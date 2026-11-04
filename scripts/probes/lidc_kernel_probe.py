#!/usr/bin/env python3
"""Does a candidate FBP apodisation give our simulated noise the SPECTRUM of Mayo's? The phase-4a
gate (planning/lidc_ct_denoising_demo.md §6, §7 b) run before a build, on a few hundred slices.

  .venv-lidc/bin/python scripts/probes/lidc_kernel_probe.py [data/lidc10/val.bin] --filter b50f \
      [--records 300] [--workers 24] [--mayo-csv runs/2026-11-02-lidc-mayo/nps.csv] [--out-dir DIR]

Takes the full-dose slices of an existing pair file with their rows' dose, spacing and own noise,
and simulates the low-dose noise exactly as preprocess_lidc.py would under `--filter` (the
sinogram, the per-slice I0 calibration under that filter, one realisation), then measures the
added noise's radial NPS the way lidc_noise_nps_check.py does (flat soft-tissue 64² ROIs,
quadratic detrend, 1/mm with each record's spacing, unit-integral normalisation) and compares it
with Mayo's vendor-inserted noise from that script's nps.csv: the peak frequency (the gate: within
10 %), the mean frequency, and the per-frequency ratio Mayo / ours, whose square root is the
correction a next kernel would multiply in. Also reports the calibration under the filter
(achieved / target σ_add, which must stay at 1) so a kernel cannot pass by moving the level.
Writes nps.csv, nps.png and summary.txt under --out-dir.
"""
import argparse
import json
import multiprocessing
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "datasets"))
sys.path.insert(0, HERE)
import lidc_records as R  # noqa: E402
import preprocess_lidc as P  # noqa: E402
from lidc_noise_nps_check import F_GRID, radial_nps, rois_of  # noqa: E402

CTX = {}


def one(k):
    full, _low, mask = R.decode(CTX["mm"][k], CTX["n"])
    row = CTX["meta"][k]
    sp = float(row["spacing"])
    sf = row.get("sigma_full_hu")
    if sf is None or not np.isfinite(sf) or not row.get("calibrated", True):
        return None
    boxes = rois_of(full, mask, CTX["roi"], CTX["grid"], float(sf), CTX["flat"])
    if not boxes:
        return None
    proj = CTX["proj"]
    hu = full
    pnum = int(row["patient"].rsplit("-", 1)[-1])
    zi = int(row["slice"])
    dose = float(row.get("dose", CTX["dose"]))  # rows before the range build carry no dose field
    gain = (1.0 / dose - 1.0) ** 0.5
    i0_ref_add = 1e5 * dose / (1.0 - dose)
    sino = proj.sinogram(hu, sp)
    flat = P.flat_soft_tissue(hu.astype(np.float64), proj.inside)
    if flat.sum() < P.CAL_MIN_PIXELS:
        return None
    own = P.noise_level(hu, flat)
    rng = np.random.default_rng([0, pnum, 1_000_000 + zi])
    s_ref = P.noise_level(proj.noise_hu(sino, sp, i0_ref_add, 0.0, rng), flat)
    i0_add = float(np.clip(i0_ref_add * (s_ref / (gain * own)) ** 2, 1e2, 1e12))
    rng = np.random.default_rng([0, pnum, zi, 0])
    d = proj.noise_hu(sino, sp, i0_add, 0.0, rng)
    achieved = P.noise_level(d, flat) / (gain * own)
    curves = []
    for y0, x0 in boxes:
        f, s, _sd = radial_nps(d[y0:y0 + CTX["roi"], x0:x0 + CTX["roi"]], sp)
        curves.append(np.interp(F_GRID, f, s, right=np.nan))
    return {"k": k, "spacing": sp, "achieved": achieved, "i0_add": i0_add, "i0_add_row": row.get("i0_add"),
            "curves": curves}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", nargs="?", default="data/lidc10/val.bin")
    ap.add_argument("--filter", default="b50f")
    ap.add_argument("--records", type=int, default=300)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--angles", type=int, default=720)
    ap.add_argument("--roi", type=int, default=64)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--flat", type=float, default=1.3)
    ap.add_argument("--mayo-csv", default="runs/2026-11-02-lidc-mayo/nps.csv")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out = args.out_dir or f"runs/2026-11-03-lidc-kernel/probe_{args.filter}"
    os.makedirs(out, exist_ok=True)
    count, n, mm = R.open_records(args.pairs)
    meta = json.load(open(args.pairs[:-4] + ".json"))
    manifest = json.load(open(os.path.join(os.path.dirname(args.pairs), "manifest_lidc.json")))
    CTX["dose"] = float(manifest["args"]["dose"])
    recs = np.linspace(0, count - 1, min(args.records, count)).round().astype(int).tolist()
    CTX.update(mm=mm, n=n, meta=meta, roi=args.roi, grid=args.grid, flat=args.flat,
               proj=P.Projector(R.S, args.angles, args.filter))
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        res = [r for r in pool.imap_unordered(one, recs, chunksize=2) if r]
    curves = np.array([c for r in res for c in r["curves"]])
    cnt = np.isfinite(curves).sum(0)
    curve = np.where(cnt > 0, np.nansum(curves, axis=0) / np.maximum(cnt, 1), 0.0)
    norm = curve / np.trapezoid(curve, F_GRID)
    sm = np.convolve(norm, np.ones(3) / 3, mode="same")
    peak = float(F_GRID[int(np.argmax(sm))])
    mean_f = float(np.trapezoid(F_GRID * norm, F_GRID))
    mayo = np.genfromtxt(args.mayo_csv, delimiter=",", names=True)
    mn = mayo["mayo_noise_nps_norm"]
    msm = np.convolve(mn, np.ones(3) / 3, mode="same")
    mpeak = float(F_GRID[int(np.argmax(msm))])
    mmean = float(np.trapezoid(F_GRID * mn, F_GRID))
    ach = np.array([r["achieved"] for r in res])
    sp = np.array([r["spacing"] for r in res])
    i0r = np.array([r["i0_add"] / r["i0_add_row"] for r in res if r["i0_add_row"]])
    lines = [f"{args.pairs}: {len(recs)} records asked, {len(res)} with a flat {args.roi}² ROI and a calibration, "
             f"{len(curves)} ROIs, spacing {np.median(sp):.3f} mm [{sp.min():.3f}, {sp.max():.3f}]; filter {args.filter}",
             f"  calibration under {args.filter}: achieved/target σ_add median {np.median(ach):.3f} "
             f"[p5 {np.percentile(ach, 5):.3f}, p95 {np.percentile(ach, 95):.3f}]; I0_add vs the file's (its filter) "
             f"median ×{np.median(i0r):.2f}" if len(i0r) else "",
             f"  added noise NPS: peak {peak:.3f} /mm, mean frequency {mean_f:.3f} /mm",
             f"  Mayo ({args.mayo_csv}): peak {mpeak:.3f} /mm, mean frequency {mmean:.3f} /mm",
             f"  peak Mayo/ours {mpeak / peak:.2f}, mean frequency Mayo/ours {mmean / mean_f:.2f}"
             f"  — gate (peak within 10%): {'PASS' if abs(mpeak / peak - 1) <= 0.10 else 'FAIL'}"]
    ratio = np.where(norm > 1e-6, mn / np.maximum(norm, 1e-6), np.nan)
    rsm = np.convolve(np.nan_to_num(ratio), np.ones(3) / 3, mode="same")
    lines.append("  √(Mayo/ours) by frequency (the correction a next kernel would multiply in):")
    lines.append("    " + "  ".join(f"{x:.2f}:{np.sqrt(np.interp(x, F_GRID, rsm)):.2f}" for x in np.arange(0.08, 0.641, 0.04)))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    with open(os.path.join(out, "nps.csv"), "w") as f:
        f.write("f_per_mm,ours_nps_hu2mm2,ours_nps_norm,mayo_nps_norm\n")
        for a, b, c, d in zip(F_GRID, curve, norm, mn):
            f.write(f"{a:.6g},{b:.6g},{c:.6g},{d:.6g}\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(F_GRID, norm, color="tab:blue", label=f"LIDC simulated, {args.filter}: peak {peak:.2f}, mean {mean_f:.2f} /mm")
    ax.plot(F_GRID, mn, color="tab:red", label=f"Mayo vendor (B50f): peak {mpeak:.2f}, mean {mmean:.2f} /mm")
    lidc_sl = mayo["lidc_noise_nps_norm"]
    ax.plot(F_GRID, lidc_sl, color="tab:gray", ls="--", label="LIDC simulated, shepp-logan (25 % build)")
    ax.set_xlabel("spatial frequency (1/mm)")
    ax.set_ylabel("NPS, unit integral")
    ax.set_xlim(0, F_GRID[-1])
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "nps.png"), dpi=130)
    print(f"→ {out}/{{summary.txt,nps.csv,nps.png}}")


if __name__ == "__main__":
    main()
