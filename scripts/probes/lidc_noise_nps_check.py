#!/usr/bin/env python3
"""Does our simulated low-dose noise have the SHAPE of a scanner's? The noise power spectrum of
the LIDC pairs' simulated noise against Mayo's vendor-inserted noise (planning/
lidc_ct_denoising_demo.md §6 phase 4a, §10.4 "first check, no GPU").

  .venv-lidc/bin/python scripts/probes/lidc_noise_nps_check.py data/lidc/val.bin data/mayo/val.bin \
      [--out-dir runs/2026-11-02-lidc-mayo] [--lidc-records 1000] [--mayo-records 0] [--roi 64] [--grid 32]

The nets are trained on LIDC slices with noise from preprocess_lidc.py's parallel-beam Poisson
simulation; the held-out set's low dose was made by Siemens' projection-domain noise insertion.
§3 checked the LEVEL (added variance = (1/dose − 1) × the scan's own, to 1%); this checks the
TEXTURE, because a denoiser learns the noise's correlation structure and a generator is judged by
reproducing it (§10.2). In each file, low − full is exactly the added noise (the full-dose image is
the target itself in both constructions). Its 2-D noise power spectrum is measured the way CT
physics does: over flat soft-tissue ROIs (`--roi`² windows on a `--grid`-px lattice, every
full-dose pixel in −200..300 HU, no reader mark, and — so that vessels and organ edges inside the
window do not pass as "grain" — the window's detrended std within `--flat` × the slice's own
noise, the pair file's `sigma_full_hu`), each ROI detrended by a fitted 2-D quadratic,
NPS(f) = |FFT2(roi)|²·Δx²/n² radially averaged, converted from cycles/px to 1/mm with the
record's own pixel spacing (LIDC's varies by patient; Mayo's is 0.742 mm), interpolated onto one
frequency grid and averaged over ROIs. The doses differ (25% vs 10%), so each spectrum is
normalised to unit integral and compared by SHAPE: the peak frequency (the gate: within ~10%)
and the mean frequency. The same measurement on the full-dose images themselves (the scanners'
own grain: LIDC's mixed STANDARD / LUNG / B-kernels against Mayo's sharp B50f) is reported
beside it, because that is the texture the demo's generator is asked to reproduce on each set —
with one caveat a patient scan forces: there is no second full-dose scan to difference, so what
anatomy survives the window test and the detrend sits in the lowest bins (below ≈ 0.1 /mm the
LIDC curve is organ texture, not grain), and the full-dose peak is therefore searched above
0.1 /mm. Writes nps.csv (the curves), nps.png and summary.txt.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
import lidc_records as R  # noqa: E402

F_GRID = np.arange(0.0, 0.651, 0.005)  # 1/mm; the axis Nyquist is 0.67 at Mayo's 0.742 mm, 0.61 at LIDC's coarsest 0.82


def detrend(roi):
    """roi minus its least-squares 2-D quadratic (1, x, y, x², xy, y²)."""
    n = roi.shape[0]
    y, x = np.mgrid[:n, :n] / (n - 1.0) - 0.5
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1).reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(A, roi.ravel(), rcond=None)
    return roi - (A @ coef).reshape(n, n)


def radial_nps(roi, spacing):
    """(f in 1/mm, NPS in HU²·mm²) radially averaged in bins of 1/n cycles/px, up to the axis Nyquist."""
    n = roi.shape[0]
    d = detrend(roi)
    p = np.abs(np.fft.fftshift(np.fft.fft2(d))) ** 2 * spacing * spacing / (n * n)
    k = np.fft.fftshift(np.fft.fftfreq(n))  # cycles/px
    r = np.hypot(k[:, None], k[None, :])
    bins = np.round(r * n).astype(int)  # bin width 1/n cycles/px
    nb = n // 2 + 1
    s = np.bincount(bins.ravel(), p.ravel(), minlength=nb)[:nb]
    c = np.bincount(bins.ravel(), minlength=nb)[:nb]
    return np.arange(nb) / n / spacing, s / np.maximum(c, 1), float(d.std())


def rois_of(full, mask, roi, grid, sigma_full, flat):
    """Top-left corners of the flat soft-tissue windows: HU in −200..300 throughout, no reader mark,
    and the detrended std at most `flat` × the slice's own noise (skipped when that is unknown)."""
    ok = (full >= -200) & (full <= 300) & (mask == 0)
    out = []
    S = full.shape[0]
    for y0 in range(0, S - roi + 1, grid):
        for x0 in range(0, S - roi + 1, grid):
            if not ok[y0:y0 + roi, x0:x0 + roi].all():
                continue
            if np.isfinite(sigma_full) and detrend(full[y0:y0 + roi, x0:x0 + roi]).std() > flat * sigma_full:
                continue
            out.append((y0, x0))
    return out


def measure(path, n_records, roi, grid, flat, label):
    count, n, mm = R.open_records(path)
    meta = json.load(open(path[: -len(".bin")] + ".json"))
    recs = list(range(count)) if n_records <= 0 or n_records >= count else \
        np.linspace(0, count - 1, n_records).round().astype(int).tolist()
    acc = {"noise": [], "full": []}
    stds = {"noise": [], "full": []}
    used = 0
    spacings = []
    for k in recs:
        full, low, mask = R.decode(mm[k], n)
        sp = float(meta[k]["spacing"])
        sf = meta[k].get("sigma_full_hu")
        boxes = rois_of(full, mask, roi, grid, float(sf) if sf is not None else float("nan"), flat)
        if not boxes:
            continue
        used += 1
        spacings.append(sp)
        d = low - full
        for y0, x0 in boxes:
            for name, img in (("noise", d), ("full", full)):
                f, s, sd = radial_nps(img[y0:y0 + roi, x0:x0 + roi], sp)
                acc[name].append(np.interp(F_GRID, f, s, right=np.nan))
                stds[name].append(sd)
    res = {"label": label, "path": path, "records": len(recs), "records_with_rois": used,
           "rois": len(acc["noise"]), "spacing_median_mm": float(np.median(spacings)) if spacings else None,
           "spacing_range_mm": [float(min(spacings)), float(max(spacings))] if spacings else None}
    for name in ("noise", "full"):
        if not acc[name]:
            continue
        arr = np.array(acc[name])
        cnt = np.isfinite(arr).sum(0)
        curve = np.where(cnt > 0, np.nansum(arr, axis=0) / np.maximum(cnt, 1), 0.0)
        norm = curve / np.trapezoid(curve, F_GRID)
        # a 3-point smooth before the argmax so one noisy bin cannot be "the peak"; the full-dose
        # peak is searched above 0.1 /mm (the docstring's caveat: below it is anatomy)
        sm = np.convolve(norm, np.ones(3) / 3, mode="same")
        if name == "full":
            sm = np.where(F_GRID >= 0.1, sm, 0.0)
        res[name] = {"nps": curve, "nps_norm": norm,
                     "peak_f": float(F_GRID[int(np.argmax(sm))]),
                     "peak_cpp": float(F_GRID[int(np.argmax(sm))]) * float(np.median(spacings)),
                     "mean_f": float(np.trapezoid(F_GRID * norm, F_GRID)),
                     "std_hu_median": float(np.median(stds[name])),
                     "std_hu_from_nps": float(np.sqrt(np.trapezoid(curve * 2 * np.pi * F_GRID, F_GRID)))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lidc_bin")
    ap.add_argument("mayo_bin")
    ap.add_argument("--out-dir", default="runs/2026-11-02-lidc-mayo")
    ap.add_argument("--lidc-records", type=int, default=1000)
    ap.add_argument("--mayo-records", type=int, default=0)
    ap.add_argument("--roi", type=int, default=64)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--flat", type=float, default=1.3, help="max detrended ROI std, in units of the slice's own noise")
    ap.add_argument("--lidc-label", default="LIDC (simulated, 25%)")
    ap.add_argument("--mayo-label", default="Mayo (vendor, 10%)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sets = [measure(args.lidc_bin, args.lidc_records, args.roi, args.grid, args.flat, args.lidc_label),
            measure(args.mayo_bin, args.mayo_records, args.roi, args.grid, args.flat, args.mayo_label)]
    lines = []
    for r in sets:
        lines.append(f"{r['label']}: {r['path']} — {r['records']} records, {r['records_with_rois']} with a flat "
                     f"{args.roi}² ROI, {r['rois']} ROIs, spacing {r['spacing_median_mm']:.3f} mm "
                     f"[{r['spacing_range_mm'][0]:.3f}, {r['spacing_range_mm'][1]:.3f}]")
        for name, what in (("noise", "added noise (low − full)"), ("full", "full-dose image (the scanner's own grain; peak above 0.1 /mm)")):
            if name in r:
                m = r[name]
                lines.append(f"  {what}: σ {m['std_hu_median']:.1f} HU (median ROI std), NPS peak {m['peak_f']:.3f} /mm "
                             f"({m['peak_cpp']:.3f} cycles/px at the median spacing), mean frequency {m['mean_f']:.3f} /mm")
    L, M = sets
    for name, what in (("noise", "added noise"), ("full", "full-dose grain")):
        if name in L and name in M:
            pk = M[name]["peak_f"] / L[name]["peak_f"] if L[name]["peak_f"] > 0 else float("nan")
            mf = M[name]["mean_f"] / L[name]["mean_f"]
            lines.append(f"{what}: peak frequency Mayo/LIDC {pk:.2f}, mean frequency Mayo/LIDC {mf:.2f}"
                         + (f"  — gate (peak within 10%): {'PASS' if abs(pk - 1) <= 0.10 else 'FAIL'}" if name == "noise" else ""))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(args.out_dir, "nps.csv"), "w") as f:
        cols = ["f_per_mm"]
        arrs = [F_GRID]
        for r, tag in ((L, "lidc"), (M, "mayo")):
            for name in ("noise", "full"):
                if name in r:
                    cols += [f"{tag}_{name}_nps_hu2mm2", f"{tag}_{name}_nps_norm"]
                    arrs += [r[name]["nps"], r[name]["nps_norm"]]
        f.write(",".join(cols) + "\n")
        for row in zip(*arrs):
            f.write(",".join(f"{v:.6g}" for v in row) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, name, title in zip(axes, ("noise", "full"), ("added noise (low − full)", "full-dose image")):
        for r, c in ((L, "tab:blue"), (M, "tab:red")):
            if name in r:
                ax.plot(F_GRID, r[name]["nps_norm"], color=c, label=f"{r['label']}: σ {r[name]['std_hu_median']:.0f} HU, "
                        f"peak {r[name]['peak_f']:.2f}, mean {r[name]['mean_f']:.2f} /mm")
        ax.set_xlabel("spatial frequency (1/mm)")
        ax.set_ylabel("NPS, normalised to unit integral")
        ax.set_title(title)
        ax.set_xlim(0, F_GRID[-1])
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "nps.png"), dpi=130)
    print(f"→ {args.out_dir}/{{summary.txt,nps.csv,nps.png}}")


if __name__ == "__main__":
    main()
