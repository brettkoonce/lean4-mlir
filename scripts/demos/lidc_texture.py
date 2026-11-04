#!/usr/bin/env python3
"""Texture, findings and spread — planning/lidc_ct_denoising_demo.md §10.2–§10.3: "too soft" as
a number, blobs the arm invents or drops in unmarked lung, and how much a generator's samples
disagree. Every arm with a dump under `<data>/out/<arm>/` (`lidc_records.find_dumps`), the
low-dose floor beside them, all on one record set: `--records=<file>` intersected with every
dump's `.records` sidecar (`common_records`).

  .venv-lidc/bin/python scripts/demos/lidc_texture.py --data=data/lidc \
      --records=runs/2026-11-02-lidc-diffusion/records_val.txt \
      [--arms=mse,n2n,diff.k0] [--json=rows.json] [--nps-csv=nps.csv] [--workers=6] [--no-blobs]
      [--blob-threshold=0.15] [--blob-records=<file>] [--by=FIELD:e1,e2,...]

Texture. The 2-D noise power spectrum (NPS, the CT physics measure of noise texture) over flat
soft-tissue ROIs: 64² tiles on a 32-px grid whose full-dose pixels all lie in [−200, 300] HU,
with no reader mark inside, and whose full-dose residual std — after subtracting the FULL dose's
local mean, a Gaussian blur of σ = 8 px — is below 2.5 × the slice's own noise
(`sigma_full_hu`; slices without a calibration, the apices and some bases, are skipped). The
same local mean is subtracted from every arm's tile, so what is left in an arm's tile is its
noise plus its deviation from the full-dose anatomy. NPS(u, v) = |FFT2(tile − mean)|² · Δx² / 64²
with Δx the pixel spacing in mm (units HU²·mm²), averaged over a slice's ROIs, then pooled over
slices. Against the full dose's own NPS on the same ROIs: the **integral ratio** (∫NPS du dv,
the residual variance — grain magnitude; the low dose should read about (36² + 62²)/36² ≈ 4 on the
25% build, an MSE net below 1, a perfect sampler 1.0) and the **mean-frequency ratio**
(f̄ = Σ |f|·NPS / Σ NPS over the 2-D grid, in 1/mm — a soft output sits at low frequency). Both
are pooled over slices (ratio of sums); their 95% intervals are a bootstrap over slices (1,000
resamples, seed 0). Beside them the plainer local-std ratio: median 7×7 local std over the ROI
pixels, arm over full dose, per slice; the median over slices with the same bootstrap. The full
dose is itself noisy (≈ 36 HU median on the 25% val split), so all three are ratios to the
scan's grain, not to a clean image. The radial NPS curves (`--nps-csv`) are the mean over slices
on a 0.02 /mm grid, for the figure.

Findings. LoG blob detection (`skimage.feature.blob_log`) in lung: full-dose HU < −500 inside
the reconstruction circle, eroded 3 px, and outside a 3-px dilation of every reader mark; on the
image windowed [−1000, −300] → [0, 1]; min_sigma = 1 mm / spacing / √2, max_sigma = 3 mm /
spacing / √2, 5 scales, threshold 0.15 on the scale-normalised LoG response (0.15 of the 700 HU
window = 105 HU; measured 2026-11-02 on the 25% val: at 0.08 the full dose alone carries 350
blobs a slice — vessel cross-sections and the grain itself, texture rather than findings — at
0.15 it carries 111, at 0.25 five, so 0.15 is the headline and 0.25 is reported beside it). For each arm against the full dose on the same slice: a blob is matched when a
full-dose blob lies within 3 px; per slice, blobs, **invented** (unmatched in the arm) and
**dropped** (full-dose blobs with no arm blob within 3 px), averaged over slices. The full dose
has its own noise blobs, so a perfect sampler invents and drops some; the low-dose arm bounds
invented from above (every blob the noise makes), the MSE arm shows the floor that smoothing
reaches.

Spread. For an arm with ≥ 2 sample seeds (`diff.k0 … diff.k3`): the per-pixel std across the
samples in HU, medians inside nodule regions (mask ≥ 2), in plain lung (full < −500, no mark,
eroded 3 px) and in soft tissue (full in [−200, 300]); and the body RMSE of the per-pixel mean
of the samples beside each sample's own (the mean of samples should approach the MSE net's).

Fidelity is `lidc_denoise_score.py`'s; body RMSE (full ≥ −950) is printed here only for
orientation, on this subset.

`--by=FIELD:e1,e2,…` repeats the texture-and-findings table once per bin of the row field FIELD
(half-open bins at the edges, below e1 first) after the table over everything — the
noise-level view of planning/lidc_ct_denoising_demo.md §11; a row without the field, when FIELD
is `sigma_add_hu`, is binned by the measured body std of low − full. The spread section and the
NPS csv stay overall.
"""
import json
import multiprocessing
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
import lidc_records as R  # noqa: E402
from preprocess_lidc import local_std  # noqa: E402

S, HW = R.S, R.HW
TILE, GRID = 64, 32
BLUR = 8.0
EDGE = 2.5
SOFT = (-200.0, 300.0)
BLOB_WINDOW = (-1000.0, -300.0)
BLOB_T = 0.15  # `--blob-threshold=` overrides; 0.08 counts texture, 0.25 only the strongest blobs
BLOB_MATCH = 3.0
FBIN, FMAX = 0.02, 1.0  # 1/mm, the common radial grid
NBINS = int(round(FMAX / FBIN))
LOW_NAME = "low dose"

W = {}  # per-worker state


def init(pairs, dumps):
    count, n, mm = R.open_records(pairs)
    W.update(count=count, n=n, mm=mm, meta=json.load(open(pairs[: -len(".bin")] + ".json")),
             eps={a: R.open_dump(p, count) for a, p in dumps.items()},
             groups={}, blobs=True)
    for a in dumps:
        if ".k" in a:
            W["groups"].setdefault(R.arm_base(a), []).append(a)
    c = S // 2
    yy, xx = np.ogrid[:S, :S]
    W["inside"] = ((yy - c) ** 2 + (xx - c) ** 2) <= c * c


def radial_f(spacing):
    f = np.fft.fftfreq(TILE, d=spacing)
    return np.hypot(f[:, None], f[None, :])


def blobs_of(img, lung, spacing):
    from skimage.feature import blob_log
    lo, hi = BLOB_WINDOW
    w = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    ys, xs = np.nonzero(lung)
    if len(ys) == 0:
        return np.zeros((0, 2))
    y0, y1, x0, x1 = max(ys.min() - 8, 0), min(ys.max() + 9, S), max(xs.min() - 8, 0), min(xs.max() + 9, S)
    b = blob_log(w[y0:y1, x0:x1], min_sigma=1.0 / spacing / np.sqrt(2), max_sigma=3.0 / spacing / np.sqrt(2),
                 num_sigma=5, threshold=BLOB_T)
    if len(b) == 0:
        return np.zeros((0, 2))
    pts = b[:, :2] + [y0, x0]
    keep = lung[pts[:, 0].astype(int), pts[:, 1].astype(int)]
    return pts[keep]


def match_counts(a, ref):
    """(unmatched in a, unmatched in ref) at BLOB_MATCH px."""
    if len(a) == 0 or len(ref) == 0:
        return len(a), len(ref)
    d = np.hypot(a[:, 0, None] - ref[None, :, 0], a[:, 1, None] - ref[None, :, 1])
    return int((d.min(1) > BLOB_MATCH).sum()), int((d.min(0) > BLOB_MATCH).sum())


def record(k):
    meta = W["meta"][k]
    full, low, mask = R.decode(W["mm"][k], W["n"])
    spacing = float(meta["spacing"])
    img = {LOW_NAME: low}
    for a, e in W["eps"].items():
        img[a] = low - 100.0 * e[k].astype(np.float64)
    body = full >= -950.0
    out = {"k": k, "nb": int(body.sum()),
           "sse": {a: float(((im - full) ** 2)[body].sum()) for a, im in img.items()}}
    if W.get("by"):
        v = meta.get(W["by"])
        v = float(v) if v is not None else float("nan")
        if not np.isfinite(v) and W["by"] == "sigma_add_hu":
            v = float((low - full)[body].std())
        out["by"] = v

    # ── texture ──
    sig = meta.get("sigma_full_hu")
    tex = None
    if sig is not None and np.isfinite(sig):
        blur = ndimage.gaussian_filter(full, BLUR)
        res_full = full - blur
        marked = mask > 0
        soft = (full >= SOFT[0]) & (full <= SOFT[1])
        rois = []
        for y0 in range(0, S - TILE + 1, GRID):
            for x0 in range(0, S - TILE + 1, GRID):
                sl = (slice(y0, y0 + TILE), slice(x0, x0 + TILE))
                if soft[sl].all() and not marked[sl].any() and res_full[sl].std() < EDGE * sig:
                    rois.append(sl)
        if rois:
            fr = radial_f(spacing)
            bins = np.minimum((fr / FBIN).astype(int), NBINS)  # the last bin collects > FMAX
            scale = spacing * spacing / (TILE * TILE)
            dudv = 1.0 / (TILE * spacing) ** 2  # Σ NPS·Δu·Δv = the residual variance, HU²
            roi_px = np.zeros((S, S), bool)
            for sl in rois:
                roi_px[sl] = True
            ls_full = np.median(local_std(full)[roi_px])
            tex = {"rois": len(rois), "arms": {}}
            for name, im in [("full dose", full)] + list(img.items()):
                res = im - blur
                nps = np.zeros((TILE, TILE))
                for sl in rois:
                    nps += np.abs(np.fft.fft2(res[sl])) ** 2 * scale
                nps /= len(rois)
                radial = np.bincount(bins.ravel(), weights=nps.ravel(), minlength=NBINS + 1)[:NBINS]
                cnt = np.bincount(bins.ravel(), minlength=NBINS + 1)[:NBINS]
                tex["arms"][name] = {"integral": float(nps.sum() * dudv), "fmom": float((fr * nps).sum() * dudv),
                                     "ls": float(np.median(local_std(im)[roi_px]) / max(ls_full, 1e-9)),
                                     "radial": (radial / np.maximum(cnt, 1)).tolist(), "radial_n": (cnt > 0).tolist()}
    out["tex"] = tex

    # ── findings ──
    if W["blobs"]:
        lung = ndimage.binary_erosion((full < -500.0) & W["inside"], iterations=3) \
            & ~ndimage.binary_dilation(mask > 0, iterations=3)
        ref = blobs_of(full, lung, spacing)
        bl = {"full dose": [len(ref), 0, 0]}
        for a, im in img.items():
            pts = blobs_of(im, lung, spacing)
            inv, drop = match_counts(pts, ref)
            bl[a] = [len(pts), inv, drop]
        out["blobs"] = bl

    # ── spread ──
    sp = {}
    for base, seeds in W["groups"].items():
        if len(seeds) < 2:
            continue
        stack = np.stack([img[a] for a in seeds])
        std = stack.std(0)
        mean = stack.mean(0)
        nodule = mask >= 2
        lung = ndimage.binary_erosion((full < -500.0) & W["inside"], iterations=3) \
            & ~ndimage.binary_dilation(mask > 0, iterations=3)
        soft = (full >= SOFT[0]) & (full <= SOFT[1])
        sp[base] = {"n": len(seeds),
                    "nodule": [float(np.median(std[nodule])), int(nodule.sum())] if nodule.any() else None,
                    "lung": [float(np.median(std[lung])), int(lung.sum())] if lung.any() else None,
                    "soft": [float(np.median(std[soft])), int(soft.sum())] if soft.any() else None,
                    "sse_mean": float(((mean - full) ** 2)[body].sum())}
    out["spread"] = sp
    return out


def boot(fn, n, rng, reps=1000):
    vals = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


def main():
    data = arg("data", "data/lidc")
    split = arg("split", "val")
    pairs = arg("pairs", f"{data}/{split}.bin")
    workers = int(arg("workers", 6))
    dumps = R.find_dumps(data, split)
    if arg("arms"):
        want = arg("arms").split(",")
        dumps = {a: p for a, p in dumps.items() if a in want or R.arm_base(a) in want}
    count, n, _ = R.open_records(pairs)
    requested = R.read_record_list(arg("records")) if arg("records") else None
    recs = R.common_records(count, dumps.values(), requested)
    do_blobs = "--no-blobs" not in sys.argv
    global BLOB_T
    BLOB_T = float(arg("blob-threshold", BLOB_T))
    blob_recs = set(R.read_record_list(arg("blob-records"))) if arg("blob-records") else None
    print(f"{pairs}: {count} records (N = {n}); scoring {len(recs)} of them"
          + (f" ({os.path.basename(arg('records'))})" if arg("records") else "") + "; arms: "
          + ", ".join(f"{a} = {os.path.relpath(p, data)}" for a, p in dumps.items()), flush=True)
    arms = [LOW_NAME] + list(dumps)
    names = {LOW_NAME: LOW_NAME, **{a: R.label(a) for a in dumps}}

    by_field, by_edges = None, []
    if arg("by"):
        by_field, es = arg("by").split(":", 1)
        by_edges = [float(e) for e in es.split(",") if e]
    ctx = multiprocessing.get_context("fork")
    init(pairs, dumps)
    W["by"] = by_field
    rows = []
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap(_run, [(k, do_blobs and (blob_recs is None or k in blob_recs)) for k in recs],
                                        chunksize=4), 1):
            rows.append(r)
            if i % 200 == 0 or i == len(recs):
                print(f"  {i}/{len(recs)} records", flush=True)

    res = summarize(rows, arms, names, do_blobs)
    bins = {}
    if by_field:
        bounds = list(zip([float("-inf")] + by_edges, by_edges + [float("inf")]))
        for lo, hi in bounds:
            sel = [r for r in rows if "by" in r and np.isfinite(r["by"]) and lo <= r["by"] < hi]
            label = f"{by_field} in [{lo:g}, {hi:g})"
            if sel:
                bins[label] = {"records": len(sel), "arms": summarize(sel, arms, names, do_blobs, title=label)}

    spread = {}
    bases = {b for r in rows for b in r["spread"]}
    if bases:
        print("\nspread across samples (per-pixel std, HU, median over slices of the slice median)")
        print(f"{'arm':<12} {'samples':>7} {'nodules':>9} {'lung':>9} {'soft':>9}  {'RMSE body, mean of samples':>26}  samples' own")
        for b in sorted(bases):
            rs = [r for r in rows if b in r["spread"]]
            def med(key):
                v = [r["spread"][b][key][0] for r in rs if r["spread"][b][key]]
                return float(np.median(v)) if v else float("nan")
            seeds = [a for a in dumps if R.arm_base(a) == b and ".k" in a]
            own = [res[a]["rmse_body"] for a in seeds]
            rm = float(np.sqrt(sum(r["spread"][b]["sse_mean"] for r in rs) / sum(r["nb"] for r in rs)))
            spread[b] = {"samples": rs[0]["spread"][b]["n"], "std_nodule": med("nodule"), "std_lung": med("lung"),
                         "std_soft": med("soft"), "rmse_body_mean_of_samples": rm, "rmse_body_samples": own}
            print(f"{R.label(b):<12} {spread[b]['samples']:7d} {med('nodule'):9.2f} {med('lung'):9.2f} {med('soft'):9.2f}  "
                  f"{rm:26.2f}  " + " / ".join(f"{v:.2f}" for v in own))

    if arg("json"):
        json.dump({"pairs": pairs, "records": len(rows), "arms": res, "spread": spread, "bins": bins,
                   "settings": {"tile": TILE, "grid": GRID, "blur_px": BLUR, "edge_x_sigma": EDGE, "soft_hu": SOFT,
                                "blob_window_hu": BLOB_WINDOW, "blob_threshold": BLOB_T, "blob_match_px": BLOB_MATCH}},
                  open(arg("json"), "w"), indent=1)
        print(f"→ {arg('json')}")
    if arg("nps-csv"):
        cols = ["full dose"] + arms
        tex_rows = [r for r in rows if r["tex"]]
        with open(arg("nps-csv"), "w") as f:
            f.write("f_per_mm," + ",".join(names.get(c, c) for c in cols) + "\n")
            acc = {c: np.zeros(NBINS) for c in cols}
            cnt = {c: np.zeros(NBINS) for c in cols}
            for r in tex_rows:
                for c in cols:
                    acc[c] += np.array(r["tex"]["arms"][c]["radial"]) * np.array(r["tex"]["arms"][c]["radial_n"])
                    cnt[c] += np.array(r["tex"]["arms"][c]["radial_n"])
            for i in range(NBINS):
                if cnt["full dose"][i] == 0:
                    continue
                f.write(f"{(i + 0.5) * FBIN:.3f}," + ",".join(f"{acc[c][i] / max(cnt[c][i], 1):.4f}" for c in cols) + "\n")
        print(f"→ {arg('nps-csv')} (radial NPS, HU²·mm², mean over slices)")


def summarize(rows, arms, names, do_blobs, title=None):
    """The texture-and-findings table over `rows`; returns the per-arm dict for the JSON."""
    rng = np.random.default_rng(0)
    nb = sum(r["nb"] for r in rows)
    if title:
        print(f"\n— {title}: {len(rows)} records")
    tex_rows = [r for r in rows if r["tex"]]
    n_roi = sum(r["tex"]["rois"] for r in tex_rows)
    print(f"\n{len(rows)} records; texture on {len(tex_rows)} calibrated slices, {n_roi} flat soft-tissue ROIs "
          f"({TILE}², ≥ 1 per slice); full dose's own grain is the reference (ratio 1.0)")
    res = {}
    if not tex_rows:
        print("  (no calibrated slice with a flat ROI in this set)")
        return {}
    integ_f = np.array([r["tex"]["arms"]["full dose"]["integral"] for r in tex_rows])
    fmom_f = np.array([r["tex"]["arms"]["full dose"]["fmom"] for r in tex_rows])
    hdr = f"{'arm':<16} {'RMSE body':>9}  {'NPS integral ratio':>24}  {'mean-freq ratio':>24}  {'local-std ratio':>24}"
    if do_blobs:
        hdr += f"  {'blobs/sl':>8} {'invented':>8} {'dropped':>8}"
    print(hdr)
    fbar_full = fmom_f.sum() / integ_f.sum()
    print(f"{'full dose':<16} {0.0:9.2f}  {'1.000':>24}  {'1.000':>24}  {'1.000':>24}"
          + (f"  {np.mean([r['blobs']['full dose'][0] for r in rows if 'blobs' in r]):8.2f} {'':>8} {'':>8}" if do_blobs else "")
          + f"    (f̄ {fbar_full:.3f} /mm; ∫NPS = residual variance, {np.sqrt(integ_f.mean()):.1f} HU rms per slice)")
    for a in arms:
        rmse = float(np.sqrt(sum(r["sse"][a] for r in rows) / nb))
        integ = np.array([r["tex"]["arms"][a]["integral"] for r in tex_rows])
        fmom = np.array([r["tex"]["arms"][a]["fmom"] for r in tex_rows])
        ls = np.array([r["tex"]["arms"][a]["ls"] for r in tex_rows])
        ir = integ.sum() / integ_f.sum()
        ir_ci = boot(lambda i: integ[i].sum() / integ_f[i].sum(), len(integ), rng)
        fr = (fmom.sum() / integ.sum()) / fbar_full
        fr_ci = boot(lambda i: (fmom[i].sum() / integ[i].sum()) / (fmom_f[i].sum() / integ_f[i].sum()), len(integ), rng)
        lr = float(np.median(ls))
        lr_ci = boot(lambda i: np.median(ls[i]), len(ls), rng)
        row = {"arm": a, "label": names[a], "rmse_body": rmse, "slices": len(rows), "texture_slices": len(tex_rows),
               "rois": n_roi, "nps_integral_ratio": [ir, *ir_ci], "mean_frequency_ratio": [fr, *fr_ci],
               "local_std_ratio": [lr, *lr_ci]}
        line = (f"{names[a]:<16} {rmse:9.2f}  {ir:6.3f} [{ir_ci[0]:6.3f}, {ir_ci[1]:6.3f}]   "
                f"{fr:6.3f} [{fr_ci[0]:6.3f}, {fr_ci[1]:6.3f}]   {lr:6.3f} [{lr_ci[0]:6.3f}, {lr_ci[1]:6.3f}]  ")
        if do_blobs:
            bl = np.array([r["blobs"][a] for r in rows if "blobs" in r], float)
            row["blobs_per_slice"], row["invented_per_slice"], row["dropped_per_slice"] = bl.mean(0).tolist()
            row["blob_slices"] = len(bl)
            line += f"  {bl[:, 0].mean():8.2f} {bl[:, 1].mean():8.2f} {bl[:, 2].mean():8.2f}"
        print(line)
        res[a] = row
    if do_blobs:
        nbl = sum(1 for r in rows if "blobs" in r)
        print(f"(blobs on {nbl} slices; invented = arm blobs with no full-dose blob within {BLOB_MATCH:.0f} px, "
              f"dropped = full-dose blobs with no arm blob within {BLOB_MATCH:.0f} px; LoG 1–3 mm, threshold {BLOB_T})")
    return res


def _run(job):
    k, do_blobs = job
    W["blobs"] = do_blobs
    return record(k)


if __name__ == "__main__":
    main()
