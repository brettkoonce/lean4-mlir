"""Score an LIDC denoiser run independently of the Lean trainer, and draw the figure.

  .venv-lidc/bin/python scripts/probes/lidc_denoise_score.py <pairs.bin> <run.eps.f32> [out.png] [--records=<file>] \
      [--by=FIELD:e1,e2,...]

`<run.eps.f32>` is a dump of the net's output for every record of `<pairs.bin>` (S² f32 each,
noise in HU/100, the realisation-0 input; `scripts/datasets/lidc_records.py` has the formats).
Reconstructs denoised = low₀ − 100·eps in HU and reports, against the scanned slice, low-dose
floor beside denoised: RMSE over the body (scanned HU ≥ −950, the trainer's region, so the two
must agree), PSNR over the full 4096 HU range, and SSIM in the soft-tissue window [−160, 240] HU.
The figure is the nodule-bearing record with the most reader-marked pixels: low / denoised / full
in the lung window, a 128² crop centred on the nodule.

Scored records: every record of the file, cut to `--records=<file>` when given and to the dump's
own `<run.eps.f32>.records` sidecar when it covers a subset (the sampler's dumps) — the numbers
then describe that subset, and the header says how many records they cover. `--by=FIELD:e1,e2,…`
prints the same block once per bin of the row field FIELD of `<pairs>.json` (half-open bins at
the edges, below e1 first) after the block over everything — the noise-level view of
planning/lidc_ct_denoising_demo.md §11; when FIELD is `sigma_add_hu` and a row has none (Mayo's
uncalibrated rows) the record is binned by the measured body std of low − full instead.
"""
import json
import os
import sys

import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
import lidc_records as R  # noqa: E402

S, HW = R.S, R.HW


def window(img, lo, hi):
    return np.clip((img - lo) / (hi - lo), 0, 1)


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    path, eps_path = pos[0], pos[1]
    out_png = pos[2] if len(pos) > 2 else None
    rec_arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--records=")), None)
    by_arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--by=")), None)
    count, n, mm = R.open_records(path)
    eps = R.open_dump(eps_path, count)
    recs = R.common_records(count, [eps_path], R.read_record_list(rec_arg) if rec_arg else None)
    field, edges, meta = None, [], None
    if by_arg:
        field, es = by_arg.split(":", 1)
        edges = [float(e) for e in es.split(",") if e]
        meta = json.load(open(path[: -len(".bin")] + ".json"))
        assert len(meta) == count, f"{len(meta)} rows for {count} records"
    labels = ["all"] + [f"{field} in [{lo}, {hi})" for lo, hi in zip(["-inf"] + edges, edges + ["inf"])]
    acc = {l: {"se": {"low": 0.0, "den": 0.0}, "sb": {"low": 0.0, "den": 0.0}, "nb": 0, "n": 0,
               "psnr": {"low": [], "den": []}, "ssim": {"low": [], "den": []}} for l in labels}
    best = (-1, 0)
    for k in recs:
        full, low, mask = R.decode(mm[k], n)
        den = low - 100.0 * eps[k].astype(np.float64)
        body = full >= -950
        into = [acc["all"]]
        if field:
            v = meta[k].get(field)
            v = float(v) if v is not None else float("nan")
            if not np.isfinite(v) and field == "sigma_add_hu":
                v = float((low - full)[body].std())
            if np.isfinite(v):
                into.append(acc[labels[1 + int(np.searchsorted(edges, v, side="right"))]])
        for a in into:
            a["nb"] += int(body.sum())
            a["n"] += 1
        for name, img in (("low", low), ("den", den)):
            d2 = (img - full) ** 2
            # A pair whose two arms are identical (two Mayo records, C158's first slices) has no
            # PSNR; it is left out of the mean rather than pulling it to infinity.
            ps = 10 * np.log10(4096.0 ** 2 / d2.mean()) if d2.mean() > 0 else None
            ss = structural_similarity(window(img, -160, 240), window(full, -160, 240), data_range=1.0)
            for a in into:
                a["se"][name] += d2.sum()
                a["sb"][name] += d2[body].sum()
                if ps is not None:
                    a["psnr"][name].append(ps)
                a["ssim"][name].append(ss)
        if (mask > 0).sum() > best[0]:
            best = ((mask > 0).sum(), k)
    subset = "" if len(recs) == count else f" ({len(recs)} of {count} records scored)"
    print(f"{path}: {len(recs)} slices{subset} — low-dose floor → denoised")
    for label in labels:
        a = acc[label]
        if a["n"] == 0:
            continue
        if label != "all":
            print(f"  — {label}: {a['n']} slices")
        se, sb, nb, psnr, ssim = a["se"], a["sb"], a["nb"], a["psnr"], a["ssim"]
        print(f"  RMSE body   {np.sqrt(sb['low'] / nb):7.2f} → {np.sqrt(sb['den'] / nb):7.2f} HU")
        print(f"  RMSE whole  {np.sqrt(se['low'] / (a['n'] * HW)):7.2f} → {np.sqrt(se['den'] / (a['n'] * HW)):7.2f} HU")
        skipped = a["n"] - len(psnr['low'])
        print(f"  PSNR        {np.mean(psnr['low']):7.2f} → {np.mean(psnr['den']):7.2f} dB (4096 HU range, mean over slices"
              + (f"; {skipped} with identical arms left out" if skipped else "") + ")")
        print(f"  SSIM        {np.mean(ssim['low']):7.4f} → {np.mean(ssim['den']):7.4f} (soft-tissue window)")
    if out_png:
        import imageio.v3 as iio
        npx, k = best
        full, low, mask = R.decode(mm[k], n)
        den = low - 100.0 * eps[k].astype(np.float64)
        if npx > 0:
            ys, xs = np.nonzero(mask)
            cy, cx = int(ys.mean()), int(xs.mean())
        else:
            cy, cx = S // 2, S // 2
        cy, cx = min(max(cy, 64), S - 64), min(max(cx, 64), S - 64)
        rows = []
        for lo, hi in ((-1350, 150), (-160, 240)):  # lung window, soft-tissue window
            tiles = [window(img[cy - 64:cy + 64, cx - 64:cx + 64], lo, hi) for img in (low, den, full)]
            rows.append(np.concatenate([np.kron(t, np.ones((3, 3))) for t in tiles], axis=1))
        fig = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
        iio.imwrite(out_png, fig)
        print(f"  figure: record {k} ({npx} nodule px), low / denoised / full, lung then soft-tissue window → {out_png}")


if __name__ == "__main__":
    main()
