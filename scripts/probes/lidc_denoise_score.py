"""Score an LIDC denoiser run independently of the Lean trainer, and draw the figure.

  .venv-lidc/bin/python scripts/probes/lidc_denoise_score.py <pairs.bin> <run.eps.f32> [out.png]

`<run.eps.f32>` is `lidc-denoise-train`'s dump of the net's output for every record of
`<pairs.bin>` (S² f32 each, noise in HU/100, the realisation-0 input). Reconstructs
denoised = low₀ − 100·eps in HU and reports, against the scanned slice, low-dose floor beside
denoised: RMSE over the body (scanned HU ≥ −950, the trainer's region, so the two must agree),
PSNR over the full 4096 HU range, and SSIM in the soft-tissue window [−160, 240] HU. The figure
is the nodule-bearing record with the most reader-marked pixels: low / denoised / full in the
lung window, a 128² crop centred on the nodule.
"""
import sys

import numpy as np
from skimage.metrics import structural_similarity

S, N = 512, 2
HW = S * S


def records(path):
    count = int(np.fromfile(path, dtype="<u4", count=1)[0])
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=4, shape=(count, (3 + 2 * N) * HW))
    return count, mm


def decode(rec):
    imgs = rec[: 2 * (1 + N) * HW].view("<u2").reshape(1 + N, S, S).astype(np.float64) - 1024.0
    return imgs[0], imgs[1], rec[2 * (1 + N) * HW:].reshape(S, S)  # full, low₀, mask


def window(img, lo, hi):
    return np.clip((img - lo) / (hi - lo), 0, 1)


def main():
    path, eps_path = sys.argv[1], sys.argv[2]
    out_png = sys.argv[3] if len(sys.argv) > 3 else None
    count, mm = records(path)
    eps = np.fromfile(eps_path, dtype="<f4").reshape(-1, S, S)
    assert len(eps) == count, f"{len(eps)} outputs for {count} records"
    se = {"low": 0.0, "den": 0.0}
    sb = {"low": 0.0, "den": 0.0}
    nb = 0
    psnr = {"low": [], "den": []}
    ssim = {"low": [], "den": []}
    best = (-1, 0)
    for k in range(count):
        full, low, mask = decode(mm[k])
        den = low - 100.0 * eps[k].astype(np.float64)
        body = full >= -950
        nb += body.sum()
        for name, img in (("low", low), ("den", den)):
            d2 = (img - full) ** 2
            se[name] += d2.sum()
            sb[name] += d2[body].sum()
            psnr[name].append(10 * np.log10(4096.0 ** 2 / d2.mean()))
            ssim[name].append(structural_similarity(window(img, -160, 240), window(full, -160, 240), data_range=1.0))
        if (mask > 0).sum() > best[0]:
            best = ((mask > 0).sum(), k)
    print(f"{path}: {count} slices — low-dose floor → denoised")
    print(f"  RMSE body   {np.sqrt(sb['low'] / nb):7.2f} → {np.sqrt(sb['den'] / nb):7.2f} HU")
    print(f"  RMSE whole  {np.sqrt(se['low'] / (count * HW)):7.2f} → {np.sqrt(se['den'] / (count * HW)):7.2f} HU")
    print(f"  PSNR        {np.mean(psnr['low']):7.2f} → {np.mean(psnr['den']):7.2f} dB (4096 HU range, mean over slices)")
    print(f"  SSIM        {np.mean(ssim['low']):7.4f} → {np.mean(ssim['den']):7.4f} (soft-tissue window)")
    if out_png:
        import imageio.v3 as iio
        npx, k = best
        full, low, mask = decode(mm[k])
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
