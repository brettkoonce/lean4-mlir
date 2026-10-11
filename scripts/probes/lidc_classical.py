"""The classical arm of the LIDC denoising demo: Gaussian, TV and BM3D, no training.

  .venv-lidc/bin/python scripts/probes/lidc_classical.py <train.bin> <val.bin> <method> <out.eps.f32> [--workers 30]

<method> is gauss | tv | bm3d. Each filter has one strength knob, chosen on 48 slices of
<train.bin> (equally spaced through the file, so spread over its patients) by the body RMSE the
nets are trained and scored on (scanned HU >= -950); the grid's table goes to stdout, and a best
value at the grid's edge is flagged. The chosen filter is then run on every record of <val.bin>
(realisation 0, the nets' scored input) and written as `lidc-denoise-train`'s dump format —
the noise (low - denoised) in HU/100, S^2 f32 per record — so `lidc_denoise_score.py` scores
it unchanged. Filters work on HU/1000; the knob is in those units (BM3D's sigma, TV's weight)
or in pixels (the Gaussian's sigma).
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

S, N = 512, 2
HW = S * S
GRIDS = {
    "gauss": [0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5],
    "tv": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
    "bm3d": [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25],
}


def records(path):
    count = int(np.fromfile(path, dtype="<u4", count=1)[0])
    return count, np.memmap(path, dtype=np.uint8, mode="r", offset=4, shape=(count, (3 + 2 * N) * HW))


def decode(path, k):
    _, mm = records(path)
    imgs = np.asarray(mm[k, : 2 * (1 + N) * HW]).view("<u2").reshape(1 + N, S, S).astype(np.float64) - 1024.0
    return imgs[0], imgs[1]  # full, low₀


def denoise(method, knob, low):
    x = low / 1000.0
    if method == "gauss":
        from scipy.ndimage import gaussian_filter
        y = gaussian_filter(x, knob, mode="nearest")
    elif method == "tv":
        from skimage.restoration import denoise_tv_chambolle
        y = denoise_tv_chambolle(x, weight=knob)
    else:
        import bm3d
        pro = bm3d.BM3DProfile()
        pro.num_threads = 1  # one per worker (its default is every core), and deterministic
        y = bm3d.bm3d(x, knob, profile=pro)
    return 1000.0 * y


def tune_one(args):
    path, k, method = args
    full, low = decode(path, k)
    body = full >= -950
    return body.sum(), [((denoise(method, w, low) - full)[body] ** 2).sum() for w in GRIDS[method]]


def run_one(args):
    path, k, method, knob = args
    _, low = decode(path, k)
    return ((low - denoise(method, knob, low)) / 100.0).astype("<f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("val")
    ap.add_argument("method", choices=sorted(GRIDS))
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--tune-slices", type=int, default=48)
    a = ap.parse_args()
    from multiprocessing import Pool

    ntrain, _ = records(a.train)
    picks = np.linspace(0, ntrain - 1, a.tune_slices).round().astype(int)
    grid = GRIDS[a.method]
    with Pool(a.workers) as pool:
        res = pool.map(tune_one, [(a.train, int(k), a.method) for k in picks])
        nb = sum(r[0] for r in res)
        rmse = np.sqrt(np.sum([r[1] for r in res], axis=0) / nb)
        floor = np.sqrt(sum(((lambda f, l: ((l - f)[f >= -950] ** 2).sum())(*decode(a.train, int(k)))) for k in picks) / nb)
        print(f"{a.method}: tuned on {len(picks)} slices of {a.train} (body RMSE, floor {floor:.2f} HU)")
        for w, r in zip(grid, rmse):
            print(f"  {w:<6} {r:7.2f} HU")
        best = int(np.argmin(rmse))
        knob = grid[best]
        edge = " ⚠ at the grid's edge" if best in (0, len(grid) - 1) else ""
        print(f"  chosen {knob}{edge}")
        nval, _ = records(a.val)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "wb") as f:
            for eps in pool.imap(run_one, [(a.val, k, a.method, knob) for k in range(nval)], chunksize=4):
                f.write(eps.tobytes())
    print(f"  {nval} val records → {a.out}")


if __name__ == "__main__":
    main()
