"""The classical arm of the LIDC denoising demo: Gaussian, TV and BM3D, no training.

  .venv-lidc/bin/python scripts/probes/lidc_classical.py <train.bin> <val.bin> <method> <out.eps.f32> [--workers 30]

<method> is gauss | tv | bm3d | bm3dpsd. Each filter has one strength knob, chosen on 48 slices of
<train.bin> (equally spaced through the file, so spread over its patients) by the body RMSE the
nets are trained and scored on (scanned HU >= -950); the grid's table goes to stdout, and a best
value at the grid's edge is flagged. The chosen filter is then run on every record of <val.bin>
(realisation 0, the nets' scored input) and written as `lidc-denoise-train`'s dump format —
the noise (low - denoised) in HU/100, S^2 f32 per record — so `lidc_denoise_score.py` scores
it unchanged. Filters work on HU/1000; the knob is in those units (BM3D's sigma, TV's weight)
or in pixels (the Gaussian's sigma). `bm3dpsd` is BM3D told the noise's measured power
spectrum instead of a white-noise sigma: E|fft2(low₀ − full)|² over 128 train slices that the
tuning does not use (bm3d's convention, so white noise of std σ would read σ²·S²; the DC bin replaced, see
measure_psd), written beside
<out> as `.psd.npy`; its knob multiplies that spectrum.
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

if not hasattr(np, "trapz"):  # bm4d 4.2.5's colored-noise path still calls np.trapz (removed in numpy 2.4)
    np.trapz = np.trapezoid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
import lidc_records as R  # noqa: E402

S, HW = R.S, R.HW  # the realisation count N is read off each file (2 on the 25% build, 1 on the 10%)
GRIDS = {
    "gauss": [0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5],
    "tv": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
    "bm3d": [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25],
    "bm3dpsd": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
}
_psd = {}


def records(path):
    count, _, mm = R.open_records(path)
    return count, mm


def decode(path, k):
    count, n, mm = R.open_records(path)
    full, low, _ = R.decode(mm[k], n)
    return full, low  # full, low₀


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
        y = bm3d.bm3d(x, knob if method == "bm3d" else knob * _psd["psd"], profile=pro)
    return 1000.0 * y


def measure_psd(path, picks):
    acc = np.zeros((S, S))
    for k in picks:
        full, low = decode(path, int(k))
        acc += np.abs(np.fft.fft2((low - full) / 1000.0)) ** 2
    acc /= len(picks)
    # The DC bin is the slice-mean offset low − full (≈ 6 HU, the low-count log bias), ~1,700× the
    # mean bin: a bias no filter removes, and it skews bm4d's PSD-shape estimate. Its neighbours stand in.
    acc[0, 0] = (acc[0, 1] + acc[1, 0] + acc[0, -1] + acc[-1, 0]) / 4
    return acc


def init_worker(psd_path):
    if psd_path:
        _psd["psd"] = np.load(psd_path)


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
    psd_path = None
    if a.method == "bm3dpsd":
        rest = np.setdiff1d(np.arange(ntrain), picks)
        held = rest[np.linspace(0, len(rest) - 1, min(128, len(rest))).round().astype(int)] if len(rest) else picks
        psd = measure_psd(a.train, held)
        psd_path = os.path.splitext(a.out)[0] + ".psd.npy"
        os.makedirs(os.path.dirname(os.path.abspath(psd_path)), exist_ok=True)
        np.save(psd_path, psd)
        print(f"noise PSD from {len(held)} train slices: std {np.sqrt(psd.mean()) / S * 1000:.1f} HU "
              f"(white would be flat; peak/mean {psd.max() / psd.mean():.1f}) → {psd_path}")
    with Pool(a.workers, initializer=init_worker, initargs=(psd_path,)) as pool:
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
