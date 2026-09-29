#!/usr/bin/env python3
"""Build the EuroSAT side of the remote-sensing demo — planning/remote_sensing_wavelengths_demo.md §2, Gate 0.

EuroSAT MS (Helber et al. 2019; Zenodo 10.5281/zenodo.7711810, MIT + Copernicus terms):
27,000 GeoTIFFs of 64×64×13 uint16, Sentinel-2A Level-1C top-of-atmosphere reflectance
× 10,000, every band cubic-spline upsampled to 10 m, ten class folders. The tif's plane
order is B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A (B8A LAST); this script
derives the order from the chips rather than trusting it (§8), refuses an unexpected
class list, and asserts torchgeo's split lists (Neumann et al. 2019, 16,200 / 5,400 /
5,400) partition the 27,000 exactly. Written, in the GW record format the trainer reads:

  data/rs/eurosat_{train,val,test}.bin        f32 [N, 13, 64, 64], per-band standardised
                                              ((DN / 10000 − mean_b) / std_b, mean/std of TRAIN)
  data/rs/labels_eurosat_{train,val,test}.bin int32, classes in alphabetical order (torchgeo's)
  data/rs/eurosat_{split}_names.txt           the chip file stems in record order
  data/rs/eurosat_quicklook.png               ten chips per class: true colour over false colour
  data/rs/manifest_rs.json                    census, band-order evidence, offset check, mean/std,
                                              split sizes, the ArASL leak audit

The Brazil preprocessor reads `band_mean` / `band_std` from the manifest and applies the
same standardisation, so a Brazilian chip and a European one are the same function of
reflectance. The leak audit is `preprocess_arasl.leak_audit`, unchanged: nearest training
chip to every val/test chip by mean |Δ| on a 16×16 grey thumbnail of the true-colour render.

  .venv-rs/bin/python scripts/datasets/preprocess_rs_eurosat.py data/rs/eurosat data/rs [--stats] [--workers=8]
"""
import argparse
import glob
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_arasl import leak_audit, THUMB, LEAK_THR  # noqa: E402  (the ArASL instrument, unchanged)

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
N_TOTAL = 27000
H = W = 64
NB = 13
SCALE = 10000.0          # L1C DN → reflectance
QL_MAX = 2750.0          # the EuroSAT README's own quicklook scaling


def read_tif(path):
    import rasterio
    with rasterio.open(path) as src:
        x = src.read()
    return x


def unzip_if_needed(src):
    tifs = glob.glob(os.path.join(src, "**", "*.tif"), recursive=True)
    if len(tifs) >= N_TOTAL:
        return
    zips = [z for z in ("EuroSAT_MS.zip", "EuroSATallBands.zip") if os.path.exists(os.path.join(src, z))]
    if not zips:
        sys.exit(f"{src}: no tifs and no EuroSAT_MS.zip / EuroSATallBands.zip — run scripts/datasets/download_rs.sh")
    z = os.path.join(src, zips[0])
    print(f"unzipping {z} ({os.path.getsize(z) / 1e9:.2f} GB) ...", flush=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="data/rs/eurosat (holds the zip or the unzipped class folders + split lists)")
    ap.add_argument("out", help="data/rs")
    ap.add_argument("--stats", action="store_true", help="census, band-order evidence and audit only; no records")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    unzip_if_needed(args.src)

    # ── census: ten class folders, 27,000 tifs, 2,000–3,000 each ──
    tifs = sorted(glob.glob(os.path.join(args.src, "**", "*.tif"), recursive=True))
    by_class = {}
    for p in tifs:
        by_class.setdefault(os.path.basename(os.path.dirname(p)), []).append(p)
    classes = sorted(by_class)
    if classes != CLASSES:
        sys.exit(f"class folders {classes} != {CLASSES}")
    census = {c: len(by_class[c]) for c in classes}
    if sum(census.values()) != N_TOTAL or any(not 2000 <= n <= 3000 for n in census.values()):
        sys.exit(f"census {census}: expected {N_TOTAL} total, 2,000–3,000 per class")
    paths = [p for c in classes for p in by_class[c]]
    labels = np.array([ci for ci, c in enumerate(classes) for _ in by_class[c]], dtype=np.int32)
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    print(f"census: {census}  ({len(paths)} tifs)", flush=True)

    # ── load: [27000, 13, 64, 64] uint16 ──
    with Pool(args.workers) as pool:
        chips = pool.map(read_tif, paths, chunksize=256)
    shapes = {x.shape for x in chips}
    dtypes = {str(x.dtype) for x in chips}
    if shapes != {(NB, H, W)} or dtypes != {"uint16"}:
        sys.exit(f"chip shapes {shapes} dtypes {dtypes}: expected {{(13, 64, 64)}} uint16")
    X = np.stack(chips)
    del chips
    print(f"loaded {X.shape} {X.dtype} ({time.time() - t0:.0f} s)", flush=True)

    # ── band-order evidence, from the chips (§8): B08 ≈ B8A over forest, both ≫ B04; B10 ≈ 0;
    #    the plane most correlated with plane 7 (B08) is plane 12 (B8A); SeaLake B12 is dark ──
    med = {c: np.median(X[labels == ci].reshape(-1, NB, H * W).mean(axis=2), axis=0) for ci, c in enumerate(classes)}
    forest = med["Forest"]
    sea = med["SeaLake"]
    sub = X[np.random.default_rng(0).choice(len(X), 2000, replace=False)].reshape(-1, NB, H * W).mean(axis=2)
    corr = np.corrcoef(sub.T.astype(np.float64))
    corr_with_b08 = corr[7].copy()
    corr_with_b08[7] = -1
    twin = int(corr_with_b08.argmax())
    band_check = {
        "forest_median_DN": {b: float(forest[i]) for i, b in enumerate(BANDS)},
        "sealake_median_DN": {b: float(sea[i]) for i, b in enumerate(BANDS)},
        "forest_B08_over_B8A": float(forest[7] / forest[12]),
        "forest_B08_over_B04": float(forest[7] / forest[3]),
        "all_B10_median_DN": float(np.median(X[:, 9])),
        "plane_most_correlated_with_B08": BANDS[twin],
        "global_min_DN": int(X.min()),
        "sealake_B12_median_DN": float(sea[11]),
    }
    ok = (0.8 < band_check["forest_B08_over_B8A"] < 1.25 and band_check["forest_B08_over_B04"] > 2.0
          and band_check["all_B10_median_DN"] < 100 and twin == 12
          and band_check["sealake_B12_median_DN"] < 500 and band_check["global_min_DN"] < 200)
    print(f"band order: forest B08/B8A {band_check['forest_B08_over_B8A']:.3f}, B08/B04 {band_check['forest_B08_over_B04']:.2f}, "
          f"B10 median {band_check['all_B10_median_DN']:.0f} DN, B08's twin is plane {twin} ({BANDS[twin]}), "
          f"SeaLake B12 median {sea[11]:.0f} DN, global min {X.min()} DN  -> {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        sys.exit("Gate 0: the plane order or the offset is not what the plan assumes — inspect band_check above")

    # ── splits: torchgeo's lists partition the 27,000 ──
    idx_of = {s: i for i, s in enumerate(stems)}
    splits = {}
    for sp in ("train", "val", "test"):
        with open(os.path.join(args.src, f"eurosat-{sp}.txt")) as f:
            names = [os.path.splitext(l.strip())[0] for l in f if l.strip()]
        missing = [n for n in names if n not in idx_of]
        if missing:
            sys.exit(f"{sp} list: {len(missing)} names not in the tifs, e.g. {missing[:3]}")
        splits[sp] = np.array([idx_of[n] for n in names], dtype=np.int64)
    allidx = np.concatenate(list(splits.values()))
    if len(np.unique(allidx)) != N_TOTAL or len(allidx) != N_TOTAL:
        sys.exit(f"split lists do not partition the {N_TOTAL} chips: {[len(v) for v in splits.values()]}, {len(np.unique(allidx))} unique")
    print("splits: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()), flush=True)

    # ── per-band mean/std of the training split, in reflectance ──
    tr = X[splits["train"]].astype(np.float32) / SCALE
    band_mean = tr.mean(axis=(0, 2, 3)).astype(np.float64)
    band_std = tr.std(axis=(0, 2, 3)).astype(np.float64)
    del tr
    print("train mean (refl): " + " ".join(f"{b}={m:.4f}" for b, m in zip(BANDS, band_mean)), flush=True)
    print("train std  (refl): " + " ".join(f"{b}={s:.4f}" for b, s in zip(BANDS, band_std)), flush=True)

    # ── thumbnails for the audit: 16×16 grey of the true-colour render ──
    rgb = X[:, [3, 2, 1]].astype(np.float32)
    grey = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]) / QL_MAX
    grey = np.clip(grey * 255.0, 0, 255).astype(np.uint8)
    thumbs = np.stack([np.asarray(Image.fromarray(g).resize((THUMB, THUMB), Image.LANCZOS)) for g in grey]).reshape(N_TOTAL, -1)
    audit = {}
    for sp in ("val", "test"):
        t1 = time.time()
        nn, nd = leak_audit(thumbs, splits["train"], splits[sp])
        same = labels[nn] == labels[splits[sp]]
        audit[sp] = dict(n=int(len(nd)), frac_under=float((nd < LEAK_THR).mean()), median_nearest=float(np.median(nd)),
                         nearest_same_class=float(same.mean()))
        print(f"leak audit {sp:4s}: {100 * audit[sp]['frac_under']:5.1f}% of chips have a train chip within {LEAK_THR:g} grey "
              f"levels at {THUMB}×{THUMB} (median nearest {np.median(nd):.1f}); nearest is same class {100 * same.mean():.1f}%  "
              f"({time.time() - t1:.0f} s)", flush=True)
    onenn = float((labels[leak_audit(thumbs, splits["train"], splits["test"])[0]] == labels[splits["test"]]).mean())
    print(f"1-NN on {THUMB}×{THUMB} grey thumbnails, test: {100 * onenn:.1f}%", flush=True)

    # ── quicklook: ten chips per class, true colour over false colour ──
    rng = np.random.default_rng(0)
    tiles_tc, tiles_fc = [], []
    for ci in range(len(classes)):
        pick = rng.choice(np.where(labels == ci)[0], 10, replace=False)
        for planes, dst in (([3, 2, 1], tiles_tc), ([7, 3, 2], tiles_fc)):
            row = np.concatenate([np.clip(X[i][planes].transpose(1, 2, 0) / QL_MAX * 255, 0, 255).astype(np.uint8) for i in pick], axis=1)
            dst.append(row)
    ql = np.concatenate([np.concatenate(tiles_tc, axis=0), np.concatenate(tiles_fc, axis=0)], axis=1)
    Image.fromarray(ql).resize((ql.shape[1] * 2, ql.shape[0] * 2), Image.NEAREST).save(os.path.join(args.out, "eurosat_quicklook.png"))

    manifest = dict(source="EuroSAT MS, Zenodo 10.5281/zenodo.7711810 (Helber et al. 2019), MIT + Copernicus Sentinel data terms",
                    level="L1C top-of-atmosphere reflectance x 10000, uint16, no radiometric offset",
                    classes=classes, census=census, band_order=BANDS, chip=[NB, H, W], scale=SCALE,
                    band_check=band_check, splits={k: int(len(v)) for k, v in splits.items()},
                    split_source="torchgeo eurosat-{train,val,test}.txt (Neumann et al. 2019, arXiv:1911.06721)",
                    band_mean=band_mean.tolist(), band_std=band_std.tolist(), audit=audit, onenn_test=onenn,
                    quicklook="eurosat_quicklook.png: rows = classes in order, left half true colour B04/B03/B02, right half false colour B08/B04/B03, 0..2750 DN")
    with open(os.path.join(args.out, "manifest_rs.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if args.stats:
        print(f"--stats: no records written ({time.time() - t0:.0f} s)")
        return

    # ── records: standardised f32 [N, 13, 64, 64] + int32 labels, in list order ──
    mean = band_mean.astype(np.float32)[None, :, None, None]
    std = band_std.astype(np.float32)[None, :, None, None]
    for sp, idx in splits.items():
        xs = (X[idx].astype(np.float32) / SCALE - mean) / std
        xs.tofile(os.path.join(args.out, f"eurosat_{sp}.bin"))
        labels[idx].tofile(os.path.join(args.out, f"labels_eurosat_{sp}.bin"))
        with open(os.path.join(args.out, f"eurosat_{sp}_names.txt"), "w") as f:
            f.write("\n".join(stems[i] for i in idx) + "\n")
        print(f"wrote eurosat_{sp}.bin {xs.shape} ({xs.nbytes / 1e9:.2f} GB)", flush=True)
    print(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
