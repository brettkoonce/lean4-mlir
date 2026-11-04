#!/usr/bin/env python3
"""Does the field's reference low-dose CT obey "added variance = (1/dose − 1) × the scan's own"?

The CT denoising demo (planning/demo_slate.md §3) simulates its low-dose arm by inserting
Poisson noise in the projection domain and calibrating the photon count so the added noise is
√(1/dose − 1) times the slice's own noise (scripts/datasets/preprocess_lidc.py). Mayo's
LDCT-and-Projection-data (TCIA, CC BY 4.0 since v7) is the field's reference set of paired
images, and its low-dose arm is made the same way, by the vendor, at 10% of the routine dose
for the chest. Both arms of a Mayo pair are reconstructions of the same scan, so low − full is
exactly the inserted noise, and this probe measures it with the preprocessor's own estimator
(the median 7×7 local std over flat soft tissue): at 10% dose, inserted/own should be √9 = 3
and total/own √10 ≈ 3.16.

Usage: .venv-lidc/bin/python scripts/probes/lidc_mayo_noise_check.py [data/mayo] [--patient C002]
  fetches the chosen chest patient's "Full Dose Images" and "Low Dose Images" series from the
  public NBIA API into <mayo_dir>/dicom/<patient>/{full,low}/ (about 300 MB; skipped when
  present — the layout `scripts/datasets/preprocess_mayo.py` fetches the whole held-out set
  into, and it keeps C002's DICOM for this probe), then prints the two ratios over 20
  mid-volume slices. The manifest is cached at <mayo_dir>/manifest_series.json.

2026-10-07, C002 (Siemens, B50f, 273 mA, 1.5 mm): own 66.1 HU, inserted 196.7, total 208.2;
inserted/own 2.99, total/own 3.15.
"""
import argparse
import json
import os
import sys

import numpy as np
import pydicom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
from preprocess_lidc import flat_soft_tissue, local_std  # noqa: E402

from preprocess_mayo import API, COLLECTION, chest_pairs, fetch, get  # noqa: E402,F401  (one definition of the API path)


def load(d):
    out = []
    for f in os.listdir(d):
        if f.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(d, f))
            hu = ds.pixel_array.astype(np.float64) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            out.append((round(float(ds.ImagePositionPatient[2]), 1), hu, ds))
    out.sort(key=lambda t: t[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mayo_dir", nargs="?", default="data/mayo")
    ap.add_argument("--patient", default="C002")
    ap.add_argument("--slices", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.mayo_dir, exist_ok=True)
    mpath = os.path.join(args.mayo_dir, "manifest_series.json")
    if not os.path.exists(mpath):
        with open(mpath, "wb") as f:
            f.write(get(f"{API}/getSeries?Collection={COLLECTION}", timeout=300))
    with open(mpath) as f:
        pairs = chest_pairs(json.load(f))
    pair = pairs[args.patient]
    for arm, series in pair.items():
        fetch(series, os.path.join(args.mayo_dir, "dicom", args.patient, arm))
    full = load(os.path.join(args.mayo_dir, "dicom", args.patient, "full"))
    low = {z: hu for z, hu, _ in load(os.path.join(args.mayo_dir, "dicom", args.patient, "low"))}
    ds = full[0][2]
    S = full[0][1].shape[0]
    yy, xx = np.ogrid[:S, :S]
    inside = ((yy - S // 2) ** 2 + (xx - S // 2) ** 2) <= (S // 2) ** 2
    own, add, tot = [], [], []
    mid = full[len(full) // 4: 3 * len(full) // 4]
    for z, f, _ in mid[:: max(1, len(mid) // args.slices)]:
        l = low.get(z)
        if l is None:
            continue
        flat = flat_soft_tissue(f, inside)
        if flat.sum() < 500:
            continue
        own.append(np.median(local_std(f)[flat]))
        tot.append(np.median(local_std(l)[flat]))
        add.append(np.median(local_std(l - f)[flat]))
    own, add, tot = map(np.array, (own, add, tot))
    print(f"{args.patient}: {getattr(ds, 'Manufacturer', '?')} {getattr(ds, 'ConvolutionKernel', '?')} "
          f"{getattr(ds, 'XRayTubeCurrent', '?')} mA {getattr(ds, 'SliceThickness', '?')} mm, {len(own)} slices")
    print(f"  own {np.median(own):.1f} HU, inserted {np.median(add):.1f}, total {np.median(tot):.1f}")
    print(f"  inserted/own {np.median(add / own):.2f} (10% dose predicts 3.00)  total/own {np.median(tot / own):.2f} (predicts 3.16)")


if __name__ == "__main__":
    main()
