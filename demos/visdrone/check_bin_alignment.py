#!/usr/bin/env python3
"""Numerical version of the overlay in inspect_lean_bin.py.

Encoding is supposed to round-trip exactly: cj=int(cx*g), tx=cx*g-cj, and
decoding does cx=(cj+tx)/g. So EVERY encoded box must reproduce some annotation
box to float32 precision. Any systematic residual is an image/target alignment
bug and invalidates every measurement in planning/yolo_assignment.md.

Eyeballing a 53-object drone scene cannot distinguish "shifted box" from
"different truck", so match numerically and report the residual distribution.

    ./.venv/bin/python3 check_bin_alignment.py [--n 200]
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_lean_bin import (BIN, SRC, IMG_SIZE, FPN_GRIDS, PER_ANCHOR,
                              load_anchor_counts, parse_ann, decode_targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="records to check")
    args = ap.parse_args()

    acounts = load_anchor_counts()
    ntot = sum(a * PER_ANCHOR * g * g for a, g in zip(acounts, FPN_GRIDS))
    img_bytes = 3 * IMG_SIZE * IMG_SIZE
    rec = img_bytes + ntot * 4
    stems = sorted(p.stem for p in (SRC / "images").glob("*.jpg"))

    res_c, res_wh, n_enc, n_ann, unmatched = [], [], 0, 0, 0
    with open(BIN, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        assert count == len(stems), f"{count} records vs {len(stems)} images"
        n = min(args.n, count)
        for i in range(n):
            f.seek(4 + i * rec + img_bytes)
            flat = np.frombuffer(f.read(ntot * 4), dtype=np.float32)
            enc = decode_targets(flat, acounts)

            stem = stems[i]
            ann = parse_ann(SRC / "annotations" / f"{stem}.txt")
            with Image.open(SRC / "images" / f"{stem}.jpg") as im:
                iw, ih = im.size
            # annotations in the same normalized frame the encoder used
            A = np.array([[(x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih,
                           (x1 - x0) / iw, (y1 - y0) / ih] for _, x0, y0, x1, y1 in ann])
            n_ann += len(A)
            n_enc += len(enc)
            if not len(A):
                continue
            for _s, cx, cy, w, h, _cid in enc:
                # match on centre AND size jointly: VisDrone has concentric
                # annotations (a truck and its cab share a centre), so a
                # centre-only match picks the wrong partner and reports a
                # phantom size error.
                d = (np.abs(A[:, 0] - cx) + np.abs(A[:, 1] - cy)
                     + np.abs(A[:, 2] - w) + np.abs(A[:, 3] - h))
                k = int(np.argmin(d))
                dc = max(abs(A[k, 0] - cx), abs(A[k, 1] - cy))
                dwh = max(abs(A[k, 2] - w), abs(A[k, 3] - h))
                res_c.append(dc)
                res_wh.append(dwh)
                unmatched += (dc > 1e-4 or dwh > 1e-4)

    c, wh = np.array(res_c), np.array(res_wh)
    print(f"checked {n} records: {n_ann} annotated boxes, {n_enc} encoded slots "
          f"({100*n_enc/max(n_ann,1):.1f}%)\n")
    print("residual of each ENCODED box against its nearest ANNOTATION box,")
    print("in normalized units (multiply by 448 for fed pixels):")
    for nm, v in (("centre", c), ("w/h", wh)):
        print(f"  {nm:<7} median {np.median(v):.2e}  p99 {np.percentile(v,99):.2e}  "
              f"max {v.max():.2e}   (= {448*v.max():.3f} px)")
    print(f"\n  {unmatched} of {len(c)} encoded boxes miss their annotation by >1e-4")
    if unmatched == 0:
        print("\n  VERDICT: encoding round-trips exactly. Image<->target alignment")
        print("  is NOT the bug -- the geometry the trainer is taught is correct.")
    else:
        print("\n  VERDICT: SYSTEMATIC MISMATCH. Investigate before trusting any")
        print("  number in planning/yolo_assignment.md.")


if __name__ == "__main__":
    main()
