#!/usr/bin/env python3
"""Carve a tiny tumour-rich BraTS subset for the overfit gate.

The overfit smoke (planning/r34_brats_retrain.md §4 step 2) exists to catch a
broken train path in minutes instead of at the end of a real epoch. For that to
mean anything the slices must actually contain tumour: a net that predicts
"background everywhere" already scores ~97% pixel accuracy on an average slice,
so overfitting a random subset proves nothing at all.

So pick the N slices with the most tumour, and require enhancing tumour (class
3) in each — that is the class that collapses, and the one whose absence would
make the gate blind.

Record format (preprocess_brats.py / lean_f32_load_brats):
    header:      4-byte LE uint32 count
    per record:  4*S*S image bytes (channel-first uint8) + S*S mask bytes
"""
import argparse
import struct
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='source .bin (e.g. data/brats/val.bin)')
    ap.add_argument('out', help='output .bin')
    ap.add_argument('--n', type=int, default=16, help='records to keep')
    ap.add_argument('--size', type=int, default=240)
    ap.add_argument('--channels', type=int, default=4)
    args = ap.parse_args()

    hw = args.size * args.size
    img_bytes = args.channels * hw
    rec_bytes = img_bytes + hw

    with open(args.src, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        blob = f.read()
    if len(blob) < count * rec_bytes:
        sys.exit(f"short file: {len(blob)} bytes for {count} records of {rec_bytes}")
    print(f"{args.src}: {count} records of {rec_bytes} bytes")

    arr = np.frombuffer(blob[:count * rec_bytes], dtype=np.uint8).reshape(count, rec_bytes)
    masks = arr[:, img_bytes:]

    tumour = (masks != 0).sum(axis=1)
    enhancing = (masks == 3).sum(axis=1)

    # Require ET, then take the largest tumour burden.
    eligible = np.where(enhancing >= 50)[0]
    if len(eligible) < args.n:
        print(f"  only {len(eligible)} slices with >=50 ET px; falling back to raw burden")
        eligible = np.arange(count)
    order = eligible[np.argsort(-tumour[eligible])]
    keep = np.sort(order[:args.n])

    print(f"  keeping {len(keep)} slices: idx {keep.tolist()}")
    print(f"  tumour px:    {tumour[keep].tolist()}")
    print(f"  enhancing px: {enhancing[keep].tolist()}")

    with open(args.out, 'wb') as f:
        f.write(struct.pack('<I', len(keep)))
        for i in keep:
            f.write(arr[i].tobytes())
    print(f"  wrote {args.out}")


if __name__ == '__main__':
    main()
