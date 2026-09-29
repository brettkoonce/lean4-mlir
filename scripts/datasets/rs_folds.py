#!/usr/bin/env python3
"""The Brazil union part and its folds — planning/remote_sensing_wavelengths_demo.md §3.8–3.9.

Concatenates the scored chips (labels 0–6; the diagnostic savanna / plantation chips stay
out) of the Brazil parts into `brazil_all` and assigns each chip a fold from a hash of its
chip id, so the wet and dry twin of a Cerrado window share a fold and the Brazil-trained
ceiling (`rs-bands train=brazil_all classes=7 fold=k`) never scores a chip it has seen in
either season.

  data/rs/brazil_all.bin, labels_brazil_all.bin, folds_brazil_all.bin (int32), meta_brazil_all.npz (chip_id, part, fold, label)

  .venv-rs/bin/python scripts/datasets/rs_folds.py [--parts amazon_dry,cerrado_dry,cerrado_wet] [--folds 5] [--data data/rs]
"""
import argparse
import os
import zlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="amazon_dry,cerrado_dry,cerrado_wet")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--data", default="data/rs")
    args = ap.parse_args()
    parts = args.parts.split(",")
    ids, src, labs, folds = [], [], [], []
    with open(os.path.join(args.data, "brazil_all.bin"), "wb") as out:
        for p in parts:
            lbl = np.fromfile(os.path.join(args.data, f"labels_{p}.bin"), dtype=np.int32)
            meta = np.load(os.path.join(args.data, f"meta_{p}.npz"))
            X = np.fromfile(os.path.join(args.data, f"{p}.bin"), dtype=np.float32).reshape(len(lbl), -1)
            keep = lbl < 7
            X[keep].tofile(out)
            for cid, l in zip(meta["chip_id"][keep], lbl[keep]):
                ids.append(str(cid)); src.append(p); labs.append(int(l)); folds.append(zlib.crc32(str(cid).encode()) % args.folds)
            print(f"{p}: {keep.sum()} scored chips ({(~keep).sum()} diagnostic left out)")
    labs = np.array(labs, dtype=np.int32); folds = np.array(folds, dtype=np.int32)
    labs.tofile(os.path.join(args.data, "labels_brazil_all.bin"))
    folds.tofile(os.path.join(args.data, "folds_brazil_all.bin"))
    np.savez_compressed(os.path.join(args.data, "meta_brazil_all.npz"), chip_id=np.array(ids), part=np.array(src), fold=folds, label=labs)
    print(f"brazil_all: {len(labs)} chips, folds " + " ".join(f"{k}:{(folds == k).sum()}" for k in range(args.folds))
          + "; per class " + " ".join(f"{c}:{(labs == c).sum()}" for c in range(7)))
    twins = {}
    for cid, f in zip(ids, folds):
        twins.setdefault(cid, set()).add(f)
    assert all(len(v) == 1 for v in twins.values()), "a chip id straddles folds"


if __name__ == "__main__":
    main()
