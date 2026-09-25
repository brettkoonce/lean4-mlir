#!/usr/bin/env python3
"""The CAM as a number — planning/plant_lab_to_field_demo.md §3.5, §4 Table 2.

`plant-leaf … cam=1` dumps the closed-form class-activation map (7×7, ReLU'd, max-normalised)
of every image of a part, for the true class and for the predicted class. This reads the dump
against the leaf mask `scripts/datasets/preprocess_plant.py` stored at the same 7×7 (the fraction of each cell
that is leaf) and reports the share of CAM mass that lands inside the leaf — the statistic the
section tests with the background fix — per image, pooled, per class, and for the leaked /
non-leaked split of the audit. `--examples` prints the images at the extremes for the figure.

  .venv/bin/python scripts/demos/plant_cam.py <prefix>_cam_pvg_test.bin --split pvg [--json OUT] [--examples 4]
"""
import argparse
import json
import os

import numpy as np

N_CLASSES = 38
G = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam", help="<prefix>_cam_<split>_test.bin")
    ap.add_argument("--split", default="pvg", choices=("pv", "pvg"))
    ap.add_argument("--data", default="data/plant")
    ap.add_argument("--json", default=None)
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()
    meta = np.load(os.path.join(args.data, "meta_plant.npz"))
    classes = [str(c).replace("___", " ").replace("_", " ") for c in meta["classes"]]
    m7 = np.load(os.path.join(args.data, f"{args.split}_test_mask7.npy"))       # [n, 7, 7] leaf fraction per cell
    n = len(m7)
    cam = np.fromfile(args.cam, dtype=np.float32).reshape(n, 2, G, G)
    pred = np.fromfile(args.cam.replace("_cam_", "_campred_"), dtype=np.int32)
    te = meta[f"{args.split}_test_index"]
    labels = meta["pv_label"][te]
    assert len(pred) == n and len(labels) == n

    out = {}
    for k, name in enumerate(("true class", "predicted class")):
        c = cam[:, k]
        total = c.reshape(n, -1).sum(axis=1)
        inside = (c * m7).reshape(n, -1).sum(axis=1)
        share = np.where(total > 0, inside / np.maximum(total, 1e-9), np.nan)
        cover = m7.reshape(n, -1).mean(axis=1)                # what a uniform map would score
        lift = share - cover
        ok = np.isfinite(share)
        print(f"CAM of the {name}, {args.split} test, n = {n}:")
        print(f"  mass inside the leaf: mean {100 * np.nanmean(share):.1f}% (a uniform map: {100 * cover.mean():.1f}%, "
              f"so the lift is {100 * np.nanmean(lift):+.1f} points); median {100 * np.nanmedian(share):.1f}%; "
              f"images with less than half inside: {100 * (share[ok] < 0.5).mean():.1f}%; "
              f"below a uniform map: {100 * (lift[ok] < 0).mean():.1f}%")
        out[name] = dict(mean_share=float(np.nanmean(share)), median_share=float(np.nanmedian(share)),
                         uniform=float(cover.mean()), lift=float(np.nanmean(lift)),
                         frac_below_half=float((share[ok] < 0.5).mean()), frac_below_uniform=float((lift[ok] < 0).mean()))
        if k == 0:
            per = sorted(((classes[c_], float(np.nanmean(share[labels == c_])), int((labels == c_).sum()))
                          for c_ in range(N_CLASSES) if (labels == c_).any()), key=lambda t: t[1])
            print("  lowest by class: " + ", ".join(f"{c_[:30]} {100 * v:.0f}%" for c_, v, _ in per[:5]))
            print("  highest: " + ", ".join(f"{c_[:30]} {100 * v:.0f}%" for c_, v, _ in per[-3:]))
            out["per_class"] = per
            if f"{args.split}_test_nn_mad" in meta:
                leaked = meta[f"{args.split}_test_nn_mad"] < 6.0
                if leaked.any():
                    print(f"  leaked images (n={int(leaked.sum())}): {100 * np.nanmean(share[leaked]):.1f}% inside; "
                          f"others: {100 * np.nanmean(share[~leaked]):.1f}%")
            correct = pred == labels
            print(f"  correctly classified (n={int(correct.sum())}): {100 * np.nanmean(share[correct]):.1f}% inside; "
                  f"misclassified (n={int((~correct).sum())}): {100 * np.nanmean(share[~correct]):.1f}%")
            out["by_correct"] = dict(correct=float(np.nanmean(share[correct])), wrong=float(np.nanmean(share[~correct])) if (~correct).any() else None)
            np.save(args.cam.replace(".bin", "_share.npy"), np.stack([share, cover]))
            if args.examples:
                order = np.argsort(np.where(ok, share, np.inf))
                print("  most-outside images (test index, class, share):",
                      [(int(te[i]), classes[labels[i]][:24], round(100 * share[i])) for i in order[:args.examples]])
                print("  most-inside:", [(int(te[i]), classes[labels[i]][:24], round(100 * share[i])) for i in order[ok.sum() - args.examples:ok.sum()]])
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
