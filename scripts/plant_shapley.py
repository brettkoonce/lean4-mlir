#!/usr/bin/env python3
"""Shapley values for the PlantVillage demo — planning/plant_lab_to_field_demo.md §3.6, §5.

Two explainers ask the same question of the same net — how much of the class evidence is the
leaf — and this is the model-agnostic one. Both modes need only the eval graph.

  two-player   The leaf region and the background are the two players; the "removed" region is
               filled with the image's median background colour, the same fill for all four
               counterfactuals (`_test`, `_test_leaf`, `_test_bg`, `_test_none`, written by
               `preprocess_plant.py --only-shapley-parts`). With two players the Shapley value is
               exact in those four evaluations:
                   φ_leaf = ½[(f(leaf) − f(none)) + (f(full) − f(bg))],  φ_bg likewise,
               and φ_leaf + φ_bg = f(full) − f(none) to the float (the efficiency axiom, checked).
               f is the logit of the true class (and of the predicted class, reported beside it).

      plant_shapley.py two-player --split pvg --logits <prefix>   # reads <prefix>_logits_pvg_test{,_leaf,_bg,_none}.bin

  grid         The 49 patches of the CAM's 7×7 grid are the players, on a few images; the
               Shapley value is estimated by permutation sampling — P random orderings, the
               marginal contribution of each patch when it joins the prefix before it — with a
               standard error over the orderings. Two steps: `--write` builds the probe part
               (one baseline + 49·P prefix images per picked image, label = true class), the
               trainer scores it with `plant-leaf eval extra=<probe.bin>`, and `--score` turns
               the logits into 7×7 maps.

      plant_shapley.py grid --write --split pvg --images 12,345,678 --perms 40 --out data/plant/shap_probe
      lake exe plant-leaf eval init=<...> extra=data/plant/shap_probe.bin ...
      plant_shapley.py grid --score <prefix>_logits_shap_probe.bin --design data/plant/shap_probe.json
"""
import argparse
import json
import math
import os

import numpy as np

N_CLASSES = 38
S = 224
G = 7
CELL = S // G


def read_part(path, side=S):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(np.frombuffer(bytes(raw[:4]), dtype=np.uint32)[0])
    rec = 1 + 3 * side * side
    body = np.asarray(raw[4:4 + n * rec]).reshape(n, rec)
    return body[:, 0].astype(np.int32), body[:, 1:].reshape(n, 3, side, side)


def write_part(path, labels, images):
    with open(path, "wb") as f:
        f.write(np.uint32(len(labels)).tobytes())
        for l, im in zip(labels, images):
            f.write(bytes([int(l)]))
            f.write(np.ascontiguousarray(im, dtype=np.uint8).tobytes())


def two_player(args):
    data = args.data
    parts = {k: os.path.join(data, f"{args.split}_test{s}.bin") for k, s in (("full", ""), ("leaf", "_test_leaf"), ("bg", "_test_bg"), ("none", "_test_none"))}
    parts["leaf"] = os.path.join(data, f"{args.split}_test_leaf.bin")
    parts["bg"] = os.path.join(data, f"{args.split}_test_bg.bin")
    parts["none"] = os.path.join(data, f"{args.split}_test_none.bin")
    labels, _ = read_part(parts["full"])
    n = len(labels)
    f = {}
    for k in ("full", "leaf", "bg", "none"):
        sfx = {"full": f"{args.split}_test", "leaf": f"{args.split}_test_leaf", "bg": f"{args.split}_test_bg", "none": f"{args.split}_test_none"}[k]
        lg = np.fromfile(f"{args.logits}_logits_{sfx}.bin", dtype=np.float32)
        assert lg.size == n * N_CLASSES, f"{k}: {lg.size} floats for {n} images"
        f[k] = lg.reshape(n, N_CLASSES)
    pred = f["full"].argmax(axis=1)
    rows = np.arange(n)
    out = {}
    for name, cls in (("true", labels), ("predicted", pred)):
        ff, fl, fb, fn = (f[k][rows, cls] for k in ("full", "leaf", "bg", "none"))
        phi_leaf = 0.5 * ((fl - fn) + (ff - fb))
        phi_bg = 0.5 * ((fb - fn) + (ff - fl))
        total = ff - fn
        eff = np.abs(phi_leaf + phi_bg - total).max()
        share = np.where(np.abs(total) > 1e-6, phi_leaf / total, np.nan)
        ok = np.isfinite(share)
        out[name] = dict(mean_share=float(np.nanmean(share)), median_share=float(np.nanmedian(share)),
                         frac_leaf_below_half=float((share[ok] < 0.5).mean()), frac_bg_positive=float((phi_bg > 0).mean()),
                         mean_phi_leaf=float(phi_leaf.mean()), mean_phi_bg=float(phi_bg.mean()), mean_total=float(total.mean()),
                         efficiency_max_abs_err=float(eff), n=int(n))
        print(f"two-player Shapley, {args.split} test, f = {name}-class logit, n = {n}:")
        print(f"  φ_leaf {phi_leaf.mean():+.3f}  φ_bg {phi_bg.mean():+.3f}  f(full)−f(none) {total.mean():+.3f}  "
              f"(efficiency: max |φ_leaf+φ_bg−Δf| = {eff:.2e})")
        print(f"  leaf share of the evidence: mean {100 * np.nanmean(share):.1f}%, median {100 * np.nanmedian(share):.1f}%; "
              f"images with the leaf under half: {100 * (share[ok] < 0.5).mean():.1f}%; background helps (φ_bg > 0): {100 * (phi_bg > 0).mean():.1f}%")
        if name == "true":
            classes = [str(c) for c in np.load(os.path.join(data, "meta_plant.npz"))["classes"]]
            per = [(classes[c].replace("___", " ")[:34], float(np.nanmean(share[labels == c]))) for c in range(N_CLASSES) if (labels == c).any()]
            per.sort(key=lambda t: t[1])
            print("  lowest leaf share by class:", ", ".join(f"{c} {100 * v:.0f}%" for c, v in per[:5]))
            print("  highest:", ", ".join(f"{c} {100 * v:.0f}%" for c, v in per[-3:]))
            out["per_class_share"] = per
            np.save(f"{args.logits}_shapley2_{args.split}.npy", np.stack([phi_leaf, phi_bg, total, share]))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
        print(f"wrote {args.json}")


def grid_write(args):
    labels, imgs = read_part(os.path.join(args.data, f"{args.split}_test.bin"))
    masks = np.load(os.path.join(args.data, f"{args.split}_test_mask.npy"))
    ids = [int(i) for i in args.images.split(",")]
    rng = np.random.RandomState(args.seed)
    design = dict(split=args.split, images=ids, perms=args.perms, order=[])
    out_l, out_i = [], []
    for i in ids:
        im = imgs[i]
        m = masks[i]
        fill = (np.median(im[:, ~m], axis=1) if (~m).any() else np.array([128, 128, 128])).astype(np.uint8)
        base = np.empty_like(im)
        base[:] = fill[:, None, None]
        out_l.append(labels[i]); out_i.append(base)                       # the empty coalition
        perms = [rng.permutation(G * G) for _ in range(args.perms)]
        design["order"].append([p.tolist() for p in perms])
        for p in perms:
            cur = base.copy()
            for patch in p:
                r, c = divmod(int(patch), G)
                cur[:, r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = im[:, r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
                out_l.append(labels[i]); out_i.append(cur.copy())
    write_part(args.out + ".bin", out_l, out_i)
    json.dump(design, open(args.out + ".json", "w"))
    print(f"wrote {args.out}.bin: {len(out_l)} records ({len(ids)} images × (1 + {G * G}·{args.perms})) and {args.out}.json")


def grid_score(args):
    design = json.load(open(args.design))
    labels, _ = read_part(os.path.join(args.data, f"{design['split']}_test.bin"))
    lg = np.fromfile(args.score, dtype=np.float32).reshape(-1, N_CLASSES)
    P = design["perms"]
    per_img = 1 + G * G * P
    assert lg.shape[0] == per_img * len(design["images"]), lg.shape
    maps, ses, checks = [], [], []
    for k, i in enumerate(design["images"]):
        cls = labels[i]
        f = lg[k * per_img:(k + 1) * per_img, cls]
        f_none = f[0]
        phi = np.zeros((P, G * G))
        for pi, order in enumerate(design["order"][k]):
            seq = f[1 + pi * G * G:1 + (pi + 1) * G * G]
            prev = f_none
            for step, patch in enumerate(order):
                phi[pi, patch] = seq[step] - prev
                prev = seq[step]
        mean, se = phi.mean(axis=0), phi.std(axis=0, ddof=1) / math.sqrt(P)
        f_full = f[G * G]                                          # the first permutation's full coalition
        checks.append(float(mean.sum() - (f_full - f_none)))
        maps.append(mean.reshape(G, G)); ses.append(se.reshape(G, G))
        print(f"image {i} (class {cls}): Σφ = {mean.sum():+.3f}, f(full)−f(none) = {f_full - f_none:+.3f}, "
              f"mean SE {se.mean():.3f}, top patch {int(mean.argmax())} φ = {mean.max():+.3f}")
    np.savez(args.score.replace(".bin", "_maps.npz"), images=np.array(design["images"]), phi=np.stack(maps), se=np.stack(ses), eff_err=np.array(checks))
    print(f"wrote {args.score.replace('.bin', '_maps.npz')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("two-player", "grid"))
    ap.add_argument("--data", default="data/plant")
    ap.add_argument("--split", default="pvg", choices=("pv", "pvg"))
    ap.add_argument("--logits", help="two-player: the run prefix (before _logits_...)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--score", default=None, help="grid: the probe's logits file")
    ap.add_argument("--design", default=None, help="grid --score: the probe's .json")
    ap.add_argument("--images", default="", help="grid --write: comma-separated test indices")
    ap.add_argument("--perms", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/plant/shap_probe")
    args = ap.parse_args()
    if args.mode == "two-player":
        two_player(args)
    elif args.write:
        grid_write(args)
    else:
        grid_score(args)


if __name__ == "__main__":
    main()
