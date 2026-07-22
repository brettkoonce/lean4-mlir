#!/usr/bin/env python3
"""IoU-target probe -- the gate on the IoU-aware-objectness training run.

fpn_objectness_readout_probe.py named the defect: among positives, sigmoid(obj)
predicts box quality at AUC 0.3497 (worse than chance), because every positive is
trained toward a constant objectness target of 1.0 no matter how bad the box that
slot emits. The proposed fix is to make the objectness TARGET the achieved IoU
(YOLOv5-style IoU-aware objectness / quality focal loss).

That fix costs a ~4.5 h training run. Gate it first, on existing logits.

The question is NOT "would perfect quality ranking help" -- trivially yes. It is:
**given the object/background signal the head already has, how much does fixing
the quality axis actually buy?** So every scenario below reuses the MEASURED
sigmoid(obj) for object/background and varies only the quality term:

  production      obj x max softmax(cls)      -- today, 9.71% USABLE
  learnable       obj x IoU-predicted-from-the-head's-own-channels  (LOWER bound:
                  a retrained head sees richer internal features than its output)
  oracle quality  obj x TRUE IoU              -- the ceiling of the lever itself
  oracle obj      perfect object/background   -- what is left for OTHER levers

USABLE = positive AND IoU>=0.5 AND inside the scorer's top-k. It is the quantity
that predicted the end-to-end A/B correctly last time, when "positives in top-k"
lied.

Reading it:
  oracle quality ~= oracle obj  -> IoU targeting is most of the remaining lever
  oracle quality ~= production  -> it is not, and the run should not be spent
  learnable near oracle quality -> the signal is already visible in the output,
                                   so the retrain has something to grab

Usage: fpn_iou_target_probe.py <logits.bin> <fpn_val.bin> [--anchors data/visdrone]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.fpn_objectness_readout_probe import (  # noqa: E402
    auc, build_features, fit_mlp)


def fit_mlp_soft(X, t, hidden=32, epochs=400, lr=0.3, seed=0):
    """MLP with BCE against a SOFT target t in [0,1] -- so the same code fits both
    the binary objectness target (t in {0,1}) and the IoU-aware target (t = IoU of
    the box that slot emits, 0 on background). That is the whole point of the
    comparison: identical features, identical capacity, only the TARGET differs."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), hidden); b2 = 0.0
    pos = t > 0.5
    pw = (~pos).sum() / max(pos.sum(), 1)
    sw = np.where(pos, pw, 1.0); sw = sw / sw.mean()
    for _ in range(epochs):
        h = np.maximum(X @ W1 + b1, 0.0)
        p = 1.0 / (1.0 + np.exp(-np.clip(h @ W2 + b2, -30, 30)))
        dz = sw * (p - t) / len(X)
        gW2 = h.T @ dz; gb2 = dz.sum()
        dh = np.outer(dz, W2) * (h > 0)
        W1 -= lr * (X.T @ dh); b1 -= lr * dh.sum(axis=0)
        W2 -= lr * gW2; b2 -= lr * gb2
    return W1, b1, W2, b2


def topk_stats(score, y, good, img, k):
    """Per-image top-k, exactly as the scorer cuts. Returns (pos%, usable%)."""
    hit = use = 0
    for r in np.unique(img):
        m = img == r
        s = score[m]
        kk = min(k, len(s))
        thr = np.partition(s, len(s) - kk)[len(s) - kk]
        ink = s >= thr
        hit += (y[m] & ink).sum()
        use += (y[m] & good[m] & ink).sum()
    return 100.0 * hit / y.sum(), 100.0 * use / y.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits"); ap.add_argument("valbin")
    ap.add_argument("--anchors", default="data/visdrone")
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--neg-sample", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y, img, names, conf, n, iouv = build_features(
        args.logits, args.valbin, args.anchors, args.imgsz)
    ntr = int(n * 0.7)
    tr_img = img < ntr; te_img = ~tr_img
    rng = np.random.default_rng(args.seed)
    good = iouv >= 0.5
    sig_obj = 1.0 / (1.0 + np.exp(-np.clip(X[:, 0], -60, 60)))
    print(f"slots {len(y)}   positives {int(y.sum())}   images {n}   "
          f"train {ntr} / test {n - ntr} (disjoint)\n")

    # ── 1. is the achieved IoU predictable from the head's OWN output? ────────
    # Fit on POSITIVES ONLY (that is the population an IoU target applies to),
    # excluding the objectness channel itself, since a retrained obj head is
    # exactly what is being simulated -- feeding it in would be circular.
    keep = [i for i, nm in enumerate(names) if nm != "obj_logit"]
    Xq = X[:, keep]
    mu = Xq[tr_img].mean(0); sd = Xq[tr_img].std(0) + 1e-9
    Zq = (Xq - mu) / sd
    trp = np.flatnonzero(tr_img & y)
    tep = np.flatnonzero(te_img & y)
    W1, b1, W2, b2 = fit_mlp(Zq[trp], good[trp].astype(float))
    qpred_all = np.maximum(Zq @ W1 + b1, 0.0) @ W2 + b2
    qpred_all = 1.0 / (1.0 + np.exp(-np.clip(qpred_all, -30, 30)))

    print("=" * 76)
    print("1  IS THE ACHIEVED IoU PREDICTABLE FROM THE HEAD'S OWN CHANNELS?")
    print("=" * 76)
    print("   (fit on positives in TRAIN images, scored on positives in TEST images;")
    print("    the objectness channel is EXCLUDED -- that is what would be retrained)\n")
    gq = good[tep].astype(float)
    for nm, sc in (("sigmoid(obj)  -- today's signal", sig_obj[tep]),
                   ("max softmax(cls)  -- best single channel", X[tep, 5]),
                   ("MLP on all non-objectness channels", qpred_all[tep])):
        print(f"   {nm:<44} AUC(IoU>=0.5) = {auc(sc, gq):.4f}")
    print()

    # ── 2. the controlled target comparison ───────────────────────────────────
    # THE TRAP, recorded because this probe fell into it first: ranking by
    # `obj x TRUE IoU` looks like a clean "oracle quality" ceiling, but iouv is 0
    # on every background slot, so that score LEAKS the positive/background label
    # and simply reproduces the oracle-obj row (both 33.65%). True IoU is not a
    # usable stand-in for a quality signal. The honest test is to fit the SAME
    # model on the SAME features and change ONLY the target.
    yt = y[te_img]; gt_ = good[te_img]; it = img[te_img]
    trs_p = np.flatnonzero(tr_img & y)
    trs_n = np.flatnonzero(tr_img & ~y)
    trs_n = rng.choice(trs_n, min(args.neg_sample, len(trs_n)), replace=False)
    trs = np.concatenate([trs_p, trs_n])
    t_bin = y.astype(float)                      # today's semantics
    t_iou = np.where(y, iouv, 0.0)               # IoU-aware objectness target

    print("=" * 76)
    print("2  CONTROLLED TARGET COMPARISON -- same features, same net, ONLY the")
    print("   target differs. This is the training change, simulated.")
    print("=" * 76)
    print(f"   {'ranker':<50} {'pos':>7} {'USABLE':>8}")
    print("   " + "-" * 68)
    res = {}
    p, u = topk_stats(conf[te_img], yt, gt_, it, args.topk)
    res["production"] = u
    print(f"   {'production: obj x max softmax(cls)':<50} {p:>6.2f}% {u:>7.2f}%")
    for nm, tgt in (("MLP, BINARY target (today's objectness)", t_bin),
                    ("MLP, IoU target (the proposed change)", t_iou)):
        Wa, ba, Wb, bb = fit_mlp_soft(Zq[trs], tgt[trs])
        sc = np.maximum(Zq[te_img] @ Wa + ba, 0.0) @ Wb + bb
        p, u = topk_stats(sc, yt, gt_, it, args.topk)
        res[nm] = u
        print(f"   {nm:<50} {p:>6.2f}% {u:>7.2f}%")

    orc = y[te_img].astype(float) + 1e-6 * conf[te_img]
    p, uo = topk_stats(orc, yt, gt_, it, args.topk)
    print(f"\n   {'ORACLE object/background (the ceiling for ANY obj fix)':<50} "
          f"{p:>6.2f}% {uo:>7.2f}%")
    print()

    # ── 3. verdict ────────────────────────────────────────────────────────────
    prod = res["production"]
    b = res["MLP, BINARY target (today's objectness)"]
    i = res["MLP, IoU target (the proposed change)"]
    print("=" * 76)
    print("3  VERDICT")
    print("=" * 76)
    print(f"   production                        {prod:>6.2f}% USABLE")
    print(f"   same net, binary target           {b:>6.2f}%   ({b - prod:+.2f} pts)")
    print(f"   same net, IoU target              {i:>6.2f}%   ({i - prod:+.2f} pts)")
    print(f"   IoU target vs binary, like for like        {i - b:+.2f} pts")
    print(f"   perfect object/background         {uo:>6.2f}%   ({uo - prod:+.2f} pts)")
    print()
    print("   The like-for-like line is the gate. It isolates the TARGET change")
    print("   from everything else, on features a real head would strictly")
    print("   improve on -- so it is a lower bound on the training run's payoff,")
    print("   but a lower bound measured rather than assumed.\n")


if __name__ == "__main__":
    main()
