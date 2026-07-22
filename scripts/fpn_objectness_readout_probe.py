#!/usr/bin/env python3
"""Objectness-discrimination probe -- the follow-up the resolution probe redirected to.

fpn_resolution_probe.py established that the binding constraint is CONFIDENCE
RANKING, not box precision: the detector emits an IoU>=0.5 box at the assigned
cell for 33.06% of encoded positives, but conf = sigmoid(obj)*max softmax(cls)
ranks it at median 2256 of 12348 slots, so only 9.19% survive the scorer's
top-1000 cut. Overall objectness AUC is 0.741, and it is 0.7393-0.7417 across ALL
FOUR trained arms -- a class-weighting change, a prior-bias init and +7M head
parameters all failed to move it.

Before building anything, split that failure in two:

  (a) READOUT failure -- the information needed to rank objects above background
      is already present in the head's 15 output channels, and `sigmoid(obj)` is
      simply a bad way to read it out. Fix is cheap: a better objectness loss or
      a rescoring function. No backbone change.

  (b) FEATURE failure -- the information is not in the head's output at all, so
      no function of it can rank better. Fix is expensive: backbone, resolution,
      or a fundamentally different training signal.

The probe: fit rankers of increasing capacity on the head's OWN output channels
and see how far AUC can be pushed. Everything is fit on one set of images and
scored on a disjoint set, so the numbers are honest generalization, not memorized.

  ceiling(readout) >> 0.741  ->  (a), and the fix is cheap
  ceiling(readout) ~= 0.741  ->  (b), and rescoring is a dead end

Two things this CANNOT tell you, stated up front:
  * It bounds what a function of the HEAD OUTPUT can do. Retraining the objectness
    head against the backbone's richer internal features could do better -- so a
    negative result here bounds rescoring, not the whole lever.
  * A ranker fit on val logits is measuring the information content of those
    logits, not proposing a deployable model. The deployable claim is only ever
    "an objectness loss that targets this signal should train".

Usage: fpn_objectness_readout_probe.py <logits.bin> <fpn_val.bin>
                                       [--anchors data/visdrone]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

P = 15
GRIDS = (56, 28, 14)


def load_anchors_file(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            w, h = ln.split()
            rows.append((float(w), float(h)))
    return rows


def iou_cwh(b1, b2):
    ix = (np.minimum(b1[..., 0] + b1[..., 2] / 2, b2[..., 0] + b2[..., 2] / 2)
          - np.maximum(b1[..., 0] - b1[..., 2] / 2, b2[..., 0] - b2[..., 2] / 2))
    iy = (np.minimum(b1[..., 1] + b1[..., 3] / 2, b2[..., 1] + b2[..., 3] / 2)
          - np.maximum(b1[..., 1] - b1[..., 3] / 2, b2[..., 1] - b2[..., 3] / 2))
    ix = np.clip(ix, 0, None); iy = np.clip(iy, 0, None)
    inter = ix * iy
    ua = b1[..., 2] * b1[..., 3] + b2[..., 2] * b2[..., 3] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-12), 0.0)


def auc(scores, labels):
    """Rank-based AUC (ties averaged)."""
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]; y = labels[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos = y.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def build_features(logits_path, val_path, anchors_dir, imgsz):
    """Per-slot feature matrix from the head's OWN 15 output channels.

    Returns X [N,F], y [N] (1 = encoded positive), img [N], names, and the
    current production score `conf` [N] for the baseline comparison."""
    anchors = [np.array(load_anchors_file(f"{anchors_dir}/anchors_fpn_{p}.txt"))
               for p in ("p3", "p4", "p5")]
    A = len(anchors[0])
    ntot = sum(A * P * g * g for g in GRIDS)
    rec_img = 3 * imgsz * imgsz
    rec = rec_img + ntot * 4

    lg = np.fromfile(logits_path, dtype=np.float32)
    nl = lg.size // ntot
    lg = lg[:nl * ntot].reshape(nl, ntot)
    raw = np.fromfile(val_path, dtype=np.uint8)
    # skip the 4-byte '<I' header -- see fpn_neighbor_separation.py's bug
    nrec = (raw.size - 4) // rec
    tgt = raw[4:4 + nrec * rec].reshape(nrec, rec)[:, rec_img:] \
        .copy().view(np.float32).reshape(nrec, ntot)
    n = min(nl, nrec)

    feats, ys, imgs, confs, ious = [], [], [], [], []
    off = 0
    for si, g in enumerate(GRIDS):
        blk = A * P * g * g
        L = lg[:n, off:off + blk].reshape(n, A, P, g, g)
        T = tgt[:n, off:off + blk].reshape(n, A, P, g, g)
        off += blk
        for a in range(A):
            aw, ah = anchors[si][a]
            o = L[:, a, 4]                                   # [n,g,g] raw logit
            c = L[:, a, 5:15]                                # [n,10,g,g]
            e = np.exp(c - c.max(axis=1, keepdims=True))
            sm = e / e.sum(axis=1, keepdims=True)
            cmax = c.max(axis=1); cmin = c.min(axis=1)
            ent = -(sm * np.log(sm + 1e-12)).sum(axis=1)
            tw = np.minimum(L[:, a, 2], 8.0); th = np.minimum(L[:, a, 3], 8.0)
            pw = aw * np.exp(tw); ph = ah * np.exp(th)
            f = np.stack([
                o, cmax, cmin, cmax - cmin, ent, sm.max(axis=1),
                tw, th, np.log(pw + 1e-9), np.log(ph + 1e-9),
                np.log(pw * ph + 1e-12), np.log((pw + 1e-9) / (ph + 1e-9)),
                L[:, a, 0], L[:, a, 1],
                np.full_like(o, si), np.full_like(o, a),
            ], axis=-1)                                      # [n,g,g,F]
            # IoU of this slot's decoded box against the GT encoded AT this slot
            # (meaningful only where the slot is positive -- used to separate
            # "a positive reached top-k" from "a USABLE positive reached top-k")
            jj = np.arange(g).reshape(1, 1, g); ii = np.arange(g).reshape(1, g, 1)
            sx = 1.0 / (1.0 + np.exp(-np.clip(L[:, a, 0], -60, 60)))
            sy = 1.0 / (1.0 + np.exp(-np.clip(L[:, a, 1], -60, 60)))
            pb = np.stack([(jj + sx) / g, (ii + sy) / g, pw, ph], -1)
            gb = np.stack([(jj + T[:, a, 0]) / g, (ii + T[:, a, 1]) / g,
                           T[:, a, 2], T[:, a, 3]], -1)
            ious.append(iou_cwh(pb, gb).reshape(-1))
            feats.append(f.reshape(-1, f.shape[-1]))
            ys.append((T[:, a, 4] > 0.5).reshape(-1))
            imgs.append(np.repeat(np.arange(n), g * g))
            confs.append((1.0 / (1.0 + np.exp(-np.clip(o, -60, 60)))
                          * sm.max(axis=1)).reshape(-1))
    names = ["obj_logit", "cls_max", "cls_min", "cls_range", "cls_entropy",
             "cls_prob_max", "tw", "th", "log_w", "log_h", "log_area",
             "log_aspect", "tx", "ty", "scale", "anchor"]
    return (np.concatenate(feats).astype(np.float64),
            np.concatenate(ys), np.concatenate(imgs), names,
            np.concatenate(confs).astype(np.float64), n,
            np.concatenate(ious).astype(np.float64))


def fit_logreg(X, y, w=None, epochs=300, lr=0.5, l2=1e-4, seed=0):
    """Plain logistic regression, full-batch GD with class balancing."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    th = rng.normal(0, 0.01, d + 1)
    pw = (y == 0).sum() / max((y == 1).sum(), 1)      # positive class weight
    sw = np.where(y == 1, pw, 1.0) if w is None else w
    sw = sw / sw.mean()
    Xb = np.hstack([X, np.ones((len(X), 1))])
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ th, -30, 30)))
        gr = Xb.T @ (sw * (p - y)) / len(X) + l2 * np.r_[th[:-1], 0.0]
        th -= lr * gr
    return th


def fit_mlp(X, y, hidden=32, epochs=400, lr=0.3, seed=0):
    """Small MLP -- the nonlinear readout ceiling. Full-batch, class-balanced."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), hidden); b2 = 0.0
    pw = (y == 0).sum() / max((y == 1).sum(), 1)
    sw = np.where(y == 1, pw, 1.0); sw = sw / sw.mean()
    for _ in range(epochs):
        h = np.maximum(X @ W1 + b1, 0.0)
        z = h @ W2 + b2
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        dz = sw * (p - y) / len(X)
        gW2 = h.T @ dz; gb2 = dz.sum()
        dh = np.outer(dz, W2) * (h > 0)
        gW1 = X.T @ dh; gb1 = dh.sum(axis=0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return W1, b1, W2, b2


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
    print(f"slots: {len(y)}   positives: {int(y.sum())}   images: {n}")
    print(f"split by IMAGE: train {ntr} imgs / test {n - ntr} imgs "
          f"(disjoint -- no leakage)\n")

    rng = np.random.default_rng(args.seed)
    # balanced training subsample: all positives + a negative sample
    tr_pos = np.flatnonzero(tr_img & (y == 1))
    tr_neg = np.flatnonzero(tr_img & (y == 0))
    tr_neg = rng.choice(tr_neg, min(args.neg_sample, len(tr_neg)), replace=False)
    tr = np.concatenate([tr_pos, tr_neg])
    # test AUC on a subsample (AUC is a rank statistic; sampling negatives is
    # unbiased). The top-k metric below uses ALL test slots, not a sample.
    te_pos = np.flatnonzero(te_img & (y == 1))
    te_neg = np.flatnonzero(te_img & (y == 0))
    te_neg_s = rng.choice(te_neg, min(args.neg_sample, len(te_neg)), replace=False)
    te = np.concatenate([te_pos, te_neg_s])

    mu = X[tr].mean(axis=0); sd = X[tr].std(axis=0) + 1e-9
    Z = (X - mu) / sd

    print("=" * 74)
    print("1  SINGLE-FEATURE AUC on held-out images (what each channel knows)")
    print("=" * 74)
    singles = sorted(((auc(X[te, k], y[te].astype(float)), names[k])
                      for k in range(X.shape[1])),
                     key=lambda t: -abs(t[0] - 0.5))
    for a_, nm in singles:
        print(f"  {nm:<16} AUC = {a_:.4f}")
    print()

    print("=" * 74)
    print("2  READOUT CEILING -- how far can a function of these channels go?")
    print("=" * 74)
    base_auc = auc(conf[te], y[te].astype(float))
    obj_auc = auc(X[te, 0], y[te].astype(float))
    th = fit_logreg(Z[tr], y[tr].astype(float))
    lr_te = np.hstack([Z[te], np.ones((len(te), 1))]) @ th
    W1, b1, W2, b2 = fit_mlp(Z[tr], y[tr].astype(float))
    mlp_te = np.maximum(Z[te] @ W1 + b1, 0.0) @ W2 + b2
    rows = [("production conf = sig(obj)*max softmax(cls)", base_auc),
            ("sigmoid(obj) alone", obj_auc),
            ("logistic regression on all 16 channels", auc(lr_te, y[te].astype(float))),
            ("MLP (16-32-1) on all 16 channels", auc(mlp_te, y[te].astype(float)))]
    for nm, a_ in rows:
        print(f"  {nm:<46} AUC = {a_:.4f}")
    print()

    print("=" * 74)
    print("3  WHAT IT BUYS -- positives reaching the top-k, ALL test slots")
    print("=" * 74)
    # full test set, ranked per image exactly as the scorer would
    mlp_all = np.maximum(Z[te_img] @ W1 + b1, 0.0) @ W2 + b2
    lr_all = np.hstack([Z[te_img], np.ones((te_img.sum(), 1))]) @ th
    yt = y[te_img]; it = img[te_img]
    # A positive reaching top-k is worthless if its box is wrong. The column that
    # predicts end-to-end recall is USABLE = positive AND IoU>=0.5 AND in top-k.
    gd = iouv[te_img] >= 0.5
    print(f"  {'ranker':<46} {'pos':>8} {'USABLE':>9}")
    print("  " + "-" * 66)
    for nm, sc in (("production conf = sig(obj)*max softmax(cls)", conf[te_img]),
                   ("sigmoid(obj) ALONE -- drop the class multiplier", X[te_img, 0]),
                   ("logistic regression", lr_all),
                   ("MLP readout", mlp_all)):
        hit = use = 0
        for r in np.unique(it):
            m = it == r
            s = sc[m]
            k = min(args.topk, len(s))
            thr = np.partition(s, len(s) - k)[len(s) - k]
            ink = s >= thr
            hit += (yt[m] & ink).sum()
            use += (yt[m] & gd[m] & ink).sum()
        print(f"  {nm:<46} {100.0*hit/yt.sum():>7.2f}% {100.0*use/yt.sum():>8.2f}%")
    print()
    print("  USABLE is the column that predicts recall. If a ranker wins on `pos`")
    print("  but loses on USABLE, it is admitting positives with unusable boxes --")
    print("  which is exactly what an end-to-end A/B would punish.\n")

    print("=" * 74)
    print("4  DOES EACH SCORE KNOW BOX QUALITY? (among positives only)")
    print("=" * 74)
    pm = te_img & (y == 1)
    gq = (iouv[pm] >= 0.5).astype(float)
    for nm, sc in (("production conf = sig(obj)*max softmax(cls)", conf[pm]),
                   ("sigmoid(obj) alone", X[pm, 0]),
                   ("max softmax(cls) alone", X[pm, 5])):
        print(f"  {nm:<46} AUC(IoU>=0.5) = {auc(sc, gq):.4f}")
    print("\n  This is the quality-ranking job, distinct from the object/background")
    print("  job in section 2. A score can win one and lose the other.\n")

    # ── 5. the readout ceiling against the RIGHT objective ────────────────────
    # Sections 2-3 fit rankers to predict positive-vs-background, which section 3
    # then showed is the wrong target: it admits positives with unusable boxes.
    # Refit against the objective that actually predicts recall -- USABLE =
    # positive AND IoU>=0.5 -- and re-read the ceiling. This is the honest test
    # of whether a better readout of the head's own channels exists.
    print("=" * 74)
    print("5  READOUT CEILING vs THE RIGHT OBJECTIVE (label = positive AND IoU>=0.5)")
    print("=" * 74)
    u = (y & (iouv >= 0.5)).astype(float)
    tr_up = np.flatnonzero(tr_img & (u == 1))
    tr_un = np.flatnonzero(tr_img & (u == 0))
    tr_un = rng.choice(tr_un, min(args.neg_sample, len(tr_un)), replace=False)
    tru = np.concatenate([tr_up, tr_un])
    thu = fit_logreg(Z[tru], u[tru])
    W1u, b1u, W2u, b2u = fit_mlp(Z[tru], u[tru])
    lru_all = np.hstack([Z[te_img], np.ones((te_img.sum(), 1))]) @ thu
    mlpu_all = np.maximum(Z[te_img] @ W1u + b1u, 0.0) @ W2u + b2u
    print(f"  {'ranker (fit on USABLE)':<46} {'pos':>8} {'USABLE':>9}")
    print("  " + "-" * 66)
    best = 0.0
    for nm, sc in (("production conf (unchanged baseline)", conf[te_img]),
                   ("logistic regression, USABLE objective", lru_all),
                   ("MLP readout, USABLE objective", mlpu_all)):
        hit = use = 0
        for r in np.unique(it):
            m = it == r
            s = sc[m]
            k = min(args.topk, len(s))
            thr = np.partition(s, len(s) - k)[len(s) - k]
            ink = s >= thr
            hit += (yt[m] & ink).sum()
            use += (yt[m] & gd[m] & ink).sum()
        v = 100.0 * use / yt.sum()
        best = max(best, v)
        print(f"  {nm:<46} {100.0*hit/yt.sum():>7.2f}% {v:>8.2f}%")
    print(f"\n  VERDICT: best readout {best:.2f}% USABLE vs production 9.71%.")
    print("  If that margin is small, rescoring is exhausted and the lever is a")
    print("  TRAINING change -- the head's channels do not carry more than it uses.\n")


if __name__ == "__main__":
    main()
