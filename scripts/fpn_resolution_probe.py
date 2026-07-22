#!/usr/bin/env python3
"""Resolution probe -- the pre-measurement for planning/yolo_assignment.md's
"suggested next lever 1" (input resolution / P2 scale).

Bite 0 of the assignment thread ended by REFRAMING the constraint: the detector
already finds 68% of the objects at IoU 0.10 but only 11.8% at IoU 0.50, so what
is missing is BOX PRECISION, not assignment, ranking or duplicates. The proposed
fix is more input resolution (448 -> 768 / a stride-4 P2 level), on the theory
that a 2-5px VisDrone object simply does not carry enough pixels to regress a box
to IoU 0.5.

Before spending an 8.3 GB re-encode + ~4.5 h train, measure whether that theory
can possibly pay. Everything here is numpy on the EXISTING e12 logits -- no
training, no GPU, no Lean.

Three parts:

  A. WHERE THE IoU ACTUALLY GOES (model-free). At each GT's own assigned positive
     cell -- the best case, the cell that was trained on it -- decompose the IoU
     shortfall by substituting the GT's own centre / own w,h into the prediction.
     If perfect centres alone clear IoU 0.5, the constraint is localization and
     resolution is the right lever. If perfect centres leave IoU pinned, the size
     regression is the binding term and more pixels will not save it.

  B. THE ERROR IN RESOLUTION-INVARIANT UNITS (model-free). Centre error measured
     in relative units, in P3 CELLS, and in GT-object DIAMETERS. The cell figure
     is the one that transfers: if the head localizes to a fixed fraction of a
     cell, then raising R shrinks the error in relative units by 448/R while the
     GT box keeps its relative size -- that is the entire mechanism by which
     resolution could buy IoU.

  C. THE TRANSFER PREDICTION (model-based -- labelled as such). Rescale the
     measured centre error by 448/R and re-score frac(IoU>=0.5) at a ladder of
     resolutions, under an optimistic and a null model. Also inverts the question:
     what error-shrink factor (hence what R) would be needed to reach a target
     recall, and is that R reachable at all.

The honest caveat, stated up front: C is an EXTRAPOLATION, not a measurement of a
detector trained at 768. It is a GATE, not a result. Its job is to be cheap enough
to run before the train and decisive enough to cancel it -- if even the optimistic
model leaves recall@0.5 flat, resolution is refuted for free.

Usage: fpn_resolution_probe.py <logits.bin> <fpn_val.bin> [--anchors data/visdrone]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

P = 15
GRIDS = (56, 28, 14)
SCALE_NAMES = ("P3", "P4", "P5")


def load_anchors_file(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            w, h = ln.split()
            rows.append((float(w), float(h)))
    return rows


def iou_cwh(b1, b2):
    """b*: [...,4] as (cx,cy,w,h) -> elementwise IoU."""
    ix = (np.minimum(b1[..., 0] + b1[..., 2] / 2, b2[..., 0] + b2[..., 2] / 2)
          - np.maximum(b1[..., 0] - b1[..., 2] / 2, b2[..., 0] - b2[..., 2] / 2))
    iy = (np.minimum(b1[..., 1] + b1[..., 3] / 2, b2[..., 1] + b2[..., 3] / 2)
          - np.maximum(b1[..., 1] - b1[..., 3] / 2, b2[..., 1] - b2[..., 3] / 2))
    ix = np.clip(ix, 0, None); iy = np.clip(iy, 0, None)
    inter = ix * iy
    ua = b1[..., 2] * b1[..., 3] + b2[..., 2] * b2[..., 3] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-12), 0.0)


def collect(logits_path, val_path, anchors_dir, imgsz, want_rank=False):
    """Return per-positive-slot arrays: gt box, predicted box, cell size (rel).

    With want_rank, also return, for each positive slot, the confidence RANK of
    that slot within its own image over ALL A*(56^2+28^2+14^2) candidate slots
    (conf = sigmoid(obj) * softmax(cls).max(), exactly the scorer's definition),
    and whether the slot's argmax class equals the GT class."""
    anchors = [np.array(load_anchors_file(f"{anchors_dir}/anchors_fpn_{p}.txt"))
               for p in ("p3", "p4", "p5")]
    A = len(anchors[0])
    ntot = sum(A * P * g * g for g in GRIDS)
    rec_img = 3 * imgsz * imgsz
    rec = rec_img + ntot * 4

    logits = np.fromfile(logits_path, dtype=np.float32)
    nl = logits.size // ntot
    logits = logits[:nl * ntot].reshape(nl, ntot)
    raw = np.fromfile(val_path, dtype=np.uint8)
    # NOTE: skip the 4-byte '<I' record-count header. Reading from byte 0 shifts
    # every target by one float32 == one cell along j -- the bug that was found
    # and fixed in fpn_neighbor_separation.py. Do not remove this.
    nrec = (raw.size - 4) // rec
    tgt = raw[4:4 + nrec * rec].reshape(nrec, rec)[:, rec_img:] \
        .copy().view(np.float32).reshape(nrec, ntot)
    n = min(nl, nrec)

    gts, preds, cells, scales = [], [], [], []
    recs, confs, clsok = [], [], []
    conf_all, cats = [], []             # per-scale [n, A*g*g] conf / category
    off = 0
    for si, g in enumerate(GRIDS):
        blk = A * P * g * g
        L = logits[:n, off:off + blk].reshape(n, A, P, g, g)
        T = tgt[:n, off:off + blk].reshape(n, A, P, g, g)
        off += blk
        if want_rank:
            o = 1.0 / (1.0 + np.exp(-np.clip(L[:, :, 4], -60, 60)))
            c = L[:, :, 5:15]
            e = np.exp(c - c.max(axis=2, keepdims=True))
            conf_all.append((o * (e.max(axis=2) / e.sum(axis=2))).reshape(n, -1))
            # slot category: 2 = positive, 1 = ring (Chebyshev-1 of ANY positive
            # cell at this scale), 0 = far background. Same split as
            # fpn_neighbor_separation.py, so the two diagnostics are comparable.
            posa = T[:, :, 4] > 0.5                      # [n,A,g,g]
            anyp = posa.any(axis=1)                      # [n,g,g]
            dil = np.zeros_like(anyp)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    dil |= np.roll(np.roll(anyp, dy, axis=1), dx, axis=2)
            cat = np.where(posa, 2,
                           np.where(dil[:, None, :, :], 1, 0)).astype(np.uint8)
            cats.append(cat.reshape(n, -1))
        for a in range(A):
            aw, ah = anchors[si][a]
            idx = np.argwhere(T[:, a, 4] > 0.5)
            if idx.size == 0:
                continue
            r_, ci_, cj_ = idx[:, 0], idx[:, 1], idx[:, 2]
            gt = np.stack([(cj_ + T[r_, a, 0, ci_, cj_]) / g,
                           (ci_ + T[r_, a, 1, ci_, cj_]) / g,
                           T[r_, a, 2, ci_, cj_], T[r_, a, 3, ci_, cj_]], -1)
            sx = 1.0 / (1.0 + np.exp(-np.clip(L[r_, a, 0, ci_, cj_], -60, 60)))
            sy = 1.0 / (1.0 + np.exp(-np.clip(L[r_, a, 1, ci_, cj_], -60, 60)))
            pw = aw * np.exp(np.minimum(L[r_, a, 2, ci_, cj_], 8.0))
            ph = ah * np.exp(np.minimum(L[r_, a, 3, ci_, cj_], 8.0))
            pr = np.stack([(cj_ + sx) / g, (ci_ + sy) / g, pw, ph], -1)
            gts.append(gt); preds.append(pr)
            cells.append(np.full(idx.shape[0], 1.0 / g))
            scales.append(np.full(idx.shape[0], si))
            if want_rank:
                recs.append(r_)
                # flat slot index within this scale's block, matching conf_all
                confs.append(np.stack([np.full(idx.shape[0], si),
                                       (a * g + ci_) * g + cj_], -1))
                pc = L[r_, a, 5:15, ci_, cj_].argmax(axis=-1)
                gc = T[r_, a, 5:15, ci_, cj_].argmax(axis=-1)
                clsok.append(pc == gc)
    out = [n, np.concatenate(gts), np.concatenate(preds),
           np.concatenate(cells), np.concatenate(scales)]
    if want_rank:
        allc = np.concatenate(conf_all, axis=1)          # [n, S]
        bounds = np.cumsum([0] + [c.shape[1] for c in conf_all])
        si_ = np.concatenate(confs)[:, 0]; fl_ = np.concatenate(confs)[:, 1]
        flat = bounds[si_] + fl_
        r_ = np.concatenate(recs)
        my = allc[r_, flat]
        # rank = how many slots in the same image score strictly higher
        order = np.argsort(-allc, axis=1)
        ranks = np.empty_like(allc, dtype=np.int32)
        np.put_along_axis(ranks, order,
                          np.broadcast_to(np.arange(allc.shape[1], dtype=np.int32),
                                          allc.shape), axis=1)
        out += [ranks[r_, flat], my, np.concatenate(clsok), allc.shape[1],
                allc, np.concatenate(cats, axis=1), r_, flat]
    return tuple(out)


def frac(v, t):
    return 100.0 * (v >= t).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits"); ap.add_argument("valbin")
    ap.add_argument("--anchors", default="data/visdrone")
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--ladder", default="448,640,768,896,1024,1280")
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()

    (n, gt, pr, cell, scale, rank, conf, clsok, nslots,
     allc, cats, prec, pflat) = collect(
        args.logits, args.valbin, args.anchors, args.imgsz, want_rank=True)
    R0 = args.imgsz
    print(f"records: {n}   positive slots: {gt.shape[0]}"
          f"   (header-corrected target read)\n")

    # ── A. where the IoU goes (model-free) ────────────────────────────────────
    base = iou_cwh(pr, gt)
    perfect_ctr = iou_cwh(np.stack([gt[:, 0], gt[:, 1], pr[:, 2], pr[:, 3]], -1), gt)
    perfect_wh = iou_cwh(np.stack([pr[:, 0], pr[:, 1], gt[:, 2], gt[:, 3]], -1), gt)

    print("=" * 78)
    print("A  WHERE THE IoU GOES -- at each GT's OWN assigned cell (model-free)")
    print("=" * 78)
    print(f"  {'variant':<38} {'mean IoU':>9} {'median':>8} "
          f"{'>=0.5':>8} {'>=0.25':>8} {'>=0.1':>8}")
    print("  " + "-" * 74)
    for nm, v in (("as predicted (both terms wrong)", base),
                  ("GT centre substituted (size only)", perfect_ctr),
                  ("GT w,h substituted (centre only)", perfect_wh)):
        print(f"  {nm:<38} {v.mean():>9.4f} {np.median(v):>8.4f} "
              f"{frac(v,0.5):>7.2f}% {frac(v,0.25):>7.2f}% {frac(v,0.1):>7.2f}%")
    print()

    # ── B. the error in resolution-invariant units (model-free) ───────────────
    d = np.hypot(pr[:, 0] - gt[:, 0], pr[:, 1] - gt[:, 1])
    diam = np.sqrt(gt[:, 2] * gt[:, 3])
    lw = np.log(np.maximum(pr[:, 2], 1e-12) / np.maximum(gt[:, 2], 1e-12))
    lh = np.log(np.maximum(pr[:, 3], 1e-12) / np.maximum(gt[:, 3], 1e-12))

    print("=" * 78)
    print("B  THE ERROR IN RESOLUTION-INVARIANT UNITS (model-free)")
    print("=" * 78)
    for nm, v in ((f"centre err / input px (@{R0})", d * R0),
                  ("centre err / own cell", d / cell),
                  ("centre err / GT diameter", d / np.maximum(diam, 1e-12)),
                  ("|log w_pred/w_gt|", np.abs(lw)),
                  ("|log h_pred/h_gt|", np.abs(lh))):
        print(f"  {nm:<32} mean={v.mean():>8.3f}  median={np.median(v):>8.3f}"
              f"  p90={np.percentile(v,90):>8.3f}")
    print(f"\n  GT size @ {R0}px input: median {np.median(diam)*R0:.2f} px, "
          f"p10 {np.percentile(diam,10)*R0:.2f} px, "
          f"p90 {np.percentile(diam,90)*R0:.2f} px")
    print(f"  median GT diameter = {np.median(diam/cell):.3f} of its own cell")
    print()

    # ── C. the transfer prediction (MODEL-BASED) ──────────────────────────────
    # Optimistic: the head keeps the SAME accuracy measured in cells, so relative
    # centre error scales by R0/R. Size log-error held fixed (conservative on that
    # term -- more pixels ought to help it too, but nothing here measures that).
    # Null: nothing scales; resolution buys zero. Both reported so the gap is
    # visible rather than assumed.
    print("=" * 78)
    print("C  RESOLUTION TRANSFER -- EXTRAPOLATION, NOT A MEASUREMENT")
    print("=" * 78)
    print("  Model: centre error is a fixed fraction of a cell, so it shrinks as")
    print("  448/R in relative units while GT boxes keep their relative size.")
    print("  This is the OPTIMISTIC case and it is a gate, not a result.\n")
    print(f"  {'input R':>8} {'shrink':>7} {'mean IoU':>9} {'>=0.5':>8} "
          f"{'>=0.25':>8}   {'null >=0.5':>10}")
    print("  " + "-" * 68)
    for R in [int(x) for x in args.ladder.split(",")]:
        s = R0 / R
        sc = np.stack([gt[:, 0] + (pr[:, 0] - gt[:, 0]) * s,
                       gt[:, 1] + (pr[:, 1] - gt[:, 1]) * s,
                       pr[:, 2], pr[:, 3]], -1)
        v = iou_cwh(sc, gt)
        print(f"  {R:>8} {s:>7.3f} {v.mean():>9.4f} {frac(v,0.5):>7.2f}% "
              f"{frac(v,0.25):>7.2f}%   {frac(base,0.5):>9.2f}%")

    # invert: what shrink factor is needed to hit a target, and what R is that?
    print("\n  INVERTED -- what would it take?")
    print(f"  {'target >=0.5':>13} {'shrink needed':>14} {'implied R':>11}")
    print("  " + "-" * 42)
    for target in (20.0, 30.0, 40.0, 50.0):
        lo, hi, got = 1e-4, 1.0, None
        for _ in range(40):
            mid = (lo + hi) / 2
            sc = np.stack([gt[:, 0] + (pr[:, 0] - gt[:, 0]) * mid,
                           gt[:, 1] + (pr[:, 1] - gt[:, 1]) * mid,
                           pr[:, 2], pr[:, 3]], -1)
            if frac(iou_cwh(sc, gt), 0.5) >= target:
                lo, got = mid, mid
            else:
                hi = mid
        if got is None:
            print(f"  {target:>12.0f}% {'UNREACHABLE':>14} "
                  f"{'--':>11}   (centre error is not the binding term)")
        else:
            print(f"  {target:>12.0f}% {got:>14.3f} {R0/got:>10.0f}px")

    # the ceiling with a PERFECT centre, which no resolution can exceed under
    # this model -- the honest asymptote of the whole lever.
    print(f"\n  ASYMPTOTE (centre error -> 0, measured w,h kept): "
          f"{frac(perfect_ctr,0.5):.2f}% at IoU>=0.5")
    print("  No amount of resolution can beat this while the size regression is")
    print("  as measured -- if it is low, resolution is the WRONG lever.\n")

    # ── D. does the correct box ever REACH the detection list? ────────────────
    # Section A says the assigned cell already emits an IoU>=0.5 box a third of
    # the time, but end-to-end class-agnostic recall is 11.8%. Those cannot both
    # describe the same pipeline unless the correct boxes are being LOST between
    # the head and the scorer. The scorer keeps conf = sigmoid(obj)*max softmax(cls)
    # and truncates to the top --topk per image before NMS, so measure the rank
    # of each correct box among its image's slots.
    good = base >= 0.5
    print("=" * 78)
    print("D  DOES THE CORRECT BOX REACH THE DETECTION LIST? (model-free)")
    print("=" * 78)
    print(f"  candidate slots per image: {nslots}"
          f"   scorer keeps top {args.topk} by conf\n")
    print(f"  {'population':<40} {'n':>7} {'med rank':>9} {'in topk':>9}")
    print("  " + "-" * 68)
    for nm, m in (("all positive slots", np.ones_like(good, dtype=bool)),
                  ("positives whose box is IoU>=0.5", good),
                  ("...and whose argmax class is right", good & clsok)):
        if m.sum() == 0:
            continue
        print(f"  {nm:<40} {m.sum():>7} {np.median(rank[m]):>9.0f} "
              f"{100.0*(rank[m] < args.topk).mean():>8.2f}%")
    print(f"\n  class head correct at the positive slot: "
          f"{100.0*clsok.mean():.2f}%  "
          f"(on IoU>=0.5 slots: {100.0*clsok[good].mean():.2f}%)")

    npos = good.size
    surv = good & (rank < args.topk)
    print("\n  RECALL BUDGET -- of every encoded positive, what survives:")
    print(f"    emits IoU>=0.5 box at its own cell      {100.0*good.mean():>7.2f}%")
    print(f"    ... and reaches the top-{args.topk} list          "
          f"{100.0*surv.mean():>7.2f}%")
    print(f"    ... and also has the right class        "
          f"{100.0*(surv & clsok).mean():>7.2f}%")
    print(f"\n  measured end-to-end class-agnostic recall is 11.80% (bite 0c).")
    print(f"  The gap between {100.0*good.mean():.2f}% and that is NOT box precision.\n")

    # ── E. how much is the top-k truncation itself costing? ───────────────────
    # If correct boxes are being cut for rank rather than for quality, the cut
    # is a SCORER parameter, not a property of the detector. Sweep it.
    print("=" * 78)
    print("E  RECALL vs THE TOP-k CUT (model-free)")
    print("=" * 78)
    print(f"  {'top-k':>8} {'IoU>=0.5 kept':>15} {'+ right class':>15}")
    print("  " + "-" * 42)
    for k in (100, 500, 1000, 2000, 4000, 8000, nslots):
        print(f"  {k:>8} {100.0*(good & (rank < k)).mean():>14.2f}% "
              f"{100.0*(good & clsok & (rank < k)).mean():>14.2f}%")
    print(f"\n  slots passing the scorer's conf>=0.001 filter, per image: "
          f"{(conf >= 0.001).sum() / n:.0f} of {nslots} "
          f"(positives only; the cut is dominated by top-k, not the threshold)")
    print()

    # ── F. WHAT is outranking the correct boxes? ──────────────────────────────
    # The assignment doc concluded the head "CANNOT split the assigned cell from
    # its neighbour" and called the task unlearnable. But ranking into a top-k
    # list is a contest against ALL slots, and there are ~34x more far-background
    # cells than ring cells. Split the slots that outrank each correct box by
    # category: if far background dominates, the binding failure is ordinary
    # object/background discrimination, NOT the unlearnable neighbour split.
    print("=" * 78)
    print("F  WHAT OUTRANKS THE CORRECT BOXES? (model-free)")
    print("=" * 78)
    order = np.argsort(-allc, axis=1)
    ordered_cat = np.take_along_axis(cats, order, axis=1)
    cum = {c: np.cumsum(ordered_cat == c, axis=1) for c in (0, 1, 2)}
    pr_rank = rank[good]; pr_rec = prec[good]
    tot = np.zeros(3)
    for c in (0, 1, 2):
        tot[c] = cum[c][pr_rec, pr_rank].mean()
    print(f"  for the {good.sum()} positives that DO emit an IoU>=0.5 box,")
    print(f"  mean number of higher-scoring slots in the same image: "
          f"{tot.sum():.0f}\n")
    print(f"  {'category':<34} {'mean count':>11} {'share':>8}")
    print("  " + "-" * 56)
    for c, nm in ((0, "far background"), (1, "ring (adjacent to a positive)"),
                  (2, "other positives")):
        print(f"  {nm:<34} {tot[c]:>11.0f} {100.0*tot[c]/max(tot.sum(),1e-9):>7.1f}%")
    print(f"\n  slot census per image: "
          f"far {100.0*(cats==0).mean():.1f}%  ring {100.0*(cats==1).mean():.1f}%  "
          f"pos {100.0*(cats==2).mean():.1f}%")
    print("\n  If far background dominates BOTH the census and the outranking,")
    print("  the ring/neighbour story is not what is costing the recall.\n")

    # ── G. which lever actually moves end-to-end recall? ──────────────────────
    # Two independent gates stand between an encoded positive and a counted
    # detection: its box must clear IoU 0.5 (precision) and its conf must clear
    # the top-k cut (ranking). A lever that fixes one is capped by the other.
    # Price them against each other, on the SAME denominator.
    k = args.topk
    ink = rank < k
    s768 = R0 / 768.0
    box768 = iou_cwh(np.stack([gt[:, 0] + (pr[:, 0] - gt[:, 0]) * s768,
                               gt[:, 1] + (pr[:, 1] - gt[:, 1]) * s768,
                               pr[:, 2], pr[:, 3]], -1), gt) >= 0.5
    print("=" * 78)
    print("G  LEVER COMPARISON -- recall of encoded positives, both gates applied")
    print("=" * 78)
    print(f"  {'scenario':<46} {'box ok':>8} {'in top-k':>9} {'BOTH':>8}")
    print("  " + "-" * 74)
    rows = [("today (measured)", good, ink),
            ("resolution 768, optimistic (section C)", box768, ink),
            ("PERFECT boxes, ranking as measured", np.ones_like(good, dtype=bool), ink),
            ("PERFECT ranking, boxes as measured", good, np.ones_like(ink, dtype=bool)),
            ("both perfect", np.ones_like(good, dtype=bool),
             np.ones_like(ink, dtype=bool))]
    for nm, b, r in rows:
        print(f"  {nm:<46} {100.0*b.mean():>7.2f}% {100.0*r.mean():>8.2f}% "
              f"{100.0*(b & r).mean():>7.2f}%")
    print(f"\n  Read the BOTH column. The lever with the larger headroom is the")
    print(f"  one to build; the other is capped by it no matter how well it works.\n")


if __name__ == "__main__":
    main()
