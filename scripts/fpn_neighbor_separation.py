#!/usr/bin/env python3
"""Is the objectness equilibrium an ASSIGNMENT problem rather than a head problem?

Three parameterizations of the detector head — no bias, bias + RetinaNet prior init,
and bias + prior + a 4-conv-per-level tower (+7M params) — all converge to the SAME
objectness logit distribution (gap ~+0.25, std ~0.24/0.33). Extra capacity did not
even lower the TRAIN loss. So the limit is not init, not capacity, and (per the T1a
refutation) not loss balance.

The remaining suspect is the target assignment. VisDrone objects are 2-5px on a 56x56
P3 grid, so a positive cell and its immediate neighbours see almost the same receptive
field. If the head assigns them almost the same logit, then positives and their
neighbours are not separable by ANY function of those features, and no amount of head
capacity can help — you cannot separate near-identical inputs.

This splits background into two populations and compares them to the positives:

  positive   : target objectness == 1
  ring       : target 0, but 8-adjacent to a positive   <- the contested cells
  far        : target 0 and not adjacent to any positive

Read it as: if mean(ring) sits close to mean(positive) and far from mean(far), the head
is doing the best a local function can and the ASSIGNMENT is what caps ranking (=> go to
Tier 3: multi-anchor-per-GT / FCOS center sampling). If ring sits with far, the head
simply is not separating, and the assignment is exonerated.

Usage:
  python3 scripts/fpn_neighbor_separation.py <logits.bin> <val.bin> [--scales 56,28,14] [-A 3]
"""
import argparse
import sys

import numpy as np

P = 15  # per-anchor channels: 4 box + 1 obj + 10 class


def dilate8(m):
    """8-neighbourhood dilation of a boolean [H,W] map."""
    out = np.zeros_like(m)
    H, W = m.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys0, ys1 = max(0, dy), H + min(0, dy)
            xs0, xs1 = max(0, dx), W + min(0, dx)
            out[ys0:ys1, xs0:xs1] |= m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits")
    ap.add_argument("valbin")
    ap.add_argument("--scales", default="56,28,14")
    ap.add_argument("-A", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=448)
    args = ap.parse_args()

    grids = [int(x) for x in args.scales.split(",")]
    A = args.A
    ntot = sum(A * P * g * g for g in grids)

    logits = np.fromfile(args.logits, dtype=np.float32)
    n = logits.size // ntot
    logits = logits[:n * ntot].reshape(n, ntot)

    # val.bin = a 4-byte <I record-count header, then records of image (u8) +
    # flat target (f32). The header MUST be skipped: starting at byte 0 shifts
    # every target by one float32 = one cell along j (the fastest-varying axis),
    # which silently mislabels each positive's right-hand neighbour as the
    # positive. (Measured: it moves pos-ring from +0.0250 to +0.0217 -- it did
    # not change this script's verdict, but it is still the wrong read.)
    rec_img = 3 * args.imgsz * args.imgsz
    rec = rec_img + ntot * 4
    raw = np.fromfile(args.valbin, dtype=np.uint8)
    nrec = (raw.size - 4) // rec
    raw = raw[4:4 + nrec * rec].reshape(nrec, rec)
    tgt = raw[:, rec_img:].copy().view(np.float32).reshape(nrec, ntot)
    n = min(n, nrec)
    print(f"records: logits {logits.shape[0]}, targets {nrec} -> using {n}; Ntot={ntot}")

    pos, ring, far = [], [], []
    off = 0
    for g in grids:
        blk = A * P * g * g
        L = logits[:n, off:off + blk].reshape(n, A * P, g, g)
        T = tgt[:n, off:off + blk].reshape(n, A * P, g, g)
        for a in range(A):
            ob = L[:, a * P + 4]          # [n,g,g] objectness logit
            tb = T[:, a * P + 4] > 0.5    # [n,g,g] positive mask
            for i in range(n):
                m = tb[i]
                if not m.any():
                    far.append(ob[i].ravel())
                    continue
                d = dilate8(m)
                pos.append(ob[i][m])
                ring.append(ob[i][d & ~m])
                far.append(ob[i][~d])
        off += blk

    pos = np.concatenate(pos) if pos else np.array([])
    ring = np.concatenate(ring) if ring else np.array([])
    far = np.concatenate(far)

    def line(name, v):
        if v.size == 0:
            print(f"  {name:9s} n=0")
            return
        print(f"  {name:9s} n={v.size:>9d}  mean={v.mean():+.4f}  std={v.std():.4f}")

    print("\nobjectness logit by target population:")
    line("positive", pos)
    line("ring(adj)", ring)
    line("far bg", far)

    if pos.size and ring.size and far.size:
        gap_pf = pos.mean() - far.mean()
        gap_pr = pos.mean() - ring.mean()
        gap_rf = ring.mean() - far.mean()
        print(f"\n  pos - far  = {gap_pf:+.4f}   (the separation that ranking needs)")
        print(f"  pos - ring = {gap_pr:+.4f}   (can the head split a cell from its neighbour?)")
        print(f"  ring - far = {gap_rf:+.4f}   (does the head know the ring is NEAR an object?)")
        if gap_pf != 0:
            frac = gap_rf / gap_pf
            print(f"\n  ring sits {frac * 100:.0f}% of the way from far background to positive.")
            if frac > 0.5:
                print("  => The head DOES localize objects; it just cannot separate the assigned")
                print("     cell from its neighbours. Positives and ring are near-identical inputs,")
                print("     so no head capacity can rank them apart. ASSIGNMENT is the cap.")
                print("     => Tier 3: multi-anchor-per-GT / FCOS center sampling.")
            else:
                print("  => The ring behaves like far background: the head is not localizing at all,")
                print("     so the assignment is exonerated and the features/objective are suspect.")


if __name__ == "__main__":
    main()
