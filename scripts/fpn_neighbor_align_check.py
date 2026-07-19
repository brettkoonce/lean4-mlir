#!/usr/bin/env python3
"""Does the val.bin 4-byte header change the neighbour-separation verdict?

scripts/fpn_neighbor_separation.py reads the FPN val.bin starting at byte 0, but
process_split_fpn() writes a 4-byte <I record-count header before the records
(verify: (filesize-4) % rec_bytes == 0, and the u32 at byte 0 is the record
count). Starting at 0 therefore shifts the whole target array LATER by exactly
one float32 -- i.e. one cell along j, the fastest-varying axis.

That shift is not neutral for this particular measurement: it marks the cell to
the RIGHT of each true positive as "positive", which pushes the TRUE positive
into the 8-adjacent "ring". A near-zero pos-ring gap is then close to guaranteed
by construction, independently of what the head learned.

This runs the identical analysis at both offsets so the two can be compared
directly. The logits file has no header (size == n*Ntot*4), so it is unaffected.

Usage: fpn_neighbor_align_check.py <logits.bin> <val.bin> [--scales 56,28,14] [-A 3]
"""
import argparse
import os

import numpy as np

P = 15


def dilate8(m):
    out = np.zeros_like(m)
    H, W = m.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ys0, ys1 = max(0, dy), H + min(0, dy)
            xs0, xs1 = max(0, dx), W + min(0, dx)
            out[ys0:ys1, xs0:xs1] |= m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def analyze(logits, tgt, grids, A, n):
    pos, ring, far = [], [], []
    off = 0
    for g in grids:
        blk = A * P * g * g
        L = logits[:n, off:off + blk].reshape(n, A * P, g, g)
        T = tgt[:n, off:off + blk].reshape(n, A * P, g, g)
        for a in range(A):
            ob = L[:, a * P + 4]
            tb = T[:, a * P + 4] > 0.5
            for i in range(n):
                m = tb[i]
                if not m.any():
                    far.append(ob[i].ravel()); continue
                d = dilate8(m)
                pos.append(ob[i][m]); ring.append(ob[i][d & ~m]); far.append(ob[i][~d])
        off += blk
    cat = lambda v: np.concatenate(v) if v else np.array([])
    return cat(pos), cat(ring), cat(far)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logits"); ap.add_argument("valbin")
    ap.add_argument("--scales", default="56,28,14")
    ap.add_argument("-A", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=448)
    args = ap.parse_args()

    grids = [int(x) for x in args.scales.split(",")]
    A = args.A
    ntot = sum(A * P * g * g for g in grids)
    rec_img = 3 * args.imgsz * args.imgsz
    rec = rec_img + ntot * 4

    sz = os.path.getsize(args.valbin)
    hdr = int(np.fromfile(args.valbin, dtype="<u4", count=1)[0])
    print(f"{args.valbin}: size={sz}  rec={rec}")
    print(f"  size % rec     = {sz % rec}")
    print(f"  (size-4) % rec = {(sz - 4) % rec}")
    print(f"  u32 at byte 0  = {hdr}   (record count in the header)")
    print(f"  => {(sz - 4) // rec} whole records after a 4-byte header\n")

    logits = np.fromfile(args.logits, dtype=np.float32)
    nl = logits.size // ntot
    print(f"{args.logits}: {logits.size} f32 = {logits.size/ntot:.4f} x Ntot "
          f"=> {nl} rows, no header\n")
    logits = logits[:nl * ntot].reshape(nl, ntot)

    raw = np.fromfile(args.valbin, dtype=np.uint8)
    for label, off0 in (("A) off=0  header NOT skipped  (what the script does)", 0),
                        ("B) off=4  header skipped      (correct)", 4)):
        nrec = (raw.size - off0) // rec
        blk = raw[off0:off0 + nrec * rec].reshape(nrec, rec)
        tgt = blk[:, rec_img:].copy().view(np.float32).reshape(nrec, ntot)
        n = min(nl, nrec)
        p, r, f = analyze(logits, tgt, grids, A, n)
        print(f"{label}   [n={n}]")
        for nm, v in (("positive", p), ("ring(adj)", r), ("far bg", f)):
            print(f"    {nm:9s} n={v.size:>9d}  mean={v.mean():+.4f}  std={v.std():.4f}")
        gap_pf, gap_pr, gap_rf = p.mean() - f.mean(), p.mean() - r.mean(), r.mean() - f.mean()
        print(f"    pos-far  = {gap_pf:+.4f}")
        print(f"    pos-ring = {gap_pr:+.4f}    <-- the doc's load-bearing number")
        print(f"    ring-far = {gap_rf:+.4f}")
        print(f"    ring sits {100*gap_rf/gap_pf:.0f}% of the way far->pos\n")


if __name__ == "__main__":
    main()
