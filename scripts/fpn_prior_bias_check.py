#!/usr/bin/env python3
"""Verify NetSpec.applyDetPriorBias wrote the RetinaNet prior into the right slots.

The FPN detector head's 3 biases are the last 3*A*15 floats of the param buffer
(the `.fpnDetect` layer is last, and its biases come after its 6 weights). Within
each [A*15] block, anchor `a`'s objectness is channel a*15+4 -- box occupies
a*15 .. a*15+4 and the 10 classes a*15+5 .. a*15+15, per emitAnchorYoloLoss.

So a correct install is: obj channels == -log((1-pi)/pi), every other channel 0.
This is the check that the prior landed on OBJECTNESS and not, say, on a box or
class channel -- an off-by-one here is silent (training still runs, the lever just
does something else) which is exactly why it gets its own script.

Run:  python3 scripts/fpn_prior_bias_check.py <init_dump.bin> [--pi 0.01] [-A 3]
"""
import argparse
import math
import sys

import numpy as np

P = 15  # per-anchor channels: 4 box + 1 obj + 10 class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="LEAN_MLIR_INIT_DUMP output")
    ap.add_argument("--pi", type=float, default=0.01)
    ap.add_argument("-A", type=int, default=3, help="anchors per scale")
    args = ap.parse_args()

    n = args.A * P
    params = np.fromfile(args.dump, dtype=np.float32)
    tail = params[-3 * n:]
    want = -math.log((1.0 - args.pi) / args.pi)

    print(f"params: {params.size}  head-bias tail: {3 * n} floats "
          f"(3 scales x A={args.A} x {P})")
    print(f"expected objectness bias: -log((1-{args.pi})/{args.pi}) = {want:.6f}")

    ok = True
    for s in range(3):
        blk = tail[s * n:(s + 1) * n]
        obj = blk[4::P]
        other = np.delete(blk, np.arange(4, n, P))
        obj_err = float(np.max(np.abs(obj - want)))
        other_max = float(np.max(np.abs(other))) if other.size else 0.0
        good = obj_err < 1e-5 and other_max == 0.0
        ok &= good
        print(f"  P{s + 3}: obj[{len(obj)}] max|b-want|={obj_err:.2e}  "
              f"non-obj[{other.size}] max|b|={other_max:.2e}  "
              f"{'PASS' if good else 'FAIL'}")

    # Guard against the buffer being all-prior (i.e. we sliced the wrong tail):
    # the 6 head/neck WEIGHTS immediately before the tail must still be He-init,
    # so they must not be constant.
    prev = params[-3 * n - 512:-3 * n]
    if prev.std() < 1e-8:
        print("  WARN: the 512 floats before the bias tail are constant — "
              "the tail slice may not be where the biases actually live")
        ok = False
    else:
        print(f"  preceding weights std={prev.std():.4f} (He-init, not clobbered)")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
