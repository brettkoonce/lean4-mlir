"""Find the index permutation between the twin's head output and Lean's dump.

validate_oracle §2 shows the two agree as a MULTISET (sorted mean|diff| 0.047)
but not positionally (0.79). That is a layout bug, and mean/std/min -- the only
things the original forward check compared -- are permutation-invariant, so it
survived that check.

Tests candidate readings of the per-scale block against Lean's flat order, which
the codegen documents as ((a*15 + c)*g + i)*g + j.
"""
import argparse
import itertools

import numpy as np
import torch

from bespoke.bn_stats import load_bn_stats
from bespoke.data import FpnBinDataset
from bespoke.lean_ckpt import load_lean_params
from bespoke.model import FPN_SCALES, NTOT, FpnDetector

P = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--bn-stats", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pad", default="lean", choices=["lean", "torchvision"],
                    help="stride-2 conv padding convention; 'lean' is the "
                         "faithful one (MlirCodegen.samePad is asymmetric)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dump = np.fromfile(args.dump, dtype=np.float32)
    ndump = dump.size // NTOT
    n = min(args.batch, ndump)
    lean = torch.from_numpy(dump.reshape(ndump, NTOT)[:n].copy()).to(dev)

    model = FpnDetector(backbone="r34", oc=256, tower=0, norm=None,
                        pretrained=False, pool="lean", pad=args.pad).to(dev)
    load_lean_params(model, args.ckpt, verbose=False)
    if args.bn_stats:
        load_bn_stats(model, args.bn_stats, verbose=False)
    model.train()          # batch stats: the closest match to Lean's train fwd
    ds = FpnBinDataset(args.data)
    imgs = torch.stack([ds[i][0] for i in range(n)]).to(dev)
    with torch.no_grad():
        tw = model(imgs)

    scale = float(lean.abs().mean())
    print(f"reference scale: mean|lean| = {scale:.4f}\n")

    off = 0
    for gsz, anchors in FPN_SCALES:
        A = len(anchors)
        cnt = A * P * gsz * gsz
        L = lean[:, off:off + cnt]
        T = tw[:, off:off + cnt]
        lvl = {56: "P3", 28: "P4", 14: "P5"}[gsz]
        print(f"--- {lvl}: A={A} P={P} g={gsz}  ({cnt} floats/img) ---")

        # Lean's documented reading of its own flat block.
        Lg = L.view(n, A * P, gsz, gsz)

        cands = {}
        # 1. identity: torch [A*P, g, g] C-order  == Lean's documented order
        cands["identity  [a,c,i,j]"] = T.view(n, A * P, gsz, gsz)
        # 2. spatial transpose: [.., j, i]
        cands["spatial^T [a,c,j,i]"] = T.view(n, A * P, gsz, gsz).transpose(-1, -2)
        # 3. slot-major channels: Lean packs [c, a] not [a, c]
        cands["slotmajor [c,a,i,j]"] = (T.view(n, A, P, gsz, gsz)
                                        .permute(0, 2, 1, 3, 4)
                                        .reshape(n, A * P, gsz, gsz))
        # 4. channels-last (NHWC): flat is [i, j, a*P]
        cands["NHWC      [i,j,ac]"] = (T.view(n, A * P, gsz, gsz)
                                        .permute(0, 2, 3, 1)
                                        .reshape(n, -1)
                                        .view(n, A * P, gsz, gsz))
        # 5. slot-major AND spatial transpose
        cands["slotmaj+^T"] = (T.view(n, A, P, gsz, gsz)
                                .permute(0, 2, 1, 4, 3)
                                .reshape(n, A * P, gsz, gsz))

        for name, cand in cands.items():
            d = (cand - Lg).abs()
            print(f"  {name:22s} mean|diff| {float(d.mean()):8.4f}  "
                  f"rel {float(d.mean())/scale:7.4f}")

        # Per-channel profile: if the CHANNEL axis is permuted, each twin
        # channel's spatial mean will match some OTHER lean channel's.
        tm = T.view(n, A * P, gsz, gsz).mean(dim=(0, 2, 3))
        lm = Lg.mean(dim=(0, 2, 3))
        # nearest-lean-channel for each twin channel
        match = (tm[:, None] - lm[None, :]).abs().argmin(dim=1)
        ident = int((match == torch.arange(A * P, device=dev)).sum())
        print(f"  channel-profile: {ident}/{A*P} twin channels are nearest to "
              f"their OWN lean channel")
        if ident < A * P:
            print(f"    twin ch -> nearest lean ch: {match.tolist()}")
        off += cnt
        print()


if __name__ == "__main__":
    main()
