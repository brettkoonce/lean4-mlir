"""§6 validation: prove the twin reproduces Lean's numbers BEFORE trusting a diff.

Three checks, in the order planning/jax_gradient_oracle.md §6 specifies:
  1. loss on Lean's OWN logits dump must hit the golden breakdown
  2. the twin's forward at the same weights vs that dump
  3. the gradient AT THE LOGITS vs alpha*w_foc*(p-t)/B

Check 1 uses no model at all, so it isolates the loss from the forward.

Usage (from visdrone/):
    .venv/bin/python3 -m bespoke.validate_oracle \
        --ckpt ../.lake/build/..._of8long_params_e2000.bin \
        --dump ../figures/yolo_of8long/logits.bin \
        --data ../data/visdrone_fpn_of8/train.bin
"""
import argparse
import struct

import numpy as np
import torch

from bespoke.data import FpnBinDataset
from bespoke.bn_stats import load_bn_stats
from bespoke.lean_ckpt import load_lean_params
from bespoke.loss import CLS_WEIGHTS, fpn_loss
from bespoke.model import FPN_SCALES, NTOT, FpnDetector


def load_targets(path, n):
    ds = FpnBinDataset(path)
    n = min(n, len(ds))
    return torch.stack([ds[i][1] for i in range(n)]), ds


def breakdown_of(logits, tgts, cls_weights, gamma=2.0):
    bd = dict.fromkeys(["box", "obj", "cls", "obj_pos", "obj_neg", "npos"], 0.0)
    total = fpn_loss(logits, tgts, FPN_SCALES, gamma, cls_weights, breakdown=bd)
    B = logits.shape[0]
    return float(total), {k: v / B for k, v in bd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pad", default="lean", choices=["lean", "torchvision"],
                    help="stride-2 conv padding convention; 'lean' is the "
                         "faithful one (MlirCodegen.samePad is asymmetric)")
    ap.add_argument("--bn-stats", default=None,
                    help="Lean *_bn_stats.bin; without it, eval mode runs on "
                         "torchvision's untrained (0,1) buffers and means nothing")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dump = np.fromfile(args.dump, dtype=np.float32)
    assert dump.size % NTOT == 0, f"dump {dump.size} not a multiple of {NTOT}"
    ndump = dump.size // NTOT
    lean_logits = torch.from_numpy(dump.reshape(ndump, NTOT)).to(dev)
    n = min(args.batch, ndump)
    tgts, _ = load_targets(args.data, n)
    tgts = tgts.to(dev)
    lean_logits = lean_logits[:n]
    print(f"dump: {ndump} records, using {n}\n")

    # ---- check 1: the loss, on Lean's own logits. No model involved. ----
    print("=== 1. loss on Lean's own logits dump ===")
    for label, cw in (("unweighted (matches fpn_loss_breakdown.py)", None),
                      ("wcls (the arm's actual loss)", CLS_WEIGHTS)):
        tot, bd = breakdown_of(lean_logits, tgts, cw)
        print(f"  {label}:")
        print(f"    total {tot:9.4f}  box {bd['box']:8.4f}  obj {bd['obj']:8.4f} "
              f"(pos {bd['obj_pos']:7.4f} / neg {bd['obj_neg']:7.4f})  "
              f"cls {bd['cls']:8.4f}")
    print("  golden (doc §6.1): total 35.422 / box 6.155 / obj 27.550 "
          "(pos 14.761 / neg 12.789) / cls 1.717\n")

    # ---- check 2: the twin's forward at the same weights ----
    print("=== 2. twin forward vs the dump ===")
    model = FpnDetector(backbone="r34", oc=256, tower=0, norm=None,
                        pretrained=False, pool="lean", pad=args.pad).to(dev)
    load_lean_params(model, args.ckpt, verbose=False)
    ds = FpnBinDataset(args.data)
    imgs = torch.stack([ds[i][0] for i in range(n)]).to(dev)

    modes = ["train", "eval"]
    if args.bn_stats:
        load_bn_stats(model, args.bn_stats)
        modes = ["train", "eval(lean bn)"]
    for mode in modes:
        model.train() if mode == "train" else model.eval()
        with torch.no_grad():
            tw = model(imgs)
        d = (tw - lean_logits).abs()
        rel = d.mean() / lean_logits.abs().mean()
        tot, bd = breakdown_of(tw, tgts, CLS_WEIGHTS)
        totu, _ = breakdown_of(tw, tgts, None)
        print(f"  BN {mode:13s}: twin mean {float(tw.mean()):8.4f} "
              f"std {float(tw.std()):7.4f} | lean mean "
              f"{float(lean_logits.mean()):8.4f} std {float(lean_logits.std()):7.4f}")
        print(f"                    mean|diff| {float(d.mean()):.4f}  max|diff| "
              f"{float(d.max()):.4f}  rel {float(rel):.4f}")
        print(f"                    loss wcls {tot:9.4f}  unweighted {totu:9.4f} "
              f"[box {bd['box']:.3f} obj {bd['obj']:.3f} cls {bd['cls']:.3f}]")
        # A permutation shows up as matching SORTED values with mismatched
        # positions -- exactly what mean/std alone cannot distinguish.
        sa, _ = torch.sort(tw.flatten())
        sb, _ = torch.sort(lean_logits.flatten())
        sd = (sa - sb).abs()
        print(f"                    sorted-value mean|diff| {float(sd.mean()):.4f} "
              f"max {float(sd.max()):.4f}  <- tiny here + large above = PERMUTATION")
        # Per-scale, to see whether one pyramid level dominates the disagreement.
        off = 0
        parts = []
        for gsz, anchors in FPN_SCALES:
            cnt = len(anchors) * 15 * gsz * gsz
            parts.append(f"P{ {56:3, 28:4, 14:5}[gsz] }:"
                         f"{float(d[:, off:off+cnt].mean()):.3f}")
            off += cnt
        print(f"                    per-scale mean|diff|  {'  '.join(parts)}")
    print("  Lean's OWN train-mode loss at e2000 (runs/fpn_of8_long.log): ~35.97\n")

    # ---- check 3: the gradient at the logits ----
    print("=== 3. d(loss)/d(logits) vs alpha*w_foc*(p-t)/B on the obj channels ===")
    z = lean_logits.clone().requires_grad_(True)
    loss = fpn_loss(z, tgts, FPN_SCALES, 2.0, CLS_WEIGHTS)
    (g,) = torch.autograd.grad(loss, z)

    B, off, worst = n, 0, 0.0
    for gsz, anchors in FPN_SCALES:
        A, P = len(anchors), 15
        cnt = A * P * gsz * gsz
        gs = g[:, off:off + cnt].view(B, A * P, gsz, gsz)
        ps = lean_logits[:, off:off + cnt].view(B, A * P, gsz, gsz)
        ts = tgts[:, off:off + cnt].view(B, A * P, gsz, gsz)
        for a in range(A):
            b = a * P
            zc, tc = ps[:, b + 4], ts[:, b + 4]
            prob = torch.sigmoid(zc)
            pt = tc * prob + (1 - tc) * (1 - prob)
            wf = (1 - pt).clamp(min=1e-12) ** 2.0
            alpha = tc + (1 - tc) * 0.5
            want = alpha * wf * (prob - tc) / B
            worst = max(worst, float((gs[:, b + 4] - want).abs().max()))
        off += cnt
    print(f"  max|autograd - emitted formula| over all objectness logits: {worst:.3e}")
    print("  (doc §6.2 wants ~1e-16; a large value means the focal weight is not "
          "detached)\n")


if __name__ == "__main__":
    main()
