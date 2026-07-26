"""Does the Lean forward agree with the twin on identical weights?

Loads a Lean checkpoint into the twin, runs the SAME records the Lean trainer
ran, and reports the loss the twin computes. The Lean side's own number is
known from the run log, so this is a direct comparison.

  Lean of8long @ step 2000 (8-image overfit, train mode, batch stats):
      total 36.0   box 6.16   obj 27.55 (pos 14.8 / neg 12.8)   cls 1.72

If the twin reproduces ~36 on those weights, the emitted FORWARD is faithful and
the defect is in the backward or the update path. If the twin reports something
far lower, the forward that produced those weights is not computing what the
architecture says -- and every FD probe would still have passed, because they
check the emitted gradient against the SAME emitted forward.

    .venv/bin/python3 -m bespoke.diff_lean \\
        --ckpt ../.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone__of8long_params_e2000.bin
"""
import argparse
import os

import torch

from bespoke.data import FpnBinDataset
from bespoke.lean_ckpt import load_lean_params
from bespoke.loss import CLS_WEIGHTS, fpn_loss
from bespoke.model import FPN_SCALES, FpnDetector

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(REPO, "data/visdrone_fpn/train.bin"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--eval-mode", action="store_true",
                    help="BN in eval mode (needs running stats; the Lean training "
                         "loss uses BATCH stats, so leave this off to compare)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FpnDetector(backbone="r34", tower=0, norm=None, pretrained=False,
                        pool="lean").to(dev)
    load_lean_params(model, args.ckpt)
    model.train(not args.eval_mode)

    ds = FpnBinDataset(args.data)
    imgs = torch.stack([ds[i][0] for i in range(args.n)]).to(dev)
    tgts = torch.stack([ds[i][1] for i in range(args.n)]).to(dev)

    bd = dict.fromkeys(["box", "obj", "cls", "obj_pos", "obj_neg", "npos"], 0.0)
    with torch.no_grad():
        logits = model(imgs)
        loss = fpn_loss(logits, tgts, FPN_SCALES, args.gamma, CLS_WEIGHTS, breakdown=bd)

    obj_logits = torch.cat([
        logits[:, o:o + len(a) * 15 * g * g].view(args.n, len(a) * 15, g, g)[:, 4::15]
        .reshape(args.n, -1)
        for o, (g, a) in zip(
            [0, 3 * 15 * 56 * 56, 3 * 15 * 56 * 56 + 3 * 15 * 28 * 28], FPN_SCALES)
    ], dim=1)

    print(f"\n{args.n} records, BN {'eval' if args.eval_mode else 'train (batch stats)'}")
    print(f"  twin total     {float(loss):10.4f}")
    print(f"  twin box       {bd['box']:10.4f}")
    print(f"  twin obj       {bd['obj']:10.4f}  (pos {bd['obj_pos']:.3f} / "
          f"neg {bd['obj_neg']:.3f})")
    print(f"  twin cls       {bd['cls']:10.4f}")
    print(f"  npos/img       {bd['npos'] / args.n:10.1f}")
    print(f"\n  Lean of8long @2000:  total 36.0  box 6.16  obj 27.55 "
          f"(pos 14.8 / neg 12.8)  cls 1.72")
    print(f"\n  objectness logits: mean {obj_logits.mean():.4f} "
          f"std {obj_logits.std():.4f} min {obj_logits.min():.4f} "
          f"max {obj_logits.max():.4f}")
    print("  (Lean measured mean -1.80, std ~0.30 on the production arm)")


if __name__ == "__main__":
    main()
