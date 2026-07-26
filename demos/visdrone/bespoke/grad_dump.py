"""Dump the twin's parameter gradients at a Lean checkpoint, in Lean's flat order.

This is the reference side of the sign probe (planning/jax_gradient_oracle.md §8a).
Adam's first step from m=v=0 is w -= lr*sign(g)*|g|/(|g|+eps), so a Lean run
resumed from W0 for exactly ONE step encodes sign(g_lean) in sign(W0-W1) --
readable with no codegen change. This script produces sign(g_reference).

Both the clip factor (a positive global scalar) and decoupled weight decay
(-wd*lr*w, ~5e-6 relative) preserve that sign for any |g| well above Adam's
eps=1e-8; `sign_probe.py` masks the rest and reports the masked fraction.

Usage (from visdrone/):
    .venv/bin/python3 -m bespoke.grad_dump \
        --ckpt ../.lake/build/..._of8long_params_e2000.bin \
        --data ../data/visdrone_fpn_of8/train.bin \
        --out /tmp/grad_w0.npz
"""
import argparse
import json
import os

import numpy as np
import torch

from bespoke.data import FpnBinDataset
from bespoke.lean_ckpt import load_lean_params, torch_backbone_params
from bespoke.loss import CLS_WEIGHTS, fpn_loss
from bespoke.model import FPN_SCALES, FpnDetector


def lean_ordered_named_params(model):
    """(name, param) in NetSpec.paramShapes order -- the flat checkpoint order.

    Must stay in lockstep with lean_ckpt.load_lean_params's `targets` list; the
    shape assertions there are what proves this ordering is the right one.
    """
    out = []
    for mod_name, mod in (("stem", model.stem), ("layer2", model.layer2),
                          ("layer3", model.layer3), ("layer4", model.layer4)):
        out.extend((f"{mod_name}.{n}", p) for n, p in mod.named_parameters())
    out += [("lat3.weight", model.lat3.weight),
            ("lat4.weight", model.lat4.weight),
            ("lat5.weight", model.lat5.weight),
            ("head3.weight", model.heads[0].weight),
            ("head4.weight", model.heads[1].weight),
            ("head5.weight", model.heads[2].weight),
            ("head3.bias", model.heads[0].bias),
            ("head4.bias", model.heads[1].bias),
            ("head5.bias", model.heads[2].bias)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pad", default="lean", choices=["lean", "torchvision"],
                    help="stride-2 conv padding convention; 'lean' is the "
                         "faithful one (MlirCodegen.samePad is asymmetric)")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--no-cls-weights", action="store_true")
    ap.add_argument("--eval-mode", action="store_true",
                    help="run BN in eval mode (needs --bn-stats); default is "
                         "train mode, which is what the Lean trainer uses")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FpnDetector(backbone="r34", oc=256, tower=0, norm=None,
                        pretrained=False, pool="lean", pad=args.pad).to(dev)
    load_lean_params(model, args.ckpt)
    model = model.to(dev)
    model.train() if not args.eval_mode else model.eval()

    ds = FpnBinDataset(args.data)
    n = min(len(ds), args.batch)
    imgs = torch.stack([ds[i][0] for i in range(n)]).to(dev)
    tgts = torch.stack([ds[i][1] for i in range(n)]).to(dev)
    print(f"batch: {n} images from {args.data}", flush=True)

    bd = dict.fromkeys(["box", "obj", "cls", "obj_pos", "obj_neg", "npos"], 0.0)
    logits = model(imgs)
    loss = fpn_loss(logits, tgts, FPN_SCALES, args.gamma,
                    None if args.no_cls_weights else CLS_WEIGHTS, breakdown=bd)
    model.zero_grad(set_to_none=True)
    loss.backward()

    B = n
    print(f"loss  : {float(loss):.4f}   [box {bd['box']/B:.3f} "
          f"obj {bd['obj']/B:.3f} (pos {bd['obj_pos']/B:.3f} / "
          f"neg {bd['obj_neg']/B:.3f}) cls {bd['cls']/B:.3f}]", flush=True)
    print(f"npos  : {bd['npos']:.0f} total, {bd['npos']/B:.1f}/img", flush=True)
    print(f"logits: mean {float(logits.mean()):.4f} std {float(logits.std()):.4f}",
          flush=True)

    named = lean_ordered_named_params(model)
    flat, blocks, off = [], [], 0
    nmissing = 0
    for name, p in named:
        g = p.grad
        if g is None:
            nmissing += 1
            g = torch.zeros_like(p)
        arr = g.detach().float().cpu().numpy().ravel()
        flat.append(arr)
        blocks.append({"name": name, "offset": off, "count": int(arr.size),
                       "shape": list(p.shape)})
        off += arr.size
    grads = np.concatenate(flat).astype(np.float32)
    print(f"grads : {grads.size:,} floats over {len(blocks)} blocks "
          f"({nmissing} with no grad)", flush=True)

    ckpt_floats = np.fromfile(args.ckpt, dtype=np.float32).size
    if ckpt_floats != grads.size:
        raise SystemExit(f"FATAL: checkpoint has {ckpt_floats:,} floats but the "
                         f"gradient vector has {grads.size:,}; ordering is wrong")

    np.savez(args.out, grads=grads,
             blocks=np.array(json.dumps(blocks)),
             loss=np.float32(float(loss)),
             breakdown=np.array(json.dumps({k: v / B for k, v in bd.items()})))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
