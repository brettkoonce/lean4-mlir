"""Train the PyTorch twin of the Lean FPN detector.

Defaults reproduce `r34FpnDetConfig` (demos/MainYolov1VisdroneFpn.lean:78):
    lr 4e-4, batch 8, 12 epochs, Adam(W), wd 5e-4, cosine, 3 warmup epochs,
    grad-clip 4.0, focal gamma 2.0, no augmentation, RetinaNet prior pi=0.01.

THE POINT OF THIS SCRIPT: the Lean arm reaches mAP 0.0001 while a YOLOv8s on the
same data reaches 0.114-0.391. That comparison is confounded -- YOLOv8 is a
different architecture. This trains OUR architecture, on OUR encoded bytes, with
OUR loss and OUR recipe, in PyTorch. It separates "the architecture cannot work"
from "the Lean implementation is broken", which nothing measured so far does.

  python3 -m bespoke.train --backbone r34 --epochs 12          # the faithful control
  python3 -m bespoke.train --subset 8 --epochs 500 --tag of8   # the overfit gate
  python3 -m bespoke.train --backbone r50 --norm gn --tower 4  # the experiment

Run from visdrone/ with .venv/bin/python3.
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from bespoke.data import FpnBinDataset
from bespoke.loss import CLS_WEIGHTS, fpn_loss
from bespoke.model import FPN_SCALES, FpnDetector

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def lr_at(step, total_steps, warmup_steps, base_lr):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data/visdrone_fpn/train.bin"))
    ap.add_argument("--backbone", default="r34", choices=["r34", "r50"])
    ap.add_argument("--weights", default=None,
                    help="path to a backbone state_dict (e.g. an RSB-A2 R50) to load "
                         "over the torchvision init")
    ap.add_argument("--norm", default=None, choices=[None, "bn", "gn"],
                    help="normalization in neck+head. None = the faithful twin "
                         "(the Lean detector has NO norm anywhere in neck/head).")
    ap.add_argument("--tower", type=int, default=0)
    ap.add_argument("--pool", default="torchvision", choices=["torchvision", "lean"],
                    help="stem maxpool. Lean's spec is `.maxPool 2 2`; torchvision's "
                         "ResNet stem is k=3,s=2,p=1. Use 'lean' for a byte-exact twin.")
    ap.add_argument("--oc", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=4.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--subset", type=int, default=None,
                    help="train on the first N records only (overfit probe)")
    ap.add_argument("--no-cls-weights", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="twin")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "runs_bespoke"))
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = os.path.abspath(os.path.join(args.out, args.tag))
    os.makedirs(outdir, exist_ok=True)
    print(f"device={dev}  out={outdir}", flush=True)

    ds = FpnBinDataset(args.data)
    if args.subset:
        ds = Subset(ds, list(range(min(args.subset, len(ds)))))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=False, pin_memory=True,
                    persistent_workers=args.workers > 0)

    model = FpnDetector(backbone=args.backbone, oc=args.oc, tower=args.tower,
                        norm=args.norm, pretrained=not args.no_pretrained,
                        pool=args.pool).to(dev)
    if args.weights:
        sd = torch.load(args.weights, map_location="cpu")
        sd = sd.get("state_dict", sd.get("model", sd))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.weights}: {len(missing)} missing, "
              f"{len(unexpected)} unexpected", flush=True)

    nparam = sum(p.numel() for p in model.parameters())
    print(f"spec: {args.backbone} oc={args.oc} tower={args.tower} norm={args.norm} "
          f"| {nparam:,} params | {len(ds)} imgs | {args.epochs} ep", flush=True)

    opt = torch.optim.AdamW(model.param_groups(args.wd), lr=args.lr)
    cw = None if args.no_cls_weights else CLS_WEIGHTS
    steps_per_epoch = math.ceil(len(ds) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    history, step = [], 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, ep_loss, nb = time.time(), 0.0, 0
        bd = dict.fromkeys(["box", "obj", "cls", "obj_pos", "obj_neg", "npos"], 0.0)
        for img, tgt in dl:
            lr = lr_at(step, total_steps, warmup_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            img, tgt = img.to(dev, non_blocking=True), tgt.to(dev, non_blocking=True)
            logits = model(img)
            loss = fpn_loss(logits, tgt, FPN_SCALES, args.gamma, cw, breakdown=bd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            ep_loss += float(loss)
            nb += 1
            step += 1
            if args.log_every and step % args.log_every == 0:
                print(f"  step {step}/{total_steps} loss={float(loss):.3f} "
                      f"lr={lr:.2e}", flush=True)

        avg = ep_loss / max(1, nb)
        nimg = max(1, nb * args.batch)      # breakdown holds raw sums; -> per image
        row = {"epoch": ep, "loss": avg, "lr": lr, "secs": time.time() - t0,
               "box": bd["box"] / nimg, "obj": bd["obj"] / nimg, "cls": bd["cls"] / nimg,
               "obj_pos": bd["obj_pos"] / nimg, "obj_neg": bd["obj_neg"] / nimg,
               "npos_per_img": bd["npos"] / nimg}
        history.append(row)
        print(f"Epoch {ep}/{args.epochs}: loss={avg:.4f} "
              f"[box {row['box']:.2f} obj {row['obj']:.2f} "
              f"(pos {row['obj_pos']:.2f} / neg {row['obj_neg']:.2f}) "
              f"cls {row['cls']:.2f}] npos/img={row['npos_per_img']:.1f} "
              f"lr={lr:.2e} ({row['secs']:.0f}s)", flush=True)
        with open(os.path.join(outdir, "history.json"), "w") as f:
            json.dump({"args": vars(args), "history": history}, f, indent=2)
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep},
                   os.path.join(outdir, "last.pt"))

    print("done.", flush=True)


if __name__ == "__main__":
    main()
