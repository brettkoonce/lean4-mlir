"""Dump twin logits in the Lean `inferDump` format so the EXISTING scorer works.

`scripts/yolo_map_visdrone.py --fpn` already knows how to decode a [N, NTOT]
float32 dump into 3-scale boxes, merge them and run one NMS. Emitting the same
bytes means the twin's mAP is computed by the same code that produced the Lean
arm's 0.0001 -- so the two numbers are directly comparable and no second scorer
can drift from the first.

    .venv/bin/python3 -m bespoke.infer --ckpt runs_bespoke/r34_full_12ep/last.pt \\
        --out ../figures/twin_r34_12ep
    python3 scripts/yolo_map_visdrone.py figures/twin_r34_12ep/logits.bin \\
        data/visdrone448/val.bin --fpn data/visdrone --grid 14 --box-param diou

BN runs in EVAL mode here (running statistics), matching what Lean's `infer` does.
"""
import argparse
import os

import numpy as np
import torch

from bespoke.data import FpnBinDataset
from bespoke.model import FpnDetector

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(REPO, "data/visdrone_fpn/val.bin"))
    ap.add_argument("--out", required=True, help="directory to write logits.bin into")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location="cpu")
    targs = blob.get("args", {})
    model = FpnDetector(backbone=targs.get("backbone", "r34"),
                        oc=targs.get("oc", 256), tower=targs.get("tower", 0),
                        norm=targs.get("norm"), pretrained=False,
                        pool=targs.get("pool", "torchvision")).to(dev)
    model.load_state_dict(blob["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch {blob.get('epoch')}) backbone="
          f"{targs.get('backbone')} norm={targs.get('norm')} tower={targs.get('tower')}",
          flush=True)

    ds = FpnBinDataset(args.data, limit=args.limit)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "logits.bin")
    n = 0
    with torch.no_grad(), open(path, "wb") as f:
        for i0 in range(0, len(ds), args.batch):
            idx = range(i0, min(i0 + args.batch, len(ds)))
            imgs = torch.stack([ds[i][0] for i in idx]).to(dev)
            out = model(imgs).float().cpu().numpy().astype(np.float32)
            f.write(out.tobytes())
            n += out.shape[0]
            if i0 % (args.batch * 20) == 0:
                print(f"  {n}/{len(ds)}", flush=True)
    print(f"wrote {path}: {n} records x {out.shape[1]} floats", flush=True)


if __name__ == "__main__":
    main()
