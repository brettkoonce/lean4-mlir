#!/usr/bin/env python3
"""Synthesize a PERFECT prediction from the encoded targets, to measure the loss FLOOR.

The overfit probe converged at train loss 215 on 32 images. That reads as "the
trainer cannot descend" ONLY if the achievable loss is ~0 -- which nobody has
checked. If a perfect prediction also scores ~215, the model has already
converged and the defect is a mis-specified loss, not a broken optimizer, and
every "converged equilibrium" reading in planning/yolo_assignment.md was
measuring the floor rather than the model.

Emits a logits.bin that `scripts/fpn_loss_breakdown.py` can consume unchanged --
reusing the already-validated numpy replica of emitMultiScaleYoloLoss rather than
re-deriving it here, so the floor cannot disagree with the loss for a reason of
my own making.

Inverting the encoder (see encode_targets_fpn in preprocess_visdrone.py, and the
decode in fpn_loss_breakdown.diou_terms):

    target ch0,ch1 = cx*g - cj, cy*g - ci   in [0,1)   <- pred needs logit(.)
    target ch2,ch3 = w_rel, h_rel                      <- pred needs log(./anchor)
    target ch4     = 1.0 on the assigned slot          <- pred needs a big logit
    target ch5..14 = one-hot class                     <- pred needs a big logit

    ./.venv/bin/python3 make_perfect_logits.py \\
        ../data/visdrone_fpn_overfit/val.bin /tmp/perfect.bin
"""
import sys
from pathlib import Path

import numpy as np

P, NC = 15, 10
IMG_BYTES = 448 * 448 * 3
SCALES = [(56, "p3"), (28, "p4"), (14, "p5")]
ANCH_DIR = Path("../data/visdrone")

# sigmoid saturates in float32: logit(1-1e-7) is ~16, and exp(tw) is capped at
# tw<=8 by the emitted DIoU block anyway. Keep the magnitudes finite and modest
# so the floor reflects an ACHIEVABLE prediction, not an infinite-logit fiction.
OBJ_HI, OBJ_LO, CLS_HI, CLS_LO = 12.0, -12.0, 12.0, -12.0
EPS = 1e-6


def load_anchors(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            w, h = line.split()
            out.append((float(w), float(h)))
    return out


def main():
    val_path, out_path = sys.argv[1], sys.argv[2]
    anchors = {k: load_anchors(ANCH_DIR / f"anchors_fpn_{k}.txt") for _, k in SCALES}
    lens = [len(anchors[k]) * P * g * g for g, k in SCALES]
    Ntot = sum(lens)
    rec = IMG_BYTES + Ntot * 4

    raw = np.memmap(val_path, dtype=np.uint8, mode="r")
    hdr = raw.size - (raw.size // rec) * rec
    n = (raw.size - hdr) // rec
    print(f"{val_path}: {n} records, Ntot={Ntot}, header={hdr}B")

    out = np.empty((n, Ntot), dtype=np.float32)
    npos = 0
    for i in range(n):
        off = hdr + i * rec + IMG_BYTES
        t = np.frombuffer(raw[off:off + Ntot * 4].tobytes(), dtype=np.float32).astype(np.float64)
        p = np.empty_like(t)
        o = 0
        for (g, key), ln in zip(SCALES, lens):
            A = len(anchors[key])
            tg = t[o:o + ln].reshape(A * P, g, g)
            pr = np.empty_like(tg)
            for a in range(A):
                b = a * P
                m = tg[b + 4] > 0.5
                aw, ah = anchors[key][a]
                # centre: invert sigmoid on the in-cell fraction
                fr = np.clip(tg[b + 0], EPS, 1 - EPS)
                pr[b + 0] = np.log(fr / (1 - fr))
                fr = np.clip(tg[b + 1], EPS, 1 - EPS)
                pr[b + 1] = np.log(fr / (1 - fr))
                # size: invert w = anchor * exp(tw), capped the way the emitter caps it
                pr[b + 2] = np.clip(np.log(np.maximum(tg[b + 2], EPS) / aw), -8.0, 8.0)
                pr[b + 3] = np.clip(np.log(np.maximum(tg[b + 3], EPS) / ah), -8.0, 8.0)
                # objectness: confident yes on assigned slots, confident no elsewhere
                pr[b + 4] = np.where(m, OBJ_HI, OBJ_LO)
                # class: confident one-hot on assigned slots (masked off elsewhere)
                oh = tg[b + 5:b + 5 + NC] > 0.5
                pr[b + 5:b + 5 + NC] = np.where(oh, CLS_HI, CLS_LO)
                npos += int(m.sum())
            p[o:o + ln] = pr.reshape(-1)
            o += ln
        out[i] = p.astype(np.float32)

    out.tofile(out_path)
    print(f"wrote {out_path}: {out.shape}, {npos} positives ({npos/n:.1f}/img)")
    print(f"\nnow: python3 scripts/fpn_loss_breakdown.py {out_path} {val_path} data/visdrone")


if __name__ == "__main__":
    main()
