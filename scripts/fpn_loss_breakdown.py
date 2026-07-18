#!/usr/bin/env python3
"""Per-term breakdown of the FPN multi-scale YOLO loss on a dumped checkpoint.

Replicates `emitMultiScaleYoloLoss` -> `emitAnchorYoloLoss` (MlirCodegen.lean)
exactly, in numpy, so we can see WHICH term the flat training loss is made of
before spending GPU-hours on a loss-rebalancing experiment.

  python3 scripts/fpn_loss_breakdown.py figures/yolo_fpn_e12/logits.bin \
      data/visdrone_fpn/val.bin data/visdrone

Prints, per scale and per anchor: num_pos, box / obj / cls loss contributions
(in the SAME normalization the trainer uses: sum over batch, then /B), plus the
objectness pos/neg split that T1a targets.
"""
import sys
import numpy as np

P = 15          # per-anchor slot: tx,ty,tw,th,obj,cls(10)
NC = 10
LAMBDA_BOX = 5.0
GAMMA = 2.0
LN = 0.5        # lambda_noobj (alpha for negatives)
IMG_BYTES = 448 * 448 * 3


def load_anchors(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            w, h = line.split()
            out.append((float(w), float(h)))
    return out


def diou_terms(pred, tgt, mask, aw, ah, g):
    """pred/tgt: [N,4,g,g] (tx,ty,tw,th | gx,gy,gw,gh), mask: [N,g,g]."""
    tx, ty, tw, th = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx, gy, gw, gh = tgt[:, 0], tgt[:, 1], tgt[:, 2], tgt[:, 3]
    cj = np.arange(g, dtype=np.float64)[None, None, :]
    ci = np.arange(g, dtype=np.float64)[None, :, None]
    invW = invH = 1.0 / g

    cx = (cj + 1.0 / (1.0 + np.exp(-tx))) * invW
    cy = (ci + 1.0 / (1.0 + np.exp(-ty))) * invH
    w = aw * np.exp(np.minimum(tw, 8.0))
    h = ah * np.exp(np.minimum(th, 8.0))
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2

    tcx, tcy = (cj + gx) * invW, (ci + gy) * invH
    X0, X1 = tcx - gw / 2, tcx + gw / 2
    Y0, Y1 = tcy - gh / 2, tcy + gh / 2

    iw = np.maximum(np.minimum(x1, X1) - np.maximum(x0, X0), 0.0)
    ih = np.maximum(np.minimum(y1, Y1) - np.maximum(y0, Y0), 0.0)
    inter = iw * ih
    union = np.maximum(w * h + gw * gh - inter, 1e-9)
    iou = inter / union

    rho2 = (cx - tcx) ** 2 + (cy - tcy) ** 2
    cw = np.maximum(x1, X1) - np.minimum(x0, X0)
    chh = np.maximum(y1, Y1) - np.minimum(y0, Y0)
    c2 = np.maximum(cw * cw + chh * chh, 1e-9)
    diou = iou - rho2 / c2
    return (1.0 - diou) * mask, iou * mask


def focal_bce(op, t):
    """Objectness focal-BCE, cellwise, matching the emitted block."""
    p = 1.0 / (1.0 + np.exp(-op))
    pt = t * p + (1.0 - t) * (1.0 - p)
    w = np.exp(GAMMA * np.log(np.maximum(1.0 - pt, 1e-12)))
    alpha = t + (1.0 - t) * LN
    # stable BCE-with-logits: max(x,0) - x*t + log(1+exp(-|x|))
    bce = np.maximum(op, 0.0) - op * t + np.log1p(np.exp(-np.abs(op)))
    return alpha * w * bce


def main():
    logits_path, val_path, anchor_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    scales = [(56, "p3"), (28, "p4"), (14, "p5")]
    anchors = {k: load_anchors(f"{anchor_dir}/anchors_fpn_{k}.txt") for _, k in scales}
    lens = [len(anchors[k]) * P * g * g for g, k in scales]
    Ntot = sum(lens)

    logits = np.fromfile(logits_path, dtype=np.float32)
    n_img = logits.size // Ntot
    logits = logits[: n_img * Ntot].reshape(n_img, Ntot).astype(np.float64)

    rec = IMG_BYTES + Ntot * 4
    raw = np.memmap(val_path, dtype=np.uint8, mode="r")
    # tolerate a small header (file size - n*rec)
    hdr = raw.size - (raw.size // rec) * rec
    n_rec = (raw.size - hdr) // rec
    n = min(n_img, n_rec)
    print(f"records: logits {n_img}, val {n_rec} -> using {n}; Ntot={Ntot} rec={rec}B hdr={hdr}B")

    tgts = np.empty((n, Ntot), dtype=np.float32)
    for i in range(n):
        off = hdr + i * rec + IMG_BYTES
        tgts[i] = np.frombuffer(raw[off : off + Ntot * 4].tobytes(), dtype=np.float32)
    tgts = tgts.astype(np.float64)

    B = float(n)  # trainer divides the summed loss by batch; use n for a per-image mean
    tot = {"box": 0.0, "obj": 0.0, "cls": 0.0}
    obj_pos_tot = obj_neg_tot = 0.0
    npos_tot = 0
    print()
    hdr_fmt = f"{'scale':>6} {'a':>2} {'num_pos':>9} {'box/img':>10} {'obj/img':>10} " \
              f"{'cls/img':>10} {'obj+':>9} {'obj-':>9} {'meanIoU':>8} {'mean p(obj+)':>12} {'mean p(obj-)':>12}"
    print(hdr_fmt)
    print("-" * len(hdr_fmt))

    off = 0
    for (g, key), ln in zip(scales, lens):
        A = len(anchors[key])
        blk = slice(off, off + ln)
        pr = logits[:, blk].reshape(n, A * P, g, g)
        tg = tgts[:, blk].reshape(n, A * P, g, g)
        for a in range(A):
            base = a * P
            m4 = tg[:, base + 4 : base + 5]             # [n,1,g,g]
            m = m4[:, 0]                                 # [n,g,g]
            npos = float(m.sum())
            aw, ah = anchors[key][a]

            cell, iou_cell = diou_terms(pr[:, base : base + 4], tg[:, base : base + 4], m, aw, ah, g)
            box = LAMBDA_BOX * cell.sum() / B
            mean_iou = iou_cell.sum() / max(npos, 1.0)

            op = pr[:, base + 4 : base + 5]
            fb = focal_bce(op, m4)
            obj = fb.sum() / B
            obj_pos = (fb * m4).sum() / B
            obj_neg = (fb * (1.0 - m4)).sum() / B
            p = 1.0 / (1.0 + np.exp(-op))
            mp_pos = float((p * m4).sum() / max(npos, 1.0))
            mp_neg = float((p * (1 - m4)).sum() / max((1 - m4).sum(), 1.0))

            cp = pr[:, base + 5 : base + P]
            ct = tg[:, base + 5 : base + P]
            mx = cp.max(axis=1, keepdims=True)
            sh = cp - mx
            lsm = sh - np.log(np.exp(sh).sum(axis=1, keepdims=True))
            cls = -(ct * lsm * m4).sum() / B

            tot["box"] += box; tot["obj"] += obj; tot["cls"] += cls
            obj_pos_tot += obj_pos; obj_neg_tot += obj_neg
            npos_tot += int(npos)
            print(f"{key:>6} {a:>2} {int(npos):>9} {box:>10.3f} {obj:>10.3f} {cls:>10.3f} "
                  f"{obj_pos:>9.3f} {obj_neg:>9.3f} {mean_iou:>8.4f} {mp_pos:>12.4f} {mp_neg:>12.4f}")
        off += ln

    total = sum(tot.values())
    ncells = Ntot // P
    print("\n" + "=" * 78)
    print(f"TOTAL loss/img = {total:.3f}")
    for k in ("box", "obj", "cls"):
        print(f"  {k:>4}: {tot[k]:>10.3f}  ({100*tot[k]/total:5.1f}%)")
    print(f"\n  objectness split:  pos {obj_pos_tot:.3f} ({100*obj_pos_tot/max(tot['obj'],1e-9):.1f}%)"
          f"   neg {obj_neg_tot:.3f} ({100*obj_neg_tot/max(tot['obj'],1e-9):.1f}%)")
    print(f"  positives/img = {npos_tot/n:.1f} of {ncells} anchor cells "
          f"(1 : {ncells*n/max(npos_tot,1):.0f} pos:neg)")
    print(f"\n  T1a rescale preview: obj term /num_pos instead of /B would scale")
    print(f"    objectness by B/num_pos = {n/(npos_tot/n):.4f}x -> obj/img {tot['obj']*n/(npos_tot/n)/n*1:.3f}"
          if npos_tot else "")


if __name__ == "__main__":
    main()
