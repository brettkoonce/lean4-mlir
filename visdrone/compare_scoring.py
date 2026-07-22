#!/usr/bin/env python3
"""Put the Lean detector and a working detector on ONE identical metric.

planning/yolo_assignment.md measured the Lean arm's objectness AUC at 0.741 and
could not tell whether that was mediocre or catastrophic, because there was no
working detector to compare against. This computes the same quantity for both.

The definition has to be architecture-neutral, because the two detectors assign
targets completely differently (Lean: static best-anchor, one cell per GT;
YOLOv8: anchor-free dynamic TAL). So:

    a candidate slot is POSITIVE if its cell centre falls inside some GT box.

That is computable for both, involves no assignment rule, and is exactly the
question ranking has to answer: "is there an object here?"

Both sides are scored against the FULL annotations, not the MAX_BBOXES=56
truncated val.bin, so neither arm gets the truncation's easy-image bias.

    ./.venv/bin/python3 compare_scoring.py --lean ../figures/yolo_fpn_t2a_e12/logits.bin
    ./.venv/bin/python3 compare_scoring.py --yolo runs/scratch_squash_12ep_448/weights/best.pt
"""
import argparse
from pathlib import Path

import numpy as np

SRC = Path("../data/visdrone/VisDrone2019-DET-val")
ANCH = Path("../data/visdrone")
FPN_GRIDS = (56, 28, 14)
PER_ANCHOR = 15
NUM_CLASSES = 10


def load_anchors(p):
    return [tuple(map(float, ln.split()))
            for ln in Path(p).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def parse_ann(txt):
    out = []
    for ln in Path(txt).read_text().splitlines():
        ln = ln.strip().rstrip(",")
        if not ln:
            continue
        p = ln.split(",")
        if len(p) < 6:
            continue
        x, y, w, h, score, cat = (int(float(v)) for v in p[:6])
        if score == 0 or cat == 0 or cat == 11 or w <= 0 or h <= 0:
            continue
        out.append((x, y, x + w, y + h))
    return out


def gt_norm(stem, iw, ih):
    """GT boxes in normalized (0..1) coords -- the frame BOTH arms use."""
    b = parse_ann(SRC / "annotations" / f"{stem}.txt")
    if not b:
        return np.zeros((0, 4), np.float32)
    a = np.array(b, np.float32)
    a[:, [0, 2]] /= iw
    a[:, [1, 3]] /= ih
    return a


def centres_inside(cx, cy, gt):
    """boolean mask: does each (cx,cy) fall inside any GT box?"""
    if len(gt) == 0:
        return np.zeros(cx.shape, bool)
    m = np.zeros(cx.shape, bool)
    for x0, y0, x1, y1 in gt:
        m |= (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    return m


def auc(scores, pos):
    """Mann-Whitney AUC; O(n log n), fine at 7M slots."""
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def report(name, scores, pos, n_img):
    a = auc(scores, pos)
    sp, sb = scores[pos], scores[~pos]
    k = min(1000 * n_img, scores.size)
    cut = np.partition(scores, -k)[-k]
    print(f"\n=== {name} ===")
    print(f"  candidate slots      {scores.size//n_img:>8,} per image "
          f"({scores.size:,} total over {n_img} images)")
    print(f"  positives            {pos.sum()/n_img:>8.1f} per image "
          f"({100*pos.mean():.2f}% of slots)")
    print(f"  OBJECT-vs-BACKGROUND AUC   {a:.4f}")
    print(f"  score  positives     mean {sp.mean():.4f}  median {np.median(sp):.4f}  "
          f"p99 {np.percentile(sp,99):.4f}  max {sp.max():.4f}")
    print(f"  score  background    mean {sb.mean():.4f}  median {np.median(sb):.4f}  "
          f"p99 {np.percentile(sb,99):.4f}  max {sb.max():.4f}")
    print(f"  separation           {sp.mean()-sb.mean():+.4f} "
          f"({(sp.mean()-sb.mean())/max(sb.std(),1e-9):.2f} background sigma)")
    print(f"  top-1000/img cut     score {cut:.4f} -> "
          f"{100*(sp>=cut).mean():.1f}% of positives survive it")
    return a


def do_lean(path, n_img, mode="objcls"):
    scales = [(g, load_anchors(ANCH / f"anchors_fpn_{p}.txt"))
              for g, p in zip(FPN_GRIDS, ("p3", "p4", "p5"))]
    pred_w = sum(len(a) * PER_ANCHOR * g * g for g, a in scales)
    raw = np.fromfile(path, dtype=np.float32)
    n_rows = raw.size // pred_w
    rows = raw[: n_rows * pred_w].reshape(n_rows, pred_w)
    stems = sorted(p.stem for p in (SRC / "images").glob("*.jpg"))
    n_img = min(n_img, n_rows, len(stems))

    S, P = [], []
    from PIL import Image
    for r in range(n_img):
        with Image.open(SRC / "images" / f"{stems[r]}.jpg") as im:
            iw, ih = im.size
        gt = gt_norm(stems[r], iw, ih)
        off = 0
        for g, anch in scales:
            A = len(anch)
            blk = rows[r][off:off + A * PER_ANCHOR * g * g].reshape(A * PER_ANCHOR, g, g)
            off += A * PER_ANCHOR * g * g
            jj = (np.arange(g) + 0.5) / g
            cx = np.broadcast_to(jj.reshape(1, g), (g, g))
            cy = np.broadcast_to(jj.reshape(g, 1), (g, g))
            inside = centres_inside(cx, cy, gt)
            for a in range(A):
                b = a * PER_ANCHOR
                obj = 1.0 / (1.0 + np.exp(-np.clip(blk[b + 4], -60, 60)))
                cls = blk[b + 5: b + 5 + NUM_CLASSES]
                e = np.exp(cls - cls.max(axis=0, keepdims=True))
                clsp = (e / e.sum(axis=0)).max(axis=0)
                S.append((obj * clsp).ravel() if mode == "objcls" else obj.ravel())
                P.append(inside.ravel())
    return np.concatenate(S), np.concatenate(P), n_img


def do_yolo(weights, n_img):
    import torch
    from ultralytics import YOLO
    from PIL import Image

    m = YOLO(weights)
    net = m.model.eval().cuda()
    stems = sorted(p.stem for p in (SRC / "images").glob("*.jpg"))[:n_img]

    # eval-mode forward returns (pred[B, 4+nc, N], aux). The N anchors are the
    # P3/P4/P5 grids concatenated in row-major order -- the same 56/28/14 grids
    # the Lean detector uses, just without its 3 anchors per cell. Class scores
    # are already sigmoid probabilities at this point.
    cxs, cys = [], []
    for g in FPN_GRIDS:
        jj = (np.arange(g) + 0.5) / g
        cxs.append(np.broadcast_to(jj.reshape(1, g), (g, g)).ravel())
        cys.append(np.broadcast_to(jj.reshape(g, 1), (g, g)).ravel())
    cx_all, cy_all = np.concatenate(cxs), np.concatenate(cys)

    S, P = [], []
    for stem in stems:
        # feed the SQUASHED square image so the grid maps straight onto the
        # normalized frame -- no letterbox padding to undo
        with Image.open(SRC / "images" / f"{stem}.jpg") as im:
            iw, ih = im.size
            sq = im.convert("RGB").resize((448, 448), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(sq, np.float32).transpose(2, 0, 1) / 255.0)
        with torch.no_grad():
            pred = net(x[None].cuda())[0]
        score = pred[0, 4:, :].max(0).values.float().cpu().numpy()
        assert score.size == cx_all.size, f"{score.size} anchors vs {cx_all.size} grid slots"
        S.append(score)
        P.append(centres_inside(cx_all, cy_all, gt_norm(stem, iw, ih)))
    return np.concatenate(S), np.concatenate(P), len(stems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", default="../figures/yolo_fpn_t2a_e12/logits.bin")
    ap.add_argument("--yolo", default=None)
    ap.add_argument("--n", type=int, default=100, help="val images to use")
    args = ap.parse_args()

    results = {}
    if args.lean:
        s, p, n = do_lean(args.lean, args.n, "obj")
        report("LEAN FPN, score = sigmoid(obj) ALONE", s, p, n)
        s, p, n = do_lean(args.lean, args.n, "objcls")
        results["Lean FPN (T2a e12), score = sigmoid(obj) x max softmax(cls)"] = report(
            "LEAN FPN, score = PRODUCTION sigmoid(obj) x max softmax(cls)", s, p, n)
    if args.yolo:
        s, p, n = do_yolo(args.yolo, args.n)
        results["YOLOv8s, score = max sigmoid(cls)"] = report(
            "YOLOv8s (from scratch, no aug, 448)", s, p, n)

    if len(results) == 2:
        (na, a), (nb, b) = results.items()
        print(f"\n=== VERDICT ===")
        print(f"  {na}\n      AUC {a:.4f}")
        print(f"  {nb}\n      AUC {b:.4f}")
        print(f"  the working detector separates object from background "
              f"{(1-a)/(1-b):.1f}x better (in error terms)")


if __name__ == "__main__":
    main()
