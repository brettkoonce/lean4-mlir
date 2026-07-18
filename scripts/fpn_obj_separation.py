#!/usr/bin/env python3
"""Is the FPN objectness head DISCRIMINATIVE, or just uniformly low?

The loss breakdown says background is only ~25% of the objectness loss, which
rules out the "negatives swamp the positives" story. The remaining question is
whether objectness RANKS positives above negatives at all -- that is what
detection actually needs, and what conf-threshold/NMS consume.

Reports AUC (P[score(pos) > score(neg)]), the positive rate in the top-K cells,
and the class-head collapse, per scale.
"""
import sys
import numpy as np

P, NC, IMG_BYTES = 15, 10, 448 * 448 * 3


def load_anchors(path):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                w, h = s.split()
                out.append((float(w), float(h)))
    return out


def auc(pos, neg, cap=400000):
    """Rank-based AUC, subsampled for tractability."""
    rng = np.random.default_rng(0)
    if pos.size > cap:
        pos = rng.choice(pos, cap, replace=False)
    if neg.size > cap:
        neg = rng.choice(neg, cap, replace=False)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rpos = ranks[: pos.size].sum()
    return (rpos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)


def main():
    logits_path, val_path, anchor_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    scales = [(56, "p3"), (28, "p4"), (14, "p5")]
    anchors = {k: load_anchors(f"{anchor_dir}/anchors_fpn_{k}.txt") for _, k in scales}
    lens = [len(anchors[k]) * P * g * g for g, k in scales]
    Ntot = sum(lens)

    logits = np.fromfile(logits_path, dtype=np.float32)
    n = logits.size // Ntot
    logits = logits[: n * Ntot].reshape(n, Ntot)

    rec = IMG_BYTES + Ntot * 4
    raw = np.memmap(val_path, dtype=np.uint8, mode="r")
    hdr = raw.size - (raw.size // rec) * rec
    n = min(n, (raw.size - hdr) // rec)
    tgts = np.empty((n, Ntot), dtype=np.float32)
    for i in range(n):
        o = hdr + i * rec + IMG_BYTES
        tgts[i] = np.frombuffer(raw[o : o + Ntot * 4].tobytes(), dtype=np.float32)

    print(f"{n} images\n")
    print(f"{'scale':>6} {'a':>2} {'AUC':>7} {'pos':>7} {'top-100/img %pos':>17} {'objlogit p5/50/95':>26}")
    print("-" * 74)

    all_pos, all_neg = [], []
    cls_pred_hist = np.zeros(NC, dtype=np.int64)
    cls_gt_hist = np.zeros(NC, dtype=np.int64)
    off = 0
    for (g, key), ln in zip(scales, lens):
        A = len(anchors[key])
        pr = logits[:, off : off + ln].reshape(n, A * P, g, g)
        tg = tgts[:, off : off + ln].reshape(n, A * P, g, g)
        for a in range(A):
            b = a * P
            m = tg[:, b + 4].reshape(n, -1)
            o = pr[:, b + 4].reshape(n, -1).astype(np.float64)
            pos, neg = o[m > 0.5], o[m <= 0.5]
            all_pos.append(pos); all_neg.append(neg)
            # top-100 cells per image by objectness: what fraction are real?
            k = min(100, o.shape[1])
            idx = np.argpartition(-o, k - 1, axis=1)[:, :k]
            frac = np.take_along_axis(m, idx, axis=1).mean() * 100
            q = np.percentile(o, [5, 50, 95])
            print(f"{key:>6} {a:>2} {auc(pos,neg):>7.4f} {pos.size:>7} {frac:>16.2f}% "
                  f"{q[0]:>8.2f}{q[1]:>9.2f}{q[2]:>9.2f}")
            # class head: argmax on POSITIVE cells vs GT class
            cp = pr[:, b + 5 : b + P].reshape(n, NC, -1)
            ct = tg[:, b + 5 : b + P].reshape(n, NC, -1)
            sel = m > 0.5
            if sel.any():
                pk = cp.argmax(axis=1)[sel]
                gk = ct.argmax(axis=1)[sel]
                cls_pred_hist += np.bincount(pk, minlength=NC)
                cls_gt_hist += np.bincount(gk, minlength=NC)
        off += ln

    pos = np.concatenate(all_pos); neg = np.concatenate(all_neg)
    print("\n" + "=" * 74)
    print(f"OVERALL objectness AUC = {auc(pos,neg):.4f}   (0.5 = no signal, 1.0 = perfect)")
    print(f"  mean logit  pos {pos.mean():+.3f}   neg {neg.mean():+.3f}   gap {pos.mean()-neg.mean():+.3f}")
    print(f"  std   logit pos {pos.std():.3f}    neg {neg.std():.3f}")
    print("\nCLASS HEAD on positive cells (argmax vs GT):")
    tot = cls_gt_hist.sum()
    acc = None
    print(f"  {'cls':>4} {'GT':>8} {'pred':>8}")
    for c in range(NC):
        print(f"  {c:>4} {cls_gt_hist[c]:>8} {cls_pred_hist[c]:>8}")
    print(f"  argmax collapsed onto {int((cls_pred_hist>0).sum())}/{NC} classes; "
          f"most-predicted class takes {100*cls_pred_hist.max()/max(tot,1):.1f}% of positives "
          f"(GT majority class is {100*cls_gt_hist.max()/max(tot,1):.1f}%)")


if __name__ == "__main__":
    main()
