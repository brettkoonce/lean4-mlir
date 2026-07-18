#!/usr/bin/env python3
"""Class frequencies over the ENCODED FPN training targets, and the T1b weights.

Counts on the encoded targets (not the raw annotations) so the weights match
exactly what `emitAnchorYoloLoss`'s class term sees -- post size-assignment and
post cell-collision.

Weights are normalized so that E_f[w] = sum_c f_c*w_c = 1, i.e. the expected
class weight under the GT distribution is 1. That keeps the TOTAL class-loss
magnitude unchanged and makes this a pure redistribution -- one lever at a time,
no accidental reweighting of cls against box/obj.
"""
import sys
import numpy as np

P, NC, IMG_BYTES = 15, 10, 448 * 448 * 3
VISDRONE = ["pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tric", "bus", "motor"]


def load_anchors(path):
    out = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                w, h = s.split()
                out.append((float(w), float(h)))
    return out


def main():
    train_path, anchor_dir = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    scales = [(56, "p3"), (28, "p4"), (14, "p5")]
    anchors = {k: load_anchors(f"{anchor_dir}/anchors_fpn_{k}.txt") for _, k in scales}
    lens = [len(anchors[k]) * P * g * g for g, k in scales]
    Ntot = sum(lens)
    rec = IMG_BYTES + Ntot * 4

    raw = np.memmap(train_path, dtype=np.uint8, mode="r")
    hdr = raw.size - (raw.size // rec) * rec
    n_rec = (raw.size - hdr) // rec
    idxs = range(0, n_rec, stride)
    print(f"{n_rec} records, scanning every {stride} -> {len(list(idxs))}")

    counts = np.zeros(NC, dtype=np.int64)
    per_scale = {k: np.zeros(NC, dtype=np.int64) for _, k in scales}
    npos = 0
    seen = 0
    for i in range(0, n_rec, stride):
        o = hdr + i * rec + IMG_BYTES
        t = np.frombuffer(raw[o : o + Ntot * 4].tobytes(), dtype=np.float32)
        off = 0
        for (g, key), ln in zip(scales, lens):
            A = len(anchors[key])
            tg = t[off : off + ln].reshape(A * P, g, g)
            for a in range(A):
                b = a * P
                m = tg[b + 4] > 0.5
                if m.any():
                    ct = tg[b + 5 : b + P][:, m]          # [NC, npos_a]
                    k = ct.argmax(axis=0)
                    c = np.bincount(k, minlength=NC)
                    counts += c
                    per_scale[key] += c
                    npos += int(m.sum())
            off += ln
        seen += 1
        if seen % 500 == 0:
            print(f"  ...{seen} imgs, {npos} positives", flush=True)

    f = counts / counts.sum()
    print(f"\ntotal encoded positives: {npos}  ({npos/seen:.1f}/img)\n")
    print(f"{'c':>3} {'name':>13} {'count':>9} {'freq':>8} {'w_inv':>8} {'w_sqrt':>8}"
          f" {'p3':>7} {'p4':>7} {'p5':>7}")
    print("-" * 82)

    inv = 1.0 / np.maximum(f, 1e-12)
    inv /= (f * inv).sum()               # E_f[w] = 1
    sq = 1.0 / np.sqrt(np.maximum(f, 1e-12))
    sq /= (f * sq).sum()
    for c in range(NC):
        print(f"{c:>3} {VISDRONE[c]:>13} {counts[c]:>9} {f[c]:>8.4f} {inv[c]:>8.3f} {sq[c]:>8.3f}"
              f" {per_scale['p3'][c]:>7} {per_scale['p4'][c]:>7} {per_scale['p5'][c]:>7}")

    print(f"\n  w_inv  range {inv.min():.3f}..{inv.max():.3f}  ({inv.max()/inv.min():.1f}x)")
    print(f"  w_sqrt range {sq.min():.3f}..{sq.max():.3f}  ({sq.max()/sq.min():.1f}x)")
    print("\nLean literal (sqrt-inverse, E_f[w]=1):")
    print("  [" + ", ".join(f"{x:.4f}" for x in sq) + "]")
    print("Lean literal (full inverse, E_f[w]=1):")
    print("  [" + ", ".join(f"{x:.4f}" for x in inv) + "]")


if __name__ == "__main__":
    main()
