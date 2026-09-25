#!/usr/bin/env python3
"""The ArASL figure — planning/arasl_people_watching_demo.md §6.

(a) the alphabet: one crop per class, 32 tiles in the 8-column grid, the 100th frame
    of each (mid-burst, not the first shot);
(b) one 8-tile row: a test image and its nearest training image for two letters under
    `random` (the same hand, a few grey levels away) and the same two letters under
    `blocked` (a different hand, often a different letter) — the leak audit as a picture;
(c) one 8-tile row: the four most confused letter pairs on the blocked split, a test
    crop of each letter (the one the net called the other, where the logits say).
`--layout one-row` folds (b) and (c) into a single row of two pairs each. (b)/(c) need
the scorer's JSON; without it only the grid is drawn.

Reads data/arasl/images_u8.npy + meta_<split>.npz from scripts/datasets/preprocess_arasl.py; (c) reads
the confused pairs and the confusion matrix from `scripts/demos/arasl_score.py --json`.

  .venv/bin/python scripts/demos/arasl_figure.py [--score=runs/.../score_blocked.json]
                   [--logits=runs/.../*_logits_test.bin] [--layout=two-rows|one-row]
                   [--out=demos/figures/arasl_signs.png]
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

T, GAP, PAD, COLS = 96, 6, 12, 8
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BG, FG, ACCENT, COOL = (16, 16, 16), (235, 235, 235), (255, 200, 60), (120, 200, 255)


def tile(a, s=T):
    return Image.fromarray(a).resize((s, s), Image.LANCZOS).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/arasl")
    ap.add_argument("--score", default=None, help="scorer JSON for the blocked split (panel c)")
    ap.add_argument("--logits", default=None, help="blocked-split test logits, to pick the examples for (c)")
    ap.add_argument("--out", default="demos/figures/arasl_signs.png")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--layout", choices=("two-rows", "one-row"), default="two-rows",
                    help="under the grid: (b) and (c) as one 8-tile row each, or both in a single row")
    args = ap.parse_args()
    imgs = np.load(os.path.join(args.data, "images_u8.npy"))
    metas = {s: np.load(os.path.join(args.data, f"meta_{s}.npz")) for s in ("random", "blocked")}
    classes = [str(c) for c in metas["random"]["classes"]]
    n = len(imgs)
    glabel = np.zeros(n, dtype=np.int32)
    glabel[metas["random"]["index"]] = metas["random"]["label"]
    gnum = np.zeros(n, dtype=np.int32)
    gnum[metas["random"]["index"]] = metas["random"]["file_num"]
    font = ImageFont.truetype(FONT, 15)
    small = ImageFont.truetype(FONT_R, 12)
    font_r = ImageFont.truetype(FONT, 13)       # row captions
    x64 = imgs.reshape(n, -1).astype(np.int16)
    mad = lambda i, j: np.abs(x64[i] - x64[j]).mean()

    W = PAD * 2 + COLS * T + (COLS - 1) * GAP
    rows_a = (len(classes) + COLS - 1) // COLS
    LBL = 30                                    # two label lines under a tile
    ROW = 18 + T + LBL                          # caption line + tiles + labels
    score = json.load(open(args.score)) if args.score else None
    n_rows = 0 if not score else (1 if args.layout == "one-row" else 2)
    hA = 30 + rows_a * (T + 18)
    sheet = Image.new("RGB", (W, PAD + hA + n_rows * ROW + PAD), BG)
    d = ImageDraw.Draw(sheet)

    def pair_row(y, caption, items):
        """Four (left image, left 2-line label, right image, right 2-line label, colour) pairs."""
        d.text((PAD, y), caption, fill=FG, font=font_r)
        y += 18
        for k, (il, ll, ir, lr, col) in enumerate(items):
            x = PAD + k * (2 * T + 2 * GAP)
            sheet.paste(tile(il), (x, y))
            sheet.paste(tile(ir), (x + T + GAP, y))
            for dx, lab in ((0, ll), (T + GAP, lr)):
                d.text((x + dx + 2, y + T + 1), lab[0], fill=col, font=small)
                d.text((x + dx + 2, y + T + 15), lab[1], fill=FG, font=small)

    # ── (a) the alphabet ──
    y0 = PAD
    d.text((PAD, y0), "(a)  ArASL — the 32 letters, one crop per class (54,049 frames, 64×64 grey, CC BY 4.0)", fill=FG, font=font)
    y0 += 30
    for ci, c in enumerate(classes):
        sel = np.flatnonzero(glabel == ci)
        k = sel[min(99, len(sel) - 1)]
        r, cc = divmod(ci, COLS)
        x, y = PAD + cc * (T + GAP), y0 + r * (T + 18)
        sheet.paste(tile(imgs[k]), (x, y))
        d.text((x + 2, y + T + 1), c, fill=ACCENT, font=small)
    if not score:
        sheet.save(args.out)
        print(args.out, sheet.size)
        return

    # ── (b) a test image and its nearest training image, the same letters under each split ──
    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(classes), 4, replace=False)[:2]
    b_items = []
    for split in ("random", "blocked"):
        m = metas[split]
        te, nn, nd = m["test_index"], m["test_nn_train_index"], m["test_nn_mad64"]
        col = ACCENT if split == "random" else COOL
        for ci in picks:
            cand = np.flatnonzero(glabel[te] == ci)
            # the median-distance test image of that class: typical, not the best case
            j = cand[np.argsort(nd[cand])[len(cand) // 2]]
            i_te, i_tr = te[j], nn[j]
            b_items.append((imgs[i_te], (f"{classes[ci]} #{gnum[i_te]}", f"test · {split}"),
                            imgs[i_tr], (f"{classes[glabel[i_tr]]} #{gnum[i_tr]}", f"train · |Δ| {mad(i_te, i_tr):.1f}"), col))

    # ── (c) the most confused pairs on the blocked split ──
    te = metas["blocked"]["test_index"]
    pred = None
    if args.logits:
        pred = np.fromfile(args.logits, dtype=np.float32).reshape(len(te), -1).argmax(axis=1)
    n_runs = len(score["runs"])
    c_items = []
    for pr in score["confused"]:
        a, b = classes.index(pr["a"]), classes.index(pr["b"])
        ia = np.flatnonzero(glabel[te] == a)
        ib = np.flatnonzero(glabel[te] == b)
        # a crop the net actually called the other letter, when the logits say which
        if pred is not None and (pred[ia] == b).any():
            ia = ia[pred[ia] == b]
        if pred is not None and (pred[ib] == a).any():
            ib = ib[pred[ib] == a]
        c_items.append((imgs[te[ia[0]]], (pr["a"], f"→ {pr['b']}: {pr['a_as_b'] / n_runs:.0f}/run"),
                        imgs[te[ib[0]]], (pr["b"], f"→ {pr['a']}: {pr['b_as_a'] / n_runs:.0f}/run"), COOL))

    y0 = PAD + hA
    if args.layout == "one-row":
        pair_row(y0, "(b)  a test image and its nearest training image, random then blocked split;   "
                     "(c)  two confused pairs, blocked split",
                 [b_items[0], b_items[2]] + c_items[:2])
    else:
        pair_row(y0, "(b)  a test image and its nearest training image — random split (left), the same two letters blocked (right)",
                 b_items)
        pair_row(y0 + ROW, "(c)  blocked split — the four most confused letter pairs, a test crop of each letter, confusions per run",
                 c_items[:4])
    sheet.save(args.out)
    print(args.out, sheet.size)


if __name__ == "__main__":
    main()
