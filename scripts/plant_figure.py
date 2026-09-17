#!/usr/bin/env python3
"""The PlantVillage → PlantDoc figure — planning/plant_lab_to_field_demo.md §6. Real maps: the
layout is runs/2026-09-17-plant/mock_figure.py's, the heat is what the trained net produced.

Eight tiles wide throughout.
(a) four PlantVillage test leaves (tomato, corn, apple, grape; the image of median CAM share in
    its class, so typical not best), two per row: input | base CAM | base Shapley (7×7, sampled) |
    +backgrounds CAM, with the share of map mass inside the leaf under each map;
(b) the same four diseases on PlantDoc — each a field photo the base net misreads, the typical
    case at ~18% — two per row: input | base net | +backgrounds | +field labels (the fold's net
    that never trained on that image), the predicted label under each (✓/✗ against the truth);
(c) three training leaves beside their composites — what the background fix trained on.

Inputs: the parts (`data/plant/*.bin`), the masks, the CAM dumps of two runs (`<pfx>_cam_<part>.bin`
+ `_campred_`), and optionally the Shapley grid maps (`plant_shapley.py grid --score` → `_maps.npz`,
whose `images` must include the picked test indices — pick with `--print-picks` first).

  .venv/bin/python scripts/plant_figure.py --base <base run prefix> --comp <comp run prefix> [--split pvg]
        [--shap <probe>_logits_shap_probe_maps.npz] [--field <fold run prefix with {k}>] [--print-picks]
        [--out demos/figures/plant_lab_to_field.png]
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAIRS = [("Tomato___Early_blight", "tomato, early blight"), ("Corn_(maize)___Northern_Leaf_Blight", "corn, leaf blight"),
         ("Apple___Apple_scab", "apple, scab"), ("Grape___Black_rot", "grape, black rot")]
T, GAP, PAD, G, S = 150, 6, 12, 7, 224
BG, FG, ACCENT, COOL, WARN = (16, 16, 16), (235, 235, 235), (255, 200, 60), (120, 200, 255), (255, 110, 110)
FONT_B, FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
VIRIDIS = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], dtype=np.float32)


def viridis(h):
    x = np.clip(h, 0, 1) * 4
    i = np.minimum(x.astype(int), 3)
    t = (x - i)[..., None]
    return (VIRIDIS[i] * (1 - t) + VIRIDIS[i + 1] * t).astype(np.uint8)


def read_part(path, side=S):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n = int(np.frombuffer(bytes(raw[:4]), dtype=np.uint32)[0])
    rec = 1 + 3 * side * side
    return n, raw, rec, side


def image_of(part, k):
    n, raw, rec, side = part
    a = np.asarray(raw[4 + k * rec + 1:4 + (k + 1) * rec]).reshape(3, side, side).transpose(1, 2, 0)
    if side != S:
        o = (side - S) // 2
        a = a[o:o + S, o:o + S]
    return Image.fromarray(np.ascontiguousarray(a))


def up(map7):
    m = np.maximum(map7, 0)
    m = m / (m.max() + 1e-9)
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((S, S), Image.BILINEAR)) / 255.0


def overlay(img, heat, alpha=0.45):
    base = np.asarray(img, dtype=np.float32)
    return Image.fromarray(np.clip(base * (1 - alpha) + viridis(heat).astype(np.float32) * alpha, 0, 255).astype(np.uint8))


def share(map7, m7):
    m = np.maximum(map7, 0)
    return float((m * m7).sum() / (m.sum() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/plant")
    ap.add_argument("--split", default="pvg", choices=("pv", "pvg"))
    ap.add_argument("--base", required=True, help="run prefix of the base arm (has _cam_<split>_test.bin and _cam_pd_all.bin)")
    ap.add_argument("--comp", default=None, help="run prefix of the +backgrounds arm")
    ap.add_argument("--shap", default=None, help="grid Shapley maps npz for the picked test images")
    ap.add_argument("--field", default=None, help="run prefix pattern of the field-fine-tuned nets with {k} for the fold, "
                    "e.g. runs/.../plant_resnet34_grouped_base_ckpt_field{k}_s1 (their cam=1 dumps on pd_fold{k}_test)")
    ap.add_argument("--print-picks", action="store_true")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--pd-seed", type=int, default=23, help="seed for the PlantDoc picks (web photos carry watermarks; 23 is the book's draw)")
    ap.add_argument("--out", default="demos/figures/plant_lab_to_field.png")
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    meta = np.load(os.path.join(args.data, "meta_plant.npz"))
    classes = [str(c) for c in meta["classes"]]
    te = meta[f"{args.split}_test_index"]
    lab_te = meta["pv_label"][te]
    m7 = np.load(os.path.join(args.data, f"{args.split}_test_mask7.npy"))
    pv_test = read_part(os.path.join(args.data, f"{args.split}_test.bin"))
    pd_all = read_part(os.path.join(args.data, "pd_all.bin"))
    lab_pd = meta["pd_label"]
    cams = {}
    for tag, pfx in (("base", args.base), ("comp", args.comp)):
        if pfx is None:
            continue
        for part, n in ((f"{args.split}_test", len(te)), ("pd_all", len(lab_pd))):
            f = f"{pfx}_cam_{part}.bin"
            if os.path.exists(f):
                cams[(tag, part)] = (np.fromfile(f, dtype=np.float32).reshape(n, 2, G, G),
                                     np.fromfile(f"{pfx}_campred_{part}.bin", dtype=np.int32))
    shap = np.load(args.shap) if args.shap else None
    shap_ids = list(shap["images"]) if shap is not None else []
    # the field-fine-tuned nets: fold k's net scored fold k's held-out images, in np.flatnonzero(fold == k) order
    field_cam = {}
    if args.field:
        pd_fold = meta["pd_fold"]
        for k in range(5):
            idx = np.flatnonzero(pd_fold == k)
            c7 = np.fromfile(args.field.format(k=k) + f"_cam_pd_fold{k}_test.bin", dtype=np.float32).reshape(len(idx), 2, G, G)
            pr = np.fromfile(args.field.format(k=k) + f"_campred_pd_fold{k}_test.bin", dtype=np.int32)
            for pos, i in enumerate(idx):
                field_cam[int(i)] = (c7[pos], int(pr[pos]))

    # picks: per class, the test image whose base true-class CAM share is the class median
    base_cam = cams[("base", f"{args.split}_test")][0][:, 0]
    sh = np.array([share(base_cam[i], m7[i]) for i in range(len(te))])
    picks_pv = []
    for cls, _ in PAIRS:
        c = classes.index(cls)
        cand = np.flatnonzero(lab_te == c)
        picks_pv.append(int(cand[np.argsort(sh[cand])[len(cand) // 2]]))
    # PlantDoc: the typical case for a net at ~16% — a field photo of the class the base net
    # misreads (the section says so); the fix columns then show whether it is corrected
    base_pd_pred = cams[("base", "pd_all")][1]
    rng_pd = np.random.RandomState(args.pd_seed) if args.pd_seed is not None else rng
    picks_pd = []
    for cls, _ in PAIRS:
        cand = np.flatnonzero(lab_pd == classes.index(cls))
        wrong = cand[base_pd_pred[cand] != classes.index(cls)]
        picks_pd.append(int(rng_pd.choice(wrong if len(wrong) else cand)))
    if args.print_picks:
        print("PlantVillage test indices (into the test part):", picks_pv, "→ global", [int(te[i]) for i in picks_pv])
        print("PlantDoc pd_all indices:", picks_pd)
        return

    font, font_r, small = ImageFont.truetype(FONT_B, 15), ImageFont.truetype(FONT_B, 12), ImageFont.truetype(FONT_R, 12)
    # eight tiles wide throughout: (a) two leaves × 4 maps per row, (b) two photos × 4 per row, (c) three pairs
    W = PAD * 2 + 8 * (T + GAP) - GAP
    ROW = T + 34
    H = PAD + (30 + 2 * ROW) + (30 + 2 * ROW) + (30 + ROW) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)

    def put(x, y, im, l1, l2, col=ACCENT):
        sheet.paste(im.resize((T, T), Image.LANCZOS), (x, y))
        d.text((x + 2, y + T + 2), l1, fill=col, font=small)
        if l2:
            d.text((x + 2, y + T + 16), l2, fill=FG, font=small)

    short = lambda c: classes[c].split("___")[1].replace("_", " ")[:20]
    y = PAD
    d.text((PAD, y), "(a)  PlantVillage test leaves — input | CAM, base net | Shapley map, base net | CAM after the background fix;  "
                     "under each map, the share of it inside the leaf", fill=FG, font=font_r)
    y += 30
    for k, (i, (cls, nice)) in enumerate(zip(picks_pv, PAIRS)):
        img = image_of(pv_test, i)
        x0 = PAD + (k % 2) * 4 * (T + GAP)
        yy = y + (k // 2) * ROW
        put(x0, yy, img, nice, "lab photograph")
        cb = cams[("base", f"{args.split}_test")][0][i, 0]
        put(x0 + (T + GAP), yy, overlay(img, up(cb)), "CAM, base", f"{100 * share(cb, m7[i]):.0f}% inside the leaf", WARN)
        gi = int(te[i])
        if shap is not None and (gi in shap_ids or i in shap_ids):
            jj = shap_ids.index(gi) if gi in shap_ids else shap_ids.index(i)
            phi = shap["phi"][jj]
            put(x0 + 2 * (T + GAP), yy, overlay(img, up(phi)), "Shapley, base", f"{100 * share(phi, m7[i]):.0f}% inside the leaf", WARN)
        if ("comp", f"{args.split}_test") in cams:
            cc = cams[("comp", f"{args.split}_test")][0][i, 0]
            put(x0 + 3 * (T + GAP), yy, overlay(img, up(cc)), "CAM, +backgrounds", f"{100 * share(cc, m7[i]):.0f}% inside the leaf", COOL)
    y += 2 * ROW
    d.text((PAD, y), "(b)  the same four diseases in PlantDoc field photographs the base net misreads — input | base net | +backgrounds | +field labels;  "
                     "the prediction under each", fill=FG, font=font_r)
    y += 30
    for k, (i, (cls, nice)) in enumerate(zip(picks_pd, PAIRS)):
        img = image_of(pd_all, i)
        x0 = PAD + (k % 2) * 4 * (T + GAP)
        yy = y + (k // 2) * ROW
        put(x0, yy, img, nice, "field photograph")
        cols = [("base", "base net")]
        if ("comp", "pd_all") in cams:
            cols.append(("comp", "+backgrounds"))
        for col, (tag, name) in enumerate(cols, start=1):
            c7, pr = cams[(tag, "pd_all")]
            ok = pr[i] == lab_pd[i]
            put(x0 + col * (T + GAP), yy, overlay(img, up(c7[i, 1])), name, f"→ {short(pr[i])} {'✓' if ok else '✗'}", COOL if ok else WARN)
        if i in field_cam:
            c7, pr = field_cam[i]
            ok = pr == lab_pd[i]
            put(x0 + 3 * (T + GAP), yy, overlay(img, up(c7[1])), "+field labels", f"→ {short(pr)} {'✓' if ok else '✗'}", COOL if ok else WARN)
    y += 2 * ROW
    d.text((PAD, y), "(c)  what the background fix trains on — a PlantVillage training leaf and the same leaf pasted onto an unrelated photograph", fill=FG, font=font_r)
    y += 30
    tr = read_part(os.path.join(args.data, f"{args.split}_train.bin"), 256)
    comp = read_part(os.path.join(args.data, f"{args.split}_train_comp.bin"), 256)
    lab_tr = np.asarray(tr[1][4:4 + tr[0] * tr[2]:tr[2]]).astype(np.int32)
    for k, cls in enumerate(("Tomato___Late_blight", "Apple___Cedar_apple_rust", "Corn_(maize)___Common_rust_")):
        i = int(rng.choice(np.flatnonzero(lab_tr == classes.index(cls))))
        x0 = PAD + k * 2 * (T + GAP) + (k * (T + GAP)) // 3 * 0
        x0 = PAD + k * (2 * T + 2 * GAP + (T + GAP) // 2)
        put(x0, y, image_of(tr, i), cls.split("___")[1].replace("_", " "), "training leaf")
        put(x0 + (T + GAP), y, image_of(comp, i), "composite", "on an Imagenette image", COOL)
    sheet.save(args.out)
    print(args.out, sheet.size)


if __name__ == "__main__":
    main()
