#!/usr/bin/env python3
"""Look at what the Lean FPN trainer actually sees.

Eight numerical probes have run on this arm's logits and nobody has rendered a
single image. This does the check that was missing: decode a record straight out
of data/visdrone_fpn/train.bin, draw the ENCODED targets on the image the network
is fed, and draw the raw annotations (pushed through the same squash-resize) on
top as a control. If the two sets of boxes do not coincide, the image/target
correspondence is broken and every number in planning/yolo_assignment.md is
measured against noise.

Also reports the object-size distribution in fed pixels, which is what makes the
"2-5 px targets" claim concrete.

    ./.venv/bin/python3 inspect_lean_bin.py [--n 6] [--out ../figures/visdrone_bin_check]
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path("..")
BIN = REPO / "data/visdrone_fpn/train.bin"
SRC = REPO / "data/visdrone/VisDrone2019-DET-train"
ANCHOR_FILES = [REPO / f"data/visdrone/anchors_fpn_p{p}.txt" for p in (3, 4, 5)]

IMG_SIZE = 448
FPN_GRIDS = (56, 28, 14)
PER_ANCHOR = 15
NUM_CLASSES = 10
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
           "tricycle", "awning-tricycle", "bus", "motor"]


def load_anchor_counts():
    counts = []
    for f in ANCHOR_FILES:
        # each file leads with a '#' comment describing the k-means fit
        n = len([ln for ln in f.read_text().splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")])
        counts.append(n)
    return counts


def parse_ann(txt_path):
    boxes = []
    for ln in Path(txt_path).read_text().splitlines():
        ln = ln.strip().rstrip(",")
        if not ln:
            continue
        p = ln.split(",")
        if len(p) < 6:
            continue
        x, y, w, h, score, cat = (int(float(v)) for v in p[:6])
        if score == 0 or cat == 0 or cat == 11 or w <= 0 or h <= 0:
            continue
        boxes.append((cat - 1, float(x), float(y), float(x + w), float(y + h)))
    return boxes


def decode_targets(flat, acounts):
    """flat [Ntot] -> [(scale, cx, cy, w, h, cid)] in normalized 448-image coords."""
    out, off = [], 0
    for s, g in enumerate(FPN_GRIDS):
        A = acounts[s]
        n = A * PER_ANCHOR * g * g
        t = flat[off:off + n].reshape(A * PER_ANCHOR, g, g)
        off += n
        for a in range(A):
            base = a * PER_ANCHOR
            obj = t[base + 4]
            for ci, cj in zip(*np.nonzero(obj > 0.5)):
                tx, ty = t[base + 0, ci, cj], t[base + 1, ci, cj]
                w, h = t[base + 2, ci, cj], t[base + 3, ci, cj]
                cid = int(np.argmax(t[base + 5: base + 5 + NUM_CLASSES, ci, cj]))
                out.append((s, (cj + tx) / g, (ci + ty) / g, w, h, cid))
    assert off == flat.size, f"decoded {off} of {flat.size} floats"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="records to render")
    ap.add_argument("--out", default="../figures/visdrone_bin_check")
    args = ap.parse_args()

    if not BIN.exists():
        sys.exit(f"missing {BIN}")
    acounts = load_anchor_counts()
    ntot = sum(a * PER_ANCHOR * g * g for a, g in zip(acounts, FPN_GRIDS))
    img_bytes = 3 * IMG_SIZE * IMG_SIZE
    rec = img_bytes + ntot * 4
    print(f"anchors/scale {acounts}, Ntot={ntot}, record={rec} bytes")

    stems = sorted(p.stem for p in (SRC / "images").glob("*.jpg"))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    all_wpx, all_hpx, all_scale = [], [], []
    with open(BIN, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        print(f"{count} records in {BIN}\n")
        # records are written in sorted-glob order, skipping images whose
        # annotation is missing/empty -- on this split nothing is skipped
        # (6471 images, 6471 records), so record i == stems[i].
        assert count == len(stems), (
            f"{count} records vs {len(stems)} images -- record<->stem mapping "
            "is not 1:1, the overlay below would compare the wrong pair")

        for i in range(args.n):
            f.seek(4 + i * rec)
            raw = f.read(rec)
            chw = np.frombuffer(raw[:img_bytes], dtype=np.uint8).reshape(3, IMG_SIZE, IMG_SIZE)
            flat = np.frombuffer(raw[img_bytes:], dtype=np.float32)
            enc = decode_targets(flat, acounts)

            stem = stems[i]
            ann = parse_ann(SRC / "annotations" / f"{stem}.txt")
            with Image.open(SRC / "images" / f"{stem}.jpg") as im:
                iw, ih = im.size

            im448 = Image.fromarray(chw.transpose(1, 2, 0).copy())
            d = ImageDraw.Draw(im448)
            # control: raw annotations pushed through the same squash resize
            for cid, x0, y0, x1, y1 in ann:
                d.rectangle([x0 / iw * IMG_SIZE, y0 / ih * IMG_SIZE,
                             x1 / iw * IMG_SIZE, y1 / ih * IMG_SIZE],
                            outline=(0, 255, 0), width=1)
                all_wpx.append((x1 - x0) / iw * IMG_SIZE)
                all_hpx.append((y1 - y0) / ih * IMG_SIZE)
            # what the trainer is actually taught
            for s, cx, cy, w, h, cid in enc:
                d.rectangle([(cx - w / 2) * IMG_SIZE, (cy - h / 2) * IMG_SIZE,
                             (cx + w / 2) * IMG_SIZE, (cy + h / 2) * IMG_SIZE],
                            outline=(255, 0, 0), width=1)
                all_scale.append(s)

            png = outdir / f"{i:02d}_{stem}.png"
            im448.resize((896, 896), Image.NEAREST).save(png)
            print(f"{png.name}: {iw}x{ih} -> 448  |  {len(ann)} annotated, "
                  f"{len(enc)} encoded ({100*len(enc)/max(len(ann),1):.0f}%)")

    w, h = np.array(all_wpx), np.array(all_hpx)
    print(f"\nobject size in FED pixels (n={w.size}, over the {args.n} rendered images):")
    for nm, v in (("width", w), ("height", h)):
        print(f"  {nm:<7} p10 {np.percentile(v,10):5.1f}  median {np.median(v):5.1f}  "
              f"p90 {np.percentile(v,90):5.1f}  max {v.max():6.1f}")
    tiny = ((w < 8) & (h < 8)).mean()
    print(f"  {100*tiny:.1f}% of objects are smaller than ONE 8px P3 cell")
    sc = np.array(all_scale)
    print(f"  encoded to P3/P4/P5: {(sc==0).sum()} / {(sc==1).sum()} / {(sc==2).sum()}")
    print(f"\ngreen = raw annotation through the squash resize (control)")
    print(f"red   = target actually encoded in train.bin")
    print(f"they must coincide; a systematic offset between them is the bug.")


if __name__ == "__main__":
    main()
