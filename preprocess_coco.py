#!/usr/bin/env python3
"""Pre-process MS-COCO 2017 (detection) → the YOLO detection .bin formats.

Produces the SAME on-disk record formats as preprocess_visdrone.py, so the
existing Lean FFI loaders, the YOLO codegen, and the scoring scripts work
unchanged. Three formats, selected by flag:

  (default)   single-grid YOLOv1   perCell = 2*5 + C     [lean_f32_load_voc_dims]
  --anchors   anchor, one grid     perCell = A*(5 + C)   [lean_f32_load_voc_anchor]
  --fpn DIR   FPN multi-scale      Ntot = Σ_s A_s(5+C)g_s² [lean_f32_load_voc_fpn]

CLASS-COUNT CEILING — read before choosing a format. Two of the three C loaders
hardcode the class count in their record stride, so they cannot read an 80-class
record no matter what this script writes:

  ffi/f32_helpers.c:426  load_voc_dims    `30 * gridH * gridW`  ⇒ C must be 20
  ffi/f32_helpers.c:~500 load_voc_anchor  `A * 15 * gH * gW`    ⇒ C must be 10
  ffi/f32_helpers.c:529  load_voc_fpn     takes `ntot`          ⇒ C is free

So **full 80-class COCO is FPN-only** on today's C code. This script enforces
that as a hard error rather than writing a file that would be silently
misparsed — a wrong stride reads garbage as float and trains on it.

Three class maps (--classes), one per loader capacity:

  all     the standard 80 COCO categories, contiguous 0..79. The remap is read
          from the JSON's own `categories` list sorted by id — NOT hardcoded —
          because COCO's file ids run 1..90 with ten gaps (12, 26, 29, 30, 45,
          66, 68, 69, 71, 83; verified against instances_val2017.json).
          Hardcoding `id - 1` is the classic COCO bug, and it is not a small
          one: measured against the real category list, **69 of the 80 classes
          come out mislabelled** — id 13 is 'stop sign' but `id - 1` calls it
          'parking meter', and everything above the first gap shifts from
          there. Nothing errors; the net just learns the wrong names.
          FPN format only.

  voc20   the 20 PASCAL-VOC categories, all of which exist in COCO under
          slightly different names (aeroplane→airplane, motorbike→motorcycle,
          sofa→couch, tvmonitor→tv, diningtable→dining table, pottedplant→
          potted plant). Indices follow VOC's canonical alphabetical order.
          This is the subset that fits the single-grid loader's perCell=30, so
          it is the cheap smoke path — and it is a standard subset, not one
          invented here.

  vdmap   COCO emitted directly into **VisDrone's 10-class index space**, so a
          COCO-pretrained detector transfers to VisDrone with its head intact.
          Six of the ten VisDrone classes have a COCO counterpart:
            person→pedestrian(0), bicycle→bicycle(2), car→car(3),
            truck→truck(5), bus→bus(8), motorcycle→motor(9)
          The other four (people, van, tricycle, awning-tricycle) have no COCO
          equivalent and stay empty — the one-hot has 10 slots and 4 are never
          set. That is deliberate and byte-compatible with the existing spec;
          it is NOT a claim that COCO covers VisDrone's taxonomy.

Two COCO evaluation subtleties are honored (or the numbers are silently wrong):
  * iscrowd == 1  -> crowd region, skip (COCO eval ignores these, exactly as
    VisDrone's score == 0 ignored regions are skipped).
  * degenerate / out-of-frame boxes are clipped to the image and dropped if
    they survive with non-positive extent.

Usage: python3 preprocess_coco.py <coco_dir> <out_dir> [flags]
  <coco_dir> must contain train2017/, val2017/ and
  annotations/instances_{train2017,val2017}.json (as download_coco.sh leaves it).

Flags: --size N --grid N --classes all|vdmap --anchors FILE --fpn DIR
       --fpn-thresh LO HI --train-only --val-only --limit N --check
Memory: instances_train2017.json is ~450 MB of JSON; json.load peaks around
4 GB. It is parsed once per split and released before image work begins.
"""
import os, sys, json, struct, glob
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow + numpy required.", file=sys.stderr); sys.exit(1)

# Defaults mirror preprocess_visdrone.py's WS-A baseline. The Lean loader
# derives the same geometry from the spec's imageH (stride-32 ⇒ grid = size/32),
# so keep size = grid*32.
IMG_SIZE, GRID_H, GRID_W = 224, 7, 7
NUM_BOXES = 2
NUM_CLASSES = 80          # set by --classes; PER_CELL/PER_ANCHOR derive from it
MAX_BBOXES = 56

# FPN multi-scale geometry — identical to preprocess_visdrone.py so one codegen
# serves both datasets. Thresholds are overridable (--fpn-thresh) because COCO
# objects are far larger than VisDrone's; the script reports the resulting
# per-scale histogram either way, so a lopsided split is visible, not assumed.
FPN_GRIDS = (56, 28, 14)          # P3 / P4 / P5 at 448px input
FPN_T_LO, FPN_T_HI = 24.0, 64.0   # max(w,h)px scale thresholds
# Measured on instances_val2017.json at these (VisDrone-tuned) thresholds, the
# COCO GT splits P3 17.2% / P4 28.5% / P5 54.3%. Worth recording because the
# obvious worry about a COCO→VisDrone transfer pretrain — that photographic
# objects all land on P5 and the small scales never train — does NOT hold:
# 46% of COCO GT lands on P3+P4. Keeping VisDrone's thresholds (so both builds
# share one assignment rule, which is what makes a head transferable) costs
# less than expected. Re-measure with --fpn-thresh before trusting it at other
# input sizes.

# COCO name -> VisDrone class index (see --classes vdmap in the docstring).
VDMAP = {"person": 0, "bicycle": 2, "car": 3, "truck": 5, "bus": 8,
         "motorcycle": 9}
VD_CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
              "tricycle", "awning-tricycle", "bus", "motor"]

# COCO name -> PASCAL-VOC class index, VOC's canonical alphabetical order.
VOC20 = ["airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
         "chair", "cow", "dining table", "dog", "horse", "motorcycle",
         "person", "potted plant", "sheep", "couch", "train", "tv"]
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]


def per_cell():
    return NUM_BOXES * 5 + NUM_CLASSES


def per_anchor():
    return 5 + NUM_CLASSES


def record_size():
    return (3*IMG_SIZE*IMG_SIZE + per_cell()*GRID_H*GRID_W*4
            + GRID_H*GRID_W*4 + 4 + MAX_BBOXES*20)


# ── annotation parsing ────────────────────────────────────────────────────────

def build_category_map(categories, mode):
    """Return (cat_id -> class index, class name list, num_classes).

    `all`   : contiguous 0..79 in ascending file-id order, read from the JSON.
    `vdmap` : VisDrone's 10-slot space, only the six COCO counterparts populated.
    """
    by_id = sorted(categories, key=lambda c: c["id"])
    if mode == "all":
        cmap = {c["id"]: i for i, c in enumerate(by_id)}
        names = [c["name"] for c in by_id]
        return cmap, names, len(names)
    if mode == "vdmap":
        cmap = {c["id"]: VDMAP[c["name"]] for c in by_id if c["name"] in VDMAP}
        missing = sorted(set(VDMAP) - {c["name"] for c in by_id})
        if missing:
            print(f"ERROR: --classes vdmap expects COCO names {missing} "
                  f"which are absent from this annotation file", file=sys.stderr)
            sys.exit(1)
        return cmap, list(VD_CLASSES), len(VD_CLASSES)
    if mode == "voc20":
        idx = {nm: i for i, nm in enumerate(VOC20)}
        cmap = {c["id"]: idx[c["name"]] for c in by_id if c["name"] in idx}
        missing = sorted(set(VOC20) - {c["name"] for c in by_id})
        if missing:
            print(f"ERROR: --classes voc20 expects COCO names {missing} "
                  f"which are absent from this annotation file", file=sys.stderr)
            sys.exit(1)
        return cmap, list(VOC_CLASSES), len(VOC_CLASSES)
    print(f"ERROR: unknown --classes {mode} (want all|voc20|vdmap)",
          file=sys.stderr)
    sys.exit(1)


def load_split_annotations(json_path, class_mode):
    """Parse instances_*.json → (per_image, names, num_classes).

    per_image is a list of (file_name, width, height, boxes) in **file_name
    order** — a stable, content-derived order, so the record index is
    reproducible across runs and machines. boxes are
    [(cid, xmin, ymin, xmax, ymax)] in pixel coords, already filtered.
    """
    print(f"  parsing {json_path} ...")
    with open(json_path, "r") as f:
        doc = json.load(f)
    cmap, names, ncls = build_category_map(doc["categories"], class_mode)

    imgs = {im["id"]: im for im in doc["images"]}
    boxes_by_img = {}
    n_ann = n_crowd = n_degen = n_offclass = n_clipped = 0
    for a in doc["annotations"]:
        n_ann += 1
        if a.get("iscrowd", 0) == 1:
            n_crowd += 1; continue
        cid = cmap.get(a["category_id"])
        if cid is None:                    # dropped by the class map (vdmap)
            n_offclass += 1; continue
        im = imgs.get(a["image_id"])
        if im is None:
            continue
        x, y, w, h = a["bbox"]
        x0, y0, x1, y1 = float(x), float(y), float(x + w), float(y + h)
        # COCO boxes can extend a fraction of a pixel past the frame; clip
        # rather than drop, then drop only what is degenerate afterwards.
        cx0 = min(max(x0, 0.0), float(im["width"]))
        cy0 = min(max(y0, 0.0), float(im["height"]))
        cx1 = min(max(x1, 0.0), float(im["width"]))
        cy1 = min(max(y1, 0.0), float(im["height"]))
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            n_clipped += 1
        if cx1 - cx0 <= 0 or cy1 - cy0 <= 0:
            n_degen += 1; continue
        boxes_by_img.setdefault(a["image_id"], []).append((cid, cx0, cy0, cx1, cy1))

    per_image = []
    n_empty = 0
    for im in sorted(doc["images"], key=lambda i: i["file_name"]):
        b = boxes_by_img.get(im["id"])
        if not b:
            n_empty += 1; continue         # no kept object: skipped, as VisDrone does
        per_image.append((im["file_name"], int(im["width"]), int(im["height"]), b))

    kept = sum(len(p[3]) for p in per_image)
    print(f"  {len(doc['images'])} images, {n_ann} annotations → "
          f"{len(per_image)} images / {kept} boxes "
          f"({kept/max(len(per_image),1):.1f}/img)")
    print(f"  dropped: {n_crowd} iscrowd, {n_degen} degenerate, "
          f"{n_offclass} off-class, {n_empty} images with no kept box"
          f"{f'; clipped {n_clipped} to frame' if n_clipped else ''}")
    return per_image, names, ncls


# ── target encodings (byte-identical layouts to preprocess_visdrone.py) ───────

def encode_targets(img_w, img_h, boxes):
    """Single-grid YOLOv1. One box per cell: a later box in the same cell
    overwrites an earlier one — the coarse-grid limitation."""
    target = np.zeros((per_cell(), GRID_H, GRID_W), dtype=np.float32)
    mask = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for (cid, xmin, ymin, xmax, ymax) in boxes:
        cx = (xmin + xmax) / 2.0 / img_w; cy = (ymin + ymax) / 2.0 / img_h
        w_rel = (xmax - xmin) / img_w;    h_rel = (ymax - ymin) / img_h
        cj = min(int(cx * GRID_W), GRID_W - 1); ci = min(int(cy * GRID_H), GRID_H - 1)
        target[0, ci, cj] = cx * GRID_W - cj
        target[1, ci, cj] = cy * GRID_H - ci
        target[2, ci, cj] = w_rel
        target[3, ci, cj] = h_rel
        target[4, ci, cj] = 1.0
        target[NUM_BOXES*5:, ci, cj] = 0.0
        target[NUM_BOXES*5 + cid, ci, cj] = 1.0
        mask[ci, cj] = 1.0
    return target, mask


def load_anchors(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        w, h = ln.split()
        rows.append((float(w), float(h)))
    return np.array(rows, dtype=np.float32)


def best_anchor(w_rel, h_rel, anchors):
    """Index of the anchor with max wh-IoU (shape match, origin-aligned)."""
    iw = np.minimum(w_rel, anchors[:, 0])
    ih = np.minimum(h_rel, anchors[:, 1])
    inter = iw * ih
    union = w_rel * h_rel + anchors[:, 0] * anchors[:, 1] - inter
    return int(np.argmax(inter / (union + 1e-12)))


def encode_targets_anchor(img_w, img_h, boxes, anchors):
    """Return (target[A*(5+C),gH,gW], mask[A,gH,gW], n_slots_filled)."""
    A = len(anchors); PA = per_anchor()
    target = np.zeros((A * PA, GRID_H, GRID_W), dtype=np.float32)
    mask = np.zeros((A, GRID_H, GRID_W), dtype=np.float32)
    filled = set()
    for (cid, xmin, ymin, xmax, ymax) in boxes:
        cx = (xmin + xmax) / 2.0 / img_w; cy = (ymin + ymax) / 2.0 / img_h
        w_rel = (xmax - xmin) / img_w;    h_rel = (ymax - ymin) / img_h
        cj = min(int(cx * GRID_W), GRID_W - 1); ci = min(int(cy * GRID_H), GRID_H - 1)
        a = best_anchor(w_rel, h_rel, anchors)
        base = a * PA
        target[base + 0, ci, cj] = cx * GRID_W - cj
        target[base + 1, ci, cj] = cy * GRID_H - ci
        target[base + 2, ci, cj] = w_rel
        target[base + 3, ci, cj] = h_rel
        target[base + 4, ci, cj] = 1.0
        target[base + 5: base + 5 + NUM_CLASSES, ci, cj] = 0.0
        target[base + 5 + cid, ci, cj] = 1.0
        mask[a, ci, cj] = 1.0
        filled.add((a, ci, cj))
    return target, mask, len(filled)


def fpn_scale_of(w_rel, h_rel, input_px):
    m = max(w_rel, h_rel) * input_px
    return 0 if m < FPN_T_LO else (1 if m < FPN_T_HI else 2)


def encode_targets_fpn(img_w, img_h, boxes, anchors_per_scale, input_px,
                       scale_hist=None):
    """Return (targets[3], masks[3], n_slots). Each GT is routed to ONE scale by
    size, then to that scale's best-shape anchor. Later GT overwrites earlier on
    a (scale,cell,anchor) collision. Mirrors preprocess_visdrone.py exactly."""
    PA = per_anchor()
    tgts, msks = [], []
    for s, g in enumerate(FPN_GRIDS):
        A = len(anchors_per_scale[s])
        tgts.append(np.zeros((A * PA, g, g), dtype=np.float32))
        msks.append(np.zeros((A, g, g), dtype=np.float32))
    filled = set()
    for (cid, xmin, ymin, xmax, ymax) in boxes:
        cx = (xmin + xmax) / 2.0 / img_w; cy = (ymin + ymax) / 2.0 / img_h
        w_rel = (xmax - xmin) / img_w;    h_rel = (ymax - ymin) / img_h
        s = fpn_scale_of(w_rel, h_rel, input_px)
        if scale_hist is not None:
            scale_hist[s] += 1
        g = FPN_GRIDS[s]
        cj = min(int(cx * g), g - 1); ci = min(int(cy * g), g - 1)
        a = best_anchor(w_rel, h_rel, anchors_per_scale[s])
        base = a * PA
        t = tgts[s]
        t[base + 0, ci, cj] = cx * g - cj
        t[base + 1, ci, cj] = cy * g - ci
        t[base + 2, ci, cj] = w_rel
        t[base + 3, ci, cj] = h_rel
        t[base + 4, ci, cj] = 1.0
        t[base + 5: base + 5 + NUM_CLASSES, ci, cj] = 0.0
        t[base + 5 + cid, ci, cj] = 1.0
        msks[s][a, ci, cj] = 1.0
        filled.add((s, ci, cj, a))
    return tgts, msks, len(filled)


# ── raw-box blocks + the uncapped GT sidecar ─────────────────────────────────

def pack_raw_boxes(img_w, img_h, boxes):
    out = bytearray(MAX_BBOXES * 20)
    n = min(len(boxes), MAX_BBOXES)
    for i in range(n):
        cid, xmin, ymin, xmax, ymax = boxes[i]
        struct.pack_into("<i", out, i * 20, int(cid))
        struct.pack_into("<ffff", out, i * 20 + 4,
                         xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h)
    return n, bytes(out)


def pack_full_gt(img_w, img_h, boxes):
    """UNCAPPED GT, per-record: <i nb, then nb × (<i cid, <ffff x0 y0 x1 y1).
    `pack_raw_boxes` truncates at MAX_BBOXES=56; scoring mAP against a truncated
    GT is not COCO protocol, so the scorer reads GT from this sidecar."""
    out = bytearray()
    out += struct.pack("<i", len(boxes))
    for cid, xmin, ymin, xmax, ymax in boxes:
        out += struct.pack("<i", int(cid))
        out += struct.pack("<ffff",
                           xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h)
    return bytes(out)


def full_gt_path(out_path):
    return str(Path(out_path).with_suffix(".full_gt.bin"))


def open_image(images_dir, file_name, jw, jh):
    """Decode, verify the JSON's dims against the file, return (PIL, w, h).
    The dims mismatch is a real (if rare) corruption mode and it would silently
    scale every box on that image, so it is counted, not assumed away."""
    img = Image.open(Path(images_dir) / file_name)
    w, h = img.size
    mismatch = (w != jw or h != jh)
    return img, w, h, mismatch


# ── writers ──────────────────────────────────────────────────────────────────

def process_split(per_image, images_dir, out_path):
    """Single-grid format + the uncapped GT sidecar, written record-for-record
    in ONE loop so logits row k, bin record k and sidecar record k are the same
    image. Any image this loop skips is skipped in both files.

    Returns the entries actually written, in record order — the checkers verify
    against THAT, not the input list, so a mid-loop skip cannot silently shift
    the comparison by one."""
    gt_path = full_gt_path(out_path)
    written = skipped = dimbad = 0
    total_boxes = capped_boxes = 0
    kept = []
    with open(out_path, "wb") as f, open(gt_path, "wb") as g:
        f.write(struct.pack("<I", 0))
        g.write(struct.pack("<I", 0))
        for (fn, jw, jh, boxes) in per_image:
            try:
                img, iw, ih, bad = open_image(images_dir, fn, jw, jh)
                dimbad += bad
                target, mask = encode_targets(iw, ih, boxes)
                nb, blk = pack_raw_boxes(iw, ih, boxes)
                chw = np.asarray(img.convert("RGB").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                    dtype=np.uint8).transpose(2, 0, 1).copy()
                f.write(chw.tobytes()); f.write(target.tobytes()); f.write(mask.tobytes())
                f.write(struct.pack("<i", nb)); f.write(blk)
                g.write(pack_full_gt(iw, ih, boxes))
                written += 1; kept.append((fn, jw, jh, boxes))
                total_boxes += len(boxes); capped_boxes += nb
            except Exception as e:
                print(f"  skip {fn}: {e}", file=sys.stderr); skipped += 1
        f.seek(0); f.write(struct.pack("<I", written))
        g.seek(0); g.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path) / 1024 / 1024
    dropped = total_boxes - capped_boxes
    print(f"  wrote {out_path}: {written} records ({skipped} skipped), "
          f"{mb:.0f} MB | {total_boxes} boxes ({total_boxes/max(written,1):.1f}/img)")
    print(f"  wrote {gt_path}: full uncapped GT ({total_boxes} boxes); "
          f"training record keeps {capped_boxes} ({dropped} dropped by "
          f"MAX_BBOXES={MAX_BBOXES}, {100.0*dropped/max(total_boxes,1):.1f}%); "
          f"<={GRID_H*GRID_W} survive the {GRID_H}x{GRID_W} target per image")
    if dimbad:
        print(f"  WARN: {dimbad} images whose decoded size != the JSON size",
              file=sys.stderr)
    return kept


def process_split_anchor(per_image, images_dir, out_path, anchors):
    A = len(anchors)
    written = skipped = 0
    total_boxes = total_slots = 0
    kept = []
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0))
        for (fn, jw, jh, boxes) in per_image:
            try:
                img, iw, ih, _bad = open_image(images_dir, fn, jw, jh)
                target, mask, nslots = encode_targets_anchor(iw, ih, boxes, anchors)
                nb, blk = pack_raw_boxes(iw, ih, boxes)
                chw = np.asarray(img.convert("RGB").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                    dtype=np.uint8).transpose(2, 0, 1).copy()
                f.write(chw.tobytes()); f.write(target.tobytes()); f.write(mask.tobytes())
                f.write(struct.pack("<i", nb)); f.write(blk)
                written += 1; kept.append((fn, jw, jh, boxes))
                total_boxes += len(boxes); total_slots += nslots
            except Exception as e:
                print(f"  skip {fn}: {e}", file=sys.stderr); skipped += 1
        f.seek(0); f.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path) / 1024 / 1024
    cover = 100.0 * total_slots / max(total_boxes, 1)
    print(f"  wrote {out_path}: {written} records ({skipped} skipped), {mb:.0f} MB | "
          f"A={A}, {total_boxes} GT boxes → {total_slots} anchor slots "
          f"({cover:.1f}% encoded)")
    return kept


def process_split_fpn(per_image, images_dir, out_path, anchors_per_scale,
                      input_px):
    """Image u8 + the flat [P3|P4|P5] target only. No mask/boxes on disk: the
    loss derives masks from the obj channels, and eval GT comes from the
    single-box val.bin + its sidecar."""
    written = skipped = 0
    total_boxes = total_slots = 0
    scale_hist = [0, 0, 0]
    kept = []
    ntot = sum(len(anchors_per_scale[s]) * per_anchor() * g * g
               for s, g in enumerate(FPN_GRIDS))
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", 0))
        for (fn, jw, jh, boxes) in per_image:
            try:
                img, iw, ih, _bad = open_image(images_dir, fn, jw, jh)
                tgts, _msks, nslots = encode_targets_fpn(
                    iw, ih, boxes, anchors_per_scale, input_px, scale_hist)
                flat = np.concatenate([t.reshape(-1) for t in tgts]).astype(np.float32)
                assert flat.size == ntot, f"flat {flat.size} != Ntot {ntot}"
                chw = np.asarray(img.convert("RGB").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                    dtype=np.uint8).transpose(2, 0, 1).copy()
                f.write(chw.tobytes()); f.write(flat.tobytes())
                written += 1; kept.append((fn, jw, jh, boxes))
                total_boxes += len(boxes); total_slots += nslots
            except Exception as e:
                print(f"  skip {fn}: {e}", file=sys.stderr); skipped += 1
        f.seek(0); f.write(struct.pack("<I", written))
    mb = os.path.getsize(out_path) / 1024 / 1024
    cover = 100.0 * total_slots / max(total_boxes, 1)
    print(f"  wrote {out_path}: {written} records ({skipped} skipped), {mb:.0f} MB | "
          f"Ntot={ntot}, {total_boxes} GT boxes → {total_slots} multi-scale slots "
          f"({cover:.1f}% encoded)")
    tot = max(sum(scale_hist), 1)
    print(f"  scale split (thresh {FPN_T_LO:.0f}/{FPN_T_HI:.0f}px @ {input_px}): "
          f"P3 {100.0*scale_hist[0]/tot:.1f}%  P4 {100.0*scale_hist[1]/tot:.1f}%  "
          f"P5 {100.0*scale_hist[2]/tot:.1f}%")
    return kept


# ── known-answer check ───────────────────────────────────────────────────────

# The pairing guard. The failure mode it exists to catch is a record set that
# has drifted out of step with the image list — which looks exactly like a
# working file until mAP is mysteriously zero. Every recent detection bug in
# this repo was silent plumbing of that shape, not bad math, so each writer
# gets a reader that ties record k back to image k's annotations. Also checks
# the file length against the declared count (a stride error), and that the
# class one-hot is exactly one-hot wherever objectness is set (a remap error).

def _check_header(out_path, rec):
    size = os.path.getsize(out_path)
    with open(out_path, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
    if 4 + count * rec != size:
        print(f"  FAIL: size {size} != 4 + {count}*{rec} "
              f"(record stride disagrees with the writer)", file=sys.stderr)
        return None
    return count


def check_single(out_path, per_image, images_dir, n=8):
    rec = record_size()
    count = _check_header(out_path, rec)
    if count is None:
        return False
    print(f"  checking {out_path} (first {min(n,count)} of {count} records) ...")
    pix = 3 * IMG_SIZE * IMG_SIZE
    tsz = per_cell() * GRID_H * GRID_W
    bad = 0
    with open(out_path, "rb") as f:
        for k in range(min(n, count)):
            f.seek(4 + k * rec)
            buf = f.read(rec)
            target = np.frombuffer(buf, dtype=np.float32, count=tsz,
                                   offset=pix).reshape(per_cell(), GRID_H, GRID_W)
            mask = np.frombuffer(buf, dtype=np.float32, count=GRID_H*GRID_W,
                                 offset=pix + tsz*4).reshape(GRID_H, GRID_W)
            fn, _jw, _jh, boxes = per_image[k]
            iw, ih = Image.open(Path(images_dir) / fn).size
            for (_cid, x0, y0, x1, y1) in boxes:
                cx = (x0 + x1) / 2.0 / iw; cy = (y0 + y1) / 2.0 / ih
                cj = min(int(cx * GRID_W), GRID_W - 1)
                ci = min(int(cy * GRID_H), GRID_H - 1)
                if mask[ci, cj] != 1.0 or target[4, ci, cj] != 1.0:
                    print(f"  FAIL rec {k} ({fn}): box at cell ({ci},{cj}) "
                          f"not marked", file=sys.stderr)
                    bad += 1; break
            for ci in range(GRID_H):
                for cj in range(GRID_W):
                    if mask[ci, cj] == 1.0:
                        oh = target[NUM_BOXES*5:, ci, cj]
                        if abs(oh.sum() - 1.0) > 1e-6:
                            print(f"  FAIL rec {k} ({fn}): class one-hot sums to "
                                  f"{oh.sum()} at ({ci},{cj})", file=sys.stderr)
                            bad += 1
    if bad:
        return False
    print(f"  OK: {count} records, {rec} B/record, first {min(n,count)} tie to "
          f"their source annotations")
    return True


def check_anchor(out_path, per_image, images_dir, anchors, n=8):
    A = len(anchors); PA = per_anchor()
    tsz = A * PA * GRID_H * GRID_W
    msz = A * GRID_H * GRID_W
    pix = 3 * IMG_SIZE * IMG_SIZE
    rec = pix + tsz*4 + msz*4 + 4 + MAX_BBOXES*20
    count = _check_header(out_path, rec)
    if count is None:
        return False
    print(f"  checking {out_path} (first {min(n,count)} of {count} records) ...")
    bad = 0
    with open(out_path, "rb") as f:
        for k in range(min(n, count)):
            f.seek(4 + k * rec)
            buf = f.read(rec)
            target = np.frombuffer(buf, dtype=np.float32, count=tsz,
                                   offset=pix).reshape(A * PA, GRID_H, GRID_W)
            mask = np.frombuffer(buf, dtype=np.float32, count=msz,
                                 offset=pix + tsz*4).reshape(A, GRID_H, GRID_W)
            fn, _jw, _jh, boxes = per_image[k]
            iw, ih = Image.open(Path(images_dir) / fn).size
            for (_cid, x0, y0, x1, y1) in boxes:
                cx = (x0 + x1) / 2.0 / iw; cy = (y0 + y1) / 2.0 / ih
                wr = (x1 - x0) / iw; hr = (y1 - y0) / ih
                cj = min(int(cx * GRID_W), GRID_W - 1)
                ci = min(int(cy * GRID_H), GRID_H - 1)
                a = best_anchor(wr, hr, anchors)
                if mask[a, ci, cj] != 1.0 or target[a*PA + 4, ci, cj] != 1.0:
                    print(f"  FAIL rec {k} ({fn}): box at anchor {a} cell "
                          f"({ci},{cj}) not marked", file=sys.stderr)
                    bad += 1; break
            # mask must equal the objectness channels exactly
            obj = target[4::PA, :, :]
            if not np.array_equal(obj, mask):
                print(f"  FAIL rec {k} ({fn}): mask != objectness channels",
                      file=sys.stderr)
                bad += 1
    if bad:
        return False
    print(f"  OK: {count} records, {rec} B/record, A={A}, first "
          f"{min(n,count)} tie to their source annotations")
    return True


def check_fpn(out_path, per_image, images_dir, anchors_per_scale, input_px, n=8):
    PA = per_anchor()
    sizes = [len(anchors_per_scale[s]) * PA * g * g
             for s, g in enumerate(FPN_GRIDS)]
    ntot = sum(sizes)
    pix = 3 * IMG_SIZE * IMG_SIZE
    rec = pix + ntot * 4
    count = _check_header(out_path, rec)
    if count is None:
        return False
    print(f"  checking {out_path} (first {min(n,count)} of {count} records) ...")
    bad = 0
    with open(out_path, "rb") as f:
        for k in range(min(n, count)):
            f.seek(4 + k * rec)
            buf = f.read(rec)
            flat = np.frombuffer(buf, dtype=np.float32, count=ntot, offset=pix)
            blocks, off = [], 0
            for s, g in enumerate(FPN_GRIDS):
                A = len(anchors_per_scale[s])
                blocks.append(flat[off:off+sizes[s]].reshape(A * PA, g, g))
                off += sizes[s]
            fn, _jw, _jh, boxes = per_image[k]
            iw, ih = Image.open(Path(images_dir) / fn).size
            for (_cid, x0, y0, x1, y1) in boxes:
                cx = (x0 + x1) / 2.0 / iw; cy = (y0 + y1) / 2.0 / ih
                wr = (x1 - x0) / iw; hr = (y1 - y0) / ih
                s = fpn_scale_of(wr, hr, input_px)
                g = FPN_GRIDS[s]
                cj = min(int(cx * g), g - 1); ci = min(int(cy * g), g - 1)
                a = best_anchor(wr, hr, anchors_per_scale[s])
                if blocks[s][a*PA + 4, ci, cj] != 1.0:
                    print(f"  FAIL rec {k} ({fn}): box at P{s+3} anchor {a} "
                          f"cell ({ci},{cj}) not marked", file=sys.stderr)
                    bad += 1; break
            for s in range(3):
                obj = blocks[s][4::PA, :, :]
                for a, ci, cj in zip(*np.nonzero(obj)):
                    oh = blocks[s][a*PA + 5: a*PA + 5 + NUM_CLASSES, ci, cj]
                    if abs(oh.sum() - 1.0) > 1e-6:
                        print(f"  FAIL rec {k} ({fn}): class one-hot sums to "
                              f"{oh.sum()} at P{s+3} ({a},{ci},{cj})",
                              file=sys.stderr)
                        bad += 1; break
    if bad:
        return False
    print(f"  OK: {count} records, {rec} B/record, Ntot={ntot}, first "
          f"{min(n,count)} tie to their source annotations")
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    global IMG_SIZE, GRID_H, GRID_W, NUM_CLASSES, FPN_T_LO, FPN_T_HI
    argv = sys.argv[1:]
    size = 224; grid = 7; anchors_path = None; fpn_dir = None
    class_mode = "all"; limit = None; do_check = False
    splits = ("train", "val")
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] == "--size":
            size = int(argv[i+1]); i += 2
        elif argv[i] == "--grid":
            grid = int(argv[i+1]); i += 2
        elif argv[i] == "--classes":
            class_mode = argv[i+1]; i += 2
        elif argv[i] == "--anchors":
            anchors_path = argv[i+1]; i += 2
        elif argv[i] == "--fpn":
            fpn_dir = argv[i+1]; i += 2
        elif argv[i] == "--fpn-thresh":
            FPN_T_LO = float(argv[i+1]); FPN_T_HI = float(argv[i+2]); i += 3
        elif argv[i] == "--limit":
            limit = int(argv[i+1]); i += 2      # first N images: a smoke build
        elif argv[i] == "--check":
            do_check = True; i += 1
        elif argv[i] == "--val-only":
            splits = ("val",); i += 1
        elif argv[i] == "--train-only":
            splits = ("train",); i += 1
        else:
            pos.append(argv[i]); i += 1
    if len(pos) != 2:
        print(__doc__); sys.exit(1)
    coco_dir, out_dir = pos[0], pos[1]

    IMG_SIZE = size; GRID_H = grid; GRID_W = grid
    if IMG_SIZE % GRID_H != 0:
        print(f"WARN: grid {GRID_H} does not evenly divide size {IMG_SIZE} "
              f"(stride = size/grid must be integer)", file=sys.stderr)

    ann_dir = Path(coco_dir) / "annotations"
    dirs = {"train": (Path(coco_dir) / "train2017",
                      ann_dir / "instances_train2017.json"),
            "val":   (Path(coco_dir) / "val2017",
                      ann_dir / "instances_val2017.json")}
    for s in splits:
        d, j = dirs[s]
        if not d.exists() or not j.exists():
            print(f"ERROR: expected {d} and {j}", file=sys.stderr); sys.exit(1)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Peek at one categories list to fix NUM_CLASSES before any geometry is
    # printed — the class count is what decides which formats are even legal.
    _d, first_json = dirs[splits[0]]
    with open(first_json, "r") as f:
        cats = json.load(f)["categories"]
    _cmap, names, ncls = build_category_map(cats, class_mode)
    NUM_CLASSES = ncls
    print(f"class map '{class_mode}': {ncls} classes "
          f"({names[0]} … {names[-1]})")

    # ── the class-count ceiling, enforced (see the module docstring) ──────────
    if fpn_dir is None and anchors_path is None and NUM_CLASSES != 20:
        print(f"ERROR: the single-grid format is read by lean_f32_load_voc_dims, "
              f"which hardcodes perCell=30 (2*5+20). With {NUM_CLASSES} classes "
              f"this file would be misparsed.\n"
              f"       Use --classes voc20 for this format, or --fpn for the "
              f"class-agnostic loader.", file=sys.stderr)
        sys.exit(1)
    if anchors_path is not None and NUM_CLASSES != 10:
        print(f"ERROR: the anchor format is read by lean_f32_load_voc_anchor, "
              f"which hardcodes perAnchor=15 (5+10). With {NUM_CLASSES} classes "
              f"this file would be misparsed. Use --classes vdmap.",
              file=sys.stderr)
        sys.exit(1)

    if fpn_dir:
        aps = [load_anchors(os.path.join(fpn_dir, f"anchors_fpn_{p}.txt"))
               for p in ("p3", "p4", "p5")]
        if IMG_SIZE != 448:
            print(f"WARN: FPN grids {FPN_GRIDS} assume 448px input, got {IMG_SIZE}",
                  file=sys.stderr)
        ntot = sum(len(aps[s]) * per_anchor() * g * g for s, g in enumerate(FPN_GRIDS))
        rec = 3*IMG_SIZE*IMG_SIZE + ntot*4
        print(f"FPN encoding: A/scale={[len(a) for a in aps]}, grids={FPN_GRIDS}, "
              f"C={NUM_CLASSES}, perAnchor={per_anchor()}, Ntot={ntot}, "
              f"{rec} bytes/record")
    elif anchors_path:
        anchors = load_anchors(anchors_path)
        rec = (3*IMG_SIZE*IMG_SIZE + len(anchors)*per_anchor()*GRID_H*GRID_W*4
               + len(anchors)*GRID_H*GRID_W*4 + 4 + MAX_BBOXES*20)
        print(f"anchor encoding: {len(anchors)} anchors, "
              f"perCell={len(anchors)*per_anchor()}, {rec} bytes/record")
    else:
        print(f"single-grid encoding at {IMG_SIZE}px / {GRID_H}×{GRID_W} "
              f"({record_size()} bytes/record)")

    for s in splits:
        images_dir, json_path = dirs[s]
        print(f"[{s}]")
        per_image, _names, _n = load_split_annotations(json_path, class_mode)
        out_path = os.path.join(out_dir, f"{s}.bin")
        if limit:
            per_image = per_image[:limit]
        if fpn_dir:
            kept = process_split_fpn(per_image, images_dir, out_path, aps, IMG_SIZE)
            ok = (not do_check) or check_fpn(out_path, kept, images_dir,
                                             aps, IMG_SIZE)
        elif anchors_path:
            kept = process_split_anchor(per_image, images_dir, out_path, anchors)
            ok = (not do_check) or check_anchor(out_path, kept, images_dir, anchors)
        else:
            kept = process_split(per_image, images_dir, out_path)
            ok = (not do_check) or check_single(out_path, kept, images_dir)
        if not ok:
            sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
