#!/usr/bin/env python3
"""Build the PlantVillage / PlantDoc sets — planning/plant_lab_to_field_demo.md §2.

PlantVillage (Mohanty, Hughes & Salathé 2016; github spMohanty/PlantVillage-Dataset,
CC BY-SA 3.0 per its dataset card) is 54,305 lab photographs of single picked leaves,
256×256, 38 classes, with a `segmented` twin (leaf on black) and a leaf grouping
(`leaf_grouping/leaf-map.json`: most leaves were photographed four times). PlantDoc
(Singh et al. 2020; github pratikkayal/PlantDoc-Dataset, CC BY 4.0) is 2,578 field
photographs of the same crops in 28 classes that all map into PlantVillage's 38.

Writes the Imagenette record format (`count` u32 LE, then `label` u8 + 3×S×S u8
channel-planar RGB) that `F32.loadImagenetteSized` reads — S = 256 for training parts
(random 224 crop at train time), S = 224 (centre crop) for evaluation parts:

  pv_{train,val,test}.bin          PlantVillage colour, RANDOM split, 80/10/10 per class
  pvg_{train,val,test}.bin         GROUPED split: test = the maintainers' splits/color_test.txt
                                   (10,709), val = one tenth of all images carved from their
                                   train by leaf id, train = the rest
  pv{,g}_test_seg.bin              the test images from `segmented` (leaf on black)
  pv{,g}_test_bg.bin               the complement: leaf region filled with the image's median
                                   background colour (the silhouette keeps the leaf's shape)
  pv{,g}_test_mask.npy             leaf mask of the test part at 224×224 (bool) and 7×7 (mean)
  pv{,g}_train_comp.bin            [--composites] every training leaf pasted onto a random 256
                                   crop of a random Imagenette training image, same order/labels
  pv{,g}_train_aug.bin             [--aug] one photometric+geometric variant per training image
  pd_all.bin, pd_train.bin, pd_test.bin, pd_fold{0..4}_{train,test}.bin
                                   PlantDoc with PlantVillage label ids; five stratified folds
                                   over all 2,578 so every field image is scored once held-out
  meta_plant.npz                   per PlantVillage image: class, lab code, suffix number, leaf
                                   id (or -1), random/grouped part; per PlantDoc image: class,
                                   fold, official split; the leak audit of each PV test part
                                   (nearest train image at 16×16, |Δ| there and at 64×64);
                                   the class map; classes_pv.txt / classes_pd.txt beside it
  manifest_plant.json              census, leaf-map coverage, chain statistic, audit summary

  .venv/bin/python preprocess_plant.py data/plant data/plant [--stats] [--composites] [--aug]
                   [--seed=0] [--imagenette=data/imagenette/imagenette2-320/train]
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_arasl import chains_of, leak_audit, THUMB, LEAK_THR  # noqa: E402  (the ArASL instrument, unchanged)

N_PV, N_CLASSES = 54305, 38
S_TRAIN, S_EVAL = 256, 224
FRACS = (0.8, 0.9)
PD_TO_PV = {
    "Apple Scab Leaf": "Apple___Apple_scab", "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust", "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot", "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight", "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy", "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight", "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy", "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy", "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot", "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot", "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf": "Grape___healthy", "grape leaf black rot": "Grape___Black_rot",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}


# ── file naming ──
def suffix_key(fname):
    """The maintainers' leaf-map key: the part after `___` (or the whole stem), `_final_masked`
    removed, extension removed, lower-cased — `plant_village.py`'s rule."""
    stem = fname.replace("_final_masked", "")
    if "___" in stem:
        stem = stem.split("___")[-1]
    stem = stem.split("copy")[0]
    stem = re.sub(r"\.(jpg|jpeg|png)$", "", stem, flags=re.I)
    return stem.strip().lower()


def lab_and_num(key):
    m = re.match(r"^(.*?)[ _]+(\d+)$", key)
    return (m.group(1), int(m.group(2))) if m else (key, -1)


# ── records ──
class RecordWriter:
    """Imagenette format: u32 count, then u8 label + planar RGB u8 per record."""

    def __init__(self, path, size):
        self.f = open(path, "wb")
        self.f.write(np.uint32(0).tobytes())
        self.n, self.size = 0, size

    def add(self, label, img):
        a = np.asarray(img, dtype=np.uint8)
        assert a.shape == (self.size, self.size, 3), a.shape
        self.f.write(bytes([label]))
        self.f.write(np.ascontiguousarray(a.transpose(2, 0, 1)).tobytes())
        self.n += 1

    def close(self):
        self.f.seek(0)
        self.f.write(np.uint32(self.n).tobytes())
        self.f.close()
        return self.n


def center(im, size):
    """Shorter side → 256, centre crop `size` (for PlantVillage's 256² this is a plain crop)."""
    im = im.convert("RGB")
    w, h = im.size
    if min(w, h) != 256:
        s = 256 / min(w, h)
        im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.LANCZOS)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return im.crop((l, t, l + size, t + size))


def write_part(path, size, items, workers=16):
    """items: list of (label, loader) where loader() → PIL image at 256 or larger."""
    w = RecordWriter(path, size)
    with ThreadPoolExecutor(workers) as ex:
        for lbl, im in zip([l for l, _ in items], ex.map(lambda t: center(t[1](), size), items, chunksize=64)):
            w.add(lbl, im)
    return w.close()


def shapley_parts(args):
    """The two-player Shapley's counterfactuals for each test part: `_test_leaf` = the leaf on the
    image's median background colour, `_test_none` = that flat colour alone. With `_test` and
    `_test_bg` (leaf region filled the same way) the four share one fill, so
    φ_leaf + φ_bg = f(test) − f(none) exactly."""
    pv_root = os.path.join(args.src, "PlantVillage-Dataset", "raw")
    m = np.load(os.path.join(args.out, "meta_plant.npz"))
    files = [os.path.join(pv_root, f) for f in m["pv_file"]]
    labels = m["pv_label"]
    seg_index = {}
    for c in m["classes"]:
        for f in os.listdir(os.path.join(pv_root, "segmented", str(c))):
            seg_index[(str(c), suffix_key(f))] = os.path.join(pv_root, "segmented", str(c), f)
    for nm in ("pv", "pvg"):
        te = m[f"{nm}_test_index"]
        wl = RecordWriter(os.path.join(args.out, f"{nm}_test_leaf.bin"), S_EVAL)
        wn = RecordWriter(os.path.join(args.out, f"{nm}_test_none.bin"), S_EVAL)
        for j in te:
            c = str(m["classes"][labels[j]])
            col = np.asarray(center(Image.open(files[j]), S_EVAL))
            seg = seg_index.get((c, suffix_key(os.path.basename(files[j]))))
            mask = (np.asarray(center(Image.open(seg), S_EVAL)).sum(axis=2) > 40) if seg else np.zeros((S_EVAL, S_EVAL), dtype=bool)
            fill = (np.median(col[~mask], axis=0) if (~mask).any() else np.array([128, 128, 128])).astype(np.uint8)
            leaf = np.empty_like(col); leaf[:] = fill; leaf[mask] = col[mask]
            none = np.empty_like(col); none[:] = fill
            wl.add(int(labels[j]), leaf); wn.add(int(labels[j]), none)
        print(f"{nm}: wrote test_leaf + test_none ({wl.close()} / {wn.close()} records)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="data/plant (holds PlantVillage-Dataset/, PlantDoc-Dataset/, splits/)")
    ap.add_argument("out", help="data/plant")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--composites", action="store_true", help="write pv{,g}_train_comp.bin")
    ap.add_argument("--aug", action="store_true", help="write pv{,g}_train_aug.bin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imagenette", default="data/imagenette/imagenette2-320/train")
    ap.add_argument("--only-shapley-parts", action="store_true",
                    help="write pv{,g}_test_{leaf,none}.bin from an existing meta_plant.npz and exit")
    args = ap.parse_args()
    if args.only_shapley_parts:
        return shapley_parts(args)
    random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    pv_root = os.path.join(args.src, "PlantVillage-Dataset", "raw")
    pd_root = os.path.join(args.src, "PlantDoc-Dataset")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # ── PlantVillage census ──
    classes = sorted(os.listdir(os.path.join(pv_root, "color")))
    if len(classes) != N_CLASSES:
        sys.exit(f"{len(classes)} PlantVillage class folders, expected {N_CLASSES}")
    cid = {c: i for i, c in enumerate(classes)}
    seg_index = {}
    for c in classes:
        for f in os.listdir(os.path.join(pv_root, "segmented", c)):
            seg_index[(c, suffix_key(f))] = os.path.join(pv_root, "segmented", c, f)
    files, labels, keys, labs, nums, seg_paths = [], [], [], [], [], []
    for c in classes:
        for f in sorted(os.listdir(os.path.join(pv_root, "color", c))):
            k = suffix_key(f)
            files.append(os.path.join(pv_root, "color", c, f))
            labels.append(cid[c]); keys.append(k)
            lab, num = lab_and_num(k); labs.append(lab); nums.append(num)
            seg_paths.append(seg_index.get((c, k)))
    n = len(files)
    labels = np.asarray(labels, dtype=np.int32); nums = np.asarray(nums, dtype=np.int32)
    has_mask = np.asarray([p is not None for p in seg_paths])
    per_class = np.bincount(labels, minlength=N_CLASSES)
    n_seg = sum(len(os.listdir(os.path.join(pv_root, "segmented", c))) for c in classes)
    print(f"PlantVillage: {n} colour images (expected {N_PV}), {N_CLASSES} classes, per class {per_class.min()} "
          f"({classes[per_class.argmin()]}) to {per_class.max()} ({classes[per_class.argmax()]}); "
          f"segmented {n_seg} files, {int(has_mask.sum())} colour images have a mask twin ({time.time() - t0:.0f} s)")
    if n != N_PV:
        sys.exit("PlantVillage census mismatch — wrong mirror?")
    if args.stats:
        for i, c in enumerate(classes):
            sel = labels == i
            labcount = {}
            for j in np.flatnonzero(sel):
                labcount[labs[j]] = labcount.get(labs[j], 0) + 1
            miss = int((~has_mask[sel]).sum())
            print(f"  {c:52s} {per_class[i]:5d}  labs {dict(sorted(labcount.items(), key=lambda kv: -kv[1])[:3])}"
                  + (f"  ⚠ {miss} without mask twin" if miss else ""))

    # ── the leaf map ──
    with open(os.path.join(args.src, "PlantVillage-Dataset", "leaf_grouping", "leaf-map.json")) as f:
        leaf_map = json.load(f)
    leaf_id = np.full(n, -1, dtype=np.int64)
    leaf_names = {}
    for j in range(n):
        sugg = leaf_map.get(keys[j])
        if not sugg:
            continue
        pick = sugg[0] if len(sugg) == 1 else next((s for s in sugg if classes[labels[j]] in s), None)
        if pick is None:
            continue
        leaf_id[j] = leaf_names.setdefault(pick, len(leaf_names))
    covered = leaf_id >= 0
    sizes = np.bincount(leaf_id[covered])
    unc = {classes[i]: int((~covered[labels == i]).sum()) for i in range(N_CLASSES) if (~covered[labels == i]).any()}
    print(f"leaf map: {int(covered.sum())}/{n} images covered, {len(leaf_names)} leaves, images per leaf mode "
          f"{np.bincount(sizes).argmax()} (max {sizes.max()}); uncovered in {len(unc)} classes: "
          f"{dict(sorted(unc.items(), key=lambda kv: -kv[1])[:6])}…")
    # group id for the grouped split: the leaf, or a singleton (the maintainers' fallback rule)
    group = leaf_id.copy()
    nxt = len(leaf_names)
    for j in np.flatnonzero(~covered):
        group[j] = nxt; nxt += 1

    # ── the official split ──
    with open(os.path.join(args.src, "splits", "color_test.txt")) as f:
        official_test = {os.path.basename(l.strip()) for l in f if l.strip()}
    is_official_test = np.asarray([os.path.basename(p) in official_test for p in files])
    print(f"official split: {int(is_official_test.sum())} test files matched of {len(official_test)} listed")
    # does the official split respect the leaf groups?
    straddle = 0
    for g in np.unique(group[covered]):
        sel = group == g
        if is_official_test[sel].any() and (~is_official_test[sel]).any():
            straddle += 1
    print(f"  leaves straddling the official train/test line: {straddle} of {len(leaf_names)}")

    # ── splits: random (80/10/10 per class) and grouped (official test; val by leaf from train) ──
    part = {}
    p = np.zeros(n, dtype=np.int8)
    for i in range(N_CLASSES):
        sel = np.flatnonzero(labels == i)
        order = rng.permutation(len(sel))
        c0, c1 = int(round(FRACS[0] * len(sel))), int(round(FRACS[1] * len(sel)))
        p[sel[order[c0:c1]]] = 1
        p[sel[order[c1:]]] = 2
    part["pv"] = p
    p = np.zeros(n, dtype=np.int8)
    p[is_official_test] = 2
    for i in range(N_CLASSES):
        sel = np.flatnonzero((labels == i) & ~is_official_test)
        want = int(round(0.1 * (labels == i).sum()))
        groups = np.unique(group[sel])
        rng.shuffle(groups)
        taken, val_groups = 0, set()
        for g in groups:
            if taken >= want:
                break
            val_groups.add(g); taken += int((group[sel] == g).sum())
        p[sel[np.isin(group[sel], list(val_groups))]] = 1
    part["pvg"] = p
    for nm, pp in part.items():
        sz = np.bincount(pp, minlength=3)
        print(f"{nm:4s} split: train {sz[0]}  val {sz[1]}  test {sz[2]}")
    for g in np.unique(group[covered]):
        if len(set(part["pvg"][group == g])) > 1:
            sys.exit("grouped split: a leaf straddles two parts")

    # ── thumbnails once, for the audit and the chain statistic ──
    def grey_thumb(path):
        return np.asarray(Image.open(path).convert("L").resize((THUMB, THUMB), Image.BOX), dtype=np.uint8).reshape(-1)

    def grey64(path):
        return np.asarray(Image.open(path).convert("L").resize((64, 64), Image.BOX), dtype=np.uint8)

    t1 = time.time()
    with ThreadPoolExecutor(16) as ex:
        thumbs = np.stack(list(ex.map(grey_thumb, files, chunksize=256)))
        g64 = np.stack(list(ex.map(grey64, files, chunksize=256)))
    print(f"thumbnails ({time.time() - t1:.0f} s)")
    # chains along the suffix number within (class, lab): are consecutive numbers the same leaf?
    chain_rows = []
    for i in range(N_CLASSES):
        for lab in sorted(set(labs[j] for j in np.flatnonzero(labels == i))):
            sel = np.asarray([j for j in np.flatnonzero(labels == i) if labs[j] == lab and nums[j] >= 0])
            if len(sel) < 20:
                continue
            sel = sel[np.argsort(nums[sel])]
            ids, d = chains_of(g64[sel], LEAK_THR)
            same_leaf = float(np.mean(leaf_id[sel][1:] == leaf_id[sel][:-1])) if covered[sel].all() else None
            chain_rows.append(dict(cls=classes[i], lab=lab, n=int(len(sel)), median_consec=float(np.median(d)),
                                   frac_under=float((d < LEAK_THR).mean()), chains=int(ids[-1] + 1), same_leaf_consec=same_leaf))
    all_frac = np.average([r["frac_under"] for r in chain_rows], weights=[r["n"] for r in chain_rows])
    print(f"chains along suffix numbers: {100 * all_frac:.1f}% of consecutive pairs under {LEAK_THR:g} grey levels "
          f"(ArASL: 73.5%); median consecutive |Δ| {np.median([r['median_consec'] for r in chain_rows]):.1f}")
    if args.stats:
        for r in sorted(chain_rows, key=lambda r: -r["frac_under"])[:8]:
            print(f"  {r['cls'][:40]:40s} {r['lab']:14s} n={r['n']:5d} consec|Δ| {r['median_consec']:5.1f} under {100 * r['frac_under']:5.1f}%"
                  + (f"  same-leaf {100 * r['same_leaf_consec']:.0f}%" if r["same_leaf_consec"] is not None else ""))

    # ── leak audit, both splits ──
    audit = {}
    for nm in ("pv", "pvg"):
        pp = part[nm]
        tr, te = np.flatnonzero(pp == 0), np.flatnonzero(pp == 2)
        t1 = time.time()
        nn, nd = leak_audit(thumbs, tr, te)
        nd64 = np.abs(g64[te].reshape(len(te), -1).astype(np.int16) - g64[nn].reshape(len(te), -1).astype(np.int16)).mean(axis=1)
        same = labels[nn] == labels[te]
        same_leaf = (leaf_id[nn] == leaf_id[te]) & (leaf_id[te] >= 0)
        audit[nm] = dict(n_test=int(len(te)), frac_under=float((nd < LEAK_THR).mean()), frac_under_64=float((nd64 < LEAK_THR).mean()),
                         median_nn=float(np.median(nd)), frac_same_class_nn=float(same.mean()),
                         frac_same_leaf_nn=float(same_leaf.mean()), nn=nn, nd=nd.astype(np.float32), nd64=nd64.astype(np.float32))
        print(f"leak audit {nm:4s}: {100 * audit[nm]['frac_under']:5.1f}% of test images have a train image within {LEAK_THR:g} "
              f"grey levels at {THUMB}×{THUMB} ({100 * audit[nm]['frac_under_64']:.1f}% at 64×64; median nearest {np.median(nd):.1f}); "
              f"nearest is same class {100 * same.mean():.1f}%, same leaf {100 * same_leaf.mean():.1f}%  ({time.time() - t1:.0f} s)")
        if args.stats:
            per = {classes[i]: float((nd[labels[te] == i] < LEAK_THR).mean()) for i in range(N_CLASSES)}
            print("   leaked per class (top 8):", {k[:28]: round(100 * v, 1) for k, v in sorted(per.items(), key=lambda kv: -kv[1])[:8]})

    # ── PlantDoc ──
    pd_classes = sorted(os.listdir(os.path.join(pd_root, "train")))
    unmapped = [c for c in pd_classes if c not in PD_TO_PV]
    if unmapped:
        sys.exit(f"PlantDoc folders with no row in the class map: {unmapped}")
    targets = [PD_TO_PV[c] for c in pd_classes]
    if len(set(targets)) != len(targets):
        sys.exit("two PlantDoc folders map to one PlantVillage class")
    pd_files, pd_labels, pd_split = [], [], []
    for split_name, code in (("train", 0), ("test", 1)):
        for c in pd_classes:
            d = os.path.join(pd_root, split_name, c)
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                pd_files.append(os.path.join(d, f)); pd_labels.append(cid[PD_TO_PV[c]]); pd_split.append(code)
    pd_labels = np.asarray(pd_labels, dtype=np.int32); pd_split = np.asarray(pd_split, dtype=np.int8)
    pd_n = len(pd_files)
    print(f"PlantDoc: {pd_n} images ({int((pd_split == 0).sum())} train / {int((pd_split == 1).sum())} test), "
          f"{len(pd_classes)} classes all mapped; {N_CLASSES - len(set(targets))} PlantVillage classes have no twin")
    if args.stats:
        for c in pd_classes:
            print(f"  {c:38s} → {PD_TO_PV[c]:52s} {int((pd_labels == cid[PD_TO_PV[c]]).sum()):4d}")
    fold = np.zeros(pd_n, dtype=np.int8)
    for lbl in np.unique(pd_labels):
        sel = np.flatnonzero(pd_labels == lbl)
        rng.shuffle(sel)
        fold[sel] = np.arange(len(sel)) % 5

    # ── write ──
    load = lambda path: (lambda: Image.open(path))
    written = {}
    for nm in ("pv", "pvg"):
        pp = part[nm]
        for k, s in enumerate(("train", "val", "test")):
            idx = np.flatnonzero(pp == k)
            size = S_TRAIN if s == "train" else S_EVAL
            written[f"{nm}_{s}"] = write_part(os.path.join(args.out, f"{nm}_{s}.bin"), size,
                                              [(int(labels[j]), load(files[j])) for j in idx])
        # the test part from `segmented`, its background complement, and the mask
        te = np.flatnonzero(pp == 2)
        masks = np.zeros((len(te), S_EVAL, S_EVAL), dtype=bool)
        wseg = RecordWriter(os.path.join(args.out, f"{nm}_test_seg.bin"), S_EVAL)
        wbg = RecordWriter(os.path.join(args.out, f"{nm}_test_bg.bin"), S_EVAL)
        for i, j in enumerate(te):
            col = np.asarray(center(Image.open(files[j]), S_EVAL))
            if seg_paths[j] is None:
                seg = np.zeros_like(col); m = np.zeros((S_EVAL, S_EVAL), dtype=bool)
            else:
                seg = np.asarray(center(Image.open(seg_paths[j]), S_EVAL)); m = seg.sum(axis=2) > 40
            masks[i] = m
            bg = col.copy()
            fill = np.median(col[~m], axis=0) if (~m).any() else np.array([128, 128, 128])
            bg[m] = fill.astype(np.uint8)
            wseg.add(int(labels[j]), seg); wbg.add(int(labels[j]), bg)
        wseg.close(); wbg.close()
        np.save(os.path.join(args.out, f"{nm}_test_mask.npy"), masks)
        np.save(os.path.join(args.out, f"{nm}_test_mask7.npy"), masks.reshape(len(te), 7, 32, 7, 32).mean(axis=(2, 4)).astype(np.float32))
        print(f"{nm}: wrote train/val/test + test_seg/test_bg/mask (mean leaf coverage {100 * masks.mean():.0f}%) ({time.time() - t0:.0f} s)")

    # PlantDoc parts: all (eval), official train/test, five folds
    pd_items = [(int(pd_labels[j]), load(pd_files[j])) for j in range(pd_n)]
    written["pd_all"] = write_part(os.path.join(args.out, "pd_all.bin"), S_EVAL, pd_items)
    written["pd_train"] = write_part(os.path.join(args.out, "pd_train.bin"), S_TRAIN, [pd_items[j] for j in np.flatnonzero(pd_split == 0)])
    written["pd_test"] = write_part(os.path.join(args.out, "pd_test.bin"), S_EVAL, [pd_items[j] for j in np.flatnonzero(pd_split == 1)])
    for k in range(5):
        written[f"pd_fold{k}_train"] = write_part(os.path.join(args.out, f"pd_fold{k}_train.bin"), S_TRAIN, [pd_items[j] for j in np.flatnonzero(fold != k)])
        written[f"pd_fold{k}_test"] = write_part(os.path.join(args.out, f"pd_fold{k}_test.bin"), S_EVAL, [pd_items[j] for j in np.flatnonzero(fold == k)])
    print(f"PlantDoc: wrote all/train/test + 5 folds ({time.time() - t0:.0f} s)")

    # ── meta + manifest ──
    def save_meta():
        np.savez(os.path.join(args.out, "meta_plant.npz"),
                 classes=np.array(classes), pd_classes=np.array(pd_classes), pd_to_pv=np.array([cid[PD_TO_PV[c]] for c in pd_classes]),
                 pv_label=labels, pv_lab=np.array(labs), pv_num=nums, pv_leaf=leaf_id, pv_group=group, pv_has_mask=has_mask,
                 pv_part_random=part["pv"], pv_part_grouped=part["pvg"], pv_official_test=is_official_test,
                 pv_file=np.array([os.path.relpath(f, pv_root) for f in files]),
                 pd_label=pd_labels, pd_fold=fold, pd_split=pd_split, pd_file=np.array([os.path.relpath(f, pd_root) for f in pd_files]),
                 **{f"{nm}_test_index": np.flatnonzero(part[nm] == 2) for nm in ("pv", "pvg")},
                 **{f"{nm}_test_nn_train_index": audit[nm]["nn"] for nm in ("pv", "pvg")},
                 **{f"{nm}_test_nn_mad": audit[nm]["nd"] for nm in ("pv", "pvg")},
                 **{f"{nm}_test_nn_mad64": audit[nm]["nd64"] for nm in ("pv", "pvg")})
        with open(os.path.join(args.out, "classes_pv.txt"), "w") as f:
            f.write("\n".join(classes) + "\n")
        with open(os.path.join(args.out, "classes_pd.txt"), "w") as f:
            f.write("\n".join(f"{c}\t{PD_TO_PV[c]}" for c in pd_classes) + "\n")
        manifest = dict(pv_n=n, classes=classes, per_class=per_class.tolist(), pv_with_mask=int(has_mask.sum()), seg_files=n_seg,
                        leaf_map=dict(covered=int(covered.sum()), leaves=len(leaf_names), mode=int(np.bincount(sizes).argmax()),
                                      max=int(sizes.max()), uncovered_per_class=unc),
                        official_split=dict(test=int(is_official_test.sum()), straddling_leaves=straddle),
                        splits={nm: np.bincount(part[nm], minlength=3).tolist() for nm in part},
                        chains=dict(frac_consecutive_under=float(all_frac), rows=chain_rows),
                        leak_audit={nm: {k: v for k, v in a.items() if k not in ("nn", "nd", "nd64")} for nm, a in audit.items()},
                        pd=dict(n=pd_n, train=int((pd_split == 0).sum()), test=int((pd_split == 1).sum()), classes=pd_classes,
                                map=PD_TO_PV, folds=np.bincount(fold).tolist()),
                        written=written, seed=args.seed)
        with open(os.path.join(args.out, "manifest_plant.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"meta + manifest saved ({time.time() - t0:.0f} s): {sum(written.values())} records in {len(written)} parts")


    save_meta()

    # ── the slow twins last, so the base arm can start while they write ──
    for nm in ("pv", "pvg"):
        pp = part[nm]
        tr = np.flatnonzero(pp == 0)
        if args.composites:
            bgs = sorted(glob.glob(os.path.join(args.imagenette, "*", "*.JPEG")))
            if not bgs:
                sys.exit(f"no Imagenette JPEGs under {args.imagenette} — run download_imagenette.sh")

            def comp(j):
                rs = random.Random(args.seed * 1000003 + int(j))   # per-image, thread-order independent
                col = center(Image.open(files[j]), S_TRAIN)
                if seg_paths[j] is None:
                    return col
                m = (np.asarray(center(Image.open(seg_paths[j]), S_TRAIN)).sum(axis=2) > 40).astype(np.float32)
                m = gaussian_filter(m, 1.5)[..., None]
                bg = np.asarray(center(Image.open(rs.choice(bgs)), S_TRAIN), dtype=np.float32)
                return Image.fromarray((np.asarray(col, dtype=np.float32) * m + bg * (1 - m)).astype(np.uint8))

            written[f"{nm}_train_comp"] = write_part(os.path.join(args.out, f"{nm}_train_comp.bin"), S_TRAIN,
                                                     [(int(labels[j]), (lambda jj=j: comp(jj))) for j in tr])
            print(f"{nm}: wrote train_comp ({time.time() - t0:.0f} s)")
        if args.aug:
            def aug(j):
                rs = random.Random(args.seed * 7919 + int(j))
                im = center(Image.open(files[j]), S_TRAIN)
                im = ImageEnhance.Brightness(im).enhance(rs.uniform(0.7, 1.3))
                im = ImageEnhance.Contrast(im).enhance(rs.uniform(0.7, 1.3))
                im = ImageEnhance.Color(im).enhance(rs.uniform(0.7, 1.3))
                a = np.asarray(im)
                fill = tuple(int(v) for v in np.median(a.reshape(-1, 3), axis=0))
                im = im.rotate(rs.uniform(-30, 30), resample=Image.BILINEAR, fillcolor=fill)
                a = np.array(im)
                # random erasing: one grey rectangle of 10–25% of the area
                area = rs.uniform(0.10, 0.25) * S_TRAIN * S_TRAIN
                ar = rs.uniform(0.5, 2.0)
                h, w = int(min(S_TRAIN, (area * ar) ** 0.5)), int(min(S_TRAIN, (area / ar) ** 0.5))
                y, x = rs.randint(0, S_TRAIN - h), rs.randint(0, S_TRAIN - w)
                a[y:y + h, x:x + w] = 128
                return Image.fromarray(a)

            written[f"{nm}_train_aug"] = write_part(os.path.join(args.out, f"{nm}_train_aug.bin"), S_TRAIN,
                                                    [(int(labels[j]), (lambda jj=j: aug(jj))) for j in tr])
            print(f"{nm}: wrote train_aug ({time.time() - t0:.0f} s)")

    save_meta()
    print(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
