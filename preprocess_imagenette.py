#!/usr/bin/env python3
"""Pre-process Imagenette to raw binary format for Lean loader.

Usage: python3 preprocess_imagenette.py <imagenette_dir> <output_dir>

Reads JPEG images from imagenette directory structure, resizes to 224×224,
saves as binary files (train.bin, val.bin) in CIFAR-like format:
  Header: count (4 bytes, little-endian uint32)
  Records: label (1 byte) + 224×224×3 bytes (channel-first: R, G, B planes)
"""
import os, sys, struct
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow required. Install with: pip install Pillow")
    sys.exit(1)

# Imagenette class folders → label indices (alphabetical by wnid)
CLASS_MAP = {
    'n01440764': 0,  # tench
    'n02102040': 1,  # English springer
    'n02979186': 2,  # cassette player
    'n03000684': 3,  # chain saw
    'n03028079': 4,  # church
    'n03394916': 5,  # French horn
    'n03417042': 6,  # garbage truck
    'n03425413': 7,  # gas pump
    'n03445777': 8,  # golf ball
    'n03888257': 9,  # parachute
}

TRAIN_SIZE = 256  # train at 256, random crop to 224 at train time
VAL_SIZE = 224    # val center-cropped to 224

def process_split(src_dir, out_path, size=VAL_SIZE):
    records = []
    import numpy as np
    for class_dir in sorted(os.listdir(src_dir)):
        if class_dir not in CLASS_MAP:
            continue
        label = CLASS_MAP[class_dir]
        class_path = os.path.join(src_dir, class_dir)
        for fname in sorted(os.listdir(class_path)):
            if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).convert('RGB').resize((size, size), Image.BILINEAR)
            except Exception as e:
                print(f"  skipping {fpath}: {e}")
                continue
            # Channel-first: R plane, G plane, B plane (like CIFAR)
            arr = np.array(img, dtype=np.uint8)  # (H, W, 3) HWC
            arr = arr.transpose(2, 0, 1)  # (3, H, W) CHW
            records.append((label, arr.tobytes()))

    print(f"  {len(records)} images → {out_path}")
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', len(records)))
        for label, pixels in records:
            f.write(struct.pack('B', label))
            f.write(pixels)

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <imagenette_dir> <output_dir>")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)

    print("Processing training split (256×256 for random crop)...")
    process_split(os.path.join(src, 'train'), os.path.join(dst, 'train.bin'), size=TRAIN_SIZE)

    print("Processing validation split (224×224 center crop)...")
    process_split(os.path.join(src, 'val'), os.path.join(dst, 'val.bin'), size=VAL_SIZE)

    print("Done.")

if __name__ == '__main__':
    main()
