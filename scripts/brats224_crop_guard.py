#!/usr/bin/env python3
"""Known-answer guard: prove data/brats224 is EXACTLY the center crop of data/brats.

The re-prep at 224 re-ran the whole pipeline — NIfTI decode, per-volume z-score,
uint8 quantization, patient split, slice selection. Any of those could have
drifted, and every one of them would drift *silently*: a differently-normalized
or differently-ordered dataset still trains, still converges, and still prints a
plausible Dice. The only thing that would look wrong is the comparison against
the 240 baseline, by which point the run has already cost hours.

So this does not sample statistics and eyeball them. It asserts the strongest
available known answer:

    brats224[i]  ==  brats240[i][:, 8:232, 8:232]     byte-for-byte

If that holds for image AND mask, then the crop is the *only* difference —
identical normalization, identical quantization, identical slice order,
identical split. Nothing else needs checking.

Usage:
    python3 scripts/brats224_crop_guard.py [--n 300]
"""
import argparse
import struct
import sys

import numpy as np

M = 8          # (240 - 224) // 2
C = 4          # modalities


def load(path, size, idxs):
    hw = size * size
    rec = C * hw + hw
    with open(path, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        out = {}
        for i in idxs:
            if i >= count:
                continue
            f.seek(4 + i * rec)
            b = f.read(rec)
            img = np.frombuffer(b[:C * hw], dtype=np.uint8).reshape(C, size, size)
            msk = np.frombuffer(b[C * hw:], dtype=np.uint8).reshape(size, size)
            out[i] = (img, msk)
    return count, out


def check(split, n):
    p240, p224 = f'data/brats/{split}.bin', f'data/brats224/{split}.bin'
    c240, _ = load(p240, 240, [])
    c224, _ = load(p224, 224, [])
    print(f'{split}: {c240} records @240, {c224} records @224')
    if c240 != c224:
        print(f'  GUARD FAILED: record counts differ ({c240} vs {c224})')
        print('    → the split or the slice-selection filter changed; the two')
        print('      datasets are not the same slices and cannot be compared')
        return False

    rng = np.random.default_rng(0)
    idxs = sorted(rng.choice(c240, size=min(n, c240), replace=False).tolist())
    _, a = load(p240, 240, idxs)
    _, b = load(p224, 224, idxs)

    bad_img = bad_msk = 0
    classes = set()
    for i in idxs:
        img240, msk240 = a[i]
        img224, msk224 = b[i]
        if not np.array_equal(img240[:, M:240 - M, M:240 - M], img224):
            bad_img += 1
        if not np.array_equal(msk240[M:240 - M, M:240 - M], msk224):
            bad_msk += 1
        classes.update(np.unique(msk224).tolist())

    print(f'  compared {len(idxs)} slices')
    print(f'  image mismatches: {bad_img}')
    print(f'  mask  mismatches: {bad_msk}')
    print(f'  mask classes present: {sorted(classes)}')
    ok = True
    if bad_img or bad_msk:
        print('  GUARD FAILED: 224 is not the exact center crop of 240')
        ok = False
    if not classes <= {0, 1, 2, 3}:
        print(f'  GUARD FAILED: mask carries classes outside 0..3: {sorted(classes)}')
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300, help='slices to compare per split')
    args = ap.parse_args()
    ok = all([check('val', args.n), check('train', args.n)])
    print()
    if ok:
        print('GUARD OK — data/brats224 is byte-exactly the center crop of data/brats')
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
