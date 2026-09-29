#!/usr/bin/env python3
"""Pre-process the Medical Segmentation Decathlon brain-tumour task
(Task01_BrainTumour, BraTS-derived) to raw binary for the Lean loader.

Usage: python3 scripts/datasets/preprocess_brats.py <task_dir> <output_dir> [options]

  <task_dir> must contain (as shipped inside Task01_BrainTumour.tar):
    dataset.json
    imagesTr/BRATS_###.nii.gz   4D volumes, (X, Y, Z, 4) modalities
    labelsTr/BRATS_###.nii.gz   3D label volumes, (X, Y, Z)

  imagesTs/ is unlabelled (challenge submission set) and is ignored — the
  train/val split is carved out of the 484 labelled imagesTr volumes.

Writes train.bin, val.bin in the format `lean_f32_load_brats` expects:
  Header: count (4 bytes, little-endian uint32)
  Per record:
    image: C*240*240 bytes  (channel-first [channel][y][x], uint8)
    mask:  240*240   bytes  (per-pixel class label, 0..3)
  C = 4 in the 2D build (288,000 bytes per record at 240).

Modalities (channel order, from dataset.json):
  0 = FLAIR, 1 = T1w, 2 = T1gd (post-contrast), 3 = T2w

2.5D (`--context K`, optionally `--context-step S`): each record's image
carries the 2K+1 axial slices z-K*S, ..., z, ..., z+K*S of its volume, in
that order, each with its 4 modalities, so C = 4*(2K+1) and the centre
slice's modalities sit at channels 4K..4K+3 (T1gd at 4K+2). Neighbours come
from the UNFILTERED volume — a tumour-free or off-stride neighbour is still
a neighbour — and are replicated at the volume's two ends. Which centre
slices are kept, and in which order, is exactly the 2D build's choice, so a
2.5D build at the same seed sees the same records as its 2D control.

`--val-full` additionally writes val_full.bin: EVERY axial slice of every
validation volume, volume by volume in z order, no tumour filter and no
stride, with val_full.idx beside it (uint32 volume count, then one uint32
slice count per volume, in file order) and val_full.json naming the volumes.
That is what per-volume scoring reads (`lake exe brats-eval`): the
literature's Dice is over whole volumes, tumour-free slices included, and
val.bin's tumour-bearing subset cannot give it.

Mask classes (MSD's remap of the native BraTS 0/1/2/4 labels):
  0 = background   1 = edema   2 = non-enhancing tumour   3 = enhancing tumour

Two preprocessing decisions worth knowing, because they are the ones that
would silently ruin the numbers if done the obvious way instead:

  * **Normalization is over brain voxels only.** BraTS volumes are
    skull-stripped, so most of each volume is exact zero. Z-scoring over
    the whole volume would compute mean/std of mostly-background and
    crush the actual tissue contrast into a narrow band. We take the
    mean/std over nonzero voxels, per modality, per volume.

  * **The split is by patient, not by slice.** Adjacent axial slices of
    one brain are near-duplicates; splitting slices at random would put
    near-copies of the same patient on both sides and report a val score
    that is partly memorization. Volumes are split, then sliced.

Storage note: images are stored as uint8, quantizing the z-scored value
over a +/-5 sigma window (step ~0.039 sigma). This matches the on-disk
convention of every other dataset in the repo (imagenette/cifar/visdrone all
store uint8 and normalize in the C loader) and keeps train.bin 4x smaller
than f32. The quantization step is far below the tissue contrasts that
define tumour boundaries. `lean_f32_load_brats` inverts it on load.
"""
import argparse
import gzip
import json
import os
import struct
import sys

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. Install with: pip install numpy")
    sys.exit(1)

SIZE = 240        # native BraTS in-plane size; 240 = 16*15, so a depth-4
                  # UNet (4 halvings) divides evenly with no resize.
MODALITIES = 4
NUM_CLASSES = 4
CLIP_SIGMA = 5.0  # z-score clip window for uint8 quantization

# NIfTI datatype code -> numpy dtype character.
NIFTI_DTYPE = {
    2: 'u1', 4: 'i2', 8: 'i4', 16: 'f4', 64: 'f8',
    256: 'i1', 512: 'u2', 768: 'u4',
}


def read_nifti(path):
    """Minimal NIfTI-1 reader (numpy only — no nibabel dependency).

    Returns the voxel array. NIfTI-1 is a fixed 348-byte header followed by
    voxel data at `vox_offset`, stored column-major (dim[1] fastest). We
    read only what this task needs and assert loudly on anything else, so
    an unexpected file fails visibly rather than producing quiet garbage.
    """
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        raw = f.read()

    if len(raw) < 348:
        raise ValueError(f"{path}: too short to be NIfTI-1 ({len(raw)} bytes)")

    # Endianness is identified by which way sizeof_hdr reads back as 348.
    endian = None
    for e in ('<', '>'):
        if struct.unpack(e + 'i', raw[0:4])[0] == 348:
            endian = e
            break
    if endian is None:
        raise ValueError(f"{path}: bad sizeof_hdr — not a NIfTI-1 file")

    magic = raw[344:348]
    if magic not in (b'n+1\x00', b'ni1\x00'):
        raise ValueError(f"{path}: bad magic {magic!r}")
    if magic == b'ni1\x00':
        raise ValueError(f"{path}: detached .hdr/.img pairs are not supported")

    dim = struct.unpack(endian + '8h', raw[40:56])
    datatype = struct.unpack(endian + 'h', raw[70:72])[0]
    vox_offset = struct.unpack(endian + 'f', raw[108:112])[0]
    scl_slope = struct.unpack(endian + 'f', raw[112:116])[0]
    scl_inter = struct.unpack(endian + 'f', raw[116:120])[0]

    ndim = dim[0]
    if not (1 <= ndim <= 7):
        raise ValueError(f"{path}: bad dim[0]={ndim}")
    shape = tuple(int(d) for d in dim[1:1 + ndim])

    if datatype not in NIFTI_DTYPE:
        raise ValueError(f"{path}: unsupported NIfTI datatype code {datatype}")
    dt = np.dtype(endian + NIFTI_DTYPE[datatype])

    count = int(np.prod(shape))
    offset = int(vox_offset)
    arr = np.frombuffer(raw, dtype=dt, count=count, offset=offset)
    arr = arr.reshape(shape, order='F')  # NIfTI voxel data is column-major

    # scl_slope==0 means "no scaling" per the NIfTI-1 spec.
    if scl_slope != 0 and (scl_slope != 1.0 or scl_inter != 0.0):
        arr = arr.astype(np.float32) * scl_slope + scl_inter

    return arr


def znorm_brain(vol):
    """Z-score one modality over its nonzero (brain) voxels.

    Background stays exactly 0. See the module docstring for why this is
    over nonzero voxels rather than the whole volume.
    """
    vol = vol.astype(np.float32, copy=False)
    brain = vol != 0
    n = int(brain.sum())
    if n == 0:
        return np.zeros_like(vol, dtype=np.float32)
    vals = vol[brain]
    mu = float(vals.mean())
    sd = float(vals.std())
    if sd == 0.0:
        sd = 1.0
    out = np.zeros_like(vol, dtype=np.float32)
    out[brain] = (vals - mu) / sd
    return out


def quantize_u8(z):
    """z-score float -> uint8 over the +/-CLIP_SIGMA window.

    Zero is anchored to exactly 128 rather than centering the window on
    127.5, so background — exact zero in a skull-stripped volume, and 83%
    of every voxel grid — dequantizes back to exact zero instead of a small
    positive constant. Step is CLIP_SIGMA/127 ~= 0.039 sigma.

    `lean_f32_load_brats` inverts this; the two must stay in lockstep.
    """
    scaled = np.rint(z * (127.0 / CLIP_SIGMA) + 128.0)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def fit_plane(a, size):
    """Center-crop or zero-pad the leading two axes to size x size.

    MSD brain volumes are 240x240 in-plane already; this only matters if a
    volume deviates, in which case we keep the brain centered rather than
    dropping the case.
    """
    for axis in (0, 1):
        cur = a.shape[axis]
        if cur == size:
            continue
        if cur > size:
            lo = (cur - size) // 2
            a = np.take(a, range(lo, lo + size), axis=axis)
        else:
            padw = [(0, 0)] * a.ndim
            lo = (size - cur) // 2
            padw[axis] = (lo, size - cur - lo)
            a = np.pad(a, padw, mode='constant')
    return a


def channels_of(args):
    """Image channels per record: the 4 modalities of each of the 2K+1 slices."""
    return MODALITIES * (2 * args.context + 1)


def process_volume(img_path, lbl_path, args, all_slices=False):
    """One volume -> list of per-slice record bytes.

    `all_slices` keeps every z (the whole-volume export); otherwise the
    tumour filter and the stride pick the centres, exactly as in 2D.
    """
    img = read_nifti(img_path)
    lbl = read_nifti(lbl_path)

    if img.ndim != 4 or img.shape[3] != MODALITIES:
        raise ValueError(f"{img_path}: expected (X,Y,Z,{MODALITIES}), got {img.shape}")
    if lbl.ndim != 3:
        raise ValueError(f"{lbl_path}: expected 3D label, got {lbl.shape}")
    if lbl.shape != img.shape[:3]:
        raise ValueError(f"{img_path}: image/label shape mismatch {img.shape[:3]} vs {lbl.shape}")

    lbl = np.rint(lbl).astype(np.uint8)
    if lbl.max() >= NUM_CLASSES:
        raise ValueError(f"{lbl_path}: label {lbl.max()} outside 0..{NUM_CLASSES - 1}")

    # Normalize each modality over its own brain voxels, then quantize once
    # for the whole volume (cheaper than per-slice, and the statistics are
    # per-volume by definition).
    zs = [znorm_brain(img[..., m]) for m in range(MODALITIES)]
    qs = [quantize_u8(z) for z in zs]

    img_q = fit_plane(np.stack(qs, axis=-1), args.size)   # (X, Y, Z, M)
    lbl_q = fit_plane(lbl, args.size)                      # (X, Y, Z)

    depth = img_q.shape[2]
    tumor_per_slice = (lbl_q > 0).reshape(-1, depth).sum(axis=0)

    if all_slices:
        keep = list(range(depth))
    else:
        keep = [z for z in range(depth) if tumor_per_slice[z] >= args.min_tumor_px]
        keep = keep[::args.stride]

    # Slice offsets stacked into the channels: -K*S, ..., 0, ..., +K*S. The
    # 2D build is K = 0, i.e. the single offset 0.
    K, S = args.context, args.context_step
    offsets = [j * S for j in range(-K, K + 1)]
    n_ch = channels_of(args)

    records = []
    for z in keep:
        planes = []
        for off in offsets:
            # Replicate at the ends: a slice near the top of the volume sees
            # the top slice again rather than a zero plane it never trained on.
            zz = min(max(z + off, 0), depth - 1)
            # (X, Y, M) -> (M, Y, X): channel-first, row-major within each channel.
            planes.append(img_q[:, :, zz, :].transpose(2, 1, 0))
        sl = np.concatenate(planes, axis=0) if len(planes) > 1 else planes[0]
        mask = lbl_q[:, :, z].T
        img_bytes = np.ascontiguousarray(sl).tobytes()
        mask_bytes = np.ascontiguousarray(mask).tobytes()
        assert len(img_bytes) == n_ch * args.size * args.size
        assert len(mask_bytes) == args.size * args.size
        records.append(img_bytes + mask_bytes)
    return records


def write_split(task_dir, cases, out_path, args, all_slices=False, manifest=None):
    """Stream one split to `out_path`. With `manifest` (a list), append one
    `{name, start, count}` entry per volume in file order — the per-volume
    index the whole-volume export needs."""
    n_ch = channels_of(args)
    rec_bytes = n_ch * args.size * args.size + args.size * args.size
    count = 0
    class_hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    skipped = 0

    # Stream to disk: the full split is multi-GB and there is no reason to
    # hold it in memory. Header is backfilled once the count is known.
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', 0))  # placeholder
        for i, case in enumerate(cases):
            img_path = os.path.join(task_dir, 'imagesTr', case)
            lbl_path = os.path.join(task_dir, 'labelsTr', case)
            try:
                records = process_volume(img_path, lbl_path, args, all_slices=all_slices)
            except Exception as e:
                print(f"  skipping {case}: {e}", file=sys.stderr)
                skipped += 1
                continue
            if manifest is not None:
                manifest.append({'name': case, 'start': count, 'count': len(records)})
            for r in records:
                assert len(r) == rec_bytes
                f.write(r)
                m = np.frombuffer(r[n_ch * args.size * args.size:], dtype=np.uint8)
                class_hist += np.bincount(m, minlength=NUM_CLASSES)
            count += len(records)
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{len(cases)} volumes -> {count} slices")
        f.seek(0)
        f.write(struct.pack('<I', count))

    total_px = class_hist.sum()
    print(f"  {count} slices from {len(cases) - skipped} volumes ({skipped} skipped)")
    print(f"  wrote {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)")
    print(f"  f32 RAM at load: {count * n_ch * args.size * args.size * 4 / 1e9:.2f} GB")
    if total_px:
        names = ['background', 'edema', 'non-enhancing', 'enhancing']
        print("  class balance (per-pixel):")
        for c in range(NUM_CLASSES):
            print(f"    {c} {names[c]:<14} {100.0 * class_hist[c] / total_px:6.3f}%")
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task_dir', help='Task01_BrainTumour directory')
    ap.add_argument('out_dir', help='output directory for train.bin / val.bin')
    ap.add_argument('--size', type=int, default=SIZE,
                    help=f'in-plane size, must be a multiple of 16 (default {SIZE})')
    ap.add_argument('--val-frac', type=float, default=0.15,
                    help='fraction of volumes held out for validation (default 0.15)')
    ap.add_argument('--seed', type=int, default=0,
                    help='seed for the patient-level split (default 0)')
    ap.add_argument('--min-tumor-px', type=int, default=1,
                    help='keep axial slices with at least this many tumour voxels '
                         '(default 1; 0 keeps every slice)')
    ap.add_argument('--stride', type=int, default=2,
                    help='keep every Nth qualifying slice (default 2). Adjacent '
                         'axial slices are 1mm apart and highly redundant, so '
                         'stride 2 drops mostly near-duplicates while halving both '
                         'the file and the f32 footprint at load (~73 tumour slices '
                         'per volume at stride 1 => ~33 GB RAM; stride 2 => ~16 GB). '
                         'Use 1 for the full set.')
    ap.add_argument('--context', type=int, default=0, metavar='K',
                    help='2.5D: stack the K slices on each side of every kept slice '
                         'into the channels (4*(2K+1) channels per record; default 0 = 2D). '
                         'The centre slices kept are exactly the 2D build\'s.')
    ap.add_argument('--context-step', type=int, default=1, metavar='S',
                    help='spacing of the context slices in slices (default 1 = adjacent, '
                         '1 mm apart on this data; 2 = every other slice)')
    ap.add_argument('--val-full', action='store_true',
                    help='also write val_full.bin/.idx/.json: every slice of every '
                         'validation volume, for per-volume scoring (brats-eval)')
    ap.add_argument('--only-val-full', action='store_true',
                    help='write only the val_full files (implies --val-full); adds '
                         'per-volume scoring to an existing build at the same seed')
    ap.add_argument('--train-full', action='store_true',
                    help='also write train_full.bin/.idx/.json: every slice of every '
                         'TRAINING volume, volume by volume — the whole-volume corpus a '
                         '3D patch sampler reads (jax/scripts/unet3d_brats.py)')
    ap.add_argument('--only-train-full', action='store_true',
                    help='write only the train_full files (implies --train-full)')
    args = ap.parse_args()
    if args.only_val_full:
        args.val_full = True
    if args.only_train_full:
        args.train_full = True

    if args.size % 16 != 0:
        print(f"ERROR: --size {args.size} is not a multiple of 16; a depth-4 UNet "
              f"halves the resolution 4 times and would not divide evenly.")
        sys.exit(1)
    if args.context < 0 or args.context_step < 1:
        print("ERROR: --context must be >= 0 and --context-step >= 1")
        sys.exit(1)

    with open(os.path.join(args.task_dir, 'dataset.json')) as f:
        meta = json.load(f)

    # dataset.json lists training entries as {"image": "./imagesTr/BRATS_001.nii.gz", ...}
    cases = sorted({os.path.basename(e['image'] if isinstance(e, dict) else e)
                    for e in meta['training']})
    print(f"dataset: {meta.get('name', '?')} | {len(cases)} labelled volumes")
    print(f"modalities: {meta.get('modality')}")
    print(f"labels: {meta.get('labels')}")

    # Patient-level split — see the module docstring.
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(cases))
    n_val = max(1, int(round(len(cases) * args.val_frac)))
    val_idx = set(order[:n_val].tolist())
    train_cases = [c for i, c in enumerate(cases) if i not in val_idx]
    val_cases = [c for i, c in enumerate(cases) if i in val_idx]
    print(f"split (seed {args.seed}): {len(train_cases)} train / {len(val_cases)} val volumes")

    os.makedirs(args.out_dir, exist_ok=True)
    n_ch = channels_of(args)
    if args.context:
        print(f"2.5D: context ±{args.context} at step {args.context_step} -> "
              f"{n_ch} channels per record")
    if not (args.only_val_full or args.only_train_full):
        print("Processing training split...")
        n_tr = write_split(args.task_dir, train_cases, os.path.join(args.out_dir, 'train.bin'), args)
        print("Processing validation split...")
        n_va = write_split(args.task_dir, val_cases, os.path.join(args.out_dir, 'val.bin'), args)

        with open(os.path.join(args.out_dir, 'split.json'), 'w') as f:
            json.dump({'seed': args.seed, 'size': args.size, 'clip_sigma': CLIP_SIGMA,
                       'stride': args.stride, 'min_tumor_px': args.min_tumor_px,
                       'context': args.context, 'context_step': args.context_step,
                       'channels': n_ch,
                       'train_volumes': train_cases, 'val_volumes': val_cases,
                       'train_slices': n_tr, 'val_slices': n_va}, f, indent=2)
        print(f"Split manifest -> {os.path.join(args.out_dir, 'split.json')}")

    for flag, cases, stem in ((args.val_full, val_cases, 'val_full'),
                              (args.train_full, train_cases, 'train_full')):
        if not flag:
            continue
        print(f"Processing {stem.split('_')[0]} volumes, every slice ({stem})...")
        manifest = []
        n_full = write_split(args.task_dir, cases, os.path.join(args.out_dir, f'{stem}.bin'),
                             args, all_slices=True, manifest=manifest)
        # The Lean side reads the binary index; the JSON is for people.
        with open(os.path.join(args.out_dir, f'{stem}.idx'), 'wb') as f:
            f.write(struct.pack('<I', len(manifest)))
            for m in manifest:
                f.write(struct.pack('<I', m['count']))
        with open(os.path.join(args.out_dir, f'{stem}.json'), 'w') as f:
            json.dump({'seed': args.seed, 'size': args.size, 'channels': n_ch,
                       'context': args.context, 'context_step': args.context_step,
                       'slices': n_full, 'volumes': manifest}, f, indent=2)
        print(f"Per-volume index -> {os.path.join(args.out_dir, f'{stem}.idx')} "
              f"({len(manifest)} volumes, {n_full} slices)")
    print("Done.")


if __name__ == '__main__':
    main()
