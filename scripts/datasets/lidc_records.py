"""The CT pair-file format, read side — one module every LIDC / Mayo scorer imports.

`preprocess_lidc.py` (and `preprocess_mayo.py`, the held-out set) write `<split>.bin`:

    uint32 count, then per record
      full   S² uint16   HU + 1024, 0..4095                 the scanned (full-dose) slice
      low_r  S² uint16   r = 0 .. N-1                        N independent low-dose realisations
      mask   S² uint8    how many of the 4 readers marked the pixel (all zero for Mayo)

so a record is (3 + 2N)·S² bytes and N is read off the file size: no reader hard-codes it (the
25% LIDC build has N = 2, the 10% build and Mayo N = 1). `<split>.json` has one row per record,
`<split>.idx` the per-volume slice counts.

A dump (`lidc-denoise-train`, `lidc_classical.py`, `lidc-denoise-sample`) is `count` × S² f32 of
the noise in HU/100 on realisation 0, so denoised = low₀ − 100·eps. A dump that covers a subset
of the records (the sampler's) carries a sidecar `<dump>.records` listing the record indices it
wrote, one per line; the other records of the file are zero and must not be scored. `records_of`
and `common_records` keep every scorer on the same subset.
"""
import glob
import os

import numpy as np

S = 512
HW = S * S


def open_records(path, size=S):
    """(count, N, memmap [count, (3 + 2N)·S²] uint8) of a pair file; N from the file size."""
    count = int(np.fromfile(path, dtype="<u4", count=1)[0])
    body = os.path.getsize(path) - 4
    hw = size * size
    if count == 0 or body % (count * hw) != 0:
        raise ValueError(f"{path}: {body} bytes is not count={count} records of k·{size}²")
    per = body // (count * hw)
    if per < 5 or (per - 3) % 2:
        raise ValueError(f"{path}: {per}·S² bytes per record is not 3 + 2N")
    n = (per - 3) // 2
    return count, n, np.memmap(path, dtype=np.uint8, mode="r", offset=4, shape=(count, per * hw))


def decode(rec, n, size=S, realisation=0):
    """(full, low_r, mask): HU as float64 for the two images, the uint8 reader count."""
    hw = size * size
    imgs = np.asarray(rec[: 2 * (1 + n) * hw]).view("<u2").reshape(1 + n, size, size)
    full = imgs[0].astype(np.float64) - 1024.0
    low = imgs[1 + realisation].astype(np.float64) - 1024.0
    return full, low, np.asarray(rec[2 * (1 + n) * hw:]).reshape(size, size)


def open_dump(path, count, size=S):
    """memmap [count, S, S] f32 of a dump, its size checked against the pair file's count."""
    want = count * size * size * 4
    have = os.path.getsize(path)
    if have != want:
        raise ValueError(f"{path}: {have} bytes, {count} records of {size}² f32 need {want}")
    return np.memmap(path, dtype="<f4", mode="r", shape=(count, size, size))


def records_of(dump_path):
    """The record indices a dump covers (its `.records` sidecar), or None when it covers all."""
    side = dump_path + ".records"
    if not os.path.exists(side):
        return None
    return sorted({int(tok) for tok in open(side).read().split()})


def read_record_list(path):
    """A record list file: integers separated by whitespace, `#` to end of line a comment."""
    out = set()
    for line in open(path):
        line = line.split("#", 1)[0]
        out.update(int(tok) for tok in line.split())
    return sorted(out)


def common_records(count, dump_paths, requested=None):
    """The records every dump in `dump_paths` covers, intersected with `requested` (a list or
    None): the one index set a table compares its arms on."""
    recs = set(range(count)) if requested is None else set(requested)
    for p in dump_paths:
        r = records_of(p)
        if r is not None:
            recs &= set(r)
    return sorted(recs)


def find_dumps(data, split="val"):
    """{arm dir name: dump path} under `<data>/out/*/`: the `*<split>*.eps.f32` of each arm dir,
    sample seeds (`.k<n>.eps.f32`) listed under `<arm>.k<n>`."""
    out = {}
    for d in sorted(glob.glob(os.path.join(data, "out", "*"))):
        if not os.path.isdir(d):
            continue
        arm = os.path.basename(d)
        for f in sorted(glob.glob(os.path.join(d, f"*{split}*.eps.f32"))):
            stem = os.path.basename(f)[: -len(".eps.f32")]
            seed = stem.rsplit(".k", 1)[1] if ".k" in stem else None
            out[arm if seed is None else f"{arm}.k{seed}"] = f
    return out


LABELS = {"gauss": "Gaussian", "tv": "TV", "bm3d": "BM3D", "bm3dpsd": "BM3D-PSD", "mse": "MSE UNet",
          "n2n": "Noise2Noise", "diff": "diffusion",
          # the §11 arms and the cross-dose rows
          "diffmix": "diffusion, range", "diffmixsig": "diffusion, range + σ̂", "msemix": "MSE UNet, range",
          "diff_clamp": "diffusion, clamped", "diff25": "diffusion, 25 %", "mse25": "MSE UNet, 25 %",
          "n2n25": "Noise2Noise, 25 %",
          # §12: the b50f-kernel arms seen from the Shepp-Logan roots, and §9's arms seen from the b50f root
          "diff_b50": "diffusion, b50f", "mse_b50": "MSE UNet, b50f",
    "diffmixsig_b50": "diffusion, range + σ̂, b50f", "msemix_b50": "MSE UNet, range, b50f",
          "diff_sl": "diffusion, Shepp-Logan", "mse_sl": "MSE UNet, Shepp-Logan"}


def label(arm):
    """The table name of an arm dir (`find_dumps` key): `mse` → "MSE UNet", `diff.k2` → "diffusion k2"."""
    base, _, seed = arm.partition(".k")
    name = LABELS.get(base, base)
    return f"{name} k{seed}" if seed else name


def arm_base(arm):
    """`diff.k2` → `diff`: the arm a sample seed belongs to."""
    return arm.partition(".k")[0]
