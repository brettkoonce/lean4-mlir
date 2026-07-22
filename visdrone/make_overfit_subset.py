#!/usr/bin/env python3
"""Carve a tiny subset out of data/visdrone_fpn for the overfit probe.

The decisive question left by planning/yolo_scoring.md bite 0: can the Lean
trainer fit ANYTHING? Three signatures say it is not fitting its own training
data (train loss 315->300.9 and flat; +7.08M params did not lower the TRAIN
loss; objectness emits the base rate everywhere). If it cannot drive the loss
toward zero on 32 images it has memorized many times over, the defect is in the
trainer -- optimizer, LR, loss scale, update path -- not in the detector design,
and FD verification would not have caught any of those.

Records are copied VERBATIM out of the existing bin: same encoder, same images,
same targets, no re-encoding. The only variable is dataset size. train.bin and
val.bin get the same records on purpose -- for an overfit probe, val IS train.

    ./.venv/bin/python3 make_overfit_subset.py --n 32
"""
import argparse
import struct
from pathlib import Path

SRC = Path("../data/visdrone_fpn")
DST = Path("../data/visdrone_fpn_overfit")

IMG_SIZE = 448
FPN_GRIDS = (56, 28, 14)
PER_ANCHOR = 15
ANCHORS_PER_SCALE = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32,
                    help="images to keep (a multiple of batchSize=8 keeps the "
                         "last step from being a short batch)")
    args = ap.parse_args()

    ntot = sum(ANCHORS_PER_SCALE * PER_ANCHOR * g * g for g in FPN_GRIDS)
    img_bytes = 3 * IMG_SIZE * IMG_SIZE
    rec = img_bytes + ntot * 4

    src = SRC / "train.bin"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    DST.mkdir(parents=True, exist_ok=True)

    with open(src, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        n = min(args.n, count)
        payload = f.read(n * rec)
    assert len(payload) == n * rec, f"short read: {len(payload)} != {n*rec}"

    for name in ("train.bin", "val.bin"):
        with open(DST / name, "wb") as g:
            g.write(struct.pack("<I", n))
            g.write(payload)

    mb = (4 + n * rec) / 1024 / 1024
    print(f"Ntot={ntot}, record={rec} bytes")
    print(f"wrote {DST}/train.bin and val.bin: {n} records, {mb:.1f} MB each "
          f"(sliced from {count})")


if __name__ == "__main__":
    main()
