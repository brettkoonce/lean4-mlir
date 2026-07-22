#!/usr/bin/env python3
"""Build a 448x448 aspect-SQUASHED copy of VisDrone, to price the Lean arm's resize.

The Lean pipeline does `Image.resize((448,448))` -- non-aspect-preserving -- while
ultralytics letterboxes. That is self-consistent, not a bug: boxes are normalized
by the original (iw, ih), so the encoding round-trips exactly. But it injects a
per-image width scaling of 0.750 (4:3 sources, ~51% of train) or 0.562 (16:9,
~49%) that is invisible in the pixels, so no width regressor can undo it.

This isolates exactly that cost. Pre-squashing to a SQUARE image means the
subsequent letterbox at imgsz=448 is a no-op, so the model sees precisely what
the Lean trainer sees. YOLO labels are normalized, and normalization is
invariant under an axis-independent rescale, so the label files are reused
verbatim -- only the pixels change.

Pairs with runs/scratch_noaug_12ep_448 (identical recipe, letterboxed) as the
control.

    ./.venv/bin/python3 make_squash_dataset.py
"""
import shutil
from pathlib import Path

from PIL import Image

SRC = Path("../data/visdrone")
DST = Path("../data/visdrone_squash")
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val"}
SIZE = 448

CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
           "tricycle", "awning-tricycle", "bus", "motor"]


def main():
    for split, srcdir in SPLITS.items():
        img_src = SRC / srcdir / "images"
        lbl_src = SRC / srcdir / "labels"
        if not lbl_src.is_dir():
            raise SystemExit(f"missing {lbl_src} -- run prepare_yolo.py first")
        img_dst = DST / srcdir / "images"
        lbl_dst = DST / srcdir / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        n = 0
        for jpg in sorted(img_src.glob("*.jpg")):
            out = img_dst / jpg.name
            if not out.exists():
                with Image.open(jpg) as im:
                    # exactly the Lean pipeline's call
                    im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(
                        out, quality=95)
            # normalized labels are invariant under an axis-independent rescale
            shutil.copyfile(lbl_src / (jpg.stem + ".txt"), lbl_dst / (jpg.stem + ".txt"))
            n += 1
        print(f"{split}: {n} images squashed to {SIZE}x{SIZE}")

    yaml = Path("data/visdrone_squash.yaml")
    yaml.write_text(
        f"path: {DST.resolve()}\n"
        f"train: {SPLITS['train']}/images\n"
        f"val: {SPLITS['val']}/images\n"
        "\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print(f"wrote {yaml}")


if __name__ == "__main__":
    main()
