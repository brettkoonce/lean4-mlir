#!/usr/bin/env python3
"""Stock-recipe VisDrone baseline -- the reference number the Lean arm lacks.

The point of this run is NOT to be clever. It is to establish what an
off-the-shelf detector achieves on this dataset with a normal recipe, so that
the Lean FPN arm's mAP@0.5 = 0.0001 can be read as "how far from achievable"
instead of being argued about in the dark. Everything here is ultralytics
defaults on purpose; the interesting work is the ablations that come after.

Deliberately different from the Lean arm in the four ways identified in
planning/yolo_assignment.md, which is exactly what makes the diff informative:
  augmentation   mosaic + flip + HSV + scale   vs the Lean arm's `augment := false`
  schedule       100 epochs                    vs 12
  resolution     640                           vs 448
  letterbox      aspect-preserving             vs a squash that distorts w/h by
                                                  0.750 or 0.562 depending on
                                                  the source aspect family

    ./.venv/bin/python3 train_baseline.py --epochs 2 --name smoke
    ./.venv/bin/python3 train_baseline.py --epochs 100 --name v8s_640_100ep
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--data", default="data/visdrone.yaml",
                    help="data/visdrone_squash.yaml prices the Lean arm's "
                         "aspect-squashing resize against this one's letterbox")
    ap.add_argument("--amp", action="store_true", default=False,
                    help="MIOpen on gfx1100 is conv-weak; AMP is off by default "
                         "until measured to help (see memory rocm-is-the-transformer-box)")
    ap.add_argument("--no-aug", action="store_true",
                    help="disable every augmentation, mirroring the Lean arm's "
                         "`augment := false`. Isolates how much of the gap the "
                         "data regime alone accounts for.")
    args = ap.parse_args()

    aug = {}
    if args.no_aug:
        aug = dict(mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                   hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
                   scale=0.0, shear=0.0, perspective=0.0, flipud=0.0,
                   fliplr=0.0, erasing=0.0, auto_augment=None,
                   # close_mosaic rebuilds the dataloader 10 epochs from the
                   # end; with mosaic already off it buys nothing, and the
                   # rebuild hung indefinitely here (epoch 3 produced no batch
                   # in 10 min with the GPU pinned at 100%).
                   close_mosaic=0)

    data = Path(args.data).resolve()
    if not data.exists():
        raise SystemExit("run prepare_yolo.py first")

    # A relative `project` is resolved against ultralytics' global runs_dir,
    # which defaults to the Lean repo's runs/ -- absolute keeps this track's
    # output inside visdrone/ and out of the Lean training logs.
    project = Path("runs").resolve()

    model = YOLO(args.model)
    model.train(
        data=str(data),
        project=str(project),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        exist_ok=True,
        amp=args.amp,
        val=True,
        plots=True,
        seed=0,
        **aug,
    )


if __name__ == "__main__":
    main()
