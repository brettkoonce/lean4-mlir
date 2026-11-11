# visdrone/ — the PyTorch references for the VisDrone detector

Two conventional PyTorch detectors on VisDrone-DET, built to give the Lean FPN detector
([`demos/README.md`](../README.md#object-detection--r34fpn-on-visdrone), `lake run demo yolo visdrone`)
something to be measured **against**. Neither is a demo: they are the reference rung of the
validation ladder, and the book's detection table quotes both.

- **A stock YOLOv8s** (`train_baseline.py`), for the known-good number on this dataset:
  mAP@0.5 0.391 under its ordinary recipe, 0.140 at the Lean arm's budget.
- **A PyTorch twin of the Lean detector** ([`bespoke/`](bespoke/)), the same ResNet-34 + FPN
  architecture and loss, for the implementation check: 0.1532 at 12 epochs, no augmentation,
  argmax decode, where the Lean trainer on that recipe and decode scores 0.151 ± 0.004.

The Lean detector itself is at **0.232 ± 0.004** over three seeds (30 epochs, scale augmentation,
multilabel decode). The distance from 0.15 to 0.23 is recipe, not implementation.

## Layout

| file | what it does |
|---|---|
| `prepare_yolo.py` | VisDrone-DET → YOLO labels. Uses the **identical** class filter as the Lean pipeline's `parse_visdrone_txt`, so the two arms score the same GT. |
| `train_baseline.py` | stock ultralytics recipe; `--no-aug` mirrors the Lean arm's `augment := false` |
| `inspect_lean_bin.py` | renders a record straight out of `data/visdrone_fpn/train.bin` with encoded targets + raw annotations overlaid |
| `check_bin_alignment.py` | numerical version of the above: does the encoder round-trip? |
| `make_squash_dataset.py` | pre-squashes VisDrone to 448×448 with the Lean pipeline's resize, for the squash-vs-letterbox A/B below |
| `bespoke/` | the PyTorch twin of the Lean FPN detector (`model.py`, `loss.py`, `data.py`), with `diff_lean.py` and `lean_ckpt.py` to load a Lean checkpoint and diff the two forward passes |
| `compare_scoring.py`, `make_overfit_subset.py`, `make_perfect_logits.py`, `train_cripple.py` | probes from the July 2026 diagnosis (§ History) |

Setup (already done; recorded for reproducibility):

```
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
.venv/bin/pip install ultralytics pillow numpy
.venv/bin/python3 prepare_yolo.py          # writes labels/ beside the source images
HIP_VISIBLE_DEVICES=0 .venv/bin/python3 -u train_baseline.py --epochs 100 --name v8s_640_100ep
```

## Cross-checks that the two arms are measuring the same thing

`prepare_yolo.py` independently reproduces the Lean pipeline's GT exactly:

| quantity | Lean pipeline | this converter |
|---|---|---|
| train boxes | 343,204 | **343,204** |
| val boxes/img | 70.7 | **70.7** |

Same 10 classes, same `score==0` / category-0 / category-11 filter, same
`cat-1` remap.

## The reference numbers

`metrics/mAP50(B)` on the 548-image val split.

| arm | recipe | init | epochs | recall | mAP@0.5 |
|---|---|---|---|---|---|
| YOLOv8s `scratch_noaug_12ep_448` | no aug, 448 letterbox | random | 12 | 0.164 | 0.114 |
| YOLOv8s `scratch_squash_12ep_448` | no aug, 448 squash | random | 12 | 0.197 | **0.140** |
| YOLOv8s `leanmatch_noaug_12ep_448` | no aug, 448 letterbox | COCO-det | 12 | 0.240 | 0.192 |
| YOLOv8s `v8s_640_100ep` | full aug, 640 | COCO-det | 100 | 0.400 | **0.391** (best, e60) |
| PyTorch twin (`bespoke/`) | no aug, 448, argmax decode | — | 12 | — | **0.1532** |

**mAP@0.5 = 0.391 / mAP@0.5:0.95 = 0.225** sits in the normal published band for a YOLOv8s on
VisDrone, so the ladder is trustworthy. The squash row is the one matched to the Lean arm's
input and budget, and it is the 0.140 the book quotes. The no-aug COCO arm peaks at epoch 6
(0.198) and drifts to 0.192 by 12, ordinary no-augmentation overfitting.

**The encoder round-trips.** `check_bin_alignment.py` decodes 38,421 encoded targets out of
`train.bin` across 600 records and matches each against its annotation: centre residual max
2.11e-09 normalized, w/h max 2.95e-08, 0 boxes missing their annotation.

## Squash vs letterbox

`make_squash_dataset.py` pre-squashes VisDrone to 448×448 with the Lean pipeline's exact
`resize((448,448))` call, so the subsequent letterbox is a no-op and the model sees precisely
what the Lean trainer sees. Identical recipe otherwise (random init, no aug, 448, 12 epochs).

| epoch | 6 | 8 | 10 | **12** |
|---|---|---|---|---|
| letterbox | 0.0946 | 0.1014 | 0.1103 | **0.1140** |
| **squash** (Lean-style) | 0.1139 | 0.1362 | 0.1401 | **0.1399** |

Squashing is 23% better, at every epoch. The aspect distortion is real (0.750 for 4:3 sources,
0.562 for 16:9), but letterboxing a 16:9 frame into a 448 square spends about 44% of the canvas
on grey padding and shrinks objects that are already a few pixels across. On this dataset that
costs more than the distortion, so the Lean pipeline keeps the squash.

## History

This directory was built in July 2026, when every Lean detector arm scored mAP@0.5 = 0.0001 and
there was no known-good number on the dataset to test the pipeline against. The YOLOv8s rows
supplied one, and the cause turned out to be in the trainer: `lean_f32_shuffle`
permuted images by a full record and labels by a hardcoded four bytes, so every detection and
segmentation trainer saw mismatched image/target pairs. With the label stride threaded through
(2026-07-22) the same arm went 0.0001 → 0.1167. The scoring probes written before that fix
measured a detector trained on scrambled pairs; their scripts stay, their conclusions do not,
and [`planning/archive/post_shuffle_fix.md`](../../planning/archive/post_shuffle_fix.md) is the
ledger of what the fix invalidated.
