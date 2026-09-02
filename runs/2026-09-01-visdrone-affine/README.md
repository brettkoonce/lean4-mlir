# 2026-09-01 — box-aware affine: the schedule was the confound

Analysis in `planning/visdrone_detector.md` §13a / §13a-bis / §13a-ter. This is the
workings. Companion: `runs/2026-09-01-visdrone-decode-sweep/README.md` (the decode
sweep, and why `--topk 3000` is part of the scoring command).

## The result

`FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_AFFINE=50 FPN_EPOCHS=30 FPN_TAG=aff30`
→ **mAP@0.5 = 0.2363** at e28, against the previous best 0.1961. ~2 h, one 4060 Ti.

```
LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=0 \
  FPN_BACKBONE=r34 FPN_TOWER=0 FPN_TAG=aff30 \
  FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_AFFINE=50 FPN_EPOCHS=30 \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
```

⚠ `FPN_BACKBONE` defaults to **r50**, not r34 — omitting it silently trains a
different arm. `FPN_AFFINE` is a PERCENT probability; scale/translate default to
±0.25 / ±0.10 (`FPN_AFFINE_SCALE` / `_TRANSLATE`), which is the measured default
and NOT Ultralytics' ±0.50.

## Epoch curve — converged, and that is the point

| epoch | mAP@0.5 | ca-AP | recall |
|---|---|---|---|
| e12 | 0.1641 | 0.3817 | 0.7430 |
| e16 | 0.2113 | 0.4604 | 0.7529 |
| e20 | 0.2254 | 0.4751 | 0.7632 |
| e24 | 0.2338 | 0.4834 | 0.7660 |
| **e28** | **0.2363** | **0.4872** | **0.7688** |
| e30 | 0.2362 | 0.4890 | 0.7685 |

e28 ≈ e30 is what makes this a settled measurement. Every 12-epoch affine number
is not — see below.

## ⛔ The 12-epoch arms measured the schedule, not the dose

| arm | affine p | epochs | e12 mAP |
|---|---|---|---|
| `cfoc2` | 0.00 | 12 | 0.1961 |
| `aff25` | 0.25 | 12 | 0.1429 |
| `aff50` | 0.50 | 12 | 0.1763 |
| `aff30` | 0.50 | 30 | 0.1641 |

Non-monotonic in p, which no dose-response story explains. Checked and cleared at
the time: BN running stats comparable across arms (same scale, no non-finite);
`_params.bin` byte-identical to `_params_e12.bin` for each. The explanation is the
last row — **no affine arm is near converged at 12 epochs**, so those rows compare
positions on unconverged trajectories. p=0.25 vs p=0.50 is still unmeasured.

⚠ **No error bar exists and none is runnable.** Init and data order are fixed and
augmentation is seeded `epoch*10000 + bi`; `Train.lean` has no seed env var
(`LEAN_MLIR_SEED` belongs to `VerifiedTrain.lean`, a different path). Two arms at
different `p` share an init but diverge from the first coin flip, so dose and
realization are confounded by construction. Every number here is n=1.

## Scoring

Six epoch dumps, four GPUs, then scored at the canonical decode:

```
for TAG in aff30e12 aff30e16 aff30e20 aff30e24 aff30e28 aff30; do
  LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$DEV FPN_BACKBONE=r34 FPN_TOWER=0 \
    FPN_TAG=$TAG .lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn <dir>
done
python3 scripts/yolo_map_visdrone.py <dir>/logits.bin data/visdrone448/val.bin \
  --fpn data/visdrone --grid 14 --multilabel --topk 3000 --ml-k 3 --ml-floor 0.05
```

⚠ An epoch checkpoint is scored by copying THREE files to a new tag prefix —
`_params_eN.bin`, `_bn_stats_eN.bin` **and `_fwd_eval.mlir`**. `infer` hard-fails
without the graph, and `FPN_TAG` is not optional: an untagged eval silently scores
a different arm and the only tell is identical rows across an epoch sweep.

## Per class, `cfoc2` → `aff30` e28

| class (val GT) | before | after | Δ |
|---|---|---|---|
| awning-tri (532) | 0.0392 | 0.0623 | +58.9% |
| bicycle (1,287) | 0.0253 | 0.0356 | +40.7% |
| tricycle (1,045) | 0.1025 | 0.1438 | +40.3% |
| van (1,975) | 0.2146 | 0.2900 | +35.1% |
| truck (750) | 0.1325 | 0.1707 | +28.8% |
| pedestrian (8,844) | 0.1753 | 0.2255 | +28.6% |
| motor (4,886) | 0.2114 | 0.2658 | +25.7% |
| bus (251) | 0.2248 | 0.2645 | +17.7% |
| people (5,125) | 0.1929 | 0.2202 | +14.2% |
| car (14,064) | 0.6428 | 0.6851 | +6.6% |

All ten improve; the rare ones improve most, and ca-AP and recall rise together
(0.4414 → 0.4872, 0.7485 → 0.7688) rather than trading off — the failure mode that
made static class weighting net harmful.
