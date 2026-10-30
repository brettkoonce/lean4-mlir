# Imagenette, seven nets × five seeds, verified path — mars (4× RTX 3060, CUDA/XLA), 2026-10-29/30

`SUITE=imagenette scripts/sweeps/seed_sweep.sh 0,1,2,3 "1 2 3 4 5"` (planning/runs_queue_2026-10-29.md
Tier 1a): the `*-verified-adam` Imagenette drivers, 80 epochs, Chapter 5's recipe (ViT-Tiny at a
third of the rate with a five-epoch warmup), resident parameters, LPT-packed one job per card.
00:39 → 08:52 UTC, 35/35 OK, no AER, no `loss=NaN` in any log. Epoch-80 top-1 on Imagenette's 3,925
val images (`sweep.log` OK lines; per-epoch bitmaps in `bitmaps/`, gitignored). Every exe was rebuilt
on 2026-10-29 before launch (lake exe-cache).

| net | s1 | s2 | s3 | s4 | s5 | **median (range)** | mean | shipped | min / seed |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 89.66 | 89.40 | 90.09 | 89.96 | 90.04 | **89.96** (89.40–90.09) | 89.83 | 90.04 (one run) | 42 |
| ResNet-34 | 89.68 | 90.11 | 89.68 | 89.78 | 90.22 | **89.78** (89.68–90.22) | 89.90 | 89.99 (five-seed mean) | 52 |
| ResNet-50 | 89.63 | 90.06 | 89.27 | 90.01 | 89.40 | **89.63** (89.27–90.06) | 89.68 | 89.71 (one run) | 80 |
| MobileNetV2 | 88.99 | 88.74 | 89.12 | 88.76 | 88.87 | **88.87** (88.74–89.12) | 88.90 | 89.25 (one run) | 38 |
| MobileNetV4 | 85.96 | 86.27 | 85.89 | 85.71 | 86.04 | **85.96** (85.71–86.27) | 85.97 | — (side quest) | 43 |
| ConvNeXt-T | 82.04 | 83.75 | 82.90 | 83.80 | 84.03 | **83.75** (82.04–84.03) | 83.30 | 82.27 (one run) | 90 |
| ViT-Tiny | 68.61 | 68.99 | 68.00 | 68.82 | 67.87 | **68.61** (67.87–68.99) | 68.46 | 68.74 (one run) | 35 |

Shipped = `tab:imagenette_five` (and R50's 89.71 from the runs queue); those runs were on one RTX 4060 Ti
(ares), these on 3060s. One Imagenette val image is 0.025 points; a single run's 95% interval is
about ±1.0.

## The three questions the sweep was for

1. **EfficientNet-B0 after TF `SAME` padding (a89fa9af, 2026-10-12).** Median 89.96 against the
   shipped 90.04, which sits inside the new range. The padding change did not move the row.
2. **The ResNets on the re-emitted pool backward (ea934bc6, 2026-10-28).**
   - R34 against its September band, full arm at epoch 80:
     `runs/2026-09-12-r34-ablation-fp32-seeds/` (the shipped 89.99 is their mean) 90.32 / 90.06 /
     89.86 / 89.63 / 90.09, median 90.06 (89.63–90.32); `runs/2026-09-11-r34-ablation-bf16-seeds/`
     89.96 / 89.81 / 89.96 / 90.14 / 90.04, median 89.96 (89.81–90.14). The new range (89.68–90.22)
     lies inside the fp32 band; mean 89.90 against 89.99.
   - R50: no September five-seed set exists; the shipped single run (89.71) sits inside the new
     range, 0.08 above the median.
   - Neither ResNet moved. Per the runs queue's Tier 3 rule, the ImageNet ResNet verified halves
     trained on the old pool backward stand and do not become re-runs.
3. **The five-seed column for `tab:imagenette_five`.** Median ± range above for every row.

## Rows that differ from the shipped number

- **ConvNeXt-T, median 83.75 against the shipped 82.27** (`runs/2026-09-13-convnext-imagenette-ls1e-6-resident/`).
  Same artifact (549 outputs, 23 LN), same layer-scale 1e-6 init, residency on; the render has
  changed only in comments since. Seed 1 retraces the shipped run (epoch 20 77.45 vs 77.43, epoch 80
  82.04 vs 82.27); the other four finish 82.9–84.0. ConvNeXt's seed spread on this recipe is two
  points, the widest in the table, and the shipped run (and the non-resident 09-13 run, 81.78) were
  low draws. Not a code change.
- **MobileNetV2, median 88.87 against the shipped 89.25** (`runs/2026-08-12-mnv2-imagenette-xla-cuda/`).
  The shipped single run is 0.13 above this sweep's highest seed. The 2026-10-07 depthwise-fan init
  change (f58939fe) is on the ImageNet driver only (`dwFanK2` defaults false; the Imagenette driver
  doesn't set it), so the Imagenette MNv2 init is unchanged. A small offset, inside a single run's
  interval; one ares run against five mars runs.

## What lands where (not done here)

`tab:imagenette_five` as median ± range with this directory in the caption; the ConvNeXt row and its
Chapter 8 Run-it-first transcript (which quote 82.27) need a decision. `blueprint/src/content.tex`
not edited.
