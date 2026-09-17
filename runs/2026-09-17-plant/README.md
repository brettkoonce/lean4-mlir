# PlantVillage → PlantDoc, 2026-09-17 — chapter 6's ResNet-34 from lab leaves to field leaves

Plan: `planning/plant_lab_to_field_demo.md` (§11 is the log). Book: `\subsection{Agriculture}`
after People watching. This directory holds Gate 0 (`gate0.log`), the queue scripts, the
scores, the CAM / Shapley summaries and the mock figure; each arm's logs, curves, checkpoints
and logits are in `runs/2026-09-17-plant-<split>-<train>-<init>/` (the field fine-tunes in
`…-grouped-field-imagenet/`, one log per fold and label count).

## Gate 0 — `gate0.log`, `data/plant/manifest_plant.json`

- PlantVillage: 54,305 colour images, 38 classes, 152 (`Potato___healthy`) – 5,507 (`Orange___
  Haunglongbing`); every colour file has a `segmented` twin (the Corn-rust colour files carry no
  uuid prefix; twins are matched on the suffix after `___`); mean leaf coverage of the 224 crop
  59%. Licence per the repo's dataset card: CC BY-SA 3.0.
- PlantDoc: 2,578 images (2,342 train / 236 test), 28 folders, all mapped into PlantVillage's 38
  (`Tomato two spotted spider mites leaf` has 2 images and no test image); 10 PlantVillage classes
  have no twin. CC BY 4.0. Images 300²–1600×1200, a few RGBA/CMYK.
- **PlantVillage ships a leaf grouping** (`leaf_grouping/leaf-map.json`): 41,111 of 54,305 files
  covered, 7,946 leaves, the mode exactly 4 photographs per leaf, max 33; 13 classes uncovered.
  The maintainers' Hugging Face split (`splits/color_{train,test}.txt`, 43,596 / 10,709) respects
  it: 0 leaves straddle. Splits here: `random` 43,447 / 5,428 / 5,430 (80/10/10 per class, seed
  0); `grouped` 38,112 / 5,484 / 10,709 (test = the maintainers' test list exactly, val carved from
  their train by leaf id; 0 leaves straddle).
- **The ArASL audit finds PlantVillage pixel-clean.** Consecutive suffix numbers within (class,
  lab): 0.1% of pairs under 6 grey levels (ArASL: 73.5%), median |Δ| 31.3. Test → nearest train
  image at 16×16: random 1.3% under 6 (0.1% at 64×64; nearest is the same leaf 6.9%), grouped
  1.0% (same leaf 0.0%). The four photographs of a leaf are different poses, not near-duplicate
  frames; a leaf-level leak, if any, is semantic and only the two accuracy columns can show it.
- 1-NN on 16×16 thumbnails: 53.3% (grouped test) — the lab backgrounds and leaf silhouettes
  already sort the classes halfway.

## Act 1–2 — the base arm (R34, ImageNet prefix, 10 epochs, Adam 1e-3 warmup+cosine, batch 64)

| arm | PlantVillage test | PlantDoc, all 2,578, restricted argmax | 38-way | unmapped predictions |
|---|---|---|---|---|
| grouped, ImageNet prefix, s1 (best val e10, 99.49) | **99.64** [99.50, 99.73] (n=10,709) | **15.79** [14.43, 17.25] | 13.62 | 22.7% |
| random, ImageNet prefix, s1 (e10, 99.67) | 99.72 [99.55, 99.83] (n=5,430) | 16.56 [15.18, 18.05] | 13.85 | 31.5% |
| grouped, from scratch, s1 (e10, 98.81) | 99.07 [98.86, 99.24] | 11.60 [10.42, 12.89] | 9.04 | 24.8% |

Seeds 2–3 of grouped/ImageNet: PlantVillage 99.56 / 99.51, PlantDoc 19.12 / 18.50 → **3-seed mean
99.57 ± 0.06 lab, 17.80 ± 1.77 field** (pooled [16.97, 18.67]; `score_grouped_base_*_3seeds.json`).
The field number swings ±1.8 between seeds of the same recipe — a fix has to beat that. The grouped and random columns agree on PlantVillage
(99.64 vs 99.72): the leaf grouping does not move this net. ~40 min per run on one 4060 Ti with
three cards busy (237 s/epoch + 9 s eval). PlantDoc per class (grouped/ImageNet): Tomato late
blight 56.8, Pepper healthy 50.8, Corn gray leaf spot 50.0 … Peach healthy 0.0, Potato late
blight 0.0, Tomato mosaic 0.0 (`score_grouped_base_pd_all.json`); top confusions: Corn gray leaf
spot ↔ Corn northern leaf blight 79, Potato late blight → Tomato late blight 44, Cherry healthy →
Tomato YLCV 46.

## Act 3 — the diagnosis, grouped test tenth, base arm

| | value |
|---|---|
| the same leaves from `segmented` (leaf on **black**) | **49.08%** |
| leaf on the image's median background colour (`_test_leaf`) | 94.98% |
| background only: leaf region filled with the median colour (`_test_bg`; the silhouette keeps the shape) | **27.64%** (chance 2.6%) |
| a flat image of the median background colour (`_test_none`) | 3.83% (random split's net: 9.32%) |
| two-player Shapley, exact: leaf share of the true-class logit | mean **85.5%**, median 88.3%; φ_leaf +15.5, φ_bg +2.6; the background helps (φ_bg > 0) in **89.6%** of images; efficiency error 3.8e-6 |
| CAM (closed form, 7×7): mass inside the leaf mask | mean **71.5%** (uniform map 58.9%, lift +12.6); 4.3% of images under half; correct 71.5 / wrong 69.2 |
| lowest leaf share by class (Shapley) | Tomato healthy 52%, Peach healthy 56%, Tomato mosaic 61% |

Reading: the two attribution instruments agree that most of the evidence is the leaf (Shapley
85%, CAM 72% of mass with a 13-point lift over uniform) — and the counterfactuals say the
background is nonetheless decisive: black behind the same leaf costs 50 points, a flat colour
alone scores above chance, and the background's marginal contribution is positive for nine
images in ten. Saliency is not sensitivity. `shapley2_grouped_base.json`, `cam_grouped_base.json`.

CAM mass inside the leaf across the base seeds: 71.5 / 70.7 / 70.5 (`cam_grouped_base*.json`);
Shapley leaf share 85.5 / 85.5 / 84.0 — the attribution numbers barely move with the seed, the
black-background counterfactual (49 / 59 / 72) is the one that does.

Sampled 7×7 Shapley (40 permutations, 7,844 forwards) on the figure's four leaves: Σφ equals
f(full) − f(none) to 1e-3 on all four, mean SE 0.06–0.09 against top-patch φ 1.2–2.0
(`…_logits_shap_probe_maps.npz`).

## Act 4 — the fixes, all scored on the 2,578 PlantDoc images (restricted argmax, Wilson)

| arm | PlantVillage test | PlantDoc | leaf on black | bg only | flat colour | Shapley leaf share | CAM inside leaf |
|---|---|---|---|---|---|---|---|
| base (Act 1), grouped s1 | 99.64 | 15.79 [14.43, 17.25] | 49.08 | 27.64 | 3.83 | 85.5% (bg helps 89.6%) | 71.5% |
| base, 3 seeds (mean ± sd) | 99.6 ± 0.1 | 17.80 ± 1.77 | 59.9 ± 11.5 | 25.1 ± 2.3 | 4.4 ± 2.4 | 85.0% (bg helps 88.2%) | 70.9 ± 0.5% |
| + leaves on Imagenette backgrounds (train ∪ comp, 76,224/epoch, best val e9) | 99.37 | **20.09** [18.59, 21.68] | **96.57** | 14.06 | 2.33 (= chance) | **95.9%** (bg helps 70.6%) | **78.6%** |
| + backgrounds, 3 seeds (mean ± sd; s2/s3: 21.80 / 22.54 field) | 99.44 ± 0.15 | **21.48 ± 1.25**, pooled [20.58, 22.41] | 96.3 ± 0.3 | 13.5 ± 1.4 | 2.6 ± 0.6 | 96.1% (bg helps 71.4%) | (seed 1) |
| + stronger augmentation (train ∪ aug, best val e9) | 99.62 | 19.24 [17.76, 20.81] | 58.96 | 28.87 | 5.54 | 81.0% (bg helps 90.0%) | 71.0% |
| + 250 field labels, 5-fold, fixed 5 ep at 1e-4 from base | — | 22.69 [21.12, 24.35]; folds 22.9 / 23.5 / 21.4 / 26.3 / 19.4 | | | | | |
| + all field labels (2,062 per fold), 5-fold | — | **45.00** [43.08, 46.92]; folds 45.2 / 45.4 / 44.4 / 47.6 / 42.4 | | | | | |
| + backgrounds, then 250 field labels, 5-fold | — | **26.57** [24.90, 28.31]; folds 27.6 / 25.2 / 25.3 / 27.6 / 27.1 | | | | | |
| + backgrounds, then all field labels, 5-fold | — | 42.05 [40.16, 43.96]; folds 41.4 / 41.3 / 41.2 / 43.3 / 43.0 | | | | | |

Augmentation moves none of the diagnostics (black 59.0, background-only 28.9, flat 5.5, Shapley 81%
— all inside the base seeds' spread) and its field number (19.2) sits inside the base spread too:
photometric and geometric variety does not touch what the net learned about the lab.
The background fix moves every diagnostic the way the diagnosis predicted — black background
49 → 97, background-only 28 → 14, the flat colour to chance, Shapley leaf share 85 → 96, CAM lift
+12.6 → +19.7 — and PlantDoc by +3.7 points over three seeds each (17.80 ± 1.77 → 21.48 ± 1.25; pooled intervals
disjoint; every composite seed above every base seed). Field labels buy +7 at ten per
class and +29 at all of them. The two fixes stack in the few-label regime (composites then 250
labels: 26.6 against 22.7) and not at full labels (42.1 against 45.0, intervals touching): once
the net has two thousand field images the lab-background repair is no longer what limits it.
(`score_grouped_field_comp.json`.) The composite arm's 10 epochs took 4,460 s (2× data).
`score_grouped_comp_pd_all.json`, `score_grouped_field.json`, `shapley2_grouped_comp.json`,
`cam_grouped_comp.json`; the field runs' PlantVillage val drifts to ~89% during the fine-tune
(BN statistics follow the field batches), which is expected and not a number the table uses.

## How it ran

- `preprocess_plant.py --stats --composites --aug`: 733 s (twins ~10 min); `--only-shapley-parts`
  77 s. ⚠ `random.Random(numpy.int64)` raises on Python 3.12 — cast to int (the first run died at
  the composites). The evaluation parts and meta are written before the slow twins so training can
  start.
- `lake build plant-leaf` 7–30 s. The u8 records stay resident (`F32.imagenetteGather`, new in
  `ffi/f32_helpers.c`): 8.5 GB per 43k-image part instead of 34 GB as f32. 38k images load in
  4.6 s; ~330 ms/step at batch 64 with three cards busy.
- CAM dump (`cam=1`): 10,709 + 2,578 images in 17 s. Shapley probe: 7,844 images in one eval.

## Reproduce

```bash
./download_plant.sh
lake build plant-leaf
CUDA_VISIBLE_DEVICES=0 lake exe plant-leaf split=grouped train=base init=imagenet seed=1 tag=s1 out=runs/x
.venv/bin/python scripts/plant_score.py runs/x/plant_resnet34_grouped_base_imagenet_s1_logits_pd_all.bin --part pd_all --restrict
.venv/bin/python scripts/plant_shapley.py two-player --split pvg --logits runs/x/plant_resnet34_grouped_base_imagenet_s1
CUDA_VISIBLE_DEVICES=0 lake exe plant-leaf split=grouped train=base init=imagenet seed=1 tag=s1 out=runs/x eval cam=1
.venv/bin/python scripts/plant_cam.py runs/x/plant_resnet34_grouped_base_imagenet_s1_cam_pvg_test.bin --split pvg
```
