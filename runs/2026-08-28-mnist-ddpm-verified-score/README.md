# 2026-08-28 — the image DDPM gets its first real number, and then the number gets fixed

`planning/diffusion_2d_demo.md` §7's proposal, built and then acted on: score the MNIST diffusion
demo with Chapter 3's **verified** CNN instead of by looking at the grid. Three results.

1. ✅ **The metric works and is bracketed at both ends.** Real MNIST scored as if generated sits at
   the floor; unstructured pixels sit at 231× it.
2. ⛔⛔ **Its first run said the checkpoint on disk was not producing digits** — 119× the floor,
   with pixel values over **[-4.9, 11.9]** where the data is [0, 1]. Nothing in the repo said so,
   because the only output was a grid a human had to squint at.
3. ⭐⭐ **Fixed: 15× the floor, 10/10 coverage, 90.94 % confidence, legible digits** — 50 epochs
   instead of 3, plus centring the training data to [-1, 1] as all three CIFAR DDPM trainers
   already did.
   ⚠⚠ **The 2×2 was run, and epochs are the dominant term, not centring.** The first write-up of
   this run compared 3-epoch arms only and read as though centring did the work. A 50-epoch
   UNCENTRED figure from July (`demos/figures/ddpm_mnist.png`) showing legible digits is what
   prompted the missing arm. It reaches **33×** — worse than centred, better than everything else.
   ⭐⭐ **And confidence would have picked the WRONG model**: the uncentred 50-epoch arm scores
   *higher* confidence (92.82 % against 90.94 %) while dropping a whole class and doubling the
   energy distance. A single-number quality score fails on exactly this comparison.

⚠ This is the UNCONDITIONAL half. There is no accuracy number, because there is no conditioning.

---

## 1. The ladder

`score_*.log`. 1024 samples, 50 DDIM steps, 1024 real images as the positive control. The
classifier is `cnnVerified` from the committed `verified_mlir/cnn_fwd.mlir`, retrained here in 10
epochs at ~5 s each.

A full 2×2 in epochs × centring, bracketed by the two controls.

| arm | coverage | confidence | energy | × floor | pixel mean | pixel sd |
|---|---|---|---|---|---|---|
| real MNIST, scored as if generated | 10/10 | 99.40 % | 0.00196 | **1×** | 0.1221 | 0.2983 |
| **50 epochs, centred** | **10/10** | 90.94 % | **0.02365** | **15×** | 0.1381 | 0.3142 |
| 50 epochs, uncentred | 9/10 | **92.82 %** | 0.05255 | 33× | 0.1635 | 0.3360 |
| 3 epochs, centred | 8/10 | 84.19 % | 0.11545 | 73× | 0.1192 | 0.2998 |
| 3 epochs, uncentred (what was on disk) | 9/10 | 63.89 % | 0.18979 | 119× | 0.0146 | 0.1788 |
| unstructured pixels, MNIST's moments | 4/10 | 58.47 % | 0.36741 | 231× | 0.1320 | 0.3106 |
| real-vs-real floor | | | 0.00159 | 1× | 0.1325 | 0.3105 |

⭐ **Read the instrument line first.** The classifier scores **98.66 %** on the real test split.
That is the check that its weights loaded — a mispacked blob scores at chance and every number
above would then be measuring nothing. The scorer refuses to print a verdict below 90 %.

### Which change did the work

| held fixed | varied | energy |
|---|---|---|
| uncentred | 3 → 50 epochs | 119× → **33×** |
| centred | 3 → 50 epochs | 73× → **15×** |
| 3 epochs | raw → centred | 119× → 73× |
| 50 epochs | raw → centred | 33× → 15× |

⭐ **Epochs are the dominant term** (3.6× and 4.9×); centring is real but secondary (1.6× and
2.2×). Both together take 119× to 15×.

⚠⚠ **This corrects the first version of this write-up**, which had only the 3-epoch arms and read
as though centring was the fix. `demos/figures/ddpm_mnist.png` — a 50-epoch uncentred run from
July, showing legible digits — is what made the missing arm obviously necessary. ▶ Two arms of a
2×2 do not settle which factor mattered, and the pre-existing artifact was right there.

### ⭐⭐ The comparison a single quality score gets wrong

The uncentred 50-epoch arm scores **higher confidence** than the centred one (92.82 % against
90.94 %) while **dropping a class** (9/10, with "1" collapsed to 0.8 % of the mass) and **more than
doubling the energy distance** (33× against 15×). Its pixel moments are also further from the
data's (0.1635/0.3360 against 0.1381/0.3142 for real's 0.1325/0.3105).

▶ So the metric that reads most like "sample quality" is the one that ranks these two backwards.
This is the concrete instance of what `planning/diffusion_2d_demo.md` §5.7 argued from the
checkerboard target: a scorer with one number passes models that a scorer with a coverage term and
a distribution term catches.

⚠ **Coverage moved non-monotonically across the ladder**: 9/10 → 8/10 → 9/10 → 10/10. Coverage and
per-sample quality are not the same axis, which is the reason both are reported.

## 2. Why the checkpoint on disk was broken

`pixel_range_uncentred_3ep.log`. Generated pixels ran **[-4.913, 11.866]**, mean 0.0146, sd 0.1788
against real MNIST's [0, 1], 0.1325, 0.3105. That is why the grid rendered as sparse scribbles: the
PPM writer clamps to [0, 1] and almost everything landed at 0.

Two causes, both already written in the tree and neither previously connected to an output:

* ⛔ **`MainMnistDdpmTrain` never centred its data.** Its own comment said so — *"x_0 batch
  (already [0, 1]; not centered to [-1, 1] for MVP)"*. All three CIFAR DDPM trainers do
  `F32.scaleShift trainImgRaw 2.0 (-1.0)` on the line that matters. The reverse process drives
  toward `N(0, I)`; the data it was fitted against had mean 0.13 on [0, 1].
* ⛔ **The default was 3 epochs** — and `demos/README.md` has documented the recipe as
  `lake exe mnist-ddpm-train data 50` all along, so the code default and the documented recipe
  disagreed. That gap is what left a 3-epoch stub on disk under a book section asserting the
  sampler produces digits. ▶ On the evidence above this is the LARGER of the two causes.

⭐ **The scoring path was cleared before any of this was written down.** `lake exe
mnist-ddpm-sample` on the same checkpoint produced the same scribbles —
`reference_sampler_grid_uncentred_3ep.png`. The bug was in the model, not the new driver.

## 3. What the negative control bought, which is more than it cost

⚠⚠ **The classifier is 58 % confident on pure noise.** So confidence has a usable range of 58 to
99, not 0 to 100. The 50-epoch model's 90.94 % is high in that range; read against 0 it would look
mediocre.

⚠⚠ **And on the broken checkpoint the mass imbalance was partly the classifier's, not the
model's.** Noise puts **55.0 %** of its mass on "5"; the uncentred model put 41.8 % there. Without
the noise arm the obvious reading is "the model is obsessed with fives" — with it, most of that
skew is the classifier's own prior on off-distribution input. ▶ A per-class mass table is not
interpretable without a negative control beside it.

```
per-class mass, 50 ep centred :   6.2   5.3  12.1  11.9  12.6  11.6   8.5   7.4  16.0   8.4
per-class mass, 3 ep uncentred:   4.0   3.8  16.1   8.9   1.6  41.8   1.1  20.9   1.6   0.3
per-class mass, noise         :   0.4   0.7   7.5  23.2   0.2  55.0   0.3   0.8  11.9   0.0
per-class mass, real test     :   9.8  11.3  10.3  10.1   9.8   8.9   9.6  10.3   9.7  10.1
```

⚠ The 50-epoch model is still 5.3–16.0 % against a true ~10 %, so per-class mass is where its
residual now lives. That is the same place the 2-D demo's residual ended up once recall saturated.

## 4. The change, and the trap it would otherwise have set

`F32.scaleShift trainImgRaw 2.0 (-1.0)` at load, and its inverse (`scaleShift x 0.5 0.5`) in BOTH
consumers — the sampler before it renders, the scorer before it classifies.

⚠⚠ **The old and new checkpoints are shape-identical and semantically incompatible.** Loading an
uncentred checkpoint into the new path would double-transform, and nothing downstream would notice:
the parameter count matches, the graph loads, the sampler runs. So the spec `name` now carries
`centered`, which changes `buildPrefix`, which changes the checkpoint path. The incompatibility is
structural rather than a comment. Both checkpoints are archived here.

## 5. The figures

* `class_grid_centred_50ep.png` — ten rows, row *d* holding the twelve samples the classifier is
  most confident are a *d*. Coverage is the count of non-empty rows and per-class quality is the
  row itself, so one figure carries both. Every row is full and legible.
* `class_grid_uncentred_50ep.png` — the ablation arm. Legible digits, and a visibly short "1"
  row: the dropped class, readable straight off the figure.
* `class_grid_uncentred_3ep.png` — the same figure on the broken checkpoint, for contrast.
* `reference_sampler_grid_centred_50ep.png` — `mnist-ddpm-sample`'s own 4×4 grid, which now works
  because the inversion landed in it too.

⚠ The four archived checkpoints (`*_params.bin`, `*_bn_stats.bin`) are **local only** —
`.gitignore` excludes `runs/**/*.bin`. Every number here is reproducible from the commands below;
the blobs are a convenience for re-scoring without retraining, not committed evidence.

## 6. Reproducing

```
lake exe mnist-ddpm-train data 50                                     # 50 x 18.6 s
lake exe mnist-ddpm-train data 50 raw                                 # the uncentred arm
LEAN_MLIR_DUMP_PARAMS=.lake/build/cnn_verified_params.bin \
  lake exe mnist-cnn-verified data                                    # 10 ep, ~50 s
lake exe mnist-ddpm-score 1024 50
python3 scripts/mnist_ddpm_score.py
```

15.5 min to train the diffusion model, 50 s the classifier, 13 s to sample 1024 images and classify
12k. The scorer exits non-zero on coverage below 10/10, so it is gateable — but like the 2-D
demo's, nothing calls it yet.

⚠ **No energy threshold is set.** One measurement per arm is a value, not a spread, and a bar drawn
from it would be flaky in the direction that costs the most. The coverage half is the sharp one.

## 7. What this does not do

* **No accuracy number**, because there is no conditioning. That is the second half of §7's
  proposal and it needs a class embedding in the denoiser.
* **No feature-space Fréchet distance.** The energy distance runs on the classifier's 10-d softmax,
  not its 512-d penultimate layer. That layer exists inside `cnn_fwd.mlir` but exposing it means
  editing a committed verified render, which is the wrong trade for one metric.
* ✅ **The sampler step count WAS swept** — `sampler_step_sweep.log`, on the 50-epoch centred model,
  weights held fixed:

  | DDIM steps | coverage | confidence | energy | × floor | pixel mean | pixel sd |
  |---|---|---|---|---|---|---|
  | 50 | 10/10 | 90.94 % | 0.02365 | 15× | 0.1381 | 0.3142 |
  | 200 | 10/10 | 91.06 % | 0.01880 | 12× | 0.1323 | 0.3082 |
  | 500 | 10/10 | 91.40 % | 0.01831 | 12× | 0.1318 | 0.3076 |

  ⭐ **The 2-D demo's finding transfers in DIRECTION but not in size.** There it found 50 measurably
  wrong with the knee near 200 (off-manifold 15.7 % → 5.0 %). Here 50 → 200 buys 15× → 12× and
  200 → 500 buys essentially nothing, so the knee is in the same place and the prize is much
  smaller. ▶ The book may keep saying 50 for this demo, but it can now say so as a measurement.
