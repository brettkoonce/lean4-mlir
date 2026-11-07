# ResNet-50 RSB-A3, JAX path, Mixup/CutMix drawn per 256-row chunk — mars (4× RTX 3060), 2026-11-06

The A/B on the JAX reference's gap to the paper at A3: 77.60 / 77.66 / 77.64 over three seeds
(`../2026-10-30-r50-a3-seeds/RESULTS.md`) against timm's 78.1, with a 0.06 seed range. One change from
`rsb-faithful`: `mixChunks := 8`, so the step's 2048 rows take eight independent Mixup/CutMix draws
(coin, λ and box per 256-row chunk, partners flipped within the chunk) instead of one. timm draws once
per GPU batch. Everything else is `rsb-faithful`, the 0.2 BCE target threshold included.

* conf `scripts/jobs/r50-a3mix8-jax-4gpu.conf` (RUN_SEED 1), trainer
  `jax/generated/generated_resnet50_imagenet_rsbfaithful_mix8.py`
* one attempt, fresh start 01:21:48 UTC, ✅ complete 15:20:11 UTC (13.97 h, ~485 s train an epoch)
* lineage `/home/skoonce/r50_a3mix8_jax/`

## Top-1, all 50,000 val images

| run | top-1 | top-5 |
|---|---|---|
| **JAX, mix8, seed 1** | **78.14** (39,071; re-scored 39,072) | 93.70 |
| JAX, one draw a step, seeds 1 / 2 / 3 | 77.60 / 77.66 / 77.64 | — / 93.65 / 93.53 |
| verified (one draw per 512 micro-batch), seeds 1 / 2 / 3 | 78.08 / 77.82 / 77.77 | — / 93.81 / 93.70 |
| timm RSB-A3 (paper) | 78.1 | — |

Re-score: `jax/scripts/eval_full50k.py`, f32 forward, PROTOCOL=train, BATCH=200, one card
(`rescore_f32.log`, `bitmaps/a3mix8_jax_f32.bin`); within one image of the trainer's own e100 count.

## Paired tests (`scripts/demos/mcnemar.py`, the mix8 bitmap against each)

| against | discordant | McNemar exact p |
|---|---|---|
| JAX s2 | 4,037 | **0.0002** |
| JAX s3 | 4,057 | **0.00009** |
| verified 10-09 (77.91; see below) | 3,929 | 0.069 |
| verified s2 | 3,943 | **0.011** |
| verified s3 | 4,052 | **0.004** |

Seed 1's JAX bitmap is not on mars (it was scored on atlas from the ares checkpoint), so the same-seed
pair is not here; the seed-2 and seed-3 JAX pairs stand in for it. Nor is the verified seed-1 bitmap
(78.08, `../2026-10-12-r50-a3-rsb-bf16/`, ares): the "10-09" row is `../2026-10-09-r50-a3-verified-bf16-100ep/`,
the same verified A3 recipe (4 × 128, threshold 0.2) at 77.91 on mars — first written up here as "verified
s1", which it is not.

## Reading

* +0.50 over the three-seed JAX mean, about eight times that arm's seed range, and on the paper's
  number. The JAX reference's A3 shortfall was the draw granularity, not the BCE threshold.
* The verified arm draws per 512 and sat between the two (mean 77.89), consistent with the same cause.
* One seed of the new arm. The claim "per-chunk drawing is worth ~0.5" rests on the old arm's tight
  seed spread, not on seeds of this one.
* Next per the decision rule: A2 (−0.43 to its paper number at one draw a step) gets the same change.
