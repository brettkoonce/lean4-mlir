# Phase-2 (Lean→JAX) step probe for the three skeleton side quests — 2026-08-27

The blueprint carried `[todo: bf16 renderer + benchmarks]` in three side-quest
cost tables (`sec:r50_a2_a1_cost`, `sec:convnext_sb`, `sec:vit_sb`). The verified
renders for those are still short a bf16 emit, but the **JAX peers exist for
every one of them**, so the schedule can be measured instead of guessed. These
are those measurements; they are what the three sections now print.

## Method

`jax/scripts/step_probe.py`, which imports a `generated_*.py` as a module,
builds params / opt-state / EMA exactly as the trainer's `__main__` does, shards
them the same way, and times real `train_step` (+ `ema_update` where the recipe
has one) on synthetic batches.

    CUDA_VISIBLE_DEVICES=0,1,2,3 STEPS=10 \
      GEN=jax/.lake/build/generated_convnext_s_imagenet.py BATCH=256 ACCUM=1 \
      .venv/bin/python -u jax/scripts/step_probe.py

Four RTX 4060 Ti, bf16 as emitted, jax 0.11.0 from the pinned `.venv`. Median of
seven timed steps after three warmups. Peak is `peak_bytes_in_use` on device 0
against the 11.68 GiB a 16 GB card's BFC allocator actually offers.

⚠ **Compute-only.** The probe never starts the `tf.data` pipeline, so every
figure is a LOWER BOUND on the live run. Measured tax: ConvNeXt-T ~1.10×,
ViT-Ti ~1.47× (the pipeline costs the same per image whatever the model is, so
it weighs most on the cheapest net).

⚠ **The emits were regenerated first.** The `generated_*.py` for all four S/B
nets on disk were from 2026-07-26 and a month of augmentation work had landed
since. `lake exe <net>-imagenet <recipe>` in `jax/` before probing, always.

## Results

| net | recipe | eff. batch | res | ms/step | peak | min/epoch |
|---|---|---|---|---|---|---|
| ConvNeXt-T | default | 256 | 224 | 171.6 | 4.22 GiB | 14.3 |
| ConvNeXt-S | default | 256 | 224 | 260.8 | 6.78 GiB | 21.8 |
| ConvNeXt-B | default | 256 | 224 | 399.3 | 9.56 GiB | 33.3 |
| ViT-Ti | default | 512 | 224 | 123.7 | 2.89 GiB | 5.2 |
| ViT-S | default | 512 | 224 | 289.7 | 7.72 GiB | 12.1 |
| ViT-B | accum (4×128) | 512 | 224 | 700.8 | 5.36 GiB | 29.2 |
| R50 A3 | rsb-faithful (4×512) | 2048 | 160 | 715.3 | 4.30 GiB | 7.5 |
| R50 A2 | a2-accum (4×512) | 2048 | 224 | 1368.3 | 7.61 GiB | 14.3 |

⭐ **Every emitted parameter count printed back the blueprint's `#guard`ed
figure exactly** — 28,587,592 / 50,222,152 / 88,589,416 / 5,717,416 /
22,050,664 / 86,567,656 / 25,557,032. That is what says each row timed the net
it names.

## What the numbers say

- **The renderer's free-vs-costly split does not reach the clock.** ConvNeXt S
  is 1.52× T and B is 1.53× S, each for a 1.76× jump in parameters — even
  though S cost the renderer nothing and B cost it ~27 dimension literals.
- **ViT widens smoothly**: S 2.34× Ti, B 2.42× S.
- **A1 and A2 are the same graph.** The emitted trainers differ in exactly three
  constants (`EPOCHS`, `WD`, Mixup α), so A1 is A2 at 2× the schedule, to the
  digit. One probe prices both.
- **FixRes is worth the ~2× the recipe table claims**: A3's 160² step is 1.91×
  faster than A2's 224² at the same batch.
- ViT-B's peak (5.36 GiB) is BELOW ViT-S's (7.72) because B is micro-batched.
  Global 512 one-shot was run as a control and does NOT fit: peak 11.41 of the
  allocator's 11.68 GiB, `RESOURCE_EXHAUSTED` on a 9.08 GiB request
  (`sb_probe.vit-b-oneshot512.log`). Same figure the 2026-07-25 probe got.

Reproduces the 2026-07-25 table in `planning/vit_convnext_sb_scaleup.md` closely
(ConvNeXt-B 399 vs 395 then, same 9.56 GiB; ViT-B 701 vs 712) on a newer JAX, so
the 0.10.2 → 0.11.0 move did not shift these. ViT-S came in 7% faster (290 vs
313).

## Probe fixes this needed

`jax/scripts/step_probe.py` had gone stale against the emitter and failed on
first contact:

1. `ema_update` grew a `step` argument when the EMA warmup ramp landed → the
   probe died with a `TypeError` on the first timed step. Now dispatches on the
   signature.
2. A recipe with `useEMA := false` (RSB-A3) emits no `ema_update` at all → the
   probe died with an `AttributeError`. Now optional.
3. FixRes recipes train at `_TRAIN_SIZE` (160), not 224; the probe hardcoded a
   224² flat input, which would have timed the wrong net.
