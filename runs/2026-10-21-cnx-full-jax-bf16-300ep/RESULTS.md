# ConvNeXt-T, the official recipe (facebookresearch/ConvNeXt's own), bf16 — JAX reference path, 300 epochs

**✅ 81.990 % top-1 / 95.87 % top-5** (40,995 / 50,000, EMA shadow, fp32 eval), epoch 300/300,
95 % CI 81.65–82.32.
* Landed 2026-10-24 15:57:19 UTC.
* **74.07 h** of trainer time, on attempt 1.
* Zero restarts, zero thermal rests, EDAC 0 / 0 throughout, no NaN.

The JAX half of the official-recipe ConvNeXt-T pair (`planning/recipe_prep.md` §4; ch 8's marker).
The verified half, `runs/2026-10-21-cnx-verified-bf16-300ep/`, was chained behind this run
(`chain_after_jax.sh`) and launched 2026-10-24 15:58:39 UTC. The chain's forced resume test (stop
mid-epoch 3, relaunch) passed: the relaunch resumed from the epoch-2 checkpoint.
* Job `scripts/jobs/cnx-default-jax-4gpu.conf`, unit `cnx-off-jax`, launched 2026-10-21 13:52:57 UTC
  on the 4× 3060 box (mars), tree at `b7af4de3`, unchanged for the whole run.
* ⛔ The unit carried `XLA_PYTHON_CLIENT_MEM_FRACTION=0.97` (`LAUNCH.md`). At XLA's default 0.75
  arena the first `jit_train_step` died RESOURCE_EXHAUSTED twice on the 12 GB cards
  (`oom_default_arena/`); the conf had been smoked only on ares (16 GB).
* Fresh start: `/home/skoonce/cnx_t_jax_2611/` was a new lineage. The 09-04 run's checkpoints
  (`/home/skoonce/convnext_t300_3060/`) are untouched.

## The run, from its own output

* `lr=0.004000  batch_size=4096 (4 devices x 1024)  epochs=300  params=28589128`
* `steps_per_epoch=312  total_steps=93600`
* `grad-accum=16x  micro_batch=256 (4 devices x 64)  -> effective_batch=4096  (LR targets effective)`
* `eval: f32 forward (evalF32)` on all 300 epochs; lr ends at 1e-06 (`--min_lr`)

Trainer: `jax/.lake/build/generated_convnext_tiny_imagenet_full.py`, byte-identical to the committed
`jax/generated/` copy, on `/home/skoonce/.venv-cuda`.

## Curve (top-1 / top-5, EMA shadow, scored every epoch)

| epoch | 10 | 25 | 50 | 60 | 75 | 100 | 150 | 200 | 250 | **300** |
|---|---|---|---|---|---|---|---|---|---|---|
| top-1 | 0.54 | 0.73 | 6.37 | 26.99 | 57.42 | 73.03 | 78.63 | 80.55 | 81.59 | **81.99** |
| top-5 | 2.22 | 2.81 | 23.32 | 52.31 | 80.90 | 91.50 | 94.46 | 95.40 | 95.90 | **95.87** |
| train loss | 5.365 | 4.298 | 3.848 | 3.806 | 3.737 | 3.638 | 3.435 | 3.179 | 3.021 | 2.869 |

* The last 25 epochs average **81.933 / 95.892** (range 81.87–81.99; best 81.992 at e297). The curve
  is flat at the end.
* ⚠ **The first ~50 epochs score near chance, and that is the recipe, not a fault.** `ModelEma` runs at
  0.9999 constant from step 0, and at batch 4096 there are only 312 optimizer steps an epoch. The
  initial weights' share of the shadow is 0.9999^(312·e): 55 % at e19, 21 % at e50, 4 % at e100. The
  scored model climbs once that share is gone (e48 4.4 % → e64 37.1 % → e92 70.4 %). Train loss fell
  steadily throughout (the train loss is under Mixup/CutMix and label smoothing, so it ends high).
* Matches the paper's ConvNeXt-T 82.1 within the CI.

## Against the 09-04 JAX run (81.53; the global-256 form of the recipe)

That run is the reference the 09-18 verified run (81.302) was paired with. Its per-epoch EMA curve is
`runs/2026-09-18-cnx-verified-300ep/reference_curve.tsv`, which `summarize.sh` reads in place.

| epochs | 101–150 | 151–200 | 201–250 | 251–300 | e276–300 | e300 |
|---|---|---|---|---|---|---|
| mean Δtop-1 | −2.76 | −0.83 | −0.10 | +0.37 | **+0.42** | **+0.46** |

* **+0.46 at the endpoint, +0.42 over the last 25 epochs.** The old run is ahead until ~e240 (its
  EMA had 5,004 steps an epoch to forget the init, against 312 here), and this run passes it in the
  last 60 epochs.
* What changed between the two, per the conf's header: the 4096 effective batch (16 accumulated
  micro-batches) and lr 4e-3 in place of the batch-256 scaling; torchvision's RandomResizedCrop at
  bicubic in place of TF's sampler; the exact erf GELU in place of tanh; the post-C6 data feed. One
  seed each, so none of these is separated.
* The pair this run belongs to is with the verified half on the same recipe, not with 09-04: that
  run landed 82.146 / 95.910, a tie (McNemar EMA p = 0.18, live p = 0.64;
  `runs/2026-10-21-cnx-verified-bf16-300ep/RESULTS.md`).

## Rescored, per image

The final `.state.npz` through `jax/scripts/eval_full50k.py` (`EVAL_DTYPE=f32 PROTOCOL=train
BATCH=256`, `DUMP_CORRECT`), for the McNemar pairing with the verified half:

| arm | top-1 | top-5 | log |
|---|---|---|---|
| EMA (scored) | 40,994 (81.988) — the log's 40,995 within one image | 95.87 | `rescore_ema.log` |
| live weights | 40,961 (81.922) | 95.74 | `rescore_live.log` |

The EMA is worth +0.07 at the end of this recipe.

## Pace

* **888.7 s an epoch** by the epoch clock (min 886, max 892), eval of all 50,000 every epoch
  included; 861.3 s of training (2.76 s a 4096-row optimizer step) + 26.5 s of val.
* 74.07 h in all, against the conf's ~76 h estimate. The 09-04 run took 76.5 h on the same box.

## Artifacts

* Checkpoints, outside the repo in `/home/skoonce/cnx_t_jax_2611/`:
  * the EMA weights for every epoch (`convnext_tiny_imagenet_full_e{1..300}.bin`)
  * the full train state for the last 3 epochs (`_e{298,299,300}.state.npz`)
  * the final `convnext_tiny_imagenet_full.state.npz` and EMA `convnext_tiny_imagenet_full.bin`
* `attempt.log` / `full.log` / `master.log`, `epoch_clock.tsv`, `edac.tsv`, the two watcher scripts,
  `LAUNCH.md`, `oom_default_arena/` (the two failed default-arena attempts)
* `rescore_ema.log` / `rescore_live.log`; their bitmaps `bitmaps/jax_e300_{ema,live}.bin` are local
  only (`runs/**/*.bin` is gitignored)
* `cnx_jax_curve.csv`, written by `summarize.sh`

## Reproduce

    bash runs/2026-10-21-cnx-full-jax-bf16-300ep/summarize.sh
