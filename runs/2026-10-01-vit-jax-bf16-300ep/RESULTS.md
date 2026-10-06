# ViT-Ti (DeiT-Ti), `default` recipe, bf16 — JAX reference path, 300 epochs

**✅ 72.072 % top-1 / 91.000 % top-5** (36,036 / 50,000, EMA shadow), epoch 300/300. Landed
2026-10-03 14:09:45 UTC. **45.91 h** trainer time, `attempt 1`, zero restarts, zero thermal rests,
EDAC 0 / 0 throughout.

The JAX half of the post-C6 ViT-Ti pair (`planning/vit_parity_todo.md`, `planning/next_session_vit.md`
D1 = rerun both). Job `scripts/jobs/vit-default-jax-4gpu.conf`, unit `vit-jax`, launched
2026-10-01 16:15:13 UTC on the 4× 3060 box as a fresh start. The verified half is
`runs/2026-10-01-vit-verified-bf16-300ep/` (72.416 / 91.212). It was chained behind this run and
launched 50 s after this run's ✅ COMPLETE.

## The run, from its own output

`lr=0.000500  batch_size=512 (4 devices x 128)  epochs=300  params=5717416` ·
`steps_per_epoch=2502  total_steps=750600` · `repeated-aug=3x`. The recipe is
`vitTinyImagenetConfig`, DeiT-Ti at batch 512:
* AdamW 5e-4, wd 0.05 off norm, bias, pos and CLS
* 5 warmup epochs, then cosine to a **1e-5 floor** (DeiT `--min-lr`)
* clip 1.0
* mixup 0.8 / cutmix 1.0 with **label smoothing 0.1 folded into the mixed target** (4b765be7)
* RandAugment m9 mstd0.5, random erasing 0.25 (C6 `pixel` mode), drop-path 0.1
* EMA 0.99996
* timm/DeiT trunc_normal(0.02) init including CLS and pos-embed
* **LayerNorm ε 1e-6**, with the **patch embed and head in fp32** (`f32StemHead`); every other matmul is bf16

The conf's precheck asserted the smoothing fold, the init, ε 1e-6, the 1e-5 floor and the two fp32
matmuls against the emitted file.

Trainer: `jax/.lake/build/generated_vit_tiny_imagenet.py`, byte-identical to the committed
`jax/generated/` copy (`44abf272`), on `/home/skoonce/.venv-cuda` (jax 0.11.1, cuda13).

## Curve (top-1 / top-5, EMA shadow, scored every 5th epoch)

| epoch | 5 | 10 | 25 | 50 | 100 | 150 | 200 | 250 | **300** |
|---|---|---|---|---|---|---|---|---|---|
| top-1 | 13.99 | 30.31 | 48.24 | 57.05 | 63.15 | 66.31 | 68.75 | 71.02 | **72.07** |
| top-5 | 31.79 | 54.41 | 73.34 | 80.79 | 85.59 | 87.63 | 89.25 | 90.40 | **91.00** |

The full table is in `vit_jax_curve.csv`: train loss and lr every epoch, top-1/top-5 on the 60
scored epochs. The tail sits on the 1e-5 floor. The scored epochs e280–300 average
**71.955 / 90.920** (range 71.81–72.07), so the endpoint is the top of a still-rising tail, about
0.1 above its mean.

## Re-scored per image (2026-10-06, after the run)

Per-image scoring of `vit_tiny_imagenet_e300.state.npz`. This adds three things the trainer's own
eval cannot: the live weights, an fp32 eval, and a bitmap to pair with the verified run. Bitmaps
are in `bitmaps/` (local only), produced by `score_arms.py` on one GPU at batch 250.

| | bf16 matmuls (the trainer's eval) | fp32 matmuls (the verified path's eval) |
|---|---|---|
| EMA shadow | 72.054 / 90.984 | 72.062 / 91.000 |
| live weights | 71.824 / 90.872 | 71.860 / 90.878 |

* **Eval precision** (`vit_parity_todo.md` G1) is worth +0.008 on the EMA and +0.036 on live, i.e.
  nothing. The 72.07 needs no fp32 asterisk.
* **EMA is worth +0.23** over the live weights (McNemar p = 0.002, 739 vs 624 discordant). DeiT
  reports the live model (P-C), so **71.82–71.86 is the number comparable to DeiT-Ti's 72.2**.
* The re-score's EMA bf16 is 9 images (0.018 pt) below the trainer's own 36,036. The trainer
  shards eval over 4 devices; this scorer runs on 1. bf16 reduction order moves a handful of
  near-ties.

## Against what came before

* The previous JAX reference, **72.31 / 91.12** (`deit-init`, 2026-08-29, the book's §9.6 number),
  trained one day before 4b765be7. Its mixup/cutmix built targets from a raw one-hot, so every
  step trained at label smoothing 0. It also trained with the pre-C6 augmentation, ε 1e-5, no LR
  floor, and a bf16 patch embed and head. This run changes all of those at once, so the −0.24
  compares two recipes, not a re-run.
* DeiT-Ti paper: 72.2 / 91.1 (live model). This run's live weights: −0.34 / −0.23.

## Pace

**545.3 s/epoch** of training, flat (539.5–557.6), 220 ms/step. Eval runs every 5th epoch (~25 s).
That matches the 2026-09-26 probe (220.3 ms/step). The conf's ~44.5–46 h held at 45.91 h. The
pre-C6 run took 34.2 h, so C6 costs 1.34× here.

## Artifacts

* Checkpoints outside the repo, in `/home/skoonce/vit_tiny_default_300ep/`:
  * EMA weights for every epoch, `vit_tiny_imagenet_e{1..300}.bin` (22.9 MB each)
  * the full train state `(params, opt_state, ema_params)` for e298–300
  * the final `vit_tiny_imagenet.{bin,state.npz}`
* `attempt.log` / `full.log` / `master.log`, `epoch_clock.tsv`, `edac.tsv`, the two watcher scripts
* `vit_jax_curve.csv`, written by `summarize.sh`
* `score_arms.py` and `bitmaps/jax_e300_{ema,live}_{bf16,f32}.bin` + `jax_e300_labels.bin`
  (LOCAL ONLY, since `runs/**/*.bin` is gitignored; regenerate with `score_arms.py`)

## Reproduce

    bash runs/2026-10-01-vit-jax-bf16-300ep/summarize.sh
    cd jax && GEN=.lake/build/generated_vit_tiny_imagenet.py \
      CKPT=/home/skoonce/vit_tiny_default_300ep/vit_tiny_imagenet_e300.state.npz \
      OUT=../runs/2026-10-01-vit-jax-bf16-300ep/bitmaps/jax_e300 CUDA_VISIBLE_DEVICES=0 \
      /home/skoonce/.venv-cuda/bin/python3 ../runs/2026-10-01-vit-jax-bf16-300ep/score_arms.py
