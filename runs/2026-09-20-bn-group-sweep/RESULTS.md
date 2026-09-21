# BN statistic-group sweep — Imagenette R50, **no-augmentation probe**

**2026-09-20.** `planning/global_bn_verified.md` §1, first run ever of
`jax/scripts/bn_sharding_demo.py`. Four configs, one per 4060 Ti, 80 epochs, fully annealed.

⚠⚠ **READ THE REGIME BEFORE THE NUMBERS.** This round used
`probe/probe_resnet50_imagenette_noaug.py`, the demo's hard-coded default and the model
`probe/benchmark.py` uses for throughput. It differs from `probe_resnet50_imagenette.py` by
exactly one line — no `augment_batch` (the pad-252/crop-224 random crop), flips only — and
`conv_bn` is byte-identical between them. R50's 23.5 M parameters on 9,469 images without a crop
**overfit**: the global-BN arm lands at 38.70 % with eval loss 5.18 against `ln 10 = 2.30`. In
that regime BN noise is the only regulariser present, so the gap below is **inflated and does not
transfer**. Every reference recipe in this tree runs `augment := true`. The augmented re-run is
`runs/2026-09-20-bn-group-aug/`; **quote that one**, not this.

What this round IS good for: the *shape* of the curve, which is clean, monotone, and worth having.

## Setup

Paired by construction — identical init (`PRNGKey 314159`), identical shuffle and flip stream
(`RandomState(42)`), identical schedule. The only difference between arms is how many images each
BN layer averages over. `GHOST_BN_GROUPS=k` reshapes `[N,C,H,W]` to `[k, N/k, C,H,W]` and reduces
over the inner axes, which is bit-identical to what `k` physical shards compute.

Batch 192, Adam lr 1e-3, wd 1e-4, 3-epoch warmup, cosine to 0, 80 epochs, f32, one card each.

⭐ **Eval is pinned to global BN in every arm** (fixed 2026-09-20). The probe has no `training`
flag, so before the fix `groups=k` also moved the *eval* BN group to `512/k` and each arm differed
from its neighbour in two places at once. Verified: across arms the eval path now differs by
exactly 0, `groups=1` stays bit-identical to the stock global `conv_bn`, and training genuinely
differs.

## Results

| shards `k` | BN group | top-1 | top-5 | Δ top-1 vs global | sec |
|---|---|---|---|---|---|
| 1 | **192 (global)** — what the JAX reference does | 38.70 | 77.96 | — | 4,503 |
| 2 | 96 | 40.79 | 78.10 | **+2.09** | 4,516 |
| 4 | **48** — the fourfold split the verified path has | 43.30 | 80.19 | **+4.60** | 4,390 |
| 8 | 24 | 47.32 | 80.44 | **+8.62** | 4,292 |

**Monotone, and the small group wins.** Smaller BN groups mean noisier per-batch statistics, which
in an overfitting regime is free regularisation — so the direction is unsurprising and the
magnitude is a property of the regime, not of BatchNorm.

⚠ **Sign note for the ImageNet question.** Here the *verified* side's grouping (48) would look
**+4.60 better** than the reference's (192). On the real pairs the reference won slightly at
convergence (R34 $-0.10$, MNv2 $+0.01$). Opposite sign. That is the regime difference doing the
talking, and it is exactly why this round is not quotable for §3.5.

## Operational

⛔⛔ The first launch had to be killed: one of four simultaneously-starting JAX processes lost its
CUDA init ("a CUDA-enabled jaxlib is not installed. Falling back to cpu"), then ran at 1316 % CPU
and 56 GB host RAM and dragged the three genuine GPU runs from 100 % GPU to 0 %. The demo now
aborts unless the backend is `gpu` (`ALLOW_CPU=1` overrides). See memory
`concurrent-jax-cpu-fallback`. Use `python -u`: block-buffered logs showed nothing for 20 minutes
while this was happening.

## Provenance

`GHOST_BN_GROUPS={1,2,4,8} SEED=0 EPOCHS=80 CUDA_VISIBLE_DEVICES=<n> python -u
jax/scripts/bn_sharding_demo.py`, with `PROBE_NOAUG=1` under the current script (this round
predates that flag, when `_noaug` was the hard-coded default). Logs `g{k}_s0.log`.
