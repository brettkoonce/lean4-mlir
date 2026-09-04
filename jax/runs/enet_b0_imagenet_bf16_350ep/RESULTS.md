# EfficientNet-B0 / ImageNet-1k — 350-epoch paper-faithful bf16 run (2026-09-02→04)

**Final (canonical, full 50,000-image validation, EMA weights):**

| Metric | Value |
|--------|-------|
| **Top-1** | **77.15%** (38573/50000) |
| **Top-5** | **93.30%** (46652/50000) |

Matches EfficientNet-B0's published **77.1% / 93.3%**. In-training val at epoch
350 reads 77.18% / 93.31% over the same 50,000 — the 0.03 gap is bf16
batch-shape noise (rescore ran at batch 200 against the run's 256), not drift
between the checkpoint and the log.

⭐ **EMA is worth +0.82 points.** Same checkpoint, raw (non-averaged) weights:
**76.33% / 92.90%**. The `.bin` and every `_e{N}.bin` hold the EMA params.

## Setup
- Hardware: **4× RTX 3060** (12 GB), CUDA 13.3, jax 0.11.1 / `xla_cuda13`.
- Arch: EfficientNet-B0, MBConv + squeeze-excite + swish. **5,288,548 params**.
- Path: phase-2 Lean→JAX (`jax/MainEfficientNetImagenet.lean`, `full` recipe).
- Precision: **bf16** matmul + **bf16Conv** (MBConv expand/depthwise/project;
  SE 1×1s and the sigmoid gate stay fp32). Master weights / BN fp32.
- Recipe (the paper's own): **RMSProp in TF form** (ρ=0.9, μ=0.9, ε=1e-3,
  `g/√(sq+ε)`, mean-square init 1.0), lr **0.016** at batch 256 (4×64), 5-epoch
  warmup then **exponential decay ×0.97 every 2.4 epochs**, weight decay 1e-5,
  label smoothing 0.1, **AutoAugment** (full ImageNet policy), **stochastic
  depth** 0.2, classifier **dropout** 0.2, **EMA 0.9999 over weights AND BN
  buffers**, running-BN eval.
- Throughput: **94.5 ms/step differenced, 8.0 min/epoch, 46.6 h total.**

## Run notes
- One unplanned interruption at epoch 149 (host power loss). Resumed from the
  epoch-149 `.state.npz` — params, RMSProp accumulators, EMA shadow, BN buffers
  and global step — and epoch 150 returned 75.07% against epoch 149's 75.07%,
  i.e. the resume was exact, not a restarted schedule.
- Zero thermal rests: the supervisor rests on temperature, and at 51–65 °C the
  cards never approached the 80 °C threshold.
- ⚠ This is the first B0 run with `convBnAct := .swish` (`4d79b49a`, 2026-08-30).
  Earlier runs trained the stem and 1×1 head with ReLU, so they are a different
  network and their numbers are not comparable to this one.

## Per-epoch validation (every 10th epoch, all over 50,000)

| Epoch | top-1 | top-5 | | Epoch | top-1 | top-5 |
|---|---|---|---|---|---|---|
| 10 | 0.6169 | 0.8394 | | 190 | 0.7584 | 0.9262 |
| 20 | 0.6635 | 0.8703 | | 200 | 0.7593 | 0.9274 |
| 30 | 0.6883 | 0.8865 | | 210 | 0.7607 | 0.9275 |
| 40 | 0.7002 | 0.8936 | | 220 | 0.7620 | 0.9290 |
| 50 | 0.7078 | 0.8977 | | 230 | 0.7622 | 0.9294 |
| 60 | 0.7170 | 0.9022 | | 240 | 0.7635 | 0.9302 |
| 70 | 0.7195 | 0.9038 | | 250 | 0.7642 | 0.9311 |
| 80 | 0.7231 | 0.9064 | | 260 | 0.7652 | 0.9312 |
| 90 | 0.7312 | 0.9110 | | 270 | 0.7665 | 0.9310 |
| 100 | 0.7314 | 0.9120 | | 280 | 0.7667 | 0.9313 |
| 110 | 0.7357 | 0.9140 | | 290 | 0.7676 | 0.9319 |
| 120 | 0.7415 | 0.9173 | | 300 | 0.7692 | 0.9325 |
| 130 | 0.7433 | 0.9180 | | 310 | 0.7684 | 0.9325 |
| 140 | 0.7484 | 0.9213 | | 320 | 0.7695 | 0.9330 |
| 150 | 0.7507 | 0.9216 | | 330 | 0.7718 | 0.9328 |
| 160 | 0.7530 | 0.9234 | | 340 | 0.7719 | 0.9326 |
| 170 | 0.7547 | 0.9251 | | 350 | 0.7718 | 0.9331 |
| 180 | 0.7565 | 0.9254 | | | | | |

Final epoch 350: 77.18% / 93.31% in-training, **77.15% / 93.30% full-50k rescore**.

## Files
- `/home/skoonce/enet_b0_350_4gpu/efficientnet_b0_imagenet.bin` (21.2 MB) — final
  **EMA** weights. ⚠ Params only: BN running stats are *not* in the `.bin`.
- `..._e{N}.bin` — per-epoch EMA checkpoints (350 of them, ~7.4 GB).
- `..._.state.npz` / `..._e{N}.state.npz` — full resumable state (params, opt,
  EMA weights, EMA BN, BN, step). **Any eval must take `ema_bn` from here.**
- `jax/scripts/supervise_enet_b0_350ep_3060.sh` — the supervisor for this box.
- `enet_val_curve.tex` — pgfplots snippet (this dir), reprinted in Ch 7.
- `training_epochs.log` — all 350 `[Epoch N]` lines; `training_tail.log` — the end.

⛔ `jax/scripts/eval_enet_full50k.py` does NOT run: it calls `m.forward(params, x)`
against the 4-arg `forward(params, x, bn, training)` and sources no BN stats at
all. The rescore above was done with a corrected one-off.
