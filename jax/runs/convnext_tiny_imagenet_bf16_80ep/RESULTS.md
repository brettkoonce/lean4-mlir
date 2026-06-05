# ConvNeXt-T / ImageNet-1k — 80-epoch bf16 run (2026-06-04→05)

**Final (canonical, full 50,000-image validation on `convnext_tiny_imagenet_bf16.bin`, EMA weights):**

| Metric | Value |
|--------|-------|
| **Top-1** | **75.93%** (37965/50000) |
| **Top-5** | **92.27%** (46134/50000) |

In-training val (49,896 imgs, drop_remainder): peaked ~76.3% (epoch 48), settled
~75.9% at the LR floor (mild late drift; val_loss crept up — expected without the
full aug pack). **Sweep accuracy leader** by a wide margin (next: R34 72.0%,
ENet 72.3%). Under ConvNeXt-T's ~82% paper number — gap is geometric AutoAugment +
the 80→300-epoch schedule (EMA, stochastic depth, LayerScale, AdamW all present).

## Setup
- Hardware: ares, **all 6× RTX 4060 Ti** (post-Gen3 BIOS fix). **0 AER, 0 resumes
  the entire ~15.5 hr run** — cleanest 6-GPU run yet (heavier net, but no AER trigger).
- Arch: ConvNeXt-T — patchify stem, depthwise-7×7 + channel-LN + inverted-bottleneck
  + GELU + LayerScale, dedicated 2×2 downsamples. **28.6M params** (heaviest in sweep).
- Path: phase-2 Lean→JAX (`jax/MainConvNeXtImagenet.lean`), first JAX-path ConvNeXt.
- Precision: **bf16** matmul + **bf16Conv** (dw-7×7 + 1×1 expand/project; channel-LN
  + GELU + LayerScale stay fp32). Master weights fp32.
- Recipe: AdamW, peak LR 4e-4 (= 4e-3@4096 scaled), batch 252 (6×42), 80 epochs,
  cosine + 5-epoch warmup, weight decay 0.05, label smoothing 0.1, **grad-clip 1.0**,
  RRC + flip, **EMA (0.9999) + stochastic depth (dropPath 0.1)**. No mixup/cutmix.
- Throughput: ~143 ms/step (~12.6 min/epoch); ~15.5 wall-clock hr.

## 6-GPU scaling (the predicted result)
- **4→6 GPU speedup: ~1.27×** (185→143 ms/step) — **better than ENet's 1.17×**, as
  predicted: ConvNeXt is heavier/more compute-bound (GPUs hit 95–100% util vs ENet's
  ~65% input-starved), so more of the added GPUs actually turns. Still under the ideal
  1.5× because the tfds CPU data pipeline remains the partial ceiling.
- **Gen3 reliability: 0 AER over the full run** (vs ENet's 2, vs Gen4's death-in-30s).

## Per-epoch validation (sampled — live weights, val=49,896 drop_remainder)
EMA catches up slowly early (decay 0.9999) — the low E1–5 val is the EMA shadow lagging;
train loss is the real early signal. Reconstructed from monitoring snapshots:

| Epoch | top-1 | top-5 |  | Epoch | top-1 | top-5 |
|---|---|---|---|---|---|---|
| 2  | 0.0143 | 0.0518 | | 41 | 0.7602 | 0.9264 |
| 3  | 0.0285 | 0.0868 | | 42 | 0.7603 | 0.9261 |
| 4  | 0.0405 | 0.1137 | | 43 | 0.7606 | 0.9266 |
| 5  | 0.1120 | 0.2699 | | 44 | 0.7602 | 0.9267 |
| 7  | 0.3364 | 0.5874 | | 46 | 0.7611 | 0.9261 |
| 8  | 0.4150 | 0.6714 | | 47 | 0.7615 | 0.9262 |
| 9  | 0.4869 | 0.7359 | | 48 | **0.7628** | **0.9268** |
| 10 | 0.5526 | 0.7895 | | 49 | 0.7623 | 0.9262 |
| 11 | 0.6044 | 0.8281 | | 51 | 0.7627 | 0.9262 |
| 12 | 0.6392 | 0.8521 | | 52 | 0.7622 | 0.9264 |
| 13 | 0.6644 | 0.8682 | | 53 | 0.7623 | 0.9267 |
| 14 | 0.6802 | 0.8794 | | 54 | 0.7620 | 0.9266 |
| 16 | 0.7020 | 0.8925 | | 56 | 0.7623 | 0.9258 |
| 17 | 0.7097 | 0.8965 | | 57 | 0.7618 | 0.9259 |
| 18 | 0.7158 | 0.9011 | | 58 | 0.7614 | 0.9254 |
| 19 | 0.7211 | 0.9044 | | 59 | 0.7612 | 0.9257 |
| 21 | 0.7294 | 0.9101 | | 60 | 0.7613 | 0.9256 |
| 22 | 0.7323 | 0.9127 | | 61 | 0.7607 | 0.9252 |
| 23 | 0.7353 | 0.9142 | | 62 | 0.7603 | 0.9247 |
| 24 | 0.7375 | 0.9157 | | 63 | 0.7599 | 0.9243 |
| 26 | 0.7431 | 0.9186 | | 66 | 0.7598 | 0.9244 |
| 27 | 0.7441 | 0.9194 | | 67 | 0.7592 | 0.9243 |
| 28 | 0.7465 | 0.9198 | | 68 | 0.7602 | 0.9243 |
| 29 | 0.7480 | 0.9208 | | 69 | 0.7599 | 0.9236 |
| 31 | 0.7505 | 0.9226 | | 70 | 0.7603 | 0.9238 |
| 32 | 0.7515 | 0.9232 | | 71 | 0.7600 | 0.9235 |
| 33 | 0.7522 | 0.9238 | | 79 | 0.7594 | 0.9224 |
| 34 | 0.7547 | 0.9234 | | 80 | 0.7593 | 0.9224 |
| 36 | 0.7563 | 0.9243 | |    |        |        |
| 37 | 0.7563 | 0.9251 | |    |        |        |
| 38 | 0.7573 | 0.9254 | |    |        |        |
| 39 | 0.7586 | 0.9258 | |    |        |        |

Best (live) epoch 48: 76.28%. Final full-50k (EMA): **75.93%**.

## Files
- `convnext_tiny_imagenet_bf16.bin` (114 MB) — final weights (at /home/skoonce/).
- `_e1.bin … _e80.bin` — per-epoch checkpoints (at /home/skoonce/).
- `scripts/eval_convnext_full50k.py` — canonical no-drop full-50k eval.
- `scripts/supervise_convnext_t_80ep_6gpu.sh` — 6-GPU checkpoint + AER-watchdog supervisor.
- `convnext_val_curve.tex` — pgfplots chart snippet (this dir).
