# EfficientNet-B0 / ImageNet-1k — 80-epoch bf16 run (2026-06-04→05)

**Final (canonical, full 50,000-image validation on `enet_b0_imagenet_bf16.bin`):**

| Metric | Value |
|--------|-------|
| **Top-1** | **72.31%** (36157/50000) |
| **Top-5** | **90.31%** (45154/50000) |

In-training val (49,896 imgs, tfds drop_remainder) at epoch 80: top-1 72.43%,
top-5 90.33% — agrees with full-50k. **Sweep accuracy leader**: edges ResNet-34
(72.02%) at ~¼ the parameters (5.29M vs 21.8M). A few points under the ~77%
paper number — gap is the missing RMSProp + AutoAugment + original longer
schedule, not precision/pipeline.

## Setup
- Hardware: ares, **all 6× RTX 4060 Ti** (first full 6-GPU run, post-Gen3 BIOS fix).
- Arch: EfficientNet-B0, MBConv + squeeze-excite + swish. **5.29M params**.
- Path: phase-2 Lean→JAX (`jax/MainEfficientNetImagenet.lean`).
- Precision: **bf16** matmul + **bf16Conv** (MBConv expand/depthwise/project cast,
  SE 1×1s + swish/sigmoid stay fp32). Master weights / BN fp32.
- Recipe: SGD + momentum 0.9, peak LR 0.1, batch 252 (6×42), 80 epochs, cosine +
  5-epoch warmup, weight decay 1e-5, label smoothing 0.1, RRC + flip,
  **EMA (decay 0.9999) + stochastic depth (dropPath 0.2)**. No mixup/cutmix.
- Throughput: ~90 ms/step (~8.0 min/epoch); ~10.5 wall-clock hr.

## 6-GPU + Gen3 (the infra results)
- **First full 6-GPU run since the Gen4→Gen3 BIOS fix.** Completed clean.
- **AER: 2 auto-resumes** over ~10.5 hr (epochs 25, 74). Gen3 makes 6-GPU AER
  **much rarer, but not zero** — vs Gen4, where every 6-GPU attempt died in
  ~20–30 s. The watchdog+resume absorbed both; no host reset.
- **4→6 GPU speedup: ~1.17×** (107→90 ms/step). Smaller than the hoped ~1.35×
  because the tfds CPU data pipeline can't fully feed 6 GPUs (GPUs at ~60% util,
  input-starved) — NOT bandwidth (gradient all-reduce is ~2 ms of a ~90 ms step).
  Light nets hit this wall first. Mitigation: more data-loader workers / caching /
  GPU-side JPEG decode / larger per-GPU batch.

## Per-epoch validation (sampled — log truncates on each resume)
val over 49,896 imgs (drop_remainder). Reconstructed from monitoring snapshots:

| Epoch | top-1 | top-5 |  | Epoch | top-1 | top-5 |
|---|---|---|---|---|---|---|
| 2  | 0.0420 | 0.1299 | | 44 | 0.7109 | 0.8970 |
| 3  | 0.1640 | 0.3642 | | 45 | 0.7133 | 0.8975 |
| 4  | 0.2861 | 0.5369 | | 49 | 0.7164 | 0.8993 |
| 5  | 0.3810 | 0.6401 | | 50 | 0.7173 | 0.8998 |
| 6  | 0.4490 | 0.7051 | | 51 | 0.7172 | 0.9007 |
| 7  | 0.4972 | 0.7462 | | 52 | 0.7179 | 0.9012 |
| 12 | 0.6112 | 0.8334 | | 57 | 0.7197 | 0.9018 |
| 13 | 0.6218 | 0.8414 | | 58 | 0.7202 | 0.9017 |
| 14 | 0.6293 | 0.8476 | | 59 | 0.7204 | 0.9022 |
| 15 | 0.6378 | 0.8531 | | 60 | 0.7208 | 0.9025 |
| 19 | 0.6616 | 0.8676 | | 66 | 0.7233 | 0.9026 |
| 20 | 0.6660 | 0.8704 | | 67 | 0.7236 | 0.9034 |
| 21 | 0.6711 | 0.8732 | | 68 | 0.7237 | 0.9031 |
| 22 | 0.6744 | 0.8754 | | 69 | 0.7237 | 0.9032 |
| 26 | 0.6845 | 0.8821 | | 70 | 0.7240 | 0.9030 |
| 27 | 0.6848 | 0.8828 | | 71 | 0.7240 | 0.9033 |
| 28 | 0.6872 | 0.8841 | | 72 | 0.7236 | 0.9034 |
| 29 | 0.6898 | 0.8849 | | 73 | 0.7239 | 0.9035 |
| 34 | 0.6990 | 0.8900 | | 74 | 0.7241 | 0.9034 |
| 35 | 0.7002 | 0.8905 | | 76 | 0.7238 | 0.9036 |
| 36 | 0.7016 | 0.8918 | | 77 | 0.7244 | 0.9033 |
| 37 | 0.7035 | 0.8928 | | 78 | 0.7241 | 0.9034 |
| 42 | 0.7100 | 0.8964 | | 79 | 0.7245 | 0.9034 |
| 43 | 0.7103 | 0.8970 | | 80 | **0.7243** | **0.9033** |

Best epoch ≈ 79 (0.7245). Final epoch 80: 0.7243 (in-train) / **0.7231 full-50k**.

## Files
- `enet_b0_imagenet_bf16.bin` (21.2 MB) — final weights (at /home/skoonce/).
- `_e1.bin … _e80.bin` — per-epoch checkpoints (at /home/skoonce/).
- `scripts/eval_enet_full50k.py` — canonical no-drop full-50k eval.
- `scripts/supervise_enet_b0_80ep_6gpu.sh` — 6-GPU checkpoint + AER-watchdog supervisor.
- `enet_val_curve.tex` — pgfplots chart snippet (this dir).
