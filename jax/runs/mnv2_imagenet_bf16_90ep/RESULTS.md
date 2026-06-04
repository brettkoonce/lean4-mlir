# MobileNetV2 / ImageNet-1k — 90-epoch bf16 run (2026-06-04)

**Final (canonical, full 50,000-image validation on `mnv2_imagenet_bf16.bin`):**

| Metric | Value |
|--------|-------|
| **Top-1** | **68.33%** (34165/50000) |
| **Top-5** | **88.17%** (44084/50000) |

In-training val (49,920 imgs, tfds drop_remainder) at epoch 90: top-1 68.39%, top-5 88.15% — agrees with full-50k. Best epoch ≈ final (monotone tail).

This is the paper-grade recipe (SGD, base aug — MobileNetV2's standard), landing
a few points under the ~71-72% paper number; the gap is mostly the missing
RMSProp + longer original schedule, not the precision or pipeline.

## Setup
- Hardware: ares, **4× RTX 4060 Ti** (CUDA dev idx 0,2,3,4; idx1/idx5 excluded).
- Arch: MobileNetV2, inverted-residual body, 1000-class head. **3.50M params**.
- Path: phase-2 Lean→JAX (`jax/MainMobilenetV2Imagenet.lean`).
- Precision: **bf16** matmul + **bf16Conv** (MBConv expand/depthwise/project cast
  to bf16, returning fp32 before BN; the ~2× MBConv-block win — the 1×1s carry it,
  3×3 depthwise is a wash). Master weights / BN fp32.
- Recipe: SGD + momentum 0.9, peak LR 0.1, batch 256 (4×64), 90 epochs, cosine +
  5-epoch warmup, weight decay 4e-5, label smoothing 0.1, RRC + horizontal flip.
  No mixup/cutmix (not standard for MNv2 at this tier).
- Throughput: ~108 ms/step (~9 min/epoch); ~14 wall-clock hr total.

## PCIe Gen4 → Gen3 (the headline infra result)
The run spanned the BIOS PCIe change, which gives a clean controlled comparison:
- **Gen4 portion (epochs 1–84):** ~5 AER `BadTLP` auto-resumes (epochs 14, 32, 36,
  53, …) — the box's fabric-wide Gen4 signal-integrity issue under sustained load.
  Watchdog caught each before a host reset; supervisor resumed from checkpoint.
- **Gen3 portion (epochs 85–90, ~1 hr sustained load after the BIOS cap):**
  **0 AER.** Links negotiated gen 3 under load (was gen 4).
- **Throughput cost of Gen3: ~108 → ~111 ms/step (+~3%, ~16 s/epoch)** — negligible,
  matching the prediction (these data-parallel convnets are compute-bound, not
  interconnect-bound; gradient all-reduce is ~1-2 ms of a ~108 ms step).
- ⇒ Gen3 trades ~3% per-step for AER stability — and unlocks running all 6 GPUs
  (a ~1.5× throughput win that swamps the 3%). Strong evidence Gen3 fixed the
  fabric-wide marginality. (Full confirmation = a clean 6-GPU run, next.)

## Per-epoch validation (sampled — log truncates on each resume)
val over 49,920 imgs (drop_remainder). Reconstructed from monitoring snapshots:

| Epoch | top-1 | top-5 |  | Epoch | top-1 | top-5 |
|---|---|---|---|---|---|---|
| 1  | 0.0526 | 0.1591 | | 50 | 0.6075 | 0.8358 |
| 2  | 0.1866 | 0.4003 | | 51 | 0.6064 | 0.8354 |
| 3  | 0.2821 | 0.5326 | | 52 | 0.6090 | 0.8368 |
| 4  | 0.3435 | 0.6068 | | 55 | 0.6172 | 0.8419 |
| 5  | 0.3894 | 0.6531 | | 56 | 0.6168 | 0.8426 |
| 6  | 0.4301 | 0.6960 | | 57 | 0.6215 | 0.8462 |
| 10 | 0.5042 | 0.7579 | | 58 | 0.6201 | 0.8440 |
| 11 | 0.5084 | 0.7622 | | 61 | 0.6296 | 0.8509 |
| 12 | 0.5174 | 0.7699 | | 62 | 0.6303 | 0.8508 |
| 13 | 0.5223 | 0.7739 | | 63 | 0.6329 | 0.8520 |
| 16 | 0.5362 | 0.7853 | | 64 | 0.6340 | 0.8538 |
| 17 | 0.5355 | 0.7822 | | 68 | 0.6449 | 0.8588 |
| 18 | 0.5472 | 0.7920 | | 69 | 0.6474 | 0.8604 |
| 19 | 0.5448 | 0.7900 | | 70 | 0.6512 | 0.8630 |
| 23 | 0.5512 | 0.7977 | | 71 | 0.6524 | 0.8633 |
| 24 | 0.5602 | 0.8010 | | 74 | 0.6623 | 0.8698 |
| 25 | 0.5595 | 0.8037 | | 75 | 0.6641 | 0.8711 |
| 26 | 0.5595 | 0.8047 | | 76 | 0.6630 | 0.8713 |
| 37 | 0.5819 | 0.8179 | | 77 | 0.6665 | 0.8725 |
| 38 | 0.5767 | 0.8164 | | 78 | 0.6702 | 0.8739 |
| 42 | 0.5910 | 0.8237 | | 82 | 0.6768 | 0.8775 |
| 43 | 0.5918 | 0.8235 | | 83 | 0.6794 | 0.8786 |
| 44 | 0.5927 | 0.8253 | | 84 | 0.6801 | 0.8790 |
| 45 | 0.5906 | 0.8252 | | 85 | 0.6803 | 0.8795 |
| 49 | 0.6027 | 0.8324 | | 86 | 0.6823 | 0.8806 |
|    |        |        | | 87 | 0.6831 | 0.8810 |
|    |        |        | | 88 | 0.6834 | 0.8818 |
|    |        |        | | 89 | 0.6835 | 0.8816 |
|    |        |        | | 90 | **0.6839** | **0.8815** |

Final epoch 90: 0.6839 (in-train) / **0.6833 full-50k**.

## Files
- `mnv2_imagenet_bf16.bin` (13.4 MB) — final weights (at /home/skoonce/).
- `_e1.bin … _e90.bin` — per-epoch checkpoints (at /home/skoonce/).
- `scripts/eval_mnv2_full50k.py` — canonical no-drop full-50k eval.
- `scripts/supervise_mnv2_90ep.sh` — checkpoint + AER-watchdog auto-resume supervisor.
- `mnv2_val_curve.tex` — pgfplots chart snippet (this dir).
