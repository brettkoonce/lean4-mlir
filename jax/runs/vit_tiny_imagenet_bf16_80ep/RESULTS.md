# ViT-Tiny / ImageNet-1k — 80-epoch bf16 run (2026-05-31 → 2026-06-01)

**Final (canonical, full 50,000-image validation on `vit_tiny_imagenet_bf16.bin`):**

| Metric | Value |
|--------|-------|
| **Top-1** | **65.64%** (32819/50000) |
| **Top-5** | **87.06%** (43531/50000) |

In-training val (49,664 imgs, tfds drop_remainder) at epoch 80: top-1 65.68%, top-5 87.08% — agrees with the full-50k figure.

## Setup
- Hardware: ares, **4× RTX 4060 Ti** (CUDA dev idx 0,2,3,4; idx1/idx5 excluded — PCIe AER).
- Arch: ViT-Tiny / DeiT-Ti — patch16, embed 192, 12 transformer blocks, 3 heads, MLP 768, 1000-class head. **5.72M params**.
- Path: phase-2 Lean→JAX (`jax/MainVitImagenet.lean`), JAX multi-GPU mesh.
- Precision: **bf16** matmul compute (patch-embed, attention QKV/scores/out, MLP, head), fp32 master weights, fp32 LayerNorm/softmax/GELU. (No bf16Conv — ViT has no convolutions.)
- Recipe (DeiT-flavored): AdamW, **LR 5e-4** peak, batch 512 (4×128), 80 epochs, 5-epoch warmup + cosine, weight decay 0.05, label smoothing 0.1, **grad-clip global-norm 1.0**, mixup (α 0.8) + cutmix (α 1.0) alternating, RandAugment (color), random erasing (p 0.25).
- Throughput: ~185 ms/step steady (~7.7 min/epoch); full 80 epochs in ~11 wall-clock hr.

## Stability note (the point of the run)
The DeiT recipe at LR 5e-4 **collapses to chance without gradient clipping** (train loss pins at ln(1000)≈6.9) — documented in `planning/vit_imagenet.md`. With grad-clip 1.0 it trains cleanly: warmup ramped LR through the collapse threshold (~1.6e-4) by epoch 5 and the model kept learning. Confirmed the fix holds on CUDA.

## Reliability
5 AER auto-resumes over the run (epochs 12, 28, 29, 77, 79) — the box's PCIe BadTLP-under-load issue (see memory `reference_ares_pcie_aer`). Each was caught by the watchdog before a host reset and resumed from the latest per-epoch checkpoint; no host reset, no lost accuracy. Supervisor: `jax/scripts/supervise_vit_80ep.sh`.

## Per-epoch validation (SAMPLED)
The per-attempt training log truncates on each of the 5 resumes, so a complete
1–80 sequence isn't recoverable from disk. These are the epochs captured in
live monitoring snapshots (val over 49,664 imgs, drop_remainder):

| Epoch | top-1 | top-5 |  | Epoch | top-1 | top-5 |
|---|---|---|---|---|---|---|
| 1  | 0.0179 | 0.0643 | | 36 | 0.5627 | 0.8054 |
| 2  | 0.0478 | 0.1398 | | 37 | 0.5634 | 0.8066 |
| 3  | 0.0720 | 0.1960 | | 38 | 0.5664 | 0.8086 |
| 4  | 0.1064 | 0.2620 | | 46 | 0.5911 | 0.8249 |
| 5  | 0.1534 | 0.3391 | | 52 | 0.6078 | 0.8388 |
| 6  | 0.2005 | 0.4179 | | 53 | 0.6125 | 0.8403 |
| 7  | 0.2464 | 0.4763 | | 61 | 0.6337 | 0.8550 |
| 8  | 0.2883 | 0.5316 | | 69 | 0.6476 | 0.8654 |
| 13 | 0.4052 | 0.6633 | | 73 | 0.6540 | 0.8697 |
| 14 | 0.4153 | 0.6747 | | 74 | 0.6547 | 0.8695 |
| 15 | 0.4321 | 0.6891 | | 75 | 0.6553 | 0.8704 |
| 16 | 0.4433 | 0.6956 | | 78 | 0.6566 | 0.8707 |
| 20 | 0.4787 | 0.7348 | | 79 | 0.6567 | 0.8708 |
| 21 | 0.4900 | 0.7421 | | 80 | **0.6568** | **0.8708** |
| 22 | 0.4964 | 0.7473 | |    |        |        |
| 23 | 0.5011 | 0.7547 | |    |        |        |
| 24 | 0.5027 | 0.7540 | |    |        |        |
| 30 | 0.5326 | 0.7791 | |    |        |        |
| 34 | 0.5510 | 0.7937 | |    |        |        |
| 35 | 0.5570 | 0.8008 | |    |        |        |

Final epoch 80: 0.6568 (in-train) / **0.6564 full-50k**. Monotone climb, no
collapse; curve flattens at the LR floor (epochs ~73–80 within 0.3%).

## Files
- `vit_tiny_imagenet_bf16.bin` (22.9 MB) — final weights (at /home/skoonce/).
- `_e1.bin … _e80.bin` — per-epoch checkpoints (at /home/skoonce/).
- `scripts/eval_vit_full50k.py` — canonical no-drop full-50k eval.
- `scripts/supervise_vit_80ep.sh` — checkpoint + AER-watchdog auto-resume supervisor.
- `vit_val_curve.tex` — pgfplots chart snippet (this dir).
