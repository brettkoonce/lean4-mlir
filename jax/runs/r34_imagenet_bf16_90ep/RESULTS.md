# ResNet-34 / ImageNet-1k — 90-epoch bf16 run (2026-05-30 → 2026-05-31)

**Final (canonical, full 50,000-image validation on `r34_imagenet_bf16.bin`):**

| Metric | Value |
|--------|-------|
| **Top-1** | **72.02%** (36011/50000) |
| **Top-5** | **90.62%** (45310/50000) |

In-training val (49,920 imgs, tfds drop_remainder) at epoch 90: top-1 72.08%, top-5 90.63% — agrees with the full-50k figure.

## Setup
- Hardware: ares, **4× RTX 4060 Ti** (CUDA dev idx 0,2,3,4; idx1/idx5 excluded — see PCIe AER note).
- Path: phase-2 Lean→JAX (`jax/MainResnetImagenet.lean`), JAX multi-GPU mesh.
- Precision: **bf16** compute (matmul + conv via `bf16Conv` flag), fp32 master weights.
- Recipe: SGD+momentum 0.9, batch 256 (4×64), 90 epochs, cosine LR peak 0.1 + 5-epoch warmup,
  weight decay 1e-4, label smoothing 0.1, random-crop + horizontal flip.
- Throughput: ~139 ms/step (~10.2 min/epoch); **bf16 1.60× faster than fp32** (223 ms/step), matching the 1.59× conv microbench.
- Wall-clock: ~15 hr for 90 epochs (one PCIe-AER interruption at epoch 85, auto-resumed from checkpoint).

## Per-epoch validation top-1 / top-5
(Epochs 1–85 reconstructed from live monitoring reports — raw stdout for 1–85 was
truncated when the run was resumed at epoch 86; epochs 86–90 are from
`training_epochs86-90.log`.)

| Epoch | top-1 | top-5 |  | Epoch | top-1 | top-5 |
|---|---|---|---|---|---|---|
| 1  | 0.1525 | 0.3544 | | 46 | 0.5876 | 0.8264 |
| 2  | 0.2916 | 0.5547 | | 47 | 0.5892 | 0.8293 |
| 3  | 0.3700 | 0.6413 | | 49 | 0.5934 | 0.8297 |
| 4  | 0.4099 | 0.6826 | | 50 | 0.5932 | 0.8302 |
| 5  | 0.4279 | 0.6958 | | 51 | 0.6014 | 0.8349 |
| 6  | 0.4570 | 0.7241 | | 52 | 0.5988 | 0.8327 |
| 7  | 0.4722 | 0.7363 | | 53 | 0.6053 | 0.8370 |
| 8  | 0.4863 | 0.7474 | | 55 | 0.6114 | 0.8419 |
| 9  | 0.4870 | 0.7488 | | 56 | 0.6124 | 0.8434 |
| 10 | 0.4919 | 0.7576 | | 57 | 0.6168 | 0.8445 |
| 11 | 0.5036 | 0.7657 | | 58 | 0.6169 | 0.8450 |
| 12 | 0.5045 | 0.7657 | | 59 | 0.6277 | 0.8524 |
| 14 | 0.5133 | 0.7724 | | 61 | 0.6326 | 0.8538 |
| 15 | 0.5184 | 0.7750 | | 62 | 0.6325 | 0.8547 |
| 16 | 0.5197 | 0.7778 | | 63 | 0.6392 | 0.8586 |
| 17 | 0.5184 | 0.7742 | | 64 | 0.6459 | 0.8598 |
| 18 | 0.5252 | 0.7807 | | 65 | 0.6450 | 0.8642 |
| 20 | 0.5323 | 0.7876 | | 67 | 0.6508 | 0.8678 |
| 21 | 0.5334 | 0.7882 | | 68 | 0.6505 | 0.8672 |
| 22 | 0.5354 | 0.7849 | | 69 | 0.6589 | 0.8729 |
| 23 | 0.5340 | 0.7866 | | 70 | 0.6635 | 0.8752 |
| 24 | 0.5419 | 0.7917 | | 71 | 0.6685 | 0.8768 |
| 25 | 0.5415 | 0.7931 | | 73 | 0.6743 | 0.8830 |
| 26 | 0.5417 | 0.7923 | | 74 | 0.6800 | 0.8832 |
| 27 | 0.5408 | 0.7940 | | 75 | 0.6811 | 0.8833 |
| 28 | 0.5470 | 0.7953 | | 76 | 0.6868 | 0.8874 |
| 29 | 0.5517 | 0.8024 | | 77 | 0.6927 | 0.8903 |
| 31 | 0.5506 | 0.7979 | | 78 | 0.6950 | 0.8928 |
| 32 | 0.5537 | 0.8007 | | 79 | 0.6981 | 0.8934 |
| 33 | 0.5545 | 0.8014 | | 80 | 0.7028 | 0.8960 |
| 34 | 0.5542 | 0.8043 | | 81 | 0.7040 | 0.8971 |
| 35 | 0.5637 | 0.8107 | | 82 | 0.7094 | 0.8986 |
| 37 | 0.5653 | 0.8100 | | 83 | 0.7115 | 0.9014 |
| 38 | 0.5695 | 0.8118 | | 85 | 0.7160 | 0.9034 |
| 39 | 0.5644 | 0.8111 | | 86 | 0.7193 | 0.9048 |
| 40 | 0.5735 | 0.8163 | | 87 | 0.7198 | 0.9061 |
| 41 | 0.5719 | 0.8157 | | 88 | **0.7212** | 0.9061 |
| 43 | 0.5782 | 0.8184 | | 89 | 0.7206 | 0.9066 |
| 44 | 0.5803 | 0.8201 | | 90 | 0.7208 | 0.9063 |
| 45 | 0.5835 | 0.8225 | |    |        |        |

Best top-1: 0.7212 (epoch 88). Final epoch 90: 0.7208 (in-train) / **0.7202 full-50k**.

(A few epoch rows missing where a monitoring snapshot landed mid-epoch; the curve is monotone enough that the gaps are immaterial.)

## Files
- `r34_imagenet_bf16.bin` (=`_e90.bin`), 87 MB — final weights.
- `_e1.bin … _e90.bin` — per-epoch checkpoints (on disk at /home/skoonce/).
- `scripts/eval_r34_full50k.py` — canonical full-50k eval.
- `scripts/supervise_r34_90ep.sh` — checkpoint+auto-resume supervisor.
