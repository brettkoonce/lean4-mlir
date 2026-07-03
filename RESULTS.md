# Training Results

All models trained via the Lean → MLIR → IREE pipeline using
`LeanMlir.Train.train` (the unified training loop). Adam, cosine LR
with linear warmup, label smoothing 0.1, weight decay 1e-4. Eval uses
running BN statistics (EMA momentum 0.1) — not batch statistics.

Hardware: AMD Radeon 7900 XTX (gfx1100) via ROCm 7.2 / IREE.

## Imagenette (10 classes, 224×224)

~9.5K train / 3.9K val. Augmentation: random crop 256→224 + horizontal flip.

| Model | Params | Val accuracy | Notes |
|---|---|---|---|
| ResNet-34 | 21.3M | **90.29%** | basic residual blocks (3x3 + 3x3) |
| ResNet-50 | 23.5M | **89.40%** | bottleneck blocks (1x1 → 3x3 → 1x1) |
| EfficientNetV2-S | 38.2M | **88.50%** | fusedMbConv early stages + MBConv+SE later |
| EfficientNet-B0 | 7.2M | **87.58%** | MBConv with swish + sigmoid SE |
| MobileNetV2 | 2.2M | **87.09%** | depthwise separable + inverted residual |
| MobileNetV3-Large | 3.0M | **86.48%** | exact h-swish + h-sigmoid SE |
| MobileNetV4-Medium | 4.1M | **84.58%** | Universal Inverted Bottleneck (15 blocks, 4 variants from 1 primitive) |
| ViT-Tiny | 5.5M | **71.70%** | patch embed + 12 transformer blocks (data-hungry) |

## MNIST (10 classes, 28×28 grayscale)

60K train / 10K test. No augmentation.

| Model | Params | Val accuracy | Notes |
|---|---|---|---|
| MNIST-CNN | 1.7M | **99.50%** | 4× convBn + 2× dense, batch 128, 15 epochs |

## CIFAR-10 (10 classes, 32×32 RGB)

50K train / 10K test. Augmentation: random horizontal flip.

| Model | Params | Val accuracy | Notes |
|---|---|---|---|
| CIFAR-10-BN | 3.7M | **83.50%** | 4× convBn + 3× dense + 2 max pools, batch 128, 30 epochs |

## tinyshakespeare (char-level language modeling, vocab 65)

1.0M train / 111K val tokens (90/10 split of the Karpathy corpus).
Per-token CE rides the per-pixel-CE (`useSeg`) codegen path; validation
is fixed-seed val chunks through the train-step vmfb with the update
discarded. Cosine LR + 100-step warmup, Adam, wd 1e-4, batch 32.
These runs: NVIDIA RTX 4060 Ti via CUDA / IREE. Lower bits/char is
better; uniform-random = 6.02, bigram baseline = 3.55.

| Model | Params | Val bits/char | Notes |
|---|---|---|---|
| tinyGPT-nano | 212K | **2.27** | T=64, D=64, 2 heads, 4 blocks, 10K steps (train 2.00 bits) |
| tinyGPT-tiny (5K steps) | 1.2M | **2.30** | T=128, D=128, 4 heads, 6 blocks; val minimum 2.25 @ step 3500; samples more locally fluent than nano |
| tinyGPT-tiny (10K steps) | 1.2M | 2.78 | overfit: val bottomed ≈2.27 @ step ~4500, train fell to 1.26 bits — kept as the "train loss lies" exhibit |

The 10K-step tiny run is the metric's first catch: by train loss it
"beats" nano by 0.7 bits while being worse on held-out text. v1 of
this demo tracked train loss only (`planning/tinygpt_demo_v2.md`).

## TinyStories (BPE language modeling, vocab 4096)

50.3M train / 4.86M val BPE tokens (`preprocess_tinystories.py`).
Model input is `[B, T]` f32 token ids with the one-hot built in-graph
(`tokenPositionEmbed idsInput` — validated byte-identical to the host
one-hot on the char-level nano model), so there is no O(V·T) host
upload at BPE vocab. Same per-token CE (`useSeg`) loss. Bits/token
(not char) — not comparable to the char rows above.

| Model | Params | Val bits/tok | Notes |
|---|---|---|---|
| tinyStories-8m @ step 500 | 8.5M | 4.65 | T=256, D=256, 8 heads, 8 blocks, causal |
| tinyStories-8m @ step 1000 | 8.5M | 3.89 | already emits coherent grammatical stories (named characters, full narrative arc) |

The demo output: a checkpoint this early already generates complete
children's stories from a prompt — the TinyStories (Eldan & Li 2023)
result reproduced inside the Lean → MLIR → IREE pipeline. Sample in
`blueprint/src/figures/tinystories/`.

## Oxford-IIIT Pets segmentation (3-class trimap, mIoU)

~3.7K train / 3.7K val, 224×224. Per-pixel softmax CE (`useSeg`).
mIoU via the `F32.segConfusion` harness (argmax over channels →
per-class IoU = TP/(row+col−TP)). These are **3-epoch smoke runs**
(the mains take an epochs arg for a real 60–80-ep budget); reported
to validate the harness + the skip-connection ablation direction.

| Model | Params | mIoU | fg IoU | bg IoU | boundary IoU |
|---|---|---|---|---|---|
| Autoencoder (skipless) | 5.5M | 0.360 | 0.425 | 0.655 | **0.000** |
| UNet (with skips) | 7.85M | 0.344 | 0.386 | 0.646 | **0.000** |

Two honest reads at this 3-epoch budget:
1. The per-class split is the point: both models collapse the thin
   boundary class (~12% of pixels) to **zero** IoU — a mean-of-3
   alone would hide it.
2. **The skip ablation is inconclusive at 3 epochs** — UNet does
   NOT yet beat the skipless autoencoder (0.344 vs 0.360, within
   noise, both boundary-collapsed). This is a budget artifact, not a
   skip-plumbing bug (the same `unetDown`/`unetUp` codegen trains
   the DDPM UNet fine; both models here are underfit at 3 ep and the
   UNet's extra 2.3M params haven't paid off). Gate B — "do skips
   help, especially on boundary?" — needs the real 60–80-ep run the
   epochs arg now enables: `unet-pets-train data/pets 70`. See
   `planning/unet_demo_v2.md`.

## Certified robustness scorecard (proved in Lean)

The Lipschitz-margin certificate (`lipschitz_margin_certified_radius`, Tsuzuku
et al. 2018) at trained weights, scaled to a dataset-level claim
(`LeanMlir/Proofs/LipschitzCertScorecard.lean`, generated by
`scripts/lipschitz_cert_scorecard.py`): over the fixed first 100 MNIST test
images (4×4-pooled to 49 features, exact pixel-sum rationals) at fixed
ε = 0.1 (pooled-feature L2), every certified image carries an in-kernel
exact-rational margin lemma and a `∀ δ, ‖δ‖ < ε → argmax fixed` theorem —
zero sorrys, standard 3-axiom closure. Counts are honest lower bounds (an
upper-bound L cannot prove an image *un*certifiable); the PGD column is the
empirical upper bracket (L2-PGD, 100 steps, 4 restarts): cert ≤ TRUE ≤ PGD.

| Net (49→8→10 ReLU MLP, pooled MNIST) | Quantized test acc | Proved L (Schatten-8) | Certified @ ε=0.1 | PGD-robust @ ε=0.1 |
|---|---|---|---|---|
| unconstrained SGD, /128 rationals | 89.8% | 63.79 | **1/100** | 69/100 |
| σ ≤ 4 projected SGD, /256 rationals | 87.0% | 19.76 | **34/100** | 72/100 |

Same theorem, same ε — the spectral projection during training decides whether
the certificate bites: 2.8 points of clean accuracy buy 34× the certified
count. (Caps ≤ 2 cost too much at this scale: σ ≤ 2 drops clean accuracy to
66%.) Single-image radius ladder on the unconstrained net: Frobenius 0.046 →
Schatten-4 0.111 → Schatten-8 0.154 (`trained_demo_certified*` in
`LipschitzCertInstance.lean`, with the power-iteration lower bounds
sandwiching each layer's true σ₁).

## Per-epoch eval history (running BN stats)

### ResNet-34

| Epoch | Val acc |
|---|---|
| 10 | 75.74% |
| 20 | 81.99% |
| 30 | 87.81% |
| 40 | 87.04% |
| 50 | 88.93% |
| 60 | 90.14% |
| 70 | 90.19% |
| 80 | **90.29%** |

### ResNet-50

| Epoch | Val acc |
|---|---|
| 10 | 73.87% |
| 20 | 77.15% |
| 30 | 85.07% |
| 40 | 87.47% |
| 50 | 88.42% |
| 60 | 89.11% |
| 70 | 89.73% |
| 80 | **89.40%** |

### MobileNetV2

| Epoch | Val acc |
|---|---|
| 10 | 75.03% |
| 20 | 79.00% |
| 30 | 81.86% |
| 40 | 84.68% |
| 50 | 86.37% |
| 60 | 86.76% |
| 70 | 87.19% |
| 80 | **87.09%** |

### MobileNetV3-Large

Exact h-swish (`x * ReLU6(x+3) / 6`) and h-sigmoid (`ReLU6(x+3) / 6`) with
piecewise gradients. SE block uses ReLU on reduce + h-sigmoid on gate (V3
variant), not swish + sigmoid (EfficientNet variant).

| Epoch | Val acc |
|---|---|
| 10 | 77.72% |
| 20 | 82.10% |
| 30 | 83.63% |
| 40 | 84.68% |
| 50 | 85.76% |
| 60 | 85.96% |
| 70 | 86.35% |
| 80 | **86.48%** |

### EfficientNet-B0

MBConv blocks with Swish activation, Squeeze-and-Excitation, variable
kernel sizes (3×3 and 5×5).

| Epoch | Val acc |
|---|---|
| 10 | 78.82% |
| 20 | 82.17% |
| 30 | 84.07% |
| 40 | 86.14% |
| 50 | 86.40% |
| 60 | 87.35% |
| 70 | 87.30% |
| 80 | **87.58%** |

### ViT-Tiny

Vision Transformer: 16×16 patch embedding → 12 transformer blocks
(192-dim, 3 heads, 768-dim MLP) → CLS token → dense. Exact tanh-form
GELU. 5-epoch warmup. Imagenette is too small for ViT to really shine
(transformers want 100K+ images), but it learns.

| Epoch | Val acc |
|---|---|
| 10 | 54.33% |
| 20 | 57.84% |
| 30 | 64.40% |
| 40 | 67.67% |
| 50 | 69.24% |
| 60 | 71.18% |
| 70 | 71.75% |
| 80 | **71.70%** |

### MobileNetV4-Medium

15 Universal Inverted Bottleneck (UIB) blocks expressing all four
block types (ExtraDW, IB / standard MBConv, ConvNeXt, FFN) from a
single parameterized primitive. The "stop adding new block types"
philosophy in action.

| Epoch | Val acc |
|---|---|
| 10 | 75.00% |
| 20 | 79.18% |
| 30 | 82.22% |
| 40 | 82.48% |
| 50 | 83.15% |
| 60 | 84.43% |
| 70 | 84.84% |
| 80 | **84.58%** |

### EfficientNetV2-S

Three Fused-MBConv stages early (where depthwise is slow on hardware),
then three standard MBConv-with-SE stages. 38M params, 110 BN layers,
~9 min/epoch.

| Epoch | Val acc |
|---|---|
| 10 | 75.79% |
| 20 | 82.89% |
| 30 | 85.81% |
| 40 | 86.53% |
| 50 | 87.35% |
| 60 | 87.99% |
| 70 | 88.32% |
| 80 | **88.50%** |

## Training recipe ablation (ResNet-34)

| Config | Final val | Notes |
|---|---|---|
| SGD+momentum, batch 16, batch-stats eval | 24.36% | initial baseline |
| SGD + WD 5e-4 + hflip + label smooth + crop | 26.53% | partial regularization |
| Adam, batch 32, cosine LR (batch-stats eval) | 52.54% | optimizer + batch fix |
| Adam, batch 32, cosine LR + **running BN eval** | **90.29%** | full pipeline |

The single biggest jump (+38 points) came from switching eval from
mini-batch BN statistics to running BN stats. The model was learning
fine the whole time — the eval was broken because batch statistics
from 32 images are too noisy to normalize correctly.

## Pipeline

```
Lean 4 NetSpec (~15 lines)
  ↓  MlirCodegen.generateTrainStep (forward + loss + VJPs + Adam)
StableHLO MLIR (~500 KB - 760 KB)
  ↓  iree-compile (~10 min for ROCm gfx1100)
VMFB flatbuffer (1.8 - 2.4 MB)
  ↓  IREE runtime via libiree_ffi.so
GPU execution (HIP / ROCm)
```

The same Lean → MLIR pipeline handles all 5 architectures. Adding a new
architecture requires extending `MlirCodegen.lean` with:
- Forward emission for the new layer types
- VJP / backward emission
- FwdRec recording for backward intermediates

The training executable, FFI, and IREE runtime are unchanged across
architectures. Only the codegen grows.
