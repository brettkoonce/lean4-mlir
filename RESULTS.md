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

## Oxford-IIIT Pets detection (cat/dog head boxes, YOLOv1, mAP@0.5)

R34-ImageNet backbone (21.28M-float bootstrap) + deep conv head →
7×7×30 YOLOv1 grid, focal-BCE objectness, trained on class-balanced
2×2 head-box mosaics (e20 checkpoint). Scored by `scripts/yolo_map.py`
over the whole val set: per-class AP@0.5 (all-point VOC integration),
detections ranked by sigmoid(conf), class from argmax of the class
slots, per-class NMS at IoU 0.5. This is the **first real detection
metric** in the repo — v1's "64/64" was hand-counted peak
localization, not IoU-based.

| Val set | cat AP | dog AP | **mAP@0.5** | mAP@0.3 |
|---|---|---|---|---|
| mosaic (trained regime) | 0.028 | 0.053 | **0.041** | 0.227 |
| single-frame (transfer) | 0.000 | 0.000 | **0.0002** | 0.005 |

Three honest reads (Gate A baseline, `planning/yolo_demo_v2.md`):
1. **Single-frame ≈ 0** quantifies v1's "trained on mosaics → 2/16 on
   full frames" caveat: the model does not transfer to centered
   single pets (mosaic mAP@0.5 is ~200× higher). Moving this is
   Workstream B's whole job (mixed single+mosaic training).
2. **v1's "64/64" was peak-in-right-cell localization, not IoU@0.5.**
   The localization *ceiling* (best IoU over all 49 cells per GT box)
   averages 0.497 — 50% of GT heads reach IoU≥0.5, 91% reach IoU≥0.3
   — so the detector finds the right region but its boxes sit on the
   IoU=0.5 knife-edge (predicted heads run ~20% larger than GT). Hence
   mAP@0.3 (0.227) ≫ mAP@0.5 (0.041) on the mosaic set.
3. **The focal head is confidence-saturated** (~4 cells/image exceed
   sigmoid(conf)=0.5, the α-balance equilibrium ~0.55), so ranking is
   weak among the top detections — a known property (see the v2 doc),
   not a harness artifact. The AP integrator is unit-tested on
   perfect/random/mixed inputs.

### Workstream B — mixed single+mosaic training (transfer)

Retrained the same R34 spec from the R34-ImageNet bootstrap on a
50/50 blend of single full-frame pets (box-aware crop, scale 0.6–1.0,
`preprocess_pets_mosaic.py --single-frac 0.5`) and 2×2 mosaics, 80 ep,
scored on both standing val sets:

| checkpoint | mosaic@0.5 | mosaic@0.3 | single@0.5 | single@0.3 |
|---|---|---|---|---|
| v1 mosaic-only (e20) | 0.041 | 0.227 | 0.0002 | 0.005 |
| 50/50 mixed e80 | 0.034 | 0.217 | 0.011 | 0.049 |
| **75% single e80** | 0.020 | 0.143 | **0.041** | **0.163** |

Gate B verdict — **the frontier is real and the box-scale hypothesis is
confirmed.** Mixing singles works; the tuning knob is the single fraction.

1. **50/50 blend: mechanism works but the box head collapses.**
   Single-frame rose ~50× @0.5 over v1, but plateaued at ~0.01 from
   e20→e80 (more epochs don't help). Diagnosed from the e80 preds: box
   regression is per-cell, so a 50/50-by-*record* blend is ~4:1-by-*box*
   toward small quadrant boxes (mosaic = 4 small-box cells/record, single
   = 1 large-box cell). The width head collapsed to a near-constant
   **w=0.23±0.01** while single heads need w≈0.40 — so **0% of top
   single-frame boxes reached IoU≥0.5**.
2. **75% single blend confirms the fix.** Rebalancing the box counts
   (~1.3:1) un-collapsed the width head to **w=0.57±0.02** (now in the GT
   range), and single-frame **kept climbing to e80** (0.017→0.020→0.041
   @0.5) instead of plateauing: **30% of top single boxes now reach
   IoU≥0.5** (was 0%), 63% reach IoU≥0.3, mean top-box IoU 0.39 (was
   0.05). Single-frame mAP@0.5 (0.041) now equals v1's *mosaic* mAP@0.5
   — the detector localizes ordinary full-frame pets (see
   `demos/figures/yolo_pets_single_m75.png`), the money shot v1's 2/16
   couldn't produce.
3. **The tradeoff is explicit**: more singles → mosaic drops (0.020 vs
   v1's 0.041 @0.5). So the frontier is single-fraction: 50/50 favors
   mosaic, 75% favors single; a mid point (~0.6–0.65) should balance both.
   **Localization is solved; the remaining gap is class** — the head
   still mostly calls cats "dog" (the separate ~64% class-bias ceiling),
   orthogonal to the box-scale fix.

## Certified robustness scorecard (proved in Lean)

The Lipschitz-margin certificate (`lipschitz_margin_certified_radius`, Tsuzuku
et al. 2018) at trained weights, scaled to a dataset-level claim
(`LeanMlir/Proofs/Certificates/LipschitzCertScorecard.lean`, generated by
`scripts/lipschitz_cert_scorecard.py`): over the fixed first 100 MNIST test
images (4×4-pooled to 49 features, exact pixel-sum rationals) at fixed
ε = 0.1 (pooled-feature L2), every certified image carries an in-kernel
exact-rational margin lemma and a `∀ δ, ‖δ‖ < ε → argmax fixed` theorem —
zero sorrys, standard 3-axiom closure. Counts are honest lower bounds (an
upper-bound L cannot prove an image *un*certifiable); the PGD column is the
empirical upper bracket (L2-PGD, 100 steps, 4 restarts): cert ≤ TRUE ≤ PGD.

| Net (49→8→10 ReLU MLP, pooled MNIST) | Quantized test acc | Proved L (Schatten-8) | Certified @ ε=0.1 | LipSDP certified @ ε=0.1 | PGD-robust @ ε=0.1 |
|---|---|---|---|---|---|
| unconstrained SGD, /128 rationals | 89.8% | 63.79 | **1/100** | **63/100** | 69/100 |
| σ ≤ 4 projected SGD, /256 rationals | 87.0% | 19.76 | **34/100** | **69/100** | 72/100 |

Same theorem, same ε — the spectral projection during training decides whether
the certificate bites: 2.8 points of clean accuracy buy 34× the certified
count. (Caps ≤ 2 cost too much at this scale: σ ≤ 2 drops clean accuracy to
66%.) The LipSDP column replaces the global `√2·∏‖Wᵢ‖` criterion by a per-pair
LipSDP-Neuron constant (Fazlyab 2019), PSD-witnessed by exact rational LDLᵀ
(`LipschitzCertScorecardSDP{,Uncon}.lean`) — same nets, same images, same ε.
Single-image radius ladder on the unconstrained net: Frobenius 0.046 →
Schatten-4 0.111 → Schatten-8 0.154 (`trained_demo_certified*` in
`LipschitzCertInstance.lean`, with the power-iteration lower bounds
sandwiching each layer's true σ₁).

### Full 784-dim input (no pooling) — the sandwich closes

The same certificates at genuine full-input resolution (exact `k/255` pixels,
pixel-space L2 ε — directly comparable to the literature), two 784→16→10 nets,
both radii (`LipschitzCertScorecardFull*.lean` + `...SDPFull*.lean`; the
784-term dot products are kernel `dotZ` evaluations, see `ListDot.lean`):

| Net (784→16→10) | Quantized test acc | Proved L (Schatten-8) | ε | Certified (global L) | LipSDP certified | PGD-robust |
|---|---|---|---|---|---|---|
| σ ≤ 2 projected SGD | 92.4% | 4.95 | 0.1 | 92/100 | **93/100** | **93/100** |
| σ ≤ 2 projected SGD | 92.4% | 4.95 | 0.3 | 72/100 | **91/100** | 92/100 |
| unconstrained SGD | 95.1% | 29.85 | 0.1 | 76/100 | **91/100** | 94/100 |
| unconstrained SGD | 95.1% | 29.85 | 0.3 | 2/100 | **77/100** | 86/100 |

At ε = 0.1 on the capped net the per-pair LipSDP certificate **equals the
L2-PGD attack bound — cert ≤ TRUE ≤ PGD closes to an equality**, machine-
checked over the trained rational weights. The unconstrained ε = 0.3 row
(2 → 77) is the cleanest evidence that the global product constant, not the
network, was the bottleneck. At full input even σ ≤ 2 keeps 92.4% clean
accuracy — the pooled experiment's "caps ≤ 2 cost too much" was an artifact
of the 49-dim reduction. Counts are lower bounds; all 3-axiom-clean
(`tests/AuditAxioms.lean`).

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
