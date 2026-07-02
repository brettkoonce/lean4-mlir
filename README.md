# Lean 4 → MLIR → GPU

**Interactive proof blueprint: [brettkoonce.github.io/lean4-mlir/blueprint/](https://brettkoonce.github.io/lean4-mlir/blueprint/)**
(or [PDF](https://brettkoonce.github.io/lean4-mlir/blueprint.pdf))
— clickable dependency DAG for the full VJP proof suite (no `sorry`s, zero project axioms),
from `pdiv` primitives up to the whole-network VJPs (ViT, ResNet, MobileNetV2, ConvNeXt, EfficientNet).

Lean 4 as a specification language for neural networks. Declare architecture
in Lean, generate StableHLO MLIR (forward + loss + backward + optimizer all
in one fused function), compile to GPU via IREE, train end-to-end. No Python
runtime, no autograd library — the gradients are computed at codegen time
in Lean.

Companion code for the upcoming book *Verified Deep Learning with Lean 4*
(follow-up to [Convolutional Neural Networks with Swift for TensorFlow](https://doi.org/10.1007/978-1-4842-6168-2), Apress).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20402133.svg)](https://doi.org/10.5281/zenodo.20402133)

**Current version: `v0.6.1`** — verified training reaches low precision
(fp8/E4M3 + bf16-mixed), Chapter 4 recast as the MNIST→ResNet bridge,
toolchain on Lean 4.31.0. Full release history in [CHANGELOG.md](CHANGELOG.md).

## Quick start

Train a real, verified neural net end to end — fastest path first.

**No GPU, just Docker (~5 min):**

```bash
git clone https://github.com/brettkoonce/lean4-mlir.git && cd lean4-mlir
docker build -t lean4-mlir-demo .
docker run --rm lean4-mlir-demo
```

Trains the Chapter-2 MNIST MLP on CPU to ~97.9% test accuracy through the
full Lean → MLIR → IREE pipeline — no GPU, no Python, ~300 MB image. (First
build ~10 min, dominated by building the IREE CPU runtime; reruns reuse the
cached image.)

**With a GPU — one command per tier:**

After a one-time native setup (Lean 4 + an IREE runtime for your backend —
see [ROCM.md](ROCM.md) / [CUDA.md](CUDA.md), or [Native setup](#native-setup-gpu-training) below):

```bash
./download_mnist.sh     # fetch MNIST
lake run mnist          # build + train the verified MNIST nets (linear / MLP / CNN)
```

Then scale up: `lake run cifar` (the Chapter-4 BatchNorm × optimizer ablation)
and `lake run imagenette` (the five Part-I nets at 224²). Not sure how long
those take on your card? `lake run benchmark` probes your GPU and prints a
per-chapter time estimate first.

**Just the proofs (no IREE, no GPU):**

```bash
lake exe cache get         # pull prebuilt Mathlib
lake build ProofsMinimal   # smallest end-to-end-verified example, ~seconds
lake build Proofs          # type-check the entire VJP proof suite
```

The full clickable proof DAG is the [interactive blueprint](https://brettkoonce.github.io/lean4-mlir/blueprint/).

## Three phases

This project went through three implementations of the same idea — "Lean 4 as a
specification language for deep learning" — each shedding more dependencies
than the last.

**Phase 1 — Pure Lean 4.** [`mnist-lean4/`](mnist-lean4/): everything in Lean,
`Float64` as the only datatype, hand-written gradients, C FFI to OpenBLAS /
hipBLAS for the matmuls. Worked end-to-end on MNIST through ResNet-34 but
performance was poor — every operation crossed the FFI boundary, no fusion,
no autodiff, no JIT.

**Phase 2 — Lean → JAX.** [`jax/`](jax/): Lean as a metaprogramming layer
that emits idiomatic JAX Python (`jax/Jax/Codegen.lean`, ~2100 lines). The
generated script gets `value_and_grad` autodiff and XLA JIT for free, runs
on any JAX-supported device. Trades the pure-Lean story for a working stack
and real GPU performance. See [`jax/README.md`](jax/README.md) for details.

**Phase 3 — Lean → StableHLO → MLIR → device.** *(this README)* No Python
runtime at all. Lean directly emits StableHLO MLIR, IREE compiles it to a
GPU flatbuffer, a thin C FFI loads and runs it. The pure-math version of
phase 2 — autodiff is done at codegen time in Lean (`LeanMlir/MlirCodegen.lean`,
~7500 lines), not at runtime by a framework. See [`RESULTS.md`](RESULTS.md)
for the per-architecture numbers.

The VJP correctness proofs live in [`LeanMlir/Proofs/`](LeanMlir/Proofs/) —
chapter-by-chapter, for tensor ops, MLP, CNN, residual, batch norm,
depthwise, SE, LayerNorm, and attention, up to whole-network backward passes
(ViT, ResNet, MobileNetV2, ConvNeXt, EfficientNet). What they establish: each
reference forward function, written in exact real arithmetic (`ℝ`), has a
backward pass equal to its Mathlib `fderiv` Jacobian-transpose — with zero
project axioms (`#print axioms` closes under the Lean-core triple alone).

The whole-network results come in two forms, set by the architecture's
activations:

- **Unconditional** — ViT, ConvNeXt, EfficientNet. These use only smooth ops
  (GELU / Swish / sigmoid, softmax, LayerNorm, convolution — no ReLU, no
  max-pool), so the VJP holds at *every* input, with the LayerNorm/BatchNorm
  `0 < ε` positivity as the only side conditions (`vit_full_has_vjp`,
  `convnext_has_vjp`, `efficientnet_has_vjp : HasVJP …`).

- **Conditional + concretely instantiated** — MLP, MNIST-CNN, ResNet,
  MobileNetV2. ReLU, ReLU6 and max-pool are genuinely non-differentiable at
  their kinks, so the *generic* whole-network VJP is stated at a smooth point
  (`*_has_vjp_at`, under per-site "off the kink" hypotheses). Each is then
  **instantiated on a concrete (small, representative) network** with every
  smoothness hypothesis discharged, giving a hypothesis-free correctness
  theorem (`MlpConcrete`, `Spatial`/`Mini`, `CnnConcrete`,
  `MobileNetV2Concrete`) — proof that the kink-avoidance conditions are
  jointly satisfiable on the real forward, not vacuous.

Axiom closure on every one of these is a CI invariant
([`tests/AuditAxioms.lean`](tests/AuditAxioms.lean)); the generic headline
theorems are additionally re-checked by the independent
[`tests/comparator/`](tests/comparator/) kernel pass.

These proofs are about the reference `ℝ` definitions in `Proofs/`. They are now
tied to the emitted StableHLO **at the denotational level**: for every chapter
net the rendered train-step graph's `ℝ` denotation (`den : SHlo n → Vec n`) is
proven equal to the certified `fderiv`-derived loss-descent step (the §1a
whole-net ties — see Tier 3 below). What is *not* yet bridged is the
`den`→`Float32` numerics, and the separate `Float32` `MlirCodegen.lean` path the
full-recipe trainers behind the headline accuracy numbers use. Two further
checks corroborate the emitted formulas independently of `den`. (1)
*Structural*: codegen and proofs were developed independently and arrived at
the same decomposition — every backward pass factors through the standalone
gradient of one new primitive per architecture (softmax for attention, the
spatial reductions for BN, the rank-1 collapse for SE), and everything else is
composition via the chain rule on tools from earlier chapters — and the
codegen cites the matching proof inline in the MLIR it generates. (2)
*Numerical*: finite-difference checks ([`LeanMlir/Proofs/check_jacobians.py`](LeanMlir/Proofs/check_jacobians.py))
and JAX `value_and_grad` oracles ([`tests/vjp_oracle/`](tests/vjp_oracle/))
exercise the emitted formulas — including at the ReLU/MaxPool kinks, where the
codegen substitutes the standard subgradient convention. See the "Codegen
trust boundary" section of [`LeanMlir/Proofs/README.md`](LeanMlir/Proofs/README.md)
for the precise gap. The forward-and-backward extraction that ties a proven
graph to the emitted render is now done at the `den` level for all chapter nets
(the §1a ties); what remains open is carrying it across the `den`→`Float32`/IREE
boundary.

## What is and isn't verified

All proofs are over exact reals (`ℝ`). The emitted MLIR and GPU execution are
`Float32`; `iree-compile`, the IREE runtime, and the FFI are trusted. Within
that boundary, the verification is tiered by dataset / backend:

**Tier 1 — MNIST (linear, mlp, cnn): forward + backward bridged.** The reference
forward and backward are proven faithful to the Mathlib `fderiv` math as rendered
StableHLO graphs (`mlpFwdGraph_faithful`, `mlpBackGraph_faithful`,
`cnnFwdGraph_faithful`, `cnnBackGraph_faithful`; for linear also the param-grad
Jacobians `wGrad/bGrad_is*Jacobian` and `sgdW/sgdB_descends_certified_grad`).
All audited to the 3-axiom closure. The whole train-step module is now
`render(provenGraph)`: `linTrainStepFaithfulV` (the fully-tied renderer in
`StableHLO.lean` that generates `verified_mlir/linear_train_step.mlir`) renders
every node — grad/SGD tail included — as `pretty` of proven `SHlo` nodes, and
`poc_train_step_tail_certified` proves each emitted `weightSgd`/`biasSgd`
output's `den` is the certified loss-descent step (the older hand-tailed
`linearTrainStepModuleV` is kept only for reference; the committed bytes are
byte-tied to the renderer in CI). Tier 1 also now
carries the `ℝ`→`Float32` bridge (below): forward, gradient, and SGD-step
rounding budgets for all three nets (linear / MLP / CNN), each with a proven
loss-descent guarantee — the CNN (`cnn_conv2_sgd_descends` &c.) carries descent
through the max-pool selection margins.

**Tier 2 — CIFAR (cifar, cifar-bn): whole train step bridged.**
`cifarFwdGraph_faithful` / `cifarBnFwdGraph_faithful` (plus op-level
`bnBack_faithful`) hold, and the §1a whole-net ties now cover the train step:
`cifar_conv_tied_certified` / `cifarBn_convbn_tied_certified` prove every emitted
conv/BN/dense parameter-SGD node denotes (`den`) the certified loss-descent step
at the real CIFAR forward + composed softmax-CE cotangent (same `den`→`Float32`
trust boundary as Tier 3).

**Tier 3 — Imagenette (ResNet-34, MobileNetV2, ConvNeXt, EfficientNet, ViT):
ℝ whole-net VJP proven *and* the whole train step bridged to the emitted
graph.** The whole-network VJP is proven over `ℝ` (`resnet34_has_vjp_at`,
`vit_full_has_vjp`, `convnext_has_vjp`, `efficientnet_has_vjp`,
`mobilenetv2_has_vjp_at`). On top of that, the **§1a whole-net ties** now bridge
the entire train step: one capstone per net — `r34_net_tied_certified`,
`mnv2_net_tied_certified`, `cnx_net_tied_certified`, `efficientnet_net_tied`,
`vit_net_tied_certified` — proves every emitted parameter-SGD node of the committed
`verified_mlir/<net>_train_step.mlir` render denotes (`den`) the certified
`fderiv`-derived `θ − lr·∂Loss/∂θ` step, with the cotangent threaded through the
**real** full forward and the loss-driven backward composed from the proven
per-block VJPs (residual fan-in at every skip — not a free `∀`-cotangent). All
3-axiom-clean (`tests/AuditAxioms.lean`), and the `<net>-verified` exes train on
exactly that committed render. What stays trusted: the `den`→`Float32` numerics,
the per-op `pretty` lexing, and `iree-compile`/runtime/FFI (the CI drift guard
currently byte-checks `linear` + `vit` against the regenerated renderer,
extended per net; convnext has 4 even-kernel weight-grad gaps, vit has none).
The headline accuracy numbers below still come from the mature full-recipe
`*-train` trainers on the unverified `MlirCodegen.lean` path.

**Tier 4 — ImageNet-1k (phase-2 Lean→JAX bridge): scale baseline, gradients
not Lean-verified.** Full 1000-class ImageNet runs use the phase-2 path
(`jax/Jax/Codegen.lean`, ~1100 lines: `NetSpec` → idiomatic JAX Python), where
**JAX's `value_and_grad` computes the gradients and XLA does the compilation** —
the Lean VJP proofs are not in the loop. The only proof-adjacent Lean artifact is
the shared `NetSpec` ADT (the same architecture spec whose phase-3 backward is
proven over `ℝ`); the emitter itself is unverified. This tier exists to (a)
establish scale baselines the verified-IREE codegen can't yet reach — ConvNeXt-T
75.93% / EfficientNet-B0 72.31% / ResNet-34 72.02% / MobileNetV2 68.33% / ViT-Tiny
65.64% top-1, full 50k val ([`jax/runs/*/RESULTS.md`](jax/runs/)) — and (b) serve
as the differential-test **oracle**: [`tests/vjp_oracle/`](tests/vjp_oracle/) uses
JAX `value_and_grad` as ground truth to cross-check the Tier 1–3 Lean-derived VJPs
to 1–2 ULP. So Tier 4 is the least-verified tier by gradient provenance but the
one that empirically anchors the others. Whether phase-3 verified codegen can reach
ImageNet scale is open.

### The ℝ→Float32 bridge (Tier 1)

All tier proofs are over exact reals; `LeanMlir/Proofs/FloatBridge.lean` +
`SgdDescent.lean`/`SgdDescentLinear.lean`/`SgdDescentMlp.lean`/`SgdDescentCnn.lean`
close the rounding gap for the
Tier-1 nets, hypothesis-style (zero project axioms — a `FloatModel` is any
rounding operator with relative error `u`; `binary32` instantiates it with
`u = 2⁻²⁴` on the normal range, subnormals open). The named `binary32`/`fp8E4M3`
models are **constructed**, not assumed (`Binary32Instance.lean`): `rndP p` =
round-to-nearest on the unbounded-exponent `p`-bit grid, with the standard model
`|rndP p x − x| ≤ 2⁻¹⁻ᵖ·|x|` *proved* (`rndP_err`) — the repo contains zero
`axiom` declarations anywhere (CI-enforced). What stays trusted is the
kernel↔model boundary (FMA, reassociation, "the GPU rounds like this grid"),
not the operator's existence. The chain, every link in the 3-axiom audit:

- **Forward** (`mlp_float_close_uniform`): dot/dense budgets in the
  classical compounded form, valid for *every* summation association (IREE
  may reassociate reductions freely). ReLU is exact in float — the op that
  forces the off-the-kink hypotheses over `ℝ` is the free op here.
- **Backward** (`mlp_{w2,w1,w0,b2,b1,b0}_step_float_close`): every rounded
  SGD parameter entry within an explicit budget of `θ − lr·(aᵢ·cⱼ)` — the
  same `emitWeightGrad`/`emitBiasGrad` entries `mlp_render_*_certified`
  prove equal to the `pdiv`-Jacobian contractions. The ReLU masks need
  *quantitative* margins (`ez < |zᵢ|`: rounding must not flip a sign).
- **Loss head** (`softmax_ce_cot_close`): the rounded softmax−onehot
  cotangent vs the certified gradient, given an `exp` accuracy hypothesis
  (`|fexp t − exp t| ≤ eexp·exp t` — GPU `exp` has no IEEE spec; `eexp` is
  the constant `tests/vjp_oracle/` validates at 1–2 ULP).
- **Descent** (`sgd_descends`, `linear_sgd_descends`,
  `mlp_{output,hidden,input}_sgd_descends`): an η-accurate gradient step
  still *decreases the loss* — with the smoothness hypothesis proven, not
  assumed: explicit constant `2a²/(1−2aD)` for the linear net, and through
  the MLP's ReLU kinks per weight layer under quantitative **margins** (the
  step's `ℓ1` radius cannot flip a mask sign, so the sign pattern freezes
  along the segment): `2d₃w₂²a²/(1−2w₂aD)` for the hidden layer,
  `2d₃d₂²w₁²w₂²a²/(1−2w₂d₂w₁aD)` for the input layer; the output layer is
  the linear theorem at the hidden activation, margin-free. No Hessian
  anywhere (the same softmax ratio sandwich as the float budgets).

**Measured vs proven** (`scripts/margin_probe.py`, an f32/f64 twin of the
97.8% GPU run; numeric capstones instantiated at the *trained* magnitudes
`|W| ≤ 3/5`):

| quantity | worst-case theorem | measured |
|---|---|---|
| logit drift | ≤ 5100 (`mnist_mlp_float_budget`) | 1.6·10⁻⁵ |
| cotangent | ≤ 21/1000 at δ=1/100 (`mnist_cot_budget`) | 2.2·10⁻⁶ |
| W₂ SGD step | ≤ 5/4 (`mnist_w2_step_float_budget`) | 7.5·10⁻⁹ |
| ReLU mask flips | 0 under margins | **0 / 29.5M** |

The worst-case-vs-measured gap (up to ~10⁸) is the quantitative case for
a-posteriori certificates past toy depth; the zero flip count says the
margin hypotheses describe real training, not a technicality.

#### Low precision: bf16-mixed and fp8 (E4M3)

`FloatModel`'s `u` is a *parameter*, so precision is an instantiation, not a
rewrite — up to where the model's assumptions break.

- **Two-roundoff model** (`dot_close_mixed`, `dense_close_mixed`): split the
  single `u` into a leaf precision `u_leaf` (the matmul inputs) and an
  accumulate `u_acc` (the reduction). The leaf contributes only a *flat*
  `(2·u_leaf + u_leaf²)·Σ|xy|` term; the fan-in Higham γ rides entirely on
  `u_acc`. So the `1/u` fan-in wall sits at `u_acc = 2⁻²⁴`, **not** at the leaf
  — which is exactly why **bf16-mixed** (the deployed config: bf16 leaf, fp32
  accumulate — the shipped `r34_imagenet_bf16.bin` checkpoints) is non-vacuous
  where pure bf16 (`γ_k` vacuous at fan-in 256) is not.
- **fp8 (E4M3), depth-1.** MNIST-linear is a single 784→10 matmul, so the
  per-matmul leaf bound *is* the end-to-end bound — the one realistic fp8 case
  with an honest end-to-end accuracy guarantee.
  - *Empirical* (`scripts/mnist_e4m3_demo.py`): fp32 92.25% → E4M3 **92.30%**
    (per-row weight scale, per-tensor activation scale, fp32 accumulate) —
    precision drops elegantly.
  - *Accuracy* (`argmax_preserved`, `linear_e4m3_argmax_preserved`): a
    `B`-accurate matmul cannot flip the prediction on a `>2B`-margin input. At
    the trained magnitudes the worst-case `B ≤ 61` (`linear_e4m3_logit_budget`,
    the flat 12.5% leaf term dominating), so margin > 122 ⟹ provably the same
    prediction. That worst-case is vacuous on real data (mean margin ≈ 4.25);
    the demo's *measured* `B = 0.38` feeds the **same** theorem ⟹ **92.89%** of
    the MNIST test set provably unchanged (and 100% of those keep their label).
  - *Structural* (`e4m3_render_faithful`, `dequant_factors`): the emitted
    block-scaled int-matmul graph denotes the intended algorithm — the
    per-output dequant scale factors out of the fp32 accumulate
    (`(sx·sWⱼ)·∑ q q = ∑ (sx q)(sWⱼ q)`), so "int matmul then dequant" = "dequant
    then matmul". "The bytes implement block-scaled-E4M3 matmul with fp32
    accumulate," with no accuracy claim — built from existing `den`-faithful ops,
    no new IR constructors.
- The honest regime ladder: **fp32** and **bf16-mixed** are accuracy-provable;
  **fp8** is per-matmul-provable, end-to-end only a-posteriori past depth-1;
  **fp4** is structural-faithfulness + statistical robustness (the relative
  `|rnd x−x| ≤ u|x|` model gives way to block-scaled quantization). All the
  above is 3-axiom-clean and audited; see `planning/floatbridge_quantization.md`.

#### Certified robustness (Tsuzuku, instantiated at trained weights)

`lipschitz_margin_certified_radius` (`LipschitzCert.lean`) is instantiated on a
trained, rationalized 49→8→10 pooled-MNIST MLP with everything in the trust
path proved in-kernel: Frobenius → Schatten-4 → Schatten-8 Lipschitz bounds
(certified radius 0.046 → 0.111 → 0.154, `LipschitzCertInstance.lean`), plus
power-iteration *lower* bounds sandwiching each layer's true σ₁ so the bound's
looseness is itself certified. Scaled to a dataset-level **scorecard**
(`LipschitzCertScorecard.lean`): over the fixed first 100 MNIST test images at
fixed ε = 0.1 (pooled L2), a spectrally-capped (σ ≤ 4 projected-SGD) sibling
net certifies **34/100** predictions vs **1/100** unconstrained — same theorem,
same ε; training decides whether the certificate bites. Empirical L2-PGD
brackets it from above (69–72/100): cert ≤ TRUE ≤ PGD. See the table in
[`RESULTS.md`](RESULTS.md); all 3-axiom-clean.

**Not yet verified anywhere:** the ~7500-line `MlirCodegen.lean` (zero
theorems — the path behind the headline accuracy numbers); the printed `.mlir`
text that `iree-compile` actually consumes (the per-op `pretty` lexing step — the
train-step *graph* it prints is now `den`-certified for all chapter nets, not
just Tier 1); and, within the float bridge, subnormals (the model is
relative-error-only), the joint all-layers descent step and bias columns
(the per-weight-layer constants are proven for linear + MLP; for the CNN
the new ingredients are proven — quantitative max-pool selection margins
that freeze the argmax routing, pool `ℓ1`-contraction, conv-kernel drift
with the weight-sharing factor — but the conv-layer capstone assembly is
open, `planning/sgd_descent_cnn.md`; so is every-parameter-at-once, where
the logits are no longer affine in the moving parameters), and any link
from the Lean-side `FloatModel` to IREE's actual kernels beyond the
empirical probe.

**Concrete-instance honesty.** The conditional capstones (MLP, MNIST-CNN, CIFAR,
MobileNetV2, ResNet-34) are instantiated to discharge their off-the-kink
hypotheses. `MlpConcrete`, `Micro`/`Mini`/`Spatial` (MNIST) and `Tiny` (CIFAR) are
live witnesses (non-constant forward, nonzero Jacobian). The deep ReLU/BN nets now
also have **non-degenerate, nonzero-Jacobian-sealed** live witnesses: `Mnv2Live`
(`MobileNetV2JacobianSeal`), `ResNet34LivePC.liveFwd2` (`ResNet34LiveSeal`), and the
full real `[3,4,6,3]`-depth `liveFwd2Full` (`ResNet34LiveFull`) all prove a
non-constant forward **and** `fderiv ≠ 0` at a witness point ⇒ the rendered
backward is genuinely not the zero map (audited 3-axiom-clean). The old degenerate
constant-output instances `MobileNetV2Concrete`, `CnnConcrete`, `ResNet34Concrete`
(zero Jacobian) remain only as satisfiability checks; the BN-CNN live witness is the
last follow-up.

## Pipeline

```
Lean NetSpec  (~15 lines)
   │
   │  MlirCodegen.generateTrainStep
   ▼
StableHLO MLIR  (500 KB - 2 MB of text, forward+loss+backward+Adam fused)
   │
   │  iree-compile (~10-15 min for ROCm gfx1100)
   ▼
VMFB flatbuffer  (1.8-3 MB)
   │
   │  IREE runtime via libiree_ffi.so
   ▼
GPU execution  (HIP/ROCm or CUDA)
```

The same Lean → MLIR pipeline handles every architecture. Adding a new
architecture means extending `LeanMlir/MlirCodegen.lean` with:
- forward emission for the new layer types
- VJP / backward emission
- `FwdRec` recording for backward intermediates

The training executable, FFI, and IREE runtime are unchanged.

## Cross-backend verification

Phase 2 and Phase 3 share the same Lean `NetSpec` ADT but compile through
*completely independent* stacks (JAX/XLA vs IREE). Differential testing
confirms both stacks produce the same training dynamics on the same input,
for both MLP (670K params, 12 epochs) and CNN (1.7M params with conv+BN,
15 epochs):

| diff                              | MLP step 1 Δ | CNN step 1 Δ |
|-----------------------------------|--------------|--------------|
| phase 2 (JAX)  vs phase 3 (IREE)  | ~2e-7        | ~1e-5 to 1e-4 |
| phase 3 ROCm   vs phase 3 CUDA    | **0**        | **0**        |
| phase 2 CPU    vs phase 2 CUDA    | ~4e-6        | ~1e-4        |

MLP hits the float32 ULP floor because it's dense-only. CNN's noise
floor is looser by ~100× because each conv-BN layer does two
reductions over ~100k-element tensors and XLA's reduction trees differ
from IREE's — both pipelines do correct math, just with different
summation orders. Phase 3 ROCm ≡ Phase 3 CUDA is bit-identical at
step 1 on both networks. Reproducible in 5 minutes via
[`traces/CROSS_BACKEND_RESULTS.md`](traces/CROSS_BACKEND_RESULTS.md).

### VJP oracle

A separate per-axiom differential test in
[`tests/vjp_oracle/`](tests/vjp_oracle/) uses JAX's `value_and_grad` as
a correctness oracle for every hand-derived backward pass in
`LeanMlir/Proofs/`. Each test case is a minimal NetSpec exercising one
axiom in isolation; the oracle compares step-2 loss (the first step
whose value depends on the backward pass) against phase 2's
autodiff-derived gradients.

Nine cases, all green on mars (ROCm + CPU) and ares (CUDA):

| case | axiom | step 2 Δ |
|---|---|---|
| `dense` | `dense_has_vjp` + `softmaxCE_grad` | 2.7e-07 |
| `dense-relu` | `relu_has_vjp` + `vjp_comp` | 4.8e-07 |
| `conv` | `conv2d_has_vjp` + `flatten_has_vjp` | 2.2e-07 |
| `convbn` | `convBn_has_vjp` (BN-mode) | 2.2e-06 |
| `conv-pool` | `maxPool_has_vjp` (argmax tiebreaks) | 1.2e-04 |
| `residual` | `biPath_has_vjp` (additive fan-in) | 3.1e-07 |
| `depthwise` | depthwise-conv VJP via `.invertedResidual` | 1.1e-05 |
| `mbconv` | `elemwiseProduct_has_vjp` (SE gate) + Swish | 1.6e-06 |
| `attention` | patchEmbed + `transformerBlock_has_vjp_mat` + classifier | 1.8e-07 |

Run with `tests/vjp_oracle/run.sh`. Adding a new axiom means dropping
a minimal Lean spec under `tests/vjp_oracle/phase{2,3}/` plus one
line in the lakefiles — see
[`tests/vjp_oracle/README.md`](tests/vjp_oracle/README.md).

The oracle also surfaced a real `heInitParams` bug (shape-peek
heuristic misfiring at patchEmbed + transformer-block boundaries) and
a JAX-ROCm crash on gfx1100 (filed as
[ROCm/MIOpen#3955](https://github.com/ROCm/MIOpen/issues/3955); repro
lives at [`upstream-issues/2026-04-rocm-miopen-conv-segv/`](upstream-issues/2026-04-rocm-miopen-conv-segv/)).

## Results (Imagenette, 10 classes, 224×224)

Trained from scratch on a single AMD 7900 XTX (gfx1100), Adam, batch 32,
cosine LR + 3-epoch warmup, label smoothing 0.1, weight decay 1e-4, random
crop (256→224) + horizontal flip, **running BN stats for eval**.

| Model | Params | Val accuracy |
|---|---|---|
| ResNet-34 | 21.3M | **90.29%** |
| ResNet-50 | 23.5M | **89.40%** |
| EfficientNetV2-S | 38.2M | **88.50%** |
| EfficientNet-B0 | 7.2M | **87.58%** |
| MobileNetV2 | 2.2M | **87.09%** |
| MobileNetV3-Large | 3.0M | **86.48%** |
| ViT-Tiny | 5.5M | **71.70%** |

Per-epoch eval histories and ablation tables in [`RESULTS.md`](RESULTS.md).

## Native setup (GPU training)

The Quick start above is the fastest path; this is the full native install
behind it — for running the GPU tiers (`lake run mnist`/`cifar`/`imagenette`)
and individual trainers.

### 1. Install Lean 4

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### 2. Install IREE

You need the IREE runtime built for your GPU (CUDA or ROCm). The FFI shim
in `ffi/` links against `libiree_runtime_unified.a` from the IREE build tree.
See [`IREE_BUILD.md`](IREE_BUILD.md) for build instructions.

### 3. Get data

```bash
./download_mnist.sh        # MNIST (Ch 2-3 trainers)
./download_cifar.sh        # CIFAR-10 (Ch 4 trainers)
./download_imagenette.sh   # Imagenette 320px → preprocessed binary (Ch 5+)
```

### 4. Build + run a tier (or one trainer)

The `lake run` tiers build and run a curated group of verified trainers in
one command (backend auto-detected — `cuda` if `nvidia-smi` is present, else
`rocm`):

```bash
lake run mnist        # verified MNIST: linear / MLP / CNN          (~30 min)
lake run cifar        # ch.5 cifar8: SGD/momentum/Adam × bn/no-bn   (~1 hr)
lake run imagenette   # the 5 Part-I nets at 224², 80-epoch AdamW   (~37 h)
lake run benchmark    # probe this GPU, print per-chapter time estimates
```

To build and run a single trainer instead, the targets are the verified nets
those tiers bundle — e.g. `mnist-mlp-verified`, `cifar8-bn-verified`,
`resnet34-verified-adam`, `vit-verified-adam` (the six `cifar8{,-bn}-verified`
SGD/`-momentum`/`-adam` variants and the five `*-verified-adam` Imagenette
nets). Unverified `<arch>-train` targets (`vgg-train`, `resnet50-train`,
`mobilenet-v3-train`, …) also build, for nets outside the verified set.

The first run of any trainer compiles its vmfb (slow — 10–15 min for a
ResNet-sized model); reruns reuse the cache under `.lake/build/`. To run a
single binary directly with the env vars set:

```bash
HIP_VISIBLE_DEVICES=0 IREE_BACKEND=rocm .lake/build/bin/resnet34-verified-adam

# Or the shell wrapper that sets them for you
bash run.sh resnet34-verified-adam        # GPU 0, ROCm (defaults)
bash run.sh efficientnet-verified-adam 1 cuda   # GPU 1, CUDA
```

For CUDA, set `IREE_BACKEND=cuda` and use `CUDA_VISIBLE_DEVICES`; set
`IREE_CHIP` for your arch (`sm_86`/`sm_89`/`sm_90`, or `gfx1100` on ROCm).

## Lean specs

The same `NetSpec` type is used by all three phases. A spec is a list of
`Layer` values:

```lean
def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

def vitTiny : NetSpec where
  name := "ViT-Tiny"
  imageH := 224
  imageW := 224
  layers := [
    .patchEmbed 3 192 16 196,             -- (224/16)^2 = 196 patches
    .transformerEncoder 192 3 768 12,     -- 12 blocks, 3 heads, MLP dim 768
    .dense 192 10 .identity
  ]
```

## Project structure

```
lean4-mlir/
├── README.md               -- this file
├── CHANGELOG.md            -- release history
├── RESULTS.md              -- per-architecture eval histories + ablations
├── IREE_BUILD.md           -- how to build libiree_ffi.so from scratch
├── ROCM.md / CUDA.md       -- per-backend setup notes
├── BENCHMARK.md            -- ROCm vs CUDA performance comparison
├── lakefile.lean           -- Lake build (libs + ~150 execs + the
│                              `lake run mnist/cifar/imagenette/benchmark` tiers)
│
├── LeanMlir.lean           -- umbrella module
├── LeanMlir/
│   ├── MlirCodegen.lean    -- ~7500 lines, NetSpec → StableHLO MLIR (phase 3)
│   ├── Spec.lean, Types.lean       -- NetSpec / Layer / Activation / param counts
│   ├── Train.lean          -- unverified training driver (the `*-train` path)
│   ├── Verified{Spec,Nets,Train}.lean
│   │                       -- verified-render trainers (the `*-verified` path)
│   ├── ViTRender.lean      -- proof-tied StableHLO renderer (incl. AdamW tail)
│   ├── E4M3Quant.lean      -- fp8 (E4M3) quantization for the float bridge
│   ├── IreeRuntime.lean    -- Lean ↔ libiree_ffi.so bindings
│   ├── F32Array.lean       -- ByteArray-backed float32 helpers
│   └── Proofs/             -- VJP correctness proofs (~67k lines, 103 files)
│       ├── MLP, CNN, Residual, BatchNorm, Depthwise, SE, LayerNorm, Attention
│       │                          -- per-operator VJP correctness
│       ├── FloatBridge.lean        -- ℝ→Float32 rounding budgets (Tier 1)
│       └── SgdDescent{,Linear,Mlp,Cnn}.lean  -- inexact-gradient descent over ℝ
│
├── apps/                   -- one Main per exe, grouped by the Part-1 path:
│   ├── mnist/ cifar/ imagenette/   -- the verified `lake run` tiers (30 exes)
│   ├── baselines/          -- 15 unverified full-recipe trainers (resnet34, vgg, …)
│   └── ablation/           -- the cifar8 optimizer / head-width ablations
│
├── Bestiary/               -- 41 read-only NetSpec catalog entries
│                              (ResNet, ViT, AlphaZero, MuZero, CLIP, Mamba, …)
├── demos/                  -- 18 task demos (YOLO detection, UNet segmentation,
│                              TinyGPT, DDPM diffusion)
├── verified_mlir/          -- committed proof-rendered StableHLO the verified exes run
│
├── tests/                  -- unit / smoke / differential tests
│   └── vjp_oracle/         -- JAX-autodiff oracle for the hand-derived VJPs
├── jax/                    -- phase 2 (Lean → JAX Python); the ImageNet-scale path
├── mnist-lean4/            -- phase 1 (pure Lean 4 + C BLAS)
├── blueprint/              -- interactive proof-blueprint source (LaTeX / plasTeX)
├── ffi/                    -- IREE runtime wrapper + data-loading C (libiree_ffi.so)
├── traces/                 -- committed cross-backend training traces
├── upstream-issues/        -- isolated reproducers for upstream bugs
└── data/                   -- downloaded + preprocessed datasets
```

## Supported layers (phase 3 codegen)

| Layer | Description |
|-------|-------------|
| `dense` | Fully connected (with optional activation) |
| `conv2d` | Standard convolution |
| `convBn` | Conv + batch norm + ReLU/ReLU6/Swish/h-swish |
| `residualBlock` | BasicBlock (ResNet-18/34) |
| `bottleneckBlock` | Bottleneck (ResNet-50/101/152) |
| `invertedResidual` | Expand → depthwise → project + skip (MobileNetV2) |
| `mbConv` | + Squeeze-Excitation, Swish (EfficientNet) |
| `mbConvV3` | + h-swish + h-sigmoid SE (MobileNetV3, exact math) |
| `fusedMbConv` | k×k regular conv replaces (1×1 expand + depthwise) (EfficientNetV2) |
| `uib` | Universal Inverted Bottleneck — pre-DW? + expand + post-DW? + project (MobileNetV4) |
| `patchEmbed` | Conv patch projection + CLS token + positional embedding (ViT) |
| `transformerEncoder` | LN → MHSA → + → LN → MLP → +, with exact tanh-form GELU |
| `maxPool`, `globalAvgPool`, `flatten` | Structural |

Activations supported with exact backward: ReLU, ReLU6, Swish, h-swish,
h-sigmoid, GELU (tanh form). Layer-norm and batch-norm both have proper
VJPs and (for BN) running statistics for eval.

## Lean version

Tested with Lean 4.31.0 / Lake 5.0.0, IREE built from source against
ROCm 7.2.0 / gfx1100.

## Citing this work

```bibtex
@software{koonce2026leanmlir,
  author  = {Brett Koonce},
  title   = {Verified Deep Learning with Lean 4: Formal Backpropagation from MLP to Attention, via MLIR},
  url     = {https://github.com/brettkoonce/lean4-mlir},
  doi     = {10.5281/zenodo.20402133},
  version = {0.6.1},
  year    = {2026},
}
```
