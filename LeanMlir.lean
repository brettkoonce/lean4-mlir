import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.GradcheckHelpers
import LeanMlir.ViTRender
import LeanMlir.SpecHelpers
import LeanMlir.Train
import LeanMlir.VerifiedTrain
import LeanMlir.VerifiedSpec
import LeanMlir.VerifiedNets
import LeanMlir.Ddpm
import LeanMlir.Cam
-- VJP proofs (Attention pulls in Tensor/MLP/Residual/SE/LayerNorm/BatchNorm
-- transitively; CNN + Depthwise need explicit imports).
import LeanMlir.Proofs.Architectures.Attention
import LeanMlir.Proofs.Architectures.CNN
import LeanMlir.Proofs.Architectures.Depthwise
-- End-to-end whole-network VJP compositions (each builds on the CNN/Depthwise
-- machinery; their own file imports pull in everything transitively).
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXt
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNet
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2Close
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StagesPC
import LeanMlir.Proofs.Codegen.EfficientNetRenderPC
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetChainClose
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtChainClose
import LeanMlir.Proofs.Nets.ViT.ViTFwdGraph
import LeanMlir.Proofs.Nets.ViT.ViTClose
import LeanMlir.Proofs.Nets.ViT.ViTChainClose
import LeanMlir.Proofs.Nets.ViT.ViTVecLN
import LeanMlir.Proofs.Nets.ViT.ViTMultiHead
import LeanMlir.Proofs.Nets.ViT.ViTDepthK
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullPaper
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFullT
-- ℝ→Float32 bridge, Tier 1: standard-model rounding bounds for the toy nets.
import LeanMlir.Proofs.Float.FloatBridge
import LeanMlir.Proofs.Float.ConvMixedFloatBridge
import LeanMlir.Proofs.Float.ConvMixedComposeBridge
import LeanMlir.Proofs.Float.DepthwiseMixedFloatBridge
-- Inexact-gradient descent over ℝ: the keystone the float budgets plug into.
import LeanMlir.Proofs.Training.SgdDescent
import LeanMlir.Proofs.Training.SgdDescentLinear
import LeanMlir.Proofs.Training.SgdDescentMlp
import LeanMlir.Proofs.Training.SgdDescentCnn
-- Robustness certificate: the Lipschitz-margin certified radius (cert ≤ TRUE ≤ PGD).
import LeanMlir.Proofs.Certificates.LipschitzCert
-- The real Gaussian probit: Φ/Φ⁻¹ facts + the smoothing radius at the true quantile.
import LeanMlir.Proofs.Certificates.SmoothingGaussian
-- Verified-codegen bridges (denoted IR + per-op bridge theorems) so doc-gen4
-- documents them. IRPrint.lean is deliberately left out: its file-writing
-- #evals run at elaboration time (use `lake env lean …/IRPrint.lean`).
import LeanMlir.Proofs.Foundation.IR
-- Spec→math ties (rungs B/C/E). Also a Certs root + audited in
-- tests/AuditAxioms.lean since 2026-07-07: it rotted while orphaned
-- from every target (the mnv2 6→17-block spec promotion broke its rfl tie).
import LeanMlir.Proofs.SpecVJP

/-! # Verified Deep Learning with Lean 4 — the API docs

doc-gen4's rendering of the project's Lean: `LeanMlir` — the `NetSpec` architectures, the
StableHLO renderer and the trainers — and the `Proofs` corpus behind the book
[*Verified Deep Learning with Lean 4*](https://lean.brettkoonce.com/blueprint/), where every
theorem is one click from its statement here. This page is the map. The file-level one is
[`LeanMlir/Proofs/README.md`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/README.md);
the argument gap by gap is the book's
[On Verification](https://lean.brettkoonce.com/blueprint/app-verification.html); the search box
above finds any declaration by name.

## Start here

Two things are proved for every net, and the linear classifier shows both in about 650 lines:

1. **Faithfulness** — the *emitted* StableHLO train step denotes the *certified* forward,
   gradient and SGD math. [`LinearTrainStep`](LeanMlir/Proofs/Nets/Small/LinearTrainStep.html)
   is the spec and its ops; the capstone
   [`poc_train_step_tail_certified`](find/#doc/Proofs.LinPoC.poc_train_step_tail_certified) in
   [`LinearFold`](LeanMlir/Proofs/Nets/Small/LinearFold.html) is emitted step = certified math.
2. **Descent** — that step provably decreases the loss: `Proofs.linear_sgd_descends` in
   `LeanMlir.Proofs.Training.SgdDescentLinear`.

`lake build ProofsMinimal` builds exactly this slice, about a minute after `lake exe cache get`.

## The foundation

`Proofs.pdiv` is *defined* over Mathlib's `fderiv` — `pdiv f x i j := fderiv ℝ f x (basisVec i) j`,
in `LeanMlir.Proofs.Foundation.Tensor` — and the calculus is theorems from Mathlib's API:
`Proofs.pdiv_comp` (the chain rule), `Proofs.pdiv_add`, `Proofs.pdiv_mul`, `Proofs.biPath_has_vjp`
(the additive fan-in). A layer's backward is a `Proofs.HasVJP` witness whose `correct` field says
it is the Jacobian-transpose of the forward, and every layer has one: convolution and max-pool in
`LeanMlir.Proofs.Architectures.CNN`; BatchNorm's three-term backward in
`LeanMlir.Proofs.Architectures.BatchNorm`; `LeanMlir.Proofs.Architectures.Depthwise`,
`LeanMlir.Proofs.Architectures.Residual`, `LeanMlir.Proofs.Architectures.SE`,
`LeanMlir.Proofs.Architectures.LayerNorm` (with GELU); softmax, scaled-dot-product and multi-head
attention up to the ViT body in `LeanMlir.Proofs.Architectures.Attention`. The emitted graph is the
`SHlo` AST of [`StableHLO`](LeanMlir/Proofs/Codegen/StableHLO.html), and its `den` is the ℝ
denotation every tie below is stated about. Zero project axioms: every theorem closes under
`propext`, `Classical.choice` and `Quot.sound` alone, and
[`tests/comparator`](https://github.com/brettkoonce/lean4-mlir/tree/main/tests/comparator)
re-runs Lean's kernel over 73 of the headline theorems independently of the elaborator.

## The chapter nets

Whole-network VJPs — [`resnet34ForwardB_full_has_vjp_at`](find/#doc/Proofs.resnet34ForwardB_full_has_vjp_at),
`Proofs.mobilenetv2_has_vjp_at`, `Proofs.efficientnet_has_vjp`, `Proofs.convnext_has_vjp`,
`Proofs.vit_full_has_vjp` — and, for every net the tiers train, the tie of the committed
train-step render (the verified_mlir/ files the trainers load) to the certified chain, at the
batch BatchNorm and the ImageNet head its artifacts run: each emitted parameter-update node
denotes the certified descent step.

* ResNet-34 (chapter 5) — [`r34_net_tiedB`](find/#doc/Proofs.ResNet34TieB.r34_net_tiedB);
  non-degeneracy on the full-width batched net,
  [`ResNet34FullBSeal`](LeanMlir/Proofs/Nets/ResNet/ResNet34FullBSeal.html); the data-parallel
  render is synchronised BatchNorm and its step IS the single-device step at the global batch,
  [`ResNet34SyncStepTieB`](LeanMlir/Proofs/Nets/ResNet/ResNet34SyncStepTieB.html)
* ResNet-50 — [`r50_net_tiedB`](find/#doc/Proofs.ResNet50TieB.r50_net_tiedB);
  [`ResNet50FullBSeal`](LeanMlir/Proofs/Nets/ResNet/ResNet50FullBSeal.html)
* MobileNetV2 (chapter 6) — [`mnv2_net_tiedB`](find/#doc/Proofs.MobileNetV2TieB.mnv2_net_tiedB);
  [`MobileNetV2FullBSeal`](LeanMlir/Proofs/Nets/MobileNet/MobileNetV2FullBSeal.html)
* MobileNetV4-Conv-M — [`mnv4_net_tiedB`](find/#doc/Proofs.Mnv4TieB.mnv4_net_tiedB);
  [`MobileNetV4FullBSeal`](LeanMlir/Proofs/Nets/MobileNet/MobileNetV4FullBSeal.html)
* EfficientNet-B0 (chapter 7) —
  [`efficientnet_net_tiedG`](find/#doc/Proofs.EnetTiePoCG.efficientnet_net_tiedG)
* ConvNeXt-T (chapter 8) — [`cnx_net_tiedGB`](find/#doc/Proofs.CnxTiePoCGB.cnx_net_tiedGB)
* ViT-Tiny (chapter 9) — [`vit_net_tiedGB`](find/#doc/Proofs.ViTTiePoCGB.vit_net_tiedGB)

The per-net trees under `LeanMlir.Proofs.Nets` repeat the linear pattern. The four conv nets share
one file chain — `*BackB0` (block backward graphs) → `*FullB` (forward + T2 graph) → `*FullBVJP`
(T1 VJP) → `*FullBSeal` (non-degeneracy) → `*StepTieB` (T3 train-step tie) →
`*WholeBackCertifiedTieB` (T6 whole-net backward; ResNet-34's is `ResNet34BackCertifiedTieB`) →
`*SyncB` / `*SyncStepTieB` (data-parallel). EfficientNet, ConvNeXt and ViT spell the same tiers
with other suffixes; the table and the suffix legend are in the
[proofs README](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/README.md).

## Around the ties

* **Descent over ℝ** — `LeanMlir.Proofs.Training.SgdDescent`, with the linear, MLP and CNN
  capstones beside it: the SGD step decreases the loss, the float budget as its inexactness.
* **The float model** — `LeanMlir.Proofs.Float.FloatBridge`: standard-model rounding bounds and
  `FloatClose` for the toy nets; past them the ℝ→Float32 gap stays trusted.
* **Data parallelism** — the all-reduce as an AST node and which function a data-parallel run
  minimises, [`DataParallel`](LeanMlir/Proofs/Foundation/DataParallel.html); the sync-BatchNorm
  kit in [`DataParallelSync`](LeanMlir/Proofs/Foundation/DataParallelSync.html).
* **Robustness certificates** — the Lipschitz-margin radius in
  `LeanMlir.Proofs.Certificates.LipschitzCert` and the smoothing radius at the true Gaussian
  quantile in `LeanMlir.Proofs.Certificates.SmoothingGaussian`.

## What stays trusted

The ℝ→Float32 numerics, the per-op text printing of the graph, and the lowerer — XLA/PJRT, or
IREE — with its runtime; and at the kinks of ReLU, ReLU6 and max-pool, where `fderiv` is `0`,
the standard subgradient convention the codegen emits. The book's appendix takes each in turn.

## Build

```
lake exe cache get          # Mathlib oleans, ~30 s
lake build ProofsMinimal    # the linear on-ramp above, ~1 min
lake build Proofs           # the engine slice the trainers import
lake build Certs            # the certificate corpus CI checks; tests/AuditAxioms.lean is its axiom audit
```
-/
