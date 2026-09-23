import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.Nets.Small.MnistCNN
import LeanMlir.Proofs.Nets.Small.CifarCNN
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullPaper
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNet
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullB0
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXt
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtFullT
import LeanMlir.Proofs.Architectures.Attention
import LeanMlir.Proofs.Nets.ViT.ViTDepthK
import LeanMlir.Proofs.Nets.ResNet.ResNet34FullB
import LeanMlir.Proofs.Codegen.StableHLO

/-! # Spec → math (the verification tie), Rung 1: the linear classifier

The shape `#guard` beside `resnet34Verified` in `VerifiedNets.lean` only checks the
*parameter interface*
(typechecking). This file is the first rung of connecting a readable `VerifiedNetSpec`
to the actual **math** — the proven VJP — on the simplest net, the Chapter-1 linear
classifier (`dense 784→10`).

The pattern (extends to MLP → conv nets, each rigid/per-net):
  1. `denote` maps the spec's layers to the Mathlib math function the proofs are about;
  2. a `rfl` lemma ties the spec's denotation to that named function (`mnistLinear`);
  3. the whole-model VJP theorem is stated about *the spec's denotation* and discharged
     by the audited op-level VJP (`dense_has_vjp`).

If the spec's `layers` drifts from `[.dense 784 10]`, step 2/3 stop reducing and the
proofs fail to typecheck — so the readable architecture is provably the verified one,
at the math level, not just the shape level.
-/

open Proofs

/- `linearVerified` (the single dense 784→10 spec) is imported from `LeanMlir.VerifiedNets`
   — the *same* object `MainMnistLinearVerified` trains, so the VJP below is about the
   trainer's exact spec, not a copy. The shape tie (`toSpecs == …`) lives there too. -/

/-- Math denotation of the linear spec. The Chapter-1 model is a single dense layer, so
    `[.dense 784 10]` denotes to the Mathlib `dense W b`. Any other layer list is not the
    linear model (`0`), which makes the tie below drift-sensitive. -/
noncomputable def denoteLinear (layers : List VLayer) (W : Mat 784 10) (b : Vec 10) :
    Vec 784 → Vec 10 :=
  match layers with
  | [.dense 784 10] => dense W b
  | _               => fun _ => 0

/-- **Spec ≡ the proven model.** `linearVerified`'s denotation is exactly `mnistLinear`
    (the function the Chapter-1 VJP capstone is about) — by `rfl`, so it's checked by the
    kernel and breaks if `linearVerified.layers` changes. -/
theorem linearVerified_denote_eq (W : Mat 784 10) (b : Vec 10) :
    denoteLinear linearVerified.layers W b = mnistLinear W b := rfl

/-- **The spec carries the math.** The linear spec's denotation has the proven VJP —
    discharged by the audited `dense_has_vjp`. This is the whole-model verification
    stated about the *readable layer list*, not a hand-written function. -/
noncomputable def linearVerified_has_vjp (W : Mat 784 10) (b : Vec 10) :
    HasVJP (denoteLinear linearVerified.layers W b) :=
  dense_has_vjp W b

/-- …and its correctness headline carries over verbatim (the backward is the
    `pdiv`-contracted Jacobian of the spec's denotation). -/
theorem linearVerified_has_vjp_correct (W : Mat 784 10) (b : Vec 10)
    (x : Vec 784) (dy : Vec 10) (i : Fin 784) :
    (linearVerified_has_vjp W b).backward x dy i
      = ∑ j : Fin 10, pdiv (denoteLinear linearVerified.layers W b) x i j * dy j :=
  (linearVerified_has_vjp W b).correct x dy i

/-! ## Rung 2: the MLP — the first genuine `vjp_comp` fold

The linear model was the degenerate case (one layer, no fold). The MLP's denotation is a
*chain* — `dense ∘ relu ∘ dense ∘ relu ∘ dense` (`mlpForward`) — and its VJP is built by
folding `vjp_comp_at` down that chain (`mlp_has_vjp_at`). So this is where the spec→math
tie first exercises the chain rule, not just a single op. -/

/-- Math denotation of the MLP spec: the 5-layer list denotes to `mlpForward`. -/
noncomputable def denoteMLP (layers : List VLayer)
    (W₀ : Mat 784 512) (b₀ : Vec 512) (W₁ : Mat 512 512) (b₁ : Vec 512)
    (W₂ : Mat 512 10) (b₂ : Vec 10) : Vec 784 → Vec 10 :=
  match layers with
  | [.dense 784 512, .relu, .dense 512 512, .relu, .dense 512 10] =>
      mlpForward W₀ b₀ W₁ b₁ W₂ b₂
  | _ => fun _ => 0

/-- **Spec ≡ the proven model.** `mlpVerified`'s denotation is exactly `mlpForward`
    (`dense ∘ relu ∘ dense ∘ relu ∘ dense`) — by `rfl`, drift-sensitive. -/
theorem mlpVerified_denote_eq (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10) :
    denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂ = mlpForward W₀ b₀ W₁ b₁ W₂ b₂ := rfl

/-- **The spec carries the math (canonical witness).** The MLP spec's denotation has a
    VJP — the global `pdiv`-derived witness (`mlp_has_vjp`; relu uses the framework
    subgradient convention at the kinks, per `Proofs/README.md`). -/
noncomputable def mlpVerified_has_vjp (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10) :
    HasVJP (denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂) :=
  mlp_has_vjp W₀ b₀ W₁ b₁ W₂ b₂

/-- **The spec carries the math (the real fold).** At a smooth input — the two ReLU
    pre-activations avoid zero — the MLP spec's denotation has a VJP built by *folding*
    `vjp_comp_at` through `dense → relu → dense → relu → dense` (no `rfl` escape at the
    kinks). This is the chain rule applied to the spec, the step linear couldn't show. -/
noncomputable def mlpVerified_has_vjp_at (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10) (x : Vec 784)
    (h0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h1 : ∀ k, dense W₁ b₁ (relu 512 (dense W₀ b₀ x)) k ≠ 0) :
    HasVJPAt (denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂) x :=
  mlp_has_vjp_at W₀ b₀ W₁ b₁ W₂ b₂ x h0 h1

/-- …correctness headline for the canonical witness carries over to the spec. -/
theorem mlpVerified_has_vjp_correct (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10)
    (x : Vec 784) (dy : Vec 10) (i : Fin 784) :
    (mlpVerified_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).backward x dy i
      = ∑ j : Fin 10, pdiv (denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  (mlpVerified_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).correct x dy i

/-! ## Rung 3: the CNN — the fold now runs through conv + maxpool

The CNN's denotation is `mnistCnnNoBnForward` — a flat `Vec 784 → Vec 10` chain
`flatConv → relu → flatConv → relu → maxPoolFlat → dense → relu → dense → relu → dense`.
The honest chain-rule fold (via `vjp_comp_at` through conv/maxpool/dense) is the audited
`mnistCnnNoBn_has_vjp_at`, conditional on the four ReLU kinks + the maxpool being smooth at
the input. Here we headline the unconditional canonical witness (`mlp_has_vjp` style); the
spec is exactly the subject of that conditional fold via `cnnVerified_denote_eq`. -/

/-- Math denotation of the CNN spec: the 11-layer list denotes to `mnistCnnNoBnForward`
    (`c=32`, `h=w=14`, the Chapter-3 MNIST CNN). -/
noncomputable def denoteCNN (layers : List VLayer)
    (W₁ : Kernel4 32 1 3 3) (b₁ : Vec 32) (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32)
    (W₃ : Mat 6272 512) (b₃ : Vec 512) (W₄ : Mat 512 512) (b₄ : Vec 512)
    (W₅ : Mat 512 10) (b₅ : Vec 10) : Vec 784 → Vec 10 :=
  match layers with
  | [.conv 1 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2, .flatten,
     .dense 6272 512, .relu, .dense 512 512, .relu, .dense 512 10] =>
      mnistCnnNoBnForward (h := 14) (w := 14) W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
  | _ => fun _ => 0

/-- **Spec ≡ the proven model.** `cnnVerified`'s denotation is exactly `mnistCnnNoBnForward`
    — the function the Chapter-3 fold `mnistCnnNoBn_has_vjp_at` is about — by `rfl`. -/
theorem cnnVerified_denote_eq (W₁ : Kernel4 32 1 3 3) (b₁ : Vec 32)
    (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32) (W₃ : Mat 6272 512) (b₃ : Vec 512)
    (W₄ : Mat 512 512) (b₄ : Vec 512) (W₅ : Mat 512 10) (b₅ : Vec 10) :
    denoteCNN cnnVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
      = mnistCnnNoBnForward (h := 14) (w := 14) W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ := rfl

/-- **The spec carries the math.** The CNN spec's denotation (conv→relu→conv→relu→maxpool
    →dense→…) has a VJP — the canonical `pdiv`-derived witness. The conditional chain-rule
    fold through conv/maxpool is the audited `mnistCnnNoBn_has_vjp_at`. -/
noncomputable def cnnVerified_has_vjp (W₁ : Kernel4 32 1 3 3) (b₁ : Vec 32)
    (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32) (W₃ : Mat 6272 512) (b₃ : Vec 512)
    (W₄ : Mat 512 512) (b₄ : Vec 512) (W₅ : Mat 512 10) (b₅ : Vec 10) :
    HasVJP (denoteCNN cnnVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅) := HasVJP.canonical _

/-! ## Rung E (linear): the spec ↔ the *generated MLIR*

The ties above connect the spec to the **math** (`denote` = the proven forward, which has
the proven VJP). This connects the spec to the **StableHLO the trainer actually compiles
and runs**: the generated forward graph `fwdGraph` (→ `verified_mlir/linear_fwd.mlir`, the
eval path) and the train-step loss-cotangent graph `lossCotGraph` (→ `linear_train_step.mlir`)
*denote* the spec's forward and its softmax-CE gradient — via the audited faithfulness
theorems (`fwdGraph_faithful`, `lossCotGraph_isCEgrad`) composed with `denoteLinear =
mnistLinear` (`rfl`). So the generated code provably computes the spec's function.

What stays trusted (the codegen boundary, per `Proofs/README.md`): the text render
`linearFwdModuleV = pretty (emit fwdGraph)` and that the committed `.mlir` equals that
text — the pretty-printer + regeneration, NOT the semantics, which are proven here. -/

open Proofs.StableHLO in
/-- **Generated forward MLIR ↔ spec.** The forward graph (rendered to `linear_fwd.mlir`,
    the eval path) denotes the spec's forward function. -/
theorem linearVerified_fwd_faithful (W : Mat 784 10) (b : Vec 10) (x : Vec 784) :
    den (fwdGraph W b x) = denoteLinear linearVerified.layers W b x := by
  exact fwdGraph_faithful W b x

open Proofs.StableHLO in
/-- **Generated train-step cotangent ↔ spec.** The loss-cotangent graph (in
    `linear_train_step.mlir`) denotes `∂(softmax-CE)/∂logits` at the spec's logits. -/
theorem linearVerified_lossCot_isCEgrad (W : Mat 784 10) (b : Vec 10) (x : Vec 784)
    (label : Fin 10) (j : Fin 10) :
    den (lossCotGraph W b x (oneHot 10 label)) j
      = pdiv (fun (z : Vec 10) (_ : Fin 1) => crossEntropy 10 z label)
             (denoteLinear linearVerified.layers W b x) j 0 := by
  exact lossCotGraph_isCEgrad W b x label j

/-! ## Rung E (MLP): the spec ↔ the generated MLIR — both forward *and* backward

The MLP has faithfulness for the whole forward graph (`mlpFwdGraph_faithful`, the graph
`mlp_fwd.mlir` prints) AND for a whole backward input-VJP graph (`mlpBackGraph_faithful`).
Composed with `denoteMLP = mlpForward` and `mlpVerified_has_vjp_at = mlp_has_vjp_at`, both
denote the spec: the rendered forward computes the spec's forward, and `mlpBackGraph`
computes the spec's VJP backward (at a smooth input). `mlpBackGraph` is a spec-level graph
no committed artifact prints: `mlp_train_step.mlir` is `MlpRender.lean`'s
`mlpTrainStepFaithfulV`, whose parameter ops `MlpFold` ties to the certified step. -/

open Proofs.StableHLO in
/-- **Generated MLP forward MLIR ↔ spec.** The forward graph (→ `mlp_fwd.mlir`) denotes
    the spec's forward function. -/
theorem mlpVerified_fwd_faithful (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10) (x : Vec 784) :
    den (mlpFwdGraph W₀ b₀ W₁ b₁ W₂ b₂ x)
      = denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂ x := by
  exact mlpFwdGraph_faithful W₀ b₀ W₁ b₁ W₂ b₂ x

open Proofs.StableHLO in
/-- **MLP backward graph ↔ spec.** `mlpBackGraph`, the input-VJP graph (spec-level; no
    committed artifact prints it, see the section header), denotes the spec's VJP backward
    (`mlpVerified_has_vjp_at`), at a smooth input (the two ReLU pre-activations avoid zero). -/
theorem mlpVerified_back_faithful (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10) (x : Vec 784)
    (h0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h1 : ∀ k, dense W₁ b₁ (relu 512 (dense W₀ b₀ x)) k ≠ 0) (dy : Vec 10) :
    den (mlpBackGraph W₀ W₁ W₂ (dense W₀ b₀ x)
          (dense W₁ b₁ (relu 512 (dense W₀ b₀ x))) dy)
      = (mlpVerified_has_vjp_at W₀ b₀ W₁ b₁ W₂ b₂ x h0 h1).backward dy := by
  exact mlpBackGraph_faithful W₀ b₀ W₁ b₁ W₂ b₂ x h0 h1 dy

/-! ## Rung E (CNN): the spec ↔ the generated MLIR (forward)

The generated CNN forward graph (`flatConv→relu→flatConv→relu→maxPoolFlat→dense→relu→
dense→relu→dense`) denotes the spec's forward. The backward graph faithfulness exists too
(`cnnBackGraph_faithful` denotes `mnistCnnNoBn_has_vjp_at.backward` — the VJP of exactly
this spec's forward), but it carries the same five ReLU/maxpool smoothness hypotheses as
the conditional fold, so we headline the unconditional forward tie (matching
`cnnVerified_has_vjp`, the canonical witness). -/

open Proofs.StableHLO in
/-- **Generated CNN forward MLIR ↔ spec.** The forward graph (→ `cnn_fwd.mlir`) denotes
    the spec's forward (`mnistCnnNoBnForward`, c=32 / h=w=14). -/
theorem cnnVerified_fwd_faithful (W₁ : Kernel4 32 1 3 3) (b₁ : Vec 32)
    (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32) (W₃ : Mat 6272 512) (b₃ : Vec 512)
    (W₄ : Mat 512 512) (b₄ : Vec 512) (W₅ : Mat 512 10) (b₅ : Vec 10) (x : Vec 784) :
    den (cnnFwdGraph (h := 14) (w := 14) W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x)
      = denoteCNN cnnVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x := by
  exact cnnFwdGraph_faithful (h := 14) (w := 14) W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x

/-! ## Rung 4 + E (CIFAR): completing the ch5 ladder

The CIFAR-10 net (ic=3, c1=32, c2=64, h=w=8 — spatial 32→16→8) gets the spec→math
denotation (= `cifarCnnForward` by `rfl`), the canonical witness VJP, and the forward
spec→generated-MLIR tie (`cifarFwdGraph_faithful`). The conditional fold is
`cifarCnn_has_vjp_at` (six ReLU kinks + two maxpools). -/

noncomputable def denoteCifar (layers : List VLayer)
    (W₁ : Kernel4 32 3 3 3) (b₁ : Vec 32) (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32)
    (W₃ : Kernel4 64 32 3 3) (b₃ : Vec 64) (W₄ : Kernel4 64 64 3 3) (b₄ : Vec 64)
    (W₅ : Mat 4096 512) (b₅ : Vec 512) (W₆ : Mat 512 512) (b₆ : Vec 512)
    (W₇ : Mat 512 10) (b₇ : Vec 10) : Vec 3072 → Vec 10 :=
  match layers with
  | [.conv 3 32 3 1, .relu, .conv 32 32 3 1, .relu, .maxPool 2 2,
     .conv 32 64 3 1, .relu, .conv 64 64 3 1, .relu, .maxPool 2 2, .flatten,
     .dense 4096 512, .relu, .dense 512 512, .relu, .dense 512 10] =>
      cifarCnnForward (h := 8) (w := 8) W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇
  | _ => fun _ => 0

theorem cifarVerified_denote_eq
    (W₁ : Kernel4 32 3 3 3) (b₁ : Vec 32) (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32)
    (W₃ : Kernel4 64 32 3 3) (b₃ : Vec 64) (W₄ : Kernel4 64 64 3 3) (b₄ : Vec 64)
    (W₅ : Mat 4096 512) (b₅ : Vec 512) (W₆ : Mat 512 512) (b₆ : Vec 512)
    (W₇ : Mat 512 10) (b₇ : Vec 10) :
    denoteCifar cifarVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇
      = cifarCnnForward (h := 8) (w := 8) W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ := rfl

/-- **The (no-BN) CIFAR spec carries the math.** -/
noncomputable def cifarVerified_has_vjp
    (W₁ : Kernel4 32 3 3 3) (b₁ : Vec 32) (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32)
    (W₃ : Kernel4 64 32 3 3) (b₃ : Vec 64) (W₄ : Kernel4 64 64 3 3) (b₄ : Vec 64)
    (W₅ : Mat 4096 512) (b₅ : Vec 512) (W₆ : Mat 512 512) (b₆ : Vec 512)
    (W₇ : Mat 512 10) (b₇ : Vec 10) :
    HasVJP (denoteCifar cifarVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇) :=
  HasVJP.canonical _

open Proofs.StableHLO in
/-- **Generated (no-BN) CIFAR forward MLIR ↔ spec.** -/
theorem cifarVerified_fwd_faithful
    (W₁ : Kernel4 32 3 3 3) (b₁ : Vec 32) (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32)
    (W₃ : Kernel4 64 32 3 3) (b₃ : Vec 64) (W₄ : Kernel4 64 64 3 3) (b₄ : Vec 64)
    (W₅ : Mat 4096 512) (b₅ : Vec 512) (W₆ : Mat 512 512) (b₆ : Vec 512)
    (W₇ : Mat 512 10) (b₇ : Vec 10) (x : Vec 3072) :
    den (cifarFwdGraph (h := 8) (w := 8) W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x)
      = denoteCifar cifarVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x := by
  exact cifarFwdGraph_faithful (h := 8) (w := 8) W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x

/-! ## Rung B/C/E (ch7 MobileNetV2, FULL, batched): the committed spec ↔ the batch-BN net

`denoteMobilenetB` maps `mobilenetv2Verified.layers` — the committed 21-entry full-paper
`[t,c,n,s]` list the trainer runs (stem-s2 3→32 → 17 bottlenecks → 1×1 head 320→1280 → GAP →
dense 1280→10) — to `mobilenetv2ForwardB_full` (`MobileNetV2FullB.lean`: batch BN throughout,
the t=1 no-expand first block, 4 stride-2 depthwise downsamples 224→7), at every batch size.
Weights ride in the `MNV2BWeights` bundle, so the tie stays readable. The `rfl` is
drift-sensitive: any `[t,c,n,s]` edit to the spec stops the match reducing — the tripwire the
6→17-block promotion once fired while the per-example twin of this section was orphaned;
certs.yml re-elaborates it on every spec push. (That per-example twin, at the forward of the
retired SGD artifact, was retired on 2026-09-19.) -/

-- ── MobileNetV2 (FULL, BATCHED): the same 21-entry spec ↔ mobilenetv2ForwardB_full ──

/-- Math denotation of the committed MobileNetV2 spec at batch BN: the 21-entry full-paper
    layer list denotes to `mobilenetv2ForwardB_full` — the batch-statistics net every shipped
    MobileNetV2 artifact runs (`MobileNetV2FullB.lean`), at every batch size `N`. Any other
    list is not the net (`0`), so the tie below is drift-sensitive. -/
noncomputable def denoteMobilenetB (N : Nat) (layers : List VLayer) (w : MNV2BWeights 10) :
    Vec (N * (3 * 224 * 224)) → Vec (N * 10) :=
  match layers with
  | [.convBnNB 3 32 3 2,
     .invertedResidualNB 32 32 16 1,
     .invertedResidualNB 16 96 24 2, .invertedResidualNB 24 144 24 1,
     .invertedResidualNB 24 144 32 2, .invertedResidualNB 32 192 32 1, .invertedResidualNB 32 192 32 1,
     .invertedResidualNB 32 192 64 2, .invertedResidualNB 64 384 64 1, .invertedResidualNB 64 384 64 1,
     .invertedResidualNB 64 384 64 1,
     .invertedResidualNB 64 384 96 1, .invertedResidualNB 96 576 96 1, .invertedResidualNB 96 576 96 1,
     .invertedResidualNB 96 576 160 2, .invertedResidualNB 160 960 160 1, .invertedResidualNB 160 960 160 1,
     .invertedResidualNB 160 960 320 1,
     .convBnNB 320 1280 1 1, .globalAvgPool, .dense 1280 10] =>
      mobilenetv2ForwardB_full N w
  | _ => fun _ => 0

/-- **Spec ≡ the full batch-BN net.** `mobilenetv2Verified`'s denotation at batch `N` is
    exactly `mobilenetv2ForwardB_full N` — by `rfl`, drift-sensitive. -/
theorem mobilenetv2VerifiedB_denote_eq (N : Nat) (w : MNV2BWeights 10) :
    denoteMobilenetB N mobilenetv2Verified.layers w = mobilenetv2ForwardB_full N w := rfl

/-- **The committed spec carries the math at batch BN** — canonical `pdiv` witness (relu6 is
    kinked; the pointwise whole-net VJP is `mobilenetv2ForwardB_full_has_vjp_at_correct`). -/
noncomputable def mobilenetv2VerifiedB_has_vjp (N : Nat) (w : MNV2BWeights 10) :
    HasVJP (denoteMobilenetB N mobilenetv2Verified.layers w) := HasVJP.canonical _

open Proofs.StableHLO in
/-- **Rung E at the committed spec, batched.** The typed graph the shipped MobileNetV2
    artifacts are printed from denotes the committed spec's function at batch BN:
    `mobilenetv2FwdGraphB_full_faithful` composed with the tie. -/
theorem mobilenetv2VerifiedB_fwd_faithful (N : Nat) (epsStr : String) (w : MNV2BWeights 10)
    (e : SHlo (N * (3 * 224 * 224))) :
    den (mobilenetv2FwdGraphB_full N epsStr w e)
      = denoteMobilenetB N mobilenetv2Verified.layers w (den e) :=
  (mobilenetv2FwdGraphB_full_faithful N epsStr w e).trans
    (congrFun (mobilenetv2VerifiedB_denote_eq N w).symm (den e))


/-! ## Rung B/C/E (FULL, unified weight bundles): r34 / enet / convnext / vit

The mnv2 full-paper pattern applied to the remaining imagenette nets: each committed
spec's ENTIRE layer list (literal dims, drift-sensitive) denotes the full proven
forward, with weights riding a structure bundle so the ties stay readable. Existing
bundles are reused where the Full module already has one (`B0Weights`,
`CnxTWeightsCh`, `R34BWeights`); vit gets its bundle here (`ViTTinyWeights`, SpecVJP-local so
no proof module's signature changes). Rung E composes each net's
full graph-faithfulness apex with the tie (vit's is `vitFwdGraphKMHV_faithful`,
ViTDepthK §3 — the depth-`k` multi-head vector-LN graph). Rung C is the canonical
witness except vit: all-smooth, so vit's rung C is the REAL whole-net VJP
`vitForwardKV_has_vjp` (only `0 < ε`) at the committed spec. -/

-- ── ResNet-34 (FULL, batched): the committed 8-entry spec ↔ resnet34ForwardB_full ──

/-- Math denotation of the committed ResNet-34 spec at batch BN: the 8-entry stage-level list
    denotes to `resnet34ForwardB_full` — the batch-statistics net every shipped ResNet-34
    artifact runs (`ResNet34FullB.lean`), at every batch size `N`. Any other list is not the net
    (`0`), so the tie below is drift-sensitive. (The per-example rung, at the forward of the
    retired SGD artifact, was retired with it on 2026-09-19.) -/
noncomputable def denoteR34FullB (N : Nat) (layers : List VLayer) (w : R34BWeights 10) :
    Vec (N * (3 * 224 * 224)) → Vec (N * 10) :=
  match layers with
  | [.convBnNB 3 64 7 2, .maxPool 3 2,
     .residualStage 64 64 3 1, .residualStage 64 128 4 2,
     .residualStage 128 256 6 2, .residualStage 256 512 3 2,
     .globalAvgPool, .dense 512 10] => resnet34ForwardB_full N w
  | _ => fun _ => 0

/-- **Spec ≡ the full batch-BN net.** `resnet34Verified`'s denotation at batch `N` is exactly
    `resnet34ForwardB_full N` — by `rfl`, drift-sensitive. -/
theorem resnet34VerifiedB_denote_eq (N : Nat) (w : R34BWeights 10) :
    denoteR34FullB N resnet34Verified.layers w = resnet34ForwardB_full N w := rfl

/-- **The committed spec carries the math at batch BN** — canonical `pdiv` witness (relu is
    kinked; the pointwise whole-net VJP is `resnet34ForwardB_full_has_vjp_at`). -/
noncomputable def resnet34VerifiedB_has_vjp (N : Nat) (w : R34BWeights 10) :
    HasVJP (denoteR34FullB N resnet34Verified.layers w) := HasVJP.canonical _

open Proofs.StableHLO in
/-- **Rung E at the committed spec, batched.** The typed graph the shipped ResNet-34 artifacts
    are printed from denotes the committed spec's function at batch BN:
    `resnet34FwdGraphB_full_faithful` composed with the tie. -/
theorem resnet34VerifiedB_fwd_faithful (N : Nat) (epsStr : String) (w : R34BWeights 10)
    (e : SHlo (N * (3 * 224 * 224))) :
    den (resnet34FwdGraphB_full N epsStr w e)
      = denoteR34FullB N resnet34Verified.layers w (den e) :=
  (resnet34FwdGraphB_full_faithful N epsStr w e).trans
    (congrFun (resnet34VerifiedB_denote_eq N w).symm (den e))

-- ── EfficientNet-B0 (FULL, batched): the committed 21-entry spec ↔ efficientnetForwardB_full ──

/-- Math denotation of the committed EfficientNet-B0 spec at batch `N`: the 21-entry
    `[t,c,n,s,k]` layer list denotes to `efficientnetForwardB_full` (all 16 MBConv
    blocks, true batch-norm + SE). The spec ties the batched net at EVERY batch size. -/
noncomputable def denoteEfficientnetB0 (N : Nat) (layers : List VLayer) (w : B0Weights) :
    Vec (N * (3 * 224 * 224)) → Vec (N * 10) :=
  match layers with
  | [.convBnNB 3 32 3 2,
     .mbConvSENB 32 32 16 8 3,
     .mbConvSENB 16 96 24 4 3, .mbConvSENB 24 144 24 6 3,
     .mbConvSENB 24 144 40 6 5, .mbConvSENB 40 240 40 10 5,
     .mbConvSENB 40 240 80 10 3, .mbConvSENB 80 480 80 20 3, .mbConvSENB 80 480 80 20 3,
     .mbConvSENB 80 480 112 20 5, .mbConvSENB 112 672 112 28 5, .mbConvSENB 112 672 112 28 5,
     .mbConvSENB 112 672 192 28 5, .mbConvSENB 192 1152 192 48 5, .mbConvSENB 192 1152 192 48 5,
     .mbConvSENB 192 1152 192 48 5,
     .mbConvSENB 192 1152 320 48 3,
     .convBnNB 320 1280 1 1, .globalAvgPool, .dense 1280 10] =>
      efficientnetForwardB_full N w
  | _ => fun _ => 0

/-- **Spec ≡ the full proven net.** `efficientnetVerified`'s denotation is exactly
    `efficientnetForwardB_full` (16 MBConv, batched, per-channel BN + SE) — by `rfl`. -/
theorem efficientnetVerified_denote_eq (N : Nat) (w : B0Weights) :
    denoteEfficientnetB0 N efficientnetVerified.layers w
      = efficientnetForwardB_full N w := rfl

/-- **The committed spec carries the math** — canonical `pdiv` witness (swish/SE are
    smooth but relu6 clamps; the per-block differentiability lemmas live in
    `EfficientNetFullB0.lean`). -/
noncomputable def efficientnetVerified_has_vjp (N : Nat) (w : B0Weights) :
    HasVJP (denoteEfficientnetB0 N efficientnetVerified.layers w) := HasVJP.canonical _

open Proofs.StableHLO in
/-- **Rung E at the committed spec (batched).** The full 16-MBConv batched graph denotes
    the committed spec's function: `efficientnetFwdGraphB_full_faithful` ∘ the tie. -/
theorem efficientnetVerified_fwd_faithful (N : Nat) (epsStr : String) (w : B0Weights)
    (x : Vec (N * (3 * 224 * 224))) :
    den (efficientnetFwdGraphB_full N epsStr w x)
      = denoteEfficientnetB0 N efficientnetVerified.layers w x :=
  (efficientnetFwdGraphB_full_faithful N epsStr w x).trans
    (congrFun (efficientnetVerified_denote_eq N w).symm x)

-- ── ConvNeXt-T (FULL): the committed 27-entry spec ↔ convNextForwardTCh ──

/-- Math denotation of the committed ConvNeXt-T spec: the 29-entry `[3,3,9,3]` layer list
    denotes to `convNextForwardTCh` — the **channel**-LayerNorm net (§2m), whose 23 LN sites
    are 1 stem + 18 block + 3 downsample + 1 **head**, the first 22 reducing over the `c` channels
    at one spatial position with a per-channel `[c]` affine and the head one over the `[768]` GAP
    output (which is the same function at one spatial position — `rowLNVecFlat 1 768`).

    ⚠ The head LN was RESTORED 2026-08-30 (`planning/archive/next_session_execution_and_parity.md` §7.1).
    §2m/§2n had deleted it to match the JAX reference, which was itself missing it against both
    the paper and timm; the parameter count was short by exactly 2×768.

    ⚠ This used to match a `.convNextBlock`/`.bn` list and denote the SCALAR-LN net: one mean and
    one variance over the whole `c·h·w` map, two scalars, and no stem LN but a head LN. §2n
    deleted that chain outright, so the trap it guarded against — silently re-pointing this at the
    scalar function, which would typecheck by `rfl` and assert that the channel-LN layer list
    denotes the scalar-LN one (§2k's own sin, one level down) — is no longer expressible. Keeping
    the note because the SHAPE of that mistake is what §2k was about, not the specific symbol. -/
noncomputable def denoteConvnextT (layers : List VLayer) (w : CnxTWeightsCh 10) :
    Vec (3 * 224 * 224) → Vec 10 :=
  match layers with
  | [.conv 3 96 4 4, .layerNorm 96,
     .convNextBlockCh 96, .convNextBlockCh 96, .convNextBlockCh 96,
     .layerNorm 96, .conv 96 192 2 2,
     .convNextBlockCh 192, .convNextBlockCh 192, .convNextBlockCh 192,
     .layerNorm 192, .conv 192 384 2 2,
     .convNextBlockCh 384, .convNextBlockCh 384, .convNextBlockCh 384,
     .convNextBlockCh 384, .convNextBlockCh 384, .convNextBlockCh 384,
     .convNextBlockCh 384, .convNextBlockCh 384, .convNextBlockCh 384,
     .layerNorm 384, .conv 384 768 2 2,
     .convNextBlockCh 768, .convNextBlockCh 768, .convNextBlockCh 768,
     .globalAvgPool, .layerNorm 768, .dense 768 10] => convNextForwardTCh w
  | _ => fun _ => 0

/-- **Spec ≡ the full proven net.** `convnextVerified`'s denotation is exactly
    `convNextForwardTCh` ([3,3,9,3] @ [96,192,384,768], channel LN + head LN, 28,589,128 params at
    K = 1000 — the JAX reference's own count) — by `rfl`. -/
theorem convnextVerified_denote_eq (w : CnxTWeightsCh 10) :
    denoteConvnextT convnextVerified.layers w = convNextForwardTCh w := rfl

/-- **The committed spec carries the math** — canonical `pdiv` witness; the REAL
    whole-net VJP exists at full depth (`convNextForwardTCh_has_vjp_correct`,
    all-smooth, the 22 LN positivities only) on the ∘-chain form. -/
noncomputable def convnextVerified_has_vjp (w : CnxTWeightsCh 10) :
    HasVJP (denoteConvnextT convnextVerified.layers w) := HasVJP.canonical _

open Proofs.StableHLO in
/-- **Rung E at the committed spec.** The committed-config [3,3,9,3] channel-LN graph denotes
    the committed spec's function: `convNextFwdGraphTCh_faithful` ∘ the tie. -/
theorem convnextVerified_fwd_faithful (epsStr : String) (w : CnxTWeightsCh 10)
    (x : Vec (3 * 224 * 224)) :
    den (convNextFwdGraphTCh epsStr w x)
      = denoteConvnextT convnextVerified.layers w x :=
  (convNextFwdGraphTCh_faithful epsStr w x).trans
    (congrFun (convnextVerified_denote_eq w).symm x)

-- ── ViT-Tiny (FULL): the committed 17-entry spec ↔ vitForwardKV @ depth 12 ──

/-- All ViT-Tiny parameters at the committed config (D=192, 3 heads, d_head=64,
    mlpDim=768, 12 untied blocks, vector-LN): patch embed + CLS/pos + 12 per-block
    `BlockParamsV` bundles + final LN + CLS head. Shared LN ε rides along. -/
structure ViTTinyWeights where
  ε : ℝ
  Wc : Kernel4 192 3 16 16
  bc : Vec 192
  cls : Vec 192
  pos : Mat 197 192
  blocks : Fin 12 → BlockParamsV 192 768
  γF : Vec 192
  βF : Vec 192
  Wcls : Mat 192 10
  bcls : Vec 10

/-- `vitForwardKV` at the committed ViT-Tiny config (depth 12, 3 heads × 64). -/
noncomputable def vitForwardTiny (w : ViTTinyWeights) : Vec (3 * 224 * 224) → Vec 10 :=
  vitForwardKV 3 224 224 16 196 768 3 64 10 12
    w.Wc w.bc w.cls w.pos w.ε w.blocks w.γF w.βF w.Wcls w.bcls

/-- Math denotation of the committed ViT-Tiny spec: the 17-entry layer list (12 untied
    `.transformerBlock`s, per-channel `[192]` LN, 1D CLS) denotes to `vitForwardTiny`. -/
noncomputable def denoteVitTiny (layers : List VLayer) (w : ViTTinyWeights) :
    Vec (3 * 224 * 224) → Vec 10 :=
  match layers with
  | [.conv 3 192 16 16,
     .param #[192] 2, .param #[197, 192] 2,
     .transformerBlock 192 768, .transformerBlock 192 768, .transformerBlock 192 768,
     .transformerBlock 192 768, .transformerBlock 192 768, .transformerBlock 192 768,
     .transformerBlock 192 768, .transformerBlock 192 768, .transformerBlock 192 768,
     .transformerBlock 192 768, .transformerBlock 192 768, .transformerBlock 192 768,
     .layerNorm 192, .dense 192 10] => vitForwardTiny w
  | _ => fun _ => 0

/-- **Spec ≡ the full proven net.** `vitVerified`'s denotation is exactly
    `vitForwardKV` at the committed config (depth-12 DISTINCT-param multi-head,
    per-token vector-LN — `ViTDepthK.lean`) — by `rfl`. Retires the rep tie's
    weight-shared scalar-LN caveats at the spec level. -/
theorem vitVerified_denote_eq (w : ViTTinyWeights) :
    denoteVitTiny vitVerified.layers w = vitForwardTiny w := rfl

/-- **The committed spec carries the math — the REAL whole-net VJP.** ViT is all-smooth
    (GELU/softmax/LN), so unlike the conv nets the honest chain-rule fold applies
    globally: `vitForwardKV_has_vjp` at the committed config, hypothesis `0 < ε` only.
    The strongest rung C in this file — no canonical-witness fallback needed. -/
noncomputable def vitVerified_has_vjp (w : ViTTinyWeights) (hε : 0 < w.ε) :
    HasVJP (denoteVitTiny vitVerified.layers w) :=
  vitForwardKV_has_vjp 3 224 224 16 196 768 3 64 10 12
    w.Wc w.bc w.cls w.pos w.ε hε w.blocks w.γF w.βF w.Wcls w.bcls

open Proofs.StableHLO in
/-- **Rung E at the committed spec.** The depth-12 3-head vector-LN forward graph
    (`vitFwdGraphKMHV`, ViTDepthK §3 — patch embed → 12 spelled multi-head blocks →
    final vector-LN → CLS → head) denotes the committed spec's function:
    `vitFwdGraphKMHV_faithful` composed with the tie. Completes the B/C/E ladder for
    all five imagenette nets. -/
theorem vitVerified_fwd_faithful (epsStr sStr oneStr zeroStr : String)
    (w : ViTTinyWeights) (x : Vec (3 * 224 * 224)) :
    den (vitFwdGraphKMHV (ic := 3) (H := 224) (W := 224) (P := 16) (N := 196)
          (hm1 := 2) (d := 64) (mlpDim := 768) (nClasses := 10)
          epsStr sStr oneStr zeroStr w.ε (sdpa_scale 64)
          w.Wc w.bc w.cls w.pos 12 w.blocks w.γF w.βF w.Wcls w.bcls x)
      = denoteVitTiny vitVerified.layers w x :=
  (vitFwdGraphKMHV_faithful 3 224 224 16 196 2 64 768 10 epsStr sStr oneStr zeroStr
      w.Wc w.bc w.cls w.pos w.ε 12 w.blocks w.γF w.βF w.Wcls w.bcls x).trans
    (congrFun (vitVerified_denote_eq w).symm x)
