import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Foundation.MLP
import LeanMlir.Proofs.Architectures.MnistCNN
import LeanMlir.Proofs.Architectures.CifarCNN
import LeanMlir.Proofs.Architectures.MobileNetV2
import LeanMlir.Proofs.Architectures.MobileNetV2FullPaper
import LeanMlir.Proofs.Architectures.EfficientNet
import LeanMlir.Proofs.Architectures.EfficientNetFullB0
import LeanMlir.Proofs.Architectures.ConvNeXt
import LeanMlir.Proofs.Architectures.ConvNeXtFullT
import LeanMlir.Proofs.Architectures.Attention
import LeanMlir.Proofs.Architectures.ViTDepthK
import LeanMlir.Proofs.Foundation.ResNet34
import LeanMlir.Proofs.Codegen.ResNet34RenderPC
import LeanMlir.Proofs.Codegen.StableHLO

/-! # Spec → math (the verification tie), Rung 1: the linear classifier

The shape `#guard` in `MainResnet34Verified` only checks the *parameter interface*
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
    HasVJP (denoteCNN cnnVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅) where
  backward x dy i :=
    ∑ j : Fin 10, pdiv (denoteCNN cnnVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅) x i j * dy j
  correct _ _ _ := rfl

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

The MLP has faithfulness for the whole forward graph (`mlpFwdGraph_faithful`) AND the whole
backward input-VJP graph (`mlpBackGraph_faithful`). Composed with `denoteMLP = mlpForward`
and `mlpVerified_has_vjp_at = mlp_has_vjp_at`, both halves of the generated train step are
tied to the spec: the rendered forward computes the spec's forward, and the rendered
backward computes the spec's VJP backward (at a smooth input). -/

open Proofs.StableHLO in
/-- **Generated MLP forward MLIR ↔ spec.** The forward graph (→ `mlp_fwd.mlir`) denotes
    the spec's forward function. -/
theorem mlpVerified_fwd_faithful (W₀ : Mat 784 512) (b₀ : Vec 512)
    (W₁ : Mat 512 512) (b₁ : Vec 512) (W₂ : Mat 512 10) (b₂ : Vec 10) (x : Vec 784) :
    den (mlpFwdGraph W₀ b₀ W₁ b₁ W₂ b₂ x)
      = denoteMLP mlpVerified.layers W₀ b₀ W₁ b₁ W₂ b₂ x := by
  exact mlpFwdGraph_faithful W₀ b₀ W₁ b₁ W₂ b₂ x

open Proofs.StableHLO in
/-- **Generated MLP backward MLIR ↔ spec.** The backward input-VJP graph (in
    `mlp_train_step.mlir`) denotes the spec's VJP backward (`mlpVerified_has_vjp_at`), at a
    smooth input (the two ReLU pre-activations avoid zero). -/
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

/-! ## Rung 4 + E (CIFAR, both variants): completing the ch5 ladder

The two CIFAR-10 nets (ic=3, c1=32, c2=64, h=w=8 — spatial 32→16→8). Each gets the
spec→math denotation (= `cifarCnnForward` / `cifarCnnBnForward` by `rfl`), the canonical
witness VJP, and the forward spec→generated-MLIR tie (`cifarFwdGraph_faithful` /
`cifarBnFwdGraph_faithful`). The conditional folds are `cifarCnn_has_vjp_at` /
`cifarCnnBn_has_vjp_at` (six ReLU kinks + two maxpools; BN adds `0 < εᵢ`). The BN here is
the SCALAR `bnForward` (one γ/β over c·h·w), the same op ViT's LayerNorm witness reduces to. -/

-- ── CIFAR (no BN) ──
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
    HasVJP (denoteCifar cifarVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇) where
  backward x dy i :=
    ∑ j : Fin 10,
      pdiv (denoteCifar cifarVerified.layers W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇) x i j * dy j
  correct _ _ _ := rfl

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

-- ── CIFAR + scalar BatchNorm ──
noncomputable def denoteCifarBn (layers : List VLayer)
    (W₁ : Kernel4 32 3 3 3) (b₁ : Vec 32) (ε₁ : ℝ) (γ₁ β₁ : Vec 32)
    (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32) (ε₂ : ℝ) (γ₂ β₂ : Vec 32)
    (W₃ : Kernel4 64 32 3 3) (b₃ : Vec 64) (ε₃ : ℝ) (γ₃ β₃ : Vec 64)
    (W₄ : Kernel4 64 64 3 3) (b₄ : Vec 64) (ε₄ : ℝ) (γ₄ β₄ : Vec 64)
    (W₅ : Mat 4096 512) (b₅ : Vec 512) (W₆ : Mat 512 512) (b₆ : Vec 512)
    (W₇ : Mat 512 10) (b₇ : Vec 10) : Vec 3072 → Vec 10 :=
  match layers with
  | [.conv 3 32 3 1, .bnPerChannel 32, .relu, .conv 32 32 3 1, .bnPerChannel 32, .relu, .maxPool 2 2,
     .conv 32 64 3 1, .bnPerChannel 64, .relu, .conv 64 64 3 1, .bnPerChannel 64, .relu, .maxPool 2 2, .flatten,
     .dense 4096 512, .relu, .dense 512 512, .relu, .dense 512 10] =>
      cifarCnnBnForward (h := 8) (w := 8) W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
        W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇
  | _ => fun _ => 0

theorem cifarBnVerified_denote_eq
    (W₁ : Kernel4 32 3 3 3) (b₁ : Vec 32) (ε₁ : ℝ) (γ₁ β₁ : Vec 32)
    (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32) (ε₂ : ℝ) (γ₂ β₂ : Vec 32)
    (W₃ : Kernel4 64 32 3 3) (b₃ : Vec 64) (ε₃ : ℝ) (γ₃ β₃ : Vec 64)
    (W₄ : Kernel4 64 64 3 3) (b₄ : Vec 64) (ε₄ : ℝ) (γ₄ β₄ : Vec 64)
    (W₅ : Mat 4096 512) (b₅ : Vec 512) (W₆ : Mat 512 512) (b₆ : Vec 512)
    (W₇ : Mat 512 10) (b₇ : Vec 10) :
    denoteCifarBn cifarBnVerified.layers W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
        W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇
      = cifarCnnBnForward (h := 8) (w := 8) W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
          W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇ := rfl

/-- **The (per-channel-BN) CIFAR spec carries the math.** -/
noncomputable def cifarBnVerified_has_vjp
    (W₁ : Kernel4 32 3 3 3) (b₁ : Vec 32) (ε₁ : ℝ) (γ₁ β₁ : Vec 32)
    (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32) (ε₂ : ℝ) (γ₂ β₂ : Vec 32)
    (W₃ : Kernel4 64 32 3 3) (b₃ : Vec 64) (ε₃ : ℝ) (γ₃ β₃ : Vec 64)
    (W₄ : Kernel4 64 64 3 3) (b₄ : Vec 64) (ε₄ : ℝ) (γ₄ β₄ : Vec 64)
    (W₅ : Mat 4096 512) (b₅ : Vec 512) (W₆ : Mat 512 512) (b₆ : Vec 512)
    (W₇ : Mat 512 10) (b₇ : Vec 10) :
    HasVJP (denoteCifarBn cifarBnVerified.layers W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
              W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇) where
  backward x dy i :=
    ∑ j : Fin 10, pdiv (denoteCifarBn cifarBnVerified.layers W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
              W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇) x i j * dy j
  correct _ _ _ := rfl

open Proofs.StableHLO in
/-- **Generated (per-channel-BN) CIFAR forward MLIR ↔ spec.** (`epsStr` = the rendered ε text;
    the denotation uses the real `εᵢ`, so it holds for any string.) -/
theorem cifarBnVerified_fwd_faithful (epsStr : String)
    (W₁ : Kernel4 32 3 3 3) (b₁ : Vec 32) (ε₁ : ℝ) (γ₁ β₁ : Vec 32)
    (W₂ : Kernel4 32 32 3 3) (b₂ : Vec 32) (ε₂ : ℝ) (γ₂ β₂ : Vec 32)
    (W₃ : Kernel4 64 32 3 3) (b₃ : Vec 64) (ε₃ : ℝ) (γ₃ β₃ : Vec 64)
    (W₄ : Kernel4 64 64 3 3) (b₄ : Vec 64) (ε₄ : ℝ) (γ₄ β₄ : Vec 64)
    (W₅ : Mat 4096 512) (b₅ : Vec 512) (W₆ : Mat 512 512) (b₆ : Vec 512)
    (W₇ : Mat 512 10) (b₇ : Vec 10) (x : Vec 3072) :
    den (cifarBnFwdGraph (h := 8) (w := 8) epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
          W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇ x)
      = denoteCifarBn cifarBnVerified.layers W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
          W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇ x := by
  exact cifarBnFwdGraph_faithful (h := 8) (w := 8) epsStr W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
          W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇ x

/-! ## Rung B/C (ch7 MobileNetV2, representative): the strided 6-block witness

Representative tie, like the other imagenette nets below: `denoteMobilenet` maps
`mobilenetv2RepLayers` — the 10-entry strided 6-block layer list (stem-s2 → 6
inverted-residual blocks `[16→64→24, 24→96→24, 24→96→32, 32→128→32, 32→128→64,
64→256→64]` with 4 stride-2 depthwise downsamples 224→7 and 2 stride-1 skips → 1×1
conv-bn-relu6 head → GAP → dense) — to `mobilenetv2Forward_full`, the faithful 6-block
composition built in `Proofs/MobileNetV2.lean` from the strided inverted-residual VJP
infrastructure (`invresBodyStrided`, `flatConvStride2`, `depthwiseStride2Flat`). The
`rfl` tie is drift-sensitive: change any block's `[t,c,n,s]` and the match stops reducing.

**History note**: this rung used to tie `mobilenetv2Verified.layers` itself — true while
the committed spec WAS the 6-block net. The spec was promoted to the full-paper 17-block
net (e9cd890), so this rung is now representative; the committed spec's full tie is the
next section (`denoteMobilenetPaper` → `mobilenetv2ForwardPaper`).

The honest chain-rule fold is carried by the new strided inverted-residual block witness
`Proofs.invresBodyStrided_has_vjp_at` (expand-SAME → stride-2 depthwise → project-SAME,
the downsampling block the render uses) composed with the representative inverted-residual
fold `Proofs.mobilenetv2_has_vjp_at`; here the rung-C headline is the unconditional
canonical witness, matching the ch4/ch5 conv nets (`cnnVerified_has_vjp` /
`cifarVerified_has_vjp`).

**Stated gap** (intrinsic, shared with ch5-BN / every BN net here): the proof's `bnForward`
is SCALAR-global (one γ/β over the whole `c·h·w` map per example); the render uses
per-channel `[c]` BN. Topology, channel flow, stride schedule, relu6 sites and residual
placement are all faithful — only BN granularity differs. -/

/-- The representative MobileNetV2 layer list: the strided 6-block net that
    `mobilenetv2Forward_full` actually renders and proves. A prefix-shaped slice of the
    committed full-paper `mobilenetv2Verified` spec (17 blocks, 210 tensors), whose full
    tie is the next section (`denoteMobilenetPaper`). -/
def mobilenetv2RepLayers : List VLayer :=
  [.convBn 3 16 3 2,
   .invertedResidual 16 64 24 2, .invertedResidual 24 96 24 1,
   .invertedResidual 24 96 32 2, .invertedResidual 32 128 32 1,
   .invertedResidual 32 128 64 2, .invertedResidual 64 256 64 2,
   .convBn 64 128 1 1, .globalAvgPool, .dense 128 10]

/-- Math denotation of the representative MobileNetV2 spec: the 10-entry strided 6-block
    layer list denotes to `mobilenetv2Forward_full`. Any other list is not the net (`0`),
    making the tie below drift-sensitive. -/
noncomputable def denoteMobilenet (layers : List VLayer)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs γs βs : ℝ)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 γe1 βe1 : ℝ)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 γd1 βd1 : ℝ)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 γp1 βp1 : ℝ)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 γe2 βe2 : ℝ)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 γd2 βd2 : ℝ)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 γp2 βp2 : ℝ)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 γe3 βe3 : ℝ)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 γd3 βd3 : ℝ)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 γp3 βp3 : ℝ)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 γe4 βe4 : ℝ)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 γd4 βd4 : ℝ)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 γp4 βp4 : ℝ)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 γe5 βe5 : ℝ)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 γd5 βd5 : ℝ)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 γp5 βp5 : ℝ)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 γe6 βe6 : ℝ)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 γd6 βd6 : ℝ)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 γp6 βp6 : ℝ)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh γh βh : ℝ)
    (Wfc : Mat 128 10) (bfc : Vec 10) :
    Vec (3 * 224 * 224) → Vec 10 :=
  match layers with
  | [.convBn 3 16 3 2,
     .invertedResidual 16 64 24 2, .invertedResidual 24 96 24 1,
     .invertedResidual 24 96 32 2, .invertedResidual 32 128 32 1,
     .invertedResidual 32 128 64 2, .invertedResidual 64 256 64 2,
     .convBn 64 128 1 1, .globalAvgPool, .dense 128 10] =>
      mobilenetv2Forward_full Ws bs εs γs βs
        We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1
        We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2
        We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3
        We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4
        We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5
        We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6
        Wh bh εh γh βh Wfc bfc
  | _ => fun _ => 0

/-- **Spec ≡ the representative proven render.** `mobilenetv2RepLayers`'s denotation is
    exactly `mobilenetv2Forward_full` (the strided 6-block net) — by `rfl`, drift-sensitive. -/
theorem mobilenetv2Rep_denote_eq
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs γs βs : ℝ)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 γe1 βe1 : ℝ)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 γd1 βd1 : ℝ)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 γp1 βp1 : ℝ)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 γe2 βe2 : ℝ)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 γd2 βd2 : ℝ)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 γp2 βp2 : ℝ)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 γe3 βe3 : ℝ)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 γd3 βd3 : ℝ)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 γp3 βp3 : ℝ)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 γe4 βe4 : ℝ)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 γd4 βd4 : ℝ)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 γp4 βp4 : ℝ)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 γe5 βe5 : ℝ)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 γd5 βd5 : ℝ)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 γp5 βp5 : ℝ)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 γe6 βe6 : ℝ)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 γd6 βd6 : ℝ)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 γp6 βp6 : ℝ)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh γh βh : ℝ)
    (Wfc : Mat 128 10) (bfc : Vec 10) :
    denoteMobilenet mobilenetv2RepLayers Ws bs εs γs βs
        We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1
        We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2
        We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3
        We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4
        We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5
        We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6
        Wh bh εh γh βh Wfc bfc
      = mobilenetv2Forward_full Ws bs εs γs βs
        We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1
        We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2
        We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3
        We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4
        We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5
        We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6
        Wh bh εh γh βh Wfc bfc := rfl

/-- **The representative spec carries the math.** The strided 6-block MobileNetV2 spec's
    denotation has a VJP — the canonical `pdiv`-derived witness (the honest strided
    chain-rule fold is `Proofs.mobilenetv2_full_has_vjp_at`). -/
noncomputable def mobilenetv2Rep_has_vjp
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs γs βs : ℝ)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 γe1 βe1 : ℝ)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 γd1 βd1 : ℝ)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 γp1 βp1 : ℝ)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 γe2 βe2 : ℝ)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 γd2 βd2 : ℝ)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 γp2 βp2 : ℝ)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 γe3 βe3 : ℝ)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 γd3 βd3 : ℝ)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 γp3 βp3 : ℝ)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 γe4 βe4 : ℝ)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 γd4 βd4 : ℝ)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 γp4 βp4 : ℝ)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 γe5 βe5 : ℝ)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 γd5 βd5 : ℝ)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 γp5 βp5 : ℝ)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 γe6 βe6 : ℝ)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 γd6 βd6 : ℝ)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 γp6 βp6 : ℝ)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh γh βh : ℝ)
    (Wfc : Mat 128 10) (bfc : Vec 10) :
    HasVJP (denoteMobilenet mobilenetv2RepLayers Ws bs εs γs βs
        We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1
        We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2
        We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3
        We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4
        We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5
        We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6
        Wh bh εh γh βh Wfc bfc) where
  backward x dy i :=
    ∑ j : Fin 10, pdiv (denoteMobilenet mobilenetv2RepLayers Ws bs εs γs βs
        We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1
        We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2
        We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3
        We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4
        We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5
        We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6
        Wh bh εh γh βh Wfc bfc) x i j * dy j
  correct _ _ _ := rfl


/-! ## Rung B/C/E (ch7 MobileNetV2, FULL): the committed spec ↔ the paper-spec net

The real thing: `denoteMobilenetPaper` maps `mobilenetv2Verified.layers` — the committed
21-entry full-paper `[t,c,n,s]` list the trainer runs (stem-s2 3→32 → 17 bottlenecks →
1×1 head 320→1280 → GAP → dense 1280→10) — to `mobilenetv2ForwardPaper`
(`MobileNetV2FullPaper.lean`: per-channel BN throughout, the t=1 no-expand first block,
4 stride-2 depthwise downsamples 224→7). Weights ride in the `MNV2PaperWeights` bundle,
so the tie stays readable. The `rfl` is drift-sensitive: any `[t,c,n,s]` edit to the spec
stops the match reducing — exactly the tripwire the 6→17-block promotion fired while this
file was orphaned; certs.yml now re-elaborates it on every spec push.

This restores (and upgrades) the full mnv2 B/C lost in the promotion: the old full tie was
the scalar-BN 6-block net; this one is the committed per-channel-BN 17-block net, with
rung E on top (`mobilenetv2FwdGraphPaper_faithful` composed with the tie). -/

/-- Math denotation of the committed MobileNetV2 spec: the 21-entry full-paper layer list
    denotes to `mobilenetv2ForwardPaper`. Any other list is not the net (`0`), making the
    tie below drift-sensitive. -/
noncomputable def denoteMobilenetPaper (layers : List VLayer) (w : MNV2PaperWeights) :
    Vec (3 * 224 * 224) → Vec 10 :=
  match layers with
  | [.convBn 3 32 3 2,
     .invertedResidual 32 32 16 1,
     .invertedResidual 16 96 24 2, .invertedResidual 24 144 24 1,
     .invertedResidual 24 144 32 2, .invertedResidual 32 192 32 1, .invertedResidual 32 192 32 1,
     .invertedResidual 32 192 64 2, .invertedResidual 64 384 64 1, .invertedResidual 64 384 64 1,
     .invertedResidual 64 384 64 1,
     .invertedResidual 64 384 96 1, .invertedResidual 96 576 96 1, .invertedResidual 96 576 96 1,
     .invertedResidual 96 576 160 2, .invertedResidual 160 960 160 1, .invertedResidual 160 960 160 1,
     .invertedResidual 160 960 320 1,
     .convBn 320 1280 1 1, .globalAvgPool, .dense 1280 10] =>
      mobilenetv2ForwardPaper w
  | _ => fun _ => 0

/-- **Spec ≡ the full paper-spec net.** The committed `mobilenetv2Verified`'s denotation
    is exactly `mobilenetv2ForwardPaper` (all 17 bottlenecks, per-channel BN) — by `rfl`,
    drift-sensitive. -/
theorem mobilenetv2Verified_denote_eq (w : MNV2PaperWeights) :
    denoteMobilenetPaper mobilenetv2Verified.layers w = mobilenetv2ForwardPaper w := rfl

/-- **The committed spec carries the math.** The full-paper spec's denotation has a VJP —
    the canonical `pdiv`-derived witness (relu6 is kinked, so the honest whole-net
    input-VJP stays pointwise-only, the repo standard for relu-family nets; the
    dim-polymorphic `MobileNetV2Close`/`ChainClose` param-grad bridges apply at the paper
    shapes verbatim, per `MobileNetV2FullPaper.lean`'s header). -/
noncomputable def mobilenetv2Verified_has_vjp (w : MNV2PaperWeights) :
    HasVJP (denoteMobilenetPaper mobilenetv2Verified.layers w) where
  backward x dy i :=
    ∑ j : Fin 10, pdiv (denoteMobilenetPaper mobilenetv2Verified.layers w) x i j * dy j
  correct _ _ _ := rfl

open Proofs.StableHLO in
/-- **Rung E at the committed spec.** The generated full-paper StableHLO graph denotes the
    committed spec's function: `mobilenetv2FwdGraphPaper_faithful` composed with the tie. -/
theorem mobilenetv2Verified_fwd_faithful (epsStr : String) (w : MNV2PaperWeights)
    (x : Vec (3 * 224 * 224)) :
    den (mobilenetv2FwdGraphPaper epsStr w x)
      = denoteMobilenetPaper mobilenetv2Verified.layers w x :=
  (mobilenetv2FwdGraphPaper_faithful epsStr w x).trans
    (congrFun (mobilenetv2Verified_denote_eq w).symm x)


/-! ## Rung B/C/E (FULL, unified weight bundles): r34 / enet / convnext / vit

The mnv2 full-paper pattern applied to the remaining imagenette nets: each committed
spec's ENTIRE layer list (literal dims, drift-sensitive) denotes the full proven
forward, with weights riding a structure bundle so the ties stay readable. Existing
bundles are reused where the Full module already has one (`B0Weights`,
`CnxTWeights`); r34 and vit get bundles here (`R34Weights`, `ViTTinyWeights` —
SpecVJP-local so no proof module's signature changes). Rung E composes each net's
full graph-faithfulness apex with the tie (vit's is `vitFwdGraphKMHV_faithful`,
ViTDepthK §3 — the depth-`k` multi-head vector-LN graph). Rung C is the canonical
witness except vit: all-smooth, so vit's rung C is the REAL whole-net VJP
`vitForwardKV_has_vjp` (only `0 < ε`) at the committed spec. -/

-- ── ResNet-34 (FULL): the committed 8-entry spec ↔ resnet34Forward_full_pc ──

/-- Identity basic-block weights (conv-BN ×2), per-channel γ/β. -/
structure R34BlockW (c : Nat) where
  W1 : Kernel4 c c 3 3
  b1 : Vec c
  g1 : Vec c
  t1 : Vec c
  W2 : Kernel4 c c 3 3
  b2 : Vec c
  g2 : Vec c
  t2 : Vec c

/-- Downsample basic-block weights (strided conv-BN ×2 + projection conv-BN). -/
structure R34DownW (ic oc : Nat) where
  W1 : Kernel4 oc ic 3 3
  b1 : Vec oc
  g1 : Vec oc
  t1 : Vec oc
  W2 : Kernel4 oc oc 3 3
  b2 : Vec oc
  g2 : Vec oc
  t2 : Vec oc
  Wp : Kernel4 oc ic 3 3
  bp : Vec oc
  gp : Vec oc
  tp : Vec oc

/-- All ResNet-34 parameters (shared BN ε): stem + [3,4,6,3] basic blocks + dense. -/
structure R34Weights where
  ε : ℝ
  sW : Kernel4 64 3 7 7
  sb : Vec 64
  sγ : Vec 64
  sβ : Vec 64
  a0 : R34BlockW 64
  a1 : R34BlockW 64
  a2 : R34BlockW 64
  d2 : R34DownW 64 128
  b0 : R34BlockW 128
  b1 : R34BlockW 128
  b2 : R34BlockW 128
  d3 : R34DownW 128 256
  c0 : R34BlockW 256
  c1 : R34BlockW 256
  c2 : R34BlockW 256
  c3 : R34BlockW 256
  c4 : R34BlockW 256
  d4 : R34DownW 256 512
  e0 : R34BlockW 512
  e1 : R34BlockW 512
  Wd : Mat 512 10
  bd : Vec 10

/-- `resnet34Forward_full_pc` at the bundle (the 145-arg field expansion, once). -/
noncomputable def resnet34ForwardW (w : R34Weights) : Vec (3 * 224 * 224) → Vec 10 :=
  resnet34Forward_full_pc w.ε w.sW w.sb w.sγ w.sβ
    w.a0.W1 w.a0.b1 w.a0.g1 w.a0.t1 w.a0.W2 w.a0.b2 w.a0.g2 w.a0.t2
    w.a1.W1 w.a1.b1 w.a1.g1 w.a1.t1 w.a1.W2 w.a1.b2 w.a1.g2 w.a1.t2
    w.a2.W1 w.a2.b1 w.a2.g1 w.a2.t1 w.a2.W2 w.a2.b2 w.a2.g2 w.a2.t2
    w.d2.W1 w.d2.b1 w.d2.g1 w.d2.t1 w.d2.W2 w.d2.b2 w.d2.g2 w.d2.t2 w.d2.Wp w.d2.bp w.d2.gp w.d2.tp
    w.b0.W1 w.b0.b1 w.b0.g1 w.b0.t1 w.b0.W2 w.b0.b2 w.b0.g2 w.b0.t2
    w.b1.W1 w.b1.b1 w.b1.g1 w.b1.t1 w.b1.W2 w.b1.b2 w.b1.g2 w.b1.t2
    w.b2.W1 w.b2.b1 w.b2.g1 w.b2.t1 w.b2.W2 w.b2.b2 w.b2.g2 w.b2.t2
    w.d3.W1 w.d3.b1 w.d3.g1 w.d3.t1 w.d3.W2 w.d3.b2 w.d3.g2 w.d3.t2 w.d3.Wp w.d3.bp w.d3.gp w.d3.tp
    w.c0.W1 w.c0.b1 w.c0.g1 w.c0.t1 w.c0.W2 w.c0.b2 w.c0.g2 w.c0.t2
    w.c1.W1 w.c1.b1 w.c1.g1 w.c1.t1 w.c1.W2 w.c1.b2 w.c1.g2 w.c1.t2
    w.c2.W1 w.c2.b1 w.c2.g1 w.c2.t1 w.c2.W2 w.c2.b2 w.c2.g2 w.c2.t2
    w.c3.W1 w.c3.b1 w.c3.g1 w.c3.t1 w.c3.W2 w.c3.b2 w.c3.g2 w.c3.t2
    w.c4.W1 w.c4.b1 w.c4.g1 w.c4.t1 w.c4.W2 w.c4.b2 w.c4.g2 w.c4.t2
    w.d4.W1 w.d4.b1 w.d4.g1 w.d4.t1 w.d4.W2 w.d4.b2 w.d4.g2 w.d4.t2 w.d4.Wp w.d4.bp w.d4.gp w.d4.tp
    w.e0.W1 w.e0.b1 w.e0.g1 w.e0.t1 w.e0.W2 w.e0.b2 w.e0.g2 w.e0.t2
    w.e1.W1 w.e1.b1 w.e1.g1 w.e1.t1 w.e1.W2 w.e1.b2 w.e1.g2 w.e1.t2
    w.Wd w.bd

/-- Math denotation of the committed ResNet-34 spec: the 8-entry stage-level layer list
    denotes to the full per-channel [3,4,6,3] render. Any other list is not the net (`0`). -/
noncomputable def denoteR34Full (layers : List VLayer) (w : R34Weights) :
    Vec (3 * 224 * 224) → Vec 10 :=
  match layers with
  | [.convBn 3 64 7 2, .maxPool 2 2,
     .residualStage 64 64 3 1, .residualStage 64 128 4 2,
     .residualStage 128 256 6 2, .residualStage 256 512 3 2,
     .globalAvgPool, .dense 512 10] => resnet34ForwardW w
  | _ => fun _ => 0

/-- **Spec ≡ the full proven render.** `resnet34Verified`'s denotation is exactly
    `resnet34Forward_full_pc` (per-channel BN, [3,4,6,3] at 224²) — by `rfl`. -/
theorem resnet34Verified_denote_eq (w : R34Weights) :
    denoteR34Full resnet34Verified.layers w = resnet34ForwardW w := rfl

/-- **The committed spec carries the math** — canonical `pdiv` witness (relu is kinked,
    so the honest whole-net input-VJP stays pointwise; the live/seal theorems
    (`ResNet34Live*`) discharge nontriviality at full depth and realistic dims). -/
noncomputable def resnet34Verified_has_vjp (w : R34Weights) :
    HasVJP (denoteR34Full resnet34Verified.layers w) where
  backward x dy i :=
    ∑ j : Fin 10, pdiv (denoteR34Full resnet34Verified.layers w) x i j * dy j
  correct _ _ _ := rfl

open Proofs.StableHLO in
/-- **Rung E at the committed spec.** The full per-channel [3,4,6,3] graph denotes the
    committed spec's function: `resnet34FwdGraphFullPC_faithful` composed with the tie. -/
theorem resnet34Verified_fwd_faithful (epsStr : String) (w : R34Weights)
    (x : Vec (3 * 224 * 224)) :
    den (resnet34FwdGraphFullPC epsStr w.ε w.sW w.sb w.sγ w.sβ
      w.a0.W1 w.a0.b1 w.a0.g1 w.a0.t1 w.a0.W2 w.a0.b2 w.a0.g2 w.a0.t2
      w.a1.W1 w.a1.b1 w.a1.g1 w.a1.t1 w.a1.W2 w.a1.b2 w.a1.g2 w.a1.t2
      w.a2.W1 w.a2.b1 w.a2.g1 w.a2.t1 w.a2.W2 w.a2.b2 w.a2.g2 w.a2.t2
      w.d2.W1 w.d2.b1 w.d2.g1 w.d2.t1 w.d2.W2 w.d2.b2 w.d2.g2 w.d2.t2 w.d2.Wp w.d2.bp w.d2.gp w.d2.tp
      w.b0.W1 w.b0.b1 w.b0.g1 w.b0.t1 w.b0.W2 w.b0.b2 w.b0.g2 w.b0.t2
      w.b1.W1 w.b1.b1 w.b1.g1 w.b1.t1 w.b1.W2 w.b1.b2 w.b1.g2 w.b1.t2
      w.b2.W1 w.b2.b1 w.b2.g1 w.b2.t1 w.b2.W2 w.b2.b2 w.b2.g2 w.b2.t2
      w.d3.W1 w.d3.b1 w.d3.g1 w.d3.t1 w.d3.W2 w.d3.b2 w.d3.g2 w.d3.t2 w.d3.Wp w.d3.bp w.d3.gp w.d3.tp
      w.c0.W1 w.c0.b1 w.c0.g1 w.c0.t1 w.c0.W2 w.c0.b2 w.c0.g2 w.c0.t2
      w.c1.W1 w.c1.b1 w.c1.g1 w.c1.t1 w.c1.W2 w.c1.b2 w.c1.g2 w.c1.t2
      w.c2.W1 w.c2.b1 w.c2.g1 w.c2.t1 w.c2.W2 w.c2.b2 w.c2.g2 w.c2.t2
      w.c3.W1 w.c3.b1 w.c3.g1 w.c3.t1 w.c3.W2 w.c3.b2 w.c3.g2 w.c3.t2
      w.c4.W1 w.c4.b1 w.c4.g1 w.c4.t1 w.c4.W2 w.c4.b2 w.c4.g2 w.c4.t2
      w.d4.W1 w.d4.b1 w.d4.g1 w.d4.t1 w.d4.W2 w.d4.b2 w.d4.g2 w.d4.t2 w.d4.Wp w.d4.bp w.d4.gp w.d4.tp
      w.e0.W1 w.e0.b1 w.e0.g1 w.e0.t1 w.e0.W2 w.e0.b2 w.e0.g2 w.e0.t2
      w.e1.W1 w.e1.b1 w.e1.g1 w.e1.t1 w.e1.W2 w.e1.b2 w.e1.g2 w.e1.t2
      w.Wd w.bd x)
      = denoteR34Full resnet34Verified.layers w x :=
  (resnet34FwdGraphFullPC_faithful epsStr w.ε w.sW w.sb w.sγ w.sβ
      w.a0.W1 w.a0.b1 w.a0.g1 w.a0.t1 w.a0.W2 w.a0.b2 w.a0.g2 w.a0.t2
      w.a1.W1 w.a1.b1 w.a1.g1 w.a1.t1 w.a1.W2 w.a1.b2 w.a1.g2 w.a1.t2
      w.a2.W1 w.a2.b1 w.a2.g1 w.a2.t1 w.a2.W2 w.a2.b2 w.a2.g2 w.a2.t2
      w.d2.W1 w.d2.b1 w.d2.g1 w.d2.t1 w.d2.W2 w.d2.b2 w.d2.g2 w.d2.t2 w.d2.Wp w.d2.bp w.d2.gp w.d2.tp
      w.b0.W1 w.b0.b1 w.b0.g1 w.b0.t1 w.b0.W2 w.b0.b2 w.b0.g2 w.b0.t2
      w.b1.W1 w.b1.b1 w.b1.g1 w.b1.t1 w.b1.W2 w.b1.b2 w.b1.g2 w.b1.t2
      w.b2.W1 w.b2.b1 w.b2.g1 w.b2.t1 w.b2.W2 w.b2.b2 w.b2.g2 w.b2.t2
      w.d3.W1 w.d3.b1 w.d3.g1 w.d3.t1 w.d3.W2 w.d3.b2 w.d3.g2 w.d3.t2 w.d3.Wp w.d3.bp w.d3.gp w.d3.tp
      w.c0.W1 w.c0.b1 w.c0.g1 w.c0.t1 w.c0.W2 w.c0.b2 w.c0.g2 w.c0.t2
      w.c1.W1 w.c1.b1 w.c1.g1 w.c1.t1 w.c1.W2 w.c1.b2 w.c1.g2 w.c1.t2
      w.c2.W1 w.c2.b1 w.c2.g1 w.c2.t1 w.c2.W2 w.c2.b2 w.c2.g2 w.c2.t2
      w.c3.W1 w.c3.b1 w.c3.g1 w.c3.t1 w.c3.W2 w.c3.b2 w.c3.g2 w.c3.t2
      w.c4.W1 w.c4.b1 w.c4.g1 w.c4.t1 w.c4.W2 w.c4.b2 w.c4.g2 w.c4.t2
      w.d4.W1 w.d4.b1 w.d4.g1 w.d4.t1 w.d4.W2 w.d4.b2 w.d4.g2 w.d4.t2 w.d4.Wp w.d4.bp w.d4.gp w.d4.tp
      w.e0.W1 w.e0.b1 w.e0.g1 w.e0.t1 w.e0.W2 w.e0.b2 w.e0.g2 w.e0.t2
      w.e1.W1 w.e1.b1 w.e1.g1 w.e1.t1 w.e1.W2 w.e1.b2 w.e1.g2 w.e1.t2
      w.Wd w.bd x).trans
    (congrFun (resnet34Verified_denote_eq w).symm x)

-- ── EfficientNet-B0 (FULL, batched): the committed 21-entry spec ↔ efficientnetForwardB_full ──

/-- Math denotation of the committed EfficientNet-B0 spec at batch `N`: the 21-entry
    `[t,c,n,s,k]` layer list denotes to `efficientnetForwardB_full` (all 16 MBConv
    blocks, true batch-norm + SE). The spec ties the batched net at EVERY batch size. -/
noncomputable def denoteEfficientnetB0 (N : Nat) (layers : List VLayer) (w : B0Weights) :
    Vec (N * (3 * 224 * 224)) → Vec (N * 10) :=
  match layers with
  | [.convBn 3 32 3 2,
     .mbConvSE 32 32 16 8 3,
     .mbConvSE 16 96 24 4 3, .mbConvSE 24 144 24 6 3,
     .mbConvSE 24 144 40 6 5, .mbConvSE 40 240 40 10 5,
     .mbConvSE 40 240 80 10 3, .mbConvSE 80 480 80 20 3, .mbConvSE 80 480 80 20 3,
     .mbConvSE 80 480 112 20 5, .mbConvSE 112 672 112 28 5, .mbConvSE 112 672 112 28 5,
     .mbConvSE 112 672 192 28 5, .mbConvSE 192 1152 192 48 5, .mbConvSE 192 1152 192 48 5,
     .mbConvSE 192 1152 192 48 5,
     .mbConvSE 192 1152 320 48 3,
     .convBn 320 1280 1 1, .globalAvgPool, .dense 1280 10] =>
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
    HasVJP (denoteEfficientnetB0 N efficientnetVerified.layers w) where
  backward x dy i :=
    ∑ j : Fin (N * 10),
      pdiv (denoteEfficientnetB0 N efficientnetVerified.layers w) x i j * dy j
  correct _ _ _ := rfl

open Proofs.StableHLO in
/-- **Rung E at the committed spec (batched).** The full 16-MBConv batched graph denotes
    the committed spec's function: `efficientnetFwdGraphB_full_faithful` ∘ the tie. -/
theorem efficientnetVerified_fwd_faithful (N : Nat) (epsStr : String) (w : B0Weights)
    (x : Vec (N * (3 * 224 * 224))) :
    den (efficientnetFwdGraphB_full N epsStr w x)
      = denoteEfficientnetB0 N efficientnetVerified.layers w x :=
  (efficientnetFwdGraphB_full_faithful N epsStr w x).trans
    (congrFun (efficientnetVerified_denote_eq N w).symm x)

-- ── ConvNeXt-T (FULL): the committed 27-entry spec ↔ convNextForwardTC ──

/-- Math denotation of the committed ConvNeXt-T spec: the 27-entry `[3,3,9,3]` layer
    list denotes to `convNextForwardTC` (the committed-render config: no stem-LN,
    180 params — `ConvNeXtFullT.lean`). -/
noncomputable def denoteConvnextT (layers : List VLayer) (w : CnxTWeights) :
    Vec (3 * 224 * 224) → Vec 10 :=
  match layers with
  | [.conv 3 96 4 4,
     .convNextBlock 96, .convNextBlock 96, .convNextBlock 96,
     .bn, .conv 96 192 2 2,
     .convNextBlock 192, .convNextBlock 192, .convNextBlock 192,
     .bn, .conv 192 384 2 2,
     .convNextBlock 384, .convNextBlock 384, .convNextBlock 384,
     .convNextBlock 384, .convNextBlock 384, .convNextBlock 384,
     .convNextBlock 384, .convNextBlock 384, .convNextBlock 384,
     .bn, .conv 384 768 2 2,
     .convNextBlock 768, .convNextBlock 768, .convNextBlock 768,
     .globalAvgPool, .bn, .dense 768 10] => convNextForwardTC w
  | _ => fun _ => 0

/-- **Spec ≡ the full proven net.** `convnextVerified`'s denotation is exactly
    `convNextForwardTC` ([3,3,9,3] @ [96,192,384,768], committed 180-param config)
    — by `rfl`. -/
theorem convnextVerified_denote_eq (w : CnxTWeights) :
    denoteConvnextT convnextVerified.layers w = convNextForwardTC w := rfl

/-- **The committed spec carries the math** — canonical `pdiv` witness; the REAL
    whole-net VJP exists at full depth (`convNextForwardTC_has_vjp_correct`,
    all-smooth, LN positivities only) on the ∘-chain form. -/
noncomputable def convnextVerified_has_vjp (w : CnxTWeights) :
    HasVJP (denoteConvnextT convnextVerified.layers w) where
  backward x dy i :=
    ∑ j : Fin 10, pdiv (denoteConvnextT convnextVerified.layers w) x i j * dy j
  correct _ _ _ := rfl

open Proofs.StableHLO in
/-- **Rung E at the committed spec.** The committed-config [3,3,9,3] graph denotes the
    committed spec's function: `convNextFwdGraphTC_faithful` ∘ the tie. -/
theorem convnextVerified_fwd_faithful (epsStr : String) (w : CnxTWeights)
    (x : Vec (3 * 224 * 224)) :
    den (convNextFwdGraphTC epsStr w x)
      = denoteConvnextT convnextVerified.layers w x :=
  (convNextFwdGraphTC_faithful epsStr w x).trans
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


/-! ## Rung B/C (representative): the imagenette nets' proof witnesses

Every committed imagenette spec is now tied in FULL above (mnv2 `denoteMobilenetPaper`,
r34 `denoteR34Full`, enet `denoteEfficientnetB0`, convnext `denoteConvnextT`, vit
`denoteVitTiny`). The representative rungs below remain as the smaller readable
skeletons the ORIGINAL per-net proof witnesses actually state (`<net>Forward` + the
audited `<net>_has_vjp` apex), tied to generic-dim `VLayer` lists exactly like ch2–5:
`denote <rep layers> = <net>Forward := rfl` (rung B) + canonical `HasVJP` witness
(rung C; the honest fold is the apex). -/

-- ── EfficientNet (representative: stem-swish → MBConv·SE skip → MBConv·SE no-skip → GAP → dense) ──
/-- Math denotation of the representative EfficientNet layer list → `efficientnetForward`. -/
noncomputable def denoteEfficientnetRep {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁ kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    (layers : List VLayer)
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ)
    (Wh : Mat cout nClasses) (bh : Vec nClasses) :
    Vec (ic * h * w) → Vec nClasses :=
  match layers with
  | [.convBn _ _ _ _, .mbConvSE _ _ _ _ _, .mbConvSE _ _ _ _ _, .globalAvgPool, .dense _ _] =>
      efficientnetForward (h := h) (w := w) Ws bs εs γs βs We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ Wh bh
  | _ => fun _ => 0

/-- **Spec ≡ the representative proven model.** -/
theorem efficientnetRep_denote_eq {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁ kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ)
    (Wh : Mat cout nClasses) (bh : Vec nClasses) :
    denoteEfficientnetRep (h := h) (w := w)
      [.convBn ic c kHs 1, .mbConvSE c cmid₁ c r₁ kHd₁, .mbConvSE c cmid₂ cout r₂ kHd₂, .globalAvgPool, .dense cout nClasses]
      Ws bs εs γs βs We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ Wh bh
      = efficientnetForward (h := h) (w := w) Ws bs εs γs βs We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ Wh bh := rfl

/-- **The representative spec carries the math** (canonical witness; the honest
    unconditional fold is `Proofs.efficientnet_has_vjp`). -/
noncomputable def efficientnetRep_has_vjp {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁ kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ)
    (Wh : Mat cout nClasses) (bh : Vec nClasses) :
    HasVJP (denoteEfficientnetRep (h := h) (w := w)
      [.convBn ic c kHs 1, .mbConvSE c cmid₁ c r₁ kHd₁, .mbConvSE c cmid₂ cout r₂ kHd₂, .globalAvgPool, .dense cout nClasses]
      Ws bs εs γs βs We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ Wh bh) where
  backward x dy i :=
    ∑ j : Fin nClasses, pdiv (denoteEfficientnetRep (h := h) (w := w)
      [.convBn ic c kHs 1, .mbConvSE c cmid₁ c r₁ kHd₁, .mbConvSE c cmid₂ cout r₂ kHd₂, .globalAvgPool, .dense cout nClasses]
      Ws bs εs γs βs We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ Wh bh) x i j * dy j
  correct _ _ _ := rfl

-- ── ConvNeXt (representative: patchify → LN → block → block → GAP → head-LN → dense; scalar LN = `.bn`) ──
/-- Math denotation of the representative ConvNeXt layer list → `convNextForward`. -/
noncomputable def denoteConvnextRep {ic c cExp h w kH kW nClasses : Nat}
    (layers : List VLayer)
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ)
    (Wd : Mat c nClasses) (bd : Vec nClasses) :
    Vec (ic * h * w) → Vec nClasses :=
  match layers with
  | [.conv _ _ _ _, .bn, .convNextBlock _, .convNextBlock _, .globalAvgPool, .bn, .dense _ _] =>
      convNextForward (h := h) (w := w) Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd
  | _ => fun _ => 0

/-- **Spec ≡ the representative proven model.** -/
theorem convnextRep_denote_eq {ic c cExp h w kH kW nClasses : Nat}
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ)
    (Wd : Mat c nClasses) (bd : Vec nClasses) :
    denoteConvnextRep (h := h) (w := w)
      [.conv ic c 1 1, .bn, .convNextBlock c, .convNextBlock c, .globalAvgPool, .bn, .dense c nClasses]
      Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd
      = convNextForward (h := h) (w := w) Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd := rfl

/-- **The representative spec carries the math** (canonical witness; the honest
    unconditional fold is `Proofs.convnext_has_vjp`). -/
noncomputable def convnextRep_has_vjp {ic c cExp h w kH kW nClasses : Nat}
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ)
    (Wd : Mat c nClasses) (bd : Vec nClasses) :
    HasVJP (denoteConvnextRep (h := h) (w := w)
      [.conv ic c 1 1, .bn, .convNextBlock c, .convNextBlock c, .globalAvgPool, .bn, .dense c nClasses]
      Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd) where
  backward x dy i :=
    ∑ j : Fin nClasses, pdiv (denoteConvnextRep (h := h) (w := w)
      [.conv ic c 1 1, .bn, .convNextBlock c, .convNextBlock c, .globalAvgPool, .bn, .dense c nClasses]
      Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd) x i j * dy j
  correct _ _ _ := rfl

-- ── ViT (representative: patch-embed → CLS/pos → transformer body (kBlocks, weight-shared) → LN → dense) ──
/-- Math denotation of the representative ViT layer list → `vit_full`. The single
    `.transformerBlock` VLayer stands for the `kBlocks`-deep weight-shared `vit_body`;
    per the proof witness the LayerNorm is scalar (`layerNormForward = bnForward`), so this
    ties the spec to the scalar-LN witness, not the rendered per-channel `[D]` LN. -/
noncomputable def denoteVitRep
    (layers : List VLayer) (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Vec (ic * H * W) → Vec nClasses :=
  match layers with
  | [.conv _ _ _ _, .param _ _, .param _ _, .transformerBlock _ _, .layerNorm _, .dense _ _] =>
      vit_full ic H W patchSize N mlpDim heads d_head kBlocks nClasses W_conv b_conv cls_token pos_embed ε γ1 β1 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls
  | _ => fun _ => 0

/-- **Spec ≡ the representative proven model.** -/
theorem vitRep_denote_eq (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    denoteVitRep
      [.conv ic (heads * d_head) patchSize patchSize, .param #[1, heads * d_head] 2, .param #[N + 1, heads * d_head] 2, .transformerBlock (heads * d_head) mlpDim, .layerNorm (heads * d_head), .dense (heads * d_head) nClasses]
      ic H W patchSize N mlpDim heads d_head kBlocks nClasses W_conv b_conv cls_token pos_embed ε γ1 β1 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls
      = vit_full ic H W patchSize N mlpDim heads d_head kBlocks nClasses W_conv b_conv cls_token pos_embed ε γ1 β1 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls := rfl

/-- **The representative spec carries the math** (canonical witness; the honest
    unconditional fold is `Proofs.vit_full_has_vjp`). -/
noncomputable def vitRep_has_vjp (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize) (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head)) (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    HasVJP (denoteVitRep
      [.conv ic (heads * d_head) patchSize patchSize, .param #[1, heads * d_head] 2, .param #[N + 1, heads * d_head] 2, .transformerBlock (heads * d_head) mlpDim, .layerNorm (heads * d_head), .dense (heads * d_head) nClasses]
      ic H W patchSize N mlpDim heads d_head kBlocks nClasses W_conv b_conv cls_token pos_embed ε γ1 β1 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls) where
  backward x dy i :=
    ∑ j : Fin nClasses, pdiv (denoteVitRep
      [.conv ic (heads * d_head) patchSize patchSize, .param #[1, heads * d_head] 2, .param #[N + 1, heads * d_head] 2, .transformerBlock (heads * d_head) mlpDim, .layerNorm (heads * d_head), .dense (heads * d_head) nClasses]
      ic H W patchSize N mlpDim heads d_head kBlocks nClasses W_conv b_conv cls_token pos_embed ε γ1 β1 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls) x i j * dy j
  correct _ _ _ := rfl

-- ── ResNet-34 (representative: the audited parametric skeleton `resnet34_has_vjp_at`) ──
/-- Math denotation of the representative ResNet-34 layer list → the skeleton composition
    `dense ∘ gap ∘ chainComp ids4 ∘ down4 ∘ … ∘ chainComp ids1 ∘ mp ∘ stem` that the audited
    parametric apex `resnet34_has_vjp_at` is about. r34 has no concrete whole-net `Forward`
    (only this abstract [3,4,6,3]-stage skeleton over abstract block maps); the full faithful
    forward at real Imagenette dims is the deferred build. -/
noncomputable def denoteR34Rep {s0 s1 s2 s3 s4 s5 s6 s7 : Nat}
    (layers : List VLayer)
    (stem : Vec s0 → Vec s1) (mp : Vec s1 → Vec s2)
    (ids1 : List (Vec s2 → Vec s2))
    (down2 : Vec s2 → Vec s3) (ids2 : List (Vec s3 → Vec s3))
    (down3 : Vec s3 → Vec s4) (ids3 : List (Vec s4 → Vec s4))
    (down4 : Vec s4 → Vec s5) (ids4 : List (Vec s5 → Vec s5))
    (gap : Vec s5 → Vec s6) (dense : Vec s6 → Vec s7) :
    Vec s0 → Vec s7 :=
  match layers with
  | [.convBn _ _ _ _, .maxPool _ _, .residualStage _ _ _ _, .residualStage _ _ _ _,
     .residualStage _ _ _ _, .residualStage _ _ _ _, .globalAvgPool, .dense _ _] =>
      dense ∘ gap ∘ chainComp ids4 ∘ down4 ∘ chainComp ids3 ∘ down3 ∘
        chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem
  | _ => fun _ => 0

/-- **Spec ≡ the representative proven skeleton.** -/
theorem r34Rep_denote_eq {s0 s1 s2 s3 s4 s5 s6 s7 : Nat}
    (stem : Vec s0 → Vec s1) (mp : Vec s1 → Vec s2)
    (ids1 : List (Vec s2 → Vec s2))
    (down2 : Vec s2 → Vec s3) (ids2 : List (Vec s3 → Vec s3))
    (down3 : Vec s3 → Vec s4) (ids3 : List (Vec s4 → Vec s4))
    (down4 : Vec s4 → Vec s5) (ids4 : List (Vec s5 → Vec s5))
    (gap : Vec s5 → Vec s6) (dense : Vec s6 → Vec s7) :
    denoteR34Rep
      [.convBn 3 64 7 2, .maxPool 2 2, .residualStage 64 64 3 1, .residualStage 64 128 4 2,
       .residualStage 128 256 6 2, .residualStage 256 512 3 2, .globalAvgPool, .dense 512 10]
      stem mp ids1 down2 ids2 down3 ids3 down4 ids4 gap dense
      = dense ∘ gap ∘ chainComp ids4 ∘ down4 ∘ chainComp ids3 ∘ down3 ∘
        chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem := rfl

/-- **The representative spec carries the math** (canonical witness; the honest conditional
    fold through the [3,4,6,3] stages is `Proofs.resnet34_has_vjp_at`). -/
noncomputable def r34Rep_has_vjp {s0 s1 s2 s3 s4 s5 s6 s7 : Nat}
    (stem : Vec s0 → Vec s1) (mp : Vec s1 → Vec s2)
    (ids1 : List (Vec s2 → Vec s2))
    (down2 : Vec s2 → Vec s3) (ids2 : List (Vec s3 → Vec s3))
    (down3 : Vec s3 → Vec s4) (ids3 : List (Vec s4 → Vec s4))
    (down4 : Vec s4 → Vec s5) (ids4 : List (Vec s5 → Vec s5))
    (gap : Vec s5 → Vec s6) (dense : Vec s6 → Vec s7) :
    HasVJP (denoteR34Rep
      [.convBn 3 64 7 2, .maxPool 2 2, .residualStage 64 64 3 1, .residualStage 64 128 4 2,
       .residualStage 128 256 6 2, .residualStage 256 512 3 2, .globalAvgPool, .dense 512 10]
      stem mp ids1 down2 ids2 down3 ids3 down4 ids4 gap dense) where
  backward x dy i :=
    ∑ j : Fin s7, pdiv (denoteR34Rep
      [.convBn 3 64 7 2, .maxPool 2 2, .residualStage 64 64 3 1, .residualStage 64 128 4 2,
       .residualStage 128 256 6 2, .residualStage 256 512 3 2, .globalAvgPool, .dense 512 10]
      stem mp ids1 down2 ids2 down3 ids3 down4 ids4 gap dense) x i j * dy j
  correct _ _ _ := rfl

/-! ## Rung E (ch7 mnv2, representative): the spec's math ↔ the **generated** MLIR

The forward graph `mobilenetv2FwdGraphFull` (StableHLO) — the strided 6-block render —
denotes the representative spec's forward: `den graph = mobilenetv2Forward_full`
(`mobilenetv2FwdGraphFull_faithful`) composed with `mobilenetv2Rep_denote_eq` gives
`den graph = denoteMobilenet mobilenetv2RepLayers`. So the generated StableHLO provably
computes the representative spec's function — the A+B+C+E ladder at the 6-block witness
(the committed 17-block spec's E rung is `mobilenetv2Verified_fwd_faithful`, above). E is
`simp`-based, so it does NOT hit the
VJP-fold's concrete-dim `isDefEq` wall. (Forward only; the backward graph + the `.mlir` re-route
off the committed `tests/Test*` string emitter are the remaining E work — see planning doc.) -/
open Proofs.StableHLO in
theorem mobilenetv2Rep_fwd_faithful
    (epsStr : String)
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs γs βs : ℝ)
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 γe1 βe1 : ℝ)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 γd1 βd1 : ℝ)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 γp1 βp1 : ℝ)
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 γe2 βe2 : ℝ)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 γd2 βd2 : ℝ)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 γp2 βp2 : ℝ)
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 γe3 βe3 : ℝ)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 γd3 βd3 : ℝ)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 γp3 βp3 : ℝ)
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 γe4 βe4 : ℝ)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 γd4 βd4 : ℝ)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 γp4 βp4 : ℝ)
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 γe5 βe5 : ℝ)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 γd5 βd5 : ℝ)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 γp5 βp5 : ℝ)
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 γe6 βe6 : ℝ)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 γd6 βd6 : ℝ)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 γp6 βp6 : ℝ)
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh γh βh : ℝ)
    (Wfc : Mat 128 10) (bfc : Vec 10)
    (x : Vec (3 * 224 * 224)) :
    den (mobilenetv2FwdGraphFull epsStr Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc x)
      = denoteMobilenet mobilenetv2RepLayers Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc x :=
  (mobilenetv2FwdGraphFull_faithful epsStr Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc x).trans
    (congrFun (mobilenetv2Rep_denote_eq Ws bs εs γs βs We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2 We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4 We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 Wh bh εh γh βh Wfc bfc).symm x)

/-! ## Rung E (ch9 convnext, representative): the spec's math ↔ the generated MLIR

The representative forward graph `convNextFwdGraph` (StableHLO; patchify → LN → block×2 →
GAP → head-LN → dense, via `geluF`/`layerScaleF`/`bnF`/`addV`) denotes the representative
`convNextForward` (`convNextFwdGraph_faithful`), composed with `convnextRep_denote_eq` ⇒
`den graph = denoteConvnextRep <rep layers>`. So convnext has the representative A+B+C+E(fwd)
ladder. (Scalar LN; the full-render E is `convnextVerified_fwd_faithful` above.) -/
open Proofs.StableHLO in
theorem convnextRep_fwd_faithful {ic c cExp h w kH kW nClasses : Nat}
    (epsStr : String)
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) :
    den (convNextFwdGraph epsStr Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd x)
      = denoteConvnextRep (h := h) (w := w)
          [.conv ic c 1 1, .bn, .convNextBlock c, .convNextBlock c, .globalAvgPool, .bn, .dense c nClasses]
          Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd x :=
  (convNextFwdGraph_faithful epsStr Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd x).trans
    (congrFun (convnextRep_denote_eq Wst bst εst γst βst Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ εhd γhd βhd Wd bd).symm x)
