import LeanMlir.Proofs.Foundation.Tensor
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-! # The Adam / AdamW optimizer step over ℝ — the verified core (Phase 3a)

The ℝ reference for `vit-train`'s optimizer, the load-bearing rung of
`planning/vit_train_to_vit_verified.md`. Coordinatewise over `Vec`, mirroring the
emitted StableHLO update (`MlirCodegen.emitAdamUpdate`) op-for-op so the later
faithfulness theorem (`den (adamGraph) = adamWStep …`) is a structural match.

Unlike `SgdDescent`, this file proves **no** descent guarantee: Adam is not a
monotone descent method (Reddi et al. 2018, the AMSGrad counterexample), so the
verified target is *faithfulness* (the rendered update equals `adamWStep` of the
certified gradient) plus *well-definedness* (the `√v̂ + ε` denominator is strictly
positive) — NOT a loss-decrease bound. See the doc's proof/host boundary note.

`bc₁`/`bc₂` are the bias-correction denominators `1 − β₁ᵗ` / `1 − β₂ᵗ`, passed in
(host-computed per step) rather than recomputed in-graph — matching the emitter,
which threads them as scalar `tensor<f32>` function arguments. -/

namespace Proofs

variable {n : Nat}

/-- First-moment update: `m' = β₁·m + (1−β₁)·g`. -/
def adamMNext (β₁ : ℝ) (m g : Vec n) : Vec n :=
  fun i => β₁ * m i + (1 - β₁) * g i

/-- Second-moment update: `v' = β₂·v + (1−β₂)·g²`. -/
def adamVNext (β₂ : ℝ) (v g : Vec n) : Vec n :=
  fun i => β₂ * v i + (1 - β₂) * (g i) ^ 2

/-- AdamW parameter update (decoupled weight decay), coordinatewise:
    `θ' = θ − lr·( (m'/bc₁) / (√(v'/bc₂) + ε) ) − (wd·lr)·θ`. The `mh/den` shape
    and the trailing `− wd·lr·θ` mirror `emitAdamUpdate` exactly. -/
noncomputable def adamWParam (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) : Vec n :=
  fun i =>
    let mh := adamMNext β₁ m g i / bc₁
    let vh := adamVNext β₂ v g i / bc₂
    θ i - lr * (mh / (Real.sqrt vh + ε)) - (wd * lr) * θ i

/-- One AdamW step: the new parameter together with the new moments
    `(θ', m', v')` — the triple the rendered train step returns per parameter. -/
noncomputable def adamWStep (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) :
    Vec n × Vec n × Vec n :=
  (adamWParam β₁ β₂ ε lr wd bc₁ bc₂ θ m v g, adamMNext β₁ m g, adamVNext β₂ v g)

/-- **Second-moment invariant.** `v'` stays nonnegative when `0 ≤ β₂ ≤ 1` and the
    incoming `v` is nonnegative — so, starting from `v = 0`, every step keeps
    `√v̂` real and the denominator below well-defined. -/
theorem adamVNext_nonneg {β₂ : ℝ} (hβ₂0 : 0 ≤ β₂) (hβ₂1 : β₂ ≤ 1)
    {v g : Vec n} (hv : ∀ i, 0 ≤ v i) (i : Fin n) : 0 ≤ adamVNext β₂ v g i := by
  have h1 : 0 ≤ β₂ * v i := mul_nonneg hβ₂0 (hv i)
  have h2 : 0 ≤ (1 - β₂) * (g i) ^ 2 := mul_nonneg (by linarith) (sq_nonneg _)
  show 0 ≤ β₂ * v i + (1 - β₂) * (g i) ^ 2
  linarith

/-- **Well-definedness of the AdamW update.** The denominator `√(v'/bc₂) + ε` is
    strictly positive whenever `ε > 0` — `Real.sqrt` is unconditionally nonnegative,
    so there is no division by zero in `adamWParam` (the analogue of the BatchNorm
    `0 < ε` positivity side condition, but unconditional in `v`). -/
theorem adam_denom_pos {β₂ ε bc₂ : ℝ} (hε : 0 < ε) {v g : Vec n} (i : Fin n) :
    0 < Real.sqrt (adamVNext β₂ v g i / bc₂) + ε := by
  have hs : 0 ≤ Real.sqrt (adamVNext β₂ v g i / bc₂) := Real.sqrt_nonneg _
  linarith

/-- **Coordinate closed form** — the spec the emitted Adam graph must denote
    (the `adamWParam` analogue of the SGD `θ − lr·certified-grad` render close).
    Holds definitionally; stated so the future `den (adamGraph) = …` faithfulness
    proof has an explicit per-coordinate target. -/
theorem adamWParam_apply (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) (i : Fin n) :
    adamWParam β₁ β₂ ε lr wd bc₁ bc₂ θ m v g i =
      θ i - lr * (((β₁ * m i + (1 - β₁) * g i) / bc₁) /
        (Real.sqrt ((β₂ * v i + (1 - β₂) * (g i) ^ 2) / bc₂) + ε)) - (wd * lr) * θ i :=
  rfl

/-- **Plain-Adam specialization** (`wd = 0`): the decoupled weight-decay term
    vanishes, recovering textbook Adam. The bridge to the no-weight-decay nets. -/
theorem adamWParam_wd_zero (β₁ β₂ ε lr bc₁ bc₂ : ℝ) (θ m v g : Vec n) (i : Fin n) :
    adamWParam β₁ β₂ ε lr 0 bc₁ bc₂ θ m v g i =
      θ i - lr * (((β₁ * m i + (1 - β₁) * g i) / bc₁) /
        (Real.sqrt ((β₂ * v i + (1 - β₂) * (g i) ^ 2) / bc₂) + ε)) := by
  rw [adamWParam_apply]; ring

/-- **Scalar AdamW update** — one coordinate of `adamWParam`, the form the
    per-entry render-close (`AdamRender.adamW`/`adamB`) applies to a single
    weight/bias entry's certified gradient (the `θ i - lr·…` analogue used by
    `StableHLO.sgdW`). -/
noncomputable def adamWScalar (β₁ β₂ ε lr wd bc₁ bc₂ θ m v g : ℝ) : ℝ :=
  θ - lr * (((β₁ * m + (1 - β₁) * g) / bc₁) /
    (Real.sqrt ((β₂ * v + (1 - β₂) * g ^ 2) / bc₂) + ε)) - (wd * lr) * θ

/-- The `Vec` spec is the scalar update applied coordinatewise — so a render that
    drives `adamWScalar` per entry computes exactly `adamWParam`. -/
theorem adamWParam_eq_scalar (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) (i : Fin n) :
    adamWParam β₁ β₂ ε lr wd bc₁ bc₂ θ m v g i
      = adamWScalar β₁ β₂ ε lr wd bc₁ bc₂ (θ i) (m i) (v i) (g i) := rfl

end Proofs
