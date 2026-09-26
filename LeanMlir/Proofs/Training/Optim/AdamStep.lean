import LeanMlir.Proofs.Foundation.Tensor

/-! # The Adam / AdamW optimizer step over ℝ

The ℝ reference for `vit-train`'s optimizer. Coordinatewise over `Vec`, mirroring the
emitted StableHLO update (`MlirCodegen.emitAdamUpdate`) op-for-op so the faithfulness
theorem `adamW_triple_faithful` (Codegen/StableHLO/Basic.lean) is a structural match.

Unlike `SgdDescent`, this file proves **no** descent guarantee: Adam is not a
monotone descent method (Reddi et al. 2018, the AMSGrad counterexample), so the
verified target is *faithfulness* (the rendered update equals `adamWStep` of the
certified gradient) plus *well-definedness* (the `√v̂ + ε` denominator is strictly
positive) — NOT a loss-decrease bound.

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
    {v g : Vec n} (hv : ∀ i, 0 ≤ v i) (i : Fin n) : 0 ≤ adamVNext β₂ v g i :=
  add_nonneg (mul_nonneg hβ₂0 (hv i)) (mul_nonneg (sub_nonneg.2 hβ₂1) (sq_nonneg _))

/-- **Well-definedness of the AdamW update.** The denominator `√(v'/bc₂) + ε` is
    strictly positive whenever `ε > 0` — `Real.sqrt` is unconditionally nonnegative,
    so there is no division by zero in `adamWParam` (the analogue of the BatchNorm
    `0 < ε` positivity side condition, but unconditional in `v`). -/
theorem adam_denom_pos {β₂ ε bc₂ : ℝ} (hε : 0 < ε) {v g : Vec n} (i : Fin n) :
    0 < Real.sqrt (adamVNext β₂ v g i / bc₂) + ε :=
  add_pos_of_nonneg_of_pos (Real.sqrt_nonneg _) hε

end Proofs
