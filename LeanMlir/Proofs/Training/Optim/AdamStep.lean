module

public import LeanMlir.Proofs.Foundation.Tensor

@[expose] public section

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
which threads them as scalar `tensor<f32>` function arguments.

## References

- Kingma & Ba 2015, *Adam: A Method for Stochastic Optimization*. <https://arxiv.org/abs/1412.6980>
- Loshchilov & Hutter 2019, *Decoupled Weight Decay Regularization* (AdamW). <https://arxiv.org/abs/1711.05101> -/

namespace Proofs

variable {n : Nat}

/-- First-moment update: `m' = β₁·m + (1−β₁)·g`. -/
def adamMNext (β₁ : ℝ) (m g : Vec n) : Vec n :=
  fun i => β₁ * m i + (1 - β₁) * g i

/-- Second-moment update: `v' = β₂·v + (1−β₂)·g²`. -/
def adamVNext (β₂ : ℝ) (v g : Vec n) : Vec n :=
  fun i => β₂ * v i + (1 - β₂) * (g i) ^ 2

@[simp] theorem adamMNext_apply (β₁ : ℝ) (m g : Vec n) (i : Fin n) :
    adamMNext β₁ m g i = β₁ * m i + (1 - β₁) * g i := rfl

@[simp] theorem adamVNext_apply (β₂ : ℝ) (v g : Vec n) (i : Fin n) :
    adamVNext β₂ v g i = β₂ * v i + (1 - β₂) * (g i) ^ 2 := rfl

/-- AdamW parameter update (decoupled weight decay), coordinatewise:
    `θ' = θ − lr·( (m'/bc₁) / (√(v'/bc₂) + ε) ) − (wd·lr)·θ`. The `mh/den` shape
    and the trailing `− wd·lr·θ` mirror `emitAdamUpdate` exactly. -/
noncomputable def adamWParam (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) : Vec n :=
  fun i =>
    let mh := adamMNext β₁ m g i / bc₁
    let vh := adamVNext β₂ v g i / bc₂
    θ i - lr * (mh / (Real.sqrt vh + ε)) - (wd * lr) * θ i

theorem adamWParam_apply (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) (i : Fin n) :
    adamWParam β₁ β₂ ε lr wd bc₁ bc₂ θ m v g i
      = θ i - lr * ((adamMNext β₁ m g i / bc₁) / (Real.sqrt (adamVNext β₂ v g i / bc₂) + ε))
          - (wd * lr) * θ i := rfl

/-- One AdamW step: the new parameter together with the new moments
    `(θ', m', v')` — the triple the rendered train step returns per parameter. -/
noncomputable def adamWStep (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) :
    Vec n × Vec n × Vec n :=
  (adamWParam β₁ β₂ ε lr wd bc₁ bc₂ θ m v g, adamMNext β₁ m g, adamVNext β₂ v g)

@[simp] theorem adamWStep_fst (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) :
    (adamWStep β₁ β₂ ε lr wd bc₁ bc₂ θ m v g).1 = adamWParam β₁ β₂ ε lr wd bc₁ bc₂ θ m v g := rfl
@[simp] theorem adamWStep_snd_fst (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) :
    (adamWStep β₁ β₂ ε lr wd bc₁ bc₂ θ m v g).2.1 = adamMNext β₁ m g := rfl
@[simp] theorem adamWStep_snd_snd (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v g : Vec n) :
    (adamWStep β₁ β₂ ε lr wd bc₁ bc₂ θ m v g).2.2 = adamVNext β₂ v g := rfl

/-- **Second-moment invariant.** `v'` stays nonnegative when `0 ≤ β₂ ≤ 1` and the
    incoming `v` is nonnegative — so, starting from `v = 0`, every step keeps
    `√v̂` real and the denominator below well-defined. -/
theorem adamVNext_nonneg {β₂ : ℝ} (hβ₂0 : 0 ≤ β₂) (hβ₂1 : β₂ ≤ 1)
    {v g : Vec n} (hv : ∀ i, 0 ≤ v i) (i : Fin n) : 0 ≤ adamVNext β₂ v g i :=
  add_nonneg (mul_nonneg hβ₂0 (hv i)) (mul_nonneg (sub_nonneg.2 hβ₂1) (sq_nonneg _))

/-- **The AdamW update's outer denominator is positive.** `√(v'/bc₂) + ε` is
    strictly positive whenever `ε > 0` — `Real.sqrt` is unconditionally nonnegative
    (the analogue of the BatchNorm `0 < ε` positivity side condition, but unconditional
    in `v`). `adamWParam` also divides by `bc₁` and `bc₂`, which this does not touch:
    `bc₁, bc₂ > 0` is the driver's `t ≥ 1`. -/
theorem adam_denom_pos {β₂ ε bc₂ : ℝ} (hε : 0 < ε) {v g : Vec n} (i : Fin n) :
    0 < Real.sqrt (adamVNext β₂ v g i / bc₂) + ε :=
  add_pos_of_nonneg_of_pos (Real.sqrt_nonneg _) hε

-- ════════════════════════════════════════════════════════════════
-- § The model-EMA shadow
-- ════════════════════════════════════════════════════════════════

/-- **The model-EMA shadow**, `e' = d·e + (1 − d)·θ'`, read against the UPDATED parameter `θ'`.

    It is `adamMNext` at `β₁ := d`, and the render emits it as exactly that op
    (`RenderKit.optOne`'s `emaTail`: `adamMNextF` at `(%emad, %oemad)` on the optimizer's own
    output), so no new `SHlo` constructor and no new faithfulness proof were needed —
    `StableHLO.emaShadow_faithful` is `adamMNextF_faithful` at that instance. The decay `d` is a
    runtime scalar because the reference warmup-corrects it, `d = min(decay, (1 + t)/(10 + t))`;
    that schedule is the driver's and is not stated here. -/
def emaNext (d : ℝ) (e θ' : Vec n) : Vec n := adamMNext d e θ'

theorem emaNext_apply (d : ℝ) (e θ' : Vec n) (i : Fin n) :
    emaNext d e θ' i = d * e i + (1 - d) * θ' i := rfl

/-- Against the fixed updated weight `θ'`, one EMA update scales the shadow's gap by `d`. A
    one-update identity, not a bound on the lag along a trajectory (the weight moves too). -/
theorem emaNext_lag (d : ℝ) (e θ' : Vec n) (i : Fin n) :
    emaNext d e θ' i - θ' i = d * (e i - θ' i) := by
  simp only [emaNext_apply]; ring

/-- At `d = 1` the shadow is frozen; at `d = 0` it is the live weight. -/
theorem emaNext_one (e θ' : Vec n) : emaNext 1 e θ' = e := by
  funext i; simp [emaNext_apply]

theorem emaNext_zero (e θ' : Vec n) : emaNext 0 e θ' = θ' := by
  funext i; simp [emaNext_apply]

end Proofs
