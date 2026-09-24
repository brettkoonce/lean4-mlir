import LeanMlir.Proofs.Foundation.MLP
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Inv

/-! # Softmax and the softmax–cross-entropy gradient

`softmax` and `crossEntropy` are defined in `MLP.lean`; this file differentiates them:
`softmax_diff`, the Jacobian `pdiv_softmax`, the global `softmax_has_vjp`, and
`softmaxCE_grad` — ∂(crossEntropy ∘ softmax)/∂logits = softmax − onehot, the cotangent every
classifier's train step starts from. Attention (`rowSoftmax`) and the small-net IR build on it.
-/

open Finset BigOperators

namespace Proofs

/-- Differentiability of `softmax c`: each coordinate is `exp(z k) · (Σ_j exp(z j))⁻¹`, and the
    denominator is positive. -/
lemma softmax_diff (c : Nat) : Differentiable ℝ (softmax c) := by
  match c with
  | 0 => rw [Subsingleton.elim (softmax 0) fun _ => 0]; exact differentiable_const _
  | c + 1 =>
    unfold softmax; simp only [div_eq_mul_inv]
    fun_prop (disch := intro z; positivity)


/-! ## The softmax Jacobian

For `p = softmax(z)` with `p_j = exp(z_j) / sum_k exp(z_k)`, the quotient
rule gives:

    dp_j/dz_i = p_j * (delta_{ij} - p_i)

This is the famous "diag minus outer product" form:

    J = diag(p) - p * p^T

Dense (every output depends on every input), but **rank-1 correction
to a diagonal** — which means the VJP has a closed-form collapse, just
like BatchNorm did.
-/

/-- **Partial derivative of softmax** (quotient rule on the exponentials).

    `d(softmax(z))_j/dz_i = softmax(z)_j * (delta_{ij} - softmax(z)_i)`

    Proved (was an axiom). The j-th coord of `softmax c z` is
    `Real.exp (z j) / S` with `S := Σ_k Real.exp (z k) > 0`, so the j-th
    output coord function `z' ↦ exp(z' j) * (Σ_k exp(z' k))⁻¹` has
    `HasFDerivAt` derivative built from `HasFDerivAt.exp`,
    `HasFDerivAt.fun_sum`, `(hasDerivAt_inv ·).comp_hasFDerivAt`, and
    `HasFDerivAt.mul`. Evaluating that CLM at `basisVec i` and
    collapsing `Σ_k exp(z k) · δ_{ki} = exp(z i)` gives the formula. -/
theorem pdiv_softmax (c : Nat) (z : Vec c) (i j : Fin c) :
    pdiv (softmax c) z i j =
    softmax c z j * ((if i = j then 1 else 0) - softmax c z i) := by
  cases c with
  | zero => exact j.elim0
  | succ c' =>
  rw [pdiv_eq_fderiv_coord (softmax_diff (c' + 1) z)]
  rw [show (fun z' : Vec (c' + 1) => softmax (c' + 1) z' j) =
         (fun z' => Real.exp (z' j) * (∑ k : Fin (c' + 1), Real.exp (z' k))⁻¹) from by
    funext z'
    rw [softmax_apply, div_eq_mul_inv]]
  set S : ℝ := ∑ k : Fin (c' + 1), Real.exp (z k) with hS_def
  have hS_pos : (0 : ℝ) < S :=
    Finset.sum_pos (fun k _ => Real.exp_pos _) Finset.univ_nonempty
  have hS_ne : S ≠ 0 := hS_pos.ne'
  -- HasFDerivAt building blocks
  have h_proj : ∀ k : Fin (c' + 1),
      HasFDerivAt (fun z' : Vec (c' + 1) => z' k)
                  (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ) z :=
    fun k => (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ).hasFDerivAt
  have h_exp : ∀ k : Fin (c' + 1),
      HasFDerivAt (fun z' : Vec (c' + 1) => Real.exp (z' k))
                  (Real.exp (z k) • (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ)) z :=
    fun k => (h_proj k).exp
  have h_sum : HasFDerivAt
      (fun z' : Vec (c' + 1) => ∑ k : Fin (c' + 1), Real.exp (z' k))
      (∑ k : Fin (c' + 1), Real.exp (z k) •
          (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ)) z :=
    HasFDerivAt.fun_sum (fun k _ => h_exp k)
  have h_inv : HasFDerivAt
      (fun z' : Vec (c' + 1) => (∑ k : Fin (c' + 1), Real.exp (z' k))⁻¹)
      ((-(S ^ 2)⁻¹) • (∑ k : Fin (c' + 1), Real.exp (z k) •
          (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ))) z :=
    (hasDerivAt_inv hS_ne).comp_hasFDerivAt z h_sum
  have h_mul : HasFDerivAt
      (fun z' : Vec (c' + 1) =>
          Real.exp (z' j) * (∑ k : Fin (c' + 1), Real.exp (z' k))⁻¹)
      (Real.exp (z j) • ((-(S ^ 2)⁻¹) • (∑ k : Fin (c' + 1), Real.exp (z k) •
            (ContinuousLinearMap.proj k : Vec (c' + 1) →L[ℝ] ℝ))) +
       S⁻¹ • (Real.exp (z j) • (ContinuousLinearMap.proj j : Vec (c' + 1) →L[ℝ] ℝ))) z :=
    (h_exp j).mul h_inv
  rw [h_mul.fderiv]
  -- Evaluate the resulting CLM at basisVec i and simplify.
  simp only [add_apply, smul_apply, smul_eq_mul,
             _root_.sum_apply, ContinuousLinearMap.proj_apply, basisVec_apply]
  -- Collapse the Kronecker sum: Σ_k exp(z k) * (if k = i then 1 else 0) = exp(z i).
  rw [show (∑ k : Fin (c' + 1), Real.exp (z k) * (if k = i then (1 : ℝ) else 0)) =
        Real.exp (z i) from by simp]
  -- Unfold softmax on the RHS and convert `if j = i` to `if i = j`.
  show Real.exp (z j) * (-(S ^ 2)⁻¹ * Real.exp (z i)) +
       S⁻¹ * (Real.exp (z j) * (if j = i then (1 : ℝ) else 0)) =
       (Real.exp (z j) / S) * ((if i = j then (1 : ℝ) else 0) - Real.exp (z i) / S)
  simp only [@eq_comm _ j i]
  field_simp
  ring

/-- **Softmax VJP — the closed-form collapse.**

    `back(z, dy)_i = p_i * (dy_i - <p, dy>)`

    where `p = softmax(z)` and `<p, dy> = sum_j p_j * dy_j` is one scalar.

    **Read this carefully.** The naive VJP would be:
      dz_i = sum_j J_{ji} * dy_j = sum_j (p_j * (delta_{ij} - p_i)) * dy_j

    That's O(c) per entry, O(c^2) total. But expanding:
      dz_i = p_i * dy_i - p_i * sum_j p_j * dy_j
           = p_i * (dy_i - <p, dy>)

    The rank-1 correction lets you **precompute one scalar** (`<p, dy>`)
    and apply it to every entry. **Total work: O(c).** Same optimization
    pattern as BN (one reduction + a broadcast) and max-pool (one
    comparison + a select).

    **Interpretation.** Softmax outputs a probability distribution. Its
    backward subtracts the "weighted average of the incoming gradient
    under that distribution" from each entry, then scales by the
    entry's probability. Entries with low probability get small
    gradients (because the softmax flattened them in the forward);
    entries with high probability get gradients proportional to how
    much they deviate from the weighted-average cotangent.

    This is the one place where "softmax means softly select one thing"
    maps directly to "softmax backward selectively amplifies the
    gradient for the winning class." -/
noncomputable def softmax_has_vjp (c : Nat) : HasVJP (softmax c) where
  backward := fun z dy =>
    let p : Vec c := softmax c z
    let s : ℝ := ∑ j : Fin c, p j * dy j  -- <p, dy>
    fun i => p i * (dy i - s)
  correct := by
    intro z dy i
    -- `Σ_j p_j (δ_ij - p_i) dy_j`: the Kronecker term collapses to `p_i dy_i`.
    simp only [pdiv_softmax, mul_sub, sub_mul, Finset.sum_sub_distrib, mul_ite, ite_mul, mul_one,
      mul_zero, zero_mul, Finset.sum_ite_eq, Finset.mem_univ, ite_true, Finset.mul_sum]
    exact congrArg _ (Finset.sum_congr rfl fun j _ => by ring)

/-- **`crossEntropy` is differentiable in the logits.** `softmax > 0` lets `Real.log` (hence
    `crossEntropy = -log(softmax · label)`) inherit smoothness. -/
@[fun_prop]
theorem crossEntropy_differentiable (c : Nat) (label : Fin c) :
    Differentiable ℝ (fun z : Vec c => crossEntropy c z label) := by
  cases c with
  | zero => exact label.elim0
  | succ c' =>
    unfold crossEntropy softmax; simp only [div_eq_mul_inv]
    fun_prop (disch := intro z; positivity)

/-- **Softmax cross-entropy scalar gradient** — proved (was an axiom in
    MLP.lean; relocated here to use `pdiv_softmax`).

    `∂(-log softmax(z)[label])/∂z_j = softmax(z)_j - onehot(label)_j`

    Stated using `pdiv` on a `Vec 1`-valued wrapper (cross-entropy is
    naturally scalar, but `pdiv` is defined for `Vec → Vec`; we just
    take the only output index). Proof: `pdiv_eq_fderiv_coord` extracts the
    only coord, then `HasFDerivAt.log` (with `softmax z label > 0`)
    composed with `softmax_diff` gives the derivative of the inner
    `Real.log`. Negating and evaluating at `basisVec j` reduces via
    `pdiv_softmax` to the expected formula. -/
theorem softmaxCE_grad (c : Nat) (logits : Vec c) (label : Fin c) (j : Fin c) :
    pdiv (fun (z : Vec c) (_ : Fin 1) => crossEntropy c z label) logits j 0
    = softmax c logits j - oneHot c label j := by
  cases c with
  | zero => exact label.elim0
  | succ c' =>
  have h_softmax_pos : ∀ z : Vec (c' + 1), 0 < softmax (c' + 1) z label := fun z =>
    div_pos (Real.exp_pos _)
      (Finset.sum_pos (fun k _ => Real.exp_pos _) Finset.univ_nonempty)
  have hp_ne : softmax (c' + 1) logits label ≠ 0 := (h_softmax_pos logits).ne'
  -- Differentiability infrastructure.
  have h_softmax_label_diff : Differentiable ℝ
      (fun z : Vec (c' + 1) => softmax (c' + 1) z label) :=
    fun z => differentiableAt_pi.mp ((softmax_diff (c' + 1)) z) label
  have h_ce_pi_diff : Differentiable ℝ
      (fun z : Vec (c' + 1) => fun _ : Fin 1 => crossEntropy (c' + 1) z label) :=
    differentiable_pi.mpr fun _ => crossEntropy_differentiable (c' + 1) label
  -- Step 1: extract the single (0-th) coord of the Vec 1-valued function.
  rw [pdiv_eq_fderiv_coord (h_ce_pi_diff logits)]
  -- Step 2: HasFDerivAt chain for crossEntropy = -log ∘ softmax_label.
  have h_softmax_at : HasFDerivAt (fun z : Vec (c' + 1) => softmax (c' + 1) z label)
      (fderiv ℝ (fun z => softmax (c' + 1) z label) logits) logits :=
    (h_softmax_label_diff logits).hasFDerivAt
  have h_log_at : HasFDerivAt
      (fun z : Vec (c' + 1) => Real.log (softmax (c' + 1) z label))
      ((softmax (c' + 1) logits label)⁻¹ •
        fderiv ℝ (fun z => softmax (c' + 1) z label) logits) logits :=
    h_softmax_at.log hp_ne
  have h_ce_at : HasFDerivAt
      (fun z : Vec (c' + 1) => crossEntropy (c' + 1) z label)
      (-((softmax (c' + 1) logits label)⁻¹ •
          fderiv ℝ (fun z => softmax (c' + 1) z label) logits)) logits := by
    simp only [crossEntropy_def]
    exact h_log_at.neg
  rw [h_ce_at.fderiv]
  -- Step 3: simplify CLM application at basisVec j.
  simp only [neg_apply, smul_apply, smul_eq_mul]
  -- Step 4: rewrite fderiv of `softmax z label` (in z) as pdiv softmax, then apply pdiv_softmax.
  rw [← pdiv_eq_fderiv_coord (softmax_diff (c' + 1) logits)]
  rw [pdiv_softmax]
  -- Step 5: oneHot unfolds to `if j = label then 1 else 0`; algebra cancels p[label].
  rw [oneHot_apply]
  field_simp
  ring

end Proofs
