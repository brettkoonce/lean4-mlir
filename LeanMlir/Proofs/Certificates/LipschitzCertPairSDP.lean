import LeanMlir.Proofs.Certificates.LipschitzCertInstance
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Algebra.Order.Star.Real

/-! # Per-pair LipSDP certificates (Fazlyab–Robey–Hassani–Morari–Pappas 2019)

The tighter-Lipschitz lever for the scorecard: replace the global product
bound `√2·‖W₂‖·‖W₁‖` on a pairwise logit gap by the **per-pair LipSDP
constant** — for the gap `g(x) = ⟨v, relu(W₁x)⟩` with `v = W₂ᵢ − W₂ⱼ`,

    (g x − g x')² ≤ ρ·‖x − x'‖²

holds whenever `S = 2·diag(T) − vvᵀ − (1/ρ)·T G₁ T` is PSD for some
diagonal `T ⪰ 0` (`G₁ = W₁W₁ᵀ`, so `S` is h×h — hidden-width-sized, not
input-sized, by Schur complement). The PSD witness entering Lean is an
exact rational LDLᵀ factorization (`S = L·diag(d)·Lᵀ`, `d ≥ 0`) —
kernel-checkable, no `√`, no eigenvalues, no `native_decide`.

The mathematical content: ReLU is slope-restricted in `[0,1]`
(`relu_slope_restricted`), so for any `T ⪰ 0` the incremental quadratic
constraint `Σₖ Tₖ·Δyₖ(Δuₖ − Δyₖ) ≥ 0` holds between any two activation
patterns; adding it to the target and completing the square in `Δx`
reduces `(vᵀΔy)² ≤ ρ‖Δx‖²` to `ΔyᵀSΔy ≥ 0` (`pair_sq_bound`). Everything
stays in squares — the final per-image check is `Lp·ε ≤ margin` with a
rational `Lp`, `ρ ≤ Lp²` (`certified_at_eps_pair`); `√ρ` never appears.

This is the one-hidden-layer instance of LipSDP-Neuron; the SDP is solved
numerically OFF-line (`scripts/lipschitz_cert_pair_sdp.py`) and only the
rationalized certificate `(ρ, T, L, d)` enters Lean as DATA, verified
exactly. -/

namespace Proofs
namespace LipschitzCertDemo

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § ReLU slope restriction (the incremental quadratic constraint)
-- ════════════════════════════════════════════════════════════════

/-- ReLU is slope-restricted in `[0,1]`: the increment `Δy = relu a − relu b`
    satisfies `Δy·(Δu − Δy) ≥ 0` where `Δu = a − b`. The four sign cases. -/
theorem relu_slope_restricted (a b : ℝ) :
    0 ≤ (max a 0 - max b 0) * ((a - b) - (max a 0 - max b 0)) := by
  rcases le_or_gt a 0 with ha | ha <;> rcases le_or_gt b 0 with hb | hb
  · simp [max_eq_right ha, max_eq_right hb]
  · rw [max_eq_right ha, max_eq_left hb.le]
    nlinarith
  · rw [max_eq_left ha.le, max_eq_right hb]
    nlinarith
  · rw [max_eq_left ha.le, max_eq_left hb.le]
    simp

-- ════════════════════════════════════════════════════════════════
-- § The master bound: slope restriction + complete the square in Δx
-- ════════════════════════════════════════════════════════════════

/-- **Per-pair LipSDP squared bound.** If the slack inequality holds for all
    `z` (discharged from a rational LDLᵀ witness: one `linarith` over its column squares),
    then the ReLU-network gap `g(x) = Σₖ vₖ·relu((W x)ₖ)` satisfies
    `(g x − g x')² ≤ ρ·‖x − x'‖²`. No `√ρ`: everything in squares. -/
theorem pair_sq_bound {n h : ℕ} (W : Fin h → Fin n → ℝ)
    (G : Fin h → Fin h → ℝ) (hG : ∀ a b, G a b = ∑ j, W a j * W b j)
    (v T : Fin h → ℝ) (hT : ∀ k, 0 ≤ T k) {ρ : ℝ} (hρ : 0 < ρ)
    (hS : ∀ z : Fin h → ℝ,
      (∑ k, v k * z k) ^ 2
        + (1/ρ) * (∑ a, ∑ b, (T a * z a) * (G a b * (T b * z b)))
        ≤ 2 * ∑ k, T k * z k ^ 2)
    (x x' : EuclideanSpace ℝ (Fin n)) :
    ((∑ k, v k * max (denseE W x k) 0) - (∑ k, v k * max (denseE W x' k) 0)) ^ 2
      ≤ ρ * ‖x - x'‖ ^ 2 := by
  set z : Fin h → ℝ :=
    fun k => max (denseE W x k) 0 - max (denseE W x' k) 0 with hz
  have hzk : ∀ k, z k = max (denseE W x k) 0 - max (denseE W x' k) 0 :=
    fun k => rfl
  -- LHS = (Σ v·z)²
  have hlhs : (∑ k, v k * max (denseE W x k) 0)
      - (∑ k, v k * max (denseE W x' k) 0) = ∑ k, v k * z k := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun k _ => by rw [hzk k]; ring
  -- layer linearity: Δu = W·Δx coordinatewise
  have hdu : ∀ k, denseE W x k - denseE W x' k = ∑ j, W k j * (x j - x' j) := by
    intro k
    rw [denseE_apply, denseE_apply, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  -- slope restriction summed against T:  Σ T z² ≤ Σ w·Δx  (w = Wᵀ(T∘z))
  have hCD : ∑ k, T k * z k ^ 2
      ≤ ∑ j, (∑ k, W k j * (T k * z k)) * (x j - x' j) := by
    have hsum : 0 ≤ ∑ k, T k * (z k * ((∑ j, W k j * (x j - x' j)) - z k)) := by
      refine Finset.sum_nonneg fun k _ => ?_
      rw [← hdu k, hzk k]
      exact mul_nonneg (hT k) (relu_slope_restricted _ _)
    have hsplit : ∑ k, T k * (z k * ((∑ j, W k j * (x j - x' j)) - z k))
        = (∑ k, T k * (z k * (∑ j, W k j * (x j - x' j))))
          - ∑ k, T k * z k ^ 2 := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun k _ => by ring
    have hswap : ∑ k, T k * (z k * (∑ j, W k j * (x j - x' j)))
        = ∑ j, (∑ k, W k j * (T k * z k)) * (x j - x' j) := by
      calc ∑ k, T k * (z k * (∑ j, W k j * (x j - x' j)))
          = ∑ k, ∑ j, (W k j * (T k * z k)) * (x j - x' j) := by
            refine Finset.sum_congr rfl fun k _ => ?_
            rw [Finset.mul_sum, Finset.mul_sum]
            exact Finset.sum_congr rfl fun j _ => by ring
        _ = ∑ j, ∑ k, (W k j * (T k * z k)) * (x j - x' j) :=
            Finset.sum_comm
        _ = ∑ j, (∑ k, W k j * (T k * z k)) * (x j - x' j) :=
            Finset.sum_congr rfl fun j _ => (Finset.sum_mul _ _ _).symm
    rw [hsplit, hswap] at hsum
    linarith
  -- Gram identity:  Σ w² = the TGT double sum
  have hEB : ∑ j, (∑ k, W k j * (T k * z k)) ^ 2
      = ∑ a, ∑ b, (T a * z a) * (G a b * (T b * z b)) := by
    rw [sum_sq_matTvec_eq W (fun k => T k * z k) G hG]
    simp only [Finset.mul_sum]
  -- complete the square:  2ρ·Σ w·Δx ≤ ρ²·Σ Δx² + Σ w²
  have hsq : 2 * ρ * (∑ j, (∑ k, W k j * (T k * z k)) * (x j - x' j))
      ≤ ρ ^ 2 * (∑ j, (x j - x' j) ^ 2)
        + ∑ j, (∑ k, W k j * (T k * z k)) ^ 2 := by
    have hnn : 0 ≤ ∑ j, (ρ * (x j - x' j) - ∑ k, W k j * (T k * z k)) ^ 2 :=
      Finset.sum_nonneg fun j _ => sq_nonneg _
    have hexp : ∑ j, (ρ * (x j - x' j) - ∑ k, W k j * (T k * z k)) ^ 2
        = ρ ^ 2 * (∑ j, (x j - x' j) ^ 2)
          - 2 * ρ * (∑ j, (∑ k, W k j * (T k * z k)) * (x j - x' j))
          + ∑ j, (∑ k, W k j * (T k * z k)) ^ 2 := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib,
        ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [hexp] at hnn
    linarith
  -- assemble: ρ·(Σvz)² ≤ ρ²·ΣΔx², then divide by ρ
  have hslack := hS z
  have hmul : ρ * ((∑ k, v k * z k) ^ 2)
      + (∑ a, ∑ b, (T a * z a) * (G a b * (T b * z b)))
      ≤ 2 * ρ * (∑ k, T k * z k ^ 2) := by
    have h := mul_le_mul_of_nonneg_left hslack hρ.le
    have e : ρ * ((∑ k, v k * z k) ^ 2
        + (1/ρ) * (∑ a, ∑ b, (T a * z a) * (G a b * (T b * z b))))
        = ρ * ((∑ k, v k * z k) ^ 2)
          + (∑ a, ∑ b, (T a * z a) * (G a b * (T b * z b))) := by
      rw [mul_add, ← mul_assoc, mul_one_div_cancel hρ.ne', one_mul]
    rw [e] at h
    linarith
  have hCDρ : 2 * ρ * (∑ k, T k * z k ^ 2)
      ≤ 2 * ρ * (∑ j, (∑ k, W k j * (T k * z k)) * (x j - x' j)) := by
    have h2ρ : (0:ℝ) ≤ 2 * ρ := by positivity
    exact mul_le_mul_of_nonneg_left hCD h2ρ
  have hfinal : ρ * ((∑ k, v k * z k) ^ 2)
      ≤ ρ * (ρ * (∑ j, (x j - x' j) ^ 2)) := by
    rw [← hEB] at hmul
    nlinarith [hmul, hCDρ, hsq]
  have hnorm : ‖x - x'‖ ^ 2 = ∑ j, (x j - x' j) ^ 2 := by
    rw [euclid_norm_sq]
    exact Finset.sum_congr rfl fun j _ => rfl
  rw [hlhs, hnorm]
  exact le_of_mul_le_mul_left hfinal hρ

-- ════════════════════════════════════════════════════════════════
-- § Bridges to the scorecard MLP and the fixed-ε certificate
-- ════════════════════════════════════════════════════════════════

/-- A pairwise logit gap of `dense ∘ relu ∘ dense` is the ⟨rowᵢ−rowⱼ, relu·⟩
    form `pair_sq_bound` speaks about. -/
theorem mlp_gap_eq {n h k : ℕ} (W1 : Fin h → Fin n → ℝ)
    (W2 : Fin k → Fin h → ℝ) (i j : Fin k) (x : EuclideanSpace ℝ (Fin n)) :
    (denseE W2 ∘ reluE ∘ denseE W1) x i - (denseE W2 ∘ reluE ∘ denseE W1) x j
      = ∑ t, (W2 i t - W2 j t) * max (denseE W1 x t) 0 := by
  show (∑ t, W2 i t * (reluE (denseE W1 x)) t)
      - (∑ t, W2 j t * (reluE (denseE W1 x)) t) = _
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [reluE_apply]
  ring

/-- **`pair_sq_bound` on the net's own logits.** The gap `f · i − f · j` of
    `f = denseE W2 ∘ reluE ∘ denseE W1`, with `v` the row difference `W2 i − W2 j`, satisfies the
    squared LipSDP bound whenever the slack certificate `hS` does; every emitted `pairSq*` theorem
    is this at one class pair. -/
theorem pair_sq_bound_mlp {n h k : ℕ} (W1 : Fin h → Fin n → ℝ) (W2 : Fin k → Fin h → ℝ)
    (G : Fin h → Fin h → ℝ) (hG : ∀ a b, G a b = ∑ j, W1 a j * W1 b j) (i j : Fin k)
    (v T : Fin h → ℝ) (hv : ∀ t, v t = W2 i t - W2 j t) (hT : ∀ t, 0 ≤ T t) {ρ : ℝ}
    (hρ : 0 < ρ)
    (hS : ∀ z : Fin h → ℝ,
      (∑ t, v t * z t) ^ 2
        + (1/ρ) * (∑ a, ∑ b, (T a * z a) * (G a b * (T b * z b)))
        ≤ 2 * ∑ t, T t * z t ^ 2)
    (x x' : EuclideanSpace ℝ (Fin n)) :
    (((denseE W2 ∘ reluE ∘ denseE W1) x i - (denseE W2 ∘ reluE ∘ denseE W1) x j)
      - ((denseE W2 ∘ reluE ∘ denseE W1) x' i - (denseE W2 ∘ reluE ∘ denseE W1) x' j)) ^ 2
      ≤ ρ * ‖x - x'‖ ^ 2 := by
  have e : ∀ w, (denseE W2 ∘ reluE ∘ denseE W1) w i - (denseE W2 ∘ reluE ∘ denseE W1) w j
      = ∑ t, v t * max (denseE W1 w t) 0 := fun w => by
    rw [mlp_gap_eq]; exact Finset.sum_congr rfl fun t _ => by rw [hv t]
  rw [e x, e x']
  exact pair_sq_bound W1 G hG v T hT hρ hS x x'

/-- The squared pair bound is symmetric in the class pair: the gap for `(j, i)` is the negated gap
    for `(i, j)`. The emitted reverse-order `pairSq*` theorems are this. -/
theorem pair_sq_symm {n k : ℕ} {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k)}
    {i j : Fin k} {ρ : ℝ}
    (h : ∀ u u' : EuclideanSpace ℝ (Fin n),
      ((f u i - f u j) - (f u' i - f u' j)) ^ 2 ≤ ρ * ‖u - u'‖ ^ 2) :
    ∀ u u' : EuclideanSpace ℝ (Fin n),
      ((f u j - f u i) - (f u' j - f u' i)) ^ 2 ≤ ρ * ‖u - u'‖ ^ 2 := fun u u' => by
  rw [show (f u j - f u i) - (f u' j - f u' i) = -((f u i - f u j) - (f u' i - f u' j)) by ring,
    neg_sq]
  exact h u u'

/-- **Fixed-ε certificate from a per-pair squared bound.** If the gap
    `f · i − f · j` satisfies the squared LipSDP bound with constant `ρ`,
    `Lp` is a rational majorant (`ρ ≤ Lp²`), and the margin at `x` clears
    `Lp·ε`, then every `‖δ‖ < ε` keeps class `j` strictly below class `i`.
    The per-pair peer of `certified_at_eps` — no `√2`, no global `L`. -/
theorem certified_at_eps_pair {n k : ℕ}
    {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k)}
    {i j : Fin k} {ρ Lp ε : ℝ}
    (hgap : ∀ u u', ((f u i - f u j) - (f u' i - f u' j)) ^ 2
      ≤ ρ * ‖u - u'‖ ^ 2)
    (hρL : ρ ≤ Lp ^ 2) (hLp : 0 < Lp)
    {x : EuclideanSpace ℝ (Fin n)} (hmargin : Lp * ε ≤ f x i - f x j)
    (δ : EuclideanSpace ℝ (Fin n)) (hδ : ‖δ‖ < ε) :
    f (x + δ) j < f (x + δ) i := by
  have hgd := hgap (x + δ) x
  rw [add_sub_cancel_left] at hgd
  have h1 : ((f (x + δ) i - f (x + δ) j) - (f x i - f x j)) ^ 2
      ≤ (Lp * ‖δ‖) ^ 2 := by
    calc ((f (x + δ) i - f (x + δ) j) - (f x i - f x j)) ^ 2
        ≤ ρ * ‖δ‖ ^ 2 := hgd
      _ ≤ Lp ^ 2 * ‖δ‖ ^ 2 :=
          mul_le_mul_of_nonneg_right hρL (sq_nonneg _)
      _ = (Lp * ‖δ‖) ^ 2 := by ring
  have h2 : |(f (x + δ) i - f (x + δ) j) - (f x i - f x j)| ≤ Lp * ‖δ‖ :=
    abs_le_of_sq_le_sq h1 (by positivity)
  have h3 : Lp * ‖δ‖ < Lp * ε := mul_lt_mul_of_pos_left hδ hLp
  have h4 := neg_le_of_abs_le h2
  linarith [hmargin, h3, h4]

end LipschitzCertDemo
end Proofs
