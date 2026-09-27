import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Foundation.Batched.BackLinks

/-! # The batched label-smoothed loss, and its gradient is the emitted cotangent

`SmoothedLossCot` identifies the emitted loss cotangent ROW BY ROW: at example `n` it is
`(1/B)·∂softCE/∂z` at that example's logits. This file states the loss as one function of the
flat batch of logits — `smoothedBatchLoss`, the `1/B`-weighted sum of every example's soft-target
cross-entropy against its smoothed target — and proves its full gradient is the emitted cotangent,
read at the head's `N·K` index (`smoothedBatchLoss_grad`). That is the `hg` a parameter-level
statement asks for (`HasGradAt` at the logits, `ParamGrad`).
-/

namespace Proofs

open scoped BigOperators
open StableHLO
open BackLinks (rowB unrowB)

/-- Example `n`'s logits out of the flat `N·K` batch. -/
def logitRow (N K : Nat) (z : Vec (N * K)) (n : Fin N) : Vec K :=
  fun k => z (finProdFinEquiv (n, k))

/-- Example `n`'s target out of the `N·(1·K)` graph input `%onehot`. -/
noncomputable def targetRow (N K : Nat) (t : Vec (N * (1 * K))) (n : Fin N) : Vec K :=
  Mat.unflatten (batchSlice N (1 * K) t n) (0 : Fin 1)

/-- **The batched label-smoothed loss**: `Σ_n softCE(smooth(tₙ), zₙ) / B`, as a function of the
    flat logits. -/
noncomputable def smoothedBatchLoss (N K : Nat) (α B : ℝ) (t : Vec (N * (1 * K)))
    (z : Vec (N * K)) : Vec 1 :=
  fun _ => ∑ n : Fin N, softCE K (smoothTarget K α (targetRow N K t n)) (logitRow N K z n) / B

private theorem softCE_differentiable (K : Nat) (t : Vec K) :
    Differentiable ℝ (fun z : Vec K => fun _ : Fin 1 => softCE K t z) := fun z =>
  differentiableAt_pi.2 fun _ => by
    unfold softCE
    exact DifferentiableAt.fun_sum fun k _ =>
      ((crossEntropy_differentiable K k) z).const_mul (t k)

private theorem logitRow_differentiable (N K : Nat) (n : Fin N) :
    Differentiable ℝ (fun z : Vec (N * K) => logitRow N K z n) :=
  (reindexCLM (fun k : Fin K => finProdFinEquiv (n, k))).differentiable

private theorem lossTerm_differentiableAt (N K : Nat) (s : Vec K) (n : Fin N) (B : ℝ)
    (z : Vec (N * K)) :
    DifferentiableAt ℝ (fun x : Vec (N * K) => softCE K s (logitRow N K x n) / B) z := by
  have h : DifferentiableAt ℝ (fun x : Vec (N * K) => softCE K s (logitRow N K x n)) z :=
    differentiableAt_pi.1 (((softCE_differentiable K s).comp (logitRow_differentiable N K n)) z) 0
  simpa only [div_eq_mul_inv] using h.mul_const B⁻¹

theorem smoothedBatchLoss_differentiable (N K : Nat) (α B : ℝ) (t : Vec (N * (1 * K))) :
    Differentiable ℝ (smoothedBatchLoss N K α B t) := fun z =>
  differentiableAt_pi.2 fun _ => DifferentiableAt.fun_sum fun n _ =>
    lossTerm_differentiableAt N K _ n B z

/-- **The batched loss's gradient**, entry `(n, j)`: example `n`'s own soft-CE gradient at its
    logits, over `B` — no other example contributes. -/
theorem smoothedBatchLoss_pdiv (N K : Nat) (α B : ℝ) (t : Vec (N * (1 * K))) (z : Vec (N * K))
    (n : Fin N) (j : Fin K) :
    pdiv (smoothedBatchLoss N K α B t) z (finProdFinEquiv (n, j)) 0
      = pdiv (fun z' : Vec K => fun _ : Fin 1 =>
            softCE K (smoothTarget K α (targetRow N K t n)) z') (logitRow N K z n) j 0 / B := by
  have hterm : ∀ m : Fin N,
      pdiv (fun z' : Vec (N * K) => fun _ : Fin 1 =>
          softCE K (smoothTarget K α (targetRow N K t m)) (logitRow N K z' m) / B)
        z (finProdFinEquiv (n, j)) 0
      = if m = n then pdiv (fun z' : Vec K => fun _ : Fin 1 =>
            softCE K (smoothTarget K α (targetRow N K t m)) z') (logitRow N K z m) j 0 / B
        else 0 := by
    intro m
    have hc : pdiv (fun z' : Vec (N * K) => fun _ : Fin 1 =>
          B⁻¹ * softCE K (smoothTarget K α (targetRow N K t m)) (logitRow N K z' m))
          z (finProdFinEquiv (n, j)) 0
        = B⁻¹ * pdiv ((fun r : Vec K => fun _ : Fin 1 =>
          softCE K (smoothTarget K α (targetRow N K t m)) r) ∘ fun z' => logitRow N K z' m) z
          (finProdFinEquiv (n, j)) 0 :=
      pdiv_const_smul B⁻¹ ((fun r : Vec K => fun _ : Fin 1 =>
        softCE K (smoothTarget K α (targetRow N K t m)) r) ∘ fun z' => logitRow N K z' m) z
        (((softCE_differentiable K _).comp (logitRow_differentiable N K m)) z)
        (finProdFinEquiv (n, j)) 0
    rw [show (fun z' : Vec (N * K) => fun _ : Fin 1 =>
          softCE K (smoothTarget K α (targetRow N K t m)) (logitRow N K z' m) / B)
        = fun z' _ => B⁻¹ * softCE K (smoothTarget K α (targetRow N K t m)) (logitRow N K z' m)
        from by funext z' _; rw [div_eq_inv_mul], hc,
      pdiv_comp _ _ z (logitRow_differentiable N K m z) (softCE_differentiable K _ _)]
    have hr : ∀ k, pdiv (fun z' : Vec (N * K) => logitRow N K z' m) z (finProdFinEquiv (n, j)) k
        = if n = m ∧ j = k then 1 else 0 := fun k => by
      rw [show (fun z' : Vec (N * K) => logitRow N K z' m)
          = fun y k => y ((fun k' : Fin K => finProdFinEquiv (m, k')) k) from rfl, pdiv_reindex]
      simp only [Equiv.apply_eq_iff_eq, Prod.mk.injEq]
    simp_rw [hr, ite_mul, one_mul, zero_mul]
    by_cases hmn : m = n
    · subst hmn
      simp only [true_and, Finset.sum_ite_eq, Finset.mem_univ, ite_true, div_eq_inv_mul]
    · simp only [Ne.symm hmn, false_and, ite_false, Finset.sum_const_zero, mul_zero, hmn]
  unfold smoothedBatchLoss
  rw [pdiv_lift_sum _ _ _ (fun m _ => differentiableAt_pi.2 fun _ =>
      lossTerm_differentiableAt N K _ m B z)]
  simp only [hterm, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- **The emitted cotangent is the batched loss's gradient.** Read at the head's `N·K` index
    (`unrowB`), the six-op chain at the logits `rowB z` is `∇ smoothedBatchLoss` at `z`, whenever
    every example's target sums to 1. -/
theorem smoothedBatchLoss_grad (N K : Nat) (hK : 0 < K) (α B : ℝ)
    (aStr negAK bStr logN ohN : String) (t : Vec (N * (1 * K))) (z : Vec (N * K))
    (ht : ∀ n, ∑ k : Fin K, targetRow N K t n k = 1) (J : Fin (N * K)) :
    pdiv (smoothedBatchLoss N K α B t) z J 0
      = unrowB N K (den (smoothedLossCotGraph N K α B aStr negAK bStr logN ohN (rowB N K z) t))
          J := by
  obtain ⟨⟨n, j⟩, rfl⟩ := finProdFinEquiv.surjective J
  have hidx : Fin.cast (congrArg (N * ·) (Nat.one_mul K)).symm (finProdFinEquiv (n, j))
      = finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)) := by
    ext; simp [finProdFinEquiv_apply_val]
  have hrow : Mat.unflatten (batchSlice N (1 * K) (rowB N K z) n) (0 : Fin 1) = logitRow N K z n := by
    funext k
    simp only [Mat.unflatten, batchSlice, rowB, logitRow]
    exact congrArg z (Fin.ext (by simp [finProdFinEquiv_apply_val]))
  rw [smoothedBatchLoss_pdiv, unrowB, hidx,
    smoothedLossCotGraph_row N K hK α B aStr negAK bStr logN ohN _ t n j (ht n), hrow]
  rfl

end Proofs
