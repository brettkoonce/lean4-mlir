import LeanMlir.Proofs.Foundation.SmoothedBatchLoss
import LeanMlir.Proofs.Foundation.BceLossCot

/-! # The batched BCE-with-logits loss, and its gradient is the emitted cotangent

`BceLossCot` identifies the `bce := true` renders' three-op cotangent ROW BY ROW: at example `n`
it is `∂bceLogits/∂z` at that example's logits over the baked `N·K`. This file states the loss as
one function of the flat batch of logits — `bceBatchLoss`, every example's class-summed
BCE-with-logits over `N·K`, i.e. the mean over `B×K` — and proves its full gradient is the emitted
cotangent, read at the head's `N·K` index (`bceBatchLoss_grad`). It is `SmoothedBatchLoss`'s twin
for ResNet-50's BCE artifacts, and like `bceLossCotGraph_row` it needs no hypothesis on the target.
-/

namespace Proofs

open scoped BigOperators
open StableHLO
open BackLinks (rowB unrowB)

/-- **The batched BCE-with-logits loss**: `Σ_n bceLogits(tₙ, zₙ) / (N·K)`, as a function of the
    flat logits — the mean over `B×K` that timm's `BinaryCrossEntropy` computes. -/
noncomputable def bceBatchLoss (N K : Nat) (t : Vec (N * (1 * K))) (z : Vec (N * K)) : Vec 1 :=
  fun _ => ∑ n : Fin N, bceLogits K (targetRow N K t n) (logitRow N K z n) / ((N : ℝ) * (K : ℝ))

private theorem bceLogits_differentiable (K : Nat) (t : Vec K) :
    Differentiable ℝ (fun z : Vec K => fun _ : Fin 1 => bceLogits K t z) := fun z =>
  differentiableAt_pi.2 fun _ => by
    unfold bceLogits softplus
    fun_prop (disch := intro; positivity)

theorem bceBatchLoss_differentiable (N K : Nat) (t : Vec (N * (1 * K))) :
    Differentiable ℝ (bceBatchLoss N K t) := fun z =>
  differentiableAt_pi.2 fun _ => DifferentiableAt.fun_sum fun n _ => by
    have h : DifferentiableAt ℝ
        (fun x : Vec (N * K) => bceLogits K (targetRow N K t n) (logitRow N K x n)) z :=
      differentiableAt_pi.1 (((bceLogits_differentiable K (targetRow N K t n)).comp
        (logitRow_differentiable N K n)) z) 0
    simpa only [div_eq_mul_inv] using h.mul_const ((N : ℝ) * (K : ℝ))⁻¹

/-- **The batched BCE loss's gradient**, entry `(n, j)`: example `n`'s own BCE gradient at its
    logits, over `N·K`. -/
theorem bceBatchLoss_pdiv (N K : Nat) (t : Vec (N * (1 * K))) (z : Vec (N * K)) (n : Fin N)
    (j : Fin K) :
    pdiv (bceBatchLoss N K t) z (finProdFinEquiv (n, j)) 0
      = pdiv (fun z' : Vec K => fun _ : Fin 1 => bceLogits K (targetRow N K t n) z')
          (logitRow N K z n) j 0 / ((N : ℝ) * (K : ℝ)) := by
  have hℓ : ∀ m : Fin N, Differentiable ℝ (fun r : Vec K => fun _ : Fin 1 =>
      ((N : ℝ) * (K : ℝ))⁻¹ * bceLogits K (targetRow N K t m) r) := fun m r =>
    differentiableAt_pi.2 fun _ =>
      (differentiableAt_pi.1 ((bceLogits_differentiable K _) r) 0).const_mul _
  rw [show bceBatchLoss N K t = fun z' _ => ∑ m : Fin N,
      ((N : ℝ) * (K : ℝ))⁻¹ * bceLogits K (targetRow N K t m) (logitRow N K z' m) from by
    funext z' _; simp only [bceBatchLoss, div_eq_inv_mul],
    rowSumLoss_pdiv N K _ hℓ,
    pdiv_const_smul _ _ _ ((bceLogits_differentiable K _) _), div_eq_inv_mul]

/-- **The emitted BCE cotangent is the batched loss's gradient.** Read at the head's `N·K` index
    (`unrowB`), the three-op chain at the logits `rowB z`, with the committed divisor `N·K`, is
    `∇ bceBatchLoss` at `z` — for every target. -/
theorem bceBatchLoss_grad (N K : Nat) (bStr logN ohN : String) (t : Vec (N * (1 * K)))
    (z : Vec (N * K)) (J : Fin (N * K)) :
    pdiv (bceBatchLoss N K t) z J 0
      = unrowB N K (den (bceLossCotGraph N K ((N : ℝ) * (K : ℝ)) bStr logN ohN (rowB N K z) t))
          J := by
  obtain ⟨⟨n, j⟩, rfl⟩ := finProdFinEquiv.surjective J
  have hidx : Fin.cast (congrArg (N * ·) (Nat.one_mul K)).symm (finProdFinEquiv (n, j))
      = finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)) := by
    ext; simp [finProdFinEquiv_apply_val]
  have hrow : Mat.unflatten (batchSlice N (1 * K) (rowB N K z) n) (0 : Fin 1) = logitRow N K z n := by
    funext k
    simp only [Mat.unflatten, batchSlice, rowB, logitRow]
    exact congrArg z (Fin.ext (by simp [finProdFinEquiv_apply_val]))
  rw [bceBatchLoss_pdiv, unrowB, hidx, bceLossCotGraph_row_committed, hrow]
  rfl

end Proofs
