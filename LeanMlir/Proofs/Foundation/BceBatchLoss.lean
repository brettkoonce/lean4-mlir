module

public import LeanMlir.Proofs.Foundation.SmoothedBatchLoss
public import LeanMlir.Proofs.Foundation.BceLossCot
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

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
    Differentiable ℝ (bceBatchLoss N K t) :=
  rowSumLoss_div_differentiable N K _ _ fun _ => bceLogits_differentiable K _

/-- **The batched BCE loss's gradient**, entry `(n, j)`: example `n`'s own BCE gradient at its
    logits, over `N·K`. -/
theorem bceBatchLoss_pdiv (N K : Nat) (t : Vec (N * (1 * K))) (z : Vec (N * K)) (n : Fin N)
    (j : Fin K) :
    pdiv (bceBatchLoss N K t) z (finProdFinEquiv (n, j)) 0
      = pdiv (fun z' : Vec K => fun _ : Fin 1 => bceLogits K (targetRow N K t n) z')
          (logitRow N K z n) j 0 / ((N : ℝ) * (K : ℝ)) :=
  rowSumLoss_pdiv_div N K _ _ (fun _ => bceLogits_differentiable K _) z n j

/-- **The emitted BCE cotangent is the batched loss's gradient.** Read at the head's `N·K` index
    (`unrowB`), the three-op chain at the logits `rowB z`, with the committed divisor `N·K`, is
    `∇ bceBatchLoss` at `z` — for every target. -/
theorem bceBatchLoss_grad (N K : Nat) (bStr logN ohN : String) (t : Vec (N * (1 * K)))
    (z : Vec (N * K)) (J : Fin (N * K)) :
    pdiv (bceBatchLoss N K t) z J 0
      = unrowB N K (den (bceLossCotGraph N K ((N : ℝ) * (K : ℝ)) bStr logN ohN (rowB N K z) t))
          J := by
  obtain ⟨⟨n, j⟩, rfl⟩ := finProdFinEquiv.surjective J
  rw [bceBatchLoss_pdiv, unrowB, rowIdx_cast, bceLossCotGraph_row_committed, rowB_row_eq_logitRow]
  rfl

end Proofs
