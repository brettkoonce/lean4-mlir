module

public import LeanMlir.Proofs.Foundation.SmoothedLossCot
public import LeanMlir.Proofs.Foundation.Batched.BackLinks
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # The batched label-smoothed loss, and its gradient is the emitted cotangent

`SmoothedLossCot` identifies the emitted loss cotangent ROW BY ROW: at example `n` it is
`(1/B)·∂softCE/∂z` at that example's logits. This file states the loss as one function of the
flat batch of logits — `smoothedBatchLoss`, the `1/B`-weighted sum of every example's soft-target
cross-entropy against its smoothed target — and proves its full gradient is the emitted cotangent,
read at the head's `N·K` index (`smoothedBatchLoss_grad`). That is the `hg` a parameter-level
statement asks for (`HasGradAt` at the logits, `ParamGrad`).

## References

- Szegedy et al. 2016, *Rethinking the Inception Architecture for Computer Vision* (§7 label smoothing). <https://arxiv.org/abs/1512.00567>
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

theorem logitRow_differentiable (N K : Nat) (n : Fin N) :
    Differentiable ℝ (fun z : Vec (N * K) => logitRow N K z n) :=
  (reindexCLM (fun k : Fin K => finProdFinEquiv (n, k))).differentiable

/-- **A loss summed over the rows, over a constant, is differentiable** when every row's loss is.
    Shared by every batched loss the renders emit (`smoothedBatchLoss`, `smoothedBatchLossDiv`,
    `bceBatchLoss`). -/
theorem rowSumLoss_div_differentiable (N K : Nat) (c : ℝ) (ℓ : Fin N → Vec K → ℝ)
    (hℓ : ∀ m, Differentiable ℝ (fun r : Vec K => fun _ : Fin 1 => ℓ m r)) :
    Differentiable ℝ (fun z : Vec (N * K) => fun _ : Fin 1 =>
      ∑ m : Fin N, ℓ m (logitRow N K z m) / c) := fun z =>
  differentiableAt_pi.2 fun _ => DifferentiableAt.fun_sum fun m _ => by
    have h : DifferentiableAt ℝ (fun x : Vec (N * K) => ℓ m (logitRow N K x m)) z :=
      differentiableAt_pi.1 (((hℓ m).comp (logitRow_differentiable N K m)) z) 0
    simpa only [div_eq_mul_inv] using h.mul_const c⁻¹

theorem smoothedBatchLoss_differentiable (N K : Nat) (α B : ℝ) (t : Vec (N * (1 * K))) :
    Differentiable ℝ (smoothedBatchLoss N K α B t) :=
  rowSumLoss_div_differentiable N K B _ fun _ => softCE_differentiable K _

/-- **A loss summed over the rows has each row's own gradient**: for `Σ_m ℓ_m(zₘ)`, the partial at
    `(n, j)` is `∂ℓₙ/∂z_j` at example `n`'s logits — no other example contributes. Shared by every
    batched loss the renders emit (`smoothedBatchLoss`, `bceBatchLoss`). -/
theorem rowSumLoss_pdiv (N K : Nat) (ℓ : Fin N → Vec K → ℝ)
    (hℓ : ∀ m, Differentiable ℝ (fun r : Vec K => fun _ : Fin 1 => ℓ m r)) (z : Vec (N * K))
    (n : Fin N) (j : Fin K) :
    pdiv (fun z' : Vec (N * K) => fun _ : Fin 1 => ∑ m : Fin N, ℓ m (logitRow N K z' m)) z
        (finProdFinEquiv (n, j)) 0
      = pdiv (fun r : Vec K => fun _ : Fin 1 => ℓ n r) (logitRow N K z n) j 0 := by
  have hterm : ∀ m : Fin N,
      pdiv (fun z' : Vec (N * K) => fun _ : Fin 1 => ℓ m (logitRow N K z' m))
        z (finProdFinEquiv (n, j)) 0
      = if m = n then pdiv (fun r : Vec K => fun _ : Fin 1 => ℓ m r) (logitRow N K z m) j 0
        else 0 := by
    intro m
    rw [show (fun z' : Vec (N * K) => fun _ : Fin 1 => ℓ m (logitRow N K z' m))
        = (fun r : Vec K => fun _ : Fin 1 => ℓ m r) ∘ fun z' => logitRow N K z' m from rfl,
      pdiv_comp _ _ z (logitRow_differentiable N K m z) (hℓ m _)]
    have hr : ∀ k, pdiv (fun z' : Vec (N * K) => logitRow N K z' m) z (finProdFinEquiv (n, j)) k
        = if n = m ∧ j = k then 1 else 0 := fun k => by
      rw [show (fun z' : Vec (N * K) => logitRow N K z' m)
          = fun y k => y ((fun k' : Fin K => finProdFinEquiv (m, k')) k) from rfl, pdiv_reindex]
      simp only [Equiv.apply_eq_iff_eq, Prod.mk.injEq]
    simp_rw [hr, ite_mul, one_mul, zero_mul]
    by_cases hmn : m = n
    · subst hmn
      simp only [true_and, Finset.sum_ite_eq, Finset.mem_univ, ite_true]
    · simp only [Ne.symm hmn, false_and, ite_false, Finset.sum_const_zero, hmn]
  rw [pdiv_lift_sum Finset.univ (fun m z' => ℓ m (logitRow N K z' m)) z
    (fun m _ => differentiableAt_pi.2 fun _ =>
      differentiableAt_pi.1 (((hℓ m).comp (logitRow_differentiable N K m)) z) 0)]
  simp only [hterm, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- `rowSumLoss_pdiv` with every row's loss scaled by one constant `c` (the batch divisor). -/
theorem rowSumLoss_pdiv_smul (N K : Nat) (c : ℝ) (ℓ : Fin N → Vec K → ℝ)
    (hℓ : ∀ m, Differentiable ℝ (fun r : Vec K => fun _ : Fin 1 => ℓ m r)) (z : Vec (N * K))
    (n : Fin N) (j : Fin K) :
    pdiv (fun z' : Vec (N * K) => fun _ : Fin 1 => ∑ m : Fin N, c * ℓ m (logitRow N K z' m)) z
        (finProdFinEquiv (n, j)) 0
      = c * pdiv (fun r : Vec K => fun _ : Fin 1 => ℓ n r) (logitRow N K z n) j 0 := by
  rw [rowSumLoss_pdiv N K (fun m r => c * ℓ m r) (fun m r => differentiableAt_pi.2 fun _ =>
      (differentiableAt_pi.1 ((hℓ m) r) 0).const_mul c),
    pdiv_const_smul c _ _ ((hℓ n) _)]

/-- `rowSumLoss_pdiv` with every row's loss divided by one constant `c` (the batch divisor): the
    form `smoothedBatchLoss`, `smoothedBatchLossDiv` and `bceBatchLoss` are written in. -/
theorem rowSumLoss_pdiv_div (N K : Nat) (c : ℝ) (ℓ : Fin N → Vec K → ℝ)
    (hℓ : ∀ m, Differentiable ℝ (fun r : Vec K => fun _ : Fin 1 => ℓ m r)) (z : Vec (N * K))
    (n : Fin N) (j : Fin K) :
    pdiv (fun z' : Vec (N * K) => fun _ : Fin 1 => ∑ m : Fin N, ℓ m (logitRow N K z' m) / c) z
        (finProdFinEquiv (n, j)) 0
      = pdiv (fun r : Vec K => fun _ : Fin 1 => ℓ n r) (logitRow N K z n) j 0 / c := by
  simp only [div_eq_inv_mul]
  exact rowSumLoss_pdiv_smul N K c⁻¹ ℓ hℓ z n j

/-- The head's `N·K` index read at the `N·(1·K)` row layout: `(n, j)` is `(n, (0, j))`. -/
theorem rowIdx_cast (N K : Nat) (n : Fin N) (j : Fin K) :
    Fin.cast (congrArg (N * ·) (Nat.one_mul K)).symm (finProdFinEquiv (n, j))
      = finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)) := by
  ext; simp [finProdFinEquiv_apply_val]

/-- Example `n`'s row of `rowB z` is its logits. -/
theorem rowB_row_eq_logitRow (N K : Nat) (z : Vec (N * K)) (n : Fin N) :
    Mat.unflatten (batchSlice N (1 * K) (rowB N K z) n) (0 : Fin 1) = logitRow N K z n := by
  funext k
  simp only [Mat.unflatten, batchSlice, rowB, logitRow]
  exact congrArg z (Fin.ext (by simp [finProdFinEquiv_apply_val]))

/-- **The batched loss's gradient**, entry `(n, j)`: example `n`'s own soft-CE gradient at its
    logits, over `B` — no other example contributes. -/
theorem smoothedBatchLoss_pdiv (N K : Nat) (α B : ℝ) (t : Vec (N * (1 * K))) (z : Vec (N * K))
    (n : Fin N) (j : Fin K) :
    pdiv (smoothedBatchLoss N K α B t) z (finProdFinEquiv (n, j)) 0
      = pdiv (fun z' : Vec K => fun _ : Fin 1 =>
            softCE K (smoothTarget K α (targetRow N K t n)) z') (logitRow N K z n) j 0 / B :=
  rowSumLoss_pdiv_div N K B _ (fun _ => softCE_differentiable K _) z n j

/-- **The emitted cotangent is the batched loss's gradient.** Read at the head's `N·K` index
    (`unrowB`), the six-op chain at the logits `rowB z` is `∇ smoothedBatchLoss` at `z`, whenever
    every example's target sums to 1. -/
theorem smoothedBatchLoss_grad (N K : Nat) (α B : ℝ)
    (aStr negAK bStr logN ohN : String) (t : Vec (N * (1 * K))) (z : Vec (N * K))
    (ht : ∀ n, ∑ k : Fin K, targetRow N K t n k = 1) (J : Fin (N * K)) :
    pdiv (smoothedBatchLoss N K α B t) z J 0
      = unrowB N K (den (smoothedLossCotGraph N K α B aStr negAK bStr logN ohN (rowB N K z) t))
          J := by
  obtain ⟨⟨n, j⟩, rfl⟩ := finProdFinEquiv.surjective J
  rw [smoothedBatchLoss_pdiv, unrowB, rowIdx_cast,
    smoothedLossCotGraph_row N K α B aStr negAK bStr logN ohN _ t n j (ht n),
    rowB_row_eq_logitRow]
  rfl

/-- **The batched label-smoothed loss at the plain `N·K` index**, the target read per example
    with no row index — the loss whose gradient `smoothedLossCotGraphDiv` emits (ConvNeXt, ViT). -/
noncomputable def smoothedBatchLossDiv (N K : Nat) (α B : ℝ) (t : Vec (N * K)) (z : Vec (N * K)) :
    Vec 1 :=
  fun _ => ∑ n : Fin N, softCE K (smoothTarget K α (batchSlice N K t n)) (logitRow N K z n) / B

theorem smoothedBatchLossDiv_differentiable (N K : Nat) (α B : ℝ) (t : Vec (N * K)) :
    Differentiable ℝ (smoothedBatchLossDiv N K α B t) :=
  rowSumLoss_div_differentiable N K B _ fun _ => softCE_differentiable K _

/-- **The emitted `softmaxDiv` cotangent is the batched loss's gradient**, entry by entry, whenever
    every example's target sums to 1 — `smoothedBatchLoss_grad` at the plain `N·K` index. -/
theorem smoothedBatchLossDiv_grad (N K : Nat) (α B : ℝ)
    (aStr negAK bStr logN ohN : String) (t : Vec (N * K)) (z : Vec (N * K))
    (ht : ∀ n, ∑ k : Fin K, batchSlice N K t n k = 1) (J : Fin (N * K)) :
    pdiv (smoothedBatchLossDiv N K α B t) z J 0
      = den (smoothedLossCotGraphDiv N K α B aStr negAK bStr logN ohN z t) J := by
  obtain ⟨⟨n, j⟩, rfl⟩ := finProdFinEquiv.surjective J
  have h : pdiv (smoothedBatchLossDiv N K α B t) z (finProdFinEquiv (n, j)) 0 = _ :=
    rowSumLoss_pdiv_div N K B _ (fun _ => softCE_differentiable K _) z n j
  rw [h, smoothedLossCotGraphDiv_row N K α B aStr negAK bStr logN ohN z t n j (ht n)]
  rfl

end Proofs
