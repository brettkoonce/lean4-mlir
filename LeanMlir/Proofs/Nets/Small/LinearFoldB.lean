module

public import LeanMlir.Proofs.Codegen.StableHLO.Basic
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Nets.Small.LinearTrainStep
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # The MNIST-linear train step at batch `N` — the fold `LinearFold` leaves open

`LinearFold` certifies `linear_train_step.mlir`'s two outputs for ONE example: its nodes
(`weightSgd`/`biasSgd` over `lossCotGraph`) denote per-example values, and the file says so
("the committed module batch-contracts over `B` examples, which this file does not state").
This file states it. The same text is the print of a graph at the batched index — `batchOp`
lifts of `dense`/`expe`/`softmaxDiv`, `subB`, and the `denseWeightSgdB`/`denseBiasSgdB` nodes
whose `den` sums the per-example gradient over `batchSlice n` — and
[`tests/TestSmallNetsBatchedEmit.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestSmallNetsBatchedEmit.lean) pins that print byte-for-byte to `linTrainStepText` at the
committed shape. So:

* `lin_train_step_tiedB` — at a batch of `N` examples with labels `label : Fin N → Fin n`, the
  weight output denotes `W − lr·Σₖ ∂CE(dense(W,b)·xₖ, labelₖ)/∂W` and the bias output
  `b − lr·Σₖ (certified ∂dense/∂b · (softmax(zₖ) − onehotₖ))`, for every `N`.
* `lin_weight_meanlossB` — at `lr = α/N` the weight output is one step of gradient descent at
  rate `α` on the batch-MEAN loss, which is the step the trainer takes (`0.00078125 = 0.1/128`).

## Scope
* per-op `den` ⇄ MLIR text, as in `LinearFold`; the committed-bytes tie is CI's regenerate-and-diff
  plus the batched-emit pin; ℝ → Float32 is `FloatBridge`.
-/

open Proofs Proofs.StableHLO

namespace Proofs.LinFoldB

/-! ## Two node lemmas every batched dense fold uses

`denseWeightSgdB`/`denseBiasSgdB` denote `θ − lr·Σₖ (per-example gradient at slice k)` by
definition; these read that off at an index. The two collapse lemmas turn the certified dense
Jacobian contracted with a cotangent into the closed form the node computes. -/

theorem denseWeightSgdB_den_slices {N a c : Nat} (xN wN lrS : String) (x : Vec (N * a))
    (W : Mat a c) (lr : ℝ) (e : SHlo (N * c)) (i : Fin a) (j : Fin c) :
    den (SHlo.denseWeightSgdB xN wN lrS x W lr e) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ k : Fin N, batchSlice N a x k i * batchSlice N c (den e) k j := by
  show Mat.flatten (fun i j => W i j - lr * ∑ k : Fin N,
    batchSlice N a x k i * batchSlice N c (den e) k j) (finProdFinEquiv (i, j)) = _
  simp only [Mat.flatten, Equiv.symm_apply_apply]

theorem denseBiasSgdB_den_slices {N c : Nat} (bN lrS : String) (b : Vec c) (lr : ℝ)
    (e : SHlo (N * c)) (j : Fin c) :
    den (SHlo.denseBiasSgdB bN lrS b lr e) j
      = b j - lr * ∑ k : Fin N, batchSlice N c (den e) k j := rfl

/-! ## Slice lemmas for the batched backward-chain nodes -/

/-- A `selectPosB` mask reads its own example's saved pre-activation. -/
theorem batchSlice_selectPosB {N n : Nat} (xN : String) (x : Vec (N * n)) (e : SHlo (N * n))
    (k : Fin N) :
    batchSlice N n (den (SHlo.selectPosB xN x e)) k
      = fun i => if batchSlice N n x k i > 0 then batchSlice N n (den e) k i else 0 := rfl

/-- An operand leaf's slice is the slice of the value it carries. -/
theorem batchSlice_operand {N n : Nat} (nm : String) (v : Vec (N * n)) (k : Fin N) :
    batchSlice N n (den (SHlo.operand nm v)) k = batchSlice N n v k := rfl

/-- A `dotOut` lift's slice is `W · (slice of the input)`. -/
theorem batchSlice_dotOutB {N m n : Nat} (wN : String) (W : Mat m n) (e : SHlo (N * n)) (k : Fin N) :
    batchSlice N m (den (SHlo.batchOp (N := N) (.dotOut wN W) e)) k
      = Mat.mulVec W (batchSlice N n (den e) k) := by
  rw [den_batchOp, batchSlice_batchMap]; rfl

/-- The certified dense weight Jacobian contracted with a cotangent is `x i · c j`. -/
theorem dense_jacW_collapse {a c : Nat} (W : Mat a c) (b : Vec c) (x : Vec a) (cot : Vec c)
    (i : Fin a) (j : Fin c) :
    ∑ k : Fin c, pdiv (fun v : Vec (a * c) => dense (Mat.unflatten v) b x)
        (Mat.flatten W) (finProdFinEquiv (i, j)) k * cot k = x i * cot j := by
  simp only [pdiv_dense_W, ite_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

/-- The certified dense bias Jacobian contracted with a cotangent is `c j`. -/
theorem dense_jacB_collapse {a c : Nat} (W : Mat a c) (b : Vec c) (x : Vec a) (cot : Vec c)
    (j : Fin c) :
    ∑ i : Fin c, pdiv (fun b' : Vec c => dense W b' x) b j i * cot i = cot j := by
  simp only [pdiv_dense_b, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ, ite_true]

variable {m n : Nat} (N : Nat) (W : Mat m n) (b : Vec n) (xB : Vec (N * m))
  (label : Fin N → Fin n)

/-- The one-hot batch `%onehot` carries: row `k` is `oneHot n (label k)`. -/
noncomputable def oneHotB : Vec (N * n) :=
  fun idx => oneHot n (label (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2

theorem batchSlice_oneHotB (k : Fin N) :
    batchSlice N n (oneHotB N label) k = oneHot n (label k) := by
  funext j; simp [batchSlice, oneHotB]

/-- Example `k`'s slice of a batched softmax-CE cotangent `softmax(e) − onehot`, whatever graph `e`
    computes the logits: the per-example `softmax − oneHot` at `e`'s slice. Each batched fold's
    `cotB_slice` is this plus its own logits slice. -/
theorem den_softmaxCotB_slice (e : SHlo (N * n)) (k : Fin N) (j : Fin n) :
    den (.subB (.batchOp (N := N) (.softmaxDiv (n := n)) (.batchOp (N := N) (.expe (n := n)) e))
          (.operand "%onehot" (oneHotB N label))) (finProdFinEquiv (k, j))
      = softmax n (batchSlice N n (den e) k) j - oneHot n (label k) j := by
  have h1 : den (.subB (.batchOp (N := N) (.softmaxDiv (n := n))
                (.batchOp (N := N) (.expe (n := n)) e))
              (.operand "%onehot" (oneHotB N label))) (finProdFinEquiv (k, j))
      = den (.batchOp (N := N) (.softmaxDiv (n := n)) (.batchOp (N := N) (.expe (n := n)) e))
          (finProdFinEquiv (k, j))
        - oneHotB N label (finProdFinEquiv (k, j)) := rfl
  have h2 : ∀ i : Fin n,
      batchSlice N n (den (.batchOp (N := N) (.expe (n := n)) e)) k i
        = Real.exp (batchSlice N n (den e) k i) := by
    intro i
    simp only [den_batchOp, batchSlice_batchMap, denOp]
  rw [h1, den_batchOp_softmaxDiv_per_example]
  simp only [h2, oneHotB, Equiv.symm_apply_apply, softmax_apply]

/-- The batched forward: `dense W b` lifted across the batch (`%x : [N, m]`). -/
def fwdGraphB : SHlo (N * n) :=
  .batchOp (N := N) (.dense "%W0" "%b0" W b) (.operand "%x" xB)

/-- The batched loss cotangent `softmax(logits) − onehot`, every op a batched node. -/
noncomputable def lossCotGraphB : SHlo (N * n) :=
  .subB (.batchOp (N := N) (.softmaxDiv (n := n))
          (.batchOp (N := N) (.expe (n := n)) (fwdGraphB N W b xB)))
        (.operand "%onehot" (oneHotB N label))

/-- Example `k`'s slice of the batched cotangent is the per-example `softmax − oneHot`. -/
theorem lossCotB_slice (k : Fin N) (j : Fin n) :
    den (lossCotGraphB N W b xB label) (finProdFinEquiv (k, j))
      = softmax n (mnistLinear W b (batchSlice N m xB k)) j - oneHot n (label k) j := by
  have hz : batchSlice N n (den (fwdGraphB N W b xB)) k = mnistLinear W b (batchSlice N m xB k) := by
    simp only [fwdGraphB, den_batchOp, batchSlice_batchMap, denOp]
    rfl
  rw [lossCotGraphB, den_softmaxCotB_slice, hz]

/-- The slice form of `lossCotB_slice`. -/
theorem batchSlice_lossCotB (k : Fin N) :
    batchSlice N n (den (lossCotGraphB N W b xB label)) k
      = fun j => softmax n (mnistLinear W b (batchSlice N m xB k)) j - oneHot n (label k) j := by
  funext j; exact lossCotB_slice N W b xB label k j

/-- **Weight output at batch `N`.** The emitted `denseWeightSgdB` node over the batched cotangent
    denotes `W − lr·Σₖ ∂(crossEntropy ∘ dense)/∂W` at example `k` — the sum over the batch of the
    per-example loss gradient `LinearFold` proves at one example. -/
theorem weightSgdB_tied (lrStr : String) (lr : ℝ) (i : Fin m) (j : Fin n) :
    den (SHlo.denseWeightSgdB "%x" "%W0" lrStr xB W lr (lossCotGraphB N W b xB label))
        (finProdFinEquiv (i, j))
      = W i j - lr * ∑ k : Fin N,
          pdiv (fun (v : Vec (m * n)) (_ : Fin 1) =>
                  crossEntropy n (dense (Mat.unflatten v) b (batchSlice N m xB k)) (label k))
               (Mat.flatten W) (finProdFinEquiv (i, j)) 0 := by
  rw [denseWeightSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [lossWeightGrad_eq_sum, batchSlice_lossCotB, dense_jacW_collapse]

/-- **Bias output at batch `N`.** The emitted `denseBiasSgdB` node denotes
    `b − lr·Σₖ (certified ∂dense/∂b · (softmax(zₖ) − onehotₖ))`. -/
theorem biasSgdB_tied (lrStr : String) (lr : ℝ) (j : Fin n) :
    den (SHlo.denseBiasSgdB "%b0" lrStr b lr (lossCotGraphB N W b xB label)) j
      = b j - lr * ∑ k : Fin N, ∑ i : Fin n,
          pdiv (fun b' : Vec n => dense W b' (batchSlice N m xB k)) b j i
            * (softmax n (mnistLinear W b (batchSlice N m xB k)) i - oneHot n (label k) i) := by
  rw [denseBiasSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [batchSlice_lossCotB, dense_jacB_collapse]

/-- **The linear train step at batch `N`, tied.** Both outputs of `linear_train_step.mlir`, read as
    the batched graph it is the print of, denote the certified SGD step on the sum of the `N`
    examples' losses: the weight against the loss gradient, the bias against the certified bias
    Jacobian contracted with `softmax − onehot`. `N` is a binder; the artifact is `N = 128`. -/
theorem lin_train_step_tiedB (lrStr : String) (lr : ℝ) :
    (∀ (i : Fin m) (j : Fin n),
        den (SHlo.denseWeightSgdB "%x" "%W0" lrStr xB W lr (lossCotGraphB N W b xB label))
            (finProdFinEquiv (i, j))
          = W i j - lr * ∑ k : Fin N,
              pdiv (fun (v : Vec (m * n)) (_ : Fin 1) =>
                      crossEntropy n (dense (Mat.unflatten v) b (batchSlice N m xB k)) (label k))
                   (Mat.flatten W) (finProdFinEquiv (i, j)) 0)
  ∧ (∀ j : Fin n,
        den (SHlo.denseBiasSgdB "%b0" lrStr b lr (lossCotGraphB N W b xB label)) j
          = b j - lr * ∑ k : Fin N, ∑ i : Fin n,
              pdiv (fun b' : Vec n => dense W b' (batchSlice N m xB k)) b j i
                * (softmax n (mnistLinear W b (batchSlice N m xB k)) i - oneHot n (label k) i)) :=
  ⟨weightSgdB_tied N W b xB label lrStr lr, biasSgdB_tied N W b xB label lrStr lr⟩

/-- **The mean-loss reading.** At `lr = α / N` (the artifact's `0.00078125 = 0.1 / 128`) the weight
    output is one gradient-descent step at rate `α` on the batch-MEAN cross-entropy. -/
theorem lin_weight_meanlossB (lrStr : String) (α : ℝ) (hN : 0 < N) (i : Fin m) (j : Fin n) :
    den (SHlo.denseWeightSgdB "%x" "%W0" lrStr xB W (α / N) (lossCotGraphB N W b xB label))
        (finProdFinEquiv (i, j))
      = W i j - α *
          pdiv (fun (v : Vec (m * n)) (_ : Fin 1) =>
                  (1 / (N : ℝ)) * ∑ k : Fin N,
                    crossEntropy n (dense (Mat.unflatten v) b (batchSlice N m xB k)) (label k))
               (Mat.flatten W) (finProdFinEquiv (i, j)) 0 := by
  have hdiff : DifferentiableAt ℝ
      (fun (v : Vec (m * n)) (_ : Fin 1) => ∑ k : Fin N,
        crossEntropy n (dense (Mat.unflatten v) b (batchSlice N m xB k)) (label k))
      (Mat.flatten W) := by
    have : (fun (v : Vec (m * n)) (_ : Fin 1) => ∑ k : Fin N,
        crossEntropy n (dense (Mat.unflatten v) b (batchSlice N m xB k)) (label k))
        = ∑ k : Fin N, (fun (v : Vec (m * n)) (_ : Fin 1) =>
            crossEntropy n (dense (Mat.unflatten v) b (batchSlice N m xB k)) (label k)) := by
      funext v _; simp [Finset.sum_apply]
    rw [this]
    exact (Differentiable.sum fun k _ =>
      lossWeightMap_differentiable b (batchSlice N m xB k) (label k)) _
  rw [weightSgdB_tied, pdiv_const_smul _ _ _ hdiff, pdiv_lift_sum _ _ _
    (fun k _ => (lossWeightMap_differentiable b (batchSlice N m xB k) (label k)) _)]
  have hNne : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  field_simp

end Proofs.LinFoldB
