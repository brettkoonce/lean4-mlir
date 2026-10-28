module

public import LeanMlir.Proofs.Nets.Small.LinearFoldB
public import LeanMlir.Proofs.Nets.Small.MlpTrainStep
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # The MNIST-MLP train step at batch `N` — `MlpFold` with the batch stated

`MlpFold` certifies the six parameter outputs of `mlp_train_step.mlir` for ONE example.
[`tests/TestSmallNetsBatchedEmit.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestSmallNetsBatchedEmit.lean) pins that same text as the print of a graph at the batched
index — `batchOp` lifts of `dense`/`relu`/`expe`/`softmaxDiv`/`dotOut`, `subB`, `selectPosB` on
the whole-batch pre-activations, and the `denseWeightSgdB`/`denseBiasSgdB` nodes — so every
node's `den` can be read at a batch of `N` examples. This file does that reading:

* the batched forward's slices are the per-example forward (`preB0_slice`, …);
* the batched loss cotangent's slices are `softmax(logitsₖ) − onehotₖ` (`cotB_slice`);
* the batched chain cotangents' slices are the per-example `mlpCotOut1`/`mlpCotOut0`
  (`cot1B_slice`, `cot0B_slice`);
* each of the six parameter nodes denotes `θ − lr·Σₖ (certified per-layer Jacobian at example k
  · that example's chain cotangent)`, and the output weight `W₂ − lr·Σₖ ∂CE(forwardₖ)/∂W₂`
  (`mlp_train_step_tiedB`). `N` is a binder; the artifact is `N = 128`.

The chain cotangents are node-local, as in `MlpFold`: each parameter node is stated with its
cotangent input as an `.operand` leaf carrying the `den` of the batched backward subgraph, which
the slice lemmas pin. Scope otherwise as `MlpFold` (per-op `pretty` lexing; ℝ → Float32).
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.MlpFoldB

variable {d₀ d₁ d₂ d₃ : Nat} (N : Nat)
  (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
  (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (xB : Vec (N * d₀)) (label : Fin N → Fin d₃)

/-! ## Per-example shorthands, at example `k` of the batch -/

/-- Example `k`'s input. -/
noncomputable abbrev xk (k : Fin N) : Vec d₀ := batchSlice N d₀ xB k
/-- Layer-0 pre-activation at example `k`. -/
noncomputable abbrev p0 (k : Fin N) : Vec d₁ := dense W₀ b₀ (xk N xB k)
/-- Layer-1 pre-activation at example `k`. -/
noncomputable abbrev p1 (k : Fin N) : Vec d₂ := dense W₁ b₁ (relu d₁ (p0 N W₀ b₀ xB k))
/-- The softmax-CE loss cotangent at example `k`: `softmax(logitsₖ) − onehotₖ`. -/
noncomputable abbrev gk (k : Fin N) : Vec d₃ := fun j =>
  softmax d₃ (mnistLinear W₂ b₂ (relu d₂ (p1 N W₀ b₀ W₁ b₁ xB k))) j - oneHot d₃ (label k) j

/-! ## The batched graph, piece by piece (the pieces `mlpTrainStepText` renders) -/

def preB0 : SHlo (N * d₁) := .batchOp (N := N) (.dense "%W0" "%b0" W₀ b₀) (.operand "%x" xB)
def actB0 : SHlo (N * d₁) := .batchOp (N := N) (.relu (n := d₁)) (preB0 N W₀ b₀ xB)
def preB1 : SHlo (N * d₂) := .batchOp (N := N) (.dense "%W1" "%b1" W₁ b₁) (actB0 N W₀ b₀ xB)
def actB1 : SHlo (N * d₂) := .batchOp (N := N) (.relu (n := d₂)) (preB1 N W₀ b₀ W₁ b₁ xB)
def logitsB : SHlo (N * d₃) :=
  .batchOp (N := N) (.dense "%W2" "%b2" W₂ b₂) (actB1 N W₀ b₀ W₁ b₁ xB)
/-- The batched loss cotangent `sub(softmaxDiv(expe(logits)), onehot)`. -/
noncomputable def cotB : SHlo (N * d₃) :=
  .subB (.batchOp (N := N) (.softmaxDiv (n := d₃))
          (.batchOp (N := N) (.expe (n := d₃)) (logitsB N W₀ b₀ W₁ b₁ W₂ b₂ xB)))
        (.operand "%onehot" (LinFoldB.oneHotB N label))

theorem preB0_slice (k : Fin N) : batchSlice N d₁ (den (preB0 N W₀ b₀ xB)) k = p0 N W₀ b₀ xB k := by
  simp only [preB0, den_batchOp, batchSlice_batchMap, denOp]; rfl

theorem actB0_slice (k : Fin N) :
    batchSlice N d₁ (den (actB0 N W₀ b₀ xB)) k = relu d₁ (p0 N W₀ b₀ xB k) := by
  simp only [actB0, den_batchOp, batchSlice_batchMap, denOp, preB0_slice]

theorem preB1_slice (k : Fin N) :
    batchSlice N d₂ (den (preB1 N W₀ b₀ W₁ b₁ xB)) k = p1 N W₀ b₀ W₁ b₁ xB k := by
  simp only [preB1, den_batchOp, batchSlice_batchMap, denOp, actB0_slice]

theorem actB1_slice (k : Fin N) :
    batchSlice N d₂ (den (actB1 N W₀ b₀ W₁ b₁ xB)) k = relu d₂ (p1 N W₀ b₀ W₁ b₁ xB k) := by
  simp only [actB1, den_batchOp, batchSlice_batchMap, denOp, preB1_slice]

theorem logitsB_slice (k : Fin N) :
    batchSlice N d₃ (den (logitsB N W₀ b₀ W₁ b₁ W₂ b₂ xB)) k
      = mnistLinear W₂ b₂ (relu d₂ (p1 N W₀ b₀ W₁ b₁ xB k)) := by
  simp only [logitsB, den_batchOp, batchSlice_batchMap, denOp, actB1_slice]; rfl

/-- Example `k`'s slice of the batched loss cotangent is `softmax(logitsₖ) − onehotₖ`. -/
theorem cotB_slice (k : Fin N) (j : Fin d₃) :
    den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label) (finProdFinEquiv (k, j)) = gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k j := by
  rw [cotB, LinFoldB.den_softmaxCotB_slice, logitsB_slice]

theorem batchSlice_cotB (k : Fin N) :
    batchSlice N d₃ (den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label)) k = gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k := by
  funext j; exact cotB_slice N W₀ b₀ W₁ b₁ W₂ b₂ xB label k j

/-! ## The backward chain: `selectPosB` on the whole-batch pre-activation over a `dotOut` lift -/

/-- Layer-1 batched chain cotangent `selectPosB p₁ (dotOut W₂ dy)`, `dy` the batched loss
    cotangent as an operand leaf. -/
noncomputable def cot1B (p₁N dyN : String) : SHlo (N * d₂) :=
  .selectPosB p₁N (den (preB1 N W₀ b₀ W₁ b₁ xB))
    (.batchOp (N := N) (.dotOut "%W2" W₂) (.operand dyN (den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label))))

/-- Its slice at example `k` is the per-example `mlpCotOut1` at that example's cotangent. -/
theorem cot1B_slice (p₁N dyN : String) (k : Fin N) :
    batchSlice N d₂ (den (cot1B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₁N dyN)) k
      = (mlpCotOut1 W₂ (p1 N W₀ b₀ W₁ b₁ xB k)).denote (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) := by
  rw [mlpCotOut1_denote, cot1B, LinFoldB.batchSlice_selectPosB, LinFoldB.batchSlice_dotOutB,
    LinFoldB.batchSlice_operand, preB1_slice, batchSlice_cotB]

/-- Layer-0 batched chain cotangent `selectPosB p₀ (dotOut W₁ c₁)`, `c₁` the layer-1 chain
    cotangent as an operand leaf. -/
noncomputable def cot0B (p₀N p₁N dyN c₁N : String) : SHlo (N * d₁) :=
  .selectPosB p₀N (den (preB0 N W₀ b₀ xB))
    (.batchOp (N := N) (.dotOut "%W1" W₁)
      (.operand c₁N (den (cot1B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₁N dyN))))

theorem cot0B_slice (p₀N p₁N dyN c₁N : String) (k : Fin N) :
    batchSlice N d₁ (den (cot0B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₀N p₁N dyN c₁N)) k
      = (mlpCotOut0 W₁ W₂ (p0 N W₀ b₀ xB k) (p1 N W₀ b₀ W₁ b₁ xB k)).denote
          (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) := by
  rw [mlpCotOut0_denote, cot0B, LinFoldB.batchSlice_selectPosB, LinFoldB.batchSlice_dotOutB,
    LinFoldB.batchSlice_operand, preB0_slice, cot1B_slice, mlpCotOut1_denote]

/-! ## The six parameter nodes at batch `N` -/

/-- **Output weight, at the WHOLE loss, summed over the batch.** -/
theorem w2B_tied (aN lrStr dyN : String) (lr : ℝ) (i : Fin d₂) (j : Fin d₃) :
    den (SHlo.denseWeightSgdB aN "%W2" lrStr (den (actB1 N W₀ b₀ W₁ b₁ xB)) W₂ lr
          (.operand dyN (den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label)))) (finProdFinEquiv (i, j))
      = W₂ i j - lr * ∑ k : Fin N,
          pdiv (fun (v : Vec (d₂ * d₃)) (_ : Fin 1) =>
                  crossEntropy d₃ (dense (Mat.unflatten v) b₂ (relu d₂ (p1 N W₀ b₀ W₁ b₁ xB k))) (label k))
               (Mat.flatten W₂) (finProdFinEquiv (i, j)) 0 := by
  rw [LinFoldB.denseWeightSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [lossWeightGrad_eq_sum, LinFoldB.dense_jacW_collapse]
  show batchSlice N d₂ (den (actB1 N W₀ b₀ W₁ b₁ xB)) k i
      * batchSlice N d₃ (den (.operand dyN (den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label)))) k j = _
  rw [actB1_slice]
  show _ * batchSlice N d₃ (den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label)) k j = _
  rw [batchSlice_cotB]

/-- Output bias. -/
theorem b2B_tied (lrStr dyN : String) (lr : ℝ) (j : Fin d₃) :
    den (SHlo.denseBiasSgdB "%b2" lrStr b₂ lr
          (.operand dyN (den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label)))) j
      = b₂ j - lr * ∑ k : Fin N, ∑ i : Fin d₃,
          pdiv (fun b' : Vec d₃ => dense W₂ b' (relu d₂ (p1 N W₀ b₀ W₁ b₁ xB k))) b₂ j i
            * gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k i := by
  rw [LinFoldB.denseBiasSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [LinFoldB.dense_jacB_collapse]
  show batchSlice N d₃ (den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label)) k j = _
  rw [batchSlice_cotB]

/-- Hidden weight `W₁`, at the layer-1 chain cotangent. -/
theorem w1B_tied (aN lrStr p₁N dyN c₁N : String) (lr : ℝ) (i : Fin d₁) (j : Fin d₂) :
    den (SHlo.denseWeightSgdB aN "%W1" lrStr (den (actB0 N W₀ b₀ xB)) W₁ lr
          (.operand c₁N (den (cot1B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₁N dyN)))) (finProdFinEquiv (i, j))
      = W₁ i j - lr * ∑ k : Fin N, ∑ k' : Fin d₂,
          pdiv (fun v : Vec (d₁ * d₂) => dense (Mat.unflatten v) b₁ (relu d₁ (p0 N W₀ b₀ xB k)))
               (Mat.flatten W₁) (finProdFinEquiv (i, j)) k'
            * (mlpCotOut1 W₂ (p1 N W₀ b₀ W₁ b₁ xB k)).denote (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) k' := by
  rw [LinFoldB.denseWeightSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [LinFoldB.dense_jacW_collapse]
  show batchSlice N d₁ (den (actB0 N W₀ b₀ xB)) k i
      * batchSlice N d₂ (den (cot1B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₁N dyN)) k j = _
  rw [actB0_slice, cot1B_slice]

/-- Hidden bias `b₁`. -/
theorem b1B_tied (lrStr p₁N dyN c₁N : String) (lr : ℝ) (j : Fin d₂) :
    den (SHlo.denseBiasSgdB "%b1" lrStr b₁ lr
          (.operand c₁N (den (cot1B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₁N dyN)))) j
      = b₁ j - lr * ∑ k : Fin N, ∑ i : Fin d₂,
          pdiv (fun b' : Vec d₂ => dense W₁ b' (relu d₁ (p0 N W₀ b₀ xB k))) b₁ j i
            * (mlpCotOut1 W₂ (p1 N W₀ b₀ W₁ b₁ xB k)).denote (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) i := by
  rw [LinFoldB.denseBiasSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [LinFoldB.dense_jacB_collapse]
  show batchSlice N d₂ (den (cot1B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₁N dyN)) k j = _
  rw [cot1B_slice]

/-- Input weight `W₀`, at the layer-0 chain cotangent. -/
theorem w0B_tied (lrStr p₀N p₁N dyN c₁N c₀N : String) (lr : ℝ) (i : Fin d₀) (j : Fin d₁) :
    den (SHlo.denseWeightSgdB "%x" "%W0" lrStr xB W₀ lr
          (.operand c₀N (den (cot0B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₀N p₁N dyN c₁N)))) (finProdFinEquiv (i, j))
      = W₀ i j - lr * ∑ k : Fin N, ∑ k' : Fin d₁,
          pdiv (fun v : Vec (d₀ * d₁) => dense (Mat.unflatten v) b₀ (xk N xB k))
               (Mat.flatten W₀) (finProdFinEquiv (i, j)) k'
            * (mlpCotOut0 W₁ W₂ (p0 N W₀ b₀ xB k) (p1 N W₀ b₀ W₁ b₁ xB k)).denote
                (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) k' := by
  rw [LinFoldB.denseWeightSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [LinFoldB.dense_jacW_collapse]
  show batchSlice N d₀ xB k i
      * batchSlice N d₁ (den (cot0B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₀N p₁N dyN c₁N)) k j = _
  rw [cot0B_slice]

/-- Input bias `b₀`. -/
theorem b0B_tied (lrStr p₀N p₁N dyN c₁N c₀N : String) (lr : ℝ) (j : Fin d₁) :
    den (SHlo.denseBiasSgdB "%b0" lrStr b₀ lr
          (.operand c₀N (den (cot0B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₀N p₁N dyN c₁N)))) j
      = b₀ j - lr * ∑ k : Fin N, ∑ i : Fin d₁,
          pdiv (fun b' : Vec d₁ => dense W₀ b' (xk N xB k)) b₀ j i
            * (mlpCotOut0 W₁ W₂ (p0 N W₀ b₀ xB k) (p1 N W₀ b₀ W₁ b₁ xB k)).denote
                (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) i := by
  rw [LinFoldB.denseBiasSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [LinFoldB.dense_jacB_collapse]
  show batchSlice N d₁ (den (cot0B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₀N p₁N dyN c₁N)) k j = _
  rw [cot0B_slice]

/-- **Whole MLP train step at batch `N`, tied.** The output weight denotes `W₂ − lr·Σₖ ∂CE/∂W₂`
    at the whole loss of each example; the other five denote `θ − lr·Σₖ (certified per-layer
    Jacobian at example k · that example's rendered chain cotangent)`, the chain cotangents being
    the slices of the emitted `selectPosB`/`dotOut` subgraphs (`cot1B_slice`, `cot0B_slice`). -/
theorem mlp_train_step_tiedB (lrStr aN a₀N p₀N p₁N dyN c₁N c₀N : String) (lr : ℝ) :
    (∀ i j, den (SHlo.denseWeightSgdB aN "%W2" lrStr (den (actB1 N W₀ b₀ W₁ b₁ xB)) W₂ lr
              (.operand dyN (den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label)))) (finProdFinEquiv (i, j))
          = W₂ i j - lr * ∑ k : Fin N,
              pdiv (fun (v : Vec (d₂ * d₃)) (_ : Fin 1) =>
                      crossEntropy d₃ (dense (Mat.unflatten v) b₂ (relu d₂ (p1 N W₀ b₀ W₁ b₁ xB k))) (label k))
                   (Mat.flatten W₂) (finProdFinEquiv (i, j)) 0)
  ∧ (∀ j, den (SHlo.denseBiasSgdB "%b2" lrStr b₂ lr
              (.operand dyN (den (cotB N W₀ b₀ W₁ b₁ W₂ b₂ xB label)))) j
          = b₂ j - lr * ∑ k : Fin N, ∑ i : Fin d₃,
              pdiv (fun b' : Vec d₃ => dense W₂ b' (relu d₂ (p1 N W₀ b₀ W₁ b₁ xB k))) b₂ j i
                * gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k i)
  ∧ (∀ i j, den (SHlo.denseWeightSgdB a₀N "%W1" lrStr (den (actB0 N W₀ b₀ xB)) W₁ lr
              (.operand c₁N (den (cot1B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₁N dyN)))) (finProdFinEquiv (i, j))
          = W₁ i j - lr * ∑ k : Fin N, ∑ k' : Fin d₂,
              pdiv (fun v : Vec (d₁ * d₂) => dense (Mat.unflatten v) b₁ (relu d₁ (p0 N W₀ b₀ xB k)))
                   (Mat.flatten W₁) (finProdFinEquiv (i, j)) k'
                * (mlpCotOut1 W₂ (p1 N W₀ b₀ W₁ b₁ xB k)).denote (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) k')
  ∧ (∀ j, den (SHlo.denseBiasSgdB "%b1" lrStr b₁ lr
              (.operand c₁N (den (cot1B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₁N dyN)))) j
          = b₁ j - lr * ∑ k : Fin N, ∑ i : Fin d₂,
              pdiv (fun b' : Vec d₂ => dense W₁ b' (relu d₁ (p0 N W₀ b₀ xB k))) b₁ j i
                * (mlpCotOut1 W₂ (p1 N W₀ b₀ W₁ b₁ xB k)).denote (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) i)
  ∧ (∀ i j, den (SHlo.denseWeightSgdB "%x" "%W0" lrStr xB W₀ lr
              (.operand c₀N (den (cot0B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₀N p₁N dyN c₁N)))) (finProdFinEquiv (i, j))
          = W₀ i j - lr * ∑ k : Fin N, ∑ k' : Fin d₁,
              pdiv (fun v : Vec (d₀ * d₁) => dense (Mat.unflatten v) b₀ (xk N xB k))
                   (Mat.flatten W₀) (finProdFinEquiv (i, j)) k'
                * (mlpCotOut0 W₁ W₂ (p0 N W₀ b₀ xB k) (p1 N W₀ b₀ W₁ b₁ xB k)).denote
                    (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) k')
  ∧ (∀ j, den (SHlo.denseBiasSgdB "%b0" lrStr b₀ lr
              (.operand c₀N (den (cot0B N W₀ b₀ W₁ b₁ W₂ b₂ xB label p₀N p₁N dyN c₁N)))) j
          = b₀ j - lr * ∑ k : Fin N, ∑ i : Fin d₁,
              pdiv (fun b' : Vec d₁ => dense W₀ b' (xk N xB k)) b₀ j i
                * (mlpCotOut0 W₁ W₂ (p0 N W₀ b₀ xB k) (p1 N W₀ b₀ W₁ b₁ xB k)).denote
                    (gk N W₀ b₀ W₁ b₁ W₂ b₂ xB label k) i) :=
  ⟨fun i j => w2B_tied N W₀ b₀ W₁ b₁ W₂ b₂ xB label aN lrStr dyN lr i j,
   fun j => b2B_tied N W₀ b₀ W₁ b₁ W₂ b₂ xB label lrStr dyN lr j,
   fun i j => w1B_tied N W₀ b₀ W₁ b₁ W₂ b₂ xB label a₀N lrStr p₁N dyN c₁N lr i j,
   fun j => b1B_tied N W₀ b₀ W₁ b₁ W₂ b₂ xB label lrStr p₁N dyN c₁N lr j,
   fun i j => w0B_tied N W₀ b₀ W₁ b₁ W₂ b₂ xB label lrStr p₀N p₁N dyN c₁N c₀N lr i j,
   fun j => b0B_tied N W₀ b₀ W₁ b₁ W₂ b₂ xB label lrStr p₀N p₁N dyN c₁N c₀N lr j⟩

end Proofs.MlpFoldB
