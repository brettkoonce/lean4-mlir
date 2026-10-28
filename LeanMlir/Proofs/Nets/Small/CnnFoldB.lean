module

public import LeanMlir.Proofs.Nets.Small.CnnChainClose
public import LeanMlir.Proofs.Nets.Small.LinearFoldB
public import LeanMlir.Proofs.Nets.Small.MlpTrainStep
import LeanMlir.Proofs.Foundation.GradNodesB
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # The MNIST-CNN train step at batch `N` — `CnnFold` with the batch stated

`CnnFold` certifies the ten parameter outputs of `cnn_train_step.mlir` for ONE example.
[`tests/TestSmallNetsBatchedEmit.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestSmallNetsBatchedEmit.lean) pins that text, byte for byte, as the print of a graph at
the batched index: `batchOp` lifts of `conv`/`relu`/`maxPool`/`dense`/`expe`/`softmaxDiv`/
`dotOut`, `subB`, `selectPosB`, `maxPoolBackB`, the `convBackP` descriptor (the per-example
conv input-VJP spelling, at the batched index) and the `convWeightSgdB`/`convBiasSgdB`/
`denseWeightSgdB`/`denseBiasSgdB` nodes. This file reads every node's `den` at a batch of `N`:

* the batched forward's slices are the per-example forward (`hc1B_slice`, …, `logitsB_slice`);
* the batched loss cotangent's slices are `softmax(logitsₖ) − onehotₖ` (`cotB_slice`);
* the batched backward chain's slices are the per-example chain — the dense head's
  `mlpCotOut1`/`mlpCotOut0`, then `CnnChainClose`'s `cnnChainCotW2` (relu mask over the
  select-and-scatter pool-back of the head cotangent) and, for odd kernels, `cnnChainCotW1`
  (relu mask over the reversed-kernel conv-back) — `dhc2B_slice`, `dhc1B_slice`;
* each of the ten parameter nodes denotes `θ − lr·Σₖ (certified per-layer Jacobian at example k
  · that example's chain cotangent)`, the output weight at the whole loss
  (`cnn_train_step_tiedB`). `N` is a binder; the artifact is `N = 128`.

Unlike `CnnFold`, the chain cotangents here ARE the emitted subgraphs' denotations (the pool-back
and conv-back are `maxPoolBackB`/`convBackP` nodes, not the hand-named `cnnChainCot*` values), so
the "cotangent SSA ↔ chain-cot per-op trust" `CnnFold`'s Scope records is closed at the batched
index: `dhc2B_slice`/`dhc1B_slice` prove the two readings agree.

## Scope
* per-op `den` ⇄ MLIR text (shared with the suite); the committed-bytes tie is CI's
  regenerate-and-diff plus the batched-emit pin; ℝ → Float32 is `FloatBridge`.
* `dhc1B_slice`'s identification with `cnnChainCotW1` needs odd kernels (`hkH`, `hkW`): the
  reversed-kernel conv is the conv input-VJP only there (`IR.conv_back_bridge`). The node
  theorems themselves carry no such hypothesis; the artifact's kernels are 3×3.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnnFoldB

/-- The SSA names the renderer threads between the pieces. Denotation-irrelevant; kept so every
    node is stated with the leaf names the emitted text uses. -/
structure Names where
  (lrStr : String)
  (ac1N hc1N ac2N hc2N poolN a3N h3N a4N h4N : String)
  (dyN dy4N dy3N dx3N dac2N dhc2N dac1N dhc1N : String)

variable {ic c h w d1 nClasses kH kW : Nat} (N : Nat)
  (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
  (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
  (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
  (xB : Vec (N * (ic*(2*h)*(2*w)))) (label : Fin N → Fin nClasses) (nm : Names)

/-! ## Per-example shorthands at example `k` (the forward `CnnFold` states, at slice `k`) -/

noncomputable abbrev xk (k : Fin N) : Vec (ic*(2*h)*(2*w)) := batchSlice N (ic*(2*h)*(2*w)) xB k
noncomputable abbrev hc1 (k : Fin N) : Vec (c*(2*h)*(2*w)) :=
  flatConv (h := 2*h) (w := 2*w) W₁ b₁ (xk N xB k)
noncomputable abbrev ac1 (k : Fin N) : Vec (c*(2*h)*(2*w)) := relu (c*(2*h)*(2*w)) (hc1 N W₁ b₁ xB k)
noncomputable abbrev hc2 (k : Fin N) : Vec (c*(2*h)*(2*w)) :=
  flatConv (h := 2*h) (w := 2*w) W₂ b₂ (ac1 N W₁ b₁ xB k)
noncomputable abbrev ac2 (k : Fin N) : Vec (c*(2*h)*(2*w)) :=
  relu (c*(2*h)*(2*w)) (hc2 N W₁ b₁ W₂ b₂ xB k)
noncomputable abbrev pool (k : Fin N) : Vec (c*h*w) := maxPoolFlat c h w (ac2 N W₁ b₁ W₂ b₂ xB k)
noncomputable abbrev h3 (k : Fin N) : Vec d1 := dense W₃ b₃ (pool N W₁ b₁ W₂ b₂ xB k)
noncomputable abbrev h4 (k : Fin N) : Vec d1 := dense W₄ b₄ (relu d1 (h3 N W₁ b₁ W₂ b₂ W₃ b₃ xB k))
noncomputable abbrev a4 (k : Fin N) : Vec d1 := relu d1 (h4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k)
/-- The softmax-CE loss cotangent at example `k`. -/
noncomputable abbrev gk (k : Fin N) : Vec nClasses := fun j =>
  softmax nClasses (mnistLinear W₅ b₅ (a4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k)) j - oneHot nClasses (label k) j

/-! ## The batched forward, nested -/

def hc1B : SHlo (N * (c*(2*h)*(2*w))) :=
  .batchOp (N := N) (.conv (h := 2*h) (w := 2*w) "%W1" "%b1" W₁ b₁) (.operand "%x" xB)
def ac1B : SHlo (N * (c*(2*h)*(2*w))) :=
  .batchOp (N := N) (.relu (n := c*(2*h)*(2*w))) (hc1B N W₁ b₁ xB)
def hc2B : SHlo (N * (c*(2*h)*(2*w))) :=
  .batchOp (N := N) (.conv (h := 2*h) (w := 2*w) "%W2" "%b2" W₂ b₂) (ac1B N W₁ b₁ xB)
def ac2B : SHlo (N * (c*(2*h)*(2*w))) :=
  .batchOp (N := N) (.relu (n := c*(2*h)*(2*w))) (hc2B N W₁ b₁ W₂ b₂ xB)
def poolB : SHlo (N * (c*h*w)) :=
  .batchOp (N := N) (.maxPool (c := c) (h := h) (w := w)) (ac2B N W₁ b₁ W₂ b₂ xB)
def h3B : SHlo (N * d1) := .batchOp (N := N) (.dense "%W3" "%b3" W₃ b₃) (poolB N W₁ b₁ W₂ b₂ xB)
def a3B : SHlo (N * d1) := .batchOp (N := N) (.relu (n := d1)) (h3B N W₁ b₁ W₂ b₂ W₃ b₃ xB)
def h4B : SHlo (N * d1) :=
  .batchOp (N := N) (.dense "%W4" "%b4" W₄ b₄) (a3B N W₁ b₁ W₂ b₂ W₃ b₃ xB)
def a4B : SHlo (N * d1) := .batchOp (N := N) (.relu (n := d1)) (h4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB)
def logitsB : SHlo (N * nClasses) :=
  .batchOp (N := N) (.dense "%W5" "%b5" W₅ b₅) (a4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB)
noncomputable def cotB : SHlo (N * nClasses) :=
  .subB (.batchOp (N := N) (.softmaxDiv (n := nClasses))
          (.batchOp (N := N) (.expe (n := nClasses)) (logitsB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB)))
        (.operand "%onehot" (LinFoldB.oneHotB N label))

theorem hc1B_slice (k : Fin N) : batchSlice N _ (den (hc1B N W₁ b₁ xB)) k = hc1 N W₁ b₁ xB k := by
  simp only [hc1B, den_batchOp, batchSlice_batchMap, denOp]; rfl
theorem ac1B_slice (k : Fin N) : batchSlice N _ (den (ac1B N W₁ b₁ xB)) k = ac1 N W₁ b₁ xB k := by
  simp only [ac1B, den_batchOp, batchSlice_batchMap, denOp, hc1B_slice]
theorem hc2B_slice (k : Fin N) :
    batchSlice N _ (den (hc2B N W₁ b₁ W₂ b₂ xB)) k = hc2 N W₁ b₁ W₂ b₂ xB k := by
  simp only [hc2B, den_batchOp, batchSlice_batchMap, denOp, ac1B_slice]
theorem ac2B_slice (k : Fin N) :
    batchSlice N _ (den (ac2B N W₁ b₁ W₂ b₂ xB)) k = ac2 N W₁ b₁ W₂ b₂ xB k := by
  simp only [ac2B, den_batchOp, batchSlice_batchMap, denOp, hc2B_slice]
theorem poolB_slice (k : Fin N) :
    batchSlice N _ (den (poolB N W₁ b₁ W₂ b₂ xB)) k = pool N W₁ b₁ W₂ b₂ xB k := by
  simp only [poolB, den_batchOp, batchSlice_batchMap, denOp, ac2B_slice]
theorem h3B_slice (k : Fin N) :
    batchSlice N _ (den (h3B N W₁ b₁ W₂ b₂ W₃ b₃ xB)) k = h3 N W₁ b₁ W₂ b₂ W₃ b₃ xB k := by
  simp only [h3B, den_batchOp, batchSlice_batchMap, denOp, poolB_slice]
theorem a3B_slice (k : Fin N) :
    batchSlice N _ (den (a3B N W₁ b₁ W₂ b₂ W₃ b₃ xB)) k = relu d1 (h3 N W₁ b₁ W₂ b₂ W₃ b₃ xB k) := by
  simp only [a3B, den_batchOp, batchSlice_batchMap, denOp, h3B_slice]
theorem h4B_slice (k : Fin N) :
    batchSlice N _ (den (h4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB)) k = h4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k := by
  simp only [h4B, den_batchOp, batchSlice_batchMap, denOp, a3B_slice]
theorem a4B_slice (k : Fin N) :
    batchSlice N _ (den (a4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB)) k = a4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k := by
  simp only [a4B, den_batchOp, batchSlice_batchMap, denOp, h4B_slice]
theorem logitsB_slice (k : Fin N) :
    batchSlice N _ (den (logitsB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB)) k
      = mnistLinear W₅ b₅ (a4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k) := by
  simp only [logitsB, den_batchOp, batchSlice_batchMap, denOp, a4B_slice]; rfl

theorem cotB_slice (k : Fin N) (j : Fin nClasses) :
    den (cotB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label) (finProdFinEquiv (k, j))
      = gk N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label k j := by
  have h1 : den (cotB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label) (finProdFinEquiv (k, j))
      = den (.batchOp (N := N) (.softmaxDiv (n := nClasses))
              (.batchOp (N := N) (.expe (n := nClasses)) (logitsB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB)))
            (finProdFinEquiv (k, j))
        - LinFoldB.oneHotB N label (finProdFinEquiv (k, j)) := rfl
  have h2 : ∀ i : Fin nClasses,
      batchSlice N nClasses (den (.batchOp (N := N) (.expe (n := nClasses))
        (logitsB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB))) k i
        = Real.exp (mnistLinear W₅ b₅ (a4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k) i) := by
    intro i; simp only [den_batchOp, batchSlice_batchMap, denOp, logitsB_slice]
  rw [h1, den_batchOp_softmaxDiv_per_example]
  simp only [h2, LinFoldB.oneHotB, Equiv.symm_apply_apply, gk, softmax_apply]

theorem batchSlice_cotB (k : Fin N) :
    batchSlice N nClasses (den (cotB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label)) k
      = gk N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label k := by
  funext j; exact cotB_slice N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label k j

/-! ## The batched backward chain — each piece a node over an operand leaf carrying the previous
piece's `den`, as the renderer threads them -/

/-- `relu'(h4) ⊙ (W₅ · dy)`. -/
noncomputable def dy4B : SHlo (N * d1) :=
  .selectPosB nm.h4N (den (h4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB))
    (.batchOp (N := N) (.dotOut "%W5" W₅)
      (.operand nm.dyN (den (cotB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label))))
/-- `relu'(h3) ⊙ (W₄ · dy4)`. -/
noncomputable def dy3B : SHlo (N * d1) :=
  .selectPosB nm.h3N (den (h3B N W₁ b₁ W₂ b₂ W₃ b₃ xB))
    (.batchOp (N := N) (.dotOut "%W4" W₄)
      (.operand nm.dy4N (den (dy4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm))))
/-- `W₃ · dy3`, the cotangent at the pool output. -/
noncomputable def dx3B : SHlo (N * (c*h*w)) :=
  .batchOp (N := N) (.dotOut "%W3" W₃)
    (.operand nm.dy3N (den (dy3B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)))
/-- The select-and-scatter pool backward at the saved pool input. -/
noncomputable def dac2B : SHlo (N * (c*(2*h)*(2*w))) :=
  .maxPoolBackB (N := N) (c := c) (h := h) (w := w) nm.ac2N (den (ac2B N W₁ b₁ W₂ b₂ xB))
    (.operand nm.dx3N (den (dx3B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)))
/-- `relu'(hc2) ⊙ dac2`, the cotangent at conv₂'s output. -/
noncomputable def dhc2B : SHlo (N * (c*(2*h)*(2*w))) :=
  .selectPosB nm.hc2N (den (hc2B N W₁ b₁ W₂ b₂ xB))
    (.operand nm.dac2N (den (dac2B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)))
/-- The conv₂ input-VJP (the `convBackP` descriptor: `.convBack`'s text at the batched index). -/
noncomputable def dac1B : SHlo (N * (c*(2*h)*(2*w))) :=
  .batchOp (N := N) (.convBackP (h := 2*h) (w := 2*w) "%W2" W₂ b₂)
    (.operand nm.dhc2N (den (dhc2B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)))
/-- `relu'(hc1) ⊙ dac1`, the cotangent at conv₁'s output. -/
noncomputable def dhc1B : SHlo (N * (c*(2*h)*(2*w))) :=
  .selectPosB nm.hc1N (den (hc1B N W₁ b₁ xB))
    (.operand nm.dac1N (den (dac1B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)))

theorem dy4B_slice (k : Fin N) :
    batchSlice N d1 (den (dy4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)) k
      = (mlpCotOut1 W₅ (h4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k)).denote
          (gk N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label k) := by
  rw [mlpCotOut1_denote, dy4B, LinFoldB.batchSlice_selectPosB, LinFoldB.batchSlice_dotOutB,
    LinFoldB.batchSlice_operand, h4B_slice, batchSlice_cotB]

theorem dy3B_slice (k : Fin N) :
    batchSlice N d1 (den (dy3B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)) k
      = (mlpCotOut0 W₄ W₅ (h3 N W₁ b₁ W₂ b₂ W₃ b₃ xB k) (h4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k)).denote
          (gk N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label k) := by
  rw [mlpCotOut0_denote, dy3B, LinFoldB.batchSlice_selectPosB, LinFoldB.batchSlice_dotOutB,
    LinFoldB.batchSlice_operand, h3B_slice, dy4B_slice, mlpCotOut1_denote]

/-- The dense head's input cotangent (`cnnDenseHeadCot`) is `W₃ · mlpCotOut0`. -/
theorem cnnDenseHeadCot_denote (h3v h4v : Vec d1) (g : Vec nClasses) :
    (cnnDenseHeadCot W₃ W₄ W₅ h3v h4v).denote g = Mat.mulVec W₃ ((mlpCotOut0 W₄ W₅ h3v h4v).denote g) := by
  unfold cnnDenseHeadCot mlpCotOut0 mlpCotOut1
  rw [denote_subst]; rfl

theorem dx3B_slice (k : Fin N) :
    batchSlice N (c*h*w) (den (dx3B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)) k
      = (cnnDenseHeadCot W₃ W₄ W₅ (h3 N W₁ b₁ W₂ b₂ W₃ b₃ xB k) (h4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k)).denote
          (gk N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label k) := by
  rw [cnnDenseHeadCot_denote, dx3B, LinFoldB.batchSlice_dotOutB, LinFoldB.batchSlice_operand,
    dy3B_slice]

/-- `maxPoolBackFlat` is the flattened `maxPoolBackDenote`: the same first-maximum routing. -/
theorem maxPoolBackFlat_eq_flatten (xv : Vec (c*(2*h)*(2*w))) (dyv : Vec (c*h*w)) :
    maxPoolBackFlat c h w xv dyv
      = Tensor3.flatten (maxPoolBackDenote (Tensor3.unflatten xv : Tensor3 c (2*h) (2*w))
          (Tensor3.unflatten dyv)) := rfl

/-- **The conv₂-output cotangent, batched, IS `cnnChainCotW2` at each example.** The emitted
    `selectPosB` over `maxPoolBackB` over the head chain denotes, at slice `k`, exactly the
    hand-named chain cotangent `CnnFold` is stated at. -/
theorem dhc2B_slice (k : Fin N) :
    batchSlice N (c*(2*h)*(2*w)) (den (dhc2B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)) k
      = cnnChainCotW2 W₃ W₄ W₅ (h3 N W₁ b₁ W₂ b₂ W₃ b₃ xB k) (h4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k)
          (Tensor3.unflatten (ac2 N W₁ b₁ W₂ b₂ xB k)) (hc2 N W₁ b₁ W₂ b₂ xB k)
          (gk N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label k) := by
  rw [cnnChainCotW2_eq, dhc2B, LinFoldB.batchSlice_selectPosB, LinFoldB.batchSlice_operand,
    hc2B_slice, dac2B, den_maxPoolBackB, batchSlice_batchMapAux, LinFoldB.batchSlice_operand,
    ac2B_slice, dx3B_slice, maxPoolBackFlat_eq_flatten]

/-- **The conv₁-output cotangent, batched, IS `cnnChainCotW1` at each example — for odd kernels.**
    `convBackP` denotes the certified conv input-VJP; `cnnChainCotW1` is written with the
    reversed-kernel `convBackDenote`, and the two agree exactly when the kernel is odd
    (`IR.conv_back_bridge`). -/
theorem dhc1B_slice (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (k : Fin N) :
    batchSlice N (c*(2*h)*(2*w)) (den (dhc1B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)) k
      = cnnChainCotW1 W₂ (hc1 N W₁ b₁ xB k)
          (batchSlice N (c*(2*h)*(2*w)) (den (dhc2B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)) k) := by
  rw [cnnChainCotW1_eq, dhc1B, LinFoldB.batchSlice_selectPosB, LinFoldB.batchSlice_operand,
    hc1B_slice, dac1B, den_batchOp, batchSlice_batchMap, LinFoldB.batchSlice_operand]
  funext i
  congr 1
  show Tensor3.flatten ((conv2dHasVJP3 W₂ b₂).backward (Tensor3.unflatten (fun _ => 0))
      (Tensor3.unflatten _)) i = _
  rw [← conv_back_bridge hkH hkW W₂ b₂]

/-! ## The ten parameter nodes at batch `N` -/

/-- A batched dense weight SGD node, tied: `θ − lr·Σₙ (certified ∂dense/∂W at slice n · cotₙ)`. -/
def DenseWSgdTiedB {a cc : Nat} (xN wN lrStr cotN : String) (x : Vec (N * a)) (W : Mat a cc)
    (b : Vec cc) (cot : Vec (N * cc)) (lr : ℝ) : Prop :=
  ∀ (i : Fin a) (j : Fin cc),
    den (SHlo.denseWeightSgdB xN wN lrStr x W lr (.operand cotN cot)) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ n : Fin N, ∑ k : Fin cc,
          pdiv (fun v : Vec (a * cc) => dense (Mat.unflatten v) b (batchSlice N a x n))
               (Mat.flatten W) (finProdFinEquiv (i, j)) k * batchSlice N cc cot n k

/-- A batched dense bias SGD node, tied. -/
def DenseBSgdTiedB {a cc : Nat} (bN lrStr cotN : String) (W : Mat a cc) (x : Vec (N * a))
    (b : Vec cc) (cot : Vec (N * cc)) (lr : ℝ) : Prop :=
  ∀ j : Fin cc,
    den (SHlo.denseBiasSgdB bN lrStr b lr (.operand cotN cot)) j
      = b j - lr * ∑ n : Fin N, ∑ k : Fin cc,
          pdiv (fun b' : Vec cc => dense W b' (batchSlice N a x n)) b j k * batchSlice N cc cot n k

/-- A batched conv weight SGD node, tied: `θ − lr·Σₙ (certified ∂conv/∂W at slice n · cotₙ)`. -/
def ConvWSgdTiedB {ic' oc h' w' : Nat} (xN wN lrStr cotN : String) (b : Vec oc)
    (x : Vec (N * (ic' * h' * w'))) (W : Kernel4 oc ic' kH kW) (cot : Vec (N * (oc * h' * w')))
    (lr : ℝ) : Prop :=
  ∀ idx : Fin (oc * ic' * kH * kW),
    den (SHlo.convWeightSgdB xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Kernel4.flatten W idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h' * w'),
          pdiv (fun v' : Vec (oc * ic' * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (ic' * h' * w') x n))))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h' * w') cot n j

/-- A batched conv bias SGD node, tied. -/
def ConvBSgdTiedB {ic' oc h' w' : Nat} (bN lrStr cotN : String) (W : Kernel4 oc ic' kH kW)
    (x : Vec (N * (ic' * h' * w'))) (b : Vec oc) (cot : Vec (N * (oc * h' * w'))) (lr : ℝ) : Prop :=
  ∀ o : Fin oc,
    den (SHlo.convBiasSgdB bN lrStr W x b lr (.operand cotN cot)) o
      = b o - lr * ∑ n : Fin N, ∑ j : Fin (oc * h' * w'),
          pdiv (fun b' : Vec oc =>
                  Tensor3.flatten (conv2d W b' (Tensor3.unflatten (batchSlice N (ic' * h' * w') x n))))
               b o j * batchSlice N (oc * h' * w') cot n j

theorem denseWSgdTiedB_holds {a cc : Nat} {xN wN lrStr cotN : String} {x : Vec (N * a)}
    {W : Mat a cc} {b : Vec cc} {cot : Vec (N * cc)} {lr : ℝ} :
    DenseWSgdTiedB N xN wN lrStr cotN x W b cot lr := by
  intro i j
  rw [LinFoldB.denseWeightSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun n _ => ?_
  rw [LinFoldB.dense_jacW_collapse]; rfl

theorem denseBSgdTiedB_holds {a cc : Nat} {bN lrStr cotN : String} {W : Mat a cc} {x : Vec (N * a)}
    {b : Vec cc} {cot : Vec (N * cc)} {lr : ℝ} :
    DenseBSgdTiedB N bN lrStr cotN W x b cot lr := by
  intro j
  rw [LinFoldB.denseBiasSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun n _ => ?_
  rw [LinFoldB.dense_jacB_collapse]; rfl

theorem convWSgdTiedB_holds {ic' oc h' w' : Nat} {xN wN lrStr cotN : String} {b : Vec oc}
    {x : Vec (N * (ic' * h' * w'))} {W : Kernel4 oc ic' kH kW} {cot : Vec (N * (oc * h' * w'))}
    {lr : ℝ} : ConvWSgdTiedB N (kH := kH) (kW := kW) xN wN lrStr cotN b x W cot lr := by
  intro idx
  rw [convWeightSgdB_eq_grad, GradNodeB.convWGradB_den]

theorem convBSgdTiedB_holds {ic' oc h' w' : Nat} {bN lrStr cotN : String} {W : Kernel4 oc ic' kH kW}
    {x : Vec (N * (ic' * h' * w'))} {b : Vec oc} {cot : Vec (N * (oc * h' * w'))} {lr : ℝ} :
    ConvBSgdTiedB N (kH := kH) (kW := kW) bN lrStr cotN W x b cot lr := by
  intro o
  rw [convBiasSgdB_eq_grad, GradNodeB.convBGradB_den]

/-- **Output weight at the whole loss, summed over the batch.** -/
theorem w5B_tied (aN : String) (lr : ℝ) (i : Fin d1) (j : Fin nClasses) :
    den (SHlo.denseWeightSgdB aN "%W5" nm.lrStr (den (a4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB)) W₅ lr
          (.operand nm.dyN (den (cotB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label)))) (finProdFinEquiv (i, j))
      = W₅ i j - lr * ∑ k : Fin N,
          pdiv (fun (v : Vec (d1 * nClasses)) (_ : Fin 1) =>
                  crossEntropy nClasses (dense (Mat.unflatten v) b₅ (a4 N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB k)) (label k))
               (Mat.flatten W₅) (finProdFinEquiv (i, j)) 0 := by
  rw [LinFoldB.denseWeightSgdB_den_slices]
  congr 2
  refine Finset.sum_congr rfl fun k _ => ?_
  rw [lossWeightGrad_eq_sum, LinFoldB.dense_jacW_collapse]
  show batchSlice N d1 (den (a4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB)) k i
      * batchSlice N nClasses (den (.operand nm.dyN (den (cotB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label)))) k j = _
  rw [LinFoldB.batchSlice_operand, a4B_slice, batchSlice_cotB]

/-- **Whole CNN train step at batch `N`, tied.** All ten parameter nodes, each at the real batched
    forward activation and the slice-pinned chain cotangent: the dense head against
    `g`/`mlpCotOut1`/`mlpCotOut0` per example (`dy4B_slice`, `dy3B_slice`), conv₂ against the
    emitted pool-back chain (`dhc2B_slice` = `cnnChainCotW2`), conv₁ against the emitted conv-back
    chain (`dhc1B_slice` = `cnnChainCotW1` for odd kernels). The output weight folds to the whole
    loss in `w5B_tied`.

    What it does not say on its own: each per-op conjunct is the op's update law at the cotangent
    the chain feeds it and holds for any cotangent, so this theorem fixes which node each op reads;
    that those nodes are the per-example chain cotangents above is the separate `*_slice` lemmas,
    and that those are the loss's gradient is `cnn_net_lossGrad`. -/
theorem cnn_train_step_tiedB (a4N a3N poolN ac1N : String) (lr : ℝ) :
    let a4v := den (a4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ xB)
    let a3v := den (a3B N W₁ b₁ W₂ b₂ W₃ b₃ xB)
    let poolv := den (poolB N W₁ b₁ W₂ b₂ xB)
    let ac1v := den (ac1B N W₁ b₁ xB)
    let g := den (cotB N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label)
    let dy4 := den (dy4B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)
    let dy3 := den (dy3B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)
    let dhc2 := den (dhc2B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)
    let dhc1 := den (dhc1B N W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xB label nm)
    -- dense head (output layer first)
    DenseWSgdTiedB N a4N "%W5" nm.lrStr nm.dyN a4v W₅ b₅ g lr
  ∧ DenseBSgdTiedB N "%b5" nm.lrStr nm.dyN W₅ a4v b₅ g lr
  ∧ DenseWSgdTiedB N a3N "%W4" nm.lrStr nm.dy4N a3v W₄ b₄ dy4 lr
  ∧ DenseBSgdTiedB N "%b4" nm.lrStr nm.dy4N W₄ a3v b₄ dy4 lr
  ∧ DenseWSgdTiedB N poolN "%W3" nm.lrStr nm.dy3N poolv W₃ b₃ dy3 lr
  ∧ DenseBSgdTiedB N "%b3" nm.lrStr nm.dy3N W₃ poolv b₃ dy3 lr
    -- conv₂, conv₁
  ∧ ConvWSgdTiedB N (kH := kH) (kW := kW) ac1N "%W2" nm.lrStr nm.dhc2N b₂ ac1v W₂ dhc2 lr
  ∧ ConvBSgdTiedB N (kH := kH) (kW := kW) "%b2" nm.lrStr nm.dhc2N W₂ ac1v b₂ dhc2 lr
  ∧ ConvWSgdTiedB N (kH := kH) (kW := kW) "%x" "%W1" nm.lrStr nm.dhc1N b₁ xB W₁ dhc1 lr
  ∧ ConvBSgdTiedB N (kH := kH) (kW := kW) "%b1" nm.lrStr nm.dhc1N W₁ xB b₁ dhc1 lr :=
  ⟨denseWSgdTiedB_holds N, denseBSgdTiedB_holds N, denseWSgdTiedB_holds N, denseBSgdTiedB_holds N,
    denseWSgdTiedB_holds N, denseBSgdTiedB_holds N, convWSgdTiedB_holds N, convBSgdTiedB_holds N,
    convWSgdTiedB_holds N, convBSgdTiedB_holds N⟩

end Proofs.CnnFoldB
