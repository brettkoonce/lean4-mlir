import LeanMlir.Proofs.CnnChainClose
import LeanMlir.Proofs.CnnRender

/-! # PoC: the MNIST-CNN train step, proof-tied to the certified SGD step

The CNN analogue of `LinearFaithfulPoC` / `MlpFaithfulPoC`. `MainMnistCnnVerified`
trains on `verified_mlir/cnn_train_step.mlir`; this file makes the *parameter
updates* of that module `den`-faithful — each emitted SGD op denotes the certified
(`fderiv`/VJP-derived) softmax-CE loss-descent step.

The CNN has two kinds of parameter: the **dense classifier head** (`W₃,W₄,W₅` +
biases — structurally a 3-layer MLP over the flattened pool output) and the
**convolution kernels/biases** (`W₁,W₂` + biases). The dense head reuses the
`weightSgd`/`biasSgd` `SHlo` ops added in `LinearFaithfulPoC` (its `den`s certified
via the M2 `weight_grad_bridge`/`bias_grad_bridge` at the `mlpCotOut`-style chain
cotangents — the head is a 3-layer MLP, so the IR `mlpCotOut0/1` apply verbatim).
The conv layers use the **new core ops** `convWeightSgd`/`convBiasSgd`
(StableHLO.lean): their `den` is `flatten(W − lr·conv2d_weight_grad…)` /
`b − lr·conv2d_bias_grad…`, proven = certified by the chain-pinned conv bridges
`cnn_render_conv{W,b}{1,2}_chain_certified` (CnnChainClose.lean) at the cotangents
the CNN backward chain actually delivers (`cnnChainCotW1`/`cnnChainCotW2`).

(Namespace/name lengths are kept short on purpose: `tests/AuditAxioms.lean`'s
three-axiom closure check greps `#print axioms` output per line, which Lean wraps
past ~120 cols — long qualified names would split the benign triple across lines
and false-fail the check.)

## What is closed here (kernel, `[propext, Classical.choice, Quot.sound]`)

* `cW1_den`/`cb1_den`/`cW2_den`/`cb2_den` — the four emitted **conv** param ops
  (`convWeightSgd`/`convBiasSgd`), fed the chain cotangent, denote the certified
  conv kernel/bias loss-descent step. The conv tail is now "under `den`" exactly
  like the forward.
* `dW3..dW5`/`db3..db5` — the six emitted **dense-head** param ops
  (`weightSgd`/`biasSgd`) denote the certified dense loss-descent step.

## Honest residual (the boundary shared with the forward `SHlo` `den`)

* **Cotangent subgraph ⇄ rendered SHlo.** The cotangents here are the *chain*
  cotangents (`cnnChainCotW1/2`, `mlpCotOut0/1`), proven = the rendered backward
  form in `CnnChainClose` (`cnnChainCotW{1,2}_eq`) and `IR` (`mlpCotOut*_denote`);
  pinning each to the exact emitted `selectPos`/`dotOut`/`convBack`/`maxPoolBack`
  SHlo subgraph (as `MlpPoC.cot{0,1}_den` does for the MLP) is the remaining polish.
* **Per-op `pretty` lexing** (shared with the whole suite) + **ℝ → Float32**.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnnPoC

/-! ## Convolution layers — the new `convWeightSgd`/`convBiasSgd` ops denote certified

`den (convWeightSgd … (.operand _ c))` is by construction
`flatten W − lr·conv2d_weight_grad(b,x)·c` (and likewise for the bias); pinning
`c` to the cotangent the chain delivers and applying the chain-certified conv
bridge gives `θ − lr·(certified ∂conv/∂θ · the-chain-cotangent)`. (The `den`
reduction is definitional — `rfl` — exactly as `LinPoC.poc_weightSgd_den_eq`.) -/

/-- **Conv-2 weight op = certified.** The emitted `convWeightSgd` for `W₂`, fed the
    conv-2 chain cotangent, denotes `W₂ − lr·(certified ∂conv2/∂W₂ · chain cot)`. -/
theorem cW2_den {c h w d1 nClasses kH kW : Nat}
    (xN wN lrStr cotN : String) (b₂ : Vec c) (ac1 ac2 : Tensor3 c (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (W₃ : Mat (c*h*w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses)
    (h3 h4 : Vec d1) (hc2 : Vec (c*(2*h)*(2*w))) (dy : Vec nClasses) (lr : ℝ)
    (idx : Fin (c*c*kH*kW)) :
    den (SHlo.convWeightSgd xN wN lrStr b₂ ac1 W₂ lr
          (.operand cotN (cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy))) idx
      = Kernel4.flatten W₂ idx - lr * ∑ j : Fin (c*(2*h)*(2*w)),
          pdiv (fun v' : Vec (c*c*kH*kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ ac1))
               (Kernel4.flatten W₂) idx j * cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy j :=
  cnn_render_convW2_chain_certified b₂ ac1 W₃ W₄ W₅ h3 h4 ac2 hc2 dy (Kernel4.flatten W₂) lr idx

/-- **Conv-2 bias op = certified.** -/
theorem cb2_den {c h w d1 nClasses kH kW : Nat}
    (bN lrStr cotN : String) (ac1 ac2 : Tensor3 c (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (W₃ : Mat (c*h*w) d1) (W₄ : Mat d1 d1) (W₅ : Mat d1 nClasses)
    (b₂ : Vec c) (h3 h4 : Vec d1) (hc2 : Vec (c*(2*h)*(2*w))) (dy : Vec nClasses) (lr : ℝ)
    (o : Fin c) :
    den (SHlo.convBiasSgd bN lrStr W₂ ac1 b₂ lr
          (.operand cotN (cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy))) o
      = b₂ o - lr * ∑ j : Fin (c*(2*h)*(2*w)),
          pdiv (fun b' : Vec c => Tensor3.flatten (conv2d W₂ b' ac1)) b₂ o j
            * cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 dy j :=
  cnn_render_convb2_chain_certified W₂ b₂ ac1 W₃ W₄ W₅ h3 h4 ac2 hc2 dy lr o

/-- **Conv-1 weight op = certified.** The deepest conv layer, at the chain cotangent
    `cnnChainCotW1` (which crosses one more conv-back than conv-2's). -/
theorem cW1_den {ic c h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b₁ : Vec c) (x : Tensor3 ic (2*h) (2*w))
    (W₁ : Kernel4 c ic kH kW) (W₂ : Kernel4 c c kH kW)
    (hc1 cotW2 : Vec (c*(2*h)*(2*w))) (lr : ℝ) (idx : Fin (c*ic*kH*kW)) :
    den (SHlo.convWeightSgd xN wN lrStr b₁ x W₁ lr
          (.operand cotN (cnnChainCotW1 W₂ hc1 cotW2))) idx
      = Kernel4.flatten W₁ idx - lr * ∑ j : Fin (c*(2*h)*(2*w)),
          pdiv (fun v' : Vec (c*ic*kH*kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b₁ x))
               (Kernel4.flatten W₁) idx j * cnnChainCotW1 W₂ hc1 cotW2 j :=
  cnn_render_convW1_chain_certified b₁ x hc1 cotW2 W₂ (Kernel4.flatten W₁) lr idx

/-- **Conv-1 bias op = certified.** -/
theorem cb1_den {ic c h w kH kW : Nat}
    (bN lrStr cotN : String) (W₁ : Kernel4 c ic kH kW) (x : Tensor3 ic (2*h) (2*w))
    (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (hc1 cotW2 : Vec (c*(2*h)*(2*w))) (lr : ℝ)
    (o : Fin c) :
    den (SHlo.convBiasSgd bN lrStr W₁ x b₁ lr
          (.operand cotN (cnnChainCotW1 W₂ hc1 cotW2))) o
      = b₁ o - lr * ∑ j : Fin (c*(2*h)*(2*w)),
          pdiv (fun b' : Vec c => Tensor3.flatten (conv2d W₁ b' x)) b₁ o j
            * cnnChainCotW1 W₂ hc1 cotW2 j :=
  cnn_render_convb1_chain_certified W₁ b₁ x hc1 cotW2 W₂ lr o

/-! ## Dense classifier head — reuse `weightSgd`/`biasSgd` (the head is a 3-layer MLP)

The pool-output `pool : Vec (c·h·w)` flows through `W₃→relu→W₄→relu→W₅`; the
per-layer cotangents are exactly the IR `mlpCotOut0/1` (with `(W₅,W₄,W₃)` playing
the MLP's `(W₂,W₁,W₀)`). Each emitted op `den` = certified via the dense bridges,
mirroring `MlpFaithfulPoC` verbatim. -/

/-- Output-layer weight op `W₅` = certified step (cotangent = the loss cotangent `dy`). -/
theorem dW5_den {c h w d1 nClasses : Nat}
    (aN lrStr dyN : String) (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (pool : Vec (c*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i : Fin d1) (j : Fin nClasses) :
    den (SHlo.weightSgd aN "%W5" lrStr (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) W₅ lr
          (.operand dyN dy)) (finProdFinEquiv (i, j))
      = W₅ i j - lr * ∑ k : Fin nClasses,
          pdiv (fun v : Vec (d1 * nClasses) =>
                  dense (Mat.unflatten v) b₅ (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))))
               (Mat.flatten W₅) (finProdFinEquiv (i, j)) k * dy k := by
  have step : den (SHlo.weightSgd aN "%W5" lrStr (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) W₅ lr
                (.operand dyN dy)) (finProdFinEquiv (i, j))
            = W₅ i j - lr * emitWeightGrad (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool))))
                Back.cotangent dy i j := by
    simp only [den, emitWeightGrad, Mat.outer, Back.denote, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, weight_grad_bridge W₅ b₅ (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool))))
        Back.cotangent dy i j]; rfl

/-- Hidden-layer weight op `W₄` = certified step (cotangent = `mlpCotOut1 W₅ h4`). -/
theorem dW4_den {c h w d1 nClasses : Nat}
    (aN lrStr cN : String) (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (pool : Vec (c*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i j : Fin d1) :
    den (SHlo.weightSgd aN "%W4" lrStr (relu d1 (dense W₃ b₃ pool)) W₄ lr
          (.operand cN ((mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy)))
        (finProdFinEquiv (i, j))
      = W₄ i j - lr * ∑ k : Fin d1,
          pdiv (fun v : Vec (d1 * d1) => dense (Mat.unflatten v) b₄ (relu d1 (dense W₃ b₃ pool)))
               (Mat.flatten W₄) (finProdFinEquiv (i, j)) k
            * (mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy k := by
  have step : den (SHlo.weightSgd aN "%W4" lrStr (relu d1 (dense W₃ b₃ pool)) W₄ lr
                (.operand cN ((mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy)))
                (finProdFinEquiv (i, j))
            = W₄ i j - lr * emitWeightGrad (relu d1 (dense W₃ b₃ pool))
                (mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) dy i j := by
    simp only [den, emitWeightGrad, Mat.outer, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, weight_grad_bridge W₄ b₄ (relu d1 (dense W₃ b₃ pool))
        (mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) dy i j]

/-- Input-layer (pool) weight op `W₃` = certified step (cotangent = `mlpCotOut0 W₄ W₅ h3 h4`). -/
theorem dW3_den {c h w d1 nClasses : Nat}
    (lrStr cN : String) (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (pool : Vec (c*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i : Fin (c*h*w)) (j : Fin d1) :
    den (SHlo.weightSgd "%pool" "%W3" lrStr pool W₃ lr
          (.operand cN ((mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool)
                          (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy)))
        (finProdFinEquiv (i, j))
      = W₃ i j - lr * ∑ k : Fin d1,
          pdiv (fun v : Vec ((c*h*w) * d1) => dense (Mat.unflatten v) b₃ pool)
               (Mat.flatten W₃) (finProdFinEquiv (i, j)) k
            * (mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool)
                (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy k := by
  have step : den (SHlo.weightSgd "%pool" "%W3" lrStr pool W₃ lr
                (.operand cN ((mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool)
                                (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy)))
                (finProdFinEquiv (i, j))
            = W₃ i j - lr * emitWeightGrad pool
                (mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool)
                  (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) dy i j := by
    simp only [den, emitWeightGrad, Mat.outer, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, weight_grad_bridge W₃ b₃ pool
        (mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool) (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) dy i j]

/-- Output-layer bias op `b₅` = certified step. -/
theorem db5_den {c h w d1 nClasses : Nat}
    (lrStr dyN : String) (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (pool : Vec (c*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i : Fin nClasses) :
    den (SHlo.biasSgd "%b5" lrStr b₅ lr (.operand dyN dy)) i
      = b₅ i - lr * ∑ j : Fin nClasses,
          pdiv (fun b' : Vec nClasses =>
                  dense W₅ b' (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool))))) b₅ i j * dy j := by
  have step : den (SHlo.biasSgd "%b5" lrStr b₅ lr (.operand dyN dy)) i
            = b₅ i - lr * emitBiasGrad Back.cotangent dy i := by
    simp only [den, emitBiasGrad, Back.denote]
  rw [step, bias_grad_bridge W₅ b₅ (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool))))
        Back.cotangent dy i]; rfl

/-- Hidden-layer bias op `b₄` = certified step. -/
theorem db4_den {c h w d1 nClasses : Nat}
    (lrStr cN : String) (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (pool : Vec (c*h*w)) (dy : Vec nClasses) (lr : ℝ) (i : Fin d1) :
    den (SHlo.biasSgd "%b4" lrStr b₄ lr
          (.operand cN ((mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy))) i
      = b₄ i - lr * ∑ j : Fin d1,
          pdiv (fun b' : Vec d1 => dense W₄ b' (relu d1 (dense W₃ b₃ pool))) b₄ i j
            * (mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy j := by
  have step : den (SHlo.biasSgd "%b4" lrStr b₄ lr
                (.operand cN ((mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy))) i
            = b₄ i - lr * emitBiasGrad (mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) dy i := by
    simp only [den, emitBiasGrad]
  rw [step, bias_grad_bridge W₄ b₄ (relu d1 (dense W₃ b₃ pool))
        (mlpCotOut1 W₅ (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) dy i]

/-- Input-layer (pool) bias op `b₃` = certified step. -/
theorem db3_den {c h w d1 nClasses : Nat}
    (lrStr cN : String) (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (pool : Vec (c*h*w)) (dy : Vec nClasses) (lr : ℝ) (i : Fin d1) :
    den (SHlo.biasSgd "%b3" lrStr b₃ lr
          (.operand cN ((mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool)
                          (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy))) i
      = b₃ i - lr * ∑ j : Fin d1,
          pdiv (fun b' : Vec d1 => dense W₃ b' pool) b₃ i j
            * (mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool)
                (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy j := by
  have step : den (SHlo.biasSgd "%b3" lrStr b₃ lr
                (.operand cN ((mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool)
                                (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))).denote dy))) i
            = b₃ i - lr * emitBiasGrad (mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool)
                (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) dy i := by
    simp only [den, emitBiasGrad]
  rw [step, bias_grad_bridge W₃ b₃ pool
        (mlpCotOut0 W₄ W₅ (dense W₃ b₃ pool) (dense W₄ b₄ (relu d1 (dense W₃ b₃ pool)))) dy i]

/-! ## Tie (dense head) — the top loss cotangent is the composed softmax-CE of the CONV forward

The cnn `*_den_certified` above hold for a free top cotangent `dy` and a free pool output. The
renderer feeds the cotangent the emitted loss graph `sub(softmaxDiv(expe(logits)), onehot)` produces,
with `logits` the REAL conv-forward output `mnistCnnNoBnForward … x`. The lemma below pins that graph
to the composed softmax-CE gradient *of the conv forward* (the cnn analogue of `mlpLossCot_den`), and
the headline folds the dense output weight `W₅` to the whole-loss gradient `∂CE/∂W₅` — so the output
layer is tied forward(conv+dense)→softmax-CE→gradient. (The CONV layers `W₁`/`W₂` need the conv
backward chain composed — the hand-written `selMask4`/`scatter`/`convBack` rendered as `SHlo` + the
conv cotangent-subgraph pins — which is the bigger conv-side work; see §1a of the planning doc.) -/

/-- **The emitted loss-cotangent graph denotes the composed softmax-CE gradient of the CONV forward**
    (`= softmax(mnistCnnNoBnForward … x) − onehot = ∂CE/∂logits` at the real conv-forward logits). -/
theorem cnnLossCot_den {ic c h w d1 nClasses kH kW : Nat}
    (nlogN ohN : String)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) (label : Fin nClasses) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe
            (.operand nlogN (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x))))
          (.operand ohN (oneHot nClasses label)))
      = fun j => softmax nClasses (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x) j
                  - oneHot nClasses label j := by
  funext j; simp only [den, softmax]

/-- **Dense output weight op, tied to the WHOLE softmax-CE loss through the conv forward.** With the
    pool output = the real conv forward (`maxPoolFlat ∘ relu ∘ conv₂ ∘ relu ∘ conv₁`) and the
    cotangent the emitted loss graph denotes (`cnnLossCot_den`), the `weightSgd` for `W₅` denotes
    `W₅ − lr·∂(crossEntropy ∘ forward)/∂W₅`. -/
theorem cnn_W5_tied_totalloss {ic c h w d1 nClasses kH kW : Nat}
    (aN lrStr dyN : String)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) (label : Fin nClasses)
    (lr : ℝ) (i : Fin d1) (j : Fin nClasses) :
    den (SHlo.weightSgd aN "%W5" lrStr
          (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃
            (maxPoolFlat c h w (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₂ b₂
              (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₁ b₁ x))))))))) W₅ lr
          (.operand dyN (fun k => softmax nClasses (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x) k
              - oneHot nClasses label k)))
        (finProdFinEquiv (i, j))
      = W₅ i j - lr * pdiv (fun v : Vec (d1 * nClasses) => fun _ : Fin 1 =>
            crossEntropy nClasses (dense (Mat.unflatten v) b₅
              (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃
                (maxPoolFlat c h w (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₂ b₂
                  (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₁ b₁ x)))))))))) label)
          (Mat.flatten W₅) (finProdFinEquiv (i, j)) 0 := by
  rw [dW5_den aN lrStr dyN W₃ b₃ W₄ b₄ W₅ b₅
        (maxPoolFlat c h w (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₂ b₂
          (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₁ b₁ x)))))
        (fun k => softmax nClasses (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x) k
          - oneHot nClasses label k) lr i j,
      mlp_output_total_loss_grad W₅ b₅
        (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃
          (maxPoolFlat c h w (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₂ b₂
            (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₁ b₁ x))))))))) label i j]
  -- both the threaded loss cotangent (`mnistCnnNoBnForward`) and the fold's `mnistLinear W₅ b₅ a₄`
  -- are `dense W₅ b₅ (relu … pool)` — unfold both to match.
  simp only [mnistCnnNoBnForward, mnistLinear, Function.comp_apply]

end Proofs.CnnPoC
