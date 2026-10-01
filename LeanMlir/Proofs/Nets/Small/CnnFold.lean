import LeanMlir.Proofs.Nets.Small.CnnChainClose
import LeanMlir.Proofs.Nets.Small.MnistCNN
import LeanMlir.Proofs.Nets.Small.MlpTrainStep
import LeanMlir.Proofs.Foundation.SgdNodes

/-! # The MNIST-CNN train step, proof-tied to the certified SGD step

The CNN analogue of `LinearFold` / `MlpFold`. `MainMnistCnnVerified`
trains on `verified_mlir/cnn_train_step.mlir`; this file states what the *parameter
updates* of that module denote: each emitted SGD op denotes
`θ − lr·(certified per-layer Jacobian · the cotangent the rendered backward chain feeds it)`,
and the output weight `W₅` is folded to the whole-loss gradient (`cnn_W5_tied_totalloss`).

The CNN has two kinds of parameter: the **dense classifier head** (`W₃,W₄,W₅` +
biases — structurally a 3-layer MLP over the flattened pool output) and the
**convolution kernels/biases** (`W₁,W₂` + biases). The dense head reuses the
`weightSgd`/`biasSgd` `SHlo` ops added in `LinearFold` (its `den`s certified
via `weight_grad_bridge`/`bias_grad_bridge` at the `mlpCotOut`-style chain
cotangents — the head is a 3-layer MLP, so the IR `mlpCotOut0/1` apply verbatim).
The conv layers use the core ops `convWeightSgd`/`convBiasSgd`
(StableHLO/Basic.lean): their `den` is `flatten(W − lr·conv2dWeightGrad…)` /
`b − lr·conv2dBiasGrad…`, proven = certified by the chain-pinned conv bridges
`cnn_render_conv{W,b}{1,2}_chain_certified` (CnnChainClose.lean) at the cotangents
the CNN backward chain delivers (`cnnChainCotW1`/`cnnChainCotW2`).

(Namespace/name lengths are kept short on purpose: [`tests/AuditAxioms.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/AuditAxioms.lean)'s
three-axiom closure check greps `#print axioms` output per line, which Lean wraps
past ~120 cols — long qualified names would split the benign triple across lines
and false-fail the check.)

## What is proved here (kernel, `[propext, Classical.choice, Quot.sound]`)

* `cW1_den`/`cb1_den`/`cW2_den`/`cb2_den` — the four emitted **conv** param ops
  (`convWeightSgd`/`convBiasSgd`), fed the chain cotangent `c`, denote
  `θ − lr·(certified ∂conv/∂θ · c)`.
* `dW5_den` — the output-layer **dense-head** op (`weightSgd`) denotes
  `W₅ − lr·(certified ∂dense/∂W₅ · dy)`; the other five head ops are `SgdNode.denseW_den` /
  `SgdNode.denseB_den` at their layer.
* `cnn_W5_tied_totalloss` — at the emitted loss cotangent, the `W₅` op denotes
  `W₅ − lr·∂(crossEntropy ∘ forward)/∂W₅`.
* `cnn_train_step_tied_certified` — all ten parameter ops (the six dense-head ops and the four
  conv ops) at the real forward activations and the chain cotangents driven by the emitted loss
  cotangent.

## Scope

* **Chain cotangent vs loss gradient.** Below the output layer the cotangents are the rendered
  chain (`mlpCotOut1`/`mlpCotOut0` in the head; `cnnChainCotW1`, `cnnChainCotW2`: relu masks,
  select-and-scatter pool-back, conv-back). The loss-gradient statement is `cnn_net_lossGrad`
  (`CnnParamGrad`), at the un-fused `*Grad` nodes these ops step by (`θ − lr·node`,
  `SmallParamGrad.convWeightSgd_eq_grad`) and with the pool's cotangent routed to one maximal cell
  of each window, as the rendered `select_and_scatter` does. `cnnChainCotW2` reads the pool
  backward as `maxPoolBackDenote`, which routes to the first maximal cell (`maxPool2Argmax`), the
  op's own choice, so it is the capstone's chain at that selection, ties included
  (`cnnChainCotW2_eq_sel`).
* **Cotangent subgraph ⇄ rendered SHlo.** The chain cotangents (`cnnChainCotW1/2`,
  `mlpCotOut0/1`) are proven = the rendered backward form in `CnnChainClose`
  (`cnnChainCotW1_eq`, `cnnChainCotW2_eq`) and `MlpTrainStep` (`mlpCotOut0_denote`,
  `mlpCotOut1_denote`); they are not pinned
  to the emitted `selectPos`/`dotOut`/`convBack`/`maxPoolBack` SHlo subgraph (as
  `MlpFold.cot0_den`/`MlpFold.cot1_den` do for the MLP).
* **Per-op `pretty` lexing** (shared with the whole suite) + **ℝ → Float32**.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CnnFold

/-! ## Convolution layers — the new `convWeightSgd`/`convBiasSgd` ops denote certified

`den (convWeightSgd … (.operand _ c))` is by construction
`flatten W − lr·conv2dWeightGrad(b,x)·c` (and likewise for the bias); pinning
`c` to the cotangent the chain delivers and applying the chain-certified conv
bridge gives `θ − lr·(certified ∂conv/∂θ · the-chain-cotangent)`. (The `den`
reduction is definitional — `rfl` — exactly as `LinFold.poc_weightSgd_den_eq`.) -/

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
the MLP's `(W₂,W₁,W₀)`). Every head op's `den` = certified is `SgdNode.denseW_den` /
`SgdNode.denseB_den` at that layer, and `cnn_train_step_tied_certified` states all six at the
real forward; the output-layer weight op is stated here on its own because the whole-loss fold
`cnn_W5_tied_totalloss` rewrites with it. -/

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
               (Mat.flatten W₅) (finProdFinEquiv (i, j)) k * dy k :=
  SgdNode.denseW_den aN "%W5" lrStr dyN _ W₅ b₅ _ lr i j

/-! ## Tie (dense head) — the top loss cotangent is the composed softmax-CE of the CONV forward

The `*_den` theorems above hold for a free top cotangent `dy` and a free pool output. The
renderer feeds the cotangent the emitted loss graph `sub(softmaxDiv(expe(logits)), onehot)` produces,
with `logits` the REAL conv-forward output `mnistCnnNoBnForward … x`. The lemma below pins that graph
to the composed softmax-CE gradient *of the conv forward* (the cnn analogue of `mlpLossCot_den`), and
the headline folds the dense output weight `W₅` to the whole-loss gradient `∂CE/∂W₅` — so the output
layer is tied forward(conv+dense)→softmax-CE→gradient. Every parameter op, at the chain cotangent
this `g` drives, is stated in the next section. -/

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
                  - oneHot nClasses label j :=
  StableHLO.softmaxCELossCot_den nlogN ohN _ label

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
      StableHLO.lossWeightGrad_eq_sum W₅ b₅
        (relu d1 (dense W₄ b₄ (relu d1 (dense W₃ b₃
          (maxPoolFlat c h w (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₂ b₂
            (relu (c*(2*h)*(2*w)) (flatConv (h := 2*h) (w := 2*w) W₁ b₁ x))))))))) label i j]
  -- both the threaded loss cotangent (`mnistCnnNoBnForward`) and the fold's `mnistLinear W₅ b₅ a₄`
  -- are `dense W₅ b₅ (relu … pool)` — unfold both to match.
  simp only [mnistCnnNoBnForward, mnistLinear, Function.comp_apply]

/-! ## The whole step — every parameter op at the real forward

The `*_den` theorems above hold for FREE activations (`ac1`/`ac2`/`hc2`, the pool output) and a
free cotangent. The capstone below instantiates them at the **real forward** (`ac1`/`hc1`/`hc2`/`ac2`
= the actual `conv₁`/`relu`/`conv₂`/`relu` outputs, `pool` their max-pool, `h3`/`h4` the dense
pre-activations) and the chain cotangents driven by the composed top cotangent
`g = softmax(mnistCnnNoBnForward x) − onehot` (`cnnLossCot_den`). Each of the ten parameter ops
denotes `θ − lr·(certified ∂layer/∂θ · c)` with `c` the rendered backward-chain cotangent: `g` for
`W₅`/`b₅`, `mlpCotOut1` for `W₄`/`b₄`, `mlpCotOut0` for `W₃`/`b₃`, `cnnChainCotW2` for conv₂ and
`cnnChainCotW1 W₂ hc1 cotW2` for conv₁ (it crosses one more conv-back). The loss-gradient form
below the output is `cnn_net_lossGrad` (see the module's Scope); at the output layer
`cnn_W5_tied_totalloss` folds `W₅` to `∂CE/∂W₅`. The correspondence between the hand-written
conv-backward SSA values and `cnnChainCotW1`/`cnnChainCotW2` is the per-op trust the whole suite
carries. -/

/-- **Whole cnn train step, tied.** All ten parameter ops — the dense head `W₅,b₅,W₄,b₄,W₃,b₃` and
    the conv kernels/biases `W₂,b₂,W₁,b₁` — at the real forward denote
    `θ − lr·(certified ∂layer/∂θ · c)` with `c` the rendered backward-chain cotangent driven by the
    emitted softmax-CE cotangent `g` (`mlpCotOut1`/`mlpCotOut0` in the head; `cnnChainCotW2`,
    `cnnChainCotW1` below it: relu masks, select-and-scatter pool-back, conv-back). The
    loss-gradient form is `cnn_net_lossGrad` (see the module's Scope). -/
theorem cnn_train_step_tied_certified {ic c h w d1 nClasses kH kW : Nat}
    (xN wN bN lrStr cotN : String)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c*h*w) d1) (b₃ : Vec d1) (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Tensor3 ic (2*h) (2*w)) (label : Fin nClasses)
    (lr : ℝ) :
    -- the forward runs in flat `Vec` space (`flatConv`); the backward/SGD read `Tensor3`
    -- activations (`conv2d`), so each conv activation has a `Vec` form (for `flatConv`/pool) and
    -- the `Tensor3.unflatten` of it (for `conv2d`/`convWeightSgd`/`cnnChainCot`).
    let xv : Vec (ic*(2*h)*(2*w)) := Tensor3.flatten x
    let hc1 : Vec (c*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₁ b₁ xv
    let ac1v : Vec (c*(2*h)*(2*w)) := relu (c*(2*h)*(2*w)) hc1
    let ac1 : Tensor3 c (2*h) (2*w) := Tensor3.unflatten ac1v
    let hc2 : Vec (c*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₂ b₂ ac1v
    let ac2v : Vec (c*(2*h)*(2*w)) := relu (c*(2*h)*(2*w)) hc2
    let ac2 : Tensor3 c (2*h) (2*w) := Tensor3.unflatten ac2v
    let pool : Vec (c*h*w) := maxPoolFlat c h w ac2v
    let h3 : Vec d1 := dense W₃ b₃ pool
    let h4 : Vec d1 := dense W₄ b₄ (relu d1 h3)
    let g : Vec nClasses := fun k =>
      softmax nClasses (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ xv) k - oneHot nClasses label k
    let cotH4 : Vec d1 := (mlpCotOut1 W₅ h4).denote g
    let cotH3 : Vec d1 := (mlpCotOut0 W₄ W₅ h3 h4).denote g
    let cotW2 := cnnChainCotW2 W₃ W₄ W₅ h3 h4 ac2 hc2 g
    -- dense head (output layer first)
    DenseWSgdTied xN wN lrStr cotN (relu d1 h4) W₅ b₅ g lr
  ∧ DenseBSgdTied bN lrStr cotN W₅ (relu d1 h4) b₅ g lr
  ∧ DenseWSgdTied xN wN lrStr cotN (relu d1 h3) W₄ b₄ cotH4 lr
  ∧ DenseBSgdTied bN lrStr cotN W₄ (relu d1 h3) b₄ cotH4 lr
  ∧ DenseWSgdTied xN wN lrStr cotN pool W₃ b₃ cotH3 lr
  ∧ DenseBSgdTied bN lrStr cotN W₃ pool b₃ cotH3 lr
  -- conv₂, conv₁
  ∧ ConvWSgdTied xN wN lrStr cotN b₂ ac1 W₂ cotW2 lr
  ∧ ConvBSgdTied bN lrStr cotN W₂ ac1 b₂ cotW2 lr
  ∧ ConvWSgdTied xN wN lrStr cotN b₁ x W₁ (cnnChainCotW1 W₂ hc1 cotW2) lr
  ∧ ConvBSgdTied bN lrStr cotN W₁ x b₁ (cnnChainCotW1 W₂ hc1 cotW2) lr :=
  ⟨denseWSgdTied_holds, denseBSgdTied_holds, denseWSgdTied_holds, denseBSgdTied_holds,
    denseWSgdTied_holds, denseBSgdTied_holds, convWSgdTied_holds, convBSgdTied_holds,
    convWSgdTied_holds, convBSgdTied_holds⟩

end Proofs.CnnFold
