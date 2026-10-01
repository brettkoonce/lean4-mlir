import LeanMlir.Proofs.Nets.Small.Cifar8StepTie

/-! # The cifar8 step tie at its UN-FUSED gradient nodes — the packed `cifar8w_*` arms

`Cifar8Tie.cifar8_train_step_tied_certified` ties the fused-SGD `cifar8_train_step.mlir`, which no
trainer runs. The packed wide no-BN arms (`cifar8w_{sgd,mom,adam}_train_step.mlir`, from
`cifar8AdamTrainStepText`) emit the same forward and backward chain feeding `*Grad` ops to a
separate optimizer. This file states those nodes, each at the same chain cotangent: all 22
parameter tensors, via `GradNode` (Foundation/SgdNodes.lean). The optimizer update is outside the
statement.

## Scope (as the fused tie)
* Below the output layer the cotangents are the rendered chain. `cifar8_net_lossGrad`
  (`Cifar8ParamGrad`) states each node as the loss gradient in its parameter, with each pool's
  cotangent routed to one maximal cell, as the rendered `select_and_scatter` does; this chain's
  `maxPoolBackDenote` routes it to the first maximal cell (`maxPool2Argmax`), the op's own choice,
  so it is the capstone's chain at that selection (`CnnFold.cnnChainCotW2_eq_sel`,
  `CifarFold.cifarChainCotW2_eq_sel`).
* Conv backward rendered hand-written (cotangent SSA ↔ chain-cot per-op trust); per-op `pretty`
  lexing; ℝ → Float32.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.Cifar8TieG

/-- **Whole cifar8 train step at its gradient nodes.** All 22 parameter tensors (8 conv `W`+`b`,
    the dense head `W₉,b₉,Wa,ba,Wb,bb`), at the real cifar8 forward: each emitted `*Grad` node
    denotes the certified per-layer Jacobian contracted with the rendered backward-chain cotangent,
    driven by the composed softmax-CE cotangent `g` — the fused tie's chain, node for node. -/
theorem cifar8_train_step_tiedG {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (xN cotN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Tensor3 ic (2*(2*(2*(2*h)))) (2*(2*(2*(2*w))))) (label : Fin nClasses) :
    let xv : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := Tensor3.flatten x
    -- stage 1 (conv₁/conv₂ at s1, c1)
    let cc1 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := flatConv (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) W₁ b₁ xv
    let r1 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := relu (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) cc1
    let r1t : Tensor3 c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))) := Tensor3.unflatten r1
    let cc2 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := flatConv (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) W₂ b₂ r1
    let r2 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := relu (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) cc2
    let r2t : Tensor3 c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))) := Tensor3.unflatten r2
    let zp1 : Vec (c1*(2*(2*(2*h)))*(2*(2*(2*w)))) := maxPoolFlat c1 (2*(2*(2*h))) (2*(2*(2*w))) r2
    let zp1t : Tensor3 c1 (2*(2*(2*h))) (2*(2*(2*w))) := Tensor3.unflatten zp1
    -- stage 2 (conv₃/conv₄ at s2, c2)
    let cc3 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := flatConv (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) W₃ b₃ zp1
    let r3 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := relu (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) cc3
    let r3t : Tensor3 c2 (2*(2*(2*h))) (2*(2*(2*w))) := Tensor3.unflatten r3
    let cc4 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := flatConv (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) W₄ b₄ r3
    let r4 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := relu (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) cc4
    let r4t : Tensor3 c2 (2*(2*(2*h))) (2*(2*(2*w))) := Tensor3.unflatten r4
    let zp2 : Vec (c2*(2*(2*h))*(2*(2*w))) := maxPoolFlat c2 (2*(2*h)) (2*(2*w)) r4
    let zp2t : Tensor3 c2 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten zp2
    -- stage 3 (conv₅/conv₆ at s3, c3)
    let cc5 : Vec (c3*(2*(2*h))*(2*(2*w))) := flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₅ b₅ zp2
    let r5 : Vec (c3*(2*(2*h))*(2*(2*w))) := relu (c3*(2*(2*h))*(2*(2*w))) cc5
    let r5t : Tensor3 c3 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten r5
    let cc6 : Vec (c3*(2*(2*h))*(2*(2*w))) := flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₆ b₆ r5
    let r6 : Vec (c3*(2*(2*h))*(2*(2*w))) := relu (c3*(2*(2*h))*(2*(2*w))) cc6
    let r6t : Tensor3 c3 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten r6
    let zp3 : Vec (c3*(2*h)*(2*w)) := maxPoolFlat c3 (2*h) (2*w) r6
    let zp3t : Tensor3 c3 (2*h) (2*w) := Tensor3.unflatten zp3
    -- stage 4 (conv₇/conv₈ at s4, c4)
    let cc7 : Vec (c4*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₇ b₇ zp3
    let r7 : Vec (c4*(2*h)*(2*w)) := relu (c4*(2*h)*(2*w)) cc7
    let r7t : Tensor3 c4 (2*h) (2*w) := Tensor3.unflatten r7
    let cc8 : Vec (c4*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₈ b₈ r7
    let r8 : Vec (c4*(2*h)*(2*w)) := relu (c4*(2*h)*(2*w)) cc8
    let r8t : Tensor3 c4 (2*h) (2*w) := Tensor3.unflatten r8
    let zp4 : Vec (c4*h*w) := maxPoolFlat c4 h w r8
    let h9 : Vec d1 := dense W₉ b₉ zp4
    let ha : Vec d1 := dense Wa ba (relu d1 h9)
    let g : Vec nClasses := fun k =>
      softmax nClasses (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
        W₉ b₉ Wa ba Wb bb xv) k - oneHot nClasses label k
    -- the 8 conv chain cotangents (all reused constructors)
    let cotC8 : Vec (c4*(2*h)*(2*w)) := cnnChainCotW2 W₉ Wa Wb h9 ha r8t cc8 g
    let cotC7 : Vec (c4*(2*h)*(2*w)) := cnnChainCotW1 W₈ cc7 cotC8
    let cotC6 : Vec (c3*(2*(2*h))*(2*(2*w))) := CifarFold.cifarChainCotW2 W₇ r6t cc6 cotC7
    let cotC5 : Vec (c3*(2*(2*h))*(2*(2*w))) := cnnChainCotW1 W₆ cc5 cotC6
    let cotC4 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := CifarFold.cifarChainCotW2 W₅ r4t cc4 cotC5
    let cotC3 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := cnnChainCotW1 W₄ cc3 cotC4
    let cotC2 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := CifarFold.cifarChainCotW2 W₃ r2t cc2 cotC3
    let cotC1 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := cnnChainCotW1 W₂ cc1 cotC2
    -- conv₁
    GradNode.ConvWGradTied xN cotN b₁ x W₁ cotC1
  ∧ GradNode.ConvBGradTied cotN W₁ x b₁ cotC1
  -- conv₂
  ∧ GradNode.ConvWGradTied xN cotN b₂ r1t W₂ cotC2
  ∧ GradNode.ConvBGradTied cotN W₂ r1t b₂ cotC2
  -- conv₃
  ∧ GradNode.ConvWGradTied xN cotN b₃ zp1t W₃ cotC3
  ∧ GradNode.ConvBGradTied cotN W₃ zp1t b₃ cotC3
  -- conv₄
  ∧ GradNode.ConvWGradTied xN cotN b₄ r3t W₄ cotC4
  ∧ GradNode.ConvBGradTied cotN W₄ r3t b₄ cotC4
  -- conv₅
  ∧ GradNode.ConvWGradTied xN cotN b₅ zp2t W₅ cotC5
  ∧ GradNode.ConvBGradTied cotN W₅ zp2t b₅ cotC5
  -- conv₆
  ∧ GradNode.ConvWGradTied xN cotN b₆ r5t W₆ cotC6
  ∧ GradNode.ConvBGradTied cotN W₆ r5t b₆ cotC6
  -- conv₇
  ∧ GradNode.ConvWGradTied xN cotN b₇ zp3t W₇ cotC7
  ∧ GradNode.ConvBGradTied cotN W₇ zp3t b₇ cotC7
  -- conv₈
  ∧ GradNode.ConvWGradTied xN cotN b₈ r7t W₈ cotC8
  ∧ GradNode.ConvBGradTied cotN W₈ r7t b₈ cotC8
  -- dense head
  ∧ GradNode.DenseWGradTied xN cotN zp4 W₉ b₉ ((mlpCotOut0 Wa Wb h9 ha).denote g)
  ∧ GradNode.DenseBGradTied cotN W₉ zp4 b₉ ((mlpCotOut0 Wa Wb h9 ha).denote g)
  ∧ GradNode.DenseWGradTied xN cotN (relu d1 h9) Wa ba ((mlpCotOut1 Wb ha).denote g)
  ∧ GradNode.DenseBGradTied cotN Wa (relu d1 h9) ba ((mlpCotOut1 Wb ha).denote g)
  ∧ GradNode.DenseWGradTied xN cotN (relu d1 ha) Wb bb g
  ∧ GradNode.DenseBGradTied cotN Wb (relu d1 ha) bb g := by
  exact ⟨GradNode.convWGradTied_holds, GradNode.convBGradTied_holds, GradNode.convWGradTied_holds, GradNode.convBGradTied_holds,
    GradNode.convWGradTied_holds, GradNode.convBGradTied_holds, GradNode.convWGradTied_holds, GradNode.convBGradTied_holds,
    GradNode.convWGradTied_holds, GradNode.convBGradTied_holds, GradNode.convWGradTied_holds, GradNode.convBGradTied_holds,
    GradNode.convWGradTied_holds, GradNode.convBGradTied_holds, GradNode.convWGradTied_holds, GradNode.convBGradTied_holds,
    GradNode.denseWGradTied_holds, GradNode.denseBGradTied_holds, GradNode.denseWGradTied_holds, GradNode.denseBGradTied_holds,
    GradNode.denseWGradTied_holds, GradNode.denseBGradTied_holds⟩


end Proofs.Cifar8TieG
