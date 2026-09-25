import LeanMlir.Proofs.Nets.Small.CifarFold
import LeanMlir.Proofs.Foundation.SgdNodes

/-! # PoC: the cifar8-bn (Chapter 4 deeper, 8-conv per-channel BN) §1a TIE

cifar8's §1a tie + a BN-back at every conv. The backward
chain alternates **BN-output cotangent** `dyBnᵢ` (relu-masked — fed to the γ/β ops) and **conv-output
cotangent** `cotCᵢ` (`bnPerChannelTensor3GradInput` of `dyBnᵢ` — fed to the conv W/b ops), repeated
over 4 conv→conv→pool stages, crossing each pool as conv-back then maxpool-back.

**Zero new ops/bridges/constructors.** Conv ties reuse `CifarPoC.convW_den`/`convB_den`; BN ties reuse
`CifarBnPoC.bnGamma_den`/`bnBeta_den`; dense head + loss-cot reuse `Cifar8PoC`/cifar. All 38 params
(8 conv W/b + 8 BN γ/β + 3 dense) fold with the generics — the cifar8-bn lesson applied to the tie.

## Honest residual (same as the rest of the suite)
* Conv/BN backward rendered hand-written (cotangent SSA ↔ chain-cot per-op trust); per-op `pretty`
  lexing; BN `0 < ε` smoothness; ℝ → Float32.
-/

open Proofs Proofs.StableHLO Proofs.IR
open Proofs.CifarBnPoC (bnSgdPairTied_holds)

namespace Proofs.Cifar8BnPoC

/-- **The emitted loss-cotangent graph denotes the softmax-CE gradient of the cifar8-bn forward.** -/
theorem cifar8BnLossCot_den {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (nlogN ohN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) (label : Fin nClasses) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN
            (cifarCnnBn8Forward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
              W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x))))
          (.operand ohN (oneHot nClasses label)))
      = fun j => softmax nClasses (cifarCnnBn8Forward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
                    W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈
                    W₉ b₉ Wa ba Wb bb x) j - oneHot nClasses label j :=
  StableHLO.softmaxCELossCot_den nlogN ohN _ label

/-- **Whole cifar8-bn conv+BN tail, tied.** All 32 conv/BN params (8 conv `W`+`b`, 8 BN `γ`+`β`), at the
    real cifar8-bn forward and the composed softmax-CE cotangent, denote the certified loss-descent
    step. The conv ops are fed the BN-back cotangents `cotC1–8`; the BN ops the relu-masked
    cotangents `dyBn1–8`; both are the genuine cifar8-bn backward chain (cifar8's chain + a BN-back at
    every conv). Dense head + loss-cot reuse `Cifar8PoC`/cifar. -/
theorem cifar8Bn_convbn_tied_certified {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Tensor3 ic (2*(2*(2*(2*h)))) (2*(2*(2*(2*w))))) (label : Fin nClasses) (lr : ℝ) :
    let xv : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := Tensor3.flatten x
    -- stage 1 (conv₁/conv₂ at s1, c1)
    let cc1 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := flatConv (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) W₁ b₁ xv
    let bn1o : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := bnPerChannelTensor3 c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))) ε₁ γ₁ β₁ cc1
    let r1 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := relu (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) bn1o
    let r1t : Tensor3 c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))) := Tensor3.unflatten r1
    let cc2 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := flatConv (h := 2*(2*(2*(2*h)))) (w := 2*(2*(2*(2*w)))) W₂ b₂ r1
    let bn2o : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := bnPerChannelTensor3 c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))) ε₂ γ₂ β₂ cc2
    let r2 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := relu (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) bn2o
    let r2t : Tensor3 c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))) := Tensor3.unflatten r2
    let zp1 : Vec (c1*(2*(2*(2*h)))*(2*(2*(2*w)))) := maxPoolFlat c1 (2*(2*(2*h))) (2*(2*(2*w))) r2
    let zp1t : Tensor3 c1 (2*(2*(2*h))) (2*(2*(2*w))) := Tensor3.unflatten zp1
    -- stage 2 (conv₃/conv₄ at s2, c2)
    let cc3 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := flatConv (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) W₃ b₃ zp1
    let bn3o : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := bnPerChannelTensor3 c2 (2*(2*(2*h))) (2*(2*(2*w))) ε₃ γ₃ β₃ cc3
    let r3 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := relu (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) bn3o
    let r3t : Tensor3 c2 (2*(2*(2*h))) (2*(2*(2*w))) := Tensor3.unflatten r3
    let cc4 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := flatConv (h := 2*(2*(2*h))) (w := 2*(2*(2*w))) W₄ b₄ r3
    let bn4o : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := bnPerChannelTensor3 c2 (2*(2*(2*h))) (2*(2*(2*w))) ε₄ γ₄ β₄ cc4
    let r4 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := relu (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) bn4o
    let r4t : Tensor3 c2 (2*(2*(2*h))) (2*(2*(2*w))) := Tensor3.unflatten r4
    let zp2 : Vec (c2*(2*(2*h))*(2*(2*w))) := maxPoolFlat c2 (2*(2*h)) (2*(2*w)) r4
    let zp2t : Tensor3 c2 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten zp2
    -- stage 3 (conv₅/conv₆ at s3, c3)
    let cc5 : Vec (c3*(2*(2*h))*(2*(2*w))) := flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₅ b₅ zp2
    let bn5o : Vec (c3*(2*(2*h))*(2*(2*w))) := bnPerChannelTensor3 c3 (2*(2*h)) (2*(2*w)) ε₅ γ₅ β₅ cc5
    let r5 : Vec (c3*(2*(2*h))*(2*(2*w))) := relu (c3*(2*(2*h))*(2*(2*w))) bn5o
    let r5t : Tensor3 c3 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten r5
    let cc6 : Vec (c3*(2*(2*h))*(2*(2*w))) := flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₆ b₆ r5
    let bn6o : Vec (c3*(2*(2*h))*(2*(2*w))) := bnPerChannelTensor3 c3 (2*(2*h)) (2*(2*w)) ε₆ γ₆ β₆ cc6
    let r6 : Vec (c3*(2*(2*h))*(2*(2*w))) := relu (c3*(2*(2*h))*(2*(2*w))) bn6o
    let r6t : Tensor3 c3 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten r6
    let zp3 : Vec (c3*(2*h)*(2*w)) := maxPoolFlat c3 (2*h) (2*w) r6
    let zp3t : Tensor3 c3 (2*h) (2*w) := Tensor3.unflatten zp3
    -- stage 4 (conv₇/conv₈ at s4, c4)
    let cc7 : Vec (c4*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₇ b₇ zp3
    let bn7o : Vec (c4*(2*h)*(2*w)) := bnPerChannelTensor3 c4 (2*h) (2*w) ε₇ γ₇ β₇ cc7
    let r7 : Vec (c4*(2*h)*(2*w)) := relu (c4*(2*h)*(2*w)) bn7o
    let r7t : Tensor3 c4 (2*h) (2*w) := Tensor3.unflatten r7
    let cc8 : Vec (c4*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₈ b₈ r7
    let bn8o : Vec (c4*(2*h)*(2*w)) := bnPerChannelTensor3 c4 (2*h) (2*w) ε₈ γ₈ β₈ cc8
    let r8 : Vec (c4*(2*h)*(2*w)) := relu (c4*(2*h)*(2*w)) bn8o
    let r8t : Tensor3 c4 (2*h) (2*w) := Tensor3.unflatten r8
    let zp4 : Vec (c4*h*w) := maxPoolFlat c4 h w r8
    let h9 : Vec d1 := dense W₉ b₉ zp4
    let ha : Vec d1 := dense Wa ba (relu d1 h9)
    let g : Vec nClasses := fun k =>
      softmax nClasses (cifarCnnBn8Forward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
        W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb xv) k
        - oneHot nClasses label k
    let cpool4 : Vec (c4*h*w) := (cnnDenseHeadCot W₉ Wa Wb h9 ha).denote g
    -- BN-output cotangents (relu-masked, γ/β) and conv-output cotangents (BN-back, W/b)
    let dyBn8 : Vec (c4*(2*h)*(2*w)) := fun i => if bn8o i > 0
      then (Back3.maxpool (c₁ := c4) (h₁ := h) (w₁ := w) r8t Back3.cot).flatDenote cpool4 i else 0
    let cotC8 : Vec (c4*(2*h)*(2*w)) := bnPerChannelTensor3GradInput c4 (2*h) (2*w) ε₈ γ₈ cc8 dyBn8
    let dyBn7 : Vec (c4*(2*h)*(2*w)) := fun i => if bn7o i > 0
      then (Back3.conv (c₁ := c4) (h₁ := 2*h) (w₁ := 2*w) W₈ Back3.cot).flatDenote cotC8 i else 0
    let cotC7 : Vec (c4*(2*h)*(2*w)) := bnPerChannelTensor3GradInput c4 (2*h) (2*w) ε₇ γ₇ cc7 dyBn7
    let dyBn6 : Vec (c3*(2*(2*h))*(2*(2*w))) := fun i => if bn6o i > 0
      then (Back3.maxpool (c₁ := c3) (h₁ := 2*h) (w₁ := 2*w) r6t Back3.cot).flatDenote
             ((Back3.conv (c₁ := c4) (h₁ := 2*h) (w₁ := 2*w) W₇ Back3.cot).flatDenote cotC7) i else 0
    let cotC6 : Vec (c3*(2*(2*h))*(2*(2*w))) := bnPerChannelTensor3GradInput c3 (2*(2*h)) (2*(2*w)) ε₆ γ₆ cc6 dyBn6
    let dyBn5 : Vec (c3*(2*(2*h))*(2*(2*w))) := fun i => if bn5o i > 0
      then (Back3.conv (c₁ := c3) (h₁ := 2*(2*h)) (w₁ := 2*(2*w)) W₆ Back3.cot).flatDenote cotC6 i else 0
    let cotC5 : Vec (c3*(2*(2*h))*(2*(2*w))) := bnPerChannelTensor3GradInput c3 (2*(2*h)) (2*(2*w)) ε₅ γ₅ cc5 dyBn5
    let dyBn4 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := fun i => if bn4o i > 0
      then (Back3.maxpool (c₁ := c2) (h₁ := 2*(2*h)) (w₁ := 2*(2*w)) r4t Back3.cot).flatDenote
             ((Back3.conv (c₁ := c3) (h₁ := 2*(2*h)) (w₁ := 2*(2*w)) W₅ Back3.cot).flatDenote cotC5) i else 0
    let cotC4 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := bnPerChannelTensor3GradInput c2 (2*(2*(2*h))) (2*(2*(2*w))) ε₄ γ₄ cc4 dyBn4
    let dyBn3 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := fun i => if bn3o i > 0
      then (Back3.conv (c₁ := c2) (h₁ := 2*(2*(2*h))) (w₁ := 2*(2*(2*w))) W₄ Back3.cot).flatDenote cotC4 i else 0
    let cotC3 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := bnPerChannelTensor3GradInput c2 (2*(2*(2*h))) (2*(2*(2*w))) ε₃ γ₃ cc3 dyBn3
    let dyBn2 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := fun i => if bn2o i > 0
      then (Back3.maxpool (c₁ := c1) (h₁ := 2*(2*(2*h))) (w₁ := 2*(2*(2*w))) r2t Back3.cot).flatDenote
             ((Back3.conv (c₁ := c2) (h₁ := 2*(2*(2*h))) (w₁ := 2*(2*(2*w))) W₃ Back3.cot).flatDenote cotC3) i else 0
    let cotC2 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := bnPerChannelTensor3GradInput c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))) ε₂ γ₂ cc2 dyBn2
    let dyBn1 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := fun i => if bn1o i > 0
      then (Back3.conv (c₁ := c1) (h₁ := 2*(2*(2*(2*h)))) (w₁ := 2*(2*(2*(2*w)))) W₂ Back3.cot).flatDenote cotC2 i else 0
    let cotC1 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := bnPerChannelTensor3GradInput c1 (2*(2*(2*(2*h)))) (2*(2*(2*(2*w)))) ε₁ γ₁ cc1 dyBn1
    -- conv₁ + bn₁
    ConvWSgdTied xN wN lrStr cotN b₁ x W₁ cotC1 lr
  ∧ ConvBSgdTied bN lrStr cotN W₁ x b₁ cotC1 lr
  ∧ CifarBnPoC.BnSgdPairTied gN vN bN epsStr lrStr cotN ε₁ γ₁ β₁ cc1 dyBn1 lr
  -- conv₂ + bn₂
  ∧ ConvWSgdTied xN wN lrStr cotN b₂ r1t W₂ cotC2 lr
  ∧ ConvBSgdTied bN lrStr cotN W₂ r1t b₂ cotC2 lr
  ∧ CifarBnPoC.BnSgdPairTied gN vN bN epsStr lrStr cotN ε₂ γ₂ β₂ cc2 dyBn2 lr
  -- conv₃ + bn₃
  ∧ ConvWSgdTied xN wN lrStr cotN b₃ zp1t W₃ cotC3 lr
  ∧ ConvBSgdTied bN lrStr cotN W₃ zp1t b₃ cotC3 lr
  ∧ CifarBnPoC.BnSgdPairTied gN vN bN epsStr lrStr cotN ε₃ γ₃ β₃ cc3 dyBn3 lr
  -- conv₄ + bn₄
  ∧ ConvWSgdTied xN wN lrStr cotN b₄ r3t W₄ cotC4 lr
  ∧ ConvBSgdTied bN lrStr cotN W₄ r3t b₄ cotC4 lr
  ∧ CifarBnPoC.BnSgdPairTied gN vN bN epsStr lrStr cotN ε₄ γ₄ β₄ cc4 dyBn4 lr
  -- conv₅ + bn₅
  ∧ ConvWSgdTied xN wN lrStr cotN b₅ zp2t W₅ cotC5 lr
  ∧ ConvBSgdTied bN lrStr cotN W₅ zp2t b₅ cotC5 lr
  ∧ CifarBnPoC.BnSgdPairTied gN vN bN epsStr lrStr cotN ε₅ γ₅ β₅ cc5 dyBn5 lr
  -- conv₆ + bn₆
  ∧ ConvWSgdTied xN wN lrStr cotN b₆ r5t W₆ cotC6 lr
  ∧ ConvBSgdTied bN lrStr cotN W₆ r5t b₆ cotC6 lr
  ∧ CifarBnPoC.BnSgdPairTied gN vN bN epsStr lrStr cotN ε₆ γ₆ β₆ cc6 dyBn6 lr
  -- conv₇ + bn₇
  ∧ ConvWSgdTied xN wN lrStr cotN b₇ zp3t W₇ cotC7 lr
  ∧ ConvBSgdTied bN lrStr cotN W₇ zp3t b₇ cotC7 lr
  ∧ CifarBnPoC.BnSgdPairTied gN vN bN epsStr lrStr cotN ε₇ γ₇ β₇ cc7 dyBn7 lr
  -- conv₈ + bn₈
  ∧ ConvWSgdTied xN wN lrStr cotN b₈ r7t W₈ cotC8 lr
  ∧ ConvBSgdTied bN lrStr cotN W₈ r7t b₈ cotC8 lr
  ∧ CifarBnPoC.BnSgdPairTied gN vN bN epsStr lrStr cotN ε₈ γ₈ β₈ cc8 dyBn8 lr := by
  exact ⟨convWSgdTied_holds, convBSgdTied_holds, bnSgdPairTied_holds, convWSgdTied_holds,
    convBSgdTied_holds, bnSgdPairTied_holds, convWSgdTied_holds, convBSgdTied_holds,
    bnSgdPairTied_holds, convWSgdTied_holds, convBSgdTied_holds, bnSgdPairTied_holds,
    convWSgdTied_holds, convBSgdTied_holds, bnSgdPairTied_holds, convWSgdTied_holds,
    convBSgdTied_holds, bnSgdPairTied_holds, convWSgdTied_holds, convBSgdTied_holds,
    bnSgdPairTied_holds, convWSgdTied_holds, convBSgdTied_holds, bnSgdPairTied_holds⟩

end Proofs.Cifar8BnPoC
