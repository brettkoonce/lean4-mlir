import LeanMlir.Proofs.CifarBnFaithfulPoC
import LeanMlir.Proofs.Cifar8FaithfulPoC

/-! # PoC: the CIFAR-BN (Chapter 5, per-channel BN) §1a TIE — tied through the real forward

The BatchNorm peer of `CifarFaithfulPoC`'s §1a tie (`cifar_conv_tied_certified` etc.). cifar-bn is
cifar (ch5) with a per-channel `bnPerChannelTensor3` inserted between every conv and its ReLU, so the
backward chain alternates **BN-output cotangent** (relu-masked — what the γ/β ops consume) and
**conv-output cotangent** (the BN input-VJP `bnPerChannelTensor3_grad_input` of it — what the conv
W/b ops consume). The cross-pool₁ step is cifar's `cifarChainCotW2` move (conv₃-back then maxpool₁-back)
with a BN-back in front; the within-stage steps are conv-back + relu-mask + BN-back (the
`Cifar8Close` BN-chain recipe, here at the 2-stage cifar dims).

**Zero new ops, zero new bridges.** The conv W/b ties reuse `CifarPoC.convW_den`/`convB_den`
(generic in the cotangent); the BN γ/β ties reuse `CifarBnPoC.bnGamma_den`/`bnBeta_den` (generic in
the cotangent); the dense head + loss-cot + total-loss fold mirror cifar verbatim. The only new
content is the per-net BN backward-chain cotangents + the capstones below.

## Honest residual (same boundary as cifar)
* The block backward is rendered hand-written (cotangent SSA ↔ chain-cot per-op trust); per-op
  `pretty` lexing; BN `0 < ε` smoothness; ℝ → Float32.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CifarBnPoC

/-- **The emitted loss-cotangent graph denotes the softmax-CE gradient of the cifar-BN forward.**
    Copy of `CifarPoC.cifarLossCot_den`, swapping `cifarCnnForward → cifarCnnBnForward`. -/
theorem cifarBnLossCot_den {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (nlogN ohN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) (label : Fin nClasses) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN
            (cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
              W₅ b₅ W₆ b₆ W₇ b₇ x))))
          (.operand ohN (oneHot nClasses label)))
      = fun j => softmax nClasses (cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
                    W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇ x) j - oneHot nClasses label j := by
  funext j; simp only [den, softmax]

/-- **Dense output weight `W₇`, tied to the WHOLE softmax-CE loss through the cifar-BN forward.**
    The dense head is identical to cifar (no BN), so this is `CifarPoC.cifar_W7_tied_totalloss` with
    the BN forward — the activation feeding `W₇` is the real cifar-BN pool₂ output. -/
theorem cifarBn_W7_tied_totalloss {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (aN lrStr dyN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic*(2*(2*h))*(2*(2*w)))) (a₆ : Vec d1) (label : Fin nClasses)
    (lr : ℝ) (i : Fin d1) (j : Fin nClasses)
    (hlog : cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
              W₅ b₅ W₆ b₆ W₇ b₇ x = mnistLinear W₇ b₇ a₆) :
    den (SHlo.weightSgd aN "%W7" lrStr a₆ W₇ lr
          (.operand dyN (fun k => softmax nClasses
              (cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
                W₅ b₅ W₆ b₆ W₇ b₇ x) k - oneHot nClasses label k)))
        (finProdFinEquiv (i, j))
      = W₇ i j - lr * pdiv (fun v : Vec (d1 * nClasses) => fun _ : Fin 1 =>
            crossEntropy nClasses (dense (Mat.unflatten v) b₇ a₆) label)
          (Mat.flatten W₇) (finProdFinEquiv (i, j)) 0 := by
  rw [Cifar8PoC.denseW_den aN "%W7" lrStr dyN a₆ W₇ b₇
        (fun k => softmax nClasses (cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
            W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ W₆ b₆ W₇ b₇ x) k - oneHot nClasses label k) lr i j,
      mlp_output_total_loss_grad W₇ b₇ a₆ label i j, hlog]

set_option maxRecDepth 8000 in
/-- **Whole cifar-BN conv+BN tail, tied.** All 16 conv/BN params (4 conv `W`+`b`, 4 BN `γ`+`β`), at
    the real cifar-BN forward and the composed softmax-CE cotangent, denote the certified loss-descent
    step. The conv ops are fed the BN-back cotangents `cotC1–4`; the BN ops the relu-masked
    cotangents `dyBn1–4`; both are the genuine cifar-BN backward chain (the cifar chain + a BN-back at
    every conv). Together with the dense head (`cifarBn_W7_tied_totalloss` + `CifarPoC.{dW5,dW6,db5,
    db6,db7}` at `g`) the WHOLE cifar-BN train step is den-composed forward→loss→backward. -/
theorem cifarBn_convbn_tied_certified {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Vec c2)
    (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Tensor3 ic (2*(2*h)) (2*(2*w))) (label : Fin nClasses) (lr : ℝ) :
    -- forward activations (the real cifar-BN forward, in flat `Vec` space)
    let xv : Vec (ic*(2*(2*h))*(2*(2*w))) := Tensor3.flatten x
    let cc1 : Vec (c1*(2*(2*h))*(2*(2*w))) := flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁ xv
    let bn1o : Vec (c1*(2*(2*h))*(2*(2*w))) := bnPerChannelTensor3 c1 (2*(2*h)) (2*(2*w)) ε₁ γ₁ β₁ cc1
    let r1 : Vec (c1*(2*(2*h))*(2*(2*w))) := relu (c1*(2*(2*h))*(2*(2*w))) bn1o
    let r1t : Tensor3 c1 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten r1
    let cc2 : Vec (c1*(2*(2*h))*(2*(2*w))) := flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂ r1
    let bn2o : Vec (c1*(2*(2*h))*(2*(2*w))) := bnPerChannelTensor3 c1 (2*(2*h)) (2*(2*w)) ε₂ γ₂ β₂ cc2
    let r2 : Vec (c1*(2*(2*h))*(2*(2*w))) := relu (c1*(2*(2*h))*(2*(2*w))) bn2o
    let r2t : Tensor3 c1 (2*(2*h)) (2*(2*w)) := Tensor3.unflatten r2
    let zp1 : Vec (c1*(2*h)*(2*w)) := maxPoolFlat c1 (2*h) (2*w) r2
    let zp1t : Tensor3 c1 (2*h) (2*w) := Tensor3.unflatten zp1
    let cc3 : Vec (c2*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₃ b₃ zp1
    let bn3o : Vec (c2*(2*h)*(2*w)) := bnPerChannelTensor3 c2 (2*h) (2*w) ε₃ γ₃ β₃ cc3
    let r3 : Vec (c2*(2*h)*(2*w)) := relu (c2*(2*h)*(2*w)) bn3o
    let r3t : Tensor3 c2 (2*h) (2*w) := Tensor3.unflatten r3
    let cc4 : Vec (c2*(2*h)*(2*w)) := flatConv (h := 2*h) (w := 2*w) W₄ b₄ r3
    let bn4o : Vec (c2*(2*h)*(2*w)) := bnPerChannelTensor3 c2 (2*h) (2*w) ε₄ γ₄ β₄ cc4
    let r4 : Vec (c2*(2*h)*(2*w)) := relu (c2*(2*h)*(2*w)) bn4o
    let r4t : Tensor3 c2 (2*h) (2*w) := Tensor3.unflatten r4
    let zp2 : Vec (c2*h*w) := maxPoolFlat c2 h w r4
    let h5 : Vec d1 := dense W₅ b₅ zp2
    let h6 : Vec d1 := dense W₆ b₆ (relu d1 h5)
    let g : Vec nClasses := fun k =>
      softmax nClasses (cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
        W₅ b₅ W₆ b₆ W₇ b₇ xv) k - oneHot nClasses label k
    let cpool2 : Vec (c2*h*w) := (cnnDenseHeadCot W₅ W₆ W₇ h5 h6).denote g
    -- BN-output cotangents (relu-masked; consumed by the γ/β ops) and conv-output cotangents
    -- (BN input-VJP of them; consumed by the W/b ops), the genuine cifar-BN backward chain
    let dyBn4 : Vec (c2*(2*h)*(2*w)) := fun i => if bn4o i > 0
      then (Back3.maxpool (c₁ := c2) (h₁ := h) (w₁ := w) r4t Back3.cot).flatDenote cpool2 i else 0
    let cotC4 : Vec (c2*(2*h)*(2*w)) := bnPerChannelTensor3_grad_input c2 (2*h) (2*w) ε₄ γ₄ cc4 dyBn4
    let dyBn3 : Vec (c2*(2*h)*(2*w)) := fun i => if bn3o i > 0
      then (Back3.conv (c₁ := c2) (h₁ := 2*h) (w₁ := 2*w) W₄ Back3.cot).flatDenote cotC4 i else 0
    let cotC3 : Vec (c2*(2*h)*(2*w)) := bnPerChannelTensor3_grad_input c2 (2*h) (2*w) ε₃ γ₃ cc3 dyBn3
    let dyBn2 : Vec (c1*(2*(2*h))*(2*(2*w))) := fun i => if bn2o i > 0
      then (Back3.maxpool (c₁ := c1) (h₁ := 2*h) (w₁ := 2*w) r2t Back3.cot).flatDenote
             ((Back3.conv (c₁ := c2) (h₁ := 2*h) (w₁ := 2*w) W₃ Back3.cot).flatDenote cotC3) i else 0
    let cotC2 : Vec (c1*(2*(2*h))*(2*(2*w))) :=
      bnPerChannelTensor3_grad_input c1 (2*(2*h)) (2*(2*w)) ε₂ γ₂ cc2 dyBn2
    let dyBn1 : Vec (c1*(2*(2*h))*(2*(2*w))) := fun i => if bn1o i > 0
      then (Back3.conv (c₁ := c1) (h₁ := 2*(2*h)) (w₁ := 2*(2*w)) W₂ Back3.cot).flatDenote cotC2 i else 0
    let cotC1 : Vec (c1*(2*(2*h))*(2*(2*w))) :=
      bnPerChannelTensor3_grad_input c1 (2*(2*h)) (2*(2*w)) ε₁ γ₁ cc1 dyBn1
    -- conv₁ (W₁/b₁) + bn₁ (γ₁/β₁)
    (∀ idx : Fin (c1*ic*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₁ x W₁ lr (.operand cotN cotC1)) idx
          = Kernel4.flatten W₁ idx - lr * ∑ jj : Fin (c1*(2*(2*h))*(2*(2*w))),
              pdiv (fun v' : Vec (c1*ic*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₁ x))
                   (Kernel4.flatten W₁) idx jj * cotC1 jj)
  ∧ (∀ o : Fin c1,
        den (SHlo.convBiasSgd bN lrStr W₁ x b₁ lr (.operand cotN cotC1)) o
          = b₁ o - lr * ∑ jj : Fin (c1*(2*(2*h))*(2*(2*w))),
              pdiv (fun b' : Vec c1 => Tensor3.flatten (conv2d W₁ b' x)) b₁ o jj * cotC1 jj)
  ∧ (∀ idx : Fin c1,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε₁ γ₁ cc1 lr (.operand cotN dyBn1)) idx
          = γ₁ idx - lr * ∑ jj : Fin (c1*((2*(2*h))*(2*(2*w)))),
              pdiv (fun γ' : Vec c1 => bnPerChannelFlat c1 ((2*(2*h))*(2*(2*w))) ε₁ γ' β₁ (reassocFwd c1 (2*(2*h)) (2*(2*w)) cc1))
                   γ₁ idx jj * reassocFwd c1 (2*(2*h)) (2*(2*w)) dyBn1 jj)
  ∧ (∀ idx : Fin c1,
        den (SHlo.bnBetaSgd bN lrStr β₁ lr (.operand cotN dyBn1)) idx
          = β₁ idx - lr * ∑ jj : Fin (c1*((2*(2*h))*(2*(2*w)))),
              pdiv (fun β' : Vec c1 => bnPerChannelFlat c1 ((2*(2*h))*(2*(2*w))) ε₁ γ₁ β' (reassocFwd c1 (2*(2*h)) (2*(2*w)) cc1))
                   β₁ idx jj * reassocFwd c1 (2*(2*h)) (2*(2*w)) dyBn1 jj)
  -- conv₂ (W₂/b₂) + bn₂ (γ₂/β₂)
  ∧ (∀ idx : Fin (c1*c1*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₂ r1t W₂ lr (.operand cotN cotC2)) idx
          = Kernel4.flatten W₂ idx - lr * ∑ jj : Fin (c1*(2*(2*h))*(2*(2*w))),
              pdiv (fun v' : Vec (c1*c1*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ r1t))
                   (Kernel4.flatten W₂) idx jj * cotC2 jj)
  ∧ (∀ o : Fin c1,
        den (SHlo.convBiasSgd bN lrStr W₂ r1t b₂ lr (.operand cotN cotC2)) o
          = b₂ o - lr * ∑ jj : Fin (c1*(2*(2*h))*(2*(2*w))),
              pdiv (fun b' : Vec c1 => Tensor3.flatten (conv2d W₂ b' r1t)) b₂ o jj * cotC2 jj)
  ∧ (∀ idx : Fin c1,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε₂ γ₂ cc2 lr (.operand cotN dyBn2)) idx
          = γ₂ idx - lr * ∑ jj : Fin (c1*((2*(2*h))*(2*(2*w)))),
              pdiv (fun γ' : Vec c1 => bnPerChannelFlat c1 ((2*(2*h))*(2*(2*w))) ε₂ γ' β₂ (reassocFwd c1 (2*(2*h)) (2*(2*w)) cc2))
                   γ₂ idx jj * reassocFwd c1 (2*(2*h)) (2*(2*w)) dyBn2 jj)
  ∧ (∀ idx : Fin c1,
        den (SHlo.bnBetaSgd bN lrStr β₂ lr (.operand cotN dyBn2)) idx
          = β₂ idx - lr * ∑ jj : Fin (c1*((2*(2*h))*(2*(2*w)))),
              pdiv (fun β' : Vec c1 => bnPerChannelFlat c1 ((2*(2*h))*(2*(2*w))) ε₂ γ₂ β' (reassocFwd c1 (2*(2*h)) (2*(2*w)) cc2))
                   β₂ idx jj * reassocFwd c1 (2*(2*h)) (2*(2*w)) dyBn2 jj)
  -- conv₃ (W₃/b₃) + bn₃ (γ₃/β₃)
  ∧ (∀ idx : Fin (c2*c1*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₃ zp1t W₃ lr (.operand cotN cotC3)) idx
          = Kernel4.flatten W₃ idx - lr * ∑ jj : Fin (c2*(2*h)*(2*w)),
              pdiv (fun v' : Vec (c2*c1*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₃ zp1t))
                   (Kernel4.flatten W₃) idx jj * cotC3 jj)
  ∧ (∀ o : Fin c2,
        den (SHlo.convBiasSgd bN lrStr W₃ zp1t b₃ lr (.operand cotN cotC3)) o
          = b₃ o - lr * ∑ jj : Fin (c2*(2*h)*(2*w)),
              pdiv (fun b' : Vec c2 => Tensor3.flatten (conv2d W₃ b' zp1t)) b₃ o jj * cotC3 jj)
  ∧ (∀ idx : Fin c2,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε₃ γ₃ cc3 lr (.operand cotN dyBn3)) idx
          = γ₃ idx - lr * ∑ jj : Fin (c2*((2*h)*(2*w))),
              pdiv (fun γ' : Vec c2 => bnPerChannelFlat c2 ((2*h)*(2*w)) ε₃ γ' β₃ (reassocFwd c2 (2*h) (2*w) cc3))
                   γ₃ idx jj * reassocFwd c2 (2*h) (2*w) dyBn3 jj)
  ∧ (∀ idx : Fin c2,
        den (SHlo.bnBetaSgd bN lrStr β₃ lr (.operand cotN dyBn3)) idx
          = β₃ idx - lr * ∑ jj : Fin (c2*((2*h)*(2*w))),
              pdiv (fun β' : Vec c2 => bnPerChannelFlat c2 ((2*h)*(2*w)) ε₃ γ₃ β' (reassocFwd c2 (2*h) (2*w) cc3))
                   β₃ idx jj * reassocFwd c2 (2*h) (2*w) dyBn3 jj)
  -- conv₄ (W₄/b₄) + bn₄ (γ₄/β₄)
  ∧ (∀ idx : Fin (c2*c2*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₄ r3t W₄ lr (.operand cotN cotC4)) idx
          = Kernel4.flatten W₄ idx - lr * ∑ jj : Fin (c2*(2*h)*(2*w)),
              pdiv (fun v' : Vec (c2*c2*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₄ r3t))
                   (Kernel4.flatten W₄) idx jj * cotC4 jj)
  ∧ (∀ o : Fin c2,
        den (SHlo.convBiasSgd bN lrStr W₄ r3t b₄ lr (.operand cotN cotC4)) o
          = b₄ o - lr * ∑ jj : Fin (c2*(2*h)*(2*w)),
              pdiv (fun b' : Vec c2 => Tensor3.flatten (conv2d W₄ b' r3t)) b₄ o jj * cotC4 jj)
  ∧ (∀ idx : Fin c2,
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε₄ γ₄ cc4 lr (.operand cotN dyBn4)) idx
          = γ₄ idx - lr * ∑ jj : Fin (c2*((2*h)*(2*w))),
              pdiv (fun γ' : Vec c2 => bnPerChannelFlat c2 ((2*h)*(2*w)) ε₄ γ' β₄ (reassocFwd c2 (2*h) (2*w) cc4))
                   γ₄ idx jj * reassocFwd c2 (2*h) (2*w) dyBn4 jj)
  ∧ (∀ idx : Fin c2,
        den (SHlo.bnBetaSgd bN lrStr β₄ lr (.operand cotN dyBn4)) idx
          = β₄ idx - lr * ∑ jj : Fin (c2*((2*h)*(2*w))),
              pdiv (fun β' : Vec c2 => bnPerChannelFlat c2 ((2*h)*(2*w)) ε₄ γ₄ β' (reassocFwd c2 (2*h) (2*w) cc4))
                   β₄ idx jj * reassocFwd c2 (2*h) (2*w) dyBn4 jj) := by
  intro xv cc1 bn1o r1 r1t cc2 bn2o r2 r2t zp1 zp1t cc3 bn3o r3 r3t cc4 bn4o r4 r4t zp2 h5 h6 g cpool2
        dyBn4 cotC4 dyBn3 cotC3 dyBn2 cotC2 dyBn1 cotC1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₁ x W₁ cotC1 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₁ x b₁ cotC1 lr o
  · intro idx; exact bnGamma_den gN vN epsStr lrStr cotN ε₁ γ₁ β₁ cc1 dyBn1 lr idx
  · intro idx; exact bnBeta_den bN lrStr cotN ε₁ γ₁ β₁ cc1 dyBn1 lr idx
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₂ r1t W₂ cotC2 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₂ r1t b₂ cotC2 lr o
  · intro idx; exact bnGamma_den gN vN epsStr lrStr cotN ε₂ γ₂ β₂ cc2 dyBn2 lr idx
  · intro idx; exact bnBeta_den bN lrStr cotN ε₂ γ₂ β₂ cc2 dyBn2 lr idx
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₃ zp1t W₃ cotC3 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₃ zp1t b₃ cotC3 lr o
  · intro idx; exact bnGamma_den gN vN epsStr lrStr cotN ε₃ γ₃ β₃ cc3 dyBn3 lr idx
  · intro idx; exact bnBeta_den bN lrStr cotN ε₃ γ₃ β₃ cc3 dyBn3 lr idx
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₄ r3t W₄ cotC4 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₄ r3t b₄ cotC4 lr o
  · intro idx; exact bnGamma_den gN vN epsStr lrStr cotN ε₄ γ₄ β₄ cc4 dyBn4 lr idx
  · intro idx; exact bnBeta_den bN lrStr cotN ε₄ γ₄ β₄ cc4 dyBn4 lr idx

end Proofs.CifarBnPoC
