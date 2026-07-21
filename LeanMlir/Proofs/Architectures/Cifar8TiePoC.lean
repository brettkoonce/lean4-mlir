import LeanMlir.Proofs.Architectures.Cifar8FaithfulPoC

/-! # PoC: the cifar8 (Chapter 4 deeper, 8-conv no-BN) §1a TIE — tied through the real forward

The 4-stage peer of `CifarFaithfulPoC`'s §1a tie (`cifar_conv_tied_certified`). cifar8 is cifar (ch5)
with **four** conv→conv→pool stages instead of two, so its conv backward chain is the cifar chain
repeated: within each stage the second conv is the maxpool-back layer (`cnnChainCotW2` for the very
last, then `cifarChainCotW2`'s cross-pool move) and the first conv is the conv-back layer
(`cnnChainCotW1`). **Every chain cotangent reuses an existing constructor** (`cnnChainCotW2` /
`cnnChainCotW1` / `cifarChainCotW2`) at the 4-stage dims — no new constructor, no new ops, no new
bridges. The conv ties are `CifarPoC.convW_den`/`convB_den` (generic in the cotangent); the dense head
+ loss-cot mirror cifar.

Spatial bookkeeping (the 2-stage `(h,w)` convention nested two levels deeper): final pooled `(h,w)`;
stage 4 (conv₇/conv₈) at `(2h,2w)`; stage 3 (conv₅/conv₆) at `(2(2h),2(2w))`; stage 2 (conv₃/conv₄) at
`(2(2(2h)),…)`; stage 1 (conv₁/conv₂) at `(2(2(2(2h))),…)`.

## Honest residual (same as cifar)
* Conv backward rendered hand-written (cotangent SSA ↔ chain-cot per-op trust); per-op `pretty`
  lexing; ℝ → Float32.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.Cifar8PoC

/-- **The emitted loss-cotangent graph denotes the softmax-CE gradient of the cifar8 forward.** -/
theorem cifar8LossCot_den {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (nlogN ohN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) (label : Fin nClasses) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN
            (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x))))
          (.operand ohN (oneHot nClasses label)))
      = fun j => softmax nClasses (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
                    W₉ b₉ Wa ba Wb bb x) j - oneHot nClasses label j := by
  funext j; simp only [den, softmax]

/-- **Dense output weight `Wb`, tied to the WHOLE softmax-CE loss through the cifar8 forward.** The
    dense head is the standard 3-layer MLP; given the forward logits = `mnistLinear Wb bb a_head`
    (true by `Function.comp_apply`, supplied as `hlog`), `Wb` folds to `∂CE/∂Wb`. -/
theorem cifar8_Wb_tied_totalloss {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (aN lrStr dyN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w)))))) (a_head : Vec d1) (label : Fin nClasses)
    (lr : ℝ) (i : Fin d1) (j : Fin nClasses)
    (hlog : cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x
              = mnistLinear Wb bb a_head) :
    den (SHlo.weightSgd aN "%Wb" lrStr a_head Wb lr
          (.operand dyN (fun k => softmax nClasses
              (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x) k
                - oneHot nClasses label k)))
        (finProdFinEquiv (i, j))
      = Wb i j - lr * pdiv (fun v : Vec (d1 * nClasses) => fun _ : Fin 1 =>
            crossEntropy nClasses (dense (Mat.unflatten v) bb a_head) label)
          (Mat.flatten Wb) (finProdFinEquiv (i, j)) 0 := by
  rw [denseW_den aN "%Wb" lrStr dyN a_head Wb bb
        (fun k => softmax nClasses (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
            W₉ b₉ Wa ba Wb bb x) k - oneHot nClasses label k) lr i j,
      mlp_output_total_loss_grad Wb bb a_head label i j, hlog]

set_option maxRecDepth 16000 in
/-- **Whole cifar8 conv tail, tied.** All 16 conv params (8 conv `W`+`b`), at the real cifar8 forward
    and the composed softmax-CE cotangent, denote the certified loss-descent step. Each conv op is fed
    the cotangent the 4-stage backward chain delivers: `cnnChainCotW2` (conv₈, the last before pool₄),
    `cnnChainCotW1` (conv₇/₅/₃/₁, the within-stage conv-back), `cifarChainCotW2` (conv₆/₄/₂, the
    cross-pool move). Together with the dense head (`cifar8_Wb_tied_totalloss` + the generic
    `denseW_den`/`denseB_den` at `g`) the WHOLE cifar8 train step is den-composed forward→loss→backward. -/
theorem cifar8_convs_tied_certified {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}
    (xN wN bN lrStr cotN : String)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4*h*w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Tensor3 ic (2*(2*(2*(2*h)))) (2*(2*(2*(2*w))))) (label : Fin nClasses) (lr : ℝ) :
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
    let cotC6 : Vec (c3*(2*(2*h))*(2*(2*w))) := CifarPoC.cifarChainCotW2 W₇ r6t cc6 cotC7
    let cotC5 : Vec (c3*(2*(2*h))*(2*(2*w))) := cnnChainCotW1 W₆ cc5 cotC6
    let cotC4 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := CifarPoC.cifarChainCotW2 W₅ r4t cc4 cotC5
    let cotC3 : Vec (c2*(2*(2*(2*h)))*(2*(2*(2*w)))) := cnnChainCotW1 W₄ cc3 cotC4
    let cotC2 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := CifarPoC.cifarChainCotW2 W₃ r2t cc2 cotC3
    let cotC1 : Vec (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))) := cnnChainCotW1 W₂ cc1 cotC2
    -- conv₁
    (∀ idx : Fin (c1*ic*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₁ x W₁ lr (.operand cotN cotC1)) idx
          = Kernel4.flatten W₁ idx - lr * ∑ jj : Fin (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))),
              pdiv (fun v' : Vec (c1*ic*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₁ x))
                   (Kernel4.flatten W₁) idx jj * cotC1 jj)
  ∧ (∀ o : Fin c1,
        den (SHlo.convBiasSgd bN lrStr W₁ x b₁ lr (.operand cotN cotC1)) o
          = b₁ o - lr * ∑ jj : Fin (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))),
              pdiv (fun b' : Vec c1 => Tensor3.flatten (conv2d W₁ b' x)) b₁ o jj * cotC1 jj)
    -- conv₂
  ∧ (∀ idx : Fin (c1*c1*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₂ r1t W₂ lr (.operand cotN cotC2)) idx
          = Kernel4.flatten W₂ idx - lr * ∑ jj : Fin (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))),
              pdiv (fun v' : Vec (c1*c1*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ r1t))
                   (Kernel4.flatten W₂) idx jj * cotC2 jj)
  ∧ (∀ o : Fin c1,
        den (SHlo.convBiasSgd bN lrStr W₂ r1t b₂ lr (.operand cotN cotC2)) o
          = b₂ o - lr * ∑ jj : Fin (c1*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))),
              pdiv (fun b' : Vec c1 => Tensor3.flatten (conv2d W₂ b' r1t)) b₂ o jj * cotC2 jj)
    -- conv₃
  ∧ (∀ idx : Fin (c2*c1*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₃ zp1t W₃ lr (.operand cotN cotC3)) idx
          = Kernel4.flatten W₃ idx - lr * ∑ jj : Fin (c2*(2*(2*(2*h)))*(2*(2*(2*w)))),
              pdiv (fun v' : Vec (c2*c1*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₃ zp1t))
                   (Kernel4.flatten W₃) idx jj * cotC3 jj)
  ∧ (∀ o : Fin c2,
        den (SHlo.convBiasSgd bN lrStr W₃ zp1t b₃ lr (.operand cotN cotC3)) o
          = b₃ o - lr * ∑ jj : Fin (c2*(2*(2*(2*h)))*(2*(2*(2*w)))),
              pdiv (fun b' : Vec c2 => Tensor3.flatten (conv2d W₃ b' zp1t)) b₃ o jj * cotC3 jj)
    -- conv₄
  ∧ (∀ idx : Fin (c2*c2*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₄ r3t W₄ lr (.operand cotN cotC4)) idx
          = Kernel4.flatten W₄ idx - lr * ∑ jj : Fin (c2*(2*(2*(2*h)))*(2*(2*(2*w)))),
              pdiv (fun v' : Vec (c2*c2*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₄ r3t))
                   (Kernel4.flatten W₄) idx jj * cotC4 jj)
  ∧ (∀ o : Fin c2,
        den (SHlo.convBiasSgd bN lrStr W₄ r3t b₄ lr (.operand cotN cotC4)) o
          = b₄ o - lr * ∑ jj : Fin (c2*(2*(2*(2*h)))*(2*(2*(2*w)))),
              pdiv (fun b' : Vec c2 => Tensor3.flatten (conv2d W₄ b' r3t)) b₄ o jj * cotC4 jj)
    -- conv₅
  ∧ (∀ idx : Fin (c3*c2*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₅ zp2t W₅ lr (.operand cotN cotC5)) idx
          = Kernel4.flatten W₅ idx - lr * ∑ jj : Fin (c3*(2*(2*h))*(2*(2*w))),
              pdiv (fun v' : Vec (c3*c2*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₅ zp2t))
                   (Kernel4.flatten W₅) idx jj * cotC5 jj)
  ∧ (∀ o : Fin c3,
        den (SHlo.convBiasSgd bN lrStr W₅ zp2t b₅ lr (.operand cotN cotC5)) o
          = b₅ o - lr * ∑ jj : Fin (c3*(2*(2*h))*(2*(2*w))),
              pdiv (fun b' : Vec c3 => Tensor3.flatten (conv2d W₅ b' zp2t)) b₅ o jj * cotC5 jj)
    -- conv₆
  ∧ (∀ idx : Fin (c3*c3*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₆ r5t W₆ lr (.operand cotN cotC6)) idx
          = Kernel4.flatten W₆ idx - lr * ∑ jj : Fin (c3*(2*(2*h))*(2*(2*w))),
              pdiv (fun v' : Vec (c3*c3*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₆ r5t))
                   (Kernel4.flatten W₆) idx jj * cotC6 jj)
  ∧ (∀ o : Fin c3,
        den (SHlo.convBiasSgd bN lrStr W₆ r5t b₆ lr (.operand cotN cotC6)) o
          = b₆ o - lr * ∑ jj : Fin (c3*(2*(2*h))*(2*(2*w))),
              pdiv (fun b' : Vec c3 => Tensor3.flatten (conv2d W₆ b' r5t)) b₆ o jj * cotC6 jj)
    -- conv₇
  ∧ (∀ idx : Fin (c4*c3*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₇ zp3t W₇ lr (.operand cotN cotC7)) idx
          = Kernel4.flatten W₇ idx - lr * ∑ jj : Fin (c4*(2*h)*(2*w)),
              pdiv (fun v' : Vec (c4*c3*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₇ zp3t))
                   (Kernel4.flatten W₇) idx jj * cotC7 jj)
  ∧ (∀ o : Fin c4,
        den (SHlo.convBiasSgd bN lrStr W₇ zp3t b₇ lr (.operand cotN cotC7)) o
          = b₇ o - lr * ∑ jj : Fin (c4*(2*h)*(2*w)),
              pdiv (fun b' : Vec c4 => Tensor3.flatten (conv2d W₇ b' zp3t)) b₇ o jj * cotC7 jj)
    -- conv₈
  ∧ (∀ idx : Fin (c4*c4*kH*kW),
        den (SHlo.convWeightSgd xN wN lrStr b₈ r7t W₈ lr (.operand cotN cotC8)) idx
          = Kernel4.flatten W₈ idx - lr * ∑ jj : Fin (c4*(2*h)*(2*w)),
              pdiv (fun v' : Vec (c4*c4*kH*kW) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b₈ r7t))
                   (Kernel4.flatten W₈) idx jj * cotC8 jj)
  ∧ (∀ o : Fin c4,
        den (SHlo.convBiasSgd bN lrStr W₈ r7t b₈ lr (.operand cotN cotC8)) o
          = b₈ o - lr * ∑ jj : Fin (c4*(2*h)*(2*w)),
              pdiv (fun b' : Vec c4 => Tensor3.flatten (conv2d W₈ b' r7t)) b₈ o jj * cotC8 jj) := by
  intro xv cc1 r1 r1t cc2 r2 r2t zp1 zp1t cc3 r3 r3t cc4 r4 r4t zp2 zp2t cc5 r5 r5t cc6 r6 r6t zp3 zp3t
        cc7 r7 r7t cc8 r8 r8t zp4 h9 ha g cotC8 cotC7 cotC6 cotC5 cotC4 cotC3 cotC2 cotC1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₁ x W₁ cotC1 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₁ x b₁ cotC1 lr o
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₂ r1t W₂ cotC2 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₂ r1t b₂ cotC2 lr o
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₃ zp1t W₃ cotC3 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₃ zp1t b₃ cotC3 lr o
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₄ r3t W₄ cotC4 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₄ r3t b₄ cotC4 lr o
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₅ zp2t W₅ cotC5 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₅ zp2t b₅ cotC5 lr o
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₆ r5t W₆ cotC6 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₆ r5t b₆ cotC6 lr o
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₇ zp3t W₇ cotC7 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₇ zp3t b₇ cotC7 lr o
  · intro idx; exact CifarPoC.convW_den xN wN lrStr cotN b₈ r7t W₈ cotC8 lr idx
  · intro o;   exact CifarPoC.convB_den bN lrStr cotN W₈ r7t b₈ cotC8 lr o

end Proofs.Cifar8PoC
