import LeanMlir.Proofs.Nets.ResNet.ResNet34StepTieB
import LeanMlir.Proofs.Foundation.ParamGrad

/-! # ResNet-34 — a parameter gradient node IS the loss's derivative in that parameter

`r34_net_tiedB` says every parameter gradient node denotes its layer's parameter Jacobian
contracted with the cotangent the emitted backward chain threads to it; `r34IdCotIn_eq_vjp` /
`r34DownCotIn_eq_vjp` say those cotangents are certified VJP backwards. This file closes the loop
for one node: block `e1`'s conv₂ weight gradient — the node the emitted backward reaches first
below the head — equals `∂L/∂W₂` of the whole net, `L` any loss whose gradient at the logits is the
cotangent the chain starts from (`hg`).

**How.** With `W₂` varied and everything else fixed, the net is `L ∘ T ∘ layer`
(`r34E1W2_forward_factor`): `layer θ` is conv₂ at parameter `θ` on its saved input, and `T` is the
rest of the block — `bn₂`, the identity skip (a constant here), the outer relu — then the head.
`T`'s certified VJP (`r34E1C2SuffixHasVJPAt`) is built from the stage library and
`addConstHasVJPAt`, its backward of `g` is the chain's own `r34IdCotC2` (`r34E1C2Suffix_backward`),
and `pdiv_param_chain_batchMap` does the rest.

**Hypotheses.** `0 < ε₂` and the block's outer relu off its kink — `R34IdSmoothAt.hout` at block
`e1`. The body's mid relu sits before `W₂` and needs nothing.
-/

open Proofs Proofs.StableHLO

namespace Proofs.ResNet34TieB

open Proofs.BackLinks (bnInB bnInB_eq_bnBackB reluMaskB)
open scoped BigOperators

/-- The weights with block `e1`'s conv₂ kernel set to the flat parameter `θ`. -/
noncomputable def r34WithE1W2 {nCls : Nat} (w : R34BWeights nCls) (θ : Vec (512 * 512 * 3 * 3)) :
    R34BWeights nCls :=
  { w with e1 := { w.e1 with W₂ := Kernel4.unflatten θ } }

/-- Everything after block `e1`'s conv₂, with the block input `v` (the skip) held fixed: `bn₂`, the
    identity add, the outer relu, the head. -/
noncomputable def r34E1C2Suffix (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (v : Vec (N * (512 * 7 * 7))) : Vec (N * (512 * 7 * 7)) → Vec (N * nCls) :=
  r34HeadB N 7 7 w.Wd w.bd ∘ (relu (N * (512 * 7 * 7)) ∘
    fun u i => bnBatchLA N 512 7 7 w.e1.ε₂ w.e1.γ₂ w.e1.β₂ u i + v i)

/-- **The net with `W₂` varied is the suffix after conv₂ at parameter `θ`.** -/
theorem r34E1W2_forward_factor (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (θ : Vec (512 * 512 * 3 * 3)) :
    resnet34ForwardBFull N (r34WithE1W2 w θ) x
      = r34E1C2Suffix N w (r34Pre15 N w x)
          (batchMap N (flatConv (Kernel4.unflatten θ) w.e1.b₂)
            (cbReluB N (h := 7) (w := 7) w.e1.W₁ w.e1.b₁ w.e1.ε₁ w.e1.γ₁ w.e1.β₁
              (r34Pre15 N w x))) := by
  rw [resnet34ForwardBFull_eq_chain, Function.comp_apply, r34Pre16_apply]
  rfl

/-- The suffix is differentiable at `z` when the outer relu is off its kink there. -/
theorem r34E1C2Suffix_differentiableAt (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hε : 0 < w.e1.ε₂) (v z : Vec (N * (512 * 7 * 7)))
    (hs : ∀ k, bnBatchLA N 512 7 7 w.e1.ε₂ w.e1.γ₂ w.e1.β₂ z k + v k ≠ 0) :
    DifferentiableAt ℝ (r34E1C2Suffix N w v) z :=
  have hA : DifferentiableAt ℝ
      (fun u i => bnBatchLA N 512 7 7 w.e1.ε₂ w.e1.γ₂ w.e1.β₂ u i + v i) z :=
    ((bnBatchLA_differentiable N 512 7 7 w.e1.ε₂ hε w.e1.γ₂ w.e1.β₂) z).add
      (differentiableAt_const v)
  (((batchMap_differentiable _ (dense_differentiable w.Wd w.bd)).comp
      (batchMap_differentiable _ (globalAvgPoolFlat_differentiable 512 7 7))) _).comp z
    ((relu_differentiableAt_of_smooth (N * (512 * 7 * 7)) _ hs).comp z hA)

/-- The suffix's certified VJP at `z`, when the outer relu is off its kink there. -/
noncomputable def r34E1C2SuffixHasVJPAt (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hε : 0 < w.e1.ε₂) (v z : Vec (N * (512 * 7 * 7)))
    (hs : ∀ k, bnBatchLA N 512 7 7 w.e1.ε₂ w.e1.γ₂ w.e1.β₂ z k + v k ≠ 0) :
    HasVJPAt (r34E1C2Suffix N w v) z :=
  have hbn := bnBatchLA_differentiable N 512 7 7 w.e1.ε₂ hε w.e1.γ₂ w.e1.β₂
  have hA : DifferentiableAt ℝ
      (fun u i => bnBatchLA N 512 7 7 w.e1.ε₂ w.e1.γ₂ w.e1.β₂ u i + v i) z :=
    (hbn z).add (differentiableAt_const v)
  have hR := relu_differentiableAt_of_smooth (N * (512 * 7 * 7)) _ hs
  vjpCompAt _ (r34HeadB N 7 7 w.Wd w.bd) z (hR.comp z hA)
    (((batchMap_differentiable _ (dense_differentiable w.Wd w.bd)).comp
      (batchMap_differentiable _ (globalAvgPoolFlat_differentiable 512 7 7))) _)
    (vjpCompAt _ (relu (N * (512 * 7 * 7))) z hA hR
      (addConstHasVJPAt _ v z (hbn z)
        ((bnBatchLAHasVJP N 512 7 7 w.e1.ε₂ hε w.e1.γ₂ w.e1.β₂).toHasVJPAt z))
      (reluHasVJPAt _ _ hs))
    ((r34HeadBHasVJP N 7 7 w.Wd w.bd).toHasVJPAt _)

/-- The suffix's backward, in the emitted chain's vocabulary: the head backward, the outer relu's
    mask, `bn₂`'s backward. -/
theorem r34E1C2Suffix_backward (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hε : 0 < w.e1.ε₂) (v z : Vec (N * (512 * 7 * 7)))
    (hs : ∀ k, bnBatchLA N 512 7 7 w.e1.ε₂ w.e1.γ₂ w.e1.β₂ z k + v k ≠ 0) (g : Vec (N * nCls)) :
    (r34E1C2SuffixHasVJPAt N w hε v z hs).backward g
      = bnInB N 512 7 7 w.e1.ε₂ w.e1.γ₂ z
          (reluMaskB (N * (512 * 7 * 7))
            (fun i => bnBatchLA N 512 7 7 w.e1.ε₂ w.e1.γ₂ w.e1.β₂ z i + v i)
            ((r34HeadBHasVJP N 7 7 w.Wd w.bd).backward
              (relu (N * (512 * 7 * 7))
                (fun i => bnBatchLA N 512 7 7 w.e1.ε₂ w.e1.γ₂ w.e1.β₂ z i + v i)) g)) := by
  rw [bnInB_eq_bnBackB N 512 7 7 w.e1.ε₂ hε w.e1.γ₂ w.e1.β₂]
  rfl

/-- **Block `e1`'s conv₂ weight gradient node is the loss's derivative in `W₂`.** The node, at the
    cotangent the emitted chain threads from `g` (the head's certified backward, then the block's
    outer-relu mask and `bn₂` backward), equals `∂L/∂W₂` of the whole ResNet-34 forward — for any
    loss `L` differentiable at the logits with gradient `g` there. Holds at `0 < ε₂` with the block's
    outer relu off its kink (`R34IdSmoothAt.hout`). -/
theorem r34E1ConvW2Grad_eq_pdiv_loss (N : Nat) {nCls : Nat} (xN cotN : String)
    (w : R34BWeights nCls) (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (L : Vec (N * nCls) → Vec 1) (g : Vec (N * nCls)) (hε : 0 < w.e1.ε₂)
    (hout : ∀ k, residual (projB N (h := 7) (w := 7) w.e1.W₂ w.e1.b₂ w.e1.ε₂ w.e1.γ₂ w.e1.β₂ ∘
      cbReluB N (h := 7) (w := 7) w.e1.W₁ w.e1.b₁ w.e1.ε₁ w.e1.γ₁ w.e1.β₁) (r34Pre15 N w x) k ≠ 0)
    (hL : DifferentiableAt ℝ L (resnet34ForwardBFull N w x))
    (hg : ∀ j, pdiv L (resnet34ForwardBFull N w x) j 0 = g j)
    (idx : Fin (512 * 512 * 3 * 3)) :
    den (SHlo.convWeightGradB xN w.e1.b₂
          (cbReluB N (h := 7) (w := 7) w.e1.W₁ w.e1.b₁ w.e1.ε₁ w.e1.γ₁ w.e1.β₁ (r34Pre15 N w x))
          w.e1.W₂
          (.operand cotN (r34IdCotC2 N 7 7 w.e1 (r34Pre15 N w x)
            (r34HeadCotBlk N 7 7 w.Wd w.bd (r34Pre16 N w x) g)))) idx
      = pdiv (fun θ => L (resnet34ForwardBFull N (r34WithE1W2 w θ) x))
          (Kernel4.flatten w.e1.W₂) idx 0 := by
  set v := r34Pre15 N w x
  set r1 := cbReluB N (h := 7) (w := 7) w.e1.W₁ w.e1.b₁ w.e1.ε₁ w.e1.γ₁ w.e1.β₁ v
  set per : Vec (512 * 512 * 3 * 3) → Vec (512 * 7 * 7) → Vec (512 * 7 * 7) :=
    fun θ y => flatConv (Kernel4.unflatten θ) w.e1.b₂ y
  set θ₀ := Kernel4.flatten w.e1.W₂
  have hz : batchMap N (per θ₀) r1 = batchMap N (flatConv w.e1.W₂ w.e1.b₂) r1 := by
    simp only [per, θ₀, Kernel4.unflatten_flatten]
  have hs : ∀ k, bnBatchLA N 512 7 7 w.e1.ε₂ w.e1.γ₂ w.e1.β₂ (batchMap N (per θ₀) r1) k + v k
      ≠ 0 := by
    rw [hz]; exact hout
  have hT : r34E1C2Suffix N w v (batchMap N (per θ₀) r1) = resnet34ForwardBFull N w x := by
    rw [hz, resnet34ForwardBFull_eq_chain, Function.comp_apply, r34Pre16_apply]; rfl
  have hper : ∀ y, DifferentiableAt ℝ (fun θ' => per θ' y) θ₀ := fun y =>
    (conv2d_weight_differentiable w.e1.b₂ (Tensor3.unflatten y)) θ₀
  have hfun : (fun θ => L (resnet34ForwardBFull N (r34WithE1W2 w θ) x))
      = fun θ => L (r34E1C2Suffix N w v (batchMap N (per θ) r1)) := by
    funext θ; rw [r34E1W2_forward_factor]
  rw [hfun, pdiv_param_chain_batchMap per r1 (r34E1C2Suffix N w v) L θ₀ g hper
    (r34E1C2Suffix_differentiableAt N w hε v _ hs) (by rw [hT]; exact hL)
    (r34E1C2SuffixHasVJPAt N w hε v _ hs) (by rw [hT]; exact hg) idx,
    r34E1C2Suffix_backward, GradNodeB.convWGradB_den]
  simp only [hz]
  rfl

end Proofs.ResNet34TieB
