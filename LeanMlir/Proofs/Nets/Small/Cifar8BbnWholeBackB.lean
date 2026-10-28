module

public import LeanMlir.Proofs.Foundation.HeadLayers
public import LeanMlir.Proofs.Nets.Small.ChapterGraphTies
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # The batch-statistics cifar8-BN backward chain IS the certified whole-net VJP

`Cifar8BbnStepTieB.cifar8Bbn_train_step_tiedB` ties the thirty-eight gradient NODES of the
`cifar8wb_bbn_*` train steps: each denotes the layer's Jacobian contracted with the cotangent the
rendered backward chain hands it. That theorem takes the cotangents as the chain delivers them.
This file is the other half — that the chain delivers the right ones: the backward chain those
artifacts render, read in the `BackLinks` vocabulary the node tie uses (`reluMaskB`, `bnBackB`,
`cInB`, the per-example pool scatter, `rowDenseBackFlat`), IS the backward of the certified VJP of
the whole batched forward, `relu ∘ dense ∘ relu ∘ dense ∘ dense ∘ pool ∘ (conv → BN → relu)² ∘ …`,
at every batch size `N`. ResNet-50's `r50InputGradB_eq_r34B_full_vjp` is the statement this one
mirrors; what ResNet-50 does with sixteen opaque bottleneck witnesses, this net does with
seventeen concrete `CertLayer`s, because every stage of it is a layer the suite already certifies.

## How it is built

* `maxPool2Layer` — the batched 2×2/s2 max pool as a `CertLayer`: `batchMapHasVJPAt` over the
  per-example `maxPoolFlatHasVJPAt'`, certified where every example's windows attain their max
  once (`MaxPool2Smooth`), its graph the render's `.maxPoolBackB`, faithful by
  `batchMapAux_eq_batchMapHasVJPAt` with each row's tie `rfl`. The 2×2 peer of `HeadLayers`'
  3×3/s2 stem pool.
* `cifar8BbnLayer` — the net: eight `cbReluLayer`s (conv → true BN → relu) and four pools in four
  stages, then `denseLayer`, `reluOut`, `denseLayer`, `reluOut`, `denseLayer`. `CertLayer.comp`
  does the composition proof; this file only names the layers.
* `Cifar8BbnSmoothAt` — the smooth-point hypotheses, one per kink: eight pre-relu activations off
  zero, four pools with unique window maxima, two head pre-activations off zero, each stated at a
  NAMED running activation (`c8A1 … c8RA`) so the `.ok` discharge stays one `comp_ok_of` per layer.
* `cifar8BbnInputGradB` — the chain, written out; `cifar8BbnBHasVJPAt_backward` says the layer's
  VJP backward is that chain, `cifar8BbnInputGradB_correct` reads it as `∑ pdiv … * dy`, and
  `cifar8BbnLayer_fwd` says the layer's forward is the committed composite `cifar8BbnForwardB`.

## Scope

A smooth-point statement at one precision: the conv nodes are the f32 kinds, as in ResNet-50's
whole-net tie (the bf16 artifacts' gradient nodes are `Cifar8BbnStepTieB`'s, on the renderers'
`bf16` switch at the identity rounding). One device: `cifar8wb_bbn` has no data-parallel render.
It is about the INPUT cotangent chain; the thirty-eight parameter gradients are the node tie's,
and that each of them is the loss's derivative in its parameter is
[`Cifar8BbnParamGrad.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Nets/Small/Cifar8BbnParamGrad.lean)
(`cifar8Bbn_net_lossGrad`). -/

open Proofs Proofs.StableHLO Proofs.BackLinks

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The batched 2×2/s2 max pool as a `CertLayer`
-- ════════════════════════════════════════════════════════════════

/-- Per-example smoothness of a batched pool input: every example's 2×2 windows attain their
    maximum at exactly one cell (`MaxPool2Smooth`, read on that example's `[C, 2h, 2w]` slice). -/
def MaxPool2SmoothB (N c h w : Nat) (x : Vec (N * (c * (2 * h) * (2 * w)))) : Prop :=
  ∀ r : Fin N, MaxPool2Smooth (Tensor3.unflatten (Mat.unflatten x r) : Tensor3 c (2 * h) (2 * w))

theorem maxPoolFlat_differentiableAt_row (N c h w : Nat) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (hx : MaxPool2SmoothB N c h w x) (r : Fin N) :
    DifferentiableAt ℝ (maxPoolFlat c h w) (Mat.unflatten x r) :=
  maxPoolFlat_differentiableAt_flat _ (hx r)

/-- **The batched 2×2 pool's VJP at a smooth point** — `batchMapHasVJPAt` over each example's own
    `maxPoolFlatHasVJPAt'`. Built field by field (no transport), so its `.backward` reduces. -/
noncomputable def maxPool2BHasVJPAt (N c h w : Nat) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (hx : MaxPool2SmoothB N c h w x) :
    HasVJPAt (batchMap N (maxPoolFlat c h w)) x :=
  batchMapHasVJPAt (maxPoolFlat c h w) x
    (fun r => maxPoolFlatHasVJPAt' (Mat.unflatten x r) (hx r))
    (fun r => maxPoolFlat_differentiableAt_row N c h w x hx r)

/-- Its backward is the render's scatter: `maxPoolBackFlat` per example on that example's own
    saved activation, which is `den (.maxPoolBackB …)`. -/
theorem maxPool2BHasVJPAt_backward (N c h w : Nat) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (hx : MaxPool2SmoothB N c h w x) :
    (maxPool2BHasVJPAt N c h w x hx).backward = batchMapAux N (maxPoolBackFlat c h w) x :=
  (batchMapAux_eq_batchMapHasVJPAt (maxPoolFlat c h w) (maxPoolBackFlat c h w) x _ _
    (fun _ => rfl)).symm

/-- Batched **2×2/s2 max pool** as a `CertLayer`, certified where every example's windows are
    smooth; its backward graph is the render's `.maxPoolBackB` at the saved pool input. -/
noncomputable def maxPool2Layer (N c h w : Nat) :
    CertLayer (N * (c * (2 * h) * (2 * w))) (N * (c * h * w)) where
  fwd := batchMap N (maxPoolFlat c h w)
  ok := MaxPool2SmoothB N c h w
  diff := fun x hx => batchMap_differentiableAt (maxPoolFlat c h w) x
    (fun r => maxPoolFlat_differentiableAt_row N c h w x hx r)
  vjp := fun x hx => maxPool2BHasVJPAt N c h w x hx
  graph := fun x e => .maxPoolBackB (N := N) (c := c) (h := h) (w := w) "%pool" x e
  faithful := fun x hx e => by
    rw [den_maxPoolBackB, ← maxPool2BHasVJPAt_backward N c h w x hx]

theorem maxPool2Layer_fwd_apply (N c h w : Nat) (v : Vec (N * (c * (2 * h) * (2 * w)))) :
    (maxPool2Layer N c h w).fwd v = batchMap N (maxPoolFlat c h w) v := rfl

/-- `comp`'s VJP backward, peeled: `L₂`'s then `L₁`'s. Definitional (`vjpCompAt_backward`); stated
    so a whole-net chain unwinds by `rw` rather than by `simp` on the structure. -/
theorem CertLayer.comp_vjp_backward {m n p : Nat} (L₁ : CertLayer m n) (L₂ : CertLayer n p)
    (x : Vec m) (hx : (L₁.comp L₂).ok x) (dy : Vec p) :
    ((L₁.comp L₂).vjp x hx).backward dy
      = (L₁.vjp x hx.1).backward ((L₂.vjp (L₁.fwd x) hx.2).backward dy) := rfl

/-- The conv → BN → relu stage's VJP backward, in the `BackLinks` vocabulary: the relu mask at the
    pre-activation, the true-BN backward at the conv output, the conv input-VJP. The relu peer of
    `cbsB_back_eq`. -/
theorem cbReluB_back_eq (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * (ic * h * w)))
    (hs : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0)
    (dy : Vec (N * (oc * h * w))) :
    (cbReluBHasVJPAt N W b ε hε γ β x hs).backward dy
      = cInB N (h := h) (w := w) W b
          (bnBackB N oc h w ε hε γ β (batchMap N (flatConv W b) x)
            (reluMaskB (N * (oc * h * w)) (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x))
              dy)) := by
  have hg := cbReluBackBatchedGraph_faithful W b ε hε γ β x (.operand "" dy) hs
  rw [den_operand] at hg
  rw [← hg, cbReluBackBatchedGraph, den_convBackBatched_eq_cInB,
    bnBatchLABack_faithful (β := β) (hε := hε), selectPos_faithful _ _ hs, den_operand]
  rfl

/-- `cbReluLayer`'s VJP backward, in the `BackLinks` vocabulary. -/
theorem cbReluLayer_vjp_backward (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * (ic * h * w)))
    (hx : (cbReluLayer N (h := h) (w := w) W b ε hε γ β).ok x) (dy : Vec (N * (oc * h * w))) :
    ((cbReluLayer N (h := h) (w := w) W b ε hε γ β).vjp x hx).backward dy
      = cInB N (h := h) (w := w) W b
          (bnBackB N oc h w ε hε γ β (batchMap N (flatConv W b) x)
            (reluMaskB (N * (oc * h * w)) (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x))
              dy)) :=
  cbReluB_back_eq N W b ε hε γ β x hx dy

/-- `maxPool2Layer`'s VJP backward is the per-example scatter. -/
theorem maxPool2Layer_vjp_backward (N c h w : Nat) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (hx : (maxPool2Layer N c h w).ok x) (dy : Vec (N * (c * h * w))) :
    ((maxPool2Layer N c h w).vjp x hx).backward dy = batchMapAux N (maxPoolBackFlat c h w) x dy :=
  congrFun (maxPool2BHasVJPAt_backward N c h w x hx) dy

/-- `denseLayer`'s VJP backward is `rowDenseBackFlat` (`den` of `.denseRowBack`). -/
theorem denseLayer_vjp_backward (N : Nat) {a nC : Nat} (W : Mat a nC) (b : Vec nC) (x : Vec (N * a))
    (hx : (denseLayer N W b).ok x) (dy : Vec (N * nC)) :
    ((denseLayer N W b).vjp x hx).backward dy = rowDenseBackFlat N a nC W dy := rfl

/-- `reluOut`'s VJP backward is the relu mask. -/
theorem reluOut_vjp_backward (n : Nat) (x : Vec n) (hx : (CertLayer.reluOut n).ok x) (dy : Vec n) :
    ((CertLayer.reluOut n).vjp x hx).backward dy = reluMaskB n x dy := rfl

end Proofs.StableHLO

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The net's weights, named running activations, and the layer
-- ════════════════════════════════════════════════════════════════

/-- The thirty-eight tensors of the eight-conv CIFAR net with BatchNorm, as one record: eight
    (kernel, bias, γ, β) and the three dense layers. One `ε` is shared by every BatchNorm, as the
    render bakes it. -/
structure C8BbnW (ic c1 c2 c3 c4 h w d1 nC kH kW : Nat) where
  W₁ : Kernel4 c1 ic kH kW
  b₁ : Vec c1
  γ₁ : Vec c1
  β₁ : Vec c1
  W₂ : Kernel4 c1 c1 kH kW
  b₂ : Vec c1
  γ₂ : Vec c1
  β₂ : Vec c1
  W₃ : Kernel4 c2 c1 kH kW
  b₃ : Vec c2
  γ₃ : Vec c2
  β₃ : Vec c2
  W₄ : Kernel4 c2 c2 kH kW
  b₄ : Vec c2
  γ₄ : Vec c2
  β₄ : Vec c2
  W₅ : Kernel4 c3 c2 kH kW
  b₅ : Vec c3
  γ₅ : Vec c3
  β₅ : Vec c3
  W₆ : Kernel4 c3 c3 kH kW
  b₆ : Vec c3
  γ₆ : Vec c3
  β₆ : Vec c3
  W₇ : Kernel4 c4 c3 kH kW
  b₇ : Vec c4
  γ₇ : Vec c4
  β₇ : Vec c4
  W₈ : Kernel4 c4 c4 kH kW
  b₈ : Vec c4
  γ₈ : Vec c4
  β₈ : Vec c4
  W₉ : Mat (c4 * h * w) d1
  b₉ : Vec d1
  Wa : Mat d1 d1
  ba : Vec d1
  Wb : Mat d1 nC
  bb : Vec nC

variable {ic c1 c2 c3 c4 h w d1 nC kH kW : Nat}

/-! The running activations, one `def` each, in forward order. Stage `k` runs at spatial
`2^(5-k)·h`, written as the explicit `2 * (…)` nest the render's types use. -/

/-- conv₁ → BN → relu, at 32². -/
noncomputable def c8A1 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :=
  cbReluB N (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) p.W₁ p.b₁ ε p.γ₁ p.β₁ x

/-- conv₂ → BN → relu, at 32². -/
noncomputable def c8A2 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :=
  cbReluB N (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) p.W₂ p.b₂ ε p.γ₂ p.β₂
    (c8A1 N p ε x)

/-- pool₁: 32² → 16². -/
noncomputable def c8P1 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c1 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))) :=
  batchMap N (maxPoolFlat c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w)))) (c8A2 N p ε x)

/-- conv₃ → BN → relu, at 16². -/
noncomputable def c8A3 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))) :=
  cbReluB N (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) p.W₃ p.b₃ ε p.γ₃ p.β₃ (c8P1 N p ε x)

/-- conv₄ → BN → relu, at 16². -/
noncomputable def c8A4 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))) :=
  cbReluB N (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) p.W₄ p.b₄ ε p.γ₄ p.β₄ (c8A3 N p ε x)

/-- pool₂: 16² → 8². -/
noncomputable def c8P2 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c2 * (2 * (2 * h)) * (2 * (2 * w)))) :=
  batchMap N (maxPoolFlat c2 (2 * (2 * h)) (2 * (2 * w))) (c8A4 N p ε x)

/-- conv₅ → BN → relu, at 8². -/
noncomputable def c8A5 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w)))) :=
  cbReluB N (h := 2 * (2 * h)) (w := 2 * (2 * w)) p.W₅ p.b₅ ε p.γ₅ p.β₅ (c8P2 N p ε x)

/-- conv₆ → BN → relu, at 8². -/
noncomputable def c8A6 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c3 * (2 * (2 * h)) * (2 * (2 * w)))) :=
  cbReluB N (h := 2 * (2 * h)) (w := 2 * (2 * w)) p.W₆ p.b₆ ε p.γ₆ p.β₆ (c8A5 N p ε x)

/-- pool₃: 8² → 4². -/
noncomputable def c8P3 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c3 * (2 * h) * (2 * w))) :=
  batchMap N (maxPoolFlat c3 (2 * h) (2 * w)) (c8A6 N p ε x)

/-- conv₇ → BN → relu, at 4². -/
noncomputable def c8A7 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c4 * (2 * h) * (2 * w))) :=
  cbReluB N (h := 2 * h) (w := 2 * w) p.W₇ p.b₇ ε p.γ₇ p.β₇ (c8P3 N p ε x)

/-- conv₈ → BN → relu, at 4². -/
noncomputable def c8A8 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c4 * (2 * h) * (2 * w))) :=
  cbReluB N (h := 2 * h) (w := 2 * w) p.W₈ p.b₈ ε p.γ₈ p.β₈ (c8A7 N p ε x)

/-- pool₄: 4² → 2², the flatten the head reads (`c4·h·w` wide). -/
noncomputable def c8P4 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * (c4 * h * w)) :=
  batchMap N (maxPoolFlat c4 h w) (c8A8 N p ε x)

/-- dense₉ pre-activation. -/
noncomputable def c8H9 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * d1) :=
  batchMap N (dense p.W₉ p.b₉) (c8P4 N p ε x)

/-- relu₉. -/
noncomputable def c8R9 (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * d1) :=
  relu (N * d1) (c8H9 N p ε x)

/-- denseₐ pre-activation. -/
noncomputable def c8HA (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * d1) :=
  batchMap N (dense p.Wa p.ba) (c8R9 N p ε x)

/-- reluₐ. -/
noncomputable def c8RA (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * d1) :=
  relu (N * d1) (c8HA N p ε x)

/-- **The batched eight-conv CIFAR forward with BatchNorm at batch statistics**, to the logits:
    the composite `dense_b ∘ relu ∘ dense_a ∘ relu ∘ dense₉ ∘ pool ∘ (conv → BN → relu)² ∘ … `
    at batch `N`, through the named running activations above. -/
noncomputable def cifar8BbnForwardB (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    Vec (N * nC) :=
  batchMap N (dense p.Wb p.bb) (c8RA N p ε x)

/-- **The net as one `CertLayer`**: seventeen certified layers composed. -/
noncomputable def cifar8BbnLayer (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW)
    (ε : ℝ) (hε : 0 < ε) :
    CertLayer (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) (N * nC) :=
  (cbReluLayer N (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) p.W₁ p.b₁ ε hε p.γ₁ p.β₁).comp <|
  (cbReluLayer N (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) p.W₂ p.b₂ ε hε p.γ₂ p.β₂).comp <|
  (maxPool2Layer N c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w)))).comp <|
  (cbReluLayer N (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) p.W₃ p.b₃ ε hε p.γ₃ p.β₃).comp <|
  (cbReluLayer N (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) p.W₄ p.b₄ ε hε p.γ₄ p.β₄).comp <|
  (maxPool2Layer N c2 (2 * (2 * h)) (2 * (2 * w))).comp <|
  (cbReluLayer N (h := 2 * (2 * h)) (w := 2 * (2 * w)) p.W₅ p.b₅ ε hε p.γ₅ p.β₅).comp <|
  (cbReluLayer N (h := 2 * (2 * h)) (w := 2 * (2 * w)) p.W₆ p.b₆ ε hε p.γ₆ p.β₆).comp <|
  (maxPool2Layer N c3 (2 * h) (2 * w)).comp <|
  (cbReluLayer N (h := 2 * h) (w := 2 * w) p.W₇ p.b₇ ε hε p.γ₇ p.β₇).comp <|
  (cbReluLayer N (h := 2 * h) (w := 2 * w) p.W₈ p.b₈ ε hε p.γ₈ p.β₈).comp <|
  (maxPool2Layer N c4 h w).comp <|
  (denseLayer N p.W₉ p.b₉).comp <|
  (CertLayer.reluOut (N * d1)).comp <|
  (denseLayer N p.Wa p.ba).comp <|
  (CertLayer.reluOut (N * d1)).comp <|
  denseLayer N p.Wb p.bb

/-- **The shape check**: the composed layer's forward IS `cifar8BbnForwardB`. -/
theorem cifar8BbnLayer_fwd (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW)
    (ε : ℝ) (hε : 0 < ε) :
    (cifar8BbnLayer N p ε hε).fwd = cifar8BbnForwardB N p ε := rfl

-- ════════════════════════════════════════════════════════════════
-- § The smooth-point hypotheses, and the layer's `.ok`
-- ════════════════════════════════════════════════════════════════

/-- **Where the net is certified**: each of the eight pre-relu activations off its kink, each of
    the four pool inputs with unique window maxima per example, the two head pre-activations off
    zero — at the running activations. Everything else in the net (conv, BN at `0 < ε`, dense) is
    smooth everywhere. -/
structure Cifar8BbnSmoothAt (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW) (ε : ℝ)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) : Prop where
  s1 : ∀ k, bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₁ p.β₁
    (batchMap N (flatConv p.W₁ p.b₁) x) k ≠ 0
  s2 : ∀ k, bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂
    (batchMap N (flatConv p.W₂ p.b₂) (c8A1 N p ε x)) k ≠ 0
  q1 : MaxPool2SmoothB N c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) (c8A2 N p ε x)
  s3 : ∀ k, bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃
    (batchMap N (flatConv p.W₃ p.b₃) (c8P1 N p ε x)) k ≠ 0
  s4 : ∀ k, bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄
    (batchMap N (flatConv p.W₄ p.b₄) (c8A3 N p ε x)) k ≠ 0
  q2 : MaxPool2SmoothB N c2 (2 * (2 * h)) (2 * (2 * w)) (c8A4 N p ε x)
  s5 : ∀ k, bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅
    (batchMap N (flatConv p.W₅ p.b₅) (c8P2 N p ε x)) k ≠ 0
  s6 : ∀ k, bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆
    (batchMap N (flatConv p.W₆ p.b₆) (c8A5 N p ε x)) k ≠ 0
  q3 : MaxPool2SmoothB N c3 (2 * h) (2 * w) (c8A6 N p ε x)
  s7 : ∀ k, bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇
    (batchMap N (flatConv p.W₇ p.b₇) (c8P3 N p ε x)) k ≠ 0
  s8 : ∀ k, bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈
    (batchMap N (flatConv p.W₈ p.b₈) (c8A7 N p ε x)) k ≠ 0
  q4 : MaxPool2SmoothB N c4 h w (c8A8 N p ε x)
  r9 : ∀ k, c8H9 N p ε x k ≠ 0
  ra : ∀ k, c8HA N p ε x k ≠ 0

/-- `Cifar8BbnSmoothAt` is the layer's `.ok`: one `comp_ok_of` per layer, each naming its input. -/
theorem cifar8BbnSmoothAt_ok (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW)
    (ε : ℝ) (hε : 0 < ε)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (hs : Cifar8BbnSmoothAt N p ε x) : (cifar8BbnLayer N p ε hε).ok x := by
  refine CertLayer.comp_ok_of hs.s1 (c8A1 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.s2 (c8A2 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.q1 (c8P1 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.s3 (c8A3 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.s4 (c8A4 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.q2 (c8P2 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.s5 (c8A5 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.s6 (c8A6 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.q3 (c8P3 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.s7 (c8A7 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.s8 (c8A8 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.q4 (c8P4 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of trivial (c8H9 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.r9 (c8R9 N p ε x) rfl ?_
  refine CertLayer.comp_ok_of trivial (c8HA N p ε x) rfl ?_
  refine CertLayer.comp_ok_of hs.ra (c8RA N p ε x) rfl ?_
  exact trivial

/-- **The certified whole-net VJP** of `cifar8BbnForwardB` at a smooth point. -/
noncomputable def cifar8BbnBHasVJPAt (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW)
    (ε : ℝ) (hε : 0 < ε)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (hs : Cifar8BbnSmoothAt N p ε x) : HasVJPAt (cifar8BbnForwardB N p ε) x :=
  (cifar8BbnLayer N p ε hε).vjp x (cifar8BbnSmoothAt_ok N p ε hε x hs)

-- ════════════════════════════════════════════════════════════════
-- § The chain, written out, and the tie
-- ════════════════════════════════════════════════════════════════

/-- **The rendered backward chain, read in the `BackLinks` vocabulary**, from the logits cotangent
    `dy` to the input cotangent: three `rowDenseBackFlat`s with the two head relu masks between
    them, then per stage the pool scatter, and twice `reluMaskB → bnBackB → cInB`. Every step is
    the `den` of the node the artifact emits (`den_convBackBatched_eq_cInB`, `den_maxPoolBackB`,
    `den_selectPosB`, `den` of `.denseRowBack`), the BatchNorm link through `bnInB_eq_bnBackB`. -/
noncomputable def cifar8BbnInputGradB (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW)
    (ε : ℝ) (hε : 0 < ε)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (dy : Vec (N * nC)) :
    Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :=
  -- head
  let dRA := rowDenseBackFlat N d1 nC p.Wb dy
  let dHA := reluMaskB (N * d1) (c8HA N p ε x) dRA
  let dR9 := rowDenseBackFlat N d1 d1 p.Wa dHA
  let dH9 := reluMaskB (N * d1) (c8H9 N p ε x) dR9
  let dP4 := rowDenseBackFlat N (c4 * h * w) d1 p.W₉ dH9
  -- stage 4
  let dA8 := batchMapAux N (maxPoolBackFlat c4 h w) (c8A8 N p ε x) dP4
  let dA7 := cInB N (h := 2 * h) (w := 2 * w) p.W₈ p.b₈
    (bnBackB N c4 (2 * h) (2 * w) ε hε p.γ₈ p.β₈ (batchMap N (flatConv p.W₈ p.b₈) (c8A7 N p ε x))
      (reluMaskB _ (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₈ p.β₈
        (batchMap N (flatConv p.W₈ p.b₈) (c8A7 N p ε x))) dA8))
  let dP3 := cInB N (h := 2 * h) (w := 2 * w) p.W₇ p.b₇
    (bnBackB N c4 (2 * h) (2 * w) ε hε p.γ₇ p.β₇ (batchMap N (flatConv p.W₇ p.b₇) (c8P3 N p ε x))
      (reluMaskB _ (bnBatchLA N c4 (2 * h) (2 * w) ε p.γ₇ p.β₇
        (batchMap N (flatConv p.W₇ p.b₇) (c8P3 N p ε x))) dA7))
  -- stage 3
  let dA6 := batchMapAux N (maxPoolBackFlat c3 (2 * h) (2 * w)) (c8A6 N p ε x) dP3
  let dA5 := cInB N (h := 2 * (2 * h)) (w := 2 * (2 * w)) p.W₆ p.b₆
    (bnBackB N c3 (2 * (2 * h)) (2 * (2 * w)) ε hε p.γ₆ p.β₆
      (batchMap N (flatConv p.W₆ p.b₆) (c8A5 N p ε x))
      (reluMaskB _ (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₆ p.β₆
        (batchMap N (flatConv p.W₆ p.b₆) (c8A5 N p ε x))) dA6))
  let dP2 := cInB N (h := 2 * (2 * h)) (w := 2 * (2 * w)) p.W₅ p.b₅
    (bnBackB N c3 (2 * (2 * h)) (2 * (2 * w)) ε hε p.γ₅ p.β₅
      (batchMap N (flatConv p.W₅ p.b₅) (c8P2 N p ε x))
      (reluMaskB _ (bnBatchLA N c3 (2 * (2 * h)) (2 * (2 * w)) ε p.γ₅ p.β₅
        (batchMap N (flatConv p.W₅ p.b₅) (c8P2 N p ε x))) dA5))
  -- stage 2
  let dA4 := batchMapAux N (maxPoolBackFlat c2 (2 * (2 * h)) (2 * (2 * w))) (c8A4 N p ε x) dP2
  let dA3 := cInB N (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) p.W₄ p.b₄
    (bnBackB N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε hε p.γ₄ p.β₄
      (batchMap N (flatConv p.W₄ p.b₄) (c8A3 N p ε x))
      (reluMaskB _ (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₄ p.β₄
        (batchMap N (flatConv p.W₄ p.b₄) (c8A3 N p ε x))) dA4))
  let dP1 := cInB N (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) p.W₃ p.b₃
    (bnBackB N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε hε p.γ₃ p.β₃
      (batchMap N (flatConv p.W₃ p.b₃) (c8P1 N p ε x))
      (reluMaskB _ (bnBatchLA N c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) ε p.γ₃ p.β₃
        (batchMap N (flatConv p.W₃ p.b₃) (c8P1 N p ε x))) dA3))
  -- stage 1
  let dA2 := batchMapAux N (maxPoolBackFlat c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))))
    (c8A2 N p ε x) dP1
  let dA1 := cInB N (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) p.W₂ p.b₂
    (bnBackB N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε hε p.γ₂ p.β₂
      (batchMap N (flatConv p.W₂ p.b₂) (c8A1 N p ε x))
      (reluMaskB _ (bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₂ p.β₂
        (batchMap N (flatConv p.W₂ p.b₂) (c8A1 N p ε x))) dA2))
  cInB N (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) p.W₁ p.b₁
    (bnBackB N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε hε p.γ₁ p.β₁
      (batchMap N (flatConv p.W₁ p.b₁) x)
      (reluMaskB _ (bnBatchLA N c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))) ε p.γ₁ p.β₁
        (batchMap N (flatConv p.W₁ p.b₁) x)) dA1))

/-- **The chain IS the certified whole-net backward.** The layer's VJP backward, peeled one
    `comp` at a time, is each stage's backward in turn, and each of those is its `BackLinks`
    form: `cbReluB_back_eq` at the eight conv stages, `maxPool2BHasVJPAt_backward` at the four
    pools, and `rfl` at the dense and relu layers. -/
theorem cifar8BbnLayer_vjp_backward (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW)
    (ε : ℝ) (hε : 0 < ε)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (hok : (cifar8BbnLayer N p ε hε).ok x) (dy : Vec (N * nC)) :
    ((cifar8BbnLayer N p ε hε).vjp x hok).backward dy = cifar8BbnInputGradB N p ε hε x dy := by
  unfold cifar8BbnLayer at hok ⊢
  repeat rw [CertLayer.comp_vjp_backward]
  simp only [cbReluLayer_vjp_backward, maxPool2Layer_vjp_backward, denseLayer_vjp_backward,
    reluOut_vjp_backward]
  rfl

/-- …and so is the certified whole-net VJP's. -/
theorem cifar8BbnBHasVJPAt_backward (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW)
    (ε : ℝ) (hε : 0 < ε)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (hs : Cifar8BbnSmoothAt N p ε x) (dy : Vec (N * nC)) :
    (cifar8BbnBHasVJPAt N p ε hε x hs).backward dy = cifar8BbnInputGradB N p ε hε x dy :=
  cifar8BbnLayer_vjp_backward N p ε hε x (cifar8BbnSmoothAt_ok N p ε hε x hs) dy

/-- **The chain IS the `pdiv`-contracted Jacobian of the whole net** — at every batch size, every
    logits cotangent and every input coordinate, wherever the eight relus and the two head relus
    are off their kinks and the four pools have unique window maxima (`Cifar8BbnSmoothAt`), with
    `0 < ε`. -/
theorem cifar8BbnInputGradB_correct (N : Nat) (p : C8BbnW ic c1 c2 c3 c4 h w d1 nC kH kW)
    (ε : ℝ) (hε : 0 < ε)
    (x : Vec (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))))
    (hs : Cifar8BbnSmoothAt N p ε x) (dy : Vec (N * nC))
    (i : Fin (N * (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))) :
    cifar8BbnInputGradB N p ε hε x dy i
      = ∑ j : Fin (N * nC), pdiv (cifar8BbnForwardB N p ε) x i j * dy j := by
  rw [← cifar8BbnBHasVJPAt_backward N p ε hε x hs dy]
  exact (cifar8BbnBHasVJPAt N p ε hε x hs).correct dy i

end Proofs
