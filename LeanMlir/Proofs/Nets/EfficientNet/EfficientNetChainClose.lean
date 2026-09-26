import LeanMlir.Proofs.Codegen.EfficientNetRender.PC

/-! # EfficientNet — the batched backward (cotangent) math, step by step

The batched analogue of the per-example cotangent chains. The forward graph
(`EfficientNetRender.PC`) lives at the batched index `N·(c·h·w)`; here we sort the **backward**
math at that same index — proving the per-block gradient (`HasVJP`) by composing the proven per-op
VJPs, lifted to the batch.

The lemma everything rests on is `batchMapHasVJP` (now in `BatchMapVJPAt`, with `bnBatchLAHasVJP`; the
per-stage VJPs are in `Batched.Stages`): a batch-separable op `batchMap N f` (every spatial op
in the forward graph) has a **block-diagonal** VJP — `f`'s VJP applied per example. This is what lets
`seBlockFullHasVJP`, the conv/depthwise/dense VJPs, etc. lift from one example to the whole batch.
Mechanically it reuses the existing row-wise machinery: `batchMap N f` IS `Mat.flatten ∘ (apply f to
each row) ∘ Mat.unflatten`, so `rowwiseHasVJPMat` + `HasVJPMat.toHasVJP` (Tensor.lean) close it.

(The one batch-coupled op, true batch-norm, is handled separately by the proven `bnBatchTensor4HasVJP`
— it is NOT a `batchMap`. swish/sigmoid are pointwise, so `swishHasVJP`/`sigmoidHasVJP` apply
directly at the batched index. The file proves the `batchMap` VJP first, then `bnBatchLA`, then the
per-block chains.)
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-block VJPs — `vjpComp` over the batched stages (residual via `residualHasVJP`)
-- ════════════════════════════════════════════════════════════════

/-- **MBConv1 (no expand) gradient.** `dw-bn-swish → SE → project-bn`. -/
theorem mbNoExpFwdB_differentiable (N : Nat) {ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    Differentiable ℝ (mbNoExpFwdB N (h := h) (w := w) Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbNoExpFwdB dwbsB swish; fun_prop (disch := assumption)
noncomputable def mbNoExpFwdBHasVJP (N : Nat) {ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    HasVJP (mbNoExpFwdB N (h := h) (w := w) Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbNoExpFwdB
  have dDw := dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  have dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  exact vjpComp _ _ (dSe.comp dDw) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjpComp _ _ dDw dSe (dwbsBHasVJP N (h := h) (w := w) Wd bd εd hεd γd βd)
      (seBHasVJP N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projBHasVJP N (h := h) (w := w) Wp bp εp hεp γp βp)

/-- **MBConv6 strided gradient.** `expand-bn-swish → strided dw-bn-swish → SE → project-bn`. -/
theorem mbStridedFwdB_differentiable (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    Differentiable ℝ (mbStridedFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbStridedFwdB cbsB dwbsSB depthwiseStride2Flat swish; fun_prop (disch := assumption)
noncomputable def mbStridedFwdBHasVJP (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    HasVJP (mbStridedFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) :=
  let dE := cbsB_differentiable N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe
  let dDw := dwbsSB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  let dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  vjpComp _ _ (dSe.comp (dDw.comp dE)) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjpComp _ _ (dDw.comp dE) dSe
      (vjpComp _ _ dE dDw (cbsBHasVJP N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe)
        (dwbsSBHasVJP N (h := h) (w := w) Wd bd εd hεd γd βd))
      (seBHasVJP N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projBHasVJP N (h := h) (w := w) Wp bp εp hεp γp βp)

/-- **MBConv6 expand, stride-1, NO residual** (stages 5/7's first block, `ic ≠ oc`): `project-bn ∘ SE ∘ dw-bn-swish ∘ expand-bn-swish` (the
    `mbResidFwdB` body without the identity skip). -/
noncomputable def mbExpFwdB (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  projB N (h := h) (w := w) Wp bp εp γp βp ∘
    seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘
    dwbsB N (h := h) (w := w) Wd bd εd γd βd ∘
    cbsB N (h := h) (w := w) We be εe γe βe

theorem mbExpFwdB_differentiable (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    Differentiable ℝ (mbExpFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbExpFwdB cbsB dwbsB swish; fun_prop (disch := assumption)
noncomputable def mbExpFwdBHasVJP (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    HasVJP (mbExpFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) :=
  let dE := cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe
  let dDw := dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  let dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  vjpComp _ _ (dSe.comp (dDw.comp dE)) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjpComp _ _ (dDw.comp dE) dSe
      (vjpComp _ _ dE dDw (cbsBHasVJP N (h := h) (w := w) We be εe hεe γe βe)
        (dwbsBHasVJP N (h := h) (w := w) Wd bd εd hεd γd βd))
      (seBHasVJP N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projBHasVJP N (h := h) (w := w) Wp bp εp hεp γp βp)

/-- **MBConv6 residual gradient.** `x + (project-bn ∘ SE ∘ dw-bn-swish ∘ expand-bn-swish)(x)`. -/
theorem mbResidFwdB_differentiable (N : Nat) {c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c) :
    Differentiable ℝ (mbResidFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbResidFwdB residual biPath cbsB dwbsB swish; fun_prop (disch := assumption)
noncomputable def mbResidFwdBHasVJP (N : Nat) {c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c) :
    HasVJP (mbResidFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbResidFwdB
  have dE := cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe
  have dDw := dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  have dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  have dBody := (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp).comp (dSe.comp (dDw.comp dE))
  exact residualHasVJP _ dBody (mbExpFwdBHasVJP N We be εe hεe γe βe Wd bd εd hεd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp)

/-- **Head gradient.** `1×1 conv-bn-swish → GAP → dense`. -/
theorem headFwdB_differentiable (N : Nat) {c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) :
    Differentiable ℝ (headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc) := by
  unfold headFwdB cbsB swish; fun_prop (disch := assumption)
noncomputable def headFwdBHasVJP (N : Nat) {c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) :
    HasVJP (headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc) := by
  unfold headFwdB
  have dCbs := cbsB_differentiable N (h := h) (w := w) Wh bh εh hεh γh βh
  have dGap := batchMap_differentiable (N := N) (globalAvgPoolFlat oc h w)
    (globalAvgPoolFlat_differentiable oc h w)
  have vGap := batchMapHasVJP (N := N) (globalAvgPoolFlat oc h w) (globalAvgPoolFlatHasVJP oc h w)
    (globalAvgPoolFlat_differentiable oc h w)
  exact vjpComp _ _ (dGap.comp dCbs)
    (batchMap_differentiable (N := N) (dense Wfc bfc) (dense_differentiable Wfc bfc))
    (vjpComp _ _ dCbs dGap (cbsBHasVJP N (h := h) (w := w) Wh bh εh hεh γh βh) vGap)
    (batchMapHasVJP (N := N) (dense Wfc bfc) (denseHasVJP Wfc bfc) (dense_differentiable Wfc bfc))

end Proofs
