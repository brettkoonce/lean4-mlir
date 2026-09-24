import LeanMlir.Proofs.Codegen.EfficientNetRenderPC

/-! # EfficientNet Item D — the batched backward (cotangent) math, step by step

The batched analogue of the per-example Item D closes. The forward graph (Item A,
`EfficientNetRenderPC.lean`) lives at the batched index `N·(c·h·w)`; here we sort the **backward**
math at that same index — proving the per-block gradient (`HasVJP`) by composing the proven per-op
VJPs, lifted to the batch.

The lemma everything rests on is `batchMap_has_vjp` (now in `BatchMapVJPAt`, with `bnBatchLA_has_vjp`; the
per-stage VJPs are in `BatchedStages`): a batch-separable op `batchMap N f` (every spatial op
in the forward graph) has a **block-diagonal** VJP — `f`'s VJP applied per example. This is what lets
`seBlockFull_has_vjp`, the conv/depthwise/dense VJPs, etc. lift from one example to the whole batch.
Mechanically it reuses the existing row-wise machinery: `batchMap N f` IS `Mat.flatten ∘ (apply f to
each row) ∘ Mat.unflatten`, so `rowwise_has_vjp_mat` + `hasVJPMat_to_hasVJP` (Tensor.lean) close it.

(The one batch-coupled op, true batch-norm, is handled separately by the proven `bnBatchTensor4_has_vjp`
— it is NOT a `batchMap`. swish/sigmoid are pointwise, so `swish_has_vjp`/`sigmoid_has_vjp` apply
directly at the batched index. Step-by-step, per the plan: `batchMap` VJP first, then `bnBatchLA`, then
the per-block chains.)
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-block VJPs — `vjp_comp` over the batched stages (residual via `residual_has_vjp`)
-- ════════════════════════════════════════════════════════════════

/-- **MBConv1 (no expand) gradient.** `dw-bn-swish → SE → project-bn`. -/
theorem mbNoExpFwdB_differentiable (N : Nat) {ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    Differentiable ℝ (mbNoExpFwdB N (h := h) (w := w) Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbNoExpFwdB dwbsB swish; fun_prop (disch := assumption)
noncomputable def mbNoExpFwdB_has_vjp (N : Nat) {ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    HasVJP (mbNoExpFwdB N (h := h) (w := w) Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbNoExpFwdB
  have dDw := dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  have dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  exact vjp_comp _ _ (dSe.comp dDw) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjp_comp _ _ dDw dSe (dwbsB_has_vjp N (h := h) (w := w) Wd bd εd hεd γd βd)
      (seB_has_vjp N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projB_has_vjp N (h := h) (w := w) Wp bp εp hεp γp βp)

/-- **MBConv6 strided gradient.** `expand-bn-swish → strided dw-bn-swish → SE → project-bn`. -/
theorem mbStridedFwdB_differentiable (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    Differentiable ℝ (mbStridedFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbStridedFwdB cbsB dwbsSB depthwiseStride2Flat swish; fun_prop (disch := assumption)
noncomputable def mbStridedFwdB_has_vjp (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    HasVJP (mbStridedFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) :=
  let dE := cbsB_differentiable N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe
  let dDw := dwbsSB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  let dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  vjp_comp _ _ (dSe.comp (dDw.comp dE)) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjp_comp _ _ (dDw.comp dE) dSe
      (vjp_comp _ _ dE dDw (cbsB_has_vjp N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe)
        (dwbsSB_has_vjp N (h := h) (w := w) Wd bd εd hεd γd βd))
      (seB_has_vjp N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projB_has_vjp N (h := h) (w := w) Wp bp εp hεp γp βp)

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
noncomputable def mbExpFwdB_has_vjp (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    HasVJP (mbExpFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) :=
  let dE := cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe
  let dDw := dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  let dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  vjp_comp _ _ (dSe.comp (dDw.comp dE)) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjp_comp _ _ (dDw.comp dE) dSe
      (vjp_comp _ _ dE dDw (cbsB_has_vjp N (h := h) (w := w) We be εe hεe γe βe)
        (dwbsB_has_vjp N (h := h) (w := w) Wd bd εd hεd γd βd))
      (seB_has_vjp N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projB_has_vjp N (h := h) (w := w) Wp bp εp hεp γp βp)

/-- **MBConv6 residual gradient.** `x + (project-bn ∘ SE ∘ dw-bn-swish ∘ expand-bn-swish)(x)`. -/
theorem mbResidFwdB_differentiable (N : Nat) {c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c) :
    Differentiable ℝ (mbResidFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbResidFwdB residual biPath cbsB dwbsB swish; fun_prop (disch := assumption)
noncomputable def mbResidFwdB_has_vjp (N : Nat) {c mid h w kHd kWd r : Nat}
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
  exact residual_has_vjp _ dBody (mbExpFwdB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp)

/-- **Head gradient.** `1×1 conv-bn-swish → GAP → dense`. -/
theorem headFwdB_differentiable (N : Nat) {c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) :
    Differentiable ℝ (headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc) := by
  unfold headFwdB cbsB swish; fun_prop (disch := assumption)
noncomputable def headFwdB_has_vjp (N : Nat) {c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) :
    HasVJP (headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc) := by
  unfold headFwdB
  have dCbs := cbsB_differentiable N (h := h) (w := w) Wh bh εh hεh γh βh
  have dGap := batchMap_differentiable (N := N) (globalAvgPoolFlat oc h w)
    (globalAvgPoolFlat_differentiable oc h w)
  have vGap := batchMap_has_vjp (N := N) (globalAvgPoolFlat oc h w) (globalAvgPoolFlat_has_vjp oc h w)
    (globalAvgPoolFlat_differentiable oc h w)
  exact vjp_comp _ _ (dGap.comp dCbs)
    (batchMap_differentiable (N := N) (dense Wfc bfc) (dense_differentiable Wfc bfc))
    (vjp_comp _ _ dCbs dGap (cbsB_has_vjp N (h := h) (w := w) Wh bh εh hεh γh βh) vGap)
    (batchMap_has_vjp (N := N) (dense Wfc bfc) (dense_has_vjp Wfc bfc) (dense_differentiable Wfc bfc))

end Proofs
