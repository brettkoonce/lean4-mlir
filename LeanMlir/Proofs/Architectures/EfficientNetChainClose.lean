import LeanMlir.Proofs.EfficientNetRenderPC

/-! # EfficientNet Item D — the batched backward (cotangent) math, step by step

The batched analogue of `MobileNetV2ChainClose` / `ResNet34ChainClose`. The forward graph (Item A,
`EfficientNetRenderPC.lean`) lives at the batched index `N·(c·h·w)`; here we sort the **backward**
math at that same index — proving the per-block gradient (`HasVJP`) by composing the proven per-op
VJPs, lifted to the batch.

The genuinely-new lemma is `batchMap_has_vjp`: a batch-separable op `batchMap N f` (every spatial op
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
-- § `batchMap` is differentiable, and its VJP is block-diagonal (the per-example VJP, batched)
-- ════════════════════════════════════════════════════════════════

/-- **`batchMap N f` is the flattened row-wise application of `f`.** Reading the output at flat index
    `idx` (decoding to example `m`, coord `c`) gives `f (row m of the input) c` on both sides — the
    `Mat.flatten`/`unflatten` row-major convention is exactly `batchMap`'s `finProdFinEquiv` split. -/
theorem batchMap_eq_rowwiseFlat {N a b : Nat} (f : Vec a → Vec b) :
    StableHLO.batchMap N f
      = fun v : Vec (N * a) => Mat.flatten ((fun A : Mat N a => fun r => f (A r)) (Mat.unflatten v)) := by
  funext v idx
  rfl

/-- **`batchMap N f` is differentiable** when `f` is — it is `f` applied independently per example. -/
theorem batchMap_differentiable {N a b : Nat} (f : Vec a → Vec b) (hf : Differentiable ℝ f) :
    Differentiable ℝ (StableHLO.batchMap N f) := by
  rw [batchMap_eq_rowwiseFlat]
  apply differentiable_pi.mpr
  intro idx
  have hcoord :
      (fun v : Vec (N * a) =>
          Mat.flatten ((fun A : Mat N a => fun r => f (A r)) (Mat.unflatten v)) idx)
        = (fun w : Vec a => f w (finProdFinEquiv.symm idx).2) ∘
            (reindexCLM (fun i : Fin a => finProdFinEquiv ((finProdFinEquiv.symm idx).1, i))) := by
    funext v; rfl
  rw [hcoord]
  exact Differentiable.comp
    (differentiable_pi.mp hf (finProdFinEquiv.symm idx).2)
    (reindexCLM _).differentiable

/-- **`batchMap N f` VJP — block-diagonal (the genuinely-new lemma).** A batch-separable op's VJP
    applies `f`'s proven VJP independently per example. The backward, like the forward, reshapes to
    `[N, ·]` and runs `f.backward` row-wise. Reuses `rowwise_has_vjp_mat` + `hasVJPMat_to_hasVJP`. This
    is `seBlockFull_has_vjp` / the conv-depthwise-dense VJPs "lifted by batchMap" to the whole batch. -/
noncomputable def batchMap_has_vjp {N a b : Nat} (f : Vec a → Vec b)
    (hf : HasVJP f) (hf_diff : Differentiable ℝ f) :
    HasVJP (StableHLO.batchMap N f) :=
  (batchMap_eq_rowwiseFlat f).symm ▸ hasVJPMat_to_hasVJP (rowwise_has_vjp_mat hf hf_diff)

-- ════════════════════════════════════════════════════════════════
-- § True batch-norm `bnBatchLA` VJP — the proven `bnBatchTensor4`, reindex-conjugated
-- ════════════════════════════════════════════════════════════════

/-- **Generic reindex VJP.** `reindexCLM σ` (gather `y ↦ y ∘ σ`) is linear; its backward scatters each
    output cotangent back to the inputs that map to it (the adjoint). Generalizes the manual reindex
    VJPs (`broadcastFlat_has_vjp`, `bnchwFwd/Back_has_vjp`). -/
noncomputable def reindex_has_vjp {a b : Nat} (σ : Fin b → Fin a) :
    HasVJP (reindexCLM σ) where
  backward := fun _v dy => fun i => ∑ k : Fin b, (if i = σ k then dy k else 0)
  correct := by
    intro v dy i
    show (∑ k : Fin b, if i = σ k then dy k else 0)
        = ∑ j : Fin b, pdiv (reindexCLM σ) v i j * dy j
    have hpd : ∀ j : Fin b, pdiv (reindexCLM σ) v i j = if i = σ j then 1 else 0 := by
      intro j; exact pdiv_reindex σ v i j
    simp_rw [hpd]
    apply Finset.sum_congr rfl
    intro j _
    by_cases hij : i = σ j
    · rw [if_pos hij, if_pos hij, one_mul]
    · rw [if_neg hij, if_neg hij, zero_mul]

/-- **`bnBatchLA` is the proven `bnBatchTensor4`, conjugated by the `mul_assoc` reindex.** Both reindex
    maps are `reindexCLM (Fin.cast …)`; the middle is the genuinely batch-coupled true batch-norm. -/
theorem bnBatchLA_eq_comp (N oc h w : Nat) (ε : ℝ) (γ β : Vec oc) :
    StableHLO.bnBatchLA N oc h w ε γ β
      = (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)))) ∘
          bnBatchTensor4 N oc h w ε γ β ∘
          (reindexCLM (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm)) := by
  rfl

theorem bnBatchLA_differentiable (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (StableHLO.bnBatchLA N oc h w ε γ β) := by
  rw [bnBatchLA_eq_comp]
  exact (reindexCLM _).differentiable.comp
    ((bnBatchTensor4_differentiable N oc h w ε hε γ β).comp (reindexCLM _).differentiable)

/-- **True batch-norm VJP at the network's flat index.** `bnBatchLA`'s backward is the proven
    `bnBatchTensor4` VJP (batch-coupled — NOT a `batchMap`), conjugated by the reindex isos. -/
noncomputable def bnBatchLA_has_vjp (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (StableHLO.bnBatchLA N oc h w ε γ β) := by
  rw [bnBatchLA_eq_comp]
  exact vjp_comp _ _
    ((bnBatchTensor4_differentiable N oc h w ε hε γ β).comp (reindexCLM _).differentiable)
    (reindexCLM _).differentiable
    (vjp_comp _ _ (reindexCLM _).differentiable (bnBatchTensor4_differentiable N oc h w ε hε γ β)
      (reindex_has_vjp _) (bnBatchTensor4_has_vjp N oc h w ε hε γ β))
    (reindex_has_vjp _)

-- ════════════════════════════════════════════════════════════════
-- § Per-stage VJPs — the batched stage abbreviations (`EfficientNetRenderPC`) compose the
--   batched per-op VJPs (`batchMap`-lifted) + true-BN (`bnBatchLA`) + pointwise swish.
-- ════════════════════════════════════════════════════════════════

/-- `flatConv W b` VJP — the per-example 1×1/3×3 conv input-VJP (the `HasVJP3`-bridged `conv2d`). -/
noncomputable def flatConv_has_vjp {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    HasVJP (flatConv W b : Vec (ic * h * w) → Vec (oc * h * w)) :=
  hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)

/-- Differentiability of a batched `conv/depthwise → bn → swish` stage. -/
theorem bnSwishStage_differentiable (N : Nat) {a oc h w : Nat} (op : Vec a → Vec (oc * h * w))
    (hop : Differentiable ℝ op) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (swish (N * (oc * h * w)) ∘ StableHLO.bnBatchLA N oc h w ε γ β ∘
      StableHLO.batchMap N op) :=
  (swish_diff _).comp ((bnBatchLA_differentiable N oc h w ε hε γ β).comp
    (batchMap_differentiable op hop))

/-- VJP of a batched `conv/depthwise → bn → swish` stage: lift `op`'s VJP per example, then the proven
    true-BN VJP, then the pointwise swish VJP. -/
noncomputable def bnSwishStage_has_vjp (N : Nat) {a oc h w : Nat} (op : Vec a → Vec (oc * h * w))
    (hop : Differentiable ℝ op) (hopv : HasVJP op) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (swish (N * (oc * h * w)) ∘ StableHLO.bnBatchLA N oc h w ε γ β ∘
      StableHLO.batchMap N op) :=
  vjp_comp _ _
    ((bnBatchLA_differentiable N oc h w ε hε γ β).comp (batchMap_differentiable op hop))
    (swish_diff _)
    (vjp_comp _ _ (batchMap_differentiable op hop) (bnBatchLA_differentiable N oc h w ε hε γ β)
      (batchMap_has_vjp op hopv hop) (bnBatchLA_has_vjp N oc h w ε hε γ β))
    (swish_has_vjp _)

/-- Differentiability of a batched `conv → bn` stage (project bottleneck, no swish). -/
theorem bnStage_differentiable (N : Nat) {a oc h w : Nat} (op : Vec a → Vec (oc * h * w))
    (hop : Differentiable ℝ op) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (StableHLO.bnBatchLA N oc h w ε γ β ∘ StableHLO.batchMap N op) :=
  (bnBatchLA_differentiable N oc h w ε hε γ β).comp (batchMap_differentiable op hop)

/-- VJP of a batched `conv → bn` stage. -/
noncomputable def bnStage_has_vjp (N : Nat) {a oc h w : Nat} (op : Vec a → Vec (oc * h * w))
    (hop : Differentiable ℝ op) (hopv : HasVJP op) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (StableHLO.bnBatchLA N oc h w ε γ β ∘ StableHLO.batchMap N op) :=
  vjp_comp _ _ (batchMap_differentiable op hop) (bnBatchLA_differentiable N oc h w ε hε γ β)
    (batchMap_has_vjp op hopv hop) (bnBatchLA_has_vjp N oc h w ε hε γ β)

-- The six stage abbreviations (reducible to the forms above), each with differentiability + VJP.

theorem cbsB_differentiable (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (cbsB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_differentiable N (flatConv W b) (flatConv_differentiable W b) ε hε γ β
noncomputable def cbsB_has_vjp (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (cbsB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_has_vjp N (flatConv W b) (flatConv_differentiable W b) (flatConv_has_vjp W b) ε hε γ β

theorem stemB_differentiable (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (stemB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_differentiable N (flatConvStride2 W b) (flatConvStride2_differentiable W b) ε hε γ β
noncomputable def stemB_has_vjp (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (stemB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_has_vjp N (flatConvStride2 W b) (flatConvStride2_differentiable W b)
    (flatConvStride2_has_vjp W b) ε hε γ β

theorem dwbsB_differentiable (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    Differentiable ℝ (dwbsB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_differentiable N (depthwiseFlat W b) (depthwiseFlat_differentiable W b) ε hε γ β
noncomputable def dwbsB_has_vjp (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    HasVJP (dwbsB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_has_vjp N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    (depthwiseFlat_has_vjp W b) ε hε γ β

theorem dwbsSB_differentiable (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    Differentiable ℝ (dwbsSB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_differentiable N (depthwiseStride2Flat W b) (depthwiseStride2Flat_differentiable W b)
    ε hε γ β
noncomputable def dwbsSB_has_vjp (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    HasVJP (dwbsSB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_has_vjp N (depthwiseStride2Flat W b) (depthwiseStride2Flat_differentiable W b)
    (depthwiseStride2Flat_has_vjp W b) ε hε γ β

theorem seB_differentiable (N : Nat) {c h w r : Nat} (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c)
    (b₂ : Vec c) :
    Differentiable ℝ (seB N (h := h) (w := w) W₁ b₁ W₂ b₂) :=
  batchMap_differentiable _ (seBlockFull_differentiable W₁ b₁ W₂ b₂)
noncomputable def seB_has_vjp (N : Nat) {c h w r : Nat} (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c)
    (b₂ : Vec c) :
    HasVJP (seB N (h := h) (w := w) W₁ b₁ W₂ b₂) :=
  batchMap_has_vjp _ (seBlockFull_has_vjp W₁ b₁ W₂ b₂) (seBlockFull_differentiable W₁ b₁ W₂ b₂)

theorem projB_differentiable (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (projB N (h := h) (w := w) W b ε γ β) :=
  bnStage_differentiable N (flatConv W b) (flatConv_differentiable W b) ε hε γ β
noncomputable def projB_has_vjp (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (projB N (h := h) (w := w) W b ε γ β) :=
  bnStage_has_vjp N (flatConv W b) (flatConv_differentiable W b) (flatConv_has_vjp W b) ε hε γ β

-- ════════════════════════════════════════════════════════════════
-- § Per-block VJPs — `vjp_comp` over the batched stages (residual via `residual_has_vjp`)
-- ════════════════════════════════════════════════════════════════

/-- **MBConv1 (no expand) gradient.** `dw-bn-swish → SE → project-bn`. -/
theorem mbNoExpFwdB_differentiable (N : Nat) {ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    Differentiable ℝ (mbNoExpFwdB N (h := h) (w := w) Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbNoExpFwdB
  exact (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp).comp
    ((seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂).comp
      (dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd))
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
  unfold mbStridedFwdB
  exact (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp).comp
    ((seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂).comp
      ((dwbsSB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd).comp
        (cbsB_differentiable N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe)))
noncomputable def mbStridedFwdB_has_vjp (N : Nat) {ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    HasVJP (mbStridedFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbStridedFwdB
  have dE := cbsB_differentiable N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe
  have dDw := dwbsSB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd
  have dSe := seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂
  have vEdw : HasVJP _ := vjp_comp _ _ dE dDw (cbsB_has_vjp N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe)
    (dwbsSB_has_vjp N (h := h) (w := w) Wd bd εd hεd γd βd)
  exact vjp_comp _ _ (dSe.comp (dDw.comp dE)) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
    (vjp_comp _ _ (dDw.comp dE) dSe vEdw (seB_has_vjp N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
    (projB_has_vjp N (h := h) (w := w) Wp bp εp hεp γp βp)

/-- **MBConv6 residual gradient.** `x + (project-bn ∘ SE ∘ dw-bn-swish ∘ expand-bn-swish)(x)`. -/
theorem mbResidFwdB_differentiable (N : Nat) {c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c) :
    Differentiable ℝ (mbResidFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  have dBody : Differentiable ℝ (projB N (h := h) (w := w) Wp bp εp γp βp ∘
      seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ ∘ dwbsB N (h := h) (w := w) Wd bd εd γd βd ∘
      cbsB N (h := h) (w := w) We be εe γe βe) :=
    (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp).comp
      ((seB_differentiable N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂).comp
        ((dwbsB_differentiable N (h := h) (w := w) Wd bd εd hεd γd βd).comp
          (cbsB_differentiable N (h := h) (w := w) We be εe hεe γe βe)))
  unfold mbResidFwdB residual biPath
  apply differentiable_pi.mpr; intro i
  exact (differentiable_pi.mp dBody i).add (differentiable_apply i)
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
  have vEdw : HasVJP _ := vjp_comp _ _ dE dDw (cbsB_has_vjp N (h := h) (w := w) We be εe hεe γe βe)
    (dwbsB_has_vjp N (h := h) (w := w) Wd bd εd hεd γd βd)
  have vBody : HasVJP _ :=
    vjp_comp _ _ (dSe.comp (dDw.comp dE)) (projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp)
      (vjp_comp _ _ (dDw.comp dE) dSe vEdw (seB_has_vjp N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂))
      (projB_has_vjp N (h := h) (w := w) Wp bp εp hεp γp βp)
  exact residual_has_vjp _ dBody vBody

/-- **Head gradient.** `1×1 conv-bn-swish → GAP → dense`. -/
theorem headFwdB_differentiable (N : Nat) {c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC) :
    Differentiable ℝ (headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc) := by
  unfold headFwdB
  exact (batchMap_differentiable (dense Wfc bfc) (dense_differentiable Wfc bfc)).comp
    ((batchMap_differentiable (globalAvgPoolFlat oc h w) (globalAvgPoolFlat_differentiable oc h w)).comp
      (cbsB_differentiable N (h := h) (w := w) Wh bh εh hεh γh βh))
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

-- ════════════════════════════════════════════════════════════════
-- § The whole batched subnet VJP — the capstone (batched analogue of `efficientnet_has_vjp`)
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000 in
/-- **The representative batched EfficientNet-B0 has a (correct) VJP.** Chained from the per-block
    gradients — stem → MBConv1 → MBConv6-strided → MBConv6-residual → head — via `vjp_comp`. The
    backward is genuinely composed from the proven per-op VJPs, `batchMap`-lifted to the batch and
    crossing the batch-coupled true batch-norm; `HasVJP.correct` pins it to the true Jacobian-transpose.
    The batched, true-batch-norm + SE analogue of `efficientnet_has_vjp` (EfficientNet.lean).

    Stated on the `∘`-composition of the blocks — which IS `efficientnetForwardB` (its nested-application
    spelling, used by the forward proof, is definitionally this composition); `vjp_comp` builds exactly
    this composition, so the proof closes structurally without re-reducing the whole net. -/
noncomputable def efficientnetForwardB_has_vjp
    (N : Nat)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec 32)
    (Wd1 : DepthwiseKernel 32 3 3) (bd1 : Vec 32) (εd1 : ℝ) (hεd1 : 0 < εd1) (γd1 βd1 : Vec 32)
    (Wz1a : Mat 32 8) (bz1a : Vec 8) (Wz1b : Mat 8 32) (bz1b : Vec 32)
    (Wp1 : Kernel4 16 32 1 1) (bp1 : Vec 16) (εp1 : ℝ) (hεp1 : 0 < εp1) (γp1 βp1 : Vec 16)
    (We2 : Kernel4 96 16 1 1) (be2 : Vec 96) (εe2 : ℝ) (hεe2 : 0 < εe2) (γe2 βe2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 : ℝ) (hεd2 : 0 < εd2) (γd2 βd2 : Vec 96)
    (Wz2a : Mat 96 4) (bz2a : Vec 4) (Wz2b : Mat 4 96) (bz2b : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 : ℝ) (hεp2 : 0 < εp2) (γp2 βp2 : Vec 24)
    (We3 : Kernel4 144 24 1 1) (be3 : Vec 144) (εe3 : ℝ) (hεe3 : 0 < εe3) (γe3 βe3 : Vec 144)
    (Wd3 : DepthwiseKernel 144 5 5) (bd3 : Vec 144) (εd3 : ℝ) (hεd3 : 0 < εd3) (γd3 βd3 : Vec 144)
    (Wz3a : Mat 144 6) (bz3a : Vec 6) (Wz3b : Mat 6 144) (bz3b : Vec 144)
    (Wp3 : Kernel4 24 144 1 1) (bp3 : Vec 24) (εp3 : ℝ) (hεp3 : 0 < εp3) (γp3 βp3 : Vec 24)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10) :
    HasVJP (headFwdB N (h := 56) (w := 56) Wh bh εh γh βh Wfc bfc ∘
      mbResidFwdB N (h := 56) (w := 56) We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3
        Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3 γp3 βp3 ∘
      mbStridedFwdB N (h := 56) (w := 56) We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2
        Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 γp2 βp2 ∘
      mbNoExpFwdB N (h := 112) (w := 112) Wd1 bd1 εd1 γd1 βd1 Wz1a bz1a Wz1b bz1b
        Wp1 bp1 εp1 γp1 βp1 ∘
      stemB N (h := 112) (w := 112) Ws bs εs γs βs) := by
  have dStem := stemB_differentiable N (h := 112) (w := 112) Ws bs εs hεs γs βs
  have vStem := stemB_has_vjp N (h := 112) (w := 112) Ws bs εs hεs γs βs
  have dB1 := mbNoExpFwdB_differentiable N (h := 112) (w := 112) Wd1 bd1 εd1 hεd1 γd1 βd1
    Wz1a bz1a Wz1b bz1b Wp1 bp1 εp1 hεp1 γp1 βp1
  have vB1 := mbNoExpFwdB_has_vjp N (h := 112) (w := 112) Wd1 bd1 εd1 hεd1 γd1 βd1
    Wz1a bz1a Wz1b bz1b Wp1 bp1 εp1 hεp1 γp1 βp1
  have dB2 := mbStridedFwdB_differentiable N (h := 56) (w := 56) We2 be2 εe2 hεe2 γe2 βe2
    Wd2 bd2 εd2 hεd2 γd2 βd2 Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 hεp2 γp2 βp2
  have vB2 := mbStridedFwdB_has_vjp N (h := 56) (w := 56) We2 be2 εe2 hεe2 γe2 βe2
    Wd2 bd2 εd2 hεd2 γd2 βd2 Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 hεp2 γp2 βp2
  have dB3 := mbResidFwdB_differentiable N (h := 56) (w := 56) We3 be3 εe3 hεe3 γe3 βe3
    Wd3 bd3 εd3 hεd3 γd3 βd3 Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3 hεp3 γp3 βp3
  have vB3 := mbResidFwdB_has_vjp N (h := 56) (w := 56) We3 be3 εe3 hεe3 γe3 βe3
    Wd3 bd3 εd3 hεd3 γd3 βd3 Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3 hεp3 γp3 βp3
  have dH := headFwdB_differentiable N (h := 56) (w := 56) Wh bh εh hεh γh βh Wfc bfc
  have vH := headFwdB_has_vjp N (h := 56) (w := 56) Wh bh εh hεh γh βh Wfc bfc
  have v1 := vjp_comp _ _ dStem dB1 vStem vB1
  have d1 := dB1.comp dStem
  have v2 := vjp_comp _ _ d1 dB2 v1 vB2
  have d2 := dB2.comp d1
  have v3 := vjp_comp _ _ d2 dB3 v2 vB3
  have d3 := dB3.comp d2
  exact vjp_comp _ _ d3 dH v3 vH

end Proofs
