import LeanMlir.Proofs.Foundation.BatchMapVJPAt

/-! # Batched stages — conv/depthwise → true batch-norm (→ swish) at the batched index, with VJPs

The building blocks every batched conv-net forward is written in, at the flat batched index
`N·(c·h·w)`: each stage is `batchMap N` of a proven per-example op, then true batch-norm
(`bnBatchLA`, which couples the batch), then pointwise swish where the net has it.

| stage | forward | VJP |
|---|---|---|
| 1×1 / stride-1 conv → BN → swish | `cbsB` | `cbsB_has_vjp` |
| XLA-`SAME` stride-2 stem conv → BN → swish | `stemB` | `stemB_has_vjp` |
| depthwise (stride 1 / symmetric stride 2) → BN → swish | `dwbsB` / `dwbsSB` | `dwbsB_has_vjp` / `dwbsSB_has_vjp` |
| squeeze-excite | `seB` | `seB_has_vjp` |
| 1×1 projection → BN (no activation) | `projB` | `projB_has_vjp` |

The generic pieces they compose — `batchMap_has_vjp` (block-diagonal VJP of a batch-separable op)
and `bnBatchLA_has_vjp` — are in `BatchMapVJPAt`. EfficientNet-B0, MobileNetV2/V4 and the ResNets
build their blocks from these; the MBConv blocks themselves are in `EfficientNetRenderPC` and
`EfficientNetChainClose`.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Batched stage abbreviations (ℝ-forward), all at the batched index `N·(c·h·w)`.
--   Each is `batchMap N` of a proven per-example op (+ true batch-norm + pointwise swish).
-- ════════════════════════════════════════════════════════════════

/-- Batched conv → bn → swish (1×1 expand / generic stride-1 conv). -/
@[reducible] noncomputable def cbsB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  swish (N * (oc * h * w)) ∘ StableHLO.bnBatchLA N oc h w ε γ β ∘ StableHLO.batchMap N (flatConv W b)

/-- Batched strided (3×3 s2) stem conv → bn → swish (halves spatial). ⚠ At the XLA-`SAME` phase
    (`flatConvStride2Xla` = `decimateOddFlat ∘ flatConv`): the TF-origin B0 pads its stem `(0,1)`,
    and the shipped render has emitted `convStridedXla` there since 2026-08-08. The symmetric
    `flatConvStride2` has the same type and output shape; nothing structural would notice the
    wrong one (re-spelled 2026-09-05, `planning/archive/xla_same_respell_and_blueprint_audit.md`). -/
noncomputable def stemB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  swish (N * (oc * h * w)) ∘ StableHLO.bnBatchLA N oc h w ε γ β ∘
    StableHLO.batchMap N (flatConvStride2Xla W b)

/-- Batched depthwise (stride-1, k×k) → bn → swish. -/
@[reducible] noncomputable def dwbsB (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  swish (N * (c * h * w)) ∘ StableHLO.bnBatchLA N c h w ε γ β ∘ StableHLO.batchMap N (depthwiseFlat W b)

/-- Batched depthwise (stride-2 downsample, k×k) → bn → swish. -/
@[reducible] noncomputable def dwbsSB (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * (2 * h) * (2 * w))) → Vec (N * (c * h * w)) :=
  swish (N * (c * h * w)) ∘ StableHLO.bnBatchLA N c h w ε γ β ∘
    StableHLO.batchMap N (depthwiseStride2Flat W b)

/-- Batched squeeze-excite block `x ⊙ gate(x)` (the proven `seBlockFull`, per example). -/
@[reducible] noncomputable def seB (N : Nat) {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  StableHLO.batchMap N (seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂)

/-- Batched project: 1×1 conv → bn (no swish — the linear bottleneck). -/
@[reducible] noncomputable def projB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  StableHLO.bnBatchLA N oc h w ε γ β ∘ StableHLO.batchMap N (flatConv W b)

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
  bnSwishStage_differentiable N (flatConvStride2Xla W b) (flatConvStride2Xla_differentiable W b) ε hε γ β
noncomputable def stemB_has_vjp (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (stemB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_has_vjp N (flatConvStride2Xla W b) (flatConvStride2Xla_differentiable W b)
    (flatConvStride2Xla_has_vjp W b) ε hε γ β

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

end Proofs
