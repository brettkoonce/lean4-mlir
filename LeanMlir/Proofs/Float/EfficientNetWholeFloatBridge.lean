import LeanMlir.Proofs.Float.EnetFloatBridge
import LeanMlir.Proofs.Float.Resnet34WholeFloatBridge
import LeanMlir.Proofs.Codegen.EfficientNetRenderPC
import LeanMlir.Proofs.Float.EfficientNetBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE batched EfficientNet-B0 FORWARD

The forward peer of the EfficientNet backward float story (`EfficientNetBackFloatBridge.lean`), and
the EfficientNet entry in the whole-net forward fold (`Resnet34WholeFloatBridge.lean` is the worked
example). The repo had the EfficientNet forward float bridge only at *MBConv-body* level
(`floatBridges_mbconvBody`, `EnetFloatBridge.lean`); this folds the whole representative B0 net.

Unlike r34/mnv2, EfficientNet's render is genuinely **batched** with **true batch-norm**
(`bnBatchLA`, reduce μ/var over the batch+spatial axes — it COUPLES the batch). So the whole net
lives at the batched index `N·(c·h·w)`: every batch-separable op is `batchMap N` of its proven
per-example op (`FloatBridges.batchMap` lifts each op-bridge), the pointwise swish is already
block-diagonal at the batched index (`floatBridges_swish` directly), and the one batch-coupled op —
true batch-norm — is **supplied as a `FloatBridges` fact** (exactly as r34 supplies its stem BN and
the backward supplies its `bnBack`s; `rsqrt`/`exp` have no IEEE spec, so the honest move is to defer
the BN as an operating-point hypothesis, discharged separately).

The whole net is `headFwdB ∘ mbResidFwdB ∘ mbStridedFwdB ∘ mbNoExpFwdB ∘ stemB`; this file states the
bridge **on the actual `efficientnetForwardB` def** (not a fresh skeleton), so the forward whole-net
now stands exactly as strong as `efficientnetForwardB_has_vjp` (the certified ℝ-forward VJP).

One forward op-bridge was missing and is built here (a thin wrap, the strided-depthwise peer of r34's
`floatBridges_flatConvStride2`): `floatBridges_depthwiseStride2Flat` — `depthwiseStride2Flat =
decimateFlat ∘ depthwiseFlat` is the stride-1 depthwise read at the decimated coordinate, so its
`FloatClose` is `floatClose_depthwise` (on the `2h×2w` grid) evaluated at `decimateIdx`.
-/

namespace Proofs

open scoped Real
open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § Forward stride-2 depthwise as a `FloatBridges`  (mbStrided's downsample)
-- ════════════════════════════════════════════════════════════════

/-- The float stride-2 depthwise: decimate the float stride-1 depthwise (the float peer of
    `depthwiseStride2Flat = decimateFlat ∘ depthwiseFlat`). The depthwise analogue of
    `FloatModel.flatConvStride2F`. -/
noncomputable def FloatModel.depthwiseStride2FlatF {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w) :=
  decimateFlat c h w ∘ (M.depthwiseFlatF (h := 2 * h) (w := 2 * w) W b)

/-- **Stride-2 depthwise is `FloatClose`.** `depthwiseStride2Flat W b = decimateFlat ∘ depthwiseFlat`
    selects the stride-1 depthwise's output at the even (`decimateIdx`) coordinates, so every
    magnitude/perturbation bound is `floatClose_depthwise` (on the `2h×2w` grid) read at `decimateIdx`.
    Same depthwise fan-in `layerBudget` (the `kH·kW` fan-in is stride-independent). The depthwise peer
    of `floatClose_flatConvStride2`. -/
theorem floatClose_depthwiseStride2Flat {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (b : Vec c) {w' bb A : ℝ}
    (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hA : 0 ≤ A) (hn : 0 < c * (2 * h) * (2 * w))
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |b ch| ≤ bb) :
    FloatClose A
      (layerAct (kH * kW) w' bb A + layerBudget M.u (kH * kW) w' bb A 0)
      (depthwiseStride2Flat (h := h) (w := w) W b) (M.depthwiseStride2FlatF (h := h) (w := w) W b)
      (fun e => layerBudget M.u (kH * kW) w' bb A e) := by
  obtain ⟨hm, he⟩ := floatClose_depthwise (h := 2 * h) (w := 2 * w) M W b hw' hbb hA hn hW hb
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · exact hm v hv (decimateIdx c h w i)
  · exact he vt va e hva hvt hd (decimateIdx c h w i)

/-- **Stride-2 depthwise float-bridges** (output magnitude `layerAct + layerBudget`, fan-in `kH·kW`). -/
theorem floatBridges_depthwiseStride2Flat {c h w kH kW : Nat} (M : FloatModel)
    (W : DepthwiseKernel c kH kW) (b : Vec c) {w' bb : ℝ}
    (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hn : 0 < c * (2 * h) * (2 * w))
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |b ch| ≤ bb) :
    FloatBridges (depthwiseStride2Flat (h := h) (w := w) W b) :=
  fun _A hA => ⟨_, _, _,
    add_nonneg (layerAct_nonneg hw' hbb hA) (layerBudget_nonneg M.u_nonneg hw' hbb hA le_rfl),
    floatClose_depthwiseStride2Flat M W b hw' hbb hA hn hW hb⟩

-- ════════════════════════════════════════════════════════════════
-- § The batched stage bridges (each `batchMap N` of a per-example op + supplied BN)
-- ════════════════════════════════════════════════════════════════

/-- Batched conv → bn → swish float-bridges. The `batchMap`-lifted conv (`FloatBridges.batchMap` of
    `floatBridges_flatConv`), the supplied true-batch-norm `bnBatchLA`, then the block-diagonal swish. -/
theorem floatBridges_cbsB {ic oc h w kH kW : Nat} (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig) (hn : 0 < ic * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ bb)
    (hbn : FloatBridges (StableHLO.bnBatchLA N oc h w ε γ β)) :
    FloatBridges (cbsB N (h := h) (w := w) W b ε γ β) := by
  unfold cbsB
  exact ((FloatBridges.batchMap N
      (floatBridges_flatConv (h := h) (w := w) M W b hw' hbb hn hW hb)).comp hbn).comp
    (floatBridges_swish M fsig hesig hsig)

/-- Batched stride-2 stem conv → bn → swish float-bridges (the `flatConvStride2` downsample). -/
theorem floatBridges_stemB {ic oc h w kH kW : Nat} (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig)
    (hn : 0 < ic * (2 * h) * (2 * w))
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ bb)
    (hbn : FloatBridges (StableHLO.bnBatchLA N oc h w ε γ β)) :
    FloatBridges (stemB N (h := h) (w := w) W b ε γ β) := by
  unfold stemB
  exact ((FloatBridges.batchMap N
      (floatBridges_flatConvStride2 (h := h) (w := w) M W b hw' hbb hn hW hb)).comp hbn).comp
    (floatBridges_swish M fsig hesig hsig)

/-- Batched depthwise → bn → swish float-bridges (stride-1). -/
theorem floatBridges_dwbsB {c h w kH kW : Nat} (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig) (hn : 0 < c * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |b ch| ≤ bb)
    (hbn : FloatBridges (StableHLO.bnBatchLA N c h w ε γ β)) :
    FloatBridges (dwbsB N (h := h) (w := w) W b ε γ β) := by
  unfold dwbsB
  exact ((FloatBridges.batchMap N
      (floatBridges_depthwise (h := h) (w := w) M W b hw' hbb hn hW hb)).comp hbn).comp
    (floatBridges_swish M fsig hesig hsig)

/-- Batched stride-2 depthwise → bn → swish float-bridges (the mbStrided downsample). -/
theorem floatBridges_dwbsSB {c h w kH kW : Nat} (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig)
    (hn : 0 < c * (2 * h) * (2 * w))
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW : ∀ ch kh kw, |W ch kh kw| ≤ w') (hb : ∀ ch, |b ch| ≤ bb)
    (hbn : FloatBridges (StableHLO.bnBatchLA N c h w ε γ β)) :
    FloatBridges (dwbsSB N (h := h) (w := w) W b ε γ β) := by
  unfold dwbsSB
  exact ((FloatBridges.batchMap N
      (floatBridges_depthwiseStride2Flat (h := h) (w := w) M W b hw' hbb hn hW hb)).comp hbn).comp
    (floatBridges_swish M fsig hesig hsig)

/-- Batched squeeze-excite block float-bridges (`batchMap N` of `floatBridges_seBlockFull`). -/
theorem floatBridges_seB {c h w r : Nat} (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hc : 0 < c) (hr : 0 < r) (hn : 0 < c * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ bb)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ bb) :
    FloatBridges (seB N (h := h) (w := w) W₁ b₁ W₂ b₂) := by
  unfold seB
  exact FloatBridges.batchMap N
    (floatBridges_seBlockFull M fsig W₁ b₁ W₂ b₂ hw' hbb hesig hhw hc hr hn hsig hW₁ hb₁ hW₂ hb₂)

/-- Batched project conv → bn float-bridges (no swish — the linear bottleneck). -/
theorem floatBridges_projB {ic oc h w kH kW : Nat} (N : Nat) (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    {w' bb : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hn : 0 < ic * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ bb)
    (hbn : FloatBridges (StableHLO.bnBatchLA N oc h w ε γ β)) :
    FloatBridges (projB N (h := h) (w := w) W b ε γ β) := by
  unfold projB
  exact (FloatBridges.batchMap N
    (floatBridges_flatConv (h := h) (w := w) M W b hw' hbb hn hW hb)).comp hbn

-- ════════════════════════════════════════════════════════════════
-- § The batched per-block bridges (peers of the backward's batched MBConv blocks)
-- ════════════════════════════════════════════════════════════════

/-- **MBConv1 (no expand) float-bridges** — `projB ∘ seB ∘ dwbsB`. The depthwise-bn-swish, the SE
    gate, and the project-bn, each batched; the two BNs supplied. -/
theorem floatBridges_mbNoExpFwdB {ic oc h w kHd kWd r : Nat} (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hic : 0 < ic) (hr : 0 < r) (hn : 0 < ic * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ bb)
    (hWz₁ : ∀ i j, |Wz₁ i j| ≤ w') (hbz₁ : ∀ j, |bz₁ j| ≤ bb)
    (hWz₂ : ∀ i j, |Wz₂ i j| ≤ w') (hbz₂ : ∀ j, |bz₂ j| ≤ bb)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ w') (hbp : ∀ o, |bp o| ≤ bb)
    (hbnD : FloatBridges (StableHLO.bnBatchLA N ic h w εd γd βd))
    (hbnP : FloatBridges (StableHLO.bnBatchLA N oc h w εp γp βp)) :
    FloatBridges (mbNoExpFwdB N (h := h) (w := w) Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbNoExpFwdB
  exact ((floatBridges_dwbsB N M fsig Wd bd εd γd βd hw' hbb hesig hn hsig hWd hbd hbnD).comp
    (floatBridges_seB N M fsig Wz₁ bz₁ Wz₂ bz₂ hw' hbb hesig hhw hic hr hn hsig
      hWz₁ hbz₁ hWz₂ hbz₂)).comp
    (floatBridges_projB N M Wp bp εp γp βp hw' hbb hn hWp hbp hbnP)

/-- **MBConv6 strided float-bridges** — `projB ∘ seB ∘ dwbsSB ∘ cbsB`. Expand-bn-swish (at `2h×2w`),
    strided depthwise-bn-swish, SE, project-bn; the three BNs supplied. -/
theorem floatBridges_mbStridedFwdB {ic mid oc h w kHd kWd r : Nat} (N : Nat) (M : FloatModel)
    (fsig : ℝ → ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hmid : 0 < mid) (hr : 0 < r)
    (hnE : 0 < ic * (2 * h) * (2 * w)) (hnD : 0 < mid * (2 * h) * (2 * w)) (hn : 0 < mid * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ w') (hbe : ∀ o, |be o| ≤ bb)
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ bb)
    (hWz₁ : ∀ i j, |Wz₁ i j| ≤ w') (hbz₁ : ∀ j, |bz₁ j| ≤ bb)
    (hWz₂ : ∀ i j, |Wz₂ i j| ≤ w') (hbz₂ : ∀ j, |bz₂ j| ≤ bb)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ w') (hbp : ∀ o, |bp o| ≤ bb)
    (hbnE : FloatBridges (StableHLO.bnBatchLA N mid (2 * h) (2 * w) εe γe βe))
    (hbnD : FloatBridges (StableHLO.bnBatchLA N mid h w εd γd βd))
    (hbnP : FloatBridges (StableHLO.bnBatchLA N oc h w εp γp βp)) :
    FloatBridges (mbStridedFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbStridedFwdB
  exact (((floatBridges_cbsB N M fsig We be εe γe βe hw' hbb hesig hnE hsig hWe hbe hbnE).comp
    (floatBridges_dwbsSB N M fsig Wd bd εd γd βd hw' hbb hesig hnD hsig hWd hbd hbnD)).comp
    (floatBridges_seB N M fsig Wz₁ bz₁ Wz₂ bz₂ hw' hbb hesig hhw hmid hr hn hsig
      hWz₁ hbz₁ hWz₂ hbz₂)).comp
    (floatBridges_projB N M Wp bp εp γp βp hw' hbb hn hWp hbp hbnP)

/-- **MBConv6 residual float-bridges** — `residual (projB ∘ seB ∘ dwbsB ∘ cbsB)`. The matched-channel
    skip via `FloatBridges.residual` over the expand → depthwise → SE → project body; the three BNs
    supplied. The dominant MBConv block. -/
theorem floatBridges_mbResidFwdB {c mid h w kHd kWd r : Nat} (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (_hc : 0 < c) (hmid : 0 < mid) (hr : 0 < r)
    (hnC : 0 < c * h * w) (hn : 0 < mid * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ w') (hbe : ∀ o, |be o| ≤ bb)
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ w') (hbd : ∀ ch, |bd ch| ≤ bb)
    (hWz₁ : ∀ i j, |Wz₁ i j| ≤ w') (hbz₁ : ∀ j, |bz₁ j| ≤ bb)
    (hWz₂ : ∀ i j, |Wz₂ i j| ≤ w') (hbz₂ : ∀ j, |bz₂ j| ≤ bb)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ w') (hbp : ∀ o, |bp o| ≤ bb)
    (hbnE : FloatBridges (StableHLO.bnBatchLA N mid h w εe γe βe))
    (hbnD : FloatBridges (StableHLO.bnBatchLA N mid h w εd γd βd))
    (hbnP : FloatBridges (StableHLO.bnBatchLA N c h w εp γp βp)) :
    FloatBridges (mbResidFwdB N (h := h) (w := w) We be εe γe βe Wd bd εd γd βd
      Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp) := by
  unfold mbResidFwdB
  exact FloatBridges.residual M
    ((((floatBridges_cbsB N M fsig We be εe γe βe hw' hbb hesig hnC hsig hWe hbe hbnE).comp
      (floatBridges_dwbsB N M fsig Wd bd εd γd βd hw' hbb hesig hn hsig hWd hbd hbnD)).comp
      (floatBridges_seB N M fsig Wz₁ bz₁ Wz₂ bz₂ hw' hbb hesig hhw hmid hr hn hsig
        hWz₁ hbz₁ hWz₂ hbz₂)).comp
      (floatBridges_projB N M Wp bp εp γp βp hw' hbb hn hWp hbp hbnP))

/-- **Head float-bridges** — `batchMap(dense) ∘ batchMap(GAP) ∘ cbsB`. The 1×1 conv-bn-swish, the
    batched global-avg-pool, and the batched dense classifier; the head BN supplied. -/
theorem floatBridges_headFwdB {c oc h w nC : Nat} (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig)
    (hhw : 0 < h * w) (hoc : 0 < oc) (hnC : 0 < c * h * w)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hWh : ∀ o c kh kw, |Wh o c kh kw| ≤ w') (hbh : ∀ o, |bh o| ≤ bb)
    (hWfc : ∀ i j, |Wfc i j| ≤ w') (hbfc : ∀ j, |bfc j| ≤ bb)
    (hbnH : FloatBridges (StableHLO.bnBatchLA N oc h w εh γh βh)) :
    FloatBridges (headFwdB N (h := h) (w := w) Wh bh εh γh βh Wfc bfc) := by
  unfold headFwdB
  exact ((floatBridges_cbsB N M fsig Wh bh εh γh βh hw' hbb hesig hnC hsig hWh hbh hbnH).comp
    (FloatBridges.batchMap N (floatBridges_gap (c := oc) (h := h) (w := w) M hoc hhw))).comp
    (FloatBridges.batchMap N (floatBridges_dense M Wfc bfc hw' hbb hoc hWfc hbfc))

-- ════════════════════════════════════════════════════════════════
-- § The whole batched EfficientNet-B0 forward (the stem→3×MBConv→head fold)
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 100000 in
/-- **The whole batched EfficientNet-B0 forward float-bridges** — the representative B0 that exercises
    every B0 element: stride-2 stem, MBConv1 no-expand, MBConv6 stride-2, MBConv6 5×5 + residual, 1×1
    head + GAP + dense, all batched with true batch-norm and squeeze-excite. One `.comp` chain over the
    per-block batched bridges; every conv/depthwise/SE/GAP/dense enters `batchMap`-lifted, swish is
    block-diagonal at the batched index, and the ten true-batch-norms are supplied as `FloatBridges`
    facts (the one batch-coupled op, deferred exactly as r34 defers its stem BN). The deployed float
    forward of the whole net is within an explicit budget of the certified ℝ forward.

    Stated on the `∘`-composition of the blocks — which **IS** `efficientnetForwardB` (its nested-
    application spelling is definitionally this composition; `FloatBridges.comp` builds exactly this
    composition, so the proof closes structurally without re-reducing the whole net). The forward peer
    of `efficientnetForwardB_has_vjp`, stated identically; closes under
    `[propext, Classical.choice, Quot.sound]`. -/
theorem efficientnetForwardB_floatBridges (N : Nat) (M : FloatModel) (fsig : ℝ → ℝ)
    (Ws : Kernel4 32 3 3 3) (bs : Vec 32) (εs : ℝ) (γs βs : Vec 32)
    (Wd1 : DepthwiseKernel 32 3 3) (bd1 : Vec 32) (εd1 : ℝ) (γd1 βd1 : Vec 32)
    (Wz1a : Mat 32 8) (bz1a : Vec 8) (Wz1b : Mat 8 32) (bz1b : Vec 32)
    (Wp1 : Kernel4 16 32 1 1) (bp1 : Vec 16) (εp1 : ℝ) (γp1 βp1 : Vec 16)
    (We2 : Kernel4 96 16 1 1) (be2 : Vec 96) (εe2 : ℝ) (γe2 βe2 : Vec 96)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 : ℝ) (γd2 βd2 : Vec 96)
    (Wz2a : Mat 96 4) (bz2a : Vec 4) (Wz2b : Mat 4 96) (bz2b : Vec 96)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 : ℝ) (γp2 βp2 : Vec 24)
    (We3 : Kernel4 144 24 1 1) (be3 : Vec 144) (εe3 : ℝ) (γe3 βe3 : Vec 144)
    (Wd3 : DepthwiseKernel 144 5 5) (bd3 : Vec 144) (εd3 : ℝ) (γd3 βd3 : Vec 144)
    (Wz3a : Mat 144 6) (bz3a : Vec 6) (Wz3b : Mat 6 144) (bz3b : Vec 144)
    (Wp3 : Kernel4 24 144 1 1) (bp3 : Vec 24) (εp3 : ℝ) (γp3 βp3 : Vec 24)
    (Wh : Kernel4 1280 24 1 1) (bh : Vec 1280) (εh : ℝ) (γh βh : Vec 1280)
    (Wfc : Mat 1280 10) (bfc : Vec 10)
    {w' bb esig : ℝ} (hw' : 0 ≤ w') (hbb : 0 ≤ bb) (hesig : 0 ≤ esig)
    (hsig : ∀ t, |fsig t - sigmoidScalar t| ≤ esig)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ w') (hbs : ∀ o, |bs o| ≤ bb)
    (hWd1 : ∀ ch kh kw, |Wd1 ch kh kw| ≤ w') (hbd1 : ∀ ch, |bd1 ch| ≤ bb)
    (hWz1a : ∀ i j, |Wz1a i j| ≤ w') (hbz1a : ∀ j, |bz1a j| ≤ bb)
    (hWz1b : ∀ i j, |Wz1b i j| ≤ w') (hbz1b : ∀ j, |bz1b j| ≤ bb)
    (hWp1 : ∀ o c kh kw, |Wp1 o c kh kw| ≤ w') (hbp1 : ∀ o, |bp1 o| ≤ bb)
    (hWe2 : ∀ o c kh kw, |We2 o c kh kw| ≤ w') (hbe2 : ∀ o, |be2 o| ≤ bb)
    (hWd2 : ∀ ch kh kw, |Wd2 ch kh kw| ≤ w') (hbd2 : ∀ ch, |bd2 ch| ≤ bb)
    (hWz2a : ∀ i j, |Wz2a i j| ≤ w') (hbz2a : ∀ j, |bz2a j| ≤ bb)
    (hWz2b : ∀ i j, |Wz2b i j| ≤ w') (hbz2b : ∀ j, |bz2b j| ≤ bb)
    (hWp2 : ∀ o c kh kw, |Wp2 o c kh kw| ≤ w') (hbp2 : ∀ o, |bp2 o| ≤ bb)
    (hWe3 : ∀ o c kh kw, |We3 o c kh kw| ≤ w') (hbe3 : ∀ o, |be3 o| ≤ bb)
    (hWd3 : ∀ ch kh kw, |Wd3 ch kh kw| ≤ w') (hbd3 : ∀ ch, |bd3 ch| ≤ bb)
    (hWz3a : ∀ i j, |Wz3a i j| ≤ w') (hbz3a : ∀ j, |bz3a j| ≤ bb)
    (hWz3b : ∀ i j, |Wz3b i j| ≤ w') (hbz3b : ∀ j, |bz3b j| ≤ bb)
    (hWp3 : ∀ o c kh kw, |Wp3 o c kh kw| ≤ w') (hbp3 : ∀ o, |bp3 o| ≤ bb)
    (hWh : ∀ o c kh kw, |Wh o c kh kw| ≤ w') (hbh : ∀ o, |bh o| ≤ bb)
    (hWfc : ∀ i j, |Wfc i j| ≤ w') (hbfc : ∀ j, |bfc j| ≤ bb)
    (hbnS : FloatBridges (StableHLO.bnBatchLA N 32 112 112 εs γs βs))
    (hbnD1 : FloatBridges (StableHLO.bnBatchLA N 32 112 112 εd1 γd1 βd1))
    (hbnP1 : FloatBridges (StableHLO.bnBatchLA N 16 112 112 εp1 γp1 βp1))
    (hbnE2 : FloatBridges (StableHLO.bnBatchLA N 96 112 112 εe2 γe2 βe2))
    (hbnD2 : FloatBridges (StableHLO.bnBatchLA N 96 56 56 εd2 γd2 βd2))
    (hbnP2 : FloatBridges (StableHLO.bnBatchLA N 24 56 56 εp2 γp2 βp2))
    (hbnE3 : FloatBridges (StableHLO.bnBatchLA N 144 56 56 εe3 γe3 βe3))
    (hbnD3 : FloatBridges (StableHLO.bnBatchLA N 144 56 56 εd3 γd3 βd3))
    (hbnP3 : FloatBridges (StableHLO.bnBatchLA N 24 56 56 εp3 γp3 βp3))
    (hbnH : FloatBridges (StableHLO.bnBatchLA N 1280 56 56 εh γh βh)) :
    FloatBridges (headFwdB N (h := 56) (w := 56) Wh bh εh γh βh Wfc bfc ∘
      mbResidFwdB N (h := 56) (w := 56) We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3
        Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3 γp3 βp3 ∘
      mbStridedFwdB N (h := 56) (w := 56) We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2
        Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 γp2 βp2 ∘
      mbNoExpFwdB N (h := 112) (w := 112) Wd1 bd1 εd1 γd1 βd1 Wz1a bz1a Wz1b bz1b
        Wp1 bp1 εp1 γp1 βp1 ∘
      stemB N (h := 112) (w := 112) Ws bs εs γs βs) := by
  have hStem := floatBridges_stemB N M fsig Ws bs εs γs βs hw' hbb hesig (by norm_num) hsig
    hWs hbs hbnS
  have hB1 := floatBridges_mbNoExpFwdB N M fsig Wd1 bd1 εd1 γd1 βd1 Wz1a bz1a Wz1b bz1b
    Wp1 bp1 εp1 γp1 βp1 hw' hbb hesig (by norm_num) (by norm_num) (by norm_num) (by norm_num) hsig
    hWd1 hbd1 hWz1a hbz1a hWz1b hbz1b hWp1 hbp1 hbnD1 hbnP1
  have hB2 := floatBridges_mbStridedFwdB N M fsig We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2
    Wz2a bz2a Wz2b bz2b Wp2 bp2 εp2 γp2 βp2 hw' hbb hesig (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) hsig
    hWe2 hbe2 hWd2 hbd2 hWz2a hbz2a hWz2b hbz2b hWp2 hbp2 hbnE2 hbnD2 hbnP2
  have hB3 := floatBridges_mbResidFwdB N M fsig We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3
    Wz3a bz3a Wz3b bz3b Wp3 bp3 εp3 γp3 βp3 hw' hbb hesig (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) hsig
    hWe3 hbe3 hWd3 hbd3 hWz3a hbz3a hWz3b hbz3b hWp3 hbp3 hbnE3 hbnD3 hbnP3
  have hHead := floatBridges_headFwdB N M fsig Wh bh εh γh βh Wfc bfc hw' hbb hesig
    (by norm_num) (by norm_num) (by norm_num) hsig hWh hbh hWfc hbfc hbnH
  exact (((hStem.comp hB1).comp hB2).comp hB3).comp hHead

end Proofs
