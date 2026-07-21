import LeanMlir.Proofs.Float.DepthwiseBackFloatBridge
import LeanMlir.Proofs.Float.SEBackFloatBridge

/-! # ℝ→Float32 bridge: the WHOLE ConvNeXt-T backward — the [3,3,9,3] input-gradient fold

A3 (planning/a3_backward_deepnet_assembly.md Part 2): the per-example ConvNeXt-T backward (the forward
`convNextForwardT` is `Vec (3·224²) → Vec 10`, per-example like r34/mnv2). The forward is
`dense ∘ LN ∘ GAP ∘ stage₄ ∘ down₃ ∘ stage₃ ∘ down₂ ∘ stage₂ ∘ down₁ ∘ stage₁ ∘ LN ∘ stem`; the
backward is its exact reverse, threaded through one `FloatBridges.comp` chain — the `r34InputGrad`
blueprint at the ConvNeXt topology.

The per-net payoff is the **ConvNeXt block body backward** (`floatBridges_cnxBlockBodyBack`): the
reverse of `layerScale ∘ project ∘ GELU ∘ expand ∘ LN ∘ depthwise` is
`depthwiseBack ∘ lnBack ∘ expandBack ∘ geluBack ∘ projectBack ∘ layerScaleBack`, where the depthwise
stage reverses through the §1e `depthwiseFlatBack` (concrete). The expand/project convs reverse through
`convFlatBack`; GELU is the saved-derivative `diagBack` (`geluB`, supplied); the per-channel layer-scale
is a `diagBack` too (`lsB`, supplied — its float weight is exact); LN-back = BN-back (`lnB`, supplied).
The full block is `residual (body)`, so the block backward is `residual (bodyBack)`
(`floatBridges_cnxBlockBack`). The stage-boundary downsample `flatConvStride2 ∘ LN` reverses to
`lnBack ∘ flatConvStride2Back` (`floatBridges_cnxDownBack`, reusing the §A3 strided-conv backward).

The stem (`flatConvStride4Back` — the 4×4/s4 patchify backward, `StridedConvBackFloatBridge`), GAP and
dense endpoints are concrete; the 4 stages, 3 downsamples and the two LN backwards are supplied as
`FloatBridges` facts — exactly as `r34_grad_floatBridges` supplies its 16 blocks — discharged by the
per-block bridges here (and `floatBridges_bnBack` for the LN-backs).
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The ConvNeXt block body backward (where §1e depthwiseBack lands)
-- ════════════════════════════════════════════════════════════════

/-- The ConvNeXt block body input-gradient VJP at a smooth point — the **reverse of
    `convNextBlockBody = layerScale ∘ project ∘ GELU ∘ expand ∘ LN ∘ depthwise`**:

      `depthwiseFlatBack Wdw ∘ lnB ∘ convFlatBack Wex ∘ geluB ∘ convFlatBack Wpr ∘ lsB`

    `lsB = diagBack γls` (the per-channel layer-scale backward); `convFlatBack Wpr` the project back;
    `geluB = diagBack (gelu'(saved))`; `convFlatBack Wex` the expand back; `lnB` the LayerNorm back
    (= BN-back); `depthwiseFlatBack Wdw` the §1e depthwise input-VJP. -/
noncomputable def cnxBlockBodyBack {c cExp h w kHd kWd : Nat}
    (Wdw : DepthwiseKernel c kHd kWd) (Wex : Kernel4 cExp c 1 1) (Wpr : Kernel4 c cExp 1 1)
    (lnB lsB : Vec (c * h * w) → Vec (c * h * w))
    (geluB : Vec (cExp * h * w) → Vec (cExp * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  depthwiseFlatBack (h := h) (w := w) Wdw
  ∘ lnB
  ∘ convFlatBack (h := h) (w := w) Wex
  ∘ geluB
  ∘ convFlatBack (h := h) (w := w) Wpr
  ∘ lsB

/-- **The ConvNeXt block body backward float-bridges.** One `.comp` chain: the layer-scale `lsB`, the
    project `convFlatBack Wpr`, the GELU `geluB`, the expand `convFlatBack Wex`, the LayerNorm `lnB`,
    and the §1e depthwise `depthwiseFlatBack Wdw` (concrete). The `lsB`/`geluB`/`lnB` smooth/norm pieces
    are supplied (discharged by `floatBridges_diagBack` / `floatBridges_bnBack`). -/
theorem floatBridges_cnxBlockBodyBack {c cExp h w kHd kWd : Nat} (M : FloatModel)
    (Wdw : DepthwiseKernel c kHd kWd) (Wex : Kernel4 cExp c 1 1) (Wpr : Kernel4 c cExp 1 1)
    (lnB lsB : Vec (c * h * w) → Vec (c * h * w))
    (geluB : Vec (cExp * h * w) → Vec (cExp * h * w))
    {wdw wex wpr : ℝ} (hwdw : 0 ≤ wdw) (hwex : 0 ≤ wex) (hwpr : 0 ≤ wpr)
    (hWdw : ∀ ch kh kw, |Wdw ch kh kw| ≤ wdw) (hWex : ∀ o cc kh kw, |Wex o cc kh kw| ≤ wex)
    (hWpr : ∀ o cc kh kw, |Wpr o cc kh kw| ≤ wpr)
    (hc : 0 < c) (hcExp : 0 < cExp) (hh : 0 < h) (hw : 0 < w)
    (hlnB : FloatBridges lnB) (hlsB : FloatBridges lsB) (hgeluB : FloatBridges geluB) :
    FloatBridges (cnxBlockBodyBack Wdw Wex Wpr lnB lsB geluB) := by
  unfold cnxBlockBodyBack
  exact (((((hlsB.comp (floatBridges_convBack (h := h) (w := w) M Wpr hwpr (by positivity) hWpr)).comp
    hgeluB).comp
    (floatBridges_convBack (h := h) (w := w) M Wex hwex (by positivity) hWex)).comp hlnB).comp
    (floatBridges_depthwiseBack (h := h) (w := w) M Wdw hwdw (by positivity) hWdw))

/-- **The ConvNeXt block backward float-bridges** — the full block is `residual (body)`, so the
    backward is `residual (bodyBack)` (`dy ↦ bodyBack(dy) + dy`, the additive skip). -/
theorem floatBridges_cnxBlockBack {c cExp h w kHd kWd : Nat} (M : FloatModel)
    (Wdw : DepthwiseKernel c kHd kWd) (Wex : Kernel4 cExp c 1 1) (Wpr : Kernel4 c cExp 1 1)
    (lnB lsB : Vec (c * h * w) → Vec (c * h * w))
    (geluB : Vec (cExp * h * w) → Vec (cExp * h * w))
    {wdw wex wpr : ℝ} (hwdw : 0 ≤ wdw) (hwex : 0 ≤ wex) (hwpr : 0 ≤ wpr)
    (hWdw : ∀ ch kh kw, |Wdw ch kh kw| ≤ wdw) (hWex : ∀ o cc kh kw, |Wex o cc kh kw| ≤ wex)
    (hWpr : ∀ o cc kh kw, |Wpr o cc kh kw| ≤ wpr)
    (hc : 0 < c) (hcExp : 0 < cExp) (hh : 0 < h) (hw : 0 < w)
    (hlnB : FloatBridges lnB) (hlsB : FloatBridges lsB) (hgeluB : FloatBridges geluB) :
    FloatBridges (Proofs.residual (cnxBlockBodyBack Wdw Wex Wpr lnB lsB geluB)) :=
  FloatBridges.residual M
    (floatBridges_cnxBlockBodyBack M Wdw Wex Wpr lnB lsB geluB hwdw hwex hwpr hWdw hWex hWpr
      hc hcExp hh hw hlnB hlsB hgeluB)

-- ════════════════════════════════════════════════════════════════
-- § The stage-boundary downsample backward
-- ════════════════════════════════════════════════════════════════

/-- The ConvNeXt downsample input-gradient VJP — the **reverse of `cnxDownW = flatConvStride2 W ∘ LN`**:
    `lnB ∘ flatConvStride2Back W` (run the strided-conv backward, then the LayerNorm back). -/
noncomputable def cnxDownBack {cin cout h w : Nat} (W : Kernel4 cout cin 2 2)
    (lnB : Vec (cin * (2 * h) * (2 * w)) → Vec (cin * (2 * h) * (2 * w))) :
    Vec (cout * h * w) → Vec (cin * (2 * h) * (2 * w)) :=
  lnB ∘ flatConvStride2Back (h := h) (w := w) W

/-- **The ConvNeXt downsample backward float-bridges** — the §A3 strided-conv backward then the
    supplied LayerNorm back. -/
theorem floatBridges_cnxDownBack {cin cout h w : Nat} (M : FloatModel) (W : Kernel4 cout cin 2 2)
    (lnB : Vec (cin * (2 * h) * (2 * w)) → Vec (cin * (2 * h) * (2 * w)))
    {wd : ℝ} (hwd : 0 ≤ wd) (hW : ∀ o c kh kw, |W o c kh kw| ≤ wd)
    (hn : 0 < cout * (2 * h) * (2 * w)) (hlnB : FloatBridges lnB) :
    FloatBridges (cnxDownBack W lnB) := by
  unfold cnxDownBack
  exact (floatBridges_flatConvStride2Back (h := h) (w := w) M W hwd hn hW).comp hlnB

-- ════════════════════════════════════════════════════════════════
-- § The whole-net input-gradient VJP (the [3,3,9,3] fold)
-- ════════════════════════════════════════════════════════════════

/-- The whole ConvNeXt-T input-gradient VJP at a smooth point — the **exact reverse of
    `convNextForwardT`**: `dense ∘ LN ∘ GAP ∘ stage₄ ∘ down₃ ∘ stage₃ ∘ down₂ ∘ stage₂ ∘ down₁ ∘
    stage₁ ∘ LN ∘ stem` reversed. The stem (`flatConvStride4Back sW ∘ lnBstem`, the 4×4/s4 patchify
    backward), GAP and dense endpoints are concrete; the head-LN, stem-LN, 4 stage backwards and 3
    downsample backwards are supplied as `FloatBridges` (the stages discharged by folding
    `floatBridges_cnxBlockBack`, the downsamples by `floatBridges_cnxDownBack`, the LN-backs by
    `floatBridges_bnBack`). The `[3,3,9,3]` structure is in the stage maps' depths; the channel/spatial
    schedule (96→192→384→768, 56→28→14→7) in their dims. -/
noncomputable def convnextInputGrad (Wd : Mat 768 10) (sW : Kernel4 96 3 4 4)
    (lnBstem : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (lnBhead : Vec 768 → Vec 768)
    (s1B : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (d1B : Vec (192 * 28 * 28) → Vec (96 * 56 * 56))
    (s2B : Vec (192 * 28 * 28) → Vec (192 * 28 * 28))
    (d2B : Vec (384 * 14 * 14) → Vec (192 * 28 * 28))
    (s3B : Vec (384 * 14 * 14) → Vec (384 * 14 * 14))
    (d3B : Vec (768 * 7 * 7) → Vec (384 * 14 * 14))
    (s4B : Vec (768 * 7 * 7) → Vec (768 * 7 * 7)) :
    Vec 10 → Vec (3 * 224 * 224) :=
  (flatConvStride4Back (h := 56) (w := 56) sW ∘ lnBstem)
  ∘ s1B ∘ d1B ∘ s2B ∘ d2B ∘ s3B ∘ d3B ∘ s4B
  ∘ gapBack 768 7 7
  ∘ lnBhead
  ∘ dense (Mat.transpose Wd) (0 : Vec 768)

set_option maxRecDepth 100000 in
/-- **The whole ConvNeXt-T input-gradient VJP float-bridges** — the per-example ConvNeXt backward. One
    `.comp` chain over `linBack` (dense), the head `lnBhead`, `gapBack`, the 4 supplied stage backwards,
    the 3 supplied downsample backwards, and the concrete stem `flatConvStride4Back sW ∘ lnBstem` (the
    4×4/s4 patchify backward). The deployed float backward of the whole net is within an explicit budget
    of the certified `ℝ` backward — the backward peer of `convNextForwardT`. Closes under
    `[propext, Classical.choice, Quot.sound]`. -/
theorem convnext_grad_floatBridges (M : FloatModel) (Wd : Mat 768 10) (sW : Kernel4 96 3 4 4)
    (lnBstem : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (lnBhead : Vec 768 → Vec 768)
    (s1B : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (d1B : Vec (192 * 28 * 28) → Vec (96 * 56 * 56))
    (s2B : Vec (192 * 28 * 28) → Vec (192 * 28 * 28))
    (d2B : Vec (384 * 14 * 14) → Vec (192 * 28 * 28))
    (s3B : Vec (384 * 14 * 14) → Vec (384 * 14 * 14))
    (d3B : Vec (768 * 7 * 7) → Vec (384 * 14 * 14))
    (s4B : Vec (768 * 7 * 7) → Vec (768 * 7 * 7))
    {wd ws : ℝ} (hwd : 0 ≤ wd) (hWd : ∀ i j, |Wd i j| ≤ wd)
    (hws : 0 ≤ ws) (hsW : ∀ o c kh kw, |sW o c kh kw| ≤ ws)
    (hlnBstem : FloatBridges lnBstem) (hlnBhead : FloatBridges lnBhead)
    (hs1B : FloatBridges s1B) (hd1B : FloatBridges d1B) (hs2B : FloatBridges s2B)
    (hd2B : FloatBridges d2B) (hs3B : FloatBridges s3B) (hd3B : FloatBridges d3B)
    (hs4B : FloatBridges s4B) :
    FloatBridges (convnextInputGrad Wd sW lnBstem lnBhead s1B d1B s2B d2B s3B d3B s4B) := by
  unfold convnextInputGrad
  have hstem : FloatBridges (flatConvStride4Back (h := 56) (w := 56) sW ∘ lnBstem) :=
    hlnBstem.comp (floatBridges_flatConvStride4Back (h := 56) (w := 56) M sW hws (by norm_num) hsW)
  have h0 := ((floatBridges_linBack M Wd hwd (by norm_num) hWd).comp hlnBhead).comp
    (floatBridges_gapBack M 768 7 7 (by norm_num) (by norm_num) (by norm_num))
  have hS4 := h0.comp hs4B
  have hD3 := hS4.comp hd3B
  have hS3 := hD3.comp hs3B
  have hD2 := hS3.comp hd2B
  have hS2 := hD2.comp hs2B
  have hD1 := hS2.comp hd1B
  have hS1 := hD1.comp hs1B
  exact hS1.comp hstem

end Proofs
