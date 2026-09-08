import LeanMlir.Proofs.Foundation.BackwardMaps

/-! # The ConvNeXt-T backward chains — the ℝ maps the ConvNeXt ties are about

The hand-composed reverse of the committed ConvNeXt-T forward, as plain `def`s on the cotangent:
the block body backward `cnxBlockBodyBack` (the reverse of
`layerScale ∘ project ∘ GELU ∘ expand ∘ LN ∘ depthwise`), the stage-boundary downsample backward
`cnxDownBack` (the reverse of `flatConvStride2 ∘ LN`), and the per-example whole-net chain
`convnextInputGrad` (the reverse of `convNextForwardTCh`, the `[3,3,9,3]` ladder at
96→192→384→768 and 56→28→14→7). The LayerNorm backwards, the GELU and layer-scale diagonal
scales, the four stage backwards and the three downsample backwards enter as *supplied* maps; only
the depthwise, the two 1×1 convs, the 4×4/s4 patchify stem, the GAP and the dense are spelled, so
that the certified tie (`ConvNeXtBackCertifiedTie`, `ConvNeXtWholeBackCertifiedTie`) is a
statement about a NAMED chain of the forward's shape. The channel-LN backward those ties fill the
`lnB` slots with is `chanLNTensor3Back` (`ChannelLNBack.lean`).

⚠ Padding is SYMMETRIC at the downsamples (`flatConvStride2Back`) and the stem is the stride-4
patchify (`flatConvStride4Back`: two scatters, then the reversed-kernel conv); the 7×7 depthwise
is odd, but `EvenKernelConvBack.lean` is where the 4×4 and 2×2 kernels' even-kernel ties live.

Moved here from the float bridge that defined them beside their float twins on 2026-09-08
(`planning/float_second_pass.md`); no number is stated about any of these chains. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The block body and the downsample backwards
-- ════════════════════════════════════════════════════════════════

/-- The ConvNeXt block body input-gradient VJP at a smooth point — the **reverse of
    `convNextBlockBody = layerScale ∘ project ∘ GELU ∘ expand ∘ LN ∘ depthwise`**:

      `depthwiseFlatBack Wdw ∘ lnB ∘ convFlatBack Wex ∘ geluB ∘ convFlatBack Wpr ∘ lsB`

    `lsB = diagBack γls` (the per-channel layer-scale backward); `convFlatBack Wpr` the project back;
    `geluB = diagBack (gelu'(saved))`; `convFlatBack Wex` the expand back; `lnB` the LayerNorm back
    (= BN-back); `depthwiseFlatBack Wdw` the depthwise input-VJP. The full block is
    `residual (body)`, so the block backward is `residual (cnxBlockBodyBack …)`;
    `cnxBlockBodyBack_eq_convNextBlockBody_vjp` pins the supplied slots to their certified
    backwards and shows the chain IS `convNextBlockBody_has_vjp`'s. -/
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

/-- The ConvNeXt downsample input-gradient VJP — the **reverse of `cnxDownChW = flatConvStride2 W ∘ LN`**:
    `lnB ∘ flatConvStride2Back W` (run the strided-conv backward, then the LayerNorm back). -/
noncomputable def cnxDownBack {cin cout h w kH kW : Nat} (W : Kernel4 cout cin kH kW)
    (lnB : Vec (cin * (2 * h) * (2 * w)) → Vec (cin * (2 * h) * (2 * w))) :
    Vec (cout * h * w) → Vec (cin * (2 * h) * (2 * w)) :=
  lnB ∘ flatConvStride2Back (h := h) (w := w) W

-- ════════════════════════════════════════════════════════════════
-- § The per-example whole-net chain (the [3,3,9,3] fold at 224 px, 10 classes)
-- ════════════════════════════════════════════════════════════════

/-- The whole ConvNeXt-T input-gradient VJP at a smooth point — the **exact reverse of
    `convNextForwardTCh`**: `dense ∘ LN ∘ GAP ∘ stage₄ ∘ down₃ ∘ stage₃ ∘ down₂ ∘ stage₂ ∘ down₁ ∘
    stage₁ ∘ LN ∘ stem` reversed. The stem (`flatConvStride4Back sW ∘ lnBstem`, the 4×4/s4 patchify
    backward), GAP and dense endpoints are concrete; the head-LN, stem-LN, 4 stage backwards and 3
    downsample backwards are supplied (the stages fold `residual (cnxBlockBodyBack …)`, the
    downsamples are `cnxDownBack`). The `[3,3,9,3]` structure is in the stage maps' depths; the
    channel/spatial schedule (96→192→384→768, 56→28→14→7) in their dims. ⚠ The reference has no
    head LN, so the tie fills `lnBhead` with `id`. -/
noncomputable def convnextInputGrad {kH kW : Nat} (Wd : Mat 768 10) (sW : Kernel4 96 3 kH kW)
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

end Proofs
