import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The ConvNeXt-T backward chains — the ℝ maps the ConvNeXt ties are about

The hand-composed reverse of the committed ConvNeXt-T forward, as plain `def`s on the cotangent:
the block body backward `cnxBlockBodyBack` (the reverse of
`layerScale ∘ project ∘ GELU ∘ expand ∘ LN ∘ depthwise`), the stage-boundary downsample backward
`cnxDownBack` (the reverse of `flatConvStride2 ∘ LN`), the per-example whole-net chain
`convnextInputGrad` (the reverse of `convNextForwardTCh`, the `[3,3,9,3]` ladder at
96→192→384→768 and 56→28→14→7, `nC` classes) and its batched form `convnextInputGradB` (the same
chain at each of `B` examples, every saved-activation slot a `StableHLO.batchMapAux B` lift). The
LayerNorm backwards, the GELU and layer-scale diagonal scales, the four stage backwards and the
three downsample backwards enter as *supplied* maps; only the depthwise, the two 1×1 convs, the
4×4/s4 patchify stem, the GAP and the dense are spelled, so that the certified ties
(`ConvNeXtBackCertifiedTie`, `ConvNeXtWholeBackCertifiedTie`, `ConvNeXtWholeBackCertifiedTieB`)
are statements about a NAMED chain of the forward's shape. The channel-LN backward those ties fill
the `lnB` slots with is `chanLNTensor3Back` (`ChannelLNBack.lean`).

⚠ Padding is SYMMETRIC at the downsamples (`flatConvStride2Back`) and the stem is the stride-4
patchify (`flatConvStride4Back`: two scatters, then the reversed-kernel conv); the 7×7 depthwise
is odd, but `EvenKernelConvBack.lean` is where the 4×4 and 2×2 kernels' even-kernel ties live.

Moved here from the float bridge that defined them beside their float twins on 2026-09-08
(`planning/archive/float_second_pass.md`); no number is stated about any of these chains. -/

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
-- § The per-example whole-net chain (the [3,3,9,3] fold at 224 px, `nC` classes)
-- ════════════════════════════════════════════════════════════════

/-- The whole ConvNeXt-T input-gradient VJP at a smooth point — the **exact reverse of
    `convNextForwardTCh`**: `dense ∘ LN ∘ GAP ∘ stage₄ ∘ down₃ ∘ stage₃ ∘ down₂ ∘ stage₂ ∘ down₁ ∘
    stage₁ ∘ LN ∘ stem` reversed. The stem (`flatConvStride4Back sW ∘ lnBstem`, the 4×4/s4 patchify
    backward), GAP and dense endpoints are concrete; the head-LN, stem-LN, 4 stage backwards and 3
    downsample backwards are supplied (the stages fold `residual (cnxBlockBodyBack …)`, the
    downsamples are `cnxDownBack`). The `[3,3,9,3]` structure is in the stage maps' depths; the
    channel/spatial schedule (96→192→384→768, 56→28→14→7) in their dims; the class count `nC` is
    a binder (10 at Imagenette, 1000 at the `convnextin_*` artifacts). The tie fills `lnBhead`
    with the head LayerNorm's backward at the GAP output (`rowLNVecFlatBack 1 768`). -/
noncomputable def convnextInputGrad {kH kW nC : Nat} (Wd : Mat 768 nC) (sW : Kernel4 96 3 kH kW)
    (lnBstem : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (lnBhead : Vec 768 → Vec 768)
    (s1B : Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (d1B : Vec (192 * 28 * 28) → Vec (96 * 56 * 56))
    (s2B : Vec (192 * 28 * 28) → Vec (192 * 28 * 28))
    (d2B : Vec (384 * 14 * 14) → Vec (192 * 28 * 28))
    (s3B : Vec (384 * 14 * 14) → Vec (384 * 14 * 14))
    (d3B : Vec (768 * 7 * 7) → Vec (384 * 14 * 14))
    (s4B : Vec (768 * 7 * 7) → Vec (768 * 7 * 7)) :
    Vec nC → Vec (3 * 224 * 224) :=
  (flatConvStride4Back (h := 56) (w := 56) sW ∘ lnBstem)
  ∘ s1B ∘ d1B ∘ s2B ∘ d2B ∘ s3B ∘ d3B ∘ s4B
  ∘ gapBack 768 7 7
  ∘ lnBhead
  ∘ dense (Mat.transpose Wd) (0 : Vec 768)

-- ════════════════════════════════════════════════════════════════
-- § The batched whole-net chain — a variable batch `B`, every slot lifted at its saved batch
-- ════════════════════════════════════════════════════════════════

/-- **THE BATCHED WHOLE-NET ConvNeXt-T INPUT GRADIENT** — `convnextInputGrad` at each of `B`
    examples, stage by stage, as the batched render computes it. The three input-independent
    leaves (the stem's reversed-kernel conv, GAP, the dense head) are `StableHLO.batchMap B` of
    the per-example leaf. Every saved-activation slot is `StableHLO.batchMapAux B` of a
    per-example FAMILY `saved ↦ (cotangent ↦ gradient)` at a batched saved activation: `lnBstem`
    at `a0` (the stem conv's output), the four stage backwards `s1B … s4B` at their stage inputs
    `a1 a3 a5 a7`, the three downsample backwards `d1B … d3B` at `a2 a4 a6`, and `lnBhead` at `a9`
    (the GAP output). Where the per-example chain fills a slot at ONE saved value, this one fills
    it at `B` of them, which is the only way a batch enters — LayerNorm is per-example and no
    ConvNeXt op couples examples, the honesty argument `ConvNeXtStepTieGB` makes for the batched
    T3 tie. `B` and `nC` are variables: this chain carries no batch numeral and no class count.
    `convnextInputGradB_eq_batchMap_convNextForwardTCh_vjp` (`ConvNeXtWholeBackCertifiedTieB.lean`)
    says it IS the certified gradient of `batchMap B convNextForwardTCh`. -/
noncomputable def convnextInputGradB (B : Nat) {kH kW nC : Nat} (Wd : Mat 768 nC)
    (sW : Kernel4 96 3 kH kW)
    (lnBstem : Vec (96 * 56 * 56) → Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (a0 : Vec (B * (96 * 56 * 56)))
    (lnBhead : Vec 768 → Vec 768 → Vec 768) (a9 : Vec (B * 768))
    (s1B : Vec (96 * 56 * 56) → Vec (96 * 56 * 56) → Vec (96 * 56 * 56))
    (a1 : Vec (B * (96 * 56 * 56)))
    (d1B : Vec (96 * 56 * 56) → Vec (192 * 28 * 28) → Vec (96 * 56 * 56))
    (a2 : Vec (B * (96 * 56 * 56)))
    (s2B : Vec (192 * 28 * 28) → Vec (192 * 28 * 28) → Vec (192 * 28 * 28))
    (a3 : Vec (B * (192 * 28 * 28)))
    (d2B : Vec (192 * 28 * 28) → Vec (384 * 14 * 14) → Vec (192 * 28 * 28))
    (a4 : Vec (B * (192 * 28 * 28)))
    (s3B : Vec (384 * 14 * 14) → Vec (384 * 14 * 14) → Vec (384 * 14 * 14))
    (a5 : Vec (B * (384 * 14 * 14)))
    (d3B : Vec (384 * 14 * 14) → Vec (768 * 7 * 7) → Vec (384 * 14 * 14))
    (a6 : Vec (B * (384 * 14 * 14)))
    (s4B : Vec (768 * 7 * 7) → Vec (768 * 7 * 7) → Vec (768 * 7 * 7))
    (a7 : Vec (B * (768 * 7 * 7))) :
    Vec (B * nC) → Vec (B * (3 * 224 * 224)) :=
  StableHLO.batchMap B (flatConvStride4Back (h := 56) (w := 56) sW)
  ∘ StableHLO.batchMapAux B lnBstem a0
  ∘ StableHLO.batchMapAux B s1B a1
  ∘ StableHLO.batchMapAux B d1B a2
  ∘ StableHLO.batchMapAux B s2B a3
  ∘ StableHLO.batchMapAux B d2B a4
  ∘ StableHLO.batchMapAux B s3B a5
  ∘ StableHLO.batchMapAux B d3B a6
  ∘ StableHLO.batchMapAux B s4B a7
  ∘ StableHLO.batchMap B (gapBack 768 7 7)
  ∘ StableHLO.batchMapAux B lnBhead a9
  ∘ StableHLO.batchMap B (dense (Mat.transpose Wd) (0 : Vec 768))

end Proofs
