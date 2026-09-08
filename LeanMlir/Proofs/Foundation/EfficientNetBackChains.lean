import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The EfficientNet-B0 backward chains — the ℝ maps the B0 ties are about

The hand-composed reverse of the committed EfficientNet-B0 forwards, as plain `def`s on the
cotangent: the per-example MBConv body backward `mbconvBodyBack` (the block where both the
depthwise input-VJP and the squeeze-excite product-rule backward land), the batched three-block
representative `efficientnetInputGradB` (the reverse of `efficientnetForwardB`), and the batched
sixteen-block paper net `efficientnetInputGradB_full` (the reverse of `efficientnetForwardB_full`,
at a variable batch `N` and class count). Each chain keeps its block backwards, its BatchNorm
backwards, its swish backwards and its SE backward as *supplied* maps and spells only the
endpoints, so that the certified tie (`EfficientNetBackCertifiedTie`,
`EfficientNetWholeBackCertifiedTie`, `EfficientNetFullWholeBackCertifiedTie`) is a statement
about a NAMED chain of the forward's shape.

⚠ The stem is XLA-`SAME` (`flatConvStride2XlaBack`, the odd-phase scatter), the TF-origin
convention; the strided depthwises inside the blocks are symmetric and sit in the supplied block
backwards. Neither net has a stem pool, so every batched endpoint is `StableHLO.batchMap N` of a
per-example leaf.

Moved here from the float bridges that defined them beside their float twins on 2026-09-08
(`planning/float_second_pass.md`); no number is stated about any of these chains. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The MBConv body backward (per-example)
-- ════════════════════════════════════════════════════════════════

/-- The EfficientNet MBConv body input-gradient VJP at a smooth point — the **reverse of
    `mbconvBody = (BN∘conv Wp) ∘ seBlockFull ∘ (swish∘BN∘depthwise Wd) ∘ (swish∘BN∘conv We)`**:

      `expandBack ∘ depthwiseBack ∘ seBack ∘ projectBack`

    `projectBack = convFlatBack Wp ∘ bnBp`; `seBack = seB` (the SE product-rule backward, supplied);
    `depthwiseBack = depthwiseFlatBack Wd ∘ bnBd ∘ swBd`; `expandBack = convFlatBack We ∘ bnBe ∘ swBe`.
    The swish backs `swBe`/`swBd` are the saved-derivative `diagBack`s; the BN-backs and the SE-back
    are supplied. `mbconvBodyBack_eq_mbconvBody_vjp` pins every supplied slot to its certified
    backward at the saved activation and shows the chain IS `mbconvBody_has_vjp`'s backward. -/
noncomputable def mbconvBodyBack {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    (bnBe bnBd swBe swBd : Vec (cmid * h * w) → Vec (cmid * h * w))
    (seB : Vec (cmid * h * w) → Vec (cmid * h * w))
    (bnBp : Vec (cout * h * w) → Vec (cout * h * w)) :
    Vec (cout * h * w) → Vec (cin * h * w) :=
  (convFlatBack (h := h) (w := w) We ∘ bnBe ∘ swBe)
  ∘ (depthwiseFlatBack (h := h) (w := w) Wd ∘ bnBd ∘ swBd)
  ∘ seB
  ∘ (convFlatBack (h := h) (w := w) Wp ∘ bnBp)

-- ════════════════════════════════════════════════════════════════
-- § The batched whole-net chains (variable N; the paper net also at a variable class count)
-- ════════════════════════════════════════════════════════════════

/-- **The batched whole-net EfficientNet input-gradient backward** — the reverse of
    `efficientnetForwardB = head ∘ mbResid ∘ mbStrided ∘ mbNoExp ∘ stem` (the representative 3-block
    batched B0): classifier-back → GAP-back → head-conv-bn-swish-back → the three MBConv block backs →
    stem-conv-bn-swish-back. The block backs `b1B`/`b2B`/`b3B` and the stem/head BN+swish backs are
    supplied; the conv/GAP/dense leaves are concrete, `batchMap`-lifted over the `N` examples. -/
noncomputable def efficientnetInputGradB (N : Nat)
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 24 1 1) (Wfc : Mat 1280 10)
    (bnBs swBs : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh swBh : Vec (N * (1280 * 56 * 56)) → Vec (N * (1280 * 56 * 56)))
    (b1B : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2B : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3B : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56))) :
    Vec (N * 10) → Vec (N * (3 * 224 * 224)) :=
  (StableHLO.batchMap N (flatConvStride2XlaBack (h := 112) (w := 112) Ws) ∘ bnBs ∘ swBs)
  ∘ b1B ∘ b2B ∘ b3B
  ∘ (StableHLO.batchMap N (convFlatBack (h := 56) (w := 56) Wh) ∘ bnBh ∘ swBh)
  ∘ StableHLO.batchMap N (gapBack 1280 56 56)
  ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wfc) (0 : Vec 1280))

/-- **The batched whole-net input-gradient backward of the sixteen-block EfficientNet-B0** —
    the reverse of `efficientnetForwardB_full = head ∘ b16 ∘ … ∘ b1 ∘ stem`: classifier-back →
    GAP-back → head-conv-bn-swish-back → the sixteen MBConv block backs → stem-conv-bn-swish-back.
    The block backs and the stem/head BN+swish backs are supplied; the conv/GAP/dense leaves are
    concrete, `batchMap`-lifted over the `N` examples, the stem at the XLA-`SAME` phase. -/
noncomputable def efficientnetInputGradB_full {nCls : Nat} (N : Nat)
    (Ws : Kernel4 32 3 3 3) (Wh : Kernel4 1280 320 1 1) (Wfc : Mat 1280 nCls)
    (bnBs swBs : Vec (N * (32 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (bnBh swBh : Vec (N * (1280 * 7 * 7)) → Vec (N * (1280 * 7 * 7)))
    (b1B : Vec (N * (16 * 112 * 112)) → Vec (N * (32 * 112 * 112)))
    (b2B : Vec (N * (24 * 56 * 56)) → Vec (N * (16 * 112 * 112)))
    (b3B : Vec (N * (24 * 56 * 56)) → Vec (N * (24 * 56 * 56)))
    (b4B : Vec (N * (40 * 28 * 28)) → Vec (N * (24 * 56 * 56)))
    (b5B : Vec (N * (40 * 28 * 28)) → Vec (N * (40 * 28 * 28)))
    (b6B : Vec (N * (80 * 14 * 14)) → Vec (N * (40 * 28 * 28)))
    (b7B : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b8B : Vec (N * (80 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b9B : Vec (N * (112 * 14 * 14)) → Vec (N * (80 * 14 * 14)))
    (b10B : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b11B : Vec (N * (112 * 14 * 14)) → Vec (N * (112 * 14 * 14)))
    (b12B : Vec (N * (192 * 7 * 7)) → Vec (N * (112 * 14 * 14)))
    (b13B : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b14B : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b15B : Vec (N * (192 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    (b16B : Vec (N * (320 * 7 * 7)) → Vec (N * (192 * 7 * 7)))
    : Vec (N * nCls) → Vec (N * (3 * 224 * 224)) :=
  (StableHLO.batchMap N (flatConvStride2XlaBack (h := 112) (w := 112) Ws) ∘ bnBs ∘ swBs)
  ∘ b1B ∘ b2B ∘ b3B ∘ b4B ∘ b5B ∘ b6B ∘ b7B ∘ b8B ∘ b9B ∘ b10B ∘ b11B ∘ b12B ∘ b13B ∘ b14B ∘ b15B ∘ b16B
  ∘ (StableHLO.batchMap N (convFlatBack (h := 7) (w := 7) Wh) ∘ bnBh ∘ swBh)
  ∘ StableHLO.batchMap N (gapBack 1280 7 7)
  ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wfc) (0 : Vec 1280))

end Proofs
