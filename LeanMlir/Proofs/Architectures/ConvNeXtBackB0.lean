import LeanMlir.Proofs.Architectures.ConvNeXt
import LeanMlir.Proofs.Architectures.ConvNeXtFullT
import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Architectures.EfficientNetBackB0

/-! # ConvNeXt whole-block backward-graph faithfulness (per-example / batch-1)

The ConvNeXt analogue of `mbconvResidual_backGraph_faithful` (EfficientNet) and
`r34*BackBatchedGraph_faithful` (ResNet-34): a *backward* StableHLO graph that
denotes the proven whole-block VJP.

ConvNeXt's whole verified stack is **per-example / batch-1** — LayerNorm here is
the per-example separable `layerNormForward` (= `bnForward` on the feature axis),
so NONE of EfficientNet's `batchMap`/`bnBatchLA` batched machinery is needed
(`ConvNeXtChainClose.lean:8`). So this file targets the per-example VJP
`convNextBlock_has_vjp` directly, modeled on the per-example section of
`EfficientNetBackB0.lean` (`residualBackGraph`, `convBnSwishBackGraph`,
`mbconvResidual_backGraph_faithful`).

One capstone:

  * **Identity/residual block** — `cnxResidBlockBackGraph` denotes
    `convNextBlock_has_vjp`'s backward. The block is `residual (block body)` with an
    identity skip, so the brick is `residualBackGraph (bodyBack …) dy`, closed via
    `residualBackGraph_faithful`.

⚠ **§2n (2026-07-31) removed the second one.** `cnxDownBackGraph` denoted `cnxDownW_has_vjp`'s
backward — the SCALAR-LN downsample — and went with the scalar chain. It is not ported: the
shipped downsample normalises with `chanLNTensor3`, whose graph-side backward tie does not exist
(§2m built the render and the math VJP; `ConvNeXtBackCertifiedTie.chanLNTensor3Back_eq_chanLN_vjp`
is the MATH-side tie, not a `den`-level one). **So the channel-LN downsample has no graph-side
backward capstone.** That is a real gap, and it is one the drop EXPOSED rather than created: it was
always the scalar net that had the capstone. What this block-level file still covers — the ch9
representative's residual block — is unaffected, because that net has no downsample.

The block body is `layerScale ∘ project(1×1) ∘ gelu ∘ expand(1×1) ∘ LN ∘
depthwise(7×7)`; everything is smooth (GELU is smooth, conv/layerScale linear, LN
smooth given `ε>0`), so the body VJP is the unconditional `vjp_comp` chain
`convNextBlockBody_has_vjp` and the only side condition is the LayerNorm positivity
`0 < εn`. The LN backward is the one non-`rfl` op: `layerNorm_has_vjp` is
*definitionally* `bn_has_vjp`, so `bnBack_faithful_fn` (the `∑ pdiv` bridge) closes it.

Same scalar-LN representation caveat as `convNextBlockBody`'s doc-comment.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt block body backward graph (per-example)
-- ════════════════════════════════════════════════════════════════

/-- The ConvNeXt block-body backward graph `D⁻¹ ∘ LN⁻¹ ∘ EX⁻¹ ∘ GE⁻¹ ∘ PR⁻¹ ∘ LS⁻¹`,
    each op's backward applied at its forward input activation (outermost backward
    token = earliest forward op `depthwise`; innermost child applied to the cotangent
    subgraph `e`). The reverse-order chain of `convNextBlockBody`'s VJP:

      `depthwiseBack ∘ bnBack(LN) ∘ convBack(expand) ∘ geluBack ∘ convBack(project)
        ∘ layerScaleF`

    `layerScaleF` is the (input-independent diagonal) layer-scale backward; the two
    `convBack`s are the 1×1 expand/project; `bnBack` is the LayerNorm backward
    (LN = BN on the feature axis). -/
noncomputable def cnxBlockBodyBackGraph {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) : SHlo (c * h * w) :=
  let d  := depthwiseFlat (h := h) (w := w) Wdw bdw x              -- LN's input
  let nl := layerNormForward (c * h * w) εn γn βn d                -- EX's input
  let ex := flatConv (h := h) (w := w) Wex bex nl                  -- GE's input
  let ge := gelu (cExp * h * w) ex                                 -- PR's input
  .depthwiseBack "%cnxWdw" Wdw bdw x
    (.bnBack "%cnxGn" "%cnxXn" "cnxE" εn γn d
      (.convBack "%cnxWex" Wex bex nl
        (.geluBack "%cnxGe" ex
          (.convBack "%cnxWpr" Wpr bpr ge
            (.layerScaleF "%cnxGls" γls e)))))

/-- **ConvNeXt block-body backward-graph faithfulness.** The reverse-order graph
    denotes the proven `convNextBlockBody_has_vjp` backward, under `0 < εn`. The LN
    backward is the one non-`rfl` op (`bnBack_faithful_fn`); the rest
    (`depthwiseBack`/`convBack`/`geluBack`/`layerScaleF`) are `rfl`-faithful per-op
    tokens, and `layerScaleF` denotes the input-independent diagonal layer-scale
    backward. -/
theorem cnxBlockBodyBackGraph_faithful {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x : Vec (c * h * w)) (e : SHlo (c * h * w)) :
    den (cnxBlockBodyBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x e)
      = (convNextBlockBody_has_vjp Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls).backward
          x (den e) := by
  simp only [cnxBlockBodyBackGraph, convNextBlockBody_has_vjp, vjp_comp,
    depthwiseBack_faithful, bnBack_faithful_fn (β := βn) (hε := hεn),
    convBack_faithful, geluBack_faithful, layerScaleF_faithful, Function.comp_apply]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Identity/residual block capstone (per-example)
-- ════════════════════════════════════════════════════════════════

/-- The whole ConvNeXt residual block backward graph (block body + identity skip):
    `residualBackGraph (bodyBack … (%dy)) dy`. The identity skip contributes the
    cotangent verbatim; `addV` sums the two paths. -/
noncomputable def cnxResidBlockBackGraph {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x dy : Vec (c * h * w)) : SHlo (c * h * w) :=
  residualBackGraph
    (cnxBlockBodyBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x (.operand "%dy" dy)) dy

/-- **The whole per-example ConvNeXt residual block: backward graph ↔ proven VJP**
    (identity/residual capstone), no hypotheses beyond `0 < εn`. Assembles the body
    backward graph (`cnxBlockBodyBackGraph`) + the identity skip into the proven
    `convNextBlock_has_vjp` backward via `residualBackGraph_faithful`. -/
theorem cnxResidBlockBackGraph_faithful {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w))
    (x dy : Vec (c * h * w)) :
    den (cnxResidBlockBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x dy)
      = (convNextBlock_has_vjp Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls).backward x dy :=
  residualBackGraph_faithful
    (convNextBlockBody Wdw bdw εn γn βn Wex bex Wpr bpr γls)
    (convNextBlockBody_differentiable Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls)
    (convNextBlockBody_has_vjp Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls)
    x dy
    (cnxBlockBodyBackGraph Wdw bdw εn γn βn Wex bex Wpr bpr γls x (.operand "%dy" dy))
    (cnxBlockBodyBackGraph_faithful Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls x
      (.operand "%dy" dy))

end Proofs.StableHLO
