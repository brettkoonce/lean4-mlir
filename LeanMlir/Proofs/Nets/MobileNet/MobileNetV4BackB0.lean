import LeanMlir.Proofs.Nets.ResNet.ResNet50BackB0
import LeanMlir.Proofs.Foundation.HeadLayers
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4Spec
import LeanMlir.Proofs.Foundation.IndexCast

/-! # MobileNetV4 — the batched UIB backward, and the four families as one chain

The block- and stage-level backward of MobileNetV4-Conv-M at the batched index: the depthwise-relu
stages, the UIB body, the skip block, the stride-2 form, the fused stage and the head, each a
`CertLayer` whose backward graph is faithful to its certified VJP.

## The four families are one chain

The UIB body is `preDW? → expand → postDW? → project`, and `k = 0` omits a depthwise. Both
depthwise positions are channel- and shape-preserving — `preDW : ic → ic`, `postDW : mid → mid` —
so an absent one is the **identity layer** (`CertLayer.id'`) in the same slot:

| family | pre | post | as a chain |
|---|---|---|---|
| ExtraDW | ✓ | ✓ | `chain [preDW, expand, postDW, project]` |
| ConvNeXt-like | ✓ | ✗ | `chain [preDW, expand, id', project]` |
| IB / MBConv | ✗ | ✓ | `chain [id', expand, postDW, project]` |
| FFN | ✗ | ✗ | `chain [id', expand, id', project]` |

`mnv4UibBody` takes the two depthwise slots as `CertLayer` arguments and the caller passes `id'`
where the table says `k = 0` — one body, no case split, and no dispatch that could disagree with
the forward's (`mnv4BodyOfRow` reads the slots off the `UibSpec` row).

## The two depthwise stages, and where the activation is

timm's `mobilenetv4_conv_medium` (the pinned spec, `planning/mnv4_timm_parity.md`): the pre-DW
(`dw_start`) is depthwise → BN with **no activation** (`dwbB`, globally certified); the post-DW
(`dw_mid`) is depthwise → BN → **relu** (`dwbReluB`, and its strided peer `dwbReluBstrided`).
Relu, not relu6 — MobileNetV2 uses relu6 one file over. The relu stage is
`bnReluStageHasVJPAt` (`Foundation/BatchedStageLayers`, generic in the op) at `depthwiseFlat`;
its backward graph masks with `.selectPos`, relu's one-sided mask.

## Contents

* the depthwise stages — BN-only (`mnv4DWBnLayer`, the pre-DW) and BN-relu, stride-1 and strided
  (`mnv4DWReluLayer`, `mnv4DWReluStridedLayer`, the post-DW) — with backward graphs;
* the family-collapsing body `mnv4UibBody`, the skip block, and the stride-2 form
  `mnv4UibStridedBody` (the post-DW carries the stride, timm's `dw_mid` rule);
* the fused stage (`mnv4FusedStage`, stage 0): `cbReluStridedLayer` then `projLayer`;
* the head `mnv4Head`: conv-bn-relu, GAP, `conv_head`-bn-relu on the pooled features, dense;
* `UibParams`, the row-typed weight record, and the table-driven `k = 0` dispatch.

The net level consumes these block by block: T1 and T2 are `MobileNetV4FullB` /
`MobileNetV4FullBVJP`, T3 is `MobileNetV4StepTieB`, T6 is `MobileNetV4WholeBackCertifiedTieB`.

⚠ The stem is not here and is not a `CertLayer`: no render emits a gradient into `%x`, so there is
no backward graph for it, and a net-level forward composes the stem's VJP by `vjpCompAt` rather
than `CertLayer.comp`.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The pre-DW stage: depthwise → bn, NO activation (timm's `dw_start`)
-- ════════════════════════════════════════════════════════════════

/-- Batched **depthwise → bn** stage, no activation — MobileNetV4's pre-DW (`dw_start` in timm's
    `UniversalInvertedResidual`, whose `BatchNormAct2d` carries `Identity`). -/
@[reducible] noncomputable def dwbB (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  bnBatchLA N c h w ε γ β ∘ batchMap N (depthwiseFlat W b)

theorem dwbB_differentiable (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    Differentiable ℝ (dwbB N (h := h) (w := w) W b ε γ β) :=
  bnStage_differentiable N (depthwiseFlat W b) (depthwiseFlat_differentiable W b) ε hε γ β

noncomputable def dwbBHasVJP (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    HasVJP (dwbB N (h := h) (w := w) W b ε γ β) :=
  bnStageHasVJP N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    (depthwiseFlatHasVJP W b) ε hε γ β

/-- `dwbB`'s backward graph: the BN backward, then the depthwise input-VJP — no mask. -/
noncomputable def dwbBackBatchedGraph {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ _β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .depthwiseBackBatched (N := N) "%dwbW" W b
    (.bnBatchLABack "%dwbG" "%dwbX" "dwbE" ε γ (batchMap N (depthwiseFlat W b) x) e)

theorem dwbBackBatchedGraph_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) :
    den (dwbBackBatchedGraph W b ε γ β x e)
      = (dwbBHasVJP N W b ε hε γ β).backward x (den e) := by
  rw [dwbBackBatchedGraph, depthwiseBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε)]
  simp only [dwbBHasVJP, bnStageHasVJP, vjpComp_backward]

/-- The pre-DW stage as a `CertLayer`. Globally certified (`ok = True`): with no activation there
    is no kink — the `projLayer` situation, at a depthwise. -/
noncomputable def mnv4DWBnLayer (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := dwbB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (dwbB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (dwbBHasVJP N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => dwbBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => dwbBackBatchedGraph_faithful W b ε hε γ β x e

-- ════════════════════════════════════════════════════════════════
-- § The post-DW stage: depthwise → bn → RELU (timm's `dw_mid`)
-- ════════════════════════════════════════════════════════════════

/-- Batched **depthwise → bn → relu** stage. ⚠ Plain `relu`, not relu6: MNv4's UIB blocks use
    relu where MobileNetV2's use relu6, and `dwbrB` (one file over) is the relu6 one. -/
@[reducible] noncomputable def dwbReluB (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  relu (N * (c * h * w)) ∘ bnBatchLA N c h w ε γ β ∘ batchMap N (depthwiseFlat W b)

/-- Batched **STRIDE-2 depthwise → bn → relu** stage — the post-DW of a downsampling UIB block,
    which carries its stride (timm's `dw_mid`). -/
@[reducible] noncomputable def dwbReluBstrided (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * (2 * h) * (2 * w))) → Vec (N * (c * h * w)) :=
  relu (N * (c * h * w)) ∘ bnBatchLA N c h w ε γ β ∘ batchMap N (depthwiseStride2Flat W b)

/-- `dwbReluB`'s `_at` VJP — ⭐ one instantiation of `bnReluStageHasVJPAt` at `depthwiseFlat`.
    The same lemma `cbReluBHasVJPAt` uses at `flatConv`; nothing analytic is new. -/
noncomputable def dwbReluBHasVJPAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0) :
    HasVJPAt (dwbReluB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStageHasVJPAt N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    (depthwiseFlatHasVJP W b) ε hε γ β x h_smooth

theorem dwbReluB_differentiableAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0) :
    DifferentiableAt ℝ (dwbReluB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_differentiableAt N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    ε hε γ β x h_smooth

noncomputable def dwbReluBstridedHasVJPAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w))))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2Flat W b) x) k ≠ 0) :
    HasVJPAt (dwbReluBstrided N (h := h) (w := w) W b ε γ β) x :=
  bnReluStageHasVJPAt N (depthwiseStride2Flat W b) (depthwiseStride2Flat_differentiable W b)
    (depthwiseStride2FlatHasVJP W b) ε hε γ β x h_smooth

theorem dwbReluBstrided_differentiableAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w))))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2Flat W b) x) k ≠ 0) :
    DifferentiableAt ℝ (dwbReluBstrided N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_differentiableAt N (depthwiseStride2Flat W b)
    (depthwiseStride2Flat_differentiable W b) ε hε γ β x h_smooth

/-- `dwbReluB`'s backward graph. ⚠ `.selectPos` (relu's ONE-sided mask) where `dwbrBackBatchedGraph`
    uses `.selectMid` (relu6's two-sided one) — that token is the whole relu-vs-relu6 difference at
    the backward, and swapping them is well-typed. -/
noncomputable def dwbReluBackBatchedGraph {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .depthwiseBackBatched (N := N) "%dwrpW" W b
    (.bnBatchLABack "%dwrpG" "%dwrpX" "dwrpE" ε γ (batchMap N (depthwiseFlat W b) x)
      (.selectPos "%dwrpR" (bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x)) e))

theorem dwbReluBackBatchedGraph_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0) :
    den (dwbReluBackBatchedGraph W b ε γ β x e)
      = (dwbReluBHasVJPAt N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [dwbReluBackBatchedGraph, depthwiseBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectPos_faithful _ _ h_smooth]
  simp only [dwbReluBHasVJPAt, bnReluStageHasVJPAt, stageHasVJPAt, vjpCompAt_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

/-- The strided depthwise-relu stage's backward graph. -/
noncomputable def dwbReluBstridedBackBatchedGraph {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (e : SHlo (N * (c * h * w))) :
    SHlo (N * (c * (2 * h) * (2 * w))) :=
  .depthwiseStridedBackBatched (N := N) "%dwrpsW" W b
    (.bnBatchLABack "%dwrpsG" "%dwrpsX" "dwrpsE" ε γ (batchMap N (depthwiseStride2Flat W b) x)
      (.selectPos "%dwrpsR"
        (bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2Flat W b) x)) e))

theorem dwbReluBstridedBackBatchedGraph_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (e : SHlo (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2Flat W b) x) k ≠ 0) :
    den (dwbReluBstridedBackBatchedGraph W b ε γ β x e)
      = (dwbReluBstridedHasVJPAt N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [dwbReluBstridedBackBatchedGraph, depthwiseStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectPos_faithful _ _ h_smooth]
  simp only [dwbReluBstridedHasVJPAt, bnReluStageHasVJPAt, stageHasVJPAt, vjpCompAt_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The four UIB stages, as CertLayers
-- ════════════════════════════════════════════════════════════════

/-- A depthwise-bn-relu stage as a `CertLayer` — the stride-1 post-DW (`c := mid`). The pre-DW is
    `mnv4DWBnLayer` above: same depthwise, no activation. -/
noncomputable def mnv4DWReluLayer (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := dwbReluB N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0
  diff := fun x hx => dwbReluB_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => dwbReluBHasVJPAt N W b ε hε γ β x hx
  graph := fun x e => dwbReluBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => dwbReluBackBatchedGraph_faithful W b ε hε γ β x e hx

-- The UIB **expand** (1×1 conv → bn → relu) and **project** (1×1 conv → bn) stages are
-- `ResNet34BackB0`'s `cbReluLayer` and `projLayer`. The project stage has no activation and so no
-- kink, which is why a UIB block has three smoothness families and not four.

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE FAMILY COLLAPSE — one body, four families, `id'` in the empty slots
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **The UIB body, for ALL FOUR families at once.**

    `preDW` and `postDW` are `CertLayer` *arguments*, so the caller passes `mnv4DWReluLayer` where
    the block table has `k > 0` and `CertLayer.id'` where it has `k = 0`. ExtraDW, IB, ConvNeXt-like
    and FFN are then four **applications** of this one definition, not four proofs.

    This is only possible because both depthwise positions are shape-preserving: an absent
    depthwise leaves the chain's types unchanged, so `id'` slots in without a case split. -/
noncomputable def mnv4UibBody (N : Nat) {ic mid oc h w : Nat}
    (preDW : CertLayer (N * (ic * h * w)) (N * (ic * h * w)))
    (expand : CertLayer (N * (ic * h * w)) (N * (mid * h * w)))
    (postDW : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) :=
  preDW.comp (expand.comp (postDW.comp project))

-- ⭐ The prose above is CHECKED, but not here: `UibSpec.family` is defined further down (it needs
-- the `UibFamily` inductive), so the guard sits with the other table guards in the dispatch
-- section (the family order of the `h = 14` stride-1 rows).

-- ════════════════════════════════════════════════════════════════
-- § THE STRIDE-2 BLOCKS — and why `id'` CANNOT collapse these
-- ════════════════════════════════════════════════════════════════

/-! ⚠⚠ **The stride-1 collapse does not extend here, and the reason is the TYPE.**

At stride 1 an absent depthwise is `id'` because the slot is shape-preserving. At stride 2 the
depthwise that carries the stride maps `(2h, 2w) ↦ (h, w)` — a *different type* — so it cannot be
replaced by an identity. timm puts the stride on `dw_mid` whenever a block has one, so in all three
Conv-M downsamples (rows 1, 3, 11, all ExtraDW) the optional pre-DW and the expand run at the INPUT
resolution `2h` and the strided post-DW takes it to `h`. The pre-DW is still a slot (at `2h`).

⚠ Until 2026-09-24 these rows were PRE-strided — the pre-DW carried the stride and the expand ran
at `h` — which is a different function with the same parameter shapes.

⚠ All three stride-2 blocks change channels (`ic ≠ oc`), so **none has a skip**: the block IS the
body, with no `CertLayer.residual` wrapper. Adding one would not typecheck, which is the good case.
-/

/-- The STRIDE-2 depthwise-bn-relu stage as a `CertLayer` — the depthwise that carries a UIB
    block's stride. Not an endomorphism (that is the whole point), so it composes via `comp`. -/
noncomputable def mnv4DWReluStridedLayer (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * (2 * h) * (2 * w))) (N * (c * h * w)) where
  fwd := dwbReluBstrided N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2Flat W b) x) k ≠ 0
  diff := fun x hx => dwbReluBstrided_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => dwbReluBstridedHasVJPAt N W b ε hε γ β x hx
  graph := fun x e => dwbReluBstridedBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => dwbReluBstridedBackBatchedGraph_faithful W b ε hε γ β x e hx

/-- **Stride-2 UIB body**: the optional pre-DW and the expand at the input resolution `2h`, the
    post-DW carrying the stride to `h`, the project at `h`. -/
noncomputable def mnv4UibStridedBody (N : Nat) {ic mid oc h w : Nat}
    (preDW : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (ic * (2 * h) * (2 * w))))
    (expand : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * (2 * h) * (2 * w))))
    (postDW : CertLayer (N * (mid * (2 * h) * (2 * w))) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  preDW.comp (expand.comp (postDW.comp project))

-- ════════════════════════════════════════════════════════════════
-- § THE FUSED STAGE (stage 0)
-- ════════════════════════════════════════════════════════════════

/-! MNv4's stage 0 is `.fusedMbConv 32 48 4 3 2 1 false .relu` (timm's `EdgeResidual`): a regular
k×k conv (not a depthwise) doing expansion and downsampling at once, then a 1×1 project.
`32 → mid = 32·4 = 128 → 48`, stride 2, symmetric padding, **relu** — so the strided conv-bn-relu
stage is ResNet's `cbReluStridedLayer` and the project is `projLayer`. Until 2026-09-24 this stage
was swish (inherited from the JAX block being shared with EfficientNetV2). -/

/-- ⭐ **MNv4's fused stage (stage 0)** — the strided k×k conv-bn-relu, then the 1×1 project.
    No skip: `ic = 32 ≠ 48 = oc` and stride 2, so the stage IS the body. -/
noncomputable def mnv4FusedStage (N : Nat) {ic mid oc h w : Nat}
    (fusedConv : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  fusedConv.comp project

-- ════════════════════════════════════════════════════════════════
-- § THE HEAD — conv-bn-relu → GAP → conv_head-bn-relu → dense (timm's order)
-- ════════════════════════════════════════════════════════════════

/-! timm's head: Conv-M's `cn_r1_k1_s1_c960` (1×1 256 → 960, BN, relu) at 7×7, then **global
pool**, then `conv_head` (1×1 960 → 1280) → `norm_head` (BN) → relu on the POOLED `[N, 960, 1, 1]`,
then the classifier. The second conv-bn-relu is `cbReluLayer` at `h = w = 1`, so its BatchNorm
normalises over the batch alone. Until 2026-09-24 `conv_head` ran at 7×7 before the pool, which
batch BN and relu do not commute with. -/

/-- **An index relabelling as a `CertLayer`**: the same vector read at `m` instead of `n` along a
    proved `n = m`. The forward is the `Fin.cast` gather, the VJP `reindexHasVJP`, and the graph
    `castIdx`, which emits no text (`skel` erases indices). The head needs two: the pooled `[N, c]`
    is `conv_head`'s `[N, c, 1, 1]`, and its `[N, oc, 1, 1]` output is the classifier's `[N, oc]`;
    `c * 1 * 1` and `c` are equal but not definitionally so at a variable `c`. -/
noncomputable def castLayer {n m : Nat} (h : n = m) : CertLayer n m where
  fwd := reindexCLM (Fin.cast h.symm)
  ok := fun _ => True
  diff := fun x _ => (reindexCLM (Fin.cast h.symm)).differentiableAt
  vjp := fun x _ => (reindexHasVJP (Fin.cast h.symm)).toHasVJPAt x
  graph := fun _ e => castIdx h.symm e
  faithful := fun x _ e => by
    funext i
    rw [den_castIdx]
    show den e (Fin.cast h.symm.symm i) = ∑ k : Fin m, (if i = Fin.cast h.symm k then den e k else 0)
    have hk : ∀ k : Fin m, i = Fin.cast h.symm k ↔ Fin.cast h i = k := fun k => by
      constructor
      · rintro rfl; exact Fin.ext rfl
      · rintro rfl; exact Fin.ext rfl
    simp only [hk, Finset.sum_ite_eq, Finset.mem_univ, ite_true]

theorem castLayer_fwd_apply {n m : Nat} (h : n = m) (v : Vec n) :
    (castLayer h).fwd v = fun j => v (Fin.cast h.symm j) := rfl

/-- The pooled `[N, c]` read as `[N, c, 1, 1]`. -/
theorem mnv4_pool11 (N c : Nat) : N * c = N * (c * 1 * 1) := by rw [Nat.mul_one, Nat.mul_one]

/-- ⭐ **MNv4's head**: conv-bn-relu, GAP, `conv_head`-bn-relu on the pooled features, classifier,
    with the two `1×1` relabellings between. Only the two conv stages carry a smoothness condition
    (their relus); GAP, the casts and dense are global. -/
noncomputable def mnv4Head (N : Nat) {c mid oc nC h w : Nat}
    (headConv : CertLayer (N * (c * h * w)) (N * (mid * h * w)))
    (gap : CertLayer (N * (mid * h * w)) (N * mid))
    (convHead : CertLayer (N * (mid * 1 * 1)) (N * (oc * 1 * 1)))
    (cls : CertLayer (N * oc) (N * nC)) :
    CertLayer (N * (c * h * w)) (N * nC) :=
  headConv.comp (gap.comp ((castLayer (mnv4_pool11 N mid)).comp
    (convHead.comp ((castLayer (mnv4_pool11 N oc).symm).comp cls))))

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE DISPATCH READS THE TABLE — `mnv4Blocks`, not the caller
-- ════════════════════════════════════════════════════════════════

/-! ⛔ **What this section fixes.** Above, `mnv4UibBody` takes its depthwise slots as *arguments*,
so passing `id'` where block 4's real pre-DW belongs is **well-typed and still certified** — graph
and VJP both move with the caller's arguments, so the theorem stays true *about the wrong net*.
Types catch the stride split (resolution is in the type); they catch nothing about `k = 0` vs
`k > 0`, because that slot is shape-preserving — the very property that made the collapse possible.
That is §3's trap, one level up from the render.

⭐ The fix is to make the slot a **function of the block table's `k`**, so the proof side runs the
same `k = 0` dispatch the render does, off the same `mnv4Blocks` list — one table, not two
readings. `mnv4-fwd-smoke` already pins the render against that table; these `#guard`s pin the
table's own shape, so a bad edit fails at `lake env lean` rather than becoming a silent net. -/

/-- The pre-depthwise **slot**, dispatched on the table's `preDWk`. ⭐ `k = 0` ⇒ `id'` — the same
    rule `uibFwdSkipB` emits, computed rather than chosen. `k > 0` ⇒ the BN-only `mnv4DWBnLayer`. -/
noncomputable def mnv4PreDWSlot (N : Nat) {c h w kH kW : Nat} (preDWk : Nat)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  if preDWk = 0 then CertLayer.id' _ else mnv4DWBnLayer N W b ε hε γ β

/-- The post-depthwise slot, same dispatch on `postDWk`. -/
noncomputable def mnv4PostDWSlot (N : Nat) {c h w kH kW : Nat} (postDWk : Nat)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  if postDWk = 0 then CertLayer.id' _ else mnv4DWReluLayer N W b ε hε γ β

/-- The four families, named — read off the two kernel slots by **exactly** the rule the slots
    dispatch on and the render emits. -/
inductive UibFamily where
  | extraDW | ib | convNeXtLike | ffn
deriving DecidableEq, Repr, BEq

/-- Which family a table row denotes. -/
def UibSpec.family (s : UibSpec) : UibFamily :=
  match s.preDWk, s.postDWk with
  | 0, 0 => .ffn
  | 0, _ => .ib
  | _, 0 => .convNeXtLike
  | _, _ => .extraDW

-- ⭐⭐ THE TABLE GUARDS. `mnv4Blocks`'s docstring (`MobileNetV4Spec`) states the family sequence and the
-- dispatch counts in PROSE; these turn that prose into checks. A wrong `preDWk` is exactly §3's
-- silent defect — same ops, same channel counts, same types, different net — and it now fails at
-- `lake env lean`.
-- ⚠⚠ **REWRITTEN FOR CONV-M (2026-08-14).** `ed5a797` swapped the verified net from Conv-S to
-- Conv-M and left every number below asserting Conv-S's 14 rows, so the corpus build went red on
-- the first push that carried it.
--
-- ▶ **The replacements were NOT re-read off `mnv4Blocks`** — that would make them restate their own
-- input and gate nothing at all. Every one was extracted from **timm 1.0.28**, the pinned spec, by
-- instantiating `mobilenetv4_conv_medium` and walking `model.blocks[1:4]`, reading `dw_start` /
-- `pw_exp` / `dw_mid` / `pw_proj` off each `UniversalInvertedResidual`. All 21 rows agree with timm
-- exactly on `(ic, oc, expand, preDWk, postDWk, h, stride2)`, so the TABLE was right and only these
-- guards were stale. ⚠ Note the naming: our `postDWk` is timm's `dw_mid`, not a third convolution.
#guard mnv4Blocks.map (·.family) =
  [.extraDW, .extraDW, .extraDW, .extraDW, .extraDW, .extraDW, .extraDW, .convNeXtLike, .ffn,
   .convNeXtLike, .extraDW, .extraDW, .extraDW, .extraDW, .ffn, .convNeXtLike, .extraDW,
   .extraDW, .ffn, .ffn, .convNeXtLike]
-- ⭐⭐ **CONV-M USES NO `ib` BLOCK AT ALL**, where Conv-S used three. Stated on its own because it
-- is the one family fact a reader carrying the old table over would be confidently wrong about.
#guard mnv4Blocks.all (fun s => s.family != .ib)
#guard (mnv4Blocks.filter (fun s => s.family == .extraDW)).length = 13
#guard (mnv4Blocks.filter (fun s => s.family == .convNeXtLike)).length = 4
#guard (mnv4Blocks.filter (fun s => s.family == .ffn)).length = 4

-- ⭐ The family order of the seven `h = 14` stride-1 rows (blocks 4–10), pinned. A docstring once
-- named Conv-S's families here (an `ib` block, of which Conv-M has none) for four weeks while the
-- guards above already said Conv-M; this makes the prose a check.
#guard (mnv4Blocks.filter (fun s => s.h == 14 && !s.stride2)).map (·.family) =
  [.extraDW, .extraDW, .extraDW, .extraDW, .convNeXtLike, .ffn, .convNeXtLike]

-- The two forward forms' split: skip (ic = oc, stride 1) and strided (the post-DW carries it).
-- Recomputed from the table rather than trusted: 18 / 3, and every strided row has a post-DW to
-- carry the stride and a pre-DW in its slot.
#guard (mnv4Blocks.filter (fun s => s.ic == s.oc && !s.stride2)).length = 18
#guard (mnv4Blocks.filter (fun s => s.stride2 && s.postDWk != 0)).length = 3
#guard (mnv4Blocks.filter (fun s => s.stride2 && s.preDWk != 0)).length = 3
-- and those three are ALL of them — no row falls through the dispatch.
#guard mnv4Blocks.length = 21

-- The spatial ladder 56 → 28 → 14 → 7 (`h` is each block's OUTPUT size) and the stride flags.
#guard mnv4Blocks.map (·.h) =
  [28, 28, 14, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
#guard mnv4Blocks.map (·.stride2) =
  [true, false, true, false, false, false, false, false, false, false,
   true, false, false, false, false, false, false, false, false, false, false]
-- Every stride-2 block changes channels, which is why none of the three has a skip.
#guard mnv4Blocks.all (fun s => !s.stride2 || s.ic != s.oc)
-- ...and every stride-1 block preserves them, which is why all eighteen do.
#guard mnv4Blocks.all (fun s => s.stride2 || s.ic == s.oc)

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ WEIGHT WIRING — the parameters are TYPED BY THEIR TABLE ROW
-- ════════════════════════════════════════════════════════════════

/-! ⛔ **The gap this closes.** A block builder that reads the *dispatch* from the table but takes
its weights as separate arguments lets a caller pair row 4's `k`s with row 7's widths. The dispatch
was table-driven; the wiring was not.

⭐ **Fix: index the parameter record by the row.** Every width in `UibParams s` is a *projection of
`s`* — `s.ic`, `s.oc`, `s.ic * s.expand`, `s.preDWk`, `s.postDWk`. A record with widths that
disagree with its row **cannot be constructed**, so the block builder needs no side conditions and
no `#guard`: it is impossible by typing rather than checked after the fact.

⚠ **What this still does not pin**, stated precisely so the next reader does not overclaim: rows
4, 5 and 10 are all `160 → 160, expand 4`, and 4/10 share `k = 3,3`. Their records are therefore
the *same type*, so swapping those two blocks' weights typechecks. Typing pins **shape**
(channels, expand ratio, kernel extents, resolution); it cannot pin **identity** between rows that
are shape-identical. Closing that needs the weights to come from one indexed array — a renderer
concern, since the render already folds `mnv4Blocks` in order. -/

/-- **One UIB block's parameters, typed by its table row.** Every width is a projection of `s`, so
    a record whose widths disagree with the row is not constructible. Bias-free convs still carry a
    `b` because the stage vocabulary takes one; the render binds it to `%zb{c}`. -/
structure UibParams (s : UibSpec) where
  /-- pre-depthwise, at `s.ic` channels and `s.preDWk` extent (degenerate when `k = 0`). -/
  Wq : DepthwiseKernel s.ic s.preDWk s.preDWk
  bq : Vec s.ic
  eq_ : ℝ
  hq : 0 < eq_
  gq : Vec s.ic
  bq2 : Vec s.ic
  /-- expand `1x1`, `s.ic -> s.ic * s.expand`. -/
  We : Kernel4 (s.ic * s.expand) s.ic 1 1
  be : Vec (s.ic * s.expand)
  ee : ℝ
  he : 0 < ee
  ge : Vec (s.ic * s.expand)
  be2 : Vec (s.ic * s.expand)
  /-- post-depthwise, at the EXPANDED width and `s.postDWk` extent. -/
  Wd : DepthwiseKernel (s.ic * s.expand) s.postDWk s.postDWk
  bd : Vec (s.ic * s.expand)
  ed : ℝ
  hd : 0 < ed
  gd : Vec (s.ic * s.expand)
  bd2 : Vec (s.ic * s.expand)
  /-- project `1x1`, `s.ic * s.expand -> s.oc`. -/
  Wz : Kernel4 s.oc (s.ic * s.expand) 1 1
  bz : Vec s.oc
  ez : ℝ
  hz : 0 < ez
  gz : Vec s.oc
  bz2 : Vec s.oc

/-- ⭐⭐ **A UIB body built ENTIRELY from its table row.** Dispatch from `s.preDWk`/`s.postDWk`,
    widths and resolution from `s`, weights from a record that cannot disagree with `s`. Nothing
    here is a free argument: given `s`, the only freedom left is the numeric values.

    ⚠ This is the BODY (`ic -> oc`). The identity skip is `CertLayer.residual` on top and needs
    `oc = ic`, which holds for exactly the eighteen non-`stride2` rows (guarded below) — applied by
    the caller at a concrete row, where it is `rfl` and needs no transport. -/
noncomputable def mnv4BodyOfRow (N : Nat) (s : UibSpec) (p : UibParams s) :
    CertLayer (N * (s.ic * s.h * s.h)) (N * (s.oc * s.h * s.h)) :=
  mnv4UibBody N
    (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2)
    (cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2)
    (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2)
    (projLayer (h := s.h) (w := s.h) N p.Wz p.bz p.ez p.hz p.gz p.bz2)

-- Every non-`stride2` row has `oc = ic`, so `CertLayer.residual` applies to all eighteen of them.
#guard (mnv4Blocks.filter (fun s => !s.stride2)).all (fun s => s.oc == s.ic)

/-- ⭐ **A STRIDED body built entirely from its table row** — `mnv4BodyOfRow`'s sibling for the
    three stride-2 rows (1, 3, 11). The pre-DW is a slot at the input resolution `2h`, dispatched
    on `s.preDWk`; the post-DW carries the stride and so is NOT a slot (it cannot be `id'`, which is
    why every strided row must have one — guarded above). -/
noncomputable def mnv4StridedBodyOfRow (N : Nat) (s : UibSpec) (p : UibParams s) :
    CertLayer (N * (s.ic * (2 * s.h) * (2 * s.h))) (N * (s.oc * s.h * s.h)) :=
  mnv4UibStridedBody N
    (mnv4PreDWSlot (h := 2 * s.h) (w := 2 * s.h) N s.preDWk p.Wq p.bq p.eq_ p.hq p.gq p.bq2)
    (cbReluLayer (h := 2 * s.h) (w := 2 * s.h) N p.We p.be p.ee p.he p.ge p.be2)
    (mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wd p.bd p.ed p.hd p.gd p.bd2)
    (projLayer (h := s.h) (w := s.h) N p.Wz p.bz p.ez p.hz p.gz p.bz2)

end Proofs.StableHLO
