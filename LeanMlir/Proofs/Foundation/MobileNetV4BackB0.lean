import LeanMlir.Proofs.Foundation.BackNetFolds

/-! # MobileNetV4 — the batched UIB backward, and the four families as ONE chain

MNv4 was the last net with **no backward of any kind** (`planning/mnv4_verified.md` §8): the
strongest empirical evidence in the repo — forward tied at 1.423e-06, gradient at 0/147 — and
nothing in Lean beyond the render. This file is its phase 1–3, and the fold falls out with it.

## ⭐⭐ THE ANSWER TO §8's OPEN QUESTION: the four families COLLAPSE

§8 asked whether MNv4's four block families — ExtraDW / IB / ConvNeXt-like / FFN — "collapse to one
parameterised theorem or need a case split", and called it "the difference between a small file and
a large one". **They collapse, and the mechanism is `CertLayer.id'`.**

The UIB body is `preDW? → expand → postDW? → project`, and `k = 0` omits a depthwise. Crucially
**both depthwise positions are channel- and shape-preserving** — `preDW : ic → ic`,
`postDW : mid → mid` — so an absent one is not a different composition, it is the **identity
layer** in the same slot:

| family | pre | post | as a chain |
|---|---|---|---|
| ExtraDW | ✓ | ✓ | `chain [preDW, expand, postDW, project]` |
| ConvNeXt-like | ✓ | ✗ | `chain [preDW, expand, id', project]` |
| IB / MBConv | ✗ | ✓ | `chain [id', expand, postDW, project]` |
| FFN | ✗ | ✗ | `chain [id', expand, id', project]` |

⭐ One `mnv4UibBody` takes the two depthwise slots as `CertLayer` arguments; the caller passes
`id'` where the table says `k = 0`. **No case split, no four proofs** — and no dispatch that could
silently disagree with the forward's, which is §3's trap ("a wrong `k = 0` dispatch is silent…
produces a valid net that trains and descends and is not MobileNetV4").

⚠ This is exactly the §6 claim — *"a family from one constructor"* — landing on the proof side,
the way §3i records it landing on the backward render.

## What was genuinely new: a depthwise-bn-RELU stage

Measured before building: the repo had batched depthwise stages at **relu6** (`dwbrB`, MobileNetV2)
and at **swish** (`dwbsB`, EfficientNet), and **none at plain relu**. MNv4 is relu throughout its
14 UIB blocks (⚠ *not* relu6 — `MobileNetV4RenderB` flags this explicitly, and mnv2 sitting one
file over makes it an easy thing to get wrong).

⭐ It cost almost nothing, because `bnReluStage_has_vjp_at` (`ResNet34BackB0`) is **generic in the
op**: it takes any differentiable `op` with a `HasVJP` and builds `relu ∘ bnBatchLA ∘ batchMap op`.
`cbReluB` is that at `flatConv`; `dwbReluB` is the same lemma at `depthwiseFlat`. Zero new analytic
content — one instantiation, plus the backward graph's `.selectPos` (relu's one-sided mask) where
mnv2's uses `.selectMid`.

## Scope

Built here: the two depthwise-relu stages (stride-1 + strided), the four stage `CertLayer`s, the
family-collapsing body, and the skip block. ⚠ **Not** built: the fused stage (swish, stage 0) and
the head. The three stride-2 blocks need the strided body assembled from the same pieces — the
stage is here, the assembly is not.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The one new stage: depthwise → bn → RELU (batched)
-- ════════════════════════════════════════════════════════════════

/-- Batched **depthwise → bn → relu** stage. ⚠ Plain `relu`, not relu6: MNv4's UIB blocks use
    relu where MobileNetV2's use relu6, and `dwbrB` (one file over) is the relu6 one. -/
@[reducible] noncomputable def dwbReluB (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  relu (N * (c * h * w)) ∘ bnBatchLA N c h w ε γ β ∘ batchMap N (depthwiseFlat W b)

/-- Batched **STRIDE-2 depthwise → bn → relu** stage — the depthwise that consumes a UIB block's
    stride. -/
@[reducible] noncomputable def dwbReluBstrided (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * (2 * h) * (2 * w))) → Vec (N * (c * h * w)) :=
  relu (N * (c * h * w)) ∘ bnBatchLA N c h w ε γ β ∘ batchMap N (depthwiseStride2Flat W b)

/-- `dwbReluB`'s `_at` VJP — ⭐ one instantiation of `bnReluStage_has_vjp_at` at `depthwiseFlat`.
    The same lemma `cbReluB_has_vjp_at` uses at `flatConv`; nothing analytic is new. -/
noncomputable def dwbReluB_has_vjp_at (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0) :
    HasVJPAt (dwbReluB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_has_vjp_at N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    (depthwiseFlat_has_vjp W b) ε hε γ β x h_smooth

theorem dwbReluB_differentiableAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0) :
    DifferentiableAt ℝ (dwbReluB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_differentiableAt N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    ε hε γ β x h_smooth

noncomputable def dwbReluBstrided_has_vjp_at (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w))))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2Flat W b) x) k ≠ 0) :
    HasVJPAt (dwbReluBstrided N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_has_vjp_at N (depthwiseStride2Flat W b) (depthwiseStride2Flat_differentiable W b)
    (depthwiseStride2Flat_has_vjp W b) ε hε γ β x h_smooth

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
      = (dwbReluB_has_vjp_at N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [dwbReluBackBatchedGraph, depthwiseBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectPos_faithful _ _ h_smooth]
  simp only [dwbReluB_has_vjp_at, bnReluStage_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt,
    Function.comp_apply]

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
      = (dwbReluBstrided_has_vjp_at N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [dwbReluBstridedBackBatchedGraph, depthwiseStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectPos_faithful _ _ h_smooth]
  simp only [dwbReluBstrided_has_vjp_at, bnReluStage_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt,
    Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The four UIB stages, as CertLayers
-- ════════════════════════════════════════════════════════════════

/-- A depthwise-bn-relu stage as a `CertLayer`. Used at BOTH UIB depthwise positions — pre
    (`c := ic`) and post (`c := mid`) — because the op is channel-parameterised. ⭐ That single
    fact is what §2 records as retiring MNv4's one supposedly-new primitive: a leading depthwise is
    the same constructor at a different channel count. -/
noncomputable def mnv4DWReluLayer (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := dwbReluB N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0
  diff := fun x hx => dwbReluB_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => dwbReluB_has_vjp_at N W b ε hε γ β x hx
  graph := fun x e => dwbReluBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => dwbReluBackBatchedGraph_faithful W b ε hε γ β x e hx

/-- The UIB **expand** (1×1 conv → bn → relu) as a `CertLayer` — `cbReluB`, reused verbatim from
    `ResNet34BackB0`. The kernel extent is a binder, so 1×1 is an argument. -/
noncomputable def mnv4ExpandLayer (N : Nat) {ic mid h w kH kW : Nat}
    (W : Kernel4 mid ic kH kW) (b : Vec mid) (ε : ℝ) (hε : 0 < ε) (γ β : Vec mid) :
    CertLayer (N * (ic * h * w)) (N * (mid * h * w)) where
  fwd := cbReluB N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N mid h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0
  diff := fun x hx => cbReluB_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => cbReluB_has_vjp_at N W b ε hε γ β x hx
  graph := fun x e => cbReluBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => cbReluBackBatchedGraph_faithful W b ε hε γ β x e hx

/-- The UIB **project** (1×1 conv → bn, NO activation) as a `CertLayer` — `projB`, reused verbatim.
    ⚠ Globally certified (`ok = True`): with no activation there is no kink, which is why a UIB
    block has three smoothness families and not four. -/
noncomputable def mnv4ProjectLayer (N : Nat) {mid oc h w kH kW : Nat}
    (W : Kernel4 oc mid kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (mid * h * w)) (N * (oc * h * w)) where
  fwd := projB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (projB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (projB_has_vjp N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => projBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => projBackBatchedGraph_faithful W b ε hε γ β x e

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

/-- ⭐ **The UIB block with its identity skip** — the 11 of MNv4's 14 blocks that have `ic = oc` at
    stride 1. `CertLayer.residual` of the body; the remaining 3 are stride-2 and skipless. -/
noncomputable def mnv4UibSkipBlock (N : Nat) {c mid h w : Nat}
    (preDW : CertLayer (N * (c * h * w)) (N * (c * h * w)))
    (expand : CertLayer (N * (c * h * w)) (N * (mid * h * w)))
    (postDW : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (c * h * w))) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  CertLayer.residual (mnv4UibBody N preDW expand postDW project)

/-- ⭐⭐ **THE MNv4 BLOCK THEOREM.** The UIB block's backward graph denotes its VJP — for every
    family, because the family is an argument. Immediate from `CertLayer.faithful`. -/
theorem mnv4UibSkipBlock_faithful (N : Nat) {c mid h w : Nat}
    (preDW : CertLayer (N * (c * h * w)) (N * (c * h * w)))
    (expand : CertLayer (N * (c * h * w)) (N * (mid * h * w)))
    (postDW : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (c * h * w)))
    (x : Vec (N * (c * h * w))) (hx : (mnv4UibSkipBlock N preDW expand postDW project).ok x)
    (e : SHlo (N * (c * h * w))) :
    den ((mnv4UibSkipBlock N preDW expand postDW project).graph x e)
      = ((mnv4UibSkipBlock N preDW expand postDW project).vjp x hx).backward (den e) :=
  (mnv4UibSkipBlock N preDW expand postDW project).faithful x hx e

/-- **The four families, as four applications.** Type-level check that each of MNv4's block forms
    is this one body with `id'` in the empty depthwise slots — the collapse §8 asked about, made
    concrete. `ExtraDW` is the general case and needs no wrapper. -/
noncomputable def mnv4FamilyIB (N : Nat) {c mid h w : Nat}
    (expand : CertLayer (N * (c * h * w)) (N * (mid * h * w)))
    (postDW : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (c * h * w))) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  mnv4UibSkipBlock N (CertLayer.id' _) expand postDW project

noncomputable def mnv4FamilyConvNeXt (N : Nat) {c mid h w : Nat}
    (preDW : CertLayer (N * (c * h * w)) (N * (c * h * w)))
    (expand : CertLayer (N * (c * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (c * h * w))) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  mnv4UibSkipBlock N preDW expand (CertLayer.id' _) project

noncomputable def mnv4FamilyFFN (N : Nat) {c mid h w : Nat}
    (expand : CertLayer (N * (c * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (c * h * w))) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  mnv4UibSkipBlock N (CertLayer.id' _) expand (CertLayer.id' _) project

/-- **MNv4's stride-1 trunk section, as a type-level check on the block table.** Blocks 4–10 all sit
    at `160 → 160`, `h = 14`, and cover every family: ExtraDW (4, 5, 10), ConvNeXt (6, 8), IB (7),
    FFN (9). Chaining them is one `CertLayer.chain`, and its faithfulness is `chain_faithful`. -/
noncomputable def mnv4Stage14 (N : Nat) {c mid : Nat}
    (extraDW convNeXt ib ffn : CertLayer (N * (c * 14 * 14)) (N * (c * 14 * 14))) :
    CertLayer (N * (c * 14 * 14)) (N * (c * 14 * 14)) :=
  CertLayer.chain [extraDW, extraDW, convNeXt, ib, convNeXt, ffn, extraDW]

end Proofs.StableHLO
