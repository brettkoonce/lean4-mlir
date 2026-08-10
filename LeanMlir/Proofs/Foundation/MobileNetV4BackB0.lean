import LeanMlir.Proofs.Foundation.BackNetFolds
import LeanMlir.Proofs.Codegen.MobileNetV4RenderB

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

-- ════════════════════════════════════════════════════════════════
-- § THE STRIDE-2 BLOCKS — and why `id'` CANNOT collapse these
-- ════════════════════════════════════════════════════════════════

/-! ⚠⚠ **The stride-1 collapse does not extend here, and the reason is the TYPE.**

At stride 1 an absent depthwise is `id'` because the slot is shape-preserving. At stride 2 the
depthwise that carries the stride maps `(2h, 2w) ↦ (h, w)` — a *different type* — so it cannot be
replaced by an identity, and **which** depthwise carries it decides the resolution every later
stage runs at. That is not a dispatch detail; it is two genuinely different compositions:

| form | blocks | who eats the stride | expand runs at |
|---|---|---|---|
| **pre-strided** | 1 (48→80), 11 (160→256) | the **pre**-DW | `h` (already reduced) |
| **post-strided** | 3 (80→160) | the **post**-DW | `2h` (not yet reduced) |

⭐ This mirrors the render exactly — `uibFwdPreStridedB` / `uibFwdPostStridedB` are two functions
for the same reason (`MobileNetV4RenderB`: *"a stride-polymorphic block cannot typecheck"*). The
proof side reproducing that split independently is a small piece of evidence that the split is real
and not a renderer artifact.

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
  vjp := fun x hx => dwbReluBstrided_has_vjp_at N W b ε hε γ β x hx
  graph := fun x e => dwbReluBstridedBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => dwbReluBstridedBackBatchedGraph_faithful W b ε hε γ β x e hx

/-- **Pre-strided UIB body** — MNv4 blocks 1 and 11. The pre-DW carries the stride, so everything
    downstream of it runs at the REDUCED resolution `h`.

    ⭐ `postDW` is still a slot: blocks 1 and 11 both have `postDWk > 0`, but passing `id'` here is
    well-typed and expresses a pre-strided ConvNeXt-family block, so the collapse still applies to
    the *stride-1* slot even though it cannot apply to the strided one. -/
noncomputable def mnv4UibPreStridedBody (N : Nat) {ic mid oc h w : Nat}
    (preDW : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (ic * h * w)))
    (expand : CertLayer (N * (ic * h * w)) (N * (mid * h * w)))
    (postDW : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  preDW.comp (expand.comp (postDW.comp project))

/-- **Post-strided UIB body** — MNv4 block 3 (80→160), the only one. No pre-DW, so the EXPAND runs
    at the full `2h` resolution and the post-DW does the reduction.

    ⚠ Note what this costs: the expand here is a 1×1 over `mid = ic·expand` channels at `2h×2w`,
    i.e. 4× the spatial positions of the pre-strided form. Reading the pre-strided body onto this
    block would be a type error, which is the good case — but reading the *render* wrongly would
    not have been, which is why `MobileNetV4RenderB` splits the two. -/
noncomputable def mnv4UibPostStridedBody (N : Nat) {ic mid oc h w : Nat}
    (expand : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * (2 * h) * (2 * w))))
    (postDW : CertLayer (N * (mid * (2 * h) * (2 * w))) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  expand.comp (postDW.comp project)

/-- ⭐ **The pre-strided block's backward graph denotes its VJP.** No skip (channels change), so the
    block IS the body. Immediate from `CertLayer.faithful`. -/
theorem mnv4UibPreStridedBody_faithful (N : Nat) {ic mid oc h w : Nat}
    (preDW : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (ic * h * w)))
    (expand : CertLayer (N * (ic * h * w)) (N * (mid * h * w)))
    (postDW : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w)))
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hx : (mnv4UibPreStridedBody N preDW expand postDW project).ok x)
    (e : SHlo (N * (oc * h * w))) :
    den ((mnv4UibPreStridedBody N preDW expand postDW project).graph x e)
      = ((mnv4UibPreStridedBody N preDW expand postDW project).vjp x hx).backward (den e) :=
  (mnv4UibPreStridedBody N preDW expand postDW project).faithful x hx e

/-- ⭐ **The post-strided block's backward graph denotes its VJP.** -/
theorem mnv4UibPostStridedBody_faithful (N : Nat) {ic mid oc h w : Nat}
    (expand : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * (2 * h) * (2 * w))))
    (postDW : CertLayer (N * (mid * (2 * h) * (2 * w))) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w)))
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hx : (mnv4UibPostStridedBody N expand postDW project).ok x)
    (e : SHlo (N * (oc * h * w))) :
    den ((mnv4UibPostStridedBody N expand postDW project).graph x e)
      = ((mnv4UibPostStridedBody N expand postDW project).vjp x hx).backward (den e) :=
  (mnv4UibPostStridedBody N expand postDW project).faithful x hx e

-- ════════════════════════════════════════════════════════════════
-- § THE FUSED STAGE (stage 0) — swish, and globally smooth
-- ════════════════════════════════════════════════════════════════

/-! MNv4's stage 0 is `.fusedMbConv 32 48 4 3 2 1 false`: a **regular k×k conv** (not a depthwise)
doing expansion and downsampling at once, then a 1×1 project. `32 → mid = 32·4 = 128 → 48`, stride
2, and ⚠ **swish, not relu** — a deliberate paper deviation that both emitters behind the 84.58%
share (§1b).

⭐ Swish is smooth, so this whole stage is the **globally-certified** kind: `ok = True`, no
smoothness side conditions, and the VJPs are global `HasVJP`s rather than `_at`. That makes stage 0
the cheapest part of MNv4's backward despite being the part §1b records as *missed by the original
scoping*.

⭐⭐ **The forward stage already existed and was already certified** — `stemB` (EfficientNet's
strided conv-bn-swish, `EfficientNetRenderPC`) is exactly this shape, with `stemB_has_vjp` and
`stemB_differentiable` in `EfficientNetChainClose`. What was missing repo-wide is its **backward
graph**: grep found no `stemBackBatchedGraph`. So EfficientNet's own stem was not graph-certified
either, and building it here closes that for both nets. -/

/-- Batched **strided conv → bn → swish** backward graph — the `cbsBackBatchedGraph` sibling with
    `convStridedBackBatched` for `convBackBatched`. Serves MNv4's fused stage AND EfficientNet's
    stem, neither of which had one. -/
noncomputable def stemBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  .convStridedBackBatched (N := N) "%stmW" W b
    (.bnBatchLABack "%stmG" "%stmX" "stmE" ε γ (batchMap N (flatConvStride2 W b) x)
      (.swishBack "%stmSw"
        (bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x)) e))

theorem stemBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    den (stemBackBatchedGraph W b ε γ β x e)
      = (stemB_has_vjp N W b ε hε γ β).backward x (den e) := by
  rw [stemBackBatchedGraph, convStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε), swishBack_faithful]
  simp only [stemB_has_vjp, bnSwishStage_has_vjp, vjp_comp, Function.comp_apply]

/-- The fused stage's **k×k strided conv → bn → swish** as a `CertLayer`. ⚠ Globally certified
    (`ok = True`) — swish has no kink, so unlike every UIB stage this one carries no hypothesis. -/
noncomputable def mnv4FusedConvLayer (N : Nat) {ic mid h w kH kW : Nat}
    (W : Kernel4 mid ic kH kW) (b : Vec mid) (ε : ℝ) (hε : 0 < ε) (γ β : Vec mid) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * h * w)) where
  fwd := stemB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (stemB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (stemB_has_vjp N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => stemBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => stemBackBatchedGraph_faithful W b ε hε γ β x e

/-- ⭐ **MNv4's fused stage (stage 0)** — the strided k×k conv-bn-swish, then the 1×1 project.
    No skip: `ic = 32 ≠ 48 = oc` and stride 2, so the stage IS the body. -/
noncomputable def mnv4FusedStage (N : Nat) {ic mid oc h w : Nat}
    (fusedConv : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  fusedConv.comp project

/-- ⭐ **The fused stage's backward graph denotes its VJP.** ⚠ Note the `ok` here: because both
    constituents are globally smooth, `(mnv4FusedStage …).ok` is `True ∧ True` — the stage is
    certified at **every** input, with nothing to discharge. The only such stage in MNv4. -/
theorem mnv4FusedStage_faithful (N : Nat) {ic mid oc h w : Nat}
    (fusedConv : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w)))
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hx : (mnv4FusedStage N fusedConv project).ok x) (e : SHlo (N * (oc * h * w))) :
    den ((mnv4FusedStage N fusedConv project).graph x e)
      = ((mnv4FusedStage N fusedConv project).vjp x hx).backward (den e) :=
  (mnv4FusedStage N fusedConv project).faithful x hx e

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
    rule `uibFwdSkipB` emits, computed rather than chosen. -/
noncomputable def mnv4PreDWSlot (N : Nat) {c h w kH kW : Nat} (preDWk : Nat)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  if preDWk = 0 then CertLayer.id' _ else mnv4DWReluLayer N W b ε hε γ β

/-- The post-depthwise slot, same dispatch on `postDWk`. -/
noncomputable def mnv4PostDWSlot (N : Nat) {c h w kH kW : Nat} (postDWk : Nat)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  if postDWk = 0 then CertLayer.id' _ else mnv4DWReluLayer N W b ε hε γ β

@[simp] theorem mnv4PreDWSlot_zero (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    mnv4PreDWSlot (h := h) (w := w) N 0 W b ε hε γ β = CertLayer.id' _ := if_pos rfl

@[simp] theorem mnv4PreDWSlot_succ (N : Nat) {c h w kH kW : Nat} (k : Nat)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    mnv4PreDWSlot (h := h) (w := w) N (k + 1) W b ε hε γ β = mnv4DWReluLayer N W b ε hε γ β :=
  if_neg (Nat.succ_ne_zero k)

@[simp] theorem mnv4PostDWSlot_zero (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    mnv4PostDWSlot (h := h) (w := w) N 0 W b ε hε γ β = CertLayer.id' _ := if_pos rfl

@[simp] theorem mnv4PostDWSlot_succ (N : Nat) {c h w kH kW : Nat} (k : Nat)
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    mnv4PostDWSlot (h := h) (w := w) N (k + 1) W b ε hε γ β = mnv4DWReluLayer N W b ε hε γ β :=
  if_neg (Nat.succ_ne_zero k)

/-- ⭐ **A UIB skip block built from the table's two `k`s.** The family is now *computed* from
    `preDWk`/`postDWk` rather than selected by the caller, so a family mis-dispatch has to be a
    wrong number in `mnv4Blocks` — which the `#guard`s below catch — instead of a silent argument. -/
noncomputable def mnv4UibSkipBlockOfKs (N : Nat) {c mid h w kHp kWp kHd kWd kHe kWe kHz kWz : Nat}
    (preDWk postDWk : Nat)
    (Wq : DepthwiseKernel c kHp kWp) (bq : Vec c) (εq : ℝ) (hεq : 0 < εq) (γq βq : Vec c)
    (We : Kernel4 mid c kHe kWe) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz : Kernel4 c mid kHz kWz) (bz : Vec c) (εz : ℝ) (hεz : 0 < εz) (γz βz : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  mnv4UibSkipBlock N
    (mnv4PreDWSlot (h := h) (w := w) N preDWk Wq bq εq hεq γq βq)
    (mnv4ExpandLayer N We be εe hεe γe βe)
    (mnv4PostDWSlot (h := h) (w := w) N postDWk Wd bd εd hεd γd βd)
    (mnv4ProjectLayer N Wz bz εz hεz γz βz)

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

-- ⭐⭐ THE TABLE GUARDS. `MobileNetV4RenderB`'s docstring states the family sequence and the
-- dispatch counts in PROSE; these turn that prose into checks. A wrong `preDWk` is exactly §3's
-- silent defect — same ops, same channel counts, same types, different net — and it now fails at
-- `lake env lean`.
#guard mnv4Blocks.map (·.family) =
  [.extraDW, .extraDW, .ib, .extraDW, .extraDW, .convNeXtLike, .ib, .convNeXtLike, .ffn,
   .extraDW, .extraDW, .extraDW, .ib, .convNeXtLike]

-- The render documents its three forward functions as covering 11 / 2 / 1 blocks. Same split,
-- recomputed from the table rather than trusted: skip (ic = oc, stride 1), pre-strided, post-strided.
#guard (mnv4Blocks.filter (fun s => s.ic == s.oc && !s.stride2)).length = 11
#guard (mnv4Blocks.filter (fun s => s.stride2 && s.preDWk != 0)).length = 2
#guard (mnv4Blocks.filter (fun s => s.stride2 && s.preDWk == 0)).length = 1
-- and those three are ALL of them — no row falls through the dispatch.
#guard mnv4Blocks.length = 14

-- The spatial ladder 56 → 28 → 14 → 7 (`h` is each block's OUTPUT size) and the stride flags.
#guard mnv4Blocks.map (·.h) = [28, 28, 14, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7, 7]
#guard mnv4Blocks.map (·.stride2) =
  [true, false, true, false, false, false, false, false, false, false, true, false, false, false]
-- Every stride-2 block changes channels, which is why none of the three has a skip.
#guard mnv4Blocks.all (fun s => !s.stride2 || s.ic != s.oc)
-- ...and every stride-1 block preserves them, which is why all eleven do.
#guard mnv4Blocks.all (fun s => s.stride2 || s.ic == s.oc)

/-- **MNv4's full block ladder, as a type-level check on `mnv4Blocks`.**

    The spatial ladder is 56 → 28 → 14 → 7 with the reductions at blocks 1, 3 and 11, and the
    channel ladder is 48 → 80 → 160 → 256. Written with nested doublings (`2*(2*(2*h))`) for the
    reason `r50Trunk` documents: Nat multiplication is not definitionally associative in a
    variable, so `8*h` would not line the stage types up.

    If any block's stride, resolution or channel count were transcribed wrongly this would not
    elaborate — the same role `r50Trunk_3463` and `r34Trunk_3463` play for their nets. -/
noncomputable def mnv4BlockLadder (N : Nat) {c₀ c₁ c₂ c₃ h w : Nat}
    (blk1  : CertLayer (N * (c₀ * (2*(2*(2*h))) * (2*(2*(2*w)))))
                       (N * (c₁ * (2*(2*h)) * (2*(2*w)))))
    (blk2  : CertLayer (N * (c₁ * (2*(2*h)) * (2*(2*w)))) (N * (c₁ * (2*(2*h)) * (2*(2*w)))))
    (blk3  : CertLayer (N * (c₁ * (2*(2*h)) * (2*(2*w)))) (N * (c₂ * (2*h) * (2*w))))
    (mid14 : CertLayer (N * (c₂ * (2*h) * (2*w))) (N * (c₂ * (2*h) * (2*w))))
    (blk11 : CertLayer (N * (c₂ * (2*h) * (2*w))) (N * (c₃ * h * w)))
    (tail  : CertLayer (N * (c₃ * h * w)) (N * (c₃ * h * w))) :
    CertLayer (N * (c₀ * (2*(2*(2*h))) * (2*(2*(2*w))))) (N * (c₃ * h * w)) :=
  blk1.comp (blk2.comp (blk3.comp (mid14.comp (blk11.comp tail))))

end Proofs.StableHLO
