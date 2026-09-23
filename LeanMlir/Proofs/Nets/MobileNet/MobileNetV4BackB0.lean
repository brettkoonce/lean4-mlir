import LeanMlir.Proofs.Nets.ResNet.ResNet50BackB0
import LeanMlir.Proofs.Foundation.HeadLayers
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4Spec

/-! # MobileNetV4 — the batched UIB backward, and the four families as ONE chain

MNv4 was the last net with **no backward of any kind** (`planning/archive/mnv4_verified.md` §8): the
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
21 UIB blocks (⚠ *not* relu6 — `MobileNetV4RenderB` flags this explicitly, and mnv2 sitting one
file over makes it an easy thing to get wrong).

⭐ It cost almost nothing, because `bnReluStage_has_vjp_at` (`ResNet34BackB0`) is **generic in the
op**: it takes any differentiable `op` with a `HasVJP` and builds `relu ∘ bnBatchLA ∘ batchMap op`.
`cbReluB` is that at `flatConv`; `dwbReluB` is the same lemma at `depthwiseFlat`. Zero new analytic
content — one instantiation, plus the backward graph's `.selectPos` (relu's one-sided mask) where
mnv2's uses `.selectMid`.

## Scope

⚠⚠ **This paragraph was WRONG from the day it was written, and a planning row copied it.** It
said the fused stage, the head and the strided body assembly were not built. All three landed on
**2026-08-10**, the same day, in the three commits that follow the one carrying this header —
`e25a011` (stride-2 blocks), `61eb512` (fused stage) and `411b1a5`, whose own message reads *"the
head — MNv4 complete at stage level"*. Nobody came back to the header, so for four weeks the file
asserted a gap its own commit log had already closed, and `proofs_tier_to_paper_nets.md` §3.6
priced a session against it (`planning/archive/mnv4_proofs_tier.md` §0 — seventh instance of that
pattern, and the cheapest: the declaration list was one `grep` away).
▶ **When a session lands a piece, edit the header that said it was missing, in the same commit.**

**Built here — everything at the BLOCK and STAGE level, which is this file's whole remit:** the two
depthwise-relu stages (stride-1 + strided) and their backward graphs; the four stage `CertLayer`s;
the family-collapsing body and the skip block; the stride-2 form (`mnv4UibPreStridedBody`; Conv-M has no post-strided row); the **fused stage** (swish, stage 0)
with `stemBackBatchedGraph` — the symmetric-padding strided conv-bn-swish backward that closed
EfficientNet's stem hole at the same time; the **head** (`mnv4Head`, its GAP and dense layers both
tying by `rfl`); the table-driven `k = 0` dispatch; and `UibParams`, the row-typed weight record.
Nine of these are in [`tests/AuditAxioms.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/AuditAxioms.lean), 3-axiom clean.

**Not built here — the NET level, which is four other files as of 2026-09-07.** T1 (the whole-net
forward and its input-VJP) is [`Nets/MobileNet/MobileNetV4FullB.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Nets/MobileNet/MobileNetV4FullB.lean) +
`MobileNetV4FullBVJP.lean`, T2 (the typed forward graph at `mnv4FwdChainB`'s tokens) is in the
first of those, T3 (the tie at the emitted gradient nodes, all 233) is
`MobileNetV4StepTieB.lean`, and T6 (the certified whole-net
backward tie) is `MobileNetV4WholeBackCertifiedTieB.lean` + its float chain. Each of them consumes
what is built here, block by block. `planning/archive/mnv4_proofs_tier.md` is the record; ResNet-50 — which
was in exactly this position, block-level only with no per-example legacy — was the file-by-file
precedent, closed over 2026-09-06/07.

⚠ **Two things this file's certificates do NOT give you.** (i) The head models ONE conv stage;
Conv-M's render has **two** (`%h1W` 256→960, then `%hW` 960→1280), so a whole-net use composes
`cbReluLayer` twice. (ii) The **stem** is not here and cannot be a `CertLayer`: no render
emits a gradient into `%x`, so there is no `convStridedXlaBackBatched` token and hence no backward
graph to be faithful to. That is B0's situation exactly, and a net-level forward must compose the stem's VJP by
`vjp_comp_at` rather than by `CertLayer.comp`.
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
  simp only [dwbReluB_has_vjp_at, bnReluStage_has_vjp_at, stage_has_vjp_at, vjp_comp_at_backward,
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
      = (dwbReluBstrided_has_vjp_at N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [dwbReluBstridedBackBatchedGraph, depthwiseStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectPos_faithful _ _ h_smooth]
  simp only [dwbReluBstrided_has_vjp_at, bnReluStage_has_vjp_at, stage_has_vjp_at, vjp_comp_at_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

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

⭐⭐ **The forward stage is `fusedConvB`, EfficientNet's `stemB` shape at SYMMETRIC padding.** Until
2026-09-05 the two nets shared `stemB` (`EfficientNetRenderPC`) outright. B0's stem then moved to
the XLA-`SAME` phase (`flatConvStride2Xla`, the TF-origin convention its render has shipped since
2026-08-08), while MNv4's fused 3×3/s2 stays symmetric — the reference's `fused_ib` passes an
explicit `(p,p)` tuple and `scripts/convention_audit.py` reads the render at `sym` there — so the
stage gets its own name with the same `bnSwishStage_*` lemmas. What was missing repo-wide was the
**backward graph**: `stemBackBatchedGraph` below, at the symmetric `convStridedBackBatched`. ⚠ It
serves MNv4's fused stage only; B0's XLA stem has no batched input-VJP token (no render emits a
gradient into the image), so B0's stem stays un-graph-certified — recorded in `planning/archive/proofs_tier_to_paper_nets.md`. -/

/-- MNv4's fused stage forward: **symmetric** strided k×k conv → bn → swish. -/
noncomputable def fusedConvB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  swish (N * (oc * h * w)) ∘ StableHLO.bnBatchLA N oc h w ε γ β ∘
    StableHLO.batchMap N (flatConvStride2 W b)

theorem fusedConvB_differentiable (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (fusedConvB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_differentiable N (flatConvStride2 W b) (flatConvStride2_differentiable W b) ε hε γ β

noncomputable def fusedConvB_has_vjp (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (fusedConvB N (h := h) (w := w) W b ε γ β) :=
  bnSwishStage_has_vjp N (flatConvStride2 W b) (flatConvStride2_differentiable W b)
    (flatConvStride2_has_vjp W b) ε hε γ β

/-- Batched **strided conv → bn → swish** backward graph — the `cbsBackBatchedGraph` sibling with
    `convStridedBackBatched` for `convBackBatched`, at symmetric padding: MNv4's fused stage. -/
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
      = (fusedConvB_has_vjp N W b ε hε γ β).backward x (den e) := by
  rw [stemBackBatchedGraph, convStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε), swishBack_faithful]
  simp only [fusedConvB_has_vjp, bnSwishStage_has_vjp, vjp_comp_backward, Function.comp_apply]

/-- The fused stage's **k×k strided conv → bn → swish** as a `CertLayer`. ⚠ Globally certified
    (`ok = True`) — swish has no kink, so unlike every UIB stage this one carries no hypothesis. -/
noncomputable def mnv4FusedConvLayer (N : Nat) {ic mid h w kH kW : Nat}
    (W : Kernel4 mid ic kH kW) (b : Vec mid) (ε : ℝ) (hε : 0 < ε) (γ β : Vec mid) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * h * w)) where
  fwd := fusedConvB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (fusedConvB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (fusedConvB_has_vjp N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => stemBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => stemBackBatchedGraph_faithful W b ε hε γ β x e

/-- ⭐ **MNv4's fused stage (stage 0)** — the strided k×k conv-bn-swish, then the 1×1 project.
    No skip: `ic = 32 ≠ 48 = oc` and stride 2, so the stage IS the body. -/
noncomputable def mnv4FusedStage (N : Nat) {ic mid oc h w : Nat}
    (fusedConv : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  fusedConv.comp project

-- ════════════════════════════════════════════════════════════════
-- § THE HEAD — 1×1 conv-bn-relu → GAP → dense
-- ════════════════════════════════════════════════════════════════

/-! MNv4's head is `1×1 conv (256 → 1280) → BN → relu → GAP(7×7) → dense`. The conv stage is
`cbReluLayer` again (conv-bn-relu is conv-bn-relu, and the kernel extent is a binder); GAP and the
classifier are the shared `StableHLO.gapLayer` / `StableHLO.denseLayer`. -/

/-- ⭐ **MNv4's head**: the 1×1 conv-bn-relu, then GAP, then the classifier.

    ⚠ Only the conv stage carries a smoothness condition (its relu); GAP and dense are global. So
    `(mnv4Head …).ok` reduces to the head conv's relu condition alone. -/
noncomputable def mnv4Head (N : Nat) {c oc h w nC : Nat}
    (headConv : CertLayer (N * (c * h * w)) (N * (oc * h * w)))
    (gap : CertLayer (N * (oc * h * w)) (N * oc))
    (cls : CertLayer (N * oc) (N * nC)) :
    CertLayer (N * (c * h * w)) (N * nC) :=
  headConv.comp (gap.comp cls)

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

-- The three forward functions' split: skip (ic = oc, stride 1), pre-strided, post-strided.
-- Recomputed from the table rather than trusted. ⚠ Conv-S was 11 / 2 / 1; Conv-M is 18 / 3 / 0.
#guard (mnv4Blocks.filter (fun s => s.ic == s.oc && !s.stride2)).length = 18
#guard (mnv4Blocks.filter (fun s => s.stride2 && s.preDWk != 0)).length = 3
-- ⚠⚠ **ZERO, where Conv-S had one.** Every Conv-M stride-2 block carries a start-DW, so the
-- POST-STRIDED forward has no rows in the shipped table. The arm is kept because the dispatch is
-- total and Conv-S remains expressible — but nothing here exercises it, so a green corpus is not
-- coverage of that path. ▶ Read this before concluding the three forwards are all gated.
#guard (mnv4Blocks.filter (fun s => s.stride2 && s.preDWk == 0)).length = 0
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

/-- ⭐ **A PRE-STRIDED body built entirely from its table row** — `mnv4BodyOfRow`'s sibling for the
    three stride-2 rows (1, 3, 11), and the row-typed section's third member.

    ⚠ The pre-DW is NOT a slot here: it carries the stride, so it maps `(2h, 2w) ↦ (h, w)` and
    cannot be `id'` — that is the whole reason the collapse stops at stride 2. The POST-DW still is
    a slot, dispatched on `s.postDWk` off the same table row, even though all three Conv-M rows
    happen to fill it.

    ⛔ There is deliberately no `mnv4PostStridedBodyOfRow`: Conv-M has **no** post-strided row
    (Conv-S had one), so a row-typed wrapper for that arm would have no possible argument. -/
noncomputable def mnv4PreStridedBodyOfRow (N : Nat) (s : UibSpec) (p : UibParams s) :
    CertLayer (N * (s.ic * (2 * s.h) * (2 * s.h))) (N * (s.oc * s.h * s.h)) :=
  mnv4UibPreStridedBody N
    (mnv4DWReluStridedLayer (h := s.h) (w := s.h) N p.Wq p.bq p.eq_ p.hq p.gq p.bq2)
    (cbReluLayer (h := s.h) (w := s.h) N p.We p.be p.ee p.he p.ge p.be2)
    (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2)
    (projLayer (h := s.h) (w := s.h) N p.Wz p.bz p.ez p.hz p.gz p.bz2)

end Proofs.StableHLO
