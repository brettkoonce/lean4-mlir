import LeanMlir.Proofs.Foundation.ResNet34BackB0

/-! # Backward-graph faithfulness for the VERIFIED ResNet-50 bottleneck block

The R50 peer of `ResNet34BackB0.lean`, and the step `planning/mnv4_verified.md` §8 calls
`<blk>BackBatchedGraph` + `<blk>BackBatchedGraph_faithful` — the two theorems that make the
render's backward *the certified one*. R50 shipped a trained number (89.86%, Imagenette) off a
certified renderer with no whole-net backward at all; this file is that gap.

## ⚠⚠ WHAT §8 GOT WRONG — "R50 is one step from done" was measured against the WRONG phase 1

§8 records R50's block-level VJP as ✓ (`Foundation/Resnet50BlocksCertified.lean`) and concludes the
job is only (2) + (3). That certificate is real, but it is for the **per-channel, non-batched**
forms — `bblkPC` / `bblkPProjPC` / `bblkPStridedPC` are built from `bnPerChannelTensor3` and plain
`flatConv`, with no `N`. The backward-graph vocabulary is **batched**: `bnBatchLA`, `batchMap`,
`convBackBatched`. Grepped before starting: there is **no batched R50 block VJP anywhere in the
repo**. So phase 1 had to be redone in the batched world here, exactly as §8 says MNv4 needs — R50
was *two* steps from done, not one.

⭐ It was still cheap, and for the reason §1 of the R50 file already gives: **every stage this
needs already exists.** `ResNet34BackB0` builds its own batched stages rather than lifting the PC
ones, and those stages are generic in `{ic oc h w kH kW}`, so R50 reuses all four **verbatim**:

| stage | what R50 uses it for | from |
|---|---|---|
| `cbReluB` | the 1×1 reduce AND the 3×3 (stride-1 blocks) | `ResNet34BackB0` |
| `cbReluStridedB` | the 3×3 in a downsample block | `ResNet34BackB0` |
| `projB` | the 1×1 expand (no activation) and the stride-1 skip | `EfficientNetBackB0` |
| `projStridedB` | the strided projection skip | `ResNet34BackB0` |

**Zero new stages, zero new SHlo ops, zero new VJP obligations.** The bottleneck's third conv is
one more `vjp_comp_at` link, and the whole file is composition — §6's "a family from one
constructor" landing on R50's backward the way §3i records it landing on MNv4's.

## The three forms, and why the third exists

| form | where in R50 | R34 analogue |
|---|---|---|
| `r50Bottleneck…` — identity | 12 blocks | `r34BasicBlock…` |
| `r50DownBlock…` — strided projection | stages 2/3/4, block 0 | `r34DownBlock…` |
| ⭐ `r50ProjBlock…` — **stride-1** projection | **stage 1 block 0 ONLY** | ⛔ **none** |

R34 never needed the third: its stage 1 is `ic = oc = 64`, so block 0 is an identity block. R50's
stage 1 goes `64 → 256` at stride 1 — the channels change so it needs a projection, the resolution
does not so that projection is not strided. ⚠ `r50DownBlock` cannot be substituted for it: its type
reads `Vec (N * (ic * (2*h) * (2*w))) → Vec (N * (oc * h * w))`, so the halving is in the
*signature* and the substitution is a shape error. Reaching for the identity form instead is the
dangerous one — an identity skip where a projection belongs is well-typed only if `ic = oc`.

## ⚠ THE STRIDE IS ON THE 3×3, NOT THE LEADING 1×1

`r50DownBody` puts `cbReluStridedB` on the **second** conv. That is ResNet **v1.5** / torchvision,
which is what `jax/MainResnet50Imagenet.lean` trains. The v1 placement (stride on the leading 1×1)
compiles, trains and descends — and is a different net (§3's trap, and `VerifiedSpec.lean:46`
records it costing ~0.5 pt of top-1). The leading 1×1 therefore runs at the INPUT resolution
`(2*h)×(2*w)` and carries `mid` channels there until `W₂` decimates.

## Relu, and why every statement here is `_at`

R50 is relu throughout, so — per §8's design note — the VJPs are pointwise and hypothesis-threaded
via `vjp_comp_at`, never the global form. A bottleneck has **three** kinks, not the basic block's
two: the two interior relus (`h_s1`, `h_s2`) and the outer post-residual relu (`h_out`). The
per-op backward token is `.selectPos`, whose faithfulness is the already-proven `selectPos_faithful`.

## Structure

* `r50BodyB` — the stride-1 bottleneck body `projB ∘ cbReluB ∘ cbReluB`, its `_at` VJP, and
  `r50BodyBackBatchedGraph` + `…_faithful`.
* `r50BottleneckBackBatchedGraph_faithful` — **CAPSTONE 1**, the identity block
  `relu ∘ residual(F)`.
* `r50ProjBlockBackBatchedGraph_faithful` — **CAPSTONE 2**, `relu ∘ residualProj(projB, F)`, the
  form with no R34 analogue.
* `r50DownBodyB` / `r50DownBlockBackBatchedGraph_faithful` — **CAPSTONE 3**, the strided
  projection block `relu ∘ residualProj(projStridedB, F_s)`.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The stride-1 bottleneck body: `projB ∘ cbReluB ∘ cbReluB`
-- ════════════════════════════════════════════════════════════════

/-- The batched R50 bottleneck body's VJP at a smooth point — `projB ∘ cbReluB ∘ cbReluB`, i.e.
    1×1-reduce-relu → 3×3-relu → 1×1-expand (no activation; the outer relu comes after the
    residual add). Two `vjp_comp_at` links threading the two interior relu smoothness families;
    `projB` (global, no activation) is lifted via `.toHasVJPAt`.

    The 3-conv peer of `r34BodyB_has_vjp_at`. ⚠ `h_s2` is stated at the SECOND stage's pre-relu
    activation, which lives at `cbReluB … x` — writing it at `x` typechecks nowhere, and that is
    the only place a reader can get this shape wrong. -/
noncomputable def r50BodyB_has_vjp_at (N : Nat) {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_s1 : ∀ k, bnBatchLA N mid h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConv W₂ b₂)
                (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0) :
    HasVJPAt (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
              cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x := by
  have h1_vjp : HasVJPAt (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    cbReluB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1
  have h1_diff : DifferentiableAt ℝ (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    cbReluB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1
  have h2_vjp : HasVJPAt (cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂)
      (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x) :=
    cbReluB_has_vjp_at N W₂ b₂ ε₂ hε₂ γ₂ β₂ _ h_s2
  have h2_diff : DifferentiableAt ℝ (cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂)
      (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x) :=
    cbReluB_differentiableAt N W₂ b₂ ε₂ hε₂ γ₂ β₂ _ h_s2
  have inner_vjp : HasVJPAt (cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
      cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    vjp_comp_at _ _ x h1_diff h2_diff h1_vjp h2_vjp
  have inner_diff : DifferentiableAt ℝ (cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
      cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x := h2_diff.comp x h1_diff
  exact vjp_comp_at _ (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃) x
    inner_diff
    ((projB_differentiable N (h := h) (w := w) W₃ b₃ ε₃ hε₃ γ₃ β₃) _)
    inner_vjp
    ((projB_has_vjp N (h := h) (w := w) W₃ b₃ ε₃ hε₃ γ₃ β₃).toHasVJPAt _)

theorem r50BodyB_differentiableAt (N : Nat) {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_s1 : ∀ k, bnBatchLA N mid h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConv W₂ b₂)
                (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0) :
    DifferentiableAt ℝ (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
              cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
  ((projB_differentiable N (h := h) (w := w) W₃ b₃ ε₃ hε₃ γ₃ β₃) _).comp x
    ((cbReluB_differentiableAt N W₂ b₂ ε₂ hε₂ γ₂ β₂ _ h_s2).comp x
      (cbReluB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1))

/-- The batched R50 bottleneck body backward graph: the three stage graphs chained at their
    cumulative forward activations (`cbReluB⁻¹ ∘ cbReluB⁻¹ ∘ projB⁻¹`). -/
noncomputable def r50BodyBackBatchedGraph {N ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (γ₃ β₃ : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  let x1 := cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x
  let x2 := cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ x1
  cbReluBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ x
    (cbReluBackBatchedGraph W₂ b₂ ε₂ γ₂ β₂ x1
      (projBackBatchedGraph W₃ b₃ ε₃ γ₃ β₃ x2 e))

theorem r50BodyBackBatchedGraph_faithful {N ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w)))
    (h_s1 : ∀ k, bnBatchLA N mid h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConv W₂ b₂)
                (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0) :
    den (r50BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x e)
      = (r50BodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2).backward (den e) := by
  rw [r50BodyBackBatchedGraph,
      cbReluBackBatchedGraph_faithful (hε := hε₁) (h_smooth := h_s1),
      cbReluBackBatchedGraph_faithful (hε := hε₂) (h_smooth := h_s2),
      projBackBatchedGraph_faithful (hε := hε₃)]
  simp only [r50BodyB_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § CAPSTONE 1 — the identity bottleneck `relu ∘ residual(F)`
-- ════════════════════════════════════════════════════════════════

/-- The batched R50 identity bottleneck's VJP at a smooth point — `relu ∘ residual(F)` with body
    `F = projB ∘ cbReluB ∘ cbReluB`. One `vjp_comp_at` composing the residual fan-in VJP with the
    OUTER relu's pointwise VJP at the pre-relu activation `residual(F)(x)`.

    Three smoothness families, where `r34BasicBlockB_has_vjp_at` needs two: the bottleneck's extra
    interior conv brings its own relu. -/
noncomputable def r50BottleneckB_has_vjp_at (N : Nat) {c mid h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid c kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 c mid kH₃ kW₃) (b₃ : Vec c) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_s1 : ∀ k, bnBatchLA N mid h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConv W₂ b₂)
                (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0)
    (h_out : ∀ k, residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                    cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                    cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    HasVJPAt (relu (N * (c * h * w)) ∘
              residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                        cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                        cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x := by
  have hbody_vjp := r50BodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2
  have hbody_diff := r50BodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2
  have hres_vjp := residual_has_vjp_at _ x hbody_diff hbody_vjp
  have hres_diff : DifferentiableAt ℝ (residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
        cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
        cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x :=
    hbody_diff.add (differentiable_id.differentiableAt)
  exact vjp_comp_at _ (relu (N * (c * h * w))) x
    hres_diff
    (relu_differentiableAt_of_smooth (N * (c * h * w)) _ h_out)
    hres_vjp
    (relu_has_vjp_at (N * (c * h * w)) _ h_out)

/-- The whole batched R50 identity bottleneck backward graph: `selectPos` (outer relu) ∘ residual
    fan-in (body backward + identity skip). Same shape as `r34BasicBlockBackBatchedGraph` — the
    outer relu is the LAST forward op, so its `.selectPos` is the OUTERMOST backward op, and the
    masked cotangent feeds BOTH the body chain and the verbatim `%dy` skip. -/
noncomputable def r50BottleneckBackBatchedGraph {N c mid h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid c kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 c mid kH₃ kW₃) (b₃ : Vec c) (ε₃ : ℝ) (γ₃ β₃ : Vec c)
    (x : Vec (N * (c * h * w))) (ecot : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  let preRelu := residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                  cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                  cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x
  let masked : SHlo (N * (c * h * w)) := .selectPos "%outR" preRelu ecot
  .addV
    (r50BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x masked)
    masked

/-- **CAPSTONE 1 — the whole batched R50 identity bottleneck: backward graph ↔ the proven VJP.**

    The three batched stage backward graphs chained at their forward activations, wrapped in the
    residual additive fan-in (body cotangent + identity skip) and the OUTER post-residual relu,
    proven equal to `r50BottleneckB_has_vjp_at`. The 3-conv peer of
    `r34BasicBlockBackBatchedGraph_faithful`, threaded through all three relu smoothness families.

    Key fact, unchanged from R34: the outer relu's `.selectPos` mask is applied ONCE to the
    incoming `dy`, and that masked cotangent is what the residual fan-in sees. -/
theorem r50BottleneckBackBatchedGraph_faithful
    {N c mid h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid c kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 c mid kH₃ kW₃) (b₃ : Vec c) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec c)
    (x : Vec (N * (c * h * w))) (ecot : SHlo (N * (c * h * w)))
    (h_s1 : ∀ k, bnBatchLA N mid h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConv W₂ b₂)
                (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0)
    (h_out : ∀ k, residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                    cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                    cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    den (r50BottleneckBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x ecot)
      = (r50BottleneckB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2 h_out).backward (den ecot) := by
  have hmask : den (SHlo.selectPos "%outR"
        (residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
          cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
          cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) ecot)
      = (relu_has_vjp_at (N * (c * h * w)) _ h_out).backward (den ecot) :=
    selectPos_faithful _ _ h_out ecot
  have hbody : den (r50BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x
        (SHlo.selectPos "%outR"
          (residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
            cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
            cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) ecot))
      = (r50BodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2).backward
          ((relu_has_vjp_at (N * (c * h * w)) _ h_out).backward (den ecot)) := by
    rw [r50BodyBackBatchedGraph_faithful (hε₁ := hε₁) (hε₂ := hε₂) (hε₃ := hε₃)
      (h_s1 := h_s1) (h_s2 := h_s2), hmask]
  funext i
  have hsum : den (r50BottleneckBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
        W₃ b₃ ε₃ γ₃ β₃ x ecot) i
      = den (r50BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x
            (SHlo.selectPos "%outR"
              (residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) ecot)) i
        + den (SHlo.selectPos "%outR"
              (residual (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) ecot) i := rfl
  rw [hsum, hbody, hmask]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § CAPSTONE 2 — the STRIDE-1 projection bottleneck (R50 stage 1 block 0)
-- ════════════════════════════════════════════════════════════════

/-- ⭐ The batched R50 **stride-1 projection** bottleneck's VJP at a smooth point —
    `relu ∘ residualProj(projB, F)` with body `F = projB ∘ cbReluB ∘ cbReluB` and a **stride-1**
    `bn∘conv` projection skip.

    ⚠ **This is the form with no R34 analogue**, and it exists in exactly one place in R50: stage 1
    block 0, where channels go `64 → 256` but the resolution does not change. R34's stage 1 is
    `ic = oc = 64`, so its block 0 is an identity block and this shape never arises.

    Structurally it is CAPSTONE 1 with `residual` (identity skip) replaced by `residualProj` (both
    paths nontrivial), and CAPSTONE 3 with every stride-2 op replaced by its stride-1 peer. -/
noncomputable def r50ProjBlockB_has_vjp_at (N : Nat)
    {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_s1 : ∀ k, bnBatchLA N mid h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConv W₂ b₂)
                (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0)
    (h_out : ∀ k, residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
                    (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                     cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                     cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    HasVJPAt (relu (N * (oc * h * w)) ∘
              residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
                (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                 cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                 cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x := by
  have hbody_vjp := r50BodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2
  have hbody_diff := r50BodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2
  have hproj_vjp : HasVJPAt (projB N (h := h) (w := w) Wp bp εp γp βp) x :=
    (projB_has_vjp N Wp bp εp hεp γp βp).toHasVJPAt x
  have hproj_diff : DifferentiableAt ℝ (projB N (h := h) (w := w) Wp bp εp γp βp) x :=
    (projB_differentiable N Wp bp εp hεp γp βp) x
  have hres_vjp := residualProj_has_vjp_at _ _ x hproj_diff hbody_diff hproj_vjp hbody_vjp
  have hres_diff : DifferentiableAt ℝ (residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
        (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
         cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
         cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x := hproj_diff.add hbody_diff
  exact vjp_comp_at _ (relu (N * (oc * h * w))) x
    hres_diff
    (relu_differentiableAt_of_smooth (N * (oc * h * w)) _ h_out)
    hres_vjp
    (relu_has_vjp_at (N * (oc * h * w)) _ h_out)

/-- The whole batched R50 stride-1 projection bottleneck backward graph: `selectPos` (outer relu) ∘
    projected-residual fan-in (**stride-1** projection-skip backward + body backward). Unlike the
    identity block, both `addV` operands are nontrivial backward subgraphs; unlike CAPSTONE 3, the
    skip is `projBackBatchedGraph`, not its strided sibling. -/
noncomputable def r50ProjBlockBackBatchedGraph
    {N ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w))) (ecot : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  let preRelu := residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
                  (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                   cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                   cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x
  let masked : SHlo (N * (oc * h * w)) := .selectPos "%outR" preRelu ecot
  .addV
    (projBackBatchedGraph Wp bp εp γp βp x masked)
    (r50BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x masked)

/-- **CAPSTONE 2 — the whole batched R50 STRIDE-1 PROJECTION bottleneck: backward graph ↔ the
    proven VJP.** The R50-only block form (stage 1 block 0), with no R34 analogue to mirror. -/
theorem r50ProjBlockBackBatchedGraph_faithful
    {N ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w))) (ecot : SHlo (N * (oc * h * w)))
    (h_s1 : ∀ k, bnBatchLA N mid h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConv W₂ b₂)
                (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0)
    (h_out : ∀ k, residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
                    (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                     cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                     cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    den (r50ProjBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
          Wp bp εp γp βp x ecot)
      = (r50ProjBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp x h_s1 h_s2 h_out).backward (den ecot) := by
  have hmask : den (SHlo.selectPos "%outR"
        (residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
          (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
           cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
           cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) ecot)
      = (relu_has_vjp_at (N * (oc * h * w)) _ h_out).backward (den ecot) :=
    selectPos_faithful _ _ h_out ecot
  have hbody : den (r50BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x
        (SHlo.selectPos "%outR"
          (residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
            (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
             cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
             cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) ecot))
      = (r50BodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2).backward
          ((relu_has_vjp_at (N * (oc * h * w)) _ h_out).backward (den ecot)) := by
    rw [r50BodyBackBatchedGraph_faithful (hε₁ := hε₁) (hε₂ := hε₂) (hε₃ := hε₃)
      (h_s1 := h_s1) (h_s2 := h_s2), hmask]
  have hproj : den (projBackBatchedGraph Wp bp εp γp βp x
        (SHlo.selectPos "%outR"
          (residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
            (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
             cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
             cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) ecot))
      = (projB_has_vjp N Wp bp εp hεp γp βp).backward x
          ((relu_has_vjp_at (N * (oc * h * w)) _ h_out).backward (den ecot)) := by
    rw [projBackBatchedGraph_faithful (hε := hεp), hmask]
  funext i
  have hsum : den (r50ProjBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
        W₃ b₃ ε₃ γ₃ β₃ Wp bp εp γp βp x ecot) i
      = den (projBackBatchedGraph Wp bp εp γp βp x
            (SHlo.selectPos "%outR"
              (residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
                (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                 cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                 cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) ecot)) i
        + den (r50BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x
              (SHlo.selectPos "%outR"
                (residualProj (projB N (h := h) (w := w) Wp bp εp γp βp)
                  (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                   cbReluB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                   cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) ecot)) i := rfl
  rw [hsum, hbody, hproj]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The DOWNSAMPLE body: `projB ∘ cbReluStridedB ∘ cbReluB`
-- ════════════════════════════════════════════════════════════════

/-- The batched R50 downsample bottleneck body's VJP at a smooth point.

    ⚠⚠ **The stride is on the 3×3 (`W₂`), not the leading 1×1** — ResNet v1.5 / torchvision, which
    is what the reference trains. Read it in the types: `W₁`'s stage is `cbReluB` at the INPUT
    resolution `(2*h)×(2*w)` (a stride-1 1×1 that changes channels only), and `W₂`'s is
    `cbReluStridedB`, which is where `(2*h, 2*w) ↦ (h, w)` happens. Putting the stride on `W₁`
    instead compiles, trains and descends — and is ResNet v1, a different net.

    `h_s1` therefore lives at `(2*h, 2*w)` and `h_s2` at `(h, w)`; that asymmetry is the load-bearing
    detail of this whole section. -/
noncomputable def r50DownBodyB_has_vjp_at (N : Nat)
    {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_s1 : ∀ k, bnBatchLA N mid (2 * h) (2 * w) ε₁ γ₁ β₁
              (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConvStride2 W₂ b₂)
                (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0) :
    HasVJPAt (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
              cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x := by
  have h1_vjp : HasVJPAt (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x :=
    cbReluB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1
  have h1_diff : DifferentiableAt ℝ (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x :=
    cbReluB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1
  have h2_vjp : HasVJPAt (cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂)
      (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x) :=
    cbReluStridedB_has_vjp_at N W₂ b₂ ε₂ hε₂ γ₂ β₂ _ h_s2
  have h2_diff : DifferentiableAt ℝ (cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂)
      (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x) :=
    cbReluStridedB_differentiableAt N W₂ b₂ ε₂ hε₂ γ₂ β₂ _ h_s2
  have inner_vjp : HasVJPAt (cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
      cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x :=
    vjp_comp_at _ _ x h1_diff h2_diff h1_vjp h2_vjp
  have inner_diff : DifferentiableAt ℝ (cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
      cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x := h2_diff.comp x h1_diff
  exact vjp_comp_at _ (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃) x
    inner_diff
    ((projB_differentiable N (h := h) (w := w) W₃ b₃ ε₃ hε₃ γ₃ β₃) _)
    inner_vjp
    ((projB_has_vjp N (h := h) (w := w) W₃ b₃ ε₃ hε₃ γ₃ β₃).toHasVJPAt _)

theorem r50DownBodyB_differentiableAt (N : Nat)
    {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_s1 : ∀ k, bnBatchLA N mid (2 * h) (2 * w) ε₁ γ₁ β₁
              (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConvStride2 W₂ b₂)
                (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0) :
    DifferentiableAt ℝ (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
              cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x :=
  ((projB_differentiable N (h := h) (w := w) W₃ b₃ ε₃ hε₃ γ₃ β₃) _).comp x
    ((cbReluStridedB_differentiableAt N W₂ b₂ ε₂ hε₂ γ₂ β₂ _ h_s2).comp x
      (cbReluB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1))

/-- The batched R50 downsample body backward graph: the three stage graphs chained at their
    cumulative forward activations. `convStridedBackBatched` appears at the **3×3**, matching the
    forward's stride placement. -/
noncomputable def r50DownBodyBackBatchedGraph
    {N ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (γ₃ β₃ : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  let x1 := cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x
  let x2 := cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ x1
  cbReluBackBatchedGraph (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x
    (cbReluStridedBackBatchedGraph W₂ b₂ ε₂ γ₂ β₂ x1
      (projBackBatchedGraph W₃ b₃ ε₃ γ₃ β₃ x2 e))

theorem r50DownBodyBackBatchedGraph_faithful
    {N ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w)))
    (h_s1 : ∀ k, bnBatchLA N mid (2 * h) (2 * w) ε₁ γ₁ β₁
              (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConvStride2 W₂ b₂)
                (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0) :
    den (r50DownBodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x e)
      = (r50DownBodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2).backward (den e) := by
  rw [r50DownBodyBackBatchedGraph,
      cbReluBackBatchedGraph_faithful (hε := hε₁) (h_smooth := h_s1),
      cbReluStridedBackBatchedGraph_faithful (hε := hε₂) (h_smooth := h_s2),
      projBackBatchedGraph_faithful (hε := hε₃)]
  simp only [r50DownBodyB_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § CAPSTONE 3 — the strided projection bottleneck (stages 2/3/4, block 0)
-- ════════════════════════════════════════════════════════════════

/-- The batched R50 **strided projection** bottleneck's VJP at a smooth point —
    `relu ∘ residualProj(projStridedB, F_s)`. The R50 peer of `r34DownBlockB_has_vjp_at`, with the
    bottleneck's third conv and its extra relu family. -/
noncomputable def r50DownBlockB_has_vjp_at (N : Nat)
    {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_s1 : ∀ k, bnBatchLA N mid (2 * h) (2 * w) ε₁ γ₁ β₁
              (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConvStride2 W₂ b₂)
                (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0)
    (h_out : ∀ k, residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                    (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                     cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                     cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    HasVJPAt (relu (N * (oc * h * w)) ∘
              residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                 cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                 cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁)) x := by
  have hbody_vjp := r50DownBodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2
  have hbody_diff := r50DownBodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2
  have hproj_vjp : HasVJPAt (projStridedB N (h := h) (w := w) Wp bp εp γp βp) x :=
    (projStridedB_has_vjp N Wp bp εp hεp γp βp).toHasVJPAt x
  have hproj_diff : DifferentiableAt ℝ (projStridedB N (h := h) (w := w) Wp bp εp γp βp) x :=
    (projStridedB_differentiable N Wp bp εp hεp γp βp) x
  have hres_vjp := residualProj_has_vjp_at _ _ x hproj_diff hbody_diff hproj_vjp hbody_vjp
  have hres_diff : DifferentiableAt ℝ
      (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
        (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
         cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
         cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁)) x := hproj_diff.add hbody_diff
  exact vjp_comp_at _ (relu (N * (oc * h * w))) x
    hres_diff
    (relu_differentiableAt_of_smooth (N * (oc * h * w)) _ h_out)
    hres_vjp
    (relu_has_vjp_at (N * (oc * h * w)) _ h_out)

/-- The whole batched R50 strided projection bottleneck backward graph: `selectPos` (outer relu) ∘
    projected-residual fan-in (strided projection-skip backward + body backward). -/
noncomputable def r50DownBlockBackBatchedGraph
    {N ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (ecot : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  let preRelu := residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                  (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                   cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                   cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x
  let masked : SHlo (N * (oc * h * w)) := .selectPos "%outR" preRelu ecot
  .addV
    (projStridedBackBatchedGraph Wp bp εp γp βp x masked)
    (r50DownBodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x masked)

/-- **CAPSTONE 3 — the whole batched R50 STRIDED PROJECTION bottleneck: backward graph ↔ the proven
    VJP.** With CAPSTONES 1 and 2 this closes every block form in ResNet-50: 12 identity blocks,
    3 strided projections and the one stride-1 projection. -/
theorem r50DownBlockBackBatchedGraph_faithful
    {N ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (ecot : SHlo (N * (oc * h * w)))
    (h_s1 : ∀ k, bnBatchLA N mid (2 * h) (2 * w) ε₁ γ₁ β₁
              (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_s2 : ∀ k, bnBatchLA N mid h w ε₂ γ₂ β₂
              (batchMap N (flatConvStride2 W₂ b₂)
                (cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁ x)) k ≠ 0)
    (h_out : ∀ k, residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                    (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                     cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                     cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    den (r50DownBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
          Wp bp εp γp βp x ecot)
      = (r50DownBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp x h_s1 h_s2 h_out).backward (den ecot) := by
  have hmask : den (SHlo.selectPos "%outR"
        (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
          (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
           cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
           cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x) ecot)
      = (relu_has_vjp_at (N * (oc * h * w)) _ h_out).backward (den ecot) :=
    selectPos_faithful _ _ h_out ecot
  have hbody : den (r50DownBodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x
        (SHlo.selectPos "%outR"
          (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
            (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
             cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
             cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x) ecot))
      = (r50DownBodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2).backward
          ((relu_has_vjp_at (N * (oc * h * w)) _ h_out).backward (den ecot)) := by
    rw [r50DownBodyBackBatchedGraph_faithful (hε₁ := hε₁) (hε₂ := hε₂) (hε₃ := hε₃)
      (h_s1 := h_s1) (h_s2 := h_s2), hmask]
  have hproj : den (projStridedBackBatchedGraph Wp bp εp γp βp x
        (SHlo.selectPos "%outR"
          (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
            (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
             cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
             cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x) ecot))
      = (projStridedB_has_vjp N Wp bp εp hεp γp βp).backward x
          ((relu_has_vjp_at (N * (oc * h * w)) _ h_out).backward (den ecot)) := by
    rw [projStridedBackBatchedGraph_faithful (hε := hεp), hmask]
  funext i
  have hsum : den (r50DownBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂
        W₃ b₃ ε₃ γ₃ β₃ Wp bp εp γp βp x ecot) i
      = den (projStridedBackBatchedGraph Wp bp εp γp βp x
            (SHlo.selectPos "%outR"
              (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                 cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                 cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x) ecot)) i
        + den (r50DownBodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ x
              (SHlo.selectPos "%outR"
                (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                  (projB N (h := h) (w := w) W₃ b₃ ε₃ γ₃ β₃ ∘
                   cbReluStridedB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                   cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁) x)
                ecot)) i := rfl
  rw [hsum, hbody, hproj]
  rfl

end Proofs.StableHLO
