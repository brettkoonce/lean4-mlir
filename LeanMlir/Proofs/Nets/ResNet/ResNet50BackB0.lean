import LeanMlir.Proofs.Nets.ResNet.ResNet34BackB0

/-! # Backward-graph faithfulness for the ResNet-50 bottleneck block (batched)

Each bottleneck form's backward graph (`<blk>BackBatchedGraph`) and the theorem that it denotes the
certified VJP (`<blk>BackBatchedGraph_faithful`) — the ResNet-50 peer of `ResNet34BackB0`, at the
batched index (`bnBatchLA`, `batchMap`, `convBackBatched`). Every stage is one other nets already
use, generic in `{ic oc h w kH kW}`, so the file is composition — no new stage, `SHlo` op or VJP:

| stage | what R50 uses it for | from |
|---|---|---|
| `cbReluB` | the 1×1 reduce and the 3×3 (stride-1 blocks) | `Foundation/BatchedStageLayers` |
| `cbReluStridedB` | the 3×3 of a downsample block | `Foundation/BatchedStageLayers` |
| `projB` | the 1×1 expand (no activation) and the stride-1 skip | `Foundation/BatchedStages` |
| `projStridedB` | the strided projection skip | `Foundation/BatchedStageLayers` |

## The three forms

| form | where in R50 | R34 analogue |
|---|---|---|
| `r50Bottleneck…` — identity | 12 blocks | `r34BasicBlock…` |
| `r50DownBlock…` — strided projection | stages 2/3/4, block 0 | `r34DownBlock…` |
| `r50ProjBlock…` — **stride-1** projection | stage 1 block 0 only | none |

R34's stage 1 is `ic = oc = 64`, so its block 0 is an identity block. R50's stage 1 goes
`64 → 256` at stride 1: the channels change, so it needs a projection, and the resolution does not,
so the projection is not strided. `r50DownBlock` cannot stand in for it — its type
`Vec (N * (ic * (2*h) * (2*w))) → Vec (N * (oc * h * w))` puts the halving in the signature.

## The stride is on the 3×3, not the leading 1×1

`r50DownBody` puts `cbReluStridedB` on the **second** conv: ResNet **v1.5** / torchvision, what
[`jax/MainResnet50Imagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainResnet50Imagenet.lean) trains. The v1 placement (stride on the leading 1×1)
compiles, trains and descends, and is a different net (`VerifiedSpec` records it costing ~0.5 pt
of top-1). The leading 1×1 therefore runs at the input resolution `(2*h)×(2*w)`.

## Relu, and why every statement is `_at`

R50 is relu throughout, so the VJPs are pointwise and hypothesis-threaded via `vjp_comp_at`. A
bottleneck has three kinks: the two interior relus (`h_s1`, `h_s2`) and the post-residual relu
(`h_out`). The backward token is `.selectPos`, faithful by `selectPos_faithful`.

## Contents

* `r50BodyBackBatchedGraph` — the stride-1 body `projB ∘ cbReluB ∘ cbReluB`'s backward graph;
  `r50DownBodyBackBatchedGraph` is its strided peer.
* `r50BottleneckBackBatchedGraph_faithful` — the identity block `relu ∘ residual(F)`.
* `r50ProjBlockBackBatchedGraph_faithful` — `relu ∘ residualProj(projB, F)`.
* `r50DownBlockBackBatchedGraph_faithful` — `relu ∘ residualProj(projStridedB, F_s)`.

Each block is a `CertLayer` (`r50BottleneckLayer`, `r50ProjBlockLayer`, `r50DownBlockLayer`)
composed from the stage layers, and each theorem is that layer's `faithful`.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The stride-1 bottleneck body: `projB ∘ cbReluB ∘ cbReluB`
-- ════════════════════════════════════════════════════════════════

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

-- ════════════════════════════════════════════════════════════════
-- § CAPSTONE 1 — the identity bottleneck `relu ∘ residual(F)`
-- ════════════════════════════════════════════════════════════════

/-- The identity bottleneck as a `CertLayer`: `residual (cbReluLayer ; cbReluLayer ; projLayer)`,
    then `reluOut`. ⭐ **An endomorphism** (`ic = oc`, resolution unchanged), which is what lets
    `CertLayer.chain` iterate it — a stage tail is n of these. -/
noncomputable def r50BottleneckLayer (N : Nat) {c mid h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid c kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 c mid kH₃ kW₃) (b₃ : Vec c) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  (CertLayer.residual (((cbReluLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁).comp
    (cbReluLayer N W₂ b₂ ε₂ hε₂ γ₂ β₂)).comp (projLayer N W₃ b₃ ε₃ hε₃ γ₃ β₃))).comp
    (CertLayer.reluOut _)

/-- The batched R50 identity bottleneck's VJP at a smooth point — `relu ∘ residual(F)` with body
    `F = projB ∘ cbReluB ∘ cbReluB`: the residual fan-in VJP, then the OUTER relu's pointwise VJP
    at the pre-relu activation `residual(F)(x)` (`r50BottleneckLayer`'s VJP). ⚠ `h_s2` is stated
    at the SECOND stage's pre-relu activation, which lives at `cbReluB … x` — writing it at `x`
    typechecks nowhere.

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
                        cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x :=
  (r50BottleneckLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃).vjp x ⟨⟨⟨h_s1, h_s2⟩, trivial⟩, h_out⟩

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
    incoming `dy`, and that masked cotangent is what the residual fan-in sees. It is
    `r50BottleneckLayer`'s `faithful`. -/
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
          W₃ b₃ ε₃ hε₃ γ₃ β₃ x h_s1 h_s2 h_out).backward (den ecot) :=
  (r50BottleneckLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃).faithful x ⟨⟨⟨h_s1, h_s2⟩, trivial⟩, h_out⟩ ecot

-- ════════════════════════════════════════════════════════════════
-- § CAPSTONE 2 — the STRIDE-1 projection bottleneck (R50 stage 1 block 0)
-- ════════════════════════════════════════════════════════════════

/-- ⭐ The **stride-1 projection** bottleneck as a `CertLayer` — R50 stage 1 block 0, the form with
    no R34 analogue: `residualProj (projLayer) (cbReluLayer ; cbReluLayer ; projLayer)`, then
    `reluOut`. Changes channels, keeps resolution, so it is NOT an endomorphism and composes via
    `comp` rather than `chain`. -/
noncomputable def r50ProjBlockLayer (N : Nat)
    {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) :=
  (CertLayer.residualProj (projLayer N (h := h) (w := w) Wp bp εp hεp γp βp)
    (((cbReluLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁).comp
      (cbReluLayer N W₂ b₂ ε₂ hε₂ γ₂ β₂)).comp (projLayer N W₃ b₃ ε₃ hε₃ γ₃ β₃))).comp
    (CertLayer.reluOut _)

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
                 cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x :=
  (r50ProjBlockLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp).vjp x ⟨⟨trivial, ⟨h_s1, h_s2⟩, trivial⟩, h_out⟩

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
    proven VJP.** The R50-only block form (stage 1 block 0), with no R34 analogue to mirror. It is
    `r50ProjBlockLayer`'s `faithful`. -/
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
          W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp x h_s1 h_s2 h_out).backward (den ecot) :=
  (r50ProjBlockLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp).faithful x ⟨⟨trivial, ⟨h_s1, h_s2⟩, trivial⟩, h_out⟩ ecot

-- ════════════════════════════════════════════════════════════════
-- § The DOWNSAMPLE body: `projB ∘ cbReluStridedB ∘ cbReluB`
-- ════════════════════════════════════════════════════════════════

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

-- ════════════════════════════════════════════════════════════════
-- § CAPSTONE 3 — the strided projection bottleneck (stages 2/3/4, block 0)
-- ════════════════════════════════════════════════════════════════

/-- The **strided projection** bottleneck as a `CertLayer` — stages 2/3/4, block 0:
    `residualProj (projStridedLayer) (cbReluLayer ; cbReluStridedLayer ; projLayer)`, then
    `reluOut`. Halves the resolution, which is why its input type carries `2*h`/`2*w`. ⚠ The stride
    is on the 3×3 (`cbReluStridedLayer` at `W₂`), so `h_s1` is stated at the input resolution and
    `h_s2` at the output one. -/
noncomputable def r50DownBlockLayer (N : Nat)
    {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (hε₃ : 0 < ε₃) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  (CertLayer.residualProj (projStridedLayer N (h := h) (w := w) Wp bp εp hεp γp βp)
    (((cbReluLayer N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ hε₁ γ₁ β₁).comp
      (cbReluStridedLayer N (h := h) (w := w) W₂ b₂ ε₂ hε₂ γ₂ β₂)).comp
      (projLayer N W₃ b₃ ε₃ hε₃ γ₃ β₃))).comp (CertLayer.reluOut _)

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
                 cbReluB N (h := 2 * h) (w := 2 * w) W₁ b₁ ε₁ γ₁ β₁)) x :=
  (r50DownBlockLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp).vjp x ⟨⟨trivial, ⟨h_s1, h_s2⟩, trivial⟩, h_out⟩

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
    3 strided projections and the one stride-1 projection. It is `r50DownBlockLayer`'s
    `faithful`. -/
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
          W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp x h_s1 h_s2 h_out).backward (den ecot) :=
  (r50DownBlockLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    W₃ b₃ ε₃ hε₃ γ₃ β₃ Wp bp εp hεp γp βp).faithful x ⟨⟨trivial, ⟨h_s1, h_s2⟩, trivial⟩, h_out⟩ ecot

end Proofs.StableHLO
