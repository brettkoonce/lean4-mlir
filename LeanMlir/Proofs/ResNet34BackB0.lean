import LeanMlir.Proofs.MobileNetV2BackB0

/-! # Backward-graph faithfulness for the VERIFIED ResNet-34 basic block

The ResNet-34 peer of `EfficientNetBackB0.lean` / `MobileNetV2BackB0.lean`: a
*backward* StableHLO graph that denotes the proven VJP of the batched ResNet-34
basic block.

The basic block is `relu ∘ residual(F)` (identity block), with body
`F = (conv-bn) ∘ (conv-bn-relu)` — a 3×3 conv → bn → relu, then a 3×3 conv → bn
(no activation), and an identity skip, followed by an **outer relu after the
residual add**. This last fact is the structural difference from the
MobileNetV2/EfficientNet residual blocks (whose residual add is the block
output): r34 wraps the residual add in one more relu.

## The relu wrinkle (vs relu6 / swish)

r34 uses **relu** (one kink, at 0): its VJP is only the *pointwise*
`relu_has_vjp_at`, conditioned on the smoothness hypothesis `∀ k, x k ≠ 0` at the
pre-activation — simpler than relu6's two-sided `x k ≠ 0 ∧ x k ≠ 6`, but the same
`_at` machinery (`vjp_comp_at` + `HasVJP.toHasVJPAt`). Its per-op backward token
is `.selectPos` (the mask `if x>0 then dy else 0`), whose denotation faithfulness
is the already-proven (`rfl`) `selectPos_faithful` (`StableHLO.lean:782`).

Because there are TWO relu kinks (the body's mid-relu AND the outer post-residual
relu), the whole-block VJP and its backward-graph faithfulness are `_at` /
hypothesis-threaded: one smoothness family at the body's mid-relu pre-activation,
one at the outer relu's pre-activation `residual(F)(x)`.

## Structure

* `cbReluB` — batched conv → bn → **relu** stage (`cbrB` from MobileNetV2BackB0
  with relu for relu6), `_at` VJP + backward-graph faithfulness
  (`cbReluBackBatchedGraph` + `…_faithful`), chaining `selectPos_faithful`
  + `bnBatchLABack_faithful` + `convBackBatched_faithful`.
* `cbB` (= `projB`, conv → bn, no activation) backward is reused VERBATIM from
  EfficientNetBackB0 (`projBackBatchedGraph` / `projBackBatchedGraph_faithful`).
* `r34BodyB_has_vjp_at` — the body `cbB ∘ cbReluB`, composed via `vjp_comp_at`
  over the mid-relu smoothness family, with its backward graph
  `r34BodyBackBatchedGraph` + `…_faithful`.
* `r34BasicBlockBackBatchedGraph_faithful` — the **CAPSTONE**: the whole batched
  ResNet-34 identity basic block backward graph (outer-relu `selectPos` ∘
  residual-fan-in(body-back) + identity skip) denotes the proven
  `relu ∘ residual(F)` VJP (`vjp_comp_at(residual_has_vjp_at(body), relu)`),
  threaded through both relu smoothness hypotheses.

## The strided/downsample block (`relu ∘ residualProj(proj, F_s)`)

The downsample-block capstone (`r34DownBlockBackBatchedGraph_faithful`) reuses the
new **strided** batched-conv backward primitive `convStridedBackBatched`
(`StableHLO.lean`, the stride-2 analog of `convBackBatched`; its `_faithful` lives
in `EfficientNetBackB0`). The body `F_s = projB ∘ cbReluStridedB` has a stride-2
conv1 (`cbReluStridedB`, the strided sibling of `cbReluB`) and a stride-1 conv2
(`projB`); the projection skip `projStridedB` is a stride-2 conv-bn. The whole
block composes `vjp_comp_at(residualProj_has_vjp_at(proj, F_s), relu)` exactly like
the identity block, but with the *projection* skip (`residualProj`, both paths
nontrivial) instead of the identity skip (`residual`), and the strided convs in the
body+skip.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Batched relu stage (`cbrB` with `relu` for `relu6`)
-- ════════════════════════════════════════════════════════════════

/-- Batched **conv → bn → relu** stage (ResNet basic-block first stage), at the
    network layout `N·(oc·h·w)`. The relu analogue of MobileNetV2's `cbrB`
    (relu for relu6). -/
@[reducible] noncomputable def cbReluB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N (flatConv W b)

-- ════════════════════════════════════════════════════════════════
-- § Stage `_at`-VJP + differentiability (relu smoothness threaded)
-- ════════════════════════════════════════════════════════════════

/-- **Generic relu-on-batched-bn-stage `_at` VJP.** The relu analogue of
    `bnRelu6Stage_has_vjp_at` (and of `bnSwishStage_has_vjp`, but `_at` — relu
    only has a pointwise VJP): compose the batched-op VJP, the true-BN VJP (both
    global, lifted via `.toHasVJPAt`), and relu's pointwise VJP at the pre-relu
    activation. The smoothness hypothesis is the one-sided `≠ 0`. -/
noncomputable def bnReluStage_has_vjp_at (N : Nat) {a oc h w : Nat}
    (op : Vec a → Vec (oc * h * w)) (hop : Differentiable ℝ op) (hopv : HasVJP op)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * a))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 0) :
    HasVJPAt (relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x := by
  have hbatch_diff : Differentiable ℝ (batchMap N op) := batchMap_differentiable op hop
  have hbn_diff : Differentiable ℝ (bnBatchLA N oc h w ε γ β) :=
    bnBatchLA_differentiable N oc h w ε hε γ β
  have inner_diff : DifferentiableAt ℝ (bnBatchLA N oc h w ε γ β ∘ batchMap N op) x :=
    (hbn_diff.comp hbatch_diff) x
  have inner_vjp : HasVJPAt (bnBatchLA N oc h w ε γ β ∘ batchMap N op) x :=
    vjp_comp_at (batchMap N op) (bnBatchLA N oc h w ε γ β) x
      (hbatch_diff x) (hbn_diff _)
      ((batchMap_has_vjp op hopv hop).toHasVJPAt x)
      ((bnBatchLA_has_vjp N oc h w ε hε γ β).toHasVJPAt _)
  exact vjp_comp_at (bnBatchLA N oc h w ε γ β ∘ batchMap N op)
    (relu (N * (oc * h * w))) x inner_diff
    (relu_differentiableAt_of_smooth (N * (oc * h * w)) _ h_smooth)
    inner_vjp
    (relu_has_vjp_at (N * (oc * h * w)) _ h_smooth)

/-- Differentiability of the generic relu-on-batched-bn-stage at a smooth point. -/
theorem bnReluStage_differentiableAt (N : Nat) {a oc h w : Nat}
    (op : Vec a → Vec (oc * h * w)) (hop : Differentiable ℝ op)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * a))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 0) :
    DifferentiableAt ℝ (relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x := by
  have inner : DifferentiableAt ℝ (bnBatchLA N oc h w ε γ β ∘ batchMap N op) x :=
    ((bnBatchLA_differentiable N oc h w ε hε γ β).comp (batchMap_differentiable op hop)) x
  exact (relu_differentiableAt_of_smooth (N * (oc * h * w)) _ h_smooth).comp x inner

/-- `cbReluB` (conv-bn-relu) `_at` VJP at a smooth point. -/
noncomputable def cbReluB_has_vjp_at (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0) :
    HasVJPAt (cbReluB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_has_vjp_at N (flatConv W b) (flatConv_differentiable W b)
    (flatConv_has_vjp W b) ε hε γ β x h_smooth

theorem cbReluB_differentiableAt (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0) :
    DifferentiableAt ℝ (cbReluB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_differentiableAt N (flatConv W b) (flatConv_differentiable W b) ε hε γ β x h_smooth

-- ════════════════════════════════════════════════════════════════
-- § Batched relu-stage backward graph (selectPos for the relu kink)
-- ════════════════════════════════════════════════════════════════

/-- Batched **conv → bn → relu** stage backward graph (ResNet basic-block stage 1):
    `convBackBatched ∘ bnBatchLABack ∘ selectPos`, each at its cumulative forward
    activation. The relu analogue of MobileNetV2's `cbrBackBatchedGraph` —
    `.selectPos` (the relu one-sided-kink mask) replaces `.selectMid`. -/
noncomputable def cbReluBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  .convBackBatched (N := N) "%cbrW" W b
    (.bnBatchLABack "%cbrG" "%cbrX" "cbrE" ε γ (batchMap N (flatConv W b) x)
      (.selectPos "%cbrR" (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x)) e))

theorem cbReluBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0) :
    den (cbReluBackBatchedGraph W b ε γ β x e)
      = (cbReluB_has_vjp_at N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [cbReluBackBatchedGraph, convBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectPos_faithful _ _ h_smooth]
  simp only [cbReluB_has_vjp_at, bnReluStage_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt,
    Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The body: `cbB ∘ cbReluB`  (= projB ∘ cbReluB)
-- ════════════════════════════════════════════════════════════════

/-- The batched ResNet-34 basic-block body's VJP at a smooth point —
    `projB ∘ cbReluB` (conv-bn after conv-bn-relu). One `vjp_comp_at` threading
    the mid-relu smoothness family; `projB` (global, no activation) is lifted via
    `.toHasVJPAt`.

    `h_s1` is the stage-1 relu smoothness (at the cbReluB pre-relu activation). -/
noncomputable def r34BodyB_has_vjp_at (N : Nat) {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_s1 : ∀ k, bnBatchLA N c h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0) :
    HasVJPAt (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x := by
  have h1_vjp : HasVJPAt (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    cbReluB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1
  have h1_diff : DifferentiableAt ℝ (cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    cbReluB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1
  exact vjp_comp_at _ (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂) x
    h1_diff
    ((projB_differentiable N (h := h) (w := w) W₂ b₂ ε₂ hε₂ γ₂ β₂) _)
    h1_vjp
    ((projB_has_vjp N (h := h) (w := w) W₂ b₂ ε₂ hε₂ γ₂ β₂).toHasVJPAt _)

theorem r34BodyB_differentiableAt (N : Nat) {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_s1 : ∀ k, bnBatchLA N c h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0) :
    DifferentiableAt ℝ (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
  ((projB_differentiable N (h := h) (w := w) W₂ b₂ ε₂ hε₂ γ₂ β₂) _).comp x
    (cbReluB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1)

/-- The batched ResNet-34 body backward graph: the two stage graphs chained at
    their cumulative forward activations (`cbReluB⁻¹ ∘ projB⁻¹`). -/
noncomputable def r34BodyBackBatchedGraph {N c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  let x1 := cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x
  cbReluBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ x
    (projBackBatchedGraph W₂ b₂ ε₂ γ₂ β₂ x1 e)

theorem r34BodyBackBatchedGraph_faithful {N c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w)))
    (h_s1 : ∀ k, bnBatchLA N c h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0) :
    den (r34BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x e)
      = (r34BodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1).backward (den e) := by
  rw [r34BodyBackBatchedGraph, cbReluBackBatchedGraph_faithful (hε := hε₁) (h_smooth := h_s1),
      projBackBatchedGraph_faithful (hε := hε₂)]
  simp only [r34BodyB_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The whole-block VJP: `relu ∘ residual(F)` (outer relu after the add)
-- ════════════════════════════════════════════════════════════════

/-- The batched ResNet-34 identity basic block's VJP at a smooth point —
    `relu ∘ residual(F)` with body `F = projB ∘ cbReluB`. One `vjp_comp_at`
    composing the residual fan-in VJP (`residual_has_vjp_at` of the body) with the
    OUTER relu's pointwise VJP at the pre-relu activation `residual(F)(x)`.

    `h_s1` is the body's mid-relu smoothness; `h_out` is the outer-relu smoothness
    (at `residual(F)(x)`). -/
noncomputable def r34BasicBlockB_has_vjp_at (N : Nat) {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_s1 : ∀ k, bnBatchLA N c h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_out : ∀ k, residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                    cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    HasVJPAt (relu (N * (c * h * w)) ∘
              residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                        cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x := by
  have hbody_vjp : HasVJPAt (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
        cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    r34BodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1
  have hbody_diff : DifferentiableAt ℝ (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
        cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    r34BodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1
  have hres_vjp : HasVJPAt (residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
        cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x :=
    residual_has_vjp_at _ x hbody_diff hbody_vjp
  have hres_diff : DifferentiableAt ℝ (residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
        cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x :=
    hbody_diff.add (differentiable_id.differentiableAt)
  exact vjp_comp_at _ (relu (N * (c * h * w))) x
    hres_diff
    (relu_differentiableAt_of_smooth (N * (c * h * w)) _ h_out)
    hres_vjp
    (relu_has_vjp_at (N * (c * h * w)) _ h_out)

-- ════════════════════════════════════════════════════════════════
-- § The whole-block backward graph (body fan-in + outer relu)
-- ════════════════════════════════════════════════════════════════

/-- The whole batched ResNet-34 identity basic block backward graph:
    `selectPos` (outer relu) ∘ residual fan-in (body backward + identity skip).
    The outer relu is the LAST forward op, so its `.selectPos` backward is the
    OUTERMOST backward op; inside, the residual `addV` sums the body's
    input-cotangent (`r34BodyBackBatchedGraph` fed the relu-masked cotangent) and
    the identity skip's verbatim cotangent (`%dy`). -/
noncomputable def r34BasicBlockBackBatchedGraph {N c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ : Vec c)
    (x dy : Vec (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  -- pre-relu activation = residual(F)(x); its relu mask gates the incoming `dy`
  let preRelu := residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                  cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x
  -- the relu-masked cotangent flows into BOTH the body fan-in and the skip
  let masked : SHlo (N * (c * h * w)) := .selectPos "%outR" preRelu (.operand "%dy" dy)
  .addV
    (r34BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x masked)
    masked

/-- **CAPSTONE — the whole batched ResNet-34 identity basic block: backward graph
    ↔ the proven VJP.** The two batched stage backward graphs (`cbReluB`/`projB`)
    chained at their forward activations, wrapped in the residual additive fan-in
    (body cotangent + identity skip) and the OUTER post-residual relu, proven
    equal to `r34BasicBlockB_has_vjp_at` (= `vjp_comp_at(residual_has_vjp_at(F),
    relu)`). The ResNet-34 analogue of `mbResidBlockBackBatchedGraph_faithful` /
    `mnv2ResidBlockBackBatchedGraph_faithful`, with the extra outer-relu factor,
    threaded through both relu smoothness hypotheses.

    Key fact: the outer relu's `.selectPos` mask is applied ONCE to the incoming
    `dy` (giving `masked = relu_has_vjp_at.backward dy`), and that masked cotangent
    is what the residual fan-in (`r34BodyBackBatchedGraph` + identity skip) sees —
    exactly matching `vjp_comp_at(residual, relu)`'s structure: first apply relu's
    backward, then residual's backward to the result. -/
theorem r34BasicBlockBackBatchedGraph_faithful {N c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec c)
    (x dy : Vec (N * (c * h * w)))
    (h_s1 : ∀ k, bnBatchLA N c h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_out : ∀ k, residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                    cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    den (r34BasicBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x dy)
      = (r34BasicBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1 h_out).backward dy := by
  -- The masked cotangent denotes relu's backward applied to dy.
  have hmask : den (SHlo.selectPos "%outR"
        (residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
          cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) (.operand "%dy" dy))
      = (relu_has_vjp_at (N * (c * h * w)) _ h_out).backward dy :=
    selectPos_faithful _ _ h_out (.operand "%dy" dy)
  -- The body backward (fed the masked cotangent) denotes the body VJP at that masked cotangent.
  have hbody : den (r34BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x
        (SHlo.selectPos "%outR"
          (residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
            cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) (.operand "%dy" dy)))
      = (r34BodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1).backward
          ((relu_has_vjp_at (N * (c * h * w)) _ h_out).backward dy) := by
    rw [r34BodyBackBatchedGraph_faithful (hε₁ := hε₁) (hε₂ := hε₂) (h_s1 := h_s1), hmask]
  funext i
  -- LHS unfolds: addV(bodyBack(masked), masked) i = bodyBack(masked) i + masked i.
  have hsum : den (r34BasicBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x dy) i
      = den (r34BodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x
            (SHlo.selectPos "%outR"
              (residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) (.operand "%dy" dy))) i
        + den (SHlo.selectPos "%outR"
              (residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) (.operand "%dy" dy)) i := rfl
  rw [hsum, hbody, hmask]
  -- RHS: vjp_comp_at(residual, relu).backward dy = residual.backward (relu.backward dy)
  --    = bodyBack(relu.backward dy) + (relu.backward dy)  [residual_has_vjp_at = biPath f id]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § DOWNSAMPLE BLOCK — `relu ∘ residualProj(proj, F_s)`
--   stage 1: STRIDED conv-bn-relu (`cbReluStridedB`, uses `convStridedBackBatched`)
-- ════════════════════════════════════════════════════════════════

/-- Batched **STRIDE-2 conv → bn → relu** stage (downsample basic-block first
    stage), at the network layout `N·(oc·h·w)` ← `N·(ic·(2h)·(2w))`. The strided
    sibling of `cbReluB` (`flatConvStride2` for `flatConv`); halves spatial. -/
@[reducible] noncomputable def cbReluStridedB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N (flatConvStride2 W b)

/-- `cbReluStridedB` (strided conv-bn-relu) `_at` VJP at a smooth point. The strided
    sibling of `cbReluB_has_vjp_at` (`flatConvStride2` for `flatConv`). -/
noncomputable def cbReluStridedB_has_vjp_at (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x) k ≠ 0) :
    HasVJPAt (cbReluStridedB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_has_vjp_at N (flatConvStride2 W b) (flatConvStride2_differentiable W b)
    (flatConvStride2_has_vjp W b) ε hε γ β x h_smooth

theorem cbReluStridedB_differentiableAt (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x) k ≠ 0) :
    DifferentiableAt ℝ (cbReluStridedB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_differentiableAt N (flatConvStride2 W b) (flatConvStride2_differentiable W b)
    ε hε γ β x h_smooth

/-- Batched **strided conv → bn → relu** stage backward graph:
    `convStridedBackBatched ∘ bnBatchLABack ∘ selectPos`, each at its cumulative
    forward activation. The strided sibling of `cbReluBackBatchedGraph` —
    `convStridedBackBatched` (the new stride-2 batched-conv VJP) replaces
    `convBackBatched`. -/
noncomputable def cbReluStridedBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  .convStridedBackBatched (N := N) "%cbsrW" W b
    (.bnBatchLABack "%cbsrG" "%cbsrX" "cbsrE" ε γ (batchMap N (flatConvStride2 W b) x)
      (.selectPos "%cbsrR" (bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x)) e))

theorem cbReluStridedBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x) k ≠ 0) :
    den (cbReluStridedBackBatchedGraph W b ε γ β x e)
      = (cbReluStridedB_has_vjp_at N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [cbReluStridedBackBatchedGraph, convStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectPos_faithful _ _ h_smooth]
  simp only [cbReluStridedB_has_vjp_at, bnReluStage_has_vjp_at, vjp_comp_at,
    HasVJP.toHasVJPAt, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The strided projection skip — `projStridedB` (conv_strided → bn, no relu)
-- ════════════════════════════════════════════════════════════════

/-- Batched **strided conv → bn** projection skip (downsample basic-block skip):
    `bnBatchLA ∘ batchMap (flatConvStride2)` — the 3×3 stride-2 projection that
    matches the body's downsampled `oc·h·w` output. The strided sibling of `projB`
    (`flatConvStride2` for `flatConv`); no activation (linear bottleneck). -/
@[reducible] noncomputable def projStridedB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  bnBatchLA N oc h w ε γ β ∘ batchMap N (flatConvStride2 W b)

theorem projStridedB_differentiable (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (projStridedB N (h := h) (w := w) W b ε γ β) :=
  bnStage_differentiable N (flatConvStride2 W b) (flatConvStride2_differentiable W b) ε hε γ β

noncomputable def projStridedB_has_vjp (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (projStridedB N (h := h) (w := w) W b ε γ β) :=
  bnStage_has_vjp N (flatConvStride2 W b) (flatConvStride2_differentiable W b)
    (flatConvStride2_has_vjp W b) ε hε γ β

/-- Batched **strided conv → bn** projection-skip backward graph:
    `convStridedBackBatched ∘ bnBatchLABack`, at the skip's forward activation. The
    strided sibling of `projBackBatchedGraph`. -/
noncomputable def projStridedBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  .convStridedBackBatched (N := N) "%psW" W b
    (.bnBatchLABack "%psG" "%psX" "psE" ε γ (batchMap N (flatConvStride2 W b) x) e)

theorem projStridedBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    den (projStridedBackBatchedGraph W b ε γ β x e)
      = (projStridedB_has_vjp N W b ε hε γ β).backward x (den e) := by
  rw [projStridedBackBatchedGraph, convStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε)]
  simp only [projStridedB_has_vjp, bnStage_has_vjp, vjp_comp]

-- ════════════════════════════════════════════════════════════════
-- § The downsample body: `projB ∘ cbReluStridedB`  (strided conv1, stride-1 conv2)
-- ════════════════════════════════════════════════════════════════

/-- The batched ResNet-34 downsample-block body's VJP at a smooth point —
    `projB ∘ cbReluStridedB` (stride-1 conv-bn after STRIDED conv-bn-relu). One
    `vjp_comp_at` threading the mid-relu smoothness family; `projB` (global, no
    activation) is lifted via `.toHasVJPAt`. The strided sibling of
    `r34BodyB_has_vjp_at` (`cbReluStridedB` for `cbReluB`). -/
noncomputable def r34DownBodyB_has_vjp_at (N : Nat) {ic oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_s1 : ∀ k, bnBatchLA N oc h w ε₁ γ₁ β₁ (batchMap N (flatConvStride2 W₁ b₁) x) k ≠ 0) :
    HasVJPAt (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x := by
  have h1_vjp : HasVJPAt (cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    cbReluStridedB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1
  have h1_diff : DifferentiableAt ℝ (cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    cbReluStridedB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1
  exact vjp_comp_at _ (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂) x
    h1_diff
    ((projB_differentiable N (h := h) (w := w) W₂ b₂ ε₂ hε₂ γ₂ β₂) _)
    h1_vjp
    ((projB_has_vjp N (h := h) (w := w) W₂ b₂ ε₂ hε₂ γ₂ β₂).toHasVJPAt _)

theorem r34DownBodyB_differentiableAt (N : Nat) {ic oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_s1 : ∀ k, bnBatchLA N oc h w ε₁ γ₁ β₁ (batchMap N (flatConvStride2 W₁ b₁) x) k ≠ 0) :
    DifferentiableAt ℝ (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
              cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
  ((projB_differentiable N (h := h) (w := w) W₂ b₂ ε₂ hε₂ γ₂ β₂) _).comp x
    (cbReluStridedB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ x h_s1)

/-- The batched ResNet-34 downsample body backward graph: the two stage graphs
    chained at their cumulative forward activations (`cbReluStridedB⁻¹ ∘ projB⁻¹`).
    The strided sibling of `r34BodyBackBatchedGraph`. -/
noncomputable def r34DownBodyBackBatchedGraph {N ic oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  let x1 := cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x
  cbReluStridedBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ x
    (projBackBatchedGraph W₂ b₂ ε₂ γ₂ β₂ x1 e)

theorem r34DownBodyBackBatchedGraph_faithful {N ic oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w)))
    (h_s1 : ∀ k, bnBatchLA N oc h w ε₁ γ₁ β₁ (batchMap N (flatConvStride2 W₁ b₁) x) k ≠ 0) :
    den (r34DownBodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x e)
      = (r34DownBodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1).backward (den e) := by
  rw [r34DownBodyBackBatchedGraph,
      cbReluStridedBackBatchedGraph_faithful (hε := hε₁) (h_smooth := h_s1),
      projBackBatchedGraph_faithful (hε := hε₂)]
  simp only [r34DownBodyB_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The whole downsample-block VJP: `relu ∘ residualProj(proj, F_s)`
-- ════════════════════════════════════════════════════════════════

/-- The batched ResNet-34 downsample basic block's VJP at a smooth point —
    `relu ∘ residualProj(proj, F_s)` with body `F_s = projB ∘ cbReluStridedB` and
    projection skip `proj = projStridedB`. One `vjp_comp_at` composing the projected
    residual fan-in VJP (`residualProj_has_vjp_at` of skip + body) with the OUTER
    relu's pointwise VJP at the pre-relu activation `residualProj(proj, F_s)(x)`.

    The strided sibling of `r34BasicBlockB_has_vjp_at`: `residualProj` (BOTH paths
    nontrivial) for `residual` (identity skip), strided convs in body+skip.

    `h_s1` is the body's mid-relu smoothness; `h_out` is the outer-relu smoothness. -/
noncomputable def r34DownBlockB_has_vjp_at (N : Nat) {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_s1 : ∀ k, bnBatchLA N oc h w ε₁ γ₁ β₁ (batchMap N (flatConvStride2 W₁ b₁) x) k ≠ 0)
    (h_out : ∀ k, residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                    (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                     cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    HasVJPAt (relu (N * (oc * h * w)) ∘
              residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                 cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x := by
  have hbody_vjp : HasVJPAt (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
        cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    r34DownBodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1
  have hbody_diff : DifferentiableAt ℝ (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
        cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x :=
    r34DownBodyB_differentiableAt N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1
  have hproj_vjp : HasVJPAt (projStridedB N (h := h) (w := w) Wp bp εp γp βp) x :=
    (projStridedB_has_vjp N Wp bp εp hεp γp βp).toHasVJPAt x
  have hproj_diff : DifferentiableAt ℝ (projStridedB N (h := h) (w := w) Wp bp εp γp βp) x :=
    (projStridedB_differentiable N Wp bp εp hεp γp βp) x
  have hres_vjp : HasVJPAt (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
        (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
         cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x :=
    residualProj_has_vjp_at _ _ x hproj_diff hbody_diff hproj_vjp hbody_vjp
  have hres_diff : DifferentiableAt ℝ (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
        (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
         cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x :=
    hproj_diff.add hbody_diff
  exact vjp_comp_at _ (relu (N * (oc * h * w))) x
    hres_diff
    (relu_differentiableAt_of_smooth (N * (oc * h * w)) _ h_out)
    hres_vjp
    (relu_has_vjp_at (N * (oc * h * w)) _ h_out)

-- ════════════════════════════════════════════════════════════════
-- § The whole downsample-block backward graph (proj+body fan-in + outer relu)
-- ════════════════════════════════════════════════════════════════

/-- The whole batched ResNet-34 downsample basic block backward graph:
    `selectPos` (outer relu) ∘ projected-residual fan-in (body backward +
    PROJECTION skip backward). The outer relu is the LAST forward op, so its
    `.selectPos` backward is the OUTERMOST backward op; inside, the `residualProj`
    `addV` sums the projection skip's input-cotangent (`projStridedBackBatchedGraph`
    fed the relu-masked cotangent) and the body's input-cotangent
    (`r34DownBodyBackBatchedGraph`, same masked cotangent). Unlike the identity
    block, BOTH operands are nontrivial backward subgraphs (the skip is a strided
    conv-bn, not a verbatim `%dy` passthrough). -/
noncomputable def r34DownBlockBackBatchedGraph {N ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  -- pre-relu activation = residualProj(proj, F_s)(x); its relu mask gates `dy`
  let preRelu := residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                  (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                   cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x
  -- the relu-masked cotangent flows into BOTH the body fan-in AND the projection skip
  let masked : SHlo (N * (oc * h * w)) := .selectPos "%outR" preRelu (.operand "%dy" dy)
  .addV
    (projStridedBackBatchedGraph Wp bp εp γp βp x masked)
    (r34DownBodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x masked)

/-- **CAPSTONE — the whole batched ResNet-34 DOWNSAMPLE basic block: backward graph
    ↔ the proven VJP.** The two batched stage backward graphs of the body
    (`cbReluStridedB`/`projB`) chained at their forward activations, wrapped in the
    PROJECTED-residual additive fan-in (body cotangent + STRIDED projection-skip
    cotangent) and the OUTER post-residual relu, proven equal to
    `r34DownBlockB_has_vjp_at` (= `vjp_comp_at(residualProj_has_vjp_at(proj, F_s),
    relu)`). The strided sibling of `r34BasicBlockBackBatchedGraph_faithful`:
    `residualProj` (both backward paths nontrivial) for `residual` (identity skip),
    `convStridedBackBatched` in the body's conv1 and the whole projection skip.

    Key fact: the outer relu's `.selectPos` mask is applied ONCE to the incoming
    `dy` (giving `masked = relu_has_vjp_at.backward dy`), and that masked cotangent
    is what BOTH residualProj fan-in operands see — matching
    `vjp_comp_at(residualProj, relu)`'s structure: first relu's backward, then
    `residualProj`'s backward (= `proj.backward + body.backward` at the masked
    cotangent). -/
theorem r34DownBlockBackBatchedGraph_faithful {N ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w)))
    (h_s1 : ∀ k, bnBatchLA N oc h w ε₁ γ₁ β₁ (batchMap N (flatConvStride2 W₁ b₁) x) k ≠ 0)
    (h_out : ∀ k, residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                    (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                     cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    den (r34DownBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ Wp bp εp γp βp x dy)
      = (r34DownBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          Wp bp εp hεp γp βp x h_s1 h_out).backward dy := by
  -- The masked cotangent denotes relu's backward applied to dy.
  have hmask : den (SHlo.selectPos "%outR"
        (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
          (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
           cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) (.operand "%dy" dy))
      = (relu_has_vjp_at (N * (oc * h * w)) _ h_out).backward dy :=
    selectPos_faithful _ _ h_out (.operand "%dy" dy)
  -- The body backward (fed the masked cotangent) denotes the body VJP at it.
  have hbody : den (r34DownBodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x
        (SHlo.selectPos "%outR"
          (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
            (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
             cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) (.operand "%dy" dy)))
      = (r34DownBodyB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1).backward
          ((relu_has_vjp_at (N * (oc * h * w)) _ h_out).backward dy) := by
    rw [r34DownBodyBackBatchedGraph_faithful (hε₁ := hε₁) (hε₂ := hε₂) (h_s1 := h_s1), hmask]
  -- The projection-skip backward (same masked cotangent) denotes the skip VJP at it.
  have hproj : den (projStridedBackBatchedGraph Wp bp εp γp βp x
        (SHlo.selectPos "%outR"
          (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
            (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
             cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) (.operand "%dy" dy)))
      = (projStridedB_has_vjp N Wp bp εp hεp γp βp).backward x
          ((relu_has_vjp_at (N * (oc * h * w)) _ h_out).backward dy) := by
    rw [projStridedBackBatchedGraph_faithful (hε := hεp), hmask]
  funext i
  -- LHS unfolds: addV(projBack(masked), bodyBack(masked)) i = projBack(masked) i + bodyBack(masked) i.
  have hsum : den (r34DownBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ Wp bp εp γp βp x dy) i
      = den (projStridedBackBatchedGraph Wp bp εp γp βp x
            (SHlo.selectPos "%outR"
              (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                 cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) (.operand "%dy" dy))) i
        + den (r34DownBodyBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x
              (SHlo.selectPos "%outR"
                (residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                  (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                   cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x) (.operand "%dy" dy))) i := rfl
  rw [hsum, hbody, hproj]
  -- RHS: vjp_comp_at(residualProj, relu).backward dy = residualProj.backward (relu.backward dy)
  --   = proj.backward(relu.backward dy) + body.backward(relu.backward dy)
  --   [residualProj_has_vjp_at = biPath_has_vjp_at proj body]
  rfl

end Proofs.StableHLO
