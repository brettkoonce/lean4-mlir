import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2BackB0
import LeanMlir.Proofs.Foundation.CertifiedChain

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
* the body `cbB ∘ cbReluB` — its backward graph `r34BodyBackBatchedGraph`, the two stage
  graphs chained at their cumulative activations, certified through the block layer.
* `r34BasicBlockBackBatchedGraph_faithful` — the **CAPSTONE**: the whole batched
  ResNet-34 identity basic block backward graph (outer-relu `selectPos` ∘
  residual-fan-in(body-back) + identity skip) denotes the proven
  `relu ∘ residual(F)` VJP (`vjp_comp_at(residual_has_vjp_at(body), relu)`),
  threaded through both relu smoothness hypotheses.
* `cbReluLayer` / `projLayer` (from `MobileNetV2BackB0`) / `cbReluStridedLayer` /
  `projStridedLayer` — the four stages as `CertLayer`s. `r34BasicBlockLayer` / `r34DownBlockLayer` compose them with
  `CertLayer.comp`, `residual` / `residualProj` and `reluOut`, and each body/block VJP
  and capstone here is that composite's `.vjp` / `.faithful`.

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
    HasVJPAt (relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x :=
  stage_has_vjp_at (batchMap N op) (bnBatchLA N oc h w ε γ β) (relu (N * (oc * h * w))) x
    (batchMap_differentiable op hop) (batchMap_has_vjp op hopv hop)
    (bnBatchLA_differentiable N oc h w ε hε γ β) (bnBatchLA_has_vjp N oc h w ε hε γ β)
    (relu_differentiableAt_of_smooth (N * (oc * h * w)) _ h_smooth)
    (relu_has_vjp_at (N * (oc * h * w)) _ h_smooth)

/-- Differentiability of the generic relu-on-batched-bn-stage at a smooth point. -/
theorem bnReluStage_differentiableAt (N : Nat) {a oc h w : Nat}
    (op : Vec a → Vec (oc * h * w)) (hop : Differentiable ℝ op)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * a))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 0) :
    DifferentiableAt ℝ (relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x := by
  fun_prop (disch := assumption)

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
  simp only [cbReluB_has_vjp_at, bnReluStage_has_vjp_at, stage_has_vjp_at, vjp_comp_at_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

/-- The conv → bn → relu stage as a `CertLayer`, certified where its pre-relu activation misses 0.
    The kernel extent is a binder, so the same layer is a 1×1 or a 3×3. -/
noncomputable def cbReluLayer (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) where
  fwd := cbReluB N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0
  diff := fun x hx => cbReluB_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => cbReluB_has_vjp_at N W b ε hε γ β x hx
  graph := fun x e => cbReluBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => cbReluBackBatchedGraph_faithful W b ε hε γ β x e hx

/-- `cbReluLayer`'s forward is `cbReluB`. -/
theorem cbReluLayer_fwd_apply (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v : Vec (N * (ic * h * w))) :
    (cbReluLayer (h := h) (w := w) N W b ε hε γ β).fwd v
      = cbReluB N (h := h) (w := w) W b ε γ β v := rfl

-- ════════════════════════════════════════════════════════════════
-- § The body: `cbB ∘ cbReluB`  (= projB ∘ cbReluB)
-- ════════════════════════════════════════════════════════════════

/-- The batched ResNet-34 body backward graph: the two stage graphs chained at
    their cumulative forward activations (`cbReluB⁻¹ ∘ projB⁻¹`). -/
noncomputable def r34BodyBackBatchedGraph {N c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  let x1 := cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ x
  cbReluBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ x
    (projBackBatchedGraph W₂ b₂ ε₂ γ₂ β₂ x1 e)

-- ════════════════════════════════════════════════════════════════
-- § The whole-block VJP: `relu ∘ residual(F)` (outer relu after the add)
-- ════════════════════════════════════════════════════════════════

/-- The batched R34 identity basic block as a `CertLayer`: `residual (cbReluLayer ; projLayer)`,
    then `reluOut`. Its `ok` is the body's mid-relu and the OUTER post-residual relu — the extra
    factor R34 has over the MBConv/inverted-residual blocks (`projLayer` contributes `True`). An
    endomorphism, so `chain` iterates it into a stage tail. -/
noncomputable def r34BasicBlockLayer (N : Nat) {c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  (CertLayer.residual ((cbReluLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁).comp
    (projLayer N W₂ b₂ ε₂ hε₂ γ₂ β₂))).comp (CertLayer.reluOut _)

/-- The batched ResNet-34 identity basic block's VJP at a smooth point —
    `relu ∘ residual(F)` with body `F = projB ∘ cbReluB`: the residual fan-in VJP
    of the body, then the OUTER relu's pointwise VJP at the pre-relu activation
    `residual(F)(x)` (`r34BasicBlockLayer`'s VJP).

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
                        cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x :=
  (r34BasicBlockLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂).vjp x
    ⟨⟨h_s1, trivial⟩, h_out⟩

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
    (x : Vec (N * (c * h * w))) (ecot : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  -- pre-relu activation = residual(F)(x); its relu mask gates the incoming `dy`
  let preRelu := residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                  cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x
  -- the relu-masked cotangent flows into BOTH the body fan-in and the skip
  let masked : SHlo (N * (c * h * w)) := .selectPos "%outR" preRelu ecot
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
    threaded through both relu smoothness hypotheses. It is `r34BasicBlockLayer`'s
    `faithful`.

    Key fact: the outer relu's `.selectPos` mask is applied ONCE to the incoming
    `dy` (giving `masked = relu_has_vjp_at.backward (den ecot)`), and that masked cotangent
    is what the residual fan-in (`r34BodyBackBatchedGraph` + identity skip) sees —
    exactly matching `vjp_comp_at(residual, relu)`'s structure: first apply relu's
    backward, then residual's backward to the result. -/
theorem r34BasicBlockBackBatchedGraph_faithful {N c h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec c)
    (x : Vec (N * (c * h * w))) (ecot : SHlo (N * (c * h * w)))
    (h_s1 : ∀ k, bnBatchLA N c h w ε₁ γ₁ β₁ (batchMap N (flatConv W₁ b₁) x) k ≠ 0)
    (h_out : ∀ k, residual (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                    cbReluB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    den (r34BasicBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ x ecot)
      = (r34BasicBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂ x h_s1 h_out).backward (den ecot) :=
  (r34BasicBlockLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂).faithful x
    ⟨⟨h_s1, trivial⟩, h_out⟩ ecot

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
  simp only [cbReluStridedB_has_vjp_at, bnReluStage_has_vjp_at, stage_has_vjp_at, vjp_comp_at_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

/-- The strided conv → bn → relu stage as a `CertLayer` — `cbReluLayer` with `flatConvStride2`. -/
noncomputable def cbReluStridedLayer (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) where
  fwd := cbReluStridedB N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x) k ≠ 0
  diff := fun x hx => cbReluStridedB_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => cbReluStridedB_has_vjp_at N W b ε hε γ β x hx
  graph := fun x e => cbReluStridedBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => cbReluStridedBackBatchedGraph_faithful W b ε hε γ β x e hx

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
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ _β : Vec oc)
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
  simp only [projStridedB_has_vjp, bnStage_has_vjp, vjp_comp_backward]

/-- The strided conv → bn projection skip as a `CertLayer` — `projLayer` with `flatConvStride2`,
    globally certified. -/
noncomputable def projStridedLayer (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) where
  fwd := projStridedB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (projStridedB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (projStridedB_has_vjp N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => projStridedBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => projStridedBackBatchedGraph_faithful W b ε hε γ β x e

-- ════════════════════════════════════════════════════════════════
-- § The downsample body: `projB ∘ cbReluStridedB`  (strided conv1, stride-1 conv2)
-- ════════════════════════════════════════════════════════════════

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

-- ════════════════════════════════════════════════════════════════
-- § The whole downsample-block VJP: `relu ∘ residualProj(proj, F_s)`
-- ════════════════════════════════════════════════════════════════

/-- The batched R34 downsample basic block as a `CertLayer`: `residualProj (projStridedLayer)
    (cbReluStridedLayer ; projLayer)`, then `reluOut`. Halves resolution (hence the `2*h` in the
    input type), with a strided conv1 and a strided projection skip. -/
noncomputable def r34DownBlockLayer (N : Nat) {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  (CertLayer.residualProj (projStridedLayer N (h := h) (w := w) Wp bp εp hεp γp βp)
    ((cbReluStridedLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁).comp
      (projLayer N W₂ b₂ ε₂ hε₂ γ₂ β₂))).comp (CertLayer.reluOut _)

/-- The batched ResNet-34 downsample basic block's VJP at a smooth point —
    `relu ∘ residualProj(proj, F_s)` with body `F_s = projB ∘ cbReluStridedB` and
    projection skip `proj = projStridedB`: the projected residual fan-in VJP (skip +
    body), then the OUTER relu's pointwise VJP at the pre-relu activation
    `residualProj(proj, F_s)(x)` (`r34DownBlockLayer`'s VJP).

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
                 cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁)) x :=
  (r34DownBlockLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    Wp bp εp hεp γp βp).vjp x ⟨⟨trivial, h_s1, trivial⟩, h_out⟩

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
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (ecot : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  -- pre-relu activation = residualProj(proj, F_s)(x); its relu mask gates `dy`
  let preRelu := residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                  (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                   cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x
  -- the relu-masked cotangent flows into BOTH the body fan-in AND the projection skip
  let masked : SHlo (N * (oc * h * w)) := .selectPos "%outR" preRelu ecot
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
    `convStridedBackBatched` in the body's conv1 and the whole projection skip. It is
    `r34DownBlockLayer`'s `faithful`.

    Key fact: the outer relu's `.selectPos` mask is applied ONCE to the incoming
    `dy` (giving `masked = relu_has_vjp_at.backward (den ecot)`), and that masked cotangent
    is what BOTH residualProj fan-in operands see — matching
    `vjp_comp_at(residualProj, relu)`'s structure: first relu's backward, then
    `residualProj`'s backward (= `proj.backward + body.backward` at the masked
    cotangent). -/
theorem r34DownBlockBackBatchedGraph_faithful {N ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc) (ε₁ : ℝ) (hε₁ : 0 < ε₁) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc) (ε₂ : ℝ) (hε₂ : 0 < ε₂) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (ecot : SHlo (N * (oc * h * w)))
    (h_s1 : ∀ k, bnBatchLA N oc h w ε₁ γ₁ β₁ (batchMap N (flatConvStride2 W₁ b₁) x) k ≠ 0)
    (h_out : ∀ k, residualProj (projStridedB N (h := h) (w := w) Wp bp εp γp βp)
                    (projB N (h := h) (w := w) W₂ b₂ ε₂ γ₂ β₂ ∘
                     cbReluStridedB N (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁) x k ≠ 0) :
    den (r34DownBlockBackBatchedGraph W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ Wp bp εp γp βp x ecot)
      = (r34DownBlockB_has_vjp_at N W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
          Wp bp εp hεp γp βp x h_s1 h_out).backward (den ecot) :=
  (r34DownBlockLayer N (h := h) (w := w) W₁ b₁ ε₁ hε₁ γ₁ β₁ W₂ b₂ ε₂ hε₂ γ₂ β₂
    Wp bp εp hεp γp βp).faithful x ⟨⟨trivial, h_s1, trivial⟩, h_out⟩ ecot

end Proofs.StableHLO
