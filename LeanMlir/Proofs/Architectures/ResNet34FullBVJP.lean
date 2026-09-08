import LeanMlir.Proofs.Architectures.ResNet34FullB
import LeanMlir.Proofs.Foundation.BatchMapVJPAt
import LeanMlir.Proofs.Foundation.BackwardMaps

/-! # ResNet-34's whole-net input-VJP at TRUE BATCH-NORM (T1, the VJP half)

`ResNet34FullB.lean` states the batch-BN forward and its typed graph. This file gives that forward
a certified `HasVJPAt` at the paper depth — the batched peer of what `MobileNetV2FullVJP.lean` does
for MobileNetV2's seventeen bottlenecks, and the last piece of T1 in `formalization.yaml` 4e's port.

## No new mathematics, and one new lemma one tier down

Every block VJP is already proven at `bnBatchLA`: `r34BasicBlockB_has_vjp_at` and
`r34DownBlockB_has_vjp_at` (`ResNet34BackB0.lean`) are exactly the two block shapes
`ResNet34FullB.lean`'s `r34IdB` / `r34DownB` unfold to. The eight bundle lemmas below are
delegations in the `EfficientNetFullB0` style.

⭐ The one thing that did not exist is `batchMap_has_vjp_at` (`Foundation/BatchMapVJPAt.lean`,
written for this): r34's stem ends in `batchMap N (maxPool3s2Flat 64 56 56)` and a max-pool has no
derivative at a tie, so the GLOBAL `batchMap_has_vjp` cannot lift it. The per-example pool VJP at a
`Vec` point was already there (`maxPool3s2Flat_has_vjp_at_vec`), so the batched pool is two lines.

## The hypothesis budget

⚠ **Pointwise (`HasVJPAt`), not global, and necessarily.** relu is kinked. ⛔ And r34 carries
**two** kink clauses per block, not one: the body's mid-relu AND the post-residual outer relu.
That outer relu is ResNet's structural difference from MobileNetV2/EfficientNet, whose residual
add IS the block output. Sixteen blocks therefore carry 32 clauses, plus the stem's relu and the
stem pool's no-tie condition — bundled per block into `R34IdSmoothAt` / `R34DownSmoothAt` so the
apex binds 18 smoothness bundles rather than 34 loose hypotheses, exactly as
`MobileNetV2FullVJP.lean` bundles `IVSmoothAt`.

⚠ The pool's condition is **per example** (`∀ r : Fin N, MaxPool3s2Smooth …` on that example's
row), because a tie is a property of one image's window, not of the batch. That is the shape
`batchMap_has_vjp_at` consumes.

The running activations are named `r34Pre1 … r34Pre16` so each bundle can be STATED at the
activation entering its block without a sixteen-deep nested application inline; `r34Pre16` doubles
as the trunk, and `resnet34ForwardB_full_eq_chain` bridges it back to the committed
nested-application forward.

⭐ `N` is a variable throughout: this tier carries no numerals.
-/

namespace Proofs

open scoped BigOperators

set_option maxHeartbeats 1000000

-- ════════════════════════════════════════════════════════════════
-- § Per-block hypothesis bundles
-- ════════════════════════════════════════════════════════════════

/-- Both BN epsilons of an identity basic block are positive. -/
structure R34IdPos {c : Nat} (q : R34IdW c) : Prop where
  h1 : 0 < q.ε₁
  h2 : 0 < q.ε₂

/-- All three BN epsilons of a downsample basic block are positive (body twice, projection once). -/
structure R34DownPos {ic oc : Nat} (q : R34DownW ic oc) : Prop where
  h1 : 0 < q.ε₁
  h2 : 0 < q.ε₂
  hp : 0 < q.εp

/-- Both relu sites of an identity basic block are away from the kink at `v`: the body's mid-relu
    (at the first BN's output) and the OUTER relu (at the post-residual sum). -/
structure R34IdSmoothAt (N h w : Nat) {c : Nat} (q : R34IdW c)
    (v : Vec (N * (c * h * w))) : Prop where
  hmid : ∀ k, StableHLO.bnBatchLA N c h w q.ε₁ q.γ₁ q.β₁
    (StableHLO.batchMap N (flatConv q.W₁ q.b₁) v) k ≠ 0
  hout : ∀ k, residual (projB N (h := h) (w := w) q.W₂ q.b₂ q.ε₂ q.γ₂ q.β₂ ∘
    StableHLO.cbReluB N (h := h) (w := w) q.W₁ q.b₁ q.ε₁ q.γ₁ q.β₁) v k ≠ 0

/-- Both relu sites of a downsample basic block are away from the kink at `v`. The mid-relu is
    after the STRIDED conv's BN (already at `h×w`); the outer relu is at the projected residual. -/
structure R34DownSmoothAt (N h w : Nat) {ic oc : Nat} (q : R34DownW ic oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Prop where
  hmid : ∀ k, StableHLO.bnBatchLA N oc h w q.ε₁ q.γ₁ q.β₁
    (StableHLO.batchMap N (flatConvStride2 q.W₁ q.b₁) v) k ≠ 0
  hout : ∀ k, residualProj (StableHLO.projStridedB N (h := h) (w := w) q.Wp q.bp q.εp q.γp q.βp)
    (projB N (h := h) (w := w) q.W₂ q.b₂ q.ε₂ q.γ₂ q.β₂ ∘
      StableHLO.cbReluStridedB N (h := h) (w := w) q.W₁ q.b₁ q.ε₁ q.γ₁ q.β₁) v k ≠ 0

/-- The stem's relu is away from the kink at the input `x` (the 7×7/s2 conv's BN output, at the
    pre-pool `2h×2w` grid). -/
def R34StemSmoothAt (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) : Prop :=
  ∀ k, StableHLO.bnBatchLA N oc (2 * h) (2 * w) εs γs βs
    (StableHLO.batchMap N (flatConvStride2 Ws bs) x) k ≠ 0

/-- The stem pool has no argmax tie, **per example**: a tie is a property of one image's 3×3
    window, so the condition is stated on each row of the batched activation. This is the shape
    `batchMap_has_vjp_at` consumes. -/
def R34PoolSmoothAt (N h w : Nat) {oc : Nat} (v : Vec (N * (oc * (2 * h) * (2 * w)))) : Prop :=
  ∀ r : Fin N,
    MaxPool3s2Smooth (Tensor3.unflatten (Mat.unflatten v r) : Tensor3 oc (2 * h) (2 * w))

-- ════════════════════════════════════════════════════════════════
-- § Block, stem and head bundle lemmas (delegation)
-- ════════════════════════════════════════════════════════════════

/-- Identity basic block VJP — `r34BasicBlockB_has_vjp_at` at the bundle's fields. -/
noncomputable def r34IdB_has_vjp_at (N h w : Nat) {c : Nat} (p : R34IdW c)
    (hq : R34IdPos p) (v : Vec (N * (c * h * w))) (hs : R34IdSmoothAt N h w p v) :
    HasVJPAt (r34IdB N h w p) v :=
  StableHLO.r34BasicBlockB_has_vjp_at N p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ v hs.hmid hs.hout

theorem r34IdB_differentiableAt (N h w : Nat) {c : Nat} (p : R34IdW c)
    (hq : R34IdPos p) (v : Vec (N * (c * h * w))) (hs : R34IdSmoothAt N h w p v) :
    DifferentiableAt ℝ (r34IdB N h w p) v := by
  have hbody : DifferentiableAt ℝ (projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      StableHLO.cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v :=
    StableHLO.r34BodyB_differentiableAt N p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
      p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ v hs.hmid
  exact (relu_differentiableAt_of_smooth (N * (c * h * w)) _ hs.hout).comp v
    (hbody.add differentiable_id.differentiableAt)

/-- Downsample basic block VJP — `r34DownBlockB_has_vjp_at` at the bundle's fields. -/
noncomputable def r34DownB_has_vjp_at (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (hq : R34DownPos p) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : R34DownSmoothAt N h w p v) :
    HasVJPAt (r34DownB N h w p) v :=
  StableHLO.r34DownBlockB_has_vjp_at N p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.Wp p.bp p.εp hq.hp p.γp p.βp v hs.hmid hs.hout

theorem r34DownB_differentiableAt (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (hq : R34DownPos p) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : R34DownSmoothAt N h w p v) :
    DifferentiableAt ℝ (r34DownB N h w p) v := by
  have hbody : DifferentiableAt ℝ (projB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      StableHLO.cbReluStridedB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v :=
    StableHLO.r34DownBodyB_differentiableAt N p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
      p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ v hs.hmid
  have hproj : DifferentiableAt ℝ
      (StableHLO.projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp) v :=
    (StableHLO.projStridedB_differentiable N p.Wp p.bp p.εp hq.hp p.γp p.βp) v
  exact (relu_differentiableAt_of_smooth (N * (oc * h * w)) _ hs.hout).comp v
    (hproj.add hbody)

/-- ⭐ Stem VJP: the 7×7/s2 conv-bn-relu, then the batched 3×3/s2 pool. The pool half is where
    `batchMap_has_vjp_at` earns its existence. -/
noncomputable def r34StemB_has_vjp_at (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (hc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (hrelu : R34StemSmoothAt N h w Ws bs εs γs βs x)
    (hpool : R34PoolSmoothAt N h w
      (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x)) :
    HasVJPAt (r34StemB N h w Ws bs εs γs βs) x :=
  vjp_comp_at _ (StableHLO.batchMap N (maxPool3s2Flat oc h w)) x
    (StableHLO.cbReluStridedB_differentiableAt N Ws bs εs hεs γs βs x hrelu)
    (batchMap_differentiableAt _ _
      (fun r => maxPool3s2Flat_differentiableAt_vec _ (hpool r) hc hh hw))
    (StableHLO.cbReluStridedB_has_vjp_at N Ws bs εs hεs γs βs x hrelu)
    (batchMap_has_vjp_at _ _
      (fun r => maxPool3s2Flat_has_vjp_at_vec _ (hpool r))
      (fun r => maxPool3s2Flat_differentiableAt_vec _ (hpool r) hc hh hw))

theorem r34StemB_differentiableAt (N h w : Nat) {ic oc : Nat}
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (hc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (hrelu : R34StemSmoothAt N h w Ws bs εs γs βs x)
    (hpool : R34PoolSmoothAt N h w
      (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) Ws bs εs γs βs x)) :
    DifferentiableAt ℝ (r34StemB N h w Ws bs εs γs βs) x :=
  (batchMap_differentiableAt _ _
      (fun r => maxPool3s2Flat_differentiableAt_vec _ (hpool r) hc hh hw)).comp x
    (StableHLO.cbReluStridedB_differentiableAt N Ws bs εs hεs γs βs x hrelu)

/-- ⭐ The head is GLOBAL — GAP and dense are both smooth everywhere, and each is `batchMap` of a
    per-example op, so `batchMap_has_vjp` suffices and no smoothness hypothesis appears. -/
noncomputable def r34HeadB_has_vjp (N h w : Nat) {c nCls : Nat}
    (Wd : Mat c nCls) (bd : Vec nCls) : HasVJP (r34HeadB N h w Wd bd) :=
  vjp_comp _ _
    (batchMap_differentiable _ (globalAvgPoolFlat_differentiable c h w))
    (batchMap_differentiable _ (dense_differentiable Wd bd))
    (batchMap_has_vjp _ (globalAvgPoolFlat_has_vjp c h w) (globalAvgPoolFlat_differentiable c h w))
    (batchMap_has_vjp _ (dense_has_vjp Wd bd) (dense_differentiable Wd bd))

theorem r34HeadB_differentiable (N h w : Nat) {c nCls : Nat}
    (Wd : Mat c nCls) (bd : Vec nCls) : Differentiable ℝ (r34HeadB N h w Wd bd) :=
  (batchMap_differentiable _ (dense_differentiable Wd bd)).comp
    (batchMap_differentiable _ (globalAvgPoolFlat_differentiable c h w))


-- ════════════════════════════════════════════════════════════════
-- § The running activations — `r34PreK` = the net truncated after block `K`
--   `r34Pre0` is the stem (conv-bn-relu + pool); each later one is one `∘` deeper, and
--   `r34Pre16` is the whole [3,4,6,3] trunk. They exist so the 18 kink bundles can be STATED
--   at the activation entering their block without a sixteen-deep nested application inline.
-- ════════════════════════════════════════════════════════════════

noncomputable def r34Pre0 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (64 * 56 * 56)) :=
  r34StemB N 56 56 w.sW w.sb w.sε w.sγ w.sβ
noncomputable def r34Pre1 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (64 * 56 * 56)) :=
  r34IdB N 56 56 w.a0 ∘ r34Pre0 N w
noncomputable def r34Pre2 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (64 * 56 * 56)) :=
  r34IdB N 56 56 w.a1 ∘ r34Pre1 N w
noncomputable def r34Pre3 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (64 * 56 * 56)) :=
  r34IdB N 56 56 w.a2 ∘ r34Pre2 N w
noncomputable def r34Pre4 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (128 * 28 * 28)) :=
  r34DownB N 28 28 w.d2 ∘ r34Pre3 N w
noncomputable def r34Pre5 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (128 * 28 * 28)) :=
  r34IdB N 28 28 w.b0 ∘ r34Pre4 N w
noncomputable def r34Pre6 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (128 * 28 * 28)) :=
  r34IdB N 28 28 w.b1 ∘ r34Pre5 N w
noncomputable def r34Pre7 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (128 * 28 * 28)) :=
  r34IdB N 28 28 w.b2 ∘ r34Pre6 N w
noncomputable def r34Pre8 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (256 * 14 * 14)) :=
  r34DownB N 14 14 w.d3 ∘ r34Pre7 N w
noncomputable def r34Pre9 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (256 * 14 * 14)) :=
  r34IdB N 14 14 w.c0 ∘ r34Pre8 N w
noncomputable def r34Pre10 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (256 * 14 * 14)) :=
  r34IdB N 14 14 w.c1 ∘ r34Pre9 N w
noncomputable def r34Pre11 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (256 * 14 * 14)) :=
  r34IdB N 14 14 w.c2 ∘ r34Pre10 N w
noncomputable def r34Pre12 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (256 * 14 * 14)) :=
  r34IdB N 14 14 w.c3 ∘ r34Pre11 N w
noncomputable def r34Pre13 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (256 * 14 * 14)) :=
  r34IdB N 14 14 w.c4 ∘ r34Pre12 N w
noncomputable def r34Pre14 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (512 * 7 * 7)) :=
  r34DownB N 7 7 w.d4 ∘ r34Pre13 N w
noncomputable def r34Pre15 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (512 * 7 * 7)) :=
  r34IdB N 7 7 w.e0 ∘ r34Pre14 N w
noncomputable def r34Pre16 (N : Nat) {nCls : Nat} (w : R34BWeights nCls) :
    Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) → Vec (N * (512 * 7 * 7)) :=
  r34IdB N 7 7 w.e1 ∘ r34Pre15 N w

-- ════════════════════════════════════════════════════════════════
-- § The apex
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **ResNet-34 at TRUE BATCH-NORM has a certified input-VJP at a smooth point — all sixteen
    basic blocks.** Chains stem → the [3,4,6,3] ladder → head with `vjp_comp_at`, one positivity
    bundle and one smoothness bundle per block. T1's VJP half for `formalization.yaml` 4e's port.

    ⚠ Pointwise, and necessarily: relu is kinked. ⛔ Each block contributes TWO clauses — the
    body's mid-relu and the post-residual OUTER relu — where MobileNetV2's bottleneck contributes
    two relu6 clauses and EfficientNet's MBConv contributes none.

    ⭐ The head takes no hypothesis at all (GAP and dense are smooth, and each is `batchMap` of a
    per-example op), and `N` is a variable: this tier carries no numerals. -/
noncomputable def resnet34ForwardB_full_has_vjp_at (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hsε : 0 < w.sε)
    (qa0 : R34IdPos w.a0)
    (qa1 : R34IdPos w.a1)
    (qa2 : R34IdPos w.a2)
    (qd2 : R34DownPos w.d2)
    (qb0 : R34IdPos w.b0)
    (qb1 : R34IdPos w.b1)
    (qb2 : R34IdPos w.b2)
    (qd3 : R34DownPos w.d3)
    (qc0 : R34IdPos w.c0)
    (qc1 : R34IdPos w.c1)
    (qc2 : R34IdPos w.c2)
    (qc3 : R34IdPos w.c3)
    (qc4 : R34IdPos w.c4)
    (qd4 : R34DownPos w.d4)
    (qe0 : R34IdPos w.e0)
    (qe1 : R34IdPos w.e1)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (h_stem : R34StemSmoothAt N 56 56 w.sW w.sb w.sε w.sγ w.sβ x)
    (h_pool : R34PoolSmoothAt N 56 56
      (StableHLO.cbReluStridedB N (h := 2 * 56) (w := 2 * 56) w.sW w.sb w.sε w.sγ w.sβ x))
    (sa0 : R34IdSmoothAt N 56 56 w.a0 (r34Pre0 N w x))
    (sa1 : R34IdSmoothAt N 56 56 w.a1 (r34Pre1 N w x))
    (sa2 : R34IdSmoothAt N 56 56 w.a2 (r34Pre2 N w x))
    (sd2 : R34DownSmoothAt N 28 28 w.d2 (r34Pre3 N w x))
    (sb0 : R34IdSmoothAt N 28 28 w.b0 (r34Pre4 N w x))
    (sb1 : R34IdSmoothAt N 28 28 w.b1 (r34Pre5 N w x))
    (sb2 : R34IdSmoothAt N 28 28 w.b2 (r34Pre6 N w x))
    (sd3 : R34DownSmoothAt N 14 14 w.d3 (r34Pre7 N w x))
    (sc0 : R34IdSmoothAt N 14 14 w.c0 (r34Pre8 N w x))
    (sc1 : R34IdSmoothAt N 14 14 w.c1 (r34Pre9 N w x))
    (sc2 : R34IdSmoothAt N 14 14 w.c2 (r34Pre10 N w x))
    (sc3 : R34IdSmoothAt N 14 14 w.c3 (r34Pre11 N w x))
    (sc4 : R34IdSmoothAt N 14 14 w.c4 (r34Pre12 N w x))
    (sd4 : R34DownSmoothAt N 7 7 w.d4 (r34Pre13 N w x))
    (se0 : R34IdSmoothAt N 7 7 w.e0 (r34Pre14 N w x))
    (se1 : R34IdSmoothAt N 7 7 w.e1 (r34Pre15 N w x))
    :
    HasVJPAt (r34HeadB N 7 7 w.Wd w.bd ∘ r34Pre16 N w) x := by
  have dS : DifferentiableAt ℝ (r34Pre0 N w) x :=
    r34StemB_differentiableAt N 56 56 w.sW w.sb w.sε hsε w.sγ w.sβ
      (by norm_num) (by norm_num) (by norm_num) x h_stem h_pool
  have vS : HasVJPAt (r34Pre0 N w) x :=
    r34StemB_has_vjp_at N 56 56 w.sW w.sb w.sε hsε w.sγ w.sβ
      (by norm_num) (by norm_num) (by norm_num) x h_stem h_pool
  have d1 := r34IdB_differentiableAt N 56 56 w.a0 qa0 _ sa0
  have e1 : HasVJPAt (r34Pre1 N w) x :=
    vjp_comp_at _ _ x dS d1 vS (r34IdB_has_vjp_at N 56 56 w.a0 qa0 _ sa0)
  have f1 : DifferentiableAt ℝ (r34Pre1 N w) x := d1.comp x dS
  have d2 := r34IdB_differentiableAt N 56 56 w.a1 qa1 _ sa1
  have e2 : HasVJPAt (r34Pre2 N w) x :=
    vjp_comp_at _ _ x f1 d2 e1 (r34IdB_has_vjp_at N 56 56 w.a1 qa1 _ sa1)
  have f2 : DifferentiableAt ℝ (r34Pre2 N w) x := d2.comp x f1
  have d3 := r34IdB_differentiableAt N 56 56 w.a2 qa2 _ sa2
  have e3 : HasVJPAt (r34Pre3 N w) x :=
    vjp_comp_at _ _ x f2 d3 e2 (r34IdB_has_vjp_at N 56 56 w.a2 qa2 _ sa2)
  have f3 : DifferentiableAt ℝ (r34Pre3 N w) x := d3.comp x f2
  have d4 := r34DownB_differentiableAt N 28 28 w.d2 qd2 _ sd2
  have e4 : HasVJPAt (r34Pre4 N w) x :=
    vjp_comp_at _ _ x f3 d4 e3 (r34DownB_has_vjp_at N 28 28 w.d2 qd2 _ sd2)
  have f4 : DifferentiableAt ℝ (r34Pre4 N w) x := d4.comp x f3
  have d5 := r34IdB_differentiableAt N 28 28 w.b0 qb0 _ sb0
  have e5 : HasVJPAt (r34Pre5 N w) x :=
    vjp_comp_at _ _ x f4 d5 e4 (r34IdB_has_vjp_at N 28 28 w.b0 qb0 _ sb0)
  have f5 : DifferentiableAt ℝ (r34Pre5 N w) x := d5.comp x f4
  have d6 := r34IdB_differentiableAt N 28 28 w.b1 qb1 _ sb1
  have e6 : HasVJPAt (r34Pre6 N w) x :=
    vjp_comp_at _ _ x f5 d6 e5 (r34IdB_has_vjp_at N 28 28 w.b1 qb1 _ sb1)
  have f6 : DifferentiableAt ℝ (r34Pre6 N w) x := d6.comp x f5
  have d7 := r34IdB_differentiableAt N 28 28 w.b2 qb2 _ sb2
  have e7 : HasVJPAt (r34Pre7 N w) x :=
    vjp_comp_at _ _ x f6 d7 e6 (r34IdB_has_vjp_at N 28 28 w.b2 qb2 _ sb2)
  have f7 : DifferentiableAt ℝ (r34Pre7 N w) x := d7.comp x f6
  have d8 := r34DownB_differentiableAt N 14 14 w.d3 qd3 _ sd3
  have e8 : HasVJPAt (r34Pre8 N w) x :=
    vjp_comp_at _ _ x f7 d8 e7 (r34DownB_has_vjp_at N 14 14 w.d3 qd3 _ sd3)
  have f8 : DifferentiableAt ℝ (r34Pre8 N w) x := d8.comp x f7
  have d9 := r34IdB_differentiableAt N 14 14 w.c0 qc0 _ sc0
  have e9 : HasVJPAt (r34Pre9 N w) x :=
    vjp_comp_at _ _ x f8 d9 e8 (r34IdB_has_vjp_at N 14 14 w.c0 qc0 _ sc0)
  have f9 : DifferentiableAt ℝ (r34Pre9 N w) x := d9.comp x f8
  have d10 := r34IdB_differentiableAt N 14 14 w.c1 qc1 _ sc1
  have e10 : HasVJPAt (r34Pre10 N w) x :=
    vjp_comp_at _ _ x f9 d10 e9 (r34IdB_has_vjp_at N 14 14 w.c1 qc1 _ sc1)
  have f10 : DifferentiableAt ℝ (r34Pre10 N w) x := d10.comp x f9
  have d11 := r34IdB_differentiableAt N 14 14 w.c2 qc2 _ sc2
  have e11 : HasVJPAt (r34Pre11 N w) x :=
    vjp_comp_at _ _ x f10 d11 e10 (r34IdB_has_vjp_at N 14 14 w.c2 qc2 _ sc2)
  have f11 : DifferentiableAt ℝ (r34Pre11 N w) x := d11.comp x f10
  have d12 := r34IdB_differentiableAt N 14 14 w.c3 qc3 _ sc3
  have e12 : HasVJPAt (r34Pre12 N w) x :=
    vjp_comp_at _ _ x f11 d12 e11 (r34IdB_has_vjp_at N 14 14 w.c3 qc3 _ sc3)
  have f12 : DifferentiableAt ℝ (r34Pre12 N w) x := d12.comp x f11
  have d13 := r34IdB_differentiableAt N 14 14 w.c4 qc4 _ sc4
  have e13 : HasVJPAt (r34Pre13 N w) x :=
    vjp_comp_at _ _ x f12 d13 e12 (r34IdB_has_vjp_at N 14 14 w.c4 qc4 _ sc4)
  have f13 : DifferentiableAt ℝ (r34Pre13 N w) x := d13.comp x f12
  have d14 := r34DownB_differentiableAt N 7 7 w.d4 qd4 _ sd4
  have e14 : HasVJPAt (r34Pre14 N w) x :=
    vjp_comp_at _ _ x f13 d14 e13 (r34DownB_has_vjp_at N 7 7 w.d4 qd4 _ sd4)
  have f14 : DifferentiableAt ℝ (r34Pre14 N w) x := d14.comp x f13
  have d15 := r34IdB_differentiableAt N 7 7 w.e0 qe0 _ se0
  have e15 : HasVJPAt (r34Pre15 N w) x :=
    vjp_comp_at _ _ x f14 d15 e14 (r34IdB_has_vjp_at N 7 7 w.e0 qe0 _ se0)
  have f15 : DifferentiableAt ℝ (r34Pre15 N w) x := d15.comp x f14
  have d16 := r34IdB_differentiableAt N 7 7 w.e1 qe1 _ se1
  have e16 : HasVJPAt (r34Pre16 N w) x :=
    vjp_comp_at _ _ x f15 d16 e15 (r34IdB_has_vjp_at N 7 7 w.e1 qe1 _ se1)
  have f16 : DifferentiableAt ℝ (r34Pre16 N w) x := d16.comp x f15
  exact vjp_comp_at _ (r34HeadB N 7 7 w.Wd w.bd) x f16
    ((r34HeadB_differentiable N 7 7 w.Wd w.bd) _) e16
    ((r34HeadB_has_vjp N 7 7 w.Wd w.bd).toHasVJPAt _)


-- ════════════════════════════════════════════════════════════════
-- § The chain equation — the layered `r34PreK` form IS the committed forward
--   ⚠ Peeled one layer at a time through `*_apply`. A one-step `rfl` against a sixteen-deep
--   nested application does not survive (MobileNetV2FullVJP.lean's section header records the
--   kernel deterministic timeout); `rw [<the def>, Function.comp_apply]` closes on syntactically
--   identical terms and never unfolds an inner layer.
-- ════════════════════════════════════════════════════════════════

theorem r34Pre0_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre0 N w x = r34StemB N 56 56 w.sW w.sb w.sε w.sγ w.sβ x := by
  rw [r34Pre0]
theorem r34Pre1_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre1 N w x = r34IdB N 56 56 w.a0 (r34Pre0 N w x) := by
  rw [r34Pre1, Function.comp_apply]
theorem r34Pre2_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre2 N w x = r34IdB N 56 56 w.a1 (r34Pre1 N w x) := by
  rw [r34Pre2, Function.comp_apply]
theorem r34Pre3_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre3 N w x = r34IdB N 56 56 w.a2 (r34Pre2 N w x) := by
  rw [r34Pre3, Function.comp_apply]
theorem r34Pre4_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre4 N w x = r34DownB N 28 28 w.d2 (r34Pre3 N w x) := by
  rw [r34Pre4, Function.comp_apply]
theorem r34Pre5_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre5 N w x = r34IdB N 28 28 w.b0 (r34Pre4 N w x) := by
  rw [r34Pre5, Function.comp_apply]
theorem r34Pre6_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre6 N w x = r34IdB N 28 28 w.b1 (r34Pre5 N w x) := by
  rw [r34Pre6, Function.comp_apply]
theorem r34Pre7_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre7 N w x = r34IdB N 28 28 w.b2 (r34Pre6 N w x) := by
  rw [r34Pre7, Function.comp_apply]
theorem r34Pre8_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre8 N w x = r34DownB N 14 14 w.d3 (r34Pre7 N w x) := by
  rw [r34Pre8, Function.comp_apply]
theorem r34Pre9_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre9 N w x = r34IdB N 14 14 w.c0 (r34Pre8 N w x) := by
  rw [r34Pre9, Function.comp_apply]
theorem r34Pre10_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre10 N w x = r34IdB N 14 14 w.c1 (r34Pre9 N w x) := by
  rw [r34Pre10, Function.comp_apply]
theorem r34Pre11_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre11 N w x = r34IdB N 14 14 w.c2 (r34Pre10 N w x) := by
  rw [r34Pre11, Function.comp_apply]
theorem r34Pre12_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre12 N w x = r34IdB N 14 14 w.c3 (r34Pre11 N w x) := by
  rw [r34Pre12, Function.comp_apply]
theorem r34Pre13_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre13 N w x = r34IdB N 14 14 w.c4 (r34Pre12 N w x) := by
  rw [r34Pre13, Function.comp_apply]
theorem r34Pre14_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre14 N w x = r34DownB N 7 7 w.d4 (r34Pre13 N w x) := by
  rw [r34Pre14, Function.comp_apply]
theorem r34Pre15_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre15 N w x = r34IdB N 7 7 w.e0 (r34Pre14 N w x) := by
  rw [r34Pre15, Function.comp_apply]
theorem r34Pre16_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    r34Pre16 N w x = r34IdB N 7 7 w.e1 (r34Pre15 N w x) := by
  rw [r34Pre16, Function.comp_apply]

/-- ⭐ **The committed nested-application forward IS the layered chain the VJP is stated on** —
    the batched peer of `mobilenetv2ForwardPaper_eq_chain`, and what lets the VJP be about
    `resnet34ForwardB_full` rather than about a re-spelling of it. -/
theorem resnet34ForwardB_full_eq_chain (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    resnet34ForwardB_full N w x = (r34HeadB N 7 7 w.Wd w.bd ∘ r34Pre16 N w) x := by
  rw [resnet34ForwardB_full, Function.comp_apply, r34Pre16_apply, r34Pre15_apply, r34Pre14_apply, r34Pre13_apply, r34Pre12_apply, r34Pre11_apply, r34Pre10_apply, r34Pre9_apply, r34Pre8_apply, r34Pre7_apply, r34Pre6_apply, r34Pre5_apply, r34Pre4_apply, r34Pre3_apply, r34Pre2_apply, r34Pre1_apply, r34Pre0_apply]


/-- ⭐⭐ **Public correctness theorem**: the sixteen-block batch-BN backward equals the
    `pdiv`-contracted Jacobian of `resnet34ForwardB_full` ITSELF — the committed
    nested-application forward `ResNet34FullB.lean` defines and
    `resnet34FwdGraphB_full_faithful` proves the typed graph denotes — not of the layered chain
    the VJP is assembled on. Tied back through `resnet34ForwardB_full_eq_chain`. -/
theorem resnet34ForwardB_full_has_vjp_at_correct (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hsε : 0 < w.sε)
    (qa0 : R34IdPos w.a0)
    (qa1 : R34IdPos w.a1)
    (qa2 : R34IdPos w.a2)
    (qd2 : R34DownPos w.d2)
    (qb0 : R34IdPos w.b0)
    (qb1 : R34IdPos w.b1)
    (qb2 : R34IdPos w.b2)
    (qd3 : R34DownPos w.d3)
    (qc0 : R34IdPos w.c0)
    (qc1 : R34IdPos w.c1)
    (qc2 : R34IdPos w.c2)
    (qc3 : R34IdPos w.c3)
    (qc4 : R34IdPos w.c4)
    (qd4 : R34DownPos w.d4)
    (qe0 : R34IdPos w.e0)
    (qe1 : R34IdPos w.e1)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (h_stem : R34StemSmoothAt N 56 56 w.sW w.sb w.sε w.sγ w.sβ x)
    (h_pool : R34PoolSmoothAt N 56 56
      (StableHLO.cbReluStridedB N (h := 2 * 56) (w := 2 * 56) w.sW w.sb w.sε w.sγ w.sβ x))
    (sa0 : R34IdSmoothAt N 56 56 w.a0 (r34Pre0 N w x))
    (sa1 : R34IdSmoothAt N 56 56 w.a1 (r34Pre1 N w x))
    (sa2 : R34IdSmoothAt N 56 56 w.a2 (r34Pre2 N w x))
    (sd2 : R34DownSmoothAt N 28 28 w.d2 (r34Pre3 N w x))
    (sb0 : R34IdSmoothAt N 28 28 w.b0 (r34Pre4 N w x))
    (sb1 : R34IdSmoothAt N 28 28 w.b1 (r34Pre5 N w x))
    (sb2 : R34IdSmoothAt N 28 28 w.b2 (r34Pre6 N w x))
    (sd3 : R34DownSmoothAt N 14 14 w.d3 (r34Pre7 N w x))
    (sc0 : R34IdSmoothAt N 14 14 w.c0 (r34Pre8 N w x))
    (sc1 : R34IdSmoothAt N 14 14 w.c1 (r34Pre9 N w x))
    (sc2 : R34IdSmoothAt N 14 14 w.c2 (r34Pre10 N w x))
    (sc3 : R34IdSmoothAt N 14 14 w.c3 (r34Pre11 N w x))
    (sc4 : R34IdSmoothAt N 14 14 w.c4 (r34Pre12 N w x))
    (sd4 : R34DownSmoothAt N 7 7 w.d4 (r34Pre13 N w x))
    (se0 : R34IdSmoothAt N 7 7 w.e0 (r34Pre14 N w x))
    (se1 : R34IdSmoothAt N 7 7 w.e1 (r34Pre15 N w x))
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    (resnet34ForwardB_full_has_vjp_at N w hsε qa0 qa1 qa2 qd2 qb0 qb1 qb2 qd3 qc0 qc1 qc2 qc3 qc4 qd4 qe0 qe1 x h_stem h_pool sa0 sa1 sa2 sd2 sb0 sb1 sb2 sd3 sc0 sc1 sc2 sc3 sc4 sd4 se0 se1).backward dy i =
      ∑ j : Fin (N * nCls), pdiv (resnet34ForwardB_full N w) x i j * dy j := by
  have h := (resnet34ForwardB_full_has_vjp_at N w hsε qa0 qa1 qa2 qd2 qb0 qb1 qb2 qd3 qc0 qc1 qc2 qc3 qc4 qd4 qe0 qe1 x h_stem h_pool sa0 sa1 sa2 sd2 sb0 sb1 sb2 sd3 sc0 sc1 sc2 sc3 sc4 sd4 se0 se1).correct dy i
  rwa [show resnet34ForwardB_full N w = r34HeadB N 7 7 w.Wd w.bd ∘ r34Pre16 N w
      from funext (resnet34ForwardB_full_eq_chain N w)]

end Proofs
