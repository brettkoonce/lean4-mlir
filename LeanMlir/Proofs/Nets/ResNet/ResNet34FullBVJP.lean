import LeanMlir.Proofs.Nets.ResNet.ResNet34FullB
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

⭐ The one thing that did not exist is `batchMap_has_vjp_at` ([`Foundation/BatchMapVJPAt.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Foundation/BatchMapVJPAt.lean),
written for this): r34's stem ends in `batchMap N (maxPool3s2Flat 64 56 56)` and a max-pool has no
derivative at a tie, so the GLOBAL `batchMap_has_vjp` cannot lift it. The per-example pool VJP at a
`Vec` point was already there (`maxPool3s2Flat_has_vjp_at_vec`), so the batched pool is two lines.

## The hypothesis budget

⚠ **Pointwise (`HasVJPAt`), not global, and necessarily.** relu is kinked. ⛔ And r34 carries
**two** kink clauses per block, not one: the body's mid-relu AND the post-residual outer relu.
That outer relu is ResNet's structural difference from MobileNetV2/EfficientNet, whose residual
add IS the block output. Sixteen blocks therefore carry 32 clauses, plus the stem's relu and the
stem pool's no-tie condition — bundled per block into `R34IdSmoothAt` / `R34DownSmoothAt`, and
those 18 bundles into one `R34SmoothAtB` (the positivity bundles into `R34PosB`), so the apex
binds two hypotheses, exactly as `MobileNetV2FullBVJP.lean`'s `MNV2SmoothAtB` / `MNV2PosB`.

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
    DifferentiableAt ℝ (r34IdB N h w p) v :=
  (StableHLO.r34BasicBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂).diff v ⟨⟨hs.hmid, trivial⟩, hs.hout⟩

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
    DifferentiableAt ℝ (r34DownB N h w p) v :=
  (StableHLO.r34DownBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.Wp p.bp p.εp hq.hp p.γp p.βp).diff v
    ⟨⟨trivial, hs.hmid, trivial⟩, hs.hout⟩

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
-- § The whole hypothesis budget, in two structures
-- ════════════════════════════════════════════════════════════════

/-- Every BatchNorm `ε` of the net is positive: the stem's and each block's bundle. -/
structure R34PosB {nCls : Nat} (w : R34BWeights nCls) : Prop where
  s : 0 < w.sε
  a0 : R34IdPos w.a0
  a1 : R34IdPos w.a1
  a2 : R34IdPos w.a2
  d2 : R34DownPos w.d2
  b0 : R34IdPos w.b0
  b1 : R34IdPos w.b1
  b2 : R34IdPos w.b2
  d3 : R34DownPos w.d3
  c0 : R34IdPos w.c0
  c1 : R34IdPos w.c1
  c2 : R34IdPos w.c2
  c3 : R34IdPos w.c3
  c4 : R34IdPos w.c4
  d4 : R34DownPos w.d4
  e0 : R34IdPos w.e0
  e1 : R34IdPos w.e1

/-- ⭐ **Every relu is away from its kink and the stem pool has no tie, each at the activation
    its block actually sees**: the stem's clauses at the image, block `k`'s at `r34Pre(k-1)`. The
    head has none (GAP and dense are smooth). -/
structure R34SmoothAtB (N : Nat) {nCls : Nat} (w : R34BWeights nCls) (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) : Prop where
  stem : R34StemSmoothAt N 56 56 w.sW w.sb w.sε w.sγ w.sβ x
  pool : R34PoolSmoothAt N 56 56
    (StableHLO.cbReluStridedB N (h := 2 * 56) (w := 2 * 56) w.sW w.sb w.sε w.sγ w.sβ x)
  a0 : R34IdSmoothAt N 56 56 w.a0 (r34Pre0 N w x)
  a1 : R34IdSmoothAt N 56 56 w.a1 (r34Pre1 N w x)
  a2 : R34IdSmoothAt N 56 56 w.a2 (r34Pre2 N w x)
  d2 : R34DownSmoothAt N 28 28 w.d2 (r34Pre3 N w x)
  b0 : R34IdSmoothAt N 28 28 w.b0 (r34Pre4 N w x)
  b1 : R34IdSmoothAt N 28 28 w.b1 (r34Pre5 N w x)
  b2 : R34IdSmoothAt N 28 28 w.b2 (r34Pre6 N w x)
  d3 : R34DownSmoothAt N 14 14 w.d3 (r34Pre7 N w x)
  c0 : R34IdSmoothAt N 14 14 w.c0 (r34Pre8 N w x)
  c1 : R34IdSmoothAt N 14 14 w.c1 (r34Pre9 N w x)
  c2 : R34IdSmoothAt N 14 14 w.c2 (r34Pre10 N w x)
  c3 : R34IdSmoothAt N 14 14 w.c3 (r34Pre11 N w x)
  c4 : R34IdSmoothAt N 14 14 w.c4 (r34Pre12 N w x)
  d4 : R34DownSmoothAt N 7 7 w.d4 (r34Pre13 N w x)
  e0 : R34IdSmoothAt N 7 7 w.e0 (r34Pre14 N w x)
  e1 : R34IdSmoothAt N 7 7 w.e1 (r34Pre15 N w x)

-- ════════════════════════════════════════════════════════════════
-- § The apex
-- ════════════════════════════════════════════════════════════════

/-- The chain's VJP and its differentiability together, one `vjp_comp_diff_at` per block: the
    apex is `.fst`, `resnet34ForwardB_full_differentiableAt` is `.snd`. -/
private noncomputable def r34ChainB (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hq : R34PosB w) (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (hx : R34SmoothAtB N w x) :
    PProd (HasVJPAt (r34HeadB N 7 7 w.Wd w.bd ∘ r34Pre16 N w) x)
      (DifferentiableAt ℝ (r34HeadB N 7 7 w.Wd w.bd ∘ r34Pre16 N w) x) :=
  let p0 : PProd (HasVJPAt (r34Pre0 N w) x) (DifferentiableAt ℝ (r34Pre0 N w) x) :=
    ⟨r34StemB_has_vjp_at N 56 56 w.sW w.sb w.sε hq.s w.sγ w.sβ
        (by norm_num) (by norm_num) (by norm_num) x hx.stem hx.pool,
      r34StemB_differentiableAt N 56 56 w.sW w.sb w.sε hq.s w.sγ w.sβ
        (by norm_num) (by norm_num) (by norm_num) x hx.stem hx.pool⟩
  let p1 : PProd (HasVJPAt (r34Pre1 N w) x) (DifferentiableAt ℝ (r34Pre1 N w) x) :=
    vjp_comp_diff_at _ _ x p0 ⟨r34IdB_has_vjp_at N 56 56 w.a0 hq.a0 _ hx.a0,
      r34IdB_differentiableAt N 56 56 w.a0 hq.a0 _ hx.a0⟩
  let p2 : PProd (HasVJPAt (r34Pre2 N w) x) (DifferentiableAt ℝ (r34Pre2 N w) x) :=
    vjp_comp_diff_at _ _ x p1 ⟨r34IdB_has_vjp_at N 56 56 w.a1 hq.a1 _ hx.a1,
      r34IdB_differentiableAt N 56 56 w.a1 hq.a1 _ hx.a1⟩
  let p3 : PProd (HasVJPAt (r34Pre3 N w) x) (DifferentiableAt ℝ (r34Pre3 N w) x) :=
    vjp_comp_diff_at _ _ x p2 ⟨r34IdB_has_vjp_at N 56 56 w.a2 hq.a2 _ hx.a2,
      r34IdB_differentiableAt N 56 56 w.a2 hq.a2 _ hx.a2⟩
  let p4 : PProd (HasVJPAt (r34Pre4 N w) x) (DifferentiableAt ℝ (r34Pre4 N w) x) :=
    vjp_comp_diff_at _ _ x p3 ⟨r34DownB_has_vjp_at N 28 28 w.d2 hq.d2 _ hx.d2,
      r34DownB_differentiableAt N 28 28 w.d2 hq.d2 _ hx.d2⟩
  let p5 : PProd (HasVJPAt (r34Pre5 N w) x) (DifferentiableAt ℝ (r34Pre5 N w) x) :=
    vjp_comp_diff_at _ _ x p4 ⟨r34IdB_has_vjp_at N 28 28 w.b0 hq.b0 _ hx.b0,
      r34IdB_differentiableAt N 28 28 w.b0 hq.b0 _ hx.b0⟩
  let p6 : PProd (HasVJPAt (r34Pre6 N w) x) (DifferentiableAt ℝ (r34Pre6 N w) x) :=
    vjp_comp_diff_at _ _ x p5 ⟨r34IdB_has_vjp_at N 28 28 w.b1 hq.b1 _ hx.b1,
      r34IdB_differentiableAt N 28 28 w.b1 hq.b1 _ hx.b1⟩
  let p7 : PProd (HasVJPAt (r34Pre7 N w) x) (DifferentiableAt ℝ (r34Pre7 N w) x) :=
    vjp_comp_diff_at _ _ x p6 ⟨r34IdB_has_vjp_at N 28 28 w.b2 hq.b2 _ hx.b2,
      r34IdB_differentiableAt N 28 28 w.b2 hq.b2 _ hx.b2⟩
  let p8 : PProd (HasVJPAt (r34Pre8 N w) x) (DifferentiableAt ℝ (r34Pre8 N w) x) :=
    vjp_comp_diff_at _ _ x p7 ⟨r34DownB_has_vjp_at N 14 14 w.d3 hq.d3 _ hx.d3,
      r34DownB_differentiableAt N 14 14 w.d3 hq.d3 _ hx.d3⟩
  let p9 : PProd (HasVJPAt (r34Pre9 N w) x) (DifferentiableAt ℝ (r34Pre9 N w) x) :=
    vjp_comp_diff_at _ _ x p8 ⟨r34IdB_has_vjp_at N 14 14 w.c0 hq.c0 _ hx.c0,
      r34IdB_differentiableAt N 14 14 w.c0 hq.c0 _ hx.c0⟩
  let p10 : PProd (HasVJPAt (r34Pre10 N w) x) (DifferentiableAt ℝ (r34Pre10 N w) x) :=
    vjp_comp_diff_at _ _ x p9 ⟨r34IdB_has_vjp_at N 14 14 w.c1 hq.c1 _ hx.c1,
      r34IdB_differentiableAt N 14 14 w.c1 hq.c1 _ hx.c1⟩
  let p11 : PProd (HasVJPAt (r34Pre11 N w) x) (DifferentiableAt ℝ (r34Pre11 N w) x) :=
    vjp_comp_diff_at _ _ x p10 ⟨r34IdB_has_vjp_at N 14 14 w.c2 hq.c2 _ hx.c2,
      r34IdB_differentiableAt N 14 14 w.c2 hq.c2 _ hx.c2⟩
  let p12 : PProd (HasVJPAt (r34Pre12 N w) x) (DifferentiableAt ℝ (r34Pre12 N w) x) :=
    vjp_comp_diff_at _ _ x p11 ⟨r34IdB_has_vjp_at N 14 14 w.c3 hq.c3 _ hx.c3,
      r34IdB_differentiableAt N 14 14 w.c3 hq.c3 _ hx.c3⟩
  let p13 : PProd (HasVJPAt (r34Pre13 N w) x) (DifferentiableAt ℝ (r34Pre13 N w) x) :=
    vjp_comp_diff_at _ _ x p12 ⟨r34IdB_has_vjp_at N 14 14 w.c4 hq.c4 _ hx.c4,
      r34IdB_differentiableAt N 14 14 w.c4 hq.c4 _ hx.c4⟩
  let p14 : PProd (HasVJPAt (r34Pre14 N w) x) (DifferentiableAt ℝ (r34Pre14 N w) x) :=
    vjp_comp_diff_at _ _ x p13 ⟨r34DownB_has_vjp_at N 7 7 w.d4 hq.d4 _ hx.d4,
      r34DownB_differentiableAt N 7 7 w.d4 hq.d4 _ hx.d4⟩
  let p15 : PProd (HasVJPAt (r34Pre15 N w) x) (DifferentiableAt ℝ (r34Pre15 N w) x) :=
    vjp_comp_diff_at _ _ x p14 ⟨r34IdB_has_vjp_at N 7 7 w.e0 hq.e0 _ hx.e0,
      r34IdB_differentiableAt N 7 7 w.e0 hq.e0 _ hx.e0⟩
  let p16 : PProd (HasVJPAt (r34Pre16 N w) x) (DifferentiableAt ℝ (r34Pre16 N w) x) :=
    vjp_comp_diff_at _ _ x p15 ⟨r34IdB_has_vjp_at N 7 7 w.e1 hq.e1 _ hx.e1,
      r34IdB_differentiableAt N 7 7 w.e1 hq.e1 _ hx.e1⟩
  vjp_comp_diff_at _ _ x p16
    ⟨(r34HeadB_has_vjp N 7 7 w.Wd w.bd).toHasVJPAt _, r34HeadB_differentiable N 7 7 w.Wd w.bd _⟩

/-- ⭐⭐ **ResNet-34 at TRUE BATCH-NORM has a certified input-VJP at a smooth point — all sixteen
    basic blocks.** Chains stem → the [3,4,6,3] ladder → head with `vjp_comp_diff_at` under two
    hypotheses: `R34PosB` (every `ε > 0`) and `R34SmoothAtB` (every relu clause and the pool's
    no-tie, each at its block's own input). T1's VJP half for `formalization.yaml` 4e's port.

    ⚠ Pointwise, and necessarily: relu is kinked. ⛔ Each block contributes TWO clauses — the
    body's mid-relu and the post-residual OUTER relu — where MobileNetV2's bottleneck contributes
    two relu6 clauses and EfficientNet's MBConv contributes none.

    ⭐ The head takes no hypothesis at all (GAP and dense are smooth, and each is `batchMap` of a
    per-example op), and `N` is a variable: this tier carries no numerals. -/
noncomputable def resnet34ForwardB_full_has_vjp_at (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hq : R34PosB w) (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (hx : R34SmoothAtB N w x) :
    HasVJPAt (r34HeadB N 7 7 w.Wd w.bd ∘ r34Pre16 N w) x :=
  (r34ChainB N w hq x hx).fst

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
    the r34 peer of `mobilenetv2ForwardB_full_eq_chain`, and what lets the VJP be about
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
    (hq : R34PosB w) (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (hx : R34SmoothAtB N w x)
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    (resnet34ForwardB_full_has_vjp_at N w hq x hx).backward dy i =
      ∑ j : Fin (N * nCls), pdiv (resnet34ForwardB_full N w) x i j * dy j := by
  have h := (resnet34ForwardB_full_has_vjp_at N w hq x hx).correct dy i
  rwa [show resnet34ForwardB_full N w = r34HeadB N 7 7 w.Wd w.bd ∘ r34Pre16 N w
      from funext (resnet34ForwardB_full_eq_chain N w)]

/-- ⭐ The committed forward is differentiable at every smooth point — the chain's `.snd`, read
    back through `resnet34ForwardB_full_eq_chain`. What the seal's `sealDiffAt` needs. -/
theorem resnet34ForwardB_full_differentiableAt (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hq : R34PosB w) (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (hx : R34SmoothAtB N w x) :
    DifferentiableAt ℝ (resnet34ForwardB_full N w) x := by
  rw [show resnet34ForwardB_full N w = r34HeadB N 7 7 w.Wd w.bd ∘ r34Pre16 N w
      from funext (resnet34ForwardB_full_eq_chain N w)]
  exact (r34ChainB N w hq x hx).snd

end Proofs
