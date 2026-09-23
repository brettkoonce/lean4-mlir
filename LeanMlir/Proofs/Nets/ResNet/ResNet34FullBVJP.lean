import LeanMlir.Proofs.Nets.ResNet.ResNet34FullB
import LeanMlir.Proofs.Foundation.BatchMapVJPAt
import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Foundation.HeadLayers
import LeanMlir.Proofs.Nets.ResNet.ResNetBackChains

/-! # ResNet-34's whole-net input-VJP at TRUE BATCH-NORM (T1, the VJP half)

`ResNet34FullB.lean` states the batch-BN forward and its typed graph. This file gives that forward
a certified `HasVJPAt` at the paper depth — the batched peer of what `MobileNetV2FullVJP.lean` does
for MobileNetV2's seventeen bottlenecks, and the last piece of T1 in `formalization.yaml` 4e's port.

## No new mathematics, and one new lemma one tier down

Every block VJP is already proven at `bnBatchLA`: `r34BasicBlockB_has_vjp_at` and
`r34DownBlockB_has_vjp_at` (`ResNet34BackB0.lean`) are exactly the two block shapes
`ResNet34FullB.lean`'s `r34IdB` / `r34DownB` unfold to. The bundle lemmas below are delegations
in the `EfficientNetFullB0` style, and the whole net is those blocks' `CertLayer`s composed.

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

/-- Downsample basic block VJP — `r34DownBlockB_has_vjp_at` at the bundle's fields. -/
noncomputable def r34DownB_has_vjp_at (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (hq : R34DownPos p) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : R34DownSmoothAt N h w p v) :
    HasVJPAt (r34DownB N h w p) v :=
  StableHLO.r34DownBlockB_has_vjp_at N p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.Wp p.bp p.εp hq.hp p.γp p.βp v hs.hmid hs.hout

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


/-- The identity basic block as a `CertLayer`, at its weight record. -/
noncomputable def r34IdLayer (N h w : Nat) {c : Nat} (p : R34IdW c) (hq : R34IdPos p) :
    StableHLO.CertLayer (N * (c * h * w)) (N * (c * h * w)) :=
  StableHLO.r34BasicBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂

/-- The downsample basic block as a `CertLayer`, at its weight record. -/
noncomputable def r34DownLayer (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (hq : R34DownPos p) : StableHLO.CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  StableHLO.r34DownBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.Wp p.bp p.εp hq.hp p.γp p.βp

theorem r34IdLayer_fwd (N h w : Nat) {c : Nat} (p : R34IdW c) (hq : R34IdPos p) :
    (r34IdLayer N h w p hq).fwd = r34IdB N h w p := rfl
theorem r34DownLayer_fwd (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc) (hq : R34DownPos p) :
    (r34DownLayer N h w p hq).fwd = r34DownB N h w p := rfl

-- ════════════════════════════════════════════════════════════════
-- § Stem and head as `CertLayer`s — shared with ResNet-50, whose stem and head are these
-- ════════════════════════════════════════════════════════════════

/-- The batched 3×3/s2 stem pool as a `CertLayer`, certified where no example's window ties. Its
    backward graph is the render's `maxPool3s2BackB`, and it denotes `batchMap_has_vjp_at`'s
    backward definitionally once the two spellings of the scatter are identified. -/
noncomputable def r34PoolLayer (N : Nat) {c h w : Nat} (hc : 0 < c) (hh : 0 < h) (hw : 0 < w) :
    StableHLO.CertLayer (N * (c * (2 * h) * (2 * w))) (N * (c * h * w)) where
  fwd := StableHLO.batchMap N (maxPool3s2Flat c h w)
  ok := R34PoolSmoothAt N h w
  diff := fun v hv => batchMap_differentiableAt _ _
    (fun r => maxPool3s2Flat_differentiableAt_vec _ (hv r) hc hh hw)
  vjp := fun v hv => batchMap_has_vjp_at _ _ (fun r => maxPool3s2Flat_has_vjp_at_vec _ (hv r))
    (fun r => maxPool3s2Flat_differentiableAt_vec _ (hv r) hc hh hw)
  graph := fun v e => .maxPool3s2BackB "%stemR" v e
  faithful := fun v _ e => by rw [den_maxPool3s2BackB_eq_flatBackB]; rfl

/-- The stem, 7×7/s2 conv-bn-relu then the 3×3/s2 pool, as a `CertLayer`. Its `ok` is exactly
    `R34StemSmoothAt ∧ R34PoolSmoothAt` at the conv's output. -/
noncomputable def r34StemLayer (N h w : Nat) {ic oc : Nat} (hc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc) :
    StableHLO.CertLayer (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))) (N * (oc * h * w)) :=
  (StableHLO.cbReluStridedLayer N (h := 2 * h) (w := 2 * w) Ws bs εs hεs γs βs).comp
    (r34PoolLayer N hc hh hw)

theorem r34StemLayer_fwd (N h w : Nat) {ic oc : Nat} (hc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc) :
    (r34StemLayer N h w hc hh hw Ws bs εs hεs γs βs).fwd = r34StemB N h w Ws bs εs γs βs := rfl

/-- GAP then the classifier, as a `CertLayer`. Globally certified. -/
noncomputable def r34HeadLayer (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls) (bd : Vec nCls) :
    StableHLO.CertLayer (N * (c * h * w)) (N * nCls) :=
  (StableHLO.gapLayer N (c := c) (h := h) (w := w)).comp (StableHLO.denseLayer N Wd bd)

theorem r34HeadLayer_fwd (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls) (bd : Vec nCls) :
    (r34HeadLayer N h w Wd bd).fwd = r34HeadB N h w Wd bd := rfl

/-- `comp`'s `ok`, with the intermediate activation NAMED. A whole-net chain whose hypotheses are
    stated at named prefixes proves its `.ok` one `refine` per block through this, so the goal
    stays at the prefix; one anonymous constructor for the whole chain instead makes every
    prefix a defeq check against the nested layer forwards, and that exceeds `maxRecDepth` by the
    fifth block. ⚠ Discharge `hy` by `rw` at literal widths, not `rfl` (`r34SmoothAtB_ok`). -/
theorem StableHLO.CertLayer.comp_ok_of {m n k : Nat} {L₁ : StableHLO.CertLayer m n}
    {L₂ : StableHLO.CertLayer n k} {x : Vec m} (h₁ : L₁.ok x) (y : Vec n) (hy : L₁.fwd x = y)
    (h₂ : L₂.ok y) : (L₁.comp L₂).ok x :=
  ⟨h₁, hy ▸ h₂⟩

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


-- ════════════════════════════════════════════════════════════════
-- § The apex — the whole net as ONE `CertLayer`
--   Stem, the sixteen basic blocks and the head composed with `CertLayer.comp`, so the VJP, its
--   differentiability and the backward-graph faithfulness are the layer's fields. ⚠ At literal
--   widths each stride join meets `64 * 56 * 56` against `64 * (2 * 28) * (2 * 28)`; they unify
--   by `Nat` literal arithmetic, and the peel below never has to (`planning/certlayer_nets.md` §6).
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **ResNet-34 as one certified layer**, at every batch size. Its `.faithful` is a whole-net
    backward graph, stem pool included, proven to denote the VJP. -/
noncomputable def r34NetLayer (N : Nat) {nCls : Nat} (w : R34BWeights nCls) (hq : R34PosB w) :
    StableHLO.CertLayer (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))) (N * nCls) :=
  (r34StemLayer N 56 56 (by norm_num) (by norm_num) (by norm_num)
      w.sW w.sb w.sε hq.s w.sγ w.sβ).comp <|
  (r34IdLayer N 56 56 w.a0 hq.a0).comp <|
  (r34IdLayer N 56 56 w.a1 hq.a1).comp <|
  (r34IdLayer N 56 56 w.a2 hq.a2).comp <|
  (r34DownLayer N 28 28 w.d2 hq.d2).comp <|
  (r34IdLayer N 28 28 w.b0 hq.b0).comp <|
  (r34IdLayer N 28 28 w.b1 hq.b1).comp <|
  (r34IdLayer N 28 28 w.b2 hq.b2).comp <|
  (r34DownLayer N 14 14 w.d3 hq.d3).comp <|
  (r34IdLayer N 14 14 w.c0 hq.c0).comp <|
  (r34IdLayer N 14 14 w.c1 hq.c1).comp <|
  (r34IdLayer N 14 14 w.c2 hq.c2).comp <|
  (r34IdLayer N 14 14 w.c3 hq.c3).comp <|
  (r34IdLayer N 14 14 w.c4 hq.c4).comp <|
  (r34DownLayer N 7 7 w.d4 hq.d4).comp <|
  (r34IdLayer N 7 7 w.e0 hq.e0).comp <|
  (r34IdLayer N 7 7 w.e1 hq.e1).comp <|
  r34HeadLayer N 7 7 w.Wd w.bd

/-- The composed layer's forward IS the committed nested-application forward: peeled one `comp`
    at a time, then each layer's forward by its `rfl` lemma. -/
theorem r34NetLayer_fwd_apply (N : Nat) {nCls : Nat} (w : R34BWeights nCls) (hq : R34PosB w)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) :
    (r34NetLayer N w hq).fwd x = resnet34ForwardB_full N w x := by
  rw [r34NetLayer]
  repeat rw [StableHLO.CertLayer.comp_fwd_apply]
  rw [r34StemLayer_fwd, r34IdLayer_fwd, r34IdLayer_fwd, r34IdLayer_fwd, r34DownLayer_fwd, r34IdLayer_fwd, r34IdLayer_fwd, r34IdLayer_fwd, r34DownLayer_fwd, r34IdLayer_fwd, r34IdLayer_fwd, r34IdLayer_fwd, r34IdLayer_fwd, r34IdLayer_fwd, r34DownLayer_fwd, r34IdLayer_fwd, r34IdLayer_fwd,
    r34HeadLayer_fwd, resnet34ForwardB_full]

/-- `R34SmoothAtB` is the layer's `.ok`: one `comp_ok_of` per block, each naming its block's
    input `r34PreK`. ⚠ At the literal widths the activation step must be a `rw`: the same step
    by `rfl` (fine at ResNet-50's binder `q`) is a kernel deep recursion already at block 1. -/
theorem r34SmoothAtB_ok (N : Nat) {nCls : Nat} (w : R34BWeights nCls) (hq : R34PosB w)
    (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (hx : R34SmoothAtB N w x) : (r34NetLayer N w hq).ok x := by
  refine StableHLO.CertLayer.comp_ok_of ⟨hx.stem, hx.pool⟩ (r34Pre0 N w x)
    (by rw [r34StemLayer_fwd, r34Pre0_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.a0.hmid, trivial⟩, hx.a0.hout⟩ (r34Pre1 N w x)
    (by rw [r34IdLayer_fwd, r34Pre1_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.a1.hmid, trivial⟩, hx.a1.hout⟩ (r34Pre2 N w x)
    (by rw [r34IdLayer_fwd, r34Pre2_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.a2.hmid, trivial⟩, hx.a2.hout⟩ (r34Pre3 N w x)
    (by rw [r34IdLayer_fwd, r34Pre3_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨trivial, hx.d2.hmid, trivial⟩, hx.d2.hout⟩ (r34Pre4 N w x)
    (by rw [r34DownLayer_fwd, r34Pre4_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.b0.hmid, trivial⟩, hx.b0.hout⟩ (r34Pre5 N w x)
    (by rw [r34IdLayer_fwd, r34Pre5_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.b1.hmid, trivial⟩, hx.b1.hout⟩ (r34Pre6 N w x)
    (by rw [r34IdLayer_fwd, r34Pre6_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.b2.hmid, trivial⟩, hx.b2.hout⟩ (r34Pre7 N w x)
    (by rw [r34IdLayer_fwd, r34Pre7_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨trivial, hx.d3.hmid, trivial⟩, hx.d3.hout⟩ (r34Pre8 N w x)
    (by rw [r34DownLayer_fwd, r34Pre8_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.c0.hmid, trivial⟩, hx.c0.hout⟩ (r34Pre9 N w x)
    (by rw [r34IdLayer_fwd, r34Pre9_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.c1.hmid, trivial⟩, hx.c1.hout⟩ (r34Pre10 N w x)
    (by rw [r34IdLayer_fwd, r34Pre10_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.c2.hmid, trivial⟩, hx.c2.hout⟩ (r34Pre11 N w x)
    (by rw [r34IdLayer_fwd, r34Pre11_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.c3.hmid, trivial⟩, hx.c3.hout⟩ (r34Pre12 N w x)
    (by rw [r34IdLayer_fwd, r34Pre12_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.c4.hmid, trivial⟩, hx.c4.hout⟩ (r34Pre13 N w x)
    (by rw [r34IdLayer_fwd, r34Pre13_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨trivial, hx.d4.hmid, trivial⟩, hx.d4.hout⟩ (r34Pre14 N w x)
    (by rw [r34DownLayer_fwd, r34Pre14_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.e0.hmid, trivial⟩, hx.e0.hout⟩ (r34Pre15 N w x)
    (by rw [r34IdLayer_fwd, r34Pre15_apply]) ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨hx.e1.hmid, trivial⟩, hx.e1.hout⟩ (r34Pre16 N w x)
    (by rw [r34IdLayer_fwd, r34Pre16_apply]) ?_
  exact ⟨trivial, trivial⟩

/-- ⭐⭐ **ResNet-34 at TRUE BATCH-NORM has a certified input-VJP at a smooth point — all sixteen
    basic blocks.** `r34NetLayer`'s `.vjp`, read at the layered chain, under two hypotheses: `R34PosB` (every `ε > 0`) and `R34SmoothAtB` (every relu clause and the pool's
    no-tie, each at its block's own input). T1's VJP half for `formalization.yaml` 4e's port.

    ⚠ Pointwise, and necessarily: relu is kinked. ⛔ Each block contributes TWO clauses — the
    body's mid-relu and the post-residual OUTER relu — where MobileNetV2's bottleneck contributes
    two relu6 clauses and EfficientNet's MBConv contributes none.

    ⭐ The head takes no hypothesis at all (GAP and dense are smooth, and each is `batchMap` of a
    per-example op), and `N` is a variable: this tier carries no numerals. -/
noncomputable def resnet34ForwardB_full_has_vjp_at (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hq : R34PosB w) (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (hx : R34SmoothAtB N w x) :
    HasVJPAt (r34HeadB N 7 7 w.Wd w.bd ∘ r34Pre16 N w) x :=
  (funext fun v => (r34NetLayer_fwd_apply N w hq v).trans (resnet34ForwardB_full_eq_chain N w v)
    : (r34NetLayer N w hq).fwd = _) ▸ (r34NetLayer N w hq).vjp x (r34SmoothAtB_ok N w hq x hx)

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

/-- ⭐ The committed forward is differentiable at every smooth point — the layer's `.diff`. What
    the seal's `sealDiffAt` needs. -/
theorem resnet34ForwardB_full_differentiableAt (N : Nat) {nCls : Nat} (w : R34BWeights nCls)
    (hq : R34PosB w) (x : Vec (N * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (hx : R34SmoothAtB N w x) :
    DifferentiableAt ℝ (resnet34ForwardB_full N w) x := by
  rw [← funext (r34NetLayer_fwd_apply N w hq)]
  exact (r34NetLayer N w hq).diff x (r34SmoothAtB_ok N w hq x hx)

end Proofs
