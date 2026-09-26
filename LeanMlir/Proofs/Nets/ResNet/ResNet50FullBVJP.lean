import LeanMlir.Proofs.Nets.ResNet.ResNet50FullB
import LeanMlir.Proofs.Nets.ResNet.ResNet34FullBVJP
import LeanMlir.Proofs.Nets.ResNet.ResNet50BackB0

/-! # ResNet-50's whole-net input-VJP at TRUE BATCH-NORM

`ResNet50FullB.lean` states the batch-BN forward at the [3,4,6,3] bottleneck ladder. This file
gives that forward a certified `HasVJPAt`.

## No new mathematics, and nothing new one tier down

Every block VJP is already proven at `bnBatchLA`: `r50BottleneckBHasVJPAt`,
`r50ProjBlockBHasVJPAt` and `r50DownBlockBHasVJPAt` (`ResNet50BackB0.lean`) are exactly the
three shapes `r50IdB` / `r50ProjB` / `r50DownB` unfold to. The bundle lemmas below are delegations
in `ResNet34FullBVJP.lean`'s style, and the whole net is those blocks' `CertLayer`s composed.

**No `Foundation` lemma of its own.** ResNet-50 has ResNet-34's stem and reuses
`r34StemBHasVJPAt` (which lifts the stem pool with `batchMapHasVJPAt`) verbatim. The head is
ResNet-34's too (`r34HeadBHasVJP`, GLOBAL: GAP and dense are smooth and each is `batchMap` of a
per-example op, so no hypothesis appears).

## The hypothesis budget

**Pointwise (`HasVJPAt`), not global, and necessarily.** relu is kinked. And a BOTTLENECK
carries **THREE** kink clauses, where ResNet-34's basic block carries two: the two interior relus
and the post-residual OUTER relu. Sixteen blocks give 48 clauses, plus the stem's relu and the stem
pool's no-tie condition — bundled per block into `R50IdSmoothAt` / `R50ProjSmoothAt` /
`R50DownSmoothAt`, and those 18 bundles into one `R50SmoothAtB` (the positivity bundles into
`R50PosB`), so the apex binds two hypotheses beside `0 < q`, as MobileNetV2's does.

The pool's condition is **per example** (`R34PoolSmoothAt`, reused): a tie is a property of one
image's 3×3 window, not of the batch.

**`0 < q` is a real hypothesis here, where ResNet-34 needed none.** r34's ladder is at literals,
so `0 < 56` closes by `norm_num`; R50's is at the binder `q`, and the stem pool's VJP needs its
output grid nonempty. At `q = 0` the net is degenerate and the statement says so.

The running activations are named `r50Pre0 … r50Pre16` so each bundle can be STATED at the
activation entering its block without a sixteen-deep nested application inline; `r50Pre16` doubles
as the trunk, and `resnet50ForwardBFull_eq_chain` bridges it back to the committed
nested-application forward.

`N` and `q` are both variables (the widths are literals), so ONE statement covers
`resnet50in_fwd` (`q = 7`, 224 px) and `resnet50in160_fwd` (`q = 5`, 160 px) at every batch size.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-block hypothesis bundles
-- ════════════════════════════════════════════════════════════════

/-- All three BatchNorm epsilons of an identity bottleneck are positive. -/
structure R50IdPos {mid oc : Nat} (p : R50IdW mid oc) : Prop where
  h1 : 0 < p.ε₁
  h2 : 0 < p.ε₂
  h3 : 0 < p.ε₃

/-- All four BatchNorm epsilons of a projection bottleneck are positive (body three, skip one).
    One bundle for BOTH projection forms, as `R50ProjW` is one record for both. -/
structure R50ProjPos {ic mid oc : Nat} (p : R50ProjW ic mid oc) : Prop where
  h1 : 0 < p.ε₁
  h2 : 0 < p.ε₂
  h3 : 0 < p.ε₃
  hp : 0 < p.εp

/-- The THREE relu sites of an identity bottleneck are away from the kink at `v`: the 1×1 reduce's
    BN output, the 3×3's BN output, and the post-residual sum. The 1×1 expand has no activation. -/
structure R50IdSmoothAt (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (v : Vec (N * (oc * h * w))) : Prop where
  hm1 : ∀ k, StableHLO.bnBatchLA N mid h w p.ε₁ p.γ₁ p.β₁
    (StableHLO.batchMap N (flatConv p.W₁ p.b₁) v) k ≠ 0
  hm2 : ∀ k, StableHLO.bnBatchLA N mid h w p.ε₂ p.γ₂ p.β₂
    (StableHLO.batchMap N (flatConv p.W₂ p.b₂)
      (StableHLO.cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)) k ≠ 0
  hout : ∀ k, residual (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
    StableHLO.cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
    StableHLO.cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v k ≠ 0

/-- The three relu sites of the STRIDE-1 projection bottleneck at `v`. Identical to the identity
    block's except that the outer relu sits at the PROJECTED residual — both paths nontrivial. -/
structure R50ProjSmoothAt (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (v : Vec (N * (ic * h * w))) : Prop where
  hm1 : ∀ k, StableHLO.bnBatchLA N mid h w p.ε₁ p.γ₁ p.β₁
    (StableHLO.batchMap N (flatConv p.W₁ p.b₁) v) k ≠ 0
  hm2 : ∀ k, StableHLO.bnBatchLA N mid h w p.ε₂ p.γ₂ p.β₂
    (StableHLO.batchMap N (flatConv p.W₂ p.b₂)
      (StableHLO.cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)) k ≠ 0
  hout : ∀ k, residualProj (projB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      StableHLO.cbReluB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      StableHLO.cbReluB N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v k ≠ 0

/-- The three relu sites of the STRIDED projection bottleneck at `v`.

    v1.5: the FIRST relu is at the input resolution `2h × 2w` and only the second is at
    `h × w`, because the stride is on the 3×3. Writing `hm1` at `h w` typechecks nowhere, which is
    the one place a reader can get this shape wrong. -/
structure R50DownSmoothAt (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Prop where
  hm1 : ∀ k, StableHLO.bnBatchLA N mid (2 * h) (2 * w) p.ε₁ p.γ₁ p.β₁
    (StableHLO.batchMap N (flatConv p.W₁ p.b₁) v) k ≠ 0
  hm2 : ∀ k, StableHLO.bnBatchLA N mid h w p.ε₂ p.γ₂ p.β₂
    (StableHLO.batchMap N (flatConvStride2 p.W₂ p.b₂)
      (StableHLO.cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ v)) k ≠ 0
  hout : ∀ k, residualProj (StableHLO.projStridedB N (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
    (projB N (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
      StableHLO.cbReluStridedB N (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      StableHLO.cbReluB N (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) v k ≠ 0

-- ════════════════════════════════════════════════════════════════
-- § Block bundle lemmas (delegation)
--   The stem and head are ResNet-34's — `r34StemBHasVJPAt` and `r34HeadBHasVJP` apply at
--   R50's widths with no restatement, so nothing appears here for them.
-- ════════════════════════════════════════════════════════════════

/-- Identity bottleneck VJP — `r50BottleneckBHasVJPAt` at the bundle's fields. -/
noncomputable def r50IdBHasVJPAt (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
    (hq : R50IdPos p) (v : Vec (N * (oc * h * w))) (hs : R50IdSmoothAt N h w p v) :
    HasVJPAt (r50IdB N h w p) v :=
  StableHLO.r50BottleneckBHasVJPAt N p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃ v hs.hm1 hs.hm2 hs.hout

/-- Stride-1 projection bottleneck VJP — `r50ProjBlockBHasVJPAt` at the bundle's fields. -/
noncomputable def r50ProjBHasVJPAt (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (v : Vec (N * (ic * h * w))) (hs : R50ProjSmoothAt N h w p v) :
    HasVJPAt (r50ProjB N h w p) v :=
  StableHLO.r50ProjBlockBHasVJPAt N p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃
    p.Wp p.bp p.εp hq.hp p.γp p.βp v hs.hm1 hs.hm2 hs.hout

/-- Strided projection bottleneck VJP — `r50DownBlockBHasVJPAt` at the bundle's fields. -/
noncomputable def r50DownBHasVJPAt (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : R50DownSmoothAt N h w p v) :
    HasVJPAt (r50DownB N h w p) v :=
  StableHLO.r50DownBlockBHasVJPAt N p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃
    p.Wp p.bp p.εp hq.hp p.γp p.βp v hs.hm1 hs.hm2 hs.hout

/-- The identity bottleneck as a `CertLayer`, at its weight record. -/
noncomputable def r50IdLayer (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc) (hq : R50IdPos p) :
    StableHLO.CertLayer (N * (oc * h * w)) (N * (oc * h * w)) :=
  StableHLO.r50BottleneckLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃

/-- The stride-1 projection bottleneck as a `CertLayer`, at its weight record. -/
noncomputable def r50ProjLayer (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) : StableHLO.CertLayer (N * (ic * h * w)) (N * (oc * h * w)) :=
  StableHLO.r50ProjBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃ p.Wp p.bp p.εp hq.hp p.γp p.βp

/-- The strided projection bottleneck as a `CertLayer`, at its weight record. -/
noncomputable def r50DownLayer (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) : StableHLO.CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  StableHLO.r50DownBlockLayer N (h := h) (w := w) p.W₁ p.b₁ p.ε₁ hq.h1 p.γ₁ p.β₁
    p.W₂ p.b₂ p.ε₂ hq.h2 p.γ₂ p.β₂ p.W₃ p.b₃ p.ε₃ hq.h3 p.γ₃ p.β₃ p.Wp p.bp p.εp hq.hp p.γp p.βp

theorem r50IdLayer_fwd (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc) (hq : R50IdPos p) :
    (r50IdLayer N h w p hq).fwd = r50IdB N h w p := rfl
theorem r50ProjLayer_fwd (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) : (r50ProjLayer N h w p hq).fwd = r50ProjB N h w p := rfl
theorem r50DownLayer_fwd (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
    (hq : R50ProjPos p) : (r50DownLayer N h w p hq).fwd = r50DownB N h w p := rfl

-- ════════════════════════════════════════════════════════════════
-- § The running activations — `r50PreK` = the net truncated after block `K`
--   `r50Pre0` is the stem (7×7/s2 conv-bn-relu + 3×3/s2 pool); each later one is one `∘` deeper,
--   and `r50Pre16` is the whole [3,4,6,3] bottleneck trunk.
-- ════════════════════════════════════════════════════════════════

noncomputable def r50Pre0 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (64 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) :=
  r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.sW w.sb w.sε w.sγ w.sβ
noncomputable def r50Pre1 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) :=
  r50ProjB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 ∘ r50Pre0 N q w
noncomputable def r50Pre2 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) :=
  r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 ∘ r50Pre1 N q w
noncomputable def r50Pre3 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) :=
  r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 ∘ r50Pre2 N q w
noncomputable def r50Pre4 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) :=
  r50DownB N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 ∘ r50Pre3 N q w
noncomputable def r50Pre5 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) :=
  r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 ∘ r50Pre4 N q w
noncomputable def r50Pre6 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) :=
  r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 ∘ r50Pre5 N q w
noncomputable def r50Pre7 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) :=
  r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 ∘ r50Pre6 N q w
noncomputable def r50Pre8 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50DownB N (2 * q) (2 * q) w.s3b0 ∘ r50Pre7 N q w
noncomputable def r50Pre9 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdB N (2 * q) (2 * q) w.s3b1 ∘ r50Pre8 N q w
noncomputable def r50Pre10 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdB N (2 * q) (2 * q) w.s3b2 ∘ r50Pre9 N q w
noncomputable def r50Pre11 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdB N (2 * q) (2 * q) w.s3b3 ∘ r50Pre10 N q w
noncomputable def r50Pre12 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdB N (2 * q) (2 * q) w.s3b4 ∘ r50Pre11 N q w
noncomputable def r50Pre13 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (1024 * (2 * q) * (2 * q))) :=
  r50IdB N (2 * q) (2 * q) w.s3b5 ∘ r50Pre12 N q w
noncomputable def r50Pre14 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (2048 * q * q)) :=
  r50DownB N q q w.s4b0 ∘ r50Pre13 N q w
noncomputable def r50Pre15 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (2048 * q * q)) :=
  r50IdB N q q w.s4b1 ∘ r50Pre14 N q w
noncomputable def r50Pre16 (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) :
    Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) → Vec (N * (2048 * q * q)) :=
  r50IdB N q q w.s4b2 ∘ r50Pre15 N q w

-- ════════════════════════════════════════════════════════════════
-- § The whole hypothesis budget, in two structures
-- ════════════════════════════════════════════════════════════════

/-- Every BatchNorm `ε` of the net is positive: the stem's and each bottleneck's bundle. -/
structure R50PosB {nCls : Nat} (w : R50BWeights nCls) : Prop where
  s : 0 < w.sε
  s1b0 : R50ProjPos w.s1b0
  s1b1 : R50IdPos w.s1b1
  s1b2 : R50IdPos w.s1b2
  s2b0 : R50ProjPos w.s2b0
  s2b1 : R50IdPos w.s2b1
  s2b2 : R50IdPos w.s2b2
  s2b3 : R50IdPos w.s2b3
  s3b0 : R50ProjPos w.s3b0
  s3b1 : R50IdPos w.s3b1
  s3b2 : R50IdPos w.s3b2
  s3b3 : R50IdPos w.s3b3
  s3b4 : R50IdPos w.s3b4
  s3b5 : R50IdPos w.s3b5
  s4b0 : R50ProjPos w.s4b0
  s4b1 : R50IdPos w.s4b1
  s4b2 : R50IdPos w.s4b2

/-- **Every relu is away from its kink and the stem pool has no tie, each at the activation
    its block actually sees**: the stem's clauses at the image, block `k`'s at `r50Pre(k-1)`. The
    head has none (GAP and dense are smooth). -/
structure R50SmoothAtB (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) : Prop where
  stem : R34StemSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.sW w.sb w.sε w.sγ w.sβ x
  pool : R34PoolSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q)))
    (StableHLO.cbReluStridedB N (h := 2 * (2 * (2 * (2 * q)))) (w := 2 * (2 * (2 * (2 * q)))) w.sW w.sb w.sε w.sγ w.sβ x)
  s1b0 : R50ProjSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 (r50Pre0 N q w x)
  s1b1 : R50IdSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 (r50Pre1 N q w x)
  s1b2 : R50IdSmoothAt N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 (r50Pre2 N q w x)
  s2b0 : R50DownSmoothAt N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Pre3 N q w x)
  s2b1 : R50IdSmoothAt N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Pre4 N q w x)
  s2b2 : R50IdSmoothAt N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Pre5 N q w x)
  s2b3 : R50IdSmoothAt N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Pre6 N q w x)
  s3b0 : R50DownSmoothAt N (2 * q) (2 * q) w.s3b0 (r50Pre7 N q w x)
  s3b1 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b1 (r50Pre8 N q w x)
  s3b2 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b2 (r50Pre9 N q w x)
  s3b3 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b3 (r50Pre10 N q w x)
  s3b4 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b4 (r50Pre11 N q w x)
  s3b5 : R50IdSmoothAt N (2 * q) (2 * q) w.s3b5 (r50Pre12 N q w x)
  s4b0 : R50DownSmoothAt N q q w.s4b0 (r50Pre13 N q w x)
  s4b1 : R50IdSmoothAt N q q w.s4b1 (r50Pre14 N q w x)
  s4b2 : R50IdSmoothAt N q q w.s4b2 (r50Pre15 N q w x)

-- ════════════════════════════════════════════════════════════════
-- § The chain equation — the layered `r50PreK` form IS the committed forward
--   ⚠ Peeled one layer at a time through `*_apply`, as ResNet-34's is: a one-step `rfl` against a
--   sixteen-deep nested application takes a kernel deterministic timeout.
-- ════════════════════════════════════════════════════════════════

theorem r50Pre0_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre0 N q w x = r34StemB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.sW w.sb w.sε w.sγ w.sβ x := by
  rw [r50Pre0]
theorem r50Pre1_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre1 N q w x = r50ProjB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 (r50Pre0 N q w x) := by
  rw [r50Pre1, Function.comp_apply]
theorem r50Pre2_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre2 N q w x = r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 (r50Pre1 N q w x) := by
  rw [r50Pre2, Function.comp_apply]
theorem r50Pre3_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre3 N q w x = r50IdB N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 (r50Pre2 N q w x) := by
  rw [r50Pre3, Function.comp_apply]
theorem r50Pre4_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre4 N q w x = r50DownB N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Pre3 N q w x) := by
  rw [r50Pre4, Function.comp_apply]
theorem r50Pre5_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre5 N q w x = r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Pre4 N q w x) := by
  rw [r50Pre5, Function.comp_apply]
theorem r50Pre6_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre6 N q w x = r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Pre5 N q w x) := by
  rw [r50Pre6, Function.comp_apply]
theorem r50Pre7_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre7 N q w x = r50IdB N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Pre6 N q w x) := by
  rw [r50Pre7, Function.comp_apply]
theorem r50Pre8_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre8 N q w x = r50DownB N (2 * q) (2 * q) w.s3b0 (r50Pre7 N q w x) := by
  rw [r50Pre8, Function.comp_apply]
theorem r50Pre9_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre9 N q w x = r50IdB N (2 * q) (2 * q) w.s3b1 (r50Pre8 N q w x) := by
  rw [r50Pre9, Function.comp_apply]
theorem r50Pre10_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre10 N q w x = r50IdB N (2 * q) (2 * q) w.s3b2 (r50Pre9 N q w x) := by
  rw [r50Pre10, Function.comp_apply]
theorem r50Pre11_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre11 N q w x = r50IdB N (2 * q) (2 * q) w.s3b3 (r50Pre10 N q w x) := by
  rw [r50Pre11, Function.comp_apply]
theorem r50Pre12_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre12 N q w x = r50IdB N (2 * q) (2 * q) w.s3b4 (r50Pre11 N q w x) := by
  rw [r50Pre12, Function.comp_apply]
theorem r50Pre13_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre13 N q w x = r50IdB N (2 * q) (2 * q) w.s3b5 (r50Pre12 N q w x) := by
  rw [r50Pre13, Function.comp_apply]
theorem r50Pre14_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre14 N q w x = r50DownB N q q w.s4b0 (r50Pre13 N q w x) := by
  rw [r50Pre14, Function.comp_apply]
theorem r50Pre15_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre15 N q w x = r50IdB N q q w.s4b1 (r50Pre14 N q w x) := by
  rw [r50Pre15, Function.comp_apply]
theorem r50Pre16_apply (N q : Nat) {nCls : Nat} (w : R50BWeights nCls) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    r50Pre16 N q w x = r50IdB N q q w.s4b2 (r50Pre15 N q w x) := by
  rw [r50Pre16, Function.comp_apply]

/-- **The committed nested-application forward IS the layered chain the VJP is stated on** — the
    ResNet-50 peer of `resnet34ForwardBFull_eq_chain`, and what lets the VJP be about
    `resnet50ForwardBFull` rather than about a re-spelling of it. -/
theorem resnet50ForwardBFull_eq_chain (N q : Nat) {nCls : Nat} (w : R50BWeights nCls)
    (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    resnet50ForwardBFull N q w x = (r34HeadB N q q w.Wd w.bd ∘ r50Pre16 N q w) x := by
  rw [resnet50ForwardBFull, Function.comp_apply, r50Pre16_apply, r50Pre15_apply, r50Pre14_apply, r50Pre13_apply, r50Pre12_apply, r50Pre11_apply, r50Pre10_apply, r50Pre9_apply, r50Pre8_apply, r50Pre7_apply, r50Pre6_apply, r50Pre5_apply, r50Pre4_apply, r50Pre3_apply, r50Pre2_apply, r50Pre1_apply, r50Pre0_apply]


-- ════════════════════════════════════════════════════════════════
-- § The apex — the whole net as ONE `CertLayer`
--   Stem, the sixteen bottlenecks and the head composed with `CertLayer.comp`, so the VJP, its
--   differentiability and the backward-graph faithfulness are the layer's fields. At the binder
--   `q` every width is the same syntactic `2 * (…)` nest on both sides of each `comp`, which is
--   what keeps the peel below cheap (`planning/certlayer_nets.md` §6).
-- ════════════════════════════════════════════════════════════════

/-- **ResNet-50 as one certified layer**, at every batch size and resolution. Its `.faithful`
    is a whole-net backward graph, stem pool included, proven to denote the VJP. -/
noncomputable def r50NetLayer (N q : Nat) (hq0 : 0 < q) {nCls : Nat} (w : R50BWeights nCls)
    (hp : R50PosB w) :
    StableHLO.CertLayer (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) (N * nCls) :=
  (r34StemLayer N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) (by norm_num) (by omega) (by omega)
      w.sW w.sb w.sε hp.s w.sγ w.sβ).comp <|
  (r50ProjLayer N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0 hp.s1b0).comp <|
  (r50IdLayer N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1 hp.s1b1).comp <|
  (r50IdLayer N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2 hp.s1b2).comp <|
  (r50DownLayer N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 hp.s2b0).comp <|
  (r50IdLayer N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 hp.s2b1).comp <|
  (r50IdLayer N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 hp.s2b2).comp <|
  (r50IdLayer N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 hp.s2b3).comp <|
  (r50DownLayer N (2 * q) (2 * q) w.s3b0 hp.s3b0).comp <|
  (r50IdLayer N (2 * q) (2 * q) w.s3b1 hp.s3b1).comp <|
  (r50IdLayer N (2 * q) (2 * q) w.s3b2 hp.s3b2).comp <|
  (r50IdLayer N (2 * q) (2 * q) w.s3b3 hp.s3b3).comp <|
  (r50IdLayer N (2 * q) (2 * q) w.s3b4 hp.s3b4).comp <|
  (r50IdLayer N (2 * q) (2 * q) w.s3b5 hp.s3b5).comp <|
  (r50DownLayer N q q w.s4b0 hp.s4b0).comp <|
  (r50IdLayer N q q w.s4b1 hp.s4b1).comp <|
  (r50IdLayer N q q w.s4b2 hp.s4b2).comp <|
  r34HeadLayer N q q w.Wd w.bd

/-- The composed layer's forward IS the committed nested-application forward: peeled one `comp`
    at a time, then each layer's forward by its `rfl` lemma. -/
theorem r50NetLayer_fwd_apply (N q : Nat) (hq0 : 0 < q) {nCls : Nat} (w : R50BWeights nCls)
    (hp : R50PosB w) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    (r50NetLayer N q hq0 w hp).fwd x = resnet50ForwardBFull N q w x := by
  rw [r50NetLayer]
  repeat rw [StableHLO.CertLayer.comp_fwd_apply]
  rw [r34StemLayer_fwd, r50ProjLayer_fwd, r50IdLayer_fwd, r50IdLayer_fwd, r50DownLayer_fwd,
    r50IdLayer_fwd, r50IdLayer_fwd, r50IdLayer_fwd, r50DownLayer_fwd, r50IdLayer_fwd,
    r50IdLayer_fwd, r50IdLayer_fwd, r50IdLayer_fwd, r50IdLayer_fwd, r50DownLayer_fwd,
    r50IdLayer_fwd, r50IdLayer_fwd, r34HeadLayer_fwd, resnet50ForwardBFull]

/-- `R50SmoothAtB` is the layer's `.ok`: one `comp_ok_of` per block, each naming its block's
    input `r50PreK`, so every step is a one-level `rfl`. -/
theorem r50SmoothAtB_ok (N q : Nat) (hq0 : 0 < q) {nCls : Nat} (w : R50BWeights nCls)
    (hp : R50PosB w) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (hx : R50SmoothAtB N q w x) : (r50NetLayer N q hq0 w hp).ok x := by
  refine StableHLO.CertLayer.comp_ok_of ⟨hx.stem, hx.pool⟩ (r50Pre0 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨trivial, ⟨hx.s1b0.hm1, hx.s1b0.hm2⟩, trivial⟩, hx.s1b0.hout⟩ (r50Pre1 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s1b1.hm1, hx.s1b1.hm2⟩, trivial⟩, hx.s1b1.hout⟩ (r50Pre2 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s1b2.hm1, hx.s1b2.hm2⟩, trivial⟩, hx.s1b2.hout⟩ (r50Pre3 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨trivial, ⟨hx.s2b0.hm1, hx.s2b0.hm2⟩, trivial⟩, hx.s2b0.hout⟩ (r50Pre4 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s2b1.hm1, hx.s2b1.hm2⟩, trivial⟩, hx.s2b1.hout⟩ (r50Pre5 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s2b2.hm1, hx.s2b2.hm2⟩, trivial⟩, hx.s2b2.hout⟩ (r50Pre6 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s2b3.hm1, hx.s2b3.hm2⟩, trivial⟩, hx.s2b3.hout⟩ (r50Pre7 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨trivial, ⟨hx.s3b0.hm1, hx.s3b0.hm2⟩, trivial⟩, hx.s3b0.hout⟩ (r50Pre8 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s3b1.hm1, hx.s3b1.hm2⟩, trivial⟩, hx.s3b1.hout⟩ (r50Pre9 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s3b2.hm1, hx.s3b2.hm2⟩, trivial⟩, hx.s3b2.hout⟩ (r50Pre10 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s3b3.hm1, hx.s3b3.hm2⟩, trivial⟩, hx.s3b3.hout⟩ (r50Pre11 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s3b4.hm1, hx.s3b4.hm2⟩, trivial⟩, hx.s3b4.hout⟩ (r50Pre12 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s3b5.hm1, hx.s3b5.hm2⟩, trivial⟩, hx.s3b5.hout⟩ (r50Pre13 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨trivial, ⟨hx.s4b0.hm1, hx.s4b0.hm2⟩, trivial⟩, hx.s4b0.hout⟩ (r50Pre14 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s4b1.hm1, hx.s4b1.hm2⟩, trivial⟩, hx.s4b1.hout⟩ (r50Pre15 N q w x) rfl ?_
  refine StableHLO.CertLayer.comp_ok_of ⟨⟨⟨hx.s4b2.hm1, hx.s4b2.hm2⟩, trivial⟩, hx.s4b2.hout⟩ (r50Pre16 N q w x) rfl ?_
  exact ⟨trivial, trivial⟩

/-- **ResNet-50 at TRUE BATCH-NORM has a certified input-VJP at a smooth point — all sixteen
    bottlenecks.** `r50NetLayer`'s `.vjp`, read at the layered chain, under two hypotheses: `R50PosB` (every `ε > 0`) and `R50SmoothAtB` (every relu clause and the pool's
    no-tie, each at its block's own input), beside `0 < q`.

    Pointwise, and necessarily: relu is kinked. Each block contributes THREE clauses — the
    two interior relus and the post-residual OUTER relu — where ResNet-34's basic block
    contributes two and EfficientNet's MBConv none.

    The head takes no hypothesis at all, and `N` and `q` are both variables, so this covers the
    224-px and 160-px artifacts at every batch size. `0 < q` is needed for the stem pool. -/
noncomputable def resnet50ForwardBFullHasVJPAt (N q : Nat) (hq0 : 0 < q) {nCls : Nat} (w : R50BWeights nCls)
    (hp : R50PosB w) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (hx : R50SmoothAtB N q w x) :
    HasVJPAt (r34HeadB N q q w.Wd w.bd ∘ r50Pre16 N q w) x :=
  (funext fun v => (r50NetLayer_fwd_apply N q hq0 w hp v).trans (resnet50ForwardBFull_eq_chain N q w v)
    : (r50NetLayer N q hq0 w hp).fwd = _) ▸ (r50NetLayer N q hq0 w hp).vjp x (r50SmoothAtB_ok N q hq0 w hp x hx)

/-- **Public correctness theorem**: the sixteen-bottleneck batch-BN backward equals the
    `pdiv`-contracted Jacobian of `resnet50ForwardBFull` ITSELF — the committed
    nested-application forward `ResNet50FullB.lean` defines — not of the layered chain the VJP is
    assembled on. Tied back through `resnet50ForwardBFull_eq_chain`. -/
theorem resnet50ForwardBFullHasVJPAt_correct (N q : Nat) (hq0 : 0 < q) {nCls : Nat} (w : R50BWeights nCls)
    (hp : R50PosB w) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (hx : R50SmoothAtB N q w x)
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q)))))))) :
    (resnet50ForwardBFullHasVJPAt N q hq0 w hp x hx).backward dy i =
      ∑ j : Fin (N * nCls), pdiv (resnet50ForwardBFull N q w) x i j * dy j := by
  have h := (resnet50ForwardBFullHasVJPAt N q hq0 w hp x hx).correct dy i
  rwa [show resnet50ForwardBFull N q w = r34HeadB N q q w.Wd w.bd ∘ r50Pre16 N q w
      from funext (resnet50ForwardBFull_eq_chain N q w)]

/-- The committed forward is differentiable at every smooth point — the layer's `.diff`. What
    the seal's `seal_differentiableAt` needs. -/
theorem resnet50ForwardBFull_differentiableAt (N q : Nat) (hq0 : 0 < q) {nCls : Nat} (w : R50BWeights nCls)
    (hp : R50PosB w) (x : Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (hx : R50SmoothAtB N q w x) :
    DifferentiableAt ℝ (resnet50ForwardBFull N q w) x := by
  rw [← funext (r50NetLayer_fwd_apply N q hq0 w hp)]
  exact (r50NetLayer N q hq0 w hp).diff x (r50SmoothAtB_ok N q hq0 w hp x hx)

end Proofs
