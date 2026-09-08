import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullB

/-! # MobileNetV4-Conv-M's whole-net input-VJP at TRUE BATCH-NORM (T1, the VJP half)

`MobileNetV4FullB.lean` states the batch-BN forward at the 21-row Conv-M table and gives it a
typed graph (T2). This file gives that forward a certified `HasVJPAt` — the other half of T1, and
with it MobileNetV4 has its first net-level tier.

⚠⚠ **No accuracy is quoted for this net.** Conv-M has no Imagenette run and no verified ImageNet
run; what pins the artifact to the reference's function is the pair of ties re-run 2026-09-07
(forward `max |Δ| = 3.770e-06`, gradient inside the reference's own fp32 floor).

## ⭐⭐ Eight hypotheses, not thirty-three

ResNet-50's apex (`resnet50ForwardB_full_has_vjp_at`) takes 33: a positivity bundle and a
smoothness bundle per block, plus the stem's and the pool's, threaded through sixteen hand-written
`r50Pre_k` prefix definitions and a bottom-up `have` chain. MobileNetV4's takes **eight** — one per
prefix — and the difference is `CertLayer`:

* `0 < ε` at all 77 BatchNorm sites is already INSIDE the weights — `UibParams` carries `hq he hd
  hz` and `Mnv4BWeights` carries the stem's, the fused stage's and the head's — so **no positivity
  hypothesis appears at this tier at all**, where R50 binds sixteen.
* every relu clause is some group's `.ok`, which `CertLayer.comp` built by conjoining each
  stage's condition **at that stage's own input** as the group was assembled. Writing them out
  would be roughly sixty clauses, each at a deeply nested activation. None is written down here.

⛔ It would be seven fewer still if the whole trunk were one `CertLayer` — and it cannot be. That
composition elaborates, but every later use has to peel `CertLayer.comp` to reach `.fwd`, and at
MNv4's literal resolutions that peel costs ten minutes and then a kernel timeout.
`MobileNetV4FullB.lean` records the measurement; the seven prefixes are the price, and they are
ResNet-50's own shape at seven stages rather than eighteen.

## The stem is the only thing composed by hand, and it has to be

⚠⚠ `CertLayer` demands a backward graph and **no render emits a gradient into `%x`** — there is no
`convStridedXlaBackBatched` token, because the artifact's backward ends at the stem conv's WEIGHT
gradient. That is EfficientNet-B0's situation exactly (`enetTrunk` takes its stem as a parameter
for the same reason), so MNv4's stem stays a plain function and the apex is one `vjp_comp_at`.

⭐ **No new `Foundation` lemma was needed.** `bnReluStage_has_vjp_at` is generic in the inner op,
so the XLA-padded stride-2 stem is one instantiation at `flatConvStride2Xla` — the same lemma
`cbReluB_has_vjp_at` uses at `flatConv` and `dwbReluB_has_vjp_at` uses at `depthwiseFlat`. ⚠ Its
relu6 twin is what `MobileNetV2FullBVJP.lean` instantiates one file over at the SAME padding; the
two stems differ in exactly one token, which is why they are separate definitions.

⚠ Pointwise (`HasVJPAt`), not global, and necessarily: relu is kinked. The fused stage is the one
stage that contributes nothing — swish is smooth, so its `ok` is `True`.

⭐ `N` is a binder, so this covers every batch size, and the artifacts' `N` is the PER-REPLICA
batch (`DataParallel.lean`, §4d).
-/

namespace Proofs

open scoped BigOperators

namespace StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The stem: its kink clause, its VJP, its differentiability
-- ════════════════════════════════════════════════════════════════

/-- The stem's single relu site is away from the kink at `x`. One clause: MNv4's stem is
    conv-bn-**relu**, where MobileNetV2's is relu6 and carries two. -/
def Mnv4StemSmoothAtB (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) : Prop :=
  ∀ k, bnBatchLA N oc h w εs γs βs (batchMap N (flatConvStride2Xla Ws bs) x) k ≠ 0

/-- ⭐ The stem's VJP: `bnReluStage_has_vjp_at` at the XLA-`SAME` strided conv. That lemma takes
    the inner op as a parameter, so the stride-2 stem is the same construction as every stride-1
    stage — zero new analytic content. -/
noncomputable def mnv4StemB_has_vjp_at (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : Mnv4StemSmoothAtB N h w Ws bs εs γs βs x) :
    HasVJPAt (mnv4StemB N h w Ws bs εs γs βs) x :=
  bnReluStage_has_vjp_at N (flatConvStride2Xla Ws bs)
    (flatConvStride2Xla_differentiable Ws bs) (flatConvStride2Xla_has_vjp Ws bs) εs hεs γs βs x hs

theorem mnv4StemB_differentiableAt (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : Mnv4StemSmoothAtB N h w Ws bs εs γs βs x) :
    DifferentiableAt ℝ (mnv4StemB N h w Ws bs εs γs βs) x :=
  bnReluStage_differentiableAt N (flatConvStride2Xla Ws bs)
    (flatConvStride2Xla_differentiable Ws bs) εs hεs γs βs x hs

-- ════════════════════════════════════════════════════════════════
-- § The whole hypothesis budget, in one structure
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **Everything MobileNetV4-Conv-M's whole-net VJP needs: the stem's one kink clause, and
    each group's `.ok` at the activation that group actually sees.**

    Each group field is a conjunction `CertLayer.comp` assembled from its blocks' conditions —
    each relu's condition stated at the activation THAT stage sees, in execution order. The fused
    stage contributes nothing (swish is smooth, `ok = True`), the eighteen skips contribute their
    bodies' conditions unchanged (an identity skip adds no kink), and GAP and dense contribute
    nothing. Roughly sixty clauses in total, none of which had to be written down. -/
structure Mnv4SmoothAt (N : Nat) {nCls : Nat} (w : Mnv4BWeights nCls)
    (x : Vec (N * (3 * 224 * 224))) : Prop where
  /-- the stem's relu is away from its kink at the image. -/
  stem : Mnv4StemSmoothAtB N 112 112 w.sW w.sb w.sE w.sg w.sbt x
  /-- the fused stage at the stem's output. ⭐ Vacuous: swish is smooth, so this is `True ∧ True`. -/
  fused : (mnv4FusedStack N w).ok (mnv4Pre0 N w x)
  /-- rows 1–2 at 56×56. -/
  g28 : (mnv4Res28Layer N w).ok (mnv4Pre1 N w x)
  /-- rows 3–6 at 28×28. -/
  g14a : (mnv4Res14aLayer N w).ok (mnv4Pre2 N w x)
  /-- rows 7–10 at 14×14. -/
  g14b : (mnv4Res14bLayer N w).ok (mnv4Pre3 N w x)
  /-- rows 11–15 at 14×14. -/
  g7a : (mnv4Res7aLayer N w).ok (mnv4Pre4 N w x)
  /-- rows 16–21 at 7×7. -/
  g7b : (mnv4Res7bLayer N w).ok (mnv4Pre5 N w x)
  /-- the head's two relus at the trunk's output. -/
  head : (mnv4HeadStack N w).ok (mnv4Pre6 N w x)

-- ════════════════════════════════════════════════════════════════
-- § The apex
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **MobileNetV4-Conv-M at TRUE BATCH-NORM has a certified input-VJP at a smooth point — the
    fused stage, all 21 UIB blocks and the two-conv head.** T1's VJP half, and the first net-level
    tier this net has ever had.

    A bottom-up chain of seven `vjp_comp_at`s: the stem, the fused stage, the five resolution
    groups, the head. Every step below the stem is a group's `.vjp` — `CertLayer.comp`'s
    composition theorem already applied within it, which is `CertifiedChain.lean`'s whole reason
    for existing; only the seven joins are made here.

    ⚠ Pointwise, and necessarily: relu is kinked. ⭐ `N` and `nCls` are both binders, so this
    covers the 10-class Imagenette artifacts and the 1000-class `mnv4in` ones at every batch size,
    and no `0 < ε` hypothesis appears — those live in the weight records. -/
noncomputable def mobilenetv4ForwardB_full_has_vjp_at (N : Nat) {nCls : Nat}
    (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) (hx : Mnv4SmoothAt N w x) :
    HasVJPAt (mobilenetv4ForwardB_full N w) x := by
  have d0 : DifferentiableAt ℝ (mnv4Pre0 N w) x :=
    mnv4StemB_differentiableAt N 112 112 w.sW w.sb w.sE w.hsE w.sg w.sbt x hx.stem
  have e0 : HasVJPAt (mnv4Pre0 N w) x :=
    mnv4StemB_has_vjp_at N 112 112 w.sW w.sb w.sE w.hsE w.sg w.sbt x hx.stem
  have e1 : HasVJPAt (mnv4Pre1 N w) x :=
    vjp_comp_at _ _ x d0 ((mnv4FusedStack N w).diff _ hx.fused) e0
      ((mnv4FusedStack N w).vjp _ hx.fused)
  have d1 : DifferentiableAt ℝ (mnv4Pre1 N w) x :=
    ((mnv4FusedStack N w).diff _ hx.fused).comp x d0
  have e2 : HasVJPAt (mnv4Pre2 N w) x :=
    vjp_comp_at _ _ x d1 ((mnv4Res28Layer N w).diff _ hx.g28) e1
      ((mnv4Res28Layer N w).vjp _ hx.g28)
  have d2 : DifferentiableAt ℝ (mnv4Pre2 N w) x :=
    ((mnv4Res28Layer N w).diff _ hx.g28).comp x d1
  have e3 : HasVJPAt (mnv4Pre3 N w) x :=
    vjp_comp_at _ _ x d2 ((mnv4Res14aLayer N w).diff _ hx.g14a) e2
      ((mnv4Res14aLayer N w).vjp _ hx.g14a)
  have d3 : DifferentiableAt ℝ (mnv4Pre3 N w) x :=
    ((mnv4Res14aLayer N w).diff _ hx.g14a).comp x d2
  have e4 : HasVJPAt (mnv4Pre4 N w) x :=
    vjp_comp_at _ _ x d3 ((mnv4Res14bLayer N w).diff _ hx.g14b) e3
      ((mnv4Res14bLayer N w).vjp _ hx.g14b)
  have d4 : DifferentiableAt ℝ (mnv4Pre4 N w) x :=
    ((mnv4Res14bLayer N w).diff _ hx.g14b).comp x d3
  have e5 : HasVJPAt (mnv4Pre5 N w) x :=
    vjp_comp_at _ _ x d4 ((mnv4Res7aLayer N w).diff _ hx.g7a) e4
      ((mnv4Res7aLayer N w).vjp _ hx.g7a)
  have d5 : DifferentiableAt ℝ (mnv4Pre5 N w) x :=
    ((mnv4Res7aLayer N w).diff _ hx.g7a).comp x d4
  have e6 : HasVJPAt (mnv4Pre6 N w) x :=
    vjp_comp_at _ _ x d5 ((mnv4Res7bLayer N w).diff _ hx.g7b) e5
      ((mnv4Res7bLayer N w).vjp _ hx.g7b)
  have d6 : DifferentiableAt ℝ (mnv4Pre6 N w) x :=
    ((mnv4Res7bLayer N w).diff _ hx.g7b).comp x d5
  exact vjp_comp_at _ _ x d6 ((mnv4HeadStack N w).diff _ hx.head) e6
    ((mnv4HeadStack N w).vjp _ hx.head)

/-- ⭐ And it IS the `pdiv`-contracted Jacobian of the whole net, at every batch size and both
    shipped class counts. The reading that says the object above is the gradient rather than
    merely a function of the right type. -/
theorem mobilenetv4ForwardB_full_has_vjp_at_correct (N : Nat) {nCls : Nat}
    (w : Mnv4BWeights nCls) (x : Vec (N * (3 * 224 * 224))) (hx : Mnv4SmoothAt N w x)
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * 224 * 224))) :
    (mobilenetv4ForwardB_full_has_vjp_at N w x hx).backward dy i
      = ∑ j : Fin (N * nCls), pdiv (mobilenetv4ForwardB_full N w) x i j * dy j :=
  (mobilenetv4ForwardB_full_has_vjp_at N w x hx).correct dy i

end StableHLO

end Proofs
