import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullVJP

/-! # MobileNetV2's whole-net input-VJP at TRUE BATCH-NORM (T1, the VJP half)

`MobileNetV2FullB.lean` states the batch-BN forward and its typed graph. This file gives that
forward a certified `HasVJPAt` at the paper depth — the MobileNetV2 peer of
`ResNet34FullBVJP.lean`, and the second piece of `formalization.yaml` 4e's port.

## No new mathematics, and no new lemma one tier down either

Every block VJP is already proven at `bnBatchLA`: `mnv2BodyB_has_vjp_at` and
`mnv2DownBodyB_has_vjp_at` (`MobileNetV2BackB0.lean`) are exactly the two body shapes
`mnv2ExpOnlyB` / `mnv2StridedB` unfold to, and `residual_has_vjp_at` wraps the first for the ten
skip blocks. The bundle lemmas below are delegations in the `EfficientNetFullB0` style.

⭐ **Where r34 needed a new `Foundation` lemma, this net needs none.** ResNet-34's stem ends in
`batchMap N (maxPool3s2Flat …)` and a max-pool has no derivative at a tie, so 4.1c had to write
`batchMap_has_vjp_at`. MobileNetV2 has NO stem pool — the stem is conv-BN-relu6 and downsamples
once — and its head's GAP and dense are smooth, so `batchMap_has_vjp` (the global one) covers
every `batchMap` in the net.

⭐ Three shapes are `bnRelu6Stage_has_vjp_at` at a different inner op, and that lemma is already
generic in it: the stem is that stage at `flatConvStride2Xla`, the head's first stage is `cbrB`,
and the expand/depthwise stages are `cbrB` / `dwbrB` / `dwbrBstrided`. The only composition written
here from scratch is the `t = 1` block `projB ∘ dwbrB` (b1), which has no `mnv2*BodyB` peer.

## The hypothesis budget

⚠ **Pointwise (`HasVJPAt`), not global, and necessarily.** relu6 is kinked on BOTH sides, so each
site carries `≠ 0 ∧ ≠ 6` and a global `HasVJP` through it is false. That is the repo standard for
the relu-family nets and matches `MobileNetV2FullVJP.lean`'s per-example fold; the axis this file
moves is the BatchNorm world, not the pointwise/global one.

⛔ **Two kink clauses per bottleneck, and the second is not r34's.** ResNet-34's blocks carry the
body's mid-relu AND a post-residual OUTER relu. MobileNetV2's carry the EXPAND relu6 and the
DEPTHWISE relu6, both inside the body — the linear bottleneck has no activation after `project`,
so the residual add IS the block output and contributes nothing. Sixteen expand-bearing blocks give
32 clauses, plus b1's single depthwise clause, plus the stem's and the head's: 35 relu6 sites,
bundled into 19 binders.

⭐ **The positivity bundles are REUSED**, not re-declared: `IVPos` / `IVNoExpPos`
(`MobileNetV2FullVJP.lean`) say `0 < ε` at each BatchNorm site and know nothing about which world
reduces it. Only the smoothness bundles need batched peers, because a kink condition names the
activation and `bnBatchLA` is a different activation from `bnPerChannelTensor3`.

The running activations are named `mnv2PreB0 … mnv2PreB17` so each bundle can be STATED at the
activation entering its block without a seventeen-deep nested application inline; `mnv2PreB17`
doubles as the trunk, and `mobilenetv2ForwardB_full_eq_chain` bridges it back to the committed
nested-application forward.

⭐ `N` is a variable throughout: this tier carries no numerals.
-/

namespace Proofs

open scoped BigOperators

set_option maxHeartbeats 1000000

-- ════════════════════════════════════════════════════════════════
-- § The batched smoothness bundles
--   ⭐ `IVPos` / `IVNoExpPos` are reused from `MobileNetV2FullVJP.lean` — a BN epsilon's
--   positivity does not know which axis the norm reduces. Only these do.
-- ════════════════════════════════════════════════════════════════

/-- Both relu6 sites of a stride-1 bottleneck are away from BOTH kinks at `v`, at batch BN: the
    expand-BN output and the depthwise-BN output each avoid `0` and `6` in every coordinate. -/
structure IVSmoothAtB (N h w : Nat) {ic mid oc : Nat} (q : IVW ic mid oc)
    (v : Vec (N * (ic * h * w))) : Prop where
  he : ∀ k, (StableHLO.bnBatchLA N mid h w q.eε q.eγ q.eβ
               (StableHLO.batchMap N (flatConv q.eW q.eb) v) k ≠ 0 ∧
              StableHLO.bnBatchLA N mid h w q.eε q.eγ q.eβ
               (StableHLO.batchMap N (flatConv q.eW q.eb) v) k ≠ 6)
  hd : ∀ k, (StableHLO.bnBatchLA N mid h w q.dε q.dγ q.dβ
               (StableHLO.batchMap N (depthwiseFlat q.dW q.db)
                 (StableHLO.cbrB N (h := h) (w := w) q.eW q.eb q.eε q.eγ q.eβ v)) k ≠ 0 ∧
              StableHLO.bnBatchLA N mid h w q.dε q.dγ q.dβ
               (StableHLO.batchMap N (depthwiseFlat q.dW q.db)
                 (StableHLO.cbrB N (h := h) (w := w) q.eW q.eb q.eε q.eγ q.eβ v)) k ≠ 6)

/-- Both relu6 sites of a stride-2 bottleneck are away from both kinks at `v`, at batch BN (the
    expand runs at the pre-downsample `2h x 2w` grid, the XLA-`SAME` depthwise at `h x w`). -/
structure IVStridedSmoothAtB (N h w : Nat) {ic mid oc : Nat} (q : IVW ic mid oc)
    (v : Vec (N * (ic * (2 * h) * (2 * w)))) : Prop where
  he : ∀ k, (StableHLO.bnBatchLA N mid (2 * h) (2 * w) q.eε q.eγ q.eβ
               (StableHLO.batchMap N (flatConv q.eW q.eb) v) k ≠ 0 ∧
              StableHLO.bnBatchLA N mid (2 * h) (2 * w) q.eε q.eγ q.eβ
               (StableHLO.batchMap N (flatConv q.eW q.eb) v) k ≠ 6)
  hd : ∀ k, (StableHLO.bnBatchLA N mid h w q.dε q.dγ q.dβ
               (StableHLO.batchMap N (depthwiseStride2FlatXla q.dW q.db)
                 (StableHLO.cbrB N (h := 2 * h) (w := 2 * w) q.eW q.eb q.eε q.eγ q.eβ v)) k ≠ 0 ∧
              StableHLO.bnBatchLA N mid h w q.dε q.dγ q.dβ
               (StableHLO.batchMap N (depthwiseStride2FlatXla q.dW q.db)
                 (StableHLO.cbrB N (h := 2 * h) (w := 2 * w) q.eW q.eb q.eε q.eγ q.eβ v)) k ≠ 6)

/-- The single relu6 site of the `t = 1` bottleneck (depthwise-BN output) is away from both
    kinks at batch BN. -/
structure IVNoExpSmoothAtB (N h w : Nat) {ic oc : Nat} (q : IVWNoExp ic oc)
    (v : Vec (N * (ic * h * w))) : Prop where
  hd : ∀ k, (StableHLO.bnBatchLA N ic h w q.dε q.dγ q.dβ
               (StableHLO.batchMap N (depthwiseFlat q.dW q.db) v) k ≠ 0 ∧
              StableHLO.bnBatchLA N ic h w q.dε q.dγ q.dβ
               (StableHLO.batchMap N (depthwiseFlat q.dW q.db) v) k ≠ 6)

/-- The stem's relu6 is away from both kinks at the input `x`, at batch BN. -/
def MNV2StemSmoothAtB (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) : Prop :=
  ∀ k, (StableHLO.bnBatchLA N oc h w εs γs βs
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) k ≠ 0 ∧
        StableHLO.bnBatchLA N oc h w εs γs βs
          (StableHLO.batchMap N (flatConvStride2Xla Ws bs) x) k ≠ 6)

/-- The head's relu6 is away from both kinks at the trunk output `v`, at batch BN. -/
def MNV2HeadSmoothAtB (N h w : Nat) {ic oc : Nat}
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (v : Vec (N * (ic * h * w))) : Prop :=
  ∀ k, (StableHLO.bnBatchLA N oc h w εh γh βh
          (StableHLO.batchMap N (flatConv Wh bh) v) k ≠ 0 ∧
        StableHLO.bnBatchLA N oc h w εh γh βh
          (StableHLO.batchMap N (flatConv Wh bh) v) k ≠ 6)

-- ════════════════════════════════════════════════════════════════
-- § Stem, block and head bundle lemmas (delegation)
-- ════════════════════════════════════════════════════════════════

/-- Stem VJP: `bnRelu6Stage_has_vjp_at` at the XLA-`SAME` strided conv. That lemma takes the inner
    op as a parameter, so the stride-2 stem is the same construction as every stride-1 stage. -/
noncomputable def mnv2StemB_has_vjp_at (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : MNV2StemSmoothAtB N h w Ws bs εs γs βs x) :
    HasVJPAt (mnv2StemB N h w Ws bs εs γs βs) x :=
  StableHLO.bnRelu6Stage_has_vjp_at N (flatConvStride2Xla Ws bs)
    (flatConvStride2Xla_differentiable Ws bs) (flatConvStride2Xla_has_vjp Ws bs) εs hεs γs βs x hs

theorem mnv2StemB_differentiableAt (N h w : Nat) {ic oc kH kW : Nat}
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (hεs : 0 < εs) (γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : MNV2StemSmoothAtB N h w Ws bs εs γs βs x) :
    DifferentiableAt ℝ (mnv2StemB N h w Ws bs εs γs βs) x :=
  StableHLO.bnRelu6Stage_differentiableAt N (flatConvStride2Xla Ws bs)
    (flatConvStride2Xla_differentiable Ws bs) εs hεs γs βs x hs

/-- `t = 1` bottleneck VJP (b1): `projB ∘ dwbrB`. The one block shape with no `mnv2*BodyB` lemma
    to delegate to, so it composes its two stages here. -/
noncomputable def mnv2NoExpB_has_vjp_at (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (hq : IVNoExpPos p) (v : Vec (N * (ic * h * w))) (hs : IVNoExpSmoothAtB N h w p v) :
    HasVJPAt (mnv2NoExpB N h w p) v :=
  vjp_comp_at _ (projB N (h := h) (w := w) p.pW p.pb p.pε p.pγ p.pβ) v
    (StableHLO.dwbrB_differentiableAt N p.dW p.db p.dε hq.hd p.dγ p.dβ v hs.hd)
    ((projB_differentiable N (h := h) (w := w) p.pW p.pb p.pε hq.hp p.pγ p.pβ) _)
    (StableHLO.dwbrB_has_vjp_at N p.dW p.db p.dε hq.hd p.dγ p.dβ v hs.hd)
    ((projB_has_vjp N (h := h) (w := w) p.pW p.pb p.pε hq.hp p.pγ p.pβ).toHasVJPAt _)

theorem mnv2NoExpB_differentiableAt (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (hq : IVNoExpPos p) (v : Vec (N * (ic * h * w))) (hs : IVNoExpSmoothAtB N h w p v) :
    DifferentiableAt ℝ (mnv2NoExpB N h w p) v :=
  ((projB_differentiable N (h := h) (w := w) p.pW p.pb p.pε hq.hp p.pγ p.pβ) _).comp v
    (StableHLO.dwbrB_differentiableAt N p.dW p.db p.dε hq.hd p.dγ p.dβ v hs.hd)

/-- Stride-1 no-skip bottleneck VJP (b11, b17) — `mnv2BodyB_has_vjp_at` at the bundle's fields. -/
noncomputable def mnv2ExpOnlyB_has_vjp_at (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (hq : IVPos p) (v : Vec (N * (ic * h * w))) (hs : IVSmoothAtB N h w p v) :
    HasVJPAt (mnv2ExpOnlyB N h w p) v :=
  StableHLO.mnv2BodyB_has_vjp_at N p.eW p.eb p.eε hq.he p.eγ p.eβ
    p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ v hs.he hs.hd

theorem mnv2ExpOnlyB_differentiableAt (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (hq : IVPos p) (v : Vec (N * (ic * h * w))) (hs : IVSmoothAtB N h w p v) :
    DifferentiableAt ℝ (mnv2ExpOnlyB N h w p) v :=
  StableHLO.mnv2BodyB_differentiableAt N p.eW p.eb p.eε hq.he p.eγ p.eβ
    p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ v hs.he hs.hd

/-- Stride-1 skip bottleneck VJP — the body VJP under `residual_has_vjp_at`. The identity arm is
    smooth everywhere, so the skip adds no hypothesis. -/
noncomputable def mnv2ResidB_has_vjp_at (N h w : Nat) {c mid : Nat} (p : IVW c mid c)
    (hq : IVPos p) (v : Vec (N * (c * h * w))) (hs : IVSmoothAtB N h w p v) :
    HasVJPAt (mnv2ResidB N h w p) v :=
  residual_has_vjp_at _ v (mnv2ExpOnlyB_differentiableAt N h w p hq v hs)
    (mnv2ExpOnlyB_has_vjp_at N h w p hq v hs)

theorem mnv2ResidB_differentiableAt (N h w : Nat) {c mid : Nat} (p : IVW c mid c)
    (hq : IVPos p) (v : Vec (N * (c * h * w))) (hs : IVSmoothAtB N h w p v) :
    DifferentiableAt ℝ (mnv2ResidB N h w p) v := by
  show DifferentiableAt ℝ (biPath _ (fun y => y)) v
  exact (mnv2ExpOnlyB_differentiableAt N h w p hq v hs).add differentiable_id.differentiableAt

/-- Stride-2 downsampling bottleneck VJP — `mnv2DownBodyB_has_vjp_at` at the bundle's fields. -/
noncomputable def mnv2StridedB_has_vjp_at (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (hq : IVPos p) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : IVStridedSmoothAtB N h w p v) :
    HasVJPAt (mnv2StridedB N h w p) v :=
  StableHLO.mnv2DownBodyB_has_vjp_at N p.eW p.eb p.eε hq.he p.eγ p.eβ
    p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ v hs.he hs.hd

theorem mnv2StridedB_differentiableAt (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (hq : IVPos p) (v : Vec (N * (ic * (2 * h) * (2 * w))))
    (hs : IVStridedSmoothAtB N h w p v) :
    DifferentiableAt ℝ (mnv2StridedB N h w p) v :=
  StableHLO.mnv2DownBodyB_differentiableAt N p.eW p.eb p.eε hq.he p.eγ p.eβ
    p.dW p.db p.dε hq.hd p.dγ p.dβ p.pW p.pb p.pε hq.hp p.pγ p.pβ v hs.he hs.hd

/-- Head VJP: the 1x1 conv-bn-relu6 stage (`cbrB`, pointwise), then GAP and dense — both smooth,
    both `batchMap` of a per-example op, so both lift with the GLOBAL `batchMap_has_vjp`.
    ⚠ Unlike r34's, this head is NOT hypothesis-free: MobileNetV2 puts a relu6 in front of the
    pool, so the head carries the net's 35th kink site. -/
noncomputable def mnv2HeadB_has_vjp_at (N h w : Nat) {ic oc nCls : Nat}
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (v : Vec (N * (ic * h * w)))
    (hs : MNV2HeadSmoothAtB N h w Wh bh εh γh βh v) :
    HasVJPAt (mnv2HeadB N h w Wh bh εh γh βh Wd bd) v := by
  have c_vjp := StableHLO.cbrB_has_vjp_at N Wh bh εh hεh γh βh v hs
  have c_diff := StableHLO.cbrB_differentiableAt N Wh bh εh hεh γh βh v hs
  have g_vjp : HasVJPAt (StableHLO.batchMap N (globalAvgPoolFlat oc h w) ∘
      StableHLO.cbrB N (h := h) (w := w) Wh bh εh γh βh) v :=
    vjp_comp_at _ _ v c_diff
      ((batchMap_differentiable _ (globalAvgPoolFlat_differentiable oc h w)) _) c_vjp
      ((batchMap_has_vjp _ (globalAvgPoolFlat_has_vjp oc h w)
        (globalAvgPoolFlat_differentiable oc h w)).toHasVJPAt _)
  have g_diff : DifferentiableAt ℝ (StableHLO.batchMap N (globalAvgPoolFlat oc h w) ∘
      StableHLO.cbrB N (h := h) (w := w) Wh bh εh γh βh) v :=
    ((batchMap_differentiable _ (globalAvgPoolFlat_differentiable oc h w)) _).comp v c_diff
  exact vjp_comp_at _ (StableHLO.batchMap N (dense Wd bd)) v g_diff
    ((batchMap_differentiable _ (dense_differentiable Wd bd)) _) g_vjp
    ((batchMap_has_vjp _ (dense_has_vjp Wd bd) (dense_differentiable Wd bd)).toHasVJPAt _)

theorem mnv2HeadB_differentiableAt (N h w : Nat) {ic oc nCls : Nat}
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (v : Vec (N * (ic * h * w)))
    (hs : MNV2HeadSmoothAtB N h w Wh bh εh γh βh v) :
    DifferentiableAt ℝ (mnv2HeadB N h w Wh bh εh γh βh Wd bd) v :=
  ((batchMap_differentiable _ (dense_differentiable Wd bd)) _).comp v
    (((batchMap_differentiable _ (globalAvgPoolFlat_differentiable oc h w)) _).comp v
      (StableHLO.cbrB_differentiableAt N Wh bh εh hεh γh βh v hs))

-- ════════════════════════════════════════════════════════════════
-- § The running activations — `mnv2PreBK` = the net truncated after block `K`
--   `mnv2PreB0` is the stem; each later one is one `∘` deeper, and `mnv2PreB17` is the whole
--   seventeen-bottleneck trunk. They exist so the 19 hypothesis bundles can be STATED at the
--   activation entering their block without a seventeen-deep nested application inline.
-- ════════════════════════════════════════════════════════════════

noncomputable def mnv2PreB0 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (32 * 112 * 112)) :=
  mnv2StemB N 112 112 w.sW w.sb w.sε w.sγ w.sβ
noncomputable def mnv2PreB1 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (16 * 112 * 112)) :=
  mnv2NoExpB N 112 112 w.b1 ∘ mnv2PreB0 N w
noncomputable def mnv2PreB2 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (24 * 56 * 56)) :=
  mnv2StridedB N 56 56 w.b2 ∘ mnv2PreB1 N w
noncomputable def mnv2PreB3 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (24 * 56 * 56)) :=
  mnv2ResidB N 56 56 w.b3 ∘ mnv2PreB2 N w
noncomputable def mnv2PreB4 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (32 * 28 * 28)) :=
  mnv2StridedB N 28 28 w.b4 ∘ mnv2PreB3 N w
noncomputable def mnv2PreB5 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (32 * 28 * 28)) :=
  mnv2ResidB N 28 28 w.b5 ∘ mnv2PreB4 N w
noncomputable def mnv2PreB6 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (32 * 28 * 28)) :=
  mnv2ResidB N 28 28 w.b6 ∘ mnv2PreB5 N w
noncomputable def mnv2PreB7 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (64 * 14 * 14)) :=
  mnv2StridedB N 14 14 w.b7 ∘ mnv2PreB6 N w
noncomputable def mnv2PreB8 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (64 * 14 * 14)) :=
  mnv2ResidB N 14 14 w.b8 ∘ mnv2PreB7 N w
noncomputable def mnv2PreB9 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (64 * 14 * 14)) :=
  mnv2ResidB N 14 14 w.b9 ∘ mnv2PreB8 N w
noncomputable def mnv2PreB10 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (64 * 14 * 14)) :=
  mnv2ResidB N 14 14 w.b10 ∘ mnv2PreB9 N w
noncomputable def mnv2PreB11 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (96 * 14 * 14)) :=
  mnv2ExpOnlyB N 14 14 w.b11 ∘ mnv2PreB10 N w
noncomputable def mnv2PreB12 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (96 * 14 * 14)) :=
  mnv2ResidB N 14 14 w.b12 ∘ mnv2PreB11 N w
noncomputable def mnv2PreB13 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (96 * 14 * 14)) :=
  mnv2ResidB N 14 14 w.b13 ∘ mnv2PreB12 N w
noncomputable def mnv2PreB14 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (160 * 7 * 7)) :=
  mnv2StridedB N 7 7 w.b14 ∘ mnv2PreB13 N w
noncomputable def mnv2PreB15 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (160 * 7 * 7)) :=
  mnv2ResidB N 7 7 w.b15 ∘ mnv2PreB14 N w
noncomputable def mnv2PreB16 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (160 * 7 * 7)) :=
  mnv2ResidB N 7 7 w.b16 ∘ mnv2PreB15 N w
noncomputable def mnv2PreB17 (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls) :
    Vec (N * (3 * (2 * 112) * (2 * 112))) → Vec (N * (320 * 7 * 7)) :=
  mnv2ExpOnlyB N 7 7 w.b17 ∘ mnv2PreB16 N w

-- ════════════════════════════════════════════════════════════════
-- § The apex
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **MobileNetV2 at TRUE BATCH-NORM has a certified input-VJP at a smooth point — all
    seventeen bottlenecks.** Chains stem → the `[t,c,n,s]` ladder → head with `vjp_comp_at`, one
    positivity bundle and one smoothness bundle per block. T1's VJP half for
    `formalization.yaml` 4e's port, and the batched peer of `mobilenetv2_full_has_vjp_at`.

    ⚠ Pointwise, and necessarily: relu6 is kinked on both sides. ⛔ Each expand-bearing block
    contributes TWO clauses — the expand relu6 and the depthwise relu6, both INSIDE the body —
    where ResNet-34's basic block contributes a mid-relu and a post-residual OUTER relu. The
    linear bottleneck has no activation after `project`, so MobileNetV2's residual add is the
    block output and adds nothing.

    ⚠ Unlike r34's, the head is NOT hypothesis-free: its 1x1 conv-BN is followed by a relu6.

    ⭐ `N` is a variable: this tier carries no numerals. -/
noncomputable def mobilenetv2ForwardB_full_has_vjp_at (N : Nat) {nCls : Nat}
    (w : MNV2BWeights nCls)
    (hsε : 0 < w.sε) (hhε : 0 < w.hε)
    (qb1 : IVNoExpPos w.b1)
    (qb2 : IVPos w.b2)
    (qb3 : IVPos w.b3)
    (qb4 : IVPos w.b4)
    (qb5 : IVPos w.b5)
    (qb6 : IVPos w.b6)
    (qb7 : IVPos w.b7)
    (qb8 : IVPos w.b8)
    (qb9 : IVPos w.b9)
    (qb10 : IVPos w.b10)
    (qb11 : IVPos w.b11)
    (qb12 : IVPos w.b12)
    (qb13 : IVPos w.b13)
    (qb14 : IVPos w.b14)
    (qb15 : IVPos w.b15)
    (qb16 : IVPos w.b16)
    (qb17 : IVPos w.b17)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112))))
    (h_stem : MNV2StemSmoothAtB N 112 112 w.sW w.sb w.sε w.sγ w.sβ x)
    (sb1 : IVNoExpSmoothAtB N 112 112 w.b1 (mnv2PreB0 N w x))
    (sb2 : IVStridedSmoothAtB N 56 56 w.b2 (mnv2PreB1 N w x))
    (sb3 : IVSmoothAtB N 56 56 w.b3 (mnv2PreB2 N w x))
    (sb4 : IVStridedSmoothAtB N 28 28 w.b4 (mnv2PreB3 N w x))
    (sb5 : IVSmoothAtB N 28 28 w.b5 (mnv2PreB4 N w x))
    (sb6 : IVSmoothAtB N 28 28 w.b6 (mnv2PreB5 N w x))
    (sb7 : IVStridedSmoothAtB N 14 14 w.b7 (mnv2PreB6 N w x))
    (sb8 : IVSmoothAtB N 14 14 w.b8 (mnv2PreB7 N w x))
    (sb9 : IVSmoothAtB N 14 14 w.b9 (mnv2PreB8 N w x))
    (sb10 : IVSmoothAtB N 14 14 w.b10 (mnv2PreB9 N w x))
    (sb11 : IVSmoothAtB N 14 14 w.b11 (mnv2PreB10 N w x))
    (sb12 : IVSmoothAtB N 14 14 w.b12 (mnv2PreB11 N w x))
    (sb13 : IVSmoothAtB N 14 14 w.b13 (mnv2PreB12 N w x))
    (sb14 : IVStridedSmoothAtB N 7 7 w.b14 (mnv2PreB13 N w x))
    (sb15 : IVSmoothAtB N 7 7 w.b15 (mnv2PreB14 N w x))
    (sb16 : IVSmoothAtB N 7 7 w.b16 (mnv2PreB15 N w x))
    (sb17 : IVSmoothAtB N 7 7 w.b17 (mnv2PreB16 N w x))
    (h_head : MNV2HeadSmoothAtB N 7 7 w.hW w.hb w.hε w.hγ w.hβ (mnv2PreB17 N w x))
    :
    HasVJPAt (mnv2HeadB N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb ∘ mnv2PreB17 N w) x := by
  have dS : DifferentiableAt ℝ (mnv2PreB0 N w) x :=
    mnv2StemB_differentiableAt N 112 112 w.sW w.sb w.sε hsε w.sγ w.sβ x h_stem
  have vS : HasVJPAt (mnv2PreB0 N w) x :=
    mnv2StemB_has_vjp_at N 112 112 w.sW w.sb w.sε hsε w.sγ w.sβ x h_stem
  have d1 := mnv2NoExpB_differentiableAt N 112 112 w.b1 qb1 _ sb1
  have e1 : HasVJPAt (mnv2PreB1 N w) x :=
    vjp_comp_at _ _ x dS d1 vS (mnv2NoExpB_has_vjp_at N 112 112 w.b1 qb1 _ sb1)
  have f1 : DifferentiableAt ℝ (mnv2PreB1 N w) x := d1.comp x dS
  have d2 := mnv2StridedB_differentiableAt N 56 56 w.b2 qb2 _ sb2
  have e2 : HasVJPAt (mnv2PreB2 N w) x :=
    vjp_comp_at _ _ x f1 d2 e1 (mnv2StridedB_has_vjp_at N 56 56 w.b2 qb2 _ sb2)
  have f2 : DifferentiableAt ℝ (mnv2PreB2 N w) x := d2.comp x f1
  have d3 := mnv2ResidB_differentiableAt N 56 56 w.b3 qb3 _ sb3
  have e3 : HasVJPAt (mnv2PreB3 N w) x :=
    vjp_comp_at _ _ x f2 d3 e2 (mnv2ResidB_has_vjp_at N 56 56 w.b3 qb3 _ sb3)
  have f3 : DifferentiableAt ℝ (mnv2PreB3 N w) x := d3.comp x f2
  have d4 := mnv2StridedB_differentiableAt N 28 28 w.b4 qb4 _ sb4
  have e4 : HasVJPAt (mnv2PreB4 N w) x :=
    vjp_comp_at _ _ x f3 d4 e3 (mnv2StridedB_has_vjp_at N 28 28 w.b4 qb4 _ sb4)
  have f4 : DifferentiableAt ℝ (mnv2PreB4 N w) x := d4.comp x f3
  have d5 := mnv2ResidB_differentiableAt N 28 28 w.b5 qb5 _ sb5
  have e5 : HasVJPAt (mnv2PreB5 N w) x :=
    vjp_comp_at _ _ x f4 d5 e4 (mnv2ResidB_has_vjp_at N 28 28 w.b5 qb5 _ sb5)
  have f5 : DifferentiableAt ℝ (mnv2PreB5 N w) x := d5.comp x f4
  have d6 := mnv2ResidB_differentiableAt N 28 28 w.b6 qb6 _ sb6
  have e6 : HasVJPAt (mnv2PreB6 N w) x :=
    vjp_comp_at _ _ x f5 d6 e5 (mnv2ResidB_has_vjp_at N 28 28 w.b6 qb6 _ sb6)
  have f6 : DifferentiableAt ℝ (mnv2PreB6 N w) x := d6.comp x f5
  have d7 := mnv2StridedB_differentiableAt N 14 14 w.b7 qb7 _ sb7
  have e7 : HasVJPAt (mnv2PreB7 N w) x :=
    vjp_comp_at _ _ x f6 d7 e6 (mnv2StridedB_has_vjp_at N 14 14 w.b7 qb7 _ sb7)
  have f7 : DifferentiableAt ℝ (mnv2PreB7 N w) x := d7.comp x f6
  have d8 := mnv2ResidB_differentiableAt N 14 14 w.b8 qb8 _ sb8
  have e8 : HasVJPAt (mnv2PreB8 N w) x :=
    vjp_comp_at _ _ x f7 d8 e7 (mnv2ResidB_has_vjp_at N 14 14 w.b8 qb8 _ sb8)
  have f8 : DifferentiableAt ℝ (mnv2PreB8 N w) x := d8.comp x f7
  have d9 := mnv2ResidB_differentiableAt N 14 14 w.b9 qb9 _ sb9
  have e9 : HasVJPAt (mnv2PreB9 N w) x :=
    vjp_comp_at _ _ x f8 d9 e8 (mnv2ResidB_has_vjp_at N 14 14 w.b9 qb9 _ sb9)
  have f9 : DifferentiableAt ℝ (mnv2PreB9 N w) x := d9.comp x f8
  have d10 := mnv2ResidB_differentiableAt N 14 14 w.b10 qb10 _ sb10
  have e10 : HasVJPAt (mnv2PreB10 N w) x :=
    vjp_comp_at _ _ x f9 d10 e9 (mnv2ResidB_has_vjp_at N 14 14 w.b10 qb10 _ sb10)
  have f10 : DifferentiableAt ℝ (mnv2PreB10 N w) x := d10.comp x f9
  have d11 := mnv2ExpOnlyB_differentiableAt N 14 14 w.b11 qb11 _ sb11
  have e11 : HasVJPAt (mnv2PreB11 N w) x :=
    vjp_comp_at _ _ x f10 d11 e10 (mnv2ExpOnlyB_has_vjp_at N 14 14 w.b11 qb11 _ sb11)
  have f11 : DifferentiableAt ℝ (mnv2PreB11 N w) x := d11.comp x f10
  have d12 := mnv2ResidB_differentiableAt N 14 14 w.b12 qb12 _ sb12
  have e12 : HasVJPAt (mnv2PreB12 N w) x :=
    vjp_comp_at _ _ x f11 d12 e11 (mnv2ResidB_has_vjp_at N 14 14 w.b12 qb12 _ sb12)
  have f12 : DifferentiableAt ℝ (mnv2PreB12 N w) x := d12.comp x f11
  have d13 := mnv2ResidB_differentiableAt N 14 14 w.b13 qb13 _ sb13
  have e13 : HasVJPAt (mnv2PreB13 N w) x :=
    vjp_comp_at _ _ x f12 d13 e12 (mnv2ResidB_has_vjp_at N 14 14 w.b13 qb13 _ sb13)
  have f13 : DifferentiableAt ℝ (mnv2PreB13 N w) x := d13.comp x f12
  have d14 := mnv2StridedB_differentiableAt N 7 7 w.b14 qb14 _ sb14
  have e14 : HasVJPAt (mnv2PreB14 N w) x :=
    vjp_comp_at _ _ x f13 d14 e13 (mnv2StridedB_has_vjp_at N 7 7 w.b14 qb14 _ sb14)
  have f14 : DifferentiableAt ℝ (mnv2PreB14 N w) x := d14.comp x f13
  have d15 := mnv2ResidB_differentiableAt N 7 7 w.b15 qb15 _ sb15
  have e15 : HasVJPAt (mnv2PreB15 N w) x :=
    vjp_comp_at _ _ x f14 d15 e14 (mnv2ResidB_has_vjp_at N 7 7 w.b15 qb15 _ sb15)
  have f15 : DifferentiableAt ℝ (mnv2PreB15 N w) x := d15.comp x f14
  have d16 := mnv2ResidB_differentiableAt N 7 7 w.b16 qb16 _ sb16
  have e16 : HasVJPAt (mnv2PreB16 N w) x :=
    vjp_comp_at _ _ x f15 d16 e15 (mnv2ResidB_has_vjp_at N 7 7 w.b16 qb16 _ sb16)
  have f16 : DifferentiableAt ℝ (mnv2PreB16 N w) x := d16.comp x f15
  have d17 := mnv2ExpOnlyB_differentiableAt N 7 7 w.b17 qb17 _ sb17
  have e17 : HasVJPAt (mnv2PreB17 N w) x :=
    vjp_comp_at _ _ x f16 d17 e16 (mnv2ExpOnlyB_has_vjp_at N 7 7 w.b17 qb17 _ sb17)
  have f17 : DifferentiableAt ℝ (mnv2PreB17 N w) x := d17.comp x f16
  exact vjp_comp_at _ (mnv2HeadB N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb) x f17
    (mnv2HeadB_differentiableAt N 7 7 w.hW w.hb w.hε hhε w.hγ w.hβ w.fcW w.fcb _ h_head) e17
    (mnv2HeadB_has_vjp_at N 7 7 w.hW w.hb w.hε hhε w.hγ w.hβ w.fcW w.fcb _ h_head)


-- ════════════════════════════════════════════════════════════════
-- § The chain equation — the layered `mnv2PreBK` form IS the committed forward
--   ⚠ Peeled one layer at a time through `*_apply`. A one-step `rfl` against a seventeen-deep
--   nested application does not survive (`MobileNetV2FullVJP.lean`'s section header records the
--   kernel deterministic timeout); `rw [<the def>, Function.comp_apply]` closes on syntactically
--   identical terms and never unfolds an inner layer.
-- ════════════════════════════════════════════════════════════════

theorem mnv2PreB0_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB0 N w x = mnv2StemB N 112 112 w.sW w.sb w.sε w.sγ w.sβ x := by
  rw [mnv2PreB0]
theorem mnv2PreB1_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB1 N w x = mnv2NoExpB N 112 112 w.b1 (mnv2PreB0 N w x) := by
  rw [mnv2PreB1, Function.comp_apply]
theorem mnv2PreB2_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB2 N w x = mnv2StridedB N 56 56 w.b2 (mnv2PreB1 N w x) := by
  rw [mnv2PreB2, Function.comp_apply]
theorem mnv2PreB3_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB3 N w x = mnv2ResidB N 56 56 w.b3 (mnv2PreB2 N w x) := by
  rw [mnv2PreB3, Function.comp_apply]
theorem mnv2PreB4_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB4 N w x = mnv2StridedB N 28 28 w.b4 (mnv2PreB3 N w x) := by
  rw [mnv2PreB4, Function.comp_apply]
theorem mnv2PreB5_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB5 N w x = mnv2ResidB N 28 28 w.b5 (mnv2PreB4 N w x) := by
  rw [mnv2PreB5, Function.comp_apply]
theorem mnv2PreB6_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB6 N w x = mnv2ResidB N 28 28 w.b6 (mnv2PreB5 N w x) := by
  rw [mnv2PreB6, Function.comp_apply]
theorem mnv2PreB7_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB7 N w x = mnv2StridedB N 14 14 w.b7 (mnv2PreB6 N w x) := by
  rw [mnv2PreB7, Function.comp_apply]
theorem mnv2PreB8_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB8 N w x = mnv2ResidB N 14 14 w.b8 (mnv2PreB7 N w x) := by
  rw [mnv2PreB8, Function.comp_apply]
theorem mnv2PreB9_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB9 N w x = mnv2ResidB N 14 14 w.b9 (mnv2PreB8 N w x) := by
  rw [mnv2PreB9, Function.comp_apply]
theorem mnv2PreB10_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB10 N w x = mnv2ResidB N 14 14 w.b10 (mnv2PreB9 N w x) := by
  rw [mnv2PreB10, Function.comp_apply]
theorem mnv2PreB11_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB11 N w x = mnv2ExpOnlyB N 14 14 w.b11 (mnv2PreB10 N w x) := by
  rw [mnv2PreB11, Function.comp_apply]
theorem mnv2PreB12_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB12 N w x = mnv2ResidB N 14 14 w.b12 (mnv2PreB11 N w x) := by
  rw [mnv2PreB12, Function.comp_apply]
theorem mnv2PreB13_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB13 N w x = mnv2ResidB N 14 14 w.b13 (mnv2PreB12 N w x) := by
  rw [mnv2PreB13, Function.comp_apply]
theorem mnv2PreB14_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB14 N w x = mnv2StridedB N 7 7 w.b14 (mnv2PreB13 N w x) := by
  rw [mnv2PreB14, Function.comp_apply]
theorem mnv2PreB15_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB15 N w x = mnv2ResidB N 7 7 w.b15 (mnv2PreB14 N w x) := by
  rw [mnv2PreB15, Function.comp_apply]
theorem mnv2PreB16_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB16 N w x = mnv2ResidB N 7 7 w.b16 (mnv2PreB15 N w x) := by
  rw [mnv2PreB16, Function.comp_apply]
theorem mnv2PreB17_apply (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mnv2PreB17 N w x = mnv2ExpOnlyB N 7 7 w.b17 (mnv2PreB16 N w x) := by
  rw [mnv2PreB17, Function.comp_apply]

/-- ⭐ **The committed nested-application forward IS the layered chain the VJP is stated on** —
    the batched peer of `mobilenetv2ForwardPaper_eq_chain`, and what lets the VJP be about
    `mobilenetv2ForwardB_full` rather than about a re-spelling of it. -/
theorem mobilenetv2ForwardB_full_eq_chain (N : Nat) {nCls : Nat} (w : MNV2BWeights nCls)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112)))) :
    mobilenetv2ForwardB_full N w x
      = (mnv2HeadB N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb ∘ mnv2PreB17 N w) x := by
  rw [mobilenetv2ForwardB_full, Function.comp_apply, mnv2PreB17_apply, mnv2PreB16_apply, mnv2PreB15_apply, mnv2PreB14_apply, mnv2PreB13_apply, mnv2PreB12_apply, mnv2PreB11_apply, mnv2PreB10_apply, mnv2PreB9_apply, mnv2PreB8_apply, mnv2PreB7_apply, mnv2PreB6_apply, mnv2PreB5_apply, mnv2PreB4_apply, mnv2PreB3_apply, mnv2PreB2_apply, mnv2PreB1_apply, mnv2PreB0_apply]


/-- ⭐⭐ **Public correctness theorem**: the seventeen-bottleneck batch-BN backward equals the
    `pdiv`-contracted Jacobian of `mobilenetv2ForwardB_full` ITSELF — the committed
    nested-application forward `MobileNetV2FullB.lean` defines and
    `mobilenetv2FwdGraphB_full_faithful` proves the typed graph denotes — not of the layered chain
    the VJP is assembled on. Tied back through `mobilenetv2ForwardB_full_eq_chain`. -/
theorem mobilenetv2ForwardB_full_has_vjp_at_correct (N : Nat) {nCls : Nat}
    (w : MNV2BWeights nCls)
    (hsε : 0 < w.sε) (hhε : 0 < w.hε)
    (qb1 : IVNoExpPos w.b1)
    (qb2 : IVPos w.b2)
    (qb3 : IVPos w.b3)
    (qb4 : IVPos w.b4)
    (qb5 : IVPos w.b5)
    (qb6 : IVPos w.b6)
    (qb7 : IVPos w.b7)
    (qb8 : IVPos w.b8)
    (qb9 : IVPos w.b9)
    (qb10 : IVPos w.b10)
    (qb11 : IVPos w.b11)
    (qb12 : IVPos w.b12)
    (qb13 : IVPos w.b13)
    (qb14 : IVPos w.b14)
    (qb15 : IVPos w.b15)
    (qb16 : IVPos w.b16)
    (qb17 : IVPos w.b17)
    (x : Vec (N * (3 * (2 * 112) * (2 * 112))))
    (h_stem : MNV2StemSmoothAtB N 112 112 w.sW w.sb w.sε w.sγ w.sβ x)
    (sb1 : IVNoExpSmoothAtB N 112 112 w.b1 (mnv2PreB0 N w x))
    (sb2 : IVStridedSmoothAtB N 56 56 w.b2 (mnv2PreB1 N w x))
    (sb3 : IVSmoothAtB N 56 56 w.b3 (mnv2PreB2 N w x))
    (sb4 : IVStridedSmoothAtB N 28 28 w.b4 (mnv2PreB3 N w x))
    (sb5 : IVSmoothAtB N 28 28 w.b5 (mnv2PreB4 N w x))
    (sb6 : IVSmoothAtB N 28 28 w.b6 (mnv2PreB5 N w x))
    (sb7 : IVStridedSmoothAtB N 14 14 w.b7 (mnv2PreB6 N w x))
    (sb8 : IVSmoothAtB N 14 14 w.b8 (mnv2PreB7 N w x))
    (sb9 : IVSmoothAtB N 14 14 w.b9 (mnv2PreB8 N w x))
    (sb10 : IVSmoothAtB N 14 14 w.b10 (mnv2PreB9 N w x))
    (sb11 : IVSmoothAtB N 14 14 w.b11 (mnv2PreB10 N w x))
    (sb12 : IVSmoothAtB N 14 14 w.b12 (mnv2PreB11 N w x))
    (sb13 : IVSmoothAtB N 14 14 w.b13 (mnv2PreB12 N w x))
    (sb14 : IVStridedSmoothAtB N 7 7 w.b14 (mnv2PreB13 N w x))
    (sb15 : IVSmoothAtB N 7 7 w.b15 (mnv2PreB14 N w x))
    (sb16 : IVSmoothAtB N 7 7 w.b16 (mnv2PreB15 N w x))
    (sb17 : IVSmoothAtB N 7 7 w.b17 (mnv2PreB16 N w x))
    (h_head : MNV2HeadSmoothAtB N 7 7 w.hW w.hb w.hε w.hγ w.hβ (mnv2PreB17 N w x))
    (dy : Vec (N * nCls)) (i : Fin (N * (3 * (2 * 112) * (2 * 112)))) :
    (mobilenetv2ForwardB_full_has_vjp_at N w hsε hhε qb1 qb2 qb3 qb4 qb5 qb6 qb7 qb8 qb9 qb10 qb11 qb12 qb13 qb14 qb15 qb16 qb17 x h_stem sb1 sb2 sb3 sb4 sb5 sb6 sb7 sb8 sb9 sb10 sb11 sb12 sb13 sb14 sb15 sb16 sb17 h_head).backward dy i =
      ∑ j : Fin (N * nCls), pdiv (mobilenetv2ForwardB_full N w) x i j * dy j := by
  have h := (mobilenetv2ForwardB_full_has_vjp_at N w hsε hhε qb1 qb2 qb3 qb4 qb5 qb6 qb7 qb8 qb9 qb10 qb11 qb12 qb13 qb14 qb15 qb16 qb17 x h_stem sb1 sb2 sb3 sb4 sb5 sb6 sb7 sb8 sb9 sb10 sb11 sb12 sb13 sb14 sb15 sb16 sb17 h_head).correct dy i
  rwa [show mobilenetv2ForwardB_full N w
      = mnv2HeadB N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW w.fcb ∘ mnv2PreB17 N w
    from funext (mobilenetv2ForwardB_full_eq_chain N w)]

end Proofs
