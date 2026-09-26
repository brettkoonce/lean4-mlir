import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StepTieB
import LeanMlir.Proofs.Foundation.DataParallel.SyncKit

/-! # MobileNetV2's data-parallel step at synchronised BatchNorm is the single-device step at `R·N`

`MobileNetV2StepTieB.lean` threads the label-smoothed loss cotangent down the batch-BN
backward chain on one device and ties every parameter gradient node to the certified gradient at
that chain's cotangent.
This is its data-parallel twin, for the render `MobileNetV2RenderB` emits at `replicas > 1`: `R`
replicas at batch `N`, every one of the 52 BatchNorms synchronised, every parameter gradient
all-reduced by its mean. The capstone `mnv2_net_syncTiedB` says that, for every parameter the
render emits,

    mean over the R replicas of replica r's gradient node, loss divided by B
      = the single-device gradient node at the global batch R·N, loss divided by R·B

— the gradient node `mnv2_net_tiedB` at `N := R·N` ties to the certified gradient. The right-hand
side is the existing single-device chain at `N := R·N`, so the spec has not moved.

## The same four steps as ResNet-34's twin

`ResNet34SyncStepTieB.lean` is the template, and everything net-agnostic is imported from it
rather than restated: the replica BN link `bnSyncInB` and its shard lemma, the `ConvWSync` /
`BnSync` / `DenseSync` statements and their `*_of_scaled` closers, the divisor step
`replicaLossCot_eq`, and the homogeneity of `bnInB` and `cInB`. The inverted-residual pieces it
shares with EfficientNet-B0 — the depthwise, GAP and dense input-VJPs, the depthwise and
XLA-`SAME` stem weight collectives — come from `DataParallel.SyncKit`.

1. **Sharding** — each replica's backward chain, handed its shard of a global cotangent, computes
   the shard of the global chain. The relu6 mask is pointwise, so it shards by `rfl`; the
   conv, depthwise, XLA-`SAME` strided depthwise, GAP and dense input-VJPs are per-example maps;
   every BN link is `bnSyncInB`.
2. **The collectives** — the mean over replicas of each replica's gradient node is `1/R` of the
   global node at the global cotangent. MobileNetV2 adds three kinds the ResNet-34 kit does not
   have: the XLA-`SAME` strided conv weight (the stem), the depthwise weight and its XLA-`SAME`
   strided peer (§4).
3. **Homogeneity** — the single-device chain and its gradient nodes are linear in the loss
   cotangent (§1): `R ×` the cotangent gives `R ×` every node.
4. **The divisor** — replica `r`'s loss cotangent is `R ×` its shard of the global one, and that
   `R` cancels the collective's `1/R` at every parameter.

## What the DP render emits, and what is tied

At the committed `convBias := false`, `MobileNetV2RenderB` emits 158 parameter gradients — stem 3
(`sW`, `sg`, `sbt`), `b1` 6, sixteen blocks × 9 (`eW eg ebt dW dg dbt pW pg pbt`), head 3 (`hW`,
`hg`, `hbt`), dense 2 (`Wd`, `bd`) — and the capstone ties all 158. The 52 conv, depthwise and
project BIAS nodes the single-device tie also states are not emitted at `convBias := false` and
are not tied here (`mnv2_net_tiedB` keeps them for the flag).

## What is NOT claimed

The replicas' saved forward activations enter as the shards of the single-device forward's
(`batchShard r (mnv2PreB{k} (R*N) w X)`); that the sync forward graph computes exactly those is
`StableHLO.mobilenetv2FwdGraphSyncFull_shard`, the forward half. That the replicas' inputs are
the shards of one batch is the driver's. The lowerer's `all_reduce` is trusted as every other
op's lowering is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.MobileNetV2SyncTieB

open scoped BigOperators
open Proofs.BackLinks (reassocB cInB dInB gapInB)
open Proofs.BackLinks (bnInB unrowB rowB)
open Proofs.SyncKit
open Proofs.SyncKit
open Proofs.MobileNetV2TieB

-- ════════════════════════════════════════════════════════════════
-- § 1. Homogeneity — the single-device chain is linear in its cotangent
-- ════════════════════════════════════════════════════════════════

/-- The two-sided relu6 mask is linear in the cotangent it gates. -/
theorem relu6MaskB_smul (n : Nat) (pre : Vec n) : IsHomog (relu6MaskB n pre) := by
  intro s dy
  funext i
  unfold relu6MaskB
  split_ifs <;> simp

theorem dStridedXlaInB_smul (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    : IsHomog (dStridedXlaInB N (h := h) (w := w) W b) :=
  batchMap_smul _ (HasVJP.backward_smul _ _)

/-! The block cotangents, each one line from the previous link's. -/

theorem mnv2NoExpCotPc_smul (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2NoExpCotPc N h w p xin) :=
  bnInB_smul _ _ _ _ _ _ _

theorem mnv2NoExpCotDn_smul (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2NoExpCotDn N h w p xin) := by
  intro s dy
  unfold mnv2NoExpCotDn; rw [mnv2NoExpCotPc_smul, cInB_smul, relu6MaskB_smul]

theorem mnv2NoExpCotDc_smul (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2NoExpCotDc N h w p xin) := by
  intro s dy
  unfold mnv2NoExpCotDc; rw [mnv2NoExpCotDn_smul, bnInB_smul]

theorem mnv2NoExpCotIn_smul (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2NoExpCotIn N h w p xin) := by
  intro s dy
  unfold mnv2NoExpCotIn; rw [mnv2NoExpCotDc_smul, dInB_smul]

theorem mnv2CotPc_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2CotPc N h w p xin) :=
  bnInB_smul _ _ _ _ _ _ _

theorem mnv2CotDn_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2CotDn N h w p xin) := by
  intro s dy
  unfold mnv2CotDn; rw [mnv2CotPc_smul, cInB_smul, relu6MaskB_smul]

theorem mnv2CotDc_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2CotDc N h w p xin) := by
  intro s dy
  unfold mnv2CotDc; rw [mnv2CotDn_smul, bnInB_smul]

theorem mnv2CotEn_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2CotEn N h w p xin) := by
  intro s dy
  unfold mnv2CotEn; rw [mnv2CotDc_smul, dInB_smul, relu6MaskB_smul]

theorem mnv2CotEc_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2CotEc N h w p xin) := by
  intro s dy
  unfold mnv2CotEc; rw [mnv2CotEn_smul, bnInB_smul]

theorem mnv2CotInBody_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * h * w))) : IsHomog (mnv2CotInBody N h w p xin) := by
  intro s dy
  unfold mnv2CotInBody; rw [mnv2CotEc_smul, cInB_smul]

theorem mnv2ResidCotIn_smul (N h w : Nat) {c mid : Nat} (p : IVW c mid c)
    (xin : Vec (N * (c * h * w))) : IsHomog (mnv2ResidCotIn N h w p xin) := by
  intro s dy
  unfold mnv2ResidCotIn
  rw [mnv2CotInBody_smul]
  funext i
  ring

theorem mnv2SCotPc_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (mnv2SCotPc N h w p xin) :=
  bnInB_smul _ _ _ _ _ _ _

theorem mnv2SCotDn_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (mnv2SCotDn N h w p xin) := by
  intro s dy
  unfold mnv2SCotDn; rw [mnv2SCotPc_smul, cInB_smul, relu6MaskB_smul]

theorem mnv2SCotDc_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (mnv2SCotDc N h w p xin) := by
  intro s dy
  unfold mnv2SCotDc; rw [mnv2SCotDn_smul, bnInB_smul]

theorem mnv2SCotEn_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (mnv2SCotEn N h w p xin) := by
  intro s dy
  unfold mnv2SCotEn; rw [mnv2SCotDc_smul, dStridedXlaInB_smul, relu6MaskB_smul]

theorem mnv2SCotEc_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (mnv2SCotEc N h w p xin) := by
  intro s dy
  unfold mnv2SCotEc; rw [mnv2SCotEn_smul, bnInB_smul]

theorem mnv2StridedCotIn_smul (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (mnv2StridedCotIn N h w p xin) := by
  intro s dy
  unfold mnv2StridedCotIn; rw [mnv2SCotEc_smul, cInB_smul]

theorem mnv2StemCotN_smul (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (mnv2StemCotN N h w Ws bs εs γs βs x) :=
  relu6MaskB_smul _ _

theorem mnv2StemCotC_smul (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (mnv2StemCotC N h w Ws bs εs γs βs x) := by
  intro s dy
  unfold mnv2StemCotC; rw [mnv2StemCotN_smul, bnInB_smul]

theorem mnv2HeadCotHr_smul (N h w : Nat) {oc nCls : Nat} (Wd : Mat oc nCls) :
    IsHomog (mnv2HeadCotHr N h w Wd) := by
  intro s g
  unfold mnv2HeadCotHr mnv2HeadCotGapIn; rw [rowDenseBackFlat_smul, gapInB_smul]

theorem mnv2HeadCotHn_smul (N h w : Nat) {ic oc nCls : Nat} (Wh : Kernel4 oc ic 1 1) (bh : Vec oc)
    (εh : ℝ) (γh βh : Vec oc) (Wd : Mat oc nCls) (xin : Vec (N * (ic * h * w))) :
    IsHomog (mnv2HeadCotHn N h w Wh bh εh γh βh Wd xin) := by
  intro s g
  unfold mnv2HeadCotHn; rw [mnv2HeadCotHr_smul, relu6MaskB_smul]

theorem mnv2HeadCotHc_smul (N h w : Nat) {ic oc nCls : Nat} (Wh : Kernel4 oc ic 1 1) (bh : Vec oc)
    (εh : ℝ) (γh βh : Vec oc) (Wd : Mat oc nCls) (xin : Vec (N * (ic * h * w))) :
    IsHomog (mnv2HeadCotHc N h w Wh bh εh γh βh Wd xin) := by
  intro s g
  unfold mnv2HeadCotHc; rw [mnv2HeadCotHn_smul, bnInB_smul]

theorem mnv2HeadCotBlk_smul (N h w : Nat) {ic oc nCls : Nat} (Wh : Kernel4 oc ic 1 1) (bh : Vec oc)
    (εh : ℝ) (γh βh : Vec oc) (Wd : Mat oc nCls) (xin : Vec (N * (ic * h * w))) :
    IsHomog (mnv2HeadCotBlk N h w Wh bh εh γh βh Wd xin) := by
  intro s g
  unfold mnv2HeadCotBlk; rw [mnv2HeadCotHc_smul, cInB_smul]

/-! The gradient nodes MobileNetV2 adds to the ResNet-34 kit. -/

theorem depthwiseStridedXlaWeightGradB_smul {N c h w kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (s : ℝ) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedXlaWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.depthwiseStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [denStep, denStepApp, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [HasVJP.backward_smul]

-- ════════════════════════════════════════════════════════════════
-- § 2. Sharding — every non-BN link of the chain commutes with the batch cut
-- ════════════════════════════════════════════════════════════════

theorem dStridedXlaInB_shard {R N : Nat} {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (DY : Vec ((R * N) * (c * h * w))) (r : Fin R) :
    dStridedXlaInB N (h := h) (w := w) W b (batchShard R N (c * h * w) DY r)
      = batchShard R N (c * (2 * h) * (2 * w)) (dStridedXlaInB (R * N) (h := h) (w := w) W b DY) r :=
  (batchShard_batchMap _ DY r).symm

theorem mnv2HeadCotHr_shard {R N : Nat} (h w : Nat) {oc nCls : Nat} (Wd : Mat oc nCls)
    (G : Vec ((R * N) * nCls)) (r : Fin R) :
    mnv2HeadCotHr N h w Wd (batchShard R N nCls G r)
      = batchShard R N (oc * h * w) (mnv2HeadCotHr (R * N) h w Wd G) r := by
  unfold mnv2HeadCotHr mnv2HeadCotGapIn
  rw [rowDenseBackFlat_shard, gapInB_shard]

-- ════════════════════════════════════════════════════════════════
-- § 3. The replica chain, block by block, and its shard lemmas
--   Saved activations are the shards of the single-device forward's (the forward half,
--   `MobileNetV2SyncB`, is what says a replica computes exactly those); every cotangent is the
--   replica's own, from the family `dys` of block-output cotangents. Every BatchNorm backward is
--   `bnSyncInB` — the render's `bnSyncDyStatsB` → all-reduce → `bnSyncBack`.
-- ════════════════════════════════════════════════════════════════

section NoExp
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
  (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))

/-- `b1`, replica `r`: the project BatchNorm's sync backward of the block-output cotangent (the
    linear bottleneck has no activation after `project`). Feeds `pW`. -/
noncomputable def mnv2NoExpSyncCotPc (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.pε p.pγ
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv p.pW p.pb)
      (dwbrB (R * N) (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ XIN)) r)
    dys r

/-- `b1`, replica `r`: the depthwise BN's output cotangent. Feeds `dg`/`dbt`. -/
noncomputable def mnv2NoExpSyncCotDn (r : Fin R) : Vec (N * (ic * h * w)) :=
  relu6MaskB (N * (ic * h * w))
    (batchShard R N (ic * h * w)
      (bnBatchLA (R * N) ic h w p.dε p.dγ p.dβ (batchMap (R * N) (depthwiseFlat p.dW p.db) XIN)) r)
    (cInB N p.pW p.pb (mnv2NoExpSyncCotPc R hR N h w p XIN dys r))

/-- `b1`, replica `r`: the depthwise conv's output cotangent. Feeds `dW`. -/
noncomputable def mnv2NoExpSyncCotDc (r : Fin R) : Vec (N * (ic * h * w)) :=
  bnSyncInB R hR N ic h w p.dε p.dγ
    (fun r => batchShard R N (ic * h * w) (batchMap (R * N) (depthwiseFlat p.dW p.db) XIN) r)
    (mnv2NoExpSyncCotDn R hR N h w p XIN dys) r

/-- `b1`, replica `r`: the block-INPUT cotangent, handed to the stem. -/
noncomputable def mnv2NoExpSyncCotIn (r : Fin R) : Vec (N * (ic * h * w)) :=
  dInB N p.dW p.db (mnv2NoExpSyncCotDc R hR N h w p XIN dys r)

end NoExp

section NoExpShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (p : IVWNoExp ic oc) (XIN : Vec ((R * N) * (ic * h * w)))
  (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
  (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hN hh hw hdys

theorem mnv2NoExpSyncCotPc_shard (r : Fin R) :
    mnv2NoExpSyncCotPc R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (mnv2NoExpCotPc (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) hdys r

theorem mnv2NoExpSyncCotDn_shard (r : Fin R) :
    mnv2NoExpSyncCotDn R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (mnv2NoExpCotDn (R * N) h w p XIN DY) r := by
  unfold mnv2NoExpSyncCotDn
  rw [mnv2NoExpSyncCotPc_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

theorem mnv2NoExpSyncCotDc_shard (r : Fin R) :
    mnv2NoExpSyncCotDc R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (mnv2NoExpCotDc (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N ic h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (mnv2NoExpSyncCotDn_shard R hR N h w hN hh hw p XIN dys DY hdys) r

theorem mnv2NoExpSyncCotIn_shard (r : Fin R) :
    mnv2NoExpSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (mnv2NoExpCotIn (R * N) h w p XIN DY) r := by
  unfold mnv2NoExpSyncCotIn
  rw [mnv2NoExpSyncCotDc_shard R hR N h w hN hh hw p XIN dys DY hdys, dInB_shard]
  rfl

end NoExpShard

section Stride1
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
  (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))

/-- Stride-1 body, replica `r`: the project BatchNorm's sync backward of the block-output
    cotangent. Feeds `pW`. -/
noncomputable def mnv2SyncCotPc (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.pε p.pγ
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv p.pW p.pb)
      (dwbrB (R * N) (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ (mnv2XE (R * N) h w p XIN))) r)
    dys r

/-- Stride-1 body, replica `r`: the depthwise BN's output cotangent. Feeds `dg`/`dbt`. -/
noncomputable def mnv2SyncCotDn (r : Fin R) : Vec (N * (mid * h * w)) :=
  relu6MaskB (N * (mid * h * w))
    (batchShard R N (mid * h * w) (bnBatchLA (R * N) mid h w p.dε p.dγ p.dβ
      (batchMap (R * N) (depthwiseFlat p.dW p.db) (mnv2XE (R * N) h w p XIN))) r)
    (cInB N p.pW p.pb (mnv2SyncCotPc R hR N h w p XIN dys r))

/-- Stride-1 body, replica `r`: the depthwise conv's output cotangent. Feeds `dW`. -/
noncomputable def mnv2SyncCotDc (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w p.dε p.dγ
    (fun r => batchShard R N (mid * h * w)
      (batchMap (R * N) (depthwiseFlat p.dW p.db) (mnv2XE (R * N) h w p XIN)) r)
    (mnv2SyncCotDn R hR N h w p XIN dys) r

/-- Stride-1 body, replica `r`: the expand BN's output cotangent. Feeds `eg`/`ebt`. -/
noncomputable def mnv2SyncCotEn (r : Fin R) : Vec (N * (mid * h * w)) :=
  relu6MaskB (N * (mid * h * w))
    (batchShard R N (mid * h * w)
      (bnBatchLA (R * N) mid h w p.eε p.eγ p.eβ (batchMap (R * N) (flatConv p.eW p.eb) XIN)) r)
    (dInB N p.dW p.db (mnv2SyncCotDc R hR N h w p XIN dys r))

/-- Stride-1 body, replica `r`: the expand conv's output cotangent. Feeds `eW`. -/
noncomputable def mnv2SyncCotEc (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w p.eε p.eγ
    (fun r => batchShard R N (mid * h * w) (batchMap (R * N) (flatConv p.eW p.eb) XIN) r)
    (mnv2SyncCotEn R hR N h w p XIN dys) r

/-- Stride-1 body, replica `r`: the body's input cotangent — the whole block-input cotangent for a
    widening (`b11`, `b17`). -/
noncomputable def mnv2SyncCotInBody (r : Fin R) : Vec (N * (ic * h * w)) :=
  cInB N p.eW p.eb (mnv2SyncCotEc R hR N h w p XIN dys r)

end Stride1

/-- Skip block, replica `r`: the body branch plus the identity skip. -/
noncomputable def mnv2ResidSyncCotIn (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid : Nat}
    (p : IVW c mid c) (XIN : Vec ((R * N) * (c * h * w))) (dys : Fin R → Vec (N * (c * h * w)))
    (r : Fin R) : Vec (N * (c * h * w)) :=
  fun i => mnv2SyncCotInBody R hR N h w p XIN dys r i + dys r i

section Stride1Shard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (p : IVW ic mid oc) (XIN : Vec ((R * N) * (ic * h * w)))
  (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
  (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hN hh hw hdys

theorem mnv2SyncCotPc_shard (r : Fin R) :
    mnv2SyncCotPc R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (mnv2CotPc (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) hdys r

theorem mnv2SyncCotDn_shard (r : Fin R) :
    mnv2SyncCotDn R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (mnv2CotDn (R * N) h w p XIN DY) r := by
  unfold mnv2SyncCotDn
  rw [mnv2SyncCotPc_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

theorem mnv2SyncCotDc_shard (r : Fin R) :
    mnv2SyncCotDc R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (mnv2CotDc (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (mnv2SyncCotDn_shard R hR N h w hN hh hw p XIN dys DY hdys) r

theorem mnv2SyncCotEn_shard (r : Fin R) :
    mnv2SyncCotEn R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (mnv2CotEn (R * N) h w p XIN DY) r := by
  unfold mnv2SyncCotEn
  rw [mnv2SyncCotDc_shard R hR N h w hN hh hw p XIN dys DY hdys, dInB_shard]
  rfl

theorem mnv2SyncCotEc_shard (r : Fin R) :
    mnv2SyncCotEc R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (mnv2CotEc (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (mnv2SyncCotEn_shard R hR N h w hN hh hw p XIN dys DY hdys) r

theorem mnv2SyncCotInBody_shard (r : Fin R) :
    mnv2SyncCotInBody R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (mnv2CotInBody (R * N) h w p XIN DY) r := by
  unfold mnv2SyncCotInBody
  rw [mnv2SyncCotEc_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

end Stride1Shard

theorem mnv2ResidSyncCotIn_shard (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (p : IVW c mid c) (XIN : Vec ((R * N) * (c * h * w)))
    (dys : Fin R → Vec (N * (c * h * w))) (DY : Vec ((R * N) * (c * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (c * h * w) DY r) (r : Fin R) :
    mnv2ResidSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (c * h * w) (mnv2ResidCotIn (R * N) h w p XIN DY) r := by
  unfold mnv2ResidSyncCotIn
  rw [mnv2SyncCotInBody_shard R hR N h w hN hh hw p XIN dys DY hdys, hdys]
  rfl

section Stride2
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat} (p : IVW ic mid oc)
  (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))

/-- Stride-2 block, replica `r`: the project BatchNorm's sync backward. Feeds `pW`. -/
noncomputable def mnv2SSyncCotPc (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.pε p.pγ
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv p.pW p.pb)
      (dwbrBstrided (R * N) (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ
        (mnv2XES (R * N) h w p XIN))) r)
    dys r

/-- Stride-2 block, replica `r`: the strided depthwise BN's output cotangent. Feeds `dg`/`dbt`. -/
noncomputable def mnv2SSyncCotDn (r : Fin R) : Vec (N * (mid * h * w)) :=
  relu6MaskB (N * (mid * h * w))
    (batchShard R N (mid * h * w) (bnBatchLA (R * N) mid h w p.dε p.dγ p.dβ
      (batchMap (R * N) (depthwiseStride2FlatXla p.dW p.db) (mnv2XES (R * N) h w p XIN))) r)
    (cInB N p.pW p.pb (mnv2SSyncCotPc R hR N h w p XIN dys r))

/-- Stride-2 block, replica `r`: the strided depthwise conv's output cotangent. Feeds `dW`. -/
noncomputable def mnv2SSyncCotDc (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w p.dε p.dγ
    (fun r => batchShard R N (mid * h * w)
      (batchMap (R * N) (depthwiseStride2FlatXla p.dW p.db) (mnv2XES (R * N) h w p XIN)) r)
    (mnv2SSyncCotDn R hR N h w p XIN dys) r

/-- Stride-2 block, replica `r`: the expand BN's output cotangent at the `2h x 2w` grid — the
    strided depthwise's input-VJP (which upsamples) masked by the expand relu6. -/
noncomputable def mnv2SSyncCotEn (r : Fin R) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  relu6MaskB (N * (mid * (2 * h) * (2 * w)))
    (batchShard R N (mid * (2 * h) * (2 * w))
      (bnBatchLA (R * N) mid (2 * h) (2 * w) p.eε p.eγ p.eβ
        (batchMap (R * N) (flatConv p.eW p.eb) XIN)) r)
    (dStridedXlaInB N p.dW p.db (mnv2SSyncCotDc R hR N h w p XIN dys r))

/-- Stride-2 block, replica `r`: the expand conv's output cotangent. Feeds `eW`. -/
noncomputable def mnv2SSyncCotEc (r : Fin R) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  bnSyncInB R hR N mid (2 * h) (2 * w) p.eε p.eγ
    (fun r => batchShard R N (mid * (2 * h) * (2 * w))
      (batchMap (R * N) (flatConv p.eW p.eb) XIN) r)
    (mnv2SSyncCotEn R hR N h w p XIN dys) r

/-- Stride-2 block, replica `r`: the block-INPUT cotangent. -/
noncomputable def mnv2StridedSyncCotIn (r : Fin R) : Vec (N * (ic * (2 * h) * (2 * w))) :=
  cInB N p.eW p.eb (mnv2SSyncCotEc R hR N h w p XIN dys r)

end Stride2

section Stride2Shard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (p : IVW ic mid oc) (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
  (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
  (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hN hh hw hdys

theorem mnv2SSyncCotPc_shard (r : Fin R) :
    mnv2SSyncCotPc R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (mnv2SCotPc (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) hdys r

theorem mnv2SSyncCotDn_shard (r : Fin R) :
    mnv2SSyncCotDn R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (mnv2SCotDn (R * N) h w p XIN DY) r := by
  unfold mnv2SSyncCotDn
  rw [mnv2SSyncCotPc_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

theorem mnv2SSyncCotDc_shard (r : Fin R) :
    mnv2SSyncCotDc R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (mnv2SCotDc (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (mnv2SSyncCotDn_shard R hR N h w hN hh hw p XIN dys DY hdys) r

theorem mnv2SSyncCotEn_shard (r : Fin R) :
    mnv2SSyncCotEn R hR N h w p XIN dys r
      = batchShard R N (mid * (2 * h) * (2 * w)) (mnv2SCotEn (R * N) h w p XIN DY) r := by
  unfold mnv2SSyncCotEn
  rw [mnv2SSyncCotDc_shard R hR N h w hN hh hw p XIN dys DY hdys, dStridedXlaInB_shard]
  rfl

theorem mnv2SSyncCotEc_shard (r : Fin R) :
    mnv2SSyncCotEc R hR N h w p XIN dys r
      = batchShard R N (mid * (2 * h) * (2 * w)) (mnv2SCotEc (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid (2 * h) (2 * w)
    (nhw_ne_zero hN (Nat.mul_pos (by norm_num) hh) (Nat.mul_pos (by norm_num) hw))
    _ _ _ _ _ _ (fun _ => rfl) (mnv2SSyncCotEn_shard R hR N h w hN hh hw p XIN dys DY hdys) r

theorem mnv2StridedSyncCotIn_shard (r : Fin R) :
    mnv2StridedSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w)) (mnv2StridedCotIn (R * N) h w p XIN DY) r := by
  unfold mnv2StridedSyncCotIn
  rw [mnv2SSyncCotEc_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

end Stride2Shard

section Stem
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW)
  (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
  (dys : Fin R → Vec (N * (oc * h * w)))

/-- Stem, replica `r`: the stem relu6's mask of the cotangent `b1` hands down. Feeds `sg`/`sbt`. -/
noncomputable def mnv2StemSyncCotN (r : Fin R) : Vec (N * (oc * h * w)) :=
  relu6MaskB (N * (oc * h * w))
    (batchShard R N (oc * h * w)
      (bnBatchLA (R * N) oc h w εs γs βs (batchMap (R * N) (flatConvStride2Xla Ws bs) X)) r)
    (dys r)

/-- Stem, replica `r`: the stem BatchNorm's sync backward. Feeds `sW`. -/
noncomputable def mnv2StemSyncCotC (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w εs γs
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConvStride2Xla Ws bs) X) r)
    (mnv2StemSyncCotN R N h w Ws bs εs γs βs X dys) r

end Stem

section StemShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
  (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
  (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hdys

theorem mnv2StemSyncCotN_shard (r : Fin R) :
    mnv2StemSyncCotN R N h w Ws bs εs γs βs X dys r
      = batchShard R N (oc * h * w) (mnv2StemCotN (R * N) h w Ws bs εs γs βs X DY) r := by
  unfold mnv2StemSyncCotN; rw [hdys]; rfl

include hN hh hw in
theorem mnv2StemSyncCotC_shard (r : Fin R) :
    mnv2StemSyncCotC R hR N h w Ws bs εs γs βs X dys r
      = batchShard R N (oc * h * w) (mnv2StemCotC (R * N) h w Ws bs εs γs βs X DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (mnv2StemSyncCotN_shard R N h w Ws bs εs γs βs X dys DY hdys) r

end StemShard

section Head
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc nCls : Nat} (Wh : Kernel4 oc ic 1 1)
  (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc) (Wd : Mat oc nCls)
  (XIN : Vec ((R * N) * (ic * h * w))) (gs : Fin R → Vec (N * nCls))

/-- Head, replica `r`: the head relu6's mask of the GAP backward of the classifier's input-VJP of
    this replica's loss cotangent. Feeds `hg`/`hbt`. -/
noncomputable def mnv2HeadSyncCotHn (r : Fin R) : Vec (N * (oc * h * w)) :=
  relu6MaskB (N * (oc * h * w))
    (batchShard R N (oc * h * w)
      (bnBatchLA (R * N) oc h w εh γh βh (batchMap (R * N) (flatConv Wh bh) XIN)) r)
    (mnv2HeadCotHr N h w Wd (gs r))

/-- Head, replica `r`: the head BatchNorm's sync backward. Feeds `hW`. -/
noncomputable def mnv2HeadSyncCotHc (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w εh γh
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv Wh bh) XIN) r)
    (mnv2HeadSyncCotHn R N h w Wh bh εh γh βh Wd XIN gs) r

/-- Head, replica `r`: the cotangent handed to `b17`. -/
noncomputable def mnv2HeadSyncCotBlk (r : Fin R) : Vec (N * (ic * h * w)) :=
  cInB N Wh bh (mnv2HeadSyncCotHc R hR N h w Wh bh εh γh βh Wd XIN gs r)

end Head

section HeadShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc nCls : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
  (Wd : Mat oc nCls) (XIN : Vec ((R * N) * (ic * h * w))) (gs : Fin R → Vec (N * nCls))
  (G : Vec ((R * N) * nCls)) (hgs : ∀ r, gs r = batchShard R N nCls G r)
include hgs

theorem mnv2HeadSyncCotHn_shard (r : Fin R) :
    mnv2HeadSyncCotHn R N h w Wh bh εh γh βh Wd XIN gs r
      = batchShard R N (oc * h * w) (mnv2HeadCotHn (R * N) h w Wh bh εh γh βh Wd XIN G) r := by
  unfold mnv2HeadSyncCotHn
  rw [hgs, mnv2HeadCotHr_shard]
  rfl

include hN hh hw in
theorem mnv2HeadSyncCotHc_shard (r : Fin R) :
    mnv2HeadSyncCotHc R hR N h w Wh bh εh γh βh Wd XIN gs r
      = batchShard R N (oc * h * w) (mnv2HeadCotHc (R * N) h w Wh bh εh γh βh Wd XIN G) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (mnv2HeadSyncCotHn_shard R N h w Wh bh εh γh βh Wd XIN gs G hgs) r

include hN hh hw in
theorem mnv2HeadSyncCotBlk_shard (r : Fin R) :
    mnv2HeadSyncCotBlk R hR N h w Wh bh εh γh βh Wd XIN gs r
      = batchShard R N (ic * h * w) (mnv2HeadCotBlk (R * N) h w Wh bh εh γh βh Wd XIN G) r := by
  unfold mnv2HeadSyncCotBlk
  rw [mnv2HeadSyncCotHc_shard R hR N h w hN hh hw Wh bh εh γh βh Wd XIN gs G hgs, cInB_shard]
  rfl

end HeadShard

-- ════════════════════════════════════════════════════════════════
-- § 4. The parameter collectives MobileNetV2 adds
-- ════════════════════════════════════════════════════════════════

/-- **P4 at the XLA-`SAME` strided depthwise weight** (`b2`, `b4`, `b7`, `b14`). -/
theorem den_allReduceMeanF_depthwiseStridedXlaWeightGradB_shard {N c h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec c) (W : DepthwiseKernel c kH kW)
    (X : Vec ((R * N) * (c * (2 * h) * (2 * w)))) (DY : Vec ((R * N) * (c * h * w)))
    (dy : Fin R → SHlo (N * (c * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (c * h * w) DY r) (idx : Fin (c * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .depthwiseStridedXlaWeightGradB xN b
            (batchShard R N (c * (2 * h) * (2 * w)) X r) W (dy r))) idx
      = (1 / (R : ℝ)) * den (.depthwiseStridedXlaWeightGradB xN b X W (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [denStep, denStepApp, hdy]
  shard_sum

-- ════════════════════════════════════════════════════════════════
-- § 5. Per-parameter DP ties for the new kinds
-- ════════════════════════════════════════════════════════════════

/-- The XLA-`SAME` strided depthwise weight, DP-tied. -/
def DepthwiseStridedXlaWSync (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat}
    (t xN cotN : String) (b : Vec c) (X : Vec ((R * N) * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) (cots : Fin R → Vec (N * (c * h * w)))
    (COT : Vec ((R * N) * (c * h * w))) : Prop :=
  ∀ idx : Fin (c * kH * kW),
    den (.allReduceMeanF R hR t [c, 1, kH, kW] (fun r =>
          .depthwiseStridedXlaWeightGradB xN b (batchShard R N (c * (2 * h) * (2 * w)) X r) W
            (.operand cotN (cots r)))) idx
      = den (.depthwiseStridedXlaWeightGradB xN b X W (.operand cotN COT)) idx

theorem depthwiseStridedXlaWSync_of_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat}
    (t xN cotN : String) (b : Vec c) (X : Vec ((R * N) * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) (cots : Fin R → Vec (N * (c * h * w)))
    (COT : Vec ((R * N) * (c * h * w)))
    (hc : ∀ r, cots r = batchShard R N (c * h * w) (fun i => (R : ℝ) * COT i) r) :
    DepthwiseStridedXlaWSync R hR N h w t xN cotN b X W cots COT := by
  intro idx
  rw [den_allReduceMeanF_depthwiseStridedXlaWeightGradB_shard R hR t xN cotN _ b W X
      (fun i => (R : ℝ) * COT i) (fun r => .operand cotN (cots r)) (fun r => hc r) idx,
    depthwiseStridedXlaWeightGradB_smul, inv_mul_R R hR]

-- ════════════════════════════════════════════════════════════════
-- § 6. The per-block DP ties
--   Parameter tags are the render's names without `%` (`b{k}eW`, `b{k}eg`, …); a BatchNorm's
--   statistics collectives are its γ tag with `mu` / `var` (inside `BnSync`).
-- ════════════════════════════════════════════════════════════════

/-- **Stem, DP-tied** — the 3x3/s2 XLA-`SAME` conv weight and its BatchNorm's γ and β. -/
def mnv2StemSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ)
    (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  ConvStridedXlaWSync R hR N h w "sW" xN cotN bs X Ws
      (mnv2StemSyncCotC R hR N h w Ws bs εs γs βs X dys)
      (mnv2StemCotC (R * N) h w Ws bs εs γs βs X DY)
  ∧ BnSync R hR N oc h w "sg" "sbt" vN epsStr cotN εs
      (batchMap (R * N) (flatConvStride2Xla Ws bs) X)
      (mnv2StemSyncCotN R N h w Ws bs εs γs βs X dys)
      (mnv2StemCotN (R * N) h w Ws bs εs γs βs X DY)

theorem mnv2_stem_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    mnv2StemSyncTiedB R hR N h w xN cotN vN epsStr Ws bs εs γs βs X dys DY := by
  refine ⟨?_, ?_⟩
  · exact convStridedXlaWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2StemSyncCotC_shard R hR N h w hN hh hw Ws bs εs γs βs X dys _ hdys,
        mnv2StemCotC_smul])
  · exact bnSync_of_scaled R hR N oc h w (nhw_ne_zero hN hh hw) _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2StemSyncCotN_shard R N h w Ws bs εs γs βs X dys _ hdys, mnv2StemCotN_smul])

/-- **`t = 1` block (b1), DP-tied** — its six emitted collectives: the depthwise weight, the
    depthwise BatchNorm's γ and β, the project weight, the project BatchNorm's γ and β. -/
def mnv2NoExpSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (pfx xN cotN vN epsStr : String) (p : IVWNoExp ic oc) (XIN : Vec ((R * N) * (ic * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  let dc := batchMap (R * N) (depthwiseFlat p.dW p.db) XIN
  let dr := dwbrB (R * N) (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ XIN
  let pc := batchMap (R * N) (flatConv p.pW p.pb) dr
  DepthwiseWSync R hR N h w s!"b{pfx}dW" xN cotN p.db XIN p.dW
      (mnv2NoExpSyncCotDc R hR N h w p XIN dys) (mnv2NoExpCotDc (R * N) h w p XIN DY)
  ∧ BnSync R hR N ic h w s!"b{pfx}dg" s!"b{pfx}dbt" vN epsStr cotN p.dε dc
      (mnv2NoExpSyncCotDn R hR N h w p XIN dys) (mnv2NoExpCotDn (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"b{pfx}pW" xN cotN p.pb dr p.pW
      (mnv2NoExpSyncCotPc R hR N h w p XIN dys) (mnv2NoExpCotPc (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"b{pfx}pg" s!"b{pfx}pbt" vN epsStr cotN p.pε pc dys DY

/-- The scaled-shard invariant through `b1`: replicas at `R ×` the shards of `DY` hand the stem
    `R ×` the shards of the single-device block-input cotangent. -/
theorem mnv2NoExpSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVWNoExp ic oc) (XIN : Vec ((R * N) * (ic * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    mnv2NoExpSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (fun i => (R : ℝ) * mnv2NoExpCotIn (R * N) h w p XIN DY i) r := by
  rw [mnv2NoExpSyncCotIn_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2NoExpCotIn_smul]

theorem mnv2_noexp_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String) (p : IVWNoExp ic oc)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    mnv2NoExpSyncTiedB R hR N h w pfx xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hw
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact depthwiseWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2NoExpSyncCotDc_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2NoExpCotDc_smul])
  · exact bnSync_of_scaled R hR N ic h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2NoExpSyncCotDn_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2NoExpCotDn_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2NoExpSyncCotPc_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2NoExpCotPc_smul])
  · exact bnSync_of_scaled R hR N oc h w hm _ _ _ _ _ _ _ _ _ hdys

/-- **Stride-1 inverted-residual block, DP-tied — its nine emitted collectives** (`eW eg ebt dW dg
    dbt pW pg pbt`). One statement for all twelve stride-1 blocks, skip or widening, exactly as
    `mnv2Stride1TiedB` is: the identity skip changes only the cotangent handed to the previous
    block, never a parameter's. -/
def mnv2Stride1SyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (pfx xN cotN vN epsStr : String) (p : IVW ic mid oc) (XIN : Vec ((R * N) * (ic * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  let ec := batchMap (R * N) (flatConv p.eW p.eb) XIN
  let er := mnv2XE (R * N) h w p XIN
  let dc := batchMap (R * N) (depthwiseFlat p.dW p.db) er
  let dr := dwbrB (R * N) (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ er
  let pc := batchMap (R * N) (flatConv p.pW p.pb) dr
  ConvWSync R hR N h w s!"b{pfx}eW" xN cotN p.eb XIN p.eW
      (mnv2SyncCotEc R hR N h w p XIN dys) (mnv2CotEc (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid h w s!"b{pfx}eg" s!"b{pfx}ebt" vN epsStr cotN p.eε ec
      (mnv2SyncCotEn R hR N h w p XIN dys) (mnv2CotEn (R * N) h w p XIN DY)
  ∧ DepthwiseWSync R hR N h w s!"b{pfx}dW" xN cotN p.db er p.dW
      (mnv2SyncCotDc R hR N h w p XIN dys) (mnv2CotDc (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid h w s!"b{pfx}dg" s!"b{pfx}dbt" vN epsStr cotN p.dε dc
      (mnv2SyncCotDn R hR N h w p XIN dys) (mnv2CotDn (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"b{pfx}pW" xN cotN p.pb dr p.pW
      (mnv2SyncCotPc R hR N h w p XIN dys) (mnv2CotPc (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"b{pfx}pg" s!"b{pfx}pbt" vN epsStr cotN p.pε pc dys DY

/-- The scaled-shard invariant through a widening's body (`b11`, `b17`). -/
theorem mnv2SyncCotInBody_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVW ic mid oc) (XIN : Vec ((R * N) * (ic * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    mnv2SyncCotInBody R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (fun i => (R : ℝ) * mnv2CotInBody (R * N) h w p XIN DY i) r := by
  rw [mnv2SyncCotInBody_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2CotInBody_smul]

/-- The scaled-shard invariant through a skip block. -/
theorem mnv2ResidSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVW c mid c) (XIN : Vec ((R * N) * (c * h * w)))
    (dys : Fin R → Vec (N * (c * h * w))) (DY : Vec ((R * N) * (c * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (c * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    mnv2ResidSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (c * h * w) (fun i => (R : ℝ) * mnv2ResidCotIn (R * N) h w p XIN DY i) r := by
  rw [mnv2ResidSyncCotIn_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2ResidCotIn_smul]

theorem mnv2_stride1_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String) (p : IVW ic mid oc)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    mnv2Stride1SyncTiedB R hR N h w pfx xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hw
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SyncCotEc_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2CotEc_smul])
  · exact bnSync_of_scaled R hR N mid h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SyncCotEn_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2CotEn_smul])
  · exact depthwiseWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SyncCotDc_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2CotDc_smul])
  · exact bnSync_of_scaled R hR N mid h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SyncCotDn_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2CotDn_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SyncCotPc_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2CotPc_smul])
  · exact bnSync_of_scaled R hR N oc h w hm _ _ _ _ _ _ _ _ _ hdys

/-- **Stride-2 downsampling block, DP-tied — its nine emitted collectives** (`b2`, `b4`, `b7`,
    `b14`): the expand half at the `2h x 2w` input grid, the XLA-`SAME` strided depthwise. -/
def mnv2Stride2SyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (pfx xN cotN vN epsStr : String) (p : IVW ic mid oc)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  let ec := batchMap (R * N) (flatConv p.eW p.eb) XIN
  let er := mnv2XES (R * N) h w p XIN
  let dc := batchMap (R * N) (depthwiseStride2FlatXla p.dW p.db) er
  let dr := dwbrBstrided (R * N) (h := h) (w := w) p.dW p.db p.dε p.dγ p.dβ er
  let pc := batchMap (R * N) (flatConv p.pW p.pb) dr
  ConvWSync R hR N (2 * h) (2 * w) s!"b{pfx}eW" xN cotN p.eb XIN p.eW
      (mnv2SSyncCotEc R hR N h w p XIN dys) (mnv2SCotEc (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid (2 * h) (2 * w) s!"b{pfx}eg" s!"b{pfx}ebt" vN epsStr cotN p.eε ec
      (mnv2SSyncCotEn R hR N h w p XIN dys) (mnv2SCotEn (R * N) h w p XIN DY)
  ∧ DepthwiseStridedXlaWSync R hR N h w s!"b{pfx}dW" xN cotN p.db er p.dW
      (mnv2SSyncCotDc R hR N h w p XIN dys) (mnv2SCotDc (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid h w s!"b{pfx}dg" s!"b{pfx}dbt" vN epsStr cotN p.dε dc
      (mnv2SSyncCotDn R hR N h w p XIN dys) (mnv2SCotDn (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"b{pfx}pW" xN cotN p.pb dr p.pW
      (mnv2SSyncCotPc R hR N h w p XIN dys) (mnv2SCotPc (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"b{pfx}pg" s!"b{pfx}pbt" vN epsStr cotN p.pε pc dys DY

/-- The scaled-shard invariant through a stride-2 block. -/
theorem mnv2StridedSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : IVW ic mid oc)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    mnv2StridedSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w))
          (fun i => (R : ℝ) * mnv2StridedCotIn (R * N) h w p XIN DY i) r := by
  rw [mnv2StridedSyncCotIn_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2StridedCotIn_smul]

theorem mnv2_stride2_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String) (p : IVW ic mid oc)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    mnv2Stride2SyncTiedB R hR N h w pfx xN cotN vN epsStr p XIN dys DY := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  have hm := nhw_ne_zero hN hh hw
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N (2 * h) (2 * w) _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SSyncCotEc_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2SCotEc_smul])
  · exact bnSync_of_scaled R hR N mid (2 * h) (2 * w) (nhw_ne_zero hN h2h h2w) _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SSyncCotEn_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2SCotEn_smul])
  · exact depthwiseStridedXlaWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SSyncCotDc_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2SCotDc_smul])
  · exact bnSync_of_scaled R hR N mid h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SSyncCotDn_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2SCotDn_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2SSyncCotPc_shard R hR N h w hN hh hw p XIN dys _ hdys, mnv2SCotPc_smul])
  · exact bnSync_of_scaled R hR N oc h w hm _ _ _ _ _ _ _ _ _ hdys

/-- **Head, DP-tied** — the 1x1 conv weight, its BatchNorm's γ and β, and the classifier's weight
    and bias at the GAP output. -/
def mnv2HeadSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc nCls : Nat}
    (xN cotN vN epsStr : String) (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wd : Mat oc nCls) (XIN : Vec ((R * N) * (ic * h * w))) (gs : Fin R → Vec (N * nCls))
    (G : Vec ((R * N) * nCls)) : Prop :=
  let hc := batchMap (R * N) (flatConv Wh bh) XIN
  let hr := cbrB (R * N) (h := h) (w := w) Wh bh εh γh βh XIN
  let a := batchMap (R * N) (globalAvgPoolFlat oc h w) hr
  ConvWSync R hR N h w "hW" xN cotN bh XIN Wh
      (mnv2HeadSyncCotHc R hR N h w Wh bh εh γh βh Wd XIN gs)
      (mnv2HeadCotHc (R * N) h w Wh bh εh γh βh Wd XIN G)
  ∧ BnSync R hR N oc h w "hg" "hbt" vN epsStr cotN εh hc
      (mnv2HeadSyncCotHn R N h w Wh bh εh γh βh Wd XIN gs)
      (mnv2HeadCotHn (R * N) h w Wh bh εh γh βh Wd XIN G)
  ∧ DenseSync R hR N "Wd" "bd" xN cotN a gs G

/-- The head's cotangent handed to `b17` on a replica, at `R ×` the shards of `G`, is `R ×` the
    shard of the single-device one. -/
theorem mnv2HeadSyncCotBlk_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc nCls : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ)
    (γh βh : Vec oc) (Wd : Mat oc nCls) (XIN : Vec ((R * N) * (ic * h * w)))
    (gs : Fin R → Vec (N * nCls)) (G : Vec ((R * N) * nCls))
    (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r) (r : Fin R) :
    mnv2HeadSyncCotBlk R hR N h w Wh bh εh γh βh Wd XIN gs r
      = batchShard R N (ic * h * w)
          (fun i => (R : ℝ) * mnv2HeadCotBlk (R * N) h w Wh bh εh γh βh Wd XIN G i) r := by
  rw [mnv2HeadSyncCotBlk_shard R hR N h w hN hh hw Wh bh εh γh βh Wd XIN gs _ hgs,
    mnv2HeadCotBlk_smul]

theorem mnv2_head_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc nCls : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (xN cotN vN epsStr : String)
    (Wh : Kernel4 oc ic 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc) (Wd : Mat oc nCls)
    (XIN : Vec ((R * N) * (ic * h * w))) (gs : Fin R → Vec (N * nCls)) (G : Vec ((R * N) * nCls))
    (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r) :
    mnv2HeadSyncTiedB R hR N h w xN cotN vN epsStr Wh bh εh γh βh Wd XIN gs G := by
  refine ⟨?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2HeadSyncCotHc_shard R hR N h w hN hh hw Wh bh εh γh βh Wd XIN gs _ hgs,
        mnv2HeadCotHc_smul])
  · exact bnSync_of_scaled R hR N oc h w (nhw_ne_zero hN hh hw) _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv2HeadSyncCotHn_shard R N h w Wh bh εh γh βh Wd XIN gs _ hgs, mnv2HeadCotHn_smul])
  · exact denseSync_of_scaled R hR N _ _ _ _ _ gs G hgs

-- ════════════════════════════════════════════════════════════════
-- § 7. The whole-net capstone
-- ════════════════════════════════════════════════════════════════

/-- **The whole-net statement, named** — so the capstone (cotangents bound) and its smoothed-CE
    corollary (cotangents instantiated) state exactly one thing. The first 18 `let`s are
    `mnv2_net_tiedB`'s chain at `N := R·N`, driven by the global cotangent `G`; the next 18 are the
    replicas' sync-BN chain, driven by the family `gs`; the 19 conjuncts are one per stage, every
    emitted parameter collective against `mnv2_net_tiedB`'s node at the global batch. -/
def mnv2NetSyncTiedB (R : Nat) (hR : 0 < R) (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : MNV2BWeights nCls) (X : Vec ((R * N) * (3 * (2 * 112) * (2 * 112))))
    (G : Vec ((R * N) * nCls)) (gs : Fin R → Vec (N * nCls)) : Prop :=
  -- ── the single-device chain at the global batch `R·N` (T3's), driven by `G` ──
  let dy17 := mnv2HeadCotBlk (R * N) 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW (mnv2PreB17 (R * N) w X) G
  let dy16 := mnv2CotInBody (R * N) 7 7 w.b17 (mnv2PreB16 (R * N) w X) dy17
  let dy15 := mnv2ResidCotIn (R * N) 7 7 w.b16 (mnv2PreB15 (R * N) w X) dy16
  let dy14 := mnv2ResidCotIn (R * N) 7 7 w.b15 (mnv2PreB14 (R * N) w X) dy15
  let dy13 := mnv2StridedCotIn (R * N) 7 7 w.b14 (mnv2PreB13 (R * N) w X) dy14
  let dy12 := mnv2ResidCotIn (R * N) 14 14 w.b13 (mnv2PreB12 (R * N) w X) dy13
  let dy11 := mnv2ResidCotIn (R * N) 14 14 w.b12 (mnv2PreB11 (R * N) w X) dy12
  let dy10 := mnv2CotInBody (R * N) 14 14 w.b11 (mnv2PreB10 (R * N) w X) dy11
  let dy9 := mnv2ResidCotIn (R * N) 14 14 w.b10 (mnv2PreB9 (R * N) w X) dy10
  let dy8 := mnv2ResidCotIn (R * N) 14 14 w.b9 (mnv2PreB8 (R * N) w X) dy9
  let dy7 := mnv2ResidCotIn (R * N) 14 14 w.b8 (mnv2PreB7 (R * N) w X) dy8
  let dy6 := mnv2StridedCotIn (R * N) 14 14 w.b7 (mnv2PreB6 (R * N) w X) dy7
  let dy5 := mnv2ResidCotIn (R * N) 28 28 w.b6 (mnv2PreB5 (R * N) w X) dy6
  let dy4 := mnv2ResidCotIn (R * N) 28 28 w.b5 (mnv2PreB4 (R * N) w X) dy5
  let dy3 := mnv2StridedCotIn (R * N) 28 28 w.b4 (mnv2PreB3 (R * N) w X) dy4
  let dy2 := mnv2ResidCotIn (R * N) 56 56 w.b3 (mnv2PreB2 (R * N) w X) dy3
  let dy1 := mnv2StridedCotIn (R * N) 56 56 w.b2 (mnv2PreB1 (R * N) w X) dy2
  let cotStem := mnv2NoExpCotIn (R * N) 112 112 w.b1 (mnv2PreB0 (R * N) w X) dy1
  -- ── the replicas' sync-BN chain, driven by the family `gs` ──
  let e17 := mnv2HeadSyncCotBlk R hR N 7 7 w.hW w.hb w.hε w.hγ w.hβ w.fcW
    (mnv2PreB17 (R * N) w X) gs
  let e16 := mnv2SyncCotInBody R hR N 7 7 w.b17 (mnv2PreB16 (R * N) w X) e17
  let e15 := mnv2ResidSyncCotIn R hR N 7 7 w.b16 (mnv2PreB15 (R * N) w X) e16
  let e14 := mnv2ResidSyncCotIn R hR N 7 7 w.b15 (mnv2PreB14 (R * N) w X) e15
  let e13 := mnv2StridedSyncCotIn R hR N 7 7 w.b14 (mnv2PreB13 (R * N) w X) e14
  let e12 := mnv2ResidSyncCotIn R hR N 14 14 w.b13 (mnv2PreB12 (R * N) w X) e13
  let e11 := mnv2ResidSyncCotIn R hR N 14 14 w.b12 (mnv2PreB11 (R * N) w X) e12
  let e10 := mnv2SyncCotInBody R hR N 14 14 w.b11 (mnv2PreB10 (R * N) w X) e11
  let e9 := mnv2ResidSyncCotIn R hR N 14 14 w.b10 (mnv2PreB9 (R * N) w X) e10
  let e8 := mnv2ResidSyncCotIn R hR N 14 14 w.b9 (mnv2PreB8 (R * N) w X) e9
  let e7 := mnv2ResidSyncCotIn R hR N 14 14 w.b8 (mnv2PreB7 (R * N) w X) e8
  let e6 := mnv2StridedSyncCotIn R hR N 14 14 w.b7 (mnv2PreB6 (R * N) w X) e7
  let e5 := mnv2ResidSyncCotIn R hR N 28 28 w.b6 (mnv2PreB5 (R * N) w X) e6
  let e4 := mnv2ResidSyncCotIn R hR N 28 28 w.b5 (mnv2PreB4 (R * N) w X) e5
  let e3 := mnv2StridedSyncCotIn R hR N 28 28 w.b4 (mnv2PreB3 (R * N) w X) e4
  let e2 := mnv2ResidSyncCotIn R hR N 56 56 w.b3 (mnv2PreB2 (R * N) w X) e3
  let e1 := mnv2StridedSyncCotIn R hR N 56 56 w.b2 (mnv2PreB1 (R * N) w X) e2
  let eStem := mnv2NoExpSyncCotIn R hR N 112 112 w.b1 (mnv2PreB0 (R * N) w X) e1
  mnv2StemSyncTiedB R hR N 112 112 xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ X eStem cotStem
  ∧ mnv2NoExpSyncTiedB R hR N 112 112 "1" xN cotN vN epsStr w.b1 (mnv2PreB0 (R * N) w X) e1 dy1
  ∧ mnv2Stride2SyncTiedB R hR N 56 56 "2" xN cotN vN epsStr w.b2 (mnv2PreB1 (R * N) w X) e2 dy2
  ∧ mnv2Stride1SyncTiedB R hR N 56 56 "3" xN cotN vN epsStr w.b3 (mnv2PreB2 (R * N) w X) e3 dy3
  ∧ mnv2Stride2SyncTiedB R hR N 28 28 "4" xN cotN vN epsStr w.b4 (mnv2PreB3 (R * N) w X) e4 dy4
  ∧ mnv2Stride1SyncTiedB R hR N 28 28 "5" xN cotN vN epsStr w.b5 (mnv2PreB4 (R * N) w X) e5 dy5
  ∧ mnv2Stride1SyncTiedB R hR N 28 28 "6" xN cotN vN epsStr w.b6 (mnv2PreB5 (R * N) w X) e6 dy6
  ∧ mnv2Stride2SyncTiedB R hR N 14 14 "7" xN cotN vN epsStr w.b7 (mnv2PreB6 (R * N) w X) e7 dy7
  ∧ mnv2Stride1SyncTiedB R hR N 14 14 "8" xN cotN vN epsStr w.b8 (mnv2PreB7 (R * N) w X) e8 dy8
  ∧ mnv2Stride1SyncTiedB R hR N 14 14 "9" xN cotN vN epsStr w.b9 (mnv2PreB8 (R * N) w X) e9 dy9
  ∧ mnv2Stride1SyncTiedB R hR N 14 14 "10" xN cotN vN epsStr w.b10 (mnv2PreB9 (R * N) w X)
      e10 dy10
  ∧ mnv2Stride1SyncTiedB R hR N 14 14 "11" xN cotN vN epsStr w.b11 (mnv2PreB10 (R * N) w X)
      e11 dy11
  ∧ mnv2Stride1SyncTiedB R hR N 14 14 "12" xN cotN vN epsStr w.b12 (mnv2PreB11 (R * N) w X)
      e12 dy12
  ∧ mnv2Stride1SyncTiedB R hR N 14 14 "13" xN cotN vN epsStr w.b13 (mnv2PreB12 (R * N) w X)
      e13 dy13
  ∧ mnv2Stride2SyncTiedB R hR N 7 7 "14" xN cotN vN epsStr w.b14 (mnv2PreB13 (R * N) w X)
      e14 dy14
  ∧ mnv2Stride1SyncTiedB R hR N 7 7 "15" xN cotN vN epsStr w.b15 (mnv2PreB14 (R * N) w X)
      e15 dy15
  ∧ mnv2Stride1SyncTiedB R hR N 7 7 "16" xN cotN vN epsStr w.b16 (mnv2PreB15 (R * N) w X)
      e16 dy16
  ∧ mnv2Stride1SyncTiedB R hR N 7 7 "17" xN cotN vN epsStr w.b17 (mnv2PreB16 (R * N) w X)
      e17 dy17
  ∧ mnv2HeadSyncTiedB R hR N 7 7 xN cotN vN epsStr w.hW w.hb w.hε w.hγ w.hβ w.fcW
      (mnv2PreB17 (R * N) w X) gs G

/-- **The synchronised-BN data-parallel MobileNetV2 step is the single-device step at the
    global batch.** `R` replicas at batch `N`, each running the render's sync-BN backward chain from
    its own cotangent `gs r`, with `gs r` the `R`-scaled shard of a global cotangent `G`; every
    parameter's all-reduced mean gradient — stem 3, `b1` 6, sixteen blocks × 9, head 3, dense 2:
    the 158 the render emits — equals the single-device batch-BN gradient node at batch `R·N`, at
    the cotangent `mnv2_net_tiedB`'s chain delivers there from `G`.

    The left-hand chain is the replicas' own: sync-BN backward (`bnSyncInB`, a collective per
    BN layer), per-example conv / depthwise / relu6 / GAP / dense links. The right-hand chain is
    `mnv2_net_tiedB`'s at `N := R·N` with `g := G`, whose nodes that capstone ties to the certified
    gradient at its chain cotangent — so this and it together say every all-reduced gradient the
    DP render emits is the global-batch step's gradient node. The optimizer update that follows
    (RMSProp/AdamW) is not stated here. `mnv2_net_syncTiedB_smoothedCE` discharges the hypothesis for the
    label-smoothed chain the artifacts emit.

    With per-replica BatchNorm the corresponding statement is false in general;
    `DataParallel.dpMeanGrad_ne_globalBatchGrad` is a two-replica counterexample. -/
theorem mnv2_net_syncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) {nCls : Nat}
    (xN cotN vN epsStr : String) (w : MNV2BWeights nCls)
    (X : Vec ((R * N) * (3 * (2 * 112) * (2 * 112)))) (G : Vec ((R * N) * nCls))
    (gs : Fin R → Vec (N * nCls))
    (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r) :
    mnv2NetSyncTiedB R hR N xN cotN vN epsStr w X G gs := by
  unfold mnv2NetSyncTiedB
  intro dy17 dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 cotStem
    e17 e16 e15 e14 e13 e12 e11 e10 e9 e8 e7 e6 e5 e4 e3 e2 e1 eStem
  have h112 : 0 < 112 := by norm_num
  have h56 : 0 < 56 := by norm_num
  have h28 : 0 < 28 := by norm_num
  have h14 : 0 < 14 := by norm_num
  have h7 : 0 < 7 := by norm_num
  -- the scaled-shard invariant, block by block down the chain
  have s17 := mnv2HeadSyncCotBlk_scaled R hR N 7 7 hN h7 h7 w.hW w.hb w.hε w.hγ w.hβ w.fcW
    (mnv2PreB17 (R * N) w X) gs G hgs
  have s16 := mnv2SyncCotInBody_scaled R hR N 7 7 hN h7 h7 w.b17 (mnv2PreB16 (R * N) w X)
    e17 dy17 s17
  have s15 := mnv2ResidSyncCotIn_scaled R hR N 7 7 hN h7 h7 w.b16 (mnv2PreB15 (R * N) w X)
    e16 dy16 s16
  have s14 := mnv2ResidSyncCotIn_scaled R hR N 7 7 hN h7 h7 w.b15 (mnv2PreB14 (R * N) w X)
    e15 dy15 s15
  have s13 := mnv2StridedSyncCotIn_scaled R hR N 7 7 hN h7 h7 w.b14 (mnv2PreB13 (R * N) w X)
    e14 dy14 s14
  have s12 := mnv2ResidSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.b13 (mnv2PreB12 (R * N) w X)
    e13 dy13 s13
  have s11 := mnv2ResidSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.b12 (mnv2PreB11 (R * N) w X)
    e12 dy12 s12
  have s10 := mnv2SyncCotInBody_scaled R hR N 14 14 hN h14 h14 w.b11 (mnv2PreB10 (R * N) w X)
    e11 dy11 s11
  have s9 := mnv2ResidSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.b10 (mnv2PreB9 (R * N) w X)
    e10 dy10 s10
  have s8 := mnv2ResidSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.b9 (mnv2PreB8 (R * N) w X)
    e9 dy9 s9
  have s7 := mnv2ResidSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.b8 (mnv2PreB7 (R * N) w X)
    e8 dy8 s8
  have s6 := mnv2StridedSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.b7 (mnv2PreB6 (R * N) w X)
    e7 dy7 s7
  have s5 := mnv2ResidSyncCotIn_scaled R hR N 28 28 hN h28 h28 w.b6 (mnv2PreB5 (R * N) w X)
    e6 dy6 s6
  have s4 := mnv2ResidSyncCotIn_scaled R hR N 28 28 hN h28 h28 w.b5 (mnv2PreB4 (R * N) w X)
    e5 dy5 s5
  have s3 := mnv2StridedSyncCotIn_scaled R hR N 28 28 hN h28 h28 w.b4 (mnv2PreB3 (R * N) w X)
    e4 dy4 s4
  have s2 := mnv2ResidSyncCotIn_scaled R hR N 56 56 hN h56 h56 w.b3 (mnv2PreB2 (R * N) w X)
    e3 dy3 s3
  have s1 := mnv2StridedSyncCotIn_scaled R hR N 56 56 hN h56 h56 w.b2 (mnv2PreB1 (R * N) w X)
    e2 dy2 s2
  have sStem := mnv2NoExpSyncCotIn_scaled R hR N 112 112 hN h112 h112 w.b1
    (mnv2PreB0 (R * N) w X) e1 dy1 s1
  exact ⟨mnv2_stem_syncTiedB R hR N 112 112 hN h112 h112 xN cotN vN epsStr
      w.sW w.sb w.sε w.sγ w.sβ X eStem _ sStem,
    mnv2_noexp_syncTiedB R hR N 112 112 hN h112 h112 "1" xN cotN vN epsStr w.b1
      (mnv2PreB0 (R * N) w X) e1 dy1 s1,
    mnv2_stride2_syncTiedB R hR N 56 56 hN h56 h56 "2" xN cotN vN epsStr w.b2
      (mnv2PreB1 (R * N) w X) e2 dy2 s2,
    mnv2_stride1_syncTiedB R hR N 56 56 hN h56 h56 "3" xN cotN vN epsStr w.b3
      (mnv2PreB2 (R * N) w X) e3 dy3 s3,
    mnv2_stride2_syncTiedB R hR N 28 28 hN h28 h28 "4" xN cotN vN epsStr w.b4
      (mnv2PreB3 (R * N) w X) e4 dy4 s4,
    mnv2_stride1_syncTiedB R hR N 28 28 hN h28 h28 "5" xN cotN vN epsStr w.b5
      (mnv2PreB4 (R * N) w X) e5 dy5 s5,
    mnv2_stride1_syncTiedB R hR N 28 28 hN h28 h28 "6" xN cotN vN epsStr w.b6
      (mnv2PreB5 (R * N) w X) e6 dy6 s6,
    mnv2_stride2_syncTiedB R hR N 14 14 hN h14 h14 "7" xN cotN vN epsStr w.b7
      (mnv2PreB6 (R * N) w X) e7 dy7 s7,
    mnv2_stride1_syncTiedB R hR N 14 14 hN h14 h14 "8" xN cotN vN epsStr w.b8
      (mnv2PreB7 (R * N) w X) e8 dy8 s8,
    mnv2_stride1_syncTiedB R hR N 14 14 hN h14 h14 "9" xN cotN vN epsStr w.b9
      (mnv2PreB8 (R * N) w X) e9 dy9 s9,
    mnv2_stride1_syncTiedB R hR N 14 14 hN h14 h14 "10" xN cotN vN epsStr w.b10
      (mnv2PreB9 (R * N) w X) e10 dy10 s10,
    mnv2_stride1_syncTiedB R hR N 14 14 hN h14 h14 "11" xN cotN vN epsStr w.b11
      (mnv2PreB10 (R * N) w X) e11 dy11 s11,
    mnv2_stride1_syncTiedB R hR N 14 14 hN h14 h14 "12" xN cotN vN epsStr w.b12
      (mnv2PreB11 (R * N) w X) e12 dy12 s12,
    mnv2_stride1_syncTiedB R hR N 14 14 hN h14 h14 "13" xN cotN vN epsStr w.b13
      (mnv2PreB12 (R * N) w X) e13 dy13 s13,
    mnv2_stride2_syncTiedB R hR N 7 7 hN h7 h7 "14" xN cotN vN epsStr w.b14
      (mnv2PreB13 (R * N) w X) e14 dy14 s14,
    mnv2_stride1_syncTiedB R hR N 7 7 hN h7 h7 "15" xN cotN vN epsStr w.b15
      (mnv2PreB14 (R * N) w X) e15 dy15 s15,
    mnv2_stride1_syncTiedB R hR N 7 7 hN h7 h7 "16" xN cotN vN epsStr w.b16
      (mnv2PreB15 (R * N) w X) e16 dy16 s16,
    mnv2_stride1_syncTiedB R hR N 7 7 hN h7 h7 "17" xN cotN vN epsStr w.b17
      (mnv2PreB16 (R * N) w X) e17 dy17 s17,
    mnv2_head_syncTiedB R hR N 7 7 hN h7 h7 xN cotN vN epsStr w.hW w.hb w.hε w.hγ w.hβ w.fcW
      (mnv2PreB17 (R * N) w X) gs G hgs⟩

/-- **…and at the loss the artifacts emit.** `mnv2_net_syncTiedB` with its cotangent hypothesis
    discharged by `replicaLossCot_eq`: each replica runs the label-smoothed softmax chain
    (`smoothedLossCotGraph`) on its shard of the logits and targets with divisor `B`; the
    single-device step runs it on the whole `R·N` batch with divisor `R·B`. Then every all-reduced
    gradient the DP render emits is the single-device node at batch `R·N`, loss divided by `R·B`. -/
theorem mnv2_net_syncTiedB_smoothedCE (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) {nCls : Nat}
    (xN cotN vN epsStr : String) (aStr negAK bStr logN ohN : String) (α B : ℝ)
    (w : MNV2BWeights nCls) (X : Vec ((R * N) * (3 * (2 * 112) * (2 * 112))))
    (T : Vec ((R * N) * (1 * nCls))) :
    mnv2NetSyncTiedB R hR N xN cotN vN epsStr w X
      (unrowB (R * N) nCls (den (smoothedLossCotGraph (R * N) nCls α ((R : ℝ) * B) aStr negAK
        bStr logN ohN (rowB (R * N) nCls (mobilenetv2ForwardBFull (R * N) w X)) T)))
      (fun r => unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (batchShard R N nCls (mobilenetv2ForwardBFull (R * N) w X) r))
        (batchShard R N (1 * nCls) T r)))) :=
  mnv2_net_syncTiedB R hR N hN xN cotN vN epsStr w X _ _
    (fun r => replicaLossCot_eq R N nCls hR α B aStr negAK bStr logN ohN _ T r)

end Proofs.MobileNetV2SyncTieB
