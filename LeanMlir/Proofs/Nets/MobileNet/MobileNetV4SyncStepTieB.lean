import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4StepTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncStepTieB
import LeanMlir.Proofs.Foundation.DataParallelSyncKit
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncB

/-! # MobileNetV4-Conv-M's data-parallel step at SYNCHRONISED BatchNorm IS the single-device step at `R·N`

`MobileNetV4StepTieB.lean` (T3) threads a loss cotangent `g` down the batch-BN backward chain on
ONE device and ties every parameter gradient node to the certified gradient. This is its
data-parallel twin, for the render `MobileNetV4RenderB` emits at `replicas > 1`: `R` replicas at
batch `N`, every one of the 77 BatchNorms synchronised, every parameter gradient all-reduced by
its mean. The capstone `mnv4_net_syncTiedB` says that, for every parameter the render emits,

    mean over the R replicas of replica r's gradient node, replica cotangent gs r
      = the single-device gradient node at the global batch R·N, global cotangent G

whenever each replica's loss cotangent is `R ×` its shard of the global one — the gradient node
`mnv4_net_tiedB` at `N := R·N` ties to the certified gradient. The right-hand side is the existing
single-device chain at `N := R·N`, so the spec has not moved.

## ⚠ T3 binds the loss cotangent, so this twin does too

`mnv4_net_tiedB` takes the logits' cotangent `g` as a binder (`mnv4_lossCot_is_smoothedCE_grad`
instantiates it). The twin therefore takes the global cotangent `G` and the replica family `gs` as
binders, with the scaled-shard hypothesis `∀ r, gs r = batchShard R N nCls (fun i => R * G i) r` —
exactly the invariant ResNet-34's and MobileNetV2's twins carry down their chains, and there
discharge from `replicaLossCot_eq`. `mnv4_net_syncTiedB_smoothedCE` discharges it the same way for
the label-smoothed softmax chain the artifacts emit: replicas dividing their loss by `B`, the
single-device step by `R·B`.

## The same four steps as ResNet-34's twin

`ResNet34SyncStepTieB.lean` is the template, and everything net-agnostic is imported from it: the
replica BN link `bnSyncInB` and its shard lemma, the `ConvWSync` / `ConvStridedWSync` / `BnSync` /
`DenseSync` statements and their `*_of_scaled` closers, and the homogeneity of `bnInB`, `cInB`,
`cStridedInB` and the head. The MBConv pieces come from `DataParallelSyncKit` — the depthwise and
SYMMETRIC strided-depthwise input-VJPs and weight collectives, and the XLA-`SAME` stem collective.
Swish's backward is a certified VJP's `.backward`, so its homogeneity is `HasVJP.backward_smul`
and its sharding is definitional.

1. **Sharding** — each replica's backward chain, handed its shard of a global cotangent, computes
   the shard of the global chain. The relu mask and swish's backward are pointwise; the conv,
   depthwise, strided conv and strided depthwise input-VJPs are per-example maps; every BN link is
   `bnSyncInB`. The table's `k = 0` dispatch (`if s.postDWk = 0`, `if s.preDWk = 0`) is the same
   `if` on both sides, so it splits once.
2. **The collectives** — the mean over replicas of each replica's gradient node is `1/R` of the
   global node at the global cotangent. MobileNetV4 needs no kind the kit does not have.
3. **Homogeneity** — the single-device chain and its gradient nodes are linear in the loss
   cotangent (§1): `R ×` the cotangent gives `R ×` every node.
4. **The divisor** — the hypothesis on `gs`; its `R` cancels the collective's `1/R` at every
   parameter.

⭐ Everything is GENERIC IN THE ROW (`s : UibSpec`), as T3 is, so widths stay variables; the
capstone instantiates at the 21 concrete rows. MNv4's activation is **relu** (the fused stage's is
swish), so the masks are `reluMaskB` — MobileNetV2's `relu6MaskB` does not appear.

## The index seam

ℝ-level, as T3 is: the replica BN link `bnSyncInB` is the `den` of the emitted nodes
(`bnSyncDyStatsB` → all-reduce → `bnSyncBack`) over `.operand` leaves at `reassocB`, and
`bnSyncInB_shard` carries P2 across the `N·(c·h·w)` / `N·(c·(h·w))` seam; `BnSync`'s γ and β
collectives read `reassocB` of the pre-BN activation and of the cotangent, exactly as T3's
`BnPairTiedB` nodes do.

## What the DP render emits, and what is tied

`MobileNetV4RenderB` emits 233 parameter gradients — stem 3 (`sW`, `sg`, `sbt`), fused 6
(`f0cW f0cg f0cbt f0pW f0pg f0pbt`), thirteen ExtraDW-profile blocks × 12 (ten stride-1 rows and
the three pre-strided rows 1, 3, 11: `u{p}{q,e,d,p}{W,g,bt}`), four ConvNeXt-like × 9 (no `d`),
four FFN × 6 (no `q`, no `d`), head 8 (`h1W h1g h1bt hW hg hbt Wd bd`) — and the capstone ties all
233. ⚠ There are no conv-bias gradients to exclude: the render has no `convBias` flag, binds every
bias slot to `%zb{c}`, and emits none. ⚠ Conv-M has no post-strided row, so the render's
post-strided backward is never emitted for this table and T3 has no chain for it; neither does
this file.

## What is NOT claimed

⚠ The replicas' saved forward activations enter as the shards of the single-device forward's
(`batchShard r (mnv4Blk{k} (R*N) w X)`); that the sync forward graph computes exactly those is
`StableHLO.mnv4FwdGraphSync_full_shard`, the forward half. ⚠ That the replicas' inputs are the
shards of one batch is the driver's. ⚠ The statement is at the f32 nodes: the `*bf16` artifact's
bf16 conv twins are outside it, as for every other net. ⚠ The lowerer's `all_reduce` is trusted as
every other op's lowering is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.MobileNetV4SyncTieB

open scoped BigOperators
open Proofs.EnetTiePoC (reassocB cInB dInB dStridedInB swBackB)
open Proofs.ResNet34TieB (bnInB unrowB rowB reluMaskB cStridedInB r34HeadCotBlk)
open Proofs.ResNet34SyncTieB
open Proofs.MBConvSyncTieB
open Proofs.Mnv4TieB

-- ════════════════════════════════════════════════════════════════
-- § 1. Homogeneity — the single-device chain is linear in its cotangent
-- ════════════════════════════════════════════════════════════════

/-! The stride-1 body (ExtraDW / ConvNeXt-like / FFN), each one line from the previous link's. The
two `if`s are T3's table dispatch; they split on both sides at once. -/

theorem mnv4CotPc_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) : IsHomog (mnv4CotPc N s p xin) :=
  bnInB_smul _ _ _ _ _ _ _

theorem mnv4CotDn_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) : IsHomog (mnv4CotDn N s p xin) := by
  intro a dy
  unfold mnv4CotDn; rw [mnv4CotPc_smul, cInB_smul, reluMaskB_smul]

theorem mnv4CotDc_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) : IsHomog (mnv4CotDc N s p xin) := by
  intro a dy
  unfold mnv4CotDc; rw [mnv4CotDn_smul, bnInB_smul]

theorem mnv4CotEn_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) : IsHomog (mnv4CotEn N s p xin) := by
  intro a dy
  unfold mnv4CotEn
  split_ifs
  · rw [mnv4CotPc_smul, cInB_smul, reluMaskB_smul]
  · rw [mnv4CotDc_smul, dInB_smul, reluMaskB_smul]

theorem mnv4CotEc_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) : IsHomog (mnv4CotEc N s p xin) := by
  intro a dy
  unfold mnv4CotEc; rw [mnv4CotEn_smul, bnInB_smul]

theorem mnv4CotQn_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) : IsHomog (mnv4CotQn N s p xin) := by
  intro a dy
  unfold mnv4CotQn; rw [mnv4CotEc_smul, cInB_smul, reluMaskB_smul]

theorem mnv4CotQc_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) : IsHomog (mnv4CotQc N s p xin) := by
  intro a dy
  unfold mnv4CotQc; rw [mnv4CotQn_smul, bnInB_smul]

theorem mnv4BodyCotIn_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * s.h * s.h))) : IsHomog (mnv4BodyCotIn N s p xin) := by
  intro a dy
  unfold mnv4BodyCotIn
  split_ifs
  · rw [mnv4CotEc_smul, cInB_smul]
  · rw [mnv4CotQc_smul, dInB_smul]

/-! The pre-strided block (rows 1, 3, 11). -/

theorem mnv4SCotPc_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) : IsHomog (mnv4SCotPc N s p xin) :=
  bnInB_smul _ _ _ _ _ _ _

theorem mnv4SCotDn_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) : IsHomog (mnv4SCotDn N s p xin) := by
  intro a dy
  unfold mnv4SCotDn; rw [mnv4SCotPc_smul, cInB_smul, reluMaskB_smul]

theorem mnv4SCotDc_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) : IsHomog (mnv4SCotDc N s p xin) := by
  intro a dy
  unfold mnv4SCotDc; rw [mnv4SCotDn_smul, bnInB_smul]

theorem mnv4SCotEn_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) : IsHomog (mnv4SCotEn N s p xin) := by
  intro a dy
  unfold mnv4SCotEn; rw [mnv4SCotDc_smul, dInB_smul, reluMaskB_smul]

theorem mnv4SCotEc_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) : IsHomog (mnv4SCotEc N s p xin) := by
  intro a dy
  unfold mnv4SCotEc; rw [mnv4SCotEn_smul, bnInB_smul]

theorem mnv4SCotQn_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) : IsHomog (mnv4SCotQn N s p xin) := by
  intro a dy
  unfold mnv4SCotQn; rw [mnv4SCotEc_smul, cInB_smul, reluMaskB_smul]

theorem mnv4SCotQc_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) : IsHomog (mnv4SCotQc N s p xin) := by
  intro a dy
  unfold mnv4SCotQc; rw [mnv4SCotQn_smul, bnInB_smul]

theorem mnv4SBodyCotIn_smul (N : Nat) (s : UibSpec) (p : UibParams s)
    (xin : Vec (N * (s.ic * (2 * s.h) * (2 * s.h)))) : IsHomog (mnv4SBodyCotIn N s p xin) := by
  intro a dy
  unfold mnv4SBodyCotIn; rw [mnv4SCotQc_smul, dStridedInB_smul]

/-! The stem, the fused stage (swish, no mask) and the two-conv head. -/

theorem mnv4StemCotN_smul (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (mnv4StemCotN N h w Ws bs εs γs βs x) :=
  reluMaskB_smul _ _

theorem mnv4StemCotC_smul (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (mnv4StemCotC N h w Ws bs εs γs βs x) := by
  intro a dy
  unfold mnv4StemCotC; rw [mnv4StemCotN_smul, bnInB_smul]

theorem mnv4FusedCotPc_smul (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (mnv4FusedCotPc N h w Wc bc εc γc βc Wp bp εp γp βp xin) :=
  bnInB_smul _ _ _ _ _ _ _

theorem mnv4FusedCotN_smul (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (mnv4FusedCotN N h w Wc bc εc γc βc Wp bp εp γp βp xin) := by
  intro a dy
  unfold mnv4FusedCotN; rw [mnv4FusedCotPc_smul, cInB_smul]; exact HasVJP.backward_smul _ _ a _

theorem mnv4FusedCotC_smul (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (mnv4FusedCotC N h w Wc bc εc γc βc Wp bp εp γp βp xin) := by
  intro a dy
  unfold mnv4FusedCotC; rw [mnv4FusedCotN_smul, bnInB_smul]

theorem mnv4FusedCotIn_smul (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
    (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
    (γp βp : Vec oc) (xin : Vec (N * (ic * (2 * h) * (2 * w)))) :
    IsHomog (mnv4FusedCotIn N h w Wc bc εc γc βc Wp bp εp γp βp xin) := by
  intro a dy
  unfold mnv4FusedCotIn; rw [mnv4FusedCotC_smul, cStridedInB_smul]

theorem mnv4HeadCotHn_smul (N h w : Nat) {c mid oc nCls : Nat} (W1 : Kernel4 mid c 1 1)
    (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid) (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ)
    (γ2 β2 : Vec oc) (Wd : Mat oc nCls) (bd : Vec nCls) (xin : Vec (N * (c * h * w))) :
    IsHomog (mnv4HeadCotHn N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin) := by
  intro a g
  unfold mnv4HeadCotHn; rw [r34HeadCotBlk_smul, reluMaskB_smul]

theorem mnv4HeadCotHc_smul (N h w : Nat) {c mid oc nCls : Nat} (W1 : Kernel4 mid c 1 1)
    (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid) (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ)
    (γ2 β2 : Vec oc) (Wd : Mat oc nCls) (bd : Vec nCls) (xin : Vec (N * (c * h * w))) :
    IsHomog (mnv4HeadCotHc N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin) := by
  intro a g
  unfold mnv4HeadCotHc; rw [mnv4HeadCotHn_smul, bnInB_smul]

theorem mnv4HeadCotH1n_smul (N h w : Nat) {c mid oc nCls : Nat} (W1 : Kernel4 mid c 1 1)
    (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid) (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ)
    (γ2 β2 : Vec oc) (Wd : Mat oc nCls) (bd : Vec nCls) (xin : Vec (N * (c * h * w))) :
    IsHomog (mnv4HeadCotH1n N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin) := by
  intro a g
  unfold mnv4HeadCotH1n; rw [mnv4HeadCotHc_smul, cInB_smul, reluMaskB_smul]

theorem mnv4HeadCotH1c_smul (N h w : Nat) {c mid oc nCls : Nat} (W1 : Kernel4 mid c 1 1)
    (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid) (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ)
    (γ2 β2 : Vec oc) (Wd : Mat oc nCls) (bd : Vec nCls) (xin : Vec (N * (c * h * w))) :
    IsHomog (mnv4HeadCotH1c N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin) := by
  intro a g
  unfold mnv4HeadCotH1c; rw [mnv4HeadCotH1n_smul, bnInB_smul]

theorem mnv4HeadCotIn_smul (N h w : Nat) {c mid oc nCls : Nat} (W1 : Kernel4 mid c 1 1)
    (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid) (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ)
    (γ2 β2 : Vec oc) (Wd : Mat oc nCls) (bd : Vec nCls) (xin : Vec (N * (c * h * w))) :
    IsHomog (mnv4HeadCotIn N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xin) := by
  intro a g
  unfold mnv4HeadCotIn; rw [mnv4HeadCotH1c_smul, cInB_smul]

-- ════════════════════════════════════════════════════════════════
-- § 2–3. The replica chain, block by block, and its shard lemmas
--   Saved activations are the shards of the single-device forward's (the forward half,
--   `MobileNetV4SyncB`, is what says a replica computes exactly those); every cotangent is the
--   replica's own, from the family `dys` of block-output cotangents. Every BatchNorm backward is
--   `bnSyncInB` — the render's `bnSyncDyStatsB` → all-reduce → `bnSyncBack`.
-- ════════════════════════════════════════════════════════════════

section Stride1
variable (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec) (p : UibParams s)
  (XIN : Vec ((R * N) * (s.ic * s.h * s.h))) (dys : Fin R → Vec (N * (s.oc * s.h * s.h)))

/-- Stride-1 body, replica `r`: the project BatchNorm's sync backward of the block-output cotangent
    (the bottleneck is linear, so nothing masks it). Feeds `u{p}pW`. -/
noncomputable def mnv4SyncCotPc (r : Fin R) : Vec (N * (s.oc * s.h * s.h)) :=
  bnSyncInB R hR N s.oc s.h s.h p.ez p.gz
    (fun r => batchShard R N (s.oc * s.h * s.h) (batchMap (R * N) (flatConv p.Wz p.bz)
      ((mnv4PostDWSlot (h := s.h) (w := s.h) (R * N) s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd
        ((cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd
          ((mnv4PreDWSlot (h := s.h) (w := s.h) (R * N) s.preDWk p.Wq p.bq p.eq_ p.hq p.gq
            p.bq2).fwd XIN)))) r)
    dys r

/-- Stride-1 body, replica `r`: the post-DW BN's output cotangent. Feeds `u{p}dg`/`u{p}dbt`. -/
noncomputable def mnv4SyncCotDn (r : Fin R) : Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.expand * s.h * s.h))
    (batchShard R N (s.ic * s.expand * s.h * s.h)
      (bnBatchLA (R * N) (s.ic * s.expand) s.h s.h p.ed p.gd p.bd2
        (batchMap (R * N) (depthwiseFlat p.Wd p.bd)
          ((cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd
            ((mnv4PreDWSlot (h := s.h) (w := s.h) (R * N) s.preDWk p.Wq p.bq p.eq_ p.hq p.gq
              p.bq2).fwd XIN)))) r)
    (cInB N p.Wz p.bz (mnv4SyncCotPc R hR N s p XIN dys r))

/-- Stride-1 body, replica `r`: the post-DW conv's output cotangent. Feeds `u{p}dW`. -/
noncomputable def mnv4SyncCotDc (r : Fin R) : Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnSyncInB R hR N (s.ic * s.expand) s.h s.h p.ed p.gd
    (fun r => batchShard R N (s.ic * s.expand * s.h * s.h) (batchMap (R * N) (depthwiseFlat p.Wd p.bd)
      ((cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd
        ((mnv4PreDWSlot (h := s.h) (w := s.h) (R * N) s.preDWk p.Wq p.bq p.eq_ p.hq p.gq
          p.bq2).fwd XIN))) r)
    (mnv4SyncCotDn R hR N s p XIN dys) r

/-- Stride-1 body, replica `r`: the expand BN's output cotangent — dispatching on the row exactly
    as T3's `mnv4CotEn` does. Feeds `u{p}eg`/`u{p}ebt`. -/
noncomputable def mnv4SyncCotEn (r : Fin R) : Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.expand * s.h * s.h))
    (batchShard R N (s.ic * s.expand * s.h * s.h)
      (bnBatchLA (R * N) (s.ic * s.expand) s.h s.h p.ee p.ge p.be2
        (batchMap (R * N) (flatConv p.We p.be)
          ((mnv4PreDWSlot (h := s.h) (w := s.h) (R * N) s.preDWk p.Wq p.bq p.eq_ p.hq p.gq
            p.bq2).fwd XIN))) r)
    (if s.postDWk = 0 then cInB N p.Wz p.bz (mnv4SyncCotPc R hR N s p XIN dys r)
     else dInB N p.Wd p.bd (mnv4SyncCotDc R hR N s p XIN dys r))

/-- Stride-1 body, replica `r`: the expand conv's output cotangent. Feeds `u{p}eW`. -/
noncomputable def mnv4SyncCotEc (r : Fin R) : Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnSyncInB R hR N (s.ic * s.expand) s.h s.h p.ee p.ge
    (fun r => batchShard R N (s.ic * s.expand * s.h * s.h) (batchMap (R * N) (flatConv p.We p.be)
      ((mnv4PreDWSlot (h := s.h) (w := s.h) (R * N) s.preDWk p.Wq p.bq p.eq_ p.hq p.gq
        p.bq2).fwd XIN)) r)
    (mnv4SyncCotEn R hR N s p XIN dys) r

/-- Stride-1 body, replica `r`: the pre-DW BN's output cotangent. Feeds `u{p}qg`/`u{p}qbt`. -/
noncomputable def mnv4SyncCotQn (r : Fin R) : Vec (N * (s.ic * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.h * s.h))
    (batchShard R N (s.ic * s.h * s.h)
      (bnBatchLA (R * N) s.ic s.h s.h p.eq_ p.gq p.bq2
        (batchMap (R * N) (depthwiseFlat p.Wq p.bq) XIN)) r)
    (cInB N p.We p.be (mnv4SyncCotEc R hR N s p XIN dys r))

/-- Stride-1 body, replica `r`: the pre-DW conv's output cotangent. Feeds `u{p}qW`. -/
noncomputable def mnv4SyncCotQc (r : Fin R) : Vec (N * (s.ic * s.h * s.h)) :=
  bnSyncInB R hR N s.ic s.h s.h p.eq_ p.gq
    (fun r => batchShard R N (s.ic * s.h * s.h) (batchMap (R * N) (depthwiseFlat p.Wq p.bq) XIN) r)
    (mnv4SyncCotQn R hR N s p XIN dys) r

/-- Stride-1 body, replica `r`: the body's input cotangent, before the skip fan-in — the pre-DW's
    input-VJP, or the expand's when the row has no pre-DW. -/
noncomputable def mnv4BodySyncCotIn (r : Fin R) : Vec (N * (s.ic * s.h * s.h)) :=
  if s.preDWk = 0 then cInB N p.We p.be (mnv4SyncCotEc R hR N s p XIN dys r)
  else dInB N p.Wq p.bq (mnv4SyncCotQc R hR N s p XIN dys r)

end Stride1

section Stride1Shard
variable (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (s : UibSpec) (hh : 0 < s.h)
  (p : UibParams s) (XIN : Vec ((R * N) * (s.ic * s.h * s.h)))
  (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h)))
  (hdys : ∀ r, dys r = batchShard R N (s.oc * s.h * s.h) DY r)
include hN hh hdys

theorem mnv4SyncCotPc_shard (r : Fin R) :
    mnv4SyncCotPc R hR N s p XIN dys r
      = batchShard R N (s.oc * s.h * s.h) (mnv4CotPc (R * N) s p XIN DY) r :=
  bnSyncInB_shard R hR N s.oc s.h s.h (nhw_ne_zero hN hh hh) _ _ _ _ _ _ (fun _ => rfl) hdys r

theorem mnv4SyncCotDn_shard (r : Fin R) :
    mnv4SyncCotDn R hR N s p XIN dys r
      = batchShard R N (s.ic * s.expand * s.h * s.h) (mnv4CotDn (R * N) s p XIN DY) r := by
  unfold mnv4SyncCotDn
  rw [mnv4SyncCotPc_shard R hR N hN s hh p XIN dys DY hdys, cInB_shard]
  rfl

theorem mnv4SyncCotDc_shard (r : Fin R) :
    mnv4SyncCotDc R hR N s p XIN dys r
      = batchShard R N (s.ic * s.expand * s.h * s.h) (mnv4CotDc (R * N) s p XIN DY) r :=
  bnSyncInB_shard R hR N (s.ic * s.expand) s.h s.h (nhw_ne_zero hN hh hh) _ _ _ _ _ _ (fun _ => rfl)
    (mnv4SyncCotDn_shard R hR N hN s hh p XIN dys DY hdys) r

theorem mnv4SyncCotEn_shard (r : Fin R) :
    mnv4SyncCotEn R hR N s p XIN dys r
      = batchShard R N (s.ic * s.expand * s.h * s.h) (mnv4CotEn (R * N) s p XIN DY) r := by
  unfold mnv4SyncCotEn mnv4CotEn
  split_ifs
  · rw [mnv4SyncCotPc_shard R hR N hN s hh p XIN dys DY hdys, cInB_shard]
    rfl
  · rw [mnv4SyncCotDc_shard R hR N hN s hh p XIN dys DY hdys, dInB_shard]
    rfl

theorem mnv4SyncCotEc_shard (r : Fin R) :
    mnv4SyncCotEc R hR N s p XIN dys r
      = batchShard R N (s.ic * s.expand * s.h * s.h) (mnv4CotEc (R * N) s p XIN DY) r :=
  bnSyncInB_shard R hR N (s.ic * s.expand) s.h s.h (nhw_ne_zero hN hh hh) _ _ _ _ _ _ (fun _ => rfl)
    (mnv4SyncCotEn_shard R hR N hN s hh p XIN dys DY hdys) r

theorem mnv4SyncCotQn_shard (r : Fin R) :
    mnv4SyncCotQn R hR N s p XIN dys r
      = batchShard R N (s.ic * s.h * s.h) (mnv4CotQn (R * N) s p XIN DY) r := by
  unfold mnv4SyncCotQn
  rw [mnv4SyncCotEc_shard R hR N hN s hh p XIN dys DY hdys, cInB_shard]
  rfl

theorem mnv4SyncCotQc_shard (r : Fin R) :
    mnv4SyncCotQc R hR N s p XIN dys r
      = batchShard R N (s.ic * s.h * s.h) (mnv4CotQc (R * N) s p XIN DY) r :=
  bnSyncInB_shard R hR N s.ic s.h s.h (nhw_ne_zero hN hh hh) _ _ _ _ _ _ (fun _ => rfl)
    (mnv4SyncCotQn_shard R hR N hN s hh p XIN dys DY hdys) r

theorem mnv4BodySyncCotIn_shard (r : Fin R) :
    mnv4BodySyncCotIn R hR N s p XIN dys r
      = batchShard R N (s.ic * s.h * s.h) (mnv4BodyCotIn (R * N) s p XIN DY) r := by
  unfold mnv4BodySyncCotIn mnv4BodyCotIn
  split_ifs
  · rw [mnv4SyncCotEc_shard R hR N hN s hh p XIN dys DY hdys, cInB_shard]
  · rw [mnv4SyncCotQc_shard R hR N hN s hh p XIN dys DY hdys, dInB_shard]

end Stride1Shard

section PreStrided
variable (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec) (p : UibParams s)
  (XIN : Vec ((R * N) * (s.ic * (2 * s.h) * (2 * s.h))))
  (dys : Fin R → Vec (N * (s.oc * s.h * s.h)))

/-- Pre-strided block, replica `r`: the project BatchNorm's sync backward. Feeds `u{p}pW`. -/
noncomputable def mnv4SSyncCotPc (r : Fin R) : Vec (N * (s.oc * s.h * s.h)) :=
  bnSyncInB R hR N s.oc s.h s.h p.ez p.gz
    (fun r => batchShard R N (s.oc * s.h * s.h) (batchMap (R * N) (flatConv p.Wz p.bz)
      ((mnv4PostDWSlot (h := s.h) (w := s.h) (R * N) s.postDWk p.Wd p.bd p.ed p.hd p.gd p.bd2).fwd
        ((cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd
          ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) (R * N) p.Wq p.bq p.eq_ p.hq p.gq
            p.bq2).fwd XIN)))) r)
    dys r

/-- Pre-strided block, replica `r`: the post-DW BN's output cotangent. Feeds `u{p}dg`/`u{p}dbt`. -/
noncomputable def mnv4SSyncCotDn (r : Fin R) : Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.expand * s.h * s.h))
    (batchShard R N (s.ic * s.expand * s.h * s.h)
      (bnBatchLA (R * N) (s.ic * s.expand) s.h s.h p.ed p.gd p.bd2
        (batchMap (R * N) (depthwiseFlat p.Wd p.bd)
          ((cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd
            ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) (R * N) p.Wq p.bq p.eq_ p.hq p.gq
              p.bq2).fwd XIN)))) r)
    (cInB N p.Wz p.bz (mnv4SSyncCotPc R hR N s p XIN dys r))

/-- Pre-strided block, replica `r`: the post-DW conv's output cotangent. Feeds `u{p}dW`. -/
noncomputable def mnv4SSyncCotDc (r : Fin R) : Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnSyncInB R hR N (s.ic * s.expand) s.h s.h p.ed p.gd
    (fun r => batchShard R N (s.ic * s.expand * s.h * s.h) (batchMap (R * N) (depthwiseFlat p.Wd p.bd)
      ((cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd
        ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) (R * N) p.Wq p.bq p.eq_ p.hq p.gq
          p.bq2).fwd XIN))) r)
    (mnv4SSyncCotDn R hR N s p XIN dys) r

/-- Pre-strided block, replica `r`: the expand BN's output cotangent. Feeds `u{p}eg`/`u{p}ebt`. -/
noncomputable def mnv4SSyncCotEn (r : Fin R) : Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.expand * s.h * s.h))
    (batchShard R N (s.ic * s.expand * s.h * s.h)
      (bnBatchLA (R * N) (s.ic * s.expand) s.h s.h p.ee p.ge p.be2
        (batchMap (R * N) (flatConv p.We p.be)
          ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) (R * N) p.Wq p.bq p.eq_ p.hq p.gq
            p.bq2).fwd XIN))) r)
    (dInB N p.Wd p.bd (mnv4SSyncCotDc R hR N s p XIN dys r))

/-- Pre-strided block, replica `r`: the expand conv's output cotangent. Feeds `u{p}eW`. -/
noncomputable def mnv4SSyncCotEc (r : Fin R) : Vec (N * (s.ic * s.expand * s.h * s.h)) :=
  bnSyncInB R hR N (s.ic * s.expand) s.h s.h p.ee p.ge
    (fun r => batchShard R N (s.ic * s.expand * s.h * s.h) (batchMap (R * N) (flatConv p.We p.be)
      ((mnv4DWReluStridedLayer (h := s.h) (w := s.h) (R * N) p.Wq p.bq p.eq_ p.hq p.gq
        p.bq2).fwd XIN)) r)
    (mnv4SSyncCotEn R hR N s p XIN dys) r

/-- Pre-strided block, replica `r`: the STRIDED pre-DW BN's output cotangent. Feeds
    `u{p}qg`/`u{p}qbt`. -/
noncomputable def mnv4SSyncCotQn (r : Fin R) : Vec (N * (s.ic * s.h * s.h)) :=
  reluMaskB (N * (s.ic * s.h * s.h))
    (batchShard R N (s.ic * s.h * s.h)
      (bnBatchLA (R * N) s.ic s.h s.h p.eq_ p.gq p.bq2
        (batchMap (R * N) (depthwiseStride2Flat p.Wq p.bq) XIN)) r)
    (cInB N p.We p.be (mnv4SSyncCotEc R hR N s p XIN dys r))

/-- Pre-strided block, replica `r`: the STRIDED pre-DW conv's output cotangent. Feeds `u{p}qW`. -/
noncomputable def mnv4SSyncCotQc (r : Fin R) : Vec (N * (s.ic * s.h * s.h)) :=
  bnSyncInB R hR N s.ic s.h s.h p.eq_ p.gq
    (fun r => batchShard R N (s.ic * s.h * s.h)
      (batchMap (R * N) (depthwiseStride2Flat p.Wq p.bq) XIN) r)
    (mnv4SSyncCotQn R hR N s p XIN dys) r

/-- Pre-strided block, replica `r`: the block-INPUT cotangent — the strided depthwise's input-VJP,
    landing at `2h`. No skip. -/
noncomputable def mnv4SBodySyncCotIn (r : Fin R) : Vec (N * (s.ic * (2 * s.h) * (2 * s.h))) :=
  dStridedInB N p.Wq p.bq (mnv4SSyncCotQc R hR N s p XIN dys r)

end PreStrided

section PreStridedShard
variable (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (s : UibSpec) (hh : 0 < s.h)
  (p : UibParams s) (XIN : Vec ((R * N) * (s.ic * (2 * s.h) * (2 * s.h))))
  (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h)))
  (hdys : ∀ r, dys r = batchShard R N (s.oc * s.h * s.h) DY r)
include hN hh hdys

theorem mnv4SSyncCotPc_shard (r : Fin R) :
    mnv4SSyncCotPc R hR N s p XIN dys r
      = batchShard R N (s.oc * s.h * s.h) (mnv4SCotPc (R * N) s p XIN DY) r :=
  bnSyncInB_shard R hR N s.oc s.h s.h (nhw_ne_zero hN hh hh) _ _ _ _ _ _ (fun _ => rfl) hdys r

theorem mnv4SSyncCotDn_shard (r : Fin R) :
    mnv4SSyncCotDn R hR N s p XIN dys r
      = batchShard R N (s.ic * s.expand * s.h * s.h) (mnv4SCotDn (R * N) s p XIN DY) r := by
  unfold mnv4SSyncCotDn
  rw [mnv4SSyncCotPc_shard R hR N hN s hh p XIN dys DY hdys, cInB_shard]
  rfl

theorem mnv4SSyncCotDc_shard (r : Fin R) :
    mnv4SSyncCotDc R hR N s p XIN dys r
      = batchShard R N (s.ic * s.expand * s.h * s.h) (mnv4SCotDc (R * N) s p XIN DY) r :=
  bnSyncInB_shard R hR N (s.ic * s.expand) s.h s.h (nhw_ne_zero hN hh hh) _ _ _ _ _ _ (fun _ => rfl)
    (mnv4SSyncCotDn_shard R hR N hN s hh p XIN dys DY hdys) r

theorem mnv4SSyncCotEn_shard (r : Fin R) :
    mnv4SSyncCotEn R hR N s p XIN dys r
      = batchShard R N (s.ic * s.expand * s.h * s.h) (mnv4SCotEn (R * N) s p XIN DY) r := by
  unfold mnv4SSyncCotEn
  rw [mnv4SSyncCotDc_shard R hR N hN s hh p XIN dys DY hdys, dInB_shard]
  rfl

theorem mnv4SSyncCotEc_shard (r : Fin R) :
    mnv4SSyncCotEc R hR N s p XIN dys r
      = batchShard R N (s.ic * s.expand * s.h * s.h) (mnv4SCotEc (R * N) s p XIN DY) r :=
  bnSyncInB_shard R hR N (s.ic * s.expand) s.h s.h (nhw_ne_zero hN hh hh) _ _ _ _ _ _ (fun _ => rfl)
    (mnv4SSyncCotEn_shard R hR N hN s hh p XIN dys DY hdys) r

theorem mnv4SSyncCotQn_shard (r : Fin R) :
    mnv4SSyncCotQn R hR N s p XIN dys r
      = batchShard R N (s.ic * s.h * s.h) (mnv4SCotQn (R * N) s p XIN DY) r := by
  unfold mnv4SSyncCotQn
  rw [mnv4SSyncCotEc_shard R hR N hN s hh p XIN dys DY hdys, cInB_shard]
  rfl

theorem mnv4SSyncCotQc_shard (r : Fin R) :
    mnv4SSyncCotQc R hR N s p XIN dys r
      = batchShard R N (s.ic * s.h * s.h) (mnv4SCotQc (R * N) s p XIN DY) r :=
  bnSyncInB_shard R hR N s.ic s.h s.h (nhw_ne_zero hN hh hh) _ _ _ _ _ _ (fun _ => rfl)
    (mnv4SSyncCotQn_shard R hR N hN s hh p XIN dys DY hdys) r

theorem mnv4SBodySyncCotIn_shard (r : Fin R) :
    mnv4SBodySyncCotIn R hR N s p XIN dys r
      = batchShard R N (s.ic * (2 * s.h) * (2 * s.h)) (mnv4SBodyCotIn (R * N) s p XIN DY) r := by
  unfold mnv4SBodySyncCotIn
  rw [mnv4SSyncCotQc_shard R hR N hN s hh p XIN dys DY hdys, dStridedInB_shard]
  rfl

end PreStridedShard

section Stem
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat} (Ws : Kernel4 oc ic kH kW)
  (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
  (dys : Fin R → Vec (N * (oc * h * w)))

/-- Stem, replica `r`: the stem relu's mask of the cotangent the fused stage hands down. Feeds
    `sg`/`sbt`. -/
noncomputable def mnv4StemSyncCotN (r : Fin R) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (batchShard R N (oc * h * w)
      (bnBatchLA (R * N) oc h w εs γs βs (batchMap (R * N) (flatConvStride2Xla Ws bs) X)) r)
    (dys r)

/-- Stem, replica `r`: the stem BatchNorm's sync backward. Feeds `sW`; the chain stops here. -/
noncomputable def mnv4StemSyncCotC (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w εs γs
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConvStride2Xla Ws bs) X) r)
    (mnv4StemSyncCotN R N h w Ws bs εs γs βs X dys) r

end Stem

section StemShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
  (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
  (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hdys

theorem mnv4StemSyncCotN_shard (r : Fin R) :
    mnv4StemSyncCotN R N h w Ws bs εs γs βs X dys r
      = batchShard R N (oc * h * w) (mnv4StemCotN (R * N) h w Ws bs εs γs βs X DY) r := by
  unfold mnv4StemSyncCotN; rw [hdys]; rfl

include hN hh hw in
theorem mnv4StemSyncCotC_shard (r : Fin R) :
    mnv4StemSyncCotC R hR N h w Ws bs εs γs βs X dys r
      = batchShard R N (oc * h * w) (mnv4StemCotC (R * N) h w Ws bs εs γs βs X DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (mnv4StemSyncCotN_shard R N h w Ws bs εs γs βs X dys DY hdys) r

end StemShard

section Fused
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc kH kW : Nat} (Wc : Kernel4 mid ic kH kW)
  (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ)
  (γp βp : Vec oc) (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
  (dys : Fin R → Vec (N * (oc * h * w)))

/-- Fused stage, replica `r`: the project BatchNorm's sync backward of the stage-output cotangent
    (no activation after the project). Feeds `f0pW`. -/
noncomputable def mnv4FusedSyncCotPc (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w εp γp
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv Wp bp)
      (fusedConvB (R * N) (h := h) (w := w) Wc bc εc γc βc XIN)) r)
    dys r

/-- Fused stage, replica `r`: the fused BN's output cotangent, through **swish**'s backward — no
    mask. Feeds `f0cg`/`f0cbt`. -/
noncomputable def mnv4FusedSyncCotN (r : Fin R) : Vec (N * (mid * h * w)) :=
  swBackB (N * (mid * h * w))
    (batchShard R N (mid * h * w)
      (bnBatchLA (R * N) mid h w εc γc βc (batchMap (R * N) (flatConvStride2 Wc bc) XIN)) r)
    (cInB N Wp bp (mnv4FusedSyncCotPc R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys r))

/-- Fused stage, replica `r`: the fused conv's output cotangent. Feeds `f0cW`. -/
noncomputable def mnv4FusedSyncCotC (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w εc γc
    (fun r => batchShard R N (mid * h * w) (batchMap (R * N) (flatConvStride2 Wc bc) XIN) r)
    (mnv4FusedSyncCotN R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys) r

/-- Fused stage, replica `r`: the stage-INPUT cotangent — the SYMMETRIC strided conv's input-VJP,
    handed to the stem. -/
noncomputable def mnv4FusedSyncCotIn (r : Fin R) : Vec (N * (ic * (2 * h) * (2 * w))) :=
  cStridedInB N Wc bc (mnv4FusedSyncCotC R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys r)

end Fused

section FusedShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc kH kW : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid)
  (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
  (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
  (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hN hh hw hdys

theorem mnv4FusedSyncCotPc_shard (r : Fin R) :
    mnv4FusedSyncCotPc R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys r
      = batchShard R N (oc * h * w)
          (mnv4FusedCotPc (R * N) h w Wc bc εc γc βc Wp bp εp γp βp XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) hdys r

theorem mnv4FusedSyncCotN_shard (r : Fin R) :
    mnv4FusedSyncCotN R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys r
      = batchShard R N (mid * h * w)
          (mnv4FusedCotN (R * N) h w Wc bc εc γc βc Wp bp εp γp βp XIN DY) r := by
  unfold mnv4FusedSyncCotN
  rw [mnv4FusedSyncCotPc_shard R hR N h w hN hh hw Wc bc εc γc βc Wp bp εp γp βp XIN dys DY hdys,
    cInB_shard]
  rfl

theorem mnv4FusedSyncCotC_shard (r : Fin R) :
    mnv4FusedSyncCotC R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys r
      = batchShard R N (mid * h * w)
          (mnv4FusedCotC (R * N) h w Wc bc εc γc βc Wp bp εp γp βp XIN DY) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl)
    (mnv4FusedSyncCotN_shard R hR N h w hN hh hw Wc bc εc γc βc Wp bp εp γp βp XIN dys DY hdys) r

theorem mnv4FusedSyncCotIn_shard (r : Fin R) :
    mnv4FusedSyncCotIn R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w))
          (mnv4FusedCotIn (R * N) h w Wc bc εc γc βc Wp bp εp γp βp XIN DY) r := by
  unfold mnv4FusedSyncCotIn
  rw [mnv4FusedSyncCotC_shard R hR N h w hN hh hw Wc bc εc γc βc Wp bp εp γp βp XIN dys DY hdys,
    cStridedInB_shard]
  rfl

end FusedShard

section Head
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid oc nCls : Nat}
  (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
  (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
  (Wd : Mat oc nCls) (bd : Vec nCls) (XIN : Vec ((R * N) * (c * h * w)))
  (gs : Fin R → Vec (N * nCls))

/-- Head, replica `r`: the second head relu's mask of the GAP/dense tail's input-VJP of this
    replica's loss cotangent. Feeds `hg`/`hbt`. -/
noncomputable def mnv4HeadSyncCotHn (r : Fin R) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (batchShard R N (oc * h * w) (bnBatchLA (R * N) oc h w ε2 γ2 β2
      (batchMap (R * N) (flatConv W2 b2) (cbReluB (R * N) (h := h) (w := w) W1 b1 ε1 γ1 β1 XIN))) r)
    (r34HeadCotBlk N h w Wd bd
      (batchShard R N (oc * h * w) (cbReluB (R * N) (h := h) (w := w) W2 b2 ε2 γ2 β2
        (cbReluB (R * N) (h := h) (w := w) W1 b1 ε1 γ1 β1 XIN)) r) (gs r))

/-- Head, replica `r`: the second head BatchNorm's sync backward. Feeds `hW`. -/
noncomputable def mnv4HeadSyncCotHc (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w ε2 γ2
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv W2 b2)
      (cbReluB (R * N) (h := h) (w := w) W1 b1 ε1 γ1 β1 XIN)) r)
    (mnv4HeadSyncCotHn R N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs) r

/-- Head, replica `r`: the first head relu's mask. Feeds `h1g`/`h1bt`. -/
noncomputable def mnv4HeadSyncCotH1n (r : Fin R) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (batchShard R N (mid * h * w)
      (bnBatchLA (R * N) mid h w ε1 γ1 β1 (batchMap (R * N) (flatConv W1 b1) XIN)) r)
    (cInB N W2 b2 (mnv4HeadSyncCotHc R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs r))

/-- Head, replica `r`: the first head BatchNorm's sync backward. Feeds `h1W`. -/
noncomputable def mnv4HeadSyncCotH1c (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w ε1 γ1
    (fun r => batchShard R N (mid * h * w) (batchMap (R * N) (flatConv W1 b1) XIN) r)
    (mnv4HeadSyncCotH1n R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs) r

/-- Head, replica `r`: the cotangent handed to block 21. -/
noncomputable def mnv4HeadSyncCotIn (r : Fin R) : Vec (N * (c * h * w)) :=
  cInB N W1 b1 (mnv4HeadSyncCotH1c R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs r)

end Head

section HeadShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid oc nCls : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
  (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
  (Wd : Mat oc nCls) (bd : Vec nCls) (XIN : Vec ((R * N) * (c * h * w)))
  (gs : Fin R → Vec (N * nCls)) (G : Vec ((R * N) * nCls))
  (hgs : ∀ r, gs r = batchShard R N nCls G r)
include hgs

theorem mnv4HeadSyncCotHn_shard (r : Fin R) :
    mnv4HeadSyncCotHn R N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs r
      = batchShard R N (oc * h * w)
          (mnv4HeadCotHn (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G) r := by
  unfold mnv4HeadSyncCotHn
  rw [hgs, r34HeadCotBlk_shard]
  rfl

include hN hh hw in
theorem mnv4HeadSyncCotHc_shard (r : Fin R) :
    mnv4HeadSyncCotHc R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs r
      = batchShard R N (oc * h * w)
          (mnv4HeadCotHc (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl)
    (mnv4HeadSyncCotHn_shard R N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs G hgs) r

include hN hh hw in
theorem mnv4HeadSyncCotH1n_shard (r : Fin R) :
    mnv4HeadSyncCotH1n R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs r
      = batchShard R N (mid * h * w)
          (mnv4HeadCotH1n (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G) r := by
  unfold mnv4HeadSyncCotH1n
  rw [mnv4HeadSyncCotHc_shard R hR N h w hN hh hw W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs G hgs,
    cInB_shard]
  rfl

include hN hh hw in
theorem mnv4HeadSyncCotH1c_shard (r : Fin R) :
    mnv4HeadSyncCotH1c R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs r
      = batchShard R N (mid * h * w)
          (mnv4HeadCotH1c (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl)
    (mnv4HeadSyncCotH1n_shard R hR N h w hN hh hw W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs G
      hgs) r

include hN hh hw in
theorem mnv4HeadSyncCotIn_shard (r : Fin R) :
    mnv4HeadSyncCotIn R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs r
      = batchShard R N (c * h * w)
          (mnv4HeadCotIn (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G) r := by
  unfold mnv4HeadSyncCotIn
  rw [mnv4HeadSyncCotH1c_shard R hR N h w hN hh hw W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs G
    hgs, cInB_shard]
  rfl

end HeadShard

-- ════════════════════════════════════════════════════════════════
-- § 4. The scaled-shard invariant, block by block
--   Replicas at `R ×` the shards of a global block-output cotangent hand the previous block `R ×`
--   the shards of the single-device block-input cotangent: sharding (§2–3) then homogeneity (§1).
-- ════════════════════════════════════════════════════════════════

theorem mnv4BodySyncCotIn_scaled (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (s : UibSpec)
    (hh : 0 < s.h) (p : UibParams s) (XIN : Vec ((R * N) * (s.ic * s.h * s.h)))
    (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h)))
    (hdys : ∀ r, dys r = batchShard R N (s.oc * s.h * s.h) (fun i => (R : ℝ) * DY i) r)
    (r : Fin R) :
    mnv4BodySyncCotIn R hR N s p XIN dys r
      = batchShard R N (s.ic * s.h * s.h) (fun i => (R : ℝ) * mnv4BodyCotIn (R * N) s p XIN DY i) r := by
  rw [mnv4BodySyncCotIn_shard R hR N hN s hh p XIN dys _ hdys, mnv4BodyCotIn_smul]

/-- ⭐ **The skip fan-in carries the invariant** — `body dx + dyOut`, each at `R ×` its shard, is
    `R ×` the shard of the sum. Generic in the width, so it applies at the concrete rows where
    `s.oc = s.ic` is definitional, as T3's `mnv4SkipCotIn` is. -/
theorem mnv4SkipSyncCotIn_scaled {R N n : Nat} (a b : Fin R → Vec (N * n)) (A B : Vec ((R * N) * n))
    (ha : ∀ r, a r = batchShard R N n (fun i => (R : ℝ) * A i) r)
    (hb : ∀ r, b r = batchShard R N n (fun i => (R : ℝ) * B i) r) (r : Fin R) :
    mnv4SkipCotIn (a r) (b r) = batchShard R N n (fun i => (R : ℝ) * mnv4SkipCotIn A B i) r := by
  rw [ha, hb]
  funext i
  simp only [mnv4SkipCotIn, batchShard]
  ring

theorem mnv4SBodySyncCotIn_scaled (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (s : UibSpec)
    (hh : 0 < s.h) (p : UibParams s) (XIN : Vec ((R * N) * (s.ic * (2 * s.h) * (2 * s.h))))
    (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h)))
    (hdys : ∀ r, dys r = batchShard R N (s.oc * s.h * s.h) (fun i => (R : ℝ) * DY i) r)
    (r : Fin R) :
    mnv4SBodySyncCotIn R hR N s p XIN dys r
      = batchShard R N (s.ic * (2 * s.h) * (2 * s.h))
          (fun i => (R : ℝ) * mnv4SBodyCotIn (R * N) s p XIN DY i) r := by
  rw [mnv4SBodySyncCotIn_shard R hR N hN s hh p XIN dys _ hdys, mnv4SBodyCotIn_smul]

theorem mnv4FusedSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc kH kW : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ)
    (γc βc : Vec mid) (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    mnv4FusedSyncCotIn R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w))
          (fun i => (R : ℝ) * mnv4FusedCotIn (R * N) h w Wc bc εc γc βc Wp bp εp γp βp XIN DY i) r := by
  rw [mnv4FusedSyncCotIn_shard R hR N h w hN hh hw Wc bc εc γc βc Wp bp εp γp βp XIN dys _ hdys,
    mnv4FusedCotIn_smul]

theorem mnv4HeadSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid oc nCls : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (XIN : Vec ((R * N) * (c * h * w)))
    (gs : Fin R → Vec (N * nCls)) (G : Vec ((R * N) * nCls))
    (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r) (r : Fin R) :
    mnv4HeadSyncCotIn R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs r
      = batchShard R N (c * h * w)
          (fun i => (R : ℝ) * mnv4HeadCotIn (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G i)
          r := by
  rw [mnv4HeadSyncCotIn_shard R hR N h w hN hh hw W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs _
    hgs, mnv4HeadCotIn_smul]

-- ════════════════════════════════════════════════════════════════
-- § 5. The per-block DP ties
--   Parameter tags are the render's names without `%` (`u{p}qW`, `u{p}qg`, …, read off the row);
--   a BatchNorm's statistics collectives are its γ tag with `mu` / `var` (inside `BnSync`).
-- ════════════════════════════════════════════════════════════════

/-- **ExtraDW-profile stride-1 block, DP-tied — its twelve emitted collectives** (`u{p}qW qg qbt
    eW eg ebt dW dg dbt pW pg pbt`), each the single-device node at the global batch, at T3's chain
    cotangents there. The skip changes only the cotangent handed down, never a parameter's. -/
def mnv4ExtraDWSyncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec)
    (xN cotN vN epsStr : String) (p : UibParams s) (XIN : Vec ((R * N) * (s.ic * s.h * s.h)))
    (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) (R * N) s.preDWk p.Wq p.bq p.eq_ p.hq p.gq
    p.bq2).fwd XIN
  let er := (cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) (R * N) s.postDWk p.Wd p.bd p.ed p.hd p.gd
    p.bd2).fwd er
  let qc := batchMap (R * N) (depthwiseFlat p.Wq p.bq) XIN
  let ec := batchMap (R * N) (flatConv p.We p.be) qr
  let dc := batchMap (R * N) (depthwiseFlat p.Wd p.bd) er
  let pc := batchMap (R * N) (flatConv p.Wz p.bz) dr
  DepthwiseWSync R hR N s.h s.h s!"u{s.p}qW" xN cotN p.bq XIN p.Wq
      (mnv4SyncCotQc R hR N s p XIN dys) (mnv4CotQc (R * N) s p XIN DY)
  ∧ BnSync R hR N s.ic s.h s.h s!"u{s.p}qg" s!"u{s.p}qbt" vN epsStr cotN p.eq_ qc
      (mnv4SyncCotQn R hR N s p XIN dys) (mnv4CotQn (R * N) s p XIN DY)
  ∧ ConvWSync R hR N s.h s.h s!"u{s.p}eW" xN cotN p.be qr p.We
      (mnv4SyncCotEc R hR N s p XIN dys) (mnv4CotEc (R * N) s p XIN DY)
  ∧ BnSync R hR N (s.ic * s.expand) s.h s.h s!"u{s.p}eg" s!"u{s.p}ebt" vN epsStr cotN p.ee ec
      (mnv4SyncCotEn R hR N s p XIN dys) (mnv4CotEn (R * N) s p XIN DY)
  ∧ DepthwiseWSync R hR N s.h s.h s!"u{s.p}dW" xN cotN p.bd er p.Wd
      (mnv4SyncCotDc R hR N s p XIN dys) (mnv4CotDc (R * N) s p XIN DY)
  ∧ BnSync R hR N (s.ic * s.expand) s.h s.h s!"u{s.p}dg" s!"u{s.p}dbt" vN epsStr cotN p.ed dc
      (mnv4SyncCotDn R hR N s p XIN dys) (mnv4CotDn (R * N) s p XIN DY)
  ∧ ConvWSync R hR N s.h s.h s!"u{s.p}pW" xN cotN p.bz dr p.Wz
      (mnv4SyncCotPc R hR N s p XIN dys) (mnv4CotPc (R * N) s p XIN DY)
  ∧ BnSync R hR N s.oc s.h s.h s!"u{s.p}pg" s!"u{s.p}pbt" vN epsStr cotN p.ez pc dys DY

theorem mnv4_extradw_syncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (s : UibSpec)
    (hh : 0 < s.h) (xN cotN vN epsStr : String) (p : UibParams s)
    (XIN : Vec ((R * N) * (s.ic * s.h * s.h))) (dys : Fin R → Vec (N * (s.oc * s.h * s.h)))
    (DY : Vec ((R * N) * (s.oc * s.h * s.h)))
    (hdys : ∀ r, dys r = batchShard R N (s.oc * s.h * s.h) (fun i => (R : ℝ) * DY i) r) :
    mnv4ExtraDWSyncTiedB R hR N s xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hh
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact depthwiseWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotQc_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotQc_smul])
  · exact bnSync_of_scaled R hR N s.ic s.h s.h hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotQn_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotQn_smul])
  · exact convWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotEc_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotEc_smul])
  · exact bnSync_of_scaled R hR N (s.ic * s.expand) s.h s.h hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotEn_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotEn_smul])
  · exact depthwiseWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotDc_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotDc_smul])
  · exact bnSync_of_scaled R hR N (s.ic * s.expand) s.h s.h hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotDn_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotDn_smul])
  · exact convWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotPc_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotPc_smul])
  · exact bnSync_of_scaled R hR N s.oc s.h s.h hm _ _ _ _ _ _ _ _ _ hdys

/-- **ConvNeXt-like block (`postDWk = 0`), DP-tied — its nine emitted collectives** (no `d`). -/
def mnv4ConvNeXtSyncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec)
    (xN cotN vN epsStr : String) (p : UibParams s) (XIN : Vec ((R * N) * (s.ic * s.h * s.h)))
    (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) (R * N) s.preDWk p.Wq p.bq p.eq_ p.hq p.gq
    p.bq2).fwd XIN
  let er := (cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) (R * N) s.postDWk p.Wd p.bd p.ed p.hd p.gd
    p.bd2).fwd er
  let qc := batchMap (R * N) (depthwiseFlat p.Wq p.bq) XIN
  let ec := batchMap (R * N) (flatConv p.We p.be) qr
  let pc := batchMap (R * N) (flatConv p.Wz p.bz) dr
  DepthwiseWSync R hR N s.h s.h s!"u{s.p}qW" xN cotN p.bq XIN p.Wq
      (mnv4SyncCotQc R hR N s p XIN dys) (mnv4CotQc (R * N) s p XIN DY)
  ∧ BnSync R hR N s.ic s.h s.h s!"u{s.p}qg" s!"u{s.p}qbt" vN epsStr cotN p.eq_ qc
      (mnv4SyncCotQn R hR N s p XIN dys) (mnv4CotQn (R * N) s p XIN DY)
  ∧ ConvWSync R hR N s.h s.h s!"u{s.p}eW" xN cotN p.be qr p.We
      (mnv4SyncCotEc R hR N s p XIN dys) (mnv4CotEc (R * N) s p XIN DY)
  ∧ BnSync R hR N (s.ic * s.expand) s.h s.h s!"u{s.p}eg" s!"u{s.p}ebt" vN epsStr cotN p.ee ec
      (mnv4SyncCotEn R hR N s p XIN dys) (mnv4CotEn (R * N) s p XIN DY)
  ∧ ConvWSync R hR N s.h s.h s!"u{s.p}pW" xN cotN p.bz dr p.Wz
      (mnv4SyncCotPc R hR N s p XIN dys) (mnv4CotPc (R * N) s p XIN DY)
  ∧ BnSync R hR N s.oc s.h s.h s!"u{s.p}pg" s!"u{s.p}pbt" vN epsStr cotN p.ez pc dys DY

theorem mnv4_convnext_syncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (s : UibSpec)
    (hh : 0 < s.h) (xN cotN vN epsStr : String) (p : UibParams s)
    (XIN : Vec ((R * N) * (s.ic * s.h * s.h))) (dys : Fin R → Vec (N * (s.oc * s.h * s.h)))
    (DY : Vec ((R * N) * (s.oc * s.h * s.h)))
    (hdys : ∀ r, dys r = batchShard R N (s.oc * s.h * s.h) (fun i => (R : ℝ) * DY i) r) :
    mnv4ConvNeXtSyncTiedB R hR N s xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hh
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact depthwiseWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotQc_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotQc_smul])
  · exact bnSync_of_scaled R hR N s.ic s.h s.h hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotQn_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotQn_smul])
  · exact convWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotEc_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotEc_smul])
  · exact bnSync_of_scaled R hR N (s.ic * s.expand) s.h s.h hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotEn_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotEn_smul])
  · exact convWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotPc_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotPc_smul])
  · exact bnSync_of_scaled R hR N s.oc s.h s.h hm _ _ _ _ _ _ _ _ _ hdys

/-- **FFN block (neither depthwise), DP-tied — its six emitted collectives** (`eW eg ebt pW pg
    pbt`). -/
def mnv4FfnSyncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec)
    (xN cotN vN epsStr : String) (p : UibParams s) (XIN : Vec ((R * N) * (s.ic * s.h * s.h)))
    (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4PreDWSlot (h := s.h) (w := s.h) (R * N) s.preDWk p.Wq p.bq p.eq_ p.hq p.gq
    p.bq2).fwd XIN
  let er := (cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) (R * N) s.postDWk p.Wd p.bd p.ed p.hd p.gd
    p.bd2).fwd er
  let ec := batchMap (R * N) (flatConv p.We p.be) qr
  let pc := batchMap (R * N) (flatConv p.Wz p.bz) dr
  ConvWSync R hR N s.h s.h s!"u{s.p}eW" xN cotN p.be qr p.We
      (mnv4SyncCotEc R hR N s p XIN dys) (mnv4CotEc (R * N) s p XIN DY)
  ∧ BnSync R hR N (s.ic * s.expand) s.h s.h s!"u{s.p}eg" s!"u{s.p}ebt" vN epsStr cotN p.ee ec
      (mnv4SyncCotEn R hR N s p XIN dys) (mnv4CotEn (R * N) s p XIN DY)
  ∧ ConvWSync R hR N s.h s.h s!"u{s.p}pW" xN cotN p.bz dr p.Wz
      (mnv4SyncCotPc R hR N s p XIN dys) (mnv4CotPc (R * N) s p XIN DY)
  ∧ BnSync R hR N s.oc s.h s.h s!"u{s.p}pg" s!"u{s.p}pbt" vN epsStr cotN p.ez pc dys DY

theorem mnv4_ffn_syncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (s : UibSpec)
    (hh : 0 < s.h) (xN cotN vN epsStr : String) (p : UibParams s)
    (XIN : Vec ((R * N) * (s.ic * s.h * s.h))) (dys : Fin R → Vec (N * (s.oc * s.h * s.h)))
    (DY : Vec ((R * N) * (s.oc * s.h * s.h)))
    (hdys : ∀ r, dys r = batchShard R N (s.oc * s.h * s.h) (fun i => (R : ℝ) * DY i) r) :
    mnv4FfnSyncTiedB R hR N s xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hh
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotEc_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotEc_smul])
  · exact bnSync_of_scaled R hR N (s.ic * s.expand) s.h s.h hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotEn_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotEn_smul])
  · exact convWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SyncCotPc_shard R hR N hN s hh p XIN dys _ hdys, mnv4CotPc_smul])
  · exact bnSync_of_scaled R hR N s.oc s.h s.h hm _ _ _ _ _ _ _ _ _ hdys

/-- **Pre-strided block (rows 1, 3, 11), DP-tied — its twelve emitted collectives**, the leading
    one the SYMMETRIC strided depthwise weight `depthwiseStridedWeightGradB`. -/
def mnv4PreStridedSyncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (s : UibSpec)
    (xN cotN vN epsStr : String) (p : UibParams s)
    (XIN : Vec ((R * N) * (s.ic * (2 * s.h) * (2 * s.h))))
    (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h))) : Prop :=
  let qr := (mnv4DWReluStridedLayer (h := s.h) (w := s.h) (R * N) p.Wq p.bq p.eq_ p.hq p.gq
    p.bq2).fwd XIN
  let er := (cbReluLayer (h := s.h) (w := s.h) (R * N) p.We p.be p.ee p.he p.ge p.be2).fwd qr
  let dr := (mnv4PostDWSlot (h := s.h) (w := s.h) (R * N) s.postDWk p.Wd p.bd p.ed p.hd p.gd
    p.bd2).fwd er
  let qc := batchMap (R * N) (depthwiseStride2Flat p.Wq p.bq) XIN
  let ec := batchMap (R * N) (flatConv p.We p.be) qr
  let dc := batchMap (R * N) (depthwiseFlat p.Wd p.bd) er
  let pc := batchMap (R * N) (flatConv p.Wz p.bz) dr
  DepthwiseStridedWSync R hR N s.h s.h s!"u{s.p}qW" xN cotN p.bq XIN p.Wq
      (mnv4SSyncCotQc R hR N s p XIN dys) (mnv4SCotQc (R * N) s p XIN DY)
  ∧ BnSync R hR N s.ic s.h s.h s!"u{s.p}qg" s!"u{s.p}qbt" vN epsStr cotN p.eq_ qc
      (mnv4SSyncCotQn R hR N s p XIN dys) (mnv4SCotQn (R * N) s p XIN DY)
  ∧ ConvWSync R hR N s.h s.h s!"u{s.p}eW" xN cotN p.be qr p.We
      (mnv4SSyncCotEc R hR N s p XIN dys) (mnv4SCotEc (R * N) s p XIN DY)
  ∧ BnSync R hR N (s.ic * s.expand) s.h s.h s!"u{s.p}eg" s!"u{s.p}ebt" vN epsStr cotN p.ee ec
      (mnv4SSyncCotEn R hR N s p XIN dys) (mnv4SCotEn (R * N) s p XIN DY)
  ∧ DepthwiseWSync R hR N s.h s.h s!"u{s.p}dW" xN cotN p.bd er p.Wd
      (mnv4SSyncCotDc R hR N s p XIN dys) (mnv4SCotDc (R * N) s p XIN DY)
  ∧ BnSync R hR N (s.ic * s.expand) s.h s.h s!"u{s.p}dg" s!"u{s.p}dbt" vN epsStr cotN p.ed dc
      (mnv4SSyncCotDn R hR N s p XIN dys) (mnv4SCotDn (R * N) s p XIN DY)
  ∧ ConvWSync R hR N s.h s.h s!"u{s.p}pW" xN cotN p.bz dr p.Wz
      (mnv4SSyncCotPc R hR N s p XIN dys) (mnv4SCotPc (R * N) s p XIN DY)
  ∧ BnSync R hR N s.oc s.h s.h s!"u{s.p}pg" s!"u{s.p}pbt" vN epsStr cotN p.ez pc dys DY

theorem mnv4_prestrided_syncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (s : UibSpec)
    (hh : 0 < s.h) (xN cotN vN epsStr : String) (p : UibParams s)
    (XIN : Vec ((R * N) * (s.ic * (2 * s.h) * (2 * s.h))))
    (dys : Fin R → Vec (N * (s.oc * s.h * s.h))) (DY : Vec ((R * N) * (s.oc * s.h * s.h)))
    (hdys : ∀ r, dys r = batchShard R N (s.oc * s.h * s.h) (fun i => (R : ℝ) * DY i) r) :
    mnv4PreStridedSyncTiedB R hR N s xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hh
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact depthwiseStridedWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SSyncCotQc_shard R hR N hN s hh p XIN dys _ hdys, mnv4SCotQc_smul])
  · exact bnSync_of_scaled R hR N s.ic s.h s.h hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SSyncCotQn_shard R hR N hN s hh p XIN dys _ hdys, mnv4SCotQn_smul])
  · exact convWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SSyncCotEc_shard R hR N hN s hh p XIN dys _ hdys, mnv4SCotEc_smul])
  · exact bnSync_of_scaled R hR N (s.ic * s.expand) s.h s.h hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SSyncCotEn_shard R hR N hN s hh p XIN dys _ hdys, mnv4SCotEn_smul])
  · exact depthwiseWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SSyncCotDc_shard R hR N hN s hh p XIN dys _ hdys, mnv4SCotDc_smul])
  · exact bnSync_of_scaled R hR N (s.ic * s.expand) s.h s.h hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SSyncCotDn_shard R hR N hN s hh p XIN dys _ hdys, mnv4SCotDn_smul])
  · exact convWSync_of_scaled R hR N s.h s.h _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4SSyncCotPc_shard R hR N hN s hh p XIN dys _ hdys, mnv4SCotPc_smul])
  · exact bnSync_of_scaled R hR N s.oc s.h s.h hm _ _ _ _ _ _ _ _ _ hdys

/-- **Stem, DP-tied** — the 3×3/s2 XLA-`SAME` conv weight and its BatchNorm's γ and β. -/
def mnv4StemSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ)
    (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  ConvStridedXlaWSync R hR N h w "sW" xN cotN bs X Ws
      (mnv4StemSyncCotC R hR N h w Ws bs εs γs βs X dys)
      (mnv4StemCotC (R * N) h w Ws bs εs γs βs X DY)
  ∧ BnSync R hR N oc h w "sg" "sbt" vN epsStr cotN εs
      (batchMap (R * N) (flatConvStride2Xla Ws bs) X)
      (mnv4StemSyncCotN R N h w Ws bs εs γs βs X dys)
      (mnv4StemCotN (R * N) h w Ws bs εs γs βs X DY)

theorem mnv4_stem_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic kH kW) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    mnv4StemSyncTiedB R hR N h w xN cotN vN epsStr Ws bs εs γs βs X dys DY := by
  refine ⟨?_, ?_⟩
  · exact convStridedXlaWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4StemSyncCotC_shard R hR N h w hN hh hw Ws bs εs γs βs X dys _ hdys,
        mnv4StemCotC_smul])
  · exact bnSync_of_scaled R hR N oc h w (nhw_ne_zero hN hh hw) _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4StemSyncCotN_shard R N h w Ws bs εs γs βs X dys _ hdys, mnv4StemCotN_smul])

/-- **Fused stage, DP-tied** — its six emitted collectives: the SYMMETRIC strided conv weight
    (`f0cW`, where the stem's is the XLA-`SAME` twin), the fused BN's γ/β (through swish), the
    project weight and the project BN's γ/β. -/
def mnv4FusedSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc kH kW : Nat}
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (xN cotN vN epsStr : String) (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  let sw := fusedConvB (R * N) (h := h) (w := w) Wc bc εc γc βc XIN
  let fc := batchMap (R * N) (flatConvStride2 Wc bc) XIN
  let pc := batchMap (R * N) (flatConv Wp bp) sw
  ConvStridedWSync R hR N h w "f0cW" xN cotN bc XIN Wc
      (mnv4FusedSyncCotC R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys)
      (mnv4FusedCotC (R * N) h w Wc bc εc γc βc Wp bp εp γp βp XIN DY)
  ∧ BnSync R hR N mid h w "f0cg" "f0cbt" vN epsStr cotN εc fc
      (mnv4FusedSyncCotN R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys)
      (mnv4FusedCotN (R * N) h w Wc bc εc γc βc Wp bp εp γp βp XIN DY)
  ∧ ConvWSync R hR N h w "f0pW" xN cotN bp sw Wp
      (mnv4FusedSyncCotPc R hR N h w Wc bc εc γc βc Wp bp εp γp XIN dys)
      (mnv4FusedCotPc (R * N) h w Wc bc εc γc βc Wp bp εp γp βp XIN DY)
  ∧ BnSync R hR N oc h w "f0pg" "f0pbt" vN epsStr cotN εp pc dys DY

theorem mnv4_fused_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc kH kW : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (Wc : Kernel4 mid ic kH kW) (bc : Vec mid) (εc : ℝ) (γc βc : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (xN cotN vN epsStr : String) (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    mnv4FusedSyncTiedB R hR N h w Wc bc εc γc βc Wp bp εp γp βp xN cotN vN epsStr XIN dys DY := by
  have hm := nhw_ne_zero hN hh hw
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact convStridedWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4FusedSyncCotC_shard R hR N h w hN hh hw Wc bc εc γc βc Wp bp εp γp βp XIN dys _
        hdys, mnv4FusedCotC_smul])
  · exact bnSync_of_scaled R hR N mid h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4FusedSyncCotN_shard R hR N h w hN hh hw Wc bc εc γc βc Wp bp εp γp βp XIN dys _
        hdys, mnv4FusedCotN_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4FusedSyncCotPc_shard R hR N h w hN hh hw Wc bc εc γc βc Wp bp εp γp βp XIN dys _
        hdys, mnv4FusedCotPc_smul])
  · exact bnSync_of_scaled R hR N oc h w hm _ _ _ _ _ _ _ _ _ hdys

/-- **Head, DP-tied** — all eight emitted collectives: the two 1×1 conv weights (`h1W` 256 → 960,
    `hW` 960 → 1280), their BatchNorms' γ and β, and the classifier's weight and bias at the GAP
    output (`ResNet34SyncTieB.r34HeadSyncTiedB`, reused: MNv4's GAP-and-dense tail is ResNet-34's). -/
def mnv4HeadSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid oc nCls : Nat}
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (xN cotN vN epsStr : String)
    (XIN : Vec ((R * N) * (c * h * w))) (gs : Fin R → Vec (N * nCls)) (G : Vec ((R * N) * nCls)) :
    Prop :=
  let r1 := cbReluB (R * N) (h := h) (w := w) W1 b1 ε1 γ1 β1 XIN
  let r2 := cbReluB (R * N) (h := h) (w := w) W2 b2 ε2 γ2 β2 r1
  let c1 := batchMap (R * N) (flatConv W1 b1) XIN
  let c2 := batchMap (R * N) (flatConv W2 b2) r1
  ConvWSync R hR N h w "h1W" xN cotN b1 XIN W1
      (mnv4HeadSyncCotH1c R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs)
      (mnv4HeadCotH1c (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G)
  ∧ BnSync R hR N mid h w "h1g" "h1bt" vN epsStr cotN ε1 c1
      (mnv4HeadSyncCotH1n R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs)
      (mnv4HeadCotH1n (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G)
  ∧ ConvWSync R hR N h w "hW" xN cotN b2 r1 W2
      (mnv4HeadSyncCotHc R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs)
      (mnv4HeadCotHc (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G)
  ∧ BnSync R hR N oc h w "hg" "hbt" vN epsStr cotN ε2 c2
      (mnv4HeadSyncCotHn R N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs)
      (mnv4HeadCotHn (R * N) h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN G)
  ∧ r34HeadSyncTiedB R hR N h w xN cotN r2 gs G

theorem mnv4_head_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {c mid oc nCls : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
    (W1 : Kernel4 mid c 1 1) (b1 : Vec mid) (ε1 : ℝ) (γ1 β1 : Vec mid)
    (W2 : Kernel4 oc mid 1 1) (b2 : Vec oc) (ε2 : ℝ) (γ2 β2 : Vec oc)
    (Wd : Mat oc nCls) (bd : Vec nCls) (xN cotN vN epsStr : String)
    (XIN : Vec ((R * N) * (c * h * w))) (gs : Fin R → Vec (N * nCls)) (G : Vec ((R * N) * nCls))
    (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r) :
    mnv4HeadSyncTiedB R hR N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd xN cotN vN epsStr XIN gs G := by
  have hm := nhw_ne_zero hN hh hw
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4HeadSyncCotH1c_shard R hR N h w hN hh hw W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs
        _ hgs, mnv4HeadCotH1c_smul])
  · exact bnSync_of_scaled R hR N mid h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4HeadSyncCotH1n_shard R hR N h w hN hh hw W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs
        _ hgs, mnv4HeadCotH1n_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4HeadSyncCotHc_shard R hR N h w hN hh hw W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs
        _ hgs, mnv4HeadCotHc_smul])
  · exact bnSync_of_scaled R hR N oc h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [mnv4HeadSyncCotHn_shard R N h w W1 b1 ε1 γ1 β1 W2 b2 ε2 γ2 β2 Wd bd XIN gs _ hgs,
        mnv4HeadCotHn_smul])
  · exact r34_head_syncTiedB R hR N h w xN cotN _ gs G hgs

-- ════════════════════════════════════════════════════════════════
-- § 6. The whole-net capstone
-- ════════════════════════════════════════════════════════════════

/-- **The whole-net statement, named** — so the capstone (cotangents bound) and its smoothed-CE
    corollary (cotangents instantiated) state exactly one thing. The first 23 `let`s are
    `mnv4_net_tiedB`'s chain at `N := R·N`, driven by the global cotangent `G`; the next 23 are the
    replicas' sync-BN chain, driven by the family `gs`; the 24 conjuncts are one per stage, every
    emitted parameter collective against T3's node at the global batch. -/
def mnv4NetSyncTiedB (R : Nat) (hR : 0 < R) (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : Mnv4BWeights nCls) (X : Vec ((R * N) * (3 * 224 * 224))) (G : Vec ((R * N) * nCls))
    (gs : Fin R → Vec (N * nCls)) : Prop :=
  -- ── the single-device chain at the global batch `R·N` (T3's) ──
  let dy21 := mnv4HeadCotIn (R * N) 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
               w.Wd w.bd (mnv4Blk21 (R * N) w X) G
  let dy20 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row21 w.b21 (mnv4Blk20 (R * N) w X) dy21) dy21
  let dy19 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row20 w.b20 (mnv4Blk19 (R * N) w X) dy20) dy20
  let dy18 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row19 w.b19 (mnv4Blk18 (R * N) w X) dy19) dy19
  let dy17 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row18 w.b18 (mnv4Blk17 (R * N) w X) dy18) dy18
  let dy16 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row17 w.b17 (mnv4Blk16 (R * N) w X) dy17) dy17
  let dy15 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row16 w.b16 (mnv4Blk15 (R * N) w X) dy16) dy16
  let dy14 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row15 w.b15 (mnv4Blk14 (R * N) w X) dy15) dy15
  let dy13 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row14 w.b14 (mnv4Blk13 (R * N) w X) dy14) dy14
  let dy12 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row13 w.b13 (mnv4Blk12 (R * N) w X) dy13) dy13
  let dy11 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row12 w.b12 (mnv4Blk11 (R * N) w X) dy12) dy12
  let dy10 := mnv4SBodyCotIn (R * N) mnv4Row11 w.b11 (mnv4Blk10 (R * N) w X) dy11
  let dy9 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row10 w.b10 (mnv4Blk9 (R * N) w X) dy10) dy10
  let dy8 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row9 w.b9 (mnv4Blk8 (R * N) w X) dy9) dy9
  let dy7 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row8 w.b8 (mnv4Blk7 (R * N) w X) dy8) dy8
  let dy6 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row7 w.b7 (mnv4Blk6 (R * N) w X) dy7) dy7
  let dy5 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row6 w.b6 (mnv4Blk5 (R * N) w X) dy6) dy6
  let dy4 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row5 w.b5 (mnv4Blk4 (R * N) w X) dy5) dy5
  let dy3 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row4 w.b4 (mnv4Blk3 (R * N) w X) dy4) dy4
  let dy2 := mnv4SBodyCotIn (R * N) mnv4Row3 w.b3 (mnv4Blk2 (R * N) w X) dy3
  let dy1 := mnv4SkipCotIn (mnv4BodyCotIn (R * N) mnv4Row2 w.b2 (mnv4Blk1 (R * N) w X) dy2) dy2
  let dy0 := mnv4SBodyCotIn (R * N) mnv4Row1 w.b1 (mnv4Blk0 (R * N) w X) dy1
  let dyStem := mnv4FusedCotIn (R * N) 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                 w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt (mnv4Pre0 (R * N) w X) dy0
  -- ── replica `r`: its own sync-BN chain, from its own loss cotangent `gs r` ──
  let e21 := mnv4HeadSyncCotIn R hR N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
               w.Wd w.bd (mnv4Blk21 (R * N) w X) gs
  let e20 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row21 w.b21 (mnv4Blk20 (R * N) w X) e21 r) (e21 r)
  let e19 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row20 w.b20 (mnv4Blk19 (R * N) w X) e20 r) (e20 r)
  let e18 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row19 w.b19 (mnv4Blk18 (R * N) w X) e19 r) (e19 r)
  let e17 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row18 w.b18 (mnv4Blk17 (R * N) w X) e18 r) (e18 r)
  let e16 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row17 w.b17 (mnv4Blk16 (R * N) w X) e17 r) (e17 r)
  let e15 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row16 w.b16 (mnv4Blk15 (R * N) w X) e16 r) (e16 r)
  let e14 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row15 w.b15 (mnv4Blk14 (R * N) w X) e15 r) (e15 r)
  let e13 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row14 w.b14 (mnv4Blk13 (R * N) w X) e14 r) (e14 r)
  let e12 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row13 w.b13 (mnv4Blk12 (R * N) w X) e13 r) (e13 r)
  let e11 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row12 w.b12 (mnv4Blk11 (R * N) w X) e12 r) (e12 r)
  let e10 := mnv4SBodySyncCotIn R hR N mnv4Row11 w.b11 (mnv4Blk10 (R * N) w X) e11
  let e9 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row10 w.b10 (mnv4Blk9 (R * N) w X) e10 r) (e10 r)
  let e8 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row9 w.b9 (mnv4Blk8 (R * N) w X) e9 r) (e9 r)
  let e7 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row8 w.b8 (mnv4Blk7 (R * N) w X) e8 r) (e8 r)
  let e6 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row7 w.b7 (mnv4Blk6 (R * N) w X) e7 r) (e7 r)
  let e5 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row6 w.b6 (mnv4Blk5 (R * N) w X) e6 r) (e6 r)
  let e4 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row5 w.b5 (mnv4Blk4 (R * N) w X) e5 r) (e5 r)
  let e3 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row4 w.b4 (mnv4Blk3 (R * N) w X) e4 r) (e4 r)
  let e2 := mnv4SBodySyncCotIn R hR N mnv4Row3 w.b3 (mnv4Blk2 (R * N) w X) e3
  let e1 := fun r => mnv4SkipCotIn
    (mnv4BodySyncCotIn R hR N mnv4Row2 w.b2 (mnv4Blk1 (R * N) w X) e2 r) (e2 r)
  let e0 := mnv4SBodySyncCotIn R hR N mnv4Row1 w.b1 (mnv4Blk0 (R * N) w X) e1
  let eStem := mnv4FusedSyncCotIn R hR N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
                 w.f0pW w.f0pb w.f0pE w.f0pg (mnv4Pre0 (R * N) w X) e0
  mnv4StemSyncTiedB R hR N 112 112 xN cotN vN epsStr w.sW w.sb w.sE w.sg w.sbt X eStem dyStem
  ∧ mnv4FusedSyncTiedB R hR N 56 56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt w.f0pW w.f0pb w.f0pE
      w.f0pg w.f0pbt xN cotN vN epsStr (mnv4Pre0 (R * N) w X) e0 dy0
  ∧ mnv4PreStridedSyncTiedB R hR N mnv4Row1 xN cotN vN epsStr w.b1 (mnv4Blk0 (R * N) w X) e1 dy1
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row2 xN cotN vN epsStr w.b2 (mnv4Blk1 (R * N) w X) e2 dy2
  ∧ mnv4PreStridedSyncTiedB R hR N mnv4Row3 xN cotN vN epsStr w.b3 (mnv4Blk2 (R * N) w X) e3 dy3
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row4 xN cotN vN epsStr w.b4 (mnv4Blk3 (R * N) w X) e4 dy4
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row5 xN cotN vN epsStr w.b5 (mnv4Blk4 (R * N) w X) e5 dy5
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row6 xN cotN vN epsStr w.b6 (mnv4Blk5 (R * N) w X) e6 dy6
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row7 xN cotN vN epsStr w.b7 (mnv4Blk6 (R * N) w X) e7 dy7
  ∧ mnv4ConvNeXtSyncTiedB R hR N mnv4Row8 xN cotN vN epsStr w.b8 (mnv4Blk7 (R * N) w X) e8 dy8
  ∧ mnv4FfnSyncTiedB R hR N mnv4Row9 xN cotN vN epsStr w.b9 (mnv4Blk8 (R * N) w X) e9 dy9
  ∧ mnv4ConvNeXtSyncTiedB R hR N mnv4Row10 xN cotN vN epsStr w.b10 (mnv4Blk9 (R * N) w X) e10 dy10
  ∧ mnv4PreStridedSyncTiedB R hR N mnv4Row11 xN cotN vN epsStr w.b11 (mnv4Blk10 (R * N) w X)
      e11 dy11
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row12 xN cotN vN epsStr w.b12 (mnv4Blk11 (R * N) w X) e12 dy12
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row13 xN cotN vN epsStr w.b13 (mnv4Blk12 (R * N) w X) e13 dy13
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row14 xN cotN vN epsStr w.b14 (mnv4Blk13 (R * N) w X) e14 dy14
  ∧ mnv4FfnSyncTiedB R hR N mnv4Row15 xN cotN vN epsStr w.b15 (mnv4Blk14 (R * N) w X) e15 dy15
  ∧ mnv4ConvNeXtSyncTiedB R hR N mnv4Row16 xN cotN vN epsStr w.b16 (mnv4Blk15 (R * N) w X) e16 dy16
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row17 xN cotN vN epsStr w.b17 (mnv4Blk16 (R * N) w X) e17 dy17
  ∧ mnv4ExtraDWSyncTiedB R hR N mnv4Row18 xN cotN vN epsStr w.b18 (mnv4Blk17 (R * N) w X) e18 dy18
  ∧ mnv4FfnSyncTiedB R hR N mnv4Row19 xN cotN vN epsStr w.b19 (mnv4Blk18 (R * N) w X) e19 dy19
  ∧ mnv4FfnSyncTiedB R hR N mnv4Row20 xN cotN vN epsStr w.b20 (mnv4Blk19 (R * N) w X) e20 dy20
  ∧ mnv4ConvNeXtSyncTiedB R hR N mnv4Row21 xN cotN vN epsStr w.b21 (mnv4Blk20 (R * N) w X) e21 dy21
  ∧ mnv4HeadSyncTiedB R hR N 7 7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg w.hbt
      w.Wd w.bd xN cotN vN epsStr (mnv4Blk21 (R * N) w X) gs G

/-- ⭐⭐⭐ **The synchronised-BN data-parallel MobileNetV4-Conv-M step IS the single-device step at the
    global batch.** `R` replicas at batch `N`, each running the render's sync-BN backward chain
    from its own loss cotangent `gs r`; when each `gs r` is `R ×` its shard of a global cotangent
    `G` — the replicas' loss divisor is `R ×` smaller than the global step's — every parameter's
    all-reduced mean gradient — stem 3, fused 6, thirteen ExtraDW-profile blocks × 12, four
    ConvNeXt-like × 9, four FFN × 6, head 8: the 233 the render emits — equals the single-device
    batch-BN gradient node at batch `R·N`, at the cotangent T3's chain delivers there from `G`.

    ⭐ The left-hand chain is the replicas' own: sync-BN backward (`bnSyncInB`, a collective per BN
    layer), per-example conv / depthwise / strided / relu / swish / GAP / dense links. The
    right-hand chain is `mnv4_net_tiedB`'s at `N := R·N` with `g := G`, whose nodes that capstone
    ties to the certified gradient — so this and it together say the DP step's update is the
    certified gradient of the global-batch step. `mnv4_net_syncTiedB_smoothedCE` discharges the
    hypothesis for the label-smoothed chain the artifacts emit.

    ⛔ Before the render's sync-BN swap the DP render normalised per replica and this statement was
    false: `DataParallel.dpMeanGrad_ne_globalBatchGrad` is the witness, and stays as the statement
    of what those runs did. -/
theorem mnv4_net_syncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) {nCls : Nat}
    (xN cotN vN epsStr : String) (w : Mnv4BWeights nCls) (X : Vec ((R * N) * (3 * 224 * 224)))
    (G : Vec ((R * N) * nCls)) (gs : Fin R → Vec (N * nCls))
    (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r) :
    mnv4NetSyncTiedB R hR N xN cotN vN epsStr w X G gs := by
  unfold mnv4NetSyncTiedB
  intro dy21 dy20 dy19 dy18 dy17 dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3
    dy2 dy1 dy0 dyStem e21 e20 e19 e18 e17 e16 e15 e14 e13 e12 e11 e10 e9 e8 e7 e6 e5 e4 e3 e2
    e1 e0 eStem
  have h112 : 0 < 112 := by norm_num
  have h56 : 0 < 56 := by norm_num
  have h7 : 0 < 7 := by norm_num
  -- the scaled-shard invariant, block by block down the chain, from `hgs`
  have s21 := mnv4HeadSyncCotIn_scaled R hR N 7 7 hN h7 h7 w.h1W w.h1b w.h1E w.h1g w.h1bt
    w.hW w.hb w.hE w.hg w.hbt w.Wd w.bd (mnv4Blk21 (R * N) w X) gs G hgs
  have s20 := mnv4SkipSyncCotIn_scaled _ e21 _ dy21
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row21 (by decide) w.b21 (mnv4Blk20 (R * N) w X)
      e21 dy21 s21) s21
  have s19 := mnv4SkipSyncCotIn_scaled _ e20 _ dy20
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row20 (by decide) w.b20 (mnv4Blk19 (R * N) w X)
      e20 dy20 s20) s20
  have s18 := mnv4SkipSyncCotIn_scaled _ e19 _ dy19
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row19 (by decide) w.b19 (mnv4Blk18 (R * N) w X)
      e19 dy19 s19) s19
  have s17 := mnv4SkipSyncCotIn_scaled _ e18 _ dy18
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row18 (by decide) w.b18 (mnv4Blk17 (R * N) w X)
      e18 dy18 s18) s18
  have s16 := mnv4SkipSyncCotIn_scaled _ e17 _ dy17
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row17 (by decide) w.b17 (mnv4Blk16 (R * N) w X)
      e17 dy17 s17) s17
  have s15 := mnv4SkipSyncCotIn_scaled _ e16 _ dy16
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row16 (by decide) w.b16 (mnv4Blk15 (R * N) w X)
      e16 dy16 s16) s16
  have s14 := mnv4SkipSyncCotIn_scaled _ e15 _ dy15
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row15 (by decide) w.b15 (mnv4Blk14 (R * N) w X)
      e15 dy15 s15) s15
  have s13 := mnv4SkipSyncCotIn_scaled _ e14 _ dy14
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row14 (by decide) w.b14 (mnv4Blk13 (R * N) w X)
      e14 dy14 s14) s14
  have s12 := mnv4SkipSyncCotIn_scaled _ e13 _ dy13
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row13 (by decide) w.b13 (mnv4Blk12 (R * N) w X)
      e13 dy13 s13) s13
  have s11 := mnv4SkipSyncCotIn_scaled _ e12 _ dy12
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row12 (by decide) w.b12 (mnv4Blk11 (R * N) w X)
      e12 dy12 s12) s12
  have s10 := mnv4SBodySyncCotIn_scaled R hR N hN mnv4Row11 (by decide) w.b11
    (mnv4Blk10 (R * N) w X) e11 dy11 s11
  have s9 := mnv4SkipSyncCotIn_scaled _ e10 _ dy10
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row10 (by decide) w.b10 (mnv4Blk9 (R * N) w X)
      e10 dy10 s10) s10
  have s8 := mnv4SkipSyncCotIn_scaled _ e9 _ dy9
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row9 (by decide) w.b9 (mnv4Blk8 (R * N) w X)
      e9 dy9 s9) s9
  have s7 := mnv4SkipSyncCotIn_scaled _ e8 _ dy8
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row8 (by decide) w.b8 (mnv4Blk7 (R * N) w X)
      e8 dy8 s8) s8
  have s6 := mnv4SkipSyncCotIn_scaled _ e7 _ dy7
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row7 (by decide) w.b7 (mnv4Blk6 (R * N) w X)
      e7 dy7 s7) s7
  have s5 := mnv4SkipSyncCotIn_scaled _ e6 _ dy6
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row6 (by decide) w.b6 (mnv4Blk5 (R * N) w X)
      e6 dy6 s6) s6
  have s4 := mnv4SkipSyncCotIn_scaled _ e5 _ dy5
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row5 (by decide) w.b5 (mnv4Blk4 (R * N) w X)
      e5 dy5 s5) s5
  have s3 := mnv4SkipSyncCotIn_scaled _ e4 _ dy4
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row4 (by decide) w.b4 (mnv4Blk3 (R * N) w X)
      e4 dy4 s4) s4
  have s2 := mnv4SBodySyncCotIn_scaled R hR N hN mnv4Row3 (by decide) w.b3
    (mnv4Blk2 (R * N) w X) e3 dy3 s3
  have s1 := mnv4SkipSyncCotIn_scaled _ e2 _ dy2
    (mnv4BodySyncCotIn_scaled R hR N hN mnv4Row2 (by decide) w.b2 (mnv4Blk1 (R * N) w X)
      e2 dy2 s2) s2
  have s0 := mnv4SBodySyncCotIn_scaled R hR N hN mnv4Row1 (by decide) w.b1
    (mnv4Blk0 (R * N) w X) e1 dy1 s1
  have sStem := mnv4FusedSyncCotIn_scaled R hR N 56 56 hN h56 h56 w.f0cW w.f0cb w.f0cE w.f0cg
    w.f0cbt w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt (mnv4Pre0 (R * N) w X) e0 dy0 s0
  exact ⟨mnv4_stem_syncTiedB R hR N 112 112 hN h112 h112 xN cotN vN epsStr
      w.sW w.sb w.sE w.sg w.sbt X eStem _ sStem,
    mnv4_fused_syncTiedB R hR N 56 56 hN h56 h56 w.f0cW w.f0cb w.f0cE w.f0cg w.f0cbt
      w.f0pW w.f0pb w.f0pE w.f0pg w.f0pbt xN cotN vN epsStr (mnv4Pre0 (R * N) w X) e0 dy0 s0,
    mnv4_prestrided_syncTiedB R hR N hN mnv4Row1 (by decide) xN cotN vN epsStr w.b1
      (mnv4Blk0 (R * N) w X) e1 dy1 s1,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row2 (by decide) xN cotN vN epsStr w.b2
      (mnv4Blk1 (R * N) w X) e2 dy2 s2,
    mnv4_prestrided_syncTiedB R hR N hN mnv4Row3 (by decide) xN cotN vN epsStr w.b3
      (mnv4Blk2 (R * N) w X) e3 dy3 s3,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row4 (by decide) xN cotN vN epsStr w.b4
      (mnv4Blk3 (R * N) w X) e4 dy4 s4,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row5 (by decide) xN cotN vN epsStr w.b5
      (mnv4Blk4 (R * N) w X) e5 dy5 s5,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row6 (by decide) xN cotN vN epsStr w.b6
      (mnv4Blk5 (R * N) w X) e6 dy6 s6,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row7 (by decide) xN cotN vN epsStr w.b7
      (mnv4Blk6 (R * N) w X) e7 dy7 s7,
    mnv4_convnext_syncTiedB R hR N hN mnv4Row8 (by decide) xN cotN vN epsStr w.b8
      (mnv4Blk7 (R * N) w X) e8 dy8 s8,
    mnv4_ffn_syncTiedB R hR N hN mnv4Row9 (by decide) xN cotN vN epsStr w.b9
      (mnv4Blk8 (R * N) w X) e9 dy9 s9,
    mnv4_convnext_syncTiedB R hR N hN mnv4Row10 (by decide) xN cotN vN epsStr w.b10
      (mnv4Blk9 (R * N) w X) e10 dy10 s10,
    mnv4_prestrided_syncTiedB R hR N hN mnv4Row11 (by decide) xN cotN vN epsStr w.b11
      (mnv4Blk10 (R * N) w X) e11 dy11 s11,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row12 (by decide) xN cotN vN epsStr w.b12
      (mnv4Blk11 (R * N) w X) e12 dy12 s12,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row13 (by decide) xN cotN vN epsStr w.b13
      (mnv4Blk12 (R * N) w X) e13 dy13 s13,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row14 (by decide) xN cotN vN epsStr w.b14
      (mnv4Blk13 (R * N) w X) e14 dy14 s14,
    mnv4_ffn_syncTiedB R hR N hN mnv4Row15 (by decide) xN cotN vN epsStr w.b15
      (mnv4Blk14 (R * N) w X) e15 dy15 s15,
    mnv4_convnext_syncTiedB R hR N hN mnv4Row16 (by decide) xN cotN vN epsStr w.b16
      (mnv4Blk15 (R * N) w X) e16 dy16 s16,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row17 (by decide) xN cotN vN epsStr w.b17
      (mnv4Blk16 (R * N) w X) e17 dy17 s17,
    mnv4_extradw_syncTiedB R hR N hN mnv4Row18 (by decide) xN cotN vN epsStr w.b18
      (mnv4Blk17 (R * N) w X) e18 dy18 s18,
    mnv4_ffn_syncTiedB R hR N hN mnv4Row19 (by decide) xN cotN vN epsStr w.b19
      (mnv4Blk18 (R * N) w X) e19 dy19 s19,
    mnv4_ffn_syncTiedB R hR N hN mnv4Row20 (by decide) xN cotN vN epsStr w.b20
      (mnv4Blk19 (R * N) w X) e20 dy20 s20,
    mnv4_convnext_syncTiedB R hR N hN mnv4Row21 (by decide) xN cotN vN epsStr w.b21
      (mnv4Blk20 (R * N) w X) e21 dy21 s21,
    mnv4_head_syncTiedB R hR N 7 7 hN h7 h7 w.h1W w.h1b w.h1E w.h1g w.h1bt w.hW w.hb w.hE w.hg
      w.hbt w.Wd w.bd xN cotN vN epsStr (mnv4Blk21 (R * N) w X) gs G hgs⟩

/-- ⭐⭐ **…and at the loss the artifacts emit.** `mnv4_net_syncTiedB` with its cotangent hypothesis
    discharged by `replicaLossCot_eq`: each replica runs the label-smoothed softmax chain
    (`smoothedLossCotGraph`, the `rowB`/`unrowB` spelling `mnv4_lossCot_is_smoothedCE_grad` reads
    off the render) on its shard of the logits and targets with divisor `B`; the single-device step
    runs it on the whole `R·N` batch with divisor `R·B`. Then every all-reduced gradient the DP
    render emits IS the single-device node at batch `R·N` — the step `mnv4_net_tiedB` at
    `N := R·N`, `g := ` that step's own smoothed-CE cotangent, ties to the certified gradient. -/
theorem mnv4_net_syncTiedB_smoothedCE (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) {nCls : Nat}
    (xN cotN vN epsStr : String) (aStr negAK bStr logN ohN : String) (α B : ℝ)
    (w : Mnv4BWeights nCls) (X : Vec ((R * N) * (3 * 224 * 224)))
    (T : Vec ((R * N) * (1 * nCls))) :
    mnv4NetSyncTiedB R hR N xN cotN vN epsStr w X
      (unrowB (R * N) nCls (den (smoothedLossCotGraph (R * N) nCls α ((R : ℝ) * B) aStr negAK
        bStr logN ohN (rowB (R * N) nCls (mobilenetv4ForwardB_full (R * N) w X)) T)))
      (fun r => unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (batchShard R N nCls (mobilenetv4ForwardB_full (R * N) w X) r))
        (batchShard R N (1 * nCls) T r)))) :=
  mnv4_net_syncTiedB R hR N hN xN cotN vN epsStr w X _ _
    (fun r => replicaLossCot_eq R N nCls hR α B aStr negAK bStr logN ohN _ T r)

end Proofs.MobileNetV4SyncTieB
