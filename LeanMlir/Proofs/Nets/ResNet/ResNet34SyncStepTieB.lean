import LeanMlir.Proofs.Nets.ResNet.ResNet34StepTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncB

/-! # ResNet-34's data-parallel step at SYNCHRONISED BatchNorm IS the single-device step at `R·N`

`ResNet34StepTieB.lean` threads the label-smoothed loss cotangent down the batch-BN
backward chain on ONE device and ties every parameter gradient node to the certified per-op
gradient at the cotangent that chain delivers. This is its data-parallel twin, for the render
`ResNet34RenderB` emits at `replicas > 1`: `R` replicas at batch `N`, every BatchNorm synchronised (`bnFwdSite` / `bnBackSite` /
`bnGammaSite`), every parameter gradient all-reduced by its mean. The capstone
`r34_net_syncTiedB` says that, for every parameter,

    mean over the R replicas of replica r's gradient node, loss divided by B
      = the single-device gradient node at the global batch R·N, loss divided by R·B

— the gradient node `r34_net_tiedB` at `N := R·N` ties to the certified per-op gradient. So the sentence
the DP render header carries ("this step IS the single-device step at the global batch") is a
theorem, and the spec it is stated against has not moved: the right-hand side is the existing
single-device chain at `N := R·N`.

## Four steps

The per-op facts behind each step are in `DataParallelSyncKit`; this file states ResNet-34's block
chains and ties over them.

1. **Sharding** — each replica's backward chain, handed its shard of a global cotangent, computes
   the shard of the global chain. Every non-BN link (relu mask, conv and strided-conv input-VJP,
   the 3×3/s2 pool's scatter, the head) is a per-example map and commutes with sharding by
   definition; the BN link is `bnSyncInB`, whose shard lemma `bnSyncInB_shard` is P2 on the
   graph (`DataParallelSync.den_bnSyncBack_allReduce`) carried across the `mul_assoc` seam.
2. **The collectives** — the mean over replicas of each replica's gradient node is `1/R` of the
   global node at the global cotangent (the P4 lemmas of `DataParallelSync` and
   `DataParallelSyncKit`). The γ node is the sync one, `bnSyncGammaGradB`, reading the forward's
   all-reduced statistics — the one parameter gradient sync-BN changes.
3. **Homogeneity** — the single-device chain and its gradient nodes are linear in the loss
   cotangent (`*_smul`): `R ×` the cotangent gives `R ×` every node.
4. **The divisor** — replica `r` divides its loss by `B` and the global step by `R·B`, so replica
   `r`'s loss cotangent is `R ×` its shard of the global one (`replicaLossCot_eq`). Steps 1–3
   carry that `R` down the chain and it cancels the collective's `1/R`.

## What is NOT claimed

The replicas' saved forward activations enter as the shards of the single-device forward's
(`batchShard r (r34Pre_k (R*N) w X)`); that the sync forward graph computes exactly those is
`ResNet34SyncB.resnet34FwdGraphSyncFull_shard`, the forward half. That the replicas' inputs are
the shards of one batch is the driver's. The emitted artifacts run `convBias := false`, so the
conv-bias nodes are not emitted and are not tied here (`r34_net_tiedB` keeps them for the flag).
The lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ResNet34SyncTieB

open scoped BigOperators
open Proofs.EnetTiePoC (reassocB cInB)
open Proofs.ResNet34TieB

-- ════════════════════════════════════════════════════════════════
-- § 1. Homogeneity — the ResNet-34 block cotangents are linear in their cotangent
--   (the per-op `*_smul` facts are in `DataParallelSyncKit`)
-- ════════════════════════════════════════════════════════════════

theorem r34HeadCotBlk_smul (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) : IsHomog (r34HeadCotBlk N h w Wd bd xin) :=
  HasVJP.backward_smul _ _

/-! The block cotangents, each one line from the previous link's. -/

theorem r34IdCotA_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin : Vec (N * (c * h * w))) :
    IsHomog (r34IdCotA N h w p xin) :=
  reluMaskB_smul _ _

theorem r34IdCotC2_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin : Vec (N * (c * h * w))) :
    IsHomog (r34IdCotC2 N h w p xin) := by
  intro s dy
  unfold r34IdCotC2; rw [r34IdCotA_smul, bnInB_smul]

theorem r34IdCotN1_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin : Vec (N * (c * h * w))) :
    IsHomog (r34IdCotN1 N h w p xin) := by
  intro s dy
  unfold r34IdCotN1; rw [r34IdCotC2_smul, cInB_smul, reluMaskB_smul]

theorem r34IdCotC1_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin : Vec (N * (c * h * w))) :
    IsHomog (r34IdCotC1 N h w p xin) := by
  intro s dy
  unfold r34IdCotC1; rw [r34IdCotN1_smul, bnInB_smul]

theorem r34IdCotIn_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin : Vec (N * (c * h * w))) :
    IsHomog (r34IdCotIn N h w p xin) := by
  intro s dy
  unfold r34IdCotIn
  rw [r34IdCotC1_smul, cInB_smul, r34IdCotA_smul]
  funext i
  beta_reduce
  ring

theorem r34DownCotA_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (r34DownCotA N h w p xin) :=
  reluMaskB_smul _ _

theorem r34DownCotC2_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (r34DownCotC2 N h w p xin) := by
  intro s dy
  unfold r34DownCotC2; rw [r34DownCotA_smul, bnInB_smul]

theorem r34DownCotN1_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (r34DownCotN1 N h w p xin) := by
  intro s dy
  unfold r34DownCotN1; rw [r34DownCotC2_smul, cInB_smul, reluMaskB_smul]

theorem r34DownCotC1_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (r34DownCotC1 N h w p xin) := by
  intro s dy
  unfold r34DownCotC1; rw [r34DownCotN1_smul, bnInB_smul]

theorem r34DownCotCp_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (r34DownCotCp N h w p xin) := by
  intro s dy
  unfold r34DownCotCp; rw [r34DownCotA_smul, bnInB_smul]

theorem r34DownCotIn_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) : IsHomog (r34DownCotIn N h w p xin) := by
  intro s dy
  unfold r34DownCotIn
  rw [r34DownCotC1_smul, cStridedInB_smul, r34DownCotCp_smul, cStridedInB_smul]
  funext i
  beta_reduce
  ring

theorem r34StemCotP_smul (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ)
    (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) :
    IsHomog (r34StemCotP N h w Ws bs εs γs βs x) :=
  mpInB_smul _ _ _ _ _

theorem r34StemCotN_smul (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ)
    (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) :
    IsHomog (r34StemCotN N h w Ws bs εs γs βs x) := by
  intro s dy
  unfold r34StemCotN; rw [r34StemCotP_smul, reluMaskB_smul]

theorem r34StemCotC_smul (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ)
    (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) :
    IsHomog (r34StemCotC N h w Ws bs εs γs βs x) := by
  intro s dy
  unfold r34StemCotC; rw [r34StemCotN_smul, bnInB_smul]

/-! The gradient nodes. -/

-- ════════════════════════════════════════════════════════════════
-- § 2. Sharding — the head cotangent, on a replica, is the shard of the global one
--   (the per-op `*_shard` facts and `bnSyncInB_shard` are in `DataParallelSyncKit`)
-- ════════════════════════════════════════════════════════════════

theorem r34HeadCotBlk_shard {R N : Nat} (h w : Nat) {c nCls : Nat} (Wd : Mat c nCls)
    (bd : Vec nCls) (XIN : Vec ((R * N) * (c * h * w))) (G : Vec ((R * N) * nCls)) (r : Fin R) :
    r34HeadCotBlk N h w Wd bd (batchShard R N (c * h * w) XIN r) (batchShard R N nCls G r)
      = batchShard R N (c * h * w) (r34HeadCotBlk (R * N) h w Wd bd XIN G) r := by
  unfold r34HeadCotBlk
  rw [← r34HeadBBack_eq_vjp_backward Wd bd (batchShard R N (c * h * w) XIN r),
      ← r34HeadBBack_eq_vjp_backward Wd bd XIN]
  simp only [Function.comp_apply]
  rw [batchShard_batchMap, batchShard_batchMap]

-- ════════════════════════════════════════════════════════════════
-- § 3. The replica chain, block by block, and its shard lemmas
--   Saved activations are the shards of the single-device forward's (the forward half,
--   `ResNet34SyncB`, is what says a replica computes exactly those); every cotangent is the
--   replica's own, from the family `dys` of block-output cotangents.
-- ════════════════════════════════════════════════════════════════

section IdBlock
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {c : Nat} (p : R34IdW c)
  (XIN : Vec ((R * N) * (c * h * w))) (dys : Fin R → Vec (N * (c * h * w)))

noncomputable def r34IdSyncCotA (r : Fin R) : Vec (N * (c * h * w)) :=
  reluMaskB (N * (c * h * w))
    (batchShard R N (c * h * w) (residual (projB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
      cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) XIN) r) (dys r)

noncomputable def r34IdSyncCotC2 (r : Fin R) : Vec (N * (c * h * w)) :=
  bnSyncInB R hR N c h w p.ε₂ p.γ₂
    (fun r => batchShard R N (c * h * w) (batchMap (R * N) (flatConv p.W₂ p.b₂)
      (cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN)) r)
    (r34IdSyncCotA R N h w p XIN dys) r

noncomputable def r34IdSyncCotN1 (r : Fin R) : Vec (N * (c * h * w)) :=
  reluMaskB (N * (c * h * w))
    (batchShard R N (c * h * w)
      (bnBatchLA (R * N) c h w p.ε₁ p.γ₁ p.β₁ (batchMap (R * N) (flatConv p.W₁ p.b₁) XIN)) r)
    (cInB N p.W₂ p.b₂ (r34IdSyncCotC2 R hR N h w p XIN dys r))

noncomputable def r34IdSyncCotC1 (r : Fin R) : Vec (N * (c * h * w)) :=
  bnSyncInB R hR N c h w p.ε₁ p.γ₁
    (fun r => batchShard R N (c * h * w) (batchMap (R * N) (flatConv p.W₁ p.b₁) XIN) r)
    (r34IdSyncCotN1 R hR N h w p XIN dys) r

/-- The replica's block-INPUT cotangent: the residual fan-in, body plus identity skip. -/
noncomputable def r34IdSyncCotIn (r : Fin R) : Vec (N * (c * h * w)) :=
  fun i => cInB N p.W₁ p.b₁ (r34IdSyncCotC1 R hR N h w p XIN dys r) i
    + r34IdSyncCotA R N h w p XIN dys r i

end IdBlock

section IdBlockShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {c : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w)
  (p : R34IdW c) (XIN : Vec ((R * N) * (c * h * w))) (dys : Fin R → Vec (N * (c * h * w)))
  (DY : Vec ((R * N) * (c * h * w))) (hdys : ∀ r, dys r = batchShard R N (c * h * w) DY r)
include hdys

theorem r34IdSyncCotA_shard (r : Fin R) :
    r34IdSyncCotA R N h w p XIN dys r = batchShard R N (c * h * w) (r34IdCotA (R * N) h w p XIN DY) r := by
  unfold r34IdSyncCotA; rw [hdys]; rfl

include hN hh hw in
theorem r34IdSyncCotC2_shard (r : Fin R) :
    r34IdSyncCotC2 R hR N h w p XIN dys r
      = batchShard R N (c * h * w) (r34IdCotC2 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N c h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r34IdSyncCotA_shard R N h w p XIN dys DY hdys) r

include hN hh hw in
theorem r34IdSyncCotN1_shard (r : Fin R) :
    r34IdSyncCotN1 R hR N h w p XIN dys r
      = batchShard R N (c * h * w) (r34IdCotN1 (R * N) h w p XIN DY) r := by
  unfold r34IdSyncCotN1
  rw [r34IdSyncCotC2_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

include hN hh hw in
theorem r34IdSyncCotC1_shard (r : Fin R) :
    r34IdSyncCotC1 R hR N h w p XIN dys r
      = batchShard R N (c * h * w) (r34IdCotC1 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N c h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r34IdSyncCotN1_shard R hR N h w hN hh hw p XIN dys DY hdys) r

include hN hh hw in
theorem r34IdSyncCotIn_shard (r : Fin R) :
    r34IdSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (c * h * w) (r34IdCotIn (R * N) h w p XIN DY) r := by
  unfold r34IdSyncCotIn
  rw [r34IdSyncCotC1_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard,
    r34IdSyncCotA_shard R N h w p XIN dys DY hdys]
  rfl

end IdBlockShard

section DownBlock
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
  (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))

noncomputable def r34DownSyncCotA (r : Fin R) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w)) (batchShard R N (oc * h * w) (r34DownPre (R * N) h w p XIN) r) (dys r)

noncomputable def r34DownSyncCotC2 (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.ε₂ p.γ₂
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv p.W₂ p.b₂)
      (cbReluStridedB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN)) r)
    (r34DownSyncCotA R N h w p XIN dys) r

noncomputable def r34DownSyncCotN1 (r : Fin R) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (batchShard R N (oc * h * w)
      (bnBatchLA (R * N) oc h w p.ε₁ p.γ₁ p.β₁ (batchMap (R * N) (flatConvStride2 p.W₁ p.b₁) XIN)) r)
    (cInB N p.W₂ p.b₂ (r34DownSyncCotC2 R hR N h w p XIN dys r))

noncomputable def r34DownSyncCotC1 (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.ε₁ p.γ₁
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConvStride2 p.W₁ p.b₁) XIN) r)
    (r34DownSyncCotN1 R hR N h w p XIN dys) r

noncomputable def r34DownSyncCotCp (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.εp p.γp
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConvStride2 p.Wp p.bp) XIN) r)
    (r34DownSyncCotA R N h w p XIN dys) r

/-- The replica's block-INPUT cotangent: the projected-residual fan-in. -/
noncomputable def r34DownSyncCotIn (r : Fin R) : Vec (N * (ic * (2 * h) * (2 * w))) :=
  fun i => cStridedInB N p.W₁ p.b₁ (r34DownSyncCotC1 R hR N h w p XIN dys r) i
    + cStridedInB N p.Wp p.bp (r34DownSyncCotCp R hR N h w p XIN dys r) i

end DownBlock

section DownBlockShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (p : R34DownW ic oc) (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
  (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
  (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hdys

theorem r34DownSyncCotA_shard (r : Fin R) :
    r34DownSyncCotA R N h w p XIN dys r
      = batchShard R N (oc * h * w) (r34DownCotA (R * N) h w p XIN DY) r := by
  unfold r34DownSyncCotA; rw [hdys]; rfl

include hN hh hw in
theorem r34DownSyncCotC2_shard (r : Fin R) :
    r34DownSyncCotC2 R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r34DownCotC2 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r34DownSyncCotA_shard R N h w p XIN dys DY hdys) r

include hN hh hw in
theorem r34DownSyncCotN1_shard (r : Fin R) :
    r34DownSyncCotN1 R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r34DownCotN1 (R * N) h w p XIN DY) r := by
  unfold r34DownSyncCotN1
  rw [r34DownSyncCotC2_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

include hN hh hw in
theorem r34DownSyncCotC1_shard (r : Fin R) :
    r34DownSyncCotC1 R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r34DownCotC1 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r34DownSyncCotN1_shard R hR N h w hN hh hw p XIN dys DY hdys) r

include hN hh hw in
theorem r34DownSyncCotCp_shard (r : Fin R) :
    r34DownSyncCotCp R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r34DownCotCp (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r34DownSyncCotA_shard R N h w p XIN dys DY hdys) r

include hN hh hw in
theorem r34DownSyncCotIn_shard (r : Fin R) :
    r34DownSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w)) (r34DownCotIn (R * N) h w p XIN DY) r := by
  unfold r34DownSyncCotIn
  rw [r34DownSyncCotC1_shard R hR N h w hN hh hw p XIN dys DY hdys, cStridedInB_shard,
    r34DownSyncCotCp_shard R hR N h w hN hh hw p XIN dys DY hdys, cStridedInB_shard]
  rfl

end DownBlockShard

section Stem
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
  (εs : ℝ) (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
  (dys : Fin R → Vec (N * (oc * h * w)))

noncomputable def r34StemSyncCotP (r : Fin R) : Vec (N * (oc * (2 * h) * (2 * w))) :=
  mpInB N oc h w
    (batchShard R N _ (cbReluStridedB (R * N) (h := 2 * h) (w := 2 * w) Ws bs εs γs βs X) r) (dys r)

noncomputable def r34StemSyncCotN (r : Fin R) : Vec (N * (oc * (2 * h) * (2 * w))) :=
  reluMaskB (N * (oc * (2 * h) * (2 * w)))
    (batchShard R N _
      (bnBatchLA (R * N) oc (2 * h) (2 * w) εs γs βs (batchMap (R * N) (flatConvStride2 Ws bs) X)) r)
    (r34StemSyncCotP R N h w Ws bs εs γs βs X dys r)

noncomputable def r34StemSyncCotC (r : Fin R) : Vec (N * (oc * (2 * h) * (2 * w))) :=
  bnSyncInB R hR N oc (2 * h) (2 * w) εs γs
    (fun r => batchShard R N _ (batchMap (R * N) (flatConvStride2 Ws bs) X) r)
    (r34StemSyncCotN R N h w Ws bs εs γs βs X dys) r

end Stem

section StemShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
  (X : Vec ((R * N) * (ic * (2 * (2 * h)) * (2 * (2 * w))))) (dys : Fin R → Vec (N * (oc * h * w)))
  (DY : Vec ((R * N) * (oc * h * w))) (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hdys

theorem r34StemSyncCotP_shard (r : Fin R) :
    r34StemSyncCotP R N h w Ws bs εs γs βs X dys r
      = batchShard R N _ (r34StemCotP (R * N) h w Ws bs εs γs βs X DY) r := by
  unfold r34StemSyncCotP; rw [hdys, mpInB_shard]; rfl

theorem r34StemSyncCotN_shard (r : Fin R) :
    r34StemSyncCotN R N h w Ws bs εs γs βs X dys r
      = batchShard R N _ (r34StemCotN (R * N) h w Ws bs εs γs βs X DY) r := by
  unfold r34StemSyncCotN; rw [r34StemSyncCotP_shard R N h w Ws bs εs γs βs X dys DY hdys]; rfl

include hN hh hw in
theorem r34StemSyncCotC_shard (r : Fin R) :
    r34StemSyncCotC R hR N h w Ws bs εs γs βs X dys r
      = batchShard R N _ (r34StemCotC (R * N) h w Ws bs εs γs βs X DY) r :=
  bnSyncInB_shard R hR N oc (2 * h) (2 * w)
    (nhw_ne_zero hN (Nat.mul_pos (by norm_num) hh) (Nat.mul_pos (by norm_num) hw))
    _ _ _ _ _ _ (fun _ => rfl) (r34StemSyncCotN_shard R N h w Ws bs εs γs βs X dys DY hdys) r

end StemShard

-- ════════════════════════════════════════════════════════════════
-- § 4. The per-block DP ties
-- ════════════════════════════════════════════════════════════════

/-- **Identity basic block, DP-tied.** Its six emitted parameter collectives — conv₁/conv₂
    weights, bn₁/bn₂ γ and β — each equal the single-device node at the global batch, at the
    single-device chain cotangents driven by `DY`, when the replicas' block-output cotangents are
    `R ×` its shards. -/
def r34IdSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {c : Nat} (pfx xN cotN vN epsStr : String)
    (p : R34IdW c) (XIN : Vec ((R * N) * (c * h * w))) (dys : Fin R → Vec (N * (c * h * w)))
    (DY : Vec ((R * N) * (c * h * w))) : Prop :=
  let r1 := cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN
  let c1 := batchMap (R * N) (flatConv p.W₁ p.b₁) XIN
  let c2 := batchMap (R * N) (flatConv p.W₂ p.b₂) r1
  ConvWSync R hR N h w s!"{pfx}W1" xN cotN p.b₁ XIN p.W₁
      (r34IdSyncCotC1 R hR N h w p XIN dys) (r34IdCotC1 (R * N) h w p XIN DY)
  ∧ BnSync R hR N c h w s!"{pfx}g1" s!"{pfx}bt1" vN epsStr cotN p.ε₁ c1
      (r34IdSyncCotN1 R hR N h w p XIN dys) (r34IdCotN1 (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"{pfx}W2" xN cotN p.b₂ r1 p.W₂
      (r34IdSyncCotC2 R hR N h w p XIN dys) (r34IdCotC2 (R * N) h w p XIN DY)
  ∧ BnSync R hR N c h w s!"{pfx}g2" s!"{pfx}bt2" vN epsStr cotN p.ε₂ c2
      (r34IdSyncCotA R N h w p XIN dys) (r34IdCotA (R * N) h w p XIN DY)

/-- The scaled-shard invariant, carried through one identity block: replicas at `R ×` the shards
    of `DY` produce block-input cotangents at `R ×` the shards of the single-device one. -/
theorem r34IdSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {c : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (p : R34IdW c) (XIN : Vec ((R * N) * (c * h * w)))
    (dys : Fin R → Vec (N * (c * h * w))) (DY : Vec ((R * N) * (c * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (c * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    r34IdSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (c * h * w) (fun i => (R : ℝ) * r34IdCotIn (R * N) h w p XIN DY i) r := by
  rw [r34IdSyncCotIn_shard R hR N h w hN hh hw p XIN dys _ hdys, r34IdCotIn_smul]

theorem r34_idblock_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {c : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String) (p : R34IdW c)
    (XIN : Vec ((R * N) * (c * h * w))) (dys : Fin R → Vec (N * (c * h * w)))
    (DY : Vec ((R * N) * (c * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (c * h * w) (fun i => (R : ℝ) * DY i) r) :
    r34IdSyncTiedB R hR N h w pfx xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hw
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34IdSyncCotC1_shard R hR N h w hN hh hw p XIN dys _ hdys, r34IdCotC1_smul])
  · exact bnSync_of_scaled R hR N c h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r34IdSyncCotN1_shard R hR N h w hN hh hw p XIN dys _ hdys, r34IdCotN1_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34IdSyncCotC2_shard R hR N h w hN hh hw p XIN dys _ hdys, r34IdCotC2_smul])
  · exact bnSync_of_scaled R hR N c h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r34IdSyncCotA_shard R N h w p XIN dys _ hdys, r34IdCotA_smul])

/-- **Downsample basic block, DP-tied** — nine emitted collectives: the strided conv₁, the
    stride-1 conv₂ and the 1×1/s2 projection weights, and the three BatchNorms' γ and β. -/
def r34DownSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat}
    (pfx xN cotN vN epsStr : String) (p : R34DownW ic oc)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  let r1 := cbReluStridedB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN
  let c1 := batchMap (R * N) (flatConvStride2 p.W₁ p.b₁) XIN
  let c2 := batchMap (R * N) (flatConv p.W₂ p.b₂) r1
  let cp := batchMap (R * N) (flatConvStride2 p.Wp p.bp) XIN
  ConvStridedWSync R hR N h w s!"{pfx}W1" xN cotN p.b₁ XIN p.W₁
      (r34DownSyncCotC1 R hR N h w p XIN dys) (r34DownCotC1 (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"{pfx}g1" s!"{pfx}bt1" vN epsStr cotN p.ε₁ c1
      (r34DownSyncCotN1 R hR N h w p XIN dys) (r34DownCotN1 (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"{pfx}W2" xN cotN p.b₂ r1 p.W₂
      (r34DownSyncCotC2 R hR N h w p XIN dys) (r34DownCotC2 (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"{pfx}g2" s!"{pfx}bt2" vN epsStr cotN p.ε₂ c2
      (r34DownSyncCotA R N h w p XIN dys) (r34DownCotA (R * N) h w p XIN DY)
  ∧ ConvStridedWSync R hR N h w s!"{pfx}Wp" xN cotN p.bp XIN p.Wp
      (r34DownSyncCotCp R hR N h w p XIN dys) (r34DownCotCp (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"{pfx}gp" s!"{pfx}btp" vN epsStr cotN p.εp cp
      (r34DownSyncCotA R N h w p XIN dys) (r34DownCotA (R * N) h w p XIN DY)

theorem r34DownSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (p : R34DownW ic oc) (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    r34DownSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w))
          (fun i => (R : ℝ) * r34DownCotIn (R * N) h w p XIN DY i) r := by
  rw [r34DownSyncCotIn_shard R hR N h w hN hh hw p XIN dys _ hdys, r34DownCotIn_smul]

theorem r34_downblock_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String) (p : R34DownW ic oc)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    r34DownSyncTiedB R hR N h w pfx xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hw
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact convStridedWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotC1_shard R hR N h w hN hh hw p XIN dys _ hdys, r34DownCotC1_smul])
  · exact bnSync_of_scaled R hR N oc h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotN1_shard R hR N h w hN hh hw p XIN dys _ hdys, r34DownCotN1_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotC2_shard R hR N h w hN hh hw p XIN dys _ hdys, r34DownCotC2_smul])
  · exact bnSync_of_scaled R hR N oc h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotA_shard R N h w p XIN dys _ hdys, r34DownCotA_smul])
  · exact convStridedWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotCp_shard R hR N h w hN hh hw p XIN dys _ hdys, r34DownCotCp_smul])
  · exact bnSync_of_scaled R hR N oc h w hm _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotA_shard R N h w p XIN dys _ hdys, r34DownCotA_smul])

/-- **Stem, DP-tied** — the 7×7/s2 conv weight and its BatchNorm's γ and β. -/
def r34StemSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (xN cotN vN epsStr : String)
    (Ws : Kernel4 oc ic 7 7) (bs : Vec oc) (εs : ℝ) (γs βs : Vec oc)
    (X : Vec ((R * N) * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  ConvStridedWSync R hR N (2 * h) (2 * w) "sW" xN cotN bs X Ws
      (r34StemSyncCotC R hR N h w Ws bs εs γs βs X dys) (r34StemCotC (R * N) h w Ws bs εs γs βs X DY)
  ∧ BnSync R hR N oc (2 * h) (2 * w) "sg" "sbt" vN epsStr cotN εs
      (batchMap (R * N) (flatConvStride2 Ws bs) X)
      (r34StemSyncCotN R N h w Ws bs εs γs βs X dys) (r34StemCotN (R * N) h w Ws bs εs γs βs X DY)

theorem r34_stem_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (xN cotN vN epsStr : String) (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (X : Vec ((R * N) * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    r34StemSyncTiedB R hR N h w xN cotN vN epsStr Ws bs εs γs βs X dys DY := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  refine ⟨?_, ?_⟩
  · exact convStridedWSync_of_scaled R hR N (2 * h) (2 * w) _ _ _ _ _ _ _ _ (fun r => by
      rw [r34StemSyncCotC_shard R hR N h w hN hh hw Ws bs εs γs βs X dys _ hdys, r34StemCotC_smul])
  · exact bnSync_of_scaled R hR N oc (2 * h) (2 * w) (nhw_ne_zero hN h2h h2w) _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r34StemSyncCotN_shard R N h w Ws bs εs γs βs X dys _ hdys, r34StemCotN_smul])

/-- **Head, DP-tied** — the classifier's weight and bias, at the GAP output. -/
def r34HeadSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {c nCls : Nat} (xN cotN : String)
    (XIN : Vec ((R * N) * (c * h * w))) (gs : Fin R → Vec (N * nCls)) (G : Vec ((R * N) * nCls)) :
    Prop :=
  DenseSync R hR N "Wd" "bd" xN cotN (batchMap (R * N) (globalAvgPoolFlat c h w) XIN) gs G

theorem r34_head_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {c nCls : Nat} (xN cotN : String)
    (XIN : Vec ((R * N) * (c * h * w))) (gs : Fin R → Vec (N * nCls)) (G : Vec ((R * N) * nCls))
    (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r) :
    r34HeadSyncTiedB R hR N h w xN cotN XIN gs G :=
  denseSync_of_scaled R hR N _ _ _ _ _ gs G hgs

/-- The head's block-side cotangent on a replica, at `R ×` the shards of `G`, is `R ×` the shard of
    the single-device one. -/
theorem r34HeadCotBlk_scaled (R : Nat) (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls)
    (bd : Vec nCls) (XIN : Vec ((R * N) * (c * h * w))) (gs : Fin R → Vec (N * nCls))
    (G : Vec ((R * N) * nCls)) (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r)
    (r : Fin R) :
    r34HeadCotBlk N h w Wd bd (batchShard R N (c * h * w) XIN r) (gs r)
      = batchShard R N (c * h * w) (fun i => (R : ℝ) * r34HeadCotBlk (R * N) h w Wd bd XIN G i) r := by
  rw [hgs, r34HeadCotBlk_shard, r34HeadCotBlk_smul]

-- ════════════════════════════════════════════════════════════════
-- § 5. The whole-net capstone
-- ════════════════════════════════════════════════════════════════

/-- **The whole-net statement, named** — so the capstone (cotangents bound) and its smoothed-CE
    corollary (cotangents instantiated) state exactly one thing. The first 16 `let`s are
    `r34_net_tiedB`'s chain at `N := R·N`, driven by the global cotangent `G`; the rest are the
    replicas' sync-BN chain, driven by the family `gs`; the 18 conjuncts are one per stage, every
    emitted parameter collective against `r34_net_tiedB`'s node at the global batch. -/
def r34NetSyncTiedB (R : Nat) (hR : 0 < R) (N : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : R34BWeights nCls) (X : Vec ((R * N) * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (G : Vec ((R * N) * nCls)) (gs : Fin R → Vec (N * nCls)) : Prop :=
  -- ── the single-device chain at the global batch `R·N` (T3's), driven by `G` ──
  let dyE1 := r34HeadCotBlk (R * N) 7 7 w.Wd w.bd (r34Pre16 (R * N) w X) G
  let dyE0 := r34IdCotIn (R * N) 7 7 w.e1 (r34Pre15 (R * N) w X) dyE1
  let dyD4 := r34IdCotIn (R * N) 7 7 w.e0 (r34Pre14 (R * N) w X) dyE0
  let dyC4 := r34DownCotIn (R * N) 7 7 w.d4 (r34Pre13 (R * N) w X) dyD4
  let dyC3 := r34IdCotIn (R * N) 14 14 w.c4 (r34Pre12 (R * N) w X) dyC4
  let dyC2 := r34IdCotIn (R * N) 14 14 w.c3 (r34Pre11 (R * N) w X) dyC3
  let dyC1 := r34IdCotIn (R * N) 14 14 w.c2 (r34Pre10 (R * N) w X) dyC2
  let dyC0 := r34IdCotIn (R * N) 14 14 w.c1 (r34Pre9 (R * N) w X) dyC1
  let dyD3 := r34IdCotIn (R * N) 14 14 w.c0 (r34Pre8 (R * N) w X) dyC0
  let dyB2 := r34DownCotIn (R * N) 14 14 w.d3 (r34Pre7 (R * N) w X) dyD3
  let dyB1 := r34IdCotIn (R * N) 28 28 w.b2 (r34Pre6 (R * N) w X) dyB2
  let dyB0 := r34IdCotIn (R * N) 28 28 w.b1 (r34Pre5 (R * N) w X) dyB1
  let dyD2 := r34IdCotIn (R * N) 28 28 w.b0 (r34Pre4 (R * N) w X) dyB0
  let dyA2 := r34DownCotIn (R * N) 28 28 w.d2 (r34Pre3 (R * N) w X) dyD2
  let dyA1 := r34IdCotIn (R * N) 56 56 w.a2 (r34Pre2 (R * N) w X) dyA2
  let dyA0 := r34IdCotIn (R * N) 56 56 w.a1 (r34Pre1 (R * N) w X) dyA1
  -- ── the replicas' sync-BN chain, driven by the family `gs` ──
  let eE1 : Fin R → Vec (N * (512 * 7 * 7)) := fun r =>
    r34HeadCotBlk N 7 7 w.Wd w.bd (batchShard R N _ (r34Pre16 (R * N) w X) r) (gs r)
  let eE0 := r34IdSyncCotIn R hR N 7 7 w.e1 (r34Pre15 (R * N) w X) eE1
  let eD4 := r34IdSyncCotIn R hR N 7 7 w.e0 (r34Pre14 (R * N) w X) eE0
  let eC4 := r34DownSyncCotIn R hR N 7 7 w.d4 (r34Pre13 (R * N) w X) eD4
  let eC3 := r34IdSyncCotIn R hR N 14 14 w.c4 (r34Pre12 (R * N) w X) eC4
  let eC2 := r34IdSyncCotIn R hR N 14 14 w.c3 (r34Pre11 (R * N) w X) eC3
  let eC1 := r34IdSyncCotIn R hR N 14 14 w.c2 (r34Pre10 (R * N) w X) eC2
  let eC0 := r34IdSyncCotIn R hR N 14 14 w.c1 (r34Pre9 (R * N) w X) eC1
  let eD3 := r34IdSyncCotIn R hR N 14 14 w.c0 (r34Pre8 (R * N) w X) eC0
  let eB2 := r34DownSyncCotIn R hR N 14 14 w.d3 (r34Pre7 (R * N) w X) eD3
  let eB1 := r34IdSyncCotIn R hR N 28 28 w.b2 (r34Pre6 (R * N) w X) eB2
  let eB0 := r34IdSyncCotIn R hR N 28 28 w.b1 (r34Pre5 (R * N) w X) eB1
  let eD2 := r34IdSyncCotIn R hR N 28 28 w.b0 (r34Pre4 (R * N) w X) eB0
  let eA2 := r34DownSyncCotIn R hR N 28 28 w.d2 (r34Pre3 (R * N) w X) eD2
  let eA1 := r34IdSyncCotIn R hR N 56 56 w.a2 (r34Pre2 (R * N) w X) eA2
  let eA0 := r34IdSyncCotIn R hR N 56 56 w.a1 (r34Pre1 (R * N) w X) eA1
  let ePool := r34IdSyncCotIn R hR N 56 56 w.a0 (r34Pre0 (R * N) w X) eA0
  r34StemSyncTiedB R hR N 56 56 xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ X ePool
      (r34IdCotIn (R * N) 56 56 w.a0 (r34Pre0 (R * N) w X) dyA0)
  ∧ r34IdSyncTiedB R hR N 56 56 "s1b0" xN cotN vN epsStr w.a0 (r34Pre0 (R * N) w X) eA0 dyA0
  ∧ r34IdSyncTiedB R hR N 56 56 "s1b1" xN cotN vN epsStr w.a1 (r34Pre1 (R * N) w X) eA1 dyA1
  ∧ r34IdSyncTiedB R hR N 56 56 "s1b2" xN cotN vN epsStr w.a2 (r34Pre2 (R * N) w X) eA2 dyA2
  ∧ r34DownSyncTiedB R hR N 28 28 "d2" xN cotN vN epsStr w.d2 (r34Pre3 (R * N) w X) eD2 dyD2
  ∧ r34IdSyncTiedB R hR N 28 28 "s2b0" xN cotN vN epsStr w.b0 (r34Pre4 (R * N) w X) eB0 dyB0
  ∧ r34IdSyncTiedB R hR N 28 28 "s2b1" xN cotN vN epsStr w.b1 (r34Pre5 (R * N) w X) eB1 dyB1
  ∧ r34IdSyncTiedB R hR N 28 28 "s2b2" xN cotN vN epsStr w.b2 (r34Pre6 (R * N) w X) eB2 dyB2
  ∧ r34DownSyncTiedB R hR N 14 14 "d3" xN cotN vN epsStr w.d3 (r34Pre7 (R * N) w X) eD3 dyD3
  ∧ r34IdSyncTiedB R hR N 14 14 "s3b0" xN cotN vN epsStr w.c0 (r34Pre8 (R * N) w X) eC0 dyC0
  ∧ r34IdSyncTiedB R hR N 14 14 "s3b1" xN cotN vN epsStr w.c1 (r34Pre9 (R * N) w X) eC1 dyC1
  ∧ r34IdSyncTiedB R hR N 14 14 "s3b2" xN cotN vN epsStr w.c2 (r34Pre10 (R * N) w X) eC2 dyC2
  ∧ r34IdSyncTiedB R hR N 14 14 "s3b3" xN cotN vN epsStr w.c3 (r34Pre11 (R * N) w X) eC3 dyC3
  ∧ r34IdSyncTiedB R hR N 14 14 "s3b4" xN cotN vN epsStr w.c4 (r34Pre12 (R * N) w X) eC4 dyC4
  ∧ r34DownSyncTiedB R hR N 7 7 "d4" xN cotN vN epsStr w.d4 (r34Pre13 (R * N) w X) eD4 dyD4
  ∧ r34IdSyncTiedB R hR N 7 7 "s4b0" xN cotN vN epsStr w.e0 (r34Pre14 (R * N) w X) eE0 dyE0
  ∧ r34IdSyncTiedB R hR N 7 7 "s4b1" xN cotN vN epsStr w.e1 (r34Pre15 (R * N) w X) eE1 dyE1
  ∧ r34HeadSyncTiedB R hR N 7 7 xN cotN (r34Pre16 (R * N) w X) gs G

/-- **The synchronised-BN data-parallel ResNet-34 step IS the single-device step at the global
    batch.** `R` replicas at batch `N`, each running the render's sync-BN backward chain from its own
    cotangent `gs r`, with `gs r` the `R`-scaled shard of a global cotangent `G`; every parameter's
    all-reduced mean gradient — stem 3, thirteen identity blocks × 6, three downsample blocks × 9,
    dense 2: the 110 the render emits — equals the single-device batch-BN gradient node at batch
    `R·N`, at the cotangent `r34_net_tiedB`'s chain delivers there from `G`.

    The left-hand chain is the replicas' own: sync-BN backward (`bnSyncInB`, a collective per BN
    layer), per-example conv / relu / pool / head links. The right-hand chain is `r34_net_tiedB`'s
    at `N := R·N`, whose nodes that capstone ties to the certified per-op gradient at the chain
    cotangent — so this and it together say each all-reduced gradient equals the single-device
    node at the global batch.
    `r34_net_syncTiedB_smoothedCE` discharges the hypothesis for the label-smoothed chain the
    artifacts emit.

    Note: with per-replica BatchNorm the mean of the replica gradients is not the global-batch
    gradient in general; `DataParallel.dpMeanGrad_ne_globalBatchGrad` is a two-replica,
    one-parameter counterexample. -/
theorem r34_net_syncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) {nCls : Nat}
    (xN cotN vN epsStr : String) (w : R34BWeights nCls)
    (X : Vec ((R * N) * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (G : Vec ((R * N) * nCls))
    (gs : Fin R → Vec (N * nCls))
    (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r) :
    r34NetSyncTiedB R hR N xN cotN vN epsStr w X G gs := by
  unfold r34NetSyncTiedB
  intro dyE1 dyE0 dyD4 dyC4 dyC3 dyC2 dyC1 dyC0 dyD3 dyB2 dyB1 dyB0 dyD2 dyA2 dyA1 dyA0
    eE1 eE0 eD4 eC4 eC3 eC2 eC1 eC0 eD3 eB2 eB1 eB0 eD2 eA2 eA1 eA0 ePool
  have h56 : 0 < 56 := by norm_num
  have h28 : 0 < 28 := by norm_num
  have h14 : 0 < 14 := by norm_num
  have h7 : 0 < 7 := by norm_num
  -- the scaled-shard invariant, block by block down the chain
  have sE1 : ∀ r, eE1 r = batchShard R N _ (fun i => (R : ℝ) * dyE1 i) r :=
    fun r => r34HeadCotBlk_scaled R N 7 7 w.Wd w.bd (r34Pre16 (R * N) w X) gs G hgs r
  have sE0 := r34IdSyncCotIn_scaled R hR N 7 7 hN h7 h7 w.e1 (r34Pre15 (R * N) w X) eE1 dyE1 sE1
  have sD4 := r34IdSyncCotIn_scaled R hR N 7 7 hN h7 h7 w.e0 (r34Pre14 (R * N) w X) eE0 dyE0 sE0
  have sC4 := r34DownSyncCotIn_scaled R hR N 7 7 hN h7 h7 w.d4 (r34Pre13 (R * N) w X) eD4 dyD4 sD4
  have sC3 := r34IdSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.c4 (r34Pre12 (R * N) w X) eC4 dyC4 sC4
  have sC2 := r34IdSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.c3 (r34Pre11 (R * N) w X) eC3 dyC3 sC3
  have sC1 := r34IdSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.c2 (r34Pre10 (R * N) w X) eC2 dyC2 sC2
  have sC0 := r34IdSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.c1 (r34Pre9 (R * N) w X) eC1 dyC1 sC1
  have sD3 := r34IdSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.c0 (r34Pre8 (R * N) w X) eC0 dyC0 sC0
  have sB2 := r34DownSyncCotIn_scaled R hR N 14 14 hN h14 h14 w.d3 (r34Pre7 (R * N) w X) eD3 dyD3 sD3
  have sB1 := r34IdSyncCotIn_scaled R hR N 28 28 hN h28 h28 w.b2 (r34Pre6 (R * N) w X) eB2 dyB2 sB2
  have sB0 := r34IdSyncCotIn_scaled R hR N 28 28 hN h28 h28 w.b1 (r34Pre5 (R * N) w X) eB1 dyB1 sB1
  have sD2 := r34IdSyncCotIn_scaled R hR N 28 28 hN h28 h28 w.b0 (r34Pre4 (R * N) w X) eB0 dyB0 sB0
  have sA2 := r34DownSyncCotIn_scaled R hR N 28 28 hN h28 h28 w.d2 (r34Pre3 (R * N) w X) eD2 dyD2 sD2
  have sA1 := r34IdSyncCotIn_scaled R hR N 56 56 hN h56 h56 w.a2 (r34Pre2 (R * N) w X) eA2 dyA2 sA2
  have sA0 := r34IdSyncCotIn_scaled R hR N 56 56 hN h56 h56 w.a1 (r34Pre1 (R * N) w X) eA1 dyA1 sA1
  have sPool := r34IdSyncCotIn_scaled R hR N 56 56 hN h56 h56 w.a0 (r34Pre0 (R * N) w X) eA0 dyA0 sA0
  exact ⟨r34_stem_syncTiedB R hR N 56 56 hN h56 h56 xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ X
      ePool _ sPool,
    r34_idblock_syncTiedB R hR N 56 56 hN h56 h56 "s1b0" xN cotN vN epsStr w.a0 (r34Pre0 (R * N) w X) eA0 dyA0 sA0,
    r34_idblock_syncTiedB R hR N 56 56 hN h56 h56 "s1b1" xN cotN vN epsStr w.a1 (r34Pre1 (R * N) w X) eA1 dyA1 sA1,
    r34_idblock_syncTiedB R hR N 56 56 hN h56 h56 "s1b2" xN cotN vN epsStr w.a2 (r34Pre2 (R * N) w X) eA2 dyA2 sA2,
    r34_downblock_syncTiedB R hR N 28 28 hN h28 h28 "d2" xN cotN vN epsStr w.d2 (r34Pre3 (R * N) w X) eD2 dyD2 sD2,
    r34_idblock_syncTiedB R hR N 28 28 hN h28 h28 "s2b0" xN cotN vN epsStr w.b0 (r34Pre4 (R * N) w X) eB0 dyB0 sB0,
    r34_idblock_syncTiedB R hR N 28 28 hN h28 h28 "s2b1" xN cotN vN epsStr w.b1 (r34Pre5 (R * N) w X) eB1 dyB1 sB1,
    r34_idblock_syncTiedB R hR N 28 28 hN h28 h28 "s2b2" xN cotN vN epsStr w.b2 (r34Pre6 (R * N) w X) eB2 dyB2 sB2,
    r34_downblock_syncTiedB R hR N 14 14 hN h14 h14 "d3" xN cotN vN epsStr w.d3 (r34Pre7 (R * N) w X) eD3 dyD3 sD3,
    r34_idblock_syncTiedB R hR N 14 14 hN h14 h14 "s3b0" xN cotN vN epsStr w.c0 (r34Pre8 (R * N) w X) eC0 dyC0 sC0,
    r34_idblock_syncTiedB R hR N 14 14 hN h14 h14 "s3b1" xN cotN vN epsStr w.c1 (r34Pre9 (R * N) w X) eC1 dyC1 sC1,
    r34_idblock_syncTiedB R hR N 14 14 hN h14 h14 "s3b2" xN cotN vN epsStr w.c2 (r34Pre10 (R * N) w X) eC2 dyC2 sC2,
    r34_idblock_syncTiedB R hR N 14 14 hN h14 h14 "s3b3" xN cotN vN epsStr w.c3 (r34Pre11 (R * N) w X) eC3 dyC3 sC3,
    r34_idblock_syncTiedB R hR N 14 14 hN h14 h14 "s3b4" xN cotN vN epsStr w.c4 (r34Pre12 (R * N) w X) eC4 dyC4 sC4,
    r34_downblock_syncTiedB R hR N 7 7 hN h7 h7 "d4" xN cotN vN epsStr w.d4 (r34Pre13 (R * N) w X) eD4 dyD4 sD4,
    r34_idblock_syncTiedB R hR N 7 7 hN h7 h7 "s4b0" xN cotN vN epsStr w.e0 (r34Pre14 (R * N) w X) eE0 dyE0 sE0,
    r34_idblock_syncTiedB R hR N 7 7 hN h7 h7 "s4b1" xN cotN vN epsStr w.e1 (r34Pre15 (R * N) w X) eE1 dyE1 sE1,
    r34_head_syncTiedB R hR N 7 7 xN cotN (r34Pre16 (R * N) w X) gs G hgs⟩

/-- **…and at the loss the artifacts emit.** `r34_net_syncTiedB` with its cotangent hypothesis
    discharged by `replicaLossCot_eq`: each replica runs the label-smoothed softmax chain
    (`smoothedLossCotGraph`) on its shard of the logits and targets with divisor `B`; the
    single-device step runs it on the whole `R·N` batch with divisor `R·B`. Then every all-reduced
    gradient the DP render emits IS the single-device node at batch `R·N`, loss divided by `R·B`. -/
theorem r34_net_syncTiedB_smoothedCE (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) {nCls : Nat}
    (xN cotN vN epsStr : String) (aStr negAK bStr logN ohN : String) (α B : ℝ)
    (w : R34BWeights nCls) (X : Vec ((R * N) * (3 * (2 * (2 * 56)) * (2 * (2 * 56)))))
    (T : Vec ((R * N) * (1 * nCls))) :
    r34NetSyncTiedB R hR N xN cotN vN epsStr w X
      (unrowB (R * N) nCls (den (smoothedLossCotGraph (R * N) nCls α ((R : ℝ) * B) aStr negAK
        bStr logN ohN (rowB (R * N) nCls (resnet34ForwardBFull (R * N) w X)) T)))
      (fun r => unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (batchShard R N nCls (resnet34ForwardBFull (R * N) w X) r))
        (batchShard R N (1 * nCls) T r)))) :=
  r34_net_syncTiedB R hR N hN xN cotN vN epsStr w X _ _
    (fun r => replicaLossCot_eq R N nCls hR α B aStr negAK bStr logN ohN _ T r)

end Proofs.ResNet34SyncTieB
