import LeanMlir.Proofs.Nets.ResNet.ResNet50StepTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncStepTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet50SyncB

/-! # ResNet-50's data-parallel step at SYNCHRONISED BatchNorm IS the single-device step at `R·N`

`ResNet50StepTieB` (T3) threads a loss cotangent `g` down the batch-BN bottleneck backward chain on
ONE device and ties every parameter gradient node to the certified gradient. This is its
data-parallel twin, for the render `ResNet50RenderB` emits at `replicas > 1`: `R` replicas at batch
`N`, every BatchNorm synchronised (`bnFwdSite` / `bnBackSite` / `bnGammaSite`), every parameter
gradient all-reduced by its mean. The capstone `r50_net_syncTiedB` says that, for every one of the
161 parameters,

    mean over the R replicas of replica r's gradient node, replica cotangent gs r
      = the single-device gradient node at the global batch R·N, loss cotangent G

whenever each replica's loss cotangent is `R ×` its shard of the global one
(`∀ r, gs r = batchShard r (R • G)`) — the gradient node `r50_net_tiedB` at `N := R·N` ties to the
certified gradient. The right-hand side is the existing single-device chain at `N := R·N`, so the
spec has not moved.

⭐⭐ **The loss cotangent is a binder, as it is in T3.** ResNet-50 ships both losses, so the
capstone takes the scaled-shard relation between the replicas' cotangents and the global one as
its hypothesis, and two corollaries discharge it, one per loss:

* `r50_net_syncTiedB_smoothedCE` — the label-smoothed chain, replicas dividing by `B` and the
  global step by `R·B` (`ResNet34SyncTieB.replicaLossCot_eq`);
* `r50_net_syncTiedB_bce` — BCE-with-logits at the COMMITTED divisors, `N·K` on a replica and
  `(R·N)·K` on the global step (`replicaBceLossCot_eq`, the BCE peer, proved here).

## Four steps (ResNet-34's, with the bottleneck's links)

1. **Sharding** — each replica's backward chain, handed its shard of a global cotangent, computes
   the shard of the global chain. Every non-BN link (relu mask, the 1×1 and 3×3 conv input-VJPs,
   the strided 3×3 and strided 1×1 input-VJPs, the stem pool's scatter, the head) is a per-example
   map; the BN link is `bnSyncInB`, whose shard lemma `bnSyncInB_shard` is P2 on the graph. ⚠ In
   the strided block bn₁ runs at the INPUT grid `2h × 2w` (v1.5: the stride is on the 3×3), so that
   site's statistics reduce over `N·(2h)·(2w)` per replica.
2. **The collectives** — the mean over replicas of each replica's gradient node is `1/R` of the
   global node at the global cotangent (`DataParallelSync`'s P4 lemmas and ResNet-34's strided and
   dense ones). The γ node is the sync one, `bnSyncGammaGradB`, reading the forward's all-reduced
   statistics.
3. **Homogeneity** — the single-device chain is linear in its cotangent (`*_smul` below for the
   three bottleneck forms, ResNet-34's for the links they are built from).
4. **The divisor** — the `R` the hypothesis carries cancels the collective's `1/R`.

## The index seam

The chain runs at `N·(c·h·w)`, the BN nodes at `N·(c·(h·w))`; the replica BN link `bnSyncInB` is the
`den` of the emitted nodes over `.operand` leaves at `reassocB` (ResNet-34's), and
`reassocB_shard` commutes the relabelling with sharding.

## What is NOT claimed

⚠ The replicas' saved forward activations enter as the shards of the single-device forward's
(`batchShard r (r50Pre_k (R*N) q w X)`); that the sync forward graph computes exactly those is
`StableHLO.resnet50FwdGraphSync_full_shard`, the forward half. ⚠ No stochastic depth (drop-path):
the chain is the drop-free one, as T3's is. ⚠ The f32 nodes — the bf16 conv twins are not this
statement. ⚠ The render has no conv-bias gradient ops, so there are none here. ⚠ That the replicas'
inputs are the shards of one batch is the driver's. ⚠ The gradient accumulator and the optimizers
run after the collective, so this is per micro-step. ⚠ The lowerer's `all_reduce` is trusted as
every other op's lowering is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ResNet50SyncTieB

open scoped BigOperators
open Proofs.EnetTiePoC (cInB)
open Proofs.ResNet34TieB Proofs.ResNet34SyncTieB Proofs.ResNet50TieB

-- ════════════════════════════════════════════════════════════════
-- § 1. Homogeneity — the single-device bottleneck chains are linear in their cotangent
-- ════════════════════════════════════════════════════════════════

section IdSmul
variable (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc) (xin dy : Vec (N * (oc * h * w)))
  (s : ℝ)

theorem r50IdCotA_smul : IsHomog (r50IdCotA N h w p xin) :=
  reluMaskB_smul _ _

theorem r50IdCotC3_smul : IsHomog (r50IdCotC3 N h w p xin) := by
  intro s dy
  unfold r50IdCotC3; rw [r50IdCotA_smul, bnInB_smul]

theorem r50IdCotN2_smul : IsHomog (r50IdCotN2 N h w p xin) := by
  intro s dy
  unfold r50IdCotN2; rw [r50IdCotC3_smul, cInB_smul, reluMaskB_smul]

theorem r50IdCotC2_smul : IsHomog (r50IdCotC2 N h w p xin) := by
  intro s dy
  unfold r50IdCotC2; rw [r50IdCotN2_smul, bnInB_smul]

theorem r50IdCotN1_smul : IsHomog (r50IdCotN1 N h w p xin) := by
  intro s dy
  unfold r50IdCotN1; rw [r50IdCotC2_smul, cInB_smul, reluMaskB_smul]

theorem r50IdCotC1_smul : IsHomog (r50IdCotC1 N h w p xin) := by
  intro s dy
  unfold r50IdCotC1; rw [r50IdCotN1_smul, bnInB_smul]

theorem r50IdCotIn_smul : IsHomog (r50IdCotIn N h w p xin) := by
  intro s dy
  unfold r50IdCotIn
  rw [r50IdCotC1_smul, cInB_smul, r50IdCotA_smul]
  funext i
  beta_reduce
  ring

end IdSmul

section ProjSmul
variable (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc) (xin : Vec (N * (ic * h * w)))
  (dy : Vec (N * (oc * h * w))) (s : ℝ)

theorem r50ProjCotA_smul : IsHomog (r50ProjCotA N h w p xin) :=
  reluMaskB_smul _ _

theorem r50ProjCotC3_smul : IsHomog (r50ProjCotC3 N h w p xin) := by
  intro s dy
  unfold r50ProjCotC3; rw [r50ProjCotA_smul, bnInB_smul]

theorem r50ProjCotN2_smul : IsHomog (r50ProjCotN2 N h w p xin) := by
  intro s dy
  unfold r50ProjCotN2; rw [r50ProjCotC3_smul, cInB_smul, reluMaskB_smul]

theorem r50ProjCotC2_smul : IsHomog (r50ProjCotC2 N h w p xin) := by
  intro s dy
  unfold r50ProjCotC2; rw [r50ProjCotN2_smul, bnInB_smul]

theorem r50ProjCotN1_smul : IsHomog (r50ProjCotN1 N h w p xin) := by
  intro s dy
  unfold r50ProjCotN1; rw [r50ProjCotC2_smul, cInB_smul, reluMaskB_smul]

theorem r50ProjCotC1_smul : IsHomog (r50ProjCotC1 N h w p xin) := by
  intro s dy
  unfold r50ProjCotC1; rw [r50ProjCotN1_smul, bnInB_smul]

theorem r50ProjCotCp_smul : IsHomog (r50ProjCotCp N h w p xin) := by
  intro s dy
  unfold r50ProjCotCp; rw [r50ProjCotA_smul, bnInB_smul]

theorem r50ProjCotIn_smul : IsHomog (r50ProjCotIn N h w p xin) := by
  intro s dy
  unfold r50ProjCotIn
  rw [r50ProjCotC1_smul, cInB_smul, r50ProjCotCp_smul, cInB_smul]
  funext i
  beta_reduce
  ring

end ProjSmul

section DownSmul
variable (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
  (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) (s : ℝ)

theorem r50DownCotA_smul : IsHomog (r50DownCotA N h w p xin) :=
  reluMaskB_smul _ _

theorem r50DownCotC3_smul : IsHomog (r50DownCotC3 N h w p xin) := by
  intro s dy
  unfold r50DownCotC3; rw [r50DownCotA_smul, bnInB_smul]

theorem r50DownCotN2_smul : IsHomog (r50DownCotN2 N h w p xin) := by
  intro s dy
  unfold r50DownCotN2; rw [r50DownCotC3_smul, cInB_smul, reluMaskB_smul]

theorem r50DownCotC2_smul : IsHomog (r50DownCotC2 N h w p xin) := by
  intro s dy
  unfold r50DownCotC2; rw [r50DownCotN2_smul, bnInB_smul]

/-- ⚠ The strided 3×3's input-VJP carries the cotangent from `h × w` up to bn₁'s `2h × 2w`. -/
theorem r50DownCotN1_smul : IsHomog (r50DownCotN1 N h w p xin) := by
  intro s dy
  unfold r50DownCotN1; rw [r50DownCotC2_smul, cStridedInB_smul, reluMaskB_smul]

theorem r50DownCotC1_smul : IsHomog (r50DownCotC1 N h w p xin) := by
  intro s dy
  unfold r50DownCotC1; rw [r50DownCotN1_smul, bnInB_smul]

theorem r50DownCotCp_smul : IsHomog (r50DownCotCp N h w p xin) := by
  intro s dy
  unfold r50DownCotCp; rw [r50DownCotA_smul, bnInB_smul]

theorem r50DownCotIn_smul : IsHomog (r50DownCotIn N h w p xin) := by
  intro s dy
  unfold r50DownCotIn
  rw [r50DownCotC1_smul, cInB_smul, r50DownCotCp_smul, cStridedInB_smul]
  funext i
  beta_reduce
  ring

end DownSmul

-- ════════════════════════════════════════════════════════════════
-- § 2. The replica chains, block by block, and their shard lemmas
--   Saved activations are the shards of the single-device forward's (the forward half,
--   `ResNet50SyncB`, is what says a replica computes exactly those); every cotangent is the
--   replica's own, from the family `dys` of block-output cotangents.
-- ════════════════════════════════════════════════════════════════

section IdBlock
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc : Nat} (p : R50IdW mid oc)
  (XIN : Vec ((R * N) * (oc * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))

/-- Replica `r`'s cotangent at the pre-relu sum. Feeds bn₃'s γ/β and the identity skip. -/
noncomputable def r50IdSyncCotA (r : Fin R) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (batchShard R N (oc * h * w)
      (residual (projB (R * N) (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
        cbReluB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
        cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) XIN) r)
    (dys r)

/-- …at conv₃'s output, through bn₃'s sync backward. Feeds `W₃`. -/
noncomputable def r50IdSyncCotC3 (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.ε₃ p.γ₃
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv p.W₃ p.b₃)
      (cbReluB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN))) r)
    (r50IdSyncCotA R N h w p XIN dys) r

/-- …at bn₂'s output. Feeds `γ₂`/`β₂`. -/
noncomputable def r50IdSyncCotN2 (r : Fin R) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (batchShard R N (mid * h * w)
      (bnBatchLA (R * N) mid h w p.ε₂ p.γ₂ p.β₂ (batchMap (R * N) (flatConv p.W₂ p.b₂)
        (cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN))) r)
    (cInB N p.W₃ p.b₃ (r50IdSyncCotC3 R hR N h w p XIN dys r))

/-- …at conv₂'s output. Feeds `W₂`. -/
noncomputable def r50IdSyncCotC2 (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w p.ε₂ p.γ₂
    (fun r => batchShard R N (mid * h * w) (batchMap (R * N) (flatConv p.W₂ p.b₂)
      (cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN)) r)
    (r50IdSyncCotN2 R hR N h w p XIN dys) r

/-- …at bn₁'s output. Feeds `γ₁`/`β₁`. -/
noncomputable def r50IdSyncCotN1 (r : Fin R) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (batchShard R N (mid * h * w)
      (bnBatchLA (R * N) mid h w p.ε₁ p.γ₁ p.β₁ (batchMap (R * N) (flatConv p.W₁ p.b₁) XIN)) r)
    (cInB N p.W₂ p.b₂ (r50IdSyncCotC2 R hR N h w p XIN dys r))

/-- …at conv₁'s output. Feeds `W₁`. -/
noncomputable def r50IdSyncCotC1 (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w p.ε₁ p.γ₁
    (fun r => batchShard R N (mid * h * w) (batchMap (R * N) (flatConv p.W₁ p.b₁) XIN) r)
    (r50IdSyncCotN1 R hR N h w p XIN dys) r

/-- The replica's block-INPUT cotangent: the residual fan-in, body plus identity skip. -/
noncomputable def r50IdSyncCotIn (r : Fin R) : Vec (N * (oc * h * w)) :=
  fun i => cInB N p.W₁ p.b₁ (r50IdSyncCotC1 R hR N h w p XIN dys r) i
    + r50IdSyncCotA R N h w p XIN dys r i

end IdBlock

section IdBlockShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (p : R50IdW mid oc) (XIN : Vec ((R * N) * (oc * h * w)))
  (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
  (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hdys

theorem r50IdSyncCotA_shard (r : Fin R) :
    r50IdSyncCotA R N h w p XIN dys r
      = batchShard R N (oc * h * w) (r50IdCotA (R * N) h w p XIN DY) r := by
  unfold r50IdSyncCotA; rw [hdys]; rfl

include hN hh hw in
theorem r50IdSyncCotC3_shard (r : Fin R) :
    r50IdSyncCotC3 R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r50IdCotC3 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50IdSyncCotA_shard R N h w p XIN dys DY hdys) r

include hN hh hw in
theorem r50IdSyncCotN2_shard (r : Fin R) :
    r50IdSyncCotN2 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50IdCotN2 (R * N) h w p XIN DY) r := by
  unfold r50IdSyncCotN2
  rw [r50IdSyncCotC3_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

include hN hh hw in
theorem r50IdSyncCotC2_shard (r : Fin R) :
    r50IdSyncCotC2 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50IdCotC2 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50IdSyncCotN2_shard R hR N h w hN hh hw p XIN dys DY hdys) r

include hN hh hw in
theorem r50IdSyncCotN1_shard (r : Fin R) :
    r50IdSyncCotN1 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50IdCotN1 (R * N) h w p XIN DY) r := by
  unfold r50IdSyncCotN1
  rw [r50IdSyncCotC2_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

include hN hh hw in
theorem r50IdSyncCotC1_shard (r : Fin R) :
    r50IdSyncCotC1 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50IdCotC1 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50IdSyncCotN1_shard R hR N h w hN hh hw p XIN dys DY hdys) r

include hN hh hw in
theorem r50IdSyncCotIn_shard (r : Fin R) :
    r50IdSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r50IdCotIn (R * N) h w p XIN DY) r := by
  unfold r50IdSyncCotIn
  rw [r50IdSyncCotC1_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard,
    r50IdSyncCotA_shard R N h w p XIN dys DY hdys]
  rfl

end IdBlockShard

section ProjBlock
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
  (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))

/-- Replica `r`'s cotangent at the stride-1 projection block's pre-relu sum. Feeds bn₃'s and the
    projection's γ/β, and the projection's backward branch. -/
noncomputable def r50ProjSyncCotA (r : Fin R) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (batchShard R N (oc * h * w)
      (residualProj (projB (R * N) (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
        (projB (R * N) (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
          cbReluB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
          cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) XIN) r)
    (dys r)

noncomputable def r50ProjSyncCotC3 (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.ε₃ p.γ₃
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv p.W₃ p.b₃)
      (cbReluB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN))) r)
    (r50ProjSyncCotA R N h w p XIN dys) r

noncomputable def r50ProjSyncCotN2 (r : Fin R) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (batchShard R N (mid * h * w)
      (bnBatchLA (R * N) mid h w p.ε₂ p.γ₂ p.β₂ (batchMap (R * N) (flatConv p.W₂ p.b₂)
        (cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN))) r)
    (cInB N p.W₃ p.b₃ (r50ProjSyncCotC3 R hR N h w p XIN dys r))

noncomputable def r50ProjSyncCotC2 (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w p.ε₂ p.γ₂
    (fun r => batchShard R N (mid * h * w) (batchMap (R * N) (flatConv p.W₂ p.b₂)
      (cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN)) r)
    (r50ProjSyncCotN2 R hR N h w p XIN dys) r

noncomputable def r50ProjSyncCotN1 (r : Fin R) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (batchShard R N (mid * h * w)
      (bnBatchLA (R * N) mid h w p.ε₁ p.γ₁ p.β₁ (batchMap (R * N) (flatConv p.W₁ p.b₁) XIN)) r)
    (cInB N p.W₂ p.b₂ (r50ProjSyncCotC2 R hR N h w p XIN dys r))

noncomputable def r50ProjSyncCotC1 (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w p.ε₁ p.γ₁
    (fun r => batchShard R N (mid * h * w) (batchMap (R * N) (flatConv p.W₁ p.b₁) XIN) r)
    (r50ProjSyncCotN1 R hR N h w p XIN dys) r

/-- …at the stride-1 1×1 projection conv's output, through the skip BN's sync backward. Feeds
    `Wp`. -/
noncomputable def r50ProjSyncCotCp (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.εp p.γp
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv p.Wp p.bp) XIN) r)
    (r50ProjSyncCotA R N h w p XIN dys) r

/-- The replica's block-INPUT cotangent: both branches' input-VJPs, as the render adds them. -/
noncomputable def r50ProjSyncCotIn (r : Fin R) : Vec (N * (ic * h * w)) :=
  fun i => cInB N p.W₁ p.b₁ (r50ProjSyncCotC1 R hR N h w p XIN dys r) i
    + cInB N p.Wp p.bp (r50ProjSyncCotCp R hR N h w p XIN dys r) i

end ProjBlock

section ProjBlockShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (p : R50ProjW ic mid oc) (XIN : Vec ((R * N) * (ic * h * w)))
  (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
  (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hdys

theorem r50ProjSyncCotA_shard (r : Fin R) :
    r50ProjSyncCotA R N h w p XIN dys r
      = batchShard R N (oc * h * w) (r50ProjCotA (R * N) h w p XIN DY) r := by
  unfold r50ProjSyncCotA; rw [hdys]; rfl

include hN hh hw in
theorem r50ProjSyncCotC3_shard (r : Fin R) :
    r50ProjSyncCotC3 R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r50ProjCotC3 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50ProjSyncCotA_shard R N h w p XIN dys DY hdys) r

include hN hh hw in
theorem r50ProjSyncCotN2_shard (r : Fin R) :
    r50ProjSyncCotN2 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50ProjCotN2 (R * N) h w p XIN DY) r := by
  unfold r50ProjSyncCotN2
  rw [r50ProjSyncCotC3_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

include hN hh hw in
theorem r50ProjSyncCotC2_shard (r : Fin R) :
    r50ProjSyncCotC2 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50ProjCotC2 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50ProjSyncCotN2_shard R hR N h w hN hh hw p XIN dys DY hdys) r

include hN hh hw in
theorem r50ProjSyncCotN1_shard (r : Fin R) :
    r50ProjSyncCotN1 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50ProjCotN1 (R * N) h w p XIN DY) r := by
  unfold r50ProjSyncCotN1
  rw [r50ProjSyncCotC2_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

include hN hh hw in
theorem r50ProjSyncCotC1_shard (r : Fin R) :
    r50ProjSyncCotC1 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50ProjCotC1 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50ProjSyncCotN1_shard R hR N h w hN hh hw p XIN dys DY hdys) r

include hN hh hw in
theorem r50ProjSyncCotCp_shard (r : Fin R) :
    r50ProjSyncCotCp R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r50ProjCotCp (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50ProjSyncCotA_shard R N h w p XIN dys DY hdys) r

include hN hh hw in
theorem r50ProjSyncCotIn_shard (r : Fin R) :
    r50ProjSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (r50ProjCotIn (R * N) h w p XIN DY) r := by
  unfold r50ProjSyncCotIn
  rw [r50ProjSyncCotC1_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard,
    r50ProjSyncCotCp_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

end ProjBlockShard

section DownBlock
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat} (p : R50ProjW ic mid oc)
  (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))

/-- Replica `r`'s cotangent at the strided block's pre-relu sum. -/
noncomputable def r50DownSyncCotA (r : Fin R) : Vec (N * (oc * h * w)) :=
  reluMaskB (N * (oc * h * w))
    (batchShard R N (oc * h * w)
      (residualProj (projStridedB (R * N) (h := h) (w := w) p.Wp p.bp p.εp p.γp p.βp)
        (projB (R * N) (h := h) (w := w) p.W₃ p.b₃ p.ε₃ p.γ₃ p.β₃ ∘
          cbReluStridedB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ ∘
          cbReluB (R * N) (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁) XIN) r)
    (dys r)

noncomputable def r50DownSyncCotC3 (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.ε₃ p.γ₃
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConv p.W₃ p.b₃)
      (cbReluStridedB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂
        (cbReluB (R * N) (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN))) r)
    (r50DownSyncCotA R N h w p XIN dys) r

noncomputable def r50DownSyncCotN2 (r : Fin R) : Vec (N * (mid * h * w)) :=
  reluMaskB (N * (mid * h * w))
    (batchShard R N (mid * h * w)
      (bnBatchLA (R * N) mid h w p.ε₂ p.γ₂ p.β₂ (batchMap (R * N) (flatConvStride2 p.W₂ p.b₂)
        (cbReluB (R * N) (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN))) r)
    (cInB N p.W₃ p.b₃ (r50DownSyncCotC3 R hR N h w p XIN dys r))

/-- …at the STRIDED 3×3's output. Feeds `W₂`. -/
noncomputable def r50DownSyncCotC2 (r : Fin R) : Vec (N * (mid * h * w)) :=
  bnSyncInB R hR N mid h w p.ε₂ p.γ₂
    (fun r => batchShard R N (mid * h * w) (batchMap (R * N) (flatConvStride2 p.W₂ p.b₂)
      (cbReluB (R * N) (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN)) r)
    (r50DownSyncCotN2 R hR N h w p XIN dys) r

/-- …at bn₁'s output — at the INPUT grid `2h × 2w`, the strided 3×3's input-VJP having upsampled. -/
noncomputable def r50DownSyncCotN1 (r : Fin R) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  reluMaskB (N * (mid * (2 * h) * (2 * w)))
    (batchShard R N (mid * (2 * h) * (2 * w))
      (bnBatchLA (R * N) mid (2 * h) (2 * w) p.ε₁ p.γ₁ p.β₁
        (batchMap (R * N) (flatConv p.W₁ p.b₁) XIN)) r)
    (cStridedInB N p.W₂ p.b₂ (r50DownSyncCotC2 R hR N h w p XIN dys r))

/-- …at conv₁'s output, through bn₁'s sync backward at `2h × 2w`. Feeds `W₁`. -/
noncomputable def r50DownSyncCotC1 (r : Fin R) : Vec (N * (mid * (2 * h) * (2 * w))) :=
  bnSyncInB R hR N mid (2 * h) (2 * w) p.ε₁ p.γ₁
    (fun r => batchShard R N (mid * (2 * h) * (2 * w)) (batchMap (R * N) (flatConv p.W₁ p.b₁) XIN) r)
    (r50DownSyncCotN1 R hR N h w p XIN dys) r

/-- …at the strided 1×1 projection's output. Feeds `Wp`. -/
noncomputable def r50DownSyncCotCp (r : Fin R) : Vec (N * (oc * h * w)) :=
  bnSyncInB R hR N oc h w p.εp p.γp
    (fun r => batchShard R N (oc * h * w) (batchMap (R * N) (flatConvStride2 p.Wp p.bp) XIN) r)
    (r50DownSyncCotA R N h w p XIN dys) r

/-- The replica's block-INPUT cotangent: conv₁'s input-VJP plus the strided skip's. -/
noncomputable def r50DownSyncCotIn (r : Fin R) : Vec (N * (ic * (2 * h) * (2 * w))) :=
  fun i => cInB N p.W₁ p.b₁ (r50DownSyncCotC1 R hR N h w p XIN dys r) i
    + cStridedInB N p.Wp p.bp (r50DownSyncCotCp R hR N h w p XIN dys r) i

end DownBlock

section DownBlockShard
variable (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat} (hN : 0 < N) (hh : 0 < h)
  (hw : 0 < w) (p : R50ProjW ic mid oc) (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
  (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
  (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r)
include hdys

theorem r50DownSyncCotA_shard (r : Fin R) :
    r50DownSyncCotA R N h w p XIN dys r
      = batchShard R N (oc * h * w) (r50DownCotA (R * N) h w p XIN DY) r := by
  unfold r50DownSyncCotA; rw [hdys]; rfl

include hN hh hw in
theorem r50DownSyncCotC3_shard (r : Fin R) :
    r50DownSyncCotC3 R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r50DownCotC3 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50DownSyncCotA_shard R N h w p XIN dys DY hdys) r

include hN hh hw in
theorem r50DownSyncCotN2_shard (r : Fin R) :
    r50DownSyncCotN2 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50DownCotN2 (R * N) h w p XIN DY) r := by
  unfold r50DownSyncCotN2
  rw [r50DownSyncCotC3_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl

include hN hh hw in
theorem r50DownSyncCotC2_shard (r : Fin R) :
    r50DownSyncCotC2 R hR N h w p XIN dys r
      = batchShard R N (mid * h * w) (r50DownCotC2 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50DownSyncCotN2_shard R hR N h w hN hh hw p XIN dys DY hdys) r

include hN hh hw in
theorem r50DownSyncCotN1_shard (r : Fin R) :
    r50DownSyncCotN1 R hR N h w p XIN dys r
      = batchShard R N (mid * (2 * h) * (2 * w)) (r50DownCotN1 (R * N) h w p XIN DY) r := by
  unfold r50DownSyncCotN1
  rw [r50DownSyncCotC2_shard R hR N h w hN hh hw p XIN dys DY hdys, cStridedInB_shard]
  rfl

include hN hh hw in
/-- ⚠ bn₁'s sync site reduces over `N·(2h)·(2w)` per replica — the one site in the net where the
    reduction width is not the block's output grid. -/
theorem r50DownSyncCotC1_shard (r : Fin R) :
    r50DownSyncCotC1 R hR N h w p XIN dys r
      = batchShard R N (mid * (2 * h) * (2 * w)) (r50DownCotC1 (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N mid (2 * h) (2 * w)
    (nhw_ne_zero hN (Nat.mul_pos (by norm_num) hh) (Nat.mul_pos (by norm_num) hw))
    (nhw_ne_zero (Nat.mul_pos hR hN) (Nat.mul_pos (by norm_num) hh) (Nat.mul_pos (by norm_num) hw))
    _ _ _ _ _ _ (fun _ => rfl) (r50DownSyncCotN1_shard R hR N h w hN hh hw p XIN dys DY hdys) r

include hN hh hw in
theorem r50DownSyncCotCp_shard (r : Fin R) :
    r50DownSyncCotCp R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r50DownCotCp (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r50DownSyncCotA_shard R N h w p XIN dys DY hdys) r

include hN hh hw in
theorem r50DownSyncCotIn_shard (r : Fin R) :
    r50DownSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w)) (r50DownCotIn (R * N) h w p XIN DY) r := by
  unfold r50DownSyncCotIn
  rw [r50DownSyncCotC1_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard,
    r50DownSyncCotCp_shard R hR N h w hN hh hw p XIN dys DY hdys, cStridedInB_shard]
  rfl

end DownBlockShard

-- ════════════════════════════════════════════════════════════════
-- § 3. The per-block DP ties
--   Each conjunct is one of `ResNet34SyncTieB`'s collectives (`ConvWSync`, `ConvStridedWSync`,
--   `BnSync`), generic in the kernel size, so the 1×1 convs and the strided 1×1 skip need no new
--   collective lemma. Tags are the render's: a gradient collective is named for its parameter
--   (`{p}W1`, `{p}g1`, `{p}bt1`, …), the γ node reads the forward's `{p}g1mu` / `{p}g1var`.
-- ════════════════════════════════════════════════════════════════

/-- **Identity bottleneck, DP-tied.** Its nine emitted parameter collectives — the three conv
    weights and the three BatchNorms' γ and β — each equal the single-device node at the global
    batch, at the single-device chain cotangents driven by `DY`, when the replicas' block-output
    cotangents are `R ×` its shards. ⚠ Each BN's γ/β reads the cotangent at THAT BN's output
    (`N1`, `N2`, `A`), each conv the one at the conv's output (`C1`, `C2`, `C3`) — T3's wiring. -/
def r50IdSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc : Nat}
    (pfx xN cotN vN epsStr : String) (p : R50IdW mid oc) (XIN : Vec ((R * N) * (oc * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  let r1 := cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN
  let r2 := cbReluB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ r1
  let c1 := batchMap (R * N) (flatConv p.W₁ p.b₁) XIN
  let c2 := batchMap (R * N) (flatConv p.W₂ p.b₂) r1
  let c3 := batchMap (R * N) (flatConv p.W₃ p.b₃) r2
  ConvWSync R hR N h w s!"{pfx}W1" xN cotN p.b₁ XIN p.W₁
      (r50IdSyncCotC1 R hR N h w p XIN dys) (r50IdCotC1 (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid h w s!"{pfx}g1" s!"{pfx}bt1" vN epsStr cotN p.ε₁ c1
      (r50IdSyncCotN1 R hR N h w p XIN dys) (r50IdCotN1 (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"{pfx}W2" xN cotN p.b₂ r1 p.W₂
      (r50IdSyncCotC2 R hR N h w p XIN dys) (r50IdCotC2 (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid h w s!"{pfx}g2" s!"{pfx}bt2" vN epsStr cotN p.ε₂ c2
      (r50IdSyncCotN2 R hR N h w p XIN dys) (r50IdCotN2 (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"{pfx}W3" xN cotN p.b₃ r2 p.W₃
      (r50IdSyncCotC3 R hR N h w p XIN dys) (r50IdCotC3 (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"{pfx}g3" s!"{pfx}bt3" vN epsStr cotN p.ε₃ c3
      (r50IdSyncCotA R N h w p XIN dys) (r50IdCotA (R * N) h w p XIN DY)

/-- The scaled-shard invariant, carried through one identity bottleneck: replicas at `R ×` the
    shards of `DY` produce block-input cotangents at `R ×` the shards of the single-device one. -/
theorem r50IdSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (p : R50IdW mid oc) (XIN : Vec ((R * N) * (oc * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    r50IdSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (fun i => (R : ℝ) * r50IdCotIn (R * N) h w p XIN DY i) r := by
  rw [r50IdSyncCotIn_shard R hR N h w hN hh hw p XIN dys _ hdys, r50IdCotIn_smul]

theorem r50_idblock_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {mid oc : Nat} (hN : 0 < N)
    (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String) (p : R50IdW mid oc)
    (XIN : Vec ((R * N) * (oc * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    r50IdSyncTiedB R hR N h w pfx xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hw
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50IdSyncCotC1_shard R hR N h w hN hh hw p XIN dys _ hdys, r50IdCotC1_smul])
  · exact bnSync_of_scaled R hR N mid h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50IdSyncCotN1_shard R hR N h w hN hh hw p XIN dys _ hdys, r50IdCotN1_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50IdSyncCotC2_shard R hR N h w hN hh hw p XIN dys _ hdys, r50IdCotC2_smul])
  · exact bnSync_of_scaled R hR N mid h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50IdSyncCotN2_shard R hR N h w hN hh hw p XIN dys _ hdys, r50IdCotN2_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50IdSyncCotC3_shard R hR N h w hN hh hw p XIN dys _ hdys, r50IdCotC3_smul])
  · exact bnSync_of_scaled R hR N oc h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50IdSyncCotA_shard R N h w p XIN dys _ hdys, r50IdCotA_smul])

/-- ⭐ **Stride-1 projection bottleneck, DP-tied** — stage 1 block 0, twelve collectives: the
    identity bottleneck's nine plus the stride-1 1×1 skip's weight (an ORDINARY `ConvWSync`) and
    its BatchNorm's γ and β, which read `A`, the post-relu cotangent. -/
def r50ProjSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (pfx xN cotN vN epsStr : String) (p : R50ProjW ic mid oc) (XIN : Vec ((R * N) * (ic * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  let r1 := cbReluB (R * N) (h := h) (w := w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN
  let r2 := cbReluB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ r1
  let c1 := batchMap (R * N) (flatConv p.W₁ p.b₁) XIN
  let c2 := batchMap (R * N) (flatConv p.W₂ p.b₂) r1
  let c3 := batchMap (R * N) (flatConv p.W₃ p.b₃) r2
  let cp := batchMap (R * N) (flatConv p.Wp p.bp) XIN
  ConvWSync R hR N h w s!"{pfx}W1" xN cotN p.b₁ XIN p.W₁
      (r50ProjSyncCotC1 R hR N h w p XIN dys) (r50ProjCotC1 (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid h w s!"{pfx}g1" s!"{pfx}bt1" vN epsStr cotN p.ε₁ c1
      (r50ProjSyncCotN1 R hR N h w p XIN dys) (r50ProjCotN1 (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"{pfx}W2" xN cotN p.b₂ r1 p.W₂
      (r50ProjSyncCotC2 R hR N h w p XIN dys) (r50ProjCotC2 (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid h w s!"{pfx}g2" s!"{pfx}bt2" vN epsStr cotN p.ε₂ c2
      (r50ProjSyncCotN2 R hR N h w p XIN dys) (r50ProjCotN2 (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"{pfx}W3" xN cotN p.b₃ r2 p.W₃
      (r50ProjSyncCotC3 R hR N h w p XIN dys) (r50ProjCotC3 (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"{pfx}g3" s!"{pfx}bt3" vN epsStr cotN p.ε₃ c3
      (r50ProjSyncCotA R N h w p XIN dys) (r50ProjCotA (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"{pfx}Wp" xN cotN p.bp XIN p.Wp
      (r50ProjSyncCotCp R hR N h w p XIN dys) (r50ProjCotCp (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"{pfx}gp" s!"{pfx}btp" vN epsStr cotN p.εp cp
      (r50ProjSyncCotA R N h w p XIN dys) (r50ProjCotA (R * N) h w p XIN DY)

theorem r50ProjSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : R50ProjW ic mid oc)
    (XIN : Vec ((R * N) * (ic * h * w))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    r50ProjSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * h * w) (fun i => (R : ℝ) * r50ProjCotIn (R * N) h w p XIN DY i) r := by
  rw [r50ProjSyncCotIn_shard R hR N h w hN hh hw p XIN dys _ hdys, r50ProjCotIn_smul]

theorem r50_projblock_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String)
    (p : R50ProjW ic mid oc) (XIN : Vec ((R * N) * (ic * h * w)))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    r50ProjSyncTiedB R hR N h w pfx xN cotN vN epsStr p XIN dys DY := by
  have hm := nhw_ne_zero hN hh hw
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50ProjSyncCotC1_shard R hR N h w hN hh hw p XIN dys _ hdys, r50ProjCotC1_smul])
  · exact bnSync_of_scaled R hR N mid h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50ProjSyncCotN1_shard R hR N h w hN hh hw p XIN dys _ hdys, r50ProjCotN1_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50ProjSyncCotC2_shard R hR N h w hN hh hw p XIN dys _ hdys, r50ProjCotC2_smul])
  · exact bnSync_of_scaled R hR N mid h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50ProjSyncCotN2_shard R hR N h w hN hh hw p XIN dys _ hdys, r50ProjCotN2_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50ProjSyncCotC3_shard R hR N h w hN hh hw p XIN dys _ hdys, r50ProjCotC3_smul])
  · exact bnSync_of_scaled R hR N oc h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50ProjSyncCotA_shard R N h w p XIN dys _ hdys, r50ProjCotA_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50ProjSyncCotCp_shard R hR N h w hN hh hw p XIN dys _ hdys, r50ProjCotCp_smul])
  · exact bnSync_of_scaled R hR N oc h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50ProjSyncCotA_shard R N h w p XIN dys _ hdys, r50ProjCotA_smul])

/-- **Strided projection bottleneck, DP-tied** — stages 2/3/4 block 0, twelve collectives. ⚠⚠ v1.5:
    `W₁` is an ordinary `ConvWSync` at the INPUT grid `2h × 2w` and bn₁'s γ/β reduce there; only
    `W₂` (the 3×3) and `Wp` (the 1×1 skip) are `ConvStridedWSync`. -/
def r50DownSyncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (pfx xN cotN vN epsStr : String) (p : R50ProjW ic mid oc)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w))) : Prop :=
  let r1 := cbReluB (R * N) (h := 2 * h) (w := 2 * w) p.W₁ p.b₁ p.ε₁ p.γ₁ p.β₁ XIN
  let r2 := cbReluStridedB (R * N) (h := h) (w := w) p.W₂ p.b₂ p.ε₂ p.γ₂ p.β₂ r1
  let c1 := batchMap (R * N) (flatConv p.W₁ p.b₁) XIN
  let c2 := batchMap (R * N) (flatConvStride2 p.W₂ p.b₂) r1
  let c3 := batchMap (R * N) (flatConv p.W₃ p.b₃) r2
  let cp := batchMap (R * N) (flatConvStride2 p.Wp p.bp) XIN
  ConvWSync R hR N (2 * h) (2 * w) s!"{pfx}W1" xN cotN p.b₁ XIN p.W₁
      (r50DownSyncCotC1 R hR N h w p XIN dys) (r50DownCotC1 (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid (2 * h) (2 * w) s!"{pfx}g1" s!"{pfx}bt1" vN epsStr cotN p.ε₁ c1
      (r50DownSyncCotN1 R hR N h w p XIN dys) (r50DownCotN1 (R * N) h w p XIN DY)
  ∧ ConvStridedWSync R hR N h w s!"{pfx}W2" xN cotN p.b₂ r1 p.W₂
      (r50DownSyncCotC2 R hR N h w p XIN dys) (r50DownCotC2 (R * N) h w p XIN DY)
  ∧ BnSync R hR N mid h w s!"{pfx}g2" s!"{pfx}bt2" vN epsStr cotN p.ε₂ c2
      (r50DownSyncCotN2 R hR N h w p XIN dys) (r50DownCotN2 (R * N) h w p XIN DY)
  ∧ ConvWSync R hR N h w s!"{pfx}W3" xN cotN p.b₃ r2 p.W₃
      (r50DownSyncCotC3 R hR N h w p XIN dys) (r50DownCotC3 (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"{pfx}g3" s!"{pfx}bt3" vN epsStr cotN p.ε₃ c3
      (r50DownSyncCotA R N h w p XIN dys) (r50DownCotA (R * N) h w p XIN DY)
  ∧ ConvStridedWSync R hR N h w s!"{pfx}Wp" xN cotN p.bp XIN p.Wp
      (r50DownSyncCotCp R hR N h w p XIN dys) (r50DownCotCp (R * N) h w p XIN DY)
  ∧ BnSync R hR N oc h w s!"{pfx}gp" s!"{pfx}btp" vN epsStr cotN p.εp cp
      (r50DownSyncCotA R N h w p XIN dys) (r50DownCotA (R * N) h w p XIN DY)

theorem r50DownSyncCotIn_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (p : R50ProjW ic mid oc)
    (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (dys : Fin R → Vec (N * (oc * h * w)))
    (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) (r : Fin R) :
    r50DownSyncCotIn R hR N h w p XIN dys r
      = batchShard R N (ic * (2 * h) * (2 * w))
          (fun i => (R : ℝ) * r50DownCotIn (R * N) h w p XIN DY i) r := by
  rw [r50DownSyncCotIn_shard R hR N h w hN hh hw p XIN dys _ hdys, r50DownCotIn_smul]

theorem r50_downblock_syncTiedB (R : Nat) (hR : 0 < R) (N h w : Nat) {ic mid oc : Nat}
    (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) (pfx xN cotN vN epsStr : String)
    (p : R50ProjW ic mid oc) (XIN : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (dys : Fin R → Vec (N * (oc * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * DY i) r) :
    r50DownSyncTiedB R hR N h w pfx xN cotN vN epsStr p XIN dys DY := by
  have h2h : 0 < 2 * h := Nat.mul_pos (by norm_num) hh
  have h2w : 0 < 2 * w := Nat.mul_pos (by norm_num) hw
  have hm := nhw_ne_zero hN hh hw
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N (2 * h) (2 * w) _ _ _ _ _ _ _ _ (fun r => by
      rw [r50DownSyncCotC1_shard R hR N h w hN hh hw p XIN dys _ hdys, r50DownCotC1_smul])
  · exact bnSync_of_scaled R hR N mid (2 * h) (2 * w) (nhw_ne_zero hN h2h h2w)
      (nhw_ne_zero (Nat.mul_pos hR hN) h2h h2w) _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50DownSyncCotN1_shard R hR N h w hN hh hw p XIN dys _ hdys, r50DownCotN1_smul])
  · exact convStridedWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50DownSyncCotC2_shard R hR N h w hN hh hw p XIN dys _ hdys, r50DownCotC2_smul])
  · exact bnSync_of_scaled R hR N mid h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50DownSyncCotN2_shard R hR N h w hN hh hw p XIN dys _ hdys, r50DownCotN2_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50DownSyncCotC3_shard R hR N h w hN hh hw p XIN dys _ hdys, r50DownCotC3_smul])
  · exact bnSync_of_scaled R hR N oc h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50DownSyncCotA_shard R N h w p XIN dys _ hdys, r50DownCotA_smul])
  · exact convStridedWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r50DownSyncCotCp_shard R hR N h w hN hh hw p XIN dys _ hdys, r50DownCotCp_smul])
  · exact bnSync_of_scaled R hR N oc h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r50DownSyncCotA_shard R N h w p XIN dys _ hdys, r50DownCotA_smul])

-- ════════════════════════════════════════════════════════════════
-- § 4. The whole-net capstone
--   ⭐ The stem and head ties are ResNet-34's, reused verbatim, as T3 reuses its single-device
--   ones: `r34StemSyncTiedB` is generic in `{ic oc}` and tags `sW` / `sg` / `sbt`, the names
--   R50's stem shares; `r34HeadSyncTiedB` is generic in `{c nCls}` and tags `Wd` / `bd`.
-- ════════════════════════════════════════════════════════════════

/-- **Every all-reduced parameter gradient of the sync-BN data-parallel ResNet-50 step, tied** —
    the statement `r50_net_syncTiedB` proves, named so the per-loss corollaries can state it at
    their cotangents.

    The single-device chain is T3's (`r50_net_tiedB`'s `dy_k`) at the global batch `R·N`, driven by
    the global loss cotangent `G`. The replica chain is each replica's own: the head backward on its
    shard of the trunk's output at its own cotangent `gs r`, then sixteen bottleneck backwards,
    every BatchNorm the sync backward. The conjuncts are the 161 collectives the render emits —
    stem 3, twelve identity bottlenecks × 9, four projection bottlenecks × 12, dense 2 — each
    equal to the single-device gradient node at `N := R·N`. -/
def r50NetSyncTiedB (R : Nat) (hR : 0 < R) (N q : Nat) {nCls : Nat} (xN cotN vN epsStr : String)
    (w : R50BWeights nCls)
    (X : Vec ((R * N) * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (G : Vec ((R * N) * nCls)) (gs : Fin R → Vec (N * nCls)) : Prop :=
  -- ── the single-device chain at the global batch `R·N`, from `G` ──
  let dy16 := r34HeadCotBlk (R * N) q q w.Wd w.bd (r50Pre16 (R * N) q w X) G
  let dy15 := r50IdCotIn (R * N) q q w.s4b2 (r50Pre15 (R * N) q w X) dy16
  let dy14 := r50IdCotIn (R * N) q q w.s4b1 (r50Pre14 (R * N) q w X) dy15
  let dy13 := r50DownCotIn (R * N) q q w.s4b0 (r50Pre13 (R * N) q w X) dy14
  let dy12 := r50IdCotIn (R * N) (2 * q) (2 * q) w.s3b5 (r50Pre12 (R * N) q w X) dy13
  let dy11 := r50IdCotIn (R * N) (2 * q) (2 * q) w.s3b4 (r50Pre11 (R * N) q w X) dy12
  let dy10 := r50IdCotIn (R * N) (2 * q) (2 * q) w.s3b3 (r50Pre10 (R * N) q w X) dy11
  let dy9 := r50IdCotIn (R * N) (2 * q) (2 * q) w.s3b2 (r50Pre9 (R * N) q w X) dy10
  let dy8 := r50IdCotIn (R * N) (2 * q) (2 * q) w.s3b1 (r50Pre8 (R * N) q w X) dy9
  let dy7 := r50DownCotIn (R * N) (2 * q) (2 * q) w.s3b0 (r50Pre7 (R * N) q w X) dy8
  let dy6 := r50IdCotIn (R * N) (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Pre6 (R * N) q w X) dy7
  let dy5 := r50IdCotIn (R * N) (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Pre5 (R * N) q w X) dy6
  let dy4 := r50IdCotIn (R * N) (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Pre4 (R * N) q w X) dy5
  let dy3 := r50DownCotIn (R * N) (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Pre3 (R * N) q w X) dy4
  let dy2 := r50IdCotIn (R * N) (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2
    (r50Pre2 (R * N) q w X) dy3
  let dy1 := r50IdCotIn (R * N) (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1
    (r50Pre1 (R * N) q w X) dy2
  let cotPool := r50ProjCotIn (R * N) (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0
    (r50Pre0 (R * N) q w X) dy1
  -- ── replica `r`, from its own cotangent `gs r`, its own sync-BN chain ──
  let e16 : Fin R → Vec (N * (2048 * q * q)) := fun r =>
    r34HeadCotBlk N q q w.Wd w.bd (batchShard R N _ (r50Pre16 (R * N) q w X) r) (gs r)
  let e15 := r50IdSyncCotIn R hR N q q w.s4b2 (r50Pre15 (R * N) q w X) e16
  let e14 := r50IdSyncCotIn R hR N q q w.s4b1 (r50Pre14 (R * N) q w X) e15
  let e13 := r50DownSyncCotIn R hR N q q w.s4b0 (r50Pre13 (R * N) q w X) e14
  let e12 := r50IdSyncCotIn R hR N (2 * q) (2 * q) w.s3b5 (r50Pre12 (R * N) q w X) e13
  let e11 := r50IdSyncCotIn R hR N (2 * q) (2 * q) w.s3b4 (r50Pre11 (R * N) q w X) e12
  let e10 := r50IdSyncCotIn R hR N (2 * q) (2 * q) w.s3b3 (r50Pre10 (R * N) q w X) e11
  let e9 := r50IdSyncCotIn R hR N (2 * q) (2 * q) w.s3b2 (r50Pre9 (R * N) q w X) e10
  let e8 := r50IdSyncCotIn R hR N (2 * q) (2 * q) w.s3b1 (r50Pre8 (R * N) q w X) e9
  let e7 := r50DownSyncCotIn R hR N (2 * q) (2 * q) w.s3b0 (r50Pre7 (R * N) q w X) e8
  let e6 := r50IdSyncCotIn R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b3 (r50Pre6 (R * N) q w X) e7
  let e5 := r50IdSyncCotIn R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b2 (r50Pre5 (R * N) q w X) e6
  let e4 := r50IdSyncCotIn R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b1 (r50Pre4 (R * N) q w X) e5
  let e3 := r50DownSyncCotIn R hR N (2 * (2 * q)) (2 * (2 * q)) w.s2b0 (r50Pre3 (R * N) q w X) e4
  let e2 := r50IdSyncCotIn R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b2
    (r50Pre2 (R * N) q w X) e3
  let e1 := r50IdSyncCotIn R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b1
    (r50Pre1 (R * N) q w X) e2
  let ePool := r50ProjSyncCotIn R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) w.s1b0
    (r50Pre0 (R * N) q w X) e1
  r34StemSyncTiedB R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xN cotN vN epsStr
      w.sW w.sb w.sε w.sγ w.sβ X ePool cotPool
  ∧ r50ProjSyncTiedB R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) "s1b0" xN cotN vN epsStr
      w.s1b0 (r50Pre0 (R * N) q w X) e1 dy1
  ∧ r50IdSyncTiedB R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) "s1b1" xN cotN vN epsStr
      w.s1b1 (r50Pre1 (R * N) q w X) e2 dy2
  ∧ r50IdSyncTiedB R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) "s1b2" xN cotN vN epsStr
      w.s1b2 (r50Pre2 (R * N) q w X) e3 dy3
  ∧ r50DownSyncTiedB R hR N (2 * (2 * q)) (2 * (2 * q)) "s2b0" xN cotN vN epsStr
      w.s2b0 (r50Pre3 (R * N) q w X) e4 dy4
  ∧ r50IdSyncTiedB R hR N (2 * (2 * q)) (2 * (2 * q)) "s2b1" xN cotN vN epsStr
      w.s2b1 (r50Pre4 (R * N) q w X) e5 dy5
  ∧ r50IdSyncTiedB R hR N (2 * (2 * q)) (2 * (2 * q)) "s2b2" xN cotN vN epsStr
      w.s2b2 (r50Pre5 (R * N) q w X) e6 dy6
  ∧ r50IdSyncTiedB R hR N (2 * (2 * q)) (2 * (2 * q)) "s2b3" xN cotN vN epsStr
      w.s2b3 (r50Pre6 (R * N) q w X) e7 dy7
  ∧ r50DownSyncTiedB R hR N (2 * q) (2 * q) "s3b0" xN cotN vN epsStr
      w.s3b0 (r50Pre7 (R * N) q w X) e8 dy8
  ∧ r50IdSyncTiedB R hR N (2 * q) (2 * q) "s3b1" xN cotN vN epsStr
      w.s3b1 (r50Pre8 (R * N) q w X) e9 dy9
  ∧ r50IdSyncTiedB R hR N (2 * q) (2 * q) "s3b2" xN cotN vN epsStr
      w.s3b2 (r50Pre9 (R * N) q w X) e10 dy10
  ∧ r50IdSyncTiedB R hR N (2 * q) (2 * q) "s3b3" xN cotN vN epsStr
      w.s3b3 (r50Pre10 (R * N) q w X) e11 dy11
  ∧ r50IdSyncTiedB R hR N (2 * q) (2 * q) "s3b4" xN cotN vN epsStr
      w.s3b4 (r50Pre11 (R * N) q w X) e12 dy12
  ∧ r50IdSyncTiedB R hR N (2 * q) (2 * q) "s3b5" xN cotN vN epsStr
      w.s3b5 (r50Pre12 (R * N) q w X) e13 dy13
  ∧ r50DownSyncTiedB R hR N q q "s4b0" xN cotN vN epsStr w.s4b0 (r50Pre13 (R * N) q w X) e14 dy14
  ∧ r50IdSyncTiedB R hR N q q "s4b1" xN cotN vN epsStr w.s4b1 (r50Pre14 (R * N) q w X) e15 dy15
  ∧ r50IdSyncTiedB R hR N q q "s4b2" xN cotN vN epsStr w.s4b2 (r50Pre15 (R * N) q w X) e16 dy16
  ∧ r34HeadSyncTiedB R hR N q q xN cotN (r50Pre16 (R * N) q w X) gs G

/-- ⭐⭐⭐ **The synchronised-BN data-parallel ResNet-50 step IS the single-device step at the global
    batch.** `R` replicas at batch `N`, each running the render's sync-BN backward chain from its
    own loss cotangent `gs r`; every parameter's all-reduced mean gradient — the 161 the render
    emits — equals the single-device batch-BN gradient node at batch `R·N`, at the cotangent T3's
    chain delivers there from `G`, whenever each replica's cotangent is `R ×` its shard of `G`.

    ⭐ The hypothesis `hgs` is the divisor step, left open because T3 leaves the loss open: a
    replica divides its loss by its own batch and the global step by `R ×` that, so a replica's
    cotangent is `R ×` its shard of the global one — for the label-smoothed CE
    (`r50_net_syncTiedB_smoothedCE`) and for BCE-with-logits (`r50_net_syncTiedB_bce`) alike.

    ⭐ The right-hand chain is `r50_net_tiedB`'s at `N := R·N`, `g := G`, whose nodes that capstone
    ties to the certified gradient — so this and it together say the DP step's update is the
    certified gradient of the mean loss over all `R·N` examples. `N`, `q` are binders, so one
    statement covers the 224-px (`q = 7`) and 160-px (`q = 5`) artifacts; `0 < q` makes every
    BatchNorm's reduction width nonzero. -/
theorem r50_net_syncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (q : Nat) (hq : 0 < q)
    {nCls : Nat} (xN cotN vN epsStr : String) (w : R50BWeights nCls)
    (X : Vec ((R * N) * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (G : Vec ((R * N) * nCls)) (gs : Fin R → Vec (N * nCls))
    (hgs : ∀ r, gs r = batchShard R N nCls (fun i => (R : ℝ) * G i) r) :
    r50NetSyncTiedB R hR N q xN cotN vN epsStr w X G gs := by
  unfold r50NetSyncTiedB
  intro dy16 dy15 dy14 dy13 dy12 dy11 dy10 dy9 dy8 dy7 dy6 dy5 dy4 dy3 dy2 dy1 cotPool
    e16 e15 e14 e13 e12 e11 e10 e9 e8 e7 e6 e5 e4 e3 e2 e1 ePool
  have h1 : 0 < q := hq
  have h2 : 0 < 2 * q := by omega
  have h4 : 0 < 2 * (2 * q) := by omega
  have h8 : 0 < 2 * (2 * (2 * q)) := by omega
  -- the scaled-shard invariant, block by block down the chain
  have s16 : ∀ r, e16 r = batchShard R N _ (fun i => (R : ℝ) * dy16 i) r :=
    fun r => r34HeadCotBlk_scaled R N q q w.Wd w.bd (r50Pre16 (R * N) q w X) gs G hgs r
  have s15 := r50IdSyncCotIn_scaled R hR N q q hN h1 h1 w.s4b2 (r50Pre15 (R * N) q w X) e16 dy16 s16
  have s14 := r50IdSyncCotIn_scaled R hR N q q hN h1 h1 w.s4b1 (r50Pre14 (R * N) q w X) e15 dy15 s15
  have s13 := r50DownSyncCotIn_scaled R hR N q q hN h1 h1 w.s4b0 (r50Pre13 (R * N) q w X)
    e14 dy14 s14
  have s12 := r50IdSyncCotIn_scaled R hR N (2 * q) (2 * q) hN h2 h2 w.s3b5
    (r50Pre12 (R * N) q w X) e13 dy13 s13
  have s11 := r50IdSyncCotIn_scaled R hR N (2 * q) (2 * q) hN h2 h2 w.s3b4
    (r50Pre11 (R * N) q w X) e12 dy12 s12
  have s10 := r50IdSyncCotIn_scaled R hR N (2 * q) (2 * q) hN h2 h2 w.s3b3
    (r50Pre10 (R * N) q w X) e11 dy11 s11
  have s9 := r50IdSyncCotIn_scaled R hR N (2 * q) (2 * q) hN h2 h2 w.s3b2
    (r50Pre9 (R * N) q w X) e10 dy10 s10
  have s8 := r50IdSyncCotIn_scaled R hR N (2 * q) (2 * q) hN h2 h2 w.s3b1
    (r50Pre8 (R * N) q w X) e9 dy9 s9
  have s7 := r50DownSyncCotIn_scaled R hR N (2 * q) (2 * q) hN h2 h2 w.s3b0
    (r50Pre7 (R * N) q w X) e8 dy8 s8
  have s6 := r50IdSyncCotIn_scaled R hR N (2 * (2 * q)) (2 * (2 * q)) hN h4 h4 w.s2b3
    (r50Pre6 (R * N) q w X) e7 dy7 s7
  have s5 := r50IdSyncCotIn_scaled R hR N (2 * (2 * q)) (2 * (2 * q)) hN h4 h4 w.s2b2
    (r50Pre5 (R * N) q w X) e6 dy6 s6
  have s4 := r50IdSyncCotIn_scaled R hR N (2 * (2 * q)) (2 * (2 * q)) hN h4 h4 w.s2b1
    (r50Pre4 (R * N) q w X) e5 dy5 s5
  have s3 := r50DownSyncCotIn_scaled R hR N (2 * (2 * q)) (2 * (2 * q)) hN h4 h4 w.s2b0
    (r50Pre3 (R * N) q w X) e4 dy4 s4
  have s2 := r50IdSyncCotIn_scaled R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) hN h8 h8 w.s1b2
    (r50Pre2 (R * N) q w X) e3 dy3 s3
  have s1 := r50IdSyncCotIn_scaled R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) hN h8 h8 w.s1b1
    (r50Pre1 (R * N) q w X) e2 dy2 s2
  have sPool := r50ProjSyncCotIn_scaled R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) hN h8 h8
    w.s1b0 (r50Pre0 (R * N) q w X) e1 dy1 s1
  exact ⟨r34_stem_syncTiedB R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) hN h8 h8
      xN cotN vN epsStr w.sW w.sb w.sε w.sγ w.sβ X ePool _ sPool,
    r50_projblock_syncTiedB R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) hN h8 h8
      "s1b0" xN cotN vN epsStr w.s1b0 (r50Pre0 (R * N) q w X) e1 dy1 s1,
    r50_idblock_syncTiedB R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) hN h8 h8
      "s1b1" xN cotN vN epsStr w.s1b1 (r50Pre1 (R * N) q w X) e2 dy2 s2,
    r50_idblock_syncTiedB R hR N (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) hN h8 h8
      "s1b2" xN cotN vN epsStr w.s1b2 (r50Pre2 (R * N) q w X) e3 dy3 s3,
    r50_downblock_syncTiedB R hR N (2 * (2 * q)) (2 * (2 * q)) hN h4 h4
      "s2b0" xN cotN vN epsStr w.s2b0 (r50Pre3 (R * N) q w X) e4 dy4 s4,
    r50_idblock_syncTiedB R hR N (2 * (2 * q)) (2 * (2 * q)) hN h4 h4
      "s2b1" xN cotN vN epsStr w.s2b1 (r50Pre4 (R * N) q w X) e5 dy5 s5,
    r50_idblock_syncTiedB R hR N (2 * (2 * q)) (2 * (2 * q)) hN h4 h4
      "s2b2" xN cotN vN epsStr w.s2b2 (r50Pre5 (R * N) q w X) e6 dy6 s6,
    r50_idblock_syncTiedB R hR N (2 * (2 * q)) (2 * (2 * q)) hN h4 h4
      "s2b3" xN cotN vN epsStr w.s2b3 (r50Pre6 (R * N) q w X) e7 dy7 s7,
    r50_downblock_syncTiedB R hR N (2 * q) (2 * q) hN h2 h2
      "s3b0" xN cotN vN epsStr w.s3b0 (r50Pre7 (R * N) q w X) e8 dy8 s8,
    r50_idblock_syncTiedB R hR N (2 * q) (2 * q) hN h2 h2
      "s3b1" xN cotN vN epsStr w.s3b1 (r50Pre8 (R * N) q w X) e9 dy9 s9,
    r50_idblock_syncTiedB R hR N (2 * q) (2 * q) hN h2 h2
      "s3b2" xN cotN vN epsStr w.s3b2 (r50Pre9 (R * N) q w X) e10 dy10 s10,
    r50_idblock_syncTiedB R hR N (2 * q) (2 * q) hN h2 h2
      "s3b3" xN cotN vN epsStr w.s3b3 (r50Pre10 (R * N) q w X) e11 dy11 s11,
    r50_idblock_syncTiedB R hR N (2 * q) (2 * q) hN h2 h2
      "s3b4" xN cotN vN epsStr w.s3b4 (r50Pre11 (R * N) q w X) e12 dy12 s12,
    r50_idblock_syncTiedB R hR N (2 * q) (2 * q) hN h2 h2
      "s3b5" xN cotN vN epsStr w.s3b5 (r50Pre12 (R * N) q w X) e13 dy13 s13,
    r50_downblock_syncTiedB R hR N q q hN h1 h1
      "s4b0" xN cotN vN epsStr w.s4b0 (r50Pre13 (R * N) q w X) e14 dy14 s14,
    r50_idblock_syncTiedB R hR N q q hN h1 h1
      "s4b1" xN cotN vN epsStr w.s4b1 (r50Pre14 (R * N) q w X) e15 dy15 s15,
    r50_idblock_syncTiedB R hR N q q hN h1 h1
      "s4b2" xN cotN vN epsStr w.s4b2 (r50Pre15 (R * N) q w X) e16 dy16 s16,
    r34_head_syncTiedB R hR N q q xN cotN (r50Pre16 (R * N) q w X) gs G hgs⟩

-- ════════════════════════════════════════════════════════════════
-- § 5. The two losses — the divisor step, discharged once per loss
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **The BCE divisor step** — `ResNet34SyncTieB.replicaLossCot_eq`'s peer for BCE-with-logits.
    Replica `r`'s three-op chain divides by `bk`, the single-device one at the global batch by
    `R·bk`; at the replica's shard of the logits and targets, the replica's cotangent is `R ×` its
    shard of the global one. `σ(z) − t` is per example, so only the divisor differs. -/
theorem replicaBceLossCot_eq (R N nCls : Nat) (hR : 0 < R) (bk : ℝ) (bStr logN ohN : String)
    (Z : Vec ((R * N) * nCls)) (T : Vec ((R * N) * (1 * nCls))) (r : Fin R) :
    unrowB N nCls (den (bceLossCotGraph N nCls bk bStr logN ohN
        (rowB N nCls (batchShard R N nCls Z r)) (batchShard R N (1 * nCls) T r)))
      = batchShard R N nCls (fun i => (R : ℝ) * unrowB (R * N) nCls
          (den (bceLossCotGraph (R * N) nCls ((R : ℝ) * bk) bStr logN ohN
            (rowB (R * N) nCls Z) T)) i) r := by
  have hRr : (R : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hR)
  -- the replica's cotangent, before the row cast: the shard of `R ×` the global one
  have hden : den (bceLossCotGraph N nCls bk bStr logN ohN
        (batchShard R N (1 * nCls) (rowB (R * N) nCls Z) r) (batchShard R N (1 * nCls) T r))
      = batchShard R N (1 * nCls) (fun m => (R : ℝ) * den (bceLossCotGraph (R * N) nCls
          ((R : ℝ) * bk) bStr logN ohN (rowB (R * N) nCls Z) T) m) r := by
    funext j
    rw [bceLossCotGraph_den]
    simp only [batchShard]
    rw [bceLossCotGraph_den, mul_div_assoc', mul_div_mul_left _ _ hRr]
    rfl
  rw [rowB_shard, hden, unrowB_shard]
  rfl

/-- ⭐ **The sync-BN DP step at the label-smoothed loss** — every `bce := false` DP artifact.
    Replicas divide by `B`, the global step by `R·B`; `replicaLossCot_eq` discharges `hgs`. -/
theorem r50_net_syncTiedB_smoothedCE (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (q : Nat)
    (hq : 0 < q) {nCls : Nat} (xN cotN vN epsStr : String) (aStr negAK bStr logN ohN : String)
    (α B : ℝ) (w : R50BWeights nCls)
    (X : Vec ((R * N) * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (T : Vec ((R * N) * (1 * nCls))) :
    r50NetSyncTiedB R hR N q xN cotN vN epsStr w X
      (unrowB (R * N) nCls (den (smoothedLossCotGraph (R * N) nCls α ((R : ℝ) * B) aStr negAK
        bStr logN ohN (rowB (R * N) nCls (resnet50ForwardB_full (R * N) q w X)) T)))
      (fun r => unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (batchShard R N nCls (resnet50ForwardB_full (R * N) q w X) r))
        (batchShard R N (1 * nCls) T r)))) :=
  r50_net_syncTiedB R hR N hN q hq xN cotN vN epsStr w X _ _
    (fun r => replicaLossCot_eq R N nCls hR α B aStr negAK bStr logN ohN _ T r)

/-- ⭐⭐ **The sync-BN DP step at BCE-with-logits, at the COMMITTED divisors** — every `bce := true`
    DP artifact, including `resnet50in160_lambaccdp8x64bce` (per micro-step). A replica's chain
    divides by `N·K` (its own batch × the class count, what the render bakes at `B := N`); the
    single-device step at the global batch by `(R·N)·K` — `r50_lossCot_is_bce_grad`'s divisor at
    `N := R·N`. `replicaBceLossCot_eq` discharges `hgs` once `(R·N)·K = R·(N·K)`. -/
theorem r50_net_syncTiedB_bce (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) (q : Nat)
    (hq : 0 < q) {nCls : Nat} (xN cotN vN epsStr : String) (bStr logN ohN : String)
    (w : R50BWeights nCls)
    (X : Vec ((R * N) * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))))
    (T : Vec ((R * N) * (1 * nCls))) :
    r50NetSyncTiedB R hR N q xN cotN vN epsStr w X
      (unrowB (R * N) nCls (den (bceLossCotGraph (R * N) nCls (((R * N : Nat) : ℝ) * (nCls : ℝ))
        bStr logN ohN (rowB (R * N) nCls (resnet50ForwardB_full (R * N) q w X)) T)))
      (fun r => unrowB N nCls (den (bceLossCotGraph N nCls ((N : ℝ) * (nCls : ℝ)) bStr logN ohN
        (rowB N nCls (batchShard R N nCls (resnet50ForwardB_full (R * N) q w X) r))
        (batchShard R N (1 * nCls) T r)))) := by
  have hdiv : ((R * N : Nat) : ℝ) * (nCls : ℝ) = (R : ℝ) * ((N : ℝ) * (nCls : ℝ)) := by
    push_cast; ring
  rw [hdiv]
  exact r50_net_syncTiedB R hR N hN q hq xN cotN vN epsStr w X _ _
    (fun r => replicaBceLossCot_eq R N nCls hR _ bStr logN ohN _ T r)

end Proofs.ResNet50SyncTieB
