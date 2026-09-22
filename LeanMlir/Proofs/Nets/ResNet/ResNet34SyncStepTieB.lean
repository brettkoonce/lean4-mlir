import LeanMlir.Proofs.Nets.ResNet.ResNet34StepTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncB

/-! # ResNet-34's data-parallel step at SYNCHRONISED BatchNorm IS the single-device step at `R·N`

`ResNet34StepTieB.lean` (T3) threads the label-smoothed loss cotangent down the batch-BN
backward chain on ONE device and ties every parameter gradient node to the certified gradient.
This is its data-parallel twin, for the render `ResNet34RenderB` emits at `replicas > 1` since
2026-09-21: `R` replicas at batch `N`, every BatchNorm synchronised (`bnFwdSite` / `bnBackSite` /
`bnGammaSite`), every parameter gradient all-reduced by its mean. The capstone
`r34_net_syncTiedB` says that, for every parameter,

    mean over the R replicas of replica r's gradient node, loss divided by B
      = the single-device gradient node at the global batch R·N, loss divided by R·B

— the gradient node `r34_net_tiedB` at `N := R·N` ties to the certified gradient. So the sentence
the DP render header carries ("this step IS the single-device step at the global batch") is a
theorem, and the spec it is stated against has not moved: the right-hand side is the existing
single-device chain at `N := R·N`.

## Four steps

1. **Sharding** — each replica's backward chain, handed its shard of a global cotangent, computes
   the shard of the global chain. Every non-BN link (relu mask, conv and strided-conv input-VJP,
   the 3×3/s2 pool's scatter, the head) is a per-example map and commutes with sharding by
   definition; the BN link is `bnSyncInB`, whose shard lemma `bnSyncInB_shard` is P2 on the
   graph (`DataParallelSync.den_bnSyncBack_allReduce`) carried across the `mul_assoc` seam.
2. **The collectives** — the mean over replicas of each replica's gradient node is `1/R` of the
   global node at the global cotangent (`DataParallelSync`'s P4 lemmas, plus the strided-conv and
   dense ones here). The γ node is the sync one, `bnSyncGammaGradB`, reading the forward's
   all-reduced statistics — the one parameter gradient sync-BN changes.
3. **Homogeneity** — the single-device chain and its gradient nodes are linear in the loss
   cotangent (`*_smul`): `R ×` the cotangent gives `R ×` every node.
4. **The divisor** — replica `r` divides its loss by `B` and the global step by `R·B`, so replica
   `r`'s loss cotangent is `R ×` its shard of the global one (`replicaLossCot_eq`). Steps 1–3
   carry that `R` down the chain and it cancels the collective's `1/R`.

## What is NOT claimed

⚠ The replicas' saved forward activations enter as the shards of the single-device forward's
(`batchShard r (r34Pre_k (R*N) w X)`); that the sync forward graph computes exactly those is
`ResNet34SyncB.resnet34FwdGraphSync_full_shard`, the forward half. ⚠ That the replicas' inputs are
the shards of one batch is the driver's. ⚠ The emitted artifacts run `convBias := false`, so the
conv-bias nodes are not emitted and are not tied here (`r34_net_tiedB` keeps them for the flag).
⚠ The lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ResNet34SyncTieB

open scoped BigOperators
open Proofs.EnetTiePoC (reassocB cInB)
open Proofs.ResNet34TieB

-- ════════════════════════════════════════════════════════════════
-- § 1. Homogeneity — the single-device chain is linear in its cotangent
-- ════════════════════════════════════════════════════════════════

theorem reluMaskB_smul (n : Nat) (pre dy : Vec n) (s : ℝ) :
    reluMaskB n pre (fun i => s * dy i) = fun i => s * reluMaskB n pre dy i := by
  funext i
  unfold reluMaskB
  split_ifs <;> simp

/-- The three-term BatchNorm input-gradient is linear in `dy` — both its reductions are. -/
theorem bn_grad_input_smul (n : Nat) (ε γ : ℝ) (x dy : Vec n) (s : ℝ) :
    bn_grad_input n ε γ x (fun i => s * dy i) = fun i => s * bn_grad_input n ε γ x dy i := by
  funext i
  have h1 : ∑ j : Fin n, γ * (s * dy j) = s * ∑ j : Fin n, γ * dy j := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun _ _ => by ring)
  have h2 : ∑ j : Fin n, bnXhat n ε x j * (γ * (s * dy j))
      = s * ∑ j : Fin n, bnXhat n ε x j * (γ * dy j) := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun _ _ => by ring)
  simp only [bn_grad_input, h1, h2]
  ring

theorem bnPerChannel_grad_input_smul (oc m : Nat) (ε : ℝ) (γ : Vec oc) (x dy : Vec (oc * m))
    (s : ℝ) :
    bnPerChannel_grad_input oc m ε γ x (fun i => s * dy i)
      = fun i => s * bnPerChannel_grad_input oc m ε γ x dy i := by
  funext idx
  exact congrFun (bn_grad_input_smul m ε _ _ (Mat.unflatten dy (finProdFinEquiv.symm idx).1) s) _

theorem bnBatchTensor4_grad_input_smul (N oc h w : Nat) (ε : ℝ) (γ : Vec oc)
    (x dy : Vec (N * (oc * (h * w)))) (s : ℝ) :
    bnBatchTensor4_grad_input N oc h w ε γ x (fun i => s * dy i)
      = fun i => s * bnBatchTensor4_grad_input N oc h w ε γ x dy i := by
  funext i
  exact congrFun (bnPerChannel_grad_input_smul oc (N * (h * w)) ε γ _ (bnchwFwd N oc h w dy) s) _

theorem bnInB_smul (N oc h w : Nat) (ε : ℝ) (γ : Vec oc) (x dy : Vec (N * (oc * h * w))) (s : ℝ) :
    bnInB N oc h w ε γ x (fun i => s * dy i) = fun i => s * bnInB N oc h w ε γ x dy i := by
  rw [bnInB_eq_den_bnBatchBack, bnInB_eq_den_bnBatchBack]
  funext i
  exact congrFun (bnBatchTensor4_grad_input_smul N oc h w ε γ _ (reassocB N oc h w dy) s) _

theorem batchMap_smul {N a b : Nat} (f : Vec a → Vec b)
    (hf : ∀ (s : ℝ) (v : Vec a), f (fun i => s * v i) = fun i => s * f v i) (s : ℝ)
    (X : Vec (N * a)) :
    batchMap N f (fun i => s * X i) = fun i => s * batchMap N f X i := by
  funext idx
  exact congrFun (hf s _) _

theorem batchMapAux_smul {N t a b : Nat} (f : Vec t → Vec a → Vec b)
    (hf : ∀ (s : ℝ) (x : Vec t) (v : Vec a), f x (fun i => s * v i) = fun i => s * f x v i)
    (s : ℝ) (aux : Vec (N * t)) (X : Vec (N * a)) :
    batchMapAux N f aux (fun i => s * X i) = fun i => s * batchMapAux N f aux X i := by
  funext idx
  exact congrFun (hf s _ _) _

theorem cInB_smul (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    cInB N (h := h) (w := w) W b (fun i => s * dy i)
      = fun i => s * cInB N (h := h) (w := w) W b dy i :=
  batchMap_smul _ (fun s v => HasVJP.backward_smul _ _ s v) s dy

theorem cStridedInB_smul (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    cStridedInB N (h := h) (w := w) W b (fun i => s * dy i)
      = fun i => s * cStridedInB N (h := h) (w := w) W b dy i :=
  batchMap_smul _ (fun s v => HasVJP.backward_smul _ _ s v) s dy

/-- The 3×3/s2 pool's `select_and_scatter` is linear in the cotangent it scatters. -/
theorem maxPool3s2BackFlat_smul (c h w : Nat) (xv : Vec (c * (2 * h) * (2 * w)))
    (dyv : Vec (c * h * w)) (s : ℝ) :
    maxPool3s2BackFlat c h w xv (fun i => s * dyv i)
      = fun i => s * maxPool3s2BackFlat c h w xv dyv i := by
  funext idx
  simp only [maxPool3s2BackFlat, Tensor3.unflatten, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun _ _ => Finset.sum_congr rfl (fun _ _ =>
    Finset.sum_congr rfl (fun _ _ => by ring)))

theorem mpInB_smul (N c h w : Nat) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (dy : Vec (N * (c * h * w))) (s : ℝ) :
    mpInB N c h w x (fun i => s * dy i) = fun i => s * mpInB N c h w x dy i :=
  batchMapAux_smul _ (fun s x v => maxPool3s2BackFlat_smul c h w x v s) s x dy

theorem r34HeadCotBlk_smul (N h w : Nat) {c nCls : Nat} (Wd : Mat c nCls) (bd : Vec nCls)
    (xin : Vec (N * (c * h * w))) (dy : Vec (N * nCls)) (s : ℝ) :
    r34HeadCotBlk N h w Wd bd xin (fun i => s * dy i)
      = fun i => s * r34HeadCotBlk N h w Wd bd xin dy i :=
  HasVJP.backward_smul _ _ s dy

/-! The block cotangents, each one line from the previous link's. -/

theorem r34IdCotA_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin dy : Vec (N * (c * h * w)))
    (s : ℝ) : r34IdCotA N h w p xin (fun i => s * dy i) = fun i => s * r34IdCotA N h w p xin dy i :=
  reluMaskB_smul _ _ _ s

theorem r34IdCotC2_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin dy : Vec (N * (c * h * w)))
    (s : ℝ) :
    r34IdCotC2 N h w p xin (fun i => s * dy i) = fun i => s * r34IdCotC2 N h w p xin dy i := by
  unfold r34IdCotC2; rw [r34IdCotA_smul, bnInB_smul]

theorem r34IdCotN1_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin dy : Vec (N * (c * h * w)))
    (s : ℝ) :
    r34IdCotN1 N h w p xin (fun i => s * dy i) = fun i => s * r34IdCotN1 N h w p xin dy i := by
  unfold r34IdCotN1; rw [r34IdCotC2_smul, cInB_smul, reluMaskB_smul]

theorem r34IdCotC1_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin dy : Vec (N * (c * h * w)))
    (s : ℝ) :
    r34IdCotC1 N h w p xin (fun i => s * dy i) = fun i => s * r34IdCotC1 N h w p xin dy i := by
  unfold r34IdCotC1; rw [r34IdCotN1_smul, bnInB_smul]

theorem r34IdCotIn_smul (N h w : Nat) {c : Nat} (p : R34IdW c) (xin dy : Vec (N * (c * h * w)))
    (s : ℝ) :
    r34IdCotIn N h w p xin (fun i => s * dy i) = fun i => s * r34IdCotIn N h w p xin dy i := by
  unfold r34IdCotIn
  rw [r34IdCotC1_smul, cInB_smul, r34IdCotA_smul]
  funext i
  beta_reduce
  ring

theorem r34DownCotA_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    r34DownCotA N h w p xin (fun i => s * dy i) = fun i => s * r34DownCotA N h w p xin dy i :=
  reluMaskB_smul _ _ _ s

theorem r34DownCotC2_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    r34DownCotC2 N h w p xin (fun i => s * dy i) = fun i => s * r34DownCotC2 N h w p xin dy i := by
  unfold r34DownCotC2; rw [r34DownCotA_smul, bnInB_smul]

theorem r34DownCotN1_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    r34DownCotN1 N h w p xin (fun i => s * dy i) = fun i => s * r34DownCotN1 N h w p xin dy i := by
  unfold r34DownCotN1; rw [r34DownCotC2_smul, cInB_smul, reluMaskB_smul]

theorem r34DownCotC1_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    r34DownCotC1 N h w p xin (fun i => s * dy i) = fun i => s * r34DownCotC1 N h w p xin dy i := by
  unfold r34DownCotC1; rw [r34DownCotN1_smul, bnInB_smul]

theorem r34DownCotCp_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    r34DownCotCp N h w p xin (fun i => s * dy i) = fun i => s * r34DownCotCp N h w p xin dy i := by
  unfold r34DownCotCp; rw [r34DownCotA_smul, bnInB_smul]

theorem r34DownCotIn_smul (N h w : Nat) {ic oc : Nat} (p : R34DownW ic oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    r34DownCotIn N h w p xin (fun i => s * dy i) = fun i => s * r34DownCotIn N h w p xin dy i := by
  unfold r34DownCotIn
  rw [r34DownCotC1_smul, cStridedInB_smul, r34DownCotCp_smul, cStridedInB_smul]
  funext i
  beta_reduce
  ring

theorem r34StemCotP_smul (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    r34StemCotP N h w Ws bs εs γs βs x (fun i => s * dy i)
      = fun i => s * r34StemCotP N h w Ws bs εs γs βs x dy i :=
  mpInB_smul _ _ _ _ _ _ s

theorem r34StemCotN_smul (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    r34StemCotN N h w Ws bs εs γs βs x (fun i => s * dy i)
      = fun i => s * r34StemCotN N h w Ws bs εs γs βs x dy i := by
  unfold r34StemCotN; rw [r34StemCotP_smul, reluMaskB_smul]

theorem r34StemCotC_smul (N h w : Nat) {ic oc : Nat} (Ws : Kernel4 oc ic 7 7) (bs : Vec oc)
    (εs : ℝ) (γs βs : Vec oc) (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (dy : Vec (N * (oc * h * w))) (s : ℝ) :
    r34StemCotC N h w Ws bs εs γs βs x (fun i => s * dy i)
      = fun i => s * r34StemCotC N h w Ws bs εs γs βs x dy i := by
  unfold r34StemCotC; rw [r34StemCotN_smul, bnInB_smul]

/-! The gradient nodes. -/

theorem batchSlice_smul {N a : Nat} (X : Vec (N * a)) (s : ℝ) (n : Fin N) :
    batchSlice N a (fun i => s * X i) n = fun i => s * batchSlice N a X n i := rfl

theorem convWeightGradB_smul {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (s : ℝ)
    (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [den, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [HasVJP.backward_smul]

theorem convStridedWeightGradB_smul {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (s : ℝ) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [den, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [HasVJP.backward_smul]

theorem bnGammaGradB_smul {N oc h w : Nat} (vN epsStr cotN : String) (ε : ℝ)
    (v cot : Vec (N * (oc * (h * w)))) (s : ℝ) (k : Fin oc) :
    den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN (fun i => s * cot i))) k
      = s * den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN cot)) k := by
  simp only [den, bnPerChannel_grad_gamma, bnchwFwd, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => by ring)

theorem bnBetaGradB_smul {N oc h w : Nat} (cotN : String) (cot : Vec (N * (oc * (h * w))))
    (s : ℝ) (k : Fin oc) :
    den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN (fun i => s * cot i))) k
      = s * den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN cot)) k := by
  simp only [den, bnPerChannel_grad_beta, bnchwFwd, Finset.mul_sum]

theorem denseWeightGradB_smul {N a c : Nat} (xN cotN : String) (x : Vec (N * a))
    (cot : Vec (N * c)) (s : ℝ) (idx : Fin (a * c)) :
    den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN cot)) idx := by
  simp only [den, Mat.flatten, batchSlice_smul, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => by ring)

theorem denseBiasGradB_smul {N c : Nat} (cotN : String) (cot : Vec (N * c)) (s : ℝ) (j : Fin c) :
    den (SHlo.denseBiasGradB (N := N) (.operand cotN (fun i => s * cot i))) j
      = s * den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j := by
  simp only [den, batchSlice_smul, Finset.mul_sum]

-- ════════════════════════════════════════════════════════════════
-- § 2. Sharding — every link of the chain, on a replica, is the shard of the global link
-- ════════════════════════════════════════════════════════════════

theorem reluMaskB_shard {R N n : Nat} (PRE DY : Vec ((R * N) * n)) (r : Fin R) :
    reluMaskB (N * n) (batchShard R N n PRE r) (batchShard R N n DY r)
      = batchShard R N n (reluMaskB ((R * N) * n) PRE DY) r := rfl

theorem cInB_shard {R N : Nat} {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (DY : Vec ((R * N) * (oc * h * w))) (r : Fin R) :
    cInB N (h := h) (w := w) W b (batchShard R N (oc * h * w) DY r)
      = batchShard R N (ic * h * w) (cInB (R * N) (h := h) (w := w) W b DY) r :=
  (batchShard_batchMap _ DY r).symm

theorem cStridedInB_shard {R N : Nat} {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (b : Vec oc) (DY : Vec ((R * N) * (oc * h * w))) (r : Fin R) :
    cStridedInB N (h := h) (w := w) W b (batchShard R N (oc * h * w) DY r)
      = batchShard R N (ic * (2 * h) * (2 * w)) (cStridedInB (R * N) (h := h) (w := w) W b DY) r :=
  (batchShard_batchMap _ DY r).symm

theorem mpInB_shard {R N : Nat} (c h w : Nat) (X : Vec ((R * N) * (c * (2 * h) * (2 * w))))
    (DY : Vec ((R * N) * (c * h * w))) (r : Fin R) :
    mpInB N c h w (batchShard R N _ X r) (batchShard R N (c * h * w) DY r)
      = batchShard R N (c * (2 * h) * (2 * w)) (mpInB (R * N) c h w X DY) r :=
  (batchShard_batchMapAux _ X DY r).symm

theorem r34HeadCotBlk_shard {R N : Nat} (h w : Nat) {c nCls : Nat} (Wd : Mat c nCls)
    (bd : Vec nCls) (XIN : Vec ((R * N) * (c * h * w))) (G : Vec ((R * N) * nCls)) (r : Fin R) :
    r34HeadCotBlk N h w Wd bd (batchShard R N (c * h * w) XIN r) (batchShard R N nCls G r)
      = batchShard R N (c * h * w) (r34HeadCotBlk (R * N) h w Wd bd XIN G) r := by
  unfold r34HeadCotBlk
  rw [← r34HeadBBack_eq_vjp_backward Wd bd (batchShard R N (c * h * w) XIN r),
      ← r34HeadBBack_eq_vjp_backward Wd bd XIN]
  simp only [Function.comp_apply]
  rw [batchShard_batchMap, batchShard_batchMap]

/-- **Replica `r`'s sync-BN input cotangent**, as `bnBackSite`'s `replicas > 1` branch computes it,
    in the network layout: this replica's `[μ ‖ σ² ‖ mean(γ·dy) ‖ mean(x̂·γ·dy)]`
    (`bnSyncDyStatsB`, reading the forward's `syncStats`) all-reduced, then `bnSyncBack`. The
    replica peer of `ResNet34TieB.bnInB`, and like it written as the `den` of the emitted nodes
    over `.operand` leaves. -/
noncomputable def bnSyncInB (R : Nat) (hR : 0 < R) (N oc h w : Nat) (ε : ℝ) (γ : Vec oc)
    (xs dys : Fin R → Vec (N * (oc * h * w))) (r : Fin R) : Vec (N * (oc * h * w)) :=
  fun i => den (SHlo.bnSyncBack "" "" "" ε γ (reassocB N oc h w (xs r))
      (.operand "" (reassocB N oc h w (dys r)))
      (.allReduceMeanF R hR "" [] (fun r' => .bnSyncDyStatsB "" "" "" ε γ (reassocB N oc h w (xs r'))
        (.operand "" (reassocB N oc h w (dys r')))
        (syncStats R hR "" "" [] [] (fun r'' => .operand "" (reassocB N oc h w (xs r'')))))))
    (Fin.cast (laAssoc N oc h w) i)

/-- `reassocB` of a shard is the shard of the `reassocB`. -/
theorem reassocB_shard {R N oc h w : Nat} (X : Vec ((R * N) * (oc * h * w))) (r : Fin R) :
    reassocB N oc h w (batchShard R N (oc * h * w) X r)
      = batchShard R N (oc * (h * w)) (reassocB (R * N) oc h w X) r :=
  (batchShard_castIdx (Nat.mul_assoc oc h w) X r).symm

/-- ⭐⭐ **The sync-BN backward on replica `r` is shard `r` of the global-batch BN backward** —
    `den_bnSyncBack_allReduce` (P2 on the graph) at the network index. The right-hand side is
    `bnInB`, the single-device chain's BN link, at `N := R·N`. -/
theorem bnSyncInB_shard (R : Nat) (hR : 0 < R) (N oc h w : Nat) (hm : N * (h * w) ≠ 0)
    (hM : (R * N) * (h * w) ≠ 0) (ε : ℝ) (γ : Vec oc) (xs dys : Fin R → Vec (N * (oc * h * w)))
    (X DY : Vec ((R * N) * (oc * h * w)))
    (hxs : ∀ r, xs r = batchShard R N (oc * h * w) X r)
    (hdys : ∀ r, dys r = batchShard R N (oc * h * w) DY r) (r : Fin R) :
    bnSyncInB R hR N oc h w ε γ xs dys r
      = batchShard R N (oc * h * w) (bnInB (R * N) oc h w ε γ X DY) r := by
  have hX : ∀ r, reassocB N oc h w (xs r) = batchShard R N (oc * (h * w)) (reassocB (R * N) oc h w X) r :=
    fun r => by rw [hxs]; exact reassocB_shard X r
  have hDY : ∀ r, reassocB N oc h w (dys r)
      = batchShard R N (oc * (h * w)) (reassocB (R * N) oc h w DY) r :=
    fun r => by rw [hdys]; exact reassocB_shard DY r
  have key := den_bnSyncBack_allReduce R hR hm hM "" "" "" "" "" "" [] [] [] ε γ
    (fun r => .operand "" (reassocB N oc h w (xs r))) (fun r => reassocB N oc h w (xs r))
    (fun r => .operand "" (reassocB N oc h w (dys r)))
    (reassocB (R * N) oc h w X) (reassocB (R * N) oc h w DY) hX hX hDY r
  unfold bnSyncInB
  rw [key, bnInB_eq_den_bnBatchBack]
  exact (batchShard_castIdx (Nat.mul_assoc oc h w).symm _ r).symm

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
  bnSyncInB_shard R hR N c h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
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
  bnSyncInB_shard R hR N c h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
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
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
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
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
    _ _ _ _ _ _ (fun _ => rfl) (r34DownSyncCotN1_shard R hR N h w hN hh hw p XIN dys DY hdys) r

include hN hh hw in
theorem r34DownSyncCotCp_shard (r : Fin R) :
    r34DownSyncCotCp R hR N h w p XIN dys r
      = batchShard R N (oc * h * w) (r34DownCotCp (R * N) h w p XIN DY) r :=
  bnSyncInB_shard R hR N oc h w (nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)
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
    (nhw_ne_zero (Nat.mul_pos hR hN) (Nat.mul_pos (by norm_num) hh) (Nat.mul_pos (by norm_num) hw))
    _ _ _ _ _ _ (fun _ => rfl) (r34StemSyncCotN_shard R N h w Ws bs εs γs βs X dys DY hdys) r

end StemShard

-- ════════════════════════════════════════════════════════════════
-- § 4. The parameter collectives
-- ════════════════════════════════════════════════════════════════

/-- **P4 at the STRIDED conv weight** — `den_allReduceMeanF_convWeightGradB_shard`'s strided peer. -/
theorem den_allReduceMeanF_convStridedWeightGradB_shard {N ic oc h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec oc) (W : Kernel4 oc ic kH kW)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (DY : Vec ((R * N) * (oc * h * w)))
    (dy : Fin R → SHlo (N * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convStridedWeightGradB xN b (batchShard R N (ic * (2 * h) * (2 * w)) X r) W
            (dy r))) idx
      = (1 / (R : ℝ)) * den (.convStridedWeightGradB xN b X W (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [den, hdy]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  rw [batchSlice_batchShard, batchSlice_batchShard]

/-- **P4 at the dense weight** — the head's `Σ_n` outer product, split by replica. -/
theorem den_allReduceMeanF_denseWeightGradB_shard {N a c : Nat} (R : Nat) (hR : 0 < R)
    (t xN cotN : String) (ds : List Nat) (A : Vec ((R * N) * a)) (DY : Vec ((R * N) * c))
    (dy : Fin R → SHlo (N * c)) (hdy : ∀ r, den (dy r) = batchShard R N c DY r) (idx : Fin (a * c)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .denseWeightGradB (c := c) xN (batchShard R N a A r) (dy r))) idx
      = (1 / (R : ℝ)) * den (.denseWeightGradB (c := c) xN A (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [den, Mat.flatten, hdy]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  rw [batchSlice_batchShard, batchSlice_batchShard]

/-- **P4 at the dense bias** — `Σ_n cot`, split by replica. -/
theorem den_allReduceMeanF_denseBiasGradB_shard {N c : Nat} (R : Nat) (hR : 0 < R)
    (t cotN : String) (ds : List Nat) (DY : Vec ((R * N) * c)) (dy : Fin R → SHlo (N * c))
    (hdy : ∀ r, den (dy r) = batchShard R N c DY r) (j : Fin c) :
    den (.allReduceMeanF R hR t ds (fun r => .denseBiasGradB (N := N) (dy r))) j
      = (1 / (R : ℝ)) * den (.denseBiasGradB (N := R * N) (.operand cotN DY)) j := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [den, hdy]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  rw [batchSlice_batchShard]

-- ════════════════════════════════════════════════════════════════
-- § 5. Per-parameter DP ties — the collective IS the single-device node
-- ════════════════════════════════════════════════════════════════

/-- **One conv weight, DP-tied**: the mean over replicas of each replica's weight-gradient node
    (at its shard of the layer input and its own cotangent) IS the single-device node at the
    global batch. Tags are the render's: the collective is named for the parameter. -/
def ConvWSync (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat} (t xN cotN : String)
    (b : Vec oc) (X : Vec ((R * N) * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (cots : Fin R → Vec (N * (oc * h * w))) (COT : Vec ((R * N) * (oc * h * w))) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (.allReduceMeanF R hR t [oc, ic, kH, kW] (fun r =>
          .convWeightGradB xN b (batchShard R N (ic * h * w) X r) W (.operand cotN (cots r)))) idx
      = den (.convWeightGradB xN b X W (.operand cotN COT)) idx

/-- The strided conv weight, DP-tied. -/
def ConvStridedWSync (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat} (t xN cotN : String)
    (b : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cots : Fin R → Vec (N * (oc * h * w))) (COT : Vec ((R * N) * (oc * h * w))) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (.allReduceMeanF R hR t [oc, ic, kH, kW] (fun r =>
          .convStridedWeightGradB xN b (batchShard R N (ic * (2 * h) * (2 * w)) X r) W
            (.operand cotN (cots r)))) idx
      = den (.convStridedWeightGradB xN b X W (.operand cotN COT)) idx

/-- **One BatchNorm's γ and β, DP-tied.** Each replica's γ node is `bnSyncGammaGradB`, reading the
    forward's all-reduced `[μ ‖ σ²]` (`syncStats` over the replicas' pre-BN activations, tagged
    `{tg}mu` / `{tg}var` as `bnFwdSite` tags them), so its `x̂` is the global batch's; the β node
    reads no statistic. The right-hand sides are the single-device `bnGammaGradB` /
    `bnBetaGradB` at `N := R·N` — `BnPairTiedB`'s nodes. -/
def BnSync (R : Nat) (hR : 0 < R) (N oc h w : Nat) (tg tb vN epsStr cotN : String) (ε : ℝ)
    (V : Vec ((R * N) * (oc * h * w))) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w))) : Prop :=
  (∀ k : Fin oc,
    den (.allReduceMeanF R hR tg [oc] (fun r =>
          .bnSyncGammaGradB vN epsStr ε (reassocB N oc h w (batchShard R N (oc * h * w) V r))
            (.operand cotN (reassocB N oc h w (cots r)))
            (syncStats R hR s!"{tg}mu" s!"{tg}var" [oc] [oc]
              (fun r' => .operand vN (reassocB N oc h w (batchShard R N (oc * h * w) V r')))))) k
      = den (.bnGammaGradB vN epsStr ε (reassocB (R * N) oc h w V)
          (.operand cotN (reassocB (R * N) oc h w COT))) k)
  ∧ (∀ k : Fin oc,
    den (.allReduceMeanF R hR tb [oc] (fun r =>
          .bnBetaGradB (N := N) (oc := oc) (h := h) (w := w)
            (.operand cotN (reassocB N oc h w (cots r))))) k
      = den (.bnBetaGradB (N := R * N) (oc := oc) (h := h) (w := w)
          (.operand cotN (reassocB (R * N) oc h w COT))) k)

/-- The dense weight and bias, DP-tied. -/
def DenseSync (R : Nat) (hR : 0 < R) (N : Nat) {a c : Nat} (tW tb xN cotN : String)
    (A : Vec ((R * N) * a)) (cots : Fin R → Vec (N * c)) (COT : Vec ((R * N) * c)) : Prop :=
  (∀ idx : Fin (a * c),
    den (.allReduceMeanF R hR tW [a, c] (fun r =>
          .denseWeightGradB (c := c) xN (batchShard R N a A r) (.operand cotN (cots r)))) idx
      = den (.denseWeightGradB (c := c) xN A (.operand cotN COT)) idx)
  ∧ (∀ j : Fin c,
    den (.allReduceMeanF R hR tb [c] (fun r => .denseBiasGradB (N := N) (.operand cotN (cots r)))) j
      = den (.denseBiasGradB (N := R * N) (.operand cotN COT)) j)

/-- `(1/R)·(R·v) = v` — the collective's mean against the divisor's `R`. -/
theorem inv_mul_R (R : Nat) (hR : 0 < R) (v : ℝ) : 1 / (R : ℝ) * ((R : ℝ) * v) = v := by
  have : (R : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hR)
  field_simp

/-- ⭐ **A replica family at `R ×` the shards of `COT` gives the DP tie** — the collective's `1/R`
    (§4) cancels the `R` homogeneity (§1) carries. The same three lines for every kind below. -/
theorem convWSync_of_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (cots : Fin R → Vec (N * (oc * h * w))) (COT : Vec ((R * N) * (oc * h * w)))
    (hc : ∀ r, cots r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * COT i) r) :
    ConvWSync R hR N h w t xN cotN b X W cots COT := by
  intro idx
  rw [den_allReduceMeanF_convWeightGradB_shard R hR t xN cotN _ b W X (fun i => (R : ℝ) * COT i)
      (fun r => .operand cotN (cots r)) (fun r => hc r) idx, convWeightGradB_smul, inv_mul_R R hR]

theorem convStridedWSync_of_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w)))
    (hc : ∀ r, cots r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * COT i) r) :
    ConvStridedWSync R hR N h w t xN cotN b X W cots COT := by
  intro idx
  rw [den_allReduceMeanF_convStridedWeightGradB_shard R hR t xN cotN _ b W X
      (fun i => (R : ℝ) * COT i) (fun r => .operand cotN (cots r)) (fun r => hc r) idx,
    convStridedWeightGradB_smul, inv_mul_R R hR]

theorem bnSync_of_scaled (R : Nat) (hR : 0 < R) (N oc h w : Nat) (hm : N * (h * w) ≠ 0)
    (hM : (R * N) * (h * w) ≠ 0) (tg tb vN epsStr cotN : String) (ε : ℝ)
    (V : Vec ((R * N) * (oc * h * w))) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w)))
    (hc : ∀ r, cots r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * COT i) r) :
    BnSync R hR N oc h w tg tb vN epsStr cotN ε V cots COT := by
  have hV : ∀ r, den (SHlo.operand vN (reassocB N oc h w (batchShard R N (oc * h * w) V r)))
      = batchShard R N (oc * (h * w)) (reassocB (R * N) oc h w V) r :=
    fun r => reassocB_shard V r
  have hC : ∀ r, den (SHlo.operand cotN (reassocB N oc h w (cots r)))
      = batchShard R N (oc * (h * w)) (reassocB (R * N) oc h w (fun i => (R : ℝ) * COT i)) r :=
    fun r => by rw [den_operand, hc]; exact reassocB_shard _ r
  refine ⟨fun k => ?_, fun k => ?_⟩
  · rw [den_allReduceMeanF_bnSyncGammaGradB R hR hm hM vN epsStr s!"{tg}mu" s!"{tg}var" tg
        [oc] [oc] [oc] ε _ (fun r => reassocB N oc h w (batchShard R N (oc * h * w) V r)) _
        (reassocB (R * N) oc h w V) (reassocB (R * N) oc h w (fun i => (R : ℝ) * COT i)) hV
        (fun r => reassocB_shard V r) hC k]
    rw [show (bnPerChannel_grad_gamma oc ((R * N) * (h * w)) ε
          (bnchwFwd (R * N) oc h w (reassocB (R * N) oc h w V))
          (bnchwFwd (R * N) oc h w (reassocB (R * N) oc h w (fun i => (R : ℝ) * COT i))) k)
        = den (SHlo.bnGammaGradB vN epsStr ε (reassocB (R * N) oc h w V)
            (.operand cotN (fun i => (R : ℝ) * reassocB (R * N) oc h w COT i))) k from rfl,
      bnGammaGradB_smul, inv_mul_R R hR]
  · rw [den_allReduceMeanF_bnBetaGradB_shard R hR tb cotN [oc]
        (reassocB (R * N) oc h w (fun i => (R : ℝ) * COT i)) _ hC k]
    rw [show reassocB (R * N) oc h w (fun i => (R : ℝ) * COT i)
          = fun i => (R : ℝ) * reassocB (R * N) oc h w COT i from rfl,
      bnBetaGradB_smul, inv_mul_R R hR]

theorem denseSync_of_scaled (R : Nat) (hR : 0 < R) (N : Nat) {a c : Nat} (tW tb xN cotN : String)
    (A : Vec ((R * N) * a)) (cots : Fin R → Vec (N * c)) (COT : Vec ((R * N) * c))
    (hc : ∀ r, cots r = batchShard R N c (fun i => (R : ℝ) * COT i) r) :
    DenseSync R hR N tW tb xN cotN A cots COT := by
  refine ⟨fun idx => ?_, fun j => ?_⟩
  · rw [den_allReduceMeanF_denseWeightGradB_shard R hR tW xN cotN _ A (fun i => (R : ℝ) * COT i)
        (fun r => .operand cotN (cots r)) (fun r => hc r) idx, denseWeightGradB_smul, inv_mul_R R hR]
  · rw [den_allReduceMeanF_denseBiasGradB_shard R hR tb cotN _ (fun i => (R : ℝ) * COT i)
        (fun r => .operand cotN (cots r)) (fun r => hc r) j, denseBiasGradB_smul, inv_mul_R R hR]

-- ════════════════════════════════════════════════════════════════
-- § 6. The per-block DP ties
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
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34IdSyncCotC1_shard R hR N h w hN hh hw p XIN dys _ hdys, r34IdCotC1_smul])
  · exact bnSync_of_scaled R hR N c h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r34IdSyncCotN1_shard R hR N h w hN hh hw p XIN dys _ hdys, r34IdCotN1_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34IdSyncCotC2_shard R hR N h w hN hh hw p XIN dys _ hdys, r34IdCotC2_smul])
  · exact bnSync_of_scaled R hR N c h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
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
  have hM := nhw_ne_zero (Nat.mul_pos hR hN) hh hw
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact convStridedWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotC1_shard R hR N h w hN hh hw p XIN dys _ hdys, r34DownCotC1_smul])
  · exact bnSync_of_scaled R hR N oc h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotN1_shard R hR N h w hN hh hw p XIN dys _ hdys, r34DownCotN1_smul])
  · exact convWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotC2_shard R hR N h w hN hh hw p XIN dys _ hdys, r34DownCotC2_smul])
  · exact bnSync_of_scaled R hR N oc h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotA_shard R N h w p XIN dys _ hdys, r34DownCotA_smul])
  · exact convStridedWSync_of_scaled R hR N h w _ _ _ _ _ _ _ _ (fun r => by
      rw [r34DownSyncCotCp_shard R hR N h w hN hh hw p XIN dys _ hdys, r34DownCotCp_smul])
  · exact bnSync_of_scaled R hR N oc h w hm hM _ _ _ _ _ _ _ _ _ (fun r => by
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
  · exact bnSync_of_scaled R hR N oc (2 * h) (2 * w) (nhw_ne_zero hN h2h h2w)
      (nhw_ne_zero (Nat.mul_pos hR hN) h2h h2w) _ _ _ _ _ _ _ _ _ (fun r => by
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
-- § 7. The divisor — a replica's loss cotangent is `R ×` its shard of the global one
-- ════════════════════════════════════════════════════════════════

/-- `rowB` of a shard is the shard of the `rowB` — a relabelling inside each example. -/
theorem rowB_shard {R N K : Nat} (Z : Vec ((R * N) * K)) (r : Fin R) :
    rowB N K (batchShard R N K Z r) = batchShard R N (1 * K) (rowB (R * N) K Z) r :=
  (batchShard_castIdx (Nat.one_mul K).symm Z r).symm

theorem unrowB_shard {R N K : Nat} (Z : Vec ((R * N) * (1 * K))) (r : Fin R) :
    unrowB N K (batchShard R N (1 * K) Z r) = batchShard R N K (unrowB (R * N) K Z) r :=
  (batchShard_castIdx (Nat.one_mul K) Z r).symm

/-- ⭐ **The divisor step.** Replica `r` divides its smoothed-CE cotangent by `B` — the render's
    `divConstB` at the per-replica batch — and the single-device step at the global batch divides
    by `R·B`. At the replica's shard of the logits and targets, the replica's cotangent is `R ×`
    its shard of the global one. Nothing else about the loss differs: softmax, the label-smoothing
    shift and the target are per example. -/
theorem replicaLossCot_eq (R N nCls : Nat) (hR : 0 < R) (α B : ℝ)
    (aStr negAK bStr logN ohN : String) (Z : Vec ((R * N) * nCls)) (T : Vec ((R * N) * (1 * nCls)))
    (r : Fin R) :
    unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (batchShard R N nCls Z r)) (batchShard R N (1 * nCls) T r)))
      = batchShard R N nCls (fun i => (R : ℝ) * unrowB (R * N) nCls
          (den (smoothedLossCotGraph (R * N) nCls α ((R : ℝ) * B) aStr negAK bStr logN ohN
            (rowB (R * N) nCls Z) T)) i) r := by
  have hRr : (R : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hR)
  -- the replica's cotangent, before the row cast: the shard of `R ×` the global one
  have hden : den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (batchShard R N (1 * nCls) (rowB (R * N) nCls Z) r) (batchShard R N (1 * nCls) T r))
      = batchShard R N (1 * nCls) (fun m => (R : ℝ) * den (smoothedLossCotGraph (R * N) nCls α
          ((R : ℝ) * B) aStr negAK bStr logN ohN (rowB (R * N) nCls Z) T) m) r := by
    funext j
    rw [smoothedLossCotGraph_den, ← batchShard_batchMap]
    simp only [batchShard]
    rw [smoothedLossCotGraph_den, mul_div_assoc', mul_div_mul_left _ _ hRr]
  rw [rowB_shard, hden, unrowB_shard]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § 8. The whole-net capstone
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐⭐ **The synchronised-BN data-parallel ResNet-34 step IS the single-device step at the global
    batch.** `R` replicas at batch `N`, each dividing its loss by `B`, each running the render's
    sync-BN backward chain from its own label-smoothed cotangent; every parameter's all-reduced
    mean gradient — stem 3, thirteen identity blocks × 6, three downsample blocks × 9, dense 2:
    the 110 the render emits — equals the single-device batch-BN gradient node at batch `R·N`,
    loss divided by `R·B`, at the cotangent T3's chain delivers there.

    ⭐ The left-hand chain is the replicas' own: sync-BN backward (`bnSyncInB`, a collective per BN
    layer), per-example conv / relu / pool / head links, each replica's own loss cotangent. The
    right-hand chain is `r34_net_tiedB`'s at `N := R·N`, `B := R·B`, whose nodes that capstone ties
    to the certified gradient — so this and it together say the DP step's update is the certified
    gradient of the mean loss over all `R·N` examples.

    ⛔ Before 2026-09-21 the DP render normalised per replica and this statement was false:
    `DataParallel.dpMeanGrad_ne_globalBatchGrad` is the witness, and stays as the statement of what
    those runs did. -/
theorem r34_net_syncTiedB (R : Nat) (hR : 0 < R) (N : Nat) (hN : 0 < N) {nCls : Nat}
    (xN cotN vN epsStr : String) (aStr negAK bStr logN ohN : String) (α B : ℝ)
    (w : R34BWeights nCls)
    (X : Vec ((R * N) * (3 * (2 * (2 * 56)) * (2 * (2 * 56))))) (T : Vec ((R * N) * (1 * nCls))) :
    -- ── the single-device step at the global batch `R·N`, loss divided by `R·B` ──
    let G : Vec ((R * N) * nCls) :=
      unrowB (R * N) nCls (den (smoothedLossCotGraph (R * N) nCls α ((R : ℝ) * B) aStr negAK bStr
        logN ohN (rowB (R * N) nCls (resnet34ForwardB_full (R * N) w X)) T))
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
    -- ── replica `r`, loss divided by `B`, its own sync-BN chain ──
    let g : Fin R → Vec (N * nCls) := fun r =>
      unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (batchShard R N nCls (resnet34ForwardB_full (R * N) w X) r))
        (batchShard R N (1 * nCls) T r)))
    let eE1 : Fin R → Vec (N * (512 * 7 * 7)) := fun r =>
      r34HeadCotBlk N 7 7 w.Wd w.bd (batchShard R N _ (r34Pre16 (R * N) w X) r) (g r)
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
    ∧ r34HeadSyncTiedB R hR N 7 7 xN cotN (r34Pre16 (R * N) w X) g G := by
  intro G dyE1 dyE0 dyD4 dyC4 dyC3 dyC2 dyC1 dyC0 dyD3 dyB2 dyB1 dyB0 dyD2 dyA2 dyA1 dyA0
    g eE1 eE0 eD4 eC4 eC3 eC2 eC1 eC0 eD3 eB2 eB1 eB0 eD2 eA2 eA1 eA0 ePool
  have h56 : 0 < 56 := by norm_num
  have h28 : 0 < 28 := by norm_num
  have h14 : 0 < 14 := by norm_num
  have h7 : 0 < 7 := by norm_num
  -- the divisor: each replica's loss cotangent is `R ×` its shard of `G`
  have sG : ∀ r, g r = batchShard R N nCls (fun i => (R : ℝ) * G i) r :=
    fun r => replicaLossCot_eq R N nCls hR α B aStr negAK bStr logN ohN _ T r
  -- the scaled-shard invariant, block by block down the chain
  have sE1 : ∀ r, eE1 r = batchShard R N _ (fun i => (R : ℝ) * dyE1 i) r :=
    fun r => r34HeadCotBlk_scaled R N 7 7 w.Wd w.bd (r34Pre16 (R * N) w X) g G sG r
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
    r34_head_syncTiedB R hR N 7 7 xN cotN (r34Pre16 (R * N) w X) g G sG⟩

end Proofs.ResNet34SyncTieB
