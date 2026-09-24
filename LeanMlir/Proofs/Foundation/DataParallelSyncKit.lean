import LeanMlir.Proofs.Foundation.IndexCast
import LeanMlir.Proofs.Foundation.DataParallelSync
import LeanMlir.Proofs.Foundation.BatchedBackLinks
import LeanMlir.Proofs.Foundation.SmoothedLossCot

/-! # The sync-BN data-parallel kit — sharding, homogeneity and the collectives, per op kind

Every net's sync-BN twin (T2 forward, T3 step) says that `R` replicas at batch `N`, each running the
render's replica program over its shard, compute the single-device step at the global batch `R·N`.
The per-op facts that argument is assembled from are stated here once:

| what | names | namespace |
|---|---|---|
| the `c·h·w ↔ c·(h·w)` index cast and sharding through it; non-BN nodes commute with the batch cut; the BN sync site | `castIdx`, `laAssoc`, `batchShard_castIdx`, `den_batchOp_shard`, `den_relu_shard`, `den_addVB_shard`, `bnSyncSiteLA` | `StableHLO` |
| homogeneity — each cotangent step and gradient node is linear in its cotangent | `*_smul` | `ResNet34SyncTieB`, `MBConvSyncTieB` |
| sharding — each input-VJP is per example, so it commutes with the batch cut; sync-BN's backward is the shard of the global one | `*_shard`, `bnSyncInB_shard` | `ResNet34SyncTieB`, `MBConvSyncTieB` |
| P4 — the replica mean of a weight-gradient node is `1/R` of the global node | `den_allReduceMeanF_*_shard` | both (and `DataParallelSync` for conv W / BN β) |
| per-parameter DP ties at `R ×` the shards of a global cotangent | `*Sync`, `*Sync_of_scaled` | both |
| the divisor: a replica's loss cotangent is `R ×` its shard of the global one | `replicaLossCot_eq` | `ResNet34SyncTieB` |

The namespaces are the nets that first needed each piece; the names are cited by every twin.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.StableHLO

open scoped BigOperators

/-- **Sharding commutes with relabelling the per-example index.** The batch axis is outside the
    per-example one, so relabelling within an example and cutting the batch do not interact. -/
theorem batchShard_castIdx {R N a b : Nat} (hab : a = b) (X : Vec ((R * N) * a)) (r : Fin R) :
    batchShard R N b (fun i => X (Fin.cast (congrArg ((R * N) * ·) hab).symm i)) r
      = fun i => batchShard R N a X r (Fin.cast (congrArg (N * ·) hab).symm i) := by
  subst hab; rfl

/-- A per-example node on every replica denotes the shard of the same node on the global batch. -/
theorem den_batchOp_shard {R N a b : Nat} (op : BatchableOp a b) (e : Fin R → SHlo (N * a))
    (X : Vec ((R * N) * a)) (he : ∀ r, den (e r) = batchShard R N a X r) (r : Fin R) :
    den (.batchOp (N := N) op (e r)) = batchShard R N b (batchMap (R * N) (denOp op) X) r := by
  rw [den_batchOp, he, batchShard_batchMap]

/-- …stated for relu at the whole-batch `relu`, the form the committed forward is written in. -/
theorem den_relu_shard {R N n : Nat} (e : Fin R → SHlo (N * n)) (X : Vec ((R * N) * n))
    (he : ∀ r, den (e r) = batchShard R N n X r) (r : Fin R) :
    den (.batchOp (N := N) (.relu (n := n)) (e r)) = batchShard R N n (relu ((R * N) * n) X) r := by
  rw [den_batchOp_relu_eq_reluF, reluF_faithful, he]
  rfl

/-- The residual fan-in on every replica is the shard of the global one. -/
theorem den_addVB_shard {R N n : Nat} (a b : Fin R → SHlo (N * n)) (A B : Vec ((R * N) * n))
    (ha : ∀ r, den (a r) = batchShard R N n A r) (hb : ∀ r, den (b r) = batchShard R N n B r)
    (r : Fin R) :
    den (.addVB (a r) (b r)) = batchShard R N n (fun j => A j + B j) r := by
  rw [den_addVB, ha, hb]
  rfl

/-- **One sync-BN forward site, at the network index, on replica `r`** — `bnFwdSite`'s
    `replicas > 1` branch: `bnSyncF` of this replica's operand, reading `syncStats` over all `R`
    replicas' operands (the mean collective `t`, then Chan's variance collective `t'`), with the
    `mul_assoc` relabelling on the way in and out. -/
def bnSyncSiteLA (gN bN es t t' : String) (ds ds' : List Nat) (R : Nat) (hR : 0 < R)
    {N oc h w : Nat} (ε : ℝ) (γ β : Vec oc) (x : Fin R → SHlo (N * (oc * h * w))) (r : Fin R) :
    SHlo (N * (oc * h * w)) :=
  castIdx (laAssoc N oc h w).symm
    (.bnSyncF gN bN es ε γ β (castIdx (laAssoc N oc h w) (x r))
      (syncStats R hR t t' ds ds' (fun r' => castIdx (laAssoc N oc h w) (x r'))))

/-- ⭐⭐ **The sync-BN site on replica `r` is shard `r` of the global-batch BatchNorm.**
    `den_bnSyncF_allReduce` (P1 on the graph), carried across the `mul_assoc` seam: the right-hand
    side is `bnBatchLA` — what `bnBatchF` denotes — at `N := R·N`. -/
theorem den_bnSyncSiteLA (gN bN es t t' : String) (ds ds' : List Nat) (R : Nat) (hR : 0 < R)
    {N oc h w : Nat} (hm : N * (h * w) ≠ 0) (hM : (R * N) * (h * w) ≠ 0) (ε : ℝ) (γ β : Vec oc)
    (x : Fin R → SHlo (N * (oc * h * w))) (X : Vec ((R * N) * (oc * h * w)))
    (hx : ∀ r, den (x r) = batchShard R N (oc * h * w) X r) (r : Fin R) :
    den (bnSyncSiteLA gN bN es t t' ds ds' R hR ε γ β x r)
      = batchShard R N (oc * h * w) (bnBatchLA (R * N) oc h w ε γ β X) r := by
  have hx' : ∀ r, den (castIdx (laAssoc N oc h w) (x r))
      = batchShard R N (oc * (h * w))
          (fun i => X (Fin.cast (congrArg ((R * N) * ·) (Nat.mul_assoc oc h w)).symm i)) r := by
    intro r
    rw [den_castIdx, hx]
    exact (batchShard_castIdx (Nat.mul_assoc oc h w) X r).symm
  unfold bnSyncSiteLA
  rw [den_castIdx, den_bnSyncF_allReduce R hR hm hM gN bN es t t' ds ds' ε γ β _ _ hx' r]
  exact (batchShard_castIdx (Nat.mul_assoc oc h w).symm _ r).symm

/-- The reduction width a BatchNorm site needs nonzero, from the three positive dimensions. -/
theorem nhw_ne_zero {N h w : Nat} (hN : 0 < N) (hh : 0 < h) (hw : 0 < w) : N * (h * w) ≠ 0 :=
  Nat.pos_iff_ne_zero.mp (Nat.mul_pos hN (Nat.mul_pos hh hw))

end Proofs.StableHLO

namespace Proofs.ResNet34SyncTieB

open scoped BigOperators
open Proofs.EnetTiePoC (reassocB cInB)
open Proofs.ResNet34TieB

theorem reluMaskB_smul (n : Nat) (pre : Vec n) : IsHomog (reluMaskB n pre) := by
  intro s dy
  funext i
  unfold reluMaskB
  split_ifs <;> simp

/-- The three-term BatchNorm input-gradient is linear in `dy` — both its reductions are. -/
theorem bn_grad_input_smul (n : Nat) (ε γ : ℝ) (x : Vec n) : IsHomog (bn_grad_input n ε γ x) := by
  intro s dy
  funext i
  have h1 : ∑ j : Fin n, γ * (s * dy j) = s * ∑ j : Fin n, γ * dy j := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun _ _ => by ring)
  have h2 : ∑ j : Fin n, bnXhat n ε x j * (γ * (s * dy j))
      = s * ∑ j : Fin n, bnXhat n ε x j * (γ * dy j) := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun _ _ => by ring)
  simp only [bn_grad_input, h1, h2]
  ring

theorem bnPerChannel_grad_input_smul (oc m : Nat) (ε : ℝ) (γ : Vec oc) (x : Vec (oc * m)) :
    IsHomog (bnPerChannel_grad_input oc m ε γ x) := by
  intro s dy
  funext idx
  exact congrFun (bn_grad_input_smul m ε _ _ s (Mat.unflatten dy (finProdFinEquiv.symm idx).1)) _

theorem bnBatchTensor4_grad_input_smul (N oc h w : Nat) (ε : ℝ) (γ : Vec oc)
    (x : Vec (N * (oc * (h * w)))) : IsHomog (bnBatchTensor4_grad_input N oc h w ε γ x) := by
  intro s dy
  funext i
  exact congrFun (bnPerChannel_grad_input_smul oc (N * (h * w)) ε γ _ s (bnchwFwd N oc h w dy)) _

theorem bnInB_smul (N oc h w : Nat) (ε : ℝ) (γ : Vec oc) (x : Vec (N * (oc * h * w))) :
    IsHomog (bnInB N oc h w ε γ x) := by
  intro s dy
  rw [bnInB_eq_den_bnBatchBack, bnInB_eq_den_bnBatchBack]
  funext i
  exact congrFun (bnBatchTensor4_grad_input_smul N oc h w ε γ _ s (reassocB N oc h w dy)) _

theorem batchMap_smul {N a b : Nat} (f : Vec a → Vec b) (hf : IsHomog f) :
    IsHomog (batchMap N f) := fun s X => by
  funext idx
  exact congrFun (hf s _) _

theorem batchMapAux_smul {N t a b : Nat} (f : Vec t → Vec a → Vec b) (hf : ∀ x, IsHomog (f x))
    (aux : Vec (N * t)) : IsHomog (batchMapAux N f aux) := fun s X => by
  funext idx
  exact congrFun (hf _ s _) _

theorem cInB_smul (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    IsHomog (cInB N (h := h) (w := w) W b) :=
  batchMap_smul _ (HasVJP.backward_smul _ _)

theorem cStridedInB_smul (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    IsHomog (cStridedInB N (h := h) (w := w) W b) :=
  batchMap_smul _ (HasVJP.backward_smul _ _)

/-- The 3×3/s2 pool's `select_and_scatter` is linear in the cotangent it scatters. -/
theorem maxPool3s2BackFlat_smul (c h w : Nat) (xv : Vec (c * (2 * h) * (2 * w))) :
    IsHomog (maxPool3s2BackFlat c h w xv) := by
  intro s dyv
  funext idx
  simp only [maxPool3s2BackFlat, Tensor3.unflatten, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun _ _ => Finset.sum_congr rfl (fun _ _ =>
    Finset.sum_congr rfl (fun _ _ => by ring)))

theorem mpInB_smul (N c h w : Nat) (x : Vec (N * (c * (2 * h) * (2 * w)))) :
    IsHomog (mpInB N c h w x) :=
  batchMapAux_smul _ (maxPool3s2BackFlat_smul c h w) x

theorem batchSlice_smul {N a : Nat} (X : Vec (N * a)) (s : ℝ) (n : Fin N) :
    batchSlice N a (fun i => s * X i) n = fun i => s * batchSlice N a X n i := rfl

theorem convWeightGradB_smul {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW) (cot : Vec (N * (oc * h * w))) (s : ℝ)
    (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.convWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [denStep, denStepApp, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [HasVJP.backward_smul]

theorem convStridedWeightGradB_smul {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (s : ℝ) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.convStridedWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [denStep, denStepApp, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [HasVJP.backward_smul]

theorem bnGammaGradB_smul {N oc h w : Nat} (vN epsStr cotN : String) (ε : ℝ)
    (v cot : Vec (N * (oc * (h * w)))) (s : ℝ) (k : Fin oc) :
    den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN (fun i => s * cot i))) k
      = s * den (SHlo.bnGammaGradB vN epsStr ε v (.operand cotN cot)) k := by
  simp only [denStepApp, bnPerChannel_grad_gamma, bnchwFwd, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => by ring)

theorem bnBetaGradB_smul {N oc h w : Nat} (cotN : String) (cot : Vec (N * (oc * (h * w))))
    (s : ℝ) (k : Fin oc) :
    den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN (fun i => s * cot i))) k
      = s * den (SHlo.bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (.operand cotN cot)) k := by
  simp only [denStepApp, bnPerChannel_grad_beta, bnchwFwd, Finset.mul_sum]

theorem denseWeightGradB_smul {N a c : Nat} (xN cotN : String) (x : Vec (N * a))
    (cot : Vec (N * c)) (s : ℝ) (idx : Fin (a * c)) :
    den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.denseWeightGradB (c := c) xN x (.operand cotN cot)) idx := by
  simp only [denStep, denStepApp, Mat.flatten, batchSlice_smul, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => by ring)

theorem denseBiasGradB_smul {N c : Nat} (cotN : String) (cot : Vec (N * c)) (s : ℝ) (j : Fin c) :
    den (SHlo.denseBiasGradB (N := N) (.operand cotN (fun i => s * cot i))) j
      = s * den (SHlo.denseBiasGradB (N := N) (.operand cotN cot)) j := by
  simp only [denStep, denStepApp, batchSlice_smul, Finset.mul_sum]

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
  simp only [denStep, denStepApp, hdy]
  shard_sum

/-- **P4 at the dense weight** — the head's `Σ_n` outer product, split by replica. -/
theorem den_allReduceMeanF_denseWeightGradB_shard {N a c : Nat} (R : Nat) (hR : 0 < R)
    (t xN cotN : String) (ds : List Nat) (A : Vec ((R * N) * a)) (DY : Vec ((R * N) * c))
    (dy : Fin R → SHlo (N * c)) (hdy : ∀ r, den (dy r) = batchShard R N c DY r) (idx : Fin (a * c)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .denseWeightGradB (c := c) xN (batchShard R N a A r) (dy r))) idx
      = (1 / (R : ℝ)) * den (.denseWeightGradB (c := c) xN A (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [denStep, denStepApp, Mat.flatten, hdy]
  shard_sum

/-- **P4 at the dense bias** — `Σ_n cot`, split by replica. -/
theorem den_allReduceMeanF_denseBiasGradB_shard {N c : Nat} (R : Nat) (hR : 0 < R)
    (t cotN : String) (ds : List Nat) (DY : Vec ((R * N) * c)) (dy : Fin R → SHlo (N * c))
    (hdy : ∀ r, den (dy r) = batchShard R N c DY r) (j : Fin c) :
    den (.allReduceMeanF R hR t ds (fun r => .denseBiasGradB (N := N) (dy r))) j
      = (1 / (R : ℝ)) * den (.denseBiasGradB (N := R * N) (.operand cotN DY)) j := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [denStep, denStepApp, hdy]
  shard_sum

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
    (tg tb vN epsStr cotN : String) (ε : ℝ)
    (V : Vec ((R * N) * (oc * h * w))) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w)))
    (hc : ∀ r, cots r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * COT i) r) :
    BnSync R hR N oc h w tg tb vN epsStr cotN ε V cots COT := by
  have hM : (R * N) * (h * w) ≠ 0 := by rw [Nat.mul_assoc]; exact Nat.mul_ne_zero hR.ne' hm
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

end Proofs.ResNet34SyncTieB

namespace Proofs.MBConvSyncTieB

open scoped BigOperators
open Proofs.EnetTiePoC (dInB dStridedInB gapInB)
open Proofs.ResNet34SyncTieB

-- ════════════════════════════════════════════════════════════════
-- § 1. Homogeneity — linear in the cotangent
-- ════════════════════════════════════════════════════════════════

theorem dInB_smul (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c) :
    IsHomog (dInB N (h := h) (w := w) W b) :=
  batchMap_smul _ (HasVJP.backward_smul _ _)

theorem dStridedInB_smul (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c) :
    IsHomog (dStridedInB N (h := h) (w := w) W b) :=
  batchMap_smul _ (HasVJP.backward_smul _ _)

theorem gapInB_smul (N c h w : Nat) : IsHomog (gapInB N c h w) :=
  batchMap_smul _ (HasVJP.backward_smul _ _)

/-- The row-wise input-VJP `dX = W·dy` (the classifier's, and the SE excite dense's) is linear in
    `dy`. -/
theorem rowDenseBackFlat_smul (N a c : Nat) (W : Mat a c) : IsHomog (rowDenseBackFlat N a c W) := by
  intro s dy
  funext idx
  simp only [rowDenseBackFlat, Mat.flatten, Mat.unflatten, Mat.mulVec, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => by ring)

/-- A `HasVJP3` backward is linear in its cotangent — `HasVJP.backward_smul`'s three-axis peer,
    read off `HasVJP3.correct`. The stride-1 depthwise weight gradient is stated through one. -/
theorem hasVJP3_backward_smul {c₁ h₁ w₁ c₂ h₂ w₂ : Nat} {f : Tensor3 c₁ h₁ w₁ → Tensor3 c₂ h₂ w₂}
    (hf : HasVJP3 f) (x : Tensor3 c₁ h₁ w₁) (a : ℝ) (dy : Tensor3 c₂ h₂ w₂) :
    hf.backward x (fun i₁ i₂ i₃ => a * dy i₁ i₂ i₃)
      = fun j₁ j₂ j₃ => a * hf.backward x dy j₁ j₂ j₃ := by
  funext j₁ j₂ j₃
  rw [hf.correct, hf.correct, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun _ _ => ?_)
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun _ _ => ?_)
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => by ring)

theorem depthwiseWeightGradB_smul {N c h w kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * h * w))) (W : DepthwiseKernel c kH kW) (cot : Vec (N * (c * h * w))) (s : ℝ)
    (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.depthwiseWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [denStep, denStepApp, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [show Tensor3.unflatten (fun i => s * batchSlice N (c * h * w) cot n i)
        = fun i₁ i₂ i₃ => s * Tensor3.unflatten (batchSlice N (c * h * w) cot n) i₁ i₂ i₃ from rfl,
      hasVJP3_backward_smul]
  rfl

theorem depthwiseStridedWeightGradB_smul {N c h w kH kW : Nat} (xN cotN : String) (b : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cot : Vec (N * (c * h * w))) (s : ℝ) (idx : Fin (c * kH * kW)) :
    den (SHlo.depthwiseStridedWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.depthwiseStridedWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [denStep, denStepApp, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [HasVJP.backward_smul]

theorem convStridedXlaWeightGradB_smul {N ic oc h w kH kW : Nat} (xN cotN : String) (b : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (W : Kernel4 oc ic kH kW)
    (cot : Vec (N * (oc * h * w))) (s : ℝ) (idx : Fin (oc * ic * kH * kW)) :
    den (SHlo.convStridedXlaWeightGradB xN b x W (.operand cotN (fun i => s * cot i))) idx
      = s * den (SHlo.convStridedXlaWeightGradB xN b x W (.operand cotN cot)) idx := by
  simp only [denStep, denStepApp, batchSlice_smul, Finset.mul_sum]
  refine Finset.sum_congr rfl (fun n _ => ?_)
  rw [HasVJP.backward_smul]

-- ════════════════════════════════════════════════════════════════
-- § 2. Sharding — every per-example input-VJP commutes with the batch cut
-- ════════════════════════════════════════════════════════════════

theorem dInB_shard {R N : Nat} {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (DY : Vec ((R * N) * (c * h * w))) (r : Fin R) :
    dInB N (h := h) (w := w) W b (batchShard R N (c * h * w) DY r)
      = batchShard R N (c * h * w) (dInB (R * N) (h := h) (w := w) W b DY) r :=
  (batchShard_batchMap _ DY r).symm

theorem dStridedInB_shard {R N : Nat} {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) (DY : Vec ((R * N) * (c * h * w))) (r : Fin R) :
    dStridedInB N (h := h) (w := w) W b (batchShard R N (c * h * w) DY r)
      = batchShard R N (c * (2 * h) * (2 * w)) (dStridedInB (R * N) (h := h) (w := w) W b DY) r :=
  (batchShard_batchMap _ DY r).symm

theorem gapInB_shard {R N : Nat} (c h w : Nat) (DY : Vec ((R * N) * c)) (r : Fin R) :
    gapInB N c h w (batchShard R N c DY r) = batchShard R N (c * h * w) (gapInB (R * N) c h w DY) r :=
  (batchShard_batchMap _ DY r).symm

/-- The row-wise input-VJP is `batchMap` of `W·`, so it shards like every per-example lift. -/
theorem rowDenseBackFlat_shard {R N : Nat} (a c : Nat) (W : Mat a c) (DY : Vec ((R * N) * c))
    (r : Fin R) :
    rowDenseBackFlat N a c W (batchShard R N c DY r)
      = batchShard R N a (rowDenseBackFlat (R * N) a c W DY) r :=
  (batchShard_batchMap (Mat.mulVec W) DY r).symm

-- ════════════════════════════════════════════════════════════════
-- § 3. The collectives — `DataParallelSync`'s P4 at the three MBConv weight kinds
-- ════════════════════════════════════════════════════════════════

/-- **P4 at the depthwise weight** — each replica's `Σ_n` over its own examples, averaged, is `1/R`
    of the global batch's `Σ_n`. -/
theorem den_allReduceMeanF_depthwiseWeightGradB_shard {N c h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec c) (W : DepthwiseKernel c kH kW)
    (X DY : Vec ((R * N) * (c * h * w))) (dy : Fin R → SHlo (N * (c * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (c * h * w) DY r) (idx : Fin (c * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .depthwiseWeightGradB xN b (batchShard R N (c * h * w) X r) W (dy r))) idx
      = (1 / (R : ℝ)) * den (.depthwiseWeightGradB xN b X W (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [denStep, denStepApp, hdy]
  shard_sum

/-- **P4 at the strided depthwise weight.** -/
theorem den_allReduceMeanF_depthwiseStridedWeightGradB_shard {N c h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec c) (W : DepthwiseKernel c kH kW)
    (X : Vec ((R * N) * (c * (2 * h) * (2 * w)))) (DY : Vec ((R * N) * (c * h * w)))
    (dy : Fin R → SHlo (N * (c * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (c * h * w) DY r) (idx : Fin (c * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .depthwiseStridedWeightGradB xN b (batchShard R N (c * (2 * h) * (2 * w)) X r) W
            (dy r))) idx
      = (1 / (R : ℝ)) * den (.depthwiseStridedWeightGradB xN b X W (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [denStep, denStepApp, hdy]
  shard_sum

/-- **P4 at the XLA-`SAME` strided conv weight** (the stem) — `den_allReduceMeanF_convWeightGradB_shard`'s
    peer. Only the certificate differs from the symmetric strided one; the batch split is the same. -/
theorem den_allReduceMeanF_convStridedXlaWeightGradB_shard {N ic oc h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec oc) (W : Kernel4 oc ic kH kW)
    (X : Vec ((R * N) * (ic * (2 * h) * (2 * w)))) (DY : Vec ((R * N) * (oc * h * w)))
    (dy : Fin R → SHlo (N * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convStridedXlaWeightGradB xN b (batchShard R N (ic * (2 * h) * (2 * w)) X r) W
            (dy r))) idx
      = (1 / (R : ℝ)) * den (.convStridedXlaWeightGradB xN b X W (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [denStep, denStepApp, hdy]
  shard_sum

-- ════════════════════════════════════════════════════════════════
-- § 4. Per-parameter DP ties
-- ════════════════════════════════════════════════════════════════

/-- **A depthwise weight, DP-tied** — the collective over the `[c, 1, kH, kW]` kernel the render
    all-reduces, against the single-device node at the global batch. -/
def DepthwiseWSync (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat} (t xN cotN : String)
    (b : Vec c) (X : Vec ((R * N) * (c * h * w))) (W : DepthwiseKernel c kH kW)
    (cots : Fin R → Vec (N * (c * h * w))) (COT : Vec ((R * N) * (c * h * w))) : Prop :=
  ∀ idx : Fin (c * kH * kW),
    den (.allReduceMeanF R hR t [c, 1, kH, kW] (fun r =>
          .depthwiseWeightGradB xN b (batchShard R N (c * h * w) X r) W (.operand cotN (cots r)))) idx
      = den (.depthwiseWeightGradB xN b X W (.operand cotN COT)) idx

/-- The strided depthwise weight, DP-tied. -/
def DepthwiseStridedWSync (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat} (t xN cotN : String)
    (b : Vec c) (X : Vec ((R * N) * (c * (2 * h) * (2 * w)))) (W : DepthwiseKernel c kH kW)
    (cots : Fin R → Vec (N * (c * h * w))) (COT : Vec ((R * N) * (c * h * w))) : Prop :=
  ∀ idx : Fin (c * kH * kW),
    den (.allReduceMeanF R hR t [c, 1, kH, kW] (fun r =>
          .depthwiseStridedWeightGradB xN b (batchShard R N (c * (2 * h) * (2 * w)) X r) W
            (.operand cotN (cots r)))) idx
      = den (.depthwiseStridedWeightGradB xN b X W (.operand cotN COT)) idx

/-- The stem's XLA-`SAME` strided conv weight, DP-tied. Tags are the render's: the collective is
    named for the parameter, over its `[oc, ic, kH, kW]` shape. -/
def ConvStridedXlaWSync (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w))) : Prop :=
  ∀ idx : Fin (oc * ic * kH * kW),
    den (.allReduceMeanF R hR t [oc, ic, kH, kW] (fun r =>
          .convStridedXlaWeightGradB xN b (batchShard R N (ic * (2 * h) * (2 * w)) X r) W
            (.operand cotN (cots r)))) idx
      = den (.convStridedXlaWeightGradB xN b X W (.operand cotN COT)) idx

theorem depthwiseWSync_of_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat}
    (t xN cotN : String) (b : Vec c) (X : Vec ((R * N) * (c * h * w))) (W : DepthwiseKernel c kH kW)
    (cots : Fin R → Vec (N * (c * h * w))) (COT : Vec ((R * N) * (c * h * w)))
    (hc : ∀ r, cots r = batchShard R N (c * h * w) (fun i => (R : ℝ) * COT i) r) :
    DepthwiseWSync R hR N h w t xN cotN b X W cots COT := by
  intro idx
  rw [den_allReduceMeanF_depthwiseWeightGradB_shard R hR t xN cotN _ b W X (fun i => (R : ℝ) * COT i)
      (fun r => .operand cotN (cots r)) (fun r => hc r) idx, depthwiseWeightGradB_smul, inv_mul_R R hR]

theorem depthwiseStridedWSync_of_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {c kH kW : Nat}
    (t xN cotN : String) (b : Vec c) (X : Vec ((R * N) * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) (cots : Fin R → Vec (N * (c * h * w)))
    (COT : Vec ((R * N) * (c * h * w)))
    (hc : ∀ r, cots r = batchShard R N (c * h * w) (fun i => (R : ℝ) * COT i) r) :
    DepthwiseStridedWSync R hR N h w t xN cotN b X W cots COT := by
  intro idx
  rw [den_allReduceMeanF_depthwiseStridedWeightGradB_shard R hR t xN cotN _ b W X
      (fun i => (R : ℝ) * COT i) (fun r => .operand cotN (cots r)) (fun r => hc r) idx,
    depthwiseStridedWeightGradB_smul, inv_mul_R R hR]

theorem convStridedXlaWSync_of_scaled (R : Nat) (hR : 0 < R) (N h w : Nat) {ic oc kH kW : Nat}
    (t xN cotN : String) (b : Vec oc) (X : Vec ((R * N) * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) (cots : Fin R → Vec (N * (oc * h * w)))
    (COT : Vec ((R * N) * (oc * h * w)))
    (hc : ∀ r, cots r = batchShard R N (oc * h * w) (fun i => (R : ℝ) * COT i) r) :
    ConvStridedXlaWSync R hR N h w t xN cotN b X W cots COT := by
  intro idx
  rw [den_allReduceMeanF_convStridedXlaWeightGradB_shard R hR t xN cotN _ b W X
      (fun i => (R : ℝ) * COT i) (fun r => .operand cotN (cots r)) (fun r => hc r) idx,
    convStridedXlaWeightGradB_smul, inv_mul_R R hR]

end Proofs.MBConvSyncTieB
