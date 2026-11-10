module

public import LeanMlir.Proofs.Nets.ViT.ViTParamGrad
public import LeanMlir.Proofs.Foundation.DataParallel.Node
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # ViT-Tiny data-parallel: `R` replicas of `N` are one device at `R·N`, through the optimizer and the EMA

`ViTParamGrad` proves, at one replica, that each of ViT-Tiny's 200 gradient nodes is the
derivative of the smoothed loss of the whole net in its parameter slot
(`vit_net_lossGrad_smoothedCE`). The `vitin_*dp*` artifacts, the book's
`vitin_emadp256x4wxdropeps0000001smxerfbf16` among them, run that program on `R` replicas: each
gradient node feeds `allReduceMeanF`, the all-reduced gradient feeds the AdamW tail, and the
updated weight feeds the EMA shadow. This file composes those three with the one-replica theorem.

ViT has no BatchNorm, so no op couples examples and the composition needs no synchronised
statistic: the loss is a sum over examples, a shard's forward is the shard of the forward, and the
mean of the replicas' losses is the single-device loss at the global batch.

* `vitNetB_shard` — replica `r`'s forward at its shard of the images and of the stochastic-depth
  masks is its shard of the batch-`R·N` forward.
* `smoothedBatchLossDiv_shard` — the batch-`R·N` loss at divisor `R·B` is the mean over replicas of
  the batch-`N` losses at divisor `B`, each at its shard of the logits and targets.
* `HasGradAt.at_allReduceMeanF` — the collective, at any slot: if each replica's node is the gradient
  of that replica's loss, the all-reduced node is the gradient of the mean loss.
* `vit_dp_node_lossGrad` — the three composed, **for every parameter slot**: whenever each
  replica's gradient node is the gradient of its own shard's loss in the slot (which
  `vit_net_lossGrad_smoothedCE` proves for each of the 200 nodes), the all-reduced node is the
  gradient, in that slot, of the smoothed loss of the single-device net at the global batch
  `R·N`. `vit_dp_node_eq_global` reads it as an equality with the batch-`R·N` node.
* `vit_dp_step` — the optimizer and the shadow: the emitted AdamW triple at the all-reduced node
  is `adamWStep` at that global-batch gradient, and the EMA node over the updated weight is
  `emaNext d e θ'`, with `θ'` that step's parameter.

A slot is any `slot : Vec P → ViTTieWeights nC`, the weights with one tensor replaced by a
vector: `fun θ => { w with bcls := θ }`, `fun θ => { w with b3 := { w.b3 with Wq := Mat.unflatten θ } }`,
and so on. `ViTNetLossTiedGB`'s 200 conjuncts are exactly hypotheses `hg` of this shape, one per
slot; `vit_dp_bcls` instantiates one end to end, the classifier bias, with no hypothesis left but
`0 < ε` and targets that sum to one.

## What is not claimed

That every replica holds the same weights, that the shards are cut from one host batch and that
`replica_groups` names all `R` devices are the driver's (`DataParallel.Basic`,
`Verified.Train`) and the `*-dp-check` gates'. The learning-rate, bias-correction and decay
scalars are runtime inputs, binders here. The theorems are over ℝ.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs

open scoped BigOperators

/-- **The collective at a slot.** If each replica's node is the gradient at `θ` of that replica's
    loss `G r`, the all-reduced node is the gradient at `θ` of the mean of the `R` losses.
    `dpMeanGrad_eq_grad_meanLoss` in `HasGradAt`'s vocabulary, at the `SHlo` node. -/
theorem HasGradAt.at_allReduceMeanF {P : Nat} (R : Nat) (hR : 0 < R) (t : String) (ds : List Nat)
    (g : Fin R → SHlo P) (G : Fin R → Vec P → Vec 1) (θ : Vec P)
    (hg : ∀ r, HasGradAt (G r) θ (den (g r))) :
    HasGradAt (fun θ' => fun _ : Fin 1 => meanLoss (fun r θ'' => G r θ'' 0) θ') θ
      (den (.allReduceMeanF R hR t ds g)) := by
  have hG : ∀ r, (fun θ' : Vec P => fun _ : Fin 1 => G r θ' 0) = G r := fun r =>
    funext fun θ' => funext fun i => by rw [Fin.eq_zero i]
  have hdiff : ∀ r, LossDifferentiableAt (fun θ'' => G r θ'' 0) θ := fun r => by
    show DifferentiableAt ℝ (fun θ' : Vec P => fun _ : Fin 1 => G r θ' 0) θ
    rw [hG r]; exact (hg r).differentiableAt
  refine ⟨meanLoss_differentiableAt _ θ hdiff, fun j => ?_⟩
  have hgrad : (fun r => den (g r)) = fun r => lossGrad (fun θ'' => G r θ'' 0) θ := by
    funext r i
    show den (g r) i = pdiv (fun θ' : Vec P => fun _ : Fin 1 => G r θ' 0) θ i 0
    rw [hG r]; exact ((hg r).pdiv_eq i).symm
  rw [den_allReduceMeanF_eq_dpMean, hgrad, dpMeanGrad_eq_grad_meanLoss _ θ hdiff]
  rfl

/-- **The smoothed batch loss shards.** The loss is a sum over examples divided by a constant, so
    the batch-`R·N` loss at divisor `R·B` is the mean over the `R` replicas of the batch-`N` losses
    at divisor `B`, each at its shard of the targets and logits. -/
theorem smoothedBatchLossDiv_shard (R N K : Nat) (hR : 0 < R) (α B : ℝ)
    (t z : Vec ((R * N) * K)) :
    smoothedBatchLossDiv (R * N) K α ((R : ℝ) * B) t z
      = fun _ => (1 / (R : ℝ)) * ∑ r : Fin R,
          smoothedBatchLossDiv N K α B (batchShard R N K t r) (batchShard R N K z r) 0 := by
  have hRr : (R : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.pos_iff_ne_zero.mp hR)
  funext _
  simp only [smoothedBatchLossDiv]
  have hrow : ∀ (r : Fin R) (n : Fin N),
      logitRow N K (batchShard R N K z r) n = logitRow (R * N) K z (finProdFinEquiv (r, n)) := by
    intro r n
    funext k
    simp only [logitRow, batchShard, Equiv.symm_apply_apply]
  simp only [batchSlice_batchShard, hrow]
  rw [← Equiv.sum_comp finProdFinEquiv, Fintype.sum_prod_type, Finset.mul_sum]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun n _ => ?_
  field_simp

end Proofs

namespace Proofs.ViTTieGB

open scoped BigOperators
open Proofs.ViTTie (ViTTieWeights)

/-- Replica `r`'s shard of the stochastic-depth masks: example `n` of replica `r` reads the
    global batch's entries at `(r, n)`, at both sites of every block. `none` stays `none`. -/
def sdShard (R N : Nat) (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N))) (r : Fin R) :
    Option (Fin 12 → Vec N × Vec N) :=
  sd.map fun f k => (fun n => (f k).1 (finProdFinEquiv (r, n)),
                     fun n => (f k).2 (finProdFinEquiv (r, n)))

theorem exampleSite_vitSdA_shard (R N : Nat) (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (r : Fin R) (k : Fin 12) (n : Fin N) :
    exampleSite (vitSdA sd k) (finProdFinEquiv (r, n))
      = exampleSite (vitSdA (sdShard R N sd r) k) n := by
  cases sd <;> rfl

theorem exampleSite_vitSdM_shard (R N : Nat) (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (r : Fin R) (k : Fin 12) (n : Fin N) :
    exampleSite (vitSdM sd k) (finProdFinEquiv (r, n))
      = exampleSite (vitSdM (sdShard R N sd r) k) n := by
  cases sd <;> rfl

/-- **Replica `r`'s forward is its shard of the batch-`R·N` forward**, at its shard of the images
    and of the masks. No ViT op couples examples, so every stage commutes with the cut
    (`batchShard_batchMap`, `batchShard_batchMapIdx`). -/
theorem vitNetB_shard {gf : GeluForm} (R N : Nat) {nC : Nat} (ε : ℝ) (w : ViTTieWeights nC)
    (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (img : Vec ((R * N) * (3 * 224 * 224))) (r : Fin R) :
    batchShard R N nC (vitNetB gf (R * N) ε w sd img) r
      = vitNetB gf N ε w (sdShard R N sd r) (batchShard R N (3 * 224 * 224) img r) := by
  unfold vitNetB
  simp only [batchShard_batchMap, batchShard_batchMapIdx, exampleSite_vitSdA_shard,
    exampleSite_vitSdM_shard]

/-- Replica `r`'s loss, as a function of the weights: the smoothed loss at divisor `B` of the
    batch-`N` net on replica `r`'s shard of the images, masks and targets. -/
noncomputable def vitReplicaLoss (gf : GeluForm) (R N : Nat) {nC : Nat} (ε α B : ℝ)
    (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (img : Vec ((R * N) * (3 * 224 * 224))) (t : Vec ((R * N) * nC)) (r : Fin R)
    (w : ViTTieWeights nC) : Vec 1 :=
  smoothedBatchLossDiv N nC α B (batchShard R N nC t r)
    (vitNetB gf N ε w (sdShard R N sd r) (batchShard R N (3 * 224 * 224) img r))

/-- The single-device loss at the global batch, as a function of the weights: the smoothed loss
    at divisor `R·B` of the batch-`R·N` net. -/
noncomputable def vitGlobalLoss (gf : GeluForm) (R N : Nat) {nC : Nat} (ε α B : ℝ)
    (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (img : Vec ((R * N) * (3 * 224 * 224))) (t : Vec ((R * N) * nC))
    (w : ViTTieWeights nC) : Vec 1 :=
  smoothedBatchLossDiv (R * N) nC α ((R : ℝ) * B) t (vitNetB gf (R * N) ε w sd img)

/-- **The mean of the replicas' losses is the single-device loss at the global batch**, as
    functions of the weights: `smoothedBatchLossDiv_shard` at `vitNetB_shard`. -/
theorem vitGlobalLoss_eq_mean {gf : GeluForm} (R N : Nat) (hR : 0 < R) {nC : Nat} (ε α B : ℝ)
    (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (img : Vec ((R * N) * (3 * 224 * 224))) (t : Vec ((R * N) * nC)) (w : ViTTieWeights nC) :
    vitGlobalLoss gf R N ε α B sd img t w
      = fun _ => (1 / (R : ℝ)) * ∑ r : Fin R, vitReplicaLoss gf R N ε α B sd img t r w 0 := by
  unfold vitGlobalLoss vitReplicaLoss
  rw [smoothedBatchLossDiv_shard R N nC hR α B t]
  simp only [vitNetB_shard]

/-- **Every all-reduced ViT-Tiny gradient node is the single-device gradient at the global batch
    `R·N`.** For any parameter slot `slot` and any family `g` of per-replica gradient nodes: if
    each replica's node is the gradient, in the slot, of that replica's loss on its own shard
    (`vit_net_lossGrad_smoothedCE` at batch `N`, for each of the 200 nodes), then the node the
    render all-reduces them into is the gradient, in the slot, of the smoothed loss of the
    batch-`R·N` net at divisor `R·B`. -/
theorem vit_dp_node_lossGrad {gf : GeluForm} (R N : Nat) (hR : 0 < R) {nC : Nat} (ε α B : ℝ)
    (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (img : Vec ((R * N) * (3 * 224 * 224))) (t : Vec ((R * N) * nC))
    {P : Nat} (slot : Vec P → ViTTieWeights nC) (θ : Vec P)
    (tn : String) (ds : List Nat) (g : Fin R → SHlo P)
    (hg : ∀ r, HasGradAt (fun θ' => vitReplicaLoss gf R N ε α B sd img t r (slot θ')) θ
            (den (g r))) :
    HasGradAt (fun θ' => vitGlobalLoss gf R N ε α B sd img t (slot θ')) θ
      (den (.allReduceMeanF R hR tn ds g)) := by
  refine (HasGradAt.at_allReduceMeanF R hR tn ds g _ θ hg).congr_left ?_
  funext θ'
  rw [vitGlobalLoss_eq_mean R N hR]
  rfl

/-- **…and so equals the batch-`R·N` node.** Whatever single-device node `gG` is the gradient of
    the global-batch loss in the slot (`vit_net_lossGrad_smoothedCE` at batch `R·N`), the
    all-reduced node denotes the same vector. -/
theorem vit_dp_node_eq_global {gf : GeluForm} (R N : Nat) (hR : 0 < R) {nC : Nat} (ε α B : ℝ)
    (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (img : Vec ((R * N) * (3 * 224 * 224))) (t : Vec ((R * N) * nC))
    {P : Nat} (slot : Vec P → ViTTieWeights nC) (θ : Vec P)
    (tn : String) (ds : List Nat) (g : Fin R → SHlo P) (gG : SHlo P)
    (hg : ∀ r, HasGradAt (fun θ' => vitReplicaLoss gf R N ε α B sd img t r (slot θ')) θ
            (den (g r)))
    (hG : HasGradAt (fun θ' => vitGlobalLoss gf R N ε α B sd img t (slot θ')) θ (den gG)) :
    den (.allReduceMeanF R hR tn ds g) = den gG := by
  funext i
  rw [← (vit_dp_node_lossGrad R N hR ε α B sd img t slot θ tn ds g hg).pdiv_eq i, hG.pdiv_eq i]

/-- **One slot end to end: the classifier bias.** The all-reduced `biasGradB` node of the
    data-parallel render, each replica's at the smoothed-loss cotangent it emits on its own
    shard, is the gradient in `bcls` of the single-device loss at the global batch `R·N`. The
    hypothesis of `vit_dp_node_lossGrad` is discharged by `vit_net_lossGrad_smoothedCE` at each
    replica; the other 199 slots discharge the same way from their conjuncts. -/
theorem vit_dp_bcls {gf : GeluForm} (R N : Nat) (hR : 0 < R) {nC : Nat} (ε α B : ℝ) (hε : 0 < ε)
    (w : ViTTieWeights nC) (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (img : Vec ((R * N) * (3 * 224 * 224))) (t : Vec ((R * N) * nC))
    (ht : ∀ k, ∑ c : Fin nC, batchSlice (R * N) nC t k c = 1)
    (cotN aStr negAK bStr logN ohN tn : String) (ds : List Nat) :
    HasGradAt (fun θ' => vitGlobalLoss gf R N ε α B sd img t { w with bcls := θ' }) w.bcls
      (den (.allReduceMeanF R hR tn ds (fun r =>
        SHlo.biasGradB (N := N) (n := nC) (.operand cotN
          (den (smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN
            (vitNetB gf N ε w (sdShard R N sd r) (batchShard R N (3 * 224 * 224) img r))
            (batchShard R N nC t r))))))) := by
  refine vit_dp_node_lossGrad R N hR ε α B sd img t (fun θ' => { w with bcls := θ' }) w.bcls
    tn ds _ fun r => ?_
  have h := vit_net_lossGrad_smoothedCE (gf := gf) "%x" "%a" "eps" cotN aStr negAK bStr logN ohN
    N ε α B hε w false false (sdShard R N sd r) (batchShard R N (3 * 224 * 224) img r)
    (batchShard R N nC t r) (fun n => by rw [batchSlice_batchShard]; exact ht _)
  exact h.2.2.2.2.2.2.2.2.2.2.2.2.1.2.2.2

/-- **The data-parallel ViT-Tiny update, through the optimizer and the EMA shadow.** Under
    `vit_dp_node_lossGrad`'s hypothesis, for any slot:

    * the AdamW triple the render emits at the all-reduced node is `adamWStep` at `∇`, the
      gradient in the slot of the single-device loss at the global batch `R·N`
      (`adamW_at_allReduceMeanF`; `wd = 0` is the `wx` renders' excluded slots);
    * the EMA node over the updated weight is `emaNext d e θ'`, i.e. `d·e + (1 − d)·θ'`, with
      `θ'` that step's parameter (`emaShadow_faithful`).

    The rate, the bias corrections and the decay are runtime scalars, binders here. -/
theorem vit_dp_step {gf : GeluForm} (R N : Nat) (hR : 0 < R) {nC : Nat} (ε α B : ℝ)
    (sd : Option (Fin 12 → Vec (R * N) × Vec (R * N)))
    (img : Vec ((R * N) * (3 * 224 * 224))) (t : Vec ((R * N) * nC))
    {P : Nat} (slot : Vec P → ViTTieWeights nC) (θ : Vec P)
    (tn : String) (ds' : List Nat) (g : Fin R → SHlo P)
    (hg : ∀ r, HasGradAt (fun θ' => vitReplicaLoss gf R N ε α B sd img t r (slot θ')) θ
            (den (g r)))
    (θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN eN dN odN θN' : String) (ds : List Nat)
    (β₁ β₂ εA lr wd bc₁ bc₂ d : ℝ) (m v e : Vec P) :
    let grad : Vec P :=
      fun i => pdiv (fun θ' => vitGlobalLoss gf R N ε α B sd img t (slot θ')) θ i 0
    let θnode : SHlo P :=
      .adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds
        β₁ β₂ εA lr wd bc₁ bc₂ θ m v (.allReduceMeanF R hR tn ds' g)
    (den θnode,
     den (.adamMNextF mN b1N ob1N ds β₁ m (.allReduceMeanF R hR tn ds' g)),
     den (.adamVNextF vN b2N ob2N ds β₂ v (.allReduceMeanF R hR tn ds' g)))
      = adamWStep β₁ β₂ εA lr wd bc₁ bc₂ θ m v grad
    ∧ den (.adamMNextF eN dN odN ds d e (.operand θN' (den θnode)))
        = emaNext d e (adamWStep β₁ β₂ εA lr wd bc₁ bc₂ θ m v grad).1 := by
  intro grad θnode
  have hgrad : dpMean (fun r => den (g r)) = grad := by
    rw [← den_allReduceMeanF_eq_dpMean R hR tn ds' g]
    funext i
    exact ((vit_dp_node_lossGrad R N hR ε α B sd img t slot θ tn ds' g hg).pdiv_eq i).symm
  have hstep := adamW_at_allReduceMeanF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds
    β₁ β₂ εA lr wd bc₁ bc₂ θ m v R hR tn ds' g
  rw [hgrad] at hstep
  refine ⟨hstep, ?_⟩
  rw [emaShadow_faithful]
  exact congrArg (emaNext d e) (congrArg Prod.fst hstep)

end Proofs.ViTTieGB
