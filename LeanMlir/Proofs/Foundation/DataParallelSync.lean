import LeanMlir.Proofs.Foundation.DataParallelNode

/-! # Data parallelism, piece 3: synchronised BatchNorm — the DP step IS the global-batch step

`DataParallel.lean` (piece 1) proved that a data-parallel step on a batch-BN net is a step on
the mean of `R` per-replica losses and provably NOT the batch-`R·N` step
(`dpMeanGrad_ne_globalBatchGrad`); `DataParallelNode.lean` (piece 2) put the collective in the
AST. This is the third piece: with the sync-BN kit (`StableHLO.bnBatchVarAtB`, `bnPackB`,
`bnSyncF`, `bnSyncDyStatsB`, `bnSyncBack`, `bnSyncGammaGradB`, `bnStatsMeanB`/`VarB` —
`planning/global_bn_verified.md` §2b) the negative result reverses, and this file is what a
per-net DP twin walks its chain with.

## What is proved

* **P3 — sharding commutes with every per-example lift.** `batchShard_batchMap`,
  `batchShard_batchMapAux`, `batchShard_map` / `batchShard_zipWith`, `batchSlice_batchShard`.
  Every non-BN op in a BN net's chain is `batchMap N` (or `batchMapAux N`, or pointwise) of a
  per-example map, so replica `r`'s value at every such node is `batchShard r` of the global
  batch-`R·N` value whenever its input is. Definitional, all of them.
* ⭐⭐ **The statistics subgraph** (`syncStats`: μ all-reduced, then Chan's `σ²_r + (μ_r − μ)²`
  all-reduced, packed) denotes the global `[μ ‖ σ²]` — `den_syncStats_left` / `_right`, the
  latter by `bnVar_row_shard_chan`.
* ⭐⭐ **P1 / P2 / P2γ at the GRAPH, for any `R`** — `den_bnSyncF_allReduce`,
  `den_bnSyncBack_allReduce`, `den_allReduceMeanF_bnSyncGammaGradB`. The sync-BN subgraphs a
  DP render emits, fed by that statistics subgraph, denote
  `batchShard r` of `bnBatchTensor4` / `bnBatchTensor4_grad_input` at `N := R·N` (forward and
  input-VJP) and `1/R` of `bnPerChannel_grad_gamma` at `N := R·N` (the γ parameter gradient).
  These are the BN cases of the chain induction; `StableHLO.lean`'s `*_allReduce_R1` anchors are
  their `R := 1` instances.
* **The handed-back statistics are the global batch's own**: `den_bnStatsMeanB_allReduce` /
  `den_bnStatsVarB_allReduce` — what a sync render returns for the host's running-stat EMA is
  `bnBatchMeanB` / `bnBatchVarB` at `N := R·N`, on every replica.
* ⭐⭐ **P4 — the parameter collective is the global-batch gradient.**
  `den_allReduceMeanF_convWeightGradB_shard`, `den_allReduceMeanF_bnBetaGradB_shard` and the
  γ statement above: the all-reduced mean of the `R` per-replica gradient nodes, each on its
  shard and at the shard-`r` block of the global cotangent, is `(1/R)·` the batch-`R·N`
  gradient node at the SAME per-example cotangents. Every other `*GradB` composes identically
  (`simp only [den]`, the shard hypothesis, `sum_finProdFinEquiv`). The ℝ-level twin is
  `DataParallel.dpSyncGrad_eq_globalBatchGrad`, the positive counterpart of
  `dpMeanGrad_ne_globalBatchGrad`.

## The `1/R`, and where it goes

A DP render divides its loss cotangent by the PER-REPLICA batch (`divConstB N`); the
single-device batch-`R·N` step it is compared to divides by `R·N`. So at a common per-example
cotangent the replica mean is `1/R` of the global sum, and the two divisors differ by exactly
that `R`. A per-net twin closes it with ONE linearity step — the whole-net backward is a
`HasVJP.backward`, and `HasVJP.backward_smul` says scaling the cotangent scales the gradient —
instead of threading a factor through every op of the chain.

## What is NOT claimed

⚠ Nothing here is a whole-net statement. The chain induction is per net (§3.2 onward of the
plan): this file supplies the BN case and the commutation lemmas the non-BN cases reduce to,
and each net's DP twin walks its own chain with them. ⚠ That the `R` graphs' host inputs ARE
the replica shards of one batch — `hx` / `hxv` / `hdy` below — remains the driver's, exactly as
in piece 2. ⚠ The lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § P3 — sharding commutes with the per-example lifts
-- ════════════════════════════════════════════════════════════════

/-- Example `n` of shard `r` is example `(r, n)` of the global batch. -/
theorem batchSlice_batchShard {R N a : Nat} (X : Vec ((R * N) * a)) (r : Fin R) (n : Fin N) :
    batchSlice N a (batchShard R N a X r) n
      = batchSlice (R * N) a X (finProdFinEquiv (r, n)) := by
  funext i
  simp only [batchSlice, batchShard, Equiv.symm_apply_apply]

/-- ⭐ **`batchMap` commutes with sharding** — the block analogue of `batchSlice_batchMap`. A
    per-example lift applied to the global batch, restricted to replica `r`, is the same lift
    applied to replica `r`'s shard: so every conv, relu, pool, GAP and dense node in a DP render
    denotes `batchShard r` of its batch-`R·N` value as soon as its input does. -/
theorem batchShard_batchMap {R N a b : Nat} (f : Vec a → Vec b) (X : Vec ((R * N) * a))
    (r : Fin R) :
    batchShard R N b (batchMap (R * N) f X) r = batchMap N f (batchShard R N a X r) := by
  funext idx
  simp only [batchShard, batchMap, Equiv.symm_apply_apply]

/-- …and so does `batchMapAux`, the shape of every batched backward that recomputes from a saved
    per-example activation: example `n` is handed ITS slice of `aux`, which sharding respects. -/
theorem batchShard_batchMapAux {R N s a b : Nat} (f : Vec s → Vec a → Vec b)
    (aux : Vec ((R * N) * s)) (X : Vec ((R * N) * a)) (r : Fin R) :
    batchShard R N b (batchMapAux (R * N) f aux X) r
      = batchMapAux N f (batchShard R N s aux r) (batchShard R N a X r) := by
  funext idx
  simp only [batchShard, batchMapAux, Equiv.symm_apply_apply, batchSlice_batchShard]

/-- A pointwise map commutes with sharding (`scaleB`, `shiftB`, `divConstB`, …). -/
theorem batchShard_map {R N a : Nat} (φ : ℝ → ℝ) (X : Vec ((R * N) * a)) (r : Fin R) :
    batchShard R N a (fun i => φ (X i)) r = fun i => φ (batchShard R N a X r i) := rfl

/-- A pointwise binary map commutes with sharding (`addVB`, `subB`, the relu mask `selectPosB`
    against its saved activation, …). -/
theorem batchShard_zipWith {R N a : Nat} (φ : ℝ → ℝ → ℝ) (X Y : Vec ((R * N) * a))
    (r : Fin R) :
    batchShard R N a (fun i => φ (X i) (Y i)) r
      = fun i => φ (batchShard R N a X r i) (batchShard R N a Y r i) := rfl

/-- The replica mean of a replica-independent value is that value. `bnSyncDyStatsB` passes the
    already-global `[μ ‖ m2]` through the second collective, and this is why that costs
    nothing. -/
theorem dpMean_const_mul {R : Nat} (hR : (R : ℝ) ≠ 0) (K : ℝ) :
    (1 / (R : ℝ)) * ∑ _r : Fin R, K = K := by
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

-- ════════════════════════════════════════════════════════════════
-- § The statistics subgraph, and P1 / P2 at the graph for any R — the BN case of the chain
-- ════════════════════════════════════════════════════════════════

/-- **The sync-BN statistics subgraph a render emits**: the replicas' means all-reduced, then
    their Chan-corrected variances at that mean all-reduced, packed as `[μ ‖ σ²]`. The same
    expression on every replica — the collective is what makes it replica-independent. -/
def syncStats {N oc h w : Nat} (R : Nat) (hR : 0 < R) (t t' : String) (ds ds' : List Nat)
    (x : Fin R → SHlo (N * (oc * (h * w)))) : SHlo (oc + oc) :=
  .bnPackB (.allReduceMeanF R hR t ds (fun r => .bnBatchMeanB (x r)))
    (.allReduceMeanF R hR t' ds' (fun r => .bnBatchVarAtB (x r)
      (.allReduceMeanF R hR t ds (fun r' => .bnBatchMeanB (x r')))))

/-- ⭐⭐ **The first collective IS the global mean** — `bnMean_shard` on the channel rows. -/
theorem den_syncStats_left {N oc h w : Nat} (R : Nat) (hR : 0 < R) (hm : N * (h * w) ≠ 0)
    (t t' : String) (ds ds' : List Nat) (x : Fin R → SHlo (N * (oc * (h * w))))
    (X : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r) (c : Fin oc) :
    den (syncStats R hR t t' ds ds' x) (Fin.castAdd oc c)
      = bnMean ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c) := by
  simp only [syncStats, den_bnPackB, Fin.append_left, den_allReduceMeanF, den_bnBatchMeanB, hx]
  exact (bnMean_row_shard R N oc h w (Nat.pos_iff_ne_zero.mp hR) hm X c).symm

/-- ⭐⭐ **The second collective IS the global variance** — Chan's parallel variance
    (`bnVar_row_shard_chan`): each replica's two-pass `σ²_r` plus `(μ_r − μ)²`, averaged. This
    is the lemma the one-round `E[x²]` exchange could not have, and the reason it was replaced. -/
theorem den_syncStats_right {N oc h w : Nat} (R : Nat) (hR : 0 < R) (hm : N * (h * w) ≠ 0)
    (t t' : String) (ds ds' : List Nat) (x : Fin R → SHlo (N * (oc * (h * w))))
    (X : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r) (c : Fin oc) :
    den (syncStats R hR t t' ds ds' x) (Fin.natAdd oc c)
      = bnVar ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c) := by
  have hRne : R ≠ 0 := Nat.pos_iff_ne_zero.mp hR
  simp only [syncStats, den_bnPackB, Fin.append_right, den_allReduceMeanF, den_bnBatchVarAtB,
             den_bnBatchMeanB, hx]
  rw [← bnMean_row_shard R N oc h w hRne hm X c]
  exact (bnVar_row_shard_chan R N oc h w hRne hm X c).symm

/-- `σ² + μ²` at the global statistics is the global second moment — how a consumer's `den`,
    stated at `(μ, m2)`, reads the packed `[μ ‖ σ²]`. -/
theorem global_var_add_sq {N oc h w : Nat} (R : Nat) (hM : (R * N) * (h * w) ≠ 0)
    (X : Vec ((R * N) * (oc * (h * w)))) (c : Fin oc) :
    bnVar ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c)
      + bnMean ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c)
        * bnMean ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c)
      = bnMeanSq ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c) := by
  rw [bnVar_eq_bnMeanSq_sub_sq _ hM]; ring

/-- ⭐⭐ **P1 on the graph, any `R`.** Replica `r`'s `bnSyncF`, fed by the two-round statistics
    subgraph over the `R` replicas' inputs, denotes `batchShard r` of the batch-`R·N`
    `bnBatchTensor4` — given that each replica's operand denotes its shard (`hx`, the chain's
    induction hypothesis). `den_bnSyncF_allReduce_R1` is this at `R := 1`. -/
theorem den_bnSyncF_allReduce {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (hm : N * (h * w) ≠ 0) (hM : (R * N) * (h * w) ≠ 0) (gN bN es t t' : String)
    (ds ds' : List Nat) (ε : ℝ) (γ β : Vec oc) (x : Fin R → SHlo (N * (oc * (h * w))))
    (X : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r) (r : Fin R) :
    den (.bnSyncF gN bN es ε γ β (x r) (syncStats R hR t t' ds ds' x))
      = batchShard R N (oc * (h * w)) (bnBatchTensor4 (R * N) oc h w ε γ β X) r := by
  rw [den_bnSyncF]
  simp only [den_syncStats_left R hR hm t t' ds ds' x X hx,
             den_syncStats_right R hR hm t t' ds ds' x X hx, global_var_add_sq R hM, hx]
  rw [← bnSyncTensor4_batchShard]
  exact congrArg (fun z => batchShard R N (oc * (h * w)) z r)
    (bnSyncTensor4_at_own_stats (R * N) oc h w hM ε γ β X)

/-- ⭐⭐ **P2 on the graph, any `R`.** Replica `r`'s `bnSyncBack`, fed by the collective over the
    replicas' `bnSyncDyStatsB` (each reading the packed forward statistics), denotes
    `batchShard r` of the batch-`R·N` `bnBatchTensor4_grad_input` — given that each replica's
    saved activation and its incoming cotangent are its shards of the global ones (`hx` / `hxv`,
    `hdy`). `den_bnSyncBack_allReduce_R1` is this at `R := 1`. -/
theorem den_bnSyncBack_allReduce {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (hm : N * (h * w) ≠ 0) (hM : (R * N) * (h * w) ≠ 0) (gN xN es t t' t'' : String)
    (ds ds' ds'' : List Nat) (ε : ℝ) (γ : Vec oc)
    (x : Fin R → SHlo (N * (oc * (h * w)))) (xv : Fin R → Vec (N * (oc * (h * w))))
    (dy : Fin R → SHlo (N * (oc * (h * w)))) (X DY : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r)
    (hxv : ∀ r, xv r = batchShard R N (oc * (h * w)) X r)
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * (h * w)) DY r) (r : Fin R) :
    den (.bnSyncBack gN xN es ε γ (xv r) (dy r)
          (.allReduceMeanF R hR t'' ds'' (fun r' => .bnSyncDyStatsB gN xN es ε γ (xv r') (dy r')
            (syncStats R hR t t' ds ds' x))))
      = batchShard R N (oc * (h * w)) (bnBatchTensor4_grad_input (R * N) oc h w ε γ X DY) r := by
  have hRne : R ≠ 0 := Nat.pos_iff_ne_zero.mp hR
  have hRr : (R : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hRne
  rw [den_bnSyncBack, hxv r, hdy r]
  -- the outer collective is [μ ‖ σ² ‖ mdy ‖ mdyx]; μ and σ² pass through (`dpMean_const_mul`)
  simp only [den_allReduceMeanF, den_bnSyncDyStatsB, Fin.append_left, Fin.append_right,
             den_syncStats_left R hR hm t t' ds ds' x X hx,
             den_syncStats_right R hR hm t t' ds ds' x X hx, global_var_add_sq R hM,
             hxv, hdy, dpMean_const_mul hRr]
  exact bnSyncTensor4_grad_input_shard_eq_global R N oc h w hRne hm hM ε γ X DY r
    _ _ _ _ (by funext c; exact bnMean_row_shard R N oc h w hRne hm X c)
    (by funext c; exact bnMeanSq_row_shard R N oc h w hRne hm X c) rfl rfl

-- ════════════════════════════════════════════════════════════════
-- § P4 — the parameter collective is 1/R of the global-batch gradient node
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **P2γ on the graph, any `R`, already all-reduced.** The parameter collective over the
    replicas' `bnSyncGammaGradB` — each at its shard, its shard's cotangent and the packed global
    statistics — is `1/R` of `bnPerChannel_grad_gamma` at `N := R·N`: the committed γ gradient
    at the global batch. The BN γ node is the one parameter gradient sync-BN changes, because it
    is the one that reads `x̂`. -/
theorem den_allReduceMeanF_bnSyncGammaGradB {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (hm : N * (h * w) ≠ 0) (hM : (R * N) * (h * w) ≠ 0) (xN es t t' t'' : String)
    (ds ds' ds'' : List Nat) (ε : ℝ)
    (x : Fin R → SHlo (N * (oc * (h * w)))) (xv : Fin R → Vec (N * (oc * (h * w))))
    (dy : Fin R → SHlo (N * (oc * (h * w)))) (X DY : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r)
    (hxv : ∀ r, xv r = batchShard R N (oc * (h * w)) X r)
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * (h * w)) DY r) (c : Fin oc) :
    den (.allReduceMeanF R hR t'' ds'' (fun r => .bnSyncGammaGradB xN es ε (xv r) (dy r)
          (syncStats R hR t t' ds ds' x))) c
      = (1 / (R : ℝ)) * bnPerChannel_grad_gamma oc ((R * N) * (h * w)) ε
          (bnchwFwd (R * N) oc h w X) (bnchwFwd (R * N) oc h w DY) c := by
  simp only [den_allReduceMeanF, den_bnSyncGammaGradB,
             den_syncStats_left R hR hm t t' ds ds' x X hx,
             den_syncStats_right R hR hm t t' ds ds' x X hx, global_var_add_sq R hM, hxv, hdy]
  -- the shard sums are the global row sum, and at the global statistics that is the γ gradient
  rw [← bnSyncPerChannel_grad_gamma_row_shard,
      congrFun (bnSyncPerChannel_grad_gamma_at_own_stats oc ((R*N)*(h*w)) hM ε
        (bnchwFwd (R*N) oc h w X) (bnchwFwd (R*N) oc h w DY)) c]

/-- **The handed-back running MEAN under sync-BN is the global batch's own** — `bnStatsMeanB`
    on the packed statistics denotes what `bnBatchMeanB` at `N := R·N` denotes, so the host
    EMAs the global statistic on every replica. -/
theorem den_bnStatsMeanB_allReduce {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (hm : N * (h * w) ≠ 0) (t t' : String) (ds ds' : List Nat)
    (x : Fin R → SHlo (N * (oc * (h * w)))) (X : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r) :
    den (.bnStatsMeanB (syncStats R hR t t' ds ds' x))
      = fun c => bnMean ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c) := by
  funext c
  simp only [den_bnStatsMeanB, den_syncStats_left R hR hm t t' ds ds' x X hx]

/-- **…and so is the handed-back VARIANCE** — the global batch's `bnVar`, what `bnBatchVarB` at
    `N := R·N` denotes. ⛔ No per-replica variance is ever averaged on its own: the between-shard
    spread `(μ_r − μ)²` rides along, which is what Chan's formula is. -/
theorem den_bnStatsVarB_allReduce {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (hm : N * (h * w) ≠ 0) (t t' : String) (ds ds' : List Nat)
    (x : Fin R → SHlo (N * (oc * (h * w)))) (X : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r) :
    den (.bnStatsVarB (syncStats R hR t t' ds ds' x))
      = fun c => bnVar ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c) := by
  funext c
  simp only [den_bnStatsVarB, den_syncStats_right R hR hm t t' ds ds' x X hx]

/-- ⭐⭐ **P4 for the `Σ_n`-shaped gradients, at the conv weight.** The collective over the
    replicas' `convWeightGradB`, each on its shard at the shard-`r` block of the global
    cotangent, is `1/R` of the batch-`R·N` node at that cotangent. `den_allReduceMeanF_convWeightGradB`
    (piece 2) is the same collective with NO relation between the replicas' inputs; this is what
    it becomes once P1–P3 relate them. Every other `Σ_n` gradient (`denseWeightGradB`,
    `denseBiasGradB`, the strided / depthwise / bias kinds) composes by the same three lines. -/
theorem den_allReduceMeanF_convWeightGradB_shard {N ic oc h w kH kW : Nat} (R : Nat)
    (hR : 0 < R) (t xN cotN : String) (ds : List Nat) (b : Vec oc) (W : Kernel4 oc ic kH kW)
    (X : Vec ((R * N) * (ic * h * w))) (DY : Vec ((R * N) * (oc * h * w)))
    (dy : Fin R → SHlo (N * (oc * h * w)))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * h * w) DY r) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convWeightGradB xN b (batchShard R N (ic * h * w) X r) W (dy r))) idx
      = (1 / (R : ℝ)) * den (.convWeightGradB xN b X W (.operand cotN DY)) idx := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [den, hdy]
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro n _
  -- ⚠ `rw`, not `simp`: the `x` slice sits inside `conv2d_weight_grad_has_vjp b x`, whose TYPE
  -- depends on it, and simp has no congruence through a dependent argument.
  rw [batchSlice_batchShard, batchSlice_batchShard]

/-- **P4 at the BN β gradient**: `Σ dy` over each shard's channel row, averaged, is `1/R` of the
    global row's `Σ dy`. β reads no statistic, so this is the row split alone. -/
theorem den_allReduceMeanF_bnBetaGradB_shard {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (t cotN : String) (ds : List Nat) (DY : Vec ((R * N) * (oc * (h * w))))
    (dy : Fin R → SHlo (N * (oc * (h * w))))
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * (h * w)) DY r) (c : Fin oc) :
    den (.allReduceMeanF R hR t ds
          (fun r => .bnBetaGradB (N := N) (oc := oc) (h := h) (w := w) (dy r))) c
      = (1 / (R : ℝ)) * den (.bnBetaGradB (N := R * N) (oc := oc) (h := h) (w := w)
          (.operand cotN DY)) c := by
  simp only [den_allReduceMeanF]
  congr 1
  simp only [den, hdy]
  exact (bnPerChannel_grad_beta_row_shard R N oc h w DY c).symm

-- ════════════════════════════════════════════════════════════════
-- § The divisor step
-- ════════════════════════════════════════════════════════════════

/-- **A VJP backward is linear in its cotangent** — read off `HasVJP.correct`. This is the one
    step that reconciles a DP render's `divConstB N` with the batch-`R·N` step's `divConstB (R·N)`:
    the per-replica cotangent is `R ·` the shard of the global one, so every per-replica
    gradient is `R ·` its shard contribution, and the collective's `1/R` cancels it. -/
theorem HasVJP.backward_smul {m n : Nat} {f : Vec m → Vec n} (hf : HasVJP f) (x : Vec m)
    (a : ℝ) (dy : Vec n) :
    hf.backward x (fun j => a * dy j) = fun i => a * hf.backward x dy i := by
  funext i
  rw [hf.correct, hf.correct, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun j _ => by ring)

end Proofs
