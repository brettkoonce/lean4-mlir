import LeanMlir.Proofs.Codegen.StableHLO

/-! # Data parallelism with synchronised BatchNorm — the DP step IS the global-batch step

`DataParallel.lean` proves that a data-parallel step on a per-replica-BN net is a step on
the mean of `R` per-replica losses and not the batch-`R·N` step
(`dpMeanGrad_ne_globalBatchGrad`); `DataParallelNode.lean` puts the collective in the
AST. With the sync-BN ops (`StableHLO.bnBatchVarAtB`, `bnPackB`, `bnSyncF`, `bnSyncDyStatsB`,
`bnSyncBack`, `bnSyncGammaGradB`, `bnStatsMeanB`/`VarB`) the negative result reverses, and this
file is what a per-net DP twin walks its chain with.

## What is proved

* **P3 — sharding commutes with every per-example lift.** `batchShard_batchMap`,
  `batchShard_batchMapAux`, `batchShard_map` / `batchShard_zipWith`, `batchSlice_batchShard`.
  Every non-BN op in a BN net's chain is `batchMap N` (or `batchMapAux N`, or pointwise) of a
  per-example map, so replica `r`'s value at every such node is `batchShard r` of the global
  batch-`R·N` value whenever its input is. Definitional, all of them.
* **The statistics subgraph** (`syncStats`: μ all-reduced, then Chan's `σ²_r + (μ_r − μ)²`
  all-reduced, packed) denotes the global `[μ ‖ σ²]` — `den_syncStats_left` / `_right`, the
  latter by `bnVar_row_shard_chan`.
* **The sync-BN forward (P1), input-VJP (P2) and γ gradient (P2γ) at the graph, for any
  `R`** — `den_bnSyncF_allReduce`,
  `den_bnSyncBack_allReduce`, `den_allReduceMeanF_bnSyncGammaGradB`. The sync-BN subgraphs a
  DP render emits, fed by that statistics subgraph, denote
  `batchShard r` of `bnBatchTensor4` / `bnBatchTensor4GradInput` at `N := R·N` (forward and
  input-VJP) and `1/R` of `bnPerChannelGradGamma` at `N := R·N` (the γ parameter gradient).
  These are the BN cases of the chain induction; `StableHLO.lean`'s `*_allReduce_R1` anchors are
  their `R := 1` instances.
* **The handed-back statistics are the global batch's own**: `den_bnStatsMeanB_allReduce` /
  `den_bnStatsVarB_allReduce` — what a sync render returns for the host's running-stat EMA is
  `bnBatchMeanB` / `bnBatchVarB` at `N := R·N`, on every replica.
* **P4 — the parameter collective is the global-batch gradient.**
  `den_allReduceMeanF_convWeightGradB_shard`, `den_allReduceMeanF_bnBetaGradB_shard` and the
  γ statement above: the all-reduced mean of the `R` per-replica gradient nodes, each on its
  shard and at the shard-`r` block of the global cotangent, is `(1/R)·` the batch-`R·N`
  gradient node at the SAME per-example cotangents. Every other `*GradB` composes identically
  (`simp only [denStep, denStepApp]`, the shard hypothesis, `sum_finProdFinEquiv`). The ℝ-level twin is
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

Nothing here is a whole-net statement. The chain induction is per net: this file supplies the
BN case and the commutation lemmas the non-BN cases reduce to, and each net's DP twin walks its
own chain with them. That the `R` graphs' host inputs ARE the replica shards of one batch —
`hx` / `hxv` / `hdy` below — remains the driver's, exactly as in `DataParallelNode.lean`. The
lowerer's `all_reduce` is trusted as every other op's lowering is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Sharding the batch-BN [N,C,H,W] layout: `batchShard`, the per-shard statistics, and the
--   shard = global identities for the sync-BN forward, its γ/β gradients and its input-VJP
--   (`PerChannelBN` holds the ops themselves, which `StableHLO`'s `den` reads)
-- ════════════════════════════════════════════════════════════════

/-- The channel a `[N,C,H,W]` flat index belongs to. -/
noncomputable def bnchwChan (N oc h w : Nat) (t : Fin (N * (oc * (h * w)))) : Fin oc :=
  (finProdFinEquiv.symm (finProdFinEquiv.symm t).2).1

/-- **The sync forward is POINTWISE once its statistics are fixed.**

    Cell `t` becomes `γ_c·(x_t − μ_c)·(m2_c − μ_c² + ε)^(−1/2) + β_c` with `c` its own channel —
    no other cell is read. Cells mix ONLY when the statistics are computed, and under sync-BN
    that computation has been hoisted out into the collective.

    This is what makes the shard argument (`bnSyncTensor4_batchShard`) pure index bookkeeping:
    a pointwise map commutes with any reindexing that preserves the channel, and sharding the
    batch does. -/
theorem bnSyncTensor4_apply (N oc h w : Nat) (ε : ℝ) (γ β μ m2 : Vec oc)
    (x : Vec (N * (oc * (h * w)))) (t : Fin (N * (oc * (h * w)))) :
    bnSyncTensor4 N oc h w ε γ β μ m2 x t
      = γ (bnchwChan N oc h w t)
          * ((x t - μ (bnchwChan N oc h w t))
             * (1 / Real.sqrt (m2 (bnchwChan N oc h w t)
                  - μ (bnchwChan N oc h w t) * μ (bnchwChan N oc h w t) + ε)))
        + β (bnchwChan N oc h w t) := by
  unfold bnSyncTensor4
  simp only [Function.comp_apply, bnchwBack]
  rw [bnPerChannelEvalFlat_apply]
  have hz : bnchwFwd N oc h w x (bnchwBackIdx N oc h w t) = x t := by
    rw [bnchwFwd_apply, bnchwFwdIdx_bnchwBackIdx]
  have hc : (finProdFinEquiv.symm (bnchwBackIdx N oc h w t)).1 = bnchwChan N oc h w t := by
    unfold bnchwBackIdx bnchwChan
    simp only [Equiv.symm_apply_apply]
  rw [hz, hc]

/-- Shard `r`'s block of a global batch laid out row-major `[R·N, a]`: example `(r, n)` of the
    global batch is example `n` of shard `r`. The contiguous cut the DP shim makes. -/
noncomputable def batchShard (R N a : Nat) (X : Vec ((R * N) * a)) (r : Fin R) : Vec (N * a) :=
  fun idx => X (finProdFinEquiv
    (finProdFinEquiv (r, (finProdFinEquiv.symm idx).1), (finProdFinEquiv.symm idx).2))

/-- **Sharding the batch does not move a cell's CHANNEL.** The one fact P1 needs about the
    layout: the batch axis is outside the channel axis in `[N,C,H,W]`, so cutting the batch
    leaves every cell in the channel it was already in. -/
theorem bnchwChan_batchShard (R N oc h w : Nat) (r : Fin R)
    (idx : Fin (N * (oc * (h * w)))) :
    bnchwChan (R*N) oc h w (finProdFinEquiv
        (finProdFinEquiv (r, (finProdFinEquiv.symm idx).1), (finProdFinEquiv.symm idx).2))
      = bnchwChan N oc h w idx := by
  unfold bnchwChan
  simp only [Equiv.symm_apply_apply]

/-- The shard of a channel's ROW. Channel `c` of a global `[R·N,C,H,W]` batch is a
    `(R·N)·h·w`-wide row; this is the equiv exhibiting it as `R` blocks of `N·h·w`, one per
    replica — `(r, (n,s)) ↦ ((r,n), s)`. The row-level counterpart of `batchShard`. -/
noncomputable def bnShardEquiv (R N hw : Nat) : Fin R × Fin (N * hw) ≃ Fin ((R * N) * hw) :=
  (Equiv.prodCongr (Equiv.refl (Fin R)) finProdFinEquiv.symm).trans
    (((Equiv.prodAssoc (Fin R) (Fin N) (Fin hw)).symm).trans
      ((Equiv.prodCongr finProdFinEquiv (Equiv.refl (Fin hw))).trans finProdFinEquiv))

/-- **The global channel row, restricted to replica `r`'s block, IS that replica's own
    channel row.** The layout fact that connects a batch shard to the `[C, N·H·W]` world
    `bnPerChannelFlat` reduces in — i.e. the one step the `bnchwFwd` relabel was hiding. -/
theorem bnchwFwd_row_batchShard (R N oc h w : Nat) (X : Vec ((R * N) * (oc * (h * w))))
    (c : Fin oc) (r : Fin R) (k : Fin (N * (h * w))) :
    Mat.unflatten (bnchwFwd (R*N) oc h w X) c (bnShardEquiv R N (h*w) (r, k))
      = Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r)) c k := by
  unfold Mat.unflatten bnchwFwd bnchwFwdIdx batchShard bnShardEquiv
  simp only [Equiv.trans_apply, Equiv.prodCongr_apply, Equiv.coe_refl, Prod.map_apply, id_eq,
             Equiv.prodAssoc_symm_apply, Equiv.symm_apply_apply]

/-- **P1a — the sync forward COMMUTES WITH SHARDING, for any statistics at all.**

    `shard_r ∘ (sync BN at μ, m2) = (sync BN at μ, m2) ∘ shard_r`. No hypothesis on `μ`/`m2`:
    once the statistics are fixed the map is pointwise (`bnSyncTensor4_apply`) and sharding
    preserves each cell's channel (`bnchwChan_batchShard`), so there is nothing to prove about
    BatchNorm here — only about indices.

    This is the half of P1 that carries no mathematics, and separating it is what leaves the
    real content in one place: whether the handed-in statistics ARE the global ones, which is
    `bnMean_shard`/`bnMeanSq_shard`, i.e. exactly what `allReduceMeanF` computes. -/
theorem bnSyncTensor4_batchShard (R N oc h w : Nat) (ε : ℝ) (γ β μ m2 : Vec oc)
    (X : Vec ((R * N) * (oc * (h * w)))) (r : Fin R) :
    batchShard R N (oc * (h * w)) (bnSyncTensor4 (R*N) oc h w ε γ β μ m2 X) r
      = bnSyncTensor4 N oc h w ε γ β μ m2 (batchShard R N (oc * (h * w)) X r) := by
  funext idx
  unfold batchShard
  rw [bnSyncTensor4_apply, bnSyncTensor4_apply, bnchwChan_batchShard]

/-- **P1b — the GLOBAL per-channel mean is the mean of the replicas' per-channel means.**
    `bnMean_shard` transported along the row shard. This is precisely what `syncStats`'s first
    collective — `allReduceMeanF` over the replicas' `bnBatchMeanB` — computes. -/
theorem bnMean_row_shard (R N oc h w : Nat) (hR : R ≠ 0) (hm : N * (h * w) ≠ 0)
    (X : Vec ((R * N) * (oc * (h * w)))) (c : Fin oc) :
    bnMean ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c)
      = (1 / (R : ℝ)) * ∑ r : Fin R, bnMean (N*(h*w))
          (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r)) c) := by
  rw [bnMean_shard hR hm (bnShardEquiv R N (h*w))]
  simp only [bnchwFwd_row_batchShard]

/-- **…and so is the global per-channel SECOND MOMENT.** Note: There is no such statement for
    the variance alone — the mean of the shards' variances is not the variance of the union —
    which is why the exchange carries Chan's corrected variance (`bnVar_row_shard_chan`) rather
    than a plain `σ²_r`. -/
theorem bnMeanSq_row_shard (R N oc h w : Nat) (hR : R ≠ 0) (hm : N * (h * w) ≠ 0)
    (X : Vec ((R * N) * (oc * (h * w)))) (c : Fin oc) :
    bnMeanSq ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c)
      = (1 / (R : ℝ)) * ∑ r : Fin R, bnMeanSq (N*(h*w))
          (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r)) c) := by
  rw [bnMeanSq_shard hR hm (bnShardEquiv R N (h*w))]
  simp only [bnchwFwd_row_batchShard]

/-- **Chan's parallel variance on the channel rows**: the global channel variance is the
    replica mean of each replica's own two-pass variance plus its mean's squared offset from the
    global mean. This is what the second collective of a sync-BN forward carries
    (`StableHLO.bnBatchVarAtB`), and it is why no consumer ever forms `E[x²] − μ²`. -/
theorem bnVar_row_shard_chan (R N oc h w : Nat) (hR : R ≠ 0) (hm : N * (h * w) ≠ 0)
    (X : Vec ((R * N) * (oc * (h * w)))) (c : Fin oc) :
    bnVar ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c)
      = (1 / (R : ℝ)) * ∑ r : Fin R,
          (bnVar (N*(h*w))
              (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r)) c)
           + (bnMean (N*(h*w))
                (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r)) c)
              - bnMean ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c))
             * (bnMean (N*(h*w))
                  (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r)) c)
                - bnMean ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c))) := by
  rw [bnVar_shard_chan hR hm (bnShardEquiv R N (h*w))]
  simp only [bnchwFwd_row_batchShard]

/-- **P2b's workhorse: a mean over the global batch of ANY pointwise function of two rows
    is the mean of the replicas' means of the same.** Both dy-reductions the sync backward needs
    have this shape — `mdy` reads only `dy`, `mdyx` reads `x` and `dy` together — so one lemma
    covers both, and `bnMean_row_shard` is its one-row special case. -/
theorem bnMean_pair_row_shard (R N oc h w : Nat) (hR : R ≠ 0) (hm : N * (h * w) ≠ 0)
    (X DY : Vec ((R * N) * (oc * (h * w)))) (c : Fin oc) (f : ℝ → ℝ → ℝ) :
    bnMean ((R*N)*(h*w)) (fun k => f (Mat.unflatten (bnchwFwd (R*N) oc h w X) c k)
                                      (Mat.unflatten (bnchwFwd (R*N) oc h w DY) c k))
      = (1 / (R : ℝ)) * ∑ r : Fin R, bnMean (N*(h*w)) (fun k =>
          f (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r)) c k)
            (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) DY r)) c k)) := by
  rw [bnMean_shard hR hm (bnShardEquiv R N (h*w))]
  simp only [bnchwFwd_row_batchShard]

/-- The global batch-spatial count is nonzero when a replica's is: `(R·N)·(h·w) = R·(N·(h·w))`. -/
theorem mulR_nhw_ne_zero {R N h w : Nat} (hR : 0 < R) (hm : N * (h * w) ≠ 0) :
    (R * N) * (h * w) ≠ 0 := by
  rw [Nat.mul_assoc]; exact Nat.mul_ne_zero hR.ne' hm

/-- **P1 — SYNC-BN ON REPLICA `r` IS THE SHARD-`r` BLOCK OF THE GLOBAL-BATCH BN.**

    Handed the GLOBAL statistics — `(1/R)·Σ_r` of each replica's own `bnMean` and `bnMeanSq`,
    which is what `syncStats` denotes once its `σ²` is read back as `m2 = σ² + μ²` — replica
    `r`'s sync forward on its own shard equals `batchShard r` of `bnBatchTensor4` run on the
    whole `R·N` batch.

    **The spec does not move**: the right-hand side is the EXISTING `bnBatchTensor4`, at
    `N := R·N`. Nothing new is being specified; the render is being shown to hit a target the
    tier already names.

    It is P1a (pointwise, so sharding commutes — no mathematics) composed with P1b (the
    statistics really are the all-reduced ones — all the mathematics) through
    `bnSyncTensor4_at_own_stats`. -/
theorem bnSyncTensor4_shard_eq_global (R N oc h w : Nat) (hR : R ≠ 0) (hm : N * (h * w) ≠ 0)
    (ε : ℝ) (γ β : Vec oc)
    (X : Vec ((R * N) * (oc * (h * w)))) (r : Fin R) :
    bnSyncTensor4 N oc h w ε γ β
        (fun c => (1 / (R : ℝ)) * ∑ r' : Fin R, bnMean (N*(h*w))
          (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r')) c))
        (fun c => (1 / (R : ℝ)) * ∑ r' : Fin R, bnMeanSq (N*(h*w))
          (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r')) c))
        (batchShard R N (oc * (h * w)) X r)
      = batchShard R N (oc * (h * w)) (bnBatchTensor4 (R*N) oc h w ε γ β X) r := by
  have hM := mulR_nhw_ne_zero (Nat.pos_of_ne_zero hR) hm
  have hμ : ∀ c : Fin oc, (1 / (R : ℝ)) * ∑ r' : Fin R, bnMean (N*(h*w))
      (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r')) c)
      = bnMean ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c) :=
    fun c => (bnMean_row_shard R N oc h w hR hm X c).symm
  have hm2 : ∀ c : Fin oc, (1 / (R : ℝ)) * ∑ r' : Fin R, bnMeanSq (N*(h*w))
      (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r')) c)
      = bnMeanSq ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c) :=
    fun c => (bnMeanSq_row_shard R N oc h w hR hm X c).symm
  simp only [hμ, hm2]
  rw [← bnSyncTensor4_batchShard]
  exact congrArg (fun z => batchShard R N (oc * (h * w)) z r)
    (bnSyncTensor4_at_own_stats (R*N) oc h w hM ε γ β X)

/-- **A channel's γ gradient over the global batch is the SUM of the replicas' γ gradients
    at the same handed-in statistics** — a sum, not a mean, because this is a parameter
    gradient: the parameter collective's `1/R` is what turns it into the global-batch mean. The
    row split `bnchwFwd_row_batchShard`, under `Σ` instead of `bnMean`. -/
theorem bnSyncPerChannelGradGamma_row_shard (R N oc h w : Nat) (ε : ℝ) (μ m2 : Vec oc)
    (X DY : Vec ((R * N) * (oc * (h * w)))) (c : Fin oc) :
    bnSyncPerChannelGradGamma oc ((R*N)*(h*w)) ε μ m2
        (bnchwFwd (R*N) oc h w X) (bnchwFwd (R*N) oc h w DY) c
      = ∑ r : Fin R, bnSyncPerChannelGradGamma oc (N*(h*w)) ε μ m2
          (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r))
          (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) DY r)) c := by
  unfold bnSyncPerChannelGradGamma
  rw [← Equiv.sum_comp (bnShardEquiv R N (h*w)), Fintype.sum_prod_type]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro k _
  simp only [bnSyncXhat_apply, bnchwFwd_row_batchShard]

/-- **…and so is the β gradient**, which reads no statistic at all: `Σ dy` over the global
    channel row is the sum of the shards' `Σ dy`. -/
theorem bnPerChannelGradBeta_row_shard (R N oc h w : Nat)
    (DY : Vec ((R * N) * (oc * (h * w)))) (c : Fin oc) :
    bnPerChannelGradBeta oc ((R*N)*(h*w)) (bnchwFwd (R*N) oc h w DY) c
      = ∑ r : Fin R, bnPerChannelGradBeta oc (N*(h*w))
          (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) DY r)) c := by
  unfold bnPerChannelGradBeta
  rw [← Equiv.sum_comp (bnShardEquiv R N (h*w)), Fintype.sum_prod_type]
  apply Finset.sum_congr rfl; intro r _
  apply Finset.sum_congr rfl; intro k _
  exact bnchwFwd_row_batchShard R N oc h w DY c r k

/-- **The sync backward is POINTWISE too**, with the channel read off the index — it reads only
    `x idx` and `dy idx`. Both its reductions were hoisted into the collective. -/
theorem bnSyncPerChannelGradInput_apply (oc m : Nat) (ε : ℝ) (γ μ m2 mdy mdyx : Vec oc)
    (x dy : Vec (oc * m)) (idx : Fin (oc * m)) :
    bnSyncPerChannelGradInput oc m ε γ μ m2 mdy mdyx x dy idx
      = (1 / Real.sqrt (m2 (finProdFinEquiv.symm idx).1
             - μ (finProdFinEquiv.symm idx).1 * μ (finProdFinEquiv.symm idx).1 + ε))
          * (γ (finProdFinEquiv.symm idx).1 * dy idx
             - mdy (finProdFinEquiv.symm idx).1
             - ((x idx - μ (finProdFinEquiv.symm idx).1)
                * (1 / Real.sqrt (m2 (finProdFinEquiv.symm idx).1
                     - μ (finProdFinEquiv.symm idx).1 * μ (finProdFinEquiv.symm idx).1 + ε)))
               * mdyx (finProdFinEquiv.symm idx).1) := by
  unfold bnSyncPerChannelGradInput bnSyncGradInput bnSyncXhat Mat.unflatten
  simp only [Prod.mk.eta, Equiv.apply_symm_apply]

/-- The `[N,C,H,W]` lift: cell `t` of the sync backward depends only on `x t`, `dy t` and its
    own channel's four statistics. -/
theorem bnSyncTensor4GradInput_apply (N oc h w : Nat) (ε : ℝ) (γ μ m2 mdy mdyx : Vec oc)
    (x dy : Vec (N * (oc * (h * w)))) (t : Fin (N * (oc * (h * w)))) :
    bnSyncTensor4GradInput N oc h w ε γ μ m2 mdy mdyx x dy t
      = (1 / Real.sqrt (m2 (bnchwChan N oc h w t)
             - μ (bnchwChan N oc h w t) * μ (bnchwChan N oc h w t) + ε))
          * (γ (bnchwChan N oc h w t) * dy t
             - mdy (bnchwChan N oc h w t)
             - ((x t - μ (bnchwChan N oc h w t))
                * (1 / Real.sqrt (m2 (bnchwChan N oc h w t)
                     - μ (bnchwChan N oc h w t) * μ (bnchwChan N oc h w t) + ε)))
               * mdyx (bnchwChan N oc h w t)) := by
  unfold bnSyncTensor4GradInput
  simp only [bnchwBack]
  rw [bnSyncPerChannelGradInput_apply]
  have hx : bnchwFwd N oc h w x (bnchwBackIdx N oc h w t) = x t := by
    rw [bnchwFwd_apply, bnchwFwdIdx_bnchwBackIdx]
  have hd : bnchwFwd N oc h w dy (bnchwBackIdx N oc h w t) = dy t := by
    show dy (bnchwFwdIdx N oc h w (bnchwBackIdx N oc h w t)) = dy t
    rw [bnchwFwdIdx_bnchwBackIdx]
  have hc : (finProdFinEquiv.symm (bnchwBackIdx N oc h w t)).1 = bnchwChan N oc h w t := by
    unfold bnchwBackIdx bnchwChan
    simp only [Equiv.symm_apply_apply]
  rw [hx, hd, hc]

/-- **P2a — the sync backward COMMUTES WITH SHARDING, for any statistics at all.**
    The backward twin of `bnSyncTensor4_batchShard`, and equally free of mathematics: pointwise
    plus channel-preserving reindex. -/
theorem bnSyncTensor4GradInput_batchShard (R N oc h w : Nat) (ε : ℝ)
    (γ μ m2 mdy mdyx : Vec oc) (X DY : Vec ((R * N) * (oc * (h * w)))) (r : Fin R) :
    batchShard R N (oc * (h * w))
        (bnSyncTensor4GradInput (R*N) oc h w ε γ μ m2 mdy mdyx X DY) r
      = bnSyncTensor4GradInput N oc h w ε γ μ m2 mdy mdyx
          (batchShard R N (oc * (h * w)) X r) (batchShard R N (oc * (h * w)) DY r) := by
  funext idx
  unfold batchShard
  rw [bnSyncTensor4GradInput_apply, bnSyncTensor4GradInput_apply, bnchwChan_batchShard]

/-- The `R = 1` backward anchor restated with `bnSyncXhat` in the `mdyx` reduction — the form
    the sync GRAPH produces, since `bnSyncDyStatsB` builds `x̂` from the statistics handed to it
    rather than from `x` directly. -/
theorem bnSyncTensor4GradInput_at_own_stats' (N oc h w : Nat) (hm : N * (h * w) ≠ 0)
    (ε : ℝ) (γ : Vec oc) (x dy : Vec (N * (oc * (h * w)))) :
    bnSyncTensor4GradInput N oc h w ε γ
        (fun c => bnMean   (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c))
        (fun c => bnMeanSq (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c))
        (fun c => bnMean (N*(h*w)) (fun k =>
          γ c * Mat.unflatten (bnchwFwd N oc h w dy) c k))
        (fun c => bnMean (N*(h*w)) (fun k =>
          bnSyncXhat (N*(h*w)) ε
            (bnMean   (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c))
            (bnMeanSq (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c))
            (Mat.unflatten (bnchwFwd N oc h w x) c) k
          * (γ c * Mat.unflatten (bnchwFwd N oc h w dy) c k)))
        x dy
      = bnBatchTensor4GradInput N oc h w ε γ x dy := by
  have hxh : (fun c => bnMean (N*(h*w)) (fun k =>
              bnSyncXhat (N*(h*w)) ε
                (bnMean   (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c))
                (bnMeanSq (N*(h*w)) (Mat.unflatten (bnchwFwd N oc h w x) c))
                (Mat.unflatten (bnchwFwd N oc h w x) c) k
              * (γ c * Mat.unflatten (bnchwFwd N oc h w dy) c k)))
          = (fun c => bnMean (N*(h*w)) (fun k =>
              bnXhat (N*(h*w)) ε (Mat.unflatten (bnchwFwd N oc h w x) c) k
              * (γ c * Mat.unflatten (bnchwFwd N oc h w dy) c k))) := by
    funext c
    rw [bnSyncXhat_at_own_stats _ hm]
  rw [hxh]
  exact bnSyncTensor4GradInput_at_own_stats N oc h w hm ε γ x dy

/-- **P2 — THE SYNC BACKWARD ON REPLICA `r` IS THE SHARD-`r` BLOCK OF THE GLOBAL-BATCH
    INPUT-VJP.**

    Handed the four all-reduced statistics — each literally `(1/R)·Σ_r'` of a per-replica
    quantity, which is what `syncStats`, then `allReduceMeanF` of `bnSyncDyStatsB`,
    denotes — replica `r`'s sync backward equals `batchShard r` of `bnBatchTensor4GradInput`
    run on the whole `R·N` batch.

    **Including the cross-shard terms.** `mdyx` averages `x̂·dx̂` with `x̂` built from the GLOBAL
    `μ`, `m2` — not from the shard's own statistics — which is exactly why `bnSyncDyStatsB`
    consumes the already-reduced vector instead of recomputing one. That is what makes each
    replica's output the true shard-`r` block of the global gradient, and hence the all-reduced
    PARAMETER gradient exact rather than approximate.

    **The spec does not move**: the right-hand side is the existing
    `bnBatchTensor4GradInput` at `N := R·N`. -/
theorem bnSyncTensor4GradInput_shard_eq_global (R N oc h w : Nat) (hR : R ≠ 0)
    (hm : N * (h * w) ≠ 0) (ε : ℝ) (γ : Vec oc)
    (X DY : Vec ((R * N) * (oc * (h * w)))) (r : Fin R) (μg m2g mdyg mdyxg : Vec oc)
    (hμ : μg = fun c => (1 / (R : ℝ)) * ∑ r' : Fin R, bnMean (N*(h*w))
      (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r')) c))
    (hm2 : m2g = fun c => (1 / (R : ℝ)) * ∑ r' : Fin R, bnMeanSq (N*(h*w))
      (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r')) c))
    (hmdy : mdyg = fun c => (1 / (R : ℝ)) * ∑ r' : Fin R, bnMean (N*(h*w)) (fun k =>
      γ c * Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) DY r')) c k))
    (hmdyx : mdyxg = fun c => (1 / (R : ℝ)) * ∑ r' : Fin R, bnMean (N*(h*w)) (fun k =>
      bnSyncXhat (N*(h*w)) ε (μg c) (m2g c)
        (Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) X r')) c) k
      * (γ c * Mat.unflatten (bnchwFwd N oc h w (batchShard R N (oc * (h * w)) DY r')) c k))) :
    bnSyncTensor4GradInput N oc h w ε γ μg m2g mdyg mdyxg
        (batchShard R N (oc * (h * w)) X r) (batchShard R N (oc * (h * w)) DY r)
      = batchShard R N (oc * (h * w)) (bnBatchTensor4GradInput (R*N) oc h w ε γ X DY) r := by
  have hM := mulR_nhw_ne_zero (Nat.pos_of_ne_zero hR) hm
  -- 1. the forward statistics are the global row's own (P1b)
  have hμ' : μg = fun c => bnMean ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c) := by
    rw [hμ]; funext c; rw [← bnMean_row_shard R N oc h w hR hm X c]
  have hm2' : m2g = fun c => bnMeanSq ((R*N)*(h*w))
      (Mat.unflatten (bnchwFwd (R*N) oc h w X) c) := by
    rw [hm2]; funext c; rw [← bnMeanSq_row_shard R N oc h w hR hm X c]
  -- 2. and so are the two dy reductions (P2b), via the pair lemma
  have hmdy' : mdyg = fun c => bnMean ((R*N)*(h*w)) (fun k =>
      γ c * Mat.unflatten (bnchwFwd (R*N) oc h w DY) c k) := by
    rw [hmdy]; funext c
    exact (bnMean_pair_row_shard R N oc h w hR hm X DY c (fun _ d => γ c * d)).symm
  have hmdyx' : mdyxg = fun c => bnMean ((R*N)*(h*w)) (fun k =>
      bnSyncXhat ((R*N)*(h*w)) ε
        (bnMean   ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c))
        (bnMeanSq ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c))
        (Mat.unflatten (bnchwFwd (R*N) oc h w X) c) k
      * (γ c * Mat.unflatten (bnchwFwd (R*N) oc h w DY) c k)) := by
    rw [hmdyx, hμ', hm2']; funext c
    simp only [bnSyncXhat_apply]
    exact (bnMean_pair_row_shard R N oc h w hR hm X DY c
      (fun xv dv => (xv - bnMean ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c))
        * (1 / Real.sqrt (bnMeanSq ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c)
            - bnMean ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c)
              * bnMean ((R*N)*(h*w)) (Mat.unflatten (bnchwFwd (R*N) oc h w X) c) + ε))
        * (γ c * dv))).symm
  -- 3. P2a composed with the R = 1 anchor at the global batch
  rw [hμ', hm2', hmdy', hmdyx', ← bnSyncTensor4GradInput_batchShard]
  exact congrArg (fun z => batchShard R N (oc * (h * w)) z r)
    (bnSyncTensor4GradInput_at_own_stats' (R*N) oc h w hM ε γ X DY)

-- ════════════════════════════════════════════════════════════════
-- § P3 — sharding commutes with the per-example lifts
-- ════════════════════════════════════════════════════════════════

/-- Example `n` of shard `r` is example `(r, n)` of the global batch. -/
theorem batchSlice_batchShard {R N a : Nat} (X : Vec ((R * N) * a)) (r : Fin R) (n : Fin N) :
    batchSlice N a (batchShard R N a X r) n
      = batchSlice (R * N) a X (finProdFinEquiv (r, n)) := by
  funext i
  simp only [batchSlice, batchShard, Equiv.symm_apply_apply]

/-- **`batchMap` commutes with sharding** — the block analogue of `batchSlice_batchMap`. A
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
  simp only [Fin.sum_const, nsmul_eq_mul]
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

/-- **The first collective IS the global mean** — `bnMean_shard` on the channel rows. -/
theorem den_syncStats_left {N oc h w : Nat} (R : Nat) (hR : 0 < R) (hm : N * (h * w) ≠ 0)
    (t t' : String) (ds ds' : List Nat) (x : Fin R → SHlo (N * (oc * (h * w))))
    (X : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r) (c : Fin oc) :
    den (syncStats R hR t t' ds ds' x) (Fin.castAdd oc c)
      = bnMean ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c) := by
  simp only [syncStats, den_bnPackB, Fin.append_left, den_allReduceMeanF, den_bnBatchMeanB, hx]
  exact (bnMean_row_shard R N oc h w (Nat.pos_iff_ne_zero.mp hR) hm X c).symm

/-- **The second collective IS the global variance** — Chan's parallel variance
    (`bnVar_row_shard_chan`): each replica's two-pass `σ²_r` plus `(μ_r − μ)²`, averaged. This
    is the lemma a one-round `E[x²]` exchange does not have. -/
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

/-- **P1 on the graph, any `R`.** Replica `r`'s `bnSyncF`, fed by the two-round statistics
    subgraph over the `R` replicas' inputs, denotes `batchShard r` of the batch-`R·N`
    `bnBatchTensor4` — given that each replica's operand denotes its shard (`hx`, the chain's
    induction hypothesis). `den_bnSyncF_allReduce_R1` is this at `R := 1`. -/
theorem den_bnSyncF_allReduce {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (hm : N * (h * w) ≠ 0) (gN bN es t t' : String)
    (ds ds' : List Nat) (ε : ℝ) (γ β : Vec oc) (x : Fin R → SHlo (N * (oc * (h * w))))
    (X : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r) (r : Fin R) :
    den (.bnSyncF gN bN es ε γ β (x r) (syncStats R hR t t' ds ds' x))
      = batchShard R N (oc * (h * w)) (bnBatchTensor4 (R * N) oc h w ε γ β X) r := by
  have hM := mulR_nhw_ne_zero hR hm
  rw [den_bnSyncF]
  simp only [den_syncStats_left R hR hm t t' ds ds' x X hx,
             den_syncStats_right R hR hm t t' ds ds' x X hx, global_var_add_sq R hM, hx]
  rw [← bnSyncTensor4_batchShard]
  exact congrArg (fun z => batchShard R N (oc * (h * w)) z r)
    (bnSyncTensor4_at_own_stats (R * N) oc h w hM ε γ β X)

/-- **P2 on the graph, any `R`.** Replica `r`'s `bnSyncBack`, fed by the collective over the
    replicas' `bnSyncDyStatsB` (each reading the packed forward statistics), denotes
    `batchShard r` of the batch-`R·N` `bnBatchTensor4GradInput` — given that each replica's
    saved activation and its incoming cotangent are its shards of the global ones (`hx` / `hxv`,
    `hdy`). `den_bnSyncBack_allReduce_R1` is this at `R := 1`. -/
theorem den_bnSyncBack_allReduce {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (hm : N * (h * w) ≠ 0) (gN xN es t t' t'' : String)
    (ds ds' ds'' : List Nat) (ε : ℝ) (γ : Vec oc)
    (x : Fin R → SHlo (N * (oc * (h * w)))) (xv : Fin R → Vec (N * (oc * (h * w))))
    (dy : Fin R → SHlo (N * (oc * (h * w)))) (X DY : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r)
    (hxv : ∀ r, xv r = batchShard R N (oc * (h * w)) X r)
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * (h * w)) DY r) (r : Fin R) :
    den (.bnSyncBack gN xN es ε γ (xv r) (dy r)
          (.allReduceMeanF R hR t'' ds'' (fun r' => .bnSyncDyStatsB gN xN es ε γ (xv r') (dy r')
            (syncStats R hR t t' ds ds' x))))
      = batchShard R N (oc * (h * w)) (bnBatchTensor4GradInput (R * N) oc h w ε γ X DY) r := by
  have hM := mulR_nhw_ne_zero hR hm
  have hRne : R ≠ 0 := Nat.pos_iff_ne_zero.mp hR
  have hRr : (R : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hRne
  rw [den_bnSyncBack, hxv r, hdy r]
  -- the outer collective is [μ ‖ σ² ‖ mdy ‖ mdyx]; μ and σ² pass through (`dpMean_const_mul`)
  simp only [den_allReduceMeanF, den_bnSyncDyStatsB, Fin.append_left, Fin.append_right,
             den_syncStats_left R hR hm t t' ds ds' x X hx,
             den_syncStats_right R hR hm t t' ds ds' x X hx, global_var_add_sq R hM,
             hxv, hdy, dpMean_const_mul hRr]
  exact bnSyncTensor4GradInput_shard_eq_global R N oc h w hRne hm ε γ X DY r
    _ _ _ _ (by funext c; exact bnMean_row_shard R N oc h w hRne hm X c)
    (by funext c; exact bnMeanSq_row_shard R N oc h w hRne hm X c) rfl rfl

-- ════════════════════════════════════════════════════════════════
-- § P4 — the parameter collective is 1/R of the global-batch gradient node
-- ════════════════════════════════════════════════════════════════

/-- **P2γ on the graph, any `R`, already all-reduced.** The parameter collective over the
    replicas' `bnSyncGammaGradB` — each at its shard, its shard's cotangent and the packed global
    statistics — is `1/R` of `bnPerChannelGradGamma` at `N := R·N`: the committed γ gradient
    at the global batch. The BN γ node is the one parameter gradient sync-BN changes, because it
    is the one that reads `x̂`. -/
theorem den_allReduceMeanF_bnSyncGammaGradB {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (hm : N * (h * w) ≠ 0) (xN es t t' t'' : String)
    (ds ds' ds'' : List Nat) (ε : ℝ)
    (x : Fin R → SHlo (N * (oc * (h * w)))) (xv : Fin R → Vec (N * (oc * (h * w))))
    (dy : Fin R → SHlo (N * (oc * (h * w)))) (X DY : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r)
    (hxv : ∀ r, xv r = batchShard R N (oc * (h * w)) X r)
    (hdy : ∀ r, den (dy r) = batchShard R N (oc * (h * w)) DY r) (c : Fin oc) :
    den (.allReduceMeanF R hR t'' ds'' (fun r => .bnSyncGammaGradB xN es ε (xv r) (dy r)
          (syncStats R hR t t' ds ds' x))) c
      = (1 / (R : ℝ)) * bnPerChannelGradGamma oc ((R * N) * (h * w)) ε
          (bnchwFwd (R * N) oc h w X) (bnchwFwd (R * N) oc h w DY) c := by
  have hM := mulR_nhw_ne_zero hR hm
  simp only [den_allReduceMeanF, den_bnSyncGammaGradB,
             den_syncStats_left R hR hm t t' ds ds' x X hx,
             den_syncStats_right R hR hm t t' ds ds' x X hx, global_var_add_sq R hM, hxv, hdy]
  -- the shard sums are the global row sum, and at the global statistics that is the γ gradient
  rw [← bnSyncPerChannelGradGamma_row_shard,
      congrFun (bnSyncPerChannelGradGamma_at_own_stats oc ((R*N)*(h*w)) hM ε
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
    `N := R·N` denotes. Note: No per-replica variance is ever averaged on its own: the between-shard
    spread `(μ_r − μ)²` rides along, which is what Chan's formula is. -/
theorem den_bnStatsVarB_allReduce {N oc h w : Nat} (R : Nat) (hR : 0 < R)
    (hm : N * (h * w) ≠ 0) (t t' : String) (ds ds' : List Nat)
    (x : Fin R → SHlo (N * (oc * (h * w)))) (X : Vec ((R * N) * (oc * (h * w))))
    (hx : ∀ r, den (x r) = batchShard R N (oc * (h * w)) X r) :
    den (.bnStatsVarB (syncStats R hR t t' ds ds' x))
      = fun c => bnVar ((R * N) * (h * w)) (Mat.unflatten (bnchwFwd (R * N) oc h w X) c) := by
  funext c
  simp only [den_bnStatsVarB, den_syncStats_right R hR hm t t' ds ds' x X hx]

/-- Closes a P4 collective once `den` has been unfolded on both sides: splits the global batch
    sum `Σ_{m : R·N}` into `Σ_r Σ_n` and reads each replica's shard back as a slice of the global
    tensor. `rw`, not `simp`, for the slices: in the conv kinds the `x` slice sits inside a VJP whose
    TYPE depends on it, and simp has no congruence through a dependent argument. -/
macro "shard_sum" : tactic => `(tactic| (
  rw [sum_finProdFinEquiv]
  apply Finset.sum_congr rfl; intro _ _
  apply Finset.sum_congr rfl; intro _ _
  repeat rw [batchSlice_batchShard]))

/-- **P4 for the `Σ_n`-shaped gradients, at the conv weight.** The collective over the
    replicas' `convWeightGradB`, each on its shard at the shard-`r` block of the global
    cotangent, is `1/R` of the batch-`R·N` node at that cotangent. `den_allReduceMeanF_convWeightGradB`
    (`DataParallelNode`) is the same collective with NO relation between the replicas' inputs; this is what
    it becomes once P1–P3 relate them. Every other `Σ_n` gradient (`denseWeightGradB`,
    `denseBiasGradB`, the strided / depthwise / bias kinds) closes by the same `shard_sum`. -/
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
  simp only [denStep, denStepApp, hdy]
  shard_sum

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
  simp only [denStep, denStepApp, hdy]
  exact (bnPerChannelGradBeta_row_shard R N oc h w DY c).symm

-- ════════════════════════════════════════════════════════════════
-- § The divisor step
-- ════════════════════════════════════════════════════════════════

/-- **`f` scales with its argument**: `f (s • v) = s • f v`, spelled pointwise. The statement of
    every cotangent-chain `_smul` lemma; an `abbrev`, so `rw [h]` and `h s v` see the equation. -/
abbrev IsHomog {a b : Nat} (f : Vec a → Vec b) : Prop :=
  ∀ (s : ℝ) (v : Vec a), f (fun i => s * v i) = fun i => s * f v i

theorem IsHomog.comp {a b c : Nat} {g : Vec b → Vec c} {f : Vec a → Vec b} (hg : IsHomog g)
    (hf : IsHomog f) : IsHomog (g ∘ f) := fun s v => by
  simp only [Function.comp_apply, hf s v, hg s]

/-- **A VJP backward is linear in its cotangent** — read off `HasVJP.correct`. This is the one
    step that reconciles a DP render's `divConstB N` with the batch-`R·N` step's `divConstB (R·N)`:
    the per-replica cotangent is `R ·` the shard of the global one, so every per-replica
    gradient is `R ·` its shard contribution, and the collective's `1/R` cancels it. -/
theorem HasVJP.backward_smul {m n : Nat} {f : Vec m → Vec n} (hf : HasVJP f) (x : Vec m) :
    IsHomog (hf.backward x) := by
  intro a dy
  funext i
  rw [hf.correct, hf.correct, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun j _ => by ring)

end Proofs
