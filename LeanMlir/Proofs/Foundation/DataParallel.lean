import LeanMlir.Proofs.Foundation.Tensor

/-! # Data parallelism: the gradient mean, and what function trained

Every `*dp*` artifact in `verified_mlir/` is ONE program run on `R` replicas. Per parameter,
after the gradient node and before the optimizer tail, `emitGradAllReduce` emits
`stablehlo.all_reduce(add)` over `replica_groups = [[0..R-1]]` followed by a divide by `R`.
That text is emitted OUTSIDE the `SHlo` AST and is a declared trusted carve-out, so every tie
in the repo — `r34_net_tiedB`, `mnv2_net_tiedB`, `efficientnet_net_tiedG` — is stated at the
PER-REPLICA gradient node and says so in its own header. This file is the ℝ-level half of
closing that disclaimer: piece 1 of `planning/proofs_tier_to_paper_nets.md` §4d.

## What is proved

* `lossGrad_meanLoss` — the gradient of a mean of losses is the mean of their gradients.
  Linearity of `pdiv`, and the only analysis in the file.
* ⭐ `dpMeanGrad_eq_grad_meanLoss` — the all-reduced mean gradient `(1/R) Σ_r g_r` IS
  `∇((1/R) Σ_r L_r)`. **This is what the collective computes, named as a gradient of
  something.** It holds for any per-replica losses whatever, batch-coupled or not.
* ⭐⭐ `meanLoss_shard` / `dpMeanGrad_eq_globalBatchGrad_of_perExample` — when the replica loss
  is the MEAN OVER ITS SLICE of a per-example loss (no batch coupling: ConvNeXt, ViT, every
  inference-BN forward), the mean of the `R` replica losses is literally the mean over the
  global batch of `R·N` examples, so the DP step is the single-device step at batch `R·N`.
  Stated at an arbitrary shard `e : Fin R × Fin N ≃ Fin (R*N)` — WHICH examples land on which
  replica does not enter, only that together they are the batch. The contiguous shard the DP
  shim cuts is the `finProdFinEquiv` instance.
* ⛔ `dpMeanGrad_ne_globalBatchGrad` — and for a batch-coupled loss that is FALSE, at an
  explicit two-replica witness. A training-mode BatchNorm reads a nonlinear function of its
  own slice's statistics; `bnToyLoss` is the smallest thing with that shape. So a batch-BN net
  trained data-parallel did NOT minimise the batch-`R·N` loss, and `dpMeanGrad_eq_grad_meanLoss`
  is the honest statement of what it did minimise. `dpToyShard_eq_batch` is the lemma that
  makes it a witness rather than a comparison of two unrelated datasets.
* ⭐⭐ `dpIterate_lockstep` / `dpIterate_eq_meanLossTrain` — the lockstep induction. Identical
  initial parameters and an identical (all-reduced) update keep the `R` state copies equal at
  every step, so `n` steps of the `R`-replica system are `n` steps of ORDINARY single-device
  training on the mean loss. That is the property `VerifiedTrain.lean` relies on when it
  checkpoints from replica 0.

## What is NOT claimed

⚠ **Nothing here is about the emitted `all_reduce`.** `den (allReduceMeanF R g) = (1/R) Σ_r den (g r)`
is §4d piece 2, an `SHlo` constructor with a `den`, a `pretty` and a parser case, and it waits on
4c's batched chains. Until it lands a tie composes with these lemmas only through the reader.

⚠ **BatchNorm statistics are per replica.** Nothing all-reduces μ/var, which is why `N` in the
batch-BN tiers is the PER-CARD batch and why `dpMeanGrad_ne_globalBatchGrad` is not a curiosity.

⚠ **That every replica starts from the same parameters, that the checkpoint is read from one
replica, and that `replica_groups` names all `R` devices are the DRIVER's** (`VerifiedTrain.lean`,
`ffi/pjrt_ffi.c`, `PJRT_REPLICAS`), not theorems here. `dpIterate_lockstep` takes the shared start
as a HYPOTHESIS — it says what follows from it, not that the driver establishes it. The
`*-dp-check` gates are the empirical evidence for that half.

⭐ **WHICH examples land where is the one piece of that which does become a theorem, and only for
half the nets.** `dpMeanGrad_eq_globalBatchGrad_of_perExample` binds the shard, so for a net with
no batch coupling the partition is provably irrelevant and all the driver has to get right is that
the slices cover the batch. ⛔ For a batch-BN net the partition changes the function, and nothing
here recovers it.

`pdiv_const_smul`, the scalar-multiple rule this file needed, is `Tensor.lean`'s (moved 2026-09-08).
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Scalar losses, their gradients, and one missing `pdiv` rule
-- ════════════════════════════════════════════════════════════════

/-- **A scalar loss's gradient**, in the `Vec 1`-lifted spelling `pdiv` reads. `SmoothedLossCot`
    writes this out at every use; it is worth a name here because the whole file is about
    rearranging sums of them. -/
noncomputable def lossGrad {P : Nat} (L : Vec P → ℝ) (θ : Vec P) : Vec P :=
  fun i => pdiv (fun θ' : Vec P => fun _ : Fin 1 => L θ') θ i 0

/-- Differentiability of a scalar loss, in the same lifted spelling. -/
abbrev LossDifferentiableAt {P : Nat} (L : Vec P → ℝ) (θ : Vec P) : Prop :=
  DifferentiableAt ℝ (fun θ' : Vec P => fun _ : Fin 1 => L θ') θ


/-- **The all-reduced gradient**: `(1/R) Σ_r g_r`. `emitGradAllReduce`'s `all_reduce(add)`
    followed by its divide by `R`, read as a function of the `R` per-replica gradients. -/
noncomputable def dpMean {R P : Nat} (g : Fin R → Vec P) : Vec P :=
  fun i => (1 / (R : ℝ)) * ∑ r, g r i

/-- **A mean of losses.** Used at two different index meanings and deliberately ONE definition:
    over replicas it is the function data parallelism minimises, and over examples it is the
    batch mean a single device minimises. That those coincide under no batch coupling is
    `meanLoss_shard`, and it is the content of §4d piece 1. -/
noncomputable def meanLoss {M P : Nat} (L : Fin M → Vec P → ℝ) : Vec P → ℝ :=
  fun θ => (1 / (M : ℝ)) * ∑ m, L m θ

/-- A mean of losses is differentiable where its summands are. -/
theorem meanLoss_differentiableAt {M P : Nat} (L : Fin M → Vec P → ℝ) (θ : Vec P)
    (hdiff : ∀ m, LossDifferentiableAt (L m) θ) :
    LossDifferentiableAt (meanLoss L) θ := by
  have hsum : DifferentiableAt ℝ (fun y : Vec P => fun _ : Fin 1 => ∑ m : Fin M, L m y) θ := by
    have heq : (fun y : Vec P => fun _ : Fin 1 => ∑ m : Fin M, L m y)
             = (fun y : Vec P => ∑ m : Fin M, (fun _ : Fin 1 => L m y)) := by
      funext y k; rw [Finset.sum_apply]
    rw [heq]
    exact DifferentiableAt.fun_sum (fun m _ => hdiff m)
  show DifferentiableAt ℝ (fun y : Vec P => fun _ : Fin 1 => (1 / (M : ℝ)) * ∑ m, L m y) θ
  exact (hsum.const_smul (1 / (M : ℝ)) : _)

/-- ⭐ **The gradient of a mean is the mean of the gradients.** Linearity of `pdiv`, and the
    only analysis in this file: `pdiv_const_smul` pulls the `1/M` out and `pdiv_finset_sum`
    splits the sum. Everything downstream is this lemma read at two index meanings. -/
theorem lossGrad_meanLoss {M P : Nat} (L : Fin M → Vec P → ℝ) (θ : Vec P)
    (hdiff : ∀ m, LossDifferentiableAt (L m) θ) :
    lossGrad (meanLoss L) θ = dpMean (fun m => lossGrad (L m) θ) := by
  funext i
  have hsum : DifferentiableAt ℝ (fun y : Vec P => fun _ : Fin 1 => ∑ m : Fin M, L m y) θ := by
    have heq : (fun y : Vec P => fun _ : Fin 1 => ∑ m : Fin M, L m y)
             = (fun y : Vec P => ∑ m : Fin M, (fun _ : Fin 1 => L m y)) := by
      funext y k; rw [Finset.sum_apply]
    rw [heq]
    exact DifferentiableAt.fun_sum (fun m _ => hdiff m)
  show pdiv (fun θ' : Vec P => fun _ : Fin 1 => (1 / (M : ℝ)) * ∑ m, L m θ') θ i 0
       = (1 / (M : ℝ)) * ∑ m, pdiv (fun θ' : Vec P => fun _ : Fin 1 => L m θ') θ i 0
  rw [pdiv_const_smul (1 / (M : ℝ)) (fun θ' : Vec P => fun _ : Fin 1 => ∑ m, L m θ') θ hsum i 0]
  congr 1
  exact pdiv_finset_sum Finset.univ (fun m => fun θ' : Vec P => fun _ : Fin 1 => L m θ') θ
    (fun m _ => hdiff m) i 0

/-- ⭐⭐ **What the collective computes, named as a gradient.** The all-reduced mean of the `R`
    per-replica gradients IS the gradient of the mean of the `R` per-replica losses.

    No hypothesis on the losses beyond differentiability — in particular this holds for a
    training-mode BatchNorm net, where each `L r` genuinely depends on the whole of replica
    `r`'s slice. That is the point: it names what a data-parallel run minimises without
    claiming that function is the global-batch loss (`dpMeanGrad_ne_globalBatchGrad`). -/
theorem dpMeanGrad_eq_grad_meanLoss {R P : Nat} (L : Fin R → Vec P → ℝ) (θ : Vec P)
    (hdiff : ∀ r, LossDifferentiableAt (L r) θ) :
    dpMean (fun r => lossGrad (L r) θ) = lossGrad (meanLoss L) θ :=
  (lossGrad_meanLoss L θ hdiff).symm

-- ════════════════════════════════════════════════════════════════
-- § No batch coupling: the DP step IS the global-batch step
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **Sharding a per-example loss is invisible to the mean.** If replica `r`'s loss is the
    mean over its own `N` examples, then the mean of the `R` replica losses is the mean over
    all `R·N` examples — as FUNCTIONS, before any derivative is taken.

    Stated at an arbitrary `e`, so which examples land on which replica does not enter: the
    contiguous slices the DP shim cuts (`finProdFinEquiv`), the interleave `ds.shard` produces
    for the sharded producers, and any other partition all give the same theorem. -/
theorem meanLoss_shard {R N P : Nat} (e : Fin R × Fin N ≃ Fin (R * N))
    (ℓ : Fin (R * N) → Vec P → ℝ) :
    meanLoss (fun r => meanLoss (fun n => ℓ (e (r, n)))) = meanLoss ℓ := by
  funext θ
  show (1 / (R : ℝ)) * ∑ r : Fin R, ((1 / (N : ℝ)) * ∑ n : Fin N, ℓ (e (r, n)) θ)
       = (1 / ((R * N : Nat) : ℝ)) * ∑ k : Fin (R * N), ℓ k θ
  rw [← Finset.mul_sum, ← mul_assoc, div_mul_div_comm, one_mul, Nat.cast_mul,
      ← Equiv.sum_comp e (fun k => ℓ k θ), Fintype.sum_prod_type]

/-- ⭐⭐ **For a net with no batch coupling the data-parallel step IS the single-device step at
    the global batch `R·N`.** `meanLoss_shard` under `lossGrad`.

    "No batch coupling" is the hypothesis's shape, not a side condition: it is spelled by
    giving replica `r` the loss `meanLoss (fun n => ℓ (e (r, n)))` — a mean of per-example
    terms — which a training-mode BatchNorm net's replica loss is not. -/
theorem dpMeanGrad_eq_globalBatchGrad_of_perExample {R N P : Nat}
    (e : Fin R × Fin N ≃ Fin (R * N)) (ℓ : Fin (R * N) → Vec P → ℝ) (θ : Vec P)
    (hdiff : ∀ k, LossDifferentiableAt (ℓ k) θ) :
    dpMean (fun r => lossGrad (meanLoss (fun n => ℓ (e (r, n)))) θ)
      = lossGrad (meanLoss ℓ) θ := by
  rw [dpMeanGrad_eq_grad_meanLoss _ θ
      (fun r => meanLoss_differentiableAt _ θ (fun n => hdiff (e (r, n)))),
      meanLoss_shard e ℓ]

/-- The contiguous shard — replica `r` owns examples `[N·r, N·r + N)` — as an instance. This is
    the split `VerifiedTrain.lean`'s DP path cuts (`elems / replicas`). -/
theorem dpMeanGrad_eq_globalBatchGrad_contiguous {R N P : Nat}
    (ℓ : Fin (R * N) → Vec P → ℝ) (θ : Vec P)
    (hdiff : ∀ k, LossDifferentiableAt (ℓ k) θ) :
    dpMean (fun r => lossGrad (meanLoss (fun n => ℓ (finProdFinEquiv (r, n)))) θ)
      = lossGrad (meanLoss ℓ) θ :=
  dpMeanGrad_eq_globalBatchGrad_of_perExample finProdFinEquiv ℓ θ hdiff

-- ════════════════════════════════════════════════════════════════
-- § Batch coupling: and there it is FALSE
-- ════════════════════════════════════════════════════════════════

/-- The batch statistic a training-mode BatchNorm reduces over: the slice's mean. -/
noncomputable def sliceMean {N : Nat} (xs : Fin N → ℝ) : ℝ := (1 / (N : ℝ)) * ∑ n, xs n

/-- **A loss that reads a NONLINEAR function of its own slice's batch statistic** — the shape
    training-mode BatchNorm gives every replica loss in this repo, reduced to the smallest
    thing that still has it. Linear in the parameter, so the gradient is a constant and the
    arithmetic is visible; QUADRATIC in the slice mean, which is the coupling. -/
noncomputable def bnToyLoss {N : Nat} (xs : Fin N → ℝ) (θ : Vec 1) : ℝ :=
  (sliceMean xs) ^ 2 * θ 0

/-- Replica `r`'s slice in the witness: one example, of value `2r`. -/
def dpToyShard : Fin 2 → Fin 1 → ℝ := fun r _ => 2 * (r.val : ℝ)

/-- The global batch those two slices make: `{0, 2}`, whose mean is `1`. -/
def dpToyBatch : Fin 2 → ℝ := fun k => 2 * (k.val : ℝ)

/-- ⚠ **The two shards ARE the global batch**, under the contiguous split — replica `r` owns
    example `r`. Without this the witness below would be comparing two unrelated datasets and
    would prove nothing; a comparison against a re-derivation tests the re-derivation. -/
theorem dpToyShard_eq_batch (r : Fin 2) (n : Fin 1) :
    dpToyShard r n = dpToyBatch (finProdFinEquiv (r, n)) := by
  show 2 * (r.val : ℝ) = 2 * (((finProdFinEquiv (r, n)) : Fin 2).val : ℝ)
  rw [finProdFinEquiv_apply_val]
  simp

/-- The gradient of a linear form, which is all `bnToyLoss` needs. -/
theorem lossGrad_smul_coord {P : Nat} (c : ℝ) (j : Fin P) (θ : Vec P) (i : Fin P) :
    lossGrad (fun θ' : Vec P => c * θ' j) θ i = if i = j then c else 0 := by
  have hlin : DifferentiableAt ℝ (fun y : Vec P => fun _ : Fin 1 => y j) θ :=
    (reindexCLM (fun _ : Fin 1 => j)).differentiableAt
  show pdiv (fun θ' : Vec P => fun _ : Fin 1 => c * θ' j) θ i 0 = _
  rw [pdiv_const_smul c (fun θ' : Vec P => fun _ : Fin 1 => θ' j) θ hlin i 0,
      pdiv_reindex (fun _ : Fin 1 => j) θ i 0]
  by_cases h : i = j
  · rw [if_pos h, if_pos h, mul_one]
  · rw [if_neg h, if_neg h, mul_zero]

/-- `bnToyLoss`'s gradient is the squared slice mean, at every coordinate of `Vec 1`. -/
theorem lossGrad_bnToyLoss {N : Nat} (xs : Fin N → ℝ) (θ : Vec 1) (i : Fin 1) :
    lossGrad (bnToyLoss xs) θ i = (sliceMean xs) ^ 2 := by
  rw [show bnToyLoss xs = (fun θ' : Vec 1 => (sliceMean xs) ^ 2 * θ' 0) from rfl,
      lossGrad_smul_coord, if_pos (Subsingleton.elim i 0)]

/-- ⛔⛔ **With batch coupling the previous section is FALSE, and here is the witness.**
    Two replicas, one example each — slices `{0}` and `{2}`, global batch `{0, 2}`. The
    data-parallel mean gradient is `(0² + 2²)/2 = 2`; the gradient of the global-batch loss is
    `1² = 1`.

    So a training-BN net trained data-parallel did NOT take a step on the batch-`R·N` loss, at
    any learning rate and however small the gradients. What it took a step on is the mean of
    the `R` per-replica batch-BN losses, which is `dpMeanGrad_eq_grad_meanLoss` — a different
    function, and the only honest answer to "what trained". ⚠ The witness needs no BatchNorm:
    ANY nonlinear read of a per-slice statistic separates the two, which is why the split is
    structural rather than a property of the normalisation's formula. -/
theorem dpMeanGrad_ne_globalBatchGrad (θ : Vec 1) :
    dpMean (fun r : Fin 2 => lossGrad (bnToyLoss (dpToyShard r)) θ)
      ≠ lossGrad (bnToyLoss dpToyBatch) θ := by
  intro h
  have h0 := congrFun h 0
  rw [show dpMean (fun r : Fin 2 => lossGrad (bnToyLoss (dpToyShard r)) θ) 0
         = (1 / (2 : ℝ)) * ∑ r : Fin 2, lossGrad (bnToyLoss (dpToyShard r)) θ 0 by
        simp [dpMean]] at h0
  rw [Fin.sum_univ_two, lossGrad_bnToyLoss, lossGrad_bnToyLoss, lossGrad_bnToyLoss] at h0
  rw [show sliceMean (dpToyShard 0) = 0 by simp [sliceMean, dpToyShard],
      show sliceMean (dpToyShard 1) = 2 by simp [sliceMean, dpToyShard],
      show sliceMean dpToyBatch = 1 by
        simp [sliceMean, dpToyBatch, Fin.sum_univ_two]] at h0
  norm_num at h0

-- ════════════════════════════════════════════════════════════════
-- § The lockstep induction
-- ════════════════════════════════════════════════════════════════

section Lockstep

variable {R P : Nat} {S : Type*}

/-- **One data-parallel step, as shipped.** Every replica computes its LOCAL gradient from its
    OWN state copy and its own slice; the collective averages them; every replica applies the
    same optimizer tail to its own state.

    `S` is the replica's whole state — parameters and optimizer moments — because the tail is
    stateful (Adam's two moments, momentum's velocity, EMA's shadow). Only the gradient is
    all-reduced, which is exactly what `emitGradAllReduce` emits. -/
noncomputable def dpStep (grad : Fin R → S → Vec P) (tail : S → Vec P → S)
    (st : Fin R → S) : Fin R → S :=
  fun r => tail (st r) (dpMean (fun s => grad s (st s)))

/-- **The single-device step the `R` replicas are running in lockstep**: one state, and the
    DP-mean gradient. -/
noncomputable def dpSingleStep (grad : Fin R → S → Vec P) (tail : S → Vec P → S) (st : S) : S :=
  tail st (dpMean (fun s => grad s st))

/-- From a shared state, one DP step lands every replica on the same state again — the induction
    step, and it is definitional: the all-reduced gradient does not depend on `r`, and neither
    does the state the tail is applied to. -/
theorem dpStep_const (grad : Fin R → S → Vec P) (tail : S → Vec P → S) (st : S) :
    dpStep grad tail (fun _ => st) = fun _ => dpSingleStep grad tail st := rfl

/-- ⭐⭐ **The lockstep induction.** Identical initial states and an identical (all-reduced)
    update keep the `R` copies equal at every step, so `n` steps of the `R`-replica system are
    `n` steps of the single-device one. This is the property `VerifiedTrain.lean` relies on
    when it checkpoints from replica 0 — the checkpoint is not "replica 0's answer", it is
    every replica's.

    ⚠ The shared start is a HYPOTHESIS (the `fun _ => st` on the left). That the driver
    broadcasts it, and that `replica_groups` names all `R` devices, is calling logic and the
    `*-dp-check` gates are its evidence. -/
theorem dpIterate_lockstep (grad : Fin R → S → Vec P) (tail : S → Vec P → S) :
    ∀ (n : Nat) (st : S),
      (dpStep grad tail)^[n] (fun _ => st) = fun _ => (dpSingleStep grad tail)^[n] st := by
  intro n
  induction n with
  | zero => intro st; rfl
  | succ n ih =>
    intro st
    rw [Function.iterate_succ_apply, Function.iterate_succ_apply, dpStep_const, ih]

end Lockstep

/-- ⭐⭐ **What function trained, in one statement.** When each replica's gradient is the
    certified gradient of its own loss, one data-parallel step is one ordinary step on the
    MEAN of the `R` per-replica losses. -/
theorem dpSingleStep_eq_meanLoss_step {R P : Nat} (L : Fin R → Vec P → ℝ)
    (tail : Vec P → Vec P → Vec P) (θ : Vec P)
    (hdiff : ∀ r, LossDifferentiableAt (L r) θ) :
    dpSingleStep (fun r θ' => lossGrad (L r) θ') tail θ = tail θ (lossGrad (meanLoss L) θ) := by
  show tail θ (dpMean (fun r => lossGrad (L r) θ)) = _
  rw [dpMeanGrad_eq_grad_meanLoss L θ hdiff]

/-- ⭐⭐ **`n` steps of `R` replicas ARE `n` steps of single-device training on the mean loss.**
    The lockstep induction and the gradient mean, composed — the closing statement of §4d
    piece 1, and the one that answers the disclaimer every tie in the repo carries.

    ⚠ Differentiability is asked for at EVERY point, not just at `θ`, because the trajectory
    passes through states the statement cannot name. For a relu net that is the one place this
    file is stronger than it needs to be: the honest weakening is differentiability along the
    trajectory, and it costs a mutual induction the payoff does not justify. -/
theorem dpIterate_eq_meanLossTrain {R P : Nat} (L : Fin R → Vec P → ℝ)
    (tail : Vec P → Vec P → Vec P)
    (hdiff : ∀ (θ : Vec P) (r : Fin R), LossDifferentiableAt (L r) θ) (n : Nat) (θ : Vec P) :
    (dpStep (fun r θ' => lossGrad (L r) θ') tail)^[n] (fun _ => θ)
      = fun _ => (fun θ' => tail θ' (lossGrad (meanLoss L) θ'))^[n] θ := by
  have hstep : dpSingleStep (fun r θ' => lossGrad (L r) θ') tail
             = fun θ' => tail θ' (lossGrad (meanLoss L) θ') := by
    funext θ'
    exact dpSingleStep_eq_meanLoss_step L tail θ' (fun r => hdiff θ' r)
  rw [dpIterate_lockstep, hstep]

end Proofs
