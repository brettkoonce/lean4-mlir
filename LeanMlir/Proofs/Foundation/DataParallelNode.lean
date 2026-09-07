import LeanMlir.Proofs.Foundation.DataParallel
import LeanMlir.Proofs.Foundation.ResNet34FaithfulPoCB

/-! # Data parallelism, piece 2: the collective as an AST node

`DataParallel.lean` (piece 1) is the ℝ-level half: `dpMean`, the replica mean of gradients, IS
the gradient of the mean of the per-replica losses, and that mean is the global-batch loss for a
net with no batch coupling and provably is not for a batch-BN net. What it could not say was
anything about the ARTIFACT, because the collective was `ViTRender.emitGradAllReduce` — emitted
text outside the `SHlo` AST, a declared trusted carve-out that every train-step tie in the repo
disclaimed in its own header.

Since 2026-09-07 the collective is `SHlo.allReduceMeanF`: `R` graphs of one skeleton (the same
program on `R` replicas, each with its own values — SPMD), reduced by `all_reduce(add)` and
divided by `R`. Its `den` is `(1/R) Σ_r den (g r)`, its `skel` reads replica 0, its emit is the
old text verbatim (`allReduceMeanText`), and the parser round-trip has its case
(`StableHLOParse.parseStack`). Every batched render now calls `prettyAllReduceMean` where it
called the text function, and every committed `*dp*` artifact re-rendered byte-identically.

This file is what the node BUYS, stated once:

* `den_allReduceMeanF_eq_dpMean` — the node denotes `dpMean` of its operands' denotations, which
  is the definition piece 1 is about.
* `skel_allReduceMeanF_of_spmd` — under the SPMD hypothesis (`∀ r, skel (g r) = skel (g 0)`,
  free in every render because a render's operand family is `.operand grad` at every `r`) the
  node's skeleton is each replica's, so `pretty` prints ONE program.
* `den_allReduceMeanF_eq_lossGrad_meanLoss` — piece 1 composed: if each replica's node denotes
  its own loss gradient, the all-reduced node denotes the gradient of the MEAN loss.
* `den_allReduceMeanF_convWeightGradB` — 4b's fold composed: the all-reduced conv weight-gradient
  node denotes the replica mean of the certified `Σ_n` gradients. One op kind shown; every other
  `*GradB` composes the same way, by `Finset.sum_congr` over the replicas and its own fold lemma.
* `adamW_at_allReduceMeanF` — the tail composed: `den (adamW tail (allReduceMeanF …))` is
  `adamWStep` at `dpMean` of the per-replica gradient nodes. That is
  `planning/proofs_tier_to_paper_nets.md` §4d piece 2's target statement.

## What is NOT claimed
⚠ Piece 3 is untouched: that the `R` graphs' values are the replica slices of ONE host batch,
that every replica starts from the same parameters and that `replica_groups` names all `R`
devices are the driver's (`VerifiedTrain.lean`, `ffi/pjrt_ffi.c`) and the `*-dp-check` gates'.
⚠ The per-replica gradient node's operand values differ per replica by construction; nothing
here says what they are. ⚠ The lowerer's `all_reduce` is trusted exactly as every other op's
lowering is.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs

open scoped BigOperators

/-- **The node denotes `dpMean` of its operands.** Definitional: `den`'s arm is piece 1's formula. -/
theorem den_allReduceMeanF_eq_dpMean {n : Nat} (R : Nat) (hR : 0 < R) (t : String)
    (ds : List Nat) (g : Fin R → SHlo n) :
    den (.allReduceMeanF R hR t ds g) = dpMean (fun r => den (g r)) := rfl

/-- **SPMD: the `R` graphs of one skeleton.** Under the hypothesis that every replica's operand
    graph has replica 0's skeleton — free in every render, whose operand family is `.operand grad`
    at every `r`, and the content of "the same program on `R` replicas" — the node's skeleton is
    each replica's, so `pretty` prints ONE program and `den` sums `R` of them. -/
theorem skel_allReduceMeanF_of_spmd {n : Nat} (R : Nat) (hR : 0 < R) (t : String)
    (ds : List Nat) (g : Fin R → SHlo n) (hsp : ∀ r, skel (g r) = skel (g ⟨0, hR⟩)) (r : Fin R) :
    skel (.allReduceMeanF R hR t ds g) = .allReduceMean R t ds (skel (g r)) := by
  show Raw.allReduceMean R t ds (skel (g ⟨0, hR⟩)) = _
  rw [hsp r]

/-- ⭐ **Piece 1 composed with the node.** If each replica's gradient node denotes the gradient of
    that replica's loss, the all-reduced node denotes the gradient of the MEAN of the per-replica
    losses — the function a data-parallel run minimises (`dpMeanGrad_eq_grad_meanLoss`). -/
theorem den_allReduceMeanF_eq_lossGrad_meanLoss {P : Nat} (R : Nat) (hR : 0 < R) (t : String)
    (ds : List Nat) (g : Fin R → SHlo P) (L : Fin R → Vec P → ℝ) (θ : Vec P)
    (hdiff : ∀ r, LossDifferentiableAt (L r) θ) (hg : ∀ r, den (g r) = lossGrad (L r) θ) :
    den (.allReduceMeanF R hR t ds g) = lossGrad (meanLoss L) θ := by
  rw [den_allReduceMeanF_eq_dpMean]
  have h : (fun r => den (g r)) = fun r => lossGrad (L r) θ := funext hg
  rw [h]
  exact dpMeanGrad_eq_grad_meanLoss L θ hdiff

/-- ⭐ **4b's fold composed with the node.** The all-reduced conv weight-gradient node denotes the
    replica mean of the certified batched `Σ_n` gradients, each at its own replica's activations
    and cotangent. One op kind; every other `*GradB` composes identically (`Finset.sum_congr`
    over the replicas, then its own fold lemma). -/
theorem den_allReduceMeanF_convWeightGradB {N ic oc h w kH kW : Nat} (R : Nat) (hR : 0 < R)
    (t : String) (ds : List Nat) (xN cotN : String) (b : Vec oc)
    (x : Fin R → Vec (N * (ic * h * w))) (W : Kernel4 oc ic kH kW)
    (cot : Fin R → Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)) :
    den (.allReduceMeanF R hR t ds
          (fun r => .convWeightGradB xN b (x r) W (.operand cotN (cot r)))) idx
      = (1 / (R : ℝ)) * ∑ r : Fin R, ∑ n : Fin N, ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b
                    (Tensor3.unflatten (batchSlice N (ic * h * w) (x r) n))))
               (Kernel4.flatten W) idx j * batchSlice N (oc * h * w) (cot r) n j := by
  simp only [den_allReduceMeanF]
  congr 1
  apply Finset.sum_congr rfl
  intro r _
  exact ResNet34PoCB.convWGradB_den xN cotN b (x r) W (cot r) idx

/-- ⭐⭐ **The optimizer tail at the all-reduced node** — §4d piece 2's target statement:
    `den (tail (allReduceMeanF R g))` is `adamWStep` at `dpMean` of the per-replica gradient
    nodes. `adamW_triple_faithful` is `∀ e`, so this is that theorem at `e := allReduceMeanF …`
    and the node's `den`; the same one line closes it for every other certified tail
    (`mom_pair_faithful`, `rmsProp_triple_faithful`, `lamb_triple_faithful`). -/
theorem adamW_at_allReduceMeanF {n : Nat}
    (θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN : String) (ds : List Nat)
    (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v : Vec n)
    (R : Nat) (hR : 0 < R) (t : String) (ds' : List Nat) (g : Fin R → SHlo n) :
    (den (.adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds
            β₁ β₂ ε lr wd bc₁ bc₂ θ m v (.allReduceMeanF R hR t ds' g)),
     den (.adamMNextF mN b1N ob1N ds β₁ m (.allReduceMeanF R hR t ds' g)),
     den (.adamVNextF vN b2N ob2N ds β₂ v (.allReduceMeanF R hR t ds' g)))
      = adamWStep β₁ β₂ ε lr wd bc₁ bc₂ θ m v (dpMean (fun r => den (g r))) := by
  rw [adamW_triple_faithful, den_allReduceMeanF_eq_dpMean]

end Proofs
