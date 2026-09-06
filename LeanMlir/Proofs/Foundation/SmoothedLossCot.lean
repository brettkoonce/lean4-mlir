import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Foundation.LinearTrainStep

/-! # The label-smoothed loss cotangent, at a GENERAL target

Every whole-net T3 tie in the repo pins its top-of-chain cotangent to
`softmax(logits) − oneHot label` — the gradient of plain cross-entropy at a hard label. That is
what the SGD-inline per-example renders emit, and it is not what the batched ImageNet renders emit.

`ResNet34RenderB.lean` composes the head cotangent from six kit ops:

```
%sm  = softmaxRow(logits)          %d0 = %sm − %onehot        %lsa = α · %onehot
%d1  = %d0 + %lsa                  %d2 = %d1 − α/K            %dy  = %d2 / B
```

so `dy = (softmax(z) − t + α·t − α/K) / B`, with `α = 0.1` baked (the `ls0` variants set it to 0)
and `t` arriving as the graph INPUT `%onehot` — which under mixup or cutmix is a soft vector drawn
on the host, not a one-hot. `ConvNeXtRenderB` and `ViTRenderB` compose the same six ops. This file
is the cotangent lemma those ties need: at a general target and at the smoothed form.

## What is proved

* `softCE K t z = Σ_k t_k · crossEntropy K z k` — cross-entropy against a **target
  distribution**, not a label. `softCE_grad` is its gradient, `(Σ_k t_k)·softmax(z)_j − t_j`,
  with no hypothesis on `t` at all: it is `softmaxCE_grad` under `pdiv_finset_sum`, and the
  familiar `softmax − t` is the `Σ t = 1` case.
* `smoothTarget K α t = (1−α)·t + α/K` is label smoothing as a map on targets, and it preserves
  `Σ = 1` (`smoothTarget_sum`, the one place `0 < K` is needed).
* ⭐ `smoothedCE_grad`: the gradient of `softCE` at the SMOOTHED target is exactly the expression
  the render emits — `softmax(z)_j − t_j + α·t_j − α/K`. So the six-op chain is not an
  approximation of the smoothed loss's gradient; it is that gradient, rearranged so that the
  smoothing is two extra elementwise ops on the target rather than a change to the target itself.
* `smoothedLossCotGraph` / `smoothedLossCotGraph_den` / `smoothedLossCotGraph_row`: the emitted
  graph, its denotation, and the per-example row of that denotation as the smoothed gradient
  divided by the batch.

## What is NOT claimed

⚠ **The `/ B` is the batch mean, and it is a convention, not a theorem here.** `smoothedLossCotGraph_row`
states the row IS `(1/B)·∂softCE/∂z` at that example's logits; that the sum of `B` such rows is the
gradient of the mean loss is the linearity step, and a tie against a `*dp*` artifact needs the
replica mean on top of it (`planning/proofs_tier_to_paper_nets.md` 4d).

⚠ **`α` is a free real.** The committed renders bake `0.1`, and the `ls0` twins bake `0`; both are
instances. ⚠ Nothing here says `t` is a probability vector — only `Σ t = 1` is ever used, which is
what mixup's convex combination of two one-hots satisfies.
-/

open Proofs Proofs.StableHLO

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Cross-entropy against a target DISTRIBUTION
-- ════════════════════════════════════════════════════════════════

/-- **Soft-target cross-entropy** `−Σ_k t_k log softmax(z)_k`, written as the `t`-weighted sum of
    the per-class `crossEntropy`. At `t = oneHot label` it IS `crossEntropy K z label`
    (`softCE_oneHot`). -/
noncomputable def softCE (K : Nat) (t z : Vec K) : ℝ :=
  ∑ k : Fin K, t k * crossEntropy K z k

/-- At a one-hot target, soft-target CE is the ordinary cross-entropy. -/
theorem softCE_oneHot (K : Nat) (z : Vec K) (label : Fin K) :
    softCE K (oneHot K label) z = crossEntropy K z label := by
  simp only [softCE, oneHot]
  rw [Finset.sum_eq_single label
      (fun k _ hk => by rw [if_neg hk]; ring)
      (fun h => absurd (Finset.mem_univ label) h)]
  rw [if_pos rfl, one_mul]

/-- **The soft-target CE gradient**, with NO hypothesis on `t`:
    `∂/∂z_j (−Σ_k t_k log p_k) = (Σ_k t_k)·p_j − t_j`. Each summand is `softmaxCE_grad`; the sum
    comes out by `pdiv_finset_sum`, and each `t_k` factor by `pdiv_mul` against a constant. -/
theorem softCE_grad (K : Nat) (t z : Vec K) (j : Fin K) :
    pdiv (fun z' : Vec K => fun _ : Fin 1 => softCE K t z') z j 0
      = (∑ k : Fin K, t k) * softmax K z j - t j := by
  have hterm : ∀ k : Fin K,
      pdiv (fun z' : Vec K => fun _ : Fin 1 => t k * crossEntropy K z' k) z j 0
        = t k * (softmax K z j - oneHot K k j) := by
    intro k
    have hc : DifferentiableAt ℝ (fun _ : Vec K => fun _ : Fin 1 => t k) z :=
      differentiableAt_const _
    have hd : DifferentiableAt ℝ (fun z' : Vec K => fun _ : Fin 1 => crossEntropy K z' k) z := by
      rw [differentiableAt_pi]
      intro _
      exact (crossEntropy_differentiable K k) z
    rw [pdiv_mul (fun _ : Vec K => fun _ : Fin 1 => t k)
          (fun z' : Vec K => fun _ : Fin 1 => crossEntropy K z' k) z hc hd j 0,
        pdiv_const (fun _ : Fin 1 => t k) z j 0, softmaxCE_grad K z k j]
    ring
  have hdiffs : ∀ k : Fin K, k ∈ (Finset.univ : Finset (Fin K)) →
      DifferentiableAt ℝ (fun z' : Vec K => fun _ : Fin 1 => t k * crossEntropy K z' k) z := by
    intro k _
    rw [differentiableAt_pi]
    intro _
    exact ((crossEntropy_differentiable K k) z).const_mul (t k)
  have hsum := pdiv_finset_sum (Finset.univ : Finset (Fin K))
    (fun k => fun z' : Vec K => fun _ : Fin 1 => t k * crossEntropy K z' k) z hdiffs j 0
  rw [show (fun (z' : Vec K) (_ : Fin 1) => softCE K t z')
        = (fun (z' : Vec K) (kk : Fin 1) =>
            ∑ k : Fin K, (fun k => fun z' : Vec K => fun _ : Fin 1 => t k * crossEntropy K z' k)
              k z' kk) from rfl, hsum]
  simp only [hterm, oneHot, mul_sub, mul_ite, mul_one, mul_zero]
  rw [Finset.sum_sub_distrib, ← Finset.sum_mul,
      Finset.sum_eq_single j
        (fun k _ hk => if_neg (fun h : j = k => hk h.symm))
        (fun h => absurd (Finset.mem_univ j) h),
      if_pos rfl]

-- ════════════════════════════════════════════════════════════════
-- § Label smoothing as a map on targets
-- ════════════════════════════════════════════════════════════════

/-- **Label smoothing**: `t ↦ (1−α)·t + α/K`, the target the smoothed loss is against. -/
noncomputable def smoothTarget (K : Nat) (α : ℝ) (t : Vec K) : Vec K :=
  fun k => (1 - α) * t k + α / K

/-- Smoothing preserves total mass. The one place `0 < K` is needed — with `K = 0` the `α/K`
    term is `α/0 = 0` and the identity fails for `α ≠ 0`. -/
theorem smoothTarget_sum (K : Nat) (hK : 0 < K) (α : ℝ) (t : Vec K) (ht : ∑ k : Fin K, t k = 1) :
    ∑ k : Fin K, smoothTarget K α t k = 1 := by
  have hKR : (K : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hK.ne'
  simp only [smoothTarget]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ht, mul_one, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp
  ring

/-- ⭐ **The emitted expression IS the smoothed loss's gradient.** `∂/∂z_j` of soft-target CE at
    the SMOOTHED target `(1−α)t + α/K` equals `softmax(z)_j − t_j + α·t_j − α/K`, which is exactly
    what the render's `softmaxRow → subB → scaleB → addVB → shiftB` chain computes, before the
    batch divide. -/
theorem smoothedCE_grad (K : Nat) (hK : 0 < K) (α : ℝ) (t z : Vec K)
    (ht : ∑ k : Fin K, t k = 1) (j : Fin K) :
    pdiv (fun z' : Vec K => fun _ : Fin 1 => softCE K (smoothTarget K α t) z') z j 0
      = softmax K z j - t j + α * t j - α / K := by
  rw [softCE_grad K (smoothTarget K α t) z j, smoothTarget_sum K hK α t ht, one_mul]
  simp only [smoothTarget]
  ring

-- ════════════════════════════════════════════════════════════════
-- § The emitted graph
-- ════════════════════════════════════════════════════════════════

/-- **The six-op label-smoothed cotangent chain the batched renders emit**, at one row per example
    (`m = 1`, `n = K`) and batch `N`. `logits` is the head's output and `t` the graph input
    `%onehot`; `α` is the smoothing and `B` the batch divisor (the render bakes `B = N`). -/
noncomputable def smoothedLossCotGraph (N K : Nat) (α B : ℝ) (aStr negAK bStr logN ohN : String)
    (logits t : Vec (N * (1 * K))) : SHlo (N * (1 * K)) :=
  .divConstB bStr B
    (.shiftB negAK (-(α / K))
      (.addVB
        (.subB (.batchOp (N := N) (.softmaxRow (m := 1) (n := K)) (.operand logN logits))
               (.operand ohN t))
        (.scaleB aStr α (.operand ohN t))))

/-- **What the chain denotes**, coordinatewise: `(rowSoftmax(logits) − t + α·t − α/K) / B`. -/
theorem smoothedLossCotGraph_den (N K : Nat) (α B : ℝ) (aStr negAK bStr logN ohN : String)
    (logits t : Vec (N * (1 * K))) (i : Fin (N * (1 * K))) :
    den (smoothedLossCotGraph N K α B aStr negAK bStr logN ohN logits t) i
      = (StableHLO.batchMap N (StableHLO.rowSoftmaxFlat 1 K) logits i - t i + t i * α
          + -(α / K)) / B := by
  simp only [smoothedLossCotGraph, den, denOp]

/-- ⭐ **Each row of the emitted cotangent is the smoothed loss's gradient at that example's
    logits, divided by the batch.** `Mat.unflatten` splits the flat `N·(1·K)` activation into its
    `N` per-example rows; `smoothedCE_grad` supplies the gradient. The hypothesis is only that the
    example's target sums to 1 — a one-hot, a mixup convex combination, or any distribution. -/
theorem smoothedLossCotGraph_row (N K : Nat) (hK : 0 < K) (α B : ℝ)
    (aStr negAK bStr logN ohN : String)
    (logits t : Vec (N * (1 * K))) (n : Fin N) (j : Fin K)
    (ht : ∑ k : Fin K, Mat.unflatten (StableHLO.batchSlice N (1 * K) t n) (0 : Fin 1) k = 1) :
    den (smoothedLossCotGraph N K α B aStr negAK bStr logN ohN logits t)
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = (pdiv (fun z' : Vec K => fun _ : Fin 1 =>
              softCE K (smoothTarget K α
                (Mat.unflatten (StableHLO.batchSlice N (1 * K) t n) (0 : Fin 1)))
              z')
            (Mat.unflatten (StableHLO.batchSlice N (1 * K) logits n) (0 : Fin 1)) j 0) / B := by
  have hsm : StableHLO.batchMap N (StableHLO.rowSoftmaxFlat 1 K) logits
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = softmax K (Mat.unflatten (StableHLO.batchSlice N (1 * K) logits n) (0 : Fin 1)) j := by
    simp only [StableHLO.batchMap, StableHLO.rowSoftmaxFlat, Mat.flatten,
      Equiv.symm_apply_apply]
    rfl
  have hti : t (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = Mat.unflatten (StableHLO.batchSlice N (1 * K) t n) (0 : Fin 1) j := by
    simp only [StableHLO.batchSlice, Mat.unflatten]
  rw [smoothedLossCotGraph_den, smoothedCE_grad K hK α _ _ ht j, hsm, hti]
  ring

end Proofs
