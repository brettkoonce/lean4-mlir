import LeanMlir.Proofs.Codegen.StableHLO

/-! # BCE-with-logits: the loss, and the cotangent the RSB renders emit

`SmoothedLossCot.lean` is this file's twin for the label-smoothed softmax cross-entropy every
other net trains on. ResNet-50's RSB-A2/A3 recipes use **binary cross-entropy with logits**
instead, and `ResNet50RenderB`'s `bce := true` path swaps the loss cotangent for a three-op chain:

```
%sm = sigmoidB(logits)      %d0 = %sm − %onehot      %dy = %d0 / (B·K)
```

so `dy = (σ(z) − t)/(B·K)`, against softmax-CE's five ops and `/B`. Nothing said that chain is a
loss's gradient — `planning/proofs_tier_to_paper_nets.md` §3.5 lists "BCE has no cotangent `den`"
as one of the two items ResNet-50's T3 needs first. This is that item.

## What is proved

* `softplus`, `softplus_hasDerivAt` — `log(1 + eᶻ)` and `d/dz softplus = σ(z)`. The renderer
  already computes the loss in this form (`%lsp = %lmax + %llg`, the stable
  `max(z,0) + log(1 + e^−|z|)`), so the ℝ definition is the reference's own spelling.
* ⭐⭐ `bceLogits_eq_logSigmoid` — and it IS binary cross-entropy: `softplus(z) − t·z` equals
  `−[t·log σ(z) + (1−t)·log(1 − σ(z))]`, class by class. **Without this the gradient theorem
  would be circular** — defining the loss as whatever has the wanted derivative and then proving
  it has it. The identity is what earns the name.
* `bceLogits_grad` — `∂/∂z_j Σ_k (softplus(z_k) − t_k·z_k) = σ(z_j) − t_j`, with NO hypothesis on
  `t` at all (it is per-class, so unlike softmax-CE nothing has to sum to 1 — which is the point
  of BCE under mixup, where the target is a sum of one-hots and can exceed 1 nowhere but need not
  be a distribution either).
* `bceLossCotGraph` / `_den` / `_row` — the emitted three-op chain, its denotation, and each row
  as that gradient at that example's logits, divided by the emitted constant.

## What is NOT claimed

⚠⚠ **The divisor is `B·K`, not `B`, and the theorem carries it as a binder rather than asserting
it.** `bceLossCotGraph_row_committed` pins it to `N·K`, which is what `ResNet50RenderB` bakes.
timm's `BinaryCrossEntropy` is `reduction='mean'` over `B×C`, not the mean of the per-example sum
over classes; at `K = 1000` the two differ by 1000× on the effective step, and RSB-A2's `lr 5e-3`
is tuned to this form. `bceLogits` is the per-example SUM over classes, so the `/K` half of the
divisor is what turns it into the mean.

⚠ **No label smoothing on this path, and that is the recipe.** timm's a3 arg string is `ls0.0`;
the soft targets reach `%onehot` from mixup/cutmix on the host, not from a smoothing constant.
So `smoothTarget` does not appear here and the render emits three ops where CE emits five.

⚠ **`%loss` itself is report-only.** The renderer's `lossCodeBce` block is hand-written text, not
`pretty` of an AST node (the §5 carve-out), and nothing here is about those lines. What is proved
is about the COTANGENT chain, which is on the gradient path and is `pretty(provenGraph)`.

⚠ **One replica**, as everywhere: under `*dp*` each gradient node is followed by
`all_reduce(add)/R` outside the AST (`DataParallel.lean`, §4d).
-/

open Finset BigOperators
open Proofs Proofs.StableHLO

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Softplus, and its derivative
-- ════════════════════════════════════════════════════════════════

/-- **Softplus**, `log(1 + eᶻ)`. The renderer computes it as `max(z,0) + log(1 + e^−|z|)`, which
    is this function written so that no positive number is ever exponentiated; that rearrangement
    is a floating-point concern and not a different map. -/
noncomputable def softplus (z : ℝ) : ℝ := Real.log (1 + Real.exp z)

/-- ⭐ **`d/dz softplus(z) = σ(z)`** — the one derivative BCE-with-logits needs, and the reason
    the loss is written in this form at all. -/
theorem softplus_hasDerivAt (z : ℝ) : HasDerivAt softplus (sigmoidScalar z) z := by
  have hpos : (0 : ℝ) < 1 + Real.exp z := by positivity
  have h : HasDerivAt (fun y : ℝ => 1 + Real.exp y) (Real.exp z) z :=
    (Real.hasDerivAt_exp z).const_add 1
  have hz : Real.exp z ≠ 0 := (Real.exp_pos z).ne'
  have heq : Real.exp z / (1 + Real.exp z) = sigmoidScalar z := by
    unfold sigmoidScalar
    rw [Real.exp_neg]
    field_simp
    ring
  exact heq ▸ h.log hpos.ne'

/-- `softplus(−z) = softplus(z) − z` — the identity that collapses the reference's two softplus
    calls into one, and the bridge to `log σ`. -/
theorem softplus_neg (z : ℝ) : softplus (-z) = softplus z - z := by
  have hz : Real.exp z ≠ 0 := (Real.exp_pos z).ne'
  have hs : (1 : ℝ) + Real.exp z ≠ 0 := by positivity
  have key : (1 : ℝ) + Real.exp (-z) = (1 + Real.exp z) / Real.exp z := by
    rw [Real.exp_neg]
    field_simp
    ring
  unfold softplus
  rw [key, Real.log_div hs hz, Real.log_exp]

/-- `log σ(z) = −softplus(−z)`. -/
theorem log_sigmoidScalar (z : ℝ) : Real.log (sigmoidScalar z) = -softplus (-z) := by
  have hpos : (0 : ℝ) < 1 + Real.exp (-z) := by positivity
  unfold sigmoidScalar softplus
  rw [one_div, Real.log_inv]

/-- `1 − σ(z) = σ(−z)`, hence `log(1 − σ(z)) = −softplus(z)`. -/
theorem one_sub_sigmoidScalar (z : ℝ) : 1 - sigmoidScalar z = sigmoidScalar (-z) := by
  have hz : Real.exp z ≠ 0 := (Real.exp_pos z).ne'
  have h1 : (1 : ℝ) + Real.exp (-z) ≠ 0 := by positivity
  have h2 : (1 : ℝ) + Real.exp z ≠ 0 := by positivity
  unfold sigmoidScalar
  rw [neg_neg, Real.exp_neg]
  field_simp
  ring

-- ════════════════════════════════════════════════════════════════
-- § The loss, and that it is binary cross-entropy
-- ════════════════════════════════════════════════════════════════

/-- **BCE-with-logits at one example**, `Σ_k (softplus(z_k) − t_k·z_k)` — the SUM over the `K`
    classes, in the stable form the renderer's own `%loss` block computes. The emitted cotangent
    divides by `B·K`, so the `/K` half of that divisor is what makes the shipped objective the
    MEAN over `B×K` rather than the mean of these sums (see the module header's ⚠⚠). -/
noncomputable def bceLogits (K : Nat) (t z : Vec K) : ℝ :=
  ∑ k : Fin K, (softplus (z k) - t k * z k)

/-- ⭐⭐ **It IS binary cross-entropy**: class by class, `softplus(z) − t·z` is
    `−[t·log σ(z) + (1−t)·log(1 − σ(z))]`.

    This is what earns `bceLogits` its name. Without it, `bceLogits_grad` would be circular —
    a function defined to have the derivative the render emits, then proved to have it. The
    proof is `softplus_neg` and the two `log σ` identities; nothing analytic. -/
theorem bceLogits_eq_logSigmoid (K : Nat) (t z : Vec K) :
    bceLogits K t z
      = ∑ k : Fin K, -(t k * Real.log (sigmoidScalar (z k))
                        + (1 - t k) * Real.log (1 - sigmoidScalar (z k))) := by
  unfold bceLogits
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [one_sub_sigmoidScalar, log_sigmoidScalar, log_sigmoidScalar, neg_neg, softplus_neg]
  ring

-- ════════════════════════════════════════════════════════════════
-- § The gradient
-- ════════════════════════════════════════════════════════════════


/-- ⭐ **The emitted cotangent's numerator is this loss's gradient**:
    `∂/∂z_j Σ_k (softplus(z_k) − t_k·z_k) = σ(z_j) − t_j`.

    ⚠ NO hypothesis on `t`. Softmax-CE's gradient needed `Σ_k t_k = 1` to collapse
    `(Σ t)·softmax − t`; BCE is per-class and separable, so the mixup targets — which are convex
    combinations of one-hots, and the a3 recipe's only source of soft labels — need no clause. -/
theorem bceLogits_grad (K : Nat) (t z : Vec K) (j : Fin K) :
    pdiv (fun z' : Vec K => fun _ : Fin 1 => bceLogits K t z') z j 0
      = sigmoidScalar (z j) - t j := by
  have hder : ∀ k : Fin K,
      HasDerivAt (fun r : ℝ => softplus r - t k * r) (sigmoidScalar (z k) - t k) (z k) := by
    intro k
    have h2 : HasDerivAt (fun r : ℝ => t k * r) (t k) (z k) := by
      simpa using (hasDerivAt_id (z k)).const_mul (t k)
    exact (softplus_hasDerivAt (z k)).sub h2
  have hterm : ∀ k : Fin K,
      pdiv (fun z' : Vec K => fun _ : Fin 1 => softplus (z' k) - t k * z' k) z j 0
        = if j = k then sigmoidScalar (z k) - t k else 0 := fun k =>
    pdiv_coordFun (fun r : ℝ => softplus r - t k * r) _ k z (hder k) j
  have hdiffs : ∀ k : Fin K, k ∈ (Finset.univ : Finset (Fin K)) →
      DifferentiableAt ℝ (fun z' : Vec K => fun _ : Fin 1 => softplus (z' k) - t k * z' k) z := by
    intro k _
    rw [differentiableAt_pi]
    intro _
    exact (hder k).differentiableAt.comp z
      (ContinuousLinearMap.proj k : Vec K →L[ℝ] ℝ).differentiableAt
  have hsum := pdiv_finset_sum (Finset.univ : Finset (Fin K))
    (fun k => fun z' : Vec K => fun _ : Fin 1 => softplus (z' k) - t k * z' k) z hdiffs j 0
  rw [show (fun (z' : Vec K) (_ : Fin 1) => bceLogits K t z')
        = (fun (z' : Vec K) (kk : Fin 1) =>
            ∑ k : Fin K,
              (fun k => fun z' : Vec K => fun _ : Fin 1 => softplus (z' k) - t k * z' k)
                k z' kk) from rfl, hsum]
  simp only [hterm]
  rw [Finset.sum_ite_eq (Finset.univ : Finset (Fin K)) j
        (fun k => sigmoidScalar (z k) - t k), if_pos (Finset.mem_univ j)]

-- ════════════════════════════════════════════════════════════════
-- § The emitted graph
-- ════════════════════════════════════════════════════════════════

/-- **The three-op BCE cotangent chain `ResNet50RenderB` emits under `bce := true`**, at one row
    per example (`m = 1`, `n = K`) and batch `N`. `logits` is the head's output, `t` the graph
    input `%onehot`, and `bk` the baked divisor.

    ⚠ The render emits the sigmoid at `(N := B, n := K)` and re-enters the subtraction through an
    `.operand` at `N*(1*K)`; the two indices are equal and NOT definitionally so at a variable
    `K`, which the render's own comment records. Stated here at the one index throughout, which
    is legitimate because `sigmoid` is elementwise and carries no row structure —
    `smoothedLossCotGraph` nests through the same `.operand` seam for the same reason. -/
noncomputable def bceLossCotGraph (N K : Nat) (bk : ℝ) (bStr logN ohN : String)
    (logits t : Vec (N * (1 * K))) : SHlo (N * (1 * K)) :=
  .divConstB bStr bk
    (.subB (.sigmoidB (N := N) (n := 1 * K) (.operand logN logits))
           (.operand ohN t))

/-- **What the chain denotes**, coordinatewise: `(σ(logits) − t) / bk`. -/
theorem bceLossCotGraph_den (N K : Nat) (bk : ℝ) (bStr logN ohN : String)
    (logits t : Vec (N * (1 * K))) (i : Fin (N * (1 * K))) :
    den (bceLossCotGraph N K bk bStr logN ohN logits t) i
      = (sigmoid (N * (1 * K)) logits i - t i) / bk := by
  simp only [bceLossCotGraph, den]

/-- ⭐ **Each row of the emitted cotangent is BCE-with-logits' gradient at that example's logits,
    divided by the baked constant.** `SmoothedLossCot`'s `smoothedLossCotGraph_row` at this loss,
    and with no hypothesis at all where that one needs the target's mass. -/
theorem bceLossCotGraph_row (N K : Nat) (bk : ℝ) (bStr logN ohN : String)
    (logits t : Vec (N * (1 * K))) (n : Fin N) (j : Fin K) :
    den (bceLossCotGraph N K bk bStr logN ohN logits t)
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = (pdiv (fun z' : Vec K => fun _ : Fin 1 =>
              bceLogits K (Mat.unflatten (StableHLO.batchSlice N (1 * K) t n) (0 : Fin 1)) z')
            (Mat.unflatten (StableHLO.batchSlice N (1 * K) logits n) (0 : Fin 1)) j 0) / bk := by
  have hsg : sigmoid (N * (1 * K)) logits
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = sigmoidScalar
          (Mat.unflatten (StableHLO.batchSlice N (1 * K) logits n) (0 : Fin 1) j) := by
    simp only [sigmoid, StableHLO.batchSlice, Mat.unflatten]
  have hti : t (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = Mat.unflatten (StableHLO.batchSlice N (1 * K) t n) (0 : Fin 1) j := by
    simp only [StableHLO.batchSlice, Mat.unflatten]
  rw [bceLossCotGraph_den, bceLogits_grad, hsg, hti]

/-- ⚠⚠ **The committed divisor is `N·K`, the mean over `B×K`.** `ResNet50RenderB` bakes
    `{B * nClasses}.0`; softmax-CE's peer bakes `{B}.0`. timm's `BinaryCrossEntropy` is
    `reduction='mean'` over `B×C`, not the mean of the per-example sum over classes, and at
    `K = 1000` the two differ by 1000× on the effective step — RSB-A2's `lr 5e-3` is tuned to
    this form. Stated separately from `bceLossCotGraph_row` so that the divisor is a checked
    fact about the artifact rather than a binder nobody instantiated. -/
theorem bceLossCotGraph_row_committed (N K : Nat) (bStr logN ohN : String)
    (logits t : Vec (N * (1 * K))) (n : Fin N) (j : Fin K) :
    den (bceLossCotGraph N K ((N : ℝ) * (K : ℝ)) bStr logN ohN logits t)
        (finProdFinEquiv (n, finProdFinEquiv ((0 : Fin 1), j)))
      = (pdiv (fun z' : Vec K => fun _ : Fin 1 =>
              bceLogits K (Mat.unflatten (StableHLO.batchSlice N (1 * K) t n) (0 : Fin 1)) z')
            (Mat.unflatten (StableHLO.batchSlice N (1 * K) logits n) (0 : Fin 1)) j 0)
          / ((N : ℝ) * (K : ℝ)) :=
  bceLossCotGraph_row N K ((N : ℝ) * (K : ℝ)) bStr logN ohN logits t n j

end Proofs
