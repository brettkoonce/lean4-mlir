import LeanMlir.Proofs.Foundation.Tensor
import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-! # Inexact-gradient descent over ℝ

The theorem that makes the `FloatBridge` budgets *mean training*, not just
proximity: if the gradient oracle is within `η` of the true gradient and the
loss is smooth along the step segment, the SGD step still decreases the
loss — with an explicit decrease.

Shape, in suite style (everything coordinatewise over `Vec`, no inner-product
spaces): `gradAt f x i = fderiv ℝ f x (basisVec i)` is the scalar-loss
gradient entry — for the MLP loss these entries are exactly what the
`*_total_loss_grad` / `mlp_whole_net_weight_grads` theorems certify, and the
per-entry oracle error `η` is what `FloatBridge`'s `mulErr`/`cotErr` budgets
supply. The smoothness hypothesis is segment-local and coordinatewise
(`ℓ∞` on the gradient, `ℓ1` on the displacement): for ReLU nets it is the
descent-side cousin of the quantitative margins — the step segment must not
cross a kink.

Proved via the one-dimensional mean value theorem on `t ↦ f(x + t·d)` — no
integration; the price is the constant `C·D²` instead of the textbook
`C·D²/2`, immaterial for a descent guarantee.

`sgd_descends` is the quotable form: if `lr·η·‖∇f‖₁ ≤ lr·‖∇f‖₂²/4` and the
curvature term is similarly dominated, one inexact step decreases the loss
by at least `lr·‖∇f‖₂²/2`. Discharging its hypotheses for the concrete MNIST
nets (actual Lipschitz constants for the MLP loss) is future work; this file
is the ℝ-side keystone the float budgets plug into. -/

namespace Proofs

/-- Gradient entry of a scalar loss: `∂f/∂xᵢ` via the Mathlib `fderiv` —
    the scalar-codomain peer of `pdiv`. -/
noncomputable def gradAt {m : Nat} (f : Vec m → ℝ) (x : Vec m) (i : Fin m) :
    ℝ :=
  fderiv ℝ f x (basisVec i)

/-- Directional derivative = gradient contraction, coordinatewise. -/
theorem fderiv_apply_eq_sum_grad {m : Nat} (f : Vec m → ℝ) (x : Vec m)
    (d : Vec m) :
    fderiv ℝ f x d = ∑ i, d i * gradAt f x i := by
  have hd : d = ∑ i, d i • basisVec i := by
    funext k
    rw [Finset.sum_apply]
    simp only [Pi.smul_apply, basisVec, smul_eq_mul, mul_ite, mul_one,
      mul_zero]
    rw [Finset.sum_ite_eq]
    simp
  calc fderiv ℝ f x d = fderiv ℝ f x (∑ i, d i • basisVec i) := by rw [← hd]
    _ = ∑ i, fderiv ℝ f x (d i • basisVec i) := map_sum _ _ _
    _ = ∑ i, d i * gradAt f x i :=
        Finset.sum_congr rfl fun i _ => by
          rw [map_smul]; simp [gradAt, smul_eq_mul]

/-- **Descent lemma along a segment** (MVT form). If `f` is differentiable
    on the segment `[x, x+d]` and its gradient entries drift by at most
    `C·(t·D)` along it (`D` an `ℓ1` bound on `d`), then
    `f(x+d) ≤ f(x) + ⟨d, ∇f(x)⟩ + C·D²`. -/
theorem descent_segment {m : Nat} (f : Vec m → ℝ) (x d : Vec m) {C D : ℝ}
    (hC : 0 ≤ C)
    (hdiff : ∀ t ∈ Set.Icc (0:ℝ) 1, DifferentiableAt ℝ f (x + t • d))
    (hD : (∑ j, |d j|) ≤ D)
    (hLip : ∀ t ∈ Set.Icc (0:ℝ) 1, ∀ i,
      |gradAt f (x + t • d) i - gradAt f x i| ≤ C * (t * D)) :
    f (x + d) ≤ f x + (∑ i, d i * gradAt f x i) + C * D ^ 2 := by
  have hD0 : 0 ≤ D :=
    le_trans (Finset.sum_nonneg fun j _ => abs_nonneg _) hD
  -- the path t ↦ x + t·d and its derivative
  have hpath : ∀ t : ℝ, HasDerivAt (fun s : ℝ => x + s • d) d t := by
    intro t
    have h1 : HasDerivAt (fun s : ℝ => s • d) ((1:ℝ) • d) t :=
      (hasDerivAt_id t).smul_const d
    simpa using h1.const_add x
  have hφd : ∀ t ∈ Set.Icc (0:ℝ) 1,
      HasDerivAt (fun s : ℝ => f (x + s • d))
        (fderiv ℝ f (x + t • d) d) t := fun t ht =>
    ((hdiff t ht).hasFDerivAt).comp_hasDerivAt t (hpath t)
  have hφ' : ∀ t ∈ Set.Ioo (0:ℝ) 1,
      HasDerivAt (fun s : ℝ => f (x + s • d))
        (fderiv ℝ f (x + t • d) d) t := fun t ht =>
    hφd t (Set.Ioo_subset_Icc_self ht)
  have hφc : ContinuousOn (fun s : ℝ => f (x + s • d)) (Set.Icc 0 1) :=
    fun t ht => (hφd t ht).continuousAt.continuousWithinAt
  -- mean value theorem at an interior point c
  obtain ⟨c, hc, hceq⟩ := exists_hasDerivAt_eq_slope
    (fun s : ℝ => f (x + s • d))
    (fun t => fderiv ℝ f (x + t • d) d) (by norm_num) hφc hφ'
  have hkey : f (x + d) - f x = fderiv ℝ f (x + c • d) d := by
    have h := hceq.symm
    simp only [one_smul, zero_smul, add_zero, sub_zero, div_one] at h
    exact h
  have hcmem : c ∈ Set.Icc (0:ℝ) 1 := Set.Ioo_subset_Icc_self hc
  -- expand the directional derivative at c and split off the drift
  have hsplit : fderiv ℝ f (x + c • d) d =
      (∑ i, d i * gradAt f x i) +
        ∑ i, d i * (gradAt f (x + c • d) i - gradAt f x i) := by
    rw [fderiv_apply_eq_sum_grad, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  have herr : (∑ i, d i * (gradAt f (x + c • d) i - gradAt f x i)) ≤
      C * D ^ 2 := by
    have hc0 : 0 ≤ c := le_of_lt hc.1
    have hc1 : c ≤ 1 := le_of_lt hc.2
    calc (∑ i, d i * (gradAt f (x + c • d) i - gradAt f x i))
        ≤ |∑ i, d i * (gradAt f (x + c • d) i - gradAt f x i)| :=
          le_abs_self _
      _ ≤ ∑ i, |d i * (gradAt f (x + c • d) i - gradAt f x i)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i, |d i| * (C * (c * D)) := by
          refine Finset.sum_le_sum fun i _ => ?_
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_left (hLip c hcmem i) (abs_nonneg _)
      _ = (∑ i, |d i|) * (C * (c * D)) := by rw [Finset.sum_mul]
      _ ≤ D * (C * (1 * D)) := by
          have h1 : C * (c * D) ≤ C * (1 * D) := by
            nlinarith [mul_nonneg (mul_nonneg hC hD0) (sub_nonneg.mpr hc1)]
          have h2 : (0:ℝ) ≤ C * (c * D) :=
            mul_nonneg hC (mul_nonneg hc0 hD0)
          exact mul_le_mul hD h1 h2 hD0
      _ = C * D ^ 2 := by ring
  have hfinal : f (x + d) - f x =
      (∑ i, d i * gradAt f x i) +
        ∑ i, d i * (gradAt f (x + c • d) i - gradAt f x i) := by
    rw [hkey, hsplit]
  linarith

/-- **One inexact SGD step, explicit quadratic bound.** With a gradient
    oracle `gh` within `η` of `∇f(x)` coordinatewise, step `x − lr·gh`, and
    segment smoothness at the `ℓ1` step radius `lr·(‖∇f‖₁ + m·η)`:

    `f(x − lr·gh) ≤ f(x) − lr·‖∇f‖₂² + lr·η·‖∇f‖₁ + C·(lr·(‖∇f‖₁ + m·η))²`.

    The three terms: full descent, the oracle-error tax, the curvature tax.
    `FloatBridge` supplies `η` (per-entry certified-gradient budgets); the
    smoothness hypothesis is the user's, segment-local — for ReLU nets it
    encodes that the step doesn't cross a kink. -/
theorem sgd_descent_inexact {m : Nat} (f : Vec m → ℝ) (x gh : Vec m)
    {lr η C : ℝ} (hlr : 0 ≤ lr) (hη : 0 ≤ η) (hC : 0 ≤ C)
    (hgh : ∀ i, |gh i - gradAt f x i| ≤ η)
    (hdiff : ∀ t ∈ Set.Icc (0:ℝ) 1,
      DifferentiableAt ℝ f (x + t • (-(lr • gh))))
    (hLip : ∀ t ∈ Set.Icc (0:ℝ) 1, ∀ i,
      |gradAt f (x + t • (-(lr • gh))) i - gradAt f x i| ≤
        C * (t * (lr * ((∑ j, |gradAt f x j|) + m * η)))) :
    f (x - lr • gh) ≤
      f x - lr * (∑ i, gradAt f x i ^ 2)
        + lr * η * (∑ i, |gradAt f x i|)
        + C * (lr * ((∑ j, |gradAt f x j|) + m * η)) ^ 2 := by
  -- ℓ1 bound on the step
  have hD : (∑ j, |(-(lr • gh)) j|) ≤
      lr * ((∑ j, |gradAt f x j|) + m * η) := by
    calc (∑ j, |(-(lr • gh)) j|) = ∑ j, lr * |gh j| := by
          refine Finset.sum_congr rfl fun j _ => ?_
          simp [abs_mul, abs_of_nonneg hlr]
      _ ≤ ∑ j, lr * (|gradAt f x j| + η) := by
          refine Finset.sum_le_sum fun j _ => ?_
          refine mul_le_mul_of_nonneg_left ?_ hlr
          have h1 : |gh j| ≤ |gh j - gradAt f x j| + |gradAt f x j| := by
            simpa using abs_sub_le (gh j) (gradAt f x j) 0
          linarith [hgh j]
      _ = lr * ((∑ j, |gradAt f x j|) + m * η) := by
          rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const,
            Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  -- inner-product bound: ⟨d, ∇f⟩ ≤ −lr·‖∇f‖₂² + lr·η·‖∇f‖₁
  have hinner : (∑ i, (-(lr • gh)) i * gradAt f x i) ≤
      -(lr * ∑ i, gradAt f x i ^ 2)
        + lr * η * ∑ i, |gradAt f x i| := by
    have h1 : ∀ i, (-(lr • gh)) i * gradAt f x i ≤
        -(lr * gradAt f x i ^ 2) + lr * η * |gradAt f x i| := by
      intro i
      have h3 := abs_le.mp (hgh i)
      have h4 : -(η * |gradAt f x i|) ≤
          (gh i - gradAt f x i) * gradAt f x i := by
        rcases abs_cases (gradAt f x i) with ⟨ha, _⟩ | ⟨ha, _⟩ <;>
          nlinarith [h3.1, h3.2]
      have h5 := mul_le_mul_of_nonneg_left h4 hlr
      have h6 : (-(lr • gh)) i = -(lr * gh i) := by simp
      rw [h6]
      nlinarith [h5]
    calc (∑ i, (-(lr • gh)) i * gradAt f x i)
        ≤ ∑ i, (-(lr * gradAt f x i ^ 2) + lr * η * |gradAt f x i|) :=
          Finset.sum_le_sum fun i _ => h1 i
      _ = -(lr * ∑ i, gradAt f x i ^ 2)
            + lr * η * ∑ i, |gradAt f x i| := by
          rw [Finset.sum_add_distrib]
          congr 1
          · rw [Finset.sum_neg_distrib, ← Finset.mul_sum]
          · rw [← Finset.mul_sum]
  have hmain := descent_segment f x (-(lr • gh)) hC hdiff hD hLip
  have hstep : x - lr • gh = x + -(lr • gh) := by rw [sub_eq_add_neg]
  rw [hstep]
  linarith

/-- **Strict descent under explicit dominance.** If the oracle-error tax and
    the curvature tax are each at most a quarter of the full descent, one
    inexact SGD step decreases the loss by at least `lr·‖∇f‖₂²/2`. -/
theorem sgd_descends {m : Nat} (f : Vec m → ℝ) (x gh : Vec m)
    {lr η C : ℝ} (hlr : 0 ≤ lr) (hη : 0 ≤ η) (hC : 0 ≤ C)
    (hgh : ∀ i, |gh i - gradAt f x i| ≤ η)
    (hdiff : ∀ t ∈ Set.Icc (0:ℝ) 1,
      DifferentiableAt ℝ f (x + t • (-(lr • gh))))
    (hLip : ∀ t ∈ Set.Icc (0:ℝ) 1, ∀ i,
      |gradAt f (x + t • (-(lr • gh))) i - gradAt f x i| ≤
        C * (t * (lr * ((∑ j, |gradAt f x j|) + m * η))))
    (h1 : lr * η * (∑ i, |gradAt f x i|) ≤
      lr * (∑ i, gradAt f x i ^ 2) / 4)
    (h2 : C * (lr * ((∑ j, |gradAt f x j|) + m * η)) ^ 2 ≤
      lr * (∑ i, gradAt f x i ^ 2) / 4) :
    f (x - lr • gh) ≤ f x - lr * (∑ i, gradAt f x i ^ 2) / 2 := by
  have := sgd_descent_inexact f x gh hlr hη hC hgh hdiff hLip
  linarith

end Proofs
