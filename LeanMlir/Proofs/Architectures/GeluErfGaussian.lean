import LeanMlir.Proofs.Architectures.GeluErf
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.CDF

/-!
# `gaussPhi` is the standard normal CDF

`GeluErf` writes the density and the CDF in closed form over the interval integral, so the
modules that import the exact GELU do not carry the probability library. This file ties both to
Mathlib's Gaussian:

- `gaussPdf_eq_gaussianPDFReal`: `gaussPdf` is `gaussianPDFReal 0 1`.
- `gaussPhi_eq_measureReal_Iic`, `gaussPhi_eq_cdf`: `gaussPhi x` is the mass the standard normal
  `gaussianReal 0 1` gives `(−∞, x]`, i.e. its `cdf`.

The `½` in `gaussPhi`'s definition is `integral_Iic_zero_gaussPdf`: the density is even and
integrates to one.
-/

open MeasureTheory ProbabilityTheory Set

namespace Proofs

/-- `gaussPdf` is Mathlib's Gaussian density at mean `0`, variance `1`. -/
theorem gaussPdf_eq_gaussianPDFReal (x : ℝ) : gaussPdf x = gaussianPDFReal 0 1 x := by
  unfold gaussPdf gaussianPDFReal
  simp only [NNReal.coe_one, mul_one, sub_zero]
  rw [div_eq_inv_mul]

/-- `gaussPdf` is integrable over the line. -/
theorem gaussPdf_integrable : Integrable gaussPdf := by
  rw [funext gaussPdf_eq_gaussianPDFReal]
  exact integrable_gaussianPDFReal 0 1

/-- **Half the mass lies below zero** — the density is even and integrates to one. -/
theorem integral_Iic_zero_gaussPdf : ∫ t in Iic (0 : ℝ), gaussPdf t = 1 / 2 := by
  have htot : ∫ t, gaussPdf t = 1 := by
    rw [funext gaussPdf_eq_gaussianPDFReal]
    exact integral_gaussianPDFReal_eq_one 0 one_ne_zero
  have hsplit := integral_add_compl (measurableSet_Iic (a := (0 : ℝ))) gaussPdf_integrable
  rw [compl_Iic, htot] at hsplit
  have heven : (fun t : ℝ => gaussPdf (-t)) = gaussPdf := by
    funext t; simp [gaussPdf]
  have hsym := integral_comp_neg_Ioi (0 : ℝ) gaussPdf
  rw [heven, neg_zero] at hsym
  linarith

/-- **`gaussPhi` is the standard normal's mass on `(−∞, x]`.** -/
theorem gaussPhi_eq_measureReal_Iic (x : ℝ) : gaussPhi x = (gaussianReal 0 1).real (Iic x) := by
  have hIic : ∫ t in Iic x, gaussPdf t = gaussPhi x := by
    have h := intervalIntegral.integral_Iic_sub_Iic (a := 0) (b := x)
      gaussPdf_integrable.integrableOn gaussPdf_integrable.integrableOn
    unfold gaussPhi
    rw [← h, integral_Iic_zero_gaussPdf]; ring
  rw [measureReal_def, gaussianReal_apply_eq_integral 0 one_ne_zero,
    ENNReal.toReal_ofReal, ← funext gaussPdf_eq_gaussianPDFReal, hIic]
  exact setIntegral_nonneg measurableSet_Iic fun t _ => gaussianPDFReal_nonneg 0 1 t

/-- **`gaussPhi` is the CDF of the standard normal**, as Mathlib defines both. -/
theorem gaussPhi_eq_cdf (x : ℝ) : gaussPhi x = cdf (gaussianReal 0 1) x := by
  rw [cdf_eq_real, gaussPhi_eq_measureReal_Iic]

end Proofs
