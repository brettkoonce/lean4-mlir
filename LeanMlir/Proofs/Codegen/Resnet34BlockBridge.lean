import LeanMlir.Proofs.Codegen.BnInputBridge
import LeanMlir.Proofs.Float.Resnet34FloatBridge

/-!
# ℝ→Float32 bridge: assembling a ResNet block step

The first genuinely-composed block step: `relu(BN(·))` where the BN input is the
*perturbed* float activation (the output of an upstream float conv). This is where
the per-block composition becomes real — the BN closeness splits into

  rounding-at-fixed-input  (`bnForward_close_of`, the float BN's own roundoff)
  + input-shift            (`bnForward_input_close`, real BN moving with its input)

and `relu_close` carries it through the (exact-in-float) ReLU. `bnRelu_close`
proves exactly that. A full identity block `relu(F(x) + x)` is then this step
twice (drop the inner ReLU on the second) chained through `flatConvF_close` for
each conv and closed by `reluAdd_close` at the skip — the same parts, no new
numerical content.
-/

namespace Proofs

namespace FloatModel

variable (M : FloatModel)

/-- Budget of the BN→relu block step: the BN rounding (`bnNormBudget`) plus the
    real-BN input-shift from a per-coordinate input error `e1` at BN-input
    magnitude `A` (`δ ≤ e1` collapses the mean term). -/
noncomputable def bnReluBudget (u D S G Bbnd emean eistd A e1 ε : ℝ) : ℝ :=
  bnNormBudget u D S G Bbnd emean eistd
  + G * ((e1 + e1) * (1 / Real.sqrt ε) + 2 * A * ((8 * A * e1) / (2 * ε * Real.sqrt ε)))

/-- **BN forward block step closeness (no activation).** With the BN input `vt`
    (float) within `e1` of `va` (real) per coordinate (both magnitude `≤ A`), the
    float per-example BN mean/istd within `emean`/`eistd` of the real ones at `vt`,
    and the usual BN magnitude bounds, the rounded `bnForwardF vt` is within
    `bnReluBudget` of `bnForward va`. The composition split: rounding
    (`bnForward_close_of`) + input-shift (`bnForward_input_close`). This is the
    pre-activation bound; `bnRelu_close` is `relu_close` on top, and the BN-before-
    swish path in EfficientNet's MBConv uses this one directly. -/
theorem bnStep_close {n : Nat} {ε γ β emean eistd D S G Bbnd A e1 fμ fistdv : ℝ}
    (vt va : Vec n) (i : Fin n) (hn : 0 < n) (hε : 0 < ε)
    (he1 : ∀ k, |vt k - va k| ≤ e1)
    (hAvt : ∀ k, |vt k| ≤ A) (hAva : ∀ k, |va k| ≤ A)
    (hmean : |fμ - bnMean n vt| ≤ emean) (histd : |fistdv - bnIstd n vt ε| ≤ eistd)
    (hD : ∀ j, |vt j - bnMean n vt| ≤ D) (hSabs : |bnIstd n vt ε| ≤ S)
    (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd) :
    |M.bnForwardF γ β fμ fistdv vt i - bnForward n ε γ β va i| ≤
      bnReluBudget M.u D S G Bbnd emean eistd A e1 ε := by
  have hnR : (0:ℝ) < (n:ℝ) := by exact_mod_cast hn
  have hsε : 0 < Real.sqrt ε := Real.sqrt_pos.mpr hε
  have hA0 : 0 ≤ A := (abs_nonneg _).trans (hAvt ⟨0, hn⟩)
  have he10 : 0 ≤ e1 := (abs_nonneg _).trans (he1 ⟨0, hn⟩)
  -- δ = (Σ|vt−va|)/n ≤ e1
  have hδ : (∑ k, |vt k - va k|) / (n:ℝ) ≤ e1 := by
    rw [div_le_iff₀ hnR]
    calc ∑ k, |vt k - va k| ≤ ∑ _k : Fin n, e1 := Finset.sum_le_sum fun k _ => he1 k
      _ = e1 * (n:ℝ) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have hround := M.bnForward_close_of (ε := ε) vt i hmean histd (hD i) hSabs hγ hβ
  have hshift0 := bnForward_input_close (γ := γ) (β := β) (A := A) (ε := ε)
    vt va hn hε hAvt hAva i
  have hinner :
      (|vt i - va i| + (∑ k, |vt k - va k|) / (n:ℝ)) * (1 / Real.sqrt ε)
        + 2 * A * ((8 * A * ((∑ k, |vt k - va k|) / (n:ℝ))) / (2 * ε * Real.sqrt ε))
      ≤ (e1 + e1) * (1 / Real.sqrt ε) + 2 * A * ((8 * A * e1) / (2 * ε * Real.sqrt ε)) := by
    have hj := he1 i
    gcongr
  have hshift : |bnForward n ε γ β vt i - bnForward n ε γ β va i| ≤
      G * ((e1 + e1) * (1 / Real.sqrt ε) + 2 * A * ((8 * A * e1) / (2 * ε * Real.sqrt ε))) := by
    refine hshift0.trans ((mul_le_mul_of_nonneg_left hinner (abs_nonneg γ)).trans ?_)
    exact mul_le_mul_of_nonneg_right hγ (by positivity)
  calc |M.bnForwardF γ β fμ fistdv vt i - bnForward n ε γ β va i|
      ≤ |M.bnForwardF γ β fμ fistdv vt i - bnForward n ε γ β vt i|
        + |bnForward n ε γ β vt i - bnForward n ε γ β va i| := abs_sub_le _ _ _
    _ ≤ bnNormBudget M.u D S G Bbnd emean eistd
        + G * ((e1 + e1) * (1 / Real.sqrt ε)
               + 2 * A * ((8 * A * e1) / (2 * ε * Real.sqrt ε))) := add_le_add hround hshift
    _ = bnReluBudget M.u D S G Bbnd emean eistd A e1 ε := rfl

/-- **BN → relu block step closeness.** `relu_close` on top of `bnForward_close`:
    the rounded `relu(bnForwardF vt)` is within `bnReluBudget` of `relu(bnForward va)`
    (ReLU is exact in float and 1-Lipschitz, so it carries the pre-activation bound
    through unchanged). -/
theorem bnRelu_close {n : Nat} {ε γ β emean eistd D S G Bbnd A e1 fμ fistdv : ℝ}
    (vt va : Vec n) (i : Fin n) (hn : 0 < n) (hε : 0 < ε)
    (he1 : ∀ k, |vt k - va k| ≤ e1)
    (hAvt : ∀ k, |vt k| ≤ A) (hAva : ∀ k, |va k| ≤ A)
    (hmean : |fμ - bnMean n vt| ≤ emean) (histd : |fistdv - bnIstd n vt ε| ≤ eistd)
    (hD : ∀ j, |vt j - bnMean n vt| ≤ D) (hSabs : |bnIstd n vt ε| ≤ S)
    (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd) :
    |relu n (M.bnForwardF γ β fμ fistdv vt) i - relu n (bnForward n ε γ β va) i| ≤
      bnReluBudget M.u D S G Bbnd emean eistd A e1 ε :=
  relu_close (M.bnForwardF γ β fμ fistdv vt) (bnForward n ε γ β va)
    (bnReluBudget M.u D S G Bbnd emean eistd A e1 ε)
    (fun j => M.bnStep_close vt va j hn hε he1 hAvt hAva hmean histd hD hSabs hγ hβ) i

end FloatModel

end Proofs
