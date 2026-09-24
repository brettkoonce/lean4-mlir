import LeanMlir.Proofs.Architectures.BatchNorm
import LeanMlir.Proofs.Float.FloatBridge

/-!
# ℝ→Float32 bridge for BatchNorm: the inverse-stddev keystone

The no-BN CIFAR bridge (`CifarFloatBridge.lean`) reuses the existing relative-error
model over sums/products/exact-max. BatchNorm adds the one genuinely new numerical
op: the **inverse standard deviation** `istd = 1/√(σ²+ε)`. The relative-error model
`|rnd x − x| ≤ u·|x|` does not cover `rsqrt` (a GPU `rsqrt`, like `exp`, has no IEEE
spec), so — exactly as the softmax bridge models `exp` by a supplied `fexp` with an
`eexp` accuracy hypothesis — we model the float inverse-stddev by a supplied
`fistd : ℝ → ℝ` with a *relative* accuracy `ers`.

The keystone is that `t ↦ 1/√t` is Lipschitz on `[ε, ∞)` with constant `1/(2ε√ε)`
(`rsqrt_lipschitz`, proved by the algebraic identity
`1/√a − 1/√b = (b−a)/((√a+√b)·√a·√b)` and the `ε`-floor). Composing the `rsqrt`
accuracy with this Lipschitz bound gives `bnIstd_close`: the float `istd` is within
`ers/√ε + e_var/(2ε√ε)` of the certified `bnIstd`, where `e_var` is whatever budget
the (standard, Higham) variance rounding supplies. This is the BN analog of the
`exp` accuracy handoff — the piece that the full per-example `bnForward` rounding
budget composes from (mean/var rounding + the normalize-stage products remain the
mechanical tail).
-/

namespace Proofs

open scoped Real

/-- `t ↦ 1/√t` is Lipschitz on `[ε, ∞)` with constant `1/(2·ε·√ε) = 1/(2ε^{3/2})`.
    Proof: the algebraic identity `1/√a − 1/√b = (b−a)/((√b+√a)·√a·√b)`, then the
    `ε`-floor bounds the denominator below by `2ε√ε`. The keystone for pushing a
    variance-rounding error through the BN inverse-stddev. -/
theorem rsqrt_lipschitz {a b ε : ℝ} (hε : 0 < ε) (ha : ε ≤ a) (hb : ε ≤ b) :
    |1 / Real.sqrt a - 1 / Real.sqrt b| ≤ |a - b| / (2 * ε * Real.sqrt ε) := by
  have hε0 : 0 ≤ ε := le_of_lt hε
  have ha0 : 0 ≤ a := hε0.trans ha
  have hb0 : 0 ≤ b := hε0.trans hb
  have hsε : 0 < Real.sqrt ε := Real.sqrt_pos.mpr hε
  have hsa : Real.sqrt ε ≤ Real.sqrt a := Real.sqrt_le_sqrt ha
  have hsb : Real.sqrt ε ≤ Real.sqrt b := Real.sqrt_le_sqrt hb
  have hsa0 : 0 < Real.sqrt a := lt_of_lt_of_le hsε hsa
  have hsb0 : 0 < Real.sqrt b := lt_of_lt_of_le hsε hsb
  have haa : Real.sqrt a * Real.sqrt a = a := Real.mul_self_sqrt ha0
  have hbb : Real.sqrt b * Real.sqrt b = b := Real.mul_self_sqrt hb0
  have hεε : Real.sqrt ε * Real.sqrt ε = ε := Real.mul_self_sqrt hε0
  have hdiff : (Real.sqrt b - Real.sqrt a) * (Real.sqrt b + Real.sqrt a) = b - a := by
    have h : (Real.sqrt b - Real.sqrt a) * (Real.sqrt b + Real.sqrt a)
        = Real.sqrt b * Real.sqrt b - Real.sqrt a * Real.sqrt a := by ring
    rw [h, hbb, haa]
  have key : 1 / Real.sqrt a - 1 / Real.sqrt b
      = (b - a) / ((Real.sqrt b + Real.sqrt a) * (Real.sqrt a * Real.sqrt b)) := by
    rw [← hdiff]; field_simp
  rw [key, abs_div]
  have hDpos : 0 < (Real.sqrt b + Real.sqrt a) * (Real.sqrt a * Real.sqrt b) := by positivity
  rw [abs_of_pos hDpos, abs_sub_comm b a]
  have hprod : Real.sqrt a * Real.sqrt b ≥ ε := by
    calc Real.sqrt a * Real.sqrt b ≥ Real.sqrt ε * Real.sqrt ε :=
          mul_le_mul hsa hsb (le_of_lt hsε) (le_of_lt hsa0)
      _ = ε := hεε
  have hsum : Real.sqrt b + Real.sqrt a ≥ 2 * Real.sqrt ε := by linarith
  have hDlow : 2 * ε * Real.sqrt ε ≤
      (Real.sqrt b + Real.sqrt a) * (Real.sqrt a * Real.sqrt b) := by
    calc 2 * ε * Real.sqrt ε = (2 * Real.sqrt ε) * ε := by ring
      _ ≤ (Real.sqrt b + Real.sqrt a) * (Real.sqrt a * Real.sqrt b) :=
          mul_le_mul hsum hprod (le_of_lt hε) (by positivity)
  gcongr

/-- **BN inverse-stddev budget at the OPERATING POINT (a-posteriori).** The general
    form of `bnIstd_close` (below): the `1/√` Lipschitz floor is a *variance lower bound*
    `V ≤ σ²+ε` (both float and real), not the `ε`-floor. `rsqrt_lipschitz` is
    floor-agnostic, so the bound becomes `ers/√V + evar/(2V√V)` — and since the
    measured `σ²` is `O(1)` (never near 0), `V ≈ σ²+ε ≫ ε` makes this ~`(σ²/ε)^{3/2}`
    tighter than the `ε`-floor `bnIstd_close` (empirically ~10⁷× on the CIFAR-BN
    probe, `scripts/cifar_bn_margin_probe.py`). The non-vacuous BN certificate. -/
theorem bnIstd_close_at {n : ℕ} {ε ers evar fvarε V : ℝ} (x : Vec n)
    (fistd : ℝ → ℝ) (hV0 : 0 < V) (hers : 0 ≤ ers)
    (hVfv : V ≤ fvarε) (hVbn : V ≤ bnVar n x + ε)
    (hrs : |fistd fvarε - 1 / Real.sqrt fvarε| ≤ ers * (1 / Real.sqrt fvarε))
    (hclose : |fvarε - (bnVar n x + ε)| ≤ evar) :
    |fistd fvarε - bnIstd n x ε| ≤
      ers / Real.sqrt V + evar / (2 * V * Real.sqrt V) := by
  have hsV : 0 < Real.sqrt V := Real.sqrt_pos.mpr hV0
  -- term 1: rsqrt accuracy, lifted from 1/√fvarε up to 1/√V (fvarε ≥ V)
  have hinv : 1 / Real.sqrt fvarε ≤ 1 / Real.sqrt V :=
    one_div_le_one_div_of_le hsV (Real.sqrt_le_sqrt hVfv)
  have ht1 : |fistd fvarε - 1 / Real.sqrt fvarε| ≤ ers / Real.sqrt V := by
    calc |fistd fvarε - 1 / Real.sqrt fvarε| ≤ ers * (1 / Real.sqrt fvarε) := hrs
      _ ≤ ers * (1 / Real.sqrt V) := mul_le_mul_of_nonneg_left hinv hers
      _ = ers / Real.sqrt V := by rw [mul_one_div]
  -- term 2: the variance shift through rsqrt_lipschitz at floor V (not ε)
  have ht2 : |1 / Real.sqrt fvarε - bnIstd n x ε| ≤ evar / (2 * V * Real.sqrt V) := by
    unfold bnIstd
    calc |1 / Real.sqrt fvarε - 1 / Real.sqrt (bnVar n x + ε)|
        ≤ |fvarε - (bnVar n x + ε)| / (2 * V * Real.sqrt V) := rsqrt_lipschitz hV0 hVfv hVbn
      _ ≤ evar / (2 * V * Real.sqrt V) := by gcongr
  calc |fistd fvarε - bnIstd n x ε|
      ≤ |fistd fvarε - 1 / Real.sqrt fvarε| + |1 / Real.sqrt fvarε - bnIstd n x ε| :=
        abs_sub_le _ _ _
    _ ≤ ers / Real.sqrt V + evar / (2 * V * Real.sqrt V) := add_le_add ht1 ht2

/-- **BN inverse-stddev rounding budget (the keystone).** Model the GPU
    inverse-stddev by a supplied `fistd` with relative accuracy `ers`
    (`|fistd t − 1/√t| ≤ ers·(1/√t)`), evaluated at the rounded `fvarε ≥ ε`
    (the float `σ²+ε`, within `evar` of the real `σ²+ε`). Then the float istd is
    within `ers/√ε + evar/(2ε√ε)` of the certified `bnIstd`. The first term is
    the `rsqrt` accuracy lifted to the `ε`-floor; the second is the variance
    rounding pushed through `rsqrt_lipschitz`. The BN analog of the softmax
    `exp`-accuracy handoff; `bnIstd_close_at` at the floor `V = ε`. -/
theorem bnIstd_close {n : ℕ} {ε ers evar fvarε : ℝ} (x : Vec n)
    (fistd : ℝ → ℝ) (hε : 0 < ε) (hers : 0 ≤ ers) (hfv : ε ≤ fvarε)
    (hrs : |fistd fvarε - 1 / Real.sqrt fvarε| ≤ ers * (1 / Real.sqrt fvarε))
    (hclose : |fvarε - (bnVar n x + ε)| ≤ evar) :
    |fistd fvarε - bnIstd n x ε| ≤
      ers / Real.sqrt ε + evar / (2 * ε * Real.sqrt ε) :=
  bnIstd_close_at x fistd hε hers hfv (by linarith [bnVar_nonneg n x]) hrs hclose

-- ════════════════════════════════════════════════════════════════
-- § The normalize chain: BN forward closeness given mean + istd errors
-- ════════════════════════════════════════════════════════════════

/-- **The float BN forward** (per-example, supplied mean `fμ` and inverse-stddev
    `fistdv`): `yᵢ = fl(γ ⊙ fl(fl(xᵢ ⊖ fμ) ⊙ fistdv) ⊕ β)` — every `+`/`−`/`·`
    rounded; `γ`/`β` are the stored (exact) parameters. -/
noncomputable def FloatModel.bnForwardF {n : Nat} (M : FloatModel)
    (γ β fμ fistdv : ℝ) (x : Vec n) : Vec n :=
  fun i => M.add (M.mul γ (M.mul (M.sub (x i) fμ) fistdv)) β

/-- The closed-form budget of the BN normalize chain: a `sub` rounding, two
    `mul`s (`mulErr`), and the final `add` rounding, threading the upstream mean
    error `emean` and inverse-stddev error `eistd`. -/
noncomputable def bnNormBudget (u D S G Bbnd emean eistd : ℝ) : ℝ :=
  u * (G * (D * S)
        + FloatModel.mulErr u G (D * S) 0
            (FloatModel.mulErr u D S (u * (D + emean) + emean) eistd)
        + Bbnd)
    + FloatModel.mulErr u G (D * S) 0
        (FloatModel.mulErr u D S (u * (D + emean) + emean) eistd)

/-- **BN forward closeness, normalize chain.** Given the mean within `emean` and
    the inverse-stddev within `eistd` of the certified values (the latter from
    `bnIstd_close`), plus magnitude bounds (`|xᵢ−μ| ≤ D`, `|istd| ≤ S`, `|γ| ≤ G`,
    `|β| ≤ Bbnd`), the rounded BN forward output is within `bnNormBudget` of the
    real `bnForward` per coordinate. The output half of the per-example BN bridge;
    compose with `bnIstd_close` (istd) and the mean/var Higham budgets. -/
theorem FloatModel.bnForward_close_of {n : Nat} (M : FloatModel)
    {ε γ β fμ fistdv emean eistd D S G Bbnd : ℝ} (x : Vec n) (i : Fin n)
    (hmean : |fμ - bnMean n x| ≤ emean)
    (histd : |fistdv - bnIstd n x ε| ≤ eistd)
    (hD : |x i - bnMean n x| ≤ D) (hSabs : |bnIstd n x ε| ≤ S)
    (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd) :
    |M.bnForwardF γ β fμ fistdv x i - bnForward n ε γ β x i| ≤
      bnNormBudget M.u D S G Bbnd emean eistd := by
  set μ := bnMean n x with hμ
  set istd := bnIstd n x ε with histddef
  have hD0 : 0 ≤ D := (abs_nonneg _).trans hD
  -- stage 1: centered  fl(xᵢ ⊖ fμ) ≈ xᵢ − μ
  have hs1 : |M.sub (x i) fμ - (x i - μ)| ≤ M.u * (D + emean) + emean :=
    M.rnd_close (by rw [show x i - fμ - (x i - μ) = μ - fμ by ring, abs_sub_comm]; exact hmean)
      ((abs_sub_le _ μ _).trans (add_le_add hD (by rwa [abs_sub_comm])))
  -- stage 2: x̂  fl(centered ⊙ fistdv) ≈ (xᵢ − μ)·istd
  have hs2 : |M.mul (M.sub (x i) fμ) fistdv - (x i - μ) * istd| ≤
      FloatModel.mulErr M.u D S (M.u * (D + emean) + emean) eistd :=
    M.mul_close hs1 histd hD hSabs
  -- stage 3: γ·x̂  fl(γ ⊙ x̂) ≈ γ·((xᵢ − μ)·istd)
  have hxhat : |(x i - μ) * istd| ≤ D * S := by
    rw [abs_mul]; exact mul_le_mul hD hSabs (abs_nonneg _) hD0
  have hs3 : |M.mul γ (M.mul (M.sub (x i) fμ) fistdv) - γ * ((x i - μ) * istd)| ≤
      FloatModel.mulErr M.u G (D * S) 0
        (FloatModel.mulErr M.u D S (M.u * (D + emean) + emean) eistd) :=
    M.mul_close (by simp) hs2 hγ hxhat
  -- stage 4: + β and assemble
  set es3 := FloatModel.mulErr M.u G (D * S) 0
      (FloatModel.mulErr M.u D S (M.u * (D + emean) + emean) eistd) with hes3
  set s3 := M.mul γ (M.mul (M.sub (x i) fμ) fistdv) with hs3def
  have hgxhat : |γ * ((x i - μ) * istd)| ≤ G * (D * S) := by
    rw [abs_mul]; exact mul_le_mul hγ hxhat (abs_nonneg _) ((abs_nonneg _).trans hγ)
  have hs3mag : |s3| ≤ G * (D * S) + es3 := by
    linarith [abs_sub_abs_le_abs_sub s3 (γ * ((x i - μ) * istd))]
  have hsumβ : |s3 + β| ≤ G * (D * S) + es3 + Bbnd := (abs_add_le _ _).trans (add_le_add hs3mag hβ)
  simp only [FloatModel.bnForwardF, bnForward, bnXhat, ← hμ, ← histddef, ← hs3def]
  refine (M.rnd_close ?_ hsumβ).trans_eq (by rw [bnNormBudget, ← hes3])
  rw [show s3 + β - (γ * ((x i - μ) * istd) + β) = s3 - γ * ((x i - μ) * istd) by ring]
  exact hs3

-- ════════════════════════════════════════════════════════════════
-- § The mean reduction (the easy Higham budget)
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **The mean reduction's budget, parameterised by the REDUCTION'S OWN SPEC.**
    `bnMean_close` below is this at `fsum := M.sum`, the concrete LEFT FOLD — and a GPU does not
    reduce left to right, so a number stated through that instance is about a program we do not
    ship. This form takes any `fsum` whose forward error meets a fan-in `γn`, which is what every
    summation order satisfies: sequential summation is the worst of them at
    `γ = (1+u)^{n+1} − 1`, and a tree's is `(1+u)^{⌈log₂n⌉+1} − 1`, strictly smaller, so the
    bound holds a fortiori. **The resulting mean accuracy is DERIVED for the kernel actually
    shipped rather than supplied by analogy** (`planning/archive/float_budget_numbers_log.md` §3.31 route 3);
    it is `bnMean_close`'s proof with one hypothesis substituted.

    ⚠ Only the SUM is parameterised. The division by the exact width is `M.div`, one rounding,
    and there is nothing to model about it. -/
theorem FloatModel.bnMean_close_of {n : ℕ} (M : FloatModel) {fsum : Vec n → ℝ} {γn A : ℝ}
    (x : Vec n) (hn : 0 < n) (hγn0 : 0 ≤ γn)
    (hsc : |fsum x - ∑ i, x i| ≤ γn * ∑ i, |x i|) (hA : ∀ i, |x i| ≤ A) :
    |M.div (fsum x) (n : ℝ) - bnMean n x| ≤ M.u * ((γn + 1) * A) + γn * A := by
  have hnR : (0:ℝ) < (n:ℝ) := by exact_mod_cast hn
  have hsumabs : ∑ i, |x i| ≤ (n:ℝ) * A := by
    calc ∑ i, |x i| ≤ ∑ _i : Fin n, A := Finset.sum_le_sum fun i _ => hA i
      _ = (n:ℝ) * A := by
          rw [Fin.sum_const, nsmul_eq_mul]
  -- |Σx| ≤ Σ|x|, hence |fsum x| ≤ (γn+1)·Σ|x|
  have hSabs : |fsum x| ≤ (γn + 1) * ((n:ℝ) * A) := by
    have htri := abs_sub_le (fsum x) (∑ i, x i) 0
    simp only [sub_zero] at htri
    have hSig : |∑ i, x i| ≤ ∑ i, |x i| := Finset.abs_sum_le_sum_abs _ _
    calc |fsum x| ≤ |fsum x - ∑ i, x i| + |∑ i, x i| := htri
      _ ≤ γn * ∑ i, |x i| + ∑ i, |x i| := add_le_add hsc hSig
      _ = (γn + 1) * ∑ i, |x i| := by ring
      _ ≤ (γn + 1) * ((n:ℝ) * A) := mul_le_mul_of_nonneg_left hsumabs (by linarith)
  rw [show bnMean n x = (∑ i, x i) / (n:ℝ) from rfl]
  -- one division rounding over (fsum x)/n
  refine M.rnd_close ?_ ?_
  · rw [div_sub_div_same, abs_div, abs_of_pos hnR, div_le_iff₀ hnR]
    calc |fsum x - ∑ i, x i| ≤ γn * ∑ i, |x i| := hsc
      _ ≤ γn * ((n:ℝ) * A) := mul_le_mul_of_nonneg_left hsumabs hγn0
      _ = γn * A * (n:ℝ) := by ring
  · rw [abs_div, abs_of_pos hnR, div_le_iff₀ hnR]
    calc |fsum x| ≤ (γn + 1) * ((n:ℝ) * A) := hSabs
      _ = (γn + 1) * A * (n:ℝ) := by ring

/-- **BN mean rounding budget.** The float mean `fl((Σx)/n)` (rounded sum, then a
    rounded division by the exact `n`) is within
    `u·(1+u)^{n+1}·A + ((1+u)^{n+1}−1)·A` of the real `bnMean`, under `|xᵢ| ≤ A`.
    Standard: `sum_close`'s fan-in `γ` plus one division rounding. ⚠ Stated at the concrete
    left fold `M.sum`; `bnMean_close_of` above is the form that covers the kernels we ship. -/
theorem FloatModel.bnMean_close {n : ℕ} (M : FloatModel) (x : Vec n) {A : ℝ}
    (hn : 0 < n) (hA : ∀ i, |x i| ≤ A) :
    |M.div (M.sum x) (n : ℝ) - bnMean n x| ≤
      M.u * ((1 + M.u) ^ (n + 1) * A) + ((1 + M.u) ^ (n + 1) - 1) * A := by
  have h := M.bnMean_close_of (fsum := M.sum) x hn
    (sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))) (M.sum_close x) hA
  simpa using h

-- ════════════════════════════════════════════════════════════════
-- § The variance reduction (the coupled Higham budget)
-- ════════════════════════════════════════════════════════════════

/-- Closed-form budget of the float variance vs `bnVar`. `esq` is the per-term
    centered-square error (`mulErr` at the centered bound `D` and centered error
    `es1 = u(D+emean)+emean`); `γₙ` is the reduction fan-in. -/
noncomputable def bnVarBudget (u D emean : ℝ) (n : ℕ) : ℝ :=
  u * (D * D
        + ((1 + u) ^ (n + 1) - 1)
            * (D * D + FloatModel.mulErr u D D (u * (D + emean) + emean) (u * (D + emean) + emean))
        + FloatModel.mulErr u D D (u * (D + emean) + emean) (u * (D + emean) + emean))
    + (((1 + u) ^ (n + 1) - 1)
            * (D * D + FloatModel.mulErr u D D (u * (D + emean) + emean) (u * (D + emean) + emean))
        + FloatModel.mulErr u D D (u * (D + emean) + emean) (u * (D + emean) + emean))

/-- **BN variance rounding budget.** The float variance (rounded centered squares,
    rounded sum, rounded `/n`) is within `bnVarBudget` of the real `bnVar`, given
    the mean within `emean` and the centered bound `|xᵢ−μ| ≤ D`. The coupled Higham
    reduction: each centered term inherits `emean` (via `mul_close`), then a
    fan-in `γ` sum and a division rounding — the variance peer of `bnMean_close`. -/
theorem FloatModel.bnVar_close {n : ℕ} (M : FloatModel) (x : Vec n) {fμ emean D : ℝ}
    (hn : 0 < n) (hmean : |fμ - bnMean n x| ≤ emean)
    (hD : ∀ i, |x i - bnMean n x| ≤ D) :
    |M.div (M.sum (fun i => M.mul (M.sub (x i) fμ) (M.sub (x i) fμ))) (n : ℝ) - bnVar n x|
      ≤ bnVarBudget M.u D emean n := by
  have hnR : (0:ℝ) < (n:ℝ) := by exact_mod_cast hn
  set μ := bnMean n x with hμ
  have hD0 : 0 ≤ D := (abs_nonneg _).trans (hD ⟨0, hn⟩)
  set es1 := M.u * (D + emean) + emean with hes1
  set γn := (1 + M.u) ^ (n + 1) - 1 with hγn
  have hγn0 : 0 ≤ γn := sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
  -- per-coordinate centered error  fl(xᵢ ⊖ fμ) ≈ xᵢ − μ, within es1
  have hces : ∀ i, |M.sub (x i) fμ - (x i - μ)| ≤ es1 := fun i =>
    M.rnd_close (by rw [show x i - fμ - (x i - μ) = μ - fμ by ring, abs_sub_comm]; exact hmean)
      ((abs_sub_le _ μ _).trans (add_le_add (hD i) (by rwa [abs_sub_comm])))
  set esq := FloatModel.mulErr M.u D D es1 es1 with hesq
  -- per-coordinate square error and float-square magnitude
  have hsqerr : ∀ i, |M.mul (M.sub (x i) fμ) (M.sub (x i) fμ) - (x i - μ) * (x i - μ)| ≤ esq :=
    fun i => M.mul_close (hces i) (hces i) (hD i) (hD i)
  have hsqmag : ∀ i, |(x i - μ) * (x i - μ)| ≤ D * D := by
    intro i; rw [abs_mul]; exact mul_le_mul (hD i) (hD i) (abs_nonneg _) hD0
  set fsq := fun i => M.mul (M.sub (x i) fμ) (M.sub (x i) fμ) with hfsq
  have hfsqmag : ∀ i, |fsq i| ≤ D * D + esq := fun i => by
    linarith [abs_sub_abs_le_abs_sub (fsq i) ((x i - μ) * (x i - μ)), hsqerr i, hsqmag i]
  -- the float numerator's mean is the mean reduction's budget …
  have hmean' := M.bnMean_close_of (fsum := M.sum) fsq hn hγn0 (M.sum_close fsq) hfsqmag
  -- … and its real mean is within `esq` of the variance (term by term)
  have hshift : |bnMean n fsq - bnVar n x| ≤ esq := by
    show |(∑ i, fsq i) / (n:ℝ) - (∑ i, (x i - μ) * (x i - μ)) / (n:ℝ)| ≤ esq
    rw [div_sub_div_same, abs_div, abs_of_pos hnR, div_le_iff₀ hnR, ← Finset.sum_sub_distrib]
    calc |∑ i, (fsq i - (x i - μ) * (x i - μ))|
        ≤ ∑ i, |fsq i - (x i - μ) * (x i - μ)| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Fin n, esq := Finset.sum_le_sum fun i _ => hsqerr i
      _ = esq * (n:ℝ) := by
          rw [Fin.sum_const, nsmul_eq_mul, mul_comm]
  calc |M.div (M.sum fsq) (n:ℝ) - bnVar n x|
      ≤ |M.div (M.sum fsq) (n:ℝ) - bnMean n fsq| + |bnMean n fsq - bnVar n x| :=
        abs_sub_le _ _ _
    _ ≤ M.u * ((γn + 1) * (D * D + esq)) + γn * (D * D + esq) + esq := add_le_add hmean' hshift
    _ = bnVarBudget M.u D emean n := by
        rw [bnVarBudget, ← hes1, ← hesq, ← hγn]; ring

-- ════════════════════════════════════════════════════════════════
-- § Assembled per-example BN forward closeness
-- ════════════════════════════════════════════════════════════════

/-- **Per-example BN forward closeness (assembled).** The float BN forward at the
    rounded mean `fl((Σx)/n)` and the GPU inverse-stddev `fistd fvarε` is within
    `bnNormBudget` of the certified `bnForward`, with the mean error discharged by
    `bnMean_close`, the inverse-stddev error by the `bnIstd_close` keystone (the
    `rsqrt` accuracy + `rsqrt_lipschitz`), and the normalize chain by
    `bnForward_close_of`. The only supplied input is `fvarε`'s closeness to `σ²+ε`
    (`hvar`) — the variance Higham reduction, the one remaining mechanical piece. -/
theorem FloatModel.bnForward_close {n : Nat} (M : FloatModel)
    {ε γ β evar D S G Bbnd ers fvarε A : ℝ} (x : Vec n) (i : Fin n) (fistd : ℝ → ℝ)
    (hn : 0 < n) (hε : 0 < ε) (hers : 0 ≤ ers) (hfv : ε ≤ fvarε)
    (hrs : |fistd fvarε - 1 / Real.sqrt fvarε| ≤ ers * (1 / Real.sqrt fvarε))
    (hvar : |fvarε - (bnVar n x + ε)| ≤ evar)
    (hA : ∀ j, |x j| ≤ A)
    (hD : |x i - bnMean n x| ≤ D) (hSabs : |bnIstd n x ε| ≤ S)
    (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd) :
    |M.bnForwardF γ β (M.div (M.sum x) (n:ℝ)) (fistd fvarε) x i - bnForward n ε γ β x i| ≤
      bnNormBudget M.u D S G Bbnd
        (M.u * ((1 + M.u) ^ (n + 1) * A) + ((1 + M.u) ^ (n + 1) - 1) * A)
        (ers / Real.sqrt ε + evar / (2 * ε * Real.sqrt ε)) :=
  M.bnForward_close_of x i (M.bnMean_close x hn hA)
    (bnIstd_close x fistd hε hers hfv hrs hvar) hD hSabs hγ hβ

end Proofs
