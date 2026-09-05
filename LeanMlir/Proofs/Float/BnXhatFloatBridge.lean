import LeanMlir.Proofs.Float.FloatBudgetEnv
import LeanMlir.Proofs.Float.BnPerChannelFloatBridge
import LeanMlir.Proofs.Foundation.ResNet34

/-! # §0.1's ESCAPE 2, window half: the normalisation leaf stated at `|x̂| ≤ √n`

`floatClose_bn` (`FloatComposeBridge.lean`) bounds the normalised activation by
`|x − μ| · |istd| ≤ D · S` — the product of the two factors' own windows — and at the `ε`-floor
`S = 1/√ε = 317` that is what makes a LayerNorm net's certified window compound. But `x̂` has a
bound of its own that mentions neither factor: **`x̂ₖ² ≤ n`** (`bnXhat_sq_le`, `Foundation/
ResNet34.lean`), because `istd² = 1/(σ²+ε)` and `(vₖ−μ)² ≤ Σⱼ(vⱼ−μ)² = n·σ²`. At ConvNeXt-T's
stem that is `|x̂| ≤ 10` where `D·S` is `2A·317`.

This file restates the leaf at that bound. Worth **53 orders on ConvNeXt-T's committed window
and 57 on ViT-Tiny's**, at no new hypothesis and no new mathematics — `bnXhat_sq_le` has been in
the repo since the realistic-seal work and is the load-bearing lemma on all four whole-net
BACKWARD numbers (`planning/float_budget_numbers.md` §3.9 finding 4, §3.16 finding 5, §3.22).
The forward leaf simply threw it away. ⭐ That is §3.3.0(b)'s rule — *before writing a tighter
leaf bound, grep the whole cone for it* — at its largest instance yet: two committed numbers,
~55 orders each, from a lemma the repo already had.

**Two places charge the window, and BOTH are fixed here.**

1. **The real output's magnitude**, `|γ·x̂ + β| ≤ G·Xh + B̄` where `floatClose_bn` has
   `G·(D·S) + B̄`. One `bnXhat_abs_le_num`.
2. ⭐ **The rounding of the float product**, and this one is subtler. `bnNormBudget`'s stage-2
   term is `mulErr u D S ea ei`, whose rounding half is `u·((D+ea)·(S+ei))` — the float product
   bounded by the product of the two factors' windows. **But that product IS the normalised
   activation, one rounding away**, so the honest bound is `u·(Xh + t) + t` with `t` the
   exact-arithmetic error the same expression already computes. `bnNormBudgetX` is
   `bnNormBudget` with those two substitutions and nothing else.
   ⭐ That is `floatClose_seScale`'s fix (§3.4 finding 2) and `mhpB`'s (§3.5) at a third leaf —
   **the third time a window has been derived through an error term when a direct bound on the
   float side was available.** §0.1's checklist item, *when a window contains an error term, ask
   why*, for the third time.

⛔ **The MODULUS is deliberately untouched.** `floatClose_bnX`'s error clause is
`floatClose_bn`'s verbatim — `bnReluBudget`, still carrying §0.1's quadratic input-sensitivity
`G·2A·(8A·e/(2ε√ε))`. Escape 2 has a modulus half too (`2e·S·(1+Xh)`, free of the window: write
the two normalised outputs over a common denominator and the factor multiplying the
σ-difference is the NORMALISED activation, with `|σ_a − σ_t| ≤ 2e` by the reverse triangle
inequality), and it is worth **nothing** to either net that consumes this file, because both
their numbers are `FloatBridgesTo.capped` and the fold stays above the cap even with the
quadratic gone — 82 orders above on ConvNeXt-T, 7 on ViT-Tiny (§3.27 finding 3). Taking it
would need the reverse triangle inequality for `‖·‖₂` over `Vec n` and `s ↦ √(s²+ε)` being
1-Lipschitz, neither of which the float tier currently uses. Priced, measured, not taken.

⚠ **What survives, and it is the interesting residue.** `D·ei + ea·S` — the DEVICE's
inverse-stddev and mean accuracies — stay linear in the window, and they are what stops a
normalisation site RESETTING. §3.27 finding 4: a site resets iff `emr·S < 1`, which at the
`ε`-floor and `emr = 10⁻²` is `3.17`, so ConvNeXt's sites still multiply (measured gain 3.19)
and `Xh` is the base rather than the answer. ⭐ Before buying `emr·S < 1` with an operating
point, question `emr = 10⁻²` itself: it is a hypothesis about a reduction the device performs in
float, and a rounded mean of `n` terms is `γₙ·A`, which at `n = 96` and `u = 2⁻²⁴` is `6·10⁻⁶`.
Deriving it rather than supplying it is worth a further 49 orders on ConvNeXt-T and 55 on
ViT-Tiny (measured); it costs a `DeviceLN` field changing status, so it is its own commit.
-/

namespace Proofs

open FloatModel

/-- ⭐ **`|x̂| ≤ X` at a RATIONAL `X`.** `bnXhat_sq_le` gives `x̂² ≤ n`, and `√n` is irrational
    for all four of ConvNeXt's channel counts (96/192/384/768) and for ViT's `D = 192`, so the
    numeral form takes the CEILING root — 10/14/20/28 and 14 — wherever `n ≤ X²`. Exact for the
    square feature maps a conv net's BatchNorm reduces over (`h·w = 56² ⇒ X = 56`).

    ⚠ Moved here from `FloatBudgetEnvBack.lean`, which is the ResNet-34 BACKWARD kit: the four
    backward budget files have called it since 2026-09-03, and a FORWARD leaf must not import
    the backward cone to reach it. Same lemma, one file lower. -/
theorem bnXhat_abs_le_num {n : Nat} {ε X : ℝ} (hε : 0 < ε) (v : Vec n)
    (hX : 0 ≤ X) (hnX : (n : ℝ) ≤ X ^ 2) (k : Fin n) : |bnXhat n ε v k| ≤ X := by
  have hsq := bnXhat_sq_le ε hε v k
  have habs : |bnXhat n ε v k| ^ 2 ≤ X ^ 2 := by
    rw [sq_abs]; linarith
  nlinarith [abs_nonneg (bnXhat n ε v k)]

-- ════════════════════════════════════════════════════════════════
-- § A rounded product charged at the REAL PRODUCT's bound
-- ════════════════════════════════════════════════════════════════

/-- **The exact-arithmetic half of a rounded product's error.** With `|xt−x| ≤ ea`,
    `|yt−y| ≤ ec`, `|x| ≤ A` and `|y| ≤ C`, the UNROUNDED float product is within
    `A·ec + ea·C + ea·ec` of the real one. This is `FloatModel.mul_close`'s internal `hprod`
    step, which that lemma consumes and does not export; `mul_close_at` needs it separately
    because it charges the ROUNDING at a different bound. -/
theorem prod_sub_abs_le {xt x yt y ea ec A C : ℝ}
    (hx : |xt - x| ≤ ea) (hy : |yt - y| ≤ ec) (hA : |x| ≤ A) (hC : |y| ≤ C) :
    |xt * yt - x * y| ≤ A * ec + ea * C + ea * ec := by
  have hea0 : 0 ≤ ea := (abs_nonneg _).trans hx
  have hec0 : 0 ≤ ec := (abs_nonneg _).trans hy
  have hA0 : 0 ≤ A := (abs_nonneg _).trans hA
  have hC0 : 0 ≤ C := (abs_nonneg _).trans hC
  have hxt : |xt| ≤ A + ea := by
    have h := abs_sub_le xt x 0
    simp only [sub_zero] at h
    linarith
  have h1 : xt * yt - x * y = xt * (yt - y) + y * (xt - x) := by ring
  have h2 : |xt| * |yt - y| ≤ (A + ea) * ec := mul_le_mul hxt hy (abs_nonneg _) (by linarith)
  have h3 : |y| * |xt - x| ≤ C * ea := mul_le_mul hC hx (abs_nonneg _) hC0
  calc |xt * yt - x * y| = |xt * (yt - y) + y * (xt - x)| := by rw [h1]
    _ ≤ |xt * (yt - y)| + |y * (xt - x)| := abs_add_le _ _
    _ = |xt| * |yt - y| + |y| * |xt - x| := by rw [abs_mul, abs_mul]
    _ ≤ A * ec + ea * C + ea * ec := by nlinarith

namespace FloatModel

variable (M : FloatModel)

/-- ⭐⭐ **A rounded product whose ROUNDING is charged at the real product's own bound.**
    `mul_close` bounds `|fl(xt·yt)|` by `(A+ea)·(C+ec)`, the product of the two factors'
    windows, because that is all it is given. When the real product `x·y` has a bound `P` of its
    own that is tighter than `A·C` — which is exactly the normalised activation's situation,
    `|x̂| ≤ √n` against `|x−μ|·|istd| ≤ D·S` — the float product is within `t` of it and so is
    bounded by `P + t`, and the rounding costs `u·(P+t)` instead.

    ⚠ It takes the exact-arithmetic error `t` as a hypothesis rather than deriving it, because
    the caller usually has a tighter `t` than `prod_sub_abs_le` gives and because keeping the
    two halves separate is what lets the window and the modulus be improved independently. -/
theorem mul_close_at {xt yt x y t P : ℝ}
    (ht : |xt * yt - x * y| ≤ t) (hP : |x * y| ≤ P) :
    |M.mul xt yt - x * y| ≤ M.u * (P + t) + t := by
  have hu := M.u_nonneg
  have hrnd : |M.mul xt yt - xt * yt| ≤ M.u * |xt * yt| := M.err _
  have hbd : |xt * yt| ≤ P + t := by
    have h := abs_sub_le (xt * yt) (x * y) 0
    simp only [sub_zero] at h
    linarith
  have h2 : M.u * |xt * yt| ≤ M.u * (P + t) := mul_le_mul_of_nonneg_left hbd hu
  calc |M.mul xt yt - x * y|
      ≤ |M.mul xt yt - xt * yt| + |xt * yt - x * y| := abs_sub_le _ _ _
    _ ≤ M.u * |xt * yt| + t := add_le_add hrnd ht
    _ ≤ M.u * (P + t) + t := by linarith

end FloatModel

-- ════════════════════════════════════════════════════════════════
-- § `bnNormBudget` restated at `|x̂| ≤ Xh`
-- ════════════════════════════════════════════════════════════════

/-- The centering step's error: one rounded subtract over the supplied mean. Unchanged from
    `bnNormBudget`'s first stage; named so the monotonicity steps below are three lines each
    rather than one literal (the shape `bnGradInputBudgetG` needed, §3.7 step 1). -/
noncomputable def bnCentErr (u D emean : ℝ) : ℝ := u * (D + emean) + emean

/-- The EXACT-arithmetic error of `centred · fistd` against `(x−μ)·istd` — `prod_sub_abs_le` at
    `A := D`, `C := S`. This half is identical to `bnNormBudget`'s; it is the ROUNDING half that
    escape 2 changes. -/
noncomputable def bnProdErr (u D S emean eistd : ℝ) : ℝ :=
  D * eistd + bnCentErr u D emean * S + bnCentErr u D emean * eistd

/-- ⭐ The normalised activation's total error, with its rounding charged at `Xh` — the whole
    of escape 2's window half in one line. `bnNormBudget` has
    `u·((D+ea)·(S+ei)) + bnProdErr` here. -/
noncomputable def bnXhatErr (u Xh D S emean eistd : ℝ) : ℝ :=
  u * (Xh + bnProdErr u D S emean eistd) + bnProdErr u D S emean eistd

/-- ⭐⭐ **`bnNormBudget` at `|x̂| ≤ Xh`.** Two substitutions and nothing else: the stage-2
    rounding is `bnXhatErr` where `bnNormBudget` has `mulErr u D S ea ei`, and the real
    normalised activation's magnitude is `Xh` where `bnNormBudget` has `D · S`. -/
noncomputable def bnNormBudgetX (u Xh D S G Bbnd emean eistd : ℝ) : ℝ :=
  u * (G * Xh + FloatModel.mulErr u G Xh 0 (bnXhatErr u Xh D S emean eistd) + Bbnd)
    + FloatModel.mulErr u G Xh 0 (bnXhatErr u Xh D S emean eistd)

/-! ### The monotone forms

Named one per stage rather than proved as one literal — `bnGradInputBudgetG`'s shape (§3.7
step 1): each step is then three lines and `nlinarith` never sees the whole chain. ⭐ Note what
is NOT quantified over: `Xh` is a constant of the reduction width, so it does not move with the
window and the monotonicity is in `D`, `em`, `ei` and the rounding unit only. -/

theorem bnCentErr_nonneg {u D em : ℝ} (hu : 0 ≤ u) (hD : 0 ≤ D) (hem : 0 ≤ em) :
    0 ≤ bnCentErr u D em := by unfold bnCentErr; positivity

theorem bnCentErr_mono {u u' D D' em em' : ℝ} (hu : 0 ≤ u) (huu : u ≤ u')
    (hD : 0 ≤ D) (hDD : D ≤ D') (hem : 0 ≤ em) (hemm : em ≤ em') :
    bnCentErr u D em ≤ bnCentErr u' D' em' := by
  unfold bnCentErr
  have : u * (D + em) ≤ u' * (D' + em') :=
    mul_le_mul huu (by linarith) (by linarith) (by linarith)
  linarith

theorem bnProdErr_nonneg {u D S em ei : ℝ} (hu : 0 ≤ u) (hD : 0 ≤ D) (hS : 0 ≤ S)
    (hem : 0 ≤ em) (hei : 0 ≤ ei) : 0 ≤ bnProdErr u D S em ei := by
  unfold bnProdErr
  have := bnCentErr_nonneg hu hD hem
  positivity

theorem bnProdErr_mono {u u' D D' S em em' ei ei' : ℝ} (hu : 0 ≤ u) (huu : u ≤ u')
    (hD : 0 ≤ D) (hDD : D ≤ D') (hS : 0 ≤ S) (hem : 0 ≤ em) (hemm : em ≤ em')
    (hei : 0 ≤ ei) (heii : ei ≤ ei') :
    bnProdErr u D S em ei ≤ bnProdErr u' D' S em' ei' := by
  unfold bnProdErr
  have hc0 := bnCentErr_nonneg hu hD hem
  have hcc := bnCentErr_mono hu huu hD hDD hem hemm
  have h1 : D * ei ≤ D' * ei' := mul_le_mul hDD heii hei (by linarith)
  have h2 : bnCentErr u D em * S ≤ bnCentErr u' D' em' * S := by nlinarith
  have h3 : bnCentErr u D em * ei ≤ bnCentErr u' D' em' * ei' :=
    mul_le_mul hcc heii hei (by linarith)
  linarith

theorem bnXhatErr_nonneg {u Xh D S em ei : ℝ} (hu : 0 ≤ u) (hXh : 0 ≤ Xh) (hD : 0 ≤ D)
    (hS : 0 ≤ S) (hem : 0 ≤ em) (hei : 0 ≤ ei) : 0 ≤ bnXhatErr u Xh D S em ei := by
  unfold bnXhatErr
  have := bnProdErr_nonneg hu hD hS hem hei
  positivity

theorem bnXhatErr_mono {u u' Xh D D' S em em' ei ei' : ℝ} (hu : 0 ≤ u) (huu : u ≤ u')
    (hXh : 0 ≤ Xh) (hD : 0 ≤ D) (hDD : D ≤ D') (hS : 0 ≤ S) (hem : 0 ≤ em) (hemm : em ≤ em')
    (hei : 0 ≤ ei) (heii : ei ≤ ei') :
    bnXhatErr u Xh D S em ei ≤ bnXhatErr u' Xh D' S em' ei' := by
  unfold bnXhatErr
  have hp0 := bnProdErr_nonneg hu hD hS hem hei
  have hpp := bnProdErr_mono hu huu hD hDD hS hem hemm hei heii
  have : u * (Xh + bnProdErr u D S em ei) ≤ u' * (Xh + bnProdErr u' D' S em' ei') :=
    mul_le_mul huu (by linarith) (by linarith) (by linarith)
  linarith

/-- ⭐ **`bnNormBudgetX` is monotone** in the rounding unit, the centered bound and the two
    supplied statistic accuracies — the fact that lets a symbolic `M.u` and a symbolic window be
    replaced by rationals before `norm_num` sees the expression. `bnNormBudget_mono`'s peer. -/
theorem bnNormBudgetX_mono {u u' Xh D D' S G Bbnd em em' ei ei' : ℝ}
    (hu : 0 ≤ u) (huu : u ≤ u') (hXh : 0 ≤ Xh) (hD : 0 ≤ D) (hDD : D ≤ D') (hS : 0 ≤ S)
    (hG : 0 ≤ G) (hBb : 0 ≤ Bbnd) (hem : 0 ≤ em) (hemm : em ≤ em')
    (hei : 0 ≤ ei) (heii : ei ≤ ei') :
    bnNormBudgetX u Xh D S G Bbnd em ei ≤ bnNormBudgetX u' Xh D' S G Bbnd em' ei' := by
  have hu'0 : 0 ≤ u' := hu.trans huu
  have hx0 := bnXhatErr_nonneg hu hXh hD hS hem hei
  have hxx := bnXhatErr_mono hu huu hXh hD hDD hS hem hemm hei heii
  have houter : FloatModel.mulErr u G Xh 0 (bnXhatErr u Xh D S em ei)
      ≤ FloatModel.mulErr u' G Xh 0 (bnXhatErr u' Xh D' S em' ei') :=
    mulErr_mono hu huu hG le_rfl hXh le_rfl le_rfl le_rfl hx0 hxx
  have houter0 : 0 ≤ FloatModel.mulErr u G Xh 0 (bnXhatErr u Xh D S em ei) :=
    mulErr_nonneg hu hG hXh le_rfl hx0
  unfold bnNormBudgetX
  have hmul : u * (G * Xh + FloatModel.mulErr u G Xh 0 (bnXhatErr u Xh D S em ei) + Bbnd)
      ≤ u' * (G * Xh + FloatModel.mulErr u' G Xh 0 (bnXhatErr u' Xh D' S em' ei') + Bbnd) :=
    mul_le_mul huu (by nlinarith) (by nlinarith) hu'0
  linarith

/-- **BN forward closeness, normalize chain, at `|x̂| ≤ Xh`.** `bnForward_close_of` with the two
    substitutions above. Stages 1 and 4 are that lemma's verbatim; stage 2 goes through
    `mul_close_at` instead of `mul_close`, and stage 3's `mul_close` is applied at `Xh` rather
    than at `D · S`. -/
theorem FloatModel.bnForward_close_of_x {n : Nat} (M : FloatModel)
    {ε γ β fμ fistdv emean eistd D S G Bbnd Xh : ℝ} (x : Vec n) (i : Fin n)
    (hmean : |fμ - bnMean n x| ≤ emean)
    (histd : |fistdv - bnIstd n x ε| ≤ eistd)
    (hD : |x i - bnMean n x| ≤ D) (hSabs : |bnIstd n x ε| ≤ S)
    (hXh : |(x i - bnMean n x) * bnIstd n x ε| ≤ Xh)
    (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd) :
    |M.bnForwardF γ β fμ fistdv x i - bnForward n ε γ β x i| ≤
      bnNormBudgetX M.u Xh D S G Bbnd emean eistd := by
  have hu := M.u_nonneg
  set μ := bnMean n x with hμ
  set istd := bnIstd n x ε with histddef
  -- stage 1: the centering, `bnForward_close_of`'s verbatim
  have hxfμ : |x i - fμ| ≤ D + emean := by
    have hmean' : |μ - fμ| ≤ emean := by rw [abs_sub_comm]; exact hmean
    calc |x i - fμ| ≤ |x i - μ| + |μ - fμ| := abs_sub_le _ _ _
      _ ≤ D + emean := add_le_add hD hmean'
  have hs1 : |M.sub (x i) fμ - (x i - μ)| ≤ bnCentErr M.u D emean := by
    have h1 : |M.sub (x i) fμ - (x i - fμ)| ≤ M.u * |x i - fμ| := M.err _
    have h2 : |(x i - fμ) - (x i - μ)| ≤ emean := by
      have he : (x i - fμ) - (x i - μ) = μ - fμ := by ring
      rw [he, abs_sub_comm]; exact hmean
    unfold bnCentErr
    calc |M.sub (x i) fμ - (x i - μ)|
        ≤ |M.sub (x i) fμ - (x i - fμ)| + |(x i - fμ) - (x i - μ)| := abs_sub_le _ _ _
      _ ≤ M.u * |x i - fμ| + emean := add_le_add h1 h2
      _ ≤ M.u * (D + emean) + emean := by gcongr
  -- stage 2: ⭐ the rounding charged at `Xh`, not at `(D+ea)*(S+ei)`
  have hprod : |M.sub (x i) fμ * fistdv - (x i - μ) * istd|
      ≤ bnProdErr M.u D S emean eistd :=
    prod_sub_abs_le hs1 histd hD hSabs
  have hs2 : |M.mul (M.sub (x i) fμ) fistdv - (x i - μ) * istd|
      ≤ bnXhatErr M.u Xh D S emean eistd :=
    M.mul_close_at hprod hXh
  -- stage 3: the γ multiply, at the tighter activation bound
  have hs3 : |M.mul γ (M.mul (M.sub (x i) fμ) fistdv) - γ * ((x i - μ) * istd)| ≤
      FloatModel.mulErr M.u G Xh 0 (bnXhatErr M.u Xh D S emean eistd) :=
    M.mul_close (by simp) hs2 hγ hXh
  -- stage 4: + β and assemble, `bnForward_close_of`'s verbatim at `Xh`
  set es3 := FloatModel.mulErr M.u G Xh 0 (bnXhatErr M.u Xh D S emean eistd) with hes3
  set s3 := M.mul γ (M.mul (M.sub (x i) fμ) fistdv) with hs3def
  have hgxhat : |γ * ((x i - μ) * istd)| ≤ G * Xh := by
    rw [abs_mul]; exact mul_le_mul hγ hXh (abs_nonneg _) ((abs_nonneg _).trans hγ)
  have hs3mag : |s3| ≤ G * Xh + es3 := by
    calc |s3| = |(s3 - γ * ((x i - μ) * istd)) + γ * ((x i - μ) * istd)| := by congr 1; ring
      _ ≤ |s3 - γ * ((x i - μ) * istd)| + |γ * ((x i - μ) * istd)| := abs_add_le _ _
      _ ≤ es3 + G * Xh := add_le_add hs3 hgxhat
      _ = G * Xh + es3 := by ring
  have hsumβ : |s3 + β| ≤ G * Xh + es3 + Bbnd := by
    calc |s3 + β| ≤ |s3| + |β| := abs_add_le _ _
      _ ≤ (G * Xh + es3) + Bbnd := add_le_add hs3mag hβ
      _ = G * Xh + es3 + Bbnd := by ring
  have hgoal :
      |M.bnForwardF γ β fμ fistdv x i - bnForward n ε γ β x i| ≤
        M.u * (G * Xh + es3 + Bbnd) + es3 := by
    simp only [FloatModel.bnForwardF, bnForward, bnXhat, ← hμ, ← histddef, ← hs3def]
    have h4a : |M.add s3 β - (s3 + β)| ≤ M.u * |s3 + β| := M.err _
    have h4b : |(s3 + β) - (γ * ((x i - μ) * istd) + β)| ≤ es3 := by
      have he : (s3 + β) - (γ * ((x i - μ) * istd) + β) = s3 - γ * ((x i - μ) * istd) := by ring
      rw [he]; exact hs3
    calc |M.add s3 β - (γ * ((x i - μ) * istd) + β)|
        ≤ |M.add s3 β - (s3 + β)| + |(s3 + β) - (γ * ((x i - μ) * istd) + β)| := abs_sub_le _ _ _
      _ ≤ M.u * |s3 + β| + es3 := add_le_add h4a h4b
      _ ≤ M.u * (G * Xh + es3 + Bbnd) + es3 := by gcongr
  exact hgoal.trans_eq (by rw [bnNormBudgetX, ← hes3])

-- ════════════════════════════════════════════════════════════════
-- § The leaf, and its bridge
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **The pure normalisation is `FloatClose` at `|x̂| ≤ Xh`.** `floatClose_bn` with the
    window `G·(D·S) + B̄ + bnNormBudget` replaced by `G·Xh + B̄ + bnNormBudgetX`, and the ERROR
    CLAUSE IDENTICAL — the same `bnReluBudget`, still carrying §0.1's quadratic. That asymmetry
    is deliberate and is the whole design of this file: both consumers cap their sites, so the
    modulus never becomes a numeral and improving it would buy nothing (§3.27 finding 3).

    ⭐ `Xh` needs no hypothesis about the input: `bnXhat_sq_le` holds at every `v`, so the
    caller supplies only `0 ≤ Xh` and `m ≤ Xh²`. Contrast `D`, which is `2A` and therefore
    tracks the window — it survives, in `bnProdErr`, multiplied by the DEVICE accuracies. -/
theorem floatClose_bnX {m : Nat} (M : FloatModel)
    {ε γ β emean eistd D S G Bbnd A Xh : ℝ} (fμ fistdv : Vec m → ℝ)
    (hn : 0 < m) (hε : 0 < ε) (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd)
    (hmean : ∀ v, (∀ k, |v k| ≤ A) → |fμ v - bnMean m v| ≤ emean)
    (histd : ∀ v, (∀ k, |v k| ≤ A) → |fistdv v - bnIstd m v ε| ≤ eistd)
    (hD : ∀ v, (∀ k, |v k| ≤ A) → ∀ j, |v j - bnMean m v| ≤ D)
    (hSabs : ∀ v, (∀ k, |v k| ≤ A) → |bnIstd m v ε| ≤ S)
    (hXh0 : 0 ≤ Xh) (hmXh : (m : ℝ) ≤ Xh ^ 2) :
    FloatClose A (G * Xh + Bbnd + bnNormBudgetX M.u Xh D S G Bbnd emean eistd)
      (fun v => bnForward m ε γ β v)
      (fun v => M.bnForwardF γ β (fμ v) (fistdv v) v)
      (fun e => bnReluBudget M.u D S G Bbnd emean eistd A e ε) := by
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i =>
    M.bnStep_close vt va i hn hε hd hvt hva (hmean vt hvt) (histd vt hvt)
      (hD vt hvt) (hSabs vt hvt) hγ hβ⟩
  have hu := M.u_nonneg
  have hG0 : 0 ≤ G := (abs_nonneg _).trans hγ
  have hBbnd0 : 0 ≤ Bbnd := (abs_nonneg _).trans hβ
  have hS0 : 0 ≤ S := (abs_nonneg _).trans (hSabs v hv)
  have hD0 : 0 ≤ D := (abs_nonneg _).trans (hD v hv i)
  have hem0 : 0 ≤ emean := (abs_nonneg _).trans (hmean v hv)
  have hei0 : 0 ≤ eistd := (abs_nonneg _).trans (histd v hv)
  have hnb0 : 0 ≤ bnNormBudgetX M.u Xh D S G Bbnd emean eistd := by
    unfold bnNormBudgetX bnXhatErr bnProdErr bnCentErr FloatModel.mulErr; positivity
  -- ⭐ the escape: `|x̂| ≤ Xh` at EVERY input, from `bnXhat_sq_le`
  have hXh : |bnXhat m ε v i| ≤ Xh := bnXhat_abs_le_num hε v hXh0 hmXh i
  have hXh' : |(v i - bnMean m v) * bnIstd m v ε| ≤ Xh := by
    simpa [bnXhat] using hXh
  have hreal : |bnForward m ε γ β v i| ≤ G * Xh + Bbnd := by
    unfold bnForward
    refine (abs_add_le _ _).trans (add_le_add ?_ hβ)
    rw [abs_mul]; exact mul_le_mul hγ hXh (abs_nonneg _) ((abs_nonneg _).trans hγ)
  have hround := M.bnForward_close_of_x (ε := ε) v i (hmean v hv) (histd v hv)
    (hD v hv i) (hSabs v hv) hXh' hγ hβ
  refine ⟨hreal.trans (le_add_of_nonneg_right hnb0), ?_⟩
  calc |M.bnForwardF γ β (fμ v) (fistdv v) v i|
      ≤ |M.bnForwardF γ β (fμ v) (fistdv v) v i - bnForward m ε γ β v i|
        + |bnForward m ε γ β v i| := by
        simpa using abs_sub_le (M.bnForwardF γ β (fμ v) (fistdv v) v i) (bnForward m ε γ β v i) 0
    _ ≤ bnNormBudgetX M.u Xh D S G Bbnd emean eistd + (G * Xh + Bbnd) :=
        add_le_add hround hreal
    _ = G * Xh + Bbnd + bnNormBudgetX M.u Xh D S G Bbnd emean eistd := by ring

/-- The escape-2 leaf's output window at input window `A`. `bnLeafMag`'s peer. -/
noncomputable def bnXLeafMag (u Xh S G Bbnd : ℝ) (emean eistd : ℝ → ℝ) (A : ℝ) : ℝ :=
  G * Xh + Bbnd + bnNormBudgetX u Xh (2 * A) S G Bbnd (emean A) (eistd A)

/-- ⭐⭐ **The pure normalisation float-bridges TO its float map at `|x̂| ≤ Xh`.**
    `floatBridgesTo_bn` with `bnXLeafMag` in place of `bnLeafMag`; the modulus `bnLeafMod` is
    unchanged, so this bridge differs from that one in its WINDOW alone. `D := 2A` is
    `bn_centered_le`'s generic bound, as there. -/
noncomputable def floatBridgesTo_bnX {m : Nat} (M : FloatModel) {ε γ β : ℝ}
    (fμ fistdv : Vec m → ℝ) (emean eistd : ℝ → ℝ) {G Bbnd S Xh : ℝ}
    (hm : 0 < m) (hε : 0 < ε) (hγ : |γ| ≤ G) (hβ : |β| ≤ Bbnd)
    (hmean : ∀ A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fμ v - bnMean m v| ≤ emean A)
    (histd : ∀ A, 0 ≤ A → ∀ v : Vec m, (∀ k, |v k| ≤ A) → |fistdv v - bnIstd m v ε| ≤ eistd A)
    (hS : ∀ v : Vec m, |bnIstd m v ε| ≤ S) (hXh0 : 0 ≤ Xh) (hmXh : (m : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo (bnForward m ε γ β) (bnForwardFV M γ β fμ fistdv) :=
  ⟨bnXLeafMag M.u Xh S G Bbnd emean eistd, bnLeafMod M.u ε S G Bbnd emean eistd,
   fun A hA =>
     have hfc := floatClose_bnX M fμ fistdv hm hε hγ hβ
       (fun v hv => hmean A hA v hv) (fun v hv => histd A hA v hv)
       (fun v hv j => bn_centered_le hm v hv j) (fun v _ => hS v) hXh0 hmXh
     ⟨hfc.cod_nonneg hA hm, hfc⟩⟩

end Proofs
