import LeanMlir.Proofs.Float.FloatBudgetEnv
import LeanMlir.Proofs.Float.Resnet34WholeBackFloatBridge
import LeanMlir.Proofs.Float.BnPerChannelBackFloatBridge
import LeanMlir.Proofs.Float.Resnet34BackFloatBridge

/-! # `FloatBridgesTo.Maps` leaves for the BACKWARD (phase 2)

`FloatBudgetEnv.lean` holds the kit and ResNet-34's forward leaves; this file holds the ones a
whole-net **input-gradient VJP** needs. It is phase 2 of `planning/float_budget_numbers.md`
(§3.7), and the reason it exists as its own kit is the finding that opened phase 2:

⭐⭐ **The backward is a FOLD, at TRAINING-mode BatchNorm — the mode the forward has no number
for at all.** §0.1's quadratic came from the statistics MOVING with the input; a VJP reads its
statistics off the SAVED ACTIVATIONS, which the cotangent does not perturb. So
`floatClose_bnBack`'s modulus is `bnGradInputBudget(A) + bnGradInputReMag(e)` with
`bnGradInputReMag n G e S Xh = (1/n)·S·(2n·G·e + n·Xh²·G·e)` — **linear in `e`**, because a VJP
at a fixed point is a linear map and this is its Lipschitz constant. No cap is needed and
`budget / window` is `0.048`, where ConvNeXt's and ViT's forward numbers are `2.00`.

⛔ **The wall does not vanish, it relocates.** The saved float activations enter as two SUPPLIED
accuracies — `es` on the inverse-stddev and `exh` on the normalised activation — and unlike
`DeviceRsqrt`'s those are quantities the forward fold does speak about (at training BN it says
`10⁷⁴¹⁷`). A backward number is an honest fold GIVEN a forward-accuracy hypothesis its own
forward cannot discharge. Say it that way (§3.7, §9).

Two things in here are the actual work; everything else is a ten-line envelope.

* ⭐ **`bnGradInputBudgetG`** — `FloatModel.bnGradInputBudget` with its two opaque quantities as
  PARAMETERS: the rounding unit, and the reduction's `(1+u)^(n+1) − 1` at `n` up to 12544.
  `norm_num` can evaluate neither. This is `peRoundErrQ`'s pattern (`FloatBudgetEnvAttn.lean`)
  one tier up, and `bnGradInputBudget_le` is the one bridge from the model to the numerals.
* ⭐⭐ **`Maps.bnPerChannelBack` closes `Xh` with `bnXhat_sq_le`** — `|x̂| ≤ √n`, the
  standardisation bound. That lemma was ALREADY in the repo (`Foundation/ResNet34.lean`, written
  for the realistic-seal work) and it is what makes the whole number exist: `Xh` enters the fold
  as `Xh²`, and deriving it from the forward's certified window instead gives `10⁷²⁷¹` against
  the shipped `10²⁸⁸`. Every place the activations enter an input-gradient they enter NORMALISED
  or through `istd ≤ 1/√ε`, so the backward does not inherit the forward's window at all.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The BN backward's budget, over rationals
-- ════════════════════════════════════════════════════════════════

/-! ### The BN backward's budget over rationals

⭐ `FloatModel.bnGradInputBudget` is a fifteen-step `let` chain over `M.u` and
`(1+M.u)^(n+1) − 1` at `n` up to 12544 — two quantities `norm_num` cannot evaluate. The chain is
respelled here with those two as PARAMETERS and every intermediate NAMED, so each monotonicity
step is a three-line lemma instead of a hundred-line literal. `bnGradInputBudget_eq_G` is `rfl`.
-/

/-- `eD` — the rounding of one `γ·dyᵢ`. -/
noncomputable def bgED (u G Cdy : ℝ) : ℝ := FloatModel.mulErr u G Cdy 0 0

/-- `eSD` — the error of the reduction `Σ γ·dy`. -/
noncomputable def bgESD (u gn : ℝ) (n : Nat) (G Cdy : ℝ) : ℝ :=
  gn * ((n : ℝ) * (G * Cdy + bgED u G Cdy)) + (n : ℝ) * bgED u G Cdy

/-- `MSD` — the magnitude of that reduction. -/
noncomputable def bgMSD (u gn : ℝ) (n : Nat) (G Cdy : ℝ) : ℝ :=
  (n : ℝ) * (G * Cdy) + bgESD u gn n G Cdy

/-- `eXD` — the rounding of one `x̂ᵢ·(γ·dyᵢ)`, at the supplied `x̂` accuracy `exh`. -/
noncomputable def bgEXD (u G Cdy Xh exh : ℝ) : ℝ :=
  FloatModel.mulErr u Xh (G * Cdy) exh (bgED u G Cdy)

/-- `eSXD` — the error of the reduction `Σ x̂·(γ·dy)`. -/
noncomputable def bgESXD (u gn : ℝ) (n : Nat) (G Cdy Xh exh : ℝ) : ℝ :=
  gn * ((n : ℝ) * (Xh * (G * Cdy) + bgEXD u G Cdy Xh exh))
    + (n : ℝ) * bgEXD u G Cdy Xh exh

/-- `enD` — the rounding of `n·(γ·dyᵢ)`. -/
noncomputable def bgEND (u : ℝ) (n : Nat) (G Cdy : ℝ) : ℝ :=
  FloatModel.mulErr u (n : ℝ) (G * Cdy) 0 (bgED u G Cdy)

/-- `MnD` — its magnitude. -/
noncomputable def bgMND (u : ℝ) (n : Nat) (G Cdy : ℝ) : ℝ :=
  (n : ℝ) * (G * Cdy) + bgEND u n G Cdy

/-- `e1` — the error after the first subtraction `n·dx̂ᵢ − Σdx̂`. -/
noncomputable def bgE1 (u gn : ℝ) (n : Nat) (G Cdy : ℝ) : ℝ :=
  u * (bgMND u n G Cdy + bgMSD u gn n G Cdy) + (bgEND u n G Cdy + bgESD u gn n G Cdy)

/-- `M1` — its magnitude. -/
noncomputable def bgM1 (u gn : ℝ) (n : Nat) (G Cdy : ℝ) : ℝ :=
  (1 + u) * (bgMND u n G Cdy + bgMSD u gn n G Cdy)

/-- `eXS` — the rounding of `x̂ᵢ·Σ(x̂·dx̂)`. -/
noncomputable def bgEXS (u gn : ℝ) (n : Nat) (G Cdy Xh exh : ℝ) : ℝ :=
  FloatModel.mulErr u Xh ((n : ℝ) * (Xh * (G * Cdy))) exh (bgESXD u gn n G Cdy Xh exh)

/-- `MXSf` — its magnitude. -/
noncomputable def bgMXSf (u gn : ℝ) (n : Nat) (G Cdy Xh exh : ℝ) : ℝ :=
  Xh * ((n : ℝ) * (Xh * (G * Cdy))) + bgEXS u gn n G Cdy Xh exh

/-- `e2` — the error after the second subtraction. -/
noncomputable def bgE2 (u gn : ℝ) (n : Nat) (G Cdy Xh exh : ℝ) : ℝ :=
  u * (bgM1 u gn n G Cdy + bgMXSf u gn n G Cdy Xh exh)
    + (bgE1 u gn n G Cdy + bgEXS u gn n G Cdy Xh exh)

/-- `MTr` — the magnitude of the whole three-term bracket. -/
noncomputable def bgMTr (n : Nat) (G Cdy Xh : ℝ) : ℝ :=
  (n : ℝ) * (G * Cdy) + (n : ℝ) * (G * Cdy) + Xh * ((n : ℝ) * (Xh * (G * Cdy)))

/-- `eP` — the rounding of the prefactor `(1/n)·s`, at the supplied inverse-stddev accuracy. -/
noncomputable def bgEP (u : ℝ) (n : Nat) (S es : ℝ) : ℝ :=
  FloatModel.mulErr u (1 / (n : ℝ)) S 0 es

/-- **`FloatModel.bnGradInputBudget` with the rounding unit `u` and the reduction's
    `(1+u)^(n+1) − 1` as PARAMETERS.** `bnGradInputBudget_le` replaces both by rationals once,
    and every numeral downstream is stated on this — `peRoundErrQ`'s pattern, one tier up. -/
noncomputable def bnGradInputBudgetG (u gn : ℝ) (n : Nat) (G Cdy S Xh es exh : ℝ) : ℝ :=
  FloatModel.mulErr u (1 / (n : ℝ) * S) (bgMTr n G Cdy Xh) (bgEP u n S es)
    (bgE2 u gn n G Cdy Xh exh)

/-- The model's budget IS the parameterised one at `(M.u, (1+M.u)^(n+1) − 1)`. `rfl`. -/
theorem bnGradInputBudget_eq_G (M : FloatModel) (n : Nat) (G Cdy S Xh es exh : ℝ) :
    M.bnGradInputBudget n G Cdy S Xh es exh
      = bnGradInputBudgetG M.u ((1 + M.u) ^ (n + 1) - 1) n G Cdy S Xh es exh := rfl

section Mono

variable {u u' gn gn' Cdy Cdy' G S Xh es exh : ℝ} {n : Nat}

theorem bgED_nonneg (hu : 0 ≤ u) (hG : 0 ≤ G) (hC : 0 ≤ Cdy) : 0 ≤ bgED u G Cdy :=
  mulErr_nonneg hu hG hC le_rfl le_rfl

theorem bgED_mono (hu : 0 ≤ u) (huu : u ≤ u') (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') :
    bgED u G Cdy ≤ bgED u' G Cdy' :=
  mulErr_mono hu huu hG le_rfl hC hCC le_rfl le_rfl le_rfl le_rfl

theorem bgESD_nonneg (hu : 0 ≤ u) (hgn : 0 ≤ gn) (hG : 0 ≤ G) (hC : 0 ≤ Cdy) :
    0 ≤ bgESD u gn n G Cdy := by
  have := bgED_nonneg (u := u) (G := G) (Cdy := Cdy) hu hG hC
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  unfold bgESD; positivity

theorem bgESD_mono (hu : 0 ≤ u) (huu : u ≤ u') (hgn : 0 ≤ gn) (hgnn : gn ≤ gn')
    (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') :
    bgESD u gn n G Cdy ≤ bgESD u' gn' n G Cdy' := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have h0 := bgED_nonneg (u := u) (G := G) (Cdy := Cdy) hu hG hC
  have hm := bgED_mono (u := u) (u' := u') (G := G) hu huu hG hC hCC
  have hMD0 : 0 ≤ G * Cdy := mul_nonneg hG hC
  have hMD : G * Cdy ≤ G * Cdy' := mul_le_mul_of_nonneg_left hCC hG
  unfold bgESD
  have h1 : gn * ((n : ℝ) * (G * Cdy + bgED u G Cdy))
      ≤ gn' * ((n : ℝ) * (G * Cdy' + bgED u' G Cdy')) :=
    mul_le_mul hgnn (by nlinarith) (by positivity) (hgn.trans hgnn)
  nlinarith

theorem bgMSD_nonneg (hu : 0 ≤ u) (hgn : 0 ≤ gn) (hG : 0 ≤ G) (hC : 0 ≤ Cdy) :
    0 ≤ bgMSD u gn n G Cdy := by
  have := bgESD_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) hu hgn hG hC
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  unfold bgMSD; positivity

theorem bgMSD_mono (hu : 0 ≤ u) (huu : u ≤ u') (hgn : 0 ≤ gn) (hgnn : gn ≤ gn')
    (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') :
    bgMSD u gn n G Cdy ≤ bgMSD u' gn' n G Cdy' := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have h := bgESD_mono (u := u) (u' := u') (gn := gn) (gn' := gn') (n := n) (G := G)
    hu huu hgn hgnn hG hC hCC
  have hMD : G * Cdy ≤ G * Cdy' := mul_le_mul_of_nonneg_left hCC hG
  unfold bgMSD; nlinarith

theorem bgEXD_nonneg (hu : 0 ≤ u) (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hXh : 0 ≤ Xh)
    (hexh : 0 ≤ exh) : 0 ≤ bgEXD u G Cdy Xh exh :=
  mulErr_nonneg hu hXh (mul_nonneg hG hC) hexh (bgED_nonneg hu hG hC)

theorem bgEXD_mono (hu : 0 ≤ u) (huu : u ≤ u') (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy')
    (hXh : 0 ≤ Xh) (hexh : 0 ≤ exh) :
    bgEXD u G Cdy Xh exh ≤ bgEXD u' G Cdy' Xh exh :=
  mulErr_mono hu huu hXh le_rfl (mul_nonneg hG hC) (mul_le_mul_of_nonneg_left hCC hG)
    hexh le_rfl (bgED_nonneg hu hG hC) (bgED_mono hu huu hG hC hCC)

theorem bgESXD_nonneg (hu : 0 ≤ u) (hgn : 0 ≤ gn) (hG : 0 ≤ G) (hC : 0 ≤ Cdy)
    (hXh : 0 ≤ Xh) (hexh : 0 ≤ exh) : 0 ≤ bgESXD u gn n G Cdy Xh exh := by
  have := bgEXD_nonneg (u := u) (G := G) (Cdy := Cdy) (Xh := Xh) (exh := exh) hu hG hC hXh hexh
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hMD0 : 0 ≤ G * Cdy := mul_nonneg hG hC
  unfold bgESXD; positivity

theorem bgESXD_mono (hu : 0 ≤ u) (huu : u ≤ u') (hgn : 0 ≤ gn) (hgnn : gn ≤ gn')
    (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') (hXh : 0 ≤ Xh) (hexh : 0 ≤ exh) :
    bgESXD u gn n G Cdy Xh exh ≤ bgESXD u' gn' n G Cdy' Xh exh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have h0 := bgEXD_nonneg (u := u) (G := G) (Cdy := Cdy) (Xh := Xh) (exh := exh) hu hG hC hXh hexh
  have hm := bgEXD_mono (u := u) (u' := u') (G := G) (Xh := Xh) (exh := exh)
    hu huu hG hC hCC hXh hexh
  have hMXD0 : 0 ≤ Xh * (G * Cdy) := mul_nonneg hXh (mul_nonneg hG hC)
  have hMXD : Xh * (G * Cdy) ≤ Xh * (G * Cdy') :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hCC hG) hXh
  unfold bgESXD
  have h1 : gn * ((n : ℝ) * (Xh * (G * Cdy) + bgEXD u G Cdy Xh exh))
      ≤ gn' * ((n : ℝ) * (Xh * (G * Cdy') + bgEXD u' G Cdy' Xh exh)) :=
    mul_le_mul hgnn (by nlinarith) (by positivity) (hgn.trans hgnn)
  nlinarith

theorem bgEND_nonneg (hu : 0 ≤ u) (hG : 0 ≤ G) (hC : 0 ≤ Cdy) : 0 ≤ bgEND u n G Cdy :=
  mulErr_nonneg hu (Nat.cast_nonneg n) (mul_nonneg hG hC) le_rfl (bgED_nonneg hu hG hC)

theorem bgEND_mono (hu : 0 ≤ u) (huu : u ≤ u') (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') :
    bgEND u n G Cdy ≤ bgEND u' n G Cdy' :=
  mulErr_mono hu huu (Nat.cast_nonneg n) le_rfl (mul_nonneg hG hC)
    (mul_le_mul_of_nonneg_left hCC hG) le_rfl le_rfl (bgED_nonneg hu hG hC)
    (bgED_mono hu huu hG hC hCC)

theorem bgMND_nonneg (hu : 0 ≤ u) (hG : 0 ≤ G) (hC : 0 ≤ Cdy) : 0 ≤ bgMND u n G Cdy := by
  have := bgEND_nonneg (u := u) (n := n) (G := G) (Cdy := Cdy) hu hG hC
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  unfold bgMND; positivity

theorem bgMND_mono (hu : 0 ≤ u) (huu : u ≤ u') (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') :
    bgMND u n G Cdy ≤ bgMND u' n G Cdy' := by
  have h := bgEND_mono (u := u) (u' := u') (n := n) (G := G) hu huu hG hC hCC
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hMD : G * Cdy ≤ G * Cdy' := mul_le_mul_of_nonneg_left hCC hG
  unfold bgMND; nlinarith

theorem bgE1_nonneg (hu : 0 ≤ u) (hgn : 0 ≤ gn) (hG : 0 ≤ G) (hC : 0 ≤ Cdy) :
    0 ≤ bgE1 u gn n G Cdy := by
  have h1 := bgMND_nonneg (u := u) (n := n) (G := G) (Cdy := Cdy) hu hG hC
  have h2 := bgMSD_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) hu hgn hG hC
  have h3 := bgEND_nonneg (u := u) (n := n) (G := G) (Cdy := Cdy) hu hG hC
  have h4 := bgESD_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) hu hgn hG hC
  unfold bgE1; positivity

theorem bgE1_mono (hu : 0 ≤ u) (huu : u ≤ u') (hgn : 0 ≤ gn) (hgnn : gn ≤ gn')
    (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') :
    bgE1 u gn n G Cdy ≤ bgE1 u' gn' n G Cdy' := by
  have hnd := bgMND_nonneg (u := u) (n := n) (G := G) (Cdy := Cdy) hu hG hC
  have hsd := bgMSD_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) hu hgn hG hC
  have hmnd := bgMND_mono (u := u) (u' := u') (n := n) (G := G) hu huu hG hC hCC
  have hmsd := bgMSD_mono (u := u) (u' := u') (gn := gn) (gn' := gn') (n := n) (G := G)
    hu huu hgn hgnn hG hC hCC
  have hend := bgEND_mono (u := u) (u' := u') (n := n) (G := G) hu huu hG hC hCC
  have hesd := bgESD_mono (u := u) (u' := u') (gn := gn) (gn' := gn') (n := n) (G := G)
    hu huu hgn hgnn hG hC hCC
  unfold bgE1
  have : u * (bgMND u n G Cdy + bgMSD u gn n G Cdy)
      ≤ u' * (bgMND u' n G Cdy' + bgMSD u' gn' n G Cdy') :=
    mul_le_mul huu (by linarith) (by linarith) (hu.trans huu)
  linarith

theorem bgM1_nonneg (hu : 0 ≤ u) (hgn : 0 ≤ gn) (hG : 0 ≤ G) (hC : 0 ≤ Cdy) :
    0 ≤ bgM1 u gn n G Cdy := by
  have h1 := bgMND_nonneg (u := u) (n := n) (G := G) (Cdy := Cdy) hu hG hC
  have h2 := bgMSD_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) hu hgn hG hC
  unfold bgM1; positivity

theorem bgM1_mono (hu : 0 ≤ u) (huu : u ≤ u') (hgn : 0 ≤ gn) (hgnn : gn ≤ gn')
    (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') :
    bgM1 u gn n G Cdy ≤ bgM1 u' gn' n G Cdy' := by
  have hnd := bgMND_nonneg (u := u) (n := n) (G := G) (Cdy := Cdy) hu hG hC
  have hsd := bgMSD_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) hu hgn hG hC
  have hmnd := bgMND_mono (u := u) (u' := u') (n := n) (G := G) hu huu hG hC hCC
  have hmsd := bgMSD_mono (u := u) (u' := u') (gn := gn) (gn' := gn') (n := n) (G := G)
    hu huu hgn hgnn hG hC hCC
  unfold bgM1
  exact mul_le_mul (by linarith) (by linarith) (by linarith) (by linarith)

theorem bgEXS_nonneg (hu : 0 ≤ u) (hgn : 0 ≤ gn) (hG : 0 ≤ G) (hC : 0 ≤ Cdy)
    (hXh : 0 ≤ Xh) (hexh : 0 ≤ exh) : 0 ≤ bgEXS u gn n G Cdy Xh exh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  exact mulErr_nonneg hu hXh (by positivity) hexh (bgESXD_nonneg hu hgn hG hC hXh hexh)

theorem bgEXS_mono (hu : 0 ≤ u) (huu : u ≤ u') (hgn : 0 ≤ gn) (hgnn : gn ≤ gn')
    (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') (hXh : 0 ≤ Xh) (hexh : 0 ≤ exh) :
    bgEXS u gn n G Cdy Xh exh ≤ bgEXS u' gn' n G Cdy' Xh exh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hMD : G * Cdy ≤ G * Cdy' := mul_le_mul_of_nonneg_left hCC hG
  refine mulErr_mono hu huu hXh le_rfl (by positivity) ?_ hexh le_rfl
    (bgESXD_nonneg hu hgn hG hC hXh hexh)
    (bgESXD_mono hu huu hgn hgnn hG hC hCC hXh hexh)
  exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hMD hXh) hn0

theorem bgMXSf_nonneg (hu : 0 ≤ u) (hgn : 0 ≤ gn) (hG : 0 ≤ G) (hC : 0 ≤ Cdy)
    (hXh : 0 ≤ Xh) (hexh : 0 ≤ exh) : 0 ≤ bgMXSf u gn n G Cdy Xh exh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have := bgEXS_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) (Xh := Xh)
    (exh := exh) hu hgn hG hC hXh hexh
  have hMD0 : 0 ≤ G * Cdy := mul_nonneg hG hC
  unfold bgMXSf; positivity

theorem bgMXSf_mono (hu : 0 ≤ u) (huu : u ≤ u') (hgn : 0 ≤ gn) (hgnn : gn ≤ gn')
    (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') (hXh : 0 ≤ Xh) (hexh : 0 ≤ exh) :
    bgMXSf u gn n G Cdy Xh exh ≤ bgMXSf u' gn' n G Cdy' Xh exh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have h := bgEXS_mono (u := u) (u' := u') (gn := gn) (gn' := gn') (n := n) (G := G)
    (Xh := Xh) (exh := exh) hu huu hgn hgnn hG hC hCC hXh hexh
  have hMD : G * Cdy ≤ G * Cdy' := mul_le_mul_of_nonneg_left hCC hG
  have hlead : Xh * ((n : ℝ) * (Xh * (G * Cdy))) ≤ Xh * ((n : ℝ) * (Xh * (G * Cdy'))) :=
    mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hMD hXh) hn0) hXh
  unfold bgMXSf; linarith

theorem bgE2_nonneg (hu : 0 ≤ u) (hgn : 0 ≤ gn) (hG : 0 ≤ G) (hC : 0 ≤ Cdy)
    (hXh : 0 ≤ Xh) (hexh : 0 ≤ exh) : 0 ≤ bgE2 u gn n G Cdy Xh exh := by
  have h1 := bgM1_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) hu hgn hG hC
  have h2 := bgMXSf_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) (Xh := Xh)
    (exh := exh) hu hgn hG hC hXh hexh
  have h3 := bgE1_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) hu hgn hG hC
  have h4 := bgEXS_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) (Xh := Xh)
    (exh := exh) hu hgn hG hC hXh hexh
  unfold bgE2; positivity

theorem bgE2_mono (hu : 0 ≤ u) (huu : u ≤ u') (hgn : 0 ≤ gn) (hgnn : gn ≤ gn')
    (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') (hXh : 0 ≤ Xh) (hexh : 0 ≤ exh) :
    bgE2 u gn n G Cdy Xh exh ≤ bgE2 u' gn' n G Cdy' Xh exh := by
  have hm1 := bgM1_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) hu hgn hG hC
  have hmx := bgMXSf_nonneg (u := u) (gn := gn) (n := n) (G := G) (Cdy := Cdy) (Xh := Xh)
    (exh := exh) hu hgn hG hC hXh hexh
  have h1 := bgM1_mono (u := u) (u' := u') (gn := gn) (gn' := gn') (n := n) (G := G)
    hu huu hgn hgnn hG hC hCC
  have h2 := bgMXSf_mono (u := u) (u' := u') (gn := gn) (gn' := gn') (n := n) (G := G)
    (Xh := Xh) (exh := exh) hu huu hgn hgnn hG hC hCC hXh hexh
  have h3 := bgE1_mono (u := u) (u' := u') (gn := gn) (gn' := gn') (n := n) (G := G)
    hu huu hgn hgnn hG hC hCC
  have h4 := bgEXS_mono (u := u) (u' := u') (gn := gn) (gn' := gn') (n := n) (G := G)
    (Xh := Xh) (exh := exh) hu huu hgn hgnn hG hC hCC hXh hexh
  unfold bgE2
  have : u * (bgM1 u gn n G Cdy + bgMXSf u gn n G Cdy Xh exh)
      ≤ u' * (bgM1 u' gn' n G Cdy' + bgMXSf u' gn' n G Cdy' Xh exh) :=
    mul_le_mul huu (by linarith) (by linarith) (hu.trans huu)
  linarith

theorem bgMTr_nonneg (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hXh : 0 ≤ Xh) : 0 ≤ bgMTr n G Cdy Xh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hMD0 : 0 ≤ G * Cdy := mul_nonneg hG hC
  unfold bgMTr; positivity

theorem bgMTr_mono (hG : 0 ≤ G) (_hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') (hXh : 0 ≤ Xh) :
    bgMTr n G Cdy Xh ≤ bgMTr n G Cdy' Xh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hMD : G * Cdy ≤ G * Cdy' := mul_le_mul_of_nonneg_left hCC hG
  have h1 : (n : ℝ) * (G * Cdy) ≤ (n : ℝ) * (G * Cdy') := mul_le_mul_of_nonneg_left hMD hn0
  have h2 : Xh * ((n : ℝ) * (Xh * (G * Cdy))) ≤ Xh * ((n : ℝ) * (Xh * (G * Cdy'))) :=
    mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hMD hXh) hn0) hXh
  unfold bgMTr; linarith

/-- ⭐ **`bnGradInputBudgetG` is monotone** in the rounding unit, the γ-bound `gn` and the
    cotangent window — the fact that lets a symbolic `M.u`, a symbolic power and a symbolic
    window all be replaced by rationals before `norm_num` sees the expression. The
    `bnNormBudget_mono` of the backward. -/
theorem bnGradInputBudgetG_mono (hu : 0 ≤ u) (huu : u ≤ u') (hgn : 0 ≤ gn) (hgnn : gn ≤ gn')
    (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cdy') (hG : 0 ≤ G) (hS : 0 ≤ S) (hXh : 0 ≤ Xh)
    (hes : 0 ≤ es) (hexh : 0 ≤ exh) :
    bnGradInputBudgetG u gn n G Cdy S Xh es exh
      ≤ bnGradInputBudgetG u' gn' n G Cdy' S Xh es exh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  unfold bnGradInputBudgetG bgEP
  exact mulErr_mono hu huu (by positivity) le_rfl (bgMTr_nonneg hG hC hXh)
    (bgMTr_mono hG hC hCC hXh)
    (mulErr_nonneg hu (by positivity) hS le_rfl hes)
    (mulErr_mono hu huu (by positivity) le_rfl hS le_rfl le_rfl le_rfl hes le_rfl)
    (bgE2_nonneg hu hgn hG hC hXh hexh) (bgE2_mono hu huu hgn hgnn hG hC hCC hXh hexh)

end Mono

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ The budget is HOMOGENEOUS in the cotangent window
-- ════════════════════════════════════════════════════════════════

/-! Every term of `bnGradInputBudgetG` carries exactly one factor of the cotangent bound — the
same linearity that makes a backward fold exist at all (§3.7). `bgEP` is the one `Cdy`-free
piece, and it enters as a `mulErr` *ea*, which is multiplied by a degree-1 *C*.

⭐ **This is not a curiosity, it is what makes the numerals checkable.** Stated directly, a
site's inequality is a forty-node tree at 250-digit numerals and `norm_num` needs seconds per
site; factored as `Cdy · budget(1)`, it is `Ā * K ≤ Ā'` against ONE constant per feature-map
size, and the expensive evaluation happens five times instead of sixty-eight. ⚠ The FORWARD
budgets are NOT homogeneous — a bias breaks it — which is why they never needed this. -/

section Homog

variable (u gn : ℝ) (n : Nat) (G Cdy S Xh es exh : ℝ)

theorem bgED_homog : bgED u G Cdy = Cdy * bgED u G 1 := by
  unfold bgED FloatModel.mulErr; ring

theorem bgESD_homog : bgESD u gn n G Cdy = Cdy * bgESD u gn n G 1 := by
  unfold bgESD; rw [bgED_homog u G Cdy]; ring

theorem bgMSD_homog : bgMSD u gn n G Cdy = Cdy * bgMSD u gn n G 1 := by
  unfold bgMSD; rw [bgESD_homog u gn n G Cdy]; ring

theorem bgEXD_homog : bgEXD u G Cdy Xh exh = Cdy * bgEXD u G 1 Xh exh := by
  unfold bgEXD FloatModel.mulErr; rw [bgED_homog u G Cdy]; ring

theorem bgESXD_homog : bgESXD u gn n G Cdy Xh exh = Cdy * bgESXD u gn n G 1 Xh exh := by
  unfold bgESXD; rw [bgEXD_homog u G Cdy Xh exh]; ring

theorem bgEND_homog : bgEND u n G Cdy = Cdy * bgEND u n G 1 := by
  unfold bgEND FloatModel.mulErr; rw [bgED_homog u G Cdy]; ring

theorem bgMND_homog : bgMND u n G Cdy = Cdy * bgMND u n G 1 := by
  unfold bgMND; rw [bgEND_homog u n G Cdy]; ring

theorem bgE1_homog : bgE1 u gn n G Cdy = Cdy * bgE1 u gn n G 1 := by
  unfold bgE1
  rw [bgMND_homog u n G Cdy, bgMSD_homog u gn n G Cdy, bgEND_homog u n G Cdy,
    bgESD_homog u gn n G Cdy]
  ring

theorem bgM1_homog : bgM1 u gn n G Cdy = Cdy * bgM1 u gn n G 1 := by
  unfold bgM1; rw [bgMND_homog u n G Cdy, bgMSD_homog u gn n G Cdy]; ring

theorem bgEXS_homog : bgEXS u gn n G Cdy Xh exh = Cdy * bgEXS u gn n G 1 Xh exh := by
  unfold bgEXS FloatModel.mulErr; rw [bgESXD_homog u gn n G Cdy Xh exh]; ring

theorem bgMXSf_homog : bgMXSf u gn n G Cdy Xh exh = Cdy * bgMXSf u gn n G 1 Xh exh := by
  unfold bgMXSf; rw [bgEXS_homog u gn n G Cdy Xh exh]; ring

theorem bgE2_homog : bgE2 u gn n G Cdy Xh exh = Cdy * bgE2 u gn n G 1 Xh exh := by
  unfold bgE2
  rw [bgM1_homog u gn n G Cdy, bgMXSf_homog u gn n G Cdy Xh exh, bgE1_homog u gn n G Cdy,
    bgEXS_homog u gn n G Cdy Xh exh]
  ring

theorem bgMTr_homog : bgMTr n G Cdy Xh = Cdy * bgMTr n G 1 Xh := by
  unfold bgMTr; ring

/-- ⭐⭐ **The BatchNorm backward's budget is LINEAR in the cotangent window.** -/
theorem bnGradInputBudgetG_homog :
    bnGradInputBudgetG u gn n G Cdy S Xh es exh
      = Cdy * bnGradInputBudgetG u gn n G 1 S Xh es exh := by
  unfold bnGradInputBudgetG FloatModel.mulErr
  rw [bgMTr_homog n G Cdy Xh, bgE2_homog u gn n G Cdy Xh exh]
  ring

/-- ⭐ And so is the real BatchNorm backward's magnitude — it IS a Lipschitz constant. -/
theorem bnGradInputReMag_homog :
    bnGradInputReMag n G Cdy S Xh = Cdy * bnGradInputReMag n G 1 S Xh := by
  unfold bnGradInputReMag; ring

end Homog

/-- ⭐ **The one bridge from the model to the numerals**: at `M.u ≤ q` and a `gamma_num`-style
    rational bound on the reduction's power, the model's BN-backward budget at any window
    `Cdy ≤ Cd` is under the rational budget at `Cd`. The `patchEmbedRoundErr_le` of the
    backward. -/
theorem bnGradInputBudget_le (M : FloatModel) (n : Nat) {G Cdy Cd S Xh es exh q gn : ℝ}
    (hq : M.u ≤ q) (hgn : (1 + M.u) ^ (n + 1) - 1 ≤ gn)
    (hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cd) (hG : 0 ≤ G) (hS : 0 ≤ S) (hXh : 0 ≤ Xh)
    (hes : 0 ≤ es) (hexh : 0 ≤ exh) :
    M.bnGradInputBudget n G Cdy S Xh es exh ≤ bnGradInputBudgetG q gn n G Cd S Xh es exh := by
  rw [bnGradInputBudget_eq_G]
  exact bnGradInputBudgetG_mono M.u_nonneg hq
    (sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))) hgn hC hCC hG hS hXh hes hexh

/-- `bnGradInputReMag` is monotone in the cotangent bound — and LINEAR in it, which is the whole
    reason a backward fold exists where the forward's does not (§3.7). -/
theorem bnGradInputReMag_mono (n : Nat) {G Cdy Cd S Xh : ℝ}
    (_hC : 0 ≤ Cdy) (hCC : Cdy ≤ Cd) (hG : 0 ≤ G) (hS : 0 ≤ S) (hXh : 0 ≤ Xh) :
    bnGradInputReMag n G Cdy S Xh ≤ bnGradInputReMag n G Cd S Xh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  unfold bnGradInputReMag
  have hMD : G * Cdy ≤ G * Cd := mul_le_mul_of_nonneg_left hCC hG
  have h : (n : ℝ) * (G * Cdy) + (n : ℝ) * (G * Cdy) + Xh * ((n : ℝ) * (Xh * (G * Cdy)))
      ≤ (n : ℝ) * (G * Cd) + (n : ℝ) * (G * Cd) + Xh * ((n : ℝ) * (Xh * (G * Cd))) := by
    have h1 : (n : ℝ) * (G * Cdy) ≤ (n : ℝ) * (G * Cd) := mul_le_mul_of_nonneg_left hMD hn0
    have h2 : Xh * ((n : ℝ) * (Xh * (G * Cdy))) ≤ Xh * ((n : ℝ) * (Xh * (G * Cd))) :=
      mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hMD hXh) hn0) hXh
    linarith
  have hpre : (0 : ℝ) ≤ 1 / (n : ℝ) * S := by positivity
  exact mul_le_mul_of_nonneg_left h hpre


-- ════════════════════════════════════════════════════════════════
-- § The BatchNorm BACKWARD, migrated to `FloatBridgesTo`
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **`|x̂| ≤ X` from the standardisation bound, at a RATIONAL `X`.** `bnXhat_sq_le` gives
    `x̂² ≤ n`; this turns it into a numeral wherever `n ≤ X²`, which for the square feature maps
    a conv net has is exact (`h·w = 56² ⇒ X = 56`). ⛔ It is what makes a whole-net backward
    number exist at all: `Xh` enters the fold as `Xh²`, and deriving it from the FORWARD's
    certified window instead is the difference between `10²⁸⁸` and `10⁷²⁷¹` (§3.7). The lemma it
    rests on was written for the realistic-seal work and sat in `Foundation/ResNet34.lean`. -/
theorem bnXhat_abs_le_num {n : Nat} {ε X : ℝ} (hε : 0 < ε) (v : Vec n)
    (hX : 0 ≤ X) (hnX : (n : ℝ) ≤ X ^ 2) (k : Fin n) : |bnXhat n ε v k| ≤ X := by
  have hsq := bnXhat_sq_le ε hε v k
  have habs : |bnXhat n ε v k| ^ 2 ≤ X ^ 2 := by
    rw [sq_abs]; linarith
  nlinarith [abs_nonneg (bnXhat n ε v k)]

/-- The float flat BN backward, NAMED (`formalization.yaml` fidelity §4d) — the deployed
    three-term input-gradient at the supplied float inverse-stddev and normalised activation. -/
noncomputable def bnBackF {n : Nat} (M : FloatModel) (γ fs : ℝ) (fxh : Vec n) :
    Vec n → Vec n := fun dy i => M.bnGradInputF γ fs fxh dy i

/-- **The flat BatchNorm backward float-bridges TO its float map**, over the cotangent. ⭐ Its
    `mod` is `budget(A) + ReMag(e)` and `ReMag` is LINEAR in `e` — a VJP at a fixed point is a
    linear map, so §0.1's quadratic never appears (§3.7). -/
noncomputable def floatBridgesTo_bnBack {n : Nat} (M : FloatModel) {ε γ : ℝ} (x fxh : Vec n)
    (fs : ℝ) {G S Xh es exh : ℝ} (hn : 0 < n) (hγ : |γ| ≤ G)
    (hs : |fs - bnIstd n x ε| ≤ es) (hSabs : |bnIstd n x ε| ≤ S)
    (hxh : ∀ i, |bnXhat n ε x i| ≤ Xh) (hfxh : ∀ i, |fxh i - bnXhat n ε x i| ≤ exh) :
    FloatBridgesTo (fun dy => bn_grad_input n ε γ x dy) (bnBackF M γ fs fxh) :=
  ⟨fun A => bnGradInputReMag n G A S Xh + M.bnGradInputBudget n G A S Xh es exh,
   fun A e => M.bnGradInputBudget n G A S Xh es exh + bnGradInputReMag n G e S Xh,
   fun A hA =>
     ⟨(floatClose_bnBack M x fxh fs hn hγ hs hSabs hxh hfxh (A := A)).cod_nonneg hA hn,
      floatClose_bnBack M x fxh fs hn hγ hs hSabs hxh hfxh⟩⟩

/-- The float per-channel BN backward (Mat-split layout): channel `c` runs its own three-term
    input-gradient over its `m`-wide spatial slab. -/
noncomputable def bnPerChannelFlatBackFV {oc m : Nat} (M : FloatModel) (γ : Vec oc)
    (fs : Fin oc → ℝ) (fxh : Fin oc → Vec m) : Vec (oc * m) → Vec (oc * m) :=
  perRowIdxFlat oc m (fun c => bnBackF M (γ c) (fs c) (fxh c))

/-- **Per-channel BN backward (Mat-split layout) float-bridges TO its float map** — the
    block-diagonal `FloatClose.perRowIdx` lift of `floatClose_bnBack`, at the uniform budget the
    shared `G`/`S`/`Xh`/`es`/`exh` give. -/
noncomputable def floatBridgesTo_bnPerChannelFlatBack {oc m : Nat} (M : FloatModel) {ε : ℝ}
    (γ : Vec oc) (X : Vec (oc * m)) (fs : Fin oc → ℝ) (fxh : Fin oc → Vec m)
    {G S Xh es exh : ℝ} (hoc : 0 < oc) (hm : 0 < m)
    (hγ : ∀ c, |γ c| ≤ G)
    (hs : ∀ c, |fs c - bnIstd m (Mat.unflatten X c) ε| ≤ es)
    (hSabs : ∀ c, |bnIstd m (Mat.unflatten X c) ε| ≤ S)
    (hxh : ∀ c i, |bnXhat m ε (Mat.unflatten X c) i| ≤ Xh)
    (hfxh : ∀ c i, |fxh c i - bnXhat m ε (Mat.unflatten X c) i| ≤ exh) :
    FloatBridgesTo (fun dy => bnPerChannel_grad_input oc m ε γ X dy)
      (bnPerChannelFlatBackFV M γ fs fxh) :=
  ⟨fun A => bnGradInputReMag m G A S Xh + M.bnGradInputBudget m G A S Xh es exh,
   fun A e => M.bnGradInputBudget m G A S Xh es exh + bnGradInputReMag m G e S Xh,
   fun A hA =>
     have hg := fun c : Fin oc => floatClose_bnBack M (Mat.unflatten X c) (fxh c) (fs c) hm
       (hγ c) (hs c) (hSabs c) (hxh c) (hfxh c) (A := A)
     have hpr := FloatClose.perRowIdx (d := m) oc hg
     ⟨hpr.cod_nonneg hA (Nat.mul_pos hoc hm), hpr⟩⟩

/-- The float per-channel BN backward in the network Tensor3 layout — the same `reassoc`
    conjugation as the real op, both permutations exact in float. -/
noncomputable def bnPerChannelTensor3BackFV {oc h w : Nat} (M : FloatModel) (γ : Vec oc)
    (fs : Fin oc → ℝ) (fxh : Fin oc → Vec (h * w)) : Vec (oc * h * w) → Vec (oc * h * w) :=
  reassocBack oc h w ∘ bnPerChannelFlatBackFV M γ fs fxh ∘ reassocFwd oc h w

/-- ⭐ **Per-channel BatchNorm BACKWARD (network Tensor3 layout) float-bridges TO its float
    map** — the `FloatBridgesTo` peer of `floatBridges_bnPerChannelBack`. The backward keystone
    every deep net's input-gradient folds; `Maps.bnPerChannelBack` puts numerals on it. -/
noncomputable def floatBridgesTo_bnPerChannelBack {oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ : Vec oc) (x : Vec (oc * h * w)) (fs : Fin oc → ℝ) (fxh : Fin oc → Vec (h * w))
    {G S Xh es exh : ℝ} (hoc : 0 < oc) (hhw : 0 < h * w)
    (hγ : ∀ c, |γ c| ≤ G)
    (hs : ∀ c, |fs c - bnIstd (h * w) (Mat.unflatten (reassocFwd oc h w x) c) ε| ≤ es)
    (hSabs : ∀ c, |bnIstd (h * w) (Mat.unflatten (reassocFwd oc h w x) c) ε| ≤ S)
    (hxh : ∀ c i, |bnXhat (h * w) ε (Mat.unflatten (reassocFwd oc h w x) c) i| ≤ Xh)
    (hfxh : ∀ c i,
      |fxh c i - bnXhat (h * w) ε (Mat.unflatten (reassocFwd oc h w x) c) i| ≤ exh) :
    FloatBridgesTo (fun dy => bnPerChannelTensor3_grad_input oc h w ε γ x dy)
      (bnPerChannelTensor3BackFV M γ fs fxh) :=
  ⟨fun A => bnGradInputReMag (h * w) G A S Xh + M.bnGradInputBudget (h * w) G A S Xh es exh,
   fun A e => M.bnGradInputBudget (h * w) G A S Xh es exh + bnGradInputReMag (h * w) G e S Xh,
   (((floatBridgesTo_gather (reassocEquiv oc h w)).comp
      (floatBridgesTo_bnPerChannelFlatBack M γ (reassocFwd oc h w x) fs fxh hoc hhw
        hγ hs hSabs hxh hfxh)).comp
      (floatBridgesTo_gather (reassocEquiv oc h w).symm)).close⟩

namespace FloatBridgesTo

/-- ⭐⭐ **An envelope through one per-channel BatchNorm BACKWARD site.** Both closing
    inequalities are stated over the RATIONAL budget `bnGradInputBudgetG` at the input window
    `Ā`, and transported to every `A ≤ Ā` by the two monotonicity lemmas above — the leaf is
    linear in the cotangent, so there is nothing cleverer to do and nothing is lost.

    ⛔ Unlike a forward LayerNorm site this one is NOT capped and does not need to be: the
    statistics are read off the saved activations, which the cotangent does not perturb, so the
    modulus is linear in the inherited error (§3.7). This is the only leaf in the repo that
    carries an honest fold through a TRAINING-mode normalisation. -/
theorem Maps.bnPerChannelBack {oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ : Vec oc) (x : Vec (oc * h * w)) (fs : Fin oc → ℝ) (fxh : Fin oc → Vec (h * w))
    {G S Xh es exh : ℝ} (hoc : 0 < oc) (hhw : 0 < h * w)
    (hγ : ∀ c, |γ c| ≤ G)
    (hs : ∀ c, |fs c - bnIstd (h * w) (Mat.unflatten (reassocFwd oc h w x) c) ε| ≤ es)
    (hSabs : ∀ c, |bnIstd (h * w) (Mat.unflatten (reassocFwd oc h w x) c) ε| ≤ S)
    (hxh : ∀ c i, |bnXhat (h * w) ε (Mat.unflatten (reassocFwd oc h w x) c) i| ≤ Xh)
    (hfxh : ∀ c i,
      |fxh c i - bnXhat (h * w) ε (Mat.unflatten (reassocFwd oc h w x) c) i| ≤ exh)
    {q gn Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hgn : (1 + M.u) ^ (h * w + 1) - 1 ≤ gn)
    (hG0 : 0 ≤ G) (hS0 : 0 ≤ S) (hXh0 : 0 ≤ Xh) (hes0 : 0 ≤ es) (hexh0 : 0 ≤ exh)
    (hĀ' : bnGradInputReMag (h * w) G Ā S Xh
            + bnGradInputBudgetG q gn (h * w) G Ā S Xh es exh ≤ Ā')
    (hĒ' : bnGradInputBudgetG q gn (h * w) G Ā S Xh es exh
            + bnGradInputReMag (h * w) G Ē S Xh ≤ Ē') :
    (floatBridgesTo_bnPerChannelBack M γ x fs fxh hoc hhw hγ hs hSabs hxh hfxh).Maps
      Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show bnGradInputReMag (h * w) G A S Xh + M.bnGradInputBudget (h * w) G A S Xh es exh ≤ Ā'
    have h1 := bnGradInputReMag_mono (h * w) (G := G) (S := S) (Xh := Xh) h0 hle hG0 hS0 hXh0
    have h2 := bnGradInputBudget_le M (h * w) (G := G) (S := S) (Xh := Xh) (es := es)
      (exh := exh) hq hgn h0 hle hG0 hS0 hXh0 hes0 hexh0
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show M.bnGradInputBudget (h * w) G A S Xh es exh + bnGradInputReMag (h * w) G E S Xh ≤ Ē'
    have h1 := bnGradInputReMag_mono (h * w) (G := G) (S := S) (Xh := Xh) hE0 hEle hG0 hS0 hXh0
    have h2 := bnGradInputBudget_le M (h * w) (G := G) (S := S) (Xh := Xh) (es := es)
      (exh := exh) hq hgn h0 hle hG0 hS0 hXh0 hes0 hexh0
    linarith

/-- The real BatchNorm backward's magnitude is nonnegative. -/
theorem bnGradInputReMag_nonneg (n : Nat) {G Cdy S Xh : ℝ} (hG : 0 ≤ G) (hC : 0 ≤ Cdy)
    (hS : 0 ≤ S) (hXh : 0 ≤ Xh) : 0 ≤ bnGradInputReMag n G Cdy S Xh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  unfold bnGradInputReMag
  positivity

/-- The rational BatchNorm-backward budget is nonnegative. -/
theorem bnGradInputBudgetG_nonneg (n : Nat) {u gn G Cdy S Xh es exh : ℝ}
    (hu : 0 ≤ u) (hgn : 0 ≤ gn) (hG : 0 ≤ G) (hC : 0 ≤ Cdy) (hS : 0 ≤ S) (hXh : 0 ≤ Xh)
    (hes : 0 ≤ es) (hexh : 0 ≤ exh) : 0 ≤ bnGradInputBudgetG u gn n G Cdy S Xh es exh := by
  have hn0 : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  unfold bnGradInputBudgetG bgEP
  exact mulErr_nonneg hu (by positivity) (bgMTr_nonneg hG hC hXh)
    (mulErr_nonneg hu (by positivity) hS le_rfl hes) (bgE2_nonneg hu hgn hG hC hXh hexh)

/-- ⭐⭐ **An envelope through a BatchNorm BACKWARD site, stated at the PER-UNIT GAIN.** The same
    claim as `Maps.bnPerChannelBack`, with the site's two constants — the real map's Lipschitz
    constant `Kr` and the rounding budget `Kb`, both at cotangent window `1` — factored out by
    `bnGradInputReMag_homog` / `bnGradInputBudgetG_homog`.

    ⭐ **This is the form a whole-net chain must use.** Stated directly, each site's two closing
    inequalities are a forty-node tree at the chain's full magnitude; here they are `Ā * (Kr+Kb)`
    and `Ā * Kb + Ē * Kr`, and the expensive evaluation moves into `hKr`/`hKb`, which depend only
    on the feature-map size — five of them for a ResNet-34, not sixty-eight. ⚠ The forward
    budgets have no such form: a bias makes their leaves affine rather than linear. -/
theorem Maps.bnPerChannelBackGain {oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ : Vec oc) (x : Vec (oc * h * w)) (fs : Fin oc → ℝ) (fxh : Fin oc → Vec (h * w))
    {G S Xh es exh : ℝ} (hoc : 0 < oc) (hhw : 0 < h * w)
    (hγ : ∀ c, |γ c| ≤ G)
    (hs : ∀ c, |fs c - bnIstd (h * w) (Mat.unflatten (reassocFwd oc h w x) c) ε| ≤ es)
    (hSabs : ∀ c, |bnIstd (h * w) (Mat.unflatten (reassocFwd oc h w x) c) ε| ≤ S)
    (hxh : ∀ c i, |bnXhat (h * w) ε (Mat.unflatten (reassocFwd oc h w x) c) i| ≤ Xh)
    (hfxh : ∀ c i,
      |fxh c i - bnXhat (h * w) ε (Mat.unflatten (reassocFwd oc h w x) c) i| ≤ exh)
    {q gn Kr Kb Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hgn : (1 + M.u) ^ (h * w + 1) - 1 ≤ gn)
    (hG0 : 0 ≤ G) (hS0 : 0 ≤ S) (hXh0 : 0 ≤ Xh) (hes0 : 0 ≤ es) (hexh0 : 0 ≤ exh)
    (hKr : bnGradInputReMag (h * w) G 1 S Xh ≤ Kr)
    (hKb : bnGradInputBudgetG q gn (h * w) G 1 S Xh es exh ≤ Kb)
    (hĀ0 : 0 ≤ Ā) (hĒ0 : 0 ≤ Ē)
    (hĀ' : Ā * (Kr + Kb) ≤ Ā') (hĒ' : Ā * Kb + Ē * Kr ≤ Ē') :
    (floatBridgesTo_bnPerChannelBack M γ x fs fxh hoc hhw hγ hs hSabs hxh hfxh).Maps
      Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    show bnGradInputReMag (h * w) G A S Xh + M.bnGradInputBudget (h * w) G A S Xh es exh ≤ Ā'
    have hr1 : 0 ≤ bnGradInputReMag (h * w) G 1 S Xh :=
      bnGradInputReMag_nonneg (h * w) hG0 zero_le_one hS0 hXh0
    have hr : bnGradInputReMag (h * w) G A S Xh ≤ Ā * Kr := by
      rw [bnGradInputReMag_homog]
      exact le_trans (mul_le_mul_of_nonneg_right hle hr1)
        (mul_le_mul_of_nonneg_left hKr hĀ0)
    have hb := bnGradInputBudget_le M (h * w) (Cd := Ā) hq hgn h0 hle hG0 hS0 hXh0 hes0 hexh0
    rw [bnGradInputBudgetG_homog] at hb
    have hb2 : Ā * bnGradInputBudgetG q gn (h * w) G 1 S Xh es exh ≤ Ā * Kb :=
      mul_le_mul_of_nonneg_left hKb hĀ0
    nlinarith
  mod_le := fun A E h0 hE0 hle hEle => by
    show M.bnGradInputBudget (h * w) G A S Xh es exh + bnGradInputReMag (h * w) G E S Xh ≤ Ē'
    have hr1 : 0 ≤ bnGradInputReMag (h * w) G 1 S Xh :=
      bnGradInputReMag_nonneg (h * w) hG0 zero_le_one hS0 hXh0
    have hr : bnGradInputReMag (h * w) G E S Xh ≤ Ē * Kr := by
      rw [bnGradInputReMag_homog]
      exact le_trans (mul_le_mul_of_nonneg_right hEle hr1)
        (mul_le_mul_of_nonneg_left hKr hĒ0)
    have hb := bnGradInputBudget_le M (h * w) (Cd := Ā) hq hgn h0 hle hG0 hS0 hXh0 hes0 hexh0
    rw [bnGradInputBudgetG_homog] at hb
    have hb2 : Ā * bnGradInputBudgetG q gn (h * w) G 1 S Xh es exh ≤ Ā * Kb :=
      mul_le_mul_of_nonneg_left hKb hĀ0
    linarith

-- ════════════════════════════════════════════════════════════════
-- § The structural backward leaves (exact in float)
-- ════════════════════════════════════════════════════════════════

/-- The ReLU kink mask passes an envelope through unchanged (a select rounds nothing). -/
theorem Maps.reluMaskBack {n : Nat} (cond : Fin n → Prop) [DecidablePred cond] {Ā Ē : ℝ} :
    (floatBridgesTo_reluMaskBack cond).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

/-- The max-pool backward is a scatter to the argmax cells — exact, envelope unchanged. -/
theorem Maps.maxPoolBack {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w)) {Ā Ē : ℝ} :
    (floatBridgesTo_maxPoolBack x).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

/-- The stride-2 decimation backward is a zero-fill scatter — exact, envelope unchanged. -/
theorem Maps.decimateBack (oc h w : Nat) {Ā Ē : ℝ} :
    (floatBridgesTo_decimateBack oc h w).Maps Ā Ē Ā Ē :=
  ⟨fun _ _ hle => hle, fun _ _ _ _ _ hEle => hEle⟩

/-- **An envelope through the GAP backward** — broadcast ÷ `h·w`, one rounded multiply.
    ⭐ Magnitude-NONincreasing (`h·w ≥ 1`): the only stage of a backward chain that shrinks. -/
theorem Maps.gapBack (M : FloatModel) (c h w : Nat) (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    {q Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hĀ' : 1 / ((h : ℝ) * (w : ℝ)) * Ā
            + FloatModel.mulErr q (1 / ((h : ℝ) * (w : ℝ))) Ā 0 0 ≤ Ā')
    (hĒ' : FloatModel.mulErr q (1 / ((h : ℝ) * (w : ℝ))) Ā 0 0
            + 1 / ((h : ℝ) * (w : ℝ)) * Ē ≤ Ē') :
    (floatBridgesTo_gapBack M c h w hc hh hw).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    have hD : (0 : ℝ) < (h : ℝ) * (w : ℝ) := by
      have h1 : (0 : ℝ) < (h : ℝ) := by exact_mod_cast hh
      have h2 : (0 : ℝ) < (w : ℝ) := by exact_mod_cast hw
      positivity
    have hinv : (0 : ℝ) ≤ 1 / ((h : ℝ) * (w : ℝ)) := by positivity
    show 1 / ((h : ℝ) * (w : ℝ)) * A + FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0 ≤ Ā'
    have h1 : 1 / ((h : ℝ) * (w : ℝ)) * A ≤ 1 / ((h : ℝ) * (w : ℝ)) * Ā :=
      mul_le_mul_of_nonneg_left hle hinv
    have h2 := mulErr_mono M.u_nonneg hq hinv le_rfl h0 hle le_rfl le_rfl le_rfl le_rfl
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    have hD : (0 : ℝ) < (h : ℝ) * (w : ℝ) := by
      have h1 : (0 : ℝ) < (h : ℝ) := by exact_mod_cast hh
      have h2 : (0 : ℝ) < (w : ℝ) := by exact_mod_cast hw
      positivity
    have hinv : (0 : ℝ) ≤ 1 / ((h : ℝ) * (w : ℝ)) := by positivity
    show FloatModel.mulErr M.u (1 / ((h : ℝ) * (w : ℝ))) A 0 0
        + 1 / ((h : ℝ) * (w : ℝ)) * E ≤ Ē'
    have h1 : 1 / ((h : ℝ) * (w : ℝ)) * E ≤ 1 / ((h : ℝ) * (w : ℝ)) * Ē :=
      mul_le_mul_of_nonneg_left hEle hinv
    have h2 := mulErr_mono M.u_nonneg hq hinv le_rfl h0 hle le_rfl le_rfl le_rfl le_rfl
    linarith

/-- **An envelope through a conv INPUT-gradient.** `convFlatBack W = flatConv (reverseSwap W) 0`,
    so this is the forward conv leaf at zero bias with the channel roles swapped: the fan-in is
    the COTANGENT's channel count times the kernel window. No new arithmetic. -/
theorem Maps.convBack {ic oc h w kH kW : Nat} (M : FloatModel) (W : Kernel4 oc ic kH kW)
    {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < oc * h * w)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w')
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (oc * kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((oc * kH * kW : ℕ) : ℝ) * w' * Ā + 0) ≤ Ā')
    (hĒ' : g * (((oc * kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + 0)
            + ((oc * kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_convBack (h := h) (w := w) M W hw' hn hW).Maps Ā Ē Ā' Ē' :=
  Maps.flatConv (ic := oc) (oc := ic) (h := h) (w := w) M (IR.reverseSwap W) (fun _ => 0)
    hw' le_rfl hn (fun o c kh kw => hW c o (IR.kRev kh) (IR.kRev kw)) (fun _ => by simp)
    hg hĀ' hĒ'

/-- **An envelope through a dense INPUT-gradient.** `linBack W = dense Wᵀ 0`, so this is the
    forward dense leaf at zero bias — the fan-in is the OUTPUT dimension (`10` at r34's loss
    head, which is why a backward chain starts small and grows). -/
theorem Maps.linBack {m n : Nat} (M : FloatModel) (W : Mat m n) {w' : ℝ}
    (hw' : 0 ≤ w') (hn : 0 < n) (hW : ∀ i j, |W i j| ≤ w')
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (n + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * ((n : ℝ) * w' * Ā + 0) ≤ Ā')
    (hĒ' : g * ((n : ℝ) * w' * (Ā + Ē) + 0) + (n : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_linBack M W hw' hn hW).Maps Ā Ē Ā' Ē' :=
  Maps.dense M (Mat.transpose W) 0 hw' le_rfl hn (fun i j => hW j i) (fun j => by simp)
    hg hĀ' hĒ'

/-- **An envelope through a STRIDED conv input-gradient** — the zero-fill scatter, then the
    reversed-kernel conv at the doubled resolution. Two stages, one of them exact. -/
theorem Maps.flatConvStride2Back {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < oc * (2 * h) * (2 * w))
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w')
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (oc * kH * kW + 2) - 1 ≤ g)
    (hĀ' : (1 + g) * (((oc * kH * kW : ℕ) : ℝ) * w' * Ā + 0) ≤ Ā')
    (hĒ' : g * (((oc * kH * kW : ℕ) : ℝ) * w' * (Ā + Ē) + 0)
            + ((oc * kH * kW : ℕ) : ℝ) * w' * Ē ≤ Ē') :
    (floatBridgesTo_flatConvStride2Back (h := h) (w := w) M W hw' hn hW).Maps Ā Ē Ā' Ē' :=
  (Maps.decimateBack oc h w).comp hn
    (Maps.convBack (h := 2 * h) (w := 2 * w) M W hw' hn hW hg hĀ' hĒ')

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The two ResNet-34 block backwards, at real weights
-- ════════════════════════════════════════════════════════════════

/-- The float identity-block input-gradient — `r34IdBlockBack`'s deployed peer, every stage the
    float map its bridge names. ⭐ The residual-skip backward needs NO new combinator: the skip
    routes the cotangent to both branches and adds, so the block backward is itself a *forward*
    `Proofs.residual`, and `FloatBridgesTo.residual`'s rounded skip-add is the backward's too. -/
noncomputable def r34IdBlockBackF {c h w : Nat} (M : FloatModel) (W₁ W₂ : Kernel4 c c 3 3)
    (bnB1F bnB2F : Vec (c * h * w) → Vec (c * h * w))
    (m_out m_mid : Fin (c * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid] :
    Vec (c * h * w) → Vec (c * h * w) :=
  (fun v j => M.add
      ((M.flatConvF (h := h) (w := w) (IR.reverseSwap W₁) (fun _ => 0) ∘ bnB1F
          ∘ reluMaskBack m_mid
          ∘ M.flatConvF (h := h) (w := w) (IR.reverseSwap W₂) (fun _ => 0) ∘ bnB2F) v j) (v j))
    ∘ reluMaskBack m_out

/-- **The r34 identity-block input-gradient VJP float-bridges TO its float peer** — the
    `FloatBridgesTo` peer of `floatBridges_r34IdBlockBack`, with the float map NAMED. The two
    per-channel BN-backs are supplied as bridges (discharge with
    `floatBridgesTo_bnPerChannelBack`); everything else is concrete. -/
noncomputable def floatBridgesTo_r34IdBlockBack {c h w : Nat} (M : FloatModel)
    (W₁ W₂ : Kernel4 c c 3 3)
    {bnB1 bnB2 bnB1F bnB2F : Vec (c * h * w) → Vec (c * h * w)}
    (m_out m_mid : Fin (c * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid]
    {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < c * h * w)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w') (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w')
    (hbnB1 : FloatBridgesTo bnB1 bnB1F) (hbnB2 : FloatBridgesTo bnB2 bnB2F) :
    FloatBridgesTo (r34IdBlockBack W₁ W₂ bnB1 bnB2 m_out m_mid)
      (r34IdBlockBackF M W₁ W₂ bnB1F bnB2F m_out m_mid) :=
  (floatBridgesTo_reluMaskBack m_out).comp
    (FloatBridgesTo.residual M
      ((((hbnB2.comp (floatBridgesTo_convBack (h := h) (w := w) M W₂ hw' hn hW₂)).comp
          (floatBridgesTo_reluMaskBack m_mid)).comp hbnB1).comp
          (floatBridgesTo_convBack (h := h) (w := w) M W₁ hw' hn hW₁)))

/-- The float downsample-block input-gradient — `r34DownBlockBack`'s deployed peer. The
    two-branch fan-in is `FloatBridgesTo.biPathSum`, the forward downsample skip's combinator. -/
noncomputable def r34DownBlockBackF {ic oc h w kHp kWp : Nat} (M : FloatModel)
    (W₁ : Kernel4 oc ic 3 3) (W₂ : Kernel4 oc oc 3 3) (Wp : Kernel4 oc ic kHp kWp)
    (bnB1F bnB2F bnBpF : Vec (oc * h * w) → Vec (oc * h * w))
    (m_out m_mid : Fin (oc * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid] :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  (fun v j => M.add
      (((M.flatConvF (h := 2 * h) (w := 2 * w) (IR.reverseSwap Wp) (fun _ => 0)
            ∘ decimateBack oc h w) ∘ bnBpF) v j)
      (((M.flatConvF (h := 2 * h) (w := 2 * w) (IR.reverseSwap W₁) (fun _ => 0)
            ∘ decimateBack oc h w) ∘ bnB1F ∘ reluMaskBack m_mid
          ∘ M.flatConvF (h := h) (w := w) (IR.reverseSwap W₂) (fun _ => 0) ∘ bnB2F) v j))
    ∘ reluMaskBack m_out

/-- **The r34 downsample-block input-gradient VJP float-bridges TO its float peer** — the outer
    ReLU mask, then the two-branch fan-in of the projection backward and the body backward. The
    three per-channel BN-backs are supplied as bridges. -/
noncomputable def floatBridgesTo_r34DownBlockBack {ic oc h w kHp kWp : Nat} (M : FloatModel)
    (W₁ : Kernel4 oc ic 3 3) (W₂ : Kernel4 oc oc 3 3) (Wp : Kernel4 oc ic kHp kWp)
    {bnB1 bnB2 bnBp bnB1F bnB2F bnBpF : Vec (oc * h * w) → Vec (oc * h * w)}
    (m_out m_mid : Fin (oc * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid]
    {w₁ w₂ wp : ℝ} (hw₁ : 0 ≤ w₁) (hw₂ : 0 ≤ w₂) (hwp : 0 ≤ wp)
    (hW₁ : ∀ o c kh kw, |W₁ o c kh kw| ≤ w₁) (hW₂ : ∀ o c kh kw, |W₂ o c kh kw| ≤ w₂)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hoc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (hbnB1 : FloatBridgesTo bnB1 bnB1F) (hbnB2 : FloatBridgesTo bnB2 bnB2F)
    (hbnBp : FloatBridgesTo bnBp bnBpF) :
    FloatBridgesTo (r34DownBlockBack W₁ W₂ Wp bnB1 bnB2 bnBp m_out m_mid)
      (r34DownBlockBackF M W₁ W₂ Wp bnB1F bnB2F bnBpF m_out m_mid) :=
  (floatBridgesTo_reluMaskBack m_out).comp
    (FloatBridgesTo.biPathSum M
      (hbnBp.comp (floatBridgesTo_flatConvStride2Back (h := h) (w := w) M Wp hwp
        (by positivity) hWp))
      ((((hbnB2.comp (floatBridgesTo_convBack (h := h) (w := w) M W₂ hw₂ (by positivity)
            hW₂)).comp
          (floatBridgesTo_reluMaskBack m_mid)).comp hbnB1).comp
          (floatBridgesTo_flatConvStride2Back (h := h) (w := w) M W₁ hw₁ (by positivity) hW₁)))

namespace FloatBridgesTo

/-- ⭐ **An envelope through one r34 identity-block input-gradient.** Six numeric stages and the
    skip: `mask → bnB₂ → convBack W₂ → mask → bnB₁ → convBack W₁`, then `Maps.residual` against
    the block's own input cotangent. Twelve inequalities — the backward peer of
    `R34IdBlk.maps`'s ten, and the two ReLU masks contribute none because a select rounds
    nothing. -/
theorem Maps.r34IdBlockBack {c h w : Nat} (M : FloatModel) (W₁ W₂ : Kernel4 c c 3 3)
    {bnB1 bnB2 bnB1F bnB2F : Vec (c * h * w) → Vec (c * h * w)}
    (m_out m_mid : Fin (c * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid]
    {w' : ℝ} (hw' : 0 ≤ w') (hn : 0 < c * h * w)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w') (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w')
    (hbnB1 : FloatBridgesTo bnB1 bnB1F) (hbnB2 : FloatBridgesTo bnB2 bnB2F)
    {q g Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hg : (1 + M.u) ^ (c * 3 * 3 + 2) - 1 ≤ g)
    (mbn2 : hbnB2.Maps Ā Ē A1 E1)
    (c2A : (1 + g) * (((c * 3 * 3 : ℕ) : ℝ) * w' * A1 + 0) ≤ A2)
    (c2E : g * (((c * 3 * 3 : ℕ) : ℝ) * w' * (A1 + E1) + 0)
            + ((c * 3 * 3 : ℕ) : ℝ) * w' * E1 ≤ E2)
    (mbn1 : hbnB1.Maps A2 E2 A3 E3)
    (c1A : (1 + g) * (((c * 3 * 3 : ℕ) : ℝ) * w' * A3 + 0) ≤ A4)
    (c1E : g * (((c * 3 * 3 : ℕ) : ℝ) * w' * (A3 + E3) + 0)
            + ((c * 3 * 3 : ℕ) : ℝ) * w' * E3 ≤ E4)
    (rA : A4 + Ā + q * (A4 + Ā) ≤ Ā') (rE : q * (A4 + E4 + Ā + Ē) + (E4 + Ē) ≤ Ē') :
    (floatBridgesTo_r34IdBlockBack M W₁ W₂ m_out m_mid hw' hn hW₁ hW₂ hbnB1 hbnB2).Maps
      Ā Ē Ā' Ē' :=
  (Maps.reluMaskBack m_out).comp hn
    (Maps.residual M hn
      ((((mbn2.comp hn (Maps.convBack (h := h) (w := w) M W₂ hw' hn hW₂ hg c2A c2E)).comp hn
          (Maps.reluMaskBack m_mid)).comp hn mbn1).comp hn
          (Maps.convBack (h := h) (w := w) M W₁ hw' hn hW₁ hg c1A c1E)) hq rA rE)

set_option maxHeartbeats 1000000 in
/-- ⭐ **An envelope through one r34 downsample-block input-gradient.** The outer mask, then the
    two-branch fan-in: the projection backward (`bnBp → 1×1 strided convBack`) and the body
    backward (`bnB₂ → convBack W₂ → mask → bnB₁ → strided convBack W₁`), summed by
    `Maps.biPathSum`. Both branches read the SAME cotangent — the downsample skip is where a
    backward chain fans out, exactly as the forward's is where it fans in. -/
theorem Maps.r34DownBlockBack {ic oc h w kHp kWp : Nat} (M : FloatModel)
    (W₁ : Kernel4 oc ic 3 3) (W₂ : Kernel4 oc oc 3 3) (Wp : Kernel4 oc ic kHp kWp)
    {bnB1 bnB2 bnBp bnB1F bnB2F bnBpF : Vec (oc * h * w) → Vec (oc * h * w)}
    (m_out m_mid : Fin (oc * h * w) → Prop) [DecidablePred m_out] [DecidablePred m_mid]
    {w₁ w₂ wp : ℝ} (hw₁ : 0 ≤ w₁) (hw₂ : 0 ≤ w₂) (hwp : 0 ≤ wp)
    (hW₁ : ∀ o c kh kw, |W₁ o c kh kw| ≤ w₁) (hW₂ : ∀ o c kh kw, |W₂ o c kh kw| ≤ w₂)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hoc : 0 < oc) (hh : 0 < h) (hw : 0 < w) (hic : 0 < ic * (2 * h) * (2 * w))
    (hbnB1 : FloatBridgesTo bnB1 bnB1F) (hbnB2 : FloatBridgesTo bnB2 bnB2F)
    (hbnBp : FloatBridgesTo bnBp bnBpF)
    {q gp g1 g2 Ā Ē P1 F1 P2 F2 A1 E1 A2 E2 A3 E3 A4 E4 Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hgp : (1 + M.u) ^ (oc * kHp * kWp + 2) - 1 ≤ gp)
    (hg2 : (1 + M.u) ^ (oc * 3 * 3 + 2) - 1 ≤ g2)
    (hg1 : (1 + M.u) ^ (oc * 3 * 3 + 2) - 1 ≤ g1)
    -- projection branch
    (mbnp : hbnBp.Maps Ā Ē P1 F1)
    (cpA : (1 + gp) * (((oc * kHp * kWp : ℕ) : ℝ) * wp * P1 + 0) ≤ P2)
    (cpE : gp * (((oc * kHp * kWp : ℕ) : ℝ) * wp * (P1 + F1) + 0)
            + ((oc * kHp * kWp : ℕ) : ℝ) * wp * F1 ≤ F2)
    -- body branch
    (mbn2 : hbnB2.Maps Ā Ē A1 E1)
    (c2A : (1 + g2) * (((oc * 3 * 3 : ℕ) : ℝ) * w₂ * A1 + 0) ≤ A2)
    (c2E : g2 * (((oc * 3 * 3 : ℕ) : ℝ) * w₂ * (A1 + E1) + 0)
            + ((oc * 3 * 3 : ℕ) : ℝ) * w₂ * E1 ≤ E2)
    (mbn1 : hbnB1.Maps A2 E2 A3 E3)
    (c1A : (1 + g1) * (((oc * 3 * 3 : ℕ) : ℝ) * w₁ * A3 + 0) ≤ A4)
    (c1E : g1 * (((oc * 3 * 3 : ℕ) : ℝ) * w₁ * (A3 + E3) + 0)
            + ((oc * 3 * 3 : ℕ) : ℝ) * w₁ * E3 ≤ E4)
    (rA : P2 + A4 + q * (P2 + A4) ≤ Ā') (rE : q * (P2 + F2 + A4 + E4) + (F2 + E4) ≤ Ē') :
    (floatBridgesTo_r34DownBlockBack M W₁ W₂ Wp m_out m_mid hw₁ hw₂ hwp hW₁ hW₂ hWp
      hoc hh hw hbnB1 hbnB2 hbnBp).Maps Ā Ē Ā' Ē' :=
  (Maps.reluMaskBack m_out).comp (Nat.mul_pos (Nat.mul_pos hoc hh) hw)
    (Maps.biPathSum M hic
      (mbnp.comp (Nat.mul_pos (Nat.mul_pos hoc hh) hw)
        (Maps.flatConvStride2Back (h := h) (w := w) M Wp hwp (by positivity) hWp hgp cpA cpE))
      ((((mbn2.comp (Nat.mul_pos (Nat.mul_pos hoc hh) hw)
          (Maps.convBack (h := h) (w := w) M W₂ hw₂ (by positivity) hW₂ hg2 c2A c2E)).comp
          (Nat.mul_pos (Nat.mul_pos hoc hh) hw) (Maps.reluMaskBack m_mid)).comp
          (Nat.mul_pos (Nat.mul_pos hoc hh) hw) mbn1).comp
          (Nat.mul_pos (Nat.mul_pos hoc hh) hw)
          (Maps.flatConvStride2Back (h := h) (w := w) M W₁ hw₁ (by positivity) hW₁ hg1 c1A c1E))
      hq rA rE)

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § A worked ResNet-34 backward site, at the emitted numerals
-- ════════════════════════════════════════════════════════════════

set_option maxHeartbeats 1000000 in
/-- ⭐⭐ **The head of ResNet-34's input-gradient chain, closed at `r34_back_chain`'s numerals** —
    the loss cotangent (`|p − y| ≤ 1` for softmax cross-entropy) through the classifier's
    input-gradient, the GAP backward, and block `e1`'s second BatchNorm backward at `512 × 7 × 7`.
    Eight inequalities over four `Maps` leaves.

    In: `(1, 0)`. Out: `(8348, 2.317·10⁻²)` — and note the SHAPE of a backward chain: the dense
    input-gradient's fan-in is `10` (the class count) and the GAP backward divides by `49`, so
    the first two stages SHRINK, and the growth only starts at the first normalisation.

    ⭐ `Xh := 7` is `√(7·7)` — `bnXhat_abs_le_num` off `bnXhat_sq_le`, the standardisation bound.
    ⛔ Every other quantity in this example is a supplied device accuracy: `es` on the float
    inverse-stddev and `exh` on the float normalised activation, both `10⁻²` (§3.7's caveat —
    those are the two the forward's own training-mode fold cannot discharge).

    ⚠ This exercises the leaves against the generator's arithmetic. A `Maps` leaf nothing
    composes is a leaf nobody has checked composes. -/
example (M : FloatModel) (hMu : M.u ≤ u32) (Wd : Mat 512 10) (γ : Vec 512)
    (x : Vec (512 * 7 * 7)) (fs : Fin 512 → ℝ) (fxh : Fin 512 → Vec (7 * 7)) {ε : ℝ}
    (hε : 0 < ε)
    (hWd : ∀ i j, |Wd i j| ≤ 12 / 10) (hγ : ∀ c, |γ c| ≤ 21 / 10)
    (hs : ∀ c, |fs c - bnIstd (7 * 7) (Mat.unflatten (reassocFwd 512 7 7 x) c) ε| ≤ 1 / 100)
    (hSabs : ∀ c, |bnIstd (7 * 7) (Mat.unflatten (reassocFwd 512 7 7 x) c) ε| ≤ 317)
    (hfxh : ∀ c i,
      |fxh c i - bnXhat (7 * 7) ε (Mat.unflatten (reassocFwd 512 7 7 x) c) i| ≤ 1 / 100) :
    (((floatBridgesTo_linBack M Wd (by norm_num) (by norm_num) hWd).comp
        (floatBridgesTo_gapBack M 512 7 7 (by norm_num) (by norm_num) (by norm_num))).comp
      (floatBridgesTo_bnPerChannelBack M γ x fs fxh (by norm_num) (by norm_num) hγ hs hSabs
        (fun c i => bnXhat_abs_le_num (X := 7) hε _ (by norm_num) (by norm_num) i) hfxh)).Maps
      1 0 8348 (2317 / 10 ^ 2) := by
  refine FloatBridgesTo.Maps.comp (by norm_num)
    (FloatBridgesTo.Maps.comp (by norm_num)
    (FloatBridgesTo.Maps.linBack M Wd (by norm_num) (by norm_num) hWd
      (M.gamma_num (k := 10 + 2) (q := 7153 / 10 ^ 10) hMu (by norm_num [u32])
        (by norm_num [u32]))
      (Ā' := 1201 / 10 ^ 2) (Ē' := 8584 / 10 ^ 9) (by norm_num [u32]) (by norm_num [u32]))
    (FloatBridgesTo.Maps.gapBack M 512 7 7 (by norm_num) (by norm_num) (by norm_num)
      hMu (Ā' := 2452 / 10 ^ 4) (Ē' := 1898 / 10 ^ 10)
      (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])))
    (FloatBridgesTo.Maps.bnPerChannelBack M γ x fs fxh (by norm_num) (by norm_num) hγ hs hSabs
    (fun c i => bnXhat_abs_le_num (X := 7) hε _ (by norm_num) (by norm_num) i) hfxh
    (q := u32) (gn := 2981 / 10 ^ 9) hMu
    (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32])
      (by norm_num [u32]))
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [bnGradInputReMag, bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf,
                  bgE1, bgEXS, bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED,
                  FloatModel.mulErr, u32])
    (by norm_num [bnGradInputReMag, bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf,
                  bgE1, bgEXS, bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED,
                  FloatModel.mulErr, u32]))


set_option maxHeartbeats 2000000 in
/-- ⭐⭐ **ResNet-34's block `e1` input-gradient, end to end, at `r34_back_chain`'s numerals** —
    an identity basic block at `512 × 7 × 7`, six numeric stages and the skip, twelve
    inequalities plus the two BN sites' four, all `norm_num`.

    In: the GAP backward's `(2.452·10⁻⁴, 1.898·10⁻¹⁰)`. Out: `(8.702·10¹², 5.299·10¹⁰)` — one
    block costs ~10¹⁶ of cotangent window, and sixteen of them put the net at 10²⁸⁸.

    ⭐ Both BatchNorm sites are the real `floatBridgesTo_bnPerChannelBack`, with `Xh := 7` from
    `bnXhat_abs_le_num` (`√(7·7)`), NOT supplied envelopes — this is the exact shape the budget
    file plugs at all 33 sites. ⛔ What IS supplied is `es`/`exh` at `10⁻²`: the float
    inverse-stddev's and normalised activation's accuracies, which §3.7's caveat is about. -/
example (M : FloatModel) (hMu : M.u ≤ u32) (W₁ W₂ : Kernel4 512 512 3 3)
    (γ₁ γ₂ : Vec 512) (x₁ x₂ : Vec (512 * 7 * 7))
    (fs₁ fs₂ : Fin 512 → ℝ) (fxh₁ fxh₂ : Fin 512 → Vec (7 * 7)) {ε : ℝ} (hε : 0 < ε)
    (m_out m_mid : Fin (512 * 7 * 7) → Prop) [DecidablePred m_out] [DecidablePred m_mid]
    (hW₁ : ∀ o c kh kw, |W₁ o c kh kw| ≤ 12 / 10) (hW₂ : ∀ o c kh kw, |W₂ o c kh kw| ≤ 12 / 10)
    (hγ₁ : ∀ c, |γ₁ c| ≤ 21 / 10) (hγ₂ : ∀ c, |γ₂ c| ≤ 21 / 10)
    (hs₁ : ∀ c, |fs₁ c - bnIstd (7 * 7) (Mat.unflatten (reassocFwd 512 7 7 x₁) c) ε| ≤ 1 / 100)
    (hs₂ : ∀ c, |fs₂ c - bnIstd (7 * 7) (Mat.unflatten (reassocFwd 512 7 7 x₂) c) ε| ≤ 1 / 100)
    (hS₁ : ∀ c, |bnIstd (7 * 7) (Mat.unflatten (reassocFwd 512 7 7 x₁) c) ε| ≤ 317)
    (hS₂ : ∀ c, |bnIstd (7 * 7) (Mat.unflatten (reassocFwd 512 7 7 x₂) c) ε| ≤ 317)
    (hf₁ : ∀ c i,
      |fxh₁ c i - bnXhat (7 * 7) ε (Mat.unflatten (reassocFwd 512 7 7 x₁) c) i| ≤ 1 / 100)
    (hf₂ : ∀ c i,
      |fxh₂ c i - bnXhat (7 * 7) ε (Mat.unflatten (reassocFwd 512 7 7 x₂) c) i| ≤ 1 / 100) :
    (floatBridgesTo_r34IdBlockBack (h := 7) (w := 7) M W₁ W₂ m_out m_mid (by norm_num)
      (by norm_num) hW₁ hW₂
      (floatBridgesTo_bnPerChannelBack M γ₁ x₁ fs₁ fxh₁ (by norm_num) (by norm_num) hγ₁ hs₁ hS₁
        (fun c i => bnXhat_abs_le_num (X := 7) hε _ (by norm_num) (by norm_num) i) hf₁)
      (floatBridgesTo_bnPerChannelBack M γ₂ x₂ fs₂ fxh₂ (by norm_num) (by norm_num) hγ₂ hs₂ hS₂
        (fun c i => bnXhat_abs_le_num (X := 7) hε _ (by norm_num) (by norm_num) i)
        hf₂)).Maps
      (2452 / 10 ^ 4) (1898 / 10 ^ 10) (8702 * 10 ^ 12) (5299 * 10 ^ 10) := by
  exact FloatBridgesTo.Maps.r34IdBlockBack (h := 7) (w := 7) M W₁ W₂ m_out m_mid (by norm_num)
    (by norm_num) hW₁ hW₂ _ _
    (q := u32) (g := 2749 / 10 ^ 7)
    (A1 := 8348) (E1 := 2317 / 10 ^ 2)
    (A2 := 4618 * 10 ^ 4) (E2 := 1409 * 10 ^ 2)
    (A3 := 1573 * 10 ^ 9) (E3 := 9146 * 10 ^ 6)
    (A4 := 8701 * 10 ^ 12) (E4 := 5298 * 10 ^ 10)
    hMu
    (M.gamma_num (k := 512 * 3 * 3 + 2) (q := 2749 / 10 ^ 7) hMu (by norm_num [u32])
      (by norm_num [u32]))
    (FloatBridgesTo.Maps.bnPerChannelBack M γ₂ x₂ fs₂ fxh₂ (by norm_num) (by norm_num) hγ₂ hs₂
      hS₂ (fun c i => bnXhat_abs_le_num (X := 7) hε _ (by norm_num) (by norm_num) i) hf₂
      (q := u32) (gn := 2981 / 10 ^ 9) hMu
      (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32])
        (by norm_num [u32]))
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num [bnGradInputReMag, bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf,
                    bgE1, bgEXS, bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED,
                    FloatModel.mulErr, u32])
      (by norm_num [bnGradInputReMag, bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf,
                    bgE1, bgEXS, bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED,
                    FloatModel.mulErr, u32]))
    (by norm_num [u32]) (by norm_num [u32])
    (FloatBridgesTo.Maps.bnPerChannelBack M γ₁ x₁ fs₁ fxh₁ (by norm_num) (by norm_num) hγ₁ hs₁
      hS₁ (fun c i => bnXhat_abs_le_num (X := 7) hε _ (by norm_num) (by norm_num) i) hf₁
      (q := u32) (gn := 2981 / 10 ^ 9) hMu
      (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32])
        (by norm_num [u32]))
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num [bnGradInputReMag, bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf,
                    bgE1, bgEXS, bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED,
                    FloatModel.mulErr, u32])
      (by norm_num [bnGradInputReMag, bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf,
                    bgE1, bgEXS, bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED,
                    FloatModel.mulErr, u32]))
    (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32])

end Proofs
