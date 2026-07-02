import LeanMlir.Proofs.LipschitzCert

/-! # A concrete instantiation of the Lipschitz-margin certified radius

Closes the "certificate machinery, never instantiated" gap: a fixed-weight
network whose Lipschitz constant is PROVED in Lean (Frobenius bound — no
power iteration, no hypothesis), whose margin at a concrete input is
computed in-kernel, and whose certified radius is provably positive.

Two instances:
* `linear_demo_certified` — a 2×2 linear classifier, L = 5 (Frobenius),
  margin 3 at x = e₀, certified radius 3/(√2·5) > 0.
* `mlp_demo_certified` — a dense → ReLU → dense MLP, L = the per-layer
  product 3·(1·2) = 6 via `LipschitzL2.comp` (the exact product bound the
  PGD demos estimate numerically), margin 2, radius 2/(√2·6) > 0.
-/

namespace Proofs
namespace LipschitzCertDemo

open scoped BigOperators

/-- A bias-free dense (linear) layer on Euclidean space:
    `(denseE W x)ᵢ = Σⱼ Wᵢⱼ xⱼ`. -/
noncomputable def denseE {n k : ℕ} (W : Fin k → Fin n → ℝ) :
    EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k) :=
  fun x => WithLp.toLp 2 (fun i => ∑ j, W i j * x j)

@[simp] theorem denseE_apply {n k : ℕ} (W : Fin k → Fin n → ℝ)
    (x : EuclideanSpace ℝ (Fin n)) (i : Fin k) :
    denseE W x i = ∑ j, W i j * x j := rfl

/-- **Frobenius bound, proved.** If the entrywise square sum of `W` is at
    most `C²`, the dense layer is `C`-Lipschitz in L2. This is the certified
    replacement for the power-iteration estimate `specNormW`: `‖W‖₂ ≤ ‖W‖_F`,
    so any rational `C ≥ ‖W‖_F` is a sound Lipschitz constant. -/
theorem denseE_lipschitzL2 {n k : ℕ} (W : Fin k → Fin n → ℝ) {C : ℝ}
    (hC : 0 ≤ C) (hW : ∑ i, ∑ j, W i j ^ 2 ≤ C ^ 2) :
    LipschitzL2 C (denseE W) := by
  intro u w
  have hcoord : ∀ i : Fin k,
      (denseE W u - denseE W w) i = ∑ j, W i j * ((u - w) j) := by
    intro i
    show (∑ j, W i j * u j) - (∑ j, W i j * w j) = _
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by
      show W i j * u j - W i j * w j = W i j * (u j - w j); ring
  have hsq : ‖denseE W u - denseE W w‖ ^ 2 ≤ (C * ‖u - w‖) ^ 2 := by
    rw [euclid_norm_sq]
    calc ∑ i, ((denseE W u - denseE W w) i) ^ 2
        = ∑ i, (∑ j, W i j * ((u - w) j)) ^ 2 := by
          exact Finset.sum_congr rfl fun i _ => by rw [hcoord]
      _ ≤ ∑ i, ((∑ j, W i j ^ 2) * (∑ j, ((u - w) j) ^ 2)) :=
          Finset.sum_le_sum fun i _ =>
            Finset.sum_mul_sq_le_sq_mul_sq _ _ _
      _ = (∑ i, ∑ j, W i j ^ 2) * (∑ j, ((u - w) j) ^ 2) :=
          (Finset.sum_mul ..).symm
      _ ≤ C ^ 2 * (∑ j, ((u - w) j) ^ 2) :=
          mul_le_mul_of_nonneg_right hW
            (Finset.sum_nonneg fun j _ => sq_nonneg _)
      _ = (C * ‖u - w‖) ^ 2 := by rw [mul_pow, euclid_norm_sq]
  have h0 : 0 ≤ C * ‖u - w‖ := mul_nonneg hC (norm_nonneg _)
  calc ‖denseE W u - denseE W w‖
      = Real.sqrt (‖denseE W u - denseE W w‖ ^ 2) :=
        (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ Real.sqrt ((C * ‖u - w‖) ^ 2) := Real.sqrt_le_sqrt hsq
    _ = C * ‖u - w‖ := Real.sqrt_sq h0

/-- Coordinatewise ReLU on Euclidean space. -/
noncomputable def reluE {n : ℕ} :
    EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin n) :=
  fun x => WithLp.toLp 2 (fun i => max (x i) 0)

@[simp] theorem reluE_apply {n : ℕ} (x : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    reluE x i = max (x i) 0 := rfl

/-- **ReLU is 1-Lipschitz in L2** — coordinatewise `|max(a,0) − max(b,0)| ≤ |a − b|`
    summed. The activation contributes factor 1 to the product certificate. -/
theorem reluE_lipschitzL2 {n : ℕ} : LipschitzL2 1 (reluE (n := n)) := by
  intro u w
  have hsq : ‖reluE u - reluE w‖ ^ 2 ≤ ‖u - w‖ ^ 2 := by
    rw [euclid_norm_sq, euclid_norm_sq]
    refine Finset.sum_le_sum fun i _ => ?_
    have habs : |max (u i) 0 - max (w i) 0| ≤ |u i - w i| :=
      abs_max_sub_max_le_abs (u i) (w i) 0
    have h1 : (reluE u - reluE w) i = max (u i) 0 - max (w i) 0 := rfl
    have h2 : (u - w) i = u i - w i := rfl
    rw [h1, h2, ← sq_abs (max (u i) 0 - max (w i) 0), ← sq_abs (u i - w i)]
    exact pow_le_pow_left₀ (abs_nonneg _) habs 2
  have := Real.sqrt_le_sqrt hsq
  rwa [Real.sqrt_sq (norm_nonneg _), Real.sqrt_sq (norm_nonneg _),
       one_mul] at *

-- ════════════════════════════════════════════════════════════════
-- § Instance 1: linear classifier, everything concrete
-- ════════════════════════════════════════════════════════════════

/-- Fixed 2×2 weight matrix: logits `(3x₀, 4x₁)`. Frobenius norm exactly 5. -/
def Wlin : Fin 2 → Fin 2 → ℝ := ![![3, 0], ![0, 4]]

/-- The concrete input `x = e₀`. -/
noncomputable def xlin : EuclideanSpace ℝ (Fin 2) := WithLp.toLp 2 ![1, 0]

theorem Wlin_lip : LipschitzL2 5 (denseE Wlin) := by
  refine denseE_lipschitzL2 Wlin (by norm_num) ?_
  simp [Wlin, Fin.sum_univ_two]
  norm_num

/-- Margin computed in-kernel: logits at `xlin` are `(3, 0)`, so class 0
    leads by `3`. -/
theorem xlin_margin : ∀ j : Fin 2, j ≠ 0 →
    (3 : ℝ) ≤ denseE Wlin xlin 0 - denseE Wlin xlin j := by
  intro j hj
  fin_cases j
  · exact absurd rfl hj
  · simp [Wlin, xlin, Fin.sum_univ_two]

/-- The certified radius is strictly positive — the certificate is
    non-vacuous. Numerically `3/(√2·5) ≈ 0.424`. -/
theorem linear_radius_pos : 0 < (3 : ℝ) / (Real.sqrt 2 * 5) :=
  div_pos (by norm_num)
    (mul_pos (Real.sqrt_pos.mpr (by norm_num)) (by norm_num))

/-- **The instantiated Tsuzuku certificate (linear).** Every L2 perturbation
    of norm `< 3/(√2·5)` of the concrete input leaves class 0 the strict
    argmax of the concrete network `x ↦ (3x₀, 4x₁)`. -/
theorem linear_demo_certified (δ : EuclideanSpace ℝ (Fin 2))
    (hδ : ‖δ‖ < 3 / (Real.sqrt 2 * 5)) :
    ∀ j, j ≠ 0 → denseE Wlin (xlin + δ) j < denseE Wlin (xlin + δ) 0 :=
  lipschitz_margin_certified_radius Wlin_lip (by norm_num) xlin_margin hδ

-- ════════════════════════════════════════════════════════════════
-- § Instance 2: dense → ReLU → dense MLP, product certificate
-- ════════════════════════════════════════════════════════════════

/-- Hidden layer: identity mixing (Frobenius √2 ≤ 2). -/
def Wmlp : Fin 2 → Fin 2 → ℝ := ![![1, 0], ![0, 1]]

/-- Output layer: `(2h₀, h₁)` (Frobenius √5 ≤ 3). -/
def Vmlp : Fin 2 → Fin 2 → ℝ := ![![2, 0], ![0, 1]]

/-- The concrete 2-layer MLP `dense ∘ relu ∘ dense`. -/
noncomputable def mlp : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) :=
  denseE Vmlp ∘ reluE ∘ denseE Wmlp

/-- The per-layer **product** certificate `L = 3·(1·2) = 6`, assembled by
    `LipschitzL2.comp` from proved Frobenius bounds — the certified analogue
    of the `∏ᵢ‖Wᵢ‖₂` product the PGD demos compute numerically. -/
theorem mlp_lip : LipschitzL2 6 mlp := by
  have hW : LipschitzL2 2 (denseE Wmlp) := by
    refine denseE_lipschitzL2 Wmlp (by norm_num) ?_
    simp [Wmlp, Fin.sum_univ_two]; norm_num
  have hV : LipschitzL2 3 (denseE Vmlp) := by
    refine denseE_lipschitzL2 Vmlp (by norm_num) ?_
    simp [Vmlp, Fin.sum_univ_two]; norm_num
  have hchain : LipschitzL2 (3 * (1 * 2)) (denseE Vmlp ∘ (reluE ∘ denseE Wmlp)) :=
    hV.comp (reluE_lipschitzL2.comp hW (by norm_num)) (by norm_num)
  have : (3 : ℝ) * (1 * 2) = 6 := by norm_num
  rw [this] at hchain
  exact hchain

/-- Margin computed in-kernel through the whole MLP: forward of `e₀` is
    `(1,0) ↦ relu (1,0) = (1,0) ↦ (2,0)`, so class 0 leads by 2. -/
theorem mlp_margin : ∀ j : Fin 2, j ≠ 0 →
    (2 : ℝ) ≤ mlp xlin 0 - mlp xlin j := by
  intro j hj
  fin_cases j
  · exact absurd rfl hj
  · simp [mlp, Function.comp, Vmlp, Wmlp, xlin, Fin.sum_univ_two]

theorem mlp_radius_pos : 0 < (2 : ℝ) / (Real.sqrt 2 * 6) :=
  div_pos (by norm_num)
    (mul_pos (Real.sqrt_pos.mpr (by norm_num)) (by norm_num))

/-- **The instantiated Tsuzuku certificate (MLP).** Every L2 perturbation of
    norm `< 2/(√2·6) ≈ 0.236` leaves class 0 the strict argmax of the
    concrete dense→ReLU→dense network — kink and all: the certificate is
    architecture-agnostic, no smoothness hypotheses needed. -/
theorem mlp_demo_certified (δ : EuclideanSpace ℝ (Fin 2))
    (hδ : ‖δ‖ < 2 / (Real.sqrt 2 * 6)) :
    ∀ j, j ≠ 0 → mlp (xlin + δ) j < mlp (xlin + δ) 0 :=
  lipschitz_margin_certified_radius mlp_lip (by norm_num) mlp_margin hδ



-- ════════════════════════════════════════════════════════════
-- § Instance 3: TRAINED weights (MNIST, 4×4-pooled to 7×7), rationalized
--
-- 49→8→10 bias-free ReLU MLP trained on pooled MNIST (test acc ≈0.898);
-- weights rounded to /128 rationals (quantized test acc ≈0.898). Test image
-- #1895 (digit 2); every quantity below is exact rational arithmetic
-- checked in-kernel. Generated by scripts (see planning); weights are DATA here.
-- ════════════════════════════════════════════════════════════

/-- Trained hidden-layer weights (8×49), entries `k/128`. -/
noncomputable def W1t : Fin 8 → Fin 49 → ℝ :=
  ![![((3 : ℝ)/128), ((-18 : ℝ)/128), ((-108 : ℝ)/128), ((-264 : ℝ)/128), ((-161 : ℝ)/128), ((-38 : ℝ)/128), ((30 : ℝ)/128), ((21 : ℝ)/128), ((26 : ℝ)/128), ((64 : ℝ)/128), ((78 : ℝ)/128), ((46 : ℝ)/128), ((88 : ℝ)/128), ((136 : ℝ)/128), ((-58 : ℝ)/128), ((29 : ℝ)/128), ((174 : ℝ)/128), ((7 : ℝ)/128), ((81 : ℝ)/128), ((174 : ℝ)/128), ((73 : ℝ)/128), ((29 : ℝ)/128), ((205 : ℝ)/128), ((186 : ℝ)/128), ((85 : ℝ)/128), ((189 : ℝ)/128), ((57 : ℝ)/128), ((-132 : ℝ)/128), ((-75 : ℝ)/128), ((-118 : ℝ)/128), ((-16 : ℝ)/128), ((-60 : ℝ)/128), ((37 : ℝ)/128), ((-100 : ℝ)/128), ((-166 : ℝ)/128), ((-30 : ℝ)/128), ((-124 : ℝ)/128), ((-75 : ℝ)/128), ((44 : ℝ)/128), ((-62 : ℝ)/128), ((-86 : ℝ)/128), ((-27 : ℝ)/128), ((33 : ℝ)/128), ((63 : ℝ)/128), ((219 : ℝ)/128), ((150 : ℝ)/128), ((137 : ℝ)/128), ((141 : ℝ)/128), ((42 : ℝ)/128)],
    ![((34 : ℝ)/128), ((9 : ℝ)/128), ((3 : ℝ)/128), ((91 : ℝ)/128), ((18 : ℝ)/128), ((-53 : ℝ)/128), ((9 : ℝ)/128), ((19 : ℝ)/128), ((51 : ℝ)/128), ((101 : ℝ)/128), ((111 : ℝ)/128), ((-15 : ℝ)/128), ((-114 : ℝ)/128), ((-84 : ℝ)/128), ((28 : ℝ)/128), ((103 : ℝ)/128), ((4 : ℝ)/128), ((33 : ℝ)/128), ((70 : ℝ)/128), ((-3 : ℝ)/128), ((-183 : ℝ)/128), ((36 : ℝ)/128), ((-168 : ℝ)/128), ((-252 : ℝ)/128), ((219 : ℝ)/128), ((125 : ℝ)/128), ((19 : ℝ)/128), ((-24 : ℝ)/128), ((79 : ℝ)/128), ((-153 : ℝ)/128), ((-27 : ℝ)/128), ((137 : ℝ)/128), ((-66 : ℝ)/128), ((25 : ℝ)/128), ((8 : ℝ)/128), ((-7 : ℝ)/128), ((98 : ℝ)/128), ((52 : ℝ)/128), ((18 : ℝ)/128), ((71 : ℝ)/128), ((-57 : ℝ)/128), ((3 : ℝ)/128), ((41 : ℝ)/128), ((52 : ℝ)/128), ((35 : ℝ)/128), ((2 : ℝ)/128), ((56 : ℝ)/128), ((31 : ℝ)/128), ((-14 : ℝ)/128)],
    ![((-35 : ℝ)/128), ((-39 : ℝ)/128), ((16 : ℝ)/128), ((53 : ℝ)/128), ((-8 : ℝ)/128), ((-52 : ℝ)/128), ((23 : ℝ)/128), ((-24 : ℝ)/128), ((62 : ℝ)/128), ((173 : ℝ)/128), ((180 : ℝ)/128), ((169 : ℝ)/128), ((56 : ℝ)/128), ((75 : ℝ)/128), ((63 : ℝ)/128), ((118 : ℝ)/128), ((89 : ℝ)/128), ((187 : ℝ)/128), ((61 : ℝ)/128), ((59 : ℝ)/128), ((129 : ℝ)/128), ((64 : ℝ)/128), ((-100 : ℝ)/128), ((-74 : ℝ)/128), ((-169 : ℝ)/128), ((-57 : ℝ)/128), ((-26 : ℝ)/128), ((-36 : ℝ)/128), ((17 : ℝ)/128), ((-30 : ℝ)/128), ((-139 : ℝ)/128), ((-269 : ℝ)/128), ((27 : ℝ)/128), ((95 : ℝ)/128), ((-10 : ℝ)/128), ((65 : ℝ)/128), ((180 : ℝ)/128), ((141 : ℝ)/128), ((132 : ℝ)/128), ((25 : ℝ)/128), ((-9 : ℝ)/128), ((32 : ℝ)/128), ((12 : ℝ)/128), ((83 : ℝ)/128), ((245 : ℝ)/128), ((300 : ℝ)/128), ((163 : ℝ)/128), ((-15 : ℝ)/128), ((-47 : ℝ)/128)],
    ![((-3 : ℝ)/128), ((0 : ℝ)/128), ((78 : ℝ)/128), ((143 : ℝ)/128), ((196 : ℝ)/128), ((97 : ℝ)/128), ((-9 : ℝ)/128), ((-9 : ℝ)/128), ((-85 : ℝ)/128), ((-27 : ℝ)/128), ((1 : ℝ)/128), ((-2 : ℝ)/128), ((61 : ℝ)/128), ((133 : ℝ)/128), ((-46 : ℝ)/128), ((11 : ℝ)/128), ((96 : ℝ)/128), ((-13 : ℝ)/128), ((-269 : ℝ)/128), ((-122 : ℝ)/128), ((138 : ℝ)/128), ((16 : ℝ)/128), ((39 : ℝ)/128), ((35 : ℝ)/128), ((184 : ℝ)/128), ((2 : ℝ)/128), ((24 : ℝ)/128), ((-2 : ℝ)/128), ((-37 : ℝ)/128), ((-80 : ℝ)/128), ((67 : ℝ)/128), ((1 : ℝ)/128), ((-10 : ℝ)/128), ((42 : ℝ)/128), ((-108 : ℝ)/128), ((-9 : ℝ)/128), ((13 : ℝ)/128), ((171 : ℝ)/128), ((268 : ℝ)/128), ((160 : ℝ)/128), ((35 : ℝ)/128), ((-25 : ℝ)/128), ((-12 : ℝ)/128), ((-100 : ℝ)/128), ((-83 : ℝ)/128), ((-35 : ℝ)/128), ((-19 : ℝ)/128), ((-37 : ℝ)/128), ((41 : ℝ)/128)],
    ![((-33 : ℝ)/128), ((-5 : ℝ)/128), ((69 : ℝ)/128), ((90 : ℝ)/128), ((83 : ℝ)/128), ((40 : ℝ)/128), ((-38 : ℝ)/128), ((18 : ℝ)/128), ((49 : ℝ)/128), ((53 : ℝ)/128), ((-15 : ℝ)/128), ((-62 : ℝ)/128), ((-122 : ℝ)/128), ((-114 : ℝ)/128), ((14 : ℝ)/128), ((26 : ℝ)/128), ((84 : ℝ)/128), ((-70 : ℝ)/128), ((9 : ℝ)/128), ((131 : ℝ)/128), ((-194 : ℝ)/128), ((-16 : ℝ)/128), ((85 : ℝ)/128), ((92 : ℝ)/128), ((-136 : ℝ)/128), ((47 : ℝ)/128), ((195 : ℝ)/128), ((109 : ℝ)/128), ((-55 : ℝ)/128), ((18 : ℝ)/128), ((228 : ℝ)/128), ((114 : ℝ)/128), ((90 : ℝ)/128), ((-37 : ℝ)/128), ((52 : ℝ)/128), ((-45 : ℝ)/128), ((-80 : ℝ)/128), ((89 : ℝ)/128), ((141 : ℝ)/128), ((-5 : ℝ)/128), ((-7 : ℝ)/128), ((21 : ℝ)/128), ((-81 : ℝ)/128), ((-34 : ℝ)/128), ((-109 : ℝ)/128), ((-115 : ℝ)/128), ((-64 : ℝ)/128), ((-57 : ℝ)/128), ((47 : ℝ)/128)],
    ![((5 : ℝ)/128), ((-2 : ℝ)/128), ((86 : ℝ)/128), ((-20 : ℝ)/128), ((-62 : ℝ)/128), ((14 : ℝ)/128), ((-1 : ℝ)/128), ((29 : ℝ)/128), ((21 : ℝ)/128), ((141 : ℝ)/128), ((-247 : ℝ)/128), ((-133 : ℝ)/128), ((-22 : ℝ)/128), ((2 : ℝ)/128), ((74 : ℝ)/128), ((22 : ℝ)/128), ((14 : ℝ)/128), ((-114 : ℝ)/128), ((299 : ℝ)/128), ((259 : ℝ)/128), ((107 : ℝ)/128), ((-10 : ℝ)/128), ((10 : ℝ)/128), ((10 : ℝ)/128), ((36 : ℝ)/128), ((-108 : ℝ)/128), ((-84 : ℝ)/128), ((-7 : ℝ)/128), ((-43 : ℝ)/128), ((19 : ℝ)/128), ((-12 : ℝ)/128), ((122 : ℝ)/128), ((36 : ℝ)/128), ((-132 : ℝ)/128), ((-2 : ℝ)/128), ((7 : ℝ)/128), ((-17 : ℝ)/128), ((25 : ℝ)/128), ((-60 : ℝ)/128), ((-84 : ℝ)/128), ((-94 : ℝ)/128), ((-42 : ℝ)/128), ((-25 : ℝ)/128), ((-15 : ℝ)/128), ((-96 : ℝ)/128), ((-51 : ℝ)/128), ((-35 : ℝ)/128), ((-16 : ℝ)/128), ((18 : ℝ)/128)],
    ![((-25 : ℝ)/128), ((14 : ℝ)/128), ((2 : ℝ)/128), ((55 : ℝ)/128), ((41 : ℝ)/128), ((35 : ℝ)/128), ((31 : ℝ)/128), ((22 : ℝ)/128), ((65 : ℝ)/128), ((1 : ℝ)/128), ((-75 : ℝ)/128), ((-88 : ℝ)/128), ((51 : ℝ)/128), ((10 : ℝ)/128), ((7 : ℝ)/128), ((-113 : ℝ)/128), ((-175 : ℝ)/128), ((-156 : ℝ)/128), ((-21 : ℝ)/128), ((-80 : ℝ)/128), ((-50 : ℝ)/128), ((-21 : ℝ)/128), ((3 : ℝ)/128), ((203 : ℝ)/128), ((159 : ℝ)/128), ((101 : ℝ)/128), ((-44 : ℝ)/128), ((-20 : ℝ)/128), ((18 : ℝ)/128), ((102 : ℝ)/128), ((-55 : ℝ)/128), ((46 : ℝ)/128), ((174 : ℝ)/128), ((140 : ℝ)/128), ((1 : ℝ)/128), ((38 : ℝ)/128), ((-17 : ℝ)/128), ((-41 : ℝ)/128), ((36 : ℝ)/128), ((73 : ℝ)/128), ((9 : ℝ)/128), ((-4 : ℝ)/128), ((12 : ℝ)/128), ((45 : ℝ)/128), ((44 : ℝ)/128), ((-133 : ℝ)/128), ((-194 : ℝ)/128), ((-54 : ℝ)/128), ((38 : ℝ)/128)],
    ![((7 : ℝ)/128), ((-14 : ℝ)/128), ((-14 : ℝ)/128), ((6 : ℝ)/128), ((-75 : ℝ)/128), ((-15 : ℝ)/128), ((15 : ℝ)/128), ((-45 : ℝ)/128), ((-64 : ℝ)/128), ((52 : ℝ)/128), ((20 : ℝ)/128), ((-40 : ℝ)/128), ((-22 : ℝ)/128), ((14 : ℝ)/128), ((-32 : ℝ)/128), ((16 : ℝ)/128), ((37 : ℝ)/128), ((-145 : ℝ)/128), ((55 : ℝ)/128), ((254 : ℝ)/128), ((146 : ℝ)/128), ((-45 : ℝ)/128), ((-116 : ℝ)/128), ((50 : ℝ)/128), ((126 : ℝ)/128), ((-57 : ℝ)/128), ((-117 : ℝ)/128), ((72 : ℝ)/128), ((-3 : ℝ)/128), ((132 : ℝ)/128), ((147 : ℝ)/128), ((83 : ℝ)/128), ((19 : ℝ)/128), ((-1 : ℝ)/128), ((173 : ℝ)/128), ((6 : ℝ)/128), ((115 : ℝ)/128), ((145 : ℝ)/128), ((39 : ℝ)/128), ((129 : ℝ)/128), ((219 : ℝ)/128), ((24 : ℝ)/128), ((-13 : ℝ)/128), ((-160 : ℝ)/128), ((-13 : ℝ)/128), ((18 : ℝ)/128), ((-60 : ℝ)/128), ((3 : ℝ)/128), ((-10 : ℝ)/128)]]

/-- Trained output-layer weights (10×8), entries `k/128`. -/
noncomputable def W2t : Fin 10 → Fin 8 → ℝ :=
  ![![((-93 : ℝ)/128), ((-292 : ℝ)/128), ((295 : ℝ)/128), ((-35 : ℝ)/128), ((189 : ℝ)/128), ((-55 : ℝ)/128), ((-58 : ℝ)/128), ((96 : ℝ)/128)],
    ![((-244 : ℝ)/128), ((329 : ℝ)/128), ((-196 : ℝ)/128), ((249 : ℝ)/128), ((-381 : ℝ)/128), ((215 : ℝ)/128), ((-153 : ℝ)/128), ((23 : ℝ)/128)],
    ![((-297 : ℝ)/128), ((152 : ℝ)/128), ((86 : ℝ)/128), ((-128 : ℝ)/128), ((175 : ℝ)/128), ((-344 : ℝ)/128), ((116 : ℝ)/128), ((301 : ℝ)/128)],
    ![((-89 : ℝ)/128), ((143 : ℝ)/128), ((308 : ℝ)/128), ((-87 : ℝ)/128), ((-133 : ℝ)/128), ((-61 : ℝ)/128), ((288 : ℝ)/128), ((47 : ℝ)/128)],
    ![((308 : ℝ)/128), ((-79 : ℝ)/128), ((-471 : ℝ)/128), ((27 : ℝ)/128), ((28 : ℝ)/128), ((2 : ℝ)/128), ((152 : ℝ)/128), ((-147 : ℝ)/128)],
    ![((27 : ℝ)/128), ((-229 : ℝ)/128), ((223 : ℝ)/128), ((276 : ℝ)/128), ((-212 : ℝ)/128), ((321 : ℝ)/128), ((-13 : ℝ)/128), ((37 : ℝ)/128)],
    ![((-236 : ℝ)/128), ((-125 : ℝ)/128), ((-179 : ℝ)/128), ((427 : ℝ)/128), ((283 : ℝ)/128), ((105 : ℝ)/128), ((113 : ℝ)/128), ((-166 : ℝ)/128)],
    ![((221 : ℝ)/128), ((227 : ℝ)/128), ((120 : ℝ)/128), ((-209 : ℝ)/128), ((129 : ℝ)/128), ((128 : ℝ)/128), ((-207 : ℝ)/128), ((-401 : ℝ)/128)],
    ![((140 : ℝ)/128), ((23 : ℝ)/128), ((-61 : ℝ)/128), ((109 : ℝ)/128), ((-119 : ℝ)/128), ((-189 : ℝ)/128), ((-238 : ℝ)/128), ((270 : ℝ)/128)],
    ![((465 : ℝ)/128), ((86 : ℝ)/128), ((-184 : ℝ)/128), ((-170 : ℝ)/128), ((-65 : ℝ)/128), ((-279 : ℝ)/128), ((-193 : ℝ)/128), ((-85 : ℝ)/128)]]

/-- MNIST test image #1895 (digit 2), 4×4-average-pooled: exact pixel
    sums over 4080 (= 255·16). -/
noncomputable def xt : EuclideanSpace ℝ (Fin 49) :=
  WithLp.toLp 2 ![((0 : ℝ)/4080), ((0 : ℝ)/4080), ((707 : ℝ)/4080), ((289 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((1793 : ℝ)/4080), ((3597 : ℝ)/4080), ((1576 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((561 : ℝ)/4080), ((2965 : ℝ)/4080), ((1545 : ℝ)/4080), ((0 : ℝ)/4080), ((9 : ℝ)/4080), ((789 : ℝ)/4080), ((1408 : ℝ)/4080), ((1286 : ℝ)/4080), ((1798 : ℝ)/4080), ((2913 : ℝ)/4080), ((0 : ℝ)/4080), ((785 : ℝ)/4080), ((4048 : ℝ)/4080), ((4048 : ℝ)/4080), ((4048 : ℝ)/4080), ((4048 : ℝ)/4080), ((2946 : ℝ)/4080), ((0 : ℝ)/4080), ((38 : ℝ)/4080), ((816 : ℝ)/4080), ((1174 : ℝ)/4080), ((1167 : ℝ)/4080), ((1311 : ℝ)/4080), ((1748 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080), ((0 : ℝ)/4080)]

/-- The trained MLP: dense → ReLU → dense. -/
noncomputable def mlpT : EuclideanSpace ℝ (Fin 49) → EuclideanSpace ℝ (Fin 10) :=
  denseE W2t ∘ reluE ∘ denseE W1t

/-- Exact hidden pre-activations of `xt` (denominator 128·4080). -/
noncomputable def hpreVals : Fin 8 → ℝ :=
  ![((507504 : ℝ)/522240), ((730889 : ℝ)/522240), ((164933 : ℝ)/522240), ((245813 : ℝ)/522240), ((2734742 : ℝ)/522240), ((-21615 : ℝ)/522240), ((1483431 : ℝ)/522240), ((2664061 : ℝ)/522240)]

theorem hpre_eval : ∀ k : Fin 8, denseE W1t xt k = hpreVals k := by
  intro k
  fin_cases k <;>
    · simp [denseE_apply, W1t, xt, hpreVals, Fin.sum_univ_succ]
      norm_num

/-- Frobenius² of `W1t` is ≤ C₁² for C₁ = 2911/200 ≈ ‖W1t‖_F. -/
theorem W1t_lip : LipschitzL2 ((2911 : ℝ)/200) (denseE W1t) := by
  refine denseE_lipschitzL2 W1t (by norm_num) ?_
  simp [W1t, Fin.sum_univ_succ]
  norm_num

theorem W2t_lip : LipschitzL2 ((1822 : ℝ)/125) (denseE W2t) := by
  refine denseE_lipschitzL2 W2t (by norm_num) ?_
  simp [W2t, Fin.sum_univ_succ]
  norm_num

/-- Product certificate for the trained net: L = C₂·(1·C₁) = 2651921/12500. -/
theorem mlpT_lip : LipschitzL2 ((2651921 : ℝ)/12500) mlpT := by
  have h := W2t_lip.comp (reluE_lipschitzL2.comp W1t_lip (by norm_num)) (by norm_num)
  have e : ((1822 : ℝ)/125) * (1 * ((2911 : ℝ)/200)) = ((2651921 : ℝ)/12500) := by norm_num
  rw [e] at h; exact h

/-- In-kernel margin: class 2 leads every other class at `xt` by ≥ 6953/500. -/
theorem xt_margin : ∀ j : Fin 10, j ≠ 2 →
    ((6953 : ℝ)/500) ≤ mlpT xt 2 - mlpT xt j := by
  have hout : ∀ jj : Fin 10, mlpT xt jj = ∑ k : Fin 8, W2t jj k * max (hpreVals k) 0 := by
    intro jj
    show denseE W2t (reluE (denseE W1t xt)) jj = _
    rw [denseE_apply]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [reluE_apply, hpre_eval k]
  intro j hj
  fin_cases j <;>
    first
    | exact absurd rfl hj
    | · rw [hout, hout]
        simp [W2t, hpreVals, Fin.sum_univ_succ, max_def]
        norm_num

theorem trained_radius_pos : 0 < ((6953 : ℝ)/500) / (Real.sqrt 2 * ((2651921 : ℝ)/12500)) :=
  div_pos (by norm_num)
    (mul_pos (Real.sqrt_pos.mpr (by norm_num)) (by norm_num))

/-- **The Tsuzuku certificate at TRAINED weights.** Every L2 perturbation of
    the pooled MNIST digit-2 with `‖δ‖ < 13.906/(√2·212.154) ≈ 0.0463`
    leaves class 2 the strict argmax of the trained, rationalized network. -/
theorem trained_demo_certified (δ : EuclideanSpace ℝ (Fin 49))
    (hδ : ‖δ‖ < ((6953 : ℝ)/500) / (Real.sqrt 2 * ((2651921 : ℝ)/12500))) :
    ∀ j, j ≠ 2 → mlpT (xt + δ) j < mlpT (xt + δ) 2 :=
  lipschitz_margin_certified_radius mlpT_lip (by norm_num) xt_margin hδ

end LipschitzCertDemo
end Proofs
