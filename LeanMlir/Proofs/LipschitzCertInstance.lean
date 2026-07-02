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

/-- Row-wise Cauchy–Schwarz summed: `‖Mv‖² ≤ ‖M‖_F²·‖v‖²` at the raw-sum level. -/
theorem sum_sq_matvec_le {k n : ℕ} (M : Fin k → Fin n → ℝ) (y : Fin n → ℝ) :
    ∑ a, (∑ b, M a b * y b) ^ 2 ≤ (∑ a, ∑ b, M a b ^ 2) * (∑ b, y b ^ 2) := by
  calc ∑ a, (∑ b, M a b * y b) ^ 2
      ≤ ∑ a, ((∑ b, M a b ^ 2) * (∑ b, y b ^ 2)) :=
        Finset.sum_le_sum fun a _ => Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    _ = (∑ a, ∑ b, M a b ^ 2) * (∑ b, y b ^ 2) := (Finset.sum_mul ..).symm

/-- **Gram (Schatten-4) bound, proved.** If `G = W·Wᵀ` (supplied as data, verified
    entrywise) and `‖G‖_F² ≤ B⁴`, then the dense layer is `B`-Lipschitz in L2.
    Since `‖G‖_F = (Σᵢσᵢ⁴)^½`, this is `‖W‖₂ ≤ (Σσᵢ⁴)^¼` — strictly tighter than
    Frobenius `(Σσᵢ²)^½` whenever the spectrum has any spread. The Gram matrix is
    only `k×k` (output-side), so the kernel arithmetic stays small even for wide
    layers. -/
theorem denseE_lipschitzL2_gram {n k : ℕ} (W : Fin k → Fin n → ℝ)
    (G : Fin k → Fin k → ℝ) {B : ℝ} (hB : 0 ≤ B)
    (hG : ∀ a b, G a b = ∑ j, W a j * W b j)
    (hGF : ∑ a, ∑ b, G a b ^ 2 ≤ B ^ 4) :
    LipschitzL2 B (denseE W) := by
  intro u w
  set d : Fin n → ℝ := fun j => u j - w j with hdd
  set y : Fin k → ℝ := fun i => ∑ j, W i j * d j with hyy
  set z : Fin n → ℝ := fun j => ∑ i, W i j * y i with hzz
  set S : ℝ := ∑ i, y i ^ 2 with hS
  set Dq : ℝ := ∑ j, d j ^ 2 with hDq
  have hS0 : 0 ≤ S := Finset.sum_nonneg fun i _ => sq_nonneg _
  have hDq0 : 0 ≤ Dq := Finset.sum_nonneg fun j _ => sq_nonneg _
  -- S = ⟨d, Wᵀy⟩
  have hswap : S = ∑ j, d j * z j := by
    calc S = ∑ i, y i * ∑ j, W i j * d j := by
          exact Finset.sum_congr rfl fun i _ => by rw [pow_two]
      _ = ∑ i, ∑ j, y i * (W i j * d j) := by
          exact Finset.sum_congr rfl fun i _ => Finset.mul_sum ..
      _ = ∑ j, ∑ i, y i * (W i j * d j) := Finset.sum_comm
      _ = ∑ j, d j * z j := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [hzz, Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => by ring
  -- Σz² = ⟨y, Gy⟩ =: T
  have hTz : ∑ j, z j ^ 2 = ∑ a, y a * ∑ b, G a b * y b := by
    calc ∑ j, z j ^ 2
        = ∑ j, ∑ a, ∑ b, (W a j * y a) * (W b j * y b) := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [pow_two, hzz, Finset.sum_mul_sum]
      _ = ∑ a, ∑ j, ∑ b, (W a j * y a) * (W b j * y b) := Finset.sum_comm
      _ = ∑ a, ∑ b, ∑ j, (W a j * y a) * (W b j * y b) := by
          exact Finset.sum_congr rfl fun a _ => Finset.sum_comm
      _ = ∑ a, ∑ b, (y a * y b) * ∑ j, W a j * W b j := by
          refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun j _ => by ring
      _ = ∑ a, y a * ∑ b, G a b * y b := by
          refine Finset.sum_congr rfl fun a _ => ?_
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun b _ => by rw [hG]; ring
  have hT0 : 0 ≤ ∑ j, z j ^ 2 := Finset.sum_nonneg fun j _ => sq_nonneg _
  -- CS1: S² ≤ Dq·T
  have hCS1 : S ^ 2 ≤ Dq * ∑ j, z j ^ 2 := by
    rw [hswap]
    exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  -- CS2: T² ≤ S · (ΣG²·S) ≤ B⁴·S²
  have hCS2 : (∑ j, z j ^ 2) ^ 2 ≤ B ^ 4 * S ^ 2 := by
    have h1 : (∑ j, z j ^ 2) ^ 2 ≤ S * ∑ a, (∑ b, G a b * y b) ^ 2 := by
      rw [hTz]
      exact Finset.sum_mul_sq_le_sq_mul_sq _ _ _
    have h2 : ∑ a, (∑ b, G a b * y b) ^ 2 ≤ (∑ a, ∑ b, G a b ^ 2) * S :=
      sum_sq_matvec_le G y
    have h3 : (∑ a, ∑ b, G a b ^ 2) * S ≤ B ^ 4 * S :=
      mul_le_mul_of_nonneg_right hGF hS0
    calc (∑ j, z j ^ 2) ^ 2 ≤ S * ∑ a, (∑ b, G a b * y b) ^ 2 := h1
      _ ≤ S * (B ^ 4 * S) := by
          exact mul_le_mul_of_nonneg_left (h2.trans h3) hS0
      _ = B ^ 4 * S ^ 2 := by ring
  -- T ≤ B²·S  (both nonneg, compare squares)
  have hTle : (∑ j, z j ^ 2) ≤ B ^ 2 * S := by
    have hb2 : 0 ≤ B ^ 2 * S := mul_nonneg (sq_nonneg _) hS0
    nlinarith [hCS2, hT0, hb2]
  -- S ≤ B²·Dq  (divide S² ≤ Dq·B²·S by S, case S = 0)
  have hSle : S ≤ B ^ 2 * Dq := by
    rcases eq_or_lt_of_le hS0 with h0 | hpos
    · rw [← h0]; exact mul_nonneg (sq_nonneg _) hDq0
    · have : S ^ 2 ≤ Dq * (B ^ 2 * S) :=
        hCS1.trans (mul_le_mul_of_nonneg_left hTle hDq0)
      nlinarith [this, hpos]
  -- back to norms
  have hcoord : ∀ i, (denseE W u - denseE W w) i = y i := by
    intro i
    show (∑ j, W i j * u j) - (∑ j, W i j * w j) = _
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by
      show W i j * u j - W i j * w j = W i j * (u j - w j); ring
  have hnormsq : ‖denseE W u - denseE W w‖ ^ 2 ≤ (B * ‖u - w‖) ^ 2 := by
    rw [euclid_norm_sq, mul_pow, euclid_norm_sq]
    calc ∑ i, ((denseE W u - denseE W w) i) ^ 2
        = S := Finset.sum_congr rfl fun i _ => by rw [hcoord]
      _ ≤ B ^ 2 * Dq := hSle
      _ = B ^ 2 * ∑ j, ((u - w) j) ^ 2 := rfl
  calc ‖denseE W u - denseE W w‖
      = Real.sqrt (‖denseE W u - denseE W w‖ ^ 2) :=
        (Real.sqrt_sq (norm_nonneg _)).symm
    _ ≤ Real.sqrt ((B * ‖u - w‖) ^ 2) := Real.sqrt_le_sqrt hnormsq
    _ = B * ‖u - w‖ := Real.sqrt_sq (mul_nonneg hB (norm_nonneg _))

/-- **Certified lower bound on any L2 Lipschitz constant** (the power-iteration
    direction): if `‖f u − f w‖ ≥ ℓ·‖u − w‖` at one concrete pair (verified as a
    squared-sum inequality in-kernel), then every valid `L` satisfies `ℓ ≤ L`.
    With `u` the (rationalized) power-iteration singular vector and `w = 0`,
    this certifies how close a proven upper bound sits to the true `‖W‖₂`. -/
theorem lipschitzL2_lower_euclid {n k : ℕ} {L ℓ : ℝ}
    {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin k)}
    (hf : LipschitzL2 L f) (hℓ : 0 ≤ ℓ) (u w : EuclideanSpace ℝ (Fin n))
    (hpos : 0 < ∑ j, ((u - w) j) ^ 2)
    (hray : ℓ ^ 2 * (∑ j, ((u - w) j) ^ 2) ≤ ∑ i, ((f u - f w) i) ^ 2) :
    ℓ ≤ L := by
  have hnw : 0 < ‖u - w‖ := by
    have h2 : 0 < ‖u - w‖ ^ 2 := by rw [euclid_norm_sq]; exact hpos
    rcases (norm_nonneg (u - w)).eq_or_lt with h | h
    · exfalso; rw [← h] at h2; simp at h2
    · exact h
  have h1 : ℓ * ‖u - w‖ ≤ ‖f u - f w‖ := by
    have e : (ℓ * ‖u - w‖) ^ 2 ≤ ‖f u - f w‖ ^ 2 := by
      rw [mul_pow, euclid_norm_sq, euclid_norm_sq]
      exact hray
    calc ℓ * ‖u - w‖
        = Real.sqrt ((ℓ * ‖u - w‖) ^ 2) :=
          (Real.sqrt_sq (mul_nonneg hℓ (norm_nonneg _))).symm
      _ ≤ Real.sqrt (‖f u - f w‖ ^ 2) := Real.sqrt_le_sqrt e
      _ = ‖f u - f w‖ := Real.sqrt_sq (norm_nonneg _)
  exact le_of_mul_le_mul_right (h1.trans (hf u w)) hnw


-- ════════════════════════════════════════════════════════════
-- § Power-iteration certificate: certified two-sided spectral sandwich
--
-- Upper: the Gram (Schatten-4) bound ‖W‖₂ ≤ ‖WWᵀ‖_F^(1/2) = (Σσᵢ⁴)^(1/4) —
--   B₁=9.2 / B₂=9.661 vs Frobenius 14.555/14.576 ⇒ L drops
--   212→88.9 and the certified radius grows 0.0463→0.1106 (2.4×).
-- Lower: the power-iteration singular vector, rationalized, certifies that
--   ANY valid Lipschitz constant is ≥ ℓ₁=7.452 / ℓ₂=7.7 — so the Gram
--   bound provably sits within 24%/26% of the per-layer optimum.
-- ════════════════════════════════════════════════════════════

/-- Exact Gram matrix `G1t = W1t·W1tᵀ` (8×8, denominators 128² = 16384). -/
noncomputable def G1t : Fin 8 → Fin 8 → ℝ :=
  ![![((581457 : ℝ)/16384), ((-62530 : ℝ)/16384), ((130497 : ℝ)/16384), ((-69516 : ℝ)/16384), ((-80622 : ℝ)/16384), ((29867 : ℝ)/16384), ((-70132 : ℝ)/16384), ((-71816 : ℝ)/16384)],
    ![((-62530 : ℝ)/16384), ((352025 : ℝ)/16384), ((51933 : ℝ)/16384), ((-10049 : ℝ)/16384), ((-6301 : ℝ)/16384), ((-26056 : ℝ)/16384), ((-43168 : ℝ)/16384), ((-9355 : ℝ)/16384)],
    ![((130497 : ℝ)/16384), ((51933 : ℝ)/16384), ((615605 : ℝ)/16384), ((-15531 : ℝ)/16384), ((-164389 : ℝ)/16384), ((-108692 : ℝ)/16384), ((-184108 : ℝ)/16384), ((-14418 : ℝ)/16384)],
    ![((-69516 : ℝ)/16384), ((-10049 : ℝ)/16384), ((-15531 : ℝ)/16384), ((435472 : ℝ)/16384), ((41253 : ℝ)/16384), ((-126995 : ℝ)/16384), ((54877 : ℝ)/16384), ((48774 : ℝ)/16384)],
    ![((-80622 : ℝ)/16384), ((-6301 : ℝ)/16384), ((-164389 : ℝ)/16384), ((41253 : ℝ)/16384), ((375274 : ℝ)/16384), ((64070 : ℝ)/16384), ((26286 : ℝ)/16384), ((46006 : ℝ)/16384)],
    ![((29867 : ℝ)/16384), ((-26056 : ℝ)/16384), ((-108692 : ℝ)/16384), ((-126995 : ℝ)/16384), ((64070 : ℝ)/16384), ((392316 : ℝ)/16384), ((5746 : ℝ)/16384), ((124768 : ℝ)/16384)],
    ![((-70132 : ℝ)/16384), ((-43168 : ℝ)/16384), ((-184108 : ℝ)/16384), ((54877 : ℝ)/16384), ((26286 : ℝ)/16384), ((5746 : ℝ)/16384), ((324766 : ℝ)/16384), ((25088 : ℝ)/16384)],
    ![((-71816 : ℝ)/16384), ((-9355 : ℝ)/16384), ((-14418 : ℝ)/16384), ((48774 : ℝ)/16384), ((46006 : ℝ)/16384), ((124768 : ℝ)/16384), ((25088 : ℝ)/16384), ((393745 : ℝ)/16384)]]

/-- Exact Gram matrix `G2t = W2t·W2tᵀ` (10×10). -/
noncomputable def G2t : Fin 10 → Fin 10 → ℝ :=
  ![![((233489 : ℝ)/16384), ((-212663 : ℝ)/16384), ((87250 : ℝ)/16384), ((26452 : ℝ)/16384), ((-163212 : ℝ)/16384), ((67065 : ℝ)/16384), ((15920 : ℝ)/16384), ((-53271 : ℝ)/16384), ((-13918 : ℝ)/16384), ((-110593 : ℝ)/16384)],
    ![((-212663 : ℝ)/16384), ((483518 : ℝ)/16384), ((-77712 : ℝ)/16384), ((-18693 : ℝ)/16384), ((-38979 : ℝ)/16384), ((95714 : ℝ)/16384), ((51511 : ℝ)/16384), ((-53983 : ℝ)/16384), ((59832 : ℝ)/16384), ((-99078 : ℝ)/16384)],
    ![((87250 : ℝ)/16384), ((-77712 : ℝ)/16384), ((388111 : ℝ)/16384), ((131057 : ℝ)/16384), ((-169849 : ℝ)/16384), ((-196872 : ℝ)/16384), ((-42411 : ℝ)/16384), ((-160231 : ℝ)/16384), ((40571 : ℝ)/16384), ((-82469 : ℝ)/16384)],
    ![((26452 : ℝ)/16384), ((-18693 : ℝ)/16384), ((131057 : ℝ)/16384), ((237366 : ℝ)/16384), ((-153105 : ℝ)/16384), ((16132 : ℝ)/16384), ((-108454 : ℝ)/16384), ((-35493 : ℝ)/16384), ((-65940 : ℝ)/16384), ((-104884 : ℝ)/16384)],
    ![((-163212 : ℝ)/16384), ((-38979 : ℝ)/16384), ((-169849 : ℝ)/16384), ((-153105 : ℝ)/16384), ((369176 : ℝ)/16384), ((-83883 : ℝ)/16384), ((82737 : ℝ)/16384), ((19323 : ℝ)/16384), ((-6599 : ℝ)/16384), ((199281 : ℝ)/16384)],
    ![((67065 : ℝ)/16384), ((95714 : ℝ)/16384), ((-196872 : ℝ)/16384), ((16132 : ℝ)/16384), ((-83883 : ℝ)/16384), ((328598 : ℝ)/16384), ((66286 : ℝ)/16384), ((-75346 : ℝ)/16384), ((-7363 : ℝ)/16384), ((-171506 : ℝ)/16384)],
    ![((15920 : ℝ)/16384), ((51511 : ℝ)/16384), ((-42411 : ℝ)/16384), ((-108454 : ℝ)/16384), ((82737 : ℝ)/16384), ((66286 : ℝ)/16384), ((417130 : ℝ)/16384), ((-98132 : ℝ)/16384), ((-103689 : ℝ)/16384), ((-215533 : ℝ)/16384)],
    ![((-53271 : ℝ)/16384), ((-53983 : ℝ)/16384), ((-160231 : ℝ)/16384), ((-35493 : ℝ)/16384), ((19323 : ℝ)/16384), ((-75346 : ℝ)/16384), ((-98132 : ℝ)/16384), ((395126 : ℝ)/16384), ((-92487 : ℝ)/16384), ((165676 : ℝ)/16384)],
    ![((-13918 : ℝ)/16384), ((59832 : ℝ)/16384), ((40571 : ℝ)/16384), ((-65940 : ℝ)/16384), ((-6599 : ℝ)/16384), ((-7363 : ℝ)/16384), ((-103689 : ℝ)/16384), ((-92487 : ℝ)/16384), ((215157 : ℝ)/16384), ((143222 : ℝ)/16384)],
    ![((-110593 : ℝ)/16384), ((-99078 : ℝ)/16384), ((-82469 : ℝ)/16384), ((-104884 : ℝ)/16384), ((199281 : ℝ)/16384), ((-171506 : ℝ)/16384), ((-215533 : ℝ)/16384), ((165676 : ℝ)/16384), ((143222 : ℝ)/16384), ((412917 : ℝ)/16384)]]

set_option maxHeartbeats 3200000 in
theorem G1t_eq : ∀ a b, G1t a b = ∑ j, W1t a j * W1t b j := by
  intro a b
  fin_cases a <;> fin_cases b <;>
    · simp [G1t, W1t, Fin.sum_univ_succ]
      norm_num

set_option maxHeartbeats 3200000 in
theorem G2t_eq : ∀ a b, G2t a b = ∑ j, W2t a j * W2t b j := by
  intro a b
  fin_cases a <;> fin_cases b <;>
    · simp [G2t, W2t, Fin.sum_univ_succ]
      norm_num

/-- Schatten-4 Lipschitz bound for the hidden layer: B₁ = 46/5 ≈ (Σσ⁴)^(1/4). -/
theorem W1t_lip_gram : LipschitzL2 ((46 : ℝ)/5) (denseE W1t) := by
  refine denseE_lipschitzL2_gram W1t G1t (by norm_num) G1t_eq ?_
  simp [G1t, Fin.sum_univ_succ]
  norm_num

theorem W2t_lip_gram : LipschitzL2 ((9661 : ℝ)/1000) (denseE W2t) := by
  refine denseE_lipschitzL2_gram W2t G2t (by norm_num) G2t_eq ?_
  simp [G2t, Fin.sum_univ_succ]
  norm_num

/-- The tightened product certificate: L = B₂·(1·B₁) = 222203/2500. -/
theorem mlpT_lip_gram : LipschitzL2 ((222203 : ℝ)/2500) mlpT := by
  have h := W2t_lip_gram.comp (reluE_lipschitzL2.comp W1t_lip_gram (by norm_num)) (by norm_num)
  have e : ((9661 : ℝ)/1000) * (1 * ((46 : ℝ)/5)) = ((222203 : ℝ)/2500) := by norm_num
  rw [e] at h; exact h

theorem trained_radius_gram_pos : 0 < ((6953 : ℝ)/500) / (Real.sqrt 2 * ((222203 : ℝ)/2500)) :=
  div_pos (by norm_num)
    (mul_pos (Real.sqrt_pos.mpr (by norm_num)) (by norm_num))

/-- **The tightened trained certificate.** Same trained net, same margin, the
    Gram bound in place of Frobenius: every `‖δ‖ < 13.906/(√2·88.88) ≈ 0.1106`
    (2.4× the Frobenius radius) leaves the prediction fixed. -/
theorem trained_demo_certified_gram (δ : EuclideanSpace ℝ (Fin 49))
    (hδ : ‖δ‖ < ((6953 : ℝ)/500) / (Real.sqrt 2 * ((222203 : ℝ)/2500))) :
    ∀ j, j ≠ 2 → mlpT (xt + δ) j < mlpT (xt + δ) 2 :=
  lipschitz_margin_certified_radius mlpT_lip_gram (by norm_num) xt_margin hδ

/-- Rationalized power-iteration vector for `W1t` (top right-singular direction ×1000). -/
noncomputable def v1t : EuclideanSpace ℝ (Fin 49) :=
  WithLp.toLp 2 ![(4 : ℝ), (36 : ℝ), (91 : ℝ), (154 : ℝ), (128 : ℝ), (98 : ℝ), (-34 : ℝ), (15 : ℝ), (-41 : ℝ), (-109 : ℝ), (-247 : ℝ), (-229 : ℝ), (-105 : ℝ), (-133 : ℝ), (-10 : ℝ), (-124 : ℝ), (-154 : ℝ), (-269 : ℝ), (-61 : ℝ), (-24 : ℝ), (-138 : ℝ), (-84 : ℝ), (-2 : ℝ), (93 : ℝ), (128 : ℝ), (-39 : ℝ), (10 : ℝ), (138 : ℝ), (-8 : ℝ), (147 : ℝ), (210 : ℝ), (312 : ℝ), (65 : ℝ), (-8 : ℝ), (126 : ℝ), (-34 : ℝ), (-88 : ℝ), (7 : ℝ), (-22 : ℝ), (62 : ℝ), (82 : ℝ), (-10 : ℝ), (-62 : ℝ), (-139 : ℝ), (-343 : ℝ), (-387 : ℝ), (-296 : ℝ), (-106 : ℝ), (51 : ℝ)]

/-- Rationalized power-iteration vector for `W2t`. -/
noncomputable def v2t : EuclideanSpace ℝ (Fin 8) :=
  WithLp.toLp 2 ![(712 : ℝ), (160 : ℝ), (-442 : ℝ), (-286 : ℝ), (-38 : ℝ), (-140 : ℝ), (-248 : ℝ), (-328 : ℝ)]

/-- **Certified lower bound**: ANY `L` with `LipschitzL2 L (denseE W1t)` is ≥ 1863/250.
    With `W1t_lip_gram : LipschitzL2 9.2 …`, the true `‖W1t‖₂` is sandwiched in
    `[7.452, 9.2]` — the Gram bound is provably ≤ 1.235× optimal. -/
theorem W1t_lip_lower : ∀ L : ℝ, LipschitzL2 L (denseE W1t) → ((1863 : ℝ)/250) ≤ L := by
  intro L hL
  refine lipschitzL2_lower_euclid hL (by norm_num) v1t 0 ?_ ?_
  · simp [v1t, Fin.sum_univ_succ]
    norm_num
  · have hc : ∀ i : Fin 8, (denseE W1t v1t - denseE W1t 0) i = ∑ j, W1t i j * v1t j := by
      intro i
      show (∑ j, W1t i j * v1t j) - (∑ j, W1t i j * (0 : EuclideanSpace ℝ (Fin 49)) j) = _
      simp
    simp only [sub_zero, hc]
    simp [W1t, v1t, Fin.sum_univ_succ]
    norm_num

theorem W2t_lip_lower : ∀ L : ℝ, LipschitzL2 L (denseE W2t) → ((77 : ℝ)/10) ≤ L := by
  intro L hL
  refine lipschitzL2_lower_euclid hL (by norm_num) v2t 0 ?_ ?_
  · simp [v2t, Fin.sum_univ_succ]
    norm_num
  · have hc : ∀ i : Fin 10, (denseE W2t v2t - denseE W2t 0) i = ∑ j, W2t i j * v2t j := by
      intro i
      show (∑ j, W2t i j * v2t j) - (∑ j, W2t i j * (0 : EuclideanSpace ℝ (Fin 8)) j) = _
      simp
    simp only [sub_zero, hc]
    simp [W2t, v2t, Fin.sum_univ_succ]
    norm_num

end LipschitzCertDemo
end Proofs
