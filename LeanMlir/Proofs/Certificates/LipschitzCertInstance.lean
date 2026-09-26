import LeanMlir.Proofs.Certificates.DenseEuclid
import LeanMlir.Proofs.Foundation.GramQ

/-! # A concrete instantiation of the Lipschitz-margin certified radius

**REDUCED CERTIFICATE MODEL** — this file's concrete net is the 4×4-pooled 49-dim
MNIST family (width-8 hidden, /128–/256 rational weights), NOT the canonical
784→512→512→10 `mlpVerified`; chosen so every margin/norm/SOS check is exact rational
arithmetic in-kernel. Canonical surface: [`Proofs/MlpCanonical.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Nets/Small/MlpCanonical.lean).

Fixed-weight networks whose Lipschitz constant is proved in Lean (no power iteration, no
hypothesis), whose margin at a concrete input is computed in-kernel, and whose certified radius
is provably positive.

Three instances:
* `linear_demo_certified` — a 2×2 linear classifier, L = 5 (Frobenius),
  margin 3 at x = e₀, certified radius 3/(√2·5) > 0.
* `mlp_demo_certified` — a 2 → 2 → 2 dense → ReLU → dense MLP, L = the per-layer
  product 3·(1·2) = 6 via `LipschitzL2.comp` (the exact product bound the
  PGD demos estimate numerically), margin 2, radius 2/(√2·6) > 0.
* `trained_demo_certified`, `trained_demo_certified_gram`, `trained_demo_certified_gram2` —
  the trained 49→8→10 MLP `mlpT` at the pooled MNIST test image `xt`, with Frobenius
  (`mlpT_lip`), Schatten-4 (`mlpT_lip_gram`) and Schatten-8 (`mlpT_lip_gram2`) product
  constants. The certified lower bounds `W1t_lip_lower` / `W2t_lip_lower` bound each layer's
  best Lipschitz constant from below. `mlpT`, its weights and `mlpT_logit_continuous` are what
  `LipschitzCertScorecard.lean`, `SmoothingNetSemantics.lean` and `SmoothingNetWitness.lean`
  build on.
-/

namespace Proofs
namespace LipschitzCertDemo

open scoped BigOperators
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
def W1tQ : Fin 8 → Fin 49 → ℚ :=
  ![![((3 : ℚ)/128), ((-18 : ℚ)/128), ((-108 : ℚ)/128), ((-264 : ℚ)/128), ((-161 : ℚ)/128), ((-38 : ℚ)/128), ((30 : ℚ)/128), ((21 : ℚ)/128), ((26 : ℚ)/128), ((64 : ℚ)/128), ((78 : ℚ)/128), ((46 : ℚ)/128), ((88 : ℚ)/128), ((136 : ℚ)/128), ((-58 : ℚ)/128), ((29 : ℚ)/128), ((174 : ℚ)/128), ((7 : ℚ)/128), ((81 : ℚ)/128), ((174 : ℚ)/128), ((73 : ℚ)/128), ((29 : ℚ)/128), ((205 : ℚ)/128), ((186 : ℚ)/128), ((85 : ℚ)/128), ((189 : ℚ)/128), ((57 : ℚ)/128), ((-132 : ℚ)/128), ((-75 : ℚ)/128), ((-118 : ℚ)/128), ((-16 : ℚ)/128), ((-60 : ℚ)/128), ((37 : ℚ)/128), ((-100 : ℚ)/128), ((-166 : ℚ)/128), ((-30 : ℚ)/128), ((-124 : ℚ)/128), ((-75 : ℚ)/128), ((44 : ℚ)/128), ((-62 : ℚ)/128), ((-86 : ℚ)/128), ((-27 : ℚ)/128), ((33 : ℚ)/128), ((63 : ℚ)/128), ((219 : ℚ)/128), ((150 : ℚ)/128), ((137 : ℚ)/128), ((141 : ℚ)/128), ((42 : ℚ)/128)],
    ![((34 : ℚ)/128), ((9 : ℚ)/128), ((3 : ℚ)/128), ((91 : ℚ)/128), ((18 : ℚ)/128), ((-53 : ℚ)/128), ((9 : ℚ)/128), ((19 : ℚ)/128), ((51 : ℚ)/128), ((101 : ℚ)/128), ((111 : ℚ)/128), ((-15 : ℚ)/128), ((-114 : ℚ)/128), ((-84 : ℚ)/128), ((28 : ℚ)/128), ((103 : ℚ)/128), ((4 : ℚ)/128), ((33 : ℚ)/128), ((70 : ℚ)/128), ((-3 : ℚ)/128), ((-183 : ℚ)/128), ((36 : ℚ)/128), ((-168 : ℚ)/128), ((-252 : ℚ)/128), ((219 : ℚ)/128), ((125 : ℚ)/128), ((19 : ℚ)/128), ((-24 : ℚ)/128), ((79 : ℚ)/128), ((-153 : ℚ)/128), ((-27 : ℚ)/128), ((137 : ℚ)/128), ((-66 : ℚ)/128), ((25 : ℚ)/128), ((8 : ℚ)/128), ((-7 : ℚ)/128), ((98 : ℚ)/128), ((52 : ℚ)/128), ((18 : ℚ)/128), ((71 : ℚ)/128), ((-57 : ℚ)/128), ((3 : ℚ)/128), ((41 : ℚ)/128), ((52 : ℚ)/128), ((35 : ℚ)/128), ((2 : ℚ)/128), ((56 : ℚ)/128), ((31 : ℚ)/128), ((-14 : ℚ)/128)],
    ![((-35 : ℚ)/128), ((-39 : ℚ)/128), ((16 : ℚ)/128), ((53 : ℚ)/128), ((-8 : ℚ)/128), ((-52 : ℚ)/128), ((23 : ℚ)/128), ((-24 : ℚ)/128), ((62 : ℚ)/128), ((173 : ℚ)/128), ((180 : ℚ)/128), ((169 : ℚ)/128), ((56 : ℚ)/128), ((75 : ℚ)/128), ((63 : ℚ)/128), ((118 : ℚ)/128), ((89 : ℚ)/128), ((187 : ℚ)/128), ((61 : ℚ)/128), ((59 : ℚ)/128), ((129 : ℚ)/128), ((64 : ℚ)/128), ((-100 : ℚ)/128), ((-74 : ℚ)/128), ((-169 : ℚ)/128), ((-57 : ℚ)/128), ((-26 : ℚ)/128), ((-36 : ℚ)/128), ((17 : ℚ)/128), ((-30 : ℚ)/128), ((-139 : ℚ)/128), ((-269 : ℚ)/128), ((27 : ℚ)/128), ((95 : ℚ)/128), ((-10 : ℚ)/128), ((65 : ℚ)/128), ((180 : ℚ)/128), ((141 : ℚ)/128), ((132 : ℚ)/128), ((25 : ℚ)/128), ((-9 : ℚ)/128), ((32 : ℚ)/128), ((12 : ℚ)/128), ((83 : ℚ)/128), ((245 : ℚ)/128), ((300 : ℚ)/128), ((163 : ℚ)/128), ((-15 : ℚ)/128), ((-47 : ℚ)/128)],
    ![((-3 : ℚ)/128), ((0 : ℚ)/128), ((78 : ℚ)/128), ((143 : ℚ)/128), ((196 : ℚ)/128), ((97 : ℚ)/128), ((-9 : ℚ)/128), ((-9 : ℚ)/128), ((-85 : ℚ)/128), ((-27 : ℚ)/128), ((1 : ℚ)/128), ((-2 : ℚ)/128), ((61 : ℚ)/128), ((133 : ℚ)/128), ((-46 : ℚ)/128), ((11 : ℚ)/128), ((96 : ℚ)/128), ((-13 : ℚ)/128), ((-269 : ℚ)/128), ((-122 : ℚ)/128), ((138 : ℚ)/128), ((16 : ℚ)/128), ((39 : ℚ)/128), ((35 : ℚ)/128), ((184 : ℚ)/128), ((2 : ℚ)/128), ((24 : ℚ)/128), ((-2 : ℚ)/128), ((-37 : ℚ)/128), ((-80 : ℚ)/128), ((67 : ℚ)/128), ((1 : ℚ)/128), ((-10 : ℚ)/128), ((42 : ℚ)/128), ((-108 : ℚ)/128), ((-9 : ℚ)/128), ((13 : ℚ)/128), ((171 : ℚ)/128), ((268 : ℚ)/128), ((160 : ℚ)/128), ((35 : ℚ)/128), ((-25 : ℚ)/128), ((-12 : ℚ)/128), ((-100 : ℚ)/128), ((-83 : ℚ)/128), ((-35 : ℚ)/128), ((-19 : ℚ)/128), ((-37 : ℚ)/128), ((41 : ℚ)/128)],
    ![((-33 : ℚ)/128), ((-5 : ℚ)/128), ((69 : ℚ)/128), ((90 : ℚ)/128), ((83 : ℚ)/128), ((40 : ℚ)/128), ((-38 : ℚ)/128), ((18 : ℚ)/128), ((49 : ℚ)/128), ((53 : ℚ)/128), ((-15 : ℚ)/128), ((-62 : ℚ)/128), ((-122 : ℚ)/128), ((-114 : ℚ)/128), ((14 : ℚ)/128), ((26 : ℚ)/128), ((84 : ℚ)/128), ((-70 : ℚ)/128), ((9 : ℚ)/128), ((131 : ℚ)/128), ((-194 : ℚ)/128), ((-16 : ℚ)/128), ((85 : ℚ)/128), ((92 : ℚ)/128), ((-136 : ℚ)/128), ((47 : ℚ)/128), ((195 : ℚ)/128), ((109 : ℚ)/128), ((-55 : ℚ)/128), ((18 : ℚ)/128), ((228 : ℚ)/128), ((114 : ℚ)/128), ((90 : ℚ)/128), ((-37 : ℚ)/128), ((52 : ℚ)/128), ((-45 : ℚ)/128), ((-80 : ℚ)/128), ((89 : ℚ)/128), ((141 : ℚ)/128), ((-5 : ℚ)/128), ((-7 : ℚ)/128), ((21 : ℚ)/128), ((-81 : ℚ)/128), ((-34 : ℚ)/128), ((-109 : ℚ)/128), ((-115 : ℚ)/128), ((-64 : ℚ)/128), ((-57 : ℚ)/128), ((47 : ℚ)/128)],
    ![((5 : ℚ)/128), ((-2 : ℚ)/128), ((86 : ℚ)/128), ((-20 : ℚ)/128), ((-62 : ℚ)/128), ((14 : ℚ)/128), ((-1 : ℚ)/128), ((29 : ℚ)/128), ((21 : ℚ)/128), ((141 : ℚ)/128), ((-247 : ℚ)/128), ((-133 : ℚ)/128), ((-22 : ℚ)/128), ((2 : ℚ)/128), ((74 : ℚ)/128), ((22 : ℚ)/128), ((14 : ℚ)/128), ((-114 : ℚ)/128), ((299 : ℚ)/128), ((259 : ℚ)/128), ((107 : ℚ)/128), ((-10 : ℚ)/128), ((10 : ℚ)/128), ((10 : ℚ)/128), ((36 : ℚ)/128), ((-108 : ℚ)/128), ((-84 : ℚ)/128), ((-7 : ℚ)/128), ((-43 : ℚ)/128), ((19 : ℚ)/128), ((-12 : ℚ)/128), ((122 : ℚ)/128), ((36 : ℚ)/128), ((-132 : ℚ)/128), ((-2 : ℚ)/128), ((7 : ℚ)/128), ((-17 : ℚ)/128), ((25 : ℚ)/128), ((-60 : ℚ)/128), ((-84 : ℚ)/128), ((-94 : ℚ)/128), ((-42 : ℚ)/128), ((-25 : ℚ)/128), ((-15 : ℚ)/128), ((-96 : ℚ)/128), ((-51 : ℚ)/128), ((-35 : ℚ)/128), ((-16 : ℚ)/128), ((18 : ℚ)/128)],
    ![((-25 : ℚ)/128), ((14 : ℚ)/128), ((2 : ℚ)/128), ((55 : ℚ)/128), ((41 : ℚ)/128), ((35 : ℚ)/128), ((31 : ℚ)/128), ((22 : ℚ)/128), ((65 : ℚ)/128), ((1 : ℚ)/128), ((-75 : ℚ)/128), ((-88 : ℚ)/128), ((51 : ℚ)/128), ((10 : ℚ)/128), ((7 : ℚ)/128), ((-113 : ℚ)/128), ((-175 : ℚ)/128), ((-156 : ℚ)/128), ((-21 : ℚ)/128), ((-80 : ℚ)/128), ((-50 : ℚ)/128), ((-21 : ℚ)/128), ((3 : ℚ)/128), ((203 : ℚ)/128), ((159 : ℚ)/128), ((101 : ℚ)/128), ((-44 : ℚ)/128), ((-20 : ℚ)/128), ((18 : ℚ)/128), ((102 : ℚ)/128), ((-55 : ℚ)/128), ((46 : ℚ)/128), ((174 : ℚ)/128), ((140 : ℚ)/128), ((1 : ℚ)/128), ((38 : ℚ)/128), ((-17 : ℚ)/128), ((-41 : ℚ)/128), ((36 : ℚ)/128), ((73 : ℚ)/128), ((9 : ℚ)/128), ((-4 : ℚ)/128), ((12 : ℚ)/128), ((45 : ℚ)/128), ((44 : ℚ)/128), ((-133 : ℚ)/128), ((-194 : ℚ)/128), ((-54 : ℚ)/128), ((38 : ℚ)/128)],
    ![((7 : ℚ)/128), ((-14 : ℚ)/128), ((-14 : ℚ)/128), ((6 : ℚ)/128), ((-75 : ℚ)/128), ((-15 : ℚ)/128), ((15 : ℚ)/128), ((-45 : ℚ)/128), ((-64 : ℚ)/128), ((52 : ℚ)/128), ((20 : ℚ)/128), ((-40 : ℚ)/128), ((-22 : ℚ)/128), ((14 : ℚ)/128), ((-32 : ℚ)/128), ((16 : ℚ)/128), ((37 : ℚ)/128), ((-145 : ℚ)/128), ((55 : ℚ)/128), ((254 : ℚ)/128), ((146 : ℚ)/128), ((-45 : ℚ)/128), ((-116 : ℚ)/128), ((50 : ℚ)/128), ((126 : ℚ)/128), ((-57 : ℚ)/128), ((-117 : ℚ)/128), ((72 : ℚ)/128), ((-3 : ℚ)/128), ((132 : ℚ)/128), ((147 : ℚ)/128), ((83 : ℚ)/128), ((19 : ℚ)/128), ((-1 : ℚ)/128), ((173 : ℚ)/128), ((6 : ℚ)/128), ((115 : ℚ)/128), ((145 : ℚ)/128), ((39 : ℚ)/128), ((129 : ℚ)/128), ((219 : ℚ)/128), ((24 : ℚ)/128), ((-13 : ℚ)/128), ((-160 : ℚ)/128), ((-13 : ℚ)/128), ((18 : ℚ)/128), ((-60 : ℚ)/128), ((3 : ℚ)/128), ((-10 : ℚ)/128)]]

noncomputable def W1t : Fin 8 → Fin 49 → ℝ := castM W1tQ

/-- Trained output-layer weights (10×8), entries `k/128`. -/
def W2tQ : Fin 10 → Fin 8 → ℚ :=
  ![![((-93 : ℚ)/128), ((-292 : ℚ)/128), ((295 : ℚ)/128), ((-35 : ℚ)/128), ((189 : ℚ)/128), ((-55 : ℚ)/128), ((-58 : ℚ)/128), ((96 : ℚ)/128)],
    ![((-244 : ℚ)/128), ((329 : ℚ)/128), ((-196 : ℚ)/128), ((249 : ℚ)/128), ((-381 : ℚ)/128), ((215 : ℚ)/128), ((-153 : ℚ)/128), ((23 : ℚ)/128)],
    ![((-297 : ℚ)/128), ((152 : ℚ)/128), ((86 : ℚ)/128), ((-128 : ℚ)/128), ((175 : ℚ)/128), ((-344 : ℚ)/128), ((116 : ℚ)/128), ((301 : ℚ)/128)],
    ![((-89 : ℚ)/128), ((143 : ℚ)/128), ((308 : ℚ)/128), ((-87 : ℚ)/128), ((-133 : ℚ)/128), ((-61 : ℚ)/128), ((288 : ℚ)/128), ((47 : ℚ)/128)],
    ![((308 : ℚ)/128), ((-79 : ℚ)/128), ((-471 : ℚ)/128), ((27 : ℚ)/128), ((28 : ℚ)/128), ((2 : ℚ)/128), ((152 : ℚ)/128), ((-147 : ℚ)/128)],
    ![((27 : ℚ)/128), ((-229 : ℚ)/128), ((223 : ℚ)/128), ((276 : ℚ)/128), ((-212 : ℚ)/128), ((321 : ℚ)/128), ((-13 : ℚ)/128), ((37 : ℚ)/128)],
    ![((-236 : ℚ)/128), ((-125 : ℚ)/128), ((-179 : ℚ)/128), ((427 : ℚ)/128), ((283 : ℚ)/128), ((105 : ℚ)/128), ((113 : ℚ)/128), ((-166 : ℚ)/128)],
    ![((221 : ℚ)/128), ((227 : ℚ)/128), ((120 : ℚ)/128), ((-209 : ℚ)/128), ((129 : ℚ)/128), ((128 : ℚ)/128), ((-207 : ℚ)/128), ((-401 : ℚ)/128)],
    ![((140 : ℚ)/128), ((23 : ℚ)/128), ((-61 : ℚ)/128), ((109 : ℚ)/128), ((-119 : ℚ)/128), ((-189 : ℚ)/128), ((-238 : ℚ)/128), ((270 : ℚ)/128)],
    ![((465 : ℚ)/128), ((86 : ℚ)/128), ((-184 : ℚ)/128), ((-170 : ℚ)/128), ((-65 : ℚ)/128), ((-279 : ℚ)/128), ((-193 : ℚ)/128), ((-85 : ℚ)/128)]]

noncomputable def W2t : Fin 10 → Fin 8 → ℝ := castM W2tQ

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
    · simp [denseE_apply, W1t, W1tQ, xt, hpreVals, Fin.sum_univ_succ, castM]
      norm_num

/-- Frobenius² of `W1t` is ≤ C₁² for C₁ = 2911/200 ≈ ‖W1t‖_F. -/
theorem W1t_lip : LipschitzL2 ((2911 : ℝ)/200) (denseE W1t) := by
  refine denseE_lipschitzL2 W1t (by norm_num) ?_
  simp [W1t, W1tQ, Fin.sum_univ_succ, castM]
  norm_num

theorem W2t_lip : LipschitzL2 ((1822 : ℝ)/125) (denseE W2t) := by
  refine denseE_lipschitzL2 W2t (by norm_num) ?_
  simp [W2t, W2tQ, Fin.sum_univ_succ, castM]
  norm_num

/-- Product certificate for the trained net: L = C₂·(1·C₁) = 2651921/12500. -/
theorem mlpT_lip : LipschitzL2 ((2651921 : ℝ)/12500) mlpT := by
  have h := W2t_lip.comp (reluE_lipschitzL2.comp W1t_lip (by norm_num)) (by norm_num)
  have e : ((1822 : ℝ)/125) * (1 * ((2911 : ℝ)/200)) = ((2651921 : ℝ)/12500) := by norm_num
  rw [e] at h; exact h

/-- In-kernel margin: class 2 leads every other class at `xt` by ≥ 6953/500. -/
theorem xt_margin : ∀ j : Fin 10, j ≠ 2 →
    ((6953 : ℝ)/500) ≤ mlpT xt 2 - mlpT xt j := by
  have hout : ∀ jj : Fin 10, mlpT xt jj = ∑ k : Fin 8, W2t jj k * max (hpreVals k) 0 :=
    mlp_out_eq W1t W2t hpre_eval
  intro j hj
  fin_cases j <;>
    first
    | exact absurd rfl hj
    | · rw [hout, hout]
        simp [W2t, W2tQ, hpreVals, Fin.sum_univ_succ, max_def, castM]
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
def G1tQ : Fin 8 → Fin 8 → ℚ :=
  ![![((581457 : ℚ)/16384), ((-62530 : ℚ)/16384), ((130497 : ℚ)/16384), ((-69516 : ℚ)/16384), ((-80622 : ℚ)/16384), ((29867 : ℚ)/16384), ((-70132 : ℚ)/16384), ((-71816 : ℚ)/16384)],
    ![((-62530 : ℚ)/16384), ((352025 : ℚ)/16384), ((51933 : ℚ)/16384), ((-10049 : ℚ)/16384), ((-6301 : ℚ)/16384), ((-26056 : ℚ)/16384), ((-43168 : ℚ)/16384), ((-9355 : ℚ)/16384)],
    ![((130497 : ℚ)/16384), ((51933 : ℚ)/16384), ((615605 : ℚ)/16384), ((-15531 : ℚ)/16384), ((-164389 : ℚ)/16384), ((-108692 : ℚ)/16384), ((-184108 : ℚ)/16384), ((-14418 : ℚ)/16384)],
    ![((-69516 : ℚ)/16384), ((-10049 : ℚ)/16384), ((-15531 : ℚ)/16384), ((435472 : ℚ)/16384), ((41253 : ℚ)/16384), ((-126995 : ℚ)/16384), ((54877 : ℚ)/16384), ((48774 : ℚ)/16384)],
    ![((-80622 : ℚ)/16384), ((-6301 : ℚ)/16384), ((-164389 : ℚ)/16384), ((41253 : ℚ)/16384), ((375274 : ℚ)/16384), ((64070 : ℚ)/16384), ((26286 : ℚ)/16384), ((46006 : ℚ)/16384)],
    ![((29867 : ℚ)/16384), ((-26056 : ℚ)/16384), ((-108692 : ℚ)/16384), ((-126995 : ℚ)/16384), ((64070 : ℚ)/16384), ((392316 : ℚ)/16384), ((5746 : ℚ)/16384), ((124768 : ℚ)/16384)],
    ![((-70132 : ℚ)/16384), ((-43168 : ℚ)/16384), ((-184108 : ℚ)/16384), ((54877 : ℚ)/16384), ((26286 : ℚ)/16384), ((5746 : ℚ)/16384), ((324766 : ℚ)/16384), ((25088 : ℚ)/16384)],
    ![((-71816 : ℚ)/16384), ((-9355 : ℚ)/16384), ((-14418 : ℚ)/16384), ((48774 : ℚ)/16384), ((46006 : ℚ)/16384), ((124768 : ℚ)/16384), ((25088 : ℚ)/16384), ((393745 : ℚ)/16384)]]

noncomputable def G1t : Fin 8 → Fin 8 → ℝ := castM G1tQ

/-- Exact Gram matrix `G2t = W2t·W2tᵀ` (10×10). -/
def G2tQ : Fin 10 → Fin 10 → ℚ :=
  ![![((233489 : ℚ)/16384), ((-212663 : ℚ)/16384), ((87250 : ℚ)/16384), ((26452 : ℚ)/16384), ((-163212 : ℚ)/16384), ((67065 : ℚ)/16384), ((15920 : ℚ)/16384), ((-53271 : ℚ)/16384), ((-13918 : ℚ)/16384), ((-110593 : ℚ)/16384)],
    ![((-212663 : ℚ)/16384), ((483518 : ℚ)/16384), ((-77712 : ℚ)/16384), ((-18693 : ℚ)/16384), ((-38979 : ℚ)/16384), ((95714 : ℚ)/16384), ((51511 : ℚ)/16384), ((-53983 : ℚ)/16384), ((59832 : ℚ)/16384), ((-99078 : ℚ)/16384)],
    ![((87250 : ℚ)/16384), ((-77712 : ℚ)/16384), ((388111 : ℚ)/16384), ((131057 : ℚ)/16384), ((-169849 : ℚ)/16384), ((-196872 : ℚ)/16384), ((-42411 : ℚ)/16384), ((-160231 : ℚ)/16384), ((40571 : ℚ)/16384), ((-82469 : ℚ)/16384)],
    ![((26452 : ℚ)/16384), ((-18693 : ℚ)/16384), ((131057 : ℚ)/16384), ((237366 : ℚ)/16384), ((-153105 : ℚ)/16384), ((16132 : ℚ)/16384), ((-108454 : ℚ)/16384), ((-35493 : ℚ)/16384), ((-65940 : ℚ)/16384), ((-104884 : ℚ)/16384)],
    ![((-163212 : ℚ)/16384), ((-38979 : ℚ)/16384), ((-169849 : ℚ)/16384), ((-153105 : ℚ)/16384), ((369176 : ℚ)/16384), ((-83883 : ℚ)/16384), ((82737 : ℚ)/16384), ((19323 : ℚ)/16384), ((-6599 : ℚ)/16384), ((199281 : ℚ)/16384)],
    ![((67065 : ℚ)/16384), ((95714 : ℚ)/16384), ((-196872 : ℚ)/16384), ((16132 : ℚ)/16384), ((-83883 : ℚ)/16384), ((328598 : ℚ)/16384), ((66286 : ℚ)/16384), ((-75346 : ℚ)/16384), ((-7363 : ℚ)/16384), ((-171506 : ℚ)/16384)],
    ![((15920 : ℚ)/16384), ((51511 : ℚ)/16384), ((-42411 : ℚ)/16384), ((-108454 : ℚ)/16384), ((82737 : ℚ)/16384), ((66286 : ℚ)/16384), ((417130 : ℚ)/16384), ((-98132 : ℚ)/16384), ((-103689 : ℚ)/16384), ((-215533 : ℚ)/16384)],
    ![((-53271 : ℚ)/16384), ((-53983 : ℚ)/16384), ((-160231 : ℚ)/16384), ((-35493 : ℚ)/16384), ((19323 : ℚ)/16384), ((-75346 : ℚ)/16384), ((-98132 : ℚ)/16384), ((395126 : ℚ)/16384), ((-92487 : ℚ)/16384), ((165676 : ℚ)/16384)],
    ![((-13918 : ℚ)/16384), ((59832 : ℚ)/16384), ((40571 : ℚ)/16384), ((-65940 : ℚ)/16384), ((-6599 : ℚ)/16384), ((-7363 : ℚ)/16384), ((-103689 : ℚ)/16384), ((-92487 : ℚ)/16384), ((215157 : ℚ)/16384), ((143222 : ℚ)/16384)],
    ![((-110593 : ℚ)/16384), ((-99078 : ℚ)/16384), ((-82469 : ℚ)/16384), ((-104884 : ℚ)/16384), ((199281 : ℚ)/16384), ((-171506 : ℚ)/16384), ((-215533 : ℚ)/16384), ((165676 : ℚ)/16384), ((143222 : ℚ)/16384), ((412917 : ℚ)/16384)]]

noncomputable def G2t : Fin 10 → Fin 10 → ℝ := castM G2tQ

theorem G1t_eq : ∀ a b, G1t a b = ∑ j, W1t a j * W1t b j :=
  gram_eq_of_check G1tQ W1tQ (by decide +kernel)

theorem G2t_eq : ∀ a b, G2t a b = ∑ j, W2t a j * W2t b j :=
  gram_eq_of_check G2tQ W2tQ (by decide +kernel)

/-- Schatten-4 Lipschitz bound for the hidden layer: B₁ = 46/5 ≈ (Σσ⁴)^(1/4). -/
theorem W1t_lip_gram : LipschitzL2 ((46 : ℝ)/5) (denseE W1t) := by
  refine denseE_lipschitzL2_gram W1t G1t (by norm_num) G1t_eq ?_
  simp [G1t, G1tQ, Fin.sum_univ_succ, castM]
  norm_num

theorem W2t_lip_gram : LipschitzL2 ((9661 : ℝ)/1000) (denseE W2t) := by
  refine denseE_lipschitzL2_gram W2t G2t (by norm_num) G2t_eq ?_
  simp [G2t, G2tQ, Fin.sum_univ_succ, castM]
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
    simp [W1t, W1tQ, v1t, Fin.sum_univ_succ, castM]
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
    simp [W2t, W2tQ, v2t, Fin.sum_univ_succ, castM]
    norm_num


/-- `H1t = G1t²` (= `G1tᵀ·G1t`, 8×8, denominators 16384² = 268435456). -/
def H1tQ : Fin 8 → Fin 8 → ℚ :=
  ![![((381332164867 : ℚ)/268435456), ((-47465880949 : ℚ)/268435456), ((177999653658 : ℚ)/268435456), ((-86561467680 : ℚ)/268435456), ((-104293457991 : ℚ)/268435456), ((10828395667 : ℚ)/268435456), ((-92445641522 : ℚ)/268435456), ((-76464339304 : ℚ)/268435456)],
    ![((-47465880949 : ℚ)/268435456), ((133299230401 : ℚ)/268435456), ((54198370774 : ℚ)/268435456), ((-4149457835 : ℚ)/268435456), ((-11727690771 : ℚ)/268435456), ((-27449614525 : ℚ)/268435456), ((-35493143767 : ℚ)/268435456), ((-8348761388 : ℚ)/268435456)],
    ![((177999653658 : ℚ)/268435456), ((54198370774 : ℚ)/268435456), ((471878560057 : ℚ)/268435456), ((-29702498181 : ℚ)/268435456), ((-186845141077 : ℚ)/268435456), ((-118425395158 : ℚ)/268435456), ((-190683370373 : ℚ)/268435456), ((-50910988355 : ℚ)/268435456)],
    ![((-86561467680 : ℚ)/268435456), ((-4149457835 : ℚ)/268435456), ((-29702498181 : ℚ)/268435456), ((218030459641 : ℚ)/268435456), ((37216491814 : ℚ)/268435456), ((-96207401852 : ℚ)/268435456), ((51466358618 : ℚ)/268435456), ((33184252901 : ℚ)/268435456)],
    ![((-104293457991 : ℚ)/268435456), ((-11727690771 : ℚ)/268435456), ((-186845141077 : ℚ)/268435456), ((37216491814 : ℚ)/268435456), ((183008208623 : ℚ)/268435456), ((65455693299 : ℚ)/268435456), ((58378950753 : ℚ)/268435456), ((54263966873 : ℚ)/268435456)],
    ![((10828395667 : ℚ)/268435456), ((-27449614525 : ℚ)/268435456), ((-118425395158 : ℚ)/268435456), ((-96207401852 : ℚ)/268435456), ((65455693299 : ℚ)/268435456), ((203129512810 : ℚ)/268435456), ((21006791861 : ℚ)/268435456), ((94638911450 : ℚ)/268435456)],
    ![((-92445641522 : ℚ)/268435456), ((-35493143767 : ℚ)/268435456), ((-190683370373 : ℚ)/268435456), ((51466358618 : ℚ)/268435456), ((58378950753 : ℚ)/268435456), ((21006791861 : ℚ)/268435456), ((150515547253 : ℚ)/268435456), ((30723710906 : ℚ)/268435456)],
    ![((-76464339304 : ℚ)/268435456), ((-8348761388 : ℚ)/268435456), ((-50910988355 : ℚ)/268435456), ((33184252901 : ℚ)/268435456), ((54263966873 : ℚ)/268435456), ((94638911450 : ℚ)/268435456), ((30723710906 : ℚ)/268435456), ((181179974310 : ℚ)/268435456)]]

noncomputable def H1t : Fin 8 → Fin 8 → ℝ := castM H1tQ

def H2tQ : Fin 10 → Fin 10 → ℚ :=
  ![![((154706574877 : ℚ)/268435456), ((-133154423928 : ℚ)/268435456), ((105162270548 : ℚ)/268435456), ((66616904064 : ℚ)/268435456), ((-136227062252 : ℚ)/268435456), ((38420763311 : ℚ)/268435456), ((14283159418 : ℚ)/268435456), ((-63730236730 : ℚ)/268435456), ((-29152700930 : ℚ)/268435456), ((-118665041676 : ℚ)/268435456)],
    ![((-133154423928 : ℚ)/268435456), ((315048189041 : ℚ)/268435456), ((-83900165830 : ℚ)/268435456), ((-18997790332 : ℚ)/268435456), ((-7415928267 : ℚ)/268435456), ((105769853290 : ℚ)/268435456), ((71898469185 : ℚ)/268435456), ((-57955990883 : ℚ)/268435456), ((27856661667 : ℚ)/268435456), ((-92588568781 : ℚ)/268435456)],
    ![((105162270548 : ℚ)/268435456), ((-83900165830 : ℚ)/268435456), ((284984878483 : ℚ)/268435456), ((124823686979 : ℚ)/268435456), ((-166694153962 : ℚ)/268435456), ((-103218270754 : ℚ)/268435456), ((-48789575816 : ℚ)/268435456), ((-132305244086 : ℚ)/268435456), ((19945143209 : ℚ)/268435456), ((-93433200902 : ℚ)/268435456)],
    ![((66616904064 : ℚ)/268435456), ((-18997790332 : ℚ)/268435456), ((124823686979 : ℚ)/268435456), ((126639834428 : ℚ)/268435456), ((-150191603417 : ℚ)/268435456), ((10115431095 : ℚ)/268435456), ((-55754654619 : ℚ)/268435456), ((-48657743459 : ℚ)/268435456), ((-25610859957 : ℚ)/268435456), ((-105312396150 : ℚ)/268435456)],
    ![((-136227062252 : ℚ)/268435456), ((-7415928267 : ℚ)/268435456), ((-166694153962 : ℚ)/268435456), ((-150191603417 : ℚ)/268435456), ((270749873136 : ℚ)/268435456), ((-72340307485 : ℚ)/268435456), ((34534871586 : ℚ)/268435456), ((90044010142 : ℚ)/268435456), ((18081190359 : ℚ)/268435456), ((206644002559 : ℚ)/268435456)],
    ![((38420763311 : ℚ)/268435456), ((105769853290 : ℚ)/268435456), ((-103218270754 : ℚ)/268435456), ((10115431095 : ℚ)/268435456), ((-72340307485 : ℚ)/268435456), ((207230088439 : ℚ)/268435456), ((100211574032 : ℚ)/268435456), ((-68155927861 : ℚ)/268435456), ((-32175848646 : ℚ)/268435456), ((-174071170590 : ℚ)/268435456)],
    ![((14283159418 : ℚ)/268435456), ((71898469185 : ℚ)/268435456), ((-48789575816 : ℚ)/268435456), ((-55754654619 : ℚ)/268435456), ((34534871586 : ℚ)/268435456), ((100211574032 : ℚ)/268435456), ((268540246657 : ℚ)/268435456), ((-102206602332 : ℚ)/268435456), ((-80097153395 : ℚ)/268435456), ((-196883283183 : ℚ)/268435456)],
    ![((-63730236730 : ℚ)/268435456), ((-57955990883 : ℚ)/268435456), ((-132305244086 : ℚ)/268435456), ((-48657743459 : ℚ)/268435456), ((90044010142 : ℚ)/268435456), ((-68155927861 : ℚ)/268435456), ((-102206602332 : ℚ)/268435456), ((240492915630 : ℚ)/268435456), ((-28761135239 : ℚ)/268435456), ((186727506677 : ℚ)/268435456)],
    ![((-29152700930 : ℚ)/268435456), ((27856661667 : ℚ)/268435456), ((19945143209 : ℚ)/268435456), ((-25610859957 : ℚ)/268435456), ((18081190359 : ℚ)/268435456), ((-32175848646 : ℚ)/268435456), ((-80097153395 : ℚ)/268435456), ((-28761135239 : ℚ)/268435456), ((95975758982 : ℚ)/268435456), ((96108682451 : ℚ)/268435456)],
    ![((-118665041676 : ℚ)/268435456), ((-92588568781 : ℚ)/268435456), ((-93433200902 : ℚ)/268435456), ((-105312396150 : ℚ)/268435456), ((206644002559 : ℚ)/268435456), ((-174071170590 : ℚ)/268435456), ((-196883283183 : ℚ)/268435456), ((186727506677 : ℚ)/268435456), ((96108682451 : ℚ)/268435456), ((373892277385 : ℚ)/268435456)]]

noncomputable def H2t : Fin 10 → Fin 10 → ℝ := castM H2tQ

theorem H1t_eq : ∀ a b, H1t a b = ∑ c, G1t c a * G1t c b :=
  gram_eq_of_check H1tQ (fun a c => G1tQ c a) (by decide +kernel)

theorem H2t_eq : ∀ a b, H2t a b = ∑ c, G2t c a * G2t c b :=
  gram_eq_of_check H2tQ (fun a c => G2tQ c a) (by decide +kernel)

/-- Schatten-8 bound: B₁' = 7769/1000 ≈ (Σσ⁸)^(1/8) (true σ₁ ≈ 7.4525). -/
theorem W1t_lip_gram2 : LipschitzL2 ((7769 : ℝ)/1000) (denseE W1t) := by
  refine denseE_lipschitzL2_gram2 W1t G1t H1t (by norm_num) G1t_eq H1t_eq ?_
  simp [H1t, H1tQ, Fin.sum_univ_succ, castM]
  norm_num

theorem W2t_lip_gram2 : LipschitzL2 ((8211 : ℝ)/1000) (denseE W2t) := by
  refine denseE_lipschitzL2_gram2 W2t G2t H2t (by norm_num) G2t_eq H2t_eq ?_
  simp [H2t, H2tQ, Fin.sum_univ_succ, castM]
  norm_num

/-- Schatten-8 product certificate: L = B₂'·(1·B₁') = 63791259/1000000 ≈ 63.79 — vs the
    certified lower bounds ℓ₁·ℓ₂ = 57.38, provably within 11.2% of the
    per-layer-optimal product. -/
theorem mlpT_lip_gram2 : LipschitzL2 ((63791259 : ℝ)/1000000) mlpT := by
  have h := W2t_lip_gram2.comp (reluE_lipschitzL2.comp W1t_lip_gram2 (by norm_num)) (by norm_num)
  have e : ((8211 : ℝ)/1000) * (1 * ((7769 : ℝ)/1000)) = ((63791259 : ℝ)/1000000) := by norm_num
  rw [e] at h; exact h

theorem trained_radius_gram2_pos : 0 < ((6953 : ℝ)/500) / (Real.sqrt 2 * ((63791259 : ℝ)/1000000)) :=
  div_pos (by norm_num)
    (mul_pos (Real.sqrt_pos.mpr (by norm_num)) (by norm_num))

/-- **Schatten-8 trained certificate**: radius ≈ 0.1541 (3.3× Frobenius,
    1.4× Schatten-4; the true-σ ceiling for the product method is 0.171). -/
theorem trained_demo_certified_gram2 (δ : EuclideanSpace ℝ (Fin 49))
    (hδ : ‖δ‖ < ((6953 : ℝ)/500) / (Real.sqrt 2 * ((63791259 : ℝ)/1000000))) :
    ∀ j, j ≠ 2 → mlpT (xt + δ) j < mlpT (xt + δ) 2 :=
  lipschitz_margin_certified_radius mlpT_lip_gram2 (by norm_num) xt_margin hδ


/-- Each logit of the trained pooled-MNIST MLP is continuous: the coordinate
    formula `∑ k, W2ⱼₖ·max(∑ l, W1ₖₗ·xₗ, 0)` is definitional. -/
theorem mlpT_logit_continuous : ∀ j : Fin 10, Continuous fun x => mlpT x j := by
  intro j
  simp only [mlpT, Function.comp_apply, denseE_apply, reluE_apply]
  refine continuous_finsetSum _ fun k _ => continuous_const.mul ?_
  refine Continuous.max ?_ continuous_const
  exact continuous_finsetSum _ fun l _ =>
    continuous_const.mul (EuclideanSpace.proj l).continuous

end LipschitzCertDemo
end Proofs
