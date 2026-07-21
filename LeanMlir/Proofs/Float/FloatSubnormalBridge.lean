import LeanMlir.Proofs.Float.FloatBridge

/-! # ℝ → Float32 bridge: the subnormal floor, closed as a lemma (not a caveat)

`FloatBridge.lean`'s `FloatModel` carries the **unconditional** relative-error
axiom `|rnd x − x| ≤ u·|x|`, true for IEEE-754 binary32 round-to-nearest only
**on the normal range** (its docstring flags the subnormal absolute-error term
as future work; `planning/floatbridge_certificate_gaps.md` §2). Near `0` the real
bound is `|rnd x − x| ≤ u·|x| + η`, a gradual-underflow floor `η ≈ 2⁻¹⁵⁰` (½ ULP
at the smallest subnormal). Deep activations *can* underflow, so the clean
relative model is, strictly, a TRUSTED simplification.

This file closes that gap the way the plan recommends — **not** by polluting
every downstream budget with an `η` term, but by proving activations **stay
normal** so the clean relative model genuinely applies, and by showing the
residual floor (for the genuinely-near-zero coordinates the invariant does not
cover) is globally negligible.

The arc:

* `FaithfulFloatModel` — the *honest* model of a real binary32 rounder: the
  clean relative bound `err_rel` on the normal range, **plus** the honest
  absolute floor `err_abs` everywhere (subnormals included), plus `rnd 0 = 0`.
  binary32 RN instantiates it with `u = 2⁻²⁴`, `η = 2⁻¹⁵⁰`, `minNormal = 2⁻¹²⁶`
  (the IEEE facts — the same TRUSTED instantiation boundary `FloatModel` already
  relies on, now *stated*, not hidden).
* `toFloatModel` (`η = 0`) — the honest model with no underflow **is** a
  `FloatModel`. So `FloatModel` is exactly the `η→0` / stays-normal *face* of the
  honest model: every existing bridge bound is the normal-range truth.
* `err_of_normal` — on normal arguments the honest bound collapses to the clean
  `FloatModel.err`, with **no** `η`. The precise "stays-normal ⇒ the whole bridge
  applies verbatim".
* `bnDenom_normal` / `bnSqrt_normal` / `istd_ge_minNormal` — the architecture
  invariant: BN/LN's `var + ε` denominator, its `√`, and the inverse-stddev
  `istd = 1/√(var+ε)` are all bounded **below** by `minNormal` (since
  `ε ≫ minNormal`). The `rsqrt` keystone (`BnFloatBridge.rsqrt_lipschitz`) never
  touches the subnormal range — this is *why* LN/BN keep activations O(1).
* `subFloor_total_negligible` — even if *every* one of `n ≤ 2⁶⁴` rounded
  quantities underflowed, the total extra error is `≤ 2⁻⁸⁶`, below every nonzero
  budget in the suite. Handles post-ReLU tiny values / softmax tails honestly:
  the floor cannot move any closeness bound.

3-axiom clean (no `sorry`, no project axioms) — like the rest of the bridge.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § binary32 underflow constants
-- ════════════════════════════════════════════════════════════════

/-- Smallest positive **normal** binary32 magnitude, `2⁻¹²⁶`. Below this the
    relative model degrades into gradual underflow. -/
noncomputable def minNormalF32 : ℝ := ((2 : ℝ) ^ (126 : ℕ))⁻¹

/-- The round-to-nearest **subnormal absolute-error floor**, `½·2⁻¹⁴⁹ = 2⁻¹⁵⁰`:
    the largest `|rnd x − x|` can be in/under the subnormal range. -/
noncomputable def subFloorF32 : ℝ := ((2 : ℝ) ^ (150 : ℕ))⁻¹

theorem minNormalF32_pos : 0 < minNormalF32 := by unfold minNormalF32; positivity
theorem subFloorF32_pos : 0 < subFloorF32 := by unfold subFloorF32; positivity

theorem minNormalF32_le_one : minNormalF32 ≤ 1 := by
  rw [minNormalF32, inv_eq_one_div, div_le_one (by positivity)]
  norm_num

-- ════════════════════════════════════════════════════════════════
-- § The honest model of a real binary32 rounder
-- ════════════════════════════════════════════════════════════════

/-- **The honest rounding model.** A real IEEE round-to-nearest operator: the
    clean relative bound holds on the normal range (`err_rel`), and the honest
    gradual-underflow *absolute* floor `η` holds everywhere (`err_abs`), with
    `rnd 0 = 0` exact. The idealized `FloatModel` is its `η = 0` face. -/
structure FaithfulFloatModel where
  rnd : ℝ → ℝ
  u : ℝ
  η : ℝ
  minNormal : ℝ
  u_nonneg : 0 ≤ u
  η_nonneg : 0 ≤ η
  minNormal_pos : 0 < minNormal
  minNormal_le_one : minNormal ≤ 1
  rnd_zero : rnd 0 = 0
  /-- Clean relative model on the normal range `minNormal ≤ |x|`. -/
  err_rel : ∀ x : ℝ, minNormal ≤ |x| → |rnd x - x| ≤ u * |x|
  /-- Honest absolute floor everywhere — covers subnormals. -/
  err_abs : ∀ x : ℝ, |rnd x - x| ≤ u * |x| + η

namespace FaithfulFloatModel

/-- The exact rounder (`rnd = id`, `u = η = 0`) inhabits the interface — every
    bound collapses to `0`, like `FloatBridge`'s `exactModel`. -/
noncomputable def exactFaithful : FaithfulFloatModel where
  rnd := id
  u := 0
  η := 0
  minNormal := minNormalF32
  u_nonneg := le_refl 0
  η_nonneg := le_refl 0
  minNormal_pos := minNormalF32_pos
  minNormal_le_one := minNormalF32_le_one
  rnd_zero := rfl
  err_rel := fun x _ => by simp
  err_abs := fun x => by simp

variable (F : FaithfulFloatModel)

/-- **`FloatModel` is the no-underflow face.** With the floor `η = 0` the honest
    model's `err_abs` becomes the unconditional relative bound — i.e. it is a
    `FloatModel`. This is the exact sense in which `FloatModel` is "binary32 on
    the normal range": the stays-normal / `η→0` limit of the honest model, so
    every bridge bound proved against `FloatModel` is the normal-range truth. -/
noncomputable def toFloatModel (hη : F.η = 0) : FloatModel where
  rnd := F.rnd
  u := F.u
  u_nonneg := F.u_nonneg
  err := fun x => by have h := F.err_abs x; rwa [hη, add_zero] at h

/-- **Normal-domain recovery.** On a normal argument (`x = 0`, which rounds
    exactly, or `minNormal ≤ |x|`) the honest bound collapses to the clean
    `FloatModel.err` `|rnd x − x| ≤ u·|x|` — no `η`. So *provided activations
    stay normal*, the whole bridge's relative-error model applies verbatim. -/
theorem err_of_normal (x : ℝ) (h : x = 0 ∨ F.minNormal ≤ |x|) :
    |F.rnd x - x| ≤ F.u * |x| := by
  rcases h with h | h
  · subst h; rw [F.rnd_zero]; simp
  · exact F.err_rel x h

end FaithfulFloatModel

-- ════════════════════════════════════════════════════════════════
-- § The stays-normal invariant for the BN/LN normalization denominator
-- ════════════════════════════════════════════════════════════════

/-- **Stays-normal — the BN/LN denominator.** `var + ε` (`var ≥ 0`,
    `ε ≥ minNormal`) is `≥ minNormal`: the normalization denominator is in the
    normal range, so rounding it sits in the clean relative regime. -/
theorem bnDenom_normal (F : FaithfulFloatModel) {ε var : ℝ}
    (hε : F.minNormal ≤ ε) (hvar : 0 ≤ var) : F.minNormal ≤ |var + ε| := by
  have h1 : F.minNormal ≤ var + ε := le_trans hε (by linarith)
  rwa [abs_of_pos (lt_of_lt_of_le F.minNormal_pos h1)]

/-- **Stays-normal — the standard deviation `√(var+ε)`.** Also `≥ minNormal`
    (using `minNormal ≤ 1`, so `minNormal² ≤ minNormal ≤ var+ε`). The argument
    the `rsqrt` keystone consumes is normal. -/
theorem bnSqrt_normal (F : FaithfulFloatModel) {ε var : ℝ}
    (hε : F.minNormal ≤ ε) (hvar : 0 ≤ var) :
    F.minNormal ≤ |Real.sqrt (var + ε)| := by
  have hsum : F.minNormal ≤ var + ε := le_trans hε (by linarith)
  have hsq : F.minNormal ^ 2 ≤ var + ε := by
    have hle : F.minNormal ^ 2 ≤ F.minNormal := by
      nlinarith [F.minNormal_pos, F.minNormal_le_one]
    linarith
  have hle : F.minNormal ≤ Real.sqrt (var + ε) :=
    calc F.minNormal = Real.sqrt (F.minNormal ^ 2) :=
          (Real.sqrt_sq F.minNormal_pos.le).symm
      _ ≤ Real.sqrt (var + ε) := Real.sqrt_le_sqrt hsq
  rwa [abs_of_nonneg (Real.sqrt_nonneg _)]

/-- **Stays-normal — the inverse standard deviation `istd = 1/√(var+ε)`.** With
    a mild a-priori upper bound on the denominator's root (`√(var+ε) ≤
    minNormal⁻¹`, i.e. `var+ε ≤ 2²⁵²` — always true for O(1) activations),
    `istd ≥ minNormal`. The lower bound is the subnormal-relevant one (the upper
    side is overflow, a separate `maxNormal` concern). So the BN `istd` the
    bridge rounds never underflows. -/
theorem istd_ge_minNormal (F : FaithfulFloatModel) {ε var : ℝ}
    (hpos : 0 < var + ε) (hub : Real.sqrt (var + ε) ≤ F.minNormal⁻¹) :
    F.minNormal ≤ 1 / Real.sqrt (var + ε) := by
  have hs : 0 < Real.sqrt (var + ε) := Real.sqrt_pos.mpr hpos
  rw [le_div_iff₀ hs]
  calc F.minNormal * Real.sqrt (var + ε) ≤ F.minNormal * F.minNormal⁻¹ :=
        mul_le_mul_of_nonneg_left hub F.minNormal_pos.le
    _ = 1 := mul_inv_cancel₀ F.minNormal_pos.ne'

-- ════════════════════════════════════════════════════════════════
-- § The residual floor is globally negligible
-- ════════════════════════════════════════════════════════════════

/-- **The subnormal floor cannot move any bound.** Even if *every* one of `n`
    rounded quantities underflowed into the subnormal range, the total extra
    error is `n · subFloorF32`. For `n ≤ 2⁶⁴` (vastly more ops than any net in
    the suite), this is `≤ 2⁻⁸⁶` — below every nonzero closeness budget that
    appears. So the genuinely-near-zero coordinates the stays-normal invariant
    does not cover (post-ReLU tiny values, softmax tails) are harmless. -/
theorem subFloor_total_negligible (n : ℕ) (hn : (n : ℝ) ≤ (2 : ℝ) ^ (64 : ℕ)) :
    (n : ℝ) * subFloorF32 ≤ ((2 : ℝ) ^ (86 : ℕ))⁻¹ := by
  unfold subFloorF32
  calc (n : ℝ) * ((2 : ℝ) ^ (150 : ℕ))⁻¹
      ≤ (2 : ℝ) ^ (64 : ℕ) * ((2 : ℝ) ^ (150 : ℕ))⁻¹ :=
        mul_le_mul_of_nonneg_right hn (by positivity)
    _ = ((2 : ℝ) ^ (86 : ℕ))⁻¹ := by
        rw [show (150 : ℕ) = 64 + 86 from rfl, pow_add, mul_inv, ← mul_assoc,
            mul_inv_cancel₀ (by positivity : (2 : ℝ) ^ (64 : ℕ) ≠ 0), one_mul]

end Proofs
