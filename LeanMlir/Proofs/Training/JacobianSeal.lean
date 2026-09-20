import LeanMlir.Proofs.Foundation.MLP

/-!
# Nonzero-Jacobian seal — the generic "the backward is non-trivial here" bridge

The whole-network capstones prove `HasVJP.backward = pdiv`-Jacobian-transpose. A
*degenerate* witness (zero weights / constant output) satisfies that contract
**vacuously**: its Jacobian is identically zero, so the backward map is the zero map and
says nothing about a real gradient. A non-vacuity fact of the weaker kind
(`Mnv2FullBSeal.sealX_nonconstant`) only rules out a *constant forward* — strictly
weaker than a non-trivial backward at the witness.

This file supplies the missing **level-3 seal** (see `planning/archive/whole_network_backward.md`,
Item B): the reusable bridge from a single nonzero Jacobian entry to a provably non-trivial
backward, and the equivalence with `fderiv ℝ f x ≠ 0`. A witness then upgrades from
"forward ≠ const" to "the rendered backward at this point is not the zero map" by exhibiting
**one** `pdiv f x i j ≠ 0` — which is what a genuine (non-degenerate) gradient requires.

The bridge is stated for the pointwise `HasVJPAt` the kinked witnesses are built as. The
per-net seals discharge its `pdiv ≠ 0` premise on the full-width batched nets themselves
(`ResNet34FullBSeal`, `ResNet50FullBSeal`, `MobileNetV2FullBSeal`, `MobileNetV4FullBSeal`).
-/

namespace Proofs

open scoped BigOperators
open Finset

-- ════════════════════════════════════════════════════════════════
-- § Jacobian entries and `fderiv`
-- ════════════════════════════════════════════════════════════════

/-- The standard basis decomposition `∑ᵢ vᵢ · eᵢ = v` on `Vec m`. -/
theorem sum_smul_basisVec {m : Nat} (v : Vec m) :
    (∑ i : Fin m, v i • basisVec i) = v := by
  funext k; simp [Finset.sum_apply]

/-- **All Jacobian entries zero ⇒ the Fréchet derivative is the zero map.** `fderiv ℝ f x`
    is ℝ-linear, so it is determined by its values on the standard basis; if those all
    vanish it vanishes everywhere (`v = ∑ᵢ vᵢ·eᵢ`). No differentiability hypothesis —
    at a non-smooth point `fderiv` is its junk-`0` default and the entries are `0` too. -/
theorem fderiv_eq_zero_of_pdiv_all_zero {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (hall : ∀ i j, pdiv f x i j = 0) :
    fderiv ℝ f x = 0 := by
  -- a row of `pdiv` is `fderiv` on that basis vector, by definition
  refine ContinuousLinearMap.ext fun v => ?_
  rw [← sum_smul_basisVec v, map_sum]
  simp [fun i => (funext (hall i) : fderiv ℝ f x (basisVec i) = 0)]

/-- **The seal in `fderiv` form.** A nonzero Fréchet derivative at the witness yields a
    nonzero Jacobian entry — the clean analytic hypothesis behind
    `HasVJPAt.backward_ne_zero_of_pdiv_ne`. (Contrapositive of the all-zero lemma.) -/
theorem exists_pdiv_ne_of_fderiv_ne {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (hfd : fderiv ℝ f x ≠ 0) :
    ∃ (i : Fin m) (j : Fin n), pdiv f x i j ≠ 0 := by
  simpa [not_forall] using mt (fderiv_eq_zero_of_pdiv_all_zero f x) hfd

-- ════════════════════════════════════════════════════════════════
-- § The seal (`HasVJPAt`)
-- ════════════════════════════════════════════════════════════════

/-- **The nonzero-Jacobian seal.** If the Jacobian of `f` at the witness `x` has a nonzero
    entry `pdiv f x i₀ j₀ ≠ 0`, then the proven backward there is not the zero map: probing
    it with the basis cotangent `e_{j₀}` returns the nonzero `pdiv f x i₀ j₀` at row `i₀`.
    The cotangent collapses `HasVJPAt.correct`'s sum to its single diagonal term. -/
theorem HasVJPAt.backward_ne_zero_of_pdiv_ne {m n : Nat} {f : Vec m → Vec n}
    {x : Vec m} (h : HasVJPAt f x) {i₀ : Fin m} {j₀ : Fin n}
    (hpd : pdiv f x i₀ j₀ ≠ 0) :
    h.backward (basisVec j₀) i₀ ≠ 0 := by
  simpa [h.correct] using hpd

/-- **The seal in `fderiv` form.** A nonzero Fréchet derivative at the witness `x` ⇒ the
    proven backward there is non-trivial (some basis-cotangent probe returns a nonzero row).
    The form a whole-net witness uses: establish `fderiv ℝ forward x ≠ 0` once, get a
    non-trivial backward for free. -/
theorem HasVJPAt.backward_nontrivial_of_fderiv_ne {m n : Nat} {f : Vec m → Vec n}
    {x : Vec m} (h : HasVJPAt f x) (hfd : fderiv ℝ f x ≠ 0) :
    ∃ (j₀ : Fin n) (i₀ : Fin m), h.backward (basisVec j₀) i₀ ≠ 0 := by
  obtain ⟨i₀, j₀, hpd⟩ := exists_pdiv_ne_of_fderiv_ne f x hfd
  exact ⟨j₀, i₀, h.backward_ne_zero_of_pdiv_ne hpd⟩

-- ════════════════════════════════════════════════════════════════
-- § Discharging `fderiv ≠ 0` along a ray
--   The live witnesses exhibit one readout of the output (a channel, or a
--   channel difference) whose derivative along a ray `t ↦ x + t • v` is
--   nonzero at `t = 0`; along the ray the readout is `t · Q t`, `Q` continuous.
-- ════════════════════════════════════════════════════════════════

/-- **A nonzero directional derivative seals `fderiv ≠ 0`.** If a readout `ℓ` of `f`
    (differentiable at `f x`) has derivative `c ≠ 0` along the ray `t ↦ x + t • v` at
    `t = 0`, the Fréchet derivative of `f` at `x` is not the zero map (a zero one would
    give the readout derivative `0`). -/
theorem fderiv_ne_zero_of_ray {m n : Nat} {f : Vec m → Vec n} {x : Vec m} (v : Vec m)
    (hf : DifferentiableAt ℝ f x) (ℓ : Vec n → ℝ) (hℓ : DifferentiableAt ℝ ℓ (f x)) {c : ℝ}
    (hc : c ≠ 0) (hg : HasDerivAt (fun t : ℝ => ℓ (f (x + t • v))) c 0) :
    fderiv ℝ f x ≠ 0 := by
  intro hzero
  have hray : HasDerivAt (fun t : ℝ => x + t • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add x
  have h0 := (hℓ.hasFDerivAt.comp x (hzero ▸ hf.hasFDerivAt)).comp_hasDerivAt_of_eq (0 : ℝ)
    hray (by simp)
  simp only [ContinuousLinearMap.comp_zero, zero_apply] at h0
  exact hc (hg.unique h0)

/-- **`t · Q t` has derivative `Q 0` at `0`** for any `Q` continuous there — the
    product-rule cross-term carries the factor `t`, so no derivative of `Q` is needed
    (its slope at `0` is `Q` itself). -/
theorem hasDerivAt_mul_self_zero {Q : ℝ → ℝ} (hQ : ContinuousAt Q 0) :
    HasDerivAt (fun t : ℝ => t * Q t) (Q 0) 0 := by
  rw [hasDerivAt_iff_tendsto_slope]
  refine Filter.Tendsto.congr' ?_ (hQ.tendsto.mono_left nhdsWithin_le_nhds)
  filter_upwards [self_mem_nhdsWithin] with y hy
  rw [slope_def_field, sub_zero, zero_mul, sub_zero, mul_div_cancel_left₀ _ hy]

/-- ⭐ **`S · Q` at a zero of `S`** — `hasDerivAt_mul_self_zero` where the carrier reaches the
    readout through a smooth but NON-AFFINE stage, so it is not `t` itself that factors out.

    MobileNetV4's fused stage is swish: its two examples' outputs differ by `swishGap β u`, which
    vanishes at `u = 0` and is differentiable there but is not a multiple of `u`. `S` is that gap
    along the ray and `Q` the BatchNorm factors below it; as before no derivative of `Q` is
    needed, only its continuity at `0`. -/
theorem hasDerivAt_mul_of_zero {S Q : ℝ → ℝ} {c : ℝ} (hS : HasDerivAt S c 0) (hS0 : S 0 = 0)
    (hQ : ContinuousAt Q 0) : HasDerivAt (fun t : ℝ => S t * Q t) (c * Q 0) 0 := by
  rw [hasDerivAt_iff_tendsto_slope]
  have h1 := hasDerivAt_iff_tendsto_slope.mp hS
  have h2 := hQ.tendsto.mono_left (nhdsWithin_le_nhds (a := (0 : ℝ)) (s := {(0 : ℝ)}ᶜ))
  refine Filter.Tendsto.congr' ?_ (h1.mul h2)
  filter_upwards [self_mem_nhdsWithin] with y hy
  simp only [slope_def_field, hS0, sub_zero, zero_mul]
  ring

end Proofs
