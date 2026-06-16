import LeanMlir.Proofs.MLP

/-!
# Nonzero-Jacobian seal — the generic "the backward is non-trivial here" bridge

The whole-network capstones prove `HasVJP.backward = pdiv`-Jacobian-transpose. A
*degenerate* witness (zero weights / constant output, e.g. `MobileNetV2Concrete`,
`ResNet34Concrete`) satisfies that contract **vacuously**: its Jacobian is identically
zero, so the backward map is the zero map and says nothing about a real gradient. The
non-vacuity facts that exist today (`Mnv2Live.mnv2Live_forward_nonconstant`) only rule out
a *constant forward* — strictly weaker than a non-trivial backward at the witness.

This file supplies the missing **level-3 seal** (see `planning/whole_network_backward.md`,
Item B): the reusable bridge from a single nonzero Jacobian entry to a provably non-trivial
backward, and the equivalence with `fderiv ℝ f x ≠ 0`. A witness then upgrades from
"forward ≠ const" to "the rendered backward at this point is not the zero map" by exhibiting
**one** `pdiv f x i j ≠ 0` — which is what a genuine (non-degenerate) gradient requires.

The bridge itself is generic; discharging `pdiv ≠ 0` at a specific deep kinked witness
(`Mnv2Live`, a future `ResNet34Live`) is the per-net follow-up (Item B2). Demonstrated here
end-to-end on the linear classifier (`mnistLinear`), whose Jacobian is exactly `W`.
-/

namespace Proofs

open scoped BigOperators
open Finset

-- ════════════════════════════════════════════════════════════════
-- § The generic seal
-- ════════════════════════════════════════════════════════════════

/-- **The nonzero-Jacobian seal.** If the Jacobian of `f` at `x` has a nonzero entry
    `pdiv f x i₀ j₀ ≠ 0`, then the proven backward map is not the zero map there: probing
    it with the basis cotangent `e_{j₀}` returns the nonzero `pdiv f x i₀ j₀` at row `i₀`.
    The cotangent collapses `HasVJP.correct`'s sum to its single diagonal term. -/
theorem HasVJP.backward_ne_zero_of_pdiv_ne {m n : Nat} {f : Vec m → Vec n}
    (h : HasVJP f) (x : Vec m) {i₀ : Fin m} {j₀ : Fin n}
    (hpd : pdiv f x i₀ j₀ ≠ 0) :
    h.backward x (basisVec j₀) i₀ ≠ 0 := by
  rw [h.correct]
  have hsum : (∑ j : Fin n, pdiv f x i₀ j * basisVec j₀ j) = pdiv f x i₀ j₀ := by
    rw [Finset.sum_eq_single j₀]
    · rw [basisVec_apply, if_pos rfl, mul_one]
    · intro j _ hj; rw [basisVec_apply, if_neg hj, mul_zero]
    · intro hni; exact absurd (Finset.mem_univ j₀) hni
  rw [hsum]; exact hpd

/-- The standard basis decomposition `∑ᵢ vᵢ · eᵢ = v` on `Vec m`. -/
theorem sum_smul_basisVec {m : Nat} (v : Vec m) :
    (∑ i : Fin m, v i • basisVec i) = v := by
  funext k
  rw [Finset.sum_apply]
  simp only [Pi.smul_apply, smul_eq_mul, basisVec_apply]
  rw [Finset.sum_eq_single k]
  · rw [if_pos rfl, mul_one]
  · intro b _ hb; rw [if_neg (fun h => hb h.symm), mul_zero]
  · intro hni; exact absurd (Finset.mem_univ k) hni

/-- A whole row of the Jacobian vanishing is the Fréchet derivative vanishing on that
    basis vector: `pdiv f x i · = (fderiv ℝ f x) eᵢ`, by definition of `pdiv`. -/
theorem fderiv_basisVec_eq_zero_of_pdiv_row {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    {i : Fin m} (hrow : ∀ j, pdiv f x i j = 0) :
    fderiv ℝ f x (basisVec i) = 0 := by
  funext j; exact hrow j

/-- **All Jacobian entries zero ⇒ the Fréchet derivative is the zero map.** `fderiv ℝ f x`
    is ℝ-linear, so it is determined by its values on the standard basis; if those all
    vanish it vanishes everywhere (`v = ∑ᵢ vᵢ·eᵢ`). No differentiability hypothesis —
    at a non-smooth point `fderiv` is its junk-`0` default and the entries are `0` too. -/
theorem fderiv_eq_zero_of_pdiv_all_zero {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (hall : ∀ i j, pdiv f x i j = 0) :
    fderiv ℝ f x = 0 := by
  have hbasis : ∀ i, fderiv ℝ f x (basisVec i) = 0 := fun i =>
    fderiv_basisVec_eq_zero_of_pdiv_row f x (hall i)
  apply ContinuousLinearMap.ext
  intro v
  calc fderiv ℝ f x v
      = fderiv ℝ f x (∑ i : Fin m, v i • basisVec i) := by rw [sum_smul_basisVec]
    _ = ∑ i : Fin m, fderiv ℝ f x (v i • basisVec i) := by rw [map_sum]
    _ = ∑ i : Fin m, v i • fderiv ℝ f x (basisVec i) := by simp_rw [map_smul]
    _ = 0 := by simp_rw [hbasis]; simp

/-- **The seal in `fderiv` form.** A nonzero Fréchet derivative at the witness yields a
    nonzero Jacobian entry — the clean analytic hypothesis behind
    `HasVJP.backward_ne_zero_of_pdiv_ne`. (Contrapositive of the all-zero lemma.) -/
theorem exists_pdiv_ne_of_fderiv_ne {m n : Nat} (f : Vec m → Vec n) (x : Vec m)
    (hfd : fderiv ℝ f x ≠ 0) :
    ∃ (i : Fin m) (j : Fin n), pdiv f x i j ≠ 0 := by
  by_contra hcon
  refine hfd (fderiv_eq_zero_of_pdiv_all_zero f x ?_)
  intro i j
  by_contra hne
  exact hcon ⟨i, j, hne⟩

/-- Packaging: a nonzero Fréchet derivative ⇒ the proven backward is non-trivial at `x`
    (some basis-cotangent probe returns a nonzero row). The form a whole-net witness uses:
    establish `fderiv ℝ forward x ≠ 0` once, get a non-trivial backward for free. -/
theorem HasVJP.backward_nontrivial_of_fderiv_ne {m n : Nat} {f : Vec m → Vec n}
    (h : HasVJP f) (x : Vec m) (hfd : fderiv ℝ f x ≠ 0) :
    ∃ (j₀ : Fin n) (i₀ : Fin m), h.backward x (basisVec j₀) i₀ ≠ 0 := by
  obtain ⟨i₀, j₀, hpd⟩ := exists_pdiv_ne_of_fderiv_ne f x hfd
  exact ⟨j₀, i₀, h.backward_ne_zero_of_pdiv_ne x hpd⟩

-- ════════════════════════════════════════════════════════════════
-- § The seal, pointwise (`HasVJPAt`)
--   The deep kinked witnesses (`Mnv2Live`, a future `ResNet34Live`) are
--   built as *pointwise* `HasVJPAt f x`, not the global `HasVJP f`. Their
--   `.correct` field has the same `backward = pdiv`-contraction shape, so
--   the seal transfers verbatim — this is the form Item B2 actually consumes.
-- ════════════════════════════════════════════════════════════════

/-- **The nonzero-Jacobian seal, pointwise.** `HasVJPAt` analogue of
    `HasVJP.backward_ne_zero_of_pdiv_ne`: one nonzero Jacobian entry at the
    witness point `x` makes the proven backward there not the zero map. -/
theorem HasVJPAt.backward_ne_zero_of_pdiv_ne {m n : Nat} {f : Vec m → Vec n}
    {x : Vec m} (h : HasVJPAt f x) {i₀ : Fin m} {j₀ : Fin n}
    (hpd : pdiv f x i₀ j₀ ≠ 0) :
    h.backward (basisVec j₀) i₀ ≠ 0 := by
  rw [h.correct]
  have hsum : (∑ j : Fin n, pdiv f x i₀ j * basisVec j₀ j) = pdiv f x i₀ j₀ := by
    rw [Finset.sum_eq_single j₀]
    · rw [basisVec_apply, if_pos rfl, mul_one]
    · intro j _ hj; rw [basisVec_apply, if_neg hj, mul_zero]
    · intro hni; exact absurd (Finset.mem_univ j₀) hni
  rw [hsum]; exact hpd

/-- **The seal in `fderiv` form, pointwise.** `HasVJPAt` analogue of
    `HasVJP.backward_nontrivial_of_fderiv_ne`: a nonzero Fréchet derivative
    at the witness `x` ⇒ the proven backward there is non-trivial. -/
theorem HasVJPAt.backward_nontrivial_of_fderiv_ne {m n : Nat} {f : Vec m → Vec n}
    {x : Vec m} (h : HasVJPAt f x) (hfd : fderiv ℝ f x ≠ 0) :
    ∃ (j₀ : Fin n) (i₀ : Fin m), h.backward (basisVec j₀) i₀ ≠ 0 := by
  obtain ⟨i₀, j₀, hpd⟩ := exists_pdiv_ne_of_fderiv_ne f x hfd
  exact ⟨j₀, i₀, h.backward_ne_zero_of_pdiv_ne hpd⟩

-- ════════════════════════════════════════════════════════════════
-- § Demonstration — the linear classifier (Jacobian = W)
-- ════════════════════════════════════════════════════════════════

/-- **Non-trivial backward for the linear classifier.** `pdiv (mnistLinear W b) = W`, so any
    nonzero weight `W i₀ j₀ ≠ 0` seals the backward as non-trivial at every input. The
    simplest end-to-end instance of the seal; the deep kinked witnesses (`Mnv2Live`,
    `ResNet34Live`) discharge the same `pdiv ≠ 0` premise through their BN-deviation chains. -/
theorem mnistLinear_backward_nontrivial {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) {i₀ : Fin m} {j₀ : Fin n} (hW : W i₀ j₀ ≠ 0) :
    (dense_has_vjp W b).backward x (basisVec j₀) i₀ ≠ 0 :=
  (dense_has_vjp W b).backward_ne_zero_of_pdiv_ne x (by rw [pdiv_dense]; exact hW)

end Proofs
