# Closing the smoothing sandwich: the Gaussian (1/σ)-Lipschitz lemma

> **STATUS: COMPLETE (2026-07-07).** G1–G4 all landed in
> `LeanMlir/Proofs/SmoothingGaussian.lean`, 3-axiom clean, four commits (G1, G2, G3, G4).
> Endpoint theorems: `smoothing_probit_lipschitz` (hg discharged),
> `smoothing_certified_radius_cohen` (soft scores) and
> `smoothing_certified_radius_classifier` (hard classifier; [0,1] bounds + runner-up bound
> free from decision-region disjointness). Deviations from plan: G2 was ~150 lines, not
> 1-2k (the pointwise MLR trick needs no layer-cake); G3's rotation invariance ships with
> Mathlib (`stdGaussian_map` / `stdGaussian_eq_map_pi_orthonormalBasis`); the only
> irreducible extra hypothesis is `hp : p ∈ Ioo 0 1` (Φ⁻¹ needs it; Monte-Carlo estimates
> always satisfy it) — the plan's "unconditional" claim holds modulo exactly that.

**Goal.** Discharge the `hg` hypothesis of `Proofs.smoothing_certified_radius`
(`LipschitzCert.lean:209`) — the one quarantined assumption keeping the randomized-smoothing
radius from being an unconditional theorem. Everything else (the margin algebra, the
runner-up bound, the radius formula) is already proved; `hg` is the Cohen–Rosenfeld–Kolter
hard half:

> for `f : E → [0,1]` and `g(x) := Φ⁻¹(𝔼_{Z ~ N(0,σ²I)}[f(x+Z)])`, `g` is `(1/σ)`-Lipschitz.

This is net-independent — proving it upgrades the smoothing certs of EVERY `*-smooth`
driver, including the canonical 784→512→512→10 `mlpVerified` (the one net where the
Lipschitz-product cert is measured vacuous, so smoothing is the only paradigm that
reaches canonical scale; see `Proofs/MlpCanonical.lean` header).

## Statement plan (matching the existing abstract hypotheses)

`smoothing_certified_radius` already takes `Phiinv` abstract with `hmono : Monotone Phiinv`
and `hanti : Phiinv (1−p) = −Phiinv p`. The concrete instantiation:

- `Φ : ℝ → ℝ` := cdf of `gaussianReal 0 1` (Mathlib `Probability.Distributions.Gaussian`
  + `MeasureTheory.cdf`).
- `Φ⁻¹` on (0,1) via strict-mono continuous surjectivity (`OrderIso` restriction or
  `Function.invFun` + the cdf's `StrictMono`/`Continuous`/tendsto-at-±∞ facts).
- `p_c(x) := (Measure.pi fun _ => gaussianReal 0 σ²).map (x + ·) |>.real (f ⁻¹' {c})`
  or, cleaner, `pᶜ(x) := ∫ z, ind (f (x + z) = c) ∂(σ-Gaussian)`.

## Milestones (each a standalone commit, all 3-axiom-clean)

**G1 — Φ and Φ⁻¹ (cheap, high leverage).** Standard-normal cdf: strict mono, continuous,
`Tendsto Φ atBot (𝓝 0)` / `atTop (𝓝 1)`, symmetry `Φ(−t) = 1 − Φ(t)` (from the pdf's
evenness), hence `Φ⁻¹` with `hmono` and `hanti` DISCHARGED with the real function. Payoff:
`smoothing_certified_radius` instantiated at the true Φ⁻¹ — the remaining gap becomes
exactly the NP core, stated in one hypothesis. Mathlib has the pdf/measure and cdf API;
symmetry may need a small integral substitution lemma.

**G2 — the 1-D Neyman–Pearson core.** For `f : ℝ → [0,1]` measurable and
`μ_t := gaussianReal t σ²`: if `∫ f dμ₀ ≥ p` then `∫ f dμ_δ ≥ Φ(Φ⁻¹(p) − δ/σ)`.
Proof shape: the likelihood ratio `dμ_δ/dμ₀ (z) = exp((δz − δ²/2)/σ²)` is monotone in `z`,
so the infimum over `{f : ∫ f dμ₀ = p}` is attained at the halfspace indicator
`ind (z ≤ σ·Φ⁻¹(p))` — a rearrangement/level-set argument. This is the real work:
likely ~1–2k lines. Mathlib pieces: `gaussianReal` absolute continuity + density
(`gaussianPDF`), `exp` monotonicity, layer-cake (`MeasureTheory.lintegral` level-set
machinery). No multivariate measure theory yet.

**G3 — dimension reduction.** For direction `u = δ/‖δ‖` in `EuclideanSpace ℝ (Fin n)`:
the σ-Gaussian pi-measure is invariant under `LinearIsometryEquiv` (rotate `u` to `e₀`),
and pushing forward along `⟨·, u⟩` gives `gaussianReal 0 σ²` — so the n-D statement
collapses to G2 applied to the conditional slice. Needs: rotational invariance of the
iid-Gaussian pi measure (may not be in Mathlib — provable via `MeasurePreserving` of
orthogonal maps on `Measure.pi` gaussians through the density; this is the riskiest
Mathlib-gap item, scout first).

**G4 — assembly.** `smoothing_probit_lipschitz : LipschitzL2 (1/σ) (fun x => Φ⁻¹ (pᶜ x))`
from G2+G3 (two-sided by applying the bound in both directions), then
`smoothing_certified_radius_unconditional` — the Cohen radius as a theorem with NO
smoothing-side hypotheses (only measurability of the classifier and `0 < σ`). Update the
`ff2b8a2` "both sides of the sandwich" claim to drop its qualifier.

## Order and scoping

G1 first (small, immediately improves the headline statement); scout Mathlib for G3's
rotational invariance before committing to G2's full generality (if the invariance is
missing, G2+G3 can be fused by working directly with the n-D density ratio, which only
depends on `⟨z, δ⟩` — arguably simpler than two lemmas). Keep everything in a new
`Proofs/SmoothingGaussian.lean`; wire into `Certs` + `AuditAxioms` on first commit
(the SpecVJP lesson: no orphan proof files).

## G1 DONE (2026-07-07)

Landed exactly as scoped below, all 3-axiom-clean:

- `smoothing_certified_radius_probit` (LipschitzCert.lean) — the Ioo variant
  (`hp : p c y ∈ Ioo 0 1`, `MonotoneOn`/Ioo-`hanti`), original theorem untouched.
- `Proofs/SmoothingGaussian.lean` — `stdNormalCDF := cdf (gaussianReal 0 1)`,
  `stdNormalQuantile p := sSup {t | Φ t < p}`; strict mono (`stdGaussian_Ioo_pos` via
  `setLIntegral_pos_iff` + `support_gaussianPDF`), symmetry `stdNormalCDF_neg` (Mathlib
  HAD the map-neg lemma: `gaussianReal_map_neg`), quantile `MonotoneOn` + `hanti` via the
  no-flat-step bridge `stdNormalCDF_sSup_lt_eq_sInf_gt` (a gap ⇒ two points with Φ = q ⇒
  strict-mono contradiction; `Real.sSup_neg` turns the negated set into `−sInf`).
- Capstone `smoothing_certified_radius_gaussian`: only `hg` (the NP core) remains.
- Wired: lakefile `Certs` roots + `LeanMlir.lean` + `tests/AuditAxioms.lean` (6 new
  `#print axioms` entries) in the same change.

## G2 + G3 DONE (2026-07-07, same day)

**G2** (commit after G1): `stdNormalCDF_quantile` (Φ(Φ⁻¹p)=p — right-continuity + no-atoms
left-limit) + `gaussian_np_shift` — the 1-D NP core came in at ~150 lines, NOT the planned
1-2k: the pointwise MLR inequality `(f − 1_{z≤t})·(LR − LR(t)) ≥ 0` integrated against the
BASE Gaussian needs no layer-cake/rearrangement machinery at all.

**G3**: the rotation-invariance scout hit gold — Mathlib HAS it
(`stdGaussian_map` for any LinearIsometryEquiv, `map_pi_eq_stdGaussian`,
`stdGaussian_eq_map_pi_orthonormalBasis`), so the G2/G3 split stood. Chain:
`integral_gaussianReal_shift_eq` (1-D Cameron–Martin) → `pi_gaussian_shift_eq` (Fubini at
coordinate 0 via `measurePreserving_piFinSuccAbove`) → `pi_gaussian_np_shift` (G2's MLR
trick against the pi measure, halfspace `{z₀ ≤ t}`) → `stdGaussian_np_shift` (adapted ONB:
`Submodule.reflection_sub` carries `e₀` to `δ/‖δ‖`, `OrthonormalBasis.map`). Stated on
`Fin (n+1)` (nonempty index). All 3-axiom clean.

## Session handoff (2026-07-07, pre-G1 analysis DONE)

**Design discovery (load-bearing):** `hmono : Monotone Phiinv` in
`smoothing_certified_radius` (LipschitzCert.lean:208) is UNDISCHARGEABLE by any total
real Φ⁻¹ — the true quantile is unbounded on (0,1), so no total ℝ-valued function can be
globally (or even Icc-)monotone and agree with it. Traced the proof: hmono/hanti are used
exactly 3× (hmono at `p j x` vs `1 − p i x`; hanti at `p i x`; hmono at `p {i,j} (x+δ)`)
— all probability values. So G1 = a VARIANT theorem `smoothing_certified_radius_probit`:

- add `(hp : ∀ c y, p c y ∈ Set.Ioo (0:ℝ) 1)` — realistic, Monte-Carlo/Clopper–Pearson
  estimates are never exactly 0/1;
- weaken to `hmono : MonotoneOn Phiinv (Set.Ioo 0 1)` and
  `hanti : ∀ q ∈ Set.Ioo (0:ℝ) 1, Phiinv (1 - q) = -Phiinv q`;
- copy the original ~25-line proof, threading Ioo memberships (`1 − p ∈ Ioo` ✓).
  Keep the original theorem untouched.

**Then instantiate with the real thing** (new file `Proofs/SmoothingGaussian.lean`):
`stdNormalCDF := MeasureTheory.cdf (gaussianReal 0 1)`;
`stdNormalQuantile p := sSup {t | stdNormalCDF t < p}`. Needed facts: strict mono of the
cdf (gaussianPDF positivity ⇒ positive mass on intervals), symmetry `Φ(−t) = 1 − Φ(t)`
(via neg-map invariance of `gaussianReal 0 1` — grep Mathlib for a `map neg` lemma first,
else prove through the density's evenness), quantile MonotoneOn + hanti on Ioo.
Mathlib pin inventory (checked): `Probability/Distributions/Gaussian/Real.lean` has
`gaussianReal`/`gaussianPDF` + `IsProbabilityMeasure`; `Probability/CDF.lean` has the
generic `cdf` (StieltjesFunction). NO prebuilt Gaussian-cdf mono/symmetry facts — G1
builds them.

**Process:** wire the new file into `Certs` roots + `tests/AuditAxioms.lean` in the SAME
commit that creates it (the SpecVJP orphan lesson); iterate lemmas in a scratch file
importing the cached LipschitzCert olean (~1s) before moving them in. G2 (1-D
Neyman–Pearson) next; scout pi-Gaussian rotation invariance before choosing the G2/G3
split (fusing via the ⟨z,δ⟩-only density ratio may be simpler).
