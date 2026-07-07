# Closing the smoothing sandwich: the Gaussian (1/σ)-Lipschitz lemma

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
