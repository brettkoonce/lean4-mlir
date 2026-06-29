# Planning — Power iteration: the bridge that *computes* the spectral norm the Lipschitz cert assumes

**The gap this closes.** The robustness certificate (`LeanMlir/Proofs/LipschitzCert.lean`) composes
per-layer Lipschitz constants into a global one: `clm_lipschitzL2` (a linear map's `L` = its operator
norm `‖W‖₂`) + `LipschitzL2.comp` (compose layers) ⟹ `∏‖Wᵢ‖₂`, which feeds
`lipschitz_margin_certified_radius` / `smoothing_certified_radius`. **But the cert *assumes* you hand
it `‖Wᵢ‖₂`.** Nothing in the repo *computes* it. Power iteration is the missing computational half —
and it's a **Newton–Schulz cousin** that reuses the exact spectral engine from the Muon ladder, so it
welds the two threads: [[muon-geometry-proof-ladder]]'s `conj_diag_pow` ↔ [[robustness-ladder-thread]]'s
cert. ([[math-threads-demo-first]] Thread 1 ∩ Thread 4.)

## 0. The one idea — power iteration is `conj_diag_pow` on the spectrum

`‖W‖₂ = σ₁(W) = √(λ_max(WᵀW))`. Power iteration `xₖ₊₁ = WᵀW xₖ / ‖WᵀW xₖ‖` drives `xₖ` to the top
eigenvector of `WᵀW`, and the Rayleigh quotient `ρ(xₖ) = ⟨xₖ, WᵀW xₖ⟩ / ⟨xₖ, xₖ⟩ → λ_max` at rate
`(λ₂/λ₁)ᵏ`. **Why it's the same motif as NS:** with the spectral decomposition `WᵀW = V diag(λ) Vᵀ`
(`V` orthonormal — exactly `svd_of_isUnit`'s `V`), `(WᵀW)ᵏ = V diag(λᵏ) Vᵀ` (**`conj_diag_pow`,
already proven**). In the `V`-basis the iterate is `diag(λᵏ)` acting on coordinates, so the `λ_max`
coordinate dominates — convergence is a *scalar ratio* `(λ₂/λ₁)ᵏ → 0`, the same spectral reduction
that ran through NS P1–P3. Power iteration : eigenvector :: Newton–Schulz : polar factor.

## 1. Layered plan (spectral reduction first, then the ratio analysis)

New file `LeanMlir/Proofs/PowerIteration.lean`, `Proofs` lib root, audited, target 3-axiom clean.

- ⬜ **PI1 — the spectral-step lemma (reuse the engine).** With `A := WᵀW = V diag(λ) Vᵀ` (`VᵀV = 1`,
  `λᵢ ≥ 0` from `svd_of_isUnit` / `Matrix.IsHermitian.spectral_theorem`), `Aᵏ = V diag(λᵏ) Vᵀ` by
  `conj_diag_pow` (imported from `MuonGeometry`). So `Aᵏ x₀ = V diag(λᵏ) (Vᵀ x₀)` — the iterate's
  coordinates in the eigenbasis are `cᵢ·λᵢᵏ`. Pure reuse, the only matrix-level step.
- ⬜ **PI2 — Rayleigh-quotient convergence (the scalar ratio).** For `x₀` with a nonzero top-eigenvector
  component (`c_max ≠ 0`), `ρ(Aᵏ x₀) = (Σ cᵢ² λᵢ^{2k+1}) / (Σ cᵢ² λᵢ^{2k}) → λ_max`. Factor out
  `λ_max^{2k}`: numerator/denominator → `c_max² λ_max` / `c_max²` as `(λᵢ/λ_max)ᵏ → 0` for `i ≠ max`.
  Cousin of NS P2's `gCubic_iterate_tendsto_one` (a scalar limit driving a vector/matrix limit); the
  geometric `(λ₂/λ₁)ᵏ → 0` is `tendsto_pow_atTop_nhds_zero_of_lt_one`.
- ⬜ **PI3 — tie to the certificate.** `λ_max = σ₁² = ‖W‖₂²`, so the converged Rayleigh quotient *is*
  the per-layer Lipschitz constant `clm_lipschitzL2` consumes. State `powerIter_tendsto_opNormSq` and
  compose with `LipschitzL2.comp` → the certified radius (`lipschitz_margin_certified_radius`) now
  rests on a **computed**, convergence-proven `L`, not an assumed one. Closes the cert's open input.

## 2. Honest scoping (the load-bearing caveat — mirror NS P4)

**Direction matters for soundness.** A *certificate* needs an **upper bound** `‖W‖₂ ≤ L̄`. The Rayleigh
quotient converges **from below** (`ρ(x) ≤ λ_max` always), so a *finite* power-iteration run gives a
**lower** bound — the *wrong* direction for a sound cert (it would *overstate* robustness). The honest
options, in increasing rigor:
- **(a) asymptotic only** — prove `→ λ_max` (PI1–PI3) as the clean limit theorem, and be explicit that
  it certifies nothing at finite `k` by itself. This is the real deliverable; don't overclaim.
- **(b) a-posteriori upper bound** — after `k` steps with residual `r = Ax − ρx`, a Bauer–Fike /
  Weyl-style bound gives `λ_max ≤ ρ + ‖r‖/‖x‖`, an *honest* finite-step upper bound. This is the part
  that would make the cert *sound from a finite computation* — heavier, optional.
- **don't** claim a sound finite-step cert from bare power iteration. Tier it like NS (cubic converges
  cleanly; quintic only as a finite band): power iteration converges cleanly (a), the sound finite
  upper bound (b) is the separate harder rung.

## 3. Mathlib support (scout)

- ✅ reuse: `conj_diag_pow`, `svd_of_isUnit` / `Matrix.IsHermitian.spectral_theorem`, the
  `U Σ Vᵀ`/eigenbasis vocabulary (all in `MuonGeometry.lean`).
- ✅ `tendsto_pow_atTop_nhds_zero_of_lt_one` for `(λ₂/λ₁)ᵏ → 0`; monotone/squeeze (`Filter.Tendsto`)
  reused from NS P2.
- 🔎 `Matrix.IsHermitian.rayleigh*` / variational eigenvalue API; `Matrix.l2_opNorm` (bridge `σ₁` to
  Mathlib's operator norm if available — the `clm_lipschitzL2` side already uses `ContinuousLinearMap`
  op norm, so PI3 needs `σ₁(W) = ‖W.toCLM‖`).

## 4. Why this is the right next bridge

Cheapest high-value move that connects two invested threads: it makes the Lipschitz cert's spectral
norms *computed and convergence-proven* instead of assumed, reuses `conj_diag_pow` wholesale (≈ a
session), and is demo-able (a runnable power-iteration spectral-norm estimate vs. `numpy.linalg.svd`,
[[math-threads-demo-first]] demo-first protocol). Companion to `planning/robustness_ladder.md` and
`planning/muon_ns_convergence.md`. Sequence it before natural gradient (`planning/natural_gradient.md`).
