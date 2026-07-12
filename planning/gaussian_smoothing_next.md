# Gaussian smoothing: state of the chain + next moves

*(written 2026-07-12, for a fresh session; predecessor threads:
`planning/robustness_handoff.md`, the `mathlib-upstream-pr1-pr2` branch)*

## Where the chain stands (all 3-axiom clean, in `Certs`)

The randomized-smoothing story is now **end-to-end theorems**, one file per
layer:

| layer | file | apex |
|---|---|---|
| Neyman–Pearson / probit-Lipschitz | `LeanMlir/Proofs/SmoothingGaussian.lean` | `smoothing_probit_lipschitz` (the (1/σ)-Lipschitzness of `Φ⁻¹∘p_c` — Cohen's analytic heart, DISCHARGED, real pi-Gaussian) |
| radius algebra | same + `LipschitzCert.lean` | `smoothing_certified_radius_classifier` (radius `σ·Φ⁻¹(p)` for the TRUE class probability, measurable hard classifier, hypotheses = measurability + `p ∈ (0,1)`) |
| sampling confidence | `LeanMlir/Proofs/SmoothingMC.lean` (NEW 2026-07-12) | `smoothing_mc_certified`: with prob `≥ 1 − exp(−2Nt²)` over `N` iid Gaussian samples, the radius `σ·Φ⁻¹(p̂ − t)` reported from the EMPIRICAL frequency is genuinely certified |

`SmoothingMC.lean` internals (reusable pieces):
- `iIndepFun_eval_pi` — coordinates of `Measure.pi` are iid (via
  `iIndepFun_iff_map_fun_eq_pi_map` + `measurePreserving_eval`);
- `mc_mean_lower_bound` — one-sided Hoeffding for `[0,1]`-valued MC means
  over the product measure, built on Mathlib's `HasSubgaussianMGF`
  (`hasSubgaussianMGF_of_mem_Icc`, c = 1/4, +
  `HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun`);
- `stdNormalQuantile_of_nonpos` — `Φ⁻¹(q≤0) = 0` (junk value ⇒ the `p̂−t ≤ 0`
  case certifies vacuously). Uses `stdNormalCDF_pos`
  (`SmoothingGaussian.lean:717`).

The guarantee shape now matches Cohen–Rosenfeld–Kolter's CERTIFY exactly:
a confidence-qualified radius, quantified over the sample draw.

## The `mathlib-upstream-pr1-pr2` branch (unmerged, compiles on the pin)

Two Mathlib-PR drafts, staged in `LeanMlir/Proofs/UpstreamDraft.lean`
(scratch namespace `MathlibUpstream`, NOT in any build target) + polished
copies in `planning/mathlib_upstream_drafts/PR{1_CDF,2_GaussianReal}.lean`:

- **PR1 (generic `cdf`)**: `strictMono_cdf_iff` (⟺ `IsOpenPosMeasure`),
  `continuous_cdf_iff` (⟺ `NoAtoms`), `cdf_pos/lt_one/mem_Ioo`.
- **PR2 (`gaussianReal`)**: the PR1 facts instantiated
  (`cdf_gaussianReal_{pos,lt_one,mem_Ioo}`, `strictMono/continuous_cdf_gaussianReal`)
  + `cdf_gaussianReal_neg` (symmetry) + `cdf_gaussianReal_sub_const`
  (mean shift).

NOTE: `stdNormalCDF` is DEFINED as `cdf (gaussianReal 0 1)` — the branch
lemmas apply verbatim, no bridge needed. The MC tie did NOT end up needing
them (the repo already had bespoke `stdNormalCDF_pos`/`strictMono`); they
become load-bearing at the next rung (below). Verify the branch still
compiles: `git show origin/mathlib-upstream-pr1-pr2:LeanMlir/Proofs/UpstreamDraft.lean | lake env lean /dev/stdin`-style, or merge it.

## Next moves, in rough value order

1. **Two-sided quantile inverse via continuity (uses the branch!).**
   `stdNormalQuantile_inv` (`Φ(Φ⁻¹ p) = p` on `(0,1)`) exists
   (`SmoothingGaussian.lean:188`), and `stdNormalQuantile_cdf` (`Φ⁻¹(Φ s) = s`,
   line 708). What's missing is the clean packaging: `Φ⁻¹` STRICTLY monotone
   on `(0,1)` (currently only `MonotoneOn`), continuity of `Φ⁻¹`, and
   `Φ⁻¹ ∘ Φ = id` globally — all falling out of `continuous_cdf` +
   `strictMono_cdf` (PR1/PR2) via `StrictMono.orderIsoOfSurjective`-style
   arguments or `Continuous.strictMonoOn_inv`. This is where the branch gets
   used in anger; merge it first (it adds 3 files, no conflicts), lift
   `UpstreamDraft` into the build (Certs root + audit), then swap
   `SmoothingGaussian`'s bespoke positivity/mono proofs to cite it (optional,
   cosmetic).

2. **Exact Clopper–Pearson instead of Hoeffding.** `mc_mean_lower_bound`'s
   `exp(−2Nt²)` is the crude bound; Cohen's actual CERTIFY uses the exact
   binomial lower confidence limit. The empirical count `k = Σᵢ 1[C(x+σωᵢ)=y]`
   is `Binomial(N, p)` — Mathlib has `PMF.binomial`; the missing piece is
   "the count of successes over `Measure.pi` is binomial" (map of the sum =
   binomial PMF — provable by induction, or via `iIndepFun` + Bernoulli
   convolution if Mathlib has it). Then the CP bound is a monotonicity
   statement about the binomial CDF in `p` — continuity/monotonicity
   arguments where PR1-style lemmas again help. Payoff: the theorem
   matches the deployed driver's arithmetic exactly.

3. **Tie the `smoothCertify` DRIVER'S reported radii** (`f30f857`,
   `mnist-{mlp,cnn}-smooth`, `cifar-smooth` exes). The driver samples on GPU
   and reports `σ·Φ⁻¹(p̂ − t)`-style radii; with (2) its exact arithmetic
   becomes an instance of the theorem. A smoothing SCORECARD mirroring the
   Lipschitz one (fixed 100 images, radius per image, `1−α` column) would be
   the demo artifact — but note the scorecard data lives in `CertsHeavy`
   territory (generated files got their own lib + workflow 2026-07-12;
   keep new generated instances OUT of `Certs`).

4. **σ over ℝ≥0 / nonstandard-σ variants, and the `t`-per-class refinement**
   (Cohen uses one-sided bounds on `p_A` only; the classifier theorem's
   runner-up bound is automatic from disjointness — already exploited).
   Low priority.

## Iteration recipe / gotchas for the fresh session

- Fast loop: `lake env lean LeanMlir/Proofs/SmoothingMC.lean` (~30 s wall;
  imports cached — build `Certs` once first). No generated data anywhere in
  this thread: pure hand-written lemma iteration.
- Mathlib-name drift on this pin (4.31): `le_or_lt` → `le_or_gt`;
  `measureReal_univ_eq_one` → `probReal_univ`; `List.length_eq_zero_iff`;
  `div_le_iff₀`/`lt_div_iff₀` (₀-suffixed order lemmas);
  `measure_sum_ge_le_of_iIndepFun` lives in namespace `HasSubgaussianMGF`.
- Measure-rewriting friction: prefer `exact congrArg (∫ y, f y ∂·) h.map_eq`
  / calc blocks over `rw` when `Function.eval i` vs `fun ω => ω i` spellings
  collide; `Measure.map_id` needs the function literally `id` (use a `have
  h1 : (...) = id := rfl` first).
- `set q := ... with hq` inside a proof whose STATEMENT has set-builders:
  fine — but remember the set-builder in a THEOREM statement needs its
  closing `}` before `:= by` (bit us once).
- The audit gate: new theorems → `tests/AuditAxioms.lean` (imports + `#print
  axioms` lines; keep to the exact-triple, no `propext`-only lines — the CI
  grep counts exact matches) + `scripts/check_audit_coverage.py` must pass.
  New proof files must land in a lake target (`Certs` roots) — standing rule.
- Sanity anchors: `#print axioms Proofs.smoothing_mc_certified` →
  `[propext, Classical.choice, Quot.sound]`.
