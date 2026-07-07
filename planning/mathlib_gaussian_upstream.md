# Upstreaming the Gaussian cdf/quantile/NP lemmas to Mathlib

**Goal.** Extract the Mathlib-shaped foundations of `LeanMlir/Proofs/SmoothingGaussian.lean`
(the G1–G4 ladder, complete 2026-07-07, commits 23b40de/0738b2a/7ba0b4e/bc4a5fd) into a
sequence of small mathlib4 PRs. Verified against master docs 2026-07-07: Mathlib has **no**
Gaussian cdf lemmas, **no** quantile/inverse-cdf API anywhere, **no** Neyman–Pearson or
Cameron–Martin content. Everything below the ML layer of our file is novel upstream.

**Why.** Good karma (one minor PR already landed from this account); permanent maintenance
win (our smoothing cert then sits on Mathlib-reviewed foundations); "our Gaussian
foundations are upstream" strengthens the Diderot/comparator trust story.

**This doc is the handoff for the machine with the GitHub account.** The source repo is
`github.com/brettkoonce/lean4-mlir`, file `LeanMlir/Proofs/SmoothingGaussian.lean` — clone
or just open it on GitHub; every lemma named below lives there. Our pin is Lean/Mathlib
v4.31; PRs target master (docs-checked: master hasn't added any of this, port is
mechanical).

---

## Step 0 — Zulip first (do this before writing any code)

Post ONE thread on leanprover.zulipchat.com, stream `#mathlib4` (or `#maths` → probability
folks will find it; Rémy Degenne is the most active probability maintainer):

> **Title:** Gaussian cdf facts + a quantile API
>
> Pitch: I have proofs of (a) generic cdf lemmas — strict monotonicity from
> interval-positivity, continuity from NoAtoms; (b) Gaussian-specific cdf facts —
> `IsOpenPosMeasure (gaussianReal μ v)` (v ≠ 0), symmetry `Φ(−t) = 1 − Φ(t)`,
> `cdf ∈ Ioo 0 1`, mean-shift `cdf (gaussianReal δ 1) t = cdf (gaussianReal 0 1) (t − δ)`;
> (c) a quantile (inverse-cdf) definition with monotonicity + two-sided inversion under
> the above; (d) the 1-D and n-D Gaussian Cameron–Martin shift identity and a
> Neyman–Pearson-type halfspace-optimality bound. Motivated by a formalization of
> randomized-smoothing certificates (Cohen et al. 2019). Proposed PR sequence below —
> does the quantile design look right / where should each piece live?

Key design question to settle there: **quantile convention**. Ours is
`sSup {t | cdf t < p}`; the textbook standard is `sInf {t | p ≤ cdf t}` (left-continuous
quantile). They agree wherever the cdf is continuous + strictly monotone (our no-flat-step
lemma is exactly that bridge). Maintainers may prefer sInf, may want it as a
`StieltjesFunction`-adjacent gadget, or may want the full Galois-connection treatment. Do
not pre-commit; bring both forms.

Also disclose AI assistance in the thread and in each PR description (Mathlib policy:
human author must review, understand, and vouch for every line — the proofs are short
enough that this is genuine).

## Mechanics (once per machine)

```
git clone git@github.com:<account>/mathlib4  # fork of leanprover-community/mathlib4
cd mathlib4 && lake exe cache get            # ~5 min, downloads oleans
# per PR: branch off master, edit, then:
lake build Mathlib.Probability.CDF           # (whatever you touched)
lake exe lint-style
# push branch, open PR against leanprover-community/mathlib4 master
```

PR titles follow `feat(Probability): ...`. Every public decl needs a docstring; 100-col
lines; copyright header on new files (`Authors: Brett Koonce`). A maintainer adds
`awaiting-review`; expect days-to-weeks per round; merge is via bors (`bors r+` by a
maintainer). Keep each PR under ~300 lines.

---

## PR 1 — generic cdf lemmas (`Mathlib/Probability/CDF.lean`) — START HERE

Smallest, most reusable, zero design risk. Contents (~80–120 lines):

- `strictMono_cdf` : for `μ : Measure ℝ` `[IsProbabilityMeasure μ]` with
  `∀ s < t, 0 < μ (Ioo s t)` (or `[μ.IsOpenPosMeasure]` — cleaner): `StrictMono (cdf μ)`.
  Port of our `stdNormalCDF_strictMono`: split `Iic t = Iic s ∪ Ioc s t`
  (`measureReal_union` + `Set.Iic_union_Ioc_eq_Iic`), positivity via
  `ENNReal.toReal_pos` + `measure_ne_top`.
- `continuous_cdf` (or `leftLim_cdf_eq` + combine) : `[NoAtoms μ]` ⇒ `Continuous (cdf μ)`.
  Port of the ≤-half of our `stdNormalCDF_quantile`: `measure_cdf` +
  `StieltjesFunction.measure_singleton` + `ENNReal.ofReal_eq_zero` gives
  `leftLim (cdf μ) x = cdf μ x`; monotone + right-continuous + leftLim-agrees ⇒ continuous
  (`Monotone.continuousAt_iff_leftLim_eq` direction exists in
  `Topology/Order/LeftRightLim`).
- Optional converses (cdf strictly mono → intervals have positive mass; continuous → no
  atoms) if cheap — reviewers like iff-shaped API.

## PR 2 — Gaussian cdf facts (`Mathlib/Probability/Distributions/Gaussian/Real.lean`)

Depends on PR 1. Contents (~100–150 lines):

- `instIsOpenPosMeasure : (gaussianReal μ v).IsOpenPosMeasure` for `v ≠ 0` — the
  mathlib-idiomatic packaging of our `stdGaussian_Ioo_pos` (proof:
  `gaussianReal_apply` + `setLIntegral_pos_iff` + `support_gaussianPDF`; for a general
  open set use positivity on a contained interval). Then PR 1's `strictMono_cdf` and
  `measure_Ioo_pos` fire for free.
- `cdf_gaussianReal_neg` : `cdf (gaussianReal 0 v) (−t) = 1 − cdf (gaussianReal 0 v) t`.
  Port of `stdNormalCDF_neg` (`gaussianReal_map_neg`, `Ioi_ae_eq_Ici`,
  `measureReal_compl` + `probReal_univ`). Generalize variance — free in the proof.
- `cdf_gaussianReal_pos` / `cdf_gaussianReal_lt_one` / `∈ Ioo 0 1` — ports of
  `stdNormalCDF_pos`/`stdNormalCDF_lt_one`/`stdNormalCDF_mem_Ioo`.
- `cdf_gaussianReal_sub_const` : `cdf (gaussianReal (μ+δ) v) t = cdf (gaussianReal μ v) (t−δ)`
  — port of `cdf_gaussianReal_shift` via `gaussianReal_map_add_const`. Generalize mean —
  free.

## PR 3 — quantile API (new file, e.g. `Mathlib/Probability/Quantile.lean`) — the long pole

Blocked on the Zulip design outcome. Our material to bring:

- def (ours): `quantile μ p = sSup {t | cdf μ t < p}` — junk outside `(0,1)`, honest on it.
- `monotoneOn_quantile` on `Ioo 0 1` (port of `stdNormalQuantile_monotoneOn`;
  `csSup_le_csSup` + nonempty/bddAbove from `tendsto_cdf_atBot/atTop`).
- no-flat-step bridge `sSup {cdf < q} = sInf {q < cdf}` under `StrictMono (cdf μ)` (port
  of `stdNormalCDF_sSup_lt_eq_sInf_gt`; the two-midpoints trick) — this is ALSO the
  sSup-vs-sInf convention equivalence, so it belongs regardless of which def wins.
- inversion pair (ports of `stdNormalCDF_quantile` / `stdNormalQuantile_cdf`):
  `cdf μ (quantile μ p) = p` on `Ioo 0 1` (needs right-continuity + NoAtoms) and
  `quantile μ (cdf μ s) = s` (needs StrictMono; `csSup_Iio`).
- Gaussian oddness `quantile (gaussianReal 0 1) (1−q) = −quantile _ q` (port of
  `stdNormalQuantile_anti`; `Real.sSup_neg`) — this one may live in the Gaussian file.

## PR 4 — Cameron–Martin shift identity

- 1-D (`Gaussian/Real.lean`): port of `integral_gaussianReal_shift_eq` —
  `∫ g(s+d) dN(μ,v) = ∫ exp(likelihood ratio)·g dN(μ,v)`; our proof is 10 lines via
  `gaussianReal_map_add_const` + `integral_gaussianReal_eq_integral_smul` + the pdf ratio
  (`gaussianPDFReal_shift` port — generalize to mean μ, variance v; the exponent becomes
  `(d·(s−μ) − d²/2)/v`-shaped).
- n-D (`Gaussian/Multivariate.lean`): port of `pi_gaussian_shift_eq`, better stated for
  `stdGaussian E` and any shift vector `v : E` (rotate coordinate-0 statement to general
  `v` using the same adapted-ONB trick we used in `stdGaussian_np_shift`, or state at
  measure level: `(stdGaussian E).map (· + v)` has density `exp(⟪x,v⟫ − ‖v‖²/2)` wrt
  `stdGaussian E`). Reviewers may know Cameron–Martin as the infinite-dim theorem — flag
  that this is the finite-dim baby case and name accordingly
  (`integral_stdGaussian_add_eq` or similar, not `cameronMartin`).

## PR 5 — the Gaussian Neyman–Pearson bound

The mathematically substantial one; possibly split. Port of `gaussian_np_shift` (1-D) and
`stdGaussian_np_shift` (n-D):

> `f : E → ℝ` measurable, `0 ≤ f ≤ 1`, `∫ f d(stdGaussian E) ≥ cdf Φ t` ⇒
> `∫ f(· + δ) d(stdGaussian E) ≥ cdf Φ (t − ‖δ‖)`.

Framing options for review: (a) as-is (Gaussian shift bound / "Gaussian halfspaces are
extremal", isoperimetry-adjacent); (b) generalize the core to an abstract
monotone-likelihood-ratio Neyman–Pearson lemma (our pointwise trick
`(f − 1_{z≤t})·(LR − LR(t)) ≥ 0` is verbatim the general proof) with the Gaussian as the
instance — more work, more lasting. Ask on Zulip; offer (a) with (b) as follow-up.
Dependencies: PRs 1–4. Needed Mathlib pieces all exist (`measurePreserving_piFinSuccAbove`,
`integral_prod_symm`, `Integrable.comp_fst`, `measurePreserving_eval`,
`stdGaussian_eq_map_pi_orthonormalBasis`, `Submodule.reflection_sub`,
`OrthonormalBasis.map`, `integrable_exp_mul_gaussianReal`).

**Stays downstream (do NOT PR):** `smoothing_probit_lipschitz`,
`smoothing_certified_radius_*`, `LipschitzL2`, anything mentioning classifiers — Cohen-
certificate framing belongs in our repo. After PR 5 lands, our file shrinks to essentially
just those.

---

## Adaptation checklist (applies to every PR)

- [ ] Drop the `Proofs` namespace and the `stdNormalCDF`/`stdNormalQuantile` wrappers —
      state directly about `cdf (gaussianReal μ v)` / general `μ : Measure ℝ`.
- [ ] Generalize gratuitous specializations: mean 0 → μ, variance 1 → v ≠ 0 where the
      proof doesn't care; `Fin (n+1)` → the general statement (for the n-D NP, quantify
      over a nonempty index or state on any finite-dim inner-product space via
      `stdGaussian E` — our transfer proof already goes through `E` abstractly except for
      choosing the basis index).
- [ ] Mathlib naming: conclusion-first (`strictMono_cdf`, `cdf_gaussianReal_neg`,
      `quantile_cdf`, `cdf_quantile`); no `std` prefixes.
- [ ] Docstrings on every public decl; module docstring per file section.
- [ ] `lake exe lint-style` + build the touched modules; no `set_option` hacks left.
- [ ] PR description: motivation (randomized-smoothing formalization), AI-assistance
      disclosure, link the downstream repo file as provenance.

## Effort estimate

PR 1 + 2: a day of adaptation, then review rounds. PR 4: a day. PR 3: small code, but the
design thread is the calendar cost (weeks). PR 5: 2–3 days if framing (a), more for (b).
Total: few days of work spread over 1–3 months of review latency. Ship PR 1 immediately
after the Zulip thread — a merged small PR builds reviewer trust for the rest.
