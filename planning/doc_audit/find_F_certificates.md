# Slice F — LeanMlir/Proofs/Certificates/ (documentation audit)

**Coverage.** Read in full (every docstring against its statement): LipschitzCert, LipschitzCertPairSDP,
DenseEuclid, IntervalBound, CrownBound, GaussianQuantile, SmoothingGaussian, SmoothingMC, SmoothingCP,
SmoothingNetSemantics, SmoothingPhiBounds, LipschitzCertInstance (hand-written), plus the hand-shaped engine
part of LipschitzCertFloat and all of SmoothingNetWitness. For the 22 generated files: read every module
docstring, deduplicated every declaration docstring into its template and checked the templates against
sample statements (scorecard aggregates, per-image theorems), then located the offending text in the generator.
Checked: no `sorry` / `axiom` / `admit` / `native_decide` / `implemented_by` / `@[extern]` anywhere in the slice.
So the "3-axiom clean" claims hold as far as source inspection can show. I did not run `#print axioms`.
Naming pass: no docstring in the slice cites a pre-7dbe9c92 spelling. Every backticked Lean name resolves, except
the Mathlib names and the variables.

---

### LeanMlir/Proofs/Certificates/SmoothingGaussian.lean:453 — `smoothing_certified_radius_classifier` (same text at :45–46 module doc, and LipschitzCert.lean:202–203 `smoothing_certified_radius_probit`)

**Kind:** overclaim
**Says:** "non-degenerate class probabilities (`hp` — Φ⁻¹ needs `(0,1)`; Monte-Carlo estimates always satisfy this)". The module doc says "`hp : p c y ∈ Ioo 0 1` — the realistic regime, since Monte-Carlo/Clopper–Pearson class-probability estimates are never exactly 0 or 1". LipschitzCert.lean says "Monte-Carlo/Clopper–Pearson estimates are never exactly 0 or 1".
**Actually states:** `hp : ∀ c x, (∫ z, if C (x + σ • z) = c then 1 else 0 ∂γ) ∈ Ioo 0 1`. This is a hypothesis on the TRUE smoothed probability of EVERY class at EVERY input point. Estimates play no part in it. The justification is also false on its own terms: the repo's own SmoothingCPScorecard.lean `smooth_cp_mlp_i0` has count 10112/10112, a Monte-Carlo estimate of exactly 1. `hp` is a real hypothesis that says every decision region is non-null and non-conull. It is discharged only for an argmax net with per-class witnesses (`argmaxNet_smoothProb_mem_Ioo`, SmoothingNetSemantics.lean).
**Fix:** "`hp` — every class's smoothed probability lies in `(0,1)` at every point, i.e. no decision region is Gaussian-null or conull (Φ⁻¹ is only honest on `(0,1)`). For an argmax net it follows from one strict-argmax witness per class (`argmaxNet_smoothProb_mem_Ioo`)." Delete the Monte-Carlo sentence in all three places.

### LeanMlir/Proofs/Certificates/SmoothingGaussian.lean:11–15 — module docstring

**Kind:** overclaim
**Says:** "for a measurable classifier under `N(0,σ²I)` smoothing, every `‖δ‖ < σ·Φ⁻¹(p_A(x))` provably cannot flip the smoothed argmax, with `Φ⁻¹` the genuine standard-normal quantile and NO smoothing-side hypotheses left".
**Actually states:** `smoothing_certified_radius_classifier` still takes `hp` (interiority for all classes, all points) as a hypothesis, as do `smoothing_certified_of_le`, `smoothing_mc_certified` and `smoothing_cp_certified`.
**Fix:** "…with `Φ⁻¹` the genuine standard-normal quantile. The Neyman–Pearson `(1/σ)`-Lipschitz probit is now a theorem (`smoothing_probit_lipschitz`). The one remaining hypothesis is `hp`, that every class probability lies in `(0,1)`, which SmoothingNetSemantics.lean discharges for argmax nets."

### scripts/certs/lipschitz_cert_scorecard_ibp.py:206 — emitted module docstring of LipschitzCertScorecardIBP.lean / …IBPUncon.lean

**Kind:** overclaim
**Says:** "at ε = 1/255, 2/255, 4/255, 8/255 the box certificate proves **92/100**, **88/100**, **69/100**, **24/100** predictions robust".
**Actually states:** the aggregate theorem ties only the 8-per-radius emitted witnesses (32 / 18 per-image `CertifiedAtLinf` theorems). The same header's next paragraph says the counts "are exact-rational MEASUREMENTS", so the lead sentence contradicts it.
**Fix:** "the box certificate certifies (measured by exact rational interval propagation) **92/100**, … — see *Theorem vs. measurement* below for which of these carry Lean proofs."

### scripts/certs/lipschitz_cert_scorecard.py:436 — emitted docstring of `scorecard` (LipschitzCertScorecard.lean)

**Kind:** overclaim
**Says:** "**The scorecard, as a theorem**: at ε = 1/10 (pooled L2) the capped net certifies 34/100 of the fixed test subset and the unconstrained net 1/100".
**Actually states:** `(cappedCerts.length = 8 ∧ ∀ p ∈ cappedCerts, CertifiedAt mlpS (1/10) …) ∧ (unconCerts.length = 1 ∧ …)`. That is 8 and 1 certified witnesses. The 34 is a comment-line measurement. The later "Those are MEASUREMENTS" does not undo a bold lead that calls 34/100 a theorem. The witnesses are also not "BELOW": they appear in the statement and are defined above it.
**Fix:** "**The proved core of the scorecard**: the 8 capped-net and 1 unconstrained-net witnesses in `cappedCerts`/`unconCerts` each carry a `CertifiedAt … (1/10)` proof. The dataset counts (34/100 capped, 1/100 unconstrained) are exact-rational measurements recorded in the `certMargin*` lines above, not theorems. Lower bounds only." (This matches the wording lipschitz_cert_scorecard_full.py already uses.)

### scripts/certs/ibp_conv_scorecard.py:404 (also docstring :21–23) — emitted module docstring of IbpConvScorecardNet.lean

**Kind:** overclaim
**Says:** "the QUANTIZED net is what is certified, so the certified network is the deployed one". The script docstring adds "not a nearby real-valued idealization".
**Actually states:** `CertifiedAtLinf3 net ε x y` over the ℝ-semantics `net` (the cast of the k/256 data). The deployed forward is float-evaluated. The dense tier needed a separate LipschitzCertFloat.lean to close exactly this gap, and no float composition exists for the conv net.
**Fix:** "the QUANTIZED weights are what is certified (exact-ℝ semantics of the k/256 net); the float-evaluated forward is not covered (no FloatBridge composition exists for this net, unlike the dense `LipschitzCertFloat.lean`)."

### scripts/certs/lipschitz_cert_scorecard_full.py:426–428 — emitted module docstring of LipschitzCertScorecardFull.lean

**Kind:** overclaim (wrong direction)
**Says:** "ε here is FULL-pixel-space L2 (pixels in [0,1]), not the pooled-feature L2 of `LipschitzCertScorecard.lean` — a strictly stronger, directly comparable-to-the-literature perturbation model."
**Actually states:** 4×4 average pooling has operator norm 1/4 in L2, so a pixel ball of radius ε maps into a pooled ball of radius ε/4. A pooled-L2 certificate at ε therefore covers raw-pixel perturbations up to 4ε. At equal ε, the pixel-L2 model is the WEAKER requirement, not the stronger one. The older header's own gloss agrees: pooled ε = 0.1 "concentrated on one block" is 0.4 in pixel L2.
**Fix:** "ε here is FULL-pixel-space L2 (pixels in [0,1]), the perturbation model the literature reports. It is not the pooled-feature L2 of `LipschitzCertScorecard.lean`: a pooled-L2 radius ε corresponds to a raw-pixel radius up to 4ε, so the two tiers' ε are not directly comparable."

### LeanMlir/Proofs/Certificates/LipschitzCert.lean:207–210 — `smoothing_certified_radius_probit`

**Kind:** stale
**Says:** "`SmoothingGaussian.lean` discharges both conditions at the real `Φ⁻¹`, leaving only the Neyman–Pearson Lipschitz core `hg` as a hypothesis."
**Actually states:** SmoothingGaussian.lean also proves `hg`: `smoothing_probit_lipschitz` / `smoothing_certified_radius_cohen`, whose docstring reads "No Lipschitz hypothesis".
**Fix:** "`SmoothingGaussian.lean` discharges both conditions at the real `Φ⁻¹` and proves `hg` for Gaussian-smoothed scores (`smoothing_probit_lipschitz`); `smoothing_certified_radius_classifier` is the hypothesis-free-but-`hp` form."

### LeanMlir/Proofs/Certificates/SmoothingGaussian.lean:88–91 — `smoothing_certified_radius_gaussian`

**Kind:** stale
**Says:** "(`hg` — the Neyman–Pearson core, the ONE remaining smoothing-side hypothesis, G2–G4 of `planning/archive/smoothing_gaussian_lemma.md`)".
**Actually states:** the G2–G4 plan is complete in this same file (`smoothing_probit_lipschitz`). `hg` remains a hypothesis only because this lemma abstracts over `p`.
**Fix:** "(`hg` — `(1/σ)`-Lipschitz probit scores; for Gaussian-smoothed `[0,1]` scores this is `smoothing_probit_lipschitz`, applied in `smoothing_certified_radius_cohen`)".

### LeanMlir/Proofs/Certificates/SmoothingGaussian.lean:17–41 — module docstring

**Kind:** stale
**Says:** "This file proves the three facts Mathlib doesn't have: strict monotonicity (`stdNormalCDF_strictMono`) … symmetry (`stdNormalCDF_neg`) … `stdNormalQuantile_monotoneOn` … `stdNormalQuantile_anti` … The two-sided inverse is fully packaged: … (`stdNormalCDF_quantile`), … (`stdNormalQuantile_cdf`) … `stdNormalQuantile_strictMonoOn` … `_continuousOn` …". It also says "G2 … also lives here: `stdNormalCDF_quantile` upgrades the quantile…".
**Actually states:** none of these are declared in this file. All live in `GaussianQuantile.lean` (which this file imports). This file holds `gaussianPDFReal_shift`, the G3 shift/NP theorems (`pi_gaussian_np_shift`, `stdGaussian_np_shift`), G4 (`smoothing_probit_lipschitz`), and the radius capstones.
**Fix:** replace the first half with "`Φ`, `Φ⁻¹` and their monotonicity, symmetry, two-sided inversion and continuity live in `GaussianQuantile.lean`. This file proves the Neyman–Pearson core (`pi_gaussian_np_shift`, the n-D `stdGaussian_np_shift`), the `(1/σ)`-Lipschitz probit (`smoothing_probit_lipschitz`), and the Cohen radius at the real quantile (`smoothing_certified_radius_gaussian` / `_cohen` / `_classifier`, `smoothing_certified_of_le`)."

### LeanMlir/Proofs/Certificates/SmoothingNetSemantics.lean:17–20 — module docstring

**Kind:** stale
**Says:** "* `stdGaussian` full support — `IsOpenPosMeasure` instances for `gaussianReal 0 1` … and for the multivariate `stdGaussian E` …" (listed among this file's contents).
**Actually states:** both instances are declared in `GaussianQuantile.lean` (`instance : (gaussianReal 0 1).IsOpenPosMeasure`, `stdGaussian.instIsOpenPosMeasure`). This file only uses them.
**Fix:** "* full support of `stdGaussian` (the `IsOpenPosMeasure` instances of `GaussianQuantile.lean`) gives the witness regions positive mass;"

### LeanMlir/Proofs/Certificates/LipschitzCert.lean:16–17 — module docstring

**Kind:** stale
**Says:** "The `L` is supplied numerically by `specNormW` / `specNormConvTapSum` ([`LeanMlir/VerifiedTrain.lean`](…))".
**Actually states:** both are `private def`s in `LeanMlir/VerifiedAttack.lean` (lines 35, 104). Neither name appears in VerifiedTrain.lean.
**Fix:** point the link at `LeanMlir/VerifiedAttack.lean`.

### LeanMlir/Proofs/Certificates/LipschitzCertInstance.lean:11–21 — module docstring

**Kind:** stale / missing
**Says:** "Two instances: * `linear_demo_certified` … * `mlp_demo_certified` …"
**Actually states:** the file has a third section that is its main content. It covers the trained 49→8→10 MLP `mlpT`: `trained_demo_certified` (Frobenius), `trained_demo_certified_gram` (Schatten-4), `trained_demo_certified_gram2` (Schatten-8), the certified lower bounds `W1t_lip_lower` / `W2t_lip_lower`, and `mlpT_logit_continuous`. Five downstream files import it for these. The opener "Closes the 'certificate machinery, never instantiated' gap" is also process narrative.
**Fix:** add "* `trained_demo_certified{,_gram,_gram2}` — the trained 49→8→10 MLP `mlpT` at a pooled MNIST digit, with Frobenius / Schatten-4 / Schatten-8 product constants and certified lower bounds `W1t_lip_lower`/`W2t_lip_lower` sandwiching each layer's `‖W‖₂`; `mlpT` and its weights are what the scorecards and `SmoothingNetWitness.lean` build on." Drop the "Closes the … gap" sentence.

### LeanMlir/Proofs/Certificates/SmoothingPhiBounds.lean:88–89, 146–148 — `ratExpLB`, `ratCeil9`

**Kind:** stale
**Says:** "32 terms: relative error < e⁻²⁴ at `x = 5.12`, our largest use" and "the exact `ratPdfUB` values have ~190-digit numerators whose lcm across 640 grid points is astronomical".
**Actually states:** the only instance (SmoothingDecScorecard / Chunk1–6) runs `h = 1/1000` over 3300 panels. The largest `ratExpLB` argument is therefore `(3.299)²/2 ≈ 5.44`, not 5.12, and the grid has 3300 points, not 640.
**Fix:** "…at `x ≈ 5.44` (`a = 3.3`, the 3300-panel grid's last point)" and "…across the 3300 grid points…". Recompute the error figure at 5.44.

### LeanMlir/Proofs/Certificates/IntervalBound.lean:60 — `CertifiedAtLinf`

**Kind:** stale
**Says:** "The `L∞` peer of `CertifiedAt` (`LipschitzCertScorecard.lean`)."
**Actually states:** `CertifiedAt` is defined in `DenseEuclid.lean:257`.
**Fix:** "(`DenseEuclid.lean`)".

### LeanMlir/Proofs/Certificates/CrownBound.lean:48–53 — module docstring

**Kind:** stale (inconsistent with the emitted instance)
**Says:** "at `k = 8` the rounding costs ZERO images … so the coefficients stay at the same `/256` scale as the weights themselves."
**Actually states:** the slope is rounded to `/2^8`, but each coefficient `a = v·s` is a weight difference times a slope. The generated LipschitzCertScorecardCrown.lean header says so: "keeps coefficients at `/65536` and `A` at `/16777216`".
**Fix:** "…so the slope sits on the weights' `/256` grid and the coefficients at `/65536` (`A` at `/16777216`), instead of carrying the layer-1 denominators."

### LeanMlir/Proofs/Certificates/IntervalBound.lean:9–13 — module docstring

**Kind:** overclaim (mild)
**Says:** "this is the certificate that scales past the h=16 wall to canonical widths."
**Actually states:** the engine is width-generic, but every IBP instance in the repo is width 16 (784→16→10) or the 4-channel conv net. No canonical-width instance exists.
**Fix:** "…IBP is linear in width, so it is the tier that could scale past the h = 16 wall. Every instance here is still at h = 16."

### LeanMlir/Proofs/Certificates/DenseEuclid.lean:192–197 — `denseE_lipschitzL2_gram`

**Kind:** wrong (prose-level math)
**Says:** "`‖W‖₂ ≤ (Σσᵢ⁴)^¼` — strictly tighter than Frobenius `(Σσᵢ²)^½` whenever the spectrum has any spread."
**Actually states:** `(Σσ⁴)^¼ < (Σσ²)^½` holds exactly when at least two σᵢ are nonzero, spread or not. With all σᵢ equal and rank ≥ 2 it is still strict.
**Fix:** "…strictly tighter than Frobenius whenever `W` has rank ≥ 2."

### Process narrative in module docstrings (one finding, several sites)

**Kind:** process-narrative
**Says (examples):**
- LipschitzCertScorecard.lean title "(post_audit_roadmap §1)" and "the roadmap's caps 1.5–2 …" (scripts/certs/lipschitz_cert_scorecard.py:177, :202)
- LipschitzCertFloat.lean "The 2026-07-02 audit's gap #1, closed" (lipschitz_cert_float.py:149)
- LipschitzCertScorecardFull.lean "(the 2026-07 audit's gap #3)" (lipschitz_cert_scorecard_full.py)
- SmoothingGaussian.lean "The complete G1–G4 ladder of `planning/archive/…`"; ":118–119 The plan's riskiest item … turned out to ship with Mathlib"
- SmoothingMC.lean "the honest gap flagged since the smoothing theorems landed. This file closes it"
- SmoothingNetSemantics.lean "the last informality flagged in the scorecard headers. This file closes it"
- LipschitzCertInstance.lean "Closes the 'certificate machinery, never instantiated' gap"; section comment "Generated by scripts (see planning)" (names no script)
**Actually states:** none of this says what the material is for. It records project history and dates badly.
**Fix:** replace each with the mathematical role, e.g. "Composes the per-image Lipschitz certificates with the FloatBridge forward budgets, certifying the float-evaluated capped net." Name the actual generator script where "(see planning)" appears, or drop the phrase.

---

## Overclaims (fix before the results are quoted again)

1. **SmoothingGaussian.lean:453 / :45 and LipschitzCert.lean:203.** Justify `hp` with "Monte-Carlo estimates always satisfy this / are never exactly 0 or 1". `hp` concerns the true probabilities of every class at every point, and the repo's own CP scorecard has a count of 10112/10112.
2. **SmoothingGaussian.lean:13.** "NO smoothing-side hypotheses left", while `hp` is still a hypothesis of the classifier theorem and everything built on it.
3. **scripts/certs/lipschitz_cert_scorecard_ibp.py:206.** "the box certificate **proves** 92/100 … predictions robust". These are measurements; 8 per radius are proved.
4. **scripts/certs/lipschitz_cert_scorecard.py:436.** "**The scorecard, as a theorem** … certifies 34/100". The theorem proves 8 and 1 witnesses.
5. **scripts/certs/ibp_conv_scorecard.py:404, :23.** "the certified network is the deployed one". The certificate covers the ℝ semantics of the quantized weights, not the float forward.
6. **scripts/certs/lipschitz_cert_scorecard_full.py:427.** Full-pixel L2 called "a strictly stronger" perturbation model than pooled L2. At equal ε it is the weaker one, since pooling has L2 norm 1/4.
7. **IntervalBound.lean:11–13 (mild).** IBP "scales past the h=16 wall to canonical widths". No canonical-width instance exists.
