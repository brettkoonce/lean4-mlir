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

1. **Two-sided quantile inverse — DONE 2026-07-12 (uncommitted).**
   `SmoothingGaussian.lean` now packages the whole inverse:
   `stdNormalQuantile_strictMonoOn` (STRICT on `(0,1)`, strictness reflected
   through `Φ` via `stdNormalCDF_quantile`), `stdNormalQuantile_surjOn`
   (`Φ⁻¹` maps `(0,1)` ONTO ℝ — every `s` is `Φ⁻¹(Φ s)`),
   `stdNormalQuantile_continuousAt`/`_continuousOn`. Continuity fell to
   `StrictMonoOn.continuousAt_of_image_mem_nhds` + the full-image trick — the
   branch was NOT needed for it (no `continuous_cdf`, no OrderIso). The branch
   files are still lifted into the build (`UpstreamDraft` = `Certs` root +
   audited, 6 apex `#print axioms` lines) so the PR drafts can't rot; the
   bespoke-proof swap (cosmetic) was skipped. Landed as a single commit on
   main (Brett's call: fold it in, the branch never reached PR stage);
   `origin/mathlib-upstream-pr1-pr2` deleted, content fully subsumed.

2. **Exact Clopper–Pearson instead of Hoeffding — DONE 2026-07-12
   (`LeanMlir/Proofs/SmoothingCP.lean`, Certs root, 5 audited, 3-axiom clean).**
   - `pi_hitCount_eq_binomial`: the count of successes over `Measure.pi` IS
     binomial — the lemma Mathlib genuinely lacks (surveyed: `Bin(n,p)` exists
     as `setBer(Iio n,p).map ncard`, no iid-indicator-count law, binomial
     mean/variance still `proof_wanted`). Induction on `N` via
     `measurePreserving_piFinSuccAbove` at coordinate 0 + `Measure.prod_apply`
     Fubini + `lintegral_add_compl` split + Pascal. `hitCount` uses
     `Set.indicator` (NOT ite — no `Decidable` instance exists for `∈ A` on
     this pin, `open scoped Classical` no longer provides it).
   - `cpLower α N k = sInf {q ∈ [0,1] | α < binomTail N k q}`; coverage
     (`cp_coverage`, ≥ 1−α) needs NO tail-monotonicity-in-q and NO
     binomial-sums-to-1: minimal counterexample count `k₀` (`Nat.find`) has
     `binomTail N k₀ p ≤ α` by contrapositive of `csInf_le`, counts below `k₀`
     certify by minimality. (The predicted "CP bound is a monotonicity
     statement" turned out unnecessary — monotonicity only matters for tying
     to a SOLVED-equation form of the bound, i.e. move 3.)
   - `smoothing_cp_certified`: radius `σ·Φ⁻¹(cpLower α N k)` from the observed
     count, certified with prob ≥ 1−α. Guarantee AND arithmetic now match
     CERTIFY. (Neither PR1/PR2 nor the quantile continuity was needed here;
     they wait for move 3's solved-form tie.)

3. **Tie the `smoothCertify` DRIVER'S reported radii** (`f30f857`,
   `mnist-{mlp,cnn}-smooth`, `cifar-smooth` exes).
   **3a SOLVED-FORM MACHINERY DONE 2026-07-12 (SmoothingCP.lean §solved):**
   `binomTail_monotoneOn` — by COUPLING (tail law at uniform-[0,1] with
   nested `[0,q] ⊆ [0,p]`), no calculus, no PR1/PR2 needed after all;
   `le_cpLower_of_tail_le` — ONE kernel check `binomTail N k₀ q₀ ≤ α`
   certifies the driver's reported `q₀` (needs `binomTail_one` + set
   nonempty via q=1); `smoothing_cp_certified_solved` — the per-image
   scorecard theorem: "if the count comes out k₀, radius σ·Φ⁻¹(q₀) is
   certified" w.p. ≥ 1−α, numeric hypothesis = pure kernel arithmetic.
   **3b demo checks DONE**: 99/100 @ q₀=0.9 (decide + norm_num) and
   999/1000 @ q₀=0.985. GOTCHAS: `norm_num` pow silently no-ops above
   `exponentiation.threshold` (default 256 — raise via set_option; also
   `maxRecDepth`); `decide` on `Nat.choose N (N-1)` is fine (linear
   recursion) but `Nat.choose N j` for middling `j` blows up — use
   `Nat.choose_symm` / ladder `Nat.choose_succ_right_eq` literals.
   **3c-ENGINE DONE 2026-07-12 (SmoothingCP.lean §kernel)**: the norm_num
   route priced out at driver scale, so the ListDot recipe instead —
   `binomTailNum` (kernel-computable ℕ numerator; binomials via
   `descFactorial/factorial` on the SMALL side of the tail, exact by
   `Nat.choose_eq_descFactorial_div_factorial`, no Pascal blowup) + the
   once-proven bridge (`binomTailNum_eq` via `sum_range_reflect` +
   `choose_symm`; `binomTail_eq_kernel` cast/div) ⇒
   `binomTail_le_of_kernel_check`: per-image hypothesis = ONE
   `A * binomTailNum N k a d ≤ d ^ N` by `decide +kernel` (kernel
   Nat.pow/mul are GMP-accelerated — VALIDATED at the deployed N=10112:
   213-term tail AND 4613-term deep tail each ~0.1 s, whole file 4 s).
   The choose-ladder/power-sharing generator tricks are OBSOLETE.
   **3c-data DONE 2026-07-12**: driver patched (per-image `count,n` CSV
   columns + `SMOOTH_SIGMA_MILLI`/`SMOOTH_STRIDE` knobs on the generic
   `smoothCertify`), fixed-protocol runs via `run_smooth_scorecard.sh`
   (first-100 test images, σ=0.5, n=10112, α=0.001, 2 GPUs, ~25 min;
   noise-trained clean accs 97.3/98.0/54.5, ACR 1.15/1.32/0.39) →
   `scripts/smooth_scorecard_gen.py` (binary-searches the LARGEST 4-decimal
   `q₀ = a/10000` per certified image, verifies each in exact integer
   arithmetic — incremental-term tail, ~50 s for all 279) →
   `LeanMlir/Proofs/SmoothingCPScorecard.lean`: **MNIST-MLP 99/100,
   MNIST-CNN 100/100, CIFAR-CNN 80/100 certified**, one
   `binomTail_le_of_kernel_check` + `decide +kernel` lemma per image +
   per-net `∀ e ∈ entries` aggregates (Lipschitz-scorecard idiom). Landed
   in `Certs` (light: pure kernel bignum, no norm_num megaterms).
   **CERTIFIED DECIMAL RADII — CORE DONE 2026-07-12
   (`LeanMlir/Proofs/SmoothingPhiBounds.lean`)**: the float-Φ⁻¹ informality
   is closed at the machinery level. `stdNormalCDF_panel` (one upper-Riemann
   panel via `gaussianReal_apply` + `setLIntegral` — no FTC/improper
   integrals), kernel-computable rational pdf upper bound (`ratExpLB` 32-term
   Taylor + `√2π ≥ 2.5066282` from `pi_gt_d20` + `ratCeil9` rounding so grid
   denominators stay bounded — WITHOUT the rounding the exact-ℚ fold's
   denominators explode astronomically), grid fold `phiGridUB` +
   `le_stdNormalQuantile_of_grid`: ONE `decide +kernel` check certifies
   `m·h ≤ Φ⁻¹(q₀)`. Demos: `Φ⁻¹(0.9) ≥ 1.27` (true 1.2816, m=635 ~13s),
   `Φ⁻¹(0.9952) ≥ 2.54` (true 2.590, m=1270 ~29s), MLP-img1 radius ≥ 1.27
   in decimals (driver printout 1.295). Slack ≈ h·(φ(0)−φ(t)): ~0.001 in Φ
   at h=1/500. GOTCHAS: a `decide` that reports "stuck at ...blt" can just
   mean the inequality is FALSE (misleading error); equation-compiler
   recursion on ℕ kernel-reduces fine to m≥1270 (no brecOn blowup).
   **PREFIX-SCAN CORPUS DECIMALS DONE 2026-07-12**: `phiScanRev h n`
   (SmoothingPhiBounds.lean §prefix-scan) = the DESCENDING scan
   `[phiGridUB h n, …, phiGridUB h 0]`, built by reusing the previous head
   through a `match` (the two uses of the head stay pointer-shared, so the
   kernel's whnf cache makes ONE evaluation O(n) — measured 81 s for the
   full h=1/1000, 3300-panel scan; a `let` would NOT share: zeta-substitution
   duplicates the thunk). `phiScanRev_getD` (entry `n−m` is `phiGridUB h m`)
   + `le_stdNormalQuantile_of_scan` + protocol form `smooth_radius_dec`
   (σ=1/2, h=1/1000: `m/2000 ≤ σ·Φ⁻¹(a/10000)` from one `L.getD` lookup).
   Generated `LeanMlir/Proofs/SmoothingDecScorecard.lean`
   (scripts/smooth_dec_scorecard_gen.py: exact-ℚ Python mirror of the
   ratExpLB/ratPdfUB/ratCeil9 fold — untrusted, the emitted
   `phiScanLit_eq : phiScanRev (1/1000) 3300 = <literal>` re-verifies it in
   ONE `decide +kernel`): ALL 279 CP-scorecard images get proved decimal
   radii at grid-optimal m. Measured slack vs driver float: avg 0.012/0.019/
   0.003, max 0.036 (the q₀=0.9993 images: 1.561 proved vs 1.597 float) —
   better than the ~0.08σ feared; the Mills-ratio barrier is NOT needed.
   Certified ACR: MLP 1.151, CNN 1.297, CIFAR 0.591 (certified images).
   Gotchas: `set_option … in` must precede the docstring; a flat 3301-element
   `List ℚ` DIVISION literal never finishes elaborating (>10 min even at 400M
   heartbeats — per-numeral `OfNat ℚ` pending-mvar synthesis) — emit the
   literal as `List ℕ` NUMERATORS over the common denominator `10¹²` (every
   grid value is `1/2 + Σ (1/1000)·(k/10⁹)`) plus one `map (·/10¹²)`: ℕ
   literals are kernel-primitive and elaborate in ~1 s.
   **CI-MEMORY RETROFIT 2026-07-13** (the one whole-grid decide peaked
   15.1 GB RSS → OOM'd the blueprint runner's `lake build LeanMlir Certs`,
   16 GB; Certify Corpus survived the same build by luck): the scan is now
   verified in SIX 550-panel chunks — `phiScanRevFrom h k v j` (scan
   continued from a checkpoint value) + glue `phiScanRevFrom_append` in
   SmoothingPhiBounds, then per chunk: checkpoint literal read off the
   verified prefix via `phiScanRev_headI`, one `decide +kernel` from that
   literal, one `rw`-chain glue lemma. TWO more memory facts discovered:
   (a) kernel memory is NOT reclaimed between declarations inside one lean
   process (6 chunks in one file still peaked 12.9 GB) → each chunk lives in
   its OWN MODULE (`SmoothingDecChunk1–6`, import-chained, ~5.2 GB / ~18 s
   each); (b) 279 per-image `getD` decides against the FULL literal
   accumulated 11.3 GB in the scorecard module → each per-image check now
   targets the SMALLEST verified prefix covering its m (`phiScanEq{550,…}`;
   `getD` at index < 551 never forces the appended tail) → main module
   8.5 GB / 29 s. Worst module 8.5 GB, under the ViTBackB0 14 GB precedent.
   The LAST remaining informality: the net-semantics hypothesis
   (C = the rendered fwd's argmax + `hp` interiority) — noted in the
   scorecard header.

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
