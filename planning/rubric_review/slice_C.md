# Slice C — Certificates: rubric review 2026-09-30

Scope read in full: the hand-written engines `LipschitzCert/{Basic,Instance,PairSDP}`, `DenseEuclid`,
`IntervalBound`, `CrownBound`, `GaussianQuantile`, `Smoothing/{Gaussian,MC,CP,NetSemantics,PhiBounds}`;
the engine halves of the generated `LipschitzCert/Float`, `Smoothing/NetWitness`; headers + aggregate
statements of every generated scorecard; `Foundation/IntervalBoundConv` (the conv-IBP engine);
`planning/mathlib_upstream_drafts/PR{1,2}` against `Foundation/UpstreamDraft.lean` and the pinned
Mathlib; the `Certs`/`CertsHeavy` lakefile blocks, `tests/AuditAxiomsHeavy.lean`, `certs-heavy.yml`,
`formalization.yaml` certificate rows, `TRUST.md`. History checked: `doc_audit/find_F_certificates.md`
+ `ledger_F.md` (landed), `audit_v2_rubrics/slice_notes.md` (certificates), `api_design_audit.md` §6.2–6.5.
Three claims were typechecked against the main checkout's oleans (scratch in `scratchpad/C/`).

## Verdicts
| angle | verdict | findings |
|---|---|---|
| correctness | request_changes | 1 |
| reuse | block (rubric-mandated; the blocking item is a one-line fix) | 2 |
| scope | approve | 0 |
| attribution | request_changes | 1 |
| api-design | approve | 0 |
| generality | approve | 0 |
| placement | request_changes | 3 |
| naming | request_changes | 2 |
| documentation | request_changes | 4 |
| proof-quality | request_changes | 1 |

**Answer to the brief's shipped-network question.** The kernel-checked theorems are all true and
non-vacuous. What they certify is narrower than several prose sites say:
- **L2 Lipschitz / LipSDP / IBP / CROWN:** these cover purpose-trained, quantized, bias-free nets
  (`mlpT`, `mlpS` pooled 49→8→10; `mlpSF`/`mlpTF` 784→16→10). They are certified under exact-ℝ
  semantics over *all* δ in the ball; the certificate does not clip to [0,1], which makes it
  stronger. The float forward is covered only for pooled `mlpS` (`LipschitzCert/Float`), and only
  modulo the rounding-model trust boundary.
- **Conv IBP:** this covers the quantized net under ℝ semantics, with no float composition.
- **Smoothing:** no shipped driver checkpoint is certified. The scorecards prove tail arithmetic
  and decimal quantile bounds. Only `mlpT` gets the full net-level chain, and no Monte-Carlo count
  was ever taken on `mlpT`.
- **The `mnist-*-pgd` demos' "certified-robust acc":** this is not an instance of any theorem
  (C-cor-1).

The 09-24 finding ("69/100, 92/100 quoted as proved when 8 are") is fixed in every generated Lean
header, which now say MEASURED. It survives in `formalization.yaml`, the lakefile, the CertsHeavy CI
summary and `tests/AuditAxiomsHeavy.lean`, and a new variant appears in the smoothing CP/decimal
aggregates (C-doc-1, C-doc-2, C-nam-1).

## Findings

### correctness
- **C-cor-1** `LeanMlir/Proofs/Certificates/LipschitzCert/Basic.lean:5-17` (and
  `LeanMlir/Verified/Attack.lean:97-102, 274-279`). The problem is the claim that the demos'
  certified radius is an instance of `lipschitz_margin_certified_radius`.
  - What the docs say: the module doc calls the theorem "the certificate behind the
    `mnist-{linear,mlp,cnn}-pgd` demos … stated as a theorem", with "the `L` … supplied
    numerically by `specNormW` / `specNormConvTapSum`". `Attack.lean:278` links the demo's printed
    `m/(√2·L)` radius to this theorem.
  - Why that is wrong: `specNormW`/`specNormGet` (Attack.lean:35-95) return √(Rayleigh quotient)
    after 60 power iterations. For PSD `WᵀW` that is ≤ σ_max, so the demo's `L` is an
    **under**-estimate. The theorem's hypothesis `hf : LipschitzL2 L f` is therefore not
    established. The printed radius can exceed what the theorem certifies, and the
    "certified-robust acc" lines (Attack.lean:268, :373, :506) are not theorem instances.
  - Two more misstatements: `specNormConvTapSum`'s docstring calls itself "A **sound** (loose)
    upper bound", but its summands are the same power-iteration lower estimates. And the demo nets
    (784→512→512→10, float) are not the nets any certificate is stated about.
  - `DenseEuclid.lean:55` already calls `denseE_lipschitzL2` "the certified replacement for the
    power-iteration estimate", so the repo knows this.
  - **Fix:** the Basic.lean module doc should say:
    > The `*-pgd` demos print an *estimated* radius. `L` there is a power-iteration estimate, a
    > lower bound on each ‖Wᵢ‖₂, so that radius is not an instance of this theorem. The certified
    > instances are `LipschitzCert.Instance` / `Scorecard*`.

    Also drop "sound" from `specNormConvTapSum`, and relabel the demo's `certified-robust acc`
    print as `est. certified acc`. Alternatively, make the demo sound by multiplying the estimate by
    a proven slack, or by using the Frobenius/Gram bound the certificates use.
  - **Evidence:** Attack.lean:56-59 (`lam := Σ v·(WᵀW v)`, `‖v‖=1`, `sqrt lam`); DenseEuclid.lean:225-229
    (`lipschitzL2_lower_euclid` treats the power-iteration vector as a LOWER bound).
  - **Cost:** docstrings plus one print string; 0 pins. Attack.lean is outside the slice, so
    coordinate with its owner. **Size:** S.

### reuse
- **C-reuse-1** `LeanMlir/Proofs/Certificates/CrownBound.lean:249` — private `mlp2_apply` is exactly
  `mlp_out_eq W1 W2 (fun _ => rfl) c` (`DenseEuclid.lean:75`). `PairSDP.lean:166` already uses that
  form. **Fix:** delete `mlp2_apply`; at :311 write `rw [mlp_out_eq W1 W2 (fun _ => rfl),
  mlp_out_eq W1 W2 (fun _ => rfl)]`. **Evidence:** typechecked
  (`scratchpad/C/mlp2.lean`, exit 0). **Cost:** private, 0 pins, −6 lines; gate `lake build
  +LeanMlir.Proofs.Certificates.CrownBound` then `CertsHeavy`'s `ScorecardCrown*` consumers are
  unaffected (they don't cite it). **Size:** S.
- **C-reuse-2** `LeanMlir/Proofs/Certificates/GaussianQuantile.lean:92-136`. The 30-line private
  `stdNormalCDF_sSup_lt_eq_sInf_gt` (the "no flat step" argument) and the set algebra in
  `stdNormalQuantile_anti` re-derive what the file's own `stdNormalCDF_quantile` (:148) and
  `stdNormalCDF_strictMono.injective` already give.
  - **Fix:** move `stdNormalQuantile_anti` below `stdNormalCDF_quantile` and prove it as:
    ```lean
    apply stdNormalCDF_strictMono.injective
    rw [stdNormalCDF_neg, stdNormalCDF_quantile hq,
      stdNormalCDF_quantile ⟨by linarith [hq.2], by linarith [hq.1]⟩]
    ```
    Then delete `stdNormalCDF_sSup_lt_eq_sInf_gt`.
  - **Evidence:** typechecked (`scratchpad/C/anti.lean`, exit 0; the file is byte-identical in
    both checkouts).
  - **Cost:** the name is kept, so the pins (AuditAxioms:1571, Smoothing/Gaussian:87) are
    untouched; about −40 lines. **Size:** S.

### attribution
- **C-att-1** Named methods are followed without credit:
  - `LeanMlir/Proofs/Certificates/IntervalBound.lean:3-32` and
    `LeanMlir/Proofs/Foundation/IntervalBoundConv.lean:1-47` implement interval bound propagation
    by that name ("IBP", "literature-standard") and cite no source. The source is Gowal et al.
    2018, *On the Effectiveness of Interval Bound Propagation…*, arXiv:1810.12715.
  - `CrownBound.lean:9,19` credits CROWN (Zhang et al. 2018), but the method the file states it
    implements is **CROWN-IBP**: Zhang, Chen, Xiao, Gowal, Stanforth, Li, Boning, Hsieh, ICLR 2020,
    arXiv:1906.06316. That source is uncredited.
  - `formalization.yaml` `references:` (:50-83) lists Tsuzuku, Cohen, Salman and LipSDP but has no
    IBP / CROWN / CROWN-IBP / Clopper–Pearson (1934) entries. Its summary (:16) says "three
    robustness certificates", although the IBP (dense and conv) and CROWN families are lib roots
    (`Certs`: IntervalBound, CrownBound, IntervalBoundConv; `CertsHeavy`: ScorecardIBP*,
    ScorecardCrown*, IbpConvScorecard).

  **Fix:** add one citation line to each module docstring (Gowal 2018 in IntervalBound and
  IntervalBoundConv; Zhang 2018 + Zhang 2020 in CrownBound; optionally "Clopper & Pearson 1934,
  Biometrika" in Smoothing/CP.lean:8). Add the three `references:` rows, and name IBP/CROWN in the
  yaml summary. **Evidence:** `grep -iE 'gowal|1906.06316|crown-ibp' formalization.yaml
  LeanMlir/Proofs` → only the CrownBound "Zhang et al. 2018" hit. **Cost:** docs only; the
  comparator tier is unaffected (no new yaml declaration rows). **Size:** S.

### placement
- **C-plc-1** `LeanMlir/Proofs/Certificates/LipschitzCert/Float.lean:55-140`, emitted by
  `scripts/certs/lipschitz_cert_float.py:~190-260`. The problem is that general float-composition
  engine code only exists as a string template in a generator.
  - The engine pieces are `FloatModel.mlp2F`, `FloatModel.mlp2_float_close_uniform` (the 2-layer
    sibling of `mlp_float_close_uniform`, `Float/MlpFloatBridge.lean:68`) and
    `LipschitzCertDemo.certified_at_eps_close` (the float-widened peer of `certified_at_eps`).
  - Every other family keeps its engine hand-written, and `Certificates/README.md:5-12` says so.
    Here the proofs can only be edited through Python.
  - **Fix:** move `mlp2F` and `mlp2_float_close_uniform` into `Float/MlpFloatBridge.lean`, beside
    `mlpF`. Move `certified_at_eps_close` into `DenseEuclid.lean`, beside `certified_at_eps`.
    Namespaces stay, so no renames. Make the generator import instead of emit, and update the
    README's hand-written list.
  - **Cost:** 2 AuditAxioms pins (:1739-1740) keep their names; the generator template changes
    and proofs.yml's `_gencheck --check` regenerates. Gate on `lake build Certs` + the generator's
    `--check`. **Size:** S–M.
- **C-plc-2** `LeanMlir/Proofs/Foundation/IntervalBoundConv.lean:232-322`. This file defines
  certificate vocabulary (`CertifiedAtLinf3`, `CertifiedAtLinf3.mono`,
  `ibp3_certified_of_boxSound`, `deepNet_boxSound`). Its only non-test consumers are
  `Foundation/IntervalBoundConvQ` → `Certificates/IbpConvScorecard/Net`.
  - `Certificates/README.md:22-24` justifies the Foundation placement with "the ones with no
    certificate vocabulary live in `Foundation/` (`IntervalBoundConv`, …)", and that claim is false
    for this file.
  - **Fix:** move `IntervalBoundConv{,Q}` to `Certificates/IntervalBoundConv/{Basic,Q}.lean`
    (placement_imports_cleanup.md already lists the directory pairing as optional), or at least
    move the certificate half (:232-end).
  - **Cost:** imports in `IntervalBoundConvQ`, the `ibp_conv_scorecard.py` template (→
    `IbpConvScorecard/Net`), `tests/AuditAxioms.lean:171`, the lakefile `Certs` root and
    `certs-heavy.yml` paths. The `IBP.` namespace is unchanged (19 audit lines keep their names).
    **Size:** S.
- **C-plc-3** Mathlib-level Gaussian material lives in `Certificates/`, outside the upstream drafts:
  - `GaussianQuantile.lean:22` `stdGaussian.instIsOpenPosMeasure` (a statement purely about
    Mathlib's `stdGaussian`);
  - `Smoothing/Gaussian.lean:54` `gaussianPDFReal_shift`;
  - `Smoothing/Gaussian.lean:109` `integral_gaussianReal_shift_eq` (1-D Cameron–Martin);
  - `Smoothing/Gaussian.lean:63` `integral_indicator_Iic_eq_cdf`, which holds for any probability
    measure.

  **Fix:** add them to `Foundation/UpstreamDraft.lean` plus a PR2 (or PR3) draft, generalised to
  variance `v` as api_design_audit §6.4 proposed, and re-point the uses.
  **Cost:** each is pinned once in AuditAxioms (namespace moves to `MathlibUpstream`). **Size:** M.
  (carried: api_design_audit.md §6.4 "Not done"; the instance is new to the list.)

### naming
- **C-nam-1** The generated aggregates `smoothCpMlp_certified`, `smoothCpCnn_certified`,
  `smoothCpCifar_certified` (`Smoothing/CPScorecard.lean:622,1253,1764`) and
  `smoothDec{Mlp,Cnn,Cifar}_certified` (`Smoothing/DecScorecard.lean`) are emitted by
  `scripts/certs/smooth_scorecard_gen.py:181` and `smooth_dec_scorecard_gen.py:265`. Their names
  overstate the statements:
  - the CP aggregates conclude `binomTail … ≤ 1/1000` (tail arithmetic only);
  - the decimal aggregates conclude `m/2000 ≤ ½·Φ⁻¹(a/10000)`;
  - neither mentions a classifier.

  The README's own ⚠ line (Certificates/README.md:44-46) says these end "in their side conditions".
  **Fix:** rename in the generators to `smoothCp<Net>_tail_le` / `smoothDec<Net>_radius_le` and
  regenerate. **Cost:** 6 AuditAxioms lines, 2 generators, CI `--check` regenerates; no yaml, book
  or comparator pins. **Size:** S.
- **C-nam-2** The engine files `DenseEuclid`, `IntervalBound`, `CrownBound` and
  `LipschitzCert/PairSDP` live in namespace `Proofs.LipschitzCertDemo`. The only reason given is
  compatibility (`DenseEuclid.lean:13`: "the namespace is theirs, kept so every citation keeps its
  name"). "Demo" misdescribes reusable engines (IBP, CROWN), and the compatibility policy does not
  let citations pin a name. **Fix:** pick a name on its merits (e.g. `Proofs.Robustness`) and move
  every consumer. **Cost (stated so it can be scheduled, not waved through):** 106 lines in
  `tests/AuditAxioms.lean`, 75 in `AuditAxiomsHeavy.lean`, 1 in `formalization.yaml`, 1 in
  `gen_comparator_tier.py`, 9 generators plus regenerating every scorecard (including the two
  SDPFull files no CI builds). **Size:** M–L. (carried: api_design_audit.md §6.5 — "choose a
  namespace on its merits"; the naming pass did not take it.)

### documentation
- **C-doc-1** Measured counts are still quoted as proved outside the generated headers (residue of
  the 09-24 finding):
  - `formalization.yaml:219` comment "mechanized 100-image LipSDP scorecard … (69/100)" on
    `scorecard_sdp`, which states 8 witnesses;
  - `formalization.yaml:495` "honest lower bound (69/100 vs 72/100 PGD bracket)";
  - `formalization.yaml:320-321` "The LipSDP scorecards certify a FIXED 100-image subset";
  - `lakefile.lean:284-287` "Results (all 3-axiom …): L2 capped σ≤2 92/100 …; IBP … 92/88/69/24
    per 100 … 93/100";
  - `.github/workflows/certs-heavy.yml:253`, whose green-run step summary prints the same counts
    beside "All N theorems closed";
  - `tests/AuditAxiomsHeavy.lean:26-28, 71-72`.

  **Fix:** reword each as "8 proved witnesses per radius; N/100 measured by exact rational
  arithmetic", as the generated headers now do. **Evidence:** `scorecard_sdp` (ScorecardSDP.lean:2085)
  = `sdpCappedCerts.length = 8 ∧ …`; `scorecardFull` / `scorecard_ibp` / `scorecard_crown` state
  8/8/8/2 lengths. **Cost:** comments only. **Size:** S.
- **C-doc-2** Same class, new site: `Smoothing/CPScorecard.lean:10-11` ("Through
  `smoothing_cp_certified_solved`, each entry certifies the radius σ·Φ⁻¹(q₀) … w.p. ≥ 1−α"), plus
  the section banners (":22 MNIST-MLP: 99/100 certified", ":1278 CIFAR-CNN: 80/100 certified") and
  aggregate docstrings ("80/100 first-100 images certified").
  - Each entry discharges only `htail`. The driver classifiers' `hC`/`hp` are never proved, and the
    header's own Scope paragraph says so.
  - **Fix:** in `smooth_scorecard_gen.py:145-146, 160, 179`, write "each entry discharges the tail
    hypothesis of `smoothing_cp_certified_solved`; the driver nets' `hC`/`hp` are not proved, so
    these are side-condition checks, not certificates of the driver nets". Use "N/100 with a
    certified tail check" in the banners. Regenerate (`--check` in proofs.yml). **Size:** S.
- **C-doc-3** Stale CertsHeavy description, repeated at `lakefile.lean:277-290`,
  `tests/AuditAxiomsHeavy.lean:13-21` and `.github/workflows/certs-heavy.yml:3-9`. All three say
  "784-dim scorecard + per-pair LipSDP + IBP L∞: ~90k lines … across 8 files". They give the split
  reason as "the linarith PSD goals carry ~230-digit LDLᵀ fractions".
  - What the lib actually contains: CertsHeavy's roots are ScorecardFull, ScorecardIBP{,Uncon},
    ScorecardCrown{,Uncon} and IbpConvScorecard. They reach 15 modules and ~19.2k lines. There is
    **no** LipSDP and **zero** `linarith` in any of them.
  - The linarith PSD goals live in `ScorecardSDP{,Uncon}` (in the *light* `Certs` lib, 35
    `linarith [sq_nonneg …]` each) and in the unbuilt `SDPFull*`. CROWN and conv IBP go
    unmentioned.
  - **Fix:** describe the roots as they are (full-input L2, IBP dense + conv, CROWN; ~19k lines;
    split for per-module peak memory, ImgsA 39.5 GB historically). State that LipSDP-full is in no
    lib and the pooled LipSDP is in `Certs`. **Evidence:** `wc -l` over the roots' import closure;
    `grep -c linarith` = 0 in each. **Size:** S.
- **C-doc-4** Smaller stale or false statements:
  - (a) `Certificates/README.md:22-24`: "no certificate vocabulary" for Foundation's
    IntervalBoundConv (see C-plc-2). The README also counts twelve hand-written files, though the
    Float engine is generated (C-plc-1).
  - (b) `planning/mathlib_upstream_drafts/PR1_CDF.lean:11` and `PR2_GaussianReal.lean:17` say
    "Verified to compile … by `LeanMlir/Proofs/UpstreamDraft.lean`", but the file is
    `LeanMlir/Proofs/Foundation/UpstreamDraft.lean`. The "keep the two in sync" mirror has already
    drifted: `cdf_gaussianReal_neg` uses `haveI` in PR2:70 and `have` in UpstreamDraft.
  - (c) `GaussianQuantile.lean:9-10` says its `IsOpenPosMeasure` instances cover "the 1-D and the
    multivariate" Gaussian. The file declares only the multivariate one; the 1-D instance is
    `MathlibUpstream.instIsOpenPosMeasureGaussianReal`. `Smoothing/NetSemantics.lean:16-17`
    repeats the claim.
  - (d) `LipschitzCert/Instance.lean:132` "Generated by scripts; weights are DATA here" names no
    script. It should name `scripts/certs/lipschitz_cert_witness_s8.py` and
    `historical/lipschitz_cert_rationalize.py`, as the README does.

  **Fix:** one-line corrections each. **Size:** S.

### proof-quality
- **C-pq-1** Three capstones repeat the same ~25-line coverage→certificate scaffold:
  `smoothing_mc_certified` (`Smoothing/MC.lean:138-164`), `smoothing_cp_certified`
  (`Smoothing/CP.lean:410-439`) and `smoothing_cp_certified_solved` (`CP.lean:466-495`).
  - The shared steps are: `set γ`, `set A`, `hA`, `hpA := smoothProb_eq_real`, `hcount`, an
    `hsub` whose body is `smoothing_certified_of_le …`, then `calc … cp_coverage/mc_mean_lower_bound
    … measureReal_mono hsub`.
  - **Fix:** state one lemma in `Smoothing/Gaussian.lean`. For any `q : Ω → ℝ`, it should give
    `{ω | q ω ≤ p_y(x)} ⊆ {ω | ∀ δ, ‖δ‖ < σ·Φ⁻¹(q ω) → ∀ j ≠ y, p_j(x+δ) < p_y(x+δ)}`, with
    `p_c(x)` the smoothed integral. MC and CP then become the coverage bound followed by
    `measureReal_mono`. Solved composes the same lemma with `le_cpLower_of_tail_le`.
  - **Cost:** 0 pins (statements unchanged), about −40 lines. **Size:** S–M.

## Checked, not findings
- The theorem statements were checked adversarially.
  - `lipschitz_margin_certified_radius`, `certified_at_eps`, `CertifiedAt.of_margin`,
    `pair_sq_bound` and `certified_at_eps_pair` are correct. The LipSDP slack
    `(vᵀz)² + ρ⁻¹zᵀTGTz ≤ 2Σ Tz²` implies `ρ(vᵀz)² ≤ ρ²‖Δx‖²` via slope restriction plus
    completing the square.
  - `certified_of_boxSound`, `crown2_certified_at_eps`, `relu_upper_envelope` (chord condition
    stated multiplicatively, so rounded slopes are sound) and
    `smoothing_certified_radius_{probit,gaussian,cohen,classifier}` are correct. The latter use the
    real `Φ⁻¹` and σ-scaled N(0,I) noise, matching the drivers.
  - `smoothing_probit_lipschitz` (Salman Lemma 2, both directions), `pi_gaussian_np_shift` (MLR
    pointwise inequality) and `stdGaussian_np_shift` (reflection basis) are correct.
  - `cp_coverage` is correct: the minimal-counterexample argument is valid, and an empty `sInf` set
    gives cpLower = 0, which is harmless. So are `binomTail_monotoneOn` (coupling), the
    `binomTailNumFast` kernel bridge and `argmaxNet_smoothProb_mem_Ioo` (needs ≥ 2 classes, stated).
  - No `True` fields, no ∃-modulus shapes, and no content moved into hypotheses. `hp` is a real
    hypothesis, discharged for `mlpT` by `NetWitness`.
- `smoothing_cp_certified_solved` / `_net` / `_mlpT` are per-fixed-`k₀` statements ("IF the count
  comes out k₀"), which is weaker than CERTIFY's data-dependent guarantee. The data-dependent form
  is `smoothing_cp_certified` (proved), and the docstrings state the fixed-k₀ shape. Honest.
- `smooth_cp_mlpT_demo` already says its count comes from a 784-dim driver, not `mlpT`. TRUST.md:30
  and yaml 4b disclose the 784-dim untied gap.
- Reuse searches came back empty:
  - Mathlib has no Gaussian quantile/probit, no cdf strict-mono/continuity lemmas (hence PR1/PR2
    drafts, still needed at the pin), no Gaussian Cameron–Martin/tilt lemma, and no
    `l2_opNorm ≤ frobenius` lemma. `l2_opNorm_conjTranspose_mul_self` exists, but without the
    Frobenius comparison it does not replace the Gram chain.
  - Mathlib's `binomial` is `setBer(Iio n,p).map ncard` over `infinitePi`. Bridging `Measure.pi`
    (Fin N) to it costs about what `pi_hitCount_eq_binomial`'s 90-line induction does, so this is
    not a finding.
  - `denseE` could be `toLp (W *ᵥ x)` and `crownRow` is `Matrix.vecMul` (the in-code comment says
    so). Both are left as they are: the generated corpus rewrites through `denseE_apply`.
  - Interval boxes vs `Set.Icc` was killed in audit v2.
- Gram/Schatten-4/8 bounds (`denseE_lipschitzL2_gram{,2}`): textbook Schatten-p ≥ operator norm,
  so no attribution is required. If the iterated-Gram idea was taken from Delattre et al. 2023
  ("Gram iteration", ICML), a one-line credit would be courteous.
- Tsuzuku, Cohen–Rosenfeld–Kolter, Salman, Fazlyab (LipSDP), Hoeffding and Clopper–Pearson (by
  name) are credited in module docstrings and yaml.
- Scope: MC.lean (Hoeffding) is superseded in practice by CP but is a `Certs` root and a
  standalone valid bound. The `LipschitzCert/ScorecardSDPFull{,Uncon}` exclusion is a known,
  documented decision.
- The `show` sites in DenseEuclid:79 and CrownBound:252 are deliberate: they keep the inner
  `denseE W1 x` folded (documented at CrownBound:246-248). A `simp only` replacement unfolds it
  (tested).
- Generated files were skipped per protocol; `maxHeartbeats` appears only in generated files.
  No `sorry`/`native_decide`/axioms in the slice.

## Gaps for the humans
- `ScorecardSDPFull{,Uncon}` (6k lines) are built by no lib and no CI job. A change to `PairSDP`,
  `ScorecardFullImgs*` or the generator can break them silently until someone runs
  `scripts/certs/check_sdpfull.sh` by hand (~16 GB).
- Only 4 of the ~11 certificate generators are `--check`-regenerated in CI (`_gencheck` in
  proofs.yml: `lipschitz_cert_float`, `smooth_scorecard_gen`, `smooth_dec_scorecard_gen`,
  `smoothing_net_witness_gen`). The others need `data/` and are reproducibility-checked only by hand
  (placement_imports_cleanup §7).
- No gate keeps `planning/mathlib_upstream_drafts/*.lean` in sync with
  `Foundation/UpstreamDraft.lean`; they already drift (C-doc-4b). A small diff script over the
  declaration bodies would catch it.
- No mechanical check that a prose "N/100 certified" count matches the aggregate theorem's
  `.length`. The generated headers were fixed by hand twice. A lint could grep
  `[0-9]+/100 certified` outside generated "MEASURED" lines.
