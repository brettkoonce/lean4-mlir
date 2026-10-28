# Retiring the pooled 49-dim certificate model

**Goal.** The robustness certificates keep only the full-input MNIST family. The pooled "REDUCED
CERTIFICATE MODEL" was scaffolding and gets deleted: 4×4-pooled 49-dim input, width-8 hidden,
nets `mlpS` (capped, /256) and `mlpT` (unconstrained, /128).

**Status.** Not started. The decisions in §4 are open. The removal map in §1–§3 was made at
`ea934bc6` (2026-10-28) and should be re-checked before work starts. The quality-indicator pass that
prompted it is in the lean-quality-indicators run of the same day.

## 1. What goes

| Kind | Path | Why |
|---|---|---|
| Lean | `Certificates/LipschitzCert/{Instance,Scorecard,ScorecardSDP,ScorecardSDPUncon,Float}.lean` | the pooled family |
| Lean | `Training/Trained/MlpWitness.lean` | VJP witness on `Instance`'s weights (`W1tQ`, `W2tQ`, `xt`, `hpre_eval`) |
| Lean | `Certificates/Smoothing/NetWitness.lean` | smoothing ↔ `mlpT`; its generator parses `Instance.lean` |
| Generator | `scripts/certs/lipschitz_cert_scorecard.py` | emits `Scorecard` |
| Generator | `scripts/certs/lipschitz_cert_pair_sdp.py` | emits `ScorecardSDP{,Uncon}` |
| Generator | `scripts/certs/lipschitz_cert_float.py` | emits `Float` |
| Generator | `scripts/certs/lipschitz_cert_witness_s8.py` | emits `MlpWitness` and an `Instance` snippet |
| Generator | `scripts/certs/smoothing_net_witness_gen.py` | emits `NetWitness` |
| Logs (optional) | `runs/2026-09-30-cert-scorecards/lipschitz_cert_{scorecard,pair_sdp}.log` | pooled runs |

Nothing outside these files imports them except `lakefile.lean`, `tests/AuditAxioms.lean` and the
comparator tier.

## 2. What stays

- Full-input family, `CertsHeavy`: `ScorecardFull*`, `ScorecardIBP*`, `ScorecardCrown*`.
- Full-input family, in no lib: `ScorecardSDPFull{,Uncon}`. Their PSD `linarith` peaks at about 15 GB,
  which kills the 16 GB CI runners, so `scripts/certs/check_sdpfull.sh` re-checks them. Last run
  2026-10-28 at `ea934bc6` on the build box: both build (271 s and 285 s), and all 445 theorems
  rest on the three core axioms.
- Engines: `LipschitzCert/Basic`, `DenseEuclid`, `PairSDP`, `Foundation/GramQ`.
- `Smoothing/NetSemantics` (abstract, no net), `CPScorecard`, `DecScorecard`.
- `GramQ` leaves the `Certs` closure. After the removal only `ScorecardFullNets` (`CertsHeavy`)
  imports it.
- `content.tex` cites nothing from the pooled family. Its "pooled" hits are pooling layers.

## 3. Edits, in lockstep

These break CI if done separately.

1. **Lake roots and the audit.**
   - `lakefile.lean` `Certs` roots: drop `NetWitness`, `Instance`, `MlpWitness`, `Scorecard`,
     `ScorecardSDP`, `ScorecardSDPUncon` and `Float`; keep `PairSDP` and `ListDot`.
   - Reword the `CertsHeavy` docstring ("The pooled LipSDP files are in `Certs`").
   - `tests/AuditAxioms.lean`: drop the matching imports and pins (the `Instance`, `Scorecard`, SDP,
     `Float`, `TrainedMlp.*` and `NetWitness` sections). Keep the engine pins (`DenseEuclid`,
     `PairSDP`) and re-head their sections. `check_audit_coverage.py` passes only if roots and pins
     move together.
2. **Comparator.**
   - In `scripts/gates/gen_comparator_tier.py`, drop `Proofs.Robustness.scorecard_sdp` and its
     module, then regenerate `tests/comparator/{Challenge,Solution}Tier.lean` and `config-tier.json`.
   - Its `check_yaml()` requires every `formalization.yaml` main result in the config, so the
     yaml row changes in the same commit (§4.1).
   - `tests/comparator/README.md` lists `scorecard_sdp`.
3. **Generated files that cite the pooled family.** Edit the generator and its `--check`ed output
   together:
   - `smooth_dec_scorecard_gen.py` and `smooth_scorecard_gen.py`: the docstrings of `DecScorecard` and
     `CPScorecard` say "discharged for a concrete trained net in `Smoothing/NetWitness.lean`".
   - `lipschitz_cert_scorecard_full.py`: `ScorecardFull` docstring.
   - `ibp_conv_scorecard.py`: `IbpConvScorecard/Net` ("unlike the dense `LipschitzCert.Float`").
   - `trained_cnn_witness.py` and `trained_cnn_seal.py`: `CnnWitness` and `CnnSeal` call
     `Trained.MlpWitness` their base.
4. **Hand-written docstrings that cite deleted names.** These trip `module_refs.py` and
   docstring-checkrefs:
   - `DenseEuclid.lean:18`
   - `Foundation/ListDot.lean:17`
   - `Smoothing/NetSemantics.lean:30–33`
   - `Verified/Attack.lean:18`
   - `Nets/Small/MlpCanonical.lean:26–30`
   - `Float/MlpFloatBridge.lean:10–11`
   - `LipschitzCert/PairSDP.lean:30` (point at `lipschitz_cert_pair_sdp_full.py`)
5. **CI.** `.github/workflows/proofs.yml`: drop `lipschitz_cert_float.py` from the path filters and
   the `--check` step, and fix the "Three generators…" comment. `certs-heavy.yml`'s
   `LipschitzCert/Scorecard*.lean` glob still matches the full files.
6. **Gate baselines.**
   - `scripts/gates/name_lint_baseline.txt`: the `Instance` and `MlpWitness` rows.
   - `scripts/gates/comment_numbers_allow.tsv`: the `Instance.lean` rows.
7. **Prose.**
   - `LeanMlir/Proofs/Certificates/README.md`:
     - the generator list and the hand-written list;
     - the float-tier paragraph;
     - the table rows for pooled L2, pooled LipSDP, float tier and smoothing ↔ net;
     - the "only the pooled `mlpT` composes them" note;
     - the `mlpS`/`mlpT`, `C`/`U`, `SC`/`SU` naming paragraph.
   - `TRUST.md:29`.
   - `formalization.yaml`: the `scorecard_sdp` row, the `MlpWitness` row, the smoothing ↔ `mlpT`
     sentence, the `scorecard; scorecard_sdp / scorecard_sdp_uncon` list and its module glob.
   - `runs/2026-09-30-cert-scorecards/README.md`: the `scorecard_sdp` example and the
     "Pooled 49-dim MLP" table.

## 4. Open decisions

1. **The LipSDP headline.** `scorecard_sdp` is the only LipSDP-at-trained-weights row in
   `formalization.yaml` and the comparator tier. Options:
   - (a) Swap in `scorecard_sdp_full`, marked as checked by `check_sdpfull.sh` rather than CI, and
     leave it out of the comparator tier. The comparator CI can't build it.
   - (b) Drop the row.
   - (c) Swap in `scorecardFull` (L2, `CertsHeavy`) and make `comparator.yml` build `CertsHeavy`.

   Leaning (a).
2. **Claims that only the pooled net carries.**
   - Smoothing discharged at a concrete trained net (`smoothing_cp_certified_mlpT`; TRUST.md and the
     yaml cite it).
   - The float-tier certificate (`LipschitzCert.Float`).

   Either drop both claims and reword, or port them to `mlpTF`/`mlpSF`. Porting needs a new
   continuity lemma and generator, and the result lands in `CertsHeavy`. Leaning drop. The book's
   smoothing paragraph does not depend on either.
3. **`MlpWitness`.** Either delete it, or keep it by inlining the 49→8→10 data it reads from
   `Instance`, which keeps a reduced-model file. Leaning delete, and reword the yaml row and the
   `CnnWitness`/`CnnSeal` text.
4. **Leftovers with no remaining instance.**
   - The 2-dim toy demos in `Instance.lean` (`linear_demo_certified`, `mlp_demo_certified`, …).
   - `FloatModel.mlp2F` and `mlp2_float_close_uniform`.
   - `GramQ.absLeCheck` and `abs_le_of_check`.
   - In `DenseEuclid`: `certified_at_eps_close`, `denseE_lipschitzL2`, `denseE_lipschitzL2_gram`,
     `sum_sq_matvec_le`, `lipschitzL2_lower_euclid`, `sqrt_two_le_rat`.

   Either prune them with their pins or keep them as engine API. Leaning prune.
5. **`GramQ`.** Either leave it built only through `CertsHeavy`, or make it an explicit `Certs` root
   so `Certs:docs` keeps it. Leaning leave it in `CertsHeavy`.

## 5. Gates for the removal commit

- `lake build Certs CertsHeavy LeanMlir Apps Reference TestSupport ComparatorTests`
- `tests/AuditAxioms.lean` and `tests/AuditAxiomsHeavy.lean`, with no `sorryAx`
- `gen_comparator_tier.py --check`
- `lake lint` (name lint, comment numbers, module refs, audit coverage, import audit)
- the `--check` of every touched generator
- docstring-checkrefs, blueprint checkdecls
- `scripts/certs/check_sdpfull.sh` if §4.1 (a) is taken
