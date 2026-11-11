# Retiring the pooled 49-dim certificate model

**Goal.** The robustness certificates keep only the full-input MNIST family. The pooled "REDUCED
CERTIFICATE MODEL" was scaffolding and gets deleted: 4×4-pooled 49-dim input, width-8 hidden,
nets `mlpS` (capped, /256) and `mlpT` (unconstrained, /128).

**Status.** Not started. The decisions in §4 are open; §6 (2026-10-28, `c141448b`) re-checked the
map, measured both families' memory, and recommends an answer to each decision. The quality-indicator
pass that prompted it is in the lean-quality-indicators run of 2026-10-28.

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

## 6. Re-check at `c141448b` (2026-10-28)

**The map holds.** None of the 19 commits between `ea934bc6` and `c141448b` touched the pooled
sections of `lakefile.lean`, `tests/AuditAxioms.lean`, the comparator tier files, `formalization.yaml`
or the gate baselines. Every path in §1–§3 was re-verified. Outside the family, the only *imports* are
`tests/AuditAxioms.lean` and `tests/comparator/{Challenge,Solution}Tier.lean`; every other hit is a
docstring or prose. `blueprint/src/content.tex` cites nothing from the family.

### 6.1 Memory, measured

Lean 4.35.0, `/usr/bin/time -v`, warm imports, this box. `LEAN_NUM_THREADS=1`; the default thread
count gave the same peaks for the pooled files (one goal at a time either way).

| module | peak RSS | wall |
|---|---|---|
| pooled `Instance` / `Scorecard` / `Float` / `MlpWitness` (`lake env lean`) | 1.9–2.7 GB | 4–30 s |
| pooled `NetWitness` | 3.1 GB | 29 s |
| pooled `ScorecardSDP` / `ScorecardSDPUncon` | 4.5 / 4.6 GB | 55 s each |
| `ScorecardFull` apex, ImgsA/B/Nets prebuilt | 1.9 GB | 4 s |
| `ScorecardFullImgsA` / `FullNets` (certs-heavy.yml comment, Lean 4.32) | 11.4 / 10.1 GB | — |
| `ScorecardSDPFull`, `lake env lean` | 11.9 GB | 3:53 |
| `ScorecardSDPFull`, `lake build` | 14.1 GB | 4:49 |
| `ScorecardSDPFullUncon`, `lake env lean` | 12.5 GB | 3:57 |

So: the full-input L2 family already fits the heavy runner's one-module-at-a-time loop and is
CI-checked. The LipSDP-full pair is down from the 15–16 GB on record to 12–14 GB, but 14 GB under
`lake build` is what a 16 GB runner has free. They stay local-only unless the PSD witnesses change
shape (§6.3).

### 6.2 The five roles of the pooled net, and what replaces each

1. **L2 scorecard** (`scorecard` → `scorecardFull`). Exists in `CertsHeavy`, CI-checked. Nothing to
   port.
2. **LipSDP headline** (`scorecard_sdp` → `scorecard_sdp_full`). Exists, same shape, 16 proved images
   (8 per radius) instead of 8. Built by no lib, so it cannot enter the comparator tier
   (`comparator.yml` builds `Certs` only). → §4.1 **(a)**: point the yaml row at it, mark it checked
   by `check_sdpfull.sh`, drop it from the comparator tier.
3. **Smoothing at a concrete net** (`smoothing_cp_certified_mlpT`). `TRUST.md:29` and the yaml (4b,
   and the Cohen row) cite it, so **port** rather than drop. The port is cheap:
   - `mlpT_logit_continuous` generalises verbatim to any width → one engine lemma in `DenseEuclid`
     (`mlp_logit_continuous (W1) (W2)`), which also serves `mlpTF`.
   - `smoothing_cp_certified_net` needs one strict-argmax witness per class. The committed full-input
     images (ImgsA/B: test #0–#7, #9, #25, #71) cover digits 0, 1, 2, 4, 7, 9. Four new 784-dim images
     (first test image of digits 3, 5, 6, 8 with a strict `mlpTF` argmax) are needed, each with its 16
     `dotZ` pre-activation evaluations — the existing ImgsA recipe.
   - `smoothing_net_witness_gen.py` parses `ScorecardFullNets` instead of `Instance`. New module
     `Smoothing/NetWitnessFull` in `CertsHeavy` (a few GB; ImgsA holds 10 images at ≤ 11.4 GB).
   - Then `TRUST.md:29`, yaml 4b and the Cohen row say "the trained 784-dim `mlpTF`", and the
     "witness file runs at the 49-dim net" caveat goes.
4. **Float tier** (`certifiedC*_float` on `mlpS` → `mlpSF`). Portable: `mlp2_float_close_uniform` is
   dimension-polymorphic and the weight bound is already a kernel `decide` (`abs_le_of_check`), so
   784×16 costs nothing. What does not scale is the per-image `img*_abs_le` by `fin_cases` over the
   coordinates; it has to become a kernel check too (an `absLeCheck` over the image as a 1×784
   matrix). Budget constants are recomputed by the generator. Mostly generator work, lands in
   `CertsHeavy`. Nothing cites it except two docstrings (`Verified/Attack.lean:18`,
   `IbpConvScorecard/Net.lean:16`) and the book does not depend on it → **drop now**, keep this port
   as a later option. If dropped, `FloatModel.mlp2F` / `mlp2_float_close_uniform` lose their only
   consumer (§4.4).
5. **The VJP witness** (`Trained/MlpWitness`). Level 1 is vacuous (`HasVJPAt` is inhabited by
   `canonical`), and a 784-dim version would evaluate `pdiv_fwd_val` through a 784-term sum by
   `simp [Fin.sum_univ_succ]`, the pattern the full files abandoned for `dotZ`. → §4.3 **delete**.
   `CnnWitness` / `CnnSeal` remain the trained-weight witness rung; reword the yaml 4a sentence and
   the `CnnWitness` / `trained_cnn_witness.py` / `trained_cnn_seal.py` "extends `Trained.MlpWitness`"
   lines.

Recommended answers, then: §4.1 (a); §4.2 port the smoothing tie, drop the float tier; §4.3 delete;
§4.4 prune; §4.5 leave `GramQ` in `CertsHeavy`.

### 6.3 Modules that only build on this box: the pattern

The repo has a three-rung ladder and the gap is the top rung:

| rung | gate | ceiling | enforced by |
|---|---|---|---|
| `Certs` | every proof push + nightly | ~5 GB per module, default scheduler | certs.yml, comparator.yml |
| `CertsHeavy` | push on its paths + 1st/15th cron | ~12 GB, one module at a time, `LEAN_NUM_THREADS=1` | certs-heavy.yml; module splitting (`ImgsA–D`) pushes a file down into this rung |
| no lib + local script | nothing | this box | `check_sdpfull.sh`, by hand |

Nothing fails if `PairSDP` or the ImgsA/B/Nets files change and `check_sdpfull.sh` is not re-run.
Three ways to close that, in the order to try them:

1. **Fix at the source.** `planning/archive/certs_heavy_psd_memory.md` option 1: replace the exact
   LDLᵀ witness by a `/2^k`-rounded factor plus a diagonally dominant remainder checked entrywise by
   `norm_num` (`quad_form_nonneg_of_dd`, `lipsdp_slack_of_cert_split`). That removes the ~230-digit
   fractions behind the 12 GB goals and would move both SDP-full modules into `CertsHeavy`. Estimated
   half a day there; no draft exists in the tree (grep finds only the two planning mentions).
2. **A freshness stamp.** `check_sdpfull.sh` writes a committed stamp (e.g.
   `scripts/certs/sdpfull.stamp`) holding a hash of its inputs: the two modules, `PairSDP.lean`,
   `ScorecardFull{Nets,ImgsA,ImgsB}.lean`, `lipschitz_cert_pair_sdp_full.py`, `lean-toolchain`,
   `lake-manifest.json`. A cheap CI step (proofs.yml's Python block, or `lake lint`) recomputes the
   hash and fails on mismatch. CI then enforces that the local check ran against the *current*
   inputs without doing the work. Same ethos as the generators' `--check`; generalises to any future
   local-only module through one table of (script, input globs, stamp).
3. **A self-hosted runner on this box.** Zero runners are registered. The account is personal and on
   the free plan; GitHub's larger runners are an org-level paid feature (Team / Enterprise), so this
   box is the only bigger machine on offer. GitHub advises against self-hosted runners on public
   repos; certs-heavy.yml never runs on pull requests, which limits the exposure, but it would still
   want an ephemeral runner under a dedicated user, a `[self-hosted, certs-heavy]` label used by that
   one workflow, and the box always up.

Recommendation: the stamp now, so the removal commit can take §4.1 (a) honestly; the
diagonal-dominance rewrite as the follow-up that makes the stamp unnecessary for this pair.

### 6.4 Gates (addendum to §5)

If §6.2.3 is taken: `lake build CertsHeavy` with `NetWitnessFull` in the certs-heavy.yml
serialization loop (the "every root is in the loop" step enforces this), `tests/AuditAxiomsHeavy.lean`
pins for its theorems, and `smoothing_net_witness_gen.py --check`.

## 7. A common frame for the trained witnesses (2026-11-11)

**Decision (the user, 2026-11-11).** The trained-weight witnesses of Chapters 1–3 move to one
frame: the chapter's own architecture, narrow, on a real 28×28 MNIST test image. The reason is the
reader, not the proof: one thing to understand across the three chapters. This supersedes §4.3's
"leaning delete" for `MlpWitness`.

| Chapter | Today | Target |
|---|---|---|
| Linear | 4×4-pooled 49→10 (`Trained/LinearDescent`) | 784→10, the chapter's net at full size |
| MLP | 4×4-pooled 49→8→10 (`Trained/MlpWitness`) | 784→8→8→10 (`mlpG 8 8`'s shape) |
| CNN | 24×24 crop, 2×2 block sums to 12×12, 2 channels, 8-wide head (`Trained/CnnDescent`, `CnnDescentConv1`, `CnnWitness`, `CnnSeal`) | 28×28, 2 channels, 8-wide head |

**Sizes (counts from the architectures, not measured builds).** Linear 490 → 7,840 weights (16×).
MLP about 480 → about 6,400 (13×). CNN 576 → 3,136 pre-activations per convolution and 72 → 392
pool windows (about 5×). The chapter widths (512, 32 channels) are about 1,400× and 4,000× in
weights and about 11 million multiplies per CNN forward; that needs a reflective integer evaluator,
not larger `norm_num` tables, and is out of scope here.

**What is not a resize.**
- Per-coordinate checks by `fin_cases` over 49 inputs become one kernel check over 784 (§6.2.4's
  `absLeCheck` point).
- A 28×28 image has blank borders, so the CNN's tied pool windows grow in number; the twin
  argument (`MaxPool2MarginQUpTo`, `ConvPatchEq.of_zero`) is the same.
- The step sizes shrink with the widths' products in the small-step condition.
- Whether the MLP chapter has a trained DESCENT instance, or only the VJP witness, is unchecked.
  If only the VJP witness, a parallel story across the three chapters needs a new instance.

**Order.** Linear first, as the proof of concept for the 784-coordinate checks; then the MLP; then
the CNN descent pair; `CnnWitness` / `CnnSeal` last (the seal is the largest file). Scope added to
§1–§3: `LinearDescent` and the four CNN files, their generators (`trained_linear_descent.py`,
`trained_cnn_descent.py`, `trained_cnn_witness.py`, `trained_cnn_seal.py`), and the book's witness
sentences in `thm:cnn_sgd_descends` and its Chapter 1–2 peers.

**Status (2026-11-11).** Linear, MLP and the CNN descent pair done on `book/audit-pass`; the CNN
VJP witness and seal are blocked (below).
- `Trained/LinearDescent` is the 784→10 net on test image #8 (`trained_linear_descent.py`
  rewritten onto `Foundation/ListDot`; the fan-in factor through `gamma_num`, no big power). 27 s.
- `Trained/MlpWitness` is `mlpForward` / `mlpHasVJPAt` at 784→8→8→10, bias-free, on test image #0,
  from the new `trained_mlp_witness.py`; it no longer imports `LipschitzCert.Instance`, so §1's
  "VJP witness on `Instance`'s weights" row and §4.3 are settled (kept, moved). 29 s.
  `lipschitz_cert_witness_s8.py` no longer writes it.
- `Trained/MlpDescent` (new, `trained_mlp_descent.py`): `mlp_input_sgd_descends` at the exact
  gradient, on the SAME net as `MlpWitness` (imported) and test image #63 (a 3 read as a 2), the
  first misclassified test image off every ReLU kink with a sign-definite row of
  `mask⁰·W₁·mask¹·W₂`; `lr = 2⁻³⁵`, strict form proved. No binary32 rung: its layer-2 rounding
  margin fails on that image.
- `Trained/CnnDescent` and `CnnDescentConv1` are 1×28×28 → conv 1→2 → conv 2→2 → pool → dense
  392→8 → 8 → 10 on test image #0, tables as kernel list checks through the new
  `Architectures/ListConv` (one `decide +kernel` per conv map and one for all 392 pool windows);
  549 and 559 lines (were 2,291 and 2,269), about 70 s each. 1,568 pre-activations per
  convolution, not the 3,136 this section's size line said.
- **Blocked: `CnnWitness` / `CnnSeal` stay on their pooled 6×6 input.** `mnistCnnNoBnHasVJPAt`
  needs `MaxPool2Smooth` (a strict unique max in every window), and on a 28×28 image a window
  whose cells read blank two-conv receptive fields has four equal conv2 outputs at any weights;
  all 10,000 test images have at least six such windows per channel. Moving them needs a ReLU+pool
  "smooth or dead" input VJP for the 2×2 pool (the ResNet stem has one, `stemReluPoolLayer` /
  `MaxPool3s2SmoothOrDead`) and a net whose blank-background conv2 output is negative in both
  channels.
- `Foundation/ListGetD` (new) holds the four `getD` bridges the linear and MLP descent files share.
