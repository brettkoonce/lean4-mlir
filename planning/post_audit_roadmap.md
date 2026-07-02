# Planning — Post-audit roadmap (2026-07)

Where to go after the 2026-07-01 hostile audit and the four-commit response arc
(`1769bb6..b92305f`). Context: the audit's top-3 ranked gaps are now CLOSED —
(1) the Tsuzuku certificate is instantiated (`LipschitzCertInstance.lean`,
certified radius ladder 0.0463 → 0.1106 → 0.1541 on a trained /128-rationalized
49→8→10 pooled-MNIST MLP, with proved two-sided spectral sandwiches
‖W1‖₂∈[7.452, 7.769], ‖W2‖₂∈[7.7, 8.211]); (3) the first NON-synthetic live
witness exists (`TrainedMlpWitness.lean`, ReLU smoothness inherited from
training, level-3 sealed via the explicit Jacobian entry −85017/8192); and the
suite's two chonky files are fixed (the `mhsa_has_vjp_mat` Eq.mpr-cast de-cast:
ViTBackB0 625s→3.4s, tie 220s→2.0s — see the docstring on
`mhsa_composed_has_vjp_mat` for the kernel lesson). AuditAxioms: 1231/1231.

This doc ranks what's next by (headline value) × (tractability). Items 1–3 are
each 1–2 focused sessions; the batched hygiene pass is one short session.

## 0. TL;DR — the priority order

1. **Certified-accuracy scorecard** (§1) — turn the one-input cert into a
   dataset-level claim: "N% of the MNIST test set certified robust at ε,
   kernel-checked, zero axioms." Mostly generator scripting; strongest
   available robustness headline. Do this first.
2. **IEEE axiom discharge** (§2) — construct the rounding operator, delete the
   repo's only two axioms. "Zero axioms anywhere" + CI simplification.
3. **Descent at trained weights** (§3) — one provably loss-decreasing SGD step
   on the trained net; retires the last "satisfiability witness is degenerate
   (W=0)" caveat and fuses the descent + trained arcs.
4. **Hygiene batch** (§4) — cast-pattern grep, stale heartbeat pragmas,
   untracked stragglers.
5. Separate empirical track (§5, klawd): R50-A3 SGD+momentum probe,
   grad-accum first GPU run.
6. Parked-without-guilt list (§6).

## 1. Certified-accuracy scorecard (the headline move) — ✅ DONE 2026-07-02

**Outcome** (`LeanMlir/Proofs/LipschitzCertScorecard.lean`, generator
`scripts/lipschitz_cert_scorecard.py`): fixed first-100 test subset, fixed
ε = 1/10 (pooled L2) — capped net **34/100** certified vs **1/100**
unconstrained; PGD brackets at 72/69. Aggregate = honest lower bound only
(an upper-bound L can't prove an image uncertifiable), so only certified
images carry theorems (35 margin proofs, not 200 — whole module builds in
~2¼ min wall). AuditAxioms 1250/1250; RESULTS.md + README carry the table.

**Deviations from the plan below, learned the hard way:**
- Caps c≈1.5–2 were WRONG at this scale: σ≤2 costs 24 points of clean
  accuracy (66%); the sweet spot is **c=4, 36 epochs** (87.0% clean, L=19.8,
  /256 grid). The gap-vs-ceiling story survives (L=19.76 vs c²=16).
- Fixed-ε form needed a new lemma `certified_at_eps` (rational radius check
  via √2 ≤ 14143/10000 — no irrationals reach the kernel).
- PGD side done in numpy inside the generator (same quantized nets), not the
  `mnist-mlp-pgd` driver (which trains its own 784→512→512→10 net).

**Claim to produce:** over a fixed test subset (start: 100, stretch: 500
images), the fraction where `lipschitz_margin_certified_radius` certifies the
prediction at a fixed ε — i.e. per-image theorems
`∀ δ, ‖δ‖ < ε → argmax fixed`, plus one aggregate count statement. All
in-kernel, 3-axiom clean.

**Why it's the right next move.** The pipeline is already end-to-end: trained
rational weights → proved L (Schatten-8 Gram, `denseE_lipschitzL2_gram2`) →
in-kernel margin (`xt_margin` shape) → radius. The single-input demo
generalizes *mechanically*: L is proved ONCE per net; each additional image
costs only its margin evaluation (the `hpre_eval` + 10-logit pattern, ~2–5s
kernel each — 100 images ≈ minutes, fine as one CI-built module). Nobody can
quibble with a dataset-level number the way they can with one hand-picked
digit.

**The multiplier: spectrally-constrained training.** On the unconstrained net
the product L=63.8 vs ceiling 57.4 means radii are honest but small
(ε≈0.15 on inputs of norm ~2.8). `mnist-mlp-spectral` exists PRECISELY to cap
‖Wᵢ‖₂ during training (planning/robustness_ladder.md's gap-shrinking lever) —
retrain the 49→8→10 net with caps c≈1.5–2, rationalize, re-run the same
generator. The product cert on a capped net is near-tight, so the certified-
accuracy-vs-ε curve becomes genuinely competitive instead of
90%-of-a-loose-ceiling. Report BOTH nets (unconstrained vs capped) — the
comparison IS the chapter narrative: same theorem, training method decides
whether the certificate bites.

**Mechanics.**
- Extend `scripts/lipschitz_cert_witness_s8.py`: loop over test indices,
  emit per-image `margin_i`/`certified_i` lemmas + one
  `certified_count : (images.filter certified).card = K` style capstone
  (or simply K named theorems + a comment; keep the aggregate simple).
- One new file `LeanMlir/Proofs/LipschitzCertScorecard.lean`; wire into
  lakefile roots + AuditAxioms (spot-check ~10 of the per-image lemmas +
  the aggregate, not all N, to keep the audit file sane).
- Pick ε as a round rational ≤ the capped net's typical radius (e.g. 36/255
  in pooled-feature L2 — state the pixel-space interpretation in the
  docstring: pooled-L2 ε corresponds to a 4×4-block-averaged pixel budget).
- Also emit the PGD side (existing `mnist-mlp-pgd` driver) on the same nets
  for the `cert ≤ TRUE ≤ PGD` sandwich table — empirical, not proof, but
  it's the honest bracketing the ladder doc already uses.

**Risks/notes.** Kernel time scales linearly in images — if 100×margin blows
past ~10 min, drop to 50 images or split the module. Keep the per-image
weights SHARED (one `W1t`/`W2t` def, images as data rows) so the file doesn't
balloon. The capped-net retrain must re-check quantized accuracy (the /128
grid on small-norm weights is coarser relatively — may need /256; the
generator's exact-fraction path handles any power of two).

**Done when:** scorecard module builds in CI, AuditAxioms includes the
aggregate + spot-checks, README/RESULTS gets the certified-accuracy table
(unconstrained vs capped, cert vs PGD).

## 2. Discharge the IEEE axioms (`ieeeRnd`/`ieeeRnd_err`) — ✅ DONE 2026-07-02

**Outcome:** exactly the construction below (`rndP`/`rndP_err`, ~45 lines — the
estimate said 100–200; `Int.zpow_log_le_self` + `abs_sub_round` did all the
work). Interface kept *parametric in `p`* via `gridModel p u (hu : 2⁻¹⁻ᵖ ≤ u)`
rather than two ad-hoc instances — `binary32 = gridModel 23 u32`,
`fp8E4M3 = gridModel 3 u_e4m3`, both `binary32_u`-style rfl-lemmas intact,
downstream corollaries unchanged. Binary32Instance joined the `Proofs` roots +
main AuditAxioms closure (1254 lines); `tests/AuditTrustedBridge.lean`, the
`TrustedBridge` lib, and the CI footprint step are DELETED, replaced by a CI
zero-`axiom`-declaration grep. Bonus theorems: only `rndP_zero` (note:
`rndP_neg` is FALSE at ties — Mathlib `round` is half-up, so `round(−t) ≠
−round t` at half-integers; monotonicity/idempotence parked, no consumer).

**Goal:** replace the repo's only two axioms (`Binary32Instance.lean:47,51`)
with a constructed operator + theorem; TrustedBridge joins the 3-axiom
closure; the separate CI footprint check collapses into the main one.

**Construction** (unbounded-exponent p-bit grid — exactly the idealization the
axiom's own docstring says it models: "the ∀x form abstracts away overflow and
the subnormal floor"):

    noncomputable def rndP (p : ℕ) (x : ℝ) : ℝ :=
      if x = 0 then 0 else
        (round (x / 2 ^ (Int.log 2 |x| - p)) : ℝ) * 2 ^ (Int.log 2 |x| - p)

    theorem rndP_err (p : ℕ) : ∀ x, |rndP p x - x| ≤ 2⁻¹⁻ᵖ * |x|

Proof: scale into `[2^p, 2^(p+1))` via `Int.zpow_log_le_self` /
`Int.lt_zpow_succ_log_self`, `|round t − t| ≤ 1/2` (`abs_sub_round`), unscale.
~100–200 lines, Mathlib-only.

**Interface simplification:** Binary32Instance only ever uses u = 2⁻²⁴ and
2⁻⁴, so DROP the parametric-u axiom shape — two concrete instances
(`rnd24`, `rnd4`) suffice; `ieeeModel` becomes a def. Downstream corollaries
(`binary32_e4m3_argmax_preserved`, `binary32_linear_sgd_descends_concrete`, …)
should go through unchanged since they only consume the FloatModel interface.

**Bonus theorems** the FloatBridge currently has to hypothesize, now provable
about a concrete operator: monotonicity, `rndP p` fixes grid points
(idempotence), sign preservation, `rndP p 0 = 0`.

**Second step (optional, +~200 lines):** the bounded-exponent variant with
subnormal floor η = 2⁻¹⁵⁰ matching `FaithfulFloatModel`
(FloatSubnormalBridge) — discharges the honest model too. Overflow stays out
of scope by design (true binary32 violates any relative bound there).

**What it does NOT buy** (say so in the commit message): the kernel↔model
boundary (FMA, reassociation, "the GPU behaves like round-to-nearest on this
grid") remains trusted per floatbridge_certificate_gaps.md — the trust moves
from "an operator with this bound exists" (mathematically trivial — id
satisfies it) to a concrete, inspectable operator. The honesty win is
hygiene + inspectability, not a smaller hardware trust base.

**Done when:** `axiom` count in the repo is zero, AuditTrustedBridge's
expected footprint is the bare triple, proofs.yml's trusted-bridge step is
merged into the main closure check.

## 3. Descent at trained weights (retire the W=0 caveat)

**Gap:** `binary32_linear_sgd_descends_concrete` (the only concrete descent
instance) holds at the degenerate W=0, b=0 net — a satisfiability witness,
not a trained-net statement. We now have trained rational weights AND exact
gradient evaluation machinery.

**Target:** one theorem — at the trained /128 MLP (or start with a trained
LINEAR classifier on the pooled features, where `SgdDescentLinear`'s
machinery applies directly with PROVEN smoothness constants), a concrete
input/label and a concrete lr, the SGD step provably decreases the
cross-entropy loss, hypotheses `hsmall`/`h1`/`h2` discharged by `norm_num`
over exact rationals.

**Plan of attack (linear first):** train bias-free 49→10 linear on pooled
MNIST (~91% acc, same generator infra), rationalize to /128; a = input bound
(pooled features ≤ 1, so a = 1... but `hsmall : 2aD < 1` constrains the ℓ1
step radius D = lr·‖∇‖₁-ish — compute exact ∇ at the chosen sample, pick lr
so the window closes; the Binary32Instance proof at W=0 is the template,
swap in trained numbers). If the window is nonempty at trained weights
(compute in Python first — if not, pick a lower-loss sample or smaller a via
input normalization), the Lean side is mechanical. THEN attempt the MLP via
`SgdDescentMlp`'s quantitative ReLU margins — harder (margins at trained
weights are real numbers to verify exactly, and the per-layer product may
force tiny lr); treat MLP as stretch, linear as the deliverable.

**Done when:** `trained_linear_sgd_descends_concrete` (+ MLP if it closes) in
the audit set; the "satisfiability witness is degenerate" line in the audit
notes gets struck.

## 4. Hygiene batch (one short session)

- **Cast-pattern grep:** `grep -rn ':= by' LeanMlir/Proofs | grep -B2 'rw \[show'`
  style sweep for other structure-returning tactic defs with Eq.mpr-cast
  values (the `mhsa_has_vjp_mat` pathology, memory: kernel-eqmpr-cast). Check
  especially other `*_has_vjp_mat` / `*_has_vjp` defs. Fix = the
  `mhsa_composed_has_vjp_mat` split pattern.
- **Stale heartbeat pragmas:** ViTBackB0/ViTMhsaBackCertifiedTie carry
  10M/4M/1.6M `maxHeartbeats` from the pre-fix era; the files now build in
  seconds — right-size or drop (each test rebuild is ~3s now). Low value,
  do opportunistically.
- **Untracked stragglers:** delete or gitignore `ffi/libiree_ffi.so.apr20-bak`
  (1.3MB binary; stale FFI .so's have burned us twice — memory:
  iree-cuda-bigmodel-hang), decide fate of `runs/verified_imagenette_sweep/`
  logs and `blueprint/src/figures/arch/`.
- **lakefile nit:** duplicated `moreLinkArgs := ireeLink` at
  lakefile.lean:803-804 (`mnist-mlp-pgd`).
- **ViTBackB0 split:** no longer a build-time issue (3.4s); split into
  core/MH/vecLN only when editing that chapter for the book — readability
  call, park until then.

## 5. Separate track — empirical (klawd/mars, unchanged by the proof arc)

- **R50-A3 low-val:** cause isolated to optimization (LAMB@bs512 + wd-on-BN);
  next = SGD+momentum probe (project_r50_a3_lowval_diagnostic). Duty-cycle
  long runs per klawd-thermal memory.
- **Grad-accum lever:** accum=1 byte-identical / accum=4 compiles, but NOT yet
  GPU-run — the sharding constraint is unvalidated. One short klawd session.

## 6. Parked without guilt

- **Full-depth descent** — open BY DESIGN (per-layer operator-norm product
  compounds to vacuity; needs a different proof strategy, e.g. PL/local
  smoothness). Revisit only with a new idea, not more grinding.
- **Verified pretty-lexer** — deliberate stop stands (StableHLOLex §status);
  the CI byte-diff covers the practical risk.
- **Kernel↔model float boundary** — verified-compiler territory; the silicon
  probe script is the mitigation.
- **Muon tuned-quintic interval band** (`∀ σ ∈ [σmin,1], |φ⁵(σ)−1| ≤ δ`) —
  honesty polish, no consumer.
- **Non-product spectral bound** — the Gram lemmas applied to the joint
  masked product `W2·diag(mask)·W1` (10×49) would beat the per-layer product
  ceiling (radius 0.171 → toward the true joint σ₁); mathematically easy
  BUT the mask is input-dependent, so the statement only holds on the
  activation region — needs a region-stability hypothesis to be honest.
  Interesting, subtle; only worth it after §1 shows where the product cert
  actually saturates on capped nets.
