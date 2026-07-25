# Trimming the generated scorecards: recipe, gotchas, remaining work

*(2026-07-25. Status: **5 of 6 tiers trimmed**, generated corpus 106k → ~31k lines,
every reported number unchanged. Commits `6bb0313` (conv), `89b98d8` (Lipschitz
full-input + IBP L∞), plus the two LipSDP passes below. The stranding that §4
used to warn about is fixed, and the pooled tier cut **~19 minutes off every
proof push**. Three findings worth reading before the next one: the LipSDP tiers
are **not byte-reproducible** (§2.6), the committed files contain **hand edits no
generator knows about** (§2.7 — this one is still live for the last two targets),
and trimming **does not lower the full-input tier's peak memory** (§6).)*

## 0. Why

The certificate corpus was ~4k lines of engine carrying ~106k lines of generated
exhibit — a 26:1 ratio — and `certs-heavy.yml` is `workflow_dispatch:`-only, so
the majority of the proof corpus by volume was verified by nobody on a schedule.

The epistemic content of a scorecard decomposes into three things:

1. *the engine is sound* — a theorem, few hundred lines, permanent;
2. *it is non-vacuous at real trained weights* — needs a handful of worked images;
3. *how often it bites* — a **measurement**.

Kernel-checking 100 images spends ~100k lines buying (3) at kernel grade. But
soundness was never in the images: once (1) is proved, image #57 certifying tells
you nothing #56 didn't. So we keep (3) as an exact-rational measurement over the
full subset and cap how many images carry theorems.

**This is not a weakening of any claim.** Every count in RESULTS.md is identical
before and after; what changed is that the file says which numbers are theorems
and which are measurements.

## 1. The recipe

Two independent knobs in each generator:

```python
N_MEASURE   # fixed subset the COUNTS are measured over (exact Fractions, no Lean)
N_EMIT      # how many of those carry a per-image THEOREM
```

* Emission rule: **the first `N_EMIT` certifying images per (net, radius) column**,
  in test-set order — unbiased and reproducible, not a pick of the easy ones.
* **Per column, not global.** A global cap left the TF net at ε=0.3 (measured
  2/100) with an *empty* aggregate — sound but useless, since that column then
  proves nothing. Per-column guarantees every column with a nonzero measured
  count has witnesses.
* The `scorecard*` theorem states **only** the proved counts.
* The file header carries a "Theorem vs. measurement" paragraph naming both, so
  nobody quotes the measured number as if the kernel checked it.

## 2. Gotchas (each of these cost real time — read before starting a tier)

1. **`OUTDIR` was stale (pre-existing bug, now fixed).** The Proofs/ bucket
   refactor updated the `import` lines the generators *emit* but not the path
   they *write to*. `lipschitz_cert_scorecard_full.py` — and
   `lipschitz_cert_pair_sdp_full.py` / `lipschitz_cert_scorecard_ibp.py`, which
   do `OUTDIR = base.OUTDIR` — silently wrote to the dead flat path. Nothing
   catches this: the committed files still build, they just can't be reproduced.
2. **Always regenerate UNCAPPED first and diff.** Matching output against the
   committed corpus is what proves (a) your path fix is right, (b) the generator
   is deterministic, (c) your cap edit introduced no other drift. Do this even
   when it feels ceremonial: on the first three tiers it found nothing, which is
   what made a 44k-line change safe to trust — and on the pooled LipSDP tier it
   found a paragraph the generator had been silently deleting (§2.7). See §2.6
   for the one tier where "byte-identical" is the wrong bar.
3. **The tiers are coupled through per-image facts.** The IBP scorecards reuse
   the base scorecard's `hpre*`/`imgF*`; capping the base stranded 147
   references. Check with
   `comm -23 <(consumed names) <(defined names)` *before* building — it is
   seconds, versus a ~30-minute build that fails at the end.
4. **De-duplicate fallback data across sibling files.** With the base capped,
   both IBP files began emitting their own fallback images and collided
   (`imgF10` declared twice ⇒ importing both into one environment is a hard
   error). The second file now chains onto the first via an extra `import`.
5. **The audits pin specific per-image names** (`certSF10_85`, `certTF10_99`,
   `certIBPTFe2_91`, …). Capping deletes them. Repoint to first/middle/last of
   whatever survives per column.
6. **The LipSDP tier is not byte-reproducible — don't expect gotcha #2 to hold
   there.** Its emitted rationals come from an off-line float solve
   (Nelder-Mead over `sqrtm`/`eigvalsh`), so regenerating on a machine with a
   different scipy/BLAS gives a *different valid certificate*: same declarations,
   same counts, different ρ/T/LDLᵀ numbers. Regenerating the committed
   pre-cap files here rewrote 19 of 45 pairs (SF) and 4 of 45 (TF) with no
   structural change at all. The substitute check, which *did* pass:

   ```
   identical declaration list  +  identical measured counts
   ```

   i.e. `diff <(grep -oE '^(theorem|noncomputable def|def) [A-Za-z0-9_]+' old) <(… new)`
   is empty. The kernel-dotZ tiers have no such float step and stay byte-exact.
7. **Generated files can contain hand edits the generator doesn't know about —
   regenerating deletes them silently.** The pooled files carry a
   `**REDUCED CERTIFICATE MODEL**` scope disclaimer (this is the 4×4-pooled
   49-dim net, *not* the canonical 784→512→512→10 `mlpVerified`) that was added
   by hand and was in **none** of the three pooled generators. The uncapped diff
   is what caught it; `lipschitz_cert_pair_sdp.py` now emits it. **Still missing
   from `lipschitz_cert_scorecard.py` and `lipschitz_cert_float.py`** — add it
   there before regenerating either, or you will quietly drop the one paragraph
   that stops a reader taking these counts for the canonical net. Check with

   ```
   git show HEAD:<file> | diff - <(regenerated) | grep '^[<>]' | grep -v <numeric>
   ```

   and read every prose line it reports, rather than assuming prose diffs are
   your own edits.

## 3. Done

| tier | generator | before | after |
|---|---|---|---|
| conv IBP | `ibp_conv_scorecard.py` | 7,322 | **1,846** |
| Lipschitz full-input (×4) | `lipschitz_cert_scorecard_full.py` | 32,694 | **7,485** |
| IBP L∞ (SF + TF) | `lipschitz_cert_scorecard_ibp.py` | 11,683 | **3,485** |
| LipSDP full-input (SF + TF) | `lipschitz_cert_pair_sdp_full.py` | 30,884 | **7,241** |
| LipSDP pooled (C + U) | `lipschitz_cert_pair_sdp.py` | 15,933 | **5,251** |

Build: 3903 jobs clean. `AuditAxioms` 1474/1474, `AuditAxiomsHeavy` 46/46, all
three-axiom. Conv tier alone went 4,241 s → 488 s and ~38 GB → 14.2 GB peak.
The pooled LipSDP tier — the only one of these in the **per-push** `Certs` build
— went **583 s + 671 s → 48.5 s + 49.7 s** (45 → 35 class pairs), i.e. ~19 min
off every proof push, at 6.0 GB peak.

Regenerating uncapped is `SCORECARD_N_EMIT=100 python3 scripts/<gen>.py`. The
knob is defined twice, because the tiers do not share a base: in
`lipschitz_cert_scorecard_full.py` (the full-input family — the SDP and IBP
generators import it and read `base.N_EMIT`) and in `lipschitz_cert_pair_sdp.py`
(the pooled tier, which trains its own nets). Both default to 8.

## 4. Done: the SDP full-input files, un-stranded

`lipschitz_cert_pair_sdp_full.py` reads `base.need` exactly like the IBP one, so
capping the base had broken the committed `LipschitzCertScorecardSDPFull{,Uncon}.lean`
(81 + 79 dangling `imgF<i>`, 147 dangling `hpre{SF,TF}<i>`) — invisible to CI,
because those two modules are in no lib root. Regenerated capped, which fixes it:

* 30,884 → **7,241** lines; 45 → 39 class pairs needing an LDLᵀ PSD witness;
* measured counts unchanged — SF 93/100 @ ε=0.1 (= the PGD bound) and 91/100
  @ 0.3, TF 91/77 — so RESULTS.md and README need no edit;
* the emitted witnesses are images 0–7 for both nets at both radii, and
  `scorecard_sdp_full{,_uncon}` now states those proved counts;
* only one fallback fact was needed (`hpreTF1`: image 1 is not in the base's
  TF-certified set), so **no `imgz` fallback at all** and no collision with the
  IBP tier's fallbacks. Checked globally — across all eight generated scorecard
  files every declared name is unique, so the four tiers can share one
  environment. `SDPFullUncon` still chains onto `SDPFull` to keep that true if
  the emitted sets ever shift.

Both modules build clean at `LEAN_NUM_THREADS=1` (3:16 / 3:25) and all 14 audit
pins are three-axiom. They stay out of the lib roots — see §6 for why.

## 5. Remaining targets, ranked

| # | target | lines | lib | generator | notes |
|---|---|---|---|---|---|
| 1 | `LipschitzCertFloat` | 830 | `Certs` | `lipschitz_cert_float.py` | 538 s for 34 float-composed images; 8 would do. Cheap, per-push win. **Missing the §2.7 disclaimer — add it before regenerating.** |
| 2 | `LipschitzCertScorecard` (pooled) | 1,539 | `CertsHeavy` | `lipschitz_cert_scorecard.py` | Small, but it is the base several others reuse — cap it *last* or you re-strand its dependents. Its `img<i>`/`hpreC<i>` set is **hardcoded** into `lipschitz_cert_pair_sdp.py` as `EXISTING_IMGS`/`EXISTING_HPRE_C` (not imported), so capping it silently desynchronizes that generator — update both together. **Also missing the §2.7 disclaimer.** |
| 3 | Smoothing CP + Dec | 3,598 | `Certs` | `smooth_{,dec_}scorecard_gen.py` | **Lowest value.** These are one `decide +kernel` bignum check per image (~0.1 s), so the marginal image is already nearly free. Trim only for consistency. |

**Do not trim** `LipschitzCertScorecardFullNets.lean` (3,799): that is net weights
plus the Schatten-8 chains, i.e. the proved Lipschitz constant. It is engine-side
content, not a per-image exhibit.

What is left is small and cheap — the two big levers (the stranded full-input
tier and the per-push pooled tier) are both spent. Item 2 is the one with a real
trap in it (the hardcoded cross-generator image set); items 1 and 3 are routine.

## 6. The real prize: does this get `certs-heavy` back under CI?

Partly — and the LipSDP tier is now a measured **no**, for a reason worth
stating precisely.

Measured on mars at `LEAN_NUM_THREADS=1`, one module at a time (`/usr/bin/time -v`,
olean deleted first so the number is the fresh elaboration):

| module | before | after the cap |
|---|---|---|
| `LipschitzCertScorecardIBP` | 20.2 GB | **9.19 GB** (1:52) |
| `IbpConvScorecard` | ~38 GB | **14.2 GB** |
| `LipschitzCertScorecardSDPFull` | 17.08 GB @ 4 threads | **16.0 GB** (3:16) |
| `LipschitzCertScorecardSDPFullUncon` | — | **16.6 GB** (3:25) |

**Trimming does not lower the LipSDP tier's peak, and cannot.** At one thread
only one goal is in flight, so 16 GB *is the cost of a single `hS*` goal* — one
`linarith` over a 256-monomial slack with ~230-digit LDLᵀ coefficients. Cutting
100 images to 8 cut lines by 77% and wall-clock a lot, but only took 45 pairs to
39, and peak is per-goal, not per-file. You would have to emit zero pairs to move
it. So the cap is not a substitute for the fix in
`certs_heavy_psd_memory.md` §"Re-enable options" #1 (small-coefficient
diagonally-dominant witnesses) — that remains the only route to a green
`certs-heavy` for these two modules, and it attacks exactly the right thing.

The IBP tier is a different story: 9.19 GB fits a 16 GB runner with room. If the
conv tier's 14.2 GB holds up under a fresh per-module measurement, everything in
`CertsHeavy` *except* the two LipSDP modules can come off `workflow_dispatch:`
today — which is most of the corpus, and they are already the two modules that
are out of the lib roots. That is the next thing to try.

Whatever stays out, the repo should keep saying so plainly — a file carried as
if CI checked it, when only a local run ever did, is the failure mode this whole
exercise is about.

## 7. What this does *not* address

Trimming is hygiene, not capability. It changes how many times the corpus repeats
a claim, not which claims are provable. The things that move that needle are
unchanged and live elsewhere: tighter relaxation (CROWN-style linear bounds on the
existing `BoxSound` seam), the ℤ-list conv evaluator that lifts the 8×8 resolution
ceiling, and extending the float composition past the one pooled net.
