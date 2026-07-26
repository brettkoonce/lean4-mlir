# Trimming the generated scorecards: recipe, gotchas, remaining work

*(2026-07-26. Status: **7 of 8 tiers trimmed** — only the low-value smoothing tier
is left — generated corpus 106k → ~29k lines, every reported number unchanged.
Commits `6bb0313` (conv), `89b98d8` (Lipschitz full-input + IBP L∞), `fba43bb`
(both LipSDP tiers), `6a3b604` (float), `672d518` + this one (pooled base). The
stranding §4 warned about is fixed and the pooled LipSDP tier cut **~19 minutes
off every proof push**. Three findings worth more than the line count: the LipSDP
tiers are **not byte-reproducible** (§2.6); the generators had **drifted from
their own committed output**, one of them by 66 lines including `CertifiedAt`
itself (§2.7 — the single most dangerous thing found here); and trimming
**barely moves elaboration cost** on the tiers whose work is weight-side rather
than exhibit-side (§5, §6).)*

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
* **When another tier measures over this one's images, emit the measurement as
  data.** `LipschitzCertScorecard.lean` now carries a `-- certMargin<net> <i>
  <class> <n>/<d>` comment table covering *all* certified images, which is what
  `lipschitz_cert_float.py` reads. A count over a fixed subset is exact rational
  arithmetic, not a theorem, so a comment is the honest home for it — it costs
  zero elaboration and it means capping the theorems upstream can never silently
  shrink the population a downstream tier reports. Downstream generators should
  read the *table* for counts and the *theorem names* only for what they cite.

## 2. Gotchas (each of these cost real time — read before starting a tier)

1. **Paths rot in both directions — check the ones a generator READS too.**
   *Output:* the Proofs/ bucket refactor updated the `import` lines the
   generators *emit* but not the path they *write to*.
   `lipschitz_cert_scorecard_full.py` — and `lipschitz_cert_pair_sdp_full.py` /
   `lipschitz_cert_scorecard_ibp.py`, which do `OUTDIR = base.OUTDIR` — silently
   wrote to the dead flat path. Nothing catches this: the committed files still
   build, they just can't be reproduced.
   *Input (found later, same class):* `smooth_scorecard_gen.py` read
   `runs/smooth_<slug>_scorecard.csv`, but only the **curated** copies under
   `runs/2026-07-12-smooth-scorecard/` are committed — so on a clean checkout it
   died with "run ./run_smooth_scorecard.sh first", sending you off to re-run a
   ~25 min 2-GPU Monte-Carlo job for data already in the repo. Both smoothing
   generators now resolve via a shared `csv_path()`: fresh run if present, else
   the archive. A generator that can't find its input is *louder* than one that
   writes to the wrong place, but it is the same rot.
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
7. **Generated files contain hand edits the generator doesn't know about —
   regenerating deletes them silently. This was the worst thing found here.**
   All three pooled generators were missing the `**REDUCED CERTIFICATE MODEL**`
   scope disclaimer (this is the 4×4-pooled 49-dim net, *not* the canonical
   784→512→512→10 `mlpVerified`). Far worse, `lipschitz_cert_scorecard.py` was
   missing **66 lines** of its own output — including `CertifiedAt`, *the
   predicate every certificate tier in the repo states its scorecard in*, plus
   `cappedCerts`, `cappedCerts_certified` and `scorecard`. Running that
   generator would have detonated the corpus. All three now emit everything, and
   each was verified byte-identical uncapped before being capped. Check with

   ```
   git show HEAD:<file> | diff - <(regenerated) | grep '^[<>]' | grep -v <numeric>
   ```

   and **read every prose line it reports** rather than assuming prose diffs are
   your own edits. A generator being older than the file it generates is the
   default state, not the exception.

## 3. Done

| tier | generator | before | after |
|---|---|---|---|
| conv IBP | `ibp_conv_scorecard.py` | 7,322 | **1,846** |
| Lipschitz full-input (×4) | `lipschitz_cert_scorecard_full.py` | 32,694 | **7,485** |
| IBP L∞ (SF + TF) | `lipschitz_cert_scorecard_ibp.py` | 11,683 | **3,485** |
| LipSDP full-input (SF + TF) | `lipschitz_cert_pair_sdp_full.py` | 30,884 | **7,241** |
| LipSDP pooled (C + U) | `lipschitz_cert_pair_sdp.py` | 15,933 | **5,251** |
| Float composition | `lipschitz_cert_float.py` | 830 | **440** |
| Pooled base scorecard | `lipschitz_cert_scorecard.py` | 1,539 | **751** |

Build: 3903 jobs clean. `AuditAxioms` 1474/1474, `AuditAxiomsHeavy` 46/46, all
three-axiom. Conv tier alone went 4,241 s → 488 s and ~38 GB → 14.2 GB peak.
The pooled LipSDP tier — the only one of these in the **per-push** `Certs` build
— went **583 s + 671 s → 48.5 s + 49.7 s** (45 → 35 class pairs), i.e. ~19 min
off every proof push, at 6.0 GB peak.

**The float tier is the exception: trimming it is nearly free of benefit.** The
old note here claimed "538 s for 34 float-composed images"; a clean A/B at
`LEAN_NUM_THREADS=1` (2.5 s of that is lake replay) says otherwise:

| `LipschitzCertFloat` | wall | peak |
|---|---|---|
| uncapped (33 images) | 97.9 s | 4.58 GB |
| capped (8 images) | **90.7 s** | 3.79 GB |

The pooled base scorecard behaves the same way: 1,539 → 751 lines (34 → 8
theorem-carrying images, all 34 `img<i>` defs kept) bought **142 s → 103 s** and
5.44 → 4.22 GB. Its remaining cost is `G1s_eq`/`G2s_eq`/`H1s_eq`/`H2s_eq` —
328 `fin_cases` goals over the Gram matrices, engine-side.

For the float tier it is 7 s, ~7%. `-Dprofiler` explains it: **92 s of the ~95 s
is `simp`**, and it is
almost all `W1sV_abs_le` / `W2sV_abs_le` — 49×8 + 8×10 = 472 `fin_cases` goals
each re-unfolding the weight matrices at ~190 ms apiece. The per-image blocks
are ~0.29 s each (49 cheap coordinate goals). So this file's cost is
**weight-side, not exhibit-side**, and no per-image cap can touch it. The cap
was kept anyway — it is 390 fewer lines, it makes the theorem/measurement split
say the same thing as every other tier, and it is what let the §2.7 disclaimer
bug surface — but do not expect time back. The real win here would be replacing
those 472 simp goals with one kernel bound over the ℤ-list weights, the same
trick `ListDot.lean` plays for the 784-term dots.

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
| 1 | Smoothing CP + Dec | 3,598 | `Certs` | `smooth_{,dec_}scorecard_gen.py` | **Lowest value, and the only one left.** One `decide +kernel` bignum check per image (~0.1 s), so the marginal image is already nearly free. Trim only for consistency. **§2.7 check: PASSED** — all 9 outputs (CP, Dec, 6 Dec chunks, NetWitness) regenerate byte-identically, no hand edits, no drift. Only the stale input path of gotcha #1 had to be fixed to get there.

**Do not trim** `LipschitzCertScorecardFullNets.lean` (3,799): that is net weights
plus the Schatten-8 chains, i.e. the proved Lipschitz constant. It is engine-side
content, not a per-image exhibit.

The big levers are spent. The recurring lesson across the last three tiers is
that **line count and elaboration cost are only loosely related**: the float tier
gave back 7 s for a 47% cut, the pooled base 39 s for a 51% cut, because in both
the dominant cost is weight-side `simp` (`W*_abs_le`, `G*_eq`/`H*_eq`) that no
per-image cap touches. If the goal is per-push time rather than repo hygiene, the
next move is not another cap — it is replacing those `fin_cases`-over-the-weights
proofs with kernel bounds over the ℤ-list data, the `ListDot.lean` trick.

## 6. The real prize: does this get `certs-heavy` back under CI?

Partly — and the LipSDP tier is now a measured **no**, for a reason worth
stating precisely.

Measured on mars at `LEAN_NUM_THREADS=1`, one module at a time (`/usr/bin/time -v`,
olean deleted first so the number is the fresh elaboration). **Every module in
`CertsHeavy` now fits**, max 11.42 GB against the ~12 GB working threshold:

| module | before | now | wall |
|---|---|---|---|
| `LipschitzCertScorecardFullImgsA` | 39.5 GB | **11.42 GB** | 67 s |
| `LipschitzCertScorecardFullNets` | — | 10.14 GB | 257 s |
| `LipschitzCertScorecardIBP` | 20.2 GB | 9.19 GB | 112 s |
| `IbpConvScorecardImgsB` | — | 8.63 GB | 326 s |
| `IbpConvScorecardImgsA` | — | 8.43 GB | 314 s |
| `LipschitzCertScorecardIBPUncon` | — | 8.28 GB | 78 s |
| `LipschitzCertScorecardFullImgsB` | — | 4.39 GB | 20 s |
| `LipschitzCertScorecardFull` | — | 3.47 GB | 5 s |
| `IbpConvScorecardNet` | — | 3.10 GB | 72 s |
| `IbpConvScorecard` (aggregate) | — | 2.65 GB | 4 s |
| *(out of the lib roots)* `SDPFull` | 17.08 GB @ 4 thr | **16.0 GB** | 196 s |
| *(out of the lib roots)* `SDPFullUncon` | — | **16.6 GB** | 205 s |

~21 min total, sequential, against a 350-min budget.

**The conv tier needed a split, not a cap.** Capping it (`6bb0313`, 7,322 →
1,846 lines) still left one module at **14.68 GB** — over what a 16 GB runner
carries once the OS is on it. Peak tracks the images *in a module*, because Lean
does not reclaim between declarations in a process, so the fix is more modules:
`IbpConvScorecard` is now `Net` + `ImgsA` + `ImgsB` + aggregate, and the worst
chunk is 8.63 GB. Same reasoning as `SmoothingDecChunk1..6`. Cost: ~716 s vs
489 s wall, since each chunk re-imports the net context — a good trade for
fitting.

`certs-heavy.yml`'s sequential loop now lists **every** `CertsHeavy` module; it
was silently missing `IbpConvScorecard`, which meant the final
`lake build CertsHeavy` elaborated it unserialized.

**Trimming does not lower the LipSDP tier's peak, and cannot.** At one thread
only one goal is in flight, so 16 GB *is the cost of a single `hS*` goal* — one
`linarith` over a 256-monomial slack with ~230-digit LDLᵀ coefficients. Cutting
100 images to 8 cut lines by 77% and wall-clock a lot, but only took 45 pairs to
39, and peak is per-goal, not per-file. You would have to emit zero pairs to move
it. So the cap is not a substitute for the fix in
`certs_heavy_psd_memory.md` §"Re-enable options" #1 (small-coefficient
diagonally-dominant witnesses) — that remains the only route to a green
`certs-heavy` for these two modules, and it attacks exactly the right thing.

So the answer splits cleanly. The two LipSDP full-input modules cannot go under
CI without the `certs_heavy_psd_memory.md` fix — but they are already outside the
lib roots, so they were never what blocked the workflow. **Everything that is
actually in `CertsHeavy` now fits a 16 GB runner**, and the remaining step is a
one-line trigger change: drop `on: workflow_dispatch:` back to the push/weekly
triggers the file was written for. That would make the generated corpus
CI-verified for the first time.

Whatever stays out, the repo should keep saying so plainly — a file carried as
if CI checked it, when only a local run ever did, is the failure mode this whole
exercise is about.

## 7. What this does *not* address

Trimming is hygiene, not capability. It changes how many times the corpus repeats
a claim, not which claims are provable. The things that move that needle are
unchanged and live elsewhere: tighter relaxation (CROWN-style linear bounds on the
existing `BoxSound` seam), the ℤ-list conv evaluator that lifts the 8×8 resolution
ceiling, and extending the float composition past the one pooled net.
