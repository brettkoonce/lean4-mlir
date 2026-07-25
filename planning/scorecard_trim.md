# Trimming the generated scorecards: recipe, gotchas, remaining work

*(2026-07-25. Status: **3 of 6 tiers trimmed**, generated corpus 106k → 67k lines,
every reported number unchanged. Commits `6bb0313` (conv) and `89b98d8` (Lipschitz
full-input + IBP L∞) on `cap-scorecards`. One consequence needs immediate
attention: the SDP full-input files are now **stranded** — see §4.)*

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
2. **Always regenerate UNCAPPED first and diff.** Byte-identical output against
   the committed corpus is what proves (a) your path fix is right, (b) the
   generator is deterministic, (c) your cap edit introduced no other drift. This
   caught nothing bad this round, which is the point — it made the 44k-line
   change safe to trust.
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

## 3. Done

| tier | generator | before | after |
|---|---|---|---|
| conv IBP | `ibp_conv_scorecard.py` | 7,322 | **1,846** |
| Lipschitz full-input (×4) | `lipschitz_cert_scorecard_full.py` | 32,694 | **7,485** |
| IBP L∞ (SF + TF) | `lipschitz_cert_scorecard_ibp.py` | 11,683 | **3,485** |

Build: 3903 jobs clean. `AuditAxioms` 1474/1474, `AuditAxiomsHeavy` 46/46, all
three-axiom. Conv tier alone went 4,241 s → 488 s and ~38 GB → 14.2 GB peak.

## 4. ⚠ Immediate: the SDP full-input files are stranded

`lipschitz_cert_pair_sdp_full.py` reads `base.need` exactly like the IBP one, so
capping the base broke the committed `LipschitzCertScorecardSDPFull{,Uncon}.lean`:

```
84  dangling imgF<i> references
163 dangling hpre{SF,TF}<i> references
```

This is **invisible to CI** because those two modules are in no lib root (removed
per `certs_heavy_psd_memory.md`), so nothing builds them. They are now stale on
disk. Either regenerate them capped (§5, item 1) or, if the LipSDP tier is being
retired anyway, delete them — but do not leave them as-is, because the next person
to re-enable that workflow will hit 247 unresolved names and no explanation.

## 5. Remaining targets, ranked

| # | target | lines | lib | generator | notes |
|---|---|---|---|---|---|
| 1 | `SDPFull{,Uncon}` | 30,884 | *(none)* | `lipschitz_cert_pair_sdp_full.py` | **stranded, §4.** Biggest win. Same `base.need` coupling as IBP — regenerate in one pass. Verifying costs 17–22 GB/file. |
| 2 | `SDP{,Uncon}` (pooled) | 15,933 | **`Certs`** | `lipschitz_cert_pair_sdp.py` | In the **per-push** tier: 583 s + 671 s off every proof push. Has its own `EXISTING_HPRE_C` reuse — same coupling check applies. |
| 3 | `LipschitzCertFloat` | 830 | `Certs` | `lipschitz_cert_float.py` | 538 s for 34 float-composed images; 8 would do. Cheap, per-push win. |
| 4 | `LipschitzCertScorecard` (pooled) | 1,539 | `CertsHeavy` | `lipschitz_cert_scorecard.py` | Small, but it is the base several others reuse — cap it *last* or you re-strand its dependents. |
| 5 | Smoothing CP + Dec | 3,598 | `Certs` | `smooth_{,dec_}scorecard_gen.py` | **Lowest value.** These are one `decide +kernel` bignum check per image (~0.1 s), so the marginal image is already nearly free. Trim only for consistency. |

**Do not trim** `LipschitzCertScorecardFullNets.lean` (3,799): that is net weights
plus the Schatten-8 chains, i.e. the proved Lipschitz constant. It is engine-side
content, not a per-image exhibit.

Suggested order: **1 → 2 → 3 → 4**, then 5 only if it bothers you. Item 1 is
forced by §4; item 2 is the one that speeds up everyday work.

## 6. The real prize: does this get `certs-heavy` back under CI?

Open question, and worth measuring once items 1–2 land. The tier was disabled
because measured peaks were ImgsA 39.5 GB / IBP 20.2 GB / SDP 17.1 GB against a
16 GB runner. After trimming, the conv tier peaks at 14.2 GB and the whole
`Certs`+`CertsHeavy` build ran at 9.4 GB — but peak is per-module and needs
re-measuring per file with `LEAN_NUM_THREADS=1`.

If every module lands under ~12 GB, flip `certs-heavy.yml` off
`workflow_dispatch:` and the 67k-line corpus becomes CI-verified for the first
time. That is worth more than any individual scorecard count: right now those
numbers are re-checked only when someone remembers to run them locally.

If it still doesn't fit, the honest conclusion is that these files are local
artifacts and the repo should say so plainly rather than carrying them as if
they were checked.

## 7. What this does *not* address

Trimming is hygiene, not capability. It changes how many times the corpus repeats
a claim, not which claims are provable. The things that move that needle are
unchanged and live elsewhere: tighter relaxation (CROWN-style linear bounds on the
existing `BoxSound` seam), the ℤ-list conv evaluator that lifts the 8×8 resolution
ceiling, and extending the float composition past the one pooled net.
