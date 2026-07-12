# The LipSDP PSD witnesses vs 16 GB runners: post-mortem + re-enable plan

*(2026-07-12. Status: `LipschitzCertScorecardSDPFull{,Uncon}.lean` are
**CI-disabled** — removed from the `CertsHeavy` lib roots, their imports and
`#print axioms` lines commented in `tests/AuditAxiomsHeavy.lean`, skipped in
`.github/workflows/certs-heavy.yml`. The files remain in the repo and are
kernel-verified locally: capped σ≤2 **93/100 @ L2 ε=0.1 = the PGD bound —
sandwich closed** — 91/100 @0.3; uncon 91/77. Nothing about the RESULTS
changed; only who re-checks them on a schedule.)*

## What happened (4 CI attempts, all forensics real)

The per-pair LipSDP certificates discharge PSD-ness of
`S = 2·diag(T) − vvᵀ − (1/ρ)·T·G₁·T` by `linarith` from the LDLᵀ column
squares. At h=16 the exact-LDL elimination fractions grow to **~230-digit
denominators**, and the expanded 256-monomial slack goal + 16 giant-coefficient
hints make each `hS*` goal a multi-GB object during elaboration:

| attempt | config | outcome |
|---|---|---|
| 1 | lake default (-j≈4), `LEAN_NUM_THREADS=2` | runner killed mid-build @ ~75 min (OOM) |
| 2 | `lake build -j1` | fails in 2 min — **Lake 5.0.0 has no `-j`/`--jobs` flag at all** |
| 3 | sequential per-module `lake build` calls, 2 threads | single lean process OOM-killed @ ~20 min |
| 4 | sequential, `LEAN_NUM_THREADS=1`, **10 G swapfile**, 350-min budget | (in flight at time of writing; regardless of outcome the SDP modules are now out of the CI set) |

Local measurement: **17.08 GB peak RSS** for ONE SDP module at 4 threads
(`/usr/bin/time -v`, mars). GitHub free runners: 4 vCPU / 16 GB. The
alternative "deterministic" entrywise route (`lipsdp_slack_of_cert`) is
memory-light but **7.5× the CPU** (413 vs 55 CPU-min/file measured) — it
trades OOM for a guaranteed timeout on 4 cores. Neither fits.

## Re-enable options, ranked

1. **Small-coefficient PSD witnesses (RECOMMENDED — fixes it at the source).**
   Replace the exact LDLᵀ by a `/2^k`-grid-rounded factor plus a remainder
   shown positive-semidefinite by diagonal dominance:
   `S = L'·diag(d')·L'ᵀ + R`, with `R` symmetric and
   `R a a ≥ Σ_{b≠a} |R a b|` — all entries SMALL rationals, checked entrywise
   by `norm_num`, no linarith, no big fractions ⇒ low memory AND low CPU.
   - New core lemma `quad_form_nonneg_of_dd` (symmetric + diagonally dominant
     ⇒ `zᵀRz ≥ 0`, via `2|zₐz_b| ≤ zₐ² + z_b²`): a **full draft proof already
     exists** — it was written this session and parked when priorities moved;
     the skeleton: termwise bound
     `ite (b=a) (Rₐₐzₐ²) (−|Rₐ_b|(zₐ²+z_b²)/2) ≤ zₐ(Rₐ_b z_b)`, sum, swap the
     second cross-half by symmetry (`Finset.sum_comm` + `hsym`), collect to
     `Σₐ (Rₐₐ − Σ_{b≠a}|Rₐ_b|)·zₐ² ≥ 0`. Companion
     `lipsdp_slack_of_cert_split` = existing `lipsdp_slack_of_cert` with
     `h2 : Sm a b = (Σᵢ L'ₐᵢ·(d'ᵢ·L'_bᵢ)) + R a b` and the DD side conditions.
   - Generator side (`scripts/lipschitz_cert_pair_sdp_full.py`): compute the
     float LDL of `S − ηI`, round `L', d'` to the grid, compute `R = S −
     L'd'L'ᵀ` exactly in `Fraction`s, VERIFY diagonal dominance in python
     before emitting (bump ρ/η and retry if it fails, like the current
     `ldl_exact` bump loop).
   - Estimated effort: half a day. Also likely makes the files fast enough
     to move BACK into plain `CertsHeavy` sequential builds with headroom.
2. **Self-hosted runner on mars** (24 cores / 188 GB, this exact corpus
   builds here in ~8 min wall). Needs a runner token + daemon (Brett action);
   the workflow already never runs on fork PRs. Keep `runs-on:
   [self-hosted, mars]` for certs-heavy only.
3. **Paid larger runner** (8-core/32 GB): smallest diff, costs money.
4. **Status quo** (what this doc's state implements): SDP instances verified
   locally on every regeneration; CI re-checks the scorecard + IBP tiers
   weekly. Honest, zero cost — the audit tables in RESULTS.md remain true,
   they're just not cron-re-derived.

## POST-SCRIPT (same day): it was never just the SDP files

Run 5 (SDP-disabled, sequential, 2 threads, no swap) ALSO killed the runner
at ~63 min. Local measurements at `LEAN_NUM_THREADS=2` (`/usr/bin/time -v`):
**ImgsA 39.5 GB**, **IBP 20.2 GB** peak RSS — the ENTIRE generated tier is
over the free-runner ceiling, and `LEAN_NUM_THREADS` demonstrably does not
bound Lean 4.31's parallel-elaboration memory (the snapshots/env retention
across thousands of decls dominates). The 188 GB dev box masked all of this.

Consequence: `certs-heavy.yml` is now **workflow_dispatch-only** (never
auto-runs, never red). The generated corpus is kernel-verified locally on
every regeneration — that remains a real check; what's lost is only the
cron re-derivation. Re-enable = self-hosted runner on mars (now clearly the
front-runner — the DD-split fix alone would NOT save the Imgs/IBP files,
whose cost is decide/env-driven, not fraction-driven).

## Also relevant

- The pooled (h=8) SDP files are FINE in `Certs`/CI — small fractions.
- The IBP + base scorecard heavy files stay CI-checked (attempt-4's
  1-thread + swap config is kept as insurance for the IBP modules).
- The measured recipe table (linarith vs entrywise, both widths) lives in
  the 9cd03aa commit message and `full-input-scorecard` session memory.
