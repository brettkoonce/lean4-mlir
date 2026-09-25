# Handoff — Lean 4.34.0 bring-over (branch `lean-4.34`)

## 0. RESOLVED 2026-09-18 — read this first; §1–§7 are the 2026-09-16 state

Everything below §0 is the handoff as written, kept because its ruled-out list still holds.
The blocker and two regressions §7 had not reached are fixed; every CI gate that can run
locally is green on 4.34.0.

**The cause** of items 1–3 was not depth or Mathlib: it was the KERNEL re-deriving a
definitional step across the chain. Item 4 is a separate memory regression.

1. **TieB apex** — `_` for each level's inner map elaborates to the composed chain, so every
   witness sat at the chain applied to `x` while the saved activations are stage-by-stage
   `cnxSavedB_k B w x`. Identifying the two, alone as an `rfl`, does not finish at level 3 on
   EITHER toolchain (killed at 250 s on 4.32.2); 4.32.2 only got through the apex by some path
   4.34 no longer takes. Fix: `cnxSavedB_k` point-free `abbrev`s (`batchMap B stage ∘ cnxSavedB_{k-1} B w`)
   named as each level's inner map — apex 2 s.
2. **TieB tie** — `simp only [Function.comp_apply, convNextForwardTChB_has_vjp_at,
   vjp_comp_diff_at_fst_backward]` is pure `dsimp` (all three are `rfl`), so simp recorded no
   step and the kernel unfolded the whole net under the witnesses' `.backward`s: 40+ GB. Fix:
   the same steps as `rw`s. Whole module: timeout → 2.6 s, 2.7 GB.
3. **Per-example Tie** (`ConvNeXtWholeBackCertifiedTie.lean`, never failed, so the 09-16 run did
   not see it) — its closing `simp only [Function.comp_apply, cnxV0]`, the same hazard:
   17 s / 6 GB on 4.32.2, **6 min / 48 GB** on 4.34.0 (a 16 GB CI runner OOMs). Fix: `rw`. 2.4 s.
4. **`IbpConvScorecardImgsA/B`** (CertsHeavy, never built on 4.34 before) — 8.22 GB on 4.32.2,
   **16.11 GB** on 4.34.0, same olean and CPU-seconds: 4.34 elaborates a module's theorem proofs
   as concurrent tasks and holds twice the memory while they run. `Elab.async false` → 7.10 GB
   but serial, 29 min a chunk. Fix: the generator now emits four 2-image chunks
   (`scripts/certs/ibp_conv_scorecard.py`, `N_CHUNKS`), 9.1–9.7 GB each; `certs-heavy.yml` lists all four.

Measured against 4.32.2 and not regressed: the twelve slowest `Certs` modules (StableHLO, the
Lipschitz/Smoothing scorecards, the R34 ties, TrainedCnn*), and the CertsHeavy FullImgs/FullNets/IBP
peaks (all at or under their recorded 4.32 numbers).

The rest of the bring-over — manifest regenerated with `-Kenv=dev` (doc-gen4 pinned to
`v4.34.0`; plain `lake update` had dropped it), `jax/` manifest, lean4export pin, 4.34's new
lints (haveI, unused pattern variables, deprecated `dropRight`/`trim`/`asString`/
`Mathlib.Data.Real.Basic`, dead `convert … <;>` workarounds) — is in the commit.

## 1. State in one paragraph

Lean **4.34.0 shipped 2026-09-14**. This branch carries the whole bring-over off `origin/main`.
**`lake build Proofs` is GREEN on final 4.34.0 (0 errors, 0 deprecation warnings).**
**`lake build Certs` is blocked by exactly ONE module** —
`LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTieB.lean` — which is a **confirmed 4.34
regression** (it builds on 4.32.2). A bisected repro is in `planning/repro_434_kernel_timeout/`.
Nothing is merged to `main`; `main` already has the *separate*, 4.33-era fix landed and pushed.

## 2. What is on this branch

| change | scale |
|---|---|
| 3 toolchain pins + Mathlib tag → `v4.34.0` | `lean-toolchain`, `jax/lean-toolchain`, `tests/comparator/lean-toolchain`, `lakefile.lean:23` |
| `if_pos`/`if_neg`/`dif_pos`/`dif_neg` → `ite_eq_left`/`ite_eq_right`/`dite_eq_left`/`dite_eq_right` | 645 lines, 0 residual |
| `Set.mem_setOf_eq` → `Set.mem_ofPred_eq`, `Set.setOf_forall` → `Set.ofPred_forall` | 20 sites |
| the repro package | `planning/repro_434_kernel_timeout/` |

⚠ Both renames were verified **character-identical in statement** before swapping. ⛔ None of them
can be backported: the new names exist in neither Lean 4.32.2/4.33.x core nor Mathlib v4.32.2.

`lake update` resolved cleanly: mathlib `v4.34.0`, Cli `v4.34.0`, and every transitive dep bumped.

## 3. The blocker

`convNextForwardTChB_has_vjp_at` (the twelve-stage batched VJP apex, eleven nested
`vjp_comp_diff_at`s) fails with `(kernel) deterministic timeout`. The three
`unknown constant 'Proofs.convNextForwardTChB_has_vjp_at'` errors that follow are **cascade** — the
decl never entered the environment — not separate failures.

Bisection (truncate the apex to k levels above the stem pair):

| levels | 4.32.2 | 4.34.0 |
|---|---|---|
| 2 | builds 4s | builds 2s |
| **3** | **builds 2s** | **kernel timeout ~162s** |
| 4 | builds 2s | kernel timeout 139s |

A cliff, not a gradient, and 4.32.2 handles k=3 **and** k=4 in 2s. Reproduced 3× (165/153/139s),
always at the same site. It is therefore not the particular stage (`convNextStageChK 3 w.s2`).

## 4. ⛔ Ruled out by measurement — do not re-run these

* **Raising `maxHeartbeats`.** At 4000000 the module ran 37+ min climbing monotonically to
  **86.8 GB RSS** with no result, vs ~18 min and modest memory on 4.32.2. It is a blow-up in kernel
  reduction work, not a small budget. (`maxHeartbeats` *is* the right knob — `Lean/Message.lean:890`
  maps `deterministicTimeout` to that message and `Lean/CoreM.lean` registers no other budget — it
  just does not help.)
* **A Mathlib-free repro.** A skeleton (dependent `V n := Fin n → Nat`, structure with function +
  proof fields, composition combinator, symbolic `B * 96` indices) compiles **instantly on both
  toolchains at depth 5, 11 and 16**. Depth + symbolic batch indices are not sufficient; the Mathlib
  half (`DifferentiableAt ℝ`, `pdiv = fderiv ℝ`, the `∑ j : Fin n` in `HasVJPAt.correct`) is
  load-bearing. The repro needs the project cone.
* **Replacing the `rw`s with `simp only`** in the *other* (4.33-era) regression — fails with
  `simp made no progress` at all 6 sites.

## 5. Suggested next step

Structural split: make no single declaration carry all eleven levels (e.g. name the lower half as
its own `def` and compose two halves). That is the analogue of the `@[reducible] t3Idx` move that
fixed the 4.33-era regression in one line. Also worth filing upstream — Lean core rc2→4.34.0 final
is only 3 backport commits (InfoTree/editor + mimalloc), so this is present throughout 4.34.

## 6. Gotchas that cost real time

* **`lake` has no parallelism flag.** `-j` errors (`unknown short option`), and `--jobs` does not
  exist either. `nice`/`ionice` is the only lever.
* **Never key build pass/fail on "stderr is non-empty."** Lean emits warnings on success; a
  warning-only run looks like a failure. Key strictly on `error:`.
* **Lean cannot checkpoint mid-module.** A `timeout N` slice shorter than a module's build time can
  never finish it — repeated slices return `built=0` forever. Use a detached run:
  `setsid nohup nice -n 19 ionice -c3 lake build <target> > log 2>&1 < /dev/null &`
  (harness-tracked background tasks get reaped under low free memory; detached ones do not).
* **`set_option … in` must precede the docstring**, or the docstring detaches from the decl.
* **The batched witnesses (`cnxStemB_at`, `cnxStageB_at`, `cnxSavedB*`) live INSIDE `TieB`**, not in
  its import cone — truncate in place, a standalone importer cannot see them.
* **Stale oleans.** This worktree was first built on the rc2-era flat layout, so `.lake` holds ~123
  orphaned oleans from before `main`'s `Proofs/Nets/<family>/` reorg. A naive `find` counts 349
  against 243 real sources. Coverage must be measured against `git ls-files`.
* **`Codegen/IRPrint` is in neither `Certs` nor `CertsHeavy`** (0 roots, 0 importers) — it must not
  count against coverage. 14 of the other naively-"missing" modules are `CertsHeavy` roots.
* ⛔ **Never `lake clean`** in either worktree — `.lake/build` holds tens of GB of training
  checkpoints.
* ⛔ **Do not merge the old `lean-4.34-rc2` branch.** It forked before `main`'s cleanup and overlaps
  32 files. It is reference only; this branch supersedes it.

## 7. Not yet done

* `CertsHeavy` has never been built on any 4.34 (it has its own 350-min workflow).
* `jax/` and `tests/comparator/` have their own workspaces; their toolchain pins are bumped here but
  their `lake-manifest.json` still pins mathlib `v4.32.2` — they need their own `lake update`.
* CI is untouched; `.github/workflows/` still assumes 4.32.2 caches.
