# Handoff — Lean 4.34.0 bring-over (branch `lean-4.34`)

Written 2026-09-16. Everything here is measured, not assumed. If you are picking this up cold,
read §1 and §6 first.

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
