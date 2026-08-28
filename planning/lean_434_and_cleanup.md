# Lean 4.34.0-rc2 upgrade + the linter / Mathlib-dedup pass

Session 2026-08-28. Two independent tracks, on two branches, both **uncommitted**.

| branch | worktree | what |
|---|---|---|
| `lean-4.34-rc2` | `../verify-v2-434` (own `.lake`) | the toolchain bump, isolated so main's 4.32.2 build is untouched |
| `lint-cleanup` | main worktree | deprecations, unused bindings, import narrowing |

⛔ Never `lake clean` in either tree — `.lake/build` holds **35 GB of training checkpoints**
(`*_ckpt_xla.bin*`) alongside the 13.5 GB of Lean artifacts.

---

## 1. Lean 4.34.0-rc2

Getting there is easy: `lake update` alone did it (Mathlib's post-update hook auto-ran
`cache get`, 8747 files). The 3735-line `lakefile.lean` elaborates on 4.34. The C FFI is clean —
every core `lean.h` symbol the repo uses still exists, and `ffi/f32_helpers.c` + `ffi/lowerer.c`
compile against the 4.34 headers with `-fsyntax-only`.

The whole branch is **4 files**: `lean-toolchain`, the Mathlib tag in `lakefile.lean`,
`lake-manifest.json`, and 2 lines of `StableHLO.lean`. Only `lakefile.lean` overlaps
`lint-cleanup`, on different lines — trivial merge.

### ✅ Result

**`lake build Proofs` on 4.34.0-rc2: 2239 jobs, 0 errors**, at the file's original
`maxHeartbeats 2000000`. 422 warnings, all benign: 378 deprecations (the `if_pos`/`if_neg` family
below), 33 `unusedVariables` + 17 `if_true`/`if_false` that `lint-cleanup` already fixes on its own
branch, 8 `linter.style.haveILetI`, 3 `linter.unusedSimpArgs`.

### The one proof that breaks

`LeanMlir/Proofs/Codegen/StableHLO.lean`, `cnnBackGraph_faithful` (~line 4400):

```
rw [hasVJPAt_backward_det _ (maxPoolFlat_has_vjp_at' … h_mp)]
  -- 4.34: "did not find an occurrence of the pattern HasVJPAt.backward ?m ?dy"
```

The first read — "4.34's `simp only` already reduced the maxpool backward, so the rewrite has
nothing to hit" — was **wrong**. The real cause is one level up, and 4.34's own
`linter.unusedSimpArgs` names it: it flags `id_eq` in that same `simp only` list.

### What actually blocks it: an unreduced `id` wrapper

The heartbeat budget is a red herring. Two dead ends, both informative:
`try rw` (skip the rewrite) → `rfl` **times out** at `whnf`; budget raised to `8000000` → `rfl`
runs to completion and **genuinely fails**. So the rewrite is load-bearing; it simply cannot find
its pattern.

The `rfl` failure shows why. The RHS still reads:

```
(id { backward := fun dy idx => …, correct := … }).backward dy
```

The `id` wrapper is never stripped, so the structure projection cannot iota-reduce, and everything
beneath it stays frozen: `Mat.mulVec` never fires, and the maxpool witness stays as the
`Eq.mpr`-cast `maxPoolFlat_has_vjp_at` instead of the cast-free `'` version — which is precisely
why `rw [hasVJPAt_backward_det …]` finds no `HasVJPAt.backward ?v ?dy`. One blocked reduction,
three downstream symptoms. 4.34's `linter.unusedSimpArgs` names it directly: it reports `id_eq`
— which *is* in the `simp only` list — as **unused**.

**Fix: add `id` itself to the `simp only` list**, so it delta-unfolds rather than relying on
`id_eq` to fire. The plain `rw` and the original `maxHeartbeats 2000000` then both work. No budget
bump, no `try`. The entire proof-side diff for the whole upgrade is that one word.

#### ⚠ A tempting root cause that I could NOT confirm — don't repeat my mistake

`id` did change between the two toolchains (`Init/Prelude.lean:131`):

| | |
|---|---|
| 4.32.2 | `@[inline] def id` |
| 4.34.0-rc2 | `@[inline, implicit_reducible] def id` |

and `implicit_reducible` went from 127 occurrences / 39 files to **336 / 50** — 285 newly-marked
declarations including `cast`, `append`, `bind`, `abs`, `Bool.and`, `Array.push`, `ByteArray.*`.
The story writes itself: `id a` is now defeq to `a` at reducible transparency, simp never *needs*
`id_eq`, the term stays syntactically wrapped, the projection stops reducing.

**That story is not established.** Two minimal controls — a structure behind `id` with
`simp only [def, id_eq]`, and an indexed structure with a function-valued field projected under a
binder — both close cleanly on 4.32.2 *and* 4.34.0-rc2. So the attribute change is real but is
**not demonstrated** to be the mechanism here; something more specific about this goal is doing it.
What is verified is the observation (the wrapper survives, `id_eq` is reported unused) and that
unfolding `id` fixes it.

#### ⛔ And a prediction of mine that was WRONG

From the `id_eq` analysis I concluded the `Certs` risk was "small and enumerable": 6 `id_eq` sites,
of which `Foundation/IR.lean:712` already built clean and the two other `StableHLO.lean` sites
(2967, 3008) elaborated fine, leaving only `Float/ViTWholeFloatBridge.lean:138` and
`Foundation/MuonNewtonSchulz.lean:344` to check.

Building those two on 4.34 refuted it. `MuonNewtonSchulz` built **fine**. The failure came from a
module neither of them names — `Training/SgdDescentCnn.lean` — with **8 errors** in two clusters
(1330–1345, 7073–7088), and they are not `id_eq` at all. They are
`rw [if_pos …]` / `rw [if_neg …]`, i.e. the *deprecated* lemmas. The 338 deprecation warnings are
therefore **not just noise**: the same lemmas are also failing to rewrite.

The failures are instance-unification, not missing terms — the pattern is demonstrably in the goal:

```
pattern:  if t3Idx co hi' wi' = t3Idx co hi wi then ?m.326 else ?m.327
target:   … * (if t3Idx co hi' wi' = t3Idx co hi wi then 1 else 0) = 0
```

`if_pos`/`if_neg` have identical statements in both toolchains, so the lemma is not the variable —
the `Decidable` instance is. (`decEq` and `Decidable.decide` are both in the 285 newly
`implicit_reducible` declarations. Suggestive; **not** proven, same caveat as above.) Note the
first `rw [if_pos rfl]` at line 1325 *succeeds*; only the ones inside the `Finset.sum_eq_single`
side-goals fail.

The deprecation migration has been applied repo-wide on the `lean-4.34-rc2` branch
(`if_pos`→`ite_eq_left`, `if_neg`→`ite_eq_right`, `dif_pos`→`dite_eq_left`,
`dif_neg`→`dite_eq_right`): **53 files, 639 lines, 0 residual**. `Proofs` stayed green through it
and deprecation warnings fell 378 → 21 (the 21 are `if_true`/`if_false`, which `lint-cleanup`
already fixes, so the two branches merged land at **zero**).

⛔ **It did NOT fix `SgdDescentCnn`** — byte-identical errors, same 8 lines. Which confirms the
lemma *name* is not the variable; the `Decidable` instance is. `basisVec` is `@[reducible]` and
unfolds to `fun k => if k = i then (1:ℝ) else 0` over `Fin`, so the instance in the goal and the
one synthesised while elaborating `ite_eq_right H` are defeq but evidently no longer syntactically
equal to `rw`'s keyed matcher.

#### ⭐ Root cause, with a standalone minimal reproduction

`t3Idx_def` is a **folding** lemma — `finProdFinEquiv (finProdFinEquiv (ci, hi), wi) = t3Idx ci hi wi`
— so `simp only [t3Idx_def]` (line 1322) folds the raw encoding *into* `t3Idx` in the `ite`'s
**condition**. But simp does not rewrite inside a `Decidable` **instance** argument. The goal's
`ite` therefore ends up with a *folded condition over an unfolded instance*, while the lemma
`ite_eq_right H` has both folded. They are defeq — but `t3Idx` is a plain `def`, and neither `rw`'s
keyed matching nor `simp`'s does `default`-transparency unfolding to see through it.

This reproduces in **11 lines of core Lean, no Mathlib**:

```lean
def idx (a b : Nat) : Nat := a + b
theorem idx_def (a b : Nat) : a + b = idx a b := rfl

example (p q : Nat) (h : ¬ (idx p q = idx q p)) :
    (if p + q = q + p then (1:Nat) else 0) = 0 := by
  simp only [idx_def]   -- folds the CONDITION into `idx`, not the Decidable instance
  rw [if_neg h]
```

**Lean 4.32.2: succeeds. Lean 4.34.0-rc2: fails**, with pattern and target printing *identically* —

```
Did not find an occurrence of the pattern
  if idx p q = idx q p then ?m.26 else ?m.27
in the target expression
  (if idx p q = idx q p then 1 else 0) = 0
```

which is character-for-character the shape of the real `SgdDescentCnn` failure. Worth reporting
upstream; it is a self-contained regression.

**Fixes, measured against that repro on 4.34:**

| candidate | result |
|---|---|
| `@[reducible] def idx` | ✅ works |
| `simp only [idx_def, h, if_false]` (use the hypothesis to kill the condition) | ✅ works |
| `simp only [if_neg h]` instead of `rw` | ✗ `simp` made no progress |
| `split_ifs` | (Mathlib-only; not testable in the bare probe) |

⇒ **Repair applied: `@[reducible] def t3Idx`.** It matches this codebase's own precedent —
`basisVec` (`Foundation/Tensor.lean:83`), which produces the very `ite` in question, is already
`@[reducible]`; `t3Idx` just never was. **Measured: 8 errors → 1.**

The survivor was a consequence of the fix, not a new problem: `SgdDescentCnn.lean:3911`,
`No goals to be solved`. With `t3Idx` reducible, `rw [Tensor3.flatten_unflatten]` inside `h1`
(the `conv2d_input_pdiv3` bridge, ~line 3905) now closes the goal with its own trailing `rfl`, so
the explicit `rfl` on the next line had nothing left to do. Deleted. **8 → 1 → 0.**

So the complete `SgdDescentCnn` repair is **one attribute and one deleted line**.

⚠ Recorded so it is not re-tried: replacing the failing `rw`s with `simp only [...]` was attempted
across all 6 sites and **fails**, with `simp made no progress` — same root cause, and the bare-Lean
probe reproduces that too (row 3 above).

### ✅✅ The whole proof corpus builds on 4.34.0-rc2

`lake build Certs` on the branch: **3959 jobs, 0 errors.** 218 of the 233 proof-suite modules have
`.olean`s; the 15 without are the `*Scorecard*`/`Ibp*` generated instances plus `Codegen/IRPrint`,
which belong to **`CertsHeavy`** (its own workflow, 350-min budget) — ⚠ `CertsHeavy` is therefore
the one slice still untested on 4.34.

Getting there took, in total, **three lines**:

| file | change |
|---|---|
| `Codegen/StableHLO.lean` | add `id` to one `simp only` list |
| `Training/SgdDescentCnn.lean` | `@[reducible]` on `t3Idx` |
| `Training/SgdDescentCnn.lean` | delete one now-dead `rfl` (line 3911) |

plus the mechanical deprecation migration (53 files, 639 lines) and the toolchain/manifest bump.

**122 warnings remain**, none blocking:

| warning | count | note |
|---|---|---|
| `linter.unusedVariables` | 40 | 33 are fixed on `lint-cleanup`; 7 more are `Certs`-only files |
| `if_true` → `ite_true` | 21 | fixed on `lint-cleanup` |
| `Set.mem_setOf_eq` → `Set.mem_ofPred_eq` | 19 | **new** — Mathlib rename, all in `Certificates/Smoothing{CP,NetSemantics,MC,Gaussian}.lean` |
| `linter.style.haveILetI` | 15 | |
| `linter.ambiguousOpen` | 12 | |
| `if_false` → `ite_false` | 10 | fixed on `lint-cleanup` |
| `linter.unusedSimpArgs` | 3 | one of these was the diagnostic that cracked `StableHLO` |
| `Set.setOf_forall` → `Set.ofPred_forall` | 1 | **new** |
| `String.dropRight` → `String.dropEnd` | 1 | **new** |

The three **new** Mathlib deprecations only surface in `Certs` (the `Proofs` slice never reaches
those APIs) and are pure renames like the `ite` family. ⚠ Same caveat as before: check whether the
replacement names exist on 4.32.2 before migrating them pre-bump.

### ⭐ Bisect: what belongs to which release, and therefore what can land now

Run with the 11-line repro above (no Mathlib needed, so this costs an `elan install` and seconds):

| | 4.32.2 | 4.33.0 | 4.33.1 | 4.34.0-rc2 |
|---|---|---|---|---|
| `t3Idx` rewrite works | ✅ | ✗ | ✗ | ✗ |
| `ite_eq_left` / `dite_eq_right` exist | — | — | — | ✅ |
| `if_pos` deprecated | — | — | — | ✅ |

Two independent things, on different timelines:

* **The matching regression is 4.33.0-era**, not 4.34 — it has been latent for two releases. The
  three-line proof fix therefore has nothing to do with 4.34 and is a candidate to land on `main`
  immediately, shrinking the eventual 4.34 diff to the toolchain pin plus the migration.
* **The deprecation migration is 4.34-only.** Its four target names exist in neither 4.32.2 nor
  4.33.x, and `if_pos` is not deprecated before 4.34. It must travel with the bump.

⭐ **A third option this opens up: 4.33.1 is a STABLE release.** Moving there needs the
`@[reducible]` fix but **no** deprecation migration at all, so it is a strictly smaller change than
waiting for 4.34 — at the cost of another worktree and ~9 GB of matching Mathlib to verify.

**When 4.34 actually ships, do NOT merge this branch.** It forked before `lint-cleanup` and the two
touch 10 files in common (`lakefile.lean`, `Tensor.lean`, `Attention.lean`, `ViTClose.lean`,
`ViTBackB0.lean`, `PerChannelBN.lean`, `LipschitzCertPairSDP.lean`, `SgdDescentCnn.lean`,
`SgdDescentMlp.lean`, `TrainedCnnSeal.lean`). The migration is a mechanical rename — re-run it on
the updated `main` instead, and keep only the toolchain pin from the branch.

### ⛔ The deprecation cleanup CANNOT be done before the bump

4.34 deprecates 338 uses in the `Proofs` slice alone:

| old | new | count | portable to 4.32.2? |
|---|---|---|---|
| `if_neg` | `ite_eq_right` | 163 | **NO** |
| `if_pos` | `ite_eq_left` | 104 | **NO** |
| `dif_neg` | `dite_eq_right` | 32 | **NO** |
| `dif_pos` | `dite_eq_left` | 30 | **NO** |
| `if_true` | `ite_true` | 8 | yes |
| `if_false` | `ite_false` | 1 | yes |

The statements are exact matches, so the rename itself is mechanical — **but
`ite_eq_left`/`ite_eq_right`/`dite_eq_left`/`dite_eq_right` exist in neither Lean 4.32.2 core nor
Mathlib v4.32.2** (verified by grep against both). Renaming them now breaks the current build.
They must land *with* the bump. Only `if_true`/`if_false` were portable; those 29 sites are
already done on `lint-cleanup` and verified green on 4.32.2.

⚠ When checking this yourself: `Init/Sym/Lemmas.lean` also defines an `ite_true`. That one is
`Lean.Sym.ite_true`, namespaced — not the root lemma. Don't let it convince you the root
statement changed.

### Gates

`lake exe docstring-checkrefs` → **1393 citations across 491 files, all resolve** (exit 0).
`lake exe blueprint-checkdecls` → **106 declarations, all resolve** (exit 0). Note
`blueprint/lean_decls` is a CI side effect of `leanblueprint web`; it can be regenerated locally
by extracting `\lean{...}` from `blueprint/src/*.tex`, splitting on commas — but **filter out
non-identifier lines** (a literal `...` in the tex will otherwise fail the gate spuriously).

---

## 2. What `lint-cleanup` actually changed

**66 files, +123 / −124 — net −1 line.** Every category is a strict 1-for-1 substitution; the
single net deletion is a duplicated `moreLinkArgs := lowererLink` in `lean_exe «mnist-mlp-pgd»`.
`verified_mlir/` is byte-identical afterwards (aggregate md5 unchanged), and `lake build Proofs`
is green on 4.32.2 (2251 jobs, 0 errors).

| change | files | note |
|---|---|---|
| `Bestiary/`: `import LeanMlir` → `import LeanMlir.Spec` | 41 | see below |
| `if_true`/`if_false` → `ite_true`/`ite_false` | 10 | portable both toolchains |
| `linter.unusedVariables` → `_`-prefix | 6 | all 33 warnings |
| `.trim` → `.trimAscii` | 8 | + `lakefile.lean` |

**Bestiary is the build-time win.** All 41 files imported the umbrella `LeanMlir.lean` (54
imports: `MlirCodegen`, `VerifiedTrain`, `VerifiedNets`, the whole `LeanMlir.Proofs.*` cone, and
Mathlib behind it) to print a spec table. They use only `NetSpec` + `Layer` + `.totalParams` /
`.archStr` / `.validate`. ⚠ `import LeanMlir.Types` is **not** enough — `totalParams`/`archStr`
live in `Spec.lean`. `LeanMlir.Spec` imports only `Types`, which imports nothing. All 41 verified
to elaborate.

**`.trim` → `.trimAscii` typing rules** (`String.trim` is deprecated in favour of
`String.trimAscii`, and the *return type changes* `String → String.Slice`):
`Slice` has `toNat?`, `toString`, `startsWith` and a `ToString` instance; it does **not** have
`isEmpty`, `toLower`, or `splitOn`. So `.trim.toNat?` → `.trimAscii.toNat?` is a drop-in, and
anywhere a real `String` was produced needs `.trimAscii.toString`. `mnist-lean4/` (4 more sites)
was left alone — it is not a root-lakefile target.

### The dead bench-bracket feature — deleted

`lakefile.lean`'s last warning (`yourHi` unused) was the tip of a whole vestigial feature, now
removed. `fmtRange (lo hi)` was defined, documented *"a single duration when it is degenerate,
`lo-hi` when it is not"*, and **never called**; `yourHiTotal` and `anyBracket` were assigned and
never read (which is why only `yourHi`, a plain `let`, tripped the linter). A comment at the print
site already said the single number was deliberate — *"the transport-leaning `hi` is kept only for
the off-vendor note below"* — and there is no such note. So the bracket was leftover from a
reverted approach, not an unfinished one.

`yourSecRange : Option (Nat × Nat)` → `yourSecLo : Option Nat` (the `Nat.min` of the two factors);
`fmtRange`, `yourHiTotal`, `anyBracket`, `yourHi` deleted. Printed output is unchanged.
⚠ Worth knowing: the low end is the *optimistic* one. On ares 2026-08-04 ch5's two factors were
40m and 91m and the real run landed at **89m** — near the high one. `lake run bench` has always
printed 40m there. That is now stated in `yourSecLo`'s docstring rather than implied by dead code.

---

## 3. Verified backlog — Mathlib instead of DIY (proof side)

All proof-side only; `tests/AuditAxioms.lean` audits by **name** (1616 `#print axioms`), so
keeping the declaration name and swapping only the proof body is zero-churn there, and Mathlib is
itself 3-axiom clean. Every Mathlib name below was grep-verified in `.lake/packages/mathlib`.

1. **`le_of_sq_le_sq`** — `Mathlib/Algebra/Order/Ring/Abs.lean:134`. The repo open-codes the
   `Real.sqrt_sq`/`sqrt_le_sqrt`/`sqrt_sq` sandwich in **12 sites / 8 files**, densest in
   `Certificates/LipschitzCertInstance.lean` (5×). Bonus: `Foundation/ResNet34.lean:493` and
   `Training/MobileNetV2SealRealistic.lean:46` are character-for-character Mathlib's own proof of
   `Real.abs_le_sqrt`.
2. **`euclid_norm_sq`** (`Certificates/LipschitzCert.lean:76`, 15 uses) is verbatim
   `EuclideanSpace.real_norm_sq_eq` (`Mathlib/Analysis/InnerProductSpace/PiL2.lean:155`). Keep the
   name, replace the body with the one-liner.
3. **max/abs family, ~40 lines → 3.** `Architectures/CNN.lean:2715` (`max_close`, 17-line
   `rcases`/`linarith`), `CNN.lean:2759` (`abs_max_le`), `Training/SgdDescentMlp.lean:54`
   (`relu_entry_lipschitz`, 20-line 4-way `by_cases`). The tell: the repo **already uses**
   `abs_max_sub_max_le_abs` at `LipschitzCertInstance.lean:90` and
   `Float/MobileNetV2WholeFloatBridge.lean:56` — these three sites just missed it.
4. **`abs_sub_le_add`** (`Codegen/BnInputBridge.lean:47`, `private`) duplicates Mathlib's
   `abs_sub` (the `to_additive` image of `mabs_div`, `Algebra/Order/Group/Abs.lean:81`) — which
   the repo already calls at `Codegen/AdjointChainBridge.lean:273`.
5. **Four hand-proved `√` comparisons** → `Real.sqrt_le_left` (`Analysis/Real/Sqrt.lean:224`),
   `Real.sqrt_lt` (`:232`), `Real.sqrt_lt'` (`:235`). Sites: `LipschitzCertScorecard.lean:54`,
   `ResNet34LivePC.lean:191,230`, `ResNet34LiveRealistic.lean:32`. One line each.
6. **`MatBridge.lean` has zero `@[simp]`.** `planning/archive/mat_matrix_phase2.md` proposed
   marking them as the cheap partial win against the deferred full `Mat → Matrix` migration; never
   executed. One attribute change.
7. The `rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]` chain appears
   **43×** across `Float/` and `Codegen/`. `Finset.card_fin` collapses two of those rewrites; the
   real win is one repo helper `abs_sum_le_mul` (`Float/PatchEmbedBackFloatBridge.lean:30` is that
   lemma hand-inlined three deep).

**Checked and rejected** (don't re-check): the full `Mat → Matrix` migration (priced at 2–4 days
for "new capability: none" in `planning/archive/mat_matrix_phase2.md` — agreed, defer);
`winRow`/`winRowMod` vs Batteries' `Fin.divNat`/`modNat` (shape is codegen-load-bearing, not
defeq); `maxPool2`'s argmax machinery vs `Finset.sup'` (its shape participates in `rfl`-level
codegen ties across 19+ files — the asymmetry with `MaxPool3s2.lean` is deliberate and
documented); `quad_form_nonneg_of_ldl` vs `Matrix.PosSemidef` (Mathlib's is over `Finsupp`;
bridging costs more than the existing 20-line proof). Genuinely **absent** from Mathlib: softmax,
logSumExp, relu, sigmoid, `Real.tanh` Lipschitz/`HasDerivAt`, `Real.sqrt` Lipschitz, the Higham
γₖ *upper* bound, any quantile function.

---

## 4. Verified backlog — core instead of DIY, and duplication (non-proof)

1. ⚠ **`mkParam` — 34 copies, 3 divergent formulas.** Canonical is
   `LeanMlir/VerifiedTrain.lean:349` (`private`, which is *why* there are copies): conv = He
   **fan-OUT** `2/(d0·d2·d3)`, dense = **Glorot** `2/(d0+d1)`. **27 files** use He **fan-IN**
   `2/(d1·d2·d3)` instead; 5 match the trainer; 1 more variant in `TestR34DpShard.lean`.
   `tests/TestSgdRenderTie.lean:47` *documents itself* as "identical to the driver's
   `VerifiedTrain.mkParam`" — and is not.
   **Severity is lower than it looks**: a tie test feeds the same θ to both sides, so the
   distribution is irrelevant to the tie itself. The bug is the false docstring, plus any test that
   means to reproduce trainer numerics. Fix = drop `private`, delete the copies.
2. **`GradcheckHelpers` is copy-pasted wholesale.** `tests/TestMHSA.lean` and `tests/TestSDPA.lean`
   each redefine all 9 helpers (`pow10`, `digitsToNat`, `splitAtChar`, `parseFloat`,
   `parseResults`, `runFn`, `randVec`, `dot`, `axpy`) without importing it — while
   `TestSgdRenderTie`, `TestViTBlock`, `TestViTTiny` import it correctly.
3. ⚠ **`floatToU8` — same name, two ranges.** Four CIFAR demos map `[-1,1] → u8` (via
   `(v+1)*0.5`); `demos/MainMnistDdpmSample.lean:55` maps `[0,1] → u8`. Probably correct per
   dataset, but the shared name is a copy-paste trap. Rename to
   `floatToU8Signed` / `floatToU8Unit`.
4. **`parseFloat` (3 copies, 34 lines each) is reimplementing core.**
   `Lean.Syntax.decodeScientificLitVal?` (`Init/Meta/Defs.lean:1006` — in `Init`, no `import Lean`
   needed) + `Float.ofScientific` (`Init/Data/OfScientific.lean:48`) do it.
   ⚠⚠ **The obvious replacement is WRONG**: `decodeScientificLitVal?` rejects bare integers, so
   `parseFloat "42"` and `parseFloat "0"` return **0.0**. Measured. It needs a `toNat?` fallback:
   ```lean
   def parseFloat (tok : String) : Float :=
     let (neg, s) := if tok.startsWith "-" then (true, tok.drop 1 |>.toString) else (false, tok)
     let mag := match Lean.Syntax.decodeScientificLitVal? s with
       | some (m, sgn, e) => Float.ofScientific m sgn e
       | none => match s.toNat? with | some n => n.toFloat | none => 0.0
     if neg then -mag else mag
   ```
   Verified against `-0.00623606 / 1.3e-05 / 42 / 0 / -1.5E3 / -7 / garbage`.
   This also retires the "Lean core has no `String.toFloat?`" comment in `VerifiedTrain.lean` and
   4+ imagenette mains — true for that *name*, but the machinery exists, and the
   `LEAN_MLIR_BASE_LR_U` integer-encoding dodge is working around a gap that isn't there.
5. **Hand-rolled substring search ×2** → `String.contains` (`Init/Data/String/Search.lean:300`),
   which takes a `String` pattern. Verified by `#eval`. Sites: `tests/TestYolov1Mutex.lean:29`
   (docstring claims "core lacks String.containsSubstr") and `tests/DocstringCheckRefs.lean:143`
   (12-line hand-written scan). Both docstrings are wrong and should go with the code.
6. **`.foldl (· * ·) 1` / `.foldl (· + ·) 0`** → `.prod` / `.sum`, which exist in core `Init` for
   both `List` and `Array` (`Init/Data/Array/Basic.lean:1086,1097`,
   `Init/Data/List/Basic.lean:2067,2081`). **87 + 18 = 105 sites** across 47 + 12 files.
7. **Dead code, all confirmed unreferenced**: `dwConvAttrBlock` (`MlirCodegen.lean:92`),
   `emitChannelSplitGrad` (`:821`), `fwdRenderedBatch` (`VerifiedTrain.lean:1076`), `zout`
   (`tests/TestProj1x1Generic.lean:43`), `unassoc` (`tests/TestConvNeXtTTrainPC.lean:77`), `nPOf`
   (`tests/TestResnet34BatchCheck.lean:41`). All `private`, so file-local — 1 occurrence each.
8. **`LeanMlir/MnistData.lean` is unreachable from every build target.** Imported only by
   `historical/{MainMlpTrain,MainCnnTrain,MainCifarTrain}.lean`; `historical/` appears in
   `lakefile.lean` only inside a prose comment, and the umbrella `LeanMlir.lean` does not import
   it. Delete, or add `historical` as a real target.
9. **`imagenetteClasses` diverges**: `demos/MainGradCAM.lean:80` uses hyphens
   (`"English-springer"`, `"chainsaw"`), `demos/MainInspectConvNeXt.lean:21` uses spaces
   (`"English springer"`, `"chain saw"`). Two demos print different labels for the same class id.
10. **`summarize` — 41 copies in `Bestiary/`**, 10 distinct bodies, varying only cosmetically.
    One parameterized version belongs next to `structure NetSpec` in `Types.lean:331`.
11. `runIree` is inlined in 4 DDPM demos because `findIreeCompile`/`runIreeCached` are `private` in
    `Train.lean`. **Not a bug** — `Train.lean:98`'s `runIreeCached` *does* carry the XLA
    short-circuit (`:102`) and is the only caller of the plain `runIree` (`:112`). Duplication
    only.
12. `Array.mkEmpty` → `Array.emptyWithCapacity`: **not a deprecation** (no warning on 4.32.2 or
    4.34; verified by `#eval` on both). Consistency nit only — the same three files already call
    `ByteArray.emptyWithCapacity`.

---

## 5. Build hygiene

**267 orphaned `.olean`s** sit in `.lake/build/lib/lean` with no corresponding source — 201 under
`LeanMlir/` from the flat→nested `Proofs/` reorg (`LeanMlir/Proofs/Attention.olean` etc.).
**Zero of them is imported by any current source** (checked), so the prune is safe; this is exactly
the "stale dev .oleans" hazard `proofs.yml` warns about. Prune with a targeted `find … -delete` on
the orphan list — ⛔ **never** `lake clean`, which would take the 35 GB of checkpoints with it.

**Nothing gates linter warnings.** There is no `leanOptions` block in `lakefile.lean` and no
`warningAsError`; CI only greps the build log for `sorry`. The repo already has the right idiom
for this — a baseline ratchet (`scripts/render_guard_baseline.txt`, `docstring_ref_baseline.txt`,
`convention_baseline.txt`, all "debt can shrink but never grow", with `--update-baseline`). A
`scripts/linter_baseline.txt` peer would now start at **0**.

**Mathlib's opt-in linters are unreachable.** `LeanMlir/**` uses narrow Mathlib imports, so
`Mathlib.Tactic.Linter.*` is never in the cone and `linter.style.*`, `linter.flexible`,
`linter.minImports`, `linter.haveLet`, `linter.upstreamableDecl` never register. Only
`tests/comparator/{Challenge,Solution}.lean` do a bare `import Mathlib`. Enabling them means
adding `import Mathlib.Tactic.Linter` to a base module (e.g. `Foundation/Tensor.lean`) — a real
build-cost tradeoff, not a free switch. `linter.unusedTactic` is already on by default (it rides
in on `Mathlib.Tactic.Ring`).
⚠ Library-defined linter options need `-Dweak.linter.foo=true`; a bare `-D` on an unregistered
option is a hard error. And probing option validity against `/dev/null` gives a false answer —
validity depends on the loaded import cone.

**`sorry` count is clean**: 52, all in `tests/comparator/Challenge*.lean`, which is the Diderot
challenge set by design. Zero `native_decide` anywhere (every mention is a comment asserting the
repo avoids it).

**Three toolchain pins** must move together: `lean-toolchain`, `jax/lean-toolchain`,
`tests/comparator/lean-toolchain`. `jax.yml` compares only the first two and only *warns*;
`tests/comparator` is ungated entirely. Worth extending that gate.
