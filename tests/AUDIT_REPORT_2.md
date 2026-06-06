# Second-pass formalization audit

Independent re-audit performed 2026-05-18 against the same 12-phase
prompt as `AUDIT_REPORT.md`. The prior audit ran in mid-May and the
author landed five fixes (E.1–E.5) between then and now — so this
pass is essentially "what did the first audit miss, and is the
follow-up work as solid as it claims."

Working environment: Lean 4.30.0-rc2, mathlib pinned per
`lake-manifest.json`, `lake build LeanMlir.Proofs` clean (2102 jobs,
two unused-simp-argument warnings, no errors).

## Headline

The follow-up work landed cleanly. 54/54 headline theorems pass the
three-axiom check independently (`lake env lean tests/AuditAxioms.lean`
reproduced); the CI guardrail in `.github/workflows/proofs.yml` is
correctly structured and will fail on regression; `HasVJPAt`,
`relu_has_vjp_at`, and `maxPool2_has_vjp_at3` are real proofs, not
stubs that exist only to satisfy `#print axioms`. My 12 sanity probes
in `tests/AuditProbes.lean` all elaborate, and a mutation test in
`tests/AuditMutation.lean` confirms `correct := rfl` does pin
`backward` definitionally (substituting `backward := 42` correctly
fails to elaborate, as expected).

What's new below is what the first audit missed.

---

## A. Critical findings

**None.** No axiom, no sorry, no vacuous proof discovered. The
canonical-witness `correct := rfl` instances in `relu_has_vjp`,
`mlp_has_vjp`, and `maxPool2_has_vjp3` continue to be the only
`rfl`-based correctness fields, and they remain non-vacuous in the
mutation-test sense (the mutation file confirms `backward := 42`
fails to typecheck against `correct := rfl`). The three-axiom
closure is genuine.

---

## B. Substantive findings (new — not in the first audit)

### B.1 The codegen does NOT emit `select_and_scatter`, but the docs say it does

`MlirCodegen.lean:5188–5224` emits MaxPool backward as a
**tile-compare-select** pattern, not `stablehlo.select_and_scatter`.
The codegen itself documents this at line 5189:

> MaxPool backward via tile-compare-select (avoids `select_and_scatter`
> which IREE doesn't support). Works correctly for `stride==size`
> (non-overlapping). For overlapping pools (`size>stride`), use
> `stride==size` pooling instead.

The actual MLIR is `compare EQ → broadcast → select`, with the
gradient routed by the equality mask between pooled output and input.
Seven sites in the proofs and README still claim `select_and_scatter`:

- `LeanMlir/Proofs/CNN.lean:1493` — MLIR snippet inside `maxPool2_has_vjp3` docstring
- `LeanMlir/Proofs/CNN.lean:1494`, `1507`, `1520`, `1535`, `1972`, `2017`
- `LeanMlir/Proofs/README.md:191`
- `tests/AUDIT_REPORT.md:51`

**Bridge-theorem impact: zero at smooth points.** At a window with a
unique strict argmax, tile-compare-select, `select_and_scatter` with
GE-selector, and the canonical pdiv sum all return the same value, so
`maxPool2_codegen_matches_canonical` remains valid as stated. The
theorem proves something true about the canonical-witness backward;
it does not — and never did — prove anything about the MLIR string
the codegen produces. So the doc drift is in the prose connecting the
two, not in the math.

**Bridge-theorem impact at non-smooth points (argmax ties): the
codegen-and-canonical-witness gap is wider than the README admits.**
The README says "ties broken by the GE comparator" — but the actual
codegen uses `compare EQ`, which routes the gradient to *every* cell
tied for the max, not a single deterministic argmax. This is more
permissive than select_and_scatter would be. For ML training purposes
it still works (PyTorch and JAX both do something similar), but the
"codegen substitutes the standard subgradient convention" framing is
imprecise.

**Recommended fix.** Replace all `select_and_scatter` mentions in
`CNN.lean` and `Proofs/README.md` with "tile-compare-select (the
codegen avoids `select_and_scatter` for IREE compatibility — see
`MlirCodegen.lean:5189`)". Adjust the tie-breaking sentence in
`README.md:191` to "ties: every tied cell receives the gradient
(EQ-mask broadcast), matching PyTorch/JAX semantics."

### B.2 `HasVJPAt3` has no chain rule — `maxPool2_has_vjp_at3` is an orphan

`Tensor.lean:1666–1681` defines `HasVJPAt3` and `HasVJP3.toHasVJPAt3`,
but there is no `vjp3_comp_at` (or `vjp_comp_at3`) anywhere in the
project. The global `vjp3_comp` exists at `Tensor.lean:1621` (for
`HasVJP3`), and the pointwise `vjp_comp_at` exists at
`Tensor.lean:342` (for `HasVJPAt` on `Vec`), but the Tensor3-pointwise
chain rule is missing.

The CNN.lean docstring at line 2020 acknowledges this gap, calling
the consumer "(future) `cnn_has_vjp_at3`". As a result,
`maxPool2_has_vjp_at3` cannot compose with any other Tensor3 operator
pointwise — it stands alone with no downstream consumer.

The whole point of E.5 was to provide a pointwise framework that kills
the `correct := rfl` escape *along entire chains*, not just at single
ops. `mlp_has_vjp_at` (Vec-level, MLP.lean:536) does close the chain;
`maxPool2_has_vjp_at3` (Tensor3-level, CNN.lean:2023) does not, because
there's nothing to chain through.

**Recommended fix.** Write `vjp3_comp_at` by the same structure as
`vjp_comp_at`, using `pdiv3_comp` (which already exists, Tensor.lean:1548)
under `DifferentiableAt` on the flattened forms. ~40 lines, same shape
as `vjp_comp_at`. Then state `cnn_has_vjp_at3` as the CNN apex theorem
under input-smoothness hypotheses, mirroring `mlp_has_vjp_at`. The
infrastructure is in place; this is a missing connector, not a hard
proof.

### B.3 `HasVJPAt` instances have no in-project consumers

`relu_has_vjp_at`, `mlp_has_vjp_at`, and `maxPool2_has_vjp_at3`
appear in `tests/AuditAxioms.lean` (for the `#print axioms` check)
and nowhere else in the codebase — not in `MlirCodegen.lean`, not
in `tests/comparator/`, not in any other proof file or trainer.

This is consistent with the framing in `planning/audit.md` ("kills
rfl escape at kinks") — the value of the pointwise framework is that
it *exists* and discharges the contract without `rfl` at smooth
inputs. But it also means the pointwise instances are pure proof
hygiene; no production code path actually invokes them.

**Recommendation.** Either (a) add `tests/comparator/`-style
re-verification of the `_at` variants so they live alongside the
global `_has_vjp` checks; or (b) explicitly document in the README
that the `_at` framework's purpose is the smooth-point correctness
guarantee, with no runtime consumers expected. Currently the README
implies the `_at` framework is load-bearing for the codegen story
when in practice it stands on the side.

### B.4 Mathlib bridge still missing (prior audit's B.2, unaddressed)

The first audit flagged that `Proofs.Mat.mulVec`, `Proofs.Mat.outer`,
`Proofs.Mat.transpose`, `Proofs.Mat.mul` are defined in parallel to
mathlib's `Matrix` namespace with no bridge theorems. This was
tagged "fine for internal consistency" and not landed in the
follow-up. Grepping confirms: no `Mat.mulVec_eq_matrix_mulVec` or
similar exists.

`Vec` and `Mat` are `abbrev`s over `Fin n → ℝ` and
`Fin m → Fin n → ℝ`, so they inherit all the Pi instances from
mathlib (`NormedAddCommGroup`, `NormedSpace ℝ`, etc.). That's good
for the foundation work. But a downstream consumer who wants to
apply a mathlib theorem about `Matrix.mulVec` to a
`Proofs.Mat.mulVec` instance has to bridge by hand.

For "a machine-checked AD formalization intended to be read by the
broader Lean community," a one-time bridge theorem per operation
would make the project's outputs more usable. Not blocking;
recurring point worth keeping on the list.

---

## C. Hygiene findings

### C.1 Two unused-simp-argument warnings in `Tensor.lean`

`identity3_has_vjp.correct` (line 1727 and 1729):

    simp only [eq_self_iff_true, ite_true]

`eq_self_iff_true` is unused — `ite_true` alone closes the goal.
The mathlib `unusedSimpArgs` linter flags both. Mechanical fix:
remove `eq_self_iff_true,` from both lines.

### C.2 Stale "axioms" in section header

`CNN.lean:1595` still has a "Summary of axioms in this file" section
title (text follows "Derived (not axioms):"), even though the prior
audit fix at commit `281598f` cleaned up the body text. The section
heading itself reads as a claim that axioms exist in the file. Rename
to "Summary of derivations in this file" — which is what the prior
audit recommended in C.1 and which the follow-up partially did at
line 2106 but missed at line 1595.

(Worth checking: at line 2106 it does say "Summary of derivations in
this file." So this file has *two* summary blocks and only one was
renamed. The one at 1595 is still "axioms".)

### C.3 No `#lint` run scripted into CI

Mathlib's linters are not invoked in `.github/workflows/proofs.yml`.
Adding `import Mathlib.Tactic.Linter` + `#lint` blocks at the end of
each proof file would catch the unused-simp arguments (C.1) and any
future regressions automatically. Two-line CI step.

---

## D. Positive findings (independently verified)

### D.1 Three-axiom closure: 54/54

Independently reproduced `lake env lean tests/AuditAxioms.lean`;
every line ends in `[propext, Classical.choice, Quot.sound]`. No
`sorryAx`, no project axiom, no `Lean.ofReduceBool`.

### D.2 The CI guardrail is well-structured

`.github/workflows/proofs.yml`'s "Three-axiom closure check" step
correctly:

- counts total `depends on axioms:` lines
- counts lines matching the exact regex `[propext, Classical\.choice, Quot\.sound]$`
- fails if `total ≠ ok` and prints the offending lines
- writes a markdown summary on success

A negative-path test (simulating a sorryAx-tainted line) would fail
the workflow and name the offending theorem. Good design.

### D.3 `pdiv` and `pdiv3` are genuinely thin `fderiv` wrappers

`tests/AuditProbes.lean` confirms by `rfl`:

    example (f : Vec 2 → Vec 3) (x : Vec 2) (i : Fin 2) (j : Fin 3) :
        pdiv f x i j = fderiv ℝ f x (basisVec i) j := rfl

and analogously for `pdiv3` via the `Tensor3.flatten`/`unflatten`
bijection. Zero hidden complexity.

### D.4 `dense_has_vjp.backward = Mat.mulVec W dy` is `rfl`

Confirmed in `tests/AuditProbes.lean`. The dense backward IS the
named function, not a derived expression — every consumer can
substitute one for the other definitionally.

### D.5 `correct := rfl` is non-vacuous (mutation confirmed)

`tests/AuditMutation.lean` uses `#guard_msgs` to verify that

    noncomputable example (n : Nat) : HasVJP (relu n) where
      backward _ _ _ := 42
      correct _ _ _ := rfl

fails to elaborate with the precise expected `type mismatch` error.
So `correct := rfl` does pin `backward` to be definitionally equal
to the canonical sum — the canonical-witness pattern is not a free
pass for arbitrary backward functions.

### D.6 `relu_has_vjp_at.backward = if x i > 0 then dy i else 0` is `rfl`

The pointwise framework genuinely encodes the codegen-shape formula
directly. The `correct` field then goes through a real proof
(`pdiv_relu` + sum-collapse), not `rfl`. So the rfl-escape is
genuinely closed at the smooth-point level for ReLU.

### D.7 `Vec 0` edge case

`pdiv (id : Vec 0 → Vec 0) x i j` is vacuous because `i : Fin 0`,
and `relu_has_vjp_at 0 x h_smooth` exists (the proof handles the
empty case explicitly at `MLP.lean:338–343`). Confirmed by probe.

### D.8 The apex `vit_full_has_vjp` is genuinely composed

`Attention.lean:3665–3714` builds `vit_full_has_vjp` by chaining
`vjp_comp` over `classifier_flat ∘ body_bridge ∘ patchEmbed`, where
`body_bridge` is `hasVJPMat_to_hasVJP (vit_body_has_vjp_mat …)` and
each `_diff` discharge is a real lemma. Not a stub.

---

## E. Proposed patches (in priority order)

### E.6 (substantive) Fix the `select_and_scatter` documentation drift

Mechanical text edit across 7 sites in `CNN.lean` and `Proofs/README.md`.
Replace `select_and_scatter` references with `compare EQ + select`
(tile-compare-select) and adjust the tie-breaking sentence to reflect
EQ-mask semantics. Shortest-effort, highest-credibility fix.

### E.7 (substantive) Land `vjp3_comp_at` to close the HasVJPAt3 framework

40-ish lines, same shape as `vjp_comp_at`. Without it,
`maxPool2_has_vjp_at3` is an orphan and the pointwise framework is
asymmetric (Vec has chain rule, Tensor3 doesn't). Promotes the
`HasVJPAt3` story from "single instance" to "framework."

### E.8 (substantive, larger) Write `cnn_has_vjp_at3` once E.7 lands

The CNN apex pointwise theorem, mirroring `mlp_has_vjp_at`. Would
close the chain through conv → relu → conv → relu → maxPool → flat →
dense → relu → dense → relu → dense, requiring pointwise smoothness
on every relu and `MaxPool2Smooth` on the pre-pool tensor. This is
the load-bearing demonstration that the `_at` framework can carry
real composition, not just live alongside the global version.

### E.9 (hygiene) Drop the two unused `eq_self_iff_true` simp args

Two-character edit each at `Tensor.lean:1727` and `1729`. Resolves
the only two warnings in the proof build.

### E.10 (hygiene) Rename `CNN.lean:1595` section header

Change "Summary of axioms in this file" → "Summary of derivations in
this file" — consistent with the rename at line 2106. The prior
audit's commit `281598f` missed this site.

### E.11 (optional) Add mathlib bridge theorems for Mat operations

One-line `theorem mulVec_eq` per operation, connecting
`Proofs.Mat.mulVec` to `Matrix.mulVec`. The prior audit suggested
this and the author chose not to land it; the recommendation stands
but is genuinely optional.

---

## Files produced during this audit

- `tests/AuditProbes.lean` — 12 independent sanity probes (all pass)
- `tests/AuditMutation.lean` — `#guard_msgs` negative probe pinning `correct := rfl`
- `tests/AUDIT_REPORT_2.md` — this report

All three elaborate under the current build. No project files were
modified during the audit.

---

## What the first audit had right and what it missed

Right:
- Identified the `correct := rfl` vacuity pattern correctly.
- Identified the codegen trust boundary correctly (and the gap is
  genuine).
- Proposed the right shape of bridge theorems.
- Designed a comparator infrastructure that survives.
- Documented patches and let the author land them.

Missed (this audit's contribution):
- The codegen does not actually emit `select_and_scatter` — only the
  prose says it does. The actual MLIR is tile-compare-select. The
  README, six CNN.lean docstrings, and the first audit's own report
  all repeat the wrong claim.
- `HasVJPAt3` has no chain rule. The E.5 patch as landed kills the
  rfl escape at single Tensor3 operators but cannot compose them
  pointwise. The MLP side (`HasVJPAt` + `vjp_comp_at`) is complete;
  the CNN side is not.
- The `_at` framework is not consumed anywhere downstream — its
  audience is `#print axioms` and a future `cnn_has_vjp_at3` that
  doesn't exist yet.

---

## Status update — 2026-06-06

The audit above is preserved as a dated (2026-05-18) snapshot; this section
supersedes it where they differ.

**Counts.** 54 → **114** three-axiom `#print axioms` checks
(`tests/AuditAxioms.lean`); comparator re-check now covers **51** theorems
(`tests/comparator/config.json`).

**B.2 — RETRACTED (resolved).** The report claims "`HasVJPAt3` has no chain
rule — `maxPool2_has_vjp_at3` is an orphan." No longer true: `vjp3_comp_at`
exists at `Tensor.lean:1718`, and `maxPool2_has_vjp_at3` is consumed via
`hasVJPAt3_to_hasVJPAt` into `cnn_has_vjp_at` (`CNN.lean:2703`) and into
`IR.lean`. (The probe comment at `tests/AuditProbes.lean:24` still asserts the
old claim and is now stale — worth deleting.) Residual nit: a Tensor3-native
`cnn_has_vjp_at3` consuming `vjp3_comp_at` *directly* still doesn't exist (the
CNN apex composes at the flattened `Vec` level), so `vjp3_comp_at` itself has
no in-tree consumer yet — a cleanliness item, not the capability gap the
report described.

**B.3 — resolved.** The `HasVJPAt` instances are now load-bearing: the
conditional whole-network VJPs (`cnn` / `mobilenetv2` / `convnext_has_vjp_at`)
chain through `vjp_comp_at`, and the concrete instances (`MlpConcrete`,
`CnnConcrete`, `MobileNetV2Concrete`) consume them to discharge smoothness.

**B.4 — resolved.** The Mathlib `Matrix` bridge landed
(`LeanMlir/Proofs/MatBridge.lean`).

**C.2 — resolved.** `CNN.lean` now reads "Summary of derivations".

**B.1 — resolved.** The `select_and_scatter` doc drift is fixed. The two
remaining mentions (`CNN.lean:1803`, `Proofs/README.md`) now correctly state
that the codegen emits **tile-compare-select** and *avoids*
`select_and_scatter` (IREE doesn't support it), with the EQ-mask tie
semantics (gradient to every tied cell) spelled out.

**New since this audit — whole-network VJPs.** ViT, ConvNeXt and EfficientNet
are unconditional global `HasVJP` (correct at every input, only `0 < ε`);
MLP, MNIST-CNN, ResNet and MobileNetV2 carry concrete instances discharging
every smoothness hypothesis. See the "Whole-network VJPs" section of
`LeanMlir/Proofs/README.md`. Honest caveats: the codegen↔proof link is still
unproven, `MobileNetV2Concrete` is a degenerate (constant-output) witness, and
the concrete nets are tiny.
