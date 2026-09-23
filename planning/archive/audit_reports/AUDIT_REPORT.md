# Formalization Audit Report

Audit of `LeanMlir/Proofs/` (9 files, ~10,400 lines of proof content, 18,860
lines total including codegen). Performed against the three claims:
no project axioms, clean `lake build`, every former axiom now a theorem.

The headline: **the central claims hold.** Every one of 47 headline
theorems passes the three-axiom check (`[propext, Classical.choice,
Quot.sound]`). The foundation is genuinely Mathlib-derived, mutation
testing breaks the proofs at every mutation tried, and the comparator
infrastructure at `tests/comparator/` independently re-verifies the
axiom closure. This is a solidly constructed proof suite.

The findings below are residual: documentation drift, a vacuous-witness
pattern that the author has documented but not fully closed in code,
and a missing bridge theorem the project's prose references but the
Lean source never states.

---

## A. Critical findings

**None.** No axiom or sorry was found, every mutation tried broke the
build, and no headline theorem depends on any project axiom outside
the Lean core triple. The comparator setup at `tests/comparator/`
provides independent kernel re-verification of 38 theorems with a
hard-coded allowlist of exactly `{propext, Quot.sound, Classical.choice}`.

---

## B. Substantive findings

### B.1 Vacuous-witness pattern at the codegen trust boundary

Three `HasVJP*` instances use the `correct := rfl` pattern:

- `relu_has_vjp` (MLP.lean:379–381)
- `mlp_has_vjp` (MLP.lean:422–429)
- `maxPool2_has_vjp3` (CNN.lean:1507–1513)

In each, `backward` is defined to be exactly the right-hand side of
`correct` (`∑ j, pdiv f x i j * dy j`), so `correct := rfl` reduces to
`x = x`. The mutation test (replacing `backward` with `(42 : ℝ)`) does
break the build — `correct := rfl` is not totally toothless, because
it pins `backward` to be *definitionally equal* to the RHS. But the
RHS itself (`∑ j, pdiv (relu n) x i j * dy j`) doesn't independently
characterize what backpropagation should produce: at non-smooth points
(any coordinate exactly zero, or any pooling argmax tie), `pdiv`
returns Mathlib's junk default of zero, while the codegen emits the
standard subgradient (`if x > 0 then dy else 0`) or the
`stablehlo.select_and_scatter` formula.

The author has documented this transparently in
`LeanMlir/Proofs/README.md`'s "Codegen trust boundary" section. The
gap is real, but disclosed. What's missing is a **formal bridge
theorem** that closes the gap *at smooth points*.

The audit prompt requested this theorem, and it goes through:

```lean
theorem relu_codegen_matches_canonical (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) (dy : Vec n) (i : Fin n) :
    (relu_has_vjp n).backward x dy i = if x i > 0 then dy i else 0 := by
  show ∑ j : Fin n, pdiv (relu n) x i j * dy j = _
  simp_rw [pdiv_relu n x h_smooth i]
  rw [Finset.sum_eq_single i
      (fun j _ hne => by rw [if_neg (Ne.symm hne)]; ring)
      (fun h => absurd (Finset.mem_univ i) h)]
  rw [if_pos rfl]
  by_cases hx : x i > 0
  · rw [if_pos hx, if_pos hx]; ring
  · rw [if_neg hx, if_neg hx]; ring
```

Verified in `tests/AuditBridge.lean`. Three-axiom closure preserved.
With this theorem in `MLP.lean` and the analogous one for `maxPool2`
in `CNN.lean`, the project can promote "trust boundary disclosure"
into "trust boundary with a formal smooth-point bridge."

The current state — backward defined as the canonical pdiv sum, with
no bridge to the codegen formula — leaves the only formal content of
`relu_has_vjp` to be the pdiv-derived expression, which agrees with
the codegen only at smooth points and only as a corollary the project
does not state.

### B.2 Missing bridge to mathlib's `Matrix` API

The project defines its own `Proofs.Mat.mulVec`, `Proofs.Mat.outer`,
`Proofs.Mat.transpose`, `Proofs.Mat.mul` over `Fin m → Fin n → ℝ` (a
plain function type, not Mathlib's `Matrix`). Mathlib has
`Matrix.mulVec`, `Matrix.transpose`, etc. **There is no bridge
theorem connecting the two namespaces** — no
`Proofs.Mat.mulVec_eq_matrix_mulVec` or similar.

This is fine for internal consistency but means a downstream consumer
who wants to apply a Mathlib theorem about `Matrix.mulVec` to a
`Proofs.Mat.mulVec` instance has to re-prove the equivalence by hand.
For a "machine-checked AD formalization" intended to be read by the
broader Lean community, a one-time bridge theorem per operation would
make the project's outputs more usable.

### B.3 Composition rules require `Differentiable ℝ f` (global)

`vjp_comp`, `vjpMat_comp`, `biPath_has_vjp`, `elemwiseProduct_has_vjp`,
and `vjp3_comp` all take `Differentiable ℝ f` (global) rather than
`DifferentiableAt ℝ f x` (pointwise) as their smoothness hypothesis.
This is what blocks the composition route for `relu`-containing chains
— and is the upstream cause of the `correct := rfl` escape in
`mlp_has_vjp`.

For dense / softmax / layerNorm / gelu / patch-embed / mhsa / the full
transformer pipeline, this is fine: they *are* globally differentiable,
and the project's `_diff` lemmas discharge the hypothesis. But the gap
remains for any composition through ReLU.

A pointwise variant `HasVJPAt` with a `vjp_comp_at` rule taking
`DifferentiableAt ℝ f x` and `DifferentiableAt ℝ g (f x)` would let
the project state `mlp_at_smooth_input : HasVJPAt (mlpForward …) x`
under the input-smoothness hypothesis `∀ k, x k ≠ 0` and avoid the
canonical-witness escape entirely. This is more substantial than the
ReLU bridge theorem — a real API addition — so I haven't written it,
but it's the patch that fully resolves the criticism.

---

## C. Hygiene findings

### C.1 Documentation drift

Six locations claim things are axioms that are now theorems:

- **Tensor.lean:311** — `/-! ... The 5 local Jacobian axioms (matmul,
  scalarScale, transpose, rowIndep) remain — they're genuine calculus
  facts about specific operations, not structural framework. -/`
  All five (`pdivMat_matmul_left_const`, `pdivMat_matmul_right_const`,
  `pdivMat_rowIndep`, `pdivMat_scalarScale`, `pdivMat_transpose`)
  are now `theorem`s.

- **Tensor.lean:600** — `/-! The three axioms here are local Jacobians
  for the operations that appear in scaled dot-product attention's
  backward pass: ... -/` Same issue. All three are theorems.

- **Tensor.lean:1472–1473** — references `pdiv3_conv2d_vjp`,
  `pdiv3_maxPool2_vjp`, `pdiv3_depthwise_vjp` as "local Jacobian
  axioms remain". **Those names don't exist anywhere in the codebase
  except this comment** — they've either been renamed (to
  `conv2d_has_vjp3`, `maxPool2_has_vjp3`, `depthwise_has_vjp3`) or
  never existed. Broken cross-reference.

- **SE.lean:54** — `- 'pdiv_mul' — product rule for partial derivatives
  (axiom)`. `pdiv_mul` is a theorem, not an axiom (Tensor.lean:148).

- **Attention.lean:1927** — `- 'MHSA' — 'mhsa_has_vjp_mat' (bundled
  axiom — Phase 8)`. `mhsa_has_vjp_mat` is a `noncomputable def`
  whose `.correct` is a proven theorem (the apex composition built
  via `vjpMat_comp` on the per-head pipeline). Not an axiom.

- **CNN.lean:31** — `We state the VJP formulas as axioms (the proofs
  are standard matrix calculus on cross-correlations) ...` Stale.
  All conv VJPs are now theorems (per Phase 1, Apr 2026).

The CNN.lean file also has a `## Summary of axioms in this file`
section (line 1595) that lists "Derived (not axioms):" — but the
section header still says "axioms in this file," which an external
reader will pattern-match as a claim there *are* axioms. Cleanup:
rename to "Summary of derivations" or "Summary of contents."

### C.2 Vec/Mat duplicate mathlib (B.2 above)

### C.3 `pdiv ↔ fderiv` is definitionally `rfl`

Verified: `example (f : Vec 2 → Vec 3) (x : Vec 2) (i : Fin 2) (j : Fin 3) :
pdiv f x i j = fderiv ℝ f x (basisVec i) j := rfl` elaborates. So the
`pdiv` abstraction has zero hidden complexity — it really is a thin
named wrapper, and any theorem about `pdiv` could be restated against
`fderiv` directly. This is a positive finding, included here because
the audit prompt asked for confirmation.

---

## D. Positive findings (things I checked and confirmed)

### D.1 No project axioms

`rg '^axiom |^noncomputable axiom ' LeanMlir/` finds nothing. Every
`axiom` mention in the proof files is in a comment.

### D.2 No sorry / admit / proof_wanted in proof files

Same — only doc-comment mentions saying "no sorry's" or "proved, no sorry."

### D.3 47/47 headline theorems pass the three-axiom check

`tests/AuditAxioms.lean` runs `#print axioms` on every headline
theorem (foundation, MLP, CNN, BN, Residual, Depthwise, SE,
LayerNorm/GELU, Attention through `vit_full_has_vjp`). All 47 show
exactly `[propext, Classical.choice, Quot.sound]`. No `sorryAx`, no
project axiom, no `Lean.ofReduceBool`, no `Quot.lift` outside the
expected three.

### D.4 Foundation is genuinely Mathlib-grounded

`pdiv f x i j` is `noncomputable def pdiv := fderiv ℝ f x (basisVec i) j`
(Tensor.lean:104–106), and the structural rules really are derived:

- `pdiv_id` uses `fderiv_id`
- `pdiv_const` uses `(hasFDerivAt_const c x).fderiv`
- `pdiv_reindex` uses `ContinuousLinearMap.fderiv` on the reindex CLM
- `pdiv_mul` uses `fderiv_mul`
- `pdiv_add` uses `fderiv_add`
- `pdiv_comp` uses `fderiv_comp`
- `pdiv_finset_sum` is a `Finset.induction_on` over `pdiv_add` + `pdiv_const`

This is what was claimed and what was delivered.

### D.5 `dense_has_vjp.correct` is non-vacuous

Mutation test (replace backward with `fun _ _ => fun _ => 0`) makes
the `simp` proof fail. Confirmed not a self-equality.

### D.6 Mutation tests caught every mutation tried

1. `pdiv_id` flipped 1/0 → 3 errors propagated.
2. `dense_has_vjp.backward` set to `Mat.mulVec W.transpose dy` → type error.
3. `dense_has_vjp.backward` set to `fun _ => 0` → `simp made no progress`.
4. `relu_has_vjp.backward` set to `(42 : ℝ)` → `rfl` type mismatch
   (so even the canonical-witness instances pin `backward` to be
   definitionally the pdiv sum).
5. `pdivMat_matmul_left_const` condition flipped `l = j` → `l ≠ j` →
   rewrite failed in the case-analysis branch.

### D.7 Concrete instance examples elaborate

`tests/AuditSanity.lean` contains 12+ concrete examples covering
`pdiv_id` / `pdiv_const` / `pdiv_dense` / `pdivMat_transpose` /
`dense_has_vjp.backward = Mat.mulVec` / ReLU at smooth points (both
active and inactive coordinate) / the `pdiv = fderiv` rfl bridge /
`Vec 0` edge case. All elaborate.

### D.8 Comparator infrastructure is well-designed

`tests/comparator/{Challenge,Solution}.lean` does exactly what it
claims — 38 theorems, statements verified bit-identical, axiom
closure independently kernel-checked with a hard-coded allowlist.
This is the right design for a "zero project axioms" claim: it makes
the property a CI invariant, not just a per-build observation.

### D.9 ReLU smoothness proof is interesting and correct

`pdiv_relu` (MLP.lean:288–363) is a ~75-line proof using
`HasFDerivAt.congr_of_eventuallyEq` to transport the diagonal-indicator
CLM's self-fderiv to ReLU on `Metric.ball x (min |x k|)`. This is the
non-trivial proof the audit prompt asked about, and it goes through
cleanly with no smoothness assumption beyond `∀ k, x k ≠ 0`.

---

## E. Proposed patches

### E.1 Add `relu_codegen_matches_canonical` to `MLP.lean`

The bridge theorem from B.1, placed after `relu_has_vjp` (around
MLP.lean:382). Promotes the canonical-witness `backward` from a
self-equality to a smooth-point equivalence to the codegen formula.
Verified to typecheck and pass the three-axiom check in
`tests/AuditBridge.lean`.

### E.2 Add analogous `maxPool2_codegen_matches_canonical` to `CNN.lean`

Same shape, but for the `select_and_scatter`-emitting MaxPool2
backward. Requires a smoothness hypothesis "no ties in any 2×2
window" and a per-window argmax computation. Not written here —
would need spelling out, but the recipe matches.

### E.3 Fix the six doc-drift sites listed in C.1

Mechanical edits — change "axioms" to "theorems" or remove the
section. Lowest-effort, highest-credibility improvement.

### E.4 Add `tests/AuditAxioms.lean` (or equivalent) to CI

The file I wrote demonstrates every headline theorem passes the
three-axiom check. Worth keeping permanently as a CI guardrail
alongside the comparator suite.

### E.5 (Optional, larger) Introduce `HasVJPAt` pointwise variant

Would let `mlp_has_vjp` be re-stated as `mlp_has_vjp_at` under input
smoothness, eliminating the canonical-witness escape entirely. This
is the only "real API addition" recommended; the others are
documentation cleanup and theorem additions. If undertaken, the
`vjp_comp_at` analogue follows the same structure as the existing
`vjp_comp` with `DifferentiableAt` swapped in.

---

## Files produced during this audit

- `tests/AuditAxioms.lean` — `#print axioms` on 47 headline theorems.
- `tests/AuditBridge.lean` — `relu_codegen_matches_canonical` + axiom check.
- `tests/AuditSanity.lean` — concrete-instance pinning examples.
- `tests/AUDIT_REPORT.md` — this report.

All four typecheck under the current build. The author can land the
bridge theorem into `MLP.lean` as a one-line move, integrate
`AuditAxioms.lean` into CI alongside the comparator, and treat the
doc-drift fixes as a single small PR.

---

## Status update — 2026-06-06

The audit above is preserved as a dated snapshot; this section supersedes it
where they differ.

**Counts.** Headline `#print axioms` checks: 47 → **114**
(`tests/AuditAxioms.lean`), all still closing under exactly
`[propext, Classical.choice, Quot.sound]`. Independent comparator re-check:
38 → **51** theorems (`tests/comparator/config.json`).

**Findings resolved.**

- **B.1 / E.1, E.2 (vacuous-witness bridge).** Landed and axiom-audited:
  `relu_codegen_matches_canonical` (`MLP.lean:446`) and
  `maxPool2_codegen_matches_canonical` (`CNN.lean:2291`).
- **B.3 / E.5 (pointwise framework).** `HasVJPAt` + `vjp_comp_at`
  (`Tensor.lean:322,342`) and `mlp_has_vjp_at` (`MLP.lean:549`) are landed and
  now load-bearing — the CNN-family whole-network VJPs compose through them.
- **B.2 (Mathlib `Matrix` bridge).** Addressed by
  `LeanMlir/Proofs/MatBridge.lean` (`Proofs.Mat` ↔ Mathlib `Matrix` via
  `Matrix.of`).
- **C / E.3 (doc drift).** The CNN "Summary of axioms" header now reads
  "Summary of derivations" (`CNN.lean:2423`).
- **E.4 (CI guardrail).** `tests/AuditAxioms.lean` runs in
  `.github/workflows/proofs.yml` as a three-axiom-closure gate.

**New since this audit — whole-network VJPs.** Every architecture capstone is
now either unconditional or concretely instantiated:

- *Unconditional* global `HasVJP` (correct at every input, only `0 < ε`):
  `vit_full_has_vjp`, `convnext_has_vjp`, `efficientnet_has_vjp`.
- *Conditional `_at` + a discharged concrete instance*: MLP (`MlpConcrete`),
  MNIST-CNN (`Spatial`/`Mini`/`Micro`), ResNet (`CnnConcrete`), MobileNetV2
  (`MobileNetV2Concrete`) — each proving its per-site smoothness bundle
  jointly satisfiable on the real forward, not vacuous.

**Still open.** The codegen↔proof link remains unproven (reference `ℝ` vs the
emitted `Float32` StableHLO); `MobileNetV2Concrete` is a degenerate
(constant-output) witness; the concrete nets are tiny by construction.
