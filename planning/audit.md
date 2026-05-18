# Audit follow-ups (E.2 / E.5) — clean-session handoff

Parallel-agent audit (`tests/AUDIT_REPORT.md`) flagged five proposed patches.
Three landed on **2026-05-18**. Two remain for a future focused session.

## Status snapshot

| Patch | Status      | Commit    | Notes                                                |
|-------|-------------|-----------|------------------------------------------------------|
| E.3   | ✅ Landed   | `281598f` | 12 doc-drift fixes across 7 files; 1 broken xref     |
| E.1   | ✅ Landed   | `1c6d18c` | `relu_codegen_matches_canonical` + diagonal restate  |
| E.4   | ✅ Landed   | `122f487` | CI three-axiom closure check; 49/49 conforming       |
| E.2   | ⏸ Deferred | —         | MaxPool2 smooth-point bridge (this doc)              |
| E.5   | ⏸ Deferred | —         | `HasVJPAt` pointwise framework (this doc)            |

CI is green; the three-axiom closure invariant is now permanent.

---

## E.2 — `maxPool2_codegen_matches_canonical`

**Goal.** Prove that at points where every 2×2 window of `x` has a
unique strict argmax, the canonical pdiv-derived backward in
`maxPool2_has_vjp3` (in `LeanMlir/Proofs/CNN.lean:1509`) collapses to
the codegen formula ("route `dy` to the argmax position, zero
elsewhere"). Mirrors `relu_codegen_matches_canonical` (in
`LeanMlir/Proofs/MLP.lean:382`), but the local linearization is
per-2×2-window rather than per-coordinate.

### Why E.1 was 30 lines and E.2 isn't

E.1 was cheap because `pdiv_relu` (the smooth-point Jacobian, 75 lines
in `MLP.lean:288`) already existed in the codebase. The bridge theorem
was just a 30-line sum-collapse on top.

E.2 has no analogous `pdiv3_maxPool2`. Worse: pdiv_relu operates on
`Vec`, but `pdiv3` (used by maxPool's `HasVJP3`) operates on
`Tensor3` via the flatten bijection. So E.2 needs *both* the smooth-
point Jacobian *and* the Tensor3↔Vec bridge step that pdiv_relu didn't
have to do.

### Estimated effort

**~250 lines of dense Lean, 6–10 focused hours.** Should happen in
one session to keep the context loaded.

### Recipe

#### Step 1 — Window helpers (done, ~50 lines)

Already drafted in the framework section below: `winRow`, `winRowMod`,
`winRowInv` plus column counterparts and four roundtrip lemmas. Pick
these up verbatim and drop them into `CNN.lean` near `maxPool2`.

#### Step 2 — Smoothness predicate (done, ~10 lines)

`MaxPool2Smooth x` and `MaxPool2IsArgmax x ci hi_in wi_in`. Both
drafted below.

#### Step 3 — Local Vec-level CLM (~50 lines, the new infrastructure)

The choice is between two infrastructure paths; **A1 is preferred** as
it's reusable for future smooth-point bridges of other kinked operators.

**Path A1 — Expose `Tensor3.flatten` and `Tensor3.unflatten` as CLMs.**
The existing `Tensor3.flatten` in `LeanMlir/Proofs/Tensor.lean:1441` is
a coordinate permutation; it's a `ContinuousLinearMap` in disguise.
Build `flatten_clm : Tensor3 c h w →L[ℝ] Vec (c*h*w)` and `unflatten_clm`
with a `flatten_clm_apply` lemma showing they agree with the existing
defs. Then E.2 can work at the Tensor3 level (cleaner) and lift via
`HasFDerivAt.comp` with `flatten_clm.hasFDerivAt`.

**Path A2 — Build the full Vec-level CLM in-place.** Skip the CLM
infrastructure; construct the local linearization directly at the
flattened-Vec level as `ContinuousLinearMap.pi` indexed by output
flat indices, each entry being `ContinuousLinearMap.proj` at the
argmax's input flat index. Self-contained but doesn't help other
smooth-point bridges later.

#### Step 4 — Argmax position (~15 lines)

Under smoothness, every window has a unique argmax. Extract via
`Classical.choose` applied to `Finset.exists_max_image` over
`(Finset.univ : Finset (Fin 2 × Fin 2))`. Plus a smoothness-uniqueness
lemma showing the chosen pair is unique.

#### Step 5 — Local agreement (~80 lines, the dense part)

Within a ball of radius `r = min over windows of (max − second-max)`,
no window's argmax flips. Specifically: for each output window
`(ci, ho, wo)`, the per-window margin `m_w = max - second_max > 0`
under smoothness. Take `r = min m_w` over all windows (positive by
`Finset.inf'_lt_iff`). Within `Metric.ball x r`, the four values in
each window are perturbed by less than `r/2`, so the strict argmax
order is preserved.

This is structurally the same as pdiv_relu's `h_local` step
(`MLP.lean:319-346`), but with per-window combinatorics instead of
per-coordinate sign analysis.

#### Step 6 — `HasFDerivAt` via `congr_of_eventuallyEq` (~5 lines)

Standard pattern: `(phi.hasFDerivAt).congr_of_eventuallyEq h_evt`.

#### Step 7 — Compute pdiv3 at smooth points (~20 lines)

`pdiv3 maxPool2 x ci hi_in wi_in co ho wo = if (co, ho, wo) is the
window of (hi_in, wi_in) AND (hi_in, wi_in) is the argmax then 1
else 0`. This is the smooth-point Jacobian; mirror `pdiv_relu`'s
shape.

#### Step 8 — Sum collapse (~30 lines)

Three nested `Finset.sum_eq_single` calls to collapse the
`co, ho, wo` triple sum to the unique non-zero term. The pattern in
`MLP.lean:382-410` is the template.

#### Step 9 — Land it

- Port to `LeanMlir/Proofs/CNN.lean` near `maxPool2_has_vjp3`
- Update `relu_has_vjp`-style docstring on `maxPool2_has_vjp3`
  pointing at the new bridge (mirrors `MLP.lean:373-378`)
- Add the two new theorems to `tests/AuditAxioms.lean`
- `lake build` + verify `#print axioms` closes under
  `[propext, Classical.choice, Quot.sound]`
- CI E.4 guardrail will catch any regression automatically

### Framework already drafted (use verbatim)

The helpers and predicates below were drafted on 2026-05-18 in a
scratch file `tests/AuditMaxPoolBridge.lean` (left untracked because
of the trailing `sorry`). Copy these into `CNN.lean` to skip Step 1
and Step 2.

```lean
-- Window-index helpers

def winRow {h : Nat} (hi_in : Fin (2 * h)) : Fin h :=
  ⟨hi_in.val / 2, by have := hi_in.isLt; omega⟩

def winRowMod {h : Nat} (hi_in : Fin (2 * h)) : Fin 2 :=
  ⟨hi_in.val % 2, by omega⟩

def winCol {w : Nat} (wi_in : Fin (2 * w)) : Fin w :=
  ⟨wi_in.val / 2, by have := wi_in.isLt; omega⟩

def winColMod {w : Nat} (wi_in : Fin (2 * w)) : Fin 2 :=
  ⟨wi_in.val % 2, by omega⟩

def winRowInv {h : Nat} (hi_out : Fin h) (a : Fin 2) : Fin (2 * h) :=
  ⟨2 * hi_out.val + a.val, by have := hi_out.isLt; have := a.isLt; omega⟩

def winColInv {w : Nat} (wi_out : Fin w) (b : Fin 2) : Fin (2 * w) :=
  ⟨2 * wi_out.val + b.val, by have := wi_out.isLt; have := b.isLt; omega⟩

theorem winRowInv_winRow {h : Nat} (hi_in : Fin (2 * h)) :
    winRowInv (winRow hi_in) (winRowMod hi_in) = hi_in := by
  apply Fin.ext
  show 2 * (hi_in.val / 2) + hi_in.val % 2 = hi_in.val
  omega

theorem winColInv_winCol {w : Nat} (wi_in : Fin (2 * w)) :
    winColInv (winCol wi_in) (winColMod wi_in) = wi_in := by
  apply Fin.ext
  show 2 * (wi_in.val / 2) + wi_in.val % 2 = wi_in.val
  omega

theorem winRow_winRowInv {h : Nat} (ho : Fin h) (a : Fin 2) :
    winRow (winRowInv ho a) = ho := by
  apply Fin.ext
  show (2 * ho.val + a.val) / 2 = ho.val
  have := a.isLt; omega

theorem winRowMod_winRowInv {h : Nat} (ho : Fin h) (a : Fin 2) :
    winRowMod (winRowInv ho a) = a := by
  apply Fin.ext
  show (2 * ho.val + a.val) % 2 = a.val
  have := a.isLt; omega

theorem winCol_winColInv {w : Nat} (wo : Fin w) (b : Fin 2) :
    winCol (winColInv wo b) = wo := by
  apply Fin.ext
  show (2 * wo.val + b.val) / 2 = wo.val
  have := b.isLt; omega

theorem winColMod_winColInv {w : Nat} (wo : Fin w) (b : Fin 2) :
    winColMod (winColInv wo b) = b := by
  apply Fin.ext
  show (2 * wo.val + b.val) % 2 = b.val
  have := b.isLt; omega

-- Smoothness + argmax predicates

def MaxPool2Smooth {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w)) : Prop :=
  ∀ (ci : Fin c) (hi_out : Fin h) (wi_out : Fin w)
    (ab ab' : Fin 2 × Fin 2), ab ≠ ab' →
    x ci (winRowInv hi_out ab.1) (winColInv wi_out ab.2) ≠
    x ci (winRowInv hi_out ab'.1) (winColInv wi_out ab'.2)

def MaxPool2IsArgmax {c h w : Nat} (x : Tensor3 c (2 * h) (2 * w))
    (ci : Fin c) (hi_in : Fin (2 * h)) (wi_in : Fin (2 * w)) : Prop :=
  ∀ a b : Fin 2,
    x ci (winRowInv (winRow hi_in) a) (winColInv (winCol wi_in) b) ≤
    x ci hi_in wi_in

-- Bridge theorem statement (the proof is Steps 3-8 above)

theorem maxPool2_codegen_matches_canonical {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w))
    (h_smooth : MaxPool2Smooth x) (dy : Tensor3 c h w)
    (ci : Fin c) (hi_in : Fin (2 * h)) (wi_in : Fin (2 * w)) :
    (maxPool2_has_vjp3 :
        HasVJP3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)).backward
        x dy ci hi_in wi_in
    = (if MaxPool2IsArgmax x ci hi_in wi_in
       then dy ci (winRow hi_in) (winCol wi_in) else 0) := by
  -- TODO: Steps 3-8 above
  sorry
```

---

## E.5 — `HasVJPAt` pointwise framework

**Goal.** Eliminate the canonical-witness `correct := rfl` escape for
the three operators that use it problematically (`relu_has_vjp`,
`mlp_has_vjp`, `maxPool2_has_vjp3`). Introduces a pointwise-smoothness
variant of the entire VJP framework. The user's framing: "same
load-bearing structure, reached a more formal/defensible way."

### Estimated effort

**~200–500 lines new framework, 1–2 days.** Additive — doesn't break
existing proofs. Smooth-operator HasVJP instances (~40 of them: dense,
BatchNorm, GELU, etc.) keep their `correct := rfl` pattern because
there it's non-vacuous.

### What remains after E.5

`correct := rfl` is **still** load-bearing wherever the pdiv-sum IS the
chain-rule answer (~40 smooth-operator instances). What disappears is
the **use of `correct := rfl` to dodge a real proof obligation** at
points where the operator isn't differentiable. The framework's
invariant becomes stronger: every `correct` field either (a) chains
through `fderiv` at a smooth operator where the pdiv-sum is the chain-
rule answer, or (b) discharges under an explicit pointwise smoothness
hypothesis. No more "true but vacuous" hiding in the middle.

### Recipe

#### Step 1 — Core structure (~20 lines)

```lean
structure HasVJPAt {α β} [NormedAddCommGroup α] [NormedSpace ℝ α]
    [NormedAddCommGroup β] [NormedSpace ℝ β]
    (f : α → β) (x : α) where
  backward : β → α
  correct  : ∀ dy, /* the same pdiv-equality but only at x */
```

Probably belongs in a new file `LeanMlir/Proofs/HasVJPAt.lean`.

#### Step 2 — Chain rule and structural lemmas (~80 lines)

- `vjp_comp_at : HasVJPAt f x → HasVJPAt g (f x) →
    DifferentiableAt ℝ f x → DifferentiableAt ℝ g (f x) →
    HasVJPAt (g ∘ f) x`. Mirror of `vjp_comp` but with `DifferentiableAt`.
- `vjp_id_at`, `vjp_const_at` — trivial.
- `biPath_has_vjp_at`, `elemwiseProduct_has_vjp_at` — fan-in / fan-out
  analogues for pointwise smoothness.

#### Step 3 — Pointwise analogues for kinked operators (~50 lines)

- `relu_has_vjp_at x (h : ∀ k, x k ≠ 0)` — real proof via `pdiv_relu`,
  no canonical witness. The bridge theorem
  `relu_codegen_matches_canonical` becomes a property of this `_at`
  instance.
- `maxPool2_has_vjp3_at x (h : MaxPool2Smooth x)` — requires E.2 first
  (uses `pdiv3_maxPool2_at_smooth` from E.2).

#### Step 4 — `mlp_has_vjp_at` composed (~30 lines)

`HasVJPAt (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x` under input smoothness,
composed via `vjp_comp_at` through dense→relu→dense→relu→dense. No
canonical-witness escape. Replaces the vacuous `mlp_has_vjp.correct`
at smooth inputs with a real proof.

#### Step 5 — Trickle through Part 2 chapter proofs (~20 lines)

Optional: add `_at` variants of `cnn_has_vjp3`, `bn_has_vjp`, etc., to
restore composition through ReLU/maxpool in the larger chains. Most
likely scoped per-architecture as needed rather than blanket coverage.

### Dependency

E.2 must land before E.5's `maxPool2_has_vjp3_at` step. The `_at`
analogues of other kinked operators (just maxpool, really — dense /
softmax / layernorm / GELU are globally smooth) all reduce to the
same pattern.

---

## Files / commits to know about

**Audit artifacts (in repo, on `main`):**
- `tests/AUDIT_REPORT.md` — original parallel-agent audit writeup
- `tests/AuditAxioms.lean` — `#print axioms` over 49 headline theorems
  (the CI guardrail's data source)
- `tests/AuditBridge.lean` — independent kernel-side re-verification
  of the ReLU smooth-point bridge (mirrors what landed in MLP.lean)
- `tests/AuditSanity.lean` — concrete-instance pinning examples

**Production code (touched by E.1 / E.3):**
- `LeanMlir/Proofs/MLP.lean:382` — `relu_codegen_matches_canonical` +
  `relu_canonical_diagonal` (E.1)
- `LeanMlir/Proofs/CNN.lean:1509` — `maxPool2_has_vjp3` (still
  canonical-witness `correct := rfl`; E.2 target)
- `LeanMlir/Proofs/Tensor.lean:1441` — `Tensor3.flatten` (E.2 path A1's
  CLM target)

**CI:**
- `.github/workflows/proofs.yml` — three-axiom closure check via
  `lake env lean tests/AuditAxioms.lean`. Pattern:
  ```
  grep -cE "depends on axioms: \[propext, Classical\.choice, Quot\.sound\]$"
  ```
  Must match the total `depends on axioms:` line count.

**Local scratch (NOT in repo, regenerable from this doc):**
- `tests/AuditMaxPoolBridge.lean` — left as 90-line framework with a
  trailing `sorry` from the deferred E.2 session

---

## When you come back

1. Read this doc top to bottom (5 min).
2. Recreate `tests/AuditMaxPoolBridge.lean` from the framework block
   above (or pull the untracked file from local if still there).
3. Pick E.2 path: **A1 (flatten as CLM) is preferred** because it's
   reusable.
4. Implement Steps 3–8 in the scratch file. `lake env lean` is your
   fast iteration target; only port to `CNN.lean` once clean.
5. After E.2 lands, E.5 unblocks; do that as a separate session.

Standing rules (from memory) still apply: pause for explicit
`yes push` on every commit; rebuild blueprint PDF only if blueprint
files change (this work is pure Lean).
