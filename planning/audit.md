# Audit follow-ups — wrap-up

Parallel-agent audit (`tests/AUDIT_REPORT.md`) flagged five proposed
patches. **All five have landed.** This doc is now historical record.

## Status snapshot

| Patch | Status      | Commit    | Notes                                                |
|-------|-------------|-----------|------------------------------------------------------|
| E.3   | ✅ Landed   | `281598f` | 12 doc-drift fixes across 7 files; 1 broken xref     |
| E.1   | ✅ Landed   | `1c6d18c` | `relu_codegen_matches_canonical` + diagonal restate  |
| E.4   | ✅ Landed   | `122f487` | CI three-axiom closure check; 49/49 conforming       |
| E.2   | ✅ Landed   | `957b389` | `maxPool2_codegen_matches_canonical` + smooth pdiv3  |
| E.5   | ✅ Landed   | (this PR) | `HasVJPAt` pointwise framework + 3 kinked instances  |

CI is green; the three-axiom closure invariant now stands at **54/54**.

---

## E.2 — `maxPool2_codegen_matches_canonical` ✅ Landed

**Outcome.** Landed as `pdiv3_maxPool2_smooth` +
`maxPool2_codegen_matches_canonical` in `LeanMlir/Proofs/CNN.lean`,
plus the supporting infrastructure (window helpers, `MaxPool2Smooth`,
`MaxPool2IsArgmax`, `maxPool2Argmax`, `maxPool2LocalReindex`,
`maxPool2_flat_hasFDerivAt`). ~493 lines. Used path A2 (Vec-level
`reindexCLM` directly) — self-contained, didn't need the Tensor3-flatten
CLM detour A1 had been sketched for.

**Recipe for historical reference** (the path that worked):

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

## E.5 — `HasVJPAt` pointwise framework ✅ Landed

**Outcome.** Kills the vacuous `correct := rfl` escape at the three
problematic kinked operators (`relu_has_vjp`, `mlp_has_vjp`,
`maxPool2_has_vjp3`) by introducing pointwise-smoothness variants.
~265 lines across `Tensor.lean`, `MLP.lean`, `CNN.lean`,
`tests/AuditAxioms.lean`.

**Shipped:**

- `HasVJPAt` (Vec) and `HasVJPAt3` (Tensor3) structures —
  same pdiv-sum contract as `HasVJP` / `HasVJP3`, but only at a chosen
  smooth point.
- `HasVJP.toHasVJPAt` / `HasVJP3.toHasVJPAt3` — trivial lift, lets the
  ~40 globally-smooth instances participate without modification.
- `vjp_comp_at` — chain rule under `DifferentiableAt` (not global
  `Differentiable`). The piece that lets composition pass through
  ReLU at smooth inputs.
- `reluLinearPart` + `relu_hasFDerivAt` (refactored out of
  `pdiv_relu`) + `relu_differentiableAt_of_smooth`.
- `dense_differentiable` (global) — clears the chain-rule diff
  obligations for the dense layers.
- `relu_has_vjp_at` — backward is the codegen `if x i > 0 then dy i
  else 0` directly; `correct` is `pdiv_relu` + sum-collapse, not `rfl`.
- `mlp_has_vjp_at` — four nested `vjp_comp_at` calls through
  `dense → relu_at → dense → relu_at → dense`. No `rfl` in `correct`.
- `maxPool2_has_vjp_at3` — backward is the `select_and_scatter`
  formula directly; `correct` is `maxPool2_codegen_matches_canonical`
  flipped, not `rfl`.

**Not shipped (audit Step 5, marked optional):**

- `vjp_comp_at3` (Tensor3 chain rule under DifferentiableAt) and a
  full `cnn_has_vjp_at3` composing every CNN layer. Would need
  smoothness propagation through `conv2d` as well — a separate lift.
  The three kinked-operator instances above are the load-bearing fix
  the audit called for; the Tensor3 chain rule is additive on top.

---

## Files / commits to know about

**Audit artifacts (in repo, on `main`):**
- `tests/AUDIT_REPORT.md` — original parallel-agent audit writeup
- `tests/AuditAxioms.lean` — `#print axioms` over 49 headline theorems
  (the CI guardrail's data source)
- `tests/AuditBridge.lean` — independent kernel-side re-verification
  of the ReLU smooth-point bridge (mirrors what landed in MLP.lean)
- `tests/AuditSanity.lean` — concrete-instance pinning examples

**Production code (touched by E.1 / E.2 / E.3 / E.5):**
- `LeanMlir/Proofs/Tensor.lean` — `HasVJPAt` / `HasVJPAt3` structures,
  `vjp_comp_at` chain rule, lift helpers (E.5)
- `LeanMlir/Proofs/MLP.lean` — `relu_codegen_matches_canonical` +
  `relu_canonical_diagonal` (E.1); `reluLinearPart` +
  `relu_hasFDerivAt` + `relu_differentiableAt_of_smooth` +
  `dense_differentiable` + `relu_has_vjp_at` + `mlp_has_vjp_at` (E.5)
- `LeanMlir/Proofs/CNN.lean` — `maxPool2_has_vjp3` (kept; canonical
  witness), `pdiv3_maxPool2_smooth` + `maxPool2_codegen_matches_canonical`
  + supporting infra (E.2); `maxPool2_has_vjp_at3` (E.5)

**CI:**
- `.github/workflows/proofs.yml` — three-axiom closure check via
  `lake env lean tests/AuditAxioms.lean`. Pattern:
  ```
  grep -cE "depends on axioms: \[propext, Classical\.choice, Quot\.sound\]$"
  ```
  Must match the total `depends on axioms:` line count.

