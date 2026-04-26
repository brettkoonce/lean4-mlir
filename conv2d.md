# conv2d.md — plan for removing the conv2d / depthwise input-VJP axioms

**Status:** deferred. Two axioms remain after Phase 3 (mhsa) landed:

1. `conv2d_has_vjp3` (CNN.lean:164) — bundled VJP for the conv2d input path.
2. `depthwise_has_vjp3` (Depthwise.lean:101) — same shape, per-channel only.

Both are honestly tackleable — conv2d **is smooth** (linear in input via
`if pad-condition then x[…] else 0`, condition independent of x), so
`HasVJP3` is provable from foundation rules. Not a framework limitation.

> **One-line summary.** Removing these axioms is a real ~500–800 LOC
> calc-engineering effort (per axiom), not a quick proof. Plan it as a
> dedicated branch.

---

## Why these axioms are tackleable (and the others aren't)

Of the 8 remaining axioms, 4 are framework-level (subgradient
conventions: `pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`,
`maxPool2_has_vjp3`) and 2 are opaque-codegen (`patchEmbed_*`). The
remaining 2 — `conv2d_has_vjp3` and `depthwise_has_vjp3` — are the
**only standard-calculus axioms still left**:

```
conv2d W b x [o, ho, wo]
  = b o + Σ c kh kw, W[o,c,kh,kw] * (if pad-cond(kh,ho,kw,wo) then x[c, h_idx, w_idx] else 0)
```

The pad-condition `pH ≤ kh+ho ∧ kh+ho-pH < h ∧ pW ≤ kw+wo ∧ kw+wo-pW < w`
depends on indices `(kh, ho, kw, wo)`, **not** on the values of `x`.
So conv2d is fully smooth in `x` (in fact, linear). The same goes for
depthwise.

VJP.md historically grouped these with "non-smooth/boundary
conventions" (alongside maxPool2). That grouping is misleading: the
*boundary* is in indices not values, so the function is smooth.
The axiom is just bundled-existence-of-correct-backward, like
`mhsa_has_vjp_mat` was before Phase 3.

---

## Why the previous attempt didn't land

A first pass tried to clone the proof of `conv2d_weight_grad_has_vjp`
(CNN.lean:225, ~470 LOC, already a theorem) with the variable role
swapped from W to x. Bailed for these reasons:

1. **Inner conditional pdiv has a non-trivial case split.** In
   `conv2d_weight_grad_has_vjp`, the per-summand inner factor
   `(if pad then x[…] else 0)` is **constant in v** (the W-flat),
   so `pdiv (const) = 0` and the term collapses cleanly via
   `pdiv_const`. For input-grad, that same expression IS the
   variable: it's `(if pad then v(σ) else 0)` where σ is a specific
   reindex. As a function of v, this is a CLM (either `proj σ` if
   pad holds, or `0` if not), but its `pdiv` requires a `by_cases hpad`
   inside the per-summand computation — adding a layer that
   `conv2d_weight_grad_has_vjp` didn't have.

2. **Fin-tuple equalities for Kronecker collapse are noisy.** The
   final collapse needs equalities like
   `(cc, ⟨kh.val + ohw_hi.val − pH, h_kh_lt⟩, ⟨kw.val + ohw_wi.val − pW, h_kw_lt⟩) = (ci, hi, wi)`.
   Each component's `⟨_, h_proof⟩` introduces a Fin-equality that
   reduces to Nat-equality plus a (different, irrelevant) bound proof.
   Combined with Lean's right-associative tuple parsing and confused
   type-ascription error messages, the bookkeeping eats hours.

3. **`fun_prop` doesn't carry through the `let pH := …` bindings.**
   The conv2d body uses `let pH := (kH-1)/2; let pW := (kW-1)/2; …` —
   these `let`-bindings stay in the `pdiv_finset_sum` summand goals
   and `fun_prop` gets confused. Workaround: `unfold Tensor3.unflatten`
   first (exposes `v(fPF…)` shape), then `fun_prop`. Worked in the
   weight-grad proof but interacts oddly with the conditional-CLM
   above.

---

## Recommended architecture (Approach A: clone weight-grad + handle the case split)

**Why not approach B (CLM-based via `IsBoundedBilinearMap` of conv):**
conv2d as a bilinear map `(W, x) ↦ output` is a bounded bilinear map,
which gives joint diff "for free" via Mathlib's
`IsBoundedBilinearMap.hasFDerivAt`. From joint diff, the partial
derivatives w.r.t. each input pop out. **But** setting up
`IsBoundedBilinearMap` for conv2d requires writing the bilinear bound
explicitly (a calc proof on the operator norm) — comparable in
length to just doing the direct pdiv chain.

### Pieces to build (in order, per axiom)

| # | Piece                              | File          | Est. LOC |
|---|------------------------------------|---------------|----------|
| 1 | `conv2d_input_grad_formula` (def)  | CNN.lean      | ~15      |
| 2 | `conv2d_input_pdiv` (per-coord helper) | CNN.lean  | ~250–350 |
| 3 | `conv2d_has_vjp3` (compose pieces) | CNN.lean      | ~150–250 |
|   | **Subtotal (conv2d)**              |               | **~415–615** |
| 4 | `depthwise_input_grad_formula`     | Depthwise.lean | ~10     |
| 5 | `depthwise_has_vjp3` (parallel proof) | Depthwise.lean | ~300–450 |
|   | **Subtotal (depthwise)**           |               | **~310–460** |
|   | **Total**                          |               | **~725–1075** |

Depthwise comes out shorter because it lacks the cross-channel sum
(no `Σ c`), so the triple sum becomes a double sum. About 70% the
length of conv2d.

### Detail sketches

**Piece 1: `conv2d_input_grad_formula`.** Direct (ho, wo)-form sum,
no kernel-reverse rewrite (cleaner than the `kH-1-kh` doc-formula):

```lean
noncomputable def conv2d_input_grad_formula
    (W : Kernel4 oc ic kH kW) (dy : Tensor3 oc h w) : Tensor3 ic h w :=
  fun ci hi wi =>
    let pH := (kH - 1) / 2
    let pW := (kW - 1) / 2
    ∑ co kh kw, ...
      let ho_nat := hi.val + pH - kh.val
      let wo_nat := wi.val + pW - kw.val
      if hpad : kh.val ≤ hi.val + pH ∧ ho_nat < h ∧ kw.val ≤ wi.val + pW ∧ wo_nat < w then
        W co ci kh kw * dy co ⟨ho_nat, hpad.2.1⟩ ⟨wo_nat, hpad.2.2.2⟩
      else 0
```

**Note:** the cleanest form sums over (co, kh, kw) and reconstructs
(ho, wo) via Nat subtraction. The ho-form sums over (co, ho, wo)
and reconstructs (kh, kw). They're equivalent under the bijection
`(ho, wo) ↔ (kh, kw)` for valid pairs; either choice is fine for
the proof, but the (kh, kw)-form lines up with the doc/MLIR
"reverse-kernel" formulation.

**Piece 2: `conv2d_input_pdiv`.** Per-coord pdiv computation. Phases:

1. Decompose `flat conv2d in x` as `(constant b broadcast) + (variable W·x term)`.
   Apply `pdiv_add`; constant side gives 0.
2. Triple-distribute the W·x term over (c, kh, kw) via three
   `pdiv_finset_sum` applications.
3. Per-summand: factor as `(constant W) * (variable if-pad-conditional)`.
   `pdiv_mul` + `pdiv_const = 0` on the W factor + need `pdiv` of the
   if-pad-conditional.
4. **The case split** (the 1.5×-cost step vs weight-grad): inside
   `pdiv (if pad then v(σ) else 0)`, do `by_cases hpad`:
   - Pad holds: function is `(reindexCLM σ).toFun = fun v => v(σ)`,
     pdiv via `ContinuousLinearMap.fderiv` + `proj_apply` =
     `if idx_in = σ then 1 else 0`.
   - Pad doesn't hold: function is constant `0`, pdiv = 0.
5. Triple-collapse via `Finset.sum_eq_single` over (c, kh, kw),
   matching to `(cc, kh*, kw*) = (ci, kh-derived, kw-derived)`.

The encoding bureaucracy (Fin-tuple equalities) lives in step 5.
Pre-mitigate by phrasing the inner condition on **encoded indices**
not on Fin tuples:

```lean
-- Bad: (cc, ⟨...⟩ : Fin h, ⟨...⟩ : Fin w) = (ci, hi, wi)
-- Good: finProdFinEquiv (finProdFinEquiv (cc, ⟨...⟩), ⟨...⟩) =
--       finProdFinEquiv (finProdFinEquiv (ci, hi), wi)
```

The encoded form sidesteps Lean's tuple-parsing fragility around `:`
type ascription inside `⟨_, _⟩`.

**Piece 3: `conv2d_has_vjp3`.** Wrap pieces 1 + 2 in the
`HasVJP3` structure. The `correct` field substitutes piece 2 into
the `Σ pdiv * dy` form, then reindexes idx_out to `(co, ho, wo)` via
two `Fintype.sum_equiv`s + `sum_prod_type`. Exactly mirrors the tail
of `conv2d_weight_grad_has_vjp` (lines 644–714 of CNN.lean).

**Pieces 4 + 5: depthwise.** Same proof structure as conv2d, but with
`c = co` (no separate input-channel index, no `Σ c` outer sum).
Probably worth implementing AFTER conv2d so any helper/abbrev pattern
that emerges there can be reused.

---

## Pitfalls catalog (from the failed attempt)

These tripped the attempt. Pre-mitigate.

1. **Tuple type-ascription parse error.** The form
   `(cc, ⟨val, proof⟩ : Fin ic × Fin h, ⟨val, proof⟩ : Fin w)` doesn't
   parse — Lean reads `:` as type ascription inside the tuple. Use
   the encoded form `finProdFinEquiv (finProdFinEquiv (cc, ⟨_, _⟩), ⟨_, _⟩)`
   for tuple-equalities, not the unencoded form.

2. **`fun_prop` chokes on `let pH := …; let pW := …`.** The
   `(kH - 1) / 2` bindings inside summands confuse `fun_prop`'s
   structural traversal. Workaround: `unfold Tensor3.unflatten`
   *before* `fun_prop` so the `let`-bindings get inlined into the
   target. Worked in `conv2d_weight_grad_has_vjp`; will work here
   too. The trip-up is when this is interleaved with the case split
   on `hpad` — handle the case split *inside* `pdiv_mul`, not at
   the outer level.

3. **`HasVJP3` is on `Tensor3 → Tensor3` but `pdiv3` evaluates
   `pdiv` on the flat form.** When constructing the proof, the goal
   alternates between Tensor3 indexing `f x ci hi wi` and flat
   indexing `Tensor3.flatten (f (Tensor3.unflatten v)) (fPF (fPF (ci, hi), wi))`.
   Use `pdiv3`'s definitional unfold (`unfold pdiv3`) early and stay
   in flat form for the duration of the inner proof; convert back
   only at the outer `correct`.

4. **Three layers of `finProdFinEquiv.symm` for idx-out unpacking.**
   `Fin (oc * h * w) ↔ (Fin oc × Fin h) × Fin w` via row-major
   flatten. The outer `idx_out`'s `.1.1`, `.1.2`, `.2` decode to
   `(co, ho, wo)`. `set ohw_o := …; set ohw_hi := …; set ohw_wi := …`
   early (top of the per-coord pdiv proof) and use those bindings
   throughout — cleaner than re-deriving each time.

5. **Nat subtraction interlocking with Fin bounds.** Phrase the
   `kh_nat := hi.val + pH − ho.val` form with the *Nat* expression,
   then `Fin kH`-bound it via the `hpad.2.1` proof. Don't try to
   pre-construct `Fin kH` directly — Lean's elaborator can't
   discharge `kh_nat < kH` without the explicit `hpad.2.1` proof in
   scope.

6. **Don't try to be slick with `IsBoundedBilinearMap`.** Tempting
   because conv2d *is* bilinear, but the operator-norm bound proof
   for the bilinear is comparable in length to the direct pdiv
   chain, and `IsBoundedBilinearMap.hasFDerivAt` only gives joint
   diff — you still need to extract the per-input partial
   derivatives. Net: no savings.

---

## Suggested execution order

Aim for two commits per axiom:

1. **Commit 1 (conv2d, ~500 LOC).** `conv2d_input_grad_formula` def +
   `conv2d_input_pdiv` helper + `conv2d_has_vjp3` (theorem replacing
   the axiom). Run `#print axioms conv2d_has_vjp3` to verify pure-Mathlib
   closure.
2. **Commit 2 (depthwise, ~300 LOC).** Same pattern. Reuse any
   helpers from conv2d (e.g., the if-pad-conditional pdiv lemma).
3. **VJP.md update + mhsa.md follow-up.** Update axiom inventory
   `8 → 6`. The 6 remaining are 3 ReLU subgradient + 1 maxPool
   subgradient + 2 patchEmbed opaque-codegen — none tackleable
   without framework-level changes.

After both, **axiom count drops 8 → 6**. The remaining 6 are pure
"framework convention" axioms (subgradient cluster + opaque codegen).

---

## Time budget

Realistic estimate after the failed first pass:

- Phase 1 (conv2d): ~12–16 focused hours (1.5–2 working days). The
  case split on `hpad` is the surprise vs the weight-grad template
  — adds about 30% to the inner-pdiv proof. Encoding bureaucracy
  is ~30% of total LOC.
- Phase 2 (depthwise): ~6–8 hours. Faster because the proof
  structure is locked in and the helper lemmas from Phase 1 carry
  over.

**Total: ~18–24 focused hours, ~2.5–3 working days.**

Significantly more than mhsa Phase 3 (~6 hours) because the
case-split-on-hpad is a fundamentally harder inner step than mhsa's
`pdivMat_mhsa_g_split` chain rule (which had a cleaner
`HasFDerivAt` composition story).

---

## When to pick this back up

The right time is when:

1. You have a clean ~3-day window for focused proof work. The
   encoding bureaucracy needs momentum.
2. You've reviewed `conv2d_weight_grad_has_vjp` and
   `conv2d_bias_grad_has_vjp` (CNN.lean:225, 733) recently so the
   pdiv-distribution pattern is fresh.
3. You can stomach 500+ LOC of mechanical Lean proof. This is not
   creative math — it's calc engineering.

The bad time: as a side-quest during another task. Same lesson as
mhsa Phase 3.

---

## What `conv2d_has_vjp3` removal *would* buy us

- 8 → 6 project axioms.
- The book's claim "verified ResNet/MobileNet" becomes literal
  rather than "verified modulo this one bundled axiom about conv's
  input-path Jacobian" — same upgrade as Phase 3 was for ViT.
- Buzzard would stop pointing at it as the obvious next axiom to
  kill (assuming he was; conv2d/depthwise are the only
  smooth-but-axiomatic remnants).

The cost (2.5–3 working days, ~700+ LOC) versus that benefit is the
question to answer when scheduling. Like Phase 3 (mhsa) before it,
this is upgrade work, not correctness work — the current 8-axiom
state with a published `#print axioms` showing pure-Mathlib closure
modulo 8 named conventions is honestly defensible.

---

## What's left after this hypothetical phase

If both `conv2d_has_vjp3` and `depthwise_has_vjp3` land, the floor
becomes **6 axioms**:

- 3 ReLU subgradient (`pdiv_relu`, `relu_has_vjp`, `mlp_has_vjp`).
- 1 maxPool2 subgradient (`maxPool2_has_vjp3`).
- 2 patchEmbed opaque-codegen (`patchEmbed_flat_has_vjp`, `patchEmbed_flat_diff`).

The first 4 need a `HasVJP.correct` weakening to "smooth subset only"
— a project-wide rewrite, separate multi-week effort.

The last 2 could come down by **de-opaquing `patchEmbed_flat`**:
unseal the `noncomputable opaque` def and write the forward
concretely (conv + reshape + CLS prepend + pos embed). Then `_diff`
falls out by composition (~200 LOC), and `_has_vjp` follows via
composing the per-step VJPs (`conv2d_has_vjp3` + reindex VJPs +
biPath VJP for CLS/pos addition). De-opaquing is ~300 LOC plus
~500 LOC of composition proofs. Possible but not in scope here.

So the genuinely-immovable floor without framework changes is **4
axioms** (the ReLU/maxPool subgradient cluster). Everything else is
calc engineering of varying difficulty.
