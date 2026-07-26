# Phase 2: `Proofs.Mat` → Mathlib `Matrix`

Scoping note for the *full* migration of the suite's home-grown `Mat` onto
Mathlib's `Matrix`. **Phase 1 (shipped)** is the opt-in bridge in
`LeanMlir/Proofs/MatBridge.lean` — four lemmas relating
`Mat.mulVec`/`outer`/`mul`/`transpose` to Mathlib's `*ᵥ`/`vecMulVec`/`*`/`ᵀ`,
leaving every existing proof untouched. **Phase 2 (this doc)** is the deeper
refactor: make `Mat` *be* `Matrix` and delete the parallel vocabulary.

## Design

```lean
abbrev Mat m n := Matrix (Fin m) (Fin n) ℝ   -- was: Fin m → Fin n → ℝ
```

`Matrix (Fin m) (Fin n) ℝ` is *definitionally* `Fin m → Fin n → ℝ`, so the
type itself doesn't change shape. Keeping `Mat` as an `abbrev` means the
~525 `Mat`-as-type signatures don't churn. Delete `Mat.mul`/`mulVec`/
`outer`/`transpose`; rewrite call sites to `*` / `*ᵥ` / `Matrix.vecMulVec` /
`ᵀ`.

## Footprint (measured)

| Thing | Count | Notes |
|---|---|---|
| Op call sites — `Mat.mul` 58, `transpose` 40, `outer` 10, `mulVec` 6 | **114** | 73 of them in `Attention.lean` |
| `Mat`-as-type signatures | 525 | unchanged (abbrev keeps the name) |
| Proofs that explicitly `unfold`/`simp [Mat.<op>]` | ~5 | rewrite to `Matrix.mul_apply` / `transpose_apply` |
| `pdivMat` matmul/transpose Jacobian lemmas | 3 | statements name `Mat.mul`/`transpose`; restate + fix proofs |
| `MatBridge.lean` (phase 1) | — | becomes moot; delete or fold a couple into `@[simp]` |

**Out of scope:** `Tensor3` / `Kernel4` / `DepthwiseKernel` (683 uses) stay
custom — Mathlib has no clean 3D/4D tensor with the ops the CNN/attention
code needs. Phase 2 is strictly the 2D layer.

`pdivMat` itself is unaffected structurally: it's defined via
`flatten`/`unflatten` over the function form, which `Matrix` still satisfies
by defeq (`A i j` indexing, `fun i j => …` construction). Only the 3
Jacobian lemmas whose *statements* mention `Mat.mul`/`transpose` need
touching.

## The actual cost driver: the `Matrix.of` / simp-drift tax

The 114 call sites are mechanical. The real cost is that the suite builds
matrices from index lambdas everywhere — ~30–40 sites of
`let scores : Mat n n := fun i j => …`, concentrated in SDPA. Under
`Mat = Matrix`:

- matrix *construction* from a raw lambda picks up `Matrix.of` / `Matrix.of_apply`
  friction;
- `simp` calls that currently rely on `Mat.mul` reducing to a raw `∑ … * …`
  will drift, because `A * B` doesn't unfold the same way — each needs
  `Matrix.mul_apply` threaded in.

**This is exactly the friction the custom `Mat` was built to avoid.**
Mathlib's `Matrix` is tuned for algebra (rings, determinants, eigenstuff),
not index-level computation. Phase 2 swims against that current in the one
file where it's worst: `Attention.lean` (3792 lines, 54 matrix-building
defs, the intricate SDPA chain-rule proofs).

## Estimate

**2–4 days**, ~80% a build-fix grind in `Attention.lean`.

- **Floor (~1 day):** mechanical call-site rewrite + the ~8 explicit proof sites.
- **Ceiling (~4–5 days):** if simp-normal-form drift cascades through the
  SDPA proofs.
- **Risk: low.** The build is the oracle — correctness can't silently break;
  it type-checks or it doesn't. Tedious, not dangerous.
- **New capability: none.** Pure readability / Mathlib-nativeness.

## Recommendation

**Defer unless it's for publication polish.** Phase 1 already gives consumers
the interop. The ROI on the full rewrite is "reads as native Mathlib" against
2–4 days of papercuts in the most delicate file.

If partial value is wanted cheaply: mark the `MatBridge` lemmas `@[simp]`, so
a downstream proof can `simp` its way onto Mathlib's `Matrix` API on demand —
interop without editing a single existing proof.

## Verification numbers (as of this writing)

```
git grep -hoE 'Mat\.(mul\b|mulVec|transpose|outer)' LeanMlir/Proofs/*.lean | wc -l   # 114
git grep -hoE '\bMat [0-9A-Za-z(]' LeanMlir/Proofs/*.lean | wc -l                     # 525
git grep -nE '(simp|unfold|rw|dsimp)[^A-Za-z]*Mat\.(mul\b|mulVec|transpose|outer)' \
  LeanMlir/Proofs/*.lean   # ~5 sites outside MatBridge.lean
```
