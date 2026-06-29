# Planning — Newton–Schulz convergence: the iteration *computes* the polar factor

The capstone of the Muon-geometry ladder ([[muon-geometry-proof-ladder]],
`planning/muon_geometry.md`, `LeanMlir/Proofs/MuonGeometry.lean`). L1–L6 proved that the polar factor
`UVᵀ` is the *right object* — steepest descent under the operator norm (L3, von Neumann), the nuclear
norm's argmax, Shampoo's update (L5), the nearest orthogonal matrix to `G` (L6). **What's still open
is that the implementation actually computes it:** Muon's 5-step Newton–Schulz matmul iteration
`X ← aX + b(XXᵀ)X + c(XXᵀ)²X` converges to `UVᵀ`. This doc plans that proof. It closes the loop —
*implementation* (matmul iteration) ⟶ *object* (`UVᵀ`) ⟶ *theory* (L3–L6 optimality).

The implemented iteration: `OptimizerKind.muon` (`LeanMlir/Types.lean:319`), `emitMuonUpdate`, the
`den=` Newton–Schulz scaffold; design notes in `planning/muon.md` §"Newton–Schulz".

## 0. The one idea — Newton–Schulz is a *scalar* iteration in disguise

The iteration **never touches the singular directions, only the singular values.** If `X = U Σ Vᵀ`
(`U,V` orthonormal, `Σ = diagonal σ`), then since `XXᵀ = U Σ² Uᵀ`:
```
(XXᵀ)X = U Σ³ Vᵀ,   (XXᵀ)² X = U Σ⁵ Vᵀ   ⟹   p(X) = U (diagonal (aσ + bσ³ + cσ⁵)) Vᵀ.
```
So one NS step is the **scalar map `φ(t) = a t + b t³ + c t⁵` applied per singular value**, with `U,V`
carried along unchanged. Hence
```
X_k = U (diagonal (φ^[k] ∘ σ)) Vᵀ,    and    X_k → U Vᵀ   ⟺   φ^[k](σ_i) → 1  for every i.
```
**Matrix convergence reduces to scalar convergence of a fixed-point iteration.** This is exactly the
spectral-conjugation machinery already built for L5 (`conj_diag_pow`: `(W·diag d·Wᵀ)^k = W·diag(dᵏ)·Wᵀ`)
— the same `U Σ Vᵀ ↦ U f(Σ) Vᵀ` pattern, now for the polynomial `p` instead of a power.

## 1. Layered plan (repo style: spectral reduction first, then the scalar analysis)

New file `LeanMlir/Proofs/MuonNewtonSchulz.lean`, in the `Proofs` lib roots, audited in
`tests/AuditAxioms.lean`, target 3-axiom clean (`propext / Classical.choice / Quot.sound`).

- ✅ **P1 — the spectral step lemma** (the bridge; reuses L5's machinery). DONE 2026-06-29,
  `LeanMlir/Proofs/MuonNewtonSchulz.lean`, both theorems 3-axiom clean (`propext/Classical.choice/
  Quot.sound`), wired into `lakefile.lean` + `tests/AuditAxioms.lean`. `nsStep a b c X :=
  a • X + b • (X * Xᵀ * X) + c • (X * Xᵀ * (X * Xᵀ * X))` (quintic associated as `(XXᵀ)·((XXᵀ)X)` so
  the collapse threads), `nsScalar a b c t := a*t + b*t³ + c*t⁵`, and `nsStep_spectral`:
  `nsStep a b c (U * diagonal σ * Vᵀ) = U * diagonal (fun i => nsScalar a b c (σ i)) * Vᵀ` for `U,V`
  orthonormal. Pure `UᵀU=1`/`VᵀV=1` collapse algebra (`conj_diag_pow` motif done inline as `hcollapse`):
  `XXᵀ = U·diag(σ²)·Uᵀ`, cubic/quintic monomials collapse to `U·diag(σ³/σ⁵)·Vᵀ`, the three coeffs ride
  through `•` (`hsmul`) and sum pointwise (`hsum3`) to `φ`. **The only matrix-level work.** Iterated:
  `nsStep_iterate_spectral` — `nsStep^[k] (U diagσ Vᵀ) = U·diag(fun i => (nsScalar a b c)^[k] (σ i))·Vᵀ`
  by induction reusing `nsStep_spectral` (`Function.iterate_succ_apply'`). ⟹ matrix convergence to `UVᵀ`
  reduces to scalar `φ^[k](σᵢ)→1`, the entry point for P2/P3.
- ⬜ **P2 — scalar convergence (do the CUBIC first — see §4).** For the classic inverse-free polar
  iteration `g(t) = (3t − t³)/2` (i.e. `a,b,c = 3/2, −1/2, 0`), prove `∀ t₀ ∈ (0,1], g^[k](t₀) → 1`.
  The clean monotone argument: on `[0,1]`, `g` is increasing (`g'(t) = (3−3t²)/2 ≥ 0`), `g(t) ≥ t`
  (`g(t)−t = t(1−t²)/2 ≥ 0`), `g(t) ≤ 1` (`1 − g(t) = (1−t)²(2+t)/2 ≥ 0`), and `g(1) = 1`. So the
  orbit is monotone ↑, bounded above by 1 ⟹ converges; the limit is a fixed point of the continuous
  `g` in `[t₀,1]`, and `1` is the only one there ⟹ limit `= 1`.
- ⬜ **P3 — assemble matrix convergence.** `X_k = U·diag(g^[k]∘σ)·Vᵀ → U·diag(1)·Vᵀ = U·1·Vᵀ = UVᵀ`.
  From P2 (each diagonal entry → 1) + continuity of `diagonal` and matrix `*` (finite-dim, entrywise
  topology) ⟹ `Tendsto (fun k => nsStep^[k] G) atTop (𝓝 (U * Vᵀ))`. Needs `G` pre-normalized so all
  `σ_i ∈ (0,1]` (the implementation's `G / ‖G‖` step; state as a hypothesis `∀ i, σ i ∈ (0,1]`).
  Caveat: `σ_i = 0` stays `0` (`φ(0)=0`), giving the **partial isometry** `U·diag(1_{σ>0})·Vᵀ`, the
  "semi-orthogonal" limit of `Types.lean:319` — handle full-rank (`σ_i>0`) for the clean `→ UVᵀ`.
- ⬜ **P4 — the Jordan quintic, as a finite-step BAND bound** (harder; honest — see §4). `(a,b,c) ≈
  (3.4445, −4.7750, 2.0315)` is tuned for *speed to a band*, NOT asymptotic convergence to exactly 1,
  and likely oscillates near 1. The honest theorem is *quantitative*: `∀ σ ∈ [σ_min, 1],
  |φ^[5](σ) − 1| ≤ δ` for an explicit `δ`. Needs interval/explicit-bound reasoning (`norm_num`/
  `polyrith`/`interval_cases`-flavored), matching the implementation's fixed-5-step nature. Optional;
  P1–P3 (the cubic) is the real deliverable.

## 2. Mathlib support (scouted 2026-06-29)

- ✅ monotone-bounded → converges: `Monotone.ciSup_comp_tendsto_atTop_of_linearOrder`
  (`Order/Filter/AtTopBot/CompleteLattice.lean`); identify the limit via
  `isFixedPt_of_tendsto_iterate` (`Dynamics/FixedPoints/Topology.lean`: `f^[n]x → y ⟹ IsFixedPt f y`)
  + continuity of `g`. Alternatively the contraction route (`tendsto_iterate_fixedPoint`,
  `Topology/MetricSpace/Contracting.lean`) if a local Lipschitz constant `<1` near 1 is cleaner.
- ✅ matrix topology + continuity: `Mathlib/Topology/Instances/Matrix.lean` —
  `Continuous.matrix_diagonal`, `Continuous.matrix_transpose`, matrix `*` continuous (finite-dim,
  product topology); `Continuous.matrix_elem` for entrywise. Convergence is entrywise = norm here.
- ✅ already in hand (reuse, don't rebuild): `conj_diag_pow`, `Matrix.diagonal_mul_diagonal`,
  `mul_eq_one_comm`, the `U·diag·Vᵀ` vocabulary from `MuonGeometry.lean` (L4/L5).
- 🔎 scout next session: `Function.iterate` API for sequences (`Function.iterate_succ'`), and whether
  to phrase the orbit as `fun k => g^[k] t₀` (iterate) or a recursive `ℕ → ℝ`. Iterate composes better
  with `isFixedPt_of_tendsto_iterate`.

## 3. File / wiring

`LeanMlir/Proofs/MuonNewtonSchulz.lean` (Mathlib + topology heavy; imports `MuonGeometry` to reuse
`conj_diag_pow` etc.). Add root to `lakefile.lean` `Proofs` lib (next to `MuonGeometry`), and
`#print axioms` lines to `tests/AuditAxioms.lean`. Reference: Higham *Functions of Matrices* (matrix
sign / polar iteration family), Jordan 2024 (Muon), `planning/muon.md`.

## 4. Honest scoping (the load-bearing caveat — Brett wants tiers, not overclaim)

There are two iterations, and they converge differently:
- **The classic cubic `g(t) = ½(3t − t³)`** (Higham's inverse-free polar / `½X(3I − XᵀX)`):
  **monotonically converges `(0,1] → 1`, provably.** This IS a genuine Newton–Schulz polar iteration.
  ⟹ **P1–P3 give a real, clean `X_k → UVᵀ` theorem.** This is the deliverable to aim for.
- **Jordan's tuned quintic `(3.4445, −4.7750, 2.0315)`**: tuned for *fast approach to a band near 1
  in ~5 steps*, deliberately NOT monotone-convergent to exactly 1 (it overshoots/oscillates — that's
  what buys the speed). Asymptotic `φ^[k](σ) → 1` is likely **false** for these coefficients. The
  honest statement is the finite-step band bound (P4), which is also what the implementation does
  (fixed 5 steps, "rough is fine; we recompute next step anyway", `planning/muon.md:91`).

⟹ **Don't state `Jordan-quintic^[k] → 1`** (would be overclaim). State the cubic limit (P1–P3) and,
separately, the quintic band bound (P4). The repo's `Types.lean:319` already calls Muon `UNVERIFIED`;
this work verifies the *cubic* NS → `UVᵀ` cleanly and the *quintic* only as a quantitative 5-step
bound. Mirror the precision-tier honesty of [[trusted-bridge-and-limit-d]] / [[repo-verification-reality]].

## 5. Why this closes the loop (the payoff narrative)

L3–L6 proved `UVᵀ` is the optimal object from four angles. P1–P3 prove the GPU-friendly matmul
iteration actually lands on it. Together: **the thing the hardware computes is the thing the theory
says is optimal.** The spectral reduction (§0) is the same idea that ran through the whole ladder —
"it's secretly diagonal" — so NS convergence isn't a new technique, it's the ladder's motif applied
to a fixed-point iteration. Nice closing symmetry for the book/write-up.

## 6. Session handoff — start at P1

1. New file `MuonNewtonSchulz.lean`, import `MuonGeometry`, open the same `Matrix` scope.
2. **P1 first** — `nsStep` def + the spectral-step lemma (reuse `conj_diag_pow`'s proof shape: the
   only new bit is `XXᵀ = U·diag(σ²)·Uᵀ` and collecting `aΣ+bΣ³+cΣ⁵` via `diagonal_mul_diagonal` +
   `smul`/`add` of diagonals). Scalar-free, fast to land — do it like L5's `hGtG`/`contract`.
3. **P2** — the cubic scalar lemma `∀ t₀∈(0,1], Tendsto (fun k => g^[k] t₀) atTop (𝓝 1)`. The three
   inequalities (`g↑`, `g≥id`, `g≤1`) are `nlinarith`/`polyrith` one-liners; the convergence is the
   monotone-bounded + fixed-point-uniqueness assembly (§2 lemmas).
4. **P3** — glue via matrix continuity. Then optionally P4 (quintic band).
5. Commit per-phase (the ladder's per-rung commit rhythm), update `planning/muon_geometry.md` §4 +
   [[muon-geometry-proof-ladder]] memory. First scout: `Function.iterate` vs recursive sequence, and
   the exact monotone-convergence lemma signature.
