# Planning — Muon geometry: the optimizer as steepest descent under a metric

The math-zone thread (Brett, 2026-06-28: less ImageNet polish, more proof layer). Goal: formalize the
**geometric motivation for Muon** — why the update is the polar factor `UVᵀ` — inside the unifying
frame "every optimizer is steepest descent under a choice of norm." Ties to the implemented Muon
(`OptimizerKind.muon`, `emitMuonUpdate`, the `den=` Newton–Schulz scaffold) and the orthogonality
through-line.

## 0. The one idea

The steepest-ascent direction of a linear functional `g` under a norm `‖·‖` is
`d⋆ = argmax_{‖d‖≤1} ⟨g,d⟩`, and the optimal value is the **dual norm** `‖g‖_*`. Each optimizer is
this with a different norm:

| optimizer | norm on the step | dual norm | `d⋆` | geometry |
|---|---|---|---|---|
| SGD | `‖·‖₂` (Euclidean) | `‖·‖₂` | `g/‖g‖` | round ball / Cauchy–Schwarz |
| sign-descent (Adam-ish) | `‖·‖∞` | `‖·‖₁` | `sign(g)` | box |
| **Muon** | **operator norm `‖·‖₂→₂`** (matrices) | **nuclear `Σσᵢ`** | **`UVᵀ`** (polar factor) | **spectral ball** |
| Shampoo (1-step) | Kronecker-factored | — | `(GGᵀ)^{-1/4}G(GᵀG)^{-1/4} = UVᵀ` | = Muon |

## 1. Layered plan (repo style: state given a hypothesis, then discharge it)

`LeanMlir/Proofs/MuonGeometry.lean`, all 3-axiom clean, audited in `tests/AuditAxioms.lean`.

- ✅ **L1 — SGD rung (Euclidean)** DONE: `steepest_l2_bound` (`⟨g,d⟩ ≤ ‖g‖` over `‖d‖≤1`, Cauchy–
  Schwarz via `real_inner_le_norm`) + `steepest_l2_attained` (`g/‖g‖` unit, `⟨g, g/‖g‖⟩ = ‖g‖`).
  Dual norm of `‖·‖₂` is itself; maximizer = normalized gradient.
- ✅ **L2 — sign rung (L∞→L¹)** DONE: `steepest_linf_bound` (`Σgᵢdᵢ ≤ Σ|gᵢ|` over `|dᵢ|≤1`) +
  `steepest_linf_attained` (`sign g` feasible, `Σ gᵢ·sign gᵢ = Σ|gᵢ|`). Sign/Adam geometry.
- 🟡 **L3 — Muon rung (operator→nuclear)** ACHIEVABILITY DONE: `muon_polar_achieves_nuclear` —
  given SVD `G = U·diagonal s·Vᵀ` (`UᵀU=VᵀV=1`), `fInner G (UVᵀ) = Σ sᵢ` (polar factor realizes the
  nuclear norm; trace algebra). **OPEN: the upper bound** (`⟨G,D⟩_F ≤ Σσᵢ` ∀ `‖D‖op≤1`, so `UVᵀ` is
  the MAX) = von Neumann's trace inequality. Diagonal core is elementary (`|Dᵢᵢ| ≤ ‖D‖op`, reduces
  to L2) but needs the matrix operator norm (scout `Matrix.l2OpNorm`).
- ✅ **L4 — build the SVD** DONE (invertible/full-rank `G`): `svd_of_isUnit` — for `IsUnit G`,
  produces orthogonal `U, V` and `s ≥ 0` with `G = U (diagonal s) Vᵀ`, built from the **matrix**
  spectral theorem of `A := GᵀG`. `V` = `hAherm.eigenvectorUnitary` (eigenbasis), `sᵢ = √λᵢ` the
  singular values (`λ` = eigenvalues of `GᵀG`, all `> 0` via `PosSemidef.posDef_iff_isUnit` +
  `posDef_iff_eigenvalues_pos`), `U = G V Σ⁻¹`. **Key shortcut: no matrix square root** — only the
  spectral decomposition, scalar `√`, and diagonal inverses (`Σ⁻¹` exists ⇐ `λᵢ > 0`); the checks
  `UᵀU = Σ⁻¹(VᵀAV)Σ⁻¹ = Σ⁻¹(diag λ)Σ⁻¹ = 1` and `UΣVᵀ = GVVᵀ = G` are diagonal arithmetic.
  Capstone `muon_polar_achieves_nuclear_of_isUnit` composes L4 with L3-achievability: the polar
  factor pairs with `G` to `Σσᵢ` **unconditionally for invertible `G`** — SVD hypothesis discharged.
  Both 3-axiom clean, audited. **Decision: chose `Matrix` over abstract `LinearMap`** (the L3 algebra
  and the matrix spectral theorem are both `Matrix`-native; `LinearMap.singularValues` gives only
  values, not the factorization). **Remaining: the singular `G` case** = the orthonormal completion
  of `U` (when some `λᵢ = 0`, `Σ⁻¹` is degenerate; needs extending `U`'s columns to an orthonormal
  basis) — a genuine but bounded extra build, deferred.
- ⬜ **L5 — the jewel**: single-step Shampoo `(GGᵀ)^{-1/4}G(GᵀG)^{-1/4} = UVᵀ = Muon`. Substitute
  the SVD (L4 now provides it for invertible `G`). Pure linear algebra once the SVD is in hand —
  but needs the `^{-1/4}` matrix power = `cfc` of `GᵀG`, or, since L4 already exposes `V, Σ`, a direct
  `(VΣ⁻¹ᐟ²Vᵀ)`-style construction reusing `svd_of_isUnit`'s diagonal machinery. **Next entry point.**
- ⬜ **L6 — manifold view** (the equivariance bridge): `UVᵀ ∈` Stiefel/`O(n)`; Newton–Schulz =
  retraction onto it. Mathlib `UnitaryGroup`. The Lie-group geometry of weight space.

## 2. Mathlib support (scouted 2026-06-28)

- ✅ `Mathlib/Analysis/InnerProductSpace/SingularValues.lean` (NEW, 2026) — `LinearMap.singularValues`
  via eigenvalues of `adjoint ∘ T`, ordering + support. NO singular vectors / SVD factorization yet.
- ✅ spectral theorem for self-adjoint (`IsSymmetric.eigenvectorBasis`), `UnitaryGroup`, Cauchy–Schwarz.
- ❌ no Schatten/nuclear norm, no polar decomposition, no explicit SVD factorization.
  ⇒ L0–L2 now; L3+ requires building L4 (SVD from the spectral theorem).
- ✅ **L4 build (used 2026-06-28)**: `Matrix.IsHermitian.spectral_theorem`
  (`A = conjStarAlgAut V (diagonal (ofReal∘eigenvalues))`, unfold via `Unitary.conjStarAlgAut_apply`;
  over ℝ, `star V = Vᵀ` via `star_eq_conjTranspose` + `conjTranspose_eq_transpose_of_trivial`, and
  `RCLike.ofReal∘λ = λ` by `simp`), `Matrix.IsHermitian.eigenvectorUnitary` (+ `.2` membership ⇒
  `mem_unitaryGroup_iff{,'}` ⇒ `Vᵀ V = V Vᵀ = 1`), `posSemidef_conjTranspose_mul_self`,
  `PosSemidef.posDef_iff_isUnit`, `IsHermitian.posDef_iff_eigenvalues_pos`, `isUnit_transpose`,
  `diagonal_mul_diagonal`/`diagonal_transpose`/`diagonal_one`. Assoc rearrangements close by
  `simp only [Matrix.mul_assoc]` (right-normalizes both sides). Gotcha: an `hλpos` identifier fails —
  bare `λ` lexes as lambda; name it `hlampos`.
- ❌ still no matrix operator norm `Matrix.l2OpNorm` found (scout: not under `Analysis/Matrix.lean`)
  ⇒ the L3 von Neumann *upper* bound still needs an operator-norm route.

## 3. File

`LeanMlir/Proofs/MuonGeometry.lean` (Mathlib-heavy, in the `Proofs` lib roots; audited). Reference:
Bernstein–Newhouse (modular norms / duality), Jordan (Muon), Gupta–Koren–Singer (Shampoo). See
[[math-threads-demo-first]], `planning/muon.md`, `planning/shampoo.md`.

## 4. Session log / next-session handoff

- **Session 1 (2026-06-28) DONE + committed**: the framework (`steepest descent under a norm`) with
  the SGD (L1) and sign (L2) rungs proven outright, and the Muon rung's achievability (L3) — the
  polar factor `UVᵀ` realizes the nuclear norm given an SVD. 5 theorems, 3-axiom clean, wired into
  the `Proofs` lib + `tests/AuditAxioms.lean`. This is the *geometric motivation* of Muon as checked
  Lean.
- **Session 2 (2026-06-28) DONE — L4 (build the SVD)**: chose `Matrix` (not abstract `LinearMap`).
  `svd_of_isUnit` + `muon_polar_achieves_nuclear_of_isUnit` (2 theorems, 3-axiom clean, audited).
  For invertible `G`, the SVD is now a *constructed object* — `V`=eigenbasis of `GᵀG`, `sᵢ=√λᵢ`,
  `U=GVΣ⁻¹`, no matrix square root — and the polar factor → nuclear-norm pairing is unconditional.
  SVD hypothesis of `muon_polar_achieves_nuclear` discharged for the full-rank case. Lemma map +
  gotchas recorded in §2 above.
- **Next session — two open forks (pick one):**
  - **(a) L3 upper bound (von Neumann's trace inequality)** — `⟨G,D⟩_F ≤ Σσᵢ` for `‖D‖op ≤ 1`, so
    `UVᵀ` is the *max* not just an achiever. BLOCKER scouted: no `Matrix.l2OpNorm` found in
    `Analysis/Matrix.lean`. First scout a matrix operator-norm: try `Matrix.instL2OpNormedAddCommGroup`
    / the `EuclideanSpace`-CLM route (`Matrix.toEuclideanCLM` / `toEuclideanLin` ⇒ `‖·‖`), or prove
    the inequality directly from the SVD substitution (`⟨UΣVᵀ,D⟩ = Σᵢ σᵢ ⟨uᵢ, D vᵢ⟩` then bound each
    `⟨uᵢ,Dvᵢ⟩ ≤ ‖D‖op ≤ 1` — reduces to L2/Cauchy–Schwarz per singular vector).
  - **(b) L5 the Shampoo=Muon jewel** — `(GGᵀ)^{-1/4}G(GᵀG)^{-1/4} = UVᵀ`. L4 exposes `V, Σ`; reuse
    its diagonal machinery to build `(GᵀG)^{-1/4} = V diagonal(λ^{-1/4}) Vᵀ` and substitute. Pure
    linear algebra, no new Mathlib surface — likely the faster win.
  - (deferred) the **singular-`G` SVD** = orthonormal completion of `U`.
