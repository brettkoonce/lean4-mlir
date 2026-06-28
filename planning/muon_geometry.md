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
- ⬜ **L4 — build the SVD** from Mathlib's spectral theorem of `GᵀG` + `LinearMap.singularValues`:
  eigenbasis of `GᵀG` = `V`, singular values = `Σ`, `U = GVΣ⁺`. **THE NEXT BIG PIECE** — discharges
  L3's SVD hypothesis, makes the polar factor / nuclear norm real objects not hyps. Mathlib has the
  singular *values* + spectral theorem but NOT the factorization, so it's a genuine Mathlib-grade
  build. **Entry point for the next session.**
- ⬜ **L5 — the jewel**: single-step Shampoo `(GGᵀ)^{-1/4}G(GᵀG)^{-1/4} = UVᵀ = Muon`. Substitute
  the SVD (needs L4). Pure linear algebra once the SVD is in hand.
- ⬜ **L6 — manifold view** (the equivariance bridge): `UVᵀ ∈` Stiefel/`O(n)`; Newton–Schulz =
  retraction onto it. Mathlib `UnitaryGroup`. The Lie-group geometry of weight space.

## 2. Mathlib support (scouted 2026-06-28)

- ✅ `Mathlib/Analysis/InnerProductSpace/SingularValues.lean` (NEW, 2026) — `LinearMap.singularValues`
  via eigenvalues of `adjoint ∘ T`, ordering + support. NO singular vectors / SVD factorization yet.
- ✅ spectral theorem for self-adjoint (`IsSymmetric.eigenvectorBasis`), `UnitaryGroup`, Cauchy–Schwarz.
- ❌ no Schatten/nuclear norm, no polar decomposition, no explicit SVD factorization.
  ⇒ L0–L2 now; L3+ requires building L4 (SVD from the spectral theorem).

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
- **Next session — start at L4 (build the SVD)**: from Mathlib's spectral theorem of `T.adjoint ∘ T`
  + the new `Mathlib/Analysis/InnerProductSpace/SingularValues.lean`. Goal: produce `U, Σ, V` with
  `G = UΣVᵀ`, `U,V` orthogonal, `Σ = diagonal(singularValues)`. That discharges the SVD hypothesis
  of `muon_polar_achieves_nuclear`, then unlocks (a) the L3 upper bound (von Neumann ⇒ `UVᵀ` is the
  *max*, not just achiever) and (b) L5, the Shampoo=Muon jewel `(GGᵀ)^{-1/4}G(GᵀG)^{-1/4} = UVᵀ`.
  First scout: whether to work with `Matrix` or abstract `LinearMap` (the new SingularValues file is
  `LinearMap`-based), and Mathlib's matrix operator norm (`Matrix.l2OpNorm`?) for the upper bound.
