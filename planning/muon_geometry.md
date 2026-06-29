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
- ✅ **L3 — Muon rung (operator→nuclear)** DONE BOTH WAYS:
  - *achievability*: `muon_polar_achieves_nuclear` — given SVD `G = U·diagonal s·Vᵀ`, `fInner G (UVᵀ)
    = Σ sᵢ` (polar factor realizes the nuclear norm; trace algebra).
  - *upper bound (von Neumann's trace inequality)* DONE 2026-06-29: `muon_polar_is_max` —
    `fInner G D ≤ Σ sᵢ` for every contraction `D`. **Sidestepped the matrix operator norm entirely**:
    stated `‖D‖op≤1` elementarily as `(D*ᵥx)⬝ᵥ(D*ᵥx) ≤ x⬝ᵥx`, then the cyclic-trace reduction
    `fInner G D = Σᵢ sᵢ·(UᵀDV)ᵢᵢ` + per-singular-vector Cauchy–Schwarz (`Finset.sum_mul_sq_le_sq_mul_sq`,
    `(UᵀDV)ᵢᵢ = uᵢ·(Dvᵢ)`, `|·|≤1`). Capstone `muon_polar_steepest`: `UVᵀ` is feasible (isometry via
    `dotProduct_mulVec`+`mulVec_transpose`) ∧ attains ∧ unbeatable — the full argmax characterization,
    same `bound`+`attained` shape as L1/L2. 2 theorems, 3-axiom clean, audited.
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
- ✅ **L5 — the jewel** DONE: single-step Shampoo `(GGᵀ)^{-1/4}G(GᵀG)^{-1/4} = UVᵀ = Muon`.
  `shampoo_eq_muon` (parametrized by an SVD `G=UΣVᵀ`, `s>0`) proves the conjunction: `R⁴·(GᵀG)=1`
  and `L⁴·(GGᵀ)=1` (so `R=V·diag(s^{-1/2})·Vᵀ`, `L=U·diag(s^{-1/2})·Uᵀ` genuinely ARE the inverse
  fourth-roots) **and** the jewel `L·G·R = UVᵀ`. `shampoo_eq_muon_of_isUnit` makes it unconditional
  for invertible `G`. **No `cfc` / abstract matrix power needed** — avoided it entirely: defined the
  fourth-roots spectrally (`(√sᵢ)⁻¹ = sᵢ^{-1/2}` on the diagonal) and proved they earn the name via
  `X⁴·M = 1`. Workhorse `conj_diag_pow`: `(W·diag d·Wᵀ)^k = W·diag(dᵏ)·Wᵀ` (matrix power → pointwise
  scalar power, induction on k). Collapses: `(s^{-1/2})⁴·s² = 1` (roots) and `s^{-1/2}·s·s^{-1/2}=1`
  (jewel), both `field_simp; nlinarith [√sᵢ·√sᵢ=sᵢ]`. 3 theorems, 3-axiom clean, audited.
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
- **Session 3 (2026-06-29) DONE — L5 (the Shampoo=Muon jewel)**: `conj_diag_pow` + `shampoo_eq_muon`
  + `shampoo_eq_muon_of_isUnit` (3 theorems, 3-axiom clean, audited). Avoided `cfc` entirely by
  defining the inverse fourth-roots spectrally and proving the `X⁴·M=1` certs. Recipe + collapses in
  §1 L5 above. The Shampoo row of the §0 table is now checked Lean.
- **Session 4 (2026-06-29) DONE — L3 upper bound (von Neumann)**: `muon_polar_is_max` +
  `muon_polar_steepest` (2 theorems, 3-axiom clean, audited). The Muon rung is now complete BOTH
  ways — `UVᵀ` is the provable argmax, value = nuclear norm = the dual norm. Avoided the matrix
  operator norm by spelling `‖D‖op≤1` as an elementary contraction. Recipe in §1 L3 above.
- **Next session — remaining forks (the analytically-hard piece is now done):**
  - **(a) L6 manifold view** — `UVᵀ ∈ O(n)`/Stiefel, Newton–Schulz = retraction onto it.
    `Matrix.unitaryGroup` / `UnitaryGroup`. The Lie-group geometry of the update.
  - **(b) singular-`G` SVD** = orthonormal completion of `U` (when some `λᵢ=0`); would drop the
    invertibility hypothesis from L4/L5's `_of_isUnit` capstones (`svd` becomes total).
  - (optional polish) bridge the elementary contraction hypothesis to Mathlib's actual L2 operator
    norm (`Matrix.l2_opNorm_mulVec`, scoped `Matrix.Norms.L2Operator`) so the statements can *also*
    read `‖D‖ ≤ 1` literally — friction is the `EuclideanSpace.equiv` PiLp coercion.
