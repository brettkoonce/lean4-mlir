import LeanMlir.Proofs.MuonGeometry
import Mathlib.Logic.Function.Iterate
import Mathlib.Topology.Order.MonotoneConvergence
import Mathlib.Dynamics.FixedPoints.Topology

/-! # Newton–Schulz convergence, P1: the iteration is a *scalar* map in disguise

The capstone of the Muon-geometry ladder (`planning/muon_ns_convergence.md`,
`planning/muon_geometry.md`, `LeanMlir/Proofs/MuonGeometry.lean`). L1–L6 proved that the polar factor
`UVᵀ` is the *right object* — operator-norm steepest descent (L3, von Neumann), the nuclear norm's
argmax, Shampoo's single step (L5), the nearest orthogonal matrix to `G` (L6). **What remains is that
the implementation actually computes it:** Muon's matmul iteration
`X ↦ aX + b(XXᵀ)X + c(XXᵀ)²X` (`OptimizerKind.muon`, `emitMuonUpdate`) converges to `UVᵀ`.

**This file is P1 — the spectral-step lemma, the bridge that turns the whole problem scalar.** The one
idea: a Newton–Schulz step never touches the singular *directions*, only the singular *values*. With
`X = U Σ Vᵀ` (`U,V` orthonormal, `Σ = diagonal σ`), since `XXᵀ = U Σ² Uᵀ`,
```
(XXᵀ)X = U Σ³ Vᵀ,   (XXᵀ)²X = U Σ⁵ Vᵀ   ⟹   nsStep a b c X = U (diagonal (φ ∘ σ)) Vᵀ,
```
where `φ(t) = a t + b t³ + c t⁵` (`nsScalar`) is applied per singular value, with `U,V` carried along
unchanged (`nsStep_spectral`). Iterating, `nsStep^[k] X = U (diagonal (φ^[k] ∘ σ)) Vᵀ`
(`nsStep_iterate_spectral`): **matrix convergence to `UVᵀ` reduces to scalar convergence `φ^[k](σᵢ) → 1`
per singular value.** This is the same `U Σ Vᵀ ↦ U f(Σ) Vᵀ` motif as L5's `conj_diag_pow`, now for the
polynomial `φ`. The downstream scalar analysis (P2) and the matrix-continuity assembly (P3) build on
these two lemmas. All `propext / Classical.choice / Quot.sound`-clean. -/

namespace Proofs.MuonNewtonSchulz

open scoped Matrix

variable {n : ℕ}

/-- **The Newton–Schulz step.** One iteration of Muon's gradient orthogonalizer:
    `nsStep a b c X = aX + b(XXᵀ)X + c(XXᵀ)²X`, the odd matrix polynomial `X·p(XᵀX)` written via the
    Gram matrix `XXᵀ`. The classic inverse-free polar iteration is `(a,b,c) = (3/2, −1/2, 0)`; Muon's
    tuned quintic is `(3.4445, −4.7750, 2.0315)`. The quintic monomial is associated as
    `(XXᵀ)·((XXᵀ)X)` so the spectral collapse threads cleanly. -/
def nsStep (a b c : ℝ) (X : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  a • X + b • (X * Xᵀ * X) + c • (X * Xᵀ * (X * Xᵀ * X))

/-- **The scalar map a Newton–Schulz step induces on each singular value:** `φ(t) = a t + b t³ + c t⁵`.
    `nsStep_spectral` shows `nsStep a b c` acts as `nsScalar a b c` on the singular values of `X`, so
    the matrix iteration's convergence is exactly this scalar iteration's (`φ^[k](σᵢ) → 1`). -/
def nsScalar (a b c t : ℝ) : ℝ := a * t + b * t ^ 3 + c * t ^ 5

/-- **P1 — the spectral-step lemma: a Newton–Schulz step is `nsScalar` applied per singular value.**
    For `X = U (diagonal σ) Vᵀ` with `U,V` orthonormal (`UᵀU = VᵀV = 1`),
    `nsStep a b c X = U (diagonal (fun i ↦ nsScalar a b c (σ i))) Vᵀ`. The singular vectors `U,V` are
    carried through untouched; only the singular values move, by the scalar polynomial `φ`. This is
    the **only matrix-level work** in the convergence proof — everything downstream is scalar.

    The proof is pure `UᵀU = 1` / `VᵀV = 1` collapse algebra (the `conj_diag_pow` motif of L5): the
    Gram matrix `XXᵀ = U (diagonal σ²) Uᵀ`, and each higher monomial `(XXᵀ)ᵏX` collapses to
    `U (diagonal σ^{2k+1}) Vᵀ` because the inner `UᵀU` contracts to the identity; the three scalar
    coefficients `a, b, c` ride through `•` onto the diagonal and sum pointwise to `φ`. -/
theorem nsStep_spectral (a b c : ℝ) (U V : Matrix (Fin n) (Fin n) ℝ) (σ : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) :
    nsStep a b c (U * Matrix.diagonal σ * Vᵀ)
      = U * Matrix.diagonal (fun i => nsScalar a b c (σ i)) * Vᵀ := by
  -- a scalar `r •` slides through the conjugation triple `U (diag d) Vᵀ` onto the diagonal
  have hdiag_smul : ∀ (r : ℝ) (d : Fin n → ℝ),
      r • Matrix.diagonal d = Matrix.diagonal (fun i => r * d i) := by
    intro r d; ext i j
    by_cases h : i = j <;> simp [Matrix.smul_apply, h, smul_eq_mul]
  have hsmul : ∀ (r : ℝ) (d : Fin n → ℝ),
      r • (U * Matrix.diagonal d * Vᵀ) = U * Matrix.diagonal (fun i => r * d i) * Vᵀ := by
    intro r d; rw [← Matrix.smul_mul, ← Matrix.mul_smul, hdiag_smul]
  -- `(U diag p Uᵀ)(U diag q Vᵀ) = U diag (p·q) Vᵀ` — the `UᵀU = 1` contraction
  have hcollapse : ∀ (p q : Fin n → ℝ),
      (U * Matrix.diagonal p * Uᵀ) * (U * Matrix.diagonal q * Vᵀ)
        = U * Matrix.diagonal (fun i => p i * q i) * Vᵀ := by
    intro p q
    rw [show (U * Matrix.diagonal p * Uᵀ) * (U * Matrix.diagonal q * Vᵀ)
          = U * (Matrix.diagonal p * (Uᵀ * U) * Matrix.diagonal q) * Vᵀ from by
            simp only [Matrix.mul_assoc],
        hU, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
  -- the singular values only: `Xᵀ`, the Gram `XXᵀ`, and the cubic/quintic monomials stay `U diag(·) Vᵀ`
  have hXt : (U * Matrix.diagonal σ * Vᵀ)ᵀ = V * Matrix.diagonal σ * Uᵀ := by
    simp only [Matrix.transpose_mul, Matrix.diagonal_transpose, Matrix.transpose_transpose,
      Matrix.mul_assoc]
  have hXXt : (U * Matrix.diagonal σ * Vᵀ) * (U * Matrix.diagonal σ * Vᵀ)ᵀ
        = U * Matrix.diagonal (fun i => σ i * σ i) * Uᵀ := by
    rw [hXt, show (U * Matrix.diagonal σ * Vᵀ) * (V * Matrix.diagonal σ * Uᵀ)
          = U * (Matrix.diagonal σ * (Vᵀ * V) * Matrix.diagonal σ) * Uᵀ from by
            simp only [Matrix.mul_assoc], hV, Matrix.mul_one, Matrix.diagonal_mul_diagonal]
  have hcube : (U * Matrix.diagonal σ * Vᵀ) * (U * Matrix.diagonal σ * Vᵀ)ᵀ
                * (U * Matrix.diagonal σ * Vᵀ)
        = U * Matrix.diagonal (fun i => σ i * σ i * σ i) * Vᵀ := by
    rw [hXXt, hcollapse]
  have hquint : (U * Matrix.diagonal σ * Vᵀ) * (U * Matrix.diagonal σ * Vᵀ)ᵀ
                * ((U * Matrix.diagonal σ * Vᵀ) * (U * Matrix.diagonal σ * Vᵀ)ᵀ
                   * (U * Matrix.diagonal σ * Vᵀ))
        = U * Matrix.diagonal (fun i => σ i * σ i * (σ i * σ i * σ i)) * Vᵀ := by
    rw [hcube, hXXt, hcollapse]
  -- collect the three monomials onto a single diagonal, then identify it with `φ ∘ σ`
  have hsum3 : ∀ d1 d2 d3 : Fin n → ℝ,
      U * Matrix.diagonal d1 * Vᵀ + U * Matrix.diagonal d2 * Vᵀ + U * Matrix.diagonal d3 * Vᵀ
        = U * Matrix.diagonal (fun i => d1 i + d2 i + d3 i) * Vᵀ := by
    intro d1 d2 d3
    have hd : Matrix.diagonal (fun i => d1 i + d2 i + d3 i)
        = Matrix.diagonal d1 + Matrix.diagonal d2 + Matrix.diagonal d3 := by
      ext i j; by_cases h : i = j <;> simp [Matrix.add_apply, h]
    rw [hd, Matrix.mul_add, Matrix.mul_add, Matrix.add_mul, Matrix.add_mul]
  simp only [nsStep]
  rw [hquint, hcube, hsmul, hsmul, hsmul, hsum3,
      show (fun i => a * σ i + b * (σ i * σ i * σ i) + c * (σ i * σ i * (σ i * σ i * σ i)))
        = (fun i => nsScalar a b c (σ i)) from funext fun i => by simp only [nsScalar]; ring]

/-- **P1, iterated: `k` Newton–Schulz steps act as `nsScalar^[k]` per singular value.**
    `(nsStep a b c)^[k] (U (diagonal σ) Vᵀ) = U (diagonal (fun i ↦ (nsScalar a b c)^[k] (σ i))) Vᵀ`.
    A one-line induction reusing `nsStep_spectral` at each step: the singular vectors `U,V` are
    invariant under the whole orbit, so **convergence of the matrix iteration `nsStep^[k] X → UVᵀ`
    reduces to the scalar fixed-point convergence `(nsScalar a b c)^[k] (σ i) → 1`** for each singular
    value — the entry point for P2 (the cubic monotone argument) and P3 (the matrix-continuity glue). -/
theorem nsStep_iterate_spectral (a b c : ℝ) (U V : Matrix (Fin n) (Fin n) ℝ) (σ : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (k : ℕ) :
    (nsStep a b c)^[k] (U * Matrix.diagonal σ * Vᵀ)
      = U * Matrix.diagonal (fun i => (nsScalar a b c)^[k] (σ i)) * Vᵀ := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih, nsStep_spectral a b c U V _ hU hV]
    have : (fun i => nsScalar a b c ((nsScalar a b c)^[k] (σ i)))
        = (fun i => (nsScalar a b c)^[k + 1] (σ i)) :=
      funext fun i => (Function.iterate_succ_apply' (nsScalar a b c) k (σ i)).symm
    rw [this]

-- ════════════════════════════════════════════════════════════════
-- § P2 — the cubic scalar iteration `g(t) = ½(3t − t³)` converges to 1 on `(0,1]`
-- ════════════════════════════════════════════════════════════════

/-- **The classic inverse-free polar iteration**, `g(t) = ½(3t − t³)` — the Newton–Schulz scalar map
    `nsScalar (3/2) (−1/2) 0` (`gCubic_eq_nsScalar`). Unlike Muon's tuned quintic (which is band-landing,
    not asymptotically convergent), this cubic is a genuine fixed-point iteration: it monotonically
    drives every `t ∈ (0,1]` to `1` (`gCubic_iterate_tendsto_one`), so the matrix iteration built from
    it lands exactly on the polar factor `UVᵀ`. -/
noncomputable def gCubic (t : ℝ) : ℝ := (3 * t - t ^ 3) / 2

/-- The cubic iteration is the `(a,b,c) = (3/2, −1/2, 0)` instance of the Newton–Schulz scalar map, so
    `nsStep_iterate_spectral` carries its scalar convergence (`gCubic_iterate_tendsto_one`) up to the
    matrix level in P3. -/
theorem gCubic_eq_nsScalar : gCubic = nsScalar (3 / 2) (-1 / 2) 0 := by
  funext t; simp only [gCubic, nsScalar]; ring

/-- **P2 — the cubic Newton–Schulz scalar iteration converges to `1` on `(0,1]`.** For every
    `t₀ ∈ (0,1]`, `g^[k](t₀) → 1` as `k → ∞`, where `g = gCubic`. This is the scalar engine of the
    convergence proof: with `nsStep_iterate_spectral`, "each singular value `σᵢ ∈ (0,1]` flows to `1`"
    is exactly "the matrix iterate flows to `UVᵀ`" (P3).

    The argument is the textbook monotone one. On `[0,1]` the orbit is trapped in `[t₀,1]` (`hinv`):
    `g(t) ≥ t` (`g(t) − t = t(1−t)(1+t)/2 ≥ 0`) pushes it up, `g(t) ≤ 1` (`1 − g(t) = (1−t)²(2+t)/2 ≥ 0`)
    keeps it under `1`. So `k ↦ g^[k](t₀)` is monotone and bounded above, hence converges to its
    supremum `L` (`tendsto_atTop_ciSup`), which lies in `(0,1]`. Continuity makes `L` a fixed point
    (`isFixedPt_of_tendsto_iterate`), and the fixed points of `g` are `g(t) = t ⟺ t(1−t²) = 0 ⟺
    t ∈ {0,±1}`; the only one in `(0,1]` is `1`, so `L = 1`. -/
theorem gCubic_iterate_tendsto_one {t₀ : ℝ} (h0 : 0 < t₀) (h1 : t₀ ≤ 1) :
    Filter.Tendsto (fun k => gCubic^[k] t₀) Filter.atTop (nhds 1) := by
  -- the two monotone-iteration inequalities on `[0,1]`: `t ≤ g t ≤ 1`
  have hg_ge : ∀ t : ℝ, 0 ≤ t → t ≤ 1 → t ≤ gCubic t := by
    intro t ht0 ht1; simp only [gCubic]
    nlinarith [mul_nonneg (mul_nonneg ht0 (by linarith : (0:ℝ) ≤ 1 - t))
      (by linarith : (0:ℝ) ≤ 1 + t)]
  have hg_le : ∀ t : ℝ, 0 ≤ t → t ≤ 1 → gCubic t ≤ 1 := by
    intro t ht0 ht1; simp only [gCubic]
    nlinarith [mul_nonneg (sq_nonneg (1 - t)) (by linarith : (0:ℝ) ≤ 2 + t)]
  -- the orbit stays in `[t₀, 1]`
  have hinv : ∀ k, t₀ ≤ gCubic^[k] t₀ ∧ gCubic^[k] t₀ ≤ 1 := by
    intro k
    induction k with
    | zero => exact ⟨le_refl t₀, h1⟩
    | succ k ih =>
      obtain ⟨hlo, hhi⟩ := ih
      have h0k : 0 ≤ gCubic^[k] t₀ := h0.le.trans hlo
      rw [Function.iterate_succ_apply']
      exact ⟨hlo.trans (hg_ge _ h0k hhi), hg_le _ h0k hhi⟩
  -- monotone ↑ and bounded above by 1
  have hmono : Monotone (fun k => gCubic^[k] t₀) := by
    apply monotone_nat_of_le_succ
    intro k
    show gCubic^[k] t₀ ≤ gCubic^[k + 1] t₀
    rw [Function.iterate_succ_apply']
    exact hg_ge _ (h0.le.trans (hinv k).1) (hinv k).2
  have hbdd : BddAbove (Set.range (fun k => gCubic^[k] t₀)) := by
    refine ⟨1, ?_⟩; rintro x ⟨k, rfl⟩; exact (hinv k).2
  -- converges to its supremum `L`
  have htends : Filter.Tendsto (fun k => gCubic^[k] t₀) Filter.atTop
      (nhds (⨆ i, gCubic^[i] t₀)) := tendsto_atTop_ciSup hmono hbdd
  set L := ⨆ i, gCubic^[i] t₀ with hL
  have hL_le : L ≤ 1 := by rw [hL]; exact ciSup_le (fun k => (hinv k).2)
  have hL_ge : t₀ ≤ L := by rw [hL]; exact le_trans (hinv 0).1 (le_ciSup hbdd 0)
  have hL0 : 0 < L := lt_of_lt_of_le h0 hL_ge
  -- `L` is a fixed point of the continuous `g`, and `1` is the only fixed point in `(0,1]`
  have hcont : Continuous gCubic := by unfold gCubic; fun_prop
  have hfix : gCubic L = L := isFixedPt_of_tendsto_iterate htends hcont.continuousAt
  have hLcube : L - L ^ 3 = 0 := by
    have h := hfix; simp only [gCubic] at h; linarith
  have hfactor : (1 - L) * (L * (1 + L)) = 0 := by
    have hexp : (1 - L) * (L * (1 + L)) = L - L ^ 3 := by ring
    rw [hexp, hLcube]
  have h1pos : 0 < L * (1 + L) := mul_pos hL0 (by linarith)
  have hL_eq : L = 1 := by
    rcases mul_eq_zero.mp hfactor with h | h
    · linarith
    · exact absurd h h1pos.ne'
  exact hL_eq ▸ htends

-- ════════════════════════════════════════════════════════════════
-- § P3 — assemble: the cubic Newton–Schulz matrix iterate converges to the polar factor `UVᵀ`
-- ════════════════════════════════════════════════════════════════

/-- **P3 — the cubic Newton–Schulz iteration converges to Muon's polar factor `UVᵀ`.** For a
    pre-normalized full-rank gradient `G = U (diagonal σ) Vᵀ` with every singular value `σᵢ ∈ (0,1]`
    (the implementation's `G / ‖G‖` step), the matmul iterate `(nsStep (3/2) (−1/2) 0)^[k] G` converges
    to the polar factor `U Vᵀ` — exactly the object L3–L6 proved optimal (operator-norm steepest
    descent / nuclear-norm argmax / Shampoo's step / nearest orthogonal matrix). **This closes the
    loop: the thing the hardware computes is the thing the theory says is optimal.**

    The proof is the spectral reduction (§0) cashed out: `nsStep_iterate_spectral` (P1) makes the
    matrix iterate `U (diagonal (gCubic^[k] ∘ σ)) Vᵀ` (with `gCubic = nsScalar (3/2) (−1/2) 0`,
    `gCubic_eq_nsScalar`); each diagonal entry `gCubic^[k] (σᵢ) → 1` by `gCubic_iterate_tendsto_one`
    (P2); pointwise convergence in `Fin n → ℝ` (`tendsto_pi_nhds`) plus continuity of
    `d ↦ U (diagonal d) Vᵀ` (`Continuous.matrix_diagonal`/`Continuous.matrix_mul`) pushes the limit
    through to `U (diagonal 1) Vᵀ = U Vᵀ` (`Matrix.diagonal_one`). The rank hypothesis `σᵢ > 0` is what
    makes the per-value limit `1` (P2 needs `t₀ > 0`); `σᵢ = 0` would stay `0`, giving the partial
    isometry `U (diagonal 1_{σ>0}) Vᵀ` instead. -/
theorem nsStep_cubic_iterate_tendsto_polar
    (U V : Matrix (Fin n) (Fin n) ℝ) (σ : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (hσ : ∀ i, 0 < σ i ∧ σ i ≤ 1) :
    Filter.Tendsto (fun k => (nsStep (3 / 2) (-1 / 2) 0)^[k] (U * Matrix.diagonal σ * Vᵀ))
      Filter.atTop (nhds (U * Vᵀ)) := by
  -- P1: the matrix iterate is `U (diagonal (gCubic^[k] ∘ σ)) Vᵀ`
  have hseq : (fun k => (nsStep (3 / 2) (-1 / 2) 0)^[k] (U * Matrix.diagonal σ * Vᵀ))
      = (fun k => U * Matrix.diagonal (fun i => gCubic^[k] (σ i)) * Vᵀ) := by
    funext k
    rw [nsStep_iterate_spectral (3 / 2) (-1 / 2) 0 U V σ hU hV k, ← gCubic_eq_nsScalar]
  -- P2: each singular value flows to 1 ⟹ the diagonal vectors converge to `1` in `Fin n → ℝ`
  have hpt : ∀ i, Filter.Tendsto (fun k => gCubic^[k] (σ i)) Filter.atTop (nhds 1) :=
    fun i => gCubic_iterate_tendsto_one (hσ i).1 (hσ i).2
  have hpi : Filter.Tendsto (fun k => fun i => gCubic^[k] (σ i)) Filter.atTop
      (nhds (fun _ : Fin n => (1 : ℝ))) := tendsto_pi_nhds.mpr hpt
  -- continuity of `d ↦ U (diagonal d) Vᵀ` carries the limit through to `U (diagonal 1) Vᵀ = U Vᵀ`
  have hF : Continuous (fun d : Fin n → ℝ => U * Matrix.diagonal d * Vᵀ) :=
    (continuous_const.matrix_mul continuous_id.matrix_diagonal).matrix_mul continuous_const
  have hcomp := (hF.tendsto _).comp hpi
  rw [hseq, show U * Vᵀ = U * Matrix.diagonal (fun _ : Fin n => (1 : ℝ)) * Vᵀ from by
    rw [Matrix.diagonal_one, Matrix.mul_one]]
  exact hcomp

end Proofs.MuonNewtonSchulz
