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
-- § P2 — scalar convergence: a monotone Newton–Schulz scalar map drives `(0,1] → 1`
-- ════════════════════════════════════════════════════════════════

/-- **The monotone scalar-convergence engine — convergence is a property of the *coefficients*, not
    the degree.** Any scalar map `g` that on `[0,1]` is a self-map pushing up toward `1` (`t ≤ g t` and
    `g t ≤ 1`) with `1` its only fixed point in `(0,1]` (`hfix`) drives every `t₀ ∈ (0,1]` to `1`:
    `g^[k](t₀) → 1`. The textbook monotone argument: the orbit is trapped in `[t₀,1]` (`hinv`), monotone
    `↑` and bounded above, hence converges to its supremum `L ∈ (0,1]` (`tendsto_atTop_ciSup`);
    continuity makes `L` a fixed point (`isFixedPt_of_tendsto_iterate`), and `hfix` pins `L = 1`.

    Both the cubic (`gCubic_iterate_tendsto_one`) and the *principled* convergent quintic
    (`q5Scalar_iterate_tendsto_one`) instantiate this. Muon's *tuned* quintic (P4) fails `g t ≤ 1` —
    that single broken hypothesis is exactly why it bands instead of converging. -/
theorem scalar_iterate_tendsto_one (g : ℝ → ℝ) (hcont : Continuous g)
    (hge : ∀ t : ℝ, 0 ≤ t → t ≤ 1 → t ≤ g t) (hle : ∀ t : ℝ, 0 ≤ t → t ≤ 1 → g t ≤ 1)
    (hfix : ∀ L : ℝ, 0 < L → L ≤ 1 → g L = L → L = 1)
    {t₀ : ℝ} (h0 : 0 < t₀) (h1 : t₀ ≤ 1) :
    Filter.Tendsto (fun k => g^[k] t₀) Filter.atTop (nhds 1) := by
  -- the orbit stays in `[t₀, 1]`
  have hinv : ∀ k, t₀ ≤ g^[k] t₀ ∧ g^[k] t₀ ≤ 1 := by
    intro k
    induction k with
    | zero => exact ⟨le_refl t₀, h1⟩
    | succ k ih =>
      obtain ⟨hlo, hhi⟩ := ih
      have h0k : 0 ≤ g^[k] t₀ := h0.le.trans hlo
      rw [Function.iterate_succ_apply']
      exact ⟨hlo.trans (hge _ h0k hhi), hle _ h0k hhi⟩
  -- monotone ↑ and bounded above by 1 ⟹ converges to its supremum `L`
  have hmono : Monotone (fun k => g^[k] t₀) := by
    apply monotone_nat_of_le_succ
    intro k
    show g^[k] t₀ ≤ g^[k + 1] t₀
    rw [Function.iterate_succ_apply']
    exact hge _ (h0.le.trans (hinv k).1) (hinv k).2
  have hbdd : BddAbove (Set.range (fun k => g^[k] t₀)) := by
    refine ⟨1, ?_⟩; rintro x ⟨k, rfl⟩; exact (hinv k).2
  have htends : Filter.Tendsto (fun k => g^[k] t₀) Filter.atTop (nhds (⨆ i, g^[i] t₀)) :=
    tendsto_atTop_ciSup hmono hbdd
  set L := ⨆ i, g^[i] t₀ with hL
  have hL_le : L ≤ 1 := by rw [hL]; exact ciSup_le (fun k => (hinv k).2)
  have hL_ge : t₀ ≤ L := by rw [hL]; exact le_trans (hinv 0).1 (le_ciSup hbdd 0)
  -- the limit is a fixed point in `(0,1]`, so `hfix` forces it to be `1`
  have hfixL : g L = L := isFixedPt_of_tendsto_iterate htends hcont.continuousAt
  exact hfix L (lt_of_lt_of_le h0 hL_ge) hL_le hfixL ▸ htends

/-- **The classic inverse-free cubic** `g(t) = ½(3t − t³)` — the Newton–Schulz scalar map
    `nsScalar (3/2) (−1/2) 0` (`gCubic_eq_nsScalar`). -/
noncomputable def gCubic (t : ℝ) : ℝ := (3 * t - t ^ 3) / 2

theorem gCubic_eq_nsScalar : gCubic = nsScalar (3 / 2) (-1 / 2) 0 := by
  funext t; simp only [gCubic, nsScalar]; ring

/-- **P2 (cubic) — `gCubic^[k](t₀) → 1` on `(0,1]`** (second-order convergent). Instantiates the
    `scalar_iterate_tendsto_one` engine: increasing toward `1` on `[0,1]`
    (`g t − t = t(1−t)(1+t)/2 ≥ 0`, `1 − g t = (1−t)²(2+t)/2 ≥ 0`) with fixed points
    `g t = t ⟺ t(1−t²)=0 ⟺ t ∈ {0,±1}` — only `1` lies in `(0,1]`. -/
theorem gCubic_iterate_tendsto_one {t₀ : ℝ} (h0 : 0 < t₀) (h1 : t₀ ≤ 1) :
    Filter.Tendsto (fun k => gCubic^[k] t₀) Filter.atTop (nhds 1) := by
  refine scalar_iterate_tendsto_one gCubic (by unfold gCubic; fun_prop) ?_ ?_ ?_ h0 h1
  · intro t ht0 ht1; simp only [gCubic]
    nlinarith [mul_nonneg (mul_nonneg ht0 (by linarith : (0:ℝ) ≤ 1 - t))
      (by linarith : (0:ℝ) ≤ 1 + t)]
  · intro t ht0 ht1; simp only [gCubic]
    nlinarith [mul_nonneg (sq_nonneg (1 - t)) (by linarith : (0:ℝ) ≤ 2 + t)]
  · intro L hL0 hLle hfixL
    simp only [gCubic] at hfixL
    have hLcube : L - L ^ 3 = 0 := by linarith
    have hfactor : (1 - L) * (L * (1 + L)) = 0 := by
      have hexp : (1 - L) * (L * (1 + L)) = L - L ^ 3 := by ring
      rw [hexp, hLcube]
    rcases mul_eq_zero.mp hfactor with h | h
    · linarith
    · exact absurd h (mul_pos hL0 (by linarith)).ne'

/-- **The principled convergent quintic** `q₅(t) = (15t − 10t³ + 3t⁵)/8` — Higham's order-5
    Newton–Schulz iteration for the matrix sign / polar function, `nsScalar (15/8) (−5/4) (3/8)`. This
    is the answer to "does a *quintic* converge?": **yes — if you pick these coefficients.** Unlike
    Muon's tuned quintic (P4), here `q₅(1) = 1` is a fixed point and the iteration converges — in fact
    *faster* than the cubic (third-order: `1 − q₅(t) = (1−t)³(3t²+9t+8)/8`). Convergence is a property
    of the chosen polynomial, not of its degree. -/
noncomputable def q5Scalar (t : ℝ) : ℝ := nsScalar (15 / 8) (-5 / 4) (3 / 8) t

theorem q5Scalar_eq_nsScalar : q5Scalar = nsScalar (15 / 8) (-5 / 4) (3 / 8) := rfl

/-- **P2 (principled quintic) — `q5Scalar^[k](t₀) → 1` on `(0,1]`** (third-order convergent). The same
    `scalar_iterate_tendsto_one` engine as the cubic: `q₅` is increasing (`q₅′(t) = 15(1−t²)²/8 ≥ 0`),
    `q₅ t − t = t(7−3t²)(1−t²)/8 ≥ 0` and `1 − q₅ t = (1−t)³(3t²+9t+8)/8 ≥ 0` on `[0,1]`, and its only
    fixed point in `(0,1]` is `1` (`q₅ L = L ⟺ L(7−3L²)(1−L)(1+L)=0`, and `7−3L² > 0` there). -/
theorem q5Scalar_iterate_tendsto_one {t₀ : ℝ} (h0 : 0 < t₀) (h1 : t₀ ≤ 1) :
    Filter.Tendsto (fun k => q5Scalar^[k] t₀) Filter.atTop (nhds 1) := by
  refine scalar_iterate_tendsto_one q5Scalar (by unfold q5Scalar nsScalar; fun_prop) ?_ ?_ ?_ h0 h1
  · intro t ht0 ht1; simp only [q5Scalar, nsScalar]
    nlinarith [mul_nonneg (mul_nonneg ht0 (show (0:ℝ) ≤ 7 - 3 * t ^ 2 by nlinarith [ht0, ht1]))
      (show (0:ℝ) ≤ 1 - t ^ 2 by nlinarith [ht0, ht1])]
  · intro t ht0 ht1; simp only [q5Scalar, nsScalar]
    nlinarith [mul_nonneg (mul_nonneg (mul_nonneg (sub_nonneg.2 ht1) (sub_nonneg.2 ht1))
      (sub_nonneg.2 ht1)) (show (0:ℝ) ≤ 3 * t ^ 2 + 9 * t + 8 by nlinarith [ht0, sq_nonneg t])]
  · intro L hL0 hLle hfixL
    simp only [q5Scalar, nsScalar] at hfixL
    have hfactor : (1 - L) * (L * (7 - 3 * L ^ 2) * (1 + L)) = 0 := by
      have hexp : (1 - L) * (L * (7 - 3 * L ^ 2) * (1 + L)) = 3 * L ^ 5 - 10 * L ^ 3 + 7 * L := by
        ring
      rw [hexp]; linarith
    have hpos : 0 < L * (7 - 3 * L ^ 2) * (1 + L) :=
      mul_pos (mul_pos hL0 (by nlinarith [hL0, hLle])) (by linarith)
    rcases mul_eq_zero.mp hfactor with h | h
    · linarith
    · exact absurd h hpos.ne'

-- ════════════════════════════════════════════════════════════════
-- § P3 — assemble: a *convergent* Newton–Schulz matrix iterate lands on the polar factor `UVᵀ`
-- ════════════════════════════════════════════════════════════════

/-- **P3 — the matrix glue: any convergent scalar Newton–Schulz map lifts to `nsStep^[k] G → UVᵀ`.**
    For a pre-normalized full-rank `G = U (diagonal σ) Vᵀ` with every singular value `σᵢ ∈ (0,1]` (the
    implementation's `G / ‖G‖` step), if the scalar map `g = nsScalar a b c` drives `(0,1] → 1`
    (`hconv`), then the matmul iterate `(nsStep a b c)^[k] G` converges to the polar factor `U Vᵀ` —
    exactly the object L3–L6 proved optimal (operator-norm steepest descent / nuclear-norm argmax /
    Shampoo's step / nearest orthogonal matrix). **This closes the loop: the thing the hardware
    computes is the thing the theory says is optimal.**

    The §0 spectral reduction cashed out: `nsStep_iterate_spectral` (P1) makes the matrix iterate
    `U (diagonal (g^[k] ∘ σ)) Vᵀ`; each diagonal entry `g^[k](σᵢ) → 1` by `hconv`; pointwise
    convergence in `Fin n → ℝ` (`tendsto_pi_nhds`) plus continuity of `d ↦ U (diagonal d) Vᵀ`
    (`Continuous.matrix_diagonal`/`Continuous.matrix_mul`) pushes the limit through to
    `U (diagonal 1) Vᵀ = U Vᵀ` (`Matrix.diagonal_one`). The rank hypothesis `σᵢ > 0` is what makes the
    per-value limit `1`; `σᵢ = 0` would stay `0`, giving the partial isometry `U (diagonal 1_{σ>0}) Vᵀ`. -/
theorem nsStep_iterate_tendsto_polar (a b c : ℝ) (g : ℝ → ℝ) (hbridge : g = nsScalar a b c)
    (hconv : ∀ t₀ : ℝ, 0 < t₀ → t₀ ≤ 1 → Filter.Tendsto (fun k => g^[k] t₀) Filter.atTop (nhds 1))
    (U V : Matrix (Fin n) (Fin n) ℝ) (σ : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (hσ : ∀ i, 0 < σ i ∧ σ i ≤ 1) :
    Filter.Tendsto (fun k => (nsStep a b c)^[k] (U * Matrix.diagonal σ * Vᵀ))
      Filter.atTop (nhds (U * Vᵀ)) := by
  -- P1: the matrix iterate is `U (diagonal (g^[k] ∘ σ)) Vᵀ`
  have hseq : (fun k => (nsStep a b c)^[k] (U * Matrix.diagonal σ * Vᵀ))
      = (fun k => U * Matrix.diagonal (fun i => g^[k] (σ i)) * Vᵀ) := by
    funext k
    rw [nsStep_iterate_spectral a b c U V σ hU hV k, ← hbridge]
  -- `hconv` per singular value ⟹ the diagonal vectors converge to `1` in `Fin n → ℝ`
  have hpi : Filter.Tendsto (fun k => fun i => g^[k] (σ i)) Filter.atTop
      (nhds (fun _ : Fin n => (1 : ℝ))) :=
    tendsto_pi_nhds.mpr (fun i => hconv (σ i) (hσ i).1 (hσ i).2)
  -- continuity of `d ↦ U (diagonal d) Vᵀ` carries the limit through to `U (diagonal 1) Vᵀ = U Vᵀ`
  have hF : Continuous (fun d : Fin n → ℝ => U * Matrix.diagonal d * Vᵀ) :=
    (continuous_const.matrix_mul continuous_id.matrix_diagonal).matrix_mul continuous_const
  have hcomp := (hF.tendsto _).comp hpi
  rw [hseq, show U * Vᵀ = U * Matrix.diagonal (fun _ : Fin n => (1 : ℝ)) * Vᵀ from by
    rw [Matrix.diagonal_one, Matrix.mul_one]]
  exact hcomp

/-- **The cubic matmul iteration converges to the polar factor `UVᵀ`** — P2 (cubic) through the P3
    glue. The classic `(3/2, −1/2, 0)` Newton–Schulz iteration provably computes Muon's update. -/
theorem nsStep_cubic_iterate_tendsto_polar
    (U V : Matrix (Fin n) (Fin n) ℝ) (σ : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (hσ : ∀ i, 0 < σ i ∧ σ i ≤ 1) :
    Filter.Tendsto (fun k => (nsStep (3 / 2) (-1 / 2) 0)^[k] (U * Matrix.diagonal σ * Vᵀ))
      Filter.atTop (nhds (U * Vᵀ)) :=
  nsStep_iterate_tendsto_polar (3 / 2) (-1 / 2) 0 gCubic gCubic_eq_nsScalar
    (fun _ h0 h1 => gCubic_iterate_tendsto_one h0 h1) U V σ hU hV hσ

/-- **The principled convergent quintic's matmul iteration also lands on `UVᵀ`** — same polar factor,
    one degree up, *faster* (third-order). Convergence is the coefficient choice, not the degree:
    `(15/8, −5/4, 3/8)` converges (this theorem), Muon's tuned `(3.4445, −4.7750, 2.0315)` bands (P4). -/
theorem nsStep_q5_iterate_tendsto_polar
    (U V : Matrix (Fin n) (Fin n) ℝ) (σ : Fin n → ℝ)
    (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (hσ : ∀ i, 0 < σ i ∧ σ i ≤ 1) :
    Filter.Tendsto (fun k => (nsStep (15 / 8) (-5 / 4) (3 / 8))^[k] (U * Matrix.diagonal σ * Vᵀ))
      Filter.atTop (nhds (U * Vᵀ)) :=
  nsStep_iterate_tendsto_polar (15 / 8) (-5 / 4) (3 / 8) q5Scalar q5Scalar_eq_nsScalar
    (fun _ h0 h1 => q5Scalar_iterate_tendsto_one h0 h1) U V σ hU hV hσ

-- ════════════════════════════════════════════════════════════════
-- § P4 — Muon's *tuned* quintic: band-landing, NOT asymptotic convergence (the honest tier)
-- ════════════════════════════════════════════════════════════════

/-- **Muon's actual tuned Newton–Schulz quintic** `φ(t) = 3.4445 t − 4.7750 t³ + 2.0315 t⁵`
    (Jordan 2024 — `planning/muon.md`). This is *not* a statement about quintics in general — the
    *principled* quintic `q5Scalar` `(15/8, −5/4, 3/8)` converges (`q5Scalar_iterate_tendsto_one`,
    faster than the cubic even). Convergence is the **coefficient choice**: Jordan tuned *these*
    coefficients for *speed to a band near 1 in ~5 steps*, deliberately giving up asymptotic
    convergence. The map straddles `1` (`qScalar_one_lt_one` ∧ `qScalar_half_gt_one`) and so
    oscillates; the theorems below prove the cubic's monotone-convergence hypotheses (`scalar_iterate_
    tendsto_one`) structurally *fail* here, so one must **not** state `qScalar^[k] → 1`. -/
noncomputable def qScalar (t : ℝ) : ℝ := nsScalar 3.4445 (-4.7750) 2.0315 t

/-- At the top of the normalized range, the tuned quintic pulls **below** `1`: `φ(1) = 0.701 < 1`. So
    `1` is *not* a fixed point of `qScalar` (contrast the cubic, whose only relevant fixed point IS
    `1`) — the asymptotic limit the cubic enjoys simply does not exist here. -/
theorem qScalar_one_lt_one : qScalar 1 < 1 := by
  simp only [qScalar, nsScalar]; norm_num

/-- In mid-range the tuned quintic **overshoots** above `1`: `φ(1/2) ≈ 1.189 > 1`. Together with
    `qScalar_one_lt_one` this shows `qScalar` takes values on *both* sides of `1` — it straddles the
    target rather than approaching it monotonically. -/
theorem qScalar_half_gt_one : 1 < qScalar (1 / 2) := by
  simp only [qScalar, nsScalar]; norm_num

/-- **The cubic's key bound fails for the tuned quintic** — `qScalar` is *not* `≤ 1` on `[0,1]` (it
    overshoots at `1/2`). This is exactly the hypothesis `g(t) ≤ 1` that powered the monotone-bounded
    convergence of `gCubic_iterate_tendsto_one` (P2); its failure is *why* the clean `→ 1` proof does
    not transfer to Muon's quintic, and why claiming `qScalar^[k] → 1` would be an overclaim. -/
theorem qScalar_not_le_one : ¬ ∀ t : ℝ, 0 ≤ t → t ≤ 1 → qScalar t ≤ 1 := by
  intro h
  linarith [h (1 / 2) (by norm_num) (by norm_num), qScalar_half_gt_one]

/-- **The honest *positive* statement: a finite-5-step band bound** (the form P4 actually supports).
    Five steps of Muon's tuned quintic from `σ = 1/2` land within `0.3` of `1`:
    `|qScalar^[5] (1/2) − 1| ≤ 3/10` (the orbit `0.5 → 1.19 → 0.90 → 0.83 → 0.94 → 0.77` oscillates in a
    band around `1`, never reaching it). This matches the implementation's fixed-5-step, "rough is
    fine — we recompute next optimizer step anyway" design (`planning/muon.md`): not convergence, a
    *band*. The universal interval version `∀ σ ∈ [σ_min, 1], |φ^[5](σ) − 1| ≤ δ` is a degree-5⁵
    polynomial bound over an interval (genuine interval arithmetic) and is left open by hand. -/
theorem qScalar_iterate_band_half : |qScalar^[5] (1 / 2) - 1| ≤ 3 / 10 := by
  simp only [qScalar, Function.iterate_succ, Function.iterate_zero, Function.comp_apply, id_eq,
    nsScalar]
  rw [abs_le]
  constructor <;> norm_num

end Proofs.MuonNewtonSchulz
