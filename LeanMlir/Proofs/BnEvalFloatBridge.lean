import LeanMlir.Proofs.FloatComposeBridge

/-! # ℝ → Float32 bridge: eval-mode BatchNorm/LayerNorm as a fixed affine

Deployed accuracy runs BN/LN at **eval** with *running* statistics, not the
batch. With fixed `μ, σ², γ, β` the normalization is a **constant per-channel
affine**

  `y = γ·(x − μ)/√(σ²+ε) + β = a·x + b`,   `a = γ/√(σ²+ε)`,  `b = β − γ·μ/√(σ²+ε)`,

where `a, b` are precomputed **once, offline** — the runtime kernel does only a
scale and a shift. There is no batch reduction and **no runtime `rsqrt`**: the
square root that forced `BnFloatBridge`'s keystone (`rsqrt_lipschitz`,
`bnIstd_close_at`) is folded into the constant `a` at deploy time. So the
deployed-forward bridge for normalization is far simpler than the training-mode
one already built — a near-trivial `FloatClose` affine instance
(`planning/floatbridge_certificate_gaps.md` §4).

This file:
* `bnEvalAffine a b` (ℝ) / `bnEvalAffineF a b` (float, `fl(fl(aᵢ·xᵢ) ⊕ bᵢ)`) —
  the per-coordinate affine and its rounded peer (one `mul`, one `add`; fan-in 1,
  so **no** Higham γ amplification).
* `bnEvalAffine_fold` — the eval-mode BN formula `γ(x−μ)/√(σ²+ε)+β` **equals**
  `a·x+b` with the deploy-time constants: the affine genuinely *is* eval BN, the
  `rsqrt` is a constant.
* `floatClose_bnEval` — the affine is `FloatClose`, modulus `bnEvalErr` (a clean
  `mulErr` + one rounded-add floor, affine in the inherited error `e`). Drops
  straight into `FloatClose.comp` to build a deployed eval-forward fold.

3-axiom clean, reusing `mul_close`/`mulErr` from `FloatBridge`.
-/

namespace Proofs

open scoped Real

/-- Eval-mode BN/LN as a fixed **per-coordinate affine** `yᵢ = aᵢ·xᵢ + bᵢ`. The
    per-*channel* deploy is the special case where `a, b` are channel-broadcast
    (constant within a channel); per-coordinate subsumes it. -/
noncomputable def bnEvalAffine {n : Nat} (a b : Vec n) : Vec n → Vec n :=
  fun x i => a i * x i + b i

/-- The deploy-time scale `a = γ/√(σ²+ε)` and shift `b = β − γμ/√(σ²+ε)` **are**
    eval-mode BN: `a·x + b = γ·(x−μ)/√(σ²+ε) + β`. The `√` lives only in the
    offline constants, so the runtime map the float bridge sees is a bare affine. -/
theorem bnEvalAffine_fold (γ β μ σ2 ε x : ℝ) (h : 0 < σ2 + ε) :
    γ / Real.sqrt (σ2 + ε) * x + (β - γ * μ / Real.sqrt (σ2 + ε))
      = γ * ((x - μ) / Real.sqrt (σ2 + ε)) + β := by
  have hs : Real.sqrt (σ2 + ε) ≠ 0 := (Real.sqrt_pos.mpr h).ne'
  field_simp
  ring

namespace FloatModel

variable (M : FloatModel)

/-- Rounded eval-mode affine: `fl(fl(aᵢ·xᵢ) ⊕ bᵢ)` — one rounded product, one
    rounded sum, per coordinate. The deployed BN/LN-at-eval kernel. -/
noncomputable def bnEvalAffineF {n : Nat} (a b : Vec n) : Vec n → Vec n :=
  fun x i => M.add (M.mul (a i) (x i)) (b i)

end FloatModel

/-- Output-magnitude bound for the eval affine (both ℝ and float):
    `(1+u)·((1+u)·a'·A + β)` (one rounded mul under one rounded add). -/
noncomputable def bnEvalAct (u a' A β : ℝ) : ℝ := (1 + u) * ((1 + u) * a' * A + β)

/-- Error modulus of the eval affine: the rounded-add floor `u·((1+u)a'A + β)`
    plus the product budget `mulErr u a' A 0 e` (scale exact, activation error `e`).
    Affine in `e` — `mulErr u a' A 0 e = u·a'·(A+e) + a'·e`. -/
noncomputable def bnEvalErr (u a' A β e : ℝ) : ℝ :=
  u * ((1 + u) * a' * A + β) + FloatModel.mulErr u a' A 0 e

/-- **Eval-mode BN/LN is `FloatClose`** — the deployed-forward win. A fixed
    per-coordinate affine bridges with modulus `bnEvalErr` (no fan-in γ, no
    `rsqrt`): magnitudes thread through `bnEvalAct`, the inherited error `e`
    passes the `(1+u)·a'` scale plus a constant rounding floor. Composes via
    `FloatClose.comp` into a deployed eval-forward fold. -/
theorem floatClose_bnEval {n : Nat} (M : FloatModel) (a b : Vec n)
    {a' A β : ℝ} (hA : 0 ≤ A) (ha' : 0 ≤ a') (hβ : 0 ≤ β)
    (ha : ∀ i, |a i| ≤ a') (hb : ∀ i, |b i| ≤ β) :
    FloatClose A (bnEvalAct M.u a' A β)
      (bnEvalAffine a b) (M.bnEvalAffineF a b)
      (fun e => bnEvalErr M.u a' A β e) := by
  have hu := M.u_nonneg
  -- |fl(aᵢ·xᵢ)| ≤ (1+u)·a'·A  whenever |xᵢ| ≤ A  (one rounded product)
  have hmulMag : ∀ (x : Vec n), (∀ k, |x k| ≤ A) → ∀ i,
      |M.mul (a i) (x i)| ≤ (1 + M.u) * a' * A := by
    intro x hx i
    have hxy : |a i * x i| ≤ a' * A :=
      (abs_mul _ _).le.trans (mul_le_mul (ha i) (hx i) (abs_nonneg _) ha')
    have hrnd : |M.mul (a i) (x i) - a i * x i| ≤ M.u * |a i * x i| := M.err _
    have htri : |M.mul (a i) (x i)| ≤ |M.mul (a i) (x i) - a i * x i| + |a i * x i| := by
      simpa using abs_sub_le (M.mul (a i) (x i)) (a i * x i) 0
    calc |M.mul (a i) (x i)|
        ≤ M.u * |a i * x i| + |a i * x i| := by linarith
      _ = (1 + M.u) * |a i * x i| := by ring
      _ ≤ (1 + M.u) * (a' * A) := by
            apply mul_le_mul_of_nonneg_left hxy (by linarith)
      _ = (1 + M.u) * a' * A := by ring
  refine ⟨fun v hv i => ?_, fun vt va e hva hvt hd i => ?_⟩
  · -- magnitudes: both ℝ and float ≤ bnEvalAct
    constructor
    · -- |aᵢ·vᵢ + bᵢ| ≤ a'A + β ≤ bnEvalAct
      have : |a i * v i + b i| ≤ a' * A + β :=
        (abs_add_le _ _).trans (by
          have h1 : |a i * v i| ≤ a' * A :=
            (abs_mul _ _).le.trans (mul_le_mul (ha i) (hv i) (abs_nonneg _) ha')
          linarith [hb i])
      refine this.trans ?_
      have h1u : (1 : ℝ) ≤ 1 + M.u := by linarith
      have hpos : 0 ≤ (1 + M.u) * a' * A + β := by positivity
      calc a' * A + β ≤ (1 + M.u) * a' * A + β := by nlinarith [mul_nonneg ha' hA]
        _ ≤ (1 + M.u) * ((1 + M.u) * a' * A + β) :=
            le_mul_of_one_le_left hpos h1u
    · -- |fl(fl(aᵢ·vᵢ) ⊕ bᵢ)| ≤ (1+u)·((1+u)a'A + β) = bnEvalAct
      have hp : |M.mul (a i) (v i)| ≤ (1 + M.u) * a' * A := hmulMag v hv i
      have hsum : |M.mul (a i) (v i) + b i| ≤ (1 + M.u) * a' * A + β :=
        (abs_add_le _ _).trans (by linarith [hb i])
      have hrnd : |M.add (M.mul (a i) (v i)) (b i) - (M.mul (a i) (v i) + b i)|
          ≤ M.u * |M.mul (a i) (v i) + b i| := M.err _
      have htri : |M.add (M.mul (a i) (v i)) (b i)|
          ≤ |M.add (M.mul (a i) (v i)) (b i) - (M.mul (a i) (v i) + b i)|
            + |M.mul (a i) (v i) + b i| := by
        simpa using abs_sub_le (M.add (M.mul (a i) (v i)) (b i))
          (M.mul (a i) (v i) + b i) 0
      show |M.add (M.mul (a i) (v i)) (b i)| ≤ bnEvalAct M.u a' A β
      have hmono : M.u * |M.mul (a i) (v i) + b i| ≤ M.u * ((1 + M.u) * a' * A + β) :=
        mul_le_mul_of_nonneg_left hsum hu
      calc |M.add (M.mul (a i) (v i)) (b i)|
          ≤ M.u * |M.mul (a i) (v i) + b i| + |M.mul (a i) (v i) + b i| := by linarith
        _ ≤ M.u * ((1 + M.u) * a' * A + β) + ((1 + M.u) * a' * A + β) := by linarith
        _ = (1 + M.u) * ((1 + M.u) * a' * A + β) := by ring
        _ = bnEvalAct M.u a' A β := rfl
  · -- error: |fl(fl(aᵢ·vtᵢ) ⊕ bᵢ) − (aᵢ·vaᵢ + bᵢ)| ≤ bnEvalErr u a' A β e
    -- product piece (scale exact, activation error e): mul_close at ea=0, ec=e
    have hmul : |M.mul (a i) (vt i) - a i * va i| ≤ FloatModel.mulErr M.u a' A 0 e :=
      M.mul_close (xt := a i) (x := a i) (yt := vt i) (y := va i)
        (ea := 0) (ec := e)
        (le_of_eq (by rw [sub_self, abs_zero])) (hd i) (ha i) (hva i)
    -- add piece: round(p ⊕ bᵢ) against (p) shifted, then the product budget
    set p := M.mul (a i) (vt i) with hp
    have hpmag : |p| ≤ (1 + M.u) * a' * A := hmulMag vt hvt i
    have hsum : |p + b i| ≤ (1 + M.u) * a' * A + β :=
      (abs_add_le _ _).trans (by linarith [hb i])
    have hrnd : |M.add p (b i) - (p + b i)| ≤ M.u * |p + b i| := M.err _
    have hmono : M.u * |p + b i| ≤ M.u * ((1 + M.u) * a' * A + β) :=
      mul_le_mul_of_nonneg_left hsum hu
    -- triangle: fl(p⊕b) → (p+b) → (a·va + b)
    have htri : |M.add p (b i) - (a i * va i + b i)|
        ≤ |M.add p (b i) - (p + b i)| + |(p + b i) - (a i * va i + b i)| :=
      abs_sub_le _ _ _
    have hshift : |(p + b i) - (a i * va i + b i)| = |p - a i * va i| := by
      congr 1; ring
    show |M.bnEvalAffineF a b vt i - bnEvalAffine a b va i| ≤ bnEvalErr M.u a' A β e
    have : |M.add p (b i) - (a i * va i + b i)|
        ≤ M.u * ((1 + M.u) * a' * A + β) + FloatModel.mulErr M.u a' A 0 e := by
      have h1 : |p - a i * va i| ≤ FloatModel.mulErr M.u a' A 0 e := hmul
      calc |M.add p (b i) - (a i * va i + b i)|
          ≤ |M.add p (b i) - (p + b i)| + |(p + b i) - (a i * va i + b i)| := htri
        _ = |M.add p (b i) - (p + b i)| + |p - a i * va i| := by rw [hshift]
        _ ≤ M.u * ((1 + M.u) * a' * A + β) + FloatModel.mulErr M.u a' A 0 e := by linarith
    simpa [FloatModel.bnEvalAffineF, bnEvalAffine, bnEvalErr, hp] using this

end Proofs
