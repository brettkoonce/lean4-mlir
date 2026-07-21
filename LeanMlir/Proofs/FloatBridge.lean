import LeanMlir.Proofs.Foundation.MLP

/-! # ℝ → Float32 bridge, Tier 1: standard-model rounding bounds

Every theorem in `LeanMlir/Proofs/` is over exact reals; the executed kernels
are binary32. This file is the first bite at that gap, for the toy nets only
(linear, MLP — the Tier-1 ladder): a **standard-model** formalization of
rounded arithmetic and forward error bounds for the same `dense`/`relu`
compositions the train-step proofs certify.

The model is *hypothesis-style*, like the suite's `0 < ε` / off-the-kink
hypotheses: a `FloatModel` is any rounding operator `rnd` with relative error
`u` (`|rnd x − x| ≤ u·|x|`). No project axioms — IEEE-754 binary32
round-to-nearest satisfies the interface with `u = 2⁻²⁴` **in the normal
range** (Higham, *Accuracy and Stability*, §2.2; the standard model without
underflow — the subnormal absolute-error term is future work, as is the
gradient half). `exactModel` (`rnd = id`, `u = 0`) shows the interface is
inhabited and collapses every bound to `0`.

Design notes, in suite style:
* **Order-robustness.** `FloatModel.dot` fixes one association (left fold),
  but the bound `((1+u)^(n+1) − 1)·Σ|xᵢyᵢ|` is the classical one valid for
  *every* summation order — so the statement survives a backend that
  reassociates (IREE tiles reductions), at the cost of not benefiting from
  pairwise summation's tighter `log n` compounding.
* **ReLU is exact in floating point** — comparison/selection rounds nothing,
  which is why `mlpF` interleaves bare `relu` with no `rnd` and the bridge
  only needs `relu`'s 1-Lipschitz error propagation (`relu_close`). The op
  that forced the kink hypotheses on the `ℝ` side is the free op here.
* **Error shapes are `denseErr`**, one def reused at every layer: the rounded
  layer at a perturbed input vs the real layer at the real input. The MLP
  capstone (`mlp_float_close`) threads it three times; its `e₀`/`e₁`
  hypotheses are uniformizations of the per-coordinate layer bounds,
  dischargeable at any concrete instance by finite max.
-/

namespace Proofs

/-- **The standard model of rounded arithmetic.** Any rounding operator with
    relative error `u`. binary32 round-to-nearest instantiates this with
    `u = 2⁻²⁴` on the normal range. -/
structure FloatModel where
  rnd : ℝ → ℝ
  u : ℝ
  u_nonneg : 0 ≤ u
  err : ∀ x : ℝ, |rnd x - x| ≤ u * |x|

/-- The unit roundoff of IEEE-754 binary32 (round-to-nearest-even). -/
noncomputable def u32 : ℝ := ((2 : ℝ) ^ (24 : ℕ))⁻¹

/-- The normal-range unit roundoff of fp8 **E4M3** (1-4-3, 3 mantissa bits):
    `2⁻⁴ = 1/16` (6.25%) — the leaf precision of the §3c E4M3 MNIST demo
    (`scripts/mnist_e4m3_demo.py`). Outside the normal range (subnormals,
    near the 448 max) the relative model degrades; see §2/§5 of the plan. -/
noncomputable def u_e4m3 : ℝ := ((2 : ℝ) ^ (4 : ℕ))⁻¹

namespace FloatModel

variable (M : FloatModel)

/-- Rounded addition: `fl(x + y)`. -/
noncomputable def add (x y : ℝ) : ℝ := M.rnd (x + y)

/-- Rounded multiplication: `fl(x · y)`. -/
noncomputable def mul (x y : ℝ) : ℝ := M.rnd (x * y)

/-- Rounded dot product, left-fold association (`((x₀y₀ + x₁y₁) + …)`).
    The bound below is association-independent, so the choice is immaterial. -/
noncomputable def dot : {n : Nat} → Vec n → Vec n → ℝ
  | 0, _, _ => 0
  | n + 1, x, y =>
      M.add (dot (fun i => x i.castSucc) (fun i => y i.castSucc))
            (M.mul (x (Fin.last n)) (y (Fin.last n)))

theorem dot_zero (x y : Vec 0) : M.dot x y = 0 := rfl

theorem dot_succ {n : Nat} (x y : Vec (n + 1)) :
    M.dot x y =
      M.add (M.dot (fun i => x i.castSucc) (fun i => y i.castSucc))
            (M.mul (x (Fin.last n)) (y (Fin.last n))) := rfl

/-- Rounded dense layer — the float peer of `Proofs.dense`
    (`fl(Σᵢ xᵢ·Wᵢⱼ) ⊕ bⱼ`, every `+`/`·` rounded). -/
noncomputable def dense {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) :
    Vec n :=
  fun j => M.add (M.dot x (fun i => W i j)) (b j)

/-- Rounded MLP forward — the float peer of the Tier-1
    `dense W₂ b₂ ∘ relu ∘ dense W₁ b₁ ∘ relu ∘ dense W₀ b₀` composition
    (`MlpTrainStep.lean`). `relu` appears bare: max-with-0 is exact in
    floating point. -/
noncomputable def mlpF {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) : Vec d₃ :=
  M.dense W₂ b₂ (relu d₂ (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))))

-- ════════════════════════════════════════════════════════════════
-- § Exponent bookkeeping
-- ════════════════════════════════════════════════════════════════

private theorem one_le_one_add_u : (1 : ℝ) ≤ 1 + M.u := by
  have := M.u_nonneg; linarith

private theorem one_le_pow_one_add_u (k : ℕ) : (1 : ℝ) ≤ (1 + M.u) ^ k :=
  one_le_pow₀ M.one_le_one_add_u

private theorem one_add_u_le_pow {k : ℕ} (hk : 1 ≤ k) :
    1 + M.u ≤ (1 + M.u) ^ k := by
  have h := pow_le_pow_right₀ M.one_le_one_add_u hk
  simpa using h

/-- `(1+u)^k − 1 ≤ k·u·(1+u)^k` — the reading key from the compounded form
    back to the familiar first-order "≈ k·u" bound. -/
theorem pow_one_add_sub_one_le (u : ℝ) (hu : 0 ≤ u) (k : ℕ) :
    (1 + u) ^ k - 1 ≤ k * u * (1 + u) ^ k := by
  induction k with
  | zero => simp
  | succ k ih =>
    have h0 : (0 : ℝ) ≤ 1 + u := by linarith
    have hs : (1 + u) ^ (k + 1) = (1 + u) ^ k * (1 + u) := pow_succ _ _
    have h1k : (1 : ℝ) ≤ (1 + u) ^ (k + 1) := one_le_pow₀ (by linarith)
    have hihm : ((1 + u) ^ k - 1) * (1 + u) ≤ (k * u * (1 + u) ^ k) * (1 + u) :=
      mul_le_mul_of_nonneg_right ih h0
    have hu1 : u * 1 ≤ u * (1 + u) ^ (k + 1) := mul_le_mul_of_nonneg_left h1k hu
    push_cast
    nlinarith [hihm, hu1, hs]

-- ════════════════════════════════════════════════════════════════
-- § The two scalar assembly steps (pure-ℝ bookkeeping)
-- ════════════════════════════════════════════════════════════════

/-- One rounded `add` of an accumulated dot prefix (`st ≈ S`, budget
    `(C−1)·A`) and a rounded product (`pt ≈ p`): the compounded budget is
    `C·(1+u) − 1` over the inflated magnitude `A + |p|`. -/
private theorem step_bound {u st pt S p A C r : ℝ}
    (hu : 0 ≤ u) (hSA : |S| ≤ A) (h1uC : 1 + u ≤ C)
    (hih : |st - S| ≤ (C - 1) * A)
    (hpt : |pt - p| ≤ u * |p|)
    (hadd : |r - (st + pt)| ≤ u * |st + pt|) :
    |r - (S + p)| ≤ (C * (1 + u) - 1) * (A + |p|) := by
  have hp0 : 0 ≤ |p| := abs_nonneg p
  have hst : |st| ≤ C * A := by
    have h1 : |st| ≤ |st - S| + |S| := by simpa using abs_sub_le st S 0
    linarith
  have hptb : |pt| ≤ (1 + u) * |p| :=
    calc |pt| ≤ |pt - p| + |p| := by simpa using abs_sub_le pt p 0
      _ ≤ u * |p| + |p| := by linarith
      _ = (1 + u) * |p| := by ring
  have hsum : |st + pt| ≤ C * A + (1 + u) * |p| :=
    (abs_add_le st pt).trans (by linarith)
  have htri : |r - (S + p)| ≤ |r - (st + pt)| + (|st - S| + |pt - p|) := by
    have h1 : |r - (S + p)| ≤ |r - (st + pt)| + |st + pt - (S + p)| :=
      abs_sub_le _ _ _
    have h2 : |st + pt - (S + p)| ≤ |st - S| + |pt - p| := by
      have h3 : st + pt - (S + p) = (st - S) + (pt - p) := by ring
      rw [h3]; exact abs_add_le _ _
    linarith
  have hmono : u * |st + pt| ≤ u * (C * A + (1 + u) * |p|) :=
    mul_le_mul_of_nonneg_left hsum hu
  have hfin : (u * u + 2 * u) * |p| ≤ (C * (1 + u) - 1) * |p| := by
    have h4 : (1 + u) * (1 + u) ≤ C * (1 + u) :=
      mul_le_mul_of_nonneg_right h1uC (by linarith)
    have h5 : u * u + 2 * u ≤ C * (1 + u) - 1 := by nlinarith
    exact mul_le_mul_of_nonneg_right h5 hp0
  calc |r - (S + p)|
      ≤ u * |st + pt| + (|st - S| + |pt - p|) := by linarith
    _ ≤ u * (C * A + (1 + u) * |p|) + ((C - 1) * A + u * |p|) := by linarith
    _ = (C * (1 + u) - 1) * A + (u * u + 2 * u) * |p| := by ring
    _ ≤ (C * (1 + u) - 1) * A + (C * (1 + u) - 1) * |p| := by linarith
    _ = (C * (1 + u) - 1) * (A + |p|) := by ring

/-- One rounded bias-add on top of a rounded dot (`dt ≈ dxt`, budget
    `(C−1)·Sxt`) whose exact value `dxt` is itself `L·e`-near the real dot
    `d`: the layer budget is `(C·(1+u) − 1)·(SE + |bb|) + L·e`. -/
private theorem dense_step_bound {u r dt dxt d bb Sxt SE L e C : ℝ}
    (hu : 0 ≤ u) (hC : 1 ≤ C)
    (hdt : |dt - dxt| ≤ (C - 1) * Sxt)
    (hdxt : |dxt| ≤ Sxt) (hSxtSE : Sxt ≤ SE)
    (hlip : |dxt - d| ≤ L * e)
    (hadd : |r - (dt + bb)| ≤ u * |dt + bb|) :
    |r - (d + bb)| ≤ (C * (1 + u) - 1) * (SE + |bb|) + L * e := by
  have hb0 : 0 ≤ |bb| := abs_nonneg bb
  have hCu' : (0 : ℝ) ≤ (C - 1) * (1 + u) :=
    mul_nonneg (by linarith) (by linarith)
  have hC1 : 0 ≤ C * (1 + u) - 1 := by nlinarith [hCu']
  have hCu : u ≤ C * (1 + u) - 1 := by nlinarith [hCu']
  have hdtb : |dt| ≤ C * Sxt := by
    have h1 : |dt| ≤ |dt - dxt| + |dxt| := by simpa using abs_sub_le dt dxt 0
    linarith
  have hsum : |dt + bb| ≤ C * Sxt + |bb| :=
    (abs_add_le dt bb).trans (by linarith)
  have htri : |r - (d + bb)| ≤ |r - (dt + bb)| + (|dt - dxt| + |dxt - d|) := by
    have h1 : |r - (d + bb)| ≤ |r - (dt + bb)| + |dt + bb - (d + bb)| :=
      abs_sub_le _ _ _
    have h2 : |dt + bb - (d + bb)| ≤ |dt - dxt| + |dxt - d| := by
      have h3 : dt + bb - (d + bb) = (dt - dxt) + (dxt - d) := by ring
      rw [h3]; exact abs_add_le _ _
    linarith
  have hmono : u * |dt + bb| ≤ u * (C * Sxt + |bb|) :=
    mul_le_mul_of_nonneg_left hsum hu
  have hmonoSE : (C * (1 + u) - 1) * Sxt ≤ (C * (1 + u) - 1) * SE :=
    mul_le_mul_of_nonneg_left hSxtSE hC1
  have hmonob : u * |bb| ≤ (C * (1 + u) - 1) * |bb| :=
    mul_le_mul_of_nonneg_right hCu hb0
  calc |r - (d + bb)|
      ≤ u * |dt + bb| + (|dt - dxt| + |dxt - d|) := by linarith
    _ ≤ u * (C * Sxt + |bb|) + ((C - 1) * Sxt + L * e) := by linarith
    _ = (C * (1 + u) - 1) * Sxt + u * |bb| + L * e := by ring
    _ ≤ (C * (1 + u) - 1) * SE + (C * (1 + u) - 1) * |bb| + L * e := by
        linarith
    _ = (C * (1 + u) - 1) * (SE + |bb|) + L * e := by ring

-- ════════════════════════════════════════════════════════════════
-- § Dot product: the compounded-(1+u) bound
-- ════════════════════════════════════════════════════════════════

/-- **Rounded dot product forward error** —
    `|fl(x·y) − x·y| ≤ ((1+u)^(n+1) − 1)·Σᵢ|xᵢyᵢ|`.

    The classical bound (Higham §3.1), in the exact compounded form (no
    `n·u < 1` side condition); valid for every association of the sum, not
    just the left fold `dot` fixes. The exponent is `n+1` rather than the
    optimal `n` because `dot` rounds the seed addition with `0` too. -/
theorem dot_close : ∀ {n : ℕ} (x y : Vec n),
    |M.dot x y - ∑ i, x i * y i| ≤
      ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |x i * y i| := by
  intro n
  induction n with
  | zero => intro x y; simp [FloatModel.dot]
  | succ n ih =>
    intro x y
    rw [M.dot_succ x y]
    simp only [Fin.sum_univ_castSucc]
    rw [show ((1 : ℝ) + M.u) ^ (n + 1 + 1) = (1 + M.u) ^ (n + 1) * (1 + M.u)
        from pow_succ _ _]
    exact step_bound M.u_nonneg
      (Finset.abs_sum_le_sum_abs _ _)
      (M.one_add_u_le_pow (by omega))
      (ih (fun i => x i.castSucc) (fun i => y i.castSucc))
      (M.err _) (M.err _)

/-- `dot_close` in the first-order shape: `≤ (n+1)·u·(1+u)^(n+1)·Σ|xᵢyᵢ|`. -/
theorem dot_close_linear {n : ℕ} (x y : Vec n) :
    |M.dot x y - ∑ i, x i * y i| ≤
      (n + 1 : ℝ) * M.u * (1 + M.u) ^ (n + 1) * ∑ i, |x i * y i| := by
  refine (M.dot_close x y).trans ?_
  have h := pow_one_add_sub_one_le M.u M.u_nonneg (n + 1)
  push_cast at h
  exact mul_le_mul_of_nonneg_right h
    (Finset.sum_nonneg fun i _ => abs_nonneg _)

-- ════════════════════════════════════════════════════════════════
-- § Mixed-precision dot: a leaf roundoff `u_leaf` + an accumulate `u_acc`
-- ════════════════════════════════════════════════════════════════

/-- **Mixed-precision dot product.** The matmul inputs are first rounded by a
    *leaf* model `L` (low precision, `u_leaf` — e.g. bf16 `2⁻⁸` or fp8-E4M3
    `2⁻⁴`); the accumulation `M.dot` then rounds every `+`/`·` at the
    *accumulate* precision `M.u` (`u_acc`, typically fp32 `2⁻²⁴`). This is the
    deployed bf16-mixed kernel shape: bf16 leaf compute, fp32 accumulate. -/
noncomputable def dotMixed (L : FloatModel) {n : Nat} (x y : Vec n) : ℝ :=
  M.dot (fun i => L.rnd (x i)) (fun i => L.rnd (y i))

/-- **Mixed-precision dot forward error, decomposed.** The leaf precision
    contributes only a **flat per-leaf term** `(2·u_leaf + u_leaf²)·Σ|xᵢyᵢ|`
    (NOT fan-in amplified); the fan-in amplification rides entirely on the
    *accumulate* precision as the Higham γ-factor `((1+u_acc)^(n+1) − 1)`. That
    separation is exactly why bf16-mixed is non-vacuous where pure bf16 is not:
    the `1/u` fan-in wall sits at `u_acc = 2⁻²⁴`, not at the leaf precision. -/
theorem dot_close_mixed (L : FloatModel) {n : ℕ} (x y : Vec n) :
    |M.dotMixed L x y - ∑ i, x i * y i| ≤
      ((1 + M.u) ^ (n + 1) - 1) * (∑ i, |L.rnd (x i) * L.rnd (y i)|)
        + (2 * L.u + L.u ^ 2) * ∑ i, |x i * y i| := by
  -- the flat per-leaf perturbation: ∑x̃ỹ vs ∑xy, each term ≤ (2u+u²)|xy|
  have hleaf : |(∑ i, L.rnd (x i) * L.rnd (y i)) - ∑ i, x i * y i| ≤
      (2 * L.u + L.u ^ 2) * ∑ i, |x i * y i| := by
    rw [← Finset.sum_sub_distrib, Finset.mul_sum]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => ?_)
    have hxe : |L.rnd (x i) - x i| ≤ L.u * |x i| := L.err (x i)
    have hye : |L.rnd (y i) - y i| ≤ L.u * |y i| := L.err (y i)
    have hxb : |L.rnd (x i)| ≤ (1 + L.u) * |x i| :=
      calc |L.rnd (x i)| ≤ |L.rnd (x i) - x i| + |x i| := by
            simpa using abs_sub_le (L.rnd (x i)) (x i) 0
        _ ≤ (1 + L.u) * |x i| := by linarith
    have t1 : |L.rnd (x i)| * |L.rnd (y i) - y i| ≤
        (1 + L.u) * |x i| * (L.u * |y i|) :=
      mul_le_mul hxb hye (abs_nonneg _)
        (mul_nonneg (by linarith [L.u_nonneg]) (abs_nonneg _))
    have t2 : |y i| * |L.rnd (x i) - x i| ≤ |y i| * (L.u * |x i|) :=
      mul_le_mul_of_nonneg_left hxe (abs_nonneg _)
    calc |L.rnd (x i) * L.rnd (y i) - x i * y i|
        = |L.rnd (x i) * (L.rnd (y i) - y i) + y i * (L.rnd (x i) - x i)| := by
          rw [show L.rnd (x i) * L.rnd (y i) - x i * y i =
            L.rnd (x i) * (L.rnd (y i) - y i) + y i * (L.rnd (x i) - x i) from by
            ring]
      _ ≤ |L.rnd (x i) * (L.rnd (y i) - y i)| + |y i * (L.rnd (x i) - x i)| :=
          abs_add_le _ _
      _ = |L.rnd (x i)| * |L.rnd (y i) - y i| + |y i| * |L.rnd (x i) - x i| := by
          rw [abs_mul, abs_mul]
      _ ≤ (1 + L.u) * |x i| * (L.u * |y i|) + |y i| * (L.u * |x i|) := by linarith
      _ = (2 * L.u + L.u ^ 2) * |x i * y i| := by rw [abs_mul]; ring
  rw [FloatModel.dotMixed]
  refine (abs_sub_le _ (∑ i, L.rnd (x i) * L.rnd (y i)) _).trans ?_
  have hacc' : |M.dot (fun i => L.rnd (x i)) (fun i => L.rnd (y i)) -
      ∑ i, L.rnd (x i) * L.rnd (y i)| ≤
      ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |L.rnd (x i) * L.rnd (y i)| := by
    simpa using M.dot_close (fun i => L.rnd (x i)) (fun i => L.rnd (y i))
  linarith [hacc', hleaf]

/-- `dot_close_mixed` folded to a single `Σ|xᵢyᵢ|` factor — the directly
    instantiable form. Bounds the leaf-rounded magnitudes by `(1+u_leaf)²`,
    so the whole error is `[γ_acc·(1+u_leaf)² + (2u_leaf + u_leaf²)]·Σ|xᵢyᵢ|`.
    At bf16 leaf / fp32 accumulate (`u_leaf = 2⁻⁸`, `u_acc = 2⁻²⁴`, fan-in a
    few hundred) the bracket is ≈ the flat `2·2⁻⁸ ≈ 0.8%` leaf term plus a
    negligible accumulate γ — the shipped-artifact budget. -/
theorem dot_close_mixed_uniform (L : FloatModel) {n : ℕ} (x y : Vec n) :
    |M.dotMixed L x y - ∑ i, x i * y i| ≤
      (((1 + M.u) ^ (n + 1) - 1) * (1 + L.u) ^ 2 + (2 * L.u + L.u ^ 2))
        * ∑ i, |x i * y i| := by
  refine (M.dot_close_mixed L x y).trans ?_
  have hγ0 : (0:ℝ) ≤ (1 + M.u) ^ (n + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
  have hmag : (∑ i, |L.rnd (x i) * L.rnd (y i)|) ≤
      (1 + L.u) ^ 2 * ∑ i, |x i * y i| := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have hxb : |L.rnd (x i)| ≤ (1 + L.u) * |x i| :=
      calc |L.rnd (x i)| ≤ |L.rnd (x i) - x i| + |x i| := by
            simpa using abs_sub_le (L.rnd (x i)) (x i) 0
        _ ≤ (1 + L.u) * |x i| := by linarith [L.err (x i)]
    have hyb : |L.rnd (y i)| ≤ (1 + L.u) * |y i| :=
      calc |L.rnd (y i)| ≤ |L.rnd (y i) - y i| + |y i| := by
            simpa using abs_sub_le (L.rnd (y i)) (y i) 0
        _ ≤ (1 + L.u) * |y i| := by linarith [L.err (y i)]
    rw [abs_mul, abs_mul]
    calc |L.rnd (x i)| * |L.rnd (y i)|
        ≤ (1 + L.u) * |x i| * ((1 + L.u) * |y i|) :=
          mul_le_mul hxb hyb (abs_nonneg _)
            (mul_nonneg (by linarith [L.u_nonneg]) (abs_nonneg _))
      _ = (1 + L.u) ^ 2 * (|x i| * |y i|) := by ring
  have hsum0 : (0:ℝ) ≤ ∑ i, |x i * y i| :=
    Finset.sum_nonneg fun i _ => abs_nonneg _
  nlinarith [mul_le_mul_of_nonneg_left hmag hγ0, hsum0]

/-- **Mixed-precision dense layer** — leaf precision `L` on the matmul (the
    `dotMixed`), accumulate precision `M` on the bias add. The deployed
    bf16-mixed dense layer (fp32 master/accumulate, bf16 leaf compute). -/
noncomputable def denseMixed (L : FloatModel) {m n : Nat} (W : Mat m n)
    (b : Vec n) (x : Vec m) : Vec n :=
  fun j => M.add (M.dotMixed L x (fun i => W i j)) (b j)

/-- **Mixed-precision dense forward error.** The leaf precision enters only
    through the flat `dotMixed` term; the accumulate precision rides the bias
    add and the fan-in γ. The bf16-mixed / fp8 dense layer falls out by setting
    `L.u`. -/
theorem dense_close_mixed (L : FloatModel) {m n : Nat} (W : Mat m n)
    (b : Vec n) (x : Vec m) (j : Fin n) :
    |M.denseMixed L W b x j - Proofs.dense W b x j| ≤
      M.u * ((∑ i, |x i * W i j|) + |b j|)
        + (1 + M.u) * ((((1 + M.u) ^ (m + 1) - 1) * (1 + L.u) ^ 2
          + (2 * L.u + L.u ^ 2)) * ∑ i, |x i * W i j|) := by
  have hu := M.u_nonneg
  set br := ((1 + M.u) ^ (m + 1) - 1) * (1 + L.u) ^ 2 + (2 * L.u + L.u ^ 2)
    with hbr
  set S := ∑ i, |x i * W i j| with hS
  set P := ∑ i, x i * W i j with hP
  set p := M.dotMixed L x (fun i => W i j) with hp
  have hD : |p - P| ≤ br * S := by
    have h := M.dot_close_mixed_uniform L x (fun i => W i j)
    simpa [hp, hP, hS, hbr] using h
  have hS0 : (0:ℝ) ≤ S := Finset.sum_nonneg fun i _ => abs_nonneg _
  have hbr0 : 0 ≤ br := by
    have h1 : (0:ℝ) ≤ (1 + M.u) ^ (m + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith))
    have h2 : (0:ℝ) ≤ (1 + L.u) ^ 2 := sq_nonneg _
    have h3 : (0:ℝ) ≤ 2 * L.u + L.u ^ 2 := by
      have := L.u_nonneg; positivity
    rw [hbr]; positivity
  have hPS : |P| ≤ S := by rw [hP, hS]; exact Finset.abs_sum_le_sum_abs _ _
  have hpabs : |p| ≤ S + br * S := by
    have h1 : |p| ≤ |P| + |p - P| := by
      calc |p| = |P + (p - P)| := by ring_nf
        _ ≤ |P| + |p - P| := abs_add_le _ _
    have h2 : br * S ≤ br * S := le_rfl
    linarith [hPS, hD]
  -- denseMixed j = M.add p (b j); real = P + b j
  have hreal : Proofs.dense W b x j = P + b j := rfl
  have hmix : M.denseMixed L W b x j = M.add p (b j) := rfl
  rw [hmix, hreal]
  have hadd : |M.add p (b j) - (p + b j)| ≤ M.u * |p + b j| := M.err _
  have htri : |M.add p (b j) - (P + b j)| ≤
      M.u * |p + b j| + |p - P| := by
    have h1 := abs_sub_le (M.add p (b j)) (p + b j) (P + b j)
    have h2 : |p + b j - (P + b j)| = |p - P| := by
      rw [show p + b j - (P + b j) = p - P from by ring]
    linarith [hadd]
  have hpbj : |p + b j| ≤ (S + br * S) + |b j| :=
    (abs_add_le p (b j)).trans (by linarith [hpabs])
  calc |M.add p (b j) - (P + b j)|
      ≤ M.u * |p + b j| + |p - P| := htri
    _ ≤ M.u * ((S + br * S) + |b j|) + br * S := by
        have hm := mul_le_mul_of_nonneg_left hpbj hu
        linarith [hm, hD]
    _ = M.u * (S + |b j|) + (1 + M.u) * (br * S) := by ring

-- ════════════════════════════════════════════════════════════════
-- § Dense layer: rounded-at-perturbed-input vs real-at-real-input
-- ════════════════════════════════════════════════════════════════

/-- The per-coordinate error budget of one rounded dense layer evaluated at an
    input within `e` of the real activation `xa`: the layer's own rounding
    (compounded `(1+u)^(m+2) − 1`, on magnitudes inflated by `e`) plus the
    Lipschitz pass-through `(Σᵢ|Wᵢⱼ|)·e` of the inherited error. `e = 0`
    specializes to the fresh-input bound. -/
noncomputable def denseErr {m n : Nat} (W : Mat m n) (b : Vec n) (xa : Vec m)
    (e : ℝ) (j : Fin n) : ℝ :=
  ((1 + M.u) ^ (m + 2) - 1) * ((∑ i, |W i j| * (|xa i| + e)) + |b j|)
    + (∑ i, |W i j|) * e

/-- **Rounded dense layer forward error, with inherited input error.**
    If `xt` is within `e` of the real activation `xa` coordinatewise, then
    `|M.dense W b xt j − dense W b xa j| ≤ denseErr W b xa e j`. -/
theorem dense_close {m n : Nat} (W : Mat m n) (b : Vec n) (xt xa : Vec m)
    (e : ℝ) (he : 0 ≤ e) (hx : ∀ i, |xt i - xa i| ≤ e) (j : Fin n) :
    |M.dense W b xt j - Proofs.dense W b xa j| ≤ M.denseErr W b xa e j := by
  have hdot : |M.dot xt (fun i => W i j) - ∑ i, xt i * W i j| ≤
      ((1 + M.u) ^ (m + 1) - 1) * ∑ i, |xt i * W i j| := by
    simpa using M.dot_close xt (fun i => W i j)
  have hSxtSE : (∑ i, |xt i * W i j|) ≤ ∑ i, |W i j| * (|xa i| + e) := by
    refine Finset.sum_le_sum fun i _ => ?_
    have h1 : |xt i| ≤ |xa i| + e := by
      have h2 : |xt i| ≤ |xt i - xa i| + |xa i| := by
        simpa using abs_sub_le (xt i) (xa i) 0
      have h3 := hx i
      linarith
    calc |xt i * W i j| = |xt i| * |W i j| := abs_mul _ _
      _ ≤ (|xa i| + e) * |W i j| :=
          mul_le_mul_of_nonneg_right h1 (abs_nonneg _)
      _ = |W i j| * (|xa i| + e) := by ring
  have hlip : |(∑ i, xt i * W i j) - ∑ i, xa i * W i j| ≤
      (∑ i, |W i j|) * e := by
    have h1 : (∑ i, xt i * W i j) - ∑ i, xa i * W i j
        = ∑ i, (xt i - xa i) * W i j := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    rw [h1]
    calc |∑ i, (xt i - xa i) * W i j| ≤ ∑ i, |(xt i - xa i) * W i j| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i, |W i j| * e := by
          refine Finset.sum_le_sum fun i _ => ?_
          calc |(xt i - xa i) * W i j| = |xt i - xa i| * |W i j| := abs_mul _ _
            _ ≤ e * |W i j| := mul_le_mul (hx i) le_rfl (abs_nonneg _) he
            _ = |W i j| * e := by ring
      _ = (∑ i, |W i j|) * e := by rw [Finset.sum_mul]
  have hgoal : M.denseErr W b xa e j =
      ((1 + M.u) ^ (m + 1) * (1 + M.u) - 1)
          * ((∑ i, |W i j| * (|xa i| + e)) + |b j|)
        + (∑ i, |W i j|) * e := by
    simp only [FloatModel.denseErr]
    rw [show ((1 : ℝ) + M.u) ^ (m + 2) = (1 + M.u) ^ (m + 1) * (1 + M.u)
        from pow_succ _ _]
  rw [show Proofs.dense W b xa j = (∑ i, xa i * W i j) + b j from rfl, hgoal]
  exact dense_step_bound M.u_nonneg (M.one_le_pow_one_add_u (m + 1))
    hdot (Finset.abs_sum_le_sum_abs _ _) hSxtSE hlip (M.err _)

/-- `dense_close` at a fresh (unperturbed) input — the `e = 0` face. -/
theorem dense_close_fresh {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (j : Fin n) :
    |M.dense W b x j - Proofs.dense W b x j| ≤ M.denseErr W b x 0 j :=
  M.dense_close W b x x 0 le_rfl (fun i => by simp) j

-- ════════════════════════════════════════════════════════════════
-- § ReLU: exact in floating point, 1-Lipschitz on inherited error
-- ════════════════════════════════════════════════════════════════

/-- **ReLU propagates error without amplification.** No rounding term: max
    with 0 is exact in floating point, so the float net applies `relu` bare
    and the bridge only needs 1-Lipschitz-ness. -/
theorem relu_close {n : Nat} (xt xa : Vec n) (e : ℝ)
    (hx : ∀ i, |xt i - xa i| ≤ e) (i : Fin n) :
    |relu n xt i - relu n xa i| ≤ e := by
  have h := hx i
  have h1 := abs_le.mp h
  have he0 : 0 ≤ e := le_trans (abs_nonneg _) h
  simp only [relu]
  by_cases ht : xt i > 0
  · by_cases ha : xa i > 0
    · simpa [ht, ha] using h
    · rw [if_pos ht, if_neg ha, sub_zero, abs_of_pos ht]
      rw [not_lt] at ha
      linarith [h1.2]
  · by_cases ha : xa i > 0
    · rw [if_neg ht, if_pos ha, zero_sub, abs_neg, abs_of_pos ha]
      rw [not_lt] at ht
      linarith [h1.1]
    · rw [if_neg ht, if_neg ha]
      simpa using he0

-- ════════════════════════════════════════════════════════════════
-- § Capstones: the Tier-1 nets
-- ════════════════════════════════════════════════════════════════

/-- **Linear-net forward extraction (Chapter 1).** The rounded `mnistLinear`
    is within the explicit `denseErr` budget of the real one, per logit. With
    `u = 2⁻²⁴` this is the binary32 forward-error bound for the certified
    linear classifier. -/
theorem linear_float_close {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (j : Fin n) :
    |M.dense W b x j - mnistLinear W b x j| ≤ M.denseErr W b x 0 j :=
  M.dense_close_fresh W b x j

/-- **MLP forward extraction (Chapter 2).** The rounded 3-layer MLP is within
    the layer-2 `denseErr` budget (at inherited error `e₁`) of the real MLP —
    the same `dense/relu` composition whose train step is certified in
    `MlpTrainStep.lean`. The hypotheses `h₀`/`h₁` uniformize the per-coordinate
    layer-0/1 budgets into `e₀`/`e₁`; at any concrete net they are discharged
    by finite max over the `d₁` (resp. `d₂`) coordinates. -/
theorem mlp_float_close {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (e₀ e₁ : ℝ)
    (he₀ : 0 ≤ e₀) (he₁ : 0 ≤ e₁)
    (h₀ : ∀ j, M.denseErr W₀ b₀ x 0 j ≤ e₀)
    (h₁ : ∀ j, M.denseErr W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) e₀ j ≤ e₁)
    (k : Fin d₃) :
    |M.mlpF W₀ b₀ W₁ b₁ W₂ b₂ x k -
        Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁
          (relu d₁ (Proofs.dense W₀ b₀ x)))) k| ≤
      M.denseErr W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁
        (relu d₁ (Proofs.dense W₀ b₀ x)))) e₁ k := by
  -- layer 0, fresh input
  have l0 : ∀ j, |M.dense W₀ b₀ x j - Proofs.dense W₀ b₀ x j| ≤ e₀ :=
    fun j => (M.dense_close_fresh W₀ b₀ x j).trans (h₀ j)
  -- relu: exact, 1-Lipschitz
  have r0 : ∀ j, |relu d₁ (M.dense W₀ b₀ x) j -
      relu d₁ (Proofs.dense W₀ b₀ x) j| ≤ e₀ :=
    fun j => relu_close _ _ e₀ l0 j
  -- layer 1, inherited error e₀
  have l1 : ∀ j, |M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)) j -
      Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) j| ≤ e₁ :=
    fun j => (M.dense_close W₁ b₁ _ _ e₀ he₀ r0 j).trans (h₁ j)
  have r1 : ∀ j, |relu d₂ (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))) j -
      relu d₂ (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))) j| ≤ e₁ :=
    fun j => relu_close _ _ e₁ l1 j
  -- layer 2, inherited error e₁
  exact M.dense_close W₂ b₂ _ _ e₁ he₁ r1 k

-- ════════════════════════════════════════════════════════════════
-- § The γ-form: rational budgets for numeric instantiation
-- ════════════════════════════════════════════════════════════════

/-- `(1 − k·u)·(1+u)^k ≤ 1`, unconditionally (for `k·u ≥ 1` the left side
    is `≤ 0`). The division-free product form of the classical `γₖ` bound,
    so the induction needs no `k·u < 1` bookkeeping. -/
private theorem one_sub_mul_pow_le (u : ℝ) (hu : 0 ≤ u) (k : ℕ) :
    (1 - (k : ℝ) * u) * (1 + u) ^ k ≤ 1 := by
  induction k with
  | zero => simp
  | succ k ih =>
    have hp : (0 : ℝ) ≤ (1 + u) ^ k := pow_nonneg (by linarith) k
    have hs : (1 + u) ^ (k + 1) = (1 + u) ^ k * (1 + u) := pow_succ _ _
    have key : (1 - ((k : ℝ) + 1) * u) * (1 + u) ≤ 1 - (k : ℝ) * u := by
      nlinarith [mul_nonneg (mul_nonneg
        (add_nonneg (Nat.cast_nonneg (α := ℝ) k) zero_le_one) hu) hu]
    push_cast
    calc (1 - ((k : ℝ) + 1) * u) * (1 + u) ^ (k + 1)
        = ((1 - ((k : ℝ) + 1) * u) * (1 + u)) * (1 + u) ^ k := by rw [hs]; ring
      _ ≤ (1 - (k : ℝ) * u) * (1 + u) ^ k := mul_le_mul_of_nonneg_right key hp
      _ ≤ 1 := ih

/-- **The classical `γₖ` bound**: for `k·u < 1`,
    `(1+u)^k − 1 ≤ k·u/(1 − k·u)`. Turns the compounded budgets into plain
    rational arithmetic at a concrete `u` (e.g. `u32`) — `norm_num` country,
    no big-power evaluation. -/
theorem pow_gamma_bound (u : ℝ) (hu : 0 ≤ u) (k : ℕ)
    (hk : (k : ℝ) * u < 1) :
    (1 + u) ^ k - 1 ≤ (k : ℝ) * u / (1 - (k : ℝ) * u) := by
  have hpos : 0 < 1 - (k : ℝ) * u := by linarith
  have h0 := one_sub_mul_pow_le u hu k
  have h1 : (1 + u) ^ k ≤ 1 / (1 - (k : ℝ) * u) := by
    rw [le_div_iff₀ hpos]
    linarith [mul_comm ((1 + u) ^ k) (1 - (k : ℝ) * u)]
  have h2 : 1 / (1 - (k : ℝ) * u) - 1 = (k : ℝ) * u / (1 - (k : ℝ) * u) := by
    field_simp
    ring
  linarith

/-- `x ↦ x/(1−x)` is monotone on `[0, 1)` — lets a `u ≤ u32` hypothesis ride
    through the γ-form. -/
private theorem div_one_sub_mono {x y : ℝ} (hxy : x ≤ y)
    (hy : y < 1) : x / (1 - x) ≤ y / (1 - y) := by
  have h1 : 0 < 1 - x := by linarith
  have h2 : 0 < 1 - y := by linarith
  rw [div_le_div_iff₀ h1 h2]
  nlinarith

-- ════════════════════════════════════════════════════════════════
-- § Uniform-magnitude budgets (closed forms in dims and norms)
-- ════════════════════════════════════════════════════════════════

/-- Worst-case output magnitude of one real layer under uniform magnitude
    bounds: `|denseⱼ| ≤ m·w·A + β` (and `relu` only shrinks). -/
noncomputable def layerAct (m : ℕ) (w β A : ℝ) : ℝ := (m : ℝ) * w * A + β

/-- The uniform-magnitude form of `denseErr`: every `|Wᵢⱼ| ≤ w`, `|bⱼ| ≤ β`,
    real activation magnitude `≤ A`, inherited error `≤ E`. -/
noncomputable def layerBudget (u : ℝ) (m : ℕ) (w β A E : ℝ) : ℝ :=
  ((1 + u) ^ (m + 2) - 1) * ((m : ℝ) * w * (A + E) + β) + (m : ℝ) * w * E

theorem layerAct_nonneg {m : ℕ} {w β A : ℝ} (hw : 0 ≤ w) (hβ : 0 ≤ β)
    (hA : 0 ≤ A) : 0 ≤ layerAct m w β A :=
  add_nonneg (mul_nonneg (mul_nonneg (Nat.cast_nonneg m) hw) hA) hβ

theorem layerBudget_nonneg {u : ℝ} {m : ℕ} {w β A E : ℝ} (hu : 0 ≤ u)
    (hw : 0 ≤ w) (hβ : 0 ≤ β) (hA : 0 ≤ A) (hE : 0 ≤ E) :
    0 ≤ layerBudget u m w β A E := by
  have hG : (0 : ℝ) ≤ (1 + u) ^ (m + 2) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hmw : (0 : ℝ) ≤ (m : ℝ) * w := mul_nonneg (Nat.cast_nonneg m) hw
  exact add_nonneg
    (mul_nonneg hG (add_nonneg (mul_nonneg hmw (add_nonneg hA hE)) hβ))
    (mul_nonneg hmw hE)

/-- Replacing the power term and the inherited error in `layerBudget` by
    upper bounds gives an upper bound — the monotonicity step the numeric
    instantiations chain through. -/
private theorem layerBudget_le_of {u : ℝ} {m : ℕ} {w β A E g Ē : ℝ}
    (hu : 0 ≤ u) (hw : 0 ≤ w) (hβ : 0 ≤ β) (hA : 0 ≤ A)
    (hG : (1 + u) ^ (m + 2) - 1 ≤ g) (hE0 : 0 ≤ E) (hE : E ≤ Ē) :
    layerBudget u m w β A E ≤ g * ((m : ℝ) * w * (A + Ē) + β) + (m : ℝ) * w * Ē := by
  have hG0 : (0 : ℝ) ≤ (1 + u) ^ (m + 2) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hmw : (0 : ℝ) ≤ (m : ℝ) * w := mul_nonneg (Nat.cast_nonneg m) hw
  have hAE : (m : ℝ) * w * (A + E) ≤ (m : ℝ) * w * (A + Ē) :=
    mul_le_mul_of_nonneg_left (by linarith) hmw
  have hX0 : (0 : ℝ) ≤ (m : ℝ) * w * (A + E) + β :=
    add_nonneg (mul_nonneg hmw (add_nonneg hA hE0)) hβ
  have h1 : ((1 + u) ^ (m + 2) - 1) * ((m : ℝ) * w * (A + E) + β)
      ≤ g * ((m : ℝ) * w * (A + Ē) + β) :=
    mul_le_mul hG (by linarith) hX0 (hG0.trans hG)
  have h2 : (m : ℝ) * w * E ≤ (m : ℝ) * w * Ē :=
    mul_le_mul_of_nonneg_left hE hmw
  exact add_le_add h1 h2

/-- ReLU never grows magnitudes. -/
theorem relu_abs_le {n : ℕ} (z : Vec n) (i : Fin n) :
    |relu n z i| ≤ |z i| := by
  simp only [relu]
  by_cases h : z i > 0
  · simp [h]
  · simp [h]

/-- Real dense-layer magnitude bound: `|denseⱼ| ≤ layerAct m w β a`. -/
theorem dense_abs_le {m n : ℕ} {W : Mat m n} {b : Vec n} {x : Vec m}
    {w β a : ℝ} (ha : 0 ≤ a)
    (hW : ∀ i j, |W i j| ≤ w) (hb : ∀ j, |b j| ≤ β) (hx : ∀ i, |x i| ≤ a)
    (j : Fin n) : |Proofs.dense W b x j| ≤ layerAct m w β a := by
  have h1 : |∑ i, x i * W i j| ≤ (m : ℝ) * w * a := by
    calc |∑ i, x i * W i j| ≤ ∑ i, |x i * W i j| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _i : Fin m, a * w := by
          refine Finset.sum_le_sum fun i _ => ?_
          rw [abs_mul]
          exact mul_le_mul (hx i) (hW i j) (abs_nonneg _) ha
      _ = (m : ℝ) * (a * w) := by rw [Finset.sum_const]; simp [nsmul_eq_mul]
      _ = (m : ℝ) * w * a := by ring
  calc |Proofs.dense W b x j| = |(∑ i, x i * W i j) + b j| := rfl
    _ ≤ |∑ i, x i * W i j| + |b j| := abs_add_le _ _
    _ ≤ (m : ℝ) * w * a + β := add_le_add h1 (hb j)
    _ = layerAct m w β a := rfl

/-- `denseErr` under uniform magnitude bounds is at most the closed-form
    `layerBudget`. -/
theorem denseErr_le_uniform {m n : ℕ} {W : Mat m n} {b : Vec n} {xa : Vec m}
    {w β a e : ℝ} (hw : 0 ≤ w) (he : 0 ≤ e)
    (hW : ∀ i j, |W i j| ≤ w) (hb : ∀ j, |b j| ≤ β) (hxa : ∀ i, |xa i| ≤ a)
    (j : Fin n) :
    M.denseErr W b xa e j ≤ layerBudget M.u m w β a e := by
  have hG : (0 : ℝ) ≤ (1 + M.u) ^ (m + 2) - 1 :=
    sub_nonneg.mpr (M.one_le_pow_one_add_u (m + 2))
  have hsum1 : (∑ i, |W i j| * (|xa i| + e)) ≤ (m : ℝ) * w * (a + e) := by
    calc (∑ i, |W i j| * (|xa i| + e)) ≤ ∑ _i : Fin m, w * (a + e) := by
          refine Finset.sum_le_sum fun i _ => ?_
          exact mul_le_mul (hW i j) (by linarith [hxa i])
            (add_nonneg (abs_nonneg _) he) hw
      _ = (m : ℝ) * (w * (a + e)) := by rw [Finset.sum_const]; simp [nsmul_eq_mul]
      _ = (m : ℝ) * w * (a + e) := by ring
  have hsum2 : (∑ i, |W i j|) ≤ (m : ℝ) * w := by
    calc (∑ i, |W i j|) ≤ ∑ _i : Fin m, w :=
          Finset.sum_le_sum fun i _ => hW i j
      _ = (m : ℝ) * w := by rw [Finset.sum_const]; simp [nsmul_eq_mul]
  have hmono1 : ((1 + M.u) ^ (m + 2) - 1) * ((∑ i, |W i j| * (|xa i| + e)) + |b j|)
      ≤ ((1 + M.u) ^ (m + 2) - 1) * ((m : ℝ) * w * (a + e) + β) :=
    mul_le_mul_of_nonneg_left (add_le_add hsum1 (hb j)) hG
  have hmono2 : (∑ i, |W i j|) * e ≤ (m : ℝ) * w * e :=
    mul_le_mul_of_nonneg_right hsum2 he
  show ((1 + M.u) ^ (m + 2) - 1) * ((∑ i, |W i j| * (|xa i| + e)) + |b j|)
      + (∑ i, |W i j|) * e
    ≤ ((1 + M.u) ^ (m + 2) - 1) * ((m : ℝ) * w * (a + e) + β) + (m : ℝ) * w * e
  linarith

/-- **MLP forward extraction, uniform-magnitude budgets.** `mlp_float_close`
    with the `e₀`/`e₁` uniformization discharged once and for all from
    coordinatewise magnitude bounds `|Wᵢ| ≤ wᵢ`, `|bᵢ| ≤ βᵢ`, `|x| ≤ a`.
    The budget is a closed form in the dims and magnitudes — evaluable by
    `norm_num` at a concrete net. -/
theorem mlp_float_close_uniform {d₀ d₁ d₂ d₃ : Nat}
    {W₀ : Mat d₀ d₁} {b₀ : Vec d₁} {W₁ : Mat d₁ d₂} {b₁ : Vec d₂}
    {W₂ : Mat d₂ d₃} {b₂ : Vec d₃} {x : Vec d₀}
    {w₀ β₀ w₁ β₁ w₂ β₂ a : ℝ}
    (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁)
    (hw₂ : 0 ≤ w₂) (ha : 0 ≤ a)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w₂) (hb₂ : ∀ j, |b₂ j| ≤ β₂)
    (hx : ∀ i, |x i| ≤ a) (k : Fin d₃) :
    |M.mlpF W₀ b₀ W₁ b₁ W₂ b₂ x k -
        Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁
          (relu d₁ (Proofs.dense W₀ b₀ x)))) k| ≤
      layerBudget M.u d₂ w₂ β₂ (layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a))
        (layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
          (layerBudget M.u d₀ w₀ β₀ a 0)) := by
  have hA₁0 : 0 ≤ layerAct d₀ w₀ β₀ a := layerAct_nonneg hw₀ hβ₀ ha
  have hA₂0 : 0 ≤ layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) :=
    layerAct_nonneg hw₁ hβ₁ hA₁0
  have hE₀0 : 0 ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha le_rfl
  have hE₁0 : 0 ≤ layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
      (layerBudget M.u d₀ w₀ β₀ a 0) :=
    layerBudget_nonneg M.u_nonneg hw₁ hβ₁ hA₁0 hE₀0
  -- real activation magnitude bounds, layer by layer
  have ha₁ : ∀ i, |relu d₁ (Proofs.dense W₀ b₀ x) i| ≤ layerAct d₀ w₀ β₀ a :=
    fun i => (relu_abs_le _ i).trans (dense_abs_le ha hW₀ hb₀ hx i)
  have ha₂ : ∀ i, |relu d₂ (Proofs.dense W₁ b₁
      (relu d₁ (Proofs.dense W₀ b₀ x))) i| ≤
      layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) :=
    fun i => (relu_abs_le _ i).trans (dense_abs_le hA₁0 hW₁ hb₁ ha₁ i)
  refine (M.mlp_float_close W₀ b₀ W₁ b₁ W₂ b₂ x _ _ hE₀0 hE₁0
    (fun j => M.denseErr_le_uniform hw₀ le_rfl hW₀ hb₀ hx j)
    (fun j => M.denseErr_le_uniform hw₁ hE₀0 hW₁ hb₁ ha₁ j) k).trans ?_
  exact M.denseErr_le_uniform hw₂ hE₁0 hW₂ hb₂ ha₂ k

-- ════════════════════════════════════════════════════════════════
-- § The committed-net numeric instance (784→512→512→10, binary32)
-- ════════════════════════════════════════════════════════════════

/-- γ-form at a concrete exponent and target, monotone through `u ≤ u32`. -/
theorem gamma_num (hMu : M.u ≤ u32) {k : ℕ} {q : ℝ}
    (hk : (k : ℝ) * u32 < 1)
    (hq : (k : ℝ) * u32 / (1 - (k : ℝ) * u32) ≤ q) :
    (1 + M.u) ^ k - 1 ≤ q := by
  have hu := M.u_nonneg
  have hkM : (k : ℝ) * M.u ≤ (k : ℝ) * u32 :=
    mul_le_mul_of_nonneg_left hMu (Nat.cast_nonneg k)
  exact ((pow_gamma_bound M.u hu k (lt_of_le_of_lt hkM hk)).trans
    (div_one_sub_mono hkM hk)).trans hq

/-- Layer-0 budget at the committed MNIST dims and *trained* magnitudes
    (`|W| ≤ 3/5`, covering the measured `max|W| = 0.52`): `E₀ ≤ 0.023`. -/
private theorem mnist_E0_le (hMu : M.u ≤ u32) :
    layerBudget M.u 784 (3/5) 1 1 0 ≤ 23/1000 := by
  refine (layerBudget_le_of M.u_nonneg (by norm_num) (by norm_num)
    (by norm_num) (M.gamma_num (q := 47/1000000) hMu (by norm_num [u32])
      (by norm_num [u32])) le_rfl le_rfl).trans ?_
  norm_num

private theorem mnist_E0_nonneg : (0:ℝ) ≤ layerBudget M.u 784 (3/5) 1 1 0 :=
  layerBudget_nonneg M.u_nonneg (by norm_num) (by norm_num) (by norm_num)
    le_rfl

/-- Layer-1 budget at the committed MNIST dims and trained magnitudes:
    `E₁ ≤ 12`. -/
private theorem mnist_E1_le (hMu : M.u ≤ u32) :
    layerBudget M.u 512 (3/5) 1 (2357/5)
      (layerBudget M.u 784 (3/5) 1 1 0) ≤ 12 := by
  refine (layerBudget_le_of M.u_nonneg (by norm_num) (by norm_num)
    (by norm_num) (M.gamma_num (q := 31/1000000) hMu (by norm_num [u32])
      (by norm_num [u32])) M.mnist_E0_nonneg (M.mnist_E0_le hMu)).trans ?_
  norm_num

private theorem mnist_E1_nonneg : (0:ℝ) ≤ layerBudget M.u 512 (3/5) 1 (2357/5)
    (layerBudget M.u 784 (3/5) 1 1 0) :=
  layerBudget_nonneg M.u_nonneg (by norm_num) (by norm_num) (by norm_num)
    M.mnist_E0_nonneg

/-- **Numeric capstone at the committed MNIST-MLP dims and TRAINED
    magnitudes** (the `MainMnistMlpVerified.lean` net: 784→512→512→10;
    `|W| ≤ 3/5` covers the measured `max|W| = 0.52` of a real 12-epoch
    97.8% run — He init already exceeds the prettier `1/32` in its tails).
    For any rounding model at binary32 accuracy (`u ≤ 2⁻²⁴`), every rounded
    logit is within **5100** of the exact-real logit — the worst-case logit
    magnitude at these bounds is ≈4.5·10⁷, so ≈10⁻⁴ *relative*, the same
    relative scale as at small weights. All three layer budgets discharge by
    `norm_num` through the γ-form; no big-power evaluation.

    Measured on the live run (`scripts/margin_probe.py`): actual logit
    drift ≤ 1.6·10⁻⁵ — the ≈3·10⁸ gap between the worst-case bound and
    reality is the worst-case-composition blow-up (`307·e` Lipschitz
    amplification per layer at these magnitudes), the quantitative case for
    a-posteriori certificates past toy depth. -/
theorem mnist_mlp_float_budget (hMu : M.u ≤ u32)
    (W₀ : Mat 784 512) (b₀ : Vec 512) (W₁ : Mat 512 512) (b₁ : Vec 512)
    (W₂ : Mat 512 10) (b₂ : Vec 10) (x : Vec 784)
    (hW₀ : ∀ i j, |W₀ i j| ≤ 3/5) (hb₀ : ∀ j, |b₀ j| ≤ 1)
    (hW₁ : ∀ i j, |W₁ i j| ≤ 3/5) (hb₁ : ∀ j, |b₁ j| ≤ 1)
    (hW₂ : ∀ i j, |W₂ i j| ≤ 3/5) (hb₂ : ∀ j, |b₂ j| ≤ 1)
    (hx : ∀ i, |x i| ≤ 1) (k : Fin 10) :
    |M.mlpF W₀ b₀ W₁ b₁ W₂ b₂ x k -
        Proofs.dense W₂ b₂ (relu 512 (Proofs.dense W₁ b₁
          (relu 512 (Proofs.dense W₀ b₀ x)))) k| ≤ 5100 := by
  have hu := M.u_nonneg
  have hB₂ : layerBudget M.u 512 (3/5) 1 (3620377/25)
      (layerBudget M.u 512 (3/5) 1 (2357/5)
        (layerBudget M.u 784 (3/5) 1 1 0)) ≤ 5100 := by
    refine (layerBudget_le_of hu (by norm_num) (by norm_num) (by norm_num)
      (M.gamma_num (q := 31/1000000) hMu (by norm_num [u32])
        (by norm_num [u32])) M.mnist_E1_nonneg (M.mnist_E1_le hMu)).trans ?_
    norm_num
  -- assemble: the uniform capstone, activation constants evaluated
  have hmain := M.mlp_float_close_uniform
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
    hW₀ hb₀ hW₁ hb₁ hW₂ hb₂ hx k
  rw [show layerAct 784 (3/5) 1 1 = (2357/5 : ℝ) by norm_num [layerAct],
      show layerAct 512 (3/5) 1 (2357/5) = (3620377/25 : ℝ) by
        norm_num [layerAct]]
    at hmain
  exact hmain.trans hB₂

-- ════════════════════════════════════════════════════════════════
-- § Backward ops: rounded product, SGD update, ReLU mask
-- ════════════════════════════════════════════════════════════════

/-- Rounded subtraction: `fl(x − y)`. -/
noncomputable def sub (x y : ℝ) : ℝ := M.rnd (x - y)

/-- Budget of one rounded product of two inherited-error operands:
    `|fl(xt·yt) − x·y|` with `|xt−x| ≤ ea`, `|yt−y| ≤ ec`, `|x| ≤ A`, `|y| ≤ C`. -/
noncomputable def mulErr (u A C ea ec : ℝ) : ℝ :=
  u * ((A + ea) * (C + ec)) + (A * ec + ea * C + ea * ec)

/-- Budget of one rounded SGD update `fl(θ − fl(lr·gt))` against `θ − lr·g`,
    with `|gt−g| ≤ eg`, `|g| ≤ G`. -/
noncomputable def sgdErr (u lr Θ G eg : ℝ) : ℝ :=
  u * (Θ + (1 + u) * (lr * (G + eg))) + (u * (lr * (G + eg)) + lr * eg)

/-- **Rounded product with inherited operand errors.** -/
theorem mul_close {xt x yt y ea ec A C : ℝ}
    (hx : |xt - x| ≤ ea) (hy : |yt - y| ≤ ec)
    (hA : |x| ≤ A) (hC : |y| ≤ C) :
    |M.mul xt yt - x * y| ≤ mulErr M.u A C ea ec := by
  have hu := M.u_nonneg
  have hea0 : 0 ≤ ea := (abs_nonneg _).trans hx
  have hec0 : 0 ≤ ec := (abs_nonneg _).trans hy
  have hA0 : 0 ≤ A := (abs_nonneg _).trans hA
  have hC0 : 0 ≤ C := (abs_nonneg _).trans hC
  have hxt : |xt| ≤ A + ea := by
    have h := abs_sub_le xt x 0
    simp only [sub_zero] at h
    linarith
  have hyt : |yt| ≤ C + ec := by
    have h := abs_sub_le yt y 0
    simp only [sub_zero] at h
    linarith
  have hprod : |xt * yt - x * y| ≤ A * ec + ea * C + ea * ec := by
    have h1 : xt * yt - x * y = xt * (yt - y) + y * (xt - x) := by ring
    have h2 : |xt| * |yt - y| ≤ (A + ea) * ec :=
      mul_le_mul hxt hy (abs_nonneg _) (by linarith)
    have h3 : |y| * |xt - x| ≤ C * ea := mul_le_mul hC hx (abs_nonneg _) hC0
    calc |xt * yt - x * y| = |xt * (yt - y) + y * (xt - x)| := by rw [h1]
      _ ≤ |xt * (yt - y)| + |y * (xt - x)| := abs_add_le _ _
      _ = |xt| * |yt - y| + |y| * |xt - x| := by rw [abs_mul, abs_mul]
      _ ≤ A * ec + ea * C + ea * ec := by nlinarith
  have habs : |xt * yt| ≤ (A + ea) * (C + ec) := by
    rw [abs_mul]
    exact mul_le_mul hxt hyt (abs_nonneg _) (by linarith)
  have hrnd : |M.mul xt yt - xt * yt| ≤ M.u * |xt * yt| := M.err _
  have htri : |M.mul xt yt - x * y| ≤
      |M.mul xt yt - xt * yt| + |xt * yt - x * y| := abs_sub_le _ _ _
  have h2 : M.u * |xt * yt| ≤ M.u * ((A + ea) * (C + ec)) :=
    mul_le_mul_of_nonneg_left habs hu
  show |M.mul xt yt - x * y| ≤
    M.u * ((A + ea) * (C + ec)) + (A * ec + ea * C + ea * ec)
  linarith

/-- **Rounded SGD update**: `fl(θ − fl(lr·gt))` is within `sgdErr` of the
    real step `θ − lr·g`. Two roundings plus the inherited gradient error. -/
theorem sgd_step_close (θ : ℝ) {gt g lr G eg : ℝ}
    (hg : |gt - g| ≤ eg) (hG : |g| ≤ G) (hlr : 0 ≤ lr) :
    |M.sub θ (M.mul lr gt) - (θ - lr * g)| ≤ sgdErr M.u lr |θ| G eg := by
  have hu := M.u_nonneg
  have heg0 : 0 ≤ eg := (abs_nonneg _).trans hg
  have hG0 : 0 ≤ G := (abs_nonneg _).trans hG
  have hgt : |gt| ≤ G + eg := by
    have h := abs_sub_le gt g 0
    simp only [sub_zero] at h
    linarith
  have hlrg : |lr * gt| ≤ lr * (G + eg) := by
    rw [abs_mul, abs_of_nonneg hlr]
    exact mul_le_mul_of_nonneg_left hgt hlr
  have hp1 : |M.mul lr gt - lr * gt| ≤ M.u * |lr * gt| := M.err _
  have hp2 : |lr * gt - lr * g| ≤ lr * eg := by
    rw [show lr * gt - lr * g = lr * (gt - g) from by ring, abs_mul,
        abs_of_nonneg hlr]
    exact mul_le_mul_of_nonneg_left hg hlr
  have hmono := mul_le_mul_of_nonneg_left hlrg hu
  have hpclose : |M.mul lr gt - lr * g| ≤ M.u * (lr * (G + eg)) + lr * eg := by
    have htri := abs_sub_le (M.mul lr gt) (lr * gt) (lr * g)
    linarith
  have hpabs : |M.mul lr gt| ≤ (1 + M.u) * (lr * (G + eg)) := by
    have htri : |M.mul lr gt| ≤ |M.mul lr gt - lr * gt| + |lr * gt| := by
      have h := abs_sub_le (M.mul lr gt) (lr * gt) 0
      simp only [sub_zero] at h
      linarith
    nlinarith
  have hsub : |M.sub θ (M.mul lr gt) - (θ - M.mul lr gt)| ≤
      M.u * |θ - M.mul lr gt| := M.err _
  have hθp : |θ - M.mul lr gt| ≤ |θ| + (1 + M.u) * (lr * (G + eg)) := by
    have h := abs_sub_le θ 0 (M.mul lr gt)
    simp only [sub_zero, zero_sub, abs_neg] at h
    linarith
  have h3 : |(θ - M.mul lr gt) - (θ - lr * g)| = |M.mul lr gt - lr * g| := by
    rw [show (θ - M.mul lr gt) - (θ - lr * g) = -(M.mul lr gt - lr * g) from
        by ring, abs_neg]
  have htri2 : |M.sub θ (M.mul lr gt) - (θ - lr * g)| ≤
      |M.sub θ (M.mul lr gt) - (θ - M.mul lr gt)| +
        |(θ - M.mul lr gt) - (θ - lr * g)| := abs_sub_le _ _ _
  have h4 := mul_le_mul_of_nonneg_left hθp hu
  show |M.sub θ (M.mul lr gt) - (θ - lr * g)| ≤
    M.u * (|θ| + (1 + M.u) * (lr * (G + eg)))
      + (M.u * (lr * (G + eg)) + lr * eg)
  linarith [htri2, hsub, h3, hpclose, h4]

private theorem mulErr_nonneg {u A C ea ec : ℝ} (hu : 0 ≤ u) (hA : 0 ≤ A)
    (hC : 0 ≤ C) (hea : 0 ≤ ea) (hec : 0 ≤ ec) : 0 ≤ mulErr u A C ea ec :=
  add_nonneg
    (mul_nonneg hu (mul_nonneg (by linarith) (by linarith)))
    (by nlinarith)

private theorem mulErr_mono {u u' A C ea ea' ec : ℝ}
    (hu : 0 ≤ u) (huu : u ≤ u') (hA : 0 ≤ A) (hC : 0 ≤ C)
    (hea0 : 0 ≤ ea) (hea : ea ≤ ea') (hec : 0 ≤ ec) :
    mulErr u A C ea ec ≤ mulErr u' A C ea' ec := by
  have h1 : (A + ea) * (C + ec) ≤ (A + ea') * (C + ec) :=
    mul_le_mul_of_nonneg_right (by linarith) (by linarith)
  have h10 : (0:ℝ) ≤ (A + ea) * (C + ec) :=
    mul_nonneg (by linarith) (by linarith)
  have t1 : u * ((A + ea) * (C + ec)) ≤ u' * ((A + ea') * (C + ec)) :=
    mul_le_mul huu h1 h10 (by linarith)
  have t2 : ea * C ≤ ea' * C := mul_le_mul_of_nonneg_right hea hC
  have t3 : ea * ec ≤ ea' * ec := mul_le_mul_of_nonneg_right hea hec
  exact add_le_add t1 (by linarith)

theorem sgdErr_mono {u u' lr Θ Θ' G eg eg' : ℝ}
    (hu : 0 ≤ u) (huu : u ≤ u') (hlr : 0 ≤ lr) (hΘ0 : 0 ≤ Θ) (hΘ : Θ ≤ Θ')
    (hG : 0 ≤ G) (heg0 : 0 ≤ eg) (heg : eg ≤ eg') :
    sgdErr u lr Θ G eg ≤ sgdErr u' lr Θ' G eg' := by
  have hin : lr * (G + eg) ≤ lr * (G + eg') :=
    mul_le_mul_of_nonneg_left (by linarith) hlr
  have hin0 : (0:ℝ) ≤ lr * (G + eg) := mul_nonneg hlr (by linarith)
  have h1u : (1 + u) * (lr * (G + eg)) ≤ (1 + u') * (lr * (G + eg')) :=
    mul_le_mul (by linarith) hin hin0 (by linarith)
  have hX0 : (0:ℝ) ≤ Θ + (1 + u) * (lr * (G + eg)) :=
    add_nonneg hΘ0 (mul_nonneg (by linarith) hin0)
  have t1 : u * (Θ + (1 + u) * (lr * (G + eg))) ≤
      u' * (Θ' + (1 + u') * (lr * (G + eg'))) :=
    mul_le_mul huu (by linarith) hX0 (by linarith)
  have t2 : u * (lr * (G + eg)) ≤ u' * (lr * (G + eg')) :=
    mul_le_mul huu hin hin0 (by linarith)
  have t3 : lr * eg ≤ lr * eg' := mul_le_mul_of_nonneg_left heg hlr
  exact add_le_add t1 (add_le_add t2 t3)

/-- ReLU backward mask — `if z > 0 then v else 0`. Compare + select: exact
    in floating point, so the float chain applies it bare (the rendered
    trainer's relu-back compare reads the rendered pre-activation, exactly
    the `zt` here). -/
noncomputable def reluMask {n : ℕ} (z v : Vec n) : Vec n :=
  fun i => if z i > 0 then v i else 0

theorem reluMask_abs_le {n : ℕ} (z v : Vec n) (i : Fin n) :
    |reluMask z v i| ≤ |v i| := by
  simp only [reluMask]
  by_cases h : z i > 0
  · simp [h]
  · simp [h]

/-- **The float-side kink condition.** If the pre-activation error `ez`
    cannot flip any sign — `ez < |zᵢ|`, a *quantitative margin*, the float
    analogue of the suite's `x k ≠ 0` off-the-kink hypotheses — then the
    float and real masks agree and the mask is 1-Lipschitz in the value. -/
theorem reluMask_close {n : ℕ} {zt z vt v : Vec n} {ez ev : ℝ}
    (hz : ∀ i, |zt i - z i| ≤ ez) (hm : ∀ i, ez < |z i|)
    (hv : ∀ i, |vt i - v i| ≤ ev) (hev : 0 ≤ ev) (i : Fin n) :
    |reluMask zt vt i - reluMask z v i| ≤ ev := by
  have hzi := abs_le.mp (hz i)
  have hmi := hm i
  simp only [reluMask]
  rcases lt_trichotomy (z i) 0 with hneg | hzero | hpos
  · have h1 : ¬ z i > 0 := by linarith
    have h2 : ¬ zt i > 0 := by
      rw [not_lt]
      rw [abs_of_neg hneg] at hmi
      linarith [hzi.2]
    rw [if_neg h1, if_neg h2]
    simpa using hev
  · exfalso
    rw [hzero] at hmi
    simp only [abs_zero] at hmi
    linarith [(abs_nonneg (zt i - z i)).trans (hz i)]
  · have h2 : zt i > 0 := by
      rw [abs_of_pos hpos] at hmi
      linarith [hzi.1]
    rw [if_pos hpos, if_pos h2]
    exact hv i

/-- **Cotangent through one layer** — `mask(z, Wᵀ·c)`, float vs real. The
    transposed matvec is `dense` with zero bias, so the dot machinery is
    reused wholesale; under the quantitative margin the mask passes the
    `layerBudget` through unchanged. -/
theorem cot_step_close {m n : ℕ} (W : Mat m n) (zt z : Vec m) (ct c : Vec n)
    {w C ec ez : ℝ} (hw : 0 ≤ w) (hC0 : 0 ≤ C) (hec : 0 ≤ ec)
    (hW : ∀ i j, |W i j| ≤ w) (hC : ∀ j, |c j| ≤ C)
    (hc : ∀ j, |ct j - c j| ≤ ec)
    (hz : ∀ i, |zt i - z i| ≤ ez) (hm : ∀ i, ez < |z i|) (i : Fin m) :
    |reluMask zt (M.dense (fun j i' => W i' j) (fun _ => 0) ct) i -
      reluMask z (Proofs.dense (fun j i' => W i' j) (fun _ => 0) c) i| ≤
      layerBudget M.u n w 0 C ec := by
  have hpre : ∀ i', |M.dense (fun j i' => W i' j) (fun _ => 0) ct i' -
      Proofs.dense (fun j i' => W i' j) (fun _ => 0) c i'| ≤
      layerBudget M.u n w 0 C ec := fun i' =>
    (M.dense_close _ _ ct c ec hec hc i').trans
      (M.denseErr_le_uniform hw hec (fun j i'' => hW i'' j)
        (fun _ => by simp) hC i')
  exact reluMask_close hz hm hpre
    (layerBudget_nonneg M.u_nonneg hw le_rfl hC0 hec) i

-- ════════════════════════════════════════════════════════════════
-- § Train-step capstones: rounded SGD entries vs the certified step
-- ════════════════════════════════════════════════════════════════

/-- **Rounded output-layer weight update (W₂).** The float update
    `fl(W₂ᵢⱼ − fl(lr·fl(ã₂ᵢ·gtⱼ)))` — outer-product gradient from the *stored
    float forward activation*, as the rendered trainer computes it — is
    within an explicit budget of the real step `W₂ᵢⱼ − lr·(a₂ᵢ·gⱼ)`. The real
    target is `Mat.outer a₂ g i j = emitWeightGrad`'s entry, the quantity
    `mlp_render_W2_certified` proves equal to the pdiv-Jacobian contraction —
    so this chains the float step to the certified gradient. Takes the output
    cotangent `gt ≈ g` as a hypothesis (the softmax−onehot head needs an `exp`
    accuracy axiom — future rung). -/
theorem mlp_w2_step_float_close {d₀ d₁ d₂ d₃ : Nat}
    {W₀ : Mat d₀ d₁} {b₀ : Vec d₁} {W₁ : Mat d₁ d₂} {b₁ : Vec d₂}
    (W₂ : Mat d₂ d₃) {x : Vec d₀} {gt g : Vec d₃} {lr : ℝ}
    {w₀ β₀ w₁ β₁ a G eg : ℝ}
    (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁)
    (ha : 0 ≤ a) (hlr : 0 ≤ lr)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hx : ∀ i, |x i| ≤ a)
    (hG : ∀ j, |g j| ≤ G) (hg : ∀ j, |gt j - g j| ≤ eg)
    (i : Fin d₂) (j : Fin d₃) :
    |M.sub (W₂ i j) (M.mul lr (M.mul
        (relu d₂ (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))) i) (gt j))) -
      (W₂ i j - lr * (relu d₂ (Proofs.dense W₁ b₁
        (relu d₁ (Proofs.dense W₀ b₀ x))) i * g j))| ≤
    sgdErr M.u lr |W₂ i j|
      (layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) * G)
      (mulErr M.u (layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)) G
        (layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
          (layerBudget M.u d₀ w₀ β₀ a 0)) eg) := by
  have hA₁0 : 0 ≤ layerAct d₀ w₀ β₀ a := layerAct_nonneg hw₀ hβ₀ ha
  have hA₂0 : 0 ≤ layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) :=
    layerAct_nonneg hw₁ hβ₁ hA₁0
  have hE₀0 : 0 ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha le_rfl
  -- float forward chain: ã₂ within E₁ of a₂
  have l0 : ∀ j', |M.dense W₀ b₀ x j' - Proofs.dense W₀ b₀ x j'| ≤
      layerBudget M.u d₀ w₀ β₀ a 0 := fun j' =>
    (M.dense_close_fresh W₀ b₀ x j').trans
      (M.denseErr_le_uniform hw₀ le_rfl hW₀ hb₀ hx j')
  have r0 : ∀ j', |relu d₁ (M.dense W₀ b₀ x) j' -
      relu d₁ (Proofs.dense W₀ b₀ x) j'| ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    fun j' => relu_close _ _ _ l0 j'
  have ha₁ : ∀ i', |relu d₁ (Proofs.dense W₀ b₀ x) i'| ≤
      layerAct d₀ w₀ β₀ a :=
    fun i' => (relu_abs_le _ i').trans (dense_abs_le ha hW₀ hb₀ hx i')
  have l1 : ∀ j', |M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)) j' -
      Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) j'| ≤
      layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) := fun j' =>
    (M.dense_close W₁ b₁ _ _ _ hE₀0 r0 j').trans
      (M.denseErr_le_uniform hw₁ hE₀0 hW₁ hb₁ ha₁ j')
  have r1 : ∀ j', |relu d₂ (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))) j' -
      relu d₂ (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))) j'| ≤
      layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) :=
    fun j' => relu_close _ _ _ l1 j'
  have ha₂ : |relu d₂ (Proofs.dense W₁ b₁
      (relu d₁ (Proofs.dense W₀ b₀ x))) i| ≤
      layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) :=
    (relu_abs_le _ i).trans (dense_abs_le hA₁0 hW₁ hb₁ ha₁ i)
  have hmul := M.mul_close (r1 i) (hg j) ha₂ (hG j)
  have hac : |relu d₂ (Proofs.dense W₁ b₁
      (relu d₁ (Proofs.dense W₀ b₀ x))) i * g j| ≤
      layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) * G := by
    rw [abs_mul]
    exact mul_le_mul ha₂ (hG j) (abs_nonneg _) hA₂0
  exact M.sgd_step_close (W₂ i j) hmul hac hlr

/-- **Rounded output-layer bias update (b₂)** — the bias gradient *is* the
    cotangent (`emitBiasGrad`), so this is `sgd_step_close` directly. -/
theorem mlp_b2_step_float_close {d₃ : Nat} (b₂ : Vec d₃) {gt g : Vec d₃}
    {lr G eg : ℝ} (hlr : 0 ≤ lr)
    (hG : ∀ j, |g j| ≤ G) (hg : ∀ j, |gt j - g j| ≤ eg) (j : Fin d₃) :
    |M.sub (b₂ j) (M.mul lr (gt j)) - (b₂ j - lr * g j)| ≤
      sgdErr M.u lr |b₂ j| G eg :=
  M.sgd_step_close (b₂ j) (hg j) (hG j) hlr

/-- **Rounded hidden-layer weight update (W₁), through the backward chain.**
    The float cotangent `ct₁ = mask(pt₁, W₂ᵀ·gt)` — computed from the rendered
    pre-activation and the rounded transposed matvec, exactly the structure
    of the rendered backward — is within `layerBudget` of the real
    `c₁ = mask(p₁, W₂ᵀ·g)` (the `mlpCotOut1` closed form), **given the
    quantitative margin** `E₁ < |p₁ᵢ|` at every layer-1 pre-activation: the
    forward rounding error must not flip a ReLU. Then the update is within
    `sgdErr` of the real `W₁ᵢⱼ − lr·(a₁ᵢ·c₁ⱼ)`, the quantity
    `mlp_render_W1_certified` certifies. `W₀`/`b₁`/`b₀` are the same
    instantiation one mask deeper. -/
theorem mlp_w1_step_float_close {d₀ d₁ d₂ d₃ : Nat}
    {W₀ : Mat d₀ d₁} {b₀ : Vec d₁} (W₁ : Mat d₁ d₂) {b₁ : Vec d₂}
    {W₂ : Mat d₂ d₃} {x : Vec d₀} {gt g : Vec d₃} {lr : ℝ}
    {w₀ β₀ w₁ β₁ w₂ a G eg : ℝ}
    (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁)
    (hw₂ : 0 ≤ w₂) (ha : 0 ≤ a) (hlr : 0 ≤ lr) (hG0 : 0 ≤ G) (heg : 0 ≤ eg)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w₂)
    (hx : ∀ i, |x i| ≤ a)
    (hG : ∀ j, |g j| ≤ G) (hg : ∀ j, |gt j - g j| ≤ eg)
    (hmargin : ∀ i', layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) <
      |Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) i'|)
    (i : Fin d₁) (j : Fin d₂) :
    |M.sub (W₁ i j) (M.mul lr (M.mul
        (relu d₁ (M.dense W₀ b₀ x) i)
        (reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
          (M.dense (fun j' i' => W₂ i' j') (fun _ => 0) gt) j))) -
      (W₁ i j - lr * (relu d₁ (Proofs.dense W₀ b₀ x) i *
        reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
          (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g) j))| ≤
    sgdErr M.u lr |W₁ i j|
      (layerAct d₀ w₀ β₀ a * layerAct d₃ w₂ 0 G)
      (mulErr M.u (layerAct d₀ w₀ β₀ a) (layerAct d₃ w₂ 0 G)
        (layerBudget M.u d₀ w₀ β₀ a 0)
        (layerBudget M.u d₃ w₂ 0 G eg)) := by
  have hA₁0 : 0 ≤ layerAct d₀ w₀ β₀ a := layerAct_nonneg hw₀ hβ₀ ha
  have hC₁0 : 0 ≤ layerAct d₃ w₂ 0 G := layerAct_nonneg hw₂ le_rfl hG0
  have hE₀0 : 0 ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha le_rfl
  -- float forward chain to the layer-1 pre-activation
  have l0 : ∀ j', |M.dense W₀ b₀ x j' - Proofs.dense W₀ b₀ x j'| ≤
      layerBudget M.u d₀ w₀ β₀ a 0 := fun j' =>
    (M.dense_close_fresh W₀ b₀ x j').trans
      (M.denseErr_le_uniform hw₀ le_rfl hW₀ hb₀ hx j')
  have r0 : ∀ j', |relu d₁ (M.dense W₀ b₀ x) j' -
      relu d₁ (Proofs.dense W₀ b₀ x) j'| ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    fun j' => relu_close _ _ _ l0 j'
  have ha₁ : ∀ i', |relu d₁ (Proofs.dense W₀ b₀ x) i'| ≤
      layerAct d₀ w₀ β₀ a :=
    fun i' => (relu_abs_le _ i').trans (dense_abs_le ha hW₀ hb₀ hx i')
  have l1 : ∀ j', |M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)) j' -
      Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) j'| ≤
      layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) := fun j' =>
    (M.dense_close W₁ b₁ _ _ _ hE₀0 r0 j').trans
      (M.denseErr_le_uniform hw₁ hE₀0 hW₁ hb₁ ha₁ j')
  -- the backward cotangent through the mask, under the margin
  have hcot : ∀ j', |reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
      (M.dense (fun j'' i' => W₂ i' j'') (fun _ => 0) gt) j' -
      reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
        (Proofs.dense (fun j'' i' => W₂ i' j'') (fun _ => 0) g) j'| ≤
      layerBudget M.u d₃ w₂ 0 G eg := fun j' =>
    M.cot_step_close W₂ _ _ gt g hw₂ hG0 heg hW₂ hG hg l1 hmargin j'
  -- the real cotangent magnitude
  have hc₁ : |reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
      (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g) j| ≤
      layerAct d₃ w₂ 0 G :=
    (reluMask_abs_le _ _ j).trans
      (dense_abs_le hG0 (fun j' i' => hW₂ i' j') (fun _ => by simp) hG j)
  have hmul := M.mul_close (r0 i) (hcot j) (ha₁ i) hc₁
  have hac : |relu d₁ (Proofs.dense W₀ b₀ x) i *
      reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
        (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g) j| ≤
      layerAct d₀ w₀ β₀ a * layerAct d₃ w₂ 0 G := by
    rw [abs_mul]
    exact mul_le_mul (ha₁ i) hc₁ (abs_nonneg _) hA₁0
  exact M.sgd_step_close (W₁ i j) hmul hac hlr

/-- **Rounded hidden bias update (b₁)** — the gradient is the layer-1
    cotangent itself (`emitBiasGrad`), so this is the cotangent chain
    followed by `sgd_step_close`. -/
theorem mlp_b1_step_float_close {d₀ d₁ d₂ d₃ : Nat}
    {W₀ : Mat d₀ d₁} {b₀ : Vec d₁} {W₁ : Mat d₁ d₂} (b₁ : Vec d₂)
    {W₂ : Mat d₂ d₃} {x : Vec d₀} {gt g : Vec d₃} {lr : ℝ}
    {w₀ β₀ w₁ β₁ w₂ a G eg : ℝ}
    (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁)
    (hw₂ : 0 ≤ w₂) (ha : 0 ≤ a) (hlr : 0 ≤ lr) (hG0 : 0 ≤ G) (heg : 0 ≤ eg)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w₂)
    (hx : ∀ i, |x i| ≤ a)
    (hG : ∀ j, |g j| ≤ G) (hg : ∀ j, |gt j - g j| ≤ eg)
    (hmargin : ∀ i', layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) <
      |Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) i'|)
    (j : Fin d₂) :
    |M.sub (b₁ j) (M.mul lr
        (reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
          (M.dense (fun j' i' => W₂ i' j') (fun _ => 0) gt) j)) -
      (b₁ j - lr *
        reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
          (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g) j)| ≤
    sgdErr M.u lr |b₁ j| (layerAct d₃ w₂ 0 G)
      (layerBudget M.u d₃ w₂ 0 G eg) := by
  have hE₀0 : 0 ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha le_rfl
  have l0 : ∀ j', |M.dense W₀ b₀ x j' - Proofs.dense W₀ b₀ x j'| ≤
      layerBudget M.u d₀ w₀ β₀ a 0 := fun j' =>
    (M.dense_close_fresh W₀ b₀ x j').trans
      (M.denseErr_le_uniform hw₀ le_rfl hW₀ hb₀ hx j')
  have r0 : ∀ j', |relu d₁ (M.dense W₀ b₀ x) j' -
      relu d₁ (Proofs.dense W₀ b₀ x) j'| ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    fun j' => relu_close _ _ _ l0 j'
  have ha₁ : ∀ i', |relu d₁ (Proofs.dense W₀ b₀ x) i'| ≤
      layerAct d₀ w₀ β₀ a :=
    fun i' => (relu_abs_le _ i').trans (dense_abs_le ha hW₀ hb₀ hx i')
  have l1 : ∀ j', |M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)) j' -
      Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) j'| ≤
      layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) := fun j' =>
    (M.dense_close W₁ b₁ _ _ _ hE₀0 r0 j').trans
      (M.denseErr_le_uniform hw₁ hE₀0 hW₁ hb₁ ha₁ j')
  have hcot := fun j' =>
    M.cot_step_close W₂ _ _ gt g hw₂ hG0 heg hW₂ hG hg l1 hmargin j'
  have hc₁ : |reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
      (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g) j| ≤
      layerAct d₃ w₂ 0 G :=
    (reluMask_abs_le _ _ j).trans
      (dense_abs_le hG0 (fun j' i' => hW₂ i' j') (fun _ => by simp) hG j)
  exact M.sgd_step_close (b₁ j) (hcot j) hc₁ hlr

/-- **Rounded input-layer weight update (W₀)** — the cotangent crosses BOTH
    masks, so both quantitative margins are required; the activation operand
    is the raw input `x`, identical in both nets (zero inherited error). The
    real target `W₀ᵢⱼ − lr·(xᵢ·c₀ⱼ)` is the `mlp_render_W0_certified`
    quantity. -/
theorem mlp_w0_step_float_close {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) {b₀ : Vec d₁} {W₁ : Mat d₁ d₂} {b₁ : Vec d₂}
    {W₂ : Mat d₂ d₃} {x : Vec d₀} {gt g : Vec d₃} {lr : ℝ}
    {w₀ β₀ w₁ β₁ w₂ a G eg : ℝ}
    (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁)
    (hw₂ : 0 ≤ w₂) (ha : 0 ≤ a) (hlr : 0 ≤ lr) (hG0 : 0 ≤ G) (heg : 0 ≤ eg)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w₂)
    (hx : ∀ i, |x i| ≤ a)
    (hG : ∀ j, |g j| ≤ G) (hg : ∀ j, |gt j - g j| ≤ eg)
    (hmargin₁ : ∀ i', layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) <
      |Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) i'|)
    (hmargin₀ : ∀ i', layerBudget M.u d₀ w₀ β₀ a 0 <
      |Proofs.dense W₀ b₀ x i'|)
    (i : Fin d₀) (j : Fin d₁) :
    |M.sub (W₀ i j) (M.mul lr (M.mul (x i)
        (reluMask (M.dense W₀ b₀ x)
          (M.dense (fun j' i' => W₁ i' j') (fun _ => 0)
            (reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
              (M.dense (fun j' i' => W₂ i' j') (fun _ => 0) gt))) j))) -
      (W₀ i j - lr * (x i *
        reluMask (Proofs.dense W₀ b₀ x)
          (Proofs.dense (fun j' i' => W₁ i' j') (fun _ => 0)
            (reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
              (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g))) j))| ≤
    sgdErr M.u lr |W₀ i j|
      (a * layerAct d₂ w₁ 0 (layerAct d₃ w₂ 0 G))
      (mulErr M.u a (layerAct d₂ w₁ 0 (layerAct d₃ w₂ 0 G)) 0
        (layerBudget M.u d₂ w₁ 0 (layerAct d₃ w₂ 0 G)
          (layerBudget M.u d₃ w₂ 0 G eg))) := by
  have hC₁0 : 0 ≤ layerAct d₃ w₂ 0 G := layerAct_nonneg hw₂ le_rfl hG0
  have hEC₁0 : 0 ≤ layerBudget M.u d₃ w₂ 0 G eg :=
    layerBudget_nonneg M.u_nonneg hw₂ le_rfl hG0 heg
  have hE₀0 : 0 ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha le_rfl
  have l0 : ∀ j', |M.dense W₀ b₀ x j' - Proofs.dense W₀ b₀ x j'| ≤
      layerBudget M.u d₀ w₀ β₀ a 0 := fun j' =>
    (M.dense_close_fresh W₀ b₀ x j').trans
      (M.denseErr_le_uniform hw₀ le_rfl hW₀ hb₀ hx j')
  have r0 : ∀ j', |relu d₁ (M.dense W₀ b₀ x) j' -
      relu d₁ (Proofs.dense W₀ b₀ x) j'| ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    fun j' => relu_close _ _ _ l0 j'
  have ha₁ : ∀ i', |relu d₁ (Proofs.dense W₀ b₀ x) i'| ≤
      layerAct d₀ w₀ β₀ a :=
    fun i' => (relu_abs_le _ i').trans (dense_abs_le ha hW₀ hb₀ hx i')
  have l1 : ∀ j', |M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)) j' -
      Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) j'| ≤
      layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) := fun j' =>
    (M.dense_close W₁ b₁ _ _ _ hE₀0 r0 j').trans
      (M.denseErr_le_uniform hw₁ hE₀0 hW₁ hb₁ ha₁ j')
  -- layer-1 cotangent, then the layer-0 cotangent through the second mask
  have hcot := fun j' =>
    M.cot_step_close W₂ _ _ gt g hw₂ hG0 heg hW₂ hG hg l1 hmargin₁ j'
  have hc₁mag : ∀ j', |reluMask
      (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
      (Proofs.dense (fun j'' i' => W₂ i' j'') (fun _ => 0) g) j'| ≤
      layerAct d₃ w₂ 0 G := fun j' =>
    (reluMask_abs_le _ _ j').trans
      (dense_abs_le hG0 (fun j'' i' => hW₂ i' j'') (fun _ => by simp) hG j')
  have hcot0 := fun j' =>
    M.cot_step_close W₁ (M.dense W₀ b₀ x) (Proofs.dense W₀ b₀ x) _ _
      hw₁ hC₁0 hEC₁0 hW₁ hc₁mag hcot l0 hmargin₀ j'
  have hc₀mag : |reluMask (Proofs.dense W₀ b₀ x)
      (Proofs.dense (fun j' i' => W₁ i' j') (fun _ => 0)
        (reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
          (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g))) j| ≤
      layerAct d₂ w₁ 0 (layerAct d₃ w₂ 0 G) :=
    (reluMask_abs_le _ _ j).trans
      (dense_abs_le hC₁0 (fun j' i' => hW₁ i' j') (fun _ => by simp)
        hc₁mag j)
  have hmul := M.mul_close (show |x i - x i| ≤ 0 by simp) (hcot0 j)
    (hx i) hc₀mag
  have hac : |x i * reluMask (Proofs.dense W₀ b₀ x)
      (Proofs.dense (fun j' i' => W₁ i' j') (fun _ => 0)
        (reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
          (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g))) j| ≤
      a * layerAct d₂ w₁ 0 (layerAct d₃ w₂ 0 G) := by
    rw [abs_mul]
    exact mul_le_mul (hx i) hc₀mag (abs_nonneg _) ha
  exact M.sgd_step_close (W₀ i j) hmul hac hlr

/-- **Rounded input bias update (b₀)** — the layer-0 cotangent directly. -/
theorem mlp_b0_step_float_close {d₀ d₁ d₂ d₃ : Nat}
    {W₀ : Mat d₀ d₁} (b₀ : Vec d₁) {W₁ : Mat d₁ d₂} {b₁ : Vec d₂}
    {W₂ : Mat d₂ d₃} {x : Vec d₀} {gt g : Vec d₃} {lr : ℝ}
    {w₀ β₀ w₁ β₁ w₂ a G eg : ℝ}
    (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁)
    (hw₂ : 0 ≤ w₂) (ha : 0 ≤ a) (hlr : 0 ≤ lr) (hG0 : 0 ≤ G) (heg : 0 ≤ eg)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w₂)
    (hx : ∀ i, |x i| ≤ a)
    (hG : ∀ j, |g j| ≤ G) (hg : ∀ j, |gt j - g j| ≤ eg)
    (hmargin₁ : ∀ i', layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) <
      |Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) i'|)
    (hmargin₀ : ∀ i', layerBudget M.u d₀ w₀ β₀ a 0 <
      |Proofs.dense W₀ b₀ x i'|)
    (j : Fin d₁) :
    |M.sub (b₀ j) (M.mul lr
        (reluMask (M.dense W₀ b₀ x)
          (M.dense (fun j' i' => W₁ i' j') (fun _ => 0)
            (reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
              (M.dense (fun j' i' => W₂ i' j') (fun _ => 0) gt))) j)) -
      (b₀ j - lr *
        reluMask (Proofs.dense W₀ b₀ x)
          (Proofs.dense (fun j' i' => W₁ i' j') (fun _ => 0)
            (reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
              (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g))) j)| ≤
    sgdErr M.u lr |b₀ j| (layerAct d₂ w₁ 0 (layerAct d₃ w₂ 0 G))
      (layerBudget M.u d₂ w₁ 0 (layerAct d₃ w₂ 0 G)
        (layerBudget M.u d₃ w₂ 0 G eg)) := by
  have hC₁0 : 0 ≤ layerAct d₃ w₂ 0 G := layerAct_nonneg hw₂ le_rfl hG0
  have hEC₁0 : 0 ≤ layerBudget M.u d₃ w₂ 0 G eg :=
    layerBudget_nonneg M.u_nonneg hw₂ le_rfl hG0 heg
  have hE₀0 : 0 ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha le_rfl
  have l0 : ∀ j', |M.dense W₀ b₀ x j' - Proofs.dense W₀ b₀ x j'| ≤
      layerBudget M.u d₀ w₀ β₀ a 0 := fun j' =>
    (M.dense_close_fresh W₀ b₀ x j').trans
      (M.denseErr_le_uniform hw₀ le_rfl hW₀ hb₀ hx j')
  have r0 : ∀ j', |relu d₁ (M.dense W₀ b₀ x) j' -
      relu d₁ (Proofs.dense W₀ b₀ x) j'| ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=
    fun j' => relu_close _ _ _ l0 j'
  have ha₁ : ∀ i', |relu d₁ (Proofs.dense W₀ b₀ x) i'| ≤
      layerAct d₀ w₀ β₀ a :=
    fun i' => (relu_abs_le _ i').trans (dense_abs_le ha hW₀ hb₀ hx i')
  have l1 : ∀ j', |M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)) j' -
      Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) j'| ≤
      layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) := fun j' =>
    (M.dense_close W₁ b₁ _ _ _ hE₀0 r0 j').trans
      (M.denseErr_le_uniform hw₁ hE₀0 hW₁ hb₁ ha₁ j')
  have hcot := fun j' =>
    M.cot_step_close W₂ _ _ gt g hw₂ hG0 heg hW₂ hG hg l1 hmargin₁ j'
  have hc₁mag : ∀ j', |reluMask
      (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
      (Proofs.dense (fun j'' i' => W₂ i' j'') (fun _ => 0) g) j'| ≤
      layerAct d₃ w₂ 0 G := fun j' =>
    (reluMask_abs_le _ _ j').trans
      (dense_abs_le hG0 (fun j'' i' => hW₂ i' j'') (fun _ => by simp) hG j')
  have hcot0 := fun j' =>
    M.cot_step_close W₁ (M.dense W₀ b₀ x) (Proofs.dense W₀ b₀ x) _ _
      hw₁ hC₁0 hEC₁0 hW₁ hc₁mag hcot l0 hmargin₀ j'
  have hc₀mag : |reluMask (Proofs.dense W₀ b₀ x)
      (Proofs.dense (fun j' i' => W₁ i' j') (fun _ => 0)
        (reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
          (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0) g))) j| ≤
      layerAct d₂ w₁ 0 (layerAct d₃ w₂ 0 G) :=
    (reluMask_abs_le _ _ j).trans
      (dense_abs_le hC₁0 (fun j' i' => hW₁ i' j') (fun _ => by simp)
        hc₁mag j)
  exact M.sgd_step_close (b₀ j) (hcot0 j) hc₀mag hlr

/-- **Numeric gradient capstone at the committed dims and TRAINED
    magnitudes** (784→512→512→10, `|W| ≤ 3/5` covering the measured
    `max|W| = 0.52`): binary32 accuracy (`u ≤ 2⁻²⁴`), `lr = 1/10`,
    `|b|, |x| ≤ 1`, `|g| ≤ 1` (a softmax−onehot cotangent is always in
    `[−1,1]`), cotangent taken exact — then every rounded W₂ SGD entry is
    within **5/4** of the certified real step.

    The budget decomposes honestly: ~1.2 of it is `lr·E₁·|g|` — the
    *forward* budget riding through the gradient at learning-rate scale —
    while fresh backward rounding contributes only ~2·10⁻³. The gradient
    step is as accurate as the forward pass, no worse. Measured on the
    live run (`scripts/margin_probe.py`): actual W₂ step deviation
    ≤ 7.5·10⁻⁹ — the worst-case-vs-measured gap is the a-posteriori case
    in numbers. -/
theorem mnist_w2_step_float_budget (hMu : M.u ≤ u32)
    (W₀ : Mat 784 512) (b₀ : Vec 512) (W₁ : Mat 512 512) (b₁ : Vec 512)
    (W₂ : Mat 512 10) (x : Vec 784) (g : Vec 10)
    (hW₀ : ∀ i j, |W₀ i j| ≤ 3/5) (hb₀ : ∀ j, |b₀ j| ≤ 1)
    (hW₁ : ∀ i j, |W₁ i j| ≤ 3/5) (hb₁ : ∀ j, |b₁ j| ≤ 1)
    (hW₂ : ∀ i j, |W₂ i j| ≤ 3/5)
    (hx : ∀ i, |x i| ≤ 1) (hG : ∀ j, |g j| ≤ 1)
    (i : Fin 512) (j : Fin 10) :
    |M.sub (W₂ i j) (M.mul (1/10) (M.mul
        (relu 512 (M.dense W₁ b₁ (relu 512 (M.dense W₀ b₀ x))) i) (g j))) -
      (W₂ i j - (1/10) * (relu 512 (Proofs.dense W₁ b₁
        (relu 512 (Proofs.dense W₀ b₀ x))) i * g j))| ≤ 5/4 := by
  have hu := M.u_nonneg
  have hmain := M.mlp_w2_step_float_close (gt := g) (eg := 0) (lr := 1/10) W₂
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) hW₀ hb₀ hW₁ hb₁ hx hG (fun j' => by simp) i j
  rw [show layerAct 784 (3/5) 1 1 = (2357/5 : ℝ) by norm_num [layerAct],
      show layerAct 512 (3/5) 1 (2357/5) = (3620377/25 : ℝ) by
        norm_num [layerAct]]
    at hmain
  refine hmain.trans ?_
  have hm1 : mulErr M.u (3620377/25) 1 (layerBudget M.u 512 (3/5) 1 (2357/5)
      (layerBudget M.u 784 (3/5) 1 1 0)) 0 ≤ 121/10 := by
    refine (mulErr_mono hu hMu (by norm_num) (by norm_num)
      M.mnist_E1_nonneg (M.mnist_E1_le hMu) le_rfl).trans ?_
    norm_num [FloatModel.mulErr, u32]
  have hm0 : (0:ℝ) ≤ mulErr M.u (3620377/25) 1
      (layerBudget M.u 512 (3/5) 1 (2357/5)
        (layerBudget M.u 784 (3/5) 1 1 0)) 0 :=
    mulErr_nonneg hu (by norm_num) (by norm_num) M.mnist_E1_nonneg le_rfl
  refine (sgdErr_mono hu hMu (by norm_num) (abs_nonneg _) (hW₂ i j)
    (by norm_num) hm0 hm1).trans ?_
  norm_num [FloatModel.sgdErr, u32]

-- ════════════════════════════════════════════════════════════════
-- § The loss head: rounded softmax−onehot cotangent
-- ════════════════════════════════════════════════════════════════

/-- Rounded division: `fl(x / y)`. -/
noncomputable def div (x y : ℝ) : ℝ := M.rnd (x / y)

/-- Rounded sum, left-fold association. Like `dot`, the bound below holds
    for every association. -/
noncomputable def sum : {n : Nat} → Vec n → ℝ
  | 0, _ => 0
  | n + 1, x => M.add (sum (fun i => x i.castSucc)) (x (Fin.last n))

theorem sum_succ {n : Nat} (x : Vec (n + 1)) :
    M.sum x = M.add (M.sum (fun i => x i.castSucc)) (x (Fin.last n)) := rfl

/-- **Rounded sum forward error** — `((1+u)^(n+1) − 1)·Σ|xᵢ|`, association-
    independent (exponent `n+1` because the seed addition with `0` rounds). -/
theorem sum_close : ∀ {n : ℕ} (x : Vec n),
    |M.sum x - ∑ i, x i| ≤ ((1 + M.u) ^ (n + 1) - 1) * ∑ i, |x i| := by
  intro n
  induction n with
  | zero => intro x; simp [FloatModel.sum]
  | succ n ih =>
    intro x
    rw [M.sum_succ x]
    simp only [Fin.sum_univ_castSucc]
    rw [show ((1 : ℝ) + M.u) ^ (n + 1 + 1) = (1 + M.u) ^ (n + 1) * (1 + M.u)
        from pow_succ _ _]
    exact step_bound M.u_nonneg
      (Finset.abs_sum_le_sum_abs _ _)
      (M.one_add_u_le_pow (by omega))
      (ih (fun i => x i.castSucc))
      (by simp only [sub_self, abs_zero]
          exact mul_nonneg M.u_nonneg (abs_nonneg _))
      (M.err _)

-- ════════════════════════════════════════════════════════════════
-- § Gradient-is-a-reduction SGD step (the conv-grad reuse, planning §1b-B)
-- ════════════════════════════════════════════════════════════════

/-- **SGD step whose gradient is a rounded dot product.** When the gradient is
    a correlation `g = Σ pᵢqᵢ` computed in float as `M.dot p q` — the shape of
    a conv *weight* gradient (`Σ_{hi,wi} convPad · cot`) and of any dense weight
    gradient — the rounded update `fl(θ − fl(lr·fl(p·q)))` is within `sgdErr` of
    the real step `θ − lr·g`, with the dot's Higham γ as the gradient-error
    slot `eg`. This is `dot_close` feeding `sgd_step_close`. -/
theorem dotSgd_step_close (θ : ℝ) {n : ℕ} (p q : Vec n) {lr G : ℝ}
    (hG : |∑ i, p i * q i| ≤ G) (hlr : 0 ≤ lr) :
    |M.sub θ (M.mul lr (M.dot p q)) - (θ - lr * ∑ i, p i * q i)| ≤
      sgdErr M.u lr |θ| G (((1 + M.u) ^ (n + 1) - 1) * ∑ i, |p i * q i|) :=
  M.sgd_step_close θ (M.dot_close p q) hG hlr

/-- **SGD step whose gradient is a rounded sum.** When the gradient is a plain
    reduction `g = Σ xᵢ` computed in float as `M.sum x` — the shape of a conv
    *bias* gradient (`Σ_{hi,wi} cot`) — the rounded update is within `sgdErr` of
    the real step, with the sum's Higham γ as the `eg` slot. -/
theorem sumSgd_step_close (θ : ℝ) {n : ℕ} (x : Vec n) {lr G : ℝ}
    (hG : |∑ i, x i| ≤ G) (hlr : 0 ≤ lr) :
    |M.sub θ (M.mul lr (M.sum x)) - (θ - lr * ∑ i, x i)| ≤
      sgdErr M.u lr |θ| G (((1 + M.u) ^ (n + 1) - 1) * ∑ i, |x i|) :=
  M.sgd_step_close θ (M.sum_close x) hG hlr

/-- The float softmax: rounded `exp`, rounded sum, rounded division — the
    structure of the rendered loss head. `fexp` is hypothesis-supplied
    (GPU `exp` has no IEEE spec; its accuracy constant is exactly what the
    repo's `vjp_oracle` harness validates empirically). -/
noncomputable def softmaxF (fexp : ℝ → ℝ) {n : Nat} (z : Vec n) : Vec n :=
  fun k => M.div (fexp (z k)) (M.sum (fun j => fexp (z j)))

/-- The float softmax−onehot cotangent (one final rounded subtract; the
    onehot operand is exact). -/
noncomputable def softmaxCECotF (fexp : ℝ → ℝ) {n : Nat} (z : Vec n)
    (label : Fin n) : Vec n :=
  fun k => M.sub (M.softmaxF fexp z k) (oneHot n label k)

private theorem softmax_nonneg {n : ℕ} (z : Vec n) (k : Fin n) :
    0 ≤ softmax n z k :=
  div_nonneg (Real.exp_pos _).le
    (Finset.sum_nonneg fun j _ => (Real.exp_pos (z j)).le)

private theorem softmax_le_one {n : ℕ} (z : Vec n) (k : Fin n) :
    softmax n z k ≤ 1 := by
  have hD : 0 < ∑ j, Real.exp (z j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩
  exact (div_le_one hD).mpr
    (Finset.single_le_sum (fun j _ => (Real.exp_pos (z j)).le)
      (Finset.mem_univ k))

/-- **Softmax perturbation, elementary ratio form**: a coordinatewise logit
    error `δ` moves every softmax output by at most `e^(2δ) − 1`. Proved by
    sandwiching `softmax(z̃) ∈ [e^(−2δ), e^(2δ)]·softmax(z)` with bare `exp`
    monotonicity — no mean-value theorem. -/
theorem softmax_perturb {n : ℕ} (zt z : Vec n) {δ : ℝ}
    (hδ : ∀ k', |zt k' - z k'| ≤ δ) (k : Fin n) :
    |softmax n zt k - softmax n z k| ≤ Real.exp (2 * δ) - 1 := by
  have hδ0 : 0 ≤ δ := (abs_nonneg _).trans (hδ k)
  have hD : 0 < ∑ j, Real.exp (z j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩
  have hDt : 0 < ∑ j, Real.exp (zt j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩
  -- numerator and denominator sandwiches
  have hnum_ub : Real.exp (zt k) ≤ Real.exp δ * Real.exp (z k) := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by have := abs_le.mp (hδ k); linarith)
  have hnum_lb : Real.exp (-δ) * Real.exp (z k) ≤ Real.exp (zt k) := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by have := abs_le.mp (hδ k); linarith)
  have hden_lb : Real.exp (-δ) * ∑ j, Real.exp (z j) ≤ ∑ j, Real.exp (zt j) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun j _ => ?_
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by have := abs_le.mp (hδ j); linarith)
  have hden_ub : (∑ j, Real.exp (zt j)) ≤ Real.exp δ * ∑ j, Real.exp (z j) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun j _ => ?_
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by have := abs_le.mp (hδ j); linarith)
  -- the two-sided ratio bound
  have hub : softmax n zt k ≤ Real.exp (2 * δ) * softmax n z k := by
    have h1 : Real.exp (zt k) / (∑ j, Real.exp (zt j)) ≤
        (Real.exp δ * Real.exp (z k)) / (Real.exp (-δ) * ∑ j, Real.exp (z j)) :=
      div_le_div₀ (mul_nonneg (Real.exp_pos δ).le (Real.exp_pos _).le)
        hnum_ub (mul_pos (Real.exp_pos _) hD) hden_lb
    have h2 : (Real.exp δ * Real.exp (z k)) /
        (Real.exp (-δ) * ∑ j, Real.exp (z j)) =
        Real.exp (2 * δ) * (Real.exp (z k) / ∑ j, Real.exp (z j)) := by
      rw [mul_div_mul_comm, ← Real.exp_sub]
      ring_nf
    exact le_of_le_of_eq h1 h2
  have hlb : Real.exp (-(2 * δ)) * softmax n z k ≤ softmax n zt k := by
    have h1 : (Real.exp (-δ) * Real.exp (z k)) /
        (Real.exp δ * ∑ j, Real.exp (z j)) ≤
        Real.exp (zt k) / ∑ j, Real.exp (zt j) :=
      div_le_div₀ (Real.exp_pos _).le hnum_lb hDt hden_ub
    have h2 : (Real.exp (-δ) * Real.exp (z k)) /
        (Real.exp δ * ∑ j, Real.exp (z j)) =
        Real.exp (-(2 * δ)) * (Real.exp (z k) / ∑ j, Real.exp (z j)) := by
      rw [mul_div_mul_comm, ← Real.exp_sub]
      ring_nf
    exact le_of_eq_of_le h2.symm h1
  -- assemble
  have hs0 := softmax_nonneg z k
  have hs1 := softmax_le_one z k
  have hexp1 : 1 ≤ Real.exp (2 * δ) := by
    have := Real.add_one_le_exp (2 * δ); linarith
  have hprod : Real.exp (2 * δ) * Real.exp (-(2 * δ)) = 1 := by
    rw [← Real.exp_add]; simp
  have hsum2 : 2 ≤ Real.exp (2 * δ) + Real.exp (-(2 * δ)) := by
    nlinarith [sq_nonneg (Real.exp (2 * δ) - 1), Real.exp_pos (2 * δ)]
  rw [abs_le]
  constructor
  · nlinarith [hlb, hs1, hs0]
  · nlinarith [hub, hs1, hs0]

/-- Denominator perturbation of the float softmax: rounded-sum compounding
    on `exp`-inaccurate terms. -/
noncomputable def smRho (u eexp : ℝ) (n : ℕ) : ℝ :=
  ((1 + u) ^ (n + 1) - 1) * (1 + eexp) + eexp

/-- Relative budget of the pre-rounding float softmax against the real
    softmax at the same logits. -/
noncomputable def smKappa (u eexp : ℝ) (n : ℕ) : ℝ :=
  (eexp + smRho u eexp n) / (1 - smRho u eexp n)

/-- Absolute budget of the float softmax against the real softmax at the
    REAL logits: head rounding + the `e^(2δ) − 1` logit-perturbation term. -/
noncomputable def smErr (u eexp δ : ℝ) (n : ℕ) : ℝ :=
  u * (1 + smKappa u eexp n) + smKappa u eexp n + (Real.exp (2 * δ) - 1)

/-- Budget of the full rounded softmax−onehot cotangent against the
    certified real gradient. -/
noncomputable def cotErr (u eexp δ : ℝ) (n : ℕ) : ℝ :=
  u * (1 + smErr u eexp δ n) + smErr u eexp δ n

private theorem smRho_nonneg {eexp : ℝ} {n : ℕ} (heexp : 0 ≤ eexp) :
    0 ≤ smRho M.u eexp n :=
  add_nonneg
    (mul_nonneg (sub_nonneg.mpr (M.one_le_pow_one_add_u (n + 1)))
      (by linarith))
    heexp

/-- **`cotErr` is nonnegative** (it bounds an absolute value) — under `eexp ≥ 0`,
    `δ ≥ 0`, and the denominator condition `smRho < 1`. The `cot_step_close`
    precondition for any backward grad-close that runs the softmax−onehot head
    (e.g. the per-layer η-composition rungs). -/
theorem cotErr_nonneg {eexp δ : ℝ} {n : ℕ} (heexp : 0 ≤ eexp) (hδ0 : 0 ≤ δ)
    (hρ1 : smRho M.u eexp n < 1) : 0 ≤ cotErr M.u eexp δ n := by
  have hu := M.u_nonneg
  have hρ0 : 0 ≤ smRho M.u eexp n := M.smRho_nonneg heexp
  have hκ0 : 0 ≤ smKappa M.u eexp n := by
    simp only [smKappa]; exact div_nonneg (by linarith) (by linarith)
  have hexp0 : 0 ≤ Real.exp (2 * δ) - 1 := by
    have := Real.add_one_le_exp (2 * δ); linarith
  have hsm0 : 0 ≤ smErr M.u eexp δ n := by
    simp only [smErr]
    have := mul_nonneg hu (by linarith : (0:ℝ) ≤ 1 + smKappa M.u eexp n)
    linarith
  simp only [cotErr]
  have := mul_nonneg hu (by linarith : (0:ℝ) ≤ 1 + smErr M.u eexp δ n)
  linarith

/-- **Float softmax vs real softmax at the same logits** (part A): the
    rounded `exp`/`sum`/`div` head is within `u·(1+κ) + κ` absolutely, where
    `κ = (eexp + ρ)/(1 − ρ)` compounds the `exp` accuracy and the sum
    rounding. The sandwich is the same ratio argument as
    `softmax_perturb` — division-perturbation never appears. -/
theorem softmaxF_close (fexp : ℝ → ℝ) {eexp : ℝ} {n : ℕ} (z : Vec n)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : smRho M.u eexp n < 1) (k : Fin n) :
    |M.softmaxF fexp z k - softmax n z k| ≤
      M.u * (1 + smKappa M.u eexp n) + smKappa M.u eexp n := by
  have hu := M.u_nonneg
  have hρ0 : 0 ≤ smRho M.u eexp n := M.smRho_nonneg heexp0
  have hκ0 : 0 ≤ smKappa M.u eexp n :=
    div_nonneg (by linarith) (by linarith)
  have hD : 0 < ∑ j, Real.exp (z j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩
  have hG0 : (0:ℝ) ≤ (1 + M.u) ^ (n + 1) - 1 :=
    sub_nonneg.mpr (M.one_le_pow_one_add_u (n + 1))
  -- numerator sandwich
  have hN_ub : fexp (z k) ≤ (1 + eexp) * Real.exp (z k) := by
    nlinarith [abs_le.mp (hfexp (z k))]
  have hN_lb : (1 - eexp) * Real.exp (z k) ≤ fexp (z k) := by
    nlinarith [abs_le.mp (hfexp (z k))]
  have hN0 : 0 ≤ fexp (z k) :=
    le_trans (mul_nonneg (by linarith) (Real.exp_pos _).le) hN_lb
  -- denominator sandwich
  have habs_v : ∀ j : Fin n, |fexp (z j)| ≤ (1 + eexp) * Real.exp (z j) := by
    intro j
    have h2 : |fexp (z j)| ≤ |fexp (z j) - Real.exp (z j)| + |Real.exp (z j)| := by
      simpa using abs_sub_le (fexp (z j)) (Real.exp (z j)) 0
    rw [abs_of_pos (Real.exp_pos _)] at h2
    nlinarith [hfexp (z j)]
  have hSv_err : |(∑ j, fexp (z j)) - ∑ j, Real.exp (z j)| ≤
      eexp * ∑ j, Real.exp (z j) := by
    rw [← Finset.sum_sub_distrib, Finset.mul_sum]
    exact (Finset.abs_sum_le_sum_abs _ _).trans
      (Finset.sum_le_sum fun j _ => hfexp (z j))
  have hSabs : (∑ j, |fexp (z j)|) ≤ (1 + eexp) * ∑ j, Real.exp (z j) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ => habs_v j
  have hS_err : |M.sum (fun j => fexp (z j)) - ∑ j, Real.exp (z j)| ≤
      smRho M.u eexp n * ∑ j, Real.exp (z j) := by
    calc |M.sum (fun j => fexp (z j)) - ∑ j, Real.exp (z j)|
        ≤ |M.sum (fun j => fexp (z j)) - ∑ j, fexp (z j)| +
          |(∑ j, fexp (z j)) - ∑ j, Real.exp (z j)| := abs_sub_le _ _ _
      _ ≤ ((1 + M.u) ^ (n + 1) - 1) * ((1 + eexp) * ∑ j, Real.exp (z j)) +
          eexp * ∑ j, Real.exp (z j) :=
          add_le_add ((M.sum_close _).trans
            (mul_le_mul_of_nonneg_left hSabs hG0)) hSv_err
      _ = smRho M.u eexp n * ∑ j, Real.exp (z j) := by
          simp only [smRho]; ring
  have hS_lb : (1 - smRho M.u eexp n) * (∑ j, Real.exp (z j)) ≤
      M.sum (fun j => fexp (z j)) := by
    have := abs_le.mp hS_err; nlinarith
  have hS_ub : M.sum (fun j => fexp (z j)) ≤
      (1 + smRho M.u eexp n) * ∑ j, Real.exp (z j) := by
    have := abs_le.mp hS_err; nlinarith
  have hSden_pos : 0 < (1 - smRho M.u eexp n) * ∑ j, Real.exp (z j) :=
    mul_pos (by linarith) hD
  have hS_pos : 0 < M.sum (fun j => fexp (z j)) :=
    lt_of_lt_of_le hSden_pos hS_lb
  -- the pre-rounding quotient sandwich
  have hsdef : softmax n z k = Real.exp (z k) / ∑ j, Real.exp (z j) := rfl
  have hs0 := softmax_nonneg z k
  have hs1 := softmax_le_one z k
  have hQub : fexp (z k) / M.sum (fun j => fexp (z j)) ≤
      (1 + smKappa M.u eexp n) * softmax n z k := by
    have h1 : fexp (z k) / M.sum (fun j => fexp (z j)) ≤
        ((1 + eexp) * Real.exp (z k)) /
          ((1 - smRho M.u eexp n) * ∑ j, Real.exp (z j)) :=
      div_le_div₀ (mul_nonneg (by linarith) (Real.exp_pos _).le) hN_ub
        hSden_pos hS_lb
    have h2 : ((1 + eexp) * Real.exp (z k)) /
        ((1 - smRho M.u eexp n) * ∑ j, Real.exp (z j)) =
        ((1 + eexp) / (1 - smRho M.u eexp n)) * softmax n z k := by
      rw [mul_div_mul_comm, hsdef]
    have hne : (1:ℝ) - smRho M.u eexp n ≠ 0 := ne_of_gt (by linarith)
    have h3 : (1 + eexp) / (1 - smRho M.u eexp n) = 1 + smKappa M.u eexp n := by
      simp only [smKappa]
      rw [div_eq_iff hne, add_mul, one_mul, div_mul_cancel₀ _ hne]
      ring
    rw [h2, h3] at h1
    exact h1
  have hQlb : (1 - smKappa M.u eexp n) * softmax n z k ≤
      fexp (z k) / M.sum (fun j => fexp (z j)) := by
    have h1 : ((1 - eexp) * Real.exp (z k)) /
        ((1 + smRho M.u eexp n) * ∑ j, Real.exp (z j)) ≤
        fexp (z k) / M.sum (fun j => fexp (z j)) :=
      div_le_div₀ hN0 hN_lb hS_pos hS_ub
    have h2 : ((1 - eexp) * Real.exp (z k)) /
        ((1 + smRho M.u eexp n) * ∑ j, Real.exp (z j)) =
        ((1 - eexp) / (1 + smRho M.u eexp n)) * softmax n z k := by
      rw [mul_div_mul_comm, hsdef]
    have h3 : 1 - smKappa M.u eexp n ≤
        (1 - eexp) / (1 + smRho M.u eexp n) := by
      have hne : (1:ℝ) - smRho M.u eexp n ≠ 0 := ne_of_gt (by linarith)
      have hκdef : smKappa M.u eexp n * (1 - smRho M.u eexp n) =
          eexp + smRho M.u eexp n := by
        simp only [smKappa]
        rw [div_mul_cancel₀ _ hne]
      have hκρ : eexp + smRho M.u eexp n ≤
          smKappa M.u eexp n * (1 + smRho M.u eexp n) := by
        have h4 : smKappa M.u eexp n * (1 - smRho M.u eexp n) ≤
            smKappa M.u eexp n * (1 + smRho M.u eexp n) :=
          mul_le_mul_of_nonneg_left (by linarith) hκ0
        linarith [hκdef]
      rw [le_div_iff₀ (by linarith)]
      nlinarith [hκρ]
    calc (1 - smKappa M.u eexp n) * softmax n z k
        ≤ ((1 - eexp) / (1 + smRho M.u eexp n)) * softmax n z k :=
          mul_le_mul_of_nonneg_right h3 hs0
      _ = _ := h2.symm
      _ ≤ _ := h1
  have hQs : |fexp (z k) / M.sum (fun j => fexp (z j)) - softmax n z k| ≤
      smKappa M.u eexp n := by
    rw [abs_le]
    constructor
    · nlinarith [hQlb, hs0, hs1, hκ0]
    · nlinarith [hQub, hs0, hs1, hκ0]
  have hQabs : |fexp (z k) / M.sum (fun j => fexp (z j))| ≤
      1 + smKappa M.u eexp n := by
    have h1 : |fexp (z k) / M.sum (fun j => fexp (z j))| ≤
        |fexp (z k) / M.sum (fun j => fexp (z j)) - softmax n z k| +
          |softmax n z k| := by
      simpa using abs_sub_le (fexp (z k) / M.sum (fun j => fexp (z j)))
        (softmax n z k) 0
    rw [abs_of_nonneg hs0] at h1
    linarith
  have hrnd : |M.softmaxF fexp z k -
      fexp (z k) / M.sum (fun j => fexp (z j))| ≤
      M.u * |fexp (z k) / M.sum (fun j => fexp (z j))| := M.err _
  have htri : |M.softmaxF fexp z k - softmax n z k| ≤
      |M.softmaxF fexp z k - fexp (z k) / M.sum (fun j => fexp (z j))| +
        |fexp (z k) / M.sum (fun j => fexp (z j)) - softmax n z k| :=
    abs_sub_le _ _ _
  have h4 := mul_le_mul_of_nonneg_left hQabs hu
  linarith

/-- **The rounded softmax−onehot cotangent is within `cotErr` of the
    certified real gradient** `softmax(z) − onehot` — the `pdiv`-certified
    `∂(crossEntropy)/∂logits` (`softmaxCE_grad`). This discharges the
    `g̃ ≈ g` hypothesis of the `mlp_*_step_float_close` capstones:
    `eg := cotErr u eexp δ n`, where `δ` bounds the float-vs-real logits
    (worst case: the forward `layerBudget`; in practice: an a-posteriori
    measured value, since `e^(2δ) − 1` is only sharp for small `δ`). -/
theorem softmax_ce_cot_close (fexp : ℝ → ℝ) {eexp δ : ℝ} {n : ℕ}
    (zt z : Vec n) (label : Fin n)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : smRho M.u eexp n < 1)
    (hδ : ∀ k', |zt k' - z k'| ≤ δ) (k : Fin n) :
    |M.softmaxCECotF fexp zt label k -
      (softmax n z k - oneHot n label k)| ≤ cotErr M.u eexp δ n := by
  have hu := M.u_nonneg
  have hδ0 : 0 ≤ δ := (abs_nonneg _).trans (hδ k)
  have hκ0 : 0 ≤ smKappa M.u eexp n :=
    div_nonneg (by linarith [M.smRho_nonneg (eexp := eexp) (n := n) heexp0])
      (by linarith)
  have hexp1 : 1 ≤ Real.exp (2 * δ) := by
    have := Real.add_one_le_exp (2 * δ); linarith
  -- part A + part B
  have hA := M.softmaxF_close fexp zt heexp0 heexp1 hfexp hρ1 k
  have hB := softmax_perturb zt z hδ k
  have hsm : |M.softmaxF fexp zt k - softmax n z k| ≤
      smErr M.u eexp δ n := by
    have htri := abs_sub_le (M.softmaxF fexp zt k) (softmax n zt k)
      (softmax n z k)
    simp only [smErr]
    linarith
  have hsm0 : 0 ≤ smErr M.u eexp δ n := by
    simp only [smErr]
    nlinarith [mul_nonneg hu (by linarith : (0:ℝ) ≤ 1 + smKappa M.u eexp n)]
  -- |real softmax − onehot| ≤ 1
  have hs0 := softmax_nonneg z k
  have hs1 := softmax_le_one z k
  have hy : |softmax n z k - oneHot n label k| ≤ 1 := by
    simp only [oneHot]
    by_cases h : k = label
    · rw [if_pos h, abs_le]; constructor <;> linarith
    · rw [if_neg h, abs_le]; constructor <;> linarith
  -- the final rounded subtract
  have hrnd : |M.softmaxCECotF fexp zt label k -
      (M.softmaxF fexp zt k - oneHot n label k)| ≤
      M.u * |M.softmaxF fexp zt k - oneHot n label k| := M.err _
  have hsFy : |M.softmaxF fexp zt k - oneHot n label k| ≤
      1 + smErr M.u eexp δ n := by
    have h1 : |M.softmaxF fexp zt k - oneHot n label k| ≤
        |M.softmaxF fexp zt k - softmax n z k| +
          |softmax n z k - oneHot n label k| := abs_sub_le _ _ _
    linarith
  have htri : |M.softmaxCECotF fexp zt label k -
      (softmax n z k - oneHot n label k)| ≤
      |M.softmaxCECotF fexp zt label k -
        (M.softmaxF fexp zt k - oneHot n label k)| +
        |M.softmaxF fexp zt k - softmax n z k| := by
    have h1 := abs_sub_le (M.softmaxCECotF fexp zt label k)
      (M.softmaxF fexp zt k - oneHot n label k)
      (softmax n z k - oneHot n label k)
    have h2 : |(M.softmaxF fexp zt k - oneHot n label k) -
        (softmax n z k - oneHot n label k)| =
        |M.softmaxF fexp zt k - softmax n z k| := by
      rw [show (M.softmaxF fexp zt k - oneHot n label k) -
          (softmax n z k - oneHot n label k) =
          M.softmaxF fexp zt k - softmax n z k from by ring]
    linarith
  have h4 := mul_le_mul_of_nonneg_left hsFy hu
  simp only [cotErr]
  linarith

/-- **`|softmax z k| ≤ 1`** — the real softmax is a probability (public face of the
    `softmax_nonneg`/`softmax_le_one` pair, the magnitude attention's output matmul needs). -/
theorem softmax_abs_le_one {n : ℕ} (z : Vec n) (k : Fin n) :
    |softmax n z k| ≤ 1 := by
  rw [abs_of_nonneg (softmax_nonneg z k)]; exact softmax_le_one z k

/-- **`smErr` is nonnegative** under `eexp ≥ 0`, `δ ≥ 0`, and `smRho < 1` — the absolute
    softmax-vs-softmax budget bounds an absolute value, so it is itself `≥ 0`. (Extracted
    from `softmax_ce_cot_close`'s internal `hsm0`; needed as the `0 ≤ eweight` precondition
    of any downstream dot at perturbed softmax weights, e.g. attention's output matmul.) -/
theorem smErr_nonneg {eexp δ : ℝ} {n : ℕ} (heexp0 : 0 ≤ eexp) (hδ0 : 0 ≤ δ)
    (hρ1 : smRho M.u eexp n < 1) : 0 ≤ smErr M.u eexp δ n := by
  have hu := M.u_nonneg
  have hκ0 : 0 ≤ smKappa M.u eexp n :=
    div_nonneg (by linarith [M.smRho_nonneg (eexp := eexp) (n := n) heexp0]) (by linarith)
  have hexp0 : 0 ≤ Real.exp (2 * δ) - 1 := by have := Real.add_one_le_exp (2 * δ); linarith
  simp only [smErr]
  nlinarith [mul_nonneg hu (by linarith : (0:ℝ) ≤ 1 + smKappa M.u eexp n)]

/-- **Float softmax at float logits vs real softmax at real logits — within `smErr`.**
    The rounding half (`softmaxF_close`, float-vs-real at the *same* logits) plus the
    logit-perturbation half (`softmax_perturb`, real-vs-real under a coordinatewise
    logit error `δ`), assembled by the triangle inequality. This is the per-row engine
    for attention's softmax (each row's logits are the float scores, off the real scores
    by `δ`); extracted from `softmax_ce_cot_close`'s internal `hsm` (here without the
    onehot subtraction, so it applies to any softmax position, not just the loss head). -/
theorem softmaxF_close_at (fexp : ℝ → ℝ) {eexp δ : ℝ} {n : ℕ}
    (zt z : Vec n) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : smRho M.u eexp n < 1)
    (hδ : ∀ k', |zt k' - z k'| ≤ δ) (k : Fin n) :
    |M.softmaxF fexp zt k - softmax n z k| ≤ smErr M.u eexp δ n := by
  have hA := M.softmaxF_close fexp zt heexp0 heexp1 hfexp hρ1 k
  have hB := softmax_perturb zt z hδ k
  have htri := abs_sub_le (M.softmaxF fexp zt k) (softmax n zt k) (softmax n z k)
  simp only [smErr]
  linarith

/-- `e^x − 1 ≤ x/(1−x)` for `0 ≤ x < 1` — the exp analogue of the γ-form,
    from `1 − x ≤ e^(−x)` alone; keeps the numeric head budget in
    `norm_num` country. -/
theorem exp_sub_one_le {x : ℝ} (hx1 : x < 1) :
    Real.exp x - 1 ≤ x / (1 - x) := by
  have hp := Real.exp_pos x
  have hprod : Real.exp x * Real.exp (-x) = 1 := by
    rw [← Real.exp_add]; simp
  have h1 : (1 - x) * Real.exp x ≤ 1 := by
    nlinarith [Real.add_one_le_exp (-x), hp]
  rw [le_div_iff₀ (by linarith : (0:ℝ) < 1 - x)]
  nlinarith [h1]

/-- **Numeric head budget at the committed MNIST output** (`n = 10`): for
    any model at binary32 accuracy, `exp` accurate to `eexp ≤ 10⁻⁶`
    (GPU `exp` is ~1–2 ULP; the constant is what `vjp_oracle` validates),
    and float logits within `δ = 1/100` of real, the rounded
    softmax−onehot cotangent is within **21/1000** of the certified
    gradient — almost all of it the `e^(2δ) − 1 ≈ 2δ` logit-perturbation
    term; the head's own rounding contributes < 4·10⁻⁶.

    `δ = 1/100` is an a-posteriori-style hypothesis: the *worst-case*
    forward logit budget (≈5100 at trained magnitudes) makes `e^(2δ) − 1`
    vacuous, so a useful head budget needs the measured logit error —
    exactly the hand-off point from worst-case to a-posteriori analysis.
    Empirically validated (`scripts/margin_probe.py`): measured drift on a
    real 12-epoch run is ≤ 1.6·10⁻⁵, 600× inside the `1/100` hypothesis. -/
theorem mnist_cot_budget (hMu : M.u ≤ u32) (fexp : ℝ → ℝ) {eexp : ℝ}
    (heexp0 : 0 ≤ eexp) (heexp : eexp ≤ 1/1000000)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (zt z : Vec 10) (label : Fin 10)
    (hz : ∀ k', |zt k' - z k'| ≤ 1/100) (k : Fin 10) :
    |M.softmaxCECotF fexp zt label k -
      (softmax 10 z k - oneHot 10 label k)| ≤ 21/1000 := by
  have hu := M.u_nonneg
  have hu32 : M.u ≤ 1/16777216 := hMu.trans (by norm_num [u32])
  have hg11 : (1 + M.u) ^ (10 + 1) - 1 ≤ 7/10000000 :=
    M.gamma_num (q := 7/10000000) hMu (by norm_num [u32]) (by norm_num [u32])
  have hG0 : (0:ℝ) ≤ (1 + M.u) ^ (10 + 1) - 1 :=
    sub_nonneg.mpr (M.one_le_pow_one_add_u (10 + 1))
  have hρ : smRho M.u eexp 10 ≤ 18/10000000 := by
    simp only [smRho]
    nlinarith [mul_le_mul hg11 (by linarith : 1 + eexp ≤ 1 + 1/1000000)
      (by linarith : (0:ℝ) ≤ 1 + eexp) (by norm_num : (0:ℝ) ≤ 7/10000000)]
  have hρ0 : 0 ≤ smRho M.u eexp 10 := M.smRho_nonneg heexp0
  have hρ1 : smRho M.u eexp 10 < 1 := lt_of_le_of_lt hρ (by norm_num)
  have hκ : smKappa M.u eexp 10 ≤ 3/1000000 := by
    simp only [smKappa]
    rw [div_le_iff₀ (by linarith)]
    nlinarith
  have hκ0 : 0 ≤ smKappa M.u eexp 10 :=
    div_nonneg (by linarith) (by linarith)
  have hexp : Real.exp (2 * (1/100 : ℝ)) - 1 ≤ 1/49 := by
    rw [show (2:ℝ) * (1/100) = 1/50 from by norm_num]
    exact (exp_sub_one_le (by norm_num)).trans (by norm_num)
  have hsm : smErr M.u eexp (1/100) 10 ≤ 41/2000 := by
    simp only [smErr]
    have h1 : M.u * (1 + smKappa M.u eexp 10) ≤
        (1/16777216) * (1 + 3/1000000) :=
      mul_le_mul hu32 (by linarith) (by linarith) (by norm_num)
    have h2 : (1/16777216 : ℝ) * (1 + 3/1000000) + 3/1000000 + 1/49 ≤
        41/2000 := by norm_num
    linarith
  have hsm0 : 0 ≤ smErr M.u eexp (1/100) 10 := by
    simp only [smErr]
    have hexp1 : 1 ≤ Real.exp (2 * (1/100 : ℝ)) := by
      have := Real.add_one_le_exp (2 * (1/100 : ℝ)); linarith
    nlinarith [mul_nonneg hu (by linarith : (0:ℝ) ≤ 1 + smKappa M.u eexp 10)]
  refine (M.softmax_ce_cot_close fexp zt z label heexp0 (by linarith) hfexp
    hρ1 hz k).trans ?_
  simp only [cotErr]
  have h1 : M.u * (1 + smErr M.u eexp (1/100) 10) ≤
      (1/16777216) * (1 + 41/2000) :=
    mul_le_mul hu32 (by linarith) (by linarith) (by norm_num)
  have h2 : (1/16777216 : ℝ) * (1 + 41/2000) + 41/2000 ≤ 21/1000 := by
    norm_num
  linarith

-- ════════════════════════════════════════════════════════════════
-- § §3c: E4M3 argmax-preservation (the honest depth-1 fp8 statement)
-- ════════════════════════════════════════════════════════════════

/-- **Argmax preservation under a bounded logit perturbation** (planning §3c).
    If every coordinate of the perturbed logits `z'` is within `B` of the
    reference logits `z`, and `z`'s *strict* top-1 margin at `k` exceeds `2B`
    (`z k − z i > 2B` for every other class `i`), then `k` is still the strict
    argmax of `z'`. The depth-1 honest fp8 claim: a `B`-accurate matmul cannot
    flip the prediction on a `>2B`-margin input. `B` is a hypothesis, so the
    statement holds *both* for the proven worst-case bound
    (`dense_close_mixed`) and for any measured a-posteriori drift — the demo's
    empirical `B = max|Δlogit|` plugs into the same theorem. Conditional like
    the suite's quantitative ReLU margins. -/
theorem argmax_preserved {n : ℕ} {z z' : Vec n} {k : Fin n} {B : ℝ}
    (hB : ∀ i, |z' i - z i| ≤ B)
    (hmargin : ∀ i, i ≠ k → 2 * B < z k - z i) :
    ∀ i, i ≠ k → z' i < z' k := by
  intro i hik
  have hk := abs_le.mp (hB k)
  have hi := abs_le.mp (hB i)
  have hm := hmargin i hik
  -- z' k ≥ z k − B and z' i ≤ z i + B, while z k − z i > 2B
  linarith [hk.1, hi.2]

/-- Uniform (magnitude-bounded) per-logit budget of the mixed-precision dense
    layer — `dense_close_mixed` specialized by `∑ᵢ|xᵢWᵢⱼ| ≤ m·w·a`, `|bⱼ| ≤ β`,
    so it is one constant `B` over *all* outputs `j` (the input
    `argmax_preserved` needs). The accumulate `u_acc` rides the bias add and the
    fan-in γ-factor `(1+u_acc)^(m+1)`; the leaf `u_leaf` enters only via the flat
    `(2·u_leaf + u_leaf²)` term — the two-roundoff separation of §1c. -/
noncomputable def denseMixedBudget (uacc uleaf : ℝ) (m : ℕ) (w β a : ℝ) : ℝ :=
  uacc * ((m : ℝ) * w * a + β)
    + (1 + uacc) * ((((1 + uacc) ^ (m + 1) - 1) * (1 + uleaf) ^ 2
        + (2 * uleaf + uleaf ^ 2)) * ((m : ℝ) * w * a))

/-- **Mixed-precision dense forward error, uniform-magnitude budget.** Under
    `|Wᵢⱼ| ≤ w`, `|bⱼ| ≤ β`, `|xᵢ| ≤ a`, every E4M3-mixed logit is within the
    closed-form `denseMixedBudget` of the exact-ℝ logit — evaluable by `norm_num`
    at a concrete net. -/
theorem dense_close_mixed_uniform_budget (L : FloatModel) {m n : ℕ}
    {W : Mat m n} {b : Vec n} {x : Vec m} {w β a : ℝ} (ha : 0 ≤ a)
    (hW : ∀ i j, |W i j| ≤ w) (hb : ∀ j, |b j| ≤ β) (hx : ∀ i, |x i| ≤ a)
    (j : Fin n) :
    |M.denseMixed L W b x j - Proofs.dense W b x j| ≤
      denseMixedBudget M.u L.u m w β a := by
  have hu := M.u_nonneg
  have hbase := M.dense_close_mixed L W b x j
  have hSb : (∑ i, |x i * W i j|) ≤ (m : ℝ) * w * a := by
    calc (∑ i, |x i * W i j|) ≤ ∑ _i : Fin m, a * w := by
          refine Finset.sum_le_sum fun i _ => ?_
          rw [abs_mul]; exact mul_le_mul (hx i) (hW i j) (abs_nonneg _) ha
      _ = (m : ℝ) * (a * w) := by rw [Finset.sum_const]; simp [nsmul_eq_mul]
      _ = (m : ℝ) * w * a := by ring
  have hbabs : |b j| ≤ β := hb j
  set S := ∑ i, |x i * W i j| with hS
  set br := ((1 + M.u) ^ (m + 1) - 1) * (1 + L.u) ^ 2 + (2 * L.u + L.u ^ 2)
    with hbr
  have hbr0 : 0 ≤ br := by
    have h1 : (0 : ℝ) ≤ (1 + M.u) ^ (m + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith))
    have h3 : (0 : ℝ) ≤ 2 * L.u + L.u ^ 2 := by
      have := L.u_nonneg; nlinarith [sq_nonneg L.u]
    rw [hbr]; exact add_nonneg (mul_nonneg h1 (sq_nonneg _)) h3
  have h1u : (0 : ℝ) ≤ 1 + M.u := by linarith
  have step1 : M.u * (S + |b j|) ≤ M.u * ((m : ℝ) * w * a + β) :=
    mul_le_mul_of_nonneg_left (by linarith) hu
  have step2 : (1 + M.u) * (br * S) ≤ (1 + M.u) * (br * ((m : ℝ) * w * a)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hSb hbr0) h1u
  calc |M.denseMixed L W b x j - Proofs.dense W b x j|
      ≤ M.u * (S + |b j|) + (1 + M.u) * (br * S) := hbase
    _ ≤ M.u * ((m : ℝ) * w * a + β) + (1 + M.u) * (br * ((m : ℝ) * w * a)) := by
        linarith [step1, step2]
    _ = denseMixedBudget M.u L.u m w β a := by simp only [denseMixedBudget, hbr]

/-- Monotone upper bound for `denseMixedBudget`, keeping the `(1+uacc)^(m+1)`
    power abstract: replace the accumulate by `U ≥ uacc`, the fan-in γ-factor by
    `g ≥ (1+uacc)^(m+1)−1`, and the two leaf pieces by `P ≥ (1+uleaf)²`,
    `Q ≥ 2·uleaf+uleaf²`. The result has no power left, so a concrete instance
    (e.g. m = 784) evaluates by `norm_num` **without** unfolding the 785-fold
    `npow` — the `layerBudget_le_of` analogue for the two-roundoff budget. -/
theorem denseMixedBudget_le_of {uacc uleaf : ℝ} {m : ℕ} {w β a g P Q U : ℝ}
    (hU : 0 ≤ U) (huacc0 : 0 ≤ uacc) (huleaf0 : 0 ≤ uleaf) (huacc : uacc ≤ U)
    (hw : 0 ≤ w) (hβ : 0 ≤ β) (ha : 0 ≤ a)
    (hg0 : 0 ≤ g) (hg : (1 + uacc) ^ (m + 1) - 1 ≤ g)
    (hP : (1 + uleaf) ^ 2 ≤ P) (hQ : 2 * uleaf + uleaf ^ 2 ≤ Q) :
    denseMixedBudget uacc uleaf m w β a ≤
      U * ((m : ℝ) * w * a + β) + (1 + U) * ((g * P + Q) * ((m : ℝ) * w * a)) := by
  unfold denseMixedBudget
  have hmwa : (0 : ℝ) ≤ (m : ℝ) * w * a :=
    mul_nonneg (mul_nonneg (Nat.cast_nonneg m) hw) ha
  have hxb : (0 : ℝ) ≤ (m : ℝ) * w * a + β := by linarith
  have hpow0 : 0 ≤ (1 + uacc) ^ (m + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hleafpos : 0 ≤ 2 * uleaf + uleaf ^ 2 := by nlinarith [huleaf0, sq_nonneg uleaf]
  have hQ0 : 0 ≤ Q := le_trans hleafpos hQ
  have hbrkpos : 0 ≤ ((1 + uacc) ^ (m + 1) - 1) * (1 + uleaf) ^ 2
      + (2 * uleaf + uleaf ^ 2) :=
    add_nonneg (mul_nonneg hpow0 (sq_nonneg _)) hleafpos
  have hbrk : ((1 + uacc) ^ (m + 1) - 1) * (1 + uleaf) ^ 2 + (2 * uleaf + uleaf ^ 2)
      ≤ g * P + Q := add_le_add (mul_le_mul hg hP (sq_nonneg _) hg0) hQ
  have t1 : uacc * ((m : ℝ) * w * a + β) ≤ U * ((m : ℝ) * w * a + β) :=
    mul_le_mul_of_nonneg_right huacc hxb
  have t2 : (1 + uacc) * ((((1 + uacc) ^ (m + 1) - 1) * (1 + uleaf) ^ 2
        + (2 * uleaf + uleaf ^ 2)) * ((m : ℝ) * w * a))
      ≤ (1 + U) * ((g * P + Q) * ((m : ℝ) * w * a)) :=
    mul_le_mul (by linarith) (mul_le_mul_of_nonneg_right hbrk hmwa)
      (mul_nonneg hbrkpos hmwa) (by linarith)
  linarith [t1, t2]

/-- **The worst-case E4M3 per-logit budget at the MNIST-linear dims** (784→n;
    E4M3 leaf `u_leaf ≤ 2⁻⁴`, fp32 accumulate `u_acc ≤ 2⁻²⁴`; pixels `|x| ≤ 1`,
    trained `|W| ≤ 3/5`, `|b| ≤ 1`): every E4M3-mixed logit is within **61** of
    the exact-ℝ logit. The leaf term `(2·2⁻⁴ ≈ 12.5%)·∑|xW|` dominates (the fp32
    fan-in γ at 784 is ≈5·10⁻⁵, negligible) — this is the *worst-case*, all-errors-
    aligned figure. The demo (`scripts/mnist_e4m3_demo.py`) measures the actual
    drift at `max|Δlogit| = 0.38` (errors cancel), the a-posteriori `B`; both
    feed `argmax_preserved`. -/
theorem linear_e4m3_logit_budget (L : FloatModel) (hMu : M.u ≤ u32)
    (hLu : L.u ≤ u_e4m3) :
    denseMixedBudget M.u L.u 784 (3 / 5) 1 1 ≤ 61 := by
  have hu := M.u_nonneg
  have hLu0 := L.u_nonneg
  have hue : (u_e4m3 : ℝ) = 1 / 16 := by norm_num [u_e4m3]
  rw [hue] at hLu
  -- a clean coarse accumulate bound keeps the assembly out of 2⁻²⁴-land
  have hu6 : M.u ≤ 1 / 1000000 := hMu.trans (by norm_num [u32])
  -- the two flat E4M3 leaf pieces at u_leaf ≤ 1/16 (prove these BEFORE hγ:
  -- nlinarith ring-normalizes every in-scope hypothesis, so a concrete
  -- `(1+M.u)^785` in context would blow up the 785-fold npow)
  have hprodhint : (0 : ℝ) ≤ L.u * (1 / 16 - L.u) := mul_nonneg hLu0 (by linarith)
  have hsq : (1 + L.u) ^ 2 ≤ 289 / 256 := by nlinarith [hLu, hLu0, hprodhint]
  have hleaf : 2 * L.u + L.u ^ 2 ≤ 33 / 256 := by nlinarith [hLu, hLu0, hprodhint]
  -- the fan-in γ at 784 (cheap via gamma_num; no big-power evaluation)
  have hγ : (1 + M.u) ^ (784 + 1) - 1 ≤ 5 / 100000 :=
    M.gamma_num (k := 784 + 1) hMu (by norm_num [u32]) (by norm_num [u32])
  refine (denseMixedBudget_le_of (U := 1 / 1000000) (g := 5 / 100000)
      (P := 289 / 256) (Q := 33 / 256) (by norm_num) hu hLu0 hu6
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) hγ
      hsq hleaf).trans ?_
  norm_num

/-- **Verified E4M3 MNIST-linear argmax preservation** (planning §3c capstone).
    For the certified linear classifier at E4M3 leaf precision / fp32 accumulate,
    pixels `|x| ≤ 1`, trained `|W| ≤ 3/5`, `|b| ≤ 1`: whenever the exact-ℝ logit
    margin at the top class `k` exceeds `2·61 = 122`, the E4M3-mixed forward keeps
    `k` as the strict argmax — **provably the same prediction**. Depth-1 makes
    the single-matmul bound the end-to-end bound, so this is the one realistic
    fp8 case with an honest accuracy guarantee (no vacuous depth compounding).
    The 122 is the worst-case threshold; with the demo's measured `B = 0.38`
    the same `argmax_preserved` covers the `>0.76`-margin inputs — empirically
    92.89% of the MNIST test set (`scripts/mnist_e4m3_demo.py`). fp32 ≈ exact-ℝ
    (within `u_acc`), so the demo's fp32 margins are the relevant quantity. -/
theorem linear_e4m3_argmax_preserved (L : FloatModel) (hMu : M.u ≤ u32)
    (hLu : L.u ≤ u_e4m3) {n : ℕ} {W : Mat 784 n} {b : Vec n} {x : Vec 784}
    (hW : ∀ i j, |W i j| ≤ 3 / 5) (hb : ∀ j, |b j| ≤ 1) (hx : ∀ i, |x i| ≤ 1)
    (k : Fin n)
    (hmargin : ∀ i, i ≠ k →
      (122 : ℝ) < Proofs.dense W b x k - Proofs.dense W b x i) :
    ∀ i, i ≠ k → M.denseMixed L W b x i < M.denseMixed L W b x k := by
  set B := denseMixedBudget M.u L.u 784 (3 / 5) 1 1 with hBdef
  have hB : ∀ j, |M.denseMixed L W b x j - Proofs.dense W b x j| ≤ B := fun j => by
    rw [hBdef]
    exact M.dense_close_mixed_uniform_budget L (a := 1) (by norm_num) hW hb hx j
  have hBle : B ≤ 61 := by rw [hBdef]; exact M.linear_e4m3_logit_budget L hMu hLu
  refine argmax_preserved (z := Proofs.dense W b x) (z' := M.denseMixed L W b x)
    (k := k) (B := B) hB (fun i hik => ?_)
  have := hmargin i hik
  linarith [hBle]

-- ════════════════════════════════════════════════════════════════
-- § Sanity: the exact model inhabits the interface, budgets collapse
-- ════════════════════════════════════════════════════════════════

/-- The exact-arithmetic model: `rnd = id`, `u = 0`. Inhabits the interface
    (the standard model isn't vacuous) and collapses every budget to `0`. -/
def exactModel : FloatModel where
  rnd := id
  u := 0
  u_nonneg := le_rfl
  err := fun x => by simp

@[simp] theorem exactModel_dot : ∀ {n : ℕ} (x y : Vec n),
    exactModel.dot x y = ∑ i, x i * y i := by
  intro n
  induction n with
  | zero => intro x y; rw [exactModel.dot_zero x y]; simp
  | succ n ih =>
    intro x y
    rw [exactModel.dot_succ x y,
        Fin.sum_univ_castSucc (f := fun i => x i * y i),
        ih (fun i => x i.castSucc) (fun i => y i.castSucc)]
    simp [FloatModel.add, FloatModel.mul, exactModel]

@[simp] theorem exactModel_denseErr {m n : Nat} (W : Mat m n) (b : Vec n)
    (xa : Vec m) (j : Fin n) : exactModel.denseErr W b xa 0 j = 0 := by
  simp [FloatModel.denseErr, exactModel]

/-- **`dotMixed` with an exact leaf (`u_leaf = 0`) is the plain rounded dot.**
    The fp32 specialization: no input rounding ⇒ `dot_close_mixed` collapses to
    `dot_close` (the leaf term `2·0 + 0² = 0` vanishes, the leaf-rounded
    magnitudes become the real ones). Confirms the two-roundoff budget is a
    genuine *generalization* of the single-`u` budget, not a reparametrization. -/
@[simp] theorem dotMixed_exact_leaf {n : ℕ} (x y : Vec n) :
    M.dotMixed exactModel x y = M.dot x y := by
  simp [FloatModel.dotMixed, exactModel]

end FloatModel

end Proofs
