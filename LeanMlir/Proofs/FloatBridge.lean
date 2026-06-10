import LeanMlir.Proofs.MLP

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
    If `x̃` is within `e` of the real activation `xa` coordinatewise, then
    `|M.dense W b x̃ j − dense W b xa j| ≤ denseErr W b xa e j`. -/
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

/-- **Linear-net forward extraction (Chapter 2).** The rounded `mnistLinear`
    is within the explicit `denseErr` budget of the real one, per logit. With
    `u = 2⁻²⁴` this is the binary32 forward-error bound for the certified
    linear classifier. -/
theorem linear_float_close {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (j : Fin n) :
    |M.dense W b x j - mnistLinear W b x j| ≤ M.denseErr W b x 0 j :=
  M.dense_close_fresh W b x j

/-- **MLP forward extraction (Chapter 3).** The rounded 3-layer MLP is within
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

end FloatModel

end Proofs
