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

/-- **Numeric capstone at the committed MNIST-MLP dims** (the
    `MainMnistMlpTrain.lean` net: 784→512→512→10). For any rounding model at
    binary32 accuracy (`u ≤ 2⁻²⁴`), weights bounded by `1/32` and biases and
    pixels by `1`, every rounded logit is within **3/4** of the exact-real
    logit — while the logits themselves can reach ≈6.5·10³, i.e. ≈10⁻⁴
    relative. All three layer budgets discharge by `norm_num` through the
    γ-form (`pow_gamma_bound`); no big-power evaluation.

    The dominant term is the Lipschitz amplification of the layer-0 budget
    (`16·e` per layer), not fresh rounding — the worst-case-composition
    blow-up that makes a-posteriori/probabilistic analysis the right tool
    past toy depth. -/
theorem mnist_mlp_float_budget (hMu : M.u ≤ u32)
    (W₀ : Mat 784 512) (b₀ : Vec 512) (W₁ : Mat 512 512) (b₁ : Vec 512)
    (W₂ : Mat 512 10) (b₂ : Vec 10) (x : Vec 784)
    (hW₀ : ∀ i j, |W₀ i j| ≤ 1/32) (hb₀ : ∀ j, |b₀ j| ≤ 1)
    (hW₁ : ∀ i j, |W₁ i j| ≤ 1/32) (hb₁ : ∀ j, |b₁ j| ≤ 1)
    (hW₂ : ∀ i j, |W₂ i j| ≤ 1/32) (hb₂ : ∀ j, |b₂ j| ≤ 1)
    (hx : ∀ i, |x i| ≤ 1) (k : Fin 10) :
    |M.mlpF W₀ b₀ W₁ b₁ W₂ b₂ x k -
        Proofs.dense W₂ b₂ (relu 512 (Proofs.dense W₁ b₁
          (relu 512 (Proofs.dense W₀ b₀ x)))) k| ≤ 3/4 := by
  have hu := M.u_nonneg
  -- γ-form bounds at the two layer exponents, monotone through u ≤ u32
  have hgam : ∀ k : ℕ, (k : ℝ) * u32 < 1 →
      (1 + M.u) ^ k - 1 ≤ (k : ℝ) * u32 / (1 - (k : ℝ) * u32) := by
    intro k hk
    have hkM : (k : ℝ) * M.u ≤ (k : ℝ) * u32 :=
      mul_le_mul_of_nonneg_left hMu (Nat.cast_nonneg k)
    exact (pow_gamma_bound M.u hu k (lt_of_le_of_lt hkM hk)).trans
      (div_one_sub_mono hkM hk)
  have hg786 : (1 + M.u) ^ 786 - 1 ≤ 47 / 1000000 := by
    refine (hgam 786 (by norm_num [u32])).trans ?_
    norm_num [u32]
  have hg514 : (1 + M.u) ^ 514 - 1 ≤ 31 / 1000000 := by
    refine (hgam 514 (by norm_num [u32])).trans ?_
    norm_num [u32]
  -- the closed-form budgets, bounded layer by layer
  have hB₀ : layerBudget M.u 784 (1/32) 1 1 0 ≤ 6/5000 := by
    refine (layerBudget_le_of hu (by norm_num) (by norm_num) (by norm_num)
      hg786 le_rfl le_rfl).trans ?_
    norm_num
  have hB₀0 : (0:ℝ) ≤ layerBudget M.u 784 (1/32) 1 1 0 :=
    layerBudget_nonneg hu (by norm_num) (by norm_num) (by norm_num) le_rfl
  have hB₁ : layerBudget M.u 512 (1/32) 1 (51/2)
      (layerBudget M.u 784 (1/32) 1 1 0) ≤ 4/125 := by
    refine (layerBudget_le_of hu (by norm_num) (by norm_num) (by norm_num)
      hg514 hB₀0 hB₀).trans ?_
    norm_num
  have hB₁0 : (0:ℝ) ≤ layerBudget M.u 512 (1/32) 1 (51/2)
      (layerBudget M.u 784 (1/32) 1 1 0) :=
    layerBudget_nonneg hu (by norm_num) (by norm_num) (by norm_num) hB₀0
  have hB₂ : layerBudget M.u 512 (1/32) 1 409
      (layerBudget M.u 512 (1/32) 1 (51/2)
        (layerBudget M.u 784 (1/32) 1 1 0)) ≤ 3/4 := by
    refine (layerBudget_le_of hu (by norm_num) (by norm_num) (by norm_num)
      hg514 hB₁0 hB₁).trans ?_
    norm_num
  -- assemble: the uniform capstone, activation constants evaluated
  have hmain := M.mlp_float_close_uniform
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
    hW₀ hb₀ hW₁ hb₁ hW₂ hb₂ hx k
  rw [show layerAct 784 (1/32) 1 1 = (51/2 : ℝ) by norm_num [layerAct],
      show layerAct 512 (1/32) 1 (51/2) = (409 : ℝ) by norm_num [layerAct]]
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
