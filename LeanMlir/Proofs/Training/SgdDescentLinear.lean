import LeanMlir.Proofs.Training.SgdDescent
import LeanMlir.Proofs.Nets.Small.LinearTrainStep
import LeanMlir.Proofs.Float.FloatBridge

/-! # Lipschitz constants for the linear softmax-CE loss

The missing hypothesis of `sgd_descends`, discharged for the Chapter-1 net:
the gradient of `v ↦ crossEntropy(dense(unflatten v, b, x), label)` is
segment-Lipschitz with the **explicit** constant `2a²/(1 − 2aD)` (pixels
bounded by `a`, step `ℓ1`-radius `D`, small-step condition `2aD < 1`).

No Hessian appears. The route is the same elementary ratio argument as
`FloatBridge.lean`'s: the loss gradient is `xᵢ·(softmax(z)ⱼ − onehotⱼ)`
(`linear_loss_gradAt`, assembled from the suite's certified
`lossWeightGrad_eq_sum` + `pdiv_dense_W`), the logits move linearly in the
parameters (`dense_unflatten_drift`), and `FloatModel.softmax_perturb` +
the γ-form `FloatModel.exp_sub_one_le` turn the logit drift into a gradient
drift that is *linear in `t`* along the segment — exactly the shape
`descent_segment` consumes.

`linear_sgd_descends` is the capstone: an `η`-accurate gradient oracle
(e.g. the float budgets), the small-step condition, and the two dominance
conditions ⇒ **one inexact SGD step on the MNIST-linear classifier
provably decreases the cross-entropy loss by ≥ lr·‖∇L‖₂²/2.** Every
hypothesis is checkable arithmetic; smoothness is proven, not assumed. -/

namespace Proofs

open StableHLO

/-- `gradAt` agrees with the suite's `Vec 1`-codomain `pdiv` convention. -/
theorem gradAt_eq_pdiv {p : Nat} (f : Vec p → ℝ) (v : Vec p)
    (hf : DifferentiableAt ℝ f v) (idx : Fin p) :
    gradAt f v idx = pdiv (fun w => fun _ : Fin 1 => f w) v idx 0 := by
  unfold gradAt pdiv
  rw [fderiv_pi (fun _ => hf)]
  rfl

/-- **Closed form of the linear softmax-CE loss gradient at any parameter
    point**: `∂L/∂W_{ij} = xᵢ·(softmax(z)ⱼ − onehotⱼ)` — the suite's
    certified contraction (`lossWeightGrad_eq_sum` + `pdiv_dense_W`),
    re-expressed through `gradAt`. -/
theorem linear_loss_gradAt {m n : Nat} (b : Vec n) (x : Vec m)
    (label : Fin n) (v : Vec (m * n)) (i : Fin m) (j : Fin n) :
    gradAt (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label) v
        (finProdFinEquiv (i, j)) =
      x i * (softmax n (dense (Mat.unflatten v) b x) j -
        oneHot n label j) := by
  have hdiff : DifferentiableAt ℝ
      (fun w : Vec (m * n) =>
        crossEntropy n (dense (Mat.unflatten w) b x) label) v :=
    ((crossEntropy_differentiable n label).comp
      (denseWeightMap_differentiable b x)).differentiableAt
  calc gradAt (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        v (finProdFinEquiv (i, j))
      = pdiv (fun w => fun _ : Fin 1 =>
          crossEntropy n (dense (Mat.unflatten w) b x) label) v
          (finProdFinEquiv (i, j)) 0 := gradAt_eq_pdiv _ _ hdiff _
    _ = pdiv (fun w => fun _ : Fin 1 =>
          crossEntropy n (dense (Mat.unflatten w) b x) label)
          (Mat.flatten (Mat.unflatten v)) (finProdFinEquiv (i, j)) 0 := by
        rw [Mat.flatten_unflatten]
    _ = ∑ k : Fin n,
          pdiv (fun w : Vec (m * n) => dense (Mat.unflatten w) b x)
            (Mat.flatten (Mat.unflatten v)) (finProdFinEquiv (i, j)) k
          * (softmax n (mnistLinear (Mat.unflatten v) b x) k -
              oneHot n label k) :=
        lossWeightGrad_eq_sum (W := Mat.unflatten v) (b := b) (x := x)
          label i j
    _ = ∑ k : Fin n, (if k = j then x i else 0) *
          (softmax n (dense (Mat.unflatten v) b x) k - oneHot n label k) :=
        Finset.sum_congr rfl fun k _ => by
          rw [pdiv_dense_W b x (Mat.unflatten v) i j k]
          rfl
    _ = x i * (softmax n (dense (Mat.unflatten v) b x) j -
          oneHot n label j) := by
        simp only [ite_mul, zero_mul]
        rw [Finset.sum_ite_eq']
        simp

/-- The `ℓ1` mass of a scaled step. -/
theorem smul_l1_mass {n : Nat} (e : Vec n) {t : ℝ} (ht0 : 0 ≤ t) :
    (∑ idx, |(t • e) idx|) = t * ∑ idx, |e idx| := by
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun idx _ => by
    simp [abs_mul, abs_of_nonneg ht0]

/-- A `t`-scaled step stays inside the step radius for `t ∈ [0,1]`. -/
theorem smul_l1_mass_le {n : Nat} (e : Vec n) {t D : ℝ} (ht0 : 0 ≤ t)
    (ht1 : t ≤ 1) (he : (∑ idx, |e idx|) ≤ D) :
    (∑ idx, |(t • e) idx|) ≤ D := by
  rw [smul_l1_mass e ht0]
  calc t * ∑ idx, |e idx|
      ≤ 1 * D := mul_le_mul ht1 he
        (Finset.sum_nonneg fun _ _ => abs_nonneg _) zero_le_one
    _ = D := one_mul D

/-- The dense pre-activation difference under a weight perturbation, exactly:
    column `j` only sees the column-`j` slice of the perturbation. -/
theorem dense_unflatten_diff {m n : Nat} (b : Vec n) (x : Vec m)
    (v e : Vec (m * n)) (j : Fin n) :
    dense (Mat.unflatten (v + e)) b x j - dense (Mat.unflatten v) b x j =
      ∑ i, x i * e (finProdFinEquiv (i, j)) := by
  simp only [dense, Mat.unflatten, Pi.add_apply, add_sub_add_right_eq_sub,
    ← Finset.sum_sub_distrib, mul_add, add_sub_cancel_left]

/-- Column-refined drift: the column-`j` pre-activation moves by at most
    `a` times the column-`j` `ℓ1` mass (not the total mass — this is what
    keeps the hidden-layer Lipschitz constant width-free). -/
theorem dense_unflatten_col_drift {m n : Nat} (b : Vec n) (x : Vec m)
    {a : ℝ} (hx : ∀ i, |x i| ≤ a) (v e : Vec (m * n)) (j : Fin n) :
    |dense (Mat.unflatten (v + e)) b x j - dense (Mat.unflatten v) b x j| ≤
      a * ∑ i, |e (finProdFinEquiv (i, j))| := by
  rw [dense_unflatten_diff]
  calc |∑ i, x i * e (finProdFinEquiv (i, j))|
      ≤ ∑ i, |x i * e (finProdFinEquiv (i, j))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, a * |e (finProdFinEquiv (i, j))| :=
        Finset.sum_le_sum fun i _ => by
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_right (hx i) (abs_nonneg _)
    _ = a * ∑ i, |e (finProdFinEquiv (i, j))| := by rw [Finset.mul_sum]

/-- The logits move linearly in the parameters: a parameter perturbation of
    `ℓ1` mass `‖d‖₁` moves every logit by at most `a·‖d‖₁`. -/
theorem dense_unflatten_drift {m n : Nat} (b : Vec n) (x : Vec m)
    {a : ℝ} (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a)
    (v d : Vec (m * n)) (k : Fin n) :
    |dense (Mat.unflatten (v + d)) b x k - dense (Mat.unflatten v) b x k| ≤
      a * ∑ idx, |d idx| := by
  refine (dense_unflatten_col_drift b x hx v d k).trans (mul_le_mul_of_nonneg_left ?_ ha)
  rw [sum_finProdFinEquiv fun idx => |d idx|, Finset.sum_comm]
  exact Finset.single_le_sum (f := fun j => ∑ i : Fin m, |d (finProdFinEquiv (i, j))|)
    (fun _ _ => by positivity) (Finset.mem_univ k)

/-- **Softmax drift along a segment.** Logits that move by at most `t·δ` (`t ∈ [0, 1]`,
    `2δ < 1`) move every softmax output by at most `2tδ/(1−2δ)`: `softmax_perturb`'s
    `e^(2tδ) − 1`, the γ-form `exp_sub_one_le`, then `t ≤ 1` in the denominator. The
    linear, MLP and CNN segment-Lipschitz lemmas all end in this step. -/
theorem softmax_seg_drift {n : Nat} (zt z : Vec n) {t δ : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (hδ0 : 0 ≤ δ) (hsmall : 2 * δ < 1) (hz : ∀ k, |zt k - z k| ≤ t * δ) (k : Fin n) :
    |softmax n zt k - softmax n z k| ≤ 2 * (t * δ) / (1 - 2 * δ) := by
  have htδ : t * δ ≤ δ := mul_le_of_le_one_left hδ0 ht1
  refine (FloatModel.softmax_perturb zt z hz k).trans
    ((FloatModel.exp_sub_one_le (by linarith)).trans ?_)
  exact div_le_div_of_nonneg_left (mul_nonneg zero_le_two (mul_nonneg ht0 hδ0))
    (by linarith) (by linarith)

/-- **Segment-Lipschitz gradient for the linear softmax-CE loss, explicit
    constant.** Under the small-step condition `2aD < 1`, the gradient
    entries drift by at most `(2a²/(1−2aD))·(t·D)` along `[v, v+d]` — the
    exact shape `descent_segment` consumes. The exponential softmax
    perturbation is linearized by the γ-form, not the mean value theorem. -/
theorem linear_loss_grad_lipschitz {m n : Nat} (b : Vec n) (x : Vec m)
    (label : Fin n) {a D : ℝ} (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a)
    (v d : Vec (m * n)) (hd : (∑ idx, |d idx|) ≤ D)
    (hsmall : 2 * (a * D) < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1) (idx : Fin (m * n)) :
    |gradAt (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (v + t • d) idx -
      gradAt (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        v idx| ≤ (2 * a ^ 2 / (1 - 2 * (a * D))) * (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D :=
    le_trans (Finset.sum_nonneg fun idx _ => abs_nonneg _) hd
  have haD0 : 0 ≤ a * D := mul_nonneg ha hD0
  obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
  rw [linear_loss_gradAt, linear_loss_gradAt]
  -- the gradient drift is |xᵢ| times the softmax drift
  have hgd : x i * (softmax n (dense (Mat.unflatten (v + t • d)) b x) j -
        oneHot n label j) -
      x i * (softmax n (dense (Mat.unflatten v) b x) j - oneHot n label j) =
      x i * (softmax n (dense (Mat.unflatten (v + t • d)) b x) j -
        softmax n (dense (Mat.unflatten v) b x) j) := by ring
  rw [hgd, abs_mul]
  -- logit drift along the segment: ≤ t·(a·D)
  have hz : ∀ k, |dense (Mat.unflatten (v + t • d)) b x k -
      dense (Mat.unflatten v) b x k| ≤ t * (a * D) := by
    intro k
    have h1 := dense_unflatten_drift b x ha hx v (t • d) k
    rw [smul_l1_mass d ht0] at h1
    have h3 : a * (t * ∑ idx, |d idx|) ≤ t * (a * D) := by
      nlinarith [mul_le_mul_of_nonneg_left hd (mul_nonneg ht0 ha)]
    linarith
  -- softmax drift via the ratio sandwich + γ-form linearization
  have hsmle := softmax_seg_drift _ _ ht0 ht1 haD0 hsmall hz j
  calc |x i| * |softmax n (dense (Mat.unflatten (v + t • d)) b x) j -
        softmax n (dense (Mat.unflatten v) b x) j|
      ≤ a * (2 * (t * (a * D)) / (1 - 2 * (a * D))) :=
        mul_le_mul (hx i) hsmle (abs_nonneg _) ha
    _ = (2 * a ^ 2 / (1 - 2 * (a * D))) * (t * D) := by ring

/-- **One inexact SGD step on the MNIST-linear classifier provably
    decreases the cross-entropy loss.** All of `sgd_descends`' hypotheses
    discharged for the Chapter-1 net: differentiability is
    `lossWeightMap_differentiable`, the segment-Lipschitz constant is the
    explicit `C = 2a²/(1−2aD)` at step radius `D = lr·(‖∇L‖₁ + mn·η)`.
    Remaining hypotheses are checkable arithmetic: the oracle accuracy `η`
    (supplied by the float budgets), the small-step condition, and the two
    dominance conditions. Conclusion: the loss drops by ≥ `lr·‖∇L‖₂²/2`. -/
theorem linear_sgd_descends {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (label : Fin n) (gh : Vec (m * n)) {lr η a : ℝ}
    (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a) (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx -
      gradAt (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (Mat.flatten W) idx| ≤ η)
    (hsmall : 2 * (a * (lr * ((∑ idx, |gradAt
        (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) * η))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt
        (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (Mat.flatten W) idx|) ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (Mat.flatten W) idx ^ 2) / 4)
    (h2 : (2 * a ^ 2 / (1 - 2 * (a * (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
          (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) * η))))) *
        (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
          (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) * η)) ^ 2 ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (Mat.flatten W) idx ^ 2) / 4) :
    crossEntropy n (dense (Mat.unflatten (Mat.flatten W - lr • gh)) b x)
        label ≤
      crossEntropy n (dense (Mat.unflatten (Mat.flatten W)) b x) label -
        lr * (∑ idx, gradAt
          (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
          (Mat.flatten W) idx ^ 2) / 2 := by
  set f : Vec (m * n) → ℝ :=
    fun w => crossEntropy n (dense (Mat.unflatten w) b x) label with hf
  -- the Lipschitz constant at the step radius
  have hC0 : (0:ℝ) ≤ 2 * a ^ 2 / (1 - 2 * (a * (lr * ((∑ idx, |gradAt f
      (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) * η)))) := by
    refine div_nonneg (by positivity) ?_
    linarith
  -- everywhere differentiable (the loss is smooth in the parameters)
  have hdiffall : ∀ w : Vec (m * n), DifferentiableAt ℝ f w := fun w =>
    ((crossEntropy_differentiable n label).comp
      (denseWeightMap_differentiable b x)).differentiableAt
  -- ℓ1 radius of the step
  have hD : (∑ idx, |(-(lr • gh)) idx|) ≤
      lr * ((∑ idx, |gradAt f (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) * η) :=
    sgd_step_l1_le _ gh hlr hgh
  have hmain := sgd_descends f (Mat.flatten W) gh hlr hη hC0 hgh
    (fun t _ => hdiffall _)
    (fun t ht idx => by
      have := linear_loss_grad_lipschitz b x label ha hx (Mat.flatten W)
        (-(lr • gh)) hD hsmall t ht idx
      simpa [hf] using this)
    h1 h2
  simpa [hf] using hmain

-- ════════════════════════════════════════════════════════════════
-- § Item D: the η-composition — feed the FloatBridge budget into the
--   descent η-slot, so "one binary32 SGD step decreases the loss" holds
--   with NO abstract gradient-accuracy parameter.
-- ════════════════════════════════════════════════════════════════

/-- **The binary32 gradient of the MNIST-linear loss**, exactly as the
    rendered trainer computes it: float forward logits `z̃ = M.dense W b x`,
    the rounded softmax−onehot cotangent head, and one final rounded
    multiply by the (exact) input `xᵢ` to form the outer-product weight
    gradient `∂L/∂Wᵢⱼ = xᵢ·(softmax(z)ⱼ − onehotⱼ)`. Flattened to the
    `Vec (m*n)` parameter layout that `gradAt`/`linear_sgd_descends` use. -/
noncomputable def FloatModel.linearFloatGrad (M : FloatModel) {m n : Nat}
    (W : Mat m n) (b : Vec n) (x : Vec m) (fexp : ℝ → ℝ) (label : Fin n) :
    Vec (m * n) :=
  Mat.flatten fun i j =>
    M.mul (x i) (M.softmaxCECotF fexp (M.dense W b x) label j)

@[simp] theorem linearFloatGrad_apply (M : FloatModel) {m n : Nat}
    (W : Mat m n) (b : Vec n) (x : Vec m) (fexp : ℝ → ℝ) (label : Fin n)
    (i : Fin m) (j : Fin n) :
    M.linearFloatGrad W b x fexp label (finProdFinEquiv (i, j)) =
      M.mul (x i) (M.softmaxCECotF fexp (M.dense W b x) label j) := by
  simp [FloatModel.linearFloatGrad, Mat.flatten, Equiv.symm_apply_apply]

/-- **The binary32 gradient is within `mulErr u a 1 0 (cotErr …)` of the
    certified real gradient**, per entry. The head accuracy is the existing
    `softmax_ce_cot_close` (`cotErr`); the final input-multiply is one
    `mul_close` with an *exact* left operand (`ea = 0`) bounded by `a`, and a
    right operand `softmax−onehot ∈ [−1,1]` (`C = 1`). This is the bridge
    that discharges `linear_sgd_descends`' abstract `η`. -/
theorem linear_grad_close {m n : Nat} (M : FloatModel) (W : Mat m n)
    (b : Vec n) (x : Vec m) (label : Fin n) (fexp : ℝ → ℝ) {eexp δ a : ℝ}
    (hx : ∀ i, |x i| ≤ a)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp n < 1)
    (hδ : ∀ k', |M.dense W b x k' - dense W b x k'| ≤ δ)
    (i : Fin m) (j : Fin n) :
    |M.linearFloatGrad W b x fexp label (finProdFinEquiv (i, j)) -
        gradAt (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
          (Mat.flatten W) (finProdFinEquiv (i, j))| ≤
      FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ n) := by
  rw [linearFloatGrad_apply, linear_loss_gradAt, Mat.unflatten_flatten]
  -- head accuracy: rounded cotangent within `cotErr` of `softmax − onehot`
  have hcot := M.softmax_ce_cot_close fexp (M.dense W b x) (dense W b x)
    label heexp0 heexp1 hfexp hρ1 hδ j
  -- `softmax − onehot ∈ [−1, 1]`
  have hy := abs_softmax_sub_oneHot_le_one (dense W b x) label j
  -- the input multiply: exact left operand (`|xᵢ − xᵢ| = 0 ≤ 0`)
  have hxx : |x i - x i| ≤ (0:ℝ) := by simp
  exact M.mul_close hxx hcot (hx i) hy

/-- **One binary32 SGD step on the MNIST-linear classifier provably
    decreases the cross-entropy loss — with NO abstract gradient-accuracy
    parameter.** This is Item D / G1, the η-composition: the descent side
    (`linear_sgd_descends`) and the rounding side (FloatBridge's
    `cotErr`/`mulErr` head budget) are fused into one statement. The
    gradient `gh` is the *actual* float-computed gradient
    (`M.linearFloatGrad`), and its accuracy `η = mulErr u a 1 0 (cotErr …)`
    is *proven* by `linear_grad_close`, not assumed.

    What remains as hypotheses is exactly the honest residue: the input
    bound `a`, `0 ≤ lr`, the GPU `exp` accuracy `eexp` and the a-posteriori
    logit drift `δ` (the documented FloatModel → kernel trust boundary,
    `softmax_ce_cot_close`), and the checkable-arithmetic small-step + two
    dominance conditions. Depth-1 means there is no per-layer η-threading —
    the clean pilot for the chain `binary32 → proximity → smoothness →
    descent`, closed end-to-end for one net. -/
theorem linear_float_sgd_descends {m n : Nat} (M : FloatModel) (W : Mat m n)
    (b : Vec n) (x : Vec m) (label : Fin n) (fexp : ℝ → ℝ) {lr a eexp δ : ℝ}
    (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a) (hlr : 0 ≤ lr)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1) (hδ0 : 0 ≤ δ)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp n < 1)
    (hδ : ∀ k', |M.dense W b x k' - dense W b x k'| ≤ δ)
    (hsmall : 2 * (a * (lr * ((∑ idx, |gradAt
        (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) *
          FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ n)))) < 1)
    (h1 : lr * (FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ n)) *
        (∑ idx, |gradAt
          (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
          (Mat.flatten W) idx|) ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (Mat.flatten W) idx ^ 2) / 4)
    (h2 : (2 * a ^ 2 / (1 - 2 * (a * (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
          (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) *
            FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ n)))))) *
        (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
          (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) *
            FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ n))) ^ 2 ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (Mat.flatten W) idx ^ 2) / 4) :
    crossEntropy n (dense (Mat.unflatten (Mat.flatten W -
        lr • M.linearFloatGrad W b x fexp label)) b x) label ≤
      crossEntropy n (dense (Mat.unflatten (Mat.flatten W)) b x) label -
        lr * (∑ idx, gradAt
          (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
          (Mat.flatten W) idx ^ 2) / 2 := by
  -- the head budget is nonnegative (it bounds an absolute value)
  have hu := M.u_nonneg
  have hcot0 := M.cotErr_nonneg heexp0 hδ0 hρ1
  have hηF0 : 0 ≤ FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ n) := by
    have e1 : (0:ℝ) ≤ M.u * ((a + 0) * (1 + FloatModel.cotErr M.u eexp δ n)) :=
      mul_nonneg hu (mul_nonneg (by linarith) (by linarith))
    have e2 : (0:ℝ) ≤ a * FloatModel.cotErr M.u eexp δ n := mul_nonneg ha hcot0
    simp only [FloatModel.mulErr]
    nlinarith [e1, e2]
  -- discharge the abstract gradient-accuracy hypothesis by `linear_grad_close`
  have hgh : ∀ idx, |M.linearFloatGrad W b x fexp label idx -
      gradAt (fun w => crossEntropy n (dense (Mat.unflatten w) b x) label)
        (Mat.flatten W) idx| ≤
      FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ n) := by
    intro idx
    obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
    exact linear_grad_close M W b x label fexp hx heexp0 heexp1 hfexp hρ1
      hδ i j
  exact linear_sgd_descends W b x label (M.linearFloatGrad W b x fexp label)
    ha hx hlr hηF0 hgh hsmall h1 h2
