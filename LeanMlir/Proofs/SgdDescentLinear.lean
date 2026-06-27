import LeanMlir.Proofs.SgdDescent
import LeanMlir.Proofs.LinearTrainStep
import LeanMlir.Proofs.FloatBridge

/-! # Lipschitz constants for the linear softmax-CE loss

The missing hypothesis of `sgd_descends`, discharged for the Chapter-1 net:
the gradient of `v ↦ crossEntropy(dense(unflatten v, b, x), label)` is
segment-Lipschitz with the **explicit** constant `2a²/(1 − 2aD)` (pixels
bounded by `a`, step `ℓ1`-radius `D`, small-step condition `2aD < 1`).

No Hessian appears. The route is the same elementary ratio argument as the
float bridge: the loss gradient is `xᵢ·(softmax(z)ⱼ − onehotⱼ)`
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

/-- The logits move linearly in the parameters: a parameter perturbation of
    `ℓ1` mass `‖d‖₁` moves every logit by at most `a·‖d‖₁`. -/
theorem dense_unflatten_drift {m n : Nat} (b : Vec n) (x : Vec m)
    {a : ℝ} (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a)
    (v d : Vec (m * n)) (k : Fin n) :
    |dense (Mat.unflatten (v + d)) b x k - dense (Mat.unflatten v) b x k| ≤
      a * ∑ idx, |d idx| := by
  have h1 : dense (Mat.unflatten (v + d)) b x k -
      dense (Mat.unflatten v) b x k =
      ∑ i, x i * d (finProdFinEquiv (i, k)) := by
    have h2 : (∑ i : Fin m,
        x i * (v (finProdFinEquiv (i, k)) + d (finProdFinEquiv (i, k)))) -
        (∑ i : Fin m, x i * v (finProdFinEquiv (i, k))) =
        ∑ i : Fin m, x i * d (finProdFinEquiv (i, k)) := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    show ((∑ i : Fin m, x i * (v + d) (finProdFinEquiv (i, k))) + b k) -
        ((∑ i : Fin m, x i * v (finProdFinEquiv (i, k))) + b k) = _
    simp only [Pi.add_apply]
    linarith [h2]
  rw [h1]
  have hinj : (∑ i : Fin m, |d (finProdFinEquiv (i, k))|) ≤
      ∑ idx, |d idx| := by
    have himg : ∑ idx ∈ Finset.univ.image
        (fun i : Fin m => finProdFinEquiv (i, k)), |d idx| =
        ∑ i : Fin m, |d (finProdFinEquiv (i, k))| :=
      Finset.sum_image fun i _ i' _ h =>
        (Prod.ext_iff.mp (finProdFinEquiv.injective h)).1
    rw [← himg]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
      (fun idx _ _ => abs_nonneg _)
  calc |∑ i, x i * d (finProdFinEquiv (i, k))|
      ≤ ∑ i, |x i * d (finProdFinEquiv (i, k))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, a * |d (finProdFinEquiv (i, k))| :=
        Finset.sum_le_sum fun i _ => by
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_right (hx i) (abs_nonneg _)
    _ = a * ∑ i, |d (finProdFinEquiv (i, k))| := by rw [Finset.mul_sum]
    _ ≤ a * ∑ idx, |d idx| := mul_le_mul_of_nonneg_left hinj ha

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
    have h2 : (∑ idx, |(t • d) idx|) = t * ∑ idx, |d idx| := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun idx _ => ?_
      simp [abs_mul, abs_of_nonneg ht0]
    rw [h2] at h1
    have h3 : a * (t * ∑ idx, |d idx|) ≤ t * (a * D) := by
      nlinarith [mul_le_mul_of_nonneg_left hd (mul_nonneg ht0 ha)]
    linarith
  -- softmax drift via the ratio sandwich + γ-form linearization
  have hsm := FloatModel.softmax_perturb
    (dense (Mat.unflatten (v + t • d)) b x)
    (dense (Mat.unflatten v) b x) hz j
  have htaD : 2 * (t * (a * D)) < 1 := by nlinarith
  have hexp : Real.exp (2 * (t * (a * D))) - 1 ≤
      2 * (t * (a * D)) / (1 - 2 * (t * (a * D))) :=
    FloatModel.exp_sub_one_le htaD
  have hden : 2 * (t * (a * D)) / (1 - 2 * (t * (a * D))) ≤
      2 * (t * (a * D)) / (1 - 2 * (a * D)) := by
    refine div_le_div_of_nonneg_left (by nlinarith) (by linarith) ?_
    nlinarith
  have hsmle : |softmax n (dense (Mat.unflatten (v + t • d)) b x) j -
      softmax n (dense (Mat.unflatten v) b x) j| ≤
      2 * (t * (a * D)) / (1 - 2 * (a * D)) := by linarith
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
      lr * ((∑ idx, |gradAt f (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) * η) := by
    calc (∑ idx, |(-(lr • gh)) idx|) = ∑ idx, lr * |gh idx| := by
          refine Finset.sum_congr rfl fun idx _ => ?_
          simp [abs_mul, abs_of_nonneg hlr]
      _ ≤ ∑ idx, lr * (|gradAt f (Mat.flatten W) idx| + η) := by
          refine Finset.sum_le_sum fun idx _ => ?_
          refine mul_le_mul_of_nonneg_left ?_ hlr
          have h3 : |gh idx| ≤ |gh idx - gradAt f (Mat.flatten W) idx| +
              |gradAt f (Mat.flatten W) idx| := by
            simpa using abs_sub_le (gh idx) (gradAt f (Mat.flatten W) idx) 0
          linarith [hgh idx]
      _ = lr * ((∑ idx, |gradAt f (Mat.flatten W) idx|) + ((m * n : ℕ) : ℝ) * η) := by
          rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const,
            Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
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
  have hs0 : 0 ≤ softmax n (dense W b x) j :=
    div_nonneg (Real.exp_pos _).le
      (Finset.sum_nonneg fun k _ => (Real.exp_pos _).le)
  have hs1 : softmax n (dense W b x) j ≤ 1 := by
    have hD : 0 < ∑ k, Real.exp (dense W b x k) :=
      Finset.sum_pos (fun k _ => Real.exp_pos _) ⟨j, Finset.mem_univ j⟩
    exact (div_le_one hD).mpr
      (Finset.single_le_sum (fun k _ => (Real.exp_pos _).le)
        (Finset.mem_univ j))
  have hy : |softmax n (dense W b x) j - oneHot n label j| ≤ 1 := by
    simp only [oneHot]
    by_cases h : j = label
    · rw [if_pos h, abs_le]; constructor <;> linarith
    · rw [if_neg h, abs_le]; constructor <;> linarith
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
  have hcot0 : 0 ≤ FloatModel.cotErr M.u eexp δ n := by
    have hpow : (1:ℝ) ≤ (1 + M.u) ^ (n + 1) := one_le_pow₀ (by linarith)
    have hρ0 : 0 ≤ FloatModel.smRho M.u eexp n := by
      simp only [FloatModel.smRho]
      have := mul_nonneg (by linarith : (0:ℝ) ≤ (1 + M.u) ^ (n + 1) - 1)
        (by linarith : (0:ℝ) ≤ 1 + eexp)
      linarith
    have hκ0 : 0 ≤ FloatModel.smKappa M.u eexp n := by
      simp only [FloatModel.smKappa]
      exact div_nonneg (by linarith) (by linarith)
    have hexp0 : 0 ≤ Real.exp (2 * δ) - 1 := by
      have := Real.add_one_le_exp (2 * δ); linarith
    have hsm0 : 0 ≤ FloatModel.smErr M.u eexp δ n := by
      simp only [FloatModel.smErr]
      have := mul_nonneg hu
        (by linarith : (0:ℝ) ≤ 1 + FloatModel.smKappa M.u eexp n)
      linarith
    simp only [FloatModel.cotErr]
    have := mul_nonneg hu
      (by linarith : (0:ℝ) ≤ 1 + FloatModel.smErr M.u eexp δ n)
    linarith
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
