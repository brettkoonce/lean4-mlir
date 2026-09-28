import LeanMlir.Proofs.Training.SgdDescent.Mlp

/-! # Descent on the dense biases — the MLP rungs' bias columns

`SgdDescent.Linear` and `SgdDescent.Mlp` state one-step descent for each dense WEIGHT matrix of the
Chapter-2 MLP (`dense → relu → dense → relu → dense`). This file states it for each dense BIAS:

* **output bias `b₂`** — `linear_bias_sgd_descends`, constant `2/(1 − 2D)`;
* **hidden bias `b₁`** — `mlp_hidden_bias_sgd_descends`, one frozen mask, constant
  `2·d₃·w₂²/(1 − 2·w₂·D)`;
* **input bias `b₀`** — `mlp_input_bias_sgd_descends`, two frozen masks, constant
  `2·d₃·d₂²·w₁²·w₂²/(1 − 2·w₂·d₂·w₁·D)`.

Each is its weight rung with the layer input replaced by the constant `1`: the bias moves its
pre-activation by exactly the step (`dense_bias_drift`, no input bound `a`), so the weight rung's
`a` becomes `1` in the margins and the constants. The segment-Lipschitz step is
`MlpSlot.loss_grad_lipschitz` at a bias map (`σ = ρ = 1` for the hidden layer, `σ = w₁`,
`ρ = d₂·w₁` for the input layer), the gradient's row the channel indicator (`pdiv_dense_b`).

The rungs are generic in the layer's input, so the Chapter-3 CNN's dense-head biases are literal
instances at the pooled activation, as its head weights are of the weight rungs
(`SgdDescent.Cnn`). The oracle accuracy, the margins, the small-step and the two dominance
conditions remain hypotheses; no binary32 twin is stated for the biases.
-/

namespace Proofs

open StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The bias map: drift is exactly the step
-- ════════════════════════════════════════════════════════════════

/-- A bias step moves each pre-activation entry by exactly its own coordinate. -/
theorem dense_bias_diff {m n : Nat} (W : Mat m n) (x : Vec m) (b e : Vec n) (k : Fin n) :
    dense W (b + e) x k - dense W b x k = e k := by
  simp only [dense, Pi.add_apply]; ring

/-- …so by at most the step's `ℓ1` mass. -/
theorem dense_bias_drift {m n : Nat} (W : Mat m n) (x : Vec m) (b e : Vec n) (k : Fin n) :
    |dense W (b + e) x k - dense W b x k| ≤ 1 * ∑ idx, |e idx| := by
  rw [dense_bias_diff, one_mul]
  exact Finset.single_le_sum (f := fun idx => |e idx|) (fun _ _ => abs_nonneg _)
    (Finset.mem_univ k)

/-- …and the pre-activation's total `ℓ1` drift is the step's `ℓ1` mass. -/
theorem dense_bias_drift_sum {m n : Nat} (W : Mat m n) (x : Vec m) (b e : Vec n) :
    ∑ k, |dense W (b + e) x k - dense W b x k| ≤ 1 * ∑ idx, |e idx| := by
  simp only [dense_bias_diff, one_mul, le_refl]

/-- The loss gradient in a bias is the loss's input gradient at that layer's pre-activation:
    the bias Jacobian is the identity (`pdiv_dense_b`). -/
theorem gradAt_bias_eq_pdiv {m n : Nat} (W : Mat m n) (x : Vec m) (G : Vec n → Vec 1) (b : Vec n)
    (hG : DifferentiableAt ℝ G (dense W b x)) (j : Fin n) :
    gradAt (fun b' => G (dense W b' x) 0) b j = pdiv G (dense W b x) j 0 := by
  have hZ : DifferentiableAt ℝ (fun b' : Vec n => dense W b' x) b := by unfold dense; fun_prop
  have hf : DifferentiableAt ℝ (fun b' : Vec n => G (dense W b' x) 0) b :=
    DifferentiableAt.comp (g := fun z => G z 0) b (differentiableAt_pi.1 hG 0) hZ
  rw [gradAt_eq_pdiv _ _ hf,
    show (fun w => fun _ : Fin 1 => G (dense W w x) 0) = G ∘ fun b' => dense W b' x by
      funext w i; fin_cases i; rfl,
    pdiv_comp _ _ _ hZ hG]
  simp_rw [pdiv_dense_b, ite_mul, one_mul, zero_mul]
  rw [Finset.sum_ite_eq]; simp

-- ════════════════════════════════════════════════════════════════
-- § Output bias
-- ════════════════════════════════════════════════════════════════

/-- The linear classifier's loss as a function of its bias. -/
noncomputable def linearBiasLoss {m n : Nat} (W : Mat m n) (x : Vec m) (label : Fin n) :
    Vec n → ℝ :=
  fun b => crossEntropy n (dense W b x) label

/-- **Closed form of the output-bias loss gradient**: `∂L/∂bⱼ = softmax(z)ⱼ − onehotⱼ`. -/
theorem linear_bias_loss_gradAt {m n : Nat} (W : Mat m n) (x : Vec m) (label : Fin n)
    (b : Vec n) (j : Fin n) :
    gradAt (linearBiasLoss W x label) b j = softmax n (dense W b x) j - oneHot n label j := by
  rw [← softmaxCE_grad]
  exact gradAt_bias_eq_pdiv W x (fun z _ => crossEntropy n z label) b
    (differentiable_pi.mpr (fun _ => crossEntropy_differentiable n label) _) j

/-- **Segment-Lipschitz gradient for the output-bias loss**: under `2D < 1` the gradient entries
    drift by at most `(2/(1−2D))·(t·D)` along `[v, v+d]`. -/
theorem linear_bias_loss_grad_lipschitz {m n : Nat} (W : Mat m n) (x : Vec m) (label : Fin n)
    {D : ℝ} (v d : Vec n) (hd : (∑ idx, |d idx|) ≤ D) (hsmall : 2 * D < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1) (j : Fin n) :
    |gradAt (linearBiasLoss W x label) (v + t • d) j - gradAt (linearBiasLoss W x label) v j| ≤
      (2 / (1 - 2 * D)) * (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D := le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  rw [linear_bias_loss_gradAt, linear_bias_loss_gradAt, sub_sub_sub_cancel_right]
  have hz : ∀ k, |dense W (v + t • d) x k - dense W v x k| ≤ t * D := fun k => by
    refine (dense_bias_drift W x v (t • d) k).trans ?_
    rw [one_mul, smul_l1_mass d ht0]
    exact mul_le_mul_of_nonneg_left hd ht0
  refine (softmax_seg_drift _ _ ht0 ht1 hD0 hsmall hz j).trans_eq ?_
  ring

/-- **One inexact SGD step on the output bias decreases one example's cross-entropy loss** —
    `linear_sgd_descends` with the input replaced by `1`: example `(x, label)`, the bias moving,
    the weights fixed, constant `C = 2/(1−2D)` at step radius `D = lr·(‖∇L‖₁ + n·η)`. The oracle
    accuracy, the small-step and the two dominance conditions remain hypotheses. -/
theorem linear_bias_sgd_descends {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (label : Fin n) (gh : Vec n) {lr η : ℝ} (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx - gradAt (linearBiasLoss W x label) b idx| ≤ η)
    (hsmall : 2 * stepRadius (linearBiasLoss W x label) b lr η < 1)
    (h1 : lr * η * (∑ idx, |gradAt (linearBiasLoss W x label) b idx|) ≤
      lr * (∑ idx, gradAt (linearBiasLoss W x label) b idx ^ 2) / 4)
    (h2 : (2 / (1 - 2 * stepRadius (linearBiasLoss W x label) b lr η)) *
        stepRadius (linearBiasLoss W x label) b lr η ^ 2 ≤
      lr * (∑ idx, gradAt (linearBiasLoss W x label) b idx ^ 2) / 4) :
    linearBiasLoss W x label (b - lr • gh) ≤
      linearBiasLoss W x label b - lr * (∑ idx, gradAt (linearBiasLoss W x label) b idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  have hC0 : (0:ℝ) ≤ 2 / (1 - 2 * (lr * ((∑ idx, |gradAt (linearBiasLoss W x label) b idx|) +
      (n : ℝ) * η))) := div_nonneg zero_le_two (by linarith)
  exact sgd_descends _ b gh hlr hη hC0 hgh
    (fun _ _ => ((crossEntropy_differentiable n label).comp
      (by unfold dense; fun_prop : Differentiable ℝ (fun b' : Vec n => dense W b' x))).differentiableAt)
    (fun t ht idx => linear_bias_loss_grad_lipschitz W x label b (-(lr • gh))
      (sgd_step_l1_le _ gh hlr hgh) hsmall t ht idx)
    h1 h2

-- ════════════════════════════════════════════════════════════════
-- § Hidden bias — one frozen mask
-- ════════════════════════════════════════════════════════════════

/-- The MLP's loss as a function of the hidden bias `b₁`. -/
noncomputable def mlpHiddenBiasLoss {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (b₂ : Vec d₃) (a₀ : Vec d₁) (label : Fin d₃) : Vec d₂ → ℝ :=
  fun b => crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b a₀))) label

/-- The hidden-bias loss is differentiable wherever the hidden pre-activation is off the kinks. -/
theorem mlp_hidden_bias_loss_differentiableAt {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (b₂ : Vec d₃) (a₀ : Vec d₁) (label : Fin d₃) (b : Vec d₂)
    (hz : ∀ k, dense W₁ b a₀ k ≠ 0) :
    DifferentiableAt ℝ (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b := by
  unfold mlpHiddenBiasLoss
  unfold dense at hz ⊢
  fun_prop (disch := assumption)

/-- **Closed form of the hidden-bias loss gradient at an off-kink point**:
    `∂L/∂b₁ⱼ = relu'(z₁ⱼ)·∑ₖ W₂ⱼₖ·(softmax − onehot)ₖ`. -/
theorem mlp_hidden_bias_loss_gradAt {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (b₂ : Vec d₃) (a₀ : Vec d₁) (label : Fin d₃) (b : Vec d₂) (hz : ∀ k, dense W₁ b a₀ k ≠ 0)
    (j : Fin d₂) :
    gradAt (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b j
      = (if dense W₁ b a₀ j > 0 then (1:ℝ) else 0) *
          ∑ k, W₂ j k * (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b a₀))) k - oneHot d₃ label k) := by
  rw [← ce_head_relu_input_grad W₂ b₂ label _ hz j]
  exact gradAt_bias_eq_pdiv W₁ a₀ (fun y _ => crossEntropy d₃ (dense W₂ b₂ (relu d₂ y)) label) b
    (by unfold dense at hz ⊢; fun_prop (disch := assumption)) j

/-- **Segment-Lipschitz gradient for the hidden-bias loss**: `MlpSlot.loss_grad_lipschitz` at the
    bias map, `σ = ρ = 1`, the row the channel indicator. Constant `2·d₃·w₂²/(1−2·w₂·D)`. -/
theorem mlp_hidden_bias_loss_grad_lipschitz {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂) (W₂ : Mat d₂ d₃)
    (b₂ : Vec d₃) (a₀ : Vec d₁) (label : Fin d₃) {w₂ D : ℝ}
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ j k, |W₂ j k| ≤ w₂)
    (v d : Vec d₂) (hd : (∑ idx, |d idx|) ≤ D) (hmargin : ∀ j, D < |dense W₁ v a₀ j|)
    (hsmall : 2 * (w₂ * D) < 1) (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1) (j : Fin d₂) :
    |gradAt (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) (v + t • d) j -
      gradAt (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) v j| ≤
      (2 * (d₃ : ℝ) * w₂ ^ 2 / (1 - 2 * (w₂ * D))) * (t * D) := by
  refine (MlpSlot.loss_grad_lipschitz (σ := 1) (ρ := 1) (fun b => dense W₁ b a₀) W₂ b₂ label
    zero_le_one (dense_bias_drift W₁ a₀) (dense_bias_drift_sum W₁ a₀) hw₂ hW₂
    (fun l => if l = j then 1 else 0)
    (by rw [Finset.sum_eq_single j (fun l _ hl => by simp [hl]) (by simp)]; simp)
    j (fun _ => True)
    (fun v' _ hz => by
      show gradAt (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) v' j = _
      rw [mlp_hidden_bias_loss_gradAt W₁ W₂ b₂ a₀ label v' hz j,
        Finset.sum_eq_single j (fun l _ hl => by simp [hl]) (by simp), ite_eq_left rfl, one_mul])
    v d hd (fun l => lt_of_eq_of_lt (one_mul D) (hmargin l))
    (lt_of_eq_of_lt (by ring) hsmall) t ht trivial trivial).trans_eq ?_
  rw [one_mul, one_pow, mul_one]

/-- **One inexact SGD step on the MLP's hidden bias decreases one example's cross-entropy loss**
    — `mlp_hidden_sgd_descends` with the layer input replaced by `1`: the margin `D < |z₁ⱼ|` at the
    step radius `D = lr·(‖∇L‖₁ + d₂·η)` freezes the hidden mask, constant
    `C = 2·d₃·w₂²/(1−2·w₂·D)`. -/
theorem mlp_hidden_bias_sgd_descends {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (a₀ : Vec d₁) (label : Fin d₃) (gh : Vec d₂) {lr η w₂ : ℝ}
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ j k, |W₂ j k| ≤ w₂) (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx - gradAt (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b₁ idx| ≤ η)
    (hmargin : ∀ j, stepRadius (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b₁ lr η < |dense W₁ b₁ a₀ j|)
    (hsmall : 2 * (w₂ * stepRadius (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b₁ lr η) < 1)
    (h1 : lr * η * (∑ idx, |gradAt (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b₁ idx|) ≤
      lr * (∑ idx, gradAt (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b₁ idx ^ 2) / 4)
    (h2 : (2 * (d₃ : ℝ) * w₂ ^ 2 /
          (1 - 2 * (w₂ * stepRadius (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b₁ lr η))) *
        stepRadius (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b₁ lr η ^ 2 ≤
      lr * (∑ idx, gradAt (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b₁ idx ^ 2) / 4) :
    mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label (b₁ - lr • gh) ≤
      mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label b₁ -
        lr * (∑ idx, gradAt (mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label) b₁ idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  set f := mlpHiddenBiasLoss W₁ W₂ b₂ a₀ label
  have hC0 : (0:ℝ) ≤ 2 * (d₃ : ℝ) * w₂ ^ 2 /
      (1 - 2 * (w₂ * (lr * ((∑ idx, |gradAt f b₁ idx|) + (d₂ : ℝ) * η)))) :=
    div_nonneg (by positivity) (by linarith)
  have hD := sgd_step_l1_le _ gh hlr hgh
  exact sgd_descends f b₁ gh hlr hη hC0 hgh
    (fun t ht => mlp_hidden_bias_loss_differentiableAt W₁ W₂ b₂ a₀ label _ fun k =>
      (margin_keeps_offkink_of_drift (fun b => dense W₁ b a₀) zero_le_one (dense_bias_drift W₁ a₀)
        b₁ (-(lr • gh)) hD (fun l => lt_of_eq_of_lt (one_mul _) (hmargin l)) t ht.1 ht.2 k).1)
    (fun t ht idx => mlp_hidden_bias_loss_grad_lipschitz W₁ W₂ b₂ a₀ label hw₂ hW₂ b₁ (-(lr • gh))
      hD hmargin hsmall t ht idx)
    h1 h2

-- ════════════════════════════════════════════════════════════════
-- § Input bias — two frozen masks
-- ════════════════════════════════════════════════════════════════

/-- The MLP's loss as a function of the input-layer bias `b₀`. -/
noncomputable def mlpInputBiasLoss {d₀ d₁ d₂ d₃ : Nat} (W₀ : Mat d₀ d₁) (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) : Vec d₁ → ℝ :=
  fun b => crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b x))))) label

/-- The input-bias loss is differentiable wherever both pre-activations are off the kinks. -/
theorem mlp_input_bias_loss_differentiableAt {d₀ d₁ d₂ d₃ : Nat} (W₀ : Mat d₀ d₁) (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) (b : Vec d₁)
    (hz0 : ∀ k, dense W₀ b x k ≠ 0) (hz1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b x)) k ≠ 0) :
    DifferentiableAt ℝ (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b := by
  unfold mlpInputBiasLoss
  unfold dense at hz0 hz1 ⊢
  fun_prop (disch := assumption)

/-- **Closed form of the input-bias loss gradient at a two-margin point**:
    `∂L/∂b₀ⱼ = relu'(z₀ⱼ)·∑ₗ W₁ⱼₗ·relu'(z₁ₗ)·∑ₖ W₂ₗₖ·(softmax − onehot)ₖ`. -/
theorem mlp_input_bias_loss_gradAt {d₀ d₁ d₂ d₃ : Nat} (W₀ : Mat d₀ d₁) (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) (b : Vec d₁)
    (hz0 : ∀ k, dense W₀ b x k ≠ 0) (hz1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b x)) k ≠ 0)
    (j : Fin d₁) :
    gradAt (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b j
      = (if dense W₀ b x j > 0 then (1:ℝ) else 0) *
          ∑ l, W₁ j l *
            ((if dense W₁ b₁ (relu d₁ (dense W₀ b x)) l > 0 then (1:ℝ) else 0) *
              ∑ k, W₂ l k *
                (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b x))))) k -
                  oneHot d₃ label k)) := by
  rw [← ce_head2_input_grad W₁ b₁ W₂ b₂ label _ hz0 hz1 j]
  exact gradAt_bias_eq_pdiv W₀ x
    (fun y _ => crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ y)))) label) b
    (by unfold dense at hz0 hz1 ⊢; fun_prop (disch := assumption)) j

/-- The middle pre-activation moves by at most `w₁·‖e‖₁` per entry under a step `e` of the input
    bias — one dense crossing after a 1-Lipschitz ReLU. -/
theorem mlp_input_bias_mid_drift {d₀ d₁ d₂ : Nat} (W₀ : Mat d₀ d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (x : Vec d₀) {w₁ : ℝ} (hw₁ : 0 ≤ w₁) (hW₁ : ∀ j l, |W₁ j l| ≤ w₁) (v e : Vec d₁) (l : Fin d₂) :
    |dense W₁ b₁ (relu d₁ (dense W₀ (v + e) x)) l - dense W₁ b₁ (relu d₁ (dense W₀ v x)) l| ≤
      w₁ * ∑ idx, |e idx| :=
  (dense_input_drift W₁ b₁ hW₁ _ _ l).trans (mul_le_mul_of_nonneg_left
    ((Finset.sum_le_sum fun i _ => relu_entry_lipschitz d₁ _ _ i).trans
      ((dense_bias_drift_sum W₀ x v e).trans_eq (one_mul _))) hw₁)

/-- **Segment-Lipschitz gradient for the input-bias loss**: `MlpSlot.loss_grad_lipschitz` at the
    middle pre-activation, `σ = w₁`, `ρ = d₂·w₁`, the row relu₀'s frozen mask times `W₁`'s row.
    Constant `2·d₃·d₂²·w₁²·w₂²/(1−2·w₂·d₂·w₁·D)`. -/
theorem mlp_input_bias_loss_grad_lipschitz {d₀ d₁ d₂ d₃ : Nat} (W₀ : Mat d₀ d₁) (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) {w₁ w₂ D : ℝ}
    (hw₁ : 0 ≤ w₁) (hW₁ : ∀ j l, |W₁ j l| ≤ w₁) (hw₂ : 0 ≤ w₂) (hW₂ : ∀ l k, |W₂ l k| ≤ w₂)
    (v d : Vec d₁) (hd : (∑ idx, |d idx|) ≤ D)
    (hmargin0 : ∀ j, D < |dense W₀ v x j|)
    (hmargin1 : ∀ l, w₁ * D < |dense W₁ b₁ (relu d₁ (dense W₀ v x)) l|)
    (hsmall : 2 * (w₂ * ((d₂ : ℝ) * (w₁ * D))) < 1) (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1)
    (j : Fin d₁) :
    |gradAt (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) (v + t • d) j -
      gradAt (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) v j| ≤
      (2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 /
        (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * D))))) * (t * D) := by
  have hD0 : 0 ≤ D := le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  have hJ : ∑ l, |(if dense W₀ v x j > 0 then (1:ℝ) else 0) * W₁ j l| ≤ (d₂ : ℝ) * w₁ := by
    refine (Finset.sum_le_sum fun l _ => ?_).trans_eq (by rw [Fin.sum_const, nsmul_eq_mul])
    rw [abs_mul]
    exact (mul_le_of_le_one_left (abs_nonneg _) (by split_ifs <;> simp)).trans (hW₁ j l)
  refine (MlpSlot.loss_grad_lipschitz (σ := w₁) (ρ := (d₂ : ℝ) * w₁)
    (fun b => dense W₁ b₁ (relu d₁ (dense W₀ b x))) W₂ b₂ label hw₁
    (mlp_input_bias_mid_drift W₀ W₁ b₁ x hw₁ hW₁)
    (fun v e => (Finset.sum_le_sum fun l _ =>
      mlp_input_bias_mid_drift W₀ W₁ b₁ x hw₁ hW₁ v e l).trans_eq (by
        rw [Fin.sum_const, nsmul_eq_mul]; ring))
    hw₂ hW₂ _ hJ j
    (fun v' => ∀ k, dense W₀ v' x k ≠ 0 ∧ (0 < dense W₀ v' x k ↔ 0 < dense W₀ v x k))
    (fun v' hQ hz1 => ?_) v d hd hmargin1 (lt_of_eq_of_lt (by ring) hsmall) t ht
    (fun k => ⟨abs_pos.mp (hD0.trans_lt (hmargin0 k)), Iff.rfl⟩)
    (fun k => margin_keeps_offkink_of_drift (fun b => dense W₀ b x) zero_le_one
      (dense_bias_drift W₀ x) v d hd (fun l => lt_of_eq_of_lt (one_mul D) (hmargin0 l))
      t ht.1 ht.2 k)).trans_eq (by ring)
  -- at a frozen relu₀ point the gradient is that row contracted with the head
  show gradAt (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) v' j = _
  rw [mlp_input_bias_loss_gradAt W₀ W₁ b₁ W₂ b₂ x label v' (fun k => (hQ k).1) hz1 j]
  simp only [gt_iff_lt, (hQ j).2]
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun l _ => by rw [mul_assoc]

/-- **One inexact SGD step on the MLP's input bias decreases one example's cross-entropy loss** —
    `mlp_input_sgd_descends` with the layer input replaced by `1`: the margins `D < |z₀ⱼ|` and
    `w₁·D < |z₁ₗ|` at the step radius `D = lr·(‖∇L‖₁ + d₁·η)` freeze both masks, constant
    `C = 2·d₃·d₂²·w₁²·w₂²/(1−2·w₂·d₂·w₁·D)`. With this and the output and hidden bias rungs, each
    dense bias of the MLP has a single-layer, single-example descent statement. -/
theorem mlp_input_bias_sgd_descends {d₀ d₁ d₂ d₃ : Nat} (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃)
    (gh : Vec d₁) {lr η w₁ w₂ : ℝ}
    (hw₁ : 0 ≤ w₁) (hW₁ : ∀ j l, |W₁ j l| ≤ w₁) (hw₂ : 0 ≤ w₂) (hW₂ : ∀ l k, |W₂ l k| ≤ w₂)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx - gradAt (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ idx| ≤ η)
    (hmargin0 : ∀ j,
      stepRadius (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ lr η < |dense W₀ b₀ x j|)
    (hmargin1 : ∀ l, w₁ * stepRadius (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ lr η <
      |dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) l|)
    (hsmall : 2 * (w₂ * ((d₂ : ℝ) * (w₁ *
      stepRadius (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ lr η))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ idx|) ≤
      lr * (∑ idx, gradAt (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ idx ^ 2) / 4)
    (h2 : (2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 /
          (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ *
            stepRadius (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ lr η))))) *
        stepRadius (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ lr η ^ 2 ≤
      lr * (∑ idx, gradAt (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ idx ^ 2) / 4) :
    mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label (b₀ - lr • gh) ≤
      mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label b₀ -
        lr * (∑ idx, gradAt (mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label) b₀ idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  set f := mlpInputBiasLoss W₀ W₁ b₁ W₂ b₂ x label
  have hC0 : (0:ℝ) ≤ 2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 /
      (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (lr * ((∑ idx, |gradAt f b₀ idx|) + (d₁ : ℝ) * η)))))) :=
    div_nonneg (by positivity) (by linarith)
  have hD := sgd_step_l1_le _ gh hlr hgh
  exact sgd_descends f b₀ gh hlr hη hC0 hgh
    (fun t ht => mlp_input_bias_loss_differentiableAt W₀ W₁ b₁ W₂ b₂ x label _
      (fun k => (margin_keeps_offkink_of_drift (fun b => dense W₀ b x) zero_le_one
        (dense_bias_drift W₀ x) b₀ (-(lr • gh)) hD
        (fun l => lt_of_eq_of_lt (one_mul _) (hmargin0 l)) t ht.1 ht.2 k).1)
      (fun l => (margin_keeps_offkink_of_drift
        (fun b => dense W₁ b₁ (relu d₁ (dense W₀ b x))) hw₁
        (mlp_input_bias_mid_drift W₀ W₁ b₁ x hw₁ hW₁) b₀ (-(lr • gh)) hD hmargin1
        t ht.1 ht.2 l).1))
    (fun t ht idx => mlp_input_bias_loss_grad_lipschitz W₀ W₁ b₁ W₂ b₂ x label hw₁ hW₁ hw₂ hW₂
      b₀ (-(lr • gh)) hD hmargin0 hmargin1 hsmall t ht idx)
    h1 h2

end Proofs
