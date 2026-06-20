import LeanMlir.Proofs.SgdDescentLinear
import LeanMlir.Proofs.MlpTrainStep

/-! # Lipschitz constants for the MLP softmax-CE loss — descent through the ReLU kinks

`SgdDescentLinear` discharged `sgd_descends`' smoothness hypothesis for the
Chapter-2 linear net. This file extends the discharge through the Chapter-3
MLP (`dense → relu → dense → relu → dense`), layer by layer:

* **Output layer `W₂` — free.** The top dense layer sees the loss with no
  ReLU in between, so its descent statement IS the linear one at the hidden
  activation `a₁` (`mlp_output_sgd_descends` = `linear_sgd_descends` at
  `x := a₁`).

* **Hidden layer `W₁` — the genuinely new piece.** The chain to the loss
  crosses one ReLU kink, so the loss-of-`W₁` map is only *piecewise* smooth.
  The key is the **margin hypothesis** `a·D < |z₁ⱼ|` (step `ℓ1`-radius `D`,
  activations bounded by `a`): the parameter step then cannot flip any ReLU
  sign, the masks FREEZE along the whole segment
  (`sign_stable_of_close`), and on the frozen-mask region the same
  elementary route as the linear case (logit drift → softmax ratio sandwich
  → γ-form) yields the explicit segment-Lipschitz constant
  `2·d₃·w₂²·a²/(1 − 2·w₂·a·D)` (`mlp_hidden_loss_grad_lipschitz`).
  This is the descent-side twin of `FloatBridge`'s quantitative ReLU margin
  `ez < |zᵢ|`: there the *rounding* must not flip a mask, here the *step*.

* **Input layer `W₀` — two frozen masks.** Same shape, one more dense+ReLU
  crossing; the constant picks up the `ℓ1→ℓ1` operator factor `d₂·w₁` of the
  middle layer: `2·d₃·d₂²·w₁²·w₂²·a²/(1 − 2·d₂·w₁·w₂·a·D)`
  (`mlp_input_loss_grad_lipschitz`).

The capstones `mlp_hidden_sgd_descends` / `mlp_input_sgd_descends` mirror
`linear_sgd_descends`: an `η`-accurate gradient oracle (the float budgets),
the margin(s) at the step radius, the small-step condition, and the two
dominance conditions ⇒ **one inexact SGD step on that layer's weights
provably decreases the cross-entropy loss by ≥ lr·‖∇L‖₂²/2.** Every
hypothesis is checkable arithmetic at a concrete point; smoothness is
proven, not assumed. Bias columns are the same argument with the layer
input replaced by the constant `1` and are omitted. The joint all-layers
step (every parameter moving at once, logits no longer affine in the moving
parameters) is the remaining open rung. -/

namespace Proofs

open StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Helpers: ReLU is 1-Lipschitz; margins freeze signs; ℓ1 column mass
-- ════════════════════════════════════════════════════════════════

/-- ReLU is entrywise 1-Lipschitz. The float bridge proved the same fact for
    the rounded net (`relu` exact-in-float); this is the ℝ-side workhorse
    that lets forward drift pass through a kinked layer unamplified. -/
theorem relu_entry_lipschitz (n : Nat) (u v : Vec n) (k : Fin n) :
    |relu n u k - relu n v k| ≤ |u k - v k| := by
  simp only [relu]
  by_cases hu : u k > 0 <;> by_cases hv : v k > 0
  · rw [if_pos hu, if_pos hv]
  · rw [if_pos hu, if_neg hv]
    have hv' : v k ≤ 0 := not_lt.mp hv
    rw [sub_zero, abs_of_pos hu, abs_of_pos (by linarith : (0:ℝ) < u k - v k)]
    linarith
  · rw [if_neg hu, if_pos hv]
    have hu' : u k ≤ 0 := not_lt.mp hu
    rw [zero_sub, abs_neg, abs_of_pos hv,
      abs_of_neg (by linarith : u k - v k < 0)]
    linarith
  · rw [if_neg hu, if_neg hv]
    simp

/-- **Margins freeze signs.** If a value drifts by at most `c` and sits at
    distance more than `c` from the kink, the drifted value is still off the
    kink *with the same sign* — the ReLU mask cannot flip. -/
theorem sign_stable_of_close {zt z c : ℝ} (hc : |zt - z| ≤ c)
    (hm : c < |z|) : zt ≠ 0 ∧ (0 < zt ↔ 0 < z) := by
  have habs := abs_le.mp hc
  rcases lt_trichotomy z 0 with hneg | hzero | hpos
  · have hzt : zt < 0 := by rw [abs_of_neg hneg] at hm; linarith [habs.2]
    exact ⟨ne_of_lt hzt, by constructor <;> intro h <;> linarith⟩
  · exfalso
    rw [hzero, abs_zero] at hm
    exact absurd hm (not_lt.mpr (le_trans (abs_nonneg _) hc))
  · have hzt : 0 < zt := by rw [abs_of_pos hpos] at hm; linarith [habs.1]
    exact ⟨ne_of_gt hzt, ⟨fun _ => hpos, fun _ => hzt⟩⟩

/-- The `ℓ1` mass of a flattened weight perturbation, summed column by
    column, is the total `ℓ1` mass — `finProdFinEquiv` partitions the
    flat index set into the columns. -/
theorem sum_abs_flatten_cols {m n : Nat} (d : Vec (m * n)) :
    ∑ j : Fin n, ∑ i : Fin m, |d (finProdFinEquiv (i, j))| =
      ∑ idx, |d idx| := by
  calc ∑ j : Fin n, ∑ i : Fin m, |d (finProdFinEquiv (i, j))|
      = ∑ i : Fin m, ∑ j : Fin n, |d (finProdFinEquiv (i, j))| :=
        Finset.sum_comm
    _ = ∑ idx, |d idx| := by
        rw [← Equiv.sum_comp finProdFinEquiv fun idx => |d idx|,
          Fintype.sum_prod_type]

/-- The dense pre-activation difference under a weight perturbation, exactly:
    column `j` only sees the column-`j` slice of the perturbation. -/
theorem dense_unflatten_diff {m n : Nat} (b : Vec n) (x : Vec m)
    (v e : Vec (m * n)) (j : Fin n) :
    dense (Mat.unflatten (v + e)) b x j - dense (Mat.unflatten v) b x j =
      ∑ i, x i * e (finProdFinEquiv (i, j)) := by
  have h2 : (∑ i : Fin m,
      x i * (v (finProdFinEquiv (i, j)) + e (finProdFinEquiv (i, j)))) -
      (∑ i : Fin m, x i * v (finProdFinEquiv (i, j))) =
      ∑ i : Fin m, x i * e (finProdFinEquiv (i, j)) := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by ring
  show ((∑ i : Fin m, x i * (v + e) (finProdFinEquiv (i, j))) + b j) -
      ((∑ i : Fin m, x i * v (finProdFinEquiv (i, j))) + b j) = _
  simp only [Pi.add_apply]
  linarith [h2]

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

/-- Summed over all coordinates, the pre-activation drift is bounded by
    `a·‖e‖₁` *total* — the column masses tile the flat index set. -/
theorem dense_unflatten_drift_sum {m n : Nat} (b : Vec n) (x : Vec m)
    {a : ℝ} (hx : ∀ i, |x i| ≤ a) (v e : Vec (m * n)) :
    ∑ j, |dense (Mat.unflatten (v + e)) b x j -
        dense (Mat.unflatten v) b x j| ≤
      a * ∑ idx, |e idx| := by
  calc ∑ j, |dense (Mat.unflatten (v + e)) b x j -
        dense (Mat.unflatten v) b x j|
      ≤ ∑ j, a * ∑ i, |e (finProdFinEquiv (i, j))| :=
        Finset.sum_le_sum fun j _ => dense_unflatten_col_drift b x hx v e j
    _ = a * ∑ j, ∑ i, |e (finProdFinEquiv (i, j))| := by
        rw [Finset.mul_sum]
    _ = a * ∑ idx, |e idx| := by rw [sum_abs_flatten_cols]

/-- **The margin keeps the pre-activation off the kink along the whole
    segment.** With the step's `ℓ1` mass at most `D` and inputs bounded by
    `a`, the pre-activation drifts by at most `a·D` — strictly inside the
    margin — so every point of `[v, v + e]` is off the kink *with the
    original sign*. -/
theorem margin_keeps_offkink {m n : Nat} (b : Vec n) (x : Vec m)
    {a D : ℝ} (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a) (v e : Vec (m * n))
    (he : (∑ idx, |e idx|) ≤ D)
    (hmargin : ∀ j, a * D < |dense (Mat.unflatten v) b x j|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (j : Fin n) :
    dense (Mat.unflatten (v + t • e)) b x j ≠ 0 ∧
      (0 < dense (Mat.unflatten (v + t • e)) b x j ↔
        0 < dense (Mat.unflatten v) b x j) := by
  refine sign_stable_of_close ?_ (hmargin j)
  have h1 := dense_unflatten_drift b x ha hx v (t • e) j
  have h2 : (∑ idx, |(t • e) idx|) = t * ∑ idx, |e idx| := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun idx _ => by
      simp [abs_mul, abs_of_nonneg ht0]
  rw [h2] at h1
  have h3 : a * (t * ∑ idx, |e idx|) ≤ a * D := by
    refine mul_le_mul_of_nonneg_left ?_ ha
    calc t * ∑ idx, |e idx|
        ≤ 1 * D := mul_le_mul ht1 he
          (Finset.sum_nonneg fun _ _ => abs_nonneg _) zero_le_one
      _ = D := one_mul D
  linarith

-- ════════════════════════════════════════════════════════════════
-- § Input-gradients of the loss head — the pdiv-level closed forms
-- ════════════════════════════════════════════════════════════════

/-- **Loss input-gradient at the logits' input**:
    `∂(CE ∘ dense W₂)/∂yⱼ = ∑ₖ W₂ⱼₖ·(softmax − onehot)ₖ` — the pdiv-level
    form of the backward chain's dense-back step. -/
theorem ce_dense_input_grad {d₂ d₃ : Nat} (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (label : Fin d₃) (y : Vec d₂) (j : Fin d₂) :
    pdiv (fun z : Vec d₂ => fun _ : Fin 1 =>
        crossEntropy d₃ (dense W₂ b₂ z) label) y j 0
      = ∑ k, W₂ j k *
          (softmax d₃ (dense W₂ b₂ y) k - oneHot d₃ label k) := by
  rw [show (fun z : Vec d₂ => fun _ : Fin 1 =>
          crossEntropy d₃ (dense W₂ b₂ z) label)
        = (fun z : Vec d₃ => fun _ : Fin 1 => crossEntropy d₃ z label)
            ∘ (dense W₂ b₂) from rfl,
      pdiv_comp _ _ _ ((dense_differentiable W₂ b₂) y)
        (differentiable_pi.mpr
          (fun _ => crossEntropy_differentiable d₃ label) _)]
  exact Finset.sum_congr rfl fun k _ => by
    rw [pdiv_dense, softmaxCE_grad]

/-- **Loss input-gradient through one ReLU** — at an off-kink point the
    chain picks up the mask: `∂(CE ∘ dense W₂ ∘ relu)/∂zⱼ =
    relu'(zⱼ)·∑ₖ W₂ⱼₖ·(softmax − onehot)ₖ`. The pdiv-level form of the
    cotangent `mlpCotOut1` delivers (cf. `mlpCotOut1_denote`). -/
theorem ce_head_relu_input_grad {d₂ d₃ : Nat} (W₂ : Mat d₂ d₃)
    (b₂ : Vec d₃) (label : Fin d₃) (z : Vec d₂) (hz : ∀ k, z k ≠ 0)
    (j : Fin d₂) :
    pdiv (fun y : Vec d₂ => fun _ : Fin 1 =>
        crossEntropy d₃ (dense W₂ b₂ (relu d₂ y)) label) z j 0
      = (if z j > 0 then (1:ℝ) else 0) *
          ∑ k, W₂ j k *
            (softmax d₃ (dense W₂ b₂ (relu d₂ z)) k - oneHot d₃ label k) := by
  have hg : DifferentiableAt ℝ
      (fun z' : Vec d₂ => fun _ : Fin 1 =>
        crossEntropy d₃ (dense W₂ b₂ z') label) (relu d₂ z) := by
    rw [differentiableAt_pi]
    intro _
    exact (crossEntropy_differentiable d₃ label).differentiableAt.comp _
      ((dense_differentiable W₂ b₂) _)
  rw [show (fun y : Vec d₂ => fun _ : Fin 1 =>
          crossEntropy d₃ (dense W₂ b₂ (relu d₂ y)) label)
        = (fun z' : Vec d₂ => fun _ : Fin 1 =>
            crossEntropy d₃ (dense W₂ b₂ z') label) ∘ (relu d₂) from rfl,
      pdiv_comp _ _ _ (relu_differentiableAt_of_smooth d₂ z hz) hg]
  simp_rw [pdiv_relu d₂ z hz j, ite_mul, zero_mul]
  rw [Finset.sum_ite_eq]
  simp only [Finset.mem_univ, if_true]
  rw [ce_dense_input_grad]

-- ════════════════════════════════════════════════════════════════
-- § Hidden layer W₁: gradient closed form, frozen-mask Lipschitz, descent
-- ════════════════════════════════════════════════════════════════

/-- The loss-of-`W₁` map is differentiable wherever the hidden
    pre-activation is off the kinks. -/
theorem mlp_hidden_loss_differentiableAt {d₁ d₂ d₃ : Nat} (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (a₀ : Vec d₁) (label : Fin d₃)
    (w : Vec (d₁ * d₂))
    (hz : ∀ k, dense (Mat.unflatten w) b₁ a₀ k ≠ 0) :
    DifferentiableAt ℝ
      (fun w' : Vec (d₁ * d₂) =>
        crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w') b₁ a₀))) label)
      w := by
  have h0 : DifferentiableAt ℝ
      (fun w' : Vec (d₁ * d₂) => dense (Mat.unflatten w') b₁ a₀) w :=
    (denseWeightMap_differentiable b₁ a₀) w
  have h2 : DifferentiableAt ℝ
      (fun w' : Vec (d₁ * d₂) => relu d₂ (dense (Mat.unflatten w') b₁ a₀))
      w :=
    (relu_differentiableAt_of_smooth d₂ _ hz).comp
      (f := fun w' : Vec (d₁ * d₂) => dense (Mat.unflatten w') b₁ a₀) w h0
  have h3 : DifferentiableAt ℝ
      (fun w' : Vec (d₁ * d₂) =>
        dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w') b₁ a₀))) w :=
    ((dense_differentiable W₂ b₂) _).comp
      (f := fun w' : Vec (d₁ * d₂) =>
        relu d₂ (dense (Mat.unflatten w') b₁ a₀)) w h2
  exact (crossEntropy_differentiable d₃ label).differentiableAt.comp
    (f := fun w' : Vec (d₁ * d₂) =>
      dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w') b₁ a₀))) w h3

/-- **Closed form of the hidden-layer loss gradient at any off-kink
    parameter point**: `∂L/∂W₁_{ij} = a₀ᵢ·relu'(z₁ⱼ)·∑ₖ W₂ⱼₖ·(softmax −
    onehot)ₖ` — the suite's conditional fold (`mlp_hidden_total_loss_grad`)
    re-expressed through `gradAt` with both `pdiv` factors collapsed to
    their certified closed forms. The hidden-layer peer of
    `linear_loss_gradAt`. -/
theorem mlp_hidden_loss_gradAt {d₁ d₂ d₃ : Nat} (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (a₀ : Vec d₁) (label : Fin d₃)
    (v : Vec (d₁ * d₂)) (hz : ∀ k, dense (Mat.unflatten v) b₁ a₀ k ≠ 0)
    (i : Fin d₁) (j : Fin d₂) :
    gradAt (fun w => crossEntropy d₃
        (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label) v
        (finProdFinEquiv (i, j))
      = a₀ i * ((if dense (Mat.unflatten v) b₁ a₀ j > 0 then (1:ℝ) else 0) *
          ∑ k, W₂ j k *
            (softmax d₃
              (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) k -
              oneHot d₃ label k)) := by
  calc gradAt (fun w => crossEntropy d₃
        (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label) v
        (finProdFinEquiv (i, j))
      = pdiv (fun w => fun _ : Fin 1 => crossEntropy d₃
            (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
          v (finProdFinEquiv (i, j)) 0 :=
        gradAt_eq_pdiv _ _
          (mlp_hidden_loss_differentiableAt b₁ W₂ b₂ a₀ label v hz) _
    _ = pdiv (fun w => fun _ : Fin 1 => crossEntropy d₃
            (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
          (Mat.flatten (Mat.unflatten v)) (finProdFinEquiv (i, j)) 0 := by
        rw [Mat.flatten_unflatten]
    _ = ∑ k : Fin d₂,
          pdiv (fun w : Vec (d₁ * d₂) => dense (Mat.unflatten w) b₁ a₀)
              (Mat.flatten (Mat.unflatten v)) (finProdFinEquiv (i, j)) k
            * pdiv (fun z : Vec d₂ => fun _ : Fin 1 =>
                  crossEntropy d₃ (dense W₂ b₂ (relu d₂ z)) label)
                (dense (Mat.unflatten v) b₁ a₀) k 0 :=
        IR.mlp_hidden_total_loss_grad (Mat.unflatten v) b₁ W₂ b₂ a₀ label
          hz i j
    _ = ∑ k : Fin d₂, (if k = j then a₀ i else 0)
            * pdiv (fun z : Vec d₂ => fun _ : Fin 1 =>
                  crossEntropy d₃ (dense W₂ b₂ (relu d₂ z)) label)
                (dense (Mat.unflatten v) b₁ a₀) k 0 :=
        Finset.sum_congr rfl fun k _ => by
          rw [pdiv_dense_W b₁ a₀ (Mat.unflatten v) i j k]
    _ = a₀ i * pdiv (fun z : Vec d₂ => fun _ : Fin 1 =>
            crossEntropy d₃ (dense W₂ b₂ (relu d₂ z)) label)
          (dense (Mat.unflatten v) b₁ a₀) j 0 := by
        simp only [ite_mul, zero_mul]
        rw [Finset.sum_ite_eq']
        simp
    _ = a₀ i * ((if dense (Mat.unflatten v) b₁ a₀ j > 0 then (1:ℝ) else 0) *
          ∑ k, W₂ j k *
            (softmax d₃
              (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) k -
              oneHot d₃ label k)) := by
        rw [ce_head_relu_input_grad W₂ b₂ label _ hz j]

/-- The hidden-layer logit drift: a weight perturbation of `ℓ1` mass
    `‖e‖₁` moves every logit by at most `w₂·a·‖e‖₁` — through the frozen
    dense, the 1-Lipschitz ReLU, and the column-tiled `ℓ1` mass. No width
    factor. -/
theorem mlp_hidden_logit_drift {d₁ d₂ d₃ : Nat} (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (a₀ : Vec d₁) {a w₂ : ℝ}
    (hx : ∀ i, |a₀ i| ≤ a) (hw₂ : 0 ≤ w₂) (hW₂ : ∀ j k, |W₂ j k| ≤ w₂)
    (v e : Vec (d₁ * d₂)) (k : Fin d₃) :
    |dense W₂ b₂ (relu d₂ (dense (Mat.unflatten (v + e)) b₁ a₀)) k -
      dense W₂ b₂ (relu d₂ (dense (Mat.unflatten v) b₁ a₀)) k| ≤
      w₂ * (a * ∑ idx, |e idx|) := by
  have hdiff : dense W₂ b₂ (relu d₂ (dense (Mat.unflatten (v + e)) b₁ a₀)) k -
      dense W₂ b₂ (relu d₂ (dense (Mat.unflatten v) b₁ a₀)) k =
      ∑ j, (relu d₂ (dense (Mat.unflatten (v + e)) b₁ a₀) j -
        relu d₂ (dense (Mat.unflatten v) b₁ a₀) j) * W₂ j k := by
    have h2 : (∑ j, relu d₂ (dense (Mat.unflatten (v + e)) b₁ a₀) j * W₂ j k) -
        (∑ j, relu d₂ (dense (Mat.unflatten v) b₁ a₀) j * W₂ j k) =
        ∑ j, (relu d₂ (dense (Mat.unflatten (v + e)) b₁ a₀) j -
          relu d₂ (dense (Mat.unflatten v) b₁ a₀) j) * W₂ j k := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    show ((∑ j, relu d₂ (dense (Mat.unflatten (v + e)) b₁ a₀) j * W₂ j k) + b₂ k) -
        ((∑ j, relu d₂ (dense (Mat.unflatten v) b₁ a₀) j * W₂ j k) + b₂ k) = _
    linarith [h2]
  rw [hdiff]
  calc |∑ j, (relu d₂ (dense (Mat.unflatten (v + e)) b₁ a₀) j -
        relu d₂ (dense (Mat.unflatten v) b₁ a₀) j) * W₂ j k|
      ≤ ∑ j, |(relu d₂ (dense (Mat.unflatten (v + e)) b₁ a₀) j -
          relu d₂ (dense (Mat.unflatten v) b₁ a₀) j) * W₂ j k| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, |dense (Mat.unflatten (v + e)) b₁ a₀ j -
          dense (Mat.unflatten v) b₁ a₀ j| * w₂ :=
        Finset.sum_le_sum fun j _ => by
          rw [abs_mul]
          exact mul_le_mul (relu_entry_lipschitz d₂ _ _ j) (hW₂ j k)
            (abs_nonneg _) (abs_nonneg _)
    _ = (∑ j, |dense (Mat.unflatten (v + e)) b₁ a₀ j -
          dense (Mat.unflatten v) b₁ a₀ j|) * w₂ := by
        rw [← Finset.sum_mul]
    _ ≤ (a * ∑ idx, |e idx|) * w₂ :=
        mul_le_mul_of_nonneg_right
          (dense_unflatten_drift_sum b₁ a₀ hx v e) hw₂
    _ = w₂ * (a * ∑ idx, |e idx|) := by ring

/-- **Segment-Lipschitz gradient for the hidden-layer loss, explicit
    constant.** Under the margin `a·D < |z₁ⱼ|` (the step cannot flip a ReLU
    sign — the masks freeze along the whole segment) and the small-step
    condition `2·w₂·a·D < 1`, the gradient entries drift by at most
    `(2·d₃·w₂²·a²/(1−2·w₂·a·D))·(t·D)` along `[v, v+d]` — the exact shape
    `descent_segment` consumes. The hidden-layer peer of
    `linear_loss_grad_lipschitz`. -/
theorem mlp_hidden_loss_grad_lipschitz {d₁ d₂ d₃ : Nat} (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (a₀ : Vec d₁) (label : Fin d₃)
    {a w₂ D : ℝ} (ha : 0 ≤ a) (hx : ∀ i, |a₀ i| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ j k, |W₂ j k| ≤ w₂)
    (v d : Vec (d₁ * d₂)) (hd : (∑ idx, |d idx|) ≤ D)
    (hmargin : ∀ j, a * D < |dense (Mat.unflatten v) b₁ a₀ j|)
    (hsmall : 2 * (w₂ * (a * D)) < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1) (idx : Fin (d₁ * d₂)) :
    |gradAt (fun w => crossEntropy d₃
        (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        (v + t • d) idx -
      gradAt (fun w => crossEntropy d₃
        (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        v idx| ≤
      (2 * (d₃ : ℝ) * w₂ ^ 2 * a ^ 2 / (1 - 2 * (w₂ * (a * D)))) *
        (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D :=
    le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  have hden : (0:ℝ) < 1 - 2 * (w₂ * (a * D)) := by linarith
  obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
  -- ℓ1 mass of the scaled step
  have htmass : (∑ idx, |(t • d) idx|) = t * ∑ idx, |d idx| := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun idx _ => by
      simp [abs_mul, abs_of_nonneg ht0]
  -- the margin keeps every pre-activation off the kink, same sign, along
  -- the segment
  have hz_v : ∀ k, dense (Mat.unflatten v) b₁ a₀ k ≠ 0 := fun k => by
    intro h0
    have h := hmargin k
    rw [h0, abs_zero] at h
    exact absurd h (not_lt.mpr (mul_nonneg ha hD0))
  have hstab : ∀ k,
      dense (Mat.unflatten (v + t • d)) b₁ a₀ k ≠ 0 ∧
        (0 < dense (Mat.unflatten (v + t • d)) b₁ a₀ k ↔
          0 < dense (Mat.unflatten v) b₁ a₀ k) :=
    fun k => margin_keeps_offkink b₁ a₀ ha hx v d hd hmargin t ht0 ht1 k
  have hz_t : ∀ k, dense (Mat.unflatten (v + t • d)) b₁ a₀ k ≠ 0 :=
    fun k => (hstab k).1
  rw [mlp_hidden_loss_gradAt b₁ W₂ b₂ a₀ label (v + t • d) hz_t i j,
      mlp_hidden_loss_gradAt b₁ W₂ b₂ a₀ label v hz_v i j]
  -- the frozen mask
  have hmask : (if dense (Mat.unflatten (v + t • d)) b₁ a₀ j > 0
        then (1:ℝ) else 0) =
      (if dense (Mat.unflatten v) b₁ a₀ j > 0 then (1:ℝ) else 0) := by
    by_cases hp : dense (Mat.unflatten v) b₁ a₀ j > 0
    · rw [if_pos hp, if_pos ((hstab j).2.mpr hp)]
    · rw [if_neg hp, if_neg (fun h => hp ((hstab j).2.mp h))]
  rw [hmask]
  by_cases hp : dense (Mat.unflatten v) b₁ a₀ j > 0
  · -- live mask: the drift is `a₀ᵢ` times the contracted softmax drift
    rw [if_pos hp]
    have hcollapse : a₀ i * ((1:ℝ) *
          ∑ k, W₂ j k *
            (softmax d₃ (dense W₂ b₂
              (relu d₂ (dense (Mat.unflatten (v + t • d)) b₁ a₀))) k -
              oneHot d₃ label k)) -
        a₀ i * ((1:ℝ) *
          ∑ k, W₂ j k *
            (softmax d₃ (dense W₂ b₂
              (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) k -
              oneHot d₃ label k)) =
        a₀ i * ∑ k, W₂ j k *
          (softmax d₃ (dense W₂ b₂
            (relu d₂ (dense (Mat.unflatten (v + t • d)) b₁ a₀))) k -
            softmax d₃ (dense W₂ b₂
              (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) k) := by
      rw [one_mul, one_mul, ← mul_sub, ← Finset.sum_sub_distrib]
      congr 1
      exact Finset.sum_congr rfl fun k _ => by ring
    rw [hcollapse, abs_mul]
    -- logit drift along the segment, then the softmax ratio sandwich
    have hzdrift : ∀ k, |dense W₂ b₂
          (relu d₂ (dense (Mat.unflatten (v + t • d)) b₁ a₀)) k -
        dense W₂ b₂ (relu d₂ (dense (Mat.unflatten v) b₁ a₀)) k| ≤
        t * (w₂ * (a * D)) := by
      intro k
      have h1 := mlp_hidden_logit_drift b₁ W₂ b₂ a₀ hx hw₂ hW₂ v (t • d) k
      rw [htmass] at h1
      have h2 : w₂ * (a * (t * ∑ idx, |d idx|)) ≤ t * (w₂ * (a * D)) := by
        nlinarith [mul_le_mul_of_nonneg_left hd
          (mul_nonneg (mul_nonneg hw₂ ha) ht0)]
      linarith
    have hsm := fun k => FloatModel.softmax_perturb
      (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten (v + t • d)) b₁ a₀)))
      (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) hzdrift k
    have hδlt : 2 * (t * (w₂ * (a * D))) < 1 := by
      nlinarith [mul_le_mul_of_nonneg_right ht1
        (mul_nonneg hw₂ (mul_nonneg ha hD0))]
    have hexp : Real.exp (2 * (t * (w₂ * (a * D)))) - 1 ≤
        2 * (t * (w₂ * (a * D))) / (1 - 2 * (t * (w₂ * (a * D)))) :=
      FloatModel.exp_sub_one_le hδlt
    have hmono : 2 * (t * (w₂ * (a * D))) /
          (1 - 2 * (t * (w₂ * (a * D)))) ≤
        2 * (t * (w₂ * (a * D))) / (1 - 2 * (w₂ * (a * D))) := by
      refine div_le_div_of_nonneg_left
        (by nlinarith [mul_nonneg ht0 (mul_nonneg hw₂ (mul_nonneg ha hD0))])
        hden ?_
      nlinarith [mul_le_mul_of_nonneg_right ht1
        (mul_nonneg hw₂ (mul_nonneg ha hD0))]
    have hS : ∀ k, |softmax d₃ (dense W₂ b₂
          (relu d₂ (dense (Mat.unflatten (v + t • d)) b₁ a₀))) k -
        softmax d₃ (dense W₂ b₂
          (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) k| ≤
        2 * (t * (w₂ * (a * D))) / (1 - 2 * (w₂ * (a * D))) :=
      fun k => le_trans (hsm k) (le_trans hexp hmono)
    have hsum : |∑ k, W₂ j k *
          (softmax d₃ (dense W₂ b₂
            (relu d₂ (dense (Mat.unflatten (v + t • d)) b₁ a₀))) k -
            softmax d₃ (dense W₂ b₂
              (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) k)| ≤
        (d₃ : ℝ) * (w₂ *
          (2 * (t * (w₂ * (a * D))) / (1 - 2 * (w₂ * (a * D))))) := by
      calc |∑ k, W₂ j k *
            (softmax d₃ (dense W₂ b₂
              (relu d₂ (dense (Mat.unflatten (v + t • d)) b₁ a₀))) k -
              softmax d₃ (dense W₂ b₂
                (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) k)|
          ≤ ∑ k, |W₂ j k *
              (softmax d₃ (dense W₂ b₂
                (relu d₂ (dense (Mat.unflatten (v + t • d)) b₁ a₀))) k -
                softmax d₃ (dense W₂ b₂
                  (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) k)| :=
            Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _k : Fin d₃, w₂ *
              (2 * (t * (w₂ * (a * D))) / (1 - 2 * (w₂ * (a * D)))) :=
            Finset.sum_le_sum fun k _ => by
              rw [abs_mul]
              exact mul_le_mul (hW₂ j k) (hS k) (abs_nonneg _) hw₂
        _ = (d₃ : ℝ) * (w₂ *
              (2 * (t * (w₂ * (a * D))) / (1 - 2 * (w₂ * (a * D))))) := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
              nsmul_eq_mul]
    calc |a₀ i| * |∑ k, W₂ j k *
          (softmax d₃ (dense W₂ b₂
            (relu d₂ (dense (Mat.unflatten (v + t • d)) b₁ a₀))) k -
            softmax d₃ (dense W₂ b₂
              (relu d₂ (dense (Mat.unflatten v) b₁ a₀))) k)|
        ≤ a * ((d₃ : ℝ) * (w₂ *
            (2 * (t * (w₂ * (a * D))) / (1 - 2 * (w₂ * (a * D)))))) :=
          mul_le_mul (hx i) hsum (abs_nonneg _) ha
      _ = (2 * (d₃ : ℝ) * w₂ ^ 2 * a ^ 2 / (1 - 2 * (w₂ * (a * D)))) *
            (t * D) := by ring
  · -- dead mask: both gradients vanish
    rw [if_neg hp]
    simp only [zero_mul, mul_zero, sub_self, abs_zero]
    have hC0 : 0 ≤ 2 * (d₃ : ℝ) * w₂ ^ 2 * a ^ 2 /
        (1 - 2 * (w₂ * (a * D))) :=
      div_nonneg (by positivity) hden.le
    exact mul_nonneg hC0 (mul_nonneg ht0 hD0)

/-- **One inexact SGD step on the MLP's hidden weights provably decreases
    the cross-entropy loss.** All of `sgd_descends`' hypotheses discharged
    for the loss-of-`W₁` map: differentiability along the segment and the
    segment-Lipschitz constant `C = 2·d₃·w₂²·a²/(1−2·w₂·a·D)` at step radius
    `D = lr·(‖∇L‖₁ + d₁d₂·η)` both come from the **margin hypothesis** — the
    step radius is small enough that no hidden ReLU can change sign.
    Remaining hypotheses are checkable arithmetic: the oracle accuracy `η`
    (the float budgets), the margins, the small-step condition, and the two
    dominance conditions. Conclusion: the loss drops by ≥ `lr·‖∇L‖₂²/2`.
    The hidden-layer peer of `linear_sgd_descends`. -/
theorem mlp_hidden_sgd_descends {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (a₀ : Vec d₁)
    (label : Fin d₃) (gh : Vec (d₁ * d₂)) {lr η a w₂ : ℝ}
    (ha : 0 ≤ a) (hx : ∀ i, |a₀ i| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ j k, |W₂ j k| ≤ w₂)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx -
      gradAt (fun w => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        (Mat.flatten W₁) idx| ≤ η)
    (hmargin : ∀ j, a * (lr * ((∑ idx, |gradAt
        (fun w => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        (Mat.flatten W₁) idx|) + ((d₁ * d₂ : ℕ) : ℝ) * η)) <
      |dense W₁ b₁ a₀ j|)
    (hsmall : 2 * (w₂ * (a * (lr * ((∑ idx, |gradAt
        (fun w => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        (Mat.flatten W₁) idx|) + ((d₁ * d₂ : ℕ) : ℝ) * η)))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt
        (fun w => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        (Mat.flatten W₁) idx|) ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        (Mat.flatten W₁) idx ^ 2) / 4)
    (h2 : (2 * (d₃ : ℝ) * w₂ ^ 2 * a ^ 2 / (1 - 2 * (w₂ * (a * (lr *
          ((∑ idx, |gradAt (fun w => crossEntropy d₃
              (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
            (Mat.flatten W₁) idx|) + ((d₁ * d₂ : ℕ) : ℝ) * η)))))) *
        (lr * ((∑ idx, |gradAt (fun w => crossEntropy d₃
            (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
          (Mat.flatten W₁) idx|) + ((d₁ * d₂ : ℕ) : ℝ) * η)) ^ 2 ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        (Mat.flatten W₁) idx ^ 2) / 4) :
    crossEntropy d₃ (dense W₂ b₂ (relu d₂
        (dense (Mat.unflatten (Mat.flatten W₁ - lr • gh)) b₁ a₀))) label ≤
      crossEntropy d₃ (dense W₂ b₂ (relu d₂
        (dense (Mat.unflatten (Mat.flatten W₁)) b₁ a₀))) label -
        lr * (∑ idx, gradAt
          (fun w => crossEntropy d₃
            (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
          (Mat.flatten W₁) idx ^ 2) / 2 := by
  set f : Vec (d₁ * d₂) → ℝ :=
    fun w => crossEntropy d₃
      (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label with hf
  have hden : (0:ℝ) < 1 - 2 * (w₂ * (a * (lr * ((∑ idx,
      |gradAt f (Mat.flatten W₁) idx|) + ((d₁ * d₂ : ℕ) : ℝ) * η)))) := by
    linarith
  have hC0 : (0:ℝ) ≤ 2 * (d₃ : ℝ) * w₂ ^ 2 * a ^ 2 /
      (1 - 2 * (w₂ * (a * (lr * ((∑ idx,
        |gradAt f (Mat.flatten W₁) idx|) + ((d₁ * d₂ : ℕ) : ℝ) * η))))) :=
    div_nonneg (by positivity) hden.le
  -- the margin, restated at the `unflatten ∘ flatten` parameter point
  have hmargin' : ∀ j, a * (lr * ((∑ idx,
      |gradAt f (Mat.flatten W₁) idx|) + ((d₁ * d₂ : ℕ) : ℝ) * η)) <
      |dense (Mat.unflatten (Mat.flatten W₁)) b₁ a₀ j| := fun j => by
    rw [Mat.unflatten_flatten]
    exact hmargin j
  -- ℓ1 radius of the step
  have hD : (∑ idx, |(-(lr • gh)) idx|) ≤
      lr * ((∑ idx, |gradAt f (Mat.flatten W₁) idx|) +
        ((d₁ * d₂ : ℕ) : ℝ) * η) := by
    calc (∑ idx, |(-(lr • gh)) idx|) = ∑ idx, lr * |gh idx| := by
          refine Finset.sum_congr rfl fun idx _ => ?_
          simp [abs_mul, abs_of_nonneg hlr]
      _ ≤ ∑ idx, lr * (|gradAt f (Mat.flatten W₁) idx| + η) := by
          refine Finset.sum_le_sum fun idx _ => ?_
          refine mul_le_mul_of_nonneg_left ?_ hlr
          have h3 : |gh idx| ≤ |gh idx - gradAt f (Mat.flatten W₁) idx| +
              |gradAt f (Mat.flatten W₁) idx| := by
            simpa using abs_sub_le (gh idx) (gradAt f (Mat.flatten W₁) idx) 0
          linarith [hgh idx]
      _ = lr * ((∑ idx, |gradAt f (Mat.flatten W₁) idx|) +
            ((d₁ * d₂ : ℕ) : ℝ) * η) := by
          rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const,
            Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hmain := sgd_descends f (Mat.flatten W₁) gh hlr hη hC0 hgh
    (fun t ht => mlp_hidden_loss_differentiableAt b₁ W₂ b₂ a₀ label _
      (fun k => (margin_keeps_offkink b₁ a₀ ha hx (Mat.flatten W₁)
        (-(lr • gh)) hD hmargin' t ht.1 ht.2 k).1))
    (fun t ht idx => by
      have := mlp_hidden_loss_grad_lipschitz b₁ W₂ b₂ a₀ label ha hx hw₂
        hW₂ (Mat.flatten W₁) (-(lr • gh)) hD hmargin' hsmall t ht idx
      simpa [hf] using this)
    h1 h2
  simpa [hf] using hmain

-- ════════════════════════════════════════════════════════════════
-- § Output layer W₂: the linear descent theorem at the hidden activation
-- ════════════════════════════════════════════════════════════════

/-- **One inexact SGD step on the MLP's output weights provably decreases
    the cross-entropy loss — for free.** The top dense layer sits directly
    below the softmax-CE loss with no ReLU in between, so the loss-of-`W₂`
    map IS the linear net's loss at the hidden activation
    `a₁ = relu(dense W₁ b₁ (relu(dense W₀ b₀ x)))`: this is
    `linear_sgd_descends` instantiated there. No margin needed — the output
    layer never crosses a kink. -/
theorem mlp_output_sgd_descends {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃)
    (gh : Vec (d₂ * d₃)) {lr η a : ℝ}
    (ha : 0 ≤ a)
    (hx : ∀ i, |relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))) i| ≤ a)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx -
      gradAt (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
        (Mat.flatten W₂) idx| ≤ η)
    (hsmall : 2 * (a * (lr * ((∑ idx, |gradAt
        (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
        (Mat.flatten W₂) idx|) + ((d₂ * d₃ : ℕ) : ℝ) * η))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt
        (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
        (Mat.flatten W₂) idx|) ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
        (Mat.flatten W₂) idx ^ 2) / 4)
    (h2 : (2 * a ^ 2 / (1 - 2 * (a * (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
            (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
          (Mat.flatten W₂) idx|) + ((d₂ * d₃ : ℕ) : ℝ) * η))))) *
        (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
            (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
          (Mat.flatten W₂) idx|) + ((d₂ * d₃ : ℕ) : ℝ) * η)) ^ 2 ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
        (Mat.flatten W₂) idx ^ 2) / 4) :
    crossEntropy d₃ (dense (Mat.unflatten (Mat.flatten W₂ - lr • gh)) b₂
        (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label ≤
      crossEntropy d₃ (dense (Mat.unflatten (Mat.flatten W₂)) b₂
        (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label -
        lr * (∑ idx, gradAt
          (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
            (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
          (Mat.flatten W₂) idx ^ 2) / 2 :=
  linear_sgd_descends W₂ b₂
    (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) label gh
    ha hx hlr hη hgh hsmall h1 h2

-- ════════════════════════════════════════════════════════════════
-- § Input layer W₀: two frozen masks
-- ════════════════════════════════════════════════════════════════

/-- **Loss input-gradient through the relu→dense→relu chain** — the
    two-mask closed form `relu'(z₀ⱼ)·∑ₗ W₁ⱼₗ·relu'(z₁ₗ)·∑ₖ W₂ₗₖ·(softmax −
    onehot)ₖ` at a point with both pre-activations off the kinks. The
    pdiv-level form of the deepest cotangent `mlpCotOut0` delivers
    (cf. `mlpCotOut0_denote`). -/
theorem ce_head2_input_grad {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (label : Fin d₃)
    (z : Vec d₁) (hz0 : ∀ k, z k ≠ 0)
    (hz1 : ∀ k, dense W₁ b₁ (relu d₁ z) k ≠ 0) (j : Fin d₁) :
    pdiv (fun y : Vec d₁ => fun _ : Fin 1 => crossEntropy d₃
        (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ y)))) label) z j 0
      = (if z j > 0 then (1:ℝ) else 0) *
          ∑ l, W₁ j l *
            ((if dense W₁ b₁ (relu d₁ z) l > 0 then (1:ℝ) else 0) *
              ∑ k, W₂ l k *
                (softmax d₃
                  (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ z)))) k -
                  oneHot d₃ label k)) := by
  have hg1 : DifferentiableAt ℝ
      (fun u : Vec d₁ => fun _ : Fin 1 =>
        crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ u))) label)
      (relu d₁ z) := by
    rw [differentiableAt_pi]
    intro _
    have h2 : DifferentiableAt ℝ
        (fun u : Vec d₁ => relu d₂ (dense W₁ b₁ u)) (relu d₁ z) :=
      (relu_differentiableAt_of_smooth d₂ _ hz1).comp
        (f := fun u : Vec d₁ => dense W₁ b₁ u) _
        ((dense_differentiable W₁ b₁) _)
    have h3 : DifferentiableAt ℝ
        (fun u : Vec d₁ => dense W₂ b₂ (relu d₂ (dense W₁ b₁ u)))
        (relu d₁ z) :=
      ((dense_differentiable W₂ b₂) _).comp
        (f := fun u : Vec d₁ => relu d₂ (dense W₁ b₁ u)) _ h2
    exact (crossEntropy_differentiable d₃ label).differentiableAt.comp
      (f := fun u : Vec d₁ => dense W₂ b₂ (relu d₂ (dense W₁ b₁ u))) _ h3
  rw [show (fun y : Vec d₁ => fun _ : Fin 1 => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ y)))) label)
        = (fun u : Vec d₁ => fun _ : Fin 1 => crossEntropy d₃
            (dense W₂ b₂ (relu d₂ (dense W₁ b₁ u))) label) ∘ (relu d₁)
        from rfl,
      pdiv_comp _ _ _ (relu_differentiableAt_of_smooth d₁ z hz0) hg1]
  simp_rw [pdiv_relu d₁ z hz0 j, ite_mul, zero_mul]
  rw [Finset.sum_ite_eq]
  simp only [Finset.mem_univ, if_true]
  congr 1
  -- second hop: peel the middle dense, then reuse the one-relu head
  have hH : DifferentiableAt ℝ
      (fun u : Vec d₂ => fun _ : Fin 1 =>
        crossEntropy d₃ (dense W₂ b₂ (relu d₂ u)) label)
      (dense W₁ b₁ (relu d₁ z)) := by
    rw [differentiableAt_pi]
    intro _
    have h3 : DifferentiableAt ℝ
        (fun u : Vec d₂ => dense W₂ b₂ (relu d₂ u))
        (dense W₁ b₁ (relu d₁ z)) :=
      ((dense_differentiable W₂ b₂) _).comp (f := relu d₂) _
        (relu_differentiableAt_of_smooth d₂ _ hz1)
    exact (crossEntropy_differentiable d₃ label).differentiableAt.comp
      (f := fun u : Vec d₂ => dense W₂ b₂ (relu d₂ u)) _ h3
  rw [show (fun u : Vec d₁ => fun _ : Fin 1 => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense W₁ b₁ u))) label)
        = (fun u : Vec d₂ => fun _ : Fin 1 => crossEntropy d₃
            (dense W₂ b₂ (relu d₂ u)) label) ∘ (dense W₁ b₁) from rfl,
      pdiv_comp _ _ _ ((dense_differentiable W₁ b₁) _) hH]
  simp only [one_mul]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [pdiv_dense, ce_head_relu_input_grad W₂ b₂ label _ hz1 l, ite_mul,
    one_mul, zero_mul]

/-- The loss-of-`W₀` map is differentiable wherever both pre-activations
    are off the kinks. -/
theorem mlp_input_loss_differentiableAt {d₀ d₁ d₂ d₃ : Nat} (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀) (label : Fin d₃) (w : Vec (d₀ * d₁))
    (hz0 : ∀ k, dense (Mat.unflatten w) b₀ x k ≠ 0)
    (hz1 : ∀ k, dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x)) k ≠ 0) :
    DifferentiableAt ℝ
      (fun w' : Vec (d₀ * d₁) =>
        crossEntropy d₃ (dense W₂ b₂ (relu d₂
          (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w') b₀ x))))) label)
      w := by
  have h0 : DifferentiableAt ℝ
      (fun w' : Vec (d₀ * d₁) => dense (Mat.unflatten w') b₀ x) w :=
    (denseWeightMap_differentiable b₀ x) w
  have h1 : DifferentiableAt ℝ
      (fun w' : Vec (d₀ * d₁) => relu d₁ (dense (Mat.unflatten w') b₀ x))
      w :=
    (relu_differentiableAt_of_smooth d₁ _ hz0).comp
      (f := fun w' : Vec (d₀ * d₁) => dense (Mat.unflatten w') b₀ x) w h0
  have h2 : DifferentiableAt ℝ
      (fun w' : Vec (d₀ * d₁) =>
        dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w') b₀ x))) w :=
    ((dense_differentiable W₁ b₁) _).comp
      (f := fun w' : Vec (d₀ * d₁) =>
        relu d₁ (dense (Mat.unflatten w') b₀ x)) w h1
  have h3 : DifferentiableAt ℝ
      (fun w' : Vec (d₀ * d₁) =>
        relu d₂ (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w') b₀ x)))) w :=
    (relu_differentiableAt_of_smooth d₂ _ hz1).comp
      (f := fun w' : Vec (d₀ * d₁) =>
        dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w') b₀ x))) w h2
  have h4 : DifferentiableAt ℝ
      (fun w' : Vec (d₀ * d₁) => dense W₂ b₂ (relu d₂
        (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w') b₀ x))))) w :=
    ((dense_differentiable W₂ b₂) _).comp
      (f := fun w' : Vec (d₀ * d₁) =>
        relu d₂ (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w') b₀ x))))
      w h3
  exact (crossEntropy_differentiable d₃ label).differentiableAt.comp
    (f := fun w' : Vec (d₀ * d₁) => dense W₂ b₂ (relu d₂
      (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w') b₀ x))))) w h4

/-- **Closed form of the input-layer loss gradient at any two-margin
    point**: `∂L/∂W₀_{ij} = xᵢ·relu'(z₀ⱼ)·∑ₗ W₁ⱼₗ·relu'(z₁ₗ)·∑ₖ W₂ₗₖ·
    (softmax − onehot)ₖ` — the deepest fold (`mlp_input_total_loss_grad`)
    with both `pdiv` factors collapsed. The input-layer peer of
    `linear_loss_gradAt`. -/
theorem mlp_input_loss_gradAt {d₀ d₁ d₂ d₃ : Nat} (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀) (label : Fin d₃) (v : Vec (d₀ * d₁))
    (hz0 : ∀ k, dense (Mat.unflatten v) b₀ x k ≠ 0)
    (hz1 : ∀ k, dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) k ≠ 0)
    (i : Fin d₀) (j : Fin d₁) :
    gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
        (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label) v
        (finProdFinEquiv (i, j))
      = x i * ((if dense (Mat.unflatten v) b₀ x j > 0 then (1:ℝ) else 0) *
          ∑ l, W₁ j l *
            ((if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
                then (1:ℝ) else 0) *
              ∑ k, W₂ l k *
                (softmax d₃ (dense W₂ b₂ (relu d₂
                  (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k -
                  oneHot d₃ label k))) := by
  calc gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
        (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label) v
        (finProdFinEquiv (i, j))
      = pdiv (fun w => fun _ : Fin 1 => crossEntropy d₃ (dense W₂ b₂
            (relu d₂ (dense W₁ b₁
              (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
          v (finProdFinEquiv (i, j)) 0 :=
        gradAt_eq_pdiv _ _
          (mlp_input_loss_differentiableAt b₀ W₁ b₁ W₂ b₂ x label v
            hz0 hz1) _
    _ = pdiv (fun w => fun _ : Fin 1 => crossEntropy d₃ (dense W₂ b₂
            (relu d₂ (dense W₁ b₁
              (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
          (Mat.flatten (Mat.unflatten v)) (finProdFinEquiv (i, j)) 0 := by
        rw [Mat.flatten_unflatten]
    _ = ∑ k : Fin d₁,
          pdiv (fun w : Vec (d₀ * d₁) => dense (Mat.unflatten w) b₀ x)
              (Mat.flatten (Mat.unflatten v)) (finProdFinEquiv (i, j)) k
            * pdiv (fun z : Vec d₁ => fun _ : Fin 1 =>
                  crossEntropy d₃ (dense W₂ b₂ (relu d₂
                    (dense W₁ b₁ (relu d₁ z)))) label)
                (dense (Mat.unflatten v) b₀ x) k 0 :=
        IR.mlp_input_total_loss_grad (Mat.unflatten v) b₀ W₁ b₁ W₂ b₂ x
          label hz0 hz1 i j
    _ = ∑ k : Fin d₁, (if k = j then x i else 0)
            * pdiv (fun z : Vec d₁ => fun _ : Fin 1 =>
                  crossEntropy d₃ (dense W₂ b₂ (relu d₂
                    (dense W₁ b₁ (relu d₁ z)))) label)
                (dense (Mat.unflatten v) b₀ x) k 0 :=
        Finset.sum_congr rfl fun k _ => by
          rw [pdiv_dense_W b₀ x (Mat.unflatten v) i j k]
    _ = x i * pdiv (fun z : Vec d₁ => fun _ : Fin 1 =>
            crossEntropy d₃ (dense W₂ b₂ (relu d₂
              (dense W₁ b₁ (relu d₁ z)))) label)
          (dense (Mat.unflatten v) b₀ x) j 0 := by
        simp only [ite_mul, zero_mul]
        rw [Finset.sum_ite_eq']
        simp
    _ = x i * ((if dense (Mat.unflatten v) b₀ x j > 0 then (1:ℝ) else 0) *
          ∑ l, W₁ j l *
            ((if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
                then (1:ℝ) else 0) *
              ∑ k, W₂ l k *
                (softmax d₃ (dense W₂ b₂ (relu d₂
                  (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k -
                  oneHot d₃ label k))) := by
        rw [ce_head2_input_grad W₁ b₁ W₂ b₂ label _ hz0 hz1 j]

/-- The input-layer logit drift: through two dense layers and two
    1-Lipschitz ReLUs, a weight perturbation of `ℓ1` mass `‖e‖₁` moves every
    logit by at most `w₂·d₂·w₁·a·‖e‖₁`. The middle layer contributes its
    `ℓ1→ℓ1` operator factor `d₂·w₁` — unlike the first hop, the perturbation
    arriving at layer 1 is no longer column-structured. -/
theorem mlp_input_logit_drift {d₀ d₁ d₂ d₃ : Nat} (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀) {a w₁ w₂ : ℝ} (ha : 0 ≤ a)
    (hx : ∀ i, |x i| ≤ a) (hw₁ : 0 ≤ w₁) (hW₁ : ∀ j l, |W₁ j l| ≤ w₁)
    (hW₂ : ∀ l k, |W₂ l k| ≤ w₂)
    (v e : Vec (d₀ * d₁)) (k : Fin d₃) :
    |dense W₂ b₂ (relu d₂ (dense W₁ b₁
        (relu d₁ (dense (Mat.unflatten (v + e)) b₀ x)))) k -
      dense W₂ b₂ (relu d₂ (dense W₁ b₁
        (relu d₁ (dense (Mat.unflatten v) b₀ x)))) k| ≤
      w₂ * ((d₂ : ℝ) * (w₁ * (a * ∑ idx, |e idx|))) := by
  have hdiff : dense W₂ b₂ (relu d₂ (dense W₁ b₁
        (relu d₁ (dense (Mat.unflatten (v + e)) b₀ x)))) k -
      dense W₂ b₂ (relu d₂ (dense W₁ b₁
        (relu d₁ (dense (Mat.unflatten v) b₀ x)))) k =
      ∑ l, (relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten (v + e)) b₀ x))) l -
        relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten v) b₀ x))) l) * W₂ l k := by
    have h2 : (∑ l, relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten (v + e)) b₀ x))) l * W₂ l k) -
        (∑ l, relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten v) b₀ x))) l * W₂ l k) =
        ∑ l, (relu d₂ (dense W₁ b₁
            (relu d₁ (dense (Mat.unflatten (v + e)) b₀ x))) l -
          relu d₂ (dense W₁ b₁
            (relu d₁ (dense (Mat.unflatten v) b₀ x))) l) * W₂ l k := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun l _ => by ring
    show ((∑ l, relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten (v + e)) b₀ x))) l * W₂ l k) +
            b₂ k) -
        ((∑ l, relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten v) b₀ x))) l * W₂ l k) + b₂ k) = _
    linarith [h2]
  rw [hdiff]
  calc |∑ l, (relu d₂ (dense W₁ b₁
        (relu d₁ (dense (Mat.unflatten (v + e)) b₀ x))) l -
        relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten v) b₀ x))) l) * W₂ l k|
      ≤ ∑ l, |(relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten (v + e)) b₀ x))) l -
          relu d₂ (dense W₁ b₁
            (relu d₁ (dense (Mat.unflatten v) b₀ x))) l) * W₂ l k| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _l : Fin d₂, (w₁ * (a * ∑ idx, |e idx|)) * w₂ :=
        Finset.sum_le_sum fun l _ => by
          rw [abs_mul]
          refine mul_le_mul ?_ (hW₂ l k) (abs_nonneg _)
            (mul_nonneg hw₁ (mul_nonneg ha
              (Finset.sum_nonneg fun _ _ => abs_nonneg _)))
          exact le_trans (relu_entry_lipschitz d₂ _ _ l)
            (mlp_hidden_logit_drift b₀ W₁ b₁ x hx hw₁ hW₁ v e l)
    _ = (d₂ : ℝ) * ((w₁ * (a * ∑ idx, |e idx|)) * w₂) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]
    _ = w₂ * ((d₂ : ℝ) * (w₁ * (a * ∑ idx, |e idx|))) := by ring

/-- The layer-1 margin keeps the *middle* pre-activation off the kink along
    the segment: the perturbation arrives through one dense + ReLU, so the
    drift is at most `w₁·a·D` — the layer-1 analogue of
    `margin_keeps_offkink`. -/
theorem margin_keeps_offkink_mid {d₀ d₁ d₂ : Nat} (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (x : Vec d₀) {a w₁ D : ℝ}
    (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a)
    (hw₁ : 0 ≤ w₁) (hW₁ : ∀ j l, |W₁ j l| ≤ w₁)
    (v e : Vec (d₀ * d₁)) (he : (∑ idx, |e idx|) ≤ D)
    (hmargin1 : ∀ l, w₁ * (a * D) <
      |dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (l : Fin d₂) :
    dense W₁ b₁ (relu d₁ (dense (Mat.unflatten (v + t • e)) b₀ x)) l ≠ 0 ∧
      (0 < dense W₁ b₁ (relu d₁ (dense (Mat.unflatten (v + t • e)) b₀ x)) l ↔
        0 < dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l) := by
  refine sign_stable_of_close ?_ (hmargin1 l)
  have h1 := mlp_hidden_logit_drift b₀ W₁ b₁ x hx hw₁ hW₁ v (t • e) l
  have htm : (∑ idx, |(t • e) idx|) = t * ∑ idx, |e idx| := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun idx _ => by
      simp [abs_mul, abs_of_nonneg ht0]
  rw [htm] at h1
  have htsum : t * (∑ idx, |e idx|) ≤ D :=
    (mul_le_mul ht1 he (Finset.sum_nonneg fun _ _ => abs_nonneg _)
      zero_le_one).trans_eq (one_mul D)
  have h2 : w₁ * (a * (t * ∑ idx, |e idx|)) ≤ w₁ * (a * D) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left htsum ha) hw₁
  linarith

/-- **Segment-Lipschitz gradient for the input-layer loss, explicit
    constant.** Under both margins (neither ReLU layer's sign pattern can
    change along the step) and the small-step condition, the gradient
    entries drift by at most
    `(2·d₃·d₂²·w₁²·w₂²·a²/(1−2·w₂·d₂·w₁·a·D))·(t·D)`. The input-layer peer
    of `mlp_hidden_loss_grad_lipschitz`; the extra `d₂·w₁` is the middle
    layer's `ℓ1→ℓ1` operator factor. -/
theorem mlp_input_loss_grad_lipschitz {d₀ d₁ d₂ d₃ : Nat} (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀) (label : Fin d₃) {a w₁ w₂ D : ℝ}
    (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a)
    (hw₁ : 0 ≤ w₁) (hW₁ : ∀ j l, |W₁ j l| ≤ w₁)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ l k, |W₂ l k| ≤ w₂)
    (v d : Vec (d₀ * d₁)) (hd : (∑ idx, |d idx|) ≤ D)
    (hmargin0 : ∀ j, a * D < |dense (Mat.unflatten v) b₀ x j|)
    (hmargin1 : ∀ l, w₁ * (a * D) <
      |dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l|)
    (hsmall : 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))) < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1) (idx : Fin (d₀ * d₁)) :
    |gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
        (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x)))))  label)
        (v + t • d) idx -
      gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
        (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        v idx| ≤
      (2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 * a ^ 2 /
        (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))))) * (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D :=
    le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  have hden : (0:ℝ) < 1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))) := by
    linarith
  have hδ0 : (0:ℝ) ≤ w₂ * ((d₂ : ℝ) * (w₁ * (a * D))) :=
    mul_nonneg hw₂ (mul_nonneg (Nat.cast_nonneg d₂)
      (mul_nonneg hw₁ (mul_nonneg ha hD0)))
  obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
  have htmass : (∑ idx, |(t • d) idx|) = t * ∑ idx, |d idx| := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun idx _ => by
      simp [abs_mul, abs_of_nonneg ht0]
  -- base point off both kinks
  have hz0_v : ∀ k, dense (Mat.unflatten v) b₀ x k ≠ 0 := fun k h0 => by
    have h := hmargin0 k
    rw [h0, abs_zero] at h
    exact absurd h (not_lt.mpr (mul_nonneg ha hD0))
  have hz1_v : ∀ l,
      dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l ≠ 0 :=
    fun l h0 => by
      have h := hmargin1 l
      rw [h0, abs_zero] at h
      exact absurd h (not_lt.mpr (mul_nonneg hw₁ (mul_nonneg ha hD0)))
  -- both sign patterns frozen along the segment
  have hstab0 := fun k =>
    margin_keeps_offkink b₀ x ha hx v d hd hmargin0 t ht0 ht1 k
  have hstab1 := fun l =>
    margin_keeps_offkink_mid b₀ W₁ b₁ x ha hx hw₁ hW₁ v d hd hmargin1
      t ht0 ht1 l
  have hz0_t : ∀ k, dense (Mat.unflatten (v + t • d)) b₀ x k ≠ 0 :=
    fun k => (hstab0 k).1
  have hz1_t : ∀ l, dense W₁ b₁
      (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x)) l ≠ 0 :=
    fun l => (hstab1 l).1
  rw [mlp_input_loss_gradAt b₀ W₁ b₁ W₂ b₂ x label (v + t • d)
        hz0_t hz1_t i j,
      mlp_input_loss_gradAt b₀ W₁ b₁ W₂ b₂ x label v hz0_v hz1_v i j]
  -- the frozen masks
  have hmask0 : (if dense (Mat.unflatten (v + t • d)) b₀ x j > 0
        then (1:ℝ) else 0) =
      (if dense (Mat.unflatten v) b₀ x j > 0 then (1:ℝ) else 0) := by
    by_cases hp : dense (Mat.unflatten v) b₀ x j > 0
    · rw [if_pos hp, if_pos ((hstab0 j).2.mpr hp)]
    · rw [if_neg hp, if_neg (fun h => hp ((hstab0 j).2.mp h))]
  have hmask1 : ∀ l, (if dense W₁ b₁
        (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x)) l > 0
        then (1:ℝ) else 0) =
      (if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
        then (1:ℝ) else 0) := by
    intro l
    by_cases hp : dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
    · rw [if_pos hp, if_pos ((hstab1 l).2.mpr hp)]
    · rw [if_neg hp, if_neg (fun h => hp ((hstab1 l).2.mp h))]
  rw [hmask0]
  simp only [hmask1]
  by_cases hp : dense (Mat.unflatten v) b₀ x j > 0
  · -- live outer mask
    rw [if_pos hp]
    have hcollapse : x i * ((1:ℝ) *
          ∑ l, W₁ j l *
            ((if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
                then (1:ℝ) else 0) *
              ∑ k, W₂ l k *
                (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                  (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
                  oneHot d₃ label k))) -
        x i * ((1:ℝ) *
          ∑ l, W₁ j l *
            ((if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
                then (1:ℝ) else 0) *
              ∑ k, W₂ l k *
                (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                  (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k -
                  oneHot d₃ label k))) =
        x i * ∑ l, W₁ j l *
          ((if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
              then (1:ℝ) else 0) *
            ∑ k, W₂ l k *
              (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
                softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                  (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k)) := by
      rw [one_mul, one_mul, ← mul_sub, ← Finset.sum_sub_distrib]
      congr 1
      refine Finset.sum_congr rfl fun l _ => ?_
      rw [← mul_sub, ← mul_sub]
      congr 2
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun k _ => by ring
    rw [hcollapse, abs_mul]
    -- logit drift along the segment
    have hzdrift : ∀ k, |dense W₂ b₂ (relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x)))) k -
        dense W₂ b₂ (relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten v) b₀ x)))) k| ≤
        t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))) := by
      intro k
      have h1 := mlp_input_logit_drift b₀ W₁ b₁ W₂ b₂ x ha hx hw₁ hW₁
        hW₂ v (t • d) k
      rw [htmass] at h1
      have h2 : w₂ * ((d₂ : ℝ) * (w₁ * (a * (t * ∑ idx, |d idx|)))) ≤
          w₂ * ((d₂ : ℝ) * (w₁ * (a * (t * D)))) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left hd ht0) ha) hw₁)
          (Nat.cast_nonneg d₂)) hw₂
      have h3 : w₂ * ((d₂ : ℝ) * (w₁ * (a * (t * D)))) =
          t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))) := by ring
      linarith
    have hsm := fun k => FloatModel.softmax_perturb
      (dense W₂ b₂ (relu d₂ (dense W₁ b₁
        (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x)))))
      (dense W₂ b₂ (relu d₂ (dense W₁ b₁
        (relu d₁ (dense (Mat.unflatten v) b₀ x))))) hzdrift k
    have hδlt : 2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) < 1 := by
      nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
    have hexp : Real.exp (2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))))) -
        1 ≤ 2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
          (1 - 2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))))) :=
      FloatModel.exp_sub_one_le hδlt
    have hmono : 2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
          (1 - 2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))))) ≤
        2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
          (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) := by
      refine div_le_div_of_nonneg_left
        (by nlinarith [mul_nonneg ht0 hδ0]) hden ?_
      nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
    have hS : ∀ k, |softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
        softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
          (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k| ≤
        2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
          (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) :=
      fun k => le_trans (hsm k) (le_trans hexp hmono)
    -- contract through W₂ (per inner sum), the frozen mask, then W₁
    have hinner : ∀ l, |∑ k, W₂ l k *
          (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
            (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
            softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
              (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k)| ≤
        (d₃ : ℝ) * (w₂ * (2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
          (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))))) := by
      intro l
      calc |∑ k, W₂ l k *
            (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
              (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
              softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k)|
          ≤ ∑ k, |W₂ l k *
              (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
                softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                  (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k)| :=
            Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _k : Fin d₃, w₂ *
              (2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
                (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))))) :=
            Finset.sum_le_sum fun k _ => by
              rw [abs_mul]
              exact mul_le_mul (hW₂ l k) (hS k) (abs_nonneg _) hw₂
        _ = (d₃ : ℝ) * (w₂ *
              (2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
                (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))))) := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
              nsmul_eq_mul]
    have hsum : |∑ l, W₁ j l *
          ((if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
              then (1:ℝ) else 0) *
            ∑ k, W₂ l k *
              (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
                softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                  (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k))| ≤
        (d₂ : ℝ) * (w₁ * ((d₃ : ℝ) * (w₂ *
          (2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
            (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))))))) := by
      calc |∑ l, W₁ j l *
            ((if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
                then (1:ℝ) else 0) *
              ∑ k, W₂ l k *
                (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                  (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
                  softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                    (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k))|
          ≤ ∑ l, |W₁ j l *
              ((if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
                  then (1:ℝ) else 0) *
                ∑ k, W₂ l k *
                  (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                    (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
                    softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                      (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k))| :=
            Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _l : Fin d₂, w₁ * ((d₃ : ℝ) * (w₂ *
              (2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
                (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))))))) := by
            refine Finset.sum_le_sum fun l _ => ?_
            rw [abs_mul]
            refine mul_le_mul (hW₁ j l) ?_ (abs_nonneg _) hw₁
            rw [abs_mul]
            refine le_trans (mul_le_of_le_one_left (abs_nonneg _) ?_)
              (hinner l)
            split_ifs <;> simp
        _ = (d₂ : ℝ) * (w₁ * ((d₃ : ℝ) * (w₂ *
              (2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
                (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))))))) := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
              nsmul_eq_mul]
    calc |x i| * |∑ l, W₁ j l *
          ((if dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l > 0
              then (1:ℝ) else 0) *
            ∑ k, W₂ l k *
              (softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                (relu d₁ (dense (Mat.unflatten (v + t • d)) b₀ x))))) k -
                softmax d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁
                  (relu d₁ (dense (Mat.unflatten v) b₀ x))))) k))|
        ≤ a * ((d₂ : ℝ) * (w₁ * ((d₃ : ℝ) * (w₂ *
            (2 * (t * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) /
              (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))))))))) :=
          mul_le_mul (hx i) hsum (abs_nonneg _) ha
      _ = (2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 * a ^ 2 /
            (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D)))))) * (t * D) := by
          ring
  · -- dead outer mask: both gradients vanish
    rw [if_neg hp]
    simp only [zero_mul, mul_zero, sub_self, abs_zero]
    have hC0 : 0 ≤ 2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 * a ^ 2 /
        (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * D))))) :=
      div_nonneg (by positivity) hden.le
    exact mul_nonneg hC0 (mul_nonneg ht0 hD0)

/-- **One inexact SGD step on the MLP's input weights provably decreases
    the cross-entropy loss.** The deepest descent capstone: both ReLU
    layers' margins at the step radius `D = lr·(‖∇L‖₁ + d₀d₁·η)` freeze the
    masks, the segment-Lipschitz constant
    `C = 2·d₃·d₂²·w₁²·w₂²·a²/(1−2·w₂·d₂·w₁·a·D)` is proven, and the loss
    drops by ≥ `lr·‖∇L‖₂²/2`. Remaining hypotheses are checkable
    arithmetic. The input-layer peer of `linear_sgd_descends`; with this,
    every MLP weight layer's descent statement is discharged. -/
theorem mlp_input_sgd_descends {d₀ d₁ d₂ d₃ : Nat} (W₀ : Mat d₀ d₁)
    (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃)
    (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) (gh : Vec (d₀ * d₁))
    {lr η a w₁ w₂ : ℝ}
    (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a)
    (hw₁ : 0 ≤ w₁) (hW₁ : ∀ j l, |W₁ j l| ≤ w₁)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ l k, |W₂ l k| ≤ w₂)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx -
      gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
          (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        (Mat.flatten W₀) idx| ≤ η)
    (hmargin0 : ∀ j, a * (lr * ((∑ idx, |gradAt
        (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
          (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        (Mat.flatten W₀) idx|) + ((d₀ * d₁ : ℕ) : ℝ) * η)) <
      |dense W₀ b₀ x j|)
    (hmargin1 : ∀ l, w₁ * (a * (lr * ((∑ idx, |gradAt
        (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
          (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        (Mat.flatten W₀) idx|) + ((d₀ * d₁ : ℕ) : ℝ) * η))) <
      |dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) l|)
    (hsmall : 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * (lr * ((∑ idx, |gradAt
        (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
          (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        (Mat.flatten W₀) idx|) + ((d₀ * d₁ : ℕ) : ℝ) * η)))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt
        (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
          (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        (Mat.flatten W₀) idx|) ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
          (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        (Mat.flatten W₀) idx ^ 2) / 4)
    (h2 : (2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 * a ^ 2 /
        (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
            (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
          (Mat.flatten W₀) idx|) + ((d₀ * d₁ : ℕ) : ℝ) * η)))))))) *
        (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
            (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
          (Mat.flatten W₀) idx|) + ((d₀ * d₁ : ℕ) : ℝ) * η)) ^ 2 ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
          (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        (Mat.flatten W₀) idx ^ 2) / 4) :
    crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁
        (dense (Mat.unflatten (Mat.flatten W₀ - lr • gh)) b₀ x))))) label ≤
      crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁
        (dense (Mat.unflatten (Mat.flatten W₀)) b₀ x))))) label -
        lr * (∑ idx, gradAt
          (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
            (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
          (Mat.flatten W₀) idx ^ 2) / 2 := by
  set f : Vec (d₀ * d₁) → ℝ :=
    fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
      (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label with hf
  have hden : (0:ℝ) < 1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * (lr * ((∑ idx,
      |gradAt f (Mat.flatten W₀) idx|) +
        ((d₀ * d₁ : ℕ) : ℝ) * η)))))) := by
    linarith
  have hC0 : (0:ℝ) ≤ 2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 *
      a ^ 2 / (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * (lr * ((∑ idx,
        |gradAt f (Mat.flatten W₀) idx|) +
          ((d₀ * d₁ : ℕ) : ℝ) * η))))))) :=
    div_nonneg (by positivity) hden.le
  -- the margins, restated at the `unflatten ∘ flatten` parameter point
  have hmargin0' : ∀ j, a * (lr * ((∑ idx,
      |gradAt f (Mat.flatten W₀) idx|) + ((d₀ * d₁ : ℕ) : ℝ) * η)) <
      |dense (Mat.unflatten (Mat.flatten W₀)) b₀ x j| := fun j => by
    rw [Mat.unflatten_flatten]
    exact hmargin0 j
  have hmargin1' : ∀ l, w₁ * (a * (lr * ((∑ idx,
      |gradAt f (Mat.flatten W₀) idx|) + ((d₀ * d₁ : ℕ) : ℝ) * η))) <
      |dense W₁ b₁ (relu d₁
        (dense (Mat.unflatten (Mat.flatten W₀)) b₀ x)) l| := fun l => by
    rw [Mat.unflatten_flatten]
    exact hmargin1 l
  -- ℓ1 radius of the step
  have hD : (∑ idx, |(-(lr • gh)) idx|) ≤
      lr * ((∑ idx, |gradAt f (Mat.flatten W₀) idx|) +
        ((d₀ * d₁ : ℕ) : ℝ) * η) := by
    calc (∑ idx, |(-(lr • gh)) idx|) = ∑ idx, lr * |gh idx| := by
          refine Finset.sum_congr rfl fun idx _ => ?_
          simp [abs_mul, abs_of_nonneg hlr]
      _ ≤ ∑ idx, lr * (|gradAt f (Mat.flatten W₀) idx| + η) := by
          refine Finset.sum_le_sum fun idx _ => ?_
          refine mul_le_mul_of_nonneg_left ?_ hlr
          have h3 : |gh idx| ≤ |gh idx - gradAt f (Mat.flatten W₀) idx| +
              |gradAt f (Mat.flatten W₀) idx| := by
            simpa using abs_sub_le (gh idx) (gradAt f (Mat.flatten W₀) idx) 0
          linarith [hgh idx]
      _ = lr * ((∑ idx, |gradAt f (Mat.flatten W₀) idx|) +
            ((d₀ * d₁ : ℕ) : ℝ) * η) := by
          rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const,
            Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hmain := sgd_descends f (Mat.flatten W₀) gh hlr hη hC0 hgh
    (fun t ht => mlp_input_loss_differentiableAt b₀ W₁ b₁ W₂ b₂ x label _
      (fun k => (margin_keeps_offkink b₀ x ha hx (Mat.flatten W₀)
        (-(lr • gh)) hD hmargin0' t ht.1 ht.2 k).1)
      (fun l => (margin_keeps_offkink_mid b₀ W₁ b₁ x ha hx hw₁ hW₁
        (Mat.flatten W₀) (-(lr • gh)) hD hmargin1' t ht.1 ht.2 l).1))
    (fun t ht idx => by
      have := mlp_input_loss_grad_lipschitz b₀ W₁ b₁ W₂ b₂ x label ha hx
        hw₁ hW₁ hw₂ hW₂ (Mat.flatten W₀) (-(lr • gh)) hD hmargin0'
        hmargin1' hsmall t ht idx
      simpa [hf] using this)
    h1 h2
  simpa [hf] using hmain

-- ════════════════════════════════════════════════════════════════
-- § Output layer η-composition: feed the FloatBridge budget into the
--   output-layer descent slot, so "one binary32 output-layer SGD step
--   decreases the loss" holds with NO abstract gradient-accuracy parameter.
-- ════════════════════════════════════════════════════════════════

/-- **One binary32 SGD step on the MLP's output weights provably decreases the
    cross-entropy loss — with NO abstract gradient-accuracy parameter.** The
    output-layer rung of the η-composition (Item D / G1 for the MLP). Since the
    top dense layer sits directly below the softmax-CE loss with no ReLU between,
    the loss-of-`W₂` map *is* the linear net's loss at the hidden activation
    `a₁ = relu(dense W₁ b₁ (relu(dense W₀ b₀ x)))` — so this is
    `linear_float_sgd_descends` instantiated there. The gradient is the *actual*
    binary32 output-layer gradient `M.linearFloatGrad W₂ b₂ a₁` and its accuracy
    `η = mulErr u a 1 0 (cotErr …)` is *proven* (by `linear_grad_close`, inside
    the linear theorem), not assumed. No margin needed — the output layer never
    crosses a kink.

    The hidden/input rungs (`mlp_{hidden,input}_sgd_descends`) still take an
    abstract `η`: their float gradients run back through the ReLU masks and the
    `W₂`-cotangent fan-in, so the η-composition there needs a per-layer
    float-backward grad-close (a `mlp_w{1,0}_grad_close`) under the descent
    margins — the joint-step refinement flagged at the top of this file, left
    open. -/
theorem mlp_output_float_sgd_descends {d₀ d₁ d₂ d₃ : Nat} (M : FloatModel)
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) (fexp : ℝ → ℝ)
    {lr a eexp δ : ℝ}
    (ha : 0 ≤ a)
    (hx : ∀ i, |relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))) i| ≤ a)
    (hlr : 0 ≤ lr) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1) (hδ0 : 0 ≤ δ)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp d₃ < 1)
    (hδ : ∀ k', |M.dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) k' -
        dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) k'| ≤ δ)
    (hsmall : 2 * (a * (lr * ((∑ idx, |gradAt
        (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
        (Mat.flatten W₂) idx|) + ((d₂ * d₃ : ℕ) : ℝ) *
          FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ d₃)))) < 1)
    (h1 : lr * (FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ d₃)) *
        (∑ idx, |gradAt
          (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
            (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
          (Mat.flatten W₂) idx|) ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
        (Mat.flatten W₂) idx ^ 2) / 4)
    (h2 : (2 * a ^ 2 / (1 - 2 * (a * (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
            (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
          (Mat.flatten W₂) idx|) + ((d₂ * d₃ : ℕ) : ℝ) *
            FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ d₃)))))) *
        (lr * ((∑ idx, |gradAt
          (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
            (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
          (Mat.flatten W₂) idx|) + ((d₂ * d₃ : ℕ) : ℝ) *
            FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ d₃))) ^ 2 ≤
      lr * (∑ idx, gradAt
        (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
        (Mat.flatten W₂) idx ^ 2) / 4) :
    crossEntropy d₃ (dense (Mat.unflatten (Mat.flatten W₂ -
        lr • M.linearFloatGrad W₂ b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) fexp label)) b₂
        (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label ≤
      crossEntropy d₃ (dense (Mat.unflatten (Mat.flatten W₂)) b₂
        (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label -
        lr * (∑ idx, gradAt
          (fun w => crossEntropy d₃ (dense (Mat.unflatten w) b₂
            (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
          (Mat.flatten W₂) idx ^ 2) / 2 :=
  linear_float_sgd_descends M W₂ b₂
    (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) label fexp
    ha hx hlr heexp0 heexp1 hδ0 hfexp hρ1 hδ hsmall h1 h2

-- ════════════════════════════════════════════════════════════════
-- § Hidden layer W₁: the float-backward grad-close (the joint-step engine)
-- ════════════════════════════════════════════════════════════════

open FloatModel in
/-- **The binary32 hidden-layer (W₁) gradient is within an explicit budget of
    the certified one**, per entry — the float-backward grad-close that the
    hidden η-composition needs. With the layer-1 input activation `a₀` *frozen
    exact* (the descent moves only `W₁`), the rendered trainer computes the
    `W₁` gradient as `fl(a₀ᵢ · c̃₁ⱼ)` where the float layer-1 cotangent
    `c̃₁ = mask(z̃₁, W₂ᵀ·c̃₂)` reads the float pre-activation `z̃₁ = M.dense W₁ b₁ a₀`
    and the float softmax−onehot head `c̃₂` at the float logits. This is within
    `mulErr M.u a … 0 (layerBudget … (cotErr …))` of the certified
    `a₀ᵢ · mask(z₁, W₂ᵀ·(softmax−onehot))ⱼ` (= `mlp_hidden_loss_gradAt`), built
    from three reusable closes: the head (`softmax_ce_cot_close`, accuracy
    `cotErr`), the masked `W₂ᵀ` contraction (`cot_step_close`, **under the
    quantitative margin** `E₁ < |z₁ⱼ|` — forward rounding must not flip the
    layer-1 ReLU), and the final input multiply (`mul_close`, with the *exact*
    `a₀` operand, `ea = 0`, exactly as the linear grad-close). -/
theorem mlp_w1_grad_close {d₁ d₂ d₃ : Nat} (M : FloatModel)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (a₀ : Vec d₁) (label : Fin d₃) (fexp : ℝ → ℝ)
    {a w₁ β₁ w₂ β₂ eexp : ℝ}
    (ha : 0 ≤ a) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp d₃ < 1)
    (hx : ∀ i, |a₀ i| ≤ a)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w₂) (hb₂ : ∀ j, |b₂ j| ≤ β₂)
    (hmargin : ∀ j', layerBudget M.u d₁ w₁ β₁ a 0 <
      |Proofs.dense W₁ b₁ a₀ j'|)
    (i : Fin d₁) (j : Fin d₂) :
    |M.mul (a₀ i)
        (reluMask (M.dense W₁ b₁ a₀)
          (M.dense (fun j' i' => W₂ i' j') (fun _ => 0)
            (M.softmaxCECotF fexp
              (M.dense W₂ b₂ (relu d₂ (M.dense W₁ b₁ a₀))) label)) j) -
      a₀ i * reluMask (Proofs.dense W₁ b₁ a₀)
        (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0)
          (fun k => softmax d₃
            (Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁ a₀))) k -
            oneHot d₃ label k)) j| ≤
    FloatModel.mulErr M.u a (layerAct d₃ w₂ 0 1) 0
      (layerBudget M.u d₃ w₂ 0 1
        (FloatModel.cotErr M.u eexp
          (layerBudget M.u d₂ w₂ β₂ (layerAct d₁ w₁ β₁ a)
            (layerBudget M.u d₁ w₁ β₁ a 0)) d₃)) := by
  set E₁ := layerBudget M.u d₁ w₁ β₁ a 0 with hE₁
  have hE₁0 : 0 ≤ E₁ := layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl
  -- layer-1 forward (a₀ exact ⇒ inherited error 0)
  have l1 : ∀ j', |M.dense W₁ b₁ a₀ j' - Proofs.dense W₁ b₁ a₀ j'| ≤ E₁ :=
    fun j' => (M.dense_close_fresh W₁ b₁ a₀ j').trans
      (M.denseErr_le_uniform hw₁ le_rfl hW₁ hb₁ hx j')
  have r1 : ∀ j', |relu d₂ (M.dense W₁ b₁ a₀) j' -
      relu d₂ (Proofs.dense W₁ b₁ a₀) j'| ≤ E₁ := fun j' => relu_close _ _ _ l1 j'
  have ha₁ : ∀ j', |relu d₂ (Proofs.dense W₁ b₁ a₀) j'| ≤ layerAct d₁ w₁ β₁ a :=
    fun j' => (relu_abs_le _ j').trans (dense_abs_le ha hW₁ hb₁ hx j')
  -- layer-2 forward (logits), inherited error E₁
  set δ := layerBudget M.u d₂ w₂ β₂ (layerAct d₁ w₁ β₁ a) E₁ with hδdef
  have hδ0 : 0 ≤ δ := layerBudget_nonneg M.u_nonneg hw₂ hβ₂
    (layerAct_nonneg hw₁ hβ₁ ha) hE₁0
  have l2 : ∀ k, |M.dense W₂ b₂ (relu d₂ (M.dense W₁ b₁ a₀)) k -
      Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁ a₀)) k| ≤ δ := fun k =>
    (M.dense_close W₂ b₂ _ _ E₁ hE₁0 r1 k).trans
      (M.denseErr_le_uniform hw₂ hE₁0 hW₂ hb₂ ha₁ k)
  -- the float softmax−onehot head within `cotErr`
  have hcot2 : ∀ k, |M.softmaxCECotF fexp
      (M.dense W₂ b₂ (relu d₂ (M.dense W₁ b₁ a₀))) label k -
      (softmax d₃ (Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁ a₀))) k -
        oneHot d₃ label k)| ≤ FloatModel.cotErr M.u eexp δ d₃ := fun k =>
    M.softmax_ce_cot_close fexp _ _ label heexp0 heexp1 hfexp hρ1 l2 k
  -- the real cotangent `softmax − onehot ∈ [−1, 1]`
  have hC2 : ∀ k, |softmax d₃
      (Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁ a₀))) k -
      oneHot d₃ label k| ≤ 1 := by
    intro k
    have hD : 0 < ∑ t, Real.exp (Proofs.dense W₂ b₂
        (relu d₂ (Proofs.dense W₁ b₁ a₀)) t) :=
      Finset.sum_pos (fun t _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩
    have hs0 : 0 ≤ softmax d₃
        (Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁ a₀))) k :=
      div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun t _ => (Real.exp_pos _).le)
    have hs1 : softmax d₃
        (Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁ a₀))) k ≤ 1 :=
      (div_le_one hD).mpr
        (Finset.single_le_sum (fun t _ => (Real.exp_pos _).le) (Finset.mem_univ k))
    simp only [oneHot]
    by_cases h : k = label
    · rw [if_pos h, abs_le]; constructor <;> linarith
    · rw [if_neg h, abs_le]; constructor <;> linarith
  -- the masked W₂ᵀ contraction within `layerBudget … cotErr`
  have hcot1 := M.cot_step_close W₂ (M.dense W₁ b₁ a₀) (Proofs.dense W₁ b₁ a₀)
    (M.softmaxCECotF fexp (M.dense W₂ b₂ (relu d₂ (M.dense W₁ b₁ a₀))) label)
    (fun k => softmax d₃
      (Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁ a₀))) k - oneHot d₃ label k)
    hw₂ (by norm_num) (M.cotErr_nonneg heexp0 hδ0 hρ1) hW₂ hC2 hcot2 l1 hmargin j
  -- the real layer-1 cotangent magnitude
  have hc1 : |reluMask (Proofs.dense W₁ b₁ a₀)
      (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0)
        (fun k => softmax d₃
          (Proofs.dense W₂ b₂ (relu d₂ (Proofs.dense W₁ b₁ a₀))) k -
          oneHot d₃ label k)) j| ≤ layerAct d₃ w₂ 0 1 :=
    (reluMask_abs_le _ _ j).trans
      (dense_abs_le (by norm_num) (fun j' i' => hW₂ i' j') (fun _ => by simp) hC2 j)
  -- the final input multiply: exact left operand `a₀` (`ea = 0`)
  exact M.mul_close (by simp : |a₀ i - a₀ i| ≤ (0:ℝ)) hcot1 (hx i) hc1

end Proofs
