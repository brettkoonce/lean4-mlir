import LeanMlir.Proofs.Training.SgdDescent.Linear
import LeanMlir.Proofs.Nets.Small.MlpTrainStep

/-! # Lipschitz constants for the MLP softmax-CE loss — descent through the ReLU kinks

`SgdDescent.Linear` discharged `sgd_descends`' smoothness hypothesis for the
Chapter-1 linear net. This file extends the discharge through the Chapter-2
MLP (`dense → relu → dense → relu → dense`), layer by layer:

* **Output layer `W₂` — free.** The top dense layer sees the loss with no
  ReLU in between, so its descent statement IS the linear one at the hidden
  activation `a₁` (`linear_sgd_descends` at `x := a₁`).

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
dominance conditions ⇒ **one inexact SGD step on that layer's weights, at one
example `(x, label)` with every other parameter fixed, decreases that example's
cross-entropy loss by ≥ lr·‖∇L‖₂²/2.** Smoothness is proven, not assumed; the
oracle accuracy, the margins, the small-step and the two dominance conditions
remain hypotheses. `mlp_output_float_sgd_descends`, `mlp_hidden_float_sgd_descends`
and `mlp_input_float_sgd_descends` replace the oracle accuracy by the proven
accuracy of the FloatModel binary32 gradient. Bias columns are the same argument
with the layer input replaced by the constant `1` and are omitted. The joint
all-layers step (every parameter moving at once, logits no longer affine in the
moving parameters) is not proved here. -/

namespace Proofs

open StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Helpers: ReLU is 1-Lipschitz; margins freeze signs; ℓ1 column mass
-- ════════════════════════════════════════════════════════════════

/-- **Margins freeze signs.** If a value drifts by at most `c` and sits at
    distance more than `c` from the kink, the drifted value is still off the
    kink *with the same sign* — the ReLU mask cannot flip. -/
theorem sign_stable_of_close {zt z c : ℝ} (hc : |zt - z| ≤ c)
    (hm : c < |z|) : zt ≠ 0 ∧ (0 < zt ↔ 0 < z) := by
  have habs := abs_le.mp hc
  rcases (abs_pos.mp (((abs_nonneg _).trans hc).trans_lt hm)).lt_or_gt with hneg | hpos
  · have hzt : zt < 0 := by rw [abs_of_neg hneg] at hm; linarith [habs.2]
    exact ⟨ne_of_lt hzt, by constructor <;> intro h <;> linarith⟩
  · have hzt : 0 < zt := by rw [abs_of_pos hpos] at hm; linarith [habs.1]
    exact ⟨ne_of_gt hzt, ⟨fun _ => hpos, fun _ => hzt⟩⟩

/-- A dense layer's output moves by at most `w·‖Δinput‖₁` per entry — the
    `ℓ1→ℓ∞` operator bound used at every dense crossing of the chain. -/
theorem dense_input_drift {m n : Nat} (W : Mat m n) (b : Vec n)
    {wb : ℝ} (hW : ∀ i j, |W i j| ≤ wb)
    (u u' : Vec m) (j : Fin n) :
    |dense W b u' j - dense W b u j| ≤ wb * ∑ i, |u' i - u i| := by
  have hdiff : dense W b u' j - dense W b u j =
      ∑ i, (u' i - u i) * W i j := by
    simp only [dense, add_sub_add_right_eq_sub, ← Finset.sum_sub_distrib, sub_mul]
  rw [hdiff]
  calc |∑ i, (u' i - u i) * W i j|
      ≤ ∑ i, |(u' i - u i) * W i j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, |u' i - u i| * wb :=
        Finset.sum_le_sum fun i _ => by
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_left (hW i j) (abs_nonneg _)
    _ = wb * ∑ i, |u' i - u i| := by
        rw [← Finset.sum_mul]
        ring

/-- **A per-entry drift inside the margin keeps a pre-activation off the kink along the
    whole segment** — for any parameter map `Z` whose entries move by at most `ρ·‖e‖₁`, the
    margin `ρ·D < |Z v k|` at step radius `D` freezes every sign on `[v, v+e]`. -/
theorem margin_keeps_offkink_of_drift {P n : Nat} (Z : Vec P → Vec n) {ρ D : ℝ}
    (hρ : 0 ≤ ρ) (hZ : ∀ v e k, |Z (v + e) k - Z v k| ≤ ρ * ∑ idx, |e idx|) (v e : Vec P)
    (he : (∑ idx, |e idx|) ≤ D) (hm : ∀ k, ρ * D < |Z v k|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (k : Fin n) :
    Z (v + t • e) k ≠ 0 ∧ (0 < Z (v + t • e) k ↔ 0 < Z v k) :=
  sign_stable_of_close ((hZ v (t • e) k).trans
    (mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) hρ)) (hm k)

/-- The `ℓ1` mass of a flattened weight perturbation, summed column by
    column, is the total `ℓ1` mass — `finProdFinEquiv` partitions the
    flat index set into the columns. -/
theorem sum_abs_flatten_cols {m n : Nat} (d : Vec (m * n)) :
    ∑ j : Fin n, ∑ i : Fin m, |d (finProdFinEquiv (i, j))| =
      ∑ idx, |d idx| := by
  rw [sum_finProdFinEquiv fun idx => |d idx|]; exact Finset.sum_comm

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
        0 < dense (Mat.unflatten v) b x j) :=
  margin_keeps_offkink_of_drift (fun w => dense (Mat.unflatten w) b x) ha
    (dense_unflatten_drift b x ha hx) v e he hmargin t ht0 ht1 j

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
    fun_prop
  rw [show (fun y : Vec d₂ => fun _ : Fin 1 =>
          crossEntropy d₃ (dense W₂ b₂ (relu d₂ y)) label)
        = (fun z' : Vec d₂ => fun _ : Fin 1 =>
            crossEntropy d₃ (dense W₂ b₂ z') label) ∘ (relu d₂) from rfl,
      pdiv_comp _ _ _ (relu_differentiableAt_of_smooth d₂ z hz) hg]
  simp_rw [pdiv_relu d₂ z hz j, ite_mul, zero_mul]
  rw [Finset.sum_ite_eq]
  simp only [Finset.mem_univ, ite_true]
  rw [ce_dense_input_grad]

/-- ReLU then a dense layer: an input drift of `ℓ1` mass `B` moves each output entry by at
    most `w·B`. -/
theorem dense_relu_drift {m n : Nat} (W : Mat m n) (b : Vec n) {wb : ℝ} (hw : 0 ≤ wb)
    (hW : ∀ i j, |W i j| ≤ wb) (u u' : Vec m) (j : Fin n) :
    |dense W b (relu m u') j - dense W b (relu m u) j| ≤ wb * ∑ i, |u' i - u i| :=
  (dense_input_drift W b hW _ _ j).trans
    (mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => relu_entry_lipschitz m _ _ i) hw)

namespace MlpSlot

/-- **Segment-Lipschitz gradient for an MLP-slot loss, explicit constant.** For a map `Z`
    into a ReLU layer's pre-activation whose entries move by at most `σ·‖e‖₁` (`hZ`) and whose
    `ℓ1` drift is at most `ρ·‖e‖₁` (`hZ1`), and whose loss gradient at every off-kink point is
    a fixed row `J` (row mass `≤ ρ`) contracted with the mask and the `W₂` head (`hgrad`, needed
    only where `Q` holds): the margin `σ·D` freezes the mask along `[v, v+d]`, the row factors
    out, and the difference collapses to the softmax drift. The hidden layer is the instance
    `σ = ρ = a`; the input layer takes `Z` = the middle pre-activation, `σ = w₁·a`,
    `ρ = d₂·w₁·a`, `J` = `xᵢ`·relu₀'s frozen mask·`W₁`'s row, `Q` = relu₀'s signs frozen. -/
theorem loss_grad_lipschitz {P d₂ d₃ : Nat} (Z : Vec P → Vec d₂) (W₂ : Mat d₂ d₃)
    (b₂ : Vec d₃) (label : Fin d₃) {σ ρ w₂ D : ℝ} (hσ : 0 ≤ σ)
    (hZ : ∀ v e l, |Z (v + e) l - Z v l| ≤ σ * ∑ idx, |e idx|)
    (hZ1 : ∀ v e, ∑ l, |Z (v + e) l - Z v l| ≤ ρ * ∑ idx, |e idx|)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ j k, |W₂ j k| ≤ w₂)
    (J : Fin d₂ → ℝ) (hJ : ∑ l, |J l| ≤ ρ) (idx : Fin P) (Q : Vec P → Prop)
    (hgrad : ∀ v' : Vec P, Q v' → (∀ l, Z v' l ≠ 0) →
      gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂ (Z w))) label) v' idx =
        ∑ l, J l * ((if Z v' l > 0 then (1:ℝ) else 0) *
          ∑ k, W₂ l k *
            (softmax d₃ (dense W₂ b₂ (relu d₂ (Z v'))) k - oneHot d₃ label k)))
    (v d : Vec P) (hd : (∑ idx, |d idx|) ≤ D) (hm : ∀ l, σ * D < |Z v l|)
    (hsmall : 2 * (w₂ * (ρ * D)) < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1) (hQv : Q v) (hQt : Q (v + t • d)) :
    |gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂ (Z w))) label)
        (v + t • d) idx -
      gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂ (Z w))) label) v idx| ≤
      (2 * (d₃ : ℝ) * w₂ ^ 2 * ρ ^ 2 / (1 - 2 * (w₂ * (ρ * D)))) * (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D := le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  have hρ : 0 ≤ ρ := le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hJ
  have hden : (0:ℝ) < 1 - 2 * (w₂ * (ρ * D)) := by linarith
  have hδ0 : (0:ℝ) ≤ w₂ * (ρ * D) := mul_nonneg hw₂ (mul_nonneg hρ hD0)
  -- the margin freezes every mask along the segment
  have hstab := fun l => margin_keeps_offkink_of_drift Z hσ hZ v d hd hm t ht0 ht1 l
  have hz_v : ∀ l, Z v l ≠ 0 := fun l => abs_pos.mp ((mul_nonneg hσ hD0).trans_lt (hm l))
  rw [hgrad (v + t • d) hQt (fun l => (hstab l).1), hgrad v hQv hz_v]
  have hmask : ∀ l, (if Z (v + t • d) l > 0 then (1:ℝ) else 0) =
      if Z v l > 0 then (1:ℝ) else 0 := fun l => if_congr (hstab l).2 rfl rfl
  simp only [hmask]
  -- the softmax drift along the segment
  have hzdrift : ∀ k, |dense W₂ b₂ (relu d₂ (Z (v + t • d))) k -
      dense W₂ b₂ (relu d₂ (Z v)) k| ≤ t * (w₂ * (ρ * D)) := fun k => by
    refine (dense_relu_drift W₂ b₂ hw₂ hW₂ _ _ k).trans ?_
    have h1 := hZ1 v (t • d)
    rw [smul_l1_mass d ht0] at h1
    calc w₂ * ∑ l, |Z (v + t • d) l - Z v l| ≤ w₂ * (ρ * (t * D)) :=
          mul_le_mul_of_nonneg_left (h1.trans (mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left hd ht0) hρ)) hw₂
      _ = t * (w₂ * (ρ * D)) := by ring
  have hS := softmax_seg_drift _ _ ht0 ht1 hδ0 hsmall hzdrift
  have hM0 : (0:ℝ) ≤ (d₃ : ℝ) * (w₂ * (2 * (t * (w₂ * (ρ * D))) /
      (1 - 2 * (w₂ * (ρ * D))))) :=
    mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₂
      (div_nonneg (by positivity) hden.le))
  -- per row: the frozen mask, then the `W₂` contraction of the softmax drift
  have hrow : ∀ l, |J l * ((if Z v l > 0 then (1:ℝ) else 0) *
        ∑ k, W₂ l k * (softmax d₃ (dense W₂ b₂ (relu d₂ (Z (v + t • d)))) k -
          oneHot d₃ label k)) -
      J l * ((if Z v l > 0 then (1:ℝ) else 0) *
        ∑ k, W₂ l k * (softmax d₃ (dense W₂ b₂ (relu d₂ (Z v))) k -
          oneHot d₃ label k))| ≤
      |J l| * ((d₃ : ℝ) * (w₂ * (2 * (t * (w₂ * (ρ * D))) /
        (1 - 2 * (w₂ * (ρ * D)))))) := fun l => by
    rw [← mul_sub, ← mul_sub, abs_mul, abs_mul, ← Finset.sum_sub_distrib]
    refine mul_le_mul_of_nonneg_left (le_trans (mul_le_of_le_one_left (abs_nonneg _)
      (by split_ifs <;> simp)) ?_) (abs_nonneg _)
    refine (Finset.abs_sum_le_sum_abs _ _).trans ((Finset.sum_le_sum fun k _ => ?_).trans
      (by rw [Fin.sum_const, nsmul_eq_mul]))
    rw [← mul_sub, sub_sub_sub_cancel_right, abs_mul]
    exact mul_le_mul (hW₂ l k) (hS k) (abs_nonneg _) hw₂
  rw [← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ((Finset.sum_le_sum fun l _ => hrow l).trans ?_)
  rw [← Finset.sum_mul]
  refine (mul_le_mul_of_nonneg_right hJ hM0).trans_eq ?_
  ring

end MlpSlot

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
  unfold dense Mat.unflatten at hz ⊢
  fun_prop (disch := assumption)

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
      w₂ * (a * ∑ idx, |e idx|) :=
  (dense_relu_drift W₂ b₂ hw₂ hW₂ _ _ k).trans
    (mul_le_mul_of_nonneg_left (dense_unflatten_drift_sum b₁ a₀ hx v e) hw₂)

/-- **Segment-Lipschitz gradient for the hidden-layer loss, explicit
    constant.** Under the margin `a·D < |z₁ⱼ|` (the step cannot flip a ReLU
    sign — the masks freeze along the whole segment) and the small-step
    condition `2·w₂·a·D < 1`, the gradient entries drift by at most
    `(2·d₃·w₂²·a²/(1−2·w₂·a·D))·(t·D)` along `[v, v+d]` — the exact shape
    `descent_segment` consumes. The hidden-layer peer of
    `linear_loss_grad_lipschitz`; `MlpSlot.loss_grad_lipschitz` at `σ = ρ = a`. -/
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
  obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
  exact MlpSlot.loss_grad_lipschitz (fun w => dense (Mat.unflatten w) b₁ a₀) W₂ b₂ label ha
    (dense_unflatten_drift b₁ a₀ ha hx) (dense_unflatten_drift_sum b₁ a₀ hx) hw₂ hW₂
    (fun l => if l = j then a₀ i else 0)
    (by rw [Finset.sum_eq_single j (fun l _ hl => by simp [hl]) (by simp)]; simpa using hx i)
    _ (fun _ => True)
    (fun v' _ hz => by
      rw [mlp_hidden_loss_gradAt b₁ W₂ b₂ a₀ label v' hz i j,
        Finset.sum_eq_single j (fun l _ hl => by simp [hl]) (by simp), ite_eq_left rfl])
    v d hd hmargin hsmall t ht trivial trivial

/-- The loss as a function of the flattened hidden-layer weights. -/
noncomputable def mlpHiddenLoss {d₁ d₂ d₃ : Nat} (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (a₀ : Vec d₁) (label : Fin d₃) : Vec (d₁ * d₂) → ℝ :=
  fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label

/-- **One inexact SGD step on the MLP's hidden weights decreases one example's
    cross-entropy loss** (example `(a₀, label)`, `W₁` moving, every other
    parameter fixed). `sgd_descends`' smoothness hypotheses are discharged
    for the loss-of-`W₁` map: differentiability along the segment and the
    segment-Lipschitz constant `C = 2·d₃·w₂²·a²/(1−2·w₂·a·D)` at step radius
    `D = lr·(‖∇L‖₁ + d₁d₂·η)` both come from the **margin hypothesis** — the
    step radius is small enough that no hidden ReLU can change sign.
    Remaining hypotheses: the oracle accuracy `η` (the float budgets), the
    margins, the small-step condition, and the two dominance conditions.
    Conclusion: the loss drops by ≥ `lr·‖∇L‖₂²/2`. The hidden-layer peer of
    `linear_sgd_descends`. -/
theorem mlp_hidden_sgd_descends {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (a₀ : Vec d₁)
    (label : Fin d₃) (gh : Vec (d₁ * d₂)) {lr η a w₂ : ℝ}
    (ha : 0 ≤ a) (hx : ∀ i, |a₀ i| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ j k, |W₂ j k| ≤ w₂)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx -
      gradAt (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
        (Mat.flatten W₁) idx| ≤ η)
    (hmargin : ∀ j, a * (stepRadius (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁) lr η) <
      |dense W₁ b₁ a₀ j|)
    (hsmall : 2 * (w₂ * (a * (stepRadius (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
      (Mat.flatten W₁) lr η))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt
        (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
        (Mat.flatten W₁) idx|) ≤
      lr * (∑ idx, gradAt
        (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
        (Mat.flatten W₁) idx ^ 2) / 4)
    (h2 : (2 * (d₃ : ℝ) * w₂ ^ 2 * a ^ 2 / (1 - 2 * (w₂ * (a * (stepRadius
      (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁) lr η))))) *
        (stepRadius (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁) lr η) ^ 2 ≤
      lr * (∑ idx, gradAt
        (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
        (Mat.flatten W₁) idx ^ 2) / 4) :
    (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁ - lr • gh) ≤
      (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁) -
        lr * (∑ idx, gradAt
          (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
          (Mat.flatten W₁) idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  unfold mlpHiddenLoss at *
  set f : Vec (d₁ * d₂) → ℝ :=
    fun w => crossEntropy d₃
      (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label
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
        ((d₁ * d₂ : ℕ) : ℝ) * η) :=
    sgd_step_l1_le _ gh hlr hgh
  have hmain := sgd_descends f (Mat.flatten W₁) gh hlr hη hC0 hgh
    (fun t ht => mlp_hidden_loss_differentiableAt b₁ W₂ b₂ a₀ label _
      (fun k => (margin_keeps_offkink b₁ a₀ ha hx (Mat.flatten W₁)
        (-(lr • gh)) hD hmargin' t ht.1 ht.2 k).1))
    (fun t ht idx => by
      have := mlp_hidden_loss_grad_lipschitz b₁ W₂ b₂ a₀ label ha hx hw₂
        hW₂ (Mat.flatten W₁) (-(lr • gh)) hD hmargin' hsmall t ht idx
      exact this)
    h1 h2
  exact hmain

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
    fun_prop (disch := assumption)
  rw [show (fun y : Vec d₁ => fun _ : Fin 1 => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ y)))) label)
        = (fun u : Vec d₁ => fun _ : Fin 1 => crossEntropy d₃
            (dense W₂ b₂ (relu d₂ (dense W₁ b₁ u))) label) ∘ (relu d₁)
        from rfl,
      pdiv_comp _ _ _ (relu_differentiableAt_of_smooth d₁ z hz0) hg1]
  simp_rw [pdiv_relu d₁ z hz0 j, ite_mul, zero_mul]
  rw [Finset.sum_ite_eq]
  simp only [Finset.mem_univ, ite_true]
  congr 1
  -- second hop: peel the middle dense, then reuse the one-relu head
  have hH : DifferentiableAt ℝ
      (fun u : Vec d₂ => fun _ : Fin 1 =>
        crossEntropy d₃ (dense W₂ b₂ (relu d₂ u)) label)
      (dense W₁ b₁ (relu d₁ z)) := by
    fun_prop (disch := assumption)
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
  unfold dense Mat.unflatten at hz0 hz1 ⊢
  fun_prop (disch := assumption)

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
        0 < dense W₁ b₁ (relu d₁ (dense (Mat.unflatten v) b₀ x)) l) :=
  margin_keeps_offkink_of_drift (ρ := w₁ * a)
    (fun w => dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))) (mul_nonneg hw₁ ha)
    (fun v e l => (mlp_hidden_logit_drift b₀ W₁ b₁ x hx hw₁ hW₁ v e l).trans_eq (by ring))
    v e he (fun l => lt_of_eq_of_lt (by ring) (hmargin1 l)) t ht0 ht1 l

/-- **Segment-Lipschitz gradient for the input-layer loss, explicit
    constant.** Under both margins (neither ReLU layer's sign pattern can
    change along the step) and the small-step condition, the gradient
    entries drift by at most
    `(2·d₃·d₂²·w₁²·w₂²·a²/(1−2·w₂·d₂·w₁·a·D))·(t·D)`. The input-layer peer
    of `mlp_hidden_loss_grad_lipschitz`; the extra `d₂·w₁` is the middle
    layer's `ℓ1→ℓ1` operator factor. `MlpSlot.loss_grad_lipschitz` at the middle
    pre-activation, `σ = w₁·a`, `ρ = d₂·w₁·a`. -/
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
  have hD0 : 0 ≤ D := le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
  -- relu₀'s frozen mask, `xᵢ` and `W₁`'s row `j`: one fixed row at the middle pre-activation
  have hJ : ∑ l, |x i * ((if dense (Mat.unflatten v) b₀ x j > 0 then (1:ℝ) else 0) *
      W₁ j l)| ≤ (d₂ : ℝ) * (w₁ * a) := by
    refine (Finset.sum_le_sum fun l _ => ?_).trans_eq
      (by rw [Fin.sum_const, nsmul_eq_mul])
    rw [abs_mul, abs_mul, mul_comm w₁ a]
    exact mul_le_mul (hx i) ((mul_le_of_le_one_left (abs_nonneg _)
      (by split_ifs <;> simp)).trans (hW₁ j l)) (by positivity) ha
  refine (MlpSlot.loss_grad_lipschitz (σ := w₁ * a) (ρ := (d₂ : ℝ) * (w₁ * a))
    (fun w => dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))) W₂ b₂ label
    (mul_nonneg hw₁ ha)
    (fun v e l => (mlp_hidden_logit_drift b₀ W₁ b₁ x hx hw₁ hW₁ v e l).trans_eq (by ring))
    (fun v e => (Finset.sum_le_sum fun l _ =>
      mlp_hidden_logit_drift b₀ W₁ b₁ x hx hw₁ hW₁ v e l).trans_eq (by
        rw [Fin.sum_const, nsmul_eq_mul]; ring))
    hw₂ hW₂ _ hJ _
    (fun v' => ∀ k, dense (Mat.unflatten v') b₀ x k ≠ 0 ∧
      (0 < dense (Mat.unflatten v') b₀ x k ↔ 0 < dense (Mat.unflatten v) b₀ x k))
    (fun v' hQ hz1 => ?_) v d hd (fun l => lt_of_eq_of_lt (by ring) (hmargin1 l))
    (lt_of_eq_of_lt (by ring) hsmall) t ht
    (fun k => ⟨abs_pos.mp ((mul_nonneg ha hD0).trans_lt (hmargin0 k)), Iff.rfl⟩)
    (fun k => margin_keeps_offkink b₀ x ha hx v d hd hmargin0 t ht.1 ht.2 k)).trans_eq
    (by ring)
  -- at a frozen relu₀ point the input gradient is that row contracted with the head
  rw [mlp_input_loss_gradAt b₀ W₁ b₁ W₂ b₂ x label v' (fun k => (hQ k).1) hz1 i j]
  simp only [gt_iff_lt, (hQ j).2]
  rw [← mul_assoc, Finset.mul_sum]
  exact Finset.sum_congr rfl fun l _ => by rw [mul_assoc, mul_assoc, mul_assoc]

/-- The loss as a function of the flattened input-layer weights. -/
noncomputable def mlpInputLoss {d₀ d₁ d₂ d₃ : Nat} (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) : Vec (d₀ * d₁) → ℝ :=
  fun w => crossEntropy d₃
    (dense W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label

/-- **One inexact SGD step on the MLP's input weights decreases one example's
    cross-entropy loss** (example `(x, label)`, `W₀` moving, every other
    parameter fixed). The deepest descent capstone: both ReLU
    layers' margins at the step radius `D = lr·(‖∇L‖₁ + d₀d₁·η)` freeze the
    masks, the segment-Lipschitz constant
    `C = 2·d₃·d₂²·w₁²·w₂²·a²/(1−2·w₂·d₂·w₁·a·D)` is proven, and the loss
    drops by ≥ `lr·‖∇L‖₂²/2`. The oracle accuracy, the margins, the small-step
    and the two dominance conditions remain hypotheses. The input-layer peer of
    `linear_sgd_descends`; with this each MLP weight layer has a single-layer,
    single-example descent statement. -/
theorem mlp_input_sgd_descends {d₀ d₁ d₂ d₃ : Nat} (W₀ : Mat d₀ d₁)
    (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃)
    (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) (gh : Vec (d₀ * d₁))
    {lr η a w₁ w₂ : ℝ}
    (ha : 0 ≤ a) (hx : ∀ i, |x i| ≤ a)
    (hw₁ : 0 ≤ w₁) (hW₁ : ∀ j l, |W₁ j l| ≤ w₁)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ l k, |W₂ l k| ≤ w₂)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx -
      gradAt (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
        (Mat.flatten W₀) idx| ≤ η)
    (hmargin0 : ∀ j, a * (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀) lr η) <
      |dense W₀ b₀ x j|)
    (hmargin1 : ∀ l, w₁ * (a * (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
      (Mat.flatten W₀) lr η)) <
      |dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) l|)
    (hsmall : 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
      (Mat.flatten W₀) lr η))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt
        (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
        (Mat.flatten W₀) idx|) ≤
      lr * (∑ idx, gradAt
        (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
        (Mat.flatten W₀) idx ^ 2) / 4)
    (h2 : (2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 * a ^ 2 /
        (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
          (Mat.flatten W₀) lr η))))))) *
        (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀) lr η) ^ 2 ≤
      lr * (∑ idx, gradAt
        (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
        (Mat.flatten W₀) idx ^ 2) / 4) :
    (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀ - lr • gh) ≤
      (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀) -
        lr * (∑ idx, gradAt
          (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
          (Mat.flatten W₀) idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  unfold mlpInputLoss at *
  set f : Vec (d₀ * d₁) → ℝ :=
    fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
      (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label
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
        ((d₀ * d₁ : ℕ) : ℝ) * η) :=
    sgd_step_l1_le _ gh hlr hgh
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
      exact this)
    h1 h2
  exact hmain

-- ════════════════════════════════════════════════════════════════
-- § Output layer η-composition: feed the FloatBridge budget into the
--   output-layer descent slot, so "one binary32 output-layer SGD step
--   decreases the loss" holds with NO abstract gradient-accuracy parameter.
-- ════════════════════════════════════════════════════════════════

/-- The loss as a function of the flattened output-layer weights. -/
noncomputable def mlpOutputLoss {d₀ d₁ d₂ d₃ : Nat} (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) : Vec (d₂ * d₃) → ℝ :=
  fun w => crossEntropy d₃
    (dense (Mat.unflatten w) b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label

/-- **One SGD step with the FloatModel binary32 output-layer gradient decreases one
    example's cross-entropy loss; the gradient's accuracy is proven, not assumed.**
    The output-layer rung of the η-composition. Since the
    top dense layer sits directly below the softmax-CE loss with no ReLU between,
    the loss-of-`W₂` map *is* the linear net's loss at the hidden activation
    `a₁ = relu(dense W₁ b₁ (relu(dense W₀ b₀ x)))` — so this is
    `linear_float_sgd_descends` instantiated there, with the same scope (one
    example, `W₂` moving, update in ℝ). The gradient is the FloatModel
    binary32 output-layer gradient `M.linearFloatGrad W₂ b₂ a₁` and its accuracy
    `η = mulErr u a 1 0 (cotErr …)` is *proven* (by `linear_grad_close`, inside
    the linear theorem), not assumed. No margin needed — the output layer never
    crosses a kink.

    The hidden and input rungs are `mlp_hidden_float_sgd_descends` and
    `mlp_input_float_sgd_descends`. -/
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
    (hsmall : 2 * (a * (stepRadius (mlpOutputLoss W₀ b₀ W₁ b₁ b₂ x label) (Mat.flatten W₂) lr (FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ d₃)))) < 1)
    (h1 : lr * (FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ d₃)) *
        (∑ idx, |gradAt
          (mlpOutputLoss W₀ b₀ W₁ b₁ b₂ x label)
          (Mat.flatten W₂) idx|) ≤
      lr * (∑ idx, gradAt
        (mlpOutputLoss W₀ b₀ W₁ b₁ b₂ x label)
        (Mat.flatten W₂) idx ^ 2) / 4)
    (h2 : (2 * a ^ 2 / (1 - 2 * (a * (stepRadius (mlpOutputLoss W₀ b₀ W₁ b₁ b₂ x label) (Mat.flatten W₂) lr (FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ d₃)))))) *
        (stepRadius (mlpOutputLoss W₀ b₀ W₁ b₁ b₂ x label) (Mat.flatten W₂) lr (FloatModel.mulErr M.u a 1 0 (FloatModel.cotErr M.u eexp δ d₃))) ^ 2 ≤
      lr * (∑ idx, gradAt
        (mlpOutputLoss W₀ b₀ W₁ b₁ b₂ x label)
        (Mat.flatten W₂) idx ^ 2) / 4) :
    crossEntropy d₃ (dense (Mat.unflatten (Mat.flatten W₂ -
        lr • M.linearFloatGrad W₂ b₂
          (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) fexp label)) b₂
        (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label ≤
      (mlpOutputLoss W₀ b₀ W₁ b₁ b₂ x label) (Mat.flatten W₂) -
        lr * (∑ idx, gradAt
          (mlpOutputLoss W₀ b₀ W₁ b₁ b₂ x label)
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
    exact* (the descent moves only `W₁`), the FloatModel transcription computes the
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
      oneHot d₃ label k| ≤ 1 :=
    fun k => abs_softmax_sub_oneHot_le_one _ label k
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

-- ════════════════════════════════════════════════════════════════
-- § Hidden layer η-composition: feed the FloatBridge `W₁` grad-close
--   budget into the hidden-layer descent slot, so "one binary32
--   hidden-layer SGD step decreases the loss" holds with NO abstract
--   gradient-accuracy parameter.
-- ════════════════════════════════════════════════════════════════

/-- **The binary32 hidden-layer (`W₁`) gradient of the MLP loss** — the
    FloatModel transcription of the per-example gradient (with the layer-1 input
    activation `a₀` frozen exact): `fl(a₀ᵢ · c̃₁ⱼ)` where the float layer-1 cotangent
    `c̃₁ = mask(z̃₁, W₂ᵀ·c̃₂)` reads the float pre-activation
    `z̃₁ = M.dense W₁ b₁ a₀` and the float softmax−onehot head `c̃₂` at the
    float logits. Flattened to the `Vec (d₁*d₂)` parameter layout that
    `gradAt`/`mlp_hidden_sgd_descends` use. The hidden-layer peer of
    `linearFloatGrad`. -/
noncomputable def FloatModel.mlpHiddenFloatGrad (M : FloatModel)
    {d₁ d₂ d₃ : Nat} (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃)
    (b₂ : Vec d₃) (a₀ : Vec d₁) (fexp : ℝ → ℝ) (label : Fin d₃) :
    Vec (d₁ * d₂) :=
  Mat.flatten fun i j =>
    M.mul (a₀ i)
      (FloatModel.reluMask (M.dense W₁ b₁ a₀)
        (M.dense (fun j' i' => W₂ i' j') (fun _ => 0)
          (M.softmaxCECotF fexp
            (M.dense W₂ b₂ (relu d₂ (M.dense W₁ b₁ a₀))) label)) j)

@[simp] theorem mlpHiddenFloatGrad_apply (M : FloatModel) {d₁ d₂ d₃ : Nat}
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (a₀ : Vec d₁) (fexp : ℝ → ℝ) (label : Fin d₃) (i : Fin d₁) (j : Fin d₂) :
    M.mlpHiddenFloatGrad W₁ b₁ W₂ b₂ a₀ fexp label (finProdFinEquiv (i, j)) =
      M.mul (a₀ i)
        (FloatModel.reluMask (M.dense W₁ b₁ a₀)
          (M.dense (fun j' i' => W₂ i' j') (fun _ => 0)
            (M.softmaxCECotF fexp
              (M.dense W₂ b₂ (relu d₂ (M.dense W₁ b₁ a₀))) label)) j) := by
  simp [FloatModel.mlpHiddenFloatGrad, Mat.flatten, Equiv.symm_apply_apply]

/-- **The certified hidden-layer loss gradient, in the `reluMask` form that
    `mlp_w1_grad_close` bounds against.** At an off-kink parameter point
    (`hz`), `mlp_hidden_loss_gradAt`'s closed form
    `a₀ᵢ·relu'(z₁ⱼ)·∑ₖ W₂ⱼₖ·(softmax−onehot)ₖ` equals the masked-`W₂ᵀ`-
    contraction form `a₀ᵢ · reluMask(z₁, dense (fun j' i' => W₂ i' j') 0 (softmax−onehot))ⱼ`.
    The bridge that lets the float grad-close (stated with `reluMask`) discharge
    `mlp_hidden_sgd_descends`' abstract `η` (stated with `gradAt`). -/
theorem mlp_hidden_loss_gradAt_reluMask {d₁ d₂ d₃ : Nat}
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (a₀ : Vec d₁) (label : Fin d₃)
    (hz : ∀ k, dense W₁ b₁ a₀ k ≠ 0) (i : Fin d₁) (j : Fin d₂) :
    gradAt (fun w => crossEntropy d₃
        (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        (Mat.flatten W₁) (finProdFinEquiv (i, j))
      = a₀ i * FloatModel.reluMask (dense W₁ b₁ a₀)
          (dense (fun j' i' => W₂ i' j') (fun _ => 0)
            (fun k => softmax d₃
              (dense W₂ b₂ (relu d₂ (dense W₁ b₁ a₀))) k -
              oneHot d₃ label k)) j := by
  rw [mlp_hidden_loss_gradAt b₁ W₂ b₂ a₀ label (Mat.flatten W₁)
        (fun k => by rw [Mat.unflatten_flatten]; exact hz k) i j,
      Mat.unflatten_flatten]
  congr 1
  rw [FloatModel.reluMask]
  split_ifs <;> simp [dense, mul_comm]

/-- **One SGD step with the FloatModel binary32 hidden-layer gradient decreases
    one example's cross-entropy loss; the gradient's accuracy is proven, not
    assumed.** The hidden-layer rung of the η-composition. The gradient is the
    FloatModel binary32 `W₁` gradient
    `M.mlpHiddenFloatGrad W₁ b₁ W₂ b₂ a₀ fexp label`, and its accuracy
    `η = mulErr u a (layerAct …) 0 (layerBudget … (cotErr …))` is *proven*
    by `mlp_w1_grad_close` (via the `reluMask`↔`gradAt` bridge
    `mlp_hidden_loss_gradAt_reluMask`), not assumed.

    Two margins are carried as hypotheses: the **rounding** margin
    `hmargin_round` (`layerBudget < |z₁|`, forward rounding must not flip the
    layer-1 ReLU — the grad-close precondition) and the **step** margin
    `hmargin_step` (`a·D < |z₁|`, the parameter step must not flip it along
    the segment — the smoothness precondition). They are the same shape
    ("nothing flips the layer-1 ReLU") and are not collapsed into one here.
    This is the hidden-layer peer of
    `linear_float_sgd_descends` / `mlp_output_float_sgd_descends`.

    Scope: one example `(a₀, label)`, the layer's weights only (other parameters
    fixed), and the update taken in ℝ — only the gradient is float-modelled. -/
theorem mlp_hidden_float_sgd_descends {d₁ d₂ d₃ : Nat} (M : FloatModel)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (a₀ : Vec d₁) (label : Fin d₃) (fexp : ℝ → ℝ)
    {lr a w₁ β₁ w₂ β₂ eexp : ℝ}
    (ha : 0 ≤ a) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hlr : 0 ≤ lr) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp d₃ < 1)
    (hx : ∀ i, |a₀ i| ≤ a)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w₂) (hb₂ : ∀ j, |b₂ j| ≤ β₂)
    (hmargin_round : ∀ j', FloatModel.layerBudget M.u d₁ w₁ β₁ a 0 <
      |dense W₁ b₁ a₀ j'|)
    (hmargin_step : ∀ j, a * (stepRadius (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁) lr (FloatModel.mulErr M.u a (FloatModel.layerAct d₃ w₂ 0 1) 0 (FloatModel.layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ a) (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃)))) <
      |dense W₁ b₁ a₀ j|)
    (hsmall : 2 * (w₂ * (a * (stepRadius (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁) lr (FloatModel.mulErr M.u a (FloatModel.layerAct d₃ w₂ 0 1) 0 (FloatModel.layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ a) (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃)))))) < 1)
    (h1 : lr * (FloatModel.mulErr M.u a (FloatModel.layerAct d₃ w₂ 0 1) 0
          (FloatModel.layerBudget M.u d₃ w₂ 0 1
            (FloatModel.cotErr M.u eexp
              (FloatModel.layerBudget M.u d₂ w₂ β₂
                (FloatModel.layerAct d₁ w₁ β₁ a)
                (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃))) *
        (∑ idx, |gradAt
          (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
          (Mat.flatten W₁) idx|) ≤
      lr * (∑ idx, gradAt
        (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
        (Mat.flatten W₁) idx ^ 2) / 4)
    (h2 : (2 * (d₃ : ℝ) * w₂ ^ 2 * a ^ 2 / (1 - 2 * (w₂ * (a * (lr *
          ((∑ idx, |gradAt (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
            (Mat.flatten W₁) idx|) + ((d₁ * d₂ : ℕ) : ℝ) *
              FloatModel.mulErr M.u a (FloatModel.layerAct d₃ w₂ 0 1) 0
                (FloatModel.layerBudget M.u d₃ w₂ 0 1
                  (FloatModel.cotErr M.u eexp
                    (FloatModel.layerBudget M.u d₂ w₂ β₂
                      (FloatModel.layerAct d₁ w₁ β₁ a)
                      (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃)))))))) *
        (stepRadius (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁) lr (FloatModel.mulErr M.u a (FloatModel.layerAct d₃ w₂ 0 1) 0 (FloatModel.layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ a) (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃)))) ^ 2 ≤
      lr * (∑ idx, gradAt
        (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
        (Mat.flatten W₁) idx ^ 2) / 4) :
    (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁ -
          lr • M.mlpHiddenFloatGrad W₁ b₁ W₂ b₂ a₀ fexp label) ≤
      (mlpHiddenLoss b₁ W₂ b₂ a₀ label) (Mat.flatten W₁) -
        lr * (∑ idx, gradAt
          (mlpHiddenLoss b₁ W₂ b₂ a₀ label)
          (Mat.flatten W₁) idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  unfold mlpHiddenLoss at *
  have hu := M.u_nonneg
  -- the proven accuracy budget η of `mlp_w1_grad_close`
  set η : ℝ := FloatModel.mulErr M.u a (FloatModel.layerAct d₃ w₂ 0 1) 0
    (FloatModel.layerBudget M.u d₃ w₂ 0 1
      (FloatModel.cotErr M.u eexp
        (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ a)
          (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃)) with hη
  -- the budget is nonnegative (it bounds an absolute value)
  have hB1 : 0 ≤ FloatModel.layerBudget M.u d₁ w₁ β₁ a 0 :=
    FloatModel.layerBudget_nonneg hu hw₁ hβ₁ ha le_rfl
  have hcotB : 0 ≤ FloatModel.cotErr M.u eexp
      (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ a)
        (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃ :=
    M.cotErr_nonneg heexp0
      (FloatModel.layerBudget_nonneg hu hw₂ hβ₂
        (FloatModel.layerAct_nonneg hw₁ hβ₁ ha) hB1) hρ1
  have hcotB2 : 0 ≤ FloatModel.layerBudget M.u d₃ w₂ 0 1
      (FloatModel.cotErr M.u eexp
        (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ a)
          (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃) :=
    FloatModel.layerBudget_nonneg hu hw₂ le_rfl zero_le_one hcotB
  have hAct : 0 ≤ FloatModel.layerAct d₃ w₂ 0 1 :=
    FloatModel.layerAct_nonneg hw₂ le_rfl zero_le_one
  have hη0 : 0 ≤ η := by
    rw [hη]
    have e1 : (0:ℝ) ≤ M.u * ((a + 0) * (FloatModel.layerAct d₃ w₂ 0 1 +
        FloatModel.layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp
          (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ a)
            (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃))) :=
      mul_nonneg hu (mul_nonneg (by linarith) (by linarith))
    have e2 : (0:ℝ) ≤ a * FloatModel.layerBudget M.u d₃ w₂ 0 1
        (FloatModel.cotErr M.u eexp (FloatModel.layerBudget M.u d₂ w₂ β₂
          (FloatModel.layerAct d₁ w₁ β₁ a)
          (FloatModel.layerBudget M.u d₁ w₁ β₁ a 0)) d₃) := mul_nonneg ha hcotB2
    simp only [FloatModel.mulErr]
    nlinarith [e1, e2]
  -- the layer-1 pre-activations are off the kink (from the rounding margin)
  have hz : ∀ k, dense W₁ b₁ a₀ k ≠ 0 := fun k => abs_pos.mp (hB1.trans_lt (hmargin_round k))
  -- discharge `mlp_hidden_sgd_descends`' abstract η by the proven grad-close
  have hgh : ∀ idx, |M.mlpHiddenFloatGrad W₁ b₁ W₂ b₂ a₀ fexp label idx -
      gradAt (fun w => crossEntropy d₃
          (dense W₂ b₂ (relu d₂ (dense (Mat.unflatten w) b₁ a₀))) label)
        (Mat.flatten W₁) idx| ≤ η := by
    intro idx
    obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
    rw [mlpHiddenFloatGrad_apply,
      mlp_hidden_loss_gradAt_reluMask W₁ b₁ W₂ b₂ a₀ label hz i j]
    exact mlp_w1_grad_close M W₁ b₁ W₂ b₂ a₀ label fexp ha hw₁ hβ₁ hw₂ hβ₂
      heexp0 heexp1 hfexp hρ1 hx hW₁ hb₁ hW₂ hb₂ hmargin_round i j
  exact mlp_hidden_sgd_descends W₁ b₁ W₂ b₂ a₀ label
    (M.mlpHiddenFloatGrad W₁ b₁ W₂ b₂ a₀ fexp label)
    ha hx hw₂ hW₂ hlr hη0 hgh hmargin_step hsmall h1 h2

-- ════════════════════════════════════════════════════════════════
-- § Input layer W₀: the float-backward grad-close + η-composition
--   (one mask deeper than the hidden rung — two ReLU layers, two
--   masked Wᵀ contractions, two rounding margins).
-- ════════════════════════════════════════════════════════════════

/-- **A masked `Wᵀ` contraction in if-then-else form equals the `reluMask`
    form.** `(relu'(zₗ))·∑ₖ Wₗₖ·cₖ = reluMask z (Wᵀ·c) l` — the per-step
    identity behind the `gradAt`↔`reluMask` bridges (`mlp_hidden_/`
    `mlp_input_loss_gradAt_reluMask`): one ReLU-sign case split + `mul_comm`
    (the transpose `dense (fun j i' => W i' j) 0 c` reads `∑ₖ cₖ·Wₗₖ`). -/
theorem reluMask_dense_transpose_eq {p n : Nat} (z : Vec p) (W : Mat p n)
    (c : Vec n) (l : Fin p) :
    (if z l > 0 then (1:ℝ) else 0) * ∑ k, W l k * c k =
      FloatModel.reluMask z (dense (fun j i' => W i' j) (fun _ => 0) c) l := by
  rw [FloatModel.reluMask]
  split_ifs <;> simp [dense, mul_comm]

/-- **The binary32 input-layer (`W₀`) gradient of the MLP loss** — the FloatModel
    transcription of the per-example gradient (`x` the exact input): `fl(xᵢ · c̃₀ⱼ)`
    where the float layer-0 cotangent `c̃₀ = mask(z̃₀, W₁ᵀ·c̃₁)` reads the float
    layer-1 cotangent `c̃₁ = mask(z̃₁, W₂ᵀ·c̃₂)` and the float softmax−onehot head
    `c̃₂`, all at the float pre-activations. Flattened to the `Vec (d₀*d₁)`
    parameter layout. The two-mask peer of `mlpHiddenFloatGrad`. -/
noncomputable def FloatModel.mlpInputFloatGrad (M : FloatModel)
    {d₀ d₁ d₂ d₃ : Nat} (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂)
    (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (fexp : ℝ → ℝ)
    (label : Fin d₃) : Vec (d₀ * d₁) :=
  Mat.flatten fun i j =>
    M.mul (x i)
      (FloatModel.reluMask (M.dense W₀ b₀ x)
        (M.dense (fun j' i' => W₁ i' j') (fun _ => 0)
          (FloatModel.reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
            (M.dense (fun j' i' => W₂ i' j') (fun _ => 0)
              (M.softmaxCECotF fexp
                (M.dense W₂ b₂ (relu d₂
                  (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))))) label)))) j)

@[simp] theorem mlpInputFloatGrad_apply (M : FloatModel) {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (fexp : ℝ → ℝ) (label : Fin d₃)
    (i : Fin d₀) (j : Fin d₁) :
    M.mlpInputFloatGrad W₀ b₀ W₁ b₁ W₂ b₂ x fexp label
        (finProdFinEquiv (i, j)) =
      M.mul (x i)
        (FloatModel.reluMask (M.dense W₀ b₀ x)
          (M.dense (fun j' i' => W₁ i' j') (fun _ => 0)
            (FloatModel.reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
              (M.dense (fun j' i' => W₂ i' j') (fun _ => 0)
                (M.softmaxCECotF fexp
                  (M.dense W₂ b₂ (relu d₂
                    (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))))) label)))) j) := by
  simp [FloatModel.mlpInputFloatGrad, Mat.flatten, Equiv.symm_apply_apply]

/-- **The certified input-layer loss gradient, in the nested `reluMask` form
    that `mlp_w0_grad_close` bounds against.** At an off-kink point (both
    `hz0`, `hz1`), `mlp_input_loss_gradAt`'s two-mask if-then-else closed form
    equals `xᵢ · reluMask(z₀, W₁ᵀ·reluMask(z₁, W₂ᵀ·(softmax−onehot)))ⱼ`. Two
    applications of `reluMask_dense_transpose_eq` (inner W₂ᵀ then outer W₁ᵀ),
    fired by `simp_rw`. -/
theorem mlp_input_loss_gradAt_reluMask {d₀ d₁ d₂ d₃ : Nat} (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂) (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (W₀ : Mat d₀ d₁) (x : Vec d₀) (label : Fin d₃)
    (hz0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (hz1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0)
    (i : Fin d₀) (j : Fin d₁) :
    gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
        (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        (Mat.flatten W₀) (finProdFinEquiv (i, j))
      = x i * FloatModel.reluMask (dense W₀ b₀ x)
          (dense (fun j' i' => W₁ i' j') (fun _ => 0)
            (FloatModel.reluMask (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))
              (dense (fun j' i' => W₂ i' j') (fun _ => 0)
                (fun k => softmax d₃ (dense W₂ b₂ (relu d₂
                  (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k -
                  oneHot d₃ label k)))) j := by
  rw [mlp_input_loss_gradAt b₀ W₁ b₁ W₂ b₂ x label (Mat.flatten W₀)
        (fun k => by rw [Mat.unflatten_flatten]; exact hz0 k)
        (fun k => by rw [Mat.unflatten_flatten]; exact hz1 k) i j,
      Mat.unflatten_flatten]
  simp_rw [reluMask_dense_transpose_eq]

open FloatModel in
/-- **The binary32 input-layer (`W₀`) gradient is within an explicit budget of
    the certified one**, per entry — the float-backward grad-close for the
    deepest rung. With `x` exact, the FloatModel transcription computes the `W₀`
    gradient `fl(xᵢ·c̃₀ⱼ)`, `c̃₀ = mask(z̃₀, W₁ᵀ·mask(z̃₁, W₂ᵀ·c̃₂))` from the float
    softmax−onehot head `c̃₂` back through *two* ReLU masks. This is within
    `mulErr … 0 (layerBudget … (layerBudget … (cotErr …)))` of the certified
    `xᵢ·mask(z₀, W₁ᵀ·mask(z₁, W₂ᵀ·(softmax−onehot)))ⱼ` (= `mlp_input_loss_gradAt`,
    via `mlp_input_loss_gradAt_reluMask`). Built like `mlp_w1_grad_close` with
    one more `cot_step_close`: head (`softmax_ce_cot_close`), masked `W₂ᵀ`
    contraction (`cot_step_close`, **under the layer-1 margin** `E₁ < |z₁|`),
    masked `W₁ᵀ` contraction (`cot_step_close`, **under the layer-0 margin**
    `E₀ < |z₀|`), final exact-`x` multiply (`mul_close`, `ea = 0`). -/
theorem mlp_w0_grad_close {d₀ d₁ d₂ d₃ : Nat} (M : FloatModel)
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) (fexp : ℝ → ℝ)
    {a w₀ β₀ w₁ β₁ w₂ β₂ eexp : ℝ}
    (ha : 0 ≤ a) (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁)
    (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp d₃ < 1)
    (hx : ∀ i, |x i| ≤ a)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w₂) (hb₂ : ∀ j, |b₂ j| ≤ β₂)
    (hmargin0 : ∀ j', layerBudget M.u d₀ w₀ β₀ a 0 < |Proofs.dense W₀ b₀ x j'|)
    (hmargin1 : ∀ l', layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
        (layerBudget M.u d₀ w₀ β₀ a 0) <
      |Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) l'|)
    (i : Fin d₀) (j : Fin d₁) :
    |M.mul (x i)
        (reluMask (M.dense W₀ b₀ x)
          (M.dense (fun j' i' => W₁ i' j') (fun _ => 0)
            (reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
              (M.dense (fun j' i' => W₂ i' j') (fun _ => 0)
                (M.softmaxCECotF fexp
                  (M.dense W₂ b₂ (relu d₂
                    (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))))) label)))) j) -
      x i * reluMask (Proofs.dense W₀ b₀ x)
        (Proofs.dense (fun j' i' => W₁ i' j') (fun _ => 0)
          (reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
            (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0)
              (fun k => softmax d₃ (Proofs.dense W₂ b₂ (relu d₂
                (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))))) k -
                oneHot d₃ label k)))) j| ≤
    mulErr M.u a (layerAct d₂ w₁ 0 (layerAct d₃ w₂ 0 1)) 0
      (layerBudget M.u d₂ w₁ 0 (layerAct d₃ w₂ 0 1)
        (layerBudget M.u d₃ w₂ 0 1
          (FloatModel.cotErr M.u eexp
            (layerBudget M.u d₂ w₂ β₂
              (layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a))
              (layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)
                (layerBudget M.u d₀ w₀ β₀ a 0))) d₃))) := by
  -- layer-0 forward (x exact ⇒ inherited error 0)
  set E₀ := layerBudget M.u d₀ w₀ β₀ a 0 with hE₀
  have hE₀0 : 0 ≤ E₀ := layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha le_rfl
  have hA₀0 : 0 ≤ layerAct d₀ w₀ β₀ a := layerAct_nonneg hw₀ hβ₀ ha
  have l0 : ∀ k, |M.dense W₀ b₀ x k - Proofs.dense W₀ b₀ x k| ≤ E₀ :=
    fun k => (M.dense_close_fresh W₀ b₀ x k).trans
      (M.denseErr_le_uniform hw₀ le_rfl hW₀ hb₀ hx k)
  have r0 : ∀ k, |relu d₁ (M.dense W₀ b₀ x) k -
      relu d₁ (Proofs.dense W₀ b₀ x) k| ≤ E₀ := fun k => relu_close _ _ _ l0 k
  have ha₀ : ∀ k, |relu d₁ (Proofs.dense W₀ b₀ x) k| ≤ layerAct d₀ w₀ β₀ a :=
    fun k => (relu_abs_le _ k).trans (dense_abs_le ha hW₀ hb₀ hx k)
  -- layer-1 forward, inherited E₀
  set E₁ := layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) E₀ with hE₁
  have hA₁0 : 0 ≤ layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) :=
    layerAct_nonneg hw₁ hβ₁ hA₀0
  have hE₁0 : 0 ≤ E₁ := layerBudget_nonneg M.u_nonneg hw₁ hβ₁ hA₀0 hE₀0
  have l1 : ∀ k, |M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)) k -
      Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) k| ≤ E₁ := fun k =>
    (M.dense_close W₁ b₁ _ _ E₀ hE₀0 r0 k).trans
      (M.denseErr_le_uniform hw₁ hE₀0 hW₁ hb₁ ha₀ k)
  have r1 : ∀ k, |relu d₂ (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))) k -
      relu d₂ (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))) k| ≤ E₁ :=
    fun k => relu_close _ _ _ l1 k
  have ha₁ : ∀ k, |relu d₂ (Proofs.dense W₁ b₁
      (relu d₁ (Proofs.dense W₀ b₀ x))) k| ≤
      layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) :=
    fun k => (relu_abs_le _ k).trans (dense_abs_le hA₀0 hW₁ hb₁ ha₀ k)
  -- layer-2 forward (logits), inherited E₁
  set δ := layerBudget M.u d₂ w₂ β₂ (layerAct d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a)) E₁
    with hδdef
  have hδ0 : 0 ≤ δ := layerBudget_nonneg M.u_nonneg hw₂ hβ₂ hA₁0 hE₁0
  have l2 : ∀ k, |M.dense W₂ b₂ (relu d₂
      (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))) k -
      Proofs.dense W₂ b₂ (relu d₂
        (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))) k| ≤ δ := fun k =>
    (M.dense_close W₂ b₂ _ _ E₁ hE₁0 r1 k).trans
      (M.denseErr_le_uniform hw₂ hE₁0 hW₂ hb₂ ha₁ k)
  -- head: float softmax−onehot within `cotErr`
  have hcot2 : ∀ k, |M.softmaxCECotF fexp
      (M.dense W₂ b₂ (relu d₂ (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))))) label k -
      (softmax d₃ (Proofs.dense W₂ b₂ (relu d₂
        (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))))) k -
        oneHot d₃ label k)| ≤ FloatModel.cotErr M.u eexp δ d₃ := fun k =>
    M.softmax_ce_cot_close fexp _ _ label heexp0 heexp1 hfexp hρ1 l2 k
  -- real head cotangent `softmax − onehot ∈ [−1, 1]`
  have hC2 : ∀ k, |softmax d₃ (Proofs.dense W₂ b₂ (relu d₂
      (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))))) k -
      oneHot d₃ label k| ≤ 1 :=
    fun k => abs_softmax_sub_oneHot_le_one _ label k
  -- first masked W₂ᵀ contraction: layer-1 cotangent (under the layer-1 margin)
  have hcot1 : ∀ l, |reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
        (M.dense (fun j' i' => W₂ i' j') (fun _ => 0)
          (M.softmaxCECotF fexp (M.dense W₂ b₂ (relu d₂
            (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))))) label)) l -
      reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
        (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0)
          (fun k => softmax d₃ (Proofs.dense W₂ b₂ (relu d₂
            (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))))) k -
            oneHot d₃ label k)) l| ≤
      layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp δ d₃) := fun l =>
    M.cot_step_close W₂ (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
      (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
      (M.softmaxCECotF fexp (M.dense W₂ b₂ (relu d₂
        (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))))) label)
      (fun k => softmax d₃ (Proofs.dense W₂ b₂ (relu d₂
        (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))))) k -
        oneHot d₃ label k)
      hw₂ (by norm_num) (M.cotErr_nonneg heexp0 hδ0 hρ1) hW₂ hC2 hcot2 l1
      hmargin1 l
  -- real layer-1 cotangent magnitude
  have hC1 : ∀ l, |reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
      (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0)
        (fun k => softmax d₃ (Proofs.dense W₂ b₂ (relu d₂
          (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))))) k -
          oneHot d₃ label k)) l| ≤ layerAct d₃ w₂ 0 1 := fun l =>
    (reluMask_abs_le _ _ l).trans
      (dense_abs_le (by norm_num) (fun j' i' => hW₂ i' j') (fun _ => by simp) hC2 l)
  -- second masked W₁ᵀ contraction: layer-0 cotangent (under the layer-0 margin)
  have hcot0 := M.cot_step_close W₁ (M.dense W₀ b₀ x) (Proofs.dense W₀ b₀ x)
    (reluMask (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)))
      (M.dense (fun j' i' => W₂ i' j') (fun _ => 0)
        (M.softmaxCECotF fexp (M.dense W₂ b₂ (relu d₂
          (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))))) label)))
    (reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
      (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0)
        (fun k => softmax d₃ (Proofs.dense W₂ b₂ (relu d₂
          (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))))) k -
          oneHot d₃ label k)))
    hw₁ (layerAct_nonneg hw₂ le_rfl zero_le_one)
    (layerBudget_nonneg M.u_nonneg hw₂ le_rfl zero_le_one
      (M.cotErr_nonneg heexp0 hδ0 hρ1)) hW₁ hC1 hcot1 l0 hmargin0 j
  -- real layer-0 cotangent magnitude
  have hC0 : |reluMask (Proofs.dense W₀ b₀ x)
      (Proofs.dense (fun j' i' => W₁ i' j') (fun _ => 0)
        (reluMask (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)))
          (Proofs.dense (fun j' i' => W₂ i' j') (fun _ => 0)
            (fun k => softmax d₃ (Proofs.dense W₂ b₂ (relu d₂
              (Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x))))) k -
              oneHot d₃ label k)))) j| ≤
      layerAct d₂ w₁ 0 (layerAct d₃ w₂ 0 1) :=
    (reluMask_abs_le _ _ j).trans
      (dense_abs_le (layerAct_nonneg hw₂ le_rfl zero_le_one)
        (fun j' i' => hW₁ i' j') (fun _ => by simp) hC1 j)
  -- the final input multiply: exact left operand `x` (`ea = 0`)
  exact M.mul_close (by simp : |x i - x i| ≤ (0:ℝ)) hcot0 (hx i) hC0

/-- **One SGD step with the FloatModel binary32 input-layer gradient decreases one
    example's cross-entropy loss; the gradient's accuracy is proven, not
    assumed.** The input-layer rung of the η-composition, one mask deeper than
    the hidden rung. The gradient is the FloatModel binary32 `W₀`
    gradient `M.mlpInputFloatGrad …`, and its accuracy is *proven* by
    `mlp_w0_grad_close` (via the nested `reluMask`↔`gradAt` bridge
    `mlp_input_loss_gradAt_reluMask`), not assumed.

    Four margins are carried as hypotheses: the two
    **rounding** margins `hmargin0_round`/`hmargin1_round` (forward rounding
    must not flip either ReLU — the grad-close preconditions) and the two
    **step** margins `hmargin0_step`/`hmargin1_step` (the parameter step must
    not flip either along the segment — the smoothness preconditions). With
    this each of the three MLP weight layers has a float-gradient descent
    statement.

    Scope: one example `(x, label)`, the layer's weights only (other parameters
    fixed), and the update taken in ℝ — only the gradient is float-modelled. -/
theorem mlp_input_float_sgd_descends {d₀ d₁ d₂ d₃ : Nat} (M : FloatModel)
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (label : Fin d₃) (fexp : ℝ → ℝ)
    {lr a w₀ β₀ w₁ β₁ w₂ β₂ eexp : ℝ}
    (ha : 0 ≤ a) (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁)
    (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂) (hlr : 0 ≤ lr)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp d₃ < 1)
    (hx : ∀ i, |x i| ≤ a)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w₂) (hb₂ : ∀ j, |b₂ j| ≤ β₂)
    (hmargin0_round : ∀ j', FloatModel.layerBudget M.u d₀ w₀ β₀ a 0 <
      |dense W₀ b₀ x j'|)
    (hmargin1_round : ∀ l', FloatModel.layerBudget M.u d₁ w₁ β₁
        (FloatModel.layerAct d₀ w₀ β₀ a)
        (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0) <
      |dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) l'|)
    (hmargin0_step : ∀ j, a * (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀) lr (FloatModel.mulErr M.u a (FloatModel.layerAct d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)) 0 (FloatModel.layerBudget M.u d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1) (FloatModel.layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)) (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a) (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃))))) <
      |dense W₀ b₀ x j|)
    (hmargin1_step : ∀ l, w₁ * (a * (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀) lr (FloatModel.mulErr M.u a (FloatModel.layerAct d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)) 0 (FloatModel.layerBudget M.u d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1) (FloatModel.layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)) (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a) (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃)))))) <
      |dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) l|)
    (hsmall : 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀) lr (FloatModel.mulErr M.u a (FloatModel.layerAct d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)) 0 (FloatModel.layerBudget M.u d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1) (FloatModel.layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)) (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a) (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃))))))))) < 1)
    (h1 : lr * (FloatModel.mulErr M.u a
          (FloatModel.layerAct d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)) 0
          (FloatModel.layerBudget M.u d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)
            (FloatModel.layerBudget M.u d₃ w₂ 0 1
              (FloatModel.cotErr M.u eexp
                (FloatModel.layerBudget M.u d₂ w₂ β₂
                  (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a))
                  (FloatModel.layerBudget M.u d₁ w₁ β₁
                    (FloatModel.layerAct d₀ w₀ β₀ a)
                    (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃)))) *
        (∑ idx, |gradAt
          (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
          (Mat.flatten W₀) idx|) ≤
      lr * (∑ idx, gradAt
        (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
        (Mat.flatten W₀) idx ^ 2) / 4)
    (h2 : (2 * (d₃ : ℝ) * (d₂ : ℝ) ^ 2 * w₁ ^ 2 * w₂ ^ 2 * a ^ 2 /
        (1 - 2 * (w₂ * ((d₂ : ℝ) * (w₁ * (a * (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀) lr (FloatModel.mulErr M.u a (FloatModel.layerAct d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)) 0 (FloatModel.layerBudget M.u d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1) (FloatModel.layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)) (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a) (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃))))))))))) *
        (stepRadius (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀) lr (FloatModel.mulErr M.u a (FloatModel.layerAct d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)) 0 (FloatModel.layerBudget M.u d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1) (FloatModel.layerBudget M.u d₃ w₂ 0 1 (FloatModel.cotErr M.u eexp (FloatModel.layerBudget M.u d₂ w₂ β₂ (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)) (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a) (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃))))) ^ 2 ≤
      lr * (∑ idx, gradAt
        (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
        (Mat.flatten W₀) idx ^ 2) / 4) :
    (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀ -
          lr • M.mlpInputFloatGrad W₀ b₀ W₁ b₁ W₂ b₂ x fexp label) ≤
      (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label) (Mat.flatten W₀) -
        lr * (∑ idx, gradAt
          (mlpInputLoss b₀ W₁ b₁ W₂ b₂ x label)
          (Mat.flatten W₀) idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  unfold mlpInputLoss at *
  have hu := M.u_nonneg
  -- the proven accuracy budget η of `mlp_w0_grad_close`
  set η : ℝ := FloatModel.mulErr M.u a
    (FloatModel.layerAct d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)) 0
    (FloatModel.layerBudget M.u d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)
      (FloatModel.layerBudget M.u d₃ w₂ 0 1
        (FloatModel.cotErr M.u eexp
          (FloatModel.layerBudget M.u d₂ w₂ β₂
            (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a))
            (FloatModel.layerBudget M.u d₁ w₁ β₁
              (FloatModel.layerAct d₀ w₀ β₀ a)
              (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃))) with hη
  -- nonnegativity of every layer in the budget
  have hE₀0 : 0 ≤ FloatModel.layerBudget M.u d₀ w₀ β₀ a 0 :=
    FloatModel.layerBudget_nonneg hu hw₀ hβ₀ ha le_rfl
  have hA₀0 : 0 ≤ FloatModel.layerAct d₀ w₀ β₀ a := FloatModel.layerAct_nonneg hw₀ hβ₀ ha
  have hE₁0 : 0 ≤ FloatModel.layerBudget M.u d₁ w₁ β₁
      (FloatModel.layerAct d₀ w₀ β₀ a)
      (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0) :=
    FloatModel.layerBudget_nonneg hu hw₁ hβ₁ hA₀0 hE₀0
  have hA₁0 : 0 ≤ FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a) :=
    FloatModel.layerAct_nonneg hw₁ hβ₁ hA₀0
  have hδ0 : 0 ≤ FloatModel.layerBudget M.u d₂ w₂ β₂
      (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a))
      (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)
        (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0)) :=
    FloatModel.layerBudget_nonneg hu hw₂ hβ₂ hA₁0 hE₁0
  have hcotδ : 0 ≤ FloatModel.cotErr M.u eexp
      (FloatModel.layerBudget M.u d₂ w₂ β₂
        (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a))
        (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)
          (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃ :=
    M.cotErr_nonneg heexp0 hδ0 hρ1
  have hAct1 : 0 ≤ FloatModel.layerAct d₃ w₂ 0 1 :=
    FloatModel.layerAct_nonneg hw₂ le_rfl zero_le_one
  have hec1 : 0 ≤ FloatModel.layerBudget M.u d₃ w₂ 0 1
      (FloatModel.cotErr M.u eexp
        (FloatModel.layerBudget M.u d₂ w₂ β₂
          (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a))
          (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)
            (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃) :=
    FloatModel.layerBudget_nonneg hu hw₂ le_rfl zero_le_one hcotδ
  have hAct0 : 0 ≤ FloatModel.layerAct d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1) :=
    FloatModel.layerAct_nonneg hw₁ le_rfl hAct1
  have hec0 : 0 ≤ FloatModel.layerBudget M.u d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)
      (FloatModel.layerBudget M.u d₃ w₂ 0 1
        (FloatModel.cotErr M.u eexp
          (FloatModel.layerBudget M.u d₂ w₂ β₂
            (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a))
            (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)
              (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃)) :=
    FloatModel.layerBudget_nonneg hu hw₁ le_rfl hAct1 hec1
  have hη0 : 0 ≤ η := by
    rw [hη]
    have e1 : (0:ℝ) ≤ M.u * ((a + 0) *
        (FloatModel.layerAct d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1) +
          FloatModel.layerBudget M.u d₂ w₁ 0 (FloatModel.layerAct d₃ w₂ 0 1)
            (FloatModel.layerBudget M.u d₃ w₂ 0 1
              (FloatModel.cotErr M.u eexp
                (FloatModel.layerBudget M.u d₂ w₂ β₂
                  (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a))
                  (FloatModel.layerBudget M.u d₁ w₁ β₁
                    (FloatModel.layerAct d₀ w₀ β₀ a)
                    (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃)))) :=
      mul_nonneg hu (mul_nonneg (by linarith) (by linarith))
    have e2 : (0:ℝ) ≤ a * FloatModel.layerBudget M.u d₂ w₁ 0
        (FloatModel.layerAct d₃ w₂ 0 1)
        (FloatModel.layerBudget M.u d₃ w₂ 0 1
          (FloatModel.cotErr M.u eexp
            (FloatModel.layerBudget M.u d₂ w₂ β₂
              (FloatModel.layerAct d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a))
              (FloatModel.layerBudget M.u d₁ w₁ β₁ (FloatModel.layerAct d₀ w₀ β₀ a)
                (FloatModel.layerBudget M.u d₀ w₀ β₀ a 0))) d₃)) :=
      mul_nonneg ha hec0
    simp only [FloatModel.mulErr]
    nlinarith [e1, e2]
  -- the two pre-activations are off the kink (from the rounding margins)
  have hz0 : ∀ k, dense W₀ b₀ x k ≠ 0 := fun k => abs_pos.mp (hE₀0.trans_lt (hmargin0_round k))
  have hz1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0 := fun k =>
    abs_pos.mp (hE₁0.trans_lt (hmargin1_round k))
  -- discharge `mlp_input_sgd_descends`' abstract η by the proven grad-close
  have hgh : ∀ idx, |M.mlpInputFloatGrad W₀ b₀ W₁ b₁ W₂ b₂ x fexp label idx -
      gradAt (fun w => crossEntropy d₃ (dense W₂ b₂ (relu d₂
          (dense W₁ b₁ (relu d₁ (dense (Mat.unflatten w) b₀ x))))) label)
        (Mat.flatten W₀) idx| ≤ η := by
    intro idx
    obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective idx
    rw [mlpInputFloatGrad_apply,
      mlp_input_loss_gradAt_reluMask b₀ W₁ b₁ W₂ b₂ W₀ x label hz0 hz1 i j]
    exact mlp_w0_grad_close M W₀ b₀ W₁ b₁ W₂ b₂ x label fexp ha hw₀ hβ₀ hw₁ hβ₁
      hw₂ hβ₂ heexp0 heexp1 hfexp hρ1 hx hW₀ hb₀ hW₁ hb₁ hW₂ hb₂
      hmargin0_round hmargin1_round i j
  exact mlp_input_sgd_descends W₀ b₀ W₁ b₁ W₂ b₂ x label
    (M.mlpInputFloatGrad W₀ b₀ W₁ b₁ W₂ b₂ x fexp label)
    ha hx hw₁ hW₁ hw₂ hW₂ hlr hη0 hgh hmargin0_step hmargin1_step hsmall h1 h2

end Proofs
