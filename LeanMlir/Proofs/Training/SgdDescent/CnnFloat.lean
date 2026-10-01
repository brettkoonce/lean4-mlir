import LeanMlir.Proofs.Training.SgdDescent.Cnn

/-! # The MNIST CNN descent rungs with the FloatModel binary32 gradient

`SgdDescent.Cnn` proves one-step descent for both conv kernels and both conv biases of the
Chapter-3 MNIST CNN, assuming the gradient is accurate to `η` (an oracle). This file replaces
that oracle by a proof: `cnn_conv2_float_sgd_descends`, `cnn_conv1_float_sgd_descends`,
`cnn_conv2_bias_float_sgd_descends` and `cnn_conv1_bias_float_sgd_descends` take the FloatModel
transcription of the per-example gradient (`FloatModel.cnnConv2FloatGrad` and its three peers),
bound its distance from the certified gradient (`cnn_conv2_grad_close` and peers, budgets
`FloatModel.cnnConv2GradBudget` and peers), and feed that bound to the real rung. The update is
taken in ℝ; only the gradient is float-modelled.

The conv2-output cotangent chain is shared: `cnn_conv2_cot_close` bounds it at any (float or
exact) conv2 input, and the conv1 rungs reuse it at the float conv2 input, adding the rounded
transpose conv (`convTap_back_close`) and the conv1 ReLU mask (`mask_scalar_close`).
`cnn_conv1_cot_close` is that conv1-output cotangent, shared by the conv1 kernel and bias rungs.

The pool margins here stay `MaxPool2MarginQ`, with no twins. The proof model's pool backward
(`MaxPool2IsArgmax`) routes the cotangent to every cell attaining the window max, so at a tie it
is not the loss gradient, and a float rung needs windows without ties
(`MaxPool2MarginQ.to_marginQUpTo_flat` hands them to the real rungs). The rendered trainers'
`select_and_scatter` routes each window's cotangent to one cell instead; the two agree off ties.
-/

namespace Proofs

open StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The conv2 float-backward grad-close (two generic cores)
-- ════════════════════════════════════════════════════════════════

/-- **Scalar ReLU-mask freeze** — the `(if z>0 then 1 else 0)·x` peer of
    `reluMask_close`. Under the sign margin `ez < |z|` the float and real masks
    agree, so the masked value is 1-Lipschitz in `x`. The conv-output ReLU mask
    `𝟙[z₂>0]` in the conv-2 grad-close sits on a scalar cell (not a `Vec`), so
    it needs this rather than the vector `reluMask_close`. -/
theorem mask_scalar_close {zt z xt x ez ex : ℝ}
    (hz : |zt - z| ≤ ez) (hm : ez < |z|) (hx : |xt - x| ≤ ex) :
    |(if zt > 0 then (1:ℝ) else 0) * xt -
      (if z > 0 then (1:ℝ) else 0) * x| ≤ ex := by
  rw [if_congr (sign_stable_of_close hz hm).2 rfl rfl, ← mul_sub, abs_mul]
  exact (mul_le_of_le_one_left (abs_nonneg _) (by split_ifs <;> simp)).trans hx

/-- **The binary32 conv-2 weight gradient (FloatModel transcription of the
    per-example gradient)** — the conv peer of `mlpInputFloatGrad`. At kernel entry `(o,cc,kh,kw)` it is the
    float dot of the (exact) padded-input window `convPadWin` against the float
    conv-output cotangent slab `cotWin c̃Conv o`, where the float cotangent
    `c̃Conv` rounds every step of the backward — conv-output ReLU mask `𝟙[z̃₂>0]`,
    pool argmax selector (read on the FLOAT post-relu), and the head
    `W₃ᵀ·mask(z̃₃)·W₄ᵀ·mask(z̃₄)·W₅ᵀ·(float softmax−onehot)` at the float
    pre-activations. The `M`-free `reluMask`/`maxPoolFlat`/`relu` are exact in
    float; `M.convF`/`M.dense`/`M.softmaxCECotF` carry the rounding. -/
noncomputable def FloatModel.cnnConv2FloatGrad {c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC)
    (v : Vec (c * c * kH * kW)) : Vec (c * c * kH * kW) :=
  Kernel4.flatten fun o cc kh kw =>
    M.dot (convPadWin kH kW x₁ cc kh kw)
      (cotWin (fun ci hi wi =>
        (if Tensor3.flatten (M.convF (Kernel4.unflatten v) b₂ x₁)
              (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
          (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (M.convF (Kernel4.unflatten v) b₂ x₁)))) ci hi wi
            then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
              (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                  (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                    (M.convF (Kernel4.unflatten v) b₂ x₁)))))
                (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                  (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                      (M.dense W₃ b₃ (maxPoolFlat c h w
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (M.convF (Kernel4.unflatten v) b₂ x₁)))))))
                    (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                      (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                          (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                            (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                              (Tensor3.flatten (M.convF (Kernel4.unflatten v)
                                b₂ x₁))))))))) label)))))
              (t3Idx ci (winRow hi) (winCol wi))
            else 0)) o)

@[simp] theorem FloatModel.cnnConv2FloatGrad_apply {c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC)
    (v : Vec (c * c * kH * kW)) (o cc : Fin c) (kh : Fin kH) (kw : Fin kW) :
    M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label v (k4Idx o cc kh kw) =
      M.dot (convPadWin kH kW x₁ cc kh kw)
        (cotWin (fun ci hi wi =>
          (if Tensor3.flatten (M.convF (Kernel4.unflatten v) b₂ x₁)
                (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
            (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (M.convF (Kernel4.unflatten v) b₂ x₁)))) ci hi wi
              then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
                (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (M.convF (Kernel4.unflatten v) b₂ x₁)))))
                  (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                    (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                        (M.dense W₃ b₃ (maxPoolFlat c h w
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (M.convF (Kernel4.unflatten v) b₂ x₁)))))))
                      (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                        (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                            (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                              (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                                (Tensor3.flatten (M.convF (Kernel4.unflatten v)
                                  b₂ x₁))))))))) label)))))
                (t3Idx ci (winRow hi) (winCol wi))
              else 0)) o) := by
  simp only [FloatModel.cnnConv2FloatGrad, Kernel4.flatten, k4Idx,
    Equiv.symm_apply_apply]

/-- **The conv-2 float-backward grad-close budget** — the closed-form `η` the
    rounded `W₂` gradient stays within of the certified one. Bottom-up: the
    forward rounding nest (`Econv → E₃ → E₄ → δlogit`, conv at fan-in
    `c·kH·kW`, the dense head at `c·h·w / d₃`) feeds the head `cotErr`; the
    backward then rides two `cot_step` `layerBudget`s (W₅/W₄) and the unmasked
    W₃ `layerBudget` to `econv`; finally the spatial dot (fan-in `(2h)·(2w)`)
    contributes its Higham γ on the float-cotangent magnitude `Ctilde` plus the
    per-entry cotangent drift `econv`. The conv peer of the MLP's
    `mulErr/layerBudget/cotErr` nest, deeper by the pool + the dot. -/
noncomputable def FloatModel.cnnConv2GradBudget (M : FloatModel)
    (c h w d₃ d₄ nC kH kW : ℕ) (a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ) : ℝ :=
  let A2 := FloatModel.layerAct (c * kH * kW) w₂ β₂ a
  let A3 := FloatModel.layerAct (c * h * w) w₃ β₃ A2
  let A4 := FloatModel.layerAct d₃ w₄ β₄ A3
  let Econv := FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0
  let E3 := FloatModel.layerBudget M.u (c * h * w) w₃ β₃ A2 Econv
  let E4 := FloatModel.layerBudget M.u d₃ w₄ β₄ A3 E3
  let δlogit := FloatModel.layerBudget M.u d₄ w₅ β₅ A4 E4
  let C4 := FloatModel.layerAct nC w₅ 0 1
  let C3 := FloatModel.layerAct d₄ w₄ 0 C4
  let CPooled := FloatModel.layerAct d₃ w₃ 0 C3
  let ecHead := FloatModel.cotErr M.u eexp δlogit nC
  let ec4 := FloatModel.layerBudget M.u nC w₅ 0 1 ecHead
  let ec3 := FloatModel.layerBudget M.u d₄ w₄ 0 C4 ec4
  let econv := FloatModel.layerBudget M.u d₃ w₃ 0 C3 ec3
  let Ctilde := CPooled + econv
  ((1 + M.u) ^ ((2 * h) * (2 * w) + 1) - 1) *
      (((2 * h) * (2 * w) : ℕ) * (a * Ctilde)) +
    (((2 * h) * (2 * w) : ℕ) * (a * econv))

/-- The conv-2-output cotangent error budget, as a function of the conv-2
    input magnitude `aX2` and rounding `eX2` — the `e₂` of `cnnConv2GradBudget`
    (where `aX2 = a`, `eX2 = 0`) and the `e₂` inside `cnnConv1GradBudget` (where
    `aX2 = A₁`, `eX2 = E₁`). Factored so the conv-1 rung reuses the conv-2
    cotangent chain at a FLOAT conv-2 input. -/
noncomputable def FloatModel.cnnConv2CotBudget (M : FloatModel)
    (c h w d₃ d₄ nC kH kW : ℕ) (aX2 eX2 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ) : ℝ :=
  let A2 := FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2
  let A3 := FloatModel.layerAct (c * h * w) w₃ β₃ A2
  let A4 := FloatModel.layerAct d₃ w₄ β₄ A3
  let E2 := FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2
  let E3 := FloatModel.layerBudget M.u (c * h * w) w₃ β₃ A2 E2
  let E4 := FloatModel.layerBudget M.u d₃ w₄ β₄ A3 E3
  let δlogit := FloatModel.layerBudget M.u d₄ w₅ β₅ A4 E4
  let C4 := FloatModel.layerAct nC w₅ 0 1
  let C3 := FloatModel.layerAct d₄ w₄ 0 C4
  let ecHead := FloatModel.cotErr M.u eexp δlogit nC
  let ec4 := FloatModel.layerBudget M.u nC w₅ 0 1 ecHead
  let ec3 := FloatModel.layerBudget M.u d₄ w₄ 0 C4 ec4
  FloatModel.layerBudget M.u d₃ w₃ 0 C3 ec3

/-- The real conv-2-output cotangent magnitude bound — `aX2`/`eX2`-independent
    (the head cotangent and the two masked `Wᵀ` steps are magnitude-frozen). -/
noncomputable def FloatModel.cnnConv2CotMag (d₃ d₄ nC : ℕ)
    (w₃ w₄ w₅ : ℝ) : ℝ :=
  FloatModel.layerAct d₃ w₃ 0 (FloatModel.layerAct d₄ w₄ 0
    (FloatModel.layerAct nC w₅ 0 1))

open FloatModel in
/-- `cnnConv2CotMag` is nonnegative. -/
private theorem FloatModel.cnnConv2CotMag_nonneg {d₃ d₄ nC : ℕ} {w₃ w₄ w₅ : ℝ}
    (hw₃ : 0 ≤ w₃) (hw₄ : 0 ≤ w₄) (hw₅ : 0 ≤ w₅) :
    0 ≤ FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ :=
  layerAct_nonneg hw₃ le_rfl (layerAct_nonneg hw₄ le_rfl
    (layerAct_nonneg hw₅ le_rfl zero_le_one))

open FloatModel in
/-- `cnnConv2CotBudget` is nonnegative for nonnegative input magnitude/rounding and layer
    bounds. -/
private theorem FloatModel.cnnConv2CotBudget_nonneg (M : FloatModel) {c h w d₃ d₄ nC kH kW : ℕ}
    {aX2 eX2 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ} (haX2 : 0 ≤ aX2) (heX2 : 0 ≤ eX2)
    (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂) (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄)
    (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅) (hβ₅ : 0 ≤ β₅) (heexp0 : 0 ≤ eexp)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1) :
    0 ≤ M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW aX2 eX2 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp := by
  have hu := M.u_nonneg
  rw [FloatModel.cnnConv2CotBudget]
  exact layerBudget_nonneg hu hw₃ le_rfl
    (layerAct_nonneg hw₄ le_rfl (layerAct_nonneg hw₅ le_rfl zero_le_one))
    (layerBudget_nonneg hu hw₄ le_rfl (layerAct_nonneg hw₅ le_rfl zero_le_one)
      (layerBudget_nonneg hu hw₅ le_rfl zero_le_one
        (M.cotErr_nonneg heexp0 (layerBudget_nonneg hu hw₅ hβ₅
          (layerAct_nonneg hw₄ hβ₄ (layerAct_nonneg hw₃ hβ₃
            (layerAct_nonneg hw₂ hβ₂ haX2)))
          (layerBudget_nonneg hu hw₄ hβ₄ (layerAct_nonneg hw₃ hβ₃
            (layerAct_nonneg hw₂ hβ₂ haX2))
            (layerBudget_nonneg hu hw₃ hβ₃ (layerAct_nonneg hw₂ hβ₂ haX2)
              (layerBudget_nonneg hu hw₂ hβ₂ haX2 heX2)))) hρ1)))

open FloatModel in
/-- **The conv-2-output cotangent is float-close at a float conv-2 input**
    — the conv-2 cotangent chain of `cnn_conv2_grad_close`, factored to
    take the conv-2 input `(X2, X2F)` with `|X2F − X2| ≤ eX2`, `|X2| ≤ aX2`. The
    conv-2 rungs instantiate the exact input `X2 = X2F = x₁`, `eX2 = 0`; the
    conv-1 rung `X2 = relu(z₁)`, `X2F = relu(z̃₁)`, `eX2 = E₁`. The
    chain: float forward from `X2` (`convF_close` → `dense_close`×3) → head
    (`softmax_ce_cot_close`) → `cot_step_close`×2 → unmasked W₃ `dense_close` →
    pool-back (`poolBack_close`) → conv-2 ReLU mask (`mask_scalar_close`). -/
theorem cnn_conv2_cot_close {c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (X2 X2F : Tensor3 c (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    {aX2 eX2 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (haX2 : 0 ≤ aX2) (heX2 : 0 ≤ eX2) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅)
    (hβ₅ : 0 ≤ β₅) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hX2 : ∀ co i j, |X2F co i j - X2 co i j| ≤ eX2)
    (hX2mag : ∀ co i j, |X2 co i j| ≤ aX2)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmarginConv : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2 <
      |Tensor3.flatten (conv2d W₂ b₂ X2) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2)
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ X2)))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2)
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ X2)))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2)
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2)) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ X2)))))) q|)
    (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) :
    |((if Tensor3.flatten (M.convF W₂ b₂ X2F) (t3Idx co ho wo) > 0
          then (1:ℝ) else 0) *
        (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (M.convF W₂ b₂ X2F)))) co ho wo
          then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
            (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (M.convF W₂ b₂ X2F)))))
              (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                    (M.dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                      (Tensor3.flatten (M.convF W₂ b₂ X2F)))))))
                  (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                    (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                        (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                          (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                            (Tensor3.flatten (M.convF W₂ b₂ X2F))))))))) label)))))
            (t3Idx co (winRow ho) (winCol wo))
          else 0)) -
      ((if Tensor3.flatten (conv2d W₂ b₂ X2) (t3Idx co ho wo) > 0
          then (1:ℝ) else 0) *
        (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ X2)))) co ho wo
          then dense (fun j i' => W₃ i' j) (fun _ => 0)
            (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ X2)))))
              (dense (fun j i' => W₄ i' j) (fun _ => 0)
                (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                    (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                      (Tensor3.flatten (conv2d W₂ b₂ X2)))))))
                  (dense (fun j i' => W₅ i' j) (fun _ => 0)
                    (fun k => softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                        (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu
                          (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ X2))))))))) k - oneHot nC label k)))))
            (t3Idx co (winRow ho) (winCol wo))
          else 0))| ≤
      M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW aX2 eX2 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
        eexp := by
  -- abbreviate the forward values (real / float) from the conv-2 input X2/X2F
  set Z2C := Tensor3.flatten (conv2d W₂ b₂ X2) with hZ2C
  set Z2CF := Tensor3.flatten (M.convF W₂ b₂ X2F) with hZ2CF
  set PR := maxPoolFlat c h w (relu (c * (2*h) * (2*w)) Z2C) with hPR
  set PF := maxPoolFlat c h w (relu (c * (2*h) * (2*w)) Z2CF) with hPF
  set Z3 := dense W₃ b₃ PR with hZ3
  set Z3F := M.dense W₃ b₃ PF with hZ3F
  set Z4 := dense W₄ b₄ (relu d₃ Z3) with hZ4
  set Z4F := M.dense W₄ b₄ (relu d₃ Z3F) with hZ4F
  set Z5 := dense W₅ b₅ (relu d₄ Z4) with hZ5
  set Z5F := M.dense W₅ b₅ (relu d₄ Z4F) with hZ5F
  set A2 := FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2 with hA2
  set A3 := FloatModel.layerAct (c * h * w) w₃ β₃ A2 with hA3
  set A4 := FloatModel.layerAct d₃ w₄ β₄ A3 with hA4
  set E2 := FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2 with hE2
  set E3 := FloatModel.layerBudget M.u (c * h * w) w₃ β₃ A2 E2 with hE3
  set E4 := FloatModel.layerBudget M.u d₃ w₄ β₄ A3 E3 with hE4
  set DL := FloatModel.layerBudget M.u d₄ w₅ β₅ A4 E4 with hDL
  set C4 := FloatModel.layerAct nC w₅ 0 1 with hC4
  set C3 := FloatModel.layerAct d₄ w₄ 0 C4 with hC3
  set ecH := FloatModel.cotErr M.u eexp DL nC with hecH
  set ec4 := FloatModel.layerBudget M.u nC w₅ 0 1 ecH with hec4
  set ec3 := FloatModel.layerBudget M.u d₄ w₄ 0 C4 ec4 with hec3
  have A2nn : 0 ≤ A2 := layerAct_nonneg hw₂ hβ₂ haX2
  have A3nn : 0 ≤ A3 := layerAct_nonneg hw₃ hβ₃ A2nn
  have A4nn : 0 ≤ A4 := layerAct_nonneg hw₄ hβ₄ A3nn
  have E2nn : 0 ≤ E2 := layerBudget_nonneg M.u_nonneg hw₂ hβ₂ haX2 heX2
  have E3nn : 0 ≤ E3 := layerBudget_nonneg M.u_nonneg hw₃ hβ₃ A2nn E2nn
  have E4nn : 0 ≤ E4 := layerBudget_nonneg M.u_nonneg hw₄ hβ₄ A3nn E3nn
  have DLnn : 0 ≤ DL := layerBudget_nonneg M.u_nonneg hw₅ hβ₅ A4nn E4nn
  have C4nn : 0 ≤ C4 := layerAct_nonneg hw₅ le_rfl zero_le_one
  have ecHnn : 0 ≤ ecH := M.cotErr_nonneg heexp0 DLnn hρ1
  have ec4nn : 0 ≤ ec4 := layerBudget_nonneg M.u_nonneg hw₅ le_rfl zero_le_one ecHnn
  have ec3nn : 0 ≤ ec3 := layerBudget_nonneg M.u_nonneg hw₄ le_rfl C4nn ec4nn
  -- forward magnitudes (real)
  have hMconv : ∀ k, |Z2C k| ≤ A2 := by
    intro k; obtain ⟨ci, hi, wi, rfl⟩ := t3Idx_surj k
    rw [hZ2C, flatten_t3Idx]; exact conv2d_abs_le haX2 hW₂ hb₂ hX2mag ci hi wi
  have hMpool : ∀ j, |PR j| ≤ A2 :=
    fun j => maxPoolFlat_abs_le (fun k => (relu_abs_le _ k).trans (hMconv k)) j
  have hM3 : ∀ l, |relu d₃ Z3 l| ≤ A3 :=
    fun l => (relu_abs_le _ l).trans (dense_abs_le A2nn hW₃ hb₃ hMpool l)
  have hM4 : ∀ q, |relu d₄ Z4 q| ≤ A4 :=
    fun q => (relu_abs_le _ q).trans (dense_abs_le A3nn hW₄ hb₄ hM3 q)
  -- forward closeness (float vs real)
  have hEconv : ∀ k, |Z2CF k - Z2C k| ≤ E2 := by
    intro k; obtain ⟨ci, hi, wi, rfl⟩ := t3Idx_surj k
    rw [hZ2CF, hZ2C, flatten_t3Idx, flatten_t3Idx]
    exact (M.convF_close W₂ b₂ X2F X2 heX2 hX2 ci hi wi).trans
      (M.denseErr_le_uniform hw₂ heX2 (fun i j => convKernelMat_abs_le hW₂ i j)
        hb₂ (fun idx => convWindow_abs_le haX2 hX2mag hi wi idx) ci)
  have hRelu : ∀ k, |relu (c * (2*h) * (2*w)) Z2CF k -
      relu (c * (2*h) * (2*w)) Z2C k| ≤ E2 := fun k => relu_close _ _ _ hEconv k
  have hPool : ∀ k, |PF k - PR k| ≤ E2 := fun k => maxPoolFlat_close _ _ hRelu k
  have hE3close : ∀ l, |Z3F l - Z3 l| ≤ E3 := fun l =>
    (M.dense_close W₃ b₃ PF PR E2 E2nn hPool l).trans
      (M.denseErr_le_uniform hw₃ E2nn hW₃ hb₃ hMpool l)
  have hRelu3 : ∀ l, |relu d₃ Z3F l - relu d₃ Z3 l| ≤ E3 :=
    fun l => relu_close _ _ _ hE3close l
  have hE4close : ∀ q, |Z4F q - Z4 q| ≤ E4 := fun q =>
    (M.dense_close W₄ b₄ (relu d₃ Z3F) (relu d₃ Z3) E3 E3nn hRelu3 q).trans
      (M.denseErr_le_uniform hw₄ E3nn hW₄ hb₄ hM3 q)
  have hRelu4 : ∀ q, |relu d₄ Z4F q - relu d₄ Z4 q| ≤ E4 :=
    fun q => relu_close _ _ _ hE4close q
  have hDLclose : ∀ k, |Z5F k - Z5 k| ≤ DL := fun k =>
    (M.dense_close W₅ b₅ (relu d₄ Z4F) (relu d₄ Z4) E4 E4nn hRelu4 k).trans
      (M.denseErr_le_uniform hw₅ E4nn hW₅ hb₅ hM4 k)
  -- head cotangent + real head magnitude
  have hHeadCot : ∀ k, |M.softmaxCECotF fexp Z5F label k -
      (softmax nC Z5 k - oneHot nC label k)| ≤ ecH := fun k =>
    M.softmax_ce_cot_close fexp Z5F Z5 label heexp0 heexp1 hfexp hρ1 hDLclose k
  have hHeadMag : ∀ k, |softmax nC Z5 k - oneHot nC label k| ≤ 1 :=
    fun k => abs_softmax_sub_oneHot_le_one _ label k
  -- two masked Wᵀ cotangent steps + unmasked W₃ step
  have hc4 : ∀ q, |FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j)
        (fun _ => 0) (M.softmaxCECotF fexp Z5F label)) q -
      FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
        (fun k => softmax nC Z5 k - oneHot nC label k)) q| ≤ ec4 := fun q =>
    M.cot_step_close W₅ Z4F Z4 (M.softmaxCECotF fexp Z5F label)
      (fun k => softmax nC Z5 k - oneHot nC label k) hw₅ zero_le_one ecHnn hW₅
      hHeadMag hHeadCot hE4close hmargin4 q
  have hc4Mag : ∀ q, |FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j)
      (fun _ => 0) (fun k => softmax nC Z5 k - oneHot nC label k)) q| ≤ C4 :=
    fun q => (reluMask_abs_le _ _ q).trans
      (dense_abs_le zero_le_one (fun i j => hW₅ j i) (fun _ => by simp) hHeadMag q)
  have hc3 : ∀ l, |FloatModel.reluMask Z3F (M.dense (fun j i' => W₄ i' j)
        (fun _ => 0) (FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j)
          (fun _ => 0) (M.softmaxCECotF fexp Z5F label)))) l -
      FloatModel.reluMask Z3 (dense (fun j i' => W₄ i' j) (fun _ => 0)
        (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
          (fun k => softmax nC Z5 k - oneHot nC label k)))) l| ≤ ec3 := fun l =>
    M.cot_step_close W₄ Z3F Z3
      (FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
        (M.softmaxCECotF fexp Z5F label)))
      (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
        (fun k => softmax nC Z5 k - oneHot nC label k)))
      hw₄ C4nn ec4nn hW₄ hc4Mag hc4 hE3close hmargin3 l
  have hc3Mag : ∀ l, |FloatModel.reluMask Z3 (dense (fun j i' => W₄ i' j)
      (fun _ => 0) (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j)
        (fun _ => 0) (fun k => softmax nC Z5 k - oneHot nC label k)))) l| ≤ C3 :=
    fun l => (reluMask_abs_le _ _ l).trans
      (dense_abs_le C4nn (fun i j => hW₄ j i) (fun _ => by simp) hc4Mag l)
  have hcPool : ∀ j, |M.dense (fun j' i' => W₃ i' j') (fun _ => 0)
        (FloatModel.reluMask Z3F (M.dense (fun j' i' => W₄ i' j') (fun _ => 0)
          (FloatModel.reluMask Z4F (M.dense (fun j' i' => W₅ i' j') (fun _ => 0)
            (M.softmaxCECotF fexp Z5F label))))) j -
      dense (fun j' i' => W₃ i' j') (fun _ => 0)
        (FloatModel.reluMask Z3 (dense (fun j' i' => W₄ i' j') (fun _ => 0)
          (FloatModel.reluMask Z4 (dense (fun j' i' => W₅ i' j') (fun _ => 0)
            (fun k => softmax nC Z5 k - oneHot nC label k))))) j| ≤
      FloatModel.layerBudget M.u d₃ w₃ 0 C3 ec3 := fun j =>
    (M.dense_close (fun j' i' => W₃ i' j') (fun _ => 0) _ _ ec3 ec3nn hc3 j).trans
      (M.denseErr_le_uniform hw₃ ec3nn (fun i j' => hW₃ j' i) (fun _ => by simp)
        hc3Mag j)
  -- pool freeze + conv-2 ReLU mask freeze → the per-cell cotangent close
  have hPostRelu : ∀ ci hi wi,
      |Tensor3.unflatten (relu (c * (2*h) * (2*w)) Z2CF) ci hi wi -
        Tensor3.unflatten (relu (c * (2*h) * (2*w)) Z2C) ci hi wi| ≤ E2 := by
    intro ci hi wi; rw [unflatten_t3Idx, unflatten_t3Idx]
    exact hRelu (t3Idx ci hi wi)
  rw [FloatModel.cnnConv2CotBudget]
  have hpb := hmarginPool.poolBack_close hPostRelu co ho wo
    (hcPool (t3Idx co (winRow ho) (winCol wo)))
  exact mask_scalar_close (hEconv (t3Idx co ho wo)) (hmarginConv (t3Idx co ho wo))
    hpb

open FloatModel in
/-- **The real conv-2-output cotangent is magnitude-bounded** by `cnnConv2CotMag`
    — the `aX2`/`eX2`-independent ℓ∞ bound (the conv-2 ReLU mask and pool
    selector only shrink, the head cotangent is in `[−1,1]`, the two masked `Wᵀ`
    steps and the unmasked W₃ ride `layerAct`). Used to bound the real conv-1
    cotangent `∑ convTap·c₂` in the conv-1 rung. -/
theorem cnn_conv2_cot_real_abs_le {c h w d₃ d₄ nC kH kW : Nat}
    (X2 : Tensor3 c (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    {w₃ w₄ w₅ : ℝ} (hw₃ : 0 ≤ w₃) (hw₄ : 0 ≤ w₄) (hw₅ : 0 ≤ w₅)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) :
    |(if Tensor3.flatten (conv2d W₂ b₂ X2) (t3Idx co ho wo) > 0
          then (1:ℝ) else 0) *
        (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ X2)))) co ho wo
          then dense (fun j i' => W₃ i' j) (fun _ => 0)
            (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ X2)))))
              (dense (fun j i' => W₄ i' j) (fun _ => 0)
                (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                    (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                      (Tensor3.flatten (conv2d W₂ b₂ X2)))))))
                  (dense (fun j i' => W₅ i' j) (fun _ => 0)
                    (fun k => softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                        (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu
                          (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ X2))))))))) k - oneHot nC label k)))))
            (t3Idx co (winRow ho) (winCol wo))
          else 0)| ≤ FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ := by
  rw [FloatModel.cnnConv2CotMag]
  set PR := maxPoolFlat c h w (relu (c * (2*h) * (2*w))
    (Tensor3.flatten (conv2d W₂ b₂ X2))) with hPR
  set Z5 := dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ PR)))) with hZ5
  have hHeadMag : ∀ k, |softmax nC Z5 k - oneHot nC label k| ≤ 1 :=
    fun k => abs_softmax_sub_oneHot_le_one _ label k
  have hc4Mag : ∀ q, |FloatModel.reluMask
      (dense W₄ b₄ (relu d₃ (dense W₃ b₃ PR))) (dense (fun j i' => W₅ i' j)
      (fun _ => 0) (fun k => softmax nC Z5 k - oneHot nC label k)) q| ≤
      FloatModel.layerAct nC w₅ 0 1 := fun q => (reluMask_abs_le _ _ q).trans
    (dense_abs_le zero_le_one (fun i j => hW₅ j i) (fun _ => by simp) hHeadMag q)
  have hc3Mag : ∀ l, |FloatModel.reluMask (dense W₃ b₃ PR)
      (dense (fun j i' => W₄ i' j) (fun _ => 0)
        (FloatModel.reluMask (dense W₄ b₄ (relu d₃ (dense W₃ b₃ PR)))
          (dense (fun j i' => W₅ i' j) (fun _ => 0)
            (fun k => softmax nC Z5 k - oneHot nC label k)))) l| ≤
      FloatModel.layerAct d₄ w₄ 0 (FloatModel.layerAct nC w₅ 0 1) :=
    fun l => (reluMask_abs_le _ _ l).trans
      (dense_abs_le (layerAct_nonneg hw₅ le_rfl zero_le_one) (fun i j => hW₄ j i)
        (fun _ => by simp) hc4Mag l)
  have hcPoolMag : ∀ j, |dense (fun j' i' => W₃ i' j') (fun _ => 0)
      (FloatModel.reluMask (dense W₃ b₃ PR) (dense (fun j' i' => W₄ i' j')
        (fun _ => 0) (FloatModel.reluMask (dense W₄ b₄ (relu d₃ (dense W₃ b₃ PR)))
          (dense (fun j' i' => W₅ i' j') (fun _ => 0)
            (fun k => softmax nC Z5 k - oneHot nC label k))))) j| ≤
      FloatModel.layerAct d₃ w₃ 0 (FloatModel.layerAct d₄ w₄ 0
        (FloatModel.layerAct nC w₅ 0 1)) :=
    fun j => dense_abs_le (layerAct_nonneg hw₄ le_rfl
      (layerAct_nonneg hw₅ le_rfl zero_le_one)) (fun i j' => hW₃ j' i)
      (fun _ => by simp) hc3Mag j
  have h0 : 0 ≤ FloatModel.layerAct d₃ w₃ 0 (FloatModel.layerAct d₄ w₄ 0
      (FloatModel.layerAct nC w₅ 0 1)) := FloatModel.layerAct_nonneg hw₃ le_rfl
    (FloatModel.layerAct_nonneg hw₄ le_rfl (FloatModel.layerAct_nonneg hw₅ le_rfl zero_le_one))
  split_ifs <;> simp [h0, hcPoolMag]

open FloatModel in
/-- **The binary32 conv-2 weight gradient is within an explicit budget of the
    certified one** — the conv-layer peer of `mlp_w0_grad_close`. With
    the conv-2 input `x₁` exact, the FloatModel `W₂` gradient
    `M.cnnConv2FloatGrad …` stays within `cnnConv2GradBudget` of the certified
    `gradAt`. The chain: float forward (`convF_close` → `dense_close`×3, relu
    and pool error-transparent) ⟶ head (`softmax_ce_cot_close`) ⟶ two masked
    `Wᵀ` `cot_step_close` (W₅ under z̃₄, W₄ under z̃₃) ⟶ unmasked W₃ `dense_close`
    ⟶ pool-backward freeze (`poolBack_close`) ⟶ conv-output ReLU mask freeze
    (`mask_scalar_close`) ⟶ the spatial dot (`dot_perturbed_close`). Four
    quantitative margins are carried (conv-output `Econv`, pool `Econv` POST-relu,
    z̃₃ `E₃`, z̃₄ `E₄`); the bridge `cnn_conv2_loss_gradAt_reluMask` turns the
    `gradAt` into the dot the float gradient rounds. Everything up to the dot is
    `cnn_conv2_cot_close` at the exact conv-2 input. -/
theorem cnn_conv2_grad_close {c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    (v : Vec (c * c * kH * kW))
    {a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (ha : 0 ≤ a) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂) (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃)
    (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅) (hβ₅ : 0 ≤ β₅)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx₁ : ∀ ci i j, |x₁ ci i j| ≤ a)
    (hv2 : ∀ idx, |v idx| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmarginConv : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0 <
      |Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q|)
    (o cc : Fin c) (kh : Fin kH) (kw : Fin kW) :
    |M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label v (k4Idx o cc kh kw) -
      gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label) v (k4Idx o cc kh kw)|
      ≤ M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp := by
  have hv2' : ∀ o' c' kh' kw', |Kernel4.unflatten v o' c' kh' kw'| ≤ w₂ :=
    fun o' c' kh' kw' => by rw [unflatten_k4Idx]; exact hv2 _
  have Ecnn : 0 ≤ FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0 :=
    layerBudget_nonneg M.u_nonneg hw₂ hβ₂ ha le_rfl
  -- per-cell conv-2 cotangent closeness / magnitudes (exact conv-2 input x₁)
  have hc2close := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    cnn_conv2_cot_close M x₁ x₁ (Kernel4.unflatten v) b₂ W₃ b₃ W₄ b₄ W₅ b₅ label fexp ha
      (le_refl 0) hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 heexp1 hfexp hρ1
      (fun _ _ _ => by simp) hx₁ hv2' hb₂ hW₃ hb₃ hW₄ hb₄ hW₅ hb₅
      hmarginConv hmarginPool hmargin3 hmargin4 co ho wo
  have hc2realmag := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    cnn_conv2_cot_real_abs_le x₁ (Kernel4.unflatten v) b₂ W₃ b₃ W₄ b₄ W₅ b₅ label
      hw₃ hw₄ hw₅ hW₃ hW₄ hW₅ co ho wo
  -- assemble: rewrite to the dot form (apply + bridge), then the dot composite
  rw [M.cnnConv2FloatGrad_apply b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label v o cc kh kw,
    cnn_conv2_loss_gradAt_reluMask b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label v
      (fun k => abs_pos.mp (lt_of_le_of_lt Ecnn (hmarginConv k)))
      (hmarginPool.smooth Ecnn)
      (fun l => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₃ hβ₃
        (layerAct_nonneg hw₂ hβ₂ ha) Ecnn) (hmargin3 l)))
      (fun q => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₄ hβ₄
        (layerAct_nonneg hw₃ hβ₃ (layerAct_nonneg hw₂ hβ₂ ha))
        (layerBudget_nonneg M.u_nonneg hw₃ hβ₃ (layerAct_nonneg hw₂ hβ₂ ha) Ecnn))
        (hmargin4 q)))
      o cc kh kw]
  exact M.dot_perturbed_close (convPadWin kH kW x₁ cc kh kw) _ _ ha
    (fun s => by simp only [convPadWin]; exact abs_convPad_le x₁ ha hx₁ cc kh kw _ _)
    (fun s => by simp only [cotWin]; exact abs_le_of_close (hc2close o _ _) (hc2realmag o _ _))
    (fun s => by simp only [cotWin]; exact hc2close o _ _)

-- ════════════════════════════════════════════════════════════════
-- § The conv2 float rung
-- ════════════════════════════════════════════════════════════════

open FloatModel in
/-- **One SGD step with the FloatModel binary32 conv-2 kernel gradient decreases
    one example's cross-entropy loss; the gradient's accuracy is proven, not
    assumed.** The conv peer of `mlp_input_float_sgd_descends`: the gradient is the
    FloatModel binary32 `W₂` gradient `M.cnnConv2FloatGrad …`, and its accuracy is *proven* by
    `cnn_conv2_grad_close` (η := `cnnConv2GradBudget`, discharged per kernel
    entry via `k4Idx_surj`), not assumed. The two rounding-margin families are
    carried as hypotheses: the per-layer ROUND margins
    (`hmarginConv/Pool/3/4`, feeding the grad-close) and the gradient-radius
    STEP margins + `hsmall`/`h1`/`h2` (feeding `cnn_conv2_sgd_descends`'s
    drift-freeze and the descent geometry). The conv-2 input `x₁` is exact.

    Scope: one example, `W₂` moving with every other parameter fixed, and the
    update taken in ℝ — only the gradient is float-modelled. -/
theorem cnn_conv2_float_sgd_descends {c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    {lr a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (ha : 0 ≤ a) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂) (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃)
    (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅) (hβ₅ : 0 ≤ β₅) (hlr : 0 ≤ lr)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmarginConv : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0 <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hm2 : ∀ k, a * (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr (M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)) <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmq : MaxPool2MarginQ (a * (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr (M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr (M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr (M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr (M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))))))))) < 1)
    (h1 : lr * (M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
          eexp) * (∑ idx, |gradAt
        (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) idx|) ≤
      lr * (∑ idx, gradAt
        (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr (M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))))))))))) *
        (stepRadius (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) lr (M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)) ^ 2 ≤
      lr * (∑ idx, gradAt
        (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) idx ^ 2) / 4) :
    (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂ -
              lr • M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
                (Kernel4.flatten W₂)) ≤
      (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₂) -
        lr * (∑ idx, gradAt
          (cnnConv2KernelLoss b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
          (Kernel4.flatten W₂) idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  unfold cnnConv2KernelLoss at *
  have hu := M.u_nonneg
  -- nonnegativity of the proven budget
  have CPnn : 0 ≤ FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ :=
    FloatModel.cnnConv2CotMag_nonneg hw₃ hw₄ hw₅
  have ecvnn : 0 ≤ M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW a 0 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
      eexp := M.cnnConv2CotBudget_nonneg ha le_rfl hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 hρ1
  have hη0 : 0 ≤ M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄
      w₅ β₅ eexp := by
    simp only [FloatModel.cnnConv2GradBudget]
    have hγ : (0:ℝ) ≤ (1 + M.u) ^ ((2 * h) * (2 * w) + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith))
    have hn : (0:ℝ) ≤ (((2 * h) * (2 * w) : ℕ) : ℝ) := Nat.cast_nonneg _
    exact add_nonneg
      (mul_nonneg hγ (mul_nonneg hn (mul_nonneg ha (add_nonneg CPnn ecvnn))))
      (mul_nonneg hn (mul_nonneg ha ecvnn))
  -- the flattened kernel inherits the per-entry bound
  have hv2 : ∀ idx, |Kernel4.flatten W₂ idx| ≤ w₂ := by
    intro idx
    obtain ⟨o', c', kh', kw', rfl⟩ := k4Idx_surj idx
    rw [flatten_k4Idx]; exact hW₂ o' c' kh' kw'
  -- discharge the abstract gradient accuracy by the proven grad-close
  have hgh : ∀ idx, |M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
      (Kernel4.flatten W₂) idx -
      gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label) (Kernel4.flatten W₂) idx| ≤
      M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp := by
    intro idx
    obtain ⟨o', c', kh', kw', rfl⟩ := k4Idx_surj idx
    exact cnn_conv2_grad_close M b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label fexp
      (Kernel4.flatten W₂) ha hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅
      heexp0 heexp1 hfexp hρ1 hx hv2 hb₂ hW₃ hb₃ hW₄ hb₄ hW₅ hb₅
      (fun k => by rw [Kernel4.unflatten_flatten]; exact hmarginConv k)
      (by rw [Kernel4.unflatten_flatten]; exact hmarginPool)
      (fun l => by rw [Kernel4.unflatten_flatten]; exact hmargin3 l)
      (fun q => by rw [Kernel4.unflatten_flatten]; exact hmargin4 q)
      o' c' kh' kw'
  exact cnn_conv2_sgd_descends W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label
    (M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label (Kernel4.flatten W₂))
    (fun _ _ => False) ha hx (fun _ _ h => h.elim) hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ hlr hη0 hgh hm2
    (MaxPool2MarginQ.to_marginQUpTo_flat _ hmq) hm3 hm4 hsmall h1 h2

-- ════════════════════════════════════════════════════════════════
-- § The conv1 float-backward grad-close
-- ════════════════════════════════════════════════════════════════

/-- **The float conv-1-output cotangent**: the conv-1 ReLU mask `𝟙[z̃₁>0]` times the float conv-2
    backward `M.dot (convTap W₂ slab) (float conv-2-output cotangent slab)`, the conv-2 cotangent
    taken at the FLOAT conv-2 input `relu(z̃₁)`. The conv-1 kernel and bias gradients
    (`cnnConv1FloatGrad`, `cnnConv1BiasFloatGrad`) are its rounded dot against the padded input
    window and its rounded sum. -/
noncomputable def FloatModel.cnnConv1CotF {ic c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC) :
    Tensor3 c (2*h) (2*w) :=
  fun ci hi wi =>
    (if Tensor3.flatten (M.convF W₁ b₁ x₀)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
      M.dot (Tensor3.flatten (fun co ho wo => convTap W₂ ci hi wi co ho wo))
        (Tensor3.flatten (fun co ho wo =>
          (if Tensor3.flatten (M.convF W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (M.convF W₁ b₁ x₀)))))
                (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
            (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (M.convF W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (M.convF W₁ b₁ x₀)))))))) co ho wo
              then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
                (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (M.convF W₂ b₂ (Tensor3.unflatten (relu
                        (c * (2*h) * (2*w)) (Tensor3.flatten
                          (M.convF W₁ b₁ x₀)))))))))
                  (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                    (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                        (M.dense W₃ b₃ (maxPoolFlat c h w (relu
                          (c * (2*h) * (2*w)) (Tensor3.flatten
                            (M.convF W₂ b₂ (Tensor3.unflatten (relu
                              (c * (2*h) * (2*w)) (Tensor3.flatten
                                (M.convF W₁ b₁ x₀)))))))))))
                      (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                        (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                            (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                              (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                                (Tensor3.flatten (M.convF W₂ b₂
                                  (Tensor3.unflatten (relu
                                    (c * (2*h) * (2*w)) (Tensor3.flatten
                                      (M.convF W₁ b₁ x₀))))))))))))) label)))))
                (t3Idx co (winRow ho) (winCol wo))
              else 0)))

/-- **The certified conv-1-output cotangent**, in the `reluMask` form that
    `cnn_conv1_loss_gradAt_reluMask` and `cnn_conv1_bias_loss_gradAt_reluMask` contract against
    the padded input window and sum: the conv-1 ReLU mask times the conv-2 backward
    `∑ convTap·c₂` of the conv-2-output cotangent. -/
noncomputable def cnnConv1CotR {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) : Tensor3 c (2*h) (2*w) :=
  fun ci hi wi =>
    (if Tensor3.flatten (conv2d W₁ b₁ x₀)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
      ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
        convTap W₂ ci hi wi co ho wo *
          ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁ b₁ x₀)))))
                (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
            (if MaxPool2IsArgmax (Tensor3.unflatten
                  (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                    (conv2d W₂ b₂ (Tensor3.unflatten
                      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                        (conv2d W₁ b₁ x₀))))))))
                  co ho wo
              then dense (fun j i' => W₃ i' j) (fun _ => 0)
                (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d W₁ b₁ x₀)))))))))
                  (dense (fun j i' => W₄ i' j) (fun _ => 0)
                    (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                        (dense W₃ b₃ (maxPoolFlat c h w (relu
                          (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten (relu
                              (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₁ b₁ x₀)))))))))))
                      (dense (fun j i' => W₅ i' j) (fun _ => 0)
                        (fun k => softmax nC (dense W₅ b₅ (relu d₄
                            (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                              (maxPoolFlat c h w (relu
                                (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d W₂ b₂ (Tensor3.unflatten
                                    (relu (c * (2*h) * (2*w))
                                      (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))))))) k -
                          oneHot nC label k)))))
                (t3Idx co (winRow ho) (winCol wo))
              else 0))

/-- **The binary32 conv-1 weight gradient (FloatModel transcription of the
    per-example gradient)** — the conv-1 peer of `cnnConv2FloatGrad`, one conv-backward deeper. At kernel
    entry `(o,cc,kh,kw)` it is the float dot of the (exact) padded-input window
    `convPadWin x₀` against the slab of the float conv-1-output cotangent `cnnConv1CotF` at the
    conv-1 kernel `u`. All `M`-ops carry the rounding; `reluMask`/`maxPoolFlat`/`relu`/`convTap`
    are exact. -/
noncomputable def FloatModel.cnnConv1FloatGrad {ic c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC)
    (u : Vec (c * ic * kH * kW)) : Vec (c * ic * kH * kW) :=
  Kernel4.flatten fun o cc kh kw =>
    M.dot (convPadWin kH kW x₀ cc kh kw)
      (cotWin (M.cnnConv1CotF (Kernel4.unflatten u) b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label) o)

@[simp] theorem FloatModel.cnnConv1FloatGrad_apply {ic c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC)
    (u : Vec (c * ic * kH * kW)) (o : Fin c) (cc : Fin ic) (kh : Fin kH)
    (kw : Fin kW) :
    M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label u
        (k4Idx o cc kh kw) =
    M.dot (convPadWin kH kW x₀ cc kh kw)
      (cotWin (M.cnnConv1CotF (Kernel4.unflatten u) b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label) o) := by
  simp only [FloatModel.cnnConv1FloatGrad, Kernel4.flatten, k4Idx,
    Equiv.symm_apply_apply]

/-- **The conv-1 cotangent drift budget**: the conv-2 backward (transpose conv) of the float
    conv-2-output cotangent, a rounded dot over the slab `c·(2h)·(2w)` — its Higham γ against the
    float-cotangent magnitude `CP + e₂` plus the per-entry drift `e₂`, with the conv-2 cotangent
    chain (`cnnConv2CotMag` / `cnnConv2CotBudget`) at the FLOAT conv-2 input (`aX2 = A₁`,
    `eX2 = E₁`). -/
noncomputable def FloatModel.cnnConv1CotBudget (M : FloatModel)
    (ic c h w d₃ d₄ nC kH kW : ℕ)
    (a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ) : ℝ :=
  let A1 := FloatModel.layerAct (ic * kH * kW) w₁ β₁ a
  let E1 := FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0
  let CP := FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅
  let e2 := M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW A1 E1 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp
  ((1 + M.u) ^ ((c * (2 * h) * (2 * w)) + 1) - 1) *
      (((c * (2 * h) * (2 * w) : ℕ) : ℝ) * (w₂ * (CP + e2))) +
    (((c * (2 * h) * (2 * w) : ℕ) : ℝ) * (w₂ * e2))

/-- **The conv-1 float-backward grad-close budget** — the conv-2 budget
    (`cnnConv2GradBudget`-shaped) deepened by one conv layer: the conv-1 spatial dot (fan-in
    `(2h)·(2w)`) rides the conv-1 cotangent drift `eback = cnnConv1CotBudget` and the float
    conv-1-cotangent magnitude `C1t = c·(2h)·(2w)·w₂·CP + eback`. -/
noncomputable def FloatModel.cnnConv1GradBudget (M : FloatModel)
    (ic c h w d₃ d₄ nC kH kW : ℕ)
    (a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ) : ℝ :=
  let CP := FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅
  let eback := M.cnnConv1CotBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp
  let C1t := ((c * (2 * h) * (2 * w) : ℕ) : ℝ) * (w₂ * CP) + eback
  ((1 + M.u) ^ ((2 * h) * (2 * w) + 1) - 1) *
      (((2 * h) * (2 * w) : ℕ) * (a * C1t)) +
    (((2 * h) * (2 * w) : ℕ) * (a * eback))

/-- **The float conv-2 backward (transpose conv) against a perturbed
    cotangent.** The rounded `M.dot` of the exact `convTap` slab against the
    float conv-2-output cotangent `c2F`, vs the certified `∑ convTap·c2R` — the
    `convTap`-flattening (`sum_t3`) plus `dot_perturbed_close` (fan-in
    `c·(2h)·(2w)`, per-entry tap bound `w₂`, float-cotangent magnitude `C2t`,
    drift `e₂`). Generic in `(c2F, c2R)` so the conv-1 rung passes the conv-2
    cotangent tensors abstractly. -/
theorem convTap_back_close {c h w kH kW : Nat} (M : FloatModel)
    (W₂ : Kernel4 c c kH kW) (c2F c2R : Tensor3 c (2*h) (2*w))
    {w₂ C2t e2 : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hc2F : ∀ co ho wo, |c2F co ho wo| ≤ C2t)
    (hc2close : ∀ co ho wo, |c2F co ho wo - c2R co ho wo| ≤ e2)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |M.dot (Tensor3.flatten (fun co ho wo => convTap W₂ ci hi wi co ho wo))
        (Tensor3.flatten c2F) -
      ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
        convTap W₂ ci hi wi co ho wo * c2R co ho wo| ≤
      ((1 + M.u) ^ ((c * (2*h) * (2*w)) + 1) - 1) *
          (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * C2t)) +
        (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * e2)) := by
  have htap_eq : (∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
        convTap W₂ ci hi wi co ho wo * c2R co ho wo) =
      ∑ s, (Tensor3.flatten (fun co ho wo => convTap W₂ ci hi wi co ho wo)) s *
        (Tensor3.flatten c2R) s := by
    rw [sum_t3 (fun s => (Tensor3.flatten
      (fun co ho wo => convTap W₂ ci hi wi co ho wo)) s * (Tensor3.flatten c2R) s)]
    refine Finset.sum_congr rfl fun co _ => Finset.sum_congr rfl fun ho _ =>
      Finset.sum_congr rfl fun wo _ => ?_
    rw [flatten_t3Idx, flatten_t3Idx]
  rw [htap_eq]
  exact M.dot_perturbed_close
    (Tensor3.flatten (fun co ho wo => convTap W₂ ci hi wi co ho wo))
    (Tensor3.flatten c2F) (Tensor3.flatten c2R) hw₂
    (fun s => by obtain ⟨co, ho, wo, rfl⟩ := t3Idx_surj s
                 rw [flatten_t3Idx]; exact convTap_abs_le hw₂ hW₂ ci hi wi co ho wo)
    (fun s => by obtain ⟨co, ho, wo, rfl⟩ := t3Idx_surj s
                 rw [flatten_t3Idx]; exact hc2F co ho wo)
    (fun s => by obtain ⟨co, ho, wo, rfl⟩ := t3Idx_surj s
                 rw [flatten_t3Idx, flatten_t3Idx]; exact hc2close co ho wo)

open FloatModel in
/-- **The conv-1-output cotangent is float-close** — the part `cnn_conv1_grad_close` and
    `cnn_conv1_bias_grad_close` share. Under the five margins, at every conv-1 output cell the
    float cotangent `cnnConv1CotF` is within `cnnConv1CotBudget` of the certified `cnnConv1CotR`,
    and its magnitude within `c·(2h)·(2w)·w₂·cnnConv2CotMag + cnnConv1CotBudget`. The chain: the
    conv-1 forward closes (`convF_close`), so the conv-2 input does (`relu_close`); the conv-2
    cotangent chain at that float input (`cnn_conv2_cot_close`, with its real magnitude
    `cnn_conv2_cot_real_abs_le`); the rounded transpose conv (`convTap_back_close`, with
    `convTap_back_abs_le`); the conv-1 ReLU mask freezes (`mask_scalar_close`). -/
theorem cnn_conv1_cot_close {ic c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (fexp : ℝ → ℝ)
    {a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (ha : 0 ≤ a) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅)
    (hβ₅ : 0 ≤ β₅) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx₀ : ∀ ci i j, |x₀ ci i j| ≤ a)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmargin1 : ∀ k, FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 <
      |Tensor3.flatten (conv2d W₁ b₁ x₀) k|)
    (hmargin2 : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
          (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃ (FloatModel.layerAct (c * kH * kW)
          w₂ β₂ (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
            (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))) q|)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |M.cnnConv1CotF W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label ci hi wi -
        cnnConv1CotR W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label ci hi wi| ≤
      M.cnnConv1CotBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp ∧
    |M.cnnConv1CotF W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label ci hi wi| ≤
      ((c * (2 * h) * (2 * w) : ℕ) : ℝ) * (w₂ * FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅) +
        M.cnnConv1CotBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp := by
  set Z1C := Tensor3.flatten (conv2d W₁ b₁ x₀) with hZ1C
  set Z1CF := Tensor3.flatten (M.convF W₁ b₁ x₀) with hZ1CF
  set X2 := Tensor3.unflatten (relu (c * (2*h) * (2*w)) Z1C) with hX2
  set X2F := Tensor3.unflatten (relu (c * (2*h) * (2*w)) Z1CF) with hX2F
  set A1 := FloatModel.layerAct (ic * kH * kW) w₁ β₁ a with hA1
  set E1 := FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 with hE1
  have hA1nn : 0 ≤ A1 := layerAct_nonneg hw₁ hβ₁ ha
  have hE1nn : 0 ≤ E1 := layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl
  -- conv-1 forward closeness, conv-2 input closeness + magnitude
  have hZ1close : ∀ k, |Z1CF k - Z1C k| ≤ E1 := by
    intro k; obtain ⟨ci, hi, wi, rfl⟩ := t3Idx_surj k
    rw [hZ1CF, hZ1C, flatten_t3Idx, flatten_t3Idx]
    exact (M.convF_close W₁ b₁ x₀ x₀ le_rfl
        (fun _ _ _ => by simp) ci hi wi).trans
      (M.denseErr_le_uniform hw₁ le_rfl (fun i j => convKernelMat_abs_le hW₁ i j)
        hb₁ (fun idx => convWindow_abs_le ha hx₀ hi wi idx) ci)
  have hX2close : ∀ co i j, |X2F co i j - X2 co i j| ≤ E1 := by
    intro co i j; rw [hX2F, hX2, unflatten_t3Idx, unflatten_t3Idx]
    exact relu_close _ _ _ hZ1close (t3Idx co i j)
  have hX2mag : ∀ co i j, |X2 co i j| ≤ A1 := by
    intro co i j; rw [hX2, unflatten_t3Idx]
    refine (relu_abs_le _ _).trans ?_
    rw [hZ1C, flatten_t3Idx]; exact conv2d_abs_le ha hW₁ hb₁ hx₀ co i j
  -- conv-2 cotangent closeness / magnitudes (factored, at the float conv-2 input)
  have hc2close := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    cnn_conv2_cot_close M X2 X2F W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label fexp hA1nn hE1nn
      hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 heexp1 hfexp hρ1 hX2close hX2mag
      hW₂ hb₂ hW₃ hb₃ hW₄ hb₄ hW₅ hb₅ hmargin2 hmarginPool hmargin3 hmargin4 co ho wo
  have hc2realmag := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    cnn_conv2_cot_real_abs_le X2 W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label hw₃ hw₄ hw₅
      hW₃ hW₄ hW₅ co ho wo
  have hc2floatmag := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    abs_le_of_close (hc2close co ho wo) (hc2realmag co ho wo)
  have hback := convTap_back_close M W₂ _ _ hw₂ hW₂ hc2floatmag hc2close ci hi wi
  unfold FloatModel.cnnConv1CotF cnnConv1CotR FloatModel.cnnConv1CotBudget
  refine ⟨mask_scalar_close (hZ1close _) (hmargin1 _) hback, ?_⟩
  rw [abs_mul]
  refine le_trans (mul_le_mul (by split_ifs <;> simp) ?_ (abs_nonneg _)
    zero_le_one) (le_of_eq (one_mul _))
  exact abs_le_of_close hback (convTap_back_abs_le W₂ _ hw₂ hW₂ hc2realmag ci hi wi)

open FloatModel in
/-- **The binary32 conv-1 weight gradient is within an explicit budget of the
    certified one** — the conv-1 peer of
    `cnn_conv2_grad_close`, one conv-backward deeper. With `x₀` exact, the
    FloatModel `W₁` gradient `M.cnnConv1FloatGrad …` stays within
    `cnnConv1GradBudget`. The conv-2 cotangent chain is reused at a FLOAT conv-2
    input `relu(z̃₁)` (`cnn_conv2_cot_close`); the conv-2 backward is a rounded
    dot of the (exact) `convTap` slab against the float conv-2 cotangent slab
    (`dot_perturbed_close` over `c·(2h)·(2w)`); the conv-1 ReLU mask freezes
    (`mask_scalar_close`); the conv-1 weight dot rounds it
    (`dot_perturbed_close` over `(2h)·(2w)`). Five quantitative margins are
    carried; the bridge `cnn_conv1_loss_gradAt_reluMask` turns the `gradAt`
    into the dot. -/
theorem cnn_conv1_grad_close {ic c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄)
    (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    (u : Vec (c * ic * kH * kW))
    {a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (ha : 0 ≤ a) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅)
    (hβ₅ : 0 ≤ β₅) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx₀ : ∀ ci i j, |x₀ ci i j| ≤ a)
    (hu1 : ∀ idx, |u idx| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmargin1 : ∀ k, FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 <
      |Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k|)
    (hmargin2 : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀))))) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
          (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃ (FloatModel.layerAct (c * kH * kW)
          w₂ β₂ (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
            (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q|)
    (o : Fin c) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    |M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label u
        (k4Idx o cc kh kw) -
      gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label) u (k4Idx o cc kh kw)|
      ≤ M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
          w₅ β₅ eexp := by
  have hu2' : ∀ o' c' kh' kw', |Kernel4.unflatten u o' c' kh' kw'| ≤ w₁ :=
    fun o' c' kh' kw' => by rw [unflatten_k4Idx]; exact hu1 _
  -- off-kink + smooth conditions from the quantitative margins
  have hz1 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k ≠ 0 :=
    fun k => abs_pos.mp (lt_of_le_of_lt
      (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl) (hmargin1 k))
  have hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten u) b₁ x₀))))) k ≠ 0 :=
    fun k => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₂ hβ₂
      (layerAct_nonneg hw₁ hβ₁ ha)
      (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl)) (hmargin2 k))
  have hmp := hmarginPool.smooth (layerBudget_nonneg M.u_nonneg hw₂ hβ₂
    (layerAct_nonneg hw₁ hβ₁ ha)
    (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl))
  have hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l ≠ 0 :=
    fun l => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₃ hβ₃
      (layerAct_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha))
      (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha)
        (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl))) (hmargin3 l))
  have hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))))))) ) q ≠ 0 :=
    fun q => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₄ hβ₄
      (layerAct_nonneg hw₃ hβ₃ (layerAct_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha)))
      (layerBudget_nonneg M.u_nonneg hw₃ hβ₃
        (layerAct_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha))
        (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha)
          (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl)))) (hmargin4 q))
  rw [M.cnnConv1FloatGrad_apply b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label u
      o cc kh kw,
    cnn_conv1_loss_gradAt_reluMask b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label u
      hz1 hz2 hmp hz3 hz4 o cc kh kw]
  simp only [FloatModel.cnnConv1GradBudget]
  have hcot := cnn_conv1_cot_close M (Kernel4.unflatten u) b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label
    fexp ha hw₁ hβ₁ hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 heexp1 hfexp hρ1 hx₀ hu2' hb₁ hW₂ hb₂
    hW₃ hb₃ hW₄ hb₄ hW₅ hb₅ hmargin1 hmargin2 hmarginPool hmargin3 hmargin4
  refine M.dot_perturbed_close (convPadWin kH kW x₀ cc kh kw) _ _ ha
    (fun s => by simp only [convPadWin]; exact abs_convPad_le x₀ ha hx₀ cc kh kw _ _)
    (fun s => by exact (hcot o _ _).2) (fun s => by exact (hcot o _ _).1)

-- ════════════════════════════════════════════════════════════════
-- § The conv1 float rung
-- ════════════════════════════════════════════════════════════════

open FloatModel in
/-- **One SGD step with the FloatModel binary32 conv-1 kernel gradient decreases
    one example's cross-entropy loss; the gradient's accuracy is proven, not
    assumed.** The conv-1 peer of `cnn_conv2_float_sgd_descends`: the gradient is
    the FloatModel binary32 `W₁` gradient `M.cnnConv1FloatGrad …`, accuracy *proven* by `cnn_conv1_grad_close`
    (η := `cnnConv1GradBudget`, discharged per kernel entry via `k4Idx_surj`),
    wired into the abstract `cnn_conv1_sgd_descends`. Five per-layer ROUND
    margins feed the grad-close; the gradient-radius STEP margins +
    `hsmall`/`h1`/`h2` feed the drift-freeze and descent geometry. Both conv
    kernels of the Chapter-3 CNN now have a float-gradient descent statement.

    Scope: one example, `W₁` moving with every other parameter fixed, and the
    update taken in ℝ — only the gradient is float-modelled. -/
theorem cnn_conv1_float_sgd_descends {ic c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    {lr a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (ha : 0 ≤ a) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅)
    (hβ₅ : 0 ≤ β₅) (hlr : 0 ≤ lr) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx₀ : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hr1 : ∀ k, FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 <
      |Tensor3.flatten (conv2d W₁ b₁ x₀) k|)
    (hr2 : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))) k|)
    (hrPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hr3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
          (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))) l|)
    (hr4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃ (FloatModel.layerAct (c * kH * kW)
          w₂ β₂ (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
            (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))) q|)
    (hm1 : ∀ k, a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)) < |(Tensor3.flatten (conv2d W₁ b₁ x₀)) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))) < |(Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))) k|)
    (hmq : MaxPool2MarginQ (((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))))) (Tensor3.unflatten (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu
                (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))))) < |(dense W₃ b₃ (maxPoolFlat c h w (relu
              (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁ b₁ x₀))))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))))))) < |(dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat
              c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
                (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁ b₁ x₀))))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) :
        ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))))))))))) < 1)
    (h1 : lr * (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃
          w₄ β₄ w₅ β₅ eexp) * (∑ idx,
              |gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              (Kernel4.flatten W₁) idx|) ≤
      lr * (∑ idx, (gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              (Kernel4.flatten W₁)) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2
        * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) :
          ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))))))))))))) * (stepRadius (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) lr (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)) ^ 2 ≤
      lr * (∑ idx, (gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              (Kernel4.flatten W₁)) idx ^ 2) / 4) :
    (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁ -
              lr • M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
                (Kernel4.flatten W₁)) ≤
      (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) (Kernel4.flatten W₁) -
        lr * (∑ idx, (gradAt (cnnConv1KernelLoss b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              (Kernel4.flatten W₁)) idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  unfold cnnConv1KernelLoss at *
  have hu := M.u_nonneg
  -- nonnegativity of the proven budget
  have hA1nn : 0 ≤ FloatModel.layerAct (ic * kH * kW) w₁ β₁ a :=
    layerAct_nonneg hw₁ hβ₁ ha
  have hE1nn : 0 ≤ FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 :=
    layerBudget_nonneg hu hw₁ hβ₁ ha le_rfl
  have hCPnn : 0 ≤ FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ :=
    FloatModel.cnnConv2CotMag_nonneg hw₃ hw₄ hw₅
  have he2nn : 0 ≤ M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW
      (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
      (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)
      w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp :=
    M.cnnConv2CotBudget_nonneg hA1nn hE1nn hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 hρ1
  have hη0 : 0 ≤ M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃
      w₄ β₄ w₅ β₅ eexp := by
    simp only [FloatModel.cnnConv1GradBudget]
    have hγ : ∀ m : ℕ, (0:ℝ) ≤ (1 + M.u) ^ (m + 1) - 1 :=
      fun m => sub_nonneg.mpr (one_le_pow₀ (by linarith))
    have hn : ∀ m : ℕ, (0:ℝ) ≤ ((m : ℕ) : ℝ) := fun m => Nat.cast_nonneg _
    have hebacknn : (0:ℝ) ≤ ((1 + M.u) ^ ((c * (2*h) * (2*w)) + 1) - 1) *
          (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ *
            (FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ +
              M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW
                (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
                (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)
                w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))) +
          (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * M.cnnConv2CotBudget c h w d₃ d₄
            nC kH kW (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
            (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)
            w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)) :=
      add_nonneg (mul_nonneg (hγ _) (mul_nonneg (hn _) (mul_nonneg hw₂
        (add_nonneg hCPnn he2nn)))) (mul_nonneg (hn _) (mul_nonneg hw₂ he2nn))
    exact add_nonneg (mul_nonneg (hγ _) (mul_nonneg (hn _) (mul_nonneg ha
      (add_nonneg (mul_nonneg (hn _) (mul_nonneg hw₂ hCPnn)) hebacknn))))
      (mul_nonneg (hn _) (mul_nonneg ha hebacknn))
  -- the flattened conv-1 kernel inherits the per-entry bound
  have huf : ∀ idx, |Kernel4.flatten W₁ idx| ≤ w₁ := by
    intro idx
    obtain ⟨o', c', kh', kw', rfl⟩ := k4Idx_surj idx
    rw [flatten_k4Idx]; exact hW₁ o' c' kh' kw'
  -- discharge the abstract gradient accuracy by the proven grad-close
  have hgh : ∀ idx, |M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
      (Kernel4.flatten W₁) idx -
      gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label) (Kernel4.flatten W₁) idx| ≤
      M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅
        β₅ eexp := by
    intro idx
    obtain ⟨o', c', kh', kw', rfl⟩ := k4Idx_surj idx
    exact cnn_conv1_grad_close M b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label fexp
      (Kernel4.flatten W₁) ha hw₁ hβ₁ hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅
      heexp0 heexp1 hfexp hρ1 hx₀ huf hb₁ hW₂ hb₂ hW₃ hb₃ hW₄ hb₄ hW₅ hb₅
      (fun k => by rw [Kernel4.unflatten_flatten]; exact hr1 k)
      (fun k => by rw [Kernel4.unflatten_flatten]; exact hr2 k)
      (by rw [Kernel4.unflatten_flatten]; exact hrPool)
      (fun l => by rw [Kernel4.unflatten_flatten]; exact hr3 l)
      (fun q => by rw [Kernel4.unflatten_flatten]; exact hr4 q)
      o' c' kh' kw'
  exact cnn_conv1_sgd_descends W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label
    (M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
      (Kernel4.flatten W₁))
    (fun _ _ => False) ha hx₀ (fun _ _ h => h.elim) hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ hlr hη0 hgh
    hm1 hm2 (MaxPool2MarginQ.to_marginQUpTo_flat _ hmq) hm3 hm4 hsmall h1 h2

-- ════════════════════════════════════════════════════════════════
-- § The conv BIASES (the last descent rung)
--
-- The bias gradient is the spatial SUM of the conv-output cotangent
-- (`convBiasGrad_eq_sum`), the Kronecker channel-indicator Jacobian collapsing the
-- `∑ ci` to `ci = o`. The cotangent chains are reused wholesale from the
-- conv-WEIGHT rungs: the one new core is `sum_perturbed_close`, the `M.sum` peer
-- of `dot_perturbed_close`.
-- ════════════════════════════════════════════════════════════════

/-- **The binary32 conv-2 bias gradient (FloatModel transcription of the
    per-example gradient)** — the bias peer of `cnnConv2FloatGrad`: at output channel `o` it is the float SUM
    `M.sum (cotWin c̃Conv o)` of the same float conv-2-output cotangent slab
    (the bias Jacobian is the channel indicator, so there is no `convPadWin`
    left operand and no per-slot kernel index — one entry per channel). -/
noncomputable def FloatModel.cnnConv2BiasFloatGrad {c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC) : Vec c :=
  fun o =>
    M.sum (cotWin (fun ci hi wi =>
      (if Tensor3.flatten (M.convF W₂ b₂ x₁) (t3Idx ci hi wi) > 0
          then (1:ℝ) else 0) *
        (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (M.convF W₂ b₂ x₁)))) ci hi wi
          then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
            (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten (M.convF W₂ b₂ x₁)))))
              (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                    (M.dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                      (Tensor3.flatten (M.convF W₂ b₂ x₁)))))))
                  (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                    (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                        (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                          (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                            (Tensor3.flatten (M.convF W₂ b₂ x₁))))))))) label)))))
            (t3Idx ci (winRow hi) (winCol wi))
          else 0)) o)

/-- **The conv-2 bias-gradient grad-close budget** — `cnnConv2GradBudget` with
    the `a·` input factor stripped (the bias Jacobian carries no input window):
    the spatial sum's Higham γ over `(2h)·(2w)` against the float-cotangent
    magnitude (`cnnConv2CotMag + cnnConv2CotBudget`) plus the per-entry
    cotangent drift `cnnConv2CotBudget`. The cotangent chain is the exact
    conv-2-input (`aX2 = a`, `eX2 = 0`) instance of the factored
    `cnnConv2CotBudget` / `cnnConv2CotMag`. -/
noncomputable def FloatModel.cnnConv2BiasGradBudget (M : FloatModel)
    (c h w d₃ d₄ nC kH kW : ℕ) (a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ) : ℝ :=
  let CotMag := FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅
  let CotBudget := M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW a 0 w₂ β₂ w₃ β₃ w₄ β₄
    w₅ β₅ eexp
  ((1 + M.u) ^ ((2 * h) * (2 * w) + 1) - 1) *
      (((2 * h) * (2 * w) : ℕ) * (CotMag + CotBudget)) +
    (((2 * h) * (2 * w) : ℕ) * CotBudget)

open FloatModel in
/-- **The binary32 conv-2 BIAS gradient is within an explicit budget of the
    certified one** — the bias peer of `cnn_conv2_grad_close`, built on the
    factored conv-2 cotangent chain at the exact conv-2 input `x₁`
    (`cnn_conv2_cot_close` with `aX2 = a`, `eX2 = 0`) and the spatial-SUM core
    `sum_perturbed_close`. The bridge `cnn_conv2_bias_loss_gradAt_reluMask`
    turns the `gradAt` into the sum the float bias gradient rounds. Four
    quantitative margins (conv-output, pool POST-relu, z̃₃, z̃₄) freeze the
    routing. -/
theorem cnn_conv2_bias_grad_close {c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    {a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (ha : 0 ≤ a) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂) (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃)
    (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅) (hβ₅ : 0 ≤ β₅)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx₁ : ∀ ci i j, |x₁ ci i j| ≤ a)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmarginConv : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0 <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (o : Fin c) :
    |M.cnnConv2BiasFloatGrad W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label o -
      gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o|
      ≤ M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
          eexp := by
  -- off-kink + smooth conditions from the quantitative margins
  have hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ x₁) k ≠ 0 :=
    fun k => abs_pos.mp (lt_of_le_of_lt
      (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ ha le_rfl) (hmarginConv k))
  have hmp := hmarginPool.smooth (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ ha le_rfl)
  have hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l ≠ 0 :=
    fun l => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₃ hβ₃
      (layerAct_nonneg hw₂ hβ₂ ha)
      (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ ha le_rfl)) (hmargin3 l))
  have hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q ≠ 0 :=
    fun q => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₄ hβ₄
      (layerAct_nonneg hw₃ hβ₃ (layerAct_nonneg hw₂ hβ₂ ha))
      (layerBudget_nonneg M.u_nonneg hw₃ hβ₃ (layerAct_nonneg hw₂ hβ₂ ha)
        (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ ha le_rfl))) (hmargin4 q))
  -- per-cell conv-2 cotangent closeness / magnitudes (exact conv-2 input x₁)
  have hc2close := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    cnn_conv2_cot_close M x₁ x₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label fexp ha (le_refl 0)
      hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 heexp1 hfexp hρ1
      (fun _ _ _ => by simp) hx₁ hW₂ hb₂ hW₃ hb₃ hW₄ hb₄ hW₅ hb₅
      hmarginConv hmarginPool hmargin3 hmargin4 co ho wo
  have hc2realmag := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    cnn_conv2_cot_real_abs_le x₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label hw₃ hw₄ hw₅
      hW₃ hW₄ hW₅ co ho wo
  have hc2floatmag := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    abs_le_of_close (hc2close co ho wo) (hc2realmag co ho wo)
  -- assemble: unfold the float grad, rewrite gradAt to the sum (bridge), apply
  simp only [FloatModel.cnnConv2BiasFloatGrad]
  rw [cnn_conv2_bias_loss_gradAt_reluMask W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label
      b₂ hz2 hmp hz3 hz4 o]
  simp only [FloatModel.cnnConv2BiasGradBudget]
  refine M.sum_perturbed_close _ _
    (fun s => by simp only [cotWin]; exact hc2floatmag o _ _)
    (fun s => by simp only [cotWin]; exact hc2close o _ _)

open FloatModel in
/-- **One SGD step with the FloatModel binary32 conv-2 bias gradient decreases one
    example's cross-entropy loss; the gradient's accuracy is proven, not
    assumed** — the bias peer of `cnn_conv2_float_sgd_descends`: the gradient is
    the FloatModel binary32 bias gradient `M.cnnConv2BiasFloatGrad …`,
    and its accuracy is *proven* by `cnn_conv2_bias_grad_close`
    (η := `cnnConv2BiasGradBudget`, discharged per output channel — the bias IS
    a vector, so no flatten/unflatten plumbing), not assumed. The two
    rounding-margin families are carried as hypotheses, exactly as in the
    weight rungs. Scope: one example, `b₂` moving, update taken in ℝ. -/
theorem cnn_conv2_bias_float_sgd_descends {c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    {lr a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (ha : 0 ≤ a) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂) (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃)
    (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅) (hβ₅ : 0 ≤ β₅) (hlr : 0 ≤ lr)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmarginConv : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0 <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hm2 : ∀ k, stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr (M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp) <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmq : MaxPool2MarginQ (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr (M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr (M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr (M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr (M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))))))) < 1)
    (h1 : lr * (M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄
          w₅ β₅ eexp) * (∑ o, |gradAt
        (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ o|) ≤
      lr * (∑ o, gradAt
        (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ o ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr (M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))))))))) *
        (stepRadius (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ lr (M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)) ^ 2 ≤
      lr * (∑ o, gradAt
        (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label) b₂ o ^ 2) / 4) :
    (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (b₂ - lr • M.cnnConv2BiasFloatGrad W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp
              label) ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁))))))))) label -
        lr * (∑ o, gradAt
          (cnnConv2BiasLoss W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label)
            b₂ o ^ 2) / 2 := by
  simp only [stepRadius] at *
  unfold cnnConv2BiasLoss at *
  have hCotMagnn : 0 ≤ FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ :=
    FloatModel.cnnConv2CotMag_nonneg hw₃ hw₄ hw₅
  have hCotBudnn : 0 ≤ M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW a 0 w₂ β₂ w₃ β₃ w₄
      β₄ w₅ β₅ eexp :=
    M.cnnConv2CotBudget_nonneg ha le_rfl hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 hρ1
  have hη0 : 0 ≤ M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄
      w₅ β₅ eexp := by
    simp only [FloatModel.cnnConv2BiasGradBudget]
    have hγ : (0:ℝ) ≤ (1 + M.u) ^ ((2 * h) * (2 * w) + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
    have hn : (0:ℝ) ≤ (((2 * h) * (2 * w) : ℕ) : ℝ) := Nat.cast_nonneg _
    exact add_nonneg
      (mul_nonneg hγ (mul_nonneg hn (add_nonneg hCotMagnn hCotBudnn)))
      (mul_nonneg hn hCotBudnn)
  -- discharge the abstract gradient accuracy per output channel (bias = vector)
  have hgh : ∀ o, |M.cnnConv2BiasFloatGrad W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label o -
      gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o| ≤
      M.cnnConv2BiasGradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp :=
    fun o => cnn_conv2_bias_grad_close M W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label fexp
      ha hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 heexp1 hfexp hρ1 hx hW₂ hb₂
      hW₃ hb₃ hW₄ hb₄ hW₅ hb₅ hmarginConv hmarginPool hmargin3 hmargin4 o
  exact cnn_conv2_bias_sgd_descends W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label
    (M.cnnConv2BiasFloatGrad W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label)
    (fun _ _ => False) (fun _ _ h => h.elim) hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ hlr hη0 hgh hm2
    (MaxPool2MarginQ.to_marginQUpTo_flat _ hmq) hm3 hm4 hsmall h1 h2

/-- **The binary32 conv-1 bias gradient (FloatModel transcription of the
    per-example gradient)** — the bias peer of `cnnConv1FloatGrad`: at output channel `o` it is
    the float SUM `M.sum (cotWin …)` of the same float conv-1-output cotangent slab
    (`cnnConv1CotF`), with no `convPadWin` left operand. -/
noncomputable def FloatModel.cnnConv1BiasFloatGrad {ic c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC) : Vec c :=
  fun o => M.sum (cotWin (M.cnnConv1CotF W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label) o)

/-- **The conv-1 bias-gradient grad-close budget** — `cnnConv1GradBudget` with
    the `a·` input factors stripped (the bias Jacobian carries no input window):
    the spatial sum's Higham γ over `(2h)·(2w)` against the float conv-1
    cotangent magnitude `C1t = c(2h)(2w)·w₂·CP + eback` plus the per-entry drift
    `eback = cnnConv1CotBudget`. -/
noncomputable def FloatModel.cnnConv1BiasGradBudget (M : FloatModel)
    (ic c h w d₃ d₄ nC kH kW : ℕ)
    (a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ) : ℝ :=
  let CP := FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅
  let eback := M.cnnConv1CotBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp
  let C1t := ((c * (2 * h) * (2 * w) : ℕ) : ℝ) * (w₂ * CP) + eback
  ((1 + M.u) ^ ((2 * h) * (2 * w) + 1) - 1) *
      (((2 * h) * (2 * w) : ℕ) * C1t) +
    (((2 * h) * (2 * w) : ℕ) * eback)

open FloatModel in
/-- **The binary32 conv-1 BIAS gradient is within an explicit budget of the
    certified one** — the bias peer of `cnn_conv1_grad_close`, built on the
    factored conv-2 cotangent chain at the FLOAT conv-2 input `relu(z̃₁)`
    (`cnn_conv2_cot_close`), the conv-2 backward `convTap_back_close`, the
    conv-1 ReLU-mask freeze (`mask_scalar_close`), and the spatial-SUM core
    `sum_perturbed_close`. Five quantitative margins freeze the routing; the
    bridge `cnn_conv1_bias_loss_gradAt_reluMask` turns the `gradAt` into the
    sum. -/
theorem cnn_conv1_bias_grad_close {ic c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (fexp : ℝ → ℝ)
    {a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (ha : 0 ≤ a) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅)
    (hβ₅ : 0 ≤ β₅) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx₀ : ∀ ci i j, |x₀ ci i j| ≤ a)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmargin1 : ∀ k, FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 <
      |Tensor3.flatten (conv2d W₁ b₁ x₀) k|)
    (hmargin2 : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
          (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃ (FloatModel.layerAct (c * kH * kW)
          w₂ β₂ (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
            (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))) q|)
    (o : Fin c) :
    |M.cnnConv1BiasFloatGrad W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label o -
      gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b' x₀)))))))))))))
          label) b₁ o|
      ≤ M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
          w₅ β₅ eexp := by
  have hz1 : ∀ k, Tensor3.flatten (conv2d W₁ b₁ x₀) k ≠ 0 :=
    fun k => abs_pos.mp (lt_of_le_of_lt
      (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl) (hmargin1 k))
  have hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))) k ≠ 0 :=
    fun k => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₂ hβ₂
      (layerAct_nonneg hw₁ hβ₁ ha)
      (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl)) (hmargin2 k))
  have hmp := hmarginPool.smooth (layerBudget_nonneg M.u_nonneg hw₂ hβ₂
    (layerAct_nonneg hw₁ hβ₁ ha)
    (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl))
  have hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))) l ≠ 0 :=
    fun l => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₃ hβ₃
      (layerAct_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha))
      (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha)
        (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl))) (hmargin3 l))
  have hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))) ) q ≠ 0 :=
    fun q => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₄ hβ₄
      (layerAct_nonneg hw₃ hβ₃ (layerAct_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha)))
      (layerBudget_nonneg M.u_nonneg hw₃ hβ₃
        (layerAct_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha))
        (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha)
          (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl)))) (hmargin4 q))
  simp only [FloatModel.cnnConv1BiasFloatGrad]
  rw [cnn_conv1_bias_loss_gradAt_reluMask W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label
      b₁ hz1 hz2 hmp hz3 hz4 o]
  simp only [FloatModel.cnnConv1BiasGradBudget]
  have hcot := cnn_conv1_cot_close M W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label fexp ha hw₁ hβ₁
    hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 heexp1 hfexp hρ1 hx₀ hW₁ hb₁ hW₂ hb₂ hW₃ hb₃ hW₄ hb₄
    hW₅ hb₅ hmargin1 hmargin2 hmarginPool hmargin3 hmargin4
  exact M.sum_perturbed_close _ _ (fun s => by exact (hcot o _ _).2)
    (fun s => by exact (hcot o _ _).1)

open FloatModel in
/-- **One SGD step with the FloatModel binary32 conv-1 bias gradient decreases one
    example's cross-entropy loss; the gradient's accuracy is proven, not
    assumed** — the bias peer of `cnn_conv1_float_sgd_descends`, the deepest
    descent rung: the gradient is the FloatModel binary32 bias gradient
    `M.cnnConv1BiasFloatGrad …`, accuracy *proven* by `cnn_conv1_bias_grad_close`
    (η := `cnnConv1BiasGradBudget`, discharged per output channel — the bias IS a
    vector), not assumed. With this, both conv kernels and both conv biases of
    the Chapter-3 CNN have a float-gradient descent statement. Scope: one
    example, `b₁` moving, update taken in ℝ. -/
theorem cnn_conv1_bias_float_sgd_descends {ic c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    {lr a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (hc : 0 < c)
    (ha : 0 ≤ a) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅)
    (hβ₅ : 0 ≤ β₅) (hlr : 0 ≤ lr)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmargin1 : ∀ k, FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 <
      |Tensor3.flatten (conv2d W₁ b₁ x₀) k|)
    (hmargin2 : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
          (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃ (FloatModel.layerAct (c * kH * kW)
          w₂ β₂ (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
            (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))) q|)
    (hm1 : ∀ k, lr * (((∑ idx, |gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * (M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))) < |(Tensor3.flatten (conv2d W₁ b₁ x₀)) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * (M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))))) < |(Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))) k|)
    (hmq : MaxPool2MarginQ (((c * kH * kW : ℕ) : ℝ) * (w₂ * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * (M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))))) (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (lr *
      (((∑ idx, |gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * (M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))))))) < |(dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h *
      (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * (M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))))))))
      < |(dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) * (M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)))))))))))) < 1)
    (h1 : lr * (M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
      eexp) * (∑ idx, |gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) ≤
      lr * (∑ idx, (gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 *
      (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 / (1 - 2 * (w₅ *
      ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h *
      (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt
      (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁ idx|) + (c : ℝ) *
                (M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
                eexp)))))))))))))) * (stepRadius (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label) b₁ lr (M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)) ^ 2 ≤
      lr * (∑ idx, (gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁) idx ^ 2) / 4) :
    (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
      (b₁ - lr • M.cnnConv1BiasFloatGrad W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label) ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))))))) label -
        lr * (∑ idx, (gradAt (cnnConv1BiasLoss W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label)
              b₁) idx ^ 2) / 2 := by
  simp only [stepRadius] at *
  unfold cnnConv1BiasLoss at *
  have hgh : ∀ idx, |M.cnnConv1BiasFloatGrad W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp
      label idx -
      gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b' x₀)))))))))))))
          label) b₁ idx| ≤
      M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
        w₅ β₅ eexp :=
    fun idx => cnn_conv1_bias_grad_close M W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label
      fexp ha hw₁ hβ₁ hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 heexp1 hfexp hρ1
      hx hW₁ hb₁ hW₂ hb₂ hW₃ hb₃ hW₄ hb₄ hW₅ hb₅
      hmargin1 hmargin2 hmarginPool hmargin3 hmargin4 idx
  have hη0 : 0 ≤ M.cnnConv1BiasGradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃
      w₄ β₄ w₅ β₅ eexp := le_trans (abs_nonneg _) (hgh ⟨0, hc⟩)
  exact cnn_conv1_bias_sgd_descends W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label
    (M.cnnConv1BiasFloatGrad W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label)
    (fun _ _ => False) (fun _ _ h => h.elim) hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ hlr hη0 hgh
    hm1 hm2 (MaxPool2MarginQ.to_marginQUpTo_flat _ hmq) hm3 hm4 hsmall h1 h2

end Proofs
