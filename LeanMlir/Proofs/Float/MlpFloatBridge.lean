import LeanMlir.Proofs.Float.FloatBridge

/-! # The MNIST MLP and E4M3 linear-net float chains

`FloatBridge`'s per-op bounds, assembled for two nets: the MNIST MLP (784→512→512→10), as
the forward (`mlp_float_close`, `mlp_float_close_uniform`), its binary32 numeric instance
(`mnist_mlp_float_budget`), the six float SGD steps (`mlp_{w,b}{0,1,2}_step_float_close`) with the
`W₂` instance (`mnist_w2_step_float_budget`) and the loss-head instance (`mnist_cot_budget`); and
the depth-1 fp8 linear net (`linear_e4m3_logit_budget`, `linear_e4m3_argmax_preserved`). Each
theorem is stated over any `FloatModel`; `Binary32Instance` instantiates the E4M3 one. The
ResNet-34 peer is `ResNet34FloatBridge`.
-/

namespace Proofs
namespace FloatModel

variable (M : FloatModel)

/-- Rounded MLP forward — the float peer of the
    `dense W₂ b₂ ∘ relu ∘ dense W₁ b₁ ∘ relu ∘ dense W₀ b₀` composition
    (`MlpTrainStep.lean`). `relu` appears bare: max-with-0 is exact in
    floating point. -/
noncomputable def mlpF {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) : Vec d₃ :=
  M.dense W₂ b₂ (relu d₂ (M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x))))

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

    Measured on the live run (`scripts/certs/margin_probe.py`): actual logit
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

/-- **The MLP's first two rounded layers** from a fresh input: layer 0's pre-activation (and
    its ReLU) within `E₀ = layerBudget … a 0`, the real layer-0 activation within
    `A₁ = layerAct d₀ w₀ β₀ a`, and layer 1's pre-activation within
    `layerBudget … A₁ E₀` — the forward prefix every step capstone below starts from. -/
theorem mlp_l1_close {d₀ d₁ d₂ : Nat} {W₀ : Mat d₀ d₁} {b₀ : Vec d₁} {W₁ : Mat d₁ d₂}
    {b₁ : Vec d₂} {x : Vec d₀} {w₀ β₀ w₁ β₁ a : ℝ}
    (hw₀ : 0 ≤ w₀) (hβ₀ : 0 ≤ β₀) (hw₁ : 0 ≤ w₁) (ha : 0 ≤ a)
    (hW₀ : ∀ i j, |W₀ i j| ≤ w₀) (hb₀ : ∀ j, |b₀ j| ≤ β₀)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w₁) (hb₁ : ∀ j, |b₁ j| ≤ β₁) (hx : ∀ i, |x i| ≤ a) :
    (∀ j, |M.dense W₀ b₀ x j - Proofs.dense W₀ b₀ x j| ≤ layerBudget M.u d₀ w₀ β₀ a 0) ∧
    (∀ j, |relu d₁ (M.dense W₀ b₀ x) j - relu d₁ (Proofs.dense W₀ b₀ x) j| ≤
      layerBudget M.u d₀ w₀ β₀ a 0) ∧
    (∀ i, |relu d₁ (Proofs.dense W₀ b₀ x) i| ≤ layerAct d₀ w₀ β₀ a) ∧
    (∀ j, |M.dense W₁ b₁ (relu d₁ (M.dense W₀ b₀ x)) j -
      Proofs.dense W₁ b₁ (relu d₁ (Proofs.dense W₀ b₀ x)) j| ≤
      layerBudget M.u d₁ w₁ β₁ (layerAct d₀ w₀ β₀ a) (layerBudget M.u d₀ w₀ β₀ a 0)) := by
  have hE₀0 := layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha le_rfl (m := d₀)
  have l0 : ∀ j, |M.dense W₀ b₀ x j - Proofs.dense W₀ b₀ x j| ≤
      layerBudget M.u d₀ w₀ β₀ a 0 := fun j =>
    (M.dense_close_fresh W₀ b₀ x j).trans (M.denseErr_le_uniform hw₀ le_rfl hW₀ hb₀ hx j)
  have r0 := fun j => relu_close _ _ _ l0 j
  have ha₁ : ∀ i, |relu d₁ (Proofs.dense W₀ b₀ x) i| ≤ layerAct d₀ w₀ β₀ a :=
    fun i => (relu_abs_le _ i).trans (dense_abs_le ha hW₀ hb₀ hx i)
  exact ⟨l0, r0, ha₁, fun j => (M.dense_close W₁ b₁ _ _ _ hE₀0 r0 j).trans
    (M.denseErr_le_uniform hw₁ hE₀0 hW₁ hb₁ ha₁ j)⟩

/-- **Rounded output-layer weight update (W₂).** The float update
    `fl(W₂ᵢⱼ − fl(lr·fl(ã₂ᵢ·gtⱼ)))` — outer-product gradient from the *stored
    float forward activation*, as the rendered trainer computes it — is
    within an explicit budget of the real step `W₂ᵢⱼ − lr·(a₂ᵢ·gⱼ)`. The real
    target is `Mat.outer a₂ g i j = emitWeightGrad`'s entry, the quantity
    `mlp_layer2_weight_grad_bridge` proves equal to the pdiv-Jacobian contraction —
    so this chains the float step to the certified gradient. Takes the output
    cotangent `gt ≈ g` as a hypothesis; `softmax_ce_cot_close` discharges it
    with `eg := cotErr u eexp δ n`. -/
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
  obtain ⟨l0, r0, ha₁, l1⟩ := M.mlp_l1_close hw₀ hβ₀ hw₁ ha hW₀ hb₀ hW₁ hb₁ hx
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
    `mlp_layer1_weight_grad_bridge` certifies. `W₀`/`b₁`/`b₀` are the same
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
  obtain ⟨l0, r0, ha₁, l1⟩ := M.mlp_l1_close hw₀ hβ₀ hw₁ ha hW₀ hb₀ hW₁ hb₁ hx
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
  obtain ⟨l0, r0, ha₁, l1⟩ := M.mlp_l1_close hw₀ hβ₀ hw₁ ha hW₀ hb₀ hW₁ hb₁ hx
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
    real target `W₀ᵢⱼ − lr·(xᵢ·c₀ⱼ)` is the certified layer-0 step
    (`mlp_layer0_weight_grad_bridge`). -/
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
  obtain ⟨l0, r0, ha₁, l1⟩ := M.mlp_l1_close hw₀ hβ₀ hw₁ ha hW₀ hb₀ hW₁ hb₁ hx
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
  obtain ⟨l0, r0, ha₁, l1⟩ := M.mlp_l1_close hw₀ hβ₀ hw₁ ha hW₀ hb₀ hW₁ hb₁ hx
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
    live run (`scripts/certs/margin_probe.py`): actual W₂ step deviation
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

/-- **Numeric head budget at the committed MNIST output** (`n = 10`): for
    any model at binary32 accuracy, `exp` accurate to `eexp ≤ 10⁻⁶`
    (an assumed accuracy for GPU `exp`, not measured in the repo),
    and float logits within `δ = 1/100` of real, the rounded
    softmax−onehot cotangent is within **21/1000** of the certified
    gradient — almost all of it the `e^(2δ) − 1 ≈ 2δ` logit-perturbation
    term; the head's own rounding contributes < 4·10⁻⁶.

    `δ = 1/100` is an a-posteriori-style hypothesis: the *worst-case*
    forward logit budget (≈5100 at trained magnitudes) makes `e^(2δ) − 1`
    vacuous, so a useful head budget needs the measured logit error —
    exactly the hand-off point from worst-case to a-posteriori analysis.
    Empirically validated (`scripts/certs/margin_probe.py`): measured drift on a
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
    linarith [mul_le_mul hg11 (by linarith : 1 + eexp ≤ 1 + 1/1000000)
      (by linarith : (0:ℝ) ≤ 1 + eexp) (by norm_num : (0:ℝ) ≤ 7/10000000)]
  have hρ0 : 0 ≤ smRho M.u eexp 10 := M.smRho_nonneg heexp0
  have hρ1 : smRho M.u eexp 10 < 1 := lt_of_le_of_lt hρ (by norm_num)
  have hκ : smKappa M.u eexp 10 ≤ 3/1000000 := by
    simp only [smKappa]
    rw [div_le_iff₀ (by linarith)]
    linarith
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
    linarith [mul_nonneg hu (by linarith : (0:ℝ) ≤ 1 + smKappa M.u eexp 10)]
  refine (M.softmax_ce_cot_close fexp zt z label heexp0 (by linarith) hfexp
    hρ1 hz k).trans ?_
  simp only [cotErr]
  have h1 : M.u * (1 + smErr M.u eexp (1/100) 10) ≤
      (1/16777216) * (1 + 41/2000) :=
    mul_le_mul hu32 (by linarith) (by linarith) (by norm_num)
  have h2 : (1/16777216 : ℝ) * (1 + 41/2000) + 41/2000 ≤ 21/1000 := by
    norm_num
  linarith

/-- **The worst-case E4M3 per-logit budget at the MNIST-linear dims** (784→n;
    E4M3 leaf `u_leaf ≤ 2⁻⁴`, fp32 accumulate `u_acc ≤ 2⁻²⁴`; pixels `|x| ≤ 1`,
    trained `|W| ≤ 3/5`, `|b| ≤ 1`): every E4M3-mixed logit is within **61** of
    the exact-ℝ logit. The leaf term `(2·2⁻⁴ ≈ 12.5%)·∑|xW|` dominates (the fp32
    fan-in γ at 784 is ≈5·10⁻⁵, negligible) — this is the *worst-case*, all-errors-
    aligned figure. The demo (`scripts/demos/mnist_e4m3_demo.py`) measures the actual
    drift at `max|Δlogit| = 0.38` (errors cancel), the a-posteriori `B`; both
    feed `argmax_preserved`. -/
theorem linear_e4m3_logit_budget (L : FloatModel) (hMu : M.u ≤ u32)
    (hLu : L.u ≤ uE4M3) :
    denseMixedBudget M.u L.u 784 (3 / 5) 1 1 ≤ 61 := by
  have hu := M.u_nonneg
  have hLu0 := L.u_nonneg
  have hue : (uE4M3 : ℝ) = 1 / 16 := by norm_num [uE4M3]
  rw [hue] at hLu
  -- a clean coarse accumulate bound keeps the assembly out of 2⁻²⁴-land
  have hu6 : M.u ≤ 1 / 1000000 := hMu.trans (by norm_num [u32])
  -- the two flat E4M3 leaf pieces at u_leaf ≤ 1/16 (prove these BEFORE hγ:
  -- linarith ring-normalizes every in-scope hypothesis, so a concrete
  -- `(1+M.u)^785` in context would blow up the 785-fold npow)
  have hprodhint : (0 : ℝ) ≤ L.u * (1 / 16 - L.u) := mul_nonneg hLu0 (by linarith)
  have hsq : (1 + L.u) ^ 2 ≤ 289 / 256 := by linarith [hLu, hLu0, hprodhint]
  have hleaf : 2 * L.u + L.u ^ 2 ≤ 33 / 256 := by linarith [hLu, hLu0, hprodhint]
  -- the fan-in γ at 784 (cheap via gamma_num; no big-power evaluation)
  have hγ : (1 + M.u) ^ (784 + 1) - 1 ≤ 5 / 100000 :=
    M.gamma_num (k := 784 + 1) hMu (by norm_num [u32]) (by norm_num [u32])
  refine (denseMixedBudget_le_of (U := 1 / 1000000) (g := 5 / 100000)
      (P := 289 / 256) (Q := 33 / 256) (by norm_num) hu hLu0 hu6
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) hγ
      hsq hleaf).trans ?_
  norm_num

/-- **Verified E4M3 MNIST-linear argmax preservation.**
    For the certified linear classifier at E4M3 leaf precision / fp32 accumulate,
    pixels `|x| ≤ 1`, trained `|W| ≤ 3/5`, `|b| ≤ 1`: whenever the exact-ℝ logit
    margin at the top class `k` exceeds `2·61 = 122`, the E4M3-mixed forward keeps
    `k` as the strict argmax — **provably the same prediction**. Depth-1 makes
    the single-matmul bound the end-to-end bound, so this is the one realistic
    fp8 case with an honest accuracy guarantee (no vacuous depth compounding).
    The 122 is the worst-case threshold; with the demo's measured `B = 0.38`
    the same `argmax_preserved` covers the `>0.76`-margin inputs — empirically
    92.89% of the MNIST test set (`scripts/demos/mnist_e4m3_demo.py`). fp32 ≈ exact-ℝ
    (within `u_acc`), so the demo's fp32 margins are the relevant quantity. -/
theorem linear_e4m3_argmax_preserved (L : FloatModel) (hMu : M.u ≤ u32)
    (hLu : L.u ≤ uE4M3) {n : ℕ} {W : Mat 784 n} {b : Vec n} {x : Vec 784}
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

end FloatModel
end Proofs
