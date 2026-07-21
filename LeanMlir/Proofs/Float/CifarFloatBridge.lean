import LeanMlir.Proofs.Training.SgdDescentCnn
import LeanMlir.Proofs.Architectures.CifarCNN

/-!
# ℝ→Float32 forward rounding budget for the Chapter-4 CIFAR CNN (no BN)

The Tier-1 `cnn_float_close` (`SgdDescentCnn.lean`) bound scaled from the
MNIST CNN (2 conv + 1 maxpool + 3 dense) to the **no-BN CIFAR CNN**
(`cifarCnnForward`: 4 conv in two `conv→conv→pool` stages + 3 dense). Same
machinery, zero new numerical primitives: each conv threads as a `dense` at
its fan-in via `flatConvF_close`, `relu`/`maxPoolFlat` pass error through
exactly (no rounding, no amplification), and the per-layer budgets nest
through `layerBudget`/`layerAct`.

This is the no-BN first pass of the MNIST→CIFAR bridge step; BatchNorm (the
reciprocal-√ istd) is deliberately out of scope and handled separately.
-/

namespace Proofs

open StableHLO Classical

/-- **The float CIFAR-CNN (no BN) forward** — the float peer of
    `cifarCnnForward`: rounded conv (`flatConvF`) and rounded dense
    (`M.dense`); `relu` and `maxPoolFlat` appear bare (exact in float). -/
noncomputable def FloatModel.cifarCnnForwardF
    {ic c1 c2 h w d1 nClasses kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) :
    Vec (ic * (2*(2*h)) * (2*(2*w))) → Vec nClasses :=
  M.dense W₇ b₇
  ∘ (relu d1 ∘ M.dense W₆ b₆)
  ∘ (relu d1 ∘ M.dense W₅ b₅)
  ∘ maxPoolFlat c2 h w
  ∘ (relu (c2 * (2*h) * (2*w)) ∘ M.flatConvF (h := 2*h) (w := 2*w) W₄ b₄)
  ∘ (relu (c2 * (2*h) * (2*w)) ∘ M.flatConvF (h := 2*h) (w := 2*w) W₃ b₃)
  ∘ maxPoolFlat c1 (2*h) (2*w)
  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w))) ∘ M.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w))) ∘ M.flatConvF (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)

/-- The closed-form forward rounding budget for the no-BN CIFAR CNN — the
    seven-layer `layerBudget`/`layerAct` nest (conv₁..conv₄ then dense₅..dense₇).
    `norm_num`-evaluable at a concrete net and magnitude profile. -/
noncomputable def cifarFwdBudget (u : ℝ)
    (ic c1 c2 h w d1 kH kW : ℕ)
    (w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ w₆ β₆ w₇ β₇ a : ℝ) : ℝ :=
  FloatModel.layerBudget u d1 w₇ β₇
    (FloatModel.layerAct d1 w₆ β₆
      (FloatModel.layerAct (c2*h*w) w₅ β₅
        (FloatModel.layerAct (c2*kH*kW) w₄ β₄
          (FloatModel.layerAct (c1*kH*kW) w₃ β₃
            (FloatModel.layerAct (c1*kH*kW) w₂ β₂
              (FloatModel.layerAct (ic*kH*kW) w₁ β₁ a))))))
    (FloatModel.layerBudget u d1 w₆ β₆
      (FloatModel.layerAct (c2*h*w) w₅ β₅
        (FloatModel.layerAct (c2*kH*kW) w₄ β₄
          (FloatModel.layerAct (c1*kH*kW) w₃ β₃
            (FloatModel.layerAct (c1*kH*kW) w₂ β₂
              (FloatModel.layerAct (ic*kH*kW) w₁ β₁ a)))))
      (FloatModel.layerBudget u (c2*h*w) w₅ β₅
        (FloatModel.layerAct (c2*kH*kW) w₄ β₄
          (FloatModel.layerAct (c1*kH*kW) w₃ β₃
            (FloatModel.layerAct (c1*kH*kW) w₂ β₂
              (FloatModel.layerAct (ic*kH*kW) w₁ β₁ a))))
        (FloatModel.layerBudget u (c2*kH*kW) w₄ β₄
          (FloatModel.layerAct (c1*kH*kW) w₃ β₃
            (FloatModel.layerAct (c1*kH*kW) w₂ β₂
              (FloatModel.layerAct (ic*kH*kW) w₁ β₁ a)))
          (FloatModel.layerBudget u (c1*kH*kW) w₃ β₃
            (FloatModel.layerAct (c1*kH*kW) w₂ β₂
              (FloatModel.layerAct (ic*kH*kW) w₁ β₁ a))
            (FloatModel.layerBudget u (c1*kH*kW) w₂ β₂
              (FloatModel.layerAct (ic*kH*kW) w₁ β₁ a)
              (FloatModel.layerBudget u (ic*kH*kW) w₁ β₁ a 0))))))

/-- **Whole-net no-BN CIFAR-CNN forward rounding budget.** The rounded forward
    is within the explicit `cifarFwdBudget` of the real
    `(conv→relu)²→pool→(conv→relu)²→pool→dense→relu→dense→relu→dense` forward,
    per output logit — the binary32 forward-error bound for the Chapter-4
    no-BN CIFAR net. The CIFAR peer of `cnn_float_close`. -/
theorem FloatModel.cifar_float_close
    {ic c1 c2 h w d1 nClasses kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (x : Vec (ic * (2*(2*h)) * (2*(2*w))))
    {w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ w₆ β₆ w₇ β₇ a : ℝ}
    (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄)
    (hw₅ : 0 ≤ w₅) (hβ₅ : 0 ≤ β₅) (hw₆ : 0 ≤ w₆) (hβ₆ : 0 ≤ β₆)
    (hw₇ : 0 ≤ w₇) (ha : 0 ≤ a)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ o cc kh kw, |W₃ o cc kh kw| ≤ w₃) (hb₃ : ∀ o, |b₃ o| ≤ β₃)
    (hW₄ : ∀ o cc kh kw, |W₄ o cc kh kw| ≤ w₄) (hb₄ : ∀ o, |b₄ o| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hW₆ : ∀ i j, |W₆ i j| ≤ w₆) (hb₆ : ∀ j, |b₆ j| ≤ β₆)
    (hW₇ : ∀ i j, |W₇ i j| ≤ w₇) (hb₇ : ∀ j, |b₇ j| ≤ β₇)
    (hx : ∀ i, |x i| ≤ a) (k : Fin nClasses) :
    |M.cifarCnnForwardF W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x k -
        cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x k| ≤
      cifarFwdBudget M.u ic c1 c2 h w d1 kH kW
        w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ w₆ β₆ w₇ β₇ a := by
  simp only [FloatModel.cifarCnnForwardF, cifarCnnForward, Function.comp, cifarFwdBudget]
  -- per-layer scalar magnitudes (A) and budgets (E)
  set A1 := FloatModel.layerAct (ic * kH * kW) w₁ β₁ a with hA1
  set A2 := FloatModel.layerAct (c1 * kH * kW) w₂ β₂ A1 with hA2
  set A3 := FloatModel.layerAct (c1 * kH * kW) w₃ β₃ A2 with hA3
  set A4 := FloatModel.layerAct (c2 * kH * kW) w₄ β₄ A3 with hA4
  set A5 := FloatModel.layerAct (c2 * h * w) w₅ β₅ A4 with hA5
  set A6 := FloatModel.layerAct d1 w₆ β₆ A5 with hA6
  set E1 := FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 with hE1
  set E2 := FloatModel.layerBudget M.u (c1 * kH * kW) w₂ β₂ A1 E1 with hE2
  set E3 := FloatModel.layerBudget M.u (c1 * kH * kW) w₃ β₃ A2 E2 with hE3
  set E4 := FloatModel.layerBudget M.u (c2 * kH * kW) w₄ β₄ A3 E3 with hE4
  set E5 := FloatModel.layerBudget M.u (c2 * h * w) w₅ β₅ A4 E4 with hE5
  set E6 := FloatModel.layerBudget M.u d1 w₆ β₆ A5 E5 with hE6
  -- nonnegativity
  have hA1_0 : 0 ≤ A1 := FloatModel.layerAct_nonneg hw₁ hβ₁ ha
  have hE1_0 : 0 ≤ E1 := FloatModel.layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl
  have hA2_0 : 0 ≤ A2 := FloatModel.layerAct_nonneg hw₂ hβ₂ hA1_0
  have hE2_0 : 0 ≤ E2 := FloatModel.layerBudget_nonneg M.u_nonneg hw₂ hβ₂ hA1_0 hE1_0
  have hA3_0 : 0 ≤ A3 := FloatModel.layerAct_nonneg hw₃ hβ₃ hA2_0
  have hE3_0 : 0 ≤ E3 := FloatModel.layerBudget_nonneg M.u_nonneg hw₃ hβ₃ hA2_0 hE2_0
  have hA4_0 : 0 ≤ A4 := FloatModel.layerAct_nonneg hw₄ hβ₄ hA3_0
  have hE4_0 : 0 ≤ E4 := FloatModel.layerBudget_nonneg M.u_nonneg hw₄ hβ₄ hA3_0 hE3_0
  have hA5_0 : 0 ≤ A5 := FloatModel.layerAct_nonneg hw₅ hβ₅ hA4_0
  have hE5_0 : 0 ≤ E5 := FloatModel.layerBudget_nonneg M.u_nonneg hw₅ hβ₅ hA4_0 hE4_0
  have hA6_0 : 0 ≤ A6 := FloatModel.layerAct_nonneg hw₆ hβ₆ hA5_0
  have hE6_0 : 0 ≤ E6 := FloatModel.layerBudget_nonneg M.u_nonneg hw₆ hβ₆ hA5_0 hE5_0
  -- name the real (xr*) and float (xf*) post-activation tensors, bottom-up
  set xr1 := relu (c1 * (2*(2*h)) * (2*(2*w))) (flatConv W₁ b₁ x) with hxr1
  set xf1 := relu (c1 * (2*(2*h)) * (2*(2*w))) (M.flatConvF W₁ b₁ x) with hxf1
  set xr2 := relu (c1 * (2*(2*h)) * (2*(2*w))) (flatConv W₂ b₂ xr1) with hxr2
  set xf2 := relu (c1 * (2*(2*h)) * (2*(2*w))) (M.flatConvF W₂ b₂ xf1) with hxf2
  set xr2p := maxPoolFlat c1 (2*h) (2*w) xr2 with hxr2p
  set xf2p := maxPoolFlat c1 (2*h) (2*w) xf2 with hxf2p
  set xr3 := relu (c2 * (2*h) * (2*w)) (flatConv W₃ b₃ xr2p) with hxr3
  set xf3 := relu (c2 * (2*h) * (2*w)) (M.flatConvF W₃ b₃ xf2p) with hxf3
  set xr4 := relu (c2 * (2*h) * (2*w)) (flatConv W₄ b₄ xr3) with hxr4
  set xf4 := relu (c2 * (2*h) * (2*w)) (M.flatConvF W₄ b₄ xf3) with hxf4
  set xr4p := maxPoolFlat c2 h w xr4 with hxr4p
  set xf4p := maxPoolFlat c2 h w xf4 with hxf4p
  set xr5 := relu d1 (Proofs.dense W₅ b₅ xr4p) with hxr5
  set xf5 := relu d1 (M.dense W₅ b₅ xf4p) with hxf5
  set xr6 := relu d1 (Proofs.dense W₆ b₆ xr5) with hxr6
  set xf6 := relu d1 (M.dense W₆ b₆ xf5) with hxf6
  -- real activation magnitude bounds
  have mA1 : ∀ j, |xr1 j| ≤ A1 :=
    fun j => (FloatModel.relu_abs_le _ j).trans (flatConv_abs_le ha hW₁ hb₁ hx j)
  have mA2 : ∀ j, |xr2 j| ≤ A2 :=
    fun j => (FloatModel.relu_abs_le _ j).trans (flatConv_abs_le hA1_0 hW₂ hb₂ mA1 j)
  have mAp1 : ∀ j, |xr2p j| ≤ A2 := fun j => maxPoolFlat_abs_le mA2 j
  have mA3 : ∀ j, |xr3 j| ≤ A3 :=
    fun j => (FloatModel.relu_abs_le _ j).trans (flatConv_abs_le hA2_0 hW₃ hb₃ mAp1 j)
  have mA4 : ∀ j, |xr4 j| ≤ A4 :=
    fun j => (FloatModel.relu_abs_le _ j).trans (flatConv_abs_le hA3_0 hW₄ hb₄ mA3 j)
  have mAp2 : ∀ j, |xr4p j| ≤ A4 := fun j => maxPoolFlat_abs_le mA4 j
  have mA5 : ∀ j, |xr5 j| ≤ A5 :=
    fun j => (FloatModel.relu_abs_le _ j).trans (FloatModel.dense_abs_le hA4_0 hW₅ hb₅ mAp2 j)
  have mA6 : ∀ j, |xr6 j| ≤ A6 :=
    fun j => (FloatModel.relu_abs_le _ j).trans (FloatModel.dense_abs_le hA5_0 hW₆ hb₆ mA5 j)
  -- float-vs-real error, layer by layer
  have e1 : ∀ j, |M.flatConvF W₁ b₁ x j - flatConv W₁ b₁ x j| ≤ E1 :=
    fun j => M.flatConvF_close W₁ b₁ x x hw₁ ha le_rfl hW₁ hb₁ hx (fun i => by simp) j
  have r1 : ∀ j, |xf1 j - xr1 j| ≤ E1 := fun j => FloatModel.relu_close _ _ E1 e1 j
  have e2 : ∀ j, |M.flatConvF W₂ b₂ xf1 j - flatConv W₂ b₂ xr1 j| ≤ E2 :=
    fun j => M.flatConvF_close W₂ b₂ xf1 xr1 hw₂ hA1_0 hE1_0 hW₂ hb₂ mA1 r1 j
  have r2 : ∀ j, |xf2 j - xr2 j| ≤ E2 := fun j => FloatModel.relu_close _ _ E2 e2 j
  have ep1 : ∀ j, |xf2p j - xr2p j| ≤ E2 := fun j => maxPoolFlat_close _ _ r2 j
  have e3 : ∀ j, |M.flatConvF W₃ b₃ xf2p j - flatConv W₃ b₃ xr2p j| ≤ E3 :=
    fun j => M.flatConvF_close W₃ b₃ xf2p xr2p hw₃ hA2_0 hE2_0 hW₃ hb₃ mAp1 ep1 j
  have r3 : ∀ j, |xf3 j - xr3 j| ≤ E3 := fun j => FloatModel.relu_close _ _ E3 e3 j
  have e4 : ∀ j, |M.flatConvF W₄ b₄ xf3 j - flatConv W₄ b₄ xr3 j| ≤ E4 :=
    fun j => M.flatConvF_close W₄ b₄ xf3 xr3 hw₄ hA3_0 hE3_0 hW₄ hb₄ mA3 r3 j
  have r4 : ∀ j, |xf4 j - xr4 j| ≤ E4 := fun j => FloatModel.relu_close _ _ E4 e4 j
  have ep2 : ∀ j, |xf4p j - xr4p j| ≤ E4 := fun j => maxPoolFlat_close _ _ r4 j
  have e5 : ∀ j, |M.dense W₅ b₅ xf4p j - Proofs.dense W₅ b₅ xr4p j| ≤ E5 :=
    fun j => (M.dense_close W₅ b₅ _ _ E4 hE4_0 ep2 j).trans
      (M.denseErr_le_uniform hw₅ hE4_0 hW₅ hb₅ mAp2 j)
  have r5 : ∀ j, |xf5 j - xr5 j| ≤ E5 := fun j => FloatModel.relu_close _ _ E5 e5 j
  have e6 : ∀ j, |M.dense W₆ b₆ xf5 j - Proofs.dense W₆ b₆ xr5 j| ≤ E6 :=
    fun j => (M.dense_close W₆ b₆ _ _ E5 hE5_0 r5 j).trans
      (M.denseErr_le_uniform hw₆ hE5_0 hW₆ hb₆ mA5 j)
  have r6 : ∀ j, |xf6 j - xr6 j| ≤ E6 := fun j => FloatModel.relu_close _ _ E6 e6 j
  -- final dense layer
  exact (M.dense_close W₇ b₇ _ _ E6 hE6_0 r6 k).trans
    (M.denseErr_le_uniform hw₇ hE6_0 hW₇ hb₇ mA6 k)

/-! ## SGD-step rounding budgets for the no-BN CIFAR conv layers

The per-parameter rounded SGD step is `cnn_convW_step_float_close` /
`cnn_convb_step_float_close` (dimension-generic in the cotangent, so they apply
to CIFAR's convs verbatim). What is CIFAR-specific is the *numeric* budget at the
two committed spatial scales: the committed `cifarVerified` net (3→32→32→pool→
32→64→64→pool→512→512→10 at 32×32) has its first two convs at the **32×32**
output grid (weight-grad dot over `1024` spatial positions) and its last two at
**16×16** (over `256`). At binary32 (`u ≤ 2⁻²⁴`), `lr = 1/10`, kernel `|W| ≤ 3/5`,
each rounded conv weight/bias SGD entry is as accurate as its gradient — the
CIFAR peers of `mnist_cnn_convW_step_float_budget`. `a` bounds the conv-input
activation, `g` the back-propagated cotangent (both a-posteriori, supplied). The
dense head (512×512 / 4096×512) reuses the MLP weight/bias step closes directly.
-/

/-- **Stage-1 conv weight step (32×32 grid, fan-in 1024).** Covers conv₁ (3→32)
    and conv₂ (32→32): every rounded weight SGD entry is within `(a·g)/150 + 10⁻⁷`
    of the certified step. The `1/150 ≈ 0.67%` rate is `lr·γ₁₀₂₅` (the dot's
    Higham error at learning-rate scale) — the step is as accurate as the gradient. -/
theorem FloatModel.cifar_stage1_convW_step_float_budget {ic oc : Nat} (M : FloatModel)
    (hMu : M.u ≤ u32) (W : Kernel4 oc ic 3 3) (act : Tensor3 ic 32 32)
    (cot : Tensor3 oc 32 32) {a g : ℝ} (ha : 0 ≤ a) (hg : 0 ≤ g)
    (hW : ∀ o cc kh kw, |W o cc kh kw| ≤ 3/5)
    (hact : ∀ c i j, |act c i j| ≤ a) (hcot : ∀ o i j, |cot o i j| ≤ g)
    (o : Fin oc) (cc : Fin ic) (kh kw : Fin 3) :
    |M.sub (W o cc kh kw)
        (M.mul (1/10) (M.dot (convPadWin 3 3 act cc kh kw) (cotWin cot o))) -
      (W o cc kh kw - (1/10) * ∑ s,
        convPadWin 3 3 act cc kh kw s * cotWin cot o s)| ≤
      (a * g) / 150 + 1/10000000 := by
  have hu := M.u_nonneg
  have hterm : ∀ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤ a * g := by
    intro s
    rw [abs_mul]
    refine mul_le_mul ?_ ?_ (abs_nonneg _) ha
    · simp only [convPadWin]; exact abs_convPad_le act ha hact _ _ _ _ _
    · simp only [cotWin]; exact hcot _ _ _
  have hsum : ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤
      1024 * (a * g) := by
    calc ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s|
        ≤ ∑ _s : Fin (32 * 32), a * g := Finset.sum_le_sum fun s _ => hterm s
      _ = 1024 * (a * g) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul]; norm_num
  have hG : |∑ s, convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤ 1024 * (a * g) :=
    (Finset.abs_sum_le_sum_abs _ _).trans hsum
  have hstep := M.cnn_convW_step_float_close W act cot o cc kh kw hG
    (by norm_num : (0:ℝ) ≤ 1/10)
  refine hstep.trans ?_
  have hk1 : ((32 * 32 + 1 : ℕ) : ℝ) * u32 < 1 := by norm_num [u32]
  have hk2 : ((32 * 32 + 1 : ℕ) : ℝ) * u32 / (1 - ((32 * 32 + 1 : ℕ) : ℝ) * u32)
      ≤ 62/1000000 := by norm_num [u32]
  have hhigham : (1 + M.u) ^ (32 * 32 + 1) - 1 ≤ 62/1000000 :=
    M.gamma_num hMu hk1 hk2
  have hhigham0 : 0 ≤ (1 + M.u) ^ (32 * 32 + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hsum0 : 0 ≤ ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| :=
    Finset.sum_nonneg fun s _ => abs_nonneg _
  have heg : ((1 + M.u) ^ (32 * 32 + 1) - 1) *
      ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤
      (62/1000000) * (1024 * (a * g)) :=
    mul_le_mul hhigham hsum hsum0 (by norm_num)
  have hag0 : (0:ℝ) ≤ a * g := mul_nonneg ha hg
  have h1 : u32 ≤ 1/16000000 := by norm_num [u32]
  refine (sgdErr_mono hu (hMu.trans h1) (by norm_num) (abs_nonneg _)
    (hW o cc kh kw) (mul_nonneg (by norm_num) hag0)
    (mul_nonneg hhigham0 hsum0) heg).trans ?_
  set s := a * g with hs
  have hs0 : (0:ℝ) ≤ s := hag0
  unfold FloatModel.sgdErr
  linarith [hs0]

/-- **Stage-1 conv bias step (32×32 grid, fan-in 1024).** The bias gradient is
    the spatial sum `Σ cot`, so the rounded update is within `g/150 + 10⁻⁷`
    of the certified step (`|b| ≤ 1`). -/
theorem FloatModel.cifar_stage1_convb_step_float_budget {oc : Nat} (M : FloatModel)
    (hMu : M.u ≤ u32) (b : Vec oc) (cot : Tensor3 oc 32 32) {g : ℝ} (hg : 0 ≤ g)
    (hb : ∀ o, |b o| ≤ 1) (hcot : ∀ o i j, |cot o i j| ≤ g) (o : Fin oc) :
    |M.sub (b o) (M.mul (1/10) (M.sum (cotWin cot o))) -
      (b o - (1/10) * ∑ s, cotWin cot o s)| ≤
      g / 150 + 1/10000000 := by
  have hu := M.u_nonneg
  have hterm : ∀ s, |cotWin cot o s| ≤ g := fun s => by
    simp only [cotWin]; exact hcot _ _ _
  have hsum : ∑ s, |cotWin cot o s| ≤ 1024 * g := by
    calc ∑ s, |cotWin cot o s| ≤ ∑ _s : Fin (32 * 32), g :=
          Finset.sum_le_sum fun s _ => hterm s
      _ = 1024 * g := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul]; norm_num
  have hG : |∑ s, cotWin cot o s| ≤ 1024 * g :=
    (Finset.abs_sum_le_sum_abs _ _).trans hsum
  have hstep := M.cnn_convb_step_float_close b cot o hG (by norm_num : (0:ℝ) ≤ 1/10)
  refine hstep.trans ?_
  have hk1 : ((32 * 32 + 1 : ℕ) : ℝ) * u32 < 1 := by norm_num [u32]
  have hk2 : ((32 * 32 + 1 : ℕ) : ℝ) * u32 / (1 - ((32 * 32 + 1 : ℕ) : ℝ) * u32)
      ≤ 62/1000000 := by norm_num [u32]
  have hhigham : (1 + M.u) ^ (32 * 32 + 1) - 1 ≤ 62/1000000 :=
    M.gamma_num hMu hk1 hk2
  have hhigham0 : 0 ≤ (1 + M.u) ^ (32 * 32 + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hsum0 : 0 ≤ ∑ s, |cotWin cot o s| := Finset.sum_nonneg fun s _ => abs_nonneg _
  have heg : ((1 + M.u) ^ (32 * 32 + 1) - 1) * ∑ s, |cotWin cot o s| ≤
      (62/1000000) * (1024 * g) := mul_le_mul hhigham hsum hsum0 (by norm_num)
  have h1 : u32 ≤ 1/16000000 := by norm_num [u32]
  refine (sgdErr_mono hu (hMu.trans h1) (by norm_num) (abs_nonneg _)
    (hb o) (mul_nonneg (by norm_num) hg)
    (mul_nonneg hhigham0 hsum0) heg).trans ?_
  set s := g with hs
  have hs0 : (0:ℝ) ≤ s := hg
  unfold FloatModel.sgdErr
  linarith [hs0]

/-- **Stage-2 conv weight step (16×16 grid, fan-in 256).** Covers conv₃ (32→64)
    and conv₄ (64→64): every rounded weight SGD entry is within `(a·g)/2000 + 10⁻⁷`
    of the certified step — tighter than stage-1, the smaller `16×16` grid. -/
theorem FloatModel.cifar_stage2_convW_step_float_budget {ic oc : Nat} (M : FloatModel)
    (hMu : M.u ≤ u32) (W : Kernel4 oc ic 3 3) (act : Tensor3 ic 16 16)
    (cot : Tensor3 oc 16 16) {a g : ℝ} (ha : 0 ≤ a) (hg : 0 ≤ g)
    (hW : ∀ o cc kh kw, |W o cc kh kw| ≤ 3/5)
    (hact : ∀ c i j, |act c i j| ≤ a) (hcot : ∀ o i j, |cot o i j| ≤ g)
    (o : Fin oc) (cc : Fin ic) (kh kw : Fin 3) :
    |M.sub (W o cc kh kw)
        (M.mul (1/10) (M.dot (convPadWin 3 3 act cc kh kw) (cotWin cot o))) -
      (W o cc kh kw - (1/10) * ∑ s,
        convPadWin 3 3 act cc kh kw s * cotWin cot o s)| ≤
      (a * g) / 2000 + 1/10000000 := by
  have hu := M.u_nonneg
  have hterm : ∀ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤ a * g := by
    intro s
    rw [abs_mul]
    refine mul_le_mul ?_ ?_ (abs_nonneg _) ha
    · simp only [convPadWin]; exact abs_convPad_le act ha hact _ _ _ _ _
    · simp only [cotWin]; exact hcot _ _ _
  have hsum : ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤
      256 * (a * g) := by
    calc ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s|
        ≤ ∑ _s : Fin (16 * 16), a * g := Finset.sum_le_sum fun s _ => hterm s
      _ = 256 * (a * g) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul]; norm_num
  have hG : |∑ s, convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤ 256 * (a * g) :=
    (Finset.abs_sum_le_sum_abs _ _).trans hsum
  have hstep := M.cnn_convW_step_float_close W act cot o cc kh kw hG
    (by norm_num : (0:ℝ) ≤ 1/10)
  refine hstep.trans ?_
  have hk1 : ((16 * 16 + 1 : ℕ) : ℝ) * u32 < 1 := by norm_num [u32]
  have hk2 : ((16 * 16 + 1 : ℕ) : ℝ) * u32 / (1 - ((16 * 16 + 1 : ℕ) : ℝ) * u32)
      ≤ 16/1000000 := by norm_num [u32]
  have hhigham : (1 + M.u) ^ (16 * 16 + 1) - 1 ≤ 16/1000000 :=
    M.gamma_num hMu hk1 hk2
  have hhigham0 : 0 ≤ (1 + M.u) ^ (16 * 16 + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hsum0 : 0 ≤ ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| :=
    Finset.sum_nonneg fun s _ => abs_nonneg _
  have heg : ((1 + M.u) ^ (16 * 16 + 1) - 1) *
      ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤
      (16/1000000) * (256 * (a * g)) :=
    mul_le_mul hhigham hsum hsum0 (by norm_num)
  have hag0 : (0:ℝ) ≤ a * g := mul_nonneg ha hg
  have h1 : u32 ≤ 1/16000000 := by norm_num [u32]
  refine (sgdErr_mono hu (hMu.trans h1) (by norm_num) (abs_nonneg _)
    (hW o cc kh kw) (mul_nonneg (by norm_num) hag0)
    (mul_nonneg hhigham0 hsum0) heg).trans ?_
  set s := a * g with hs
  have hs0 : (0:ℝ) ≤ s := hag0
  unfold FloatModel.sgdErr
  linarith [hs0]

/-- **Stage-2 conv bias step (16×16 grid, fan-in 256).** Within `g/2000 + 10⁻⁷`
    of the certified step (`|b| ≤ 1`). -/
theorem FloatModel.cifar_stage2_convb_step_float_budget {oc : Nat} (M : FloatModel)
    (hMu : M.u ≤ u32) (b : Vec oc) (cot : Tensor3 oc 16 16) {g : ℝ} (hg : 0 ≤ g)
    (hb : ∀ o, |b o| ≤ 1) (hcot : ∀ o i j, |cot o i j| ≤ g) (o : Fin oc) :
    |M.sub (b o) (M.mul (1/10) (M.sum (cotWin cot o))) -
      (b o - (1/10) * ∑ s, cotWin cot o s)| ≤
      g / 2000 + 1/10000000 := by
  have hu := M.u_nonneg
  have hterm : ∀ s, |cotWin cot o s| ≤ g := fun s => by
    simp only [cotWin]; exact hcot _ _ _
  have hsum : ∑ s, |cotWin cot o s| ≤ 256 * g := by
    calc ∑ s, |cotWin cot o s| ≤ ∑ _s : Fin (16 * 16), g :=
          Finset.sum_le_sum fun s _ => hterm s
      _ = 256 * g := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul]; norm_num
  have hG : |∑ s, cotWin cot o s| ≤ 256 * g :=
    (Finset.abs_sum_le_sum_abs _ _).trans hsum
  have hstep := M.cnn_convb_step_float_close b cot o hG (by norm_num : (0:ℝ) ≤ 1/10)
  refine hstep.trans ?_
  have hk1 : ((16 * 16 + 1 : ℕ) : ℝ) * u32 < 1 := by norm_num [u32]
  have hk2 : ((16 * 16 + 1 : ℕ) : ℝ) * u32 / (1 - ((16 * 16 + 1 : ℕ) : ℝ) * u32)
      ≤ 16/1000000 := by norm_num [u32]
  have hhigham : (1 + M.u) ^ (16 * 16 + 1) - 1 ≤ 16/1000000 :=
    M.gamma_num hMu hk1 hk2
  have hhigham0 : 0 ≤ (1 + M.u) ^ (16 * 16 + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hsum0 : 0 ≤ ∑ s, |cotWin cot o s| := Finset.sum_nonneg fun s _ => abs_nonneg _
  have heg : ((1 + M.u) ^ (16 * 16 + 1) - 1) * ∑ s, |cotWin cot o s| ≤
      (16/1000000) * (256 * g) := mul_le_mul hhigham hsum hsum0 (by norm_num)
  have h1 : u32 ≤ 1/16000000 := by norm_num [u32]
  refine (sgdErr_mono hu (hMu.trans h1) (by norm_num) (abs_nonneg _)
    (hb o) (mul_nonneg (by norm_num) hg)
    (mul_nonneg hhigham0 hsum0) heg).trans ?_
  set s := g with hs
  have hs0 : (0:ℝ) ≤ s := hg
  unfold FloatModel.sgdErr
  linarith [hs0]

end Proofs
