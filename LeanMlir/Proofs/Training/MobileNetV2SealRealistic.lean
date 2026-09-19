import LeanMlir.Proofs.Training.MobileNetV2JacobianSeal
import LeanMlir.Proofs.Training.ResNet34LiveRealisticSeal
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Realistic-dimension `Mnv2Live` nonzero-Jacobian seal (level 3, 224×224)

`MobileNetV2JacobianSeal` seals the live MobileNetV2 at a **1×2×2** toy input. This file
lifts it to real **ImageNet 224×224** spatial resolution.

## The obstacle (vs. ResNet-34): ReLU6 is a *bounded* window, not one-sided

ResNet-34's ReLU is one-sided, so its smoothness uses the **positivity route**
`β − |γ|·√n > 0` with `β` free to grow with `√n`. MobileNetV2's **ReLU6** needs the BN
output inside the *bounded* `(0,6)`: `β` large would clamp at 6. The toy keeps `β = 3`
(centered) and `γ = 1`, which forces `√n < 3` (`bn13_window` needs `n ≤ 8`). At realistic
`n = 2·112·112` the window fails. The fix is to **scale `γ` down** — `|γ|·√n < 3` keeps
`bn ∈ (3 − |γ|√n, 3 + |γ|√n) ⊆ (0,6)`; here `γ = 1/128` gives `(1/128)·√100352 ≈ 2.48 < 3`.
The MobileNetV2 weights (`Ws`, `We₁`, …) are all 1×1 channel maps, hence **dimension-
independent** and reused verbatim; only the spatial size and `γ` change.

The seal itself reuses the `UDiff` channel-difference machinery of `R34RealSeal` (a uniform
channel-0 perturbation makes `channel0 = channel1 + δ` everywhere), threaded through the
ReLU6-free twin (the window makes every ReLU6 the identity). The stem's asymmetric channel
weights turn a uniform input perturbation `t` into the channel difference `−t`; each BN then
multiplies it by `γ·istd`, so the output difference is `−t · Rr` with `Rr` a product of four
positive `γ·istd`s, and `g'(0) = −Rr 0 ≠ 0`.
-/

namespace Proofs
namespace Mnv2RealSeal

open scoped BigOperators
open Finset Filter Topology
open Proofs Mnv2Live R34RealSeal

-- ════════════════════════════════════════════════════════════════
-- § Foundation: BN upper bound + the γ-scaled ReLU6 window
-- ════════════════════════════════════════════════════════════════

/-- **BN upper bound** `bn ≤ β + |γ|·√n` (the companion of `bnForward_lb`). -/
theorem bnForward_ub {n : Nat} (ε γ β : ℝ) (hε : 0 < ε) (v : Vec n) (k : Fin n) :
    bnForward n ε γ β v k ≤ β + |γ| * Real.sqrt (n : ℝ) := by
  have hsq := bnXhat_sq_le ε hε v k
  have habs : |bnXhat n ε v k| ≤ Real.sqrt (n : ℝ) := Real.abs_le_sqrt hsq
  have hmul : |γ * bnXhat n ε v k| ≤ |γ| * Real.sqrt (n : ℝ) := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left habs (abs_nonneg γ)
  have hle : γ * bnXhat n ε v k ≤ |γ| * Real.sqrt (n : ℝ) := le_trans (le_abs_self _) hmul
  simp only [bnForward]; linarith

/-- `√100352 < 384` (so `(1/128)·√100352 < 3`). -/
theorem sqrtN_lt : Real.sqrt ((2 * 112 * 112 : ℕ) : ℝ) < 384 :=
  ResNet34LiveRealistic.sqrt_lt_param (2 * 112 * 112) 384 (by norm_num) (by norm_num)

/-- `(1/128)·√100352 < 3`. -/
theorem gsqrt_lt : (1 / 128 : ℝ) * Real.sqrt ((2 * 112 * 112 : ℕ) : ℝ) < 3 := by
  rw [show (3 : ℝ) = (1 / 128) * 384 by norm_num]
  exact mul_lt_mul_of_pos_left sqrtN_lt (by norm_num)

/-- **The γ-scaled ReLU6 window**: at `γ = 1/128`, `β = 3` and `n = 2·112·112`, every BN
    output lands strictly inside `(0,6)`, so ReLU6 never binds (for *any* input). -/
theorem winR (z : Vec (2 * 112 * 112)) (k : Fin (2 * 112 * 112)) :
    0 < bnForward (2 * 112 * 112) 1 (1 / 128) 3 z k ∧
      bnForward (2 * 112 * 112) 1 (1 / 128) 3 z k < 6 := by
  have hlb := bnForward_lb (n := 2 * 112 * 112) 1 (1 / 128) 3 (by norm_num) z k
  have hub := bnForward_ub (n := 2 * 112 * 112) 1 (1 / 128) 3 (by norm_num) z k
  rw [show |(1 / 128 : ℝ)| = 1 / 128 from by norm_num] at hlb hub
  exact ⟨by linarith [gsqrt_lt], by linarith [gsqrt_lt]⟩

theorem winR_ne (z : Vec (2 * 112 * 112)) (k : Fin (2 * 112 * 112)) :
    bnForward (2 * 112 * 112) 1 (1 / 128) 3 z k ≠ 0 ∧
      bnForward (2 * 112 * 112) 1 (1 / 128) 3 z k ≠ 6 := by
  obtain ⟨h0, h6⟩ := winR z k; exact ⟨h0.ne', h6.ne⟩

-- ════════════════════════════════════════════════════════════════
-- § The realistic forward (1×224×224 input, 2 channels, γ = 1/128)
-- ════════════════════════════════════════════════════════════════

/-- The live MobileNetV2 at ImageNet resolution: the toy's 1×1 channel-map weights, lifted
    to `h = w = 112` spatial (input `1×224×224`) and `γ = 1/128` (the ReLU6-window scale). -/
noncomputable abbrev fwdR : Vec (1 * 112 * 112) → Vec 2 :=
  mobilenetv2Forward Ws bs 1 (1 / 128) 3 We₁ be₁ 1 (1 / 128) 3 Wd₁ bd₁ 1 (1 / 128) 3 Wp₁ bp₁ 1 (1 / 128) 3
    We₂ be₂ 1 (1 / 128) 3 Wd₂ bd₂ 1 (1 / 128) 3 Wp₂ bp₂ 1 (1 / 128) 3 Wh bh

/-- **Whole-network VJP at every input** — the ReLU6 window `winR` discharges all five
    smoothness sites regardless of activation, at realistic dims. -/
noncomputable def fwdR_has_vjp_at (v : Vec (1 * 112 * 112)) : HasVJPAt fwdR v :=
  mobilenetv2_has_vjp_at Ws bs 1 (1 / 128) 3 one_pos
    We₁ be₁ 1 (1 / 128) 3 one_pos Wd₁ bd₁ 1 (1 / 128) 3 one_pos Wp₁ bp₁ 1 (1 / 128) 3 one_pos
    We₂ be₂ 1 (1 / 128) 3 one_pos Wd₂ bd₂ 1 (1 / 128) 3 one_pos Wp₂ bp₂ 1 (1 / 128) 3 one_pos Wh bh v
    (fun k => winR_ne _ k) (fun k => winR_ne _ k) (fun k => winR_ne _ k)
    (fun k => winR_ne _ k) (fun k => winR_ne _ k)

theorem fwdR_has_vjp_correct (v : Vec (1 * 112 * 112)) (dy : Vec 2) (i : Fin (1 * 112 * 112)) :
    (fwdR_has_vjp_at v).backward dy i = ∑ j : Fin 2, pdiv fwdR v i j * dy j :=
  (fwdR_has_vjp_at v).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § Generic (dim-free) block reductions at γ = 1/128
-- ════════════════════════════════════════════════════════════════

/-- The unit 1×1 depthwise conv (`W = 1`, `b = 0`) is the identity, any spatial dims. -/
theorem depthwiseFlat_unit_id {h w : Nat} (W : DepthwiseKernel 2 1 1) (b : Vec 2)
    (hW : ∀ ch, W ch 0 0 = 1) (hb : ∀ ch, b ch = 0) (v : Vec (2 * h * w)) :
    depthwiseFlat (h := h) (w := w) W b v = v := by
  have hc : depthwiseConv2d W b (Tensor3.unflatten v) = Tensor3.unflatten v := by
    funext ch hi wi
    simp only [depthwiseConv2d, hb, Fin.sum_univ_one, hW, one_mul, zero_add]
    split
    · exact congrArg₂ (Tensor3.unflatten v ch)
        (by apply Fin.ext; simp only [Fin.val_zero]; omega)
        (by apply Fin.ext; simp only [Fin.val_zero]; omega)
    · rename_i hcond
      exact absurd (by
        have := hi.isLt; have := wi.isLt
        refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only [Fin.val_zero] <;> omega) hcond
  simp only [depthwiseFlat, hc, Tensor3.flatten_unflatten]

/-- ReLU6 fixes every BN output (the window `winR` lands in `(0,6)`). -/
theorem relu6_bnR (z : Vec (2 * 112 * 112)) :
    relu6 (2 * 112 * 112) (bnForward (2 * 112 * 112) 1 (1 / 128) 3 z)
      = bnForward (2 * 112 * 112) 1 (1 / 128) 3 z :=
  relu6_id_window _ _ (fun k => winR z k)

/-- ReLU6 fixes the constant-`3` vector. -/
theorem relu6_const3R : relu6 (2 * 112 * 112) (fun _ => 3) = (fun _ => 3) :=
  relu6_id_window _ _ (fun _ => ⟨by norm_num, by norm_num⟩)

/-- BN sends a constant vector to `β = 3`. -/
theorem bnR_const (c : ℝ) : bnForward (2 * 112 * 112) 1 (1 / 128) 3 (fun _ => c) = (fun _ => 3) :=
  bnForward_const (by norm_num) 1 (1 / 128) 3 c

/-- **The zeroed skip-block body is constantly `3`** (γ-independent: zeroed convs ⇒ BN of a
    constant ⇒ `β = 3`; the centered window keeps every ReLU6 the identity). -/
theorem invresBody₁_constR (y : Vec (2 * 112 * 112)) :
    invresBody (h := 112) (w := 112) We₁ be₁ 1 (1 / 128) 3 Wd₁ bd₁ 1 (1 / 128) 3 Wp₁ bp₁ 1 (1 / 128) 3 y
      = (fun _ => 3) := by
  simp only [invresBody, ivProject, ivDepthwise, ivExpand, Function.comp_apply,
    flatConv_eq_zero We₁ be₁ (fun _ _ _ _ => rfl) (fun _ => rfl),
    flatConv_eq_zero Wp₁ bp₁ (fun _ _ _ _ => rfl) (fun _ => rfl),
    depthwiseFlat_eq_zero Wd₁ bd₁ (fun _ _ _ => rfl) (fun _ => rfl),
    bnR_const, relu6_const3R]

-- ════════════════════════════════════════════════════════════════
-- § The ReLU6-free twin `fwdRS` (= the toy's `fwdCF`, at 112×112, γ=1/128)
-- ════════════════════════════════════════════════════════════════

noncomputable def fwdRS : Vec (1 * 112 * 112) → Vec 2 :=
  fun v => dense Wh bh (globalAvgPoolFlat 2 112 112
    (bnForward (2 * 112 * 112) 1 (1 / 128) 3 (bnForward (2 * 112 * 112) 1 (1 / 128) 3
      (bnForward (2 * 112 * 112) 1 (1 / 128) 3
        (fun k => 3 + bnForward (2 * 112 * 112) 1 (1 / 128) 3 (flatConv (h := 112) (w := 112) Ws bs v) k)))))

theorem fwdR_eq (v : Vec (1 * 112 * 112)) : fwdR v = fwdRS v := by
  have hb1 :
      residual (invresBody (h := 112) (w := 112) We₁ be₁ 1 (1 / 128) 3 Wd₁ bd₁ 1 (1 / 128) 3 Wp₁ bp₁ 1 (1 / 128) 3)
        (bnForward (2 * 112 * 112) 1 (1 / 128) 3 (flatConv (h := 112) (w := 112) Ws bs v))
        = (fun k => 3 + bnForward (2 * 112 * 112) 1 (1 / 128) 3 (flatConv (h := 112) (w := 112) Ws bs v) k) := by
    funext k; simp only [residual, biPath, invresBody₁_constR]
  show mobilenetv2Forward Ws bs 1 (1 / 128) 3 We₁ be₁ 1 (1 / 128) 3 Wd₁ bd₁ 1 (1 / 128) 3 Wp₁ bp₁ 1 (1 / 128) 3
      We₂ be₂ 1 (1 / 128) 3 Wd₂ bd₂ 1 (1 / 128) 3 Wp₂ bp₂ 1 (1 / 128) 3 Wh bh v = fwdRS v
  simp only [mobilenetv2Forward, Function.comp_apply, relu6_bnR,
    ResNet34LivePC.flatConv_diag_id (h := 112) (w := 112) We₂ be₂ (fun o i => rfl) (fun _ => rfl),
    ResNet34LivePC.flatConv_diag_id (h := 112) (w := 112) Wp₂ bp₂ (fun o i => rfl) (fun _ => rfl),
    depthwiseFlat_unit_id Wd₂ bd₂ (fun _ => rfl) (fun _ => rfl),
    hb1, fwdRS]

/-- The twin is differentiable everywhere (conv, BN under `ε=1>0`, the affine `+3`, GAP, dense;
    no ReLU6 obstruction since it never binds). -/
theorem fwdRS_differentiable : Differentiable ℝ fwdRS := by
  have hbn : Differentiable ℝ (bnForward (2 * 112 * 112) 1 (1 / 128) 3) :=
    bnForward_differentiable (2 * 112 * 112) 1 (1 / 128) 3 one_pos
  have hconv : Differentiable ℝ
      (flatConv (h := 112) (w := 112) Ws bs : Vec (1 * 112 * 112) → Vec (2 * 112 * 112)) :=
    flatConv_differentiable Ws bs
  have hgap : Differentiable ℝ (globalAvgPoolFlat 2 112 112 : Vec (2 * 112 * 112) → Vec 2) :=
    globalAvgPoolFlat_differentiable 2 112 112
  have hdense : Differentiable ℝ (dense Wh bh) := dense_differentiable Wh bh
  have hshift : Differentiable ℝ (fun u : Vec (2 * 112 * 112) => (fun k => (3 : ℝ) + u k)) := by fun_prop
  unfold fwdRS
  exact hdense.comp (hgap.comp (hbn.comp (hbn.comp (hbn.comp (hshift.comp (hbn.comp hconv))))))

-- ════════════════════════════════════════════════════════════════
-- § The uniform channel-0 perturbation and the carrier
-- ════════════════════════════════════════════════════════════════

/-- The uniform perturbation: the whole 1-channel input. -/
noncomputable def VR : Vec (1 * 112 * 112) := fun _ => 1

/-- **The asymmetric stem turns a uniform input perturbation `t` into the channel
    difference `−t`** (`conv` sends channel weights `1` and `2`, so `ch0 = t`, `ch1 = 2t`). -/
theorem conv_udiff (t : ℝ) : UDiff (-t) (flatConv (h := 112) (w := 112) Ws bs (t • VR)) := by
  have key : ∀ c : Fin 2, ∀ (i j : Fin 112),
      (Tensor3.unflatten (flatConv (h := 112) (w := 112) Ws bs (t • VR)) : Tensor3 2 112 112) c i j
        = (if c = 0 then (1 : ℝ) else 2) * t := by
    intro c i j
    show flatConv (h := 112) (w := 112) Ws bs (t • VR)
        (finProdFinEquiv (finProdFinEquiv (c, i), j)) = _
    show Tensor3.flatten (conv2d Ws bs (Tensor3.unflatten (t • VR)))
        (finProdFinEquiv (finProdFinEquiv (c, i), j)) = _
    rw [show Tensor3.flatten (conv2d Ws bs (Tensor3.unflatten (t • VR)))
          (finProdFinEquiv (finProdFinEquiv (c, i), j))
          = conv2d Ws bs (Tensor3.unflatten (t • VR)) c i j from by
        simp only [Tensor3.flatten, Equiv.symm_apply_apply]]
    rw [conv2d_1x1]
    simp only [bs, Ws, Fin.sum_univ_one, zero_add, Tensor3.unflatten, Pi.smul_apply, VR,
      smul_eq_mul, mul_one]
  intro i j
  rw [key 0 i j, key 1 i j]
  simp; ring

-- ════════════════════════════════════════════════════════════════
-- § The running activations and the positive istd product `Rr`
-- ════════════════════════════════════════════════════════════════

noncomputable def A0 (t : ℝ) : Vec (2 * 112 * 112) := flatConv (h := 112) (w := 112) Ws bs (t • VR)
noncomputable def A1 (t : ℝ) : Vec (2 * 112 * 112) := bnForward (2 * 112 * 112) 1 (1 / 128) 3 (A0 t)
noncomputable def A2 (t : ℝ) : Vec (2 * 112 * 112) := fun k => 3 + A1 t k
noncomputable def A3 (t : ℝ) : Vec (2 * 112 * 112) := bnForward (2 * 112 * 112) 1 (1 / 128) 3 (A2 t)
noncomputable def A4 (t : ℝ) : Vec (2 * 112 * 112) := bnForward (2 * 112 * 112) 1 (1 / 128) 3 (A3 t)

/-- The product of the four `γ·istd` factors (the positive nonlinear part). -/
noncomputable def Rr (t : ℝ) : ℝ :=
  (1 / 128 * bnIstd (2 * 112 * 112) (A0 t) 1)
  * ((1 / 128 * bnIstd (2 * 112 * 112) (A2 t) 1)
     * ((1 / 128 * bnIstd (2 * 112 * 112) (A3 t) 1)
        * (1 / 128 * bnIstd (2 * 112 * 112) (A4 t) 1)))

theorem Rr_pos (t : ℝ) : 0 < Rr t := by
  unfold Rr
  have hi : ∀ z : Vec (2 * 112 * 112), 0 < (1 / 128 : ℝ) * bnIstd (2 * 112 * 112) z 1 := fun z =>
    mul_pos (by norm_num) (bnIstd_pos z 1 one_pos)
  exact mul_pos (hi _) (mul_pos (hi _) (mul_pos (hi _) (hi _)))

/-- Adding a channel-symmetric constant on the LEFT preserves the uniform channel difference. -/
theorem UDiff_const_add {h w : Nat} (δ c : ℝ) (u : Vec (2 * h * w)) (hu : UDiff δ u) :
    UDiff δ (fun k => c + u k) := by
  intro i j
  have h0 : (Tensor3.unflatten (fun k => c + u k) : Tensor3 2 h w) 0 i j
      = c + (Tensor3.unflatten u) 0 i j := rfl
  have h1 : (Tensor3.unflatten (fun k => c + u k) : Tensor3 2 h w) 1 i j
      = c + (Tensor3.unflatten u) 1 i j := rfl
  rw [h0, h1, hu i j]; ring

-- ════════════════════════════════════════════════════════════════
-- § The output difference along the ray is `−t · Rr t`
-- ════════════════════════════════════════════════════════════════

theorem gd_ray_mnv2 (t : ℝ) : fwdRS (t • VR) 0 - fwdRS (t • VR) 1 = -t * Rr t := by
  have u0 : UDiff (-t) (A0 t) := conv_udiff t
  have u1 : UDiff (1 / 128 * -t * bnIstd (2 * 112 * 112) (A0 t) 1) (A1 t) :=
    UDiff_bn_γ (1 / 128) 3 (-t) (A0 t) u0
  have u2 : UDiff (1 / 128 * -t * bnIstd (2 * 112 * 112) (A0 t) 1) (A2 t) :=
    UDiff_const_add _ 3 _ u1
  have u3 := UDiff_bn_γ (1 / 128) 3 _ (A2 t) u2
  have u4 := UDiff_bn_γ (1 / 128) 3 _ (A3 t) u3
  have u5 := UDiff_bn_γ (1 / 128) 3 _ (A4 t) u4
  have hgap := UDiff_gap (h := 112) (w := 112) (by norm_num) (by norm_num) _ _ u5
  show dense Wh bh (globalAvgPoolFlat 2 112 112
    (bnForward (2 * 112 * 112) 1 (1 / 128) 3 (A4 t))) 0
      - dense Wh bh (globalAvgPoolFlat 2 112 112
    (bnForward (2 * 112 * 112) 1 (1 / 128) 3 (A4 t))) 1 = -t * Rr t
  rw [dense_Wh_apply, dense_Wh_apply, hgap]
  simp only [Rr]
  ring

-- ════════════════════════════════════════════════════════════════
-- § Continuity of `Rr`, the directional derivative, and the seal
-- ════════════════════════════════════════════════════════════════

theorem A0_continuous : Continuous A0 :=
  (flatConv_differentiable (h := 112) (w := 112) Ws bs).continuous.comp
    (continuous_id.smul continuous_const)
theorem A1_continuous : Continuous A1 :=
  (bnForward_differentiable (2 * 112 * 112) 1 (1 / 128) 3 one_pos).continuous.comp A0_continuous
theorem A2_continuous : Continuous A2 :=
  continuous_pi (fun k => continuous_const.add ((continuous_apply k).comp A1_continuous))
theorem A3_continuous : Continuous A3 :=
  (bnForward_differentiable (2 * 112 * 112) 1 (1 / 128) 3 one_pos).continuous.comp A2_continuous
theorem A4_continuous : Continuous A4 :=
  (bnForward_differentiable (2 * 112 * 112) 1 (1 / 128) 3 one_pos).continuous.comp A3_continuous

theorem Rr_continuous : Continuous Rr := by
  have istdc : ∀ (g : ℝ → Vec (2 * 112 * 112)), Continuous g →
      Continuous (fun t => (1 / 128 : ℝ) * bnIstd (2 * 112 * 112) (g t) 1) := fun g hg =>
    continuous_const.mul ((ResNet34LiveSeal.bnIstd_cont (n := 2 * 112 * 112) 1 one_pos ⟨0, by norm_num⟩).comp hg)
  exact (istdc _ A0_continuous).mul
    ((istdc _ A2_continuous).mul ((istdc _ A3_continuous).mul (istdc _ A4_continuous)))

theorem gd_hasDerivAt_mnv2 :
    HasDerivAt (fun t : ℝ => fwdRS (t • VR) 0 - fwdRS (t • VR) 1) (-Rr 0) 0 :=
  HasDerivAt.congr_of_eventuallyEq
    (hasDerivAt_mul_self_zero (Q := fun t => -Rr t) Rr_continuous.neg.continuousAt)
    (Filter.Eventually.of_forall (fun t => (gd_ray_mnv2 t).trans (neg_mul_comm t (Rr t))))

/-- **`fderiv ℝ fwdR 0 ≠ 0`** — the 224×224 live MobileNetV2's whole-net Jacobian is genuinely
    non-trivial at the witness input `0` (level-3 seal, realistic dims). -/
theorem fwdR_jacobian_nonzero : fderiv ℝ fwdR 0 ≠ 0 := by
  rw [show fwdR = fwdRS from funext fwdR_eq]
  -- the output channel difference along `t • VR` has derivative `-Rr 0 ≠ 0`
  exact fderiv_ne_zero_of_ray VR (fwdRS_differentiable 0) (fun y => y 0 - y 1) (by fun_prop)
    (neg_ne_zero.mpr (Rr_pos 0).ne') (by simpa only [zero_add] using gd_hasDerivAt_mnv2)

/-- **The level-3 seal for the 224×224 live MobileNetV2** (Item D, level 3): the proven
    whole-network backward of the nonzero-weight live MobileNetV2 at real ImageNet resolution
    is **not the zero map** at the witness input `0`. -/
theorem fwdR_backward_nontrivial :
    ∃ (j₀ : Fin 2) (i₀ : Fin (1 * 112 * 112)),
      (fwdR_has_vjp_at 0).backward (basisVec j₀) i₀ ≠ 0 :=
  (fwdR_has_vjp_at 0).backward_nontrivial_of_fderiv_ne fwdR_jacobian_nonzero

end Mnv2RealSeal
end Proofs
