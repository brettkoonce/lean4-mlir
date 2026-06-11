import LeanMlir.Proofs.Depthwise

/-!
# MobileNetV2 — end-to-end inverted-residual VJP (flattened Vec space)

Builds a representative MobileNetV2 forward and proves its end-to-end
vector–Jacobian product correct, analogous to `cnn_has_vjp_at` for the
ResNet basic block. Everything lives in flattened `Vec` space and reuses
the foundation rules from `CNN.lean`, `Depthwise.lean`, `BatchNorm.lean`,
`MLP.lean`, and `Residual.lean` through `vjp_comp_at` chaining.

## What's new here

* **`relu6`** — the clamped activation `min (max x 0) 6`. Unlike `relu`,
  the saturated branch is the *constant* 6 (not linear-through-origin),
  so the local-linearization trick matches `relu6` against an *affine*
  surrogate `g` (proj on the active region, constant 0 below, constant 6
  above) whose fderiv is the diagonal indicator `1_{0<x<6}`. Smoothness
  is the two-sided `x k ≠ 0 ∧ x k ≠ 6`.

* **Inverted-residual body** — `project ∘ depthwise ∘ expand`:
    - expand   : 1×1 conv → bn → relu6   (`ic → mid`, mid = t·ic)
    - depthwise: depthwise conv → bn → relu6   (`mid → mid`)
    - project  : 1×1 conv → bn   (linear bottleneck, no activation; `mid → oc`)
  With a stride-1 / `ic = oc` skip it becomes `residual body` (no final
  activation — MobileNetV2's linear-bottleneck design).

* **End-to-end** — stem (3×3 conv-bn-relu6) → skip inverted-residual →
  no-skip inverted-residual (channel change) → global average pool →
  dense head. Fixed block counts; generic channel/kernel dims; spatial
  `h w` preserved throughout (SAME convs).

All new defs/theorems certify to exactly `[propext, Classical.choice,
Quot.sound]`.
-/


namespace Proofs
open Finset BigOperators

-- ════════════════════════════════════════════════════════════════
-- § ReLU6  y = min(max(x,0), 6)   (MobileNetV2 activation)
-- ════════════════════════════════════════════════════════════════

noncomputable def relu6 (n : Nat) (x : Vec n) : Vec n :=
  fun i => min (max (x i) 0) 6

/-- ReLU6's local linear part at a smooth point: projects to `y k` when
    `0 < x k < 6`, otherwise zero. -/
noncomputable def relu6LinearPart (n : Nat) (x : Vec n) : Vec n →L[ℝ] Vec n :=
  ContinuousLinearMap.pi fun k =>
    if 0 < x k ∧ x k < 6 then ContinuousLinearMap.proj k else (0 : Vec n →L[ℝ] ℝ)

@[simp] theorem relu6LinearPart_apply (n : Nat) (x y : Vec n) (k : Fin n) :
    relu6LinearPart n x y k = if 0 < x k ∧ x k < 6 then y k else 0 := by
  show (ContinuousLinearMap.pi (fun k' =>
          if 0 < x k' ∧ x k' < 6 then ContinuousLinearMap.proj k'
                      else (0 : Vec n →L[ℝ] ℝ))) y k = _
  rw [ContinuousLinearMap.pi_apply]
  by_cases hxk : 0 < x k ∧ x k < 6
  · rw [if_pos hxk, if_pos hxk]; rfl
  · rw [if_neg hxk, if_neg hxk]; rfl

theorem relu6_hasFDerivAt (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0 ∧ x k ≠ 6) :
    HasFDerivAt (relu6 n) (relu6LinearPart n x) x := by
  rcases Nat.eq_zero_or_pos n with hn0 | hn_pos
  · subst hn0
    have h_eq : (relu6 0 : Vec 0 → Vec 0) = (⇑(relu6LinearPart 0 x) : Vec 0 → Vec 0) := by
      funext _ k; exact k.elim0
    rw [h_eq]; exact (relu6LinearPart 0 x).hasFDerivAt
  haveI : Nonempty (Fin n) := ⟨⟨0, hn_pos⟩⟩
  -- `g` is the locally-matching AFFINE function: identity-proj on the active
  -- region, constant 0 below and constant 6 above. Its fderiv is exactly
  -- `relu6LinearPart` (constant branches contribute 0). We show
  -- `HasFDerivAt g (relu6LinearPart) x` and `relu6 =ᶠ g` near x.
  let g : Vec n → Vec n := fun y k =>
    if 0 < x k ∧ x k < 6 then y k else (if x k ≤ 0 then 0 else 6)
  -- radius: distance to nearest kink (0 or 6) over all coordinates
  let r : ℝ := Finset.univ.inf' Finset.univ_nonempty
    (fun k : Fin n => min |x k| |x k - 6|)
  have hr_pos : 0 < r := by
    refine (Finset.lt_inf'_iff _).mpr ?_
    intro k _
    apply lt_min
    · exact abs_pos.mpr (h_smooth k).1
    · exact abs_pos.mpr (sub_ne_zero.mpr (h_smooth k).2)
  have hr_le : ∀ k : Fin n, r ≤ min |x k| |x k - 6| := fun k =>
    Finset.inf'_le _ (Finset.mem_univ k)
  -- Step A: g has fderiv relu6LinearPart at x.
  have hg_fderiv : HasFDerivAt g (relu6LinearPart n x) x := by
    unfold relu6LinearPart
    rw [hasFDerivAt_pi]
    intro k
    show HasFDerivAt (fun y : Vec n => g y k)
      ((ContinuousLinearMap.proj k).comp (ContinuousLinearMap.pi fun k' =>
        if 0 < x k' ∧ x k' < 6 then ContinuousLinearMap.proj k'
                    else (0 : Vec n →L[ℝ] ℝ))) x
    -- The component derivative `(proj k).comp(pi ...)` equals the branch CLM
    -- `if 0<x k<6 then proj k else 0`. Provide that simpler CLM and convert.
    have hcomp : (ContinuousLinearMap.proj k : Vec n →L[ℝ] ℝ).comp
          (ContinuousLinearMap.pi fun k' =>
            if 0 < x k' ∧ x k' < 6 then ContinuousLinearMap.proj k'
                        else (0 : Vec n →L[ℝ] ℝ)) =
        (if 0 < x k ∧ x k < 6 then (ContinuousLinearMap.proj k : Vec n →L[ℝ] ℝ)
          else 0) := by
      apply ContinuousLinearMap.ext; intro y
      show (ContinuousLinearMap.pi fun k' =>
            if 0 < x k' ∧ x k' < 6 then ContinuousLinearMap.proj k'
                        else (0 : Vec n →L[ℝ] ℝ)) y k =
        (if 0 < x k ∧ x k < 6 then (ContinuousLinearMap.proj k : Vec n →L[ℝ] ℝ) else 0) y
      rw [ContinuousLinearMap.pi_apply]
    rw [hcomp]
    by_cases hk : 0 < x k ∧ x k < 6
    · have h1 : (fun y : Vec n => g y k) = fun y : Vec n => y k := by
        funext y; show (if 0 < x k ∧ x k < 6 then y k else _) = y k; rw [if_pos hk]
      rw [h1, if_pos hk]
      exact (ContinuousLinearMap.proj k : Vec n →L[ℝ] ℝ).hasFDerivAt
    · have h1 : (fun y : Vec n => g y k) = fun _ : Vec n => (if x k ≤ 0 then (0:ℝ) else 6) := by
        funext y; show (if 0 < x k ∧ x k < 6 then y k else _) = _; rw [if_neg hk]
      rw [h1, if_neg hk]
      exact hasFDerivAt_const _ _
  -- Step B: relu6 =ᶠ g near x.
  have h_local : Set.EqOn (relu6 n) g (Metric.ball x r) := by
    intro y hy
    have hy_norm : ‖y - x‖ < r := by
      rw [Metric.mem_ball, dist_eq_norm] at hy; exact hy
    funext k
    have h_close : |y k - x k| < min |x k| |x k - 6| := by
      have h1 : |y k - x k| ≤ ‖y - x‖ := by
        have h2 : ‖(y - x) k‖ ≤ ‖y - x‖ := norm_le_pi_norm (y - x) k
        rw [Real.norm_eq_abs] at h2
        exact h2
      linarith [hr_le k]
    have h_close0 : |y k - x k| < |x k| := lt_of_lt_of_le h_close (min_le_left _ _)
    have h_close6 : |y k - x k| < |x k - 6| := lt_of_lt_of_le h_close (min_le_right _ _)
    show (relu6 n y) k = g y k
    show min (max (y k) 0) 6 = if 0 < x k ∧ x k < 6 then y k else (if x k ≤ 0 then 0 else 6)
    -- Three cases on x k: < 0, in (0,6), > 6.
    rcases lt_trichotomy (x k) 0 with hxneg | hx0 | hxpos
    · -- x k < 0: y k < 0 too; result 0; condition false
      have hyk_neg : y k < 0 := by
        have h_abs : |y k - x k| < -x k := by rwa [abs_of_neg hxneg] at h_close0
        have := (abs_lt.mp h_abs).2; linarith
      rw [if_neg (by rintro ⟨h, _⟩; linarith), if_pos (le_of_lt hxneg)]
      rw [max_eq_right (le_of_lt hyk_neg), min_eq_left (by linarith)]
    · exact absurd hx0 (h_smooth k).1
    · -- x k > 0. Subcase on x k vs 6.
      rcases lt_trichotomy (x k) 6 with hx6 | hx6 | hx6
      · -- 0 < x k < 6: y k in (0,6); result y k; condition true
        have hyk_pos : 0 < y k := by
          have h_abs : |y k - x k| < x k := by rwa [abs_of_pos hxpos] at h_close0
          have := (abs_lt.mp h_abs).1; linarith
        have hyk_lt6 : y k < 6 := by
          have h_abs : |y k - x k| < |x k - 6| := h_close6
          rw [abs_of_neg (by linarith : x k - 6 < 0)] at h_abs
          have := (abs_lt.mp h_abs).2; linarith
        rw [if_pos ⟨hxpos, hx6⟩]
        rw [max_eq_left (le_of_lt hyk_pos), min_eq_left (le_of_lt hyk_lt6)]
      · exact absurd hx6 (h_smooth k).2
      · -- x k > 6: y k > 6; result 6; condition false
        have hyk_gt6 : 6 < y k := by
          have h_abs : |y k - x k| < |x k - 6| := h_close6
          rw [abs_of_pos (by linarith : 0 < x k - 6)] at h_abs
          have := (abs_lt.mp h_abs).1; linarith
        rw [if_neg (by rintro ⟨_, h⟩; linarith), if_neg (by linarith)]
        rw [max_eq_left (by linarith : (0:ℝ) ≤ y k), min_eq_right (le_of_lt hyk_gt6)]
  have h_evt : (relu6 n) =ᶠ[nhds x] g :=
    h_local.eventuallyEq_of_mem (Metric.ball_mem_nhds x hr_pos)
  exact hg_fderiv.congr_of_eventuallyEq h_evt

theorem relu6_differentiableAt_of_smooth (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0 ∧ x k ≠ 6) : DifferentiableAt ℝ (relu6 n) x :=
  (relu6_hasFDerivAt n x h_smooth).differentiableAt

theorem pdiv_relu6 (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0 ∧ x k ≠ 6) (i j : Fin n) :
    pdiv (relu6 n) x i j =
      if i = j then (if 0 < x i ∧ x i < 6 then 1 else 0) else 0 := by
  rcases Nat.eq_zero_or_pos n with hn0 | hn_pos
  · subst hn0; exact i.elim0
  unfold pdiv
  rw [(relu6_hasFDerivAt n x h_smooth).fderiv, relu6LinearPart_apply, basisVec_apply]
  by_cases hij : i = j
  · subst hij; rw [if_pos rfl, if_pos rfl]
  · rw [if_neg (fun h : j = i => hij h.symm), if_neg hij]
    by_cases hxj : 0 < x j ∧ x j < 6
    · rw [if_pos hxj]
    · rw [if_neg hxj]

noncomputable def relu6_has_vjp_at (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0 ∧ x k ≠ 6) : HasVJPAt (relu6 n) x where
  backward dy i := if 0 < x i ∧ x i < 6 then dy i else 0
  correct := by
    intro dy i
    simp_rw [pdiv_relu6 n x h_smooth]
    rw [Finset.sum_eq_single i
        (fun j _ hne => by rw [if_neg (Ne.symm hne)]; ring)
        (fun h => absurd (Finset.mem_univ i) h)]
    rw [if_pos rfl]
    by_cases hxi : 0 < x i ∧ x i < 6
    · rw [if_pos hxi, if_pos hxi]; ring
    · rw [if_neg hxi, if_neg hxi]; ring

-- ════════════════════════════════════════════════════════════════
-- § Conv/Depthwise + BN + ReLU6 blocks (flat Vec space)
-- ════════════════════════════════════════════════════════════════

/-- **1×1 conv → bn → relu6** (expand stage / stem). Mirror of
    `convBnRelu_has_vjp_at` with relu6 in place of relu. -/
noncomputable def convBnRelu6_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, (bnForward (oc * h * w) ε γ β (flatConv W b v) k ≠ 0 ∧
                       bnForward (oc * h * w) ε γ β (flatConv W b v) k ≠ 6)) :
    HasVJPAt (relu6 (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConv W b) v := by
  have hconv_diff : Differentiable ℝ (flatConv W b : Vec (ic * h * w) → Vec (oc * h * w)) :=
    flatConv_differentiable W b
  have hbn_diff : Differentiable ℝ (bnForward (oc * h * w) ε γ β) :=
    bnForward_differentiable (oc * h * w) ε γ β hε
  have step1 : HasVJPAt (bnForward (oc * h * w) ε γ β ∘ flatConv W b) v :=
    vjp_comp_at (flatConv W b) (bnForward (oc * h * w) ε γ β) v
      (hconv_diff v) (hbn_diff _)
      ((hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).toHasVJPAt v)
      ((bn_has_vjp (oc * h * w) ε γ β hε).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnForward (oc * h * w) ε γ β ∘ flatConv W b) v :=
    DifferentiableAt.comp v (hbn_diff (flatConv W b v)) (hconv_diff v)
  exact vjp_comp_at (bnForward (oc * h * w) ε γ β ∘ flatConv W b)
    (relu6 (oc * h * w)) v
    step1_diff
    (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    step1
    (relu6_has_vjp_at (oc * h * w) _ h_smooth)

theorem convBnRelu6_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, (bnForward (oc * h * w) ε γ β (flatConv W b v) k ≠ 0 ∧
                       bnForward (oc * h * w) ε γ β (flatConv W b v) k ≠ 6)) :
    DifferentiableAt ℝ (relu6 (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConv W b) v := by
  have hinner : DifferentiableAt ℝ (bnForward (oc * h * w) ε γ β ∘ flatConv W b) v :=
    ((bnForward_differentiable (oc * h * w) ε γ β hε).comp (flatConv_differentiable W b)) v
  exact (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth).comp v hinner

/-- **Depthwise → bn → relu6** (depthwise stage of an inverted residual).
    Channels & spatial dims preserved: `Vec (c*h*w) → Vec (c*h*w)`. -/
noncomputable def dwBnRelu6_has_vjp_at {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (c * h * w))
    (h_smooth : ∀ k, (bnForward (c * h * w) ε γ β (depthwiseFlat W b v) k ≠ 0 ∧
                       bnForward (c * h * w) ε γ β (depthwiseFlat W b v) k ≠ 6)) :
    HasVJPAt (relu6 (c * h * w) ∘ bnForward (c * h * w) ε γ β ∘ depthwiseFlat W b) v := by
  have hdw_diff : Differentiable ℝ (depthwiseFlat W b : Vec (c * h * w) → Vec (c * h * w)) :=
    depthwiseFlat_differentiable W b
  have hbn_diff : Differentiable ℝ (bnForward (c * h * w) ε γ β) :=
    bnForward_differentiable (c * h * w) ε γ β hε
  have step1 : HasVJPAt (bnForward (c * h * w) ε γ β ∘ depthwiseFlat W b) v :=
    vjp_comp_at (depthwiseFlat W b) (bnForward (c * h * w) ε γ β) v
      (hdw_diff v) (hbn_diff _)
      ((depthwiseFlat_has_vjp W b).toHasVJPAt v)
      ((bn_has_vjp (c * h * w) ε γ β hε).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnForward (c * h * w) ε γ β ∘ depthwiseFlat W b) v :=
    DifferentiableAt.comp v (hbn_diff (depthwiseFlat W b v)) (hdw_diff v)
  exact vjp_comp_at (bnForward (c * h * w) ε γ β ∘ depthwiseFlat W b)
    (relu6 (c * h * w)) v
    step1_diff
    (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth)
    step1
    (relu6_has_vjp_at (c * h * w) _ h_smooth)

theorem dwBnRelu6_differentiableAt {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (c * h * w))
    (h_smooth : ∀ k, (bnForward (c * h * w) ε γ β (depthwiseFlat W b v) k ≠ 0 ∧
                       bnForward (c * h * w) ε γ β (depthwiseFlat W b v) k ≠ 6)) :
    DifferentiableAt ℝ (relu6 (c * h * w) ∘ bnForward (c * h * w) ε γ β ∘ depthwiseFlat W b) v := by
  have hinner : DifferentiableAt ℝ (bnForward (c * h * w) ε γ β ∘ depthwiseFlat W b) v :=
    ((bnForward_differentiable (c * h * w) ε γ β hε).comp (depthwiseFlat_differentiable W b)) v
  exact (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth).comp v hinner

/-- **conv → bn (no activation)** — the project (linear bottleneck) stage.
    Everywhere differentiable, global `HasVJP`. -/
noncomputable def convBn'_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε γ β : ℝ) (hε : 0 < ε) :
    HasVJP (bnForward (oc * h * w) ε γ β ∘ flatConv W b
      : Vec (ic * h * w) → Vec (oc * h * w)) :=
  vjp_comp (flatConv W b) (bnForward (oc * h * w) ε γ β)
    (flatConv_differentiable W b)
    (bnForward_differentiable (oc * h * w) ε γ β hε)
    (hasVJP3_to_hasVJP (conv2d_has_vjp3 W b))
    (bn_has_vjp (oc * h * w) ε γ β hε)

theorem convBn'_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (bnForward (oc * h * w) ε γ β ∘ flatConv W b
      : Vec (ic * h * w) → Vec (oc * h * w)) :=
  (bnForward_differentiable (oc * h * w) ε γ β hε).comp (flatConv_differentiable W b)

-- ════════════════════════════════════════════════════════════════
-- § Inverted-residual body  (MobileNetV2 core)
--   body = project(1×1 bn) ∘ depthwise(bn-relu6) ∘ expand(1×1 bn-relu6)
--   Channel flow:  ic → mid → mid → oc   (mid = t·ic = expansion).
--   Spatial dims h w preserved (stride-1 SAME conv).
-- ════════════════════════════════════════════════════════════════

/-- The expand stage as a flat map. -/
@[reducible] noncomputable def ivExpand {ic mid h w kHe kWe : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe γe βe : ℝ) :
    Vec (ic * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnForward (mid * h * w) εe γe βe ∘ flatConv We be

/-- The depthwise stage as a flat map. -/
@[reducible] noncomputable def ivDepthwise {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ) :
    Vec (mid * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnForward (mid * h * w) εd γd βd ∘ depthwiseFlat Wd bd

/-- The project (linear bottleneck) stage as a flat map. -/
@[reducible] noncomputable def ivProject {mid oc h w kHp kWp : Nat}
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp γp βp : ℝ) :
    Vec (mid * h * w) → Vec (oc * h * w) :=
  bnForward (oc * h * w) εp γp βp ∘ flatConv Wp bp

/-- **Inverted-residual body** = `project ∘ depthwise ∘ expand`. Flat
    `Vec (ic*h*w) → Vec (oc*h*w)`. -/
@[reducible] noncomputable def invresBody {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe γe βe : ℝ)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp γp βp : ℝ) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  ivProject (h := h) (w := w) Wp bp εp γp βp ∘
    (ivDepthwise (h := h) (w := w) Wd bd εd γd βd ∘
      ivExpand (h := h) (w := w) We be εe γe βe)

/-- **Inverted-residual body VJP at a smooth point.** Two `vjp_comp_at`
    chains: (1) `depthwise ∘ expand` over the two relu6 smoothness families,
    (2) `project` (everywhere) on top. -/
noncomputable def invresBody_has_vjp_at {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp γp βp : ℝ) (hεp : 0 < εp)
    (v : Vec (ic * h * w))
    (h_se : ∀ k, (bnForward (mid * h * w) εe γe βe (flatConv We be v) k ≠ 0 ∧
                   bnForward (mid * h * w) εe γe βe (flatConv We be v) k ≠ 6))
    (h_sd : ∀ k, (bnForward (mid * h * w) εd γd βd
                    (depthwiseFlat Wd bd (ivExpand (h := h) (w := w) We be εe γe βe v)) k ≠ 0 ∧
                   bnForward (mid * h * w) εd γd βd
                    (depthwiseFlat Wd bd (ivExpand (h := h) (w := w) We be εe γe βe v)) k ≠ 6)) :
    HasVJPAt (invresBody (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v := by
  -- expand
  have hexp_vjp : HasVJPAt (ivExpand (h := h) (w := w) We be εe γe βe) v :=
    convBnRelu6_has_vjp_at We be εe γe βe hεe v h_se
  have hexp_diff : DifferentiableAt ℝ (ivExpand (h := h) (w := w) We be εe γe βe) v :=
    convBnRelu6_differentiableAt We be εe γe βe hεe v h_se
  -- depthwise (at the expand output)
  have hdw_vjp : HasVJPAt (ivDepthwise (h := h) (w := w) Wd bd εd γd βd)
      (ivExpand (h := h) (w := w) We be εe γe βe v) :=
    dwBnRelu6_has_vjp_at Wd bd εd γd βd hεd _ h_sd
  have hdw_diff : DifferentiableAt ℝ (ivDepthwise (h := h) (w := w) Wd bd εd γd βd)
      (ivExpand (h := h) (w := w) We be εe γe βe v) :=
    dwBnRelu6_differentiableAt Wd bd εd γd βd hεd _ h_sd
  -- depthwise ∘ expand
  have hde_vjp : HasVJPAt
      (ivDepthwise (h := h) (w := w) Wd bd εd γd βd ∘
        ivExpand (h := h) (w := w) We be εe γe βe) v :=
    vjp_comp_at _ _ v hexp_diff hdw_diff hexp_vjp hdw_vjp
  have hde_diff : DifferentiableAt ℝ
      (ivDepthwise (h := h) (w := w) Wd bd εd γd βd ∘
        ivExpand (h := h) (w := w) We be εe γe βe) v :=
    hdw_diff.comp v hexp_diff
  -- project (everywhere)
  exact vjp_comp_at _ (ivProject (h := h) (w := w) Wp bp εp γp βp) v
    hde_diff
    ((convBn'_differentiable Wp bp εp γp βp hεp) _)
    hde_vjp
    ((convBn'_has_vjp Wp bp εp γp βp hεp).toHasVJPAt _)

theorem invresBody_differentiableAt {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp γp βp : ℝ) (hεp : 0 < εp)
    (v : Vec (ic * h * w))
    (h_se : ∀ k, (bnForward (mid * h * w) εe γe βe (flatConv We be v) k ≠ 0 ∧
                   bnForward (mid * h * w) εe γe βe (flatConv We be v) k ≠ 6))
    (h_sd : ∀ k, (bnForward (mid * h * w) εd γd βd
                    (depthwiseFlat Wd bd (ivExpand (h := h) (w := w) We be εe γe βe v)) k ≠ 0 ∧
                   bnForward (mid * h * w) εd γd βd
                    (depthwiseFlat Wd bd (ivExpand (h := h) (w := w) We be εe γe βe v)) k ≠ 6)) :
    DifferentiableAt ℝ (invresBody (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v := by
  have hexp_diff : DifferentiableAt ℝ (ivExpand (h := h) (w := w) We be εe γe βe) v :=
    convBnRelu6_differentiableAt We be εe γe βe hεe v h_se
  have hdw_diff : DifferentiableAt ℝ (ivDepthwise (h := h) (w := w) Wd bd εd γd βd)
      (ivExpand (h := h) (w := w) We be εe γe βe v) :=
    dwBnRelu6_differentiableAt Wd bd εd γd βd hεd _ h_sd
  exact ((convBn'_differentiable Wp bp εp γp βp hεp) _).comp v (hdw_diff.comp v hexp_diff)

/-- **Inverted-residual block WITH skip** (stride 1, `ic = oc = c`):
    `residual (invresBody)` — `body(x) + x`. No final activation
    (MobileNetV2 uses linear bottleneck; the project stage has no relu6,
    and the residual add is the block output). -/
noncomputable def invresSkip_has_vjp_at {c mid h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid c kHe kWe) (be : Vec mid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Wp : Kernel4 c mid kHp kWp) (bp : Vec c) (εp γp βp : ℝ) (hεp : 0 < εp)
    (v : Vec (c * h * w))
    (h_se : ∀ k, (bnForward (mid * h * w) εe γe βe (flatConv We be v) k ≠ 0 ∧
                   bnForward (mid * h * w) εe γe βe (flatConv We be v) k ≠ 6))
    (h_sd : ∀ k, (bnForward (mid * h * w) εd γd βd
                    (depthwiseFlat Wd bd (ivExpand (h := h) (w := w) We be εe γe βe v)) k ≠ 0 ∧
                   bnForward (mid * h * w) εd γd βd
                    (depthwiseFlat Wd bd (ivExpand (h := h) (w := w) We be εe γe βe v)) k ≠ 6)) :
    HasVJPAt (residual (invresBody (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp)) v := by
  have hF_diff : DifferentiableAt ℝ
      (invresBody (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v :=
    invresBody_differentiableAt We be εe γe βe hεe Wd bd εd γd βd hεd Wp bp εp γp βp hεp v h_se h_sd
  have hF : HasVJPAt
      (invresBody (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v :=
    invresBody_has_vjp_at We be εe γe βe hεe Wd bd εd γd βd hεd Wp bp εp γp βp hεp v h_se h_sd
  exact residual_has_vjp_at _ v hF_diff hF

theorem invresSkip_differentiableAt {c mid h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid c kHe kWe) (be : Vec mid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Wp : Kernel4 c mid kHp kWp) (bp : Vec c) (εp γp βp : ℝ) (hεp : 0 < εp)
    (v : Vec (c * h * w))
    (h_se : ∀ k, (bnForward (mid * h * w) εe γe βe (flatConv We be v) k ≠ 0 ∧
                   bnForward (mid * h * w) εe γe βe (flatConv We be v) k ≠ 6))
    (h_sd : ∀ k, (bnForward (mid * h * w) εd γd βd
                    (depthwiseFlat Wd bd (ivExpand (h := h) (w := w) We be εe γe βe v)) k ≠ 0 ∧
                   bnForward (mid * h * w) εd γd βd
                    (depthwiseFlat Wd bd (ivExpand (h := h) (w := w) We be εe γe βe v)) k ≠ 6)) :
    DifferentiableAt ℝ (residual (invresBody (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp)) v := by
  have hF_diff : DifferentiableAt ℝ
      (invresBody (h := h) (w := w) We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v :=
    invresBody_differentiableAt We be εe γe βe hεe Wd bd εd γd βd hεd Wp bp εp γp βp hεp v h_se h_sd
  show DifferentiableAt ℝ (biPath _ (fun x => x)) v
  exact DifferentiableAt.add hF_diff differentiable_id.differentiableAt

-- ════════════════════════════════════════════════════════════════
-- § End-to-end representative MobileNetV2
--
--   stem    : convBnRelu6        ic → c          (3×3, SAME, h×w)
--   block1  : residual(invresBody)  c → mid₁ → c (skip; stride-1 IR)
--   block2  : invresBody           c → mid₂ → oc (no skip; channel change)
--   gap      : globalAvgPool        oc, h×w → Vec oc
--   head     : dense               oc → nClasses
--
--   Fixed block counts (one skip IR + one no-skip IR). Spatial dims h w
--   kept constant (SAME depthwise/pointwise convs preserve them). All
--   channel/kernel dims are implicit Nat params. mid₁/mid₂ are the
--   expansion widths (t·in). The chain mirrors `cnn_has_vjp_at` with
--   inverted-residual blocks in place of basic resblocks.
-- ════════════════════════════════════════════════════════════════

/-- The forward MobileNetV2. -/
noncomputable def mobilenetv2Forward
    {ic c mid₁ oc mid₂ h w kHs kWs
     kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
     kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ nClasses : Nat}
    -- stem
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    -- block1 (skip IR: c → mid₁ → c)
    (We₁ : Kernel4 mid₁ c kHe₁ kWe₁) (be₁ : Vec mid₁) (e₁ ge₁ be1 : ℝ)
    (Wd₁ : DepthwiseKernel mid₁ kHd₁ kWd₁) (bd₁ : Vec mid₁) (d₁ gd₁ bd1 : ℝ)
    (Wp₁ : Kernel4 c mid₁ kHp₁ kWp₁) (bp₁ : Vec c) (p₁ gp₁ bp1 : ℝ)
    -- block2 (no-skip IR: c → mid₂ → oc)
    (We₂ : Kernel4 mid₂ c kHe₂ kWe₂) (be₂ : Vec mid₂) (e₂ ge₂ be2 : ℝ)
    (Wd₂ : DepthwiseKernel mid₂ kHd₂ kWd₂) (bd₂ : Vec mid₂) (d₂ gd₂ bd2 : ℝ)
    (Wp₂ : Kernel4 oc mid₂ kHp₂ kWp₂) (bp₂ : Vec oc) (p₂ gp₂ bp2 : ℝ)
    -- head
    (Wh : Mat oc nClasses) (bh : Vec nClasses) :
    Vec (ic * h * w) → Vec nClasses :=
  (dense Wh bh) ∘
  (globalAvgPoolFlat oc h w) ∘
  (invresBody (h := h) (w := w) We₂ be₂ e₂ ge₂ be2 Wd₂ bd₂ d₂ gd₂ bd2 Wp₂ bp₂ p₂ gp₂ bp2) ∘
  (residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1)) ∘
  (relu6 (c * h * w) ∘ bnForward (c * h * w) εs γs βs ∘ flatConv Ws bs)

/-- **MobileNetV2 end-to-end VJP at a smooth point.** Chains the stem,
    a skip inverted-residual, a no-skip inverted-residual, global avg
    pool, and dense head with `vjp_comp_at` under one bundled smoothness
    family (one `≠0∧≠6` hypothesis per relu6 site, evaluated at the
    running activation). -/
noncomputable def mobilenetv2_has_vjp_at
    {ic c mid₁ oc mid₂ h w kHs kWs
     kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
     kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (We₁ : Kernel4 mid₁ c kHe₁ kWe₁) (be₁ : Vec mid₁) (e₁ ge₁ be1 : ℝ) (he₁ : 0 < e₁)
    (Wd₁ : DepthwiseKernel mid₁ kHd₁ kWd₁) (bd₁ : Vec mid₁) (d₁ gd₁ bd1 : ℝ) (hd₁ : 0 < d₁)
    (Wp₁ : Kernel4 c mid₁ kHp₁ kWp₁) (bp₁ : Vec c) (p₁ gp₁ bp1 : ℝ) (hp₁ : 0 < p₁)
    (We₂ : Kernel4 mid₂ c kHe₂ kWe₂) (be₂ : Vec mid₂) (e₂ ge₂ be2 : ℝ) (he₂ : 0 < e₂)
    (Wd₂ : DepthwiseKernel mid₂ kHd₂ kWd₂) (bd₂ : Vec mid₂) (d₂ gd₂ bd2 : ℝ) (hd₂ : 0 < d₂)
    (Wp₂ : Kernel4 oc mid₂ kHp₂ kWp₂) (bp₂ : Vec oc) (p₂ gp₂ bp2 : ℝ) (hp₂ : 0 < p₂)
    (Wh : Mat oc nClasses) (bh : Vec nClasses)
    (x : Vec (ic * h * w))
    -- stem relu6 smoothness
    (h_stem : ∀ k, (bnForward (c * h * w) εs γs βs (flatConv Ws bs x) k ≠ 0 ∧
                     bnForward (c * h * w) εs γs βs (flatConv Ws bs x) k ≠ 6))
    -- block1 expand / depthwise relu6 smoothness (at the stem output)
    (h_b1e : ∀ k, (bnForward (mid₁ * h * w) e₁ ge₁ be1
        (flatConv We₁ be₁
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)) k ≠ 0 ∧
                   bnForward (mid₁ * h * w) e₁ ge₁ be1
        (flatConv We₁ be₁
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)) k ≠ 6))
    (h_b1d : ∀ k, (bnForward (mid₁ * h * w) d₁ gd₁ bd1
        (depthwiseFlat Wd₁ bd₁ (ivExpand (h := h) (w := w) We₁ be₁ e₁ ge₁ be1
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 0 ∧
                   bnForward (mid₁ * h * w) d₁ gd₁ bd1
        (depthwiseFlat Wd₁ bd₁ (ivExpand (h := h) (w := w) We₁ be₁ e₁ ge₁ be1
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 6))
    -- block2 expand / depthwise relu6 smoothness (at block1 output)
    (h_b2e : ∀ k, (bnForward (mid₂ * h * w) e₂ ge₂ be2
        (flatConv We₂ be₂
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 0 ∧
                   bnForward (mid₂ * h * w) e₂ ge₂ be2
        (flatConv We₂ be₂
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 6))
    (h_b2d : ∀ k, (bnForward (mid₂ * h * w) d₂ gd₂ bd2
        (depthwiseFlat Wd₂ bd₂ (ivExpand (h := h) (w := w) We₂ be₂ e₂ ge₂ be2
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)))) k ≠ 0 ∧
                   bnForward (mid₂ * h * w) d₂ gd₂ bd2
        (depthwiseFlat Wd₂ bd₂ (ivExpand (h := h) (w := w) We₂ be₂ e₂ ge₂ be2
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)))) k ≠ 6)) :
    HasVJPAt (mobilenetv2Forward Ws bs εs γs βs
      We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1
      We₂ be₂ e₂ ge₂ be2 Wd₂ bd₂ d₂ gd₂ bd2 Wp₂ bp₂ p₂ gp₂ bp2 Wh bh) x := by
  unfold mobilenetv2Forward
  -- s0: stem
  set S0 := (relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) with hS0
  have s0_vjp : HasVJPAt S0 x := convBnRelu6_has_vjp_at Ws bs εs γs βs hεs x h_stem
  have s0_diff : DifferentiableAt ℝ S0 x := convBnRelu6_differentiableAt Ws bs εs γs βs hεs x h_stem
  -- s1: block1 (skip IR) ∘ S0
  set B1 := residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1) with hB1
  have b1_vjp : HasVJPAt B1 (S0 x) :=
    invresSkip_has_vjp_at We₁ be₁ e₁ ge₁ be1 he₁ Wd₁ bd₁ d₁ gd₁ bd1 hd₁ Wp₁ bp₁ p₁ gp₁ bp1 hp₁
      (S0 x) h_b1e h_b1d
  have b1_diff : DifferentiableAt ℝ B1 (S0 x) :=
    invresSkip_differentiableAt We₁ be₁ e₁ ge₁ be1 he₁ Wd₁ bd₁ d₁ gd₁ bd1 hd₁ Wp₁ bp₁ p₁ gp₁ bp1 hp₁
      (S0 x) h_b1e h_b1d
  have s1_vjp : HasVJPAt (B1 ∘ S0) x := vjp_comp_at S0 B1 x s0_diff b1_diff s0_vjp b1_vjp
  have s1_diff : DifferentiableAt ℝ (B1 ∘ S0) x := b1_diff.comp x s0_diff
  -- s2: block2 (no-skip IR) ∘ (above)
  set B2 := invresBody (h := h) (w := w) We₂ be₂ e₂ ge₂ be2 Wd₂ bd₂ d₂ gd₂ bd2 Wp₂ bp₂ p₂ gp₂ bp2 with hB2
  have b2_vjp : HasVJPAt B2 (B1 (S0 x)) :=
    invresBody_has_vjp_at We₂ be₂ e₂ ge₂ be2 he₂ Wd₂ bd₂ d₂ gd₂ bd2 hd₂ Wp₂ bp₂ p₂ gp₂ bp2 hp₂
      (B1 (S0 x)) h_b2e h_b2d
  have b2_diff : DifferentiableAt ℝ B2 (B1 (S0 x)) :=
    invresBody_differentiableAt We₂ be₂ e₂ ge₂ be2 he₂ Wd₂ bd₂ d₂ gd₂ bd2 hd₂ Wp₂ bp₂ p₂ gp₂ bp2 hp₂
      (B1 (S0 x)) h_b2e h_b2d
  have s2_vjp : HasVJPAt (B2 ∘ (B1 ∘ S0)) x := vjp_comp_at (B1 ∘ S0) B2 x s1_diff b2_diff s1_vjp b2_vjp
  have s2_diff : DifferentiableAt ℝ (B2 ∘ (B1 ∘ S0)) x := b2_diff.comp x s1_diff
  -- s3: gap ∘ (above)
  set P := B2 ∘ (B1 ∘ S0) with hP
  have gap_diff : DifferentiableAt ℝ (globalAvgPoolFlat oc h w) (P x) :=
    (globalAvgPoolFlat_differentiable oc h w) (P x)
  have s3_vjp : HasVJPAt (globalAvgPoolFlat oc h w ∘ P) x :=
    vjp_comp_at P (globalAvgPoolFlat oc h w) x s2_diff gap_diff s2_vjp
      ((globalAvgPoolFlat_has_vjp oc h w).toHasVJPAt (P x))
  have s3_diff : DifferentiableAt ℝ (globalAvgPoolFlat oc h w ∘ P) x := gap_diff.comp x s2_diff
  -- s4: dense head
  exact vjp_comp_at (globalAvgPoolFlat oc h w ∘ P) (dense Wh bh) x s3_diff
    ((dense_differentiable Wh bh) _) s3_vjp
    ((dense_has_vjp Wh bh).toHasVJPAt _)

/-- **Public correctness theorem for `mobilenetv2_has_vjp_at`** — exposes
    the witness's `.correct` field: the full MobileNetV2 backward equals
    the `pdiv`-contracted Jacobian (Jacobian-transpose applied to the
    cotangent). MobileNetV2 analogue of `cnn_has_vjp_at_correct`. -/
theorem mobilenetv2_has_vjp_at_correct
    {ic c mid₁ oc mid₂ h w kHs kWs
     kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
     kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (We₁ : Kernel4 mid₁ c kHe₁ kWe₁) (be₁ : Vec mid₁) (e₁ ge₁ be1 : ℝ) (he₁ : 0 < e₁)
    (Wd₁ : DepthwiseKernel mid₁ kHd₁ kWd₁) (bd₁ : Vec mid₁) (d₁ gd₁ bd1 : ℝ) (hd₁ : 0 < d₁)
    (Wp₁ : Kernel4 c mid₁ kHp₁ kWp₁) (bp₁ : Vec c) (p₁ gp₁ bp1 : ℝ) (hp₁ : 0 < p₁)
    (We₂ : Kernel4 mid₂ c kHe₂ kWe₂) (be₂ : Vec mid₂) (e₂ ge₂ be2 : ℝ) (he₂ : 0 < e₂)
    (Wd₂ : DepthwiseKernel mid₂ kHd₂ kWd₂) (bd₂ : Vec mid₂) (d₂ gd₂ bd2 : ℝ) (hd₂ : 0 < d₂)
    (Wp₂ : Kernel4 oc mid₂ kHp₂ kWp₂) (bp₂ : Vec oc) (p₂ gp₂ bp2 : ℝ) (hp₂ : 0 < p₂)
    (Wh : Mat oc nClasses) (bh : Vec nClasses)
    (x : Vec (ic * h * w))
    (h_stem : ∀ k, (bnForward (c * h * w) εs γs βs (flatConv Ws bs x) k ≠ 0 ∧
                     bnForward (c * h * w) εs γs βs (flatConv Ws bs x) k ≠ 6))
    (h_b1e : ∀ k, (bnForward (mid₁ * h * w) e₁ ge₁ be1
        (flatConv We₁ be₁
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)) k ≠ 0 ∧
                   bnForward (mid₁ * h * w) e₁ ge₁ be1
        (flatConv We₁ be₁
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)) k ≠ 6))
    (h_b1d : ∀ k, (bnForward (mid₁ * h * w) d₁ gd₁ bd1
        (depthwiseFlat Wd₁ bd₁ (ivExpand (h := h) (w := w) We₁ be₁ e₁ ge₁ be1
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 0 ∧
                   bnForward (mid₁ * h * w) d₁ gd₁ bd1
        (depthwiseFlat Wd₁ bd₁ (ivExpand (h := h) (w := w) We₁ be₁ e₁ ge₁ be1
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 6))
    (h_b2e : ∀ k, (bnForward (mid₂ * h * w) e₂ ge₂ be2
        (flatConv We₂ be₂
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 0 ∧
                   bnForward (mid₂ * h * w) e₂ ge₂ be2
        (flatConv We₂ be₂
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 6))
    (h_b2d : ∀ k, (bnForward (mid₂ * h * w) d₂ gd₂ bd2
        (depthwiseFlat Wd₂ bd₂ (ivExpand (h := h) (w := w) We₂ be₂ e₂ ge₂ be2
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)))) k ≠ 0 ∧
                   bnForward (mid₂ * h * w) d₂ gd₂ bd2
        (depthwiseFlat Wd₂ bd₂ (ivExpand (h := h) (w := w) We₂ be₂ e₂ ge₂ be2
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)))) k ≠ 6))
    (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (mobilenetv2_has_vjp_at Ws bs εs γs βs hεs
        We₁ be₁ e₁ ge₁ be1 he₁ Wd₁ bd₁ d₁ gd₁ bd1 hd₁ Wp₁ bp₁ p₁ gp₁ bp1 hp₁
        We₂ be₂ e₂ ge₂ be2 he₂ Wd₂ bd₂ d₂ gd₂ bd2 hd₂ Wp₂ bp₂ p₂ gp₂ bp2 hp₂ Wh bh
        x h_stem h_b1e h_b1d h_b2e h_b2d).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (mobilenetv2Forward Ws bs εs γs βs
                We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1
                We₂ be₂ e₂ ge₂ be2 Wd₂ bd₂ d₂ gd₂ bd2 Wp₂ bp₂ p₂ gp₂ bp2 Wh bh)
             x i j * dy j :=
  (mobilenetv2_has_vjp_at Ws bs εs γs βs hεs
      We₁ be₁ e₁ ge₁ be1 he₁ Wd₁ bd₁ d₁ gd₁ bd1 hd₁ Wp₁ bp₁ p₁ gp₁ bp1 hp₁
      We₂ be₂ e₂ ge₂ be2 he₂ Wd₂ bd₂ d₂ gd₂ bd2 hd₂ Wp₂ bp₂ p₂ gp₂ bp2 hp₂ Wh bh
      x h_stem h_b1e h_b1d h_b2e h_b2d).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § Strided (downsampling) inverted-residual infrastructure
--
--   The representative `mobilenetv2Forward` above keeps spatial dims
--   constant (SAME convs). The *real* MobileNetV2 render downsamples with
--   a stride-2 depthwise inside the strided blocks (and a stride-2 stem).
--   These mirror the SAME op-level lemmas with `flatConvStride2` /
--   `depthwiseStride2Flat` (input spatial `2h×2w`, output `h×w`); the
--   expand/project 1×1s stay SAME (at the input resp. output resolution).
-- ════════════════════════════════════════════════════════════════

/-- **Stride-2 conv → bn → relu6** (the strided stem). Strided mirror of
    `convBnRelu6_has_vjp_at` with `flatConvStride2`; input spatial halves
    (`2h×2w → h×w`). -/
noncomputable def convBnRelu6Strided_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnForward (oc * h * w) ε γ β (flatConvStride2 W b v) k ≠ 0 ∧
                       bnForward (oc * h * w) ε γ β (flatConvStride2 W b v) k ≠ 6)) :
    HasVJPAt (relu6 (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v := by
  have hconv_diff : Differentiable ℝ (flatConvStride2 W b
      : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W b
  have hbn_diff : Differentiable ℝ (bnForward (oc * h * w) ε γ β) :=
    bnForward_differentiable (oc * h * w) ε γ β hε
  have step1 : HasVJPAt (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v :=
    vjp_comp_at (flatConvStride2 W b) (bnForward (oc * h * w) ε γ β) v
      (hconv_diff v) (hbn_diff _)
      ((flatConvStride2_has_vjp W b).toHasVJPAt v)
      ((bn_has_vjp (oc * h * w) ε γ β hε).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v :=
    DifferentiableAt.comp v (hbn_diff (flatConvStride2 W b v)) (hconv_diff v)
  exact vjp_comp_at (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b)
    (relu6 (oc * h * w)) v
    step1_diff
    (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    step1
    (relu6_has_vjp_at (oc * h * w) _ h_smooth)

theorem convBnRelu6Strided_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnForward (oc * h * w) ε γ β (flatConvStride2 W b v) k ≠ 0 ∧
                       bnForward (oc * h * w) ε γ β (flatConvStride2 W b v) k ≠ 6)) :
    DifferentiableAt ℝ
      (relu6 (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v := by
  have hinner : DifferentiableAt ℝ (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v :=
    ((bnForward_differentiable (oc * h * w) ε γ β hε).comp (flatConvStride2_differentiable W b)) v
  exact (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth).comp v hinner

/-- The strided depthwise stage as a flat map (`Vec (mid*(2h)*(2w)) → Vec (mid*h*w)`). -/
@[reducible] noncomputable def ivDepthwiseStrided {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ) :
    Vec (mid * (2 * h) * (2 * w)) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnForward (mid * h * w) εd γd βd ∘ depthwiseStride2Flat Wd bd

/-- **Stride-2 depthwise → bn → relu6** (downsampling depthwise stage). Strided
    mirror of `dwBnRelu6_has_vjp_at` with `depthwiseStride2Flat`. -/
noncomputable def dwBnRelu6Strided_has_vjp_at {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (c * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnForward (c * h * w) ε γ β (depthwiseStride2Flat W b v) k ≠ 0 ∧
                       bnForward (c * h * w) ε γ β (depthwiseStride2Flat W b v) k ≠ 6)) :
    HasVJPAt (relu6 (c * h * w) ∘ bnForward (c * h * w) ε γ β ∘ depthwiseStride2Flat W b) v := by
  have hdw_diff : Differentiable ℝ (depthwiseStride2Flat W b
      : Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w)) :=
    depthwiseStride2Flat_differentiable W b
  have hbn_diff : Differentiable ℝ (bnForward (c * h * w) ε γ β) :=
    bnForward_differentiable (c * h * w) ε γ β hε
  have step1 : HasVJPAt (bnForward (c * h * w) ε γ β ∘ depthwiseStride2Flat W b) v :=
    vjp_comp_at (depthwiseStride2Flat W b) (bnForward (c * h * w) ε γ β) v
      (hdw_diff v) (hbn_diff _)
      ((depthwiseStride2Flat_has_vjp W b).toHasVJPAt v)
      ((bn_has_vjp (c * h * w) ε γ β hε).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnForward (c * h * w) ε γ β ∘ depthwiseStride2Flat W b) v :=
    DifferentiableAt.comp v (hbn_diff (depthwiseStride2Flat W b v)) (hdw_diff v)
  exact vjp_comp_at (bnForward (c * h * w) ε γ β ∘ depthwiseStride2Flat W b)
    (relu6 (c * h * w)) v
    step1_diff
    (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth)
    step1
    (relu6_has_vjp_at (c * h * w) _ h_smooth)

theorem dwBnRelu6Strided_differentiableAt {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (c * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnForward (c * h * w) ε γ β (depthwiseStride2Flat W b v) k ≠ 0 ∧
                       bnForward (c * h * w) ε γ β (depthwiseStride2Flat W b v) k ≠ 6)) :
    DifferentiableAt ℝ
      (relu6 (c * h * w) ∘ bnForward (c * h * w) ε γ β ∘ depthwiseStride2Flat W b) v := by
  have hinner : DifferentiableAt ℝ (bnForward (c * h * w) ε γ β ∘ depthwiseStride2Flat W b) v :=
    ((bnForward_differentiable (c * h * w) ε γ β hε).comp (depthwiseStride2Flat_differentiable W b)) v
  exact (relu6_differentiableAt_of_smooth (c * h * w) _ h_smooth).comp v hinner

/-- **Strided inverted-residual body** = `project ∘ depthwiseStrided ∘ expand`.
    Expand is SAME at the input resolution (`2h×2w`); the stride-2 depthwise
    halves spatial (`2h×2w → h×w`); project is SAME at the output resolution.
    Flat `Vec (ic*(2h)*(2w)) → Vec (oc*h*w)`. (No skip: strided blocks change
    spatial / channels, so MobileNetV2 never wraps them in a residual.) -/
@[reducible] noncomputable def invresBodyStrided
    {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe γe βe : ℝ)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp γp βp : ℝ) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  ivProject (h := h) (w := w) Wp bp εp γp βp ∘
    (ivDepthwiseStrided (h := h) (w := w) Wd bd εd γd βd ∘
      ivExpand (h := 2 * h) (w := 2 * w) We be εe γe βe)

/-- **Strided inverted-residual body VJP at a smooth point.** Strided mirror of
    `invresBody_has_vjp_at`: expand SAME (at `2h×2w`) → depthwise-strided → project. -/
noncomputable def invresBodyStrided_has_vjp_at
    {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp γp βp : ℝ) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_se : ∀ k, (bnForward (mid * (2*h) * (2*w)) εe γe βe (flatConv We be v) k ≠ 0 ∧
                   bnForward (mid * (2*h) * (2*w)) εe γe βe (flatConv We be v) k ≠ 6))
    (h_sd : ∀ k, (bnForward (mid * h * w) εd γd βd
                    (depthwiseStride2Flat Wd bd
                      (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe v)) k ≠ 0 ∧
                   bnForward (mid * h * w) εd γd βd
                    (depthwiseStride2Flat Wd bd
                      (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe v)) k ≠ 6)) :
    HasVJPAt (invresBodyStrided (h := h) (w := w)
      We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v := by
  -- expand (SAME at 2h×2w)
  have hexp_vjp : HasVJPAt (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe) v :=
    convBnRelu6_has_vjp_at (h := 2*h) (w := 2*w) We be εe γe βe hεe v h_se
  have hexp_diff : DifferentiableAt ℝ (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe) v :=
    convBnRelu6_differentiableAt (h := 2*h) (w := 2*w) We be εe γe βe hεe v h_se
  -- depthwise strided (at the expand output)
  have hdw_vjp : HasVJPAt (ivDepthwiseStrided (h := h) (w := w) Wd bd εd γd βd)
      (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe v) :=
    dwBnRelu6Strided_has_vjp_at Wd bd εd γd βd hεd _ h_sd
  have hdw_diff : DifferentiableAt ℝ (ivDepthwiseStrided (h := h) (w := w) Wd bd εd γd βd)
      (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe v) :=
    dwBnRelu6Strided_differentiableAt Wd bd εd γd βd hεd _ h_sd
  -- depthwise ∘ expand
  have hde_vjp : HasVJPAt
      (ivDepthwiseStrided (h := h) (w := w) Wd bd εd γd βd ∘
        ivExpand (h := 2*h) (w := 2*w) We be εe γe βe) v :=
    vjp_comp_at _ _ v hexp_diff hdw_diff hexp_vjp hdw_vjp
  have hde_diff : DifferentiableAt ℝ
      (ivDepthwiseStrided (h := h) (w := w) Wd bd εd γd βd ∘
        ivExpand (h := 2*h) (w := 2*w) We be εe γe βe) v :=
    hdw_diff.comp v hexp_diff
  -- project (everywhere)
  exact vjp_comp_at _ (ivProject (h := h) (w := w) Wp bp εp γp βp) v
    hde_diff
    ((convBn'_differentiable Wp bp εp γp βp hεp) _)
    hde_vjp
    ((convBn'_has_vjp Wp bp εp γp βp hεp).toHasVJPAt _)

theorem invresBodyStrided_differentiableAt
    {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp γp βp : ℝ) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_se : ∀ k, (bnForward (mid * (2*h) * (2*w)) εe γe βe (flatConv We be v) k ≠ 0 ∧
                   bnForward (mid * (2*h) * (2*w)) εe γe βe (flatConv We be v) k ≠ 6))
    (h_sd : ∀ k, (bnForward (mid * h * w) εd γd βd
                    (depthwiseStride2Flat Wd bd
                      (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe v)) k ≠ 0 ∧
                   bnForward (mid * h * w) εd γd βd
                    (depthwiseStride2Flat Wd bd
                      (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe v)) k ≠ 6)) :
    DifferentiableAt ℝ (invresBodyStrided (h := h) (w := w)
      We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp) v := by
  have hexp_diff : DifferentiableAt ℝ (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe) v :=
    convBnRelu6_differentiableAt (h := 2*h) (w := 2*w) We be εe γe βe hεe v h_se
  have hdw_diff : DifferentiableAt ℝ (ivDepthwiseStrided (h := h) (w := w) Wd bd εd γd βd)
      (ivExpand (h := 2*h) (w := 2*w) We be εe γe βe v) :=
    dwBnRelu6Strided_differentiableAt Wd bd εd γd βd hεd _ h_sd
  exact ((convBn'_differentiable Wp bp εp γp βp hεp) _).comp v (hdw_diff.comp v hexp_diff)

-- ════════════════════════════════════════════════════════════════
-- § The full MobileNetV2 render — `mobilenetv2Forward_full`
--
--   The faithful 6-inverted-residual-block net the trainer actually runs
--   (`mobilenetv2Verified`, ch7), at the real Imagenette-224² channel flow
--   and spatial schedule:
--
--     stem  3→16   3×3 s2            224→112
--     b1   16→64→24  s2 (no skip)    112→56
--     b2   24→96→24  s1 (skip)       @56
--     b3   24→96→32  s2 (no skip)    56→28
--     b4   32→128→32 s1 (skip)       @28
--     b5   32→128→64 s2 (no skip)    28→14
--     b6   64→256→64 s2 (no skip)    14→7
--     head 64→128   1×1 s1           @7   (conv-bn-relu6)
--     gap → dense 128→10
--
--   Stated gap (intrinsic, shared with every BN net in this repo): the
--   `bnForward` here is SCALAR-global (one γ/β over c·h·w per example); the
--   render uses per-channel `[c]` BN. The block topology, channel flow,
--   stride schedule, relu6 sites and residual placement are all faithful.
-- ════════════════════════════════════════════════════════════════

/-- The full MobileNetV2 forward (ch7 render): stem-s2 → 6 inverted-residual
    blocks (`b1/b3/b5/b6` stride-2 downsample, `b2/b4` stride-1 skip) → 1×1
    conv-bn-relu6 head → global-avg-pool → dense. Scalar BN; faithful topology. -/
noncomputable def mobilenetv2Forward_full
    -- stem (3→16, 3×3 s2): 224→112
    (Ws : Kernel4 16 3 3 3) (bs : Vec 16) (εs γs βs : ℝ)
    -- b1 (16→64→24, s2): 112→56
    (We1 : Kernel4 64 16 1 1) (be1 : Vec 64) (εe1 γe1 βe1 : ℝ)
    (Wd1 : DepthwiseKernel 64 3 3) (bd1 : Vec 64) (εd1 γd1 βd1 : ℝ)
    (Wp1 : Kernel4 24 64 1 1) (bp1 : Vec 24) (εp1 γp1 βp1 : ℝ)
    -- b2 (24→96→24, s1 skip): @56
    (We2 : Kernel4 96 24 1 1) (be2 : Vec 96) (εe2 γe2 βe2 : ℝ)
    (Wd2 : DepthwiseKernel 96 3 3) (bd2 : Vec 96) (εd2 γd2 βd2 : ℝ)
    (Wp2 : Kernel4 24 96 1 1) (bp2 : Vec 24) (εp2 γp2 βp2 : ℝ)
    -- b3 (24→96→32, s2): 56→28
    (We3 : Kernel4 96 24 1 1) (be3 : Vec 96) (εe3 γe3 βe3 : ℝ)
    (Wd3 : DepthwiseKernel 96 3 3) (bd3 : Vec 96) (εd3 γd3 βd3 : ℝ)
    (Wp3 : Kernel4 32 96 1 1) (bp3 : Vec 32) (εp3 γp3 βp3 : ℝ)
    -- b4 (32→128→32, s1 skip): @28
    (We4 : Kernel4 128 32 1 1) (be4 : Vec 128) (εe4 γe4 βe4 : ℝ)
    (Wd4 : DepthwiseKernel 128 3 3) (bd4 : Vec 128) (εd4 γd4 βd4 : ℝ)
    (Wp4 : Kernel4 32 128 1 1) (bp4 : Vec 32) (εp4 γp4 βp4 : ℝ)
    -- b5 (32→128→64, s2): 28→14
    (We5 : Kernel4 128 32 1 1) (be5 : Vec 128) (εe5 γe5 βe5 : ℝ)
    (Wd5 : DepthwiseKernel 128 3 3) (bd5 : Vec 128) (εd5 γd5 βd5 : ℝ)
    (Wp5 : Kernel4 64 128 1 1) (bp5 : Vec 64) (εp5 γp5 βp5 : ℝ)
    -- b6 (64→256→64, s2): 14→7
    (We6 : Kernel4 256 64 1 1) (be6 : Vec 256) (εe6 γe6 βe6 : ℝ)
    (Wd6 : DepthwiseKernel 256 3 3) (bd6 : Vec 256) (εd6 γd6 βd6 : ℝ)
    (Wp6 : Kernel4 64 256 1 1) (bp6 : Vec 64) (εp6 γp6 βp6 : ℝ)
    -- head (64→128, 1×1 s1): @7  conv-bn-relu6
    (Wh : Kernel4 128 64 1 1) (bh : Vec 128) (εh γh βh : ℝ)
    -- dense head (128→10)
    (Wfc : Mat 128 10) (bfc : Vec 10) :
    Vec (3 * 224 * 224) → Vec 10 :=
  dense Wfc bfc ∘
  globalAvgPoolFlat 128 7 7 ∘
  (relu6 (128 * 7 * 7) ∘ bnForward (128 * 7 * 7) εh γh βh ∘ flatConv (h := 7) (w := 7) Wh bh) ∘
  invresBodyStrided (h := 7) (w := 7)
    We6 be6 εe6 γe6 βe6 Wd6 bd6 εd6 γd6 βd6 Wp6 bp6 εp6 γp6 βp6 ∘
  invresBodyStrided (h := 14) (w := 14)
    We5 be5 εe5 γe5 βe5 Wd5 bd5 εd5 γd5 βd5 Wp5 bp5 εp5 γp5 βp5 ∘
  residual (invresBody (h := 28) (w := 28)
    We4 be4 εe4 γe4 βe4 Wd4 bd4 εd4 γd4 βd4 Wp4 bp4 εp4 γp4 βp4) ∘
  invresBodyStrided (h := 28) (w := 28)
    We3 be3 εe3 γe3 βe3 Wd3 bd3 εd3 γd3 βd3 Wp3 bp3 εp3 γp3 βp3 ∘
  residual (invresBody (h := 56) (w := 56)
    We2 be2 εe2 γe2 βe2 Wd2 bd2 εd2 γd2 βd2 Wp2 bp2 εp2 γp2 βp2) ∘
  invresBodyStrided (h := 56) (w := 56)
    We1 be1 εe1 γe1 βe1 Wd1 bd1 εd1 γd1 βd1 Wp1 bp1 εp1 γp1 βp1 ∘
  (relu6 (16 * 112 * 112) ∘ bnForward (16 * 112 * 112) εs γs βs ∘
    flatConvStride2 (h := 112) (w := 112) Ws bs)

-- ════════════════════════════════════════════════════════════════
-- Concrete whole-network instance: every ReLU6 smoothness hypothesis
-- discharged (the `mobilenetv2_has_vjp_at` analogue of the MnistCNN
-- concrete instances). Closes the gap that the relu6 smoothness bundle is
-- never shown jointly satisfiable on the real `mobilenetv2Forward`.
--
-- NOTE: this is a *degenerate* witness. The discharge uses `bnForward_const`
-- (BN of a constant = its shift β), which requires every BN input to be
-- constant — forcing constant pre-GAP activations and hence a constant
-- network (zero Jacobian). That is intrinsic to ReLU6's two-sided kink: the
-- only cheap way to land every relu6 input strictly inside (0,6) is to pin
-- it to a single β. A *live* MobileNetV2 witness (non-trivial Jacobian)
-- needs genuine per-coordinate (0,6) bounds and is left as follow-up.
-- ════════════════════════════════════════════════════════════════

/-- BN of a constant vector is the (constant) shift `β` — centering zeroes
    the normalized term, killing the `√`. Keystone for discharging ReLU6
    smoothness on a constant-activation net. -/
theorem bnForward_const {n : Nat} (hn : 0 < n) (ε γ β c : ℝ) :
    bnForward n ε γ β (fun _ => c) = (fun _ => β) := by
  have hmean : bnMean n (fun _ : Fin n => c) = c := by
    unfold bnMean
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
        mul_comm, mul_div_assoc, div_self (Nat.cast_ne_zero.mpr hn.ne'), mul_one]
  funext i
  simp only [bnForward, bnXhat]
  rw [hmean]; ring

/-- A conv with everywhere-zero kernel and bias maps anything to `0`. -/
theorem flatConv_eq_zero {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (hW : ∀ o c kh kw, W o c kh kw = 0) (hb : ∀ o, b o = 0) (v : Vec (ic * h * w)) :
    flatConv (h := h) (w := w) W b v = (fun _ => (0:ℝ)) := by
  funext k; simp [flatConv, conv2d, Tensor3.flatten, hW, hb]

/-- A depthwise conv with everywhere-zero kernel and bias maps anything to `0`. -/
theorem depthwiseFlat_eq_zero {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (hW : ∀ ch kh kw, W ch kh kw = 0) (hb : ∀ ch, b ch = 0) (v : Vec (c * h * w)) :
    depthwiseFlat (h := h) (w := w) W b v = (fun _ => (0:ℝ)) := by
  funext k; simp [depthwiseFlat, depthwiseConv2d, Tensor3.flatten, hW, hb]

namespace MobileNetV2Concrete

-- ic=1, c=mid₁=oc=mid₂=2, h=w=2, nClasses=2, 1×1 kernels, zero weights,
-- all BN (ε,γ,β) = (1,1,1) so every relu6 input lands at β = 1 ∈ (0,6).
noncomputable def Ws  : Kernel4 2 1 1 1 := fun _ _ _ _ => 0
noncomputable def bs  : Vec 2 := fun _ => 0
noncomputable def We₁ : Kernel4 2 2 1 1 := fun _ _ _ _ => 0
noncomputable def bE₁ : Vec 2 := fun _ => 0
noncomputable def Wd₁ : DepthwiseKernel 2 1 1 := fun _ _ _ => 0
noncomputable def bD₁ : Vec 2 := fun _ => 0
noncomputable def Wp₁ : Kernel4 2 2 1 1 := fun _ _ _ _ => 0
noncomputable def bP₁ : Vec 2 := fun _ => 0
noncomputable def We₂ : Kernel4 2 2 1 1 := fun _ _ _ _ => 0
noncomputable def bE₂ : Vec 2 := fun _ => 0
noncomputable def Wd₂ : DepthwiseKernel 2 1 1 := fun _ _ _ => 0
noncomputable def bD₂ : Vec 2 := fun _ => 0
noncomputable def Wp₂ : Kernel4 2 2 1 1 := fun _ _ _ _ => 0
noncomputable def bP₂ : Vec 2 := fun _ => 0
noncomputable def Wh  : Mat 2 2 := fun _ _ => 0
noncomputable def bh  : Vec 2 := fun _ => 0
noncomputable def X   : Vec (1 * 2 * 2) := fun _ => 0

/-- **Whole-network VJP for a concrete MobileNetV2** — every ReLU6
    smoothness hypothesis (`bn ≠ 0 ∧ bn ≠ 6` at the five relu6 sites)
    discharged: every BN input is the zero vector (zero kernels), so each
    BN output is its shift `β = 1 ∈ (0,6)` via `bnForward_const`. -/
noncomputable def mnv2Concrete_has_vjp_at :
    HasVJPAt (mobilenetv2Forward Ws bs 1 1 1 We₁ bE₁ 1 1 1 Wd₁ bD₁ 1 1 1 Wp₁ bP₁ 1 1 1
      We₂ bE₂ 1 1 1 Wd₂ bD₂ 1 1 1 Wp₂ bP₂ 1 1 1 Wh bh) X :=
  mobilenetv2_has_vjp_at Ws bs 1 1 1 (by norm_num)
    We₁ bE₁ 1 1 1 (by norm_num) Wd₁ bD₁ 1 1 1 (by norm_num) Wp₁ bP₁ 1 1 1 (by norm_num)
    We₂ bE₂ 1 1 1 (by norm_num) Wd₂ bD₂ 1 1 1 (by norm_num) Wp₂ bP₂ 1 1 1 (by norm_num)
    Wh bh X
    (by intro k
        rw [flatConv_eq_zero Ws bs (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const (by norm_num)]
        exact ⟨by norm_num, by norm_num⟩)
    (by intro k
        rw [flatConv_eq_zero We₁ bE₁ (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const (by norm_num)]
        exact ⟨by norm_num, by norm_num⟩)
    (by intro k
        rw [depthwiseFlat_eq_zero Wd₁ bD₁ (fun _ _ _ => rfl) (fun _ => rfl), bnForward_const (by norm_num)]
        exact ⟨by norm_num, by norm_num⟩)
    (by intro k
        rw [flatConv_eq_zero We₂ bE₂ (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const (by norm_num)]
        exact ⟨by norm_num, by norm_num⟩)
    (by intro k
        rw [depthwiseFlat_eq_zero Wd₂ bD₂ (fun _ _ _ => rfl) (fun _ => rfl), bnForward_const (by norm_num)]
        exact ⟨by norm_num, by norm_num⟩)

/-- **Public unconditional correctness theorem** — the concrete MobileNetV2's
    backward equals the `pdiv`-Jacobian VJP, no hypotheses. -/
theorem mnv2Concrete_has_vjp_correct (dy : Vec 2) (i : Fin (1 * 2 * 2)) :
    mnv2Concrete_has_vjp_at.backward dy i =
      ∑ j : Fin 2, pdiv (mobilenetv2Forward Ws bs 1 1 1 We₁ bE₁ 1 1 1 Wd₁ bD₁ 1 1 1 Wp₁ bP₁ 1 1 1
        We₂ bE₂ 1 1 1 Wd₂ bD₂ 1 1 1 Wp₂ bP₂ 1 1 1 Wh bh) X i j * dy j :=
  mnv2Concrete_has_vjp_at.correct dy i

end MobileNetV2Concrete

-- ════════════════════════════════════════════════════════════════
-- § The *live* counterpart of `MobileNetV2Concrete`
--
--   `MobileNetV2Concrete` (above) discharges the off-the-kink bundle by
--   zeroing every kernel → constant output → zero Jacobian. `Mnv2Live`
--   discharges the SAME bundle on a NONZERO, non-collapsed net, defeating
--   BN's `√(σ²+ε)` with the `γ=1,β=3, n≤8` window (`bn13_window`) instead
--   of a constant collapse. The formal nonzero-Jacobian seal is the
--   documented residual (see the closing docstring); `bnForward_mean` /
--   `bn1_devSum_scale` / `bnIstd_pos` are its reusable, layout-free core.
-- ════════════════════════════════════════════════════════════════

namespace Mnv2Live

open Proofs

-- ════════════════════════════════════════════════════════════════
-- § The sqrt-defeating window lemma (the reusable contribution)
-- ════════════════════════════════════════════════════════════════

/-- **With `γ=1, β=3` and length `n ≤ 8`, every BN output is in `(0,6)`** — for
    an *arbitrary* input `z` and *arbitrary* `ε>0`. No constant-collapse, no
    sqrt computed: the bound reduces to `(zₖ−μ)² < 9(σ²+ε)`. -/
theorem bn13_window (n : Nat) (hn : 0 < n) (hn8 : n ≤ 8)
    (ε : ℝ) (hε : 0 < ε) (z : Vec n) (k : Fin n) :
    0 < bnForward n ε 1 3 z k ∧ bnForward n ε 1 3 z k < 6 := by
  set μ := bnMean n z with hμ
  set v := bnVar n z with hvdef
  have hv0 : 0 ≤ v := by
    rw [hvdef, bnVar]
    apply div_nonneg
    · exact Finset.sum_nonneg (fun i _ => mul_self_nonneg _)
    · positivity
  set s := v + ε with hsdef
  have hs0 : 0 < s := by rw [hsdef]; linarith
  have hsqrt : 0 < Real.sqrt s := Real.sqrt_pos.mpr hs0
  set d := z k - μ with hddef
  have hnv : (n : ℝ) * v = ∑ i, (z i - μ) * (z i - μ) := by
    rw [hvdef, bnVar]
    rw [mul_div_cancel₀]
    exact_mod_cast hn.ne'
  have hsingle : d * d ≤ ∑ i, (z i - μ) * (z i - μ) := by
    rw [hddef]
    exact Finset.single_le_sum (f := fun i => (z i - μ) * (z i - μ))
      (fun i _ => mul_self_nonneg _) (Finset.mem_univ k)
  have hd_le_nv : d * d ≤ (n : ℝ) * v := by rw [hnv]; exact hsingle
  have hn8' : (n : ℝ) ≤ 8 := by exact_mod_cast hn8
  have hnv_le : (n : ℝ) * v ≤ 8 * v := mul_le_mul_of_nonneg_right hn8' hv0
  have hkey : d * d < 9 * s := by rw [hsdef]; nlinarith [hd_le_nv, hnv_le, hv0, hε]
  have hd2 : d ^ 2 < 9 * s := by rw [sq]; exact hkey
  have habs : |d| < 3 * Real.sqrt s := by
    have h1 : Real.sqrt (d ^ 2) < Real.sqrt (9 * s) :=
      Real.sqrt_lt_sqrt (sq_nonneg d) hd2
    rw [Real.sqrt_sq_eq_abs] at h1
    rwa [Real.sqrt_mul (by norm_num) s, show Real.sqrt 9 = 3 by
      rw [show (9 : ℝ) = 3 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]] at h1
  have hval : bnForward n ε 1 3 z k = d / Real.sqrt s + 3 := by
    simp only [bnForward, bnXhat, bnIstd, hddef, one_mul]
    rw [mul_one_div]
  have hquot : |d / Real.sqrt s| < 3 := by
    rw [abs_div, abs_of_pos hsqrt, div_lt_iff₀ hsqrt]
    linarith [habs]
  rw [abs_lt] at hquot
  rw [hval]
  constructor <;> linarith [hquot.1, hquot.2]

-- ════════════════════════════════════════════════════════════════
-- § Reusable, layout-free core of the liveness seal
--   BN forces output mean β and rescales every deviation by istd > 0,
--   so a cross-channel asymmetry planted at the stem cannot vanish.
-- ════════════════════════════════════════════════════════════════

/-- Deviations sum to zero: `Σₖ (zₖ − μ) = 0`. -/
theorem dev_sum_zero (n : Nat) (hn : 0 < n) (z : Vec n) :
    ∑ k, (z k - bnMean n z) = 0 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, bnMean,
    nsmul_eq_mul]
  have h : (n : ℝ) * ((∑ k, z k) / n) = ∑ k, z k := by field_simp
  rw [h, sub_self]

/-- **BN forces the output mean to `β`** (for `n > 0`). -/
theorem bnForward_mean (n : Nat) (hn : 0 < n) (ε γ β : ℝ) (z : Vec n) :
    bnMean n (bnForward n ε γ β z) = β := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hxhat : ∑ k, bnXhat n ε z k = 0 := by
    simp only [bnXhat]
    rw [← Finset.sum_mul, dev_sum_zero n hn z, zero_mul]
  have hsum : ∑ k, bnForward n ε γ β z k = (n : ℝ) * β := by
    simp only [bnForward]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, hxhat, mul_zero, zero_add,
      Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [bnMean, hsum, mul_comm (n : ℝ) β, mul_div_assoc, div_self hn', mul_one]

/-- **BN rescales every deviation by `istd`** (the `γ=1` case). Over *any*
    index set `S`, the BN-output deviation-sum is the input deviation-sum
    scaled by the positive `bnIstd`. This is what carries a stem-planted
    cross-channel asymmetry through the four BN layers undamped. -/
theorem bn1_devSum_scale (n : Nat) (hn : 0 < n) (ε β : ℝ) (z : Vec n)
    (S : Finset (Fin n)) :
    ∑ k ∈ S, (bnForward n ε 1 β z k - bnMean n (bnForward n ε 1 β z))
      = bnIstd n z ε * ∑ k ∈ S, (z k - bnMean n z) := by
  rw [bnForward_mean n hn ε 1 β z, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k _
  simp only [bnForward, bnXhat, one_mul]
  ring

/-- `bnIstd` is strictly positive (so the rescaling above never kills the sign). -/
theorem bnIstd_pos (n : Nat) (ε : ℝ) (hε : 0 < ε) (z : Vec n) :
    0 < bnIstd n z ε := by
  rw [bnIstd]
  apply div_pos one_pos
  apply Real.sqrt_pos.mpr
  have hv0 : 0 ≤ bnVar n z := by
    rw [bnVar]; apply div_nonneg
    · exact Finset.sum_nonneg (fun i _ => mul_self_nonneg _)
    · positivity
  linarith

-- ════════════════════════════════════════════════════════════════
-- § A concrete net with NONZERO, NON-COLLAPSED weights
--   dims: ic=1, c=mid₁=oc=mid₂=2, h=w=2, nClasses=2  (every BN vec len 8).
--
--   • stem: ASYMMETRIC (channel 0 ·1, channel 1 ·2) — plants the
--     cross-channel asymmetry the seal relies on.
--   • block1 (skip): body zeroed; `invresBody₁ ≡ const`, so block1 is a
--     constant shift of its input (injective; preserves deviations).
--   • block2 (no skip): IDENTITY convs ⇒ three genuine BN layers on the
--     signal (the irreducible nonlinearity — block2 has no skip to hide in).
--   • head: IDENTITY dense ⇒ output = per-channel GAP (reads each channel,
--     unlike v1's all-ones head which BN-mean-collapsed to a constant).
-- ════════════════════════════════════════════════════════════════

noncomputable def Ws  : Kernel4 2 1 1 1 := fun o _ _ _ => if o = 0 then 1 else 2
noncomputable def bs  : Vec 2 := fun _ => 0
noncomputable def We₁ : Kernel4 2 2 1 1 := fun _ _ _ _ => 0
noncomputable def be₁ : Vec 2 := fun _ => 0
noncomputable def Wd₁ : DepthwiseKernel 2 1 1 := fun _ _ _ => 0
noncomputable def bd₁ : Vec 2 := fun _ => 0
noncomputable def Wp₁ : Kernel4 2 2 1 1 := fun _ _ _ _ => 0
noncomputable def bp₁ : Vec 2 := fun _ => 0
/-- block2 expand: identity channel map. -/
noncomputable def We₂ : Kernel4 2 2 1 1 := fun o i _ _ => if o = i then 1 else 0
noncomputable def be₂ : Vec 2 := fun _ => 0
/-- block2 depthwise: identity (single 1×1 tap). -/
noncomputable def Wd₂ : DepthwiseKernel 2 1 1 := fun _ _ _ => 1
noncomputable def bd₂ : Vec 2 := fun _ => 0
/-- block2 project: identity channel map. -/
noncomputable def Wp₂ : Kernel4 2 2 1 1 := fun o i _ _ => if o = i then 1 else 0
noncomputable def bp₂ : Vec 2 := fun _ => 0
/-- identity dense head ⇒ output = per-channel GAP. -/
noncomputable def Wh  : Mat 2 2 := fun i j => if i = j then 1 else 0
noncomputable def bh  : Vec 2 := fun _ => 0
/-- Non-constant input. -/
noncomputable def X : Vec (1 * 2 * 2) := fun i => (i.val : ℝ)

/-- The five ReLU6 sites all discharge through the one window lemma (length 8,
    γ=1, β=3, ε=1), regardless of the weights feeding them. -/
private theorem win (z : Vec (2 * 2 * 2)) (k : Fin (2 * 2 * 2)) :
    bnForward (2 * 2 * 2) 1 1 3 z k ≠ 0 ∧ bnForward (2 * 2 * 2) 1 1 3 z k ≠ 6 := by
  obtain ⟨h0, h6⟩ := bn13_window (2 * 2 * 2) (by norm_num) (by norm_num) 1 one_pos z k
  exact ⟨h0.ne', h6.ne⟩

/-- **Unconditional whole-network VJP on a nonzero, non-collapsed MobileNetV2.**
    Every ReLU6 smoothness hypothesis of `mobilenetv2_has_vjp_at` is discharged
    by `win` (the window lemma) — *not* by a constant collapse. No side
    conditions; three-axiom closure. -/
noncomputable def mnv2Live_has_vjp_at :
    HasVJPAt (mobilenetv2Forward Ws bs 1 1 3
      We₁ be₁ 1 1 3 Wd₁ bd₁ 1 1 3 Wp₁ bp₁ 1 1 3
      We₂ be₂ 1 1 3 Wd₂ bd₂ 1 1 3 Wp₂ bp₂ 1 1 3 Wh bh) X :=
  mobilenetv2_has_vjp_at Ws bs 1 1 3 one_pos
    We₁ be₁ 1 1 3 one_pos Wd₁ bd₁ 1 1 3 one_pos Wp₁ bp₁ 1 1 3 one_pos
    We₂ be₂ 1 1 3 one_pos Wd₂ bd₂ 1 1 3 one_pos Wp₂ bp₂ 1 1 3 one_pos Wh bh X
    (fun k => win _ k) (fun k => win _ k) (fun k => win _ k)
    (fun k => win _ k) (fun k => win _ k)

/-- **Public unconditional correctness theorem** — the nonzero-weight
    MobileNetV2's backward equals the `pdiv`-Jacobian VJP, no hypotheses. -/
theorem mnv2Live_has_vjp_correct (dy : Vec 2) (i : Fin (1 * 2 * 2)) :
    mnv2Live_has_vjp_at.backward dy i =
      ∑ j : Fin 2,
        pdiv (mobilenetv2Forward Ws bs 1 1 3
          We₁ be₁ 1 1 3 Wd₁ bd₁ 1 1 3 Wp₁ bp₁ 1 1 3
          We₂ be₂ 1 1 3 Wd₂ bd₂ 1 1 3 Wp₂ bp₂ 1 1 3 Wh bh) X i j * dy j :=
  mnv2Live_has_vjp_at.correct dy i

/-! ## The remaining obligation: non-vacuity (nonzero Jacobian)

```
theorem mnv2Live_jacobian_ne_zero :
    ∃ (i : Fin (1 * 2 * 2)) (j : Fin 2),
      pdiv (mobilenetv2Forward Ws bs 1 1 3 …) X i j ≠ 0
```

At `X` every ReLU6 is strictly inside `(0,6)` (`win`), so each is *locally the
identity*, and with the identity convs of block2 the forward agrees on a
neighborhood with `dense_id ∘ gap ∘ BN ∘ BN ∘ BN ∘ BN ∘ flatConv`. The
asymmetric stem plants `Σ_{channel 0}(z − μ) = −3 ≠ 0`; `bnForward_mean` +
`bn1_devSum_scale` + `bnIstd_pos` carry that deviation through all four BN
layers scaled by a positive constant, so `gap₀ = 3 − 3·∏istd/4 < 3`, whereas
the constant input gives `gap₀ = 3` (`bnForward_const`). Hence the forward is
non-constant and the Jacobian is nonzero. The residual is purely the layer
reduction + the concrete `finProdFinEquiv` evaluation of `flatConv`/`gap`.
-/

end Mnv2Live

end Proofs
