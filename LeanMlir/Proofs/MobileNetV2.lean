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


end Proofs