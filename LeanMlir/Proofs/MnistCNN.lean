import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.MLP

/-! # Chapter 4: MNIST 2D CNN (no BatchNorm) — whole-network VJP

The Chapter-4 demo model `mnistCnnNoBn`:

  conv2d 1→c (relu) → conv2d c→c (relu) → maxPool 2×2 → flatten
    → dense (relu) → dense (relu) → dense (identity)

This file builds two things:

* `mnistCnnNoBn_has_vjp_at` — the **structural** whole-network VJP: the
  composed backward equals the `pdiv`-Jacobian VJP of the full forward
  pass, *conditional* on smoothness hypotheses (no ReLU kink / MaxPool
  tie at the running activations). The Chapter-4 sibling of
  `cnn_has_vjp_at`, minus BN and residual blocks.

* `mnistMicroCnn_has_vjp_correct` — a **concrete tiny instance** where
  every smoothness hypothesis is *discharged* (`norm_num`/explicit), so
  the statement is **unconditional** and still closes under the
  three-axiom kernel. The witness that the conditional machinery is
  instantiable. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Building blocks: conv→relu and dense→relu (no BN)
-- ════════════════════════════════════════════════════════════════

/-- **conv → relu block VJP at a smooth point** (no BatchNorm).
    `relu ∘ flatConv W b`. The plain-conv analogue of
    `convBnRelu_has_vjp_at` — conv is linear (global VJP via the
    `HasVJP3` bridge), relu carries the smoothness hypothesis. -/
noncomputable def convRelu_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, flatConv W b v k ≠ 0) :
    HasVJPAt (relu (oc * h * w) ∘ flatConv W b) v :=
  vjp_comp_at (flatConv W b) (relu (oc * h * w)) v
    ((flatConv_differentiable W b) v)
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    ((hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)).toHasVJPAt v)
    (relu_has_vjp_at (oc * h * w) _ h_smooth)

/-- `relu ∘ flatConv W b` is differentiable at a smooth point. -/
theorem convRelu_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic * h * w))
    (h_smooth : ∀ k, flatConv W b v k ≠ 0) :
    DifferentiableAt ℝ (relu (oc * h * w) ∘ flatConv W b) v :=
  (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth).comp v
    ((flatConv_differentiable W b) v)

/-- **dense → relu block VJP at a smooth point.** `relu ∘ dense W b`. -/
noncomputable def denseRelu_has_vjp_at {m n : Nat}
    (W : Mat m n) (b : Vec n) (v : Vec m)
    (h_smooth : ∀ k, dense W b v k ≠ 0) :
    HasVJPAt (relu n ∘ dense W b) v :=
  vjp_comp_at (dense W b) (relu n) v
    ((dense_differentiable W b) v)
    (relu_differentiableAt_of_smooth n _ h_smooth)
    ((dense_has_vjp W b).toHasVJPAt v)
    (relu_has_vjp_at n _ h_smooth)

/-- `relu ∘ dense W b` is differentiable at a smooth point. -/
theorem denseRelu_differentiableAt {m n : Nat}
    (W : Mat m n) (b : Vec n) (v : Vec m)
    (h_smooth : ∀ k, dense W b v k ≠ 0) :
    DifferentiableAt ℝ (relu n ∘ dense W b) v :=
  (relu_differentiableAt_of_smooth n _ h_smooth).comp v ((dense_differentiable W b) v)

-- ════════════════════════════════════════════════════════════════
-- § Chapter-4 forward pass (BN-free)
-- ════════════════════════════════════════════════════════════════

/-- The Chapter-4 `mnistCnnNoBn` forward, in flattened `Vec` space.
    Conv stage runs at spatial `(2*h, 2*w)`; the `maxPool` halves it to
    `(h, w)`; then three dense layers (two with ReLU). -/
noncomputable def mnistCnnNoBnForward
    {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) :
    Vec (ic * (2*h) * (2*w)) → Vec nClasses :=
  dense W₅ b₅
  ∘ (relu d1 ∘ dense W₄ b₄)
  ∘ (relu d1 ∘ dense W₃ b₃)
  ∘ maxPoolFlat c h w
  ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
  ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)

-- ════════════════════════════════════════════════════════════════
-- § Structural whole-network VJP (Chapter-4 capstone, conditional)
-- ════════════════════════════════════════════════════════════════

/-- **MNIST 2D CNN (no BN) whole-network VJP at a smooth point.**

    The composed backward of the full Chapter-4 forward equals the
    `pdiv`-contracted Jacobian (Jacobian-transpose applied to the
    cotangent), conditional on smoothness at the four ReLU kinks and
    the one MaxPool. Built by `vjp_comp_at` through
    `convRelu → convRelu → maxPool → denseRelu → denseRelu → dense`.
    The Chapter-4 sibling of `cnn_has_vjp_at` (BN-free, no resblocks). -/
noncomputable def mnistCnnNoBn_has_vjp_at
    {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*h) * (2*w)))
    (h1 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₁ b₁ x k ≠ 0)
    (h2 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₂ b₂
            ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x) k ≠ 0)
    (h_mp : MaxPool2Smooth (Tensor3.unflatten
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)
            : Tensor3 c (2*h) (2*w)))
    (h3 : ∀ k, dense W₃ b₃ (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)) k ≠ 0)
    (h4 : ∀ k, dense W₄ b₄ ((relu d1 ∘ dense W₃ b₃) (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x))) k ≠ 0) :
    HasVJPAt (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅) x := by
  unfold mnistCnnNoBnForward
  -- conv→relu block 1 at x
  have s1 := convRelu_has_vjp_at W₁ b₁ x h1
  have s1d := convRelu_differentiableAt W₁ b₁ x h1
  -- conv→relu block 2 at (block-1 output)
  have s2v := convRelu_has_vjp_at W₂ b₂ _ h2
  have s2d2 := convRelu_differentiableAt W₂ b₂ _ h2
  have s2 := vjp_comp_at _ _ x s1d s2d2 s1 s2v
  have s2d := s2d2.comp x s1d
  -- maxpool at (block-2 output); align the point via flatten ∘ unflatten = id
  set zmp := (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x) with hzmp
  have hpt : Tensor3.flatten (Tensor3.unflatten zmp : Tensor3 c (2*h) (2*w)) = zmp :=
    Tensor3.flatten_unflatten zmp
  have mp_v : HasVJPAt (maxPoolFlat c h w) zmp := by
    rw [← hpt]; exact maxPoolFlat_has_vjp_at _ h_mp
  have mp_d : DifferentiableAt ℝ (maxPoolFlat c h w) zmp := by
    rw [← hpt]; exact maxPoolFlat_differentiableAt _ h_mp hc hh hw
  have s3 := vjp_comp_at _ _ x s2d mp_d s2 mp_v
  have s3d := mp_d.comp x s2d
  -- dense→relu block 3
  set zd3 := maxPoolFlat c h w zmp with hzd3
  have s4v := denseRelu_has_vjp_at W₃ b₃ zd3 h3
  have s4d3 := denseRelu_differentiableAt W₃ b₃ zd3 h3
  have s4 := vjp_comp_at _ _ x s3d s4d3 s3 s4v
  have s4d := s4d3.comp x s3d
  -- dense→relu block 4
  set zd4 := (relu d1 ∘ dense W₃ b₃) zd3 with hzd4
  have s5v := denseRelu_has_vjp_at W₄ b₄ zd4 h4
  have s5d4 := denseRelu_differentiableAt W₄ b₄ zd4 h4
  have s5 := vjp_comp_at _ _ x s4d s5d4 s4 s5v
  have s5d := s5d4.comp x s4d
  -- final dense (linear, no smoothness)
  exact vjp_comp_at _ _ x s5d ((dense_differentiable W₅ b₅) _) s5
    ((dense_has_vjp W₅ b₅).toHasVJPAt _)

/-- **Public correctness theorem for `mnistCnnNoBn_has_vjp_at`** — the
    Chapter-4 CNN's backward equals the `pdiv`-contracted Jacobian. -/
theorem mnistCnnNoBn_has_vjp_at_correct
    {ic c h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*h) * (2*w)))
    (h1 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₁ b₁ x k ≠ 0)
    (h2 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₂ b₂
            ((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁) x) k ≠ 0)
    (h_mp : MaxPool2Smooth (Tensor3.unflatten
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)
            : Tensor3 c (2*h) (2*w)))
    (h3 : ∀ k, dense W₃ b₃ (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x)) k ≠ 0)
    (h4 : ∀ k, dense W₄ b₄ ((relu d1 ∘ dense W₃ b₃) (maxPoolFlat c h w
            (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x))) k ≠ 0)
    (dy : Vec nClasses) (i : Fin (ic * (2*h) * (2*w))) :
    (mnistCnnNoBn_has_vjp_at W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
        hc hh hw x h1 h2 h_mp h3 h4).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅) x i j * dy j :=
  (mnistCnnNoBn_has_vjp_at W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
      hc hh hw x h1 h2 h_mp h3 h4).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § Concrete tiny instance — every smoothness hypothesis DISCHARGED
-- ════════════════════════════════════════════════════════════════

/-! A minimal `mnistCnnNoBn` at `ic=c=h=w=d1=nClasses=1`, `1×1` kernels,
    with hand-picked weights so the conv stack is the identity, every
    ReLU sees strictly-positive input, and the single MaxPool window has
    four distinct values. All five smoothness hypotheses of
    `mnistCnnNoBn_has_vjp_at` are then *proved*, yielding an
    **unconditional** whole-network VJP correctness theorem — the
    non-vacuity witness for the conditional capstone above. -/

namespace Micro
open Matrix

/-- Input `Vec 4` with four distinct strictly-positive entries `1,2,3,4`. -/
noncomputable def X0 : Vec (1 * (2*1) * (2*1)) := fun k => (k.val : ℝ) + 1
/-- `1×1×1×1` all-ones kernel ⇒ conv is the identity. -/
noncomputable def K1 : Kernel4 1 1 1 1 := fun _ _ _ _ => 1
/-- Zero bias. -/
noncomputable def Bz : Vec 1 := fun _ => 0
/-- All-ones dense weight (`Mat 1 1`). -/
noncomputable def Wd : Mat (1 * 1 * 1) 1 := fun _ _ => 1
/-- Unit bias (keeps dense outputs strictly positive). -/
noncomputable def B1 : Vec 1 := fun _ => 1

/-- The `1×1` ones-kernel conv2d is the identity. -/
theorem conv2d_K1_id (t : Tensor3 1 (2*1) (2*1)) : conv2d K1 Bz t = t := by
  funext o hi wi
  fin_cases o
  fin_cases hi <;> fin_cases wi <;> simp [conv2d, K1, Bz]

/-- Hence `flatConv K1 Bz` is the identity. -/
theorem flatConv_K1_id (v : Vec (1 * (2*1) * (2*1))) :
    flatConv (h := 2*1) (w := 2*1) K1 Bz v = v := by
  simp [flatConv, conv2d_K1_id, Tensor3.flatten_unflatten]

/-- ReLU is the identity on strictly-positive vectors. -/
theorem relu_pos {n : Nat} (v : Vec n) (hv : ∀ i, 0 < v i) : relu n v = v := by
  funext i; simp [relu, hv i]

theorem X0_pos : ∀ i, 0 < X0 i := by
  intro i; simp only [X0]; positivity

theorem X0_ge_one : ∀ i, (1 : ℝ) ≤ X0 i := by
  intro i; simp only [X0]
  have h : (0:ℝ) ≤ (i.val : ℝ) := Nat.cast_nonneg _
  linarith

/-- The conv→relu block returns its (positive) input unchanged. -/
theorem CR_id (v : Vec (1 * (2*1) * (2*1))) (hv : ∀ i, 0 < v i) :
    (relu (1 * (2*1) * (2*1)) ∘ flatConv (h := 2*1) (w := 2*1) K1 Bz) v = v := by
  simp only [Function.comp_apply, flatConv_K1_id]
  exact relu_pos v hv

/-- Two conv→relu blocks on `X0` collapse back to `X0` (the MaxPool input). -/
theorem maxpool_input_eq :
    ((relu (1 * (2*1) * (2*1)) ∘ flatConv (h := 2*1) (w := 2*1) K1 Bz)
      ∘ (relu (1 * (2*1) * (2*1)) ∘ flatConv (h := 2*1) (w := 2*1) K1 Bz)) X0 = X0 := by
  simp only [Function.comp_apply, flatConv_K1_id, relu_pos X0 X0_pos]

/-- The single 2×2 MaxPool window of `X0` has four distinct values. -/
theorem mp_smooth_X0 : MaxPool2Smooth (Tensor3.unflatten X0 : Tensor3 1 (2*1) (2*1)) := by
  intro ci hi wi ab ab' hne
  fin_cases ci
  fin_cases hi
  fin_cases wi
  fin_cases ab <;> fin_cases ab' <;>
    simp_all [X0, Tensor3.unflatten, winRowInv, winColInv] <;> decide

/-- The pooled value is ≥ 1 (so the dense head stays nonzero). -/
theorem maxPoolFlat_X0_ge_one : ∀ k, (1 : ℝ) ≤ maxPoolFlat 1 1 1 X0 k := by
  intro k
  fin_cases k
  simp only [maxPoolFlat, maxPool2, Tensor3.flatten, Tensor3.unflatten]
  exact le_trans (X0_ge_one _) (le_trans (le_max_left _ _) (le_max_left _ _))

/-- `dense Wd B1` (ones weight, unit bias) maps a positive vector to a
    strictly-positive one. The dense head stays off the ReLU kink. -/
theorem dense_Wd_pos (u : Vec (1 * 1 * 1)) (hu : ∀ j, 0 < u j) (k : Fin 1) :
    0 < dense Wd B1 u k := by
  simp only [dense, Wd, B1, mul_one]
  have hsum : (0:ℝ) ≤ ∑ i, u i := Finset.sum_nonneg (fun i _ => le_of_lt (hu i))
  linarith

/-- The pooled vector is strictly positive (feeds the dense head). -/
theorem maxPoolFlat_X0_pos : ∀ j, 0 < maxPoolFlat 1 1 1 X0 j :=
  fun j => lt_of_lt_of_le zero_lt_one (maxPoolFlat_X0_ge_one j)

/-- **Unconditional whole-network VJP for a concrete tiny CNN.**
    Every smoothness hypothesis of `mnistCnnNoBn_has_vjp_at` is
    discharged here, so this statement carries no side conditions. -/
noncomputable def mnistMicroCnn_has_vjp_at :
    HasVJPAt (mnistCnnNoBnForward K1 Bz K1 Bz Wd B1 Wd B1 Wd Bz) X0 :=
  mnistCnnNoBn_has_vjp_at K1 Bz K1 Bz Wd B1 Wd B1 Wd Bz
    (by norm_num) (by norm_num) (by norm_num) X0
    -- h1: conv1 preactivation nonzero
    (by intro k; rw [flatConv_K1_id]; exact ne_of_gt (X0_pos k))
    -- h2: conv2 preactivation nonzero
    (by intro k; rw [CR_id X0 X0_pos, flatConv_K1_id]; exact ne_of_gt (X0_pos k))
    -- h_mp: no MaxPool ties
    (by rw [maxpool_input_eq]; exact mp_smooth_X0)
    -- h3: dense3 preactivation nonzero
    (by
      intro k
      rw [maxpool_input_eq]
      exact ne_of_gt (dense_Wd_pos _ maxPoolFlat_X0_pos k))
    -- h4: dense4 preactivation nonzero
    (by
      intro k
      rw [maxpool_input_eq]
      have h_inner : ∀ j, 0 < dense Wd B1 (maxPoolFlat 1 1 1 X0) j :=
        fun j => dense_Wd_pos _ maxPoolFlat_X0_pos j
      rw [Function.comp_apply, relu_pos _ h_inner]
      exact ne_of_gt (dense_Wd_pos _ h_inner k))

/-- **Public unconditional correctness theorem** — the concrete tiny
    CNN's backward equals the `pdiv`-Jacobian VJP, no hypotheses. -/
theorem mnistMicroCnn_has_vjp_correct (dy : Vec 1) (i : Fin (1 * (2*1) * (2*1))) :
    mnistMicroCnn_has_vjp_at.backward dy i =
      ∑ j : Fin 1, pdiv (mnistCnnNoBnForward K1 Bz K1 Bz Wd B1 Wd B1 Wd Bz) X0 i j * dy j :=
  mnistMicroCnn_has_vjp_at.correct dy i

end Micro

end Proofs
