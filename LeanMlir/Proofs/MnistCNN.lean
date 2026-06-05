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

-- ════════════════════════════════════════════════════════════════
-- § Reusable discharge lemmas for the smoothness hypotheses
--
-- The `Micro` instance above discharges `MaxPool2Smooth` by `fin_cases`
-- over its single 2×2 window. That does not scale: `MaxPool2Smooth` is
-- `6·c·h·w` pairwise inequalities, so a realistic spatial size turns it
-- into thousands of `decide`s. These lemmas replace the case-bashing
-- with structural arguments (positional injectivity for the no-tie
-- condition; positivity propagation for the ReLU `≠ 0` conditions), so a
-- many-window instance is dischargeable — and stays inside the
-- `[propext, Classical.choice, Quot.sound]` closure (no `native_decide`).
-- ════════════════════════════════════════════════════════════════

/-- **Positional injectivity ⇒ `MaxPool2Smooth`.** If, on each channel,
    the position map `(r, s) ↦ x ci r s` is injective, then every 2×2
    window has pairwise-distinct values. One injectivity argument in
    place of `6·c·h·w` per-window `decide`s. -/
theorem maxPool2Smooth_of_injective {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (hinj : ∀ (ci : Fin c) (r r' : Fin (2*h)) (s s' : Fin (2*w)),
              x ci r s = x ci r' s' → r = r' ∧ s = s') :
    MaxPool2Smooth x := by
  intro ci hi_out wi_out ab ab' hne hval
  apply hne
  obtain ⟨hr, hs⟩ := hinj ci _ _ _ _ hval
  have ha : ab.1 = ab'.1 := by
    have h := hr; unfold winRowInv at h; rw [Fin.mk.injEq] at h
    exact Fin.ext (by omega)
  have hb : ab.2 = ab'.2 := by
    have h := hs; unfold winColInv at h; rw [Fin.mk.injEq] at h
    exact Fin.ext (by omega)
  exact Prod.ext_iff.mpr ⟨ha, hb⟩

/-- A tensor that is positive everywhere flattens to a positive vector
    (`flatten T k` just reads `T` at the decoded index). Discharges the
    ReLU `∀ k, … ≠ 0` conditions once the layer is shown positive. -/
theorem flatten_pos_of_pos {c h w : Nat} {T : Tensor3 c h w}
    (hT : ∀ ci hi wi, 0 < T ci hi wi) (k : Fin (c * h * w)) :
    0 < Tensor3.flatten T k := by
  unfold Tensor3.flatten; exact hT _ _ _

/-- A 2×2 max-pool of an everywhere-positive tensor is positive (the max
    dominates the top-left cell). -/
theorem maxPool2_pos {c h w : Nat} {x : Tensor3 c (2*h) (2*w)}
    (hx : ∀ ci r s, 0 < x ci r s) (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    0 < maxPool2 x ci hi wi := by
  unfold maxPool2
  exact lt_of_lt_of_le (hx _ _ _) (le_trans (le_max_left _ _) (le_max_left _ _))

/-- ReLU is the identity on a strictly-positive vector. Discharges the
    ReLU-as-identity steps that fold the composition into a plain conv
    stack at a smooth (everywhere-positive) point. -/
theorem relu_id_of_pos {n : Nat} {v : Vec n} (hv : ∀ i, 0 < v i) : relu n v = v := by
  funext i; simp only [relu]; rw [if_pos (hv i)]

/-- A dense layer with nonnegative weights, a strictly-positive bias, and a
    nonnegative input is strictly positive — the propagating positivity
    invariant that discharges the dense ReLU `≠ 0` conditions without
    per-coordinate case analysis. -/
theorem dense_pos_of_nonneg {m n : Nat} {W : Mat m n} {b : Vec n} {u : Vec m}
    (hW : ∀ i j, 0 ≤ W i j) (hb : ∀ j, 0 < b j) (hu : ∀ i, 0 ≤ u i) (j : Fin n) :
    0 < dense W b u j := by
  simp only [dense]
  have h1 : (0:ℝ) ≤ ∑ i : Fin m, u i * W i j :=
    Finset.sum_nonneg (fun i _ => mul_nonneg (hu i) (hW i j))
  have h2 : (0:ℝ) < b j := hb j
  linarith

/-- **1×1 conv collapses to a per-pixel channel mix.** With a 1×1 kernel
    (SAME padding is a no-op), `conv2d` at each pixel is just the bias
    plus a channel-weighted sum of that same pixel — the closed form a
    center-structured instance computes its forward pass with. -/
theorem conv2d_1x1 {ic oc h w : Nat} (W : Kernel4 oc ic 1 1) (b : Vec oc)
    (t : Tensor3 ic h w) (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W b t o hi wi = b o + ∑ c : Fin ic, W o c 0 0 * t c hi wi := by
  unfold conv2d
  congr 1
  refine Finset.sum_congr rfl (fun c _ => ?_)
  rw [Fin.sum_univ_one, Fin.sum_univ_one]
  congr 1
  dsimp only
  split
  · refine congrArg₂ (t c) ?_ ?_ <;> (apply Fin.ext; simp only [Fin.val_zero]; omega)
  · rename_i hcond
    exact absurd (by
      have := hi.isLt; have := wi.isLt
      refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only [Fin.val_zero] <;> omega) hcond

/-- **3×3 conv with a center-only kernel collapses to a per-pixel channel
    mix.** If `W` vanishes off the center tap `(1,1)`, the full 3×3
    SAME-padding sum — all nine taps and their padding branches — reduces
    to `b o + ∑ c, W o c 1 1 · t c hi wi`. The 3×3 analogue of
    `conv2d_1x1`: a center-structured instance exercises genuine spatial
    convolution (the padding `if`s are evaluated) while keeping a closed
    forward form. -/
theorem conv2d_center3x3 {ic oc h w : Nat}
    (W : Kernel4 oc ic 3 3) (b : Vec oc)
    (hW : ∀ (o : Fin oc) (c : Fin ic) (kh kw : Fin 3),
            ¬ (kh = 1 ∧ kw = 1) → W o c kh kw = 0)
    (t : Tensor3 ic h w) (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W b t o hi wi = b o + ∑ c : Fin ic, W o c 1 1 * t c hi wi := by
  unfold conv2d
  congr 1
  refine Finset.sum_congr rfl (fun c _ => ?_)
  -- Expand the 3×3 tap sum; the eight off-center taps vanish (`W = 0`).
  simp only [Fin.sum_univ_three]
  rw [hW o c 0 0 (by decide), hW o c 0 1 (by decide), hW o c 0 2 (by decide),
      hW o c 1 0 (by decide), hW o c 1 2 (by decide),
      hW o c 2 0 (by decide), hW o c 2 1 (by decide), hW o c 2 2 (by decide)]
  simp only [zero_mul, add_zero, zero_add]
  -- Center tap (1,1): SAME-padding keeps it in range, value is `t c hi wi`.
  congr 1
  dsimp only
  split
  · refine congrArg₂ (t c) ?_ ?_ <;> (apply Fin.ext; simp only [Fin.val_one]; omega)
  · rename_i hcond
    exact absurd (by
      have := hi.isLt; have := wi.isLt
      refine ⟨?_, ?_, ?_, ?_⟩ <;> simp only [Fin.val_one] <;> omega) hcond

-- ════════════════════════════════════════════════════════════════
-- § Tier-1 instance — a genuinely multi-channel, multi-window,
--   10-class `mnistCnnNoBn` with every smoothness hypothesis DISCHARGED
--
-- Unlike `Micro` (all dims `1`, 1×1 kernel chosen so conv = identity, a
-- single 2×2 pool window, 1 class), this instance turns on the structure
-- the conditional capstone is interesting for:
--   • 2 input→2 output→2 output channels (a real 2-channel mixing conv2),
--   • 4×4 → 2×2 spatial: FOUR pool windows per channel (eight total),
--   • a 10-class classifier head.
-- It still uses 1×1 kernels (so the forward has a clean closed form), with
-- weights chosen so the conv stack is everywhere positive and positionally
-- injective. The MaxPool no-tie condition is discharged by
-- `maxPool2Smooth_of_injective` (one argument, not eight windows of
-- `decide`s) and the ReLU conditions by positivity propagation — all
-- inside the three-axiom closure (no `native_decide`).
-- ════════════════════════════════════════════════════════════════

namespace Mini

/-- Input tensor with 16 distinct strictly-positive integer values, so
    `(hi, wi) ↦ T0 0 hi wi` is injective in position. -/
noncomputable def T0 : Tensor3 1 (2*2) (2*2) :=
  fun _ hi wi => ((4 * hi.val + wi.val + 1 : ℕ) : ℝ)
/-- Whole-network input, the flattened `T0`. -/
noncomputable def X : Vec (1 * (2*2) * (2*2)) := Tensor3.flatten T0
/-- conv1: 1→2 channels, 1×1, unit tap. -/
noncomputable def W1 : Kernel4 2 1 1 1 := fun _ _ _ _ => 1
/-- conv1 bias `(1, 2)` — gives the two output channels distinct values. -/
noncomputable def b1 : Vec 2 := fun o => if o = 0 then 1 else 2
/-- conv2: 2→2 channels, 1×1. Row depends on the output channel
    (`1` for channel 0, `2` for channel 1), so the two output channels
    differ and each has a strictly-positive input-pixel coefficient. -/
noncomputable def W2 : Kernel4 2 2 1 1 := fun o _ _ _ => if o = 0 then 1 else 2
/-- conv2 bias. -/
noncomputable def b2 : Vec 2 := fun _ => 1
/-- Dense heads: nonnegative weights + strictly-positive biases keep every
    activation off the ReLU kink. -/
noncomputable def W3 : Mat (2*2*2) 3 := fun _ _ => 1
noncomputable def b3 : Vec 3 := fun _ => 1
noncomputable def W4 : Mat 3 3 := fun _ _ => 1
noncomputable def b4 : Vec 3 := fun _ => 1
noncomputable def W5 : Mat 3 10 := fun _ _ => 1
noncomputable def b5 : Vec 10 := fun _ => 0

/-- conv1 in closed form: bias plus the (unit-tap) input pixel. -/
theorem conv1_eq (o : Fin 2) (hi wi : Fin (2*2)) :
    conv2d W1 b1 T0 o hi wi = b1 o + T0 0 hi wi := by
  rw [conv2d_1x1, Fin.sum_univ_one]; simp [W1]

/-- conv1 is everywhere positive (bias ≥ 1, pixel ≥ 0). -/
theorem conv1_pos (o : Fin 2) (hi wi : Fin (2*2)) : 0 < conv2d W1 b1 T0 o hi wi := by
  rw [conv1_eq]
  have hb : (0:ℝ) < b1 o := by simp only [b1]; split <;> norm_num
  have ht : (0:ℝ) ≤ T0 0 hi wi := by simp only [T0]; positivity
  linarith

/-- conv2 ∘ conv1 in closed form. -/
theorem conv2_eq (o : Fin 2) (hi wi : Fin (2*2)) :
    conv2d W2 b2 (conv2d W1 b1 T0) o hi wi
      = b2 o + (W2 o 0 0 0 * (b1 0 + T0 0 hi wi) + W2 o 1 0 0 * (b1 1 + T0 0 hi wi)) := by
  rw [conv2d_1x1, Fin.sum_univ_two, conv1_eq, conv1_eq]

/-- conv2 ∘ conv1 is everywhere positive. -/
theorem conv2_pos (o : Fin 2) (hi wi : Fin (2*2)) :
    0 < conv2d W2 b2 (conv2d W1 b1 T0) o hi wi := by
  rw [conv2d_1x1]
  have hb : (0:ℝ) < b2 o := by simp only [b2]; norm_num
  have hs : (0:ℝ) ≤ ∑ i : Fin 2, W2 o i 0 0 * conv2d W1 b1 T0 i hi wi := by
    apply Finset.sum_nonneg
    intro i _
    apply mul_nonneg
    · simp only [W2]; split <;> norm_num
    · exact le_of_lt (conv1_pos i hi wi)
  linarith

/-- The max-pool input (`conv2 ∘ conv1`) is positionally injective on each
    channel: distinct positions give distinct values (the conv stack is
    affine with a strictly-positive coefficient on the injective input). -/
theorem poolTensor_inj (ci : Fin 2) (r r' s s' : Fin (2*2))
    (heq : conv2d W2 b2 (conv2d W1 b1 T0) ci r s
         = conv2d W2 b2 (conv2d W1 b1 T0) ci r' s') :
    r = r' ∧ s = s' := by
  rw [conv2_eq, conv2_eq] at heq
  have key : T0 0 r s = T0 0 r' s' := by
    fin_cases ci <;> (simp [W2, b1, b2] at heq; linarith)
  simp only [T0] at key
  rw [Nat.cast_inj] at key
  have hr := r.isLt; have hs := s.isLt; have hr' := r'.isLt; have hs' := s'.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

/-- `flatConv W1 b1 X = flatten (conv2d W1 b1 T0)` (the input round-trips
    through `unflatten ∘ flatten`). -/
theorem flatConv1_eq : flatConv W1 b1 X = Tensor3.flatten (conv2d W1 b1 T0) := by
  simp only [flatConv, X, Tensor3.unflatten_flatten]

/-- Second conv layer, post unflatten/flatten round-trip. -/
theorem convZ_eq :
    flatConv W2 b2 (Tensor3.flatten (conv2d W1 b1 T0))
      = Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0)) := by
  simp only [flatConv, Tensor3.unflatten_flatten]

/-- First conv→relu block: ReLU is the identity (conv1 is positive). -/
theorem block1_eq :
    (relu (2 * (2*2) * (2*2)) ∘ flatConv W1 b1) X
      = Tensor3.flatten (conv2d W1 b1 T0) := by
  simp only [Function.comp_apply, flatConv1_eq]
  exact relu_id_of_pos (fun k => flatten_pos_of_pos (fun o hi wi => conv1_pos o hi wi) k)

/-- Both conv→relu blocks fold (ReLUs are identities) to the flattened
    `conv2 ∘ conv1` — the tensor handed to max-pool. -/
theorem blockZ_eq :
    ((relu (2 * (2*2) * (2*2)) ∘ flatConv W2 b2) ∘
      (relu (2 * (2*2) * (2*2)) ∘ flatConv W1 b1)) X
      = Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0)) := by
  rw [Function.comp_apply, block1_eq, Function.comp_apply, convZ_eq]
  exact relu_id_of_pos (fun k => flatten_pos_of_pos (fun o hi wi => conv2_pos o hi wi) k)

/-- The pooled vector in closed form. -/
theorem pooled_eq :
    maxPoolFlat 2 2 2 (Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0)))
      = Tensor3.flatten (maxPool2 (conv2d W2 b2 (conv2d W1 b1 T0))) := by
  simp only [maxPoolFlat, Tensor3.unflatten_flatten]

/-- The pooled vector is everywhere positive. -/
theorem pooled_pos (i : Fin (2*2*2)) :
    0 < maxPoolFlat 2 2 2 (Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0))) i := by
  rw [pooled_eq]
  exact flatten_pos_of_pos
    (fun ci hi wi => maxPool2_pos (fun o r s => conv2_pos o r s) ci hi wi) i

/-- The first dense layer's output is everywhere positive. -/
theorem dense3_pos (j : Fin 3) :
    0 < dense W3 b3
      (maxPoolFlat 2 2 2 (Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0)))) j :=
  dense_pos_of_nonneg (fun _ _ => by simp [W3]) (fun _ => by simp [b3])
    (fun i => le_of_lt (pooled_pos i)) j

/-- **Unconditional whole-network VJP for a multi-channel, multi-window,
    10-class CNN.** Every smoothness hypothesis of
    `mnistCnnNoBn_has_vjp_at` is discharged — the no-tie condition via
    `maxPool2Smooth_of_injective`, the ReLU conditions via positivity —
    so the statement carries no side conditions and stays in the
    three-axiom closure. -/
noncomputable def miniCnn_has_vjp_at :
    HasVJPAt (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X :=
  mnistCnnNoBn_has_vjp_at W1 b1 W2 b2 W3 b3 W4 b4 W5 b5
    (by norm_num) (by norm_num) (by norm_num) X
    -- h1: conv1 preactivation nonzero
    (by intro k; rw [flatConv1_eq]
        exact ne_of_gt (flatten_pos_of_pos (fun o hi wi => conv1_pos o hi wi) k))
    -- h2: conv2 preactivation nonzero
    (by intro k; rw [block1_eq, convZ_eq]
        exact ne_of_gt (flatten_pos_of_pos (fun o hi wi => conv2_pos o hi wi) k))
    -- h_mp: no MaxPool ties (positional injectivity on the pool input)
    (by rw [blockZ_eq, Tensor3.unflatten_flatten]
        exact maxPool2Smooth_of_injective _
          (fun ci r r' s s' h => poolTensor_inj ci r r' s s' h))
    -- h3: dense3 preactivation nonzero
    (by intro k; rw [blockZ_eq]; exact ne_of_gt (dense3_pos k))
    -- h4: dense4 preactivation nonzero
    (by intro k
        rw [blockZ_eq, Function.comp_apply, relu_id_of_pos (fun i => dense3_pos i)]
        exact ne_of_gt (dense_pos_of_nonneg (fun _ _ => by simp [W4]) (fun _ => by simp [b4])
          (fun i => le_of_lt (dense3_pos i)) k))

/-- **Public unconditional correctness theorem** — the Tier-1 CNN's
    backward equals the `pdiv`-Jacobian VJP, no hypotheses. -/
theorem miniCnn_has_vjp_correct (dy : Vec 10) (i : Fin (1 * (2*2) * (2*2))) :
    miniCnn_has_vjp_at.backward dy i =
      ∑ j : Fin 10, pdiv (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X i j * dy j :=
  miniCnn_has_vjp_at.correct dy i

end Mini

-- ════════════════════════════════════════════════════════════════
-- § Tier-2 instance — same multi-channel/multi-window/10-class CNN, now
--   with genuine 3×3 SAME-padding convolutions
--
-- Identical in spirit to `Mini`, but the two conv layers use 3×3 kernels
-- (center-structured, via `conv2d_center3x3`), so the forward exercises
-- real spatial convolution — the nine-tap sum and the SAME-padding
-- branches — not just a per-pixel 1×1 channel mix. The smoothness
-- discharge is unchanged: `maxPool2Smooth_of_injective` for the no-tie
-- condition, positivity propagation for the ReLU conditions, all inside
-- the three-axiom closure.
-- ════════════════════════════════════════════════════════════════

namespace Spatial

/-- Input tensor, 16 distinct strictly-positive values (positionally
    injective). -/
noncomputable def T0 : Tensor3 1 (2*2) (2*2) :=
  fun _ hi wi => ((4 * hi.val + wi.val + 1 : ℕ) : ℝ)
noncomputable def X : Vec (1 * (2*2) * (2*2)) := Tensor3.flatten T0
/-- conv1: 1→2 channels, 3×3, center tap `1`, zero elsewhere. -/
noncomputable def W1 : Kernel4 2 1 3 3 := fun _ _ kh kw => if kh = 1 ∧ kw = 1 then 1 else 0
noncomputable def b1 : Vec 2 := fun o => if o = 0 then 1 else 2
/-- conv2: 2→2 channels, 3×3, center tap depends on the output channel
    (`1` for channel 0, `2` for channel 1), zero elsewhere. -/
noncomputable def W2 : Kernel4 2 2 3 3 :=
  fun o _ kh kw => if kh = 1 ∧ kw = 1 then (if o = 0 then 1 else 2) else 0
noncomputable def b2 : Vec 2 := fun _ => 1
noncomputable def W3 : Mat (2*2*2) 3 := fun _ _ => 1
noncomputable def b3 : Vec 3 := fun _ => 1
noncomputable def W4 : Mat 3 3 := fun _ _ => 1
noncomputable def b4 : Vec 3 := fun _ => 1
noncomputable def W5 : Mat 3 10 := fun _ _ => 1
noncomputable def b5 : Vec 10 := fun _ => 0

/-- conv1 vanishes off the center tap (the `conv2d_center3x3` hypothesis). -/
theorem hW1 (o : Fin 2) (c : Fin 1) (kh kw : Fin 3) (hne : ¬(kh = 1 ∧ kw = 1)) :
    W1 o c kh kw = 0 := by simp only [W1]; exact if_neg hne
theorem hW2 (o c : Fin 2) (kh kw : Fin 3) (hne : ¬(kh = 1 ∧ kw = 1)) :
    W2 o c kh kw = 0 := by simp only [W2]; exact if_neg hne
/-- conv1 center tap is `1`. -/
theorem W1_center (o : Fin 2) (c : Fin 1) : W1 o c 1 1 = 1 := by simp [W1]

/-- conv1 in closed form. -/
theorem conv1_eq (o : Fin 2) (hi wi : Fin (2*2)) :
    conv2d W1 b1 T0 o hi wi = b1 o + T0 0 hi wi := by
  rw [conv2d_center3x3 W1 b1 hW1, Fin.sum_univ_one, W1_center, one_mul]

theorem conv1_pos (o : Fin 2) (hi wi : Fin (2*2)) : 0 < conv2d W1 b1 T0 o hi wi := by
  rw [conv1_eq]
  have hb : (0:ℝ) < b1 o := by simp only [b1]; split <;> norm_num
  have ht : (0:ℝ) ≤ T0 0 hi wi := by simp only [T0]; positivity
  linarith

/-- conv2 ∘ conv1 in closed form. -/
theorem conv2_eq (o : Fin 2) (hi wi : Fin (2*2)) :
    conv2d W2 b2 (conv2d W1 b1 T0) o hi wi
      = b2 o + (W2 o 0 1 1 * (b1 0 + T0 0 hi wi) + W2 o 1 1 1 * (b1 1 + T0 0 hi wi)) := by
  rw [conv2d_center3x3 W2 b2 hW2, Fin.sum_univ_two, conv1_eq, conv1_eq]

theorem conv2_pos (o : Fin 2) (hi wi : Fin (2*2)) :
    0 < conv2d W2 b2 (conv2d W1 b1 T0) o hi wi := by
  rw [conv2d_center3x3 W2 b2 hW2]
  have hb : (0:ℝ) < b2 o := by simp only [b2]; norm_num
  have hs : (0:ℝ) ≤ ∑ i : Fin 2, W2 o i 1 1 * conv2d W1 b1 T0 i hi wi := by
    apply Finset.sum_nonneg
    intro i _
    apply mul_nonneg
    · rw [show W2 o i 1 1 = (if o = 0 then (1:ℝ) else 2) from by simp [W2]]
      split <;> norm_num
    · exact le_of_lt (conv1_pos i hi wi)
  linarith

/-- The max-pool input is positionally injective on each channel. -/
theorem poolTensor_inj (ci : Fin 2) (r r' s s' : Fin (2*2))
    (heq : conv2d W2 b2 (conv2d W1 b1 T0) ci r s
         = conv2d W2 b2 (conv2d W1 b1 T0) ci r' s') :
    r = r' ∧ s = s' := by
  rw [conv2_eq, conv2_eq] at heq
  have key : T0 0 r s = T0 0 r' s' := by
    fin_cases ci <;> (simp [W2, b1, b2] at heq; linarith)
  simp only [T0] at key
  rw [Nat.cast_inj] at key
  have hr := r.isLt; have hs := s.isLt; have hr' := r'.isLt; have hs' := s'.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

theorem flatConv1_eq : flatConv W1 b1 X = Tensor3.flatten (conv2d W1 b1 T0) := by
  simp only [flatConv, X, Tensor3.unflatten_flatten]

theorem convZ_eq :
    flatConv W2 b2 (Tensor3.flatten (conv2d W1 b1 T0))
      = Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0)) := by
  simp only [flatConv, Tensor3.unflatten_flatten]

theorem block1_eq :
    (relu (2 * (2*2) * (2*2)) ∘ flatConv W1 b1) X
      = Tensor3.flatten (conv2d W1 b1 T0) := by
  simp only [Function.comp_apply, flatConv1_eq]
  exact relu_id_of_pos (fun k => flatten_pos_of_pos (fun o hi wi => conv1_pos o hi wi) k)

theorem blockZ_eq :
    ((relu (2 * (2*2) * (2*2)) ∘ flatConv W2 b2) ∘
      (relu (2 * (2*2) * (2*2)) ∘ flatConv W1 b1)) X
      = Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0)) := by
  rw [Function.comp_apply, block1_eq, Function.comp_apply, convZ_eq]
  exact relu_id_of_pos (fun k => flatten_pos_of_pos (fun o hi wi => conv2_pos o hi wi) k)

theorem pooled_eq :
    maxPoolFlat 2 2 2 (Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0)))
      = Tensor3.flatten (maxPool2 (conv2d W2 b2 (conv2d W1 b1 T0))) := by
  simp only [maxPoolFlat, Tensor3.unflatten_flatten]

theorem pooled_pos (i : Fin (2*2*2)) :
    0 < maxPoolFlat 2 2 2 (Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0))) i := by
  rw [pooled_eq]
  exact flatten_pos_of_pos
    (fun ci hi wi => maxPool2_pos (fun o r s => conv2_pos o r s) ci hi wi) i

theorem dense3_pos (j : Fin 3) :
    0 < dense W3 b3
      (maxPoolFlat 2 2 2 (Tensor3.flatten (conv2d W2 b2 (conv2d W1 b1 T0)))) j :=
  dense_pos_of_nonneg (fun _ _ => by simp [W3]) (fun _ => by simp [b3])
    (fun i => le_of_lt (pooled_pos i)) j

/-- **Unconditional whole-network VJP for a 3×3-convolution CNN.** Same
    shape as `Mini.miniCnn` (2 channels, eight pool windows, 10 classes)
    but with genuine 3×3 SAME-padding convolutions, every smoothness
    hypothesis discharged, inside the three-axiom closure. -/
noncomputable def spatialCnn_has_vjp_at :
    HasVJPAt (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X :=
  mnistCnnNoBn_has_vjp_at W1 b1 W2 b2 W3 b3 W4 b4 W5 b5
    (by norm_num) (by norm_num) (by norm_num) X
    (by intro k; rw [flatConv1_eq]
        exact ne_of_gt (flatten_pos_of_pos (fun o hi wi => conv1_pos o hi wi) k))
    (by intro k; rw [block1_eq, convZ_eq]
        exact ne_of_gt (flatten_pos_of_pos (fun o hi wi => conv2_pos o hi wi) k))
    (by rw [blockZ_eq, Tensor3.unflatten_flatten]
        exact maxPool2Smooth_of_injective _
          (fun ci r r' s s' h => poolTensor_inj ci r r' s s' h))
    (by intro k; rw [blockZ_eq]; exact ne_of_gt (dense3_pos k))
    (by intro k
        rw [blockZ_eq, Function.comp_apply, relu_id_of_pos (fun i => dense3_pos i)]
        exact ne_of_gt (dense_pos_of_nonneg (fun _ _ => by simp [W4]) (fun _ => by simp [b4])
          (fun i => le_of_lt (dense3_pos i)) k))

/-- **Public unconditional correctness theorem** — the 3×3-conv CNN's
    backward equals the `pdiv`-Jacobian VJP, no hypotheses. -/
theorem spatialCnn_has_vjp_correct (dy : Vec 10) (i : Fin (1 * (2*2) * (2*2))) :
    spatialCnn_has_vjp_at.backward dy i =
      ∑ j : Fin 10, pdiv (mnistCnnNoBnForward W1 b1 W2 b2 W3 b3 W4 b4 W5 b5) X i j * dy j :=
  spatialCnn_has_vjp_at.correct dy i

end Spatial

end Proofs
