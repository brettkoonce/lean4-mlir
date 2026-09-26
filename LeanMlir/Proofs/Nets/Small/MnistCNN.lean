import LeanMlir.Proofs.Architectures.CNN
import LeanMlir.Proofs.Foundation.MLP

/-! # Chapter 3: MNIST 2D CNN (no BatchNorm) — whole-network VJP

The Chapter-3 demo model `mnistCnnNoBn`:

  conv2d 1→c (relu) → conv2d c→c (relu) → maxPool 2×2 → flatten
    → dense (relu) → dense (relu) → dense (identity)

`mnistCnnNoBnHasVJPAt` is the **structural** whole-network VJP: the
composed backward equals the `pdiv`-Jacobian VJP of the full forward
pass, *conditional* on smoothness hypotheses (no ReLU kink / MaxPool
tie at the running activations). The Chapter-3 sibling of
`cnnHasVJPAt`, minus BN and residual blocks.
`TrainedCnn.trainedCnnHasVJPAt` discharges every hypothesis on a reduced
instance of this forward — input a 24×24 MNIST crop 4×4-average-pooled to
6×6, conv 1→2→2 (3×3), dense 18→8→8→10, /128-rational trained weights (the
max-pool no-tie hypothesis needed a pool-tie regulariser during training) —
at one pooled MNIST test digit. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Building blocks: conv→relu and dense→relu (no BN)
-- ════════════════════════════════════════════════════════════════

/-- **conv → relu block VJP at a smooth point** (no BatchNorm).
    `relu ∘ flatConv W b`. The plain-conv analogue of
    `convBnReluHasVJPAt` — conv is linear (global VJP via the
    `HasVJP3` bridge), relu carries the smoothness hypothesis. -/
noncomputable def convReluHasVJPAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, flatConv W b v k ≠ 0) :
    HasVJPAt (relu (oc * h * w) ∘ flatConv W b) v :=
  vjpCompAt (flatConv W b) (relu (oc * h * w)) v
    ((flatConv_differentiable W b) v)
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    ((HasVJP3.toHasVJP (conv2dHasVJP3 W b)).toHasVJPAt v)
    (reluHasVJPAt (oc * h * w) _ h_smooth)

/-- `relu ∘ flatConv W b` is differentiable at a smooth point. -/
theorem convRelu_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (v : Vec (ic * h * w))
    (h_smooth : ∀ k, flatConv W b v k ≠ 0) :
    DifferentiableAt ℝ (relu (oc * h * w) ∘ flatConv W b) v :=
  (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth).comp v
    ((flatConv_differentiable W b) v)

/-- **dense → relu block VJP at a smooth point.** `relu ∘ dense W b`. -/
noncomputable def denseReluHasVJPAt {m n : Nat}
    (W : Mat m n) (b : Vec n) (v : Vec m)
    (h_smooth : ∀ k, dense W b v k ≠ 0) :
    HasVJPAt (relu n ∘ dense W b) v :=
  vjpCompAt (dense W b) (relu n) v
    ((dense_differentiable W b) v)
    (relu_differentiableAt_of_smooth n _ h_smooth)
    ((denseHasVJP W b).toHasVJPAt v)
    (reluHasVJPAt n _ h_smooth)

/-- `relu ∘ dense W b` is differentiable at a smooth point. -/
theorem denseRelu_differentiableAt {m n : Nat}
    (W : Mat m n) (b : Vec n) (v : Vec m)
    (h_smooth : ∀ k, dense W b v k ≠ 0) :
    DifferentiableAt ℝ (relu n ∘ dense W b) v :=
  (relu_differentiableAt_of_smooth n _ h_smooth).comp v ((dense_differentiable W b) v)

-- ════════════════════════════════════════════════════════════════
-- § Chapter-3 forward pass (BN-free)
-- ════════════════════════════════════════════════════════════════

/-- The Chapter-3 `mnistCnnNoBn` forward, in flattened `Vec` space.
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
-- § Structural whole-network VJP (Chapter-3 capstone, conditional)
-- ════════════════════════════════════════════════════════════════

/-- **MNIST 2D CNN (no BN) whole-network VJP at a smooth point.**

    The composed backward of the full Chapter-3 forward equals the
    `pdiv`-contracted Jacobian (Jacobian-transpose applied to the
    cotangent), conditional on smoothness at the four ReLU kinks and
    the one MaxPool. Built by `vjpCompAt` through
    `convRelu → convRelu → maxPool → denseRelu → denseRelu → dense`.
    The Chapter-3 sibling of `cnnHasVJPAt` (BN-free, no resblocks). -/
noncomputable def mnistCnnNoBnHasVJPAt
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
  have s1 := convReluHasVJPAt W₁ b₁ x h1
  have s1d := convRelu_differentiableAt W₁ b₁ x h1
  -- conv→relu block 2 at (block-1 output)
  have s2v := convReluHasVJPAt W₂ b₂ _ h2
  have s2d2 := convRelu_differentiableAt W₂ b₂ _ h2
  have s2 := vjpCompAt _ _ x s1d s2d2 s1 s2v
  have s2d := s2d2.comp x s1d
  -- maxpool at (block-2 output); align the point via flatten ∘ unflatten = id
  set zmp := (((relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₂ b₂)
              ∘ (relu (c * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₁ b₁)) x) with hzmp
  have hpt : Tensor3.flatten (Tensor3.unflatten zmp : Tensor3 c (2*h) (2*w)) = zmp :=
    Tensor3.flatten_unflatten zmp
  have mp_v : HasVJPAt (maxPoolFlat c h w) zmp := by
    rw [← hpt]; exact maxPoolFlatHasVJPAt _ h_mp
  have mp_d : DifferentiableAt ℝ (maxPoolFlat c h w) zmp := by
    rw [← hpt]; exact maxPoolFlat_differentiableAt _ h_mp hc hh hw
  have s3 := vjpCompAt _ _ x s2d mp_d s2 mp_v
  have s3d := mp_d.comp x s2d
  -- dense→relu block 3
  set zd3 := maxPoolFlat c h w zmp with hzd3
  have s4v := denseReluHasVJPAt W₃ b₃ zd3 h3
  have s4d3 := denseRelu_differentiableAt W₃ b₃ zd3 h3
  have s4 := vjpCompAt _ _ x s3d s4d3 s3 s4v
  have s4d := s4d3.comp x s3d
  -- dense→relu block 4
  set zd4 := (relu d1 ∘ dense W₃ b₃) zd3 with hzd4
  have s5v := denseReluHasVJPAt W₄ b₄ zd4 h4
  have s5d4 := denseRelu_differentiableAt W₄ b₄ zd4 h4
  have s5 := vjpCompAt _ _ x s4d s5d4 s4 s5v
  have s5d := s5d4.comp x s4d
  -- final dense (linear, no smoothness)
  exact vjpCompAt _ _ x s5d ((dense_differentiable W₅ b₅) _) s5
    ((denseHasVJP W₅ b₅).toHasVJPAt _)

/-- **Public correctness theorem for `mnistCnnNoBnHasVJPAt`** — the
    Chapter-3 CNN's backward equals the `pdiv`-contracted Jacobian. -/
theorem mnistCnnNoBnHasVJPAt_correct
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
    (mnistCnnNoBnHasVJPAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
        hc hh hw x h1 h2 h_mp h3 h4).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅) x i j * dy j :=
  (mnistCnnNoBnHasVJPAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
      hc hh hw x h1 h2 h_mp h3 h4).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § Reusable discharge lemmas for the smoothness hypotheses
--
-- Discharging `MaxPool2Smooth` by `fin_cases` over each 2×2 window does
-- not scale: `MaxPool2Smooth` is `6·c·h·w` pairwise inequalities, so a realistic spatial size turns it
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
    plus a channel-weighted sum of that same pixel — the closed form the
    1×1-kernel witnesses compute their forward pass with. -/
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

-- ════════════════════════════════════════════════════════════════
-- Chapter-2 MLP: a concrete whole-network instance with every ReLU
-- smoothness hypothesis discharged (the simplest kinked capstone — one
-- non-smooth op, `relu`, two sites). Closes the gap that `mlpHasVJPAt`
-- is never instantiated. Inside the three-axiom closure.
-- ════════════════════════════════════════════════════════════════

namespace MlpConcrete

/-- A concrete 3-layer MLP (`dense → relu → dense → relu → dense`) with
    all-ones weights/biases and a positive input. Every ReLU pre-activation
    is then strictly positive (hence `≠ 0`), so both smoothness hypotheses
    of `mlpHasVJPAt` discharge. The net is non-constant, so this is a
    *live* witness (non-trivial Jacobian), not a degenerate one. -/
noncomputable def W₀ : Mat 2 2 := fun _ _ => 1
noncomputable def b₀ : Vec 2 := fun _ => 1
noncomputable def W₁ : Mat 2 2 := fun _ _ => 1
noncomputable def b₁ : Vec 2 := fun _ => 1
noncomputable def W₂ : Mat 2 2 := fun _ _ => 1
noncomputable def b₂ : Vec 2 := fun _ => 1
noncomputable def x  : Vec 2 := fun _ => 1

/-- First-layer pre-activation is strictly positive at every coordinate. -/
theorem preact0_pos (k : Fin 2) : 0 < dense W₀ b₀ x k :=
  dense_pos_of_nonneg (fun _ _ => by simp [W₀]) (fun _ => by simp [b₀]) (fun _ => by simp [x]) k

/-- **Unconditional whole-network VJP for a concrete 3-layer MLP.** Both
    ReLU `≠ 0` hypotheses are discharged via `dense_pos_of_nonneg`
    (positive bias + nonnegative weights/input propagate strict
    positivity), with `relu_id_of_pos` collapsing the inner ReLU. -/
noncomputable def mlpConcreteHasVJPAt :
    HasVJPAt (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x :=
  mlpHasVJPAt W₀ b₀ W₁ b₁ W₂ b₂ x
    (fun k => ne_of_gt (preact0_pos k))
    (by
      intro k
      rw [relu_id_of_pos (fun i => preact0_pos i)]
      exact ne_of_gt (dense_pos_of_nonneg (fun _ _ => by simp [W₁]) (fun _ => by simp [b₁])
        (fun i => le_of_lt (preact0_pos i)) k))

/-- **Public unconditional correctness theorem** — the concrete MLP's
    backward equals the `pdiv`-Jacobian VJP, no hypotheses. -/
theorem mlpConcreteHasVJP_correct (dy : Vec 2) (i : Fin 2) :
    mlpConcreteHasVJPAt.backward dy i =
      ∑ j : Fin 2, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  mlpConcreteHasVJPAt.correct dy i

end MlpConcrete

-- ════════════════════════════════════════════════════════════════
-- ResNet-style CNN *with* BatchNorm: a concrete whole-network instance
-- with every smoothness hypothesis of `cnnHasVJPAt` discharged. The maxpool `MaxPool2Smooth`
-- forbids the constant trick (constant ties every window), so the stem
-- must be genuinely injective: a 1×1 identity conv on an injective input,
-- with ε chosen so BN's istd is exact (var+ε a perfect square), giving
-- distinct positive BN outputs. Resblocks (post-maxpool, here Vec 1) use
-- BN γ=0 → constant. Inside the three-axiom closure.
-- ════════════════════════════════════════════════════════════════

open Finset BigOperators

/-- BN with γ = 0 collapses to the constant shift β (no input constraint) —
    discharges the resblock smoothness conditions, whose BN inputs need not
    be constant. -/
theorem bnForward_gamma_zero {n : Nat} (ε β : ℝ) (v : Vec n) :
    bnForward n ε 0 β v = (fun _ => β) := by
  funext k; simp [bnForward, bnXhat]

namespace CnnConcrete

-- ic=c=oc=1, h=w=1 (stem spatial 2×2), nClasses=2. Stem: 1×1 identity conv,
-- BN (ε=11/4, γ=1, β=10) ⇒ bn(X) = [9.25,9.75,10.25,10.75] (distinct,
-- positive). Resblocks: all BN γ=0 → constant.
noncomputable def Ws  : Kernel4 1 1 1 1 := fun _ _ _ _ => 1
noncomputable def bs  : Vec 1 := fun _ => 0
noncomputable def X   : Vec (1 * (2*1) * (2*1)) := fun i => (i.val : ℝ)
noncomputable def W₁  : Kernel4 1 1 1 1 := fun _ _ _ _ => 0
noncomputable def b₁  : Vec 1 := fun _ => 0
noncomputable def W₂  : Kernel4 1 1 1 1 := fun _ _ _ _ => 0
noncomputable def b₂  : Vec 1 := fun _ => 0
noncomputable def W₁' : Kernel4 1 1 1 1 := fun _ _ _ _ => 0
noncomputable def b₁' : Vec 1 := fun _ => 0
noncomputable def W₂' : Kernel4 1 1 1 1 := fun _ _ _ _ => 0
noncomputable def b₂' : Vec 1 := fun _ => 0
noncomputable def Wp  : Kernel4 1 1 1 1 := fun _ _ _ _ => 0
noncomputable def bp  : Vec 1 := fun _ => 0
noncomputable def Wd  : Mat 1 2 := fun _ _ => 0
noncomputable def bd  : Vec 2 := fun _ => 0

/-- The 1×1 identity stem conv is the identity on the (flattened) input. -/
theorem flatConv_ws : flatConv (h := 2*1) (w := 2*1) Ws bs X = X := by
  have hc : conv2d Ws bs (Tensor3.unflatten X) = Tensor3.unflatten X := by
    funext o hi wi
    rw [conv2d_1x1]
    simp only [bs, Ws, Fin.sum_univ_one, one_mul, zero_add]
    congr 1
    exact (Fin.fin_one_eq_zero o).symm ▸ rfl
  simp only [flatConv, hc, Tensor3.flatten_unflatten]

theorem bnMean_x : bnMean (1 * (2*1) * (2*1)) X = 3/2 := by
  unfold bnMean
  change (∑ i : Fin 4, X i) / ((4:ℕ):ℝ) = 3/2
  rw [Fin.sum_univ_four]; norm_num [X]

theorem bnVar_x : bnVar (1 * (2*1) * (2*1)) X = 5/4 := by
  unfold bnVar
  rw [bnMean_x]
  change (∑ i : Fin 4, (X i - 3/2) * (X i - 3/2)) / ((4:ℕ):ℝ) = 5/4
  rw [Fin.sum_univ_four]; norm_num [X]

theorem bnIstd_x : bnIstd (1 * (2*1) * (2*1)) X (11/4) = 1/2 := by
  unfold bnIstd
  rw [bnVar_x, show (5/4 + 11/4 : ℝ) = 2^2 by norm_num, Real.sqrt_sq (by norm_num)]

theorem bnX_eq (k : Fin (1 * (2*1) * (2*1))) :
    bnForward (1 * (2*1) * (2*1)) (11/4) 1 10 X k = (X k - 3/2) * (1/2) + 10 := by
  unfold bnForward bnXhat
  rw [bnMean_x, bnIstd_x]; ring

theorem bnX_pos (k : Fin (1 * (2*1) * (2*1))) :
    0 < bnForward (1 * (2*1) * (2*1)) (11/4) 1 10 X k := by
  rw [bnX_eq]
  have hx : 0 ≤ X k := by simp only [X]; positivity
  nlinarith [hx]

theorem bnX_inj : Function.Injective (bnForward (1 * (2*1) * (2*1)) (11/4) 1 10 X) := by
  intro a b hab
  rw [bnX_eq, bnX_eq] at hab
  have hXab : X a = X b := by linarith
  have : (a.val : ℝ) = (b.val : ℝ) := by simpa [X] using hXab
  exact Fin.ext (by exact_mod_cast this)

/-- `cbr X` collapses to `bn X` (identity conv, then relu of a positive). -/
theorem cbr_x : cbr (h := 2*1) (w := 2*1) Ws bs (11/4) 1 10 X
    = bnForward (1 * (2*1) * (2*1)) (11/4) 1 10 X := by
  show relu _ (bnForward _ (11/4) 1 10 (flatConv Ws bs X)) = _
  rw [flatConv_ws]
  exact relu_id_of_pos (fun k => bnX_pos k)

/-- **Whole-network VJP for a concrete ResNet-style CNN with BatchNorm** —
    every smoothness hypothesis discharged: the stem produces distinct
    positive BN outputs (so maxpool has no ties and `bn ≠ 0`), and the
    resblock BNs use γ=0 (constant). -/
noncomputable def cnnConcreteHasVJPAt :
    HasVJPAt (cnnForward Ws bs (11/4) 1 10 W₁ b₁ W₂ b₂ 1 0 1 1 0 1
      W₁' b₁' W₂' b₂' Wp bp 1 0 1 1 0 1 1 0 1 Wd bd) X :=
  cnnHasVJPAt Ws bs (11/4) 1 10 (by norm_num)
    W₁ b₁ W₂ b₂ 1 0 1 1 0 1 (by norm_num) (by norm_num)
    W₁' b₁' W₂' b₂' Wp bp
    1 0 1 1 0 1 1 0 1 (by norm_num) (by norm_num) (by norm_num)
    Wd bd (by norm_num) (by norm_num) (by norm_num) X
    -- h_stem
    (fun k => ne_of_gt (by rw [flatConv_ws]; exact bnX_pos k))
    -- h_mp (maxpool no ties): the stem output is positionally injective
    (by
      rw [cbr_x]
      apply maxPool2Smooth_of_injective
      intro ci r r' s s' heq
      simp only [Tensor3.unflatten] at heq
      simpa [Prod.ext_iff] using bnX_inj heq)
    -- h_rb1
    (fun k => by rw [bnForward_gamma_zero]; norm_num)
    -- h_rb1o
    (fun k => by
      have hmp : 0 < maxPoolFlat 1 1 1 (bnForward (1*(2*1)*(2*1)) (11/4) 1 10 X) k :=
        flatten_pos_of_pos (fun ci hi wi =>
          maxPool2_pos (fun _ _ _ => bnX_pos _) ci hi wi) k
      simp only [Function.comp_apply]
      rw [bnForward_gamma_zero, flatConv_ws, relu_id_of_pos (fun k => bnX_pos k)]
      show (1:ℝ) + maxPoolFlat 1 1 1 (bnForward (1*(2*1)*(2*1)) (11/4) 1 10 X) k ≠ 0
      exact ne_of_gt (by linarith))
    -- h_rb2
    (fun k => by rw [bnForward_gamma_zero]; norm_num)
    -- h_rb2o
    (fun k => by
      simp only [Function.comp_apply]
      rw [bnForward_gamma_zero, bnForward_gamma_zero]; norm_num)

/-- **Public unconditional correctness theorem** — the concrete CNN's
    backward equals the `pdiv`-Jacobian VJP, no hypotheses. -/
theorem cnnConcreteHasVJP_correct (dy : Vec 2) (i : Fin (1 * (2*1) * (2*1))) :
    cnnConcreteHasVJPAt.backward dy i =
      ∑ j : Fin 2, pdiv (cnnForward Ws bs (11/4) 1 10 W₁ b₁ W₂ b₂ 1 0 1 1 0 1
        W₁' b₁' W₂' b₂' Wp bp 1 0 1 1 0 1 1 0 1 Wd bd) X i j * dy j :=
  cnnConcreteHasVJPAt.correct dy i

end CnnConcrete

end Proofs
