import LeanMlir.Proofs.MnistCNN

/-! # Chapter 5: CIFAR-10 2D CNN (no BatchNorm) — whole-network VJP

The Chapter-5 demo model `cifarCnn` (the no-BN spec of `MainCifarCnnTrain`):

  conv 3→32 (relu) → conv 32→32 (relu) → maxPool 2×2 →
  conv 32→64 (relu) → conv 64→64 (relu) → maxPool 2×2 → flatten →
  dense 4096→512 (relu) → dense 512→512 (relu) → dense 512→10

i.e. **two** conv→conv→maxPool stages (channels 3→32→32, then 32→64→64) and
a three-layer dense head, on 32×32 RGB input with two 2×2 pools (32→16→8).

This is the Chapter-4 `mnistCnnNoBn` machinery scaled up: the same
`convRelu`/`denseRelu`/`maxPoolFlat` building blocks, chained through
`vjp_comp_at`, just longer and with **two** maxpool steps. Spatial bookkeeping
uses the final pooled size `(h, w)` as the unit: the second conv stage runs at
`(2h, 2w)`, the first at `(2·(2h), 2·(2w))` — exactly the Chapter-4 `(2h, 2w)`
convention nested one level deeper (so the two pools read `maxPoolFlat _ (2h) (2w)`
and `maxPoolFlat _ h w`).

* `cifarCnn_has_vjp_at` — the **structural** whole-network VJP: the composed
  backward equals the `pdiv`-Jacobian VJP of the full forward pass, *conditional*
  on smoothness at the six ReLU kinks and the two MaxPools. The Chapter-5 sibling
  of `mnistCnnNoBn_has_vjp_at`.

* `Tiny.cifarTinyCnn_has_vjp_correct` — a **concrete instance** where every
  smoothness hypothesis is *discharged* (positivity + positional injectivity
  through both pools), so the statement is **unconditional** and closes under the
  three-axiom kernel. The non-vacuity witness for the conditional capstone. -/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Chapter-5 forward pass (BN-free, two conv→conv→pool stages)
-- ════════════════════════════════════════════════════════════════

/-- The Chapter-5 `cifarCnn` forward, in flattened `Vec` space. The conv stack
    runs at spatial `(2·(2h), 2·(2w))`; the first `maxPool` halves it to
    `(2h, 2w)` (where the second conv stage runs), the second to `(h, w)`; then
    three dense layers (two with ReLU). With the real CIFAR shapes
    `ic=3, c1=32, c2=64, h=w=8, d1=512, nClasses=10, kH=kW=3` the input width is
    `3·32·32 = 3072` and the flattened pool output is `64·8·8 = 4096`. -/
noncomputable def cifarCnnForward
    {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) :
    Vec (ic * (2*(2*h)) * (2*(2*w))) → Vec nClasses :=
  dense W₇ b₇
  ∘ (relu d1 ∘ dense W₆ b₆)
  ∘ (relu d1 ∘ dense W₅ b₅)
  ∘ maxPoolFlat c2 h w
  ∘ (relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄)
  ∘ (relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃)
  ∘ maxPoolFlat c1 (2*h) (2*w)
  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w))) ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w))) ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)

-- ════════════════════════════════════════════════════════════════
-- § Structural whole-network VJP (Chapter-5 capstone, conditional)
-- ════════════════════════════════════════════════════════════════

/-- **CIFAR 2D CNN (no BN) whole-network VJP at a smooth point.**

    The composed backward of the full Chapter-5 forward equals the
    `pdiv`-contracted Jacobian, conditional on smoothness at the six ReLU kinks
    and the two MaxPools. Built by `vjp_comp_at` through
    `convRelu → convRelu → maxPool → convRelu → convRelu → maxPool →
     denseRelu → denseRelu → dense`. The Chapter-5 sibling of
    `mnistCnnNoBn_has_vjp_at` (two conv stages, two pools). -/
noncomputable def cifarCnn_has_vjp_at
    {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*(2*h)) * (2*(2*w))))
    (h1 : ∀ k, flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁ x k ≠ 0)
    (h2 : ∀ k, flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂
            ((relu (c1 * (2*(2*h)) * (2*(2*w)))
              ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁) x) k ≠ 0)
    (h_mp1 : MaxPool2Smooth (Tensor3.unflatten
            (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
              ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x)
            : Tensor3 c1 (2*(2*h)) (2*(2*w))))
    (h3 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₃ b₃
            (maxPoolFlat c1 (2*h) (2*w)
              (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                  ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                  ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x)) k ≠ 0)
    (h4 : ∀ k, flatConv (h := 2*h) (w := 2*w) W₄ b₄
            ((relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃)
              (maxPoolFlat c1 (2*h) (2*w)
                (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x))) k ≠ 0)
    (h_mp2 : MaxPool2Smooth (Tensor3.unflatten
            (((relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄)
              ∘ (relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃))
              (maxPoolFlat c1 (2*h) (2*w)
                (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x)))
            : Tensor3 c2 (2*h) (2*w)))
    (h5 : ∀ k, dense W₅ b₅ (maxPoolFlat c2 h w
            (((relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄)
              ∘ (relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃))
              (maxPoolFlat c1 (2*h) (2*w)
                (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x)))) k ≠ 0)
    (h6 : ∀ k, dense W₆ b₆ ((relu d1 ∘ dense W₅ b₅) (maxPoolFlat c2 h w
            (((relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄)
              ∘ (relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃))
              (maxPoolFlat c1 (2*h) (2*w)
                (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x))))) k ≠ 0) :
    HasVJPAt (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇) x := by
  unfold cifarCnnForward
  -- conv→relu block 1 at x
  have s1 := convRelu_has_vjp_at W₁ b₁ x h1
  have s1d := convRelu_differentiableAt W₁ b₁ x h1
  -- conv→relu block 2 at (block-1 output)
  have s2v := convRelu_has_vjp_at W₂ b₂ _ h2
  have s2d2 := convRelu_differentiableAt W₂ b₂ _ h2
  have s2 := vjp_comp_at _ _ x s1d s2d2 s1 s2v
  have s2d := s2d2.comp x s1d
  -- maxpool 1 at (block-2 output); align the point via flatten ∘ unflatten = id
  set zmp1 := (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
              ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x) with hzmp1
  have hpt1 : Tensor3.flatten (Tensor3.unflatten zmp1 : Tensor3 c1 (2*(2*h)) (2*(2*w))) = zmp1 :=
    Tensor3.flatten_unflatten zmp1
  have mp1_v : HasVJPAt (maxPoolFlat c1 (2*h) (2*w)) zmp1 := by
    rw [← hpt1]; exact maxPoolFlat_has_vjp_at _ h_mp1
  have mp1_d : DifferentiableAt ℝ (maxPoolFlat c1 (2*h) (2*w)) zmp1 := by
    rw [← hpt1]; exact maxPoolFlat_differentiableAt _ h_mp1 hc1 (by omega) (by omega)
  have s3 := vjp_comp_at _ _ x s2d mp1_d s2 mp1_v
  have s3d := mp1_d.comp x s2d
  -- conv→relu block 3 at (pool-1 output)
  set zp1 := maxPoolFlat c1 (2*h) (2*w) zmp1 with hzp1
  have s4v := convRelu_has_vjp_at W₃ b₃ zp1 h3
  have s4d3 := convRelu_differentiableAt W₃ b₃ zp1 h3
  have s4 := vjp_comp_at _ _ x s3d s4d3 s3 s4v
  have s4d := s4d3.comp x s3d
  -- conv→relu block 4 at (block-3 output)
  set z3 := (relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃) zp1 with hz3
  have s5v := convRelu_has_vjp_at W₄ b₄ z3 h4
  have s5d4 := convRelu_differentiableAt W₄ b₄ z3 h4
  have s5 := vjp_comp_at _ _ x s4d s5d4 s4 s5v
  have s5d := s5d4.comp x s4d
  -- maxpool 2 at (block-4 output)
  set zmp2 := (relu (c2 * (2*h) * (2*w)) ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄) z3 with hzmp2
  have hpt2 : Tensor3.flatten (Tensor3.unflatten zmp2 : Tensor3 c2 (2*h) (2*w)) = zmp2 :=
    Tensor3.flatten_unflatten zmp2
  have mp2_v : HasVJPAt (maxPoolFlat c2 h w) zmp2 := by
    rw [← hpt2]; exact maxPoolFlat_has_vjp_at _ h_mp2
  have mp2_d : DifferentiableAt ℝ (maxPoolFlat c2 h w) zmp2 := by
    rw [← hpt2]; exact maxPoolFlat_differentiableAt _ h_mp2 hc2 hh hw
  have s6 := vjp_comp_at _ _ x s5d mp2_d s5 mp2_v
  have s6d := mp2_d.comp x s5d
  -- dense→relu block 5 at (pool-2 output)
  set zp2 := maxPoolFlat c2 h w zmp2 with hzp2
  have s7v := denseRelu_has_vjp_at W₅ b₅ zp2 h5
  have s7d5 := denseRelu_differentiableAt W₅ b₅ zp2 h5
  have s7 := vjp_comp_at _ _ x s6d s7d5 s6 s7v
  have s7d := s7d5.comp x s6d
  -- dense→relu block 6 at (block-5 output)
  set z5 := (relu d1 ∘ dense W₅ b₅) zp2 with hz5
  have s8v := denseRelu_has_vjp_at W₆ b₆ z5 h6
  have s8d6 := denseRelu_differentiableAt W₆ b₆ z5 h6
  have s8 := vjp_comp_at _ _ x s7d s8d6 s7 s8v
  have s8d := s8d6.comp x s7d
  -- final dense (linear, no smoothness)
  exact vjp_comp_at _ _ x s8d ((dense_differentiable W₇ b₇) _) s8
    ((dense_has_vjp W₇ b₇).toHasVJPAt _)

/-- **Public correctness theorem for `cifarCnn_has_vjp_at`** — the Chapter-5
    CIFAR CNN's backward equals the `pdiv`-contracted Jacobian. -/
theorem cifarCnn_has_vjp_at_correct
    {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*(2*h)) * (2*(2*w))))
    (h1 h2 h_mp1 h3 h4 h_mp2 h5 h6)
    (dy : Vec nClasses) (i : Fin (ic * (2*(2*h)) * (2*(2*w)))) :
    (cifarCnn_has_vjp_at W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇
        hc1 hc2 hh hw x h1 h2 h_mp1 h3 h4 h_mp2 h5 h6).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇) x i j * dy j :=
  (cifarCnn_has_vjp_at W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇
      hc1 hc2 hh hw x h1 h2 h_mp1 h3 h4 h_mp2 h5 h6).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § Concrete tiny instance — every smoothness hypothesis DISCHARGED
--
-- A minimal `cifarCnn` at `ic=c1=c2=h=w=d1=1`, `nClasses=2`, `1×1` identity
-- kernels, so the conv stack is the identity, every ReLU sees a positive input,
-- and BOTH maxpools have distinct-valued windows. The crux unique to Chapter 5
-- (two pools) is the SECOND pool's no-tie condition: it needs the FIRST pool's
-- output to be positionally injective. With the 4×4 input `1,…,16` (row-major)
-- the four 2×2 window maxima are `6,8,14,16` (= `8·r+2·s+6`, proved by folding
-- the real `max`es back through `Nat.cast_max` and `omega`), so the pooled 2×2
-- tensor is injective and the second pool is smooth. All eight smoothness
-- hypotheses then discharge, yielding an **unconditional** whole-network VJP —
-- the non-vacuity witness for the conditional capstone, inside the three-axiom
-- closure (no `native_decide`). The Chapter-5 peer of `Mini`/`Spatial`.
-- ════════════════════════════════════════════════════════════════

namespace Tiny

/-- 16 distinct positive values, 4×4 row-major ⇒ positionally injective. Typed
    at the `2·(2·1)` spatial form so it lines up syntactically with the forward's
    stage-1 input shape (no `2*2` vs `2*(2*1)` cast friction). -/
noncomputable def T0 : Tensor3 1 (2*(2*1)) (2*(2*1)) :=
  fun _ hi wi => ((4 * hi.val + wi.val + 1 : ℕ) : ℝ)
/-- Whole-network input, the flattened `T0` (`Vec 16`). -/
noncomputable def X : Vec (1 * (2*(2*1)) * (2*(2*1))) := Tensor3.flatten T0
/-- Every conv: 1→1, 1×1 unit kernel ⇒ identity. -/
noncomputable def K : Kernel4 1 1 1 1 := fun _ _ _ _ => 1
noncomputable def Bz : Vec 1 := fun _ => 0
/-- Dense heads: nonnegative weights + a strictly-positive bias keep every
    activation off the ReLU kink. -/
noncomputable def Wd5 : Mat (1*1*1) 1 := fun _ _ => 1
noncomputable def Wd6 : Mat 1 1 := fun _ _ => 1
noncomputable def Bp : Vec 1 := fun _ => 1
noncomputable def Wd7 : Mat 1 2 := fun _ _ => 1
noncomputable def Bz2 : Vec 2 := fun _ => 0

theorem T0_pos : ∀ (ci : Fin 1) (hi wi : Fin (2*(2*1))), 0 < T0 ci hi wi := by
  intro ci hi wi; simp only [T0]; positivity

/-- `T0` is positionally injective: `4·hi + wi + 1` pins down `(hi, wi)`. -/
theorem T0_inj (ci : Fin 1) (r r' s s' : Fin (2*(2*1)))
    (h : T0 ci r s = T0 ci r' s') : r = r' ∧ s = s' := by
  simp only [T0, Nat.cast_inj] at h
  have hr := r.isLt; have hs := s.isLt; have hr' := r'.isLt; have hs' := s'.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

theorem X_pos : ∀ k, 0 < X k :=
  fun k => flatten_pos_of_pos (fun ci hi wi => T0_pos ci hi wi) k

/-- 1×1 unit conv is the identity (single channel). -/
theorem conv2dK_id {h w : Nat} (t : Tensor3 1 h w) : conv2d K Bz t = t := by
  funext o hi wi
  rw [conv2d_1x1]
  simp only [Bz, K, Fin.sum_univ_one, one_mul, zero_add]
  rw [Fin.fin_one_eq_zero o]

/-- Hence `flatConv K Bz` is the identity at any spatial size. -/
theorem flatConvK_id {h w : Nat} (v : Vec (1*h*w)) :
    flatConv (h := h) (w := w) K Bz v = v := by
  simp only [flatConv, conv2dK_id, Tensor3.flatten_unflatten]

/-- The conv→relu block returns its (positive) input unchanged. -/
theorem CRK_id {h w : Nat} (v : Vec (1*h*w)) (hv : ∀ i, 0 < v i) :
    (relu (1*h*w) ∘ flatConv (h := h) (w := w) K Bz) v = v := by
  simp only [Function.comp_apply, flatConvK_id]
  exact relu_id_of_pos hv

/-- Both stage-1 conv→relu blocks fold (identity convs, positive ReLUs) to `X`. -/
theorem stage1_X :
    ((relu (1*(2*(2*1))*(2*(2*1))) ∘ flatConv (h := 2*(2*1)) (w := 2*(2*1)) K Bz)
      ∘ (relu (1*(2*(2*1))*(2*(2*1))) ∘ flatConv (h := 2*(2*1)) (w := 2*(2*1)) K Bz)) X = X := by
  rw [Function.comp_apply, CRK_id X X_pos, CRK_id X X_pos]

/-- The first pool collapses (identity convs) to the maxpool of `T0`. -/
theorem pool1_eq :
    maxPoolFlat 1 (2*1) (2*1) X = Tensor3.flatten (maxPool2 T0) := by
  simp only [maxPoolFlat, X, Tensor3.unflatten_flatten]

/-- The first pool's output is everywhere positive. -/
theorem pool1X_pos : ∀ k, 0 < maxPoolFlat 1 (2*1) (2*1) X k := by
  intro k; rw [pool1_eq]
  exact flatten_pos_of_pos
    (fun ci hi wi => maxPool2_pos (fun c r s => T0_pos c r s) ci hi wi) k

/-- A positive vector unflattens to a positive tensor. -/
theorem unflatten_pos {c h w : Nat} {v : Vec (c*h*w)} (hv : ∀ k, 0 < v k) :
    ∀ ci hi wi, 0 < Tensor3.unflatten v ci hi wi := by
  intro ci hi wi; simp only [Tensor3.unflatten]; exact hv _

/-- The second pool's input (= the first pool's output, post identity stage-2)
    is everywhere positive. -/
theorem pool2X_pos : ∀ k, 0 < maxPoolFlat 1 1 1 (maxPoolFlat 1 (2*1) (2*1) X) k := by
  intro k
  exact flatten_pos_of_pos
    (fun ci hi wi => maxPool2_pos (fun c r s => unflatten_pos pool1X_pos c r s) ci hi wi) k

/-- Both stage-2 conv→relu blocks fold to the (positive) first-pool output. -/
theorem stage2_pool1 :
    ((relu (1*(2*1)*(2*1)) ∘ flatConv (h := 2*1) (w := 2*1) K Bz)
      ∘ (relu (1*(2*1)*(2*1)) ∘ flatConv (h := 2*1) (w := 2*1) K Bz))
      (maxPoolFlat 1 (2*1) (2*1) X) = maxPoolFlat 1 (2*1) (2*1) X := by
  rw [Function.comp_apply, CRK_id _ pool1X_pos, CRK_id _ pool1X_pos]

/-- **The four first-pool window maxima are `8·r + 2·s + 6`** (i.e. `6,8,14,16`):
    `T0` is strictly increasing in row-major, so each 2×2 window's max is its
    bottom-right corner. Proved by folding the real `max`es back through
    `Nat.cast_max` and discharging the resulting `Nat` identity with `omega`. -/
theorem maxPool2T0_val (r s : Fin (2*1)) :
    maxPool2 T0 0 r s = ((8 * r.val + 2 * s.val + 6 : ℕ) : ℝ) := by
  simp only [maxPool2, T0]
  rw [← Nat.cast_max, ← Nat.cast_max, ← Nat.cast_max, Nat.cast_inj]
  have hr := r.isLt; have hs := s.isLt; omega

/-- Hence the first pool's output is positionally injective (the four window
    maxima are pairwise distinct) — the second pool's no-tie condition. -/
theorem pool1_inj : ∀ (ci : Fin 1) (r r' s s' : Fin (2*1)),
    maxPool2 T0 ci r s = maxPool2 T0 ci r' s' → r = r' ∧ s = s' := by
  intro ci r r' s s' heq
  rw [Fin.fin_one_eq_zero ci, maxPool2T0_val, maxPool2T0_val, Nat.cast_inj] at heq
  have hr := r.isLt; have hs := s.isLt; have hr' := r'.isLt; have hs' := s'.isLt
  exact ⟨Fin.ext (by omega), Fin.ext (by omega)⟩

/-- **Unconditional whole-network VJP for a concrete tiny CIFAR CNN.** Every
    smoothness hypothesis of `cifarCnn_has_vjp_at` is discharged — the two
    no-tie conditions via positional injectivity (`maxPool2Smooth_of_injective`,
    the second through `pool1_inj`), the six ReLU conditions via positivity —
    so the statement carries no side conditions and stays in the three-axiom
    closure. -/
noncomputable def cifarTinyCnn_has_vjp_at :
    HasVJPAt (cifarCnnForward K Bz K Bz K Bz K Bz Wd5 Bp Wd6 Bp Wd7 Bz2) X :=
  cifarCnn_has_vjp_at K Bz K Bz K Bz K Bz Wd5 Bp Wd6 Bp Wd7 Bz2
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) X
    -- h1: conv1 preactivation nonzero
    (by intro k; rw [flatConvK_id]; exact ne_of_gt (X_pos k))
    -- h2: conv2 preactivation nonzero
    (by intro k; rw [CRK_id X X_pos, flatConvK_id]; exact ne_of_gt (X_pos k))
    -- h_mp1: no ties at the first maxpool (positionally-injective input `T0`)
    (by rw [stage1_X, show X = Tensor3.flatten T0 from rfl, Tensor3.unflatten_flatten]
        exact maxPool2Smooth_of_injective _ (fun ci r r' s s' h => T0_inj ci r r' s s' h))
    -- h3: conv3 preactivation nonzero
    (by intro k; rw [stage1_X, flatConvK_id]; exact ne_of_gt (pool1X_pos k))
    -- h4: conv4 preactivation nonzero
    (by intro k; rw [stage1_X, CRK_id _ pool1X_pos, flatConvK_id]
        exact ne_of_gt (pool1X_pos k))
    -- h_mp2: no ties at the second maxpool (first-pool output injective)
    (by rw [stage1_X, stage2_pool1, pool1_eq, Tensor3.unflatten_flatten]
        exact maxPool2Smooth_of_injective _
          (fun ci r r' s s' h => pool1_inj ci r r' s s' h))
    -- h5: dense5 preactivation nonzero
    (by intro k; rw [stage1_X, stage2_pool1]
        exact ne_of_gt (dense_pos_of_nonneg (fun _ _ => by simp [Wd5]) (fun _ => by simp [Bp])
          (fun i => le_of_lt (pool2X_pos i)) k))
    -- h6: dense6 preactivation nonzero
    (by intro k; rw [stage1_X, stage2_pool1]
        have hd5 : ∀ j, 0 < dense Wd5 Bp (maxPoolFlat 1 1 1 (maxPoolFlat 1 (2*1) (2*1) X)) j :=
          fun j => dense_pos_of_nonneg (fun _ _ => by simp [Wd5]) (fun _ => by simp [Bp])
            (fun i => le_of_lt (pool2X_pos i)) j
        rw [Function.comp_apply, relu_id_of_pos hd5]
        exact ne_of_gt (dense_pos_of_nonneg (fun _ _ => by simp [Wd6]) (fun _ => by simp [Bp])
          (fun i => le_of_lt (hd5 i)) k))

/-- **Public unconditional correctness theorem** — the concrete tiny CIFAR CNN's
    backward equals the `pdiv`-Jacobian VJP, no hypotheses. -/
theorem cifarTinyCnn_has_vjp_correct (dy : Vec 2) (i : Fin (1 * (2*(2*1)) * (2*(2*1)))) :
    cifarTinyCnn_has_vjp_at.backward dy i =
      ∑ j : Fin 2,
        pdiv (cifarCnnForward K Bz K Bz K Bz K Bz Wd5 Bp Wd6 Bp Wd7 Bz2) X i j * dy j :=
  cifarTinyCnn_has_vjp_at.correct dy i

end Tiny

-- ════════════════════════════════════════════════════════════════
-- § Chapter-5 BatchNorm variant — conv→BN→relu blocks
--
-- The BN-CIFAR net inserts a per-example BatchNorm (`bnForward`, normalize over
-- the whole oc·h·w feature vec with scalar γ/β — BatchNorm.lean) between each
-- conv and its ReLU. The forward is `cifarCnnForward` with each `convRelu` block
-- replaced by a `convBnRelu` block; the whole-network VJP chains exactly as the
-- no-BN version but through `convBnRelu_has_vjp_at` (which folds the conv, BN,
-- and ReLU VJPs via `vjp_comp_at`). BN is differentiable everywhere (ε > 0), so
-- the only new side conditions are the four `0 < εᵢ` and the smoothness moves to
-- the POST-BN pre-activations.
-- ════════════════════════════════════════════════════════════════

/-- The Chapter-5 **BatchNorm** CIFAR forward: `cifarCnnForward` with a per-example
    `bnForward` inserted between each conv and its ReLU (four BN layers, scalar
    `εᵢ, γᵢ, βᵢ`). -/
noncomputable def cifarCnnBnForward
    {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ γ₁ β₁ : ℝ)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ γ₂ β₂ : ℝ)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ γ₃ β₃ : ℝ)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ γ₄ β₄ : ℝ)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) :
    Vec (ic * (2*(2*h)) * (2*(2*w))) → Vec nClasses :=
  dense W₇ b₇
  ∘ (relu d1 ∘ dense W₆ b₆)
  ∘ (relu d1 ∘ dense W₅ b₅)
  ∘ maxPoolFlat c2 h w
  ∘ (relu (c2 * (2*h) * (2*w))
      ∘ bnForward (c2 * (2*h) * (2*w)) ε₄ γ₄ β₄ ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄)
  ∘ (relu (c2 * (2*h) * (2*w))
      ∘ bnForward (c2 * (2*h) * (2*w)) ε₃ γ₃ β₃ ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃)
  ∘ maxPoolFlat c1 (2*h) (2*w)
  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
      ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₂ γ₂ β₂
        ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
      ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
        ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)

/-- **BN-CIFAR whole-network VJP at a (post-BN) smooth point.** The composed
    backward equals the `pdiv`-contracted Jacobian, conditional on `0 < εᵢ` and
    smoothness at the six ReLU kinks (now reading the post-BN pre-activations)
    and the two MaxPools. Chains `convBnRelu → convBnRelu → maxPool →
    convBnRelu → convBnRelu → maxPool → denseRelu → denseRelu → dense` through
    `vjp_comp_at`. The BatchNorm sibling of `cifarCnn_has_vjp_at`. -/
noncomputable def cifarCnnBn_has_vjp_at
    {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ γ₁ β₁ : ℝ) (hε₁ : 0 < ε₁)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ γ₂ β₂ : ℝ) (hε₂ : 0 < ε₂)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ γ₃ β₃ : ℝ) (hε₃ : 0 < ε₃)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ γ₄ β₄ : ℝ) (hε₄ : 0 < ε₄)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*(2*h)) * (2*(2*w))))
    (h1 : ∀ k, bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
            (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁ x) k ≠ 0)
    (h2 : ∀ k, bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₂ γ₂ β₂
            (flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂
              ((relu (c1 * (2*(2*h)) * (2*(2*w)))
                ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
                  ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁) x)) k ≠ 0)
    (h_mp1 : MaxPool2Smooth (Tensor3.unflatten
            (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₂ γ₂ β₂
                  ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
              ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
                  ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x)
            : Tensor3 c1 (2*(2*h)) (2*(2*w))))
    (h3 : ∀ k, bnForward (c2 * (2*h) * (2*w)) ε₃ γ₃ β₃
            (flatConv (h := 2*h) (w := 2*w) W₃ b₃
              (maxPoolFlat c1 (2*h) (2*w)
                (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₂ γ₂ β₂
                      ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
                      ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x))) k ≠ 0)
    (h4 : ∀ k, bnForward (c2 * (2*h) * (2*w)) ε₄ γ₄ β₄
            (flatConv (h := 2*h) (w := 2*w) W₄ b₄
              ((relu (c2 * (2*h) * (2*w))
                ∘ bnForward (c2 * (2*h) * (2*w)) ε₃ γ₃ β₃ ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃)
                (maxPoolFlat c1 (2*h) (2*w)
                  (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                      ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₂ γ₂ β₂
                        ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                    ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                      ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
                        ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x)))) k ≠ 0)
    (h_mp2 : MaxPool2Smooth (Tensor3.unflatten
            (((relu (c2 * (2*h) * (2*w))
                ∘ bnForward (c2 * (2*h) * (2*w)) ε₄ γ₄ β₄ ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄)
              ∘ (relu (c2 * (2*h) * (2*w))
                ∘ bnForward (c2 * (2*h) * (2*w)) ε₃ γ₃ β₃ ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃))
              (maxPoolFlat c1 (2*h) (2*w)
                (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₂ γ₂ β₂
                      ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
                      ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x)))
            : Tensor3 c2 (2*h) (2*w)))
    (h5 : ∀ k, dense W₅ b₅ (maxPoolFlat c2 h w
            (((relu (c2 * (2*h) * (2*w))
                ∘ bnForward (c2 * (2*h) * (2*w)) ε₄ γ₄ β₄ ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄)
              ∘ (relu (c2 * (2*h) * (2*w))
                ∘ bnForward (c2 * (2*h) * (2*w)) ε₃ γ₃ β₃ ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃))
              (maxPoolFlat c1 (2*h) (2*w)
                (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₂ γ₂ β₂
                      ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
                      ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x)))) k ≠ 0)
    (h6 : ∀ k, dense W₆ b₆ ((relu d1 ∘ dense W₅ b₅) (maxPoolFlat c2 h w
            (((relu (c2 * (2*h) * (2*w))
                ∘ bnForward (c2 * (2*h) * (2*w)) ε₄ γ₄ β₄ ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄)
              ∘ (relu (c2 * (2*h) * (2*w))
                ∘ bnForward (c2 * (2*h) * (2*w)) ε₃ γ₃ β₃ ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃))
              (maxPoolFlat c1 (2*h) (2*w)
                (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₂ γ₂ β₂
                      ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
                  ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                    ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
                      ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x))))) k ≠ 0) :
    HasVJPAt (cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
      W₅ b₅ W₆ b₆ W₇ b₇) x := by
  unfold cifarCnnBnForward
  -- conv→BN→relu block 1 at x
  have s1 := convBnRelu_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ x h1
  have s1d := convBnRelu_differentiableAt W₁ b₁ ε₁ γ₁ β₁ hε₁ x h1
  -- block 2
  have s2v := convBnRelu_has_vjp_at W₂ b₂ ε₂ γ₂ β₂ hε₂ _ h2
  have s2d2 := convBnRelu_differentiableAt W₂ b₂ ε₂ γ₂ β₂ hε₂ _ h2
  have s2 := vjp_comp_at _ _ x s1d s2d2 s1 s2v
  have s2d := s2d2.comp x s1d
  -- maxpool 1
  set zmp1 := (((relu (c1 * (2*(2*h)) * (2*(2*w)))
                ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₂ γ₂ β₂
                  ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₂ b₂)
              ∘ (relu (c1 * (2*(2*h)) * (2*(2*w)))
                ∘ bnForward (c1 * (2*(2*h)) * (2*(2*w))) ε₁ γ₁ β₁
                  ∘ flatConv (h := 2*(2*h)) (w := 2*(2*w)) W₁ b₁)) x) with hzmp1
  have hpt1 : Tensor3.flatten (Tensor3.unflatten zmp1 : Tensor3 c1 (2*(2*h)) (2*(2*w))) = zmp1 :=
    Tensor3.flatten_unflatten zmp1
  have mp1_v : HasVJPAt (maxPoolFlat c1 (2*h) (2*w)) zmp1 := by
    rw [← hpt1]; exact maxPoolFlat_has_vjp_at _ h_mp1
  have mp1_d : DifferentiableAt ℝ (maxPoolFlat c1 (2*h) (2*w)) zmp1 := by
    rw [← hpt1]; exact maxPoolFlat_differentiableAt _ h_mp1 hc1 (by omega) (by omega)
  have s3 := vjp_comp_at _ _ x s2d mp1_d s2 mp1_v
  have s3d := mp1_d.comp x s2d
  -- block 3
  set zp1 := maxPoolFlat c1 (2*h) (2*w) zmp1 with hzp1
  have s4v := convBnRelu_has_vjp_at W₃ b₃ ε₃ γ₃ β₃ hε₃ zp1 h3
  have s4d3 := convBnRelu_differentiableAt W₃ b₃ ε₃ γ₃ β₃ hε₃ zp1 h3
  have s4 := vjp_comp_at _ _ x s3d s4d3 s3 s4v
  have s4d := s4d3.comp x s3d
  -- block 4
  set z3 := (relu (c2 * (2*h) * (2*w))
      ∘ bnForward (c2 * (2*h) * (2*w)) ε₃ γ₃ β₃ ∘ flatConv (h := 2*h) (w := 2*w) W₃ b₃) zp1 with hz3
  have s5v := convBnRelu_has_vjp_at W₄ b₄ ε₄ γ₄ β₄ hε₄ z3 h4
  have s5d4 := convBnRelu_differentiableAt W₄ b₄ ε₄ γ₄ β₄ hε₄ z3 h4
  have s5 := vjp_comp_at _ _ x s4d s5d4 s4 s5v
  have s5d := s5d4.comp x s4d
  -- maxpool 2
  set zmp2 := (relu (c2 * (2*h) * (2*w))
      ∘ bnForward (c2 * (2*h) * (2*w)) ε₄ γ₄ β₄ ∘ flatConv (h := 2*h) (w := 2*w) W₄ b₄) z3 with hzmp2
  have hpt2 : Tensor3.flatten (Tensor3.unflatten zmp2 : Tensor3 c2 (2*h) (2*w)) = zmp2 :=
    Tensor3.flatten_unflatten zmp2
  have mp2_v : HasVJPAt (maxPoolFlat c2 h w) zmp2 := by
    rw [← hpt2]; exact maxPoolFlat_has_vjp_at _ h_mp2
  have mp2_d : DifferentiableAt ℝ (maxPoolFlat c2 h w) zmp2 := by
    rw [← hpt2]; exact maxPoolFlat_differentiableAt _ h_mp2 hc2 hh hw
  have s6 := vjp_comp_at _ _ x s5d mp2_d s5 mp2_v
  have s6d := mp2_d.comp x s5d
  -- dense→relu block 5
  set zp2 := maxPoolFlat c2 h w zmp2 with hzp2
  have s7v := denseRelu_has_vjp_at W₅ b₅ zp2 h5
  have s7d5 := denseRelu_differentiableAt W₅ b₅ zp2 h5
  have s7 := vjp_comp_at _ _ x s6d s7d5 s6 s7v
  have s7d := s7d5.comp x s6d
  -- dense→relu block 6
  set z5 := (relu d1 ∘ dense W₅ b₅) zp2 with hz5
  have s8v := denseRelu_has_vjp_at W₆ b₆ z5 h6
  have s8d6 := denseRelu_differentiableAt W₆ b₆ z5 h6
  have s8 := vjp_comp_at _ _ x s7d s8d6 s7 s8v
  have s8d := s8d6.comp x s7d
  -- final dense
  exact vjp_comp_at _ _ x s8d ((dense_differentiable W₇ b₇) _) s8
    ((dense_has_vjp W₇ b₇).toHasVJPAt _)

/-- **Public correctness theorem for `cifarCnnBn_has_vjp_at`** — the BN-CIFAR
    CNN's backward equals the `pdiv`-contracted Jacobian. -/
theorem cifarCnnBn_has_vjp_at_correct
    {ic c1 c2 h w d1 nClasses kH kW : Nat}
    (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (ε₁ γ₁ β₁ : ℝ) (hε₁ : 0 < ε₁)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (ε₂ γ₂ β₂ : ℝ) (hε₂ : 0 < ε₂)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (ε₃ γ₃ β₃ : ℝ) (hε₃ : 0 < ε₃)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ε₄ γ₄ β₄ : ℝ) (hε₄ : 0 < ε₄)
    (W₅ : Mat (c2 * h * w) d1) (b₅ : Vec d1)
    (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses)
    (hc1 : 0 < c1) (hc2 : 0 < c2) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*(2*h)) * (2*(2*w))))
    (h1 h2 h_mp1 h3 h4 h_mp2 h5 h6)
    (dy : Vec nClasses) (i : Fin (ic * (2*(2*h)) * (2*(2*w)))) :
    (cifarCnnBn_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ W₂ b₂ ε₂ γ₂ β₂ hε₂ W₃ b₃ ε₃ γ₃ β₃ hε₃
        W₄ b₄ ε₄ γ₄ β₄ hε₄ W₅ b₅ W₆ b₆ W₇ b₇ hc1 hc2 hh hw x
        h1 h2 h_mp1 h3 h4 h_mp2 h5 h6).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (cifarCnnBnForward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄
          W₅ b₅ W₆ b₆ W₇ b₇) x i j * dy j :=
  (cifarCnnBn_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ W₂ b₂ ε₂ γ₂ β₂ hε₂ W₃ b₃ ε₃ γ₃ β₃ hε₃
      W₄ b₄ ε₄ γ₄ β₄ hε₄ W₅ b₅ W₆ b₆ W₇ b₇ hc1 hc2 hh hw x
      h1 h2 h_mp1 h3 h4 h_mp2 h5 h6).correct dy i

end Proofs
