import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.SE
import LeanMlir.Proofs.Architectures.LayerNorm

/-!
# EfficientNet — MBConv with Squeeze-Excite, end-to-end VJP

The hardest of the three flagship CNN VJPs in this stack (alongside the
ResNet `cnn_has_vjp_at` and the MobileNet depthwise chain), because the
**squeeze-excite gate** is a genuine fan-out sub-network multiplied back
into the main path.  We reuse `seBlock_has_vjp` (`SE.lean`), which already
carries the product-rule fan-in for `x ⊙ gate(x)`; here we supply the
concrete gate (`seGate`) — a real `Vec → Vec` differentiable map with its
own composed VJP — plus its differentiability.

## What this file provides

* `sigmoid` / `sigmoid_has_vjp` — smooth logistic activation (chosen over
  the kinked h-sigmoid so the gate is differentiable *everywhere*; same
  diagonal-Jacobian proof template as `swish`/`gelu` in `LayerNorm.lean`).
* `broadcastFlat` / `broadcastFlat_has_vjp` — per-channel scalar broadcast
  to spatial layout (the adjoint reindex of `globalAvgPoolFlat`).
* `seGate` / `seGate_has_vjp` — the concrete squeeze-excite gate
  `broadcast ∘ sigmoid ∘ dense ∘ swish ∘ dense ∘ GAP`, assembled with the
  chain rule; fed into `seBlock_has_vjp` to give `seBlockFull_has_vjp`
  (the full `x ⊙ gate(x)`).
* `mbconvBody` / `mbconvBody_has_vjp` — one MBConv block body
  `project(1×1 conv-bn) ∘ SE ∘ depthwise(bn-swish) ∘ expand(1×1 conv-bn-swish)`,
  smooth everywhere (global `HasVJP`).
* `mbconvResidual_has_vjp_at` — the stride-1, `cin = cout` block wrapped in
  the identity residual skip.
* `efficientnet_has_vjp_at` / `_correct` — a representative end-to-end
  EfficientNet (stem → MBConv-with-SE-and-residual → MBConv-with-SE →
  globalAvgPool → dense head), built by `vjp_comp_at`, exposing the
  `pdiv`-contracted Jacobian.  Spatial dims held constant (stride-1; the
  separable striding/pooling plumbing is already in `CNN.lean`).  Only the
  `0 < ε` batch-norm hypotheses are required — swish and sigmoid are
  smooth, so there are no relu-style kink hypotheses anywhere in the block.
-/

namespace Proofs

open Finset BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Sigmoid activation (smooth, logistic)
-- ════════════════════════════════════════════════════════════════

noncomputable def sigmoidScalar (x : ℝ) : ℝ :=
  1 / (1 + Real.exp (-x))

noncomputable def sigmoid (n : Nat) (x : Vec n) : Vec n :=
  fun i => sigmoidScalar (x i)

noncomputable def sigmoidScalarDeriv (x : ℝ) : ℝ :=
  deriv sigmoidScalar x

@[fun_prop]
lemma sigmoidScalar_diff : Differentiable ℝ sigmoidScalar := by
  unfold sigmoidScalar
  intro x
  have h_pos : (0 : ℝ) < 1 + Real.exp (-x) := by positivity
  exact DifferentiableAt.div (differentiableAt_const _) (by fun_prop) h_pos.ne'

lemma sigmoid_diff (D : Nat) : Differentiable ℝ (sigmoid D) := by
  unfold sigmoid; fun_prop

theorem pdiv_sigmoid (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (sigmoid n) x i j =
    if i = j then sigmoidScalarDeriv (x i) else 0 := by
  unfold pdiv
  have h_swap : fderiv ℝ (sigmoid n) x (basisVec i) j =
                fderiv ℝ (fun y : Vec n => sigmoid n y j) x (basisVec i) := by
    rw [fderiv_apply ((sigmoid_diff n) x) j]
    rfl
  rw [h_swap]
  have h_decomp : (fun y : Vec n => sigmoid n y j) =
                  sigmoidScalar ∘ (ContinuousLinearMap.proj j : Vec n →L[ℝ] ℝ) := by
    funext y; rfl
  rw [h_decomp]
  rw [fderiv_comp _ (sigmoidScalar_diff _)
        (ContinuousLinearMap.proj j : Vec n →L[ℝ] ℝ).differentiableAt]
  rw [(ContinuousLinearMap.proj j : Vec n →L[ℝ] ℝ).fderiv]
  simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply]
  rw [fderiv_eq_smul_deriv]
  show basisVec i j • deriv sigmoidScalar (x j) = if i = j then sigmoidScalarDeriv (x i) else 0
  show basisVec i j * deriv sigmoidScalar (x j) = _
  by_cases hij : i = j
  · subst hij
    simp only [if_pos rfl, one_mul]
    rfl
  · have h_basis : basisVec i j = 0 := by
      simp only [basisVec_apply]
      rw [if_neg]; intro heq; exact hij heq.symm
    rw [h_basis, zero_mul, if_neg hij]

noncomputable def sigmoid_has_vjp (n : Nat) : HasVJP (sigmoid n) where
  backward := fun x dy i => dy i * sigmoidScalarDeriv (x i)
  correct := by
    intro x dy i
    simp [pdiv_sigmoid, mul_comm]

theorem sigmoid_has_vjp_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (sigmoid_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (sigmoid n) x i j * dy j :=
  (sigmoid_has_vjp n).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § Broadcast: per-channel scalar → spatial (adjoint of GAP)
-- ════════════════════════════════════════════════════════════════

/-- **Broadcast a per-channel vector back to spatial layout.**
    `broadcastFlat c h w v idx = v (flatChannel c h w idx)` — every spatial
    cell of channel `k` receives `v k`. This is the reindex map along
    `flatChannel`, i.e. the adjoint of `globalAvgPoolFlat` (up to the
    1/(h·w) scale). `Vec c → Vec (c*h*w)`. -/
noncomputable def broadcastFlat (c h w : Nat) : Vec c → Vec (c * h * w) :=
  fun v => fun idx => v (flatChannel c h w idx)

theorem broadcastFlat_differentiable (c h w : Nat) :
    Differentiable ℝ (broadcastFlat c h w) :=
  (reindexCLM (flatChannel c h w)).differentiable

/-- **Broadcast VJP** — linear reindex; backward sums each channel's
    spatial cotangents (the adjoint of broadcast = sum-over-spatial). -/
noncomputable def broadcastFlat_has_vjp (c h w : Nat) :
    HasVJP (broadcastFlat c h w) where
  backward := fun _v dy => fun k =>
    ∑ idx : Fin (c * h * w), (if flatChannel c h w idx = k then dy idx else 0)
  correct := by
    intro v dy k
    show (∑ idx : Fin (c * h * w),
            (if flatChannel c h w idx = k then dy idx else 0)) =
      ∑ j : Fin (c * h * w), pdiv (broadcastFlat c h w) v k j * dy j
    have hpd : ∀ j : Fin (c * h * w),
        pdiv (broadcastFlat c h w) v k j =
          if k = flatChannel c h w j then 1 else 0 := by
      intro j
      exact pdiv_reindex (flatChannel c h w) v k j
    simp_rw [hpd]
    apply Finset.sum_congr rfl
    intro j _
    by_cases hkj : flatChannel c h w j = k
    · rw [if_pos hkj, if_pos hkj.symm, one_mul]
    · rw [if_neg hkj, if_neg (fun he => hkj he.symm), zero_mul]

-- ════════════════════════════════════════════════════════════════
-- § SE gate: squeeze → reduce(swish) → expand → sigmoid → broadcast
-- ════════════════════════════════════════════════════════════════

/-- **The squeeze-excite gate.** Maps `Vec (c*h*w) → Vec (c*h*w)`:
      broadcast ∘ sigmoid ∘ dense(W₂,b₂) ∘ swish ∘ dense(W₁,b₁) ∘ GAP
    Squeeze (GAP `c·h·w → c`), reduce (dense `c → r`), swish, expand
    (dense `r → c`), sigmoid gate, broadcast back to spatial.  Every
    stage is smooth everywhere (swish/sigmoid smooth, dense/GAP/broadcast
    linear-affine), so the gate is differentiable everywhere and has a
    global `HasVJP`. -/
noncomputable def seGate {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  broadcastFlat c h w ∘ sigmoid c ∘ dense W₂ b₂ ∘ swish r ∘
    dense W₁ b₁ ∘ globalAvgPoolFlat c h w

theorem seGate_differentiable {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Differentiable ℝ (seGate (h := h) (w := w) W₁ b₁ W₂ b₂) :=
  (broadcastFlat_differentiable c h w).comp
    ((sigmoid_diff c).comp
      ((dense_differentiable W₂ b₂).comp
        ((swish_diff r).comp
          ((dense_differentiable W₁ b₁).comp
            (globalAvgPoolFlat_differentiable c h w)))))

noncomputable def seGate_has_vjp {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    HasVJP (seGate (h := h) (w := w) W₁ b₁ W₂ b₂) :=
  vjp_comp _ (broadcastFlat c h w)
    ((sigmoid_diff c).comp
      ((dense_differentiable W₂ b₂).comp
        ((swish_diff r).comp
          ((dense_differentiable W₁ b₁).comp
            (globalAvgPoolFlat_differentiable c h w)))))
    (broadcastFlat_differentiable c h w)
    (vjp_comp _ (sigmoid c)
      ((dense_differentiable W₂ b₂).comp
        ((swish_diff r).comp
          ((dense_differentiable W₁ b₁).comp
            (globalAvgPoolFlat_differentiable c h w))))
      (sigmoid_diff c)
      (vjp_comp _ (dense W₂ b₂)
        ((swish_diff r).comp
          ((dense_differentiable W₁ b₁).comp
            (globalAvgPoolFlat_differentiable c h w)))
        (dense_differentiable W₂ b₂)
        (vjp_comp _ (swish r)
          ((dense_differentiable W₁ b₁).comp
            (globalAvgPoolFlat_differentiable c h w))
          (swish_diff r)
          (vjp_comp _ (dense W₁ b₁)
            (globalAvgPoolFlat_differentiable c h w)
            (dense_differentiable W₁ b₁)
            (globalAvgPoolFlat_has_vjp c h w)
            (dense_has_vjp W₁ b₁))
          (swish_has_vjp r))
        (dense_has_vjp W₂ b₂))
      (sigmoid_has_vjp c))
    (broadcastFlat_has_vjp c h w)

/-- **The full SE block** with the concrete gate: `x ⊙ seGate(x)`. -/
noncomputable def seBlockFull {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  seBlock (seGate (h := h) (w := w) W₁ b₁ W₂ b₂)

noncomputable def seBlockFull_has_vjp {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    HasVJP (seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂) :=
  seBlock_has_vjp (seGate (h := h) (w := w) W₁ b₁ W₂ b₂)
    (seGate_differentiable W₁ b₁ W₂ b₂)
    (seGate_has_vjp W₁ b₁ W₂ b₂)

theorem seBlockFull_differentiable {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    Differentiable ℝ (seBlockFull (h := h) (w := w) W₁ b₁ W₂ b₂) := by
  show Differentiable ℝ (seBlock (seGate (h := h) (w := w) W₁ b₁ W₂ b₂))
  generalize hgate : seGate (h := h) (w := w) W₁ b₁ W₂ b₂ = gate
  have hg : Differentiable ℝ gate := hgate ▸ seGate_differentiable W₁ b₁ W₂ b₂
  show Differentiable ℝ (fun x : Vec (c * h * w) => fun i => x i * gate x i)
  apply differentiable_pi.mpr; intro i
  exact (differentiable_apply i).mul (differentiable_pi.mp hg i)

-- ════════════════════════════════════════════════════════════════
-- § conv → bn → swish  (smooth expand stage; swish has no kink)
-- ════════════════════════════════════════════════════════════════

/-- **conv → bn → swish block — everywhere VJP.** Like `convBnRelu` but
    with swish (smooth) instead of relu, so no smoothness hypothesis is
    needed; this is a global `HasVJP`.  `Vec (ic*h*w) → Vec (oc*h*w)`. -/
noncomputable def convBnSwish_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε) :
    HasVJP (swish (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConv W b
      : Vec (ic * h * w) → Vec (oc * h * w)) :=
  vjp_comp (bnForward (oc * h * w) ε γ β ∘ flatConv W b) (swish (oc * h * w))
    (convBn_differentiable W b ε γ β hε)
    (swish_diff (oc * h * w))
    (convBn_has_vjp W b ε γ β hε)
    (swish_has_vjp (oc * h * w))

theorem convBnSwish_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (swish (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConv W b
      : Vec (ic * h * w) → Vec (oc * h * w)) :=
  (swish_diff (oc * h * w)).comp (convBn_differentiable W b ε γ β hε)

-- ════════════════════════════════════════════════════════════════
-- § depthwise → bn → swish  (smooth depthwise stage)
-- ════════════════════════════════════════════════════════════════

/-- **depthwise → bn → swish block — everywhere VJP.** Depthwise conv
    keeps channel count `c`; bn over `c*h*w`; swish smooth.  Global
    `HasVJP`.  `Vec (c*h*w) → Vec (c*h*w)`. -/
noncomputable def dwBnSwish_has_vjp {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε γ β : ℝ) (hε : 0 < ε) :
    HasVJP (swish (c * h * w) ∘ bnForward (c * h * w) ε γ β ∘ depthwiseFlat W b
      : Vec (c * h * w) → Vec (c * h * w)) :=
  vjp_comp (bnForward (c * h * w) ε γ β ∘ depthwiseFlat W b) (swish (c * h * w))
    ((bnForward_differentiable (c * h * w) ε γ β hε).comp (depthwiseFlat_differentiable W b))
    (swish_diff (c * h * w))
    (vjp_comp (depthwiseFlat W b) (bnForward (c * h * w) ε γ β)
      (depthwiseFlat_differentiable W b)
      (bnForward_differentiable (c * h * w) ε γ β hε)
      (depthwiseFlat_has_vjp W b)
      (bn_has_vjp (c * h * w) ε γ β hε))
    (swish_has_vjp (c * h * w))

theorem dwBnSwish_differentiable {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (swish (c * h * w) ∘ bnForward (c * h * w) ε γ β ∘ depthwiseFlat W b
      : Vec (c * h * w) → Vec (c * h * w)) :=
  (swish_diff (c * h * w)).comp
    ((bnForward_differentiable (c * h * w) ε γ β hε).comp (depthwiseFlat_differentiable W b))

-- ════════════════════════════════════════════════════════════════
-- § MBConv block body (no residual): expand → dw → SE → project
-- ════════════════════════════════════════════════════════════════

/-- **MBConv block body** (EfficientNet MBConv with squeeze-excite), in
    flattened `Vec` space:

      project(1×1 conv-bn) ∘ seBlockFull ∘ depthwise(bn-swish) ∘ expand(1×1 conv-bn-swish)

    Channels: `cin → cmid` (expand 1×1), depthwise keeps `cmid`, SE keeps
    `cmid`, project `cmid → cout` (1×1).  Spatial `h, w` constant
    (stride 1).  Every stage is smooth everywhere (swish/sigmoid smooth;
    convs/bn/depthwise/SE differentiable), so the body has a global
    `HasVJP`.  `Vec (cin*h*w) → Vec (cout*h*w)`. -/
noncomputable def mbconvBody {cin cmid cout h w kHe kWe kHd kWd kHp kWp r : Nat}
    -- expand 1×1
    (We : Kernel4 cmid cin kHe kWe) (be : Vec cmid) (εe γe βe : ℝ)
    -- depthwise
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ)
    -- SE (squeeze cmid → r → cmid)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    -- project 1×1
    (Wp : Kernel4 cout cmid kHp kWp) (bp : Vec cout) (εp γp βp : ℝ) :
    Vec (cin * h * w) → Vec (cout * h * w) :=
  (bnForward (cout * h * w) εp γp βp ∘ flatConv Wp bp) ∘
  seBlockFull (h := h) (w := w) Ws₁ bs₁ Ws₂ bs₂ ∘
  (swish (cmid * h * w) ∘ bnForward (cmid * h * w) εd γd βd ∘ depthwiseFlat Wd bd) ∘
  (swish (cmid * h * w) ∘ bnForward (cmid * h * w) εe γe βe ∘ flatConv We be)

noncomputable def mbconvBody_has_vjp {cin cmid cout h w kHe kWe kHd kWd kHp kWp r : Nat}
    (We : Kernel4 cmid cin kHe kWe) (be : Vec cmid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 cout cmid kHp kWp) (bp : Vec cout) (εp γp βp : ℝ) (hεp : 0 < εp) :
    HasVJP (mbconvBody We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp
      : Vec (cin * h * w) → Vec (cout * h * w)) := by
  unfold mbconvBody
  -- s0 : expand (conv-bn-swish)
  set E := swish (cmid * h * w) ∘ bnForward (cmid * h * w) εe γe βe ∘ flatConv We be with hE
  -- s1 : depthwise-bn-swish
  set D := swish (cmid * h * w) ∘ bnForward (cmid * h * w) εd γd βd ∘ depthwiseFlat Wd bd with hD
  -- s2 : SE block
  set S := seBlockFull (h := h) (w := w) Ws₁ bs₁ Ws₂ bs₂ with hS
  -- s3 : project (conv-bn)
  set P := bnForward (cout * h * w) εp γp βp ∘ flatConv Wp bp with hP
  -- differentiability witnesses
  have hE_diff : Differentiable ℝ E := convBnSwish_differentiable We be εe γe βe hεe
  have hD_diff : Differentiable ℝ D := dwBnSwish_differentiable Wd bd εd γd βd hεd
  have hS_diff : Differentiable ℝ S := seBlockFull_differentiable Ws₁ bs₁ Ws₂ bs₂
  have hP_diff : Differentiable ℝ P := convBn_differentiable Wp bp εp γp βp hεp
  -- compose: P ∘ (S ∘ (D ∘ E))
  have hDE : HasVJP (D ∘ E) :=
    vjp_comp E D hE_diff hD_diff
      (convBnSwish_has_vjp We be εe γe βe hεe)
      (dwBnSwish_has_vjp Wd bd εd γd βd hεd)
  have hSDE : HasVJP (S ∘ (D ∘ E)) :=
    vjp_comp (D ∘ E) S (hD_diff.comp hE_diff) hS_diff
      hDE (seBlockFull_has_vjp Ws₁ bs₁ Ws₂ bs₂)
  exact vjp_comp (S ∘ (D ∘ E)) P
    (hS_diff.comp (hD_diff.comp hE_diff)) hP_diff
    hSDE (convBn_has_vjp Wp bp εp γp βp hεp)

theorem mbconvBody_differentiable {cin cmid cout h w kHe kWe kHd kWd kHp kWp r : Nat}
    (We : Kernel4 cmid cin kHe kWe) (be : Vec cmid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 cout cmid kHp kWp) (bp : Vec cout) (εp γp βp : ℝ) (hεp : 0 < εp) :
    Differentiable ℝ (mbconvBody We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp
      : Vec (cin * h * w) → Vec (cout * h * w)) := by
  unfold mbconvBody
  exact (convBn_differentiable Wp bp εp γp βp hεp).comp
    ((seBlockFull_differentiable Ws₁ bs₁ Ws₂ bs₂).comp
      ((dwBnSwish_differentiable Wd bd εd γd βd hεd).comp
        (convBnSwish_differentiable We be εe γe βe hεe)))

-- ════════════════════════════════════════════════════════════════
-- § Residual MBConv (stride-1, cin = cout = c): identity skip
-- ════════════════════════════════════════════════════════════════

/-- **Residual MBConv VJP (global).** When stride is 1 and `cin = cout = c`,
    the MBConv body's input and output shapes match, so the identity skip
    applies: `residual (mbconvBody …)`. The body is differentiable
    everywhere (global `HasVJP`), so the residual VJP is global too. -/
noncomputable def mbconvResidual_has_vjp {c cmid h w kHe kWe kHd kWd kHp kWp r : Nat}
    (We : Kernel4 cmid c kHe kWe) (be : Vec cmid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 c cmid kHp kWp) (bp : Vec c) (εp γp βp : ℝ) (hεp : 0 < εp) :
    HasVJP (residual (mbconvBody (h := h) (w := w)
        We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp)) :=
  residual_has_vjp _
    (mbconvBody_differentiable We be εe γe βe hεe Wd bd εd γd βd hεd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp)
    (mbconvBody_has_vjp We be εe γe βe hεe Wd bd εd γd βd hεd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp)

/-- **Residual MBConv VJP at a point** — the global witness restricted to a
    point. Kept for downstream `_at` consumers. -/
noncomputable def mbconvResidual_has_vjp_at {c cmid h w kHe kWe kHd kWd kHp kWp r : Nat}
    (We : Kernel4 cmid c kHe kWe) (be : Vec cmid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 c cmid kHp kWp) (bp : Vec c) (εp γp βp : ℝ) (hεp : 0 < εp)
    (x : Vec (c * h * w)) :
    HasVJPAt (residual (mbconvBody (h := h) (w := w)
        We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp)) x :=
  (mbconvResidual_has_vjp We be εe γe βe hεe Wd bd εd γd βd hεd
    Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp).toHasVJPAt x

theorem mbconvResidual_differentiable {c cmid h w kHe kWe kHd kWd kHp kWp r : Nat}
    (We : Kernel4 cmid c kHe kWe) (be : Vec cmid) (εe γe βe : ℝ) (hεe : 0 < εe)
    (Wd : DepthwiseKernel cmid kHd kWd) (bd : Vec cmid) (εd γd βd : ℝ) (hεd : 0 < εd)
    (Ws₁ : Mat cmid r) (bs₁ : Vec r) (Ws₂ : Mat r cmid) (bs₂ : Vec cmid)
    (Wp : Kernel4 c cmid kHp kWp) (bp : Vec c) (εp γp βp : ℝ) (hεp : 0 < εp) :
    Differentiable ℝ (residual (mbconvBody (h := h) (w := w)
        We be εe γe βe Wd bd εd γd βd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp)) := by
  unfold residual biPath
  apply differentiable_pi.mpr; intro i
  have hb := mbconvBody_differentiable (h := h) (w := w) We be εe γe βe hεe Wd bd εd γd βd hεd Ws₁ bs₁ Ws₂ bs₂ Wp bp εp γp βp hεp
  exact (differentiable_pi.mp hb i).add (differentiable_apply i)

-- ════════════════════════════════════════════════════════════════
-- § End-to-end representative EfficientNet
-- ════════════════════════════════════════════════════════════════

/-! **Architectural choices (documented).**

We assemble a representative EfficientNet, all in flattened `Vec` space,
spatial dims held constant (stride-1 throughout — pooling/striding is a
separable concern already covered by `maxPoolFlat`/strided conv in
`CNN.lean`; the VJP plumbing is identical):

  stem (3×3 conv-bn-swish, `ic → c`)
    → MBConv₁ **with SE, residual** (stride-1, `c → c` identity skip)
    → MBConv₂ **with SE, no skip** (channel change `c → cout`)
    → globalAvgPool (`cout·h·w → cout`)
    → dense head (`cout → nClasses`)

`MBConv₁` is the headline block: a genuine squeeze-excite gate
(`seBlockFull`) inside an identity residual.  `MBConv₂` exercises the
channel-changing path (no skip).  Both blocks are smooth everywhere
(swish + sigmoid + convs + bn + SE), so only the `0 < ε` batch-norm
hypotheses are needed — no relu-style kink hypotheses.  -/
noncomputable def efficientnetForward
    {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
      kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    -- stem
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ)
    -- MBConv₁ (residual, c → c)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ)
    -- MBConv₂ (no skip, c → cout)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ)
    -- head
    (Wh : Mat cout nClasses) (bh : Vec nClasses) :
    Vec (ic * h * w) → Vec nClasses :=
  dense Wh bh ∘
  globalAvgPoolFlat cout h w ∘
  mbconvBody (h := h) (w := w)
    We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ ∘
  residual (mbconvBody (h := h) (w := w)
    We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁) ∘
  (swish (c * h * w) ∘ bnForward (c * h * w) εs γs βs ∘ flatConv Ws bs)

/-- **End-to-end EfficientNet VJP (global).** Every block is smooth
    everywhere (swish + sigmoid SE gate + convs + BN, no ReLU/maxpool), so
    the only hypotheses are the `0 < ε` batch-norm conditions and the VJP
    holds at *every* input — putting EfficientNet alongside
    `vit_full_has_vjp` and `convnext_has_vjp` as an unconditional
    whole-network VJP. Chained through the global `vjp_comp`. -/
noncomputable def efficientnet_has_vjp
    {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
      kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ) (hεe₁ : 0 < εe₁)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ) (hεd₁ : 0 < εd₁)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ) (hεp₁ : 0 < εp₁)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ) (hεe₂ : 0 < εe₂)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ) (hεd₂ : 0 < εd₂)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ) (hεp₂ : 0 < εp₂)
    (Wh : Mat cout nClasses) (bh : Vec nClasses) :
    HasVJP (efficientnetForward (h := h) (w := w) Ws bs εs γs βs
        We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁
        We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂
        Wh bh) := by
  unfold efficientnetForward
  set STEM := swish (c * h * w) ∘ bnForward (c * h * w) εs γs βs ∘ flatConv Ws bs with hSTEM
  have stem_diff : Differentiable ℝ STEM := convBnSwish_differentiable Ws bs εs γs βs hεs
  have stem_vjp : HasVJP STEM := convBnSwish_has_vjp Ws bs εs γs βs hεs
  set MB1 := residual (mbconvBody (h := h) (w := w)
    We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁) with hMB1
  have mb1_diff : Differentiable ℝ MB1 :=
    mbconvResidual_differentiable We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁
      Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
  have mb1_vjp : HasVJP MB1 :=
    mbconvResidual_has_vjp We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁
      Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
  set MB2 := mbconvBody (h := h) (w := w)
    We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ with hMB2
  have mb2_diff : Differentiable ℝ MB2 :=
    mbconvBody_differentiable We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂
      Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
  have mb2_vjp : HasVJP MB2 :=
    mbconvBody_has_vjp We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂
      Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
  have s1_vjp : HasVJP (MB1 ∘ STEM) := vjp_comp STEM MB1 stem_diff mb1_diff stem_vjp mb1_vjp
  have s1_diff : Differentiable ℝ (MB1 ∘ STEM) := mb1_diff.comp stem_diff
  have s2_vjp : HasVJP (MB2 ∘ (MB1 ∘ STEM)) := vjp_comp (MB1 ∘ STEM) MB2 s1_diff mb2_diff s1_vjp mb2_vjp
  have s2_diff : Differentiable ℝ (MB2 ∘ (MB1 ∘ STEM)) := mb2_diff.comp s1_diff
  set P2 := MB2 ∘ (MB1 ∘ STEM) with hP2
  have gap_diff : Differentiable ℝ (globalAvgPoolFlat cout h w) := globalAvgPoolFlat_differentiable cout h w
  have gap_vjp : HasVJP (globalAvgPoolFlat cout h w) := globalAvgPoolFlat_has_vjp cout h w
  have s3_vjp : HasVJP (globalAvgPoolFlat cout h w ∘ P2) :=
    vjp_comp P2 (globalAvgPoolFlat cout h w) s2_diff gap_diff s2_vjp gap_vjp
  have s3_diff : Differentiable ℝ (globalAvgPoolFlat cout h w ∘ P2) := gap_diff.comp s2_diff
  exact vjp_comp (globalAvgPoolFlat cout h w ∘ P2) (dense Wh bh) s3_diff
    (dense_differentiable Wh bh) s3_vjp (dense_has_vjp Wh bh)

/-- **End-to-end EfficientNet VJP at a point** — the global witness
    restricted to a point. Kept for downstream `_at` consumers and the
    comparator. -/
noncomputable def efficientnet_has_vjp_at
    {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
      kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ) (hεe₁ : 0 < εe₁)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ) (hεd₁ : 0 < εd₁)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ) (hεp₁ : 0 < εp₁)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ) (hεe₂ : 0 < εe₂)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ) (hεd₂ : 0 < εd₂)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ) (hεp₂ : 0 < εp₂)
    (Wh : Mat cout nClasses) (bh : Vec nClasses)
    (x : Vec (ic * h * w)) :
    HasVJPAt (efficientnetForward Ws bs εs γs βs
        We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁
        We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂
        Wh bh) x :=
  (efficientnet_has_vjp (h := h) (w := w) Ws bs εs γs βs hεs
      We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
      We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
      Wh bh).toHasVJPAt x

/-- **Public correctness theorem for `efficientnet_has_vjp` (global)** — the
    full EfficientNet's backward equals the `pdiv`-contracted Jacobian
    (Jacobian-transpose on the cotangent), at *every* input `x`. The
    unconditional EfficientNet analogue of `vit_full_has_vjp_correct`. -/
theorem efficientnet_has_vjp_correct
    {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
      kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ) (hεe₁ : 0 < εe₁)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ) (hεd₁ : 0 < εd₁)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ) (hεp₁ : 0 < εp₁)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ) (hεe₂ : 0 < εe₂)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ) (hεd₂ : 0 < εd₂)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ) (hεp₂ : 0 < εp₂)
    (Wh : Mat cout nClasses) (bh : Vec nClasses)
    (x : Vec (ic * h * w)) (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (efficientnet_has_vjp (h := h) (w := w) Ws bs εs γs βs hεs
        We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
        We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
        Wh bh).backward x dy i =
      ∑ j : Fin nClasses,
        pdiv (efficientnetForward Ws bs εs γs βs
                We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁
                We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂
                Wh bh)
             x i j * dy j :=
  (efficientnet_has_vjp (h := h) (w := w) Ws bs εs γs βs hεs
      We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
      We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
      Wh bh).correct x dy i

/-- **Public correctness theorem for `efficientnet_has_vjp_at`** — exposes
    the witness's `.correct` field: the full EfficientNet's backward equals
    the `pdiv`-contracted Jacobian (Jacobian-transpose on the cotangent).
    EfficientNet analogue of `cnn_has_vjp_at_correct`. -/
theorem efficientnet_has_vjp_at_correct
    {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
      kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ) (hεe₁ : 0 < εe₁)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ) (hεd₁ : 0 < εd₁)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ) (hεp₁ : 0 < εp₁)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ) (hεe₂ : 0 < εe₂)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ) (hεd₂ : 0 < εd₂)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ) (hεp₂ : 0 < εp₂)
    (Wh : Mat cout nClasses) (bh : Vec nClasses)
    (x : Vec (ic * h * w)) (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (efficientnet_has_vjp_at Ws bs εs γs βs hεs
        We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
        We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
        Wh bh x).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (efficientnetForward Ws bs εs γs βs
                We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁
                We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂
                Wh bh)
             x i j * dy j :=
  (efficientnet_has_vjp_at Ws bs εs γs βs hεs
      We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
      We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
      Wh bh x).correct dy i


end Proofs
