import LeanMlir.Proofs.Architectures.CNN

/-! # Strided convolution (stride-2 SAME) — Chapter 5 Milestone B, the hard new op

Real ResNet-34 downsamples with **stride-2 convolutions**, the one genuinely-new
operator the Chapter-5 handoff (`planning/verified_r34.md` §3.6) flags as gating
the jump from the ch6-A ResNet-*style* net to a true 34-layer ResNet.

**The key identity that makes this tractable.** A stride-2 SAME convolution is
exactly a stride-1 SAME convolution followed by spatial decimation (keep every
other position):

  `conv_stride2 W b X = decimate2 (conv2d W b X)`,    `X : Tensor3 ic (2h) (2w)`

because both read `x_pad[c, 2·hi+kh−pH, 2·wi+kw−pW]` — the stride-1 conv computes
that at *every* output position, and decimation throws away the odd ones. So we do
**not** re-derive the ~800-line conv input-VJP / weight-grad with stride arithmetic;
we reuse `conv2d_has_vjp3` and `conv2d_weight_grad_has_vjp` verbatim and only add a
small linear **decimation** map `decimateFlat` (a `reindex`, hence a CLM) with its
VJP (the backward is the "zero-upsampling" / `lhs_dilation` scatter). The strided
conv's input- and weight-VJPs then fall out of `vjp_comp`.

Everything closes under `[propext, Classical.choice, Quot.sound]`.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Spatial decimation `decimate2` (keep every other position)
-- ════════════════════════════════════════════════════════════════

/-- The decimation index map: a small output flat index `k ↔ (co, ho, wo)` maps to
    the **even** input position `(co, 2·ho, 2·wo)` in the `(2h)×(2w)` grid. A pure
    reindex `Fin (oc·h·w) → Fin (oc·2h·2w)`; `decimateFlat` reads through it. -/
noncomputable def decimateIdx (oc h w : Nat) (k : Fin (oc * h * w)) :
    Fin (oc * (2 * h) * (2 * w)) :=
  let p := finProdFinEquiv.symm k         -- (Fin (oc*h), Fin w)
  let q := finProdFinEquiv.symm p.1       -- (Fin oc, Fin h)
  finProdFinEquiv
    (finProdFinEquiv (q.1, (⟨2 * q.2.val, by have := q.2.isLt; omega⟩ : Fin (2 * h))),
     (⟨2 * p.2.val, by have := p.2.isLt; omega⟩ : Fin (2 * w)))

/-- **Flat spatial decimation** `Vec (oc·2h·2w) → Vec (oc·h·w)`: keep the even
    spatial positions. A coordinate reindex `fun y k => y (decimateIdx k)` — i.e.
    `reindexCLM decimateIdx` — so it is continuous-linear (hence differentiable),
    and `decimate2 (conv2d …) = conv_stride2 …`. -/
noncomputable def decimateFlat (oc h w : Nat) :
    Vec (oc * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  fun y k => y (decimateIdx oc h w k)

theorem decimateFlat_differentiable (oc h w : Nat) :
    Differentiable ℝ (decimateFlat oc h w) :=
  (reindexCLM (decimateIdx oc h w)).differentiable

/-- **Decimation VJP.** `decimateFlat` is a reindex, so its Jacobian is the sparse
    `δ(idx = decimateIdx j)` (`pdiv_reindex`); the backward scatters `dy` back to
    the even positions (zero elsewhere) — the "zero-upsampling" that StableHLO
    renders as `lhs_dilation = [2,2]`. Stated in the universal `∑ pdiv · dy` form. -/
noncomputable def decimateFlat_has_vjp (oc h w : Nat) :
    HasVJP (decimateFlat oc h w) where
  backward := fun _v dy => fun idx =>
    ∑ k : Fin (oc * h * w), (if idx = decimateIdx oc h w k then (1 : ℝ) else 0) * dy k
  correct := by
    intro v dy idx
    apply Finset.sum_congr rfl
    intro j _
    rw [show decimateFlat oc h w = (fun y : Vec (oc * (2*h) * (2*w)) =>
            fun k : Fin (oc * h * w) => y (decimateIdx oc h w k)) from rfl,
        pdiv_reindex]

-- ════════════════════════════════════════════════════════════════
-- § Stride-2 SAME convolution = decimate ∘ (stride-1 SAME conv)
-- ════════════════════════════════════════════════════════════════

/-- **Stride-2 SAME convolution**, flattened: `Vec (ic·2h·2w) → Vec (oc·h·w)`.
    Defined as `decimateFlat ∘ flatConv` (the stride-1 SAME conv on the `2h×2w`
    grid, then keep even positions) — provably the genuine stride-2 conv. -/
noncomputable def flatConvStride2 {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  decimateFlat oc h w ∘ (flatConv (h := 2 * h) (w := 2 * w) W b)

theorem flatConvStride2_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Differentiable ℝ (flatConvStride2 W b
      : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) := by
  unfold flatConvStride2
  -- pin the stride-1 conv's spatial dims to 2h×2w structurally (else the
  -- nonlinear `oc*?*? = oc*(2h)*(2w)` won't unify)
  have hf : Differentiable ℝ (flatConv (h := 2 * h) (w := 2 * w) W b) :=
    flatConv_differentiable W b
  have hg : Differentiable ℝ (decimateFlat oc h w) := decimateFlat_differentiable oc h w
  exact hg.comp hf

/-- **Stride-2 conv input-VJP** — the centerpiece. By the chain rule
    (`vjp_comp`) on `decimateFlat ∘ flatConv`, reusing the proven stride-1 conv
    input-VJP (`conv2d_has_vjp3` via the flatten bridge) and the decimation VJP.
    The backward is `flatConv.back (decimate.back dy)` — i.e. zero-upsample the
    cotangent, then run the reversed-kernel conv (StableHLO: `lhs_dilation=[2,2]`
    on the transpose-reverse convolution). -/
noncomputable def flatConvStride2_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    HasVJP (flatConvStride2 W b
      : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
  let hf_diff : Differentiable ℝ (flatConv (h := 2 * h) (w := 2 * w) W b) :=
    flatConv_differentiable W b
  let hf_vjp : HasVJP (flatConv (h := 2 * h) (w := 2 * w) W b) :=
    hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)
  show HasVJP (decimateFlat oc h w ∘ (flatConv (h := 2 * h) (w := 2 * w) W b)) from
  vjp_comp _ _ hf_diff (decimateFlat_differentiable oc h w) hf_vjp (decimateFlat_has_vjp oc h w)

/-- **Stride-2 conv input-VJP correctness** (the ℝ-carrying audit headline): the
    backward equals the `pdiv`-contracted Jacobian of `flatConvStride2`. -/
theorem flatConvStride2_has_vjp_correct {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Vec (ic * (2 * h) * (2 * w))) (dy : Vec (oc * h * w)) (i : Fin (ic * (2 * h) * (2 * w))) :
    (flatConvStride2_has_vjp W b).backward x dy i
      = ∑ j : Fin (oc * h * w), pdiv (flatConvStride2 W b) x i j * dy j :=
  (flatConvStride2_has_vjp W b).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § Stride-2 conv weight-VJP (reuses the stride-1 weight-grad)
-- ════════════════════════════════════════════════════════════════

/-- **Conv2d (as a function of its flattened kernel) is differentiable** — it is
    affine in the weights (`b o + ∑ v(idx)·pad-eval x`, the pad-eval being a
    weight-independent constant). Needed as the `vjp_comp` hypothesis for the
    strided weight-grad. -/
theorem conv2d_weight_differentiable {ic oc h w kH kW : Nat} (b : Vec oc) (x : Tensor3 ic h w) :
    Differentiable ℝ (fun v : Vec (oc * ic * kH * kW) =>
      Tensor3.flatten (conv2d (Kernel4.unflatten v) b x)) := by
  unfold conv2d Tensor3.flatten Kernel4.unflatten
  fun_prop

/-- **Stride-2 conv weight-VJP.** The same composition `decimate ∘ conv` viewed
    as a function of the *kernel* (input `x` fixed): the weight-grad is
    `conv_weight_grad` run on the zero-upsampled cotangent. By `vjp_comp`,
    reusing the proven stride-1 `conv2d_weight_grad_has_vjp` + `decimateFlat_has_vjp`. -/
noncomputable def flatConvStride2_weight_grad_has_vjp {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) :
    HasVJP (fun v : Vec (oc * ic * kH * kW) =>
      flatConvStride2 (Kernel4.unflatten v) b x) :=
  let f : Vec (oc * ic * kH * kW) → Vec (oc * (2 * h) * (2 * w)) :=
    fun v => Tensor3.flatten (conv2d (Kernel4.unflatten v) b (Tensor3.unflatten x))
  let hf_diff : Differentiable ℝ f :=
    conv2d_weight_differentiable (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten x)
  let hf_vjp : HasVJP f :=
    conv2d_weight_grad_has_vjp (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten x)
  show HasVJP (decimateFlat oc h w ∘ f) from
  vjp_comp f (decimateFlat oc h w) hf_diff (decimateFlat_differentiable oc h w)
    hf_vjp (decimateFlat_has_vjp oc h w)

/-- **Stride-2 conv weight-VJP correctness** (ℝ-headline): backward = the
    `pdiv`-Jacobian of the strided conv in its kernel. -/
theorem flatConvStride2_weight_grad_has_vjp_correct {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w)))
    (v : Vec (oc * ic * kH * kW)) (dy : Vec (oc * h * w)) (i : Fin (oc * ic * kH * kW)) :
    (flatConvStride2_weight_grad_has_vjp b x).backward v dy i
      = ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) => flatConvStride2 (Kernel4.unflatten v') b x) v i j * dy j :=
  (flatConvStride2_weight_grad_has_vjp b x).correct v dy i

-- ════════════════════════════════════════════════════════════════
-- § Stride-2 conv bias-VJP (reuses the stride-1 bias-grad)
-- ════════════════════════════════════════════════════════════════
-- (Relocated here from `MobileNetV2Close` so the `convStridedBiasSgd` op's `den` in
--  `StableHLO` can reference it — same upstream-move pattern as the per-channel BN grads.)

/-- **`conv2d` (as a function of its bias) is differentiable** — affine in `b` (bias broadcast
    plus a `b`-independent `W,x` term). The `vjp_comp` hypothesis for the strided-conv bias-grad. -/
theorem conv2d_bias_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) :
    Differentiable ℝ (fun b : Vec oc => Tensor3.flatten (conv2d W b x)) := by
  unfold conv2d Tensor3.flatten
  fun_prop

/-- **Stride-2 conv bias-VJP.** `fun b => flatConvStride2 W b x = decimate ∘ (conv2d-in-b)`; by
    `vjp_comp` of the proven stride-1 `conv2d_bias_grad_has_vjp` with `decimateFlat_has_vjp`. The
    bias peer of `flatConvStride2_weight_grad_has_vjp`. -/
noncomputable def flatConvStride2_bias_grad_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Vec (ic * (2 * h) * (2 * w))) :
    HasVJP (fun b : Vec oc =>
      flatConvStride2 W b x : Vec oc → Vec (oc * h * w)) :=
  let g : Vec oc → Vec (oc * (2 * h) * (2 * w)) :=
    fun b => Tensor3.flatten (conv2d W b (Tensor3.unflatten x))
  let hg_diff : Differentiable ℝ g :=
    conv2d_bias_differentiable (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten x)
  let hg_vjp : HasVJP g :=
    conv2d_bias_grad_has_vjp (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten x)
  show HasVJP (decimateFlat oc h w ∘ g) from
  vjp_comp g (decimateFlat oc h w) hg_diff (decimateFlat_differentiable oc h w)
    hg_vjp (decimateFlat_has_vjp oc h w)

-- ════════════════════════════════════════════════════════════════
-- § Stride-4 patchify convolution = decimate ∘ decimateOdd ∘ (stride-1 SAME conv)
--   (the ConvNeXt 4×4/s4 patchify stem, ch9 scaling pass)
-- ════════════════════════════════════════════════════════════════

/-- The odd decimation index map: like `decimateIdx` but keeping the **odd**
    positions `(co, 2·ho+1, 2·wo+1)`. Composed under an even decimation it reads
    `4·ho+1` — which for the SAME conv at `pad (k-1)/2 = 1` (k = 4) makes the
    window exactly the **left-aligned** `x[4i .. 4i+3]`: the real (paper/render)
    pad-0 stride-4 patchify, never touching the boundary padding. -/
noncomputable def decimateOddIdx (oc h w : Nat) (k : Fin (oc * h * w)) :
    Fin (oc * (2 * h) * (2 * w)) :=
  let p := finProdFinEquiv.symm k         -- (Fin (oc*h), Fin w)
  let q := finProdFinEquiv.symm p.1       -- (Fin oc, Fin h)
  finProdFinEquiv
    (finProdFinEquiv (q.1, (⟨2 * q.2.val + 1, by have := q.2.isLt; omega⟩ : Fin (2 * h))),
     (⟨2 * p.2.val + 1, by have := p.2.isLt; omega⟩ : Fin (2 * w)))

/-- **Flat odd spatial decimation** `Vec (oc·2h·2w) → Vec (oc·h·w)`: keep the odd
    spatial positions. A coordinate reindex, exactly as `decimateFlat`. -/
noncomputable def decimateOddFlat (oc h w : Nat) :
    Vec (oc * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  fun y k => y (decimateOddIdx oc h w k)

theorem decimateOddFlat_differentiable (oc h w : Nat) :
    Differentiable ℝ (decimateOddFlat oc h w) :=
  (reindexCLM (decimateOddIdx oc h w)).differentiable

/-- **Odd-decimation VJP** — the same sparse-δ reindex Jacobian as
    `decimateFlat_has_vjp`, at the odd positions. -/
noncomputable def decimateOddFlat_has_vjp (oc h w : Nat) :
    HasVJP (decimateOddFlat oc h w) where
  backward := fun _v dy => fun idx =>
    ∑ k : Fin (oc * h * w), (if idx = decimateOddIdx oc h w k then (1 : ℝ) else 0) * dy k
  correct := by
    intro v dy idx
    apply Finset.sum_congr rfl
    intro j _
    rw [show decimateOddFlat oc h w = (fun y : Vec (oc * (2*h) * (2*w)) =>
            fun k : Fin (oc * h * w) => y (decimateOddIdx oc h w k)) from rfl,
        pdiv_reindex]

/-- **Stride-4 patchify convolution**, flattened: `Vec (ic·4h·4w) → Vec (oc·h·w)`.
    `decimateFlat ∘ decimateOddFlat ∘ (stride-1 SAME conv)` — reads the SAME conv
    (pad `(k-1)/2`) at positions `4i+1`, which for the 4×4 stem is the
    **left-aligned window** `x[4i .. 4i+3]`: the paper's pad-0 `Conv2d(k=4, s=4)`
    and the committed render's patchify, in-bounds at every tap. -/
noncomputable def flatConvStride4 {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * (2 * (2 * h)) * (2 * (2 * w))) → Vec (oc * h * w) :=
  decimateFlat oc h w ∘ decimateOddFlat oc (2 * h) (2 * w) ∘
    flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b

theorem flatConvStride4_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Differentiable ℝ (flatConvStride4 W b
      : Vec (ic * (2 * (2 * h)) * (2 * (2 * w))) → Vec (oc * h * w)) := by
  unfold flatConvStride4
  have hf : Differentiable ℝ (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b) :=
    flatConv_differentiable W b
  exact (decimateFlat_differentiable oc h w).comp
    ((decimateOddFlat_differentiable oc (2 * h) (2 * w)).comp hf)

/-- **Stride-4 conv input-VJP** — two `vjp_comp` steps over the proven stride-1
    conv input-VJP and the two decimation VJPs (backward = zero-upsample twice,
    then the reversed-kernel conv). -/
noncomputable def flatConvStride4_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    HasVJP (flatConvStride4 W b
      : Vec (ic * (2 * (2 * h)) * (2 * (2 * w))) → Vec (oc * h * w)) := by
  unfold flatConvStride4
  have hf_diff : Differentiable ℝ (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b) :=
    flatConv_differentiable W b
  have hf_vjp : HasVJP (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b) :=
    hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)
  have s1_vjp : HasVJP (decimateOddFlat oc (2 * h) (2 * w) ∘
      flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b) :=
    vjp_comp _ _ hf_diff (decimateOddFlat_differentiable oc (2 * h) (2 * w))
      hf_vjp (decimateOddFlat_has_vjp oc (2 * h) (2 * w))
  have s1_diff : Differentiable ℝ (decimateOddFlat oc (2 * h) (2 * w) ∘
      flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b) :=
    (decimateOddFlat_differentiable oc (2 * h) (2 * w)).comp hf_diff
  exact vjp_comp _ _ s1_diff (decimateFlat_differentiable oc h w)
    s1_vjp (decimateFlat_has_vjp oc h w)

/-- **Stride-4 conv weight-VJP.** The kernel-side peer of `flatConvStride4_has_vjp`, and the
    stride-4 analogue of `flatConvStride2_weight_grad_has_vjp`: the same
    `decimateFlat ∘ decimateOddFlat ∘ conv` composition viewed as a function of the *kernel*
    (input `x` fixed), so the weight-grad is `conv2d_weight_grad` run on the twice-zero-upsampled
    cotangent. Two `vjp_comp` steps over the proven stride-1 weight-VJP and the two decimation
    VJPs — no new mathematics, only the composition the stride-2 sibling already does once.

    This is the cert that ConvNeXt's 4×4/s4 patchify stem (`psW`) was missing: its forward
    (`flatConvStride4`) and input-VJP (`flatConvStride4_has_vjp`) were already proven, so the stem's
    weight gradient was the last hand-written emitter in that render. -/
noncomputable def flatConvStride4_weight_grad_has_vjp {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w)))) :
    HasVJP (fun v : Vec (oc * ic * kH * kW) =>
      flatConvStride4 (Kernel4.unflatten v) b x) :=
  let f : Vec (oc * ic * kH * kW) → Vec (oc * (2 * (2 * h)) * (2 * (2 * w))) :=
    fun v => Tensor3.flatten (conv2d (Kernel4.unflatten v) b (Tensor3.unflatten x))
  let hf_diff : Differentiable ℝ f :=
    conv2d_weight_differentiable (h := 2 * (2 * h)) (w := 2 * (2 * w)) b (Tensor3.unflatten x)
  let hf_vjp : HasVJP f :=
    conv2d_weight_grad_has_vjp (h := 2 * (2 * h)) (w := 2 * (2 * w)) b (Tensor3.unflatten x)
  let s1_diff : Differentiable ℝ (decimateOddFlat oc (2 * h) (2 * w) ∘ f) :=
    (decimateOddFlat_differentiable oc (2 * h) (2 * w)).comp hf_diff
  let s1_vjp : HasVJP (decimateOddFlat oc (2 * h) (2 * w) ∘ f) :=
    vjp_comp f (decimateOddFlat oc (2 * h) (2 * w)) hf_diff
      (decimateOddFlat_differentiable oc (2 * h) (2 * w)) hf_vjp
      (decimateOddFlat_has_vjp oc (2 * h) (2 * w))
  show HasVJP (decimateFlat oc h w ∘ (decimateOddFlat oc (2 * h) (2 * w) ∘ f)) from
  vjp_comp _ (decimateFlat oc h w) s1_diff (decimateFlat_differentiable oc h w)
    s1_vjp (decimateFlat_has_vjp oc h w)

/-- **Stride-4 conv weight-VJP correctness** (ℝ-headline): backward = the `pdiv`-Jacobian of the
    stride-4 conv in its kernel. The peer of `flatConvStride2_weight_grad_has_vjp_correct`. -/
theorem flatConvStride4_weight_grad_has_vjp_correct {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w))))
    (v : Vec (oc * ic * kH * kW)) (dy : Vec (oc * h * w)) (i : Fin (oc * ic * kH * kW)) :
    (flatConvStride4_weight_grad_has_vjp b x).backward v dy i
      = ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
            flatConvStride4 (Kernel4.unflatten v') b x) v i j * dy j :=
  (flatConvStride4_weight_grad_has_vjp b x).correct v dy i

-- ════════════════════════════════════════════════════════════════
-- § Stride-2 conv at XLA `SAME` = decimateODD ∘ (stride-1 SAME conv)
--   (`planning/mnv4_verified.md` §3b/§3d — the TF-origin padding convention)
-- ════════════════════════════════════════════════════════════════

/-! **Why a second stride-2 convolution exists, and why it is not a fix to the first.**

The repo carries two genuinely different stride-2 conventions, and both are correct references
for the nets that use them:

* **`flatConvStride2`** (above) pads symmetrically, `(k-1)/2` on each side. This is
  He et al. / torchvision — `nn.Conv2d(padding=k//2)` — and it is what ResNet-34, ResNet-50 and
  ConvNeXt's references do (`jax/Jax/Codegen.lean:462`, symmetric ON PURPOSE since 2026-08-04).
* **`flatConvStride2Xla`** (here) pads the way XLA `'SAME'` does: at an **even** input the total
  padding is `k-2`, split **asymmetrically** as `((k-2)/2, k/2)` — `(0,1)` at `k=3`, `(1,2)` at
  `k=5`, `(2,3)` at `k=7`. This is what the TF-origin ports do — MobileNetV2, MobileNetV4,
  EfficientNet — where `padding='SAME'` **is** the reference and must not be "fixed".

⭐ **The identity that makes this nearly free.** A stride-2 XLA-`SAME` conv is the *same*
symmetric stride-1 conv the even-decimation op already uses, decimated at the **odd** offsets:

  `convXlaSame_s2 W b X = decimateOddFlat (flatConv W b X)`,   `X : Tensor3 ic (2h) (2w)`

because output `ho` then reads `x[2·ho + 1 + kh − (k−1)/2] = x[2·ho + kh − ((k−2)/2)]`, and
`(k−2)/2` is exactly XLA's `pad_low` at an even input. So the whole asymmetry is **a phase shift
in the decimation**, not new padding arithmetic: `flatConv` is reused verbatim, and so is every
one of its VJPs. `decimateOddFlat` and `decimateOddFlat_has_vjp` already exist above (they were
added for ConvNeXt's 4×4/s4 patchify stem), so this section adds **no new proof obligation** —
only compositions of results already closed under the three standard axioms.

⚠ **This holds at EVEN inputs only, which is the only case any net in this repo hits** (224, 112,
56, 28, 14 — every strided site in mnv2/mnv4/enet). At an *odd* input XLA `SAME` pads
`((k-1)/2, (k-1)/2)` — symmetric — so `flatConvStride2` is already the right op there and this one
would be wrong. The type enforces it: the input index is `ic*(2*h)*(2*w)`, structurally even.
Verified against `jax.lax.conv_general_dilated(…, 'SAME')` over
`H ∈ {224,112,56,28,14,32,16,9,7,15,33} × k ∈ {3,5,7}` — 33 configs, all agreeing with this rule
(`planning/mnv4_verified.md` §3e). -/

/-- **Stride-2 XLA-`SAME` convolution**, flattened: `Vec (ic·2h·2w) → Vec (oc·h·w)`.
    `decimateOddFlat ∘ flatConv` — the stride-1 symmetric-SAME conv on the `2h×2w` grid, then keep
    the **odd** positions. The asymmetric-pad peer of `flatConvStride2`. -/
noncomputable def flatConvStride2Xla {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  decimateOddFlat oc h w ∘ (flatConv (h := 2 * h) (w := 2 * w) W b)

theorem flatConvStride2Xla_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Differentiable ℝ (flatConvStride2Xla W b
      : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) := by
  unfold flatConvStride2Xla
  have hf : Differentiable ℝ (flatConv (h := 2 * h) (w := 2 * w) W b) :=
    flatConv_differentiable W b
  exact (decimateOddFlat_differentiable oc h w).comp hf

/-- **Stride-2 XLA-`SAME` input-VJP.** `vjp_comp` on `decimateOddFlat ∘ flatConv`, reusing the
    proven stride-1 conv input-VJP and the odd-decimation VJP. The backward zero-upsamples the
    cotangent **onto the odd positions** and then runs the reversed-kernel conv — i.e. StableHLO's
    `lhs_dilation = [2,2]` with the transposed padding shifted by one, which is exactly the
    asymmetry the forward introduced. ⚠ A symmetric backward against this forward is a silent
    wrong-gradient (`planning/mnv4_verified.md` §3b), and it is this composition that rules it out:
    the offset lives in one place and both directions read it. -/
noncomputable def flatConvStride2Xla_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    HasVJP (flatConvStride2Xla W b
      : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
  let hf_diff : Differentiable ℝ (flatConv (h := 2 * h) (w := 2 * w) W b) :=
    flatConv_differentiable W b
  let hf_vjp : HasVJP (flatConv (h := 2 * h) (w := 2 * w) W b) :=
    hasVJP3_to_hasVJP (conv2d_has_vjp3 W b)
  show HasVJP (decimateOddFlat oc h w ∘ (flatConv (h := 2 * h) (w := 2 * w) W b)) from
  vjp_comp _ _ hf_diff (decimateOddFlat_differentiable oc h w) hf_vjp
    (decimateOddFlat_has_vjp oc h w)

/-- **Stride-2 XLA-`SAME` input-VJP correctness** (the ℝ-carrying audit headline): the backward
    equals the `pdiv`-contracted Jacobian. Peer of `flatConvStride2_has_vjp_correct`. -/
theorem flatConvStride2Xla_has_vjp_correct {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Vec (ic * (2 * h) * (2 * w))) (dy : Vec (oc * h * w)) (i : Fin (ic * (2 * h) * (2 * w))) :
    (flatConvStride2Xla_has_vjp W b).backward x dy i
      = ∑ j : Fin (oc * h * w), pdiv (flatConvStride2Xla W b) x i j * dy j :=
  (flatConvStride2Xla_has_vjp W b).correct x dy i

/-- **Stride-2 XLA-`SAME` weight-VJP.** The same composition viewed as a function of the *kernel*
    (input `x` fixed): `conv2d_weight_grad` run on the odd-zero-upsampled cotangent. -/
noncomputable def flatConvStride2Xla_weight_grad_has_vjp {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) :
    HasVJP (fun v : Vec (oc * ic * kH * kW) =>
      flatConvStride2Xla (Kernel4.unflatten v) b x) :=
  let f : Vec (oc * ic * kH * kW) → Vec (oc * (2 * h) * (2 * w)) :=
    fun v => Tensor3.flatten (conv2d (Kernel4.unflatten v) b (Tensor3.unflatten x))
  let hf_diff : Differentiable ℝ f :=
    conv2d_weight_differentiable (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten x)
  let hf_vjp : HasVJP f :=
    conv2d_weight_grad_has_vjp (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten x)
  show HasVJP (decimateOddFlat oc h w ∘ f) from
  vjp_comp f (decimateOddFlat oc h w) hf_diff (decimateOddFlat_differentiable oc h w)
    hf_vjp (decimateOddFlat_has_vjp oc h w)

/-- **Stride-2 XLA-`SAME` weight-VJP correctness** (ℝ-headline). -/
theorem flatConvStride2Xla_weight_grad_has_vjp_correct {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w)))
    (v : Vec (oc * ic * kH * kW)) (dy : Vec (oc * h * w)) (i : Fin (oc * ic * kH * kW)) :
    (flatConvStride2Xla_weight_grad_has_vjp b x).backward v dy i
      = ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
            flatConvStride2Xla (Kernel4.unflatten v') b x) v i j * dy j :=
  (flatConvStride2Xla_weight_grad_has_vjp b x).correct v dy i

/-- **Stride-2 XLA-`SAME` bias-VJP.** `fun b => flatConvStride2Xla W b x` = the odd decimation of
    the stride-1 conv-in-`b`; by `vjp_comp` of `conv2d_bias_grad_has_vjp` with the odd-decimation
    VJP. The bias peer of `flatConvStride2_bias_grad_has_vjp`. -/
noncomputable def flatConvStride2Xla_bias_grad_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Vec (ic * (2 * h) * (2 * w))) :
    HasVJP (fun b : Vec oc =>
      flatConvStride2Xla W b x : Vec oc → Vec (oc * h * w)) :=
  let g : Vec oc → Vec (oc * (2 * h) * (2 * w)) :=
    fun b => Tensor3.flatten (conv2d W b (Tensor3.unflatten x))
  let hg_diff : Differentiable ℝ g :=
    conv2d_bias_differentiable (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten x)
  let hg_vjp : HasVJP g :=
    conv2d_bias_grad_has_vjp (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten x)
  show HasVJP (decimateOddFlat oc h w ∘ g) from
  vjp_comp g (decimateOddFlat oc h w) hg_diff (decimateOddFlat_differentiable oc h w)
    hg_vjp (decimateOddFlat_has_vjp oc h w)

end Proofs
