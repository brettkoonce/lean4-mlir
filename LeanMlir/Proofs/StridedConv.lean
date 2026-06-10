import LeanMlir.Proofs.CNN

/-! # Strided convolution (stride-2 SAME) — Chapter 6 Milestone B, the hard new op

Real ResNet-34 downsamples with **stride-2 convolutions**, the one genuinely-new
operator the Chapter-6 handoff (`planning/verified_r34.md` §3.6) flags as gating
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
-- § Stride-4 SAME convolution = decimate ∘ decimate ∘ (stride-1 SAME conv)
--   (the ConvNeXt 4×4/s4 patchify stem, ch9 scaling pass)
-- ════════════════════════════════════════════════════════════════

/-- **Stride-4 SAME convolution**, flattened: `Vec (ic·4h·4w) → Vec (oc·h·w)`.
    Double decimation of the stride-1 SAME conv — keep every 4th position
    (even-of-even), the ch6 SAME-strided convention extended to stride 4. -/
noncomputable def flatConvStride4 {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * (2 * (2 * h)) * (2 * (2 * w))) → Vec (oc * h * w) :=
  decimateFlat oc h w ∘ decimateFlat oc (2 * h) (2 * w) ∘
    flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b

theorem flatConvStride4_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Differentiable ℝ (flatConvStride4 W b
      : Vec (ic * (2 * (2 * h)) * (2 * (2 * w))) → Vec (oc * h * w)) := by
  unfold flatConvStride4
  have hf : Differentiable ℝ (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b) :=
    flatConv_differentiable W b
  exact (decimateFlat_differentiable oc h w).comp
    ((decimateFlat_differentiable oc (2 * h) (2 * w)).comp hf)

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
  have s1_vjp : HasVJP (decimateFlat oc (2 * h) (2 * w) ∘
      flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b) :=
    vjp_comp _ _ hf_diff (decimateFlat_differentiable oc (2 * h) (2 * w))
      hf_vjp (decimateFlat_has_vjp oc (2 * h) (2 * w))
  have s1_diff : Differentiable ℝ (decimateFlat oc (2 * h) (2 * w) ∘
      flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W b) :=
    (decimateFlat_differentiable oc (2 * h) (2 * w)).comp hf_diff
  exact vjp_comp _ _ s1_diff (decimateFlat_differentiable oc h w)
    s1_vjp (decimateFlat_has_vjp oc h w)

end Proofs
