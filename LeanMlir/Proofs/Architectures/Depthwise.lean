import LeanMlir.Proofs.Foundation.Tensor
import LeanMlir.Proofs.Architectures.CNN
import LeanMlir.Proofs.Architectures.StridedConv

/-!
# Depthwise Convolution

The structural simplification at the heart of MobileNet, EfficientNet,
and every "mobile" CNN since ~2017. Standard conv2d does cross-channel
mixing (every output channel sees every input channel) plus spatial
filtering. Depthwise conv **drops the cross-channel mixing**: each
input channel gets its own 2D filter and produces its own output channel.

Math-wise it's "regular conv with a constraint." Practical-wise it's
~10× cheaper because you avoid the `O(ic · oc)` cross-channel sum.

Architecturally, depthwise is always paired with a 1×1 "pointwise" conv
that does the cross-channel mixing separately. Together they form the
**depthwise-separable convolution** (Xception, MobileNet) — the same
expressive power as a regular conv, factored into two cheaper steps.

## What this file proves

The depthwise conv is **structurally a special case of regular conv**:
- Regular conv kernel: `(oc, ic, kH, kW)` — full mixing.
- Depthwise kernel:    `(c, 1, kH, kW)` — diagonal in the channel pair.

The VJPs (`depthwiseHasVJP3`, `depthwiseWeightGradHasVJP3`,
`depthwiseBiasGradHasVJP`) are proved directly with `pdiv_of_affine`, not
derived from the regular-conv ones; they have the same shape as
`conv2dInputGrad` / `conv2dWeightGrad` from `CNN.lean` with the sum over
input channels removed. The transpose trick and the reversed-kernel trick
still apply — they just operate per-channel.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Types
-- ════════════════════════════════════════════════════════════════

/-- A depthwise kernel: `(c, kH, kW)` — one filter per channel, no
    `in_channels` axis. (Equivalently, a `(c, 1, kH, kW)` kernel where
    the singleton dim has been squeezed out.)

    In the MLIR backend this is represented as a regular `(c, 1, kH, kW)`
    kernel with `feature_group_count = c`, telling StableHLO to apply
    each kernel only to its own input channel. -/
abbrev DepthwiseKernel (c kH kW : Nat) := Fin c → Fin kH → Fin kW → ℝ

-- ════════════════════════════════════════════════════════════════
-- § Forward
-- ════════════════════════════════════════════════════════════════

/-- **Depthwise conv2d forward** (SAME padding, stride 1).

    `y[c, h, w] = (Σ_{kh, kw} x[c, h+kh−p, w+kw−p] · W[c, kh, kw]) + b[c]`

    Compare to regular conv2d (`CNN.lean` `conv2d`):
      regular: `y[o,h,w] = Σ_{c, kh, kw} x[c,...] · W[o,c,kh,kw] + b[o]`
      depthwise: same minus the `Σ_c` (no cross-channel mixing).

    Output has the same number of channels as the input (`c`, not `oc`).

    MLIR (`MlirCodegen.lean` `emitDepthwiseConvBn`):
      uses `feature_group_count = c` to tell StableHLO that each kernel
      applies only within its own channel group. -/
noncomputable def depthwiseConv2d {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) : Tensor3 c h w :=
  fun ch hi wi =>
    b ch + ∑ kh : Fin kH, ∑ kw : Fin kW,
      W ch kh kw *
        (let pH := (kH - 1) / 2
         let pW := (kW - 1) / 2
         let hh := kh.val + hi.val
         let ww := kw.val + wi.val
         if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
           x ch ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
         else 0)

-- ════════════════════════════════════════════════════════════════
-- § Backward — three pieces, each per-channel
-- ════════════════════════════════════════════════════════════════

/-- **Closed-form input gradient for depthwise conv2d** — direct formula,
    written as a sum over output positions `(ho, wo)` with reconstructed
    kernel offsets `kh_nat = hi + pH − ho`, `kw_nat = wi + pW − wo`. The
    body is nonzero only when the reconstructed `(kh_nat, kw_nat)` lies
    in `[0, kH) × [0, kW)`. No `Σ co` like regular conv2d — input channel
    `ci` reads only from kernel-channel `ci` and gradient-channel `ci`,
    because depthwise has no cross-channel mixing.

    Equivalent (under the `(ho, wo) ↔ (kh, kw)` partial bijection) to the
    MLIR-aligned reversed-kernel formula
    `dx[c, h, w] = Σ_{kh, kw} W[c, kH−1−kh, kW−1−kw] · dy[c, h+kh−p, w+kw−p]`. -/
noncomputable def depthwiseConv2dInputGradFormula {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (dy : Tensor3 c h w) : Tensor3 c h w :=
  fun ci hi wi =>
    ∑ ho : Fin h, ∑ wo : Fin w,
      let pH := (kH - 1) / 2
      let pW := (kW - 1) / 2
      let kh_nat := hi.val + pH - ho.val
      let kw_nat := wi.val + pW - wo.val
      if hpad : ho.val ≤ hi.val + pH ∧ kh_nat < kH ∧
                 wo.val ≤ wi.val + pW ∧ kw_nat < kW then
        W ci ⟨kh_nat, hpad.2.1⟩ ⟨kw_nat, hpad.2.2.2⟩ * dy ci ho wo
      else 0

/-- **Depthwise conv input-VJP** — proved from foundation rules.

    The function `v ↦ flatten (depthwiseConv2d W b (unflatten v))` is
    affine in `v`: a constant `b ohw_o(idx_out)` plus a double sum over
    `(kh, kw)` of `W ohw_o kh kw * (if pad-cond then v(reindex) else 0)`.
    Mirrors `conv2dHasVJP3` but with one fewer sum level (no Σ c) and
    the channel for the `v`-read is the same as `ohw_o` (forced by
    structure: input-channel = output-channel in depthwise).

    The closing collapse first folds `Σ co → co=ci` (since for `co ≠ ci`,
    the indicator `idx_in = finProdFinEquiv (..., co, ...)` is false by
    channel-mismatch on the first projection), then proceeds per-(ho, wo)
    with a 2-conjunct `h_indicator` (just `kh+ho = hi+pH` and
    `kw+wo = wi+pW`; no `c = ci` since `c` isn't summed).

    The backward function (accessed as `(depthwiseHasVJP3 W b).backward`,
    or via the `depthwiseConv2dInputGrad` abbrev below) implements
    `depthwiseConv2dInputGradFormula W dy ci hi wi`. Equivalent to the
    MLIR-aligned reversed-kernel formula
    `dx[c, h, w] = Σ_{kh, kw} W[c, kH−1−kh, kW−1−kw] · dy[c, h+kh−p, w+kw−p]`. -/
noncomputable def depthwiseHasVJP3 {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    HasVJP3 (depthwiseConv2d W b : Tensor3 c h w → Tensor3 c h w) where
  backward := fun _x dy => depthwiseConv2dInputGradFormula W dy
  correct := by
    intro x dy ci hi wi
    set idx_in : Fin (c * h * w) :=
      finProdFinEquiv (finProdFinEquiv (ci, hi), wi) with hidx_in
    -- Step 1: per-(idx_in, idx_out) pdiv formula. UN-collapsed in (kh, kw)
    -- to avoid a partial bijection between Fin h and Fin kH; the closing
    -- collapse reindexes naturally.
    have h_pdiv : ∀ idx_out : Fin (c * h * w),
        pdiv (fun v' : Vec (c * h * w) =>
                Tensor3.flatten (depthwiseConv2d W b (Tensor3.unflatten v')))
              (Tensor3.flatten x) idx_in idx_out =
        ∑ kh : Fin kH, ∑ kw : Fin kW,
          W ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1) kh kw *
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := kh.val +
               (finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).2.val
             let ww := kw.val + (finProdFinEquiv.symm idx_out).2.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               (if idx_in = finProdFinEquiv (finProdFinEquiv
                   ((finProdFinEquiv.symm (finProdFinEquiv.symm idx_out).1).1,
                    ⟨hh - pH, hpad.2.1⟩), ⟨ww - pW, hpad.2.2.2⟩) then
                 (1 : ℝ) else 0)
             else 0) := by
      intro idx_out
      -- Affine in the input: the bias-free depthwise conv plus the broadcast bias.
      have hsplit : (fun v' : Vec (c * h * w) =>
            Tensor3.flatten (depthwiseConv2d W b (Tensor3.unflatten v'))) =
          fun v => Tensor3.flatten (depthwiseConv2d W 0 (Tensor3.unflatten v)) +
            (fun k => b (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1) := by
        funext v k
        simp only [Tensor3.flatten, depthwiseConv2d, Pi.add_apply, Pi.zero_apply, zero_add]
        ring
      rw [hsplit, pdiv_of_affine]
      · simp only [Tensor3.flatten, depthwiseConv2d, Tensor3.unflatten, Pi.zero_apply, zero_add,
          basisVec_apply, @eq_comm _ idx_in]
      · intro u v; funext k
        simp only [Tensor3.flatten, depthwiseConv2d, Tensor3.unflatten, Pi.add_apply,
          Pi.zero_apply, zero_add, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun kh _ => Finset.sum_congr rfl fun kw _ => ?_
        split_ifs <;> ring
      · intro a v; funext k
        simp only [Tensor3.flatten, depthwiseConv2d, Tensor3.unflatten, Pi.smul_apply,
          Pi.zero_apply, zero_add, smul_eq_mul, Finset.mul_sum]
        refine Finset.sum_congr rfl fun kh _ => Finset.sum_congr rfl fun kw _ => ?_
        split_ifs <;> ring
    -- Step 2: substitute h_pdiv, rewrite each indicator to `co = ci ∧ tap lands`, collapse `co`.
    show depthwiseConv2dInputGradFormula W dy ci hi wi =
      ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
        pdiv3 (depthwiseConv2d W b) x ci hi wi co ho wo * dy co ho wo
    unfold depthwiseConv2dInputGradFormula pdiv3
    rw [← hidx_in]
    simp only [h_pdiv]
    simp only [Equiv.symm_apply_apply, hidx_in, padTap_indicator, ite_and, mul_ite,
      ite_mul, mul_one, mul_zero, zero_mul, Finset.sum_ite_irrel, Finset.sum_const_zero,
      Finset.sum_ite_eq', Finset.mem_univ, ite_true, sum_fin_ite_add_eq]
    clear h_pdiv
    refine Finset.sum_congr rfl fun ho _ => Finset.sum_congr rfl fun wo _ => ?_
    split_ifs <;> first | rfl | (exfalso; omega) | exact (zero_mul _).symm

/-- Named accessor for the depthwise input backward — aligns with MLIR
    codegen (per-channel `stablehlo.convolution` in the backward pass). -/
noncomputable abbrev depthwiseConv2dInputGrad {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w) : Tensor3 c h w :=
  (depthwiseHasVJP3 W b).backward x dy

-- ════════════════════════════════════════════════════════════════
-- § Differentiability + flattened-Vec witnesses (shared prereq for
--   MobileNetV2 / EfficientNet / ConvNeXt). Mirrors `conv2d_differentiable`
--   and `flatConv`/`flatConv_differentiable` in CNN.lean, with one fewer
--   sum level (no `Σ c`) since depthwise has no cross-channel mixing.
-- ════════════════════════════════════════════════════════════════

/-- **`depthwiseConv2d` is differentiable everywhere.** Mirror of
    `conv2d_differentiable`: `depthwiseConv2d W b x ch hi wi` is the affine
    map `b ch + ∑_{kh,kw} W ch kh kw · (pad-eval x)` — a constant bias plus a
    finite ℝ-linear combination of pad-guarded input reads
    (`differentiable_dite_zero`; no `Σ c` — depthwise reads only its own channel). -/
@[fun_prop]
theorem depthwise_differentiable {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Differentiable ℝ (depthwiseConv2d W b : Tensor3 c h w → Tensor3 c h w) := by
  unfold depthwiseConv2d; fun_prop

/-- **Flat depthwise conv** — `depthwiseConv2d` bridged into flattened
    `Vec → Vec` space: `flatten ∘ depthwiseConv2d W b ∘ unflatten`. Channels
    and spatial dims are preserved (`c h w → c h w`), so this is
    `Vec (c*h*w) → Vec (c*h*w)`. Mirror of `flatConv`; the form the
    MobileNet/EfficientNet/ConvNeXt VJP composition uses (flat Vec space). -/
noncomputable def depthwiseFlat {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun v => Tensor3.flatten (depthwiseConv2d W b (Tensor3.unflatten v))

/-- **`depthwiseFlat` is differentiable everywhere.** Composition of the
    three differentiable maps `unflatten`, `depthwiseConv2d`, `flatten`.
    Mirror of `flatConv_differentiable`. -/
@[fun_prop]
theorem depthwiseFlat_differentiable {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Differentiable ℝ (depthwiseFlat W b : Vec (c * h * w) → Vec (c * h * w)) :=
  Tensor3.flatten_differentiable.comp
    ((depthwise_differentiable W b).comp Tensor3.unflatten_differentiable)

@[fun_prop]
theorem depthwiseFlat_continuous {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Continuous (depthwiseFlat (h := h) (w := w) W b) :=
  (depthwiseFlat_differentiable W b).continuous

/-- **Flat depthwise conv input-VJP.** `depthwiseFlat W b` is defeq to the
    generic bridge's `fun v => flatten (depthwiseConv2d W b (unflatten v))`,
    so `HasVJP3.toHasVJP` applied to `depthwiseHasVJP3` lands the witness
    directly. Mirror of the regular-conv flat VJP. -/
noncomputable def depthwiseFlatHasVJP {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    HasVJP (depthwiseFlat W b : Vec (c * h * w) → Vec (c * h * w)) :=
  HasVJP3.toHasVJP (depthwiseHasVJP3 W b)

-- ════════════════════════════════════════════════════════════════
-- § Strided (stride-2) depthwise conv — `decimate ∘ depthwise` (ch7 C3)
-- ════════════════════════════════════════════════════════════════

/-- **Stride-2 SAME depthwise conv**, flattened: `Vec (c·2h·2w) → Vec (c·h·w)`.
    Defined as `decimateFlat ∘ depthwiseFlat` (the stride-1 SAME depthwise on the
    `2h×2w` grid, then keep even positions) — exactly the strided-conv recipe
    (`flatConvStride2`, StridedConv.lean) with the depthwise kernel. This is how
    MobileNetV2 downsamples (stride-2 depthwise inside an inverted-residual block);
    channels are unchanged (`c → c`), spatial halves. -/
noncomputable def depthwiseStride2Flat {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w) :=
  decimateFlat c h w ∘ (depthwiseFlat (h := 2 * h) (w := 2 * w) W b)

theorem depthwiseStride2Flat_differentiable {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Differentiable ℝ (depthwiseStride2Flat W b
      : Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w)) := by
  unfold depthwiseStride2Flat; fun_prop

@[fun_prop]
theorem depthwiseStride2Flat_continuous {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) : Continuous (depthwiseStride2Flat (h := h) (w := w) W b) :=
  (depthwiseStride2Flat_differentiable W b).continuous

/-- **Stride-2 depthwise input-VJP** — by the chain rule (`vjpComp`) on
    `decimateFlat ∘ depthwiseFlat`, reusing the proven stride-1 depthwise input-VJP
    (`depthwiseFlatHasVJP`) and the decimation VJP. The backward is
    `depthwise.back (decimate.back dy)` — i.e. zero-upsample the cotangent then run
    the reversed-kernel stride-1 depthwise (StableHLO: `stablehlo.pad` interior=1
    then `feature_group_count = c` reversed-kernel conv), exactly the `convStridedBack`
    shape with the per-channel grouping. -/
noncomputable def depthwiseStride2FlatHasVJP {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    HasVJP (depthwiseStride2Flat W b
      : Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w)) :=
  let hf_diff : Differentiable ℝ (depthwiseFlat (h := 2 * h) (w := 2 * w) W b) :=
    depthwiseFlat_differentiable W b
  let hf_vjp : HasVJP (depthwiseFlat (h := 2 * h) (w := 2 * w) W b) :=
    depthwiseFlatHasVJP W b
  show HasVJP (decimateFlat c h w ∘ (depthwiseFlat (h := 2 * h) (w := 2 * w) W b)) from
  vjpComp _ _ hf_diff (decimateFlat_differentiable c h w) hf_vjp (decimateFlatHasVJP c h w)

/-- **Stride-2 depthwise input-VJP correctness**:
    the backward equals the `pdiv`-contracted Jacobian of `depthwiseStride2Flat`. -/
theorem depthwiseStride2FlatHasVJP_correct {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Vec (c * (2 * h) * (2 * w))) (dy : Vec (c * h * w)) (i : Fin (c * (2 * h) * (2 * w))) :
    (depthwiseStride2FlatHasVJP W b).backward x dy i
      = ∑ j : Fin (c * h * w), pdiv (depthwiseStride2Flat W b) x i j * dy j :=
  (depthwiseStride2FlatHasVJP W b).correct x dy i

/-! ### Depthwise weight gradient (proved from foundation rules)

Per-channel transpose trick:

    `dW[c, kh, kw] = Σ_{h, w} x[c, h+kh−p, w+kw−p] · dy[c, h, w]`

Compare to the regular conv weight gradient:
- Regular conv: produces `(oc, ic, kH, kW)` — every (oc, ic) pair.
- Depthwise:    produces `(c, kH, kW)` — only the diagonal `(c, c)`
                pairs survive (the rest are zero by construction).

The transpose trick works the same way: view `x` and `dy` with
channel and batch axes swapped, do a standard conv, the spatial
dims of `dy` become the kernel dims. The only difference is that
`feature_group_count` is set so the conv stays per-channel.

MLIR (the depthwise variant of the transpose trick is in
`MlirCodegen.emitDepthwiseConvBnBackward`):

    "For depthwise: dW[c,1,kH,kW] = sum_b input[b,c,:,:] conv grad[b,c,:,:]"

**Framework.** Unlike the regular-conv weight gradient (which needs
`Kernel4.flatten` because the kernel is 4D), the depthwise kernel
`DepthwiseKernel c kH kW` is 3D — same shape as `Tensor3 c kH kW`, and
in fact definitionally equal. So we can reuse the existing `HasVJP3`
framework directly, parameterized over `W` instead of `x`. -/

/-- **Depthwise weight-VJP** — proved from foundation rules.

    `DepthwiseKernel c kH kW` is definitionally `Tensor3 c kH kW`, so
    `HasVJP3` applies directly. The function `W ↦ depthwiseConv2d W b x`
    is affine in W: at output (co, ho, wo) it's
    `b co + Σ_{kh, kw} W co kh kw * x_pad_term(co, kh, kw, ho, wo)`.
    Same recipe as `conv2dWeightGradHasVJP` but with two inner
    dims (kh, kw) instead of three (c, kh, kw) — depthwise has no
    cross-channel sum, so the "channel match" condition `co = ci` is
    a single equality rather than a packed comparison. -/
noncomputable def depthwiseWeightGradHasVJP3 {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w) :
    HasVJP3 (fun W : DepthwiseKernel c kH kW => depthwiseConv2d W b x) where
  backward := fun _W dy => fun ci hi_k wi_k =>
    ∑ ho : Fin h, ∑ wo : Fin w,
      (let pH := (kH - 1) / 2
       let pW := (kW - 1) / 2
       let hh := hi_k.val + ho.val
       let ww := wi_k.val + wo.val
       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
         x ci ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
       else 0)
      * dy ci ho wo
  correct := by
    intro W dy ci hi_k wi_k
    -- Per-(co, ho, wo) pdiv3 formula.
    have h_pdiv3 : ∀ co : Fin c, ∀ ho : Fin h, ∀ wo : Fin w,
        pdiv3 (fun W' : DepthwiseKernel c kH kW => depthwiseConv2d W' b x)
          W ci hi_k wi_k co ho wo =
        (if co = ci then
          (let pH := (kH - 1) / 2
           let pW := (kW - 1) / 2
           let hh := hi_k.val + ho.val
           let ww := wi_k.val + wo.val
           if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
             x ci ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
           else 0)
        else 0) := by
      intro co ho wo
      unfold pdiv3
      -- Affine in the kernel: the bias-free depthwise conv plus the broadcast bias.
      have hsplit : (fun v' : Vec (c * kH * kW) =>
            Tensor3.flatten (depthwiseConv2d
              (Tensor3.unflatten v' : DepthwiseKernel c kH kW) b x)) =
          fun v => Tensor3.flatten (depthwiseConv2d
              (Tensor3.unflatten v : DepthwiseKernel c kH kW) 0 x) +
            (fun k => b (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1) := by
        funext v k
        simp only [Tensor3.flatten, depthwiseConv2d, Pi.add_apply, Pi.zero_apply, zero_add]
        ring
      rw [hsplit, pdiv_of_affine]
      · simp only [Tensor3.flatten, depthwiseConv2d, Tensor3.unflatten, Pi.zero_apply, zero_add,
          basisVec_apply, Equiv.symm_apply_apply, EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq,
          ite_mul, one_mul, zero_mul]
        by_cases hc : co = ci
        · subst hc; simp [ite_and, Finset.sum_ite_irrel]
        · simp [hc]
      · intro u v; funext k
        simp only [Tensor3.flatten, depthwiseConv2d, Tensor3.unflatten, Pi.add_apply,
          Pi.zero_apply, zero_add, add_mul, Finset.sum_add_distrib]
      · intro a v; funext k
        simp only [Tensor3.flatten, depthwiseConv2d, Tensor3.unflatten, Pi.smul_apply,
          Pi.zero_apply, zero_add, smul_eq_mul, Finset.mul_sum, mul_assoc]
    -- Step 2: collapse the triple sum using h_pdiv3.
    show (∑ ho : Fin h, ∑ wo : Fin w,
            (let pH := (kH - 1) / 2
             let pW := (kW - 1) / 2
             let hh := hi_k.val + ho.val
             let ww := wi_k.val + wo.val
             if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
               x ci ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
             else 0)
            * dy ci ho wo) =
          ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
            pdiv3 (fun W' : DepthwiseKernel c kH kW => depthwiseConv2d W' b x)
              W ci hi_k wi_k co ho wo * dy co ho wo
    simp_rw [h_pdiv3]
    -- Now: LHS = ∑ co ho wo, (if co = ci then x_pad else 0) * dy co ho wo. Collapse co.
    rw [Finset.sum_eq_single ci
          (fun co _ hco_ne =>
            Finset.sum_eq_zero (fun ho _ =>
              Finset.sum_eq_zero (fun wo _ => by
                rw [ite_eq_right hco_ne, zero_mul])))
          (fun hni => absurd (Finset.mem_univ ci) hni)]
    -- Now: LHS = ∑ ho wo, (if ci = ci then x_pad(ci) else 0) * dy ci ho wo. The if is true.
    apply Finset.sum_congr rfl; intro ho _
    apply Finset.sum_congr rfl; intro wo _
    rw [ite_eq_left rfl]

/-- Named accessor for the depthwise weight backward. -/
noncomputable abbrev depthwiseConv2dWeightGrad {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w) : DepthwiseKernel c kH kW :=
  (depthwiseWeightGradHasVJP3 b x).backward W dy

/-- **Depthwise bias-VJP** — proved from foundation rules. Same shape
    as `conv2dBiasGradHasVJP`, just simpler: depthwise has no
    Σ over input channels (input channel = output channel). The
    function `b ↦ flatten(depthwiseConv2d W b x)` decomposes as
    `(channel-reindex from b) + (W,x term constant in b)`, exactly
    like conv2d's case. -/
noncomputable def depthwiseBiasGradHasVJP {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) :
    HasVJP (fun b : Vec c => Tensor3.flatten (depthwiseConv2d W b x)) where
  backward := fun _b dy => fun cc =>
    ∑ hi : Fin h, ∑ wi : Fin w,
      dy (finProdFinEquiv (finProdFinEquiv (cc, hi), wi))
  correct := by
    intro b dy cc
    -- Affine in the bias: the channel broadcast of `b` plus the bias-free conv.
    have hsplit : (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) =
        fun b' => (fun k : Fin (c * h * w) =>
            b' (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1) +
          Tensor3.flatten (depthwiseConv2d W 0 x) := by
      funext b' k
      simp only [Tensor3.flatten, depthwiseConv2d, Pi.add_apply, Pi.zero_apply, zero_add]
    simp only [hsplit, pdiv_of_affine (fun (b' : Vec c) (k : Fin (c * h * w)) =>
      b' (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1) _ (fun _ _ => rfl) (fun _ _ => rfl),
      basisVec_apply, ite_mul, one_mul, zero_mul]
    rw [sum_finProdFinEquiv₃]
    simp

/-- Named accessor for the depthwise bias backward via the VJP framework. -/
noncomputable def depthwiseConv2dBiasGrad {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w) : Vec c :=
  (depthwiseBiasGradHasVJP W x).backward b (Tensor3.flatten dy)

/-- **Depthwise bias gradient — closed-form formula.**

    `db[c] = Σ_{h, w} dy[c, h, w]`

    This is the sum `depthwiseBiasGradHasVJP`'s backward is defined as, on
    `Tensor3` instead of the flattened `dy`. No theorem in this file states
    the equation with `depthwiseConv2dBiasGrad`.

    Identical to regular conv's bias gradient — the bias is per-channel
    in both cases, and it adds the same value to every spatial cell
    of its channel. The reduction is the same. -/
noncomputable def depthwiseConv2dBiasGradFormula {c h w : Nat}
    (dy : Tensor3 c h w) : Vec c :=
  fun cc => ∑ y : Fin h, ∑ x : Fin w, dy cc y x

-- ════════════════════════════════════════════════════════════════
-- § The relationship to regular conv
-- ════════════════════════════════════════════════════════════════

/-! ## Depthwise = constrained regular conv

Conceptually, depthwise conv is regular conv with a sparsity pattern
on the kernel: `W_regular[o, c, kh, kw]` is zero unless `o = c`. Equivalently,
`W_regular` is **block-diagonal** in the `(o, c)` channel pair.

Two consequences for the VJPs:

1. **Forward**: the cross-channel sum `Σ_c` collapses to a single term
   (the diagonal one), giving the per-channel formula above.

2. **Backward**: every formula involving `Σ_o` or `Σ_c` over channel
   indices collapses similarly. The transpose trick still produces a
   `(ic, oc, kH, kW)` tensor in principle, but only the diagonal slice
   is nonzero, and the implementation just stores the diagonal.

So the depthwise VJPs have the regular-conv shape specialized to the
sparsity pattern — a constraint on the forward propagates to a constraint
on the backward. (This file proves them directly, not by specializing the
regular-conv theorems.)

## Cost

The forward op cost goes from `O(B · oc · ic · H · W · kH · kW)` for
regular conv to `O(B · c · H · W · kH · kW)` for depthwise — saves a
factor of `oc` (typically 32–512). The backward cost reduces by the
same factor. This is why mobile architectures pair depthwise with a
cheap 1×1 pointwise conv for cross-channel mixing — together they have
the same expressive power as a regular conv at a fraction of the FLOPs.

## Where it's used

- **MobileNet v1/v2/v3** (`MainMobilenet.lean`, `MainMobilenetV2.lean`,
  `MainMobilenetV3.lean`) — depthwise everywhere.
- **EfficientNet** (`MainEfficientNet.lean`) — depthwise inside MBConv blocks.
- **MBConv** (`MainEfficientNet.lean`, `MainEfficientNetV2.lean`) — the
  block that pairs an expand 1×1 → depthwise k×k → project 1×1, with
  optional Squeeze-and-Excitation. See `SE.lean` for the SE part.
-/

/-! ## Summary of derivations in this file

**None.** The forward `depthwiseConv2d` is a concrete definition (not a
black-box), and all three VJPs are theorems proved from the foundation
rules in `Tensor.lean`:

- `depthwiseHasVJP3` — input-path VJP, proved with `pdiv_of_affine` (the
  depthwise conv is affine in its input).
  Mirrors `conv2dHasVJP3` with one fewer sum level (no Σ c) and a
  prepended Σ co collapse.
- `depthwiseWeightGradHasVJP3` — weight-path VJP, bundled as
  `HasVJP3` directly (no flattening needed; see framework note above).
  Gradient-checked numerically.
- `depthwiseBiasGradHasVJP` — bias-path VJP, bundled `HasVJP` on
  the flattened output. Same pattern as conv2d's bias VJP.

Pure-Mathlib closure verified via `#print axioms` (only `propext`,
`Classical.choice`, `Quot.sound`).

Derived helpers (not axioms):
- `depthwiseConv2dInputGrad`, `depthwiseConv2dWeightGrad`,
  `depthwiseConv2dBiasGrad` — named accessors, `.backward` of the
  corresponding VJP.
- `depthwiseConv2dInputGradFormula` — the concrete sum-over-output-
  positions closed-form, used as the backward of `depthwiseHasVJP3`.
- `depthwiseConv2dBiasGradFormula` — the concrete sum-over-spatial
  closed-form; the same sum as `depthwiseBiasGradHasVJP`'s backward, with
  no theorem stating the equation. -/

/-- **Public correctness theorem for `depthwiseHasVJP3`**: the
proved input-VJP's backward equals the `pdiv3`-contracted Jacobian. -/
theorem depthwiseHasVJP3_correct {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    (depthwiseHasVJP3 (h := h) (w := w) W b).backward x dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (depthwiseConv2d W b : Tensor3 c h w → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  (depthwiseHasVJP3 (h := h) (w := w) W b).correct x dy ci hi wi

-- ════════════════════════════════════════════════════════════════
-- § Strided (stride-2) depthwise param VJPs — RELOCATED here from
--   `MobileNetV2Close.lean` so the `depthwiseStrided{Weight,Bias}Sgd` ops'
--   `den` in `StableHLO` can reference them upstream (the same move the strided
--   *conv* bias VJP made into `StridedConv.lean`). Each strided forward is
--   `decimateFlat ∘ (stride-1 depthwise op)`, so the param VJP is `vjpComp`
--   of a proven stride-1 depthwise VJP with the decimation VJP — the backward
--   is "zero-upsample the cotangent (StableHLO `pad` interior=1), then the
--   stride-1 grad", exactly the render's `dwconvWGradStrided`.
-- ════════════════════════════════════════════════════════════════

/-- **`depthwiseConv2d` (as a function of its kernel) is differentiable** — affine in `W`. The
    depthwise peer of `conv2d_weight_differentiable`; the `vjpComp` hypothesis for the strided
    weight-grad. -/
theorem depthwise_weight_differentiable {c h w kH kW : Nat} (b : Vec c) (x : Tensor3 c h w) :
    Differentiable ℝ (fun v : Vec (c * kH * kW) =>
      Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v : DepthwiseKernel c kH kW) b x)) := by
  unfold depthwiseConv2d Tensor3.flatten Tensor3.unflatten
  fun_prop

/-- **`depthwiseConv2d` (as a function of its bias) is differentiable** — affine in `b`. The
    `vjpComp` hypothesis for the strided depthwise bias-grad. -/
theorem depthwise_bias_differentiable {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) :
    Differentiable ℝ (fun b : Vec c => Tensor3.flatten (depthwiseConv2d W b x)) := by
  unfold depthwiseConv2d Tensor3.flatten
  fun_prop

/-- **Stride-2 depthwise weight-VJP.** `fun v => depthwiseStride2Flat (unflatten v) b x =
    decimate ∘ (depthwise-weight-in-v)`; by `vjpComp` of the proven stride-1
    `depthwiseWeightGradHasVJP3` (flattened via `HasVJP3.toHasVJP`) with `decimateFlatHasVJP`.
    The depthwise peer of `flatConvStride2WeightGradHasVJP`. -/
noncomputable def depthwiseStride2WeightGradHasVJP {c h w kH kW : Nat}
    (b : Vec c) (x : Vec (c * (2 * h) * (2 * w))) :
    HasVJP (fun v : Vec (c * kH * kW) =>
      depthwiseStride2Flat (Tensor3.unflatten v : DepthwiseKernel c kH kW) b x) :=
  let f : Vec (c * kH * kW) → Vec (c * (2 * h) * (2 * w)) :=
    fun v => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v : DepthwiseKernel c kH kW) b
              (Tensor3.unflatten x))
  let hf_diff : Differentiable ℝ f :=
    depthwise_weight_differentiable (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten x)
  let hf_vjp : HasVJP f :=
    HasVJP3.toHasVJP (depthwiseWeightGradHasVJP3 (h := 2 * h) (w := 2 * w) b
      (Tensor3.unflatten x))
  show HasVJP (decimateFlat c h w ∘ f) from
  vjpComp f (decimateFlat c h w) hf_diff (decimateFlat_differentiable c h w)
    hf_vjp (decimateFlatHasVJP c h w)

/-- **Stride-2 depthwise bias-VJP.** `fun b => depthwiseStride2Flat W b x = decimate ∘
    (depthwise-bias-in-b)`; by `vjpComp` of the proven stride-1 `depthwiseBiasGradHasVJP` with
    `decimateFlatHasVJP`. -/
noncomputable def depthwiseStride2BiasGradHasVJP {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (c * (2 * h) * (2 * w))) :
    HasVJP (fun b : Vec c =>
      depthwiseStride2Flat W b x : Vec c → Vec (c * h * w)) :=
  let g : Vec c → Vec (c * (2 * h) * (2 * w)) :=
    fun b => Tensor3.flatten (depthwiseConv2d W b (Tensor3.unflatten x))
  let hg_diff : Differentiable ℝ g :=
    depthwise_bias_differentiable (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten x)
  let hg_vjp : HasVJP g :=
    depthwiseBiasGradHasVJP (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten x)
  show HasVJP (decimateFlat c h w ∘ g) from
  vjpComp g (decimateFlat c h w) hg_diff (decimateFlat_differentiable c h w)
    hg_vjp (decimateFlatHasVJP c h w)

-- ════════════════════════════════════════════════════════════════
-- § Stride-2 depthwise at XLA `SAME` = decimateODD ∘ (stride-1 depthwise)
--   (`planning/archive/mnv4_verified.md` §3e — the TF-origin padding convention)
-- ════════════════════════════════════════════════════════════════

/-! **The depthwise peer of `flatConvStride2Xla`** ([`Architectures/StridedConv.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Architectures/StridedConv.lean)), and it exists
for the same reason: `depthwise_conv` in [`jax/Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean)'s `depthwise_conv` defaults to `padding='SAME'`,
so MobileNetV2's four strided depthwises — and EfficientNet's — pad **asymmetrically**, while
`depthwiseStride2Flat` above pads symmetrically. Both give the same output size, so only a forward
tie can see the difference.

Identical structure to the regular-conv case, so identical cost: the asymmetry is a **phase
shift in the decimation**, `decimateOddFlat` instead of `decimateFlat`. `depthwiseFlat` and all of
its VJPs are reused verbatim, and `decimateOddFlatHasVJP` is already proven, so nothing here is
a new obligation.

Even inputs only — which the type enforces (`c*(2*h)*(2*w)`) and which is every strided
depthwise in mnv2/mnv4/enet (112, 56, 28, 14). At an odd input XLA `SAME` is symmetric and
`depthwiseStride2Flat` is already correct. -/
-- Measured (planning/archive/mnv4_verified.md §3d): at MNv2's five sites the symmetric/`SAME`
-- difference was 2.9e-1 of a ~1.05 logit range in its trainer's BN world.

/-- **Stride-2 XLA-`SAME` depthwise conv**, flattened: `Vec (c·2h·2w) → Vec (c·h·w)`.
    `decimateOddFlat ∘ depthwiseFlat` — the asymmetric-pad peer of `depthwiseStride2Flat`. -/
noncomputable def depthwiseStride2FlatXla {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w) :=
  decimateOddFlat c h w ∘ (depthwiseFlat (h := 2 * h) (w := 2 * w) W b)

@[fun_prop]
theorem depthwiseStride2FlatXla_differentiable {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    Differentiable ℝ (depthwiseStride2FlatXla W b
      : Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w)) := by
  unfold depthwiseStride2FlatXla; fun_prop

@[fun_prop]
theorem depthwiseStride2FlatXla_continuous {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW)
    (b : Vec c) : Continuous (depthwiseStride2FlatXla (h := h) (w := w) W b) :=
  (depthwiseStride2FlatXla_differentiable W b).continuous

/-- **Stride-2 XLA-`SAME` depthwise input-VJP.** `vjpComp` on `decimateOddFlat ∘ depthwiseFlat`.
    The backward zero-upsamples the cotangent onto the **odd** positions, then runs the
    reversed-kernel grouped conv — so the forward's asymmetry is placed by the backward too. A
    symmetric backward against this forward is a silent wrong-gradient. -/
noncomputable def depthwiseStride2FlatXlaHasVJP {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) :
    HasVJP (depthwiseStride2FlatXla W b
      : Vec (c * (2 * h) * (2 * w)) → Vec (c * h * w)) :=
  let hf_diff : Differentiable ℝ (depthwiseFlat (h := 2 * h) (w := 2 * w) W b) :=
    depthwiseFlat_differentiable W b
  let hf_vjp : HasVJP (depthwiseFlat (h := 2 * h) (w := 2 * w) W b) :=
    depthwiseFlatHasVJP W b
  show HasVJP (decimateOddFlat c h w ∘ (depthwiseFlat (h := 2 * h) (w := 2 * w) W b)) from
  vjpComp _ _ hf_diff (decimateOddFlat_differentiable c h w) hf_vjp
    (decimateOddFlatHasVJP c h w)

/-- **Stride-2 XLA-`SAME` depthwise weight-VJP.** The kernel-side peer, by `vjpComp` of the proven
    stride-1 `depthwiseWeightGradHasVJP3` with the odd-decimation VJP. -/
noncomputable def depthwiseStride2XlaWeightGradHasVJP {c h w kH kW : Nat}
    (b : Vec c) (x : Vec (c * (2 * h) * (2 * w))) :
    HasVJP (fun v : Vec (c * kH * kW) =>
      depthwiseStride2FlatXla (Tensor3.unflatten v : DepthwiseKernel c kH kW) b x) :=
  let f : Vec (c * kH * kW) → Vec (c * (2 * h) * (2 * w)) :=
    fun v => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v : DepthwiseKernel c kH kW) b
              (Tensor3.unflatten x))
  let hf_diff : Differentiable ℝ f :=
    depthwise_weight_differentiable (h := 2 * h) (w := 2 * w) b (Tensor3.unflatten x)
  let hf_vjp : HasVJP f :=
    HasVJP3.toHasVJP (depthwiseWeightGradHasVJP3 (h := 2 * h) (w := 2 * w) b
      (Tensor3.unflatten x))
  show HasVJP (decimateOddFlat c h w ∘ f) from
  vjpComp f (decimateOddFlat c h w) hf_diff (decimateOddFlat_differentiable c h w)
    hf_vjp (decimateOddFlatHasVJP c h w)

/-- **Stride-2 XLA-`SAME` depthwise bias-VJP.** -/
noncomputable def depthwiseStride2XlaBiasGradHasVJP {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (c * (2 * h) * (2 * w))) :
    HasVJP (fun b : Vec c =>
      depthwiseStride2FlatXla W b x : Vec c → Vec (c * h * w)) :=
  let g : Vec c → Vec (c * (2 * h) * (2 * w)) :=
    fun b => Tensor3.flatten (depthwiseConv2d W b (Tensor3.unflatten x))
  let hg_diff : Differentiable ℝ g :=
    depthwise_bias_differentiable (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten x)
  let hg_vjp : HasVJP g :=
    depthwiseBiasGradHasVJP (h := 2 * h) (w := 2 * w) W (Tensor3.unflatten x)
  show HasVJP (decimateOddFlat c h w ∘ g) from
  vjpComp g (decimateOddFlat c h w) hg_diff (decimateOddFlat_differentiable c h w)
    hg_vjp (decimateOddFlatHasVJP c h w)

-- ════════════════════════════════════════════════════════════════
-- § Depthwise SGD-tail denotations — non-reducing wrappers for the `SHlo`
--   `depthwise{,Strided}{Weight,Bias}Sgd` `den` arms. Defined here (not inlined
--   in `den`) so the `den` match stays small: `depthwiseWeightGradHasVJP3` /
--   `depthwiseBiasGradHasVJP` are STRUCTURE LITERALS whose `.backward`
--   reduces to a big sum, so inlining them in `den` would bloat the match and
--   blow the heartbeat limit of every `simp only [den]` proof. The `*Fold`
--   `den = certified` lemmas unfold these first, then close via the
--   `mnv2_render_depthwise*_certified` bridges.
-- ════════════════════════════════════════════════════════════════

/-- Stride-1 depthwise weight SGD step: `flatten W − lr·flatten(dwconv_weight_grad(b,x)·dy)`. -/
noncomputable def depthwiseWeightSgdDen {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW) (lr : ℝ) (dy : Vec (c*h*w)) :
    Vec (c*kH*kW) :=
  fun idx => Tensor3.flatten W idx
    - lr * Tensor3.flatten ((depthwiseWeightGradHasVJP3 b x).backward W (Tensor3.unflatten dy)) idx

/-- Stride-1 depthwise bias SGD step: `b − lr·(dwconv_bias_grad(W,x)·dy)`. -/
noncomputable def depthwiseBiasSgdDen {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w) (b : Vec c) (lr : ℝ) (dy : Vec (c*h*w)) :
    Vec c :=
  fun o => b o - lr * (depthwiseBiasGradHasVJP W x).backward b dy o

/-- Stride-2 depthwise weight SGD step: `flatten W − lr·(dwconvStride2_weight_grad(b,x)·dy)`. -/
noncomputable def depthwiseStridedWeightSgdDen {c h w kH kW : Nat}
    (b : Vec c) (x : Vec (c*(2*h)*(2*w))) (W : DepthwiseKernel c kH kW) (lr : ℝ) (dy : Vec (c*h*w)) :
    Vec (c*kH*kW) :=
  fun idx => Tensor3.flatten W idx
    - lr * (depthwiseStride2WeightGradHasVJP b x).backward (Tensor3.flatten W) dy idx

/-- Stride-2 depthwise bias SGD step: `b − lr·(dwconvStride2_bias_grad(W,x)·dy)`. -/
noncomputable def depthwiseStridedBiasSgdDen {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (c*(2*h)*(2*w))) (b : Vec c) (lr : ℝ) (dy : Vec (c*h*w)) :
    Vec c :=
  fun o => b o - lr * (depthwiseStride2BiasGradHasVJP W x).backward b dy o

/-- Stride-2 **XLA-`SAME`** depthwise weight SGD step — `depthwiseStridedWeightSgdDen`'s peer at
    the odd decimation phase (`depthwiseStride2XlaWeightGradHasVJP`). Same non-reducing
    wrapper, for the same `den`-match-size reason. -/
noncomputable def depthwiseStridedXlaWeightSgdDen {c h w kH kW : Nat}
    (b : Vec c) (x : Vec (c*(2*h)*(2*w))) (W : DepthwiseKernel c kH kW) (lr : ℝ) (dy : Vec (c*h*w)) :
    Vec (c*kH*kW) :=
  fun idx => Tensor3.flatten W idx
    - lr * (depthwiseStride2XlaWeightGradHasVJP b x).backward (Tensor3.flatten W) dy idx

/-- Stride-2 **XLA-`SAME`** depthwise bias SGD step. -/
noncomputable def depthwiseStridedXlaBiasSgdDen {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (c*(2*h)*(2*w))) (b : Vec c) (lr : ℝ) (dy : Vec (c*h*w)) :
    Vec c :=
  fun o => b o - lr * (depthwiseStride2XlaBiasGradHasVJP W x).backward b dy o

/-- A depthwise conv with everywhere-zero kernel and bias maps anything to `0`. -/
theorem depthwiseFlat_eq_zero {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (hW : ∀ ch kh kw, W ch kh kw = 0) (hb : ∀ ch, b ch = 0) (v : Vec (c * h * w)) :
    depthwiseFlat (h := h) (w := w) W b v = (fun _ => (0:ℝ)) := by
  funext k; simp [depthwiseFlat, depthwiseConv2d, Tensor3.flatten, hW, hb]

end Proofs
