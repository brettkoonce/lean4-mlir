import LeanMlir.Proofs.Depthwise
import LeanMlir.Proofs.Foundation.StridedConv
import LeanMlir.Proofs.Foundation.CnnTrainStep
import LeanMlir.Proofs.CifarBnClose

/-! # Closing the MobileNetV2 render — the depthwise / strided parameter-gradient bridges

`planning/mobilenetv2_close.md` Item C — the "free close" (generic in the cotangent the
backward chain delivers at each layer's output, the CIFAR-non-BN-style close): every
MobileNetV2 train-step parameter output denotes `θ − lr·(certified Jacobian · cotangent)`.

The MobileNetV2 train step (`tests/TestMobilenetV2Train.lean`) has these parameter families,
and each is now certified by the bridge in the right column:

| family (render SSA)                         | forward fn          | certified by                                  |
|---------------------------------------------|---------------------|-----------------------------------------------|
| 1×1 conv W (expand `eW` / project `pW` / head `hW`) | `conv2d` (stride 1) | `cnn_render_convW_certified` (M3, **reuse**)  |
| 1×1 conv b (`eb` / `pb` / `hb`)             | `conv2d`            | `cnn_render_convb_certified` (M3, **reuse**)  |
| BN γ (`eg`/`dg`/`pg`/`hg`/`sg`)             | `bnPerChannelFlat`  | `cifar_bn_render_gamma_certified` (**reuse**) |
| BN β (`ebt`/`dbt`/`pbt`/`hbt`/`sbt`)        | `bnPerChannelFlat`  | `cifar_bn_render_beta_certified` (**reuse**)  |
| dense `Wd` / `bd`                           | matmul / +bias      | `weight_grad_bridge` / `bias_grad_bridge` (M2, **reuse**) |
| stem 3×3 conv W (`sW`, stride 2)            | `flatConvStride2`   | `mnv2_render_stem_convW_certified` (**new wrapper**) |
| stem 3×3 conv b (`sb`, stride 2)            | `flatConvStride2`   | `mnv2_render_stem_convb_certified` (**new**)  |
| depthwise W stride 1 (`dW`, blocks b2,b4)   | `depthwiseConv2d`   | `mnv2_render_depthwiseW_certified` (**new**)  |
| depthwise b stride 1 (`db`, blocks b2,b4)   | `depthwiseConv2d`   | `mnv2_render_depthwiseb_certified` (**new**)  |
| depthwise W stride 2 (`dW`, blocks b1,b3,b5,b6) | `depthwiseStride2Flat` | `mnv2_render_depthwiseW_strided_certified` (**new**) |
| depthwise b stride 2 (`db`, blocks b1,b3,b5,b6) | `depthwiseStride2Flat` | `mnv2_render_depthwiseb_strided_certified` (**new**) |

The reuse families need no new theorem — the generic M2/M3/CIFAR-BN bridges apply verbatim at
the MobileNetV2 shapes. This file supplies the genuinely-new pieces:

* **Depthwise (stride-1) W/b** — the `.correct` fields of the proven `depthwise_weight_grad_has_vjp3`
  / `depthwise_bias_grad_has_vjp` (`Depthwise.lean`), SGD-wrapped. The "one genuinely-new bridge
  family" of the plan — instantiation, the VJP itself is already proven 3-axiom-clean.
* **Stem strided conv W/b** — wrappers of `flatConvStride2_weight_grad_has_vjp` (ch6) and a new
  strided-conv *bias* VJP.
* **Strided depthwise W/b (4 of 6 blocks downsample)** — a new strided-depthwise weight/bias VJP,
  the exact `decimate ∘ stride-1` recipe of `flatConvStride2_weight_grad_has_vjp` with the depthwise
  kernel. (The plan's Item C list omitted these; the downsampling blocks need them for honest
  coverage.) Each is `vjp_comp` of a proven stride-1 depthwise VJP with `decimateFlat`'s VJP.

All bridges are generic in the cotangent `c`/`dy` the backward chain delivers at the layer output
(pinning that cotangent to the actual inverted-residual chain is the optional Item D). The SGD
wrapping `θ − lr·∇` is identical to the linear/MLP/CNN cases.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § A. Depthwise (stride-1) parameter bridges — the genuinely-new family
--
-- `depthwise_weight_grad_has_vjp3` / `depthwise_bias_grad_has_vjp` (Depthwise.lean) are the
-- proven, foundation-rule depthwise param VJPs. Their `.correct` fields are the bridges: the
-- rendered `dwconvWGrad` (per-channel transpose trick) / `convBiasGrad` (spatial reduce) equal
-- the certified Jacobian of `depthwiseConv2d` (as a function of W / of b) contracted with the
-- cotangent. The depthwise analogue of `conv_weight_grad_bridge` / `conv_bias_grad_bridge`.
-- ════════════════════════════════════════════════════════════════

/-- **Depthwise weight-gradient bridge.** At any cotangent `dy` at the depthwise layer's output
    and any kernel `W`, the emitted per-channel depthwise kernel gradient equals the certified
    Jacobian of `depthwiseConv2d` viewed as a function of the kernel, contracted with `dy`. The
    `.correct` field of `depthwise_weight_grad_has_vjp3`. -/
theorem mnv2_depthwise_weight_grad_bridge {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w)
    (W : DepthwiseKernel c kH kW) (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin kH) (wi : Fin kW) :
    (depthwise_weight_grad_has_vjp3 b x).backward W dy ci hi wi
      = ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
          pdiv3 (fun W' : DepthwiseKernel c kH kW => depthwiseConv2d W' b x) W ci hi wi co ho wo
            * dy co ho wo :=
  (depthwise_weight_grad_has_vjp3 b x).correct W dy ci hi wi

/-- **Depthwise bias-gradient bridge.** Likewise the per-channel depthwise bias gradient
    (`db[c] = Σ_spatial dy`) is the certified Jacobian of `depthwiseConv2d` wrt the bias, contracted
    with `dy` — the `.correct` field of `depthwise_bias_grad_has_vjp`. -/
theorem mnv2_depthwise_bias_grad_bridge {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w)
    (b : Vec c) (dy : Vec (c * h * w)) (cc : Fin c) :
    (depthwise_bias_grad_has_vjp W x).backward b dy cc
      = ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b cc j * dy j :=
  (depthwise_bias_grad_has_vjp W x).correct b dy cc

/-- **Depthwise weight output, certified.** `Wⁿ = W − lr·(per-channel transpose-trick grad)`
    denotes, at the kernel `W`, `W − lr·(certified ∂(depthwiseConv2d)/∂W · cotangent)`. The
    depthwise peer of `cnn_render_convW_certified`. -/
theorem mnv2_render_depthwiseW_certified {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w)
    (W : DepthwiseKernel c kH kW) (dy : Tensor3 c h w) (lr : ℝ)
    (ci : Fin c) (hi : Fin kH) (wi : Fin kW) :
    W ci hi wi - lr * (depthwise_weight_grad_has_vjp3 b x).backward W dy ci hi wi
      = W ci hi wi - lr * ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
          pdiv3 (fun W' : DepthwiseKernel c kH kW => depthwiseConv2d W' b x) W ci hi wi co ho wo
            * dy co ho wo := by
  rw [mnv2_depthwise_weight_grad_bridge]

/-- **Depthwise bias output, certified.** Likewise `bⁿ = b − lr·(spatial reduce)` denotes
    `b − lr·(certified ∂(depthwiseConv2d)/∂b · cotangent)`. -/
theorem mnv2_render_depthwiseb_certified {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w)
    (b : Vec c) (dy : Vec (c * h * w)) (lr : ℝ) (cc : Fin c) :
    b cc - lr * (depthwise_bias_grad_has_vjp W x).backward b dy cc
      = b cc - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b cc j * dy j := by
  rw [mnv2_depthwise_bias_grad_bridge]

-- ════════════════════════════════════════════════════════════════
-- § B. Stem strided 3×3 conv — weight reuses ch6, bias is new
--
-- The stem (`conv3WGradStrided`) reuses `flatConvStride2_weight_grad_has_vjp` (StridedConv.lean)
-- for the kernel; the SGD wrapper is the only new content. The stem bias needs a strided-conv
-- *bias* VJP (added in § C alongside the strided-depthwise bias).
-- ════════════════════════════════════════════════════════════════

/-- **Stem conv weight output, certified.** `sWⁿ = sW − lr·(strided transpose-trick grad)` denotes
    `sW − lr·(certified ∂(flatConvStride2)/∂sW · cotangent)`, via `flatConvStride2_weight_grad_has_vjp`
    (the ch6 strided conv weight VJP). -/
theorem mnv2_render_stem_convW_certified {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w)))
    (v : Vec (oc * ic * kH * kW)) (dy : Vec (oc * h * w)) (lr : ℝ)
    (i : Fin (oc * ic * kH * kW)) :
    v i - lr * (flatConvStride2_weight_grad_has_vjp b x).backward v dy i
      = v i - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) => flatConvStride2 (Kernel4.unflatten v') b x)
            v i j * dy j := by
  rw [flatConvStride2_weight_grad_has_vjp_correct]

-- ════════════════════════════════════════════════════════════════
-- § C. Strided (stride-2) param VJPs — the 4 downsampling blocks + the stem bias
--
-- 4 of the 6 inverted-residual blocks (b1,b3,b5,b6) downsample via a stride-2 depthwise, so
-- their depthwise W/b feed `depthwiseStride2Flat`, not `depthwiseConv2d`; likewise the stem bias
-- feeds the strided `flatConvStride2`. The strided-depthwise VJPs themselves
-- (`depthwise_weight_differentiable`, `depthwise_bias_differentiable`,
-- `depthwiseStride2_weight_grad_has_vjp`, `depthwiseStride2_bias_grad_has_vjp`) were RELOCATED to
-- `Depthwise.lean` (next to their stride-1 peers) so the `depthwiseStrided{Weight,Bias}Sgd` ops'
-- `den` in `StableHLO` can reference them upstream; they are still in scope here by import. The
-- `mnv2_render_depthwise*_strided_certified` SGD-wrappers below stay here.
-- ════════════════════════════════════════════════════════════════

-- ── C.1 Stem strided-conv bias (`sb`) ──
-- `conv2d_bias_differentiable` and `flatConvStride2_bias_grad_has_vjp` were RELOCATED to
-- `StridedConv.lean` (next to their weight peers) so the `convStridedBiasSgd` op's `den` in
-- `StableHLO` can reference the bias-VJP upstream; they are still in scope here by import.

/-- **Stem conv bias output, certified.** `sbⁿ = sb − lr·(spatial reduce)` denotes
    `sb − lr·(certified ∂(flatConvStride2)/∂sb · cotangent)`. -/
theorem mnv2_render_stem_convb_certified {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Vec (ic * (2 * h) * (2 * w)))
    (b : Vec oc) (dy : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (flatConvStride2_bias_grad_has_vjp W x).backward b dy o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * dy j := by
  rw [(flatConvStride2_bias_grad_has_vjp W x).correct]

-- ── C.2 Strided depthwise weight (`dW`, blocks b1,b3,b5,b6) ──
-- (`depthwiseStride2_weight_grad_has_vjp` RELOCATED to `Depthwise.lean` — see § C header.)

/-- **Strided depthwise weight output, certified.** `Wⁿ = W − lr·(upsample-then-stride-1 grad)`
    denotes `W − lr·(certified ∂(depthwiseStride2Flat)/∂W · cotangent)`. -/
theorem mnv2_render_depthwiseW_strided_certified {c h w kH kW : Nat}
    (b : Vec c) (x : Vec (c * (2 * h) * (2 * w)))
    (v : Vec (c * kH * kW)) (dy : Vec (c * h * w)) (lr : ℝ) (i : Fin (c * kH * kW)) :
    v i - lr * (depthwiseStride2_weight_grad_has_vjp b x).backward v dy i
      = v i - lr * ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
            depthwiseStride2Flat (Tensor3.unflatten v' : DepthwiseKernel c kH kW) b x) v i j * dy j := by
  rw [(depthwiseStride2_weight_grad_has_vjp b x).correct]

-- ── C.3 Strided depthwise bias (`db`, blocks b1,b3,b5,b6) ──
-- (`depthwiseStride2_bias_grad_has_vjp` RELOCATED to `Depthwise.lean` — see § C header.)

/-- **Strided depthwise bias output, certified.** `bⁿ = b − lr·(spatial reduce)` denotes
    `b − lr·(certified ∂(depthwiseStride2Flat)/∂b · cotangent)`. -/
theorem mnv2_render_depthwiseb_strided_certified {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (c * (2 * h) * (2 * w)))
    (b : Vec c) (dy : Vec (c * h * w)) (lr : ℝ) (o : Fin c) :
    b o - lr * (depthwiseStride2_bias_grad_has_vjp W x).backward b dy o
      = b o - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => depthwiseStride2Flat W b' x) b o j * dy j := by
  rw [(depthwiseStride2_bias_grad_has_vjp W x).correct]

end Proofs
