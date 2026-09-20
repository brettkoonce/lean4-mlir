import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Foundation.StridedConv
import LeanMlir.Proofs.Nets.Small.CnnTrainStep
import LeanMlir.Proofs.Nets.Small.CifarBnClose

/-! # Closing the MobileNetV2 render — the depthwise / strided parameter-gradient bridges

`planning/archive/mobilenetv2_close.md` Item C — the "free close" (generic in the cotangent the
backward chain delivers at each layer's output, the CIFAR-non-BN-style close): every
MobileNetV2 train-step parameter output denotes `θ − lr·(certified Jacobian · cotangent)`.

The MobileNetV2 train step (`TestMobilenetV2Train.lean` (retired 2026-09-20)) has these parameter families,
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
| depthwise W stride 1 (`dW`, blocks b2,b4)   | `depthwiseConv2d`   | `Mnv2PoC.depthwiseW_den` (`MobileNetV2Fold.lean`; this file's own wrapper retired 2026-09-20) |
| depthwise b stride 1 (`db`, blocks b2,b4)   | `depthwiseConv2d`   | `mnv2_render_depthwiseb_certified` (**new**)  |

The reuse families need no new theorem — the generic M2/M3/CIFAR-BN bridges apply verbatim at
the MobileNetV2 shapes. This file supplies the genuinely-new pieces:

* **Depthwise (stride-1) b** — the `.correct` field of the proven `depthwise_bias_grad_has_vjp`
  (`Depthwise.lean`), SGD-wrapped (the W twin, superseded by `Mnv2PoC.depthwiseW_den`, was retired). The "one genuinely-new bridge
  family" of the plan — instantiation, the VJP itself is already proven 3-axiom-clean.
* **Stem strided conv W/b** — wrappers of `flatConvStride2_weight_grad_has_vjp` (ch6) and a new
  strided-conv *bias* VJP.

The shipped MobileNetV2 is batched, with XLA-`SAME` stride-2 layers. `mnv2_net_tiedB` certifies its
stride-2 depthwise and stem parameters through `Mnv2PaperPoCG.depthwiseStridedXlaWGradB_den` /
`Mnv2PaperPoCG.depthwiseStridedXlaBGradB_den`, `EnetPoCG.convStridedXlaWGradB_den` and
`Mnv2PaperPoCG.convStridedXlaBGradB_den`.

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
-- *bias* VJP (§ C).
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
-- § C. Stem strided-conv bias (`sb`)
--
-- `conv2d_bias_differentiable` and `flatConvStride2_bias_grad_has_vjp` were RELOCATED to
-- `StridedConv.lean` (next to their weight peers) so the `convStridedBiasSgd` op's `den` in
-- `StableHLO` can reference the bias-VJP upstream; they are still in scope here by import.
-- ════════════════════════════════════════════════════════════════

/-- **Stem conv bias output, certified.** `sbⁿ = sb − lr·(spatial reduce)` denotes
    `sb − lr·(certified ∂(flatConvStride2)/∂sb · cotangent)`. -/
theorem mnv2_render_stem_convb_certified {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Vec (ic * (2 * h) * (2 * w)))
    (b : Vec oc) (dy : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (flatConvStride2_bias_grad_has_vjp W x).backward b dy o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * dy j := by
  rw [(flatConvStride2_bias_grad_has_vjp W x).correct]

end Proofs
