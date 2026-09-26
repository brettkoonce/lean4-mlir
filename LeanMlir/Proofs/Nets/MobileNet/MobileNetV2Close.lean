import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.StridedConv
import LeanMlir.Proofs.Architectures.ConvGrad
import LeanMlir.Proofs.Architectures.PerChannelBNGrad

/-! # Depthwise-bias and stride-2 conv SGD bridges

Three SGD bridges, each saying that `θ − lr·(backward of the parameter's VJP)` equals
`θ − lr·(certified ∂/∂θ · c)` for a free cotangent `c`:

* `mnv2_render_depthwiseb_certified` — the stride-1 depthwise bias (`depthwiseBiasGradHasVJP`,
  the spatial reduce `db[c] = Σ dy`), through `mnv2_depthwise_bias_grad_bridge`.
* `mnv2_render_stem_convW_certified` — the stride-2 conv weight (`flatConvStride2WeightGradHasVJP`).
* `mnv2_render_stem_convb_certified` — the stride-2 conv bias (`flatConvStride2BiasGradHasVJP`).

The cotangent is a binder; nothing here ties it to a net's backward chain. `MobileNetV2Fold.lean`
builds the stride-1 depthwise op ties on the bias bridge (`ConvNeXtStepTie` uses them), and
`ResNet34Fold.lean` builds its strided-stem ties on the two conv bridges. The batched MobileNetV2
train step's strided parameters are tied in `mnv2_net_tiedB` through
`Mnv2PaperPoCG.depthwiseStridedXlaWGradB_den`, `Mnv2PaperPoCG.depthwiseStridedXlaBGradB_den`,
`EnetPoCG.convStridedXlaWGradB_den` and `Mnv2PaperPoCG.convStridedXlaBGradB_den`, not through this
file.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § A. Depthwise (stride-1) parameter bridges — the genuinely-new family
--
-- `depthwiseWeightGradHasVJP3` / `depthwiseBiasGradHasVJP` (Depthwise.lean) are the
-- proven, foundation-rule depthwise param VJPs. Their `.correct` fields are the bridges: the
-- rendered `dwconvWGrad` (per-channel transpose trick) / `convBiasGrad` (spatial reduce) equal
-- the certified Jacobian of `depthwiseConv2d` (as a function of W / of b) contracted with the
-- cotangent. The depthwise analogue of `conv_weight_grad_bridge` / `conv_bias_grad_bridge`.
-- ════════════════════════════════════════════════════════════════

/-- **Depthwise bias-gradient bridge.** Likewise the per-channel depthwise bias gradient
    (`db[c] = Σ_spatial dy`) is the certified Jacobian of `depthwiseConv2d` wrt the bias, contracted
    with `dy` — the `.correct` field of `depthwiseBiasGradHasVJP`. -/
theorem mnv2_depthwise_bias_grad_bridge {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w)
    (b : Vec c) (dy : Vec (c * h * w)) (cc : Fin c) :
    (depthwiseBiasGradHasVJP W x).backward b dy cc
      = ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b cc j * dy j :=
  (depthwiseBiasGradHasVJP W x).correct b dy cc

/-- **Depthwise bias output, certified.** Likewise `bⁿ = b − lr·(spatial reduce)` denotes
    `b − lr·(certified ∂(depthwiseConv2d)/∂b · cotangent)`. -/
theorem mnv2_render_depthwiseb_certified {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w)
    (b : Vec c) (dy : Vec (c * h * w)) (lr : ℝ) (cc : Fin c) :
    b cc - lr * (depthwiseBiasGradHasVJP W x).backward b dy cc
      = b cc - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b cc j * dy j := by
  rw [mnv2_depthwise_bias_grad_bridge]

-- ════════════════════════════════════════════════════════════════
-- § B. Stem strided 3×3 conv — weight reuses ch6, bias is new
--
-- The stem (`conv3WGradStrided`) reuses `flatConvStride2WeightGradHasVJP` (StridedConv.lean)
-- for the kernel; the SGD wrapper is the only new content. The stem bias needs a strided-conv
-- *bias* VJP (§ C).
-- ════════════════════════════════════════════════════════════════

/-- **Stem conv weight output, certified.** `sWⁿ = sW − lr·(strided transpose-trick grad)` denotes
    `sW − lr·(certified ∂(flatConvStride2)/∂sW · cotangent)`, via `flatConvStride2WeightGradHasVJP`
    (the ch6 strided conv weight VJP). -/
theorem mnv2_render_stem_convW_certified {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w)))
    (v : Vec (oc * ic * kH * kW)) (dy : Vec (oc * h * w)) (lr : ℝ)
    (i : Fin (oc * ic * kH * kW)) :
    v i - lr * (flatConvStride2WeightGradHasVJP b x).backward v dy i
      = v i - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) => flatConvStride2 (Kernel4.unflatten v') b x)
            v i j * dy j := by
  rw [flatConvStride2WeightGradHasVJP_correct]

-- ════════════════════════════════════════════════════════════════
-- § C. Stem strided-conv bias (`sb`)
--
-- `conv2d_bias_differentiable` and `flatConvStride2BiasGradHasVJP` were RELOCATED to
-- `StridedConv.lean` (next to their weight peers) so the `convStridedBiasSgd` op's `den` in
-- `StableHLO` can reference the bias-VJP upstream; they are still in scope here by import.
-- ════════════════════════════════════════════════════════════════

/-- **Stem conv bias output, certified.** `sbⁿ = sb − lr·(spatial reduce)` denotes
    `sb − lr·(certified ∂(flatConvStride2)/∂sb · cotangent)`. -/
theorem mnv2_render_stem_convb_certified {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Vec (ic * (2 * h) * (2 * w)))
    (b : Vec oc) (dy : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (flatConvStride2BiasGradHasVJP W x).backward b dy o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * dy j := by
  rw [(flatConvStride2BiasGradHasVJP W x).correct]

end Proofs
