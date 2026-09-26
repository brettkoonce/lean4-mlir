import LeanMlir.Proofs.Architectures.Depthwise

/-! # Convolution parameter-gradient bridges (the MNIST CNN train step's conv parameters)

The MNIST CNN (`conv → relu → conv → relu → maxpool → dense → relu → dense → relu → dense`)
train step has two kinds of parameters: the dense classifier head (whose grads reuse
`IR.weight_grad_bridge`/`IR.bias_grad_bridge`) and the **convolution kernels/biases**,
whose gradient is a *correlation*, not an outer product. This file supplies the conv
analogue of the dense bridges.

As for the MLP, the cotangent the backward chain delivers at each conv layer's output flows
through a backward graph — here the Tensor3-level `IR.Back3` (`convBackDenote`,
`maxPoolBackDenote`, with `IR.denote_subst3` the chain rule), exactly as the MLP used
`IR.Back` (`mlpCotOut0`/`mlpCotOut1`). Given that cotangent `c`, the conv kernel and
bias gradients (the transpose-trick `conv2dWeightGrad`/`conv2dBiasGrad`) are the
certified Jacobian of `conv2d` — as a function of the flattened kernel / of the bias —
contracted with `c`. Both bridges are the `.correct` field of the proven conv
parameter VJPs (`conv2dWeightGradHasVJP`/`conv2dBiasGradHasVJP`). The statements are
about those witnesses' backwards; no rendered text appears in them. The link from a
rendered conv gradient op to these witnesses is that op's `den` theorem (see `CnnFold`).

With the dense bridges in `IR` and the `Back3` cotangent chain, this gives a bridge for
every parameter of the CNN train step. (The SGD wrapping `θ − lr·∇` is identical to the
linear/MLP case.)

The last section adds the same bridges for the depthwise bias and the stride-2 conv kernel and
bias, which the MobileNetV2, ResNet-34 and ConvNeXt fold ties use.
-/

namespace Proofs

/-- **Conv weight-gradient bridge.** At any cotangent `c` at the conv layer's output
    (and any kernel point `v = Kernel4.flatten W`), the backward of
    `conv2dWeightGradHasVJP` (the transpose-trick kernel gradient) equals the certified Jacobian of `conv2d` viewed as a function of the flattened
    kernel, contracted with `c`. The convolution analogue of `IR.weight_grad_bridge`;
    it is the `.correct` field of `conv2dWeightGradHasVJP`. -/
theorem conv_weight_grad_bridge {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Tensor3 ic h w)
    (v : Vec (oc * ic * kH * kW)) (c : Vec (oc * h * w))
    (idx : Fin (oc * ic * kH * kW)) :
    (conv2dWeightGradHasVJP b x).backward v c idx
      = ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               v idx j * c j :=
  (conv2dWeightGradHasVJP b x).correct v c idx

/-- **Conv bias-gradient bridge.** Likewise the conv bias gradient (`db[o] = Σ
    spatial c`) is the certified Jacobian of `conv2d` wrt the bias, contracted with
    `c` — the `.correct` field of `conv2dBiasGradHasVJP`. -/
theorem conv_bias_grad_bridge {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc * h * w)) (o : Fin oc) :
    (conv2dBiasGradHasVJP W x).backward b c o
      = ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j :=
  (conv2dBiasGradHasVJP W x).correct b c o

-- ════════════════════════════════════════════════════════════════
-- § Closing the CNN render — the conv param outputs denote the certified gradients
--
-- The CNN train step (`cnnTrainStepFaithfulV` — conv→relu→conv→relu→maxpool→dense→relu→
-- dense→relu→dense) renders, per conv layer, `%dWᵢ = convWGrad` (the transpose-trick
-- kernel gradient) and `%Wᵢn = Wᵢ − lr·%dWᵢ`. These theorems are the denotation side:
-- each rendered conv SGD output equals `θ − lr·(certified conv Jacobian · the cotangent
-- the backward chain delivers)`, via the conv bridges. The conv analogue of the MLP
-- `mlp_render_W*_certified`; generic in the cotangent `c`, so one theorem covers both
-- conv layers (W₁, W₂). The three DENSE layers (W₃,W₄,W₅) reuse the M2 dense bridges
-- (`weight_grad_bridge`/`bias_grad_bridge`) exactly as the MLP render close does.
-- ════════════════════════════════════════════════════════════════

/-- **Conv weight SGD step — SGD form of `conv_weight_grad_bridge`.** At the flattened
    kernel `v`, `v − lr·(conv2dWeightGradHasVJP backward at c)` equals
    `v − lr·(certified ∂conv/∂kernel · c)`. A rewrite by the bridge; no rendered text
    appears in the statement. -/
theorem cnn_render_convW_certified {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Tensor3 ic h w)
    (v : Vec (oc * ic * kH * kW)) (c : Vec (oc * h * w)) (lr : ℝ)
    (idx : Fin (oc * ic * kH * kW)) :
    v idx - lr * (conv2dWeightGradHasVJP b x).backward v c idx
      = v idx - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               v idx j * c j := by
  rw [conv_weight_grad_bridge b x v c idx]

/-- **Conv bias SGD step — SGD form of `conv_bias_grad_bridge`.** Likewise
    `b − lr·(conv2dBiasGradHasVJP backward at c)` equals
    `b − lr·(certified ∂conv/∂bias · c)`. -/
theorem cnn_render_convb_certified {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (conv2dBiasGradHasVJP W x).backward b c o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j := by
  rw [conv_bias_grad_bridge W x b c o]

-- ════════════════════════════════════════════════════════════════
-- § Depthwise (stride-1) bias and stride-2 conv parameter bridges
--
-- The same `.correct`-field bridges for two more conv shapes: the depthwise bias
-- (`depthwiseBiasGradHasVJP`, `Depthwise.lean`, the spatial reduce `db[c] = Σ dy`) and the
-- stride-2 conv kernel and bias (`flatConvStride2WeightGradHasVJP` /
-- `flatConvStride2BiasGradHasVJP`, `StridedConv.lean`). `SgdNodes` builds the stride-1
-- depthwise op nodes on the bias bridge (`Mnv2PoC.depthwiseB_den`) and the strided conv nodes on the
-- two conv bridges (`ResNet34PoC.convStrided{W,B}_den`). The cotangent is a binder here too.
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
