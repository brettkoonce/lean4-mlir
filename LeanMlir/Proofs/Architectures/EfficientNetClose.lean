import LeanMlir.Proofs.Architectures.MobileNetV2Close

/-! # Closing the EfficientNet-B0 render — the parameter-gradient close (another FREE close)

`planning/mobilenetv2_close.md` Item C, applied to EfficientNet-B0 (`tests/TestEfficientNetTrain.lean`,
262 params, the real `[t,c,n,s,k]` B0 spec — 16 MBConv layers with squeeze-excite + swish + batch
norm, 3×3/5×5 depthwise). Like ResNet-34, **every parameter family is already certified by an existing
bridge** — even the two genuinely-new structures (squeeze-excite, true batch-norm) introduce no new
*parameter*-gradient bridge:

| family (render SSA)                              | forward fn            | certified by                                   |
|--------------------------------------------------|-----------------------|------------------------------------------------|
| 1×1 conv W/b (expand `eW`, project `pW`, head `hW`) | `conv2d`           | `cnn_render_conv{W,b}_certified` (M3, **reuse**) |
| stem 3×3 stride-2 conv W/b (`sW`)                | `flatConvStride2`     | `mnv2_render_stem_conv{W,b}_certified` (**reuse**) |
| depthwise **3×3** W/b (`dW`, stride 1/2)         | `depthwiseConv2d` / `depthwiseStride2Flat` | `mnv2_render_depthwise{W,b}[_strided]_certified` (**reuse**) |
| depthwise **5×5** W/b (`dW`, stride 1/2)         | same, `kH=kW=5`       | the same depthwise bridges (kernel-general) — pinned below |
| **SE** squeeze/excite dense `zW1/zb1/zW2/zb2`    | `dense` (`dot_general`) | `weight_grad_bridge` / `bias_grad_bridge` (M2, **reuse**) |
| batch-norm γ/β (every `g*`/`bt*`)                | `bnBatchTensor4`      | `cifar_bn_render_{gamma,beta}_certified` at `m=N·h·w` (**reuse**) |
| dense head `Wd`/`bd`                             | matmul / +bias        | M2 `weight/bias_grad_bridge` (**reuse**)        |
| swish / sigmoid / SE channel-scale / GAP        | —                     | no parameters                                  |

Two facts make this a free close despite the new structure:
* **Batch-norm γ/β = per-channel BN γ/β over the merged batch+spatial axis.** `bnBatchTensor4 =
  bnchwBack ∘ bnPerChannelFlat oc (N·h·w) ε γ β ∘ bnchwFwd` (PerChannelBN.lean): true batch-norm is
  per-channel BN over `m = N·h·w` cells, with γ/β-independent layout transposes. γ/β enter affinely,
  so the param grad is exactly `cifar_bn_render_{gamma,beta}_certified` at `m = N·h·w` — the render's
  `dγ = Σ_{[0,2,3]} dy·x̂`, `dβ = Σ_{[0,2,3]} dy`. (The hard *input*-VJP `bnBatchTensor4_grad_input`
  couples the batch; the *param* grad does not.)
* **Squeeze-excite carries no new param family.** The squeeze/excite are `dot_general` (dense), so
  their W/b reuse the M2 dense bridges; the channel-scale, sigmoid gate and swish carry no parameters.

So this file adds **no new VJP**. Its content: pin the depthwise bridges to the new **5×5** kernel
(no prior net used 5×5), and record the batch-norm γ/β = per-channel-at-`N·h·w` reuse as auditable
theorems. The SE / 1×1-conv / stem / dense families are verbatim reuse (documented above). The genuinely-
new EfficientNet work — the SE structure in the forward graph (Item A), the structured render with SE +
swish/sigmoid + batch-norm backward (Item B), and the SE-gate cotangent chain (Item D) — is separate.
3-axiom clean by inheritance.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § 5×5 depthwise (stages 3, 5, 6 — the kernel size no prior net exercised)
-- ════════════════════════════════════════════════════════════════

/-- **5×5 depthwise weight output, certified (stride-1).** The generic depthwise weight bridge at
    `kH=kW=5`; covers the stride-1 5×5 MBConv depthwise (e.g. stage 5). -/
theorem enet_render_dw5W_certified {c h w : Nat}
    (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c 5 5) (dy : Tensor3 c h w) (lr : ℝ)
    (ci : Fin c) (hi : Fin 5) (wi : Fin 5) :
    W ci hi wi - lr * (depthwise_weight_grad_has_vjp3 b x).backward W dy ci hi wi
      = W ci hi wi - lr * ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
          pdiv3 (fun W' : DepthwiseKernel c 5 5 => depthwiseConv2d W' b x) W ci hi wi co ho wo
            * dy co ho wo :=
  mnv2_render_depthwiseW_certified b x W dy lr ci hi wi

/-- **5×5 depthwise bias output, certified (stride-1).** -/
theorem enet_render_dw5b_certified {c h w : Nat}
    (W : DepthwiseKernel c 5 5) (x : Tensor3 c h w) (b : Vec c) (dy : Vec (c * h * w)) (lr : ℝ)
    (cc : Fin c) :
    b cc - lr * (depthwise_bias_grad_has_vjp W x).backward b dy cc
      = b cc - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b cc j * dy j :=
  mnv2_render_depthwiseb_certified W x b dy lr cc

/-- **5×5 depthwise weight output, certified (stride-2 downsampling).** The strided depthwise weight
    bridge at `kH=kW=5`; covers the stride-2 5×5 MBConv depthwise (stages 3, 6). -/
theorem enet_render_dw5W_strided_certified {c h w : Nat}
    (b : Vec c) (x : Vec (c * (2 * h) * (2 * w))) (v : Vec (c * 5 * 5)) (dy : Vec (c * h * w)) (lr : ℝ)
    (i : Fin (c * 5 * 5)) :
    v i - lr * (depthwiseStride2_weight_grad_has_vjp b x).backward v dy i
      = v i - lr * ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * 5 * 5) =>
            depthwiseStride2Flat (Tensor3.unflatten v' : DepthwiseKernel c 5 5) b x) v i j * dy j :=
  mnv2_render_depthwiseW_strided_certified b x v dy lr i

/-- **5×5 depthwise bias output, certified (stride-2 downsampling).** -/
theorem enet_render_dw5b_strided_certified {c h w : Nat}
    (W : DepthwiseKernel c 5 5) (x : Vec (c * (2 * h) * (2 * w))) (b : Vec c) (dy : Vec (c * h * w))
    (lr : ℝ) (o : Fin c) :
    b o - lr * (depthwiseStride2_bias_grad_has_vjp W x).backward b dy o
      = b o - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => depthwiseStride2Flat W b' x) b o j * dy j :=
  mnv2_render_depthwiseb_strided_certified W x b dy lr o

-- ════════════════════════════════════════════════════════════════
-- § Batch-norm γ/β = per-channel BN γ/β over the merged batch+spatial axis (m = N·h·w)
-- ════════════════════════════════════════════════════════════════

/-- **Batch-norm γ output, certified.** EfficientNet's `bnBatch` reduces statistics over `[0,2,3]`
    (batch + spatial); since `bnBatchTensor4 = bnchwBack ∘ bnPerChannelFlat oc (N·h·w) ε γ β ∘ bnchwFwd`,
    its per-channel `dγ_c = Σ_{[0,2,3]} dy·x̂` is exactly the per-channel BN γ-gradient over `m = N·h·w`
    cells. So `cifar_bn_render_gamma_certified` at `m = N·h·w` certifies it (γ enters affinely — no batch
    coupling in the *param* grad). -/
theorem enet_render_bngamma_certified (oc N h w : Nat) (ε : ℝ) (γ β : Vec oc)
    (v dy : Vec (oc * (N * h * w))) (lr : ℝ) (idx : Fin oc) :
    γ idx - lr * bnPerChannel_grad_gamma oc (N * h * w) ε v dy idx
      = γ idx - lr * ∑ j : Fin (oc * (N * h * w)),
          pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (N * h * w) ε γ' β v) γ idx j * dy j :=
  cifar_bn_render_gamma_certified oc (N * h * w) ε γ β v dy lr idx

/-- **Batch-norm β output, certified.** Likewise `dβ_c = Σ_{[0,2,3]} dy` is the per-channel BN
    β-gradient over `m = N·h·w`. -/
theorem enet_render_bnbeta_certified (oc N h w : Nat) (ε : ℝ) (γ β : Vec oc)
    (v dy : Vec (oc * (N * h * w))) (lr : ℝ) (idx : Fin oc) :
    β idx - lr * bnPerChannel_grad_beta oc (N * h * w) dy idx
      = β idx - lr * ∑ j : Fin (oc * (N * h * w)),
          pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * h * w) ε γ β' v) β idx j * dy j :=
  cifar_bn_render_beta_certified oc (N * h * w) ε γ β v dy lr idx

-- The 1×1 conv (expand/project/head), the strided 3×3 stem, the 3×3 depthwise (stride 1/2), the SE
-- squeeze/excite dense (`dot_general`), and the dense head are covered VERBATIM by the existing
-- `cnn_render_conv{W,b}_certified` / `mnv2_render_stem_conv{W,b}_certified` /
-- `mnv2_render_depthwise{W,b}[_strided]_certified` / M2 `weight_grad_bridge`/`bias_grad_bridge` at the
-- EfficientNet shapes — no kernel size to pin (3×3/1×1 already exercised; dense is dense). With the
-- 5×5 depthwise and batch-norm γ/β above, every EfficientNet-B0 train-step parameter output is
-- certified `θ − lr·(certified Jacobian · the cotangent)`.

end Proofs
