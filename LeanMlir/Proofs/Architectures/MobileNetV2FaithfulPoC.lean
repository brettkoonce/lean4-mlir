import LeanMlir.Proofs.Architectures.MobileNetV2ChainClose
import LeanMlir.Proofs.MobileNetV2RenderPC

/-! # PoC: the MobileNetV2 depthwise param updates, proof-tied to the certified SGD step

The mnv2 §1 fold (the depthwise half) — the four `depthwise{,Strided}{Weight,Bias}Sgd` ops
(`StableHLO.lean`) each denote the certified loss-descent step `θ − lr·(certified ∂/∂θ · c)`,
generic in the cotangent `c` the backward chain delivers. The depthwise peers of
`CifarPoC.convW_den`/`convB_den`, delegating to the `mnv2_render_depthwise*_certified` bridges
(`MobileNetV2Close.lean`).

* **Stride-2 (downsampling, blocks b1/b3/b5/b6):** `depthwiseStridedW_den` / `depthwiseStridedB_den`
  are one-line delegations to `mnv2_render_depthwiseW_strided_certified` /
  `mnv2_render_depthwiseb_strided_certified` (the strided VJP is already flat).
* **Stride-1 (skip blocks b2/b4):** `depthwiseB_den` likewise delegates to
  `mnv2_render_depthwiseb_certified`. The stride-1 **weight** is the only one needing a bridge: the
  stride-1 depthwise weight VJP is 3-index (`depthwise_weight_grad_has_vjp3`), and the emitted op's
  `den` carries it flat (`Tensor3.flatten (… .backward W (unflatten c))`), so `depthwiseW_den` first
  routes through `mnv2_render_depthwiseW_flat_certified` — the flat pdiv-Jacobian form via
  `hasVJP3_to_hasVJP.correct` (the same triple→flat reindex `flatConvStride2`'s weight VJP gets for
  free), modulo the `unflatten ∘ flatten = id` round-trip on `W`.

The expand/project 1×1 convs reuse `CifarPoC.convW_den`/`convB_den`; BN γ/β reuse
`CifarBnPoC.bnGamma_den`/`bnBeta_den`; the final dense reuses `Cifar8PoC.denseW_den`/`denseB_den` —
so the depthwise ops here are the only genuinely-new `den = certified` content for mnv2's §1 fold.

## Honest residual (same boundary as every prior fold)
* The cotangents `c` are free (∀ c); pinning each to the actual inverted-residual backward chain is
  the §1a tie (`MobileNetV2TiePoC`). Per-op `pretty` lexing + relu6 two-kink + ℝ → Float32.
-/

open Proofs Proofs.StableHLO

namespace Proofs.Mnv2PoC

open scoped BigOperators

/-- **Flat stride-1 depthwise weight render bridge.** The emitted op's flat weight grad
    `flatten W − lr·flatten((dwconv_weight_grad₃ b x).backward W (unflatten c))` equals the flat
    pdiv-Jacobian form. Via `hasVJP3_to_hasVJP.correct` (the triple→flat reindex), modulo
    `unflatten (flatten W) = W`. The stride-1 depthwise peer of `cnn_render_convW_certified`. -/
theorem mnv2_render_depthwiseW_flat_certified {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW)
    (cot : Vec (c*h*w)) (lr : ℝ) (idx : Fin (c*kH*kW)) :
    Tensor3.flatten W idx
        - lr * Tensor3.flatten
            ((depthwise_weight_grad_has_vjp3 b x).backward W (Tensor3.unflatten cot)) idx
      = Tensor3.flatten W idx - lr * ∑ j : Fin (c*h*w),
          pdiv (fun v' : Vec (c*kH*kW) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b x))
               (Tensor3.flatten W) idx j * cot j := by
  congr 1
  congr 1
  rw [← (hasVJP3_to_hasVJP (depthwise_weight_grad_has_vjp3 b x)).correct (Tensor3.flatten W) cot idx]
  simp only [hasVJP3_to_hasVJP, Tensor3.flatten, Tensor3.unflatten_flatten]

/-- **Stride-1 depthwise weight op = certified.** The `depthwiseWeightSgd` op denotes
    `flatten W − lr·(certified ∂(depthwiseConv2d)/∂W · c)` (flat pdiv form). The stride-1 depthwise
    peer of `CifarPoC.convW_den`. -/
theorem depthwiseW_den {c h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec c) (x : Tensor3 c h w)
    (W : DepthwiseKernel c kH kW) (cot : Vec (c*h*w)) (lr : ℝ) (idx : Fin (c*kH*kW)) :
    den (SHlo.depthwiseWeightSgd xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Tensor3.flatten W idx - lr * ∑ j : Fin (c*h*w),
          pdiv (fun v' : Vec (c*kH*kW) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b x))
               (Tensor3.flatten W) idx j * cot j := by
  show depthwiseWeightSgdDen b x W lr cot idx = _
  exact mnv2_render_depthwiseW_flat_certified b x W cot lr idx

/-- **Stride-1 depthwise bias op = certified.** Delegates to `mnv2_render_depthwiseb_certified`. -/
theorem depthwiseB_den {c h w kH kW : Nat}
    (bN lrStr cotN : String) (W : DepthwiseKernel c kH kW) (x : Tensor3 c h w)
    (b : Vec c) (cot : Vec (c*h*w)) (lr : ℝ) (o : Fin c) :
    den (SHlo.depthwiseBiasSgd bN lrStr W x b lr (.operand cotN cot)) o
      = b o - lr * ∑ j : Fin (c*h*w),
          pdiv (fun b' : Vec c => Tensor3.flatten (depthwiseConv2d W b' x)) b o j * cot j := by
  show depthwiseBiasSgdDen W x b lr cot o = _
  exact mnv2_render_depthwiseb_certified W x b cot lr o

/-- **Stride-2 depthwise weight op = certified.** The strided VJP is already flat; one-line
    delegation to `mnv2_render_depthwiseW_strided_certified`. -/
theorem depthwiseStridedW_den {c h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec c) (x : Vec (c*(2*h)*(2*w)))
    (W : DepthwiseKernel c kH kW) (cot : Vec (c*h*w)) (lr : ℝ) (idx : Fin (c*kH*kW)) :
    den (SHlo.depthwiseStridedWeightSgd xN wN lrStr b x W lr (.operand cotN cot)) idx
      = Tensor3.flatten W idx - lr * ∑ j : Fin (c*h*w),
          pdiv (fun v' : Vec (c*kH*kW) => depthwiseStride2Flat (Tensor3.unflatten v') b x)
               (Tensor3.flatten W) idx j * cot j := by
  show depthwiseStridedWeightSgdDen b x W lr cot idx = _
  exact mnv2_render_depthwiseW_strided_certified b x (Tensor3.flatten W) cot lr idx

/-- **Stride-2 depthwise bias op = certified.** Delegates to `mnv2_render_depthwiseb_strided_certified`. -/
theorem depthwiseStridedB_den {c h w kH kW : Nat}
    (bN lrStr cotN : String) (W : DepthwiseKernel c kH kW) (x : Vec (c*(2*h)*(2*w)))
    (b : Vec c) (cot : Vec (c*h*w)) (lr : ℝ) (o : Fin c) :
    den (SHlo.depthwiseStridedBiasSgd bN lrStr W x b lr (.operand cotN cot)) o
      = b o - lr * ∑ j : Fin (c*h*w),
          pdiv (fun b' : Vec c => depthwiseStride2Flat W b' x) b o j * cot j := by
  show depthwiseStridedBiasSgdDen W x b lr cot o = _
  exact mnv2_render_depthwiseb_strided_certified W x b cot lr o

end Proofs.Mnv2PoC
