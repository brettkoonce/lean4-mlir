import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2Close
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2StagesPC
import LeanMlir.Proofs.Foundation.SgdNodes

/-! # The stride-1 depthwise param updates, proof-tied to the certified SGD step

The two stride-1 `depthwise{Weight,Bias}Sgd` ops (`StableHLO.lean`) each denote the certified
loss-descent step `θ − lr·(certified ∂/∂θ · c)`, generic in the cotangent `c` the backward chain
delivers — the depthwise peers of `CifarPoC.convW_den`/`convB_den`, delegating to the
`mnv2_render_depthwise*_certified` bridges (`MobileNetV2Close.lean`). Written for MobileNetV2's
per-example fold; ConvNeXt-T's 7×7 depthwise ties (`ConvNeXtStepTie`) are what use them now.

`depthwiseB_den` delegates to `mnv2_render_depthwiseb_certified`. The **weight** is the one needing a
bridge: the stride-1 depthwise weight VJP is 3-index (`depthwiseWeightGradHasVJP3`), and the
emitted op's `den` carries it flat (`Tensor3.flatten (… .backward W (unflatten c))`), so
`depthwiseW_den` first routes through `mnv2_render_depthwiseW_flat_certified` — the flat
pdiv-Jacobian form via `HasVJP3.toHasVJP.correct`, modulo the `unflatten ∘ flatten = id`
round-trip on `W`.

## Honest residual
* The cotangents `c` are free (∀ c); pinning each to the actual backward chain is the §1a tie.
  Per-op `pretty` lexing + ℝ → Float32.
-/

open Proofs Proofs.StableHLO

namespace Proofs.Mnv2PoC

open scoped BigOperators

/-- **Flat stride-1 depthwise weight render bridge.** The emitted op's flat weight grad
    `flatten W − lr·flatten((dwconv_weight_grad₃ b x).backward W (unflatten c))` equals the flat
    pdiv-Jacobian form. Via `HasVJP3.toHasVJP.correct` (the triple→flat reindex), modulo
    `unflatten (flatten W) = W`. The stride-1 depthwise peer of `cnn_render_convW_certified`. -/
theorem mnv2_render_depthwiseW_flat_certified {c h w kH kW : Nat}
    (b : Vec c) (x : Tensor3 c h w) (W : DepthwiseKernel c kH kW)
    (cot : Vec (c*h*w)) (lr : ℝ) (idx : Fin (c*kH*kW)) :
    Tensor3.flatten W idx
        - lr * Tensor3.flatten
            ((depthwiseWeightGradHasVJP3 b x).backward W (Tensor3.unflatten cot)) idx
      = Tensor3.flatten W idx - lr * ∑ j : Fin (c*h*w),
          pdiv (fun v' : Vec (c*kH*kW) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b x))
               (Tensor3.flatten W) idx j * cot j := by
  congr 1
  congr 1
  rw [← (HasVJP3.toHasVJP (depthwiseWeightGradHasVJP3 b x)).correct (Tensor3.flatten W) cot idx]
  simp only [HasVJP3.toHasVJP, Tensor3.flatten, Tensor3.unflatten_flatten]

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


end Proofs.Mnv2PoC
