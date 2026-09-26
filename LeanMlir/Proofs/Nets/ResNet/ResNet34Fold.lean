import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2Close
import LeanMlir.Proofs.Nets.Small.CifarFold
import LeanMlir.Proofs.Foundation.SgdNodes

/-! # The strided-convolution SGD ops denote the certified step

Two lemmas, `convStridedW_den` and `convStridedB_den`: any emitted strided-conv weight / bias SGD
op (`convStridedWeightSgd` / `convStridedBiasSgd`, StableHLO.lean) denotes
`θ − lr·(certified ∂flatConvStride2/∂θ · c)`, generic in the conv dims, the kernel size and the
cotangent `c`. They are the strided peers of `CifarPoC.convW_den` / `CifarPoC.convB_den`, and the
per-example fused-SGD ties use them (`ConvNeXtStepTie`). The batched, un-fused peer is
`ResNet34PoCB` in `GradNodesB`.

* **`convStridedWeightSgd`** emits the strided weight-grad text (zero-upsample the cotangent —
  the decimate-backward — then the SAME transpose-trick stride-1 weight-grad conv on the 2h×2w
  grid). Its `den` reduces (`rfl`) to `flatten W − lr·(flatConvStride2_weight_grad · c)`, the LHS
  of `mnv2_render_stem_convW_certified` — generic in `kH/kW`, so the *single* lemma below covers
  a 7×7 stem and every 3×3 strided conv.
* **`convStridedBiasSgd`** — the bias grad is stride-INDEPENDENT (`Σ_{batch,spatial} dy`), so it
  emits the same `reduce` op text as `convBiasSgd` (its `skel` aliases that op); only its `den`
  differs (the strided VJP), closing via `mnv2_render_stem_convb_certified`.

Both are one-line delegations to the generic strided bridge `mnv2_render_stem_conv{W,b}_certified`,
mirroring `CifarPoC.convW_den`'s delegation to `cnn_render_convW_certified`.
-/

-- History: this file was the fold of the per-example ResNet-34 SGD train step
-- (`verified_mlir/resnet34_train_step.mlir`, renderer `ResNet34Render.lean`); both were retired
-- when every ResNet-34 artifact moved to batch BatchNorm. The two lemmas are about op kinds and an
-- arbitrary cotangent, so they stay true and in use.

open Proofs Proofs.StableHLO

namespace Proofs.ResNet34PoC

/-! ## Strided convolutions — the two `den = certified` lemmas -/

/-- **Any emitted STRIDED conv weight op = certified.** Generic in the conv dims, the kernel size
    (covers the 7×7 stem AND every 3×3 downsample/projection) and the cotangent `c`: the
    `convStridedWeightSgd` op denotes `flatten W − lr·(certified ∂(flatConvStride2)/∂W · c)`, the
    emitted op's `den` reduced (`rfl`) to the LHS of the generic strided weight bridge. The strided
    peer of `CifarPoC.convW_den`. -/
theorem convStridedW_den {ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Vec (ic*(2*h)*(2*w)))
    (W : Kernel4 oc ic kH kW) (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*kH*kW)) :
    den (SHlo.convStridedWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
      = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun v' : Vec (oc*ic*kH*kW) => flatConvStride2 (Kernel4.unflatten v') b x)
               (Kernel4.flatten W) idx j * c j :=
  mnv2_render_stem_convW_certified b x (Kernel4.flatten W) c lr idx

/-- **Any emitted STRIDED conv bias op = certified.** The bias peer of `convStridedW_den`; the
    `convStridedBiasSgd` op (which emits the same `reduce` text as `convBiasSgd`) denotes
    `b − lr·(certified ∂(flatConvStride2)/∂b · c)`. -/
theorem convStridedB_den {ic oc h w kH kW : Nat}
    (bN lrStr cotN : String) (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w)))
    (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc) :
    den (SHlo.convStridedBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * c j :=
  mnv2_render_stem_convb_certified W x b c lr o

end Proofs.ResNet34PoC
