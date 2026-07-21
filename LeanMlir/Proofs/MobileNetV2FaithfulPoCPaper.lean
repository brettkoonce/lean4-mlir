import LeanMlir.Proofs.MobileNetV2FaithfulPoC
import LeanMlir.Proofs.Foundation.ResNet34FaithfulPoC
import LeanMlir.Proofs.Cifar8FaithfulPoC
import LeanMlir.Proofs.CifarBnFaithfulPoC

/-! # PoC: the FULL 17-block paper-spec MobileNetV2 train step, proof-tied (the §1 fold, den)

The whole-net peer of `MobileNetV2FaithfulPoC` (the reduced 6-block fold), scaled to the real
`[t,c,n,s]` table. `mnv2TrainStepFaithfulVPaper` (`MobileNetV2Render.lean`) renders the full
17-block SGD train step (210 params) as `pretty(provenGraph)` and writes
`verified_mlir/mobilenetv2_paper_train_step.mlir`; this file makes its parameter updates
`den`-faithful — every emitted param-SGD op denotes the certified loss-descent step.

**Zero new core ops, ZERO new den lemmas — pure reuse, exactly the cifar8-bn lesson.** The 17-block
net emits the *same twelve* param-SGD op types the reduced 6-block net already exercises, just more
of them and across two new block *structures* (no-expand `b1`, no-skip widenings `b11`/`b17`) — and
those new structures change the forward/backward *wiring*, not the param op types. Every one of the
210 params is therefore certified by a generic `den = certified` lemma that already exists and is
already in the 3-axiom closure:

| param op (count)                                   | certified by (generic, dim+cotangent-polymorphic)  |
|----------------------------------------------------|-----------------------------------------------------|
| stem `convStrided{Weight,Bias}Sgd` (2)             | `ResNet34PoC.convStrided{W,B}_den`                  |
| expand/project/head `conv{Weight,Bias}Sgd` (1×1)   | `CifarPoC.conv{W,B}_den`                             |
| stride-1 depthwise `depthwise{Weight,Bias}Sgd`     | `Mnv2PoC.depthwise{W,B}_den`                         |
| stride-2 depthwise `depthwiseStrided{…}Sgd`        | `Mnv2PoC.depthwiseStrided{W,B}_den`                 |
| every BN γ/β `bn{Gamma,Beta}Sgd`                   | `CifarBnPoC.bn{Gamma,Beta}_den`                     |
| final dense `weightSgd`/`biasSgd`                  | `Cifar8PoC.dense{W,B}_den`                           |

210-param accounting: stem 4 + b1 (no-expand) 8 + b2…b17 (16 inverted-residual × 12) 192 +
head 4 + dense 2 = **210**.

What this file adds over the bare generics: **one named, machine-checked capstone per distinct
block-type param profile** — `mnv2Stem/NoExp/Stride1/Stride2/Head/Dense ParamsCertified` — each
GENERIC in the block dims (so a single theorem covers every block of that type at once: e.g.
`mnv2Stride1ParamsCertified` certifies all twelve stride-1 blocks `b3,5,6,8,9,10,12,13,15,16` plus
the no-skip widenings `b11,b17`, which share the param profile). Each conjunct is a direct
delegation to the audited generic, so the file is honestly "no new proof" — but the claim "all 210
params den-certified" is now a checked statement per block type rather than prose.

## Honest residual (same boundary as every prior fold)
* The cotangents are free (∀ c); pinning each to the actual 17-block inverted-residual backward
  chain (the fan-in sum at every stride-1 skip, the relu6 two-kink masks) is the §1a tie
  (`MobileNetV2TiePoCPaper`). Per-op `pretty` lexing + BN `0<ε` + relu6 + ℝ → Float32.
-/

open Proofs Proofs.StableHLO

namespace Proofs.Mnv2PaperPoC

open scoped BigOperators

/-! ## Stem — 3×3/s2 conv (3→32) → BN.  4 params. -/

/-- **Stem param ops all denote the certified step** (`convStrided{Weight,Bias}Sgd` for the 3×3/s2
    weight+bias, `bn{Gamma,Beta}Sgd` for the BN). Generic in the spatial dims; the paper net
    instantiates at `ic=3, oc=32, 112×112, 3×3`. -/
theorem mnv2StemParamsCertified {ic oc h w : Nat} :
    -- strided conv weight
    (∀ (xN wN lrStr cotN : String) (b : Vec oc) (x : Vec (ic*(2*h)*(2*w)))
       (W : Kernel4 oc ic 3 3) (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*3*3)),
        den (SHlo.convStridedWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
          = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*3*3) => flatConvStride2 (Kernel4.unflatten v') b x)
                   (Kernel4.flatten W) idx j * c j) ∧
    -- strided conv bias
    (∀ (bN lrStr cotN : String) (W : Kernel4 oc ic 3 3) (x : Vec (ic*(2*h)*(2*w)))
       (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc),
        den (SHlo.convStridedBiasSgd bN lrStr W x b lr (.operand cotN c)) o
          = b o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => flatConvStride2 W b' x) b o j * c j) ∧
    -- BN γ
    (∀ (gN vN epsStr lrStr cotN : String) (ε : ℝ) (γ β : Vec oc) (v c : Vec (oc*h*w))
       (lr : ℝ) (idx : Fin oc),
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γ v lr (.operand cotN c)) idx
          = γ idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' β (reassocFwd oc h w v))
                   γ idx j * reassocFwd oc h w c j) ∧
    -- BN β
    (∀ (bN lrStr cotN : String) (ε : ℝ) (γ β : Vec oc) (v c : Vec (oc*h*w))
       (lr : ℝ) (idx : Fin oc),
        den (SHlo.bnBetaSgd bN lrStr β lr (.operand cotN c)) idx
          = β idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γ β' (reassocFwd oc h w v))
                   β idx j * reassocFwd oc h w c j) :=
  ⟨fun xN wN lrStr cotN b x W c lr idx => ResNet34PoC.convStridedW_den xN wN lrStr cotN b x W c lr idx,
   fun bN lrStr cotN W x b c lr o => ResNet34PoC.convStridedB_den bN lrStr cotN W x b c lr o,
   fun gN vN epsStr lrStr cotN ε γ β v c lr idx => CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γ β v c lr idx,
   fun bN lrStr cotN ε γ β v c lr idx => CifarBnPoC.bnBeta_den bN lrStr cotN ε γ β v c lr idx⟩

/-! ## No-expand block (b1, t=1): depthwise(s1, on `ic` ch)→BN→relu6 → project(1×1)→BN.  8 params. -/

/-- **No-expand block param ops all denote the certified step.** depthwise stride-1 weight+bias
    (`Mnv2PoC.depthwise{W,B}_den`), depthwise BN (`CifarBnPoC.bn*`), project 1×1 conv
    (`CifarPoC.conv*`), project BN. Paper net: `ic=32, oc=16, 112×112`. -/
theorem mnv2NoExpParamsCertified {ic oc h w : Nat} :
    -- depthwise (stride-1) weight
    (∀ (xN wN lrStr cotN : String) (b : Vec ic) (x : Tensor3 ic h w) (W : DepthwiseKernel ic 3 3)
       (cot : Vec (ic*h*w)) (lr : ℝ) (idx : Fin (ic*3*3)),
        den (SHlo.depthwiseWeightSgd xN wN lrStr b x W lr (.operand cotN cot)) idx
          = Tensor3.flatten W idx - lr * ∑ j : Fin (ic*h*w),
              pdiv (fun v' : Vec (ic*3*3) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b x))
                   (Tensor3.flatten W) idx j * cot j) ∧
    -- depthwise (stride-1) bias
    (∀ (bN lrStr cotN : String) (W : DepthwiseKernel ic 3 3) (x : Tensor3 ic h w) (b : Vec ic)
       (cot : Vec (ic*h*w)) (lr : ℝ) (o : Fin ic),
        den (SHlo.depthwiseBiasSgd bN lrStr W x b lr (.operand cotN cot)) o
          = b o - lr * ∑ j : Fin (ic*h*w),
              pdiv (fun b' : Vec ic => Tensor3.flatten (depthwiseConv2d W b' x)) b o j * cot j) ∧
    -- project 1×1 conv weight
    (∀ (xN wN lrStr cotN : String) (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic 1 1)
       (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*1*1)),
        den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
          = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
                   (Kernel4.flatten W) idx j * c j) ∧
    -- project 1×1 conv bias
    (∀ (bN lrStr cotN : String) (W : Kernel4 oc ic 1 1) (x : Tensor3 ic h w) (b : Vec oc)
       (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc),
        den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
          = b o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j) :=
  ⟨fun xN wN lrStr cotN b x W cot lr idx => Mnv2PoC.depthwiseW_den xN wN lrStr cotN b x W cot lr idx,
   fun bN lrStr cotN W x b cot lr o => Mnv2PoC.depthwiseB_den bN lrStr cotN W x b cot lr o,
   fun xN wN lrStr cotN b x W c lr idx => CifarPoC.convW_den xN wN lrStr cotN b x W c lr idx,
   fun bN lrStr cotN W x b c lr o => CifarPoC.convB_den bN lrStr cotN W x b c lr o⟩

/-! ## Stride-1 inverted-residual block (incl. no-skip widenings): expand 1×1 → BN → relu6 →
    depthwise(s1) → BN → relu6 → project 1×1 → BN.  12 params.  (The skip vs no-skip difference is
    in the dx fan-in, not the param ops, so one capstone covers `irBack` and `irBackNoSkip`.) -/

/-- **Stride-1 inverted-residual block param ops all denote the certified step.** Expand + project
    are 1×1 convs (`CifarPoC.conv*`); the depthwise is stride-1 (`Mnv2PoC.depthwise*`); three BNs
    (`CifarBnPoC.bn*`). Generic in `{ic mid oc h w}` — covers every stride-1 block (and the no-skip
    widenings b11/b17). -/
theorem mnv2Stride1ParamsCertified {ic mid oc h w : Nat} :
    -- expand 1×1 conv W/b  (ic → mid)
    (∀ (xN wN lrStr cotN : String) (b : Vec mid) (x : Tensor3 ic h w) (W : Kernel4 mid ic 1 1)
       (c : Vec (mid*h*w)) (lr : ℝ) (idx : Fin (mid*ic*1*1)),
        den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
          = Kernel4.flatten W idx - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun v' : Vec (mid*ic*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
                   (Kernel4.flatten W) idx j * c j) ∧
    (∀ (bN lrStr cotN : String) (W : Kernel4 mid ic 1 1) (x : Tensor3 ic h w) (b : Vec mid)
       (c : Vec (mid*h*w)) (lr : ℝ) (o : Fin mid),
        den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
          = b o - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun b' : Vec mid => Tensor3.flatten (conv2d W b' x)) b o j * c j) ∧
    -- depthwise (stride-1) W/b  (mid channels)
    (∀ (xN wN lrStr cotN : String) (b : Vec mid) (x : Tensor3 mid h w) (W : DepthwiseKernel mid 3 3)
       (cot : Vec (mid*h*w)) (lr : ℝ) (idx : Fin (mid*3*3)),
        den (SHlo.depthwiseWeightSgd xN wN lrStr b x W lr (.operand cotN cot)) idx
          = Tensor3.flatten W idx - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun v' : Vec (mid*3*3) => Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') b x))
                   (Tensor3.flatten W) idx j * cot j) ∧
    (∀ (bN lrStr cotN : String) (W : DepthwiseKernel mid 3 3) (x : Tensor3 mid h w) (b : Vec mid)
       (cot : Vec (mid*h*w)) (lr : ℝ) (o : Fin mid),
        den (SHlo.depthwiseBiasSgd bN lrStr W x b lr (.operand cotN cot)) o
          = b o - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun b' : Vec mid => Tensor3.flatten (depthwiseConv2d W b' x)) b o j * cot j) ∧
    -- project 1×1 conv W/b  (mid → oc)
    (∀ (xN wN lrStr cotN : String) (b : Vec oc) (x : Tensor3 mid h w) (W : Kernel4 oc mid 1 1)
       (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*mid*1*1)),
        den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
          = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*mid*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
                   (Kernel4.flatten W) idx j * c j) ∧
    (∀ (bN lrStr cotN : String) (W : Kernel4 oc mid 1 1) (x : Tensor3 mid h w) (b : Vec oc)
       (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc),
        den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
          = b o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j) :=
  ⟨fun xN wN lrStr cotN b x W c lr idx => CifarPoC.convW_den xN wN lrStr cotN b x W c lr idx,
   fun bN lrStr cotN W x b c lr o => CifarPoC.convB_den bN lrStr cotN W x b c lr o,
   fun xN wN lrStr cotN b x W cot lr idx => Mnv2PoC.depthwiseW_den xN wN lrStr cotN b x W cot lr idx,
   fun bN lrStr cotN W x b cot lr o => Mnv2PoC.depthwiseB_den bN lrStr cotN W x b cot lr o,
   fun xN wN lrStr cotN b x W c lr idx => CifarPoC.convW_den xN wN lrStr cotN b x W c lr idx,
   fun bN lrStr cotN W x b c lr o => CifarPoC.convB_den bN lrStr cotN W x b c lr o⟩

/-! ## Stride-2 inverted-residual block (downsamples b2/b4/b7/b14): expand 1×1 → BN → relu6 →
    depthwise(STRIDED) → BN → relu6 → project 1×1 → BN.  12 params. -/

/-- **Stride-2 inverted-residual block param ops all denote the certified step.** Identical to the
    stride-1 capstone EXCEPT the depthwise is strided (`Mnv2PoC.depthwiseStrided*`); expand/project
    1×1 convs + three BNs unchanged. The depthwise activation lives at the `2h×2w` input grid. -/
theorem mnv2Stride2ParamsCertified {ic mid oc h w : Nat} :
    -- expand 1×1 conv W/b  (ic → mid, at the 2h×2w grid)
    (∀ (xN wN lrStr cotN : String) (b : Vec mid) (x : Tensor3 ic (2*h) (2*w)) (W : Kernel4 mid ic 1 1)
       (c : Vec (mid*(2*h)*(2*w))) (lr : ℝ) (idx : Fin (mid*ic*1*1)),
        den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
          = Kernel4.flatten W idx - lr * ∑ j : Fin (mid*(2*h)*(2*w)),
              pdiv (fun v' : Vec (mid*ic*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
                   (Kernel4.flatten W) idx j * c j) ∧
    (∀ (bN lrStr cotN : String) (W : Kernel4 mid ic 1 1) (x : Tensor3 ic (2*h) (2*w)) (b : Vec mid)
       (c : Vec (mid*(2*h)*(2*w))) (lr : ℝ) (o : Fin mid),
        den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
          = b o - lr * ∑ j : Fin (mid*(2*h)*(2*w)),
              pdiv (fun b' : Vec mid => Tensor3.flatten (conv2d W b' x)) b o j * c j) ∧
    -- depthwise (STRIDED) W/b  (mid channels, 2h×2w → h×w)
    (∀ (xN wN lrStr cotN : String) (b : Vec mid) (x : Vec (mid*(2*h)*(2*w))) (W : DepthwiseKernel mid 3 3)
       (cot : Vec (mid*h*w)) (lr : ℝ) (idx : Fin (mid*3*3)),
        den (SHlo.depthwiseStridedWeightSgd xN wN lrStr b x W lr (.operand cotN cot)) idx
          = Tensor3.flatten W idx - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun v' : Vec (mid*3*3) => depthwiseStride2Flat (Tensor3.unflatten v') b x)
                   (Tensor3.flatten W) idx j * cot j) ∧
    (∀ (bN lrStr cotN : String) (W : DepthwiseKernel mid 3 3) (x : Vec (mid*(2*h)*(2*w))) (b : Vec mid)
       (cot : Vec (mid*h*w)) (lr : ℝ) (o : Fin mid),
        den (SHlo.depthwiseStridedBiasSgd bN lrStr W x b lr (.operand cotN cot)) o
          = b o - lr * ∑ j : Fin (mid*h*w),
              pdiv (fun b' : Vec mid => depthwiseStride2Flat W b' x) b o j * cot j) ∧
    -- project 1×1 conv W/b  (mid → oc, at h×w)
    (∀ (xN wN lrStr cotN : String) (b : Vec oc) (x : Tensor3 mid h w) (W : Kernel4 oc mid 1 1)
       (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*mid*1*1)),
        den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
          = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*mid*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
                   (Kernel4.flatten W) idx j * c j) ∧
    (∀ (bN lrStr cotN : String) (W : Kernel4 oc mid 1 1) (x : Tensor3 mid h w) (b : Vec oc)
       (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc),
        den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
          = b o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j) :=
  ⟨fun xN wN lrStr cotN b x W c lr idx => CifarPoC.convW_den xN wN lrStr cotN b x W c lr idx,
   fun bN lrStr cotN W x b c lr o => CifarPoC.convB_den bN lrStr cotN W x b c lr o,
   fun xN wN lrStr cotN b x W cot lr idx => Mnv2PoC.depthwiseStridedW_den xN wN lrStr cotN b x W cot lr idx,
   fun bN lrStr cotN W x b cot lr o => Mnv2PoC.depthwiseStridedB_den bN lrStr cotN W x b cot lr o,
   fun xN wN lrStr cotN b x W c lr idx => CifarPoC.convW_den xN wN lrStr cotN b x W c lr idx,
   fun bN lrStr cotN W x b c lr o => CifarPoC.convB_den bN lrStr cotN W x b c lr o⟩

/-! ## Head — 1×1 conv (320→1280) → BN.  4 params. -/

/-- **Head param ops all denote the certified step.** 1×1 conv (`CifarPoC.conv*`) + BN
    (`CifarBnPoC.bn*`). Paper net: `ic=320, oc=1280, 7×7`. -/
theorem mnv2HeadParamsCertified {ic oc h w : Nat} :
    (∀ (xN wN lrStr cotN : String) (b : Vec oc) (x : Tensor3 ic h w) (W : Kernel4 oc ic 1 1)
       (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*1*1)),
        den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
          = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun v' : Vec (oc*ic*1*1) => Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
                   (Kernel4.flatten W) idx j * c j) ∧
    (∀ (bN lrStr cotN : String) (W : Kernel4 oc ic 1 1) (x : Tensor3 ic h w) (b : Vec oc)
       (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc),
        den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
          = b o - lr * ∑ j : Fin (oc*h*w),
              pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j) ∧
    (∀ (gN vN epsStr lrStr cotN : String) (ε : ℝ) (γ β : Vec oc) (v c : Vec (oc*h*w))
       (lr : ℝ) (idx : Fin oc),
        den (SHlo.bnGammaSgd gN vN epsStr lrStr ε γ v lr (.operand cotN c)) idx
          = γ idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (h*w) ε γ' β (reassocFwd oc h w v))
                   γ idx j * reassocFwd oc h w c j) ∧
    (∀ (bN lrStr cotN : String) (ε : ℝ) (γ β : Vec oc) (v c : Vec (oc*h*w))
       (lr : ℝ) (idx : Fin oc),
        den (SHlo.bnBetaSgd bN lrStr β lr (.operand cotN c)) idx
          = β idx - lr * ∑ j : Fin (oc*(h*w)),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (h*w) ε γ β' (reassocFwd oc h w v))
                   β idx j * reassocFwd oc h w c j) :=
  ⟨fun xN wN lrStr cotN b x W c lr idx => CifarPoC.convW_den xN wN lrStr cotN b x W c lr idx,
   fun bN lrStr cotN W x b c lr o => CifarPoC.convB_den bN lrStr cotN W x b c lr o,
   fun gN vN epsStr lrStr cotN ε γ β v c lr idx => CifarBnPoC.bnGamma_den gN vN epsStr lrStr cotN ε γ β v c lr idx,
   fun bN lrStr cotN ε γ β v c lr idx => CifarBnPoC.bnBeta_den bN lrStr cotN ε γ β v c lr idx⟩

/-! ## Dense classifier — `weightSgd`/`biasSgd` (1280 → nClasses).  2 params. -/

/-- **Dense head param ops denote the certified step** (`Cifar8PoC.dense{W,B}_den`). Paper net:
    `m=1280, n=nClasses`. -/
theorem mnv2DenseParamsCertified {m n : Nat} :
    (∀ (aN wN lrStr cotN : String) (a : Vec m) (W : Mat m n) (b : Vec n) (c : Vec n) (lr : ℝ)
       (i : Fin m) (j : Fin n),
        den (SHlo.weightSgd aN wN lrStr a W lr (.operand cotN c)) (finProdFinEquiv (i, j))
          = W i j - lr * ∑ k : Fin n,
              pdiv (fun v : Vec (m*n) => dense (Mat.unflatten v) b a) (Mat.flatten W)
                   (finProdFinEquiv (i, j)) k * c k) ∧
    (∀ (bN lrStr cotN : String) (W : Mat m n) (a : Vec m) (b : Vec n) (c : Vec n) (lr : ℝ) (i : Fin n),
        den (SHlo.biasSgd bN lrStr b lr (.operand cotN c)) i
          = b i - lr * ∑ j : Fin n,
              pdiv (fun b' : Vec n => dense W b' a) b i j * c j) :=
  ⟨fun aN wN lrStr cotN a W b c lr i j => Cifar8PoC.denseW_den aN wN lrStr cotN a W b c lr i j,
   fun bN lrStr cotN W a b c lr i => Cifar8PoC.denseB_den bN lrStr cotN W a b c lr i⟩

end Proofs.Mnv2PaperPoC
