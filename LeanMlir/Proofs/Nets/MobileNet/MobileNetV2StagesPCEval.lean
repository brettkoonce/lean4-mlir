import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.PerChannelBN

/-! # MobileNetV2 — the inference (frozen-statistics) stage vocabulary

The eval twins of `MobileNetV2StagesPC.lean`'s per-channel stage abbreviations: `ivExpandPCEval` /
`ivDepthwisePCEval` / `ivDepthwiseStridedPCEval` / `ivProjectPCEval` and the two bodies
(`invresBodyPCEval`, `invresBodyStridedPCEval`), every BN site at `bnPerChannelEvalTensor3`
(frozen running mean and variance). `MobileNetV2FullPaperEval` builds the shipped seventeen-block
eval forward and its graph from them. 3-axiom clean.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § The inference stage abbreviations (eval mirrors of ivExpandPC / … / ivProjectPC)
-- ════════════════════════════════════════════════════════════════

/-- Expand stage at inference: `relu6 ∘ bnEval ∘ conv(1×1)`. -/
@[reducible] noncomputable def ivExpandPCEval {ic mid h w kHe kWe : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe μe ve : Vec mid) :
    Vec (ic * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelEvalTensor3 mid h w εe γe βe μe ve ∘ flatConv We be

/-- Depthwise stage (stride-1) at inference. -/
@[reducible] noncomputable def ivDepthwisePCEval {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd μd vd : Vec mid) :
    Vec (mid * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelEvalTensor3 mid h w εd γd βd μd vd ∘ depthwiseFlat Wd bd

/-- Depthwise stage (stride-2 downsample) at inference. -/
@[reducible] noncomputable def ivDepthwiseStridedPCEval {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd μd vd : Vec mid) :
    Vec (mid * (2 * h) * (2 * w)) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelEvalTensor3 mid h w εd γd βd μd vd ∘
    depthwiseStride2FlatXla Wd bd

/-- Project (linear bottleneck) stage at inference — no relu6. -/
@[reducible] noncomputable def ivProjectPCEval {mid oc h w kHp kWp : Nat}
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp μp vp : Vec oc) :
    Vec (mid * h * w) → Vec (oc * h * w) :=
  bnPerChannelEvalTensor3 oc h w εp γp βp μp vp ∘ flatConv Wp bp

/-- Inverted-residual body (stride-1) at inference, at one shared ε. -/
@[reducible] noncomputable def invresBodyPCEval {ic mid oc h w : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  ivProjectPCEval (h := h) (w := w) Wp bp ε γp βp μp vp ∘
    (ivDepthwisePCEval (h := h) (w := w) Wd bd ε γd βd μd vd ∘
      ivExpandPCEval (h := h) (w := w) We be ε γe βe μe ve)

/-- Inverted-residual body (stride-2 downsample) at inference: expand at `2h×2w`, then the
    strided depthwise, then project. -/
@[reducible] noncomputable def invresBodyStridedPCEval {ic mid oc h w : Nat} (ε : ℝ)
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (γe βe μe ve : Vec mid)
    (Wd : DepthwiseKernel mid 3 3) (bd : Vec mid) (γd βd μd vd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (γp βp μp vp : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  ivProjectPCEval (h := h) (w := w) Wp bp ε γp βp μp vp ∘
    (ivDepthwiseStridedPCEval (h := h) (w := w) Wd bd ε γd βd μd vd ∘
      ivExpandPCEval (h := 2 * h) (w := 2 * w) We be ε γe βe μe ve)

end Proofs
