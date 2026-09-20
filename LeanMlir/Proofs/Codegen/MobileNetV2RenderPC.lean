import LeanMlir.Proofs.Codegen.StableHLO

/-! # The per-channel-BN MobileNetV2 stage vocabulary

Per-channel-BN mirrors (`bnPerChannelTensor3`: reduce over spatial `[2,3]`, `γ/β : Vec c`) of the
global-BN stage abbreviations `ivExpand` / … / `ivProject`: `ivExpandPC` / `ivDepthwisePC` /
`ivDepthwiseStridedPC` / `ivProjectPC`, and the two inverted-residual bodies composed from them
(`invresBodyPC`, `invresBodyStridedPC`). The per-example VJP tier (`MobileNetV2BackCertifiedTie`,
`MobileNetV2FullVJP`) is stated over these, and `MobileNetV2RenderPCEval` gives their
frozen-statistics twins.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Per-channel inverted-residual stage abbreviations
--   (per-channel-BN mirrors of ivExpand / ivDepthwise / ivProject)
-- ════════════════════════════════════════════════════════════════

/-- Expand stage, per-channel BN: `relu6 ∘ bnPC ∘ conv(1×1)`. -/
@[reducible] noncomputable def ivExpandPC {ic mid h w kHe kWe : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid) :
    Vec (ic * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelTensor3 mid h w εe γe βe ∘ flatConv We be

/-- Depthwise stage (stride-1), per-channel BN: `relu6 ∘ bnPC ∘ depthwise`. -/
@[reducible] noncomputable def ivDepthwisePC {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid) :
    Vec (mid * h * w) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelTensor3 mid h w εd γd βd ∘ depthwiseFlat Wd bd

/-- Depthwise stage (stride-2 downsample), per-channel BN: `relu6 ∘ bnPC ∘ depthwiseStrided`. -/
@[reducible] noncomputable def ivDepthwiseStridedPC {mid h w kHd kWd : Nat}
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid) :
    Vec (mid * (2 * h) * (2 * w)) → Vec (mid * h * w) :=
  relu6 (mid * h * w) ∘ bnPerChannelTensor3 mid h w εd γd βd ∘ depthwiseStride2FlatXla Wd bd

/-- Project (linear bottleneck) stage, per-channel BN: `bnPC ∘ conv(1×1)` (no relu6). -/
@[reducible] noncomputable def ivProjectPC {mid oc h w kHp kWp : Nat}
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (mid * h * w) → Vec (oc * h * w) :=
  bnPerChannelTensor3 oc h w εp γp βp ∘ flatConv Wp bp

/-- Inverted-residual body (stride-1), per-channel BN: `project ∘ depthwise ∘ expand`. -/
@[reducible] noncomputable def invresBodyPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  ivProjectPC (h := h) (w := w) Wp bp εp γp βp ∘
    (ivDepthwisePC (h := h) (w := w) Wd bd εd γd βd ∘
      ivExpandPC (h := h) (w := w) We be εe γe βe)

/-- Inverted-residual body (stride-2 downsample), per-channel BN: expand SAME (at `2h×2w`) →
    depthwise-strided → project. -/
@[reducible] noncomputable def invresBodyStridedPC {ic mid oc h w kHe kWe kHd kWd kHp kWp : Nat}
    (We : Kernel4 mid ic kHe kWe) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  ivProjectPC (h := h) (w := w) Wp bp εp γp βp ∘
    (ivDepthwiseStridedPC (h := h) (w := w) Wd bd εd γd βd ∘
      ivExpandPC (h := 2 * h) (w := 2 * w) We be εe γe βe)

end Proofs
