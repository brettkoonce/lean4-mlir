import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullPaper
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2BackCertifiedTie

/-! # MobileNetV2's per-channel strided stem VJP, and the BN-positivity bundles the batched tier binds

What is left of the per-example paper-spec fold after its retirement on 2026-09-20:

* `convBnRelu6StridedPC_has_vjp_at` / `convBnRelu6StridedPC_differentiableAt` — the per-channel
  STRIDED stem stage `relu6 ∘ bnPC ∘ flatConvStride2Xla` at a smooth point (`MobileNetV2.lean`'s
  `convBnRelu6Strided_*` is the global-`bnForward` twin). `MobileNetV2WholeBackCertifiedTie.lean`'s
  stem leaf tie is stated against it.
* `IVPos` / `IVNoExpPos` — one BN-epsilon positivity bundle per bottleneck kind, over the `IVW` /
  `IVWNoExp` weight records of `MobileNetV2FullPaper.lean`. A BN epsilon's positivity does not know
  which axis the norm reduces, so the batched fold `mobilenetv2ForwardB_full_has_vjp_at`
  (`MobileNetV2FullBVJP.lean`) and the batched step tie bind these same bundles.

The fold this file used to state — stem + 17 bottlenecks + head at per-example BN, pointwise,
with 19 smoothness bundles and named running activations — was the forward of the retired SGD
artifact; its batch-BN successor is `MobileNetV2FullBVJP.lean`, same shape. -/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The per-channel strided stem stage
-- ════════════════════════════════════════════════════════════════

/-- Strided stem stage VJP, per-channel BN: `relu6 ∘ bnPC ∘ flatConvStride2Xla`. The per-channel
    twin of `MobileNetV2.lean`'s `convBnRelu6Strided_has_vjp_at`, in the `bnPerChannelTensor3`
    vocabulary the paper-spec net renders. -/
noncomputable def convBnRelu6StridedPC_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 6)) :
    HasVJPAt (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2Xla W b) v :=
  stage_has_vjp_at (flatConvStride2Xla W b) (bnPerChannelTensor3 oc h w ε γ β)
    (relu6 (oc * h * w)) v
    (flatConvStride2Xla_differentiable W b) (flatConvStride2Xla_has_vjp W b)
    (bnPerChannelTensor3_differentiable oc h w ε hε γ β)
    (bnPerChannelTensor3_has_vjp oc h w ε hε γ β)
    (relu6_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    (relu6_has_vjp_at (oc * h * w) _ h_smooth)

theorem convBnRelu6StridedPC_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 6)) :
    DifferentiableAt ℝ
      (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2Xla W b) v := by
  fun_prop (disch := assumption)

-- ════════════════════════════════════════════════════════════════
-- § Per-block BN-epsilon positivity bundles
-- ════════════════════════════════════════════════════════════════

/-- The three BN epsilons of a full bottleneck are positive. -/
structure IVPos {ic mid oc : Nat} (q : IVW ic mid oc) : Prop where
  he : 0 < q.eε
  hd : 0 < q.dε
  hp : 0 < q.pε

/-- The two BN epsilons of the t=1 (no-expand) bottleneck are positive. -/
structure IVNoExpPos {ic oc : Nat} (q : IVWNoExp ic oc) : Prop where
  hd : 0 < q.dε
  hp : 0 < q.pε

end Proofs
