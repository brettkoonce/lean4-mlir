import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2BackCertifiedTie
import LeanMlir.Proofs.Architectures.ConvBackCertifiedTie
import LeanMlir.Proofs.Nets.MobileNet.MobileNetBackChains
import LeanMlir.Proofs.Codegen.MobileNetV2RenderPC
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullVJP

/-! # MobileNetV2's strided stem stage and its stem / head leaf ties

Two pieces MobileNetV2's whole-net ties stand on (the per-example paper tie first, retired 2026-09-19):

1. `convStridedBnRelu6PC_has_vjp_at` — the STEM stage's certified VJP, `relu6 ∘ bnPC ∘
   flatConvStride2Xla`. The repo had the strided-conv-with-**relu** peer (the per-example r34 stem, since retired) and
   the **non-strided** relu6 peer (`convBnRelu6PC_has_vjp_at`); this is the missing corner.
2. `convStridedBnRelu6PCBack_eq_vjp_backward` / `convBnRelu6PCBack_eq_vjp_backward` — the stem and
   head leaf ties, both closing on one conv-leaf rewrite (`flatConvStride2XlaBack_eq_vjp_backward` /
   `convFlatBack_eq_vjp_backward`) and then `rfl`: relu6's certified backward IS
   `reluMaskBack (0 < · ∧ · < 6)`, and the pinned BN-back is definitionally the certified one.

⚠ Both are SMOOTH-POINT statements, as every `HasVJPAt` in this cone is: the stem's and head's
post-BN clamp windows (`≠ 0 ∧ ≠ 6`) are hypotheses.
-/

namespace Proofs


/-- **conv(stride-2) → per-channel-BN → relu6 VJP at a smooth point** — MobileNetV2's STEM,
    i.e. `MobileNetV2FullVJP`'s `convBnRelu6StridedPC_has_vjp_at`. -/
noncomputable def convStridedBnRelu6PC_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 6)) :
    HasVJPAt (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β
      ∘ flatConvStride2Xla (h := h) (w := w) W b) v :=
  convBnRelu6StridedPC_has_vjp_at W b ε γ β hε v h_smooth

theorem convStridedBnRelu6PC_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 6)) :
    DifferentiableAt ℝ (relu6 (oc * h * w) ∘ bnPerChannelTensor3 oc h w ε γ β
      ∘ flatConvStride2Xla (h := h) (w := w) W b) v :=
  convBnRelu6StridedPC_differentiableAt W b ε γ β hε v h_smooth

/-- **The STEM tie.** -/
theorem convStridedBnRelu6PCBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) k ≠ 6)) :
    flatConvStride2XlaBack (h := h) (w := w) W
      ∘ (bnPerChannelTensor3_has_vjp oc h w ε hε γ β).backward (flatConvStride2Xla W b v)
      ∘ reluMaskBack (fun i => 0 < bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) i ∧
          bnPerChannelTensor3 oc h w ε γ β (flatConvStride2Xla W b v) i < 6)
      = (convStridedBnRelu6PC_has_vjp_at W b ε γ β hε v h_smooth).backward := by
  funext dy
  rw [flatConvStride2XlaBack_eq_vjp_backward hkH hkW W b v]
  rfl

/-- **The HEAD tie.** -/
theorem convBnRelu6PCBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * h * w))
    (h_smooth : ∀ k, (bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 0 ∧
                       bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) k ≠ 6)) :
    convFlatBack (h := h) (w := w) W
      ∘ (bnPerChannelTensor3_has_vjp oc h w ε hε γ β).backward (flatConv W b v)
      ∘ reluMaskBack (fun i => 0 < bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) i ∧
          bnPerChannelTensor3 oc h w ε γ β (flatConv W b v) i < 6)
      = (convBnRelu6PC_has_vjp_at W b ε γ β hε v h_smooth).backward := by
  funext dy
  rw [convFlatBack_eq_vjp_backward (W := W) (b := b) (x := v) hkH hkW]
  rfl






end Proofs
