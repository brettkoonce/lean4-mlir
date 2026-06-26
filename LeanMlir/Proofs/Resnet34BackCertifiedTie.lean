import LeanMlir.Proofs.ResNet34RenderPC
import LeanMlir.Proofs.CifarCNN
import LeanMlir.Proofs.Resnet34BackFloatBridge
import LeanMlir.Proofs.EfficientNetChainClose

/-! # §B: the r34 identity-block backward float bridge targets the CERTIFIED VJP

The A3 backward float bridges prove **deployed-float ≈ a hand-assembled reverse-mode
transcription** (`r34IdBlockBack`). This file closes the §B integrity question for the r34
**identity block**: that transcription IS the certified input-gradient VJP.

The key design choice that makes this `b1`-free (no batched↔non-batched `batchMap`
reconciliation): the float-bridge `r34IdBlockBack` is the reverse of `rblkPC` — the
**per-channel-BN, non-batched** identity block (`ResNet34RenderPC`). So the right
certified target is a VJP of `rblkPC` in the *same vocabulary*, NOT the batched true-BN
`r34BasicBlockB_has_vjp_at` (`ResNet34BackB0`). That certified object did not exist, so we
build it here (`rblkPC_has_vjp_at`), then tie.

Three pieces:
1. `convFlatBack_eq_vjp_backward` — the conv **leaf** tie: the float-bridge `convFlatBack`
   (reversed-kernel conv) IS the certified conv input-VJP, via the general odd-kernel
   `IR.convBackDenote_eq_input_grad_formula`.
2. `rblkPC_has_vjp_at` — the certified per-channel-BN identity-block VJP, assembled from the
   per-op VJPs (`convBnReluPC_has_vjp_at` + `bnPerChannelTensor3_has_vjp` + `residual_has_vjp_at`),
   mirroring the scalar-BN `resblock_has_vjp_at`.
3. `r34IdBlockBack_eq_rblkPC_vjp` — **the tie**: `r34IdBlockBack` with its abstract BN-backs
   pinned to the certified per-channel-BN backwards and its ReLU masks pinned to the actual
   pre-activation signs equals `(rblkPC_has_vjp_at …).backward`. Closes by rewriting the two
   conv leaves; everything else (residual fan-in, the `∘`-reversal, the relu masks, the pinned
   BN-backs) matches definitionally.

The honest upgrade: for the r34 identity block, the float bridge's closeness is now closeness
to **the certified gradient**, not merely to a hand-map. (Remaining §B: the down-block, the
stem/GAP/dense endpoints, and the whole-net fold — the latter still gated by the fact that the
certified whole-net VJP is parametric / only concretely instantiated at toy dims.)
-/

namespace Proofs

open Classical

/-- **Conv input-VJP leaf tie.** The float-bridge `convFlatBack W` (= reversed-kernel forward
    conv) IS the certified conv input-VJP `(flatConv_has_vjp W b).backward x` (conv is linear,
    so the saved activation `x` is ignored), for odd kernels. Routes through the general
    `IR.convBackDenote_eq_input_grad_formula`; the leaf the §B block tie reuses (×2). -/
theorem convFlatBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * h * w)) :
    convFlatBack (h := h) (w := w) W = (flatConv_has_vjp W b).backward x := by
  funext dy
  simp only [convFlatBack, flatConv, flatConv_has_vjp, hasVJP3_to_hasVJP, conv2d_has_vjp3]
  rw [IR.convBackDenote_eq_input_grad_formula hkH hkW W (Tensor3.unflatten dy)]
  rfl

/-- **Certified VJP of the per-channel-BN identity basic block `rblkPC`** (non-batched).
    `relu ∘ residual(F)` with `F = (bnPC₂∘conv₂) ∘ (relu∘bnPC₁∘conv₁)`. The same-vocabulary
    certified target for the float-bridge `r34IdBlockBack` — no batched/`batchMap`
    reconciliation. Mirrors `resblock_has_vjp_at` (scalar BN) with `bnPerChannelTensor3` for
    `bnForward`, reusing `convBnReluPC_has_vjp_at` for stage 1. -/
noncomputable def rblkPC_has_vjp_at {c h w : Nat}
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ : Vec c)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (c * h * w))
    (h_smooth₁ : ∀ k, bnPerChannelTensor3 c h w ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnPerChannelTensor3 c h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (c * h * w) ∘ bnPerChannelTensor3 c h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) v k + v k ≠ 0) :
    HasVJPAt (rblkPC (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂) v := by
  set stage1 := relu (c * h * w) ∘ bnPerChannelTensor3 c h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁ with hs1
  have hstage1_vjp : HasVJPAt stage1 v := convBnReluPC_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  have hstage1_diff : DifferentiableAt ℝ stage1 v :=
    convBnReluPC_differentiableAt W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  set stage2 := bnPerChannelTensor3 c h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂ with hs2
  have hstage2_diff : DifferentiableAt ℝ stage2 (stage1 v) :=
    ((bnPerChannelTensor3_differentiable c h w ε₂ hε₂ γ₂ β₂).comp (flatConv_differentiable W₂ b₂))
      (stage1 v)
  have hstage2_vjp : HasVJPAt stage2 (stage1 v) :=
    vjp_comp_at (flatConv W₂ b₂) (bnPerChannelTensor3 c h w ε₂ γ₂ β₂) (stage1 v)
      (flatConv_differentiable W₂ b₂ _)
      ((bnPerChannelTensor3_differentiable c h w ε₂ hε₂ γ₂ β₂) _)
      ((hasVJP3_to_hasVJP (conv2d_has_vjp3 W₂ b₂)).toHasVJPAt _)
      ((bnPerChannelTensor3_has_vjp c h w ε₂ hε₂ γ₂ β₂).toHasVJPAt _)
  set F := stage2 ∘ stage1 with hF
  have hF_vjp : HasVJPAt F v :=
    vjp_comp_at stage1 stage2 v hstage1_diff hstage2_diff hstage1_vjp hstage2_vjp
  have hF_diff : DifferentiableAt ℝ F v := DifferentiableAt.comp v hstage2_diff hstage1_diff
  have hres_vjp : HasVJPAt (residual F) v := residual_has_vjp_at F v hF_diff hF_vjp
  have hres_diff : DifferentiableAt ℝ (residual F) v := by
    show DifferentiableAt ℝ (biPath F (fun x => x)) v
    exact DifferentiableAt.add hF_diff differentiable_id.differentiableAt
  have h_smooth_res : ∀ k, residual F v k ≠ 0 := h_smooth_out
  exact vjp_comp_at (residual F) (relu (c * h * w)) v hres_diff
    (relu_differentiableAt_of_smooth (c * h * w) _ h_smooth_res) hres_vjp
    (relu_has_vjp_at (c * h * w) _ h_smooth_res)

/-- **The §B identity-block tie: float-bridge backward = certified VJP.** `r34IdBlockBack`,
    with its abstract BN-backs pinned to the certified per-channel-BN backwards
    (`bnPerChannelTensor3_has_vjp.backward` at the respective conv outputs) and its ReLU masks
    pinned to the actual pre-activation signs, equals `(rblkPC_has_vjp_at …).backward`.

    Both sides are `fun dy ↦ bodyBack(mask dy) + mask dy` (residual fan-in over the outer-relu
    mask). The bodies match because: the two conv leaves tie via `convFlatBack_eq_vjp_backward`
    (3×3 is odd), the BN-backs are pinned to the exact certified terms, and the inner ReLU mask
    is the mid pre-activation sign — so after rewriting the conv leaves everything is
    definitional. Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem r34IdBlockBack_eq_rblkPC_vjp {c h w : Nat}
    (W₁ : Kernel4 c c 3 3) (b₁ : Vec c) (ε₁ : ℝ) (γ₁ β₁ : Vec c)
    (W₂ : Kernel4 c c 3 3) (b₂ : Vec c) (ε₂ : ℝ) (γ₂ β₂ : Vec c)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (c * h * w))
    (h_smooth₁ : ∀ k, bnPerChannelTensor3 c h w ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnPerChannelTensor3 c h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (c * h * w) ∘ bnPerChannelTensor3 c h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) v k + v k ≠ 0) :
    r34IdBlockBack W₁ W₂
      ((bnPerChannelTensor3_has_vjp c h w ε₁ hε₁ γ₁ β₁).backward (flatConv W₁ b₁ v))
      ((bnPerChannelTensor3_has_vjp c h w ε₂ hε₂ γ₂ β₂).backward
        (flatConv W₂ b₂ ((relu (c*h*w) ∘ bnPerChannelTensor3 c h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁) v)))
      (fun i => Proofs.residual ((bnPerChannelTensor3 c h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (c*h*w) ∘ bnPerChannelTensor3 c h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)) v i > 0)
      (fun i => bnPerChannelTensor3 c h w ε₁ γ₁ β₁ (flatConv W₁ b₁ v) i > 0)
      = (rblkPC_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁ h_smooth_out).backward := by
  funext dy
  unfold r34IdBlockBack
  rw [convFlatBack_eq_vjp_backward (b := b₁) (x := v) (by decide) (by decide),
      convFlatBack_eq_vjp_backward (b := b₂)
        (x := (relu (c*h*w) ∘ bnPerChannelTensor3 c h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁) v)
        (by decide) (by decide)]
  rfl

end Proofs
