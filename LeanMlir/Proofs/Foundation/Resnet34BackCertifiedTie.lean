import LeanMlir.Proofs.Codegen.ResNet34RenderPC
import LeanMlir.Proofs.Architectures.CifarCNN
import LeanMlir.Proofs.Float.Resnet34BackFloatBridge
import LeanMlir.Proofs.Architectures.EfficientNetChainClose
import LeanMlir.Proofs.Float.StridedConvBackFloatBridge
import LeanMlir.Proofs.Float.Resnet34DownBackFloatBridge
import LeanMlir.Proofs.Float.Resnet34WholeBackFloatBridge
import LeanMlir.Proofs.Foundation.IR

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
to **the certified gradient**, not merely to a hand-map.

The **downsample block** is closed the same way (`§ The DOWNSAMPLE block` below): a strided-conv
leaf tie (`flatConvStride2Back_eq_vjp_backward` = conv leaf + the `decimateBack` `rfl`), the
certified strided block VJP `rblkPStridedPC_has_vjp_at` (mirrors `resblockProj_has_vjp_at`, with the
`residualProj` two-branch fan-in), and the tie `r34DownBlockBack_eq_rblkPStridedPC_vjp`. So **both r34
block types** (identity + downsample) now target the certified gradient, b1-free.

The **endpoint leaf ties** (`§ The ENDPOINT leaf ties` below) close the rest of the per-op set:
`dense_transpose_eq_vjp_backward` (the dense head, `Wᵀ·dy` = certified `Mat.mulVec W`),
`gapBack_eq_vjp_backward` (GAP broadcast-÷, `rfl`), `maxPoolFlatBack_eq_vjp_backward` (the smooth-point
arg-max scatter). With these + the conv/strided-conv leaves above, **every per-op backward of the r34
whole-net `r34InputGrad` is now individually tied to its certified VJP.**

⭐⭐ **THE WHOLE-NET FOLD IS CLOSED (2026-09-03): `r34InputGrad_eq_resnet34_vjp`.** The last
section of this file assembles the per-op ties into `r34InputGrad = (resnet34_has_vjp_at …
).backward` at the full `3×224²` dims — so the claim is no longer "every piece of the chain is
the certified gradient" but "the chain IS the certified whole-net gradient".

⛔ The blocker recorded here previously — *"`resnet34_has_vjp_at` is parametric / only concretely
instantiated at toy `resnet34Concrete` dims"* — was a misreading. That theorem is **dimension-
generic and parametric in its component maps**, so instantiating it at ImageNet dims needs no new
apex: it needs the components' `HasVJPAt`/`DifferentiableAt` witnesses, and every one already
existed except the stem's (`cbrStridedPC_has_vjp_at`, added below — the strided peer of
`convBnReluPC_has_vjp_at`). ⭐ The blocks stay OPAQUE, entering as the `ChainData`/`PProd` bundles
`resnet34_has_vjp_at` already takes, so the whole-net `isDefEq` runs between variables.

⛔⛔ **What the tie actually found was a DRIFT, and it moved the r34 backward number.**
`r34InputGrad` pooled with `maxPoolFlatBack` — the **2×2** pool's backward — while the committed
forward `resnet34Forward_full_pc` pools with `maxPool3s2Flat`, He et al.'s 3×3/s2 stem pool.
`MaxPool3s2.lean`'s header warns the two share a TYPE and are different functions; nothing forced
the two statements to unify until a theorem needed them to be about ONE net. The missing leaf is
`MaxPool3s2BackFloatBridge.lean` (the ACCUMULATING scatter — 3×3/s2 windows overlap, so an input
can be the argmax of up to four outputs, window `4A` not `A`), and `r34_grad_float_le` moved
`1.458·10²⁴⁴ → 6.894·10²⁴⁴`. ⚠ This is `imagenet_specs_drift_from_twins` for the fourth time and
ConvNeXt's stale head-LayerNorm slot for the second: *"the same net as the tie" is an unchecked
claim until something forces the two statements to unify, and what forces it is needing the tie.*

⚠ Still supplied, and unchanged by this: the SMOOTHNESS side. The tie takes the pool's
`MaxPool3s2Smooth`, the stem's post-BN no-zero, and the per-block `ChainData` bundles as
hypotheses — a smooth-point statement, as every `HasVJPAt` in this cone is.
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

-- ════════════════════════════════════════════════════════════════
-- § The DOWNSAMPLE block — `relu ∘ residualProj(proj, F_s)`, strided convs
-- ════════════════════════════════════════════════════════════════

/-- **Strided conv input-VJP leaf tie.** `flatConvStride2Back W` (= `convFlatBack ∘ decimateBack`)
    IS the certified strided conv input-VJP `(flatConvStride2_has_vjp W b).backward x`, for odd
    kernels. Decomposes into the conv leaf tie (`convFlatBack_eq_vjp_backward`) and the decimate
    leaf (`decimateBack_eq_vjp`, `rfl`), matching `flatConvStride2 = decimateFlat ∘ flatConv`. -/
theorem flatConvStride2Back_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) :
    flatConvStride2Back (h := h) (w := w) W = (flatConvStride2_has_vjp W b).backward x := by
  funext dy
  show convFlatBack (h := 2*h) (w := 2*w) W (decimateBack oc h w dy) = _
  rw [convFlatBack_eq_vjp_backward hkH hkW W b x]
  rfl

/-- **Certified VJP of the per-channel-BN downsample block `rblkPStridedPC`** (non-batched).
    `relu ∘ residualProj(proj, F_s)` — `proj = bnPC∘convStride2(Wp)` (the `kHp×kWp`-stride-2 skip),
    `F_s = (bnPC₂∘conv₂) ∘ (relu∘bnPC₁∘convStride2(W₁))` (first conv strided). The same-vocabulary
    certified target for `r34DownBlockBack`. Mirrors the scalar-BN `resblockProj_has_vjp_at` with
    `bnPerChannelTensor3` + `flatConvStride2`; per-op VJPs assembled by `vjp_comp_at`. -/
noncomputable def rblkPStridedPC_has_vjp_at {ic oc h w kHp kWp : Nat}
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      (bnPerChannelTensor3 oc h w εp γp βp ∘ flatConvStride2 Wp bp) v k
      + ((bnPerChannelTensor3 oc h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (oc*h*w) ∘ bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v k ≠ 0) :
    HasVJPAt (rblkPStridedPC (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ Wp bp εp γp βp) v := by
  set proj := bnPerChannelTensor3 oc h w εp γp βp ∘ flatConvStride2 Wp bp with hproj
  have hproj_diff : DifferentiableAt ℝ proj v :=
    ((bnPerChannelTensor3_differentiable oc h w εp hεp γp βp).comp
      (flatConvStride2_differentiable Wp bp)) v
  have hproj_vjp : HasVJPAt proj v :=
    vjp_comp_at (flatConvStride2 Wp bp) (bnPerChannelTensor3 oc h w εp γp βp) v
      (flatConvStride2_differentiable Wp bp _)
      ((bnPerChannelTensor3_differentiable oc h w εp hεp γp βp) _)
      ((flatConvStride2_has_vjp Wp bp).toHasVJPAt v)
      ((bnPerChannelTensor3_has_vjp oc h w εp hεp γp βp).toHasVJPAt _)
  set stage1 := relu (oc*h*w) ∘ bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁ with hs1
  have hs1in_diff : DifferentiableAt ℝ
      (bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁) v :=
    ((bnPerChannelTensor3_differentiable oc h w ε₁ hε₁ γ₁ β₁).comp
      (flatConvStride2_differentiable W₁ b₁)) v
  have hs1in_vjp : HasVJPAt (bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁) v :=
    vjp_comp_at (flatConvStride2 W₁ b₁) (bnPerChannelTensor3 oc h w ε₁ γ₁ β₁) v
      (flatConvStride2_differentiable W₁ b₁ _)
      ((bnPerChannelTensor3_differentiable oc h w ε₁ hε₁ γ₁ β₁) _)
      ((flatConvStride2_has_vjp W₁ b₁).toHasVJPAt v)
      ((bnPerChannelTensor3_has_vjp oc h w ε₁ hε₁ γ₁ β₁).toHasVJPAt _)
  have hstage1_vjp : HasVJPAt stage1 v :=
    vjp_comp_at _ (relu (oc*h*w)) v hs1in_diff
      (relu_differentiableAt_of_smooth (oc*h*w) _ h_smooth₁) hs1in_vjp
      (relu_has_vjp_at (oc*h*w) _ h_smooth₁)
  have hstage1_diff : DifferentiableAt ℝ stage1 v :=
    (relu_differentiableAt_of_smooth (oc*h*w) _ h_smooth₁).comp v hs1in_diff
  set stage2 := bnPerChannelTensor3 oc h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂ with hs2
  have hstage2_diff : DifferentiableAt ℝ stage2 (stage1 v) :=
    ((bnPerChannelTensor3_differentiable oc h w ε₂ hε₂ γ₂ β₂).comp (flatConv_differentiable W₂ b₂))
      (stage1 v)
  have hstage2_vjp : HasVJPAt stage2 (stage1 v) :=
    vjp_comp_at (flatConv W₂ b₂) (bnPerChannelTensor3 oc h w ε₂ γ₂ β₂) (stage1 v)
      (flatConv_differentiable W₂ b₂ _)
      ((bnPerChannelTensor3_differentiable oc h w ε₂ hε₂ γ₂ β₂) _)
      ((hasVJP3_to_hasVJP (conv2d_has_vjp3 W₂ b₂)).toHasVJPAt _)
      ((bnPerChannelTensor3_has_vjp oc h w ε₂ hε₂ γ₂ β₂).toHasVJPAt _)
  set F := stage2 ∘ stage1 with hF
  have hF_vjp : HasVJPAt F v :=
    vjp_comp_at stage1 stage2 v hstage1_diff hstage2_diff hstage1_vjp hstage2_vjp
  have hF_diff : DifferentiableAt ℝ F v := DifferentiableAt.comp v hstage2_diff hstage1_diff
  have hres_vjp : HasVJPAt (residualProj proj F) v :=
    residualProj_has_vjp_at proj F v hproj_diff hF_diff hproj_vjp hF_vjp
  have hres_diff : DifferentiableAt ℝ (residualProj proj F) v := DifferentiableAt.add hproj_diff hF_diff
  have h_smooth_res : ∀ k, residualProj proj F v k ≠ 0 := h_smooth_out
  exact vjp_comp_at (residualProj proj F) (relu (oc*h*w)) v hres_diff
    (relu_differentiableAt_of_smooth (oc*h*w) _ h_smooth_res) hres_vjp
    (relu_has_vjp_at (oc*h*w) _ h_smooth_res)

/-- **The §B downsample-block tie: float-bridge backward = certified VJP.** `r34DownBlockBack`,
    with BN-backs pinned to the certified per-channel backwards and ReLU masks pinned to the pre-
    activation signs, equals `(rblkPStridedPC_has_vjp_at …).backward`. Both sides are
    `fun dy ↦ projBack(mask dy) + bodyBack(mask dy)` (the `residualProj` two-branch fan-in over the
    outer-relu mask). Closes by rewriting the two strided-conv leaves
    (`flatConvStride2Back_eq_vjp_backward`) and the one non-strided conv leaf
    (`convFlatBack_eq_vjp_backward`); the rest is definitional. Completes the r34 block set
    (identity + downsample). 3-axiom-clean. -/
theorem r34DownBlockBack_eq_rblkPStridedPC_vjp {ic oc h w kHp kWp : Nat}
    (hkHp : 2 * ((kHp - 1) / 2) + 1 = kHp) (hkWp : 2 * ((kWp - 1) / 2) + 1 = kWp)
    (W₁ : Kernel4 oc ic 3 3) (b₁ : Vec oc) (ε₁ : ℝ) (γ₁ β₁ : Vec oc)
    (W₂ : Kernel4 oc oc 3 3) (b₂ : Vec oc) (ε₂ : ℝ) (γ₂ β₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      (bnPerChannelTensor3 oc h w εp γp βp ∘ flatConvStride2 Wp bp) v k
      + ((bnPerChannelTensor3 oc h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (oc*h*w) ∘ bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v k ≠ 0) :
    r34DownBlockBack W₁ W₂ Wp
      ((bnPerChannelTensor3_has_vjp oc h w ε₁ hε₁ γ₁ β₁).backward (flatConvStride2 W₁ b₁ v))
      ((bnPerChannelTensor3_has_vjp oc h w ε₂ hε₂ γ₂ β₂).backward
        (flatConv W₂ b₂ ((relu (oc*h*w) ∘ bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁) v)))
      ((bnPerChannelTensor3_has_vjp oc h w εp hεp γp βp).backward (flatConvStride2 Wp bp v))
      (fun i => residualProj (bnPerChannelTensor3 oc h w εp γp βp ∘ flatConvStride2 Wp bp)
        ((bnPerChannelTensor3 oc h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (oc*h*w) ∘ bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v i > 0)
      (fun i => bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) i > 0)
      = (rblkPStridedPC_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ Wp bp εp γp βp
          hε₁ hε₂ hεp v h_smooth₁ h_smooth_out).backward := by
  funext dy
  unfold r34DownBlockBack
  rw [flatConvStride2Back_eq_vjp_backward hkHp hkWp Wp bp v,
      flatConvStride2Back_eq_vjp_backward (by decide) (by decide) W₁ b₁ v,
      convFlatBack_eq_vjp_backward (by decide) (by decide) W₂ b₂
        ((relu (oc*h*w) ∘ bnPerChannelTensor3 oc h w ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁) v)]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The ENDPOINT leaf ties — dense head, GAP, maxpool
--   (the stem's strided conv is `flatConvStride2Back_eq_vjp_backward` above)
-- ════════════════════════════════════════════════════════════════

/-- **Dense head input-VJP leaf tie.** The float-bridge dense backward `dense (Wᵀ) 0` (= `Wᵀ·dy`)
    IS the certified dense input-VJP `(dense_has_vjp W b).backward x` (= `Mat.mulVec W dy`), conv is
    linear so the activation `x` is ignored. One `mul_comm` per term. -/
theorem dense_transpose_eq_vjp_backward {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) :
    dense (Mat.transpose W) (0 : Vec m) = (dense_has_vjp W b).backward x := by
  funext dy i
  simp only [dense, dense_has_vjp, Mat.transpose, Mat.mulVec, Pi.zero_apply, add_zero]
  exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- **GAP input-VJP leaf tie.** The float-bridge `gapBack c h w` (broadcast `dy(channel)/(h·w)`)
    IS the certified GAP input-VJP `(globalAvgPoolFlat_has_vjp c h w).backward x` — definitionally
    the same broadcast-÷ map (the VJP ignores its primal argument). -/
theorem gapBack_eq_vjp_backward (c h w : Nat) (x : Vec (c * h * w)) :
    gapBack c h w = (globalAvgPoolFlat_has_vjp c h w).backward x := rfl

/-- **Maxpool input-VJP leaf tie (smooth point).** The float-bridge `maxPoolFlatBack x` (scatter
    `dy` to the arg-max cell, 0 elsewhere) IS the certified maxpool input-VJP
    `(maxPoolFlat_has_vjp_at x h_smooth).backward` at a smooth point (unique arg-max per window).
    Both denote `if MaxPool2IsArgmax then dy(winRow,winCol) else 0` (`IR.maxPoolBackDenote` =
    `maxPool2_has_vjp_at3.backward`). -/
theorem maxPoolFlatBack_eq_vjp_backward {c h w : Nat} (x : Tensor3 c (2*h) (2*w))
    (h_smooth : MaxPool2Smooth x) :
    maxPoolFlatBack x = (maxPoolFlat_has_vjp_at x h_smooth).backward := by
  funext dy idx
  simp only [maxPoolFlatBack, Tensor3.flatten, maxPoolFlat_has_vjp_at, hasVJPAt3_to_hasVJPAt,
    IR.maxPoolBackDenote, maxPool2_has_vjp_at3]

-- ════════════════════════════════════════════════════════════════
-- § The STEM leaf tie — conv(stride-2) → per-channel BN → relu
-- ════════════════════════════════════════════════════════════════

/-- **conv(stride-2) → per-channel-BN → relu VJP at a smooth point** — the strided peer of
    `convBnReluPC_has_vjp_at`, i.e. the certified VJP of `cbrStridedPC`, which is r34's stem. Two
    `vjp_comp_at`s: the strided conv, the per-channel BN (differentiable everywhere at `ε > 0`),
    then the ReLU at its smooth point. -/
noncomputable def cbrStridedPC_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 0) :
    HasVJPAt (cbrStridedPC (h := h) (w := w) W b ε γ β) v := by
  have hconv_diff : Differentiable ℝ
      (flatConvStride2 W b : Vec (ic * (2*h) * (2*w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W b
  have hbn_diff : Differentiable ℝ (bnPerChannelTensor3 oc h w ε γ β) :=
    bnPerChannelTensor3_differentiable oc h w ε hε γ β
  have step1 : HasVJPAt (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v :=
    vjp_comp_at (flatConvStride2 W b) (bnPerChannelTensor3 oc h w ε γ β) v
      (hconv_diff v) (hbn_diff _)
      ((flatConvStride2_has_vjp W b).toHasVJPAt v)
      ((bnPerChannelTensor3_has_vjp oc h w ε hε γ β).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v :=
    DifferentiableAt.comp v (hbn_diff (flatConvStride2 W b v)) (hconv_diff v)
  exact vjp_comp_at (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b)
    (relu (oc * h * w)) v step1_diff
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth)
    step1 (relu_has_vjp_at (oc * h * w) _ h_smooth)

/-- `cbrStridedPC` is differentiable at a smooth point (the companion `resnet34_has_vjp_at`
    threads alongside every `HasVJPAt`). -/
theorem cbrStridedPC_differentiableAt {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 0) :
    DifferentiableAt ℝ (cbrStridedPC (h := h) (w := w) W b ε γ β) v := by
  have hinner : DifferentiableAt ℝ
      (bnPerChannelTensor3 oc h w ε γ β ∘ flatConvStride2 W b) v :=
    ((bnPerChannelTensor3_differentiable oc h w ε hε γ β).comp
      (flatConvStride2_differentiable W b)) v
  exact (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth).comp v hinner

/-- **The stem tie.** `r34InputGrad`'s stem slot — `flatConvStride2Back Ws ∘ bnB ∘ reluMaskBack`,
    with the BN-back pinned to the certified per-channel backward and the mask to the actual
    post-BN sign — IS `(cbrStridedPC_has_vjp_at …).backward`. Closes by rewriting the one strided
    conv leaf; the BN-back and the ReLU mask are definitionally the certified terms
    (`relu_has_vjp_at`'s backward is `fun dy i => if x i > 0 then dy i else 0`, which is
    `reluMaskBack (· > 0)`). -/
theorem cbrStridedPCBack_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hkH : 2 * ((kH - 1) / 2) + 1 = kH) (hkW : 2 * ((kW - 1) / 2) + 1 = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) k ≠ 0) :
    flatConvStride2Back (h := h) (w := w) W
      ∘ (bnPerChannelTensor3_has_vjp oc h w ε hε γ β).backward (flatConvStride2 W b v)
      ∘ reluMaskBack (fun i => bnPerChannelTensor3 oc h w ε γ β (flatConvStride2 W b v) i > 0)
      = (cbrStridedPC_has_vjp_at W b ε γ β hε v h_smooth).backward := by
  funext dy
  rw [flatConvStride2Back_eq_vjp_backward hkH hkW W b v]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE WHOLE-NET FOLD — `r34InputGrad` IS the certified whole-net gradient
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 400000 in
set_option maxHeartbeats 2000000 in
/-- **The whole-net certified tie.** `r34InputGrad`, with every one of its slots pinned to the
    certified per-op backward, IS `(resnet34_has_vjp_at …).backward` — the certified
    input-gradient VJP of `dense ∘ GAP ∘ [3,4,6,3] ∘ maxPool3s2 ∘ stem` at `x`.

    ⭐ This is the statement the per-op ties above could not make. Until it existed the honest
    reading of `r34_grad_float_le` was *"every piece of this chain is the certified gradient"*;
    with it, the chain IS the certified whole-net gradient.

    ⛔⛔ **And closing it is what found the drift.** `r34InputGrad` used `maxPoolFlatBack` — the
    **2×2** pool's backward — while the committed forward `resnet34Forward_full_pc` pools with
    `maxPool3s2Flat`, He et al.'s 3×3/s2 stem pool. `MaxPool3s2.lean` warns that the two share a
    TYPE and are different functions; nothing forced them to unify until this theorem needed the
    two statements to be about one net. The fix is `MaxPool3s2BackFloatBridge.lean` (the
    accumulating scatter, window `4A` not `A`), and the r34 backward number moved
    `2.188·10²⁴⁵ → 8.857·10²⁴⁵`.

    ⭐ The blocks stay OPAQUE — they enter as the `ChainData`/`PProd` witnesses
    `resnet34_has_vjp_at` already takes, and `r34InputGrad`'s block slots are pinned to their
    backwards — so the composition is checked between variables and costs nothing (§3.7(a)'s
    lesson, in the direction that works). Only the four concrete endpoints are rewritten: the
    dense head, GAP, the 3×3/s2 pool and the stem. -/
theorem r34InputGrad_eq_resnet34_vjp
    (Ws : Kernel4 64 3 7 7) (bs : Vec 64) (ε : ℝ) (γs βs : Vec 64) (hε : 0 < ε)
    (Wd : Mat 512 10) (bd : Vec 10)
    (a2 a1 a0 : Vec (64 * 56 * 56) → Vec (64 * 56 * 56))
    (down2 : Vec (64 * 56 * 56) → Vec (128 * 28 * 28))
    (b2 b1 b0 : Vec (128 * 28 * 28) → Vec (128 * 28 * 28))
    (down3 : Vec (128 * 28 * 28) → Vec (256 * 14 * 14))
    (c4 c3 c2 c1 c0 : Vec (256 * 14 * 14) → Vec (256 * 14 * 14))
    (down4 : Vec (256 * 14 * 14) → Vec (512 * 7 * 7))
    (e1 e0 : Vec (512 * 7 * 7) → Vec (512 * 7 * 7))
    (x : Vec (3 * 224 * 224))
    (hstem_smooth : ∀ k,
      bnPerChannelTensor3 64 112 112 ε γs βs (flatConvStride2 Ws bs x) k ≠ 0)
    (hmp_smooth : MaxPool3s2Smooth
      (Tensor3.unflatten (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x)
        : Tensor3 64 (2 * 56) (2 * 56)))
    (hids1 : ChainData (maxPool3s2Flat 64 56 56
      (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x)) [a2, a1, a0])
    (hdown2 : PProd (HasVJPAt down2 (chainComp [a2, a1, a0] (maxPool3s2Flat 64 56 56
        (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x))))
      (DifferentiableAt ℝ down2 (chainComp [a2, a1, a0] (maxPool3s2Flat 64 56 56
        (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x)))))
    (hids2 : ChainData (down2 (chainComp [a2, a1, a0] (maxPool3s2Flat 64 56 56
      (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x)))) [b2, b1, b0])
    (hdown3 : PProd (HasVJPAt down3 (chainComp [b2, b1, b0] (down2 (chainComp [a2, a1, a0]
        (maxPool3s2Flat 64 56 56 (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x))))))
      (DifferentiableAt ℝ down3 (chainComp [b2, b1, b0] (down2 (chainComp [a2, a1, a0]
        (maxPool3s2Flat 64 56 56 (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x)))))))
    (hids3 : ChainData (down3 (chainComp [b2, b1, b0] (down2 (chainComp [a2, a1, a0]
      (maxPool3s2Flat 64 56 56 (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x))))))
      [c4, c3, c2, c1, c0])
    (hdown4 : PProd (HasVJPAt down4 (chainComp [c4, c3, c2, c1, c0] (down3 (chainComp [b2, b1, b0]
        (down2 (chainComp [a2, a1, a0] (maxPool3s2Flat 64 56 56
          (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x))))))))
      (DifferentiableAt ℝ down4 (chainComp [c4, c3, c2, c1, c0] (down3 (chainComp [b2, b1, b0]
        (down2 (chainComp [a2, a1, a0] (maxPool3s2Flat 64 56 56
          (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x)))))))))
    (hids4 : ChainData (down4 (chainComp [c4, c3, c2, c1, c0] (down3 (chainComp [b2, b1, b0]
      (down2 (chainComp [a2, a1, a0] (maxPool3s2Flat 64 56 56
        (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x)))))))) [e1, e0]) :
    r34InputGrad Ws Wd
      ((bnPerChannelTensor3_has_vjp 64 112 112 ε hε γs βs).backward (flatConvStride2 Ws bs x))
      hids4.snd.fst.backward hids4.snd.snd.snd.fst.backward
      hdown4.fst.backward
      hids3.snd.fst.backward hids3.snd.snd.snd.fst.backward
      hids3.snd.snd.snd.snd.snd.fst.backward
      hids3.snd.snd.snd.snd.snd.snd.snd.fst.backward
      hids3.snd.snd.snd.snd.snd.snd.snd.snd.snd.fst.backward
      hdown3.fst.backward
      hids2.snd.fst.backward hids2.snd.snd.snd.fst.backward
      hids2.snd.snd.snd.snd.snd.fst.backward
      hdown2.fst.backward
      hids1.snd.fst.backward hids1.snd.snd.snd.fst.backward
      hids1.snd.snd.snd.snd.snd.fst.backward
      (Tensor3.unflatten (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x))
      (fun i => bnPerChannelTensor3 64 112 112 ε γs βs (flatConvStride2 Ws bs x) i > 0)
      = (resnet34_has_vjp_at (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs)
          (maxPool3s2Flat 64 56 56) [a2, a1, a0] down2 [b2, b1, b0] down3 [c4, c3, c2, c1, c0]
          down4 [e1, e0] (globalAvgPoolFlat 512 7 7) (dense Wd bd) x
          ⟨cbrStridedPC_has_vjp_at (ic := 3) (oc := 64) (h := 112) (w := 112)
              Ws bs ε γs βs hε x hstem_smooth,
            cbrStridedPC_differentiableAt (ic := 3) (oc := 64) (h := 112) (w := 112)
              Ws bs ε γs βs hε x hstem_smooth⟩
          ⟨maxPool3s2Flat_has_vjp_at_vec (c := 64) (h := 56) (w := 56) _ hmp_smooth,
            maxPool3s2Flat_differentiableAt_vec (c := 64) (h := 56) (w := 56) _ hmp_smooth
              (by norm_num) (by norm_num) (by norm_num)⟩
          hids1 hdown2 hids2 hdown3 hids3 hdown4 hids4
          ⟨(globalAvgPoolFlat_has_vjp 512 7 7).toHasVJPAt _,
            (globalAvgPoolFlat_differentiable 512 7 7) _⟩
          ⟨(dense_has_vjp Wd bd).toHasVJPAt _, (dense_differentiable Wd bd) _⟩).backward := by
  unfold r34InputGrad
  rw [cbrStridedPCBack_eq_vjp_backward (ic := 3) (oc := 64) (h := 112) (w := 112)
        (by decide) (by decide) Ws bs ε γs βs hε x hstem_smooth,
      dense_transpose_eq_vjp_backward Wd bd
        (globalAvgPoolFlat 512 7 7 (chainComp [e1, e0] (down4 (chainComp [c4, c3, c2, c1, c0]
          (down3 (chainComp [b2, b1, b0] (down2 (chainComp [a2, a1, a0] (maxPool3s2Flat 64 56 56
            (cbrStridedPC (h := 112) (w := 112) Ws bs ε γs βs x))))))))))]
  rfl

end Proofs
