import LeanMlir.Proofs.Codegen.ResNet34RenderPC
import LeanMlir.Proofs.Architectures.CifarCNN

/-! # R50 phase 1 — the THREE bottleneck blocks, forward + certified VJP

`planning/next_session_pipeline_then_r50.md` §3.1. ResNet-50's residual unit is the **bottleneck**:
`1×1 (ic→mid) → 3×3 (mid→mid) → 1×1 (mid→oc)`, three convs and two interior ReLUs, against the
basic block's two convs and one. `ResNet34RenderPC.lean`'s `rblkPC` / `rblkPStridedPC` are the
peers this file mirrors.

## ⚠⚠ THERE ARE THREE FORMS, AND THE THIRD IS THE ONE A FIRST LOOK MISSES

| form | where in R50 | R34 analogue |
|---|---|---|
| `bblkPC` — identity | 12 blocks | `rblkPC` |
| `bblkPStridedPC` — strided projection | stages 2/3/4, block 0 | `rblkPStridedPC` |
| ⭐ `bblkPProjPC` — **stride-1** projection | **stage 1 block 0 ONLY** | ⛔ **none** |

R34 never needed the third because its stage 1 is `ic = oc = 64`, so block 0 is an identity block.
R50's stage 1 goes **64 → 256 at stride 1**: the channel count changes, so it needs a projection,
but the resolution does not, so that projection is not strided.

⚠ **`rblkPStridedPC` cannot be reused for it, and the reason is its TYPE, not its arguments.** It
reads `Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)` — the halving is in the signature — and it
bakes `flatConvStride2` into both the body's leading conv and the projection. Substituting it at
stage 1 is a shape error and fails loudly. ⭐ **Reaching for `bblkPC` instead is the dangerous
mistake**: an identity skip where a projection belongs is well-typed only if `ic = oc`, and at
`64 → 256` it is not — but the failure mode a reader should fear is the one where a similar
substitution *does* typecheck and silently trains a different net. This file builds the stride-1
projection FIRST for exactly that reason.

## ⚠ THE STRIDE IS ON THE 3×3, NOT THE LEADING 1×1

`bblkPStridedPC` puts `flatConvStride2` on the **second** conv. That is ResNet **v1.5** /
torchvision, and it is what `jax/MainResnet50Imagenet.lean`'s reference trains. The v1 placement
(stride on the leading 1×1) compiles, trains and descends — and is a different net, worth ~0.5 pt
of top-1. The projection is strided in both conventions.

## What this cost, and why it was cheap

Nothing new was needed underneath. `convBnReluPC_has_vjp_at`, `flatConv_has_vjp`,
`flatConvStride2_has_vjp`, `bnPerChannelTensor3_has_vjp` and `residualProj_has_vjp_at` are all
already generic in `{ic oc h w kH kW}` / `{m n}`, so the bottleneck's extra conv is one more
`vjp_comp_at` link rather than a new foundation — §1's measured "zero new SHlo ops" showing up on
the proof side as zero new VJP obligations. The kernel extents stay **binders** (`kH₁ kW₁ …`)
rather than literals, so 1×1-vs-3×3 is an argument, exactly as `downFwd`'s docstring argues for
R34's projection.
-/

namespace Proofs

open Classical

-- ════════════════════════════════════════════════════════════════
-- § The three bottleneck ℝ-forwards (per-channel BN, non-batched)
-- ════════════════════════════════════════════════════════════════

/-- **Identity bottleneck** `relu(F(x) + x)`, `F = (bn₃∘conv₃) ∘ (relu∘bn₂∘conv₂) ∘ (relu∘bn₁∘conv₁)`.
    Channels go `c → mid → mid → c`; resolution is unchanged. The 3-conv peer of `rblkPC`. -/
@[reducible] noncomputable def bblkPC {c mid h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid c kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 c mid kH₃ kW₃) (b₃ : Vec c) (ε₃ : ℝ) (γ₃ β₃ : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  relu (c * h * w) ∘ residual
    ((bnPerChannelTensor3 c h w ε₃ γ₃ β₃ ∘ flatConv W₃ b₃) ∘
      ((relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)))

/-- ⭐ **Stride-1 projection bottleneck** `relu(F(x) + proj(x))`, everything at stride 1 —
    **R50 stage 1 block 0, and nothing else in the net**.

    Channels `ic → mid → mid → oc` with `ic ≠ oc` (64 → 256), resolution unchanged. The projection
    is `bn∘conv` at stride 1, where `rblkPStridedPC`'s is `bn∘conv_stride2`.

    ⚠ This is the form with **no R34 analogue**; see the module docstring. Its input and output
    types share `h`/`w`, which is precisely what distinguishes it from `bblkPStridedPC` below. -/
@[reducible] noncomputable def bblkPProjPC {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  relu (oc * h * w) ∘ residualProj
    (bnPerChannelTensor3 oc h w εp γp βp ∘ flatConv Wp bp)
    ((bnPerChannelTensor3 oc h w ε₃ γ₃ β₃ ∘ flatConv W₃ b₃) ∘
      ((relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁)))

/-- **Strided projection bottleneck** — R50 stages 2/3/4, block 0. Channels `ic → mid → mid → oc`,
    resolution halved.

    ⚠⚠ **The stride is on the 3×3 (`W₂`), not the leading 1×1.** That is v1.5 / torchvision, which
    is what the reference trains. The leading conv stays stride-1 at the INPUT resolution
    `(2*h)×(2*w)`, and `mid` channels are carried at that resolution until `W₂` decimates. -/
@[reducible] noncomputable def bblkPStridedPC {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc) :
    Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
  relu (oc * h * w) ∘ residualProj
    (bnPerChannelTensor3 oc h w εp γp βp ∘ flatConvStride2 Wp bp)
    ((bnPerChannelTensor3 oc h w ε₃ γ₃ β₃ ∘ flatConv W₃ b₃) ∘
      ((relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₂ γ₂ β₂ ∘ flatConvStride2 W₂ b₂) ∘
        (relu (mid * (2 * h) * (2 * w)) ∘ bnPerChannelTensor3 mid (2 * h) (2 * w) ε₁ γ₁ β₁ ∘
          flatConv W₁ b₁)))

-- ════════════════════════════════════════════════════════════════
-- § Certified VJPs
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **Certified VJP of the stride-1 projection bottleneck** — R50 stage 1 block 0.

    Built FIRST of the three (§3.1). Assembled from `convBnReluPC_has_vjp_at` (×2, for the two
    interior ReLU stages), a plain `bn∘conv` tail, the `bn∘conv` projection, and
    `residualProj_has_vjp_at`'s two-branch fan-in — mirroring `rblkPStridedPC_has_vjp_at` with
    every `flatConvStride2` replaced by `flatConv` and one more conv stage in the body.

    Smoothness is required at each ReLU: the two interior ones and the output. -/
noncomputable def bblkPProjPC_has_vjp_at {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hε₃ : 0 < ε₃) (hεp : 0 < εp)
    (v : Vec (ic * h * w))
    (h_smooth₁ : ∀ k, bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0)
    (h_smooth₂ : ∀ k, bnPerChannelTensor3 mid h w ε₂ γ₂ β₂
      (flatConv W₂ b₂ ((relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ ∘
        flatConv W₁ b₁) v)) k ≠ 0)
    (h_smooth_out : ∀ k,
      residualProj (bnPerChannelTensor3 oc h w εp γp βp ∘ flatConv Wp bp)
        ((bnPerChannelTensor3 oc h w ε₃ γ₃ β₃ ∘ flatConv W₃ b₃) ∘
          ((relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
            (relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))) v k ≠ 0) :
    HasVJPAt (bblkPProjPC (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
      Wp bp εp γp βp) v := by
  -- the projection: bn ∘ conv, at stride 1 — this is the whole difference from the strided form
  set proj := bnPerChannelTensor3 oc h w εp γp βp ∘ flatConv Wp bp with hproj
  have hproj_diff : DifferentiableAt ℝ proj v :=
    ((bnPerChannelTensor3_differentiable oc h w εp hεp γp βp).comp (flatConv_differentiable Wp bp)) v
  have hproj_vjp : HasVJPAt proj v :=
    vjp_comp_at (flatConv Wp bp) (bnPerChannelTensor3 oc h w εp γp βp) v
      (flatConv_differentiable Wp bp _)
      ((bnPerChannelTensor3_differentiable oc h w εp hεp γp βp) _)
      ((hasVJP3_to_hasVJP (conv2d_has_vjp3 Wp bp)).toHasVJPAt v)
      ((bnPerChannelTensor3_has_vjp oc h w εp hεp γp βp).toHasVJPAt _)
  -- body stage 1: relu ∘ bn ∘ conv (ic → mid)
  set stage1 := relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁ with hs1
  have hstage1_vjp : HasVJPAt stage1 v := convBnReluPC_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  have hstage1_diff : DifferentiableAt ℝ stage1 v :=
    convBnReluPC_differentiableAt W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  -- body stage 2: relu ∘ bn ∘ conv (mid → mid) — the 3×3, at stride 1 here
  set stage2 := relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂ with hs2
  have hstage2_vjp : HasVJPAt stage2 (stage1 v) :=
    convBnReluPC_has_vjp_at W₂ b₂ ε₂ γ₂ β₂ hε₂ (stage1 v) h_smooth₂
  have hstage2_diff : DifferentiableAt ℝ stage2 (stage1 v) :=
    convBnReluPC_differentiableAt W₂ b₂ ε₂ γ₂ β₂ hε₂ (stage1 v) h_smooth₂
  -- body stage 3: bn ∘ conv (mid → oc), NO relu — the residual is added before the activation
  set stage3 := bnPerChannelTensor3 oc h w ε₃ γ₃ β₃ ∘ flatConv W₃ b₃ with hs3
  set inner := stage2 ∘ stage1 with hinner
  have hinner_vjp : HasVJPAt inner v :=
    vjp_comp_at stage1 stage2 v hstage1_diff hstage2_diff hstage1_vjp hstage2_vjp
  have hinner_diff : DifferentiableAt ℝ inner v := DifferentiableAt.comp v hstage2_diff hstage1_diff
  have hstage3_diff : DifferentiableAt ℝ stage3 (inner v) :=
    ((bnPerChannelTensor3_differentiable oc h w ε₃ hε₃ γ₃ β₃).comp (flatConv_differentiable W₃ b₃))
      (inner v)
  have hstage3_vjp : HasVJPAt stage3 (inner v) :=
    vjp_comp_at (flatConv W₃ b₃) (bnPerChannelTensor3 oc h w ε₃ γ₃ β₃) (inner v)
      (flatConv_differentiable W₃ b₃ _)
      ((bnPerChannelTensor3_differentiable oc h w ε₃ hε₃ γ₃ β₃) _)
      ((hasVJP3_to_hasVJP (conv2d_has_vjp3 W₃ b₃)).toHasVJPAt _)
      ((bnPerChannelTensor3_has_vjp oc h w ε₃ hε₃ γ₃ β₃).toHasVJPAt _)
  set F := stage3 ∘ inner with hF
  have hF_vjp : HasVJPAt F v :=
    vjp_comp_at inner stage3 v hinner_diff hstage3_diff hinner_vjp hstage3_vjp
  have hF_diff : DifferentiableAt ℝ F v := DifferentiableAt.comp v hstage3_diff hinner_diff
  have hres_vjp : HasVJPAt (residualProj proj F) v :=
    residualProj_has_vjp_at proj F v hproj_diff hF_diff hproj_vjp hF_vjp
  have hres_diff : DifferentiableAt ℝ (residualProj proj F) v :=
    DifferentiableAt.add hproj_diff hF_diff
  have h_smooth_res : ∀ k, residualProj proj F v k ≠ 0 := h_smooth_out
  exact vjp_comp_at (residualProj proj F) (relu (oc * h * w)) v hres_diff
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth_res) hres_vjp
    (relu_has_vjp_at (oc * h * w) _ h_smooth_res)

/-- **Certified VJP of the identity bottleneck** — R50's 12 identity blocks. The 3-conv peer of
    `rblkPC_has_vjp_at`: same `residual` fan-in, one more `vjp_comp_at` link in the body. -/
noncomputable def bblkPC_has_vjp_at {c mid h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ : Nat}
    (W₁ : Kernel4 mid c kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 c mid kH₃ kW₃) (b₃ : Vec c) (ε₃ : ℝ) (γ₃ β₃ : Vec c)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hε₃ : 0 < ε₃)
    (v : Vec (c * h * w))
    (h_smooth₁ : ∀ k, bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0)
    (h_smooth₂ : ∀ k, bnPerChannelTensor3 mid h w ε₂ γ₂ β₂
      (flatConv W₂ b₂ ((relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ ∘
        flatConv W₁ b₁) v)) k ≠ 0)
    (h_smooth_out : ∀ k,
      residual ((bnPerChannelTensor3 c h w ε₃ γ₃ β₃ ∘ flatConv W₃ b₃) ∘
        ((relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))) v k ≠ 0) :
    HasVJPAt (bblkPC (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃) v := by
  set stage1 := relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₁ γ₁ β₁ ∘ flatConv W₁ b₁ with hs1
  have hstage1_vjp : HasVJPAt stage1 v := convBnReluPC_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  have hstage1_diff : DifferentiableAt ℝ stage1 v :=
    convBnReluPC_differentiableAt W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  set stage2 := relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₂ γ₂ β₂ ∘ flatConv W₂ b₂ with hs2
  have hstage2_vjp : HasVJPAt stage2 (stage1 v) :=
    convBnReluPC_has_vjp_at W₂ b₂ ε₂ γ₂ β₂ hε₂ (stage1 v) h_smooth₂
  have hstage2_diff : DifferentiableAt ℝ stage2 (stage1 v) :=
    convBnReluPC_differentiableAt W₂ b₂ ε₂ γ₂ β₂ hε₂ (stage1 v) h_smooth₂
  set stage3 := bnPerChannelTensor3 c h w ε₃ γ₃ β₃ ∘ flatConv W₃ b₃ with hs3
  set inner := stage2 ∘ stage1 with hinner
  have hinner_vjp : HasVJPAt inner v :=
    vjp_comp_at stage1 stage2 v hstage1_diff hstage2_diff hstage1_vjp hstage2_vjp
  have hinner_diff : DifferentiableAt ℝ inner v := DifferentiableAt.comp v hstage2_diff hstage1_diff
  have hstage3_diff : DifferentiableAt ℝ stage3 (inner v) :=
    ((bnPerChannelTensor3_differentiable c h w ε₃ hε₃ γ₃ β₃).comp (flatConv_differentiable W₃ b₃))
      (inner v)
  have hstage3_vjp : HasVJPAt stage3 (inner v) :=
    vjp_comp_at (flatConv W₃ b₃) (bnPerChannelTensor3 c h w ε₃ γ₃ β₃) (inner v)
      (flatConv_differentiable W₃ b₃ _)
      ((bnPerChannelTensor3_differentiable c h w ε₃ hε₃ γ₃ β₃) _)
      ((hasVJP3_to_hasVJP (conv2d_has_vjp3 W₃ b₃)).toHasVJPAt _)
      ((bnPerChannelTensor3_has_vjp c h w ε₃ hε₃ γ₃ β₃).toHasVJPAt _)
  set F := stage3 ∘ inner with hF
  have hF_vjp : HasVJPAt F v :=
    vjp_comp_at inner stage3 v hinner_diff hstage3_diff hinner_vjp hstage3_vjp
  have hF_diff : DifferentiableAt ℝ F v := DifferentiableAt.comp v hstage3_diff hinner_diff
  have hres_vjp : HasVJPAt (residual F) v := residual_has_vjp_at F v hF_diff hF_vjp
  have hres_diff : DifferentiableAt ℝ (residual F) v := by
    show DifferentiableAt ℝ (biPath F (fun x => x)) v
    exact DifferentiableAt.add hF_diff differentiable_id.differentiableAt
  have h_smooth_res : ∀ k, residual F v k ≠ 0 := h_smooth_out
  exact vjp_comp_at (residual F) (relu (c * h * w)) v hres_diff
    (relu_differentiableAt_of_smooth (c * h * w) _ h_smooth_res) hres_vjp
    (relu_has_vjp_at (c * h * w) _ h_smooth_res)

/-- **Certified VJP of the strided projection bottleneck** — R50 stages 2/3/4, block 0.

    ⚠ `flatConvStride2` sits on `W₂` (the 3×3) and on `Wp`, and `W₁` is a stride-1 conv at the
    INPUT resolution `(2*h)×(2*w)`. That asymmetry is v1.5 and it is why stage 1's smoothness
    hypothesis is stated at `(2*h)*(2*w)` while stage 2's is at `h*w`. -/
noncomputable def bblkPStridedPC_has_vjp_at {ic mid oc h w kH₁ kW₁ kH₂ kW₂ kH₃ kW₃ kHp kWp : Nat}
    (W₁ : Kernel4 mid ic kH₁ kW₁) (b₁ : Vec mid) (ε₁ : ℝ) (γ₁ β₁ : Vec mid)
    (W₂ : Kernel4 mid mid kH₂ kW₂) (b₂ : Vec mid) (ε₂ : ℝ) (γ₂ β₂ : Vec mid)
    (W₃ : Kernel4 oc mid kH₃ kW₃) (b₃ : Vec oc) (ε₃ : ℝ) (γ₃ β₃ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hε₃ : 0 < ε₃) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnPerChannelTensor3 mid (2 * h) (2 * w) ε₁ γ₁ β₁ (flatConv W₁ b₁ v) k ≠ 0)
    (h_smooth₂ : ∀ k, bnPerChannelTensor3 mid h w ε₂ γ₂ β₂
      (flatConvStride2 W₂ b₂ ((relu (mid * (2 * h) * (2 * w)) ∘
        bnPerChannelTensor3 mid (2 * h) (2 * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁) v)) k ≠ 0)
    (h_smooth_out : ∀ k,
      residualProj (bnPerChannelTensor3 oc h w εp γp βp ∘ flatConvStride2 Wp bp)
        ((bnPerChannelTensor3 oc h w ε₃ γ₃ β₃ ∘ flatConv W₃ b₃) ∘
          ((relu (mid * h * w) ∘ bnPerChannelTensor3 mid h w ε₂ γ₂ β₂ ∘ flatConvStride2 W₂ b₂) ∘
            (relu (mid * (2 * h) * (2 * w)) ∘
              bnPerChannelTensor3 mid (2 * h) (2 * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁))) v k ≠ 0) :
    HasVJPAt (bblkPStridedPC (h := h) (w := w) W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
      Wp bp εp γp βp) v := by
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
  -- stage 1 is stride-1 at the INPUT resolution; only stage 2 decimates
  set stage1 := relu (mid * (2 * h) * (2 * w)) ∘
    bnPerChannelTensor3 mid (2 * h) (2 * w) ε₁ γ₁ β₁ ∘ flatConv W₁ b₁ with hs1
  have hstage1_vjp : HasVJPAt stage1 v := convBnReluPC_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  have hstage1_diff : DifferentiableAt ℝ stage1 v :=
    convBnReluPC_differentiableAt W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  -- stage 2: the 3×3 STRIDE-2 conv — no `convBnReluPC` helper for the strided form, so the
  -- relu ∘ bn ∘ conv_s2 chain is assembled here exactly as `rblkPStridedPC_has_vjp_at` does.
  set s2in := bnPerChannelTensor3 mid h w ε₂ γ₂ β₂ ∘ flatConvStride2 W₂ b₂ with hs2in
  have hs2in_diff : DifferentiableAt ℝ s2in (stage1 v) :=
    ((bnPerChannelTensor3_differentiable mid h w ε₂ hε₂ γ₂ β₂).comp
      (flatConvStride2_differentiable W₂ b₂)) (stage1 v)
  have hs2in_vjp : HasVJPAt s2in (stage1 v) :=
    vjp_comp_at (flatConvStride2 W₂ b₂) (bnPerChannelTensor3 mid h w ε₂ γ₂ β₂) (stage1 v)
      (flatConvStride2_differentiable W₂ b₂ _)
      ((bnPerChannelTensor3_differentiable mid h w ε₂ hε₂ γ₂ β₂) _)
      ((flatConvStride2_has_vjp W₂ b₂).toHasVJPAt _)
      ((bnPerChannelTensor3_has_vjp mid h w ε₂ hε₂ γ₂ β₂).toHasVJPAt _)
  set stage2 := relu (mid * h * w) ∘ s2in with hs2
  have hstage2_vjp : HasVJPAt stage2 (stage1 v) :=
    vjp_comp_at s2in (relu (mid * h * w)) (stage1 v) hs2in_diff
      (relu_differentiableAt_of_smooth (mid * h * w) _ h_smooth₂) hs2in_vjp
      (relu_has_vjp_at (mid * h * w) _ h_smooth₂)
  have hstage2_diff : DifferentiableAt ℝ stage2 (stage1 v) :=
    (relu_differentiableAt_of_smooth (mid * h * w) _ h_smooth₂).comp (stage1 v) hs2in_diff
  set stage3 := bnPerChannelTensor3 oc h w ε₃ γ₃ β₃ ∘ flatConv W₃ b₃ with hs3
  set inner := stage2 ∘ stage1 with hinner
  have hinner_vjp : HasVJPAt inner v :=
    vjp_comp_at stage1 stage2 v hstage1_diff hstage2_diff hstage1_vjp hstage2_vjp
  have hinner_diff : DifferentiableAt ℝ inner v := DifferentiableAt.comp v hstage2_diff hstage1_diff
  have hstage3_diff : DifferentiableAt ℝ stage3 (inner v) :=
    ((bnPerChannelTensor3_differentiable oc h w ε₃ hε₃ γ₃ β₃).comp (flatConv_differentiable W₃ b₃))
      (inner v)
  have hstage3_vjp : HasVJPAt stage3 (inner v) :=
    vjp_comp_at (flatConv W₃ b₃) (bnPerChannelTensor3 oc h w ε₃ γ₃ β₃) (inner v)
      (flatConv_differentiable W₃ b₃ _)
      ((bnPerChannelTensor3_differentiable oc h w ε₃ hε₃ γ₃ β₃) _)
      ((hasVJP3_to_hasVJP (conv2d_has_vjp3 W₃ b₃)).toHasVJPAt _)
      ((bnPerChannelTensor3_has_vjp oc h w ε₃ hε₃ γ₃ β₃).toHasVJPAt _)
  set F := stage3 ∘ inner with hF
  have hF_vjp : HasVJPAt F v :=
    vjp_comp_at inner stage3 v hinner_diff hstage3_diff hinner_vjp hstage3_vjp
  have hF_diff : DifferentiableAt ℝ F v := DifferentiableAt.comp v hstage3_diff hinner_diff
  have hres_vjp : HasVJPAt (residualProj proj F) v :=
    residualProj_has_vjp_at proj F v hproj_diff hF_diff hproj_vjp hF_vjp
  have hres_diff : DifferentiableAt ℝ (residualProj proj F) v :=
    DifferentiableAt.add hproj_diff hF_diff
  have h_smooth_res : ∀ k, residualProj proj F v k ≠ 0 := h_smooth_out
  exact vjp_comp_at (residualProj proj F) (relu (oc * h * w)) v hres_diff
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth_res) hres_vjp
    (relu_has_vjp_at (oc * h * w) _ h_smooth_res)

-- ════════════════════════════════════════════════════════════════
-- § Wiring check at R50's REAL dimensions
-- ════════════════════════════════════════════════════════════════

/-! The three forwards above are generic in `{ic mid oc h w}`, which is what makes them reusable and
also what makes a mis-wired `mid` or `oc` typecheck happily in the abstract. This section pins them
at ResNet-50's actual positions and **composes a whole stage**, so the types have to agree
end-to-end — the shape error surfaces here rather than in a render three phases later.

⭐ `stage1` below is the load-bearing one: its block 0 is `bblkPProjPC` (64 → 256 at 56², stride 1),
and its blocks 1–2 are `bblkPC` at `c = 256, mid = 64`. Note what that means — **the identity
blocks' `mid` is a QUARTER of their `c`**, which is the bottleneck's whole point and the easiest
thing to get backwards. -/
section R50Dims

/-- **Stage 1 (56², `[64→256] ×3`)** — projection block 0 then two identity blocks, composed. The
    only place in R50 where a *stride-1* projection appears. -/
noncomputable example (ε : ℝ)
    -- block 0: the stride-1 projection, 64 → 256
    (p1 : Kernel4 64 64 1 1) (pb1 : Vec 64) (pg1 pt1 : Vec 64)
    (p2 : Kernel4 64 64 3 3) (pb2 : Vec 64) (pg2 pt2 : Vec 64)
    (p3 : Kernel4 256 64 1 1) (pb3 : Vec 256) (pg3 pt3 : Vec 256)
    (pp : Kernel4 256 64 1 1) (pbp : Vec 256) (pgp ptp : Vec 256)
    -- blocks 1,2: identity at c = 256 with mid = 64
    (a1 : Kernel4 64 256 1 1) (ab1 : Vec 64) (ag1 at1 : Vec 64)
    (a2 : Kernel4 64 64 3 3) (ab2 : Vec 64) (ag2 at2 : Vec 64)
    (a3 : Kernel4 256 64 1 1) (ab3 : Vec 256) (ag3 at3 : Vec 256) :
    Vec (64 * 56 * 56) → Vec (256 * 56 * 56) :=
  bblkPC (h := 56) (w := 56) a1 ab1 ε ag1 at1 a2 ab2 ε ag2 at2 a3 ab3 ε ag3 at3 ∘
  bblkPC (h := 56) (w := 56) a1 ab1 ε ag1 at1 a2 ab2 ε ag2 at2 a3 ab3 ε ag3 at3 ∘
  bblkPProjPC (h := 56) (w := 56) p1 pb1 ε pg1 pt1 p2 pb2 ε pg2 pt2 p3 pb3 ε pg3 pt3
    pp pbp ε pgp ptp

/-- **Stage 2 (56² → 28², `[256→512] ×4`)** — strided projection block 0 then three identity
    blocks. ⚠ `h`/`w` are the OUTPUT resolution (28); `bblkPStridedPC` reads its input at `2*h`. -/
noncomputable example (ε : ℝ)
    (s1 : Kernel4 128 256 1 1) (sb1 : Vec 128) (sg1 st1 : Vec 128)
    (s2 : Kernel4 128 128 3 3) (sb2 : Vec 128) (sg2 st2 : Vec 128)
    (s3 : Kernel4 512 128 1 1) (sb3 : Vec 512) (sg3 st3 : Vec 512)
    (sp : Kernel4 512 256 1 1) (sbp : Vec 512) (sgp stp : Vec 512)
    (b1 : Kernel4 128 512 1 1) (bb1 : Vec 128) (bg1 bt1 : Vec 128)
    (b2 : Kernel4 128 128 3 3) (bb2 : Vec 128) (bg2 bt2 : Vec 128)
    (b3 : Kernel4 512 128 1 1) (bb3 : Vec 512) (bg3 bt3 : Vec 512) :
    Vec (256 * (2 * 28) * (2 * 28)) → Vec (512 * 28 * 28) :=
  bblkPC (h := 28) (w := 28) b1 bb1 ε bg1 bt1 b2 bb2 ε bg2 bt2 b3 bb3 ε bg3 bt3 ∘
  bblkPC (h := 28) (w := 28) b1 bb1 ε bg1 bt1 b2 bb2 ε bg2 bt2 b3 bb3 ε bg3 bt3 ∘
  bblkPC (h := 28) (w := 28) b1 bb1 ε bg1 bt1 b2 bb2 ε bg2 bt2 b3 bb3 ε bg3 bt3 ∘
  bblkPStridedPC (h := 28) (w := 28) s1 sb1 ε sg1 st1 s2 sb2 ε sg2 st2 s3 sb3 ε sg3 st3
    sp sbp ε sgp stp

/-- **Stage 4 (14² → 7², `[1024→2048] ×3`)** — the last stage, feeding GAP at 2048×7×7. -/
noncomputable example (ε : ℝ)
    (s1 : Kernel4 512 1024 1 1) (sb1 : Vec 512) (sg1 st1 : Vec 512)
    (s2 : Kernel4 512 512 3 3) (sb2 : Vec 512) (sg2 st2 : Vec 512)
    (s3 : Kernel4 2048 512 1 1) (sb3 : Vec 2048) (sg3 st3 : Vec 2048)
    (sp : Kernel4 2048 1024 1 1) (sbp : Vec 2048) (sgp stp : Vec 2048)
    (e1 : Kernel4 512 2048 1 1) (eb1 : Vec 512) (eg1 et1 : Vec 512)
    (e2 : Kernel4 512 512 3 3) (eb2 : Vec 512) (eg2 et2 : Vec 512)
    (e3 : Kernel4 2048 512 1 1) (eb3 : Vec 2048) (eg3 et3 : Vec 2048) :
    Vec (1024 * (2 * 7) * (2 * 7)) → Vec (2048 * 7 * 7) :=
  bblkPC (h := 7) (w := 7) e1 eb1 ε eg1 et1 e2 eb2 ε eg2 et2 e3 eb3 ε eg3 et3 ∘
  bblkPC (h := 7) (w := 7) e1 eb1 ε eg1 et1 e2 eb2 ε eg2 et2 e3 eb3 ε eg3 et3 ∘
  bblkPStridedPC (h := 7) (w := 7) s1 sb1 ε sg1 st1 s2 sb2 ε sg2 st2 s3 sb3 ε sg3 st3
    sp sbp ε sgp stp

end R50Dims

end Proofs
