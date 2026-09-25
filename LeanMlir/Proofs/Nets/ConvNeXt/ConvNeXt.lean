import LeanMlir.Proofs.Architectures.Depthwise
import LeanMlir.Proofs.Architectures.LayerNorm

/-!
# ConvNeXt

A representative ConvNeXt block and a small end-to-end ConvNeXt VJP in
flattened `Vec` space — the ConvNeXt analogue of `cnnHasVJPAt`.

ConvNeXt is the "ResNet, modernized" architecture: it keeps the residual
skeleton but swaps in the ViT-era ingredients — a large-kernel (7×7)
*depthwise* conv for spatial mixing, *LayerNorm* instead of BatchNorm,
GELU instead of ReLU, an inverted-bottleneck MLP (1×1 expand → GELU →
1×1 project), and a learnable per-channel *layer scale*. Crucially, the
block has **no non-smooth activation** (GELU is smooth everywhere) and
**no max-pool**, so — unlike `cnnHasVJPAt` — the entire VJP composes
with *no kink hypotheses*: the only side conditions are the LayerNorm
positivity arguments `0 < ε`.

This file contributes three genuinely new pieces:

  * `layerScale` — per-channel learnable elementwise scale, a diagonal
    linear map; `Differentiable` + a `HasVJP` (`back i = γ i · dy i`),
    its Jacobian read off the basis vector by `pdiv_of_linear`.
  * the **ConvNeXt block body** `layerScale ∘ project ∘ gelu ∘ expand ∘
    LayerNorm ∘ depthwise`, everywhere-differentiable, with a pointwise
    VJP built by chaining the piece VJPs through `vjpCompAt`; and the
    full block `residual (block body)` (identity skip, no post-add act).
  * `convnextHasVJPAt` / `convnextHasVJPAt_correct` — a fixed-depth
    (two-block) end-to-end network: stem-patchify → stem-LN → block₁ →
    block₂ → global-avg-pool → head-LN → dense.

## LayerNorm representation caveat

The `layerNormForward` reused here is the proof's Vec→Vec LayerNorm with
*scalar* `γ, β` that normalizes over the *whole* flattened vector,
whereas true ConvNeXt LayerNorm is per-spatial-position over the channel
axis (LayerNorm over NCHW's C). This is the same representation
simplification the audit flagged for the LN family; a faithful
channel-LN-over-NCHW lift is a follow-up. Every other piece (depthwise
7×7, 1×1 convs, GELU, layer scale, GAP, dense) is exact.
-/

namespace Proofs

open Finset BigOperators

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt block body
-- ════════════════════════════════════════════════════════════════

/-- **ConvNeXt block body** (Vec→Vec, no skip):

      `layerScale γ ∘ project(1×1) ∘ gelu ∘ expand(1×1) ∘ layerNorm ∘ depthwise(7×7)`

    Channel/spatial dims are generic Nat params; the depthwise kernel is
    `c × kH × kW` (the 7×7 in ConvNeXt), the expand conv lifts `c → cExp`
    channels (the usual 4× inverted-bottleneck), gelu is applied on the
    expanded activation, the project conv brings `cExp → c` back, and the
    per-channel layer scale closes the block.

    **LN representation caveat.** The `layerNormForward` used here is the
    proof's Vec→Vec LayerNorm with *scalar* `γ_n, β_n` that normalizes
    over the *whole* flattened `c·h·w` vector, whereas true ConvNeXt LN is
    per-spatial-position over the channel axis (LayerNorm over NCHW's C).
    This is the same representation simplification the audit flagged for
    the LN family; a faithful channel-LN-over-NCHW is a follow-up. Every
    other piece is exact. Because gelu is smooth everywhere, LN is smooth
    given `ε>0`, and conv/layerScale are linear, the whole body is
    differentiable everywhere — no ReLU-style kink hypotheses needed. -/
noncomputable def convNextBlockBody {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  layerScale γls ∘
  (flatConv (h := h) (w := w) Wpr bpr) ∘
  (gelu (cExp * h * w)) ∘
  (flatConv (h := h) (w := w) Wex bex) ∘
  (layerNormForward (c * h * w) εn γn βn) ∘
  (depthwiseFlat (h := h) (w := w) Wdw bdw)

/-- The block body is differentiable everywhere (composition of
    everywhere-differentiable maps). -/
theorem convNextBlockBody_differentiable {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    Differentiable ℝ (convNextBlockBody Wdw bdw εn γn βn Wex bex Wpr bpr γls) := by
  unfold convNextBlockBody layerNormForward layerScale gelu
  fun_prop (disch := assumption)

/-- **ConvNeXt block body VJP (global)** — built by chaining the
    everywhere-differentiable piece VJPs through `vjpComp`. Needs only
    `0 < εn` (the LayerNorm positivity); no kink hypotheses since gelu is
    smooth and the rest are linear. Because the body is differentiable
    everywhere, the VJP is global (`HasVJP`), not pointwise. -/
noncomputable def convNextBlockBodyHasVJP {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    HasVJP (convNextBlockBody Wdw bdw εn γn βn Wex bex Wpr bpr γls) := by
  unfold convNextBlockBody
  have hdw := depthwiseFlat_differentiable (h := h) (w := w) Wdw bdw
  have hln := bnForward_differentiable (c * h * w) εn γn βn hεn
  have hex := flatConv_differentiable (h := h) (w := w) Wex bex
  have hge := gelu_differentiable (cExp * h * w)
  have hpr := flatConv_differentiable (h := h) (w := w) Wpr bpr
  have hls := layerScale_differentiable γls
  set D := depthwiseFlat (h := h) (w := w) Wdw bdw with hD
  set LN := layerNormForward (c * h * w) εn γn βn with hLN
  have d_vjp : HasVJP D := depthwiseFlatHasVJP (h := h) (w := w) Wdw bdw
  have ln_vjp : HasVJP LN := layerNormHasVJP (c * h * w) εn γn βn hεn
  have s1_vjp : HasVJP (LN ∘ D) := vjpComp D LN hdw hln d_vjp ln_vjp
  have s1_diff : Differentiable ℝ (LN ∘ D) := hln.comp hdw
  set EX := flatConv (h := h) (w := w) Wex bex with hEX
  have ex_vjp : HasVJP EX := HasVJP3.toHasVJP (conv2dHasVJP3 Wex bex)
  have s2_vjp : HasVJP (EX ∘ (LN ∘ D)) := vjpComp (LN ∘ D) EX s1_diff hex s1_vjp ex_vjp
  have s2_diff : Differentiable ℝ (EX ∘ (LN ∘ D)) := hex.comp s1_diff
  set GE := gelu (cExp * h * w) with hGE
  have ge_vjp : HasVJP GE := geluHasVJP (cExp * h * w)
  have s3_vjp : HasVJP (GE ∘ (EX ∘ (LN ∘ D))) := vjpComp (EX ∘ (LN ∘ D)) GE s2_diff hge s2_vjp ge_vjp
  have s3_diff : Differentiable ℝ (GE ∘ (EX ∘ (LN ∘ D))) := hge.comp s2_diff
  set PR := flatConv (h := h) (w := w) Wpr bpr with hPR
  have pr_vjp : HasVJP PR := HasVJP3.toHasVJP (conv2dHasVJP3 Wpr bpr)
  have s4_vjp : HasVJP (PR ∘ (GE ∘ (EX ∘ (LN ∘ D)))) :=
    vjpComp (GE ∘ (EX ∘ (LN ∘ D))) PR s3_diff hpr s3_vjp pr_vjp
  have s4_diff : Differentiable ℝ (PR ∘ (GE ∘ (EX ∘ (LN ∘ D)))) := hpr.comp s3_diff
  set LS := layerScale γls with hLS
  exact vjpComp (PR ∘ (GE ∘ (EX ∘ (LN ∘ D)))) LS s4_diff hls s4_vjp (layerScaleHasVJP γls)

/-- **Full ConvNeXt block** = `residual (block body)`. ConvNeXt uses an
    identity skip (no projection, no post-add activation), so this is the
    plain `residual` of the block body. -/
noncomputable def convNextBlock {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  residual (convNextBlockBody Wdw bdw εn γn βn Wex bex Wpr bpr γls)

/-- The full ConvNeXt block is differentiable everywhere (residual of an
    everywhere-differentiable body). -/
theorem convNextBlock_differentiable {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    Differentiable ℝ (convNextBlock Wdw bdw εn γn βn Wex bex Wpr bpr γls) := by
  unfold convNextBlock residual
  exact (convNextBlockBody_differentiable Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls).add
    differentiable_id

/-- **ConvNeXt block VJP (global)** — `residualHasVJP` on top of the
    block-body VJP. Needs only `0 < εn`. Global since the body is
    everywhere-differentiable and the skip is the identity. -/
noncomputable def convNextBlockHasVJP {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) :
    HasVJP (convNextBlock Wdw bdw εn γn βn Wex bex Wpr bpr γls) :=
  residualHasVJP (convNextBlockBody Wdw bdw εn γn βn Wex bex Wpr bpr γls)
    (convNextBlockBody_differentiable Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls)
    (convNextBlockBodyHasVJP Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls)

-- ════════════════════════════════════════════════════════════════
-- § End-to-end ConvNeXt
-- ════════════════════════════════════════════════════════════════

/-- **Forward ConvNeXt** (representative, fixed block count = 2):

      stem-patchify(1×1 conv) → stem-LN → block₁ → block₂
      → globalAvgPool → head-LN → dense

    Generic channel/spatial dims; `ic`→`c` patchify, two identity-skip
    ConvNeXt blocks at `c`, GAP to `Vec c`, a final LN over the pooled
    `Vec c`, and a `Mat c nClasses` linear head. Same LN representation
    caveat as `convNextBlockBody` applies to the stem-LN and head-LN. -/
noncomputable def convNextForward
    {ic c cExp h w kH kW nClasses : Nat}
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ)
    (Wd : Mat c nClasses) (bd : Vec nClasses) :
    Vec (ic * h * w) → Vec nClasses :=
  (dense Wd bd) ∘
  (layerNormForward c εhd γhd βhd) ∘
  (globalAvgPoolFlat c h w) ∘
  (convNextBlock Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂) ∘
  (convNextBlock Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁) ∘
  (layerNormForward (c * h * w) εst γst βst) ∘
  (flatConv (h := h) (w := w) Wst bst)

/-- **End-to-end ConvNeXt VJP (global).** Everything is smooth, so the
    only hypotheses are the four LayerNorm positivity conditions
    (`0 < εst, εn₁, εn₂, εhd`) — no ReLU/maxpool kink conditions, unlike
    `cnnHasVJPAt`. Chained entirely through the global `vjpComp`, so the
    VJP holds at *every* input, not just a fixed point — putting ConvNeXt
    alongside `vitFullHasVJP` as an unconditional whole-network VJP. -/
noncomputable def convnextHasVJP
    {ic c cExp h w kH kW nClasses : Nat}
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ) (hεst : 0 < εst)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ) (hεn₁ : 0 < εn₁)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ) (hεn₂ : 0 < εn₂)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ) (hεhd : 0 < εhd)
    (Wd : Mat c nClasses) (bd : Vec nClasses) :
    HasVJP (convNextForward Wst bst εst γst βst
      Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
      Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
      εhd γhd βhd Wd bd) := by
  unfold convNextForward
  set ST := flatConv (h := h) (w := w) Wst bst with hST
  have st_diff := flatConv_differentiable (h := h) (w := w) Wst bst
  have st_vjp : HasVJP ST := HasVJP3.toHasVJP (conv2dHasVJP3 Wst bst)
  set LNs := layerNormForward (c * h * w) εst γst βst with hLNs
  have lns_diff := bnForward_differentiable (c * h * w) εst γst βst hεst
  have lns_vjp : HasVJP LNs := layerNormHasVJP (c * h * w) εst γst βst hεst
  have s1_vjp : HasVJP (LNs ∘ ST) := vjpComp ST LNs st_diff lns_diff st_vjp lns_vjp
  have s1_diff : Differentiable ℝ (LNs ∘ ST) := lns_diff.comp st_diff
  set B1 := convNextBlock Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ with hB1
  have b1_diff := convNextBlock_differentiable Wdw₁ bdw₁ εn₁ hεn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
  have b1_vjp : HasVJP B1 := convNextBlockHasVJP Wdw₁ bdw₁ εn₁ hεn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
  have s2_vjp : HasVJP (B1 ∘ (LNs ∘ ST)) := vjpComp (LNs ∘ ST) B1 s1_diff b1_diff s1_vjp b1_vjp
  have s2_diff : Differentiable ℝ (B1 ∘ (LNs ∘ ST)) := b1_diff.comp s1_diff
  set B2 := convNextBlock Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ with hB2
  have b2_diff := convNextBlock_differentiable Wdw₂ bdw₂ εn₂ hεn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
  have b2_vjp : HasVJP B2 := convNextBlockHasVJP Wdw₂ bdw₂ εn₂ hεn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
  have s3_vjp : HasVJP (B2 ∘ (B1 ∘ (LNs ∘ ST))) := vjpComp (B1 ∘ (LNs ∘ ST)) B2 s2_diff b2_diff s2_vjp b2_vjp
  have s3_diff : Differentiable ℝ (B2 ∘ (B1 ∘ (LNs ∘ ST))) := b2_diff.comp s2_diff
  set P3 := B2 ∘ (B1 ∘ (LNs ∘ ST)) with hP3
  set GAP := globalAvgPoolFlat c h w with hGAP
  have gap_diff := globalAvgPoolFlat_differentiable c h w
  have gap_vjp : HasVJP GAP := globalAvgPoolFlatHasVJP c h w
  have s4_vjp : HasVJP (GAP ∘ P3) := vjpComp P3 GAP s3_diff gap_diff s3_vjp gap_vjp
  have s4_diff : Differentiable ℝ (GAP ∘ P3) := gap_diff.comp s3_diff
  set LNh := layerNormForward c εhd γhd βhd with hLNh
  have lnh_diff := bnForward_differentiable c εhd γhd βhd hεhd
  have lnh_vjp : HasVJP LNh := layerNormHasVJP c εhd γhd βhd hεhd
  have s5_vjp : HasVJP (LNh ∘ (GAP ∘ P3)) := vjpComp (GAP ∘ P3) LNh s4_diff lnh_diff s4_vjp lnh_vjp
  have s5_diff : Differentiable ℝ (LNh ∘ (GAP ∘ P3)) := lnh_diff.comp s4_diff
  exact vjpComp (LNh ∘ (GAP ∘ P3)) (dense Wd bd) s5_diff
    (dense_differentiable Wd bd) s5_vjp (denseHasVJP Wd bd)

/-- **End-to-end ConvNeXt VJP at a point** — the global witness restricted
    to a point. Kept for downstream `_at` consumers and the comparator. -/
noncomputable def convnextHasVJPAt
    {ic c cExp h w kH kW nClasses : Nat}
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ) (hεst : 0 < εst)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ) (hεn₁ : 0 < εn₁)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ) (hεn₂ : 0 < εn₂)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ) (hεhd : 0 < εhd)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) :
    HasVJPAt (convNextForward Wst bst εst γst βst
      Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
      Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
      εhd γhd βhd Wd bd) x :=
  (convnextHasVJP Wst bst εst γst βst hεst
    Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
    Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
    εhd γhd βhd hεhd Wd bd).toHasVJPAt x

/-- **Public correctness theorem for `convnextHasVJP` (global)** — the
    end-to-end ConvNeXt's backward equals the `pdiv`-contracted Jacobian
    (Jacobian-transpose applied to the cotangent), at *every* input `x`.
    The unconditional ConvNeXt analogue of `vitFullHasVJP_correct`. -/
theorem convnextHasVJP_correct
    {ic c cExp h w kH kW nClasses : Nat}
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ) (hεst : 0 < εst)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ) (hεn₁ : 0 < εn₁)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ) (hεn₂ : 0 < εn₂)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ) (hεhd : 0 < εhd)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (convnextHasVJP Wst bst εst γst βst hεst
      Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
      Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
      εhd γhd βhd hεhd Wd bd).backward x dy i =
      ∑ j : Fin nClasses,
        pdiv (convNextForward Wst bst εst γst βst
          Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
          Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
          εhd γhd βhd Wd bd) x i j * dy j :=
  (convnextHasVJP Wst bst εst γst βst hεst
    Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
    Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
    εhd γhd βhd hεhd Wd bd).correct x dy i

/-- **Public correctness theorem for `convnextHasVJPAt`** — exposes the
    witness's `.correct` field: the end-to-end ConvNeXt's backward equals
    the `pdiv`-contracted Jacobian (Jacobian-transpose applied to the
    cotangent). Analogue of `cnnHasVJPAt_correct`. -/
theorem convnextHasVJPAt_correct
    {ic c cExp h w kH kW nClasses : Nat}
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ) (hεst : 0 < εst)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ) (hεn₁ : 0 < εn₁)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ) (hεn₂ : 0 < εn₂)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ) (hεhd : 0 < εhd)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (convnextHasVJPAt Wst bst εst γst βst hεst
      Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
      Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
      εhd γhd βhd hεhd Wd bd x).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (convNextForward Wst bst εst γst βst
          Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
          Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
          εhd γhd βhd Wd bd) x i j * dy j :=
  (convnextHasVJPAt Wst bst εst γst βst hεst
    Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
    Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
    εhd γhd βhd hεhd Wd bd x).correct dy i

end Proofs
