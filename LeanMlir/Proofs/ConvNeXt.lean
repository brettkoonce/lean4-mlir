import LeanMlir.Proofs.Depthwise
import LeanMlir.Proofs.LayerNorm

/-!
# ConvNeXt

A representative ConvNeXt block and a small end-to-end ConvNeXt VJP in
flattened `Vec` space — the ConvNeXt analogue of `cnn_has_vjp_at`.

ConvNeXt is the "ResNet, modernized" architecture: it keeps the residual
skeleton but swaps in the ViT-era ingredients — a large-kernel (7×7)
*depthwise* conv for spatial mixing, *LayerNorm* instead of BatchNorm,
GELU instead of ReLU, an inverted-bottleneck MLP (1×1 expand → GELU →
1×1 project), and a learnable per-channel *layer scale*. Crucially, the
block has **no non-smooth activation** (GELU is smooth everywhere) and
**no max-pool**, so — unlike `cnn_has_vjp_at` — the entire VJP composes
with *no kink hypotheses*: the only side conditions are the LayerNorm
positivity arguments `0 < ε`.

This file contributes three genuinely new pieces:

  * `layerScale` — per-channel learnable elementwise scale, a diagonal
    linear map; `Differentiable` + a `HasVJP` (`back i = γ i · dy i`),
    its Jacobian derived from `pdiv_mul`/`pdiv_const`/`pdiv_id`.
  * the **ConvNeXt block body** `layerScale ∘ project ∘ gelu ∘ expand ∘
    LayerNorm ∘ depthwise`, everywhere-differentiable, with a pointwise
    VJP built by chaining the piece VJPs through `vjp_comp_at`; and the
    full block `residual (block body)` (identity skip, no post-add act).
  * `convnext_has_vjp_at` / `convnext_has_vjp_at_correct` — a fixed-depth
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
-- § Layer scale (per-channel learnable elementwise scale)
-- ════════════════════════════════════════════════════════════════

/-- **Layer scale** — per-channel learnable elementwise multiply by `γ`.
    `layerScale γ x i = γ i * x i`. A diagonal linear map. -/
noncomputable def layerScale {n : Nat} (γ : Vec n) (x : Vec n) : Vec n :=
  fun i => γ i * x i

/-- `layerScale γ` is differentiable everywhere (diagonal linear). -/
theorem layerScale_differentiable {n : Nat} (γ : Vec n) :
    Differentiable ℝ (layerScale γ) := by
  unfold layerScale; fun_prop

/-- **Jacobian of `layerScale`** — `∂(γ_j x_j)/∂x_i = γ_i δ_{ij}`. -/
theorem pdiv_layerScale {n : Nat} (γ : Vec n) (x : Vec n) (i j : Fin n) :
    pdiv (layerScale γ) x i j = if i = j then γ i else 0 := by
  have h_eq : (layerScale γ) =
      (fun y : Vec n => fun k => (fun _ : Vec n => γ) y k * (fun w : Vec n => w) y k) := by
    funext y k; rfl
  rw [h_eq]
  rw [pdiv_mul (fun _ : Vec n => γ) (fun w : Vec n => w) x
        (differentiableAt_const γ) differentiableAt_id i j]
  rw [pdiv_const, pdiv_id]
  by_cases hij : i = j
  · subst hij; simp
  · rw [if_neg hij, if_neg hij]; ring

/-- **Layer scale VJP**: `back(x, dy)_i = γ i * dy i`. -/
noncomputable def layerScale_has_vjp {n : Nat} (γ : Vec n) :
    HasVJP (layerScale γ) where
  backward := fun _x dy i => γ i * dy i
  correct := by
    intro x dy i
    simp_rw [pdiv_layerScale]
    rw [Finset.sum_eq_single i]
    · rw [if_pos rfl]
    · intro b _ hne; rw [if_neg (fun h => hne h.symm)]; ring
    · intro hp; exact absurd (Finset.mem_univ _) hp

theorem layerScale_has_vjp_correct {n : Nat} (γ : Vec n)
    (x dy : Vec n) (i : Fin n) :
    (layerScale_has_vjp γ).backward x dy i =
    ∑ j : Fin n, pdiv (layerScale γ) x i j * dy j :=
  (layerScale_has_vjp γ).correct x dy i

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
  unfold convNextBlockBody
  have hdw := depthwiseFlat_differentiable (h := h) (w := w) Wdw bdw
  have hln := bnForward_differentiable (c * h * w) εn γn βn hεn
  have hex := flatConv_differentiable (h := h) (w := w) Wex bex
  have hge := gelu_diff (cExp * h * w)
  have hpr := flatConv_differentiable (h := h) (w := w) Wpr bpr
  have hls := layerScale_differentiable γls
  -- layerNormForward = bnForward definitionally
  exact hls.comp (hpr.comp (hge.comp (hex.comp (hln.comp hdw))))

/-- **ConvNeXt block body VJP at a point** — built by chaining the
    everywhere-differentiable piece VJPs through `vjp_comp_at`. Needs only
    `0 < εn` (the LayerNorm positivity); no kink hypotheses since gelu is
    smooth and the rest are linear. -/
noncomputable def convNextBlockBody_has_vjp_at {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) (v : Vec (c * h * w)) :
    HasVJPAt (convNextBlockBody Wdw bdw εn γn βn Wex bex Wpr bpr γls) v := by
  unfold convNextBlockBody
  -- differentiability witnesses (everywhere)
  have hdw := depthwiseFlat_differentiable (h := h) (w := w) Wdw bdw
  have hln := bnForward_differentiable (c * h * w) εn γn βn hεn
  have hex := flatConv_differentiable (h := h) (w := w) Wex bex
  have hge := gelu_diff (cExp * h * w)
  have hpr := flatConv_differentiable (h := h) (w := w) Wpr bpr
  have hls := layerScale_differentiable γls
  -- step1: layerNorm ∘ depthwise
  set D := depthwiseFlat (h := h) (w := w) Wdw bdw with hD
  set LN := layerNormForward (c * h * w) εn γn βn with hLN
  have d_vjp : HasVJPAt D v := (depthwiseFlat_has_vjp (h := h) (w := w) Wdw bdw).toHasVJPAt v
  have ln_vjp : HasVJPAt LN (D v) :=
    (layerNorm_has_vjp (c * h * w) εn γn βn hεn).toHasVJPAt (D v)
  have s1_vjp : HasVJPAt (LN ∘ D) v :=
    vjp_comp_at D LN v (hdw v) (hln (D v)) d_vjp ln_vjp
  have s1_diff : DifferentiableAt ℝ (LN ∘ D) v := (hln (D v)).comp v (hdw v)
  -- step2: expand ∘ (LN ∘ D)
  set EX := flatConv (h := h) (w := w) Wex bex with hEX
  have ex_vjp : HasVJPAt EX (LN (D v)) :=
    (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wex bex)).toHasVJPAt (LN (D v))
  have s2_vjp : HasVJPAt (EX ∘ (LN ∘ D)) v :=
    vjp_comp_at (LN ∘ D) EX v s1_diff (hex (LN (D v))) s1_vjp ex_vjp
  have s2_diff : DifferentiableAt ℝ (EX ∘ (LN ∘ D)) v := (hex (LN (D v))).comp v s1_diff
  -- step3: gelu ∘ (EX ∘ LN ∘ D)
  set GE := gelu (cExp * h * w) with hGE
  have ge_vjp : HasVJPAt GE (EX (LN (D v))) :=
    (gelu_has_vjp (cExp * h * w)).toHasVJPAt (EX (LN (D v)))
  have s3_vjp : HasVJPAt (GE ∘ (EX ∘ (LN ∘ D))) v :=
    vjp_comp_at (EX ∘ (LN ∘ D)) GE v s2_diff (hge (EX (LN (D v)))) s2_vjp ge_vjp
  have s3_diff : DifferentiableAt ℝ (GE ∘ (EX ∘ (LN ∘ D))) v :=
    (hge (EX (LN (D v)))).comp v s2_diff
  -- step4: project ∘ (above)
  set PR := flatConv (h := h) (w := w) Wpr bpr with hPR
  have pr_vjp : HasVJPAt PR (GE (EX (LN (D v)))) :=
    (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wpr bpr)).toHasVJPAt (GE (EX (LN (D v))))
  have s4_vjp : HasVJPAt (PR ∘ (GE ∘ (EX ∘ (LN ∘ D)))) v :=
    vjp_comp_at (GE ∘ (EX ∘ (LN ∘ D))) PR v s3_diff (hpr (GE (EX (LN (D v))))) s3_vjp pr_vjp
  have s4_diff : DifferentiableAt ℝ (PR ∘ (GE ∘ (EX ∘ (LN ∘ D)))) v :=
    (hpr (GE (EX (LN (D v))))).comp v s3_diff
  -- step5: layerScale ∘ (above)
  set LS := layerScale γls with hLS
  exact vjp_comp_at (PR ∘ (GE ∘ (EX ∘ (LN ∘ D)))) LS v s4_diff
    (hls _) s4_vjp ((layerScale_has_vjp γls).toHasVJPAt _)

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
  intro v
  exact DifferentiableAt.add
    ((convNextBlockBody_differentiable Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls) v)
    differentiable_id.differentiableAt

/-- **ConvNeXt block VJP at a point** — `residual_has_vjp_at` on top of the
    block-body VJP. Needs only `0 < εn`. -/
noncomputable def convNextBlock_has_vjp_at {c cExp h w kH kW : Nat}
    (Wdw : DepthwiseKernel c kH kW) (bdw : Vec c)
    (εn : ℝ) (hεn : 0 < εn) (γn βn : ℝ)
    (Wex : Kernel4 cExp c 1 1) (bex : Vec cExp)
    (Wpr : Kernel4 c cExp 1 1) (bpr : Vec c)
    (γls : Vec (c * h * w)) (v : Vec (c * h * w)) :
    HasVJPAt (convNextBlock Wdw bdw εn γn βn Wex bex Wpr bpr γls) v :=
  residual_has_vjp_at (convNextBlockBody Wdw bdw εn γn βn Wex bex Wpr bpr γls) v
    ((convNextBlockBody_differentiable Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls) v)
    (convNextBlockBody_has_vjp_at Wdw bdw εn hεn γn βn Wex bex Wpr bpr γls v)

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

/-- **End-to-end ConvNeXt VJP at a point.** Everything is smooth, so the
    only hypotheses are the four LayerNorm positivity conditions
    (`0 < εst, εn₁, εn₂, εhd`) — no ReLU/maxpool kink conditions, unlike
    `cnn_has_vjp_at`. Chained entirely through `vjp_comp_at` with global
    `Differentiable`/`.toHasVJPAt` lifts. -/
noncomputable def convnext_has_vjp_at
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
      εhd γhd βhd Wd bd) x := by
  unfold convNextForward
  -- s0: stem conv at x
  set ST := flatConv (h := h) (w := w) Wst bst with hST
  have st_diff := flatConv_differentiable (h := h) (w := w) Wst bst
  have st_vjp : HasVJPAt ST x :=
    (hasVJP3_to_hasVJP (conv2d_has_vjp3 Wst bst)).toHasVJPAt x
  -- s1: stem LN ∘ stem conv
  set LNs := layerNormForward (c * h * w) εst γst βst with hLNs
  have lns_diff := bnForward_differentiable (c * h * w) εst γst βst hεst
  have lns_vjp : HasVJPAt LNs (ST x) :=
    (layerNorm_has_vjp (c * h * w) εst γst βst hεst).toHasVJPAt (ST x)
  have s1_vjp : HasVJPAt (LNs ∘ ST) x :=
    vjp_comp_at ST LNs x (st_diff x) (lns_diff (ST x)) st_vjp lns_vjp
  have s1_diff : DifferentiableAt ℝ (LNs ∘ ST) x := (lns_diff (ST x)).comp x (st_diff x)
  -- s2: block₁ ∘ (above)
  set B1 := convNextBlock Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ with hB1
  have b1_diff := convNextBlock_differentiable Wdw₁ bdw₁ εn₁ hεn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
  have b1_vjp : HasVJPAt B1 (LNs (ST x)) :=
    convNextBlock_has_vjp_at Wdw₁ bdw₁ εn₁ hεn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁ (LNs (ST x))
  have s2_vjp : HasVJPAt (B1 ∘ (LNs ∘ ST)) x :=
    vjp_comp_at (LNs ∘ ST) B1 x s1_diff (b1_diff (LNs (ST x))) s1_vjp b1_vjp
  have s2_diff : DifferentiableAt ℝ (B1 ∘ (LNs ∘ ST)) x := (b1_diff (LNs (ST x))).comp x s1_diff
  -- s3: block₂ ∘ (above)
  set B2 := convNextBlock Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ with hB2
  have b2_diff := convNextBlock_differentiable Wdw₂ bdw₂ εn₂ hεn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
  have b2_vjp : HasVJPAt B2 (B1 (LNs (ST x))) :=
    convNextBlock_has_vjp_at Wdw₂ bdw₂ εn₂ hεn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂ (B1 (LNs (ST x)))
  have s3_vjp : HasVJPAt (B2 ∘ (B1 ∘ (LNs ∘ ST))) x :=
    vjp_comp_at (B1 ∘ (LNs ∘ ST)) B2 x s2_diff (b2_diff (B1 (LNs (ST x)))) s2_vjp b2_vjp
  have s3_diff : DifferentiableAt ℝ (B2 ∘ (B1 ∘ (LNs ∘ ST))) x :=
    (b2_diff (B1 (LNs (ST x)))).comp x s2_diff
  -- s4: GAP ∘ (above)
  set P3 := B2 ∘ (B1 ∘ (LNs ∘ ST)) with hP3
  set GAP := globalAvgPoolFlat c h w with hGAP
  have gap_diff := globalAvgPoolFlat_differentiable c h w
  have s4_vjp : HasVJPAt (GAP ∘ P3) x :=
    vjp_comp_at P3 GAP x s3_diff (gap_diff (P3 x)) s3_vjp
      ((globalAvgPoolFlat_has_vjp c h w).toHasVJPAt (P3 x))
  have s4_diff : DifferentiableAt ℝ (GAP ∘ P3) x := (gap_diff (P3 x)).comp x s3_diff
  -- s5: head LN ∘ (above)
  set LNh := layerNormForward c εhd γhd βhd with hLNh
  have lnh_diff := bnForward_differentiable c εhd γhd βhd hεhd
  have s5_vjp : HasVJPAt (LNh ∘ (GAP ∘ P3)) x :=
    vjp_comp_at (GAP ∘ P3) LNh x s4_diff (lnh_diff (GAP (P3 x))) s4_vjp
      ((layerNorm_has_vjp c εhd γhd βhd hεhd).toHasVJPAt (GAP (P3 x)))
  have s5_diff : DifferentiableAt ℝ (LNh ∘ (GAP ∘ P3)) x := (lnh_diff (GAP (P3 x))).comp x s4_diff
  -- s6: dense ∘ (above)
  exact vjp_comp_at (LNh ∘ (GAP ∘ P3)) (dense Wd bd) x s5_diff
    ((dense_differentiable Wd bd) _) s5_vjp
    ((dense_has_vjp Wd bd).toHasVJPAt _)

/-- **Public correctness theorem for `convnext_has_vjp_at`** — exposes the
    witness's `.correct` field: the end-to-end ConvNeXt's backward equals
    the `pdiv`-contracted Jacobian (Jacobian-transpose applied to the
    cotangent). Analogue of `cnn_has_vjp_at_correct`. -/
theorem convnext_has_vjp_at_correct
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
    (convnext_has_vjp_at Wst bst εst γst βst hεst
      Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
      Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
      εhd γhd βhd hεhd Wd bd x).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (convNextForward Wst bst εst γst βst
          Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
          Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
          εhd γhd βhd Wd bd) x i j * dy j :=
  (convnext_has_vjp_at Wst bst εst γst βst hεst
    Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
    Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
    εhd γhd βhd hεhd Wd bd x).correct dy i

end Proofs
