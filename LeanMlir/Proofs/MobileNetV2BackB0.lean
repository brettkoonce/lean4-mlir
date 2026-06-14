import LeanMlir.Proofs.EfficientNetBackB0

/-! # Backward-graph faithfulness for the VERIFIED MobileNetV2 inverted-residual block

The MobileNetV2 peer of `EfficientNetBackB0.lean`: a *backward* StableHLO graph
that denotes the proven VJP of the batched MobileNetV2 inverted-residual block —
`project ∘ depthwise-bn-relu6 ∘ expand-bn-relu6` with the linear-bottleneck skip.

The block is the EfficientNet MBConv body **minus the squeeze-excite stage**, with
**relu6 in place of swish** and the **same** linear-bottleneck `projB` (1×1 conv →
bn, no activation). The project stage and the residual fan-in are reused VERBATIM
from the EfficientNet file (they are global/clean — no smoothness wrinkle).

## The relu6 wrinkle

Unlike swish (smooth everywhere, GLOBAL `swish_has_vjp`), relu6 has a TWO-SIDED kink
(at 0 and at 6), so its VJP is only the *pointwise* `relu6_has_vjp_at`, conditioned
on the smoothness hypothesis `∀ k, x k ≠ 0 ∧ x k ≠ 6` at the pre-activation. Its
per-op backward token is `.selectMid` (the mask `if 0<x<6 then dy else 0`), whose
denotation faithfulness is the already-proven (`rfl`) `selectMid_faithful`
(`StableHLO.lean:794`).

Because relu6's VJP is `_at`, the whole MobileNetV2 stage/body VJP and its backward-
graph faithfulness are stated in the **`_at` / hypothesis-threaded** form (via
`vjp_comp_at`, lifting the global `bnBatchLA`/`batchMap`/conv/depthwise VJPs through
`HasVJP.toHasVJPAt`), NOT the EfficientNet *global* form. The relu6 smoothness
hypothesis at the pre-relu6 activation `bnBatchLA(…)(batchMap(conv)(x))` is threaded
through; the bn/conv/depthwise pieces stay activation-independent (linear) or global.

## Structure

* `cbrB` / `dwbrB` — batched conv/depthwise → bn → **relu6** stages (`cbsB`/`dwbsB`
  with `relu6` for `swish`), with `_at` differentiability + VJP and backward-graph
  faithfulness (`cbrBackBatchedGraph` + `…_faithful`).
* `mnv2BodyB_has_vjp_at` — the SE-less body `projB ∘ dwbrB ∘ cbrB`, composed via
  `vjp_comp_at` over the relu6 smoothness families, with its backward graph
  `mnv2BodyBackBatchedGraph` + `…_faithful`.
* `mnv2ResidBlockBackBatchedGraph_faithful` — the **CAPSTONE**: the whole batched
  MobileNetV2 inverted-residual block backward graph (body + identity skip) denotes
  the proven `residual_has_vjp_at` of the SE-less body. Mirrors the EfficientNet
  `mbResidBlockBackBatchedGraph_faithful` without the `seB` factor, threaded through
  the relu6 smoothness hypotheses.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Batched relu6 stages (`cbsB`/`dwbsB` with `relu6` for `swish`)
-- ════════════════════════════════════════════════════════════════

/-- Batched **conv → bn → relu6** stage (MobileNetV2 expand), at the network layout
    `N·(oc·h·w)`. Identical to EfficientNet's `cbsB` but with `relu6` for `swish`. -/
@[reducible] noncomputable def cbrB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  relu6 (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N (flatConv W b)

/-- Batched **depthwise → bn → relu6** stage (MobileNetV2 depthwise), at the network
    layout. Identical to EfficientNet's `dwbsB` but with `relu6` for `swish`. -/
@[reducible] noncomputable def dwbrB (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * h * w)) → Vec (N * (c * h * w)) :=
  relu6 (N * (c * h * w)) ∘ bnBatchLA N c h w ε γ β ∘ batchMap N (depthwiseFlat W b)

-- ════════════════════════════════════════════════════════════════
-- § Stage `_at`-VJP + differentiability (relu6 smoothness threaded)
-- ════════════════════════════════════════════════════════════════

/-- **Generic relu6-on-batched-bn-stage `_at` VJP.** The relu6 analogue of
    `bnSwishStage_has_vjp`, but `_at` (relu6 only has a pointwise VJP): compose the
    batched-op VJP, the true-BN VJP (both global, lifted via `.toHasVJPAt`), and
    relu6's pointwise VJP at the pre-relu6 activation. -/
noncomputable def bnRelu6Stage_has_vjp_at (N : Nat) {a oc h w : Nat}
    (op : Vec a → Vec (oc * h * w)) (hop : Differentiable ℝ op) (hopv : HasVJP op)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * a))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 6) :
    HasVJPAt (relu6 (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x := by
  -- inner = bnBatchLA ∘ batchMap op (global, lifted to `_at`)
  have hbatch_diff : Differentiable ℝ (batchMap N op) := batchMap_differentiable op hop
  have hbn_diff : Differentiable ℝ (bnBatchLA N oc h w ε γ β) :=
    bnBatchLA_differentiable N oc h w ε hε γ β
  have inner_diff : DifferentiableAt ℝ (bnBatchLA N oc h w ε γ β ∘ batchMap N op) x :=
    (hbn_diff.comp hbatch_diff) x
  have inner_vjp : HasVJPAt (bnBatchLA N oc h w ε γ β ∘ batchMap N op) x :=
    vjp_comp_at (batchMap N op) (bnBatchLA N oc h w ε γ β) x
      (hbatch_diff x) (hbn_diff _)
      ((batchMap_has_vjp op hopv hop).toHasVJPAt x)
      ((bnBatchLA_has_vjp N oc h w ε hε γ β).toHasVJPAt _)
  exact vjp_comp_at (bnBatchLA N oc h w ε γ β ∘ batchMap N op)
    (relu6 (N * (oc * h * w))) x inner_diff
    (relu6_differentiableAt_of_smooth (N * (oc * h * w)) _ h_smooth)
    inner_vjp
    (relu6_has_vjp_at (N * (oc * h * w)) _ h_smooth)

/-- Differentiability of the generic relu6-on-batched-bn-stage at a smooth point. -/
theorem bnRelu6Stage_differentiableAt (N : Nat) {a oc h w : Nat}
    (op : Vec a → Vec (oc * h * w)) (hop : Differentiable ℝ op)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * a))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 6) :
    DifferentiableAt ℝ (relu6 (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x := by
  have inner : DifferentiableAt ℝ (bnBatchLA N oc h w ε γ β ∘ batchMap N op) x :=
    ((bnBatchLA_differentiable N oc h w ε hε γ β).comp (batchMap_differentiable op hop)) x
  exact (relu6_differentiableAt_of_smooth (N * (oc * h * w)) _ h_smooth).comp x inner

/-- `cbrB` (conv-bn-relu6) `_at` VJP at a smooth point. -/
noncomputable def cbrB_has_vjp_at (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 6) :
    HasVJPAt (cbrB N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6Stage_has_vjp_at N (flatConv W b) (flatConv_differentiable W b)
    (flatConv_has_vjp W b) ε hε γ β x h_smooth

theorem cbrB_differentiableAt (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 6) :
    DifferentiableAt ℝ (cbrB N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6Stage_differentiableAt N (flatConv W b) (flatConv_differentiable W b) ε hε γ β x h_smooth

/-- `dwbrB` (depthwise-bn-relu6) `_at` VJP at a smooth point. -/
noncomputable def dwbrB_has_vjp_at (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0 ∧
                     bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 6) :
    HasVJPAt (dwbrB N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6Stage_has_vjp_at N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    (depthwiseFlat_has_vjp W b) ε hε γ β x h_smooth

theorem dwbrB_differentiableAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0 ∧
                     bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 6) :
    DifferentiableAt ℝ (dwbrB N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6Stage_differentiableAt N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    ε hε γ β x h_smooth

-- ════════════════════════════════════════════════════════════════
-- § Batched relu6-stage backward graphs (selectMid for the relu6 kink)
-- ════════════════════════════════════════════════════════════════

/-- Batched **conv → bn → relu6** stage backward graph (MobileNetV2 expand):
    `convBackBatched ∘ bnBatchLABack ∘ selectMid`, each at its cumulative forward
    activation. The relu6 analogue of `cbsBackBatchedGraph` — `.selectMid` (the
    relu6 two-sided-kink mask) replaces `.swishBack`. -/
noncomputable def cbrBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  .convBackBatched (N := N) "%cbrW" W b
    (.bnBatchLABack "%cbrG" "%cbrX" "cbrE" ε γ (batchMap N (flatConv W b) x)
      (.selectMid "%cbrR6" (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x)) e))

theorem cbrBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 6) :
    den (cbrBackBatchedGraph W b ε γ β x e)
      = (cbrB_has_vjp_at N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [cbrBackBatchedGraph, convBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectMid_faithful _ _ h_smooth]
  simp only [cbrB_has_vjp_at, bnRelu6Stage_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt,
    Function.comp_apply]

/-- Batched **depthwise → bn → relu6** stage backward graph (MobileNetV2 depthwise). -/
noncomputable def dwbrBackBatchedGraph {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  .depthwiseBackBatched (N := N) "%dwrW" W b
    (.bnBatchLABack "%dwrG" "%dwrX" "dwrE" ε γ (batchMap N (depthwiseFlat W b) x)
      (.selectMid "%dwrR6" (bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x)) e))

theorem dwbrBackBatchedGraph_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0 ∧
                     bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 6) :
    den (dwbrBackBatchedGraph W b ε γ β x e)
      = (dwbrB_has_vjp_at N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [dwbrBackBatchedGraph, depthwiseBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectMid_faithful _ _ h_smooth]
  simp only [dwbrB_has_vjp_at, bnRelu6Stage_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt,
    Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The SE-less body: `projB ∘ dwbrB ∘ cbrB`
-- ════════════════════════════════════════════════════════════════

/-- The batched MobileNetV2 inverted-residual body's VJP at a smooth point —
    `projB ∘ dwbrB ∘ cbrB` (the EfficientNet MBConv body MINUS `seB`, with relu6
    for swish). Two `vjp_comp_at` chains threading the two relu6 smoothness families:
    (1) `dwbrB ∘ cbrB` over the expand+depthwise relu6 kinks, (2) `projB` (global,
    lifted) on top.

    `h_se` is the expand relu6 smoothness (at the cbrB pre-relu6 activation);
    `h_sd` is the depthwise relu6 smoothness (at the dwbrB pre-relu6 activation,
    fed the cbrB output). -/
noncomputable def mnv2BodyB_has_vjp_at (N : Nat) {c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_se : ∀ k, bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 6) :
    HasVJPAt (projB N (h := h) (w := w) Wp bp εp γp βp ∘
              dwbrB N (h := h) (w := w) Wd bd εd γd βd ∘ cbrB N (h := h) (w := w) We be εe γe βe) x := by
  -- expand stage
  have he_vjp : HasVJPAt (cbrB N (h := h) (w := w) We be εe γe βe) x :=
    cbrB_has_vjp_at N We be εe hεe γe βe x h_se
  have he_diff : DifferentiableAt ℝ (cbrB N (h := h) (w := w) We be εe γe βe) x :=
    cbrB_differentiableAt N We be εe hεe γe βe x h_se
  -- depthwise stage (at the expand output)
  have hd_vjp : HasVJPAt (dwbrB N (h := h) (w := w) Wd bd εd γd βd)
      (cbrB N (h := h) (w := w) We be εe γe βe x) :=
    dwbrB_has_vjp_at N Wd bd εd hεd γd βd _ h_sd
  have hd_diff : DifferentiableAt ℝ (dwbrB N (h := h) (w := w) Wd bd εd γd βd)
      (cbrB N (h := h) (w := w) We be εe γe βe x) :=
    dwbrB_differentiableAt N Wd bd εd hεd γd βd _ h_sd
  -- depthwise ∘ expand
  have hde_vjp : HasVJPAt
      (dwbrB N (h := h) (w := w) Wd bd εd γd βd ∘ cbrB N (h := h) (w := w) We be εe γe βe) x :=
    vjp_comp_at _ _ x he_diff hd_diff he_vjp hd_vjp
  have hde_diff : DifferentiableAt ℝ
      (dwbrB N (h := h) (w := w) Wd bd εd γd βd ∘ cbrB N (h := h) (w := w) We be εe γe βe) x :=
    hd_diff.comp x he_diff
  -- project (global, lifted)
  exact vjp_comp_at _ (projB N (h := h) (w := w) Wp bp εp γp βp) x
    hde_diff
    ((projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp) _)
    hde_vjp
    ((projB_has_vjp N (h := h) (w := w) Wp bp εp hεp γp βp).toHasVJPAt _)

theorem mnv2BodyB_differentiableAt (N : Nat) {c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_se : ∀ k, bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 6) :
    DifferentiableAt ℝ (projB N (h := h) (w := w) Wp bp εp γp βp ∘
              dwbrB N (h := h) (w := w) Wd bd εd γd βd ∘ cbrB N (h := h) (w := w) We be εe γe βe) x := by
  have he_diff : DifferentiableAt ℝ (cbrB N (h := h) (w := w) We be εe γe βe) x :=
    cbrB_differentiableAt N We be εe hεe γe βe x h_se
  have hd_diff : DifferentiableAt ℝ (dwbrB N (h := h) (w := w) Wd bd εd γd βd)
      (cbrB N (h := h) (w := w) We be εe γe βe x) :=
    dwbrB_differentiableAt N Wd bd εd hεd γd βd _ h_sd
  exact ((projB_differentiable N (h := h) (w := w) Wp bp εp hεp γp βp) _).comp x
    (hd_diff.comp x he_diff)

/-- The batched MobileNetV2 body backward graph: the three stage graphs chained at
    their cumulative forward activations (`cbrB⁻¹ ∘ dwbrB⁻¹ ∘ projB⁻¹`). -/
noncomputable def mnv2BodyBackBatchedGraph {N c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  let xE := cbrB N (h := h) (w := w) We be εe γe βe x
  let xD := dwbrB N (h := h) (w := w) Wd bd εd γd βd xE
  cbrBackBatchedGraph We be εe γe βe x
    (dwbrBackBatchedGraph Wd bd εd γd βd xE
      (projBackBatchedGraph Wp bp εp γp βp xD e))

theorem mnv2BodyBackBatchedGraph_faithful {N c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w)))
    (h_se : ∀ k, bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 6) :
    den (mnv2BodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp x e)
      = (mnv2BodyB_has_vjp_at N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wp bp εp hεp γp βp x h_se h_sd).backward (den e) := by
  rw [mnv2BodyBackBatchedGraph, cbrBackBatchedGraph_faithful (hε := hεe) (h_smooth := h_se),
      dwbrBackBatchedGraph_faithful (hε := hεd) (h_smooth := h_sd),
      projBackBatchedGraph_faithful (hε := hεp)]
  simp only [mnv2BodyB_has_vjp_at, vjp_comp_at, HasVJP.toHasVJPAt, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § Capstone: the whole batched MobileNetV2 inverted-residual block
-- ════════════════════════════════════════════════════════════════

/-- The whole batched MobileNetV2 inverted-residual block backward graph
    (body + identity skip). -/
noncomputable def mnv2ResidBlockBackBatchedGraph {N c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    (x dy : Vec (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  residualBackGraph
    (mnv2BodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp
      x (.operand "%dy" dy)) dy

/-- **CAPSTONE — the whole batched MobileNetV2 inverted-residual block: backward
    graph ↔ the proven VJP.** The three batched stage backward graphs
    (`cbrB`/`dwbrB`/`projB`) chained at their forward activations + the identity
    skip, proven equal to `residual_has_vjp_at` of the SE-less body
    `projB ∘ dwbrB ∘ cbrB`. The MobileNetV2 analogue of the EfficientNet
    `mbResidBlockBackBatchedGraph_faithful`, without the `seB` factor, threaded
    through the relu6 smoothness hypotheses (relu6's VJP is only `_at`).

    Uses `residualBackGraph_faithful`'s `_at`-form analogue: the residual fan-in
    backward (body cotangent + the identity skip's verbatim `%dy`), with the body
    hypothesis discharged by `mnv2BodyBackBatchedGraph_faithful`. -/
theorem mnv2ResidBlockBackBatchedGraph_faithful {N c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x dy : Vec (N * (c * h * w)))
    (h_se : ∀ k, bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 6) :
    den (mnv2ResidBlockBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp x dy)
      = (residual_has_vjp_at
          (projB N (h := h) (w := w) Wp bp εp γp βp ∘
            dwbrB N (h := h) (w := w) Wd bd εd γd βd ∘ cbrB N (h := h) (w := w) We be εe γe βe)
          x
          (mnv2BodyB_differentiableAt N We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp x h_se h_sd)
          (mnv2BodyB_has_vjp_at N We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp x h_se h_sd)).backward dy := by
  -- the residual `_at` backward = body backward at cotangent dy + identity skip dy
  have hbody : den (mnv2BodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp
        x (.operand "%dy" dy))
      = (mnv2BodyB_has_vjp_at N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wp bp εp hεp γp βp x h_se h_sd).backward dy :=
    mnv2BodyBackBatchedGraph_faithful We be εe hεe γe βe Wd bd εd hεd γd βd
      Wp bp εp hεp γp βp x (.operand "%dy" dy) h_se h_sd
  funext i
  have hsum : den (mnv2ResidBlockBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp x dy) i
      = den (mnv2BodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp
              x (.operand "%dy" dy)) i + dy i := rfl
  rw [hsum, hbody]
  rfl

end Proofs.StableHLO
