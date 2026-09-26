import LeanMlir.Proofs.Foundation.BatchedStageLayers

/-! # Backward-graph faithfulness for the batched MobileNetV2 inverted-residual block

The MobileNetV2 peer of `EfficientNetBackB0.lean`: a *backward* StableHLO graph
that denotes the proven VJP of the batched MobileNetV2 inverted-residual block —
`project ∘ depthwise-bn-relu6 ∘ expand-bn-relu6` with the linear-bottleneck skip.

The block is the EfficientNet MBConv body **minus the squeeze-excite stage**, with
**relu6 in place of swish** and the **same** linear-bottleneck `projB` (1×1 conv →
bn, no activation). The stages come from `BatchedStageLayers` (`cbrB`, `dwbrB`,
`dwbrBstrided` and their `CertLayer`s `cbrLayer`, `dwbrLayer`, `dwbrStridedLayer`, `projLayer`)
and `BatchedStages` (`projB`); the residual fan-in's backward graph is
`residualBackGraph` (`BatchedBackLinks`). The project stage and the fan-in need no
smoothness hypothesis.

## The relu6 wrinkle

Unlike swish (smooth everywhere, GLOBAL `swishHasVJP`), relu6 has a TWO-SIDED kink
(at 0 and at 6), so its VJP is only the *pointwise* `relu6HasVJPAt`, conditioned
on the smoothness hypothesis `∀ k, x k ≠ 0 ∧ x k ≠ 6` at the pre-activation. Its
per-op backward token is `.selectMid` (the mask `if 0<x<6 then dy else 0`), whose
denotation faithfulness is the already-proven (`rfl`) `StableHLO.selectMid_faithful`.

Because relu6's VJP is `_at`, the whole MobileNetV2 stage/body VJP and its backward-
graph faithfulness are stated in the **`_at` / hypothesis-threaded** form (via
`vjpCompAt`, lifting the global `bnBatchLA`/`batchMap`/conv/depthwise VJPs through
`HasVJP.toHasVJPAt`), NOT the EfficientNet *global* form. The relu6 smoothness
hypothesis at the pre-relu6 activation `bnBatchLA(…)(batchMap(conv)(x))` is threaded
through; the bn/conv/depthwise pieces stay activation-independent (linear) or global.

## Structure

* `mnv2BodyLayer` / `mnv2DownBodyLayer` — the stride-1 and stride-2 bodies as `CertLayer`s,
  composed from the stage layers with `CertLayer.comp`; each body VJP, differentiability lemma
  and graph `_faithful` below is that layer's `vjp` / `diff` / `faithful`.
* `mnv2BodyBHasVJPAt` — the SE-less body `projB ∘ dwbrB ∘ cbrB` at the relu6
  smoothness families, with its backward graph `mnv2BodyBackBatchedGraph` + `…_faithful`.
  It takes `ic` and `oc` separately, so it also covers the stride-1 bodies with `ic ≠ oc`
  (`b11`, 64 → 96, and `b17`, 160 → 320; `MobileNetV2FullB.lean`'s `mnv2ExpOnlyB`).
* `mnv2ResidBlockBackBatchedGraph_faithful` — the whole batched
  MobileNetV2 inverted-residual block backward graph (body + identity skip) denotes
  the proven `residualHasVJPAt` of the SE-less body. Mirrors the EfficientNet
  `mbResidBlockBackBatchedGraph_faithful` without the `seB` factor, threaded through
  the relu6 smoothness hypotheses; `CertLayer.residual mnv2BodyLayer`'s `faithful`.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The SE-less body: `projB ∘ dwbrB ∘ cbrB`
-- ════════════════════════════════════════════════════════════════

/-- The SE-less body as a `CertLayer`: `cbrLayer ; dwbrLayer ; projLayer`, left-nested so its
    `fwd` is `projB ∘ (dwbrB ∘ cbrB)`, the association the stated types use. Its `ok` is the
    expand and depthwise relu6 clauses; `projLayer` contributes `True`. -/
noncomputable def mnv2BodyLayer (N : Nat) {ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) :=
  ((cbrLayer N (h := h) (w := w) We be εe hεe γe βe).comp (dwbrLayer N Wd bd εd hεd γd βd)).comp
    (projLayer N Wp bp εp hεp γp βp)

/-- The batched MobileNetV2 inverted-residual body's VJP at a smooth point —
    `projB ∘ dwbrB ∘ cbrB` (the EfficientNet MBConv body MINUS `seB`, with relu6
    for swish), `mnv2BodyLayer`'s VJP.

    `h_se` is the expand relu6 smoothness (at the cbrB pre-relu6 activation);
    `h_sd` is the depthwise relu6 smoothness (at the dwbrB pre-relu6 activation,
    fed the cbrB output). -/
noncomputable def mnv2BodyBHasVJPAt (N : Nat) {ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_se : ∀ k, bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 6) :
    HasVJPAt (projB N (h := h) (w := w) Wp bp εp γp βp ∘
              dwbrB N (h := h) (w := w) Wd bd εd γd βd ∘ cbrB N (h := h) (w := w) We be εe γe βe) x :=
  (mnv2BodyLayer N (h := h) (w := w) We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp).vjp x
    ⟨⟨h_se, h_sd⟩, trivial⟩

theorem mnv2BodyB_differentiableAt (N : Nat) {ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_se : ∀ k, bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 6) :
    DifferentiableAt ℝ (projB N (h := h) (w := w) Wp bp εp γp βp ∘
              dwbrB N (h := h) (w := w) Wd bd εd γd βd ∘ cbrB N (h := h) (w := w) We be εe γe βe) x :=
  (mnv2BodyLayer N (h := h) (w := w) We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp).diff x
    ⟨⟨h_se, h_sd⟩, trivial⟩

/-- The batched MobileNetV2 body backward graph: the three stage graphs chained at
    their cumulative forward activations (`cbrB⁻¹ ∘ dwbrB⁻¹ ∘ projB⁻¹`). -/
noncomputable def mnv2BodyBackBatchedGraph {N ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  let xE := cbrB N (h := h) (w := w) We be εe γe βe x
  let xD := dwbrB N (h := h) (w := w) Wd bd εd γd βd xE
  cbrBackBatchedGraph We be εe γe βe x
    (dwbrBackBatchedGraph Wd bd εd γd βd xE
      (projBackBatchedGraph Wp bp εp γp βp xD e))

theorem mnv2BodyBackBatchedGraph_faithful {N ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w)))
    (h_se : ∀ k, bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 6) :
    den (mnv2BodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp x e)
      = (mnv2BodyBHasVJPAt N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wp bp εp hεp γp βp x h_se h_sd).backward (den e) :=
  (mnv2BodyLayer N (h := h) (w := w) We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp).faithful
    x ⟨⟨h_se, h_sd⟩, trivial⟩ e

-- ════════════════════════════════════════════════════════════════
-- § The DOWNSAMPLE body: `projB ∘ dwbrBstrided ∘ cbrB` (strided, NO residual)
-- ════════════════════════════════════════════════════════════════

/-- The downsample body as a `CertLayer`: `cbrLayer` at `2h × 2w`, then `dwbrStridedLayer` and
    `projLayer` at `h × w`. -/
noncomputable def mnv2DownBodyLayer (N : Nat) {ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  ((cbrLayer N (h := 2 * h) (w := 2 * w) We be εe hεe γe βe).comp
    (dwbrStridedLayer N (h := h) (w := w) Wd bd εd hεd γd βd)).comp
    (projLayer N (h := h) (w := w) Wp bp εp hεp γp βp)

/-- The batched MobileNetV2 DOWNSAMPLE inverted-residual body's VJP at a smooth
    point — `projB ∘ dwbrBstrided ∘ cbrB`, the stride-2 analogue of
    `mnv2BodyBHasVJPAt` (swaps the stride-1 `dwbrB` depthwise stage for the
    STRIDED `dwbrBstrided`). The expand `cbrB` runs at the larger `2h×2w` (1×1
    conv keeps spatial), the strided depthwise then halves spatial to `h×w`; project
    runs at `h×w`. NO residual (spatial/channels change), so this is the body alone:
    `mnv2DownBodyLayer`'s VJP.

    `h_se` is the expand relu6 smoothness (at the cbrB pre-relu6 activation, at
    `2h×2w`); `h_sd` is the strided-depthwise relu6 smoothness (at the dwbrBstrided
    pre-relu6 activation, fed the cbrB output). -/
noncomputable def mnv2DownBodyBHasVJPAt (N : Nat) {ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_se : ∀ k, bnBatchLA N mid (2 * h) (2 * w) εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid (2 * h) (2 * w) εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseStride2FlatXla Wd bd) (cbrB N (h := 2 * h) (w := 2 * w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseStride2FlatXla Wd bd) (cbrB N (h := 2 * h) (w := 2 * w) We be εe γe βe x)) k ≠ 6) :
    HasVJPAt (projB N (h := h) (w := w) Wp bp εp γp βp ∘
              dwbrBstrided N (h := h) (w := w) Wd bd εd γd βd ∘
              cbrB N (h := 2 * h) (w := 2 * w) We be εe γe βe) x :=
  (mnv2DownBodyLayer N (h := h) (w := w) We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp).vjp
    x ⟨⟨h_se, h_sd⟩, trivial⟩

theorem mnv2DownBodyB_differentiableAt (N : Nat) {ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_se : ∀ k, bnBatchLA N mid (2 * h) (2 * w) εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid (2 * h) (2 * w) εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseStride2FlatXla Wd bd) (cbrB N (h := 2 * h) (w := 2 * w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseStride2FlatXla Wd bd) (cbrB N (h := 2 * h) (w := 2 * w) We be εe γe βe x)) k ≠ 6) :
    DifferentiableAt ℝ (projB N (h := h) (w := w) Wp bp εp γp βp ∘
              dwbrBstrided N (h := h) (w := w) Wd bd εd γd βd ∘
              cbrB N (h := 2 * h) (w := 2 * w) We be εe γe βe) x :=
  (mnv2DownBodyLayer N (h := h) (w := w) We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp).diff
    x ⟨⟨h_se, h_sd⟩, trivial⟩

/-- The batched MobileNetV2 downsample body backward graph: the three stage graphs
    chained at their cumulative forward activations
    (`cbrB⁻¹ ∘ dwbrBstrided⁻¹ ∘ projB⁻¹`). Stride-2 analogue of
    `mnv2BodyBackBatchedGraph` (strided depthwise stage graph, no residual). -/
noncomputable def mnv2DownBodyBackBatchedGraph {N ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  let xE := cbrB N (h := 2 * h) (w := 2 * w) We be εe γe βe x
  let xD := dwbrBstrided N (h := h) (w := w) Wd bd εd γd βd xE
  cbrBackBatchedGraph We be εe γe βe x
    (dwbrBstridedBackBatchedGraph Wd bd εd γd βd xE
      (projBackBatchedGraph Wp bp εp γp βp xD e))

/-- **CAPSTONE — the batched MobileNetV2 DOWNSAMPLE inverted-residual body: backward
    graph ↔ the proven `mnv2DownBodyBHasVJPAt`.** The three batched stage backward
    graphs (`cbrB`/`dwbrBstrided`/`projB`) chained at their forward activations,
    proven equal to the downsample-body VJP. The stride-2 analogue of
    `mnv2BodyBackBatchedGraph_faithful` (no residual skip — the downsample block
    changes spatial/channels, so the body alone is the block), threaded through the
    two relu6 smoothness hypotheses (relu6's VJP is only `_at`). The MobileNetV2
    relu6 peer of the EfficientNet `mbDownBodyBackBatchedGraph_faithful`; it is
    `mnv2DownBodyLayer`'s `faithful`. -/
theorem mnv2DownBodyBackBatchedGraph_faithful {N ic mid oc h w kHd kWd : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w)))
    (h_se : ∀ k, bnBatchLA N mid (2 * h) (2 * w) εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid (2 * h) (2 * w) εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseStride2FlatXla Wd bd) (cbrB N (h := 2 * h) (w := 2 * w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseStride2FlatXla Wd bd) (cbrB N (h := 2 * h) (w := 2 * w) We be εe γe βe x)) k ≠ 6) :
    den (mnv2DownBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp x e)
      = (mnv2DownBodyBHasVJPAt N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wp bp εp hεp γp βp x h_se h_sd).backward (den e) :=
  (mnv2DownBodyLayer N (h := h) (w := w) We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp).faithful
    x ⟨⟨h_se, h_sd⟩, trivial⟩ e

-- ════════════════════════════════════════════════════════════════
-- § Capstone: the whole batched MobileNetV2 inverted-residual block
-- ════════════════════════════════════════════════════════════════

/-- The whole batched MobileNetV2 inverted-residual block backward graph
    (body + identity skip). -/
noncomputable def mnv2ResidBlockBackBatchedGraph {N c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (ecot : SHlo (N * (c * h * w))) : SHlo (N * (c * h * w)) :=
  residualBackGraph
    (mnv2BodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp
      x ecot) ecot

/-- **CAPSTONE — the whole batched MobileNetV2 inverted-residual block: backward
    graph ↔ the proven VJP.** The three batched stage backward graphs
    (`cbrB`/`dwbrB`/`projB`) chained at their forward activations + the identity
    skip, proven equal to `residualHasVJPAt` of the SE-less body
    `projB ∘ dwbrB ∘ cbrB`. The MobileNetV2 analogue of the EfficientNet
    `mbResidBlockBackBatchedGraph_faithful`, without the `seB` factor, threaded
    through the relu6 smoothness hypotheses (relu6's VJP is only `_at`).

    It is `CertLayer.residual mnv2BodyLayer`'s `faithful`: the residual fan-in backward
    (body cotangent + the identity skip's verbatim `%dy`). -/
theorem mnv2ResidBlockBackBatchedGraph_faithful {N c mid h w kHd kWd : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (ecot : SHlo (N * (c * h * w)))
    (h_se : ∀ k, bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 0 ∧
                 bnBatchLA N mid h w εe γe βe (batchMap N (flatConv We be) x) k ≠ 6)
    (h_sd : ∀ k, bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 0 ∧
                 bnBatchLA N mid h w εd γd βd
                    (batchMap N (depthwiseFlat Wd bd) (cbrB N (h := h) (w := w) We be εe γe βe x)) k ≠ 6) :
    den (mnv2ResidBlockBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wp bp εp γp βp x ecot)
      = (residualHasVJPAt
          (projB N (h := h) (w := w) Wp bp εp γp βp ∘
            dwbrB N (h := h) (w := w) Wd bd εd γd βd ∘ cbrB N (h := h) (w := w) We be εe γe βe)
          x
          (mnv2BodyB_differentiableAt N We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp x h_se h_sd)
          (mnv2BodyBHasVJPAt N We be εe hεe γe βe Wd bd εd hεd γd βd Wp bp εp hεp γp βp x h_se h_sd)).backward (den ecot) :=
  (CertLayer.residual (mnv2BodyLayer N (h := h) (w := w) We be εe hεe γe βe Wd bd εd hεd γd βd
    Wp bp εp hεp γp βp)).faithful x ⟨⟨h_se, h_sd⟩, trivial⟩ ecot

end Proofs.StableHLO
