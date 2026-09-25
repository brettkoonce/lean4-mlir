import LeanMlir.Proofs.Foundation.BatchedBackLinks
import LeanMlir.Proofs.Foundation.CertifiedChain

/-! # Batched activation stages — conv/depthwise → true BN → relu or relu6, as `CertLayer`s

The kinked stages the ResNets, MobileNetV2 and MobileNetV4 are built from, at the batched index
`N·(c·h·w)`. Each comes with its `_at` VJP (certified where the pre-activation misses the kink),
its backward graph (`.selectPos` for relu, `.selectMid` for relu6) with `_faithful`, and a
`CertLayer` bundling the two. The smooth stages (swish, projection) are in `BatchedStages`.

| stage | forward | `CertLayer` |
|---|---|---|
| conv → BN → relu6; depthwise → BN → relu6 (stride 1 / XLA-`SAME` stride 2) | `cbrB`, `dwbrB`, `dwbrBstrided` | `cbrLayer`, `dwbrLayer`, `dwbrStridedLayer` |
| 1×1 projection → BN | `projB` | `projLayer` |
| conv → BN → relu; strided conv → BN → relu | `cbReluB`, `cbReluStridedB` | `cbReluLayer`, `cbReluStridedLayer` |
| strided projection → BN (the downsample skip) | `projStridedB` | `projStridedLayer` |
-/

namespace Proofs.StableHLO

open scoped BigOperators

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

/-- Batched **STRIDE-2 depthwise → bn → relu6** stage (MobileNetV2 downsample
    depthwise), at the network layout. The stride-2 analogue of `dwbrB`: maps the
    larger input spatial `c·(2h)·(2w)` to the output spatial `c·h·w`. Identical to
    EfficientNet's `dwbsSB` but with `relu6` for `swish`. -/
@[reducible] noncomputable def dwbrBstrided (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c) :
    Vec (N * (c * (2 * h) * (2 * w))) → Vec (N * (c * h * w)) :=
  relu6 (N * (c * h * w)) ∘ bnBatchLA N c h w ε γ β ∘ batchMap N (depthwiseStride2FlatXla W b)

-- ════════════════════════════════════════════════════════════════
-- § Stage `_at`-VJP + differentiability (relu6 smoothness threaded)
-- ════════════════════════════════════════════════════════════════

/-- **Generic relu6-on-batched-bn-stage `_at` VJP.** The relu6 analogue of
    `bnSwishStageHasVJP`, but `_at` (relu6 only has a pointwise VJP): compose the
    batched-op VJP, the true-BN VJP (both global, lifted via `.toHasVJPAt`), and
    relu6's pointwise VJP at the pre-relu6 activation. -/
noncomputable def bnRelu6StageHasVJPAt (N : Nat) {a oc h w : Nat}
    (op : Vec a → Vec (oc * h * w)) (hop : Differentiable ℝ op) (hopv : HasVJP op)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * a))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 6) :
    HasVJPAt (relu6 (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x :=
  stageHasVJPAt (batchMap N op) (bnBatchLA N oc h w ε γ β) (relu6 (N * (oc * h * w))) x
    (batchMap_differentiable op hop) (batchMapHasVJP op hopv hop)
    (bnBatchLA_differentiable N oc h w ε hε γ β) (bnBatchLAHasVJP N oc h w ε hε γ β)
    (relu6_differentiableAt_of_smooth (N * (oc * h * w)) _ h_smooth)
    (relu6HasVJPAt (N * (oc * h * w)) _ h_smooth)

/-- Differentiability of the generic relu6-on-batched-bn-stage at a smooth point. -/
theorem bnRelu6Stage_differentiableAt (N : Nat) {a oc h w : Nat}
    (op : Vec a → Vec (oc * h * w)) (hop : Differentiable ℝ op)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * a))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 6) :
    DifferentiableAt ℝ (relu6 (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x := by
  fun_prop (disch := assumption)

/-- `cbrB` (conv-bn-relu6) `_at` VJP at a smooth point. -/
noncomputable def cbrBHasVJPAt (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 6) :
    HasVJPAt (cbrB N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6StageHasVJPAt N (flatConv W b) (flatConv_differentiable W b)
    (flatConvHasVJP W b) ε hε γ β x h_smooth

theorem cbrB_differentiableAt (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 6) :
    DifferentiableAt ℝ (cbrB N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6Stage_differentiableAt N (flatConv W b) (flatConv_differentiable W b) ε hε γ β x h_smooth

/-- `dwbrB` (depthwise-bn-relu6) `_at` VJP at a smooth point. -/
noncomputable def dwbrBHasVJPAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0 ∧
                     bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 6) :
    HasVJPAt (dwbrB N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6StageHasVJPAt N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    (depthwiseFlatHasVJP W b) ε hε γ β x h_smooth

theorem dwbrB_differentiableAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0 ∧
                     bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 6) :
    DifferentiableAt ℝ (dwbrB N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6Stage_differentiableAt N (depthwiseFlat W b) (depthwiseFlat_differentiable W b)
    ε hε γ β x h_smooth

/-- `dwbrBstrided` (STRIDE-2 depthwise-bn-relu6) `_at` VJP at a smooth point. The
    stride-2 analogue of `dwbrBHasVJPAt`: lifts `depthwiseStride2FlatXlaHasVJP`
    (the strided per-channel conv input-VJP) through the generic relu6-bn stage. -/
noncomputable def dwbrBstridedHasVJPAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w))))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2FlatXla W b) x) k ≠ 0 ∧
                     bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2FlatXla W b) x) k ≠ 6) :
    HasVJPAt (dwbrBstrided N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6StageHasVJPAt N (depthwiseStride2FlatXla W b) (depthwiseStride2FlatXla_differentiable W b)
    (depthwiseStride2FlatXlaHasVJP W b) ε hε γ β x h_smooth

theorem dwbrBstrided_differentiableAt (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w))))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2FlatXla W b) x) k ≠ 0 ∧
                     bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2FlatXla W b) x) k ≠ 6) :
    DifferentiableAt ℝ (dwbrBstrided N (h := h) (w := w) W b ε γ β) x :=
  bnRelu6Stage_differentiableAt N (depthwiseStride2FlatXla W b) (depthwiseStride2FlatXla_differentiable W b)
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
    (.bnBatchLABack "%cbrG" "%cbr_x" "cbrE" ε γ (batchMap N (flatConv W b) x)
      (.selectMid "%cbrR6" (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x)) e))

theorem cbrBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0 ∧
                     bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 6) :
    den (cbrBackBatchedGraph W b ε γ β x e)
      = (cbrBHasVJPAt N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [cbrBackBatchedGraph, convBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectMid_faithful _ _ h_smooth]
  simp only [cbrBHasVJPAt, bnRelu6StageHasVJPAt, stageHasVJPAt, vjpCompAt_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

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
      = (dwbrBHasVJPAt N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [dwbrBackBatchedGraph, depthwiseBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectMid_faithful _ _ h_smooth]
  simp only [dwbrBHasVJPAt, bnRelu6StageHasVJPAt, stageHasVJPAt, vjpCompAt_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

/-- Batched **STRIDE-2 depthwise → bn → relu6** stage backward graph (MobileNetV2
    downsample depthwise). The stride-2 analogue of `dwbrBackBatchedGraph`: the
    bn/relu6 run at the OUTPUT spatial `h×w`, then `depthwiseStridedBackBatched` maps
    the bn-cotangent back to the larger input `c·(2h)·(2w)` (zero-upsample +
    reversed-kernel per-channel depthwise). The relu6 back is `.selectMid` as before. -/
noncomputable def dwbrBstridedBackBatchedGraph {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (e : SHlo (N * (c * h * w))) :
    SHlo (N * (c * (2 * h) * (2 * w))) :=
  .depthwiseStridedXlaBackBatched (N := N) "%dwsrW" W b
    (.bnBatchLABack "%dwsrG" "%dwsrX" "dwsrE" ε γ (batchMap N (depthwiseStride2FlatXla W b) x)
      (.selectMid "%dwsrR6" (bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2FlatXla W b) x)) e))

theorem dwbrBstridedBackBatchedGraph_faithful {N c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (c * (2 * h) * (2 * w)))) (e : SHlo (N * (c * h * w)))
    (h_smooth : ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2FlatXla W b) x) k ≠ 0 ∧
                     bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2FlatXla W b) x) k ≠ 6) :
    den (dwbrBstridedBackBatchedGraph W b ε γ β x e)
      = (dwbrBstridedHasVJPAt N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [dwbrBstridedBackBatchedGraph, depthwiseStridedXlaBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectMid_faithful _ _ h_smooth]
  simp only [dwbrBstridedHasVJPAt, bnRelu6StageHasVJPAt, stageHasVJPAt, vjpCompAt_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

-- ════════════════════════════════════════════════════════════════
-- § The stages as `CertLayer`s
-- ════════════════════════════════════════════════════════════════

/-- The conv → bn → relu6 stage as a `CertLayer`, certified where its pre-relu6 activation misses
    both kinks. The kernel extent is a binder, so the same layer is a 1×1 or a 3×3. -/
noncomputable def cbrLayer (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) where
  fwd := cbrB N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0 ∧
                      bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 6
  diff := fun x hx => cbrB_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => cbrBHasVJPAt N W b ε hε γ β x hx
  graph := fun x e => cbrBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => cbrBackBatchedGraph_faithful W b ε hε γ β x e hx

/-- The depthwise → bn → relu6 stage as a `CertLayer`. -/
noncomputable def dwbrLayer (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := dwbrB N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 0 ∧
                      bnBatchLA N c h w ε γ β (batchMap N (depthwiseFlat W b) x) k ≠ 6
  diff := fun x hx => dwbrB_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => dwbrBHasVJPAt N W b ε hε γ β x hx
  graph := fun x e => dwbrBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => dwbrBackBatchedGraph_faithful W b ε hε γ β x e hx

/-- The stride-2 depthwise → bn → relu6 stage as a `CertLayer`: `2h × 2w` in, `h × w` out. -/
noncomputable def dwbrStridedLayer (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * (2 * h) * (2 * w))) (N * (c * h * w)) where
  fwd := dwbrBstrided N (h := h) (w := w) W b ε γ β
  ok := fun x =>
    ∀ k, bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2FlatXla W b) x) k ≠ 0 ∧
         bnBatchLA N c h w ε γ β (batchMap N (depthwiseStride2FlatXla W b) x) k ≠ 6
  diff := fun x hx => dwbrBstrided_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => dwbrBstridedHasVJPAt N W b ε hε γ β x hx
  graph := fun x e => dwbrBstridedBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => dwbrBstridedBackBatchedGraph_faithful W b ε hε γ β x e hx

/-- The conv → bn stage (`projB`, no activation) as a `CertLayer`. Globally certified
    (`ok = True`): with no activation there is no kink. -/
noncomputable def projLayer (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) where
  fwd := projB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (projB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (projBHasVJP N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => projBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => projBackBatchedGraph_faithful W b ε hε γ β x e

-- ════════════════════════════════════════════════════════════════
-- § Batched relu stage (`cbrB` with `relu` for `relu6`)
-- ════════════════════════════════════════════════════════════════

/-- Batched **conv → bn → relu** stage (ResNet basic-block first stage), at the
    network layout `N·(oc·h·w)`. The relu analogue of MobileNetV2's `cbrB`
    (relu for relu6). -/
@[reducible] noncomputable def cbReluB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * h * w)) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N (flatConv W b)

-- ════════════════════════════════════════════════════════════════
-- § Stage `_at`-VJP + differentiability (relu smoothness threaded)
-- ════════════════════════════════════════════════════════════════

/-- **Generic relu-on-batched-bn-stage `_at` VJP.** The relu analogue of
    `bnRelu6StageHasVJPAt` (and of `bnSwishStageHasVJP`, but `_at` — relu
    only has a pointwise VJP): compose the batched-op VJP, the true-BN VJP (both
    global, lifted via `.toHasVJPAt`), and relu's pointwise VJP at the pre-relu
    activation. The smoothness hypothesis is the one-sided `≠ 0`. -/
noncomputable def bnReluStageHasVJPAt (N : Nat) {a oc h w : Nat}
    (op : Vec a → Vec (oc * h * w)) (hop : Differentiable ℝ op) (hopv : HasVJP op)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * a))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 0) :
    HasVJPAt (relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x :=
  stageHasVJPAt (batchMap N op) (bnBatchLA N oc h w ε γ β) (relu (N * (oc * h * w))) x
    (batchMap_differentiable op hop) (batchMapHasVJP op hopv hop)
    (bnBatchLA_differentiable N oc h w ε hε γ β) (bnBatchLAHasVJP N oc h w ε hε γ β)
    (relu_differentiableAt_of_smooth (N * (oc * h * w)) _ h_smooth)
    (reluHasVJPAt (N * (oc * h * w)) _ h_smooth)

/-- Differentiability of the generic relu-on-batched-bn-stage at a smooth point. -/
theorem bnReluStage_differentiableAt (N : Nat) {a oc h w : Nat}
    (op : Vec a → Vec (oc * h * w)) (hop : Differentiable ℝ op)
    (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) (x : Vec (N * a))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N op x) k ≠ 0) :
    DifferentiableAt ℝ (relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N op) x := by
  fun_prop (disch := assumption)

/-- `cbReluB` (conv-bn-relu) `_at` VJP at a smooth point. -/
noncomputable def cbReluBHasVJPAt (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0) :
    HasVJPAt (cbReluB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStageHasVJPAt N (flatConv W b) (flatConv_differentiable W b)
    (flatConvHasVJP W b) ε hε γ β x h_smooth

theorem cbReluB_differentiableAt (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0) :
    DifferentiableAt ℝ (cbReluB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_differentiableAt N (flatConv W b) (flatConv_differentiable W b) ε hε γ β x h_smooth

-- ════════════════════════════════════════════════════════════════
-- § Batched relu-stage backward graph (selectPos for the relu kink)
-- ════════════════════════════════════════════════════════════════

/-- Batched **conv → bn → relu** stage backward graph (ResNet basic-block stage 1):
    `convBackBatched ∘ bnBatchLABack ∘ selectPos`, each at its cumulative forward
    activation. The relu analogue of MobileNetV2's `cbrBackBatchedGraph` —
    `.selectPos` (the relu one-sided-kink mask) replaces `.selectMid`. -/
noncomputable def cbReluBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  .convBackBatched (N := N) "%cbrW" W b
    (.bnBatchLABack "%cbrG" "%cbr_x" "cbrE" ε γ (batchMap N (flatConv W b) x)
      (.selectPos "%cbrR" (bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x)) e))

theorem cbReluBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0) :
    den (cbReluBackBatchedGraph W b ε γ β x e)
      = (cbReluBHasVJPAt N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [cbReluBackBatchedGraph, convBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectPos_faithful _ _ h_smooth]
  simp only [cbReluBHasVJPAt, bnReluStageHasVJPAt, stageHasVJPAt, vjpCompAt_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

/-- The conv → bn → relu stage as a `CertLayer`, certified where its pre-relu activation misses 0.
    The kernel extent is a binder, so the same layer is a 1×1 or a 3×3. -/
noncomputable def cbReluLayer (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) where
  fwd := cbReluB N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConv W b) x) k ≠ 0
  diff := fun x hx => cbReluB_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => cbReluBHasVJPAt N W b ε hε γ β x hx
  graph := fun x e => cbReluBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => cbReluBackBatchedGraph_faithful W b ε hε γ β x e hx

/-- `cbReluLayer`'s forward is `cbReluB`. -/
theorem cbReluLayer_fwd_apply (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (v : Vec (N * (ic * h * w))) :
    (cbReluLayer (h := h) (w := w) N W b ε hε γ β).fwd v
      = cbReluB N (h := h) (w := w) W b ε γ β v := rfl

-- ════════════════════════════════════════════════════════════════
-- § DOWNSAMPLE BLOCK — `relu ∘ residualProj(proj, F_s)`
--   stage 1: STRIDED conv-bn-relu (`cbReluStridedB`, uses `convStridedBackBatched`)
-- ════════════════════════════════════════════════════════════════

/-- Batched **STRIDE-2 conv → bn → relu** stage (downsample basic-block first
    stage), at the network layout `N·(oc·h·w)` ← `N·(ic·(2h)·(2w))`. The strided
    sibling of `cbReluB` (`flatConvStride2` for `flatConv`); halves spatial. -/
@[reducible] noncomputable def cbReluStridedB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  relu (N * (oc * h * w)) ∘ bnBatchLA N oc h w ε γ β ∘ batchMap N (flatConvStride2 W b)

/-- `cbReluStridedB` (strided conv-bn-relu) `_at` VJP at a smooth point. The strided
    sibling of `cbReluBHasVJPAt` (`flatConvStride2` for `flatConv`). -/
noncomputable def cbReluStridedBHasVJPAt (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x) k ≠ 0) :
    HasVJPAt (cbReluStridedB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStageHasVJPAt N (flatConvStride2 W b) (flatConvStride2_differentiable W b)
    (flatConvStride2HasVJP W b) ε hε γ β x h_smooth

theorem cbReluStridedB_differentiableAt (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x) k ≠ 0) :
    DifferentiableAt ℝ (cbReluStridedB N (h := h) (w := w) W b ε γ β) x :=
  bnReluStage_differentiableAt N (flatConvStride2 W b) (flatConvStride2_differentiable W b)
    ε hε γ β x h_smooth

/-- Batched **strided conv → bn → relu** stage backward graph:
    `convStridedBackBatched ∘ bnBatchLABack ∘ selectPos`, each at its cumulative
    forward activation. The strided sibling of `cbReluBackBatchedGraph` —
    `convStridedBackBatched` (the new stride-2 batched-conv VJP) replaces
    `convBackBatched`. -/
noncomputable def cbReluStridedBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  .convStridedBackBatched (N := N) "%cbsrW" W b
    (.bnBatchLABack "%cbsrG" "%cbsrX" "cbsrE" ε γ (batchMap N (flatConvStride2 W b) x)
      (.selectPos "%cbsrR" (bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x)) e))

theorem cbReluStridedBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w)))
    (h_smooth : ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x) k ≠ 0) :
    den (cbReluStridedBackBatchedGraph W b ε γ β x e)
      = (cbReluStridedBHasVJPAt N W b ε hε γ β x h_smooth).backward (den e) := by
  rw [cbReluStridedBackBatchedGraph, convStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectPos_faithful _ _ h_smooth]
  simp only [cbReluStridedBHasVJPAt, bnReluStageHasVJPAt, stageHasVJPAt, vjpCompAt_backward,
    HasVJP.toHasVJPAt, Function.comp_apply]

/-- The strided conv → bn → relu stage as a `CertLayer` — `cbReluLayer` with `flatConvStride2`. -/
noncomputable def cbReluStridedLayer (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) where
  fwd := cbReluStridedB N (h := h) (w := w) W b ε γ β
  ok := fun x => ∀ k, bnBatchLA N oc h w ε γ β (batchMap N (flatConvStride2 W b) x) k ≠ 0
  diff := fun x hx => cbReluStridedB_differentiableAt N W b ε hε γ β x hx
  vjp := fun x hx => cbReluStridedBHasVJPAt N W b ε hε γ β x hx
  graph := fun x e => cbReluStridedBackBatchedGraph W b ε γ β x e
  faithful := fun x hx e => cbReluStridedBackBatchedGraph_faithful W b ε hε γ β x e hx

-- ════════════════════════════════════════════════════════════════
-- § The strided projection skip — `projStridedB` (conv_strided → bn, no relu)
-- ════════════════════════════════════════════════════════════════

/-- Batched **strided conv → bn** projection skip (downsample basic-block skip):
    `bnBatchLA ∘ batchMap (flatConvStride2)` — the 3×3 stride-2 projection that
    matches the body's downsampled `oc·h·w` output. The strided sibling of `projB`
    (`flatConvStride2` for `flatConv`); no activation (linear bottleneck). -/
@[reducible] noncomputable def projStridedB (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ β : Vec oc) :
    Vec (N * (ic * (2 * h) * (2 * w))) → Vec (N * (oc * h * w)) :=
  bnBatchLA N oc h w ε γ β ∘ batchMap N (flatConvStride2 W b)

theorem projStridedB_differentiable (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    Differentiable ℝ (projStridedB N (h := h) (w := w) W b ε γ β) :=
  bnStage_differentiable N (flatConvStride2 W b) (flatConvStride2_differentiable W b) ε hε γ β

noncomputable def projStridedBHasVJP (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    HasVJP (projStridedB N (h := h) (w := w) W b ε γ β) :=
  bnStageHasVJP N (flatConvStride2 W b) (flatConvStride2_differentiable W b)
    (flatConvStride2HasVJP W b) ε hε γ β

/-- Batched **strided conv → bn** projection-skip backward graph:
    `convStridedBackBatched ∘ bnBatchLABack`, at the skip's forward activation. The
    strided sibling of `projBackBatchedGraph`. -/
noncomputable def projStridedBackBatchedGraph {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (γ _β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    SHlo (N * (ic * (2 * h) * (2 * w))) :=
  .convStridedBackBatched (N := N) "%psW" W b
    (.bnBatchLABack "%psG" "%psX" "psE" ε γ (batchMap N (flatConvStride2 W b) x) e)

theorem projStridedBackBatchedGraph_faithful {N ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    den (projStridedBackBatchedGraph W b ε γ β x e)
      = (projStridedBHasVJP N W b ε hε γ β).backward x (den e) := by
  rw [projStridedBackBatchedGraph, convStridedBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε)]
  simp only [projStridedBHasVJP, bnStageHasVJP, vjpComp_backward]

/-- The strided conv → bn projection skip as a `CertLayer` — `projLayer` with `flatConvStride2`,
    globally certified. -/
noncomputable def projStridedLayer (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) where
  fwd := projStridedB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (projStridedB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (projStridedBHasVJP N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => projStridedBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => projStridedBackBatchedGraph_faithful W b ε hε γ β x e

end Proofs.StableHLO
