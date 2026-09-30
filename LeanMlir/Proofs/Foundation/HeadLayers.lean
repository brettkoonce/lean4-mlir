import LeanMlir.Proofs.Foundation.BackwardMaps
import LeanMlir.Proofs.Foundation.Batched.StageLayers

/-! # Shared stem-pool and head `CertLayer`s — batched GAP, the dense classifier, the 3×3/s2 pool

Every conv net in the suite ends in global average pooling and a dense classifier, and both are
globally certified (GAP is linear, dense is affine), so `ok := True`. Written once here and shared
by MobileNetV4, ResNet-34 and ResNet-50. The ResNets' stem is here too, conv-BN-ReLU and the 3×3/s2
pool as one layer (`stemReluPoolLayer`), certified where every window is smooth or dead
(`StemPoolSmoothAt`).

**Both backward graphs tie by `rfl`.** `den` of `.gapBackBatched` is definitionally the row-wise
GAP VJP, and `den` of `.denseRowBack` is `rowDenseBackFlat`, which is what `batchMapHasVJP`
reduces to. Note: GAP's VJP does not depend on its input, and `den .gapBackBatched` uses that by
evaluating the backward at `fun _ => 0`. That is sound because GAP is linear, and it is why the tie
holds at every `x`.
-/

namespace Proofs.StableHLO

/-- Batched **global average pool** as a `CertLayer`. Globally certified — GAP is linear. -/
noncomputable def gapLayer (N : Nat) {c h w : Nat} :
    CertLayer (N * (c * h * w)) (N * c) where
  fwd := batchMap N (globalAvgPoolFlat c h w)
  ok := fun _ => True
  diff := fun x _ => (batchMap_differentiable (globalAvgPoolFlat c h w)
    (globalAvgPoolFlat_differentiable c h w)) x
  vjp := fun x _ => (batchMapHasVJP (N := N) (globalAvgPoolFlat c h w)
    (globalAvgPoolFlatHasVJP c h w) (globalAvgPoolFlat_differentiable c h w)).toHasVJPAt x
  graph := fun _ e => .gapBackBatched (N := N) (c := c) (h := h) (w := w) e
  faithful := fun _ _ _ => rfl

/-- The GAP layer's forward is the batched global average pool. -/
theorem gapLayer_fwd_apply (N : Nat) {c h w : Nat} (v : Vec (N * (c * h * w))) :
    (gapLayer N (c := c) (h := h) (w := w)).fwd v
      = StableHLO.batchMap N (globalAvgPoolFlat c h w) v := rfl

/-- Batched **dense classifier** as a `CertLayer`. Globally certified — dense is affine. -/
noncomputable def denseLayer (N : Nat) {a nC : Nat} (W : Mat a nC) (b : Vec nC) :
    CertLayer (N * a) (N * nC) where
  fwd := batchMap N (dense W b)
  ok := fun _ => True
  diff := fun x _ => (batchMap_differentiable (dense W b) (dense_differentiable W b)) x
  vjp := fun x _ => (batchMapHasVJP (N := N) (dense W b) (denseHasVJP W b)
    (dense_differentiable W b)).toHasVJPAt x
  graph := fun _ e => .denseRowBack (N := N) (a := a) (c := nC) "%Wd" W e
  faithful := fun _ _ _ => rfl

/-- The classifier layer's forward is the batched dense. -/
theorem denseLayer_fwd_apply (N : Nat) {a nC : Nat} (W : Mat a nC) (b : Vec nC)
    (v : Vec (N * a)) :
    (denseLayer N W b).fwd v = StableHLO.batchMap N (Proofs.dense W b) v := rfl

end Proofs.StableHLO

namespace Proofs

/-- **Transport a VJP along a local equality.** `pdiv` reads only the germ of the map at the
    point (it is an `fderiv`), so a witness for `f` is one for any `g` that agrees with `f` near
    `x`, with the same backward. How a map that is not globally a composite of certified pieces
    still gets a certified VJP: the ReLU-then-pool stem is locally a gather. -/
def HasVJPAt.congrOfEventuallyEq {m n : Nat} {f g : Vec m → Vec n} {x : Vec m}
    (h : f =ᶠ[nhds x] g) (hf : HasVJPAt f x) : HasVJPAt g x where
  backward := hf.backward
  correct dy i := by
    rw [hf.correct]
    unfold pdiv
    rw [h.fderiv_eq]

/-- **ReLU then the 3×3/s2 pool is locally a gather.** Near a pre-activation `z` with no zero
    entry, where every window of `relu z` is smooth or dead (`MaxPool3s2SmoothOrDead`), the pooled
    ReLU reads each output from the cell `maxPool3s2LocalReindex (relu z)` names. A smooth window
    keeps its strict maximum nearby. A dead window has every pre-activation strictly negative, so
    nearby its ReLUs are all `0` and any choice of cell reads `0`. This is why a window of dead
    ReLUs, where the pool alone has no derivative, costs the composite nothing. -/
theorem maxPool3s2Flat_relu_eventuallyEq {c h w : Nat} (z : Vec (c * (2 * h) * (2 * w)))
    (hz : ∀ k, z k ≠ 0)
    (hs : MaxPool3s2SmoothOrDead
      (Tensor3.unflatten (relu _ z) : Tensor3 c (2 * h) (2 * w))) :
    (fun v => maxPool3s2Flat c h w (relu _ v)) =ᶠ[nhds z]
      (fun v k => relu _ v (maxPool3s2LocalReindex
        (Tensor3.unflatten (relu _ z) : Tensor3 c (2 * h) (2 * w)) k)) := by
  set y : Tensor3 c (2 * h) (2 * w) := Tensor3.unflatten (relu _ z) with hy
  have hcont : ∀ k, Continuous (fun v : Vec (c * (2 * h) * (2 * w)) => relu _ v k) := by
    intro k
    have : (fun v : Vec (c * (2 * h) * (2 * w)) => relu _ v k) = fun v => max (v k) 0 := by
      funext v; exact relu_apply_eq_max v k
    rw [this]; fun_prop
  -- every window cell stays at or below the argmax cell nearby
  have hmax : ∀ᶠ v in nhds z, ∀ (co : Fin c) (ho : Fin h) (wo : Fin w) (a' b' : Fin 3),
      (Tensor3.unflatten (relu _ v) : Tensor3 c (2 * h) (2 * w)) co (win3RowInv ho a')
          (win3ColInv wo b') ≤
        (Tensor3.unflatten (relu _ v) : Tensor3 c (2 * h) (2 * w)) co
          (win3RowInv ho (maxPool3s2Argmax y co ho wo).1)
          (win3ColInv wo (maxPool3s2Argmax y co ho wo).2) := by
    simp only [Filter.eventually_all]
    intro co ho wo a' b'
    by_cases hab : (win3RowInv ho a', win3ColInv wo b') =
        (win3RowInv ho (maxPool3s2Argmax y co ho wo).1,
          win3ColInv wo (maxPool3s2Argmax y co ho wo).2)
    · obtain ⟨hr, hs⟩ := Prod.mk.inj hab
      exact Filter.Eventually.of_forall fun _ => by rw [hr, hs]
    · rcases hs co ho wo with hdead | hsm
      · -- dead window: the pre-activation is strictly negative, so it stays negative nearby
        set k := finProdFinEquiv (finProdFinEquiv (co, win3RowInv ho a'), win3ColInv wo b')
        have hyk : relu _ z k ≤ 0 := hdead (a', b')
        have hzk : z k < 0 := by
          rcases lt_or_gt_of_ne (hz k) with hlt | hgt
          · exact hlt
          · exfalso
            have : relu _ z k = z k := by simp [relu, hgt]
            linarith
        have hev : ∀ᶠ v in nhds z, v k < 0 :=
          (continuous_apply k).continuousAt.eventually (gt_mem_nhds hzk)
        refine hev.mono fun v hv => ?_
        have h0 : relu _ v k = 0 := by simp [relu, not_lt.mpr hv.le]
        show relu _ v k ≤ _
        rw [h0]; exact relu_nonneg _ _ _
      · have hlt := hsm _ (a', b') (Ne.symm hab) (maxPool3s2Argmax_max y co ho wo)
        exact ((hcont _).continuousAt.eventually_lt (hcont _).continuousAt hlt).mono
          fun _ => le_of_lt
  filter_upwards [hmax] with v hv
  funext k_out
  exact maxPool3s2_eq_at_max (Tensor3.unflatten (relu _ v)) _ _ _ _ _ (hv _ _ _)

/-- **The stem pool's condition, per example**: every 3×3 window of the post-ReLU stem activation
    has its maximum at one position, or is entirely zero (a window of dead ReLUs). A tie is a
    property of one image's window, so the condition is stated on each row of the batch.

    Real batches have many all-zero windows, and the pool alone has no derivative at any of them;
    the stem layer is certified as ReLU and pool together (`stemReluPoolLayer`), where a dead window
    is locally constant. What is still excluded is a window whose maximum is POSITIVE and attained
    at two positions. On real ImageNet batches such windows occur in every batch, and they come
    from two cells reading IDENTICAL input patches (flat image regions); there the stem has no
    derivative in the image, so an input-gradient statement must exclude them, but the two cells
    are the same function of the stem's weights, so the loss is still differentiable in the
    parameters. The probe script scripts/probes/stem_pool_smooth_probe.py counts all three kinds of
    window. -/
def StemPoolSmoothAt (N h w : Nat) {oc : Nat} (v : Vec (N * (oc * (2 * h) * (2 * w)))) : Prop :=
  ∀ r : Fin N,
    MaxPool3s2SmoothOrDead (Tensor3.unflatten (Mat.unflatten v r) : Tensor3 oc (2 * h) (2 * w))

/-- The batched argmax gather: output `(r, k)` reads example `r`'s argmax cell for `k`. -/
noncomputable def maxPool3s2LocalReindexB (N c h w : Nat) (y : Vec (N * (c * (2 * h) * (2 * w)))) :
    Fin (N * (c * h * w)) → Fin (N * (c * (2 * h) * (2 * w))) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    finProdFinEquiv (p.1, maxPool3s2LocalReindex
      (Tensor3.unflatten (StableHLO.batchSlice N _ y p.1) : Tensor3 c (2 * h) (2 * w)) p.2)

/-- `maxPool3s2Flat_relu_eventuallyEq` over the batch, one example per row. -/
theorem batchMap_maxPool3s2Flat_relu_eventuallyEq (N : Nat) {c h w : Nat}
    (Z : Vec (N * (c * (2 * h) * (2 * w)))) (hz : ∀ k, Z k ≠ 0)
    (hs : StemPoolSmoothAt N h w (relu _ Z)) :
    (fun v => StableHLO.batchMap N (maxPool3s2Flat c h w) (relu _ v)) =ᶠ[nhds Z]
      (fun v k => relu _ v (maxPool3s2LocalReindexB N c h w (relu _ Z) k)) := by
  have hrow : ∀ r : Fin N, ∀ᶠ v in nhds Z,
      maxPool3s2Flat c h w (relu _ (Mat.unflatten v r)) =
        fun k => relu _ (Mat.unflatten v r) (maxPool3s2LocalReindex
          (Tensor3.unflatten (relu _ (Mat.unflatten Z r)) : Tensor3 c (2 * h) (2 * w)) k) := by
    intro r
    have hT : Filter.Tendsto (fun v : Vec (N * (c * (2 * h) * (2 * w))) => Mat.unflatten v r)
        (nhds Z) (nhds (Mat.unflatten Z r)) :=
      (continuous_pi fun _ => continuous_apply _).continuousAt
    exact (maxPool3s2Flat_relu_eventuallyEq (Mat.unflatten Z r) (fun k => hz _) (hs r)).comp_tendsto hT
  filter_upwards [Filter.eventually_all.mpr hrow] with v hv
  funext idx
  exact congrFun (hv (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2

/-- **The render's pool backward is the gather's scatter.** `maxPool3s2FlatBackB` (what
    `den maxPool3s2BackB` is) sends each output cotangent to its argmax cell, which is exactly
    the adjoint of the gather `maxPool3s2LocalReindexB`, at every input. -/
theorem maxPool3s2FlatBackB_eq_reindex (N c h w : Nat) (y : Vec (N * (c * (2 * h) * (2 * w))))
    (v : Vec (N * (c * (2 * h) * (2 * w)))) (dy : Vec (N * (c * h * w))) :
    maxPool3s2FlatBackB N c h w y dy =
      (reindexVJP (maxPool3s2LocalReindexB N c h w y)).backward v dy := by
  funext idx
  obtain ⟨⟨r, i⟩, rfl⟩ := finProdFinEquiv.surjective idx
  simp only [maxPool3s2FlatBackB, StableHLO.batchMapAux, reindexVJP, maxPool3s2FlatBack,
    Equiv.symm_apply_apply, maxPool3s2LocalReindexB]
  conv_rhs => rw [sum_finProdFinEquiv]
  simp only [Equiv.symm_apply_apply, EmbeddingLike.apply_eq_iff_eq, Prod.mk.injEq]
  rw [Finset.sum_eq_single r (fun r' _ hr' => by simp [Ne.symm hr']) (by simp)]
  refine Finset.sum_congr rfl fun k _ => ?_
  by_cases hk : maxPool3s2LocalReindex
      (Tensor3.unflatten (StableHLO.batchSlice N _ y r) : Tensor3 c (2 * h) (2 * w)) k = i
  · simp [hk, StableHLO.batchSlice]
  · simp [hk, Ne.symm hk]

/-- The batched ReLU-then-pool is differentiable at a smooth-or-dead pre-activation. -/
theorem batchMap_maxPool3s2Flat_relu_differentiableAt (N : Nat) {c h w : Nat}
    (Z : Vec (N * (c * (2 * h) * (2 * w)))) (hz : ∀ k, Z k ≠ 0)
    (hs : StemPoolSmoothAt N h w (relu _ Z)) :
    DifferentiableAt ℝ (fun v => StableHLO.batchMap N (maxPool3s2Flat c h w) (relu _ v)) Z :=
  (((reindexCLM _).differentiableAt).comp Z (relu_differentiableAt_of_smooth _ Z hz)).congr_of_eventuallyEq
    (batchMap_maxPool3s2Flat_relu_eventuallyEq N Z hz hs)

/-- The batched ReLU-then-pool's VJP at a smooth-or-dead pre-activation: scatter to the argmax
    cells, then the ReLU mask. -/
noncomputable def batchMapMaxPool3s2FlatReluHasVJPAt (N : Nat) {c h w : Nat}
    (Z : Vec (N * (c * (2 * h) * (2 * w)))) (hz : ∀ k, Z k ≠ 0)
    (hs : StemPoolSmoothAt N h w (relu _ Z)) :
    HasVJPAt (fun v => StableHLO.batchMap N (maxPool3s2Flat c h w) (relu _ v)) Z :=
  HasVJPAt.congrOfEventuallyEq (batchMap_maxPool3s2Flat_relu_eventuallyEq N Z hz hs).symm
    (vjpCompAt _ _ Z (relu_differentiableAt_of_smooth _ Z hz) (reindexCLM _).differentiableAt
      (reluHasVJPAt _ Z hz) ((reindexVJP _).toHasVJPAt _))

/-- Its backward is the render's pool scatter, then the ReLU mask. -/
theorem batchMapMaxPool3s2FlatReluHasVJPAt_backward (N : Nat) {c h w : Nat}
    (Z : Vec (N * (c * (2 * h) * (2 * w)))) (hz : ∀ k, Z k ≠ 0)
    (hs : StemPoolSmoothAt N h w (relu _ Z)) (dy : Vec (N * (c * h * w))) :
    (batchMapMaxPool3s2FlatReluHasVJPAt N Z hz hs).backward dy =
      (reluHasVJPAt _ Z hz).backward (maxPool3s2FlatBackB N c h w (relu _ Z) dy) := by
  rw [maxPool3s2FlatBackB_eq_reindex _ _ _ _ _ Z]; rfl

/-- The pooled stem agrees near `x` with the argmax gather of the stem's post-ReLU output. -/
theorem stemPool_eventuallyEq (N : Nat) {ic c h w kH kW : Nat}
    (W : Kernel4 c ic kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))))
    (hz : ∀ k, StableHLO.bnBatchLA N c (2 * h) (2 * w) ε γ β
      (StableHLO.batchMap N (flatConvStride2 W b) x) k ≠ 0)
    (hs : StemPoolSmoothAt N h w
      (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β x)) :
    (StableHLO.batchMap N (maxPool3s2Flat c h w) ∘
        StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β) =ᶠ[nhds x]
      ((fun y k => y (maxPool3s2LocalReindexB N c h w
          (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β x) k)) ∘
        StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β) := by
  have hA : Continuous (StableHLO.bnBatchLA N c (2 * h) (2 * w) ε γ β ∘
      StableHLO.batchMap N (flatConvStride2 W b)) :=
    (bnBatchLA_continuous N c (2 * h) (2 * w) ε hε γ β).comp
      (batchMap_continuous _ (flatConvStride2_differentiable W b).continuous)
  exact (batchMap_maxPool3s2Flat_relu_eventuallyEq N _ hz hs).comp_tendsto hA.continuousAt

/-- **The ResNet stem, strided conv-BN-ReLU then the 3×3/s2 pool, as ONE certified layer.**

    It is one layer, not `cbReluStridedLayer.comp` of a pool layer, because the pool alone is not
    differentiable at a window of dead ReLUs, and real batches have many. Together they are:
    near the input the pooled stem is the argmax gather of the ReLU output
    (`stemPool_eventuallyEq`), so its VJP is the conv-BN-ReLU stage's after the gather's scatter,
    and that scatter is what the render's `maxPool3s2BackB` denotes
    (`maxPool3s2FlatBackB_eq_reindex`). The forward and the backward graph are the two-layer
    composite's, so the rendered stem is unchanged; only the certification domain grows. -/
noncomputable def stemReluPoolLayer (N : Nat) {ic c h w kH kW : Nat}
    (W : Kernel4 c ic kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    StableHLO.CertLayer (N * (ic * (2 * (2 * h)) * (2 * (2 * w)))) (N * (c * h * w)) where
  fwd := StableHLO.batchMap N (maxPool3s2Flat c h w) ∘
    StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β
  ok := fun x => (∀ k, StableHLO.bnBatchLA N c (2 * h) (2 * w) ε γ β
      (StableHLO.batchMap N (flatConvStride2 W b) x) k ≠ 0) ∧
    StemPoolSmoothAt N h w (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β x)
  diff := fun x hx =>
    (((reindexCLM _).differentiableAt).comp x
      (StableHLO.cbReluStridedB_differentiableAt N W b ε hε γ β x hx.1)).congr_of_eventuallyEq
      (stemPool_eventuallyEq N W b ε hε γ β x hx.1 hx.2)
  vjp := fun x hx =>
    HasVJPAt.congrOfEventuallyEq (stemPool_eventuallyEq N W b ε hε γ β x hx.1 hx.2).symm
      (vjpCompAt _ _ x (StableHLO.cbReluStridedB_differentiableAt N W b ε hε γ β x hx.1)
        (reindexCLM _).differentiableAt
        (StableHLO.cbReluStridedBHasVJPAt N W b ε hε γ β x hx.1)
        ((reindexVJP _).toHasVJPAt _))
  graph := fun x e => StableHLO.cbReluStridedBackBatchedGraph W b ε γ β x
    (.maxPool3s2BackB "%stemR" (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β x) e)
  faithful := fun x hx e => by
    rw [StableHLO.cbReluStridedBackBatchedGraph_faithful W b ε hε γ β x _ hx.1,
      den_maxPool3s2BackB_eq_flatBackB, maxPool3s2FlatBackB_eq_reindex _ _ _ _ _
        (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β x)]
    rfl

/-- The stem layer's VJP is the conv-BN-ReLU stage's after the render's pool scatter. -/
theorem stemReluPoolLayer_vjp_backward (N : Nat) {ic c h w kH kW : Nat}
    (W : Kernel4 c ic kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (x : Vec (N * (ic * (2 * (2 * h)) * (2 * (2 * w))))) (hx : (stemReluPoolLayer N (h := h) (w := w) W b ε hε γ β).ok x)
    (dy : Vec (N * (c * h * w))) :
    ((stemReluPoolLayer N W b ε hε γ β).vjp x hx).backward dy =
      (StableHLO.cbReluStridedBHasVJPAt N W b ε hε γ β x hx.1).backward
        (maxPool3s2FlatBackB N c h w
          (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β x) dy) := by
  rw [maxPool3s2FlatBackB_eq_reindex _ _ _ _ _
    (StableHLO.cbReluStridedB N (h := 2 * h) (w := 2 * w) W b ε γ β x)]
  rfl

end Proofs
