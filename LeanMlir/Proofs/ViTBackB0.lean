import LeanMlir.Proofs.Attention
import LeanMlir.Proofs.ViTFwdGraph
import LeanMlir.Proofs.ViTChainClose
import LeanMlir.Proofs.StableHLO

/-! # ViT whole-block backward-graph faithfulness (heads = 1, per-token Mat VJP)

The ViT analogue of the other four nets' `*BackB0` capstones: a *backward*
StableHLO graph (over the ch10 backward tokens — `denseRowBack`/`geluBack`/
`lnRowBack`/`softmaxRowBack`/`matmulF`/`transposeF`/`scaleF`/`addV`) whose
denotation IS the proven whole transformer-block VJP
`transformerBlock_has_vjp_mat` (Attention.lean), at **heads = 1** (matching the
committed render config that `ViTFwdGraph.lean`'s `vitFwdGraph_faithful` uses —
all `1 * d_head`).

Unlike the conv nets — whose blocks live natively as `Vec → Vec` (`HasVJP`) — a
transformer block lives in the per-token matrix framework `HasVJPMat` (`Mat N D →
Mat N D`). The block VJP's `.backward` is therefore `Mat`-valued, while `den` is
`Vec`-valued; the faithfulness statements bridge the two through `Mat.flatten`
(exactly the convention `vitFwdGraph_faithful` uses on the forward side):

    den (…BackGraph A e) = Mat.flatten ((… _has_vjp_mat …).backward A (Mat.unflatten (den e)))

Three stages, each `lake build`-green before the next:

  * **Stage 1** — `mlpSublayerBackGraph` ↔ `transformerMlpSublayer_has_vjp_mat`.
    Residual fan-in (`addV (inner) (%dz)`) where inner =
    `lnRowBack(LN₂) ∘ denseRowBack(Wfc1) ∘ geluBack ∘ denseRowBack(Wfc2)` at the
    cumulative forward activations.
  * **Stage 2** — `attnSublayerBackGraph` ↔ `transformerAttnSublayer_has_vjp_mat`.
    The crux: MHSA backward = `denseRowBack(qkv) ∘ [SDPA backward] ∘
    denseRowBack(Wo)` with the SDPA backward assembled from
    `matmulF`/`transposeF`/`scaleF`/`softmaxRowBack`, bridged at heads = 1 to
    `mhsa_g_has_vjp_mat`, then LN₁-back + residual fan-in.
  * **Stage 3** — `transformerBlockBackGraph` ↔ `transformerBlock_has_vjp_mat`,
    via the `vjpMat_comp` structure `block.backward A dY = attn.backward A
    (mlp.backward (attn A) dY)`.

The framework `.backward` rules (Tensor.lean):
  * `vjpMat_comp F G … .backward A dY = hF.backward A (hG.backward (F A) dY)`
  * `biPathMat_has_vjp F G … .backward A dY i j = hF.backward A dY i j +
    hG.backward A dY i j`; the identity skip's `.backward A dY = dY`.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Flat-form ↔ rowwise `.backward` bridges
-- ════════════════════════════════════════════════════════════════

/-! Each ch10 backward den helper is a flattened rowwise op; here we tie it to
the `HasVJPMat` `.backward` of the corresponding framework piece. These are the
ViT-backward analogues of ViTFwdGraph's `rowDenseFlat_flat`/`rowLNFlat_flat`
forward bridges. -/

/-- Per-token dense input-VJP: the flat `rowDenseBackFlat` IS the flatten of the
    rowwise `dense_per_token_has_vjp_mat.backward` (which ignores the saved
    activation `A`, dense being affine — `dense_has_vjp.backward _ dy = Mat.mulVec W dy`). -/
lemma rowDenseBackFlat_eq_backward {N a c : Nat} (W : Mat a c) (b : Vec c)
    (A : Mat N a) (dY : Mat N c) :
    rowDenseBackFlat N a c W (Mat.flatten dY)
      = Mat.flatten ((dense_per_token_has_vjp_mat N a c W b).backward A dY) := by
  unfold rowDenseBackFlat dense_per_token_has_vjp_mat rowwise_has_vjp_mat
  simp only [dense_has_vjp, Mat.unflatten_flatten]

/-- Per-token LayerNorm input-VJP: `rowLNBackFlat` IS the flatten of the rowwise
    `layerNorm_per_token_has_vjp_mat.backward` at the saved pre-LN activation `A`.
    (`layerNorm_has_vjp` is definitionally `bn_has_vjp`, whose backward is
    `bn_grad_input`; `rowLNBackFlat` is the rowwise `bn_grad_input`.) -/
lemma rowLNBackFlat_eq_backward {N D : Nat} (ε γ β : ℝ) (hε : 0 < ε)
    (A : Mat N D) (dY : Mat N D) :
    rowLNBackFlat N D ε γ (Mat.flatten A) (Mat.flatten dY)
      = Mat.flatten ((layerNorm_per_token_has_vjp_mat N D ε γ β hε).backward A dY) := by
  unfold rowLNBackFlat layerNorm_per_token_has_vjp_mat rowwise_has_vjp_mat
  simp only [Mat.unflatten_flatten]
  rfl

/-- Per-token GELU input-VJP: the flat `gelu_has_vjp (N*D)` backward IS the
    flatten of the rowwise `gelu_per_token_has_vjp_mat.backward` at the saved
    pre-GELU activation `A` (GELU is elementwise, so flat and rowwise agree). -/
lemma geluFlat_eq_backward {N D : Nat} (A : Mat N D) (dY : Mat N D) :
    (gelu_has_vjp (N * D)).backward (Mat.flatten A) (Mat.flatten dY)
      = Mat.flatten ((gelu_per_token_has_vjp_mat N D).backward A dY) := by
  unfold gelu_per_token_has_vjp_mat rowwise_has_vjp_mat gelu_has_vjp Mat.flatten
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Stage 1 — MLP sublayer backward graph
-- ════════════════════════════════════════════════════════════════

/-- The transformer MLP backward graph (reverse-order chain of
    `transformerMlp_has_vjp_mat = dense2 ∘ gelu ∘ dense1` per-token; outermost
    backward token = earliest forward op = dense1):

      `denseRowBack(Wfc1) ∘ geluBack(@ pre-GELU = dense1(Y)) ∘ denseRowBack(Wfc2)`

    where `Y` is the MLP input (= `LN₂ h`). The GELU backward reads its saved
    pre-activation `m1 = dense1 Y`. -/
noncomputable def transformerMlpBackGraph {Np1 D mlpDim : Nat}
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D)
    (Y : Mat Np1 D) (e : SHlo (Np1 * D)) : SHlo (Np1 * D) :=
  .denseRowBack "%Wfc1" Wfc1
    (.geluBack "%m1" (Mat.flatten (fun r => dense Wfc1 bfc1 (Y r)))
      (.denseRowBack "%Wfc2" Wfc2 e))

/-- **MLP backward-graph faithfulness.** The reverse-order chain denotes the
    proven `transformerMlp_has_vjp_mat.backward` at the saved MLP input `Y`. The
    two dense backs ignore the activation; GELU's reads `dense1 Y`. -/
theorem transformerMlpBackGraph_faithful {Np1 D mlpDim : Nat}
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (Y : Mat Np1 D) (dz : Mat Np1 D) :
    den (transformerMlpBackGraph Wfc1 bfc1 Wfc2 Y (.operand "%dz" (Mat.flatten dz)))
      = Mat.flatten ((transformerMlp_has_vjp_mat Np1 D mlpDim Wfc1 bfc1 Wfc2 bfc2).backward
          Y dz) := by
  simp only [transformerMlpBackGraph, denseRowBack_faithful, geluBack_faithful,
    den_operand, transformerMlp_has_vjp_mat, vjpMat_comp]
  -- transformerMlp.backward Y dz = dense1.backward Y (gelu.backward (dense1 Y)
  --   (dense2.backward (gelu (dense1 Y)) dz))
  -- denseRowBack(Wfc2) at dz = flatten (dense2.backward · dz)
  rw [rowDenseBackFlat_eq_backward Wfc2 bfc2
        (fun r => gelu mlpDim (dense Wfc1 bfc1 (Y r)))]
  -- geluBack at (dense1 Y) of that = flatten (gelu.backward (dense1 Y) ·)
  rw [geluFlat_eq_backward (N := Np1) (D := mlpDim)
        (fun r => dense Wfc1 bfc1 (Y r))]
  -- denseRowBack(Wfc1) of that = flatten (dense1.backward Y ·)
  rw [rowDenseBackFlat_eq_backward Wfc1 bfc1 Y]
  rfl

/-- The transformer MLP-sublayer non-trivial arm backward graph
    (`transformerMlp ∘ LN₂`; outermost backward token = earliest forward op = LN₂):

      `lnRowBack(LN₂ @ h) ∘ transformerMlpBackGraph(@ Y = LN₂ h)`

    The LN₂ backward reads its saved pre-norm input `h`; the MLP-body back reads
    the MLP input `Y = LN₂ h`. -/
noncomputable def mlpSublayerInnerBackGraph {Np1 D mlpDim : Nat}
    (ε γ2 : ℝ)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D)
    (h : Vec (Np1 * D)) (Y : Mat Np1 D) (e : SHlo (Np1 * D)) : SHlo (Np1 * D) :=
  .lnRowBack "%g2" "%h" "ε" ε γ2 h
    (transformerMlpBackGraph Wfc1 bfc1 Wfc2 Y e)

/-- **MLP-sublayer inner-arm backward-graph faithfulness.** The reverse-order
    chain denotes the proven `(vjpMat_comp LN₂ transformerMlp).backward h ·`, the
    sublayer's non-trivial arm. `Y = LN₂ h` is the saved MLP input. -/
theorem mlpSublayerInnerBackGraph_faithful {Np1 D mlpDim : Nat}
    (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (h : Mat Np1 D) (dz : Mat Np1 D) :
    den (mlpSublayerInnerBackGraph ε γ2 Wfc1 bfc1 Wfc2 (Mat.flatten h)
          (fun r => layerNormForward D ε γ2 β2 (h r)) (.operand "%dz" (Mat.flatten dz)))
      = Mat.flatten
          ((layerNorm_per_token_has_vjp_mat Np1 D ε γ2 β2 hε).backward h
            ((transformerMlp_has_vjp_mat Np1 D mlpDim Wfc1 bfc1 Wfc2 bfc2).backward
              (fun r => layerNormForward D ε γ2 β2 (h r)) dz)) := by
  simp only [mlpSublayerInnerBackGraph, lnRowBack_faithful]
  rw [transformerMlpBackGraph_faithful Wfc1 bfc1 Wfc2 bfc2
        (fun r => layerNormForward D ε γ2 β2 (h r)) dz]
  rw [rowLNBackFlat_eq_backward (β := β2) ε γ2 hε h
        ((transformerMlp_has_vjp_mat Np1 D mlpDim Wfc1 bfc1 Wfc2 bfc2).backward
          (fun r => layerNormForward D ε γ2 β2 (h r)) dz)]

/-- The whole transformer MLP-sublayer backward graph (inner arm + identity skip):
    `addV (innerBack … (%dz)) (%dz)`. The identity skip contributes the cotangent
    verbatim; `addV` sums the two residual paths. -/
noncomputable def mlpSublayerBackGraph {Np1 D mlpDim : Nat}
    (ε γ2 : ℝ)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D)
    (h : Vec (Np1 * D)) (Y : Mat Np1 D) (dz : Vec (Np1 * D)) : SHlo (Np1 * D) :=
  .addV (mlpSublayerInnerBackGraph ε γ2 Wfc1 bfc1 Wfc2 h Y (.operand "%dz" dz))
        (.operand "%dz" dz)

/-- **MLP sublayer backward-graph faithfulness (Stage 1 capstone).** The
    residual-fan-in graph denotes the proven `transformerMlpSublayer_has_vjp_mat`
    backward (heads = 1, `D := 1 * d_head` at the call site), under `0 < ε`. The
    `biPathMat` skip arm is the identity (contributes the cotangent verbatim); the
    non-trivial arm is the `LN₂`-back of the MLP-body back. The forward
    activations fed: `h` (LN₂'s pre-norm input, = the MLP input via `LN₂ h`). -/
theorem mlpSublayerBackGraph_faithful {Np1 d_head mlpDim : Nat}
    (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat (1 * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d_head)) (bfc2 : Vec (1 * d_head))
    (h : Mat Np1 (1 * d_head)) (dz : Mat Np1 (1 * d_head)) :
    den (mlpSublayerBackGraph ε γ2 Wfc1 bfc1 Wfc2 (Mat.flatten h)
          (fun r => layerNormForward (1 * d_head) ε γ2 β2 (h r)) (Mat.flatten dz))
      = Mat.flatten ((transformerMlpSublayer_has_vjp_mat Np1 1 d_head mlpDim ε γ2 β2 hε
          Wfc1 bfc1 Wfc2 bfc2).backward h dz) := by
  funext j
  -- LHS: den (addV (innerBack at h) (%dz)) j = den innerBack j + flatten dz j
  show den (mlpSublayerInnerBackGraph ε γ2 Wfc1 bfc1 Wfc2 (Mat.flatten h)
            (fun r => layerNormForward (1 * d_head) ε γ2 β2 (h r))
            (.operand "%dz" (Mat.flatten dz))) j + Mat.flatten dz j = _
  rw [mlpSublayerInnerBackGraph_faithful ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2 h dz]
  -- RHS: flatten (biPathMat_has_vjp.backward h dz) j
  --    = flatten (fun i k => dz i k + inner.backward h dz i k) j
  show _ = Mat.flatten ((transformerMlpSublayer_has_vjp_mat Np1 1 d_head mlpDim
              ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2).backward h dz) j
  -- the sublayer backward = `fun i k => dz i k + LN2.backward h (mlp.backward (LN2 h) dz) i k`
  show Mat.flatten ((layerNorm_per_token_has_vjp_mat Np1 (1 * d_head) ε γ2 β2 hε).backward h
          ((transformerMlp_has_vjp_mat Np1 (1 * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2).backward
            (fun r => layerNormForward (1 * d_head) ε γ2 β2 (h r)) dz)) j
        + Mat.flatten dz j = _
  -- Both sides are flatten of a pointwise sum; commute the addition.
  unfold Mat.flatten
  exact add_comm _ _

-- ════════════════════════════════════════════════════════════════
-- § Stage 2 — MHSA backward heads = 1 collapse (the crux)
-- ════════════════════════════════════════════════════════════════

/-! The proven `mhsa_has_vjp_mat` witness is built via `by rw [mhsa_layer_eq_compose];
exact vjpMat_comp …`, so its `.backward` field does NOT reduce by `rfl` (the
`Eq.mpr` transport blocks whnf). We instead build a *clean* witness `mhsaClean`
for the same factored function (whose `.backward` unfolds transparently), tie it
to `mhsa_has_vjp_mat` by VJP determinism, and collapse it at heads = 1 to the
plain three-way dense fan-in over `sdpa_back_{Q,K,V}` (the form ViTChainClose's
`vitCotD{Q,K,V}_eq_sdpa_back_*` ties target). -/

/-- **`HasVJPMat` backward determinism** — two matrix VJPs of the same function
    agree (both equal the shared `pdivMat`-contraction). -/
theorem hasVJPMat_backward_det {a b c d : Nat} {f : Mat a b → Mat c d}
    (v v' : HasVJPMat f) (A : Mat a b) (dY : Mat c d) :
    v.backward A dY = v'.backward A dY := by
  funext i j; rw [v.correct, v'.correct]

/-- Determinism across *propositionally-equal* functions. -/
theorem hasVJPMat_backward_det' {a b c d : Nat} {f g : Mat a b → Mat c d}
    (hfg : f = g) (v : HasVJPMat f) (v' : HasVJPMat g) (A : Mat a b) (dY : Mat c d) :
    v.backward A dY = v'.backward A dY := by
  subst hfg; exact hasVJPMat_backward_det v v' A dY

/-- A *clean* `HasVJPMat` witness for the factored MHSA — the same `vjpMat_comp`
    chain `mhsa_has_vjp_mat`'s body uses, but stated for the explicit composition
    `Wo-dense ∘ colSlabApply mhsa_g ∘ qkv-dense` so its `.backward` reduces by
    `rfl` (no `mhsa_layer_eq_compose` transport in the way). -/
noncomputable def mhsaClean (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat
      ((fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n)) ∘
       (colSlabApply (mhsa_g N d_head) (heads := heads)) ∘
       (fun X' : Mat N (heads * d_head) => fun n =>
          dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n))) :=
  vjpMat_comp _ _
    (by
      have h1 := dense_per_token_flat_diff (N := N) (mhsa_qkv_W heads d_head Wq Wk Wv)
        (mhsa_qkv_b heads d_head bq bk bv)
      have h2 := colSlabApply_flat_diff (mhsa_g N d_head) (mhsa_g_flat_diff N d_head) (heads := heads)
      have h_eq : (fun v : Vec (N * (heads * d_head)) =>
          Mat.flatten ((colSlabApply (mhsa_g N d_head) (heads := heads) ∘
            (fun X' : Mat N (heads * d_head) => fun n =>
              dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n)))
            (Mat.unflatten v) : Mat N (heads * d_head))) =
          (fun u : Vec (N * (heads * (3 * d_head))) =>
            Mat.flatten ((colSlabApply (mhsa_g N d_head) (heads := heads)) (Mat.unflatten u)
                         : Mat N (heads * d_head))) ∘
          (fun v : Vec (N * (heads * d_head)) =>
            Mat.flatten ((fun X' : Mat N (heads * d_head) => fun n =>
              dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n))
              (Mat.unflatten v))) := by
        funext v; simp [Function.comp, Mat.unflatten_flatten]
      rw [h_eq]; exact h2.comp h1)
    (dense_per_token_flat_diff (N := N) Wo bo)
    (vjpMat_comp _ _
      (dense_per_token_flat_diff (N := N) (mhsa_qkv_W heads d_head Wq Wk Wv)
        (mhsa_qkv_b heads d_head bq bk bv))
      (colSlabApply_flat_diff (mhsa_g N d_head) (mhsa_g_flat_diff N d_head) (heads := heads))
      (rowwise_has_vjp_mat (dense_has_vjp (mhsa_qkv_W heads d_head Wq Wk Wv)
        (mhsa_qkv_b heads d_head bq bk bv))
        (dense_diff (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv)))
      (colSlabwise_has_vjp_mat (mhsa_g_has_vjp_mat N d_head) (mhsa_g_flat_diff N d_head)))
    (rowwise_has_vjp_mat (dense_has_vjp Wo bo) (dense_diff Wo bo))

/-- The clean witness's backward IS `mhsa_has_vjp_mat`'s backward (both VJPs of
    `mhsa_layer`, tied by determinism). -/
theorem mhsaClean_backward_eq (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (X dY : Mat N (heads * d_head)) :
    (mhsaClean N heads d_head Wq Wk Wv Wo bq bk bv bo).backward X dY
      = (mhsa_has_vjp_mat N heads d_head Wq Wk Wv Wo bq bk bv bo).backward X dY := by
  have hfun : ((fun M : Mat N (heads * d_head) => fun n => dense Wo bo (M n)) ∘
       (colSlabApply (mhsa_g N d_head) (heads := heads)) ∘
       (fun X' : Mat N (heads * d_head) => fun n =>
          dense (mhsa_qkv_W heads d_head Wq Wk Wv) (mhsa_qkv_b heads d_head bq bk bv) (X' n)))
      = mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo := by
    funext X'; exact (mhsa_layer_eq_compose N heads d_head Wq Wk Wv Wo bq bk bv bo X').symm
  exact hasVJPMat_backward_det' hfun (mhsaClean N heads d_head Wq Wk Wv Wo bq bk bv bo)
    (mhsa_has_vjp_mat N heads d_head Wq Wk Wv Wo bq bk bv bo) X dY

private lemma sum_one_3d {M : Type*} [AddCommMonoid M] (d : Nat) (f : Fin (1 * (3 * d)) → M) :
    (∑ k : Fin (1 * (3 * d)), f k)
      = ∑ c : Fin 3, ∑ j : Fin d, f (finProdFinEquiv ((0 : Fin 1), finProdFinEquiv (c, j))) := by
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin 1 × Fin (3*d) ≃ Fin (1 * (3*d))) f]
  rw [Fintype.sum_prod_type, Fin.sum_univ_one]
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin 3 × Fin d ≃ Fin (3*d))
      (fun kk => f (finProdFinEquiv ((0:Fin 1), kk)))]
  rw [Fintype.sum_prod_type]

set_option maxHeartbeats 800000 in
/-- The three-way QKV-stack dense-back fan-in at heads = 1: contracting the
    column-stacked SDPA backward against `mhsa_qkv_W` splits into the three
    per-projection dense-backs (`Mat.mulVec Wq/Wk/Wv`). -/
private lemma qkv_back_fanin (N d : Nat) (Wq Wk Wv : Mat (1 * d) (1 * d))
    (dQg dKg dVg : Mat N d) (r : Fin N) (c : Fin (1 * d)) :
    Mat.mulVec (mhsa_qkv_W 1 d Wq Wk Wv)
      (fun (idx : Fin (1 * (3 * d))) =>
        let p := finProdFinEquiv.symm idx
        let q := finProdFinEquiv.symm p.2
        if q.1 = (0 : Fin 3) then dQg r q.2
        else if q.1 = (1 : Fin 3) then dKg r q.2
        else dVg r q.2) c
      = Mat.mulVec Wq (fun cc => dQg r (finProdFinEquiv.symm cc).2) c
        + Mat.mulVec Wk (fun cc => dKg r (finProdFinEquiv.symm cc).2) c
        + Mat.mulVec Wv (fun cc => dVg r (finProdFinEquiv.symm cc).2) c := by
  unfold Mat.mulVec
  rw [sum_one_3d]
  simp only [Equiv.symm_apply_apply]
  rw [Fin.sum_univ_three]
  simp only [mhsa_qkv_W_eq0, mhsa_qkv_W_eq1, mhsa_qkv_W_eq2,
    show (1 : Fin 3) ≠ (0 : Fin 3) from by decide,
    show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
    show (2 : Fin 3) ≠ (1 : Fin 3) from by decide, if_true, if_false]
  have hr : ∀ (W : Mat (1*d) (1*d)) (g : Mat N d),
      (∑ x : Fin (1*d), W c x * g r (finProdFinEquiv.symm x).2)
        = ∑ j : Fin d, W c (finProdFinEquiv ((0:Fin 1), j)) * g r j := by
    intro W g
    rw [← Equiv.sum_comp (finProdFinEquiv : Fin 1 × Fin d ≃ Fin (1*d))
        (fun cc => W c cc * g r (finProdFinEquiv.symm cc).2)]
    rw [Fintype.sum_prod_type, Fin.sum_univ_one]
    apply Finset.sum_congr rfl; intro j _; rw [Equiv.symm_apply_apply]
  rw [hr Wq dQg, hr Wk dKg, hr Wv dVg]

/-- The collapsed heads = 1 MHSA backward: the three-way dense fan-in
    (`Mat.mulVec Wq/Wk/Wv`, reshaping the inner-`d` SDPA backwards up to `1*d`)
    over `sdpa_back_{Q,K,V}` at the inner-`d` dense projections of `X` and the
    `Wo`-back of the cotangent `dh`. -/
noncomputable def mhsaBackCollapsed (N d : Nat)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (X dh : Mat N (1 * d)) : Mat N (1 * d) :=
  let Qg : Mat N d := fun r j => dense Wq bq (X r) (finProdFinEquiv ((0:Fin 1), j))
  let Kg : Mat N d := fun r j => dense Wk bk (X r) (finProdFinEquiv ((0:Fin 1), j))
  let Vg : Mat N d := fun r j => dense Wv bv (X r) (finProdFinEquiv ((0:Fin 1), j))
  let dAttg : Mat N d := fun r j => Mat.mulVec Wo (dh r) (finProdFinEquiv ((0:Fin 1), j))
  let dQg : Mat N d := sdpa_back_Q N d Qg Kg Vg dAttg
  let dKg : Mat N d := sdpa_back_K N d Qg Kg Vg dAttg
  let dVg : Mat N d := sdpa_back_V N d Qg Kg Vg dAttg
  fun r c =>
    Mat.mulVec Wq (fun cc => dQg r (finProdFinEquiv.symm cc).2) c
    + Mat.mulVec Wk (fun cc => dKg r (finProdFinEquiv.symm cc).2) c
    + Mat.mulVec Wv (fun cc => dVg r (finProdFinEquiv.symm cc).2) c

set_option maxHeartbeats 4000000 in
/-- **MHSA backward heads = 1 collapse.** The clean MHSA witness's backward equals
    the plain three-way dense fan-in over `sdpa_back_{Q,K,V}` (`mhsaBackCollapsed`).
    Composes: qkv-stack dense-back = `Mat.mulVec (mhsa_qkv_W)`, colSlabwise-back at
    one head = `mhsa_g_has_vjp_mat.backward` on the single slab (the if-third
    `sdpa_back_*` form), the QKV-stack column-projections collapse to plain
    `dense Wq/Wk/Wv`, and the qkv-back fan-in splits per projection. -/
theorem mhsaClean_backward_collapse (N d : Nat)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (X dh : Mat N (1 * d)) :
    (mhsaClean N 1 d Wq Wk Wv Wo bq bk bv bo).backward X dh
      = mhsaBackCollapsed N d Wq Wk Wv Wo bq bk bv bo X dh := by
  funext r c
  show (rowwise_has_vjp_mat (dense_has_vjp (mhsa_qkv_W 1 d Wq Wk Wv)
                                          (mhsa_qkv_b 1 d bq bk bv))
                           (dense_diff (mhsa_qkv_W 1 d Wq Wk Wv)
                                       (mhsa_qkv_b 1 d bq bk bv))).backward X
        ((colSlabwise_has_vjp_mat (mhsa_g_has_vjp_mat N d) (mhsa_g_flat_diff N d)
            (heads := 1)).backward
          ((fun X' : Mat N (1 * d) => fun n =>
              dense (mhsa_qkv_W 1 d Wq Wk Wv) (mhsa_qkv_b 1 d bq bk bv) (X' n)) X)
          ((rowwise_has_vjp_mat (dense_has_vjp Wo bo) (dense_diff Wo bo)).backward
            (colSlabApply (mhsa_g N d) (heads := 1)
              ((fun X' : Mat N (1 * d) => fun n =>
                 dense (mhsa_qkv_W 1 d Wq Wk Wv) (mhsa_qkv_b 1 d bq bk bv) (X' n)) X))
            dh)) r c = _
  show Mat.mulVec (mhsa_qkv_W 1 d Wq Wk Wv)
        (fun kj => (colSlabwise_has_vjp_mat (mhsa_g_has_vjp_mat N d) (mhsa_g_flat_diff N d)
            (heads := 1)).backward
          (fun n => dense (mhsa_qkv_W 1 d Wq Wk Wv) (mhsa_qkv_b 1 d bq bk bv) (X n))
          (fun n => Mat.mulVec Wo (dh n))
          r kj) c = _
  set M0 : Mat N (1 * (3 * d)) :=
    fun n => dense (mhsa_qkv_W 1 d Wq Wk Wv) (mhsa_qkv_b 1 d bq bk bv) (X n) with hM0
  set dY0 : Mat N (1 * d) := fun n => Mat.mulVec Wo (dh n) with hdY0
  have hslab : ∀ kj : Fin (1 * (3 * d)),
      (colSlabwise_has_vjp_mat (mhsa_g_has_vjp_mat N d) (mhsa_g_flat_diff N d)
          (heads := 1)).backward M0 dY0 r kj
        = (mhsa_g_has_vjp_mat N d).backward
            (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv ((0 : Fin 1), j_in)))
            (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv ((0 : Fin 1), j_out)))
            r (finProdFinEquiv.symm kj).2 := by
    intro kj
    show (mhsa_g_has_vjp_mat N d).backward
          (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv ((finProdFinEquiv.symm kj).1, j_in)))
          (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv ((finProdFinEquiv.symm kj).1, j_out)))
          r (finProdFinEquiv.symm kj).2 = _
    rw [Subsingleton.elim (finProdFinEquiv.symm kj).1 (0 : Fin 1)]
  simp only [hslab]
  have hproj0 : mhsa_proj_c (0 : Fin 3)
      (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv ((0 : Fin 1), j_in)))
      = (fun r' (j : Fin d) => dense Wq bq (X r') (finProdFinEquiv ((0:Fin 1), j))) := by
    funext r' j; unfold mhsa_proj_c; show dense _ _ _ _ = _
    unfold dense; simp only [mhsa_qkv_W_eq0, mhsa_qkv_b_eq0]
  have hproj1 : mhsa_proj_c (1 : Fin 3)
      (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv ((0 : Fin 1), j_in)))
      = (fun r' (j : Fin d) => dense Wk bk (X r') (finProdFinEquiv ((0:Fin 1), j))) := by
    funext r' j; unfold mhsa_proj_c; show dense _ _ _ _ = _
    unfold dense; simp only [mhsa_qkv_W_eq1, mhsa_qkv_b_eq1]
  have hproj2 : mhsa_proj_c (2 : Fin 3)
      (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv ((0 : Fin 1), j_in)))
      = (fun r' (j : Fin d) => dense Wv bv (X r') (finProdFinEquiv ((0:Fin 1), j))) := by
    funext r' j; unfold mhsa_proj_c; show dense _ _ _ _ = _
    unfold dense; simp only [mhsa_qkv_W_eq2, mhsa_qkv_b_eq2]
  have hdz : (fun kj : Fin (1 * (3 * d)) =>
        (mhsa_g_has_vjp_mat N d).backward
          (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv ((0 : Fin 1), j_in)))
          (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv ((0 : Fin 1), j_out)))
          r (finProdFinEquiv.symm kj).2)
      = (fun (idx : Fin (1 * (3 * d))) =>
          let p := finProdFinEquiv.symm idx
          let q := finProdFinEquiv.symm p.2
          if q.1 = (0 : Fin 3) then
            sdpa_back_Q N d
              (fun r' j => dense Wq bq (X r') (finProdFinEquiv ((0:Fin 1), j)))
              (fun r' j => dense Wk bk (X r') (finProdFinEquiv ((0:Fin 1), j)))
              (fun r' j => dense Wv bv (X r') (finProdFinEquiv ((0:Fin 1), j)))
              (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv ((0:Fin 1), j))) r q.2
          else if q.1 = (1 : Fin 3) then
            sdpa_back_K N d
              (fun r' j => dense Wq bq (X r') (finProdFinEquiv ((0:Fin 1), j)))
              (fun r' j => dense Wk bk (X r') (finProdFinEquiv ((0:Fin 1), j)))
              (fun r' j => dense Wv bv (X r') (finProdFinEquiv ((0:Fin 1), j)))
              (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv ((0:Fin 1), j))) r q.2
          else
            sdpa_back_V N d
              (fun r' j => dense Wq bq (X r') (finProdFinEquiv ((0:Fin 1), j)))
              (fun r' j => dense Wk bk (X r') (finProdFinEquiv ((0:Fin 1), j)))
              (fun r' j => dense Wv bv (X r') (finProdFinEquiv ((0:Fin 1), j)))
              (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv ((0:Fin 1), j))) r q.2) := by
    funext kj
    rw [show (mhsa_g_has_vjp_mat N d).backward
            (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv ((0 : Fin 1), j_in)))
            (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv ((0 : Fin 1), j_out)))
            r (finProdFinEquiv.symm kj).2
          = (let p := finProdFinEquiv.symm (finProdFinEquiv.symm kj).2
             if p.1 = (0 : Fin 3) then
               sdpa_back_Q N d
                 (mhsa_proj_c 0 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv ((0:Fin 1), j_in))))
                 (mhsa_proj_c 1 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv ((0:Fin 1), j_in))))
                 (mhsa_proj_c 2 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv ((0:Fin 1), j_in))))
                 (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv ((0:Fin 1), j_out))) r p.2
             else if p.1 = (1 : Fin 3) then
               sdpa_back_K N d
                 (mhsa_proj_c 0 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv ((0:Fin 1), j_in))))
                 (mhsa_proj_c 1 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv ((0:Fin 1), j_in))))
                 (mhsa_proj_c 2 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv ((0:Fin 1), j_in))))
                 (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv ((0:Fin 1), j_out))) r p.2
             else
               sdpa_back_V N d
                 (mhsa_proj_c 0 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv ((0:Fin 1), j_in))))
                 (mhsa_proj_c 1 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv ((0:Fin 1), j_in))))
                 (mhsa_proj_c 2 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv ((0:Fin 1), j_in))))
                 (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv ((0:Fin 1), j_out))) r p.2)
        from rfl]
    rw [hproj0, hproj1, hproj2, hdY0]
  rw [hdz]
  rw [qkv_back_fanin N d Wq Wk Wv
        (sdpa_back_Q N d _ _ _ _) (sdpa_back_K N d _ _ _ _) (sdpa_back_V N d _ _ _ _) r c]
  rfl

/-- The proven MHSA VJP's backward at heads = 1 IS the collapsed three-way fan-in. -/
theorem mhsa_backward_collapse (N d : Nat)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (X dh : Mat N (1 * d)) :
    (mhsa_has_vjp_mat N 1 d Wq Wk Wv Wo bq bk bv bo).backward X dh
      = mhsaBackCollapsed N d Wq Wk Wv Wo bq bk bv bo X dh := by
  rw [← mhsaClean_backward_eq, mhsaClean_backward_collapse]

-- ── SDPA-back width bridges: lifting the inner-`d` `sdpa_back_*` (which
--    `mhsaBackCollapsed` uses, via `mhsa_g`) to the `1*d`-spelled `sdpa_back_*`
--    (which the ViTChainClose ties + the `1*d` forward graph use). The one-head
--    column gather collapses the matmuls (`matmul_oh`), and `sdpa_scale (1*d) =
--    sdpa_scale d`. ──

/-- Gather a `Mat N (1*d)` to `Mat N d` through the head-0 column slice. -/
private noncomputable def gth {N d : Nat} (M1 : Mat N (1 * d)) : Mat N d :=
  fun r j => M1 r (finProdFinEquiv ((0:Fin 1), j))

private lemma matmul_oh {N d : Nat} (A1 B1 : Mat N (1 * d)) :
    Mat.mul (gth A1) (Mat.transpose (gth B1)) = Mat.mul A1 (Mat.transpose B1) := by
  funext i j; unfold Mat.mul Mat.transpose gth
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin 1 × Fin d ≃ Fin (1*d)) (fun k => A1 i k * B1 j k)]
  rw [Fintype.sum_prod_type, Fin.sum_univ_one]

private lemma weights_oh {N d : Nat} (Q1 K1 : Mat N (1 * d)) :
    sdpa_weights N d (gth Q1) (gth K1) = sdpa_weights N (1 * d) Q1 K1 := by
  unfold sdpa_weights
  have hsc : sdpa_scale (1 * d) = sdpa_scale d := by unfold sdpa_scale; rw [Nat.one_mul]
  rw [matmul_oh, hsc]

private lemma dweights_oh {N d : Nat} (V1 dAtt1 : Mat N (1 * d)) :
    sdpa_dWeights (gth V1) (gth dAtt1) = sdpa_dWeights V1 dAtt1 := by
  unfold sdpa_dWeights; funext i j; unfold Mat.mul Mat.transpose gth
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin 1 × Fin d ≃ Fin (1*d)) (fun k => dAtt1 i k * V1 j k)]
  rw [Fintype.sum_prod_type, Fin.sum_univ_one]

private lemma dscores_oh {N d : Nat} (Q1 K1 V1 dAtt1 : Mat N (1 * d)) :
    sdpa_dScores N d (gth Q1) (gth K1) (gth V1) (gth dAtt1)
      = sdpa_dScores N (1 * d) Q1 K1 V1 dAtt1 := by
  unfold sdpa_dScores sdpa_dScaled
  have hsc : sdpa_scale (1 * d) = sdpa_scale d := by unfold sdpa_scale; rw [Nat.one_mul]
  rw [weights_oh, dweights_oh, hsc]

/-- The `1*d`-lifted flatten of a `Mat N d` (the cotangent shape the per-token
    `denseRowBack Wq/Wk/Wv` of `mhsaBackCollapsed` reads). -/
noncomputable def liftFlat (N d : Nat) (M : Mat N d) : Vec (N * (1 * d)) :=
  Mat.flatten (fun r (cc : Fin (1 * d)) => M r (finProdFinEquiv.symm cc).2)

private lemma liftFlat_mul_gth_right {N d : Nat} (DS : Mat N N) (B1 : Mat N (1 * d)) :
    liftFlat N d (Mat.mul DS (gth B1)) = Mat.flatten (Mat.mul DS B1) := by
  unfold liftFlat; funext k; unfold Mat.flatten
  have hmul : ∀ (r : Fin N) (j' : Fin d),
      Mat.mul DS (gth B1) r j' = Mat.mul DS B1 r (finProdFinEquiv ((0:Fin 1), j')) := by
    intro r j'; unfold Mat.mul gth; rfl
  show Mat.mul DS (gth B1) (finProdFinEquiv.symm k).1
        (finProdFinEquiv.symm (finProdFinEquiv.symm k).2).2 = _
  rw [hmul]
  show Mat.mul DS B1 (finProdFinEquiv.symm k).1
        (finProdFinEquiv ((0:Fin 1), (finProdFinEquiv.symm (finProdFinEquiv.symm k).2).2)) = _
  congr 1
  conv_rhs => rw [show (finProdFinEquiv.symm k).2
      = finProdFinEquiv (finProdFinEquiv.symm (finProdFinEquiv.symm k).2) from
        (Equiv.apply_symm_apply _ _).symm]
  congr 1; exact Prod.ext (Subsingleton.elim _ _) rfl

theorem sdpa_back_Q_oh {N d : Nat} (Q1 K1 V1 dAtt1 : Mat N (1 * d)) :
    liftFlat N d (sdpa_back_Q N d (gth Q1) (gth K1) (gth V1) (gth dAtt1))
      = Mat.flatten (sdpa_back_Q N (1 * d) Q1 K1 V1 dAtt1) := by
  unfold sdpa_back_Q; rw [dscores_oh]
  exact liftFlat_mul_gth_right (sdpa_dScores N (1 * d) Q1 K1 V1 dAtt1) K1

theorem sdpa_back_K_oh {N d : Nat} (Q1 K1 V1 dAtt1 : Mat N (1 * d)) :
    liftFlat N d (sdpa_back_K N d (gth Q1) (gth K1) (gth V1) (gth dAtt1))
      = Mat.flatten (sdpa_back_K N (1 * d) Q1 K1 V1 dAtt1) := by
  unfold sdpa_back_K; rw [dscores_oh]
  exact liftFlat_mul_gth_right (Mat.transpose (sdpa_dScores N (1 * d) Q1 K1 V1 dAtt1)) Q1

theorem sdpa_back_V_oh {N d : Nat} (Q1 K1 V1 dAtt1 : Mat N (1 * d)) :
    liftFlat N d (sdpa_back_V N d (gth Q1) (gth K1) (gth V1) (gth dAtt1))
      = Mat.flatten (sdpa_back_V N (1 * d) Q1 K1 V1 dAtt1) := by
  unfold sdpa_back_V; rw [weights_oh]
  exact liftFlat_mul_gth_right (Mat.transpose (sdpa_weights N (1 * d) Q1 K1)) dAtt1

-- ── The MHSA backward in ViTChainClose's renderable cotangent forms ──

set_option maxHeartbeats 2000000 in
/-- The collapsed MHSA backward (flattened) IS the ViTChainClose three-way
    LN₁-fan-in over `vitCotD{Q,K,V}` (the `matmulF`/`scaleF`/`softmaxRowBack`/
    `transposeF` rendered SDPA backwards). The per-projection cotangents are the
    `1*d`-spelled `sdpa_back_{Q,K,V}` (`vitCotD*_eq_sdpa_back_*`), lifted from the
    inner-`d` ones `mhsaBackCollapsed` carries via the `sdpa_back_*_oh` bridges. -/
theorem mhsaBackCollapsed_eq_vitCot (N d : Nat)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (X dh : Mat N (1 * d)) :
    Mat.flatten (mhsaBackCollapsed N d Wq Wk Wv Wo bq bk bv bo X dh)
      = vitCotLn1 Wq Wk Wv
          (vitCotDQ (1 * d)
            (Mat.flatten (fun i j => sdpa_scale (1*d) *
              Mat.mul (fun r => dense Wq bq (X r)) (Mat.transpose (fun r => dense Wk bk (X r))) i j))
            (Mat.flatten (fun r => dense Wk bk (X r)))
            (Mat.flatten (fun r => dense Wv bv (X r)))
            (Mat.flatten (fun r => Mat.mulVec Wo (dh r))))
          (vitCotDK (1 * d)
            (Mat.flatten (fun i j => sdpa_scale (1*d) *
              Mat.mul (fun r => dense Wq bq (X r)) (Mat.transpose (fun r => dense Wk bk (X r))) i j))
            (Mat.flatten (fun r => dense Wq bq (X r)))
            (Mat.flatten (fun r => dense Wv bv (X r)))
            (Mat.flatten (fun r => Mat.mulVec Wo (dh r))))
          (vitCotDV
            (Mat.flatten (sdpa_weights N (1*d) (fun r => dense Wq bq (X r)) (fun r => dense Wk bk (X r))))
            (Mat.flatten (fun r => Mat.mulVec Wo (dh r)))) := by
  set Q1 : Mat N (1*d) := fun r => dense Wq bq (X r) with hQ1
  set K1 : Mat N (1*d) := fun r => dense Wk bk (X r) with hK1
  set V1 : Mat N (1*d) := fun r => dense Wv bv (X r) with hV1
  set dAtt1 : Mat N (1*d) := fun r => Mat.mulVec Wo (dh r) with hdAtt1
  rw [vitCotDQ_eq_sdpa_back_Q N (1*d) Q1 K1 V1 dAtt1,
      vitCotDK_eq_sdpa_back_K N (1*d) Q1 K1 V1 dAtt1,
      vitCotDV_eq_sdpa_back_V N (1*d) Q1 K1 V1 dAtt1]
  -- The collapsed cotangents `Qg/Kg/Vg/dAttg` are `gth`-gathers of `Q1/K1/V1/dAtt1`.
  have hgQ : (fun (r : Fin N) (j : Fin d) => dense Wq bq (X r) (finProdFinEquiv ((0:Fin 1), j)))
              = gth Q1 := rfl
  have hgK : (fun (r : Fin N) (j : Fin d) => dense Wk bk (X r) (finProdFinEquiv ((0:Fin 1), j)))
              = gth K1 := rfl
  have hgV : (fun (r : Fin N) (j : Fin d) => dense Wv bv (X r) (finProdFinEquiv ((0:Fin 1), j)))
              = gth V1 := rfl
  have hgA : (fun (r : Fin N) (j : Fin d) => Mat.mulVec Wo (dh r) (finProdFinEquiv ((0:Fin 1), j)))
              = gth dAtt1 := rfl
  have hQ : (fun (r : Fin N) (cc : Fin (1*d)) =>
              sdpa_back_Q N d (gth Q1) (gth K1) (gth V1) (gth dAtt1) r (finProdFinEquiv.symm cc).2)
            = Mat.unflatten (Mat.flatten (sdpa_back_Q N (1*d) Q1 K1 V1 dAtt1)) := by
    rw [← sdpa_back_Q_oh Q1 K1 V1 dAtt1]; unfold liftFlat; rw [Mat.unflatten_flatten]
  have hK : (fun (r : Fin N) (cc : Fin (1*d)) =>
              sdpa_back_K N d (gth Q1) (gth K1) (gth V1) (gth dAtt1) r (finProdFinEquiv.symm cc).2)
            = Mat.unflatten (Mat.flatten (sdpa_back_K N (1*d) Q1 K1 V1 dAtt1)) := by
    rw [← sdpa_back_K_oh Q1 K1 V1 dAtt1]; unfold liftFlat; rw [Mat.unflatten_flatten]
  have hV : (fun (r : Fin N) (cc : Fin (1*d)) =>
              sdpa_back_V N d (gth Q1) (gth K1) (gth V1) (gth dAtt1) r (finProdFinEquiv.symm cc).2)
            = Mat.unflatten (Mat.flatten (sdpa_back_V N (1*d) Q1 K1 V1 dAtt1)) := by
    rw [← sdpa_back_V_oh Q1 K1 V1 dAtt1]; unfold liftFlat; rw [Mat.unflatten_flatten]
  unfold mhsaBackCollapsed vitCotLn1 rowDenseBackFlat
  simp only [hgQ, hgK, hgV, hgA]
  funext k
  unfold Mat.flatten
  dsimp only []
  rw [show (fun cc => sdpa_back_Q N d (gth Q1) (gth K1) (gth V1) (gth dAtt1)
              (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm cc).2)
        = Mat.unflatten (Mat.flatten (sdpa_back_Q N (1*d) Q1 K1 V1 dAtt1)) (finProdFinEquiv.symm k).1
      from congrFun hQ _,
      show (fun cc => sdpa_back_K N d (gth Q1) (gth K1) (gth V1) (gth dAtt1)
              (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm cc).2)
        = Mat.unflatten (Mat.flatten (sdpa_back_K N (1*d) Q1 K1 V1 dAtt1)) (finProdFinEquiv.symm k).1
      from congrFun hK _,
      show (fun cc => sdpa_back_V N d (gth Q1) (gth K1) (gth V1) (gth dAtt1)
              (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm cc).2)
        = Mat.unflatten (Mat.flatten (sdpa_back_V N (1*d) Q1 K1 V1 dAtt1)) (finProdFinEquiv.symm k).1
      from congrFun hV _]
  rfl

-- ── The MHSA backward graph (heads = 1) ──

/-- SDPA dQ-segment subgraph: `matmulF(scaleF(softmaxRowBack(matmulF(dAtt, transposeF v))), k)`
    — denotes `vitCotDQ`. `ss` = saved pre-softmax scaled scores; `k`/`v`/`dAtt` saved. -/
noncomputable def sdpaBackQGraph (Np1 D : Nat) (ss : Vec (Np1*Np1))
    (k v : Vec (Np1*D)) (e : SHlo (Np1*D)) : SHlo (Np1*D) :=
  .matmulF (m := Np1) (k := Np1) (n := D)
    (.scaleF "%sdpaS" (sdpa_scale D)
      (.softmaxRowBack "%ss" ss
        (.matmulF (m := Np1) (k := D) (n := Np1) e (.transposeF (.operand "%v" v)))))
    (.operand "%k" k)

/-- SDPA dK-segment subgraph — denotes `vitCotDK`. -/
noncomputable def sdpaBackKGraph (Np1 D : Nat) (ss : Vec (Np1*Np1))
    (q v : Vec (Np1*D)) (e : SHlo (Np1*D)) : SHlo (Np1*D) :=
  .matmulF (m := Np1) (k := Np1) (n := D)
    (.transposeF (m := Np1) (n := Np1)
      (.scaleF "%sdpaS" (sdpa_scale D)
        (.softmaxRowBack "%ss" ss
          (.matmulF (m := Np1) (k := D) (n := Np1) e (.transposeF (.operand "%v" v))))))
    (.operand "%q" q)

/-- SDPA dV-segment subgraph — denotes `vitCotDV`. `p` = saved post-softmax weights. -/
noncomputable def sdpaBackVGraph (Np1 D : Nat) (p : Vec (Np1*Np1))
    (e : SHlo (Np1*D)) : SHlo (Np1*D) :=
  .matmulF (m := Np1) (k := Np1) (n := D)
    (.transposeF (m := Np1) (n := Np1) (.operand "%p" p)) e

theorem sdpaBackQGraph_faithful (Np1 D : Nat) (ss : Vec (Np1*Np1))
    (k v : Vec (Np1*D)) (e : SHlo (Np1*D)) :
    den (sdpaBackQGraph Np1 D ss k v e) = vitCotDQ D ss k v (den e) := by
  unfold sdpaBackQGraph vitCotDQ vitCotDS vitCotDP
  simp only [matmulF_faithful, scaleF_faithful, softmaxRowBack_faithful,
    transposeF_faithful, den_operand]

theorem sdpaBackKGraph_faithful (Np1 D : Nat) (ss : Vec (Np1*Np1))
    (q v : Vec (Np1*D)) (e : SHlo (Np1*D)) :
    den (sdpaBackKGraph Np1 D ss q v e) = vitCotDK D ss q v (den e) := by
  unfold sdpaBackKGraph vitCotDK vitCotDS vitCotDP
  simp only [matmulF_faithful, transposeF_faithful, scaleF_faithful, softmaxRowBack_faithful,
    den_operand]

theorem sdpaBackVGraph_faithful (Np1 D : Nat) (p : Vec (Np1*Np1)) (e : SHlo (Np1*D)) :
    den (sdpaBackVGraph Np1 D p e) = vitCotDV p (den e) := by
  unfold sdpaBackVGraph vitCotDV
  simp only [matmulF_faithful, transposeF_faithful, den_operand]

/-- The whole MHSA backward graph (heads = 1): the three-way LN₁-fan-in
    `addV (addV (denseRowBack Wq (dQ)) (denseRowBack Wk (dK))) (denseRowBack Wv (dV))`,
    where each `d{Q,K,V}` is the SDPA backward subgraph fed the `Wo`-backward
    `denseRowBack Wo (%dh)` (the cotangent at the SDPA output). Saved activations:
    `Q/K/V` (the dense projections of `X`), the pre-softmax scores `ss`, and the
    post-softmax weights `p`. -/
noncomputable def mhsaBackGraph (Np1 d : Nat)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d))
    (Q K V : Vec (Np1*(1*d))) (ss : Vec (Np1*Np1)) (p : Vec (Np1*Np1))
    (dh : Vec (Np1*(1*d))) : SHlo (Np1*(1*d)) :=
  let dAtt := SHlo.denseRowBack "%Wo" Wo (.operand "%dh" dh)
  .addV
    (.addV
      (.denseRowBack "%Wq" Wq (sdpaBackQGraph Np1 (1*d) ss K V dAtt))
      (.denseRowBack "%Wk" Wk (sdpaBackKGraph Np1 (1*d) ss Q V dAtt)))
    (.denseRowBack "%Wv" Wv (sdpaBackVGraph Np1 (1*d) p dAtt))

set_option maxHeartbeats 2000000 in
/-- **MHSA backward-graph faithfulness (heads = 1).** The three-way SDPA-backward
    fan-in graph denotes the proven `mhsa_has_vjp_mat.backward` (flattened), at the
    saved dense projections `Q = dense Wq, K = dense Wk, V = dense Wv`, pre-softmax
    scores `ss`, and post-softmax weights `p`. -/
theorem mhsaBackGraph_faithful (Np1 d : Nat)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (X dh : Mat Np1 (1 * d)) :
    den (mhsaBackGraph Np1 d Wq Wk Wv Wo
          (Mat.flatten (fun r => dense Wq bq (X r)))
          (Mat.flatten (fun r => dense Wk bk (X r)))
          (Mat.flatten (fun r => dense Wv bv (X r)))
          (Mat.flatten (fun i j => sdpa_scale (1*d) *
            Mat.mul (fun r => dense Wq bq (X r)) (Mat.transpose (fun r => dense Wk bk (X r))) i j))
          (Mat.flatten (sdpa_weights Np1 (1*d) (fun r => dense Wq bq (X r))
            (fun r => dense Wk bk (X r))))
          (Mat.flatten dh))
      = Mat.flatten ((mhsa_has_vjp_mat Np1 1 d Wq Wk Wv Wo bq bk bv bo).backward X dh) := by
  rw [mhsa_backward_collapse, mhsaBackCollapsed_eq_vitCot]
  -- den (addV (addV Q K) V) = den Q + den K + den V; each branch is `denseRowBack` of
  -- the SDPA-back subgraph (which denotes `vitCotD{Q,K,V}` at the `Wo`-back cotangent).
  unfold mhsaBackGraph
  rw [show den (.addV
            (.addV (.denseRowBack "%Wq" Wq (sdpaBackQGraph Np1 (1*d) _ _ _
                      (.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh)))))
                   (.denseRowBack "%Wk" Wk (sdpaBackKGraph Np1 (1*d) _ _ _
                      (.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh))))))
            (.denseRowBack "%Wv" Wv (sdpaBackVGraph Np1 (1*d) _
                      (.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh))))))
        = fun j =>
            den (.denseRowBack "%Wq" Wq (sdpaBackQGraph Np1 (1*d)
                  (Mat.flatten (fun i j => sdpa_scale (1*d) *
                    Mat.mul (fun r => dense Wq bq (X r)) (Mat.transpose (fun r => dense Wk bk (X r))) i j))
                  (Mat.flatten (fun r => dense Wk bk (X r)))
                  (Mat.flatten (fun r => dense Wv bv (X r)))
                  (.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh))))) j
            + den (.denseRowBack "%Wk" Wk (sdpaBackKGraph Np1 (1*d)
                  (Mat.flatten (fun i j => sdpa_scale (1*d) *
                    Mat.mul (fun r => dense Wq bq (X r)) (Mat.transpose (fun r => dense Wk bk (X r))) i j))
                  (Mat.flatten (fun r => dense Wq bq (X r)))
                  (Mat.flatten (fun r => dense Wv bv (X r)))
                  (.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh))))) j
            + den (.denseRowBack "%Wv" Wv (sdpaBackVGraph Np1 (1*d)
                  (Mat.flatten (sdpa_weights Np1 (1*d) (fun r => dense Wq bq (X r))
                    (fun r => dense Wk bk (X r))))
                  (.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh))))) j
        from rfl]
  -- Each branch: denseRowBack Wq of the SDPA-back subgraph denoting vitCotDQ at the Wo-back.
  have hdAtt : den (SHlo.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh)))
      = Mat.flatten (fun r => Mat.mulVec Wo (dh r)) := by
    rw [denseRowBack_faithful, den_operand]; unfold rowDenseBackFlat
    rw [Mat.unflatten_flatten]
  funext j
  rw [denseRowBack_faithful, denseRowBack_faithful, denseRowBack_faithful,
      sdpaBackQGraph_faithful, sdpaBackKGraph_faithful, sdpaBackVGraph_faithful, hdAtt]
  -- RHS is `vitCotLn1 Wq Wk Wv (vitCotDQ …) (vitCotDK …) (vitCotDV …)`.
  unfold vitCotLn1
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Stage 2 — Attention sublayer backward graph (MHSA + LN₁ + residual)
-- ════════════════════════════════════════════════════════════════

/-- The attention-sublayer non-trivial arm backward graph (`mhsa ∘ LN₁`; outermost
    backward token = earliest forward op = LN₁):

      `lnRowBack(LN₁ @ x) ∘ mhsaBackGraph(@ X = LN₁ x)`

    `x` (saved pre-LN₁ block input) feeds the LN₁ backward; the saved Q/K/V/scores/
    weights inside `mhsaBackGraph` are computed at `X = LN₁ x`. -/
noncomputable def attnSublayerInnerBackGraph (Np1 d : Nat) (ε γ1 : ℝ)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (x : Vec (Np1 * (1 * d))) (X : Mat Np1 (1 * d)) (e : SHlo (Np1 * (1 * d))) :
    SHlo (Np1 * (1 * d)) :=
  .lnRowBack "%g1" "%x" "ε" ε γ1 x
    (mhsaBackGraph Np1 d Wq Wk Wv Wo
      (Mat.flatten (fun r => dense Wq bq (X r)))
      (Mat.flatten (fun r => dense Wk bk (X r)))
      (Mat.flatten (fun r => dense Wv bv (X r)))
      (Mat.flatten (fun i j => sdpa_scale (1*d) *
        Mat.mul (fun r => dense Wq bq (X r)) (Mat.transpose (fun r => dense Wk bk (X r))) i j))
      (Mat.flatten (sdpa_weights Np1 (1*d) (fun r => dense Wq bq (X r))
        (fun r => dense Wk bk (X r))))
      (den e))

/-- **Attention-sublayer inner-arm backward-graph faithfulness.** The reverse-order
    chain denotes the proven `(vjpMat_comp LN₁ mhsa).backward x ·`. `X = LN₁ x` is
    the saved MHSA input. -/
theorem attnSublayerInnerBackGraph_faithful (Np1 d : Nat) (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (x : Mat Np1 (1 * d)) (dh : Mat Np1 (1 * d)) :
    den (attnSublayerInnerBackGraph Np1 d ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten x)
          (fun r => layerNormForward (1 * d) ε γ1 β1 (x r)) (.operand "%dh" (Mat.flatten dh)))
      = Mat.flatten
          ((layerNorm_per_token_has_vjp_mat Np1 (1 * d) ε γ1 β1 hε).backward x
            ((mhsa_has_vjp_mat Np1 1 d Wq Wk Wv Wo bq bk bv bo).backward
              (fun r => layerNormForward (1 * d) ε γ1 β1 (x r)) dh)) := by
  simp only [attnSublayerInnerBackGraph, lnRowBack_faithful, den_operand]
  rw [mhsaBackGraph_faithful Np1 d Wq Wk Wv Wo bq bk bv bo
        (fun r => layerNormForward (1 * d) ε γ1 β1 (x r)) dh]
  rw [rowLNBackFlat_eq_backward (β := β1) ε γ1 hε x
        ((mhsa_has_vjp_mat Np1 1 d Wq Wk Wv Wo bq bk bv bo).backward
          (fun r => layerNormForward (1 * d) ε γ1 β1 (x r)) dh)]

/-- The whole attention-sublayer backward graph (inner arm + identity skip):
    `addV (innerBack … (%dh)) (%dh)`. -/
noncomputable def attnSublayerBackGraph (Np1 d : Nat) (ε γ1 : ℝ)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (x : Vec (Np1 * (1 * d))) (X : Mat Np1 (1 * d)) (dh : Vec (Np1 * (1 * d))) :
    SHlo (Np1 * (1 * d)) :=
  .addV (attnSublayerInnerBackGraph Np1 d ε γ1 Wq Wk Wv Wo bq bk bv bo x X
          (.operand "%dh" dh))
        (.operand "%dh" dh)

set_option maxHeartbeats 1000000 in
/-- The attention-sublayer VJP's backward unfolds (structurally) to the `biPathMat`
    residual fan-in: `dh + LN₁.backward x (mhsa.backward (LN₁ x) dh)`. -/
private theorem attnSublayer_backward_unfold (Np1 d : Nat) (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (x dh : Mat Np1 (1 * d)) :
    (transformerAttnSublayer_has_vjp_mat Np1 1 d ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo).backward x dh
      = fun i k => dh i k +
          (layerNorm_per_token_has_vjp_mat Np1 (1 * d) ε γ1 β1 hε).backward x
            ((mhsa_has_vjp_mat Np1 1 d Wq Wk Wv Wo bq bk bv bo).backward
              (fun r => layerNormForward (1 * d) ε γ1 β1 (x r)) dh) i k := rfl

/-- **Attention sublayer backward-graph faithfulness (Stage 2 capstone).** The
    residual-fan-in graph denotes the proven `transformerAttnSublayer_has_vjp_mat`
    backward (heads = 1), under `0 < ε`. The `biPathMat` skip arm is the identity;
    the non-trivial arm is the LN₁-back of the MHSA back. The forward activations
    fed: `x` (LN₁'s pre-norm input, the block input) and `X = LN₁ x` (MHSA input). -/
theorem attnSublayerBackGraph_faithful (Np1 d : Nat) (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (x : Mat Np1 (1 * d)) (dh : Mat Np1 (1 * d)) :
    den (attnSublayerBackGraph Np1 d ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten x)
          (fun r => layerNormForward (1 * d) ε γ1 β1 (x r)) (Mat.flatten dh))
      = Mat.flatten ((transformerAttnSublayer_has_vjp_mat Np1 1 d ε γ1 β1 hε
          Wq Wk Wv Wo bq bk bv bo).backward x dh) := by
  rw [attnSublayer_backward_unfold]
  funext j
  show den (attnSublayerInnerBackGraph Np1 d ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten x)
            (fun r => layerNormForward (1 * d) ε γ1 β1 (x r))
            (.operand "%dh" (Mat.flatten dh))) j + Mat.flatten dh j = _
  rw [attnSublayerInnerBackGraph_faithful Np1 d ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo x dh]
  -- Both sides are flatten of the pointwise sum; commute the addition.
  unfold Mat.flatten
  exact add_comm _ _

-- ════════════════════════════════════════════════════════════════
-- § Stage 3 — Whole transformer-block backward graph
-- ════════════════════════════════════════════════════════════════

/-- The whole transformer-block backward graph (heads = 1). `transformerBlock =
    mlpSublayer ∘ attnSublayer`, so `block.backward A dY = attn.backward A
    (mlp.backward (attn A) dY)`: the MLP-sublayer backward graph (at the saved
    attention-sublayer output `h`) feeds the attention-sublayer backward graph's
    cotangent (at the saved block input `A`). Saved activations: `A` (block input),
    `h = attnSublayer A` (attention-sublayer output / MLP-sublayer input). -/
noncomputable def transformerBlockBackGraph (Np1 d mlpDim : Nat)
    (ε γ1 β1 γ2 β2 : ℝ)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (Wfc1 : Mat (1 * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d))
    (A : Mat Np1 (1 * d)) (h : Mat Np1 (1 * d)) (dY : Vec (Np1 * (1 * d))) :
    SHlo (Np1 * (1 * d)) :=
  -- MLP sublayer backward at `h`, producing the cotangent at the attn-sublayer output;
  -- then the attention sublayer backward at `A`.
  attnSublayerBackGraph Np1 d ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten A)
    (fun r => layerNormForward (1 * d) ε γ1 β1 (A r))
    (den (mlpSublayerBackGraph ε γ2 Wfc1 bfc1 Wfc2 (Mat.flatten h)
            (fun r => layerNormForward (1 * d) ε γ2 β2 (h r)) dY))

set_option maxHeartbeats 1000000 in
/-- The transformer-block VJP's backward unfolds (structurally) to the `vjpMat_comp`
    chain: `attn.backward A (mlp.backward (attn A) dY)`. -/
private theorem transformerBlock_backward_unfold (Np1 d mlpDim : Nat)
    (ε γ1 β1 γ2 β2 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (Wfc1 : Mat (1 * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d)) (bfc2 : Vec (1 * d))
    (A dY : Mat Np1 (1 * d)) :
    (transformerBlock_has_vjp_mat Np1 1 d mlpDim ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward A dY
      = (transformerAttnSublayer_has_vjp_mat Np1 1 d ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo).backward A
          ((transformerMlpSublayer_has_vjp_mat Np1 1 d mlpDim ε γ2 β2 hε
              Wfc1 bfc1 Wfc2 bfc2).backward
            (transformerAttnSublayer Np1 1 d ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A) dY) := rfl

/-- **Whole transformer-block backward-graph faithfulness (Stage 3 capstone).**
    The block backward graph denotes the proven `transformerBlock_has_vjp_mat`
    backward (heads = 1), under `0 < ε`. Wires the MLP-sublayer backward (at the
    saved attention-sublayer output `h = attnSublayer A`) into the attention-sublayer
    backward (at the saved block input `A`), per `block.backward A dY = attn.backward
    A (mlp.backward (attn A) dY)`. -/
theorem transformerBlockBackGraph_faithful (Np1 d mlpDim : Nat)
    (ε γ1 β1 γ2 β2 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (Wfc1 : Mat (1 * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d)) (bfc2 : Vec (1 * d))
    (A dY : Mat Np1 (1 * d)) :
    den (transformerBlockBackGraph Np1 d mlpDim ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
          Wfc1 bfc1 Wfc2 A
          (transformerAttnSublayer Np1 1 d ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A)
          (Mat.flatten dY))
      = Mat.flatten ((transformerBlock_has_vjp_mat Np1 1 d mlpDim ε γ1 β1 hε
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward A dY) := by
  rw [transformerBlock_backward_unfold (bfc2 := bfc2)]
  -- abbreviate the attn-sublayer output `h`.
  set h : Mat Np1 (1 * d) := transformerAttnSublayer Np1 1 d ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A with hh
  -- The block graph = attnSublayerBackGraph at `A` fed `den (mlpSublayerBackGraph at h dY)`.
  show den (attnSublayerBackGraph Np1 d ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten A)
            (fun r => layerNormForward (1 * d) ε γ1 β1 (A r))
            (den (mlpSublayerBackGraph ε γ2 Wfc1 bfc1 Wfc2 (Mat.flatten h)
                    (fun r => layerNormForward (1 * d) ε γ2 β2 (h r)) (Mat.flatten dY)))) = _
  -- the MLP sublayer back graph denotes `mlp.backward h dY` (flattened).
  rw [mlpSublayerBackGraph_faithful ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2 h dY]
  -- the attn sublayer back graph (fed that flattened cotangent) denotes `attn.backward A (·)`.
  rw [attnSublayerBackGraph_faithful Np1 d ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo A
        ((transformerMlpSublayer_has_vjp_mat Np1 1 d mlpDim ε γ2 β2 hε
            Wfc1 bfc1 Wfc2 bfc2).backward h dY)]

end Proofs.StableHLO
