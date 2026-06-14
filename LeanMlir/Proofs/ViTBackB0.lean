import LeanMlir.Proofs.Attention
import LeanMlir.Proofs.ViTFwdGraph
import LeanMlir.Proofs.ViTChainClose
import LeanMlir.Proofs.ViTMultiHead
import LeanMlir.Proofs.ViTDepthK
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

/-- **`HasVJP` backward determinism** — two Vec-level VJPs of the same function
    agree (both equal the shared `pdiv`-contraction, via `.correct`). The Vec
    analogue of `hasVJPMat_backward_det`; lets us tie a CLEANLY-projecting
    `vjp_comp` chain to an opaque tactic-built whole-net VJP without forcing a
    deep `rfl` reduction of the latter. -/
theorem hasVJP_backward_det {m n : Nat} {f : Vec m → Vec n}
    (v v' : HasVJP f) (x : Vec m) (dy : Vec n) :
    v.backward x dy = v'.backward x dy := by
  funext i; rw [v.correct, v'.correct]

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

-- ════════════════════════════════════════════════════════════════
-- § Stage 1 (MULTI-HEAD) — MHSA backward general-`heads` collapse
-- ════════════════════════════════════════════════════════════════

/-! The multi-head analogue of the heads = 1 collapse. Where heads = 1 collapses
the colSlab-lifted backward to a single three-way dense fan-in, general heads
collapses to a SUM over heads: each head `h` slices the dense Q/K/V projections
and the `Wo`-back cotangent to head `h`'s columns, runs `sdpa_back_{Q,K,V}` at
`d_head`, and the qkv-stack dense-back contracts head `h`'s SDPA backward against
the `finProdFinEquiv (h, ·)` columns of `Wq/Wk/Wv`. -/

/-- General-`heads` reindex of a sum over the qkv-slab column axis
    `Fin (heads * (3 * d))` into `(h, c, j)`. -/
private lemma sum_heads_3d {M : Type*} [AddCommMonoid M] (heads d : Nat)
    (f : Fin (heads * (3 * d)) → M) :
    (∑ k : Fin (heads * (3 * d)), f k)
      = ∑ h : Fin heads, ∑ c : Fin 3, ∑ j : Fin d,
          f (finProdFinEquiv (h, finProdFinEquiv (c, j))) := by
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin heads × Fin (3*d) ≃ Fin (heads * (3*d))) f]
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl; intro h _
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin 3 × Fin d ≃ Fin (3*d))
      (fun kk => f (finProdFinEquiv (h, kk)))]
  rw [Fintype.sum_prod_type]

set_option maxHeartbeats 1600000 in
/-- The multi-head QKV-stack dense-back fan-in: contracting the column-stacked
    per-head SDPA backward against `mhsa_qkv_W` splits into a sum over heads of
    the three per-projection dense-backs at head `h`'s columns. -/
private lemma qkv_back_fanin_MH (N heads d : Nat)
    (Wq Wk Wv : Mat (heads * d) (heads * d))
    (dQg dKg dVg : Fin heads → Mat N d) (r : Fin N) (c : Fin (heads * d)) :
    Mat.mulVec (mhsa_qkv_W heads d Wq Wk Wv)
      (fun (kj : Fin (heads * (3 * d))) =>
        let p := finProdFinEquiv.symm kj
        let q := finProdFinEquiv.symm p.2
        if q.1 = (0 : Fin 3) then dQg p.1 r q.2
        else if q.1 = (1 : Fin 3) then dKg p.1 r q.2
        else dVg p.1 r q.2) c
      = ∑ h : Fin heads,
          ((∑ j : Fin d, Wq c (finProdFinEquiv (h, j)) * dQg h r j)
           + (∑ j : Fin d, Wk c (finProdFinEquiv (h, j)) * dKg h r j)
           + (∑ j : Fin d, Wv c (finProdFinEquiv (h, j)) * dVg h r j)) := by
  unfold Mat.mulVec
  rw [sum_heads_3d]
  apply Finset.sum_congr rfl; intro h _
  rw [Fin.sum_univ_three]
  simp only [Equiv.symm_apply_apply,
    mhsa_qkv_W_eq0, mhsa_qkv_W_eq1, mhsa_qkv_W_eq2,
    show (1 : Fin 3) ≠ (0 : Fin 3) from by decide,
    show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
    show (2 : Fin 3) ≠ (1 : Fin 3) from by decide, if_true, if_false]

/-- The collapsed general-`heads` MHSA backward: for each head `h`, slice the
    dense Q/K/V projections and the `Wo`-back cotangent to head `h`'s columns,
    run `sdpa_back_{Q,K,V}` at `d`, then contract per-head against the
    `finProdFinEquiv (h, ·)` columns of `Wq/Wk/Wv`, summed over heads. -/
noncomputable def mhsaBackCollapsedMH (N heads d : Nat)
    (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (X dh : Mat N (heads * d)) : Mat N (heads * d) :=
  let Qg : Fin heads → Mat N d := fun h r j => dense Wq bq (X r) (finProdFinEquiv (h, j))
  let Kg : Fin heads → Mat N d := fun h r j => dense Wk bk (X r) (finProdFinEquiv (h, j))
  let Vg : Fin heads → Mat N d := fun h r j => dense Wv bv (X r) (finProdFinEquiv (h, j))
  let dAttg : Fin heads → Mat N d := fun h r j => Mat.mulVec Wo (dh r) (finProdFinEquiv (h, j))
  let dQg : Fin heads → Mat N d := fun h => sdpa_back_Q N d (Qg h) (Kg h) (Vg h) (dAttg h)
  let dKg : Fin heads → Mat N d := fun h => sdpa_back_K N d (Qg h) (Kg h) (Vg h) (dAttg h)
  let dVg : Fin heads → Mat N d := fun h => sdpa_back_V N d (Qg h) (Kg h) (Vg h) (dAttg h)
  fun r c =>
    ∑ h : Fin heads,
      ((∑ j : Fin d, Wq c (finProdFinEquiv (h, j)) * dQg h r j)
       + (∑ j : Fin d, Wk c (finProdFinEquiv (h, j)) * dKg h r j)
       + (∑ j : Fin d, Wv c (finProdFinEquiv (h, j)) * dVg h r j))

set_option maxHeartbeats 4000000 in
/-- **MHSA backward general-`heads` collapse.** The clean MHSA witness's backward
    equals the per-head sum fan-in `mhsaBackCollapsedMH`. -/
theorem mhsaClean_backward_collapseMH (N heads d : Nat)
    (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (X dh : Mat N (heads * d)) :
    (mhsaClean N heads d Wq Wk Wv Wo bq bk bv bo).backward X dh
      = mhsaBackCollapsedMH N heads d Wq Wk Wv Wo bq bk bv bo X dh := by
  funext r c
  show (rowwise_has_vjp_mat (dense_has_vjp (mhsa_qkv_W heads d Wq Wk Wv)
                                          (mhsa_qkv_b heads d bq bk bv))
                           (dense_diff (mhsa_qkv_W heads d Wq Wk Wv)
                                       (mhsa_qkv_b heads d bq bk bv))).backward X
        ((colSlabwise_has_vjp_mat (mhsa_g_has_vjp_mat N d) (mhsa_g_flat_diff N d)
            (heads := heads)).backward
          ((fun X' : Mat N (heads * d) => fun n =>
              dense (mhsa_qkv_W heads d Wq Wk Wv) (mhsa_qkv_b heads d bq bk bv) (X' n)) X)
          ((rowwise_has_vjp_mat (dense_has_vjp Wo bo) (dense_diff Wo bo)).backward
            (colSlabApply (mhsa_g N d) (heads := heads)
              ((fun X' : Mat N (heads * d) => fun n =>
                 dense (mhsa_qkv_W heads d Wq Wk Wv) (mhsa_qkv_b heads d bq bk bv) (X' n)) X))
            dh)) r c = _
  show Mat.mulVec (mhsa_qkv_W heads d Wq Wk Wv)
        (fun kj => (colSlabwise_has_vjp_mat (mhsa_g_has_vjp_mat N d) (mhsa_g_flat_diff N d)
            (heads := heads)).backward
          (fun n => dense (mhsa_qkv_W heads d Wq Wk Wv) (mhsa_qkv_b heads d bq bk bv) (X n))
          (fun n => Mat.mulVec Wo (dh n))
          r kj) c = _
  set M0 : Mat N (heads * (3 * d)) :=
    fun n => dense (mhsa_qkv_W heads d Wq Wk Wv) (mhsa_qkv_b heads d bq bk bv) (X n) with hM0
  set dY0 : Mat N (heads * d) := fun n => Mat.mulVec Wo (dh n) with hdY0
  -- Per slab column kj: the colSlabwise backward slices to head `(symm kj).1` and runs
  -- `mhsa_g.backward` at that head's slab, read at `(symm kj).2`.
  have hslab : ∀ kj : Fin (heads * (3 * d)),
      (colSlabwise_has_vjp_mat (mhsa_g_has_vjp_mat N d) (mhsa_g_flat_diff N d)
          (heads := heads)).backward M0 dY0 r kj
        = (mhsa_g_has_vjp_mat N d).backward
            (fun r' (j_in : Fin (3 * d)) =>
              M0 r' (finProdFinEquiv ((finProdFinEquiv.symm kj).1, j_in)))
            (fun r' (j_out : Fin d) =>
              dY0 r' (finProdFinEquiv ((finProdFinEquiv.symm kj).1, j_out)))
            r (finProdFinEquiv.symm kj).2 := fun _ => rfl
  -- Rewrite each slab backward into the if-third `sdpa_back_*` form, with the
  -- `mhsa_proj_c` projections collapsed to head-`h`-sliced dense Q/K/V.
  have hdz : (fun kj : Fin (heads * (3 * d)) =>
        (colSlabwise_has_vjp_mat (mhsa_g_has_vjp_mat N d) (mhsa_g_flat_diff N d)
            (heads := heads)).backward M0 dY0 r kj)
      = (fun (kj : Fin (heads * (3 * d))) =>
          let p := finProdFinEquiv.symm kj
          let q := finProdFinEquiv.symm p.2
          if q.1 = (0 : Fin 3) then
            sdpa_back_Q N d
              (fun r' j => dense Wq bq (X r') (finProdFinEquiv (p.1, j)))
              (fun r' j => dense Wk bk (X r') (finProdFinEquiv (p.1, j)))
              (fun r' j => dense Wv bv (X r') (finProdFinEquiv (p.1, j)))
              (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv (p.1, j))) r q.2
          else if q.1 = (1 : Fin 3) then
            sdpa_back_K N d
              (fun r' j => dense Wq bq (X r') (finProdFinEquiv (p.1, j)))
              (fun r' j => dense Wk bk (X r') (finProdFinEquiv (p.1, j)))
              (fun r' j => dense Wv bv (X r') (finProdFinEquiv (p.1, j)))
              (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv (p.1, j))) r q.2
          else
            sdpa_back_V N d
              (fun r' j => dense Wq bq (X r') (finProdFinEquiv (p.1, j)))
              (fun r' j => dense Wk bk (X r') (finProdFinEquiv (p.1, j)))
              (fun r' j => dense Wv bv (X r') (finProdFinEquiv (p.1, j)))
              (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv (p.1, j))) r q.2) := by
    funext kj
    rw [hslab kj]
    set h := (finProdFinEquiv.symm kj).1 with hh
    -- the head-`h`-sliced slab projections collapse to head-`h`-sliced dense Q/K/V.
    have hproj0 : mhsa_proj_c (0 : Fin 3)
        (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv (h, j_in)))
        = (fun r' (j : Fin d) => dense Wq bq (X r') (finProdFinEquiv (h, j))) := by
      funext r' j; unfold mhsa_proj_c; show dense _ _ _ _ = _
      unfold dense; simp only [mhsa_qkv_W_eq0, mhsa_qkv_b_eq0]
    have hproj1 : mhsa_proj_c (1 : Fin 3)
        (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv (h, j_in)))
        = (fun r' (j : Fin d) => dense Wk bk (X r') (finProdFinEquiv (h, j))) := by
      funext r' j; unfold mhsa_proj_c; show dense _ _ _ _ = _
      unfold dense; simp only [mhsa_qkv_W_eq1, mhsa_qkv_b_eq1]
    have hproj2 : mhsa_proj_c (2 : Fin 3)
        (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv (h, j_in)))
        = (fun r' (j : Fin d) => dense Wv bv (X r') (finProdFinEquiv (h, j))) := by
      funext r' j; unfold mhsa_proj_c; show dense _ _ _ _ = _
      unfold dense; simp only [mhsa_qkv_W_eq2, mhsa_qkv_b_eq2]
    rw [show (mhsa_g_has_vjp_mat N d).backward
            (fun r' (j_in : Fin (3 * d)) => M0 r' (finProdFinEquiv (h, j_in)))
            (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv (h, j_out)))
            r (finProdFinEquiv.symm kj).2
          = (let p := finProdFinEquiv.symm (finProdFinEquiv.symm kj).2
             if p.1 = (0 : Fin 3) then
               sdpa_back_Q N d
                 (mhsa_proj_c 0 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv (h, j_in))))
                 (mhsa_proj_c 1 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv (h, j_in))))
                 (mhsa_proj_c 2 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv (h, j_in))))
                 (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv (h, j_out))) r p.2
             else if p.1 = (1 : Fin 3) then
               sdpa_back_K N d
                 (mhsa_proj_c 0 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv (h, j_in))))
                 (mhsa_proj_c 1 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv (h, j_in))))
                 (mhsa_proj_c 2 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv (h, j_in))))
                 (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv (h, j_out))) r p.2
             else
               sdpa_back_V N d
                 (mhsa_proj_c 0 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv (h, j_in))))
                 (mhsa_proj_c 1 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv (h, j_in))))
                 (mhsa_proj_c 2 (fun r' (j_in : Fin (3*d)) => M0 r' (finProdFinEquiv (h, j_in))))
                 (fun r' (j_out : Fin d) => dY0 r' (finProdFinEquiv (h, j_out))) r p.2)
        from rfl]
    rw [hproj0, hproj1, hproj2, hdY0]
  -- Transit through the if-form: `congr` (defeq-tolerant on eta) folds the slab
  -- backward into the if-form, then `qkv_back_fanin_MH` splits the fan-in per head.
  trans (Mat.mulVec (mhsa_qkv_W heads d Wq Wk Wv)
          (fun (kj : Fin (heads * (3 * d))) =>
            let p := finProdFinEquiv.symm kj
            let q := finProdFinEquiv.symm p.2
            if q.1 = (0 : Fin 3) then
              sdpa_back_Q N d
                (fun r' j => dense Wq bq (X r') (finProdFinEquiv (p.1, j)))
                (fun r' j => dense Wk bk (X r') (finProdFinEquiv (p.1, j)))
                (fun r' j => dense Wv bv (X r') (finProdFinEquiv (p.1, j)))
                (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv (p.1, j))) r q.2
            else if q.1 = (1 : Fin 3) then
              sdpa_back_K N d
                (fun r' j => dense Wq bq (X r') (finProdFinEquiv (p.1, j)))
                (fun r' j => dense Wk bk (X r') (finProdFinEquiv (p.1, j)))
                (fun r' j => dense Wv bv (X r') (finProdFinEquiv (p.1, j)))
                (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv (p.1, j))) r q.2
            else
              sdpa_back_V N d
                (fun r' j => dense Wq bq (X r') (finProdFinEquiv (p.1, j)))
                (fun r' j => dense Wk bk (X r') (finProdFinEquiv (p.1, j)))
                (fun r' j => dense Wv bv (X r') (finProdFinEquiv (p.1, j)))
                (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv (p.1, j))) r q.2) c)
  · exact congrFun (congrArg (Mat.mulVec (mhsa_qkv_W heads d Wq Wk Wv)) hdz) c
  · -- `qkv_back_fanin_MH` (defeq-applied via `exact`, beta-tolerant) splits the
    -- fan-in; `mhsaBackCollapsedMH` unfolds to the same per-head sum.
    exact qkv_back_fanin_MH N heads d Wq Wk Wv
          (fun h => sdpa_back_Q N d
            (fun r' j => dense Wq bq (X r') (finProdFinEquiv (h, j)))
            (fun r' j => dense Wk bk (X r') (finProdFinEquiv (h, j)))
            (fun r' j => dense Wv bv (X r') (finProdFinEquiv (h, j)))
            (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv (h, j))))
          (fun h => sdpa_back_K N d
            (fun r' j => dense Wq bq (X r') (finProdFinEquiv (h, j)))
            (fun r' j => dense Wk bk (X r') (finProdFinEquiv (h, j)))
            (fun r' j => dense Wv bv (X r') (finProdFinEquiv (h, j)))
            (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv (h, j))))
          (fun h => sdpa_back_V N d
            (fun r' j => dense Wq bq (X r') (finProdFinEquiv (h, j)))
            (fun r' j => dense Wk bk (X r') (finProdFinEquiv (h, j)))
            (fun r' j => dense Wv bv (X r') (finProdFinEquiv (h, j)))
            (fun r' j => Mat.mulVec Wo (dh r') (finProdFinEquiv (h, j)))) r c

/-- The proven MHSA VJP's backward at general `heads` IS the per-head collapse. -/
theorem mhsa_backward_collapseMH (N heads d : Nat)
    (Wq Wk Wv Wo : Mat (heads * d) (heads * d)) (bq bk bv bo : Vec (heads * d))
    (X dh : Mat N (heads * d)) :
    (mhsa_has_vjp_mat N heads d Wq Wk Wv Wo bq bk bv bo).backward X dh
      = mhsaBackCollapsedMH N heads d Wq Wk Wv Wo bq bk bv bo X dh := by
  rw [← mhsaClean_backward_eq, mhsaClean_backward_collapseMH]

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


-- ════════════════════════════════════════════════════════════════
-- § Stage 2 (MULTI-HEAD) — MHSA backward graph
-- ════════════════════════════════════════════════════════════════

/-- Contracting `W` against head `h`'s pad-scatter of an `[N,d]` cotangent reads
    off head `h`'s columns. -/
private lemma mulVec_headPadMat {N heads d : Nat} (W : Mat (heads * d) (heads * d))
    (h : Fin heads) (M : Mat N d) (r : Fin N) (c : Fin (heads * d)) :
    Mat.mulVec W (headPadMat N heads d h M r) c
      = ∑ j : Fin d, W c (finProdFinEquiv (h, j)) * M r j := by
  unfold Mat.mulVec headPadMat
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin heads × Fin d ≃ Fin (heads * d))
        (fun kj => W c kj *
          (if (finProdFinEquiv.symm kj).1 = h then M r (finProdFinEquiv.symm kj).2 else 0))]
  rw [Fintype.sum_prod_type]
  simp only [Equiv.symm_apply_apply]
  rw [Finset.sum_eq_single h]
  · apply Finset.sum_congr rfl; intro j _; rw [if_pos rfl]
  · intro h' _ hne
    apply Finset.sum_eq_zero; intro j _; rw [if_neg hne, mul_zero]
  · intro hc; exact absurd (Finset.mem_univ h) hc

/-- `denseRowBack W` of head `h`'s pad of a flattened `[N,d]` matrix `M` reads off
    head `h`'s columns of `W` per output column. -/
private lemma denseRowBack_headPad_eq {Np1 heads d : Nat} (wN : String)
    (W : Mat (heads * d) (heads * d)) (h : Fin heads) (M : Mat Np1 d)
    (e : SHlo (Np1 * d)) (he : den e = Mat.flatten M) :
    den (SHlo.denseRowBack wN W (SHlo.headPadF h e))
      = Mat.flatten (fun r c => ∑ jj : Fin d, W c (finProdFinEquiv (h, jj)) * M r jj) := by
  rw [denseRowBack_faithful, headPadF_faithful, he, headPadFlat_flat]
  unfold rowDenseBackFlat
  funext k
  simp only [Mat.unflatten_flatten]
  unfold Mat.flatten
  exact mulVec_headPadMat W h M _ _

/-- Per-head SDPA-back subgraph denotes head `h`'s `sdpa_back_Q` at `d`. Saved:
    `ss h` = head-`h` scaled scores, `K h`/`V h` = head-`h` K/V, cotangent =
    head-`h` slice of the `Wo`-back. -/
private lemma sdpaBackQGraph_head_eq {Np1 heads d : Nat}
    (Qg Kg Vg dAttg : Fin heads → Mat Np1 d) (h : Fin heads) (e : SHlo (Np1 * d))
    (he : den e = Mat.flatten (dAttg h)) :
    den (sdpaBackQGraph Np1 d
          (Mat.flatten (fun i j => sdpa_scale d *
            Mat.mul (Qg h) (Mat.transpose (Kg h)) i j))
          (Mat.flatten (Kg h)) (Mat.flatten (Vg h)) e)
      = Mat.flatten (sdpa_back_Q Np1 d (Qg h) (Kg h) (Vg h) (dAttg h)) := by
  rw [sdpaBackQGraph_faithful, he, vitCotDQ_eq_sdpa_back_Q]

private lemma sdpaBackKGraph_head_eq {Np1 heads d : Nat}
    (Qg Kg Vg dAttg : Fin heads → Mat Np1 d) (h : Fin heads) (e : SHlo (Np1 * d))
    (he : den e = Mat.flatten (dAttg h)) :
    den (sdpaBackKGraph Np1 d
          (Mat.flatten (fun i j => sdpa_scale d *
            Mat.mul (Qg h) (Mat.transpose (Kg h)) i j))
          (Mat.flatten (Qg h)) (Mat.flatten (Vg h)) e)
      = Mat.flatten (sdpa_back_K Np1 d (Qg h) (Kg h) (Vg h) (dAttg h)) := by
  rw [sdpaBackKGraph_faithful, he, vitCotDK_eq_sdpa_back_K]

private lemma sdpaBackVGraph_head_eq {Np1 heads d : Nat}
    (Qg Kg Vg dAttg : Fin heads → Mat Np1 d) (h : Fin heads) (e : SHlo (Np1 * d))
    (he : den e = Mat.flatten (dAttg h)) :
    den (sdpaBackVGraph Np1 d
          (Mat.flatten (sdpa_weights Np1 d (Qg h) (Kg h))) e)
      = Mat.flatten (sdpa_back_V Np1 d (Qg h) (Kg h) (Vg h) (dAttg h)) := by
  rw [sdpaBackVGraph_faithful, he, vitCotDV_eq_sdpa_back_V]

/-- The whole multi-head MHSA backward graph (`heads = hm1 + 1`): for each head `h`,
    the three-way LN₁-fan-in over the per-head SDPA backward subgraphs (fed the
    head-`h` slice of the `Wo`-back), padded into head `h`'s columns and contracted
    against `Wq/Wk/Wv`; summed over heads (`headsSumG`). Saved (per head): the dense
    Q/K/V projections, the scaled pre-softmax scores `ss`, the post-softmax weights
    `p`; plus the block cotangent `dh`. -/
noncomputable def mhsaBackGraphMH {Np1 hm1 d : Nat}
    (Wq Wk Wv Wo : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (Q K V : Fin (hm1 + 1) → Vec (Np1 * d))
    (ss p : Fin (hm1 + 1) → Vec (Np1 * Np1))
    (dh : Vec (Np1 * ((hm1 + 1) * d))) : SHlo (Np1 * ((hm1 + 1) * d)) :=
  let dAtt := SHlo.denseRowBack "%Wo" Wo (.operand "%dh" dh)
  headsSumG (fun h : Fin (hm1 + 1) =>
    SHlo.addV
      (SHlo.addV
        (SHlo.denseRowBack "%Wq" Wq (SHlo.headPadF h
          (sdpaBackQGraph Np1 d (ss h) (K h) (V h) (SHlo.headSliceF h dAtt))))
        (SHlo.denseRowBack "%Wk" Wk (SHlo.headPadF h
          (sdpaBackKGraph Np1 d (ss h) (Q h) (V h) (SHlo.headSliceF h dAtt)))))
      (SHlo.denseRowBack "%Wv" Wv (SHlo.headPadF h
        (sdpaBackVGraph Np1 d (p h) (SHlo.headSliceF h dAtt)))))

set_option maxHeartbeats 2000000 in
/-- **MHSA backward-graph faithfulness (multi-head, `heads = hm1 + 1`).** The
    per-head fan-in graph denotes the proven `mhsa_has_vjp_mat.backward` (flattened)
    at general heads, with the saved per-head dense projections, scaled scores, and
    post-softmax weights. -/
theorem mhsaBackGraphMH_faithful {Np1 hm1 d : Nat}
    (Wq Wk Wv Wo : Mat ((hm1 + 1) * d) ((hm1 + 1) * d))
    (bq bk bv bo : Vec ((hm1 + 1) * d))
    (X dh : Mat Np1 ((hm1 + 1) * d)) :
    den (mhsaBackGraphMH Wq Wk Wv Wo
          (fun h => Mat.flatten (fun r j => dense Wq bq (X r) (finProdFinEquiv (h, j))))
          (fun h => Mat.flatten (fun r j => dense Wk bk (X r) (finProdFinEquiv (h, j))))
          (fun h => Mat.flatten (fun r j => dense Wv bv (X r) (finProdFinEquiv (h, j))))
          (fun h => Mat.flatten (fun i j => sdpa_scale d *
            Mat.mul (fun r j' => dense Wq bq (X r) (finProdFinEquiv (h, j')))
              (Mat.transpose (fun r j' => dense Wk bk (X r) (finProdFinEquiv (h, j')))) i j))
          (fun h => Mat.flatten (sdpa_weights Np1 d
            (fun r j' => dense Wq bq (X r) (finProdFinEquiv (h, j')))
            (fun r j' => dense Wk bk (X r) (finProdFinEquiv (h, j')))))
          (Mat.flatten dh))
      = Mat.flatten ((mhsa_has_vjp_mat Np1 (hm1 + 1) d Wq Wk Wv Wo bq bk bv bo).backward X dh) := by
  rw [mhsa_backward_collapseMH]
  -- Name the per-head sliced activations.
  set Qg : Fin (hm1 + 1) → Mat Np1 d :=
    fun h r j => dense Wq bq (X r) (finProdFinEquiv (h, j)) with hQg
  set Kg : Fin (hm1 + 1) → Mat Np1 d :=
    fun h r j => dense Wk bk (X r) (finProdFinEquiv (h, j)) with hKg
  set Vg : Fin (hm1 + 1) → Mat Np1 d :=
    fun h r j => dense Wv bv (X r) (finProdFinEquiv (h, j)) with hVg
  set dAttg : Fin (hm1 + 1) → Mat Np1 d :=
    fun h r j => Mat.mulVec Wo (dh r) (finProdFinEquiv (h, j)) with hdAttg
  -- The head-`h` slice of the `Wo`-back cotangent.
  have hdAtt : ∀ h : Fin (hm1 + 1),
      den (SHlo.headSliceF h (SHlo.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh))))
        = Mat.flatten (dAttg h) := by
    intro h
    rw [headSliceF_faithful, denseRowBack_faithful, den_operand]
    unfold rowDenseBackFlat
    rw [show (fun i => Mat.mulVec Wo ((Mat.unflatten (Mat.flatten dh)) i))
          = (fun r => Mat.mulVec Wo (dh r)) from by rw [Mat.unflatten_flatten]]
    rw [headSliceFlat_flat]
    rfl
  -- Each per-head branch denotes the flatten of head `h`'s fan-in part.
  unfold mhsaBackGraphMH
  rw [den_headsSumG]
  funext j
  -- RHS: flatten of the per-head sum.
  show (∑ h : Fin (hm1 + 1),
          den (SHlo.addV
            (SHlo.addV
              (SHlo.denseRowBack "%Wq" Wq (SHlo.headPadF h
                (sdpaBackQGraph Np1 d _ (Mat.flatten (Kg h)) (Mat.flatten (Vg h))
                  (SHlo.headSliceF h _))))
              (SHlo.denseRowBack "%Wk" Wk (SHlo.headPadF h
                (sdpaBackKGraph Np1 d _ (Mat.flatten (Qg h)) (Mat.flatten (Vg h))
                  (SHlo.headSliceF h _)))))
            (SHlo.denseRowBack "%Wv" Wv (SHlo.headPadF h
              (sdpaBackVGraph Np1 d _ (SHlo.headSliceF h _))))) j) = _
  simp only [den_addV]
  -- Compute each branch via the per-head helpers.
  have hQbr : ∀ h : Fin (hm1 + 1),
      den (SHlo.denseRowBack "%Wq" Wq (SHlo.headPadF h
            (sdpaBackQGraph Np1 d
              (Mat.flatten (fun i jj => sdpa_scale d *
                Mat.mul (Qg h) (Mat.transpose (Kg h)) i jj))
              (Mat.flatten (Kg h)) (Mat.flatten (Vg h))
              (SHlo.headSliceF h (SHlo.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh)))))))
        = Mat.flatten (fun r c => ∑ jj : Fin d, Wq c (finProdFinEquiv (h, jj)) *
            sdpa_back_Q Np1 d (Qg h) (Kg h) (Vg h) (dAttg h) r jj) := by
    intro h
    rw [denseRowBack_headPad_eq "%Wq" Wq h
          (sdpa_back_Q Np1 d (Qg h) (Kg h) (Vg h) (dAttg h)) _
          (sdpaBackQGraph_head_eq Qg Kg Vg dAttg h _ (hdAtt h))]
  have hKbr : ∀ h : Fin (hm1 + 1),
      den (SHlo.denseRowBack "%Wk" Wk (SHlo.headPadF h
            (sdpaBackKGraph Np1 d
              (Mat.flatten (fun i jj => sdpa_scale d *
                Mat.mul (Qg h) (Mat.transpose (Kg h)) i jj))
              (Mat.flatten (Qg h)) (Mat.flatten (Vg h))
              (SHlo.headSliceF h (SHlo.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh)))))))
        = Mat.flatten (fun r c => ∑ jj : Fin d, Wk c (finProdFinEquiv (h, jj)) *
            sdpa_back_K Np1 d (Qg h) (Kg h) (Vg h) (dAttg h) r jj) := by
    intro h
    rw [denseRowBack_headPad_eq "%Wk" Wk h
          (sdpa_back_K Np1 d (Qg h) (Kg h) (Vg h) (dAttg h)) _
          (sdpaBackKGraph_head_eq Qg Kg Vg dAttg h _ (hdAtt h))]
  have hVbr : ∀ h : Fin (hm1 + 1),
      den (SHlo.denseRowBack "%Wv" Wv (SHlo.headPadF h
            (sdpaBackVGraph Np1 d (Mat.flatten (sdpa_weights Np1 d (Qg h) (Kg h)))
              (SHlo.headSliceF h (SHlo.denseRowBack "%Wo" Wo (.operand "%dh" (Mat.flatten dh)))))))
        = Mat.flatten (fun r c => ∑ jj : Fin d, Wv c (finProdFinEquiv (h, jj)) *
            sdpa_back_V Np1 d (Qg h) (Kg h) (Vg h) (dAttg h) r jj) := by
    intro h
    rw [denseRowBack_headPad_eq "%Wv" Wv h
          (sdpa_back_V Np1 d (Qg h) (Kg h) (Vg h) (dAttg h)) _
          (sdpaBackVGraph_head_eq Qg Kg Vg dAttg h _ (hdAtt h))]
  rw [show (∑ h : Fin (hm1 + 1),
        ((den (SHlo.denseRowBack "%Wq" Wq (SHlo.headPadF h
            (sdpaBackQGraph Np1 d _ (Mat.flatten (Kg h)) (Mat.flatten (Vg h))
              (SHlo.headSliceF h _)))) j
         + den (SHlo.denseRowBack "%Wk" Wk (SHlo.headPadF h
            (sdpaBackKGraph Np1 d _ (Mat.flatten (Qg h)) (Mat.flatten (Vg h))
              (SHlo.headSliceF h _)))) j)
         + den (SHlo.denseRowBack "%Wv" Wv (SHlo.headPadF h
            (sdpaBackVGraph Np1 d _ (SHlo.headSliceF h _)))) j))
      = ∑ h : Fin (hm1 + 1),
        ((Mat.flatten (fun r c => ∑ jj : Fin d, Wq c (finProdFinEquiv (h, jj)) *
            sdpa_back_Q Np1 d (Qg h) (Kg h) (Vg h) (dAttg h) r jj) j
          + Mat.flatten (fun r c => ∑ jj : Fin d, Wk c (finProdFinEquiv (h, jj)) *
            sdpa_back_K Np1 d (Qg h) (Kg h) (Vg h) (dAttg h) r jj) j)
          + Mat.flatten (fun r c => ∑ jj : Fin d, Wv c (finProdFinEquiv (h, jj)) *
            sdpa_back_V Np1 d (Qg h) (Kg h) (Vg h) (dAttg h) r jj) j)
      from by
        apply Finset.sum_congr rfl; intro h _; rw [hQbr h, hKbr h, hVbr h]]
  -- RHS: flatten of mhsaBackCollapsedMH = the per-head sum.
  unfold mhsaBackCollapsedMH
  unfold Mat.flatten
  rfl

-- ════════════════════════════════════════════════════════════════
-- § Stage 3 (MULTI-HEAD) — attn sublayer + whole block
-- ════════════════════════════════════════════════════════════════

/-- Attn-sublayer non-trivial arm (`mhsa ∘ LN₁`), multi-head. -/
noncomputable def attnSublayerInnerBackGraphMH {Np1 hm1 d : Nat} (ε γ1 : ℝ)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x : Vec (Np1 * ((hm1+1) * d))) (X : Mat Np1 ((hm1+1) * d))
    (e : SHlo (Np1 * ((hm1+1) * d))) : SHlo (Np1 * ((hm1+1) * d)) :=
  SHlo.lnRowBack "%g1" "%x" "ε" ε γ1 x
    (mhsaBackGraphMH Wq Wk Wv Wo
      (fun h => Mat.flatten (fun r j => dense Wq bq (X r) (finProdFinEquiv (h, j))))
      (fun h => Mat.flatten (fun r j => dense Wk bk (X r) (finProdFinEquiv (h, j))))
      (fun h => Mat.flatten (fun r j => dense Wv bv (X r) (finProdFinEquiv (h, j))))
      (fun h => Mat.flatten (fun i j => sdpa_scale d *
        Mat.mul (fun r j' => dense Wq bq (X r) (finProdFinEquiv (h, j')))
          (Mat.transpose (fun r j' => dense Wk bk (X r) (finProdFinEquiv (h, j')))) i j))
      (fun h => Mat.flatten (sdpa_weights Np1 d
        (fun r j' => dense Wq bq (X r) (finProdFinEquiv (h, j')))
        (fun r j' => dense Wk bk (X r) (finProdFinEquiv (h, j')))))
      (den e))

theorem attnSublayerInnerBackGraphMH_faithful {Np1 hm1 d : Nat} (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x : Mat Np1 ((hm1+1) * d)) (dh : Mat Np1 ((hm1+1) * d)) :
    den (attnSublayerInnerBackGraphMH ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten x)
          (fun r => layerNormForward ((hm1+1) * d) ε γ1 β1 (x r)) (.operand "%dh" (Mat.flatten dh)))
      = Mat.flatten
          ((layerNorm_per_token_has_vjp_mat Np1 ((hm1+1) * d) ε γ1 β1 hε).backward x
            ((mhsa_has_vjp_mat Np1 (hm1+1) d Wq Wk Wv Wo bq bk bv bo).backward
              (fun r => layerNormForward ((hm1+1) * d) ε γ1 β1 (x r)) dh)) := by
  simp only [attnSublayerInnerBackGraphMH, lnRowBack_faithful, den_operand]
  rw [mhsaBackGraphMH_faithful Wq Wk Wv Wo bq bk bv bo
        (fun r => layerNormForward ((hm1+1) * d) ε γ1 β1 (x r)) dh]
  rw [rowLNBackFlat_eq_backward (β := β1) ε γ1 hε x
        ((mhsa_has_vjp_mat Np1 (hm1+1) d Wq Wk Wv Wo bq bk bv bo).backward
          (fun r => layerNormForward ((hm1+1) * d) ε γ1 β1 (x r)) dh)]

noncomputable def attnSublayerBackGraphMH {Np1 hm1 d : Nat} (ε γ1 : ℝ)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x : Vec (Np1 * ((hm1+1) * d))) (X : Mat Np1 ((hm1+1) * d))
    (dh : Vec (Np1 * ((hm1+1) * d))) : SHlo (Np1 * ((hm1+1) * d)) :=
  SHlo.addV (attnSublayerInnerBackGraphMH ε γ1 Wq Wk Wv Wo bq bk bv bo x X
          (.operand "%dh" dh))
        (.operand "%dh" dh)

set_option maxHeartbeats 1000000 in
private theorem attnSublayer_backward_unfoldMH {Np1 hm1 d : Nat} (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x dh : Mat Np1 ((hm1+1) * d)) :
    (transformerAttnSublayer_has_vjp_mat Np1 (hm1+1) d ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo).backward x dh
      = fun i k => dh i k +
          (layerNorm_per_token_has_vjp_mat Np1 ((hm1+1) * d) ε γ1 β1 hε).backward x
            ((mhsa_has_vjp_mat Np1 (hm1+1) d Wq Wk Wv Wo bq bk bv bo).backward
              (fun r => layerNormForward ((hm1+1) * d) ε γ1 β1 (x r)) dh) i k := rfl

theorem attnSublayerBackGraphMH_faithful {Np1 hm1 d : Nat} (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x : Mat Np1 ((hm1+1) * d)) (dh : Mat Np1 ((hm1+1) * d)) :
    den (attnSublayerBackGraphMH ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten x)
          (fun r => layerNormForward ((hm1+1) * d) ε γ1 β1 (x r)) (Mat.flatten dh))
      = Mat.flatten ((transformerAttnSublayer_has_vjp_mat Np1 (hm1+1) d ε γ1 β1 hε
          Wq Wk Wv Wo bq bk bv bo).backward x dh) := by
  rw [attnSublayer_backward_unfoldMH (bo := bo)]
  funext j
  show den (attnSublayerInnerBackGraphMH ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten x)
            (fun r => layerNormForward ((hm1+1) * d) ε γ1 β1 (x r))
            (.operand "%dh" (Mat.flatten dh))) j + Mat.flatten dh j = _
  rw [attnSublayerInnerBackGraphMH_faithful ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo x dh]
  unfold Mat.flatten
  exact add_comm _ _

noncomputable def transformerBlockBackGraphMH {Np1 hm1 d mlpDim : Nat}
    (ε γ1 β1 γ2 β2 : ℝ)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (Wfc1 : Mat ((hm1+1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1+1) * d))
    (A : Mat Np1 ((hm1+1) * d)) (h : Mat Np1 ((hm1+1) * d)) (dY : Vec (Np1 * ((hm1+1) * d))) :
    SHlo (Np1 * ((hm1+1) * d)) :=
  attnSublayerBackGraphMH ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten A)
    (fun r => layerNormForward ((hm1+1) * d) ε γ1 β1 (A r))
    (den (mlpSublayerBackGraph ε γ2 Wfc1 bfc1 Wfc2 (Mat.flatten h)
            (fun r => layerNormForward ((hm1+1) * d) ε γ2 β2 (h r)) dY))

set_option maxHeartbeats 1000000 in
private theorem transformerBlock_backward_unfoldMH {Np1 hm1 d mlpDim : Nat}
    (ε γ1 β1 γ2 β2 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (Wfc1 : Mat ((hm1+1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1+1) * d)) (bfc2 : Vec ((hm1+1) * d))
    (A dY : Mat Np1 ((hm1+1) * d)) :
    (transformerBlock_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward A dY
      = (transformerAttnSublayer_has_vjp_mat Np1 (hm1+1) d ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo).backward A
          ((transformerMlpSublayer_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ2 β2 hε
              Wfc1 bfc1 Wfc2 bfc2).backward
            (transformerAttnSublayer Np1 (hm1+1) d ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A) dY) := rfl


-- ── MLP sublayer faithfulness at general (hm1+1)*d (used by the block capstone) ──
theorem mlpSublayerBackGraph_faithfulMH {Np1 hm1 d mlpDim : Nat}
    (ε γ2 β2 : ℝ) (hε : 0 < ε)
    (Wfc1 : Mat ((hm1+1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1+1) * d)) (bfc2 : Vec ((hm1+1) * d))
    (h : Mat Np1 ((hm1+1) * d)) (dz : Mat Np1 ((hm1+1) * d)) :
    den (mlpSublayerBackGraph ε γ2 Wfc1 bfc1 Wfc2 (Mat.flatten h)
          (fun r => layerNormForward ((hm1+1) * d) ε γ2 β2 (h r)) (Mat.flatten dz))
      = Mat.flatten ((transformerMlpSublayer_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ2 β2 hε
          Wfc1 bfc1 Wfc2 bfc2).backward h dz) := by
  funext j
  show den (mlpSublayerInnerBackGraph ε γ2 Wfc1 bfc1 Wfc2 (Mat.flatten h)
            (fun r => layerNormForward ((hm1+1) * d) ε γ2 β2 (h r))
            (.operand "%dz" (Mat.flatten dz))) j + Mat.flatten dz j = _
  rw [mlpSublayerInnerBackGraph_faithful ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2 h dz]
  show _ = Mat.flatten ((transformerMlpSublayer_has_vjp_mat Np1 (hm1+1) d mlpDim
              ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2).backward h dz) j
  show Mat.flatten ((layerNorm_per_token_has_vjp_mat Np1 ((hm1+1) * d) ε γ2 β2 hε).backward h
          ((transformerMlp_has_vjp_mat Np1 ((hm1+1) * d) mlpDim Wfc1 bfc1 Wfc2 bfc2).backward
            (fun r => layerNormForward ((hm1+1) * d) ε γ2 β2 (h r)) dz)) j
        + Mat.flatten dz j = _
  unfold Mat.flatten
  exact add_comm _ _

theorem transformerBlockBackGraphMH_faithful {Np1 hm1 d mlpDim : Nat}
    (ε γ1 β1 γ2 β2 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (Wfc1 : Mat ((hm1+1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1+1) * d)) (bfc2 : Vec ((hm1+1) * d))
    (A dY : Mat Np1 ((hm1+1) * d)) :
    den (transformerBlockBackGraphMH ε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
          Wfc1 bfc1 Wfc2 A
          (transformerAttnSublayer Np1 (hm1+1) d ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A)
          (Mat.flatten dY))
      = Mat.flatten ((transformerBlock_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ1 β1 hε
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward A dY) := by
  rw [transformerBlock_backward_unfoldMH (bfc2 := bfc2)]
  set h : Mat Np1 ((hm1+1) * d) :=
    transformerAttnSublayer Np1 (hm1+1) d ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A with hh
  show den (attnSublayerBackGraphMH ε γ1 Wq Wk Wv Wo bq bk bv bo (Mat.flatten A)
            (fun r => layerNormForward ((hm1+1) * d) ε γ1 β1 (A r))
            (den (mlpSublayerBackGraph ε γ2 Wfc1 bfc1 Wfc2 (Mat.flatten h)
                    (fun r => layerNormForward ((hm1+1) * d) ε γ2 β2 (h r)) (Mat.flatten dY)))) = _
  rw [mlpSublayerBackGraph_faithfulMH ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2 h dY]
  rw [attnSublayerBackGraphMH_faithful ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo A
        ((transformerMlpSublayer_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ2 β2 hε
            Wfc1 bfc1 Wfc2 bfc2).backward h dY)]


-- ════════════════════════════════════════════════════════════════
-- § VECTOR-LN VARIANT — production-parity capstone (multi-head + vec-LN)
-- ════════════════════════════════════════════════════════════════

/-! The committed `verified_mlir/vit_train_step.mlir` ViT-Tiny render uses VECTOR
γ/β per LN site, decomposed as `(+βv) ∘ layerScale γv ∘ LN(1,0)` (`layerNormVec`,
ViTVecLN.lean). The forward/backward MHSA and MLP-body are IDENTICAL to the scalar
block; the only difference is the two LN sites. So we REUSE `mhsaBackGraphMH` and
`transformerMlpBackGraph` verbatim and swap the LN-back fragment to the vec-LN one:
apply `rowScaleF γv` to the incoming cotangent, then `lnRowBack` at γ=1 (the bias
backward is the identity, so β drops out of the input cotangent).

The new structural facts:
  * `layerNormVec_per_token_backward_eq` — the vec-LN per-token backward IS the
    normalize-only (LN at γ=1,β=0) backward of the rowwise-`layerScale γv` cotangent
    (the bias backward = id collapses).
  * `rowVecLNBack_eq_backward` — the flat composition `lnRowBack(γ=1) ∘ rowScaleF γv`
    denotes that vec-LN per-token backward (the crux bridge).
Then the sublayer + whole-block capstones mirror the scalar/MH templates exactly,
re-targeting the `_has_vjp_mat` to the `…V…` (vec-LN) versions. -/

-- ── Stage 1: the vec-LN LN-back bridge ──

/-- **The vec-LN per-token backward collapses to normalize-only-of-scaled.** The
    vec-LN VJP `(+βv) ∘ layerScale γv ∘ LN(1,0)` has, by `vjp_comp`, backward
    `LN(1,0).backward x (layerScale_has_vjp.backward _ (biasAdd.backward _ dy))`;
    `biasAdd.backward = id`, `layerScale_has_vjp.backward _ dy = (γv · * dy ·) =
    layerScale γv dy`. Rowwise-lifted, this is the normalize-only (`layerNorm` at
    γ=1, β=0) per-token backward fed the rowwise `layerScale γv` of the cotangent. -/
theorem layerNormVec_per_token_backward_eq {N D : Nat} (ε : ℝ) (γv βv : Vec D)
    (hε : 0 < ε) (X dY : Mat N D) :
    (layerNormVec_per_token_has_vjp_mat N D ε γv βv hε).backward X dY
      = (layerNorm_per_token_has_vjp_mat N D ε 1 0 hε).backward X
          (fun r => layerScale γv (dY r)) := by
  funext r c
  -- Both sides are the rowwise lift; reduce to the per-row `HasVJP.backward`.
  show (layerNormVec_has_vjp D ε γv βv hε).backward (X r) (dY r) c
        = (layerNorm_has_vjp D ε 1 0 hε).backward (X r) (layerScale γv (dY r)) c
  -- `layerNormVec_has_vjp = vjp_comp _ (+βv) … (vjp_comp LN (layerScale γv) …) (biasAdd βv)`,
  -- so `.backward x dy = LN.backward x (layerScale_has_vjp.backward _ (biasAdd.backward _ dy))`.
  -- `biasAdd.backward = id`; `layerScale_has_vjp.backward _ dy = fun k => γv k * dy k`, which is
  -- definitionally `layerScale γv dy`. All collapses by `rfl`.
  rfl

/-- **Vec-LN LN-back bridge (Stage 1 crux).** The flat composition `lnRowBack(γ=1)
    of (rowScaleF γv applied to the cotangent)` denotes the vec-LN per-token VJP's
    `.backward` (flattened) at the saved pre-LN input `X`. The `rowScaleF` realizes
    the rowwise `layerScale γv` on the incoming cotangent; the `lnRowBack` at γ=1 is
    the normalize-only backward; the bias backward (identity) has dropped out. -/
theorem rowVecLNBack_eq_backward {N D : Nat} (ε : ℝ) (γv βv : Vec D) (hε : 0 < ε)
    (X dY : Mat N D) :
    rowLNBackFlat N D ε 1 (Mat.flatten X) (rowScaleFlat N D γv (Mat.flatten dY))
      = Mat.flatten ((layerNormVec_per_token_has_vjp_mat N D ε γv βv hε).backward X dY) := by
  rw [layerNormVec_per_token_backward_eq ε γv βv hε X dY]
  rw [rowScaleFlat_flat γv dY]
  rw [rowLNBackFlat_eq_backward (β := (0 : ℝ)) ε 1 hε X (fun r => layerScale γv (dY r))]

-- ── Stage 2: Vec-LN MLP sublayer backward graph ──

/-- The vec-LN MLP-sublayer non-trivial arm backward graph (`transformerMlp ∘ LNᵥ₂`;
    outermost backward token = earliest forward op = LN₂). REUSES `transformerMlpBackGraph`
    verbatim (the MLP body is LN-agnostic); the only change vs the scalar
    `mlpSublayerInnerBackGraph` is the LN-back fragment: `lnRowBack(γ=1) ∘ rowScaleF γ2v`
    instead of `lnRowBack(γ2)`. `Y = LNᵥ₂ h` is the saved MLP input. -/
noncomputable def mlpSublayerVInnerBackGraph {Np1 D mlpDim : Nat}
    (ε : ℝ) (γ2v : Vec D)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim D)
    (h : Vec (Np1 * D)) (Y : Mat Np1 D) (e : SHlo (Np1 * D)) : SHlo (Np1 * D) :=
  .lnRowBack "%g2" "%h" "ε" ε 1 h
    (.rowScaleF "%g2v" γ2v (transformerMlpBackGraph Wfc1 bfc1 Wfc2 Y e))

/-- **Vec-LN MLP-sublayer inner-arm backward-graph faithfulness.** Denotes the proven
    `(vjpMat_comp LNᵥ₂ transformerMlp).backward h ·`. `Y = LNᵥ₂ h`. -/
theorem mlpSublayerVInnerBackGraph_faithful {Np1 D mlpDim : Nat}
    (ε : ℝ) (γ2v β2v : Vec D) (hε : 0 < ε)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (h : Mat Np1 D) (dz : Mat Np1 D) :
    den (mlpSublayerVInnerBackGraph ε γ2v Wfc1 bfc1 Wfc2 (Mat.flatten h)
          (fun r => layerNormVec D ε γ2v β2v (h r)) (.operand "%dz" (Mat.flatten dz)))
      = Mat.flatten
          ((layerNormVec_per_token_has_vjp_mat Np1 D ε γ2v β2v hε).backward h
            ((transformerMlp_has_vjp_mat Np1 D mlpDim Wfc1 bfc1 Wfc2 bfc2).backward
              (fun r => layerNormVec D ε γ2v β2v (h r)) dz)) := by
  simp only [mlpSublayerVInnerBackGraph, lnRowBack_faithful, rowScaleF_faithful]
  rw [transformerMlpBackGraph_faithful Wfc1 bfc1 Wfc2 bfc2
        (fun r => layerNormVec D ε γ2v β2v (h r)) dz]
  rw [rowVecLNBack_eq_backward (βv := β2v) ε γ2v hε h
        ((transformerMlp_has_vjp_mat Np1 D mlpDim Wfc1 bfc1 Wfc2 bfc2).backward
          (fun r => layerNormVec D ε γ2v β2v (h r)) dz)]

/-- The whole vec-LN MLP-sublayer backward graph (inner arm + identity skip). -/
noncomputable def mlpSublayerVBackGraph {Np1 D mlpDim : Nat}
    (ε : ℝ) (γ2v : Vec D)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim) (Wfc2 : Mat mlpDim D)
    (h : Vec (Np1 * D)) (Y : Mat Np1 D) (dz : Vec (Np1 * D)) : SHlo (Np1 * D) :=
  .addV (mlpSublayerVInnerBackGraph ε γ2v Wfc1 bfc1 Wfc2 h Y (.operand "%dz" dz))
        (.operand "%dz" dz)

/-- **Vec-LN MLP sublayer backward-graph faithfulness (Stage 2 capstone), at general
    `(hm1+1)*d`.** Denotes the proven `transformerMlpSublayerV_has_vjp_mat` backward. -/
theorem mlpSublayerVBackGraph_faithfulMH {Np1 hm1 d mlpDim : Nat}
    (ε : ℝ) (γ2v β2v : Vec ((hm1+1) * d)) (hε : 0 < ε)
    (Wfc1 : Mat ((hm1+1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1+1) * d)) (bfc2 : Vec ((hm1+1) * d))
    (h : Mat Np1 ((hm1+1) * d)) (dz : Mat Np1 ((hm1+1) * d)) :
    den (mlpSublayerVBackGraph ε γ2v Wfc1 bfc1 Wfc2 (Mat.flatten h)
          (fun r => layerNormVec ((hm1+1) * d) ε γ2v β2v (h r)) (Mat.flatten dz))
      = Mat.flatten ((transformerMlpSublayerV_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ2v β2v hε
          Wfc1 bfc1 Wfc2 bfc2).backward h dz) := by
  funext j
  show den (mlpSublayerVInnerBackGraph ε γ2v Wfc1 bfc1 Wfc2 (Mat.flatten h)
            (fun r => layerNormVec ((hm1+1) * d) ε γ2v β2v (h r))
            (.operand "%dz" (Mat.flatten dz))) j + Mat.flatten dz j = _
  rw [mlpSublayerVInnerBackGraph_faithful ε γ2v β2v hε Wfc1 bfc1 Wfc2 bfc2 h dz]
  show _ = Mat.flatten ((transformerMlpSublayerV_has_vjp_mat Np1 (hm1+1) d mlpDim
              ε γ2v β2v hε Wfc1 bfc1 Wfc2 bfc2).backward h dz) j
  show Mat.flatten ((layerNormVec_per_token_has_vjp_mat Np1 ((hm1+1) * d) ε γ2v β2v hε).backward h
          ((transformerMlp_has_vjp_mat Np1 ((hm1+1) * d) mlpDim Wfc1 bfc1 Wfc2 bfc2).backward
            (fun r => layerNormVec ((hm1+1) * d) ε γ2v β2v (h r)) dz)) j
        + Mat.flatten dz j = _
  unfold Mat.flatten
  exact add_comm _ _

-- ── Stage 3: Vec-LN attention sublayer backward graph (multi-head) ──

/-- The vec-LN attn-sublayer non-trivial arm (`mhsa ∘ LNᵥ₁`), multi-head. REUSES
    `mhsaBackGraphMH` verbatim; the only change vs the scalar `attnSublayerInnerBackGraphMH`
    is the LN-back fragment: `lnRowBack(γ=1) ∘ rowScaleF γ1v`. -/
noncomputable def attnSublayerVInnerBackGraphMH {Np1 hm1 d : Nat} (ε : ℝ)
    (γ1v : Vec ((hm1+1) * d))
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x : Vec (Np1 * ((hm1+1) * d))) (X : Mat Np1 ((hm1+1) * d))
    (e : SHlo (Np1 * ((hm1+1) * d))) : SHlo (Np1 * ((hm1+1) * d)) :=
  SHlo.lnRowBack "%g1" "%x" "ε" ε 1 x
    (SHlo.rowScaleF "%g1v" γ1v
      (mhsaBackGraphMH Wq Wk Wv Wo
        (fun h => Mat.flatten (fun r j => dense Wq bq (X r) (finProdFinEquiv (h, j))))
        (fun h => Mat.flatten (fun r j => dense Wk bk (X r) (finProdFinEquiv (h, j))))
        (fun h => Mat.flatten (fun r j => dense Wv bv (X r) (finProdFinEquiv (h, j))))
        (fun h => Mat.flatten (fun i j => sdpa_scale d *
          Mat.mul (fun r j' => dense Wq bq (X r) (finProdFinEquiv (h, j')))
            (Mat.transpose (fun r j' => dense Wk bk (X r) (finProdFinEquiv (h, j')))) i j))
        (fun h => Mat.flatten (sdpa_weights Np1 d
          (fun r j' => dense Wq bq (X r) (finProdFinEquiv (h, j')))
          (fun r j' => dense Wk bk (X r) (finProdFinEquiv (h, j')))))
        (den e)))

theorem attnSublayerVInnerBackGraphMH_faithful {Np1 hm1 d : Nat} (ε : ℝ)
    (γ1v β1v : Vec ((hm1+1) * d)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x : Mat Np1 ((hm1+1) * d)) (dh : Mat Np1 ((hm1+1) * d)) :
    den (attnSublayerVInnerBackGraphMH ε γ1v Wq Wk Wv Wo bq bk bv bo (Mat.flatten x)
          (fun r => layerNormVec ((hm1+1) * d) ε γ1v β1v (x r)) (.operand "%dh" (Mat.flatten dh)))
      = Mat.flatten
          ((layerNormVec_per_token_has_vjp_mat Np1 ((hm1+1) * d) ε γ1v β1v hε).backward x
            ((mhsa_has_vjp_mat Np1 (hm1+1) d Wq Wk Wv Wo bq bk bv bo).backward
              (fun r => layerNormVec ((hm1+1) * d) ε γ1v β1v (x r)) dh)) := by
  simp only [attnSublayerVInnerBackGraphMH, lnRowBack_faithful, rowScaleF_faithful, den_operand]
  rw [mhsaBackGraphMH_faithful Wq Wk Wv Wo bq bk bv bo
        (fun r => layerNormVec ((hm1+1) * d) ε γ1v β1v (x r)) dh]
  rw [rowVecLNBack_eq_backward (βv := β1v) ε γ1v hε x
        ((mhsa_has_vjp_mat Np1 (hm1+1) d Wq Wk Wv Wo bq bk bv bo).backward
          (fun r => layerNormVec ((hm1+1) * d) ε γ1v β1v (x r)) dh)]

noncomputable def attnSublayerVBackGraphMH {Np1 hm1 d : Nat} (ε : ℝ)
    (γ1v : Vec ((hm1+1) * d))
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x : Vec (Np1 * ((hm1+1) * d))) (X : Mat Np1 ((hm1+1) * d))
    (dh : Vec (Np1 * ((hm1+1) * d))) : SHlo (Np1 * ((hm1+1) * d)) :=
  SHlo.addV (attnSublayerVInnerBackGraphMH ε γ1v Wq Wk Wv Wo bq bk bv bo x X
          (.operand "%dh" dh))
        (.operand "%dh" dh)

set_option maxHeartbeats 1000000 in
private theorem attnSublayerV_backward_unfoldMH {Np1 hm1 d : Nat} (ε : ℝ)
    (γ1v β1v : Vec ((hm1+1) * d)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x dh : Mat Np1 ((hm1+1) * d)) :
    (transformerAttnSublayerV_has_vjp_mat Np1 (hm1+1) d ε γ1v β1v hε Wq Wk Wv Wo bq bk bv bo).backward x dh
      = fun i k => dh i k +
          (layerNormVec_per_token_has_vjp_mat Np1 ((hm1+1) * d) ε γ1v β1v hε).backward x
            ((mhsa_has_vjp_mat Np1 (hm1+1) d Wq Wk Wv Wo bq bk bv bo).backward
              (fun r => layerNormVec ((hm1+1) * d) ε γ1v β1v (x r)) dh) i k := rfl

/-- **Vec-LN attention sublayer backward-graph faithfulness (Stage 3 capstone, MH).** -/
theorem attnSublayerVBackGraphMH_faithful {Np1 hm1 d : Nat} (ε : ℝ)
    (γ1v β1v : Vec ((hm1+1) * d)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (x : Mat Np1 ((hm1+1) * d)) (dh : Mat Np1 ((hm1+1) * d)) :
    den (attnSublayerVBackGraphMH ε γ1v Wq Wk Wv Wo bq bk bv bo (Mat.flatten x)
          (fun r => layerNormVec ((hm1+1) * d) ε γ1v β1v (x r)) (Mat.flatten dh))
      = Mat.flatten ((transformerAttnSublayerV_has_vjp_mat Np1 (hm1+1) d ε γ1v β1v hε
          Wq Wk Wv Wo bq bk bv bo).backward x dh) := by
  rw [attnSublayerV_backward_unfoldMH (bo := bo)]
  funext j
  show den (attnSublayerVInnerBackGraphMH ε γ1v Wq Wk Wv Wo bq bk bv bo (Mat.flatten x)
            (fun r => layerNormVec ((hm1+1) * d) ε γ1v β1v (x r))
            (.operand "%dh" (Mat.flatten dh))) j + Mat.flatten dh j = _
  rw [attnSublayerVInnerBackGraphMH_faithful ε γ1v β1v hε Wq Wk Wv Wo bq bk bv bo x dh]
  unfold Mat.flatten
  exact add_comm _ _

-- ── Stage 4: Whole vec-LN transformer-block backward graph (multi-head) ──

/-- The whole vec-LN transformer-block backward graph (multi-head). `transformerBlockV =
    mlpSublayerV ∘ attnSublayerV`, so `block.backward A dY = attn.backward A
    (mlp.backward (attn A) dY)`. Saved: `A` (block input), `h = attnSublayerV A`. -/
noncomputable def transformerBlockVBackGraphMH {Np1 hm1 d mlpDim : Nat}
    (ε : ℝ) (γ1v β1v γ2v β2v : Vec ((hm1+1) * d))
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (Wfc1 : Mat ((hm1+1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1+1) * d))
    (A : Mat Np1 ((hm1+1) * d)) (h : Mat Np1 ((hm1+1) * d)) (dY : Vec (Np1 * ((hm1+1) * d))) :
    SHlo (Np1 * ((hm1+1) * d)) :=
  attnSublayerVBackGraphMH ε γ1v Wq Wk Wv Wo bq bk bv bo (Mat.flatten A)
    (fun r => layerNormVec ((hm1+1) * d) ε γ1v β1v (A r))
    (den (mlpSublayerVBackGraph ε γ2v Wfc1 bfc1 Wfc2 (Mat.flatten h)
            (fun r => layerNormVec ((hm1+1) * d) ε γ2v β2v (h r)) dY))

set_option maxHeartbeats 1000000 in
private theorem transformerBlockV_backward_unfoldMH {Np1 hm1 d mlpDim : Nat}
    (ε : ℝ) (γ1v β1v γ2v β2v : Vec ((hm1+1) * d)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (Wfc1 : Mat ((hm1+1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1+1) * d)) (bfc2 : Vec ((hm1+1) * d))
    (A dY : Mat Np1 ((hm1+1) * d)) :
    (transformerBlockV_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ1v β1v hε Wq Wk Wv Wo bq bk bv bo
        γ2v β2v Wfc1 bfc1 Wfc2 bfc2).backward A dY
      = (transformerAttnSublayerV_has_vjp_mat Np1 (hm1+1) d ε γ1v β1v hε Wq Wk Wv Wo bq bk bv bo).backward A
          ((transformerMlpSublayerV_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ2v β2v hε
              Wfc1 bfc1 Wfc2 bfc2).backward
            (transformerAttnSublayerV Np1 (hm1+1) d ε γ1v β1v Wq Wk Wv Wo bq bk bv bo A) dY) := rfl

/-- **Whole vec-LN transformer-block backward-graph faithfulness (Stage 4 capstone, MH).**
    The production-parity capstone — multi-head (`heads = hm1+1`) + vector-LN, matching
    the committed `verified_mlir/vit_train_step.mlir` ViT-Tiny config. Wires the vec-LN
    MLP-sublayer backward (at the saved attn-sublayer output `h = attnSublayerV A`) into
    the vec-LN attention-sublayer backward (at the saved block input `A`), per
    `block.backward A dY = attn.backward A (mlp.backward (attn A) dY)`. -/
theorem transformerBlockVBackGraphMH_faithful {Np1 hm1 d mlpDim : Nat}
    (ε : ℝ) (γ1v β1v γ2v β2v : Vec ((hm1+1) * d)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat ((hm1+1) * d) ((hm1+1) * d)) (bq bk bv bo : Vec ((hm1+1) * d))
    (Wfc1 : Mat ((hm1+1) * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim ((hm1+1) * d)) (bfc2 : Vec ((hm1+1) * d))
    (A dY : Mat Np1 ((hm1+1) * d)) :
    den (transformerBlockVBackGraphMH ε γ1v β1v γ2v β2v Wq Wk Wv Wo bq bk bv bo
          Wfc1 bfc1 Wfc2 A
          (transformerAttnSublayerV Np1 (hm1+1) d ε γ1v β1v Wq Wk Wv Wo bq bk bv bo A)
          (Mat.flatten dY))
      = Mat.flatten ((transformerBlockV_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ1v β1v hε
          Wq Wk Wv Wo bq bk bv bo γ2v β2v Wfc1 bfc1 Wfc2 bfc2).backward A dY) := by
  rw [transformerBlockV_backward_unfoldMH (bfc2 := bfc2)]
  set h : Mat Np1 ((hm1+1) * d) :=
    transformerAttnSublayerV Np1 (hm1+1) d ε γ1v β1v Wq Wk Wv Wo bq bk bv bo A with hh
  show den (attnSublayerVBackGraphMH ε γ1v Wq Wk Wv Wo bq bk bv bo (Mat.flatten A)
            (fun r => layerNormVec ((hm1+1) * d) ε γ1v β1v (A r))
            (den (mlpSublayerVBackGraph ε γ2v Wfc1 bfc1 Wfc2 (Mat.flatten h)
                    (fun r => layerNormVec ((hm1+1) * d) ε γ2v β2v (h r)) (Mat.flatten dY)))) = _
  rw [mlpSublayerVBackGraph_faithfulMH ε γ2v β2v hε Wfc1 bfc1 Wfc2 bfc2 h dY]
  rw [attnSublayerVBackGraphMH_faithful ε γ1v β1v hε Wq Wk Wv Wo bq bk bv bo A
        ((transformerMlpSublayerV_has_vjp_mat Np1 (hm1+1) d mlpDim ε γ2v β2v hε
            Wfc1 bfc1 Wfc2 bfc2).backward h dY)]

-- ════════════════════════════════════════════════════════════════
-- § WHOLE-NET backward graph — depth-`k`, multi-head, vector-LN
-- ════════════════════════════════════════════════════════════════

/-! The whole-net `vitForwardKV_has_vjp` (ViTDepthK.lean) is a `Vec → Vec`
`HasVJP` built by three `vjp_comp` steps:

    classifier_flat ∘ [finalLN] ∘ vitBodyKVFlat(k) ∘ patchEmbed_flat

so its `.backward x dy` chains the four stages in REVERSE (head first, image last):

    PE.back x (BODY.back (PE x) (LNF.back (BODY (PE x)) (classifier.back (… ) dy)))

We mirror this with a backward graph over the ch10 backward tokens. The blocks
live in `HasVJPMat` and are bridged to `HasVJP` by `hasVJPMat_to_hasVJP`, so each
block-back faithfulness statement is stated through `Mat.flatten`/`Mat.unflatten`
(exactly the `transformerBlockVBackGraphMH_faithful` convention). Four stages:

  * **Stage 1** — `classifierBackGraph` ↔ `classifier_flat_has_vjp.backward`:
    `clsPadF (dotOut Wcls (%dy))` (dense-back into row 0 of a zero `[N+1,D]`).
  * **Stage 2** — `finalLNBackGraph` ↔ the bridged per-token vec-LN VJP back:
    the vec-LN fragment `lnRowBack(γ=1) ∘ rowScaleF γF`, chained onto Stage 1.
  * **Stage 3** — `vitBodyBackGraphKMHV`, a depth-`k` reverse fold of
    `transformerBlockVBackGraphMH` (last forward block = first backward block),
    threading each block's saved forward activation; `_den` by induction on `k`.
  * **Stage 4** — `patchEmbedBackGraph` ↔ `patchEmbed_flat_has_vjp.backward`:
    the strided-patchify conv input-VJP (the `patchEmbedBack` token).
-/

-- ── Stage 1: Classifier backward graph ──

/-- The classifier-head backward graph: `clsPadF (dotOut Wcls (%dy))`. The
    `dotOut Wcls` is the dense head's input-VJP (`Mat.mulVec Wcls`), scattered by
    `clsPadF` into row 0 of a zero `[N+1,D]` (the CLS-slice's input-VJP). -/
noncomputable def classifierBackGraph (N D nClasses : Nat)
    (Wcls : Mat D nClasses) (dy : Vec nClasses) : SHlo ((N+1)*D) :=
  .clsPadF (N := N) (.dotOut "%Wcls" Wcls (.operand "%dy" dy))

/-- **Classifier backward-graph faithfulness (Stage 1).** Denotes the proven
    `classifier_flat_has_vjp.backward` at any input `v` (dense + CLS-slice are
    both linear, so the saved activation is irrelevant). -/
theorem classifierBackGraph_faithful (N D nClasses : Nat)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses)
    (v : Vec ((N + 1) * D)) (dy : Vec nClasses) :
    den (classifierBackGraph N D nClasses Wcls dy)
      = (classifier_flat_has_vjp N D nClasses Wcls bcls).backward v dy := by
  funext idx
  -- LHS: clsPadFlat N D (Mat.mulVec Wcls dy)
  show clsPadFlat N D (fun i => ∑ j, Wcls i j * dy j) idx = _
  -- RHS: classifier_flat_has_vjp.backward = cls_slice.back v (dense.back _ dy)
  --    = cls_slice_flat_has_vjp.backward v (Mat.mulVec Wcls dy)
  show clsPadFlat N D (fun i => ∑ j, Wcls i j * dy j) idx
      = (cls_slice_flat_has_vjp N D).backward v (Mat.mulVec Wcls dy) idx
  show (let p := finProdFinEquiv.symm idx
        if p.1 = (0 : Fin (N + 1)) then (fun i => ∑ j, Wcls i j * dy j) p.2 else 0)
      = (let p := finProdFinEquiv.symm idx
         if p.1 = (0 : Fin (N + 1)) then Mat.mulVec Wcls dy p.2 else 0)
  rfl

-- ── Stage 2: Final vec-LN backward graph (over N+1 tokens) ──

/-- The final (pre-head) vector-LN backward graph over `(N+1)` tokens. REUSES the
    vec-LN LN-back fragment (`lnRowBack(γ=1) ∘ rowScaleF γF`) chained onto the
    classifier back. The bias backward (identity) drops out; `X` is the saved
    pre-LN input (the body output). -/
noncomputable def finalLNBackGraph (N D nClasses : Nat) (ε : ℝ) (γF : Vec D)
    (Wcls : Mat D nClasses)
    (X : Vec ((N+1)*D)) (dy : Vec nClasses) : SHlo ((N+1)*D) :=
  .lnRowBack "%gF" "%XF" "ε" ε 1 X
    (.rowScaleF "%gFv" γF (classifierBackGraph N D nClasses Wcls dy))

/-- **Final vec-LN backward-graph faithfulness (Stage 2).** Denotes the bridged
    per-token vec-LN VJP back of the classifier back, at the saved body output `X`. -/
theorem finalLNBackGraph_faithful (N D nClasses : Nat) (ε : ℝ) (γF βF : Vec D) (hε : 0 < ε)
    (Wcls : Mat D nClasses) (bcls : Vec nClasses)
    (X : Mat (N+1) D) (dy : Vec nClasses) :
    den (finalLNBackGraph N D nClasses ε γF Wcls (Mat.flatten X) dy)
      = Mat.flatten ((layerNormVec_per_token_has_vjp_mat (N+1) D ε γF βF hε).backward X
          (Mat.unflatten
            ((classifier_flat_has_vjp N D nClasses Wcls bcls).backward (Mat.flatten X) dy))) := by
  simp only [finalLNBackGraph, lnRowBack_faithful, rowScaleF_faithful]
  rw [classifierBackGraph_faithful N D nClasses Wcls bcls (Mat.flatten X) dy]
  -- The classifier-back cotangent is a raw Vec; present it as `Mat.flatten (unflatten ·)`
  -- so the vec-LN LN-back bridge (which expects a flattened Mat cotangent) applies.
  -- Rewrite ONLY the LHS `rowScaleFlat`-fed cotangent (NOT the RHS's `Mat.unflatten`).
  set cb : Vec ((N + 1) * D) :=
    (classifier_flat_has_vjp N D nClasses Wcls bcls).backward (Mat.flatten X) dy with hcb
  conv_lhs => rw [show cb = Mat.flatten (Mat.unflatten cb) from (Mat.flatten_unflatten cb).symm]
  rw [rowVecLNBack_eq_backward (βv := βF) ε γF hε X (Mat.unflatten cb)]

-- ── Stage 3: Depth-`k` tower backward (reverse fold) ──

/-- `transformerBlockVBackGraphMH` at a bundled `BlockParamsV` block (the backward
    analogue of `vitBlockGraphMHVP`). Saved: the block input `A` and its
    attn-sublayer output `h = attnSublayerV A`. -/
noncomputable def transformerBlockVBackGraphMHP {Np1 hm1 d mlpDim : Nat} (ε : ℝ)
    (p : BlockParamsV ((hm1+1) * d) mlpDim)
    (A : Mat Np1 ((hm1+1) * d)) (dY : Vec (Np1 * ((hm1+1) * d))) :
    SHlo (Np1 * ((hm1+1) * d)) :=
  transformerBlockVBackGraphMH ε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo
    p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 A
    (transformerAttnSublayerV Np1 (hm1+1) d ε p.γ1 p.β1 p.Wq p.Wk p.Wv p.Wo
      p.bq p.bk p.bv p.bo A) dY

/-- **Bundled per-block backward faithfulness.** The block back graph at a bundled
    `BlockParamsV` block (saved input `A`, cotangent `dY`) denotes the flatten of
    the proven `blockV`'s VJP backward — `transformerBlockV_has_vjp_mat.backward`,
    spelled ONCE here over a generic `p` (so the depth-`k` induction never re-spells
    the 16-field tuple). The block VJP is bundled as `transformerBlockV_has_vjp_matP`. -/
noncomputable def transformerBlockV_has_vjp_matP {N hm1 d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV ((hm1+1) * d) mlpDim) :
    HasVJPMat (blockV N (hm1+1) d mlpDim ε p) :=
  transformerBlockV_has_vjp_mat N (hm1+1) d mlpDim ε p.γ1 p.β1 hε p.Wq p.Wk p.Wv p.Wo
    p.bq p.bk p.bv p.bo p.γ2 p.β2 p.Wfc1 p.bfc1 p.Wfc2 p.bfc2

theorem transformerBlockVBackGraphMHP_faithful {Np1 hm1 d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV ((hm1+1) * d) mlpDim) (A dY : Mat Np1 ((hm1+1) * d)) :
    den (transformerBlockVBackGraphMHP ε p A (Mat.flatten dY))
      = Mat.flatten ((transformerBlockV_has_vjp_matP ε hε p).backward A dY) :=
  transformerBlockVBackGraphMH_faithful ε p.γ1 p.β1 p.γ2 p.β2 hε
    p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 A dY

/-- **Bundled per-block flat-VJP unfold.** The depth-`(k+1)` tower flat-VJP backward
    factors (by `vjp_comp`'s rule) as block `0`'s `hasVJPMat`-bridged backward of the
    depth-`k` tail backward, at the post-block-`0` flat activation. Spells the
    16-tuple ONCE; the induction only chains this. Proven by the SHALLOW projection
    equations of `vjp_comp` and `hasVJPMat_to_hasVJP` (NOT a deep `rfl`, which the
    kernel times out on — the tail VJP term is enormous). -/
theorem vitBodyKVFlat_has_vjp_succ_backward {Np1 hm1 d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε)
    (k : Nat) (ps : Fin (k+1) → BlockParamsV ((hm1+1) * d) mlpDim)
    (vA vc : Vec (Np1 * ((hm1+1) * d))) :
    (vitBodyKVFlat_has_vjp Np1 (hm1+1) d mlpDim ε hε (k+1) ps).backward vA vc
      = Mat.flatten ((transformerBlockV_has_vjp_matP ε hε (ps 0)).backward (Mat.unflatten vA)
          (Mat.unflatten
            ((vitBodyKVFlat_has_vjp Np1 (hm1+1) d mlpDim ε hε k (fun i => ps i.succ)).backward
              (blockVFlat Np1 (hm1+1) d mlpDim ε (ps 0) vA) vc))) := by
  -- Unfold ONLY the outer recursion equation + the two SHALLOW projection rules
  -- (`vjp_comp.backward`, `hasVJPMat_to_hasVJP.backward`), then collapse the
  -- `Mat.flatten`/`Mat.unflatten` bridge. The tail VJP stays opaque — no deep `rfl`.
  rw [vitBodyKVFlat_has_vjp]
  show (hasVJPMat_to_hasVJP (transformerBlockV_has_vjp_matP ε hε (ps 0))).backward vA
        ((vitBodyKVFlat_has_vjp Np1 (hm1+1) d mlpDim ε hε k (fun i => ps i.succ)).backward
          (blockVFlat Np1 (hm1+1) d mlpDim ε (ps 0) vA) vc) = _
  rfl

/-- **Depth-`k` tower backward graph** — the REVERSE fold of
    `transformerBlockVBackGraphMHP`. The forward body runs block `0` first
    (`vitBodyKV (k+1) ps = vitBodyKV k (ps∘succ) ∘ blockV (ps 0)`), so the
    backward runs block `0` LAST: the incoming cotangent flows through the tail
    (blocks `1..k`, at the post-block-`0` activation `blockV (ps 0) A`), then
    through block `0` (at the saved block input `A`). Mirrors
    `vitBodyKVFlat_has_vjp`'s `vjp_comp` chain. -/
noncomputable def vitBodyBackGraphKMHV {Np1 hm1 d mlpDim : Nat} (ε : ℝ) :
    (k : Nat) → (Fin k → BlockParamsV ((hm1+1) * d) mlpDim) →
    Mat Np1 ((hm1+1) * d) → SHlo (Np1 * ((hm1+1) * d)) → SHlo (Np1 * ((hm1+1) * d))
  | 0, _, _, e => e
  | k + 1, ps, A, e =>
      transformerBlockVBackGraphMHP ε (ps 0) A
        (den (vitBodyBackGraphKMHV ε k (fun i => ps i.succ)
          (blockV Np1 (hm1+1) d mlpDim ε (ps 0) A) e))

/-- **Depth-`k` tower backward den** — by induction on `k`, chaining the bundled
    `transformerBlockVBackGraphMHP_faithful` per block: the reverse fold denotes
    the flatten of the inductive tower VJP backward
    (`vitBodyKVFlat_has_vjp.backward`) at the saved block input `A`, fed any
    incoming cotangent whose den is the flatten of `dInner`. The backward
    analogue of `vitBodyGraphKMHV_den`. -/
theorem vitBodyBackGraphKMHV_den {Np1 hm1 d mlpDim : Nat} (ε : ℝ) (hε : 0 < ε) :
    ∀ (k : Nat) (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
      (A : Mat Np1 ((hm1+1) * d)) (e : SHlo (Np1 * ((hm1+1) * d)))
      (dInner : Mat Np1 ((hm1+1) * d)),
      den e = Mat.flatten dInner →
      den (vitBodyBackGraphKMHV ε k ps A e)
        = (vitBodyKVFlat_has_vjp Np1 (hm1+1) d mlpDim ε hε k ps).backward
              (Mat.flatten A) (Mat.flatten dInner)
  | 0, _, _, e, dInner, he => by
      -- depth-0 VJP backward = identity.
      show den e = Mat.flatten dInner
      exact he
  | k + 1, ps, A, e, dInner, he => by
      -- blockVFlat (ps 0) (flatten A) = flatten (blockV (ps 0) A).
      have hblock : blockVFlat Np1 (hm1+1) d mlpDim ε (ps 0) (Mat.flatten A)
          = Mat.flatten (blockV Np1 (hm1+1) d mlpDim ε (ps 0) A) := by
        unfold blockVFlat; rw [Mat.unflatten_flatten]
      -- IH at the post-block-0 activation `B := blockV (ps 0) A`.
      set B : Mat Np1 ((hm1+1) * d) := blockV Np1 (hm1+1) d mlpDim ε (ps 0) A with hB
      have ih := vitBodyBackGraphKMHV_den ε hε k (fun i => ps i.succ) B e dInner he
      -- LHS: den (block-back graph at A, fed the tail-back graph den).
      show den (transformerBlockVBackGraphMHP ε (ps 0) A
            (den (vitBodyBackGraphKMHV ε k (fun i => ps i.succ) B e))) = _
      -- The tail-back graph den IS the tail VJP backward at flatten B (by IH).
      rw [ih]
      -- Present the tail cotangent as `flatten (unflatten ·)` and apply the block faithful.
      set tc : Vec (Np1 * ((hm1+1) * d)) :=
        (vitBodyKVFlat_has_vjp Np1 (hm1+1) d mlpDim ε hε k (fun i => ps i.succ)).backward
          (Mat.flatten B) (Mat.flatten dInner) with htc
      rw [show tc = Mat.flatten (Mat.unflatten tc) from (Mat.flatten_unflatten tc).symm]
      rw [transformerBlockVBackGraphMHP_faithful ε hε (ps 0) A (Mat.unflatten tc)]
      -- RHS: reduce the depth-(k+1) tower VJP backward one block (bundled unfold),
      -- then match flatten/unflatten round-trips: unflatten (flatten A) → A, then
      -- fold the tail cotangent (blockVFlat → flatten B via hblock) back into `tc`.
      rw [vitBodyKVFlat_has_vjp_succ_backward ε hε k ps (Mat.flatten A) (Mat.flatten dInner)]
      rw [Mat.unflatten_flatten, hblock, ← htc]

-- ── Stage 4: patchEmbed input-backward graph ──

/-- The patch-embedding input-backward graph: the `patchEmbedBack` token (the
    strided-patchify conv's input-VJP) on the patch-embed-output cotangent. -/
noncomputable def patchEmbedBackGraph (ic H W P N D : Nat)
    (Wc : Kernel4 D ic P P) (e : SHlo ((N+1)*D)) : SHlo (ic*H*W) :=
  .patchEmbedBack "%Wp" Wc e

/-- **patchEmbed input-backward tie.** The local re-spelling `patchEmbedBackFlat`
    IS the proven `patchEmbed_flat_has_vjp.backward` (`patchEmbed_input_grad_formula`,
    activation-independent — patchEmbed is affine). (`rfl`, like the forward tie.) -/
theorem patchEmbedBackFlat_eq_backward (ic H W P N D : Nat)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N+1) D)
    (img : Vec (ic*H*W)) (dy : Vec ((N+1)*D)) :
    patchEmbedBackFlat ic H W P N D Wc dy
      = (patchEmbed_flat_has_vjp ic H W P N D Wc bc cls pos).backward img dy := rfl

/-- **patchEmbed input-backward-graph faithfulness (Stage 4).** Denotes the proven
    `patchEmbed_flat_has_vjp.backward` at any saved image `img` (linear — the
    activation is irrelevant). -/
theorem patchEmbedBackGraph_faithful (ic H W P N D : Nat)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N+1) D)
    (img : Vec (ic*H*W)) (e : SHlo ((N+1)*D)) :
    den (patchEmbedBackGraph ic H W P N D Wc e)
      = (patchEmbed_flat_has_vjp ic H W P N D Wc bc cls pos).backward img (den e) := by
  simp only [patchEmbedBackGraph, patchEmbedBack_faithful]
  rw [patchEmbedBackFlat_eq_backward ic H W P N D Wc bc cls pos img (den e)]

-- ── Stage 5: Whole-net backward graph + faithfulness ──

/-- **Whole-net depth-`k` multi-head vector-LN ViT backward graph.** Mirrors the
    forward `vitFwdGraphKMHV` in REVERSE: classifier-back → final-vec-LN-back →
    depth-`k` tower-back (reverse fold) → patchEmbed-back. Each stage is fed the
    saved forward activation it differentiates at:
      * `xin` — the saved image (input to patchEmbed; patchEmbed-back is linear).
      * `bodyOut` — the saved body output (input to the final LN).
      * the tower-back fold threads each block's saved input internally from the
        patchEmbed output `embOut` (= the body input). -/
noncomputable def vitNetBackGraph
    (ic H W patchSize N mlpDim hm1 d nClasses k : Nat) (ε : ℝ)
    (Wc : Kernel4 ((hm1+1) * d) ic patchSize patchSize)
    (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
    (γF : Vec ((hm1+1) * d)) (Wcls : Mat ((hm1+1) * d) nClasses)
    (embOut : Mat (N+1) ((hm1+1) * d)) (bodyOut : Mat (N+1) ((hm1+1) * d))
    (dy : Vec nClasses) : SHlo (ic*H*W) :=
  patchEmbedBackGraph ic H W patchSize N ((hm1+1) * d) Wc
    (vitBodyBackGraphKMHV ε k ps embOut
      (finalLNBackGraph N ((hm1+1) * d) nClasses ε γF Wcls (Mat.flatten bodyOut) dy))

set_option maxHeartbeats 1000000 in
/-- **Whole-net backward-graph faithfulness (capstone).** The reverse-composed
    backward graph denotes the proven whole-net VJP `vitForwardKV_has_vjp.backward`
    at every input image `x` and output cotangent `dy`, at EVERY depth `k`, for the
    production config (depth-`k`, multi-head `heads = hm1+1`, vector-LN). The
    backward analogue of `vitFwdGraphKMHV_faithful`: it composes the four stages by
    unfolding the three `vjp_comp` backward rules and bridging Vec↔Mat via
    `Mat.flatten`/`Mat.unflatten`, mirroring how the forward faithfulness composes
    the forward stages. The saved activations (`embOut`, `bodyOut`) are pinned to
    the forward intermediates `patchEmbed_flat x` and `vitBodyKVFlat (patchEmbed_flat x)`. -/
theorem vitNetBackGraph_faithful
    (ic H W patchSize N mlpDim hm1 d nClasses k : Nat) (ε : ℝ) (hε : 0 < ε)
    (Wc : Kernel4 ((hm1+1) * d) ic patchSize patchSize)
    (bc cls : Vec ((hm1+1) * d)) (pos : Mat (N + 1) ((hm1+1) * d))
    (ps : Fin k → BlockParamsV ((hm1+1) * d) mlpDim)
    (γF βF : Vec ((hm1+1) * d))
    (Wcls : Mat ((hm1+1) * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) (dy : Vec nClasses) :
    den (vitNetBackGraph ic H W patchSize N mlpDim hm1 d nClasses k ε Wc ps γF Wcls
          (Mat.unflatten
            (patchEmbed_flat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x))
          (Mat.unflatten
            (vitBodyKVFlat (N + 1) (hm1+1) d mlpDim ε k ps
              (patchEmbed_flat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x)))
          dy)
      = (vitForwardKV_has_vjp ic H W patchSize N mlpDim (hm1+1) d nClasses k
          Wc bc cls pos ε hε ps γF βF Wcls bcls).backward x dy := by
  -- Abbreviate the forward stages exactly as `vitForwardKV_has_vjp`'s proof does.
  set PE := patchEmbed_flat ic H W patchSize N ((hm1+1) * d) Wc bc cls pos with hPE
  set BODY := vitBodyKVFlat (N + 1) (hm1+1) d mlpDim ε k ps with hBODY
  set LNF := (fun v : Vec ((N + 1) * ((hm1+1) * d)) =>
    Mat.flatten (fun n => layerNormVec ((hm1+1) * d) ε γF βF ((Mat.unflatten v) n))) with hLNF
  -- The whole-net VJP's backward unfolds (vjp_comp backward rule, rfl-transparent) to:
  --   PE.back x (BODY.back (PE x) (LNF.back (BODY (PE x)) (classifier.back (LNF (BODY (PE x))) dy)))
  have hunfold :
      (vitForwardKV_has_vjp ic H W patchSize N mlpDim (hm1+1) d nClasses k
          Wc bc cls pos ε hε ps γF βF Wcls bcls).backward x dy
        = (patchEmbed_flat_has_vjp ic H W patchSize N ((hm1+1) * d) Wc bc cls pos).backward x
            ((vitBodyKVFlat_has_vjp (N + 1) (hm1+1) d mlpDim ε hε k ps).backward (PE x)
              ((hasVJPMat_to_hasVJP (layerNormVec_per_token_has_vjp_mat (N + 1) ((hm1+1) * d)
                  ε γF βF hε)).backward (BODY (PE x))
                ((classifier_flat_has_vjp N ((hm1+1) * d) nClasses Wcls bcls).backward
                  (LNF (BODY (PE x))) dy))) := rfl
  rw [hunfold]
  -- Now build the graph side: patchEmbedBack (tower (finalLN (classifier))).
  simp only [vitNetBackGraph]
  -- Stage 4: patchEmbedBack at saved image x.
  rw [patchEmbedBackGraph_faithful ic H W patchSize N ((hm1+1) * d) Wc bc cls pos x]
  congr 1
  -- Inner: den (tower-back graph at embOut = PE x, fed the finalLN back graph).
  -- The tower-back den (Stage 3) needs `den (finalLN graph) = flatten dInner`.
  set cClsf : Vec ((N + 1) * ((hm1+1) * d)) :=
    (hasVJPMat_to_hasVJP (layerNormVec_per_token_has_vjp_mat (N + 1) ((hm1+1) * d)
        ε γF βF hε)).backward (BODY (PE x))
      ((classifier_flat_has_vjp N ((hm1+1) * d) nClasses Wcls bcls).backward
        (LNF (BODY (PE x))) dy) with hcClsf
  -- finalLN graph's den = flatten of the vec-LN per-token back of the classifier back,
  -- at the saved body output bodyOut = unflatten (BODY (PE x)).
  have hLNgraph :
      den (finalLNBackGraph N ((hm1+1) * d) nClasses ε γF Wcls
            (Mat.flatten (Mat.unflatten (BODY (PE x)))) dy)
        = Mat.flatten (Mat.unflatten cClsf) := by
    rw [finalLNBackGraph_faithful N ((hm1+1) * d) nClasses ε γF βF hε Wcls bcls
          (Mat.unflatten (BODY (PE x))) dy]
    -- `unflatten cClsf = (lnVec).backward (unflatten (BODY (PE x))) (unflatten (classifier.back …))`:
    -- `cClsf = hasVJPMat_to_hasVJP.backward (BODY (PE x)) cc = flatten ((lnVec).backward (unflatten (BODY (PE x))) (unflatten cc))`
    -- (definitional), so `unflatten cClsf` cancels via `Mat.unflatten_flatten` (a LEMMA, not rfl).
    -- The classifier back is activation-INDEPENDENT, so the two `cc` legs are then defeq.
    have hcc : Mat.unflatten cClsf
        = (layerNormVec_per_token_has_vjp_mat (N + 1) ((hm1+1) * d) ε γF βF hε).backward
            (Mat.unflatten (BODY (PE x)))
            (Mat.unflatten ((classifier_flat_has_vjp N ((hm1+1) * d) nClasses Wcls bcls).backward
              (LNF (BODY (PE x))) dy)) := by
      rw [hcClsf]
      show Mat.unflatten (Mat.flatten
            ((layerNormVec_per_token_has_vjp_mat (N + 1) ((hm1+1) * d) ε γF βF hε).backward
              (Mat.unflatten (BODY (PE x)))
              (Mat.unflatten ((classifier_flat_has_vjp N ((hm1+1) * d) nClasses Wcls bcls).backward
                (LNF (BODY (PE x))) dy)))) = _
      rw [Mat.unflatten_flatten]
    rw [hcc]
    -- Remaining: the two sides differ only in the classifier-back's saved activation
    -- (`flatten (unflatten (BODY (PE x)))` vs `LNF (BODY (PE x))`). The classifier VJP
    -- backward IGNORES its activation (`vjp_comp` of two activation-free linear backs),
    -- so they are equal — close by the activation-independence equation.
    have hclsf_indep : ∀ a b : Vec ((N + 1) * ((hm1+1) * d)),
        (classifier_flat_has_vjp N ((hm1+1) * d) nClasses Wcls bcls).backward a dy
          = (classifier_flat_has_vjp N ((hm1+1) * d) nClasses Wcls bcls).backward b dy :=
      fun _ _ => rfl
    rw [hclsf_indep (Mat.flatten (Mat.unflatten (BODY (PE x)))) (LNF (BODY (PE x)))]
  -- Feed hLNgraph to the tower den (Stage 3), at embOut = unflatten (PE x).
  rw [vitBodyBackGraphKMHV_den ε hε k ps (Mat.unflatten (PE x))
        (finalLNBackGraph N ((hm1+1) * d) nClasses ε γF Wcls
          (Mat.flatten (Mat.unflatten (BODY (PE x)))) dy)
        (Mat.unflatten cClsf) hLNgraph]
  -- tower den RHS = tail.backward (flatten (unflatten (PE x))) (flatten (unflatten cClsf));
  -- round-trips cancel to BODY.backward (PE x) cClsf.
  rw [Mat.flatten_unflatten, Mat.flatten_unflatten]

end Proofs.StableHLO
