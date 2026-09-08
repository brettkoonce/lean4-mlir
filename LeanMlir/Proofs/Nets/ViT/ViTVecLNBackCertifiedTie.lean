import LeanMlir.Proofs.Nets.ViT.ViTMhsaBackCertifiedTie
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackCertifiedTie
import LeanMlir.Proofs.Nets.ViT.ViTBackChains

/-! # §B: the ViT encoder-block backward tie at the VECTOR LayerNorm the net runs

`ViTMhsaBackCertifiedTie.lean` closes the block tie at `γ1 β1 γ2 β2 : ℝ` — the *scalar* LayerNorm
of the representative close — against `transformerBlock_has_vjp_mat`. The shipped depth-12 net is
`vitForwardKV`, whose blocks are `transformerBlockV` at `γ β : Vec D`, the form
`ViTRender.lean` emits and `ViTVecLN.lean` calls *"MORE faithful"*. This file re-states the tie
there, so the whole-net fold in `ViTWholeBackCertifiedTie.lean` is about the blocks the artifact
contains. It is package 3.1's move for ConvNeXt-T, one architecture over.

⭐ **There is no new analysis, and the reason is that ConvNeXt already built ViT's LayerNorm
backward.** `rowLNVecFlatBack` (`ChannelLNBack.lean`) is `perRowFlatPR` of
`bn_grad_input c ε 1 (X r) ∘ diagBack γ` — the γ scale in front of the unit-γ input gradient at
each row's own saved activation — and its own header says it is *"literally ViT's per-token LN
with 'token' read as 'spatial position'"*. `rowLNVecFlat_has_vjp_backward_eq` already pins it to
`layerNormVec_per_token_has_vjp_mat`. So the vector-LN seam (`rowLNVecFlatBack_eq_vecLN_vjp`) is
that lemma read at a flat saved input, and everything else in the block — `mhsaBackFlat`, the
`dense Wᵀ 0` input-VJPs, the `diagBack` GELU derivative, the `perRowFlatPR` residual seams — is
LayerNorm-agnostic and is reused from the scalar file verbatim.

⚠ Both sublayer decompositions and the block unfold are `rfl` at the vector LN exactly as they
are at the scalar one: `transformerBlockV_has_vjp_mat` is the same `vjpMat_comp` /
`biPathMat_has_vjp` assembly with one argument changed, so the projections reduce identically.
3-axiom-clean.
-/

namespace Proofs

variable {h N dh : Nat}

-- ════════════════════════════════════════════════════════════════
-- § The vector-LayerNorm seam
-- ════════════════════════════════════════════════════════════════

/-- **The vector-LN backward at a flat saved input IS the certified per-token vector-LN VJP,
    flattened.** `rowLNVecFlat_has_vjp_backward_eq` (ConvNeXt's) read through
    `hasVJPMat_to_hasVJP`'s projection. The vector-LN peer of `perRowFlatPR_LN_back`, and the
    only LayerNorm-specific step in this file. -/
theorem rowLNVecFlatBack_eq_vecLN_vjp (n D : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec D)
    (X dy : Vec (n * D)) :
    rowLNVecFlatBack n D ε γ X dy
      = Mat.flatten ((layerNormVec_per_token_has_vjp_mat n D ε γ β hε).backward
          (Mat.unflatten X) (Mat.unflatten dy)) := by
  rw [← rowLNVecFlat_has_vjp_backward_eq ε hε γ β]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § The two sublayer backward decompositions (both `rfl`)
-- ════════════════════════════════════════════════════════════════

/-- **The vector-LN attention sublayer's VJP backward decomposes** (`biPathMat` unfold, `rfl`):
    the residual skip passes the cotangent through, and the non-trivial arm is
    `LNᵥ₁-back ∘ mhsa-back` at the saved `LNᵥ₁` output. -/
theorem transformerAttnSublayerV_backward_decomp (ε : ℝ) (hε : 0 < ε) (γ1 β1 : Vec (h * dh))
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh)) (A dY : Mat N (h * dh)) :
    (transformerAttnSublayerV_has_vjp_mat N h dh ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo).backward A dY
      = (fun i j => dY i j +
          (layerNormVec_per_token_has_vjp_mat N (h * dh) ε γ1 β1 hε).backward A
            ((mhsa_has_vjp_mat N h dh Wq Wk Wv Wo bq bk bv bo).backward
              (fun n => layerNormVec (h * dh) ε γ1 β1 (A n)) dY) i j) := rfl

/-- **The vector-LN MLP sublayer's VJP backward decomposes** — the MLP peer, also `rfl`. -/
theorem transformerMlpSublayerV_backward_decomp (dff : Nat) (ε : ℝ) (hε : 0 < ε)
    (γ2 β2 : Vec (h * dh))
    (Wfc1 : Mat (h * dh) dff) (bfc1 : Vec dff) (Wfc2 : Mat dff (h * dh)) (bfc2 : Vec (h * dh))
    (hM dz : Mat N (h * dh)) :
    (transformerMlpSublayerV_has_vjp_mat N h dh dff ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2).backward hM dz
      = fun i j => dz i j +
          (layerNormVec_per_token_has_vjp_mat N (h * dh) ε γ2 β2 hε).backward hM
            ((transformerMlp_has_vjp_mat N (h * dh) dff Wfc1 bfc1 Wfc2 bfc2).backward
              (fun n => layerNormVec (h * dh) ε γ2 β2 (hM n)) dz) i j := rfl

/-- **The vector-LN block VJP backward unfolds**: `block.backward A dz = attn.backward A
    (mlp.backward (attn A) dz)`. The outer `vjpMat_comp`'s projection, `rfl`. -/
theorem transformerBlockV_backward_unfold (dff : Nat) (ε : ℝ) (hε : 0 < ε)
    (γ1 β1 γ2 β2 : Vec (h * dh))
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh))
    (Wfc1 : Mat (h * dh) dff) (bfc1 : Vec dff) (Wfc2 : Mat dff (h * dh)) (bfc2 : Vec (h * dh))
    (A dz : Mat N (h * dh)) :
    (transformerBlockV_has_vjp_mat N h dh dff ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward A dz
      = (transformerAttnSublayerV_has_vjp_mat N h dh ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo).backward A
          ((transformerMlpSublayerV_has_vjp_mat N h dh dff ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2).backward
            (transformerAttnSublayerV N h dh ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A) dz) := rfl

-- ════════════════════════════════════════════════════════════════
-- § The two sublayer flat ties
-- ════════════════════════════════════════════════════════════════

/-- **The attention-sublayer backward float-half IS the certified sublayer VJP, flat.** The
    vector-LN peer of `attnSubFlatTie`: `residual (rowLNVecFlatBack ∘ mhsaBackFlat)` with Q/K/V
    pinned at `LNᵥ₁(A)` and the LN backward at the block's own saved input. The sdpa leaf is
    `mhsaBackFlat_eq_mhsa_vjp`, which never mentions a LayerNorm. -/
theorem attnSubFlatTieV (ε : ℝ) (hε : 0 < ε) (γ1 β1 : Vec (h * dh))
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh))
    (A : Mat N (h * dh)) (w : Vec (N * (h * dh))) :
    Proofs.residual (rowLNVecFlatBack N (h * dh) ε γ1 (Mat.flatten A)
        ∘ mhsaBackFlat Wq Wk Wv Wo
            (fun r => Proofs.dense Wq bq (layerNormVec (h * dh) ε γ1 β1 (A r)))
            (fun r => Proofs.dense Wk bk (layerNormVec (h * dh) ε γ1 β1 (A r)))
            (fun r => Proofs.dense Wv bv (layerNormVec (h * dh) ε γ1 β1 (A r)))) w
      = Mat.flatten ((transformerAttnSublayerV_has_vjp_mat N h dh ε γ1 β1 hε
          Wq Wk Wv Wo bq bk bv bo).backward A (Mat.unflatten w)) := by
  have hmhsa : (mhsa_has_vjp_mat N h dh Wq Wk Wv Wo bq bk bv bo).backward
        (fun n => layerNormVec (h * dh) ε γ1 β1 (A n)) (Mat.unflatten w)
      = Mat.unflatten (mhsaBackFlat Wq Wk Wv Wo
          (fun r => Proofs.dense Wq bq (layerNormVec (h * dh) ε γ1 β1 (A r)))
          (fun r => Proofs.dense Wk bk (layerNormVec (h * dh) ε γ1 β1 (A r)))
          (fun r => Proofs.dense Wv bv (layerNormVec (h * dh) ε γ1 β1 (A r))) w) := by
    rw [congrFun (mhsaBackFlat_eq_mhsa_vjp Wq Wk Wv Wo bq bk bv bo
          (fun n => layerNormVec (h * dh) ε γ1 β1 (A n))) w, Mat.unflatten_flatten]
  rw [transformerAttnSublayerV_backward_decomp ε hε γ1 β1 Wq Wk Wv Wo bq bk bv bo A
        (Mat.unflatten w), hmhsa]
  funext idx
  simp only [Proofs.residual, biPath, Function.comp_apply, Mat.flatten]
  rw [rowLNVecFlatBack_eq_vecLN_vjp (β := β1) N (h * dh) ε hε, Mat.unflatten_flatten]
  have hw : Mat.unflatten w (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = w idx := by
    show w (finProdFinEquiv ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2)) = w idx
    rw [Prod.mk.eta, Equiv.apply_symm_apply]
  rw [hw]
  exact add_comm _ _

/-- **The MLP-sublayer backward float-half IS the certified sublayer VJP, flat.** The vector-LN
    peer of `mlpSubFlatTie`, with the residual lifted out of the per-token fold. The per-token
    body tie (`transformerMlp_back_flat_eq_perRowFlatPR`) is LayerNorm-agnostic and reused
    verbatim; only the LN₂-back seam changes. -/
theorem mlpSubFlatTieV (dff : Nat) (ε : ℝ) (hε : 0 < ε) (γ2 β2 : Vec (h * dh))
    (Wfc1 : Mat (h * dh) dff) (bfc1 : Vec dff) (Wfc2 : Mat dff (h * dh)) (bfc2 : Vec (h * dh))
    (hM : Mat N (h * dh)) (v : Vec (N * (h * dh))) :
    Proofs.residual (rowLNVecFlatBack N (h * dh) ε γ2 (Mat.flatten hM)
        ∘ perRowFlatPR N (h * dh) (fun r =>
            Proofs.dense (Mat.transpose Wfc1) (0 : Vec (h * dh))
              ∘ diagBack (fun c => geluScalarDeriv (Proofs.dense Wfc1 bfc1
                  (layerNormVec (h * dh) ε γ2 β2 (hM r)) c))
              ∘ Proofs.dense (Mat.transpose Wfc2) (0 : Vec dff))) v
      = Mat.flatten ((transformerMlpSublayerV_has_vjp_mat N h dh dff ε γ2 β2 hε
          Wfc1 bfc1 Wfc2 bfc2).backward hM (Mat.unflatten v)) := by
  rw [transformerMlpSublayerV_backward_decomp dff ε hε γ2 β2 Wfc1 bfc1 Wfc2 bfc2 hM
        (Mat.unflatten v)]
  funext idx
  simp only [Proofs.residual, biPath, Function.comp_apply]
  rw [← transformerMlp_back_flat_eq_perRowFlatPR N (h * dh) dff Wfc1 bfc1 Wfc2 bfc2
        (fun n => layerNormVec (h * dh) ε γ2 β2 (hM n)) v,
      rowLNVecFlatBack_eq_vecLN_vjp (β := β2) N (h * dh) ε hε, Mat.unflatten_flatten,
      Mat.unflatten_flatten]
  simp only [Mat.flatten]
  have hv : Mat.unflatten v (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = v idx := by
    show v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2)) = v idx
    rw [Prod.mk.eta, Equiv.apply_symm_apply]
  rw [hv]
  exact add_comm _ _

-- ════════════════════════════════════════════════════════════════
-- § THE CAPSTONE — the vector-LN ViT block backward tie
-- ════════════════════════════════════════════════════════════════

/-- ⭐⭐ **THE VECTOR-LN ViT BLOCK §B TIE.** `vitBlockBackV`, with every saved activation pinned
    to the real forward (Q/K/V at `LNᵥ₁ A`; the LN₁ backward at the block's own input `A` and the
    LN₂ backward at the attention sublayer's output; the GELU derivative at
    `dense₁(LNᵥ₂(attn A))`), IS the certified `transformerBlockV` input-gradient VJP, flattened.

    This is `vitBlockBackPR_eq_transformerBlock_vjp` at the LayerNorm the shipped net actually
    has. Assembled from the block unfold and the two sublayer flat ties; no new analysis, and
    general in the head count exactly as the scalar tie is. -/
theorem vitBlockBackV_eq_transformerBlockV_vjp (dff : Nat) (ε : ℝ) (hε : 0 < ε)
    (γ1 β1 γ2 β2 : Vec (h * dh))
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (bq bk bv bo : Vec (h * dh))
    (Wfc1 : Mat (h * dh) dff) (bfc1 : Vec dff) (Wfc2 : Mat dff (h * dh)) (bfc2 : Vec (h * dh))
    (A : Mat N (h * dh)) :
    vitBlockBackV Wq Wk Wv Wo
        (fun r => Proofs.dense Wq bq (layerNormVec (h * dh) ε γ1 β1 (A r)))
        (fun r => Proofs.dense Wk bk (layerNormVec (h * dh) ε γ1 β1 (A r)))
        (fun r => Proofs.dense Wv bv (layerNormVec (h * dh) ε γ1 β1 (A r)))
        ε γ1 (Mat.flatten A) Wfc1 Wfc2
        (fun r => fun c => geluScalarDeriv (Proofs.dense Wfc1 bfc1
          (layerNormVec (h * dh) ε γ2 β2
            (transformerAttnSublayerV N h dh ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A r)) c))
        γ2 (Mat.flatten (transformerAttnSublayerV N h dh ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A))
      = fun dY => Mat.flatten ((transformerBlockV_has_vjp_mat N h dh dff ε γ1 β1 hε
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward A (Mat.unflatten dY)) := by
  funext dY
  set hM : Mat N (h * dh) :=
    transformerAttnSublayerV N h dh ε γ1 β1 Wq Wk Wv Wo bq bk bv bo A with hhM
  show Proofs.residual (rowLNVecFlatBack N (h * dh) ε γ1 (Mat.flatten A)
      ∘ mhsaBackFlat Wq Wk Wv Wo
          (fun r => Proofs.dense Wq bq (layerNormVec (h * dh) ε γ1 β1 (A r)))
          (fun r => Proofs.dense Wk bk (layerNormVec (h * dh) ε γ1 β1 (A r)))
          (fun r => Proofs.dense Wv bv (layerNormVec (h * dh) ε γ1 β1 (A r))))
      (Proofs.residual (rowLNVecFlatBack N (h * dh) ε γ2 (Mat.flatten hM)
        ∘ perRowFlatPR N (h * dh) (fun r =>
            Proofs.dense (Mat.transpose Wfc1) (0 : Vec (h * dh))
              ∘ diagBack (fun c => geluScalarDeriv (Proofs.dense Wfc1 bfc1
                  (layerNormVec (h * dh) ε γ2 β2 (hM r)) c))
              ∘ Proofs.dense (Mat.transpose Wfc2) (0 : Vec dff))) dY) = _
  rw [mlpSubFlatTieV dff ε hε γ2 β2 Wfc1 bfc1 Wfc2 bfc2 hM dY,
      attnSubFlatTieV ε hε γ1 β1 Wq Wk Wv Wo bq bk bv bo A _,
      Mat.unflatten_flatten,
      ← transformerBlockV_backward_unfold dff ε hε γ1 β1 γ2 β2 Wq Wk Wv Wo bq bk bv bo
        Wfc1 bfc1 Wfc2 bfc2 A (Mat.unflatten dY)]

/-- **The block tie in the form the tower recursion needs** — `vitBlockBackVAt` at a FLAT saved
    input `v` is the flat block's VJP backward at `v`, i.e. exactly the `HasVJP` that
    `vitBodyKVFlat_has_vjp`'s chain step consumes. `Mat.flatten_unflatten` is the only step. -/
theorem vitBlockBackVAt_eq_vjp (Np1 heads d_head mlpDim : Nat) (ε : ℝ) (hε : 0 < ε)
    (p : BlockParamsV (heads * d_head) mlpDim) (v : Vec (Np1 * (heads * d_head))) :
    vitBlockBackVAt Np1 heads d_head mlpDim ε p v
      = (hasVJPMat_to_hasVJP (transformerBlockV_has_vjp_mat Np1 heads d_head mlpDim ε
          p.γ1 p.β1 hε p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo p.γ2 p.β2
          p.Wfc1 p.bfc1 p.Wfc2 p.bfc2)).backward v := by
  have h := vitBlockBackV_eq_transformerBlockV_vjp (N := Np1) (h := heads) (dh := d_head)
    mlpDim ε hε p.γ1 p.β1 p.γ2 p.β2 p.Wq p.Wk p.Wv p.Wo p.bq p.bk p.bv p.bo
    p.Wfc1 p.bfc1 p.Wfc2 p.bfc2 (Mat.unflatten v)
  rw [Mat.flatten_unflatten] at h
  exact h

end Proofs
