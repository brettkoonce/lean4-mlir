import LeanMlir.Proofs.Nets.ViT.ViTChainClose

/-! # ViT at vector-[D] LayerNorm

The committed ViT render (`ViTRender.lean`, the GPU-trained ViT-Tiny) carries a vector `γ, β : [D]`
per LN site, spelled `scalar-LN(1,0) ∘ per-channel scale γ ∘ + β`. The op itself — `layerNormVec`,
its VJP, its per-token lift and its γ/β parameter gradients (`vecLNGradGamma` / `_beta` and the
`vit_render_vecln*_certified` step forms) — is in `Architectures/LayerNorm`, which ConvNeXt's
channel-LN shares. This file is ViT's use of it:

* §1 **`transformerBlockV`** — the vector-LN block, with its VJP composed through the sublayer
  recipe (`biPathMatHasVJP` + `vjpMatComp` + `rowwiseHasVJPMat`). UNCONDITIONAL except
  `0 < ε`. The depth-`k` net built from it is `ViTDepthK`'s.
* §2 **the row-broadcast flat bridges** (`rowScaleFlat_flat`, `rowBiasFlat_flat`) for the tokens
  each vector-LN site is spelled with: `lnRowF`(1,0) → `rowScaleF γ` → `rowBiasF β`.
* §3 **the chain cotangents** at vector LN.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § 1. The vector-LN transformer block (sublayers + block + VJP)
-- ════════════════════════════════════════════════════════════════

/-- Attention sublayer with vector-LN: `X ↦ X + MHSA(LNᵥ(X))`. -/
noncomputable def transformerAttnSublayerV (N heads d_head : Nat) (ε : ℝ)
    (γ1 β1 : Vec (heads * d_head))
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  biPathMat
    (fun X => X)
    ((mhsaLayer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
     (fun X : Mat N (heads * d_head) => fun n =>
        layerNormVec (heads * d_head) ε γ1 β1 (X n)))

/-- MLP sublayer with vector-LN: `h ↦ h + MLP(LNᵥ(h))`. -/
noncomputable def transformerMlpSublayerV (N heads d_head mlpDim : Nat) (ε : ℝ)
    (γ2 β2 : Vec (heads * d_head))
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  biPathMat
    (fun X => X)
    ((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
     (fun X : Mat N (heads * d_head) => fun n =>
        layerNormVec (heads * d_head) ε γ2 β2 (X n)))

/-- **Vector-LN transformer block**: MLPᵥ-sublayer ∘ attentionᵥ-sublayer —
    the `ViTRender` block form. -/
noncomputable def transformerBlockV (N heads d_head mlpDim : Nat) (ε : ℝ)
    (γ1 β1 : Vec (heads * d_head))
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : Vec (heads * d_head))
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
  (transformerAttnSublayerV N heads d_head ε γ1 β1 Wq Wk Wv Wo bq bk bv bo)

/-- Flat Diff of the attentionᵥ sublayer's non-trivial arm (`mhsa ∘ LNᵥ`). -/
lemma transformerAttnSublayerV_inner_flat_differentiable
    (N heads d_head : Nat) (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten
        (((mhsaLayer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
          (fun X : Mat N (heads * d_head) => fun n =>
            layerNormVec (heads * d_head) ε γ1 β1 (X n)))
         (Mat.unflatten v))) := by
  simpa [Function.comp_def, Mat.unflatten_flatten] using
    (mhsaLayer_flat_differentiable N heads d_head Wq Wk Wv Wo bq bk bv bo).comp
      (layerNormVec_per_token_flat_differentiable N (heads * d_head) ε γ1 β1 hε)

/-- Flat Diff of the attentionᵥ sublayer. -/
lemma transformerAttnSublayerV_flat_differentiable
    (N heads d_head : Nat) (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerAttnSublayerV N heads d_head ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo (Mat.unflatten v))) := by
  exact (identity_mat_flat_differentiable N (heads * d_head)).add
    (transformerAttnSublayerV_inner_flat_differentiable N heads d_head ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo)

/-- Attentionᵥ sublayer VJP. -/
noncomputable def transformerAttnSublayerVHasVJPMat (N heads d_head : Nat)
    (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat (transformerAttnSublayerV N heads d_head ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo) :=
  preLNResHasVJPMat _ _ (layerNormVec_per_token_flat_differentiable N (heads * d_head) ε γ1 β1 hε)
    (mhsaLayer_flat_differentiable N heads d_head Wq Wk Wv Wo bq bk bv bo)
    (layerNormVecPerTokenHasVJPMat N (heads * d_head) ε γ1 β1 hε)
    (mhsaHasVJPMat N heads d_head Wq Wk Wv Wo bq bk bv bo)

/-- Flat Diff of the MLPᵥ sublayer's non-trivial arm. -/
lemma transformerMlpSublayerV_inner_flat_differentiable
    (N heads d_head mlpDim : Nat) (ε : ℝ) (γ2 β2 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten
        (((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
          (fun X : Mat N (heads * d_head) => fun n =>
            layerNormVec (heads * d_head) ε γ2 β2 (X n)))
         (Mat.unflatten v))) := by
  simpa [Function.comp_def, Mat.unflatten_flatten] using
    (transformerMlp_flat_differentiable N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2).comp
      (layerNormVec_per_token_flat_differentiable N (heads * d_head) ε γ2 β2 hε)

/-- Flat Diff of the MLPᵥ sublayer. -/
lemma transformerMlpSublayerV_flat_differentiable
    (N heads d_head mlpDim : Nat) (ε : ℝ) (γ2 β2 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2
                     Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v))) := by
  exact (identity_mat_flat_differentiable N (heads * d_head)).add
    (transformerMlpSublayerV_inner_flat_differentiable N heads d_head mlpDim ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2)

/-- MLPᵥ sublayer VJP. -/
noncomputable def transformerMlpSublayerVHasVJPMat (N heads d_head mlpDim : Nat)
    (ε : ℝ) (γ2 β2 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2
                 Wfc1 bfc1 Wfc2 bfc2) :=
  preLNResHasVJPMat _ _ (layerNormVec_per_token_flat_differentiable N (heads * d_head) ε γ2 β2 hε)
    (transformerMlp_flat_differentiable N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)
    (layerNormVecPerTokenHasVJPMat N (heads * d_head) ε γ2 β2 hε)
    (transformerMlpHasVJPMat N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)

/-- Flat Diff of the vector-LN block. -/
lemma transformerBlockV_flat_differentiable (N heads d_head mlpDim : Nat)
    (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : Vec (heads * d_head))
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerBlockV N heads d_head mlpDim ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2
                   (Mat.unflatten v))) := by
  simpa [transformerBlockV, Function.comp_def, Mat.unflatten_flatten] using
    (transformerMlpSublayerV_flat_differentiable N heads d_head mlpDim ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2).comp
      (transformerAttnSublayerV_flat_differentiable N heads d_head ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo)

/-- **Vector-LN block VJP** — one `vjpMatComp` of the two sublayer witnesses. -/
noncomputable def transformerBlockVHasVJPMat (N heads d_head mlpDim : Nat)
    (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : Vec (heads * d_head))
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerBlockV N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo
                 γ2 β2 Wfc1 bfc1 Wfc2 bfc2) :=
  vjpMatComp _ (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
    (transformerAttnSublayerV_flat_differentiable N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (transformerMlpSublayerV_flat_differentiable N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)
    (transformerAttnSublayerVHasVJPMat N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (transformerMlpSublayerVHasVJPMat N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § 2. Row-broadcast flat bridges
-- ════════════════════════════════════════════════════════════════

/-- Row-broadcast scale commutation bridge. -/
lemma rowScaleFlat_flat {m n : Nat} (γ : Vec n) (A : Mat m n) :
    rowScaleFlat m n γ (Mat.flatten A) = Mat.flatten (fun r => layerScale γ (A r)) := by
  unfold rowScaleFlat
  rw [Mat.unflatten_flatten]

/-- Row-broadcast bias commutation bridge. -/
lemma rowBiasFlat_flat {m n : Nat} (β : Vec n) (A : Mat m n) :
    rowBiasFlat m n β (Mat.flatten A) = Mat.flatten (fun r k => A r k + β k) := by
  unfold rowBiasFlat
  rw [Mat.unflatten_flatten]

end Proofs.StableHLO

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § 3. Vector-LN chain cotangents
--
-- The vector-LN block backward decomposes each LN input-VJP as the render emits
-- it: `rowScaleF γ` on the cotangent (diagonal — the forward token), then
-- `lnRowBack`(γ=1) at the saved pre-LN input. The MLP/attention dense segments
-- and the SDPA ties are LN-form-agnostic (`vitCot{G,M1,Ln2,DP,DS,DQ,DK,DV,Ln1}`
-- from `ViTChainClose` hold verbatim); only the residual fan-ins change.
-- ════════════════════════════════════════════════════════════════

/-- Cot at the attention-sublayer output `h`, vector-LN form: `dyOut` + the
    decomposed LN₂ input-VJP (`rowScaleFlat γ2` then `rowLNBackFlat` at γ=1). -/
noncomputable def vitCotHV {Np1 D mlpDim : Nat} (ε : ℝ) (γ2 : Vec D)
    (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D) (h : Vec (Np1 * D))
    (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) : Vec (Np1 * D) :=
  fun i => dyOut i + StableHLO.rowLNBackFlat Np1 D ε 1 h
    (StableHLO.rowScaleFlat Np1 D γ2 (vitCotLn2 Wfc1 Wfc2 m1 dyOut)) i

/-- Cot at the SDPA output, vector-LN form. -/
noncomputable def vitCotAttV {Np1 D mlpDim : Nat} (ε : ℝ) (γ2 : Vec D)
    (Wo : Mat D D) (Wfc1 : Mat D mlpDim) (Wfc2 : Mat mlpDim D)
    (h : Vec (Np1 * D)) (m1 : Vec (Np1 * mlpDim)) (dyOut : Vec (Np1 * D)) :
    Vec (Np1 * D) :=
  StableHLO.rowDenseBackFlat Np1 D D Wo (vitCotHV ε γ2 Wfc1 Wfc2 h m1 dyOut)

/-- Cot at the block input, vector-LN form: `cotH` + the decomposed LN₁
    input-VJP of the three-way Q/K/V fan-in. -/
noncomputable def vitCotXinV {Np1 D : Nat} (ε : ℝ) (γ1 : Vec D)
    (Wq Wk Wv : Mat D D) (xin : Vec (Np1 * D)) (dQ dK dV cotH : Vec (Np1 * D)) :
    Vec (Np1 * D) :=
  fun i => cotH i + StableHLO.rowLNBackFlat Np1 D ε 1 xin
    (StableHLO.rowScaleFlat Np1 D γ1 (vitCotLn1 Wq Wk Wv dQ dK dV)) i

/-- Cot at block 2's output, vector-LN form: the decomposed final-LN input-VJP
    of the classifier-back row-0 scatter. -/
noncomputable def vitCotB2outV (N D nClasses : Nat) (ε : ℝ) (γF : Vec D)
    (Wcls : Mat D nClasses) (b2out : Vec ((N + 1) * D)) (dy : Vec nClasses) :
    Vec ((N + 1) * D) :=
  StableHLO.rowLNBackFlat (N + 1) D ε 1 b2out
    (StableHLO.rowScaleFlat (N + 1) D γF (vitCotFl N D nClasses Wcls dy))

end Proofs
