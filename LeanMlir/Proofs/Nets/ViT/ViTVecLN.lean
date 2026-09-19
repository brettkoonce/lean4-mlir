import LeanMlir.Proofs.Nets.ViT.ViTChainClose

/-! # ViT scaling pass — the vector-[D] LayerNorm upgrade

The representative ViT close (Items A–D) used the proof's *scalar* LN γ/β. The
committed production render (`ViTRender.lean`, the GPU-trained ViT-Tiny) is MORE
faithful: vector `γ, β : [D]` per LN site, decomposed as
`scalar-LN(1,0) ∘ per-channel scale γ ∘ + β`. This file brings the close to that
form — `planning/archive/vit_close.md`'s top scaling-pass item:

* **`layerNormVec`** — per-token normalize (scalar-LN at γ=1, β=0) then the
  per-channel affine `γ ⊙ · + β`, with `HasVJP` composed from `layerNorm_has_vjp`
  (at 1,0), `layerScale_has_vjp` (ch9), and the bias translation.
* **`transformerBlockV`** — the vector-LN block, with its VJP composed through the
  sublayer recipe (`biPathMat_has_vjp` + `vjpMat_comp` + `rowwise_has_vjp_mat`).
  UNCONDITIONAL except `0 < ε`. The depth-`k` net built from it is `ViTDepthK`'s.
* **The row-broadcast flat bridges** (`rowScaleFlat_flat`, `rowBiasFlat_flat`) for the
  tokens each vector-LN site is spelled with: `lnRowF`(1,0) → `rowScaleF γ` → `rowBiasF β`
  (exactly the ViTRender decomposition).
* **Vector γ/β param bridges** (`vit_render_vecln{gamma,beta}_certified`) — the
  per-channel grads `dγ_k = Σ_tokens dy_(r,k)·x̂_r(k)`, `dβ_k = Σ_tokens dy_(r,k)`
  (reduce over batch+tokens, KEEP the channel axis), certified via the
  masked-gather Jacobian recipe.
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § 1. Vector-[D] LayerNorm — per-token forward + VJP
-- ════════════════════════════════════════════════════════════════

/-- **Vector-[D] LayerNorm**: per-token normalize (the scalar LN at γ=1, β=0 — pure
    x̂), then the per-channel affine `γ ⊙ x̂ + β`. The committed `ViTRender` LN form. -/
noncomputable def layerNormVec (D : Nat) (ε : ℝ) (γv βv : Vec D) (x : Vec D) : Vec D :=
  fun k => γv k * layerNormForward D ε 1 0 x k + βv k

lemma layerNormVec_diff (D : Nat) (ε : ℝ) (γv βv : Vec D) (hε : 0 < ε) :
    Differentiable ℝ (layerNormVec D ε γv βv) := by
  unfold layerNormVec; fun_prop (disch := assumption)

/-- The bias translation's VJP — backward is the identity (`dx = dy`). -/
noncomputable def biasAdd_has_vjp {n : Nat} (βv : Vec n) :
    HasVJP (fun z : Vec n => fun k => z k + βv k) where
  backward := fun _z dy => dy
  correct := by
    intro z dy i
    simp [pdiv_id_add_const βv z]

/-- **Vector-LN VJP** — `(+β) ∘ layerScale γ ∘ LN(1,0)`, three proven pieces glued
    by `vjp_comp`. Only `0 < ε`. -/
noncomputable def layerNormVec_has_vjp (D : Nat) (ε : ℝ) (γv βv : Vec D)
    (hε : 0 < ε) : HasVJP (layerNormVec D ε γv βv) :=
  have h1 : Differentiable ℝ (layerNormForward D ε 1 0) :=
    bnForward_differentiable D ε 1 0 hε
  have h2 : Differentiable ℝ (layerScale γv) := layerScale_differentiable γv
  have h3 : Differentiable ℝ (fun z : Vec D => fun k => z k + βv k) := by
    rw [differentiable_pi]; intro k
    exact (differentiable_pi.mp differentiable_id k).add_const (βv k)
  vjp_comp _ (fun z : Vec D => fun k => z k + βv k) (h2.comp h1) h3
    (vjp_comp (layerNormForward D ε 1 0) (layerScale γv) h1 h2
      (layerNorm_has_vjp D ε 1 0 hε) (layerScale_has_vjp γv))
    (biasAdd_has_vjp βv)

/-- Per-token vector-LN across a sequence — the rowwise lift. -/
noncomputable def layerNormVec_per_token_has_vjp_mat (N D : Nat) (ε : ℝ)
    (γv βv : Vec D) (hε : 0 < ε) :
    HasVJPMat (fun X : Mat N D => fun r => layerNormVec D ε γv βv (X r)) :=
  rowwise_has_vjp_mat (layerNormVec_has_vjp D ε γv βv hε)
    (layerNormVec_diff D ε γv βv hε)

/-- Generic flat differentiability of a rowwise lift — each output coordinate
    is a coordinate of the per-row map applied to one row of the input. -/
lemma rowwise_flat_diff {N D P : Nat} (g : Vec D → Vec P)
    (hg : Differentiable ℝ g) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => g (X n)) (Mat.unflatten v))) := by
  unfold Mat.flatten Mat.unflatten; fun_prop

lemma layerNormVec_per_token_flat_diff (N D : Nat) (ε : ℝ) (γv βv : Vec D)
    (hε : 0 < ε) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => layerNormVec D ε γv βv (X n))
                   (Mat.unflatten v))) :=
  rowwise_flat_diff _ (layerNormVec_diff D ε γv βv hε)

-- ════════════════════════════════════════════════════════════════
-- § 2. The vector-LN transformer block (sublayers + block + VJP)
-- ════════════════════════════════════════════════════════════════

/-- Attention sublayer with vector-LN: `X ↦ X + MHSA(LNᵥ(X))`. -/
noncomputable def transformerAttnSublayerV (N heads d_head : Nat) (ε : ℝ)
    (γ1 β1 : Vec (heads * d_head))
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Mat N (heads * d_head) → Mat N (heads * d_head) :=
  biPathMat
    (fun X => X)
    ((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
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
lemma transformerAttnSublayerV_inner_flat_diff
    (N heads d_head : Nat) (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten
        (((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
          (fun X : Mat N (heads * d_head) => fun n =>
            layerNormVec (heads * d_head) ε γ1 β1 (X n)))
         (Mat.unflatten v))) := by
  simpa [Function.comp_def, Mat.unflatten_flatten] using
    (mhsa_layer_flat_diff N heads d_head Wq Wk Wv Wo bq bk bv bo).comp
      (layerNormVec_per_token_flat_diff N (heads * d_head) ε γ1 β1 hε)

/-- Flat Diff of the attentionᵥ sublayer. -/
lemma transformerAttnSublayerV_flat_diff
    (N heads d_head : Nat) (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerAttnSublayerV N heads d_head ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo (Mat.unflatten v))) := by
  exact (identity_mat_flat_diff N (heads * d_head)).add
    (transformerAttnSublayerV_inner_flat_diff N heads d_head ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo)

/-- Attentionᵥ sublayer VJP. -/
noncomputable def transformerAttnSublayerV_has_vjp_mat (N heads d_head : Nat)
    (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat (transformerAttnSublayerV N heads d_head ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo) :=
  preLNRes_has_vjp_mat _ _ (layerNormVec_per_token_flat_diff N (heads * d_head) ε γ1 β1 hε)
    (mhsa_layer_flat_diff N heads d_head Wq Wk Wv Wo bq bk bv bo)
    (layerNormVec_per_token_has_vjp_mat N (heads * d_head) ε γ1 β1 hε)
    (mhsa_has_vjp_mat N heads d_head Wq Wk Wv Wo bq bk bv bo)

/-- Flat Diff of the MLPᵥ sublayer's non-trivial arm. -/
lemma transformerMlpSublayerV_inner_flat_diff
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
    (transformerMlp_flat_diff N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2).comp
      (layerNormVec_per_token_flat_diff N (heads * d_head) ε γ2 β2 hε)

/-- Flat Diff of the MLPᵥ sublayer. -/
lemma transformerMlpSublayerV_flat_diff
    (N heads d_head mlpDim : Nat) (ε : ℝ) (γ2 β2 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2
                     Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v))) := by
  exact (identity_mat_flat_diff N (heads * d_head)).add
    (transformerMlpSublayerV_inner_flat_diff N heads d_head mlpDim ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2)

/-- MLPᵥ sublayer VJP. -/
noncomputable def transformerMlpSublayerV_has_vjp_mat (N heads d_head mlpDim : Nat)
    (ε : ℝ) (γ2 β2 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2
                 Wfc1 bfc1 Wfc2 bfc2) :=
  preLNRes_has_vjp_mat _ _ (layerNormVec_per_token_flat_diff N (heads * d_head) ε γ2 β2 hε)
    (transformerMlp_flat_diff N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)
    (layerNormVec_per_token_has_vjp_mat N (heads * d_head) ε γ2 β2 hε)
    (transformerMlp_has_vjp_mat N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)

/-- Flat Diff of the vector-LN block. -/
lemma transformerBlockV_flat_diff (N heads d_head mlpDim : Nat)
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
    (transformerMlpSublayerV_flat_diff N heads d_head mlpDim ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2).comp
      (transformerAttnSublayerV_flat_diff N heads d_head ε γ1 β1 hε Wq Wk Wv Wo bq bk bv bo)

/-- **Vector-LN block VJP** — one `vjpMat_comp` of the two sublayer witnesses. -/
noncomputable def transformerBlockV_has_vjp_mat (N heads d_head mlpDim : Nat)
    (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : Vec (heads * d_head))
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerBlockV N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo
                 γ2 β2 Wfc1 bfc1 Wfc2 bfc2) :=
  vjpMat_comp _ (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
    (transformerAttnSublayerV_flat_diff N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (transformerMlpSublayerV_flat_diff N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)
    (transformerAttnSublayerV_has_vjp_mat N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (transformerMlpSublayerV_has_vjp_mat N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § 3. Row-broadcast flat bridges
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
-- § 4. Vector γ/β param bridges (the Item C analogue at vector LN)
--
-- As a function of `γv : Vec D`, the rowwise vector-LN site is a coefficient-gather:
-- `y_(r,k) = x̂_r(k)·γv(k) + βv(k)` — the masked-gather Jacobian recipe
-- (`pdiv_maskGather_add_const`) with the per-row x̂ as the coefficient. The
-- per-channel grads keep the channel axis: `dγ_k = Σ_tokens dy_(r,k)·x̂_r(k)`,
-- `dβ_k = Σ_tokens dy_(r,k)` — `ViTRender`'s LN param-grad reduces.
-- ════════════════════════════════════════════════════════════════

/-- **Jacobian of the rowwise vector-LN site w.r.t. γv** —
    `∂y_(r,k)/∂γv_i = δ_(i,k)·x̂_r(k)`. -/
theorem pdiv_vecLN_gamma {N D : Nat} (ε : ℝ) (βv : Vec D) (X : Mat N D)
    (γ : Vec D) (i : Fin D) (o : Fin (N * D)) :
    pdiv (fun gv : Vec D =>
            Mat.flatten (fun r => layerNormVec D ε gv βv (X r))) γ i o
      = layerNormForward D ε 1 0 (X (finProdFinEquiv.symm o).1)
          (finProdFinEquiv.symm o).2 *
        (if i = (finProdFinEquiv.symm o).2 then 1 else 0) := by
  rw [show (fun gv : Vec D => Mat.flatten (fun r => layerNormVec D ε gv βv (X r)))
        = (fun gv : Vec D => fun o' : Fin (N * D) =>
            (fun o'' : Fin (N * D) =>
              layerNormForward D ε 1 0 (X (finProdFinEquiv.symm o'').1)
                (finProdFinEquiv.symm o'').2) o' *
              gv ((fun o'' : Fin (N * D) => (finProdFinEquiv.symm o'').2) o') +
            (fun o'' : Fin (N * D) =>
              βv (finProdFinEquiv.symm o'').2) o') from by
      funext gv o'
      unfold layerNormVec Mat.flatten
      ring]
  exact pdiv_maskGather_add_const _ _ _ γ i o

/-- **Jacobian of the rowwise vector-LN site w.r.t. βv** — `∂y_(r,k)/∂βv_i = δ_(i,k)`. -/
theorem pdiv_vecLN_beta {N D : Nat} (ε : ℝ) (γv : Vec D) (X : Mat N D)
    (β : Vec D) (i : Fin D) (o : Fin (N * D)) :
    pdiv (fun bv : Vec D =>
            Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o
      = if i = (finProdFinEquiv.symm o).2 then 1 else 0 := by
  rw [show (fun bv : Vec D => Mat.flatten (fun r => layerNormVec D ε γv bv (X r)))
        = fun bv => (fun o' : Fin (N * D) => bv (finProdFinEquiv.symm o').2) +
            fun o' => γv (finProdFinEquiv.symm o').2 *
              layerNormForward D ε 1 0 (X (finProdFinEquiv.symm o').1)
                (finProdFinEquiv.symm o').2 from by
      funext bv o'
      unfold layerNormVec Mat.flatten
      exact add_comm _ _,
    pdiv_of_affine _ _ (fun _ _ => rfl) (fun _ _ => rfl)]
  simp [@eq_comm _ i]

/-- The rendered **vector-LN γ gradient**: per-channel, the batch+token reduce
    `dγ_k = Σ_r dY_(r,k)·x̂_r(k)` (KEEPS the channel axis — `ViTRender`'s form). -/
noncomputable def vecLN_grad_gamma (N D : Nat) (ε : ℝ) (X dY : Mat N D) : Vec D :=
  fun i => ∑ r : Fin N, dY r i * layerNormForward D ε 1 0 (X r) i

/-- The rendered **vector-LN β gradient**: `dβ_k = Σ_r dY_(r,k)`. -/
noncomputable def vecLN_grad_beta (N D : Nat) (dY : Mat N D) : Vec D :=
  fun i => ∑ r : Fin N, dY r i

/-- **Vector-LN γ-gradient bridge.** -/
theorem vit_veclnGamma_grad_bridge {N D : Nat} (ε : ℝ) (βv : Vec D) (γ : Vec D)
    (X : Mat N D) (dy : Vec (N * D)) (i : Fin D) :
    vecLN_grad_gamma N D ε X (Mat.unflatten dy) i
      = ∑ o : Fin (N * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε gv βv (X r))) γ i o
            * dy o := by
  simp_rw [pdiv_vecLN_gamma]
  rw [sum_finProdFinEquiv (m := N) (n := D)]
  simp [vecLN_grad_gamma, Mat.unflatten, mul_comm]

/-- **Vector-LN β-gradient bridge.** -/
theorem vit_veclnBeta_grad_bridge {N D : Nat} (ε : ℝ) (γv : Vec D) (β : Vec D)
    (X : Mat N D) (dy : Vec (N * D)) (i : Fin D) :
    vecLN_grad_beta N D (Mat.unflatten dy) i
      = ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o
            * dy o := by
  simp_rw [pdiv_vecLN_beta]
  rw [sum_finProdFinEquiv (m := N) (n := D)]
  simp [vecLN_grad_beta, Mat.unflatten]

/-- **Vector-LN γ output, certified.** `γvⁿ_k = γv_k − lr·(Σ_tokens dy·x̂)_k` denotes
    the certified rowwise vector-LN ∂/∂γv contraction. Covers all five LN sites of
    the vector-LN representative (and is the `ViTRender` per-channel LN-γ reduce). -/
theorem vit_render_veclngamma_certified {N D : Nat} (ε : ℝ) (βv : Vec D)
    (γ : Vec D) (X : Mat N D) (dy : Vec (N * D)) (lr : ℝ) (i : Fin D) :
    γ i - lr * vecLN_grad_gamma N D ε X (Mat.unflatten dy) i
      = γ i - lr * ∑ o : Fin (N * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε gv βv (X r))) γ i o
            * dy o := by
  rw [vit_veclnGamma_grad_bridge ε βv γ X dy i]

/-- **Vector-LN β output, certified.** -/
theorem vit_render_veclnbeta_certified {N D : Nat} (ε : ℝ) (γv : Vec D)
    (β : Vec D) (X : Mat N D) (dy : Vec (N * D)) (lr : ℝ) (i : Fin D) :
    β i - lr * vecLN_grad_beta N D (Mat.unflatten dy) i
      = β i - lr * ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o
            * dy o := by
  rw [vit_veclnBeta_grad_bridge ε γv β X dy i]

end Proofs

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § 5. Vector-LN chain cotangents (the Item D analogue)
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
