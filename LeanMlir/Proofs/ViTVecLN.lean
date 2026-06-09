import LeanMlir.Proofs.ViTChainClose

/-! # ViT scaling pass — the vector-[D] LayerNorm upgrade

The representative ViT close (Items A–D) used the proof's *scalar* LN γ/β. The
committed production render (`ViTRender.lean`, the GPU-trained ViT-Tiny) is MORE
faithful: vector `γ, β : [D]` per LN site, decomposed as
`scalar-LN(1,0) ∘ per-channel scale γ ∘ + β`. This file brings the close to that
form — `planning/vit_close.md`'s top scaling-pass item:

* **`layerNormVec`** — per-token normalize (scalar-LN at γ=1, β=0) then the
  per-channel affine `γ ⊙ · + β`, with `HasVJP` composed from `layerNorm_has_vjp`
  (at 1,0), `layerScale_has_vjp` (ch9), and the bias translation.
* **`transformerBlockV`** / **`vitForward2V`** — the vector-LN block and 2-block
  net, with the whole-net VJP re-composed through the same sublayer recipe as the
  scalar one (`biPathMat_has_vjp` + `vjpMat_comp` + `rowwise_has_vjp_mat`).
  UNCONDITIONAL except `0 < ε`.
* **`vitFwdGraphV` + `vitFwdGraphV_faithful`** — the graph spells each LN site
  with the new broadcast tokens: `lnRowF`(1,0) → `rowScaleF γ` → `rowBiasF β`
  (exactly the ViTRender decomposition); faithful at heads = 1.
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
  unfold layerNormVec
  have h : Differentiable ℝ (layerNormForward D ε 1 0) :=
    bnForward_differentiable D ε 1 0 hε
  rw [differentiable_pi]
  intro k
  have hk : Differentiable ℝ (fun x : Vec D => layerNormForward D ε 1 0 x k) :=
    fun x => differentiableAt_pi.mp (h x) k
  exact (hk.const_mul (γv k)).add_const (βv k)

/-- The bias translation's VJP — backward is the identity (`dx = dy`). -/
noncomputable def biasAdd_has_vjp {n : Nat} (βv : Vec n) :
    HasVJP (fun z : Vec n => fun k => z k + βv k) where
  backward := fun _z dy => dy
  correct := by
    intro z dy i
    simp_rw [pdiv_id_add_const βv z]
    rw [Finset.sum_eq_single i
        (fun j _ hne => by rw [if_neg (Ne.symm hne), zero_mul])
        (fun h => absurd (Finset.mem_univ i) h)]
    rw [if_pos rfl, one_mul]

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
    projects through a row-projection CLM into the per-row map (the
    `layerNorm_per_token_flat_diff` recipe with the row map abstracted). -/
lemma rowwise_flat_diff {N D P : Nat} (g : Vec D → Vec P)
    (hg : Differentiable ℝ g) :
    Differentiable ℝ (fun v : Vec (N * D) =>
      Mat.flatten ((fun X : Mat N D => fun n => g (X n)) (Mat.unflatten v))) := by
  rw [differentiable_pi]
  intro idx
  have h_eq : (fun v : Vec (N * D) =>
        Mat.flatten ((fun X : Mat N D => fun n => g (X n)) (Mat.unflatten v)) idx) =
      (fun w : Vec D => g w (finProdFinEquiv.symm idx).2) ∘
      (fun v : Vec (N * D) => fun j' : Fin D =>
        v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))) := by
    funext v; rfl
  rw [h_eq]
  have h_outer : Differentiable ℝ (fun w : Vec D => g w (finProdFinEquiv.symm idx).2) :=
    fun w => differentiableAt_pi.mp (hg w) (finProdFinEquiv.symm idx).2
  have h_proj : Differentiable ℝ
      (fun v : Vec (N * D) => fun j' : Fin D =>
        v (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))) :=
    (reindexCLM (fun j' : Fin D =>
      finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))).differentiable
  exact h_outer.comp h_proj

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
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten
          (((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε γ1 β1 (X n)))
           (Mat.unflatten v))) =
      (fun u : Vec (N * (heads * d_head)) => Mat.flatten
          (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo (Mat.unflatten u))) ∘
      (fun v : Vec (N * (heads * d_head)) => Mat.flatten
          ((fun X : Mat N (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε γ1 β1 (X n)) (Mat.unflatten v))) := by
    funext v; simp [Function.comp, Mat.unflatten_flatten]
  rw [h_eq]
  exact (mhsa_layer_flat_diff N heads d_head Wq Wk Wv Wo bq bk bv bo).comp
        (layerNormVec_per_token_flat_diff N (heads * d_head) ε γ1 β1 hε)

/-- Flat Diff of the attentionᵥ sublayer. -/
lemma transformerAttnSublayerV_flat_diff
    (N heads d_head : Nat) (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerAttnSublayerV N heads d_head ε γ1 β1
                     Wq Wk Wv Wo bq bk bv bo (Mat.unflatten v))) := by
  unfold transformerAttnSublayerV biPathMat
  have h_id := identity_mat_flat_diff N (heads * d_head)
  have h_inner := transformerAttnSublayerV_inner_flat_diff N heads d_head ε γ1 β1 hε
                    Wq Wk Wv Wo bq bk bv bo
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten (fun (r : Fin N) (s : Fin (heads * d_head)) =>
          (fun X : Mat N (heads * d_head) => X) (Mat.unflatten v) r s +
          ((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε γ1 β1 (X n)))
            (Mat.unflatten v) r s)) =
      fun v => fun k =>
        (fun v' : Vec (N * (heads * d_head)) =>
          Mat.flatten ((fun X : Mat N (heads * d_head) => X) (Mat.unflatten v'))) v k +
        (fun v' : Vec (N * (heads * d_head)) =>
          Mat.flatten (((mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε γ1 β1 (X n)))
            (Mat.unflatten v'))) v k := by
    funext v k; unfold Mat.flatten; rfl
  rw [h_eq]
  exact h_id.add h_inner

/-- Attentionᵥ sublayer VJP. -/
noncomputable def transformerAttnSublayerV_has_vjp_mat (N heads d_head : Nat)
    (ε : ℝ) (γ1 β1 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head)) :
    HasVJPMat (transformerAttnSublayerV N heads d_head ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo) :=
  let inner_has_vjp :=
    vjpMat_comp _ (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo)
      (layerNormVec_per_token_flat_diff N (heads * d_head) ε γ1 β1 hε)
      (mhsa_layer_flat_diff N heads d_head Wq Wk Wv Wo bq bk bv bo)
      (layerNormVec_per_token_has_vjp_mat N (heads * d_head) ε γ1 β1 hε)
      (mhsa_has_vjp_mat N heads d_head Wq Wk Wv Wo bq bk bv bo)
  biPathMat_has_vjp _ _
    (identity_mat_flat_diff N (heads * d_head))
    (transformerAttnSublayerV_inner_flat_diff N heads d_head ε γ1 β1 hε
       Wq Wk Wv Wo bq bk bv bo)
    (identityMat_has_vjp N (heads * d_head))
    inner_has_vjp

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
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten
          (((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε γ2 β2 (X n)))
           (Mat.unflatten v))) =
      (fun u : Vec (N * (heads * d_head)) => Mat.flatten
          (transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2
             (Mat.unflatten u))) ∘
      (fun v : Vec (N * (heads * d_head)) => Mat.flatten
          ((fun X : Mat N (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε γ2 β2 (X n)) (Mat.unflatten v))) := by
    funext v; simp [Function.comp, Mat.unflatten_flatten]
  rw [h_eq]
  exact (transformerMlp_flat_diff N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2).comp
        (layerNormVec_per_token_flat_diff N (heads * d_head) ε γ2 β2 hε)

/-- Flat Diff of the MLPᵥ sublayer. -/
lemma transformerMlpSublayerV_flat_diff
    (N heads d_head mlpDim : Nat) (ε : ℝ) (γ2 β2 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    Differentiable ℝ (fun v : Vec (N * (heads * d_head)) =>
      Mat.flatten (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2
                     Wfc1 bfc1 Wfc2 bfc2 (Mat.unflatten v))) := by
  unfold transformerMlpSublayerV biPathMat
  have h_id := identity_mat_flat_diff N (heads * d_head)
  have h_inner := transformerMlpSublayerV_inner_flat_diff N heads d_head mlpDim
                    ε γ2 β2 hε Wfc1 bfc1 Wfc2 bfc2
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten (fun (r : Fin N) (s : Fin (heads * d_head)) =>
          (fun X : Mat N (heads * d_head) => X) (Mat.unflatten v) r s +
          ((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε γ2 β2 (X n)))
            (Mat.unflatten v) r s)) =
      fun v => fun k =>
        (fun v' : Vec (N * (heads * d_head)) =>
          Mat.flatten ((fun X : Mat N (heads * d_head) => X) (Mat.unflatten v'))) v k +
        (fun v' : Vec (N * (heads * d_head)) =>
          Mat.flatten (((transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2) ∘
            (fun X : Mat N (heads * d_head) => fun n =>
              layerNormVec (heads * d_head) ε γ2 β2 (X n)))
            (Mat.unflatten v'))) v k := by
    funext v k; unfold Mat.flatten; rfl
  rw [h_eq]
  exact h_id.add h_inner

/-- MLPᵥ sublayer VJP. -/
noncomputable def transformerMlpSublayerV_has_vjp_mat (N heads d_head mlpDim : Nat)
    (ε : ℝ) (γ2 β2 : Vec (heads * d_head)) (hε : 0 < ε)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head)) :
    HasVJPMat (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2
                 Wfc1 bfc1 Wfc2 bfc2) :=
  let inner_has_vjp :=
    vjpMat_comp _ (transformerMlp N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)
      (layerNormVec_per_token_flat_diff N (heads * d_head) ε γ2 β2 hε)
      (transformerMlp_flat_diff N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)
      (layerNormVec_per_token_has_vjp_mat N (heads * d_head) ε γ2 β2 hε)
      (transformerMlp_has_vjp_mat N (heads * d_head) mlpDim Wfc1 bfc1 Wfc2 bfc2)
  biPathMat_has_vjp _ _
    (identity_mat_flat_diff N (heads * d_head))
    (transformerMlpSublayerV_inner_flat_diff N heads d_head mlpDim ε γ2 β2 hε
       Wfc1 bfc1 Wfc2 bfc2)
    (identityMat_has_vjp N (heads * d_head))
    inner_has_vjp

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
  unfold transformerBlockV
  have h_eq : (fun v : Vec (N * (heads * d_head)) =>
        Mat.flatten
          (((transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2) ∘
            (transformerAttnSublayerV N heads d_head ε γ1 β1 Wq Wk Wv Wo bq bk bv bo))
           (Mat.unflatten v))) =
      (fun u : Vec (N * (heads * d_head)) => Mat.flatten
          (transformerMlpSublayerV N heads d_head mlpDim ε γ2 β2 Wfc1 bfc1 Wfc2 bfc2
             (Mat.unflatten u))) ∘
      (fun v : Vec (N * (heads * d_head)) => Mat.flatten
          (transformerAttnSublayerV N heads d_head ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
             (Mat.unflatten v))) := by
    funext v; simp [Function.comp, Mat.unflatten_flatten]
  rw [h_eq]
  exact (transformerMlpSublayerV_flat_diff N heads d_head mlpDim ε γ2 β2 hε
            Wfc1 bfc1 Wfc2 bfc2).comp
        (transformerAttnSublayerV_flat_diff N heads d_head ε γ1 β1 hε
            Wq Wk Wv Wo bq bk bv bo)

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

-- ════════════════════════════════════════════════════════════════
-- § 3. The vector-LN 2-block ViT + whole-net VJP
-- ════════════════════════════════════════════════════════════════

/-- **Distinct-param 2-block ViT forward at vector-[D] LN** — the production
    `ViTRender` LN form at the representative architecture. -/
noncomputable def vitForward2V
    (ic H W patchSize N mlpDim heads d_head nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ)
    (γ1₁ β1₁ : Vec (heads * d_head))
    (Wq₁ Wk₁ Wv₁ Wo₁ : Mat (heads * d_head) (heads * d_head))
    (bq₁ bk₁ bv₁ bo₁ : Vec (heads * d_head))
    (γ2₁ β2₁ : Vec (heads * d_head))
    (Wfc1₁ : Mat (heads * d_head) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim (heads * d_head)) (bfc2₁ : Vec (heads * d_head))
    (γ1₂ β1₂ : Vec (heads * d_head))
    (Wq₂ Wk₂ Wv₂ Wo₂ : Mat (heads * d_head) (heads * d_head))
    (bq₂ bk₂ bv₂ bo₂ : Vec (heads * d_head))
    (γ2₂ β2₂ : Vec (heads * d_head))
    (Wfc1₂ : Mat (heads * d_head) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim (heads * d_head)) (bfc2₂ : Vec (heads * d_head))
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Vec (ic * H * W) → Vec nClasses :=
  (classifier_flat N (heads * d_head) nClasses Wcls bcls) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (fun n => layerNormVec (heads * d_head) ε γF βF
      ((Mat.unflatten v) n))) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (transformerBlockV (N + 1) heads d_head mlpDim ε γ1₂ β1₂
      Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
      (Mat.unflatten v))) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (transformerBlockV (N + 1) heads d_head mlpDim ε γ1₁ β1₁
      Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
      (Mat.unflatten v))) ∘
  (patchEmbed_flat ic H W patchSize N (heads * d_head)
    W_conv b_conv cls_token pos_embed)

/-- **Whole-net VJP for the vector-LN 2-block ViT (global)** — only `0 < ε`. -/
noncomputable def vitForward2V_has_vjp
    (ic H W patchSize N mlpDim heads d_head nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (γ1₁ β1₁ : Vec (heads * d_head))
    (Wq₁ Wk₁ Wv₁ Wo₁ : Mat (heads * d_head) (heads * d_head))
    (bq₁ bk₁ bv₁ bo₁ : Vec (heads * d_head))
    (γ2₁ β2₁ : Vec (heads * d_head))
    (Wfc1₁ : Mat (heads * d_head) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim (heads * d_head)) (bfc2₁ : Vec (heads * d_head))
    (γ1₂ β1₂ : Vec (heads * d_head))
    (Wq₂ Wk₂ Wv₂ Wo₂ : Mat (heads * d_head) (heads * d_head))
    (bq₂ bk₂ bv₂ bo₂ : Vec (heads * d_head))
    (γ2₂ β2₂ : Vec (heads * d_head))
    (Wfc1₂ : Mat (heads * d_head) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim (heads * d_head)) (bfc2₂ : Vec (heads * d_head))
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    HasVJP (vitForward2V ic H W patchSize N mlpDim heads d_head nClasses
      W_conv b_conv cls_token pos_embed ε
      γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
      γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
      γF βF Wcls bcls) := by
  unfold vitForward2V
  set PE := patchEmbed_flat ic H W patchSize N (heads * d_head)
              W_conv b_conv cls_token pos_embed with hPE
  have pe_diff := patchEmbed_flat_diff ic H W patchSize N (heads * d_head)
                    W_conv b_conv cls_token pos_embed
  have pe_vjp : HasVJP PE := patchEmbed_flat_has_vjp ic H W patchSize N
                    (heads * d_head) W_conv b_conv cls_token pos_embed
  set B1 := (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (transformerBlockV (N + 1) heads d_head mlpDim ε γ1₁ β1₁
      Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
      (Mat.unflatten v))) with hB1
  have b1_diff := transformerBlockV_flat_diff (N + 1) heads d_head mlpDim
                    ε γ1₁ β1₁ hε Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁
                    γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
  have b1_vjp : HasVJP B1 :=
    hasVJPMat_to_hasVJP (transformerBlockV_has_vjp_mat (N + 1) heads d_head mlpDim
      ε γ1₁ β1₁ hε Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁)
  have s1_vjp : HasVJP (B1 ∘ PE) := vjp_comp PE B1 pe_diff b1_diff pe_vjp b1_vjp
  have s1_diff : Differentiable ℝ (B1 ∘ PE) := b1_diff.comp pe_diff
  set B2 := (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (transformerBlockV (N + 1) heads d_head mlpDim ε γ1₂ β1₂
      Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
      (Mat.unflatten v))) with hB2
  have b2_diff := transformerBlockV_flat_diff (N + 1) heads d_head mlpDim
                    ε γ1₂ β1₂ hε Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂
                    γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
  have b2_vjp : HasVJP B2 :=
    hasVJPMat_to_hasVJP (transformerBlockV_has_vjp_mat (N + 1) heads d_head mlpDim
      ε γ1₂ β1₂ hε Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂)
  have s2_vjp : HasVJP (B2 ∘ (B1 ∘ PE)) :=
    vjp_comp (B1 ∘ PE) B2 s1_diff b2_diff s1_vjp b2_vjp
  have s2_diff : Differentiable ℝ (B2 ∘ (B1 ∘ PE)) := b2_diff.comp s1_diff
  set LNF := (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (fun n => layerNormVec (heads * d_head) ε γF βF
      ((Mat.unflatten v) n))) with hLNF
  have lnf_diff : Differentiable ℝ LNF :=
    layerNormVec_per_token_flat_diff (N + 1) (heads * d_head) ε γF βF hε
  have lnf_vjp : HasVJP LNF :=
    hasVJPMat_to_hasVJP (layerNormVec_per_token_has_vjp_mat (N + 1) (heads * d_head)
      ε γF βF hε)
  have s3_vjp : HasVJP (LNF ∘ (B2 ∘ (B1 ∘ PE))) :=
    vjp_comp (B2 ∘ (B1 ∘ PE)) LNF s2_diff lnf_diff s2_vjp lnf_vjp
  have s3_diff : Differentiable ℝ (LNF ∘ (B2 ∘ (B1 ∘ PE))) := lnf_diff.comp s2_diff
  exact vjp_comp (LNF ∘ (B2 ∘ (B1 ∘ PE)))
    (classifier_flat N (heads * d_head) nClasses Wcls bcls)
    s3_diff (classifier_flat_diff N (heads * d_head) nClasses Wcls bcls)
    s3_vjp (classifier_flat_has_vjp N (heads * d_head) nClasses Wcls bcls)

/-- **Public correctness theorem for `vitForward2V_has_vjp`** — the vector-LN
    2-block ViT's backward equals the `pdiv`-contracted Jacobian at every input. -/
theorem vitForward2V_has_vjp_correct
    (ic H W patchSize N mlpDim heads d_head nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (γ1₁ β1₁ : Vec (heads * d_head))
    (Wq₁ Wk₁ Wv₁ Wo₁ : Mat (heads * d_head) (heads * d_head))
    (bq₁ bk₁ bv₁ bo₁ : Vec (heads * d_head))
    (γ2₁ β2₁ : Vec (heads * d_head))
    (Wfc1₁ : Mat (heads * d_head) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim (heads * d_head)) (bfc2₁ : Vec (heads * d_head))
    (γ1₂ β1₂ : Vec (heads * d_head))
    (Wq₂ Wk₂ Wv₂ Wo₂ : Mat (heads * d_head) (heads * d_head))
    (bq₂ bk₂ bv₂ bo₂ : Vec (heads * d_head))
    (γ2₂ β2₂ : Vec (heads * d_head))
    (Wfc1₂ : Mat (heads * d_head) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim (heads * d_head)) (bfc2₂ : Vec (heads * d_head))
    (γF βF : Vec (heads * d_head))
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) (dy : Vec nClasses) (i : Fin (ic * H * W)) :
    (vitForward2V_has_vjp ic H W patchSize N mlpDim heads d_head nClasses
      W_conv b_conv cls_token pos_embed ε hε
      γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
      γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
      γF βF Wcls bcls).backward x dy i =
      ∑ j : Fin nClasses,
        pdiv (vitForward2V ic H W patchSize N mlpDim heads d_head nClasses
          W_conv b_conv cls_token pos_embed ε
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls) x i j * dy j :=
  (vitForward2V_has_vjp ic H W patchSize N mlpDim heads d_head nClasses
    W_conv b_conv cls_token pos_embed ε hε
    γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
    γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
    γF βF Wcls bcls).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § 4. The vector-LN spelled block (Mat level)
-- ════════════════════════════════════════════════════════════════

/-- The spelled vector-LN block at heads = 1 — each LN site decomposed as the graph
    (and `ViTRender`) emit it: pure normalize (scalar-LN at 1,0) → per-channel scale
    → per-channel bias. -/
noncomputable def vitBlockSpelledV (Np1 d mlpDim : Nat) (ε : ℝ)
    (γ1 β1 : Vec (1 * d))
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (γ2 β2 : Vec (1 * d))
    (Wfc1 : Mat (1 * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d)) (bfc2 : Vec (1 * d))
    (X : Mat Np1 (1 * d)) : Mat Np1 (1 * d) :=
  let xh1 : Mat Np1 (1 * d) := fun r => layerNormForward (1 * d) ε 1 0 (X r)
  let sc1 : Mat Np1 (1 * d) := fun r => layerScale γ1 (xh1 r)
  let ln1 : Mat Np1 (1 * d) := fun r k => sc1 r k + β1 k
  let Q : Mat Np1 (1 * d) := fun r => dense Wq bq (ln1 r)
  let K : Mat Np1 (1 * d) := fun r => dense Wk bk (ln1 r)
  let V : Mat Np1 (1 * d) := fun r => dense Wv bv (ln1 r)
  let P : Mat Np1 Np1 :=
    rowSoftmax (fun i j => sdpa_scale d * Mat.mul Q (Mat.transpose K) i j)
  let O : Mat Np1 (1 * d) := fun r => dense Wo bo (Mat.mul P V r)
  let h : Mat Np1 (1 * d) := fun r s => X r s + O r s
  let xh2 : Mat Np1 (1 * d) := fun r => layerNormForward (1 * d) ε 1 0 (h r)
  let sc2 : Mat Np1 (1 * d) := fun r => layerScale γ2 (xh2 r)
  let ln2 : Mat Np1 (1 * d) := fun r k => sc2 r k + β2 k
  let m1 : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let g : Mat Np1 mlpDim := fun r => gelu mlpDim (m1 r)
  let m2 : Mat Np1 (1 * d) := fun r => dense Wfc2 bfc2 (g r)
  fun r s => h r s + m2 r s

/-- **The spelled vector-LN block IS `transformerBlockV` at one head** — the
    three-stage LN decomposition collapses to `layerNormVec` definitionally; the
    per-head plumbing collapses via `mhsa_layer_one_head`. -/
lemma vitBlockSpelledV_eq (Np1 d mlpDim : Nat) (ε : ℝ)
    (γ1 β1 : Vec (1 * d))
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (γ2 β2 : Vec (1 * d))
    (Wfc1 : Mat (1 * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d)) (bfc2 : Vec (1 * d))
    (X : Mat Np1 (1 * d)) :
    vitBlockSpelledV Np1 d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X =
      transformerBlockV Np1 1 d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X := by
  unfold transformerBlockV transformerMlpSublayerV transformerAttnSublayerV
         transformerMlp biPathMat vitBlockSpelledV
  simp only [Function.comp_apply]
  rw [mhsa_layer_one_head]
  rfl

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § 5. The vector-LN graph + faithfulness
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

/-- One spelled vector-LN block over the tokens: each LN site is
    `lnRowF`(1,0) → `rowScaleF γ` → `rowBiasF β` (the `ViTRender` decomposition);
    everything else as `vitBlockGraph`. `oneStr`/`zeroStr` name the rendered
    constant-1/0 scalars the pure-normalize sites reference. -/
def vitBlockGraphV {Np1 D mlpDim : Nat} (pfx epsStr sStr oneStr zeroStr : String)
    (ε s : ℝ) (γ1 β1 : Vec D)
    (Wq Wk Wv Wo : Mat D D) (bq bk bv bo : Vec D)
    (γ2 β2 : Vec D)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (x : SHlo (Np1 * D)) : SHlo (Np1 * D) :=
  let ln1 := SHlo.rowBiasF s!"%{pfx}bt1" β1
    (SHlo.rowScaleF s!"%{pfx}g1" γ1
      (SHlo.lnRowF oneStr zeroStr epsStr ε 1 0 x))
  let q := SHlo.denseRowF s!"%{pfx}Wq" s!"%{pfx}bq" Wq bq ln1
  let k := SHlo.denseRowF s!"%{pfx}Wk" s!"%{pfx}bk" Wk bk ln1
  let v := SHlo.denseRowF s!"%{pfx}Wv" s!"%{pfx}bv" Wv bv ln1
  let p := SHlo.softmaxRowF (SHlo.scaleF sStr s (SHlo.matmulF q (SHlo.transposeF k)))
  let o := SHlo.denseRowF s!"%{pfx}Wo" s!"%{pfx}bo" Wo bo (SHlo.matmulF p v)
  let h := SHlo.addV x o
  let ln2 := SHlo.rowBiasF s!"%{pfx}bt2" β2
    (SHlo.rowScaleF s!"%{pfx}g2" γ2
      (SHlo.lnRowF oneStr zeroStr epsStr ε 1 0 h))
  let m2 := SHlo.denseRowF s!"%{pfx}Wfc2" s!"%{pfx}bfc2" Wfc2 bfc2
    (SHlo.geluF (SHlo.denseRowF s!"%{pfx}Wfc1" s!"%{pfx}bfc1" Wfc1 bfc1 ln2))
  SHlo.addV h m2

/-- Block-graph denotation (vector-LN), generalized over the input's Mat form. -/
private lemma vitBlockGraphV_den_aux {Np1 d mlpDim : Nat}
    (pfx epsStr sStr oneStr zeroStr : String) (ε : ℝ)
    (γ1 β1 : Vec (1 * d))
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (γ2 β2 : Vec (1 * d))
    (Wfc1 : Mat (1 * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d)) (bfc2 : Vec (1 * d))
    (e : SHlo (Np1 * (1 * d))) (A : Mat Np1 (1 * d)) (hA : den e = Mat.flatten A) :
    den (vitBlockGraphV pfx epsStr sStr oneStr zeroStr ε (sdpa_scale d) γ1 β1
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 e) =
      Mat.flatten (vitBlockSpelledV Np1 d mlpDim ε γ1 β1
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 A) := by
  simp only [vitBlockGraphV, lnRowF_faithful, rowScaleF_faithful, rowBiasF_faithful,
             denseRowF_faithful, matmulF_faithful, transposeF_faithful, scaleF_faithful,
             softmaxRowF_faithful, geluF_faithful, den_addV, hA]
  simp only [rowLNFlat_flat, rowScaleFlat_flat, rowBiasFlat_flat, rowDenseFlat_flat,
             transposeFlat_flat, matMulFlat_flat, scale_flat, rowSoftmaxFlat_flat,
             gelu_flat, add_flat_pt]
  rfl

/-- Whole **vector-LN ViT forward** graph: patch embed → 2 spelled vector-LN blocks
    (distinct params) → final vector-LN (same three-token decomposition) → CLS slice
    → dense head. -/
def vitFwdGraphV {ic H W P N D mlpDim nClasses : Nat}
    (epsStr sStr oneStr zeroStr : String) (ε s : ℝ)
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (γ1₁ β1₁ : Vec D) (Wq₁ Wk₁ Wv₁ Wo₁ : Mat D D) (bq₁ bk₁ bv₁ bo₁ : Vec D)
    (γ2₁ β2₁ : Vec D) (Wfc1₁ : Mat D mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim D) (bfc2₁ : Vec D)
    (γ1₂ β1₂ : Vec D) (Wq₂ Wk₂ Wv₂ Wo₂ : Mat D D) (bq₂ bk₂ bv₂ bo₂ : Vec D)
    (γ2₂ β2₂ : Vec D) (Wfc1₂ : Mat D mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim D) (bfc2₂ : Vec D)
    (γF βF : Vec D) (Wcls : Mat D nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) : SHlo nClasses :=
  let embed : SHlo ((N + 1) * D) :=
    .patchEmbedF "%Wp" "%bp" "%cls" "%pos" Wc bc cls pos (.operand "%x" x)
  let b1 := vitBlockGraphV "b1_" epsStr sStr oneStr zeroStr ε s γ1₁ β1₁
    Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ embed
  let b2 := vitBlockGraphV "b2_" epsStr sStr oneStr zeroStr ε s γ1₂ β1₂
    Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ b1
  let fl := SHlo.rowBiasF "%btF" βF
    (SHlo.rowScaleF "%gF" γF
      (SHlo.lnRowF oneStr zeroStr epsStr ε 1 0 b2))
  denseF "%Wcls" "%bcls" Wcls bcls (.clsSliceF fl)

/-- **Vector-LN ViT forward faithfulness** — the graph denotes `vitForward2V` at
    one head. The scaling-pass peer of `vitFwdGraph_faithful`. -/
theorem vitFwdGraphV_faithful
    (ic H W patchSize N d mlpDim nClasses : Nat)
    (epsStr sStr oneStr zeroStr : String)
    (Wc : Kernel4 (1 * d) ic patchSize patchSize) (bc : Vec (1 * d))
    (cls : Vec (1 * d)) (pos : Mat (N + 1) (1 * d))
    (ε : ℝ)
    (γ1₁ β1₁ : Vec (1 * d)) (Wq₁ Wk₁ Wv₁ Wo₁ : Mat (1 * d) (1 * d))
    (bq₁ bk₁ bv₁ bo₁ : Vec (1 * d))
    (γ2₁ β2₁ : Vec (1 * d)) (Wfc1₁ : Mat (1 * d) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim (1 * d)) (bfc2₁ : Vec (1 * d))
    (γ1₂ β1₂ : Vec (1 * d)) (Wq₂ Wk₂ Wv₂ Wo₂ : Mat (1 * d) (1 * d))
    (bq₂ bk₂ bv₂ bo₂ : Vec (1 * d))
    (γ2₂ β2₂ : Vec (1 * d)) (Wfc1₂ : Mat (1 * d) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim (1 * d)) (bfc2₂ : Vec (1 * d))
    (γF βF : Vec (1 * d)) (Wcls : Mat (1 * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) :
    den (vitFwdGraphV epsStr sStr oneStr zeroStr ε (sdpa_scale d) Wc bc cls pos
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls x)
      = vitForward2V ic H W patchSize N mlpDim 1 d nClasses Wc bc cls pos ε
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls x := by
  have h0 : den (SHlo.patchEmbedF (P := patchSize) "%Wp" "%bp" "%cls" "%pos"
        Wc bc cls pos (.operand "%x" x))
      = Mat.flatten (Mat.unflatten
          (patchEmbed_flat ic H W patchSize N (1 * d) Wc bc cls pos x)) := by
    simp only [patchEmbedF_faithful, den_operand]
    rw [Mat.flatten_unflatten]
    rfl
  have h1 := vitBlockGraphV_den_aux "b1_" epsStr sStr oneStr zeroStr ε γ1₁ β1₁
    Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
    _ _ h0
  have h2 := vitBlockGraphV_den_aux "b2_" epsStr sStr oneStr zeroStr ε γ1₂ β1₂
    Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
    _ _ h1
  simp only [vitFwdGraphV, denseF_faithful, clsSliceF_faithful, rowBiasF_faithful,
             rowScaleF_faithful, lnRowF_faithful, h2]
  simp only [rowLNFlat_flat, rowScaleFlat_flat, rowBiasFlat_flat, vitBlockSpelledV_eq]
  unfold vitForward2V classifier_flat
  simp only [Function.comp_apply, Mat.unflatten_flatten]
  rfl

end Proofs.StableHLO

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § 6. Vector γ/β param bridges (the Item C analogue at vector LN)
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
        = (fun bv : Vec D => fun o' : Fin (N * D) =>
            (fun o'' : Fin (N * D) =>
              γv (finProdFinEquiv.symm o'').2 *
                layerNormForward D ε 1 0 (X (finProdFinEquiv.symm o'').1)
                  (finProdFinEquiv.symm o'').2) o' +
            bv ((fun o'' : Fin (N * D) => (finProdFinEquiv.symm o'').2) o')) from by
      funext bv o'
      unfold layerNormVec Mat.flatten
      rfl]
  have h_const : DifferentiableAt ℝ
      (fun (_ : Vec D) (o'' : Fin (N * D)) =>
        γv (finProdFinEquiv.symm o'').2 *
          layerNormForward D ε 1 0 (X (finProdFinEquiv.symm o'').1)
            (finProdFinEquiv.symm o'').2) β := differentiableAt_const _
  have h_gather : DifferentiableAt ℝ
      (fun (w : Vec D) (o'' : Fin (N * D)) =>
        w ((fun o''' : Fin (N * D) => (finProdFinEquiv.symm o''').2) o'')) β :=
    (reindexCLM (fun o''' : Fin (N * D) => (finProdFinEquiv.symm o''').2)).differentiableAt
  rw [pdiv_add _ _ _ h_const h_gather, pdiv_const, zero_add,
      pdiv_reindex (fun o''' : Fin (N * D) => (finProdFinEquiv.symm o''').2) β i o]

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
  rw [sum_fin_prod N D]
  unfold vecLN_grad_gamma Mat.unflatten
  apply Finset.sum_congr rfl
  intro r _
  rw [Finset.sum_eq_single i
      (fun k _ hne => by
        rw [Equiv.symm_apply_apply]
        dsimp only
        rw [if_neg (Ne.symm hne), mul_zero, zero_mul])
      (fun h => absurd (Finset.mem_univ i) h)]
  rw [Equiv.symm_apply_apply]
  dsimp only
  rw [if_pos rfl, mul_one]
  ring

/-- **Vector-LN β-gradient bridge.** -/
theorem vit_veclnBeta_grad_bridge {N D : Nat} (ε : ℝ) (γv : Vec D) (β : Vec D)
    (X : Mat N D) (dy : Vec (N * D)) (i : Fin D) :
    vecLN_grad_beta N D (Mat.unflatten dy) i
      = ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o
            * dy o := by
  simp_rw [pdiv_vecLN_beta]
  rw [sum_fin_prod N D]
  unfold vecLN_grad_beta Mat.unflatten
  apply Finset.sum_congr rfl
  intro r _
  rw [Finset.sum_eq_single i
      (fun k _ hne => by
        rw [Equiv.symm_apply_apply]
        dsimp only
        rw [if_neg (Ne.symm hne), zero_mul])
      (fun h => absurd (Finset.mem_univ i) h)]
  rw [Equiv.symm_apply_apply]
  dsimp only
  rw [if_pos rfl, one_mul]

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
