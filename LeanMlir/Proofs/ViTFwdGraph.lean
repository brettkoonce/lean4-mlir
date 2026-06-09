import LeanMlir.Proofs.StableHLO
import LeanMlir.Proofs.Attention

/-!
# ViT forward graph — ch10 close Item A (planning/vit_close.md)

Two halves, both living here because `StableHLO.lean` cannot import
`Attention.lean` (Attention is the proof capstone; StableHLO is the token
layer — the ViT den helpers there are local re-spellings, tied back to the
proven Attention forms in THIS file):

1. **`vitForward2`** — the representative *distinct-param* 2-block ViT
   forward at the `Vec` level: `classifier ∘ finalLN ∘ block₂ ∘ block₁ ∘
   patchEmbed`. The proven `transformerTower`/`vit_full` share ONE param
   tuple across blocks; a train step needs distinct per-block params, so the
   2-block forward is composed here from `transformerBlock` directly (the
   tower proof does exactly this composition — with shared params).
   `vitForward2_has_vjp` is the whole-net VJP: `vjp_comp` chains
   `patchEmbed_flat_has_vjp`, two bridged `transformerBlock_has_vjp_mat`
   witnesses, the bridged per-token final-LN, and `classifier_flat_has_vjp`.
   UNCONDITIONAL except `0 < ε` (all-smooth — softmax/GELU/LN have no kinks).

2. **`vitFwdGraph`** — the typed `SHlo` forward graph over the ch10 token
   vocabulary (`patchEmbedF`/`lnRowF`/`denseRowF`/`matmulF`/`transposeF`/
   `scaleF`/`softmaxRowF`/`geluF`/`addV`/`clsSliceF`), heads = 1 (SDPA = three
   matmuls + a row-softmax — the representative granularity trade, like
   ConvNeXt's 2-block/1×1-stem). **`vitFwdGraph_faithful`**: its denotation
   IS `vitForward2` at `heads := 1` — the ViT analogue of
   `convNextFwdGraph_faithful`.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § 1. The distinct-param 2-block ViT forward + whole-net VJP
-- ════════════════════════════════════════════════════════════════

/-- **Distinct-param 2-block ViT forward** (the ch10 representative):

      patchEmbed (stride-P conv + CLS + pos) → block₁ → block₂
      → final-LN (per-token, scalar γ/β) → CLS slice → dense head

    Generic dims; the two `transformerBlock`s carry *distinct* parameter
    sets (`…₁`/`…₂`) — beyond the shared-param `transformerTower` witness,
    composed from the same proven block VJP. One shared LN ε across all
    five LN sites (the proof convention, as in `vit_full`). -/
noncomputable def vitForward2
    (ic H W patchSize N mlpDim heads d_head nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ)
    (γ1₁ β1₁ : ℝ)
    (Wq₁ Wk₁ Wv₁ Wo₁ : Mat (heads * d_head) (heads * d_head))
    (bq₁ bk₁ bv₁ bo₁ : Vec (heads * d_head))
    (γ2₁ β2₁ : ℝ)
    (Wfc1₁ : Mat (heads * d_head) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim (heads * d_head)) (bfc2₁ : Vec (heads * d_head))
    (γ1₂ β1₂ : ℝ)
    (Wq₂ Wk₂ Wv₂ Wo₂ : Mat (heads * d_head) (heads * d_head))
    (bq₂ bk₂ bv₂ bo₂ : Vec (heads * d_head))
    (γ2₂ β2₂ : ℝ)
    (Wfc1₂ : Mat (heads * d_head) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim (heads * d_head)) (bfc2₂ : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    Vec (ic * H * W) → Vec nClasses :=
  (classifier_flat N (heads * d_head) nClasses Wcls bcls) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (fun n => layerNormForward (heads * d_head) ε γF βF
      ((Mat.unflatten v) n))) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (transformerBlock (N + 1) heads d_head mlpDim ε γ1₂ β1₂
      Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
      (Mat.unflatten v))) ∘
  (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (transformerBlock (N + 1) heads d_head mlpDim ε γ1₁ β1₁
      Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
      (Mat.unflatten v))) ∘
  (patchEmbed_flat ic H W patchSize N (heads * d_head)
    W_conv b_conv cls_token pos_embed)

/-- **Whole-net VJP for the distinct-param 2-block ViT (global).** All-smooth,
    so the only hypothesis is the LayerNorm positivity `0 < ε` — joins
    `vit_full_has_vjp`/`convnext_has_vjp` as an unconditional whole-network
    VJP holding at every input. Four `vjp_comp` steps glueing
    `patchEmbed_flat_has_vjp`, two bridged distinct-param
    `transformerBlock_has_vjp_mat` witnesses, the bridged per-token final-LN,
    and `classifier_flat_has_vjp`. -/
noncomputable def vitForward2_has_vjp
    (ic H W patchSize N mlpDim heads d_head nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (γ1₁ β1₁ : ℝ)
    (Wq₁ Wk₁ Wv₁ Wo₁ : Mat (heads * d_head) (heads * d_head))
    (bq₁ bk₁ bv₁ bo₁ : Vec (heads * d_head))
    (γ2₁ β2₁ : ℝ)
    (Wfc1₁ : Mat (heads * d_head) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim (heads * d_head)) (bfc2₁ : Vec (heads * d_head))
    (γ1₂ β1₂ : ℝ)
    (Wq₂ Wk₂ Wv₂ Wo₂ : Mat (heads * d_head) (heads * d_head))
    (bq₂ bk₂ bv₂ bo₂ : Vec (heads * d_head))
    (γ2₂ β2₂ : ℝ)
    (Wfc1₂ : Mat (heads * d_head) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim (heads * d_head)) (bfc2₂ : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses) :
    HasVJP (vitForward2 ic H W patchSize N mlpDim heads d_head nClasses
      W_conv b_conv cls_token pos_embed ε
      γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
      γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
      γF βF Wcls bcls) := by
  unfold vitForward2
  -- Stage facts: patch embed
  set PE := patchEmbed_flat ic H W patchSize N (heads * d_head)
              W_conv b_conv cls_token pos_embed with hPE
  have pe_diff := patchEmbed_flat_diff ic H W patchSize N (heads * d_head)
                    W_conv b_conv cls_token pos_embed
  have pe_vjp : HasVJP PE := patchEmbed_flat_has_vjp ic H W patchSize N
                    (heads * d_head) W_conv b_conv cls_token pos_embed
  -- Block 1 (params …₁), bridged to the flat index
  set B1 := (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (transformerBlock (N + 1) heads d_head mlpDim ε γ1₁ β1₁
      Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
      (Mat.unflatten v))) with hB1
  have b1_diff := transformerBlock_flat_diff (N + 1) heads d_head mlpDim
                    ε γ1₁ β1₁ hε Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁
                    γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
  have b1_vjp : HasVJP B1 :=
    hasVJPMat_to_hasVJP (transformerBlock_has_vjp_mat (N + 1) heads d_head mlpDim
      ε γ1₁ β1₁ hε Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁)
  have s1_vjp : HasVJP (B1 ∘ PE) := vjp_comp PE B1 pe_diff b1_diff pe_vjp b1_vjp
  have s1_diff : Differentiable ℝ (B1 ∘ PE) := b1_diff.comp pe_diff
  -- Block 2 (params …₂)
  set B2 := (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (transformerBlock (N + 1) heads d_head mlpDim ε γ1₂ β1₂
      Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
      (Mat.unflatten v))) with hB2
  have b2_diff := transformerBlock_flat_diff (N + 1) heads d_head mlpDim
                    ε γ1₂ β1₂ hε Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂
                    γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
  have b2_vjp : HasVJP B2 :=
    hasVJPMat_to_hasVJP (transformerBlock_has_vjp_mat (N + 1) heads d_head mlpDim
      ε γ1₂ β1₂ hε Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂)
  have s2_vjp : HasVJP (B2 ∘ (B1 ∘ PE)) :=
    vjp_comp (B1 ∘ PE) B2 s1_diff b2_diff s1_vjp b2_vjp
  have s2_diff : Differentiable ℝ (B2 ∘ (B1 ∘ PE)) := b2_diff.comp s1_diff
  -- Final LN (per-token, scalar γF/βF), bridged to the flat index
  set LNF := (fun v : Vec ((N + 1) * (heads * d_head)) =>
    Mat.flatten (fun n => layerNormForward (heads * d_head) ε γF βF
      ((Mat.unflatten v) n))) with hLNF
  have lnf_diff : Differentiable ℝ LNF :=
    layerNorm_per_token_flat_diff (N + 1) (heads * d_head) ε γF βF hε
  have lnf_vjp : HasVJP LNF :=
    hasVJPMat_to_hasVJP (layerNorm_per_token_has_vjp_mat (N + 1) (heads * d_head)
      ε γF βF hε)
  have s3_vjp : HasVJP (LNF ∘ (B2 ∘ (B1 ∘ PE))) :=
    vjp_comp (B2 ∘ (B1 ∘ PE)) LNF s2_diff lnf_diff s2_vjp lnf_vjp
  have s3_diff : Differentiable ℝ (LNF ∘ (B2 ∘ (B1 ∘ PE))) := lnf_diff.comp s2_diff
  -- Classifier head
  exact vjp_comp (LNF ∘ (B2 ∘ (B1 ∘ PE)))
    (classifier_flat N (heads * d_head) nClasses Wcls bcls)
    s3_diff (classifier_flat_diff N (heads * d_head) nClasses Wcls bcls)
    s3_vjp (classifier_flat_has_vjp N (heads * d_head) nClasses Wcls bcls)

/-- **Public correctness theorem for `vitForward2_has_vjp`** — the
    distinct-param 2-block ViT's backward equals the `pdiv`-contracted
    Jacobian (Jacobian-transpose applied to the cotangent), at *every*
    input. The ch10 analogue of `convnext_has_vjp_correct`. -/
theorem vitForward2_has_vjp_correct
    (ic H W patchSize N mlpDim heads d_head nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε : ℝ) (hε : 0 < ε)
    (γ1₁ β1₁ : ℝ)
    (Wq₁ Wk₁ Wv₁ Wo₁ : Mat (heads * d_head) (heads * d_head))
    (bq₁ bk₁ bv₁ bo₁ : Vec (heads * d_head))
    (γ2₁ β2₁ : ℝ)
    (Wfc1₁ : Mat (heads * d_head) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim (heads * d_head)) (bfc2₁ : Vec (heads * d_head))
    (γ1₂ β1₂ : ℝ)
    (Wq₂ Wk₂ Wv₂ Wo₂ : Mat (heads * d_head) (heads * d_head))
    (bq₂ bk₂ bv₂ bo₂ : Vec (heads * d_head))
    (γ2₂ β2₂ : ℝ)
    (Wfc1₂ : Mat (heads * d_head) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim (heads * d_head)) (bfc2₂ : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) (dy : Vec nClasses) (i : Fin (ic * H * W)) :
    (vitForward2_has_vjp ic H W patchSize N mlpDim heads d_head nClasses
      W_conv b_conv cls_token pos_embed ε hε
      γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
      γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
      γF βF Wcls bcls).backward x dy i =
      ∑ j : Fin nClasses,
        pdiv (vitForward2 ic H W patchSize N mlpDim heads d_head nClasses
          W_conv b_conv cls_token pos_embed ε
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls) x i j * dy j :=
  (vitForward2_has_vjp ic H W patchSize N mlpDim heads d_head nClasses
    W_conv b_conv cls_token pos_embed ε hε
    γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
    γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
    γF βF Wcls bcls).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § 2. heads = 1: MHSA is three matmuls + a row-softmax
-- ════════════════════════════════════════════════════════════════

/-- Sum over `Fin (1 * d)` re-indexed through `finProdFinEquiv (0, ·)` —
    the head axis of a 1-head reshape is trivial. -/
private lemma sum_fin_one_mul {M : Type*} [AddCommMonoid M] (d : Nat)
    (f : Fin (1 * d) → M) :
    (∑ k : Fin (1 * d), f k) = ∑ j : Fin d, f (finProdFinEquiv ((0 : Fin 1), j)) := by
  rw [← Equiv.sum_comp (finProdFinEquiv : Fin 1 × Fin d ≃ Fin (1 * d)) f]
  rw [Fintype.sum_prod_type]
  exact Fin.sum_univ_one _

/-- At one head, the per-head column gather is the identity modulo the
    `Fin 1 × Fin d ≃ Fin (1 * d)` reindex: contracting the gathered
    columns equals contracting the full rows. -/
private lemma matmul_one_head {Np1 d : Nat} (A B : Mat Np1 (1 * d)) :
    Mat.mul (fun r c => A r (finProdFinEquiv ((0 : Fin 1), c)))
      (Mat.transpose fun r c => B r (finProdFinEquiv ((0 : Fin 1), c))) =
    Mat.mul A (Mat.transpose B) := by
  funext i j
  unfold Mat.mul Mat.transpose
  exact (sum_fin_one_mul d fun k => A i k * B j k).symm

/-- The 1-head reshape round-trip: scattering through row 0 of the head
    axis and gathering back is the identity index. -/
private lemma fpf_one_head {d : Nat} (k : Fin (1 * d)) :
    finProdFinEquiv ((0 : Fin 1), (finProdFinEquiv.symm k).2) = k := by
  have h0 : (finProdFinEquiv.symm k).1 = (0 : Fin 1) := Fin.eq_zero _
  calc finProdFinEquiv ((0 : Fin 1), (finProdFinEquiv.symm k).2)
      = finProdFinEquiv ((finProdFinEquiv.symm k).1, (finProdFinEquiv.symm k).2) := by
        rw [h0]
    _ = k := Equiv.apply_symm_apply _ _

/-- **MHSA at heads = 1 is three matmuls + a row-softmax.** The per-head
    slice/concat plumbing of `mhsa_layer` collapses (the head axis is
    `Fin 1`), leaving exactly the ch10 graph spelling: Q/K/V per-token
    dense → `Q·Kᵀ` → `·1/√d` → row-softmax → `P·V` → output dense.
    This is the load-bearing tie for `vitFwdGraph_faithful`. -/
lemma mhsa_layer_one_head (Np1 d : Nat)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (X : Mat Np1 (1 * d)) :
    mhsa_layer Np1 1 d Wq Wk Wv Wo bq bk bv bo X =
      fun n => dense Wo bo
        (Mat.mul
          (rowSoftmax (fun i j => sdpa_scale d *
            Mat.mul (fun r c => dense Wq bq (X r) c)
              (Mat.transpose (fun r c => dense Wk bk (X r) c)) i j))
          (fun r c => dense Wv bv (X r) c) n) := by
  funext n j
  unfold mhsa_layer sdpa sdpa_scale dense
  dsimp only
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  have h0 : (finProdFinEquiv.symm k).1 = (0 : Fin 1) := Fin.eq_zero _
  rw [h0]
  -- Factor the beta-expanded Q/K gathers so `matmul_one_head` applies
  -- (the `have` type is the goal's syntactic form; the proof term is the
  -- factored form — they are beta-defeq).
  have hQK : Mat.mul
      (fun (n' : Fin Np1) (j' : Fin d) =>
        (∑ k' : Fin (1 * d), X n' k' * Wq k' (finProdFinEquiv ((0 : Fin 1), j'))) +
          bq (finProdFinEquiv ((0 : Fin 1), j')))
      (Mat.transpose fun (n' : Fin Np1) (j' : Fin d) =>
        (∑ k' : Fin (1 * d), X n' k' * Wk k' (finProdFinEquiv ((0 : Fin 1), j'))) +
          bk (finProdFinEquiv ((0 : Fin 1), j'))) =
    Mat.mul
      (fun (r : Fin Np1) (c : Fin (1 * d)) =>
        (∑ i : Fin (1 * d), X r i * Wq i c) + bq c)
      (Mat.transpose fun (r : Fin Np1) (c : Fin (1 * d)) =>
        (∑ i : Fin (1 * d), X r i * Wk i c) + bk c) :=
    matmul_one_head
      (fun (r : Fin Np1) (c : Fin (1 * d)) =>
        (∑ i : Fin (1 * d), X r i * Wq i c) + bq c)
      (fun (r : Fin Np1) (c : Fin (1 * d)) =>
        (∑ i : Fin (1 * d), X r i * Wk i c) + bk c)
  simp only [hQK]
  unfold Mat.mul
  dsimp only
  simp only [fpf_one_head]

-- ════════════════════════════════════════════════════════════════
-- § 3. The spelled block (Mat level) — the graph's op sequence
-- ════════════════════════════════════════════════════════════════

/-- The ch10 spelled pre-norm transformer block at heads = 1 (Mat level) —
    the exact op sequence `vitBlockGraph` denotes: LN₁ → Q/K/V per-token
    dense → `Q·Kᵀ` → `·1/√d` → row-softmax → `P·V` → output dense → +res →
    LN₂ → fc1 → GELU → fc2 → +res. Equals `transformerBlock` at one head
    (`vitBlockSpelled_eq`). -/
noncomputable def vitBlockSpelled (Np1 d mlpDim : Nat) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (1 * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d)) (bfc2 : Vec (1 * d))
    (X : Mat Np1 (1 * d)) : Mat Np1 (1 * d) :=
  let ln1 : Mat Np1 (1 * d) := fun r => layerNormForward (1 * d) ε γ1 β1 (X r)
  let Q : Mat Np1 (1 * d) := fun r => dense Wq bq (ln1 r)
  let K : Mat Np1 (1 * d) := fun r => dense Wk bk (ln1 r)
  let V : Mat Np1 (1 * d) := fun r => dense Wv bv (ln1 r)
  let P : Mat Np1 Np1 :=
    rowSoftmax (fun i j => sdpa_scale d * Mat.mul Q (Mat.transpose K) i j)
  let O : Mat Np1 (1 * d) := fun r => dense Wo bo (Mat.mul P V r)
  let h : Mat Np1 (1 * d) := fun r s => X r s + O r s
  let ln2 : Mat Np1 (1 * d) := fun r => layerNormForward (1 * d) ε γ2 β2 (h r)
  let m1 : Mat Np1 mlpDim := fun r => dense Wfc1 bfc1 (ln2 r)
  let g : Mat Np1 mlpDim := fun r => gelu mlpDim (m1 r)
  let m2 : Mat Np1 (1 * d) := fun r => dense Wfc2 bfc2 (g r)
  fun r s => h r s + m2 r s

/-- **The spelled block IS `transformerBlock` at one head.** The
    sublayer/residual structure matches definitionally once
    `mhsa_layer_one_head` collapses the per-head plumbing. -/
lemma vitBlockSpelled_eq (Np1 d mlpDim : Nat) (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (1 * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d)) (bfc2 : Vec (1 * d))
    (X : Mat Np1 (1 * d)) :
    vitBlockSpelled Np1 d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X =
      transformerBlock Np1 1 d mlpDim ε γ1 β1 Wq Wk Wv Wo bq bk bv bo
        γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X := by
  unfold transformerBlock transformerMlpSublayer transformerAttnSublayer
         transformerMlp biPathMat
  simp only [Function.comp_apply]
  rw [mhsa_layer_one_head]
  rfl

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § 4. The ViT forward graph (SHlo tokens) + faithfulness
-- ════════════════════════════════════════════════════════════════

/-- One spelled pre-norm transformer block over the ch10 tokens (heads = 1):
    `lnRowF` → Q/K/V `denseRowF` → `matmulF`(Q, `transposeF` K) → `scaleF` →
    `softmaxRowF` → `matmulF`(P, V) → output `denseRowF` → `addV` residual →
    `lnRowF` → fc1 → `geluF` → fc2 → `addV` residual. Generic `D`; the
    faithfulness theorem instantiates `D := 1 * d`, `s := sdpa_scale d`. -/
def vitBlockGraph {Np1 D mlpDim : Nat} (pfx epsStr sStr : String)
    (ε s : ℝ) (γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat D D) (bq bk bv bo : Vec D)
    (γ2 β2 : ℝ)
    (Wfc1 : Mat D mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim D) (bfc2 : Vec D)
    (x : SHlo (Np1 * D)) : SHlo (Np1 * D) :=
  let ln1 := SHlo.lnRowF s!"%{pfx}g1" s!"%{pfx}bt1" epsStr ε γ1 β1 x
  let q := SHlo.denseRowF s!"%{pfx}Wq" s!"%{pfx}bq" Wq bq ln1
  let k := SHlo.denseRowF s!"%{pfx}Wk" s!"%{pfx}bk" Wk bk ln1
  let v := SHlo.denseRowF s!"%{pfx}Wv" s!"%{pfx}bv" Wv bv ln1
  let p := SHlo.softmaxRowF (SHlo.scaleF sStr s (SHlo.matmulF q (SHlo.transposeF k)))
  let o := SHlo.denseRowF s!"%{pfx}Wo" s!"%{pfx}bo" Wo bo (SHlo.matmulF p v)
  let h := SHlo.addV x o
  let ln2 := SHlo.lnRowF s!"%{pfx}g2" s!"%{pfx}bt2" epsStr ε γ2 β2 h
  let m2 := SHlo.denseRowF s!"%{pfx}Wfc2" s!"%{pfx}bfc2" Wfc2 bfc2
    (SHlo.geluF (SHlo.denseRowF s!"%{pfx}Wfc1" s!"%{pfx}bfc1" Wfc1 bfc1 ln2))
  SHlo.addV h m2

/-! ### Flat ↔ Mat commutation bridges

Each ch10 den helper applied to a `Mat.flatten` is the flatten of the
corresponding Mat-level op (the `Mat.unflatten_flatten` round-trip
cancels); the pointwise ops (`scaleF`/`geluF`/`addV`) commute with
flattening definitionally. -/

private lemma rowLNFlat_flat {m n : Nat} (ε γ β : ℝ) (A : Mat m n) :
    rowLNFlat m n ε γ β (Mat.flatten A) =
      Mat.flatten (fun r => layerNormForward n ε γ β (A r)) := by
  unfold rowLNFlat layerNormForward
  rw [Mat.unflatten_flatten]

private lemma rowDenseFlat_flat {N a c : Nat} (W : Mat a c) (b : Vec c) (A : Mat N a) :
    rowDenseFlat N a c W b (Mat.flatten A) = Mat.flatten (fun r => dense W b (A r)) := by
  unfold rowDenseFlat
  rw [Mat.unflatten_flatten]

private lemma matMulFlat_flat {m k n : Nat} (A : Mat m k) (B : Mat k n) :
    matMulFlat m k n (Mat.flatten A) (Mat.flatten B) = Mat.flatten (Mat.mul A B) := by
  unfold matMulFlat
  rw [Mat.unflatten_flatten, Mat.unflatten_flatten]

private lemma transposeFlat_flat {m n : Nat} (A : Mat m n) :
    transposeFlat m n (Mat.flatten A) = Mat.flatten (Mat.transpose A) := by
  unfold transposeFlat
  rw [Mat.unflatten_flatten]

private lemma rowSoftmaxFlat_flat {m n : Nat} (A : Mat m n) :
    rowSoftmaxFlat m n (Mat.flatten A) = Mat.flatten (rowSoftmax A) := by
  unfold rowSoftmaxFlat rowSoftmax
  rw [Mat.unflatten_flatten]

private lemma scale_flat {m n : Nat} (s : ℝ) (A : Mat m n) :
    (fun i => s * Mat.flatten A i) = Mat.flatten (fun r c => s * A r c) := rfl

private lemma gelu_flat {m n : Nat} (A : Mat m n) :
    gelu (m * n) (Mat.flatten A) = Mat.flatten (fun r => gelu n (A r)) := rfl

private lemma add_flat_pt {m n : Nat} (A B : Mat m n) (j : Fin (m * n)) :
    Mat.flatten A j + Mat.flatten B j = Mat.flatten (fun r s => A r s + B r s) j := rfl

/-- **Block-graph denotation** (generalized over the input's Mat form):
    the spelled token block denotes `Mat.flatten ∘ vitBlockSpelled ∘
    Mat.unflatten` at `D := 1 * d`, `s := sdpa_scale d`. -/
private lemma vitBlockGraph_den_aux {Np1 d mlpDim : Nat} (pfx epsStr sStr : String)
    (ε γ1 β1 : ℝ)
    (Wq Wk Wv Wo : Mat (1 * d) (1 * d)) (bq bk bv bo : Vec (1 * d))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (1 * d) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (1 * d)) (bfc2 : Vec (1 * d))
    (e : SHlo (Np1 * (1 * d))) (A : Mat Np1 (1 * d)) (hA : den e = Mat.flatten A) :
    den (vitBlockGraph pfx epsStr sStr ε (sdpa_scale d) γ1 β1
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 e) =
      Mat.flatten (vitBlockSpelled Np1 d mlpDim ε γ1 β1
          Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 A) := by
  simp only [vitBlockGraph, lnRowF_faithful, denseRowF_faithful, matmulF_faithful,
             transposeF_faithful, scaleF_faithful, softmaxRowF_faithful, geluF_faithful,
             den_addV, hA]
  simp only [rowLNFlat_flat, rowDenseFlat_flat, transposeFlat_flat, matMulFlat_flat,
             scale_flat, rowSoftmaxFlat_flat, gelu_flat, add_flat_pt]
  rfl

/-- Whole **ViT forward** graph (the ch10 representative, peer of
    `convNextFwdGraph`): patch embed (stride-P conv + CLS + pos-embed) →
    2 spelled transformer blocks (distinct params) → final per-token LN →
    CLS slice → dense head. Generic `D`/`s`; faithful at `D := 1 * d`,
    `s := sdpa_scale d` (heads = 1). -/
def vitFwdGraph {ic H W P N D mlpDim nClasses : Nat}
    (epsStr sStr : String) (ε s : ℝ)
    (Wc : Kernel4 D ic P P) (bc : Vec D) (cls : Vec D) (pos : Mat (N + 1) D)
    (γ1₁ β1₁ : ℝ) (Wq₁ Wk₁ Wv₁ Wo₁ : Mat D D) (bq₁ bk₁ bv₁ bo₁ : Vec D)
    (γ2₁ β2₁ : ℝ) (Wfc1₁ : Mat D mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim D) (bfc2₁ : Vec D)
    (γ1₂ β1₂ : ℝ) (Wq₂ Wk₂ Wv₂ Wo₂ : Mat D D) (bq₂ bk₂ bv₂ bo₂ : Vec D)
    (γ2₂ β2₂ : ℝ) (Wfc1₂ : Mat D mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim D) (bfc2₂ : Vec D)
    (γF βF : ℝ) (Wcls : Mat D nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) : SHlo nClasses :=
  let embed : SHlo ((N + 1) * D) :=
    .patchEmbedF "%Wp" "%bp" "%cls" "%pos" Wc bc cls pos (.operand "%x" x)
  let b1 := vitBlockGraph "b1_" epsStr sStr ε s γ1₁ β1₁
    Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁ embed
  let b2 := vitBlockGraph "b2_" epsStr sStr ε s γ1₂ β1₂
    Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂ b1
  let fl := SHlo.lnRowF "%gF" "%btF" epsStr ε γF βF b2
  denseF "%Wcls" "%bcls" Wcls bcls (.clsSliceF fl)

/-- **ViT forward faithfulness** — the ch10 close's Item A apex: the
    representative forward graph denotes the proven distinct-param 2-block
    `vitForward2` at one head (`heads := 1`, `D := 1 * d`,
    `s := sdpa_scale d`). The ViT analogue of `convNextFwdGraph_faithful`:
    per-block `vitBlockGraph_den_aux` + `vitBlockSpelled_eq`
    (`mhsa_layer_one_head` under the hood), then the patch-embed / CLS-slice
    den helpers are the proven Attention forms verbatim. -/
theorem vitFwdGraph_faithful
    (ic H W patchSize N d mlpDim nClasses : Nat)
    (epsStr sStr : String)
    (Wc : Kernel4 (1 * d) ic patchSize patchSize) (bc : Vec (1 * d))
    (cls : Vec (1 * d)) (pos : Mat (N + 1) (1 * d))
    (ε : ℝ)
    (γ1₁ β1₁ : ℝ) (Wq₁ Wk₁ Wv₁ Wo₁ : Mat (1 * d) (1 * d))
    (bq₁ bk₁ bv₁ bo₁ : Vec (1 * d))
    (γ2₁ β2₁ : ℝ) (Wfc1₁ : Mat (1 * d) mlpDim) (bfc1₁ : Vec mlpDim)
    (Wfc2₁ : Mat mlpDim (1 * d)) (bfc2₁ : Vec (1 * d))
    (γ1₂ β1₂ : ℝ) (Wq₂ Wk₂ Wv₂ Wo₂ : Mat (1 * d) (1 * d))
    (bq₂ bk₂ bv₂ bo₂ : Vec (1 * d))
    (γ2₂ β2₂ : ℝ) (Wfc1₂ : Mat (1 * d) mlpDim) (bfc1₂ : Vec mlpDim)
    (Wfc2₂ : Mat mlpDim (1 * d)) (bfc2₂ : Vec (1 * d))
    (γF βF : ℝ) (Wcls : Mat (1 * d) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) :
    den (vitFwdGraph epsStr sStr ε (sdpa_scale d) Wc bc cls pos
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls x)
      = vitForward2 ic H W patchSize N mlpDim 1 d nClasses Wc bc cls pos ε
          γ1₁ β1₁ Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
          γ1₂ β1₂ Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
          γF βF Wcls bcls x := by
  -- Stage 0: the patch embedding (the den helper IS `patchEmbed_flat`).
  have h0 : den (SHlo.patchEmbedF (P := patchSize) "%Wp" "%bp" "%cls" "%pos"
        Wc bc cls pos (.operand "%x" x))
      = Mat.flatten (Mat.unflatten
          (patchEmbed_flat ic H W patchSize N (1 * d) Wc bc cls pos x)) := by
    simp only [patchEmbedF_faithful, den_operand]
    rw [Mat.flatten_unflatten]
    rfl
  -- Stage 1/2: the two blocks, chained through their Mat forms.
  have h1 := vitBlockGraph_den_aux "b1_" epsStr sStr ε γ1₁ β1₁
    Wq₁ Wk₁ Wv₁ Wo₁ bq₁ bk₁ bv₁ bo₁ γ2₁ β2₁ Wfc1₁ bfc1₁ Wfc2₁ bfc2₁
    _ _ h0
  have h2 := vitBlockGraph_den_aux "b2_" epsStr sStr ε γ1₂ β1₂
    Wq₂ Wk₂ Wv₂ Wo₂ bq₂ bk₂ bv₂ bo₂ γ2₂ β2₂ Wfc1₂ bfc1₂ Wfc2₂ bfc2₂
    _ _ h1
  -- Assemble: head ∘ final-LN over the stage-2 denotation.
  simp only [vitFwdGraph, denseF_faithful, clsSliceF_faithful, lnRowF_faithful, h2]
  simp only [rowLNFlat_flat, vitBlockSpelled_eq]
  -- The right-hand side is the same composition, written through
  -- `Mat.unflatten ∘ Mat.flatten` round-trips.
  unfold vitForward2 classifier_flat
  simp only [Function.comp_apply, Mat.unflatten_flatten]
  rfl

end Proofs.StableHLO
