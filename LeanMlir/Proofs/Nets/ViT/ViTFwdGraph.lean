import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Architectures.Attention

/-!
# ViT forward-graph pieces — the heads = 1 MHSA reduction and the flat ↔ Mat bridges

The ch10 token layer's ViT den helpers (`StableHLO.lean`) are local re-spellings; this file
ties them back to the proven Attention forms, for the forward-graph faithfulness proofs
(`ViTVecLN`, `ViTMultiHead`, `ViTDepthK`) to build on:

1. **`mhsa_layer_one_head`** — at heads = 1, MHSA is three matmuls + a row-softmax.
2. **The flat ↔ Mat bridges** — each den helper applied to a `Mat.flatten` is the flatten of
   the Mat-level op — and `patchEmbedF_x_den`, the patch-embed stage every ViT forward-graph
   faithfulness proof starts from.
-/

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § 1. heads = 1: MHSA is three matmuls + a row-softmax
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
    `ViTVecLN`'s spelled-block tie (`vitBlockSpelledV_eq`) rewrites with it. -/
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

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § 2. Flat ↔ Mat bridges + the patch-embed stage
-- ════════════════════════════════════════════════════════════════

/-! ### Flat ↔ Mat commutation bridges

Each ch10 den helper applied to a `Mat.flatten` is the flatten of the
corresponding Mat-level op (the `Mat.unflatten_flatten` round-trip
cancels); the pointwise ops (`scaleF`/`geluF`/`addV`) commute with
flattening definitionally. Public — `ViTChainClose` reuses them to tie
the matmul-spelled SDPA backward to the proven closed forms. -/

lemma rowLNFlat_flat {m n : Nat} (ε γ β : ℝ) (A : Mat m n) :
    rowLNFlat m n ε γ β (Mat.flatten A) =
      Mat.flatten (fun r => layerNormForward n ε γ β (A r)) := by
  unfold rowLNFlat layerNormForward
  rw [Mat.unflatten_flatten]

lemma rowDenseFlat_flat {N a c : Nat} (W : Mat a c) (b : Vec c) (A : Mat N a) :
    rowDenseFlat N a c W b (Mat.flatten A) = Mat.flatten (fun r => dense W b (A r)) := by
  unfold rowDenseFlat
  rw [Mat.unflatten_flatten]

lemma matMulFlat_flat {m k n : Nat} (A : Mat m k) (B : Mat k n) :
    matMulFlat m k n (Mat.flatten A) (Mat.flatten B) = Mat.flatten (Mat.mul A B) := by
  unfold matMulFlat
  rw [Mat.unflatten_flatten, Mat.unflatten_flatten]

lemma transposeFlat_flat {m n : Nat} (A : Mat m n) :
    transposeFlat m n (Mat.flatten A) = Mat.flatten (Mat.transpose A) := by
  unfold transposeFlat
  rw [Mat.unflatten_flatten]

lemma rowSoftmaxFlat_flat {m n : Nat} (A : Mat m n) :
    rowSoftmaxFlat m n (Mat.flatten A) = Mat.flatten (rowSoftmax A) := by
  unfold rowSoftmaxFlat rowSoftmax
  rw [Mat.unflatten_flatten]

lemma scale_flat {m n : Nat} (s : ℝ) (A : Mat m n) :
    (fun i => s * Mat.flatten A i) = Mat.flatten (fun r c => s * A r c) := rfl

lemma gelu_flat {m n : Nat} (A : Mat m n) :
    gelu (m * n) (Mat.flatten A) = Mat.flatten (fun r => gelu n (A r)) := rfl

lemma add_flat_pt {m n : Nat} (A B : Mat m n) (j : Fin (m * n)) :
    Mat.flatten A j + Mat.flatten B j = Mat.flatten (fun r s => A r s + B r s) j := rfl

/-- The patch-embed stage over the image operand `%x`, in the `Mat.flatten (Mat.unflatten …)` form
    the block `den_aux` lemmas take — stage 0 of every ViT forward-graph faithfulness proof. -/
lemma patchEmbedF_x_den (ic H W patchSize N D : Nat) (Wc : Kernel4 D ic patchSize patchSize)
    (bc cls : Vec D) (pos : Mat (N + 1) D) (x : Vec (ic * H * W)) :
    den (SHlo.patchEmbedF (P := patchSize) "%Wp" "%bp" "%cls" "%pos"
        Wc bc cls pos (.operand "%x" x))
      = Mat.flatten (Mat.unflatten (patchEmbed_flat ic H W patchSize N D Wc bc cls pos x)) := by
  rw [Mat.flatten_unflatten]
  rfl

end Proofs.StableHLO
