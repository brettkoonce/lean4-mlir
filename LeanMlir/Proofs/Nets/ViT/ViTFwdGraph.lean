import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Architectures.Attention

/-!
# ViT forward-graph pieces — the flat ↔ Mat bridges

The ch10 token layer's ViT den helpers (`StableHLO.lean`) are local re-spellings; this file
ties them back to the proven Attention forms for the forward-graph faithfulness proofs
(`ViTMultiHead`, `ViTDepthK`): each den helper applied to a `Mat.flatten` is the flatten of
the Mat-level op, and `patchEmbedF_x_den` is the patch-embed stage every ViT forward-graph
faithfulness proof starts from.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § Flat ↔ Mat bridges + the patch-embed stage
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
