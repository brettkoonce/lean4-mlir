import LeanMlir.Proofs.Nets.ViT.ViTFaithfulPoC

/-! # T3 §1 fold for ViT-Tiny at the UN-FUSED gradient — the Adam artifact's op set

`ViTFaithfulPoC.lean` makes every parameter output of the SGD-inline `vit_train_step.mlir`
`den`-faithful at the fused `θ − lr·g` ops. `ViTRender.lean` renders one backward traversal with
two endings under an `adam : Bool` (its own comment: *"one backward traversal, two endings, which is
what keeps `vit_train_step.mlir` byte-identical while `vit_adam_train_step.mlir` gets its
gradients"*), and every non-SGD artifact of this net — `vit_adam_train_step.mlir`, and the ImageNet
`vitin_adamdp128x4wxclipdrop` whose accuracy the book quotes — is on the `adam := true` branch,
which emits the RAW gradient and hands it to the AdamW triple. This file is the fold at those nodes.

⭐ **One lemma per op kind certifies every optimizer tail at once**: AdamW, the clip and
weight-decay variants, the 4× gradient accumulation and plain SGD all consume the same gradient
node. The fusion itself is `rfl` (`StableHLO.lean`'s `*Sgd_eq_grad` family — the transformer peers
of that set exist precisely because "the ViT family §2a left fused, which is why
`vit_adam_train_step` had no certified render until these existed"), so nothing here is new
mathematics: each proof is the fused lemma's proof with the `θ − lr·` wrapper dropped, which means
each lands directly on the raw grad bridge the wrapped cert was built from.

## The op table of `vit_adam_train_step.mlir`

| emitted node | lemma | fused peer it un-fuses |
|---|---|---|
| `veclnGammaGrad` (25 LN γ: LN1/LN2 × 12 + final) | `veclnGammaGrad_den` | `ViTPoC.veclnGammaSgd_den` |
| `rowDenseBiasGrad` (25 LN β) | `rowDenseBiasGrad_den_lnbeta` | `ViTPoC.rowDenseBiasSgd_den_lnbeta` |
| `rowDenseWeightGrad` (Wq/Wk/Wv/Wo/Wfc1/Wfc2 × 12) | `rowDenseWeightGrad_den` | `ViTPoC.rowDenseWeightSgd_den` |
| `rowDenseBiasGrad` (bq/bk/bv/bo/bfc1/bfc2 × 12) | `rowDenseBiasGrad_den` | `ViTPoC.rowDenseBiasSgd_den` |
| `patchEmbedWeightGrad` / `patchEmbedBiasGrad` | `patchEmbedWeightGrad_den` / `patchEmbedBiasGrad_den` | `ViTPoC.patchEmbedWeightSgd_den` / `patchEmbedBiasSgd_den` |
| `posEmbedGrad` | `posEmbedGrad_den` | `ViTPoC.posEmbedSgd_den` |
| `denseBiasGradB` at `N = 1` (the CLS token) | `clsGrad_den` | `ViTTiePoC.vit_cls_den` |
| `weightGrad` / `biasGrad` (the classifier) | `headWGrad_den` / `headBGrad_den` | `ViTPoC.headW_den` / `headB_den` |

⭐ **The same op serves two different forwards, and both readings survive un-fusing.**
`rowDenseBiasGrad` is `Σ_tokens dy` whether the parameter it belongs to is a dense bias or a
LayerNorm β, so it appears twice below against two different certified Jacobians — as it does in
the fused file. That is not an ambiguity: the tie (`ViTTiePoC.lean`) is what says which forward a
given SSA name's operand came from.

⭐ **The LayerNorm form is the VECTOR one** (`γ β : Vec D`), which is what the shipped
`vitForwardKV` runs; the scalar-affine spelling this cone was caught on three times is nowhere in
this file.

## Honest residual
* Every lemma is `∀ cot`, so each holds at the actual backward-chain cotangent without naming it.
  Pinning them is the §1a tie (`ViTTiePoC.lean`, 200 params); re-pointing that capstone at these
  nodes needs the smoothed-target loss cotangent, scoped with r34's batched tie.
* `vitin_adamdp128x4*` is four replicas: the all-reduce is emitted text outside the AST, so these
  lemmas are about the per-replica gradient node, and the 4× accumulation is `momVNextF` at
  `(μ := akeep)`, a separately certified tail.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.ViTPoCG

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The vector LayerNorm — 25 γ sites and 25 β sites
-- ════════════════════════════════════════════════════════════════

/-- **Vector-LN γ GRADIENT denotes the certified γ gradient** `Σ_tokens dy⊙x̂`. All 25 sites
    (LN1/LN2 × 12 blocks + the final LN). -/
theorem veclnGammaGrad_den {N D : Nat} (xN epsStr cotN : String)
    (ε : ℝ) (βv : Vec D) (x : Vec (N * D)) (γ : Vec D) (dy : Vec (N * D)) (k : Fin D) :
    den (SHlo.veclnGammaGrad xN epsStr ε x (.operand cotN dy)) k
      = ∑ o : Fin (N * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε gv βv (Mat.unflatten x r))) γ k o * dy o := by
  simp only [den]
  exact vit_veclnGamma_grad_bridge ε βv γ (Mat.unflatten x) dy k

/-- **The SAME row-reduce op, certified against the vector-LN β forward.** The LN β gradient is
    `Σ_tokens dy` — the identical reduce to a dense bias — so `rowDenseBiasGrad` denotes this too.
    All 25 β sites. -/
theorem rowDenseBiasGrad_den_lnbeta {N D : Nat} (cotN : String)
    (ε : ℝ) (γv : Vec D) (X : Mat N D) (β : Vec D) (dy : Vec (N * D)) (i : Fin D) :
    den (SHlo.rowDenseBiasGrad (N := N) (c := D) (.operand cotN dy)) i
      = ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D => Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o * dy o := by
  simp only [den]
  exact vit_veclnBeta_grad_bridge ε γv β X dy i

-- ════════════════════════════════════════════════════════════════
-- § The per-token denses — Wq/Wk/Wv/Wo/Wfc1/Wfc2 and their biases
-- ════════════════════════════════════════════════════════════════

/-- **Per-token dense weight GRADIENT denotes the certified `Σ_tokens x⊗dy`.** All six denses in
    each of the 12 blocks. -/
theorem rowDenseWeightGrad_den {N a c : Nat} (xN cotN : String)
    (bb : Vec c) (x : Vec (N * a)) (W : Mat a c) (dy : Vec (N * c)) (i : Fin a) (j : Fin c) :
    den (SHlo.rowDenseWeightGrad xN x (.operand cotN dy)) (finProdFinEquiv (i, j))
      = ∑ o : Fin (N * c),
          pdiv (fun v : Vec (a * c) =>
                  Mat.flatten (fun r => dense (Mat.unflatten v) bb (Mat.unflatten x r)))
               (Mat.flatten W) (finProdFinEquiv (i, j)) o * dy o := by
  simp only [den, Mat.flatten, Equiv.symm_apply_apply]
  exact vit_rowDenseW_grad_bridge bb (Mat.unflatten x) W dy i j

/-- **Per-token dense bias GRADIENT denotes the certified `Σ_tokens dy`.** bq/bk/bv/bo/bfc1/bfc2. -/
theorem rowDenseBiasGrad_den {N a c : Nat} (cotN : String)
    (W : Mat a c) (X : Mat N a) (b : Vec c) (dy : Vec (N * c)) (i : Fin c) :
    den (SHlo.rowDenseBiasGrad (N := N) (c := c) (.operand cotN dy)) i
      = ∑ o : Fin (N * c),
          pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r))) b i o * dy o := by
  simp only [den]
  exact vit_rowDenseb_grad_bridge W X b dy i

-- ════════════════════════════════════════════════════════════════
-- § The patch embedding — conv weight/bias, the CLS token, the positional table
-- ════════════════════════════════════════════════════════════════

/-- **Patch-embed conv weight GRADIENT denotes the certified patchify weight gradient.** ViT's
    analogue of ConvNeXt's stem 4×4/s4 weight — but with a VJP cert, so it ties. -/
theorem patchEmbedWeightGrad_den {ic H W P N D : Nat} (xN cotN : String)
    (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W)) (Wp : Kernel4 D ic P P)
    (dy : Vec ((N + 1) * D)) (d : Fin D) (c : Fin ic) (kh kw : Fin P) :
    den (SHlo.patchEmbedWeightGrad (N := N) xN img (.operand cotN dy))
        (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw))
      = ∑ o : Fin ((N + 1) * D),
          pdiv (fun v : Vec (D * ic * P * P) =>
                  patchEmbed_flat ic H W P N D (Kernel4.unflatten v) bc cls pos img)
            (Kernel4.flatten Wp)
            (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o * dy o := by
  simp only [den, patchEmbedWeightGradFlat, Kernel4.flatten, Equiv.symm_apply_apply]
  exact vit_patchW_grad_bridge Wp bc cls pos img dy d c kh kw

/-- **Patch-embed conv bias GRADIENT denotes the certified bias gradient** (`Σ_patches dy`, the
    CLS row excluded). -/
theorem patchEmbedBiasGrad_den {ic H W P N D : Nat} (cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W))
    (dy : Vec ((N + 1) * D)) (i : Fin D) :
    den (SHlo.patchEmbedBiasGrad (N := N) (.operand cotN dy)) i
      = ∑ o : Fin ((N + 1) * D),
          pdiv (fun b' : Vec D => patchEmbed_flat ic H W P N D Wc b' cls pos img) bc i o * dy o := by
  simp only [den]
  exact vit_patchb_grad_bridge Wc bc cls pos img dy i

/-- **Positional-embed GRADIENT denotes the certified gradient** — the cotangent itself, since the
    positional table is added to every token and its Jacobian is the identity. -/
theorem posEmbedGrad_den {ic H W P N D : Nat} (cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W))
    (dy : Vec ((N + 1) * D)) (i : Fin ((N + 1) * D)) :
    den (SHlo.posEmbedGrad (.operand cotN dy)) i
      = ∑ j : Fin ((N + 1) * D),
          pdiv (fun p : Vec ((N + 1) * D) =>
                  patchEmbed_flat ic H W P N D Wc bc cls (Mat.unflatten p) img)
            (Mat.flatten pos) i j * dy j := by
  simp only [den]
  simp_rw [pdiv_patchEmbed_pos]
  rw [Finset.sum_eq_single i
      (fun j _ hne => by rw [if_neg (Ne.symm (Ne.symm hne).symm), zero_mul])
      (fun h => absurd (Finset.mem_univ i) h)]
  rw [if_pos rfl, one_mul]

/-- **CLS-token GRADIENT denotes the certified gradient.** The render slices row 0 of the embed
    cotangent (`clsSliceF`) and then reduces it as a `[1, D]` batch, so the op is
    `denseBiasGradB` at `N = 1` and its `den` IS `cls_token_grad`. ⚠ Stated at the committed
    ViT-Tiny dims rather than generically, for the reason `ViTTiePoC.vit_cls_den` is: the operand's
    type is `Vec (1 * D)`, which reduces to `Vec D` only at a literal `D`.

    ⭐ The fused peer's proof ends in `vit_render_cls_certified`, whose statement carries the
    `θ − lr·` wrapper and has no un-wrapped twin; instantiating it at `lr = 1` un-fuses it, which
    is the same content as a `*Sgd_eq_grad` `rfl` read backwards. -/
theorem clsGrad_den (cotN : String)
    (Wc : Kernel4 192 3 16 16) (bc cls : Vec 192) (pos : Mat 197 192)
    (img : Vec (3 * 224 * 224)) (dyEmbed : Vec (197 * 192)) (i : Fin 192) :
    den (SHlo.denseBiasGradB (N := 1) (c := 192)
            (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i
      = ∑ j : Fin (197 * 192),
          pdiv (fun cl : Vec 192 =>
                  patchEmbed_flat 3 224 224 16 196 192 Wc bc cl pos img) cls i j * dyEmbed j := by
  have hstep : den (SHlo.denseBiasGradB (N := 1) (c := 192)
            (.operand cotN (clsSliceFlat 196 192 dyEmbed))) i = cls_token_grad dyEmbed i := by
    simp only [den, batchSlice, cls_token_grad]; rw [Fin.sum_univ_one]; rfl
  rw [hstep]
  have h := vit_render_cls_certified Wc bc cls pos img dyEmbed 1 i
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The classifier head — the single CLS vector, no row lift
-- ════════════════════════════════════════════════════════════════

/-- **Classifier weight GRADIENT denotes the certified outer product.** -/
theorem headWGrad_den {D nC : Nat} (aN cotN : String)
    (a : Vec D) (Wc : Mat D nC) (bc : Vec nC) (cot : Vec nC) (i : Fin D) (j : Fin nC) :
    den (SHlo.weightGrad aN a (.operand cotN cot)) (finProdFinEquiv (i, j))
      = ∑ k : Fin nC,
          pdiv (fun v : Vec (D * nC) => dense (Mat.unflatten v) bc a) (Mat.flatten Wc)
               (finProdFinEquiv (i, j)) k * cot k := by
  simp only [den, Mat.flatten, Equiv.symm_apply_apply]
  exact dense_weight_grad_correct Wc bc a cot i j

/-- **Classifier bias GRADIENT denotes the certified cotangent.** -/
theorem headBGrad_den {D nC : Nat} (cotN : String)
    (Wc : Mat D nC) (a : Vec D) (bc : Vec nC) (cot : Vec nC) (i : Fin nC) :
    den (SHlo.biasGrad (.operand cotN cot)) i
      = ∑ j : Fin nC, pdiv (fun b' : Vec nC => dense Wc b' a) bc i j * cot j := by
  simp only [den]
  exact dense_bias_grad_correct Wc bc a cot i

end Proofs.ViTPoCG
