import LeanMlir.Proofs.StableHLO
import LeanMlir.Proofs.Architectures.ViTClose
import LeanMlir.Proofs.Architectures.ViTVecLN
import LeanMlir.Proofs.Architectures.Cifar8FaithfulPoC

/-! # ViT-Tiny §1 fold — each emitted param-SGD op `den`otes the certified loss-descent step

The ViT peer of `MobileNetV2FaithfulPoC`/`ConvNeXtFaithfulPoC`/`EfficientNetFaithfulPoC`: for every
param-SGD op the `vitTrainStepRenderV` renderer emits, prove `den(op) = θ − lr·(certified Jacobian ·
cotangent)`. Each is a one-or-few-line delegation to the already-proven render certs in
`ViTVecLN` (vector-[D] LN γ/β) and `ViTClose` (rowwise dense W/b, patch conv W/b, cls, pos); the
classifier head reuses the M2 `Cifar8PoC.dense{W,B}_den`. Together these cover EVERY parameter family
of the depth-12 ViT-Tiny train step (200 params), so the §1a tie (`ViTTiePoC`) can thread them at the
real backward chain cotangents.

The op `den`s and the cert LHSs line up by construction (the core ops were built to denote exactly
these grads): `veclnGammaSgd`→`vit_render_veclngamma_certified`, `rowDenseWeightSgd`→
`vit_render_rowdenseW_certified`, `rowDenseBiasSgd`→`vit_render_rowdenseb_certified` (dense bias) and
`vit_render_veclnbeta_certified` (LN β — same op, different forward in the pdiv), `patchEmbedWeightSgd`
→`vit_render_patchW_certified`, `patchEmbedBiasSgd`→`vit_render_patchb_certified`, `posEmbedSgd`→
`vit_render_pos_certified`, cls (`clsSliceF`→`denseBiasSgdB`)→`vit_render_cls_certified`. -/

namespace Proofs.ViTPoC

open scoped BigOperators
open Proofs Proofs.StableHLO

/-- **Vector-LN γ op denotes the certified step.** `den(veclnGammaSgd)` = `γ − lr·(Σ_tokens dy·x̂)`,
    the certified ∂(rowwise vector-LN)/∂γ contraction. Covers all 25 LN-γ sites (LN1/LN2 × 12 + final).
    One-line delegation to `vit_render_veclngamma_certified` (the den's sum IS `vecLN_grad_gamma`). -/
theorem veclnGammaSgd_den {N D : Nat} (gN xN epsStr lrStr cotN : String)
    (ε : ℝ) (βv : Vec D) (x : Vec (N * D)) (γ : Vec D) (dy : Vec (N * D)) (lr : ℝ) (k : Fin D) :
    den (SHlo.veclnGammaSgd gN xN epsStr lrStr ε x γ lr (.operand cotN dy)) k
      = γ k - lr * ∑ o : Fin (N * D),
          pdiv (fun gv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε gv βv (Mat.unflatten x r))) γ k o * dy o := by
  simp only [den]
  exact vit_render_veclngamma_certified ε βv γ (Mat.unflatten x) dy lr k

/-- **Per-token dense weight op denotes the certified step.** `den(rowDenseWeightSgd) (flat (i,j))` =
    `W_ij − lr·(Σ_tokens x·dy)`, the certified ∂(rowwise dense)/∂W contraction. Covers Wq/Wk/Wv/Wo/
    Wfc1/Wfc2 (all 6 per-block denses). Delegation to `vit_render_rowdenseW_certified`. -/
theorem rowDenseWeightSgd_den {N a c : Nat} (xN wN lrStr cotN : String)
    (bb : Vec c) (x : Vec (N * a)) (W : Mat a c) (dy : Vec (N * c)) (lr : ℝ) (i : Fin a) (j : Fin c) :
    den (SHlo.rowDenseWeightSgd xN wN lrStr x W lr (.operand cotN dy)) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ o : Fin (N * c),
          pdiv (fun v : Vec (a * c) =>
                  Mat.flatten (fun r => dense (Mat.unflatten v) bb (Mat.unflatten x r)))
               (Mat.flatten W) (finProdFinEquiv (i, j)) o * dy o := by
  simp only [den, Mat.flatten, Equiv.symm_apply_apply]
  exact vit_render_rowdenseW_certified bb (Mat.unflatten x) W dy lr i j

/-- **Per-token dense bias op denotes the certified step** (dense-bias forward). `den(rowDenseBiasSgd)`
    = `b − lr·(Σ_tokens dy)`. Covers bq/bk/bv/bo/bfc1/bfc2. Delegation to `vit_render_rowdenseb_certified`. -/
theorem rowDenseBiasSgd_den {N a c : Nat} (bN lrStr cotN : String)
    (W : Mat a c) (X : Mat N a) (b : Vec c) (dy : Vec (N * c)) (lr : ℝ) (i : Fin c) :
    den (SHlo.rowDenseBiasSgd bN lrStr b lr (.operand cotN dy)) i
      = b i - lr * ∑ o : Fin (N * c),
          pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r))) b i o * dy o := by
  simp only [den]
  exact vit_render_rowdenseb_certified W X b dy lr i

/-- **The SAME per-token bias op, certified against the vector-LN β forward.** The LN β grad is
    `Σ_tokens dy` — identical reduce to the dense bias — so `rowDenseBiasSgd` ALSO denotes the certified
    ∂(rowwise vector-LN)/∂β contraction. Covers all 25 LN-β sites. Delegation to `vit_render_veclnbeta_certified`. -/
theorem rowDenseBiasSgd_den_lnbeta {N D : Nat} (bN lrStr cotN : String)
    (ε : ℝ) (γv : Vec D) (X : Mat N D) (β : Vec D) (dy : Vec (N * D)) (lr : ℝ) (i : Fin D) :
    den (SHlo.rowDenseBiasSgd bN lrStr β lr (.operand cotN dy)) i
      = β i - lr * ∑ o : Fin (N * D),
          pdiv (fun bv : Vec D =>
                  Mat.flatten (fun r => layerNormVec D ε γv bv (X r))) β i o * dy o := by
  simp only [den]
  exact vit_render_veclnbeta_certified ε γv β X dy lr i

/-- **Patch-embed conv weight op denotes the certified step.** `den(patchEmbedWeightSgd) (flat
    (d,c,kh,kw))` = `W − lr·(certified patchify-conv weight grad)`. The ViT analogue of ConvNeXt's
    stem 4×4/s4 weight — but here a VJP cert EXISTS, so it ties (vit has no even-kernel weight gap).
    Delegation to `vit_render_patchW_certified`. -/
theorem patchEmbedWeightSgd_den {ic H W P N D : Nat} (wN xN lrStr cotN : String)
    (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W)) (Wp : Kernel4 D ic P P)
    (dy : Vec ((N + 1) * D)) (lr : ℝ) (d : Fin D) (c : Fin ic) (kh kw : Fin P) :
    den (SHlo.patchEmbedWeightSgd wN xN lrStr img Wp lr (.operand cotN dy))
        (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw))
      = Wp d c kh kw - lr * ∑ o : Fin ((N + 1) * D),
          pdiv (fun v : Vec (D * ic * P * P) =>
                  patchEmbed_flat ic H W P N D (Kernel4.unflatten v) bc cls pos img)
            (Kernel4.flatten Wp)
            (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o * dy o := by
  simp only [den, patchEmbedWeightGradFlat, Kernel4.flatten, Equiv.symm_apply_apply]
  exact vit_render_patchW_certified Wp bc cls pos img dy lr d c kh kw

/-- **Patch-embed conv bias op denotes the certified step.** `den(patchEmbedBiasSgd)` = `b − lr·(Σ_patches
    dy)` (CLS row 0 excluded). Delegation to `vit_render_patchb_certified`. -/
theorem patchEmbedBiasSgd_den {ic H W P N D : Nat} (bN lrStr cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W))
    (dy : Vec ((N + 1) * D)) (lr : ℝ) (i : Fin D) :
    den (SHlo.patchEmbedBiasSgd bN lrStr bc lr (.operand cotN dy)) i
      = bc i - lr * ∑ o : Fin ((N + 1) * D),
          pdiv (fun b' : Vec D => patchEmbed_flat ic H W P N D Wc b' cls pos img) bc i o * dy o := by
  simp only [den]
  exact vit_render_patchb_certified Wc bc cls pos img dy lr i

/-- **Positional-embed op denotes the certified step.** `den(posEmbedSgd)` = `pos − lr·dy` (the pos
    Jacobian is the identity — pos is added to every token). Delegation to `vit_render_pos_certified`. -/
theorem posEmbedSgd_den {ic H W P N D : Nat} (pN lrStr cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W))
    (dy : Vec ((N + 1) * D)) (lr : ℝ) (i : Fin ((N + 1) * D)) :
    den (SHlo.posEmbedSgd pN lrStr pos lr (.operand cotN dy)) i
      = Mat.flatten pos i - lr * ∑ j : Fin ((N + 1) * D),
          pdiv (fun p : Vec ((N + 1) * D) =>
                  patchEmbed_flat ic H W P N D Wc bc cls (Mat.unflatten p) img)
            (Mat.flatten pos) i j * dy j := by
  simp only [den]
  exact vit_render_pos_certified Wc bc cls pos img dy lr i

-- The **CLS token** reuses the batched `denseBiasSgdB` (the render takes `clsSliceF`'s row-0 slice of
-- the embed cotangent, then the `{N=1}` batch reduce); its patch-embed cls-Jacobian connection
-- (`vit_render_cls_certified`) is threaded at the §1a tie (where the cls cotangent IS the cls slice),
-- exactly as the reused conv/dense/BN ops are in the mnv2/r34/convnext ties — no NEW fold lemma here.

/-- **Classifier head weight op denotes the certified step** — the CLS-vector dense `[D,nClasses]`,
    covered VERBATIM by the M2 generic (single-vector dense, nothing to row-lift). -/
theorem headW_den {D nC : Nat} (aN wN lrStr cotN : String)
    (a : Vec D) (Wc : Mat D nC) (bc : Vec nC) (cot : Vec nC) (lr : ℝ) (i : Fin D) (j : Fin nC) :
    den (SHlo.weightSgd aN wN lrStr a Wc lr (.operand cotN cot)) (finProdFinEquiv (i, j))
      = Wc i j - lr * ∑ k : Fin nC,
          pdiv (fun v : Vec (D * nC) => dense (Mat.unflatten v) bc a) (Mat.flatten Wc)
               (finProdFinEquiv (i, j)) k * cot k :=
  Proofs.Cifar8PoC.denseW_den aN wN lrStr cotN a Wc bc cot lr i j

/-- **Classifier head bias op denotes the certified step.** Peer of `headW_den`. -/
theorem headB_den {D nC : Nat} (bN lrStr cotN : String)
    (Wc : Mat D nC) (a : Vec D) (bc : Vec nC) (cot : Vec nC) (lr : ℝ) (i : Fin nC) :
    den (SHlo.biasSgd bN lrStr bc lr (.operand cotN cot)) i
      = bc i - lr * ∑ j : Fin nC,
          pdiv (fun b' : Vec nC => dense Wc b' a) bc i j * cot j :=
  Proofs.Cifar8PoC.denseB_den bN lrStr cotN Wc a bc cot lr i

end Proofs.ViTPoC
