import LeanMlir.Proofs.Architectures.TokenParamGrad
import LeanMlir.Proofs.Foundation.SgdNodes

/-! # ViT-Tiny fold — each emitted param-SGD op `den`otes the certified loss-descent step

The ViT peer of `ConvNeXtFold`/`EfficientNetFold`: for every param-SGD op the
`vitTrainStepRenderV` renderer emits, prove `den(op) = θ − lr·(certified Jacobian · cotangent)`. The
vector-LN γ / β nodes (`veclnGammaSgd_den`, `rowDenseBiasSgd_den_lnbeta` and their tie clauses) are
in `SgdNodes`, under this namespace, because ConvNeXt's head uses them too. Each is a one-or-few-line delegation to the already-proven render certs in
`ViTVecLN` (vector-[D] LN γ/β) and `TokenParamGrad` (rowwise dense W/b, patch conv W/b, cls, pos); the
classifier head reuses `Cifar8PoC.dense{W,B}_den`. Together these cover EVERY parameter family
of the depth-12 ViT-Tiny train step (200 params), so the step tie (`ViTStepTie`) can thread them at
the real backward chain cotangents.

The op `den`s and the cert LHSs line up by construction (the core ops were built to denote exactly
these grads): `veclnGammaSgd`→`layerNormVec_gamma_sgd_certified`, `rowDenseWeightSgd`→
`rowDense_weight_sgd_certified`, `rowDenseBiasSgd`→`rowDense_bias_sgd_certified` (dense bias) and
`layerNormVec_beta_sgd_certified` (LN β — same op, different forward in the pdiv), `patchEmbedWeightSgd`
→`patchEmbed_weight_sgd_certified`, `patchEmbedBiasSgd`→`patchEmbed_bias_sgd_certified`, `posEmbedSgd`→
`posEmbed_sgd_certified`, cls (`clsSliceF`→`denseBiasSgdB`)→`clsToken_sgd_certified`. -/

namespace Proofs.ViTPoC
open Proofs.SgdNode

open scoped BigOperators
open Proofs Proofs.StableHLO

/-- **Per-token dense weight op denotes the certified step.** `den(rowDenseWeightSgd) (flat (i,j))` =
    `W_ij − lr·(Σ_tokens x·dy)`, the certified ∂(rowwise dense)/∂W contraction. Covers Wq/Wk/Wv/Wo/
    Wfc1/Wfc2 (all 6 per-block denses). Delegation to `rowDense_weight_sgd_certified`. -/
theorem rowDenseWeightSgd_den {N a c : Nat} (xN wN lrStr cotN : String)
    (bb : Vec c) (x : Vec (N * a)) (W : Mat a c) (dy : Vec (N * c)) (lr : ℝ) (i : Fin a) (j : Fin c) :
    den (SHlo.rowDenseWeightSgd xN wN lrStr x W lr (.operand cotN dy)) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ o : Fin (N * c),
          pdiv (fun v : Vec (a * c) =>
                  Mat.flatten (fun r => dense (Mat.unflatten v) bb (Mat.unflatten x r)))
               (Mat.flatten W) (finProdFinEquiv (i, j)) o * dy o := by
  simp only [denStep, denStepApp, Mat.flatten, Equiv.symm_apply_apply]
  exact rowDense_weight_sgd_certified bb (Mat.unflatten x) W dy lr i j

/-- **Per-token dense bias op denotes the certified step** (dense-bias forward). `den(rowDenseBiasSgd)`
    = `b − lr·(Σ_tokens dy)`. Covers bq/bk/bv/bo/bfc1/bfc2. Delegation to `rowDense_bias_sgd_certified`. -/
theorem rowDenseBiasSgd_den {N a c : Nat} (bN lrStr cotN : String)
    (W : Mat a c) (X : Mat N a) (b : Vec c) (dy : Vec (N * c)) (lr : ℝ) (i : Fin c) :
    den (SHlo.rowDenseBiasSgd bN lrStr b lr (.operand cotN dy)) i
      = b i - lr * ∑ o : Fin (N * c),
          pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (X r))) b i o * dy o := by
  simp only [denStep, denStepApp]
  exact rowDense_bias_sgd_certified W X b dy lr i

/-- **Patch-embed conv weight op denotes the certified step.** `den(patchEmbedWeightSgd) (flat
    (d,c,kh,kw))` = `W − lr·(certified patchify-conv weight grad)`. The ViT analogue of ConvNeXt's
    stem 4×4/s4 weight — but here a VJP cert EXISTS, so it ties (vit has no even-kernel weight gap).
    Delegation to `patchEmbed_weight_sgd_certified`. -/
theorem patchEmbedWeightSgd_den {ic H W P N D : Nat} (wN xN lrStr cotN : String)
    (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W)) (Wp : Kernel4 D ic P P)
    (dy : Vec ((N + 1) * D)) (lr : ℝ) (d : Fin D) (c : Fin ic) (kh kw : Fin P) :
    den (SHlo.patchEmbedWeightSgd wN xN lrStr img Wp lr (.operand cotN dy))
        (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw))
      = Wp d c kh kw - lr * ∑ o : Fin ((N + 1) * D),
          pdiv (fun v : Vec (D * ic * P * P) =>
                  patchEmbedFlat ic H W P N D (Kernel4.unflatten v) bc cls pos img)
            (Kernel4.flatten Wp)
            (finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (d, c), kh), kw)) o * dy o := by
  simp only [denStepApp, patchEmbedWeightGradFlat, Kernel4.flatten, Equiv.symm_apply_apply]
  exact patchEmbed_weight_sgd_certified Wp bc cls pos img dy lr d c kh kw

/-- **Patch-embed conv bias op denotes the certified step.** `den(patchEmbedBiasSgd)` = `b − lr·(Σ_patches
    dy)` (CLS row 0 excluded). Delegation to `patchEmbed_bias_sgd_certified`. -/
theorem patchEmbedBiasSgd_den {ic H W P N D : Nat} (bN lrStr cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W))
    (dy : Vec ((N + 1) * D)) (lr : ℝ) (i : Fin D) :
    den (SHlo.patchEmbedBiasSgd bN lrStr bc lr (.operand cotN dy)) i
      = bc i - lr * ∑ o : Fin ((N + 1) * D),
          pdiv (fun b' : Vec D => patchEmbedFlat ic H W P N D Wc b' cls pos img) bc i o * dy o := by
  simp only [denStep, denStepApp]
  exact patchEmbed_bias_sgd_certified Wc bc cls pos img dy lr i

/-- **Positional-embed op denotes the certified step.** `den(posEmbedSgd)` = `pos − lr·dy` (the pos
    Jacobian is the identity — pos is added to every token). Delegation to `posEmbed_sgd_certified`. -/
theorem posEmbedSgd_den {ic H W P N D : Nat} (pN lrStr cotN : String)
    (Wc : Kernel4 D ic P P) (bc cls : Vec D) (pos : Mat (N + 1) D) (img : Vec (ic * H * W))
    (dy : Vec ((N + 1) * D)) (lr : ℝ) (i : Fin ((N + 1) * D)) :
    den (SHlo.posEmbedSgd pN lrStr pos lr (.operand cotN dy)) i
      = Mat.flatten pos i - lr * ∑ j : Fin ((N + 1) * D),
          pdiv (fun p : Vec ((N + 1) * D) =>
                  patchEmbedFlat ic H W P N D Wc bc cls (Mat.unflatten p) img)
            (Mat.flatten pos) i j * dy j := by
  simp only [denStepApp]
  exact posEmbed_sgd_certified Wc bc cls pos img dy lr i

-- The **CLS token** reuses the batched `denseBiasSgdB` (the render takes `clsSliceF`'s row-0 slice of
-- the embed cotangent, then the `{N=1}` batch reduce); its patch-embed cls-Jacobian connection
-- (`clsToken_sgd_certified`) is threaded at the §1a tie (where the cls cotangent IS the cls slice),
-- exactly as the reused conv/dense/BN ops are in the mnv2/r34/convnext ties — no NEW fold lemma here.

-- ════════════════════════════════════════════════════════════════
-- § Tie clauses — one per-token SGD node each (each its `_den` lemma's statement with the index
--   bound, over the flat input `x`; each `…_holds` below proves it)
-- ════════════════════════════════════════════════════════════════

/-- A per-token dense weight SGD node, tied (`rowDenseWeightSgd_den`). -/
def RowDenseWSgdTied (N : Nat) {a c : Nat} (xN wN lrStr cotN : String) (bb : Vec c)
    (x : Vec (N * a)) (W : Mat a c) (dy : Vec (N * c)) (lr : ℝ) : Prop :=
  ∀ (i : Fin a) (j : Fin c),
    den (SHlo.rowDenseWeightSgd xN wN lrStr x W lr (.operand cotN dy)) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ o : Fin (N * c),
          pdiv (fun v : Vec (a * c) =>
                  Mat.flatten (fun r => dense (Mat.unflatten v) bb (Mat.unflatten x r)))
               (Mat.flatten W) (finProdFinEquiv (i, j)) o * dy o

/-- A per-token dense bias SGD node, tied (`rowDenseBiasSgd_den`). -/
def RowDenseBSgdTied (N : Nat) {a c : Nat} (bN lrStr cotN : String) (W : Mat a c)
    (x : Vec (N * a)) (b : Vec c) (dy : Vec (N * c)) (lr : ℝ) : Prop :=
  ∀ i : Fin c,
    den (SHlo.rowDenseBiasSgd bN lrStr b lr (.operand cotN dy)) i
      = b i - lr * ∑ o : Fin (N * c),
          pdiv (fun b' : Vec c => Mat.flatten (fun r => dense W b' (Mat.unflatten x r))) b i o
            * dy o

/-! Each clause holds, every argument implicit (read off the goal by a step tie's constructor). -/

theorem rowDenseWSgdTied_holds {N a c : Nat} {xN wN lrStr cotN : String} {bb : Vec c}
    {x : Vec (N * a)} {W : Mat a c} {dy : Vec (N * c)} {lr : ℝ} :
    RowDenseWSgdTied N xN wN lrStr cotN bb x W dy lr := fun i j =>
  rowDenseWeightSgd_den xN wN lrStr cotN bb x W dy lr i j

theorem rowDenseBSgdTied_holds {N a c : Nat} {bN lrStr cotN : String} {W : Mat a c}
    {x : Vec (N * a)} {b : Vec c} {dy : Vec (N * c)} {lr : ℝ} :
    RowDenseBSgdTied N bN lrStr cotN W x b dy lr := fun i =>
  rowDenseBiasSgd_den bN lrStr cotN W (Mat.unflatten x) b dy lr i

end Proofs.ViTPoC
