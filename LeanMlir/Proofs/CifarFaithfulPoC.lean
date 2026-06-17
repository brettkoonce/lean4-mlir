import LeanMlir.Proofs.CnnChainClose
import LeanMlir.Proofs.CnnRender

/-! # PoC: the CIFAR-CNN (Chapter 5, no-BN) train step, proof-tied to the certified SGD step

The Chapter-5 peer of `CnnFaithfulPoC` — a deeper, two-spatial-scale conv net
(`(conv→relu)×2 → pool → (conv→relu)×2 → pool → (dense→relu)×2 → dense`; 14 params:
4 conv kernels/biases + 3 dense layers). `MainCifarVerified` trains on
`verified_mlir/cifar_train_step.mlir`; this file makes its parameter updates
`den`-faithful — each emitted SGD op denotes the certified loss-descent step.

**Zero new core ops.** The conv layers reuse the `convWeightSgd`/`convBiasSgd` ops
added for cnn (CnnFaithfulPoC); the dense head reuses `weightSgd`/`biasSgd`. The
only new content is the per-net `den = certified` capstones below.

* **Conv layers (all four):** `convW_den`/`convB_den` are *generic* in the conv dims
  and the cotangent `c` — `den (convWeightSgd … (.operand _ c)) = θ − lr·(certified
  ∂conv/∂θ · c)`, the emitted op's `den` reduced (`rfl`) to the LHS of the generic
  `cnn_render_conv{W,b}_certified`. Instantiated at each conv layer's `(b,x,W)` and the
  cotangent the renderer feeds there, they certify W₁/b₁ … W₄/b₄ (one lemma each, all
  four layers — conv2d's weight/bias VJP is dim-generic).
* **Dense head (W₅/W₆/W₇):** the classifier head is a 3-layer MLP over the flattened
  pool output, so its cotangents are the IR `mlpCotOut0/1` and its `den`s close via the
  M2 `weight_grad_bridge`/`bias_grad_bridge` — verbatim `CnnFaithfulPoC` (`dW3..5`).

## Honest residual (same boundary as cnn/mlp/linear)
* The conv cotangents here are free variables `c` (the `convW_den`/`convB_den` statement
  is ∀ c) — so the lemmas hold at the actual backward-chain cotangent the renderer
  feeds, without naming it. Pinning each `c` to the exact emitted backward subgraph
  (the `CnnChainClose` recipe, scaled to two stages) is the remaining polish.
* Per-op `pretty` lexing + ℝ → Float32.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.CifarPoC

/-! ## Conv layers — generic `den = certified` (covers all four conv layers) -/

/-- **Any emitted conv weight op = certified.** Generic in the conv dims and the
    cotangent `c`: the `convWeightSgd` op denotes `flatten W − lr·(certified
    ∂conv/∂W · c)`. Instantiated at each layer's `(b,x,W,c)` it certifies W₁…W₄. -/
theorem convW_den {ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Tensor3 ic h w)
    (W : Kernel4 oc ic kH kW) (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*kH*kW)) :
    den (SHlo.convWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
      = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun v' : Vec (oc*ic*kH*kW) =>
                  Tensor3.flatten (conv2d (Kernel4.unflatten v') b x))
               (Kernel4.flatten W) idx j * c j :=
  cnn_render_convW_certified b x (Kernel4.flatten W) c lr idx

/-- **Any emitted conv bias op = certified.** Generic peer of `convW_den`. -/
theorem convB_den {ic oc h w kH kW : Nat}
    (bN lrStr cotN : String) (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc) :
    den (SHlo.convBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j * c j :=
  cnn_render_convb_certified W x b c lr o

/-! ## Dense classifier head (W₅/W₆/W₇) — `weightSgd`/`biasSgd`, mirrors `CnnPoC`

The head `pool2 → W₅→relu→W₆→relu→W₇` is a 3-layer MLP; per-layer cotangents are the
IR `mlpCotOut0/1` (with `(W₇,W₆,W₅)` playing the MLP's `(W₂,W₁,W₀)`). -/

/-- Output-layer weight op `W₇` = certified step (cotangent = the loss cotangent `dy`). -/
theorem dW7_den {c2 h w d1 nClasses : Nat}
    (aN lrStr dyN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i : Fin d1) (j : Fin nClasses) :
    den (SHlo.weightSgd aN "%W7" lrStr (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) W₇ lr
          (.operand dyN dy)) (finProdFinEquiv (i, j))
      = W₇ i j - lr * ∑ k : Fin nClasses,
          pdiv (fun v : Vec (d1 * nClasses) =>
                  dense (Mat.unflatten v) b₇ (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))))
               (Mat.flatten W₇) (finProdFinEquiv (i, j)) k * dy k := by
  have step : den (SHlo.weightSgd aN "%W7" lrStr (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) W₇ lr
                (.operand dyN dy)) (finProdFinEquiv (i, j))
            = W₇ i j - lr * emitWeightGrad (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool))))
                Back.cotangent dy i j := by
    simp only [den, emitWeightGrad, Mat.outer, Back.denote, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, weight_grad_bridge W₇ b₇ (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool))))
        Back.cotangent dy i j]; rfl

/-- Hidden-layer weight op `W₆` = certified step (cotangent = `mlpCotOut1 W₇ h6`). -/
theorem dW6_den {c2 h w d1 nClasses : Nat}
    (aN lrStr cN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses) (lr : ℝ) (i j : Fin d1) :
    den (SHlo.weightSgd aN "%W6" lrStr (relu d1 (dense W₅ b₅ pool)) W₆ lr
          (.operand cN ((mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy)))
        (finProdFinEquiv (i, j))
      = W₆ i j - lr * ∑ k : Fin d1,
          pdiv (fun v : Vec (d1 * d1) => dense (Mat.unflatten v) b₆ (relu d1 (dense W₅ b₅ pool)))
               (Mat.flatten W₆) (finProdFinEquiv (i, j)) k
            * (mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy k := by
  have step : den (SHlo.weightSgd aN "%W6" lrStr (relu d1 (dense W₅ b₅ pool)) W₆ lr
                (.operand cN ((mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy)))
                (finProdFinEquiv (i, j))
            = W₆ i j - lr * emitWeightGrad (relu d1 (dense W₅ b₅ pool))
                (mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) dy i j := by
    simp only [den, emitWeightGrad, Mat.outer, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, weight_grad_bridge W₆ b₆ (relu d1 (dense W₅ b₅ pool))
        (mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) dy i j]

/-- Input-layer (pool) weight op `W₅` = certified step (cotangent = `mlpCotOut0 W₆ W₇ h5 h6`). -/
theorem dW5_den {c2 h w d1 nClasses : Nat}
    (lrStr cN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i : Fin (c2*h*w)) (j : Fin d1) :
    den (SHlo.weightSgd "%pool2" "%W5" lrStr pool W₅ lr
          (.operand cN ((mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                          (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy)))
        (finProdFinEquiv (i, j))
      = W₅ i j - lr * ∑ k : Fin d1,
          pdiv (fun v : Vec ((c2*h*w) * d1) => dense (Mat.unflatten v) b₅ pool)
               (Mat.flatten W₅) (finProdFinEquiv (i, j)) k
            * (mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy k := by
  have step : den (SHlo.weightSgd "%pool2" "%W5" lrStr pool W₅ lr
                (.operand cN ((mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                                (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy)))
                (finProdFinEquiv (i, j))
            = W₅ i j - lr * emitWeightGrad pool
                (mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                  (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) dy i j := by
    simp only [den, emitWeightGrad, Mat.outer, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, weight_grad_bridge W₅ b₅ pool
        (mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool) (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) dy i j]

/-- Output-layer bias op `b₇` = certified step. -/
theorem db7_den {c2 h w d1 nClasses : Nat}
    (lrStr dyN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (b₇ : Vec nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses)
    (lr : ℝ) (i : Fin nClasses) :
    den (SHlo.biasSgd "%b7" lrStr b₇ lr (.operand dyN dy)) i
      = b₇ i - lr * ∑ j : Fin nClasses,
          pdiv (fun b' : Vec nClasses =>
                  dense W₇ b' (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool))))) b₇ i j * dy j := by
  have step : den (SHlo.biasSgd "%b7" lrStr b₇ lr (.operand dyN dy)) i
            = b₇ i - lr * emitBiasGrad Back.cotangent dy i := by
    simp only [den, emitBiasGrad, Back.denote]
  rw [step, bias_grad_bridge W₇ b₇ (relu d1 (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool))))
        Back.cotangent dy i]; rfl

/-- Hidden-layer bias op `b₆` = certified step. -/
theorem db6_den {c2 h w d1 nClasses : Nat}
    (lrStr cN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses) (lr : ℝ) (i : Fin d1) :
    den (SHlo.biasSgd "%b6" lrStr b₆ lr
          (.operand cN ((mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy))) i
      = b₆ i - lr * ∑ j : Fin d1,
          pdiv (fun b' : Vec d1 => dense W₆ b' (relu d1 (dense W₅ b₅ pool))) b₆ i j
            * (mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy j := by
  have step : den (SHlo.biasSgd "%b6" lrStr b₆ lr
                (.operand cN ((mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy))) i
            = b₆ i - lr * emitBiasGrad (mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) dy i := by
    simp only [den, emitBiasGrad]
  rw [step, bias_grad_bridge W₆ b₆ (relu d1 (dense W₅ b₅ pool))
        (mlpCotOut1 W₇ (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) dy i]

/-- Input-layer (pool) bias op `b₅` = certified step. -/
theorem db5_den {c2 h w d1 nClasses : Nat}
    (lrStr cN : String) (W₅ : Mat (c2*h*w) d1) (b₅ : Vec d1) (W₆ : Mat d1 d1) (b₆ : Vec d1)
    (W₇ : Mat d1 nClasses) (pool : Vec (c2*h*w)) (dy : Vec nClasses) (lr : ℝ) (i : Fin d1) :
    den (SHlo.biasSgd "%b5" lrStr b₅ lr
          (.operand cN ((mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                          (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy))) i
      = b₅ i - lr * ∑ j : Fin d1,
          pdiv (fun b' : Vec d1 => dense W₅ b' pool) b₅ i j
            * (mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy j := by
  have step : den (SHlo.biasSgd "%b5" lrStr b₅ lr
                (.operand cN ((mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                                (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))).denote dy))) i
            = b₅ i - lr * emitBiasGrad (mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool)
                (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) dy i := by
    simp only [den, emitBiasGrad]
  rw [step, bias_grad_bridge W₅ b₅ pool
        (mlpCotOut0 W₆ W₇ (dense W₅ b₅ pool) (dense W₆ b₆ (relu d1 (dense W₅ b₅ pool)))) dy i]

end Proofs.CifarPoC
