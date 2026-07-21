import LeanMlir.Proofs.Foundation.MlpTrainStep
import LeanMlir.Proofs.MlpRender
import LeanMlir.Proofs.Foundation.LinearFaithfulPoC

/-! # PoC: the MNIST-MLP train step, proof-tied to the certified SGD step

The MLP analogue of `LinearFaithfulPoC`. `MainMnistMlpVerified` trains on
`verified_mlir/mlp_train_step.mlir`; this file makes the *whole* module
`pretty(provenGraph)` — forward (`denseF`/`reluF`), the loss cotangent, the
backward chain (`dotOut`/`selectPos`), and the six parameter SGD updates as the
`weightSgd`/`biasSgd` `SHlo` ops added in `LinearFaithfulPoC`'s core extension —
and proves each output's `den` equals the certified `fderiv`-derived loss-descent
step, reusing `mlp_render_{W,b}*_certified` (the per-layer bridges) and
`mlpCotOut{0,1}_denote` (the explicit chain cotangents).

No new core `SHlo` ops are needed: the backward chain uses the existing
`dotOut`/`selectPos`, and the param updates reuse `weightSgd`/`biasSgd`.

Residual (as for linear): per-op `pretty` lexing; B=1 (the emitted module
batch-contracts; `den` is per-example); the ReLU smooth-point hypotheses are
inherited from the bridges; ℝ→Float32.
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.MlpPoC

variable {d₀ d₁ d₂ d₃ : Nat}
  (W₀ : Mat d₀ d₁) (b₀ : Vec d₁) (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
  (W₂ : Mat d₂ d₃) (b₂ : Vec d₃) (x : Vec d₀) (g : Vec d₃) (lr : ℝ)

/-! ## The backward-chain cotangent subgraphs denote the proven `mlpCotOut*`

The emitted `selectPos`/`dotOut` chain — what the faithful renderer prints for the
per-layer pre-activation cotangents — denotes exactly `mlpCotOut{1,0}.denote g`. -/

/-- Layer-1 cotangent subgraph `selectPos p₁ (dotOut W₂ dy)` denotes `mlpCotOut1.denote g`. -/
theorem cot1_den (p₁name dyName : String) :
    den (SHlo.selectPos p₁name (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))
          (SHlo.dotOut "%W2" W₂ (.operand dyName g)))
      = (mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g := by
  rw [mlpCotOut1_denote]
  funext i
  simp only [den, Mat.mulVec]

/-- Layer-0 cotangent subgraph `selectPos p₀ (dotOut W₁ cot1)` denotes `mlpCotOut0.denote g`. -/
theorem cot0_den (p₀name c1name : String) :
    den (SHlo.selectPos p₀name (dense W₀ b₀ x)
          (SHlo.dotOut "%W1" W₁ (.operand c1name
            ((mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g))))
      = (mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g := by
  rw [mlpCotOut0_denote, mlpCotOut1_denote]
  funext i
  simp only [den, Mat.mulVec]

/-! ## The six emitted parameter ops denote the certified SGD step

Each `weightSgd`/`biasSgd` op, fed the right activation (`x` field) and the
cotangent the chain delivers (the `.operand` value), denotes `θ − lr·(certified
per-layer Jacobian · cotangent)` — via the op `den` = `emitWeightGrad`/`emitBiasGrad`
(outer / reduce) and the `mlp_render_*_certified` bridges. -/

/-- Output-layer weight op `weightSgd a1 W₂ (cot = dy)` = certified `W₂` step. -/
theorem W2_den_certified (aN lrStr dyN : String) (i : Fin d₂) (j : Fin d₃) :
    den (SHlo.weightSgd aN "%W2" lrStr (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) W₂ lr
          (.operand dyN g)) (finProdFinEquiv (i, j))
      = W₂ i j - lr * ∑ k : Fin d₃,
          pdiv (fun v : Vec (d₂ * d₃) =>
                  dense (Mat.unflatten v) b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))))
               (Mat.flatten W₂) (finProdFinEquiv (i, j)) k * g k := by
  have step : den (SHlo.weightSgd aN "%W2" lrStr (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) W₂ lr
                (.operand dyN g)) (finProdFinEquiv (i, j))
            = W₂ i j - lr * emitWeightGrad (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))
                Back.cotangent g i j := by
    simp only [den, emitWeightGrad, Mat.outer, Back.denote, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, mlp_render_W2_certified W₀ b₀ W₁ b₁ W₂ b₂ x g lr i j]

/-- Hidden-layer weight op `weightSgd a0 W₁ (cot = mlpCotOut1)` = certified `W₁` step. -/
theorem W1_den_certified (aN lrStr cN : String) (i : Fin d₁) (j : Fin d₂) :
    den (SHlo.weightSgd aN "%W1" lrStr (relu d₁ (dense W₀ b₀ x)) W₁ lr
          (.operand cN ((mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g)))
        (finProdFinEquiv (i, j))
      = W₁ i j - lr * ∑ k : Fin d₂,
          pdiv (fun v : Vec (d₁ * d₂) => dense (Mat.unflatten v) b₁ (relu d₁ (dense W₀ b₀ x)))
               (Mat.flatten W₁) (finProdFinEquiv (i, j)) k
            * (mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g k := by
  have step : den (SHlo.weightSgd aN "%W1" lrStr (relu d₁ (dense W₀ b₀ x)) W₁ lr
                (.operand cN ((mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g)))
                (finProdFinEquiv (i, j))
            = W₁ i j - lr * emitWeightGrad (relu d₁ (dense W₀ b₀ x))
                (mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) g i j := by
    simp only [den, emitWeightGrad, Mat.outer, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, mlp_render_W1_certified W₀ b₀ W₁ b₁ W₂ x g lr i j]

/-- Input-layer weight op `weightSgd x W₀ (cot = mlpCotOut0)` = certified `W₀` step. -/
theorem W0_den_certified (lrStr cN : String) (i : Fin d₀) (j : Fin d₁) :
    den (SHlo.weightSgd "%x" "%W0" lrStr x W₀ lr
          (.operand cN ((mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g)))
        (finProdFinEquiv (i, j))
      = W₀ i j - lr * ∑ k : Fin d₁,
          pdiv (fun v : Vec (d₀ * d₁) => dense (Mat.unflatten v) b₀ x)
               (Mat.flatten W₀) (finProdFinEquiv (i, j)) k
            * (mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g k := by
  have step : den (SHlo.weightSgd "%x" "%W0" lrStr x W₀ lr
                (.operand cN ((mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g)))
                (finProdFinEquiv (i, j))
            = W₀ i j - lr * emitWeightGrad x
                (mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) g i j := by
    simp only [den, emitWeightGrad, Mat.outer, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, mlp_render_W0_certified W₀ b₀ W₁ b₁ W₂ x g lr i j]

/-- Output-layer bias op = certified `b₂` step. -/
theorem b2_den_certified (lrStr dyN : String) (i : Fin d₃) :
    den (SHlo.biasSgd "%b2" lrStr b₂ lr (.operand dyN g)) i
      = b₂ i - lr * ∑ j : Fin d₃,
          pdiv (fun b' : Vec d₃ => dense W₂ b' (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) b₂ i j
            * g j := by
  have step : den (SHlo.biasSgd "%b2" lrStr b₂ lr (.operand dyN g)) i
            = b₂ i - lr * emitBiasGrad (Back.cotangent) g i := by
    simp only [den, emitBiasGrad, Back.denote]
  rw [step, mlp_render_b2_certified W₀ b₀ W₁ b₁ W₂ b₂ x g lr i]

/-- Hidden-layer bias op = certified `b₁` step. -/
theorem b1_den_certified (lrStr cN : String) (i : Fin d₂) :
    den (SHlo.biasSgd "%b1" lrStr b₁ lr
          (.operand cN ((mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g))) i
      = b₁ i - lr * ∑ j : Fin d₂,
          pdiv (fun b' : Vec d₂ => dense W₁ b' (relu d₁ (dense W₀ b₀ x))) b₁ i j
            * (mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g j := by
  have step : den (SHlo.biasSgd "%b1" lrStr b₁ lr
                (.operand cN ((mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g))) i
            = b₁ i - lr * emitBiasGrad (mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) g i := by
    simp only [den, emitBiasGrad]
  rw [step, mlp_render_b1_certified W₀ b₀ W₁ b₁ W₂ x g lr i]

/-- Input-layer bias op = certified `b₀` step. -/
theorem b0_den_certified (lrStr cN : String) (i : Fin d₁) :
    den (SHlo.biasSgd "%b0" lrStr b₀ lr
          (.operand cN ((mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g))) i
      = b₀ i - lr * ∑ j : Fin d₁,
          pdiv (fun b' : Vec d₁ => dense W₀ b' x) b₀ i j
            * (mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g j := by
  have step : den (SHlo.biasSgd "%b0" lrStr b₀ lr
                (.operand cN ((mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote g))) i
            = b₀ i - lr * emitBiasGrad (mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) g i := by
    simp only [den, emitBiasGrad]
  rw [step, mlp_render_b0_certified W₀ b₀ W₁ b₁ W₂ x g lr i]

/-! ## Fully tied — the top loss cotangent is the composed softmax-CE of the forward

The six `*_den_certified` above hold `∀ g` (a symbolic top loss cotangent). The renderer feeds
the cotangent the EMITTED loss graph `sub(softmaxDiv(expe(logits)), onehot)` produces, where
`logits` is the real forward output. The lemma below pins that graph's `den` to the composed
softmax-CE gradient of the forward (`∂CE/∂logits`, the mlp analogue of `lossCotGraph_isCEgrad`);
instantiating the six certified theorems at it ties the train step forward→loss→backward, with
the output layer folding to the WHOLE-loss gradient `∂CE/∂W₂` (`mlp_output_total_loss_grad`). -/

/-- **The emitted loss-cotangent graph denotes the composed softmax-CE gradient of the forward**
    (`= softmax(logits) − onehot = ∂CE/∂logits` at the real forward logits). -/
theorem mlpLossCot_den (nlogN ohN : String) (label : Fin d₃) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe
            (.operand nlogN (mnistLinear W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))))))
          (.operand ohN (oneHot d₃ label)))
      = fun j => softmax d₃ (mnistLinear W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) j
                  - oneHot d₃ label j := by
  funext j; simp only [den, softmax]

/-- **Output-layer weight op, fully tied to the WHOLE softmax-CE loss.** Fed the cotangent the
    emitted loss graph denotes (`mlpLossCot_den`), the `weightSgd` for `W₂` denotes
    `W₂ − lr·∂(crossEntropy ∘ forward)/∂W₂` — the certified TOTAL-loss-descent step, composed
    forward→softmax-CE→gradient, no symbolic cotangent. -/
theorem mlp_W2_tied_totalloss (aN lrStr dyN : String) (label : Fin d₃) (i : Fin d₂) (j : Fin d₃) :
    den (SHlo.weightSgd aN "%W2" lrStr (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) W₂ lr
          (.operand dyN (fun k => softmax d₃ (mnistLinear W₂ b₂
                  (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k)))
        (finProdFinEquiv (i, j))
      = W₂ i j - lr * pdiv (fun v : Vec (d₂ * d₃) => fun _ : Fin 1 =>
            crossEntropy d₃ (dense (Mat.unflatten v) b₂
              (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
          (Mat.flatten W₂) (finProdFinEquiv (i, j)) 0 := by
  rw [W2_den_certified W₀ b₀ W₁ b₁ W₂ b₂ x
        (fun k => softmax d₃ (mnistLinear W₂ b₂
            (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k) lr aN lrStr dyN i j,
      mlp_output_total_loss_grad W₂ b₂ (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) label i j]

/-- **Whole mlp train step, tied.** With the top loss cotangent `g` pinned to the composed
    softmax-CE gradient of the forward (`mlpLossCot_den`), all six emitted parameter ops denote
    the certified loss-descent step — the output weight `W₂` folded to the WHOLE-loss gradient
    `∂CE/∂W₂` (`mlp_W2_tied_totalloss`), the other five to `θ − lr·(certified ∂layer/∂θ · the
    backward-chain cotangent the real loss drives)` (the `*_den_certified` at the composed `g`).
    No symbolic cotangent remains; the forward is shared (single render, correctly-threaded SSAs). -/
theorem mlp_train_step_tied_certified (lrStr aN dyN cN : String) (label : Fin d₃) :
    -- `g` = the softmax-CE gradient of the REAL forward logits (= `den` of the emitted loss graph)
    (∀ i j, den (SHlo.weightSgd aN "%W2" lrStr (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))) W₂ lr
              (.operand dyN (fun k => softmax d₃ (mnistLinear W₂ b₂
                  (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k)))
            (finProdFinEquiv (i, j))
          = W₂ i j - lr * pdiv (fun v : Vec (d₂ * d₃) => fun _ : Fin 1 =>
                crossEntropy d₃ (dense (Mat.unflatten v) b₂
                  (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) label)
              (Mat.flatten W₂) (finProdFinEquiv (i, j)) 0)
  ∧ (∀ i, den (SHlo.biasSgd "%b2" lrStr b₂ lr
              (.operand dyN (fun k => softmax d₃ (mnistLinear W₂ b₂
                  (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k))) i
          = b₂ i - lr * ∑ j : Fin d₃,
              pdiv (fun b' : Vec d₃ => dense W₂ b' (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) b₂ i j
                * (fun k => softmax d₃ (mnistLinear W₂ b₂
                    (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k) j)
  ∧ (∀ i j, den (SHlo.weightSgd aN "%W1" lrStr (relu d₁ (dense W₀ b₀ x)) W₁ lr
              (.operand cN ((mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote
                  (fun k => softmax d₃ (mnistLinear W₂ b₂
                    (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k))))
            (finProdFinEquiv (i, j))
          = W₁ i j - lr * ∑ k : Fin d₂,
              pdiv (fun v : Vec (d₁ * d₂) => dense (Mat.unflatten v) b₁ (relu d₁ (dense W₀ b₀ x)))
                   (Mat.flatten W₁) (finProdFinEquiv (i, j)) k
                * (mlpCotOut1 W₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote
                    (fun k => softmax d₃ (mnistLinear W₂ b₂
                      (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k) k)
  ∧ (∀ i j, den (SHlo.weightSgd "%x" "%W0" lrStr x W₀ lr
              (.operand cN ((mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote
                  (fun k => softmax d₃ (mnistLinear W₂ b₂
                    (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k))))
            (finProdFinEquiv (i, j))
          = W₀ i j - lr * ∑ k : Fin d₁,
              pdiv (fun v : Vec (d₀ * d₁) => dense (Mat.unflatten v) b₀ x)
                   (Mat.flatten W₀) (finProdFinEquiv (i, j)) k
                * (mlpCotOut0 W₁ W₂ (dense W₀ b₀ x) (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)))).denote
                    (fun k => softmax d₃ (mnistLinear W₂ b₂
                      (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k) k) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i j; exact mlp_W2_tied_totalloss W₀ b₀ W₁ b₁ W₂ b₂ x lr aN lrStr dyN label i j
  · intro i
    exact b2_den_certified W₀ b₀ W₁ b₁ W₂ b₂ x
      (fun k => softmax d₃ (mnistLinear W₂ b₂
        (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k) lr lrStr dyN i
  · intro i j
    exact W1_den_certified W₀ b₀ W₁ b₁ W₂ x
      (fun k => softmax d₃ (mnistLinear W₂ b₂
        (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k) lr aN lrStr cN i j
  · intro i j
    exact W0_den_certified W₀ b₀ W₁ b₁ W₂ x
      (fun k => softmax d₃ (mnistLinear W₂ b₂
        (relu d₂ (dense W₁ b₁ (relu d₁ (dense W₀ b₀ x))))) k - oneHot d₃ label k) lr lrStr cN i j

end Proofs.MlpPoC
