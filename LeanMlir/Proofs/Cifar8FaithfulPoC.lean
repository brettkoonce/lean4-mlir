import LeanMlir.Proofs.CifarFaithfulPoC

/-! # PoC: the deeper 8-conv CIFAR (cifar8, no-BN) train step, proof-tied

The 4-stage (8-conv) peer of `CifarFaithfulPoC`: `(conv→relu)×2 → pool, four times,
→ (dense→relu)×2 → dense` — 22 params (8 conv kernels/biases, 3 dense layers).
`MainCifar8Verified` trains on `verified_mlir/cifar8_train_step.mlir`.

**Zero new core ops, and almost zero new proof.** Every conv layer is covered by the
*generic* `CifarPoC.convW_den`/`convB_den` (dim- and cotangent-generic — they certify
W₁…W₈ by instantiation). The only thing this file adds is the *generic* dense lemmas
`denseW_den`/`denseB_den` (the dense analogue of `convW_den`/`convB_den`: free in the
activation, weight, bias and cotangent), which certify the three dense layers (and are
reusable by any future dense head). Both close via the M2 `weight_grad_bridge` /
`bias_grad_bridge` at `Back.cotangent`.

Residual: as the non-BN cifar fold (conv/dense cotangents are free vars; cotangent-
subgraph⇄SHlo pin; per-op `pretty` lexing; ℝ→Float32).
-/

open Proofs Proofs.StableHLO Proofs.IR

namespace Proofs.Cifar8PoC

/-- **Any emitted dense weight op = certified.** Generic in the layer dims, the
    activation `a`, bias `b` and cotangent `c`: `den (weightSgd a W (.operand _ c)) =
    W − lr·(certified ∂dense/∂W · c)`. Instantiated per dense layer it certifies W₉/Wₐ/W_b. -/
theorem denseW_den {m n : Nat} (aN wN lrStr cotN : String)
    (a : Vec m) (W : Mat m n) (b : Vec n) (c : Vec n) (lr : ℝ) (i : Fin m) (j : Fin n) :
    den (SHlo.weightSgd aN wN lrStr a W lr (.operand cotN c)) (finProdFinEquiv (i, j))
      = W i j - lr * ∑ k : Fin n,
          pdiv (fun v : Vec (m*n) => dense (Mat.unflatten v) b a) (Mat.flatten W)
               (finProdFinEquiv (i, j)) k * c k := by
  have step : den (SHlo.weightSgd aN wN lrStr a W lr (.operand cotN c)) (finProdFinEquiv (i, j))
            = W i j - lr * emitWeightGrad a Back.cotangent c i j := by
    simp only [den, emitWeightGrad, Mat.outer, Back.denote, Mat.flatten, Equiv.symm_apply_apply]
  rw [step, weight_grad_bridge W b a Back.cotangent c i j]; rfl

/-- **Any emitted dense bias op = certified.** Generic peer of `denseW_den`. -/
theorem denseB_den {m n : Nat} (bN lrStr cotN : String)
    (W : Mat m n) (a : Vec m) (b : Vec n) (c : Vec n) (lr : ℝ) (i : Fin n) :
    den (SHlo.biasSgd bN lrStr b lr (.operand cotN c)) i
      = b i - lr * ∑ j : Fin n,
          pdiv (fun b' : Vec n => dense W b' a) b i j * c j := by
  have step : den (SHlo.biasSgd bN lrStr b lr (.operand cotN c)) i
            = b i - lr * emitBiasGrad Back.cotangent c i := by
    simp only [den, emitBiasGrad, Back.denote]
  rw [step, bias_grad_bridge W b a Back.cotangent c i]; rfl

end Proofs.Cifar8PoC
