import LeanMlir.Proofs.StableHLO

/-! # M1 — the linear train step descends the certified softmax-CE gradient

`StableHLO.lean` proves the Chapter-2 linear train step piecewise: the forward
graph (`fwdGraph_faithful`), the loss cotangent (`lossCotGraph_isCEgrad`), the
per-parameter Jacobians (`wGrad/bGrad_isWeightJacobian`), and the plain-SGD update
(`sgdW/sgdB_descends_certified_grad`). Each of those, however, still mentions the
emitted cotangent as `den (lossCotGraph …)` — a denotation of an emitted graph,
not yet a named closed form.

This file bundles them into a single statement per parameter: the emitted SGD
update subtracts `lr` times **[the certified ∂logits/∂θ Jacobian]** contracted with
**[the certified closed-form softmax-CE gradient `softmax − onehot`]**. Every factor
is now a named, axiom-audited certified quantity — no residual `den`-of-graph, no
trusted optimizer step. This is the *denotation* half of milestone M1 for `linear`
(what the emitted train step computes).

Two things are deliberately NOT done here, and are tracked in
`planning/verified_train_step.md`:

* **The chain-rule fold.** The two-factor sum below is, by `pdiv_comp`, the single
  gradient `∂/∂θ (crossEntropy ∘ mnistLinear)` — i.e. literally one step of gradient
  descent on the loss. Stating it in folded form needs `DifferentiableAt` for
  `crossEntropy` (no such lemma exists yet) and for the dense-wrt-flattened-weights
  map; left as the next proof step. The unfolded form here carries the same content
  with no smoothness obligation.
* **The rendering half.** `den`/`SHlo` is a single-example semantics with no
  constructors for the batched weight-grad `dot_general`, bias-grad `reduce`, or SGD
  `multiply`/`subtract`; that tail of `verified_mlir/linear_train_step.mlir` is still
  hand-written string concat. Closing `emitted text = render(provenGraph)` needs the
  batched multi-output AST (Stage 1 of the plan).
-/

namespace Proofs.StableHLO

open Proofs

variable {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)

/-- The emitted loss cotangent is the certified closed-form softmax-CE gradient
    `softmax(logits) − onehot`, by `lossCotGraph_isCEgrad` then `softmaxCE_grad`. -/
theorem lossCot_eq_softmax_sub_onehot (label : Fin n) (k : Fin n) :
    den (lossCotGraph W b x (oneHot n label)) k
      = softmax n (mnistLinear W b x) k - oneHot n label k := by
  rw [lossCotGraph_isCEgrad W b x label k,
      softmaxCE_grad n (mnistLinear W b x) label k]

/-- **M1 (weight).** The emitted linear SGD weight update subtracts `lr` times the
    certified ∂logits/∂W Jacobian contracted with the certified closed-form
    softmax-CE gradient `softmax − onehot`. -/
theorem sgdW_descends_softmaxCE_grad (lr : ℝ) (label : Fin n) (i : Fin m) (j : Fin n) :
    sgdW W b x lr label i j
      = W i j - lr * ∑ k : Fin n,
          pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
               (Mat.flatten W) (finProdFinEquiv (i, j)) k
            * (softmax n (mnistLinear W b x) k - oneHot n label k) := by
  rw [sgdW_descends_certified_grad W b x lr label i j]
  simp_rw [lossCot_eq_softmax_sub_onehot W b x label]

/-- **M1 (bias).** The emitted linear SGD bias update subtracts `lr` times the
    certified ∂logits/∂b Jacobian contracted with the same certified softmax-CE
    gradient. -/
theorem sgdB_descends_softmaxCE_grad (lr : ℝ) (label : Fin n) (j : Fin n) :
    sgdB W b x lr label j
      = b j - lr * ∑ i : Fin n,
          pdiv (fun b' : Vec n => dense W b' x) b j i
            * (softmax n (mnistLinear W b x) i - oneHot n label i) := by
  rw [sgdB_descends_certified_grad W b x lr label j]
  simp_rw [lossCot_eq_softmax_sub_onehot W b x label]

end Proofs.StableHLO
