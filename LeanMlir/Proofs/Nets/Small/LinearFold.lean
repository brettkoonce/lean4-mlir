import LeanMlir.Proofs.Nets.Small.LinearTrainStep
import LeanMlir.Proofs.Codegen.ChapterGraphs

/-! # PoC: the MNIST-linear train step, proof-tied to the certified SGD step

`MainMnistLinearVerified`
trains on `verified_mlir/linear_train_step.mlir`, which is written by
`Proofs.StableHLO.linTrainStepFaithfulV` (the `#eval` writer at the end of
`StableHLO.Basic`). This file certifies *that* renderer for a single example `x`: the emitted
weight output denotes `W − lr·∂(crossEntropy ∘ dense)/∂W` (Mathlib-`fderiv`-derived) and the bias
output denotes `b − lr·(certified ∂dense/∂b · (softmax − onehot))`. The committed module
batch-contracts over `B` examples, which this file does not state.

(Namespace/name lengths are kept short on purpose: [`tests/AuditAxioms.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/AuditAxioms.lean)'s
three-axiom closure check greps `#print axioms` output per line, which Lean wraps
past ~120 cols — long qualified names would split the benign triple across lines
and false-fail the check. Keep future per-chapter capstone names short.)

## What is closed here (kernel, `[propext, Classical.choice, Quot.sound]`)

* `fwdGraph_faithful` + `poc_fwd_is_render` — the forward-eval module *is*
  `renderModule` of a graph whose `den` is `mnistLinear` (text = `render(graph)`
  ∧ `den(graph) = math`). **Forward: end-to-end tied.**
* `poc_train_step_tail_certified` — **fully tied.** The two emitted SGD ops consume
  the proven `lossCotGraph` node *directly* (no SSA-name pin), so the weight output's `den`
  is proven = `W − lr·∂CE/∂W` and the bias output's = the certified bias Jacobian contracted
  with `softmax − onehot`, with the forward = the proven `fwdGraph` (nested in
  `lossCotGraph`). `_fwd` and `_train_step` share the same forward graph.

The committed-bytes tie (`verified_mlir/linear_train_step.mlir ==
linTrainStepFaithfulV(…)`) is enforced in CI (regenerate + `git diff`, the
"Verified-render drift guard" step in `proofs.yml`), not here.

## Scope (the boundary shared with the forward `SHlo` `den`)

* **Per-op `den` ⇄ MLIR text** for `weightSgd`/`biasSgd`: that the printed
  `dot_general`/`reduce`/`multiply`/`subtract` compute their `den` is the same trusted
  op-level modelling the forward `den` relies on (the weight-grad piece is `wGrad`, tied to
  `IR.emitWeightGrad` by `wGrad_faithful`).
* **Single example (B = 1):** `wGrad x dy = x ⊗ dy`; the emitted module
  batch-contracts. The mean-loss cotangent makes the batch sum the mean gradient.
* **ℝ → Float32:** not stated here; `FloatBridge.lean` and `MlpFloatBridge.lean` cover the
  linear/MLP rounding budget separately.
-/

open Proofs Proofs.StableHLO

namespace Proofs.LinPoC

variable {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)

/-! ## Forward eval — end-to-end tied -/

/-- The committed `linear_fwd.mlir` generator `linearFwdModuleV` *is* `renderModule`
    applied to the proven `fwdGraph` — the emitted bytes are literally the print of
    the graph `fwdGraph_faithful` is about. -/
theorem poc_fwd_is_render (B : Nat) :
    ∃ argSig : String,
      linearFwdModuleV B m n W b x
        = renderModule "linear_fwd" argSig B n (fwdGraph W b x) :=
  ⟨_, rfl⟩

/-! ## The tail fold (closed) — the emitted tail ops are `pretty(provenNode)`

`StableHLO.linTrainStepFaithfulV` (what generates `verified_mlir/linear_train_step.mlir`)
renders the *whole* module as `pretty` of `SHlo` nodes, **fully tied**: each of
`SHlo.weightSgd` / `SHlo.biasSgd` consumes the proven `lossCotGraph` node DIRECTLY
(not a `.operand %dy <placeholder>` name-pin), so the forward = the proven `fwdGraph`
(nested inside `lossCotGraph`) and `den(output) = certified` is one composed theorem
below — no trusted SSA-name linkage between the cotangent and the SGD ops. The shared
cotangent is rendered once per output (2×); iree CSEs the duplicate. (`lr`/`W,b,x`
*values* are `skel`-erased, so the render is value-independent — the placeholders
`linTrainStepFaithfulV` passes print the same text as the live graph here.) -/

/-- The emitted `weightSgd` op — consuming the proven `lossCotGraph` node DIRECTLY (the
    fully-tied render) — denotes `linWeightDen` (the certified `sgdW` step). -/
theorem poc_weightSgd_den_eq (lrStr : String) (lr : ℝ) (label : Fin n) :
    den (SHlo.weightSgd "%x" "%W0" lrStr x W lr (lossCotGraph W b x (oneHot n label)))
      = linWeightDen W b x lr label := rfl

/-- The emitted `biasSgd` op — consuming the proven `lossCotGraph` node DIRECTLY — denotes
    `linBiasDen` (the certified bias step). -/
theorem poc_biasSgd_den_eq (lrStr : String) (lr : ℝ) (label : Fin n) :
    den (SHlo.biasSgd "%b0" lrStr b lr (lossCotGraph W b x (oneHot n label)))
      = linBiasDen W b x lr label := rfl

/-- **Tail fold.** For one example `x`, the two emitted tail ops `weightSgd`/`biasSgd` — the
    actual `SHlo` nodes `linTrainStepFaithfulV` prints — denote, respectively,
    `W − lr·∂(crossEntropy ∘ dense)/∂W` and `b − lr·(certified ∂dense/∂b · (softmax − onehot))`. -/
theorem poc_train_step_tail_certified (lrStr : String) (lr : ℝ) (label : Fin n) :
    (∀ (i : Fin m) (j : Fin n),
        den (SHlo.weightSgd "%x" "%W0" lrStr x W lr
              (lossCotGraph W b x (oneHot n label))) (finProdFinEquiv (i, j))
          = W i j - lr * pdiv
              (fun (v : Vec (m * n)) (_ : Fin 1) =>
                  crossEntropy n (dense (Mat.unflatten v) b x) label)
              (Mat.flatten W) (finProdFinEquiv (i, j)) 0)
  ∧ (∀ j : Fin n,
        den (SHlo.biasSgd "%b0" lrStr b lr
              (lossCotGraph W b x (oneHot n label))) j
          = b j - lr * ∑ i : Fin n,
              pdiv (fun b' : Vec n => dense W b' x) b j i
                * (softmax n (mnistLinear W b x) i - oneHot n label i)) := by
  refine ⟨?_, ?_⟩
  · intro i j
    rw [poc_weightSgd_den_eq W b x lrStr lr label]
    exact linWeightDen_is_loss_descent W b x lr label i j
  · intro j
    rw [poc_biasSgd_den_eq W b x lrStr lr label]
    exact linBiasDen_is_certified W b x lr label j

end Proofs.LinPoC
