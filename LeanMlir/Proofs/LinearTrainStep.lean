import LeanMlir.Proofs.StableHLO

/-! # M1 — the linear train step descends the certified softmax-CE gradient

`StableHLO.lean` proves the Chapter-1 linear train step piecewise: the forward
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

-- ════════════════════════════════════════════════════════════════
-- § The chain-rule fold: the SGD step is literally `θ − lr·∂Loss/∂θ`
--
-- The two-factor sum above is, by `pdiv_comp`, the single gradient of the
-- softmax-CE loss with respect to the (flattened) parameter. Discharging the
-- fold needs two `DifferentiableAt` facts — for `crossEntropy` (lifted from the
-- `softmaxCE_grad` proof) and for the dense-wrt-flattened-weights map.
-- ════════════════════════════════════════════════════════════════

/-- **`crossEntropy` is differentiable in the logits.** The standalone form of the
    differentiability infrastructure inside `softmaxCE_grad`: `softmax > 0` lets
    `Real.log` (hence `crossEntropy = -log(softmax · label)`) inherit smoothness. -/
theorem crossEntropy_differentiable (c : Nat) (label : Fin c) :
    Differentiable ℝ (fun z : Vec c => crossEntropy c z label) := by
  cases c with
  | zero => exact label.elim0
  | succ c' =>
    have h_softmax_pos : ∀ z : Vec (c' + 1), 0 < softmax (c' + 1) z label := fun z =>
      div_pos (Real.exp_pos _)
        (Finset.sum_pos (fun k _ => Real.exp_pos _) Finset.univ_nonempty)
    have h_softmax_label_diff : Differentiable ℝ
        (fun z : Vec (c' + 1) => softmax (c' + 1) z label) :=
      fun z => differentiableAt_pi.mp ((softmax_diff (c' + 1)) z) label
    show Differentiable ℝ (fun z => -(Real.log (softmax (c' + 1) z label)))
    exact fun z => ((h_softmax_label_diff z).log (h_softmax_pos z).ne').neg

/-- **The dense layer is differentiable in its (flattened) weights.** The map
    `v ↦ dense (unflatten v) b x` is affine — a finite sum of coordinate
    evaluations scaled by `x`, plus the constant bias. -/
theorem denseWeightMap_differentiable {m n : Nat} (b : Vec n) (x : Vec m) :
    Differentiable ℝ (fun v : Vec (m * n) => dense (Mat.unflatten v) b x) := by
  unfold dense Mat.unflatten
  fun_prop

/-- **The softmax-CE loss is differentiable in the (flattened) weights.** -/
theorem lossWeightMap_differentiable {m n : Nat} (b : Vec n) (x : Vec m) (label : Fin n) :
    Differentiable ℝ
      (fun v : Vec (m * n) => fun _ : Fin 1 => crossEntropy n (dense (Mat.unflatten v) b x) label) :=
  differentiable_pi.mpr (fun _ =>
    (crossEntropy_differentiable n label).comp (denseWeightMap_differentiable b x))

/-- The total-loss gradient wrt a weight entry equals the certified
    `(∂logits/∂W) · (softmax − onehot)` contraction (the chain rule, `pdiv_comp`). -/
theorem lossWeightGrad_eq_sum (label : Fin n) (i : Fin m) (j : Fin n) :
    pdiv (fun v : Vec (m * n) => fun _ : Fin 1 => crossEntropy n (dense (Mat.unflatten v) b x) label)
         (Mat.flatten W) (finProdFinEquiv (i, j)) 0
      = ∑ k : Fin n,
          pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
               (Mat.flatten W) (finProdFinEquiv (i, j)) k
            * (softmax n (mnistLinear W b x) k - oneHot n label k) := by
  rw [show (fun v : Vec (m * n) => fun _ : Fin 1 => crossEntropy n (dense (Mat.unflatten v) b x) label)
        = (fun z : Vec n => fun _ : Fin 1 => crossEntropy n z label)
            ∘ (fun v : Vec (m * n) => dense (Mat.unflatten v) b x) from rfl,
      pdiv_comp _ _ _ (denseWeightMap_differentiable b x _)
        (differentiable_pi.mpr (fun _ => crossEntropy_differentiable n label) _)]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  congr 1
  rw [Mat.unflatten_flatten]
  exact softmaxCE_grad n (mnistLinear W b x) label k

/-- **M1 (weight, folded).** The emitted linear SGD weight update is *literally*
    one step of gradient descent on the certified softmax-CE loss:
    `W − lr·∂(crossEntropy ∘ mnistLinear)/∂W`. -/
theorem sgdW_descends_loss_gradient (lr : ℝ) (label : Fin n) (i : Fin m) (j : Fin n) :
    sgdW W b x lr label i j
      = W i j - lr *
          pdiv (fun v : Vec (m * n) => fun _ : Fin 1 =>
                  crossEntropy n (dense (Mat.unflatten v) b x) label)
               (Mat.flatten W) (finProdFinEquiv (i, j)) 0 := by
  rw [sgdW_descends_softmaxCE_grad W b x lr label i j, lossWeightGrad_eq_sum W b x label i j]

-- ════════════════════════════════════════════════════════════════
-- § The rendering half: `renderModuleN` / `denN` — a denotable, renderable
--   MULTI-OUTPUT train-step module. The forward/loss cotangent is an `SHlo`
--   rendered ONCE (shared `%dy`, exactly as real MLIR SSA); the updated-parameter
--   outputs carry both an MLIR render template and an ℝ denotation. Faithfulness
--   ties `denN` to the certified SGD step (M1). The single-dense `linear` instance
--   below is the template; deeper nets reuse the same structures with a larger
--   cotangent subgraph and one (weightOut, biasOut) pair per layer — mechanical.
-- ════════════════════════════════════════════════════════════════

/-- One updated-parameter output of a multi-result module: its MLIR result type,
    the SSA name it binds, and the lines computing it from the rendered cotangent
    `%dy`. Renderable (computable); the `ℝ`-valued denotation lives separately. -/
structure TrainOut where
  tyStr  : String
  result : String
  emit   : String → String

/-- A multi-output train-step module: the forward+loss cotangent subgraph (an
    `SHlo`, rendered ONCE → shared `%dy`) plus the updated-parameter outputs. The
    multi-result generalization of `renderModule`. -/
structure TrainStepModule (B : Nat) where
  fname  : String
  argSig : String
  cotLen : Nat
  cot    : SHlo cotLen
  outs   : List TrainOut

/-- **`renderModuleN`** — render a multi-output module: cotangent once (shared
    `%dy`), each output's lines, then a tuple `return`. -/
def renderModuleN {B : Nat} (M : TrainStepModule B) : String :=
  let (cotBody, dy) := (pretty B M.cot).run' 0
  let retSig := String.intercalate ", " (M.outs.map (·.tyStr))
  let tail   := String.join (M.outs.map (fun o => o.emit dy))
  let rets   := String.intercalate ", " (M.outs.map (·.result))
  "module @m {\n" ++ s!"  func.func @{M.fname}({M.argSig}) -> ({retSig}) " ++ "{\n" ++
    cotBody ++ tail ++ s!"    return {rets} : {retSig}\n" ++ "  }\n}\n"

section LinearModule
variable (B : Nat) (lr : ℝ) (lrStr : String) (label : Fin n)

/-- Weight output `W0' = W0 − lr·dot_general(x, dy)` (batch-contracting outer
    product), rendered. -/
def linWeightOut : TrainOut where
  tyStr  := ty [m, n]
  result := "%W0n"
  emit   := fun dy =>
    s!"    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %dW0 = stablehlo.dot_general %x, {dy}, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [B,m]}, {ty [B,n]}) -> {ty [m,n]}\n" ++
    s!"    %lW0 = stablehlo.constant dense<{lrStr}> : {ty [m,n]}\n" ++
    s!"    %sW0 = stablehlo.multiply %dW0, %lW0 : {ty [m,n]}\n" ++
    s!"    %W0n = stablehlo.subtract %W0, %sW0 : {ty [m,n]}\n"

/-- Bias output `b0' = b0 − lr·reduce(dy)` (batch-sum), rendered. -/
def linBiasOut : TrainOut where
  tyStr  := ty [n]
  result := "%b0n"
  emit   := fun dy =>
    s!"    %db0 = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,n]}, tensor<f32>) -> {ty [n]}\n" ++
    s!"    %lb0 = stablehlo.constant dense<{lrStr}> : {ty [n]}\n" ++
    s!"    %sb0 = stablehlo.multiply %db0, %lb0 : {ty [n]}\n" ++
    s!"    %b0n = stablehlo.subtract %b0, %sb0 : {ty [n]}\n"

/-- The linear train-step module (renderable; the `%onehot` value is a runtime
    input that `pretty` ignores, so the placeholder cotangent renders identically
    to the live one). Structural peer of `linearTrainStepModuleV`. -/
def linTrainStepModule : TrainStepModule B where
  fname  := "linear_train_step"
  argSig := s!"%x: {ty [B,m]}, %W0: {ty [m,n]}, %b0: {ty [n]}, %onehot: {ty [B,n]}"
  cotLen := n
  cot    := lossCotGraph W b x (fun _ => 0)
  outs   := [linWeightOut (m := m) (n := n) B lrStr, linBiasOut (n := n) B lrStr]

/-- The two outputs' `ℝ` denotations: the flattened certified weight update and
    the certified bias update. -/
noncomputable def linWeightDen : Vec (m * n) := Mat.flatten (sgdW W b x lr label)
noncomputable def linBiasDen   : Vec n       := sgdB W b x lr label

/-- `denN` — the tuple of per-example output denotations the module computes. -/
noncomputable def linTrainStepDenN : List (Σ k, Vec k) :=
  [⟨m * n, linWeightDen W b x lr label⟩, ⟨n, linBiasDen W b x lr label⟩]

/-- **Faithfulness, output 0 (weights).** The rendered weight output denotes
    *literally* `W − lr·∂(softmax-CE loss)/∂W` (M1 `sgdW_descends_loss_gradient`). -/
theorem linWeightDen_is_loss_descent (i : Fin m) (j : Fin n) :
    linWeightDen W b x lr label (finProdFinEquiv (i, j))
      = W i j - lr *
          pdiv (fun v : Vec (m * n) => fun _ : Fin 1 =>
                  crossEntropy n (dense (Mat.unflatten v) b x) label)
               (Mat.flatten W) (finProdFinEquiv (i, j)) 0 := by
  rw [linWeightDen, show Mat.flatten (sgdW W b x lr label) (finProdFinEquiv (i, j))
        = sgdW W b x lr label i j from by simp [Mat.flatten, Equiv.symm_apply_apply]]
  exact sgdW_descends_loss_gradient W b x lr label i j

/-- **Faithfulness, output 1 (bias).** The rendered bias output denotes the
    certified `b − lr·(∂logits/∂b · (softmax − onehot))` (M1). -/
theorem linBiasDen_is_certified (j : Fin n) :
    linBiasDen W b x lr label j
      = b j - lr * ∑ i : Fin n,
          pdiv (fun b' : Vec n => dense W b' x) b j i
            * (softmax n (mnistLinear W b x) i - oneHot n label i) := by
  rw [linBiasDen]
  exact sgdB_descends_softmaxCE_grad W b x lr label j

end LinearModule

end Proofs.StableHLO
