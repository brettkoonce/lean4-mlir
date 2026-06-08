# Verifying the train step — roadmap

**Goal.** Make the bytes `iree-compile` consumes a *proven* function of the
certified gradient, for real models, over `ℝ` — with everything that can't
feasibly be verified (the IREE compiler, Float32 arithmetic, the FFI) named as a
small, explicit trusted computing base (TCB) rather than silently assumed.

"Verify the train step" is not one theorem. It is a chain of links, most of which
are currently trusted or hand-written. The plan is to move the provable ones into
Lean and shrink the rest to a one-paragraph TCB.

## The chain (and where each link stands)

A train step is `θ ↦ θ' = θ − lr·∇_θ L(f_θ(x), y)`. To trust the emitted program:

| # | Link | Statement | Status |
|---|------|-----------|--------|
| 1 | **Math** | `∇_θ L` is the real Jacobian | ✅ over ℝ for input-VJPs; ⚠️ param-VJPs only per-layer, not chained (Crux A) |
| 2 | **AST denotes math** | `denN(trainGraph) = certifiedStep` | ◑ fwd+back graphs for linear/mlp/cnn; ❌ param-grad/SGD not in the AST (Crux B) |
| 3 | **Text is the AST** | `emittedText = render(trainGraph)` | ◑ fwd + cotangent rendered for linear; ❌ grad/SGD tail hand-written |
| 4 | **Text means StableHLO** | `den` agrees with real StableHLO op semantics | ⚠️ per-op token↔text hand-audited; `roundtrip` only covers `skel` through the project's own parser |
| 5 | **Float32 ≈ ℝ** | execution stays near the real algorithm | ❌ trusted; empirically bounded by FD/ULP oracles |
| 6 | **IREE** | `iree-compile` lowers correctly | ❌ trusted (large compiler) |
| 7 | **FFI/runtime** | loads & runs faithfully | ❌ trusted |

Owned-in-Lean target: links 1–3. Link 4 shrunk to a small audited op table.
Links 5–7 are the stated TCB, with empirical backing (`check_jacobians.py`,
`tests/vjp_oracle/`, the cross-backend traces).

## Two cruxes that aren't plumbing

### Crux A — parameter-VJP ≠ input-VJP
The whole-net theorems (`mlp_has_vjp_at`, `cnn_has_vjp_at`, …) prove
`∂output/∂input`. Training needs `∂L/∂Wᵢ` for every layer. The reduction is
standard — `∂L/∂Wᵢ = (cotangent arriving at layer i) ⊗ (cached input to layer i)`
— and the *local* pieces exist (`dense_weight_grad_correct`,
`conv2d_weight_grad_has_vjp`, the batched `wGrad/bGrad_is*Jacobian` in
`StableHLO.lean`). What is missing for anything deeper than `linear` is the
**assembly theorem**: that the per-layer outer products, taken along the
intermediate cotangents the proven backward chain produces, equal `fderiv (L ∘
forward)` w.r.t. the full parameter vector. `linear` has it (one layer, no chain).
MLP/CNN is where the genuinely-new math lives. The codegen already records exactly
those intermediates (`FwdRec`), so the structure matches; only the proof is absent.

### Crux B — the denotation is single-example; the optimizer tail isn't in the AST
`SHlo : Nat → Type` / `den : SHlo n → Vec n` (StableHLO.lean:57, :209) is a
**single-example** semantics. Its constructors cover forward ops and *input*-VJP
ops only. There are **no** constructors for the batched train-step tail — the
weight-grad `dot_general` (contract the batch axis → outer product), the bias-grad
`reduce` (sum over the batch), the scalar `multiply` by `lr`, or the SGD
`subtract`. That is precisely why `mlpTrainStepText`/`linearTrainStepModuleV`
write that tail as hand-assembled `s!"…"` strings (StableHLO.lean:1896, :1954): it
lives in a batched, multi-output regime `den` does not model. The corresponding
math theorems (`wGrad_isWeightJacobian`, `sgdW_descends_certified_grad`, …) are
stated directly over `Mat`/`Vec` under a "per-example shortcut," not as `den` of a
graph.

So closing link 3 needs new **infrastructure**, not just new proofs:

```lean
-- a batched, multi-output module: a list of result subgraphs sharing inputs
-- (each output a flattened Vec), with a denotation into a tuple and a renderer
-- that threads ONE SSA counter and emits a tuple `return`.
def denN  : List (Σ k, SHlo k) → List (Σ k, Vec k)
def renderModuleN (name argSig : String) (B : Nat)
                  (outs : List (Σ k, SHlo k)) : String
-- new constructors (or a small batched layer over SHlo) with den-correctness =
-- the existing op theorems:
--   batchedWeightGrad : den = certified dW   (wGrad_isWeightJacobian, per layer)
--   batchedBiasGrad   : den = certified db   (bGrad_isBiasJacobian)
--   axpyConst lr      : den = θ − lr·g       (sgd*_descends_certified_grad)
```

Because `den` and `render` would then act on the *same* `SHlo` value, `text =
render(graph)` becomes true by construction (the property `renderModule` already
gives for the *forward* graphs `linearFwdModuleV`/`cnnFwdModuleV`).

## M1 — `linear`, status

**Done (this milestone), denotation half.** `LeanMlir/Proofs/LinearTrainStep.lean`
bundles the piecewise linear theorems into one statement per parameter: the
emitted SGD update subtracts `lr` times **[the certified ∂logits/∂θ Jacobian]**
contracted with **[the certified closed-form softmax-CE gradient `softmax −
onehot`]** — every factor a named, axiom-audited certified quantity, no residual
`den`-of-graph, no trusted optimizer step.

* `lossCot_eq_softmax_sub_onehot` — the emitted cotangent `den(lossCotGraph …)` is
  the closed form `softmax(logits) − onehot` (via `lossCotGraph_isCEgrad` +
  `softmaxCE_grad`).
* `sgdW_descends_softmaxCE_grad`, `sgdB_descends_softmaxCE_grad` — the bundled
  weight/bias updates.

3-axiom clean (added to `tests/AuditAxioms.lean`; audit is 160/160 under
`[propext, Classical.choice, Quot.sound]`), builds under the default `Proofs`
target.

**Remaining for `linear`:**

1. **The chain-rule fold (small, gated).** The two-factor sum above is, by
   `pdiv_comp`, the single gradient `∂/∂θ (crossEntropy ∘ mnistLinear)` — i.e.
   literally one SGD step on the loss. Stating it folded needs two `DifferentiableAt`
   facts: for `crossEntropy` at the logits (**no such lemma exists** — log-softmax
   smoothness, a real sub-proof) and for `fun v => dense (Mat.unflatten v) b x` at
   `Mat.flatten W` (affine; provable via a CLM/`fun_prop` argument). Land
   `crossEntropy_differentiable` first; the fold is then ~10 lines.
2. **The rendering half (Crux B infra).** Build `renderModuleN`/`denN` + the three
   batched tail nodes, redefine `linearTrainStepTextV := renderModuleN … (linearTrainStepGraph …)`,
   prove `denN(graph) = certifiedStep`, and retrofit the on-disk file with
   `linearTrainStepText = …TextV := by native_decide` (byte-identical, so
   `verified_mlir/linear_train_step.mlir` is unchanged but now provably
   `render(provenGraph)`). After this, `linear` is end-to-end inside Lean up to the
   links-4–7 TCB.

## Staged roadmap

* **M1 — `linear`** (in progress; denotation done). Smallest tail, already has the
  SGD-descent theorems and a rendered fwd+cotangent prefix. Proves out the whole
  approach.
* **M2 — `MLP`.** Adds Crux A (the multi-layer param-grad assembly theorem — the
  real new math) and the ReLU smooth-point caveat as a hypothesis. Reuses M1's
  module infra.
* **M3 — `MNIST-CNN`.** Conv/maxpool into the train graph; conv param-grad exists,
  so mostly assembly + the maxpool argmax-tie caveat.
* **M4+ — CIFAR/BN → deep nets.** Need their *backward* graph first (Tier 2/3 are
  forward-only today), then the param-grad chain through BN/residual/strided ops.

Everything M1–M3 targets the MNIST-scale models — lightweight to build and run on
CPU/GPU, so each milestone can be validated by an actual `*-verified` training run,
not just `lake build`.

## Strategic fork — unify the two emitters

There are currently **two** emitters: the proven-but-partial `StableHLO.lean` AST,
and the unverified-but-complete `MlirCodegen.lean` (~7500 lines) that produces the
headline numbers. The `native_decide` retrofit connects them per-model but is a
treadmill — the verified one perpetually lags. The durable move is to grow the
proven AST + `renderModuleN` until it covers what `MlirCodegen` emits and either
delete the parallel emitter or generate it *from* the AST, so "the codegen" and
"the proven renderer" are one artifact. Bigger lift; it's the difference between
"N models retrofitted" and "the codegen is verified."

## What stays trusted (and why that's fine)

* **Float32 vs ℝ.** Do *not* attempt a full forward floating-point error analysis
  for training — SGD is robust to per-op ε perturbations, so the marginal assurance
  isn't worth it. State it as a boundary, backed by the FD checks (~1e-11 in f64)
  and the JAX ULP oracles.
* **IREE + FFI.** Unverifiable in practice; cross-checked by the phase-2/phase-3
  differential traces (bit-identical ROCm≡CUDA at step 1). Name them in the TCB.

**Finish line:** links 1–3 proven in Lean, link 4 an op-by-op StableHLO-token
table, links 5–7 a stated four-item TCB with empirical backing — first for
`linear` (M1), then MLP/CNN, with emitter unification as the backbone if it should
scale past hand-picked models.
