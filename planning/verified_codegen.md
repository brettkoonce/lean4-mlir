# verified_codegen.md — a small verified backprop codegen: MLP first, then expand

Closing **R3** (the mirror-drift gap from `typed_ir.md`): today the
`Back`/`Back3` IR is hand-written to *look like* what `MlirCodegen.lean`
emits, with no formal link. The fix is to make the emitted StableHLO **be
the rendering of the proof-backed IR**, so `mlp_whole_bridge` (and friends)
cover the actual artifact, not a mirror.

Strategy: **don't retarget the monolith** (`generateTrainStep` is ~7.5k
lines and emits forward+loss+backward+optimizer as one fused, hand-SSA'd
string — rewriting its backward risks the working trainers). Instead build
a **parallel, self-contained verified pipeline for one network at a time**,
where the emitted text *is* `print (emitBack net)` by construction (so R3
is closed for that net, no string-diffing), validated on IREE. **MLP first,
land it, then expand.**

Status: **MLP fully landed (Phases 0–2 + Phase-4 param-grad/SGD and loss
slices); CNN landed through the whole backward chain (Phase 3).** The MLP
train step (forward → loss → backward → grads) and the CNN input-gradient
backward (conv → relu → maxpool → flatten → dense) are both renderings of
proof-backed IR, GPU-validated on rocm/gfx1100. Remaining: CNN weight-gradient
+ a full CNN train step; optionally SGD-IR / migrating `generateTrainStep`.

## Done (MLP)

- **Phase 0** — `IRPrint.lean` renders the backward graph to StableHLO
  (`Hlo` mirror of `Back`); `mlpModule`/`linearModule` emit full `func.func`s.
- **Phase 1** — `check_ir_codegen.py` compiles them on IREE and runs them on
  CPU (`llvm-cpu`) **and the real GPU** (`rocm/gfx1100`, Radeon RX 7900 XTX),
  matching an independent numpy VJP to ~1e-7. R3 closed for the MLP backward,
  GPU-validated.
- **Param grads + SGD (slice of Phase 4)** — `mlpTrainStepModule` renders a
  full SGD step: proof-backed backward (the dx chain
  `⟦emitMlpBack⟧ = mlp_has_vjp_at.backward`, plus `dWℓ`/`dbℓ` =
  `IR.emitWeightGrad`/`emitBiasGrad`, bridged to the certified Jacobians by
  `weight_grad_bridge`/`bias_grad_bridge`) → trusted elementwise SGD. The six
  updated parameters match numpy SGD to 0.0 on CPU and 1.19e-7 on the GPU —
  the proof-backed gradients drive a correct weight update on real hardware.
- **Phase 2** — forward IR `Fwd` + `mlp_fwd_bridge` (`⟦emitMlpFwd⟧ =
  mlpForward`) + `mlp_fwd_preact0/1` (the forward sub-graphs denote exactly the
  pre-activations the backward reads — the splice contract, proven). Printer:
  `HloF` (forward AST, peer of `Hlo`) renders the forward; the standalone
  `@mlp_fwd` module and the train step's forward are now `render(mlpFwd…)`,
  matching numpy forward to 2.38e-7. The forward is no longer trusted: it is
  the rendering of an IR proven `= mlpForward`.
- **Loss cotangent (rest of Phase 4)** — `IR.emitLossCot` + `lossCot_bridge`
  (`⟦emitLossCot⟧ = ∂(crossEntropy)/∂logits = softmax(logits) − onehot`, via
  the repo's `softmaxCE_grad`). Printer: `renderLossCot` emits the softmax head
  (`exp`+`reduce`+`broadcast`+`divide`) + `subtract` target; the standalone
  `@loss_cot` module (5.96e-8 vs numpy) and the train step now COMPUTE the
  cotangent in-module from the logits + target distribution (input changed from
  `%dy` to `%onehot`). So the cotangent is proof-backed too — the train step is
  `forward → loss → backward → grads`, every mathematical op proof-backed; only
  the SGD arithmetic stays trusted. Train step: 1.19e-7 on CPU and GPU.
- **Phase 3 (CNN)** — the proof-backed pattern generalized past dense/relu to
  real spatial ops. The repo's `conv2d` is SAME-pad stride-1 cross-correlation
  = `stablehlo.convolution`; `maxPool2` = `reduce_window`(max). The proven
  backwards render as: conv input-VJP = `transpose`+`reverse`+`convolution`
  (`conv_back_bridge_1to2`, the reversed-kernel identity); maxpool VJP =
  `select_and_scatter` (`maxpool_back_bridge`, route dy to argmax, GE tie-break
  = the no-tie smooth-point hypothesis). `cnnModule` then composes the lot —
  `conv → relu → maxpool → flatten → dense` forward + the full input-gradient
  backward (`dense_back → reshape → maxpool_back → relu_back → conv_back`), the
  reshape proof-faithful because `Tensor3.flatten` is C-order = `stablehlo
  .reshape`. Validated vs an independent numpy CNN VJP: conv 1.91e-6, maxpool
  exact, the whole `cnn_back` chain 9.54e-7 — on CPU and GPU.

## The pipeline

```
NetSpec ──▶ forward MLIR        (Phase 0: reuse existing emitters / hand)
        ──▶ loss MLIR           (reuse)
        ──▶ Back IR  ──print──▶ backward MLIR     ← the verified piece
            (⟦·⟧ = proven VJP, mlp_whole_bridge)
        ──▶ optimizer MLIR      (reuse)
                 │
                 ▼  splice into one func, hand to IREE → GPU → train
```

The **only new trusted component is the printer** `Back.toStablehlo`. The
proof gives `⟦emitBack⟧ = VJP`; the printer renders `emitBack` to text
trusted (validated numerically, not proven — see R4) to denote `⟦·⟧`.

## The splice contract (why this is the crux, not the printer ops)

The backward isn't standalone: it consumes forward-saved activations and
produces the parameter gradients the optimizer expects. So the rendered
backward must wire into the existing forward/optimizer by SSA name:

- **in:** the initial cotangent (`d_logits` from the loss), plus each
  layer's saved forward intermediate the backward reads (the dense weights
  `%W*`; the ReLU pre-activations for the `compare`/`select` masks).
- **out:** `%dW0, %dW1, …` (and bias grads) in the names the optimizer block
  consumes.

Match this contract and the rendered backward drops into the existing
train-step unchanged.

## The printer — key design decisions

**D1 — The `Back` IR holds *abstract* `Vec`/`Mat` (`Fin n → ℝ`,
noncomputable). The printer cannot read values from it.** It walks the
*structure* (which constructor) to choose ops, and takes **operand SSA
names from an external wiring map** (the forward pass exports them). The
held `Vec`s are for `denote`/proofs only; codegen never inspects them.
→ Phase 0: supply the wiring explicitly (the MLP has few operands: 3
weights, 2 ReLU pre-activations). A later phase can formalize a
codegen-IR ↔ math-IR refinement if we want the wiring itself checked.

**D2 — SSA generation.** Thread an `Nat` counter (`StateM Nat`), emit
`%bk0, %bk1, …`. Expression-tree `Back` has no sharing; the printer emits
one SSA per node (a `let`-free tree → a straight-line SSA sequence). Sharing
is an optional later optimization (CSE on the printed sequence); it doesn't
affect semantics.

**D3 — op mapping** (`Back` node → StableHLO):

| `Back` | StableHLO |
|---|---|
| `dotGeneral W` | `stablehlo.dot_general` (operand `%W`) |
| `scale s` | `stablehlo.multiply` (operand `%s` — a saved activation) |
| `scaleConst c` | `stablehlo.multiply` by `stablehlo.constant` |
| `sumBroadcast` | `stablehlo.reduce`(add) + `broadcast_in_dim` |
| `sub` / `add` | `stablehlo.subtract` / `add` |
| `selectPos x` | `stablehlo.compare GT 0` (on `%x`) + `stablehlo.select` |
| `cotangent` | the input SSA |

(`Back3`: `conv` → `convolution` on the `reverse`d/transposed kernel;
`maxpool` → tile-`compare`-`select`. Phase 3.)

**D4 — forward.** Phase 0 reuses the existing string forward (trusted) so we
get a runnable module fast. Forward-IR + a forward bridge
(`⟦fwdIR⟧ = mlpForward`) is Phase 2 — then the *whole* module is proof-backed,
not just the backward.

## Phases

| Phase | Scope | Who | Status |
|---|---|---|---|
| **0** | `Back.toStablehlo` printer (Vec ops: dot_general/multiply/reduce/broadcast/sub/add/select) + render the MLP backward (`emitMlpBack`) → one `.mlir`. | Lean-only. | ✅ `IRPrint.lean` |
| **1** | **Land it.** Compile on IREE, run the MLP backward, confirm it matches the numpy VJP. R3 closed for the MLP backward, validated on GPU. | GPU. | ✅ CPU + rocm/gfx1100 |
| **4 (param-grad + SGD slice)** | `dWℓ`/`dbℓ` emitters + bridges + a full SGD `mlpTrainStepModule`; run a train step where the weights move correctly. | Lean + GPU. | ✅ CPU 0.0 / GPU 1.19e-7 |
| **2** | Forward IR + `⟦fwdIR⟧ = mlpForward` bridge → the *whole* MLP module is proof-backed (forward + backward), not just backward. | Lean + IREE. | ✅ `Fwd`/`HloF`, mlp_fwd 2.38e-7 |
| **4 (loss-cotangent slice)** | Loss IR (softmax-CE cotangent) instead of the supplied `%dy` — `emitLossCot`/`lossCot_bridge` + `renderLossCot`; train step computes dy in-module. | Lean + GPU. | ✅ loss_cot 5.96e-8 |
| **3** | conv `convolution` + maxpool `reduce_window`/`select_and_scatter` + flatten reshape → a small **CNN** backward chain end-to-end. | Lean + GPU. | ✅ cnn_back 9.54e-7 |
| **3 (rest)** | CNN weight-gradient (conv dW = a convolution) + a full CNN train step (param grads + SGD). | Lean + GPU. | next |
| **4 (rest)** | SGD-IR (make the optimizer step proof-backed too); then decide whether to migrate `generateTrainStep` or keep the parallel path as the verified reference. | Optional. | — |

## R3 closure & residual trust, per phase

- After **Phase 1**: the StableHLO *backward* that runs is `print(emitMlpBack)`,
  and `⟦emitMlpBack⟧ = mlp_has_vjp_at.backward` (proven). So the running
  backward is proof-backed **up to**: the printer (faithful rendering —
  trusted/tested, not proven; proving it needs a StableHLO *text* semantics =
  R4), IREE/XLA lowering, and float. Forward/loss/optimizer still reused
  (trusted) until Phase 2/4.
- The honest end-state claim (Phase 2, MLP): *"the emitted forward and
  backward StableHLO are renderings of IRs whose denotations are the proven
  forward map and its exact VJP; validated to match on GPU; trusted below
  the printer + IREE + float."* Far past "trust the comment."

## Risks

- **R-wiring.** The splice contract (matching SSA names + the optimizer's
  expected gradient names) is the fiddly part. Mitigation: start with a
  1-layer MLP (one weight, no ReLU) — minimal wiring — then 2-layer (one
  ReLU mask), then full.
- **R-printer-faithfulness.** Trusted, validated only numerically. To
  *prove* it, you'd need a formal StableHLO text semantics (out of scope;
  this is the irreducible R4 surface, now centralized in one ~200-line
  printer instead of scattered across 7.5k lines of `s!"..."`).
- **R-IREE-shape.** The printed StableHLO must be exactly IREE-acceptable
  (dim numbers, layouts). Mitigation: diff against a known-good emitted
  module for the same op; iterate on Phase 1.
- **R-regression.** None to the production path — this is parallel and
  touches nothing in `generateTrainStep` until Phase 4.

## Success criteria

- Phase 0: a complete `.mlir` whose backward section is `print(emitMlpBack)`,
  type-checks/parses.
- Phase 1: it compiles on IREE and a train step's gradients match the oracle
  to FD tolerance — i.e., the proof-backed backward runs correctly on GPU.
- Phase 2: forward bridge lands; whole MLP module proof-backed.
- Phase 3: a small CNN rendered end-to-end the same way.

## Strategic note

For the **book**, what already exists (per-op bridges + `mlp_whole_bridge` +
numerical oracle) is a strong "formal spec of the codegen" chapter. This
plan is the **research artifact**: a backprop codegen whose output is
proof-backed and GPU-validated — a genuine "verified deep learning" claim
(up to printer/IREE/float). Phases 0–1 on the MLP are the proof of concept;
everything after is scaling the same pattern.

## Immediate next step

**Phase 0**, MLP, smallest first: the `Back.toStablehlo` printer + render
`emitMlpBack` (start 1-layer to nail the wiring, then 2-layer) into a full
module. Lean-only; produces a `.mlir` to throw at IREE in Phase 1.
