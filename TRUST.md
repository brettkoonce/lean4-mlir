# What is proved, and what is trusted

Every claim this repository makes rests on one of the rows below. A sentence that makes a claim
names its row: a theorem and its hypotheses, a guard, a gate and its tolerance, a run directory, or
the word "trusted". The first three row names are the book's
([On Verification](https://lean.brettkoonce.com/blueprint/app-verification.html)); the appendix
is the full argument, and `formalization.yaml` (`fidelity`) is the itemised record.

| Row | What it covers here | Checked by | Cite as |
|---|---|---|---|
| Proven | Layer VJPs, whole-net VJPs, the train-step ties (each emitted update node's `den` is the certified step), data-parallel and sync-BN identities, the certificates. All over ℝ. | Lean's kernel against the three core axioms (`tests/AuditAxioms.lean`); 87 re-checked independently by `tests/comparator/` | the theorem name and its hypotheses |
| By construction | The committed `verified_mlir/` renders are the renderers' output, byte for byte. | the render drift guard in `proofs.yml`; `scripts/gates/check_render_coverage.py` holds the unguarded remainder at its baseline | the artifact and its guard |
| Hypothesis | A condition the theorem assumes and the caller supplies: smooth-input clauses, positivity, the rounding model. | nothing, unless a theorem discharges it (table below) | the hypothesis, beside the theorem |
| Cross-checked | Agreement with an independent computation: `tests/vjp_oracle/` (JAX through IREE), `check_jacobians.py` finite differences, the `*-dp-check` and `*-syncbn-check` gates, `scripts/parity/` against [timm](https://github.com/huggingface/pytorch-image-models). | a script that passes within a stated tolerance | the gate and its tolerance |
| Trusted | Per-op text printing, [XLA/PJRT](https://github.com/openxla/xla) and [IREE](https://github.com/iree-org/iree) compilation, the C runtime in `ffi/`, GPU rounding, reassociation and transcendentals. | nothing in this repository; the cross-checks watch it | "trusted" |
| Measured | Accuracies, ms/step, memory. | the run that produced it | its `runs/<date>-<name>/` directory |

"The tie is proven" and "the gate passed" are different claims; so are "the render is the proof's
text" and "XLA ran that text correctly". One sentence, one row.

## Hypotheses and what discharges them

| Hypothesis | Assumed by | Discharged | Not discharged |
|---|---|---|---|
| Smooth input: `R34SmoothAtB`, `R50SmoothAtB`, the MobileNetV2/V4 relu6/relu clauses | the whole-net VJPs at a point | at each net's seal witness (`seal_smooth`); MobileNetV2/V4's clauses are weight-only | at trained weights on real inputs; the kink convention is `fidelity` item 2 |
| Positivity: `R34PosB`, `R50PosB`, `MNV2PosB`, `B0Weights.EpsPos` | the batched whole-net chains and step ties | at the seal weights (`seal_pos`) for ResNet-34/50 and MobileNetV2 | `B0Weights.EpsPos` at any concrete weights; no net at the shipped constants |
| `FloatModel` (every theorem is `∀ M`) | the float bridges and the SGD-descent chain | `binary32`, `fp8E4M3`: round-to-nearest on a p-bit grid (`rndP_err`) | that the hardware rounds on that grid (Trusted) |
| `FaithfulFloatModel` (subnormal floor) | `FloatSubnormalBridge.lean` only | `exactFaithful` (`rnd = id`) | a binary32 instance; nothing downstream consumes one |
| `FloatClose` magnitude window `A → B` | per-operator closeness and its compositions | per operator, by the caller | a whole-net window (`fidelity` 4c/4d) |
| Smoothed class probability in (0,1) | `smoothing_certified_radius_classifier` | `smoothing_cp_certified_net`, instantiated at the trained pooled `mlpT` (`smoothing_cp_certified_mlpT`) | the 784-dim driver checkpoints |

The declarations Lean treats specially, for an audit:

```bash
rg -n '\baxiom\b|\bopaque\b|@\[implemented_by|@\[extern' LeanMlir -g '*.lean'
```

No `axiom` and no `@[implemented_by]`. Every `opaque` is an `@[extern]` runtime entry point
(`F32Array.lean`, `IreeRuntime.lean`, `Ddpm.lean`, `Verified/Train.lean`) or the runtime's session
handle type (`IreeRuntime.lean`), all on the Trusted row.
