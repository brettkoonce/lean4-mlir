# Comparator-based independent kernel re-check

This directory holds an end-to-end verification of 110 theorems from the
proof suite using `lake comparator` — Lean's bundled frontend (since 4.35) to
[leanprover/comparator](https://github.com/leanprover/comparator), the
trustworthy-judge tool the Lean Zulip community recommended for projects
that claim "zero project axioms".

`#print axioms` and `lake build` both run via Lean's elaborator, so they
share a trust path with whatever the elaborator did. **comparator
re-runs Lean's kernel typechecker independently and verifies the
transitive axiom closure of each theorem statically**, sandboxed in
`bwrap`. `run.sh` passes `--paranoid`, so five independent checkers that ship
with the toolchain must accept each solution as well: `leanchecker-paranoid`
(Lean's kernel on a hardened allocator), `lean4lean`,
[nanoda](https://github.com/ammkrn/nanoda_lib), `con-leche` and `con-ron`.

## Three challenge files

`Challenge.lean` **imports Mathlib and nothing else.** It holds the 13
architecture-free theorems — chapter 1's `pdiv` calculus rules and chapter 9's
matrix-level rules — with the handful of definitions they need (`Vec`, `Mat`,
`basisVec`, `pdiv`, `pdivMat`, `Mat.{mul,transpose,flatten,unflatten}`) copied
inline. Every one is a one-liner over Mathlib's `fderiv`, and
`chk_pdiv_is_fderiv` pins that by `rfl`. A reviewer can read this file and
know exactly what is claimed without reading a line of this project.

Copying those definitions is safe in a way that copying an architecture would
not be: comparator compares the Challenge and Solution statements
bit-identically, so a copy that drifted would fail the run rather than quietly
prove something weaker.

`ChallengeArch.lean` holds 39 more and **does** import
`LeanMlir.Proofs.*`, because each of them is a statement *about a specific
network* — "ResNet-34's rendered backward equals its Fréchet derivative"
cannot be phrased without ResNet-34 in scope. Making those Mathlib-only would
mean a second copy of every architecture, which must then never drift from the
first; that is a weaker guarantee than an import, not a stronger one.

`ChallengeTier.lean` holds the remaining 35 and is **machine-generated**
(`scripts/gates/gen_comparator_tier.py`). It is the layer above the Jacobians: the step
ties, the codegen faithfulness results, the whole-net back-chains, the
data-parallel results, the float bridge, the descent result and the three
certificate theorems — and the three nets the other two files never mention
(ResNet-34, ResNet-50, MobileNetV4). Its statements run to hundreds of lines
apiece (`cnx_net_tiedGB` alone is 186), so they are printed from each
declaration's own type rather than transcribed: the challenge and the solution
carry the same text by construction, and the solution's proof is the bare
constant.

Its contents are exactly the declarations `formalization.yaml` advertises as the
audited set, minus the four `config-arch.json` already covered. That is the
point of it: **every declaration this project puts forward is independently
kernel-rechecked**, and `scripts/gates/gen_comparator_tier.py --check` fails if a yaml
row stops naming a config that contains it.

The division is the point: once the `fderiv` pin and the structural rules are
checked over Mathlib alone, the architecture theorems are applications of
them, and what a reviewer must additionally trust is the forward functions and
nothing else.

## What gets verified

The 110 theorems span foundation rules (incl. `chk_pdiv_is_fderiv`, which pins
`pdiv` to Mathlib's `fderiv`), every chapter's headline Jacobian, the public
`*HasVJP_correct` wrappers, six whole-network VJPs, and the tie /
faithfulness / certificate tier. The first two buckets are `Challenge.lean`,
the next nine `ChallengeArch.lean`, the last `ChallengeTier.lean`:

| Bucket | Theorems |
|---|---|
| Foundation calculus rules | `pdiv_is_fderiv`, `pdiv_comp`, `pdiv_add`, `pdiv_mul`, `pdiv_id`, `pdiv_const`, `pdiv_reindex`, `pdiv_finset_sum`, `pdivMat_rowIndep` |
| Mat-level structural rules | `pdivMat_comp`, `pdivMat_matmul_left_const`, `pdivMat_scalarScale`, `pdivMat_transpose` |
| Ch 3 MLP | `pdiv_dense`, `pdiv_dense_W`, `pdiv_dense_b`, `denseWeightGrad_correct`, `denseBiasGrad_correct`, `reluHasVJP_correct`, `mlpHasVJP_correct`, `reluHasVJPAt_correct`, `mlpHasVJPAt_correct` |
| Ch 4 CNN | `maxPool2HasVJP3_correct`, `maxPool2HasVJPAt3_correct`, `conv2dHasVJP3_correct`, `globalAvgPoolFlatHasVJP_correct` |
| Ch 5 BN | `pdiv_bnAffine`, `pdiv_bnCentered`, `pdiv_bnIstdBroadcast`, `pdiv_bnNormalize` (the famous 3-term cancellation) |
| Ch 6 Residual | `residualHasVJP_correct`, `residualProjHasVJP_correct` |
| Ch 7 Depthwise | `depthwiseHasVJP3_correct` |
| Ch 8 SE | `seBlockHasVJP_correct` |
| Ch 9 LN+GELU | `pdiv_gelu`, `geluHasVJP_correct`, `layerNormHasVJP_correct` |
| Ch 10 Attention | `pdiv_softmax`, `softmaxCE_grad`, `sdpaBackQ/K/V_correct`, `mhsaHasVJPMat_correct`, `transformerBlockHasVJPMat_correct` |
| Whole-network VJPs | `mnistLinearHasVJP_correct`, `vitFullHasVJP_correct`, `cnnHasVJPAt_correct`, `mobilenetv2HasVJPAt_correct`, `convnextHasVJP{,_at}_correct`, `efficientnetHasVJP{,_at}_correct` |
| **Tier: step ties** | `r50_net_tiedB`, `vit_net_tied`, `cnx_net_tiedGB` |
| **Tier: codegen faithfulness** | `mnv4FwdGraphBFull_faithful`, `convStridedWGradB_den` |
| **Tier: whole-net back-chains** | `resnet50ForwardBFullHasVJPAt_correct`, `r34InputGradB_eq_r34B_full_vjp`, `efficientnetInputGradBFull_correct`, `convnextImagenetInputGradB_eq_vjp`, `vitTinyHasVJP_correct`, `bn_input_grad_correct`, `smoothedCE_grad`, `sealX_nonconstant` |
| **Tier: data parallel** | `dpMeanGrad_ne_globalBatchGrad`, `adamW_at_allReduceMeanF` |
| **Tier: float / descent** | `linear_e4m3_argmax_preserved`, `trained_linear_sgd_strictly_descends` |
| **Tier: certificates** | `lipschitz_margin_certified_radius`, `scorecard_sdp`, `smoothing_certified_radius_classifier`, `shampoo_eq_muon` |

For each, comparator confirms:

1. The Solution's theorem statement is bit-identical to the Challenge's
   (which has `:= by sorry`) — the prover didn't redefine the goal. For the
   Mathlib-only pair this also pins the inlined vocabulary: `Solution.lean`
   carries its own copy, and a divergence between the two would surface here.
2. The Solution's proof uses **only** `propext`, `Quot.sound`,
   `Classical.choice` — no project axioms anywhere in the transitive
   closure.
3. The Solution typechecks against Lean's kernel, re-run from an export of the
   compiled `.olean` independently of the elaborator — and against each of the
   five bundled checkers above.

`reluHasVJPAt_correct`, `mlpHasVJPAt_correct` and
`maxPool2HasVJPAt3_correct` are pointwise (smooth-input) variants whose
underlying `.correct` field is a real proof rather than `rfl` — closing the
kink-rfl-escape at smooth inputs for ReLU, the composed MLP, and MaxPool2. See
`LeanMlir/Proofs/README.md`'s codegen trust boundary section for the math.
⚠ Those three were checked HERE and nowhere else until 2026-09-20: they were
missing from `tests/AuditAxioms.lean`, so the per-push axiom sweep never saw
them. They are in both now.

## Prerequisites (one-time)

- **The project toolchain** (Lean ≥ 4.35). `lake comparator`, the exporter and
  every checker come with it; nothing else is cloned or built.
- **bubblewrap** (`bwrap`) on `PATH`, able to create user namespaces:
  ```
  sudo apt install bubblewrap
  bwrap --ro-bind / / --dev /dev --proc /proc --unshare-all true && echo ok
  ```
  If that fails with a permission error, the distribution restricts
  unprivileged user namespaces; on Ubuntu 24.04 that is
  `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, or install
  `bwrap` setuid root. CI does the former (`comparator.yml`).

## Layout

The six modules are the **root package's** `ComparatorTests` lib
(`lakefile.lean`), not a nested package. `lake comparator` resolves, builds and
exports inside a sandbox that mounts `/` read-only, hides `/home` and `/tmp`,
and lets the build write only the project's `.lake` — so a nested package's
`path = "../.."` require cannot be seen from inside it. As part of the root
package the sandboxed build reuses the workspace's own oleans; build them first
(`lake build Certs ComparatorTests`) or the sandbox compiles the corpus itself.

`Challenge*` and `Solution*` declare the same theorem names, so the two halves
of a pair must never be imported into one environment;
`tests/BlueprintCheckDecls.lean` and `tests/DocstringCheckRefs.lean`, which
import every lib's roots, skip `ComparatorTests` by name.

## Running

```
tests/comparator/run.sh                 # all three configs, --paranoid
tests/comparator/run.sh --no-paranoid   # Lean's kernel only
lake comparator --paranoid --config tests/comparator/config.json   # one config
```

`run.sh` runs the three configs in order from the repo root and stops at the
first failure (`set -e`). Expected output per config:

```
Resolving dependencies
Building Challenge
…
Exporting #[chk_pdiv_comp, …, propext, Quot.sound, Classical.choice, …] from Solution
Running Lean paranoid kernel on solution
…
Running nanoda kernel on solution
nanoda kernel accepts the solution
…
Lean default kernel accepts the solution
Your solution is okay!
[comparator] config.json: 13 theorems OK in … s
```

The `Exporting` line is the audit list. The only allowed axioms are
`propext`, `Quot.sound`, `Classical.choice` (Lean core); everything
else in that list is a `Nat.*` primitive or `String.ofList` (also Lean
core, present in any non-trivial Lean program).

## What's *not* covered

- **The remaining theorems in the proof suite** (downstream compositions,
  `_diff` smoothness lemmas, `_eq_compose` rewrites, the per-leaf ties beneath
  each whole-net chain). `tests/AuditAxioms.lean` prints the axiom closure of
  every one of them on every proof-path push (count its `#print axioms`
  lines); this directory re-checks 110 of them with an independent kernel. The
  gap between the two is deliberate: what the comparator adds is a second, non-elaborator opinion, and
  a second opinion on the advertised set plus the calculus floor it rests on is
  the claim being made. It is not a claim that 110 is all that is proved.
- **`noncomputable def` *witnesses*** themselves like `vitFullHasVJP`,
  `cnnHasVJPAt`, `mhsaLayerHasVJPMat`, etc. comparator's
  `theorem_names` matches `Lean.ConstantInfo.thm`, not `defn`, so the
  witness *defs* aren't run directly — but their public `_correct`
  theorem wrappers (`vitFullHasVJP_correct`, `cnnHasVJPAt_correct`,
  and the per-architecture `*HasVJPAt_correct`) **are** in the suite
  above. `#print axioms` on the underlying defs confirms the same
  allowlist closure for the composition shortcuts.
