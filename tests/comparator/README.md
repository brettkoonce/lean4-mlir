# Comparator-based independent kernel re-check

This directory holds an end-to-end verification of 73 theorems from the
proof suite using
[leanprover/comparator](https://github.com/leanprover/comparator) — the
trustworthy-judge tool the Lean Zulip community recommended for projects
that claim "zero project axioms".

`#print axioms` and `lake build` both run via Lean's elaborator, so they
share a trust path with whatever the elaborator did. **comparator
re-runs Lean's kernel typechecker independently and verifies the
transitive axiom closure of each theorem statically**, sandboxed in
landlock, with the option to pile on the
[nanoda](https://github.com/ammkrn/nanoda_lib) kernel as a second
opinion.

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

`ChallengeTier.lean` holds the remaining 21 and is **machine-generated**
(`scripts/gen_comparator_tier.py`). It is the layer above the Jacobians: the step
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
kernel-rechecked**, and `scripts/gen_comparator_tier.py --check` fails if a yaml
row stops naming a config that contains it.

The division is the point: once the `fderiv` pin and the structural rules are
checked over Mathlib alone, the architecture theorems are applications of
them, and what a reviewer must additionally trust is the forward functions and
nothing else.

## What gets verified

The 73 theorems span foundation rules (incl. `chk_pdiv_is_fderiv`, which pins
`pdiv` to Mathlib's `fderiv`), every chapter's headline Jacobian, the public
`*_has_vjp_correct` wrappers, six whole-network VJPs, and the tie /
faithfulness / certificate tier. The first two buckets are `Challenge.lean`,
the next nine `ChallengeArch.lean`, the last `ChallengeTier.lean`:

| Bucket | Theorems |
|---|---|
| Foundation calculus rules | `pdiv_is_fderiv`, `pdiv_comp`, `pdiv_add`, `pdiv_mul`, `pdiv_id`, `pdiv_const`, `pdiv_reindex`, `pdiv_finset_sum`, `pdivMat_rowIndep` |
| Mat-level structural rules | `pdivMat_comp`, `pdivMat_matmul_left_const`, `pdivMat_scalarScale`, `pdivMat_transpose` |
| Ch 3 MLP | `pdiv_dense`, `pdiv_dense_W`, `pdiv_dense_b`, `dense_weight_grad_correct`, `dense_bias_grad_correct`, `relu_has_vjp_correct`, `mlp_has_vjp_correct`, `relu_has_vjp_at_correct`, `mlp_has_vjp_at_correct` |
| Ch 4 CNN | `maxPool2_has_vjp3_correct`, `maxPool2_has_vjp_at3_correct`, `conv2d_has_vjp3_correct`, `globalAvgPoolFlat_has_vjp_correct` |
| Ch 5 BN | `pdiv_bnAffine`, `pdiv_bnCentered`, `pdiv_bnIstdBroadcast`, `pdiv_bnNormalize` (the famous 3-term cancellation) |
| Ch 6 Residual | `residual_has_vjp_correct`, `residualProj_has_vjp_correct` |
| Ch 7 Depthwise | `depthwise_has_vjp3_correct` |
| Ch 8 SE | `seBlock_has_vjp_correct` |
| Ch 9 LN+GELU | `pdiv_gelu`, `gelu_has_vjp_correct`, `layerNorm_has_vjp_correct` |
| Ch 10 Attention | `pdiv_softmax`, `softmaxCE_grad`, `sdpa_back_Q/K/V_correct`, `mhsa_has_vjp_mat_correct`, `transformerBlock_has_vjp_mat_correct` |
| Whole-network VJPs | `mnistLinear_has_vjp_correct`, `vit_full_has_vjp_correct`, `cnn_has_vjp_at_correct`, `mobilenetv2_has_vjp_at_correct`, `convnext_has_vjp{,_at}_correct`, `efficientnet_has_vjp{,_at}_correct` |
| **Tier: step ties** | `r50_net_tiedB`, `vit_net_tied_certified`, `cnx_net_tiedGB` |
| **Tier: codegen faithfulness** | `mnv4FwdGraphB_full_faithful`, `convStridedWGradB_den` |
| **Tier: whole-net back-chains** | `resnet50ForwardB_full_has_vjp_at_correct`, `r34InputGradB_eq_r34B_full_vjp`, `efficientnetInputGradB_full_correct`, `convnextImagenetInputGradB_eq_vjp`, `vitTiny_has_vjp_correct`, `bn_input_grad_correct`, `smoothedCE_grad`, `mnv2Live_forward_nonconstant` |
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
3. The Solution typechecks against Lean's kernel, re-run from the
   compiled `.olean` independently of the elaborator.

`relu_has_vjp_at_correct`, `mlp_has_vjp_at_correct` and
`maxPool2_has_vjp_at3_correct` are pointwise (smooth-input) variants whose
underlying `.correct` field is a real proof rather than `rfl` — closing the
kink-rfl-escape at smooth inputs for ReLU, the composed MLP, and MaxPool2. See
`LeanMlir/Proofs/README.md`'s codegen trust boundary section for the math.
⚠ Those three were checked HERE and nowhere else until 2026-09-20: they were
missing from `tests/AuditAxioms.lean`, so the per-push axiom sweep never saw
them. They are in both now.

## Prerequisites (one-time)

- **A kernel with Landlock.** ⚠ This used to read "Linux kernel ≥ 6.10 for
  Landlock ABI v5", as a hard requirement. It is not one: comparator passes
  `--best-effort` to landrun unconditionally (`Main.lean:83`), so landrun
  enforces whatever ABI the running kernel offers instead of refusing.
  **Measured 2026-09-20 on Ubuntu 24.04.4 / kernel 6.8.0-138 / Landlock ABI 4:
  all three configs run to `Your solution is okay!`** — the kernel re-check and
  the axiom-closure check are unaffected, because neither depends on the
  sandbox.

  What the older kernel costs is sandbox STRENGTH, and it costs it silently:
  landrun prints nothing about the downgrade, so a green run here is not
  evidence that the ABI v5 profile held. If the sandbox is what you are
  relying on — a judge you do not trust not to be tampered with — get v5:
  ```
  uname -r                                        # what you have
  python3 -c 'import ctypes; print(ctypes.CDLL(None).syscall(444, None, 0, 1))'   # ABI version
  sudo apt install linux-image-generic-hwe-24.04  # 24.04: 7.0 as of 2026-09
  sudo reboot
  ```
- **landrun** ≥ v0.1.13 — sandbox runner using Linux Landlock LSM:
  ```
  curl -fsSL -o ~/.local/bin/landrun \
    https://github.com/Zouuup/landrun/releases/latest/download/landrun-linux-amd64
  chmod +x ~/.local/bin/landrun
  ```
- **lean4export** — Lean's `.olean` → text exporter:
  ```
  git clone --depth=1 https://github.com/leanprover/lean4export ~/lean4export
  (cd ~/lean4export && lake build)
  ln -sf ~/lean4export/.lake/build/bin/lean4export ~/.local/bin/lean4export
  ```
- **comparator** itself:
  ```
  git clone --depth=1 https://github.com/leanprover/comparator ~/comparator
  (cd ~/comparator && lake build)
  ln -sf ~/comparator/.lake/build/bin/comparator ~/.local/bin/comparator
  ```

`landrun`, `lean4export`, and `comparator` must be on `PATH` (they
should be after the `ln -sf` lines above if `~/.local/bin` is on PATH).
Alternatively comparator reads `COMPARATOR_LANDRUN`, `COMPARATOR_LEAN4EXPORT`
and `COMPARATOR_NANODA` (`Main.lean:286-288`) — pointing those at absolute paths
is tidier than the `PATH` juggling `run.sh` does, and would let the landrun shim
below be passed directly rather than shadowed onto `PATH`.

⚠ The v0.1.14 release binary self-reports `landrun version 0.1.13`; that is
upstream's metadata, not a wrong download. It accepts comparator's single-dash
`-ldd` / `-add-exec`, so the shim's flag translation is a no-op against this
version — it is kept for older ones. The `--rox /usr` half is still needed.

## Running

```
./run.sh
```

`run.sh` runs all three configs in order and stops at the first failure
(`set -e`). Expected output (full Mathlib decompress on the first run,
~5 minutes; seconds on subsequent runs):

```
[1/3] lake update (resolving Mathlib)…
[2/3] lake build Solution (outside sandbox)…
[3/3] lake env comparator config.json…
Building Challenge
…
Exporting #[chk_pdiv_comp, …, propext, Quot.sound, Classical.choice, …] from Solution
Running Lean default kernel on solution.
Lean default kernel accepts the solution
Your solution is okay!
```

The `Exporting` line is the audit list. The only allowed axioms are
`propext`, `Quot.sound`, `Classical.choice` (Lean core); everything
else in that list is a `Nat.*` primitive or `String.ofList` (also Lean
core, present in any non-trivial Lean program).

## Notes on the wrapper hack

`run.sh` injects a shim around `landrun` that does two things:

1. Translates single-dash flags `-ldd` and `-add-exec` (which comparator
   emits) into the double-dash form (`--ldd`, `--add-exec`) that
   landrun's CLI parser expects.
2. Prepends `--rox /usr` to the sandbox config — comparator only allows
   `/usr/bin/git`, but `lake` does generic path lookups in `/usr` and
   needs that broader exec permission.

Both look like upstream comparator/landrun polish items; the workaround
keeps the audit reproducible until they land.

## What's *not* covered

- **The remaining theorems in the proof suite** (downstream compositions,
  `_diff` smoothness lemmas, `_eq_compose` rewrites, the per-leaf ties beneath
  each whole-net chain). `tests/AuditAxioms.lean` prints the axiom closure of
  all 1,377 of them on every proof-path push; this directory re-checks 73 of
  them with an independent kernel. The gap between those two numbers is
  deliberate: what the comparator adds is a second, non-elaborator opinion, and
  a second opinion on the advertised set plus the calculus floor it rests on is
  the claim being made. It is not a claim that 73 is all that is proved.
- **`noncomputable def` *witnesses*** themselves like `vit_full_has_vjp`,
  `cnn_has_vjp_at`, `mhsa_layer_has_vjp_mat`, etc. comparator's
  `theorem_names` matches `Lean.ConstantInfo.thm`, not `defn`, so the
  witness *defs* aren't run directly — but their public `_correct`
  theorem wrappers (`vit_full_has_vjp_correct`, `cnn_has_vjp_at_correct`,
  and the per-architecture `*_has_vjp_at_correct`) **are** in the suite
  above. `#print axioms` on the underlying defs confirms the same
  allowlist closure for the composition shortcuts.
- **nanoda second-kernel re-check.** Set `enable_nanoda: true` in
  `config.json` and add nanoda to PATH (Rust build, ~5 min) for that
  upgrade.
