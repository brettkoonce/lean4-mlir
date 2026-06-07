# Validation-sweep refactor — VerifiedNetSpec + the spec→MLIR ladder

Handoff for migrating the `*-verified` trainers onto a declarative `VerifiedNetSpec`
and tying each spec to its math + its generated MLIR. ch2–5 done; imagenette nets next
(order: r34 → mnv2 → efficientnet → convnext → vit). Keep context lean: this doc + the
per-net pointers below should be enough — don't read whole proof files, grep to the names.

## The thesis (what "done" means per net)

A net is a readable `VerifiedNetSpec` layer list, and we prove a ladder of ties:

| rung | statement | where |
|---|---|---|
| A shapes | `#guard spec.toSpecs == <layout>` (param interface) | `VerifiedNets.lean` |
| B spec→math | `denoteX spec.layers … = <proof Forward fn>` by `rfl` | `Proofs/SpecVJP.lean` |
| C VJP | `xVerified_has_vjp : HasVJP (denoteX spec.layers …)` | `Proofs/SpecVJP.lean` |
| E spec→MLIR | `den (xFwdGraph …) = denoteX spec.layers …` (generated StableHLO denotes the spec) | `Proofs/SpecVJP.lean` |

Plus: the trainer is a one-line `main` and is **GPU-validated** to a sane accuracy.

## The 4 modules (all light except SpecVJP)

- `LeanMlir/VerifiedSpec.lean` — `VLayer` DSL + `VerifiedNetSpec` + `toSpecs` (folds layers →
  `(dims,initKind)` param layout). initKind 0=He, 1=ones(γ), 2=zeros(β/bias).
  Existing constructors: `convBn`, `maxPool`, `residualStage`, `globalAvgPool`, `dense`,
  `relu`, `conv` (plain, no BN), `flatten`, `bn` (scalar-global `bnForward`, rank-0 γ/β).
- `LeanMlir/VerifiedTrain.lean` — `VerifiedNet`/`VerifiedConfig` + the driver. Two methods:
  `train` (packed-params `mlpTrainStepV`, He-init — the conv/MLP nets) and `trainLinear`
  (2-arg `linearTrainStepV`, zero-init — only ch2). Both share `compileVmfb`/`loadData`
  (`.mnist`/`.cifar`/`.imagenette`)/eval. `loadData` for `.imagenette` center-crops 256→224.
- `LeanMlir/VerifiedNets.lean` — the shared concrete specs (a spec lives here once a proof
  imports it; r34's lives in its Main until then). Holds the A-rung `#guard`s.
- `LeanMlir/Proofs/SpecVJP.lean` — rungs B/C/E. Imports VerifiedNets + the relevant proof
  modules + StableHLO. **NOT in the `lake build` aggregator** — verify with
  `lake env lean LeanMlir/Proofs/SpecVJP.lean` (heavy; Mathlib).

## Migration recipe (per net)

1. **Trainer**: add `xVerified : VerifiedNetSpec` to `VerifiedNets.lean` (layer list +
   slug + inC/imageH/imageW + nClasses + data) + an A-rung `#guard xVerified.toSpecs == …`
   (read `XLayout` in `IreeRuntime.lean` for the expected `(dims,kind)` list). Rewrite
   `MainXVerified.lean` to `import LeanMlir.VerifiedNets` + a `VerifiedConfig` + one-line
   `main := xVerified.train xConfig (argv.head?.getD "data")`.
2. **GPU-validate** (recipe below) — confirm a sane accuracy.
3. **Proof ties** in `SpecVJP.lean`: `denoteX` (match the exact layer list → the proof's
   Forward fn, pinning dependent dims like `(h:=…)(w:=…)`); `xVerified_denote_eq … := rfl`;
   `xVerified_has_vjp` (canonical witness: `backward := ∑ pdiv …; correct _ _ _ := rfl`);
   and `xVerified_fwd_faithful := xFwdGraph_faithful …` if the faithfulness infra exists.

### Gotchas (bit us already)
- **`#guard` cannot follow a `/-- … -/` doc-comment** (it's a command, not a decl). Use a
  plain `-- …` or `/- … -/` comment before `#guard`. (Hit this 3×.)
- **Re-seed**: the generic spec-driven init seeds every param slot (biases too), so the
  legacy hand-init nets (mnist/cifar) re-initialize differently — accepted (statistically
  equiv, not bit-identical to old runs).
- **Dependent dims**: conv forwards take `{h w}` only via products (`Mat (c*h*w) d1`), so
  `denote`/faithfulness must pass `(h:=…)(w:=…)` explicitly. `Vec 784 ≟ Vec (1*(2*14)*(2*14))`
  is defeq (Nat reduces) — `exact …`/`rfl` handle it.
- **`Kernel4 oc ic kH kW`** is the conv-kernel type (in `Proofs`, via CNN.lean).
- The spec must be in a **lib** for a proof to import it (can't import a Main exe root) —
  hence `VerifiedNets.lean`. (We deduped linear after first defining a twin.)

## GPU run recipe (this sandbox)
```
PATH=$PWD/.venv/bin:$PATH  LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:$LD_LIBRARY_PATH \
IREE_BACKEND=rocm IREE_CHIP=gfx1100 IREE_DEVICE=hip \
.lake/build/bin/<trainer> <dataDir>
```
- MNIST idx: `/home/skoonce/lean/mnist-lean4/data`.  CIFAR-10 bin: `/home/skoonce/lean/claude_max/lean4-jax/data` (has `cifar-10/`).  Imagenette: needs `<dir>/imagenette/{train.bin@256,val.bin@224}` — locate before running.
- `ffi/libiree_ffi.so` is a HIP+local-task build; `.venv/bin/iree-compile` supports rocm gfx1100. CPU fallback = `IREE_BACKEND=llvm-cpu IREE_DEVICE=local-task` (slow for wide nets). Single-GPU only. See `memory/lean4-mlir-env-assets.md`.

## Status

DONE (full ladder A+B+C+E, GPU-validated):
- ch2 linear (92%, fwd+cotangent E) · ch3 MLP (97.78%, fwd+**back** E) · ch4 CNN (98.99%,
  fwd E) · ch5 cifar (~67%, fwd E) · ch5 cifar-bn (~57% slow, fwd E).
- Commits 649ef6a … 25847f2 on `main`, **NOT pushed** (~72 ahead).

## Imagenette nets — per-net pointers (do in this order)

All Layouts (`ResNet34/MobileNetV2/EfficientNet/ConvNeXt/ViTLayout`) are in
`IreeRuntime.lean` and already expose `.specs` → rung A is trivial. The trainers already
use `mlpTrainStepV` + `.imagenette` data → they fit `VerifiedNet.train` (one-line `main`).
**Key caveat for rung E**: `den(graph)=math` faithfulness in `StableHLO.lean` exists for
ch2–5 **and a `resnetFwdGraph_faithful` (line ~808)** — but **NOT** for mnv2/efficientnet/
convnext/vit (those MLIRs are hand-rendered by `tests/TestX{Train,Fwd}.lean` and
gradcheck-validated, not denotation-proven). So expect A+B+C for those; **E is r34-only
(maybe) until/unless that infra is extended**. Flag E as out-of-scope per net if so.

- **r34** (`Proofs/ResNet34.lean`): apex `resnet34_has_vjp_at` (:174, **parametric skeleton**
  over stem/mp/4×(down,ids)/gap/dense — depth is a `List.length`). VLayers all exist
  (`convBn`/`maxPool`/`residualStage`/`globalAvgPool`/`dense` — built in the r34 spike).
  `toBlocks` (the spec→renderer-block fold) was prototyped + proven `== blocks` in a reverted
  edit — re-derivable. E: `resnetFwdGraph_faithful` exists (StableHLO ~808) — check it matches
  the full `ResNet34Layout` render or is representative.
- **mnv2** (`Proofs/MobileNetV2.lean`): apex `mobilenetv2_has_vjp_at`. Inverted-residual /
  MBConv blocks (expand 1×1 → depthwise k×k strided → project 1×1, per-channel BN, **relu6**,
  residual when s=1∧ic=oc). New VLayers needed: depthwise, relu6, invertedResidual (+ per-channel BN
  — distinct from scalar `.bn`). No fwd-graph faithfulness → E likely out of scope.
- **efficientnet** (`Proofs/EfficientNet.lean`): apex `efficientnet_has_vjp(_correct)` (:456/536,
  unconditional), `efficientnetForward` (:424). MBConv + **squeeze-excite** + **swish** +
  **batch-norm** (per-channel). New VLayers: swish, sigmoid, SE, MBConv-w-SE, per-channel BN.
- **convnext** (`Proofs/ConvNeXt.lean`): apex `convnext_has_vjp(_correct)` (:293/368),
  `convNextForward` (:267) — **representative ~2-block witness**, render is [3,3,9,3] (depth gap
  to handle). Block: depthwise 7×7 → scalar-LN → 1×1 expand → **GELU** → 1×1 project → layerScale;
  patchify 4×4/s4 + 2×2/s2 downsample (even-kernel transposed backward). New VLayers: convNextBlock,
  gelu, layerScale, patchify, downsample. LN is **scalar-global** = the existing `.bn` op.
- **vit** (`Proofs/Attention.lean`): apex `vit_full_has_vjp_correct` (softmax/SDPA/MHSA/block/
  patchEmbed stack). Transformer blocks (MHSA + MLP), patch-16 embed, CLS+pos. New VLayers:
  patchEmbed, transformerBlock (MHSA/SDPA), gelu, LN. **Proof-vs-render granularity gap**:
  the LN proof witness is **scalar** (`layerNormForward` = `bnForward`), but the rendered/trained
  ViT uses **per-channel `[D]` LN** — so denote can only tie the spec to the scalar witness; the
  `[D]` render is beyond it (state this explicitly). vit_full witness is also scalar-LN.

## Loose ends (do AFTER the rest are in the new pattern)

- **Wire `SpecVJP.lean` into the build** — currently only `lake env lean`-checked, not in the
  `LeanMlir.lean` aggregator, so it isn't verified on `lake build` (could bit-rot). Adding it
  pulls Mathlib into the default build (slow) — decide if worth it (maybe a separate test target).
- **Backward-E for the conv nets** — only forward-E is done for CNN/CIFAR. `cnnBackGraph_faithful`
  (StableHLO ~900) and the cifar backward equivalents exist but carry the 5 ReLU/maxpool
  smoothness hyps; wrap them as `xVerified_back_faithful` (transcribe the hyps; verbose). MLP
  already has backward-E (only 2 hyps).
- **Text/file trust boundary** — rung E proves `den(graph) = spec`; still trusted: `xFwdModuleV =
  pretty (emit graph)` (the pretty-printer) and that the committed `verified_mlir/*.mlir` equals
  that text (regeneration). `Proofs/StableHLOParse.lean` could close the text→IR round-trip.
- **CIFAR-BN GPU perf** — ~2 min/epoch (scalar-BN global reductions over c·h·w are slow on the
  rocm reduction path); fine, just slow. Not a correctness issue.
- **Push** — ~72 local commits on `main`, unpushed (needs explicit per-push OK per
  `memory/git-workflow-commit-to-main.md`).
- **Imagenette E infra** — if we want rung E for mnv2/enet/convnext/vit, the `den(emit g)=math`
  faithfulness for their op-graphs would need building (currently those nets are gradcheck-validated
  string renders, not denotation-proven). Big; likely a separate effort.
