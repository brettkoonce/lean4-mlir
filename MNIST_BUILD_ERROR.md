# mnist-cnn-train: stale SGD FFI vs Adam-flavored codegen

**Status:** broken on `main` as of 2026-04-11. Reproduces instantly from
a clean checkout. Needs a refactor, not a targeted patch — slated to
be brought in line with the `resnet34-train` template.

## Symptom

```
$ .lake/build/bin/mnist-cnn-train
MNIST CNN: 1676458 params
Generating train step MLIR...
  71683 chars
Compiling...
  compiled
Loading IREE...
Loading MNIST...
  60000 images
Init params...
  1676458 params
Training (468 batches/epoch, batch=128)
iree_ffi: gen_train_invoke failed:
iree/runtime/src/iree/vm/invocation.c:99: INVALID_ARGUMENT;
  input list and function mismatch; expected 52 arguments but passed 35
uncaught exception: f32 train_step failed
```

## Root cause

`MlirCodegen.generateTrainStep` emits an **Adam** training step for any
spec containing `convBn` layers. Per the generated signature in
`.lake/build/mnist_cnn_train_step.mlir`, the function takes:

```
16 params             (W/γ/β for 4 convBn + W/b for 2 dense)
16 first-moment (m_*) (Adam)
16 second-moment (v_*) (Adam)
 1 x_flat
 1 y (labels)
 1 lr scalar
 1 t  (bias-correction step counter)
---
52 total
```

`MainMnistCnnTrain.lean:135` still calls the **SGD-era** FFI:

```lean
let packed := p.append v                              -- only 2 copies
let out ← IreeSession.trainStepF32 sess
            "jit_mnist_cnn_train_step.main"
            packed allShapes xba xSh yb lr batch      -- no m, no t
```

So it pushes `x + 32 params + y + lr = 35` and the IREE VM rejects the
call before the first batch runs.

## Why it's out of date

`mnist-cnn-train` predates the codegen migration to Adam + BN. Every
later trainer was updated; this one never was. For reference, the
working pattern from `MainResnetTrain.lean:209-212`:

```lean
let packed := (p.append m).append v
let out ← IreeSession.trainStepAdamF32 sess
            "jit_resnet34_train_step.main"
            packed allShapes xba xSh yb
            lr globalStep.toFloat bnShapes batch
```

Three separate fixes packed into one call: `trainStepF32 →
trainStepAdamF32`, `p.append v → (p.append m).append v`, and the new
`t` + `bnShapes` parameters.

## Scope of refactor (not just a one-liner)

The right fix isn't patching line 135 — it's bringing
`MainMnistCnnTrain.lean` in line with the full `MainResnetTrain.lean`
template:

1. **Adam packed params** — three copies, not two.
2. **Running BN statistics.** `convBn` layers need `bnShapes` threaded
   through `trainStepAdamF32` and the stats carried across steps for
   eval-time normalization (see `RESULTS.md` — every BN-using model
   relies on these).
3. **Split eval forward.** Resnet/efficientnet/etc. compile a separate
   `*_eval.forward_eval` vmfb and call it via `IreeSession.forwardF32`.
   `MainMnistCnnTrain.lean` currently only tracks loss, with a TODO
   comment about wiring up test accuracy.
4. **Step counter `t`.** Needs to be a `Float` passed per batch for
   Adam bias correction.

None of the plumbing is missing from `LeanMlir/` — `trainStepAdamF32`,
`forwardF32`, `packShapes` all exist and all work (ResNet-34 uses
them end-to-end). This is purely a `Main*Train.lean` rewrite.

## Other trainers

| Trainer | Status |
|---|---|
| `mnist-mlp-train` | works (historical 2-vmfb dance, see `IREE_BUILD.md` §6) |
| `mnist-cnn-train` | **broken — this file** |
| `cifar-cnn-train` / `cifar-bn-train` | **not verified** — may have the same drift, worth checking |
| `resnet34-train` and later (mobilenet, efficientnet, vit, vgg) | believed working per source inspection; not personally rerun from this clean checkout |

`cifar-cnn-train` and `cifar-bn-train` are the most likely candidates
for the same bug — they're from the same phase-2 generation as
`mnist-cnn-train`. Worth grepping for `trainStepF32` vs
`trainStepAdamF32` before trusting them:

```bash
grep -n "trainStepF32\b" Main*.lean
# Any hit on a file with .convBn in its spec is probably broken.
```

## Quick repro

```bash
# From a clean checkout with libiree_ffi.so built per IREE_BUILD.md:
./download_mnist.sh
source .venv/bin/activate
lake build mnist-cnn-train
.lake/build/bin/mnist-cnn-train
# → fails at "Training (468 batches/epoch, batch=128)" with the 52-vs-35 error
```
