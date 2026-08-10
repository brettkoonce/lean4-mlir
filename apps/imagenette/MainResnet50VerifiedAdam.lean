import LeanMlir.VerifiedNets

/-! # `resnet50-verified-adam` — ResNet-50 on Imagenette, via XLA/PJRT


The bottleneck peer of `resnet34-verified-adam` (on XLA): same harness, same 80-epoch
bs32 AdamW schedule, one net swapped. `resnet50Verified` was a layout skeleton
with no artifacts until `Proofs/Codegen/ResNet50RenderB.lean` grew its
`resnet50_*` renders.

**This does not move the verification tier**, and it moves it less than R34's row
does: R34/Imagenette carries §1a ties, `resnet50Verified` carries none yet. What
this trainer produces is a measurement on the certified renderer's output.

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet50-verified-adam
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet50-verified-adam data
```

⚠ `IREE_BACKEND` is inert here — PJRT does not read it.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything: the
backend is a run-time choice about transport, not a different program.
-/

/-- Matches `resnet34AdamConfig` — 80 epochs, bs 32 — so the R50 row of the
    Imagenette tier is read against the other nets at the same schedule, and any
    difference is the architecture rather than the recipe. -/
def resnet50AdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- Entry point for both backends. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear
    warmup then cosine decay — `resnet34AdamConfig`'s schedule exactly.

    `LEAN_MLIR_VARIANT` selects `verified_mlir/resnet50_<variant>_train_step.mlir`.
    Only **`adam`** (the default) is rendered: the certified single-device render
    at bs32. The DP / large-batch variants that R34 carries (`adamdp`, `adam256`,
    `adamdp128`) have no R50 counterpart yet — asking for one gets a missing-file
    error, which is the intended loud failure.

    `LEAN_MLIR_BATCH` overrides the batch and **must match the batch the variant
    was rendered at** — batch is baked into the graph, not a runtime dimension, so
    a mismatch is a shape error at the first invoke rather than something that
    silently limps. The forwards are rendered at 32 as well, so any other batch
    also needs `LEAN_MLIR_SKIP_EVAL=1` and then reports no validation accuracy. -/
def runResnet50Adam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD resnet50AdamConfig.batchSize
  -- `LEAN_MLIR_BASE_LR_U` — base LR in MICRO-units (1e-6), so `100000` is 0.1.
  -- Integer-encoded because this toolchain has no `String.toFloat?`. Same knob,
  -- same units, same default as the R34 trainer.
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.001
  resnet50Verified.toNet.trainAdamSched { resnet50AdamConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 3 variant

def main (argv : List String) : IO Unit := runResnet50Adam argv
