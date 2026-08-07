import LeanMlir.VerifiedNets

/-! # Shared body of the verified ResNet-50 + AdamW Imagenette trainer

`resnet50-verified-adam` (IREE) and `resnet50-verified-adam-xla` (XLA/PJRT) are
the same program — same `VerifiedNetSpec` (`resnet50Verified`), same
`verified_mlir/resnet50_adam_train_step.mlir`, same schedule and He-init seed.
Only the linked trusted lowerer differs.

This is the **bottleneck** peer of `Resnet34AdamCommon`: identical harness, one
net swapped. `resnet50Verified` carried the note "LAYOUT SKELETON: no render, no
proof chain, no artifact yet" — the render half is now discharged
(`Proofs/Codegen/ResNet50RenderB.lean`, the `resnet50_*` block at the end).

⚠ **The proof chain is NOT.** R34's Imagenette pairing carries §1a ties;
`resnet50Verified` has none yet, so a number off this trainer is a *measurement
on the certified renderer's output*, not a tied one. Name that when quoting it.

Lake requires a distinct root module per executable, so the config and entry
point live here rather than being duplicated; drift in `epochs`, `batchSize`, the
seed, or any AdamW hyperparameter would quietly invalidate the comparison against
the other Imagenette nets.
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
