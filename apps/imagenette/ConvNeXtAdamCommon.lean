import LeanMlir.VerifiedNets

/-! # Shared body of the verified ConvNeXt-T + AdamW Imagenette trainer

`convnext-verified-adam` (IREE) and `convnext-verified-adam-xla` (XLA/PJRT) are the same program —
same `VerifiedNetSpec`, same `verified_mlir/convnext_<variant>_train_step.mlir`, same schedule and
He-init seed. Only the linked trusted lowerer differs.

Lake requires a distinct root module per executable, so the config and entry point live here rather
than being duplicated; drift in `epochs`, `batchSize`, the seed, or any AdamW hyperparameter would
quietly invalidate any cross-backend comparison. Same reason as `Resnet34AdamCommon.lean` and
`EfficientNetAdamCommon.lean`.
-/

-- Matches MainConvNeXtTrain.lean's `convNextTinyConfig`: 80 epochs, bs 32, AdamW lr 1e-3 / wd 1e-4,
-- cosine + 3-epoch warmup, label smoothing 0.1, augment.
def convnextAdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- Entry point for both backends. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine
    decay (`convNextTinyConfig`).

    `LEAN_MLIR_VARIANT` selects the rendered train step, i.e. which
    `verified_mlir/convnext_<variant>_train_step.mlir` is loaded (and with it a distinct vmfb and
    checkpoint). **Only `adam` exists today** — it is `pretty(provenGraph)` out of
    `Proofs/Codegen/ConvNeXtRender.lean`, tied bit-exactly against the retired hand-written emitter
    on all 83,434,629 returned floats (handoff §2f-bis, §5: all 180 params are `pretty(AST)` since
    the two weight-gradient gaps were closed). The knob is threaded so that adding a DP variant
    later is a renderer change only — **ConvNeXt's renderer has no `replicas` parameter at all**
    today, the one large net with no DP path whatsoever. Pair a DP variant with
    `LEAN_MLIR_REPLICAS=N PJRT_REPLICAS=N` and `HIP_VISIBLE_DEVICES` unset, and use the XLA build:
    collectives exist only on the PJRT path.

    No `LEAN_MLIR_BATCH` here, unlike EfficientNet: the batch is baked into the graph and bs32 is
    the only ConvNeXt render that exists, so the knob could only ever produce a shape error. -/
def runConvNeXtAdam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  convnextVerified.toNet.trainAdamSched convnextAdamConfig
    (argv.head?.getD "data") 0.001 0.9 0.999 3 variant
