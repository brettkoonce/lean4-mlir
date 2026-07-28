import LeanMlir.VerifiedNets

/-! # Shared body of the verified EfficientNet-B0 + AdamW Imagenette trainer

`efficientnet-verified-adam` (IREE) and `efficientnet-verified-adam-xla` (XLA/PJRT) are the same
program — same `VerifiedNetSpec`, same `verified_mlir/efficientnet_<variant>_train_step.mlir`, same
schedule and He-init seed. Only the linked trusted lowerer differs.

Lake requires a distinct root module per executable, so the config and entry point live here rather
than being duplicated; drift in `epochs`, `batchSize`, the seed, or any AdamW hyperparameter would
quietly invalidate any cross-backend comparison. Same reason as `Resnet34AdamCommon.lean`.
-/

-- Matches MainEfficientNetTrain.lean's `efficientNetB0Config`: 80 epochs, bs 32, AdamW lr 1e-3 /
-- wd 1e-4, cosine + 3-epoch warmup, label smoothing 0.1, augment.
def efficientnetAdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- Entry point for both backends. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine
    decay (`efficientNetB0Config`).

    `LEAN_MLIR_VARIANT` selects the rendered train step, i.e. which
    `verified_mlir/efficientnet_<variant>_train_step.mlir` is loaded (and with it a distinct vmfb
    and checkpoint). Two exist, and **both are `pretty(provenGraph)` out of
    `Proofs/Codegen/EfficientNetRender.lean`** — unlike ResNet-34, this net never had a
    hand-written DP emitter to migrate off:

    * **`adam`** (default) — the certified single-device render, tied bit-exactly against the
      retired hand-written emitter (handoff §2e).
    * **`adamdp`** — the DATA-PARALLEL render: the same graph plus one `all_reduce(add)/N` per
      parameter gradient between the certified gradient and the certified AdamW triple. That
      collective is a **declared trusted carve-out** (handoff §5) and the render says so in its own
      output banner. Pair it with `LEAN_MLIR_REPLICAS=N PJRT_REPLICAS=N` and `HIP_VISIBLE_DEVICES`
      unset; it needs the XLA build, because collectives only exist on the PJRT path and the IREE
      shim refuses the DP entry point outright rather than silently running single-device. -/
def runEfficientNetAdam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  efficientnetVerified.toNet.trainAdamSched efficientnetAdamConfig
    (argv.head?.getD "data") 0.001 0.9 0.999 3 variant
