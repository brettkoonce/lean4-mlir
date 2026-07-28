import LeanMlir.VerifiedNets

/-! # Shared body of the verified ResNet-34 + AdamW Imagenette trainer

`resnet34-verified-adam` (IREE) and `resnet34-verified-adam-xla` (XLA/PJRT) are
the same program — same `VerifiedNetSpec`, same
`verified_mlir/resnet34_adam_train_step.mlir`, same §1a ties, same schedule and
He-init seed. Only the linked trusted lowerer differs.

This is rung 3 of `planning/xla_pjrt_ladder.md`: full scale, and the first net
whose `bnChannels` is non-empty, so the train step carries **BN running statistics**
out in passthrough slots and eval runs `@resnet34_fwd_eval` with them.

Lake requires a distinct root module per executable, so the config and entry point
live here rather than being duplicated; drift in `epochs`, `batchSize`, the seed,
or any AdamW hyperparameter would quietly invalidate the G2 comparison.
-/

-- Matches MainResnetTrain.lean's `resnet34Config`: 80 epochs, bs 32, AdamW lr 1e-3 / wd 1e-4,
-- cosine + 3-epoch warmup, label smoothing 0.1, augment.
def resnet34AdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- Entry point for both backends. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear
    warmup then cosine decay (`resnet34Config`).

    `LEAN_MLIR_VARIANT` selects the rendered train step, i.e. which
    `verified_mlir/resnet34_<variant>_train_step.mlir` is loaded (and, with it, a distinct vmfb and
    checkpoint). Two exist:

    * **`adam`** (default) — the certified single-device render,
      `pretty(provenGraph)` out of `Proofs/Codegen/ResNet34RenderB.lean` (handoff §2b-ter).
    * **`adamdp`** — the DATA-PARALLEL render, from the hand-written emitter in
      `tests/TestResnet34Train.lean` at `REPLICAS > 1`. It all-reduces each gradient before the
      optimizer (§2c). **Not certified** — the batched renderer cannot emit collectives yet — so
      it is opt-in by name rather than something a plain run can land on by accident. Pair it with
      `LEAN_MLIR_REPLICAS=N PJRT_REPLICAS=N` and `HIP_VISIBLE_DEVICES` unset. -/
def runResnet34Adam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  resnet34Verified.toNet.trainAdamSched resnet34AdamConfig
    (argv.head?.getD "data") 0.001 0.9 0.999 3 variant
