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
    checkpoint). Two exist, both written by `Proofs/Codegen/ConvNeXtRender.lean`:

    * **`adam`** (default) — `pretty(provenGraph)`, tied bit-exactly against the retired
      hand-written emitter on all 83,434,629 returned floats (handoff §2f-bis, §5: all 180 params
      are `pretty(AST)` since the two weight-gradient gaps were closed).
    * **`adamdp`** — the same graph plus one `all_reduce(add)/N` per parameter gradient, a declared
      trusted carve-out (§2h-quater). Gated by `convnext-dp-check`: on a duplicated batch the DP
      step reproduces the single-device one, `%loss` bit-exact and gradient norm-rel 1.1e-8, with
      the sum-not-mean control firing at 1.114. Pair it with `LEAN_MLIR_REPLICAS=N PJRT_REPLICAS=N`
      and `HIP_VISIBLE_DEVICES` unset, and use the **XLA** build — collectives exist only on the
      PJRT path, and the IREE shim refuses a DP entry point outright rather than silently running
      single-device.

    No `LEAN_MLIR_BATCH` here, unlike EfficientNet: the batch is baked into the graph and bs32 is
    the only ConvNeXt render that exists, so the knob could only ever produce a shape error. -/
def runConvNeXtAdam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  -- ▶ `ema`/`emadp` select the EMA-shadow render (`planning/ema.md`): same AdamW graph plus a 4th
  -- `[θ|m|v|ema]` blob region updated by `adamMNextF` at `(β₁ := d)`, with EVAL AND THE CHECKPOINT
  -- scoring the shadow. ConvNeXt is the right first net for it — LayerNorm means there is no
  -- `ema_bn` peer to carry, so it is the parameter shadow alone.
  --
  -- `LEAN_MLIR_EMA_DECAY_U` sets the decay in MICRO-units (`999900` = 0.9999, the reference's
  -- value), because this toolchain has no `String.toFloat?` — the `LEAN_MLIR_BASE_LR_U` dodge.
  -- ⚠ **`0` is meaningful and is the gate**: at decay 0 the shadow must be BIT-IDENTICAL to the
  -- live weights, since `d = min(0, ·) = 0 ⇒ ema' = θ'`. That is a free exact endpoint and it is
  -- what pins the wiring — a shadow reading the wrong slot fails it immediately.
  let emaDecay := match (← IO.getEnv "LEAN_MLIR_EMA_DECAY_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.9999
  convnextVerified.toNet.trainAdamSched convnextAdamConfig
    (argv.head?.getD "data") 0.001 0.9 0.999 3 variant 0.0 1.0 emaDecay
