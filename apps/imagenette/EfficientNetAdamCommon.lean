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
      shim refuses the DP entry point outright rather than silently running single-device.

    `LEAN_MLIR_BATCH` overrides the batch, and **must match the batch the selected variant was
    rendered at** — the batch is baked into the graph, not a runtime dimension, so a mismatch is a
    shape error at the first invoke rather than something that silently limps. The pairs that exist:
    `adam`/`adamdp` at 32, `adam128`/`adamdp128` at 128 (§2e-quater). **The eval forwards are still
    rendered at bs32**, so anything other than 32 needs `LEAN_MLIR_SKIP_EVAL=1` or re-rendered
    forwards — the same caveat §2d.1 carries for R34's bs256. -/
def runEfficientNetAdam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD efficientnetAdamConfig.batchSize
  -- ▶ `rms` = the RMSProp render, at this shape for the same reason mnv2's is: a descent check that
  -- needs neither ImageNet nor the shim. ⚠ **Not the reference's recipe** — that is ImageNet at
  -- global 256 — so the peak is `enetRmsSchedule.lr` linearly scaled by batch, and only the
  -- optimizer and the schedule SHAPE (warmup → ×0.97 every 2.4 epochs, mean-square init 1.0) carry
  -- over. `LEAN_MLIR_BASE_LR_U` overrides in micro-units (`2000` = 0.002).
  let sched := enetRmsSchedule
  -- ⚠ SUBSTRING, not prefix: `emarms` (RMSProp + EMA) does not START with "rms", so a prefix test
  -- would quietly hand this net the AdamW LR and a cosine schedule. Same trap as the driver's.
  let rms := (variant.splitOn "rms").length > 1
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => if rms then sched.lr * bs.toFloat / 256.0 else 0.001
  -- ▶ `ema*` variants carry the 4th `[θ|m|v|ema]` blob region. `emarms` is RMSProp + EMA, which is
  -- this net's REFERENCE recipe (`efficientNetB0ImagenetConfig`) — and its 72.31% is the shadow's
  -- number, not the live weights'. ⚠ On this net EMA also shadows the BN running buffers
  -- (driver-side, `ema_bn`), because eval must pair EMA weights with EMA-lagged statistics.
  -- `LEAN_MLIR_EMA_DECAY_U` is micro-units (`999900` = 0.9999, the reference's value); ⚠ `0` is
  -- meaningful and is the gate — at decay 0 the shadow must be BIT-IDENTICAL to the live weights.
  let emaDecay := match (← IO.getEnv "LEAN_MLIR_EMA_DECAY_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.9999
  -- ▶ `LEAN_MLIR_DROP_RATE_U` — stochastic-depth rate in MICRO-units (`200000` = 0.2, the
  -- reference's `efficientNetB0ImagenetConfig.dropPath`). Unset ⇒ the spec's ramp.
  -- ⚠ **`0` is meaningful and it is THE GATE**: every keep becomes 1.0, so every supplied scale is
  -- exactly 1.0 and the drop op is the identity in IEEE (`Proofs.dropPath_ones_id`). The `adamsd`
  -- render must then train the same parameters as the plain `adam` render — a free exact endpoint,
  -- the peer of EMA's `decay = 0`, and it is what pins the WIRING: a drop site on the wrong side of
  -- the skip add, or a scale reaching the identity path, fails it immediately.
  let dropNet := match (← IO.getEnv "LEAN_MLIR_DROP_RATE_U").bind (·.toNat?) with
    | some 0 => { efficientnetVerified.toNet with
                    dropKeeps := efficientnetVerified.dropKeeps.map (fun _ => 1.0) }
    | some u =>
        -- Re-derive the ramp at a different rate from the SPEC's own keeps, so the block indices
        -- stay the renderer's: `i = (1 − keep) · 15 / 0.2` recovers the index the spec encoded.
        let rate := u.toFloat * 1e-6
        { efficientnetVerified.toNet with
            dropKeeps := efficientnetVerified.dropKeeps.map
              (fun k => 1.0 - rate * ((1.0 - k) * 15.0 / 0.2) / 15.0) }
    | none   => efficientnetVerified.toNet

  dropNet.trainAdamSched { efficientnetAdamConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 3 variant
    (if rms then sched.decayRate else 0.0) sched.decayEpochs emaDecay
