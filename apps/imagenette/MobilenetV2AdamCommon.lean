import LeanMlir.VerifiedNets

/-! # Shared body of the verified MobileNetV2 + AdamW Imagenette trainer

`mobilenetv2-verified-adam` (IREE) and `mobilenetv2-verified-adam-xla` (XLA/PJRT) are the same
program — same `VerifiedNetSpec`, same `verified_mlir/mobilenetv2_<variant>_train_step.mlir`, same
schedule and He-init seed. Only the linked trusted lowerer differs.

Lake requires a distinct root module per executable, so the config and entry point live here rather
than being duplicated; drift in `epochs`, `batchSize`, the seed, or any AdamW hyperparameter would
quietly invalidate any cross-backend comparison. Same reason as `MainResnet34VerifiedAdam.lean` and
`EfficientNetAdamCommon.lean`.
-/

-- Matches MainMobilenetV2Train.lean's `mobilenetV2Config`: 80 epochs, bs 32, AdamW lr 1e-3 /
-- wd 1e-4, cosine + 3-epoch warmup, label smoothing 0.1, augment.
def mobilenetv2AdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- Entry point for both backends. baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine
    decay (`mobilenetV2Config`).

    `LEAN_MLIR_VARIANT` selects the rendered train step, i.e. which
    `verified_mlir/mobilenetv2_<variant>_train_step.mlir` is loaded (and with it a distinct vmfb and
    checkpoint). **Only `adam` exists today** — it is `pretty(provenGraph)` out of
    `Proofs/Codegen/MobileNetV2RenderB.lean`, tied bit-exactly against the retired hand-written
    emitter on all 6,795,329 returned floats (handoff §2f). The knob is threaded anyway because
    `mnv2AdamVariant` already returns an `adamdp` name and the renderer already takes `replicas`;
    when someone writes that artifact's `#eval` the driver side is done. Pair a DP variant with
    `LEAN_MLIR_REPLICAS=N PJRT_REPLICAS=N` and `HIP_VISIBLE_DEVICES` unset, and use the XLA build —
    collectives exist only on the PJRT path, and the IREE shim refuses a DP entry point outright
    rather than silently running single-device.

    No `LEAN_MLIR_BATCH` here, unlike EfficientNet: the batch is baked into the graph and bs32 is
    the only mnv2 render that exists, so the knob could only ever produce a shape error. -/
def runMobilenetV2Adam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  -- ▶ `rms` = the RMSProp render (`recipe_gaps.md` v1.2), rendered at THIS shape deliberately so
  -- the optimizer has a runnable descent check that needs neither ImageNet nor the tfds shim.
  --
  -- ⚠ **This is not the reference's recipe and must not be quoted as one.** MobileNetV2's is
  -- ImageNet at global batch 256; this is Imagenette at 32. What carries over is the OPTIMIZER and
  -- the SHAPE of the schedule (warmup → ×0.98/epoch, mean-square init 1.0); the peak LR is
  -- `mnv2RmsSchedule.lr` **linearly scaled by batch**, which is the standard rule and is stated here
  -- rather than being a second hardcoded number. `LEAN_MLIR_BASE_LR_U` overrides it in micro-units
  -- (`5625` = 0.005625), since this toolchain has no `String.toFloat?`.
  let sched := mnv2RmsSchedule
  let rms := variant.startsWith "rms"
  let bs := mobilenetv2AdamConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => if rms then sched.lr * bs.toFloat / 256.0 else 0.001
  mobilenetv2Verified.toNet.trainAdamSched mobilenetv2AdamConfig
    (argv.head?.getD "data") baseLR 0.9 0.999 3 variant
    (if rms then sched.decayRate else 0.0) sched.decayEpochs
