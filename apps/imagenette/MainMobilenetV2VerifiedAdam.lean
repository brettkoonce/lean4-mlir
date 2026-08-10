import LeanMlir.VerifiedNets

/-! # `mobilenetv2-verified-adam` — train MobileNetV2 with the VERIFIED-rendered **AdamW** step

The mnv2 peer of `vit`/`enet-verified-adam`: the proof-rendered MobileNetV2 train step
(`LeanMlir/Proofs/Codegen/MobileNetV2RenderB.lean → verified_mlir/mobilenetv2_adam_train_step.mlir`,
`@mobilenetv2_adam_train_step`) — 17-block inverted-residual net with relu6 + per-channel batch-norm,
with the gradients un-fused and handed to the proven `adamMNextF`/`adamVNextF`/`adamWParamF` triple —
driven by the generic `VerifiedNet.trainAdamSched`: `[θ|m|v]` (158 params) packed as one blob +
runtime `lr`/`bc₁`/`bc₂` scalars (cosine + warmup + per-step bias correction) through the unchanged
FFI (`n_params = 3k`).

**The artifact is `pretty(provenGraph)` since 2026-07-28** (handoff §2f). It used to come from a
hand-written string emitter in `tests/TestMobilenetV2TrainPC.lean`; the swap was licensed by
`mobilenetv2-adam-tie`, which found the two **bit-exact** on all 6,795,329 returned floats — forward
(the 52 BN batch statistics), `%loss`, and the gradient — with three negative controls each firing
its own gate. The driver itself needed no change: it resolves the path from the net slug.

Recipe matches `mobilenet-v2-train` (`MainMobilenetV2Train.lean`'s `mobilenetV2Config`): AdamW
lr 1e-3 / wd 1e-4, cosine + 3-epoch warmup, label smoothing 0.1, augment, 80 epochs, bs 32
(no EMA, no grad-clip). **Exact BN parity**: TRUE batch-norm (reduce `[0,2,3]`) in the train step
+ running-stats eval — `mobilenetv2Verified.bnChannels` (52 layers, full-paper 17-block net) is
non-empty, so the generic `trainAdamSched` threads per-layer EMA batch stats and evals through
`@mobilenetv2_fwd_eval` (affine BN with the running stats), class-batch-independent on the sorted
val set. Weight decay is applied uniformly (incl. BN/bias), matching the ViT path.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mobilenetv2-verified-adam data` (loader reads
`data/imagenette`). For the XLA/PJRT lowerer — measurably faster on this box, and the only path to

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file: the config and entry point below ARE the program.
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

def main (argv : List String) : IO Unit := runMobilenetV2Adam argv
