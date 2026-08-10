import LeanMlir.VerifiedNets

/-! # `resnet34-verified-adam` — train ResNet-34 with the VERIFIED-rendered **AdamW** step

The ResNet-34 peer of the `vit`/`mnv2`/`enet` verified-adam trainers: the proof-rendered train step
(`tests/TestResnet34Train.lean → verified_mlir/resnet34_adam_train_step.mlir`,
`@resnet34_adam_train_step`) — 7×7-s2 stem → maxpool → [3,4,6,3] basic blocks (per-channel BN +
strided downsamples) → GAP → dense — with the SGD update swapped for AdamW via
`ViTRender.emitAdamV`, driven by the generic `VerifiedNet.trainAdamSched`: `[θ|m|v]` (146 params)
packed as one blob + runtime `lr`/`bc₁`/`bc₂` scalars (cosine + warmup + per-step bias correction)
through the unchanged FFI (`n_params = 3k`).

Recipe matches the reference (`MainResnetTrain.lean`'s `resnet34Config`): AdamW lr 1e-3 / wd 1e-4,
cosine + 3-epoch warmup, label smoothing 0.1, augment, 80 epochs, bs 32. **Exact BN parity**:
true batch-norm (reduce `[0,2,3]`) in train + running-stats eval — `resnet34Verified.bnChannels`
(36 layers) drives the generic `trainAdamSched` to thread per-layer EMA batch stats and eval
through `@resnet34_fwd_eval` (class-batch-independent on the sorted val set). Weight decay
uniform (incl. BN/bias), matching the other verified paths.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/resnet34-verified-adam data` (loader reads
`data/imagenette`).

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file: the config and entry point below ARE the program.
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
    checkpoint). **All four are `pretty(provenGraph)` out of
    `Proofs/Codegen/ResNet34RenderB.lean`, which is the sole writer of every one:**

    * **`adam`** (default) — the certified single-device render at bs32 (handoff §2b-ter).
    * **`adamdp`** — the DATA-PARALLEL render at bs32: the same graph plus one `all_reduce(add)/N`
      per parameter gradient before its AdamW triple, a **declared trusted carve-out** (§5) that the
      render announces in its own output banner. *(An earlier version of this docstring called this
      variant "not certified — the batched renderer cannot emit collectives yet" and pointed at a
      hand-written emitter in `tests/TestResnet34Train.lean`. Both were true until §2b-quater, which
      moved it onto the certified renderer and DELETED that emitter. Corrected 2026-07-29.)*
    * **`adam256`** — bs256, single device (§2d.1), worth **1.78×** img/s over bs32.
    * **`adamdp128`** — bs128 × N replicas, i.e. global 256 data-parallel.

    `LEAN_MLIR_BATCH` overrides the batch and **must match the batch the selected variant was
    rendered at** — batch is baked into the graph, not a runtime dimension, so a mismatch is a shape
    error at the first invoke rather than something that silently limps. Pairs: `adam`/`adamdp` at
    32, `adam256` at 256, `adamdp128` at 128.

    ⚠ **The eval forwards are rendered at bs32**, so any other batch needs `LEAN_MLIR_SKIP_EVAL=1`
    (or re-rendered forwards) and therefore reports no validation accuracy.

    Pair a DP variant with `LEAN_MLIR_REPLICAS=N PJRT_REPLICAS=N` and `HIP_VISIBLE_DEVICES` unset,
    and select XLA (the default; `LEAN_MLIR_LOWERER=xla` to be explicit) — collectives exist only
    on the PJRT path, and the IREE shim refuses a DP entry point outright rather than silently
    running single-device. This is now a RUN-TIME choice, not a separate build. -/
def runResnet34Adam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD resnet34AdamConfig.batchSize
  -- `LEAN_MLIR_BASE_LR_U` — base LR in MICRO-units (1e-6), so `100000` is 0.1. Integer-encoded
  -- because this toolchain has no `String.toFloat?`; the same dodge `LEAN_MLIR_PERTURB_R` uses
  -- (1e-9 there). The default 0.001 is unchanged, so every existing run is bit-for-bit unaffected.
  --
  -- ⚠ **Effectively REQUIRED for `LEAN_MLIR_VARIANT=mom`.** 0.001 is an *AdamW* rate; the
  -- heavy-ball render (`R34Opt.heavyBall`) matches `jax/MainResnetImagenet.lean`, which uses
  -- **0.1** at batch 256. Running `mom` at the default under-steps by ~100×, which looks exactly
  -- like a broken render rather than a wrong knob — and "the optimizer render is fine, the LR was
  -- an Adam one" is a debugging session nobody should have to repeat.
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.001
  resnet34Verified.toNet.trainAdamSched { resnet34AdamConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 3 variant

def main (argv : List String) : IO Unit := runResnet34Adam argv
