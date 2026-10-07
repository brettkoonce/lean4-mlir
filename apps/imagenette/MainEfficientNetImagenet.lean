import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Train

/-! # `efficientnet-imagenet-verified` — EfficientNet-B0 on full ImageNet-1k, verified → XLA

One of the ImageNet scale-tier trainers. `B` and `nClasses` are renderer parameters, so this
needs only a `slug`, and the report-only loss's −α/K is derived rather than hardcoded.

Does NOT move the verification tier. The optimizer follows the variant: the `rms*`/`emarms*`
renders are the reference's RMSProp with its ×0.97-every-2.4 decay, EMA, drop-connect and
classifier dropout; the `adam*` renders are AdamW + cosine. The shipping
`emarmsdp64dropdowxeps0001bf16` is the TF recipe the JAX `full` config carries: `wx`, BN ε 1e-3,
the staircase on the global step (`enetImagenetRmsSchedule`) and the i/16 drop-connect ramp.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The target name has no
`-xla` suffix because the backend does not distinguish the program.
-/

/-- 350 epochs at 64 per device — the tier this net is measured against, and at four
    replicas its global batch of 256. `batchSize` is PER DEVICE and must match the batch the
    variant was rendered at.

    The JAX reference the chapter prints is the 350-epoch RMSProp `full` run (`jax/runs/enet_b0_imagenet_bf16_350ep/`,
    against B0's paper 77.1 / 93.3). A config must carry the epoch count of the tier whose number its chapter prints, because
    `totalSteps := cfg.epochs * nb / accK` is what the schedule anneals over — 80 vs 350 is a
    different LR curve end to end, not a prefix of one, so the two results would not be comparable.

    The committed variants are the AdamW family (`adam64`, `adamdp64`) and the RMSProp family
    (`rms64`, `rmsdp64`, `emarms64*`, `emarmsdp64*`); only the RMSProp family matches the
    reference. -/
def efficientnetImagenetConfig : VerifiedConfig where
  epochs    := 350
  batchSize := 64
  -- **Depthwise fan = k², as the JAX reference.** Every JAX depthwise emitter hard-codes
  -- fan = k² (TF/timm's `variance_scaling` on a `(k,k,C,1)` kernel); `mkParam`'s He fan-out gave
  -- `2/(C·k²)`, a fraction of the reference's std on the 16 depthwise kernels (planning/init_parity.md §2a,
  -- imagenet_parity.md D4). Host-side, no re-render. On since 2026-10-07; verified runs before it
  -- started at the narrow fan, and `LEAN_MLIR_DW_FAN_K2=0` reproduces those from their seed.
  dwFanK2 := true
  -- **SE FCs at the reference's fan.** The 17 squeeze / excite denses are kind 7; the JAX
  -- reference inits them as 1×1 convs at var 2/out (TF fan-out), where Glorot gave a fraction of the std
  -- on every reduce FC and so started every SE gate near σ(0) (planning/init_parity.md §2b,
  -- imagenet_parity.md D5). Host-side. On since 2026-10-07; `LEAN_MLIR_SE_FAN_OUT=0` reproduces
  -- the runs before it from their seed.
  seFanOutInit := true

/-- Entry point. Defaults to the single-device `adam64` variant, matching the other three ImageNet
    drivers: a DP default makes a plain invocation die at the first step on a replica-count
    refusal, which reads as a broken build rather than a missing flag. -/
def runEfficientNetImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam64"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD efficientnetImagenetConfig.batchSize
  -- `rms*` selects the reference's OWN optimizer, so it also selects the reference's own schedule:
  -- RMSProp (ρ .9, μ .9, ε **1e-3**, coupled wd 1e-5 — baked by `rmsConstsBlock enetRmsHyper`) at
  -- peak 0.016 with 5-epoch warmup and ×0.97 every **2.4** epochs, mean-square init 1.0. Note the
  -- decay period is not 1 epoch here and mnv2's is; that difference is the whole reason
  -- `RmsSchedule` carries `decayEpochs` rather than the two nets sharing one constant.
  --
  -- With EMA (`ema…`), drop-connect (`drop`) and classifier dropout (`do`) the `emarmsdp64dropdo*`
  -- renders carry the whole reference recipe; the masks are drawn on the host rather than on
  -- device.
  let sched := enetImagenetRmsSchedule
  -- SUBSTRING, not prefix. Optimizer and EMA are INDEPENDENT axes in this net's variant names, so
  -- RMSProp+EMA is spelled `emarms`, which does NOT start with "rms" — and six committed artifacts
  -- are spelled that way, including the paper recipe `emarmsdp64dropdo`. Under a prefix test the
  -- SHARED TRAINER still classifies them as RMSProp (`Verified.Train`'s own test is a substring)
  -- and initialises the mean-square to 1.0, while this file hands them AdamW's 0.001 and a cosine
  -- schedule instead of the paper's 0.016 with ×0.97 every 2.4 epochs. That split is not loud: it
  -- descends and prints a normal log.
  -- `tests/TestVariantPredicates.lean` is the collision table, and this is its case 1.
  let rms := variant.contains "rms"
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => if rms then sched.lr
                else 0.001   -- NOT the reference's 0.016: that is an RMSProp rate and this path
                             -- is AdamW. 1e-3 is the AdamW default the other verified nets train
                             -- at, and it is the first knob to tune if this under- or over-steps.
  -- `LEAN_MLIR_EPOCHS` SETS the schedule where `LEAN_MLIR_MAX_EPOCHS` only CAPS it
  -- (`min n cfg.epochs`). `totalSteps := cfg.epochs * nb / accK` is what the schedule anneals over,
  -- so EPOCHS=80 is a complete 80-epoch experiment while MAX_EPOCHS=80 is a PREFIX of the committed
  -- schedule stopped with the LR high. Without this knob the committed count is also unprobeable:
  -- a short smoke run cannot be asked for. Spelled as in `MainResnet50Imagenet.lean`.
  -- Clear checkpoints when switching schedules; resuming across them fuses two LR curves silently.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD efficientnetImagenetConfig.epochs
  efficientnetImagenetVerified.toNet.trainAdamSched
    { efficientnetImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 (if rms then sched.warmup else 5) variant
    (if rms then sched.decayRate else 0.0) sched.decayEpochs
    (expStaircase := rms && sched.staircase)

def main (argv : List String) : IO Unit := runEfficientNetImagenet argv
