import LeanMlir.VerifiedNetsCore
import LeanMlir.VerifiedTrain

/-! # `efficientnet-imagenet-verified` — EfficientNet-B0 on full ImageNet-1k, verified → XLA

The fourth and last of the ImageNet scale-tier trainers (§2p). `B` and `nClasses` were already
renderer parameters, so this needed only a `slug` — plus the derived −α/K that turned up the third
copy of §2k's hardcoded-K bug, this time in EfficientNet's report-only loss.

⚠ Does NOT move the verification tier. The optimizer follows the variant: the `rms*`/`emarms*`
renders are the reference's RMSProp with its ×0.97-every-2.4 decay, EMA, drop-connect and
classifier dropout; the `adam*` renders are AdamW + cosine. The shipping
`emarmsdp64dropdowxeps0001bf16` is the TF recipe the JAX `full` config now carries: `wx`, BN ε 1e-3,
the staircase on the global step (`enetImagenetRmsSchedule`) and the i/16 drop-connect ramp.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything.
-/

/-- 350 epochs at 64 per device — the phase-2 tier this net is measured against, and at four
    replicas its global batch of 256. `batchSize` is PER DEVICE and must match the batch the
    variant was rendered at.

    ⚠ **This was 80, and 80 was the wrong number to carry.** The JAX reference the chapter prints
    is the 350-epoch RMSProp `full` run (77.15% / 93.30%, against B0's paper 77.1 / 93.3). A
    phase-4 config must carry the epoch count of the tier whose number its chapter prints, because
    `totalSteps := cfg.epochs * nb / accK` is what the schedule anneals over — 80 vs 350 is a
    different LR curve end to end, not a prefix of one, so the two results would not be comparable.

    The committed variants are the AdamW family (`adam64`, `adamdp64`) and the RMSProp family
    (`rms64`, `rmsdp64`, `emarms64*`, `emarmsdp64*`); only the RMSProp family matches the
    reference. -/
def efficientnetImagenetConfig : VerifiedConfig where
  epochs    := 350
  batchSize := 64

/-- Entry point. Defaults to the single-device `adam64` variant, matching the other three ImageNet
    drivers: a DP default makes a plain invocation die at the first step on a replica-count
    refusal, which reads as a broken build rather than a missing flag. -/
def runEfficientNetImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam64"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD efficientnetImagenetConfig.batchSize
  -- ▶ `rms*` selects the reference's OWN optimizer, so it also selects the reference's own schedule:
  -- RMSProp (ρ .9, μ .9, ε **1e-3**, coupled wd 1e-5 — baked by `rmsConstsBlock enetRmsHyper`) at
  -- peak 0.016 with 5-epoch warmup and ×0.97 every **2.4** epochs, mean-square init 1.0. ⚠ Note the
  -- decay period is not 1 epoch here and mnv2's is; that difference is the whole reason
  -- `RmsSchedule` carries `decayEpochs` rather than the two nets sharing one constant.
  --
  -- ▶ With EMA (`ema…`), drop-connect (`drop`) and classifier dropout (`do`) the `emarmsdp64dropdo*`
  -- renders carry the whole reference recipe; the remaining differences are listed in
  -- planning/imagenet_parity.md §2.2 (BN group before the sync-BN render, host-drawn masks).
  let sched := enetImagenetRmsSchedule
  -- ⚠ SUBSTRING, not prefix, and the prefix version was a live bug here. Optimizer and EMA are
  -- INDEPENDENT axes in this net's variant names, so RMSProp+EMA is spelled `emarms`, which does
  -- NOT start with "rms" — and six committed artifacts are spelled that way, including the paper
  -- recipe `emarmsdp64dropdo`. Under a prefix test the SHARED TRAINER still classified them as
  -- RMSProp (`VerifiedTrain.lean`'s own test is a substring) and initialised the mean-square to
  -- 1.0, while this file handed them AdamW's 0.001 and a cosine schedule instead of the paper's
  -- 0.016 with ×0.97 every 2.4 epochs. That split is not loud: it descends and prints a normal log.
  -- `tests/TestVariantPredicates.lean` is the collision table, and this is its case 1.
  let rms := variant.contains "rms"
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => if rms then sched.lr
                else 0.001   -- ⚠ NOT the reference's 0.016: that is an RMSProp rate and this path
                             -- is AdamW. 1e-3 is the AdamW default the other verified nets train
                             -- at, and it is the first knob to tune if this under- or over-steps.
  -- ⚠ `LEAN_MLIR_EPOCHS` SETS the schedule where `LEAN_MLIR_MAX_EPOCHS` only CAPS it
  -- (`min n cfg.epochs`). `totalSteps := cfg.epochs * nb / accK` is what the schedule anneals over,
  -- so EPOCHS=80 is a complete 80-epoch experiment while MAX_EPOCHS=80 is a PREFIX of the committed
  -- schedule stopped with the LR high. Without this knob the committed count is also unprobeable:
  -- a short smoke run cannot be asked for. Spelled as in `MainResnet50Imagenet.lean`.
  -- ⚠ Clear checkpoints when switching schedules; resuming across them fuses two LR curves silently.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD efficientnetImagenetConfig.epochs
  efficientnetImagenetVerified.toNet.trainAdamSched
    { efficientnetImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 (if rms then sched.warmup else 5) variant
    (if rms then sched.decayRate else 0.0) sched.decayEpochs
    (expStaircase := rms && sched.staircase)

def main (argv : List String) : IO Unit := runEfficientNetImagenet argv
