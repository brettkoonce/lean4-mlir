import LeanMlir.VerifiedNets

/-! # `efficientnet-imagenet-verified` — EfficientNet-B0 on full ImageNet-1k, verified → XLA

The fourth and last of the ImageNet scale-tier trainers (§2p). `B` and `nClasses` were already
renderer parameters, so this needed only a `slug` — plus the derived −α/K that turned up the third
copy of §2k's hardcoded-K bug, this time in EfficientNet's report-only loss.

⚠ Does NOT move the verification tier, and the OPTIMIZER does not match the reference (RMSProp +
exponential decay there, AdamW + cosine here). See `the net's Main file`.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything.
-/

/-- 350 epochs at 64 per device — the phase-2 tier this net is measured against, and at four
    replicas its global batch of 256. `batchSize` is PER DEVICE and must match the batch the
    variant was rendered at.

    ⚠ **This was 80, and 80 was the wrong number to carry.** The chapter reports two phase-2 tiers:
    an 80-epoch SGD validation tier (72.31%) and the faithful 350-epoch RMSProp run (76.80% /
    93.26%, against B0's paper 77.1 / 93.3). A phase-4 config must carry the epoch count of the
    tier whose number its chapter prints, because `totalSteps := cfg.epochs * nb / accK` is what the
    schedule anneals over — 80 vs 350 is a different LR curve end to end, not a prefix of one, so
    the two results would not be comparable.

    ▶ And here the choice is forced, not merely preferred: **there is no momentum/SGD render for
    `efficientnetin`**. The committed variants are the AdamW family (`adam64`, `adamdp64`) and the
    RMSProp family (`rms64`, `rmsdp64`, `emarms64*`, `emarmsdp64*`). The 80-epoch SGD tier is
    therefore not reproducible on the verified path at all, and the 350-epoch RMSProp tier is the
    only phase-2 result these artifacts can be pointed at. -/
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
  -- ⚠ RMSProp is one of TWO gaps to the reference's 72.31% on this net (`recipe_gaps.md` §2) —
  -- stochastic depth and EMA are still missing — so this is not yet a matched pair the way mnv2 is.
  let sched := enetRmsSchedule
  -- ⚠ SUBSTRING, not prefix, and the prefix version was a live bug here. Optimizer and EMA are
  -- INDEPENDENT axes in this net's variant names, so RMSProp+EMA is spelled `emarms`, which does
  -- NOT start with "rms" — and six committed artifacts are spelled that way, including the paper
  -- recipe `emarmsdp64dropdo`. Under a prefix test the SHARED TRAINER still classified them as
  -- RMSProp (`VerifiedTrain.lean`'s own test is a substring) and initialised the mean-square to
  -- 1.0, while this file handed them AdamW's 0.001 and a cosine schedule instead of the paper's
  -- 0.016 with ×0.97 every 2.4 epochs. That split is not loud: it descends and prints a normal log.
  -- `tests/TestVariantPredicates.lean` is the collision table, and this is its case 1.
  let rms := (variant.splitOn "rms").length > 1
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

def main (argv : List String) : IO Unit := runEfficientNetImagenet argv
