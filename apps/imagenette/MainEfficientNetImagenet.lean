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

/-- 80 epochs at 64 per device — the reference's schedule length, and at four replicas its global
    batch of 256. `batchSize` is PER DEVICE and must match the batch the variant was rendered at. -/
def efficientnetImagenetConfig : VerifiedConfig where
  epochs    := 80
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
  let rms := variant.startsWith "rms"
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => if rms then sched.lr
                else 0.001   -- ⚠ NOT the reference's 0.016: that is an RMSProp rate and this path
                             -- is AdamW. 1e-3 is the AdamW default the other verified nets train
                             -- at, and it is the first knob to tune if this under- or over-steps.
  efficientnetImagenetVerified.toNet.trainAdamSched
    { efficientnetImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 (if rms then sched.warmup else 5) variant
    (if rms then sched.decayRate else 0.0) sched.decayEpochs

def main (argv : List String) : IO Unit := runEfficientNetImagenet argv
