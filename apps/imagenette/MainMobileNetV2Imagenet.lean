import LeanMlir.VerifiedNets

/-! # `mobilenetv2-imagenet-verified` — MobileNetV2 on full ImageNet-1k, verified → XLA

The last of the five scale-tier trainers (§2p). Needed only a `slug` plus derived label-smoothing
constants — and mnv2 was the worst of the five on that axis, carrying the K=10 value in the
COTANGENT (the gradient path, §2k's original bug) as well as in the report-only loss.

⚠ Does NOT move the verification tier; optimizer does not match the reference (RMSProp there,
AdamW here). See `MobileNetV2ImagenetCommon`.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything.
-/

/-- 300 epochs at 64 per device — the reference's schedule length, and at four replicas its global
    batch of 256. `batchSize` is PER DEVICE and must match the batch the variant was rendered at. -/
def mobilenetv2ImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 64

/-- Entry point. Defaults to the single-device `adam64` variant, matching the other four ImageNet
    drivers — a DP default dies at the first step on a replica-count refusal, which reads as a
    broken build rather than a missing flag. -/
def runMobileNetV2Imagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam64"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD mobilenetv2ImagenetConfig.batchSize
  -- ▶ `rms*` selects the reference's OWN optimizer, so it also selects the reference's own schedule.
  -- At `rms64`/`rmsdp64` this driver is the MobileNetV2 recipe: RMSProp (ρ .9, μ .9, ε 1.0, coupled
  -- wd 4e-5 — all baked by `rmsConstsBlock mnv2RmsHyper`) at peak LR 0.045 with 5-epoch warmup and
  -- ×0.98 per epoch after it, mean-square initialised to 1.0 by the driver. Every one of those is
  -- `mobilenetV2ImagenetConfig`'s value, and the LR/schedule half comes off `mnv2RmsSchedule` so
  -- the Imagenette peer cannot carry a different 0.98.
  let sched := mnv2RmsSchedule
  let rms := variant.startsWith "rms"
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => if rms then sched.lr
                else 0.001   -- ⚠ NOT the reference's 0.045: that is an RMSProp rate, this is AdamW.
  mobilenetv2ImagenetVerified.toNet.trainAdamSched
    { mobilenetv2ImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 (if rms then sched.warmup else 5) variant
    (if rms then sched.decayRate else 0.0) sched.decayEpochs

def main (argv : List String) : IO Unit := runMobileNetV2Imagenet argv
