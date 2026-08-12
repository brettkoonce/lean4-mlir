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

/-- **350 epochs** at 64 per device, and at four replicas the reference's global batch of 256.
    `batchSize` is PER DEVICE and must match the batch the variant was rendered at.

    ⭐ 350, not 300, because **this trainer's job is to match the phase-2 run**, and the phase-2
    number this net is measured against (71.44% / 90.34%) is the paper-faithful **350-epoch**
    exponential-decay tier — `blueprint`'s "The paper-faithful tier: 350 epochs". A 300-epoch
    phase-4 run would not be comparable to it: `totalSteps := cfg.epochs * nb / accK` is what the
    schedule anneals over, so 300 vs 350 is a different LR curve end to end, not a shorter version
    of the same one. ⚠ It was 300 until 2026-08-12, which made the driver and the section it is
    supposed to reproduce quietly disagree. -/
def mobilenetv2ImagenetConfig : VerifiedConfig where
  epochs    := 350
  batchSize := 64

/-- Entry point. Defaults to the single-device `adam64` variant, matching the other four ImageNet
    drivers — a DP default dies at the first step on a replica-count refusal, which reads as a
    broken build rather than a missing flag. -/
def runMobileNetV2Imagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam64"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD mobilenetv2ImagenetConfig.batchSize
  -- ⚠ `LEAN_MLIR_EPOCHS` SETS the schedule where `LEAN_MLIR_MAX_EPOCHS` only CAPS it: `totalSteps
  -- := cfg.epochs * nb / accK` is what the cosine anneals over, so `EPOCHS=30` is a complete
  -- 30-epoch experiment while `MAX_EPOCHS=30` is a PREFIX of the 350-epoch decay stopped with the
  -- LR still high. Present on the R50 and MNv4 ImageNet drivers since they were written; this one
  -- went without it until 2026-08-12, which made a short probe of this net IMPOSSIBLE — the
  -- knob was silently ignored and the run went for the full committed schedule.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD mobilenetv2ImagenetConfig.epochs
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
    { mobilenetv2ImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 (if rms then sched.warmup else 5) variant
    (if rms then sched.decayRate else 0.0) sched.decayEpochs

def main (argv : List String) : IO Unit := runMobileNetV2Imagenet argv
