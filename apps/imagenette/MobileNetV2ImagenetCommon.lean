import LeanMlir.VerifiedNets

/-! # Shared body of the verified MobileNetV2 / **full ImageNet-1k** trainer

The fifth and last of the scale-tier trainers (§2p). Same certified renderer at `nClasses := 1000`,
`B := 64` — four replicas is **global batch 256**, which is `mobilenetV2ImagenetConfig.batchSize`,
so the 5004 steps/epoch match the reference exactly.

⚠ A batch-BN net: eval goes through `@mnv2in_fwd_eval` with frozen running stats, and the step
buffer carries a 2×52-tensor stat region. §2g's forward/train-step skew was found on THIS net —
`mobilenetv2_fwd` was batch-BN against a per-example-BN train step, so the trainer scored a
different net than it trained (logits rel 1.86). The forwards here come off the same chain the
train step differentiates, under their own slug.

⚠ **The optimizer does not match the reference**: MobileNetV2's recipe is **RMSProp at LR 0.045**
with the paper's schedule, where the verified path has AdamW + cosine — the same gap EfficientNet
has. On top of the usual missing mixup/cutmix/EMA. Read a result as the verified renderer training
this architecture, not as a MobileNetV2 reproduction.

```bash
(cd jax && lake exe resnet34-imagenet default --shim)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build mobilenetv2-imagenet-verified-xla
cat .lake/build/mnv2in_adamdp64_ckpt_xla.bin.epoch 2>/dev/null   # ⚠ READ THIS FIRST (§4)

PJRT_FFI_RESIDENT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LEAN_MLIR_VARIANT=adamdp64 LEAN_MLIR_BATCH=64 \
  LEAN_MLIR_REPLICAS=4 PJRT_REPLICAS=4 \
  .lake/build/bin/mobilenetv2-imagenet-verified-xla data 2>&1 | tee runs/mnv2in_4gpu.log
```
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
