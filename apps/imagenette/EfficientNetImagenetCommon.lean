import LeanMlir.VerifiedNets

/-! # Shared body of the verified EfficientNet-B0 / **full ImageNet-1k** trainer

The EfficientNet peer of the R34, ViT and ConvNeXt ImageNet trainers (§2p). Same certified renderer
at `nClasses := 1000`, `B := 64` — so four replicas is **global batch 256**, which is
`efficientNetB0ImagenetConfig.batchSize`. Matching the reference's global batch is what keeps the
two runs a comparable pair.

⚠ **First ImageNet net here with BatchNorm.** Consequences: eval goes through `@enetin_fwd_eval`
with frozen running stats (batch-BN eval is degenerate on a sorted val split), and the per-step
buffer carries a 2×49-tensor running-stat region that the LayerNorm nets do not have.

⚠ **The optimizer does NOT match the reference and this is the largest recipe gap of the four.**
`efficientNetB0ImagenetConfig` uses **RMSProp with exponential LR decay** (×0.97 every 2.4 epochs,
base LR 0.016@bs256, `cosineDecay := false`); the verified path has AdamW + cosine only. That is on
top of the usual missing mixup/cutmix/EMA. Do not read a number from this as an EfficientNet-B0
reproduction — read it as the verified renderer training the same architecture.

```bash
(cd jax && lake exe resnet34-imagenet default --shim)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build efficientnet-imagenet-verified-xla
cat .lake/build/enetin_adamdp64_ckpt_xla.bin.epoch 2>/dev/null   # ⚠ READ THIS FIRST (§4)

PJRT_FFI_RESIDENT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LEAN_MLIR_VARIANT=adamdp64 LEAN_MLIR_BATCH=64 \
  LEAN_MLIR_REPLICAS=4 PJRT_REPLICAS=4 \
  .lake/build/bin/efficientnet-imagenet-verified-xla data 2>&1 | tee runs/enetin_4gpu.log
```
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
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.001   -- ⚠ NOT the reference's 0.016: that is an RMSProp rate and this path is
                        -- AdamW. 1e-3 is the AdamW default the other verified nets train at, and
                        -- it is the first knob to tune if this under- or over-steps.
  efficientnetImagenetVerified.toNet.trainAdamSched
    { efficientnetImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant
