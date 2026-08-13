import LeanMlir.VerifiedNets

/-! # `vit-imagenet-verified` — ViT-Tiny on full ImageNet-1k, verified renderer → XLA/PJRT

The ViT peer of `resnet34-imagenet-verified` (handoff §2p). Nothing new was needed in the
renderer to get here — `nClasses`, `bs` and `replicas` are all parameters of it, so the three
ImageNet artifacts are three `#eval`s, exactly as §2k found for ResNet-34.

XLA-only by construction: collectives live on the PJRT path, and IREE has no measured
ImageNet-scale number here to compare against.

⚠ **This does not move the verification tier** — the proof-carrying claims stop at Imagenette. See
`the net's Main file`'s claim-ceiling note, and note in particular that this is not the DeiT recipe.

⚠ **Set `SHIM_WORKERS=2`.** ViT is the first net here whose step rate outruns a single shim
producer (~1,940 img/s wanted against ~1,530 delivered); without it the GPUs wait on data.

```bash
scripts/gen_shims.sh                       # this net's OWN data shim (⚠ NOT R34's — see VerifiedNet.shimScript)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build vit-imagenet-verified
PJRT_FFI_RESIDENT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 SHIM_WORKERS=2 \
  LEAN_MLIR_VARIANT=adamdp128x4 LEAN_MLIR_BATCH=128 \
  LEAN_MLIR_REPLICAS=4 PJRT_REPLICAS=4 \
  .lake/build/bin/vit-imagenet-verified data
```

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything.
-/

/-- 300 epochs at 128 per device — the DeiT-Ti schedule length, and at four replicas the
    reference's global batch of 512. `batchSize` is PER DEVICE, as everywhere else here
    (`LEAN_MLIR_BATCH` overrides, and must match the batch the selected variant was rendered at,
    since it is baked into the graph). -/
def vitImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 128

/-- Entry point. Defaults to the **single-device** `adam128` variant rather than the four-replica
    one, matching `runResnet34Imagenet`: a DP default would make a plain invocation fail at the
    first step with a replica-count refusal, which reads as a broken build rather than a missing
    flag. `LEAN_MLIR_VARIANT=adamdp128x4` selects the 4-GPU render. -/
def runViTImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam128"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD vitImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.0005   -- `vitTinyImagenetConfig.learningRate`, the DeiT batch-512 rate. The
                         -- Imagenette ViT driver's 3e-4 is tuned for global batch 32 and would
                         -- under-step this by ~1.7x.
  -- ⚠ `LEAN_MLIR_EPOCHS` SETS the schedule where `LEAN_MLIR_MAX_EPOCHS` only CAPS it
  -- (`min n cfg.epochs`). `totalSteps := cfg.epochs * nb / accK` is what the cosine anneals over,
  -- so `EPOCHS=30` is a complete 30-epoch experiment while `MAX_EPOCHS=30` is a PREFIX of the
  -- committed 300-epoch decay stopped with the LR still high. ⚠ Clear checkpoints when switching
  -- schedules; resuming across them fuses two LR curves silently. Without this knob a 300-epoch
  -- commitment makes the net unprobeable, which is what it was before 2026-08-12.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD vitImagenetConfig.epochs
  vitImagenetVerified.toNet.trainAdamSched
    { vitImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant

def main (argv : List String) : IO Unit := runViTImagenet argv
