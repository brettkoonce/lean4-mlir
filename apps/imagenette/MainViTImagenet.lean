import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Train

/-! # `vit-imagenet-verified` — ViT-Tiny on full ImageNet-1k, verified renderer → XLA/PJRT

The ViT peer of `resnet34-imagenet-verified` (handoff §2p). Nothing new was needed in the
renderer to get here — `nClasses`, `bs` and `replicas` are all parameters of it, so the three
ImageNet artifacts are three `#eval`s, exactly as §2k found for ResNet-34.

XLA-only by construction: collectives live on the PJRT path, and IREE has no measured
ImageNet-scale number here to compare against.

⚠ **This does not move the verification tier** — the proof-carrying claims stop at Imagenette.
See `vitImagenetVerified`'s claim-ceiling note for what the variant carries of the DeiT recipe.

▶ The job is `scripts/jobs/vit-default-emabf16-4gpu.conf` (`emadp128x4wxclipdropbf16`: EMA, clip,
drop-path, weight decay off norm/bias, bf16, 4 × 128 = global 512). It sets `SHIM_WORKERS`,
`PJRT_FFI_RESIDENT`, `LEAN_MLIR_EPOCHS` and both replica knobs; run the job, not the bare binary,
for anything printable. The worker count is an open probe (planning/imagenet_parity.md VT-2).

```bash
scripts/gen_shims.sh                       # this net's OWN data shim (⚠ NOT R34's — see VerifiedNet.shimScript)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build vit-imagenet-verified
scripts/supervise.sh vit-default-emabf16-4gpu
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
  -- ⭐ timm/DeiT init, matching the phase-2 reference run's `deit-init` recipe (blueprint §9.6).
  -- Without it every transformer Linear is Glorot, which at ViT-Ti's d=192 is 3.6× wider than
  -- timm's fixed 0.02, and the CLS token is 5× wider — the one axis §9.6 credits for reaching
  -- the paper's number. Init is host-side, so this needs no re-render.
  vitInit   := true

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
  -- ⚠⚠ `emaDecay` IS NAMED, and it has to be. `trainAdamSched`'s default is **0.9999**, while
  -- `vitTinyImagenetConfig.emaDecay` — the DeiT default the reference run used — is **0.99996**.
  -- Positional args stop at `variant`, so an EMA variant launched without this line trains against
  -- a shadow that averages ~4× faster than its reference's and reports it as the pair.
  -- ⚠ Inert unless the variant selects EMA (`VerifiedVariant.emaOn`, i.e. a name starting "ema"),
  -- so it costs the non-EMA renders nothing.
  vitImagenetVerified.toNet.trainAdamSched
    { vitImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant (emaDecay := 0.99996)

def main (argv : List String) : IO Unit := runViTImagenet argv
