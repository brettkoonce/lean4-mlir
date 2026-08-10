import LeanMlir.VerifiedNets

/-! # `resnet34-imagenet-verified` — ResNet-34 on full ImageNet-1k, verified renderer → XLA/PJRT

The scale tier of handoff §2k, and the payoff the whole verified track was pointed at: the certified
renderer at ImageNet shapes, fed by the same augmentation pipeline the Lean→JAX reference trainer
uses, so the two are a **matched pair** and JAX is an external oracle rather than a separate story.

Nothing new was needed in the renderer to get here — `nClasses`, `B`, `opt` and `slug` are all
parameters of it, so the three ImageNet artifacts are three `#eval`s. What was needed was the data
path: `VerifiedData.imagenet` and the tfds shim.

XLA-only by construction: the shim path is where the throughput is, and IREE has no measured
ImageNet-scale number here to compare against.

⚠ **This does not move the verification tier** — the proof-carrying claims stop at Imagenette. See
`the net's Main file`'s claim-ceiling note before quoting a result from it.

```bash
scripts/gen_shims.sh                       # this net's OWN data shim (⚠ NOT R34's — see VerifiedNet.shimScript)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet34-imagenet-verified
HIP_VISIBLE_DEVICES=0 LEAN_MLIR_BASE_LR_U=100000 \
  .lake/build/bin/resnet34-imagenet-verified data
```

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). The `-xla` suffix is
gone from the target name because it no longer distinguishes anything.
-/

/-- 30 epochs at batch 256 — the `resnet34ImagenetConfigShort` tier on the JAX side, i.e. the
    validation subrun rather than the 90-epoch paper recipe. At the measured bs256 rate that is
    ~28 h on one 7900 XTX (~17 h on two), against the reference's ~5 h for the same tier on 4 CUDA
    cards at bf16. 5-epoch warmup matches the reference; the schedule is cosine as everywhere here. -/
def resnet34ImagenetConfig : VerifiedConfig where
  epochs    := 30
  batchSize := 256

/-- Entry point. Defaults to the `mom256` variant — the ImageNet artifact — rather than `adam`,
    since that is the only render this spec's slug has; `LEAN_MLIR_VARIANT` still overrides. -/
def runResnet34Imagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "mom256"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD resnet34ImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.1        -- the reference rate; unlike the Imagenette driver this net has no
                           -- AdamW variant, so an Adam default here would only ever be wrong.
  resnet34ImagenetVerified.toNet.trainAdamSched
    { resnet34ImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant

def main (argv : List String) : IO Unit := runResnet34Imagenet argv
