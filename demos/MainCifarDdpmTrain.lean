import LeanMlir

/-! Tiny DDPM trainer on CIFAR-10 — DDPM demo Phase 3.

    Same shape as the MNIST DDPM trainer, but on 32×32 RGB:
      input = 3 image channels + 1 t-channel = 4 channels
      output = 3 channels (predicted ε per RGB channel)

    Tiny UNet, ~120K params. Loss is per-pixel MSE. CIFAR-10 is
    significantly harder than MNIST (real photos, not centered
    geometric shapes), so this run is honestly a "does the pipeline
    handle multi-channel" check more than a "produces sharp samples"
    expectation. With only 50 epochs and a 120K model, expect
    blurry-but-coherent CIFAR-style chromatic blobs.

    Usage: lake exe cifar-ddpm-train [data] [epochs]
-/

/-- Base 80 (~3M params). 2.8× the base-48 capacity; previous base-48
    run plateaued in MSE around epoch 30, suggesting capacity was
    the binding constraint. Per-step ~280ms vs 100ms; 70 epochs
    targets an 8-9 hour overnight run. Data is centered to [-1, 1]
    in the trainer (inverted in the sampler), standard DDPM. -/
def tinyCifarDdpm : NetSpec where
  name := "DDPM UNet T-cond base80 centered (CIFAR 32x32x3)"
  imageH := 32
  imageW := 32
  layers := [
    .unetDown 4 80,                 -- 4 input channels (3 RGB + 1 t)
    .unetDown 80 160,
    .convBn 160 320 3 1 .same,
    .convBn 320 320 3 1 .same,
    .unetUp 320 160,
    .unetUp 160 80,
    .conv2d 80 3 1 .same .identity  -- 3 RGB output channels
  ]

def cifarDdpmConfig : TrainConfig where
  learningRate := 0.0005
  batchSize    := 32
  epochs       := 3
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

/-- Inline CIFAR-10 loader. Reads the 5 train .bin files and returns
    the f32 image buffer + count. No labels (DDPM is unconditional). -/
private def loadCifar (dataDir : String) : IO (ByteArray × Nat) := do
  let mut raw : ByteArray := .empty
  let mut nTotal : Nat := 0
  for i in [1:6] do
    let batchRaw ← IO.FS.readBinFile s!"{dataDir}/cifar-10/data_batch_{i}.bin"
    let n := batchRaw.size / 3073
    raw := raw.append batchRaw
    nTotal := nTotal + n
  let imgs ← F32.cifarBatch raw 0 nTotal.toUSize
  return (imgs, nTotal)

def main (args : List String) : IO Unit := do
  let dataDir := args.head?.getD "data"
  let epochsOverride : Option Nat := match args with
    | _ :: e :: _ => e.toNat?
    | _ => none
  let spec := tinyCifarDdpm
  let cfg := { cifarDdpmConfig with
    epochs := epochsOverride.getD cifarDdpmConfig.epochs }
  IO.eprintln s!"{spec.name}: {spec.totalParams} params (epochs={cfg.epochs})"

  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix
  let outShape : List Nat := [cfg.batchSize, 3, spec.imageH, spec.imageW]

  IO.eprintln "Generating train step MLIR..."
  let trainMlir := MlirCodegen.generateTrainStep spec cfg.batchSize
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := cfg.weightDecay)
    (useAdam := cfg.useAdam)
    (useDdpm := true) (ddpmOutShape := outShape)
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.eprintln s!"  {trainMlir.length} chars"

  IO.eprintln "Compiling vmfb..."
  let compileMlir : String → String → IO Bool := fun mlirPath outPath => do
    let args ← ireeCompileArgs mlirPath outPath
    let compiler ← if (← System.FilePath.pathExists ".venv/bin/iree-compile")
                   then pure ".venv/bin/iree-compile" else pure "iree-compile"
    let r ← IO.Process.output { cmd := compiler, args := args }
    if r.exitCode != 0 then
      IO.eprintln s!"iree-compile failed: {r.stderr.take 3000}"
      return false
    return true
  let vmfbPath := s!"{pfx}_train_step.vmfb"
  unless (← compileMlir s!"{pfx}_train_step.mlir" vmfbPath) do IO.Process.exit 1
  IO.eprintln "  train step compiled"

  IO.eprintln "Loading CIFAR-10..."
  let (trainImgRaw, nTrain) ← loadCifar dataDir
  -- DDPM standard: center data to [-1, 1] so the SNR of N(0, I) noise
  -- isn't biased by an off-center signal.
  let trainImg ← F32.scaleShift trainImgRaw 2.0 (-1.0)
  IO.eprintln s!"  train: {nTrain} images (centered to [-1, 1])"

  let params ← spec.heInitParams
  let nP := spec.totalParams
  let adamM ← F32.const nP.toUSize 0.0
  let adamV ← F32.const nP.toUSize 0.0
  IO.eprintln s!"  {nP} params, {(params.size + adamM.size + adamV.size) / 1024 / 1024} MB"

  let alphaBar ← Ddpm.cosineSchedule 1000

  let sess ← IreeSession.create vmfbPath
  IO.eprintln "  session loaded"

  -- Per-image element count: 3 channels × 32 × 32 = 3072 floats.
  let nPix : Nat := 3 * spec.imageH * spec.imageW
  let Tmax : Nat := 1000
  let bpE := nTrain / cfg.batchSize
  let allShapes := spec.shapesBA
  let bnShapes := spec.bnShapesBA
  let xSh := spec.xShape cfg.batchSize
  let nT := 3 * nP
  let batch : USize := cfg.batchSize.toUSize
  let imgC : USize := 3
  let outH : USize := spec.imageH.toUSize
  let outW : USize := spec.imageW.toUSize

  let mut p := params
  let mut m := adamM
  let mut v := adamV
  let mut globalStep : Nat := 0
  let mut runningBnStats ← F32.const spec.nBnStats.toUSize 0.0

  IO.eprintln s!"training: {bpE} batches/epoch, batch={cfg.batchSize}, lr={cfg.learningRate}"
  for epoch in [:cfg.epochs] do
    let t0 ← IO.monoMsNow
    let mut epochLoss : Float := 0.0
    for bi in [:bpE] do
      globalStep := globalStep + 1
      let x0 := F32.sliceImages trainImg (bi * cfg.batchSize) cfg.batchSize nPix
      let stepSeed : USize := (epoch * 1000000 + bi).toUSize
      let (xt, ddpmRest) ← Ddpm.stepInputs x0 alphaBar batch nPix.toUSize stepSeed
      let (eps, tba) := ddpmRest
      let xtCond ← Ddpm.prependTChannel xt tba batch imgC outH outW Tmax.toUSize
      let packed := (p.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepAdamF32Ddpm sess spec.trainFnName
                  packed allShapes xtCond xSh eps
                  cfg.learningRate globalStep.toFloat
                  bnShapes batch imgC outH outW
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      p := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      let batchBnStats := out.extract ((nT + 1) * 4) ((nT + 1 + spec.nBnStats) * 4)
      let bnMom : Float := if globalStep == 1 then 1.0 else 0.1
      runningBnStats ← F32.ema runningBnStats batchBnStats bnMom
      if bi < 3 || bi % 200 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    IO.eprintln s!"Epoch {epoch+1}/{cfg.epochs}: avg loss={epochLoss / bpE.toFloat} ({t1-t0}ms)"

  IO.FS.writeBinFile s!"{pfx}_params.bin" p
  IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + BN stats."
