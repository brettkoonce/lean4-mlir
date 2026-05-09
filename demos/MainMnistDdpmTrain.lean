import LeanMlir

/-! Tiny DDPM trainer on MNIST — DDPM demo Phase 1 smoke test.

    Architecture: tiny UNet (1-channel 28×28, 2 encoder/decoder pairs,
    base 16 channels, ~250K params). The model is trained to predict
    the noise ε that was added to a clean image x_0:

        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        loss = || ε_θ(x_t) - ε ||²  (per-pixel MSE, mean over B·C·H·W)

    No time conditioning yet — the model has no idea which `t` the
    input came from. Generated samples will be coarser than the
    canonical DDPM but the pipeline correctness can still be verified
    from training loss decreasing.

    Usage:
      lake exe mnist-ddpm-train [data/mnist]
-/

/-- 2-channel input: image + a scalar t/T_max timestep encoding tiled
    to the same spatial dims. The model output stays single-channel
    (predicted ε for the image). -/
def tinyDdpmUnet : NetSpec where
  name := "tiny DDPM UNet T-cond (MNIST 28x28x1)"
  imageH := 28
  imageW := 28
  layers := [
    .unetDown 2 16,
    .unetDown 16 32,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .unetUp 64 32,
    .unetUp 32 16,
    .conv2d 16 1 1 .same .identity
  ]

def tinyDdpmConfig : TrainConfig where
  learningRate := 0.0005
  batchSize    := 32
  epochs       := 3
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

def main (args : List String) : IO Unit := do
  let dataDir := args.head?.getD "data/mnist"
  let epochsOverride : Option Nat := match args with
    | _ :: e :: _ => e.toNat?
    | _ => none
  let spec := tinyDdpmUnet
  let cfg := { tinyDdpmConfig with
    epochs := epochsOverride.getD tinyDdpmConfig.epochs }
  IO.eprintln s!"{spec.name}: {spec.totalParams} params (epochs={cfg.epochs})"

  -- ── Compile vmfbs (forward + train_step with useDdpm) ──
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix
  let outShape : List Nat := [cfg.batchSize, 1, spec.imageH, spec.imageW]

  IO.eprintln "Generating train step MLIR..."
  let trainMlir := MlirCodegen.generateTrainStep spec cfg.batchSize
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := cfg.weightDecay)
    (useAdam := cfg.useAdam)
    (useDdpm := true) (ddpmOutShape := outShape)
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.eprintln s!"  {trainMlir.length} chars"

  -- iree-compile train step
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

  -- ── Load MNIST (60K × 28×28 f32 in [0, 1]) ──
  IO.eprintln "Loading MNIST..."
  let (trainImg, nTrain) ← F32.loadIdxImages s!"{dataDir}/train-images-idx3-ubyte"
  IO.eprintln s!"  train: {nTrain} images"

  -- ── Init params + Adam state ──
  let params ← spec.heInitParams
  let nP := spec.totalParams
  let adamM ← F32.const nP.toUSize 0.0
  let adamV ← F32.const nP.toUSize 0.0
  IO.eprintln s!"  {nP} params, {(params.size + adamM.size + adamV.size) / 1024 / 1024} MB"

  -- ── DDPM schedule (precomputed once) ──
  let alphaBar ← Ddpm.cosineSchedule 1000

  -- ── Training session ──
  let sess ← IreeSession.create vmfbPath
  IO.eprintln "  session loaded"

  -- Per-image element count for the IMAGE channel (1 channel pre-conditioning).
  -- After `prependTChannel` this doubles to 2 * H * W for the network input.
  let nPix : Nat := 1 * spec.imageH * spec.imageW
  let nPixCond : Nat := 2 * spec.imageH * spec.imageW
  let Tmax : Nat := 1000
  let bpE := nTrain / cfg.batchSize
  let allShapes := spec.shapesBA
  let bnShapes := spec.bnShapesBA
  let xSh := spec.xShape cfg.batchSize
  let nT := 3 * nP
  let batch : USize := cfg.batchSize.toUSize
  let outC : USize := 1
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
      -- x_0 batch (already [0, 1]; not centered to [-1, 1] for MVP)
      let x0 := F32.sliceImages trainImg (bi * cfg.batchSize) cfg.batchSize nPix
      -- Sample (x_t, eps, t) for this batch
      let stepSeed : USize := (epoch * 1000000 + bi).toUSize
      let (xt, ddpmRest) ← Ddpm.stepInputs x0 alphaBar batch nPix.toUSize stepSeed
      let (eps, tba) := ddpmRest
      -- Time conditioning: prepend a t/T_max-filled second channel so
      -- the network can see which timestep it's denoising. Output is
      -- [B, 2, H, W] flat = [B, 2*H*W].
      let xtCond ← Ddpm.prependTChannel xt tba batch (1 : USize) outH outW Tmax.toUSize
      let packed := (p.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepAdamF32Ddpm sess spec.trainFnName
                  packed allShapes xtCond xSh eps
                  cfg.learningRate globalStep.toFloat
                  bnShapes batch outC outH outW
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      p := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      let batchBnStats := out.extract ((nT + 1) * 4) ((nT + 1 + spec.nBnStats) * 4)
      let bnMom : Float := if globalStep == 1 then 1.0 else 0.1
      runningBnStats ← F32.ema runningBnStats batchBnStats bnMom
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    IO.eprintln s!"Epoch {epoch+1}/{cfg.epochs}: avg loss={epochLoss / bpE.toFloat} ({t1-t0}ms)"

  IO.FS.writeBinFile s!"{pfx}_params.bin" p
  IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + BN stats."
