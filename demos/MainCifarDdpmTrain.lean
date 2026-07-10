import LeanMlir

/-! Tiny DDPM trainer on CIFAR-10 — DDPM demo, v2 Phase 0.

    Same base80 T-conditioned UNet as before, now with the three
    Phase-0 recipe levers from planning/ddpm_demo_v2.md (Workstream B),
    none of which need new codegen:

      * EMA of weights (decay 0.9999) — the plan's single cheapest
        untested quality lever. A shadow param buffer tracks
        `ema = 0.9999·ema + 0.0001·p` every step; saved as
        `_params_ema.bin` beside `_params.bin`.
      * Fixed-seed DDIM sample grid every `sampleEvery` epochs (fixed
        noise seed) → `runs/.../samples_ep{N}.ppm` from the RAW
        weights (EMA weights mismatch the raw-weight BN running stats
        — Gate-A verdict). Makes every DDPM A/B honest (MSE ≠ sample
        quality).
      * Per-image horizontal-flip augment (`F32.hflipNCHW`).

    Usage: lake exe cifar-ddpm-train [data] [epochs] [maxSteps]
      maxSteps>0 caps steps/epoch (smoke testing).
-/

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

/-- v2 Workstream A variant: per-block `timeCondAdd` injects a learned
    sin/cos time projection after each stage (in-graph timestep, no ABI
    change). `nFreq=8` → 16 sin/cos features per site. -/
def tinyCifarDdpmTC : NetSpec where
  name := "DDPM UNet timeCondAdd base80 (CIFAR 32x32x3)"
  imageH := 32
  imageW := 32
  layers := [
    .unetDown 4 80,
    .timeCondAdd 80 8,
    .unetDown 80 160,
    .timeCondAdd 160 8,
    .convBn 160 320 3 1 .same,
    .timeCondAdd 320 8,
    .convBn 320 320 3 1 .same,
    .timeCondAdd 320 8,
    .unetUp 320 160,
    .timeCondAdd 160 8,
    .unetUp 160 80,
    .timeCondAdd 80 8,
    .conv2d 80 3 1 .same .identity
  ]

/-- Phase-2 capacity leg (B4): same plain UNet at base96 (~4.2M params). -/
def tinyCifarDdpm96 : NetSpec where
  name := "DDPM UNet T-cond base96 centered (CIFAR 32x32x3)"
  imageH := 32
  imageW := 32
  layers := [
    .unetDown 4 96,
    .unetDown 96 192,
    .convBn 192 384 3 1 .same,
    .convBn 384 384 3 1 .same,
    .unetUp 384 192,
    .unetUp 192 96,
    .conv2d 96 3 1 .same .identity
  ]

def cifarDdpmConfig : TrainConfig where
  learningRate := 0.0005
  batchSize    := 32
  epochs       := 3
  useAdam      := true
  weightDecay  := 0.0
  -- Constant LR wanders late in training: the 100-ep base80 run cycled
  -- through global color modes ep75→100 (magenta collapse at ep100, best
  -- grid ep90) at dead-flat MSE. Cosine decay pins the endpoint.
  cosineDecay  := true
  warmupEpochs := 0
  augment      := false

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

/-- Map [-1, 1] (DDPM-centered) back to [0, 255] uint8, clamped. -/
private def floatToU8 (v : Float) : UInt8 :=
  let scaled := (v + 1.0) * 0.5
  let p := if scaled < 0.0 then 0.0 else if scaled > 1.0 then 1.0 else scaled
  (p * 255.0).toUInt8

/-- Render a fixed-seed 4×4 DDIM sample grid from `evalParams` (raw
    weights + matching BN running stats) to `outPath`. 50 DDIM steps,
    seed fixed so grids are comparable across epochs. -/
private def sampleGrid (spec : NetSpec) (evalSess : IreeSession)
    (evalParams evalShapes : ByteArray) (alphaBar : ByteArray)
    (outPath : String) : IO Unit := do
  let B : Nat := 16
  let imgC : Nat := 3
  let H := spec.imageH; let W := spec.imageW
  let nPix : Nat := imgC * H * W
  let xShape := spec.xShape B
  let T : Nat := 1000
  let nSteps : Nat := 50
  let stride : Nat := T / nSteps
  let alphaBarF : Nat → Float := fun t => F32.read alphaBar t.toUSize
  let mut x ← Ddpm.sampleNoise (B * nPix).toUSize 0xfeed5eed
  let nTotal : USize := (B * nPix).toUSize
  for k in [:nSteps] do
    let t := T - 1 - k * stride
    let tPrev : Nat := if k + 1 < nSteps then T - 1 - (k + 1) * stride else 0
    let xCond ← Ddpm.prependTChannelScalar x B.toUSize imgC.toUSize
                  H.toUSize W.toUSize t.toUSize T.toUSize
    let eps ← IreeSession.forwardF32 evalSess spec.evalFnName
                evalParams evalShapes xCond xShape B.toUSize nPix.toUSize
    let aBarT := alphaBarF t
    let aBarP := if k + 1 < nSteps then alphaBarF tPrev else 0.9999
    let a := Float.sqrt aBarP / Float.sqrt aBarT
    let b := Float.sqrt (1.0 - aBarP) - a * Float.sqrt (1.0 - aBarT)
    x ← Ddpm.ddimStep x eps a b nTotal
  -- 4×4 RGB grid
  let mut ppm : ByteArray := ByteArray.empty
  ppm := ppm.append s!"P6\n{4*W} {4*H}\n255\n".toUTF8
  let chanStride := H * W
  let imgStride := imgC * chanStride
  for gy in [:4] do
    for h in [:H] do
      for gx in [:4] do
        let idx := gy * 4 + gx
        for w in [:W] do
          let r := F32.read x (idx * imgStride + 0 * chanStride + h * W + w).toUSize
          let g := F32.read x (idx * imgStride + 1 * chanStride + h * W + w).toUSize
          let bch := F32.read x (idx * imgStride + 2 * chanStride + h * W + w).toUSize
          ppm := ppm.push (floatToU8 r) |>.push (floatToU8 g) |>.push (floatToU8 bch)
  IO.FS.writeBinFile outPath ppm
  IO.eprintln s!"    sample grid → {outPath}"

def main (args : List String) : IO Unit := do
  let dataDir := args.head?.getD "data"
  let epochsOverride : Option Nat := (args[1]?).bind String.toNat?
  let maxSteps : Nat := ((args[2]?).bind String.toNat?).getD 0
  -- 4th arg: "tc" = per-block timeCondAdd variant (v2 Workstream A);
  -- "b96" = base96 capacity leg (Phase 2 / B4); default = plain base80.
  let variant := (args[3]?).getD ""
  let spec := if variant == "tc" then tinyCifarDdpmTC
              else if variant == "b96" then tinyCifarDdpm96
              else tinyCifarDdpm
  let cfg := { cifarDdpmConfig with epochs := epochsOverride.getD cifarDdpmConfig.epochs }
  IO.eprintln s!"{spec.name}: {spec.totalParams} params (epochs={cfg.epochs})"

  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix
  let outShape : List Nat := [cfg.batchSize, 3, spec.imageH, spec.imageW]

  IO.eprintln "Generating train step MLIR..."
  let trainMlir := MlirCodegen.generateTrainStep spec cfg.batchSize
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := cfg.weightDecay) (useAdam := cfg.useAdam)
    (useDdpm := true) (ddpmOutShape := outShape)
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir

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

  -- Eval forward for periodic sampling (batch 16, matches sampleGrid).
  let evalMlirPath := s!"{pfx}_fwd_eval.mlir"
  let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
  IO.FS.writeFile evalMlirPath (MlirCodegen.generateEval spec 16)
  unless (← compileMlir evalMlirPath evalVmfb) do IO.Process.exit 1
  IO.eprintln "  eval forward compiled"

  IO.eprintln "Loading CIFAR-10..."
  let (trainImgRaw, nTrain) ← loadCifar dataDir
  let trainImg ← F32.scaleShift trainImgRaw 2.0 (-1.0)
  IO.eprintln s!"  train: {nTrain} images (centered to [-1, 1])"

  let params ← spec.heInitParams
  let nP := spec.totalParams
  let adamM ← F32.const nP.toUSize 0.0
  let adamV ← F32.const nP.toUSize 0.0
  let alphaBar ← Ddpm.cosineSchedule 1000

  let sess ← IreeSession.create vmfbPath
  let evalSess ← IreeSession.create evalVmfb

  let nPix : Nat := 3 * spec.imageH * spec.imageW
  let Tmax : Nat := 1000
  let bpEfull := nTrain / cfg.batchSize
  let bpE := if maxSteps > 0 then min maxSteps bpEfull else bpEfull
  let allShapes := spec.shapesBA
  let bnShapes := spec.bnShapesBA
  let xSh := spec.xShape cfg.batchSize
  let evalShapes := spec.evalShapesBA
  let nT := 3 * nP
  let batch : USize := cfg.batchSize.toUSize
  let imgC : USize := 3
  let outH : USize := spec.imageH.toUSize
  let outW : USize := spec.imageW.toUSize
  let runDir := if variant == "tc" then "runs/ddpm_v2_tc"
                else if variant == "b96" then "runs/ddpm_v2_base96"
                else "runs/ddpm_v2_base80"
  IO.FS.createDirAll runDir

  let mut p := params
  let mut m := adamM
  let mut v := adamV
  let mut emaP := params            -- EMA shadow, init = params
  let mut globalStep : Nat := 0
  let mut runningBnStats ← F32.const spec.nBnStats.toUSize 0.0
  let sampleEvery : Nat := 5

  IO.eprintln s!"training: {bpE} batches/epoch, batch={cfg.batchSize}, lr={cfg.learningRate} (cosine={cfg.cosineDecay}), EMA 0.9999 + hflip"
  for epoch in [:cfg.epochs] do
    let t0 ← IO.monoMsNow
    let mut epochLoss : Float := 0.0
    for bi in [:bpE] do
      globalStep := globalStep + 1
      -- Per-step cosine LR schedule (same idiom as Train.lean:560).
      let lr := if cfg.cosineDecay then
                  let prog := globalStep.toFloat / (cfg.epochs * bpE).toFloat
                  cfg.learningRate * 0.5 * (1.0 + Float.cos (3.14159265358979 * prog))
                else cfg.learningRate
      let x0raw := F32.sliceImages trainImg (bi * cfg.batchSize) cfg.batchSize nPix
      -- hflip augment (per-image p=0.5).
      let x0 ← F32.hflipNCHW x0raw batch imgC outH outW (globalStep * 2654435761).toUSize
      let stepSeed : USize := (epoch * 1000000 + bi).toUSize
      let (xt, ddpmRest) ← Ddpm.stepInputs x0 alphaBar batch nPix.toUSize stepSeed
      let (eps, tba) := ddpmRest
      let xtCond ← Ddpm.prependTChannel xt tba batch imgC outH outW Tmax.toUSize
      let packed := (p.append m).append v
      let out ← IreeSession.trainStepAdamF32Ddpm sess spec.trainFnName
                  packed allShapes xtCond xSh eps
                  lr globalStep.toFloat
                  bnShapes batch imgC outH outW
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      p := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      -- EMA weight update.
      emaP ← F32.ema emaP p 0.0001
      let batchBnStats := out.extract ((nT + 1) * 4) ((nT + 1 + spec.nBnStats) * 4)
      let bnMom : Float := if globalStep == 1 then 1.0 else 0.1
      runningBnStats ← F32.ema runningBnStats batchBnStats bnMom
      if bi < 3 || bi % 200 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss}"
    let t1 ← IO.monoMsNow
    IO.eprintln s!"Epoch {epoch+1}/{cfg.epochs}: avg loss={epochLoss / bpE.toFloat} ({t1-t0}ms)"
    -- Periodic fixed-seed sample grid from RAW weights. NOT the EMA:
    -- the BN running stats are accumulated under the raw weights, and
    -- EMA-weights + raw-weight BN stats is a normalization mismatch that
    -- the DDIM chain amplifies into confetti (Gate-A verdict,
    -- planning/ddpm_demo_v2.md). EMA weights are still checkpointed; using
    -- them needs a BN-stat recalibration pass first.
    if (epoch + 1) % sampleEvery == 0 || epoch + 1 == cfg.epochs then
      let evalParams := p.append runningBnStats
      sampleGrid spec evalSess evalParams evalShapes alphaBar
        s!"{runDir}/samples_ep{epoch+1}.ppm"
      -- Periodic checkpoint (long runs survive interruption).
      IO.FS.writeBinFile s!"{pfx}_params.bin" p
      IO.FS.writeBinFile s!"{pfx}_params_ema.bin" emaP
      IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" runningBnStats

  IO.FS.writeBinFile s!"{pfx}_params.bin" p
  IO.FS.writeBinFile s!"{pfx}_params_ema.bin" emaP
  IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" runningBnStats
  IO.eprintln "Saved params + EMA params + BN stats."
