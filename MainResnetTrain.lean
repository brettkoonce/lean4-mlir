import LeanJax.IreeRuntime
import LeanJax.F32Array
import LeanJax.Types
import LeanJax.Spec
import LeanJax.MlirCodegen

/-! ResNet-34 on Imagenette — full training pipeline.
    Generates train_step MLIR → compiles with IREE → SGD training loop.
    ~21.3M params, 224×224 input, 10 classes. -/

def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,                    -- 2/2 instead of 3/2 (IREE compat)
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

namespace ResnetLayout

-- convBn params: W (oc*ic*k*k) + gamma (oc) + beta (oc)
-- residualBlock: 2 convBn per block + optional projection convBn
-- See Spec.lean Layer.nParams for the formula.

def nParams : Nat := resnet34.totalParams  -- 21289802

-- Build param shapes array by walking the spec
def paramShapes : Array (Array Nat) := Id.run do
  let mut shapes : Array (Array Nat) := #[]
  for l in resnet34.layers do
    match l with
    | .convBn ic oc k _ _ =>
      shapes := shapes.push #[oc, ic, k, k] |>.push #[oc] |>.push #[oc]
    | .dense fi fo _ =>
      shapes := shapes.push #[fi, fo] |>.push #[fo]
    | .residualBlock ic oc nBlocks firstStride =>
      let needsProj := !(ic == oc && firstStride == 1)
      for bi in [:nBlocks] do
        let blockIc := if bi == 0 then ic else oc
        shapes := shapes.push #[oc, blockIc, 3, 3] |>.push #[oc] |>.push #[oc]
        shapes := shapes.push #[oc, oc, 3, 3] |>.push #[oc] |>.push #[oc]
        if bi == 0 && needsProj then
          shapes := shapes.push #[oc, ic, 1, 1] |>.push #[oc] |>.push #[oc]
    | _ => pure ()
  return shapes

-- Full shapes: params ++ m (1st moment) ++ v (2nd moment) for Adam
def allShapes : Array (Array Nat) := paramShapes ++ paramShapes ++ paramShapes
def shapesBA : ByteArray := packShapes allShapes
def nTotal : Nat := 3 * nParams  -- params + m + v

def xShape (batch : Nat) : ByteArray :=
  packXShape #[batch, 3 * 224 * 224]

end ResnetLayout

def main (args : List String) : IO Unit := do
  let dataDir := args.head? |>.getD "data/imagenette"
  IO.eprintln s!"ResNet-34: {ResnetLayout.nParams} params"

  let batchN : Nat := 32
  let batch : USize := 32

  -- Generate + compile train step MLIR (self-compile like R50)
  IO.FS.createDirAll ".lake/build"
  IO.eprintln "Generating train step MLIR..."
  let mlir := MlirCodegen.generateTrainStep resnet34 batchN "jit_resnet34_train_step"
  IO.FS.writeFile ".lake/build/resnet34_train_step.mlir" mlir
  IO.eprintln s!"  {mlir.length} chars"

  let fwdMlir := MlirCodegen.generate resnet34 batchN
  IO.FS.writeFile ".lake/build/resnet34_fwd.mlir" fwdMlir

  IO.eprintln "Compiling vmfbs..."
  let fwdCompileArgs ← ireeCompileArgs ".lake/build/resnet34_fwd.mlir" ".lake/build/resnet34_fwd.vmfb"
  let rf ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := fwdCompileArgs }
  if rf.exitCode != 0 then
    IO.eprintln s!"forward compile failed: {rf.stderr.take 1000}"
  else
    IO.eprintln "  forward compiled"

  let compileArgs ← ireeCompileArgs ".lake/build/resnet34_train_step.mlir" ".lake/build/resnet34_train_step.vmfb"
  let r ← IO.Process.output { cmd := ".venv/bin/iree-compile", args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"compile failed: {r.stderr.take 3000}"
    IO.Process.exit 1
  IO.eprintln "  compiled"

  let sess ← IreeSession.create ".lake/build/resnet34_train_step.vmfb"
  IO.eprintln "  session loaded"

  -- Load Imagenette data (train at 256×256 for random crop, val at 224×224)
  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (dataDir ++ "/train.bin") 256
  IO.eprintln s!"  train: {nTrain} images (256×256)"

  -- Initialize params: He init for weights, 1.0 for gamma, 0.0 for beta/bias
  let mut paramParts : Array ByteArray := #[]
  let mut seed : USize := 42
  let shapes := ResnetLayout.paramShapes
  let mut si : Nat := 0
  while si < shapes.size do
    let shape := shapes[si]!
    let n := shape.foldl (· * ·) 1
    if shape.size >= 2 then
      let fanIn := if shape.size == 4 then shape[1]! * shape[2]! * shape[3]! else shape[0]!
      paramParts := paramParts.push (← F32.heInit seed n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
      seed := seed + 1
      si := si + 1
      if si < shapes.size && shapes[si]!.size == 1 then
        let n1 := shapes[si]![0]!
        if si + 1 < shapes.size && shapes[si + 1]!.size == 1 then
          paramParts := paramParts.push (← F32.const n1.toUSize 1.0)
          si := si + 1
          paramParts := paramParts.push (← F32.const (shapes[si]![0]!).toUSize 0.0)
          si := si + 1
        else
          paramParts := paramParts.push (← F32.const n1.toUSize 0.0)
          si := si + 1
    else
      si := si + 1
  let params := F32.concat paramParts
  -- Adam state: m (1st moment) and v (2nd moment), both zero-initialized
  let adamM ← F32.const (F32.size params).toUSize 0.0
  let adamV ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params + m + v ({(params.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  -- Training loop: Adam optimizer, cosine LR schedule, batch 32
  let epochs := 80
  let bpE := nTrain / batchN
  let trainPixels := 3 * 256 * 256
  let allShapes := ResnetLayout.shapesBA
  let xSh := ResnetLayout.xShape batchN
  let nP := ResnetLayout.nParams
  let nT := ResnetLayout.nTotal  -- params + m + v
  let baseLR : Float := 0.001

  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, Adam, lr={baseLR}, cosine, label_smooth=0.1, wd=1e-4"
  let mut p := params
  let mut m := adamM
  let mut v := adamV
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  let mut globalStep : Nat := 0
  for epoch in [:epochs] do
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPixels.toUSize (epoch + 42).toUSize
    curImg := sImg; curLbl := sLbl
    -- Cosine LR with 3-epoch warmup
    let lr : Float := if epoch < 3 then
      baseLR * (epoch.toFloat + 1.0) / 3.0
    else
      baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (epoch.toFloat - 3.0) / (epochs.toFloat - 3.0)))
    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    for bi in [:bpE] do
      globalStep := globalStep + 1
      let xba256 := F32.sliceImages curImg (bi * batchN) batchN trainPixels
      let xbaCropped ← F32.randomCrop xba256 batch 3 256 256 224 224 (epoch * 10000 + bi).toUSize
      let xba ← F32.randomHFlip xbaCropped batch 3 224 224 (epoch * 10000 + bi + 7777).toUSize
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN
      -- Pack params ++ m ++ v
      let packed := (p.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← IreeSession.trainStepAdamF32 sess "jit_resnet34_train_step.main"
                  packed allShapes xba xSh yb lr globalStep.toFloat batch
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      -- Unpack: params, m, v
      p := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"

    -- Val eval every 10 epochs
    if (epoch + 1) % 10 == 0 || epoch + 1 == epochs then
      let fwdVmfb := ".lake/build/resnet34_fwd.vmfb"
      if ← System.FilePath.pathExists fwdVmfb then
        let evalSess ← IreeSession.create fwdVmfb
        let (valImg, valLbl, nVal) ← F32.loadImagenette (dataDir ++ "/val.bin")
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let paramShapesBA := packShapes ResnetLayout.paramShapes
        let evalXSh := ResnetLayout.xShape evalBatch
        let mut correct : Nat := 0
        let mut total : Nat := 0
        for bi in [:evalSteps] do
          let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch (3 * 224 * 224)
          let logits ← IreeSession.forwardF32 evalSess "resnet_34.forward"
                          p paramShapesBA xba evalXSh evalBatch.toUSize 10
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch
          for i in [:evalBatch] do
            let pred := F32.argmax10 logits (i * 10).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy: {correct}/{total} = {acc}%"
  IO.FS.writeBinFile ".lake/build/resnet34_params.bin" p
  IO.eprintln "Saved params."
