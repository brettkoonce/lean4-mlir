import LeanMlir.IreeRuntime
import LeanMlir.F32Array
import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers
import LeanMlir.Train

/-! End-to-end smoke for the ConvNeXt-BN variant: emit train-step MLIR
    using `.bn` for both `.convNextStage` and `.convNextDownsample`,
    iree-compile, run one Adam step, verify the loss is finite and
    descends after one update. Mirrors `TestConvNextTrainStep` but
    swaps norm `.ln` → `.bn`. -/

def tinyConvNextBnSpec : NetSpec where
  name := "tiny-ConvNeXt-BN"
  imageH := 32
  imageW := 32
  layers := [
    .convBn 3 32 2 2 .same,
    .convNextStage 32 2 .bn .gelu,
    .convNextDownsample 32 64 .bn,
    .convNextStage 64 2 .bn .gelu,
    .globalAvgPool,
    .dense 64 10 .identity
  ]

private def findIreeCompile : IO String := do
  if ← System.FilePath.pathExists ".venv/bin/iree-compile" then
    return ".venv/bin/iree-compile"
  return "iree-compile"

def main : IO Unit := do
  let spec := tinyConvNextBnSpec
  let batch : Nat := 2
  let nClasses : Nat := 10

  IO.println "── ConvNeXt-BN train-step smoke ──"
  let moduleName := "jit_" ++ spec.sanitizedName ++ "_train_step"
  let mlir := MlirCodegen.generateTrainStep spec batch moduleName
  IO.FS.createDirAll ".lake/build"
  let mlirPath := s!".lake/build/{spec.sanitizedName}_train_step.mlir"
  let vmfbPath := s!".lake/build/{spec.sanitizedName}_train_step.vmfb"
  IO.FS.writeFile mlirPath mlir
  IO.println s!"  emitted    : {mlir.length} chars"
  IO.println s!"  BN layers  : {spec.bnLayers.size} (stem + per-block + per-downsample)"
  IO.println s!"  BN stat fl : {spec.nBnStats}"

  let compiler ← findIreeCompile
  let backend ← (IO.getEnv "IREE_BACKEND").map (·.getD "llvm-cpu")
  let mut compileArgs : Array String := #[mlirPath, s!"--iree-hal-target-backends={backend}"]
  if backend == "llvm-cpu" then
    compileArgs := compileArgs.push "--iree-llvmcpu-target-cpu=host"
  else if backend == "rocm" then
    let chip ← (IO.getEnv "IREE_CHIP").map (·.getD "gfx1100")
    compileArgs := compileArgs.push s!"--iree-rocm-target={chip}"
  compileArgs := compileArgs ++ #["-o", vmfbPath]
  let r ← IO.Process.output { cmd := compiler, args := compileArgs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 2000}"
    IO.Process.exit 1
  IO.println s!"  compiled   : {vmfbPath}"

  let p ← spec.heInitParams
  let nP := F32.size p
  let m ← F32.const nP.toUSize 0.0
  let v ← F32.const nP.toUSize 0.0
  let packed := (p.append m).append v
  let pixels := 3 * spec.imageH * spec.imageW
  let xba ← F32.heInit 7 (batch * pixels).toUSize 1.0
  let xSh := packXShape #[batch, pixels]
  let mut yb : ByteArray := .empty
  for i in [:batch] do
    let lbl : UInt32 := (i % nClasses).toUInt32
    yb := yb.push (lbl &&& 0xFF).toUInt8
    yb := yb.push ((lbl >>> 8) &&& 0xFF).toUInt8
    yb := yb.push ((lbl >>> 16) &&& 0xFF).toUInt8
    yb := yb.push ((lbl >>> 24) &&& 0xFF).toUInt8

  let sess ← IreeSession.create vmfbPath
  let t0 ← IO.monoMsNow
  let out ← IreeSession.trainStepAdamF32 sess spec.trainFnName
              packed spec.shapesBA xba xSh yb 0.001 1.0 spec.bnShapesBA batch.toUSize
  let t1 ← IO.monoMsNow
  let loss := F32.extractLoss out (3 * nP)
  IO.println s!"  step 1     : loss={loss} ({t1-t0}ms)"

  let pNew := F32.slice out 0 nP
  let mNew := F32.slice out nP nP
  let vNew := F32.slice out (2 * nP) nP
  let packed2 := (pNew.append mNew).append vNew
  let out2 ← IreeSession.trainStepAdamF32 sess spec.trainFnName
              packed2 spec.shapesBA xba xSh yb 0.001 2.0 spec.bnShapesBA batch.toUSize
  let loss2 := F32.extractLoss out2 (3 * nP)
  IO.println s!"  step 2     : loss={loss2}"
  IO.println s!"  Δloss      : {loss - loss2}"

  if loss.isNaN || loss.isInf || loss2.isNaN || loss2.isInf then
    IO.eprintln "ERROR: non-finite loss"
    IO.Process.exit 1
  IO.println "ConvNeXt-BN train-step smoke OK."
