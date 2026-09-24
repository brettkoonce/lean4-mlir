import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.MlirCodegen
import LeanMlir.ReferenceNets

/-! Test: generate ResNet-34 forward + train step MLIR with real residualBlock layers
    (skip connections, projections) and compile with IREE. -/

def compile (src out : String) : IO Bool := do
  let args ← ireeCompileArgs src out
  let r ← IO.Process.output { cmd := (← findIreeCompile), args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"Compile FAILED:\n{r.stderr.take 3000}"
    return false
  return true

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  IO.println s!"ResNet-34: {ReferenceNets.resnet34.totalParams} params"

  -- Forward
  let fwd := MlirCodegen.generate ReferenceNets.resnet34 16
  IO.FS.writeFile ".lake/build/resnet34_fwd.mlir" fwd
  IO.println s!"Forward: {fwd.length} chars"
  if ← compile ".lake/build/resnet34_fwd.mlir" ".lake/build/resnet34_fwd.vmfb"
  then IO.println "✓ Forward compiled"

  -- Train step
  let ts := MlirCodegen.generateTrainStep ReferenceNets.resnet34 16 "jit_resnet34_train_step"
  IO.FS.writeFile ".lake/build/resnet34_train_step.mlir" ts
  IO.println s!"Train step: {ts.length} chars"
  if ← compile ".lake/build/resnet34_train_step.mlir" ".lake/build/resnet34_train_step.vmfb"
  then IO.println "✓ Train step compiled"
