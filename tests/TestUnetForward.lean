import LeanMlir
import LeanMlir.ReferenceNets

/-! Forward-only smoke test for the new `unetDown` / `unetUp` codegen.

Generates the forward MLIR and the segmentation train step for `unetBrats`
(depth 4, base 32, 240×240 4-modality MRI → 4-class tumour), writes them to
disk, runs iree-compile on each, and exits 0 if both compile.

Usage: `lake exe test-unet-forward` (`IREE_BACKEND` / `IREE_CHIP` pick the
target; default cuda / sm_86) -/

private def compileOne (label mlir mlirPath vmfbPath : String) : IO Unit := do
  IO.FS.writeFile mlirPath mlir
  IO.eprintln s!"  [{label}] wrote {mlirPath} ({mlir.length} chars)"
  let args ← ireeCompileArgs mlirPath vmfbPath
  let compiler ← findIreeCompile
  let r ← IO.Process.output { cmd := compiler, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"  [{label}] FAIL: iree-compile exit {r.exitCode}"
    IO.eprintln (r.stderr.take 4000)
    IO.Process.exit 1
  IO.eprintln s!"  [{label}] OK: {vmfbPath} produced"

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  let fwd := MlirCodegen.generate ReferenceNets.unetBrats 2
  compileOne "forward" fwd
    ".lake/build/test_unet_forward.mlir" ".lake/build/test_unet_forward.vmfb"
  let train := MlirCodegen.generateTrainStep ReferenceNets.unetBrats 2 "jit_test_unet_train_step"
    (useAdam := true) (useSeg := true)
  compileOne "train-step" train
    ".lake/build/test_unet_train.mlir" ".lake/build/test_unet_train.vmfb"
