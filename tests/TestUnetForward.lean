import LeanMlir
import LeanMlir.ReferenceNets

/-! Forward-only smoke test for the new `unetDown` / `unetUp` codegen.

Generates the forward MLIR for `unetPets` (depth 4, base 32, 224×224
RGB → 3-class trimap), writes it to disk, runs iree-compile to produce
a .vmfb, and exits 0 if iree-compile succeeded.

Only validates the **forward** path. Train-step backward (with
skip-grad slots) is the next session.

Usage: `IREE_BACKEND=rocm IREE_CHIP=gfx1100 lake exe test-unet-forward` -/

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
  let fwd := MlirCodegen.generate ReferenceNets.unetPets 2
  compileOne "forward" fwd
    ".lake/build/test_unet_forward.mlir" ".lake/build/test_unet_forward.vmfb"
  let train := MlirCodegen.generateTrainStep ReferenceNets.unetPets 2 "jit_test_unet_train_step"
    (useAdam := true) (useSeg := true)
  compileOne "train-step" train
    ".lake/build/test_unet_train.mlir" ".lake/build/test_unet_train.vmfb"
