import LeanMlir

/-! Compile-only smoke test: tiny DDPM-shaped UNet train-step MLIR
should generate cleanly through the new `useDdpm` codegen branch. -/

def tinyDdpmUnet : NetSpec where
  name := "tiny DDPM UNet (MNIST 28x28x1)"
  imageH := 28
  imageW := 28
  layers := [
    .unetDown 1 16,
    .unetDown 16 32,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .unetUp 64 32,
    .unetUp 32 16,
    .conv2d 16 1 1 .same .identity
  ]

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  let train := MlirCodegen.generateTrainStep tinyDdpmUnet 8 "jit_test_ddpm_train_step"
    (useAdam := true) (useDdpm := true) (ddpmOutShape := [8, 1, 28, 28])
  let path := ".lake/build/test_ddpm_train.mlir"
  IO.FS.writeFile path train
  IO.eprintln s!"  wrote {path} ({train.length} chars)"
  let args ← ireeCompileArgs path ".lake/build/test_ddpm_train.vmfb"
  let compiler := if (← System.FilePath.pathExists ".venv/bin/iree-compile")
                  then ".venv/bin/iree-compile"
                  else "iree-compile"
  let r ← IO.Process.output { cmd := compiler, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"FAIL: iree-compile exit {r.exitCode}"
    IO.eprintln (r.stderr.take 4000)
    IO.Process.exit 1
  IO.eprintln s!"OK: .lake/build/test_ddpm_train.vmfb produced"
