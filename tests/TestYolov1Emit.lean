import LeanMlir

/-! T1 from planning/yolo_demo_v2.md Phase 1 — YOLOv1 codegen compile-only
    smoke test. Decision D11: full ResNet-34 + YOLO head, batch=1.

    Generates the train-step MLIR with `useYolov1 := true`, writes it to
    .lake/build/, runs `iree-compile`, and asserts exit code 0. Matches
    the existing `TestDdpmTrainEmit.lean` pattern.

    Channel layout (per planning/yolo_demo_v2.md Phase 1):
      perCell = 2 boxes × 5 + 20 classes = 30
      gridH = gridW = 7
      Total flat output dim = 7 × 7 × 30 = 1470

    Backbone shape walk (224×224 input):
      convBn 7 stride 2:  112×112×64
      maxPool 3 stride 2:  56×56×64
      residualBlock s1:    56×56×64
      residualBlock s2:    28×28×128
      residualBlock s2:    14×14×256
      residualBlock s2:     7×7×512   ← matches grid resolution exactly
      flatten:             [B, 7*7*512 = 25088]
      dense 25088→1470:    [B, 1470]
-/

def r34Yolov1 : NetSpec where
  name := "ResNet-34 + YOLOv1 head (smoke)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,                    -- 2/2 instead of paper's 3/2 (IREE compat; matches MainResnetTrain)
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .flatten,
    .dense 25088 1470 .identity
  ]

private def findIreeCompile : IO String := do
  if ← System.FilePath.pathExists ".venv/bin/iree-compile" then
    return ".venv/bin/iree-compile"
  return "iree-compile"

def main : IO Unit := do
  let spec := r34Yolov1
  -- Sanity check the channel chain.
  match spec.validate with
  | some err => IO.eprintln s!"NetSpec validation failed: {err}"; IO.Process.exit 1
  | none     => pure ()
  IO.println s!"{spec.name}: {spec.totalParams} params"

  -- D11: batch=1.
  let batch : Nat := 1
  let train := MlirCodegen.generateTrainStep spec batch "jit_test_yolov1_train_step"
    (useAdam := true) (useYolov1 := true)
    (yoloGridH := 7) (yoloGridW := 7) (yoloNumBoxes := 2) (yoloNumClasses := 20)
  IO.FS.createDirAll ".lake/build"
  let mlirPath := ".lake/build/test_yolov1_train.mlir"
  let vmfbPath := ".lake/build/test_yolov1_train.vmfb"
  IO.FS.writeFile mlirPath train
  IO.println s!"wrote {mlirPath} ({train.length} chars)"

  let args ← ireeCompileArgs mlirPath vmfbPath
  let compiler ← findIreeCompile
  IO.println s!"running {compiler} {String.intercalate " " args.toList}"
  let r ← IO.Process.output { cmd := compiler, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"T1 FAIL: iree-compile exit {r.exitCode}"
    IO.eprintln (r.stderr.take 6000)
    IO.Process.exit 1
  IO.println s!"T1 PASS: {vmfbPath} produced"
