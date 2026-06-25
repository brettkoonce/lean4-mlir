import LeanMlir

/-! YOLOv1 inference dump on the Pets detection val set. Loads the
    bootstrap-trained checkpoint, runs N forward passes, writes the raw
    `[N, 1470]` logits + the val image bytes + per-image IDs to a directory
    so `scripts/yolo_render.py` can draw the predicted boxes.

    See `planning/yolo_final.md`. Usage:
      lake build yolov1-pets-infer
      .lake/build/bin/yolov1-pets-infer [n] [data_dir] [out_dir]

    n defaults to 16, data_dir to data/pets_mosaic_bal, out_dir to figures/yolo_pets.

    Outputs (in out_dir):
      logits.bin       : [N, 1470] float32 — raw YOLOv1 outputs
      images.bin       : [N, 3, 224, 224] float32 — preprocessed inputs (de-normalize in Python to RGB)
      indices.txt      : N lines of per-image IDs (cosmetic; index labels if no ids.txt)

    Spec is identical to MainYolov1PetsTrainBootstrap so the buildPrefix
    matches and the eval vmfb + params + bn_stats files line up. -/

def r34Yolov1 : NetSpec where
  -- Must match the trainer spec (name → checkpoint prefix; layers → eval graph).
  name := "ResNet-34 + YOLOv1 deep-head (Pets)"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .conv2d 512 256 3 .same .relu,
    .conv2d 256 30 1 .same .identity,
    .flatten
  ]

def main (args : List String) : IO Unit := do
  let n        : Nat    := (args[0]?.bind String.toNat?).getD 16
  let dataDir  : String := args[1]?.getD "data/pets_mosaic_bal"
  let outDir   : String := args[2]?.getD "figures/yolo_pets"
  IO.FS.createDirAll outDir
  let spec := r34Yolov1
  IO.println s!"YOLOv1 Pets inference dump"
  IO.println s!"  spec       : {spec.name}"
  IO.println s!"  build pfx  : {spec.buildPrefix}"
  IO.println s!"  data dir   : {dataDir}"
  IO.println s!"  output dir : {outDir}"
  IO.println s!"  n images   : {n}"

  let evalVmfb := s!"{spec.buildPrefix}_fwd_eval.vmfb"
  let paramsPath := s!"{spec.buildPrefix}_params.bin"
  let bnPath := s!"{spec.buildPrefix}_bn_stats.bin"

  if !(← System.FilePath.pathExists evalVmfb) then
    IO.eprintln s!"ERROR: no eval vmfb at {evalVmfb}; train first via yolov1-pets-train-bootstrap"
    IO.Process.exit 1
  if !(← System.FilePath.pathExists paramsPath) then
    IO.eprintln s!"ERROR: no params at {paramsPath}; train first"
    IO.Process.exit 1

  IO.println s!"loading checkpoint..."
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ←
    if ← System.FilePath.pathExists bnPath
    then IO.FS.readBinFile bnPath
    else do
      IO.eprintln s!"  WARN: no BN stats at {bnPath}; using zeros (predictions will be poor)"
      F32.const spec.nBnStats.toUSize 0.0
  let evalParams := params.append bnStats

  let sess ← IreeSession.create evalVmfb
  IO.println s!"  session loaded"

  -- Load val images.
  let (valImg, valLbl, nVal) ← F32.loadDetBin (dataDir ++ "/val.bin")
  let n := min n nVal
  IO.println s!"  loaded {nVal} val records; using first {n}"
  let _ := valLbl

  -- The eval vmfb is compiled at the training batch size (16), so we must
  -- feed exactly that many images in one forward pass (batch=1 → shape error).
  let batch : Nat := 16
  let n := batch
  let xShape := spec.xShape batch
  let pixelsPerImage := 3 * 224 * 224
  let evalShapesBA := spec.evalShapesBA
  let nClasses : USize := 1470  -- flat YOLOv1 output, treated as [B, 1470] "logits"

  let logitsPath := s!"{outDir}/logits.bin"
  let imagesPath := s!"{outDir}/images.bin"
  let indicesPath := s!"{outDir}/indices.txt"

  -- Optional per-image IDs (cosmetic label on each render). Read `ids.txt`
  -- from the data dir if present; otherwise the loop below falls back to
  -- index labels — fine for mosaics, whose IDs are meaningless anyway.
  let idsPath := s!"{dataDir}/ids.txt"
  let testIds ← if ← System.FilePath.pathExists idsPath then do
      let s ← IO.FS.readFile idsPath
      pure ((s.trim.splitOn "\n").map String.trim |>.filter (· != ""))
    else pure []

  -- Single batched forward over the first `n` (=batch) val images.
  let imagesOut := F32.sliceImages valImg 0 n pixelsPerImage
  let logitsOut ← IreeSession.forwardF32 sess spec.evalFnName
                    evalParams evalShapesBA imagesOut xShape batch.toUSize nClasses
  IO.println s!"  inferred batch of {n}"
  let mut idsOut : String := ""
  for i in [:n] do
    if h : i < testIds.length then
      idsOut := idsOut ++ testIds[i] ++ "\n"
    else
      idsOut := idsOut ++ s!"unknown_{i}\n"

  IO.FS.writeBinFile logitsPath logitsOut
  IO.FS.writeBinFile imagesPath imagesOut
  IO.FS.writeFile    indicesPath idsOut

  IO.println s!"wrote:"
  IO.println s!"  {logitsPath}  ({logitsOut.size} bytes — {n}×1470 float32)"
  IO.println s!"  {imagesPath}  ({imagesOut.size} bytes — {n}×3×224×224 float32)"
  IO.println s!"  {indicesPath} ({idsOut.length} chars — {n} image IDs)"
  IO.println s!"next: python3 scripts/yolo_render.py {outDir}"
