import LeanMlir

/-! # `yolov1-neudet448` — the single-grid arm on NEU-DET steel defects

The archived VisDrone rung `demos/archive/MainYolov1VisDrone448.lean` — ResNet-34
backbone, deep conv head, one 14×14 grid at 448 input, the YOLOv1 masked loss —
with two edits: the name (so its checkpoints never collide with the VisDrone
arm's) and an epoch override. It scored mAP 0.0000 on VisDrone: seventy 20-px
objects per frame cannot share 196 cells. NEU-DET has 2.3 defects per crop, most
of them larger than a cell, so the same head is expected to be a wash here, and
that pair of numbers is what the section is about (planning/neu_det_fpn_demo.md).

```
lake build yolov1-neudet448
# train (default): data dir with 448/14 train.bin + val.bin from preprocess_neu_det.py
CUDA_VISIBLE_DEVICES=1 YOLO_EPOCHS=30 .lake/build/bin/yolov1-neudet448 data/neu_det448
# infer: dump [N,5880] logits.bin for scripts/yolo_map_visdrone.py --grid 14 --classes neu
.lake/build/bin/yolov1-neudet448 infer data/neu_det448 runs/neudet_grid
```

`YOLO_EPOCHS` overrides the archived 12 so the arm can run the FPN arm's
schedule; `YOLO_EVAL_SPLIT=test` points `infer` at `test.bin` instead of `val.bin`;
`YOLO_TAG` suffixes the name, so a probe never lands on the measured arm's
checkpoints — and, as with `FPN_TAG`, it must be set on `infer` too;
`YOLO_EVAL_EPOCH=N` evaluates the `_params_eN.bin` checkpoint instead of the final.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`).
-/

def r34Yolov1_448NeuDet : NetSpec where
  -- Identical architecture to the VisDrone `r34Yolov1_448`, at 448² input:
  -- backbone strides 2·2·1·2·2·2 = 32 ⇒ 14×14 grid ⇒ head [B,30,14,14] ⇒ flatten
  -- [B,5880]. Distinct name ⇒ distinct buildPrefix ⇒ own graphs/checkpoints.
  name := "ResNet-34 + YOLOv1 448 (NEU-DET)"
  imageH := 448
  imageW := 448
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .conv2d 512 256 3 .same .relu,      -- deep head L1: 3×3, spatial context
    .conv2d 256 30 1 .same .identity,   -- deep head L2: 1×1 → [B,30,14,14]
    .flatten                            -- → [B,5880] for the YOLOv1 masked loss
  ]

def r34Yolov1_448NeuDetConfig : TrainConfig where
  -- The archived VisDrone-448 recipe, unchanged: same LR / clip / focal, so the
  -- ONLY change against the 0.0000 arm is the dataset.
  learningRate := 7.0e-4
  batchSize    := 16
  epochs       := 12
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := true
  warmupEpochs := 3
  gradClipNorm := 4.0
  headLrMult   := 1.0
  checkpointEveryNEpochs := 2
  augment      := true
  lossKind     := LossKind.yolov1Masked
  useFocal     := true
  focalGamma   := 2.0
  bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)

/-- Epoch-count override (`YOLO_EPOCHS`), so this arm can run the FPN arm's
    schedule; defaults to the archived 12. -/
def yoloEpochsFromEnv (dflt : Nat) : IO Nat := do
  match (← IO.getEnv "YOLO_EPOCHS") with
  | none => return dflt
  | some v => return (v.trimAscii.toNat?).getD dflt

/-- Which split `infer` dumps (`YOLO_EVAL_SPLIT`, `val` by default; `test` for the
    table's row once the epoch is chosen on val). -/
def yoloEvalSplitFromEnv : IO String := do
  match (← IO.getEnv "YOLO_EVAL_SPLIT") with
  | none => return "val"
  | some v => return if v.trimAscii.toString.isEmpty then "val" else v.trimAscii.toString

/-- Name suffix (`YOLO_TAG`): the name IS the checkpoint prefix. -/
def yoloTagFromEnv : IO String := do
  match (← IO.getEnv "YOLO_TAG") with
  | none => return ""
  | some v => return if v.trimAscii.toString.isEmpty then "" else s!" {v.trimAscii.toString}"

def specFromEnv : IO NetSpec := do
  let tag ← yoloTagFromEnv
  return { r34Yolov1_448NeuDet with name := r34Yolov1_448NeuDet.name ++ tag }

/-- Infer mode: dump `[N, 5880]` logits of the chosen split to `outDir/logits.bin`. -/
def inferDump (dataDir outDir : String) : IO Unit := do
  IO.FS.createDirAll outDir
  let spec ← specFromEnv
  let split ← yoloEvalSplitFromEnv
  let esfx ← match (← IO.getEnv "YOLO_EVAL_EPOCH") with
    | none => pure ""
    | some v => pure (if v.trimAscii.toString.isEmpty then "" else s!"_e{v.trimAscii.toString}")
  let gH := spec.imageH / 32
  let gW := spec.imageW / 32
  let flat : Nat := 30 * gH * gW            -- 5880
  let evalVmfb ← NetSpec.graphArtifact spec.buildPrefix "fwd_eval"
  let paramsPath := s!"{spec.buildPrefix}_params{esfx}.bin"
  let bnPath := s!"{spec.buildPrefix}_bn_stats{esfx}.bin"
  IO.println s!"  spec   : {spec.name}"
  IO.println s!"  prefix : {spec.buildPrefix}"
  IO.println s!"  split  : {split}"
  if !(← System.FilePath.pathExists evalVmfb) then
    IO.eprintln s!"ERROR: no eval graph at {evalVmfb}; train first"; IO.Process.exit 1
  if !(← System.FilePath.pathExists paramsPath) then
    IO.eprintln s!"ERROR: no params at {paramsPath}; train first"; IO.Process.exit 1
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ←
    if ← System.FilePath.pathExists bnPath then IO.FS.readBinFile bnPath
    else do
      IO.eprintln s!"  WARN: no BN stats at {bnPath}; using zeros"
      F32.const spec.nBnStats.toUSize 0.0
  let evalParams := params.append bnStats
  let sess ← LowererSession.create evalVmfb
  let (valImg, _valLbl, nVal) ← F32.loadDetBinDims (dataDir ++ s!"/{split}.bin")
                                  spec.imageH.toUSize gH.toUSize gW.toUSize
  IO.println s!"  loaded {nVal} {split} records ({flat}-wide output); dumping logits"
  let batch : Nat := 16
  let xShape := spec.xShape batch
  let pixelsPerImage := 3 * spec.imageH * spec.imageW
  let evalShapesBA := spec.evalShapesBA
  let nOut : USize := flat.toUSize
  let rowBytes : Nat := flat * 4
  let nBatches := (nVal + batch - 1) / batch
  let mut logitsAll : ByteArray := ByteArray.empty
  for b in [:nBatches] do
    let start := b * batch
    let real  := min batch (nVal - start)
    let mut imgs := F32.sliceImages valImg start real pixelsPerImage
    if real < batch then
      let lastImg := F32.sliceImages valImg (start + real - 1) 1 pixelsPerImage
      for _ in [:batch - real] do
        imgs := imgs ++ lastImg
    let logitsB ← LowererSession.forwardF32 sess spec.evalFnName
                    evalParams evalShapesBA imgs xShape batch.toUSize nOut
    logitsAll := logitsAll ++ logitsB.extract 0 (real * rowBytes)
  IO.FS.writeBinFile s!"{outDir}/logits.bin" logitsAll
  IO.println s!"  wrote {outDir}/logits.bin ({logitsAll.size} bytes — {nVal}×{flat} f32)"
  IO.println s!"next: python3 scripts/yolo_map_visdrone.py {outDir}/logits.bin {dataDir}/{split}.bin --grid {gH} --classes neu"

def main (args : List String) : IO Unit := do
  match args with
  | "infer" :: rest =>
    let dataDir := rest[0]?.getD "data/neu_det448"
    let outDir  := rest[1]?.getD "runs/neudet_grid"
    IO.println s!"YOLOv1 NEU-DET-448 inference dump — data {dataDir} → {outDir}"
    inferDump dataDir outDir
  | _ =>
    let dataDir := args.head?.getD "data/neu_det448"
    let epochs ← yoloEpochsFromEnv r34Yolov1_448NeuDetConfig.epochs
    let cfg := { r34Yolov1_448NeuDetConfig with epochs := epochs }
    let spec ← specFromEnv
    IO.println s!"YOLOv1 NEU-DET-448 (single 14×14 grid) — data {dataDir} — epochs {epochs} — box loss: sqrt-MSE"
    IO.println s!"  spec   : {spec.name}"
    spec.train cfg dataDir DatasetKind.petsDet
