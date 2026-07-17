import LeanMlir

/-! Anchor-based YOLO detector on VisDrone at 448 / 14×14 grid, A=6 anchors
    (brick #2, planning/yolo_drone.md WS-C). Each cell predicts 6 anchor slots
    of [tx,ty,tw,th, obj, cls(10)] (head 256→90 = A·15); box_a = anchor_a·exp(pred),
    FD-verified DIoU box loss + focal objectness + softmax class (emitAnchorYoloLoss).
    Anchors are the k-means priors from scripts/visdrone_anchors.py (recall@0.5
    ceiling 76%). Data: data/visdrone448_a6 (preprocess_visdrone --anchors).

    Usage:
      lake build yolov1-visdrone-anchor
      IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 \
        .lake/build/bin/yolov1-visdrone-anchor data/visdrone448_a6
      .lake/build/bin/yolov1-visdrone-anchor infer data/visdrone448_a6 figures/yolo_anchor
-/

-- k-means priors, A=6 (data/visdrone/anchors_a6.txt).
def visdroneAnchors6 : List (Float × Float) :=
  [(0.007192, 0.015628), (0.017894, 0.027307), (0.023569, 0.061085),
   (0.044307, 0.038472), (0.062762, 0.087849), (0.124852, 0.151875)]

def r34Yolov1Anchor : NetSpec where
  name := "ResNet-34 + YOLO anchor A6 448 (VisDrone)"
  imageH := 448
  imageW := 448
  detStride := 32                       -- grid 448/32 = 14
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .conv2d 512 256 3 .same .relu,      -- deep head L1
    .conv2d 256 90 1 .same .identity,   -- A·15 = 90 → [B,90,14,14]
    .flatten                            -- → [B,17640]
  ]

def r34Yolov1AnchorConfig : TrainConfig where
  learningRate := 7.0e-4
  batchSize    := 16
  epochs       := 12
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := true
  warmupEpochs := 3
  gradClipNorm := 4.0
  checkpointEveryNEpochs := 2
  augment      := false                 -- yoloAugment is single-box-format only
  lossKind     := LossKind.yolov1Masked
  useFocal     := true
  focalGamma   := 2.0
  anchors      := visdroneAnchors6      -- routes the loss to emitAnchorYoloLoss
  bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)

/-- Infer: dump [N, 17640] val logits for scripts/yolo_map_visdrone.py --anchors. -/
def inferDump (dataDir outDir : String) : IO Unit := do
  IO.FS.createDirAll outDir
  let spec := r34Yolov1Anchor
  let gH := spec.imageH / spec.detStride
  let gW := spec.imageW / spec.detStride
  let A := visdroneAnchors6.length
  let flat : Nat := A * 15 * gH * gW           -- 17640
  let evalVmfb := s!"{spec.buildPrefix}_fwd_eval.vmfb"
  let paramsPath := s!"{spec.buildPrefix}_params.bin"
  let bnPath := s!"{spec.buildPrefix}_bn_stats.bin"
  if !(← System.FilePath.pathExists evalVmfb) then
    IO.eprintln s!"ERROR: no eval vmfb at {evalVmfb}; train first"; IO.Process.exit 1
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ←
    if ← System.FilePath.pathExists bnPath then IO.FS.readBinFile bnPath
    else F32.const spec.nBnStats.toUSize 0.0
  let evalParams := params.append bnStats
  let sess ← IreeSession.create evalVmfb
  let (valImg, _t, nVal) ← F32.loadDetBinAnchor (dataDir ++ "/val.bin")
                             spec.imageH.toUSize gH.toUSize gW.toUSize A.toUSize
  IO.println s!"  loaded {nVal} val records ({flat}-wide output); dumping logits"
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
      for _ in [:batch - real] do imgs := imgs ++ lastImg
    let logitsB ← IreeSession.forwardF32 sess spec.evalFnName
                    evalParams evalShapesBA imgs xShape batch.toUSize nOut
    logitsAll := logitsAll ++ logitsB.extract 0 (real * rowBytes)
  IO.FS.writeBinFile s!"{outDir}/logits.bin" logitsAll
  IO.println s!"  wrote {outDir}/logits.bin ({logitsAll.size} bytes — {nVal}×{flat} f32)"

def main (args : List String) : IO Unit := do
  match args with
  | "infer" :: rest =>
    let dataDir := rest[0]?.getD "data/visdrone448_a6"
    let outDir  := rest[1]?.getD "figures/yolo_anchor"
    IO.println s!"YOLO anchor VisDrone inference dump — {dataDir} → {outDir}"
    inferDump dataDir outDir
  | _ =>
    let dataDir := args.head?.getD "data/visdrone448_a6"
    IO.println s!"YOLO anchor A6 VisDrone (14×14, 6 anchors) — data dir: {dataDir}"
    r34Yolov1Anchor.train r34Yolov1AnchorConfig dataDir DatasetKind.petsDet
