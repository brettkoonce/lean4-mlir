import LeanMlir

/-! YOLOv1 detector on VisDrone at 448 input / **28×28 grid** (stride-16 tap).

    The "different head" hedge (planning/yolo_drone.md): the WS-A collapse and the
    448/14 rung both point at the grid being too coarse for VisDrone's density
    (~70 objects/image). This drops the ResNet-34 backbone's last downsample
    (last residual block stride 2 → 1) so the detection feature map is 448/16 = 28
    → 784 cells (vs 196 at stride-32), ~10× headroom over the object count. Same
    deep conv head + √-MSE YOLOv1 loss otherwise; the ONLY change vs the 448/14 run
    is the grid. Tests whether more grid alone recovers detection, or whether the
    box loss / anchors are the real blocker.

    Backbone conv weights are stride-independent, so the R34-ImageNet bootstrap
    prefix (21,284,672) still loads. Runs on GPU 1 (own build prefix) in parallel
    with the 448/14 run on GPU 0.

    Usage:
      lake build yolov1-visdrone448s16
      IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 \
        .lake/build/bin/yolov1-visdrone448s16 data/visdrone448_g28
      .lake/build/bin/yolov1-visdrone448s16 infer data/visdrone448_g28 figures/yolo_visdrone448s16
-/

def r34Yolov1_448s16 : NetSpec where
  name := "ResNet-34 + YOLOv1 448 s16 (VisDrone)"
  imageH := 448
  imageW := 448
  detStride := 16                       -- last block stride 1 ⇒ grid 448/16 = 28
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 1,         -- stride 1 (was 2) → keep 28×28 resolution
    .conv2d 512 256 3 .same .relu,      -- deep head L1
    .conv2d 256 30 1 .same .identity,   -- deep head L2 → [B,30,28,28]
    .flatten                            -- → [B,23520]
  ]

def r34Yolov1_448s16Config : TrainConfig where
  learningRate := 7.0e-4
  batchSize    := 8             -- 28×28×512 activations are 4× heavier; halve batch
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

def inferDump (dataDir outDir : String) : IO Unit := do
  IO.FS.createDirAll outDir
  let spec := r34Yolov1_448s16
  let gH := spec.imageH / spec.detStride
  let gW := spec.imageW / spec.detStride
  let flat : Nat := 30 * gH * gW            -- 23520
  let evalVmfb := s!"{spec.buildPrefix}_fwd_eval.vmfb"
  let paramsPath := s!"{spec.buildPrefix}_params.bin"
  let bnPath := s!"{spec.buildPrefix}_bn_stats.bin"
  if !(← System.FilePath.pathExists evalVmfb) then
    IO.eprintln s!"ERROR: no eval vmfb at {evalVmfb}; train first"; IO.Process.exit 1
  if !(← System.FilePath.pathExists paramsPath) then
    IO.eprintln s!"ERROR: no params at {paramsPath}; train first"; IO.Process.exit 1
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ←
    if ← System.FilePath.pathExists bnPath then IO.FS.readBinFile bnPath
    else do
      IO.eprintln s!"  WARN: no BN stats at {bnPath}; using zeros"
      F32.const spec.nBnStats.toUSize 0.0
  let evalParams := params.append bnStats
  let sess ← IreeSession.create evalVmfb
  let (valImg, _valLbl, nVal) ← F32.loadDetBinDims (dataDir ++ "/val.bin")
                                  spec.imageH.toUSize gH.toUSize gW.toUSize
  IO.println s!"  loaded {nVal} val records ({flat}-wide output); dumping logits"
  let batch : Nat := 8
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
    let logitsB ← IreeSession.forwardF32 sess spec.evalFnName
                    evalParams evalShapesBA imgs xShape batch.toUSize nOut
    logitsAll := logitsAll ++ logitsB.extract 0 (real * rowBytes)
  IO.FS.writeBinFile s!"{outDir}/logits.bin" logitsAll
  IO.println s!"  wrote {outDir}/logits.bin ({logitsAll.size} bytes — {nVal}×{flat} f32)"
  IO.println s!"next: python3 scripts/yolo_map_visdrone.py {outDir}/logits.bin {dataDir}/val.bin --grid {gH}"

def main (args : List String) : IO Unit := do
  match args with
  | "infer" :: rest =>
    let dataDir := rest[0]?.getD "data/visdrone448_g28"
    let outDir  := rest[1]?.getD "figures/yolo_visdrone448s16"
    IO.println s!"YOLOv1 VisDrone-448-s16 inference dump — data {dataDir} → {outDir}"
    inferDump dataDir outDir
  | _ =>
    let dataDir := args.head?.getD "data/visdrone448_g28"
    IO.println s!"YOLOv1 VisDrone-448 stride-16 detector (28×28 grid) — data dir: {dataDir}"
    r34Yolov1_448s16.train r34Yolov1_448s16Config dataDir DatasetKind.petsDet
