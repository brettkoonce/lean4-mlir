import LeanMlir

/-! YOLOv1 single-scale detector on VisDrone at 448 input / 14×14 grid.

    The validation-ladder rung above the 224/7×7 baseline (planning/yolo_drone.md
    WS-A, which collapsed to mAP 0.0000): change ONE thing — input resolution —
    and see whether it alone lifts detection off zero, before committing to the
    multi-scale build. Same ResNet-34 backbone + deep conv head as the Pets
    detector; at 448 the stride-32 backbone yields a 14×14 grid (196 cells vs 49),
    and a median VisDrone object grows from ~2×5 px to ~5×10 px — small, but
    visible to the backbone. Its own build prefix so it never collides with the
    Pets checkpoints.

    Usage:
      lake build yolov1-visdrone448
      # train (default): data dir with 448/14 train.bin + val.bin
      IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 \
        .lake/build/bin/yolov1-visdrone448 data/visdrone448
      # infer: dump [N,5880] logits.bin for scripts/yolo_map_visdrone.py
      .lake/build/bin/yolov1-visdrone448 infer data/visdrone448 figures/yolo_visdrone448
-/

def r34Yolov1_448 : NetSpec where
  -- Identical architecture to the Pets r34Yolov1, at 448² input. Backbone
  -- strides 2·2·1·2·2·2 = 32 ⇒ 448/32 = 14 ⇒ head output [B,30,14,14],
  -- flatten [B,5880]. Distinct name ⇒ distinct buildPrefix ⇒ own vmfbs/ckpts.
  name := "ResNet-34 + YOLOv1 448 (VisDrone)"
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

def r34Yolov1_448Config : TrainConfig where
  -- Same recipe as the 224 baseline (planning/yolo_final.md), shorter run: this
  -- rung only needs to answer "does resolution move mAP off zero", read from
  -- early checkpoints. Same LR/clip/focal so the ONLY change vs WS-A is the input.
  learningRate := 7.0e-4
  batchSize    := 16
  epochs       := 12
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := true
  warmupEpochs := 3
  gradClipNorm := 4.0
  headLrMult   := 1.0
  checkpointEveryNEpochs := 2   -- e2/e4/... for early-signal eval
  augment      := true
  lossKind     := LossKind.yolov1Masked
  useFocal     := true
  focalGamma   := 2.0
  bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)

/-- Infer mode: dump `[N, 5880]` val logits to `outDir/logits.bin`, matching the
    224 infer demo but at 448/14 dims (loader + flat width derived from the spec). -/
def inferDump (dataDir outDir : String) : IO Unit := do
  IO.FS.createDirAll outDir
  let spec := r34Yolov1_448
  let gH := spec.imageH / 32
  let gW := spec.imageW / 32
  let flat : Nat := 30 * gH * gW            -- 5880
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
    let logitsB ← IreeSession.forwardF32 sess spec.evalFnName
                    evalParams evalShapesBA imgs xShape batch.toUSize nOut
    logitsAll := logitsAll ++ logitsB.extract 0 (real * rowBytes)
  IO.FS.writeBinFile s!"{outDir}/logits.bin" logitsAll
  IO.println s!"  wrote {outDir}/logits.bin ({logitsAll.size} bytes — {nVal}×{flat} f32)"
  IO.println s!"next: python3 scripts/yolo_map_visdrone.py {outDir}/logits.bin {dataDir}/val.bin --grid {gH}"

def main (args : List String) : IO Unit := do
  match args with
  | "infer" :: rest =>
    let dataDir := rest[0]?.getD "data/visdrone448"
    let outDir  := rest[1]?.getD "figures/yolo_visdrone448"
    IO.println s!"YOLOv1 VisDrone-448 inference dump — data {dataDir} → {outDir}"
    inferDump dataDir outDir
  | _ =>
    let dataDir := args.head?.getD "data/visdrone448"
    -- USE_DIOU=1 switches the box loss to the FD-verified DIoU term (brick #1);
    -- its own build tag so the arm never collides with the √-MSE run.
    let useDiou := (← IO.getEnv "USE_DIOU").isSome
    let cfg := if useDiou then { r34Yolov1_448Config with useDiouBox := true }
               else r34Yolov1_448Config
    let spec := if useDiou then r34Yolov1_448.withBuildTag "diou" else r34Yolov1_448
    let boxName := if useDiou then "DIoU" else "sqrt-MSE"
    IO.println s!"YOLOv1 VisDrone-448 (14×14) — data {dataDir} — box loss: {boxName}"
    spec.train cfg dataDir DatasetKind.petsDet
