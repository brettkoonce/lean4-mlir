import LeanMlir

/-! FPN multi-scale YOLO detector on VisDrone at 448 (detection brick #3,
    planning/yolo_fpn.md bite 7/8). R34-ImageNet backbone tapped at C3/C4/C5
    (strides 8/16/32 → grids 56/28/14) → top-down neck (all → 256ch) → per-scale
    1×1 head (256 → A·15) → flat [B, Ntot] concat (Ntot = 3·15·(56²+28²+14²) =
    185220). Loss = multi-scale anchor YOLO (per-scale DIoU box + focal objectness
    + softmax class), summed over the 3 scales.

    Breaks the single-14×14-grid wall: only ~61% of GT are encodable at one scale
    (the anchor A=6 detector plateaus at 5.08% recall); the 3-scale FPN lifts
    encodable coverage to 88.2% (77% of GT are <24px → the new P3 scale).

    Data: data/visdrone_fpn (preprocess_visdrone.py --fpn). Anchors: the per-scale
    k-means priors in data/visdrone/anchors_fpn_{p3,p4,p5}.txt.

    Usage:
      lake build yolov1-visdrone-fpn
      IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 \
        .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
      .lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn figures/yolo_fpn
-/

-- Per-scale k-means priors (data/visdrone/anchors_fpn_{p3,p4,p5}.txt).
def fpnAnchorsP3 : List (Float × Float) :=
  [(0.006935, 0.014941), (0.015750, 0.028005), (0.033728, 0.035028)]
def fpnAnchorsP4 : List (Float × Float) :=
  [(0.023961, 0.070528), (0.055662, 0.068706), (0.093187, 0.094324)]
def fpnAnchorsP5 : List (Float × Float) :=
  [(0.060280, 0.168604), (0.107559, 0.204684), (0.181239, 0.149031)]

/-- Per-scale (grid, anchors) for P3/P4/P5. Order MUST match the codegen concat
    ([P3|P4|P5]) and the on-disk flat target laid out by process_split_fpn. -/
def fpnDetScales : List (Nat × List (Float × Float)) :=
  [(56, fpnAnchorsP3), (28, fpnAnchorsP4), (14, fpnAnchorsP5)]

def fpnNtot : Nat :=
  (fpnDetScales.map (fun sc => sc.2.length * 15 * sc.1 * sc.1)).foldl (·+·) 0

def r34FpnDet : NetSpec where
  name := "ResNet-34 + FPN detector 448 (VisDrone)"   -- DISTINCT from the anchor arm
  imageH := 448
  imageW := 448
  detStride := 32
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,   -- stride 4
    .residualBlock  64 128 4 2,   -- C3: 128ch, 56×56
    .residualBlock 128 256 6 2,   -- C4: 256ch, 28×28
    .residualBlock 256 512 3 2,   -- C5: 512ch, 14×14
    .fpnDetect 256 128 256 512 14 3
  ]

def r34FpnDetConfig : TrainConfig where
  learningRate := 4.0e-4                -- below the anchor arm's 7e-4: the 3-scale
                                        -- loss sums ~10× the cells ⇒ larger grads
  batchSize    := 8                     -- larger graph than the anchor arm
  epochs       := 12
  useAdam      := true
  weightDecay  := 0.0005
  cosineDecay  := true
  warmupEpochs := 3
  gradClipNorm := 4.0
  checkpointEveryNEpochs := 2
  augment      := false                 -- yoloAugment is single-box-format only
  focalGamma   := 2.0                   -- objectness focal γ (used by the FPN loss)
  fpnScales    := fpnDetScales          -- routes the loss to emitMultiScaleYoloLoss
  bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)

/-- Infer: dump [N, Ntot] val logits for scripts/yolo_map_visdrone.py --fpn. -/
def inferDump (dataDir outDir : String) : IO Unit := do
  IO.FS.createDirAll outDir
  let spec := r34FpnDet
  let flat : Nat := fpnNtot
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
  let (valImg, _t, nVal) ← F32.loadDetBinFpn (dataDir ++ "/val.bin")
                             spec.imageH.toUSize flat.toUSize
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
      for _ in [:batch - real] do imgs := imgs ++ lastImg
    let logitsB ← IreeSession.forwardF32 sess spec.evalFnName
                    evalParams evalShapesBA imgs xShape batch.toUSize nOut
    logitsAll := logitsAll ++ logitsB.extract 0 (real * rowBytes)
  IO.FS.writeBinFile s!"{outDir}/logits.bin" logitsAll
  IO.println s!"  wrote {outDir}/logits.bin ({logitsAll.size} bytes — {nVal}×{flat} f32)"

def main (args : List String) : IO Unit := do
  match args with
  | "infer" :: rest =>
    let dataDir := rest[0]?.getD "data/visdrone_fpn"
    let outDir  := rest[1]?.getD "figures/yolo_fpn"
    IO.println s!"FPN VisDrone inference dump — {dataDir} → {outDir}"
    inferDump dataDir outDir
  | _ =>
    let dataDir := args.head?.getD "data/visdrone_fpn"
    IO.println s!"FPN multi-scale VisDrone (56/28/14, 3 anchors/scale, Ntot={fpnNtot}) — data dir: {dataDir}"
    r34FpnDet.train r34FpnDetConfig dataDir DatasetKind.petsDet
