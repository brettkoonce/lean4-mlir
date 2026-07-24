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

/-- T1b class weights (planning/yolo_fpn.md): sqrt-inverse encoded-target class
    frequency, normalized so `Σ_c f_c·w_c = 1` — a pure redistribution that leaves
    the class term's total magnitude (and so its balance against box/objectness)
    unchanged. Counts from `scripts/fpn_class_freq.py` over data/visdrone_fpn:
    car 44.1% and pedestrian 21.2% of positives, and the unweighted e12 head
    predicted ONLY those two (5/10 classes never emitted). Full inverse frequency
    spans 45× and is needlessly violent; sqrt spans 6.7×. -/
def fpnClsWeights : List Float :=
  [0.8058, 1.4377, 2.1196, 0.5579, 1.3407, 1.7916, 2.9778, 3.7281, 2.6187, 1.2694]

/-- `tower` = number of 3×3 convs in the RetinaNet head tower per pyramid level
    (T2a). **0 = the minimal 1×1 head**, which is the T2-bias arm currently on the
    board; 4 is the RetinaNet default. Selected at run time by `FPN_TOWER` so the
    two arms are one binary, and folded into `name` so their checkpoints and vmfbs
    can never collide. -/
def r34FpnDetT (tower : Nat) : NetSpec where
  -- name is the on-disk checkpoint prefix: keep it DISTINCT from the anchor arm,
  -- from the unweighted FPN baseline, AND from the T1b (wcls) arm — all of their
  -- e2..e12 checkpoints are live A/B references and must not be clobbered.
  name := if tower == 0 then "ResNet-34 + FPN detector 448 wcls pb (VisDrone)"
          else s!"ResNet-34 + FPN detector 448 wcls pb tower{tower} (VisDrone)"
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
    .fpnDetect 256 128 256 512 14 3 tower
  ]

/-- The towerless (T2-bias) arm. `tower = 0` emits ZERO tower ops, so this spec is
    byte-identical to the pre-T2a codegen and the in-flight run stays reproducible. -/
def r34FpnDet : NetSpec := r34FpnDetT 0

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
  yoloClsWeights := fpnClsWeights       -- T1b: kept on (free, and better class spread)
  -- Tier 2, lever 1: the detector head now HAS a bias, initialized to the
  -- RetinaNet prior. This is the only change vs the T1b arm — a zero-init bias
  -- reproduces the biasless head exactly, so the T1b run is the control and no
  -- separate bias-off arm is needed. Targets the measured failure: objectness
  -- had AUC 0.742 but every logit squeezed into [−2.7, −1.2], because a
  -- bias-free 1×1 conv must synthesize the background offset from its weights.
  detPriorPi   := 0.01
  bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)

/-- Read the head-tower depth (T2a) from `FPN_TOWER`; 0 = the minimal 1×1 head. -/
def towerDepthFromEnv : IO Nat := do
  match (← IO.getEnv "FPN_TOWER") with
  | none => return 0
  | some v => return (v.trim.toNat?).getD 0

/-- Epoch-count override (`FPN_EPOCHS`), for the overfit probe: point the trainer
    at a 32-image subset and give it enough epochs to fit it. Defaults to the
    arm's configured 12 so every existing runbook is unchanged. -/
def epochsFromEnv (dflt : Nat) : IO Nat := do
  match (← IO.getEnv "FPN_EPOCHS") with
  | none => return dflt
  | some v => return (v.trim.toNat?).getD dflt

/-- Checkpoint interval override (`FPN_CKPT_EVERY`). The overfit probe runs
    hundreds of epochs and wants the loss trajectory, not 100 × 86 MB of
    snapshots. Defaults to the arm's configured 2. -/
def ckptEveryFromEnv (dflt : Nat) : IO Nat := do
  match (← IO.getEnv "FPN_CKPT_EVERY") with
  | none => return dflt
  | some v => return (v.trim.toNat?).getD dflt

/-- Learning-rate multiplier (`FPN_LR_MULT`). The overfit probe needs to separate
    "the trainer is THROTTLED" from "the trainer is BROKEN": if 10× the LR fits
    32 images, the update path works and the schedule is wrong; if it still
    cannot, the defect is in the gradient or update path itself. LR is a runtime
    scalar (the cosine schedule is computed host-side), so this needs no
    recompile. An integer multiplier rather than an absolute value because this
    toolchain has no `String.toFloat?`. Defaults to 1. -/
def lrMultFromEnv : IO Float := do
  match (← IO.getEnv "FPN_LR_MULT") with
  | none => return 1.0
  | some v => return ((v.trim.toNat?).getD 1).toFloat

/-- Global-norm gradient-clip override (`FPN_CLIP`), as a Nat; 0 disables the
    clip entirely. Measuring `%gcnorm` would only say whether the clip is ACTIVE;
    turning it off says whether it is CAUSAL, which is the actual question, and
    it needs no change to the train step's return arity. Note the clip threshold
    is baked into the emitted IR, so changing this forces a vmfb recompile.
    Defaults to the arm's configured 4.0. -/
def clipFromEnv (dflt : Float) : IO Float := do
  match (← IO.getEnv "FPN_CLIP") with
  | none => return dflt
  | some v => match v.trim.toNat? with
              | none => return dflt
              | some n => return n.toFloat

/-- Name suffix (`FPN_TAG`). The name IS the on-disk checkpoint prefix, so a probe
    run without a distinct tag silently overwrites the live arm's e2..e12
    checkpoints — which are the artifacts every measurement in
    planning/yolo_assignment.md is computed from. Empty by default. -/
def tagFromEnv : IO String := do
  match (← IO.getEnv "FPN_TAG") with
  | none => return ""
  | some v => return if v.trim.isEmpty then "" else s!" {v.trim}"

/-- Augmentation toggle (`FPN_AUG=1`). Turns on the FPN-path augmentation pack —
    YOLO-style HSV jitter (photometric, image-only) + horizontal flip (geometric,
    re-encoded on the flat [P3|P4|P5] target). OFF by default so the in-flight
    baseline arm stays byte-reproducible; this is an explicit A/B arm and MUST run
    under its own `FPN_TAG` so its checkpoints don't clobber the no-aug control. -/
def augFromEnv : IO Bool := do
  match (← IO.getEnv "FPN_AUG") with
  | none => return false
  | some v => return (v.trim == "1" || v.trim.toLower == "true")

/-- Infer: dump [N, Ntot] val logits for scripts/yolo_map_visdrone.py --fpn. -/
def inferDump (spec : NetSpec) (dataDir outDir : String) : IO Unit := do
  IO.FS.createDirAll outDir
  let flat : Nat := fpnNtot
  let evalVmfb := s!"{spec.buildPrefix}_fwd_eval.vmfb"
  let paramsPath := s!"{spec.buildPrefix}_params.bin"
  let bnPath := s!"{spec.buildPrefix}_bn_stats.bin"
  -- Announce WHICH ARM is being evaluated. The arm is selected by FPN_TOWER, and
  -- forgetting it silently evaluates a DIFFERENT arm's checkpoint rather than
  -- failing: every prefix/size/vmfb is self-consistent for the wrong spec, so no
  -- size check can catch it. (Cost one full 12-epoch eval sweep that reproduced
  -- the previous arm's numbers exactly — six identical rows was the only tell.)
  IO.println s!"  spec   : {spec.name}"
  IO.println s!"  prefix : {spec.buildPrefix}"
  IO.println s!"  params : {paramsPath} ({spec.totalParams} floats expected)"
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
  let tower ← towerDepthFromEnv
  let tag ← tagFromEnv
  let spec := { r34FpnDetT tower with name := (r34FpnDetT tower).name ++ tag }
  match args with
  | "infer" :: rest =>
    let dataDir := rest[0]?.getD "data/visdrone_fpn"
    let outDir  := rest[1]?.getD "figures/yolo_fpn"
    IO.println s!"FPN VisDrone inference dump (tower={tower}) — {dataDir} → {outDir}"
    inferDump spec dataDir outDir
  | _ =>
    let dataDir := args.head?.getD "data/visdrone_fpn"
    let epochs ← epochsFromEnv r34FpnDetConfig.epochs
    let ckptEvery ← ckptEveryFromEnv r34FpnDetConfig.checkpointEveryNEpochs
    let lrMult ← lrMultFromEnv
    let lr := r34FpnDetConfig.learningRate * lrMult
    let clip ← clipFromEnv r34FpnDetConfig.gradClipNorm
    let aug ← augFromEnv
    let cfg := { r34FpnDetConfig with epochs := epochs,
                                      checkpointEveryNEpochs := ckptEvery,
                                      learningRate := lr,
                                      gradClipNorm := clip,
                                      augment := aug }
    IO.println s!"FPN multi-scale VisDrone (56/28/14, 3 anchors/scale, Ntot={fpnNtot}, head tower={tower}) — data dir: {dataDir}"
    IO.println s!"  spec   : {spec.name}"
    IO.println s!"  epochs : {epochs}"
    IO.println s!"  lr     : {lr}  clip: {clip}"
    IO.println s!"  augment: {aug} (HSV jitter + hflip on the FPN path)"
    spec.train cfg dataDir DatasetKind.petsDet
