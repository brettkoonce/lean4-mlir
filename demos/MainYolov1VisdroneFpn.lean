import LeanMlir

/-! # `yolov1-visdrone-fpn` — the multi-scale FPN detector

```
lake build yolov1-visdrone-fpn
CUDA_VISIBLE_DEVICES=0 .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
```

Two backbones, selected by `FPN_BACKBONE`: `r50` (default, RSB-A3 77.2% top-1)
and `r34`. They carry different names, so their checkpoints and compiled graphs
cannot collide. The other knobs are `FPN_TOWER`, `FPN_TAG`, `FPN_AUG`,
`FPN_EPOCHS`, `FPN_CKPT_EVERY`, `FPN_LR_MULT` and `FPN_CLIP`.

⚠ `FPN_TAG` selects the checkpoint prefix and must be set on `infer` as well as
on training. Forgetting it does not fail — it silently evaluates a different
arm's weights, and the only tell is an epoch sweep whose rows are identical.

⚠ Only on an IREE box: `IREE_BACKEND=rocm` is required on gfx1100, because the
reduction workaround in `ireeCompileArgs` (`LeanMlir/Types.lean`) is gated on it
and the multi-scale loss's N-D→scalar reductions otherwise abort with
`'func.func' op failed to distribute`. Irrelevant under XLA/PJRT.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file: the config and entry point below ARE the program.
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

/-- ResNet-50 backbone variant. The first six layers are copied VERBATIM from
    `jax/MainResnet50Imagenet.lean`'s `resnet50Imagenet` — same order, same
    channels, same `convPadStyle` — because `bootstrapBackbone` is a PREFIX copy:
    it drops the checkpoint's leading floats onto the init's leading floats and
    only verifies that the bytes landed, not that they mean the same thing. Any
    divergence in these six lines silently loads a correct-sized, wrong-layout
    backbone. The classifier (`.globalAvgPool` + `.dense 2048 1000`) is what we
    drop, which is why the bootstrap count is 23,508,032 and not the file's full
    25,557,032 floats.

    At 448 input the stages land on 112 / 56 / 28 / 14, so C3/C4/C5 are 56/28/14
    — the FPN scales already in `fpnDetScales`, unchanged. Only the tap WIDTHS
    move (512/1024/2048 vs R34's 128/256/512), and `.fpnDetect` takes those as
    arguments, so this is a spec change with no new codegen. -/
def r50FpnDetT (tower : Nat) : NetSpec where
  name := if tower == 0 then "ResNet-50 + FPN detector 448 wcls pb (VisDrone)"
          else s!"ResNet-50 + FPN detector 448 wcls pb tower{tower} (VisDrone)"
  -- torchvision's Conv2d(3,64,7,stride=2,padding=3), matching the ImageNet render
  -- the A3 checkpoint was trained under. Omitting this mismatches the stem.
  convPadStyle := .symmetric
  imageH := 448
  imageW := 448
  detStride := 32
  layers := [
    .convBn 3 64 7 2 .same,
    -- ⚠ DELIBERATELY 2×2, where `resnet50Imagenet` has `.maxPool 3 2`.
    -- The train-step emitter's max-pool backward is a tile-compare-select that
    -- is correct ONLY for non-overlapping windows (MlirCodegen.lean:7698 says so
    -- outright), and size 3 > stride 2 overlaps: one input can be the max of
    -- several windows, so its gradient is a SUM the tiling never forms. It also
    -- happens to fail loudly first — the forward pads 224→225 but the backward
    -- reads `inShape` (224) against the padded SSA, so the graph does not even
    -- parse. Fixing that type error alone would trade a compile failure for a
    -- silently wrong gradient, which is worse.
    -- Safe to change because pooling is PARAMETER-FREE: the bootstrap prefix is
    -- untouched, and both windows take 224→112, so every downstream shape and
    -- the C3/C4/C5 taps are identical. The cost is a one-layer distribution
    -- shift — the backbone was pretrained under 3×3 pooling — which fine-tuning
    -- absorbs. The R34 detector arm above has always done exactly this.
    .maxPool 2 2,
    .bottleneckBlock   64  256 3 1,   -- C2: stride 4,  112×112
    .bottleneckBlock  256  512 4 2,   -- C3: 512ch,      56×56
    .bottleneckBlock  512 1024 6 2,   -- C4: 1024ch,     28×28
    .bottleneckBlock 1024 2048 3 2,   -- C5: 2048ch,     14×14
    .fpnDetect 256 512 1024 2048 14 3 tower
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
  yoloClsWeights := fpnClsWeights       -- T1b: kept on (free, and better class spread)
  -- Tier 2, lever 1: the detector head now HAS a bias, initialized to the
  -- RetinaNet prior. This is the only change vs the T1b arm — a zero-init bias
  -- reproduces the biasless head exactly, so the T1b run is the control and no
  -- separate bias-off arm is needed. Targets the measured failure: objectness
  -- had AUC 0.742 but every logit squeezed into [−2.7, −1.2], because a
  -- bias-free 1×1 conv must synthesize the background offset from its weights.
  detPriorPi   := 0.01
  bootstrapBackbone := some (".lake/build/jax_r34_imagenet.bin", 21284672)

/-- The R50 arm's config. Identical to `r34FpnDetConfig` except the backbone, so
    a same-schedule A/B attributes any delta to the base and nothing else.

    `23508032` = the A3 checkpoint's 25,557,032 floats minus the 1000-way
    classifier (2048·1000 + 1000). The count is the ONLY thing standing between a
    real bootstrap and a silent misload, so it is derived, not guessed.

    ⛔ It does NOT load `r50_a3_params.bin` (the A3 checkpoint) directly, because
    that file's BN slots are not in this emitter's convention. Its conv weights are
    perfect — std 0.17, every segment matching `ckpt_e100.state.npz` — but its
    per-channel BN values run to **5.2e6**, against R34's γ∈[0,0.60] / β∈[−0.68,0.77].
    Bootstrapped raw it starts at loss 5.8e8 where R34 starts at 717.7. Those values
    are presumably meaningful to the render that wrote them (that run really did
    score 77.2%), most likely a scale folded against running statistics kept in the
    `.state.npz` — which the generic bootstrap loads as zeros, so nothing cancels.

    So we transfer what is portable: **conv weights from A3, BN reset to γ=1, β=0**
    (`r50_a3convonly_params.bin`, built by walking the npz and replacing each conv's
    two following per-channel tensors). BN re-learns in a few hundred steps; the
    convolutional features are the bulk of what pretraining buys. Max abs in the
    resulting body is 1.98.

    Not a codegen bug: the same spec under `FPN_NOBOOTSTRAP=1` starts at 8,284.8 and
    reaches ~1,018 by step 100, so `.bottleneckBlock` emits fine. The clean fix, if
    full transfer is ever wanted, is to re-export A3 through the same path that
    produced R34's working `jax_r34_imagenet.bin` (`LEAN_MLIR_PARAMS_OUT`). -/
def r50FpnDetConfig : TrainConfig :=
  { r34FpnDetConfig with
      bootstrapBackbone := some (".lake/build/r50_a3convonly_params.bin", 23508032) }

/-- Backbone selector (`FPN_BACKBONE`), `r50` (default) or `r34`.

    R50 is the default because it is both the better base — RSB-A3, 77.2% top-1
    against R34's ~74% — and the only one that still has a loadable checkpoint:
    `jax_r34_imagenet.bin` was deleted and can only be regenerated by a full
    ImageNet run, whereas the A3 weights are a plain float32 parameter file.
    `r34` is kept selectable so the 0.1386 arm can be reproduced if that file
    ever comes back. -/
def backboneFromEnv : IO String := do
  match (← IO.getEnv "FPN_BACKBONE") with
  | none => return "r50"
  | some v => return if v.trimAscii.toString.isEmpty then "r50" else v.trimAscii.toString.toLower

/-- Drop the pretrained bootstrap (`FPN_NOBOOTSTRAP=1`), leaving a pure He init.

    This is the control that separates "the checkpoint is being misread" from "the
    emitted graph is wrong", and it is the first thing to run when an arm's loss
    scale does not match a known arm's. R50-FPN starts at 5.8e8 where R34-FPN
    starts at 7.2e2; if He init reproduces the 5.8e8 the weights are innocent and
    the bottleneck emit is the suspect, and if it lands near R34's He-init range
    the weights are being interpreted wrongly despite matching in count and order. -/
def noBootstrapFromEnv : IO Bool := do
  match (← IO.getEnv "FPN_NOBOOTSTRAP") with
  | none => return false
  | some v => return (v.trimAscii.toString == "1" || v.trimAscii.toString.toLower == "true")

/-- Read the head-tower depth (T2a) from `FPN_TOWER`; 0 = the minimal 1×1 head. -/
def towerDepthFromEnv : IO Nat := do
  match (← IO.getEnv "FPN_TOWER") with
  | none => return 0
  | some v => return (v.trimAscii.toNat?).getD 0

/-- Epoch-count override (`FPN_EPOCHS`), for the overfit probe: point the trainer
    at a 32-image subset and give it enough epochs to fit it. Defaults to the
    arm's configured 12 so every existing runbook is unchanged. -/
def epochsFromEnv (dflt : Nat) : IO Nat := do
  match (← IO.getEnv "FPN_EPOCHS") with
  | none => return dflt
  | some v => return (v.trimAscii.toNat?).getD dflt

/-- Checkpoint interval override (`FPN_CKPT_EVERY`). The overfit probe runs
    hundreds of epochs and wants the loss trajectory, not 100 × 86 MB of
    snapshots. Defaults to the arm's configured 2. -/
def ckptEveryFromEnv (dflt : Nat) : IO Nat := do
  match (← IO.getEnv "FPN_CKPT_EVERY") with
  | none => return dflt
  | some v => return (v.trimAscii.toNat?).getD dflt

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
  | some v => return ((v.trimAscii.toNat?).getD 1).toFloat

/-- Global-norm gradient-clip override (`FPN_CLIP`), as a Nat; 0 disables the
    clip entirely. Measuring `%gcnorm` would only say whether the clip is ACTIVE;
    turning it off says whether it is CAUSAL, which is the actual question, and
    it needs no change to the train step's return arity. Note the clip threshold
    is baked into the emitted IR, so changing this forces a vmfb recompile.
    Defaults to the arm's configured 4.0. -/
def clipFromEnv (dflt : Float) : IO Float := do
  match (← IO.getEnv "FPN_CLIP") with
  | none => return dflt
  | some v => match v.trimAscii.toNat? with
              | none => return dflt
              | some n => return n.toFloat

/-- Name suffix (`FPN_TAG`). The name IS the on-disk checkpoint prefix, so a probe
    run without a distinct tag silently overwrites the live arm's e2..e12
    checkpoints — which are the artifacts every measurement in
    planning/yolo_assignment.md is computed from. Empty by default. -/
def tagFromEnv : IO String := do
  match (← IO.getEnv "FPN_TAG") with
  | none => return ""
  | some v => return if v.trimAscii.toString.isEmpty then "" else s!" {v.trimAscii.toString}"

/-- Augmentation toggle (`FPN_AUG=1`). Turns on the FPN-path augmentation pack —
    YOLO-style HSV jitter (photometric, image-only) + horizontal flip (geometric,
    re-encoded on the flat [P3|P4|P5] target). OFF by default so the in-flight
    baseline arm stays byte-reproducible; this is an explicit A/B arm and MUST run
    under its own `FPN_TAG` so its checkpoints don't clobber the no-aug control. -/
def augFromEnv : IO Bool := do
  match (← IO.getEnv "FPN_AUG") with
  | none => return false
  | some v => return (v.trimAscii.toString == "1" || v.trimAscii.toString.toLower == "true")

/-- Box-aware affine augmentation, as three PERCENT knobs (this toolchain has no
    `String.toFloat?`, same reason `FPN_LR_MULT` is an integer):

      `FPN_AFFINE`           firing probability per image, 0-100. 0 = off = default.
      `FPN_AFFINE_SCALE`     scale gain, so 25 ⇒ per-image scale in [0.75, 1.25].
      `FPN_AFFINE_TRANSLATE` translate gain as a % of the frame, so 10 ⇒ ±0.1.

    ⚠ The defaults are deliberately GENTLER than Ultralytics' (scale 0.5,
    translate 0.1). Halving an image is standard practice on COCO, where objects
    are large; here the median object is a handful of pixels across, and at
    scale 0.5 a 4 px car becomes 2 px and falls under P3's stride-8 grid. The
    honest experiment is a scale LADDER under separate tags, not one setting
    copied from a dataset with the opposite size distribution.

    Independent of `FPN_AUG`: this stacks ON TOP of the HSV+hflip pack, which is
    already measured as worth 0.1243 → 0.1674 at 50 epochs. Run it as
    `FPN_AUG=1 FPN_AFFINE=<p>` against an `FPN_AUG=1` control on the same
    schedule, or the comparison confounds two changes. -/
def affinePctFromEnv (name : String) (dflt : Nat) : IO Float := do
  match (← IO.getEnv name) with
  | none => return dflt.toFloat / 100.0
  | some v => return ((v.trimAscii.toNat?).getD dflt).toFloat / 100.0

/-- Infer: dump [N, Ntot] val logits for scripts/yolo_map_visdrone.py --fpn. -/
def inferDump (spec : NetSpec) (dataDir outDir : String) : IO Unit := do
  IO.FS.createDirAll outDir
  let flat : Nat := fpnNtot
  -- Backend-aware: `.vmfb` on IREE, the `.mlir` itself on XLA/PJRT. Without this
  -- the guard below hard-fails on XLA — `infer` would break while `train` worked,
  -- and training is what gets exercised first.
  let evalVmfb ← NetSpec.graphArtifact spec.buildPrefix "fwd_eval"
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
    IO.eprintln s!"ERROR: no eval graph at {evalVmfb}; train first"; IO.Process.exit 1
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ←
    if ← System.FilePath.pathExists bnPath then IO.FS.readBinFile bnPath
    else F32.const spec.nBnStats.toUSize 0.0
  let evalParams := params.append bnStats
  let sess ← LowererSession.create evalVmfb
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
    let logitsB ← LowererSession.forwardF32 sess spec.evalFnName
                    evalParams evalShapesBA imgs xShape batch.toUSize nOut
    logitsAll := logitsAll ++ logitsB.extract 0 (real * rowBytes)
  IO.FS.writeBinFile s!"{outDir}/logits.bin" logitsAll
  IO.println s!"  wrote {outDir}/logits.bin ({logitsAll.size} bytes — {nVal}×{flat} f32)"

def runYolov1VisdroneFpn (args : List String) : IO Unit := do
  let tower ← towerDepthFromEnv
  let tag ← tagFromEnv
  let backbone ← backboneFromEnv
  -- The backbone selects BOTH the spec and the config, and the name carries it,
  -- so the two arms can never share a checkpoint prefix or a compiled graph.
  let baseSpec := if backbone == "r34" then r34FpnDetT tower else r50FpnDetT tower
  let baseCfg  := if backbone == "r34" then r34FpnDetConfig else r50FpnDetConfig
  let spec := { baseSpec with name := baseSpec.name ++ tag }
  match args with
  | "emit-deploy" :: rest =>
    -- Write a BATCH-1 eval graph for edge deployment. The trained artifacts are
    -- emitted at the training batch size (8); a camera runs one frame at a time,
    -- and paying for 8 would divide the frame rate by 8.
    -- Nothing else changes: same spec, same weights, same 190-argument calling
    -- convention (image, then 189 weight tensors = 21,548,743 params ++ 17,024
    -- BN stats, in that order). deploy/orin_detect.py parses that signature out
    -- of this file rather than being told it, so the two cannot drift.
    let outDir := rest[0]?.getD "deploy/build"
    IO.FS.createDirAll outDir
    let mlir := MlirCodegen.generateEval spec 1
    let path := s!"{outDir}/detector_fwd_eval_b1.mlir"
    IO.FS.writeFile path mlir
    IO.println s!"  spec   : {spec.name}"
    IO.println s!"  params : {spec.totalParams} floats + {spec.nBnStats} BN stats"
    IO.println s!"  wrote  : {path} ({mlir.length} chars, batch 1)"
    IO.println s!"  weights: {spec.buildPrefix}_params.bin / _bn_stats.bin"
  | "infer" :: rest =>
    let dataDir := rest[0]?.getD "data/visdrone_fpn"
    let outDir  := rest[1]?.getD "runs/yolo_fpn"
    IO.println s!"FPN VisDrone inference dump (tower={tower}) — {dataDir} → {outDir}"
    inferDump spec dataDir outDir
  | _ =>
    let dataDir := args.head?.getD "data/visdrone_fpn"
    let epochs ← epochsFromEnv baseCfg.epochs
    let ckptEvery ← ckptEveryFromEnv baseCfg.checkpointEveryNEpochs
    let lrMult ← lrMultFromEnv
    let lr := baseCfg.learningRate * lrMult
    let clip ← clipFromEnv baseCfg.gradClipNorm
    let aug ← augFromEnv
    let noBoot ← noBootstrapFromEnv
    let baseCfg := if noBoot then { baseCfg with bootstrapBackbone := none } else baseCfg
    let affProb ← affinePctFromEnv "FPN_AFFINE" 0
    let affScale ← affinePctFromEnv "FPN_AFFINE_SCALE" 25
    let affTrans ← affinePctFromEnv "FPN_AFFINE_TRANSLATE" 10
    let cfg := { baseCfg with epochs := epochs,
                                      checkpointEveryNEpochs := ckptEvery,
                                      learningRate := lr,
                                      gradClipNorm := clip,
                                      augment := aug,
                                      fpnAffineProb := affProb,
                                      fpnAffineScale := affScale,
                                      fpnAffineTranslate := affTrans }
    IO.println s!"FPN multi-scale VisDrone (56/28/14, 3 anchors/scale, Ntot={fpnNtot}, head tower={tower}) — data dir: {dataDir}"
    IO.println s!"  spec   : {spec.name}"
    IO.println s!"  epochs : {epochs}"
    IO.println s!"  lr     : {lr}  clip: {clip}"
    IO.println s!"  augment: {aug} (HSV jitter + hflip on the FPN path)"
    if affProb > 0.0 then
      IO.println s!"  affine : p={affProb} scale=±{affScale} translate=±{affTrans} \
(box-aware, target re-encoded)"
    else
      IO.println s!"  affine : off (FPN_AFFINE=<percent> to enable)"
    spec.train cfg dataDir DatasetKind.petsDet

def main (args : List String) : IO Unit := runYolov1VisdroneFpn args
