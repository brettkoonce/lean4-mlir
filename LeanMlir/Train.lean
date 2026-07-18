import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.SpecHelpers

/-! Unified training loop for any spec that uses the Adam codegen path.

Each Main*Train.lean used to inline ~250 lines of identical
code: generate MLIR → compile vmfbs → init params → for-each-epoch
{ shuffle, for-each-batch { trainStepAdamF32, EMA, log }, val eval } →
save. The body was the same for ResNet/MobileNet/EfficientNet/ViT/VGG
modulo a handful of name strings.

`NetSpec.train` is the extracted function. A trainer is now:

    def main (args : List String) : IO Unit :=
      resnet34.train resnet34Config (args.head?.getD "data/imagenette")

The 250 lines collapse to one call. Adding a new architecture means
defining the spec + a `TrainConfig` value — no copy-paste plumbing. -/

namespace NetSpec

/-- File-path prefix for the generated MLIR / vmfb / saved-params files
    associated with this spec. Uses the sanitized spec name so adding
    a new spec automatically gets a unique non-colliding prefix without
    the trainer author having to pick one. -/
def buildPrefix (spec : NetSpec) : String :=
  ".lake/build/" ++ spec.sanitizedName ++
    (if spec.buildTag.isEmpty then "" else "_" ++ MlirCodegen.sanitize spec.buildTag)

/-- Set `buildTag`, so an ablation arm owns its own artifacts. Reads as
    `(net.withBuildTag "ce").train …` at the call site. (Not `buildTag` — that
    name belongs to the field itself.) -/
def withBuildTag (spec : NetSpec) (tag : String) : NetSpec :=
  { spec with buildTag := tag }

/-- The fully qualified `<module>.<func>` name the train step's main
    function lives at — what `trainStepAdamF32` wants as its `fnName`.
    Mirrors how `MlirCodegen.generateTrainStep` builds the module
    name from the third argument it's given (`jit_<sanitized>_train_step`). -/
def trainFnName (spec : NetSpec) : String :=
  "jit_" ++ spec.sanitizedName ++ "_train_step.main"

/-- Cache key for an MLIR string under the current IREE backend.
    Combines the MLIR content with the backend env var so switching
    `IREE_BACKEND=rocm` ↔ `cuda` invalidates the cache. -/
private def cacheKey (mlir : String) : IO String := do
  let backend ← (IO.getEnv "IREE_BACKEND").map (·.getD "cuda")
  return toString (mlir ++ "::" ++ backend).hash

/-- Find iree-compile: check .venv/bin first (local dev), then PATH. -/
private def findIreeCompile : IO String := do
  if ← System.FilePath.pathExists ".venv/bin/iree-compile" then
    return ".venv/bin/iree-compile"
  -- Fall back to PATH (Docker, system install)
  return "iree-compile"

private def runIree (mlirPath outPath : String) : IO Bool := do
  let args ← ireeCompileArgs mlirPath outPath
  let compiler ← findIreeCompile
  let r ← IO.Process.output { cmd := compiler, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile failed for {mlirPath}: {r.stderr.take 3000}"
    return false
  return true

/-- Compile `mlirPath` to `outPath` via iree-compile, but skip the work
    if `outPath` exists and `outPath ++ ".hash"` matches the cache key
    for the current MLIR + IREE backend. Returns (compiledNewly, success). -/
private def runIreeCached (mlirPath outPath mlir : String) : IO (Bool × Bool) := do
  let key ← cacheKey mlir
  let hashPath := outPath ++ ".hash"
  let vmfbExists ← System.FilePath.pathExists outPath
  let hashExists ← System.FilePath.pathExists hashPath
  if vmfbExists && hashExists then
    let cached ← IO.FS.readFile hashPath
    if cached.trimAscii == key then
      return (false, true)  -- cache hit, skip the slow path
  -- Cache miss → compile and write the sidecar.
  let ok ← runIree mlirPath outPath
  if ok then
    IO.FS.writeFile hashPath key
  return (true, ok)

/-- Generate forward / eval-forward / train-step MLIR for `spec`,
    write them to `.lake/build/<sanitized>_*.mlir`, and compile each
    to a `.vmfb`. Cached on the MLIR content + IREE backend so a
    second run with no codegen changes skips iree-compile entirely
    (saves ~10-15 min for ResNet-sized models). Returns the path to
    the train-step vmfb.

    `useSeg = true` switches the train-step codegen to per-pixel
    softmax CE (segmentation), where labels are an int32 `[B, H, W]`
    tensor instead of a `[B]` class index. Mutually exclusive with
    soft-label / focal / label-smoothing flows. -/
def compileVmfbs (spec : NetSpec) (cfg : TrainConfig)
    (useSeg : Bool := false) : IO String := do
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix

  IO.eprintln "Generating train step MLIR..."
  -- Mixup / CutMix / KNN-Mixup produce fractional labels; switch to the soft-label codegen.
  let useSoftLabels := cfg.useMixup || cfg.useCutmix || cfg.useKnnMixup
  -- ── Resolve effective LossKind (planning/yolo_final.md R1) ──
  -- If `cfg.lossKind` is at its default `.classCE`, derive from the
  -- legacy booleans for back-compat with every existing trainer.
  let lossKind : LossKind :=
    if cfg.lossKind != .classCE then cfg.lossKind
    else if cfg.useYolov1 then .yolov1Masked
    else if useSeg then .perPixelCE
    else if useSoftLabels then .softLabelCE
    else .classCE
  -- ── Single-match mutex validation (replaces the prior chain of throws) ──
  -- Modifiers (useFocal, labelSmoothing, useMixup/Cutmix/KnnMixup) only
  -- apply to specific kinds. Reject the rest at compile time so misuse
  -- shows up here, not as a wrong-loss-value 100 steps in.
  match lossKind with
  | .classCE =>
    -- useFocal + labelSmoothing forbidden together; useFocal forbids
    -- soft labels (mixup et al.) which would have routed us to softLabelCE.
    if cfg.useFocal && cfg.labelSmoothing != 0.0 then
      throw <| IO.userError "useFocal requires labelSmoothing = 0 (focal mixes poorly with smoothing)"
  | .softLabelCE =>
    if cfg.useFocal then
      throw <| IO.userError "softLabelCE (mixup/cutmix/knnMixup) is incompatible with useFocal — focal applies to int-label CE only"
  | .perPixelCE =>
    if useSoftLabels then
      throw <| IO.userError "perPixelCE (segmentation) is incompatible with mixup/cutmix/knnMixup — per-pixel labels can't be mixed batch-wise"
    if cfg.useFocal then
      throw <| IO.userError "perPixelCE + focal not yet supported on this path — use lossKind := .perPixelFocalCE γ, which is the segmentation focal emitter"
    -- Label smoothing IS accepted, as of the FD check at ls=0.1
    -- (`seg_loss_probe_check.py`, grad vs central differences 1.07e-07). The
    -- emitter has had it since Phase 0 (`emitPerPixelCEBlock`'s
    -- smoothOn/smoothOff) and this guard was the only thing gating it — but it
    -- was gating an UNVERIFIED path, because nothing in the probe ran at ls > 0
    -- until then. Verified, then lifted; not lifted because it looked free.
  | .perPixelWeightedCE weights =>
    if useSoftLabels then
      throw <| IO.userError "perPixelWeightedCE (segmentation) is incompatible with mixup/cutmix/knnMixup — per-pixel labels can't be mixed batch-wise"
    if cfg.useFocal then
      throw <| IO.userError "perPixelWeightedCE + focal not yet supported (both reweight the loss; composing them wants its own scoping)"
    -- Label smoothing IS accepted (FD 1.31e-07 at ls=0.1). The weight rides on
    -- `%seg_mask`, the raw EQ comparison, so it stays a weight on the TRUE
    -- class while the *target* softens — which is the composition you want, and
    -- is what the numpy model FD checks against.
    -- Caught here rather than in the emitter: a wrong-length list becomes a
    -- `dense<[...]> : tensor<NCxf32>` shape mismatch thousands of lines into
    -- the generated MLIR, which is a miserable way to learn you typed one
    -- weight too few.
    let nc := (spec.layers.getLast?.map (·.outChannels)).getD 0
    if weights.length != nc then
      throw <| IO.userError s!"perPixelWeightedCE: got {weights.length} weights for a {nc}-class head — they must match 1:1"
    if weights.any (· <= 0.0) then
      throw <| IO.userError "perPixelWeightedCE: weights must be strictly positive (a zero weight silently deletes a class from the loss; a negative one ascends it)"
  | .perPixelFocalCE gamma =>
    if useSoftLabels then
      throw <| IO.userError "perPixelFocalCE (segmentation) is incompatible with mixup/cutmix/knnMixup — per-pixel labels can't be mixed batch-wise"
    if cfg.useFocal then
      throw <| IO.userError "perPixelFocalCE already IS the focal path — cfg.useFocal selects the classCE/YOLOv1 focal modifier and would double-apply; set useFocal := false and put γ in the LossKind"
    if cfg.labelSmoothing != 0.0 then
      throw <| IO.userError "perPixelFocalCE + label smoothing not yet supported — focal is defined on p_t, the probability of *the* true class, which a smoothed target does not name"
    if gamma < 0.0 then
      throw <| IO.userError "perPixelFocalCE: γ must be ≥ 0 (γ=0 is plain perPixelCE; γ<0 up-weights the examples already correct)"
  | .perPixelDice | .perPixelDiceCE =>
    if useSoftLabels then
      throw <| IO.userError "perPixelDice/DiceCE (segmentation) is incompatible with mixup/cutmix/knnMixup — per-pixel labels can't be mixed batch-wise"
    if cfg.useFocal then
      throw <| IO.userError "perPixelDice/DiceCE + focal not yet supported"
    -- Label smoothing IS accepted here, unlike .perPixelCE: `.diceCE` applies
    -- it to its CE half only (the Dice half one-hots raw — smoothing a
    -- set-overlap ratio is meaningless). `.perPixelDice` alone ignores it,
    -- so reject that combination rather than silently dropping it.
    if lossKind == .perPixelDice && cfg.labelSmoothing != 0.0 then
      throw <| IO.userError "perPixelDice + label smoothing is meaningless (Dice is a set-overlap ratio, not a log-likelihood target) — use .perPixelDiceCE if you want smoothing on the CE half"
  | .floatTargetMse =>
    -- DDPM bypasses compileVmfbs entirely (see demos/MainCifarDdpm*Train.lean);
    -- reaching this branch via compileVmfbs is currently unused but reserved.
    pure ()
  | .yolov1Masked =>
    if useSoftLabels then
      throw <| IO.userError "yolov1Masked is incompatible with useMixup/useCutmix/useKnnMixup — YOLOv1 targets are per-cell float tensors, not class-mixable"
    -- useFocal IS allowed here: for YOLOv1 it selects the sigmoid focal-BCE
    -- objectness path (T3/T4/T5) instead of raw-MSE — the fg/bg imbalance fix
    -- (planning/yolo_final.md §3). The class term (T6) stays softmax-CE either way.
    if useSeg then
      throw <| IO.userError "yolov1Masked is incompatible with segmentation — different target shape ([B,30,7,7] float vs [B,H,W] int32)"
    if cfg.labelSmoothing != 0.0 then
      throw <| IO.userError "yolov1Masked is incompatible with labelSmoothing — smoothing applies to one-hot CE, not box-regression MSE"
  | .bce =>
    -- BCE-with-logits (RSB-A2) is implemented on the JAX backend only; the
    -- IREE/MLIR train-step codegen (this path) has no sigmoid-BCE emitter.
    throw <| IO.userError "lossKind = .bce is JAX-only (RSB-A2) — the IREE/MLIR backend does not implement BCE-with-logits; run it via runJax"
  let useYolov1Codegen := lossKind == .yolov1Masked
  let trainMlir := MlirCodegen.generateTrainStep spec cfg.batchSize ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (labelSmoothing := cfg.labelSmoothing)
    (weightDecay := cfg.weightDecay)
    (useAdam := cfg.useAdam)
    (useMuon := cfg.useMuon)
    (useShampoo := cfg.useShampoo)
    (useSoftLabels := useSoftLabels)
    (useFocal := cfg.useFocal)
    (focalGamma := cfg.focalGamma)
    (useSeg := useSeg)
    (useYolov1 := useYolov1Codegen)
    -- YOLOv1 grid follows the input resolution (stride-32 backbone): 224→7,
    -- 448→14. Derived so the higher-res VisDrone path emits its loss at the
    -- right grid without a separate codegen entry point.
    (yoloGridH := spec.imageH / spec.detStride) (yoloGridW := spec.imageW / spec.detStride)
    (gradClipNorm := cfg.gradClipNorm)
    (headLrMult := cfg.headLrMult)
    (segLoss := lossKind.segLoss)
    (useDiouBox := cfg.useDiouBox)
    (yoloAnchors := cfg.anchors)
    (fpnScales := cfg.fpnScales)
    (yoloClsWeights := cfg.yoloClsWeights)
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.eprintln s!"  {trainMlir.length} chars"

  let fwdMlir := MlirCodegen.generate spec cfg.batchSize
  IO.FS.writeFile s!"{pfx}_fwd.mlir" fwdMlir

  let evalFwdMlir := MlirCodegen.generateEval spec cfg.batchSize
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" evalFwdMlir

  IO.eprintln "Compiling vmfbs..."
  let (fwdNew, fwdOk) ← runIreeCached s!"{pfx}_fwd.mlir" s!"{pfx}_fwd.vmfb" fwdMlir
  if fwdOk then IO.eprintln (if fwdNew then "  forward compiled" else "  forward (cached)")

  let (evalNew, evalOk) ← runIreeCached s!"{pfx}_fwd_eval.mlir" s!"{pfx}_fwd_eval.vmfb" evalFwdMlir
  if evalOk then IO.eprintln (if evalNew then "  eval forward compiled" else "  eval forward (cached)")

  let (trainNew, trainOk) ← runIreeCached s!"{pfx}_train_step.mlir" s!"{pfx}_train_step.vmfb" trainMlir
  if trainOk then
    IO.eprintln (if trainNew then "  train step compiled" else "  train step (cached)")
  else
    IO.Process.exit 1
  return s!"{pfx}_train_step.vmfb"

/-- Dataset-specific I/O: where the training/val data lives, how big each
    image is on disk, and what augmentation to apply per batch. Each
    dataset (Imagenette / CIFAR-10 / MNIST) constructs one of these and
    `runTraining` consumes it. -/
private structure DatasetIO where
  /-- Pixels per training image as stored on disk (before any crop). -/
  trainPixels : Nat
  /-- Pixels per val image as stored on disk. -/
  valPixels : Nat
  /-- Channels (1 for MNIST, 3 for CIFAR/Imagenette). -/
  channels : Nat
  /-- Bytes per label record. 4 for int32-class classification (default),
      H*W for per-pixel segmentation masks (Pets: 224*224 = 50176). -/
  labelBytesPerRecord : Nat := 4
  /-- Named unions of raw classes to additionally report **Dice** for at
      eval. Empty (the default) for datasets whose raw classes are already
      the entity the field reports — pets' trimap, where per-class IoU is
      the whole story.

      BraTS is the exception this exists for: the literature scores nested
      unions (WT/TC/ET), not the raw labels, and because the unions are
      nested they cannot be recovered from per-class IoU after the fact.
      See planning/brats_demo.md Workstream F. -/
  segRegions : List (String × List Nat) := []
  loadTrain : String → IO (ByteArray × ByteArray × Nat)
  loadVal   : String → IO (ByteArray × ByteArray × Nat)
  /-- Apply augmentation to a slice of training data. Receives the raw
      slice, the batch size, and a seed (epoch * 10000 + bi). Must
      return a buffer matching `inputFlatDim spec` floats per image. -/
  augmentBatch : (raw : ByteArray) → (batch : USize) → (seed : Nat) → IO ByteArray
  /-- Deterministic preprocessing for **training** when cfg.augment=false.
      Datasets whose on-disk training size differs from the model's
      input size (e.g. Imagenette: 256 stored, 224 expected) override
      this to do a center-crop. MNIST and CIFAR default to identity
      since their stored size already matches the input. NOT used for
      validation — val data is loaded at the model's input size
      directly, see `valPreprocessBatch`. -/
  preprocessBatch : (raw : ByteArray) → (batch : USize) → IO ByteArray := fun raw _ => return raw
  /-- Deterministic preprocessing for **validation**. Defaults to
      identity because val data is loaded at the model's input size in
      every dataset (Imagenette: 224 via `loadImagenette` even though
      train is 256; MNIST/CIFAR: same size as train). Override only if
      a future dataset stores val data at a different size from the
      input. -/
  valPreprocessBatch : (raw : ByteArray) → (batch : USize) → IO ByteArray := fun raw _ => return raw
  deriving Inhabited

/-- Imagenette: 256×256 train, 224×224 val, random-crop-to-224 + hflip.
    When augment=false, falls back to deterministic center-crop 256→224
    so the model still receives correctly-shaped input. -/
private def imagenetteIO : DatasetIO where
  trainPixels := 3 * 256 * 256
  valPixels   := 3 * 224 * 224
  channels    := 3
  loadTrain   := fun dir => F32.loadImagenetteSized (dir ++ "/train.bin") 256
  loadVal     := fun dir => F32.loadImagenette (dir ++ "/val.bin")
  augmentBatch := fun raw batch seed => do
    let cropped ← F32.randomCrop raw batch 3 256 256 224 224 seed.toUSize
    F32.randomHFlip cropped batch 3 224 224 (seed + 7777).toUSize
  -- Train data is stored at 256×256 and crops down to 224×224 here
  -- (augment=false path). Val data is stored at 224×224 (see
  -- `loadImagenette`'s `img_size=224` in f32_helpers.c) — DOES NOT
  -- need cropping. The eval pipeline must NOT call this on val data;
  -- it'd read 256-pixel-strided memory across a 224-pixel-strided
  -- buffer and pull pixels from neighbouring images in the batch.
  -- Eval uses the input directly; see runTraining's eval loop.
  preprocessBatch := fun raw batch =>
    F32.centerCrop raw batch 3 256 256 224 224

/-- MNIST: 28×28 IDX format, no augmentation. -/
private def mnistIO : DatasetIO where
  trainPixels := 1 * 28 * 28
  valPixels   := 1 * 28 * 28
  channels    := 1
  loadTrain := fun dir => do
    let (imgs, n) ← F32.loadIdxImages (dir ++ "/train-images-idx3-ubyte")
    let (lbls, _) ← F32.loadIdxLabels (dir ++ "/train-labels-idx1-ubyte")
    return (imgs, lbls, n)
  loadVal := fun dir => do
    let (imgs, n) ← F32.loadIdxImages (dir ++ "/t10k-images-idx3-ubyte")
    let (lbls, _) ← F32.loadIdxLabels (dir ++ "/t10k-labels-idx1-ubyte")
    return (imgs, lbls, n)
  augmentBatch := fun raw _ _ => return raw

/-- CIFAR-10: 5 train batches concatenated, 1 test batch, random hflip. -/
private def cifar10IO : DatasetIO where
  trainPixels := 3 * 32 * 32
  valPixels   := 3 * 32 * 32
  channels    := 3
  loadTrain := fun dir => do
    let mut raw : ByteArray := .empty
    let mut labels : ByteArray := .empty
    let mut nTotal : Nat := 0
    for i in [1:6] do
      let batchRaw ← IO.FS.readBinFile s!"{dir}/cifar-10/data_batch_{i}.bin"
      let n := batchRaw.size / 3073
      for j in [:n] do
        labels := labels.push batchRaw[j * 3073]!
        labels := labels.push 0
        labels := labels.push 0
        labels := labels.push 0
      raw := raw.append batchRaw
      nTotal := nTotal + n
    let imgs ← F32.cifarBatch raw 0 nTotal.toUSize
    return (imgs, labels, nTotal)
  loadVal := fun dir => do
    let raw ← IO.FS.readBinFile s!"{dir}/cifar-10/test_batch.bin"
    let n := raw.size / 3073
    let mut labels : ByteArray := .empty
    for j in [:n] do
      labels := labels.push raw[j * 3073]!
      labels := labels.push 0
      labels := labels.push 0
      labels := labels.push 0
    let imgs ← F32.cifarBatch raw 0 n.toUSize
    return (imgs, labels, n)
  augmentBatch := fun raw batch seed =>
    F32.randomHFlip raw batch 3 32 32 seed.toUSize

/-- Oxford-IIIT Pets: 224×224 RGB images with 224×224 per-pixel
    masks (3-class trimap: 0=foreground, 1=background, 2=boundary).
    No augmentation in Phase 0 of the UNet demo — mask-aware aug
    is a follow-on. The "labels" buffer in this DatasetIO is the
    mask buffer (224*224 bytes per record, not 4 bytes per record
    like the classification datasets); downstream segmentation
    code reads it as a per-pixel layout. -/
private def petsIO : DatasetIO where
  trainPixels := 3 * 224 * 224
  valPixels   := 3 * 224 * 224
  channels    := 3
  labelBytesPerRecord := 224 * 224
  loadTrain := fun dir => F32.loadPets (dir ++ "/train.bin")
  loadVal   := fun dir => F32.loadPets (dir ++ "/val.bin")
  augmentBatch := fun raw _ _ => return raw

/-- BraTS brain-tumour segmentation (MSD Task01_BrainTumour), 2D axial
    slices. 240×240 images with 4 MRI modalities as channels (FLAIR / T1w /
    T1gd / T2w) and 240×240 per-pixel masks (4-class: 0=background, 1=edema,
    2=non-enhancing tumour, 3=enhancing tumour). Like `petsIO`, the "labels"
    buffer is the mask buffer (240*240 bytes per record), which is what puts
    `runTraining` on the `.perPixelCE` segmentation path.

    240 is the native BraTS in-plane size and 240 = 16*15, so a depth-4 UNet's
    four halvings divide evenly — no resize is needed anywhere in the pipeline.
    Augmentation is identity for now (same starting point as the pets demo). -/
private def bratsIO : DatasetIO where
  trainPixels := 4 * 240 * 240
  valPixels   := 4 * 240 * 240
  channels    := 4
  labelBytesPerRecord := 240 * 240
  loadTrain := fun dir => F32.loadBrats (dir ++ "/train.bin") 240
  loadVal   := fun dir => F32.loadBrats (dir ++ "/val.bin") 240
  augmentBatch := fun raw _ _ => return raw
  -- The three BraTS regions, in MSD's label numbering. Read them off *MSD's*
  -- remap, not the BraTS paper's — MSD permuted the native 1/2/4 to 2/1/3, so
  -- MSD 1 = edema (= BraTS 2) and MSD 2 = non-enhancing/necrotic (= BraTS 1).
  -- Getting this backwards silently swaps TC's contents and is invisible in
  -- the output.
  --   WT (whole tumour)     = everything abnormal
  --   TC (tumour core)      = the resectable core; excludes edema
  --   ET (enhancing tumour) = the surgical target, and the class that collapses
  segRegions := [("WT", [1, 2, 3]), ("TC", [2, 3]), ("ET", [3])]

/-- YOLOv1 detection (Oxford-IIIT Pets). 224×224 RGB images; the "labels"
    buffer carries the 30×7×7 float32 target tensor concatenated with the
    7×7 float32 per-cell objectness mask (6076 bytes per record). The
    `runTraining` dispatch splits this into target + mask before calling
    `trainStepAdamF32Yolov1`. See `preprocess_pets_mosaic.py` for the on-disk
    format and `planning/yolo_final.md` for the recipe. -/
private def petsDetIO : DatasetIO where
  trainPixels := 3 * 224 * 224
  valPixels   := 3 * 224 * 224
  channels    := 3
  -- Phase 3b layout: target (5880) + mask (196) + numBoxes (4) +
  -- raw_boxes (56 × 20 = 1120) = 7200 bytes/record.
  labelBytesPerRecord := 30 * 7 * 7 * 4 + 7 * 7 * 4 + 4 + 56 * 20
  loadTrain := fun dir => F32.loadDetBin (dir ++ "/train.bin")
  loadVal   := fun dir => F32.loadDetBin (dir ++ "/val.bin")
  augmentBatch := fun raw _ _ => return raw

private def datasetIO : DatasetKind → DatasetIO
  | .imagenette => imagenetteIO
  | .mnist      => mnistIO
  | .cifar10    => cifar10IO
  | .pets       => petsIO
  | .petsDet  => petsDetIO
  | .brats      => bratsIO
  | .imagenet   =>
    -- Phase 3 doesn't yet support full 1000-class ImageNet — the 1.28M
    -- training set needs a C-side streaming reader, not the current
    -- read-everything-into-a-ByteArray pattern. Phase 2 (jax/) wires it
    -- through tfds. Until phase 3 streams, this kind is JAX-only.
    panic! "DatasetKind.imagenet not supported by phase 3; use phase 2 (jax/) for now"

/-- Adam + cosine-LR + running-BN-stats training loop, generic over
    `DatasetKind`. The dataset specifies how to load the train/val
    data and what augmentation to apply per batch; everything else
    (init, optimizer, BN EMA, val eval, save) is identical across
    datasets.

    `spec` must have been compiled via `compileVmfbs` first. -/
def runTraining (spec : NetSpec) (cfg : TrainConfig) (ds : DatasetKind)
    (dataDir : String) (sess : IreeSession) : IO Unit := do
  let pfx := spec.buildPrefix
  let batchN : Nat := cfg.batchSize
  let batch  : USize := cfg.batchSize.toUSize
  let dio0 := datasetIO ds
  -- Higher-resolution detection (VisDrone 448/14×14): the det DatasetIO is
  -- hardcoded to 224/7×7 (Pets). When the spec runs at a larger input, override
  -- the record geometry + loader to the dims-parameterized path so the same
  -- .bin format works at any resolution. Pets (imageH=224) is left untouched.
  -- FPN multi-scale total target width Σ_s Aₛ·15·g_s² (0 when not an FPN run).
  let fpnNtot : Nat := (cfg.fpnScales.map (fun sc => sc.2.length * 15 * sc.1 * sc.1)).foldl (·+·) 0
  let dio :=
    if ds == .petsDet && !cfg.fpnScales.isEmpty then
      -- FPN mode (brick #3): the loader returns the flat [P3|P4|P5] target
      -- (ntot f32/record); mask derived from obj channels in the loss.
      { dio0 with
        trainPixels := 3 * spec.imageH * spec.imageW
        valPixels   := 3 * spec.imageH * spec.imageW
        labelBytesPerRecord := fpnNtot * 4
        loadTrain := fun d => F32.loadDetBinFpn (d ++ "/train.bin") spec.imageH.toUSize fpnNtot.toUSize
        loadVal   := fun d => F32.loadDetBinFpn (d ++ "/val.bin") spec.imageH.toUSize fpnNtot.toUSize }
    else if ds == .petsDet && !cfg.anchors.isEmpty then
      -- Anchor mode (brick #2): the loader returns TARGET-ONLY labels
      -- [A·15,gH,gW] (mask derived from target obj channels in the loss).
      let gH := spec.imageH / spec.detStride
      let gW := spec.imageW / spec.detStride
      let A := cfg.anchors.length
      { dio0 with
        trainPixels := 3 * spec.imageH * spec.imageW
        valPixels   := 3 * spec.imageH * spec.imageW
        labelBytesPerRecord := A * 15 * gH * gW * 4
        loadTrain := fun d => F32.loadDetBinAnchor (d ++ "/train.bin") spec.imageH.toUSize gH.toUSize gW.toUSize A.toUSize
        loadVal   := fun d => F32.loadDetBinAnchor (d ++ "/val.bin") spec.imageH.toUSize gH.toUSize gW.toUSize A.toUSize }
    else if ds == .petsDet && spec.imageH != 224 then
      let gH := spec.imageH / spec.detStride
      let gW := spec.imageW / spec.detStride
      let imgSz := spec.imageH.toUSize
      let gHu := gH.toUSize
      let gWu := gW.toUSize
      { dio0 with
        trainPixels := 3 * spec.imageH * spec.imageW
        valPixels   := 3 * spec.imageH * spec.imageW
        labelBytesPerRecord := 30 * gH * gW * 4 + gH * gW * 4 + 4 + 56 * 20
        loadTrain := fun d => F32.loadDetBinDims (d ++ "/train.bin") imgSz gHu gWu
        loadVal   := fun d => F32.loadDetBinDims (d ++ "/val.bin") imgSz gHu gWu }
    else dio0
  -- Derive effective LossKind (planning/yolo_final.md R1). Existing
  -- callers leave cfg.lossKind at the default .classCE and we infer from
  -- the older booleans + the dataset kind:
  --   * petsDet + useYolov1   → yolov1Masked (target+mask f32 batch dispatch)
  --   * label record != 4 bytes → perPixelCE  (segmentation; pets path)
  --   * mixup/cutmix/knnMixup   → softLabelCE
  --   * else                    → classCE (with optional useFocal modifier)
  let useSoftLabelsTop := cfg.useMixup || cfg.useCutmix || cfg.useKnnMixup
  let lossKind : LossKind :=
    if cfg.lossKind != .classCE then cfg.lossKind
    else if cfg.useYolov1 || ds matches .petsDet then .yolov1Masked
    else if dio.labelBytesPerRecord != 4 then .perPixelCE
    else if useSoftLabelsTop then .softLabelCE
    else .classCE
  let useSeg := lossKind.isSeg
  let useFpnRun := !cfg.fpnScales.isEmpty
  let useYolov1Run := lossKind == .yolov1Masked && !useFpnRun

  let (trainImg, trainLbl, nTrain) ← dio.loadTrain dataDir
  IO.eprintln s!"  train: {nTrain} images ({dio.trainPixels} floats/image)"

  -- Phase 4: optional pretrained-backbone bootstrap. Replaces the prefix
  -- of the He-init with bytes read from a saved checkpoint (e.g. R34
  -- Imagenette weights loaded into a YOLOv1 init). Auto-loads BN stats
  -- from the companion file if the path is `*_params.bin`.
  let params ← match cfg.bootstrapBackbone with
    | none => spec.heInitParams
    | some (path, prefixFloats) => do
        IO.eprintln s!"  bootstrap: loading first {prefixFloats} floats from {path}"
        let init ← spec.heInitParams
        NetSpec.patchInitWithPretrainedPrefix init path (prefixFloats * 4)
  -- RetinaNet prior-bias init. Applied AFTER the bootstrap patch (which only
  -- ever rewrites a prefix — the backbone — and never reaches the head) and
  -- BEFORE the checkpoint resume below, which is a full restore and must win.
  let params ← if cfg.headPriorBias.isEmpty then pure params else do
    let p ← NetSpec.applyHeadPriorBias spec params cfg.headPriorBias
    let z := cfg.headPriorBias.map Float.log
    IO.eprintln s!"  head prior-bias init: bias = log π = {z}"
    pure p
  -- LEAN_MLIR_INIT_LOAD: resume from a full checkpoint (overrides any bootstrap
  -- init above). Pairs with LEAN_MLIR_START_STEP below for crash auto-resume on
  -- the IREE/Lean path (e.g. the YOLO segfault wrapper). Adam moments are not
  -- checkpointed, so they restart at zero (re-warm within ~1/(1-β) steps).
  let params ← match (← IO.getEnv "LEAN_MLIR_INIT_LOAD") with
    | none => pure params
    | some ckpt => do
        let loaded ← IO.FS.readBinFile ckpt
        if loaded.size == params.size then
          IO.eprintln s!"  resume: loaded full checkpoint {ckpt} ({loaded.size} bytes)"
          pure loaded
        else
          IO.eprintln s!"  WARN: resume checkpoint {ckpt} size {loaded.size} ≠ expected {params.size}; keeping init"
          pure params
  -- If LEAN_MLIR_INIT_DUMP is set, save the raw init-params buffer to disk.
  -- Used by phase 2 (jax/) to load bit-identical initial parameters for
  -- step-level cross-compiler diffing. See traces/TRACE_FORMAT.md.
  match (← IO.getEnv "LEAN_MLIR_INIT_DUMP") with
  | some path => do
      IO.FS.writeBinFile path params
      IO.eprintln s!"  init-dump  : {path} ({params.size} bytes)"
  | none => pure ()
  let adamM ← F32.const (F32.size params).toUSize 0.0
  let adamV ← F32.const (F32.size params).toUSize 0.0
  IO.eprintln s!"  {F32.size params} params + m + v ({(params.size + adamM.size + adamV.size) / 1024 / 1024} MB)"

  let bpE := nTrain / batchN
  let trainPixels := dio.trainPixels
  let allShapes := spec.shapesBA
  let xSh := spec.xShape batchN
  let nP := spec.totalParams
  let nT := 3 * nP
  let baseLR : Float := cfg.learningRate
  let warmup : Nat := cfg.warmupEpochs
  let epochs : Nat := cfg.epochs
  let nClasses : USize := spec.numClasses.toUSize

  let bnShapes := spec.bnShapesBA
  let nBnStats := spec.nBnStats

  let optName := if cfg.useAdam then "Adam" else "SGD+momentum"
  IO.eprintln s!"training: {bpE} batches/epoch, batch={batchN}, {optName}, lr={baseLR}, cosine warmup={warmup}, label_smooth={cfg.labelSmoothing}, wd={cfg.weightDecay}"
  IO.eprintln s!"  BN layers: {spec.bnLayers.size}, BN stat floats: {nBnStats}"

  -- Trace emission (opt-in via LEAN_MLIR_TRACE_OUT env var).
  -- Writes a JSON Lines file with a header + one record per training
  -- step. See traces/TRACE_FORMAT.md for the contract.
  let traceHandle : Option IO.FS.Handle ←
    match (← IO.getEnv "LEAN_MLIR_TRACE_OUT") with
    | some path => do
        let h ← IO.FS.Handle.mk path .write
        let jsBool := fun (b : Bool) => if b then "true" else "false"
        let dsName : String := match ds with
          | .mnist      => "mnist"
          | .cifar10    => "cifar10"
          | .imagenette => "imagenette"
          | .pets       => "pets"
          | .imagenet   => "imagenet"
          | .petsDet  => "pets_det"
          | .brats      => "brats"
        let hdr :=
          "{\"kind\":\"header\",\"phase\":\"phase3\"" ++
          s!",\"netspec_name\":\"{spec.name}\"" ++
          ",\"config\":{" ++
            s!"\"lr\":{baseLR}" ++
            s!",\"batch_size\":{batchN}" ++
            s!",\"epochs\":{epochs}" ++
            s!",\"use_adam\":{jsBool cfg.useAdam}" ++
            s!",\"weight_decay\":{cfg.weightDecay}" ++
            s!",\"cosine\":{jsBool cfg.cosineDecay}" ++
            s!",\"warmup_epochs\":{warmup}" ++
            s!",\"augment\":{jsBool cfg.augment}" ++
            s!",\"label_smoothing\":{cfg.labelSmoothing}" ++
            s!",\"seed\":{cfg.seed}" ++
          "}" ++
          s!",\"total_params\":{nP}" ++
          s!",\"dataset\":\"{dsName}\"" ++
          ",\"emitter_version\":\"1\"}\n"
        h.putStr hdr
        pure (some h)
    | none => pure none

  let mut p := params
  let mut m := adamM
  let mut v := adamV
  -- Phase 4: bootstrap BN stats from the companion `_bn_stats.bin` if
  -- the pretrained backbone has BN layers matching the spec's. Size
  -- must match exactly (any mismatch == architecture drift between R34
  -- backbone and YOLOv1 backbone, which would be a bug).
  let bnInit : ByteArray ← match cfg.bootstrapBackbone with
    | none => F32.const nBnStats.toUSize 0.0
    | some (path, _) => do
        let bnPath := path.replace "_params.bin" "_bn_stats.bin"
        if ← System.FilePath.pathExists bnPath then
          let bn ← IO.FS.readBinFile bnPath
          let expectedBytes := nBnStats * 4
          if bn.size == expectedBytes then
            IO.eprintln s!"  bootstrap: loaded {bn.size}-byte BN stats from {bnPath}"
            pure bn
          else
            IO.eprintln s!"  WARN: bootstrap BN stats size {bn.size} ≠ expected {expectedBytes}; using zeros"
            F32.const nBnStats.toUSize 0.0
        else
          IO.eprintln s!"  bootstrap: no BN stats file at {bnPath}; using zeros"
          F32.const nBnStats.toUSize 0.0
  -- On resume (LEAN_MLIR_INIT_LOAD), prefer the checkpoint's own BN-stats
  -- companion (`..._params_eN.bin` → `..._bn_stats_eN.bin`) over the bootstrap.
  let bnInit ← match (← IO.getEnv "LEAN_MLIR_INIT_LOAD") with
    | none => pure bnInit
    | some ckpt => do
        let bnPath := ckpt.replace "_params_e" "_bn_stats_e"
        if ← System.FilePath.pathExists bnPath then
          let bn ← IO.FS.readBinFile bnPath
          if bn.size == nBnStats * 4 then
            IO.eprintln s!"  resume: loaded BN stats from {bnPath}"
            pure bn
          else
            IO.eprintln s!"  WARN: resume BN stats size {bn.size} ≠ {nBnStats * 4}; keeping init"
            pure bnInit
        else
          IO.eprintln s!"  resume: no BN stats companion at {bnPath}; keeping init"
          pure bnInit
  let mut runningBnStats : ByteArray := bnInit
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  -- LEAN_MLIR_START_STEP: resume the epoch loop + LR schedule at this global
  -- step (epoch = startStep / batches-per-epoch). Aligns the cosine/warmup LR
  -- and skips already-trained epochs; pairs with LEAN_MLIR_INIT_LOAD.
  let startStep : Nat := ((← IO.getEnv "LEAN_MLIR_START_STEP").bind (·.toNat?)).getD 0
  let startEpoch : Nat := if bpE > 0 then startStep / bpE else 0
  if startEpoch > 0 then
    IO.eprintln s!"  resume: starting at epoch {startEpoch} (step {startEpoch * bpE})"
  let mut globalStep : Nat := startEpoch * bpE
  -- EMA / SWA running averages of θ, used for the final eval checkpoint
  -- when their respective TrainConfig knobs are enabled. Both initialize
  -- to a copy of `params` (so the EMA hasn't moved yet at step 0).
  let mut emaParams : ByteArray := if cfg.useEMA then params else .empty
  let mut swaParams : ByteArray := if cfg.useSWA then params else .empty
  let mut swaCount  : Nat := 0
  -- SWAG: extends SWA with running E[θ²] (for the diagonal of the
  -- weight covariance) plus a circular buffer of the last `swagK`
  -- per-epoch deviations from the SWA mean (the low-rank component).
  let mut swaSqParams   : ByteArray := if cfg.useSWAG then params else .empty
  let mut swagDeviations : Array ByteArray := #[]

  -- Best-by-val checkpoint for the segmentation path. The BraTS runs
  -- *oscillate* — wcesqrt cos pb peaked at WT Dice 0.329 (epoch 3) then drifted
  -- to 0.226 by epoch 10 — and the plain per-N-epoch checkpoint keeps only the
  -- last epoch, so the endpoint is not the best model the run ever produced.
  -- This tracks the best val score seen and saves `{pfx}_best_{params,bn_stats}`
  -- whenever a new best lands. `-1` so the first eval always wins.
  let mut bestSegScore : Float := -1.0

  -- LEAN_MLIR_NO_SHUFFLE=1 disables per-epoch shuffling — used for
  -- phase-2/phase-3 cross-verification where both sides need to see
  -- batches in the same order.
  let skipShuffle := (← IO.getEnv "LEAN_MLIR_NO_SHUFFLE").isSome

  for epoch in [startEpoch:epochs] do
    if !skipShuffle then
      let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPixels.toUSize (epoch + 42).toUSize
      curImg := sImg; curLbl := sLbl

    -- LR is computed PER STEP (see inside the bi loop) so warmup ramps smoothly
    -- from ~0 over warmup*bpE steps rather than jumping to a fraction of peak at
    -- each epoch boundary. The per-epoch jump made Adam's early steps (small v →
    -- ~LR-sized param moves) blow up a hot head LR (headLrMult); per-step warmup
    -- keeps those first steps tiny. `lr` here is just the running value (also
    -- used for the epoch-end log line).
    let mut lr : Float := baseLR

    let mut epochLoss : Float := 0.0
    let t0 ← IO.monoMsNow
    let useSoftLabels := cfg.useMixup || cfg.useCutmix || cfg.useKnnMixup
    let nClassesNat := spec.numClasses
    -- Spatial dims used by mixup/cutmix/RE C kernels. For 4D NCHW
    -- post-augment shapes: imagenette is [b, 3, 224, 224]; cifar
    -- is [b, 3, 32, 32]; mnist is [b, 1, 28, 28]. We derive H = W
    -- from `dio.valPixels / channels` then sqrt — but easier: trust
    -- spec.imageH / imageW for the post-augmentBatch shape (since
    -- for imagenette augmentBatch crops 256→224).
    let augH := spec.imageH
    let augW := spec.imageW
    let augC := dio.channels
    for bi in [:bpE] do
      globalStep := globalStep + 1
      -- Per-step warmup + cosine schedule (keyed off globalStep, resume-safe).
      lr := if cfg.cosineDecay then
              let ws := warmup * bpE
              if ws > 0 && globalStep ≤ ws then
                baseLR * globalStep.toFloat / ws.toFloat
              else
                let denom := (epochs * bpE - ws).toFloat
                let prog := if denom > 0.0 then (globalStep - ws).toFloat / denom else 1.0
                baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * prog))
            else baseLR
      let xbaRaw := F32.sliceImages curImg (bi * batchN) batchN trainPixels
      let xbaInit ← if cfg.augment then dio.augmentBatch xbaRaw batch (epoch * 10000 + bi)
                                   else dio.preprocessBatch xbaRaw batch
      let mut xba : ByteArray := xbaInit
      let yb := F32.sliceLabels curLbl (bi * batchN) batchN dio.labelBytesPerRecord
      let stepSeed : USize := (epoch * 100000 + bi).toUSize
      let mut yArg : ByteArray := yb
      -- Segmentation aug: paired hflip of image+mask (one coin per image), so
      -- the per-pixel correspondence is preserved. The classification aug
      -- pack below (RandAugment / mixup / cutmix) does not apply to per-pixel
      -- masks and stays gated by its own flags; this is the seg path's only
      -- geometric aug. Guarded by useSeg so no non-seg trainer is touched.
      if useSeg && cfg.augment then
        let (xf, mf) ← F32.segHflipPair xba yb batchN.toUSize augC.toUSize
                          augH.toUSize augW.toUSize (stepSeed ^^^ (11 : USize))
        xba := xf
        yArg := mf
      -- DeiT-style aug. RandAugment first (color-only subset), then RE.
      if cfg.useRandAugment then
        let raSeed : USize := stepSeed ^^^ (5 : USize)
        -- ImageNet-normalized inputs need a de-norm/re-norm round-trip
        -- around the color ops (which assume [0,1] sRGB). For CIFAR /
        -- MNIST images already in [0,1], pass 0 to skip the round-trip.
        let imagenetNorm : USize := match ds with | .imagenette => 1 | _ => 0
        xba ← F32.randAugment xba batch augC.toUSize augH.toUSize augW.toUSize cfg.randAugmentN.toUSize cfg.randAugmentM imagenetNorm raSeed
      if cfg.randomErasing then
        xba ← F32.randomErasing xba batch augC.toUSize augH.toUSize augW.toUSize cfg.randomErasingProb stepSeed
      -- Mixup XOR CutMix XOR KNN-Mixup (paper convention: pick one per batch).
      -- All three emit soft labels, switching the train-step path below.
      -- KNN takes precedence over plain Mixup if both are set.
      if cfg.useKnnMixup then
        let mixSeed : USize := stepSeed ^^^ (3 : USize)
        let xbaPre := xba
        xba ← F32.knnMixupImages xbaPre batch augC.toUSize augH.toUSize augW.toUSize cfg.knnMixupAlpha mixSeed
        yArg ← F32.knnMixupSoftLabels yb xbaPre batch nClassesNat.toUSize augC.toUSize augH.toUSize augW.toUSize cfg.knnMixupAlpha cfg.labelSmoothing mixSeed
      else if cfg.useMixup then
        let mixSeed : USize := stepSeed ^^^ (1 : USize)
        xba ← F32.mixupImages xba batch augC.toUSize augH.toUSize augW.toUSize cfg.mixupAlpha mixSeed
        yArg ← F32.mixupSoftLabels yb batch nClassesNat.toUSize cfg.mixupAlpha cfg.labelSmoothing mixSeed
      else if cfg.useCutmix then
        let mixSeed : USize := stepSeed ^^^ (2 : USize)
        xba ← F32.cutmixImages xba batch augC.toUSize augH.toUSize augW.toUSize cfg.cutmixAlpha mixSeed
        yArg ← F32.cutmixSoftLabels yb batch nClassesNat.toUSize augH.toUSize augW.toUSize cfg.cutmixAlpha cfg.labelSmoothing mixSeed
      let packed := (p.append m).append v
      let ts0 ← IO.monoMsNow
      let out ← if useFpnRun then do
                  -- FPN (brick #3): one flat target [B, Ntot] (loader gives it
                  -- target-only). The FPN codegen sig is x + %y_fpn:[B,Ntot,1,1] +
                  -- lr + t — identical to the single-target DDPM protocol — so
                  -- reuse that FFI verbatim (outC=Ntot, outH=outW=1), no mask.
                  IreeSession.trainStepAdamF32Ddpm sess spec.trainFnName
                    packed allShapes xba xSh yArg lr globalStep.toFloat bnShapes batch
                    fpnNtot.toUSize 1 1
                else if useYolov1Run then do
                  let gHu := (spec.imageH / spec.detStride).toUSize
                  let gWu := (spec.imageW / spec.detStride).toUSize
                  if !cfg.anchors.isEmpty then do
                    -- Anchor path (brick #2): yArg IS the target [B, A·15, gH, gW]
                    -- (target-only loader); the mask is derived from the target's
                    -- obj channels in the loss, so pass a zero dummy mask. No augment
                    -- (yoloAugment is single-box-format only).
                    let A := cfg.anchors.length
                    let yMsk ← F32.const (batch * gHu * gWu) 0.0
                    IreeSession.trainStepAdamF32Yolov1 sess spec.trainFnName
                      packed allShapes xba xSh yArg yMsk lr globalStep.toFloat bnShapes batch
                      gHu gWu (A * 15).toUSize
                  else do
                  -- Single-box: yArg per record is target (5880) + mask (196) +
                  -- numBoxes (4) + raw_boxes (56×20) = 7200 bytes (Phase 3b format).
                  --   * cfg.augment := false → split into target+mask.
                  --   * cfg.augment := true  → yoloAugment (bbox-aware hflip+crop).
                  let (xAug, yTgtAug, yMskAug) ← if cfg.augment then
                        F32.yoloAugment xba yArg batch augC.toUSize
                          augH.toUSize augW.toUSize
                          gHu gWu (30 : USize) (20 : USize)
                          0.5 0.5 0.8 (stepSeed ^^^ (11 : USize))
                      else do
                        let (yTgt, yMsk) ← F32.detSplitBatch yArg batch
                        pure (xba, yTgt, yMsk)
                  IreeSession.trainStepAdamF32Yolov1 sess spec.trainFnName
                    packed allShapes xAug xSh yTgtAug yMskAug lr globalStep.toFloat bnShapes batch
                    gHu gWu (30 : USize)
                else if useSeg then do
                  -- Pets `loadPets` returns uint8 masks; the seg train step
                  -- expects int32 LE [B, H, W]. yArg here is the raw uint8
                  -- batch slice (sliceLabels with bytesPerRecord = H*W).
                  let yI32 ← F32.maskU8ToI32 yArg
                  IreeSession.trainStepAdamF32Seg sess spec.trainFnName
                    packed allShapes xba xSh yI32 lr globalStep.toFloat bnShapes batch
                    spec.imageH.toUSize spec.imageW.toUSize
                else if useSoftLabels then
                  IreeSession.trainStepAdamF32Soft sess spec.trainFnName
                    packed allShapes xba xSh yArg lr globalStep.toFloat bnShapes batch nClassesNat.toUSize
                else
                  IreeSession.trainStepAdamF32 sess spec.trainFnName
                    packed allShapes xba xSh yArg lr globalStep.toFloat bnShapes batch
      let ts1 ← IO.monoMsNow
      let loss := F32.extractLoss out nT
      epochLoss := epochLoss + loss
      p := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      -- EMA: per-step `ema = decay·ema + (1-decay)·θ`. F32.ema's signature
      -- is `mom·new + (1-mom)·running`, so pass mom = 1-decay.
      if cfg.useEMA then
        emaParams ← F32.ema emaParams p (1.0 - cfg.emaDecay)
      if let some h := traceHandle then
        let line :=
          "{\"kind\":\"step\"" ++
          s!",\"step\":{globalStep}" ++
          s!",\"epoch\":{epoch}" ++
          s!",\"loss\":{loss}" ++
          s!",\"lr\":{lr}" ++
          "}\n"
        h.putStr line
      let batchBnStats := out.extract ((nT + 1) * 4) ((nT + 1 + nBnStats) * 4)
      let bnMom : Float := if globalStep == 1 then 1.0 else 0.1
      runningBnStats ← F32.ema runningBnStats batchBnStats bnMom
      if bi < 3 || bi % 100 == 0 then
        IO.eprintln s!"  step {bi}/{bpE}: loss={loss} ({ts1-ts0}ms)"
    let t1 ← IO.monoMsNow
    let avgLoss := epochLoss / bpE.toFloat
    IO.eprintln s!"Epoch {epoch+1}/{epochs}: loss={avgLoss} lr={lr} ({t1-t0}ms)"

    -- SWA: equal-weight average of params from epochs ≥ swaStartEpoch.
    -- Update once per epoch boundary using running-average semantics:
    --   swa_θ = (n·swa_θ + θ) / (n+1) = ema(swa_θ, θ, 1/(n+1)).
    if cfg.useSWA && epoch >= cfg.swaStartEpoch then
      let mom : Float := 1.0 / (swaCount.toFloat + 1.0)
      swaParams ← F32.ema swaParams p mom
      if cfg.useSWAG then
        -- E[θ²] for the diagonal of the SWAG covariance.
        swaSqParams ← F32.emaSq swaSqParams p mom
        -- Push the deviation `p − swaParams` into the circular buffer.
        let dev ← F32.subtract p swaParams
        if swagDeviations.size >= cfg.swagK then
          swagDeviations := swagDeviations.eraseIdx! 0
        swagDeviations := swagDeviations.push dev
      swaCount := swaCount + 1

    -- Per-N-epoch checkpoint (planning/yolo_final.md Phase 4
    -- infrastructure). Writes params + BN stats with the epoch number
    -- in the filename so a killed training run can pick up from the
    -- last save, OR downstream tasks (e.g. YOLOv1 bootstrap) can borrow
    -- a mid-training checkpoint without waiting for the full schedule.
    if cfg.checkpointEveryNEpochs > 0 && (epoch + 1) % cfg.checkpointEveryNEpochs == 0 then
      IO.FS.writeBinFile s!"{pfx}_params_e{epoch + 1}.bin" p
      IO.FS.writeBinFile s!"{pfx}_bn_stats_e{epoch + 1}.bin" runningBnStats
      IO.eprintln s!"  checkpoint: wrote {pfx}_params_e{epoch + 1}.bin + bn_stats_e{epoch + 1}.bin"

    -- Guard the modulus: `evalEveryNEpochs := 0` means "final epoch only", not
    -- a division by zero.
    let evalNow :=
      (cfg.evalEveryNEpochs > 0 && (epoch + 1) % cfg.evalEveryNEpochs == 0)
        || epoch + 1 == epochs
    if evalNow then
     if useSeg then
       -- Per-class IoU + mIoU over the val set (planning/unet_demo_v2.md
       -- Workstream A). Eval-forward → argmax over the NC channels →
       -- confusion matrix accumulated in exact Nat across batches →
       -- IoU_c = conf[c][c] / (row_c + col_c − conf[c][c]).
       let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
       if ← System.FilePath.pathExists evalVmfb then
         let evalSess ← IreeSession.create evalVmfb
         let (valImg, valLbl, nVal) ← dio.loadVal dataDir
         let evalBatch := batchN
         let evalSteps := nVal / evalBatch
         let evalXSh := spec.xShape evalBatch
         let evalShapesBA := spec.evalShapesBA
         let evalParams := p.append runningBnStats
         let H := spec.imageH; let W := spec.imageW
         let plane := H * W
         let NC := (spec.layers.getLast?.map (·.outChannels)).getD 3
         let outElems : USize := (NC * plane).toUSize
         let readI64 : ByteArray → Nat → Nat := fun ba idx => Id.run do
           let mut v : Nat := 0
           for k in [:8] do
             v := v + ba.data[idx * 8 + k]!.toNat * (2 ^ (8 * k))
           return v
         let mut conf : Array Nat := Array.replicate (NC * NC) 0
         for bi in [:evalSteps] do
           let xba := F32.sliceImages valImg (bi * evalBatch) evalBatch dio.valPixels
           let logits ← IreeSession.forwardF32 evalSess spec.evalFnName
                           evalParams evalShapesBA xba evalXSh evalBatch.toUSize outElems
           let maskSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch dio.labelBytesPerRecord
           let cb ← F32.segConfusion logits maskSlice
                       evalBatch.toUSize NC.toUSize H.toUSize W.toUSize
           for j in [:NC * NC] do
             conf := conf.set! j (conf[j]! + readI64 cb j)
         -- Per-class IoU + mean.
         let mut ious : Array Float := #[]
         for c in [:NC] do
           let tp := conf[c * NC + c]!
           let mut row : Nat := 0
           let mut col : Nat := 0
           for k in [:NC] do
             row := row + conf[c * NC + k]!
             col := col + conf[k * NC + c]!
           let uni := row + col - tp
           let iou := if uni == 0 then 0.0 else tp.toFloat / uni.toFloat
           ious := ious.push iou
         let miou := (ious.foldl (· + ·) 0.0) / NC.toFloat
         let iouStr := String.intercalate " " (ious.toList.mapIdx fun i v => s!"c{i}={v}")
         IO.eprintln s!"  val mIoU: {miou}  (per-class: {iouStr})"
         -- Dice on named class-unions (planning/brats_demo.md Workstream F).
         -- Falls straight out of the confusion matrix already accumulated
         -- above — no new kernel, no second pass over val. For a region
         -- R ⊆ classes, reading C[gt][pred]:
         --   inter_R = Σ_{g∈R} Σ_{p∈R} C[g][p]
         --   |gt_R|  = Σ_{g∈R} Σ_p C[g][p]
         --   |pr_R|  = Σ_{p∈R} Σ_g C[g][p]
         --   Dice_R  = 2·inter_R / (|gt_R| + |pr_R|)
         -- Counts stay exact `Nat` until the final divide. IoU is kept
         -- alongside rather than replaced: it is what makes this demo
         -- comparable to the pets one, while Dice is what makes it
         -- comparable to the BraTS literature.
         let mut regionDices : Array Float := #[]
         for (name, cls) in dio.segRegions do
           let inR : Nat → Bool := fun c => cls.contains c
           let mut inter : Nat := 0
           let mut gtR : Nat := 0
           let mut prR : Nat := 0
           for g in [:NC] do
             for pd in [:NC] do
               let cij := conf[g * NC + pd]!
               if inR g && inR pd then inter := inter + cij
               if inR g then gtR := gtR + cij
               if inR pd then prR := prR + cij
           let den := gtR + prR
           -- den = 0 ⇒ the region is absent from ground truth *and* from the
           -- prediction across the whole val set: vacuously perfect, which is
           -- the field's convention. It does not arise on real BraTS val —
           -- a collapsed model still has |gt_R| > 0 and so scores 0, which is
           -- the reading we want.
           let dice := if den == 0 then 1.0 else (2 * inter).toFloat / den.toFloat
           regionDices := regionDices.push dice
           IO.eprintln s!"  val Dice {name}: {dice}  (inter={inter} gt={gtR} pred={prR})"
         -- Best-by-val checkpoint. Selection metric: mean region Dice when the
         -- dataset defines regions (BraTS — this is its headline number and
         -- captures "found the tumour"), else mean of the non-background class
         -- IoUs (a general seg proxy that is ~0 for the trivial predictor).
         -- Saving the whole-tumour region overlap rather than mean-tumour-IoU
         -- is deliberate: on the oscillating BraTS run the two disagree, and WT
         -- Dice is the one the field reports.
         let segScore : Float :=
           if regionDices.isEmpty then
             (((List.range NC).drop 1).foldl (fun a c => a + ious[c]!) 0.0)
               / (max 1 (NC - 1)).toFloat
           else
             (regionDices.foldl (· + ·) 0.0) / regionDices.size.toFloat
         if segScore > bestSegScore then
           bestSegScore := segScore
           IO.FS.writeBinFile s!"{pfx}_best_params.bin" p
           IO.FS.writeBinFile s!"{pfx}_best_bn_stats.bin" runningBnStats
           IO.eprintln s!"  ✓ new best (score {segScore}) — saved {pfx}_best_*"
     else if useYolov1Run || useFpnRun then
       -- mAP@0.5 eval for detection is a separate offline pass (inferDump →
       -- scripts/yolo_map_visdrone.py). The train step runs + loss drops on real
       -- detection data; the classification eval block below would interpret the
       -- flat detection output as class logits, which is nonsensical.
       IO.eprintln s!"  ({if useFpnRun then "fpn" else "yolov1"} eval skipped — offline mAP pass; train loss above is the signal)"
     else
      let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
      if ← System.FilePath.pathExists evalVmfb then
        let evalSess ← IreeSession.create evalVmfb
        let (valImg, valLbl, nVal) ← dio.loadVal dataDir
        let evalBatch := batchN
        let evalSteps := nVal / evalBatch
        let evalXSh := spec.xShape evalBatch
        let evalShapesBA := spec.evalShapesBA
        -- EMA gets priority over SWA when both are enabled (DeiT default).
        -- For models with BN, the running-stat block at the end of
        -- evalParams is a stale-vs-averaged-weights mismatch; for now we
        -- accept that, and recommend BN-free architectures for EMA/SWA
        -- experiments. A future BN-recompute pass would fix this.
        let evalLabel : String :=
          if cfg.useEMA then "EMA"
          else if cfg.useSWA && epoch + 1 == epochs then "SWA"
          else "running BN"
        let evalWeights : ByteArray :=
          if cfg.useEMA then emaParams
          else if cfg.useSWA && epoch + 1 == epochs then swaParams
          else p
        let evalParams := evalWeights.append runningBnStats
        let mut correct : Nat := 0
        let mut total : Nat := 0
        let ttaM : Nat := if cfg.useTTA then cfg.ttaSamples else 1
        let evalLabel2 := if cfg.useTTA then s!"{evalLabel}+TTA{ttaM}" else evalLabel
        for bi in [:evalSteps] do
          let xbaRaw := F32.sliceImages valImg (bi * evalBatch) evalBatch dio.valPixels
          -- TTA: average logits over M independently-augmented passes.
          -- M=1 (the !cfg.useTTA path) reduces to a single deterministic
          -- preprocess + forward, matching the prior behavior exactly.
          let mut logitsAcc ← F32.const (evalBatch * spec.numClasses).toUSize 0.0
          for k in [:ttaM] do
            let xba ← if cfg.useTTA then
                        dio.augmentBatch xbaRaw evalBatch.toUSize (epoch * 100000 + bi * 31 + k)
                      else
                        dio.valPreprocessBatch xbaRaw evalBatch.toUSize
            let logits ← IreeSession.forwardF32 evalSess spec.evalFnName
                            evalParams evalShapesBA xba evalXSh evalBatch.toUSize nClasses
            let mom : Float := 1.0 / (k.toFloat + 1.0)
            logitsAcc ← F32.ema logitsAcc logits mom
          let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch dio.labelBytesPerRecord
          for i in [:evalBatch] do
            let pred := F32.argmax10 logitsAcc (i * spec.numClasses).toUSize
            let label := lblSlice.data[i * 4]!.toNat
            if pred.toNat == label then correct := correct + 1
            total := total + 1
        let acc := correct.toFloat / total.toFloat * 100.0
        IO.eprintln s!"  val accuracy ({evalLabel2}): {correct}/{total} = {acc}%"

  -- SWAG sample-and-average eval. Runs *after* the regular eval at
  -- training's end so the comparison vs single-checkpoint accuracy is
  -- visible side-by-side. Each sample draws θ_s ~ N(swaMean, Σ_SWAG)
  -- from `swagSample`, replaces the param block, and runs forward;
  -- logits are averaged per batch across `swagSamples`. Caveat: BN
  -- running stats are stale relative to sampled weights — the same
  -- caveat as EMA/SWA's eval block.
  if cfg.useSWAG && swagDeviations.size > 0 then
    let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
    if ← System.FilePath.pathExists evalVmfb then
      let evalSess ← IreeSession.create evalVmfb
      let (valImg, valLbl, nVal) ← dio.loadVal dataDir
      let evalBatch := batchN
      let evalSteps := nVal / evalBatch
      let evalXSh := spec.xShape evalBatch
      let evalShapesBA := spec.evalShapesBA
      let concatDev := F32.concat swagDeviations
      let nPUSize : USize := nP.toUSize
      let kUSize : USize := swagDeviations.size.toUSize
      let mut logitsAccs : Array ByteArray := #[]
      for _ in [:evalSteps] do
        logitsAccs := logitsAccs.push (← F32.const (evalBatch * spec.numClasses).toUSize 0.0)
      for k in [:cfg.swagSamples] do
        let sampledWeights ← F32.swagSample swaParams swaSqParams concatDev nPUSize kUSize (k * 7919 + 13).toUSize
        let sampledEvalParams := sampledWeights.append runningBnStats
        for bi in [:evalSteps] do
          let xbaRaw := F32.sliceImages valImg (bi * evalBatch) evalBatch dio.valPixels
          let xba ← dio.valPreprocessBatch xbaRaw evalBatch.toUSize
          let logits ← IreeSession.forwardF32 evalSess spec.evalFnName
                          sampledEvalParams evalShapesBA xba evalXSh evalBatch.toUSize nClasses
          let mom : Float := 1.0 / (k.toFloat + 1.0)
          logitsAccs := logitsAccs.set! bi (← F32.ema logitsAccs[bi]! logits mom)
      let mut correct : Nat := 0
      let mut total : Nat := 0
      for bi in [:evalSteps] do
        let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch dio.labelBytesPerRecord
        for i in [:evalBatch] do
          let pred := F32.argmax10 logitsAccs[bi]! (i * spec.numClasses).toUSize
          let label := lblSlice.data[i * 4]!.toNat
          if pred.toNat == label then correct := correct + 1
          total := total + 1
      let acc := correct.toFloat / total.toFloat * 100.0
      IO.eprintln s!"  val accuracy (SWAG×{cfg.swagSamples}): {correct}/{total} = {acc}%"

  IO.FS.writeBinFile s!"{pfx}_params.bin" p
  IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" runningBnStats
  if cfg.useEMA then
    IO.FS.writeBinFile s!"{pfx}_ema_params.bin" emaParams
  if cfg.useSWA then
    IO.FS.writeBinFile s!"{pfx}_swa_params.bin" swaParams
  if cfg.useSWAG && swagDeviations.size > 0 then
    IO.FS.writeBinFile s!"{pfx}_swa_sq_params.bin" swaSqParams
    IO.FS.writeBinFile s!"{pfx}_swag_deviations.bin" (F32.concat swagDeviations)
  IO.eprintln "Saved params + BN stats."

/-- Eval-only mode: skip training entirely, load saved params + bn_stats,
    run the eval forward on val data, print accuracy. Used to re-eval
    existing checkpoints against a fixed eval pipeline (e.g. after the
    Imagenette `centerCrop` bug was fixed). Requires `compileVmfbs` to
    have produced (or cached) the `_fwd_eval.vmfb` already. -/
def evalOnly (spec : NetSpec) (cfg : TrainConfig) (ds : DatasetKind)
    (dataDir : String) : IO Unit := do
  let pfx := spec.buildPrefix
  let dio := datasetIO ds
  if dio.labelBytesPerRecord != 4 then
    IO.eprintln "evalOnly is classification-only (argmax vs int32 label). Seg eval (mIoU) is Phase 2 of the UNet demo."
    return ()

  let paramsPath := s!"{pfx}_params.bin"
  let bnPath := s!"{pfx}_bn_stats.bin"
  if !(← System.FilePath.pathExists paramsPath) then
    IO.eprintln s!"ERROR: no saved params at {paramsPath}"
    IO.eprintln "  (train this cell once, or check that pfx matches its saved files)"
    IO.Process.exit 1
  IO.eprintln s!"loading {paramsPath}..."
  let params ← IO.FS.readBinFile paramsPath
  let runningBnStats ← if (← System.FilePath.pathExists bnPath)
                       then IO.FS.readBinFile bnPath
                       else pure ByteArray.empty
  let evalParams := params.append runningBnStats

  let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
  if !(← System.FilePath.pathExists evalVmfb) then
    IO.eprintln s!"ERROR: no eval vmfb at {evalVmfb}"
    IO.eprintln "  (run a training cycle first to compile, or call compileVmfbs)"
    IO.Process.exit 1
  let evalSess ← IreeSession.create evalVmfb
  IO.eprintln "  eval session loaded"

  let (valImg, valLbl, nVal) ← dio.loadVal dataDir
  let evalBatch : Nat := cfg.batchSize
  let evalSteps := nVal / evalBatch
  let evalXSh := spec.xShape evalBatch
  let evalShapesBA := spec.evalShapesBA
  let nClasses := spec.numClasses.toUSize

  let mut correct : Nat := 0
  let mut total : Nat := 0
  for bi in [:evalSteps] do
    let xbaRaw := F32.sliceImages valImg (bi * evalBatch) evalBatch dio.valPixels
    let xba ← dio.valPreprocessBatch xbaRaw evalBatch.toUSize
    let logits ← IreeSession.forwardF32 evalSess spec.evalFnName
                    evalParams evalShapesBA xba evalXSh evalBatch.toUSize nClasses
    let lblSlice := F32.sliceLabels valLbl (bi * evalBatch) evalBatch dio.labelBytesPerRecord
    for i in [:evalBatch] do
      let pred := F32.argmax10 logits (i * spec.numClasses).toUSize
      let label := lblSlice.data[i * 4]!.toNat
      if pred.toNat == label then correct := correct + 1
      total := total + 1
  let acc := correct.toFloat / total.toFloat * 100.0
  IO.eprintln s!"EVAL ONLY  {spec.name}: {correct}/{total} = {acc}%"

/-- End-to-end: compile all three vmfbs, load the train-step session,
    and run the training loop on the chosen dataset. The high-level
    entry point that every Main*Train.lean now calls.

    `LEAN_MLIR_EVAL_ONLY=1` short-circuits to `evalOnly` — skips
    training, loads the saved checkpoint, runs eval. Used for re-eval
    after a fixed eval pipeline.

    Defaults to Imagenette so existing trainers don't have to change. -/
def train (spec : NetSpec) (cfg : TrainConfig) (dataDir : String)
    (ds : DatasetKind := .imagenette) : IO Unit := do
  IO.eprintln s!"{spec.name}: {spec.totalParams} params"
  -- Match runTraining's lossKind derivation so the codegen flag aligns
  -- with the dispatch path. The per-pixel kinds (CE / Dice / DiceCE) all set
  -- `useSeg` — see `LossKind.isSeg`. YOLOv1 has a non-4-byte label record too
  -- but routes through the yolov1 codegen path, not the seg one.
  let useSoftLabelsTop := cfg.useMixup || cfg.useCutmix || cfg.useKnnMixup
  let lossKind : LossKind :=
    if cfg.lossKind != .classCE then cfg.lossKind
    else if cfg.useYolov1 || ds matches .petsDet then .yolov1Masked
    else if (datasetIO ds).labelBytesPerRecord != 4 then .perPixelCE
    else if useSoftLabelsTop then .softLabelCE
    else .classCE
  let useSeg := lossKind.isSeg
  match (← IO.getEnv "LEAN_MLIR_EVAL_ONLY") with
  | some _ =>
    -- compileVmfbs is cheap when cached and produces the eval vmfb we need.
    let _ ← spec.compileVmfbs cfg useSeg
    spec.evalOnly cfg ds dataDir
  | none => do
    let trainVmfb ← spec.compileVmfbs cfg useSeg
    let sess ← IreeSession.create trainVmfb
    IO.eprintln "  session loaded"
    spec.runTraining cfg ds dataDir sess

end NetSpec
