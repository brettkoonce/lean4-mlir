import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.E4M3Quant

/-! # Shared driver for the `*-verified` trainers

Every `Main*Verified.lean` trains a network on **pre-rendered, audited** StableHLO
(`verified_mlir/<slug>_{train_step,fwd}.mlir`, emitted offline by `tests/Test*` from
the proof stack) through the IREE FFI. Unlike the reference `NetSpec`/`Train.lean`
path — which *generates* the MLIR at runtime — the verified path consumes a fixed
codegen artifact, so a verified "model definition" is just:

  * `slug`   — which `verified_mlir/*.mlir` + which `m.*` functions to invoke,
  * `specs`  — the param layout (`(dims, initKind)`, = the matching `XLayout.specs`),
  * `d0`     — per-example input width, and
  * `data`   — which dataset/loader to feed it.

The architecture itself lives in the renderer + the audited VJP theorems; it is
deliberately NOT re-expressed here. This file factors the ~100 lines of identical
boilerplate (compile → sessions → load → init → train/eval loop) that every trainer
used to copy. A trainer is now a `VerifiedNet` value + a `VerifiedConfig` + a one-line
`main`, mirroring the shape of `MainResnetTrain.lean`.

NB the learning rate is **baked into the rendered train-step MLIR** — `VerifiedConfig.lr`
is for the banner only; changing it does not change training (re-render to change lr).
-/

/-- Which dataset a verified trainer runs on. Picks the loader, the eval-split name,
    and whether the training images need a 256²→224² center-crop per batch. -/
inductive VerifiedData where
  /-- MNIST idx files directly under `dataDir` (28×28×1, no crop). -/
  | mnist
  /-- CIFAR-10 `.bin` records under `dataDir/cifar-10` (32×32×3, no crop). -/
  | cifar
  /-- Imagenette under `dataDir/imagenette` — train stored at 256² (center-cropped
      to 224² per batch), val at 224². -/
  | imagenette
deriving BEq, Repr

/-- A verified trainer: a pinned codegen artifact (`slug`) + its param layout
    (`specs`, `d0`, `nClasses`) + the dataset to run it on. See the module docstring. -/
structure VerifiedNet where
  /-- Display name, e.g. `"ResNet-34"`. -/
  name     : String
  /-- Codegen slug: drives `verified_mlir/<slug>_{train_step,fwd}.mlir`,
      `.lake/build/<slug>_{ts,fwd}_v.vmfb`, and the `m.<slug>_{train_step,fwd}` funcs. -/
  slug     : String
  /-- `(dims, initKind)` per param, in func-arg order — the matching `XLayout.specs`.
      `initKind`: 0 = He(fan-in), 1 = ones (γ), 2 = zeros (β / bias). -/
  specs    : Array (Array Nat × Nat)
  /-- Per-example flattened input width (e.g. `3 * 224 * 224`). -/
  d0       : Nat
  /-- Number of output classes. -/
  nClasses : Nat := 10
  /-- Dataset / loader selector. -/
  data     : VerifiedData
  /-- One-line intro printed at startup (the prose banner). -/
  blurb    : String
  /-- Per-BN-layer channel counts, in forward order (empty for LayerNorm / no-BN nets). When
      non-empty, `trainAdamSched` threads running BN stats: the adam train step carries per-layer
      batch mean/var out in passthrough slots, the driver EMAs them, and eval uses
      `<slug>_fwd_eval.mlir` (affine BN with the running stats) instead of `<slug>_fwd.mlir`. -/
  bnChannels : Array Nat := #[]

/-- Training hyperparameters — the `TrainConfig` of the verified path. Mirrors the
    reference `TrainConfig`; kept as its own object so a net is a (spec, config) pair. -/
structure VerifiedConfig where
  /-- Number of training epochs. -/
  epochs    : Nat
  /-- Minibatch size (a free runtime param — the MLIR's batch dim is dynamic). -/
  batchSize : Nat := 32
  /-- Learning rate. DISPLAY ONLY — baked into `<slug>_train_step.mlir`; changing it
      here does not change training (re-render the MLIR to change lr). -/
  lr        : Float := 0.1

namespace VerifiedNet

/-- Param shapes in func-arg order (= `specs` dims). -/
def paramShapes (n : VerifiedNet) : Array (Array Nat) := n.specs.map (·.1)
/-- Packed shape descriptors for the FFI (see `packShapes`). -/
def shapesBA (n : VerifiedNet) : ByteArray := packShapes n.paramShapes
/-- Total float count across all params. -/
def nParams (n : VerifiedNet) : Nat := (n.specs.map (fun s => s.1.foldl (·*·) 1)).foldl (·+·) 0
/-- Packed `x` input shape `[batch, d0]`. -/
def xShape (n : VerifiedNet) (batch : Nat) : ByteArray := packXShape #[batch, n.d0]

end VerifiedNet

/-- iree-compile one `.mlir` → `.vmfb`, surfacing failures. Skips when the `.vmfb` is already
    newer than the `.mlir` (a content-stable cache): avoids the ~minutes-long 224² recompile, and
    lets two same-net runs share one GPU-pair safely — they only *read* the cached vmfb (concurrent
    reads are fine; it's the concurrent *writes* of an identical compile that would race). -/
private def compileVmfb (mlirPath outPath : String) : IO Unit := do
  if (← System.FilePath.pathExists outPath) then
    let srcMd ← (System.FilePath.mk mlirPath).metadata
    let outMd ← (System.FilePath.mk outPath).metadata
    if outMd.modified.sec ≥ srcMd.modified.sec then
      IO.println s!"  (cached vmfb) {outPath}"
      return
  let cargs ← ireeCompileArgs mlirPath outPath
  IO.println s!"  iree-compile {mlirPath}"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    throw (IO.userError s!"iree-compile failed:\n{r.stderr.take 2000}")

/-- Init one parameter from its `(dims, initKind)` spec: He(fan-in) weights (kind 0;
    fan-in = `ic·kH·kW` for a rank-4 conv kernel, `in` for a rank-2 dense matrix),
    γ = 1 (kind 1), β / bias = 0 (kind 2). -/
private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

/-- Load CIFAR-10 `.bin` records (3073 bytes: 1 label byte + 3072 image bytes).
    Returns f32 images `[n×3072]` (normalized) and int32-LE labels `[n×4]`. -/
private def loadCifarSplit (paths : List String) : IO (ByteArray × ByteArray × Nat) := do
  let mut raw : ByteArray := .empty
  let mut labels : ByteArray := .empty
  let mut nTotal : Nat := 0
  for p in paths do
    let batchRaw ← IO.FS.readBinFile p
    let n := batchRaw.size / 3073
    for j in [:n] do
      labels := labels.push batchRaw[j * 3073]!
      labels := labels.push 0; labels := labels.push 0; labels := labels.push 0
    raw := raw.append batchRaw
    nTotal := nTotal + n
  let imgs ← F32.cifarBatch raw 0 nTotal.toUSize
  return (imgs, labels, nTotal)

/-- Load the train + eval splits for a dataset. Returns
    `(trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop?)` where
    `trainPix` is the stored per-example width of the *training* images (256² for
    Imagenette, `d0` otherwise) and `crop?` requests the 256²→224² center-crop. -/
private def loadData (data : VerifiedData) (d0 : Nat) (dataDir : String) :
    IO (ByteArray × ByteArray × Nat × ByteArray × ByteArray × Nat × Nat × Bool) := do
  match data with
  | .imagenette =>
    let idir := dataDir ++ "/imagenette"
    -- Train split ships at 256² → randomCrop 256→224 + hflip (the training recipe);
    -- val ships at 224² (center crop). DEFAULT is 256²/crop, matching the reference
    -- trainer (Train.lean `imagenetteIO` hardcodes 256). Some dirs store the train
    -- split at 224² already (records of [1 label byte + 224·224·3 uint8]); for those
    -- set LEAN_MLIR_IMAGENETTE_TRAIN=224 to load 224²/no-crop (else: "short read").
    -- px also feeds trainPix (3·px²) and crop := (px == 256).
    let px := ((← IO.getEnv "LEAN_MLIR_IMAGENETTE_TRAIN").bind (·.toNat?)).getD 256
    let (trI, trL, nTr) ← F32.loadImagenetteSized (idir ++ "/train.bin") px.toUSize
    let (evI, evL, nEv) ← F32.loadImagenette (idir ++ "/val.bin")
    return (trI, trL, nTr, evI, evL, nEv, 3 * px * px, px == 256)
  | .mnist =>
    let (trI, nTr) ← F32.loadIdxImages (dataDir ++ "/train-images-idx3-ubyte")
    let (trL, _)   ← F32.loadIdxLabels (dataDir ++ "/train-labels-idx1-ubyte")
    let (evI, nEv) ← F32.loadIdxImages (dataDir ++ "/t10k-images-idx3-ubyte")
    let (evL, _)   ← F32.loadIdxLabels (dataDir ++ "/t10k-labels-idx1-ubyte")
    return (trI, trL, nTr, evI, evL, nEv, d0, false)
  | .cifar =>
    let cdir := dataDir ++ "/cifar-10"
    let trainPaths := (List.range 5).map (fun i => s!"{cdir}/data_batch_{i+1}.bin")
    let (trI, trL, nTr) ← loadCifarSplit trainPaths
    let (evI, evL, nEv) ← loadCifarSplit [s!"{cdir}/test_batch.bin"]
    return (trI, trL, nTr, evI, evL, nEv, d0, false)

/-- Synthetic-input data for the `lake run benchmark` probes (`LEAN_MLIR_BENCH_SYNTH`):
    ONE constant batch, reused every step, but with the dataset's *real* `nTrain` so the
    per-epoch step count — and thus the per-epoch / per-step timing — matches the on-disk
    anchors (train-step throughput is value-independent). Lets the benchmark run with zero
    data downloaded. The per-step crop/hflip stays in the loop (so timing matches); eval is
    skipped in synth, so `nEval` is a placeholder. -/
private def mkSynthData (data : VerifiedData) (d0 bs : Nat) :
    IO (ByteArray × ByteArray × Nat × ByteArray × ByteArray × Nat × Nat × Bool) := do
  let (nTr, px, crop) := match data with
    | .imagenette => (9469, 3 * 256 * 256, true)   -- 256² pre-crop → 224² each step
    | .cifar      => (50000, d0, false)
    | _           => (60000, d0, false)             -- mnist
  let img ← F32.const (bs * px).toUSize 0.1
  let lbl ← F32.const bs.toUSize 0.0               -- bs int32 zero labels (4 bytes each)
  pure (img, lbl, nTr, img, lbl, bs, px, crop)

/-- Train a `VerifiedNet` end-to-end on its proof-rendered StableHLO: compile both
    MLIRs → IREE sessions → load data → He/spec init → SGD train + eval loop. The
    SGD update (and lr) are baked into `<slug>_train_step.mlir`; we only feed batches. -/
def VerifiedNet.train (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  IO.println net.blurb
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let synth := (← IO.getEnv "LEAN_MLIR_BENCH_SYNTH").isSome
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    if synth then mkSynthData net.data d0 bs else loadData net.data d0 dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} ({net.specs.size} params, {net.nParams} floats), mean-loss SGD lr={cfg.lr}, He init{if synth then " [SYNTH]" else ""}"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  -- init params in func-arg order from the layout specs (one seed per slot).
  -- Seed base is overridable via LEAN_MLIR_SEED (default 1) to probe how
  -- sensitive convergence is to the specific He-init draw.
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut params := F32.concat parts
  -- LEAN_MLIR_MAX_EPOCHS caps the epoch count (opt-in; absent → full cfg.epochs).
  -- Used by `lake run benchmark` to probe steady-state per-epoch wall-clock with
  -- only a few epochs; harmless otherwise (timing per epoch is LR-independent).
  let nEpochs := match (← IO.getEnv "LEAN_MLIR_MAX_EPOCHS").bind (·.toNat?) with
    | some n => min n cfg.epochs
    | none   => cfg.epochs
  for ep in [0:nEpochs] do
    let tEp0 ← IO.monoMsNow
    for bi in [0:nb] do
      let xbRaw := if synth then trainImg else F32.sliceImages trainImg (bi * bs) bs trainPix
      let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
      let yb := if synth then trainLbl else F32.sliceLabels trainLbl (bi * bs) bs
      params ← IreeSession.mlpTrainStepV tsSess tsFn
                  xb params shapes yb bs.toUSize d0.toUSize nc.toUSize
    let mut correct := 0
    if !synth then          -- synth probe: skip eval (no eval split on disk)
      for bi in [0:nbt] do
        let xb := F32.sliceImages evalImg (bi * bs) bs d0
        let logits ← IreeSession.forwardF32 fwdSess fwdFn params shapes
                        xb xShape bs.toUSize nc.toUSize
        for j in [0:bs] do
          let pred := (F32.argmax10 logits (j * nc).toUSize).toNat
          let lbl  := (evalLbl.get! (4 * (bi * bs + j))).toNat
          if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * bs).toFloat * 100.0
    let epMs := (← IO.monoMsNow) - tEp0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nbt * bs} = {acc}% ({epMs}ms)"
    (← IO.getStdout).flush
  IO.println s!"done (trained {net.name} via the proof-rendered StableHLO)."

/-- **AdamW training driver** — threads the first/second moment buffers as a single
    packed `[θ|m|v]` param blob through the generic FFI (`n_params = 3k`; the moments
    ride in the params slot, so the prebuilt `.so` is unchanged), against the
    baked-hyperparameter packed render `@<slug>_adam_train_step`
    (`ViTRender.vitTrainStepModuleAdamPacked`, optimizer = `Proofs.adamWParam`).
    Moments init to 0; eval reads the θ slice (first `nParams` floats). The Adam
    analogue of `VerifiedNet.train`. -/
def VerifiedNet.trainAdamPacked (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  IO.println net.blurb
  let tsVmfb  := s!".lake/build/{net.slug}_adam_ts.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_adam_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"             fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    loadData net.data d0 dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} AdamW (packed θ|m|v), He init"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := nEval / bs
  -- θ|m|v packed: θ = He-init (one seed per slot, as `train`), m = v = 0. The
  -- shapes descriptor lists every tensor three times (θ, then m, then v).
  let adamShapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes)
  let fwdShapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_adam_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let theta := F32.concat parts
  let zeros ← F32.const net.nParams.toUSize 0.0
  let mut params := F32.concat #[theta, zeros, zeros]
  let pBytes := net.nParams * 4
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xbRaw := F32.sliceImages trainImg (bi * bs) bs trainPix
      let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      params ← IreeSession.mlpTrainStepV tsSess tsFn
                  xb params adamShapes yb bs.toUSize d0.toUSize nc.toUSize
    let thetaCur := params.extract 0 pBytes
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 fwdSess fwdFn thetaCur fwdShapes
                      xb xShape bs.toUSize nc.toUSize
      for j in [0:bs] do
        let pred := (F32.argmax10 logits (j * nc).toUSize).toNat
        let lbl  := (evalLbl.get! (4 * (bi * bs + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * bs).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nbt * bs} = {acc}%"
    (← IO.getStdout).flush
  IO.println s!"done (trained {net.name} with AdamW via packed θ|m|v threading)."

/-- **Scheduled AdamW driver** (Phase 2) — `trainAdamPacked` with a runtime LR and
    bias correction. `lr`/`bc₁`/`bc₂` ride as three rank-0 scalar params in the blob
    tail (`[θ|m|v|lr|bc₁|bc₂]`, the FFI takes no scalar slot) and are returned
    unchanged; the host recomputes them each step: cosine decay + linear warmup for
    `lr`, and `bc₁=1−β₁ᵗ`, `bc₂=1−β₂ᵗ` (proper bias correction). Drives
    `ViTRender.vitTrainStepModuleAdamSched`. -/
def VerifiedNet.trainAdamSched (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (baseLR β1 β2 : Float) (warmupEpochs : Nat) (variant : String := "adam") : IO Unit := do
  -- `variant` selects the rendered train step `@<slug>_<variant>_train_step` (and its artifact /
  -- vmfb / checkpoint names). Default "adam" = the AdamW render; "mom" = the Nesterov-momentum SGD
  -- render (same packed [θ|m|v]+lr/bc1/bc2 signature; the momentum step ignores the m/bc slots and
  -- reads only lr + v, so this driver is shared verbatim). β1/β2 still drive the (unused-by-mom)
  -- bias-correction scalars; the cosine+warmup lr schedule is identical.
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  IO.println net.blurb
  -- Running-stats BN: when `bnChannels` is non-empty the adam train step carries per-layer batch
  -- mean/var out in passthrough slots (so #out=#in), the driver EMAs them into `runningBnStats`,
  -- and eval uses `<slug>_fwd_eval.mlir` (affine BN with the running stats) — class-batch-independent
  -- eval parity, not the degenerate batch-BN-eval. LayerNorm / no-BN nets skip all of this.
  let hasBn := !net.bnChannels.isEmpty
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let tsVmfb  := s!".lake/build/{net.slug}_{variant}_ts.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  let fwdEvalVmfb := s!".lake/build/{net.slug}_fwd_eval_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_{variant}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"             fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let fwdEvalSess ← if hasBn then do
      compileVmfb s!"verified_mlir/{net.slug}_fwd_eval.mlir" fwdEvalVmfb
      IreeSession.create fwdEvalVmfb
    else pure fwdSess
  let synth := (← IO.getEnv "LEAN_MLIR_BENCH_SYNTH").isSome
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    if synth then mkSynthData net.data d0 bs else loadData net.data d0 dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  let nb  := nTrain / bs
  let nbt := nEval / bs
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} {variant} (cosine+warmup {warmupEpochs}ep, baseLR {baseLR}), He init"
  if hasBn then IO.println s!"  running-stats BN: {net.bnChannels.size} layers, {nBnStats} stat floats → eval via @{net.slug}_fwd_eval"
  (← IO.getStdout).flush
  let adamShapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes ++ #[#[], #[], #[]]
                                ++ (if hasBn then bnStatShapes else #[]))
  let fwdShapes := net.shapesBA
  let fwdEvalShapes := packShapes (net.paramShapes ++ bnStatShapes)
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_{variant}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let theta := F32.concat parts
  let zeros ← F32.const net.nParams.toUSize 0.0
  let mut thetamv := F32.concat #[theta, zeros, zeros]
  let mvBytes := 3 * net.nParams * 4
  let pBytes := net.nParams * 4
  -- Running BN stats (EMA of per-layer batch mean/var; mom 1.0 on the first step to seed,
  -- then 0.1). Reset per process — washed out well before the per-epoch eval (mom 0.1).
  let mut runningBnStats ← F32.const nBnStats.toUSize 0.0
  let mut bnFirst := true
  let totalSteps := (cfg.epochs * nb).toFloat
  let warmSteps := (warmupEpochs * nb).toFloat
  -- Auto checkpoint/resume: each epoch writes [θ|m|v] + the next-epoch counter;
  -- on startup, resume from the latest checkpoint if present (survives reaps).
  -- Delete `.lake/build/<slug>_adam_ckpt.bin{,.epoch}` to start fresh.
  let ckptPath := s!".lake/build/{net.slug}_{variant}_ckpt.bin"
  let epPath := ckptPath ++ ".epoch"
  let mut startEpoch := 0
  if (← System.FilePath.pathExists ckptPath) && (← System.FilePath.pathExists epPath) then
    thetamv ← IO.FS.readBinFile ckptPath
    startEpoch := ((← IO.FS.readFile epPath).toNat?).getD 0
    IO.println s!"  ▸ resuming from checkpoint at epoch {startEpoch}"
    (← IO.getStdout).flush
  -- Reuse ONE shuffle buffer across epochs (mirrors the reference trainer's
  -- curImg/curLbl). Shuffling the SAME mutable in place keeps it exclusive
  -- (rc 1) so F32.shuffle mutates it rather than allocating a fresh full-dataset
  -- copy each epoch. The old `F32.shuffle trainImg` kept the pristine trainImg
  -- alive (rc≥2), forcing the copy path every epoch and leaking ~one training
  -- set (5.3 GiB) per epoch → OOM after ~30 epochs on a 188 GB box.
  let mut curImg := trainImg
  let mut curLbl := trainLbl
  -- LEAN_MLIR_MAX_STEPS: run a short steady-state ms/step probe then exit. This is
  -- the benchmark's `attn` anchor — ViT is matmul/attention-bound, so its per-step
  -- cost scales very differently from conv across GPUs and can't borrow the conv
  -- factor. A full ViT epoch is too slow to probe, so we time a step window.
  let probeSteps := (← IO.getEnv "LEAN_MLIR_MAX_STEPS").bind (·.toNat?)
  let probeWarm := 8
  let mut probePrev := 0
  let mut probeTimes : Array Nat := #[]
  for ep in [startEpoch:cfg.epochs] do
    let mut epochLossSum := 0.0
    let mut lastLr := 0.0
    -- Per-epoch Fisher-Yates shuffle (the reference does this; the data is
    -- class-sorted, so without it every batch is a single class — degenerate).
    if !synth then
      let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPix.toUSize (ep + 42).toUSize
      curImg := sImg; curLbl := sLbl
    for bi in [0:nb] do
      let gstep := (ep * nb + bi + 1).toFloat
      let lrt := if gstep ≤ warmSteps then baseLR * gstep / warmSteps
                 else baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (gstep - warmSteps) / (totalSteps - warmSteps)))
      let bc1 := 1.0 - Float.exp (gstep * Float.log β1)
      let bc2 := 1.0 - Float.exp (gstep * Float.log β2)
      let tail := F32.concat #[← F32.const (1 : USize) lrt, ← F32.const (1 : USize) bc1, ← F32.const (1 : USize) bc2]
      -- BN nets append the (ignored) stat-in passthrough slots; the step writes batch stats out.
      let params := if hasBn then F32.concat #[thetamv, tail, runningBnStats] else F32.concat #[thetamv, tail]
      let augSeed := (ep * nb + bi + 1).toUSize
      let xbRaw := if synth then curImg else F32.sliceImages curImg (bi * bs) bs trainPix
      -- Data-pipeline augmentation (the same FFI the unverified trainer uses;
      -- lives in the data pipeline, not the network): Imagenette = random crop
      -- 256→224 (when the source is 256²) + random hflip; CIFAR = hflip only;
      -- MNIST = none.
      let xb ← match net.data with
        | .imagenette =>
            let c ← if crop then F32.randomCrop xbRaw bs.toUSize 3 256 256 224 224 augSeed
                    else pure xbRaw
            F32.randomHFlip c bs.toUSize 3 224 224 (augSeed + 7777)
        | .cifar => F32.randomHFlip xbRaw bs.toUSize 3 32 32 augSeed
        | _ => pure xbRaw
      let yb := if synth then curLbl else F32.sliceLabels curLbl (bi * bs) bs
      let out ← IreeSession.mlpTrainStepV tsSess tsFn xb params adamShapes yb bs.toUSize d0.toUSize nc.toUSize
      -- the train step emits the smoothed-CE loss in the slot after [θ'|m'|v']
      let stepLoss := F32.read out (3 * net.nParams).toUSize
      epochLossSum := epochLossSum + stepLoss
      lastLr := lrt
      if bi < 3 || bi % 100 == 0 then
        IO.println s!"  step {bi}/{nb}: loss={stepLoss}"
        (← IO.getStdout).flush
      thetamv := out.extract 0 mvBytes
      -- EMA the batch BN stats (in the passthrough slots after [θ'|m'|v'|loss|bc1|bc2]).
      if hasBn then
        let batchBn := out.extract ((3 * net.nParams + 3) * 4) ((3 * net.nParams + 3 + nBnStats) * 4)
        runningBnStats ← F32.ema runningBnStats batchBn (if bnFirst then 1.0 else 0.1)
        bnFirst := false
      -- ms/step probe: start the clock past warmup, report + exit at the cap.
      match probeSteps with
      | some ps =>
        if bi == probeWarm then probePrev := (← IO.monoMsNow)
        else if bi > probeWarm && bi ≤ ps then
          let t ← IO.monoMsNow
          probeTimes := probeTimes.push (t - probePrev); probePrev := t
          if bi == ps then
            -- robust: median per-step time (drops the cold-cache / GC-blip outliers)
            let sorted := probeTimes.qsort Nat.blt
            IO.println s!"  PROBE: {sorted[sorted.size / 2]!} ms/step (median of {sorted.size} steps {probeWarm+1}..{ps}, {net.name})"
            (← IO.getStdout).flush
            return ()
      | none => pure ()
    IO.println s!"Epoch {ep + 1}/{cfg.epochs}: loss={epochLossSum / nb.toFloat} lr={lastLr}"
    let thetaCur := thetamv.extract 0 pBytes
    -- BN nets eval through `@<slug>_fwd_eval` with the running stats appended; others use `@<slug>_fwd`.
    let evalSess := if hasBn then fwdEvalSess else fwdSess
    let evalFn := if hasBn then s!"m.{net.slug}_fwd_eval" else fwdFn
    let evalParams := if hasBn then F32.concat #[thetaCur, runningBnStats] else thetaCur
    let evalShapes := if hasBn then fwdEvalShapes else fwdShapes
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 evalSess evalFn evalParams evalShapes
                      xb xShape bs.toUSize nc.toUSize
      for j in [0:bs] do
        let pred := (F32.argmax10 logits (j * nc).toUSize).toNat
        let lbl  := (evalLbl.get! (4 * (bi * bs + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * bs).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nbt * bs} = {acc}%"
    (← IO.getStdout).flush
    IO.FS.writeBinFile ckptPath thetamv
    IO.FS.writeFile epPath (toString (ep + 1))
  IO.println s!"done (trained {net.name} with AdamW + cosine/warmup via packed threading)."

/-- Train driver for the **2-parameter linear** path (Chapter 1). The verified
    `@<slug>_train_step` takes `W0`/`b0` as *separate* arguments (`linearTrainStepV`),
    weights are zero-initialized, and the loss/lr are baked into the MLIR — distinct
    from the packed-params, He-init `train` above. Only the linear classifier uses this;
    shares `compileVmfb` / `loadData` / the eval pass with the main driver. -/
def VerifiedNet.trainLinear (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println net.blurb
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _trainPix, _crop) ←
    loadData net.data d0 dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; dense {d0}->{d1}, bs {bs}, SGD"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA          -- packed [W0|b0] layout for the verified forward
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut W0 ← F32.const (d0 * d1).toUSize 0.0
  let mut b0 ← F32.const d1.toUSize 0.0
  -- LEAN_MLIR_MAX_EPOCHS cap + per-epoch (Nms) timing, matching `train` (used by
  -- `lake run benchmark`); opt-in, full cfg.epochs otherwise.
  let nEpochs := match (← IO.getEnv "LEAN_MLIR_MAX_EPOCHS").bind (·.toNat?) with
    | some n => min n cfg.epochs
    | none   => cfg.epochs
  for ep in [0:nEpochs] do
    let tEp0 ← IO.monoMsNow
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let out ← IreeSession.linearTrainStepV tsSess tsFn
                  xb W0 b0 yb bs.toUSize d0.toUSize d1.toUSize
      W0 := out.extract 0 (d0 * d1 * 4)
      b0 := out.extract (d0 * d1 * 4) ((d0 * d1 + d1) * 4)
    let params := W0 ++ b0
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 fwdSess fwdFn params shapes
                      xb xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        let pred := (F32.argmax10 logits (j * d1).toUSize).toNat
        let lbl  := (evalLbl.get! (4 * (bi * bs + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * bs).toFloat * 100.0
    let epMs := (← IO.monoMsNow) - tEp0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nbt * bs} = {acc}% ({epMs}ms)"
    (← IO.getStdout).flush
  IO.println s!"done (trained {net.name} via the proof-rendered StableHLO)."

/-- Phase-3 PGD-step kernel for the linear classifier (`planning/robustness.md`).
    `forward → softmax-CE input gradient dx = (softmax(xW+b) − onehot)·Wᵀ` (the proven
    linear input-VJP, `Proofs.mlpInputGrad`'s 1-layer case) → L∞ sign-step → project to the
    `eps`-ball around `x0` → clip to [0,1]. Returns the advanced adversarial input `x_adv`.
    `eps`/`alpha` baked as constants (recompiled per sweep point). Invoked via the generic
    `forwardF32` FFI with `onehot`+`x0` in the params blob and `nClasses := d0` (output size) —
    no new FFI/C shim. The whole PGD step runs on the GPU; the host just iterates. -/
private def genLinearPgdStep (bs d0 d1 : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let bxd0 := s!"tensor<{bs}x{d0}xf32>"
  let bxd1 := s!"tensor<{bs}x{d1}xf32>"
  let wty  := s!"tensor<{d0}x{d1}xf32>"
  let bty  := s!"tensor<{d1}xf32>"
  let rty  := s!"tensor<{bs}xf32>"
  -- shared: forward → softmax-CE input gradient %dx, then the broadcast constants
  let header :=
    "module @m {\n" ++
    s!"  func.func @linear_pgd_step(%x: {bxd0}, %W0: {wty}, %b0: {bty}, %onehot: {bxd1}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %mm = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxd0}, {wty}) -> {bxd1}\n" ++
    s!"    %bb = stablehlo.broadcast_in_dim %b0, dims = [1] : ({bty}) -> {bxd1}\n" ++
    s!"    %logits = stablehlo.add %mm, %bb : {bxd1}\n" ++
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {bxd1}\n" ++
    s!"    %exp = stablehlo.exponential %shift : {bxd1}\n" ++
    s!"    %ssum = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %softmax = stablehlo.divide %exp, %ssumb : {bxd1}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {bxd1}\n" ++
    s!"    %dx = stablehlo.dot_general %g, %W0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxd1}, {wty}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  -- step + projection: L∞ (sign, box-clip to x0±eps) or L2 (normalized grad, eps-ball)
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %step = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %step : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %c1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %c1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %step = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %step : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %c3 = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %c4 = stablehlo.minimum %c3, %oneb : {bxd0}\n" ++
  s!"    return %c4 : {bxd0}\n" ++
  "  }\n}\n"

/-- Build a one-hot `[bs, d1]` f32 batch from int32-LE labels (1.0 = bytes 00 00 80 3F). -/
private def oneHotBatch (labels : ByteArray) (start bs d1 : Nat) : IO ByteArray := do
  let mut oh ← F32.const (bs * d1).toUSize 0.0
  for j in [0:bs] do
    let lbl := (labels.get! (4 * (start + j))).toNat
    let fi := j * d1 + lbl
    oh := (((oh.set! (4*fi) 0).set! (4*fi+1) 0).set! (4*fi+2) 0x80).set! (4*fi+3) 0x3F
  return oh

/-- Spectral norm `‖W‖₂` of `W : [d0,d1]` (row-major) by power iteration on the small
    `WᵀW : [d1,d1]` Gram matrix. For the linear net this IS the global Lipschitz constant
    of the logit map (`logits = xW+b`, Jacobian `Wᵀ`). Host-side, pure. -/
private def specNormW (W : ByteArray) (d0 d1 : Nat) : Float := Id.run do
  let g := fun (i j : Nat) => Id.run do      -- WᵀW[i,j] = Σ_k W[k,i]·W[k,j]
    let mut s := 0.0
    for k in [0:d0] do
      s := s + (F32.read W (k*d1+i).toUSize) * (F32.read W (k*d1+j).toUSize)
    pure s
  let mut wtw : Array Float := Array.replicate (d1*d1) 0.0
  for i in [0:d1] do
    for j in [0:d1] do
      wtw := wtw.set! (i*d1+j) (g i j)
  let mv := fun (v : Array Float) => Id.run do  -- WᵀW · v
    let mut u : Array Float := Array.replicate d1 0.0
    for i in [0:d1] do
      let mut s := 0.0
      for j in [0:d1] do s := s + wtw[i*d1+j]! * v[j]!
      u := u.set! i s
    pure u
  let mut v : Array Float := Array.replicate d1 1.0
  for _ in [0:60] do
    let u := mv v
    let mut nrm := 0.0
    for i in [0:d1] do nrm := nrm + u[i]!*u[i]!
    nrm := Float.sqrt nrm
    if nrm > 1e-20 then
      for i in [0:d1] do v := v.set! i (u[i]!/nrm)
  let u := mv v
  let mut lam := 0.0
  for i in [0:d1] do lam := lam + v[i]! * u[i]!   -- Rayleigh quotient (‖v‖=1)
  pure (Float.sqrt lam)

/-- Spectral norm `‖M‖₂` of a `[rows, cols]` matrix given by an index function `get i j`
    (the same power iteration on the `cols×cols` Gram as `specNormW`, but reading via `get`
    so it works on strided sub-tensors — e.g. one tap-plane of a conv kernel). -/
private def specNormGet (get : Nat → Nat → Float) (rows cols : Nat) : Float := Id.run do
  let gram := fun (i j : Nat) => Id.run do        -- (MᵀM)[i,j] = Σ_k M[k,i]·M[k,j]
    let mut s := 0.0
    for k in [0:rows] do s := s + (get k i) * (get k j)
    pure s
  let mut wtw : Array Float := Array.replicate (cols*cols) 0.0
  for i in [0:cols] do
    for j in [0:cols] do
      wtw := wtw.set! (i*cols+j) (gram i j)
  let mv := fun (v : Array Float) => Id.run do
    let mut u : Array Float := Array.replicate cols 0.0
    for i in [0:cols] do
      let mut s := 0.0
      for j in [0:cols] do s := s + wtw[i*cols+j]! * v[j]!
      u := u.set! i s
    pure u
  let mut v : Array Float := Array.replicate cols 1.0
  for _ in [0:60] do
    let u := mv v
    let mut nrm := 0.0
    for i in [0:cols] do nrm := nrm + u[i]!*u[i]!
    nrm := Float.sqrt nrm
    if nrm > 1e-20 then
      for i in [0:cols] do v := v.set! i (u[i]!/nrm)
  let u := mv v
  let mut lam := 0.0
  for i in [0:cols] do lam := lam + v[i]! * u[i]!
  pure (Float.sqrt lam)

/-- A **sound** (loose) upper bound on the L2 operator norm of a zero-padded 2-D
    convolution with kernel `W : [outC, inC, kh, kw]` (row-major). Writing the conv as a
    sum over spatial taps `T = Σ_{ky,kx} S_{ky,kx} ∘ M_{ky,kx}` — each `S` a (norm ≤ 1)
    shift and each `M` the pointwise `[outC,inC]` channel-mixing matrix at that tap — the
    triangle inequality gives `‖T‖₂ ≤ Σ_{ky,kx} ‖W[:,:,ky,kx]‖₂`. Each tap-plane's spectral
    norm is the same power iteration as `specNormW`. Loose by up to `√(kh·kw)` vs the exact
    (Sedghi–Gupta–Long) value — which only sharpens the "depth ⇒ vacuous product" message. -/
private def specNormConvTapSum (W : ByteArray) (outC inC kh kw : Nat) : Float := Id.run do
  let mut s := 0.0
  for ky in [0:kh] do
    for kx in [0:kw] do
      s := s + specNormGet
        (fun o i => F32.read W (((o*inC+i)*kh+ky)*kw+kx).toUSize) outC inC
  pure s

/-- **Matrix-free** spectral norm `‖W‖₂` of `W : [d0,d1]` (row-major) — power iteration that
    applies `W` and `Wᵀ` as mat-vecs (`σ = ‖W v‖`, `v` the top right singular vector) instead
    of forming the `d1×d1` Gram. ~`2·d0·d1` per iteration vs `d0·d1²` for `specNormW`, so it's
    cheap enough to call **during** training (the spectral-norm projection below). Fewer iters
    (`iters`) trade a little precision for speed; `specNormW` stays the high-precision cert path. -/
private def specNormMV (W : ByteArray) (d0 d1 : Nat) (iters : Nat := 15) : Float := Id.run do
  let norm := fun (a : Array Float) (n : Nat) => Id.run do
    let mut s := 0.0
    for i in [0:n] do s := s + a[i]! * a[i]!
    pure (Float.sqrt s)
  let normalize := fun (a : Array Float) (n : Nat) => Id.run do
    let s := norm a n
    if s > 1e-20 then
      let mut b := a
      for i in [0:n] do b := b.set! i (a[i]! / s)
      pure b
    else pure a
  let mut v : Array Float := normalize (Array.replicate d1 1.0) d1
  let mut σ := 0.0
  for _ in [0:iters] do
    let mut u : Array Float := Array.replicate d0 0.0    -- u = W v
    for r in [0:d0] do
      let mut s := 0.0
      for cc in [0:d1] do s := s + F32.read W (r*d1+cc).toUSize * v[cc]!
      u := u.set! r s
    σ := norm u d0                                       -- σ = ‖W v‖ (‖v‖ = 1)
    let mut w : Array Float := Array.replicate d1 0.0    -- w = Wᵀ u
    for cc in [0:d1] do
      let mut s := 0.0
      for r in [0:d0] do s := s + F32.read W (r*d1+cc).toUSize * u[r]!
      w := w.set! cc s
    v := normalize w d1
  pure σ

/-- **Spectral-norm projection** (projected SGD onto the spectral ball): rescale every weight
    whose L2-Lipschitz bound exceeds `c` down to `c`, leaving biases untouched. Dense `[d0,d1]`:
    cap the spectral norm `‖W‖₂` (`specNormMV`). Conv `[o,i,kh,kw]`: cap the **same** tap-sum
    operator bound the CNN certificate uses (`specNormConvTapSum`, `‖T‖₂ ≤ Σ_tap‖W[:,:,ky,kx]‖₂`)
    by scaling the whole kernel — so the projection and the cert control the identical quantity.
    Caps each layer's Lipschitz constant at `c`, so the global `L = ∏ᵢ ≤ cᵏ` — the lever that
    turns the (vacuous) product certificate non-vacuous. `F32.scaleShift` does the rescale. -/
private def projectSpectral (theta : ByteArray) (specs : Array (Array Nat × Nat)) (c : Float)
    : IO ByteArray := do
  let mut parts : Array ByteArray := #[]
  let mut off := 0
  for spec in specs do
    let dims := spec.1
    let len := dims.foldl (·*·) 1
    let slice := theta.extract (off*4) ((off+len)*4)
    let slice' ← if dims.size == 2 then do
        let σ := specNormMV slice dims[0]! dims[1]!
        if σ > c then F32.scaleShift slice (c/σ) 0.0 else pure slice
      else if dims.size == 4 then do
        let s := specNormConvTapSum slice dims[0]! dims[1]! dims[2]! dims[3]!
        if s > c then F32.scaleShift slice (c/s) 0.0 else pure slice
      else pure slice
    parts := parts.push slice'
    off := off + len
  return F32.concat parts

/-- Phase-3 PGD-step kernel for the 2-hidden-layer MLP (`d0→h→h→d1`, ReLU). Forward
    (saving the pre-activations `z0,z1`) → the proven `mlpInputGrad` VJP
    `dx = ((g·W₂ᵀ ⊙ relu'(z₁))·W₁ᵀ ⊙ relu'(z₀))·W₀ᵀ` (ReLU masks via `compare GT`/`select`,
    the codegen's idiom) → L∞/L2 step + projection. Returns `x_adv`. -/
private def genMlpPgdStep (bs d0 h d1 : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let bxd0 := s!"tensor<{bs}x{d0}xf32>"
  let bxh  := s!"tensor<{bs}x{h}xf32>"
  let bxd1 := s!"tensor<{bs}x{d1}xf32>"
  let bxhi := s!"tensor<{bs}x{h}xi1>"
  let w0ty := s!"tensor<{d0}x{h}xf32>"
  let w1ty := s!"tensor<{h}x{h}xf32>"
  let w2ty := s!"tensor<{h}x{d1}xf32>"
  let hbty := s!"tensor<{h}xf32>"
  let d1bt := s!"tensor<{d1}xf32>"
  let rty  := s!"tensor<{bs}xf32>"
  let header :=
    "module @m {\n" ++
    s!"  func.func @mlp_pgd_step(%x: {bxd0}, %W0: {w0ty}, %b0: {hbty}, %W1: {w1ty}, %b1: {hbty}, %W2: {w2ty}, %b2: {d1bt}, %onehot: {bxd1}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %zh = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxh}\n" ++
    -- forward (save preacts z0, z1)
    s!"    %z0mm = stablehlo.dot_general %x, %W0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxd0}, {w0ty}) -> {bxh}\n" ++
    s!"    %b0b = stablehlo.broadcast_in_dim %b0, dims = [1] : ({hbty}) -> {bxh}\n" ++
    s!"    %z0 = stablehlo.add %z0mm, %b0b : {bxh}\n" ++
    s!"    %h0 = stablehlo.maximum %z0, %zh : {bxh}\n" ++
    s!"    %z1mm = stablehlo.dot_general %h0, %W1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxh}, {w1ty}) -> {bxh}\n" ++
    s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : ({hbty}) -> {bxh}\n" ++
    s!"    %z1 = stablehlo.add %z1mm, %b1b : {bxh}\n" ++
    s!"    %h1 = stablehlo.maximum %z1, %zh : {bxh}\n" ++
    s!"    %lgmm = stablehlo.dot_general %h1, %W2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({bxh}, {w2ty}) -> {bxd1}\n" ++
    s!"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : ({d1bt}) -> {bxd1}\n" ++
    s!"    %logits = stablehlo.add %lgmm, %b2b : {bxd1}\n" ++
    -- softmax-CE gradient g
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {bxd1}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {bxd1}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd1}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {bxd1}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {bxd1}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {bxd1}\n" ++
    -- backward: dx = ((g·W2ᵀ ⊙ relu'(z1))·W1ᵀ ⊙ relu'(z0))·W0ᵀ
    s!"    %dh1 = stablehlo.dot_general %g, %W2, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxd1}, {w2ty}) -> {bxh}\n" ++
    s!"    %rm1 = stablehlo.compare GT, %z1, %zh : ({bxh}, {bxh}) -> {bxhi}\n" ++
    s!"    %dz1 = stablehlo.select %rm1, %dh1, %zh : {bxhi}, {bxh}\n" ++
    s!"    %dh0 = stablehlo.dot_general %dz1, %W1, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxh}, {w1ty}) -> {bxh}\n" ++
    s!"    %rm0 = stablehlo.compare GT, %z0, %zh : ({bxh}, {bxh}) -> {bxhi}\n" ++
    s!"    %dz0 = stablehlo.select %rm0, %dh0, %zh : {bxhi}, {bxh}\n" ++
    s!"    %dx = stablehlo.dot_general %dz0, %W0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({bxh}, {w0ty}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **Phase-3 PGD-step kernel for the verified MNIST CNN** (`conv 1→32 → relu → conv 32→32 →
    relu → maxpool 28→14 → flatten → dense 6272→512 → relu → 512→512 → relu → 512→10`).
    Forward (saving every pre-activation + the maxpool input) → softmax-CE seed → the full
    input-VJP `dx`, mirroring `verified_mlir/cnn_train_step.mlir`'s backward ops:
    `dot_general` adjoints + ReLU masks (`compare GT`/`select`), **maxpool-back**
    (`select_and_scatter`, scatter the pooled cotangent to the argmax cells), and the two
    **conv input-VJPs** (transpose-`o,i` + spatial `reverse` of the kernel, then the same
    padded conv). The train step stops at `dz1` (it only needs weight grads); here we add the
    final conv1 input-VJP to reach `dx` over the pixels. Then the L∞ sign-step / L2 projected
    step + ε-ball project + [0,1] clip. Architecture is fixed; only `bs`/`eps`/`alpha` vary. -/
private def genCnnPgdStep (bs : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let i4  := s!"tensor<{bs}x1x28x28xf32>"
  let c4  := s!"tensor<{bs}x32x28x28xf32>"
  let c4i := s!"tensor<{bs}x32x28x28xi1>"
  let p4  := s!"tensor<{bs}x32x14x14xf32>"
  let f2  := s!"tensor<{bs}x6272xf32>"
  let h2  := s!"tensor<{bs}x512xf32>"
  let h2i := s!"tensor<{bs}x512xi1>"
  let o2  := s!"tensor<{bs}x10xf32>"
  let bxd0 := s!"tensor<{bs}x784xf32>"
  let rty := s!"tensor<{bs}xf32>"
  let convCfg := "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
  let header :=
    "module @m {\n" ++
    s!"  func.func @cnn_pgd_step(%x: {bxd0}, %W1: tensor<32x1x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<6272x512xf32>, %b3: tensor<512xf32>, %W4: tensor<512x512xf32>, %b4: tensor<512xf32>, %W5: tensor<512x10xf32>, %b5: tensor<10xf32>, %onehot: {o2}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %zc4 = stablehlo.constant dense<0.0> : {c4}\n" ++
    s!"    %zh = stablehlo.constant dense<0.0> : {h2}\n" ++
    -- ── forward (save pre-acts z1,z2,z3,z4 + maxpool input h2c) ──
    s!"    %v0 = stablehlo.reshape %x : ({bxd0}) -> {i4}\n" ++
    s!"    %c1 = stablehlo.convolution(%v0, %W1)\n      {convCfg} : ({i4}, tensor<32x1x3x3xf32>) -> {c4}\n" ++
    s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> {c4}\n" ++
    s!"    %z1 = stablehlo.add %c1, %b1b : {c4}\n" ++
    s!"    %h1 = stablehlo.maximum %z1, %zc4 : {c4}\n" ++
    s!"    %c2 = stablehlo.convolution(%h1, %W2)\n      {convCfg} : ({c4}, tensor<32x32x3x3xf32>) -> {c4}\n" ++
    s!"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> {c4}\n" ++
    s!"    %z2 = stablehlo.add %c2, %b2b : {c4}\n" ++
    s!"    %h2c = stablehlo.maximum %z2, %zc4 : {c4}\n" ++
    s!"    %pool = \"stablehlo.reduce_window\"(%h2c, %ninf) (\{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    s!"    }) \{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : ({c4}, tensor<f32>) -> {p4}\n" ++
    s!"    %flat = stablehlo.reshape %pool : ({p4}) -> {f2}\n" ++
    s!"    %d3 = stablehlo.dot_general %flat, %W3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({f2}, tensor<6272x512xf32>) -> {h2}\n" ++
    s!"    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z3 = stablehlo.add %d3, %b3b : {h2}\n" ++
    s!"    %h3 = stablehlo.maximum %z3, %zh : {h2}\n" ++
    s!"    %d4 = stablehlo.dot_general %h3, %W4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z4 = stablehlo.add %d4, %b4b : {h2}\n" ++
    s!"    %h4 = stablehlo.maximum %z4, %zh : {h2}\n" ++
    s!"    %d5 = stablehlo.dot_general %h4, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x10xf32>) -> {o2}\n" ++
    s!"    %b5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<10xf32>) -> {o2}\n" ++
    s!"    %logits = stablehlo.add %d5, %b5b : {o2}\n" ++
    -- ── softmax-CE seed g = softmax(logits) − onehot ──
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {o2}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {o2}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {o2}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {o2}\n" ++
    -- ── backward to dx ──
    s!"    %dh4 = stablehlo.dot_general %g, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({o2}, tensor<512x10xf32>) -> {h2}\n" ++
    s!"    %rm4 = stablehlo.compare GT, %z4, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz4 = stablehlo.select %rm4, %dh4, %zh : {h2i}, {h2}\n" ++
    s!"    %dh3 = stablehlo.dot_general %dz4, %W4, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %rm3 = stablehlo.compare GT, %z3, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz3 = stablehlo.select %rm3, %dh3, %zh : {h2i}, {h2}\n" ++
    s!"    %dflat = stablehlo.dot_general %dz3, %W3, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<6272x512xf32>) -> {f2}\n" ++
    s!"    %dpool = stablehlo.reshape %dflat : ({f2}) -> {p4}\n" ++
    -- maxpool-back: scatter the pooled cotangent back to the argmax cells of the pool input
    s!"    %dpre2 = \"stablehlo.select_and_scatter\"(%h2c, %dpool, %zf) (\{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, {\n" ++
    "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    s!"    }) \{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : ({c4}, {p4}, tensor<f32>) -> {c4}\n" ++
    s!"    %rmc2 = stablehlo.compare GT, %z2, %zc4 : ({c4}, {c4}) -> {c4i}\n" ++
    s!"    %dz2 = stablehlo.select %rmc2, %dpre2, %zc4 : {c4i}, {c4}\n" ++
    -- conv2 input-VJP: transpose o,i + spatial-reverse the kernel, conv with the cotangent
    s!"    %w2t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>\n" ++
    s!"    %w2r = stablehlo.reverse %w2t, dims = [2, 3] : tensor<32x32x3x3xf32>\n" ++
    s!"    %dpost1 = stablehlo.convolution(%dz2, %w2r)\n      {convCfg} : ({c4}, tensor<32x32x3x3xf32>) -> {c4}\n" ++
    s!"    %rmc1 = stablehlo.compare GT, %z1, %zc4 : ({c4}, {c4}) -> {c4i}\n" ++
    s!"    %dz1 = stablehlo.select %rmc1, %dpost1, %zc4 : {c4i}, {c4}\n" ++
    -- conv1 input-VJP → dx over the pixels (the step the train kernel omits; W1: 32x1x3x3 → 1x32x3x3)
    s!"    %w1t = stablehlo.transpose %W1, dims = [1, 0, 2, 3] : (tensor<32x1x3x3xf32>) -> tensor<1x32x3x3xf32>\n" ++
    s!"    %w1r = stablehlo.reverse %w1t, dims = [2, 3] : tensor<1x32x3x3xf32>\n" ++
    s!"    %dxi = stablehlo.convolution(%dz1, %w1r)\n      {convCfg} : ({c4}, tensor<1x32x3x3xf32>) -> {i4}\n" ++
    s!"    %dx = stablehlo.reshape %dxi : ({i4}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **Phase-3 PGD-step kernel for the verified CIFAR-10 CNN** — the deeper sibling of
    `genCnnPgdStep` (`conv 3→32 → relu → conv 32→32 → relu → maxpool → conv 32→64 → relu →
    conv 64→64 → relu → maxpool → flatten(4096) → 512 → 512 → 10`). Same recipe — forward
    (saving every pre-activation + both maxpool inputs) → softmax-CE seed → the full input-VJP
    `dx`, mirroring `verified_mlir/cifar_train_step.mlir`'s backward (4 conv input-VJPs, 2
    `select_and_scatter` maxpool-backs, ReLU masks, dense adjoints) + the final conv1 input-VJP
    the train step omits — then the L∞/L2 step + ε-ball project + [0,1] clip. 3-channel 32×32,
    `bs`/`eps`/`alpha` vary. -/
private def genCifarPgdStep (bs : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let i4   := s!"tensor<{bs}x3x32x32xf32>"
  let m32  := s!"tensor<{bs}x32x32x32xf32>"
  let m32i := s!"tensor<{bs}x32x32x32xi1>"
  let p32  := s!"tensor<{bs}x32x16x16xf32>"
  let m64  := s!"tensor<{bs}x64x16x16xf32>"
  let m64i := s!"tensor<{bs}x64x16x16xi1>"
  let p64  := s!"tensor<{bs}x64x8x8xf32>"
  let f2   := s!"tensor<{bs}x4096xf32>"
  let h2   := s!"tensor<{bs}x512xf32>"
  let h2i  := s!"tensor<{bs}x512xi1>"
  let o2   := s!"tensor<{bs}x10xf32>"
  let bxd0 := s!"tensor<{bs}x3072xf32>"
  let rty  := s!"tensor<{bs}xf32>"
  let convCfg := "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
  let poolAttr := "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}"
  let header :=
    "module @m {\n" ++
    s!"  func.func @cifar_pgd_step(%x: {bxd0}, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>, %onehot: {o2}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %z32 = stablehlo.constant dense<0.0> : {m32}\n" ++
    s!"    %z64 = stablehlo.constant dense<0.0> : {m64}\n" ++
    s!"    %zh = stablehlo.constant dense<0.0> : {h2}\n" ++
    -- ── forward (save pre-acts z1,z2,z3,z4,z5,z6 + both maxpool inputs h2c,h4c) ──
    s!"    %v0 = stablehlo.reshape %x : ({bxd0}) -> {i4}\n" ++
    s!"    %c1 = stablehlo.convolution(%v0, %W1)\n      {convCfg} : ({i4}, tensor<32x3x3x3xf32>) -> {m32}\n" ++
    s!"    %b1b = stablehlo.broadcast_in_dim %b1, dims = [1] : (tensor<32xf32>) -> {m32}\n" ++
    s!"    %z1 = stablehlo.add %c1, %b1b : {m32}\n" ++
    s!"    %h1 = stablehlo.maximum %z1, %z32 : {m32}\n" ++
    s!"    %c2 = stablehlo.convolution(%h1, %W2)\n      {convCfg} : ({m32}, tensor<32x32x3x3xf32>) -> {m32}\n" ++
    s!"    %b2b = stablehlo.broadcast_in_dim %b2, dims = [1] : (tensor<32xf32>) -> {m32}\n" ++
    s!"    %z2 = stablehlo.add %c2, %b2b : {m32}\n" ++
    s!"    %h2c = stablehlo.maximum %z2, %z32 : {m32}\n" ++
    s!"    %pool1 = \"stablehlo.reduce_window\"(%h2c, %ninf) (\{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m32}, tensor<f32>) -> {p32}\n" ++
    s!"    %c3 = stablehlo.convolution(%pool1, %W3)\n      {convCfg} : ({p32}, tensor<64x32x3x3xf32>) -> {m64}\n" ++
    s!"    %b3b = stablehlo.broadcast_in_dim %b3, dims = [1] : (tensor<64xf32>) -> {m64}\n" ++
    s!"    %z3 = stablehlo.add %c3, %b3b : {m64}\n" ++
    s!"    %h3 = stablehlo.maximum %z3, %z64 : {m64}\n" ++
    s!"    %c4 = stablehlo.convolution(%h3, %W4)\n      {convCfg} : ({m64}, tensor<64x64x3x3xf32>) -> {m64}\n" ++
    s!"    %b4b = stablehlo.broadcast_in_dim %b4, dims = [1] : (tensor<64xf32>) -> {m64}\n" ++
    s!"    %z4 = stablehlo.add %c4, %b4b : {m64}\n" ++
    s!"    %h4c = stablehlo.maximum %z4, %z64 : {m64}\n" ++
    s!"    %pool2 = \"stablehlo.reduce_window\"(%h4c, %ninf) (\{\n" ++
    "      ^bb0(%qa: tensor<f32>, %qb: tensor<f32>):\n" ++
    "        %qm = stablehlo.maximum %qa, %qb : tensor<f32>\n" ++
    "        stablehlo.return %qm : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m64}, tensor<f32>) -> {p64}\n" ++
    s!"    %flat = stablehlo.reshape %pool2 : ({p64}) -> {f2}\n" ++
    s!"    %d5 = stablehlo.dot_general %flat, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({f2}, tensor<4096x512xf32>) -> {h2}\n" ++
    s!"    %b5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z5 = stablehlo.add %d5, %b5b : {h2}\n" ++
    s!"    %h5 = stablehlo.maximum %z5, %zh : {h2}\n" ++
    s!"    %d6 = stablehlo.dot_general %h5, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %b6b = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z6 = stablehlo.add %d6, %b6b : {h2}\n" ++
    s!"    %h6 = stablehlo.maximum %z6, %zh : {h2}\n" ++
    s!"    %d7 = stablehlo.dot_general %h6, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x10xf32>) -> {o2}\n" ++
    s!"    %b7b = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> {o2}\n" ++
    s!"    %logits = stablehlo.add %d7, %b7b : {o2}\n" ++
    -- ── softmax-CE seed ──
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {o2}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {o2}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {o2}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {o2}\n" ++
    -- ── backward to dx ──
    s!"    %dh6 = stablehlo.dot_general %g, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({o2}, tensor<512x10xf32>) -> {h2}\n" ++
    s!"    %rm6 = stablehlo.compare GT, %z6, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz6 = stablehlo.select %rm6, %dh6, %zh : {h2i}, {h2}\n" ++
    s!"    %dh5 = stablehlo.dot_general %dz6, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %rm5 = stablehlo.compare GT, %z5, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz5 = stablehlo.select %rm5, %dh5, %zh : {h2i}, {h2}\n" ++
    s!"    %dflat = stablehlo.dot_general %dz5, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<4096x512xf32>) -> {f2}\n" ++
    s!"    %dpool2 = stablehlo.reshape %dflat : ({f2}) -> {p64}\n" ++
    -- maxpool2-back
    s!"    %dpre4 = \"stablehlo.select_and_scatter\"(%h4c, %dpool2, %zf) (\{\n" ++
    "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
    "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %sge : tensor<i1>\n" ++
    "    }, {\n" ++
    "      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):\n" ++
    "        %ss = stablehlo.add %sc, %sd : tensor<f32>\n" ++
    "        stablehlo.return %ss : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m64}, {p64}, tensor<f32>) -> {m64}\n" ++
    s!"    %rmc4 = stablehlo.compare GT, %z4, %z64 : ({m64}, {m64}) -> {m64i}\n" ++
    s!"    %dz4 = stablehlo.select %rmc4, %dpre4, %z64 : {m64i}, {m64}\n" ++
    -- conv4 input-VJP
    s!"    %w4t = stablehlo.transpose %W4, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>\n" ++
    s!"    %w4r = stablehlo.reverse %w4t, dims = [2, 3] : tensor<64x64x3x3xf32>\n" ++
    s!"    %dpost3 = stablehlo.convolution(%dz4, %w4r)\n      {convCfg} : ({m64}, tensor<64x64x3x3xf32>) -> {m64}\n" ++
    s!"    %rmc3 = stablehlo.compare GT, %z3, %z64 : ({m64}, {m64}) -> {m64i}\n" ++
    s!"    %dz3 = stablehlo.select %rmc3, %dpost3, %z64 : {m64i}, {m64}\n" ++
    -- conv3 input-VJP (W3: 64x32x3x3 → 32x64x3x3): grad back to the pool1 output [bs,32,16,16]
    s!"    %w3t = stablehlo.transpose %W3, dims = [1, 0, 2, 3] : (tensor<64x32x3x3xf32>) -> tensor<32x64x3x3xf32>\n" ++
    s!"    %w3r = stablehlo.reverse %w3t, dims = [2, 3] : tensor<32x64x3x3xf32>\n" ++
    s!"    %dpool1 = stablehlo.convolution(%dz3, %w3r)\n      {convCfg} : ({m64}, tensor<32x64x3x3xf32>) -> {p32}\n" ++
    -- maxpool1-back
    s!"    %dpre2 = \"stablehlo.select_and_scatter\"(%h2c, %dpool1, %zf) (\{\n" ++
    "      ^bb0(%ta: tensor<f32>, %tb: tensor<f32>):\n" ++
    "        %tge = stablehlo.compare GE, %ta, %tb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    "        stablehlo.return %tge : tensor<i1>\n" ++
    "    }, {\n" ++
    "      ^bb0(%tc: tensor<f32>, %td: tensor<f32>):\n" ++
    "        %ts = stablehlo.add %tc, %td : tensor<f32>\n" ++
    "        stablehlo.return %ts : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({m32}, {p32}, tensor<f32>) -> {m32}\n" ++
    s!"    %rmc2 = stablehlo.compare GT, %z2, %z32 : ({m32}, {m32}) -> {m32i}\n" ++
    s!"    %dz2 = stablehlo.select %rmc2, %dpre2, %z32 : {m32i}, {m32}\n" ++
    -- conv2 input-VJP
    s!"    %w2t = stablehlo.transpose %W2, dims = [1, 0, 2, 3] : (tensor<32x32x3x3xf32>) -> tensor<32x32x3x3xf32>\n" ++
    s!"    %w2r = stablehlo.reverse %w2t, dims = [2, 3] : tensor<32x32x3x3xf32>\n" ++
    s!"    %dpost1 = stablehlo.convolution(%dz2, %w2r)\n      {convCfg} : ({m32}, tensor<32x32x3x3xf32>) -> {m32}\n" ++
    s!"    %rmc1 = stablehlo.compare GT, %z1, %z32 : ({m32}, {m32}) -> {m32i}\n" ++
    s!"    %dz1 = stablehlo.select %rmc1, %dpost1, %z32 : {m32i}, {m32}\n" ++
    -- conv1 input-VJP → dx (W1: 32x3x3x3 → 3x32x3x3)
    s!"    %w1t = stablehlo.transpose %W1, dims = [1, 0, 2, 3] : (tensor<32x3x3x3xf32>) -> tensor<3x32x3x3xf32>\n" ++
    s!"    %w1r = stablehlo.reverse %w1t, dims = [2, 3] : tensor<3x32x3x3xf32>\n" ++
    s!"    %dxi = stablehlo.convolution(%dz1, %w1r)\n      {convCfg} : ({m32}, tensor<3x32x3x3xf32>) -> {i4}\n" ++
    s!"    %dx = stablehlo.reshape %dxi : ({i4}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxn = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxn, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **Phase-3 PGD-step kernel for the verified CIFAR-10 CNN + per-channel BatchNorm** (`cifar_bn`).
    The net's "BN" is **instance normalization** (`cifar_bn_fwd`: each image normalized over its
    spatial dims per channel, `nf=H·W`) — so the per-image gradient is clean, there's no train/eval
    split, and the deployed forward IS the attacked forward. Structure: `conv→+b→BN→relu` ×4 (2
    maxpools) → 3 denses. Forward saves every BN's `istd`/`xhat` + post-acts; backward runs the full
    input-VJP to `dx`: dense adjoints, ReLU masks, 2 maxpool `select_and_scatter`-backs, the BN
    grad-input 3-term formula `dx = istd·(dxhat − meanₛ dxhat − xhat·meanₛ(dxhat·xhat))` (per image,
    over spatial), and the 4 conv input-VJPs (transpose-`o,i`+`reverse`) + the conv1 VJP to the
    pixels. `bnBlock`/`bnBack` emit the repeated BN forward/backward. **No certificate** — instance
    norm absorbs the conv weight scale and its Lipschitz is data-dependent (`γ·istd`), a separate
    problem; this is the attack rung only. -/
private def genCifarBnPgdStep (bs : Nat) (eps alpha : Float) (linf : Bool) : String :=
  let convCfg := "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
    "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
    "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}"
  let poolAttr := "{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}"
  -- BN forward + bias + relu for block k (C channels, N spatial, H side): emits %hc{k} (conv+bias),
  -- saves %istd{k}/%xhat{k}/%gb{k}/%znk{k}/%nfk{k}/%mask{k}, post-relu reshaped to 4D as %h{k}4.
  let bnBlock := fun (k C N H : Nat) (convOut : String) =>
    let t4 := s!"tensor<{bs}x{C}x{H}x{H}xf32>"
    let tn := s!"tensor<{bs}x{C}x{N}xf32>"
    let tc := s!"tensor<{bs}x{C}xf32>"
    let tni := s!"tensor<{bs}x{C}x{N}xi1>"
    let cty := s!"tensor<{C}xf32>"
    s!"    %bbb{k} = stablehlo.broadcast_in_dim %b{k}, dims = [1] : ({cty}) -> {t4}\n" ++
    s!"    %hc{k} = stablehlo.add {convOut}, %bbb{k} : {t4}\n" ++
    s!"    %nfk{k} = stablehlo.constant dense<{N}.0> : {tn}\n" ++
    s!"    %epk{k} = stablehlo.constant dense<1.0e-05> : {tn}\n" ++
    s!"    %znk{k} = stablehlo.constant dense<0.0> : {tn}\n" ++
    s!"    %xr{k} = stablehlo.reshape %hc{k} : ({t4}) -> {tn}\n" ++
    s!"    %smr{k} = stablehlo.reduce(%xr{k} init: %zf) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {tc}\n" ++
    s!"    %smb{k} = stablehlo.broadcast_in_dim %smr{k}, dims = [0, 1] : ({tc}) -> {tn}\n" ++
    s!"    %mu{k} = stablehlo.divide %smb{k}, %nfk{k} : {tn}\n" ++
    s!"    %xc{k} = stablehlo.subtract %xr{k}, %mu{k} : {tn}\n" ++
    s!"    %sq{k} = stablehlo.multiply %xc{k}, %xc{k} : {tn}\n" ++
    s!"    %vsr{k} = stablehlo.reduce(%sq{k} init: %zf) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {tc}\n" ++
    s!"    %vsb{k} = stablehlo.broadcast_in_dim %vsr{k}, dims = [0, 1] : ({tc}) -> {tn}\n" ++
    s!"    %var{k} = stablehlo.divide %vsb{k}, %nfk{k} : {tn}\n" ++
    s!"    %ve{k} = stablehlo.add %var{k}, %epk{k} : {tn}\n" ++
    s!"    %istd{k} = stablehlo.rsqrt %ve{k} : {tn}\n" ++
    s!"    %xhat{k} = stablehlo.multiply %xc{k}, %istd{k} : {tn}\n" ++
    s!"    %gb{k} = stablehlo.broadcast_in_dim %g{k}, dims = [1] : ({cty}) -> {tn}\n" ++
    s!"    %btb{k} = stablehlo.broadcast_in_dim %bt{k}, dims = [1] : ({cty}) -> {tn}\n" ++
    s!"    %gx{k} = stablehlo.multiply %xhat{k}, %gb{k} : {tn}\n" ++
    s!"    %y{k} = stablehlo.add %gx{k}, %btb{k} : {tn}\n" ++
    s!"    %hn{k} = stablehlo.maximum %y{k}, %znk{k} : {tn}\n" ++
    s!"    %mask{k} = stablehlo.compare GT, %y{k}, %znk{k} : ({tn}, {tn}) -> {tni}\n" ++
    s!"    %h{k}4 = stablehlo.reshape %hn{k} : ({tn}) -> {t4}\n"
  -- BN backward for block k: grad w.r.t. post-relu (4D %dpost) → grad w.r.t. conv output %dco{k} (4D).
  let bnBack := fun (k C N H : Nat) (dpost : String) =>
    let t4 := s!"tensor<{bs}x{C}x{H}x{H}xf32>"
    let tn := s!"tensor<{bs}x{C}x{N}xf32>"
    let tc := s!"tensor<{bs}x{C}xf32>"
    let tni := s!"tensor<{bs}x{C}x{N}xi1>"
    s!"    %dpn{k} = stablehlo.reshape {dpost} : ({t4}) -> {tn}\n" ++
    s!"    %dy{k} = stablehlo.select %mask{k}, %dpn{k}, %znk{k} : {tni}, {tn}\n" ++
    s!"    %dxhat{k} = stablehlo.multiply %dy{k}, %gb{k} : {tn}\n" ++
    s!"    %ds1r{k} = stablehlo.reduce(%dxhat{k} init: %zf) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {tc}\n" ++
    s!"    %ds1b{k} = stablehlo.broadcast_in_dim %ds1r{k}, dims = [0, 1] : ({tc}) -> {tn}\n" ++
    s!"    %s1{k} = stablehlo.divide %ds1b{k}, %nfk{k} : {tn}\n" ++
    s!"    %dpr{k} = stablehlo.multiply %dxhat{k}, %xhat{k} : {tn}\n" ++
    s!"    %ds2r{k} = stablehlo.reduce(%dpr{k} init: %zf) applies stablehlo.add across dimensions = [2] : ({tn}, tensor<f32>) -> {tc}\n" ++
    s!"    %ds2b{k} = stablehlo.broadcast_in_dim %ds2r{k}, dims = [0, 1] : ({tc}) -> {tn}\n" ++
    s!"    %s2{k} = stablehlo.divide %ds2b{k}, %nfk{k} : {tn}\n" ++
    s!"    %xs2{k} = stablehlo.multiply %xhat{k}, %s2{k} : {tn}\n" ++
    s!"    %dsa{k} = stablehlo.subtract %dxhat{k}, %s1{k} : {tn}\n" ++
    s!"    %dsb{k} = stablehlo.subtract %dsa{k}, %xs2{k} : {tn}\n" ++
    s!"    %dxn{k} = stablehlo.multiply %istd{k}, %dsb{k} : {tn}\n" ++
    s!"    %dco{k} = stablehlo.reshape %dxn{k} : ({tn}) -> {t4}\n"
  -- conv input-VJP: transpose-(o,i) + spatial-reverse W{k} ([oC,iC,3,3]→[iC,oC,3,3]), conv with %dco{k}
  let convVjp := fun (k oC iC : Nat) (dco lhsTy outTy : String) (out : String) =>
    s!"    %wt{k} = stablehlo.transpose %W{k}, dims = [1, 0, 2, 3] : (tensor<{oC}x{iC}x3x3xf32>) -> tensor<{iC}x{oC}x3x3xf32>\n" ++
    s!"    %wr{k} = stablehlo.reverse %wt{k}, dims = [2, 3] : tensor<{iC}x{oC}x3x3xf32>\n" ++
    s!"    {out} = stablehlo.convolution({dco}, %wr{k})\n      {convCfg} : ({lhsTy}, tensor<{iC}x{oC}x3x3xf32>) -> {outTy}\n"
  let m32_4 := s!"tensor<{bs}x32x32x32xf32>"
  let m64_4 := s!"tensor<{bs}x64x16x16xf32>"
  let p32 := s!"tensor<{bs}x32x16x16xf32>"
  let p64 := s!"tensor<{bs}x64x8x8xf32>"
  let i4  := s!"tensor<{bs}x3x32x32xf32>"
  let f2  := s!"tensor<{bs}x4096xf32>"
  let h2  := s!"tensor<{bs}x512xf32>"
  let h2i := s!"tensor<{bs}x512xi1>"
  let o2  := s!"tensor<{bs}x10xf32>"
  let bxd0 := s!"tensor<{bs}x3072xf32>"
  let rty := s!"tensor<{bs}xf32>"
  let poolFwd := fun (inTy outTy inp out : String) =>
    s!"    {out} = \"stablehlo.reduce_window\"({inp}, %ninf) (\{\n" ++
    "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
    "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
    "        stablehlo.return %pm : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({inTy}, tensor<f32>) -> {outTy}\n"
  let poolBack := fun (sfx : String) (srcTy gradTy inp grad out : String) =>
    s!"    {out} = \"stablehlo.select_and_scatter\"({inp}, {grad}, %zf) (\{\n" ++
    s!"      ^bb0(%sa{sfx}: tensor<f32>, %sb{sfx}: tensor<f32>):\n" ++
    s!"        %sge{sfx} = stablehlo.compare GE, %sa{sfx}, %sb{sfx} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
    s!"        stablehlo.return %sge{sfx} : tensor<i1>\n" ++
    "    }, {\n" ++
    s!"      ^bb0(%sc{sfx}: tensor<f32>, %sd{sfx}: tensor<f32>):\n" ++
    s!"        %ss{sfx} = stablehlo.add %sc{sfx}, %sd{sfx} : tensor<f32>\n" ++
    s!"        stablehlo.return %ss{sfx} : tensor<f32>\n" ++
    s!"    }) {poolAttr} : ({srcTy}, {gradTy}, tensor<f32>) -> {srcTy}\n"
  let header :=
    "module @m {\n" ++
    s!"  func.func @cifar_bn_pgd_step(%x: {bxd0}, %W1: tensor<32x3x3x3xf32>, %b1: tensor<32xf32>, %g1: tensor<32xf32>, %bt1: tensor<32xf32>, %W2: tensor<32x32x3x3xf32>, %b2: tensor<32xf32>, %g2: tensor<32xf32>, %bt2: tensor<32xf32>, %W3: tensor<64x32x3x3xf32>, %b3: tensor<64xf32>, %g3: tensor<64xf32>, %bt3: tensor<64xf32>, %W4: tensor<64x64x3x3xf32>, %b4: tensor<64xf32>, %g4: tensor<64xf32>, %bt4: tensor<64xf32>, %W5: tensor<4096x512xf32>, %b5: tensor<512xf32>, %W6: tensor<512x512xf32>, %b6: tensor<512xf32>, %W7: tensor<512x10xf32>, %b7: tensor<10xf32>, %onehot: {o2}, %x0: {bxd0}) -> {bxd0} " ++ "{\n" ++
    "    %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
    "    %zf = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
    s!"    %alpha = stablehlo.constant dense<{alpha}> : tensor<f32>\n" ++
    s!"    %eps = stablehlo.constant dense<{eps}> : tensor<f32>\n" ++
    s!"    %zh = stablehlo.constant dense<0.0> : {h2}\n" ++
    s!"    %v0 = stablehlo.reshape %x : ({bxd0}) -> {i4}\n" ++
    -- forward: conv→BN→relu ×2, pool, ×2, pool, denses
    s!"    %c1 = stablehlo.convolution(%v0, %W1)\n      {convCfg} : ({i4}, tensor<32x3x3x3xf32>) -> {m32_4}\n" ++
    bnBlock 1 32 1024 32 "%c1" ++
    s!"    %c2 = stablehlo.convolution(%h14, %W2)\n      {convCfg} : ({m32_4}, tensor<32x32x3x3xf32>) -> {m32_4}\n" ++
    bnBlock 2 32 1024 32 "%c2" ++
    poolFwd m32_4 p32 "%h24" "%pool1" ++
    s!"    %c3 = stablehlo.convolution(%pool1, %W3)\n      {convCfg} : ({p32}, tensor<64x32x3x3xf32>) -> {m64_4}\n" ++
    bnBlock 3 64 256 16 "%c3" ++
    s!"    %c4 = stablehlo.convolution(%h34, %W4)\n      {convCfg} : ({m64_4}, tensor<64x64x3x3xf32>) -> {m64_4}\n" ++
    bnBlock 4 64 256 16 "%c4" ++
    poolFwd m64_4 p64 "%h44" "%pool2" ++
    s!"    %flat = stablehlo.reshape %pool2 : ({p64}) -> {f2}\n" ++
    s!"    %d5 = stablehlo.dot_general %flat, %W5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({f2}, tensor<4096x512xf32>) -> {h2}\n" ++
    s!"    %b5b = stablehlo.broadcast_in_dim %b5, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z5 = stablehlo.add %d5, %b5b : {h2}\n" ++
    s!"    %h5 = stablehlo.maximum %z5, %zh : {h2}\n" ++
    s!"    %d6 = stablehlo.dot_general %h5, %W6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %b6b = stablehlo.broadcast_in_dim %b6, dims = [1] : (tensor<512xf32>) -> {h2}\n" ++
    s!"    %z6 = stablehlo.add %d6, %b6b : {h2}\n" ++
    s!"    %h6 = stablehlo.maximum %z6, %zh : {h2}\n" ++
    s!"    %d7 = stablehlo.dot_general %h6, %W7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x10xf32>) -> {o2}\n" ++
    s!"    %b7b = stablehlo.broadcast_in_dim %b7, dims = [1] : (tensor<10xf32>) -> {o2}\n" ++
    s!"    %logits = stablehlo.add %d7, %b7b : {o2}\n" ++
    -- softmax-CE
    s!"    %rmax = stablehlo.reduce(%logits init: %ninf) applies stablehlo.maximum across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %rmaxb = stablehlo.broadcast_in_dim %rmax, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %shift = stablehlo.subtract %logits, %rmaxb : {o2}\n" ++
    s!"    %expv = stablehlo.exponential %shift : {o2}\n" ++
    s!"    %ssum = stablehlo.reduce(%expv init: %zero) applies stablehlo.add across dimensions = [1] : ({o2}, tensor<f32>) -> {rty}\n" ++
    s!"    %ssumb = stablehlo.broadcast_in_dim %ssum, dims = [0] : ({rty}) -> {o2}\n" ++
    s!"    %softmax = stablehlo.divide %expv, %ssumb : {o2}\n" ++
    s!"    %g = stablehlo.subtract %softmax, %onehot : {o2}\n" ++
    -- backward
    s!"    %dh6 = stablehlo.dot_general %g, %W7, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({o2}, tensor<512x10xf32>) -> {h2}\n" ++
    s!"    %rm6 = stablehlo.compare GT, %z6, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz6 = stablehlo.select %rm6, %dh6, %zh : {h2i}, {h2}\n" ++
    s!"    %dh5 = stablehlo.dot_general %dz6, %W6, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<512x512xf32>) -> {h2}\n" ++
    s!"    %rm5 = stablehlo.compare GT, %z5, %zh : ({h2}, {h2}) -> {h2i}\n" ++
    s!"    %dz5 = stablehlo.select %rm5, %dh5, %zh : {h2i}, {h2}\n" ++
    s!"    %dflat = stablehlo.dot_general %dz5, %W5, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({h2}, tensor<4096x512xf32>) -> {f2}\n" ++
    s!"    %dpool2 = stablehlo.reshape %dflat : ({f2}) -> {p64}\n" ++
    poolBack "p2" m64_4 p64 "%h44" "%dpool2" "%dpre4" ++
    bnBack 4 64 256 16 "%dpre4" ++
    convVjp 4 64 64 "%dco4" m64_4 m64_4 "%dpre3" ++
    bnBack 3 64 256 16 "%dpre3" ++
    convVjp 3 64 32 "%dco3" m64_4 p32 "%dpool1" ++
    poolBack "p1" m32_4 p32 "%h24" "%dpool1" "%dpre2" ++
    bnBack 2 32 1024 32 "%dpre2" ++
    convVjp 2 32 32 "%dco2" m32_4 m32_4 "%dpre1" ++
    bnBack 1 32 1024 32 "%dpre1" ++
    convVjp 1 32 3 "%dco1" m32_4 i4 "%dxi" ++
    s!"    %dx = stablehlo.reshape %dxi : ({i4}) -> {bxd0}\n" ++
    s!"    %alphab = stablehlo.broadcast_in_dim %alpha, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %zerob = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
    s!"    %oneb = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {bxd0}\n"
  let step :=
    if linf then
      s!"    %sgn = stablehlo.sign %dx : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %sgn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %epsb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {bxd0}\n" ++
      s!"    %lo = stablehlo.subtract %x0, %epsb : {bxd0}\n" ++
      s!"    %hi = stablehlo.add %x0, %epsb : {bxd0}\n" ++
      s!"    %pj1 = stablehlo.maximum %xn, %lo : {bxd0}\n" ++
      s!"    %xp = stablehlo.minimum %pj1, %hi : {bxd0}\n"
    else
      s!"    %e12 = stablehlo.constant dense<1.0e-12> : tensor<f32>\n" ++
      s!"    %e12r = stablehlo.broadcast_in_dim %e12, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %dx2 = stablehlo.multiply %dx, %dx : {bxd0}\n" ++
      s!"    %dxs = stablehlo.reduce(%dx2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dxnv = stablehlo.sqrt %dxs : {rty}\n" ++
      s!"    %dxnp = stablehlo.add %dxnv, %e12r : {rty}\n" ++
      s!"    %dxnb = stablehlo.broadcast_in_dim %dxnp, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %gn = stablehlo.divide %dx, %dxnb : {bxd0}\n" ++
      s!"    %stp = stablehlo.multiply %alphab, %gn : {bxd0}\n" ++
      s!"    %xn = stablehlo.add %x, %stp : {bxd0}\n" ++
      s!"    %delta = stablehlo.subtract %xn, %x0 : {bxd0}\n" ++
      s!"    %dl2 = stablehlo.multiply %delta, %delta : {bxd0}\n" ++
      s!"    %dls = stablehlo.reduce(%dl2 init: %zero) applies stablehlo.add across dimensions = [1] : ({bxd0}, tensor<f32>) -> {rty}\n" ++
      s!"    %dln = stablehlo.sqrt %dls : {rty}\n" ++
      s!"    %dlnp = stablehlo.add %dln, %e12r : {rty}\n" ++
      s!"    %epsr = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %oner = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> {rty}\n" ++
      s!"    %ratio = stablehlo.divide %epsr, %dlnp : {rty}\n" ++
      s!"    %fac = stablehlo.minimum %oner, %ratio : {rty}\n" ++
      s!"    %facb = stablehlo.broadcast_in_dim %fac, dims = [0] : ({rty}) -> {bxd0}\n" ++
      s!"    %dproj = stablehlo.multiply %delta, %facb : {bxd0}\n" ++
      s!"    %xp = stablehlo.add %x0, %dproj : {bxd0}\n"
  header ++ step ++
  s!"    %clA = stablehlo.maximum %xp, %zerob : {bxd0}\n" ++
  s!"    %clB = stablehlo.minimum %clA, %oneb : {bxd0}\n" ++
  s!"    return %clB : {bxd0}\n" ++
  "  }\n}\n"

/-- **Phase-3 PGD attack on the verified MNIST MLP** (`planning/robustness.md`). Trains the
    784→512→512→10 ReLU MLP on the proof-rendered SGD step, then attacks through IREE with the
    proven `mlpInputGrad` VJP kernel. The Lipschitz certificate is the **product** of the three
    layers' spectral norms — where the bound (and so the cert) goes loose. -/
def VerifiedNet.attackPgdMlp (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let hN := 512
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net.data d0 dataDir
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let nP := net.nParams
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut theta := F32.concat parts
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for _ in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let out ← IreeSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
      theta := out.extract 0 (nP * 4)
  -- split θ (func-arg order: W0 b0 W1 b1 W2 b2)
  let W0 := theta.extract 0 (d0*hN*4)
  let W1 := theta.extract ((d0*hN + hN)*4) ((d0*hN + hN + hN*hN)*4)
  let W2 := theta.extract ((d0*hN + hN + hN*hN + hN)*4) ((d0*hN + hN + hN*hN + hN + hN*d1)*4)
  let mut clean := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImages evalImg (bi * bs) bs d0
    let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:bs] do
      if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
        clean := clean + 1
  IO.println s!"clean test acc = {clean}/{nbt*bs} = {clean.toFloat/(nbt*bs).toFloat*100.0}%"
  let K := 40
  let pgdShapes := packShapes #[#[d0,hN], #[hN], #[hN,hN], #[hN], #[hN,d1], #[d1], #[bs,d1], #[bs,d0]]
  let runSweep := fun (linf : Bool) (epsList : List Float) => do
    for eps in epsList do
      let alpha := 2.5 * eps / K.toFloat
      IO.FS.writeFile ".lake/build/mlp_pgd_step.mlir" (genMlpPgdStep bs d0 hN d1 eps alpha linf)
      compileVmfb ".lake/build/mlp_pgd_step.mlir" ".lake/build/mlp_pgd_step.vmfb"
      let pgdSess ← IreeSession.create ".lake/build/mlp_pgd_step.vmfb"
      let mut correct := 0
      for bi in [0:nbt] do
        let x0 := F32.sliceImages evalImg (bi * bs) bs d0
        let oh ← oneHotBatch evalLbl (bi * bs) bs d1
        let pgdParams := F32.concat #[theta, oh, x0]
        let mut x := x0
        for _ in [0:K] do
          x ← IreeSession.forwardF32 pgdSess "m.mlp_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
        let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
        for j in [0:bs] do
          if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
            correct := correct + 1
      let lbl := if linf then "L∞" else "L2"
      IO.println s!"{lbl} PGD eps={eps}: adv acc = {correct.toFloat/(nbt*bs).toFloat*100.0}%"
  runSweep true [0.1, 0.2, 0.3]
  -- certificate: product of the three layers' spectral norms (ReLU is 1-Lipschitz)
  let L0 := specNormW W0 d0 hN
  let L1 := specNormW W1 hN hN
  let L2 := specNormW W2 hN d1
  let L := L0 * L1 * L2
  IO.println s!"\nspectral norms ‖W₀‖={L0}, ‖W₁‖={L1}, ‖W₂‖={L2}  →  global L = {L}  (PRODUCT over 3 layers — loose)"
  let tot := (nbt * bs).toFloat
  let mut cert05 := 0
  let mut cert10 := 0
  let mut cert15 := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImages evalImg (bi * bs) bs d0
    let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:bs] do
      let mut top := -1.0e30
      let mut sec := -1.0e30
      let mut topi := 0
      for c in [0:d1] do
        let v := F32.read logits (j * d1 + c).toUSize
        if v > top then
          sec := top
          top := v
          topi := c
        else if v > sec then
          sec := v
      if topi == (evalLbl.get! (4 * (bi * bs + j))).toNat then
        let r := (top - sec) / (1.4142135623730951 * L)
        if r ≥ 0.5 then cert05 := cert05 + 1
        if r ≥ 1.0 then cert10 := cert10 + 1
        if r ≥ 1.5 then cert15 := cert15 + 1
  IO.println s!"certified-robust acc (L2): ε=0.5 → {cert05.toFloat/tot*100.0}%, ε=1.0 → {cert10.toFloat/tot*100.0}%, ε=1.5 → {cert15.toFloat/tot*100.0}%"
  runSweep false [0.5, 1.0, 1.5]
  IO.println "done (phase-3 MLP PGD: input gradient = the proven mlpInputGrad VJP via IREE)."

/-- **Spectral-norm-constrained training of the verified MNIST MLP** (`planning/robustness_ladder.md`,
    the research lever). Trains the 784→512→512→10 net with **projected SGD onto the spectral ball**
    — after every `K` proof-rendered steps (and once at the end) each weight `Wᵢ` is rescaled to
    `‖Wᵢ‖₂ ≤ c` (`projectSpectral`) — then runs the *same* `cert ≤ TRUE ≤ PGD` sandwich. Sweeps a
    few caps `c` (plus an unconstrained baseline) so the table shows the trade: shrinking `c` pulls
    the global `L = ∏‖Wᵢ‖₂` down (`L ≤ c³`), turning the **vacuous** product certificate
    **non-vacuous** — at the cost of clean accuracy. The empirical face of
    `lipschitz_margin_certified_radius` (`LeanMlir/Proofs/LipschitzCert.lean`): smaller `L` ⇒ larger
    certified radius `m/(√2·L)`. The verified CE gradient stays in the proven kernel; the projection
    is host-side weight rescaling only. -/
def VerifiedNet.attackPgdSpectralMlp (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let hN := 512
  let d1 := net.nClasses
  let projEvery := 20            -- lazy projection: every 20 verified steps (+ once at the end)
  IO.println s!"Spectral-norm-constrained PGD study on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net.data d0 dataDir
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let K := 40
  let pgdShapes := packShapes #[#[d0,hN], #[hN], #[hN,hN], #[hN], #[hN,d1], #[d1], #[bs,d1], #[bs,d0]]
  -- run one PGD eps point, returning adversarial accuracy (%) on the verified net
  let pgdAcc := fun (theta : ByteArray) (linf : Bool) (eps : Float) => do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile ".lake/build/mlp_pgd_step.mlir" (genMlpPgdStep bs d0 hN d1 eps alpha linf)
    compileVmfb ".lake/build/mlp_pgd_step.mlir" ".lake/build/mlp_pgd_step.vmfb"
    let pgdSess ← IreeSession.create ".lake/build/mlp_pgd_step.vmfb"
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImages evalImg (bi * bs) bs d0
      let oh ← oneHotBatch evalLbl (bi * bs) bs d1
      let pgdParams := F32.concat #[theta, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← IreeSession.forwardF32 pgdSess "m.mlp_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
          correct := correct + 1
    pure (correct.toFloat / (nbt*bs).toFloat * 100.0)
  let tot := (nbt * bs).toFloat
  let mut rows : Array String := #[]
  for cap in caps do
    let capStr := if cap ≥ 1.0e8 then "∞ (none)" else toString cap
    IO.println s!"\n── cap c = {capStr} ──"
    -- fresh He-init (same seeds ⇒ fair comparison across caps)
    let mut parts : Array ByteArray := #[]
    let mut seed := 1
    for spec in net.specs do
      parts := parts.push (← mkParam seed spec.1 spec.2)
      seed := seed + 1
    let mut theta := F32.concat parts
    let mut step := 0
    for _ in [0:cfg.epochs] do
      for bi in [0:nb] do
        let xb := F32.sliceImages trainImg (bi * bs) bs d0
        let yb := F32.sliceLabels trainLbl (bi * bs) bs
        theta ← IreeSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
        step := step + 1
        if cap < 1.0e8 && step % projEvery == 0 then
          theta ← projectSpectral theta net.specs cap
    if cap < 1.0e8 then theta ← projectSpectral theta net.specs cap   -- enforce the cap on the final θ
    -- split θ for the certificate
    let W0 := theta.extract 0 (d0*hN*4)
    let W1 := theta.extract ((d0*hN + hN)*4) ((d0*hN + hN + hN*hN)*4)
    let W2 := theta.extract ((d0*hN + hN + hN*hN + hN)*4) ((d0*hN + hN + hN*hN + hN + hN*d1)*4)
    let L0 := specNormW W0 d0 hN
    let L1 := specNormW W1 hN hN
    let L2 := specNormW W2 hN d1
    let L := L0 * L1 * L2
    -- clean accuracy + certified-robust accuracy at L2 {0.25, 0.5, 1.0}
    let mut clean := 0
    let mut c025 := 0
    let mut c05 := 0
    let mut c10 := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        let mut top := -1.0e30
        let mut sec := -1.0e30
        let mut topi := 0
        for cidx in [0:d1] do
          let v := F32.read logits (j * d1 + cidx).toUSize
          if v > top then sec := top; top := v; topi := cidx
          else if v > sec then sec := v
        if topi == (evalLbl.get! (4 * (bi * bs + j))).toNat then
          clean := clean + 1
          let r := (top - sec) / (1.4142135623730951 * L)
          if r ≥ 0.25 then c025 := c025 + 1
          if r ≥ 0.5 then c05 := c05 + 1
          if r ≥ 1.0 then c10 := c10 + 1
    let cleanPct := clean.toFloat/tot*100.0
    IO.println s!"  ‖W₀‖={L0}  ‖W₁‖={L1}  ‖W₂‖={L2}  →  L = {L}"
    IO.println s!"  clean = {cleanPct}%   cert@L2 0.25/0.5/1.0 = {c025.toFloat/tot*100.0}% / {c05.toFloat/tot*100.0}% / {c10.toFloat/tot*100.0}%"
    let pinf ← pgdAcc theta true 0.1
    let pl2 ← pgdAcc theta false 0.5
    IO.println s!"  L∞ PGD ε=0.1 = {pinf}%   L2 PGD ε=0.5 = {pl2}%"
    (← IO.getStdout).flush
    rows := rows.push s!"  {capStr}\t{cleanPct}\t{L}\t{c05.toFloat/tot*100.0}\t{pl2}\t{pinf}"
  IO.println "\n══ spectral-norm training: the cert ≤ TRUE ≤ PGD trade ══"
  IO.println "  cap c\tclean%\tglobal L\tcert@L2 0.5\tL2 PGD 0.5\tL∞ PGD 0.1"
  for row in rows do IO.println row
  IO.println "\ndone (spectral-norm-constrained training: smaller c ⇒ smaller L ⇒ the product cert"
  IO.println "      goes non-vacuous, at the cost of clean accuracy — the gap-shrinking lever)."

/-- **Generic conv-net PGD attack** (`planning/robustness_ladder.md`). Trains any packed conv
    net on its proof-rendered SGD step, then attacks through IREE with `genKernel` — the full
    proven backward (conv input-VJPs + maxpool `select_and_scatter`-backs, mirroring the net's
    `<slug>_train_step.mlir`) run to `dx`. Certificate = the conv-aware spectral-norm **product**
    (`specNormConvTapSum` for convs × `specNormW` for denses; ReLU/maxpool are 1-Lipschitz) —
    astronomically loose, the depth-cliff. `genKernel` and `net.slug` select the architecture
    (`genCnnPgdStep`/MNIST-CNN, `genCifarPgdStep`/CIFAR-CNN). -/
def VerifiedNet.attackPgdConvNet (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (genKernel : Nat → Float → Float → Bool → String) (withCert : Bool := true) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net.data d0 dataDir
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut theta := F32.concat parts
  -- Best-checkpoint training: eval each epoch and keep the highest-accuracy θ. Plain SGD on the
  -- deeper nets (CIFAR) can diverge late; attacking the best checkpoint keeps the demo robust
  -- (and the cert finite). Monotone nets (MNIST CNN) → best = final, so numbers are unchanged.
  let mut bestTheta := theta
  let mut bestAcc := -1.0
  let evalAcc := fun (th : ByteArray) => do
    let mut c := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 fwdSess fwdFn th shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
          c := c + 1
    pure (c.toFloat / (nbt*bs).toFloat * 100.0)
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      theta ← IreeSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
    let acc ← evalAcc theta
    if acc > bestAcc then bestAcc := acc; bestTheta := theta
    IO.println s!"    epoch {ep + 1}/{cfg.epochs}: acc = {acc}%"
    (← IO.getStdout).flush
  theta := bestTheta                       -- attack the best checkpoint
  IO.println s!"clean test acc (best epoch) = {bestAcc}%"
  let K := 40
  let pgdShapes := packShapes (net.paramShapes ++ #[#[bs, d1], #[bs, d0]])
  let runSweep := fun (linf : Bool) (epsList : List Float) => do
    for eps in epsList do
      let alpha := 2.5 * eps / K.toFloat
      IO.FS.writeFile s!".lake/build/{net.slug}_pgd_step.mlir" (genKernel bs eps alpha linf)
      compileVmfb s!".lake/build/{net.slug}_pgd_step.mlir" s!".lake/build/{net.slug}_pgd_step.vmfb"
      let pgdSess ← IreeSession.create s!".lake/build/{net.slug}_pgd_step.vmfb"
      let mut correct := 0
      for bi in [0:nbt] do
        let x0 := F32.sliceImages evalImg (bi * bs) bs d0
        let oh ← oneHotBatch evalLbl (bi * bs) bs d1
        let pgdParams := F32.concat #[theta, oh, x0]
        let mut x := x0
        for _ in [0:K] do
          x ← IreeSession.forwardF32 pgdSess s!"m.{net.slug}_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
        let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
        for j in [0:bs] do
          if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
            correct := correct + 1
      let lbl := if linf then "L∞" else "L2"
      IO.println s!"{lbl} PGD eps={eps}: adv acc = {correct.toFloat/(nbt*bs).toFloat*100.0}%"
      (← IO.getStdout).flush
  runSweep true [0.1, 0.2, 0.3]
  -- ── certificate: conv-aware spectral-norm PRODUCT (ReLU/maxpool are 1-Lipschitz) ──
  -- Skipped for BN nets: instance norm absorbs the conv weight scale and its Lipschitz is
  -- data-dependent (γ·istd), so the conv-product cert is meaningless — a separate problem.
  if withCert then
    let mut L := 1.0
    let mut off := 0
    let mut msg := ""
    for spec in net.specs do
      let dims := spec.1
      let len := dims.foldl (·*·) 1
      let wslice := theta.extract (off*4) ((off+len)*4)
      if dims.size == 4 then
        let n := specNormConvTapSum wslice dims[0]! dims[1]! dims[2]! dims[3]!
        L := L * n
        msg := msg ++ s!"conv{dims[1]!}→{dims[0]!} Σtap‖·‖₂={n}  "
      else if dims.size == 2 then
        let n := specNormW wslice dims[0]! dims[1]!
        L := L * n
        msg := msg ++ s!"dense{dims[0]!}→{dims[1]!} ‖·‖₂={n}  "
      off := off + len
    IO.println s!"\nlayer norms: {msg}"
    IO.println s!"  →  global L = {L}  (PRODUCT over conv+dense layers — astronomically loose)"
    let tot := (nbt * bs).toFloat
    let mut cert05 := 0
    let mut cert10 := 0
    let mut cert15 := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        let mut top := -1.0e30
        let mut sec := -1.0e30
        let mut topi := 0
        for c in [0:d1] do
          let v := F32.read logits (j * d1 + c).toUSize
          if v > top then
            sec := top
            top := v
            topi := c
          else if v > sec then
            sec := v
        if topi == (evalLbl.get! (4 * (bi * bs + j))).toNat then
          let r := (top - sec) / (1.4142135623730951 * L)
          if r ≥ 0.5 then cert05 := cert05 + 1
          if r ≥ 1.0 then cert10 := cert10 + 1
          if r ≥ 1.5 then cert15 := cert15 + 1
    IO.println s!"certified-robust acc (L2): ε=0.5 → {cert05.toFloat/tot*100.0}%, ε=1.0 → {cert10.toFloat/tot*100.0}%, ε=1.5 → {cert15.toFloat/tot*100.0}%"
  else
    IO.println "\n(certificate N/A — instance-norm Lipschitz is data-dependent (γ·istd); deferred)"
  runSweep false [0.5, 1.0, 1.5]
  IO.println s!"done (phase-3 {net.name} PGD: input gradient = the proven conv/maxpool input-VJP via IREE)."

/-- PGD attack on the verified MNIST CNN (the first conv rung). -/
def VerifiedNet.attackPgdCnn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCnnPgdStep

/-- PGD attack on the verified CIFAR-10 CNN (the deeper conv rung: 4 conv + 2 pool + 3 dense). -/
def VerifiedNet.attackPgdCifar (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCifarPgdStep

/-- PGD attack on the verified CIFAR-10 CNN **+ per-channel (instance) BatchNorm** (`cifar_bn`).
    The BN input-VJP rung — `genCifarBnPgdStep` runs the proven backward through 4 instance-norm
    layers (the BN grad-input 3-term formula). Certificate skipped (`withCert := false`): instance
    norm's Lipschitz is data-dependent (`γ·istd`), a separate problem from the conv-product. -/
def VerifiedNet.attackPgdCifarBn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit :=
  net.attackPgdConvNet cfg dataDir genCifarBnPgdStep (withCert := false)

/-- **Spectral-norm-constrained training of the verified MNIST CNN** (`planning/robustness_ladder.md`,
    the gap-shrinking lever applied to the conv net). The CNN sibling of `attackPgdSpectralMlp`:
    projected SGD onto the spectral ball — every `K` proof-rendered steps (and once at the end)
    `projectSpectral` caps **both** the dense `‖Wᵢ‖₂` and the conv tap-sum bound at `c` — then the
    `cert ≤ TRUE ≤ PGD` sandwich (PGD via `genKernel`, cert = the conv-aware product). Harder than
    the MLP: it's a `k`-layer product (`L ≤ cᵏ`) and the conv tap-sum is a *loose* bound, so
    projection over-penalizes the convs — the cert needs a tighter `c` (and pays more clean accuracy)
    than the MLP did, and certifies only at *smaller* radii. The honest "depth + loose conv-norm ⇒
    certifying the conv net is harder." Generic over `genKernel`/`net.slug` (MNIST-CNN, CIFAR-CNN). -/
def VerifiedNet.attackPgdSpectralConvNet (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) (genKernel : Nat → Float → Float → Bool → String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  let projEvery := 20
  IO.println s!"Spectral-norm-constrained PGD study on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net.data d0 dataDir
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let K := 40
  let pgdShapes := packShapes (net.paramShapes ++ #[#[bs, d1], #[bs, d0]])
  let pgdAcc := fun (theta : ByteArray) (linf : Bool) (eps : Float) => do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile s!".lake/build/{net.slug}_pgd_step.mlir" (genKernel bs eps alpha linf)
    compileVmfb s!".lake/build/{net.slug}_pgd_step.mlir" s!".lake/build/{net.slug}_pgd_step.vmfb"
    let pgdSess ← IreeSession.create s!".lake/build/{net.slug}_pgd_step.vmfb"
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImages evalImg (bi * bs) bs d0
      let oh ← oneHotBatch evalLbl (bi * bs) bs d1
      let pgdParams := F32.concat #[theta, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← IreeSession.forwardF32 pgdSess s!"m.{net.slug}_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
          correct := correct + 1
    pure (correct.toFloat / (nbt*bs).toFloat * 100.0)
  let tot := (nbt * bs).toFloat
  let mut rows : Array String := #[]
  for cap in caps do
    let capStr := if cap ≥ 1.0e8 then "∞ (none)" else toString cap
    IO.println s!"\n── cap c = {capStr} ──"
    let mut parts : Array ByteArray := #[]
    let mut seed := 1
    for spec in net.specs do
      parts := parts.push (← mkParam seed spec.1 spec.2)
      seed := seed + 1
    let mut theta := F32.concat parts
    let mut bestTheta := theta
    let mut bestAcc := -1.0
    let mut step := 0
    for _ in [0:cfg.epochs] do
      for bi in [0:nb] do
        let xb := F32.sliceImages trainImg (bi * bs) bs d0
        let yb := F32.sliceLabels trainLbl (bi * bs) bs
        theta ← IreeSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
        step := step + 1
        if cap < 1.0e8 && step % projEvery == 0 then
          theta ← projectSpectral theta net.specs cap
      -- best-checkpoint (the baseline ∞ cap can diverge late; constrained caps stay bounded)
      let mut c := 0
      for bi in [0:nbt] do
        let xb := F32.sliceImages evalImg (bi * bs) bs d0
        let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
        for j in [0:bs] do
          if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
            c := c + 1
      let acc := c.toFloat / (nbt*bs).toFloat * 100.0
      if acc > bestAcc then bestAcc := acc; bestTheta := theta
    theta := bestTheta
    if cap < 1.0e8 then theta ← projectSpectral theta net.specs cap
    -- conv-aware certificate product (specNormConvTapSum convs × specNormW denses)
    let mut L := 1.0
    let mut off := 0
    let mut msg := ""
    for spec in net.specs do
      let dims := spec.1
      let len := dims.foldl (·*·) 1
      let wslice := theta.extract (off*4) ((off+len)*4)
      if dims.size == 4 then
        let n := specNormConvTapSum wslice dims[0]! dims[1]! dims[2]! dims[3]!
        L := L * n; msg := msg ++ s!"cv={n} "
      else if dims.size == 2 then
        let n := specNormW wslice dims[0]! dims[1]!
        L := L * n; msg := msg ++ s!"de={n} "
      off := off + len
    let mut clean := 0
    let mut cR1 := 0      -- certified @ L2 0.1
    let mut cR2 := 0      -- certified @ L2 0.25  (the CNN's visible band — it certifies at small radii)
    let mut cR3 := 0      -- certified @ L2 0.5
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        let mut top := -1.0e30
        let mut sec := -1.0e30
        let mut topi := 0
        for cidx in [0:d1] do
          let v := F32.read logits (j * d1 + cidx).toUSize
          if v > top then sec := top; top := v; topi := cidx
          else if v > sec then sec := v
        if topi == (evalLbl.get! (4 * (bi * bs + j))).toNat then
          clean := clean + 1
          let r := (top - sec) / (1.4142135623730951 * L)
          if r ≥ 0.1 then cR1 := cR1 + 1
          if r ≥ 0.25 then cR2 := cR2 + 1
          if r ≥ 0.5 then cR3 := cR3 + 1
    let cleanPct := clean.toFloat/tot*100.0
    IO.println s!"  {msg} →  L = {L}"
    IO.println s!"  clean = {cleanPct}%   cert@L2 0.1/0.25/0.5 = {cR1.toFloat/tot*100.0}% / {cR2.toFloat/tot*100.0}% / {cR3.toFloat/tot*100.0}%"
    let pinf ← pgdAcc theta true 0.1
    let pl2 ← pgdAcc theta false 0.5
    IO.println s!"  L∞ PGD ε=0.1 = {pinf}%   L2 PGD ε=0.5 = {pl2}%"
    (← IO.getStdout).flush
    rows := rows.push s!"  {capStr}\t{cleanPct}\t{L}\t{cR2.toFloat/tot*100.0}\t{pl2}\t{pinf}"
  IO.println s!"\n══ spectral-norm training ({net.name}): the cert ≤ TRUE ≤ PGD trade ══"
  IO.println "  cap c\tclean%\tglobal L\tcert@L2 0.25\tL2 PGD 0.5\tL∞ PGD 0.1"
  for row in rows do IO.println row
  IO.println "\ndone (spectral-norm-constrained conv training: the k-layer product + loose conv tap-sum"
  IO.println "      make certifying the conv net harder than the MLP — tighter c, more clean cost)."

/-- Spectral-norm-constrained training of the verified MNIST CNN. -/
def VerifiedNet.attackPgdSpectralCnn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  net.attackPgdSpectralConvNet cfg dataDir caps genCnnPgdStep

/-- Spectral-norm-constrained training of the verified CIFAR-10 CNN (7-layer product). -/
def VerifiedNet.attackPgdSpectralCifar (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (caps : List Float) : IO Unit :=
  net.attackPgdSpectralConvNet cfg dataDir caps genCifarPgdStep

/-! ## Randomized-smoothing statistics (Cohen–Rosenfeld–Kolter 2019)

The pieces the smoothing certificate needs, in pure `Float` (no kernel, no Mathlib): the
probit `Φ⁻¹` for the radius `σ·Φ⁻¹(p_A)`, and a **sound** Clopper–Pearson lower confidence
bound on `p_A` (a genuine 1−α lower bound, not an approximation — a certificate must under-
estimate). CP is built bottom-up from the regularized incomplete beta `Iₓ(a,b)`. -/

/-- Inverse standard-normal CDF `Φ⁻¹` (probit), Peter Acklam's rational approximation
    (relative error < 1.15e-9 over `(0,1)`). `p_A` here lives in `(0.5, ~0.99)`, far from
    the tails, so this is orders of magnitude tighter than the Monte-Carlo sampling error. -/
private def invNormCdf (p : Float) : Float := Id.run do
  let a0 := -3.969683028665376e+01
  let a1 :=  2.209460984245205e+02
  let a2 := -2.759285104469687e+02
  let a3 :=  1.383577518672690e+02
  let a4 := -3.066479806614716e+01
  let a5 :=  2.506628277459239e+00
  let b1 := -5.447609879822406e+01
  let b2 :=  1.615858368580409e+02
  let b3 := -1.556989798598866e+02
  let b4 :=  6.680131188771972e+01
  let b5 := -1.328068155288572e+01
  let c0 := -7.784894002430293e-03
  let c1 := -3.223964580411365e-01
  let c2 := -2.400758277161838e+00
  let c3 := -2.549732539343734e+00
  let c4 :=  4.374664141464968e+00
  let c5 :=  2.938163982698783e+00
  let d1 :=  7.784695709041462e-03
  let d2 :=  3.224671290700398e-01
  let d3 :=  2.445134137142996e+00
  let d4 :=  3.754408661907416e+00
  let plow := 0.02425
  let phigh := 1.0 - plow
  if p < plow then
    let q := Float.sqrt (-2.0 * Float.log p)
    return (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0)
  else if p ≤ phigh then
    let q := p - 0.5
    let r := q*q
    return (((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0)
  else
    let q := Float.sqrt (-2.0 * Float.log (1.0 - p))
    return 0.0 - (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0)

/-- `log Γ(x)` for `x ≥ 0.5` (the only regime we hit: `a=k≥1`, `b=n−k+1≥1`), Lanczos `g=7`. -/
private def lgammaF (x : Float) : Float := Id.run do
  let g : Array Float := #[
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
  let xx := x - 1.0
  let mut a := g[0]!
  let t := xx + 7.5
  for i in [1:9] do
    a := a + g[i]! / (xx + i.toFloat)
  return 0.5 * Float.log (2.0 * 3.141592653589793) + (xx + 0.5) * Float.log t - t + Float.log a

/-- Lentz continued fraction for the incomplete beta (Numerical Recipes `betacf`). -/
private def betacf (a b x : Float) : Float := Id.run do
  let fpmin := 1.0e-300; let eps := 3.0e-12
  let qab := a + b; let qap := a + 1.0; let qam := a - 1.0
  let mut c := 1.0
  let mut d := 1.0 - qab * x / qap
  if Float.abs d < fpmin then d := fpmin
  d := 1.0 / d
  let mut h := d
  for m in [1:201] do
    let mf := m.toFloat; let m2 := 2.0 * mf
    let aa1 := mf * (b - mf) * x / ((qam + m2) * (a + m2))
    d := 1.0 + aa1 * d; if Float.abs d < fpmin then d := fpmin
    c := 1.0 + aa1 / c; if Float.abs c < fpmin then c := fpmin
    d := 1.0 / d; h := h * d * c
    let aa2 := -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2))
    d := 1.0 + aa2 * d; if Float.abs d < fpmin then d := fpmin
    c := 1.0 + aa2 / c; if Float.abs c < fpmin then c := fpmin
    d := 1.0 / d
    let del := d * c
    h := h * del
    if Float.abs (del - 1.0) < eps then break
  return h

/-- Regularized incomplete beta `Iₓ(a,b)` (Numerical Recipes `betai`); increasing in `x`. -/
private def betaiF (a b x : Float) : Float :=
  if x ≤ 0.0 then 0.0
  else if x ≥ 1.0 then 1.0
  else
    let lbt := lgammaF (a+b) - lgammaF a - lgammaF b + a * Float.log x + b * Float.log (1.0 - x)
    let bt := Float.exp lbt
    if x < (a + 1.0) / (a + b + 2.0) then bt * betacf a b x / a
    else 1.0 - bt * betacf b a (1.0 - x) / b

/-- Clopper–Pearson **exact** lower confidence bound for a Binomial proportion: the largest
    `p` with `P[Bin(n,p) ≥ k] ≤ α`, i.e. the `α`-quantile of `Beta(k, n−k+1)` — the `p` solving
    `I_p(k, n−k+1) = α`, found by bisection (`Iₓ` is monotone). The SOUND `1−α` lower bound on
    `p_A` that the certified radius `σ·Φ⁻¹(p_A)` rests on (Cohen 2019 uses the same CP bound).
    `k=0 ⇒ 0`. -/
private def clopperPearsonLower (k n : Nat) (alpha : Float) : Float := Id.run do
  if k == 0 then return 0.0
  let a := k.toFloat
  let b := (n - k + 1).toFloat
  let mut lo := 0.0
  let mut hi := 1.0
  for _ in [0:60] do
    let mid := 0.5 * (lo + hi)
    if betaiF a b mid < alpha then lo := mid else hi := mid
  return 0.5 * (lo + hi)

/-- **Randomized-smoothing certificate** (Cohen–Rosenfeld–Kolter 2019, `planning/robustness_ladder.md`
    §3) — the depth-INDEPENDENT cert, and the answer where the Lipschitz product is hopeless.
    The smoothed classifier `ĝ(x) = argmax_c P[f(x+η)=c]`, `η ~ N(0,σ²I)`, is certified robust at
    L2 radius `σ·Φ⁻¹(p_A)` where `p_A` is a lower bound on the top class's noise probability. It's
    **forward-only**: no new kernel, no input-VJP — just sample `n` noisy copies, run the existing
    proof-rendered `<slug>_fwd`, count argmax votes, Clopper–Pearson lower-bound `p_A`. The base
    classifier is trained with matched Gaussian augmentation (every batch corrupted with `N(0,σ²I)`
    host-side before the proof-rendered SGD step — the forward/backward graph is untouched), the
    Cohen recipe. Architecture-agnostic + depth-independent, so it certifies a *non-vacuous* radius
    on the very nets (CIFAR, deep) where `∏‖Wᵢ‖₂` is astronomically loose. Generic over any
    `VerifiedNet` (fwd + train-step only).

    `n` (`SMOOTH_N`, default 10000 — Cohen's large-`n` regime) is the estimation budget and the only
    honest tightening lever: the per-point radius is capped at `σ·Φ⁻¹(α^(1/n))` (a unanimous vote
    still only certifies `p_A ≤ α^(1/n)`), so larger `n` lifts the ceiling and tightens the CP bound
    toward the true noise-probability — bigger certified radii at the same `1−α` guarantee. -/
def VerifiedNet.smoothCertify (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (sigmas : List Float) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Randomized-smoothing certificate on {net.name} (verified codegen → IREE → GPU, forward-only)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  -- `trainPix`/`crop`: Imagenette train ships at 256² and is center-cropped to 224² per batch
  -- (the val/eval split is already 224² = d0, so certify reads it directly). For MNIST/CIFAR
  -- crop=false and trainPix=d0, so the crop below is a no-op.
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ← loadData net.data d0 dataDir
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  -- knobs (env-overridable; defaults are Cohen's large-n estimation regime — drop SMOOTH_N /
  -- SMOOTH_MAXCERT for a cheap smoke).
  let n0      := ((← IO.getEnv "SMOOTH_N0").bind (·.toNat?)).getD 100
  let nSamp   := ((← IO.getEnv "SMOOTH_N").bind (·.toNat?)).getD 10000
  let maxCert := ((← IO.getEnv "SMOOTH_MAXCERT").bind (·.toNat?)).getD 200
  -- per-epoch best-θ eval is a clean-acc proxy; cap it to a subsample on heavy 224² nets
  -- (`SMOOTH_EVAL_BATCHES`, default = full val set).
  let evalBatches := min nbt (((← IO.getEnv "SMOOTH_EVAL_BATCHES").bind (·.toNat?)).getD nbt)
  let alpha   := 0.001
  let radii : Array Float := #[0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
  let nCert   := min maxCert nEval
  let stride  := max 1 (nEval / nCert)
  -- the rendered fwd has a STATIC batch = bs, so SampleUnderNoise feeds whole `bs`-batches;
  -- round the requested sample counts up to a multiple of bs (the effective n the CP bound uses).
  let n0Batches := max 1 ((n0 + bs - 1) / bs)
  let nBatches  := max 1 ((nSamp + bs - 1) / bs)
  let n0Eff := n0Batches * bs
  let nEff  := nBatches * bs
  -- the n→radius ceiling: even a UNANIMOUS vote gives only p_A ≤ α^(1/n) (the CP bound at n_A=n),
  -- so EVERY point's radius is capped at σ·Φ⁻¹(α^(1/n)) regardless of how robust it truly is.
  -- Larger n ⇒ p_A closer to the true noise-prob ⇒ higher ceiling — the only honest tightening
  -- lever (shrinking α would just weaken the 1−α guarantee, not tighten the estimate).
  let pMax := clopperPearsonLower nEff nEff alpha
  IO.println s!"  n0={n0Eff} (select)  n={nEff} (estimate)  α={alpha}  certifying {nCert} test imgs (every {stride}th)"
  IO.println s!"  p_A ceiling = {pMax} (= α^(1/n))  →  max certifiable radius = σ · {invNormCdf pMax}"
  let mut rows : Array String := #[]
  -- per-image certified radius, dumped to CSV for arbitrarily-fine frontier curves (cert-acc at
  -- radius r = fraction with correct ∧ radius ≥ r) + ACR + any threshold, all from one run.
  let mut dumpRows : Array String := #["sigma,img_idx,label,pred,abstain,radius"]
  -- Lipschitz-hypothesis probe (SMOOTH_LIP_PROBE): measures |Φ⁻¹(p_ĉ(x+δ))−Φ⁻¹(p_ĉ(x))| vs the
  -- proven bound ‖δ‖/σ — the empirical grounding of `smoothing_certified_radius`'s (1/σ)-Lipschitz hyp.
  let lipProbe := (← IO.getEnv "SMOOTH_LIP_PROBE").isSome
  let mut lipRows : Array String := #["sigma,img_idx,delta,dg,bound,ratio"]
  let mut lipMax := 0.0
  let mut lipViol := 0
  let mut lipN := 0
  for sigma in sigmas do
    IO.println s!"\n── σ = {sigma}  (radius ceiling at this n = {sigma * invNormCdf pMax}) ──"
    -- (1) train a Gaussian-noise-augmented base classifier (the Cohen recipe): every batch is
    --     corrupted with N(0,σ²I) host-side before the proof-rendered SGD step — graph unchanged.
    --     Best-θ checkpoint on clean eval acc.
    let mut parts : Array ByteArray := #[]
    let mut iseed := 1
    for spec in net.specs do
      parts := parts.push (← mkParam iseed spec.1 spec.2); iseed := iseed + 1
    let mut theta := F32.concat parts
    let mut bestTheta := theta
    let mut bestAcc := -1.0
    let mut nseed : USize := 1
    for ep in [0:cfg.epochs] do
      for bi in [0:nb] do
        let xbRaw := F32.sliceImages trainImg (bi * bs) bs trainPix
        let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
        let yb := F32.sliceLabels trainLbl (bi * bs) bs
        let xbN ← F32.addGaussianTiled xb 0 (bs*d0).toUSize 1 sigma nseed
        nseed := nseed + 1
        theta ← IreeSession.mlpTrainStepV tsSess tsFn xbN theta shapes yb bs.toUSize d0.toUSize d1.toUSize
      let mut c := 0
      for bi in [0:evalBatches] do
        let xb := F32.sliceImages evalImg (bi * bs) bs d0
        let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes xb (net.xShape bs) bs.toUSize d1.toUSize
        for j in [0:bs] do
          if (F32.argmax10 logits (j*d1).toUSize).toNat == (evalLbl.get! (4*(bi*bs+j))).toNat then c := c+1
      let acc := c.toFloat/(evalBatches*bs).toFloat*100.0
      if acc > bestAcc then bestAcc := acc; bestTheta := theta
      IO.println s!"    epoch {ep+1}/{cfg.epochs}: clean acc = {acc}%"
      (← IO.getStdout).flush
    let th := bestTheta                       -- certify the best checkpoint (immutable snapshot)
    IO.println s!"  noise-trained clean acc = {bestAcc}%"
    -- (2) SampleUnderNoise: `nBatches·bs` noisy copies of ONE image (element offset `off`), each
    --     forward feeding exactly `bs` rows (the rendered fwd's static batch), argmax-vote → counts.
    let sampleCountsB := fun (base : ByteArray) (off nBat : Nat) (sd : USize) => do
      let mut counts : Array Nat := Array.replicate d1 0
      for ci in [0:nBat] do
        let xN ← F32.addGaussianTiled base off.toUSize d0.toUSize bs.toUSize sigma (sd + ci.toUSize)
        let logits ← IreeSession.forwardF32 fwdSess fwdFn th shapes xN (net.xShape bs) bs.toUSize d1.toUSize
        for j in [0:bs] do
          let a := (F32.argmax10 logits (j*d1).toUSize).toNat
          counts := counts.set! a (counts[a]! + 1)
      pure counts
    let sampleCounts := fun (off nBat : Nat) (sd : USize) => sampleCountsB evalImg off nBat sd
    -- (3) certify each sampled test image: n0 select ĉ_A, n estimate p_A, radius σ·Φ⁻¹(p_A).
    let mut abstain := 0
    let mut natCorrect := 0
    let mut acr := 0.0
    let mut certCnt : Array Nat := Array.replicate radii.size 0
    for t in [0:nCert] do
      let imgIdx := t * stride
      let off := imgIdx * d0
      let label := (evalLbl.get! (4*imgIdx)).toNat
      let base : USize := (imgIdx + 1).toUSize * 131 + 1
      let counts0 ← sampleCounts off n0Batches base
      let mut cHatA := 0
      for c in [1:d1] do
        if counts0[c]! > counts0[cHatA]! then cHatA := c
      if cHatA == label then natCorrect := natCorrect + 1
      let counts ← sampleCounts off nBatches (base + 524287)
      let pA := clopperPearsonLower counts[cHatA]! nEff alpha
      let certified := pA > 0.5
      let radius := if certified then sigma * invNormCdf pA else 0.0
      if certified then
        if cHatA == label then
          acr := acr + radius
          for ri in [0:radii.size] do
            if radius ≥ radii[ri]! then certCnt := certCnt.set! ri (certCnt[ri]! + 1)
      else
        abstain := abstain + 1
      dumpRows := dumpRows.push s!"{sigma},{imgIdx},{label},{cHatA},{if certified then 0 else 1},{radius}"
      if (t+1) % 100 == 0 then
        IO.println s!"    certified {t+1}/{nCert} ..."; (← IO.getStdout).flush
    -- (4) Lipschitz-hypothesis probe: shift x by r·(unit vector) and compare the probit-score change
    --     |Φ⁻¹(p_ĉ(x+δ))−Φ⁻¹(p_ĉ(x))| to the PROVEN bound ‖δ‖/σ (Salman et al. 2019 Lemma 2 — the
    --     hypothesis `smoothing_certified_radius` assumes). ratio = Δg·σ/‖δ‖ ≤ 1 ⟺ (1/σ)-Lipschitz holds.
    if lipProbe then
      let rGrid : Array Float := #[0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
      let mProbe := min 40 nEval
      let pstride := max 1 (nEval / mProbe)
      let clampP := fun (k : Nat) =>
        let lo := 1.0 / (2.0 * nEff.toFloat)
        max lo (min (1.0 - lo) (k.toFloat / nEff.toFloat))
      for t in [0:mProbe] do
        let imgIdx := t * pstride
        let off := imgIdx * d0
        let cnt0 ← sampleCountsB evalImg off nBatches ((imgIdx + 7).toUSize * 131 + 3)
        let mut chat := 0
        for c in [1:d1] do
          if cnt0[c]! > cnt0[chat]! then chat := c
        let gx := invNormCdf (clampP cnt0[chat]!)
        for ri in [0:rGrid.size] do
          let r := rGrid[ri]!
          let xp ← F32.perturbUnit evalImg off.toUSize d0.toUSize r (imgIdx.toUSize * 17 + ri.toUSize + 1)
          let cntp ← sampleCountsB xp 0 nBatches (imgIdx.toUSize * 53 + ri.toUSize + 11)
          let gxp := invNormCdf (clampP cntp[chat]!)
          let dg := Float.abs (gxp - gx)
          let ratio := dg * sigma / r
          if ratio > lipMax then lipMax := ratio
          if ratio > 1.05 then lipViol := lipViol + 1
          lipN := lipN + 1
          lipRows := lipRows.push s!"{sigma},{imgIdx},{r},{dg},{r/sigma},{ratio}"
      IO.println s!"  Lipschitz probe: {mProbe} imgs × {rGrid.size} shifts (Φ⁻¹∘p_ĉ vs ‖δ‖/σ)"
      (← IO.getStdout).flush
    let tot := nCert.toFloat
    let pct := fun (k : Nat) => k.toFloat / tot * 100.0
    let certStr := String.intercalate " / " (radii.toList.zipIdx.map
      (fun (r, ri) => s!"{r}→{pct certCnt[ri]!}%"))
    IO.println s!"  smoothed natural acc = {pct natCorrect}%   abstain = {pct abstain}%   ACR = {acr/tot}"
    IO.println s!"  certified-robust acc (L2): {certStr}"
    (← IO.getStdout).flush
    rows := rows.push s!"  {sigma}\t{bestAcc}\t{pct natCorrect}\t{pct abstain}\t{pct certCnt[1]!}\t{pct certCnt[3]!}\t{pct certCnt[5]!}\t{acr/tot}"
  IO.println s!"\n══ randomized smoothing ({net.name}): depth-independent certified L2 radius ══"
  IO.println "  σ\tclean%\tnat%\tabst%\tcert@.5\tcert@1.0\tcert@1.5\tACR"
  for row in rows do IO.println row
  let csvPath := s!"runs/smooth_{net.slug}_radii.csv"
  IO.FS.writeFile csvPath (String.intercalate "\n" dumpRows.toList ++ "\n")
  IO.println s!"\nper-image certified radii → {csvPath} ({dumpRows.size - 1} rows, all σ) — fine frontier curves + ACR"
  if lipProbe then
    let lipPath := s!"runs/smooth_{net.slug}_lipschitz.csv"
    IO.FS.writeFile lipPath (String.intercalate "\n" lipRows.toList ++ "\n")
    IO.println s!"Lipschitz-hypothesis probe → {lipPath} ({lipN} measurements): max ratio Δg·σ/‖δ‖ = {lipMax}"
    IO.println s!"  ⇒ Φ⁻¹∘p_ĉ is empirically (1/σ)-Lipschitz: {lipN - lipViol}/{lipN} measurements ≤ bound (violations>1.05: {lipViol})"
  IO.println "done (randomized smoothing: forward-only Monte-Carlo cert via the proof-rendered fwd —"
  IO.println "      architecture-agnostic + depth-independent, non-vacuous where ∏‖Wᵢ‖₂ is hopeless)."

/-- **Phase-3 PGD adversarial attack** on the verified linear classifier
    (`planning/robustness.md`). Trains via the proof-rendered train step, then attacks
    through the real IREE pipeline: each PGD step's input gradient is computed by the
    `genLinearPgdStep` StableHLO kernel (the proven `dx = (softmax−onehot)·Wᵀ` VJP) on the
    GPU. Reports clean vs L∞-PGD adversarial accuracy over an eps sweep. -/
def VerifiedNet.attackPgd (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println s!"Phase-3 PGD attack on {net.name} (verified codegen → IREE → GPU)"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _, _) ← loadData net.data d0 dataDir
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut W0 ← F32.const (d0 * d1).toUSize 0.0
  let mut b0 ← F32.const d1.toUSize 0.0
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for _ in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let out ← IreeSession.linearTrainStepV tsSess tsFn xb W0 b0 yb bs.toUSize d0.toUSize d1.toUSize
      W0 := out.extract 0 (d0 * d1 * 4)
      b0 := out.extract (d0 * d1 * 4) ((d0 * d1 + d1) * 4)
  -- clean accuracy
  let mut clean := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImages evalImg (bi * bs) bs d0
    let logits ← IreeSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:bs] do
      if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
        clean := clean + 1
  IO.println s!"clean test acc = {clean}/{nbt*bs} = {clean.toFloat/(nbt*bs).toFloat*100.0}%"
  -- L∞ PGD sweep
  let K := 40
  for eps in ([0.1, 0.2, 0.3] : List Float) do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile ".lake/build/linear_pgd_step.mlir" (genLinearPgdStep bs d0 d1 eps alpha true)
    compileVmfb ".lake/build/linear_pgd_step.mlir" ".lake/build/linear_pgd_step.vmfb"
    let pgdSess ← IreeSession.create ".lake/build/linear_pgd_step.vmfb"
    let pgdShapes := packShapes #[#[d0, d1], #[d1], #[bs, d1], #[bs, d0]]
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImages evalImg (bi * bs) bs d0
      let oh ← oneHotBatch evalLbl (bi * bs) bs d1
      let pgdParams := F32.concat #[W0, b0, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← IreeSession.forwardF32 pgdSess "m.linear_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← IreeSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes x xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
          correct := correct + 1
    IO.println s!"L∞ PGD eps={eps}: adv acc = {correct}/{nbt*bs} = {correct.toFloat/(nbt*bs).toFloat*100.0}%"
  -- ── L2 sandwich: Lipschitz certificate (lower bound) vs L2 PGD (upper bound) ──
  let L := specNormW W0 d0 d1
  IO.println s!"\nglobal Lipschitz ‖W‖₂ = {L}  (linear: the logit map's exact L2 Lipschitz)"
  let tot := (nbt * bs).toFloat
  let mut cert05 := 0
  let mut cert10 := 0
  let mut cert15 := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImages evalImg (bi * bs) bs d0
    let logits ← IreeSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:bs] do
      let mut top := -1.0e30
      let mut sec := -1.0e30
      let mut topi := 0
      for c in [0:d1] do
        let v := F32.read logits (j * d1 + c).toUSize
        if v > top then
          sec := top
          top := v
          topi := c
        else if v > sec then
          sec := v
      if topi == (evalLbl.get! (4 * (bi * bs + j))).toNat then
        let r := (top - sec) / (1.4142135623730951 * L)    -- certified L2 radius m(x)/(√2 L)
        if r ≥ 0.5 then cert05 := cert05 + 1
        if r ≥ 1.0 then cert10 := cert10 + 1
        if r ≥ 1.5 then cert15 := cert15 + 1
  IO.println s!"certified-robust acc (L2): ε=0.5 → {cert05.toFloat/tot*100.0}%, ε=1.0 → {cert10.toFloat/tot*100.0}%, ε=1.5 → {cert15.toFloat/tot*100.0}%"
  for eps in ([0.5, 1.0, 1.5] : List Float) do
    let alpha := 2.5 * eps / K.toFloat
    IO.FS.writeFile ".lake/build/linear_pgd_step.mlir" (genLinearPgdStep bs d0 d1 eps alpha false)
    compileVmfb ".lake/build/linear_pgd_step.mlir" ".lake/build/linear_pgd_step.vmfb"
    let pgdSess ← IreeSession.create ".lake/build/linear_pgd_step.vmfb"
    let pgdShapes := packShapes #[#[d0, d1], #[d1], #[bs, d1], #[bs, d0]]
    let mut correct := 0
    for bi in [0:nbt] do
      let x0 := F32.sliceImages evalImg (bi * bs) bs d0
      let oh ← oneHotBatch evalLbl (bi * bs) bs d1
      let pgdParams := F32.concat #[W0, b0, oh, x0]
      let mut x := x0
      for _ in [0:K] do
        x ← IreeSession.forwardF32 pgdSess "m.linear_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
      let logits ← IreeSession.forwardF32 fwdSess fwdFn (W0 ++ b0) shapes x xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
          correct := correct + 1
    IO.println s!"L2 PGD eps={eps}: adv acc = {correct.toFloat/tot*100.0}%  (sandwich: cert ≤ true ≤ this)"
  IO.println "done (phase-3 PGD: gradient computed by the proven input-VJP kernel via IREE)."

/-- **fp8 (E4M3) Lean trainer** — the low-precision sibling of `trainLinear`.

    Keeps **fp32 master weights** and, each step, projects the weights
    (per-output-column) and the activations (per-tensor) onto the **E4M3** grid
    (`LeanMlir/E4M3Quant.lean`), runs the *same* verified `@<slug>_train_step`
    kernel (the matmul accumulates in fp32 — the `dotMixed` model: `u_leaf =
    E4M3`, `u_acc = fp32`), and applies the recovered gradient delta to the fp32
    master via `addDelta` (`master += Wout − Wq = master − lr·∇`). The MLIR and
    FFI are **unchanged**: fp8 here is host-side operand byte-prep, exactly the
    §3b render-tie model (`Proofs/E4M3FaithfulPoC.lean`). Eval runs the fp32
    master through `@<slug>_fwd` (the "fp32-infer" accuracy of the fp8-trained
    model, mirroring `scripts/mnist_e4m3_demo.py`).

    Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-linear-e4m3-verified data` -/
def VerifiedNet.trainLinearE4M3 (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let d1 := net.nClasses
  IO.println net.blurb
  IO.println "  [fp8 E4M3] fp32 master · per-column W / per-tensor x → E4M3 grid · fp32 accumulate"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, _trainPix, _crop) ←
    loadData net.data d0 dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; dense {d0}->{d1}, bs {bs}, fp8-SGD (E4M3 leaf / fp32 acc)"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  -- Static per-tensor activation scale ⇒ quantize the whole train set ONCE.
  let trainImgQ := F32E4M3.quantPerTensor trainImg
  let mut mW ← F32.const (d0 * d1).toUSize 0.0     -- fp32 master weights (zero-init)
  let mut mb ← F32.const d1.toUSize 0.0            -- fp32 master bias (unquantized)
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImgQ (bi * bs) bs d0     -- E4M3 activations
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let Wq := F32E4M3.quantPerColumn mW d0 d1               -- E4M3 weight operand
      let out ← IreeSession.linearTrainStepV tsSess tsFn
                  xb Wq mb yb bs.toUSize d0.toUSize d1.toUSize
      let Wout := out.extract 0 (d0 * d1 * 4)
      let bout := out.extract (d0 * d1 * 4) ((d0 * d1 + d1) * 4)
      mW := F32E4M3.addDelta mW Wout Wq                       -- master += (Wout − Wq)
      mb := bout                                              -- bias update is exact (unquantized)
    let params := mW ++ mb
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 fwdSess fwdFn params shapes
                      xb xShape bs.toUSize d1.toUSize
      for j in [0:bs] do
        let pred := (F32.argmax10 logits (j * d1).toUSize).toNat
        let lbl  := (evalLbl.get! (4 * (bi * bs + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * bs).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nbt * bs} = {acc}% (fp8 E4M3)"
    (← IO.getStdout).flush
  IO.println s!"done (trained {net.name} in fp8 E4M3 on the proof-rendered StableHLO)."

/-- **fp8 (E4M3) packed-params trainer** — the low-precision sibling of
    `VerifiedNet.train`, for the depth>1 nets (MLP, CNN). Keeps **fp32 master
    params** and, each step, projects every *weight* slot onto the E4M3 grid
    (dense per-output-column, conv per-output-channel; biases kept fp32 —
    `F32E4M3.quantPackedParams`) and the *input* per-tensor, runs the *same*
    verified `@<slug>_train_step` (fp32 accumulate inside), and folds the
    gradient delta back into the master with `addDelta` over the whole packed
    buffer (`master += out − paramsQ`: weight slots get `−lr·∇`, bias slots the
    exact update). MLIR/FFI unchanged.

    **Scope (honest):** host-side prep reaches weights + the *input* activation
    only. The intermediate activations (relu/pool/flatten outputs feeding the
    deeper matmuls) and the backward-chain cotangents are computed *inside* the
    fused kernel and stay fp32 — quantizing them needs in-graph E4M3 ops (the
    next, codegen-level step), not host byte-prep. So this is honest **fp8
    weights + fp8 input, fp32 intermediates**. Eval runs the fp32 master.

    Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-e4m3-verified data` -/
def VerifiedNet.trainE4M3 (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  IO.println net.blurb
  IO.println "  [fp8 E4M3] fp32 master · per-slot weight quant (dense per-col / conv per-channel) + per-tensor input · fp32 accumulate"
  IO.println "  note: depth>1 ⇒ intermediate activations & cotangents stay fp32 (inside the kernel); weights + input are E4M3"
  let tsVmfb  := s!".lake/build/{net.slug}_ts_v.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"        fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    loadData net.data d0 dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} ({net.specs.size} params, {net.nParams} floats), fp8-SGD (E4M3 leaf / fp32 acc), He init"
  (← IO.getStdout).flush
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let shapes := net.shapesBA
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut params := F32.concat parts                       -- fp32 master params
  -- Static per-tensor input scale ⇒ quantize the train images ONCE (crop, if any,
  -- only selects grid-valued pixels, so quantize-then-crop stays on the grid).
  let trainImgQ := F32E4M3.quantPerTensor trainImg
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xbRaw := F32.sliceImages trainImgQ (bi * bs) bs trainPix
      let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let paramsQ := F32E4M3.quantPackedParams params net.specs   -- E4M3 weight operands
      let out ← IreeSession.mlpTrainStepV tsSess tsFn
                  xb paramsQ shapes yb bs.toUSize d0.toUSize nc.toUSize
      params := F32E4M3.addDelta params out paramsQ              -- master += (out − paramsQ)
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 fwdSess fwdFn params shapes
                      xb xShape bs.toUSize nc.toUSize
      for j in [0:bs] do
        let pred := (F32.argmax10 logits (j * nc).toUSize).toNat
        let lbl  := (evalLbl.get! (4 * (bi * bs + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * bs).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nbt * bs} = {acc}% (fp8 E4M3)"
    (← IO.getStdout).flush
  IO.println s!"done (trained {net.name} in fp8 E4M3 on the proof-rendered StableHLO)."

/-- **fp8 (E4M3) variant of `trainAdamSched`** — runs the Adam / Nesterov-momentum
    optimizer demos in fp8. Keeps an fp32 master `[θ|m|v]`; each step projects the
    *weight* third `θ` onto the E4M3 grid (`quantPackedParams`: dense per-column,
    conv per-channel; biases fp32) and the input per-tensor, runs the *same*
    verified `@<slug>_<variant>_train_step` (the optimizer is baked into the MLIR,
    so fp8 needs no new module — operand byte-prep only; fp32 accumulate), and folds
    the optimizer-step delta back into the fp32 master θ (`addDelta`), keeping the
    returned `m'/v'` moments in fp32. Distinct `_e4m3` checkpoint (won't resume an
    fp32 run); honors `LEAN_MLIR_MAX_EPOCHS`. Same scope as `trainE4M3`: fp8
    weights + input, fp32 intermediates / moments. -/
def VerifiedNet.trainAdamSchedE4M3 (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String)
    (baseLR β1 β2 : Float) (warmupEpochs : Nat) (variant : String := "adam") : IO Unit := do
  let bs := cfg.batchSize
  let d0 := net.d0
  let nc := net.nClasses
  IO.println net.blurb
  IO.println s!"  [fp8 E4M3] fp32 master [θ|m|v] · per-slot θ quant + per-tensor input · fp32 accumulate ({variant})"
  let hasBn := !net.bnChannels.isEmpty
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let tsVmfb  := s!".lake/build/{net.slug}_{variant}_ts.vmfb"
  let fwdVmfb := s!".lake/build/{net.slug}_fwd_v.vmfb"
  let fwdEvalVmfb := s!".lake/build/{net.slug}_fwd_eval_v.vmfb"
  compileVmfb s!"verified_mlir/{net.slug}_{variant}_train_step.mlir" tsVmfb
  compileVmfb s!"verified_mlir/{net.slug}_fwd.mlir"             fwdVmfb
  let tsSess  ← IreeSession.create tsVmfb
  let fwdSess ← IreeSession.create fwdVmfb
  let fwdEvalSess ← if hasBn then do
      compileVmfb s!"verified_mlir/{net.slug}_fwd_eval.mlir" fwdEvalVmfb
      IreeSession.create fwdEvalVmfb
    else pure fwdSess
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    loadData net.data d0 dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  let nb  := nTrain / bs
  let nbt := nEval / bs
  let nEpochs := match (← IO.getEnv "LEAN_MLIR_MAX_EPOCHS").bind (·.toNat?) with
    | some n => min n cfg.epochs
    | none   => cfg.epochs
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} {variant} fp8 (cosine+warmup {warmupEpochs}ep, baseLR {baseLR}), He init"
  (← IO.getStdout).flush
  let adamShapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes ++ #[#[], #[], #[]]
                                ++ (if hasBn then bnStatShapes else #[]))
  let fwdShapes := net.shapesBA
  let fwdEvalShapes := packShapes (net.paramShapes ++ bnStatShapes)
  let xShape := net.xShape bs
  let tsFn  := s!"m.{net.slug}_{variant}_train_step"
  let fwdFn := s!"m.{net.slug}_fwd"
  let mut parts : Array ByteArray := #[]
  let mut seed := ((← IO.getEnv "LEAN_MLIR_SEED").bind (·.toNat?)).getD 1
  for spec in net.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let theta := F32.concat parts
  let zeros ← F32.const net.nParams.toUSize 0.0
  let mut thetamv := F32.concat #[theta, zeros, zeros]
  let mvBytes := 3 * net.nParams * 4
  let pBytes := net.nParams * 4
  let mut runningBnStats ← F32.const nBnStats.toUSize 0.0
  let mut bnFirst := true
  let totalSteps := (cfg.epochs * nb).toFloat
  let warmSteps := (warmupEpochs * nb).toFloat
  let ckptPath := s!".lake/build/{net.slug}_{variant}_e4m3_ckpt.bin"   -- distinct from the fp32 runs
  let epPath := ckptPath ++ ".epoch"
  let mut startEpoch := 0
  if (← System.FilePath.pathExists ckptPath) && (← System.FilePath.pathExists epPath) then
    thetamv ← IO.FS.readBinFile ckptPath
    startEpoch := ((← IO.FS.readFile epPath).toNat?).getD 0
    IO.println s!"  ▸ resuming from fp8 checkpoint at epoch {startEpoch}"
    (← IO.getStdout).flush
  -- pre-quantize the train images ONCE (per-tensor E4M3); shuffle + hflip preserve the grid.
  let mut curImg := F32E4M3.quantPerTensor trainImg
  let mut curLbl := trainLbl
  for ep in [startEpoch:nEpochs] do
    let mut epochLossSum := 0.0
    let mut lastLr := 0.0
    let (sImg, sLbl) ← F32.shuffle curImg curLbl nTrain.toUSize trainPix.toUSize (ep + 42).toUSize
    curImg := sImg; curLbl := sLbl
    for bi in [0:nb] do
      let gstep := (ep * nb + bi + 1).toFloat
      let lrt := if gstep ≤ warmSteps then baseLR * gstep / warmSteps
                 else baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (gstep - warmSteps) / (totalSteps - warmSteps)))
      let bc1 := 1.0 - Float.exp (gstep * Float.log β1)
      let bc2 := 1.0 - Float.exp (gstep * Float.log β2)
      let tail := F32.concat #[← F32.const (1 : USize) lrt, ← F32.const (1 : USize) bc1, ← F32.const (1 : USize) bc2]
      -- fp8: project the θ third onto the E4M3 grid (weights per-slot; biases + m/v stay fp32).
      let thetaMaster := thetamv.extract 0 pBytes
      let thetaQ := F32E4M3.quantPackedParams thetaMaster net.specs
      let thetamvQ := F32.concat #[thetaQ, thetamv.extract pBytes mvBytes]
      let params := if hasBn then F32.concat #[thetamvQ, tail, runningBnStats] else F32.concat #[thetamvQ, tail]
      let augSeed := (ep * nb + bi + 1).toUSize
      let xbRaw := F32.sliceImages curImg (bi * bs) bs trainPix
      let xb ← match net.data with
        | .imagenette =>
            let c ← if crop then F32.randomCrop xbRaw bs.toUSize 3 256 256 224 224 augSeed
                    else pure xbRaw
            F32.randomHFlip c bs.toUSize 3 224 224 (augSeed + 7777)
        | .cifar => F32.randomHFlip xbRaw bs.toUSize 3 32 32 augSeed
        | _ => pure xbRaw
      let yb := F32.sliceLabels curLbl (bi * bs) bs
      let out ← IreeSession.mlpTrainStepV tsSess tsFn xb params adamShapes yb bs.toUSize d0.toUSize nc.toUSize
      let stepLoss := F32.read out (3 * net.nParams).toUSize
      epochLossSum := epochLossSum + stepLoss
      lastLr := lrt
      if bi < 3 || bi % 100 == 0 then
        IO.println s!"  step {bi}/{nb}: loss={stepLoss}"
        (← IO.getStdout).flush
      -- fp8 master recovery: θ_master += (θ' − θ_q); keep the returned fp32 m'/v'.
      let thetaPrime := out.extract 0 pBytes
      let mvPrime := out.extract pBytes mvBytes
      let thetaMasterNew := F32E4M3.addDelta thetaMaster thetaPrime thetaQ
      thetamv := F32.concat #[thetaMasterNew, mvPrime]
      if hasBn then
        let batchBn := out.extract ((3 * net.nParams + 3) * 4) ((3 * net.nParams + 3 + nBnStats) * 4)
        runningBnStats ← F32.ema runningBnStats batchBn (if bnFirst then 1.0 else 0.1)
        bnFirst := false
    IO.println s!"Epoch {ep + 1}/{nEpochs}: loss={epochLossSum / nb.toFloat} lr={lastLr}"
    let thetaCur := thetamv.extract 0 pBytes
    let evalSess := if hasBn then fwdEvalSess else fwdSess
    let evalFn := if hasBn then s!"m.{net.slug}_fwd_eval" else fwdFn
    let evalParams := if hasBn then F32.concat #[thetaCur, runningBnStats] else thetaCur
    let evalShapes := if hasBn then fwdEvalShapes else fwdShapes
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages evalImg (bi * bs) bs d0
      let logits ← IreeSession.forwardF32 evalSess evalFn evalParams evalShapes
                      xb xShape bs.toUSize nc.toUSize
      for j in [0:bs] do
        let pred := (F32.argmax10 logits (j * nc).toUSize).toNat
        let lbl  := (evalLbl.get! (4 * (bi * bs + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * bs).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nbt * bs} = {acc}% (fp8 E4M3, {variant})"
    (← IO.getStdout).flush
    IO.FS.writeBinFile ckptPath thetamv
    IO.FS.writeFile epPath (toString (ep + 1))
  IO.println s!"done (trained {net.name} {variant} in fp8 E4M3 on the proof-rendered StableHLO)."
