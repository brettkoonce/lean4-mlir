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

/-- iree-compile one `.mlir` → `.vmfb`, surfacing failures. -/
private def compileVmfb (mlirPath outPath : String) : IO Unit := do
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

/-- Train driver for the **2-parameter linear** path (Chapter 2). The verified
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
