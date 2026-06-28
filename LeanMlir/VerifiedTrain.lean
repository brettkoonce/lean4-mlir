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

/-- **Phase-3 PGD attack on the verified MNIST CNN** (`planning/robustness_ladder.md`, the
    first conv rung). Trains the `conv→conv→pool→512→512→10` net on the proof-rendered SGD
    step, then attacks through IREE with `genCnnPgdStep` — whose input gradient is the full
    proven backward (conv input-VJP + maxpool `select_and_scatter`-back, mirroring
    `cnn_train_step.mlir`) run to `dx`. The certificate is the conv-aware spectral-norm
    **product** (`specNormConvTapSum` for the two convs × `specNormW` for the three denses);
    ReLU and maxpool are 1-Lipschitz. Over ~5 layers the product is even looser than the
    MLP's — making the linear-tight → MLP-vacuous → CNN-more-vacuous depth-cliff visual. -/
def VerifiedNet.attackPgdCnn (net : VerifiedNet) (cfg : VerifiedConfig) (dataDir : String) : IO Unit := do
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
  IO.println s!"  training {net.name} ({cfg.epochs} epochs, bs {bs}) ..."
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * bs) bs d0
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      theta ← IreeSession.mlpTrainStepV tsSess tsFn xb theta shapes yb bs.toUSize d0.toUSize d1.toUSize
    IO.println s!"    epoch {ep + 1}/{cfg.epochs} done"
    (← IO.getStdout).flush
  -- clean accuracy
  let mut clean := 0
  for bi in [0:nbt] do
    let xb := F32.sliceImages evalImg (bi * bs) bs d0
    let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes xb xShape bs.toUSize d1.toUSize
    for j in [0:bs] do
      if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
        clean := clean + 1
  IO.println s!"clean test acc = {clean}/{nbt*bs} = {clean.toFloat/(nbt*bs).toFloat*100.0}%"
  let K := 40
  let pgdShapes := packShapes (net.paramShapes ++ #[#[bs, d1], #[bs, d0]])
  let runSweep := fun (linf : Bool) (epsList : List Float) => do
    for eps in epsList do
      let alpha := 2.5 * eps / K.toFloat
      IO.FS.writeFile ".lake/build/cnn_pgd_step.mlir" (genCnnPgdStep bs eps alpha linf)
      compileVmfb ".lake/build/cnn_pgd_step.mlir" ".lake/build/cnn_pgd_step.vmfb"
      let pgdSess ← IreeSession.create ".lake/build/cnn_pgd_step.vmfb"
      let mut correct := 0
      for bi in [0:nbt] do
        let x0 := F32.sliceImages evalImg (bi * bs) bs d0
        let oh ← oneHotBatch evalLbl (bi * bs) bs d1
        let pgdParams := F32.concat #[theta, oh, x0]
        let mut x := x0
        for _ in [0:K] do
          x ← IreeSession.forwardF32 pgdSess "m.cnn_pgd_step" pgdParams pgdShapes x xShape bs.toUSize d0.toUSize
        let logits ← IreeSession.forwardF32 fwdSess fwdFn theta shapes x xShape bs.toUSize d1.toUSize
        for j in [0:bs] do
          if (F32.argmax10 logits (j * d1).toUSize).toNat == (evalLbl.get! (4 * (bi * bs + j))).toNat then
            correct := correct + 1
      let lbl := if linf then "L∞" else "L2"
      IO.println s!"{lbl} PGD eps={eps}: adv acc = {correct.toFloat/(nbt*bs).toFloat*100.0}%"
      (← IO.getStdout).flush
  runSweep true [0.1, 0.2, 0.3]
  -- ── certificate: conv-aware spectral-norm PRODUCT (ReLU/maxpool are 1-Lipschitz) ──
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
  runSweep false [0.5, 1.0, 1.5]
  IO.println "done (phase-3 CNN PGD: input gradient = the proven conv/maxpool input-VJP via IREE)."

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
