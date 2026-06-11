import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

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
    -- Some data dirs ship `train.bin` at the canonical 256² (→ 224 center-crop);
    -- others (incl. this env) store it at 224² already (4-byte header + records of
    -- [1 label byte + 224·224·3 uint8]) — calling the loader with the wrong size is
    -- the "short read". Default to 224²/no-crop (matches val); override with
    -- LEAN_MLIR_IMAGENETTE_TRAIN=256 for the 256² train split + center-crop.
    let px := ((← IO.getEnv "LEAN_MLIR_IMAGENETTE_TRAIN").bind (·.toNat?)).getD 224
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
  let (trainImg, trainLbl, nTrain, evalImg, evalLbl, nEval, trainPix, crop) ←
    loadData net.data d0 dataDir
  let evalName := match net.data with | .imagenette => "val" | _ => "test"
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} ({net.specs.size} params, {net.nParams} floats), mean-loss SGD lr={cfg.lr}, He init"
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
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let xbRaw := F32.sliceImages trainImg (bi * bs) bs trainPix
      let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      params ← IreeSession.mlpTrainStepV tsSess tsFn
                  xb params shapes yb bs.toUSize d0.toUSize nc.toUSize
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
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nbt * bs} = {acc}%"
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
    (baseLR β1 β2 : Float) (warmupEpochs : Nat) : IO Unit := do
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
  let nb  := nTrain / bs
  let nbt := nEval / bs
  IO.println s!"  train {nTrain}, {evalName} {nEval}; bs {bs}, {net.name} AdamW (cosine+warmup {warmupEpochs}ep, baseLR {baseLR}), He init"
  (← IO.getStdout).flush
  let adamShapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes ++ #[#[], #[], #[]])
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
  let mut thetamv := F32.concat #[theta, zeros, zeros]
  let mvBytes := 3 * net.nParams * 4
  let pBytes := net.nParams * 4
  let totalSteps := (cfg.epochs * nb).toFloat
  let warmSteps := (warmupEpochs * nb).toFloat
  for ep in [0:cfg.epochs] do
    for bi in [0:nb] do
      let gstep := (ep * nb + bi + 1).toFloat
      let lrt := if gstep ≤ warmSteps then baseLR * gstep / warmSteps
                 else baseLR * 0.5 * (1.0 + Float.cos (3.14159265358979 * (gstep - warmSteps) / (totalSteps - warmSteps)))
      let bc1 := 1.0 - Float.exp (gstep * Float.log β1)
      let bc2 := 1.0 - Float.exp (gstep * Float.log β2)
      let tail := F32.concat #[← F32.const (1 : USize) lrt, ← F32.const (1 : USize) bc1, ← F32.const (1 : USize) bc2]
      let params := F32.concat #[thetamv, tail]
      let xbRaw := F32.sliceImages trainImg (bi * bs) bs trainPix
      let xb ← if crop then F32.centerCrop xbRaw bs.toUSize 3 256 256 224 224 else pure xbRaw
      let yb := F32.sliceLabels trainLbl (bi * bs) bs
      let out ← IreeSession.mlpTrainStepV tsSess tsFn xb params adamShapes yb bs.toUSize d0.toUSize nc.toUSize
      thetamv := out.extract 0 mvBytes
    let thetaCur := thetamv.extract 0 pBytes
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
  for ep in [0:cfg.epochs] do
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
    IO.println s!"  epoch {ep + 1}: {evalName}_acc = {correct}/{nbt * bs} = {acc}%"
    (← IO.getStdout).flush
  IO.println s!"done (trained {net.name} via the proof-rendered StableHLO)."
