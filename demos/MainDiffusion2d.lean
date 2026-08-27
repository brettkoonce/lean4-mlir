import LeanMlir

/-! Diffusion on a 2-D toy distribution — `planning/diffusion_2d_demo.md`.

    The diffusion demo you can be *wrong* about. Every image DDPM here
    succeeds or fails by "does that look like a digit to you"; on a 2-D
    target the ground truth is a second point cloud, so correctness is a
    number (§4 of the plan: mode recall and energy distance) instead of a
    judgement — and it trains in seconds on ~35K params rather than 7 h on 3M.

    ⭐ No new codegen primitives. The denoiser is `.dense`/`.relu`, which
    Chapters 1-2 already prove, and the time conditioning reuses
    `Ddpm.prependSinCosT` with `H = W = 1` — the image path's own encoding,
    applied to a 2-vector instead of a plane.

    Usage:
      python3 preprocess_toy2d.py 8192 data/toy2d
      lake exe diffusion-2d                    # train + sample + dump
      lake exe diffusion-2d 4000               # step override
      python3 scripts/toy2d_metrics.py         # mode recall + energy distance
-/

/-- Frequencies in the sinusoidal time encoding; the input is the 2-vector
    plus `2 * nFreq` channels of `[sin(t·ω_k), cos(t·ω_k)]`. -/
def nFreq : Nat := 4
def condDim : Nat := 2 + 2 * nFreq

/-- ~35K params at 128 hidden, against the tiny image UNet's 118K. The output
    is `[B, 2]` — a predicted ε per point — which is what puts this on the
    codegen's rank-2 DDPM branch rather than the 4-D image one. -/
def diff2dDenoiser : NetSpec where
  name   := "diffusion 2d denoiser (8-gaussians)"
  imageH := 1
  imageW := 1
  layers := [
    .dense condDim 128 .relu,
    .dense 128 128 .relu,
    .dense 128 2 .identity
  ]

def diff2dConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 256
  epochs       := 1
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false

def Tmax : Nat := 1000

def main (args : List String) : IO Unit := do
  let steps  := (args[0]?.bind String.toNat?).getD 3000
  -- Sampler step count is an ARGUMENT, not a constant: the plan's open
  -- question is how many reverse steps a 2-D manifold actually needs, and
  -- the image demos' 50 is a convention nobody measured. Here it is a sweep.
  let nStepsArg := (args[1]?.bind String.toNat?).getD 50
  let spec   := diff2dDenoiser
  let cfg    := diff2dConfig
  let B      := cfg.batchSize
  IO.eprintln s!"{spec.name}: {spec.totalParams} params, {steps} steps"

  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix
  -- ⚠ RANK-4 [B, 2, 1, 1], not [B, 2]. `iree_ffi_train_step_adam_ddpm`
  -- hardcodes a rank-4 target upload; the loss branch reshapes. Same
  -- convention the FPN detector uses to ride this FFI unchanged.
  let outShape : List Nat := [B, 2, 1, 1]

  IO.eprintln "Generating train step MLIR..."
  let trainMlir := MlirCodegen.generateTrainStep spec B
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := cfg.weightDecay) (useAdam := cfg.useAdam)
    (useDdpm := true) (ddpmOutShape := outShape)
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.eprintln s!"  {trainMlir.length} chars"

  let evalMlir := MlirCodegen.generateEval spec 1
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" evalMlir

  let trainArt ← NetSpec.graphArtifact pfx "train_step"
  let evalArt  ← NetSpec.graphArtifact pfx "fwd_eval"

  -- ── data: flat f32 LE [N, 2], written by preprocess_toy2d.py ──
  let raw ← IO.FS.readBinFile "data/toy2d/eight_gaussians.bin"
  let nPts := F32.size raw / 2
  IO.eprintln s!"  data: {nPts} points"

  let params ← spec.heInitParams
  let nP := spec.totalParams
  let adamM ← F32.const nP.toUSize 0.0
  let adamV ← F32.const nP.toUSize 0.0
  let alphaBar ← Ddpm.cosineSchedule Tmax.toUSize

  let sess ← LowererSession.create trainArt
  IO.eprintln "  session loaded"

  let allShapes := spec.shapesBA
  let bnShapes  := spec.bnShapesBA
  let xSh       := spec.xShape B
  let nT        := 3 * nP
  let batch     : USize := B.toUSize
  let nPer      : Nat := 2

  let mut p := params
  let mut m := adamM
  let mut v := adamV
  let mut bnRun ← F32.const spec.nBnStats.toUSize 0.0
  let bpE := nPts / B

  IO.eprintln s!"training: {steps} steps, batch={B}, lr={cfg.learningRate}"
  let t0 ← IO.monoMsNow
  for gs in [:steps] do
    let bi := gs % bpE
    let x0 := F32.slice raw (bi * B * nPer) (B * nPer)
    let (xt, rest) ← Ddpm.stepInputs x0 alphaBar batch nPer.toUSize gs.toUSize
    let (eps, tba) := rest
    -- Time conditioning with H = W = 1: the image encoder applied to a point.
    let xtc ← Ddpm.prependSinCosT xt tba batch nPer.toUSize 1 1
                nFreq.toUSize Tmax.toUSize
    let packed := (p.append m).append v
    let out ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName
                packed allShapes xtc xSh eps
                cfg.learningRate (gs+1).toFloat
                bnShapes batch 2 1 1
    let loss := F32.extractLoss out nT
    p := F32.slice out 0 nP
    m := F32.slice out nP nP
    v := F32.slice out (2 * nP) nP
    if gs % 500 == 0 || gs + 1 == steps then
      IO.eprintln s!"  step {gs}/{steps}: loss={loss}"
  let t1 ← IO.monoMsNow
  IO.eprintln s!"trained in {t1-t0}ms"
  IO.FS.writeBinFile s!"{pfx}_params.bin" p

  -- ── sampling: DDIM (η = 0) from pure noise, batched ──
  let nGen : Nat := 2048
  IO.eprintln s!"sampling {nGen} points..."
  let evalSess ← LowererSession.create evalArt
  let evalShapes := spec.evalShapesBA
  let bnPad ← F32.const spec.nBnStats.toUSize 0.0
  let evalParams := p.append bnPad
  let xSh1 := spec.xShape 1
  -- One point at a time keeps the eval graph at batch 1 (it is 35K params;
  -- the loop is still sub-second) and avoids a second compiled batch shape.
  let nSteps : Nat := nStepsArg
  let stride := Tmax / nSteps
  let mut acc : Array ByteArray := #[]
  for i in [:nGen] do
    let mut x ← Ddpm.sampleNoise nPer.toUSize (i + 7919).toUSize
    for k in [:nSteps] do
      let tCur := Tmax - 1 - k * stride
      let tPrev := if k + 1 == nSteps then 0 else Tmax - 1 - (k+1) * stride
      let xc ← Ddpm.prependSinCosTScalar x 1 nPer.toUSize 1 1
                 tCur.toUSize nFreq.toUSize Tmax.toUSize
      let epsHat ← LowererSession.forwardF32 evalSess spec.evalFnName
                     evalParams evalShapes xc xSh1 1 nPer.toUSize
      let abT := F32.read alphaBar tCur.toUSize
      -- Matches the image sampler's convention: the final step uses 0.9999
      -- rather than a literal 1.0, which would send `a = √ᾱ_prev/√ᾱ_t` sky-high.
      let abP := if tPrev == 0 then 0.9999 else F32.read alphaBar tPrev.toUSize
      let a := Float.sqrt abP / Float.sqrt abT
      let b := Float.sqrt (1.0 - abP) - a * Float.sqrt (1.0 - abT)
      x ← Ddpm.ddimStep x epsHat a b nPer.toUSize
    acc := acc.push x
  let gen := F32.concat acc
  let outPath := s!".lake/build/diffusion2d_samples_s{steps}_n{nSteps}.bin"
  IO.FS.writeBinFile outPath gen
  IO.FS.writeBinFile ".lake/build/diffusion2d_samples.bin" gen
  IO.eprintln s!"wrote .lake/build/diffusion2d_samples.bin ({F32.size gen / 2} points)"
  IO.eprintln "▶ score it: python3 scripts/toy2d_metrics.py"
