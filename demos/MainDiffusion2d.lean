import LeanMlir

/-! Diffusion on a 2-D toy distribution — `planning/diffusion_2d_demo.md`.

    The diffusion demo you can be *wrong* about. Every image DDPM here
    succeeds or fails by "does that look like a digit to you"; on a 2-D
    target the ground truth is a second point cloud, so correctness is a
    number (§4 of the plan: cell recall and energy distance) instead of a
    judgement — and it trains in seconds on 18,178 params rather than 7 h
    on 3M.

    ⭐ No new codegen primitives. The denoiser is `.dense`/`.relu`, which
    Chapters 1-2 already prove, and the time conditioning reuses
    `Ddpm.prependSinCosT` with `H = W = 1` — the image path's own encoding,
    applied to a 2-vector instead of a plane.

    ⭐ FOUR targets (plan §2), each catching a different failure: 8-gaussians
    mode collapse, spiral corner-cutting, two-moons over-smoothing across the
    gap, checkerboard mass leaking into the empty squares. The target name is
    an argument and it flows into `spec.name`, so each one owns its MLIR,
    checkpoint and samples without a flag anywhere else.

    ⭐ `strip` dumps the reverse process (plan §5): the cloud at nine points
    along t = T … 0, which is the figure images cannot carry because their
    intermediate states are grey mush and these are legible.

    Usage:
      python3 preprocess_toy2d.py 8192 data/toy2d
      lake exe diffusion-2d                          # train + sample + dump
      lake exe diffusion-2d 20000 200 25             # steps, sampler steps, eta%
      lake exe diffusion-2d spiral                   # any of the four targets
      lake exe diffusion-2d two_moons reuse strip    # the reverse-process strip
      python3 scripts/toy2d_metrics.py --target=spiral
      python3 scripts/toy2d_strip.py two_moons
-/

/-- Frequencies in the sinusoidal time encoding; the input is the 2-vector
    plus `2 * nFreq` channels of `[sin(t·ω_k), cos(t·ω_k)]`. -/
def nFreq : Nat := 4
def condDim : Nat := 2 + 2 * nFreq

/-- The plan's §2 targets: data-file stem, and the display name that goes into
    `spec.name`. `buildPrefix` is derived from that name, so naming the target
    here is the whole of what keeps four sets of artifacts apart. ⚠ The
    8-gaussians display name must stay `8-gaussians` — it is what the existing
    checkpoint and MLIR on disk are keyed by. -/
def toy2dTargets : List (String × String) :=
  [("eight_gaussians", "8-gaussians"),
   ("spiral",          "spiral"),
   ("two_moons",       "two-moons"),
   ("checkerboard",    "checkerboard")]

/-- 18,178 params at 128 hidden, against the tiny image UNet's 118K. The output
    is `[B, 2]` — a predicted ε per point — which is what puts this on the
    codegen's rank-2 DDPM branch rather than the 4-D image one. -/
def diff2dDenoiser (label : String) : NetSpec where
  name   := s!"diffusion 2d denoiser ({label})"
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

/-- Frames in the reverse-process strip, not counting the final one: nine
    panels total, the first being pure noise at `t = T`. -/
def nStripFrames : Nat := 8

def main (args : List String) : IO Unit := do
  -- The target is matched by NAME anywhere in the args, the same way `reuse`
  -- is, so it composes with the positional numeric arguments.
  let target := (args.find? fun a => toy2dTargets.any (·.1 == a)).getD "eight_gaussians"
  let label  := (toy2dTargets.lookup target).getD "8-gaussians"
  -- ⚠ Refuse an argument that is neither a number, a known flag, nor a known
  -- target. Without this a typo'd target silently trains 8-gaussians instead,
  -- which is the failure mode this whole demo exists to make impossible to
  -- mistake for a success.
  -- ⭐ `ddim` is the shipped sampler and the default. The other three are the
  -- Score-SDE family on the SAME weights: `euler` integrates the
  -- probability-flow ODE naively, `heun` does it to second order, `sde`
  -- integrates the reverse SDE with Euler-Maruyama.
  let sampler := (args.find? fun a => Ddpm.samplerNfe.any (·.1 == a)).getD "ddim"
  -- ⚠ `logsnr` spaces the continuous solvers' grid uniformly in log σ instead of
  -- uniformly in t. It exists to SETTLE A CONFOUND, not as a tuning knob: the
  -- first sweep held spacing uniform so the comparison was between solvers, but
  -- uniform-in-t is the worst grid for a stiff VP schedule, so part of what DDIM
  -- appeared to win was its parameterisation rather than its integrator. Giving
  -- the explicit solvers the better grid is what separates the two.
  let logsnr := args.any (· == "logsnr")
  -- ⭐ `logabar` is the STABILITY-OPTIMAL grid for an explicit solver, and it is
  -- the control that actually settles the confound. β = -d/dt log ᾱ, so a grid
  -- uniform in log ᾱ holds `h·β` constant — and `h·β` is exactly the
  -- amplification factor in the Euler update `x ← x(1 - hβ/2) + ε̂(hβ/2σ)`.
  -- ⚠ `logsnr` turned out to be the WRONG control: it concentrates steps at
  -- small σ and takes one enormous step across the region where β diverges,
  -- which made Euler 20× worse rather than better (25.1 against 1.25 at NFE 10,
  -- 100 % off-support). Kept because that measurement is the evidence.
  let logabar := args.any (· == "logabar")
  let flags := ["reuse", "strip", "tframes", "logsnr", "logabar"]
               ++ Ddpm.samplerNfe.map (·.1)
  for a in args do
    unless (a.toNat?.isSome || flags.contains a || toy2dTargets.any (·.1 == a)) do
      let names := String.intercalate ", " (toy2dTargets.map (·.1))
      let fl    := String.intercalate ", " flags
      throw <| IO.userError
        s!"unrecognised argument '{a}' — targets: {names}; flags: {fl}"
  let nums   := args.filterMap String.toNat?
  let steps  := (nums[0]?).getD 3000
  -- Sampler step count is an ARGUMENT, not a constant: the plan's open
  -- question is how many reverse steps a 2-D manifold actually needs, and
  -- the image demos' 50 is a convention nobody measured. Here it is a sweep.
  let nStepsArg := (nums[1]?).getD 50
  -- eta as a PERCENT (the arg parser has only `toNat?`): 0 = DDIM,
  -- deterministic; 100 = DDPM, full stochastic. The plan's §4 asks for both.
  let etaPct := (nums[2]?).getD 0
  -- `reuse` loads the saved checkpoint instead of retraining. Training is
  -- NOT reproducible run-to-run, so an eta sweep that retrained at each
  -- point would confound eta with a different model. Sweeps hold weights fixed.
  let reuse := args.any (· == "reuse")
  -- `strip` snapshots the cloud during sampling. It costs one extra dump per
  -- frame and nothing per step, so it is a flag rather than a second binary.
  let strip := args.any (· == "strip")
  let eta : Float := etaPct.toFloat / 100.0
  let spec   := diff2dDenoiser label
  let cfg    := diff2dConfig
  let B      := cfg.batchSize
  IO.eprintln s!"{spec.name}: {spec.totalParams} params, {steps} steps, eta={eta}"

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
  let dataPath := s!"data/toy2d/{target}.bin"
  unless ← System.FilePath.pathExists dataPath do
    throw <| IO.userError
      s!"{dataPath} missing — run: python3 preprocess_toy2d.py 8192 data/toy2d"
  let raw ← IO.FS.readBinFile dataPath
  let nPts := F32.size raw / 2
  IO.eprintln s!"  data: {nPts} points from {dataPath}"

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

  let ckpt := s!"{pfx}_params.bin"
  let haveCkpt ← System.FilePath.pathExists ckpt
  if reuse && haveCkpt then
    p ← IO.FS.readBinFile ckpt
    IO.eprintln s!"  reusing checkpoint {ckpt} — NOT training"
  else
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
  -- 4th numeric arg. A scaling sweep wants many arms, and the energy distance
  -- subsamples to 2048 anyway, so a smaller cloud per arm buys the curve.
  let nGen : Nat := (nums[3]?).getD 2048
  IO.eprintln s!"sampling {nGen} points..."
  let evalSess ← LowererSession.create evalArt
  let evalShapes := spec.evalShapesBA
  let bnPad ← F32.const spec.nBnStats.toUSize 0.0
  let evalParams := p.append bnPad
  let xSh1 := spec.xShape 1
  -- One point at a time keeps the eval graph at batch 1 (it is 18K params;
  -- the loop is still sub-second) and avoids a second compiled batch shape.
  let nSteps : Nat := nStepsArg
  let stride := Tmax / nSteps
  -- ⭐ Strip frames are spaced uniformly in log σ, where σ_t = √(1-ᾱ_t) is the
  -- NOISE scale of the marginal `p(x_t) = data ⊛ N(0, σ_t²)`. That is the axis
  -- the picture actually moves along, and it is the axis the diffusion
  -- literature plots against (uniform log σ is uniform log-SNR once ᾱ ≈ 1).
  -- ⚠ Both obvious alternatives are worse, and by measurement rather than
  -- taste. Uniform in t (the `tframes` branch below, run 2026-08-28) lands at
  -- σ = 1.00, 0.98, 0.92, 0.83, 0.71, 0.56, 0.39, 0.20 — FIVE of nine panels
  -- above σ = 0.7, i.e. barely-touched noise, with the resolution crammed into
  -- the last. Uniform in ᾱ is worse still (computed from the schedule, never
  -- shipped: σ = 1.00 … 0.35, reaching only t = 224), because the cosine
  -- schedule moves ᾱ fastest exactly where nothing is visible yet. Log σ lands at σ = 1.00, 0.58, 0.34, 0.20, 0.11, 0.06, 0.04,
  -- 0.02 — which brackets the 8-gaussians' own 0.05 mode width, so the modes
  -- appear ACROSS panels instead of between the last two.
  -- `tframes` restores the naive spacing; the difference is worth being able
  -- to see rather than take on trust.
  let uniformT := args.any (· == "tframes")
  let sigAt (k : Nat) : Float :=
    Float.sqrt (1.0 - F32.read alphaBar (Tmax - 1 - k * stride).toUSize)
  let mut frameKs : Array Nat := #[]
  if uniformT then
    for f in [:nStripFrames] do
      frameKs := frameKs.push (min (f * (max 1 (nSteps / nStripFrames))) (nSteps - 1))
  else
    let lo := Float.log (sigAt 0)
    let hi := Float.log (sigAt (nSteps - 1))
    for f in [:nStripFrames] do
      let goal := lo + (hi - lo) * (f.toFloat / nStripFrames.toFloat)
      -- σ DECREASES with k, so the frame is the first k to fall to the goal.
      let mut kf := 0
      for k in [:nSteps] do
        if Float.log (sigAt k) > goal then kf := k + 1
      -- Strictly increasing, so two frames can never claim the same step and
      -- leave one panel empty.
      let prev := (frameKs[frameKs.size - 1]?).getD 0
      frameKs := frameKs.push (min (max kf (if f == 0 then 0 else prev + 1)) (nSteps - 1))
  let mut frames : Array (Array ByteArray) := Array.replicate (nStripFrames + 1) #[]
  -- ⚠ `nSteps` is the NFE BUDGET, not the step count. Heun spends two
  -- evaluations per step, so it takes half as many — that is what makes the
  -- arms comparable at all.
  let nfe := (Ddpm.samplerNfe.lookup sampler).getD 1
  let solverSteps := max 1 (nSteps / nfe)
  if strip && sampler != "ddim" then
    throw <| IO.userError "strip is implemented for the ddim sampler only — its \
frame schedule is indexed against the DDIM step grid, and silently reusing it \
for a solver with a different grid would produce a figure labelled with the \
wrong times"
  IO.eprintln s!"  sampler={sampler} ({nfe} eval/step), NFE={nSteps} \
=> {solverSteps} solver steps"
  -- Uniform-in-t grid, deliberately matching DDIM's uniform index stride so the
  -- comparison is between SOLVERS and not between spacings.
  -- ⚠ It stops at t = 1/Tmax rather than 0: σ(0) = 0 exactly and the drift
  -- carries a 1/σ. The DDIM path fudges the same singularity with ᾱ_prev = 0.9999.
  let tHi := (Tmax - 1).toFloat / Tmax.toFloat
  let tLo := 1.0 / Tmax.toFloat
  -- σ(t) is invertible in closed form: σ² = 1 - cos²θ(t)/cos²θ(0), so
  -- t = (2(1+s)/π)·arccos(cosθ₀·√(1-σ²)) - s. That makes a log-σ-uniform grid
  -- exact rather than a bisection.
  let sHi := Ddpm.sigC tHi
  let sLo := Ddpm.sigC tLo
  let uHi := -(Float.log (Ddpm.abarC tHi))
  let uLo := -(Float.log (Ddpm.abarC tLo))
  let tAt := fun (k : Nat) =>
    if logabar then
      let u := uHi + (uLo - uHi) * k.toFloat / solverSteps.toFloat
      Ddpm.tOfAbar (Float.exp (-u))
    else if logsnr then
      let lg := Float.log sHi
                + (Float.log sLo - Float.log sHi) * k.toFloat / solverSteps.toFloat
      let sg := Float.exp lg
      Ddpm.tOfAbar (1.0 - sg * sg)
    else
      tHi + (tLo - tHi) * k.toFloat / solverSteps.toFloat
  -- One network evaluation at continuous time. ⚠ QUANTIZED: the encoder takes an
  -- integer index because the model was trained at t ∈ {0 … Tmax-1}, so a
  -- continuous solver queries the nearest one. Harmless at NFE ≥ 10 over 1000
  -- indices; below that the quantization, not the solver, is the limit.
  let epsAt := fun (xv : ByteArray) (t : Float) => do
    let r := Float.round (t * Tmax.toFloat)
    let r := if r < 0.0 then 0.0
             else if r > (Tmax - 1).toFloat then (Tmax - 1).toFloat else r
    let xc ← Ddpm.prependSinCosTScalar xv 1 nPer.toUSize 1 1
               r.toUInt64.toUSize nFreq.toUSize Tmax.toUSize
    LowererSession.forwardF32 evalSess spec.evalFnName evalParams evalShapes
      xc xSh1 1 nPer.toUSize
  let mut acc : Array ByteArray := #[]
  for i in [:nGen] do
    let mut x ← Ddpm.sampleNoise nPer.toUSize (i + 7919).toUSize
    if sampler == "ddim" then
     for k in [:nSteps] do
      if strip then
        match frameKs.findIdx? (· == k) with
        | some f => frames := frames.modify f (·.push x)
        | none   => pure ()
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
      -- Generalized DDIM (Song et al. eq. 12):
      --   x_{t-1} = a·x_t + b'·ε̂ + σ_t·z
      --   σ_t = η·√((1-ᾱ_prev)/(1-ᾱ_t))·√(1 - ᾱ_t/ᾱ_prev)
      --   b'  = √(1 - ᾱ_prev - σ_t²) − a·√(1-ᾱ_t)
      -- η = 0 collapses to the deterministic form (σ = 0, b' = b) and η = 1 is
      -- ancestral DDPM sampling. ⭐ No new primitive: `ddimStep` computes
      -- `a·x + b·e`, so the noise term is a second call with (1.0, σ_t, z).
      -- ᾱ decreases in t and tPrev < tCur, so ᾱ_t/ᾱ_prev < 1 and the inner
      -- root is real; the outer one is clamped because η→1 can drive
      -- 1 − ᾱ_prev − σ² marginally negative at the ends of the schedule.
      let sigma := eta * Float.sqrt ((1.0 - abP) / (1.0 - abT))
                       * Float.sqrt (1.0 - abT / abP)
      let inner := 1.0 - abP - sigma * sigma
      let b := Float.sqrt (max inner 0.0) - a * Float.sqrt (1.0 - abT)
      x ← Ddpm.ddimStep x epsHat a b nPer.toUSize
      if sigma > 0.0 then
        let z ← Ddpm.sampleNoise nPer.toUSize (i * 131071 + k * 8191 + 17).toUSize
        x ← Ddpm.ddimStep x z 1.0 sigma nPer.toUSize
    else
     -- ── the Score-SDE family, on the same weights ──────────────────────────
     -- score(x,t) = -ε̂/σ_t, so with f = -½βx and g² = β the probability-flow
     -- ODE is  dx/dt = -½β(t)·(x - ε̂/σ_t)  and the reverse SDE adds √β·dw̄.
     for k in [:solverSteps] do
      let t  := tAt k
      let tn := tAt (k + 1)
      let h  := tn - t                    -- negative; the integration runs backwards
      let b1 := Ddpm.betaC t
      let s1 := Ddpm.sigC t
      let e1 ← epsAt x t
      if sampler == "heun" then
        -- Euler predictor, re-evaluate at the endpoint, average the two drifts.
        let xt ← Ddpm.ddimStep x e1 (1.0 - h * 0.5 * b1) (h * 0.5 * b1 / s1) nPer.toUSize
        let b2 := Ddpm.betaC tn
        let s2 := Ddpm.sigC tn
        let e2 ← epsAt xt tn
        -- x + (h/2)(d₁ + d₂), accumulated term by term because the two drifts
        -- are affine in DIFFERENT points and no single `ddimStep` spans them.
        let mut a2 ← F32.scaleShift x (1.0 - h * 0.25 * b1) 0.0
        a2 ← F32.axpySlice a2 0 e1 0 nPer.toUSize (h * 0.25 * b1 / s1)
        a2 ← F32.axpySlice a2 0 xt 0 nPer.toUSize (-(h * 0.25 * b2))
        a2 ← F32.axpySlice a2 0 e2 0 nPer.toUSize (h * 0.25 * b2 / s2)
        x := a2
      else if sampler == "sde" then
        x ← Ddpm.ddimStep x e1 (1.0 - h * 0.5 * b1) (h * b1 / s1) nPer.toUSize
        let z ← Ddpm.sampleNoise nPer.toUSize (i * 131071 + k * 8191 + 17).toUSize
        x ← Ddpm.ddimStep x z 1.0 (Float.sqrt (b1 * (-h))) nPer.toUSize
      else   -- "euler": one naive step of the probability-flow ODE
        x ← Ddpm.ddimStep x e1 (1.0 - h * 0.5 * b1) (h * 0.5 * b1 / s1) nPer.toUSize
    if strip then
      frames := frames.modify nStripFrames (·.push x)
    acc := acc.push x
  let gen := F32.concat acc
  let outPath :=
    s!".lake/build/diffusion2d_samples_{target}_{sampler}{if logabar then "_logabar" else if logsnr then "_logsnr" else ""}\
_s{steps}_n{nSteps}_e{etaPct}.bin"
  IO.FS.writeBinFile outPath gen
  IO.FS.writeBinFile s!".lake/build/diffusion2d_samples_{target}.bin" gen
  IO.eprintln s!"wrote .lake/build/diffusion2d_samples_{target}.bin ({F32.size gen / 2} points)"

  if strip then
    -- One manifest line per panel: index, the t it was taken at, the file.
    -- The renderer reads this rather than parsing filenames, so the panel
    -- ORDER is data instead of a sort convention.
    let mut manifest := ""
    for f in [:nStripFrames + 1] do
      let final := f == nStripFrames
      let t := if final then 0 else Tmax - 1 - (frameKs[f]!) * stride
      -- σ is on the line because it, not t, is what the panels are spaced by;
      -- a reader checking the figure should not have to re-derive it.
      let sg := if final then 0.0 else sigAt (frameKs[f]!)
      let path := s!".lake/build/diffusion2d_strip_{target}_f{f}.bin"
      IO.FS.writeBinFile path (F32.concat frames[f]!)
      manifest := manifest ++ s!"{f} {t} {sg} {path}\n"
    let mpath := s!".lake/build/diffusion2d_strip_{target}.txt"
    IO.FS.writeFile mpath manifest
    IO.eprintln s!"wrote {nStripFrames + 1} strip frames + {mpath}"
    IO.eprintln s!"▶ render it: python3 scripts/toy2d_strip.py {target}"

  IO.eprintln s!"▶ score it: python3 scripts/toy2d_metrics.py --target={target}"
