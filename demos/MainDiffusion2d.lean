import LeanMlir

/-! Diffusion and flow matching on a 2-D target — `planning/archive/diffusion_2d_demo.md`
    and `planning/boltzmann_generator_demo.md`.

    The diffusion demo you can be *wrong* about. Every image DDPM here
    succeeds or fails by "does that look like a digit to you"; on a 2-D
    target the ground truth is a second point cloud, so correctness is a
    number (cell recall and energy distance) instead of a judgement, and it
    trains in seconds on 18,178 params rather than 7 h on 3M.

    ⭐ No new codegen primitives. The denoiser is `.dense`/`.relu`, which
    Chapters 1-2 already prove, and the time conditioning reuses
    `Ddpm.prependSinCosT` with `H = W = 1` — the image path's own encoding,
    applied to a 2-vector instead of a plane.

    ⭐ FIVE targets. Four point clouds (8-gaussians mode collapse, spiral
    corner-cutting, two-moons over-smoothing, checkerboard leakage) and one
    DENSITY: `muller_brown`, exp(-U/kT) on the Müller-Brown surface, whose
    training set is a Langevin chain and whose every score is exact by
    quadrature. The target name is an argument and it flows into `spec.name`,
    so each one owns its MLIR, checkpoint and samples without a flag anywhere.

    ⭐ `flow` trains the SAME net on the SAME rank-2 MSE block as a
    flow-matching model: the interpolant is `x_t = (1-t)x0 + t·ε` with target
    `v = ε - x0` instead of the cosine-schedule `x_t` with target `ε`
    (`Ddpm.flowStepInputs`), and the sampler is Euler on `dx/dt = v` from
    t = 1 to 0 (`fm-euler`, or `fm-heun`). Without the flag the exe is the
    DDPM baseline on the same target — the interpolant row of the table.
    `ot` pairs each batch's noise to its data by a minimum-cost assignment
    (minibatch OT), `reflow` retrains on the trained flow's own (noise,
    sample) pairs; both are straightening moves and both are the same exe.

    ⭐ `logp` integrates the divergence of `v` beside the state, so every
    sample comes with its exact log-density by the continuity equation
    (four extra forwards per step, central differences); `nll` runs the
    same integration the other way on the exact reference draw. That is the
    KL column, and the importance weights that make a Boltzmann generator.

    ⭐ The eval graph is compiled at batch `nGen`, one forward per solver
    step for the whole cloud. The archived version ran one point at a time.

    Usage:
      python3 preprocess_toy2d.py 8192 data/toy2d
      python3 preprocess_boltzmann.py
      lake exe diffusion-2d                          # 8-gaussians: train + sample + dump
      lake exe diffusion-2d 20000 200 25             # steps, NFE budget, eta%
      lake exe diffusion-2d spiral                   # any of the targets
      lake exe diffusion-2d two_moons reuse strip    # the reverse-process strip
      lake exe diffusion-2d muller_brown flow 20000 50 logp          # the Boltzmann generator
      lake exe diffusion-2d muller_brown flow reuse fm-euler 10      # NFE sweep on one checkpoint
      lake exe diffusion-2d muller_brown 20000 50 euler              # the DDPM path, same target
      python3 scripts/toy2d_metrics.py --target=spiral
      python3 scripts/boltzmann_metrics.py score "flow NFE 50=<samples.bin>" --gate
-/

/-- Frequencies in the sinusoidal time encoding; the input is the 2-vector
    plus `2 * nFreq` channels of `[sin(t·ω_k), cos(t·ω_k)]`. -/
def nFreq : Nat := 4
def condDim : Nat := 2 + 2 * nFreq

/-- The targets: arg name, the display name that goes into `spec.name`, and
    the data file. `buildPrefix` is derived from that name, so naming the
    target here is the whole of what keeps five sets of artifacts apart. ⚠ The
    8-gaussians display name must stay `8-gaussians` — it is what the existing
    checkpoint and MLIR on disk are keyed by. -/
def toy2dTargets : List (String × String × String) :=
  [("eight_gaussians", "8-gaussians",  "data/toy2d/eight_gaussians.bin"),
   ("spiral",          "spiral",       "data/toy2d/spiral.bin"),
   ("two_moons",       "two-moons",    "data/toy2d/two_moons.bin"),
   ("checkerboard",    "checkerboard", "data/toy2d/checkerboard.bin"),
   ("muller_brown",    "muller-brown", "data/boltzmann/mb_kT20.bin")]

/-- 18,178 params at 128 hidden, against the tiny image UNet's 118K. The output
    is `[B, 2]` — a predicted ε (or velocity) per point — which is what puts
    this on the codegen's rank-2 DDPM branch rather than the 4-D image one. -/
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

/-- Points whose full trajectory the flow samplers keep for the figure. -/
def nPathPoints : Nat := 128

@[inline] def pushF32 (acc : ByteArray) (x : Float) : ByteArray :=
  let u : UInt32 := x.toFloat32.toBits
  (((acc.push (u &&& 0xff).toUInt8).push ((u >>> 8) &&& 0xff).toUInt8).push
    ((u >>> 16) &&& 0xff).toUInt8).push ((u >>> 24) &&& 0xff).toUInt8

/-- The integer the time channel takes for a continuous flow time `t ∈ [0, 1]`.
    ⚠ Train and sample must agree on this map; `Ddpm.flowStepInputs` spells the
    same `round(t · Tmax)`. -/
def flowIdx (t : Float) : Nat :=
  let r := Float.round (t * Tmax.toFloat)
  let r := if r < 0.0 then 0.0 else if r > Tmax.toFloat then Tmax.toFloat else r
  r.toUInt64.toNat

/-- A cloud of `m` standard-normal 2-vectors, one `Ddpm.sampleNoise` call per
    point at the seed `f i`. Per-point seeds are kept rather than one long
    draw so the flow and DDPM arms start from the SAME noise, point for point. -/
def noiseCloud (m : Nat) (f : Nat → Nat) : IO ByteArray := do
  let mut acc : Array ByteArray := Array.mkEmpty m
  for i in [:m] do
    acc := acc.push (← Ddpm.sampleNoise 2 (f i).toUSize)
  return F32.concat acc

/-- One batched pass of the flow ODE `dx/dt = v(x, t)` on a cloud of `m`
    points with the velocity oracle `vAt`, `nSteps` uniform steps from t = 1
    to 0 (`forward := false`, sampling) or 0 to 1 (`forward := true`, the
    likelihood of given points). Euler, or Heun when `heun`.

    With `withDiv` the density is carried beside the state. In 2-D the whole
    Jacobian `J = ∂v/∂x` is two central differences (four extra oracle calls
    per step), and each Euler step `x ← x + h·v` changes `log ρ` by exactly
    `-log|det(I + hJ)|`. That is the continuity equation `d/dt log ρ = -∇·v`
    for the DISCRETE map actually run: the trace form `-h·tr J` is its first
    order and biases the KL by ~0.03 nats at NFE 50, enough to make it
    negative. Returns the cloud, `Σ log|det(I + hJ)|` per point (so
    `log ρ_data = log N(z) - Σ` backward and `log N(x₁) + Σ` forward), the
    kinetic energy `∫ E|v|² dt`, and the trajectory of the first `keepPaths`
    points, one frame per step. -/
def flowOde (vAt : ByteArray → Float → IO ByteArray) (x0 : ByteArray) (m : Nat)
    (nSteps : Nat) (forward heun withDiv : Bool) (keepPaths : Nat := 0)
    : IO (ByteArray × Array Float × Float × Array ByteArray) := do
  let n := (2 * m).toUSize
  let hFd : Float := 1e-3
  let mut e0 := ByteArray.empty
  let mut e1 := ByteArray.empty
  if withDiv then
    for _ in [:m] do
      e0 := pushF32 (pushF32 e0 1.0) 0.0
      e1 := pushF32 (pushF32 e1 0.0) 1.0
  let mut x := x0
  let mut divInt : Array Float := Array.replicate m 0.0
  let mut kin := 0.0
  let mut frames : Array ByteArray :=
    if keepPaths > 0 then #[F32.slice x0 0 (2 * keepPaths)] else #[]
  for k in [:nSteps] do
    let t  := if forward then k.toFloat / nSteps.toFloat else 1.0 - k.toFloat / nSteps.toFloat
    let tn := if forward then (k + 1).toFloat / nSteps.toFloat else 1.0 - (k + 1).toFloat / nSteps.toFloat
    let h := tn - t
    let v1 ← vAt x t
    if withDiv then
      -- Column d of J by central differences along axis d; `axpySlice` copies
      -- because `x` is shared, so the probe never touches the state.
      let xp0 ← F32.axpySlice x 0 e0 0 n hFd
      let xm0 ← F32.axpySlice x 0 e0 0 n (-hFd)
      let xp1 ← F32.axpySlice x 0 e1 0 n hFd
      let xm1 ← F32.axpySlice x 0 e1 0 n (-hFd)
      let vp0 ← vAt xp0 t
      let vm0 ← vAt xm0 t
      let vp1 ← vAt xp1 t
      let vm1 ← vAt xm1 t
      for i in [:m] do
        let r0 := (2 * i).toUSize
        let r1 := (2 * i + 1).toUSize
        let j00 := (F32.read vp0 r0 - F32.read vm0 r0) / (2.0 * hFd)
        let j10 := (F32.read vp0 r1 - F32.read vm0 r1) / (2.0 * hFd)
        let j01 := (F32.read vp1 r0 - F32.read vm1 r0) / (2.0 * hFd)
        let j11 := (F32.read vp1 r1 - F32.read vm1 r1) / (2.0 * hFd)
        let det := (1.0 + h * j00) * (1.0 + h * j11) - h * h * j01 * j10
        divInt := divInt.set! i (divInt[i]! + Float.log (Float.abs det))
    kin := kin + (F32.dotSlice v1 0 v1 0 n) / m.toFloat * Float.abs h
    if heun then
      let xp ← F32.axpySlice x 0 v1 0 n h
      let v2 ← vAt xp tn
      let mut a ← F32.axpySlice x 0 v1 0 n (h * 0.5)
      a ← F32.axpySlice a 0 v2 0 n (h * 0.5)
      x := a
    else
      x ← F32.axpySlice x 0 v1 0 n h
    if keepPaths > 0 then frames := frames.push (F32.slice x 0 (2 * keepPaths))
  return (x, divInt, kin, frames)

/-- `log N(x; 0, I₂)` per point of a cloud. -/
def logNormal2 (x : ByteArray) (m : Nat) : Array Float := Id.run do
  let mut out : Array Float := Array.mkEmpty m
  for i in [:m] do
    let a := F32.read x (2 * i).toUSize
    let b := F32.read x (2 * i + 1).toUSize
    out := out.push (-(a * a + b * b) / 2.0 - Float.log (2.0 * Ddpm.piF))
  return out

def floatsToBytes (a : Array Float) : ByteArray :=
  a.foldl pushF32 ByteArray.empty

def main (args : List String) : IO Unit := do
  -- The target is matched by NAME anywhere in the args, the same way `reuse`
  -- is, so it composes with the positional numeric arguments.
  let target := (args.find? fun a => toy2dTargets.any (·.1 == a)).getD "eight_gaussians"
  let (label, dataPath) := (toy2dTargets.lookup target).getD ("8-gaussians", "data/toy2d/eight_gaussians.bin")
  -- ⭐ `flow` is the flow-matching arm; `ot` and `reflow` are its two coupling
  -- variants (both imply `flow`). Each arm owns its artifacts through the
  -- build tag, so the DDPM baseline and the three flow arms of one target
  -- never share a checkpoint.
  let ot     := args.any (· == "ot")
  let reflow := args.any (· == "reflow")
  let flow   := args.any (· == "flow") || ot || reflow
  if ot && reflow then
    throw <| IO.userError "ot and reflow are two different couplings — pick one"
  let arm := if reflow then "flow-reflow" else if ot then "flow-ot" else if flow then "flow" else ""
  -- ⭐ `ddim` is the shipped sampler and the DDPM arm's default; `fm-euler` is
  -- the flow arm's. The other DDPM samplers are the Score-SDE family on the
  -- SAME weights: `euler` integrates the probability-flow ODE naively, `heun`
  -- does it to second order, `sde` integrates the reverse SDE with
  -- Euler-Maruyama. The flow arm's ODE has no schedule: `fm-euler` /
  -- `fm-heun` integrate `dx/dt = v` on a uniform grid in t.
  let allSamplers := Ddpm.samplerNfe ++ Ddpm.flowSamplerNfe
  let sampler := (args.find? fun a => allSamplers.any (·.1 == a)).getD (if flow then "fm-euler" else "ddim")
  let isFm := Ddpm.flowSamplerNfe.any (·.1 == sampler)
  if flow && !isFm then
    throw <| IO.userError s!"sampler '{sampler}' integrates the VP schedule, and the flow arm's \
net predicts a velocity, not ε — use fm-euler or fm-heun"
  if !flow && isFm then
    throw <| IO.userError s!"sampler '{sampler}' integrates dx/dt = v, and this arm's net predicts \
ε — add `flow`, or use ddim / euler / heun / sde"
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
  if isFm && (logsnr || logabar) then
    throw <| IO.userError "logsnr / logabar are VP-schedule grids; the linear path has no σ \
schedule to space by, its solvers use a uniform grid in t"
  -- `logp`: the exact log-density of every sample by the continuity equation.
  -- `nll`: the same integration forward on the exact reference draw.
  -- `field`: dump v_θ on a (lattice × t) grid for the field error of §6.2.
  let logp  := args.any (· == "logp")
  let nll   := args.any (· == "nll")
  let field := args.any (· == "field")
  if (logp || nll || field) && !flow then
    throw <| IO.userError "logp / nll / field are flow-arm modes (they integrate the divergence \
of the velocity field)"
  if logp && sampler != "fm-euler" then
    throw <| IO.userError "logp integrates the divergence along the Euler path; use fm-euler"
  let flags := ["reuse", "strip", "tframes", "logsnr", "logabar", "flow", "ot", "reflow",
                "logp", "nll", "field"] ++ allSamplers.map (·.1)
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
  -- deterministic; 100 = DDPM, full stochastic.
  let etaPct := (nums[2]?).getD 0
  -- `reuse` loads the saved checkpoint instead of retraining. Training is
  -- NOT reproducible run-to-run, so an eta or NFE sweep that retrained at each
  -- point would confound the sweep with a different model. Sweeps hold weights fixed.
  let reuse := args.any (· == "reuse")
  -- `strip` snapshots the cloud during sampling. It costs one extra dump per
  -- frame and nothing per step, so it is a flag rather than a second binary.
  let strip := args.any (· == "strip")
  let eta : Float := etaPct.toFloat / 100.0
  let spec := if arm.isEmpty then diff2dDenoiser label else (diff2dDenoiser label).withBuildTag arm
  let cfg  := diff2dConfig
  let B    := cfg.batchSize
  -- 4th numeric arg. A scaling sweep wants many arms, and the energy distance
  -- subsamples to 2048 anyway, so a smaller cloud per arm buys the curve.
  let nGen : Nat := (nums[3]?).getD 2048
  IO.eprintln s!"{spec.name}{if arm.isEmpty then "" else " [" ++ arm ++ "]"}: {spec.totalParams} params, \
{steps} steps, sampler={sampler}, eta={eta}, nGen={nGen}"

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

  -- ⭐ The eval graph at batch nGen: one forward per solver step for the whole
  -- cloud. The divergence integration multiplies the evaluations by five and
  -- the field dump by a lattice, and this is what keeps everything under a minute.
  let evalMlir := MlirCodegen.generateEval spec nGen
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" evalMlir

  let trainArt ← NetSpec.graphArtifact pfx "train_step"
  let evalArt  ← NetSpec.graphArtifact pfx "fwd_eval"

  let nP := spec.totalParams
  let allShapes  := spec.shapesBA
  let bnShapes   := spec.bnShapesBA
  let evalShapes := spec.evalShapesBA
  let xSh        := spec.xShape B
  let xShN       := spec.xShape nGen
  let nT         := 3 * nP
  let batch      : USize := B.toUSize
  let nPer       : Nat := 2
  let bnPad ← F32.const spec.nBnStats.toUSize 0.0
  let m := nGen
  let nM := (2 * m).toUSize

  -- One network evaluation of a whole cloud at flow time `t`, through a given
  -- session and parameter set. The flow arm's net returns the velocity.
  let vAtWith := fun (sess : LowererSession) (params : ByteArray) (xv : ByteArray) (t : Float) => do
    let xc ← Ddpm.prependSinCosTScalar xv m.toUSize nPer.toUSize 1 1
               (flowIdx t).toUSize nFreq.toUSize Tmax.toUSize
    LowererSession.forwardF32 sess spec.evalFnName params evalShapes xc xShN m.toUSize nPer.toUSize

  -- ── data: flat f32 LE [N, 2] ──
  -- The reflow arm's data is not a file: it is the trained flow arm's own
  -- (noise, sample) pairs, drawn here through that arm's checkpoint at NFE 50.
  let mut raw := ByteArray.empty
  let mut epsData := ByteArray.empty
  if reflow then
    let parent := (diff2dDenoiser label).withBuildTag "flow"
    let ppfx := parent.buildPrefix
    unless ← System.FilePath.pathExists s!"{ppfx}_params.bin" do
      throw <| IO.userError s!"{ppfx}_params.bin missing — reflow retrains on the flow arm's \
pairs; run `lake exe diffusion-2d {target} flow` first"
    IO.FS.writeFile s!"{ppfx}_fwd_eval.mlir" (MlirCodegen.generateEval parent m)
    let pSess ← LowererSession.create (← NetSpec.graphArtifact ppfx "fwd_eval")
    let pParams := (← IO.FS.readBinFile s!"{ppfx}_params.bin").append bnPad
    let nChunks := 18
    IO.eprintln s!"  reflow: drawing {nChunks * m} (noise, sample) pairs from {ppfx} at NFE 50..."
    let t0 ← IO.monoMsNow
    let mut zs : Array ByteArray := #[]
    let mut xs : Array ByteArray := #[]
    for ch in [:nChunks] do
      let z ← noiseCloud m (fun i => ch * m + i + 424242)
      let (x, _, _, _) ← flowOde (vAtWith pSess pParams) z m 50 false false false
      zs := zs.push z
      xs := xs.push x
    raw := F32.concat xs
    epsData := F32.concat zs
    let t1 ← IO.monoMsNow
    IO.eprintln s!"  {F32.size raw / 2} pairs in {t1 - t0} ms"
  else
    unless ← System.FilePath.pathExists dataPath do
      let hint := if target == "muller_brown" then "python3 preprocess_boltzmann.py"
                  else "python3 preprocess_toy2d.py 8192 data/toy2d"
      throw <| IO.userError s!"{dataPath} missing — run: {hint}"
    raw ← IO.FS.readBinFile dataPath
  let nPts := F32.size raw / 2
  IO.eprintln s!"  data: {nPts} points{if reflow then " (reflow pairs)" else " from " ++ dataPath}"

  let params ← spec.heInitParams
  let adamM ← F32.const nP.toUSize 0.0
  let adamV ← F32.const nP.toUSize 0.0
  let alphaBar ← Ddpm.cosineSchedule Tmax.toUSize

  let mut p := params
  let mut mm := adamM
  let mut vv := adamV
  let bpE := nPts / B

  let ckpt := s!"{pfx}_params.bin"
  let haveCkpt ← System.FilePath.pathExists ckpt
  if reuse && haveCkpt then
    p ← IO.FS.readBinFile ckpt
    IO.eprintln s!"  reusing checkpoint {ckpt} — NOT training"
  else
    let sess ← LowererSession.create trainArt
    IO.eprintln "  session loaded"
    IO.eprintln s!"training: {steps} steps, batch={B}, lr={cfg.learningRate}\
{if flow then ", flow matching (" ++ (if ot then "minibatch-OT" else if reflow then "reflow" else "independent") ++ " coupling)" else ", DDPM"}"
    let t0 ← IO.monoMsNow
    for gs in [:steps] do
      let bi := gs % bpE
      let x0 := F32.slice raw (bi * B * nPer) (B * nPer)
      let (xt, target, tba) ← if flow then do
          -- The linear path: x_t = (1-t)x0 + tε, target ε - x0. The MSE
          -- block does not know whether the target is ε or v.
          let epsIn := if reflow then F32.slice epsData (bi * B * nPer) (B * nPer) else ByteArray.empty
          let mode : USize := if reflow then 1 else if ot then 2 else 0
          let (xt, rest) ← Ddpm.flowStepInputs x0 epsIn batch nPer.toUSize gs.toUSize Tmax.toUSize mode
          pure (xt, rest.1, rest.2)
        else do
          let (xt, rest) ← Ddpm.stepInputs x0 alphaBar batch nPer.toUSize gs.toUSize
          pure (xt, rest.1, rest.2)
      -- Time conditioning with H = W = 1: the image encoder applied to a point.
      let xtc ← Ddpm.prependSinCosT xt tba batch nPer.toUSize 1 1
                  nFreq.toUSize Tmax.toUSize
      let packed := (p.append mm).append vv
      let out ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName
                  packed allShapes xtc xSh target
                  cfg.learningRate (gs+1).toFloat
                  bnShapes batch 2 1 1
      let loss := F32.extractLoss out nT
      p := F32.slice out 0 nP
      mm := F32.slice out nP nP
      vv := F32.slice out (2 * nP) nP
      if gs % 500 == 0 || gs + 1 == steps then
        IO.eprintln s!"  step {gs}/{steps}: loss={loss}"
    let t1 ← IO.monoMsNow
    IO.eprintln s!"trained in {t1-t0}ms"
    IO.FS.writeBinFile ckpt p

  -- ── sampling: the whole cloud at once through the batch-nGen eval graph ──
  IO.eprintln s!"sampling {nGen} points..."
  let evalSess ← LowererSession.create evalArt
  let evalParams := p.append bnPad
  let vAt := vAtWith evalSess evalParams
  let nSteps : Nat := nStepsArg
  let stride := Tmax / nSteps
  -- ⚠ `nSteps` is the NFE BUDGET, not the step count. Heun spends two
  -- evaluations per step, so it takes half as many — that is what makes the
  -- arms comparable at all.
  let nfe := (allSamplers.lookup sampler).getD 1
  let solverSteps := max 1 (nSteps / nfe)
  if strip && sampler != "ddim" && sampler != "fm-euler" then
    throw <| IO.userError "strip is implemented for the ddim and fm-euler samplers only — its \
frame schedule is indexed against those step grids, and silently reusing it \
for a solver with a different grid would produce a figure labelled with the \
wrong times"
  IO.eprintln s!"  sampler={sampler} ({nfe} eval/step), NFE={nSteps} => {solverSteps} solver steps"

  -- The starting noise: per-point seeds, identical across arms and samplers.
  let z ← noiseCloud m (fun i => i + 7919)
  let mut x := z
  let mut logpArr : Array Float := #[]
  let mut kinetic : Float := 0.0
  let mut paths : Array ByteArray := #[]
  let mut frames : Array (Nat × Float × ByteArray) := #[]   -- (t index, σ or t, cloud)
  if isFm then
    -- Straight to `flowOde`: Euler or Heun, the divergence beside the state
    -- when `logp`, the first `nPathPoints` trajectories for the figure.
    let (xf, divInt, kin, fr) ← flowOde vAt z m solverSteps false (sampler == "fm-heun") logp nPathPoints
    x := xf
    kinetic := kin
    paths := fr
    if logp then
      let lz := logNormal2 z m
      logpArr := (Array.range m).map fun i => lz[i]! - divInt[i]!
    if strip then
      -- Uniform in t: the linear path's noise scale IS t, so the axis the
      -- picture moves along is the one the grid is uniform in.
      for f in [:nStripFrames] do
        let k := min (f * (max 1 (solverSteps / nStripFrames))) (solverSteps - 1)
        let t := 1.0 - k.toFloat / solverSteps.toFloat
        frames := frames.push (flowIdx t, t, fr[k]!)
      frames := frames.push (0, 0.0, F32.slice x 0 (2 * nPathPoints))
  else
    -- Frame schedule of the DDIM strip.
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
    -- schedule moves ᾱ fastest exactly where nothing is visible yet. Log σ lands
    -- at σ = 1.00, 0.58, 0.34, 0.20, 0.11, 0.06, 0.04, 0.02 — which brackets the
    -- 8-gaussians' own 0.05 mode width, so the modes appear ACROSS panels
    -- instead of between the last two. `tframes` restores the naive spacing.
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
    -- Uniform-in-t grid for the continuous solvers, deliberately matching
    -- DDIM's uniform index stride so the comparison is between SOLVERS and
    -- not between spacings.
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
      let xc ← Ddpm.prependSinCosTScalar xv m.toUSize nPer.toUSize 1 1
                 r.toUInt64.toUSize nFreq.toUSize Tmax.toUSize
      LowererSession.forwardF32 evalSess spec.evalFnName evalParams evalShapes
        xc xShN m.toUSize nPer.toUSize
    if sampler == "ddim" then
      for k in [:nSteps] do
        if strip then
          match frameKs.findIdx? (· == k) with
          | some _ => frames := frames.push (Tmax - 1 - k * stride, sigAt k, x)
          | none   => pure ()
        let tCur := Tmax - 1 - k * stride
        let tPrev := if k + 1 == nSteps then 0 else Tmax - 1 - (k+1) * stride
        let xc ← Ddpm.prependSinCosTScalar x m.toUSize nPer.toUSize 1 1
                   tCur.toUSize nFreq.toUSize Tmax.toUSize
        let epsHat ← LowererSession.forwardF32 evalSess spec.evalFnName evalParams evalShapes
                       xc xShN m.toUSize nPer.toUSize
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
        let sigma := eta * Float.sqrt ((1.0 - abP) / (1.0 - abT))
                         * Float.sqrt (1.0 - abT / abP)
        let inner := 1.0 - abP - sigma * sigma
        let b := Float.sqrt (max inner 0.0) - a * Float.sqrt (1.0 - abT)
        x ← Ddpm.ddimStep x epsHat a b nM
        if sigma > 0.0 then
          let zk ← noiseCloud m (fun i => i * 131071 + k * 8191 + 17)
          x ← Ddpm.ddimStep x zk 1.0 sigma nM
      if strip then frames := frames.push (0, 0.0, x)
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
          let xt ← Ddpm.ddimStep x e1 (1.0 - h * 0.5 * b1) (h * 0.5 * b1 / s1) nM
          let b2 := Ddpm.betaC tn
          let s2 := Ddpm.sigC tn
          let e2 ← epsAt xt tn
          -- x + (h/2)(d₁ + d₂), accumulated term by term because the two drifts
          -- are affine in DIFFERENT points and no single `ddimStep` spans them.
          let mut a2 ← F32.scaleShift x (1.0 - h * 0.25 * b1) 0.0
          a2 ← F32.axpySlice a2 0 e1 0 nM (h * 0.25 * b1 / s1)
          a2 ← F32.axpySlice a2 0 xt 0 nM (-(h * 0.25 * b2))
          a2 ← F32.axpySlice a2 0 e2 0 nM (h * 0.25 * b2 / s2)
          x := a2
        else if sampler == "sde" then
          x ← Ddpm.ddimStep x e1 (1.0 - h * 0.5 * b1) (h * b1 / s1) nM
          let zk ← noiseCloud m (fun i => i * 131071 + k * 8191 + 17)
          x ← Ddpm.ddimStep x zk 1.0 (Float.sqrt (b1 * (-h))) nM
        else   -- "euler": one naive step of the probability-flow ODE
          x ← Ddpm.ddimStep x e1 (1.0 - h * 0.5 * b1) (h * 0.5 * b1 / s1) nM

  let armSfx := if arm.isEmpty then "" else "-" ++ arm
  let gridSfx := if logabar then "_logabar" else if logsnr then "_logsnr" else ""
  let outPath := s!".lake/build/diffusion2d_samples_{target}{armSfx}_{sampler}{gridSfx}\
_s{steps}_n{nSteps}_e{etaPct}.bin"
  let stem := (outPath.toList.take (outPath.length - 4)).asString
  IO.FS.writeBinFile outPath x
  IO.FS.writeBinFile s!".lake/build/diffusion2d_samples_{target}.bin" x
  IO.FS.writeBinFile s!"{stem}.noise.bin" z
  IO.eprintln s!"wrote {outPath} ({F32.size x / 2} points) + .noise.bin"
  if isFm then
    -- Liu's straightness of the learned ODE: with the coupling the ODE itself
    -- induces, (z, x), how far v_θ along the straight line between them is from
    -- the constant velocity z - x. Zero iff one Euler step is exact — the
    -- quantity reflow drives down.
    let nS := 10
    let d ← F32.subtract z x
    let mut straight := 0.0
    for j in [:nS] do
      let t := (j.toFloat + 0.5) / nS.toFloat
      let mut xt ← F32.scaleShift x (1.0 - t) 0.0
      xt ← F32.axpySlice xt 0 z 0 nM t
      let v ← vAt xt t
      let e ← F32.subtract v d
      straight := straight + (F32.dotSlice e 0 e 0 nM) / m.toFloat / nS.toFloat
    IO.println s!"flow {arm} {sampler} NFE {nSteps}: kinetic energy ∫E|v|²dt = {kinetic}, \
straightness ∫E|v(x_t,t) - (z - x)|²dt = {straight}"
    IO.FS.writeBinFile s!"{stem}.paths.bin" (F32.concat paths)
    IO.eprintln s!"wrote {stem}.paths.bin ({paths.size} frames × {nPathPoints} points)"
    if logp then
      IO.FS.writeBinFile s!"{stem}.logp.bin" (floatsToBytes logpArr)
      let mean := (logpArr.foldl (· + ·) 0.0) / m.toFloat
      IO.eprintln s!"wrote {stem}.logp.bin (mean log p_θ = {mean})"

  if nll then
    -- The other direction: the exact reference draw, integrated data → noise
    -- with the divergence, gives log p_θ on the exact density's own samples,
    -- hence KL(exact ‖ model). Chunked at the graph's batch, zero-padded.
    let refPath := "data/boltzmann/mb_kT20_ref.bin"
    let refRaw ← IO.FS.readBinFile refPath
    let nRef := F32.size refRaw / 2
    let nChunks := (nRef + m - 1) / m
    let padded := refRaw.append (← F32.const ((nChunks * m - nRef) * 2).toUSize 0.0)
    let mut out : Array Float := #[]
    for ch in [:nChunks] do
      let xr := F32.slice padded (ch * m * 2) (m * 2)
      let (x1, divInt, _, _) ← flowOde vAt xr m solverSteps true false true
      let l1 := logNormal2 x1 m
      for i in [:m] do
        out := out.push (l1[i]! + divInt[i]!)
    let outRef := out.extract 0 nRef
    IO.FS.writeBinFile s!"{stem}.reflogp.bin" (floatsToBytes outRef)
    let mean := (outRef.foldl (· + ·) 0.0) / nRef.toFloat
    IO.eprintln s!"wrote {stem}.reflogp.bin ({nRef} exact points, mean log p_θ = {mean})"

  if field then
    -- v_θ on a lattice × t grid, for the field error against the exact
    -- marginal velocity (x - E[x0 | x_t]) / t that the scorer computes by quadrature.
    let side := 64
    let lo : Float := -2.2
    let hi : Float := 2.2
    let ts : List Float := [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    let mut lat := ByteArray.empty
    for iy in [:side] do
      for ix in [:side] do
        lat := pushF32 (pushF32 lat (lo + (hi - lo) * ix.toFloat / (side - 1).toFloat))
                              (lo + (hi - lo) * iy.toFloat / (side - 1).toFloat)
    let nL := side * side
    let nChunks := (nL + m - 1) / m
    let padded := lat.append (← F32.const ((nChunks * m - nL) * 2).toUSize 0.0)
    let mut out : Array ByteArray := #[]
    for t in ts do
      for ch in [:nChunks] do
        let xl := F32.slice padded (ch * m * 2) (m * 2)
        let v ← vAt xl t
        let keep := min m (nL - ch * m)
        out := out.push (F32.slice v 0 (keep * 2))
    IO.FS.writeBinFile s!"{stem}.field.bin" (F32.concat out)
    IO.FS.writeFile s!"{stem}.field.txt"
      (s!"{ts.length} {side} {lo} {hi} " ++ String.intercalate " " (ts.map toString) ++ "\n")
    IO.eprintln s!"wrote {stem}.field.bin ({ts.length} × {side}×{side} lattice)"

  if strip then
    -- One manifest line per panel: index, the t it was taken at, the file.
    -- The renderer reads this rather than parsing filenames, so the panel
    -- ORDER is data instead of a sort convention. σ is on the line because
    -- it, not t, is what the DDIM panels are spaced by; for the flow the
    -- noise scale is t itself.
    let mut manifest := ""
    for f in [:frames.size] do
      let (tIdx, sg, cloud) := frames[f]!
      let path := s!".lake/build/diffusion2d_strip_{target}{armSfx}_f{f}.bin"
      IO.FS.writeBinFile path cloud
      manifest := manifest ++ s!"{f} {tIdx} {sg} {path}\n"
    let mpath := s!".lake/build/diffusion2d_strip_{target}{armSfx}.txt"
    IO.FS.writeFile mpath manifest
    IO.eprintln s!"wrote {frames.size} strip frames + {mpath}"
    IO.eprintln s!"▶ render it: python3 scripts/toy2d_strip.py {target}"

  if target == "muller_brown" then
    IO.eprintln s!"▶ score it: python3 scripts/boltzmann_metrics.py score \"{arm} {sampler} NFE {nSteps}={outPath}\" --gate"
  else
    IO.eprintln s!"▶ score it: python3 scripts/toy2d_metrics.py --target={target}"
