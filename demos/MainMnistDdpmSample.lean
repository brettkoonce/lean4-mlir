import LeanMlir

/-! Sample digit-shaped images from a trained tiny DDPM checkpoint.

Pipeline:
  1. Compile the eval-forward vmfb from the training spec (idempotent;
     reuses cache if present).
  2. Load trained params + BN running stats.
  3. Pick a `nSteps`-subsampled DDIM schedule from the full `T = 1000`
     cosine alphaBar table.
  4. Initialize `x_T ~ N(0, I)` for `B` images.
  5. Loop t = T-1 → 0 over the subsampled schedule:
       ε_θ ← forward_eval(x_t)
       a   = √ᾱ_{t-1} / √ᾱ_t
       b   = √(1-ᾱ_{t-1}) - a·√(1-ᾱ_t)
       x_{t-1} = a·x_t + b·ε_θ
  6. Render final `x_0` batch as a 4×4 grid PPM.

No time conditioning in this MVP — the model has no input telling it
which `t` it's denoising. Generation will be coarser than canonical
DDPM but the pipeline correctness is testable from the output shape.

Usage:
  lake exe mnist-ddpm-sample [out.ppm]
-/

def tinyDdpmUnet (centred : Bool := true) : NetSpec where
  name := if centred then "tiny DDPM UNet T-cond centered (MNIST 28x28x1)"
                     else "tiny DDPM UNet T-cond (MNIST 28x28x1)"
  imageH := 28
  imageW := 28
  layers := [
    .unetDown 2 16,
    .unetDown 16 32,
    .convBn 32 64 3 1 .same,
    .convBn 64 64 3 1 .same,
    .unetUp 64 32,
    .unetUp 32 16,
    .conv2d 16 1 1 .same .identity
  ]

private def runIree (mlirPath outPath : String) : IO Bool := do
  -- XLA/PJRT has no ahead-of-time compile step: the `.mlir` IS the artifact
  -- the runtime loads. Mirrors `Train.lean`'s `runIreeCached` guard.
  if (← LowererSession.backendName) == "xla" then return true
  let args ← ireeCompileArgs mlirPath outPath
  let compiler ← if (← System.FilePath.pathExists ".venv/bin/iree-compile")
                 then pure ".venv/bin/iree-compile" else pure "iree-compile"
  let r ← IO.Process.output { cmd := compiler, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile failed: {r.stderr.take 3000}"
    return false
  return true

private def floatToU8 (v : Float) : UInt8 :=
  let p := if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v
  (p * 255.0).toUInt8

def main (args : List String) : IO Unit := do
  -- `raw` renders the uncentred ablation arm; see MainMnistDdpmTrain's note.
  let raw := args.any (· == "raw")
  -- `trajectory` renders the two-strip figure instead of the 4x4 grid: the
  -- FORWARD process (a real MNIST digit corrupted to noise) over the REVERSE
  -- process (this sampler's own intermediates, noise to digit). Both strips come
  -- from the same alphaBar table and the same DDIM loop as an ordinary sample
  -- run, so the picture cannot drift from the sampler it illustrates.
  let traj := args.any (· == "trajectory")
  -- `trajectory` extras: where MNIST lives, and which training image the forward
  -- strip corrupts. Index is a knob because most MNIST digits are visually dull
  -- at 28x28 under heavy noise and it is worth being able to pick a legible one.
  let dataDir := (args.filter (fun a => a.startsWith "data=")).head?.map
                   (fun a => (a.drop 5).toString) |>.getD "data"
  let fwdIdx : Nat := ((args.filter (fun a => a.startsWith "img=")).head?.bind
                   (fun a => (a.drop 4).toString.toNat?)).getD 7
  let outPath := (args.filter (fun a =>
                     a != "raw" && a != "trajectory"
                     && !a.startsWith "data=" && !a.startsWith "img=")).head?.getD
                   (if traj then "runs/2026-09-02-mnist-ddpm/trajectory.ppm"
                    else "runs/2026-05-07-mnist-ddpm/samples.ppm")
  IO.FS.createDirAll (System.FilePath.mk outPath).parent.get!.toString
  let spec := tinyDdpmUnet (centred := !raw)
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix

  -- ── Compile the eval forward vmfb (fixedBN=true) if not cached ──
  let evalMlirPath := s!"{pfx}_fwd_eval.mlir"
  let evalVmfb ← NetSpec.graphArtifact pfx "fwd_eval"
  let B : Nat := 16
  let nPix : Nat := spec.imageH * spec.imageW
  if !(← System.FilePath.pathExists evalVmfb) then
    let mlir := MlirCodegen.generateEval spec B
    IO.FS.writeFile evalMlirPath mlir
    IO.eprintln s!"  generated eval mlir ({mlir.length} chars), compiling..."
    unless (← runIree evalMlirPath evalVmfb) do IO.Process.exit 1
    IO.eprintln "  eval forward compiled"

  -- ── Load checkpoint ──
  let paramsPath := s!"{pfx}_params.bin"
  let bnPath := s!"{pfx}_bn_stats.bin"
  for p in [paramsPath, bnPath] do
    if !(← System.FilePath.pathExists p) then
      IO.eprintln s!"missing checkpoint: {p}"
      IO.eprintln "  run lake exe mnist-ddpm-train data first"
      IO.Process.exit 1
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ← IO.FS.readBinFile bnPath
  let evalParams := params.append bnStats
  let evalShapes := spec.evalShapesBA
  let xShape := spec.xShape B

  -- ── DDIM schedule: subsample 50 steps from T = 1000 ──
  let T : Nat := 1000
  let alphaBar ← Ddpm.cosineSchedule T.toUSize
  let nSteps : Nat := 50
  let stride : Nat := T / nSteps
  -- Step indices going DOWN: [T-1, T-1-stride, ..., stride-1]. Pair with
  -- a "previous" of one stride lower; the final step uses ᾱ ≈ 1 (clean).
  let stepTs : Array Nat := Id.run do
    let mut s : Array Nat := #[]
    for k in [:nSteps] do s := s.push (T - 1 - k * stride)
    s

  let alphaBarF : Nat → Float := fun t => F32.read alphaBar t.toUSize

  -- ── Initialize x_T ~ N(0, I) for the 16-image batch ──
  let mut x ← Ddpm.sampleNoise (B * nPix).toUSize 0xc0ffee
  let nTotal : USize := (B * nPix).toUSize

  -- ── Sampling loop ──
  let sess ← LowererSession.create evalVmfb
  IO.eprintln s!"  sampling: {nSteps} DDIM steps, batch {B}"
  -- Every reverse state, with the timestep it sits at. ⚠ Keep ALL of them and
  -- select later by NOISE LEVEL: indexing the strip by sampler step would make
  -- column c mean a different ᾱ in each row, and the two rows are only worth
  -- putting one above the other if they are comparable column by column.
  let nFrames : Nat := 9
  let mut revAll : Array ByteArray := #[]
  let mut revTs  : Array Nat := #[]
  if traj then
    revAll := revAll.push x; revTs := revTs.push (T - 1)   -- x_T, pure noise
  for k in [:nSteps] do
    let t := stepTs[k]!
    let tPrev : Nat := if k + 1 < nSteps then stepTs[k + 1]! else 0
    -- Time conditioning: prepend a constant t/T-channel to each image.
    -- Output is [B, 2, H, W] flat = the 2-channel input the network expects.
    let xCond ← Ddpm.prependTChannelScalar x B.toUSize (1 : USize)
                  spec.imageH.toUSize spec.imageW.toUSize t.toUSize T.toUSize
    -- Forward: ε_θ = model(x_t conditioned on t). nClasses = nPix because
    -- model output is [B, 1, 28, 28] = B * 784 floats per batch.
    let eps ← LowererSession.forwardF32 sess spec.evalFnName
                evalParams evalShapes xCond xShape B.toUSize nPix.toUSize
    -- DDIM coefs
    let aBarT := alphaBarF t
    let aBarP := if k + 1 < nSteps then alphaBarF tPrev else 0.9999
    let sqAT := Float.sqrt aBarT
    let sqAP := Float.sqrt aBarP
    let sqOmAT := Float.sqrt (1.0 - aBarT)
    let sqOmAP := Float.sqrt (1.0 - aBarP)
    let a := sqAP / sqAT
    let b := sqOmAP - a * sqOmAT
    x ← Ddpm.ddimStep x eps a b nTotal
    if traj then
      revAll := revAll.push x; revTs := revTs.push tPrev
    if k % 10 == 0 || k == nSteps - 1 then
      IO.eprintln s!"  step {k}/{nSteps} t={t}->{tPrev}  a={a} b={b}"

  -- ── Trajectory figure: forward strip over reverse strip ──
  if traj then
    let H := spec.imageH; let W := spec.imageW
    -- FORWARD: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε on ONE real MNIST digit.
    -- `ddimStep` is exactly `a·x + b·eps`, so the same primitive the reverse loop
    -- uses builds the forward strip — no second implementation of the schedule.
    let (imgs, nImgs) ← F32.loadIdxImages s!"{dataDir}/train-images-idx3-ubyte"
    IO.eprintln s!"  forward strip: {nImgs} MNIST images available, using #{fwdIdx}"
    let x0raw := F32.sliceImages imgs fwdIdx 1 nPix
    -- The trainer centres to [-1,1]; the strip must live in the same space as the
    -- reverse frames or the two rows are not comparable.
    let x0 ← F32.scaleShift x0raw 2.0 (-1.0)
    let fwdNoise ← Ddpm.sampleNoise nPix.toUSize 0x5eed
    let mut fwdFrames : Array ByteArray := #[]
    for f in [:nFrames] do
      -- t spread over the full schedule, clean (t=0) first, pure noise last.
      let tf : Nat := f * (T - 1) / (nFrames - 1)
      let ab := alphaBarF tf
      fwdFrames := fwdFrames.push
        (← Ddpm.ddimStep x0 fwdNoise (Float.sqrt ab) (Float.sqrt (1.0 - ab)) nPix.toUSize)
    -- Reverse row, selected to match the forward row's noise levels: column c of
    -- both rows is the same ᾱ, so the figure reads as one process and its inverse
    -- rather than two unrelated strips.
    let mut revFrames : Array ByteArray := #[]
    for f in [:nFrames] do
      let tf : Nat := f * (T - 1) / (nFrames - 1)
      let mut bestI : Nat := 0
      let mut bestD : Nat := T
      for i in [:revTs.size] do
        let d := if revTs[i]! > tf then revTs[i]! - tf else tf - revTs[i]!
        if d < bestD then bestD := d; bestI := i
      revFrames := revFrames.push revAll[bestI]!
    -- Undo the centring for display, both rows identically.
    let unc := fun (b : ByteArray) => F32.scaleShift b 0.5 0.5
    let rows := #[fwdFrames, revFrames]
    let gap : Nat := 4
    let stripW := nFrames * W + (nFrames - 1) * gap
    let stripH := 2 * H + gap
    let mut ppm : ByteArray := ByteArray.empty
    ppm := ppm.append s!"P6\n{stripW} {stripH}\n255\n".toUTF8
    for r in [:2] do
      let frames := rows[r]!
      let shown ← frames.mapM unc
      for h in [:H] do
        for c in [:nFrames] do
          let fr := shown[min c (shown.size - 1)]!
          for w in [:W] do
            let u := floatToU8 (F32.read fr (h * W + w).toUSize)
            ppm := ppm.push u |>.push u |>.push u
          if c + 1 < nFrames then
            for _ in [:gap] do ppm := ppm.push 24 |>.push 24 |>.push 28
      if r == 0 then
        for _ in [:gap] do
          for _ in [:stripW] do ppm := ppm.push 24 |>.push 24 |>.push 28
    IO.FS.writeBinFile outPath ppm
    IO.eprintln s!"  wrote {outPath} ({stripW}x{stripH}, {fwdFrames.size} forward + {revFrames.size} reverse frames)"
    return

  -- ── Render 4×4 grid ──
  -- ⚠ Invert the trainer's [-1, 1] centring FIRST. `floatToU8` clamps to
  -- [0, 1], so without this every negative pixel — half the image — renders
  -- black and the grid looks like sparse scribbles whatever the model learned.
  unless raw do x ← F32.scaleShift x 0.5 0.5
  let H := spec.imageH; let W := spec.imageW
  let gridW := 4 * W
  let gridH := 4 * H
  let mut ppm : ByteArray := ByteArray.empty
  ppm := ppm.append s!"P6\n{gridW} {gridH}\n255\n".toUTF8
  for gy in [:4] do
    for h in [:H] do
      for gx in [:4] do
        let idx := gy * 4 + gx
        for w in [:W] do
          let v := F32.read x (idx * nPix + h * W + w).toUSize
          let u := floatToU8 v
          ppm := ppm.push u |>.push u |>.push u
  IO.FS.writeBinFile outPath ppm
  IO.eprintln s!"  wrote {outPath} ({ppm.size} bytes)"
