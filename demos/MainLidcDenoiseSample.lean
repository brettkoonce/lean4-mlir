import LeanMlir
import LeanMlir.ReferenceNets

/-! The LIDC generator's sampler (`planning/lidc_ct_denoising_demo.md` §10.1): given a low-dose
    slice, draw one full-dose slice from the conditional diffusion model `lidc-denoise-train
    --arm=diff` fitted.

    The model diffuses the residual x₀ = (full − low)/100 HU conditioned on `low/1000`; the net
    sees `[x_t, low]` plus the 8 sin/cos channels of t (`Ddpm.prependSinCosTScalar`, nFreq 4,
    T = 1000, cosine ᾱ) and predicts ε. Sampling is generalized DDIM (`Ddpm.ddimCoefs`): 50 steps
    over the 1000-step table, η = 1 by default — ancestral, every step adds fresh noise, so the
    grain in a sample is the model's, not a deterministic sampler's smoothing. The final x is the
    residual sample x̂₀; the generated slice is low + 100·x̂₀ HU. The dump stores −x̂₀, which is
    the trainer's convention (denoised = low − 100·eps), so `lidc_denoise_score.py`,
    `lidc_nodules.py` and `lidc_texture.py` read a sample exactly as they read the MSE net.

    Records: all of the file, or `--records=<file>` (integers, `#` comments; the §10.5 scoring set
    from `scripts/demos/lidc_select_records.py`), split `--shard=i/n` across processes on different
    GPUs. The dump `<dump>/<parent>_<file>.k<K>.eps.f32` has the trainer's layout (record k at
    byte k·S²·4, `F32.truncateFile` + `F32.pwriteFile`, the unwritten records a hole) and a sidecar
    `.records` that lists the records written, one per line, appended as each lands — so a
    killed run resumes past them, shards share one file, and a scorer knows which records to
    read. `--k` is the sample seed: k = 0 … 3 are the four draws of the spread study; the initial
    noise and every step's noise are seeded by (k, record, step), independent of the batch and
    shard a record ends up in.

    Prints, per record and pooled, the body RMSE of the sample against the scanned slice beside
    the low-dose floor (a sample must cost RMSE against the posterior mean — §10's expected
    result; the texture and findings scorers are the arm's measurements).

    Usage:
      lidc-denoise-sample data/lidc/val.bin --seed=0 --k=0 --records=runs/…/records_val.txt \
        --shard=0/4 --dump=data/lidc/out/diff
    Flags: --seed=N (the training run's seed → its checkpoint) --k=N (sample seed) --steps=50
           --eta=100 (percent) --batch=1 --records=file --shard=i/n --limit=N --dump=dir
           --model=<root> (the data root the checkpoint was trained on: lidc, lidc10; default = the
           pairs file's own) --smoke (the smoke fit's unnamed checkpoint) --patch=256 (the fit's patch)
           --crop=C (centre-crop the slices to C² — the plumbing smoke on a small GPU slice)
           --sigma (plan §11: the noise-conditioned model; reads `<pairs.bin>.sigma.f32`, the slice's
           σ̂ from the low-dose image, as the trainer did) --sigma-cap=HU (tell the net at most this
           σ̂: plan §13.4's probe, the training median 101) --clamp-x0 (the usual clipped-x₀
           prediction, plan §13.4: x̂₀ = (x − √(1−ᾱ) ε̂)/√ᾱ kept inside the HU range as low + 100·x̂₀
           and ε̂ recomputed from it, so no step aims past the range) --target=x0 (plan §13.5: the
           checkpoint trained to predict x₀ rather than ε; the net's output is read as x̂₀ and the
           step's ε̂ is (x − √ᾱ x̂₀)/√(1−ᾱ); composes with --clamp-x0) --clamp (after every step, keep
           low + 100·x inside [−1024, 3071] HU — the valid range; the count of clamped pixels is
           reported, so the row says whether the guard ever bit)
-/

open ReferenceNets (unetCtDenoise)

def S : Nat := 512
def nFreq : Nat := 4
def Tmax : Nat := 1000

def flag (args : List String) (key : String) : Option String :=
  (args.filter (·.startsWith s!"--{key}=")).head?.map fun a => (a.drop (key.length + 3)).toString

/-- `(Σ over body pixels of (den − full)², body count)` for one `px`-pixel image at `base`, with
    den = low − 100·eps (eps none: the low dose itself). -/
def bodySq (inp tgt : ByteArray) (eps : Option (ByteArray × Nat)) (base px : Nat) : Float × Nat := Id.run do
  let mut body := 0.0
  let mut nb := 0
  for j in [0:px] do
    let low := F32.read inp (base + j).toUSize * 4096.0 - 1024.0
    let full := F32.read tgt (base + j).toUSize * 4096.0 - 1024.0
    let den := match eps with
      | some (e, eb) => low - 100.0 * F32.read e (eb + j).toUSize
      | none => low
    if full ≥ -950.0 then
      body := body + (den - full) * (den - full)
      nb := nb + 1
  return (body, nb)

/-- The integers of a record-list file; `#` to end of line is a comment. -/
def readRecordList (path : String) : IO (Array Nat) := do
  let txt ← IO.FS.readFile path
  let mut out : Array Nat := #[]
  for line in txt.splitOn "\n" do
    let body := (line.splitOn "#").head?.getD ""
    for tok in body.splitOn " " do
      for t in tok.splitOn "\t" do
        if let some n := t.trimAscii.toNat? then out := out.push n
  return out

def main (args : List String) : IO UInt32 := do
  let bins := args.filter (·.endsWith ".bin")
  let some path := bins.head? | IO.eprintln "usage: lidc-denoise-sample <pairs.bin> [--seed=N] [--k=N] [--records=file] [--shard=i/n] [--dump=dir]"; return 2
  let seed := ((flag args "seed").bind String.toNat?).getD 0
  let kSeed := ((flag args "k").bind String.toNat?).getD 0
  let nSteps := ((flag args "steps").bind String.toNat?).getD 50
  let eta : Float := (((flag args "eta").bind String.toNat?).getD 100).toFloat / 100.0
  let B := ((flag args "batch").bind String.toNat?).getD 1
  let limit := ((flag args "limit").bind String.toNat?).getD 0
  let crop := ((flag args "crop").bind String.toNat?).getD 0
  let patch := ((flag args "patch").bind String.toNat?).getD 256
  let smoke := args.contains "--smoke"
  let useSigma := args.contains "--sigma"
  let clamp := args.contains "--clamp"
  let clampX0 := args.contains "--clamp-x0"
  let predX0 := (flag args "target").getD "eps" == "x0"
  let sigCap : Float := (((flag args "sigma-cap").bind String.toNat?).getD 0).toFloat
  let dumpDir := (flag args "dump").getD "data/lidc/out/diff"
  let parent := ((System.FilePath.mk path).parent.map (·.fileName.getD "")).getD ""
  -- The data root the checkpoint was trained on names its artifacts (the trainer's tag); the
  -- default is the pairs file's own root, Mayo passes --model=lidc10.
  let model := (flag args "model").getD parent
  let (shardI, shardN) := match (flag args "shard").map (·.splitOn "/") with
    | some [i, n] => ((i.toNat?.getD 0), (n.toNat?.getD 1))
    | _ => (0, 1)
  let Sx := if crop == 0 then S else crop
  let px := Sx * Sx
  let tag := (if smoke then "diff" else s!"diff s{seed}" ++ (if model == "lidc" then "" else s!" {model}"))
    ++ (if useSigma then " sig" else "") ++ (if predX0 then " x0" else "")
    ++ (if patch == 256 then "" else s!" p{patch}")
  let baseC := 2 + 2 * nFreq
  let inC := baseC + (if useSigma then 1 else 0)
  let spec := unetCtDenoise Sx tag inC
  let pfx := spec.buildPrefix
  IO.FS.createDirAll dumpDir
  IO.FS.createDirAll ".lake/build"

  -- ── The checkpoint ──
  let paramsPath := s!"{pfx}_params.bin"
  let bnPath := s!"{pfx}_bn_stats.bin"
  for f in [paramsPath, bnPath] do
    unless ← System.FilePath.pathExists f do
      IO.eprintln s!"missing {f}: train the arm first (lidc-denoise-train --arm=diff --seed={seed})"
      return 1
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ← IO.FS.readBinFile bnPath
  if params.size != spec.totalParams * 4 then
    IO.eprintln s!"⛔ {paramsPath} holds {params.size / 4} params, this net has {spec.totalParams}"; return 1
  let evalParams := params.append bnStats

  -- ── The eval forward at the sampling size ──
  let evalMlir := MlirCodegen.generateEval spec B
  let evalMlirPath := s!"{pfx}_fwd_eval{Sx}_b{B}.mlir"
  IO.FS.writeFile evalMlirPath evalMlir
  let evalVmfb ← NetSpec.graphArtifact pfx s!"fwd_eval{Sx}_b{B}"
  unless (← NetSpec.compileArtifact evalMlirPath evalVmfb) do return 1
  let sess ← LowererSession.create evalVmfb

  -- ── The slices ──
  let nReal ← F32.ctRealisations path S
  let (inp0, tgt0, count) ← F32.loadCtPair path S.toUSize nReal.toUSize 0 0
  let (inp, tgt) ← if crop == 0 then pure (inp0, tgt0) else do
    let a ← F32.centerCrop inp0 count.toUSize 1 S.toUSize S.toUSize crop.toUSize crop.toUSize
    let b ← F32.centerCrop tgt0 count.toUSize 1 S.toUSize S.toUSize crop.toUSize crop.toUSize
    pure (a, b)
  let low ← F32.scaleShift inp 4.096 (-1.024)   -- HU/1000, the conditioning
  let sig ← if useSigma then IO.FS.readBinFile s!"{path}.sigma.f32" else pure ByteArray.empty
  if useSigma && sig.size != count * 4 then
    IO.eprintln s!"⛔ {path}.sigma.f32 holds {sig.size / 4} values, the file has {count} records"; return 1
  let requested ← match flag args "records" with
    | some f => readRecordList f
    | none => pure (Array.range count)
  let name := (System.FilePath.mk path).fileStem.getD "eval"
  let dumpPath := s!"{dumpDir}/{parent}_{name}.k{kSeed}{if crop == 0 then "" else s!".crop{crop}"}.eps.f32"
  let sidecar := dumpPath ++ ".records"
  let done ← if ← System.FilePath.pathExists sidecar then readRecordList sidecar else pure #[]
  let mut recs : Array Nat := #[]
  let mut pos := 0
  for r in requested do
    if r < count then
      if pos % shardN == shardI && !done.contains r then recs := recs.push r
      pos := pos + 1
  if limit > 0 then recs := recs.extract 0 limit
  F32.truncateFile dumpPath (count * px * 4).toUSize
  IO.eprintln s!"{spec.name}: {spec.totalParams} params; {path}: {count} records (N = {nReal}), {requested.size} requested, {done.size} already in {sidecar}, {recs.size} to draw in shard {shardI}/{shardN}; {nSteps} DDIM steps, η = {eta}, k = {kSeed}, batch {B}{if crop == 0 then "" else s!", centre crop {crop}²"}{if sigCap > 0.0 then s!", σ̂ ≤ {sigCap} HU" else ""}{if clampX0 then ", x̂₀ clamped" else ""}{if predX0 then ", x₀-target" else ""}"
  if recs.isEmpty then return 0

  -- ── DDIM over the 1000-step cosine table ──
  let alphaBar ← Ddpm.cosineSchedule Tmax.toUSize
  let stride := Tmax / nSteps
  let abAt := fun (t : Nat) => F32.read alphaBar t.toUSize
  let side ← IO.FS.Handle.mk sidecar .append
  let mut sumFloor := 0.0
  let mut sumSample := 0.0
  let mut nBody := 0
  let mut nDone := 0
  let mut clamped := 0
  let mut clampedX0 := 0
  let t0 ← IO.monoMsNow
  for bi in [:(recs.size + B - 1) / B] do
    let b0 := bi * B
    let batchRecs := recs.extract b0 (min (b0 + B) recs.size)
    let nb := batchRecs.size
    -- The conditioning block [B, 1, px] (zero rows past the end of the list).
    let mut lows : Array ByteArray := #[]
    let mut xs : Array ByteArray := #[]
    for i in [0:B] do
      let r := batchRecs[i]?.getD batchRecs[0]!
      lows := lows.push (F32.sliceImages low r 1 px)
      xs := xs.push (← Ddpm.sampleNoise px.toUSize (kSeed * 7919 + r * 1000003 + 12345).toUSize)
    let lowB := F32.concat lows
    let mut x := F32.concat xs
    let n := (B * px).toUSize
    let sigVals ← if useSigma then do
        let mut vs : Array ByteArray := #[]
        for i in [0:B] do
          let r := batchRecs[i]?.getD batchRecs[0]!
          vs := vs.push (F32.slice sig r 1)
        let raw := F32.concat vs
        -- --sigma-cap: min(σ̂, cap), the residual clamp with a zero base and unit scale.
        let raw ← if sigCap > 0.0 then do
            let zeros ← F32.const B.toUSize 0.0
            let (c, _) ← F32.clampResidual raw zeros B.toUSize 0.0 1.0 (-1.0e30) sigCap
            pure c
          else pure raw
        F32.log10Scale raw 0.01
      else pure ByteArray.empty
    for k in [:nSteps] do
      let tCur := Tmax - 1 - k * stride
      let tPrev := if k + 1 == nSteps then 0 else Tmax - 1 - (k + 1) * stride
      let xc ← F32.concatChannels x lowB B.toUSize 1 1 px.toUSize
      let cond ← Ddpm.prependSinCosTScalar xc B.toUSize 2 Sx.toUSize Sx.toUSize tCur.toUSize nFreq.toUSize Tmax.toUSize
      let cond ← if useSigma then F32.appendScalarChannel cond sigVals B.toUSize baseC.toUSize px.toUSize else pure cond
      let mut eps ← LowererSession.forwardF32 sess spec.evalFnName evalParams spec.evalShapesBA
        cond (spec.xShape B) B.toUSize px.toUSize
      let abT := abAt tCur
      if predX0 || clampX0 then
        -- x̂₀: the net's output under --target=x0, else (x − √(1−ᾱ) ε̂)/√ᾱ from its ε̂; under
        -- --clamp-x0 clamped so low + 100·x̂₀ stays in [−1024, 3071] HU. The ε̂ that would have
        -- predicted that x̂₀, (x − √ᾱ x̂₀)/√(1−ᾱ), then drives the step.
        let sT := Float.sqrt abT
        let cT := Float.sqrt (1.0 - abT)
        let mut x0 ← if predX0 then pure eps else Ddpm.ddimStep x eps (1.0 / sT) (-(cT / sT)) n
        if clampX0 then
          let (x0c, moved) ← F32.clampResidual x0 lowB n 1000.0 100.0 (-1024.0) 3071.0
          clampedX0 := clampedX0 + moved
          x0 := x0c
        eps ← Ddpm.ddimStep x x0 (1.0 / cT) (-(sT / cT)) n
      -- The image samplers' convention: the final step aims at ᾱ = 0.9999, not the table's t = 0.
      let abP := if tPrev == 0 then 0.9999 else abAt tPrev
      let (a, b, sigma) := Ddpm.ddimCoefs abT abP eta
      x ← Ddpm.ddimStep x eps a b n
      if sigma > 0.0 then
        let mut zs : Array ByteArray := #[]
        for i in [0:B] do
          let r := batchRecs[i]?.getD batchRecs[0]!
          zs := zs.push (← Ddpm.sampleNoise px.toUSize (kSeed * 7919 + r * 1000003 + k * 8191 + 17).toUSize)
        x ← Ddpm.ddimStep x (F32.concat zs) 1.0 sigma n
      if clamp then
        let (x', moved) ← F32.clampResidual x lowB n 1000.0 100.0 (-1024.0) 3071.0
        x := x'; clamped := clamped + moved
    -- x̂₀ in HU/100 of (full − low); the dump holds −x̂₀ (denoised = low − 100·eps).
    let negx ← F32.scaleShift x (-1.0) 0.0
    for i in [0:nb] do
      let r := batchRecs[i]!
      let ei := negx.extract (i * px * 4) ((i + 1) * px * 4)
      F32.pwriteFile dumpPath (r * px * 4).toUSize ei
      side.putStrLn s!"{r}"
      side.flush
      let (fb, c) := bodySq inp tgt none (r * px) px
      let (sb, _) := bodySq inp tgt (some (ei, 0)) (r * px) px
      sumFloor := sumFloor + fb; sumSample := sumSample + sb; nBody := nBody + c
      nDone := nDone + 1
      let t ← IO.monoMsNow
      if nDone ≤ 3 || nDone % 20 == 0 || nDone == recs.size then
        IO.eprintln s!"  [{nDone}/{recs.size}] record {r}: body RMSE floor {(fb / c.toFloat).sqrt} → sample {(sb / c.toFloat).sqrt} HU; pooled {(sumFloor / nBody.toFloat).sqrt} → {(sumSample / nBody.toFloat).sqrt}; {(t - t0) / nDone} ms/slice"
  IO.println s!"{dumpPath}: {nDone} samples (k = {kSeed}, {nSteps} steps, η = {eta}{if predX0 then ", x₀-target" else ""}{if useSigma then ", σ̂-conditioned" else ""}{if sigCap > 0.0 then s!", σ̂ ≤ {sigCap} HU" else ""}); body RMSE floor {(sumFloor / nBody.toFloat).sqrt} → sample {(sumSample / nBody.toFloat).sqrt} HU over {nDone} slices{if clamp then s!"; clamp moved {clamped} of {nDone * px * nSteps} pixel-steps" else ""}{if clampX0 then s!"; x̂₀ clamp moved {clampedX0} of {nDone * px * nSteps} pixel-steps" else ""}"
  return 0
