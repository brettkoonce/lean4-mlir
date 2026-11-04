import LeanMlir
import LeanMlir.ReferenceNets

/-! The LIDC low-dose CT denoiser (`planning/lidc_ct_denoising_demo.md` §5–§6, §10): the MSE arm,
    the Noise2Noise arm and the conditional diffusion arm (the generator), one trainer.

    The regression arms learn the noise a simulated low-dose slice carries:

        input  x = low / 1000                       (HU; water 0, air −1)
        target ε = (low − target) / 100              (HU)
        loss     = ‖net(x) − ε‖²  per-pixel MSE      (the DDPM trainer's ε-loss, `useDdpm`)
        denoised = low − 100 · net(x)  HU

    `--arm=mse` (default): target = the scanned slice, every low copy as an input.
    `--arm=n2n`: Noise2Noise, target = the OTHER low copy (both directions); the scanned slice is
    never seen in training. Needs a 2-realisation build. Same pair count, steps, crops and seeds.
    `--arm=diff`: conditional residual diffusion (§10.1). The variable diffused is the residual
    x₀ = (target − low) / 100 HU (std ≈ 0.7), so the MSE arm is exactly this model's posterior
    mean; the net sees `[x_t, low / 1000]` plus 8 sin/cos channels of t (`Ddpm.prependSinCosT`,
    `nFreq = 4`, T = 1000, cosine ᾱ) and predicts the noise ε of the forward process, the MNIST
    DDPM's recipe at 256² patches. A diffusion run writes no dump: `lidc-denoise-sample` draws
    the samples (DDIM, η = 1, 50 steps) and writes them in this trainer's dump format.

    Trained on random `patch`² crops (input and target cut by one seed, so at one offset);
    the regression arms are scored on whole 512² slices with the eval forward (running BN
    statistics), realisation 0 against the scanned slice: RMSE in HU, low-dose floor beside
    denoised, over the whole slice and over body pixels (scanned HU ≥ −950), and `net(x)` per
    scored record goes to `<dump>/<parent>_<file>.eps.f32` for `scripts/probes/lidc_denoise_score.py`.

    The realisation count N is read off the pair file (`F32.ctRealisations`): the 25 % build has
    two, the 10 % build one (so MSE and diffusion see 9,600 pairs an epoch there, not 19,200).

    Checkpoints `[θ, m, v, BN, step]` to `<dump>/ckpt_*` every `--ckpt-every` steps and resumes
    from them when present (the cosine schedule keys off the global step, so a resume is exact
    up to the data order). The loss curve goes to `<out>/curve.csv`.

    Usage — any `.bin` but the first is scored, a bare number is the step count:
      lidc-denoise-train data/lidc/train.bin data/lidc/val.bin 30000 --arm=mse --cosine \
        --out=runs/2026-10-10-lidc-denoise/mse --dump=data/lidc/out/mse
      lidc-denoise-train data/lidc/train.bin 60000 --arm=diff --cosine \
        --out=runs/2026-11-02-lidc-diffusion/diff25 --dump=data/lidc/out/diff
      lidc-denoise-train smoke/val.bin smoke/val.bin smoke/train.bin 1500 --out=…smoke --dump=…   -- the smoke fit
    `--sigma` (plan §11): the slice's own noise estimate, `<train.bin>.sigma.f32` (one f32 per record,
    HU, from `scripts/datasets/ct_noise_level.py`, computed from the low-dose image alone), fed as
    one more constant channel, log10(σ̂ / 100 HU), after the time channels — the noise-conditioned
    generator; the name carries `sig`.

    Flags: --arm=mse|n2n|diff --seed=N --lr=F --cosine --ckpt-every=N --sigma
           --eval-only --ckpt=dir (score the eval files with a finished run: its checkpoint dir when
           not the dump dir, e.g. the 10 % MSE net on Mayo with --dump=data/mayo/out/mse)
           --patch=256 --batch=8 (the smoke's small shapes; the artifact names carry a non-default patch)
-/

open ReferenceNets (unetCtDenoise)

def S : Nat := 512
def evalBatch : Nat := 1  -- XLA logs a ~25 GB rematerialization warning compiling the 512² eval at 4 and at 1; both run
def nFreq : Nat := 4
def Tmax : Nat := 1000

/-- `(Σ over pixels, Σ over body pixels, body count)` of the squared HU error of `low − 100·eps`
    (or of `low` when `eps` is none) against `full`, for record `k` of `px`-pixel images. -/
def sqErr (inp tgt : ByteArray) (eps : Option ByteArray) (k px : Nat) : Float × Float × Nat := Id.run do
  let base := k * px
  let mut all := 0.0
  let mut body := 0.0
  let mut nb := 0
  for j in [0:px] do
    let low := F32.read inp (base + j).toUSize * 4096.0 - 1024.0
    let full := F32.read tgt (base + j).toUSize * 4096.0 - 1024.0
    let den := match eps with
      | some e => low - 100.0 * F32.read e j.toUSize
      | none => low
    let d := (den - full) * (den - full)
    all := all + d
    if full ≥ -950.0 then
      body := body + d
      nb := nb + 1
  return (all, body, nb)

def flag (args : List String) (key : String) : Option String :=
  (args.filter (·.startsWith s!"--{key}=")).head?.map fun a => (a.drop (key.length + 3)).toString

/-- `x = low/1000` and `ε = (low − target)/100` for one pairing of `path`, the intermediates freed. -/
def loadArm (path : String) (nReal pairing r : Nat) : IO (ByteArray × ByteArray × Nat) := do
  let (inp, tgt, k) ← F32.loadCtPair path S.toUSize nReal.toUSize pairing.toUSize r.toUSize
  let e ← F32.scaleShift (← F32.subtract inp tgt) 40.96 0.0
  let x ← F32.scaleShift inp 4.096 (-1.024)
  return (x, e, k)

def main (args : List String) : IO UInt32 := do
  let bins := args.filter (·.endsWith ".bin")
  let steps := ((args.filter (!·.startsWith "--")).filterMap String.toNat?).head?.getD 1500
  let arm := (flag args "arm").getD "mse"
  let seed := ((flag args "seed").bind String.toNat?).getD 0
  let lr0 : Float := ((flag args "lr").bind fun s => (s.toNat?.map (·.toFloat / 1e6))).getD 0.0005  -- --lr in units of 1e-6
  let cosine := args.contains "--cosine"
  let ckptEvery := ((flag args "ckpt-every").bind String.toNat?).getD 1000
  let evalOnly := args.contains "--eval-only"
  let useSigma := args.contains "--sigma"
  let patch := ((flag args "patch").bind String.toNat?).getD 256
  let batch := ((flag args "batch").bind String.toNat?).getD 8
  let outDir := (flag args "out").getD "runs/lidc-denoise"
  let dumpDir := (flag args "dump").getD s!"data/lidc/out/{arm}"
  let ckptDir := (flag args "ckpt").getD dumpDir
  let some trainPath := bins.head? | IO.eprintln "usage: lidc-denoise-train <train.bin> [eval.bin …] [steps] [--arm=mse|n2n|diff] [--out=dir] [--dump=dir]"; return 2
  unless arm == "mse" || arm == "n2n" || arm == "diff" do IO.eprintln s!"--arm={arm}: mse, n2n or diff"; return 2
  let diff := arm == "diff"
  let nReal ← F32.ctRealisations trainPath S
  if arm == "n2n" && nReal < 2 then IO.eprintln s!"--arm=n2n needs two realisations; {trainPath} has {nReal}"; return 2
  let evalPaths := bins.tail
  IO.FS.createDirAll outDir
  IO.FS.createDirAll dumpDir
  IO.FS.createDirAll ".lake/build"
  -- The arm, the seed and the data root (lidc10 — the 25 % build's `lidc` is left off so the
  -- 2026-10-10 artifacts keep their names) name the run apart, so the 25 % and 10 % generators
  -- never share a train-step artifact, a params file or a checkpoint; a non-default patch too,
  -- so a smoke at 64² never reuses the 256² train step (its shapes are baked in).
  let smoke := outDir.endsWith "smoke"
  let root := ((System.FilePath.mk trainPath).parent.map (·.fileName.getD "")).getD ""
  let tag := (if smoke then (if diff then "diff" else "") else s!"{arm} s{seed}" ++ (if root == "lidc" then "" else s!" {root}"))
    ++ (if useSigma then " sig" else "") ++ (if patch == 256 then "" else s!" p{patch}")
  let baseC := if diff then 2 + 2 * nFreq else 1   -- the channels before the σ̂ one
  let inC := baseC + (if useSigma then 1 else 0)
  let spec := unetCtDenoise patch tag inC
  let evalSpec := unetCtDenoise S tag inC
  let pfx := spec.buildPrefix
  IO.eprintln s!"{spec.name}: {spec.totalParams} params; arm {arm}, seed {seed}, N = {nReal}; {steps} steps of {batch} × {patch}², lr {lr0}{if cosine then " cosine" else ""}"

  let nP := spec.totalParams
  let mut p ← spec.heInitParams (42 + (seed * 100003).toUSize)
  let mut m ← F32.const nP.toUSize 0.0
  let mut v ← F32.const nP.toUSize 0.0
  let mut bn ← F32.const spec.nBnStats.toUSize 0.0
  let mut step := 0
  let ck := fun (s : String) => s!"{ckptDir}/ckpt_{s}.bin"
  if ← System.FilePath.pathExists (ck "step") then
    step := readU32LE (← IO.FS.readBinFile (ck "step")) 0
    p ← IO.FS.readBinFile (ck "params"); m ← IO.FS.readBinFile (ck "m")
    v ← IO.FS.readBinFile (ck "v"); bn ← IO.FS.readBinFile (ck "bn")
    if p.size != nP * 4 then IO.eprintln s!"⛔ {ck "params"} holds {p.size / 4} params, this net has {nP}: another arm's checkpoint"; return 1
    IO.eprintln s!"resumed from {ckptDir}/ckpt_* at step {step}"
  else if evalOnly then
    -- Scoring another file with a finished run: its saved params, never a fresh init.
    if ← System.FilePath.pathExists s!"{pfx}_params.bin" then
      p ← IO.FS.readBinFile s!"{pfx}_params.bin"; bn ← IO.FS.readBinFile s!"{pfx}_bn_stats.bin"
      IO.eprintln s!"--eval-only: loaded {pfx}_params.bin"
    else
      IO.eprintln s!"⛔ --eval-only with no checkpoint in {ckptDir} and no {pfx}_params.bin"; return 1

  if !evalOnly && step < steps then
    -- ── Train step: the DDPM path's per-pixel MSE against an f32 target ──
    let trainMlir := MlirCodegen.generateTrainStep spec batch ("jit_" ++ spec.sanitizedName ++ "_train_step")
      (weightDecay := 0.0) (useAdam := true) (useDdpm := true) (ddpmOutShape := [batch, 1, patch, patch])
    IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
    let vmfb ← NetSpec.graphArtifact pfx "train_step"
    unless (← NetSpec.compileArtifact s!"{pfx}_train_step.mlir" vmfb) do return 1

    -- ── Data: one (input, target) pairing per realisation ──
    let pairings : List (Nat × Nat) :=
      if arm == "n2n" then [(1, 0), (2, 0)] else (List.range nReal).map fun r => (0, r)
    let mut xs : Array ByteArray := #[]
    let mut es : Array ByteArray := #[]
    let mut n := 0
    for (pr, r) in pairings do
      let (x, e, k) ← loadArm trainPath nReal pr r
      xs := xs.push x; es := es.push e; n := n + k
    let mut x := F32.concat xs
    let mut e := F32.concat es
    xs := #[]; es := #[]
    IO.eprintln s!"train: {n} (input, target) pairs from {trainPath}"
    -- The σ̂ sidecar: one float per record, repeated per pairing so it lines up with x and e; a
    -- dummy 4-byte label lets F32.shuffle carry it under the same permutation (same n, same seed).
    let mut sig : ByteArray := ByteArray.empty
    let mut sigDummy : ByteArray := ByteArray.empty
    if useSigma then
      let raw ← IO.FS.readBinFile s!"{trainPath}.sigma.f32"
      let per := n / pairings.length
      if raw.size != per * 4 then IO.eprintln s!"⛔ {trainPath}.sigma.f32 holds {raw.size / 4} values, the file has {per} records"; return 1
      sig := F32.concat ((List.replicate pairings.length raw).toArray)
      sigDummy ← F32.const n.toUSize 0.0
      IO.eprintln s!"sigma: {trainPath}.sigma.f32, median channel value log10(σ̂/100) ≈ {Float.log10 (F32.read raw (per / 2).toUSize / 100.0)}"
    let alphaBar ← Ddpm.cosineSchedule Tmax.toUSize

    let sess ← LowererSession.create vmfb
    let nT := 3 * nP
    let px := S * S
    let pp := patch * patch
    let bpE := n / batch
    let curve ← IO.FS.Handle.mk s!"{outDir}/curve.csv" (if step == 0 then .write else .append)
    if step == 0 then curve.putStrLn "step,loss,rms_hu,lr,ms_per_step"
    let mut acc := 0.0
    let mut accN := 0
    let mut t0 ← IO.monoMsNow
    let mut epoch := step / bpE
    while step < steps do
      let (x', e') ← F32.shuffle x e n.toUSize px.toUSize (px * 4).toUSize (seed * 1000003 + epoch + 1).toUSize
      x := x'; e := e'
      if useSigma then
        let (s', d') ← F32.shuffle sig sigDummy n.toUSize 1 4 (seed * 1000003 + epoch + 1).toUSize
        sig := s'; sigDummy := d'
      epoch := epoch + 1
      for bi in [:bpE] do
        if step ≥ steps then break
        step := step + 1
        let lr := if cosine then lr0 * 0.5 * (1.0 + Float.cos (3.141592653589793 * step.toFloat / steps.toFloat)) else lr0
        let cs := (seed * 1000003 + step * 7919).toUSize
        let xb ← F32.randomCrop (F32.sliceImages x (bi * batch) batch px) batch.toUSize 1 S.toUSize S.toUSize patch.toUSize patch.toUSize cs
        let eb ← F32.randomCrop (F32.sliceImages e (bi * batch) batch px) batch.toUSize 1 S.toUSize S.toUSize patch.toUSize patch.toUSize cs
        -- The diffusion arm: x₀ = −ε_resid = (target − low)/100, its forward process at a drawn
        -- t per image, the net's input [x_t, low, sin/cos t], the DDPM noise as the target.
        let (xin, tgt) ← if diff then do
            let x0 ← F32.scaleShift eb (-1.0) 0.0
            let (xt, rest) ← Ddpm.stepInputs x0 alphaBar batch.toUSize pp.toUSize (seed * 1000003 + step * 104729 + 7).toUSize
            let (eps, tba) := rest
            let xc ← F32.concatChannels xt xb batch.toUSize 1 1 pp.toUSize
            let xcond ← Ddpm.prependSinCosT xc tba batch.toUSize 2 patch.toUSize patch.toUSize nFreq.toUSize Tmax.toUSize
            pure (xcond, eps)
          else pure (xb, eb)
        let xin ← if useSigma then do
            let vals ← F32.log10Scale (F32.sliceImages sig (bi * batch) batch 1) 0.01
            F32.appendScalarChannel xin vals batch.toUSize baseC.toUSize pp.toUSize
          else pure xin
        let out ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName ((p.append m).append v) spec.shapesBA
          xin (spec.xShape batch) tgt lr step.toFloat spec.bnShapesBA batch.toUSize 1 patch.toUSize patch.toUSize
        let loss := F32.extractLoss out nT
        (p, m, v) := F32.unpackAdam out nP
        bn ← F32.ema bn (out.extract ((nT + 1) * 4) ((nT + 1 + spec.nBnStats) * 4)) (if step == 1 then 1.0 else 0.1)
        acc := acc + loss; accN := accN + 1
        if step ≤ 3 || step % 100 == 0 then
          let t ← IO.monoMsNow
          let l := acc / accN.toFloat
          -- Regression arms: loss is in (HU/100)², so √loss·100 is the RMS error of the predicted
          -- noise in HU. Diffusion: loss is the ε-MSE of a unit-normal target (1.0 = predicting 0).
          let unit := if diff then "ε-MSE" else s!"rms {l.sqrt * 100.0} HU"
          IO.eprintln s!"  step {step}/{steps}: loss {l} ({unit}), lr {lr}, {(t - t0) / accN} ms/step"
          curve.putStrLn s!"{step},{l},{if diff then 0.0 else l.sqrt * 100.0},{lr},{(t - t0) / accN}"
          curve.flush
          acc := 0.0; accN := 0; t0 := t
        if step % ckptEvery == 0 || step == steps then
          IO.FS.writeBinFile (ck "params") p; IO.FS.writeBinFile (ck "m") m
          IO.FS.writeBinFile (ck "v") v; IO.FS.writeBinFile (ck "bn") bn
          IO.FS.writeBinFile (ck "step") (pushU32LE ByteArray.empty step)
    IO.FS.writeBinFile s!"{pfx}_params.bin" p
    IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" bn
    IO.eprintln s!"saved {pfx}_params.bin + _bn_stats.bin"

  if diff then
    IO.println s!"diffusion arm trained to step {step}; draw and score samples with lidc-denoise-sample --seed={seed}"
    return 0

  -- ── Score whole 512² slices with the eval forward (regression arms) ──
  let evalMlir := MlirCodegen.generateEval evalSpec evalBatch
  IO.FS.writeFile s!"{pfx}_fwd_eval{S}.mlir" evalMlir
  let evalVmfb ← NetSpec.graphArtifact pfx s!"fwd_eval{S}"
  unless (← NetSpec.compileArtifact s!"{pfx}_fwd_eval{S}.mlir" evalVmfb) do return 1
  let esess ← LowererSession.create evalVmfb
  let evalParams := p.append bn
  let px := S * S
  IO.println s!"rmse HU vs the scanned slice: low-dose floor → denoised ({arm}, seed {seed}, step {step})"
  for path in evalPaths do
    let nR ← F32.ctRealisations path S
    let (inp, tgt, k) ← F32.loadCtPair path S.toUSize nR.toUSize 0 0
    let xin ← F32.scaleShift inp 4.096 (-1.024)
    let sigEval ← if useSigma then IO.FS.readBinFile s!"{path}.sigma.f32" else pure ByteArray.empty
    if useSigma && sigEval.size != k * 4 then IO.eprintln s!"⛔ {path}.sigma.f32 holds {sigEval.size / 4} values, the file has {k} records"; return 1
    let mut sums : Array Float := #[0, 0, 0, 0]
    let mut nb := 0
    let name := (System.FilePath.mk path).fileStem.getD "eval"
    let parent := ((System.FilePath.mk path).parent.map (·.fileName.getD "")).getD ""
    let dump ← IO.FS.Handle.mk s!"{dumpDir}/{parent}_{name}.eps.f32" .write
    for b0 in [0:k:evalBatch] do
      let xb := F32.sliceImagesPad xin b0 evalBatch px k
      let xb ← if useSigma then do
          let vals ← F32.log10Scale (F32.sliceImagesPad sigEval b0 evalBatch 1 k) 0.01
          F32.appendScalarChannel xb vals evalBatch.toUSize 1 px.toUSize
        else pure xb
      let out ← LowererSession.forwardF32 esess evalSpec.evalFnName evalParams evalSpec.evalShapesBA
        xb (evalSpec.xShape evalBatch) evalBatch.toUSize px.toUSize
      for i in [0:min evalBatch (k - b0)] do
        let ei := out.extract (i * px * 4) ((i + 1) * px * 4)
        dump.write ei
        let (fa, fb, c) := sqErr inp tgt none (b0 + i) px
        let (da, db, _) := sqErr inp tgt (some ei) (b0 + i) px
        sums := #[sums[0]! + fa, sums[1]! + fb, sums[2]! + da, sums[3]! + db]
        nb := nb + c
    let all := (k * px).toFloat
    IO.println s!"  {path} ({k} slices): whole {(sums[0]! / all).sqrt} → {(sums[2]! / all).sqrt}, body {(sums[1]! / nb.toFloat).sqrt} → {(sums[3]! / nb.toFloat).sqrt}"
  return 0
