import LeanMlir
import LeanMlir.ReferenceNets

/-! The LIDC low-dose CT denoiser (`planning/lidc_ct_denoising_demo.md` §5–§6): the MSE arm and
    the Noise2Noise arm, one trainer.

    `unetCtDenoise` learns the noise a simulated quarter-dose slice carries:

        input  x = low / 1000                       (HU; water 0, air −1)
        target ε = (low − target) / 100              (HU)
        loss     = ‖net(x) − ε‖²  per-pixel MSE      (the DDPM trainer's ε-loss, `useDdpm`)
        denoised = low − 100 · net(x)  HU

    `--arm=mse` (default): target = the scanned slice, both low copies as inputs.
    `--arm=n2n`: Noise2Noise, target = the OTHER low copy (both directions); the scanned slice is
    never seen in training. Same pair count, steps, crops and seeds as the MSE arm.

    Trained on random `patch`² crops (input and target cut by one seed, so at one offset);
    scored on whole 512² slices with the eval forward (running BN statistics), realisation 0
    against the scanned slice. Prints RMSE in HU, low-dose floor beside denoised, over the whole
    slice and over body pixels (scanned HU ≥ −950), and writes `net(x)` per scored record to
    `<dump>/<parent>_<file>.eps.f32` for `scripts/probes/lidc_denoise_score.py`.

    Checkpoints `[θ, m, v, BN, step]` to `<dump>/ckpt_*` every `--ckpt-every` steps and resumes
    from them when present (the cosine schedule keys off the global step, so a resume is exact
    up to the data order). The loss curve goes to `<out>/curve.csv`.

    Usage — any `.bin` but the first is scored, a bare number is the step count:
      lidc-denoise-train data/lidc/train.bin data/lidc/val.bin 30000 --arm=mse --cosine \
        --out=runs/2026-10-10-lidc-denoise/mse --dump=data/lidc/out/mse
      lidc-denoise-train smoke/val.bin smoke/val.bin smoke/train.bin 1500 --out=… --dump=…   -- the smoke fit
    Flags: --arm=mse|n2n --seed=N --lr=F --cosine --ckpt-every=N --eval-only
-/

open ReferenceNets (unetCtDenoise)

def S : Nat := 512
def patch : Nat := 256
def batch : Nat := 8
def evalBatch : Nat := 1  -- XLA logs a ~25 GB rematerialization warning compiling the 512² eval at 4 and at 1; both run

/-- `(Σ over pixels, Σ over body pixels, body count)` of the squared HU error of `low − 100·eps`
    (or of `low` when `eps` is none) against `full`, for record `k`. -/
def sqErr (inp tgt : ByteArray) (eps : Option ByteArray) (k : Nat) : Float × Float × Nat := Id.run do
  let base := k * S * S
  let mut all := 0.0
  let mut body := 0.0
  let mut nb := 0
  for j in [0:S*S] do
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
def loadArm (path : String) (pairing r : Nat) : IO (ByteArray × ByteArray × Nat) := do
  let (inp, tgt, k) ← F32.loadCtPair path S.toUSize 2 pairing.toUSize r.toUSize
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
  let outDir := (flag args "out").getD "runs/lidc-denoise"
  let dumpDir := (flag args "dump").getD s!"data/lidc/out/{arm}"
  let some trainPath := bins.head? | IO.eprintln "usage: lidc-denoise-train <train.bin> [eval.bin …] [steps] [--arm=mse|n2n] [--out=dir] [--dump=dir]"; return 2
  unless arm == "mse" || arm == "n2n" do IO.eprintln s!"--arm={arm}: mse or n2n"; return 2
  let evalPaths := bins.tail
  IO.FS.createDirAll outDir
  IO.FS.createDirAll dumpDir
  IO.FS.createDirAll ".lake/build"
  let tag := if outDir.endsWith "smoke" then "" else s!"{arm} s{seed}"
  let spec := unetCtDenoise patch tag
  let evalSpec := unetCtDenoise S tag
  let pfx := spec.buildPrefix
  IO.eprintln s!"{spec.name}: {spec.totalParams} params; arm {arm}, seed {seed}; {steps} steps of {batch} × {patch}², lr {lr0}{if cosine then " cosine" else ""}"

  let nP := spec.totalParams
  let mut p ← spec.heInitParams (42 + (seed * 100003).toUSize)
  let mut m ← F32.const nP.toUSize 0.0
  let mut v ← F32.const nP.toUSize 0.0
  let mut bn ← F32.const spec.nBnStats.toUSize 0.0
  let mut step := 0
  let ck := fun (s : String) => s!"{dumpDir}/ckpt_{s}.bin"
  if ← System.FilePath.pathExists (ck "step") then
    step := readU32LE (← IO.FS.readBinFile (ck "step")) 0
    p ← IO.FS.readBinFile (ck "params"); m ← IO.FS.readBinFile (ck "m")
    v ← IO.FS.readBinFile (ck "v"); bn ← IO.FS.readBinFile (ck "bn")
    IO.eprintln s!"resumed from {dumpDir}/ckpt_* at step {step}"

  if !evalOnly && step < steps then
    -- ── Train step: the DDPM path's per-pixel MSE against an f32 target ──
    let trainMlir := MlirCodegen.generateTrainStep spec batch ("jit_" ++ spec.sanitizedName ++ "_train_step")
      (weightDecay := 0.0) (useAdam := true) (useDdpm := true) (ddpmOutShape := [batch, 1, patch, patch])
    IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
    let vmfb ← NetSpec.graphArtifact pfx "train_step"
    unless (← NetSpec.compileArtifact s!"{pfx}_train_step.mlir" vmfb) do return 1

    -- ── Data: two (input, target) pairings per record ──
    let pairings : List (Nat × Nat) := if arm == "mse" then [(0, 0), (0, 1)] else [(1, 0), (2, 0)]
    let mut xs : Array ByteArray := #[]
    let mut es : Array ByteArray := #[]
    let mut n := 0
    for (pr, r) in pairings do
      let (x, e, k) ← loadArm trainPath pr r
      xs := xs.push x; es := es.push e; n := n + k
    let mut x := F32.concat xs
    let mut e := F32.concat es
    xs := #[]; es := #[]
    IO.eprintln s!"train: {n} (input, target) pairs from {trainPath}"

    let sess ← LowererSession.create vmfb
    let nT := 3 * nP
    let px := S * S
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
      epoch := epoch + 1
      for bi in [:bpE] do
        if step ≥ steps then break
        step := step + 1
        let lr := if cosine then lr0 * 0.5 * (1.0 + Float.cos (3.141592653589793 * step.toFloat / steps.toFloat)) else lr0
        let cs := (seed * 1000003 + step * 7919).toUSize
        let xb ← F32.randomCrop (F32.sliceImages x (bi * batch) batch px) batch.toUSize 1 S.toUSize S.toUSize patch.toUSize patch.toUSize cs
        let eb ← F32.randomCrop (F32.sliceImages e (bi * batch) batch px) batch.toUSize 1 S.toUSize S.toUSize patch.toUSize patch.toUSize cs
        let out ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName ((p.append m).append v) spec.shapesBA
          xb (spec.xShape batch) eb lr step.toFloat spec.bnShapesBA batch.toUSize 1 patch.toUSize patch.toUSize
        let loss := F32.extractLoss out nT
        (p, m, v) := F32.unpackAdam out nP
        bn ← F32.ema bn (out.extract ((nT + 1) * 4) ((nT + 1 + spec.nBnStats) * 4)) (if step == 1 then 1.0 else 0.1)
        acc := acc + loss; accN := accN + 1
        if step ≤ 3 || step % 100 == 0 then
          let t ← IO.monoMsNow
          let l := acc / accN.toFloat
          -- loss is in (HU/100)²: √loss·100 is the RMS error of the predicted noise in HU.
          IO.eprintln s!"  step {step}/{steps}: loss {l} (rms {l.sqrt * 100.0} HU), lr {lr}, {(t - t0) / accN} ms/step"
          curve.putStrLn s!"{step},{l},{l.sqrt * 100.0},{lr},{(t - t0) / accN}"
          curve.flush
          acc := 0.0; accN := 0; t0 := t
        if step % ckptEvery == 0 || step == steps then
          IO.FS.writeBinFile (ck "params") p; IO.FS.writeBinFile (ck "m") m
          IO.FS.writeBinFile (ck "v") v; IO.FS.writeBinFile (ck "bn") bn
          IO.FS.writeBinFile (ck "step") (pushU32LE ByteArray.empty step)
    IO.FS.writeBinFile s!"{pfx}_params.bin" p
    IO.FS.writeBinFile s!"{pfx}_bn_stats.bin" bn
    IO.eprintln s!"saved {pfx}_params.bin + _bn_stats.bin"

  -- ── Score whole 512² slices with the eval forward ──
  let evalMlir := MlirCodegen.generateEval evalSpec evalBatch
  IO.FS.writeFile s!"{pfx}_fwd_eval{S}.mlir" evalMlir
  let evalVmfb ← NetSpec.graphArtifact pfx s!"fwd_eval{S}"
  unless (← NetSpec.compileArtifact s!"{pfx}_fwd_eval{S}.mlir" evalVmfb) do return 1
  let esess ← LowererSession.create evalVmfb
  let evalParams := p.append bn
  let px := S * S
  IO.println s!"rmse HU vs the scanned slice: low-dose floor → denoised ({arm}, seed {seed}, step {step})"
  for path in evalPaths do
    let (inp, tgt, k) ← F32.loadCtPair path S.toUSize 2 0 0
    let xin ← F32.scaleShift inp 4.096 (-1.024)
    let mut sums : Array Float := #[0, 0, 0, 0]
    let mut nb := 0
    let name := (System.FilePath.mk path).fileStem.getD "eval"
    let parent := ((System.FilePath.mk path).parent.map (·.fileName.getD "")).getD ""
    let dump ← IO.FS.Handle.mk s!"{dumpDir}/{parent}_{name}.eps.f32" .write
    for b0 in [0:k:evalBatch] do
      let xb := F32.sliceImagesPad xin b0 evalBatch px k
      let out ← LowererSession.forwardF32 esess evalSpec.evalFnName evalParams evalSpec.evalShapesBA
        xb (evalSpec.xShape evalBatch) evalBatch.toUSize px.toUSize
      for i in [0:min evalBatch (k - b0)] do
        let ei := out.extract (i * px * 4) ((i + 1) * px * 4)
        dump.write ei
        let (fa, fb, c) := sqErr inp tgt none (b0 + i)
        let (da, db, _) := sqErr inp tgt (some ei) (b0 + i)
        sums := #[sums[0]! + fa, sums[1]! + fb, sums[2]! + da, sums[3]! + db]
        nb := nb + c
    let all := (k * px).toFloat
    IO.println s!"  {path} ({k} slices): whole {(sums[0]! / all).sqrt} → {(sums[2]! / all).sqrt}, body {(sums[1]! / nb.toFloat).sqrt} → {(sums[3]! / nb.toFloat).sqrt}"
  return 0
