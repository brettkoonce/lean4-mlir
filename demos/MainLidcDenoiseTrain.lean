import LeanMlir
import LeanMlir.ReferenceNets

/-! The LIDC low-dose CT denoiser, MSE arm (`planning/lidc_ct_denoising_demo.md` §5–§6).

    `unetCtDenoise` learns the noise a simulated quarter-dose slice carries:

        input  x = low / 1000                       (HU; water 0, air −1)
        target ε = (low − full) / 100                (HU)
        loss     = ‖net(x) − ε‖²  per-pixel MSE      (the DDPM trainer's ε-loss, `useDdpm`)
        denoised = low − 100 · net(x)  HU

    Trained on random `patch`² crops (input and target cut by one seed, so at one offset), both
    realisations of every record; scored on whole 512² slices with the eval forward (running BN
    statistics). Prints RMSE against the scanned slice in HU, low-dose floor beside denoised, over
    the whole slice and over body pixels (scanned HU ≥ −950), and writes `net(x)` for each scored
    record to `<out>/<name>.eps.f32` for the Python scorer and figure.

    Usage — any `.bin` but the first is scored, a bare number is the step count:
      lake exe lidc-denoise-train data/lidc/train.bin data/lidc/val.bin 20000
      lake exe lidc-denoise-train smoke/val.bin smoke/val.bin smoke/train.bin 1500   -- overfit one volume
-/

open ReferenceNets (unetCtDenoise)

def S : Nat := 512
def patch : Nat := 256
def batch : Nat := 8
def evalBatch : Nat := 1  -- XLA asks ~25 GB compiling the 512² eval at batch 4; 1 fits
def lr : Float := 0.0005

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

def main (args : List String) : IO UInt32 := do
  let bins := args.filter (·.endsWith ".bin")
  let steps := ((args.filterMap String.toNat?).head?).getD 1500
  let outDir := ((args.filter (·.startsWith "--out=")).head?.map (fun a => (a.drop 6).toString)).getD "runs/2026-10-10-lidc-denoise-smoke"
  let some trainPath := bins.head? | IO.eprintln "usage: lidc-denoise-train <train.bin> [eval.bin …] [steps] [--out=dir]"; return 2
  let evalPaths := bins.tail
  IO.FS.createDirAll outDir
  IO.FS.createDirAll ".lake/build"
  let spec := unetCtDenoise patch
  let evalSpec := unetCtDenoise S
  let pfx := spec.buildPrefix
  IO.eprintln s!"{spec.name}: {spec.totalParams} params; {steps} steps of {batch} × {patch}², lr {lr}"

  -- ── Train step: the DDPM path's per-pixel MSE against an f32 target ──
  let trainMlir := MlirCodegen.generateTrainStep spec batch ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := 0.0) (useAdam := true) (useDdpm := true) (ddpmOutShape := [batch, 1, patch, patch])
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  let vmfb ← NetSpec.graphArtifact pfx "train_step"
  unless (← NetSpec.compileArtifact s!"{pfx}_train_step.mlir" vmfb) do return 1

  -- ── Data: both realisations, x = HU/1000, ε = (low − full)/100 ──
  let mut xs : Array ByteArray := #[]
  let mut es : Array ByteArray := #[]
  let mut n := 0
  for r in [0, 1] do
    let (inp, tgt, k) ← F32.loadCtPair trainPath S.toUSize 2 0 r.toUSize
    xs := xs.push (← F32.scaleShift inp 4.096 (-1.024))
    es := es.push (← F32.scaleShift (← F32.subtract inp tgt) 40.96 0.0)
    n := n + k
  let mut x := F32.concat xs
  let mut e := F32.concat es
  IO.eprintln s!"train: {n} (slice, realisation) pairs from {trainPath}"

  let nP := spec.totalParams
  let mut p ← spec.heInitParams
  let mut m ← F32.const nP.toUSize 0.0
  let mut v ← F32.const nP.toUSize 0.0
  let mut bn ← F32.const spec.nBnStats.toUSize 0.0
  let sess ← LowererSession.create vmfb
  let nT := 3 * nP
  let px := S * S
  let bpE := n / batch
  let mut step := 0
  let mut acc := 0.0
  let mut accN := 0
  let t0 ← IO.monoMsNow
  while step < steps do
    let (x', e') ← F32.shuffle x e n.toUSize px.toUSize (px * 4).toUSize (step + 1).toUSize
    x := x'; e := e'
    for bi in [:bpE] do
      if step ≥ steps then break
      step := step + 1
      let seed := (step * 7919).toUSize
      let xb ← F32.randomCrop (F32.sliceImages x (bi * batch) batch px) batch.toUSize 1 S.toUSize S.toUSize patch.toUSize patch.toUSize seed
      let eb ← F32.randomCrop (F32.sliceImages e (bi * batch) batch px) batch.toUSize 1 S.toUSize S.toUSize patch.toUSize patch.toUSize seed
      let out ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName ((p.append m).append v) spec.shapesBA
        xb (spec.xShape batch) eb lr step.toFloat spec.bnShapesBA batch.toUSize 1 patch.toUSize patch.toUSize
      let loss := F32.extractLoss out nT
      (p, m, v) := F32.unpackAdam out nP
      bn ← F32.ema bn (out.extract ((nT + 1) * 4) ((nT + 1 + spec.nBnStats) * 4)) (if step == 1 then 1.0 else 0.1)
      acc := acc + loss; accN := accN + 1
      if step ≤ 3 || step % 100 == 0 then
        let t ← IO.monoMsNow
        -- loss is in (HU/100)²: √loss·100 is the RMS error of the predicted noise in HU.
        IO.eprintln s!"  step {step}/{steps}: loss {acc / accN.toFloat} (rms {(acc / accN.toFloat).sqrt * 100.0} HU), {(t - t0) / step} ms/step"
        acc := 0.0; accN := 0
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
  IO.println s!"rmse HU vs the scanned slice: low-dose floor → denoised"
  for path in evalPaths do
    let (inp, tgt, k) ← F32.loadCtPair path S.toUSize 2 0 0
    let xin ← F32.scaleShift inp 4.096 (-1.024)
    let mut sums : Array Float := #[0, 0, 0, 0]
    let mut nb := 0
    let mut dump := ByteArray.empty
    for b0 in [0:k:evalBatch] do
      let xb := F32.sliceImagesPad xin b0 evalBatch px k
      let out ← LowererSession.forwardF32 esess evalSpec.evalFnName evalParams evalSpec.evalShapesBA
        xb (evalSpec.xShape evalBatch) evalBatch.toUSize px.toUSize
      for i in [0:min evalBatch (k - b0)] do
        let ei := out.extract (i * px * 4) ((i + 1) * px * 4)
        dump := dump.append ei
        let (fa, fb, c) := sqErr inp tgt none (b0 + i)
        let (da, db, _) := sqErr inp tgt (some ei) (b0 + i)
        sums := #[sums[0]! + fa, sums[1]! + fb, sums[2]! + da, sums[3]! + db]
        nb := nb + c
    let name := (System.FilePath.mk path).fileStem.getD "eval"
    let parent := ((System.FilePath.mk path).parent.map (·.fileName.getD "")).getD ""
    IO.FS.writeBinFile s!"{outDir}/{parent}_{name}.eps.f32" dump
    let all := (k * px).toFloat
    IO.println s!"  {path} ({k} slices): whole {(sums[0]! / all).sqrt} → {(sums[2]! / all).sqrt}, body {(sums[1]! / nb.toFloat).sqrt} → {(sums[3]! / nb.toFloat).sqrt}"
  return 0
