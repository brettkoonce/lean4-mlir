import LeanMlir

/-! Sample CIFAR-style images from a trained tiny DDPM checkpoint.
    Mirrors `MainMnistDdpmSample` but for 3-channel RGB. -/

def tinyCifarDdpm : NetSpec where
  name := "DDPM UNet T-cond base64 bottleneck-attn (CIFAR 32x32x3)"
  imageH := 32
  imageW := 32
  layers := [
    .unetDown 4 64,
    .unetDown 64 128,
    .convBn 128 256 3 1 .same,
    .spatialFlatten,
    .transformerEncoder 256 4 1024 1
        (causalMask := false) (keepSequence := true),
    .spatialUnflatten 256 8 8,
    .convBn 256 256 3 1 .same,
    .unetUp 256 128,
    .unetUp 128 64,
    .conv2d 64 3 1 .same .identity
  ]

private def runIree (mlirPath outPath : String) : IO Bool := do
  let args ← ireeCompileArgs mlirPath outPath
  let compiler ← if (← System.FilePath.pathExists ".venv/bin/iree-compile")
                 then pure ".venv/bin/iree-compile" else pure "iree-compile"
  let r ← IO.Process.output { cmd := compiler, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile failed: {r.stderr.take 3000}"
    return false
  return true

/-- Map [-1, 1] (DDPM-centered training data) back to [0, 255] uint8
    for rendering. Clamp guards anything that escaped that range. -/
private def floatToU8 (v : Float) : UInt8 :=
  let scaled := (v + 1.0) * 0.5
  let p := if scaled < 0.0 then 0.0 else if scaled > 1.0 then 1.0 else scaled
  (p * 255.0).toUInt8

def main (args : List String) : IO Unit := do
  let outPath := args.head?.getD "runs/2026-05-07-cifar-ddpm/samples.ppm"
  IO.FS.createDirAll (System.FilePath.mk outPath).parent.get!.toString
  let spec := tinyCifarDdpm
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix
  let evalMlirPath := s!"{pfx}_fwd_eval.mlir"
  let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
  let B : Nat := 16
  let imgC : Nat := 3
  let nPix : Nat := imgC * spec.imageH * spec.imageW

  if !(← System.FilePath.pathExists evalVmfb) then
    let mlir := MlirCodegen.generateEval spec B
    IO.FS.writeFile evalMlirPath mlir
    IO.eprintln s!"  generated eval mlir ({mlir.length} chars), compiling..."
    unless (← runIree evalMlirPath evalVmfb) do IO.Process.exit 1
    IO.eprintln "  eval forward compiled"

  let paramsPath := s!"{pfx}_params.bin"
  let bnPath := s!"{pfx}_bn_stats.bin"
  for p in [paramsPath, bnPath] do
    if !(← System.FilePath.pathExists p) then
      IO.eprintln s!"missing checkpoint: {p}"
      IO.eprintln "  run lake exe cifar-ddpm-train data first"
      IO.Process.exit 1
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ← IO.FS.readBinFile bnPath
  let evalParams := params.append bnStats
  let evalShapes := spec.evalShapesBA
  let xShape := spec.xShape B

  let T : Nat := 1000
  let alphaBar ← Ddpm.cosineSchedule T.toUSize
  let nSteps : Nat := 50
  let stride : Nat := T / nSteps
  let stepTs : Array Nat := Id.run do
    let mut s : Array Nat := #[]
    for k in [:nSteps] do s := s.push (T - 1 - k * stride)
    s

  let alphaBarF : Nat → Float := fun t => F32.read alphaBar t.toUSize

  let mut x ← Ddpm.sampleNoise (B * nPix).toUSize 0xc0ffee
  let nTotal : USize := (B * nPix).toUSize

  let sess ← IreeSession.create evalVmfb
  IO.eprintln s!"  sampling: {nSteps} DDIM steps, batch {B}"
  for k in [:nSteps] do
    let t := stepTs[k]!
    let tPrev : Nat := if k + 1 < nSteps then stepTs[k + 1]! else 0
    let xCond ← Ddpm.prependTChannelScalar x B.toUSize imgC.toUSize
                  spec.imageH.toUSize spec.imageW.toUSize t.toUSize T.toUSize
    let eps ← IreeSession.forwardF32 sess spec.evalFnName
                evalParams evalShapes xCond xShape B.toUSize nPix.toUSize
    let aBarT := alphaBarF t
    let aBarP := if k + 1 < nSteps then alphaBarF tPrev else 0.9999
    let sqAT := Float.sqrt aBarT
    let sqAP := Float.sqrt aBarP
    let sqOmAT := Float.sqrt (1.0 - aBarT)
    let sqOmAP := Float.sqrt (1.0 - aBarP)
    let a := sqAP / sqAT
    let b := sqOmAP - a * sqOmAT
    x ← Ddpm.ddimStep x eps a b nTotal
    if k % 10 == 0 || k == nSteps - 1 then
      IO.eprintln s!"  step {k}/{nSteps} t={t}->{tPrev}  a={a} b={b}"

  -- ── Render 4×4 RGB grid ──
  let H := spec.imageH; let W := spec.imageW
  let gridW := 4 * W
  let gridH := 4 * H
  let mut ppm : ByteArray := ByteArray.empty
  ppm := ppm.append s!"P6\n{gridW} {gridH}\n255\n".toUTF8
  let chanStride := H * W
  let imgStride := imgC * chanStride
  for gy in [:4] do
    for h in [:H] do
      for gx in [:4] do
        let idx := gy * 4 + gx
        for w in [:W] do
          let r := F32.read x (idx * imgStride + 0 * chanStride + h * W + w).toUSize
          let g := F32.read x (idx * imgStride + 1 * chanStride + h * W + w).toUSize
          let b := F32.read x (idx * imgStride + 2 * chanStride + h * W + w).toUSize
          ppm := ppm.push (floatToU8 r) |>.push (floatToU8 g) |>.push (floatToU8 b)
  IO.FS.writeBinFile outPath ppm
  IO.eprintln s!"  wrote {outPath} ({ppm.size} bytes)"
