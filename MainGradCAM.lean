import LeanMlir

/-! GradCAM (Zhou-2016 closed form) on a trained checkpoint.

For Phase 1 we cover any spec ending in `... → globalAvgPool → dense`.
Two pre-wired models for the demo:

  * `convnext` — ConvNeXt-T-GELU on Imagenette (the convnext_tiny_gelu
    ablation run; pre-GAP feature 768×7×7).
  * `r34` — ResNet-34 on Imagenette (the r34-full ablation run; pre-GAP
    feature 512×7×7).

Usage:
  lake exe gradcam [model=convnext|r34] [N=4] [out.ppm]

The walk is:
  1. Compile a `forward_cam` vmfb that returns the pre-GAP feature map
     `[B, C, H, W]` flat (no GAP, no dense).
  2. Run it once for a batch of N images.
  3. For each image, recompute logits with `F32.camLogits`, argmax to
     pick a target class, then `F32.camCompute` for that class.
  4. Bilinear-upsample to 224×224, viridis-colormap, α-blend over the
     de-normalized image.
  5. Stack panels per image (input | overlay | viridis-heatmap) into
     one PPM strip for the blueprint.

This carries no autodiff: for nets with this head, the per-channel
GradCAM weight is just the dense matrix's row for the target class. -/

def convNextTinyGelu : NetSpec where
  name := "ConvNeXt-T-GELU"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 96 4 4 .same,
    .convNextStage 96 3 .ln .gelu,
    .convNextDownsample 96 192,
    .convNextStage 192 3 .ln .gelu,
    .convNextDownsample 192 384,
    .convNextStage 384 9 .ln .gelu,
    .convNextDownsample 384 768,
    .convNextStage 768 3 .ln .gelu,
    .globalAvgPool,
    .dense 768 10 .identity
  ]

def resnet34 : NetSpec where
  name := "ResNet-34"
  imageH := 224
  imageW := 224
  layers := [
    .convBn 3 64 7 2 .same,
    .maxPool 2 2,
    .residualBlock  64  64 3 1,
    .residualBlock  64 128 4 2,
    .residualBlock 128 256 6 2,
    .residualBlock 256 512 3 2,
    .globalAvgPool,
    .dense 512 10 .identity
  ]

/-- A picked model: spec, the file prefix where the trained params live,
    and the default output PPM path. -/
structure Model where
  spec       : NetSpec
  ckptPfx    : String
  defaultOut : String

def pickModel : String → Option Model
  | "convnext" =>
    some { spec := convNextTinyGelu
           ckptPfx := ".lake/build/convnext_t_gelu_convnext_tiny_gelu"
           defaultOut := "blueprint/src/figures/gradcam/convnext_t_gelu_strip.ppm" }
  | "r34" =>
    some { spec := resnet34
           ckptPfx := ".lake/build/resnet_34_r34_full"
           defaultOut := "blueprint/src/figures/gradcam/resnet34_strip.ppm" }
  | _ => none

def imagenetteClasses : Array String := #[
  "tench", "English-springer", "cassette-player", "chainsaw", "church",
  "French-horn", "garbage-truck", "gas-pump", "golf-ball", "parachute"
]

/-- ImageNet de-normalization for a single channel value `v`. -/
private def denormToU8 (c : Nat) (v : Float) : UInt8 :=
  let (mean, std) :=
    if c == 0      then (0.485, 0.229)
    else if c == 1 then (0.456, 0.224)
    else                 (0.406, 0.225)
  let p := (v * std + mean) * 255.0
  let pc := if p < 0.0 then 0.0 else if p > 255.0 then 255.0 else p
  pc.toUInt8

/-- Convert a single NCHW `[3, H, W]` slice into row-major `[H, W, 3]`
    UInt8 RGB suitable for PPM / overlay. -/
private def chwF32ToHwcU8 (chw : ByteArray) (start H W : Nat) : ByteArray := Id.run do
  let plane := H * W
  let mut out := ByteArray.emptyWithCapacity (H * W * 3)
  for i in [:H] do
    for j in [:W] do
      let off := i * W + j
      let r := denormToU8 0 (F32.read chw (start + 0 * plane + off).toUSize)
      let g := denormToU8 1 (F32.read chw (start + 1 * plane + off).toUSize)
      let b := denormToU8 2 (F32.read chw (start + 2 * plane + off).toUSize)
      out := out.push r |>.push g |>.push b
  return out

/-- Sum of preceding param shape sizes (in elements) up to layer index
    `targetIdx` exclusive. Used to compute the byte offset of the final
    dense W and b in the packed params buffer. -/
private def offsetBefore (spec : NetSpec) (targetIdx : Nat) : Nat := Id.run do
  let shapes := spec.paramShapes
  let mut acc : Nat := 0
  for i in [:targetIdx] do
    let sh := shapes[i]!
    let mut sz : Nat := 1
    for d in sh do sz := sz * d
    acc := acc + sz
  return acc

/-- Argmax over `n` f32s starting at byte-offset 0 of `ba`. -/
private def argmaxN (ba : ByteArray) (n : Nat) : Nat := Id.run do
  let mut best : Nat := 0
  let mut bestV : Float := F32.read ba 0
  for i in [1:n] do
    let v := F32.read ba i.toUSize
    if v > bestV then bestV := v; best := i
  return best

/-- Compile (or fetch from cache) the `forward_cam` vmfb for `spec`.
    Mirrors the pattern in `Train.compileVmfbs`. -/
private def compileCamVmfb (spec : NetSpec) (batchSize : Nat) : IO String := do
  let pfx := spec.buildPrefix
  let mlirPath := s!"{pfx}_fwd_cam.mlir"
  let vmfbPath := s!"{pfx}_fwd_cam.vmfb"
  let mlir := MlirCodegen.generateForwardCam spec batchSize
  IO.FS.writeFile mlirPath mlir
  if (← System.FilePath.pathExists vmfbPath) then
    IO.eprintln s!"  cam vmfb cached: {vmfbPath}"
    return vmfbPath
  IO.eprintln s!"  compiling cam vmfb -> {vmfbPath}"
  let compiler := if (← System.FilePath.pathExists ".venv/bin/iree-compile")
                    then ".venv/bin/iree-compile"
                    else "iree-compile"
  let args ← ireeCompileArgs mlirPath vmfbPath
  let r ← IO.Process.output { cmd := compiler, args := args }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile failed: {r.stderr.take 3000}"
    IO.Process.exit 1
  return vmfbPath

def main (args : List String) : IO Unit := do
  -- Args: [model=convnext|r34] [nVis=4] [out.ppm]
  let modelKey := args.head?.getD "convnext"
  let model ← match pickModel modelKey with
    | some m => pure m
    | none =>
      IO.eprintln s!"Unknown model '{modelKey}'. Use 'convnext' or 'r34'."
      IO.Process.exit 1
  let spec := model.spec

  let nVis : Nat := match args with
    | _ :: n :: _ => (n.toNat?).getD 4
    | _ => 4
  let outPath : String := match args with
    | _ :: _ :: out :: _ => out
    | _ => model.defaultOut

  IO.FS.createDirAll (System.FilePath.mk outPath).parent.get!.toString

  match MlirCodegen.preGAPShape spec with
  | some (c, h, w) =>
      IO.eprintln s!"model={modelKey}  pre-GAP shape: ({c}, {h}, {w})  (lastConv {c*h*w} f32 per image)"
  | none =>
      IO.eprintln "spec has no globalAvgPool — not CAM-eligible"
      IO.Process.exit 1

  let paramsPath := s!"{model.ckptPfx}_params.bin"
  let bnPath     := s!"{model.ckptPfx}_bn_stats.bin"
  for p in [paramsPath, bnPath] do
    if !(← System.FilePath.pathExists p) then
      IO.eprintln s!"missing artifact: {p}"
      IO.eprintln s!"  (run the training run for '{modelKey}' first)"
      IO.Process.exit 1

  IO.eprintln s!"  loading {paramsPath} ..."
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ← IO.FS.readBinFile bnPath
  let evalParams := params.append bnStats

  -- Locate the final dense layer; pull out W and b from the packed params.
  let (fanIn, fanOut) : Nat × Nat ← match spec.layers.getLast? with
    | some (Layer.dense fi fo _) => pure (fi, fo)
    | _ => IO.eprintln "spec must end in .dense for Phase 1 CAM"; IO.Process.exit 1
  -- A `.dense fi fo _` occupies two paramShape slots: [fi, fo] then [fo].
  let nShapes := spec.paramShapes.size
  let denseShapeIdx := nShapes - 2
  let wOffElems := offsetBefore spec denseShapeIdx
  let denseW := F32.slice params wOffElems (fanIn * fanOut)
  let denseB := F32.slice params (wOffElems + fanIn * fanOut) fanOut
  IO.eprintln s!"  dense W slice: {F32.size denseW} f32 ({fanIn}x{fanOut})"
  IO.eprintln s!"  dense b slice: {F32.size denseB} f32 ({fanOut})"

  let evalBatch : Nat := 32  -- Match the cached training batch.
  IO.eprintln "loading data/imagenette/val.bin ..."
  let (valImg, valLbl, nVal) ← F32.loadImagenette "data/imagenette/val.bin"
  if nVal < evalBatch then
    IO.eprintln s!"need ≥ {evalBatch} val records, found {nVal}"; IO.Process.exit 1
  let H := spec.imageH
  let W := spec.imageW
  let imgPixels := 3 * H * W
  let xba := F32.sliceImages valImg 0 evalBatch imgPixels

  let camVmfb ← compileCamVmfb spec evalBatch
  IO.eprintln s!"  cam vmfb: {camVmfb}"
  let camSess ← IreeSession.create camVmfb
  let evalShapesBA := spec.evalShapesBA
  let xShape := spec.xShape evalBatch
  let camFnName := s!"{spec.sanitizedName}_cam.forward_cam"
  let lcSize := match MlirCodegen.preGAPShape spec with
    | some (c, h, w) => c * h * w
    | none => 0
  IO.eprintln s!"  invoking {camFnName} (batch={evalBatch}, lastConv elems = {lcSize})"
  let lastConv ← IreeSession.forwardF32 camSess camFnName
                    evalParams evalShapesBA xba xShape evalBatch.toUSize lcSize.toUSize
  IO.eprintln s!"  forward_cam returned {lastConv.size} bytes"

  let (lcC, lcH, lcW) := match MlirCodegen.preGAPShape spec with
    | some t => t | none => (0, 0, 0)

  -- Build the output strip: nVis rows × (input | overlay | heatmap).
  let panelW := W
  let stripW := 3 * panelW
  let stripH := nVis * H
  let mut allRows : ByteArray := ByteArray.emptyWithCapacity (stripH * stripW * 3)

  for i in [:nVis] do
    let logits ← F32.camLogits denseW denseB lastConv i.toUSize
                    lcC.toUSize lcH.toUSize lcW.toUSize fanOut.toUSize
    let pred := argmaxN logits fanOut
    let label := valLbl.data[i * 4]!.toNat
    IO.eprintln s!"  img {i}: pred={pred} ({imagenetteClasses[pred]!}), label={label} ({imagenetteClasses[label]!})"

    let heatLow ← F32.camCompute denseW lastConv i.toUSize
                    lcC.toUSize lcH.toUSize lcW.toUSize fanOut.toUSize pred.toUSize
    let heatHi ← F32.bilinearUpsample2D heatLow
                    lcH.toUSize lcW.toUSize H.toUSize W.toUSize

    let imgRGB := chwF32ToHwcU8 valImg (i * imgPixels) H W
    let overlay := Cam.overlayHeatmap imgRGB heatHi H W 0.55

    let mut heatRGB : ByteArray := ByteArray.emptyWithCapacity (H * W * 3)
    for y in [:H] do
      for x in [:W] do
        let h := F32.read heatHi (y * W + x).toUSize
        let (r, g, b) := Cam.viridis h
        heatRGB := heatRGB.push r |>.push g |>.push b

    for y in [:H] do
      let rowBase := y * W * 3
      let inRow := imgRGB.extract rowBase (rowBase + W * 3)
      let ovRow := overlay.extract rowBase (rowBase + W * 3)
      let htRow := heatRGB.extract rowBase (rowBase + W * 3)
      allRows := allRows.append inRow |>.append ovRow |>.append htRow

  Cam.writePPM outPath stripH stripW allRows
  IO.eprintln s!"  wrote {outPath} ({stripH}×{stripW} = {allRows.size + 50} bytes)"
  IO.eprintln s!"  view: display {outPath}   or   convert {outPath} {outPath.dropRight 4}.png"
