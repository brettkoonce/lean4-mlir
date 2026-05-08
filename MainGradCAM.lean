import LeanMlir

/-! GradCAM (Zhou-2016 closed form) on a trained checkpoint.

For Phase 1 we cover any spec ending in `... → globalAvgPool → dense`.
The pre-built artifact for this exe is the ConvNeXt-T-GELU Imagenette
checkpoint produced by `MainAblation`. To wire up another model, swap
`spec` and the artifact prefix.

Usage:
  lake exe gradcam [N=4]    # auto-class CAM for first N val images
  Output: blueprint/src/figures/gradcam/convnext_t_gelu_strip.ppm

The walk is:
  1. Compile a `forward_cam` vmfb that returns the pre-GAP feature map
     `[B, C, H, W]` flat (no GAP, no dense).
  2. Run it once for a batch of N images.
  3. For each image, recompute logits with `F32.camLogits`, argmax to
     pick a target class, then `F32.camCompute` for that class.
  4. Bilinear-upsample to 224×224, viridis-colormap, α-blend over the
     de-normalized image.
  5. Stack 4 panels per image (input | overlay-pred | viridis-heatmap | label)
     into one PPM strip for the blueprint.

This carries no autodiff: for nets with this head, the per-channel
GradCAM weight is just the dense matrix's row for the target class. -/

def specGelu : NetSpec where
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
  let nVis : Nat := (args.head?.bind String.toNat?).getD 4
  let outPath : String := match args with
    | _ :: out :: _ => out
    | _ => "blueprint/src/figures/gradcam/convnext_t_gelu_strip.ppm"

  IO.FS.createDirAll (System.FilePath.mk outPath).parent.get!.toString

  match MlirCodegen.preGAPShape specGelu with
  | some (c, h, w) =>
      IO.eprintln s!"pre-GAP shape: ({c}, {h}, {w})  (lastConv {c}*{h}*{w} = {c*h*w} f32 per image)"
  | none =>
      IO.eprintln "spec has no globalAvgPool — not CAM-eligible"
      IO.Process.exit 1

  -- Trained checkpoint (produced by MainAblation's convnext_tiny_gelu run).
  let ckptPfx := ".lake/build/convnext_t_gelu_convnext_tiny_gelu"
  let paramsPath := s!"{ckptPfx}_params.bin"
  let bnPath     := s!"{ckptPfx}_bn_stats.bin"
  for p in [paramsPath, bnPath] do
    if !(← System.FilePath.pathExists p) then
      IO.eprintln s!"missing artifact: {p}"
      IO.eprintln "  (run MainAblation first to produce the convnext_tiny_gelu checkpoint)"
      IO.Process.exit 1

  IO.eprintln s!"  loading {paramsPath} ..."
  let params ← IO.FS.readBinFile paramsPath
  let bnStats ← IO.FS.readBinFile bnPath
  let evalParams := params.append bnStats

  -- Locate the final dense layer in the spec; pull out W ([fanIn, fanOut])
  -- and b ([fanOut]) bytes from the packed params buffer.
  let _ := specGelu.layers.length
  let (fanIn, fanOut) : Nat × Nat ← match specGelu.layers.getLast? with
    | some (Layer.dense fi fo _) => pure (fi, fo)
    | _ => IO.eprintln "spec must end in .dense for Phase 1 CAM"; IO.Process.exit 1
  -- Each layer occupies one or more shape slots in `paramShapes`. For
  -- a `.dense fi fo _`, exactly two: [fi, fo] then [fo].
  let nShapes := specGelu.paramShapes.size
  let denseShapeIdx := nShapes - 2  -- W is second-to-last shape
  let wOffElems := offsetBefore specGelu denseShapeIdx
  let denseW := F32.slice params wOffElems (fanIn * fanOut)
  let denseB := F32.slice params (wOffElems + fanIn * fanOut) fanOut
  IO.eprintln s!"  dense W slice: {F32.size denseW} f32 ({fanIn}x{fanOut})"
  IO.eprintln s!"  dense b slice: {F32.size denseB} f32 ({fanOut})"

  let evalBatch : Nat := 32  -- Match the cached training batch so we can reuse the param layout.
  -- Imagenette val is 224×224; load and slice nVis images for the strip.
  IO.eprintln "loading data/imagenette/val.bin ..."
  let (valImg, valLbl, nVal) ← F32.loadImagenette "data/imagenette/val.bin"
  if nVal < evalBatch then
    IO.eprintln s!"need ≥ {evalBatch} val records, found {nVal}"; IO.Process.exit 1
  let H := specGelu.imageH
  let W := specGelu.imageW
  let imgPixels := 3 * H * W
  -- Take the first `evalBatch` images (we'll only render `nVis` of them).
  let xba := F32.sliceImages valImg 0 evalBatch imgPixels

  -- Compile + load the cam vmfb at the same batch size (so the param
  -- layout matches the trained checkpoint).
  let camVmfb ← compileCamVmfb specGelu evalBatch
  IO.eprintln s!"  cam vmfb: {camVmfb}"
  let camSess ← IreeSession.create camVmfb
  let evalShapesBA := specGelu.evalShapesBA
  let xShape := specGelu.xShape evalBatch
  let camFnName := s!"{specGelu.sanitizedName}_cam.forward_cam"
  let lcSize := match MlirCodegen.preGAPShape specGelu with
    | some (c, h, w) => c * h * w
    | none => 0
  IO.eprintln s!"  invoking {camFnName} (batch={evalBatch}, lastConv elems = {lcSize})"
  let lastConv ← IreeSession.forwardF32 camSess camFnName
                    evalParams evalShapesBA xba xShape evalBatch.toUSize lcSize.toUSize
  IO.eprintln s!"  forward_cam returned {lastConv.size} bytes"

  -- preGAP shape — every image uses these dims.
  let (lcC, lcH, lcW) := match MlirCodegen.preGAPShape specGelu with
    | some t => t | none => (0, 0, 0)

  -- Build the output strip: [nVis × H, 3 × W, 3] = nVis rows of (input | overlay | heatmap)
  let panelW := W
  let stripW := 3 * panelW
  let stripH := nVis * H
  let mut allRows : ByteArray := ByteArray.emptyWithCapacity (stripH * stripW * 3)

  for i in [:nVis] do
    -- Logits + argmax for this image (auto class).
    let logits ← F32.camLogits denseW denseB lastConv i.toUSize
                    lcC.toUSize lcH.toUSize lcW.toUSize fanOut.toUSize
    let pred := argmaxN logits fanOut
    let label := valLbl.data[i * 4]!.toNat
    IO.eprintln s!"  img {i}: pred={pred} ({imagenetteClasses[pred]!}), label={label} ({imagenetteClasses[label]!})"

    -- CAM heatmap [lcH, lcW].
    let heatLow ← F32.camCompute denseW lastConv i.toUSize
                    lcC.toUSize lcH.toUSize lcW.toUSize fanOut.toUSize pred.toUSize
    -- Bilinear upsample to image resolution.
    let heatHi ← F32.bilinearUpsample2D heatLow
                    lcH.toUSize lcW.toUSize H.toUSize W.toUSize

    -- Decode the input image to row-major HWC UInt8.
    let imgRGB := chwF32ToHwcU8 valImg (i * imgPixels) H W

    -- Overlay heatmap.
    let overlay := Cam.overlayHeatmap imgRGB heatHi H W 0.55

    -- Pure heatmap render.
    let mut heatRGB : ByteArray := ByteArray.emptyWithCapacity (H * W * 3)
    for y in [:H] do
      for x in [:W] do
        let h := F32.read heatHi (y * W + x).toUSize
        let (r, g, b) := Cam.viridis h
        heatRGB := heatRGB.push r |>.push g |>.push b

    -- Stitch rows: for each scanline y, push [input | overlay | heatmap].
    for y in [:H] do
      let rowBase := y * W * 3
      let inRow := imgRGB.extract rowBase (rowBase + W * 3)
      let ovRow := overlay.extract rowBase (rowBase + W * 3)
      let htRow := heatRGB.extract rowBase (rowBase + W * 3)
      allRows := allRows.append inRow |>.append ovRow |>.append htRow

  Cam.writePPM outPath stripH stripW allRows
  IO.eprintln s!"  wrote {outPath} ({stripH}×{stripW} = {allRows.size + 50} bytes)"
  IO.eprintln s!"  view: display {outPath}   or   convert {outPath} {outPath.dropRight 4}.png"
