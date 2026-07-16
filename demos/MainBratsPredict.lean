import LeanMlir

/-! Render predictions from a trained BraTS segmentation checkpoint.

    The BraTS peer of `MainPetsPredict`, and for this demo's audience it is
    the deliverable rather than a garnish: a medical student evaluates a
    segmentation by *looking* at it. A collapsed model and a working one
    produce the same respectable-looking mIoU (0.243 vs the 0.243 of the
    trivial background-only predictor — planning/brats_demo.md Workstream A);
    they do not produce the same picture.

    For each of N chosen val slices, writes a PPM strip:

        T1gd | T1gd + ground truth | T1gd + prediction

    Three things differ from the pets renderer, each because MRI is not RGB:

    * **The backdrop is one modality, not the input.** The input has four
      co-registered modalities (FLAIR / T1w / T1gd / T2w) and there is no
      meaningful way to show four channels as one image. T1gd is the one
      clinicians read for enhancing tumour — the contrast agent is what makes
      the surgical target light up — so it is the backdrop, and the tumour
      being visible in it is the point.

    * **Masks are overlaid, not shown beside.** A trimap next to a photo is
      legible; a tumour mask floating on black is not — a tumour is only
      interpretable against the anatomy it sits in. Region colours follow the
      usual BraTS renderings: edema green, non-enhancing/necrotic red,
      enhancing yellow.

    * **Slices are chosen, not taken in order.** `preprocess_brats.py` keeps
      any slice with ≥1 tumour pixel, so the head of the val set is mostly
      near-empty tumour edges. Rendering those would show nothing either way.

    Usage:
        lake exe brats-predict [arm=<name>] [out.ppm] [params.bin] [bn_stats.bin]

    `arm=` selects which ablation arm's checkpoint to render, matching the tag
    `unet-brats-train` stamps on its artifacts (`NetSpec.buildTag`) — `arm=ce`,
    `arm=wce`, `arm=focal_pb`, and so on. Omit it for an untagged run.

    That is what makes the money figure a two-command job: the arms no longer
    overwrite each other's checkpoints, so both survive to be rendered.

        lake exe unet-brats-train data/brats 10 ce      # -> ..._ce_params.bin
        lake exe unet-brats-train data/brats 10 wce     # -> ..._wce_params.bin
        lake exe brats-predict arm=ce  demos/figures/brats_ce.ppm
        lake exe brats-predict arm=wce demos/figures/brats_wce.ppm

    The explicit params/bn paths still override, for checkpoints that live
    outside `.lake/build` (a copy saved aside, a file from another box).
-/

/-- MUST match `demos/MainUnetBratsTrain.lean` exactly, name string included:
    `buildPrefix` is derived from `spec.name`, so a single edited character
    here points this renderer at a checkpoint that does not exist. -/
def unetBrats : NetSpec where
  name := "UNet (BraTS, 240×240 4-modality MRI → 4-class tumour)"
  imageH := 240
  imageW := 240
  layers := [
    .unetDown 4   32,
    .unetDown 32  64,
    .unetDown 64  128,
    .unetDown 128 256,
    .convBn 256 512 3 1 .same,
    .convBn 512 512 3 1 .same,
    .unetUp 512 256,
    .unetUp 256 128,
    .unetUp 128 64,
    .unetUp 64  32,
    .conv2d 32 4 1 .same .identity
  ]

/-- Channel index of the modality used as the grayscale backdrop.
    0 = FLAIR, 1 = T1w, 2 = T1gd, 3 = T2w (order fixed by
    `preprocess_brats.py`, which reads it from MSD's dataset.json). -/
private def backdropModality : Nat := 2   -- T1gd

private def numModalities : Nat := 4
private def numClasses : Nat := 4

/-- z-score → 8-bit gray over a fixed [-2σ, +3σ] window.

    Fixed rather than per-slice min/max: `znorm_brain` already put every
    volume on a common scale (mean 0, std 1 over brain voxels), so a fixed
    window keeps two renders comparable — which is the entire point when the
    figure is a CE-vs-Dice A/B. The window is asymmetric because the
    interesting signal is bright: enhancing tumour on T1gd sits well above
    the brain mean. -/
private def zToGray (z : Float) : UInt8 :=
  let t := (z + 2.0) / 5.0
  let tc := if t < 0.0 then 0.0 else if t > 1.0 then 1.0 else t
  (tc * 255.0).toUInt8

/-- Grayscale backdrop for one pixel of batch element `b`. -/
private def backdropGray (xba : ByteArray) (b h w H W : Nat) : UInt8 :=
  let stride := H * W
  let imgOff := b * numModalities * stride
  let pix := h * W + w
  -- Brain mask: background is exact 0 in a skull-stripped volume and
  -- `znorm_brain` preserves that (it writes z only at nonzero voxels), so
  -- "0 in all four modalities" is background. Testing a single modality
  -- would instead punch black speckle through the brain: ~1.6% of brain
  -- voxels have a z within half a quantization step of 0 and dequantize to
  -- exactly 0. Agreeing across four independent modalities makes that
  -- coincidence vanish.
  let isBrain := (List.range numModalities).any fun c =>
    F32.read xba (imgOff + c * stride + pix).toUSize != 0.0
  if !isBrain then 0
  else zToGray (F32.read xba (imgOff + backdropModality * stride + pix).toUSize)

/-- Overlay colour for a tumour class, in MSD Task01's numbering
    (0 = background, 1 = edema, 2 = non-enhancing/necrotic, 3 = enhancing).
    `none` = leave the anatomy alone. -/
private def regionColor : UInt8 → Option (Float × Float × Float)
  | 1 => some (60.0, 200.0, 60.0)    -- edema: green
  | 2 => some (220.0, 50.0, 50.0)    -- non-enhancing / necrotic core: red
  | 3 => some (255.0, 215.0, 0.0)    -- enhancing tumour: yellow
  | _ => none                        -- background

/-- Alpha-blend the class colour over the grayscale anatomy. 0.55 keeps the
    underlying tissue readable — a solid mask would hide exactly the contrast
    a reader is trying to judge the boundary against. -/
private def overlayPx (gray : UInt8) (cls : UInt8) : UInt8 × UInt8 × UInt8 :=
  match regionColor cls with
  | none => (gray, gray, gray)
  | some (r, g, b) =>
    let a : Float := 0.55
    let gf := gray.toNat.toFloat
    let mix := fun (c : Float) => ((1.0 - a) * gf + a * c).toUInt8
    (mix r, mix g, mix b)

/-- Argmax over the `numClasses` channels at `(b, h, w)`. Logits are NCHW. -/
private def argmaxChannel (logits : ByteArray) (b h w H W : Nat) : UInt8 := Id.run do
  let stride := H * W
  let imgOff := b * numClasses * stride
  let pix := h * W + w
  let mut best : Nat := 0
  let mut bestV := F32.read logits (imgOff + pix).toUSize
  for c in [1:numClasses] do
    let v := F32.read logits (imgOff + c * stride + pix).toUSize
    if v > bestV then
      bestV := v
      best := c
  return best.toUInt8

/-- Display name for an arm; the empty tag is an untagged single run. -/
private def armLabel (a : String) : String :=
  if a.isEmpty then "prediction" else a

private structure SliceScore where
  idx : Nat
  /-- Pixels of any tumour class (1/2/3). -/
  tumour : Nat
  /-- Pixels of enhancing tumour (class 3) specifically. -/
  enhancing : Nat
  deriving Inhabited

def main (args : List String) : IO Unit := do
  -- `arm=<a>,<b>,…` — each name must reproduce the tag `unet-brats-train`
  -- stamped on its artifacts, or every path below points at a file nobody
  -- wrote. Several arms render as extra columns of ONE figure, on identical
  -- slices, which is the only honest way to show a loss ablation: two separate
  -- images invite the reader to wonder whether they even saw the same brain.
  let arms : List String :=
    match (args.filter (·.startsWith "arm=")).head? with
    | some a => ((a.drop 4).toString).splitOn ","
    | none => [""]
  let positional := args.filter (fun a => !a.startsWith "arm=")
  let outPath := positional[0]?.getD "demos/figures/brats_pred.ppm"
  -- The explicit params/bn override is single-arm only: with several arms
  -- there is no one checkpoint to point at, and silently applying one arm's
  -- weights to another's column would produce a plausible, wrong figure.
  if arms.length > 1 && positional.length > 1 then
    IO.eprintln "explicit params/bn paths are single-arm only — drop them, or pass one arm"
    IO.Process.exit 1
  let prefixes := arms.map (fun a => (unetBrats.withBuildTag a).buildPrefix)
  let paramPaths := match positional[1]? with
    | some p => [p]
    | none => prefixes.map (fun p => s!"{p}_params.bin")
  let bnPaths := match positional[2]? with
    | some p => [p]
    | none => prefixes.map (fun p => s!"{p}_bn_stats.bin")
  let vmfbs := prefixes.map (fun p => s!"{p}_fwd_eval.vmfb")
  IO.FS.createDirAll (System.FilePath.mk outPath).parent.get!.toString
  for p in paramPaths ++ bnPaths ++ vmfbs do
    if !(← System.FilePath.pathExists p) then
      IO.eprintln s!"missing artifact: {p}"
      IO.eprintln "  (run lake exe unet-brats-train first to produce checkpoint + vmfb)"
      IO.Process.exit 1
  let mut evalParamsList : List ByteArray := []
  for (pp, bp) in paramPaths.zip bnPaths do
    IO.eprintln s!"  loading {pp} ..."
    let params ← IO.FS.readBinFile pp
    let bnStats ← IO.FS.readBinFile bp
    evalParamsList := evalParamsList ++ [params.append bnStats]
  let spec := unetBrats
  IO.eprintln s!"  loading data/brats/val.bin ..."
  let (valImg, valMask, nVal) ← F32.loadBrats "data/brats/val.bin" 240
  -- The eval vmfb is compiled at the training batch size; we fill a full
  -- batch and render only the first `nVis` of it.
  let evalBatch : Nat := 16
  let nVis : Nat := 4
  if nVal == 0 then
    IO.eprintln "val.bin has no records"; IO.Process.exit 1
  let H := spec.imageH
  let W := spec.imageW
  let imgPixels := numModalities * H * W
  let maskPixels := H * W

  -- Slice selection. Score every val slice by tumour burden, on a stride-2
  -- subsample of the mask — this is a ranking, and 1-in-4 pixels ranks a
  -- 240² mask just as well as all of them for a fraction of the scan.
  IO.eprintln s!"  scoring {nVal} val slices for tumour burden ..."
  let scanStride : Nat := 2
  let mut cand : Array SliceScore := #[]
  for i in [:nVal] do
    let base := i * maskPixels
    let mut tum : Nat := 0
    let mut et : Nat := 0
    for h in [:H / scanStride] do
      for w in [:W / scanStride] do
        let c := valMask.get! (base + (h * scanStride) * W + w * scanStride)
        if c != 0 then
          tum := tum + 1
          if c == 3 then et := et + 1
    cand := cand.push { idx := i, tumour := tum, enhancing := et }
  -- Prefer slices that actually contain enhancing tumour: ET is the class the
  -- demo is about, and a slice without it cannot show whether ET collapsed.
  -- Fall back to raw tumour burden if too few qualify.
  let minEt : Nat := 25
  let withEt := cand.filter (fun s => s.enhancing >= minEt)
  let pool := if withEt.size >= nVis then withEt else cand
  let sorted := pool.qsort (fun a b => a.tumour > b.tumour)
  -- Spread the picks across patients. Slices are written volume by volume, so
  -- index distance is a proxy for "different brain"; without this the top-4 by
  -- burden are four adjacent slices of the single biggest tumour, which is one
  -- finding rendered four times. A proxy, not a guarantee — the manifest in
  -- data/brats/split.json has the exact volume boundaries if it ever matters.
  let minGap : Nat := 60
  let mut chosen : Array Nat := #[]
  for s in sorted do
    if chosen.size >= nVis then break
    if chosen.all (fun j => (if j > s.idx then j - s.idx else s.idx - j) >= minGap) then
      chosen := chosen.push s.idx
  if chosen.size < nVis then
    for s in sorted do
      if chosen.size >= nVis then break
      if !chosen.contains s.idx then chosen := chosen.push s.idx
  -- A val set smaller than `nVis` is not a real configuration, but the render
  -- loop indexes `chosen` directly and would panic rather than say so.
  let nRender := min nVis chosen.size
  IO.eprintln s!"  rendering slices {chosen.toList} of {nVal}"

  -- Gather the chosen slices into one batch. `sliceImages` is contiguous, so
  -- an arbitrary pick means appending them one at a time; the tail of the
  -- batch is padding (a repeat of the first pick) and is never rendered.
  let mut xba : ByteArray := ByteArray.empty
  for k in [:evalBatch] do
    let i := chosen[k % chosen.size]!
    xba := xba.append (F32.sliceImages valImg i 1 imgPixels)
  let xShape := spec.xShape evalBatch
  let evalShapes := spec.evalShapesBA
  IO.eprintln s!"  running {arms.length} eval forward(s) (batch={evalBatch}, rendering {nRender}) ..."
  let outElems : USize := (numClasses * H * W).toUSize
  -- One forward per arm, on the SAME batch — that identity is the point of
  -- rendering them together.
  let mut logitsList : List ByteArray := []
  for (vmfb, ep) in vmfbs.zip evalParamsList do
    let sess ← IreeSession.create vmfb
    let lg ← IreeSession.forwardF32 sess spec.evalFnName
                ep evalShapes xba xShape evalBatch.toUSize outElems
    logitsList := logitsList ++ [lg]
  IO.eprintln s!"  rendering PPM ..."

  -- (2 + one per arm) panels per slice: T1gd | +GT | +pred_a | +pred_b | …
  let nPanels := 2 + arms.length
  let stripW := nPanels * W
  let stripH := nRender * H
  let mut ppm : ByteArray := ByteArray.empty
  ppm := ppm.append s!"P6\n{stripW} {stripH}\n255\n".toUTF8
  for k in [:nRender] do
    let idx := chosen[k]!
    let mut gtPx : Nat := 0
    let mut predPx : Array Nat := Array.replicate arms.length 0
    for h in [:H] do
      -- panel 1: T1gd backdrop, no overlay
      for w in [:W] do
        let g := backdropGray xba k h w H W
        ppm := ppm.push g |>.push g |>.push g
      -- panel 2: ground truth over T1gd
      for w in [:W] do
        let g := backdropGray xba k h w H W
        let cls := valMask.get! (idx * maskPixels + h * W + w)
        if cls != 0 then gtPx := gtPx + 1
        let (r, gg, b) := overlayPx g cls
        ppm := ppm.push r |>.push gg |>.push b
      -- panels 3..: one prediction per arm, same slice, same backdrop
      for (logits, ai) in logitsList.zipIdx do
        for w in [:W] do
          let g := backdropGray xba k h w H W
          let cls := argmaxChannel logits k h w H W
          if cls != 0 then predPx := predPx.set! ai (predPx[ai]! + 1)
          let (r, gg, b) := overlayPx g cls
          ppm := ppm.push r |>.push gg |>.push b
    -- The scalar that captions the figure: a collapsed model prints 0 here
    -- against a four-digit ground truth.
    let per := String.intercalate "  " (arms.zipIdx.map (fun (a, i) =>
      armLabel a ++ "=" ++ toString predPx[i]!))
    IO.eprintln s!"    slice {idx}: gt tumour px={gtPx}  predicted: {per}"
  IO.FS.writeBinFile outPath ppm
  IO.eprintln s!"  wrote {outPath} ({ppm.size} bytes)"
  let panelNames := String.intercalate " | " (arms.map (fun a => "+" ++ armLabel a))
  IO.eprintln s!"  panels: T1gd | +ground truth | {panelNames}"
  IO.eprintln s!"  colours: edema green, non-enhancing red, enhancing yellow"
  IO.eprintln s!"  view with e.g. `display {outPath}` or convert:"
  IO.eprintln s!"    convert {outPath} {outPath.dropEnd 4}.png"
