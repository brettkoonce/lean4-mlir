import LeanMlir.F32Array

/-! Class Activation Map (Zhou 2016) — visualization helpers.

The heavy lifts (CAM compute, bilinear upsample, logit recomputation)
are in `F32Array.lean` via FFI. This file provides:

  * `viridis : Float → (UInt8 × UInt8 × UInt8)` — perceptually uniform
    256-entry colormap (also legible in greyscale).
  * `overlayHeatmap : ...` — α-blend a heatmap over an RGB image.
  * `writePPM : ...` — binary P6 PPM serializer (zero deps; convert to
    PNG via `convert` outside if final asset size matters).

No autodiff is needed — for any network ending in
`... → globalAvgPool → dense ic oc act` GradCAM collapses to the
closed form Zhou et al. computed three years before Selvaraju et al.
generalized it. See `planning/gradcam.md`. -/

namespace Cam

/-- 32-stop subsample of matplotlib's viridis (perceptually uniform).
    We linearly interpolate between stops for the full 256-entry LUT.
    Hardcoded so we don't carry a runtime dependency on a colormap file. -/
private def viridisStops : Array (UInt8 × UInt8 × UInt8) := #[
  (68, 1, 84),    (71, 19, 101),  (72, 36, 117),  (70, 52, 128),
  (65, 68, 135),  (59, 82, 139),  (52, 95, 141),  (47, 108, 142),
  (42, 120, 142), (37, 132, 142), (33, 144, 141), (30, 156, 138),
  (33, 168, 132), (45, 179, 124), (66, 190, 113), (93, 200, 99),
  (123, 209, 81),(157, 217, 59), (192, 222, 41), (224, 226, 28),
  (253, 231, 36),
  -- Pad the tail with the last entry so the linear interp formula
  -- below is safe even at idx=255.
  (253, 231, 36),(253, 231, 36),(253, 231, 36),(253, 231, 36),
  (253, 231, 36),(253, 231, 36),(253, 231, 36),(253, 231, 36),
  (253, 231, 36),(253, 231, 36),(253, 231, 36)
]

/-- Map `t ∈ [0, 1]` to an RGB triple using a lerped viridis palette.
    Out-of-range `t` is clamped to `[0, 1]`. -/
def viridis (t : Float) : UInt8 × UInt8 × UInt8 :=
  let tc := if t < 0.0 then 0.0 else if t > 1.0 then 1.0 else t
  let n := viridisStops.size
  -- We have 21 real stops + tail padding. Index into the first 21.
  let nReal : Nat := 21
  let f := tc * (nReal - 1).toFloat
  let i0 : Nat := f.toUInt32.toNat
  let i1 : Nat := if i0 + 1 < nReal then i0 + 1 else i0
  let frac := f - i0.toFloat
  let _ := n
  let (r0, g0, b0) := viridisStops[i0]!
  let (r1, g1, b1) := viridisStops[i1]!
  let lerp (a b : UInt8) : UInt8 :=
    let av := a.toNat.toFloat
    let bv := b.toNat.toFloat
    let v := av + (bv - av) * frac
    let vc := if v < 0.0 then 0.0 else if v > 255.0 then 255.0 else v
    vc.toUInt8
  (lerp r0 r1, lerp g0 g1, lerp b0 b1)

/-- α-blend a heatmap over an RGB image and emit the result as a
    contiguous `[H, W, 3]` UInt8 buffer (PPM-ready).

    * `imgRGB` : already-denormalized `[H, W, 3]` UInt8 (R, G, B
      interleaved row-major). Caller is responsible for the
      ImageNet-mean/std de-normalization for whatever dataset.
    * `heat`   : f32 `[H, W]` with values in `[0, 1]`. -/
def overlayHeatmap (imgRGB : ByteArray) (heat : ByteArray)
    (H W : Nat) (alpha : Float) : ByteArray := Id.run do
  let mut out : ByteArray := ByteArray.emptyWithCapacity (H * W * 3)
  for i in [:H] do
    for j in [:W] do
      let pix := i * W + j
      let r := imgRGB.get! (pix * 3 + 0)
      let g := imgRGB.get! (pix * 3 + 1)
      let b := imgRGB.get! (pix * 3 + 2)
      let h := F32.read heat pix.toUSize
      let (hr, hg, hb) := viridis h
      let blend (c h : UInt8) : UInt8 :=
        let cv := c.toNat.toFloat
        let hv := h.toNat.toFloat
        let v := (1.0 - alpha) * cv + alpha * hv
        v.toUInt8
      out := out.push (blend r hr) |>.push (blend g hg) |>.push (blend b hb)
  return out

/-- Write a binary P6 PPM file. `pixels` is `[H, W, 3]` UInt8 row-major. -/
def writePPM (path : String) (H W : Nat) (pixels : ByteArray) : IO Unit := do
  let mut ba : ByteArray := s!"P6\n{W} {H}\n255\n".toUTF8
  ba := ba.append pixels
  IO.FS.writeBinFile path ba

/-- Render a heatmap directly as a colored PPM (no overlay) — useful
    for quick "did this work at all" checks. -/
def writeHeatmapPPM (path : String) (heat : ByteArray) (H W : Nat) : IO Unit := do
  let mut pixels : ByteArray := ByteArray.emptyWithCapacity (H * W * 3)
  for i in [:H] do
    for j in [:W] do
      let h := F32.read heat (i * W + j).toUSize
      let (r, g, b) := viridis h
      pixels := pixels.push r |>.push g |>.push b
  writePPM path H W pixels

end Cam
