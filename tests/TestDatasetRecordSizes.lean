import LeanMlir.F32Array

/-!
# Do the declared record sizes match what the loaders actually return?

`DatasetIO` declares `trainPixels` and `labelBytesPerRecord`; the C loaders in
`ffi/f32_helpers.c` allocate the buffers. Nothing checked that the two agree.
When they disagree the training loop does not fail — it slices batches at the
wrong stride, or (before `F32.shuffle` was made strict) permutes only a prefix
of each label record, which destroys image/target pairing silently.

This probe loads each dataset that is present on disk and asserts

    images = n × trainPixels × 4 bytes
    labels = n × labelBytesPerRecord bytes

exactly — the same invariant `lean_f32_shuffle` now enforces at runtime. It is
the cheap differential reference the host data path otherwise lacks: it compares
the *declared* layout against the *loaded* layout, which no proof obligation,
`iree-compile` run or FD probe can see.

Note the two label conventions this pins, which are easy to conflate: pets and
BraTS masks are **uint8**, one byte per pixel, so `224 * 224` and `240 * 240`
are byte counts and correctly have no `* 4`. Every detection target is **f32**,
so those spell out `... * 4`.

Datasets that are absent are skipped, not failed — this cannot run in CI, it is
a pre-flight check to run whenever a dataset or its preprocessing changes.
-/

namespace DatasetRecordSizes

private structure Case where
  name        : String
  path        : String
  pixels      : Nat
  labelBytes  : Nat
  load        : IO (ByteArray × ByteArray × Nat)

private def check (c : Case) : IO (Option Bool) := do
  if !(← System.FilePath.pathExists c.path) then
    IO.println s!"  SKIP  {c.name} — {c.path} not present"
    return none
  let (img, lbl, n) ← c.load
  let wantImg := n * c.pixels * 4
  let wantLbl := n * c.labelBytes
  let imgOk := img.size == wantImg
  let lblOk := lbl.size == wantLbl
  if imgOk && lblOk then
    IO.println s!"  PASS  {c.name}: n={n}, {c.pixels} px/img, {c.labelBytes} B/label"
    return some true
  else
    IO.println s!"  FAIL  {c.name}: n={n}"
    if !imgOk then
      IO.println s!"        images {img.size} B, declared trainPixels implies {wantImg} B"
      IO.println s!"        (loaded {img.size / n} B/record vs declared {c.pixels * 4})"
    if !lblOk then
      IO.println s!"        labels {lbl.size} B, declared labelBytesPerRecord implies {wantLbl} B"
      IO.println s!"        (loaded {lbl.size / n} B/record vs declared {c.labelBytes})"
    return some false

end DatasetRecordSizes

open DatasetRecordSizes in
def main : IO UInt32 := do
  IO.println "=== declared record size vs loaded record size ==="
  let cases : List Case := [
    -- Classification: one int32 class id per record.
    { name := "mnist (train)", path := "data/train-images-idx3-ubyte"
      pixels := 1 * 28 * 28, labelBytes := 4
      load := do
        let (imgs, n) ← F32.loadIdxImages "data/train-images-idx3-ubyte"
        let (lbls, _) ← F32.loadIdxLabels "data/train-labels-idx1-ubyte"
        return (imgs, lbls, n) },
    { name := "imagenette (val, 224)", path := "data/imagenette/val.bin"
      pixels := 3 * 224 * 224, labelBytes := 4
      load := F32.loadImagenette "data/imagenette/val.bin" },
    -- CIFAR builds its label buffer in Lean rather than C (one class byte
    -- padded to an int32), so it is the one classification path where the
    -- 4-byte stride is not the C loader's doing. Replicated from cifar10IO.
    { name := "cifar-10 (test batch)", path := "data/cifar-10/test_batch.bin"
      pixels := 3 * 32 * 32, labelBytes := 4
      load := do
        let raw ← IO.FS.readBinFile "data/cifar-10/test_batch.bin"
        let n := raw.size / 3073
        let mut labels : ByteArray := .empty
        for j in [:n] do
          labels := labels.push raw[j * 3073]!
          labels := labels.push 0
          labels := labels.push 0
          labels := labels.push 0
        let imgs ← F32.cifarBatch raw 0 n.toUSize
        return (imgs, labels, n) },
    -- Segmentation: uint8 mask, ONE byte per pixel (hence no `* 4`).
    { name := "pets seg (val)", path := "data/pets/val.bin"
      pixels := 3 * 224 * 224, labelBytes := 224 * 224
      load := F32.loadPets "data/pets/val.bin" },
    { name := "brats (val)", path := "data/brats/val.bin"
      pixels := 4 * 240 * 240, labelBytes := 240 * 240
      load := F32.loadBrats "data/brats/val.bin" 240 },
    -- Detection: f32 target tensors.
    { name := "pets det yolov1 (val)", path := "data/pets_det/val.bin"
      pixels := 3 * 224 * 224, labelBytes := 30 * 7 * 7 * 4 + 7 * 7 * 4 + 4 + 56 * 20
      load := F32.loadDetBin "data/pets_det/val.bin" },
    { name := "visdrone fpn (overfit-8)", path := "data/visdrone_fpn_of8/train.bin"
      pixels := 3 * 448 * 448, labelBytes := 185220 * 4
      load := F32.loadDetBinFpn "data/visdrone_fpn_of8/train.bin" 448 185220 },
    -- Anchor mode: target-only [A·15, gH, gW]. A=6, 448/32 = 14.
    { name := "visdrone anchor A=6 (val)", path := "data/visdrone448_a6/val.bin"
      pixels := 3 * 448 * 448, labelBytes := 6 * 15 * 14 * 14 * 4
      load := F32.loadDetBinAnchor "data/visdrone448_a6/val.bin" 448 14 14 6 },
    -- Dims mode: the yolov1 target+mask+numBoxes+raw_boxes record at a
    -- parameterized grid. 448/32 = 14 and the 28×28 variant.
    { name := "visdrone dims 14×14 (val)", path := "data/visdrone448/val.bin"
      pixels := 3 * 448 * 448
      labelBytes := 30 * 14 * 14 * 4 + 14 * 14 * 4 + 4 + 56 * 20
      load := F32.loadDetBinDims "data/visdrone448/val.bin" 448 14 14 },
    { name := "visdrone dims 28×28 (val)", path := "data/visdrone448_g28/val.bin"
      pixels := 3 * 448 * 448
      labelBytes := 30 * 28 * 28 * 4 + 28 * 28 * 4 + 4 + 56 * 20
      load := F32.loadDetBinDims "data/visdrone448_g28/val.bin" 448 28 28 }
  ]
  let mut ran := 0
  let mut ok := true
  for c in cases do
    match ← check c with
    | none => pure ()
    | some good => ran := ran + 1; ok := good && ok
  IO.println s!"=== {ran} dataset(s) checked ==="
  if ok then
    IO.println "=== PASS ==="
    return 0
  else
    IO.println "=== FAIL ==="
    return 1
