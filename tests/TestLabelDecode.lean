import LeanMlir.F32Array

/-! # `label-check` — the WIDTH gate on eval label decoding

The second defect found in the scoring loop on 2026-08-04/05, and the twin of `argmax-check`:
both were 10-class assumptions sitting in code that also runs at 1000 classes, and both were
correct everywhere the repo GATES (Imagenette, CIFAR-10 and MNIST are all 10-class).

`ByteArray.get!` returns a **`UInt8`**, so the old `(evalLbl.get! (4 * i)).toNat` read **byte 0
only** of a little-endian int32 — i.e. `label % 256`. On ImageNet that silently discarded the high
byte of every label from class 256 upward, and since a correct prediction can then only ever match
on classes 0..255, it capped reported top-1 at roughly a quarter of the truth.

⭐ **Measured off the real val wire, 2026-08-05**: the first validation batch carries labels
**1..988**, with **193 of 256 (75.4%) above 255**. And the end-to-end effect was measured too —
the R34/ImageNet run reported **10.57%** at epoch 5 where the JAX reference on the same box, same
data, same recipe reported **34.68%**, while their TRAIN LOSSES agreed to 0.055 nats at epochs 1,
2 and 5. Training was never wrong; only the scoring was. `34.68% × (256/1000) ≈ 8.9%`, plus the
accidental collisions where a prediction `c < 256` meets a truncated label from class `c + 256k`,
lands on the observed figure.

Gate A pins the decode. Gate B is the CONTROL — the old byte-0 read, required to be WRONG on a
label above 255, so a regression cannot pass this file quietly.
-/

/-- Pack `labels` the way the shim's wire and the driver's buffers do: little-endian int32. -/
def packLabels (labels : Array Nat) : ByteArray := Id.run do
  let mut b : ByteArray := .empty
  for l in labels do
    b := b.push (UInt8.ofNat (l % 256))
    b := b.push (UInt8.ofNat ((l / 256) % 256))
    b := b.push (UInt8.ofNat ((l / 65536) % 256))
    b := b.push (UInt8.ofNat ((l / 16777216) % 256))
  b

def main : IO Unit := do
  -- Spans the ImageNet range and straddles every byte boundary that matters.
  let labels := #[0, 1, 9, 255, 256, 257, 511, 512, 700, 988, 999]
  let buf := packLabels labels
  let mut bad := 0

  -- ══ Gate A: the full int32 survives the round trip ══
  let mut wrong : Array (Nat × Nat) := #[]
  for (want, i) in labels.zipIdx do
    let got := F32.readLabel buf i
    if got != want then wrong := wrong.push (want, got)
  if wrong.isEmpty then
    IO.println s!"  ✅ A  readLabel round-trips all {labels.size} labels (0..999, incl. 255/256/257)"
  else
    IO.println s!"  ❌ A  readLabel wrong on {wrong.size}: {wrong}"; bad := bad + 1

  -- ══ Gate B (CONTROL): the OLD byte-0 read must be WRONG above 255 ══
  -- If this ever agrees, the decode has silently narrowed again.
  let hi := labels.filter (· > 255)
  let mut byte0Agrees := 0
  for l in hi do
    let i := labels.findIdx? (· == l) |>.getD 0
    if (buf.get! (4 * i)).toNat == l then byte0Agrees := byte0Agrees + 1
  if byte0Agrees == 0 then
    IO.println s!"  ✅ B  control: byte-0-only MISSES every one of the {hi.size} labels > 255 \
(e.g. 988 reads as {(buf.get! (4 * (labels.findIdx? (· == 988) |>.getD 0))).toNat})"
  else
    IO.println s!"  ❌ B  control did NOT fire: byte-0 matched {byte0Agrees} labels > 255"
    bad := bad + 1

  -- ══ Gate C: the ImageNet range is genuinely mostly out of one byte ══
  -- The scale of the old defect, asserted rather than remembered.
  let outOfByte := (List.range 1000).filter (· > 255) |>.length
  if outOfByte == 744 then
    IO.println s!"  ✅ C  {outOfByte}/1000 ImageNet classes are unrepresentable in one byte (74.4%)"
  else
    IO.println s!"  ❌ C  expected 744, got {outOfByte}"; bad := bad + 1

  if bad == 0 then
    IO.println "label-check: OK (2 gates + 1 control)"
  else
    IO.println s!"label-check: {bad} FAILED"
    throw <| IO.userError "label-check failed"
