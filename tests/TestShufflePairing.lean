import LeanMlir.F32Array

/-!
# Image/label pairing regression test for `F32.shuffle`

`lean_f32_shuffle` used to permute images by a parameterized stride
(`pixelsPerImage * 4`) and labels by a **hardcoded 4 bytes** — the size of one
f32 classification scalar. Detection and segmentation labels are whole tensors
(the FPN detector's record is 185,220 floats = 740,880 bytes; BraTS masks are
240², Pets seg 224²), so every epoch permuted the images and left the targets
where they were. The trainer then saw mismatched image/target pairs for its
entire existence and could only learn the marginal target distribution: mAP@0.5
0.0001, an 8-image probe that refused to memorize, and nine investigations that
all came back refuted.

Nothing downstream could see it. Shapes, parameter counts, the loss and its VJP
all stay self-consistent under a permutation, and an FD probe checks the emitted
gradient against the *same* mispaired batch, so it passes too. The only way to
catch this class is to assert the invariant on the host side, which is what this
file does.

**The construction.** Record `k` is an image whose every pixel is the float `k`
and a label whose every float is also `k`. Pairing therefore holds if and only
if every float of label record `k` equals every float of image record `k`, for
every `k` — a per-element check over the whole buffer, not a summary statistic
(mean, std and min are all permutation-invariant and would report agreement on
data this test rejects).

Hermetic: synthesizes its own data, reads no files, needs no GPU.
-/

namespace ShufflePairing

/-- Build `n` records: image `k` is `pixels` copies of `k`, label `k` is
    `labelFloats` copies of `k`. -/
private def build (n pixels labelFloats : Nat) : IO (ByteArray × ByteArray) := do
  let mut imgs : Array ByteArray := #[]
  let mut lbls : Array ByteArray := #[]
  for k in [0:n] do
    imgs := imgs.push (← F32.const pixels.toUSize k.toFloat)
    lbls := lbls.push (← F32.const labelFloats.toUSize k.toFloat)
  return (F32.concat imgs, F32.concat lbls)

/-- Check the pairing invariant per element, and that the result is a genuine
    permutation of the input (every original record present exactly once).
    Returns `none` on success or the first failure explanation. -/
private def checkPaired (n pixels labelFloats : Nat)
    (imgs lbls : ByteArray) : IO (Option String) := do
  -- Sizes first: a truncated buffer would make the loops below vacuous.
  if F32.size imgs != n * pixels then
    return some s!"image buffer is {F32.size imgs} floats, expected {n * pixels}"
  if F32.size lbls != n * labelFloats then
    return some s!"label buffer is {F32.size lbls} floats, expected {n * labelFloats}"
  let mut seen : Array Bool := (List.replicate n false).toArray
  let mut identity := true
  for k in [0:n] do
    let tag := F32.read imgs (k * pixels).toUSize
    -- The record must be internally intact: a partial swap shows up here.
    for p in [0:pixels] do
      let v := F32.read imgs (k * pixels + p).toUSize
      if v != tag then
        return some s!"image record {k} is not intact: pixel {p} is {v}, pixel 0 is {tag}"
    -- THE INVARIANT: the label at slot k must still describe the image at slot k.
    for j in [0:labelFloats] do
      let v := F32.read lbls (k * labelFloats + j).toUSize
      if v != tag then
        return some s!"PAIRING BROKEN at slot {k}: image is record {tag}, but label float {j} is {v}"
    -- Permutation bookkeeping.
    let idx := tag.toUInt64.toNat
    if idx >= n || tag != idx.toFloat then
      return some s!"slot {k} carries tag {tag}, which is not a record index in [0,{n})"
    if seen[idx]! then
      return some s!"record {idx} appears at least twice — not a permutation"
    seen := seen.set! idx true
    if idx != k then identity := false
  -- A shuffle that did nothing would satisfy every check above vacuously.
  if identity && n > 8 then
    return some s!"shuffle left all {n} records in place — the test would pass vacuously"
  return none

/-- One case: build, shuffle at the honest stride, assert pairing survived. -/
private def runCase (name : String) (n pixels labelFloats seed : Nat) : IO Bool := do
  let (imgs, lbls) ← build n pixels labelFloats
  let (sImg, sLbl) ← F32.shuffle imgs lbls n.toUSize pixels.toUSize
                       (labelFloats * 4).toUSize seed.toUSize
  match ← checkPaired n pixels labelFloats sImg sLbl with
  | none =>
      IO.println s!"  PASS  {name} (n={n}, {pixels} px/img, {labelFloats} f32/label)"
      return true
  | some why =>
      IO.println s!"  FAIL  {name} (n={n}, {pixels} px/img, {labelFloats} f32/label)"
      IO.println s!"        {why}"
      return false

/-- The guard: an under-reported `labelBytes` is the original bug verbatim —
    the caller claims a 4-byte label while the buffer holds whole tensors, so
    only the first float of each record moves. It must be refused, not
    silently performed. Over-reporting must be refused too (it reads past the
    end of the buffer). -/
private def runGuardCase (name : String) (n pixels labelFloats claimedBytes : Nat) : IO Bool := do
  let (imgs, lbls) ← build n pixels labelFloats
  let refused ←
    try
      let (sImg, sLbl) ← F32.shuffle imgs lbls n.toUSize pixels.toUSize
                           claimedBytes.toUSize 7
      -- It did not refuse. Did it at least leave the data paired?
      match ← checkPaired n pixels labelFloats sImg sLbl with
      | none => pure (some "accepted the stride but the data stayed paired")
      | some why => pure (some s!"accepted the stride AND corrupted the data: {why}")
    catch _ => pure none
  match refused with
  | none =>
      IO.println s!"  PASS  {name} (claimed {claimedBytes} B/label, true {labelFloats * 4} B) — refused"
      return true
  | some why =>
      IO.println s!"  FAIL  {name} (claimed {claimedBytes} B/label, true {labelFloats * 4} B)"
      IO.println s!"        {why}"
      return false

end ShufflePairing

open ShufflePairing in
def main : IO UInt32 := do
  IO.println "=== F32.shuffle image/label pairing ==="
  let mut ok := true
  -- Classification: stride 4 is correct here, and was the only case the old
  -- hardcoded FFI got right. This is the regression guard on the fix itself.
  ok := (← runCase "classification scalar" 64 12 1 1) && ok
  -- Multi-float labels: everything the old code destroyed.
  ok := (← runCase "small multi-float label" 64 12 5 2) && ok
  ok := (← runCase "segmentation-shaped label" 33 16 64 3) && ok
  -- Label record much larger than the image record — the FPN's shape, where
  -- the temp buffer must be sized off the label, not the image.
  ok := (← runCase "label larger than image" 40 4 512 4) && ok
  -- n not divisible by anything convenient, and n=1 (loop body never runs).
  ok := (← runCase "prime n" 47 7 9 5) && ok
  ok := (← runCase "single record" 1 8 16 6) && ok

  IO.println "=== stride guard ==="
  -- THE ORIGINAL BUG, stated as a call: whole-tensor labels, stride claimed 4.
  ok := (← runGuardCase "under-reported stride (the original bug)" 64 12 64 4) && ok
  -- Under-reported but still a multiple of the true stride.
  ok := (← runGuardCase "under-reported stride (half)" 64 12 64 128) && ok
  -- Over-reported: reads past the end of the label buffer.
  ok := (← runGuardCase "over-reported stride" 64 12 8 64) && ok
  ok := (← runGuardCase "zero stride" 64 12 8 0) && ok

  if ok then
    IO.println "=== PASS ==="
    return 0
  else
    IO.println "=== FAIL ==="
    return 1
