import LeanMlir.F32Array

/-! # `argmax-check` — the CLASS-COUNT gate on `F32.argmaxN`

An argmax with a literal class count (a C loop `for (i = 1; i < 10; i++)`) is correct on every
10-class net — Imagenette, CIFAR-10 and MNIST — so it agrees with the net everywhere those tests
look. It is wrong on exactly the tier with no accuracy gate — the 1000-class ImageNet trainers —
where it confines every prediction to labels 0..9 and therefore scores a net that can only ever
be right on the ~1/100 of validation images whose label falls in that window. Training stays
healthy; only the readout is wrong. A readout with nowhere to put the class count cannot be
checked against the net.

Gate A pins `F32.argmaxN` at the net's class count. Gate B is the CONTROL — it re-runs a 10-wide
window on the same data and requires it to MISS, so a regression cannot pass this file quietly.
-/

/-- A `rows × K` logit block, all zeros except one peak per row at the given index. -/
def mkLogits (rows K : Nat) (peakAt : Nat → Nat) : IO ByteArray := do
  let mut ba ← F32.const (rows * K).toUSize 0.0
  for r in [0:rows] do
    -- `write3` lays down three consecutive floats, so straddle the peak with its neighbours;
    -- that also proves the peak beats an ADJACENT entry, not just the far-away zeros.
    let idx := r * K + peakAt r
    ba ← F32.write3 ba (idx - 1).toUSize 0.0 5.0 0.0
  return ba

def main : IO Unit := do
  let K := 1000                     -- ImageNet-1k, the tier a 10-class constant is wrong on
  let peak : Nat → Nat := fun r => if r == 0 then 700 else 3
  let ba ← mkLogits 2 K peak
  let mut bad := 0

  -- ══ Gate A: the class count is honoured — a winner past index 10 is FOUND ══
  let a0 := (F32.argmaxN ba 0 K.toUSize).toNat
  let a1 := (F32.argmaxN ba K.toUSize K.toUSize).toNat
  if a0 == 700 then
    IO.println s!"  ✅ A1 argmaxN over {K}: found the peak at 700"
  else
    IO.println s!"  ❌ A1 argmaxN over {K}: got {a0}, want 700"; bad := bad + 1
  -- The second row also pins the OFFSET: a row whose peak is inside a 10-wide window still has
  -- to be read at its own base, not row 0's.
  if a1 == 3 then
    IO.println s!"  ✅ A2 argmaxN at row offset {K}: found the peak at 3"
  else
    IO.println s!"  ❌ A2 argmaxN at row offset {K}: got {a1}, want 3"; bad := bad + 1

  -- ══ Gate B (CONTROL): the OLD window, re-run on the same data, must MISS ══
  -- If this ever passes, `n` has stopped being read and the defect is back.
  let b0 := (F32.argmaxN ba 0 10).toNat
  if b0 != 700 then
    IO.println s!"  ✅ B  control: a 10-wide window MISSES the peak (returns {b0}) — the gate is live"
  else
    IO.println s!"  ❌ B  control did NOT fire: a 10-wide window returned 700, so this file proves nothing"
    bad := bad + 1

  -- ══ Gates C/D: `rankOf`, the top-5 metric ══
  -- The reference's headline is quoted as "72.02% top-1 / 90.62% top-5". `rankOf` counts strictly-greater logits, so the label is in
  -- the top-k iff rank < k — the same construction the reference uses (it avoids `top_k`, whose
  -- indices it records as broken on ROCm/gfx1100). Matching the formulation makes the two sides'
  -- top-5 comparable by construction, ties and all.
  let mut ranked ← F32.const K.toUSize 0.0
  -- six descending values at known indices; everything else is 0
  let places := #[(100, 6.0), (200, 5.0), (300, 4.0), (400, 3.0), (500, 2.0), (600, 1.0)]
  for (idx, v) in places do
    ranked ← F32.write3 ranked (idx - 1).toUSize 0.0 v 0.0
  let expect : Array (Nat × Nat) := #[(100, 0), (200, 1), (500, 4), (600, 5), (700, 6)]
  let mut rankBad : Array (Nat × Nat × Nat) := #[]
  for (lbl, want) in expect do
    let got := (F32.rankOf ranked 0 K.toUSize lbl.toUSize).toNat
    if got != want then rankBad := rankBad.push (lbl, want, got)
  if rankBad.isEmpty then
    IO.println s!"  ✅ C  rankOf: {expect.size} labels rank exactly as constructed (0,1,4,5,6)"
  else
    IO.println s!"  ❌ C  rankOf wrong (label, want, got): {rankBad}"; bad := bad + 1

  -- THE BOUNDARY, which is the whole gate: rank 4 is in the top-5 and rank 5 is NOT.
  -- An off-by-one here (`<=` for `<`) would silently inflate every top-5 number ever reported.
  let in5 (lbl : Nat) : Bool := (F32.rankOf ranked 0 K.toUSize lbl.toUSize).toNat < 5
  if in5 500 && !(in5 600) then
    IO.println "  ✅ D  top-5 boundary: rank 4 is IN, rank 5 is OUT"
  else
    IO.println s!"  ❌ D  top-5 boundary wrong: rank4 in5={in5 500}, rank5 in5={in5 600}"
    bad := bad + 1

  if bad == 0 then
    IO.println "argmax-check: OK (4 gates + 1 control)"
  else
    IO.println s!"argmax-check: {bad} FAILED"
    throw <| IO.userError "argmax-check failed"
