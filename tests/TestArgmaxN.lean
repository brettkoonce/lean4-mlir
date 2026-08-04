import LeanMlir.F32Array

/-! # `argmax-check` — the CLASS-COUNT gate on `F32.argmaxN`

The gate that would have caught the defect it exists for. `F32.argmaxN`'s predecessor,
`F32.argmax10`, had no `n` at all: its C loop was `for (i = 1; i < 10; i++)`, a literal.

⚠ **It was correct on every net this repo had ever gated.** Imagenette, CIFAR-10 and MNIST are
all 10-class, so the constant agreed with the net everywhere a test looked. It was wrong on
exactly the tier with no accuracy gate — the 1000-class ImageNet trainers — where it confined
every prediction to labels 0..9 and therefore scored a net that could only ever be right on the
~1/100 of validation images whose label fell in that window.

Measured on the R34/ImageNet 4-GPU run, 2026-08-04, before the fix: epoch 2 reported
**471/49920 = 0.94%**, which is 471 of the ~499 *reachable* images — i.e. **94% on the 10-way
sub-problem it was actually being asked**, while the true 1000-way number was ~4.4%. Training
was healthy the whole time; only the readout was wrong. The signature is the whole lesson: the
old one had nowhere to put the class count, so nothing could check it against the net.

Gate A pins the fix. Gate B is the CONTROL — it re-runs the old window on the same data and
requires it to MISS, so a regression cannot pass this file quietly.
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
  let K := 1000                     -- ImageNet-1k, the tier the old constant was wrong on
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
  -- The second row also pins the OFFSET: a row whose peak is inside the old 10-window still has
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

  if bad == 0 then
    IO.println "argmax-check: OK (2 gates + 1 control)"
  else
    IO.println s!"argmax-check: {bad} FAILED"
    throw <| IO.userError "argmax-check failed"
