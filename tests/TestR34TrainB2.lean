import LeanMlir
import LeanMlir.Proofs.Codegen.ResNet34RenderB

/-! # ResNet-34 AdamW train step at **batch 2** — the artifact `scripts/grad_tie.py` runs

The committed `resnet34_adam_train_step.mlir` is B=32, which is more than a CPU gradient check
wants to carry. This is the same renderer at `B := 2`, written to `.lake/build/` because it is a
test input rather than part of the certified corpus.

⚠ **Sole writer of this path**, deliberately. Two `#eval`s writing one artifact is the
last-writer-wins race the repo has been bitten by (§2a), and a *test* artifact is no less exposed
to it than a committed one — arguably more, since nothing diffs it.

## ⚠⚠ Why there is no forward-prefix check here, unlike `mnv4-train-smoke`

MNv4 asserts that its train step contains `@mnv4_fwd`'s body verbatim. **ResNet-34 cannot make that
claim and this test must not pretend otherwise.** Measured on the committed artifacts:

```
resnet34_fwd              reduce[0,2,3] = 0     reduce[2,3] = 73    <- PER-EXAMPLE BN
resnet34_adam_train_step  reduce[0,2,3] = 468   reduce[2,3] = 1     <- BATCH BN
```

They are different functions. That is the §3d(b) two-worlds split (`planning/mnv4_verified.md`),
live on R34, and `scripts/regen_verified_mlir.sh check` goes green anyway because it only ever
pairs a forward with the **SGD** train step — which shares the per-example world. So the artifact
that trains every quoted R34 number has never been audited against the forward that scores it.

▶ This test PINS that split as a measured fact rather than leaving it implicit. If it ever stops
holding, someone has changed a BN world and needs to know which.
-/

open Proofs.StableHLO

def main : IO Unit := do
  let B := 2
  let nClasses := 10
  let m := resnet34AdamTrainStepFaithfulB B nClasses "1.0e-05"
  IO.FS.writeFile ".lake/build/resnet34_adam_train_step_b2.mlir" m
  IO.println s!"  wrote .lake/build/resnet34_adam_train_step_b2.mlir ({(m.splitOn "\n").length} lines)"
  let mut bad := 0
  let chk (what : String) (got want : Nat) : IO Bool := do
    if got == want then IO.println s!"  ✓ {what}: {got}"; pure true
    else IO.println s!"  ✗ {what}: got {got}, want {want}"; pure false

  -- ── the interface contract: 110 params × (θ, m, v) + lr/bc1/bc2 + 72 stat slots + %x + %onehot ──
  let hdr := (m.splitOn "func.func @").getD 1 ""
  let argSig := (hdr.splitOn ") -> (").getD 0 ""
  let nIn := (argSig.splitOn ": tensor").length - 1
  if !(← chk "func inputs" nIn (1 + 3*110 + 3 + 72 + 1)) then bad := bad + 1
  let entry := s!"@resnet34_{r34AdamVariant B 1}_train_step("
  if (m.splitOn entry).length > 1 then
    IO.println s!"  ✓ entry point {entry.dropRight 1}"
  else
    IO.println s!"  ✗ entry point missing (expected {entry.dropRight 1})"; bad := bad + 1

  -- ── ⚠ THE TWO-WORLDS SPLIT, pinned. See the module docstring. ──
  -- This render is BATCH BN; `verified_mlir/resnet34_fwd.mlir` is per-example. Recording it here
  -- means a change to either world fails a test instead of silently re-pairing the nets.
  let nBatch := ((m.splitOn "dimensions = [0, 2, 3]").length - 1)
  let nPerEx := ((m.splitOn "dimensions = [2, 3]").length - 1)
  IO.println s!"  · this train step: batch reductions {nBatch}, per-example {nPerEx} (the 1 is the head GAP)"
  if nBatch == 0 then
    IO.println "  ✗ expected BATCH BN in the Adam train step"; bad := bad + 1
  if (← IO.FS.lines "verified_mlir/resnet34_fwd.mlir").any
      (fun l => (l.splitOn "dimensions = [0, 2, 3]").length > 1) then
    IO.println "  ⚠ resnet34_fwd is now BATCH BN too — the two-worlds split has been CLOSED."
    IO.println "    Good news, but `scripts/grad_tie.py --net r34` and this file's docstring both"
    IO.println "    assume the split; re-read them before trusting either."
  else
    IO.println "  · resnet34_fwd remains PER-EXAMPLE BN — the split is still open (§3d(b))"

  if bad == 0 then
    IO.println "  ✓ r34 b2 train step ready for scripts/grad_tie.py --net r34"
  else
    throw (IO.userError s!"r34 b2 emit FAILED ({bad})")
