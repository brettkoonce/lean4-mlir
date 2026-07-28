import LeanMlir.Types

/-! # MobileNetV2 SGD train step: iree-compile smoke on the COMMITTED render

**This file no longer writes `verified_mlir/mobilenetv2_train_step.mlir`.** Its only writer is the
`#eval` in `LeanMlir/Proofs/Codegen/MobileNetV2Render.lean`:
`mnv2TrainStepFaithfulVPaper 32 10 "1.0e-5" "0.3"` — `pretty(provenGraph)` at the **full paper
[t,c,n,s] config**, 210 params. The bytes `mobilenetv2-verified` trains on ARE that render.

Until 2026-07-28 this file *also* rendered that artifact, and the two writers did not even have the
same **arity**: the emitter here was the **reduced 6-block** net (head 64→128, 84 in / 82 out)
against the committed full config's 212/210. They are not two spellings of one function, so no
numeric tie was possible or meaningful — a clobber here replaced the full network with a smaller
one under the full one's canonical filename. That failure is at least *loud* (the driver refuses the
wrong arity), which is exactly why it must not set expectations for convnext/efficientnet/resnet34,
where the same clobber produces a runnable graph computing something else.

The reduced render has its own home and needs no writer here:
`Proofs/Codegen/MobileNetV2Render.lean` also emits `verified_mlir/mobilenetv2_reduced_train_step.mlir`
via `mnv2TrainStepFaithfulV`. Recover the hand-written emitter from
`git show c992a94:tests/TestMobilenetV2Train.lean` if it is ever wanted.

Retired under §2a-quinquies. The mnv2 **AdamW** render is untouched and lives in a different file,
`tests/TestMobilenetV2TrainPC.lean`.

What remains is the part the `Proofs/` `#eval` cannot do: iree-compile the committed bytes on the
rocm box, which needs the compiler on PATH and so must stay out of `lake build`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestMobilenetV2Train.lean
-/

private def main : IO Unit := do
  let path := "verified_mlir/mobilenetv2_train_step.mlir"
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/MobileNetV2Render.lean; run \
`lake build LeanMlir.Proofs.Codegen.MobileNetV2Render` first")
  IO.FS.createDirAll ".lake/build"
  IO.println s!"iree-compile smoke on the COMMITTED {path} (this file does not re-render it)"
  let cargs ← ireeCompileArgs path ".lake/build/mobilenetv2_train_step_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println s!"MobileNetV2 FULL SGD train step iree-compile OK → {path}"

#eval main
