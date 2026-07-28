import LeanMlir.Types

/-! # ResNet-34 SGD train step: iree-compile smoke on the COMMITTED render

**This file no longer writes `verified_mlir/resnet34_train_step.mlir`.** Its only writer is the
`#eval` in `LeanMlir/Proofs/Codegen/ResNet34Render.lean`:
`resnet34TrainStepFaithfulV 32 10 "1.0e-05" "0.003125"` — `pretty(provenGraph)`, **per-example** BN
(reduce `[2,3]`, n = H·W = 12544), lr 0.003125 = 0.1/32 (the mean folded into lr — the r34
convention, shared with `vit_train_step`). The bytes `resnet34-verified` trains on ARE that render.

Until 2026-07-28 this file *also* rendered that artifact, from an independent hand-written emitter,
and the two were **not two spellings of one function**: the `tests/` emitter used **batch** BN
(reduce `[0,2,3]`, n = B·H·W = 401408). On one shared (θ, x) the two forwards disagree at rel 1.13
on non-degenerate logits (handoff §2a), and whichever writer ran last decided what the trainer
optimised — observed flipping md5 `3184522f` ↔ `929074f6` merely by elaborating this file. That is
the whole hazard: a clobber here produces a *runnable* graph computing something else, silently.

Retired under §2a-quinquies. The AdamW half went earlier (§2b-ter/§2b-quater — both the
single-device and the `adamdp` data-parallel render now come from
`Proofs/Codegen/ResNet34RenderB.lean`, its only writer). Recover the emitter from
`git show c992a94:tests/TestResnet34Train.lean` if it is ever wanted; note the batch-BN semantics it
carries, which are the AdamW path's, not this one's.

What remains is the part the `Proofs/` `#eval` cannot do: iree-compile the committed bytes on the
rocm box, which needs the compiler on PATH and so must stay out of `lake build`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestResnet34Train.lean
-/

private def main : IO Unit := do
  let path := "verified_mlir/resnet34_train_step.mlir"
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/ResNet34Render.lean; run `lake build LeanMlir.Proofs.Codegen.ResNet34Render` \
first")
  IO.FS.createDirAll ".lake/build"
  IO.println s!"iree-compile smoke on the COMMITTED {path} (this file does not re-render it)"
  let cargs ← ireeCompileArgs path ".lake/build/resnet34_train_step_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println s!"ResNet-34 FULL SGD train step iree-compile OK → {path}"

#eval main
