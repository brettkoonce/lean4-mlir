import LeanMlir.Proofs.Codegen.ViTRender
import LeanMlir.ViTRender
import LeanMlir.Types

/-! # ch10 V6b — ViT-Tiny train step: iree-compile smoke on the COMMITTED render

**This file no longer writes `verified_mlir/vit_train_step.mlir`.** Its only writer is the `#eval`
in `LeanMlir/Proofs/Codegen/ViTRender.lean`:
`vitTrainStepRenderV "vit_train_step" "0.003125"` — the render
`Proofs.ViTTiePoC.vit_net_tied_certified` is about (every param-SGD op `den`otes the certified
loss-descent step; whole module = `pretty(provenGraph)`). 1D CLS `tensor<192>` matching the proof's
`cls : Vec 192` + `ViTLayout`; lr 0.003125 = 0.1/32 (mean folded into lr, r34 convention).
200 params, BS=32. The bytes `MainViTVerified` trains on ARE that render.

Until 2026-07-28 this file re-rendered it with *identical* arguments — a second writer producing the
same bytes. Verified byte-identical (md5 `f57aff00…` unchanged across a run of both writers) and
then retired: a redundant writer costs nothing until someone edits one of the two, at which point it
is a silent last-writer-wins race (§2a, §2b-ter). Being *currently* identical is not a property that
maintains itself.

What remains is the part the `Proofs/` `#eval` cannot do: iree-compile the committed bytes on the
rocm box, which needs the compiler on PATH and so must stay out of `lake build`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestViTTrain.lean
-/

private def main : IO Unit := do
  let path := "verified_mlir/vit_train_step.mlir"
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/ViTRender.lean; run `lake build LeanMlir.Proofs.Codegen.ViTRender` first")
  IO.println s!"iree-compile smoke on the COMMITTED {path} (this file does not re-render it)"
  let cargs ← ireeCompileArgs path ".lake/build/vit_train_step_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[depth-12] iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println s!"ViT-Tiny FULL train step iree-compile OK → {path}"

#eval main
