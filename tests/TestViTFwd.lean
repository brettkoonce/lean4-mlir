import LeanMlir.Proofs.Codegen.ViTRender
import LeanMlir.ViTRender
import LeanMlir.Types

/-! # ch10 V6b — ViT-Tiny forward (eval): iree-compile smoke on the COMMITTED render

**This file no longer writes `verified_mlir/vit_fwd.mlir`.** Its only writer is the `#eval` in
`LeanMlir/Proofs/Codegen/ViTRender.lean`, which renders `Proofs.StableHLO.vitFwdRenderV "vit_fwd"`
— the certified forward renderer, eval peer of `vitTrainStepRenderV`, same param order as
`ViTLayout` (1D CLS `tensor<192>`).

Until 2026-07-28 this file re-rendered the artifact with *identical* arguments, so it was a second
writer producing the same bytes. Verified byte-identical (md5 `626cc192…` unchanged across a run of
both writers) and then retired: a redundant writer costs nothing until someone edits one of the two,
at which point it is a silent last-writer-wins race — which is what happened to `resnet34_train_step`
(§2a) and `resnet34_adam_train_step` (§2b-ter). Being *currently* identical is not a property that
maintains itself.

What remains is the part the `Proofs/` `#eval` genuinely cannot do: iree-compile the committed bytes
on the rocm box, which needs the compiler on PATH and so must stay out of `lake build`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestViTFwd.lean
-/

private def main : IO Unit := do
  let path := "verified_mlir/vit_fwd.mlir"
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/ViTRender.lean; run `lake build LeanMlir.Proofs.Codegen.ViTRender` first")
  IO.println s!"iree-compile smoke on the COMMITTED {path} (this file does not re-render it)"
  let cargs ← ireeCompileArgs path ".lake/build/vit_fwd_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[vit_fwd] iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println s!"ViT-Tiny forward iree-compile OK → {path}"

#eval main
