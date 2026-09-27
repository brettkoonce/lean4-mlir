import LeanMlir.Proofs.Codegen.ViTRender
import tests.ViTRender
import LeanMlir.Types

/-! # ch10 V6b — ViT-Tiny forward (eval): iree-compile smoke on the COMMITTED render

**This file does not write `verified_mlir/vit_fwd.mlir`.** Its only writer is the `#eval` in
`LeanMlir/Proofs/Codegen/ViTRender.lean`, which renders `Proofs.StableHLO.vitFwdRenderV "vit_fwd"`
— the certified forward renderer, eval peer of `vitTrainStepRenderV`, same param order as
`ViTLayout` (1D CLS `tensor<192>`).

A second writer producing the same bytes costs nothing until someone edits one of the two, at
which point it is a silent last-writer-wins race. Being *currently* identical is not a property
that maintains itself.

What remains is the part the `Proofs/` `#eval` genuinely cannot do: iree-compile the committed bytes,
which needs the compiler on PATH and so must stay out of `lake build`.

Run (needs iree-compile on PATH):
  lake env lean tests/TestViTFwd.lean
-/

private def main : IO Unit := do
  let path := "verified_mlir/vit_fwd.mlir"
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/ViTRender.lean; run `lake build LeanMlir.Proofs.Codegen.ViTRender` first")
  IO.println s!"iree-compile smoke on the COMMITTED {path} (this file does not re-render it)"
  tryCompile path ".lake/build/vit_fwd_v.vmfb" "ViT-Tiny forward"

#eval main
