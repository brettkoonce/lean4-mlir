import LeanMlir.ViTRender
import LeanMlir.Types

/-! # ch10 V6b — production ViT-Tiny forward renderer (eval, Imagenette 224², depth-12)

Renders `@vit_fwd` (image → logits) for the depth-12 ViT-Tiny — the eval peer of
`@vit_train_step` (tests/TestViTTrain.lean), SAME `vitFwd` fragments and param order
(`ViTLayout`). Writes `verified_mlir/vit_fwd.mlir` and iree-compiles to ROCm.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestViTFwd.lean
-/

open ViTRender

private def BS : Nat := 32

private def main : IO Unit := do
  IO.FS.createDirAll "verified_mlir"
  let cfg := vitTinyConfig BS 12
  let fwd := vitFwdModule cfg (vitTinyBlocks 12)
  IO.FS.writeFile "verified_mlir/vit_fwd.mlir" fwd
  IO.println s!"rendered ViT-Tiny forward (BS={BS}, depth 12): {fwd.length} chars → verified_mlir/vit_fwd.mlir"
  let cargs ← ireeCompileArgs "verified_mlir/vit_fwd.mlir" ".lake/build/vit_fwd_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[vit_fwd] iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "ViT-Tiny forward iree-compile OK → verified_mlir/vit_fwd.mlir"

#eval main
