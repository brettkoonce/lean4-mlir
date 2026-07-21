import LeanMlir.Proofs.Codegen.ViTRender
import LeanMlir.ViTRender
import LeanMlir.Types

/-! # ch10 V6b — ViT-Tiny forward (eval), the PROOF-TIED render (Imagenette 224², depth-12)

Writes `verified_mlir/vit_fwd.mlir` from the **certified** forward renderer
`Proofs.StableHLO.vitFwdRenderV` (the eval peer of `vitTrainStepRenderV`, same param order
as `ViTLayout`), then iree-compiles to ROCm. Swapped from the hand renderer
(`ViTRender.vitFwdModule`) so eval reads the certified bytes too — 1D CLS `tensor<192>`,
matching the train step + `ViTLayout`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestViTFwd.lean
-/

private def main : IO Unit := do
  IO.FS.createDirAll "verified_mlir"
  let fwd := Proofs.StableHLO.vitFwdRenderV "vit_fwd"
  IO.FS.writeFile "verified_mlir/vit_fwd.mlir" fwd
  IO.println s!"rendered ViT-Tiny forward (proof-tied, BS=32, depth 12): {fwd.length} chars → verified_mlir/vit_fwd.mlir"
  let cargs ← ireeCompileArgs "verified_mlir/vit_fwd.mlir" ".lake/build/vit_fwd_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[vit_fwd] iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "ViT-Tiny forward iree-compile OK → verified_mlir/vit_fwd.mlir"

#eval main
