import LeanMlir.Proofs.Codegen.ViTRender
import LeanMlir.ViTRender
import LeanMlir.Types

/-! # ch10 V6b — ViT-Tiny train step, the PROOF-TIED render (Imagenette 224², depth-12)

Writes `verified_mlir/vit_train_step.mlir` from the **certified** renderer
`Proofs.StableHLO.vitTrainStepRenderV` — the render `Proofs.ViTTiePoC.vit_net_tied_certified`
is about (every param-SGD op `den`otes the certified loss-descent step; whole module =
`pretty(provenGraph)`). Then iree-compiles to ROCm.

Swapped from the hand renderer (`ViTRender.vitTrainStepModule`) so the committed bytes
`MainViTVerified` trains on ARE the certified render. 1D CLS `tensor<192>` (matching the
proof's `cls : Vec 192` + `ViTLayout`); lr 0.003125 = 0.1/32 (mean folded into lr, r34
convention — the same effective update as the old hand render's lr=0.1 + mean loss cot).
200 params, BS=32.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestViTTrain.lean
-/

private def main : IO Unit := do
  IO.FS.createDirAll "verified_mlir"
  let ts := Proofs.StableHLO.vitTrainStepRenderV "vit_train_step" "0.003125"
  IO.FS.writeFile "verified_mlir/vit_train_step.mlir" ts
  IO.println s!"rendered ViT-Tiny train step (proof-tied, BS=32, depth 12, 200 params): {ts.length} chars → verified_mlir/vit_train_step.mlir"
  let cargs ← ireeCompileArgs "verified_mlir/vit_train_step.mlir" ".lake/build/vit_train_step_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[depth-12] iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "ViT-Tiny FULL train step iree-compile OK → verified_mlir/vit_train_step.mlir"

#eval main
