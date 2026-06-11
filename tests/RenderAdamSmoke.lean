import LeanMlir.ViTRender

/-! Scratch renderer for the Adam render smoke (Phase 3b). Writes:
  * `/tmp/adam/adam_step.mlir` — a tiny 2-param `@adam_step` (a [4,4] + a [4]) for
    an exact numeric faithfulness check vs `Proofs.adamWParam` on the GPU.
  * `/tmp/adam/vit_train_step_adam.mlir` — the full depth-12 ViT-Tiny AdamW step.
Run: `lake env lean tests/RenderAdamSmoke.lean` (after `lake build LeanMlir.ViTRender`). -/

open ViTRender Proofs.StableHLO

def adamTinyModule : String :=
  let (s1, w', _, _) := emitAdamV "%w" "%dw" "%mw" "%vw" [4, 4] "w"
  let (s2, b', _, _) := emitAdamV "%b" "%db" "%mb" "%vb" [4] "b"
  "module @m {\n" ++
  "  func.func @adam_step(%w: tensor<4x4xf32>, %dw: tensor<4x4xf32>, %mw: tensor<4x4xf32>, %vw: tensor<4x4xf32>, %b: tensor<4xf32>, %db: tensor<4xf32>, %mb: tensor<4xf32>, %vb: tensor<4xf32>, %b1: tensor<f32>, %ob1: tensor<f32>, %b2: tensor<f32>, %ob2: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %lr: tensor<f32>, %eps: tensor<f32>, %wd: tensor<f32>) -> (tensor<4x4xf32>, tensor<4xf32>) {\n" ++
  s1 ++ s2 ++
  s!"    return {w'}, {b'} : tensor<4x4xf32>, tensor<4xf32>\n" ++
  "  }\n}\n"

def main : IO Unit := do
  IO.FS.createDirAll "/tmp/adam"
  IO.FS.writeFile "/tmp/adam/adam_step.mlir" adamTinyModule
  let cfg := vitTinyConfig 32 12
  IO.FS.writeFile "/tmp/adam/vit_train_step_adam.mlir" (vitTrainStepModuleAdam cfg (vitTinyBlocks 12))
  IO.println "wrote /tmp/adam/adam_step.mlir + /tmp/adam/vit_train_step_adam.mlir"

#eval main
