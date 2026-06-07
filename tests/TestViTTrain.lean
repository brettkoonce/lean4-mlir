import LeanMlir.ViTRender
import LeanMlir.Types

/-! # ch10 V6b — production ViT-Tiny train-step renderer (Imagenette 224², depth-12)

Renders `@vit_train_step` (image→logits→softmax-CE→full backward→mean-loss-SGD update)
for ViT-Tiny at the paper-native 224² resolution: 16×16/s16 patch embed (3→192,
196 patches), CLS + positional embed (197 tokens), 12 pre-norm transformer blocks
(dim 192, 3 heads, MLP 768), per-channel `[192]` final LayerNorm, CLS-slice dense head.
BS=32, 200 params. Every op fragment is the StableHLO of a proven-faithful emitter
(row-softmax V1, batched MHSA V2/V3, per-channel LN, GELU, even-kernel patch conv);
the full assembly is gradcheck-validated at a tiny config in tests/TestViTTiny.lean.

Writes `verified_mlir/vit_train_step.mlir` and iree-compiles to ROCm. The depth-4
sanity compile (smaller, faster) runs first to surface any production-shape issue
before the full depth-12 compile.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestViTTrain.lean
-/

open ViTRender

private def BS : Nat := 32
private def LR : String := "0.1"

private def main : IO Unit := do
  IO.FS.createDirAll "verified_mlir"
  -- depth-4 sanity compile first (production shapes, fewer blocks → faster)
  let cfg4 := vitTinyConfig BS 4
  let ts4 := vitTrainStepModule cfg4 (vitTinyBlocks 4) LR
  IO.FS.writeFile ".lake/build/vit_train_step_d4.mlir" ts4
  IO.println s!"rendered depth-4 sanity train step: {ts4.length} chars"
  let cargs4 ← ireeCompileArgs ".lake/build/vit_train_step_d4.mlir" ".lake/build/vit_ts_d4.vmfb"
  let r4 ← IO.Process.output { cmd := "iree-compile", args := cargs4 }
  if r4.exitCode != 0 then
    IO.eprintln s!"[depth-4] iree-compile FAILED:\n{r4.stderr.take 3000}"; return
  IO.println "[depth-4] iree-compile OK"
  -- production depth-12 train step
  let cfg := vitTinyConfig BS 12
  let ts := vitTrainStepModule cfg (vitTinyBlocks 12) LR
  IO.FS.writeFile "verified_mlir/vit_train_step.mlir" ts
  IO.println s!"rendered ViT-Tiny train step (BS={BS}, depth 12, 200 params): {ts.length} chars → verified_mlir/vit_train_step.mlir"
  let cargs ← ireeCompileArgs "verified_mlir/vit_train_step.mlir" ".lake/build/vit_train_step_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"[depth-12] iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "ViT-Tiny FULL train step iree-compile OK → verified_mlir/vit_train_step.mlir"

#eval main
