import LeanMlir.VerifiedNets
import LeanMlir.ViTRender

/-! # `vit-verified-adam` — train ViT-Tiny with the VERIFIED-rendered **AdamW** step

Phase 3c of `planning/vit_train_to_vit_verified.md`: the SGD `vit-verified` with its
optimizer swapped for AdamW. The packed train step `@vit_adam_train_step`
(`ViTRender.vitTrainStepModuleAdamPacked`, ℝ spec `Proofs.adamWParam`, the
GPU-validated render) is emitted here with the hyperparameters baked, then driven by
`VerifiedNet.trainAdamPacked` — which threads `[θ|m|v]` as a single packed param blob
through the generic FFI (`n_params = 3k`; the moments ride in the params slot, so the
prebuilt `.so` is unchanged). Moments start at 0; bias correction is omitted (a later
rung host-passes `1−βᵗ`).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/vit-verified-adam data`
-/

def vitAdamConfig : VerifiedConfig where
  epochs    := 20
  batchSize := 32

def main (argv : List String) : IO Unit := do
  -- Emit the scheduled AdamW train step (β₁ .9, β₂ .999, ε 1e-8, wd 1e-4 baked;
  -- lr/bc₁/bc₂ are runtime scalar params the driver schedules).
  let cfg := ViTRender.vitTinyConfig 32 12
  let mlir := ViTRender.vitTrainStepModuleAdamSched cfg (ViTRender.vitTinyBlocks 12)
                "0.9" "0.1" "0.999" "0.001" "1.0e-8" "0.0001" 0.1   -- label smoothing α=0.1
  IO.FS.writeFile "verified_mlir/vit_adam_train_step.mlir" mlir
  -- baseLR 1e-3, β₁ .9, β₂ .999, 5-epoch linear warmup then cosine decay.
  vitVerified.toNet.trainAdamSched vitAdamConfig (argv.head?.getD "data") 0.001 0.9 0.999 5
