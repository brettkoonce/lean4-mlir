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

-- Matches MainVitTrain.lean's `vitTinyConfig` (the reference): 80 epochs, bs 32,
-- AdamW lr 3e-4 / wd 1e-4, cosine + 5-epoch warmup, label smoothing 0.1, augment.
-- (vitTinyConfig sets NO EMA and gradClipNorm 0.0, so the verified path omits them too.)
def vitAdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

def main (argv : List String) : IO Unit := do
  -- This driver no longer WRITES `verified_mlir/vit_adam_train_step.mlir`. Until 2026-07-28 it
  -- re-emitted the hand-written `ViTRender.vitTrainStepModuleAdamSched` here on every startup,
  -- which meant the committed bytes were never authoritative and the artifact writer audit could
  -- not see the writer at all. The artifact now comes from `Proofs/Codegen/ViTRender.lean`'s
  -- `vitAdamTrainStepFaithful` — `pretty(provenGraph)`, gradients + AdamW triple both proven —
  -- and that `#eval` is its sole writer. Licensed by `lake build vit-adam-tie`: gradient norm-rel
  -- 1e-6, %loss bit-exact, 0/200 params disagreeing, on all 16,579,041 returned floats.
  -- Fail loudly if the artifact is missing rather than quietly recreating it.
  let tsPath := "verified_mlir/vit_adam_train_step.mlir"
  if !(← System.FilePath.pathExists tsPath) then
    throw (IO.userError s!"{tsPath} missing — it is written by \
LeanMlir/Proofs/Codegen/ViTRender.lean; run `lake build LeanMlir.Proofs.Codegen.ViTRender` first")
  -- baseLR 3e-4, β₁ .9, β₂ .999, 5-epoch linear warmup then cosine decay (vitTinyConfig).
  vitVerified.toNet.trainAdamSched vitAdamConfig (argv.head?.getD "data") 0.0003 0.9 0.999 5
