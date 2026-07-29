import LeanMlir.VerifiedNets

/-! # Shared body of the verified ViT-Tiny + AdamW Imagenette trainer

`vit-verified-adam` (IREE) and `vit-verified-adam-xla` (XLA/PJRT) are the same program — same
`VerifiedNetSpec`, same `verified_mlir/vit_<variant>_train_step.mlir`, same schedule and He-init
seed. Only the linked trusted lowerer differs.

Lake requires a distinct root module per executable, so the config and entry point live here rather
than being duplicated; drift in `epochs`, `batchSize`, the seed, or any AdamW hyperparameter would
quietly invalidate any cross-backend comparison. Same reason as `Resnet34AdamCommon.lean` and
`EfficientNetAdamCommon.lean`.

⛔ **The XLA binary does not run on this box** — see `MainViTVerifiedAdamXla.lean`. The plumbing is
here and correct; the graph dies in MIOpen.
-/

-- Matches MainVitTrain.lean's `vitTinyConfig` (the reference): 80 epochs, bs 32,
-- AdamW lr 3e-4 / wd 1e-4, cosine + 5-epoch warmup, label smoothing 0.1, augment.
-- (vitTinyConfig sets NO EMA and gradClipNorm 0.0, so the verified path omits them too.)
def vitAdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

/-- Entry point for both backends. baseLR 3e-4, β₁ .9, β₂ .999, 5-epoch linear warmup then cosine
    decay (`vitTinyConfig`) — note ViT's schedule differs from the other four nets' 1e-3 / 3 epochs.

    `LEAN_MLIR_VARIANT` selects the rendered train step, i.e. which
    `verified_mlir/vit_<variant>_train_step.mlir` is loaded (and with it a distinct vmfb and
    checkpoint). Two exist, and **both are `pretty(provenGraph)` out of
    `Proofs/Codegen/ViTRender.lean`**:

    * **`adam`** (default) — the certified single-device render, licensed by `vit-adam-tie`:
      gradient norm-rel 1e-6, `%loss` bit-exact, 0/200 params disagreeing, on all 16,579,041
      returned floats (handoff §2a).
    * **`adamdp`** — the DATA-PARALLEL render: the same graph plus one `all_reduce(add)/N` per
      parameter gradient between the certified gradient and the certified AdamW triple. That
      collective is a **declared trusted carve-out** (§5) and the render says so in its own output
      banner. ⛔ It is **numerically ungated** — `vit-dp-check` is written and will pass the moment
      the graph runs, but the graph does not execute on this box. Do not describe ViT multi-GPU as
      working. Pair it with `LEAN_MLIR_REPLICAS=N PJRT_REPLICAS=N` and `HIP_VISIBLE_DEVICES` unset.

    This driver no longer WRITES its artifact. Until 2026-07-28 it re-emitted the hand-written
    `ViTRender.vitTrainStepModuleAdamSched` on every startup, which meant the committed bytes were
    never authoritative and the writer audit could not see the writer at all. The artifact now comes
    from `Proofs/Codegen/ViTRender.lean`'s `vitAdamTrainStepFaithful` and that `#eval` is its sole
    writer, so a missing artifact **throws** rather than being quietly recreated. -/
def runViTAdam (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  let tsPath := s!"verified_mlir/vit_{variant}_train_step.mlir"
  if !(← System.FilePath.pathExists tsPath) then
    throw (IO.userError s!"{tsPath} missing — it is written by \
LeanMlir/Proofs/Codegen/ViTRender.lean; run `lake build LeanMlir.Proofs.Codegen.ViTRender` first")
  vitVerified.toNet.trainAdamSched vitAdamConfig (argv.head?.getD "data") 0.0003 0.9 0.999 5 variant
