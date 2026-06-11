import LeanMlir.VerifiedNets

/-! # `resnet34-verified-adam` — train ResNet-34 with the VERIFIED-rendered **AdamW** step

The ResNet-34 peer of the `vit`/`mnv2`/`enet` verified-adam trainers: the proof-rendered train step
(`tests/TestResnet34Train.lean → verified_mlir/resnet34_adam_train_step.mlir`,
`@resnet34_adam_train_step`) — 7×7-s2 stem → maxpool → [3,4,6,3] basic blocks (per-channel BN +
strided downsamples) → GAP → dense — with the SGD update swapped for AdamW via
`ViTRender.emitAdamV`, driven by the generic `VerifiedNet.trainAdamSched`: `[θ|m|v]` (146 params)
packed as one blob + runtime `lr`/`bc₁`/`bc₂` scalars (cosine + warmup + per-step bias correction)
through the unchanged FFI (`n_params = 3k`).

Recipe matches the reference (`MainResnetTrain.lean`'s `resnet34Config`): AdamW lr 1e-3 / wd 1e-4,
cosine + 3-epoch warmup, label smoothing 0.1, augment, 80 epochs, bs 32. **Loss-curve-first
parity**: both batch-norm in train mode, so the train-loss curve tracks; eval here uses the
eval-batch's own BN stats (running-stats BN deferred), so val-acc is close-not-exact. Weight decay
uniform (incl. BN/bias), matching the other verified paths.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/resnet34-verified-adam data` (loader reads
`data/imagenette`).
-/

-- Matches MainResnetTrain.lean's `resnet34Config`: 80 epochs, bs 32, AdamW lr 1e-3 / wd 1e-4,
-- cosine + 3-epoch warmup, label smoothing 0.1, augment.
def resnet34AdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

def main (argv : List String) : IO Unit :=
  -- baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine decay (resnet34Config).
  resnet34Verified.toNet.trainAdamSched resnet34AdamConfig
    (argv.head?.getD "data") 0.001 0.9 0.999 3
