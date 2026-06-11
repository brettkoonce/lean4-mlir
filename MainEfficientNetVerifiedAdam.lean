import LeanMlir.VerifiedNets

/-! # `efficientnet-verified-adam` — train EfficientNet-B0 with the VERIFIED-rendered **AdamW** step

The enet peer of `vit`/`mnv2-verified-adam`: the proof-rendered EfficientNet-B0 train step
(`tests/TestEfficientNetTrain.lean → verified_mlir/efficientnet_adam_train_step.mlir`,
`@efficientnet_adam_train_step`) — all-swish + squeeze-excite + per-channel batch-norm — with the
SGD update swapped for AdamW via `ViTRender.emitAdamV`, driven by the generic
`VerifiedNet.trainAdamSched`: `[θ|m|v]` (262 params) packed as one blob + runtime `lr`/`bc₁`/`bc₂`
scalars (cosine + warmup + per-step bias correction) through the unchanged FFI (`n_params = 3k`).

Recipe matches `efficientnet-train` (`MainEfficientNetTrain.lean`'s `efficientNetB0Config`): AdamW
lr 1e-3 / wd 1e-4, cosine + 3-epoch warmup, label smoothing 0.1, augment, 80 epochs, bs 32.
**Loss-curve-first parity**: both this and the reference batch-norm in train mode, so the train-loss
curve tracks; eval here uses the eval-batch's own BN stats (running-stats BN deferred), so val-acc
is close-not-exact. Weight decay uniform (incl. BN/bias), matching the ViT/mnv2 verified paths.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/efficientnet-verified-adam data` (loader reads
`data/imagenette`).
-/

-- Matches MainEfficientNetTrain.lean's `efficientNetB0Config`: 80 epochs, bs 32, AdamW lr 1e-3 /
-- wd 1e-4, cosine + 3-epoch warmup, label smoothing 0.1, augment.
def efficientnetAdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

def main (argv : List String) : IO Unit :=
  -- baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine decay (efficientNetB0Config).
  efficientnetVerified.toNet.trainAdamSched efficientnetAdamConfig
    (argv.head?.getD "data") 0.001 0.9 0.999 3
