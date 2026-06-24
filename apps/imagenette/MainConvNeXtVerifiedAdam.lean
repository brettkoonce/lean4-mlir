import LeanMlir.VerifiedNets

/-! # `convnext-verified-adam` — train ConvNeXt-T with the VERIFIED-rendered **AdamW** step

The ConvNeXt-T peer of the `vit`/`mnv2`/`enet`/`r34` verified-adam trainers: the proof-rendered
train step (`tests/TestConvNeXtTrain.lean → verified_mlir/convnext_adam_train_step.mlir`,
`@convnext_adam_train_step`) — patchify stem → [3,3,9,3] depthwise-7×7 blocks (LayerNorm + GELU +
layerScale) + 3 between-stage downsamples → GAP → LN → dense — with the SGD update swapped for
AdamW via `ViTRender.emitAdamV`, driven by the generic `VerifiedNet.trainAdamSched`: `[θ|m|v]`
(180 params) packed as one blob + runtime `lr`/`bc₁`/`bc₂` scalars through the unchanged FFI.

ConvNeXt is all-smooth (LayerNorm, not BN), so there's no running-stats / train-vs-eval BN gap —
this is **exact-parity** territory like ViT (eval matches train). Recipe matches the reference
(`MainConvNeXtTrain.lean`'s `convNextTinyConfig`): AdamW lr 1e-3 / wd 1e-4, cosine + 3-epoch
warmup, label smoothing 0.1, augment, 80 epochs, bs 32. Weight decay uniform (incl. LN/bias),
matching the other verified paths.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/convnext-verified-adam data` (loader reads
`data/imagenette`).
-/

-- Matches MainConvNeXtTrain.lean's `convNextTinyConfig`: 80 epochs, bs 32, AdamW lr 1e-3 / wd 1e-4,
-- cosine + 3-epoch warmup, label smoothing 0.1, augment.
def convnextAdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

def main (argv : List String) : IO Unit :=
  -- baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine decay (convNextTinyConfig).
  convnextVerified.toNet.trainAdamSched convnextAdamConfig
    (argv.head?.getD "data") 0.001 0.9 0.999 3
