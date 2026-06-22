import LeanMlir.VerifiedNets

/-! # `mobilenetv2-verified-adam` — train MobileNetV2 with the VERIFIED-rendered **AdamW** step

The mnv2 peer of `vit-verified-adam`: the proof-rendered MobileNetV2 train step
(`tests/TestMobilenetV2TrainPC.lean → verified_mlir/mobilenetv2_adam_train_step.mlir`,
`@mobilenetv2_adam_train_step`) with the SGD update swapped for AdamW via
`ViTRender.emitAdamV`, driven by the generic `VerifiedNet.trainAdamSched` — which threads
`[θ|m|v]` as one packed blob + the runtime `lr`/`bc₁`/`bc₂` scalars (cosine + warmup + per-step
bias correction) through the unchanged FFI (`n_params = 3k`).

Recipe matches `mobilenet-v2-train` (`MainMobilenetV2Train.lean`'s `mobilenetV2Config`): AdamW
lr 1e-3 / wd 1e-4, cosine + 3-epoch warmup, label smoothing 0.1, augment, 80 epochs, bs 32
(no EMA, no grad-clip). **Exact BN parity**: TRUE batch-norm (reduce `[0,2,3]`) in the train step
+ running-stats eval — `mobilenetv2Verified.bnChannels` (52 layers, full-paper 17-block net) is non-empty, so the generic
`trainAdamSched` threads per-layer EMA batch stats and evals through `@mobilenetv2_fwd_eval` (affine
BN with the running stats), class-batch-independent on the sorted val set (GPU-validated: epoch-1
loss 2.18, running-stats val_acc 32.9%). The BN is hand-emitted (not a proof token — see
`TestMobilenetV2TrainPC`). Weight decay is applied uniformly (incl. BN/bias), matching the ViT path.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mobilenetv2-verified-adam data` (loader reads
`data/imagenette`).
-/

-- Matches MainMobilenetV2Train.lean's `mobilenetV2Config`: 80 epochs, bs 32, AdamW lr 1e-3 /
-- wd 1e-4, cosine + 3-epoch warmup, label smoothing 0.1, augment.
def mobilenetv2AdamConfig : VerifiedConfig where
  epochs    := 80
  batchSize := 32

def main (argv : List String) : IO Unit :=
  -- baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch linear warmup then cosine decay (mobilenetV2Config).
  mobilenetv2Verified.toNet.trainAdamSched mobilenetv2AdamConfig
    (argv.head?.getD "data") 0.001 0.9 0.999 3
