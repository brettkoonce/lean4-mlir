import apps.imagenette.EfficientNetAdamCommon

/-! # `efficientnet-verified-adam` — train EfficientNet-B0 with the VERIFIED-rendered **AdamW** step

The enet peer of `vit`/`mnv2-verified-adam`: the proof-rendered EfficientNet-B0 train step
(`LeanMlir/Proofs/Codegen/EfficientNetRender.lean → verified_mlir/efficientnet_adam_train_step.mlir`,
`@efficientnet_adam_train_step`) — all-swish + squeeze-excite + per-channel batch-norm, with the
gradients un-fused and handed to the proven `adamMNextF`/`adamVNextF`/`adamWParamF` triple — driven
by the generic `VerifiedNet.trainAdamSched`: `[θ|m|v]` (213 params) packed as one blob + runtime
`lr`/`bc₁`/`bc₂` scalars (cosine + warmup + per-step bias correction) through the unchanged FFI
(`n_params = 3k`).

**The artifact is `pretty(provenGraph)` since 2026-07-28** (handoff §2e). It used to come from a
hand-written string emitter in `tests/TestEfficientNetTrain.lean`; the swap was licensed by
`efficientnet-adam-tie`, which found the two bit-exact on all 12,166,117 returned floats — forward
(the 98 BN batch statistics), `%loss`, and the gradient. The driver itself needed no change: it
resolves the path from the net slug.

Recipe matches `efficientnet-train` (`MainEfficientNetTrain.lean`'s `efficientNetB0Config`): AdamW
lr 1e-3 / wd 1e-4, cosine + 3-epoch warmup, label smoothing 0.1, augment, 80 epochs, bs 32.
**Exact BN parity**: true batch-norm (reduce `[0,2,3]`) in train + running-stats eval —
`efficientnetVerified.bnChannels` (49 layers) drives the generic `trainAdamSched` to thread
per-layer EMA batch stats and eval through `@efficientnet_fwd_eval` (class-batch-independent on
the sorted val set). Weight decay uniform (incl. BN/bias), matching the ViT/mnv2 verified paths.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/efficientnet-verified-adam data` (loader reads
`data/imagenette`). For multi-GPU use `efficientnet-verified-adam-xla` — collectives live only on
the PJRT path, and this binary's shim refuses the DP entry point rather than silently running
single-device.
-/

def main (argv : List String) : IO Unit := runEfficientNetAdam argv
