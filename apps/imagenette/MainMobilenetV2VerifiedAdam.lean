import apps.imagenette.MobilenetV2AdamCommon

/-! # `mobilenetv2-verified-adam` — train MobileNetV2 with the VERIFIED-rendered **AdamW** step

The mnv2 peer of `vit`/`enet-verified-adam`: the proof-rendered MobileNetV2 train step
(`LeanMlir/Proofs/Codegen/MobileNetV2RenderB.lean → verified_mlir/mobilenetv2_adam_train_step.mlir`,
`@mobilenetv2_adam_train_step`) — 17-block inverted-residual net with relu6 + per-channel batch-norm,
with the gradients un-fused and handed to the proven `adamMNextF`/`adamVNextF`/`adamWParamF` triple —
driven by the generic `VerifiedNet.trainAdamSched`: `[θ|m|v]` (210 params) packed as one blob +
runtime `lr`/`bc₁`/`bc₂` scalars (cosine + warmup + per-step bias correction) through the unchanged
FFI (`n_params = 3k`).

**The artifact is `pretty(provenGraph)` since 2026-07-28** (handoff §2f). It used to come from a
hand-written string emitter in `tests/TestMobilenetV2TrainPC.lean`; the swap was licensed by
`mobilenetv2-adam-tie`, which found the two **bit-exact** on all 6,795,329 returned floats — forward
(the 52 BN batch statistics), `%loss`, and the gradient — with three negative controls each firing
its own gate. The driver itself needed no change: it resolves the path from the net slug.

Recipe matches `mobilenet-v2-train` (`MainMobilenetV2Train.lean`'s `mobilenetV2Config`): AdamW
lr 1e-3 / wd 1e-4, cosine + 3-epoch warmup, label smoothing 0.1, augment, 80 epochs, bs 32
(no EMA, no grad-clip). **Exact BN parity**: TRUE batch-norm (reduce `[0,2,3]`) in the train step
+ running-stats eval — `mobilenetv2Verified.bnChannels` (52 layers, full-paper 17-block net) is
non-empty, so the generic `trainAdamSched` threads per-layer EMA batch stats and evals through
`@mobilenetv2_fwd_eval` (affine BN with the running stats), class-batch-independent on the sorted
val set. Weight decay is applied uniformly (incl. BN/bias), matching the ViT path.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mobilenetv2-verified-adam data` (loader reads
`data/imagenette`). For the XLA/PJRT lowerer — measurably faster on this box, and the only path to
multi-GPU — use `mobilenetv2-verified-adam-xla`; the shared body is in
`apps/imagenette/MobilenetV2AdamCommon.lean` so the two cannot drift.
-/

def main (argv : List String) : IO Unit := runMobilenetV2Adam argv
