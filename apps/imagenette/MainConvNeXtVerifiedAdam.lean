import apps.imagenette.ConvNeXtAdamCommon

/-! # `convnext-verified-adam` — train ConvNeXt-T with the VERIFIED-rendered **AdamW** step

The ConvNeXt-T peer of the `vit`/`mnv2`/`enet`/`r34` verified-adam trainers: the proof-rendered
train step (`LeanMlir/Proofs/Codegen/ConvNeXtRender.lean →
verified_mlir/convnext_adam_train_step.mlir`, `@convnext_adam_train_step`) — patchify stem →
[3,3,9,3] depthwise-7×7 blocks (LayerNorm + GELU + layerScale) + 3 between-stage downsamples → GAP →
LN → dense — with the gradients un-fused and handed to the proven
`adamMNextF`/`adamVNextF`/`adamWParamF` triple, driven by the generic `VerifiedNet.trainAdamSched`:
`[θ|m|v]` (180 params) packed as one blob + runtime `lr`/`bc₁`/`bc₂` scalars through the unchanged
FFI.

**The artifact is `pretty(provenGraph)` since 2026-07-28** (handoff §2f-bis). It used to come from a
hand-written string emitter in `tests/TestConvNeXtTrain.lean`; the swap was licensed by
`convnext-adam-tie`, and after the stem-patchify and even-kernel-downsample weight-gradient gaps were
closed (`9bb00f5`) that tie is **bit-exact** on all 83,434,629 returned floats with spread 0/180 —
so all 180 params are `pretty(AST)` end to end. The driver itself needed no change: it resolves the
path from the net slug.

ConvNeXt is all-smooth (LayerNorm, not BN), so there's no running-stats / train-vs-eval BN gap —
this is **exact-parity** territory like ViT (eval matches train), and it is why there is no
`convnext_fwd_eval` artifact and `fwd-tie convnext --eval` refuses outright. Recipe matches the
reference (`MainConvNeXtTrain.lean`'s `convNextTinyConfig`): AdamW lr 1e-3 / wd 1e-4, cosine +
3-epoch warmup, label smoothing 0.1, augment, 80 epochs, bs 32. Weight decay uniform (incl.
LN/bias), matching the other verified paths.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/convnext-verified-adam data` (loader reads
`data/imagenette`). For the XLA/PJRT lowerer — measurably faster on this box, and the only path to
multi-GPU — use `convnext-verified-adam-xla`; the shared body is in
`apps/imagenette/ConvNeXtAdamCommon.lean` so the two cannot drift.
-/

def main (argv : List String) : IO Unit := runConvNeXtAdam argv
