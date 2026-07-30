import LeanMlir.VerifiedNets

/-! # Shared body of the verified ResNet-34 / **full ImageNet-1k** trainer

The scale/reference tier of handoff §2k. Same certified renderer as the Imagenette R34, at
`nClasses := 1000` and `B := 256`, with the **heavy-ball + coupled-L2** optimizer the
`jax/MainResnetImagenet.lean` reference uses — so the two can be run as a matched pair with the
JAX side as an external oracle, rather than as two unrelated experiments.

Data comes from the **generated tfds shim** (`VerifiedData.imagenet`): the train split streams over
a pipe, already RandomResizedCrop'd / flipped / normalized by the same `build_imagenet_iter` the
reference trainer consumes, and the val split is drained into RAM once (49,920 images after tfds
`drop_remainder` — the count the reference reports). **This side does no augmentation**, which is
the point: one definition of the transform, and it is the reference's.

⚠ **Claim ceiling.** The proof-carrying tier stops at Imagenette. What is inherited here is
*provenance* — the artifacts are `pretty(provenGraph)` off the same renderer, because `nClasses`,
`B`, `opt` and `slug` are ordinary parameters of it — plus whatever the pair agreement shows. Say
"one architecture, two independent lowerings, agreeing", never "proven". See `resnet34ImagenetVerified`.

Prerequisites, in order:

```bash
(cd jax && lake exe resnet34-imagenet default --shim)   # emit the data shim
lake build resnet34-imagenet-verified-xla
HIP_VISIBLE_DEVICES=0 LEAN_MLIR_VARIANT=mom256 LEAN_MLIR_BATCH=256 \
  LEAN_MLIR_BASE_LR_U=100000 \
  .lake/build/bin/resnet34-imagenet-verified-xla data
```

`LEAN_MLIR_BASE_LR_U=100000` is **0.1**, the reference's rate. The 0.001 default is an *AdamW*
rate and would under-step heavy-ball by ~100×, which reads as a broken render rather than a wrong
knob — see `Resnet34AdamCommon`.
-/

/-- 30 epochs at batch 256 — the `resnet34ImagenetConfigShort` tier on the JAX side, i.e. the
    validation subrun rather than the 90-epoch paper recipe. At the measured bs256 rate that is
    ~28 h on one 7900 XTX (~17 h on two), against the reference's ~5 h for the same tier on 4 CUDA
    cards at bf16. 5-epoch warmup matches the reference; the schedule is cosine as everywhere here. -/
def resnet34ImagenetConfig : VerifiedConfig where
  epochs    := 30
  batchSize := 256

/-- Entry point. Defaults to the `mom256` variant — the ImageNet artifact — rather than `adam`,
    since that is the only render this spec's slug has; `LEAN_MLIR_VARIANT` still overrides. -/
def runResnet34Imagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "mom256"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD resnet34ImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.1        -- the reference rate; unlike the Imagenette driver this net has no
                           -- AdamW variant, so an Adam default here would only ever be wrong.
  resnet34ImagenetVerified.toNet.trainAdamSched
    { resnet34ImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant
