import LeanMlir.VerifiedNets

/-! # Shared body of the verified ViT-Tiny / **full ImageNet-1k** trainer

The ViT peer of `Resnet34ImagenetCommon`, and the scale tier of handoff §2p. Same certified
renderer as the Imagenette ViT, at `nClasses := 1000` and `bs := 128` — so the four-replica run is
at **global batch 512**, which is `jax/MainVitImagenet.lean`'s `vitTinyImagenetConfig.batchSize`.
Matching the reference's global batch is what makes the two runs a comparable pair instead of two
experiments.

Data comes from the **generated tfds shim** (`VerifiedData.imagenet`): the train split streams over
a pipe, already RandomResizedCrop'd / flipped / normalized by the same `build_imagenet_iter` the
reference trainer consumes, and the val split is drained into RAM once. **This side does no
augmentation**, which is the point — one definition of the transform, and it is the reference's.

⚠ **ViT is the first net here where the LOADER, not the GPU, is the ceiling.** A 4×128 step costs
~264 ms and consumes 512 images, i.e. ~1,940 img/s, while one shim process delivers ~1,530
(measured 2026-08-01, marginal). Set `SHIM_WORKERS=2` (or more) or the cards will wait on data —
R34/ImageNet never needed this because bs256 at ~900 ms/step only asks for ~380 img/s.

⚠ **Claim ceiling.** The proof-carrying tier stops at Imagenette; what is inherited here is
provenance plus whatever the pair agreement shows. See `vitImagenetVerified`. And this is **not**
the DeiT recipe — mixup, cutmix, stochastic depth, EMA and grad clipping are all in
`vitTinyImagenetConfig` and none of them exist on the verified path. Do not compare to DeiT-Ti's
72.0%; the reference's own grad-clip-only 80-epoch ancestor reached 65.6%, and this has less.

Prerequisites, in order:

```bash
scripts/gen_shims.sh                       # this net's OWN data shim (⚠ NOT R34's — see VerifiedNet.shimScript)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build vit-imagenet-verified-xla
cat .lake/build/vitin_adamdp128x4_ckpt_xla.bin.epoch 2>/dev/null   # ⚠ READ THIS FIRST (§4)

PJRT_FFI_RESIDENT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 SHIM_WORKERS=2 \
  LEAN_MLIR_VARIANT=adamdp128x4 LEAN_MLIR_BATCH=128 \
  LEAN_MLIR_REPLICAS=4 PJRT_REPLICAS=4 \
  .lake/build/bin/vit-imagenet-verified-xla data 2>&1 | tee runs/vitin_4gpu.log
```
-/

/-- 300 epochs at 128 per device — the DeiT-Ti schedule length, and at four replicas the
    reference's global batch of 512. `batchSize` is PER DEVICE, as everywhere else here
    (`LEAN_MLIR_BATCH` overrides, and must match the batch the selected variant was rendered at,
    since it is baked into the graph). -/
def vitImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 128

/-- Entry point. Defaults to the **single-device** `adam128` variant rather than the four-replica
    one, matching `runResnet34Imagenet`: a DP default would make a plain invocation fail at the
    first step with a replica-count refusal, which reads as a broken build rather than a missing
    flag. `LEAN_MLIR_VARIANT=adamdp128x4` selects the 4-GPU render. -/
def runViTImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam128"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD vitImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.0005   -- `vitTinyImagenetConfig.learningRate`, the DeiT batch-512 rate. The
                         -- Imagenette ViT driver's 3e-4 is tuned for global batch 32 and would
                         -- under-step this by ~1.7x.
  vitImagenetVerified.toNet.trainAdamSched
    { vitImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant
