import LeanMlir.VerifiedNets

/-! # Shared body of the verified ConvNeXt-T / **full ImageNet-1k** trainer

The ConvNeXt peer of `Resnet34ImagenetCommon` and `ViTImagenetCommon` (handoff §2p). Same certified
renderer as the Imagenette ConvNeXt at `nClasses := 1000`, fed by the generated tfds shim.

⚠ **Batch is 32 per device**, unlike the R34 (256) and ViT (128) ImageNet peers, because `cBS` is
still a private constant in `ConvNeXtRender.lean` while `nClasses` is now a parameter. At four
replicas that is **global 128 and 10,009 steps/epoch** — *more* optimizer steps than the JAX
reference's 5,004 at batch 256, and §2d.2 measured accuracy tracking step count, so this is a
defensible config rather than a degraded one. Threading `cBS` is a separate refactor if the
wall-clock is wanted.

⚠ **ConvNeXt is the most expensive net here per step** — 170 ms at bs32 on one card against ViT's
52 (measured 2026-08-01, ares, residency on). Budget accordingly before starting anything long.

⚠ **Claim ceiling.** The proof-carrying tier stops at Imagenette; what is inherited is provenance
plus whatever the pair comparison against `jax/MainConvNeXtImagenet.lean` shows. And this is **not**
the ConvNeXt paper recipe: mixup 0.8, cutmix 1.0, stochastic depth 0.1, EMA 0.9999, grad clip 1.0
and `wdExcludeNormBias` are all in the reference config and none exist on the verified path. The
pipeline-level augs (RandAugment geometric, random erasing) DO come across free via the shim. The
reference's own 80-epoch no-RandAugment run reached 75.93%; do not compare this to ConvNeXt-T's ~82%.

```bash
scripts/gen_shims.sh                       # this net's OWN data shim (⚠ NOT R34's — see VerifiedNet.shimScript)
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build convnext-imagenet-verified-xla
cat .lake/build/cnxin_adamdp_ckpt_xla.bin.epoch 2>/dev/null   # ⚠ READ THIS FIRST (§4)

PJRT_FFI_RESIDENT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LEAN_MLIR_VARIANT=adamdp LEAN_MLIR_BATCH=32 \
  LEAN_MLIR_REPLICAS=4 PJRT_REPLICAS=4 \
  .lake/build/bin/convnext-imagenet-verified-xla data 2>&1 | tee runs/cnxin_4gpu.log
```
-/

/-- 300 epochs at 32 per device — the ConvNeXt paper's schedule length. `batchSize` is PER DEVICE
    and must match the batch the selected variant was rendered at (32; it is baked into the graph,
    so a mismatch is a shape error at the first invoke rather than a silent limp). -/
def convnextImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 32

/-- Entry point. Defaults to the single-device `adam` variant rather than `adamdp`, matching the
    R34 and ViT ImageNet drivers: a DP default makes a plain invocation fail at the first step with
    a replica-count refusal, which reads as a broken build rather than a missing flag. -/
def runConvNeXtImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD convnextImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.00025   -- `convNeXtTinyImagenetConfig.learningRate`: 4e-3@bs4096 scaled to bs256.
                          -- ⚠ this run is at global 128, so the linear-scaling rule would put it
                          -- near 1.25e-4; 2.5e-4 is kept to match the reference knob and left as
                          -- the thing to tune first if it under- or over-steps.
  convnextImagenetVerified.toNet.trainAdamSched
    { convnextImagenetConfig with batchSize := bs }
    (argv.head?.getD "data") baseLR 0.9 0.999 20 variant
    -- warmup 20 epochs, not 5: `convNeXtTinyImagenetConfig.warmupEpochs := 20` is a ConvNeXt-paper
    -- value and differs from every other net in this repo.
