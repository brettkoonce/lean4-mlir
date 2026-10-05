import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Train

/-! # `vit-s-imagenet-verified` — ViT-**Small** on full ImageNet-1k, verified → XLA

Added by WIDENING an existing net rather than by writing a new chain.
ViT-S is ViT-Ti widened: `D = 384 = 6 heads × 64` against Tiny's `192 = 3 × 64`, MLP 1536 against
768, and everything else identical — same depth 12, same 16×16 patch grid, same block structure,
same drop-path ramp. 22,050,664 parameters against Tiny's 5,717,416.

**The proof side needs nothing new.** `Proofs.vitForwardKVHasVJP` is
`∀ heads d_head mlpDim k`, and it is a GLOBAL `HasVJP` rather than the pointwise `_at` form the
relu-family nets carry, because GELU/softmax/LayerNorm have no kink. The same theorem covers Tiny
and Small at different arguments. The widths live in the RENDERER: `ViTRenderB.lean` threads a
`VitDims` record as a trailing defaulted parameter, so one renderer serves both sizes.

**Only the 4-replica variant is rendered**, so unlike the other ImageNet drivers this one has no
single-device default to fall back to. `adamdp128x4wxclipdrop` at 128 per device × 4 = the global
512 the DeiT recipe uses. There is no `adam128` peer for ViT-S; asking for one fails at load.

ViT has no BatchNorm, so there is no running-stats eval forward. `vitsin_drop_fwd` plus the
train step is the complete artifact set for this net.

**Nothing has been trained**, and no accuracy has been measured. The wall clock HAS been
probed on real ImageNet on four cards, fp32 and bf16: `runs/2026-08-27-vitb-global512/`.

**Run it through the job config**, which owns the device list, the epoch budget and the restart
policy:
```
scripts/supervise.sh vits-default-emabf16-4gpu
DRY_RUN=1 scripts/supervise.sh vits-default-emabf16-4gpu   # print the plan, run nothing
```
`vits-default-emabf16-4gpu` is the pair job: `vitsin_emadp128x4wxclipdropeps0000001bf16`, Tiny's
pair recipe (EMA 0.99996, timm/DeiT init, bf16, DeiT's LayerNorm ε 1e-6 and min lr 1e-5) at this
width; its eval forward is `vitsin_emadp128x4wxclipdropeps0000001bf16_fwd.mlir`. `vits-default-g512-4gpu` is the non-EMA
f32 sibling.

By hand (4 GPUs — BOTH replica knobs are required):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  LEAN_MLIR_VARIANT=adamdp128x4wxclipdrop LEAN_MLIR_BATCH=128 \
  SHIM_WORKERS=8 \
  .lake/build/bin/vit-s-imagenet-verified data
```
**DO NOT SET `LEAN_MLIR_MEM_FRACTION` FOR THIS NET.** S's graph
fits the plugin's default arena; ViT-B's does not, which is why B's driver refuses
without the option. Setting it here looks like free headroom. It is not free: on ConvNeXt-S
and -B the same 0.97 makes both bf16 arms die
`CUDA_ERROR_OUT_OF_MEMORY` in `d2h(res)` — B with a core dump — where the identical runs complete
at the default, because a BFC pool at fraction 0.97 starves what lives OUTSIDE it (device-to-host staging, NCCL,
workspaces). Probed both ways, this net reads 528 → 319 with the option and **531 → 323 without**:
it buys nothing. The rule: raise the fraction for a graph that does not otherwise fit, and
leave it alone for one that does. `runs/2026-08-28-convnext-sb-jobs/`.
-/

/-- 300 epochs at 128 per device — the DeiT schedule length, and at four replicas the reference's
    global batch of 512. Same config as the Tiny driver: S changes the width, not the recipe. -/
def vitSImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 128
  -- timm/DeiT init, as the Tiny driver and the JAX reference (`vitInit := true` in every JAX
  -- ViT base config since VT-1). Without it every transformer Linear is Glorot. Host-side, so no
  -- re-render.
  vitInit   := true

/-- Entry point. Defaults to the FOUR-REPLICA variant, unlike every other ImageNet driver here,
    because it is the only one rendered for this net. A plain invocation therefore needs
    `PJRT_REPLICAS=4` and `LEAN_MLIR_REPLICAS=4` or it fails at the first step on a replica-count
    refusal. That is the honest failure: the alternative is a single-device default that names an
    artifact which does not exist. -/
def runViTSImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adamdp128x4wxclipdrop"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD vitSImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.0005   -- the DeiT batch-512 rate, as the Tiny driver uses. NOT retuned for S:
                         -- DeiT-S uses the same 5e-4 at batch 512, so this matches the reference
                         -- rather than being an untuned carry-over.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD vitSImagenetConfig.epochs
  -- DeiT's EMA decay, named because `trainAdamSched` defaults to 0.9999 (see the Tiny driver).
  -- Inert unless the variant starts with `ema`.
  vitSImagenetVerified.toNet.trainAdamSched
    { vitSImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant (emaDecay := 0.99996)
    -- DeiT `--min-lr 1e-5`, the reference's `vitSImagenetConfig.minLR`, as the Tiny driver.
    (minLR := 0.00001)

def main (argv : List String) : IO Unit := runViTSImagenet argv
