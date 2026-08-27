import LeanMlir.VerifiedNets

/-! # `vit-s-imagenet-verified` — ViT-**Small** on full ImageNet-1k, verified → XLA

The first net in this repo added by WIDENING an existing one rather than by writing a new chain.
ViT-S is ViT-Ti widened: `D = 384 = 6 heads × 64` against Tiny's `192 = 3 × 64`, MLP 1536 against
768, and everything else identical — same depth 12, same 16×16 patch grid, same block structure,
same drop-path ramp. 22,050,664 parameters against Tiny's 5,717,416.

⭐⭐ **The proof side needed nothing.** `Proofs.vitForwardKV_has_vjp` is already
`∀ heads d_head mlpDim k`, and it is a GLOBAL `HasVJP` rather than the pointwise `_at` form the
relu-family nets carry, because GELU/softmax/LayerNorm have no kink. The same theorem covers Tiny
and Small at different arguments. What changed was the RENDERER: `ViTRenderB.lean`'s six
`private def` width constants became a `VitDims` record threaded as a trailing defaulted
parameter, so one renderer serves both sizes. Every ViT-Tiny artifact re-renders byte-identical,
which is the gate that says the parameterisation was inert.

⚠ **Only the 4-replica variant is rendered**, so unlike the other ImageNet drivers this one has no
single-device default to fall back to. `adamdp128x4wxclipdrop` at 128 per device × 4 = the global
512 the DeiT recipe uses. There is no `adam128` peer for ViT-S; asking for one fails at load.

⚠ ViT has no BatchNorm, so there is no running-stats eval forward. `vitsin_drop_fwd` plus the
train step is the complete artifact set for this net.

⚠ **Nothing has been trained.** The artifacts render, the shapes tie to `VLayer.toSpecs`, and the
parameter count is `#guard`ed. No accuracy has been measured and no wall clock has been probed.

Run (4 GPUs — and BOTH replica knobs are required, see `planning/mnv4_convm_ties_todo.md`):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  LEAN_MLIR_VARIANT=adamdp128x4wxclipdrop LEAN_MLIR_BATCH=128 \
  PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
  .lake/build/bin/vit-s-imagenet-verified data
```
-/

/-- 300 epochs at 128 per device — the DeiT schedule length, and at four replicas the reference's
    global batch of 512. Same config as the Tiny driver: S changes the width, not the recipe. -/
def vitSImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 128

/-- Entry point. ⚠ Defaults to the FOUR-REPLICA variant, unlike every other ImageNet driver here,
    because it is the only one rendered for this net. A plain invocation therefore needs
    `PJRT_REPLICAS=4` and `LEAN_MLIR_REPLICAS=4` or it fails at the first step on a replica-count
    refusal. That is the honest failure: the alternative is a single-device default that names an
    artifact which does not exist. -/
def runViTSImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adamdp128x4wxclipdrop"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD vitSImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.0005   -- the DeiT batch-512 rate, as the Tiny driver uses. ⚠ NOT retuned for S:
                         -- DeiT-S uses the same 5e-4 at batch 512, so this matches the reference
                         -- rather than being an untuned carry-over.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD vitSImagenetConfig.epochs
  vitSImagenetVerified.toNet.trainAdamSched
    { vitSImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant

def main (argv : List String) : IO Unit := runViTSImagenet argv
