import LeanMlir.VerifiedNets

/-! # `vitb-imagenet-verified` — ViT-**Base** (DeiT-B) on full ImageNet-1k, verified → XLA

The third width off one renderer, and the widest. ViT-B is ViT-Ti widened twice over:
`D = 768 = 12 heads × 64` against Small's `384 = 6 × 64` and Tiny's `192 = 3 × 64`, MLP 3072
against 1536 and 768, and everything else identical — same depth 12, same 16×16 patch grid, same
block structure, same drop-path ramp. 86,567,656 parameters — DeiT-B's published 86.57M — in the
SAME 200 tensors as Ti and S.

⭐⭐ **The proof side needed nothing.** `Proofs.vitForwardKV_has_vjp` is already
`∀ heads d_head mlpDim k`, and it is a GLOBAL `HasVJP` rather than the pointwise `_at` form the
relu-family nets carry, because GELU/softmax/LayerNorm have no kink. The same theorem covers Tiny,
Small and Base at different arguments. What changed was the RENDERER: `ViTRenderB.lean`'s six
`private def` width constants became a `VitDims` record threaded as a trailing defaulted
parameter, so one renderer serves all three sizes. Every ViT-Tiny artifact re-renders byte-identical,
which is the gate that says the parameterisation was inert.

⚠⚠ **GLOBAL BATCH 128, NOT THE DeiT 512, AND THAT IS A MEASURED MEMORY LIMIT.** ViT-B at 4×128
was rendered and run: it OOMs, asking 11.90 GiB against the 11.68 GiB the BFC allocator gets on a
16 GB card, on all four devices. 4×32 = global 128 fits and trains at 432 ms/step. Reaching the
recipe's global 512 needs GRADIENT ACCUMULATION, and ViT has no accumulation render — only R50
does (`resnet50in_accdp4x64_train_step.mlir` and peers). That is a renderer feature, not a config.
▶ **So no run of this driver is comparable to DeiT-B's 81.8%**: the batch is 4× short and the LR
here is the batch-512 rate, un-rescaled.

⚠ **Only the 4-replica variant is rendered**, so unlike the other ImageNet drivers this one has no
single-device default to fall back to. `adamdp32x4wxclipdrop` is 32 per device × 4 = the global
128 above, NOT the DeiT 512. There is no single-device peer for ViT-B; asking for one fails at load.

⚠ ViT has no BatchNorm, so there is no running-stats eval forward. `vitbin_drop_fwd` plus the
train step is the complete artifact set for this net.

⚠ **Nothing has been trained.** The artifacts render, the shapes tie to `VLayer.toSpecs`, and the
parameter count is `#guard`ed. No accuracy has been measured and no wall clock has been probed.

Run (4 GPUs — and BOTH replica knobs are required, see `planning/mnv4_convm_ties_todo.md`):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  LEAN_MLIR_VARIANT=adamdp32x4wxclipdrop LEAN_MLIR_BATCH=32 \
  PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
  .lake/build/bin/vitb-imagenet-verified data
```
-/

/-- 300 epochs — the DeiT schedule length. ⚠ But 32 per device, so four replicas is global 128 and
    NOT the reference's 512: see the memory paragraph above. The schedule is DeiT's; the batch is
    not, and the LR here is still the batch-512 rate. -/
def vitBImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 32

/-- Entry point. ⚠ Defaults to the FOUR-REPLICA variant, unlike every other ImageNet driver here,
    because it is the only one rendered for this net. A plain invocation therefore needs
    `PJRT_REPLICAS=4` and `LEAN_MLIR_REPLICAS=4` or it fails at the first step on a replica-count
    refusal. That is the honest failure: the alternative is a single-device default that names an
    artifact which does not exist. -/
def runViTBImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adamdp32x4wxclipdrop"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD vitBImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.0005   -- the DeiT batch-512 rate, as the Tiny driver uses. ⚠ NOT retuned for S:
                         -- DeiT-S uses the same 5e-4 at batch 512, so this matches the reference
                         -- rather than being an untuned carry-over.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD vitBImagenetConfig.epochs
  vitBImagenetVerified.toNet.trainAdamSched
    { vitBImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant

def main (argv : List String) : IO Unit := runViTBImagenet argv
