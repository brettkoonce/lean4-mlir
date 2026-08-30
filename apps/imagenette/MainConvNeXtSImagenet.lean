import LeanMlir.VerifiedNets

/-! # `convnext-s-imagenet-verified` — ConvNeXt-**Small** on full ImageNet-1k, verified → XLA

The second net here added by RESHAPING an existing renderer rather than by writing a new chain, and
the cheapest of them so far. ConvNeXt-S is ConvNeXt-T DEEPENED: stage 3 goes 9 → 27 blocks, and the
dims do not move — still `[96,192,384,768]`, still 7×7 depthwise, still the same block. 344
parameter tensors and 50,223,688 scalars against T's 182 / 28,589,128 — both timm's figures
exactly, since the head LN was restored 2026-08-30 (planning §7.1).

⭐ **What it cost.** One `Array Nat` threaded as a trailing defaulted parameter through both
renderers. ViT-S needed six width constants turned into a `VitDims` record because ViT widens;
ConvNeXt-S only deepens, and the renderers already folded over the stage table in both directions
(`for si in [0:4] do for j in [0:depths[si]!]`, reversed in the backward). Because no DIMENSION
moves, the twelve hardcoded `768`s and thirteen `96`s in the head and the GAP backward stayed
correct untouched.

⭐ **The proof side needed nothing**, and for a different reason than ViT-S's. ViT's was that
`vitForwardKV_has_vjp` is already `∀ heads d_head mlpDim k`. ConvNeXt's is more basic: the
certificates are per-SITE and generic in `c`/`e`/`h` already, so 18 more blocks is 18 more uses of
theorems that were never indexed by depth. Depth was not a hypothesis.

⚠ **ConvNeXt-B is NOT one line away from here**, and the shape of this file should not suggest
otherwise. B is depth AND width (`[128,256,512,1024]`), which moves every dimension literal the S
work was able to leave alone — `planning/vit_convnext_sb_scaleup.md` counts them: the literals
outnumber the symbolic table uses in both renderers. That is a dims-threading pass with its own
byte-identity gate at T, not an instance.

⚠ **The stochastic-depth RATE differs from Tiny's and that is deliberate**: 0.4, the ConvNeXt
paper's value for S at 300 epochs, against T's 0.1. It is the one recipe knob that moves with model
size, and it is DATA (the spec's `dropKeeps`, supplied per step by the driver), not a render knob —
so `LEAN_MLIR_DROP_RATE_U` retunes it without touching an artifact. ▶ At the 80-epoch tier use
`LEAN_MLIR_DROP_RATE_U=200000` (0.2): the paper's 0.4 underfits on the short schedule
(`vit_convnext_sb_scaleup.md`, measured on the JAX side).

⚠ **Batch is 32 per device**, because `cBS` is still a private constant in the renderer. At four
replicas that is global 128 and 10,009 steps/epoch — more steps than the paper's 5,004 at batch
256, which §2d.2 says is the axis accuracy actually tracks. Threading `cBS` is a separate refactor.

⚠ ConvNeXt has no BatchNorm, so there is no running-stats eval forward: `convnextsin_fwd.mlir`
(drop-free, the identity an all-ones mask would give anyway) plus the train step is the complete
artifact set. Two variants are rendered — `adamwxclipdrop` (single device, the default) and
`adamdpwxclipdrop` (4 replicas, what a real run loads).

⚠⚠ **NOTHING HAS BEEN TRAINED.** The artifacts render, the shapes tie to `VLayer.toSpecs`, the
parameter count is `#guard`ed against the published 50.22M and against the independent JAX emitter,
and the drop ramp is tied to the renderer's own `cnxBlockIdx`. No accuracy has been measured and
none is claimed.

Run (4 GPUs — BOTH replica knobs are required, see `planning/mnv4_convm_ties_todo.md`):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  LEAN_MLIR_VARIANT=adamdpwxclipdrop LEAN_MLIR_BATCH=32 \
  PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
  .lake/build/bin/convnext-s-imagenet-verified data
```
-/

/-- 300 epochs at 32 per device — the ConvNeXt paper's schedule length, unchanged from Tiny. The
    paper changes the stochastic-depth rate between T and S, not the schedule.
    ⚠ `batchSize` is PER DEVICE and must match the batch the variant was rendered at (32; it is
    baked into the graph, so a mismatch is a shape error at the first invoke, not a silent limp). -/
def convnextSImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 32

/-- Entry point. Defaults to the SINGLE-DEVICE `adamwxclipdrop`, matching the ConvNeXt-T ImageNet
    driver rather than the ViT-S one: a DP default makes a plain invocation fail at the first step
    with a replica-count refusal, which reads as a broken build rather than a missing flag. Both
    variants are rendered, so either default would have been runnable — this one fails better. -/
def runConvNeXtSImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adamwxclipdrop"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD convnextSImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.00025   -- `convNeXtTinyImagenetConfig.learningRate`: 4e-3@bs4096 scaled to bs256.
                          -- ⚠ NOT retuned for S, and that matches the reference: the ConvNeXt paper
                          -- uses one LR across T/S/B and varies only the stochastic-depth rate.
                          -- ⚠ This run is at global 128, so the linear rule would put it near
                          -- 1.25e-4; 2.5e-4 is kept to match the reference knob, exactly as the
                          -- Tiny driver does, and is the first thing to tune if it over-steps.
  -- ⚠ `LEAN_MLIR_EPOCHS` SETS the schedule where `LEAN_MLIR_MAX_EPOCHS` only CAPS it. The cosine
  -- anneals over `cfg.epochs`, so EPOCHS=80 is a complete 80-epoch experiment while MAX_EPOCHS=80
  -- is a PREFIX of the 300-epoch decay stopped with the LR high. ▶ If you set EPOCHS=80, set
  -- `LEAN_MLIR_DROP_RATE_U=200000` too — the paper's 0.4 is a 300-epoch value.
  -- ⚠ Clear checkpoints when switching schedules; resuming across them fuses two LR curves silently.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD convnextSImagenetConfig.epochs
  -- ▶ Stochastic-depth rate in MICRO-units (`400000` = 0.4, this spec's committed value; `200000`
  -- = 0.2 for the 80-epoch tier). Unset ⇒ the spec's ramp.
  -- ⚠⚠ `0` is meaningful and it is THE GATE: every keep becomes 1.0, so every supplied scale is
  -- exactly 1.0 and each drop op is the identity in IEEE (`Proofs.dropPath_ones_id`) — the
  -- `*drop` render must then train what a drop-free render trains. ⚠ It is an ENDPOINT gate and
  -- endpoint gates are blind to PLACEMENT (`1 ⊙ (branch + x) = branch + x` exactly), which is what
  -- `scripts/misplace_drop_sites.py` is the control for.
  let dropNet := match (← IO.getEnv "LEAN_MLIR_DROP_RATE_U").bind (·.toNat?) with
    | some 0 => { convnextSImagenetVerified.toNet with
                    dropKeeps := convnextSImagenetVerified.dropKeeps.map (fun _ => 1.0) }
    | some u =>
        -- Re-derive the ramp at a different rate from the SPEC's own keeps, so the block indices
        -- stay the renderer's: `(1 − k)/0.4` recovers `i/35`, the index the spec encoded.
        -- ⚠ The `0.4` here is S's committed rate, NOT Tiny's 0.1 — the Tiny driver's copy of this
        -- line divides by 0.1, and carrying it over would rescale every requested rate by 4×.
        let rate := u.toFloat * 1e-6
        { convnextSImagenetVerified.toNet with
            dropKeeps := convnextSImagenetVerified.dropKeeps.map
              (fun k => 1.0 - rate * ((1.0 - k) / 0.4)) }
    | none   => convnextSImagenetVerified.toNet
  dropNet.trainAdamSched
    { convnextSImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 20 variant
    -- warmup 20 epochs, not 5: `convNeXtTinyImagenetConfig.warmupEpochs := 20` is a ConvNeXt-paper
    -- value and differs from every other net in this repo. It does not move with model size.

def main (argv : List String) : IO Unit := runConvNeXtSImagenet argv
