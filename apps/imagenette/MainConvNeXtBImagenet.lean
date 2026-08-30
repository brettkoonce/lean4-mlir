import LeanMlir.VerifiedNets

/-! # `convnext-b-imagenet-verified` — ConvNeXt-**Base** on full ImageNet-1k, verified → XLA

ConvNeXt-B is ConvNeXt-S's depth (`[3,3,27,3]`) at `[128,256,512,1024]`: same 344 parameter
tensors, every one of them wider. 88,591,464 scalars against S's 50,223,688 and T's 28,589,128 — all three timm's figures exactly
(the head LN came back 2026-08-30; each size was short by 2×dim before).

⚠⚠ **B is the size that made the DIMENSIONS a renderer parameter.** S was pure depth and never
touched a dimension literal, so it was served by threading one `Array Nat`. B moves the stem
(96 → 128), the head (768 → 1024) and all four stages — ~27 hardcoded literals across the two
renderers, concentrated in the stem, the head and the hand-written GAP backward. Depths and dims
are now a single `CnxDims` record precisely so that `(S depths, T dims)` — a net that exists
nowhere, yet type-checks, renders and trains — cannot be spelled.

⚠ **B shares S's depth table exactly**, so anything keyed on block count cannot separate them. That
is not hypothetical: the renderer's banner function keyed on block count while S was the only new
size, and every B artifact would have opened by introducing itself as a ConvNeXt-S.

⭐ **The proof side needed nothing**, and B is stronger evidence for that than S was: S reused the
per-site certificates at the *same* widths, where B instantiates them at four widths no committed
artifact had ever used. They are generic in `c`/`e`/`h`; neither depth nor width was a hypothesis.

⚠ Stochastic depth is **0.5** — the ConvNeXt paper's B value at 300 epochs, against S's 0.4 and
T's 0.1. Third distinct rate across three sizes, and still DATA (the spec's `dropKeeps`, supplied
per step), so `LEAN_MLIR_DROP_RATE_U` retunes it without touching an artifact. ▶ On the 80-epoch
tier use `LEAN_MLIR_DROP_RATE_U=300000` (0.3): the paper's per-size values underfit on the short
schedule (`planning/vit_convnext_sb_scaleup.md`, measured on the JAX side).

⚠ **Batch is 32 per device** — `cBS` is still a private constant. At four replicas that is global
128, half the paper's 256. ▶ For B this is the binding constraint rather than a scope note: see
the wall-clock and memory rows in `planning/vit_convnext_sb_scaleup.md` before proposing a run.

⚠ ConvNeXt has no BatchNorm, so there is no running-stats eval forward. `convnextbin_fwd.mlir`
plus the train step is the complete artifact set; `convnextbin_drop_fwd.mlir` is the SD render's
structural prefix partner, not what the driver evals.

⚠⚠ **NOTHING HAS BEEN TRAINED.** The artifacts render, the shapes tie to `VLayer.toSpecs`, the
count is `#guard`ed against the published 88.59M and against the independent JAX emitter. No
accuracy has been measured and none is claimed.

Run (4 GPUs — BOTH replica knobs are required):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  LEAN_MLIR_VARIANT=adamdpwxclipdrop LEAN_MLIR_BATCH=32 \
  PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
  .lake/build/bin/convnext-b-imagenet-verified data
```
-/

/-- 300 epochs at 32 per device — the ConvNeXt paper's schedule length, unchanged across T/S/B.
    The paper varies the stochastic-depth rate with size, not the schedule or the LR. -/
def convnextBImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 32

/-- Entry point. Defaults to the single-device `adamwxclipdrop`, as the T and S drivers do: a DP
    default makes a plain invocation fail at the first step on a replica-count refusal, which reads
    as a broken build rather than a missing flag. -/
def runConvNeXtBImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adamwxclipdrop"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD convnextBImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.00025   -- `convNeXtTinyImagenetConfig.learningRate`: 4e-3@bs4096 scaled to bs256.
                          -- ⚠ NOT retuned for B, matching the reference: the ConvNeXt paper uses
                          -- one LR across T/S/B and varies only the stochastic-depth rate.
                          -- ⚠ This runs at global 128, so the linear rule would put it near
                          -- 1.25e-4; 2.5e-4 matches the reference knob and is the first thing to
                          -- tune if it over-steps. Eight replicas would make it exactly right.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD convnextBImagenetConfig.epochs
  -- ▶ Stochastic-depth rate in MICRO-units (`500000` = 0.5, this spec's committed value). Unset ⇒
  -- the spec's ramp. ⚠⚠ `0` is THE GATE: every keep becomes 1.0, so each drop op is the identity
  -- in IEEE (`Proofs.dropPath_ones_id`) and the `*drop` render must train what a drop-free render
  -- trains. It is an ENDPOINT gate and blind to PLACEMENT, which `scripts/misplace_drop_sites.py`
  -- is the control for.
  let dropNet := match (← IO.getEnv "LEAN_MLIR_DROP_RATE_U").bind (·.toNat?) with
    | some 0 => { convnextBImagenetVerified.toNet with
                    dropKeeps := convnextBImagenetVerified.dropKeeps.map (fun _ => 1.0) }
    | some u =>
        -- Re-derive the ramp at a different rate from the SPEC's own keeps, so the block indices
        -- stay the renderer's: `(1 − k)/0.5` recovers `i/35`.
        -- ⚠ The `0.5` is B's committed rate — S's copy of this line divides by 0.4 and T's by 0.1.
        -- Three sizes, three divisors; carrying the wrong one silently rescales every request.
        let rate := u.toFloat * 1e-6
        { convnextBImagenetVerified.toNet with
            dropKeeps := convnextBImagenetVerified.dropKeeps.map
              (fun k => 1.0 - rate * ((1.0 - k) / 0.5)) }
    | none   => convnextBImagenetVerified.toNet
  dropNet.trainAdamSched
    { convnextBImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 20 variant
    -- warmup 20 epochs: `convNeXtTinyImagenetConfig.warmupEpochs := 20`, a ConvNeXt-paper value
    -- that differs from every other net here and does not move with model size.

def main (argv : List String) : IO Unit := runConvNeXtBImagenet argv
