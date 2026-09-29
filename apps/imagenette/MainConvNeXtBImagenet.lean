import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Train

/-! # `convnext-b-imagenet-verified` — ConvNeXt-**Base** on full ImageNet-1k, verified → XLA

ConvNeXt-B is ConvNeXt-S's depth (`[3,3,27,3]`) at `[128,256,512,1024]`: same 344 parameter
tensors, every one of them wider. 88,591,464 scalars against S's 50,223,688 and T's 28,589,128 — all three timm's figures exactly.

**The DIMENSIONS are a renderer parameter.** B moves the stem (96 → 128), the head (768 → 1024)
and all four stages. Depths and dims are a single `CnxDims` record precisely so that `(S depths, T dims)` — a net that exists
nowhere, yet would type-check, render and train — cannot be spelled.

**B shares S's depth table exactly**, so anything keyed on block count cannot separate them: a
banner keyed on block count introduces every B artifact as a ConvNeXt-S.

**The proof side needs nothing new**: B instantiates the per-site certificates at four widths no
other committed artifact uses. They are generic in `c`/`e`/`h`; neither depth nor width is a
hypothesis.

Stochastic depth is **0.5** — the ConvNeXt paper's B value at 300 epochs, against S's 0.4 and
T's 0.1. Third distinct rate across three sizes, and still DATA (the spec's `dropKeeps`, supplied
per step), so `LEAN_MLIR_DROP_RATE_U` retunes it without touching an artifact. On the 80-epoch
tier use `LEAN_MLIR_DROP_RATE_U=300000` (0.3): the paper's per-size values underfit on the short
schedule (measured on the JAX side).

**Batch is 64 per device** for the pair, ConvNeXt-T's rescope: global 256, the batch the LR is
scaled to. The pair variant is `emadpwxclipdropbf16` (EMA shadow, bf16), `emawxclipdropbf16` its
single-device peer and the default here; it peaks at 9.53 GiB of the plugin's 11.68 default arena
(compile probe, 2026-09-29), so no accumulation render and no `LEAN_MLIR_MEM_FRACTION`. The
`adam*wxclipdrop{,bf16}` siblings are rendered at 32 and need `LEAN_MLIR_BATCH=32`.

ConvNeXt has no BatchNorm, so there is no running-stats eval forward. `convnextbin_fwd.mlir` (at
64) plus the train step is the complete artifact set, and `convnextbin_fwd_s288` scores at timm's
test size; `convnextbin_drop_fwd.mlir` is the 32-batch SD renders' structural prefix partner, not
what the driver evals.

**NOTHING HAS BEEN TRAINED.** The artifacts render, the shapes tie to `VLayer.toSpecs`, the
count is `#guard`ed against the published 88.59M and against the independent JAX emitter. No
accuracy has been measured and none is claimed.

Run through the job config: `scripts/supervise.sh cnxb-default-emabf16-4gpu`.
-/

/-- 300 epochs at 64 per device — the ConvNeXt paper's schedule length, unchanged across T/S/B.
    The paper varies the stochastic-depth rate with size, not the schedule or the LR. -/
def convnextBImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 64
  -- ConvNeXt `_init_weights` (σ = 0.02 on every conv and the head), as the Tiny driver and the JAX
  -- reference. Host-side, so no re-render.
  cnxInit   := true
  -- The reference samples validation every 5 epochs (`jax/MainConvNeXtBImagenet.lean`); so does this.
  valEveryEpochs := 5

/-- Entry point. Defaults to the single-device `emawxclipdropbf16` at 64, as the S driver does: a DP
    default makes a plain invocation fail at the first step on a replica-count refusal, which reads
    as a broken build rather than a missing flag. -/
def runConvNeXtBImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "emawxclipdropbf16"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD convnextBImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.00025   -- `convNeXtTinyImagenetConfig.learningRate`: 4e-3@bs4096 scaled to bs256.
                          -- NOT retuned for B, matching the reference: the ConvNeXt paper uses
                          -- one LR across T/S/B and varies only the stochastic-depth rate.
                          -- At 64 × 4 this is global 256, the batch the rate is scaled to.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD convnextBImagenetConfig.epochs
  -- Stochastic-depth rate in MICRO-units (`500000` = 0.5, this spec's committed value). Unset ⇒
  -- the spec's ramp. `0` is THE GATE: every keep becomes 1.0, so each drop op is the identity
  -- in IEEE (`Proofs.dropPath_ones_id`) and the `*drop` render must train what a drop-free render
  -- trains. It is an ENDPOINT gate and blind to PLACEMENT, which `scripts/probes/misplace_drop_sites.py`
  -- is the control for.
  let dropNet := match (← IO.getEnv "LEAN_MLIR_DROP_RATE_U").bind (·.toNat?) with
    | some u => { convnextBImagenetVerified.toNet with
                  dropKeeps := convnextBImagenetVerified.dropKeepsAt (u.toFloat * 1e-6) }
    | none   => convnextBImagenetVerified.toNet
  dropNet.trainAdamSched
    { convnextBImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 20 variant
    -- warmup 20 epochs: `convNeXtTinyImagenetConfig.warmupEpochs := 20`, a ConvNeXt-paper value
    -- that differs from every other net here and does not move with model size.

def main (argv : List String) : IO Unit := runConvNeXtBImagenet argv
