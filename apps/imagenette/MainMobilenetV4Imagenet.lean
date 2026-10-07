import LeanMlir.Verified.NetsCore
import LeanMlir.Verified.Train

/-! # `mobilenetv4-imagenet-verified` — MobileNetV4-Conv-M on full ImageNet-1k, verified → XLA

The sixth scale-tier trainer, and the last of the Imagenette nets to get one. Built the way
`resnet50-imagenet-verified` was: the Imagenette block table with a 1000-class head
(`mnv4ImagenetVerified`, slug `mnv4in`), rendered by the SAME chain as the 10-class artifacts
(`Proofs/Codegen/MobileNetV4RenderB.lean`), driven by the generic `VerifiedNet.trainAdamSched`.

**Conv-M, on the timm `mobilenetv4_conv_medium` layout** (stride on the post-DW, BN-only
pre-DW, ReLU stage 0, symmetric stem, head pooled before `conv_head`), the same network as
jax/MainMobilenetV4Imagenet.lean; `scripts/parity/mnv4_timm_parity.py` ties both to timm on
shared weights. 9,715,512 parameters. The 100-epoch pair on the timm net:
**76.68 / 93.14** here (`mnv4-default-4gpu`, runs/2026-09-26-mnv4-verified-bf16-100ep) against
76.57 / 92.98 on the JAX path (`mnv4-default-jax-4gpu`, runs/2026-09-26-mnv4-jax-bf16-100ep), a
tie inside one Wilson half-width. The pre-timm reference's 75.48 is a different network.

✅ **The 4× renders are tied.** `mnv4-dp-check` (duplicated batch) covers the 4-replica renders,
and `imagenet-syncbn-check mnv4` (split batch: 4×64 IS 1×256) covers `adamdp64`. Their BatchNorm
is synchronised, so a 4×B step IS the single-device step at batch 4B. The split-batch gate does
not yet cover the shipping `emaaccdp8x128wxdowd005bf16`.

The job is `scripts/jobs/mnv4-default-4gpu.conf`: `emaaccdp8x128wxdowd005bf16`, i.e. AdamW 0.004
(`LEAN_MLIR_BASE_LR_U=4000`) at 8 accumulated micro-batches of 4 × 128 = effective 4096, sync-BN
over 512, wd 0.05 off norm/bias, classifier dropout 0.1, EMA 0.9999, bf16, 100 epochs, RandAugment
N2 m9 from the shim: the JAX reference's `default` recipe. The optimizer, schedule and
regularisers are selected by the variant string; this file only supplies the defaults below.
The 500-epoch paper tier is `LEAN_MLIR_RECIPE=full` with `accdp8x128wxdropdowd01bf16`
(`scripts/jobs/mnv4-full-4gpu.conf`): the `full` shim's RandAugment m15, dropout 0.2, drop-path
0.075 on the 18 skip blocks, wd 0.1, no EMA.

A single-card figure off this driver is not comparable to the book's other ImageNet rows, which
were all measured at 4×. Run the job, not the bare binary, for anything printable.

**One file, one binary, either lowerer.** The proven graph goes to whichever trusted lowerer
`$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with `=iree` -- resolved by dlopen at
run time (`ffi/lowerer.h`).

Run (GPU, single device):
```
SHIM_WORKERS=8 \
  .lake/build/bin/mobilenetv4-imagenet-verified data
```

Run (4 GPUs, the 100-epoch pair's verified side, as overnight chunks):
```
START_AT=00:00 STOP_AT=08:00 setsid nohup scripts/supervise.sh mnv4-default-4gpu >/dev/null 2>&1 &
```
-/

/-- 100 epochs at 64 PER DEVICE; four replicas would give the global 256 the other ImageNet
    drivers use. 100 matches the Conv-M reference's tier-2 schedule length, which is the closest
    thing to a target this block table has.

    `batchSize` is PER DEVICE and must match the batch the variant was rendered at (64 — see the
    `#eval`s in `MobileNetV4RenderB.lean`). `LEAN_MLIR_EPOCHS` SETS this, where
    `LEAN_MLIR_MAX_EPOCHS` only caps it: `totalSteps := cfg.epochs * nb / accK` is what the cosine
    anneals over, so `EPOCHS=30` is a complete 30-epoch experiment while `MAX_EPOCHS=30` is a
    PREFIX of a 100-epoch decay stopped with the LR still high. -/
def mnv4ImagenetConfig : VerifiedConfig where
  epochs    := 100
  batchSize := 64
  -- **Depthwise fan = k², as the JAX reference.** Every JAX depthwise emitter hard-codes
  -- fan = k² (TF/timm's `variance_scaling` on a `(k,k,C,1)` kernel); `mkParam`'s He fan-out gave
  -- `2/(C·k²)`, 0.03–0.18× the reference's std on the 30 depthwise kernels (planning/init_parity.md §2a,
  -- imagenet_parity.md D4). Host-side, no re-render. On since 2026-10-07; verified runs before it
  -- started at the narrow fan, and `LEAN_MLIR_DW_FAN_K2=0` reproduces those from their seed.
  dwFanK2 := true

/-- Entry point. Defaults to the single-device `adam64` variant, matching the other five ImageNet
    drivers — a DP default dies at the first step on a replica-count refusal, which reads as a
    broken build rather than a missing flag.

    The `…dp…` variants are 4-REPLICA artifacts and need `PJRT_REPLICAS=4` AND
    `LEAN_MLIR_REPLICAS=4`. There is no 2-replica peer, so a 2-GPU attempt hits the shim's
    replica-count guard rather than degrading. `scripts/jobs/mnv4-default-4gpu.conf` sets both. -/
def runMnv4Imagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam64"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD mnv4ImagenetConfig.batchSize
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD mnv4ImagenetConfig.epochs
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.001   -- NOT the reference's 0.004: that is a batch-4096 rate, this is 256.
  -- `LEAN_MLIR_RECIPE` picks the NET SPEC, hence the shim and the host-drawn masks: `default` (the
  -- 100-epoch pair, RandAugment m9, dropout 0.1) or `full` (the paper tier: m15, dropout 0.2,
  -- drop-path 0.075 on the skip blocks). A `drop` variant on `default` would draw no masks
  -- (`default` has no drop sites) and `full` without one would skip the drop-path, so the two
  -- must agree.
  let recipe := ((← IO.getEnv "LEAN_MLIR_RECIPE").getD "default").trimAscii.toString
  let net ← match recipe with
    | "default" => pure mnv4ImagenetVerified
    | "full"    => pure mnv4ImagenetFullVerified
    | r => throw <| IO.userError s!"LEAN_MLIR_RECIPE={r}: `default` or `full`"
  if VerifiedVariant.sdOn variant != (recipe == "full") then
    throw <| IO.userError s!"LEAN_MLIR_RECIPE={recipe} with LEAN_MLIR_VARIANT={variant}: the `full` \
      recipe trains a `drop` variant (e.g. accdp8x128wxdropdowd01bf16) and `default` does not."
  IO.println s!"  ▸ RECIPE: {recipe} — augmentation from {net.shimScript}"
  net.toNet.trainAdamSched
    { mnv4ImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant

def main (argv : List String) : IO Unit := runMnv4Imagenet argv
