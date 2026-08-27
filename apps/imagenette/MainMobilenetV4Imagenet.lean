import LeanMlir.VerifiedNets

/-! # `mobilenetv4-imagenet-verified` — MobileNetV4-Conv-M on full ImageNet-1k, verified → XLA

The sixth scale-tier trainer, and the last of the Imagenette nets to get one. Built the way
`resnet50-imagenet-verified` was: the Imagenette block table with a 1000-class head
(`mnv4ImagenetVerified`, slug `mnv4in`), rendered by the SAME chain as the 10-class artifacts
(`Proofs/Codegen/MobileNetV4RenderB.lean`), driven by the generic `VerifiedNet.trainAdamSched`.

⭐ **Conv-M as of 2026-08-14**, so this driver and `jax/MainMobilenetV4Imagenet.lean` are the
same network at last, and the reference's 75.48% / 92.37% is this driver's target. 9,715,512
parameters. ⚠ Target, not result: nothing has been run to convergence on this path.

✅ **THE 4× RENDER EXISTS AND IS TIED, as of 2026-08-27.** This paragraph used to say there was
none, and that it was what blocked a printable phase-4 row. `mnv4in_adamdp64` and its bf16 peer are
4-replica renders, and both halves of the collective tie are green — `mnv4-dp-check` (duplicated
batch: fp32 `bnstat` bit-exact, gradient 8.45e-7; bf16 bit-exact on all 9,715,512 floats) and
`shard-check mnv4in` (asymmetric batch: TEST 1.10e-6 against a CONTROL of 2.00). Both go red on a
sum-not-mean render. ▶ `scripts/jobs/mnv4-default-4gpu.conf` is the job;
`runs/2026-08-27-mnv4-dp-shard-gates/` is the evidence.

⚠ A single-card figure off this driver is still not comparable to the book's other ImageNet rows,
which were all measured at 4× on this box. Run the job, not the bare binary, for anything printable.

⚠ Optimizer does NOT match the MNv4 reference: AdamW at 1e-3 here, where the paper is
AdamW at 0.004 on effective batch 4096 with drop-path, EMA and RandAugment m15. Those live in the
reference `TrainConfig`, and the ones that are not yet expressible on this path are listed in
`planning/chapter_makeover.md` under the MNv4 phase-4 gaps.

**One file, one binary, either lowerer.** The proven graph goes to whichever trusted lowerer
`$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with `=iree` -- resolved by dlopen at
run time (`ffi/lowerer.h`).

Run (GPU, single device):
```
PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
  .lake/build/bin/mobilenetv4-imagenet-verified data
```

Run (4 GPUs — the measured 25.6 h configuration, and the one the book quotes):
```
scripts/supervise.sh mnv4-default-4gpu
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

/-- Entry point. Defaults to the single-device `adam64` variant, matching the other five ImageNet
    drivers — a DP default dies at the first step on a replica-count refusal, which reads as a
    broken build rather than a missing flag.

    ⭐ **`adamdp64` IS rendered and tied** (2026-08-27), so asking for it works — but it is a
    4-REPLICA artifact and needs `PJRT_REPLICAS=4` AND `LEAN_MLIR_REPLICAS=4`. There is no
    2-replica peer, so a 2-GPU attempt hits the shim's replica-count guard rather than degrading.
    `scripts/jobs/mnv4-default-4gpu.conf` sets both. -/
def runMnv4Imagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adam64"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD mnv4ImagenetConfig.batchSize
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD mnv4ImagenetConfig.epochs
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.001   -- ⚠ NOT the reference's 0.004: that is a batch-4096 rate, this is 256.
  mnv4ImagenetVerified.toNet.trainAdamSched
    { mnv4ImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant

def main (argv : List String) : IO Unit := runMnv4Imagenet argv
