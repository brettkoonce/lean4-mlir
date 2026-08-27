import LeanMlir.VerifiedNets

/-! # `mobilenetv4-imagenet-verified` — MobileNetV4-Conv-M on full ImageNet-1k, verified → XLA

The sixth scale-tier trainer, and the last of the Imagenette nets to get one. Built the way
`resnet50-imagenet-verified` was: the Imagenette block table with a 1000-class head
(`mnv4ImagenetVerified`, slug `mnv4in`), rendered by the SAME chain as the 10-class artifacts
(`Proofs/Codegen/MobileNetV4RenderB.lean`), driven by the generic `VerifiedNet.trainAdamSched`.

⭐ **Conv-M as of 2026-08-14**, so this driver and `jax/MainMobilenetV4Imagenet.lean` are the
same network at last, and the reference's 75.48% / 92.37% is this driver's target. 9,715,512
parameters. ⚠ Target, not result: nothing has been run to convergence on this path.

⚠⚠ **There is no 4× render, and that is what blocks a printable phase-4 row.** `adamdp64` is
named by `mnv4AdamVariant 64 2` and the renderer would emit it, but MNv4 has no `shard-check` row
and no dp-check peer, so nothing ties its collectives and it is not rendered. Every other ImageNet
row in the book was measured at 4×, so a single-card figure here is not comparable to them.

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

    ⚠ **`adamdp64` is NOT rendered.** `mnv4AdamVariant 64 2` names it and the renderer would emit
    it, but MNv4 has no `shard-check` row and no dp-check peer, so nothing ties its collectives.
    Asking for it here fails at load with "artifact not found", which is the honest failure. -/
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
