import LeanMlir.VerifiedNets

/-! # Shared body of the verified CIFAR-8-BN (plain SGD) trainer

`cifar8-bn-verified` (IREE) and `cifar8-bn-verified-xla` (XLA/PJRT) are the same
program — same `VerifiedNetSpec`, same `verified_mlir/cifar8_bn_train_step.mlir`
and `cifar8_bn_fwd.mlir`, same `epochs`/`batchSize`/He-init seed. Only the linked
trusted lowerer differs.

Lake requires a distinct root module per executable, so the config and entry point
live here rather than being duplicated; drift in `epochs`, `batchSize` or the seed
would quietly invalidate any cross-backend comparison
(`planning/xla_pjrt_ladder.md` §3).

This is the **conv anchor** for both `lake run benchmark` and `lake run
benchmark-xla` (`planning/xla_pjrt_handoff.md` §2j). The two benchmarks must probe
the *same net* for their per-family factors to mean the same thing, which is why
the XLA peer exists at all rather than the probe being re-pointed at
`cifar8-bn-verified-adam-xla` — a different optimizer would need its own reference
constant and would stop the two tables being comparable row by row.

Unlike the AdamW arms this net trains through `VerifiedNet.train`, whose SGD update
and lr are **baked into the artifact**; it keeps no optimizer state and writes no
checkpoint, so the §4 stale-checkpoint and cross-backend `.vmfb` hazards do not
arise here (the XLA path compiles in-process and never writes a `.vmfb`).
-/

def cifar8BnConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

/-- Entry point for both backends. The lr is baked into the render, not passed. -/
def runCifar8Bn (argv : List String) : IO Unit :=
  cifar8BnVerified.train cifar8BnConfig (argv.head?.getD "data")
