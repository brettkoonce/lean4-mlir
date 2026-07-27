import LeanMlir.VerifiedNets

/-! # Shared body of the verified linear trainer

`mnist-linear-verified` (IREE) and `mnist-linear-verified-xla` (XLA/PJRT) are the
**same program** — same `VerifiedNetSpec`, same `verified_mlir/linear_*.mlir`,
same §1a ties, same loop. They differ only in which trusted lowerer is linked,
which `VerifiedNet.mkSession` detects at run time.

Lake requires a distinct root module per executable, so the config and entry
point live here rather than being duplicated in the two `Main*` files — a
divergence in `epochs` or `batchSize` between the two would quietly invalidate
the G2 comparison they exist to support (`planning/xla_pjrt_ladder.md` §3).
-/

/-- The model `linearVerified` (a single dense 784→10) lives in
    `LeanMlir.VerifiedNets` so this trainer and the `Proofs.SpecVJP` VJP
    theorem share one object. -/
def linearConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

/-- Entry point for both backends. The train loop lives in the driver
    (`VerifiedNet.trainLinear`); linear uses the 2-argument `linearTrainStepV`
    FFI rather than the packed-params path the other nets use. -/
def runLinearVerified (argv : List String) : IO Unit :=
  linearVerified.trainLinear linearConfig (argv.head?.getD "data")
