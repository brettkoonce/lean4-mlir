import LeanMlir.VerifiedNets

/-! # Shared body of the verified CNN trainer

`mnist-cnn-verified` (IREE) and `mnist-cnn-verified-xla` (XLA/PJRT) are the same
program — same `VerifiedNetSpec`, same `verified_mlir/cnn_*.mlir`, same §1a ties,
same loop and same He-init seed. Only the linked trusted lowerer differs.

Lake requires a distinct root module per executable, so the config and entry point
live here rather than being duplicated; drift in `epochs`, `batchSize`, or the init
seed would quietly invalidate the G2 comparison (`planning/xla_pjrt_ladder.md` §3).
-/

def cnnConfig : VerifiedConfig where
  epochs    := 10
  batchSize := 128

/-- Entry point for both backends. Trains through the packed-params
    `VerifiedNet.train` driver (`mlpTrainStepV`, He-init, 4-D kernels). -/
def runCnnVerified (argv : List String) : IO Unit :=
  cnnVerified.train cnnConfig (argv.head?.getD "data")
