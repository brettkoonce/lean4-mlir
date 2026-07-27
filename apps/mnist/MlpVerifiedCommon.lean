import LeanMlir.VerifiedNets

/-! # Shared body of the verified MLP trainer

`mnist-mlp-verified` (IREE) and `mnist-mlp-verified-xla` (XLA/PJRT) are the same
program — same `VerifiedNetSpec`, same `verified_mlir/mlp_*.mlir`, same §1a ties,
same loop and same He-init seed. Only the linked trusted lowerer differs, which
`VerifiedNet.mkSession` detects at run time.

Lake requires a distinct root module per executable, so the config and entry point
live here rather than being duplicated — a divergence in `epochs`, `batchSize`, or
the init seed would quietly invalidate the G2 comparison the two builds exist to
support (`planning/xla_pjrt_ladder.md` §3).
-/

def mlpConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

/-- Entry point for both backends. Trains through the packed-params
    `VerifiedNet.train` driver (`mlpTrainStepV`, He-init). -/
def runMlpVerified (argv : List String) : IO Unit :=
  mlpVerified.train mlpConfig (argv.head?.getD "data")
