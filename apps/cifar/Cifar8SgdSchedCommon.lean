import LeanMlir.VerifiedNets

/-! # Shared body of the CIFAR-8 (no BN) + plain SGD on the momentum/Adam pipeline trainer

`cifar8-sgdsched` reads `-verified-` in its exe name; the IREE and XLA executables are the
same program — same `VerifiedNetSpec`, same certified train-step artifact, same schedule and
He-init seed. Only the linked trusted lowerer differs.

Lake requires a distinct root module per executable, so the config and entry point live here
rather than being duplicated: drift in `epochs`, `batchSize`, the seed or the learning rate
would quietly invalidate any cross-backend comparison
(`planning/xla_pjrt_handoff.md` §2h, `xla_pjrt_ladder.md` §3).
-/

def cifar8SgdSchedConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

/-- Entry point for both backends.
    baseLR 0.1, 3-epoch warmup + cosine, same per-epoch shuffle + random hflip as the momentum
    and Adam runs — which is the point: the optimizer is the ONLY free variable. -/
def runCifar8SgdSched (argv : List String) : IO Unit :=
  cifar8Verified.toNet.trainAdamSched cifar8SgdSchedConfig (argv.head?.getD "data") 0.1 0.9 0.999 3 "sgd"
