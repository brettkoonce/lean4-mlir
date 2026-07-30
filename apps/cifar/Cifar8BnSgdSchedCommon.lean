import LeanMlir.VerifiedNets

/-! # Shared body of the CIFAR-8 + per-channel BN + plain SGD on the momentum/Adam pipeline trainer

`cifar8-bn-sgdsched` reads `-verified-` in its exe name; the IREE and XLA executables are the
same program — same `VerifiedNetSpec`, same certified train-step artifact, same schedule and
He-init seed. Only the linked trusted lowerer differs.

Lake requires a distinct root module per executable, so the config and entry point live here
rather than being duplicated: drift in `epochs`, `batchSize`, the seed or the learning rate
would quietly invalidate any cross-backend comparison
(`planning/xla_pjrt_handoff.md` §2h, `xla_pjrt_ladder.md` §3).
-/

def cifar8BnSgdSchedConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

/-- Entry point for both backends.
    baseLR 0.1, 3-epoch warmup + cosine. Per-channel BN ⇒ train = eval, eval via
    `@cifar8_bn_fwd`. -/
def runCifar8BnSgdSched (argv : List String) : IO Unit :=
  cifar8BnVerified.toNet.trainAdamSched cifar8BnSgdSchedConfig (argv.head?.getD "data") 0.1 0.9 0.999 3 "sgd"
