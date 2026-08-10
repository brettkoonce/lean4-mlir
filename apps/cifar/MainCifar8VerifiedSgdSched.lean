import LeanMlir.VerifiedNets

/-! # `cifar8-verified-sgdsched` — plain SGD on the SAME pipeline as the momentum/Adam runs

The controlled-baseline peer for the optimizer ablation. `cifar8-verified` (plain SGD via
`.train`) uses NO shuffle / NO augmentation / flat lr, so comparing it to the
momentum/Adam runs (which go through `trainAdamSched` with per-epoch shuffle + random hflip +
cosine-warmup) confounds the optimizer with the data pipeline. This exe runs **plain SGD
through `trainAdamSched` itself** (`variant := "sgd"`, update `θ←θ−lr·∇`, the m/v slots
passthrough), so SGD/momentum/Adam differ ONLY in the update rule — a clean 3-way ablation.

baseLR 0.1 (no momentum amplification ⇒ same neighborhood as plain SGD's flat optimum),
3-epoch warmup + cosine decay, same shuffle + hflip as the other two.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-verified-sgdsched data`

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects -- XLA/PJRT by default, IREE with
`=iree` -- resolved by dlopen at run time (`ffi/lowerer.h`). There is no `-xla`
peer and no shared-body file: the config and entry point below ARE the program.
The backend is a run-time choice about transport, not a different program, which
is what the G2 gate asserts.
-/

def cifar8SgdSchedConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

/-- Entry point for both backends.
    baseLR 0.1, 3-epoch warmup + cosine, same per-epoch shuffle + random hflip as the momentum
    and Adam runs — which is the point: the optimizer is the ONLY free variable. -/
def runCifar8SgdSched (argv : List String) : IO Unit :=
  cifar8Verified.toNet.trainAdamSched cifar8SgdSchedConfig (argv.head?.getD "data") 0.1 0.9 0.999 3 "sgd"

def main (argv : List String) : IO Unit := runCifar8SgdSched argv
