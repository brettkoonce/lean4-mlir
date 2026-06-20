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
-/

def cifar8SgdSchedConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  -- baseLR 0.1 (β args unused by the plain-SGD update), 3-epoch warmup + cosine.
  cifar8Verified.toNet.trainAdamSched cifar8SgdSchedConfig (argv.head?.getD "data") 0.1 0.9 0.999 3 "sgd"
