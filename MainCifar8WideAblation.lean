import LeanMlir.VerifiedNets

/-! # `cifar8w-ablation` — wide-head (d1=512) cifar8, no-BN, all three optimizers

The MNIST-style wide-head (2×512 dense) peer of the cifar8 optimizer ablation. Runs the
`cifar8wVerified` net (8-conv backbone + 128→512→512→10 head, 373,626 floats) three ways in
sequence — SGD / Nesterov-momentum / AdamW — all on the identical controlled pipeline
(per-epoch shuffle + hflip + cosine-warmup) via `trainAdamSched`, so only the optimizer
varies. Renders: `tests/TestCifar8WideTrain.lean`. 40 epochs, bs 128.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8w-ablation data`
-/

def cfg : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit := do
  let d := argv.head?.getD "data"
  IO.println "════════ cifar8w (wide head) — SGD (lr 0.1) ════════"
  cifar8wVerified.toNet.trainAdamSched cfg d 0.1 0.9 0.999 3 "sgd"
  IO.println "════════ cifar8w (wide head) — Nesterov momentum (μ.9, lr 0.02) ════════"
  cifar8wVerified.toNet.trainAdamSched cfg d 0.02 0.9 0.999 3 "mom"
  IO.println "════════ cifar8w (wide head) — AdamW (lr 1e-3) ════════"
  cifar8wVerified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 3 "adam"
