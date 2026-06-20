import LeanMlir.VerifiedNets

/-! # `cifar8w-bn-ablation` — wide-head (d1=512) cifar8 **+ per-channel BN**, all three optimizers

BN peer of `cifar8w-ablation`: the `cifar8wBnVerified` net (8× conv→BN→relu, 128→512→512→10
head) run SGD / Nesterov-momentum / AdamW in sequence on the same controlled pipeline
(shuffle + hflip + cosine-warmup) via `trainAdamSched`. Per-channel BN is per-example ⇒
train=eval (eval via `@cifar8w_bn_fwd`). Renders: `tests/TestCifar8WideTrain.lean`. 40 ep, bs 128.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8w-bn-ablation data`
-/

def cfg : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit := do
  let d := argv.head?.getD "data"
  IO.println "════════ cifar8w-BN (wide head) — SGD (lr 0.1) ════════"
  cifar8wBnVerified.toNet.trainAdamSched cfg d 0.1 0.9 0.999 3 "sgd"
  IO.println "════════ cifar8w-BN (wide head) — Nesterov momentum (μ.9, lr 0.02) ════════"
  cifar8wBnVerified.toNet.trainAdamSched cfg d 0.02 0.9 0.999 3 "mom"
  IO.println "════════ cifar8w-BN (wide head) — AdamW (lr 1e-3) ════════"
  cifar8wBnVerified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 3 "adam"
