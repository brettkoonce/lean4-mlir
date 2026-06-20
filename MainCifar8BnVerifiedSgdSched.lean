import LeanMlir.VerifiedNets

/-! # `cifar8-bn-verified-sgdsched` — plain SGD (BN net) on the momentum/Adam pipeline

BN peer of `cifar8-verified-sgdsched`: plain SGD `θ←θ−lr·∇` through `trainAdamSched`
(`variant := "sgd"`) with the same per-epoch shuffle + random hflip + cosine-warmup schedule
as the BN momentum/Adam runs, so the optimizer is the only free variable. 38 params
(8× per-channel BN γ/β); per-example BN ⇒ train=eval, eval via `@cifar8_bn_fwd`.

baseLR 0.1, 3-epoch warmup + cosine decay.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar8-bn-verified-sgdsched data`
-/

def cifar8BnSgdSchedConfig : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifar8BnVerified.toNet.trainAdamSched cifar8BnSgdSchedConfig (argv.head?.getD "data") 0.1 0.9 0.999 3 "sgd"
