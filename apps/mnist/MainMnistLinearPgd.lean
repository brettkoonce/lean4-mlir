import LeanMlir.VerifiedNets

/-! # `mnist-linear-pgd` — phase-3 PGD adversarial attack on the verified linear net

Trains the Chapter-2 linear classifier on the proof-rendered StableHLO (same as
`mnist-linear-verified`), then runs an L∞ **PGD adversarial attack** through the real
IREE pipeline: each attack step's input gradient `dx = (softmax(xW+b) − onehot)·Wᵀ` is
computed by a StableHLO kernel (the proven linear input-VJP, `Proofs.mlpInputGrad`'s
1-layer case) on the GPU — NOT host autodiff. The whole PGD step (forward, gradient,
sign-step, eps-ball projection, [0,1] clip) runs as one IREE kernel; the host iterates.

This is the phase-3 counterpart to `jax/demos/pgd_mnist.py` (which trained a throwaway
JAX net): here the attack hits the *actual verified net* through the *real codegen path*.
See `planning/robustness.md`.

Run (GPU): `IREE_BACKEND=rocm IREE_CHIP=gfx1100 .lake/build/bin/mnist-linear-pgd data`
-/

def linearConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit :=
  linearVerified.attackPgd linearConfig (argv.head?.getD "data")
