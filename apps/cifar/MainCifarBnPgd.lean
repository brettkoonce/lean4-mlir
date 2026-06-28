import LeanMlir.VerifiedNets

/-! # `cifar-bn-pgd` — phase-3 PGD attack on the verified CIFAR-10 CNN + (instance) BatchNorm

The BatchNorm rung of the robustness ladder. `cifar_bn` is `conv→BN→relu` ×4 (2 maxpools) → 3
denses, where "BN" is **instance normalization** (each image normalized over its spatial dims per
channel) — so the per-image gradient is clean and the deployed `cifar_bn_fwd` IS the attacked
forward (no running stats). `genCifarBnPgdStep` runs the full proven input-VJP to `dx`, including
the **BN grad-input 3-term formula** `dx = istd·(dxhat − meanₛ dxhat − xhat·meanₛ(dxhat·xhat))`
through all 4 BN layers — the first BN-backward in the attack path.

**No certificate** (the cert is N/A here): instance norm absorbs the conv weight scale, and its
Lipschitz is data-dependent (`γ·istd`) — a separate problem from the conv-product cert. This is the
attack rung only: clean → adversarial collapse.

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm .lake/build/bin/cifar-bn-pgd data`
-/

def cifarBnPgdConfig : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit := do
  -- CIFAR_PGD_EPOCHS overrides the epoch count (cheap smoke test); absent → full 12.
  let ep := ((← IO.getEnv "CIFAR_PGD_EPOCHS").bind (·.toNat?)).getD cifarBnPgdConfig.epochs
  cifarBnVerified.attackPgdCifarBn { cifarBnPgdConfig with epochs := ep } (argv.head?.getD "data")
