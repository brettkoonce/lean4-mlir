import LeanMlir.VerifiedNets

/-! # `mnist-linear-e4m3-verified` — fp8 (E4M3) training on the VERIFIED codegen

The low-precision sibling of `MainMnistLinearVerified`. Trains the Chapter-2 linear
classifier on the *same* proof-rendered StableHLO (`verified_mlir/linear_train_step.mlir`
= `Proofs.StableHLO.linearTrainStepModuleV`, audited 3-axiom-clean), but each step
projects the weights (per-output-column) and activations (per-tensor) onto the **E4M3
fp8 grid** before the matmul, accumulates in fp32 inside the verified kernel, and keeps
**fp32 master weights** (the `u_leaf = E4M3`, `u_acc = fp32` mixed model).

This is the runnable Lean side of the fp8 story whose structural faithfulness is proved
by `Proofs/E4M3FaithfulPoC.lean` (§3b render-tie) and whose accuracy bound is
`Proofs/FloatBridge.lean` (§3c argmax-preservation). The E4M3 quantizer is pure Lean
(`LeanMlir/E4M3Quant.lean`) — no fp8 hardware or fp8 StableHLO type; `q` runs host-side
as operand byte-prep, exactly the §3b model. The numpy oracle
(`scripts/mnist_e4m3_demo.py` / `mnist_e4m3_train_demo.py`) uses the identical grid.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-linear-e4m3-verified data`
-/

def linearE4M3Config : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit :=
  (linearVerified.toNet).trainLinearE4M3 linearE4M3Config (argv.head?.getD "data")
