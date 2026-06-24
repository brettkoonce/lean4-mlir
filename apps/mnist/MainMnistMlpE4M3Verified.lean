import LeanMlir.VerifiedNets

/-! # `mnist-mlp-e4m3-verified` — fp8 (E4M3) MLP training on the VERIFIED codegen

The low-precision sibling of `MainMnistMlpVerified`. Trains the Chapter-3 MLP
(`dense 784→512 → relu → dense 512→512 → relu → dense 512→10`) on the *same*
proof-rendered StableHLO (`verified_mlir/mlp_train_step.mlir`), but each step
projects every weight matrix onto the **E4M3** grid (per output column) and the
input per-tensor, accumulating in fp32 inside the verified kernel, with **fp32
master weights** (`VerifiedNet.trainE4M3`).

Honest scope: this is **fp8 weights + fp8 input, fp32 intermediate activations**
— the relu/hidden activations feeding `W1`/`W2` live inside the fused kernel, so
host-side byte-prep can't quantize them (depth > 1). Quantizing those needs
in-graph E4M3 ops, the next codegen-level step. The E4M3 grid is the pure-Lean
`LeanMlir/E4M3Quant.lean` (same grid as the numpy oracle).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-e4m3-verified data`
-/

def mlpE4M3Config : VerifiedConfig where
  epochs    := 12
  batchSize := 128

def main (argv : List String) : IO Unit :=
  (mlpVerified.toNet).trainE4M3 mlpE4M3Config (argv.head?.getD "data")
