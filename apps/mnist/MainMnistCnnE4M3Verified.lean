import LeanMlir.VerifiedNets

/-! # `mnist-cnn-e4m3-verified` — fp8 (E4M3) CNN training on the VERIFIED codegen

The low-precision sibling of `MainMnistCnnVerified`. Trains the Chapter-4 CNN
(`conv 1→32 → relu → conv 32→32 → relu → maxpool → flatten → dense 6272→512 →
relu → dense 512→512 → relu → dense 512→10`) on the *same* proof-rendered
StableHLO (`verified_mlir/cnn_train_step.mlir`), but each step projects every
weight onto the **E4M3** grid — conv kernels per output channel, dense weights
per output column — and the input per-tensor, accumulating in fp32 inside the
verified kernel, with **fp32 master weights** (`VerifiedNet.trainE4M3`).

Honest scope: **fp8 weights + fp8 input, fp32 intermediate activations** — the
relu/maxpool/flatten activations feeding `conv2`/the dense head live inside the
fused kernel, so host-side byte-prep can't quantize them (depth > 1). Those need
in-graph E4M3 ops (the next codegen-level step). E4M3 grid: pure-Lean
`LeanMlir/E4M3Quant.lean`.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-cnn-e4m3-verified data`
-/

def cnnE4M3Config : VerifiedConfig where
  epochs    := 10
  batchSize := 128

def main (argv : List String) : IO Unit :=
  (cnnVerified.toNet).trainE4M3 cnnE4M3Config (argv.head?.getD "data")
