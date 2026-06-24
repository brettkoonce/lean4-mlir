import LeanMlir.VerifiedNets

/-! # `cifar-e4m3-verified` — fp8 (E4M3) CIFAR-10 training on the VERIFIED codegen

The low-precision sibling of `MainCifarVerified` (Chapter 5, no BatchNorm):
`conv 3→32 → relu → conv 32→32 → relu → maxpool → conv 32→64 → relu →
conv 64→64 → relu → maxpool → flatten → dense 4096→512 → relu → dense 512→512 →
relu → dense 512→10` + softmax-CE. Trains on the *same* proof-rendered StableHLO
(`verified_mlir/cifar_train_step.mlir`, `cifarCnn_has_vjp_at`, audited
3-axiom-clean), but each step projects every weight onto the **E4M3** grid — conv
kernels per output channel, dense weights per output column — and the input
per-tensor, accumulating in fp32 inside the verified kernel, with **fp32 master
weights** (`VerifiedNet.trainE4M3`, the same packed driver the MLP/CNN fp8
trainers use; CIFAR just plugs in via its `.cifar` loader).

Honest scope: **fp8 weights + fp8 input, fp32 intermediate activations** — the
relu/maxpool/flatten activations and backward cotangents live inside the fused
kernel, so host-side byte-prep can't reach them (depth > 1). All-operand fp8
needs an in-graph E4M3 round op (the next codegen-level step). E4M3 grid: pure-Lean
`LeanMlir/E4M3Quant.lean` (same grid as the numpy oracle).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar-e4m3-verified data`
-/

def cifarE4M3Config : VerifiedConfig where
  epochs    := 10
  batchSize := 128

def main (argv : List String) : IO Unit :=
  (cifarVerified.toNet).trainE4M3 cifarE4M3Config (argv.head?.getD "data")
