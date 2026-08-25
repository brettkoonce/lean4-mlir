import LeanMlir.VerifiedNets

/-! # `cifar8-bf16-verified` — 8-conv CIFAR-10 CNN, **bf16 forward convs**, plain SGD (const lr)

The bf16 arm of the §5.2 optimizer sweep (`planning/cifar_lowprec_stability.md`). Identical
net, identical init, identical hyperparameters to its fp32 peer `cifar8-verified` — the ONLY
difference is that `cifar8Bf16Verified`'s slug points `mkSession` at
`verified_mlir/cifar8_bf16_train_step.mlir`, emitted by the SAME renderer with `bf16 := true`.

⭐ The point is the ORDERING, not the accuracy: the CIFAR chapter shows optimizers scale
(SGD < AdamW < Nesterov), and running that ladder at fp32 / bf16 / fp8 shows the MATH scales
with it. A controlled comparison needs the model held fixed, which a slug-only change gives.

⚠ bf16 is FORWARD-ONLY — cifar8's backward runs on the per-example `convBack`/`dotOut`, which
have no bf16 twin (§4.1). ⚠⚠ Expect NO speedup: §5.3 measured bf16 at 0.87× across cifar8's
conv stack, since these convs are launch-bound at ≤32 channels. Eval runs in f32.

Run: `LEAN_MLIR_LOWERER=xla .lake/build/bin/cifar8-bf16-verified data`
-/

def cifar8Bf16Config : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit :=
  cifar8Bf16Verified.train cifar8Bf16Config (argv.head?.getD "data")
