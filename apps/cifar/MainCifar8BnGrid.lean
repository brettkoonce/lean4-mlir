import LeanMlir.VerifiedNets

/-! # `cifar8-bn-grid` — FC-head width sweep of the 8-conv CIFAR CNN + per-channel BN, AdamW

The CIFAR peer of `mnist-cnn-grid`: the conv feature extractor is held fixed
(8× conv→BN→relu, channels `[16,16,32,32]`, 4 pools) and only the dense classifier head is
swept — `flatten(128) → dense 128→d → relu → dense d→d → relu → dense d→10`. Reads the head
width `d` from argv and trains `cifar8BnG d` on the width-slugged verified renders
`verified_mlir/cifar8_bn_{d}_{adam_train_step,fwd}.mlir` (rendered offline by
`tests/TestCifar8AdamTrain.lean`, whose `D1` is now parametric), via the packed-`[θ|m|v]`
AdamW driver `VerifiedNet.trainAdamSched` (variant `"adam"`, the same one
`cifar8-bn-verified-adam` uses). Per-channel BN ⇒ train=eval (no running stats).

baseLR 1e-3, β₁ .9, β₂ .999, 3-epoch warmup + cosine decay. Epochs default 25 (argv 2).

Run (GPU): `IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 .lake/build/bin/cifar8-bn-grid 128 25 data`
-/

def main (argv : List String) : IO Unit := do
  match argv with
  | ds :: rest =>
    let some d := ds.toNat? | throw (.userError s!"bad FC-head width: {ds}")
    let epochs := (rest.head?.bind (·.toNat?)).getD 25
    let dataDir := rest[1]?.getD "data"
    let cfg : VerifiedConfig := { epochs := epochs, batchSize := 128 }
    (cifar8BnG d).toNet.trainAdamSched cfg dataDir 0.001 0.9 0.999 3 "adam"
  | _ => throw (.userError "usage: cifar8-bn-grid <fc-head-width> [epochs] [dataDir]")
