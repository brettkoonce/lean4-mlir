import LeanMlir.VerifiedNets

/-! # `cifar8wb-bn-ablation` — wide head, BATCHED render, BatchNorm, f32 and bf16

Chapter 4's **Lever 3 on the normalized net** (`planning/bf16_batchnorm.md`). The existing bf16
lever could only be measured on the un-normalized net, because the 27 bf16 ops are batched-only
and the BN train step was rendered per-example. `cifar8BnTrainStepFaithfulB` moves the BN net onto
the batched family, so bf16 reaches all 23 convolutions — forward, input-VJP and weight gradients.

**Six arms, three optimizers × two precisions**, from ONE renderer: precision is the only thing
that moves inside a pair. ⚠ BatchNorm itself stays f32 and per-example in both arms — that is
mixed precision as every ImageNet net in this repo practises it, and it is the recipe Chapter 5
actually uses.

Same hyperparameters as `cifar8w-bn-ablation`: SGD lr 0.1, Nesterov μ0.9 lr 0.02, AdamW lr 1e-3,
**constant** learning rate (`warmupEpochs = 0`, `expDecayRate = 1.0`), 40 epochs, bs 128.

⚠ Expect no speedup — bf16 measures 0.87× at cifar8's conv shapes. This is a stability and
accuracy result, never a throughput one.

Run: `LEAN_MLIR_LOWERER=xla .lake/build/bin/cifar8wb-bn-ablation data`
-/

def cfg : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit := do
  let d := argv.head?.getD "data"
  IO.println "════════ cifar8wb-bn f32 — SGD (lr 0.1) ════════"
  cifar8wbBnVerified.toNet.trainAdamSched cfg d 0.1 0.9 0.999 0 "sgd" 1.0 1.0
  IO.println "════════ cifar8wb-bn f32 — Nesterov momentum (μ.9, lr 0.02) ════════"
  cifar8wbBnVerified.toNet.trainAdamSched cfg d 0.02 0.9 0.999 0 "mom" 1.0 1.0
  IO.println "════════ cifar8wb-bn f32 — AdamW (lr 1e-3) ════════"
  cifar8wbBnVerified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 0 "adam" 1.0 1.0
  IO.println "════════ cifar8wb-bn bf16 — SGD (lr 0.1) ════════"
  cifar8wbBnVerified.toNet.trainAdamSched cfg d 0.1 0.9 0.999 0 "bf16sgd" 1.0 1.0
  IO.println "════════ cifar8wb-bn bf16 — Nesterov momentum (μ.9, lr 0.02) ════════"
  cifar8wbBnVerified.toNet.trainAdamSched cfg d 0.02 0.9 0.999 0 "bf16mom" 1.0 1.0
  IO.println "════════ cifar8wb-bn bf16 — AdamW (lr 1e-3) ════════"
  cifar8wbBnVerified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 0 "bf16adam" 1.0 1.0
