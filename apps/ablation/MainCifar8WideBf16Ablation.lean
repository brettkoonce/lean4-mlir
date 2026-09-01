import LeanMlir.VerifiedNets

/-! # `cifar8wb-bf16-ablation` — wide head, BATCHED render, bf16

One arm of the §4.3 "Lever 3: precision" sweep (`planning/cifar_lowprec_stability.md` §5.2).
Runs SGD / Nesterov / AdamW in sequence on the wide-head (d1=512) net that Levers 1–2 already
measure, so the new lever is read against the existing table rather than a different network.

Same hyperparameters as `cifar8w-ablation` (SGD lr 0.1, momentum μ0.9 lr 0.02, AdamW lr 1e-3,
3-epoch warmup + cosine, 40 epochs, bs 128). bf16 reaches the FORWARD, the input-VJP and the weight gradients — 23/23 convolutions — because the batched family is the one the 27 bf16 ops were built for. ⚠ Expect no speedup (§5.3: 0.87× at these shapes); this is a numerics result.

Run: `LEAN_MLIR_LOWERER=xla .lake/build/bin/cifar8wb-bf16-ablation data`
-/

def cfg : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit := do
  let d := argv.head?.getD "data"
  IO.println "════════ cifar8wb bf16 — SGD (lr 0.1) ════════"
  cifar8wbVerified.toNet.trainAdamSched cfg d 0.1 0.9 0.999 0 "bf16sgd" 1.0 1.0
  IO.println "════════ cifar8wb bf16 — Nesterov momentum (μ.9, lr 0.02) ════════"
  cifar8wbVerified.toNet.trainAdamSched cfg d 0.02 0.9 0.999 0 "bf16mom" 1.0 1.0
  IO.println "════════ cifar8wb bf16 — AdamW (lr 1e-3) ════════"
  cifar8wbVerified.toNet.trainAdamSched cfg d 0.001 0.9 0.999 0 "bf16adam" 1.0 1.0
