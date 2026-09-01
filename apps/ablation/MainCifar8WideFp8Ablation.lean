import LeanMlir.VerifiedNets

/-! # `cifar8w-fp8-ablation` — wide head, fp8 (E4M3, emulated)

One arm of the §4.3 "Lever 3: precision" sweep (`planning/cifar_lowprec_stability.md` §5.2).
Runs SGD / Nesterov / AdamW in sequence on the wide-head (d1=512) net that Levers 1–2 already
measure, so the new lever is read against the existing table rather than a different network.

Same hyperparameters as `cifar8w-ablation` (SGD lr 0.1, momentum μ0.9 lr 0.02, AdamW lr 1e-3,
3-epoch warmup + cosine, 40 epochs, bs 128). ⚠ fp8 here is HOST-SIDE: weights and input are projected onto the E4M3 grid, the graph stays f32, accumulate is f32 and the master copy is f32. No f8 type reaches the StableHLO, so this arm rides the SAME artifacts as the f32 one.

Run: `LEAN_MLIR_LOWERER=xla .lake/build/bin/cifar8w-fp8-ablation data`
-/

def cfg : VerifiedConfig where
  epochs    := 40
  batchSize := 128

def main (argv : List String) : IO Unit := do
  let d := argv.head?.getD "data"
  IO.println "════════ cifar8w fp8 E4M3 — SGD (lr 0.1) ════════"
  cifar8wVerified.toNet.trainAdamSchedE4M3 cfg d 0.1 0.9 0.999 0 "sgd" 1.0
  IO.println "════════ cifar8w fp8 E4M3 — Nesterov momentum (μ.9, lr 0.02) ════════"
  cifar8wVerified.toNet.trainAdamSchedE4M3 cfg d 0.02 0.9 0.999 0 "mom" 1.0
  IO.println "════════ cifar8w fp8 E4M3 — AdamW (lr 1e-3) ════════"
  cifar8wVerified.toNet.trainAdamSchedE4M3 cfg d 0.001 0.9 0.999 0 "adam" 1.0
