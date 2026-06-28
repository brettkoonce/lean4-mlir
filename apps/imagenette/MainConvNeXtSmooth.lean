import LeanMlir.VerifiedNets

/-! # `convnext-smooth` — randomized-smoothing certificate on the verified ConvNeXt-T (Imagenette 224²)

The deep-net / real-resolution rung of the smoothing certificate (`planning/robustness_ladder.md` §3,
Cohen–Rosenfeld–Kolter 2019). Same forward-only Monte-Carlo procedure as the MNIST/CIFAR `*-smooth`
demos, now on a 224² ConvNeXt-T — the depth-independence claim taken to a real ImageNet-scale net,
where the Lipschitz product is hopeless. ConvNeXt uses **LayerNorm** (per-sample, no BN running
stats), so the rendered `convnext_fwd.mlir` processes each noisy sample independently — exactly the
condition randomized smoothing needs — and the generic `smoothCertify` driver applies unchanged
(noise-augmented SGD on `convnext_train_step.mlir`, then certify through `convnext_fwd`).

`batchSize` is pinned to **32** (the baked static batch of `convnext_fwd.mlir`). σ is selected by
`SMOOTH_SIGMA_MILLI` (σ×1000, e.g. 250 → σ=0.25) so the two σ values can run on the two gfx1100 GPUs
in parallel; unset → the default `[0.25, 0.5]` sweep. Knobs: `SMOOTH_EPOCHS`, `SMOOTH_N`,
`SMOOTH_MAXCERT` (the 224² forward is heavy — start light).

Run (GPU): `PATH=$PWD/.venv/bin:$PATH IREE_BACKEND=rocm SMOOTH_N=2000 SMOOTH_MAXCERT=50 \
  SMOOTH_EPOCHS=12 SMOOTH_SIGMA_MILLI=500 .lake/build/bin/convnext-smooth data`
-/

def convnextSmoothConfig : VerifiedConfig where
  epochs    := 15
  batchSize := 32          -- MUST match the baked batch of convnext_fwd.mlir

def main (argv : List String) : IO Unit := do
  let ep := ((← IO.getEnv "SMOOTH_EPOCHS").bind (·.toNat?)).getD convnextSmoothConfig.epochs
  let sigmas := match (← IO.getEnv "SMOOTH_SIGMA_MILLI").bind (·.toNat?) with
    | some m => [m.toFloat / 1000.0]
    | none   => [0.25, 0.5]
  convnextVerified.smoothCertify { convnextSmoothConfig with epochs := ep } (argv.head?.getD "data") sigmas
