import LeanMlir.VerifiedNets

/-! # `resnet34-verified` — train a real ResNet-34 on the VERIFIED-rendered codegen

Chapter 6 Milestone B9: the whole 34-layer ResNet whose architecture VJP is the audited
parametric skeleton `Proofs.resnet34_has_vjp_at` (depth = a `List.length`, folded from
`vjp_comp_at`/`vjp_chain_at`). IMAGENETTE 3×224×224 (paper-native ImageNet resolution):

  conv(3→64,7×7,stride-2,SAME) → BN → relu → maxpool(112→56) →
  stage1: 3 identity blocks @64           (56×56) →
  stage2: downsample 64→128 + 3 identity  (28×28) →
  stage3: downsample 128→256 + 5 identity (14×14) →
  stage4: downsample 256→512 + 2 identity (7×7)   →
  global-average-pool → dense 512→10 + softmax-CE

The model is the `resnet34Verified` `VerifiedNetSpec` (in `LeanMlir.VerifiedNets`); its
derived 146-param layout is kernel-`#guard`ed against the audited `ResNet34Layout`. Trains
on `verified_mlir/resnet34_{train_step,fwd}.mlir` — **both** rendered by
`LeanMlir/Proofs/Codegen/ResNet34Render.lean` as `pretty(provenGraph)` — through the
packed-params `VerifiedNet.train` driver (`mlpTrainStepV`, per-channel BN, He-init,
mean-loss SGD lr=0.1).

BN here is **per-channel, per-example** (μ/σ over `H·W`, reduce `[2,3]`) — instance-norm-shaped,
not batch-norm. That makes train and eval the same function, so no running stats are needed and
`resnet34_fwd.mlir` is a byte-prefix of `resnet34_train_step.mlir`. Until 2026-07-27 the forward
was a separate hand-written render that normalised over the **batch** (reduce `[0,2,3]`,
n = B·H·W), i.e. eval scored a different function than training optimised; see
`planning/xla_pjrt_handoff.md` §2a. (The AdamW sibling `resnet34-verified-adam` is the true
batch-norm path, and evals through `@resnet34_fwd_eval` with EMA'd running stats.)

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/resnet34-verified data`
-/

def resnet34Config : VerifiedConfig where
  epochs := 10
  lr     := 0.1

def main (argv : List String) : IO Unit :=
  resnet34Verified.train resnet34Config (argv.head?.getD "data")
