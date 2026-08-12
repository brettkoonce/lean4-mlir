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

Run (GPU): `.lake/build/bin/resnet34-verified data`

⚠⚠ **THIS DRIVER CANNOT PRODUCE A MEANINGFUL ACCURACY ON THIS NET, and the
number it prints is not a slow-learning curve — it is a constant predictor.**
Measured 2026-08-12 on XLA/CUDA, all ten epochs: `390/3925 = 9.936306%`, byte
identical every epoch, which is chance on Imagenette's ten classes.

The cause is structural. `resnet34Verified` carries 36 BatchNorm layers, and
**running-statistic threading lives only in `VerifiedNet.trainAdamSched`**, not
in `VerifiedNet.train`. So this driver trains parameters but never accumulates
BN running stats, then evaluates through `@resnet34_fwd` — which needs them —
instead of `@resnet34_fwd_eval`. The forward sees garbage statistics and returns
the same class for every image.

▶ **For a real number on this net use `resnet34-verified-adam`**, which goes
through `trainAdamSched`, announces `running-stats BN: 36 layers, 17024 stat
floats → eval via @resnet34_fwd_eval`, and reaches 22.2% top-1 after one epoch
against this driver's flat 9.94% after ten.

▶ This binary remains useful as a **structural smoke test**: it exercises
compile, the 110-output train step, and the packed-parameter round trip. Do not
quote its accuracy. Fixing it means teaching `.train` the same BN threading
`trainAdamSched` has, at which point this note comes out.

**One file, one binary, either lowerer.** The proven graph goes to whichever
trusted lowerer `$LEAN_MLIR_LOWERER` selects — XLA/PJRT by default, IREE with
`=iree` — resolved by dlopen at run time (`ffi/lowerer.h`).
-/

def resnet34Config : VerifiedConfig where
  epochs := 10
  lr     := 0.1

def main (argv : List String) : IO Unit :=
  resnet34Verified.train resnet34Config (argv.head?.getD "data")
