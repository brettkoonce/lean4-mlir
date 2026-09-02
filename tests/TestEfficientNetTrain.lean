import LeanMlir.VerifiedNets

/-! # E6 — EfficientNet-B0 artifact smoke (iree-compile over the COMMITTED bytes)

**This file no longer renders anything.** Both EfficientNet train-step artifacts are written by the
`#eval`s in `LeanMlir/Proofs/Codegen/EfficientNetRender.lean` as `pretty(provenGraph)`, and those
are their only writers:

| artifact | renderer |
|---|---|
| `verified_mlir/efficientnet_train_step.mlir` (SGD) | `efficientnetTrainStepFaithfulV` |
| `verified_mlir/efficientnet_adam_train_step.mlir` (AdamW) | `efficientnetAdamTrainStepFaithful` |

What remains here is the part `lake build` genuinely cannot do: **iree-compile the committed
bytes**, which needs the compiler on PATH. It reads them and throws if they are missing, rather
than quietly recreating them — recreating them is what made this a double writer in the first
place, twice, and neither time were the two renders the same function.

## Why both emitters that used to live here are gone

**The SGD one (retired §2a-quinquies).** The numeric tie run before deleting it FAILED, and that is
the finding, not a problem with the deletion:

| | loss cotangent | baked lr | effective lr on the MEAN loss |
|---|---|---|---|
| committed (`Proofs/`) | **sum**-CE — `softmax − onehot` straight into `dot_general` | 0.05 | 0.05 × 32 = **1.6** |
| retired (here) | **mean**-CE — `divide %dyr, dense<32.0>` | 0.1 | **0.1** |

`sgd-render-tie efficientnet <Proofs> 0.05 <tests> 0.1` reported all 262 parameters disagreeing at
norm-relative **0.96875 = 31/32** — the exact signature of `g_tests = g_Proofs / 32` — against a
bit-exact A-vs-A determinism floor. So this file was a live instance of the `RenderCifar8Sgd02`
hazard (§2a-quater): a `tests/` writer that, on elaboration, silently replaced a committed certified
artifact with **different hyperparameters**, here a 16× smaller effective step.

On the convention: the house style (`resnet34_train_step`, `vit_train_step`) is sum-CE with the mean
folded into lr, lr = 0.003125 = 0.1/32, and `convnext_train_step` reaches the same effective 0.1 by
spelling the mean explicitly. The committed EfficientNet SGD render sits at an effective **1.6** —
a *tuned* value, not a slip: `runs/efficientnet_verified_crop_gpu1.log` descends 40.6% → **87.81%**
over 80 epochs, matching README's 87.58%. Leave the number alone; it was only the absence of a
stated convention that misled. Recover the emitter from
`git show c992a94:tests/TestEfficientNetTrain.lean`.

**The AdamW one (retired 2026-07-28, the EfficientNet AdamW thread).** Unlike the SGD case this one
was a genuine peer of the certified render, and the tie **passed** before it was deleted.
`lake build efficientnet-adam-tie`, one AdamW step, all 12,166,117 returned floats: the forward
**BIT-EXACT** over the 98 BN batch statistics, `%loss` bit-exact, the gradient bit-exact, against a
bit-exact A-vs-A determinism floor. Recover it from
`git show c96bd36:tests/TestEfficientNetTrain.lean`; the retired *artifact* (what the tie ran as
side A) is `git show c96bd36:verified_mlir/efficientnet_adam_train_step.mlir`.

The `bnChannels` layout this file used to print lives in `efficientnetVerified.bnChannels`
(49 layers, `LeanMlir/VerifiedNets.lean`), and the certified AdamW render now derives its 49 stat
slots from the same forward traversal that computes them.

Run (rocm): export IREE_BACKEND=rocm; lake env lean tests/TestEfficientNetTrain.lean
-/


/-- Compile a COMMITTED artifact. Throws if it is missing: this file is not its writer, and
    recreating it here is exactly the double-writer race that shipped two different functions. -/
private def smoke (path dst label : String) : IO Unit := do
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/EfficientNetRender.lean; run \
`lake build LeanMlir.Proofs.Codegen.EfficientNetRender` first")
  tryCompile path dst label

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  smoke "verified_mlir/efficientnet_adam_train_step.mlir"
    ".lake/build/efficientnet_adam_ts.vmfb" "AdamW (committed bytes, not re-rendered)"
  smoke "verified_mlir/efficientnet_train_step.mlir"
    ".lake/build/efficientnet_train_step_v.vmfb" "SGD (committed bytes, not re-rendered)"

#eval main
