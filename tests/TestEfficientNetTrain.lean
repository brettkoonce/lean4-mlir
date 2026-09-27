import LeanMlir.Verified.Train

/-! # EfficientNet-B0 artifact smoke (iree-compile over the COMMITTED bytes)

**This file renders nothing.** Both EfficientNet train-step artifacts are written by the
`#eval`s in `LeanMlir/Proofs/Codegen/EfficientNetRender/Basic.lean` as `pretty(provenGraph)`, and those
are their only writers:

| artifact | renderer |
|---|---|
| `verified_mlir/efficientnet_train_step.mlir` (SGD) | `efficientnetTrainStepFaithfulV` |
| `verified_mlir/efficientnet_adam_train_step.mlir` (AdamW) | `efficientnetAdamTrainStepFaithful` |

What remains here is the part `lake build` genuinely cannot do: **iree-compile the committed
bytes**, which needs the compiler on PATH. It reads them and throws if they are missing, rather
than quietly recreating them — recreating them would make this file a second writer, and a
second writer need not render the same function.

## The SGD render's learning-rate convention

The committed SGD render is **sum**-CE — `softmax − onehot` straight into `dot_general` — at a
baked lr of 0.05, an effective lr of 0.05 × 32 = **1.6** on the MEAN loss. The house style
(`resnet34_train_step`, `vit_train_step`) is sum-CE with the mean folded into lr, lr = 0.003125 =
0.1/32, and `convnext_train_step` reaches the same effective 0.1 by spelling the mean explicitly.
The EfficientNet SGD render's effective **1.6** is a *tuned* value, not a slip:
`runs/efficientnet_verified_crop_gpu1.log` descends 40.6% → **87.81%** over 80 epochs, matching
README's 87.58%. Leave the number alone. A `tests/` writer that re-rendered it as **mean**-CE at
lr 0.1 would, on elaboration, silently replace a committed certified artifact with **different
hyperparameters** (a 16× smaller effective step); `sgd-render-tie` reads that split as every
parameter disagreeing at norm-relative **0.96875 = 31/32**, the signature of `g = g_committed / 32`.

The `bnChannels` layout lives in `efficientnetVerified.bnChannels`
(49 layers, `LeanMlir/Verified/NetsCore.lean`), and the certified AdamW render derives its 49 stat
slots from the same forward traversal that computes them.

Run (needs iree-compile on PATH): lake env lean tests/TestEfficientNetTrain.lean
-/


/-- Compile a COMMITTED artifact. Throws if it is missing: this file is not its writer, and
    recreating it here is the double-writer race that can ship two different functions. -/
private def smoke (path dst label : String) : IO Unit := do
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/EfficientNetRender/Basic.lean; run \
`lake build LeanMlir.Proofs.Codegen.EfficientNetRender.Basic` first")
  tryCompile path dst label

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  smoke "verified_mlir/efficientnet_adam_train_step.mlir"
    ".lake/build/efficientnet_adam_ts.vmfb" "AdamW (committed bytes, not re-rendered)"
  smoke "verified_mlir/efficientnet_train_step.mlir"
    ".lake/build/efficientnet_train_step_v.vmfb" "SGD (committed bytes, not re-rendered)"

#eval main
