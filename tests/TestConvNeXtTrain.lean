import LeanMlir.VerifiedNets

/-! # ConvNeXt-T artifact smoke (iree-compile over the COMMITTED bytes)

**This file no longer renders anything.** Both ConvNeXt train-step artifacts are written by the
`#eval`s in `LeanMlir/Proofs/Codegen/ConvNeXtRender.lean` as `pretty(provenGraph)`, and those are
their only writers:

| artifact | renderer |
|---|---|
| `verified_mlir/convnext_train_step.mlir` (SGD) | `convNextTrainStepFaithfulV` |
| `verified_mlir/convnext_adam_train_step.mlir` (AdamW) | `convNextAdamTrainStepFaithful` |

What remains is the part `lake build` cannot do: **iree-compile the committed bytes**, which needs
the compiler on PATH. It reads them and throws if they are missing rather than quietly recreating
them — recreating them is what made this a double writer in the first place.

## Why both emitters that used to live here are gone

**The SGD one (retired §2a-quinquies).** Its tie came back **gradient BIT-EXACT** over 27,811,542
coordinates against a structurally different emitter (4483 vs 3590 lines), so that deletion was
provably lossless.

**The AdamW one (retired 2026-07-28, §2f).** `lake build convnext-adam-tie`, one AdamW step, all
83,434,629 returned floats: `%loss` **BIT-EXACT**, and 179 of 180 parameter gradients bit-exact.

The one that differs is worth recording, because it is a *conditioning* result and not a defect.
`s3b2lg` — the last block's per-channel layer-scale γ — disagrees at norm-rel 6.9e-3. Its gradient is
`reduce[0,2,3](project-out ⊙ block-cotangent)`, a **cancelling** sum (|Σ|/Σ|·| ≈ 0.09 across
channels, far worse within one), and it is the only parameter in the block whose gradient reads a
forward *value* rather than a cotangent. Running the reference render against **itself** on the same
batch in reversed row order — semantics-preserving, since every gradient here sums over the batch —
moves it by **6.7e-3**, i.e. as much as the two emitters differ, and disturbs **6** parameters where
the two emitters disturb **1**.

So the gate is calibrated against that reorder control rather than an absolute bound, and it gates
the **spread** as well as the magnitude. The spread gate is not decoration: perturbing the cotangent
(α 0.1 → 0.11) clears the 4×-magnitude gate at 9.1e-3 while disturbing **178/180** parameters.
Conditioning noise is local to the ill-conditioned op; a different function is global.

Recover the retired emitter from `git show b94e8e9:tests/TestConvNeXtTrain.lean`; the retired
*artifact* is `git show b94e8e9:verified_mlir/convnext_adam_train_step.mlir`.

Run (rocm): export IREE_BACKEND=rocm; lake env lean tests/TestConvNeXtTrain.lean
-/

/-- iree-compile smoke that degrades gracefully when the compiler isn't on PATH (the render +
    write already happened, so the artifact exists regardless). -/
private def tryCompile (src dst label : String) : IO Unit := do
  try
    let cargs ← ireeCompileArgs src dst
    let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
    if r.exitCode != 0 then IO.eprintln s!"iree-compile ({label}) FAILED:\n{r.stderr.take 3000}"
    else IO.println s!"{label} iree-compile OK → {src}"
  catch e => IO.eprintln s!"iree-compile ({label}) skipped (compiler unavailable): {e}"

/-- Compile a COMMITTED artifact. Throws if it is missing: this file is not its writer, and
    recreating it here is exactly the double-writer race that shipped two different functions. -/
private def smoke (path dst label : String) : IO Unit := do
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/ConvNeXtRender.lean; run `lake build LeanMlir.Proofs.Codegen.ConvNeXtRender` \
first")
  tryCompile path dst label

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  smoke "verified_mlir/convnext_adam_train_step.mlir"
    ".lake/build/convnext_adam_ts.vmfb" "AdamW (committed bytes, not re-rendered)"
  smoke "verified_mlir/convnext_train_step.mlir"
    ".lake/build/convnext_train_step_v.vmfb" "SGD (committed bytes, not re-rendered)"

#eval main
