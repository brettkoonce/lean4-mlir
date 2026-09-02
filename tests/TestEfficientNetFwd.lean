import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # E6 — EfficientNet-B0 forward: the `iree-compile` smoke over the COMMITTED bytes

**The hand-written emitter that used to live here is RETIRED (2026-07-28).**
`verified_mlir/efficientnet_fwd.mlir` and `verified_mlir/efficientnet_fwd_eval.mlir` are now
written by `LeanMlir/Proofs/Codegen/EfficientNetRender.lean`'s `efficientnetFwd{,Eval}FaithfulV` —
`pretty(provenGraph)`, both off the single `enetFwdChain` the train steps differentiate — and those
`#eval`s are their only writers. This file keeps only the part `lake build` genuinely cannot do:
running `iree-compile`, which needs the compiler on PATH.

The net, unchanged (EfficientNet-B0, Tan & Le 2019, Imagenette 224², 10 classes):

  stem : 3×3 stride-2 conv (3→32) + BN + swish   (224→112)
  B0 stages [expand t, channels c, repeats n, stride s, kernel k]:
    s1: t1 c16  n1 s1 k3   (MBConv1, NO expand conv)   @112
    s2: t6 c24  n2 s2 k3                                112→56
    s3: t6 c40  n2 s2 k5                                56→28
    s4: t6 c80  n3 s2 k3                                28→14
    s5: t6 c112 n3 s1 k5                                @14
    s6: t6 c192 n4 s2 k5                                14→7
    s7: t6 c320 n1 s1 k3                                @7
  head : 1×1 conv (320→1280) + BN + swish  →  GAP → dense(1280→10)
  16 MBConv layers total; SE (ratio 0.25 of block-input ch) in every block.

**The two artifacts differ in ONE thing and it is now structural**: `@efficientnet_fwd` renders the
chain at `BnMode.train` (batch statistics reduced out of the activation, `bnBatchF`) and
`@efficientnet_fwd_eval` at `.eval` (frozen per-channel running stats as graph inputs, the new
`bnEval` descriptor). Previously they were two independently hand-written spellings of that
distinction — precisely the shape of the ResNet-34 §2a bug, a net trained with one normalisation
and scored with the other.

**Why the emitter is gone rather than dormant** (§2b-quater): a second emitter that can write is one
more thing to drift. The swap was licensed by `lake build fwd-tie`:

    git show 17413f0:verified_mlir/efficientnet_fwd.mlir      > /tmp/r_fwd.mlir
    git show 17413f0:verified_mlir/efficientnet_fwd_eval.mlir > /tmp/r_eval.mlir
    .lake/build/bin/fwd-tie efficientnet        /tmp/r_fwd.mlir  verified_mlir/efficientnet_fwd.mlir
    .lake/build/bin/fwd-tie efficientnet --eval /tmp/r_eval.mlir verified_mlir/efficientnet_fwd_eval.mlir

both **BIT-EXACT, 320/320 logits**, on non-degenerate output, with interfaces positionally identical
down to the argument NAMES (263 in / 361 in). Shown capable of failing, two ways:

* perturbing the 49 BN ε constants (1e-5 → 1e-3) fires the forward tie at rel 4.4e-1, 0/320
  bit-exact;
* **swapping which stat slot two BN sites READ** (`b13en` ↔ `b14en`, same 1152 channels so the
  types still match, signature order untouched) fires the eval tie at rel 5.6e-1. That is §2e's
  "a misaligned stat slot is silent" hazard, now covered by an executable check rather than by
  care — the arities still match, and nothing but the numbers can tell.

  The first attempt at that control was a no-op worth remembering: renaming the parameter in the
  signature AND at the use site is an alpha-rename, and came back bit-identical. The func-arg
  POSITION is what binds a statistic to a slot.

Recover the retired emitter from `git show 17413f0:tests/TestEfficientNetFwd.lean`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestEfficientNetFwd.lean
-/

open Proofs Proofs.StableHLO


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
  smoke "verified_mlir/efficientnet_fwd.mlir"
    ".lake/build/efficientnet_fwd_v.vmfb" "forward (committed bytes, not re-rendered)"
  smoke "verified_mlir/efficientnet_fwd_eval.mlir"
    ".lake/build/efficientnet_fwd_eval_v.vmfb" "eval forward (committed bytes, not re-rendered)"

#eval main
