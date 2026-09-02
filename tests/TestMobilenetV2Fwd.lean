import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # C4a/D3 — MobileNetV2 forward: the `iree-compile` smoke over the COMMITTED bytes

**The hand-written emitter that used to live here is RETIRED (2026-07-28), and retiring it FIXED A
BUG.** `verified_mlir/mobilenetv2_fwd.mlir` and `verified_mlir/mobilenetv2_fwd_eval.mlir` are now
written by `LeanMlir/Proofs/Codegen/MobileNetV2Render.lean`'s `mnv2Fwd{,Eval}FaithfulV` —
`pretty(provenGraph)`, both off the single `mnv2FwdChain` the train step differentiates — and those
`#eval`s are their only writers. This file keeps only the part `lake build` genuinely cannot do:
running `iree-compile`, which needs the compiler on PATH.

## ▶ The bug: `@mobilenetv2_fwd` was normalising over the wrong axis

The emitter here rendered **batch** BatchNorm (reduce `[0, 2, 3]`, n = B·H·W — its own docstring
said so) while the `mobilenetv2_train_step.mlir` it partners normalises **per example**
(reduce `[2, 3]`, n = H·W). So `mobilenetv2-verified` trained a per-example-BN net and **scored it
with batch statistics**: two different functions under one artifact name.

This is precisely the ResNet-34 defect of `planning/xla_pjrt_handoff.md` §2a, fixed there on
2026-07-27 and still live here. Measured on one shared (θ, x) with the real He init, via
`lake build fwd-tie`:

    git show 17413f0:verified_mlir/mobilenetv2_fwd.mlir > /tmp/retired.mlir
    .lake/build/bin/fwd-tie mobilenetv2 /tmp/retired.mlir verified_mlir/mobilenetv2_fwd.mlir

| | result |
|---|---|
| retired (batch BN) vs certified (per-example BN) | **max rel 1.86**, 0/320 logits bit-exact |
| certified vs ITSELF (the determinism floor) | bit-exact 320/320 |

A relative difference above 1 means the logits disagree in sign, not just magnitude. The floor
being bit-exact is what makes that number graph-attributable rather than backend noise.

**Consequence for published numbers:** the 86.89% in `runs/mobilenetv2_verified_crop_gpu0.log` was
measured through the old forward and needs re-running, exactly as §3 already records for
`resnet34-verified`. The training run itself is unaffected — only the scoring was wrong.

`@mobilenetv2_fwd_eval` was NOT affected: it is frozen-stat affine BN, which performs no reduction
and so is the same graph in either BN world. It ties the retired render **BIT-EXACT, 320/320**.

The net, unchanged (full paper `[t,c,n,s]` MobileNetV2, Imagenette 3×224×224):

  stem  : 3×3 stride-2 conv (3→32, 224→112) + BN + relu6
  b1-b17: full-paper inverted-residual stack (t=1 no-expand b1, then 16→24→32→64→96→160→320,
          4 stride-2 depthwise downsamples 112→56→28→14→7)
  head  : 1×1 conv (320→1280) + BN + relu6 → GAP → dense(1280→10)

Recover the retired emitter from `git show 17413f0:tests/TestMobilenetV2Fwd.lean`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestMobilenetV2Fwd.lean
-/

open Proofs Proofs.StableHLO


/-- Compile a COMMITTED artifact. Throws if it is missing: this file is not its writer, and
    recreating it here is exactly the double-writer race that shipped two different functions. -/
private def smoke (path dst label : String) : IO Unit := do
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/MobileNetV2Render.lean; run \
`lake build LeanMlir.Proofs.Codegen.MobileNetV2Render` first")
  tryCompile path dst label

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  smoke "verified_mlir/mobilenetv2_fwd.mlir"
    ".lake/build/mobilenetv2_fwd_v.vmfb" "forward (committed bytes, not re-rendered)"
  smoke "verified_mlir/mobilenetv2_fwd_eval.mlir"
    ".lake/build/mobilenetv2_fwd_eval_v.vmfb" "eval forward (committed bytes, not re-rendered)"

#eval main
