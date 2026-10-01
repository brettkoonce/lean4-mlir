import LeanMlir.Proofs.Codegen.StableHLO.Basic
import LeanMlir.Types

/-! # MobileNetV2 forward: the `iree-compile` smoke over the COMMITTED bytes

`verified_mlir/mobilenetv2_fwd.mlir` and `verified_mlir/mobilenetv2_fwd_eval.mlir` are
written by `LeanMlir/Proofs/Codegen/MobileNetV2RenderB.lean`'s `mobilenetv2FwdText` and
`mnv2FwdEvalText` — `pretty(provenGraph)`, off the `mnv2FwdChainB` the train step
differentiates — and those
`#eval`s are their only writers. This file keeps only the part `lake build` genuinely cannot do:
running `iree-compile`, which needs the compiler on PATH.

The forward must normalise over the same axes as the `mobilenetv2_train_step.mlir` it partners
(per example: reduce `[2, 3]`, n = H·W). A **batch** BatchNorm forward (reduce `[0, 2, 3]`,
n = B·H·W) scores a different function from the one the train step trains: on one shared (θ, x)
with the real He init, `lake build fwd-tie` measures **max rel 1.86**, 0/320 logits bit-exact,
against a bit-exact 320/320 determinism floor. A relative difference above 1 means the logits
disagree in sign, not just magnitude. `@mobilenetv2_fwd_eval` is frozen-stat affine BN, which
performs no reduction and so is the same graph in either BN world.

The net (full paper `[t,c,n,s]` MobileNetV2, Imagenette 3×224×224):

  stem  : 3×3 stride-2 conv (3→32, 224→112) + BN + relu6
  b1-b17: full-paper inverted-residual stack (t=1 no-expand b1, then 16→24→32→64→96→160→320,
          4 stride-2 depthwise downsamples 112→56→28→14→7)
  head  : 1×1 conv (320→1280) + BN + relu6 → GAP → dense(1280→10)

Run (needs iree-compile on PATH):
  lake env lean tests/TestMobilenetV2Fwd.lean
-/

open Proofs Proofs.StableHLO


/-- Compile a COMMITTED artifact. Throws if it is missing: this file is not its writer, and
    recreating it here is exactly the double-writer race that can ship two different functions. -/
private def smoke (path dst label : String) : IO Unit := do
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/MobileNetV2RenderB.lean; run \
`lake build LeanMlir.Proofs.Codegen.MobileNetV2RenderB` first")
  tryCompile path dst label

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  smoke "verified_mlir/mobilenetv2_fwd.mlir"
    ".lake/build/mobilenetv2_fwd_v.vmfb" "forward (committed bytes, not re-rendered)"
  smoke "verified_mlir/mobilenetv2_fwd_eval.mlir"
    ".lake/build/mobilenetv2_fwd_eval_v.vmfb" "eval forward (committed bytes, not re-rendered)"

#eval main
