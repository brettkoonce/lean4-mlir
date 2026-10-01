import LeanMlir.Proofs.Codegen.StableHLO.Basic
import LeanMlir.Types

/-! # ch9 N5 — ConvNeXt-T forward: the `iree-compile` smoke over the COMMITTED bytes

`verified_mlir/convnext_fwd.mlir` is written by `LeanMlir/Proofs/Codegen/ConvNeXtRender.lean`'s
`convNextFwdText` — `pretty(provenGraph)`, sharing `convNextFwdChain` with both train steps —
and that `#eval` is its only writer. This file keeps only the part `lake build` genuinely cannot do:
running `iree-compile`, which needs the compiler on PATH.

The net (ConvNeXt-T, Liu et al. 2022, Imagenette 224², 10 classes):

  stem    : 4×4 conv stride 4 (3→96)  "patchify"             224→56
  stage 1 : 3× ConvNeXt block @ 96                           @56
  downsmpl: LN + 2×2 conv stride 2 (96→192)                  56→28
  stage 2 : 3× ConvNeXt block @ 192                          @28
  downsmpl: LN + 2×2 conv stride 2 (192→384)                 28→14
  stage 3 : 9× ConvNeXt block @ 384                          @14
  downsmpl: LN + 2×2 conv stride 2 (384→768)                 14→7
  stage 4 : 3× ConvNeXt block @ 768                          @7
  head    : globalAvgPool → LN(768) → dense 768→10

ConvNeXt block (dim c, expand 4c): depthwise 7×7 → LN → 1×1 expand c→4c → GELU
→ 1×1 project 4c→c → layerScale (per-channel γ) → + x (identity skip; stride-1 only).

**Why this file has no emitter**: a second emitter that can write is one more thing to drift, and
two different functions can ship under one artifact name that way.

`lake build fwd-tie` compares two forward renders logit by logit. On this net it is capable of
failing: perturbing the 22 LayerNorm ε constants (1e-6 → 1e-3) fires it at rel 5.9e-2, and the
GELU cubic coefficient (0.044715 → 0.05) at 3.7e-1, both 0/320 bit-exact.

One control does NOT fire, and it is worth knowing why rather than filing it as a weakness:
rescaling the GAP divisor (49 → 48) moves the logits by only 5.1e-5. The head LayerNorm
immediately normalises its input, and LN is scale-invariant up to ε — so a rescale of the pooled
features is absorbed before the dense layer. It is a genuinely near-identity perturbation of THIS
net, not a gap in the gate. (It also means a GAP-normaliser bug here could not change eval
accuracy.)

Run (needs iree-compile on PATH):
  lake env lean tests/TestConvNeXtFwd.lean
-/

open Proofs Proofs.StableHLO


/-- Compile a COMMITTED artifact. Throws if it is missing: this file is not its writer, and
    recreating it here is exactly the double-writer race that can ship two different functions. -/
private def smoke (path dst label : String) : IO Unit := do
  if !(← System.FilePath.pathExists path) then
    throw (IO.userError s!"{path} missing — it is written by \
LeanMlir/Proofs/Codegen/ConvNeXtRender.lean; run `lake build LeanMlir.Proofs.Codegen.ConvNeXtRender` \
first")
  tryCompile path dst label

def main : IO Unit := do
  IO.FS.createDirAll ".lake/build"
  smoke "verified_mlir/convnext_fwd.mlir"
    ".lake/build/convnext_fwd_v.vmfb" "forward (committed bytes, not re-rendered)"

#eval main
