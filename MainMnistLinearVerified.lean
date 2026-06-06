import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `mnist-linear-verified` — train MNIST on the VERIFIED-rendered codegen

Trains the Chapter-2 linear classifier on the StableHLO that the **verified
renderer** emits — `verified_mlir/linear_train_step.mlir`, which is
`Proofs.StableHLO.linearTrainStepModuleV` = `pretty (emit g)`, the text whose
denotation is machine-proven equal to the Mathlib `fderiv` math
(`LeanMlir/Proofs/StableHLO.lean`, audited 3-axiom-clean). The forward, softmax-
CE cotangent, parameter gradients, and SGD update are all the proof-backed ops.

This is the *real* path: the Lean loop → `IreeRuntime` FFI (`libiree_ffi.so`) →
in-process IREE → GPU — not the Python/CLI stand-in. The training step runs via
`IreeSession.linearTrainStepV` (the verified module's signature); eval runs the
verified `@linear_fwd` via `IreeSession.forwardF32`.

Regenerate the `verified_mlir/*.mlir` with
`lake env lean LeanMlir/Proofs/StableHLO.lean`.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-linear-verified data`
-/

private def D0 : Nat := 784
private def D1 : Nat := 10
private def BS : Nat := 128

private def compileVmfb (mlirPath outPath : String) : IO Unit := do
  let cargs ← ireeCompileArgs mlirPath outPath
  IO.println s!"  iree-compile {mlirPath}"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    throw (IO.userError s!"iree-compile failed:\n{r.stderr.take 2000}")

def main (argv : List String) : IO Unit := do
  let dataDir := argv.head?.getD "data"
  IO.println "MNIST-Linear via the VERIFIED renderer (pretty∘emit) → IREE FFI → GPU"
  -- compile the verified-rendered StableHLO
  compileVmfb "verified_mlir/linear_train_step.mlir" ".lake/build/linear_ts_v.vmfb"
  compileVmfb "verified_mlir/linear_fwd.mlir"        ".lake/build/linear_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/linear_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/linear_fwd_v.vmfb"
  -- data (f32 images in [0,1], int32 labels)
  let (trainImg, nTrain) ← F32.loadIdxImages (dataDir ++ "/train-images-idx3-ubyte")
  let (trainLbl, _)      ← F32.loadIdxLabels (dataDir ++ "/train-labels-idx1-ubyte")
  let (testImg,  nTest)  ← F32.loadIdxImages (dataDir ++ "/t10k-images-idx3-ubyte")
  let (testLbl,  _)      ← F32.loadIdxLabels (dataDir ++ "/t10k-labels-idx1-ubyte")
  IO.println s!"  train {nTrain}, test {nTest}; dense {D0}->{D1}, bs {BS}, SGD"
  let nb  := nTrain / BS
  let nbt := nTest / BS
  let fwdShapes := packShapes #[#[D0, D1], #[D1]]
  let xShape    := packXShape #[BS, D0]
  let mut W0 ← F32.const (D0 * D1).toUSize 0.0
  let mut b0 ← F32.const D1.toUSize 0.0
  for ep in [0:12] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * BS) BS D0
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      let out ← IreeSession.linearTrainStepV tsSess "m.linear_train_step"
                  xb W0 b0 yb BS.toUSize D0.toUSize D1.toUSize
      W0 := out.extract 0 (D0 * D1 * 4)
      b0 := out.extract (D0 * D1 * 4) ((D0 * D1 + D1) * 4)
    -- eval via the verified forward
    let params := W0 ++ b0
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages testImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.linear_fwd" params fwdShapes
                      xb xShape BS.toUSize D1.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * D1).toUSize).toNat
        let lbl  := (testLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: test_acc = {correct}/{nbt * BS} = {acc}%"
  IO.println "done (trained on the proof-rendered StableHLO)."
