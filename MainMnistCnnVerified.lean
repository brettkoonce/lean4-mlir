import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `mnist-cnn-verified` — train the MNIST CNN on the VERIFIED-rendered codegen

Chapter 4: `conv 1→32 → relu → conv 32→32 → relu → maxpool 28→14 →
flatten → dense 6272→512 → relu → dense 512→512 → relu → dense 512→10` +
softmax-CE. Trains on `verified_mlir/cnn_train_step.mlir`
(`Proofs.StableHLO.cnnTrainStepText`), whose forward/backward/grad ops are each
proven faithful to the Mathlib `fderiv` math (`cnnFwdGraph_faithful`,
`convBack_faithful`, `maxPoolBack_faithful`, `reluF_faithful`,
`selectPos_faithful`, `wGrad/bGrad_is*Jacobian`, `lossCotGraph_isCEgrad`) —
audited 3-axiom-clean. The conv weight grad is the transpose-trick render,
validated by this run.

Reuses the params-general FFI binding `mlpTrainStepV` (x, packed params per
`CnnLayout.shapesBA` incl. 4-D conv kernels, int32 labels → one-hot in C).

Real path: Lean loop → IreeRuntime FFI → in-process IREE → GPU.
Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-cnn-verified data`
-/

private def IC : Nat := 1
private def C : Nat := 32
private def KH : Nat := 3
private def KW : Nat := 3
private def FLAT : Nat := 6272      -- C * 14 * 14
private def D1 : Nat := 512
private def NCLASS : Nat := 10
private def D0 : Nat := 784         -- IC * 28 * 28 (flattened input width)
private def BS : Nat := 128

private def compileVmfb (mlirPath outPath : String) : IO Unit := do
  let cargs ← ireeCompileArgs mlirPath outPath
  IO.println s!"  iree-compile {mlirPath}"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    throw (IO.userError s!"iree-compile failed:\n{r.stderr.take 2000}")

def main (argv : List String) : IO Unit := do
  let dataDir := argv.head?.getD "data"
  IO.println "MNIST-CNN via the VERIFIED renderer (conv→conv→pool→512→512→10) → IREE FFI → GPU"
  compileVmfb "verified_mlir/cnn_train_step.mlir" ".lake/build/cnn_ts_v.vmfb"
  compileVmfb "verified_mlir/cnn_fwd.mlir"        ".lake/build/cnn_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/cnn_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/cnn_fwd_v.vmfb"
  let (trainImg, nTrain) ← F32.loadIdxImages (dataDir ++ "/train-images-idx3-ubyte")
  let (trainLbl, _)      ← F32.loadIdxLabels (dataDir ++ "/train-labels-idx1-ubyte")
  let (testImg,  nTest)  ← F32.loadIdxImages (dataDir ++ "/t10k-images-idx3-ubyte")
  let (testLbl,  _)      ← F32.loadIdxLabels (dataDir ++ "/t10k-labels-idx1-ubyte")
  IO.println s!"  train {nTrain}, test {nTest}; bs {BS}, SGD"
  let nb  := nTrain / BS
  let nbt := nTest / BS
  let shapes := CnnLayout.shapesBA
  let xShape := CnnLayout.xShape BS
  -- He-init weights (conv fanin = ic·kH·kW), zero biases; pack in arg order
  -- W1|b1|W2|b2|W3|b3|W4|b4|W5|b5 (= CnnLayout.paramShapes).
  let mkW (seed n fanIn : Nat) : IO ByteArray :=
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))
  let W1 ← mkW 1 (C*IC*KH*KW)   (IC*KH*KW)
  let b1 ← F32.const C.toUSize 0.0
  let W2 ← mkW 2 (C*C*KH*KW)    (C*KH*KW)
  let b2 ← F32.const C.toUSize 0.0
  let W3 ← mkW 3 (FLAT*D1)      FLAT
  let b3 ← F32.const D1.toUSize 0.0
  let W4 ← mkW 4 (D1*D1)        D1
  let b4 ← F32.const D1.toUSize 0.0
  let W5 ← mkW 5 (D1*NCLASS)    D1
  let b5 ← F32.const NCLASS.toUSize 0.0
  let mut params := F32.concat #[W1, b1, W2, b2, W3, b3, W4, b4, W5, b5]
  for ep in [0:10] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * BS) BS D0
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.cnn_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages testImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.cnn_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (testLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: test_acc = {correct}/{nbt * BS} = {acc}%"
  IO.println "done (trained on the proof-rendered CNN StableHLO)."
