import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `mnist-mlp-verified` — train the MNIST MLP on the VERIFIED-rendered codegen

Chapter 3: `dense 784→512 → relu → dense 512→512 → relu → dense 512→10` +
softmax-CE. Trains on `verified_mlir/mlp_train_step.mlir`
(`Proofs.StableHLO.mlpTrainStepText`), whose forward/backward/grad ops are each
proven faithful to the Mathlib `fderiv` math (`mlpFwdGraph_faithful`,
`mlpBackGraph_faithful`, `reluF_faithful`, `selectPos_faithful`,
`wGrad/bGrad_is*Jacobian`, `lossCotGraph_isCEgrad`) — audited 3-axiom-clean.

Real path: Lean loop → IreeRuntime FFI → in-process IREE → GPU.
Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mnist-mlp-verified data`
-/

private def D0 : Nat := 784
private def D1 : Nat := 512
private def D2 : Nat := 512
private def D3 : Nat := 10
private def BS : Nat := 128

private def compileVmfb (mlirPath outPath : String) : IO Unit := do
  let cargs ← ireeCompileArgs mlirPath outPath
  IO.println s!"  iree-compile {mlirPath}"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    throw (IO.userError s!"iree-compile failed:\n{r.stderr.take 2000}")

def main (argv : List String) : IO Unit := do
  let dataDir := argv.head?.getD "data"
  IO.println "MNIST-MLP via the VERIFIED renderer (784→512→512→10) → IREE FFI → GPU"
  compileVmfb "verified_mlir/mlp_train_step.mlir" ".lake/build/mlp_ts_v.vmfb"
  compileVmfb "verified_mlir/mlp_fwd.mlir"        ".lake/build/mlp_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/mlp_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/mlp_fwd_v.vmfb"
  let (trainImg, nTrain) ← F32.loadIdxImages (dataDir ++ "/train-images-idx3-ubyte")
  let (trainLbl, _)      ← F32.loadIdxLabels (dataDir ++ "/train-labels-idx1-ubyte")
  let (testImg,  nTest)  ← F32.loadIdxImages (dataDir ++ "/t10k-images-idx3-ubyte")
  let (testLbl,  _)      ← F32.loadIdxLabels (dataDir ++ "/t10k-labels-idx1-ubyte")
  IO.println s!"  train {nTrain}, test {nTest}; bs {BS}, SGD"
  let nb  := nTrain / BS
  let nbt := nTest / BS
  let shapes := packShapes #[#[D0, D1], #[D1], #[D1, D2], #[D2], #[D2, D3], #[D3]]
  let xShape := packXShape #[BS, D0]
  -- He-init weights, zero biases; pack W0|b0|W1|b1|W2|b2
  let mkW (seed n fanIn : Nat) : IO ByteArray := F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))
  let W0 ← mkW 1 (D0*D1) D0
  let b0 ← F32.const D1.toUSize 0.0
  let W1 ← mkW 2 (D1*D2) D1
  let b1 ← F32.const D2.toUSize 0.0
  let W2 ← mkW 3 (D2*D3) D2
  let b2 ← F32.const D3.toUSize 0.0
  let mut params := F32.concat #[W0, b0, W1, b1, W2, b2]
  for ep in [0:12] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * BS) BS D0
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.mlp_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize D3.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages testImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.mlp_fwd" params shapes
                      xb xShape BS.toUSize D3.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * D3).toUSize).toNat
        let lbl  := (testLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: test_acc = {correct}/{nbt * BS} = {acc}%"
  IO.println "done (trained on the proof-rendered MLP StableHLO)."
