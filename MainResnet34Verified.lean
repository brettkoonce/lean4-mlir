import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `resnet34-verified` — train a real ResNet-34 on the VERIFIED-rendered codegen

Chapter 6 Milestone B9: the whole 34-layer ResNet whose architecture VJP is the
audited `Proofs.resnet34_has_vjp_at` (unconditional via `resnet34Concrete_has_vjp_correct`),
now rendered + GPU-trained. CIFAR 3×32×32:

  conv(3→64,3×3,SAME) → BN → relu → maxpool(32→16) →
  stage1: 3 identity blocks @64           (16×16) →
  stage2: downsample 64→128 + 3 identity  (8×8)   →
  stage3: downsample 128→256 + 5 identity (4×4)   →
  stage4: downsample 256→512 + 2 identity (2×2)   →
  global-average-pool → dense 512→10 + softmax-CE

Trains on `verified_mlir/resnet34_train_step.mlir` (146 params), evals via
`verified_mlir/resnet34_fwd.mlir` — both rendered by tests/TestResnet34{Train,Fwd}.lean
from the same `Blk` list, every op fragment the StableHLO of a proven-faithful emitter:
strided conv (`flatConvStride2_has_vjp`), **per-channel BN** (B8b `bnPerChannelF/Back`,
`bnPerChannelTensor3_grad_input_correct`), residual `addV` fan-in, GAP, dense, conv /
maxpool / relu. Both MLIRs iree-compile to ROCm gfx1100.

146 params packed per `ResNet34Layout` (per-channel γ/β are rank-1 `[c]`). Reuses the
params-general `mlpTrainStepV` FFI. He init for conv/dense weights, γ=1, β=0, biases=0;
mean-loss SGD lr=0.1 (baked into the rendered train step).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/resnet34-verified data`
-/

private def BS : Nat := 128
private def D0 : Nat := 3072
private def NCLASS : Nat := 10

private def compileVmfb (mlirPath outPath : String) : IO Unit := do
  let cargs ← ireeCompileArgs mlirPath outPath
  IO.println s!"  iree-compile {mlirPath}"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    throw (IO.userError s!"iree-compile failed:\n{r.stderr.take 2000}")

private def loadCifarSplit (paths : List String) : IO (ByteArray × ByteArray × Nat) := do
  let mut raw : ByteArray := .empty
  let mut labels : ByteArray := .empty
  let mut nTotal : Nat := 0
  for p in paths do
    let batchRaw ← IO.FS.readBinFile p
    let n := batchRaw.size / 3073
    for j in [:n] do
      labels := labels.push batchRaw[j * 3073]!
      labels := labels.push 0; labels := labels.push 0; labels := labels.push 0
    raw := raw.append batchRaw
    nTotal := nTotal + n
  let imgs ← F32.cifarBatch raw 0 nTotal.toUSize
  return (imgs, labels, nTotal)

/-- Init one parameter from its `(dims, initKind)` spec: He(fan-in) weights
    (kind 0; fan-in = `ic·kH·kW` for a rank-4 conv kernel, `in` for a rank-2 dense
    matrix), γ = 1 (kind 1), β / bias = 0 (kind 2). -/
private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

def main (argv : List String) : IO Unit := do
  let dataDir := argv.head?.getD "data"
  IO.println "Real ResNet-34 (stem→pool→[3,4,6,3] blocks w/ per-channel BN + strided downsamples→GAP→dense) via the VERIFIED renderer → IREE FFI → GPU"
  compileVmfb "verified_mlir/resnet34_train_step.mlir" ".lake/build/resnet34_ts_v.vmfb"
  compileVmfb "verified_mlir/resnet34_fwd.mlir"        ".lake/build/resnet34_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/resnet34_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/resnet34_fwd_v.vmfb"
  let cdir := dataDir ++ "/cifar-10"
  let trainPaths := (List.range 5).map (fun i => s!"{cdir}/data_batch_{i+1}.bin")
  let (trainImg, trainLbl, nTrain) ← loadCifarSplit trainPaths
  let (testImg,  testLbl,  nTest)  ← loadCifarSplit [s!"{cdir}/test_batch.bin"]
  IO.println s!"  train {nTrain}, test {nTest}; bs {BS}, ResNet-34 ({ResNet34Layout.specs.size} params, {ResNet34Layout.nParams} floats), per-channel BN, mean-loss SGD lr=0.1, He init"
  (← IO.getStdout).flush
  let nb  := nTrain / BS
  let nbt := nTest / BS
  let shapes := ResNet34Layout.shapesBA
  let xShape := ResNet34Layout.xShape BS
  -- init the 146 params in func-arg order from the layout specs
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in ResNet34Layout.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut params := F32.concat parts
  for ep in [0:10] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * BS) BS D0
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.resnet34_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages testImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.resnet34_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (testLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: test_acc = {correct}/{nbt * BS} = {acc}%"
    (← IO.getStdout).flush
  IO.println "done (trained a real ResNet-34 on the proof-rendered StableHLO)."
