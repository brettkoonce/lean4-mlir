import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `mobilenetv2-verified` — train a small MobileNetV2 on the VERIFIED-rendered codegen

Chapter 7 (C4 + C3): a real, DOWNSAMPLING MobileNetV2 whose architecture VJP is the
audited `Proofs.mobilenetv2_has_vjp_at` (unconditional via `mnv2Concrete_has_vjp_correct`;
the stride-2 depthwise input-VJP `depthwiseStride2Flat_has_vjp_correct` is also audited),
now rendered + GPU-trained. CIFAR 3×32×32, the reference inverted-residual `[t,c,n,s]`
shape with STRIDE-2 DEPTHWISE downsampling:

  stem  3×3 stride-2 conv (3→16, 32→16) → BN → relu6 →
  b1    IR 16→24, mid 64,  stride 2 (16→8)  [no skip] →
  b2    IR 24→24, mid 96,  stride 1 (8×8)   [skip]    →
  b3    IR 24→32, mid 96,  stride 2 (8→4)   [no skip] →
  b4    IR 32→32, mid 128, stride 1 (4×4)   [skip]    →
  b5    IR 32→64, mid 128, stride 1 (4×4)   [no skip] →
  b6    IR 64→64, mid 256, stride 1 (4×4)   [skip]    →
  head  1×1 conv (64→128) → BN → relu6  (the MNv2 "features" layer @4×4) →
  global-average-pool → dense 128→10 + softmax-CE

The head's relu6 before GAP is essential: per-example instance-norm zeroes each
channel's spatial mean, so GAP of a raw linear-bottleneck BN is the constant β
(input-independent); the relu6 gives the pooled tensor a per-input mean.

Trains on `verified_mlir/mobilenetv2_train_step.mlir` (82 params), evals via
`verified_mlir/mobilenetv2_fwd.mlir` — both rendered by
tests/TestMobilenetV2{Train,Fwd}.lean from the same `allParams`, every op fragment
tests/TestMobilenetV2{Train,Fwd}.lean from the same `blocks`/`allParams`, every op
fragment the StableHLO of a proven-faithful emitter: depthwise conv stride-1
(`depthwise_has_vjp3_correct`, C1 `depthwiseF/Back`) + stride-2
(`depthwiseStride2Flat_has_vjp_correct`, C3 `depthwiseStridedF/Back`), relu6
(`relu6_has_vjp_at`, C2 `relu6F/selectMid`), per-channel BN
(`bnPerChannelTensor3_grad_input_correct`), residual `addV`, 1×1 convs, GAP, dense,
regular stride-2 stem conv. Both MLIRs iree-compile to ROCm gfx1100.

82 params packed per `MobileNetV2Layout` (per-channel γ/β rank-1 `[c]`; depthwise
kernels `[mid,1,3,3]`). Reuses the params-general `mlpTrainStepV` FFI. He init for
conv/dense weights (depthwise fan-in = 9), γ=1, β=0, biases=0; mean-loss SGD lr=0.3
(baked into the rendered train step).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/mobilenetv2-verified data`
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
    (kind 0; fan-in = `ic·kH·kW` for a rank-4 conv kernel — for a depthwise
    `[c,1,3,3]` kernel that is `1·3·3 = 9` — or `in` for a rank-2 dense matrix),
    γ = 1 (kind 1), β / bias = 0 (kind 2). -/
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
  IO.println "MobileNetV2 (stem-s2 → 6 inverted-residual blocks, 2 stride-2 depthwise downsamples → head conv-BN-relu6 → GAP → dense) via the VERIFIED renderer → IREE FFI → GPU"
  compileVmfb "verified_mlir/mobilenetv2_train_step.mlir" ".lake/build/mobilenetv2_ts_v.vmfb"
  compileVmfb "verified_mlir/mobilenetv2_fwd.mlir"        ".lake/build/mobilenetv2_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/mobilenetv2_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/mobilenetv2_fwd_v.vmfb"
  let cdir := dataDir ++ "/cifar-10"
  let trainPaths := (List.range 5).map (fun i => s!"{cdir}/data_batch_{i+1}.bin")
  let (trainImg, trainLbl, nTrain) ← loadCifarSplit trainPaths
  let (testImg,  testLbl,  nTest)  ← loadCifarSplit [s!"{cdir}/test_batch.bin"]
  IO.println s!"  train {nTrain}, test {nTest}; bs {BS}, MobileNetV2 ({MobileNetV2Layout.specs.size} params, {MobileNetV2Layout.nParams} floats), per-channel BN, mean-loss SGD lr=0.3, He init"
  (← IO.getStdout).flush
  let nb  := nTrain / BS
  let nbt := nTest / BS
  let shapes := MobileNetV2Layout.shapesBA
  let xShape := MobileNetV2Layout.xShape BS
  -- init the 82 params in func-arg order from the layout specs
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in MobileNetV2Layout.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut params := F32.concat parts
  for ep in [0:20] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * BS) BS D0
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.mobilenetv2_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages testImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.mobilenetv2_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (testLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: test_acc = {correct}/{nbt * BS} = {acc}%"
    (← IO.getStdout).flush
  IO.println "done (trained a small MobileNetV2 on the proof-rendered StableHLO)."
