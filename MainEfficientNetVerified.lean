import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `efficientnet-verified` — train a small EfficientNet on the VERIFIED-rendered codegen

Chapter 8 (E1–E4): a real, DOWNSAMPLING EfficientNet whose whole-network VJP is the
audited, UNCONDITIONAL `Proofs.efficientnet_has_vjp(_correct)` (the swish/sigmoid/SE/
MBConv VJP stack — all smooth, no kinks), now rendered + GPU-trained. CIFAR 3×32×32, the
MBConv (inverted-residual + squeeze-excite + swish) shape with STRIDE-2 DEPTHWISE
downsampling:

  stem  3×3 stride-2 conv (3→16, 32→16) → BN → swish →
  b1    MBConv 16→24, mid 64,  r 4,  stride 2 (16→8)  [no skip] →
  b2    MBConv 24→24, mid 96,  r 6,  stride 1 (8×8)   [skip]    →
  b3    MBConv 24→32, mid 96,  r 6,  stride 2 (8→4)   [no skip] →
  b4    MBConv 32→32, mid 128, r 8,  stride 1 (4×4)   [skip]    →
  b5    MBConv 32→64, mid 128, r 8,  stride 1 (4×4)   [no skip] →
  b6    MBConv 64→64, mid 256, r 16, stride 1 (4×4)   [skip]    →
  head  1×1 conv (64→128) → BN → swish  (the EfficientNet "features" layer @4×4) →
  global-average-pool → dense 128→10 + softmax-CE

MBConv: expand 1×1 conv→BN→swish → depthwise 3×3 (stride 1/2)→BN→swish → SE gate
(squeeze C→r→C, sigmoid, ×main) → project 1×1 conv→BN; + residual iff s=1 ∧ ic=oc. The
head's swish before GAP is essential (per-example instance-norm zeroes each channel's
spatial mean, so GAP of a raw linear-bottleneck BN is the constant β).

Trains on `verified_mlir/efficientnet_train_step.mlir` (106 params), evals via
`verified_mlir/efficientnet_fwd.mlir` — both rendered by tests/TestEfficientNet{Train,Fwd}.lean
from the same `blocks`/`allParams`, every op fragment the StableHLO of a proven-faithful
emitter: swish (`swish_has_vjp_correct`, E1 `swishF/swishBack`), sigmoid (`sigmoid_has_vjp`,
E2 `sigmoidF`), the SE gate (`seBlock_has_vjp_correct`/`broadcastFlat_has_vjp`, E3 — gradcheck-
validated standalone), depthwise stride-1/2 (`depthwiseFlat`/`depthwiseStride2Flat_has_vjp`),
per-channel BN (`bnPerChannelTensor3_grad_input_correct`), residual `addV`, 1×1/3×3 convs,
GAP, dense. Both MLIRs iree-compile to ROCm gfx1100.

106 params packed per `EfficientNetLayout` (per-channel γ/β rank-1 `[c]`; depthwise kernels
`[mid,1,3,3]`; SE dense Ws₁`[mid,r]`/Ws₂`[r,mid]`). Reuses the params-general `mlpTrainStepV`
FFI. He init for conv/dense/SE weights (depthwise fan-in = 9), γ=1, β=0, biases=0; mean-loss
SGD lr=0.3 (baked into the rendered train step).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/efficientnet-verified data`
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

/-- Init one parameter from its `(dims, initKind)` spec: He(fan-in) weights (kind 0;
    fan-in = `ic·kH·kW` for a rank-4 conv kernel — for a depthwise `[c,1,3,3]` kernel
    that is `1·3·3 = 9` — or `in` for a rank-2 dense/SE matrix), γ = 1 (kind 1),
    β / bias = 0 (kind 2). -/
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
  IO.println "EfficientNet (stem-s2 → 6 MBConv blocks: depthwise + swish + squeeze-excite, 2 stride-2 downsamples → head conv-BN-swish → GAP → dense) via the VERIFIED renderer → IREE FFI → GPU"
  compileVmfb "verified_mlir/efficientnet_train_step.mlir" ".lake/build/efficientnet_ts_v.vmfb"
  compileVmfb "verified_mlir/efficientnet_fwd.mlir"        ".lake/build/efficientnet_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/efficientnet_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/efficientnet_fwd_v.vmfb"
  let cdir := dataDir ++ "/cifar-10"
  let trainPaths := (List.range 5).map (fun i => s!"{cdir}/data_batch_{i+1}.bin")
  let (trainImg, trainLbl, nTrain) ← loadCifarSplit trainPaths
  let (testImg,  testLbl,  nTest)  ← loadCifarSplit [s!"{cdir}/test_batch.bin"]
  IO.println s!"  train {nTrain}, test {nTest}; bs {BS}, EfficientNet ({EfficientNetLayout.specs.size} params, {EfficientNetLayout.nParams} floats), per-channel BN, mean-loss SGD lr=0.3, He init"
  (← IO.getStdout).flush
  let nb  := nTrain / BS
  let nbt := nTest / BS
  let shapes := EfficientNetLayout.shapesBA
  let xShape := EfficientNetLayout.xShape BS
  -- init the 106 params in func-arg order from the layout specs
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in EfficientNetLayout.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut params := F32.concat parts
  for ep in [0:20] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * BS) BS D0
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.efficientnet_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages testImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.efficientnet_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (testLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: test_acc = {correct}/{nbt * BS} = {acc}%"
    (← IO.getStdout).flush
  IO.println "done (trained a small EfficientNet on the proof-rendered StableHLO)."
