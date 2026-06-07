import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `efficientnet-verified` — train a small EfficientNet on the VERIFIED-rendered codegen

Chapter 8 (E1–E6): the EfficientNet-B0 architecture (Tan & Le 2019) whose whole-network
VJP is the audited, UNCONDITIONAL `Proofs.efficientnet_has_vjp(_correct)` (the swish/sigmoid/
SE/MBConv VJP stack — all smooth, no kinks), now rendered + GPU-trained. ALL-SWISH with
BATCH norm (E5). FAITHFUL B0 `[t,c,n,s,k]` config (E6) on IMAGENETTE 224×224, 10 classes
(the native B0 resolution — stem stride 2, 5 downsamples 224→7, no degenerate tiny maps):

  stem  3×3 stride-2 conv (3→32) → BN → swish   (224→112)
  B0 stages [expand t, channels c, repeats n, stride s, kernel k]:
    s1 t1 c16  n1 s1 k3 (MBConv1, no expand) @112  s2 t6 c24  n2 s2 k3  112→56
    s3 t6 c40  n2 s2 k5                      56→28  s4 t6 c80  n3 s2 k3   28→14
    s5 t6 c112 n3 s1 k5                      @14    s6 t6 c192 n4 s2 k5   14→7
    s7 t6 c320 n1 s1 k3                      @7
  head  1×1 conv (320→1280) → BN → swish → GAP → dense 1280→10 + softmax-CE
  16 MBConv layers; SE (ratio 0.25 of block-input ch) in every block.

MBConv: expand 1×1 conv→BN→swish → depthwise 3×3 (stride 1/2)→BN→swish → SE gate
(squeeze C→r→C, sigmoid, ×main) → project 1×1 conv→BN; + residual iff s=1 ∧ ic=oc.
BATCH norm (not the ch7 per-example instance-norm) keeps inter-image variance in the
pooled features, so swish works at the final GAP — genuinely all-swish, no relu6 head.

Trains on `verified_mlir/efficientnet_train_step.mlir` (262 params), evals via
`verified_mlir/efficientnet_fwd.mlir` — both rendered by tests/TestEfficientNet{Train,Fwd}.lean
from the same B0 `stages`/`blocks` generator, every op fragment the StableHLO of a proven-faithful
emitter: swish (`swish_has_vjp_correct`, E1 `swishF/swishBack`), sigmoid (`sigmoid_has_vjp`,
E2 `sigmoidF`), the SE gate (`seBlock_has_vjp_correct`/`broadcastFlat_has_vjp`, E3 — gradcheck-
validated standalone), depthwise k×k stride-1/2 (`depthwiseFlat`/`depthwiseStride2Flat_has_vjp`,
the op is kernel-general — 3×3 and 5×5), BATCH norm (`bnBatchTensor4_grad_input_correct`, E5 —
reduce over batch+spatial [0,2,3]), residual `addV`, 1×1/3×3 convs, GAP, dense. Both MLIRs
iree-compile to ROCm gfx1100. NB eval uses batch stats (BS=128); population-stats EMA is out of scope.

262 params packed per `EfficientNetLayout` (batch-norm γ/β rank-1 `[c]`; depthwise kernels
`[mid,1,k,k]`; SE dense Ws₁`[mid,r]`/Ws₂`[r,mid]`; MBConv1 blocks have no expand params). Reuses
the params-general `mlpTrainStepV` FFI. He init for conv/dense/SE weights (depthwise fan-in = k²),
γ=1, β=0, biases=0; mean-loss SGD lr=0.1 (baked into the rendered train step).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/efficientnet-verified data`
-/

private def BS : Nat := 32
private def D0 : Nat := 3 * 224 * 224       -- 150528 (Imagenette 224²)
private def TRAINPIX : Nat := 3 * 256 * 256 -- train stored at 256², center-cropped to 224
private def NCLASS : Nat := 10

private def compileVmfb (mlirPath outPath : String) : IO Unit := do
  let cargs ← ireeCompileArgs mlirPath outPath
  IO.println s!"  iree-compile {mlirPath}"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    throw (IO.userError s!"iree-compile failed:\n{r.stderr.take 2000}")

/-- Init one parameter from its `(dims, initKind)` spec: He(fan-in) weights (kind 0;
    fan-in = `ic·kH·kW` for a rank-4 conv kernel — for a depthwise `[c,1,k,k]` kernel
    that is `k²` — or `in` for a rank-2 dense/SE matrix), γ = 1 (kind 1),
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
  IO.println "EfficientNet-B0 on Imagenette 224² (stem-s2 → 16 MBConv layers [t,c,n,s,k], swish + squeeze-excite + batch-norm, 5 downsamples 224→7 → head 320→1280 conv-BN-swish → GAP → dense 10) via the VERIFIED renderer → IREE FFI → GPU"
  compileVmfb "verified_mlir/efficientnet_train_step.mlir" ".lake/build/efficientnet_ts_v.vmfb"
  compileVmfb "verified_mlir/efficientnet_fwd.mlir"        ".lake/build/efficientnet_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/efficientnet_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/efficientnet_fwd_v.vmfb"
  let idir := dataDir ++ "/imagenette"
  -- train stored at 256² (center-crop to 224 per batch); val at 224²
  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (idir ++ "/train.bin") 256
  let (valImg,   valLbl,   nVal)   ← F32.loadImagenette (idir ++ "/val.bin")
  IO.println s!"  train {nTrain}, val {nVal}; bs {BS}, EfficientNet-B0 224² ({EfficientNetLayout.specs.size} params, {EfficientNetLayout.nParams} floats), batch-norm, mean-loss SGD lr=0.1, He init"
  (← IO.getStdout).flush
  let nb  := nTrain / BS
  let nbt := nVal / BS
  let shapes := EfficientNetLayout.shapesBA
  let xShape := EfficientNetLayout.xShape BS
  -- init the 262 params in func-arg order from the layout specs
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in EfficientNetLayout.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut params := F32.concat parts
  for ep in [0:20] do
    for bi in [0:nb] do
      let xb256 := F32.sliceImages trainImg (bi * BS) BS TRAINPIX
      let xb ← F32.centerCrop xb256 BS.toUSize 3 256 256 224 224   -- 256→224 (deterministic)
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.efficientnet_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages valImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.efficientnet_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (valLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: val_acc = {correct}/{nbt * BS} = {acc}%"
    (← IO.getStdout).flush
  IO.println "done (trained EfficientNet-B0 on Imagenette via the proof-rendered StableHLO)."
