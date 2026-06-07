import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `convnext-verified` — train ConvNeXt-T on the VERIFIED-rendered codegen

Chapter 9: the ConvNeXt-T architecture (Liu et al. 2022, "A ConvNet for the 2020s")
whose whole-network VJP is the audited, UNCONDITIONAL `Proofs.convnext_has_vjp(_correct)`
(the GELU/LayerNorm/layerScale/convNextBlock VJP stack — all smooth, only the 4× `0<ε`
LayerNorm conditions), now rendered + GPU-trained on IMAGENETTE 224×224, 10 classes (the
paper-native resolution — patchify /4, then /2 between stages, 224→56→28→14→7):

  stem    4×4 stride-4 conv (3→96) "patchify"                 224→56
  stage 1 3× ConvNeXt block @ 96                              @56
  downsmpl LN + 2×2 stride-2 conv (96→192)                    56→28
  stage 2 3× ConvNeXt block @ 192                             @28
  downsmpl LN + 2×2 stride-2 conv (192→384)                   28→14
  stage 3 9× ConvNeXt block @ 384                             @14
  downsmpl LN + 2×2 stride-2 conv (384→768)                   14→7
  stage 4 3× ConvNeXt block @ 768                             @7
  head    GAP → LN(768) → dense 768→10 + softmax-CE

ConvNeXt block: depthwise 7×7 → LN → 1×1 expand c→4c → GELU → 1×1 project 4c→c →
layerScale (per-channel γ) → + x (identity skip). LN here is the proof's per-example
GLOBAL scalar-γ/β LayerNorm (`= layerNormForward = bnForward` over the flat c·h·w vec),
NOT the paper's per-token channel-LN — faithful to the theorem, a simplification of the
paper. GELU = the tanh approximation (`geluF`, N1; closed-form derivative `geluScalarDeriv_eq`).

Trains on `verified_mlir/convnext_train_step.mlir` (180 params), evals via
`verified_mlir/convnext_fwd.mlir` — both rendered by tests/TestConvNeXt{Train,Fwd}.lean
from the same [3,3,9,3] generator, every op fragment the StableHLO of a proven-faithful
emitter: GELU (`gelu_has_vjp_correct`, `geluF`/`geluBack`), LayerNorm (global scalar BN,
`bn_grad_input`, `bnF`/`bnBack`), layerScale (`layerScale_has_vjp`), depthwise 7×7
(`depthwiseFlat_has_vjp`), 1×1 convs, the even-kernel 4×4/s4 patchify + 2×2/s2 downsamples
(hand-verified transposed backward), residual `addV`, GAP, dense. Both MLIRs iree-compile
to ROCm. NB eval uses batch stats; the paper recipe (AdamW/cosine/stoch-depth/EMA) is out
of scope — the win is verified codegen on native-resolution data.

180 params packed per `ConvNeXtLayout` (LN γ/β rank-0 scalars `#[]`; layerScale γ rank-1
`[c]`; depthwise `[c,1,7,7]`; patchify `[96,3,4,4]`; downsample `[2c,c,2,2]`). Reuses the
params-general `mlpTrainStepV` FFI. He init for conv/dense weights (depthwise fan-in 49,
patchify 48), LN γ = layerScale γ = 1, β = 0, biases = 0; mean-loss SGD lr=0.1 (baked in).

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/convnext-verified data`
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
    fan-in = `ic·kH·kW` for a rank-4 conv kernel — depthwise `[c,1,7,7]` ⇒ 49,
    patchify `[96,3,4,4]` ⇒ 48 — or `in` for the rank-2 dense matrix), γ = 1 (kind 1;
    LN scalar / layerScale `[c]`), β / bias = 0 (kind 2; including rank-0 LN β). -/
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
  IO.println "ConvNeXt-T on Imagenette 224² (patchify /4 → [3,3,9,3] blocks @ [96,192,384,768] with depthwise-7×7 + LN + GELU + layerScale + 3 downsamples 56→7 → GAP → LN → dense 10) via the VERIFIED renderer → IREE FFI → GPU"
  compileVmfb "verified_mlir/convnext_train_step.mlir" ".lake/build/convnext_ts_v.vmfb"
  compileVmfb "verified_mlir/convnext_fwd.mlir"        ".lake/build/convnext_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/convnext_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/convnext_fwd_v.vmfb"
  let idir := dataDir ++ "/imagenette"
  -- train stored at 256² (center-crop to 224 per batch); val at 224²
  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (idir ++ "/train.bin") 256
  let (valImg,   valLbl,   nVal)   ← F32.loadImagenette (idir ++ "/val.bin")
  IO.println s!"  train {nTrain}, val {nVal}; bs {BS}, ConvNeXt-T 224² ({ConvNeXtLayout.specs.size} params, {ConvNeXtLayout.nParams} floats), global-scalar LN, mean-loss SGD lr=0.1, He init"
  (← IO.getStdout).flush
  let nb  := nTrain / BS
  let nbt := nVal / BS
  let shapes := ConvNeXtLayout.shapesBA
  let xShape := ConvNeXtLayout.xShape BS
  -- init the 180 params in func-arg order from the layout specs
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in ConvNeXtLayout.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut params := F32.concat parts
  for ep in [0:20] do
    for bi in [0:nb] do
      let xb256 := F32.sliceImages trainImg (bi * BS) BS TRAINPIX
      let xb ← F32.centerCrop xb256 BS.toUSize 3 256 256 224 224   -- 256→224 (deterministic)
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.convnext_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages valImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.convnext_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (valLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: val_acc = {correct}/{nbt * BS} = {acc}%"
    (← IO.getStdout).flush
  IO.println "done (trained ConvNeXt-T on Imagenette via the proof-rendered StableHLO)."
