import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `vit-verified` — train ViT-Tiny on the VERIFIED-rendered codegen

Chapter 10: the Vision Transformer (Dosovitskiy et al. 2021, "An Image is Worth
16×16 Words") whose whole-network VJP is the audited, UNCONDITIONAL
`Proofs.vit_full_has_vjp(_correct)` (softmax / SDPA-3-path / MHSA / transformer-block /
patch-embed VJP stack — all smooth, only the `0<ε` LayerNorm conditions), now rendered
+ GPU-trained on IMAGENETTE 224×224, 10 classes (paper-native resolution, patch-16):

  patch   16×16 stride-16 conv (3→192) "patchify"            224→14×14 = 196 patches
  tokens  flatten → [196,192], prepend CLS → [197,192], + positional embed
  tower   12× pre-norm transformer block @ dim 192, 3 heads (d_head 64), MLP 768
  final   LayerNorm(192)  (per-channel [192] γ/β)
  head    CLS token (row 0) → dense 192→10 + softmax-CE

Transformer block (pre-norm): x → LN → MHSA → +x → LN → MLP(192→768→GELU→768→192) → +.
MHSA: Q/K/V dense → 3 heads × 64 → per-head SDPA softmax(QKᵀ/√64)·V → concat → out-proj.

Trains on `verified_mlir/vit_train_step.mlir` (200 params), evals via
`verified_mlir/vit_fwd.mlir` — both rendered by tests/TestViT{Train,Fwd}.lean from the
shared `ViTRender` fragments, every op the StableHLO of a proven-faithful emitter:
row-softmax (`softmaxRowF`/`softmaxRowBack`, the V1 op), batched multi-head SDPA
(`dot_general` batching_dims + the 3-path `sdpa_back_{Q,K,V}`), per-channel LayerNorm
(normalize ∘ affine), GELU (`geluF`/`geluBack`), even-kernel 16×16/s16 patch conv
(hand-verified weight-grad), residual `addV`, dense. The full assembly is numerically
gradcheck-validated (tests/TestSDPA, TestMHSA, TestViTBlock, TestViTTiny — all PASS).

LayerNorm γ/β are per-channel `[192]` (the non-scalar form — beyond the scalar proof
witness `vit_full`, faithful per-op). 200 params packed per `ViTLayout`. He init for
conv/dense weights (patch fan-in 3·16·16=768, QKV/MLP from `in`), LN γ = 1, β/bias/CLS/
pos = 0; mean-loss SGD lr=0.1 (baked into the train step). ViT from scratch on small
data is notoriously hard — a low Imagenette number is fine; the win is verified attention
codegen on native-resolution data. Per-token LN ⇒ eval uses the same stats as train.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/vit-verified data`
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
    fan-in = `ic·kH·kW` for a rank-4 conv kernel — patch `[192,3,16,16]` ⇒ 768 — or
    `in` for the rank-2 dense matrix), γ = 1 (kind 1), β / bias / CLS / pos = 0 (kind 2). -/
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
  IO.println "ViT-Tiny on Imagenette 224² (patch-16 → CLS+pos → 12 transformer blocks @ dim192/3heads/MLP768 → final LN → CLS-head 10) via the VERIFIED renderer → IREE FFI → GPU"
  compileVmfb "verified_mlir/vit_train_step.mlir" ".lake/build/vit_ts_v.vmfb"
  compileVmfb "verified_mlir/vit_fwd.mlir"        ".lake/build/vit_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/vit_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/vit_fwd_v.vmfb"
  let idir := dataDir ++ "/imagenette"
  let (trainImg, trainLbl, nTrain) ← F32.loadImagenetteSized (idir ++ "/train.bin") 256
  let (valImg,   valLbl,   nVal)   ← F32.loadImagenette (idir ++ "/val.bin")
  IO.println s!"  train {nTrain}, val {nVal}; bs {BS}, ViT-Tiny 224² patch-16 ({ViTLayout.specs.size} params, {ViTLayout.nParams} floats), per-channel LN, mean-loss SGD lr=0.1, He init"
  (← IO.getStdout).flush
  let nb  := nTrain / BS
  let nbt := nVal / BS
  let shapes := ViTLayout.shapesBA
  let xShape := ViTLayout.xShape BS
  -- init the 200 params in func-arg order from the layout specs
  let mut parts : Array ByteArray := #[]
  let mut seed := 1
  for spec in ViTLayout.specs do
    parts := parts.push (← mkParam seed spec.1 spec.2)
    seed := seed + 1
  let mut params := F32.concat parts
  for ep in [0:20] do
    for bi in [0:nb] do
      let xb256 := F32.sliceImages trainImg (bi * BS) BS TRAINPIX
      let xb ← F32.centerCrop xb256 BS.toUSize 3 256 256 224 224   -- 256→224 (deterministic)
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.vit_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages valImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.vit_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (valLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: val_acc = {correct}/{nbt * BS} = {acc}%"
    (← IO.getStdout).flush
  IO.println "done (trained ViT-Tiny on Imagenette via the proof-rendered StableHLO)."
