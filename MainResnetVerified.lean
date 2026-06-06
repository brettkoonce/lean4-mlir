import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `resnet-verified` — train a ResNet-style net on the VERIFIED-rendered codegen

Chapter 6: the residual/skip + global-average-pool architecture whose whole-network
VJP is the already-audited `Proofs.cnn_has_vjp_at` (discharged unconditionally by
`CnnConcrete.cnnConcrete_has_vjp_correct`):

  convBnRelu(stem 3→32, 32×32) → maxpool 16×16 →
  identity block  (relu(F(x) + x),       32→32) →
  projection block(relu(proj(x) + F'(x)), 32→64) →
  global-average-pool → dense 64→10  + softmax-CE

Trains on `verified_mlir/resnet_train_step.mlir` (`Proofs.StableHLO.resnetTrainStepText`).
Every op is proof-backed:
* forward conv / maxpool / dense / relu + per-example BN (`resnetFwdGraph_faithful`,
  `bnF_faithful`), **residual add** (`addV` → `+`, `den_addV`), **global-average-pool**
  (`gapF` → `globalAvgPoolFlat`, `gapF_faithful`);
* the residual backward is the proven fan-IN `dx = f.back(dy) + skip(dy)`
  (`residual_has_vjp` / `residualProj_has_vjp`); BN input-VJP is the consolidated
  O(N) three-term `bn_grad_input` (`bnBack_faithful`); GAP backward broadcasts
  `dy/(H·W)` (`globalAvgPoolFlat_has_vjp`); conv / maxpool / dense grads + SGD as before;
* whole-network backward = `cnn_has_vjp_at` — audited 3-axiom-clean.

Reuses `mlpTrainStepV` (params-general FFI). 26 params packed per `ResnetLayout`
(γ/β are rank-0 scalars). Forward eval module `resnet_fwd.mlir` is rendered from the
verified AST `resnetFwdGraph` (the residual skips recompute their subtree — a tree,
not a DAG — so the forward graph is larger than a CSE'd one but semantically exact).

The BN here is the proven per-example **global, scalar** γ/β normalization (as in
Chapter 5), NOT per-channel batch-BN — so as in `cifar-bn-verified` it is unlikely
to beat the no-BN baselines on accuracy. The Chapter-6 result is a *correctness /
codegen* one: residual connections + GAP + the deep block structure are verified
end-to-end and GPU-trained, not a SOTA number.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/resnet-verified data`
-/

private def IC : Nat := 3
private def C  : Nat := 32
private def OC : Nat := 64
private def KH : Nat := 3
private def KW : Nat := 3
private def NCLASS : Nat := 10
private def D0 : Nat := 3072
private def BS : Nat := 128

private def compileVmfb (mlirPath outPath : String) : IO Unit := do
  let cargs ← ireeCompileArgs mlirPath outPath
  IO.println s!"  iree-compile {mlirPath}"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    throw (IO.userError s!"iree-compile failed:\n{r.stderr.take 2000}")

/-- Load CIFAR-10 `.bin` records (3073 bytes: 1 label + 3072 image). Returns f32
    images `[n×3072]` (normalized) and int32-LE labels `[n×4]`. -/
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

def main (argv : List String) : IO Unit := do
  let dataDir := argv.head?.getD "data"
  IO.println "ResNet-style net (stem→pool→id-block→proj-block→GAP→dense) via the VERIFIED renderer → IREE FFI → GPU"
  compileVmfb "verified_mlir/resnet_train_step.mlir" ".lake/build/resnet_ts_v.vmfb"
  compileVmfb "verified_mlir/resnet_fwd.mlir"        ".lake/build/resnet_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/resnet_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/resnet_fwd_v.vmfb"
  let cdir := dataDir ++ "/cifar-10"
  let trainPaths := (List.range 5).map (fun i => s!"{cdir}/data_batch_{i+1}.bin")
  let (trainImg, trainLbl, nTrain) ← loadCifarSplit trainPaths
  let (testImg,  testLbl,  nTest)  ← loadCifarSplit [s!"{cdir}/test_batch.bin"]
  IO.println s!"  train {nTrain}, test {nTest}; bs {BS}, residual+GAP, BN + SGD lr=0.1 (mean-loss), He init, γ=1 β=0"
  (← IO.getStdout).flush
  let nb  := nTrain / BS
  let nbt := nTest / BS
  let shapes := ResnetLayout.shapesBA
  let xShape := ResnetLayout.xShape BS
  let mkW (seed n fanIn : Nat) : IO ByteArray :=
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))
  let one (_ : Unit) : IO ByteArray := F32.const 1 1.0
  let zero (_ : Unit) : IO ByteArray := F32.const 1 0.0
  -- stem
  let Ws ← mkW 1 (C*IC*KH*KW) (IC*KH*KW); let bs ← F32.const C.toUSize 0.0
  let gs ← one (); let bts ← zero ()
  -- identity block
  let W1 ← mkW 2 (C*C*KH*KW) (C*KH*KW);   let b1 ← F32.const C.toUSize 0.0
  let g1 ← one (); let bt1 ← zero ()
  let W2 ← mkW 3 (C*C*KH*KW) (C*KH*KW);   let b2 ← F32.const C.toUSize 0.0
  let g2 ← one (); let bt2 ← zero ()
  -- projection block
  let W1p ← mkW 4 (OC*C*KH*KW) (C*KH*KW);  let b1p ← F32.const OC.toUSize 0.0
  let g1p ← one (); let bt1p ← zero ()
  let W2p ← mkW 5 (OC*OC*KH*KW) (OC*KH*KW);let b2p ← F32.const OC.toUSize 0.0
  let g2p ← one (); let bt2p ← zero ()
  let Wp ← mkW 6 (OC*C*KH*KW) (C*KH*KW);   let bp ← F32.const OC.toUSize 0.0
  let gp ← one (); let btp ← zero ()
  -- dense head
  let Wd ← mkW 7 (OC*NCLASS) OC;           let bd ← F32.const NCLASS.toUSize 0.0
  let mut params := F32.concat #[Ws, bs, gs, bts, W1, b1, g1, bt1, W2, b2, g2, bt2,
                                 W1p, b1p, g1p, bt1p, W2p, b2p, g2p, bt2p, Wp, bp, gp, btp,
                                 Wd, bd]
  for ep in [0:10] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * BS) BS D0
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.resnet_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages testImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.resnet_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (testLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: test_acc = {correct}/{nbt * BS} = {acc}%"
    (← IO.getStdout).flush
  IO.println "done (trained on the proof-rendered ResNet-style StableHLO)."
