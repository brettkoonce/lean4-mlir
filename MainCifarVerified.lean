import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `cifar-verified` — train the CIFAR-10 CNN on the VERIFIED-rendered codegen

Chapter 5 (no BatchNorm): `conv 3→32 → relu → conv 32→32 → relu → maxpool 32→16
→ conv 32→64 → relu → conv 64→64 → relu → maxpool 16→8 → flatten 4096 →
dense 4096→512 → relu → dense 512→512 → relu → dense 512→10` + softmax-CE.
Trains on `verified_mlir/cifar_train_step.mlir`
(`Proofs.StableHLO.cifarTrainStepText`), whose forward/backward/grad ops are each
proven faithful to the Mathlib `fderiv` math (`cifarFwdGraph_faithful`,
`convBack_faithful`, `maxPoolBack_faithful`, `reluF_faithful`,
`selectPos_faithful`, `wGrad/bGrad_is*Jacobian`, `lossCotGraph_isCEgrad`; the
whole-network VJP is `cifarCnn_has_vjp_at`) — audited 3-axiom-clean. The two
conv weight grads per stage are the transpose-trick render, validated by this run.

Reuses the params-general FFI binding `mlpTrainStepV` (x, packed params per
`CifarLayout.shapesBA` incl. 4-D conv kernels, int32 labels → one-hot in C).

**Observed result (measured, GPU/ROCm):** with textbook He init
(conv fanin = ic·kH·kW) and the verified trainers' mean-loss learning rate
(`lr = 0.1/128`, i.e. plain SGD lr = 0.1 on the *mean* batch loss), this no-BN
net **does train** — test accuracy climbs to ~66% (peak 66.9% at epoch 8, ~65.6%
at epoch 10), no augmentation / schedule. The Chapter-5 handoff predicted a
no-BN *failure* (~10%, loss ≈ log 10); empirically the failure is **not
intrinsic** — it is sensitive to the lr scale and init. The production
`MainCifarCnnTrain` stalls at random under its lr convention (literal `0.1` on
the summed-batch gradient is ~128× larger here ⇒ divergence); under the
mean-loss convention every verified trainer uses, the proof-rendered conv stack
scales up and learns. So this milestone shows the verified codegen *training*
CIFAR end-to-end; the BN "lift" (Milestone B) is a smaller, init/lr-robustness
story than the handoff assumed, not a 10%→trains cliff.

Real path: Lean loop → IreeRuntime FFI → in-process IREE → GPU.
Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar-verified data`
-/

private def IC : Nat := 3
private def C1 : Nat := 32
private def C2 : Nat := 64
private def KH : Nat := 3
private def KW : Nat := 3
private def FLAT : Nat := 4096      -- C2 * 8 * 8
private def D1 : Nat := 512
private def NCLASS : Nat := 10
private def D0 : Nat := 3072        -- IC * 32 * 32 (flattened input width)
private def BS : Nat := 128

private def compileVmfb (mlirPath outPath : String) : IO Unit := do
  let cargs ← ireeCompileArgs mlirPath outPath
  IO.println s!"  iree-compile {mlirPath}"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    throw (IO.userError s!"iree-compile failed:\n{r.stderr.take 2000}")

/-- Load CIFAR-10 `.bin` records (3073 bytes each: 1 label byte + 3072 image
    bytes). Returns images as f32 `[n×3072]` normalized to [0,1] (`F32.cifarBatch`)
    and labels as int32-LE `[n×4]` (label byte then three zero bytes). -/
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
  IO.println "CIFAR-10 CNN via the VERIFIED renderer (3→32→32→pool→32→64→64→pool→512→512→10) → IREE FFI → GPU"
  compileVmfb "verified_mlir/cifar_train_step.mlir" ".lake/build/cifar_ts_v.vmfb"
  compileVmfb "verified_mlir/cifar_fwd.mlir"        ".lake/build/cifar_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/cifar_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/cifar_fwd_v.vmfb"
  let cdir := dataDir ++ "/cifar-10"
  let trainPaths := (List.range 5).map (fun i => s!"{cdir}/data_batch_{i+1}.bin")
  let (trainImg, trainLbl, nTrain) ← loadCifarSplit trainPaths
  let (testImg,  testLbl,  nTest)  ← loadCifarSplit [s!"{cdir}/test_batch.bin"]
  IO.println s!"  train {nTrain}, test {nTest}; bs {BS}, no-BN SGD lr=0.1 (mean-loss), He init"
  let nb  := nTrain / BS
  let nbt := nTest / BS
  let shapes := CifarLayout.shapesBA
  let xShape := CifarLayout.xShape BS
  -- He-init weights (conv fanin = ic·kH·kW), zero biases; pack in arg order
  -- W1|b1|…|W7|b7 (= CifarLayout.paramShapes).
  let mkW (seed n fanIn : Nat) : IO ByteArray :=
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))
  let W1 ← mkW 1 (C1*IC*KH*KW)   (IC*KH*KW)
  let b1 ← F32.const C1.toUSize 0.0
  let W2 ← mkW 2 (C1*C1*KH*KW)   (C1*KH*KW)
  let b2 ← F32.const C1.toUSize 0.0
  let W3 ← mkW 3 (C2*C1*KH*KW)   (C1*KH*KW)
  let b3 ← F32.const C2.toUSize 0.0
  let W4 ← mkW 4 (C2*C2*KH*KW)   (C2*KH*KW)
  let b4 ← F32.const C2.toUSize 0.0
  let W5 ← mkW 5 (FLAT*D1)       FLAT
  let b5 ← F32.const D1.toUSize 0.0
  let W6 ← mkW 6 (D1*D1)         D1
  let b6 ← F32.const D1.toUSize 0.0
  let W7 ← mkW 7 (D1*NCLASS)     D1
  let b7 ← F32.const NCLASS.toUSize 0.0
  let mut params := F32.concat #[W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7]
  for ep in [0:10] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * BS) BS D0
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.cifar_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages testImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.cifar_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (testLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: test_acc = {correct}/{nbt * BS} = {acc}%"
  IO.println "done (trained on the proof-rendered CIFAR CNN StableHLO — no-BN SGD)."
