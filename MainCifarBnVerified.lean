import LeanMlir.Types
import LeanMlir.F32Array
import LeanMlir.IreeRuntime

/-! # `cifar-bn-verified` — train the CIFAR-10 CNN **with BatchNorm** on the
VERIFIED-rendered codegen

Chapter 5, BatchNorm variant: `conv 3→32 → BN → relu → conv 32→32 → BN → relu →
maxpool → conv 32→64 → BN → relu → conv 64→64 → BN → relu → maxpool → flatten →
dense 4096→512 → relu → dense 512→512 → relu → dense 512→10` + softmax-CE. The BN
is the proven per-example normalization (`bnForward`: reduce μ/var over the whole
oc·H·W feature vec, scalar γ/β), inserted after each conv.

Trains on `verified_mlir/cifar_bn_train_step.mlir`
(`Proofs.StableHLO.cifarBnTrainStepText`). Every op is proof-backed:
* forward conv/maxpool/dense/relu + per-example **BN** (`cifarBnFwdGraph_faithful`,
  `bnF_faithful`);
* BN input-VJP — the consolidated O(N) three-term gradient (`bnBack_faithful`,
  `bn_input_grad_correct`); BN param grads `dγ = Σ dy·x̂`, `dβ = Σ dy`;
* conv input-VJP / weight-grad, maxpool `select_and_scatter`, dense grads, SGD;
* whole-network backward = `cifarCnnBn_has_vjp_at` — audited 3-axiom-clean.

Reuses `mlpTrainStepV` (params-general FFI). The four BN layers each add a scalar
γ (init 1) and β (init 0), packed per `CifarBnLayout` (γ/β are rank-0 tensors).

**Observed result (measured, GPU/ROCm):** trains healthily and monotonically —
test acc 19% → 57.1% over 10 epochs (still climbing), no divergence. Notably this
is *below* the no-BN `cifar-verified` (~66% in the same budget). That is faithful,
not a bug: the proven `bnForward` is a **per-example GLOBAL normalization** (reduce
μ/var over the entire oc·H·W feature vec) with a single **scalar** γ/β per layer —
far more aggressive than per-channel batch-BN, and a scalar affine can't restore
per-channel scale, so it slows early feature learning here. The Chapter-5 point
that lands is the hard one: BN's forward AND its consolidated O(N) three-term
input-VJP are *verified* (`bnBack_faithful`, `cifarCnnBn_has_vjp_at`) and run on
GPU. The accuracy "lift" the handoff imagined is a per-channel-BN / init-robustness
story; this faithful scalar/per-example BN is a different (more constrained) object.

Companion to `cifar-verified` (no-BN, ~66%). Run both to compare BN vs no-BN.
Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/cifar-bn-verified data`
-/

private def IC : Nat := 3
private def C1 : Nat := 32
private def C2 : Nat := 64
private def KH : Nat := 3
private def KW : Nat := 3
private def FLAT : Nat := 4096
private def D1 : Nat := 512
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
  IO.println "CIFAR-10 CNN + BatchNorm via the VERIFIED renderer (conv→BN→relu ×4, 2 pools, 512→512→10) → IREE FFI → GPU"
  compileVmfb "verified_mlir/cifar_bn_train_step.mlir" ".lake/build/cifar_bn_ts_v.vmfb"
  compileVmfb "verified_mlir/cifar_bn_fwd.mlir"        ".lake/build/cifar_bn_fwd_v.vmfb"
  let tsSess  ← IreeSession.create ".lake/build/cifar_bn_ts_v.vmfb"
  let fwdSess ← IreeSession.create ".lake/build/cifar_bn_fwd_v.vmfb"
  let cdir := dataDir ++ "/cifar-10"
  let trainPaths := (List.range 5).map (fun i => s!"{cdir}/data_batch_{i+1}.bin")
  let (trainImg, trainLbl, nTrain) ← loadCifarSplit trainPaths
  let (testImg,  testLbl,  nTest)  ← loadCifarSplit [s!"{cdir}/test_batch.bin"]
  IO.println s!"  train {nTrain}, test {nTest}; bs {BS}, BN + SGD lr=0.1 (mean-loss), He init, γ=1 β=0"
  let nb  := nTrain / BS
  let nbt := nTest / BS
  let shapes := CifarBnLayout.shapesBA
  let xShape := CifarBnLayout.xShape BS
  let mkW (seed n fanIn : Nat) : IO ByteArray :=
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))
  -- per-BN scalar γ (init 1), β (init 0)
  let g1 ← F32.const 1 1.0; let bt1 ← F32.const 1 0.0
  let g2 ← F32.const 1 1.0; let bt2 ← F32.const 1 0.0
  let g3 ← F32.const 1 1.0; let bt3 ← F32.const 1 0.0
  let g4 ← F32.const 1 1.0; let bt4 ← F32.const 1 0.0
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
  let mut params := F32.concat #[W1, b1, g1, bt1, W2, b2, g2, bt2, W3, b3, g3, bt3,
                                 W4, b4, g4, bt4, W5, b5, W6, b6, W7, b7]
  for ep in [0:10] do
    for bi in [0:nb] do
      let xb := F32.sliceImages trainImg (bi * BS) BS D0
      let yb := F32.sliceLabels trainLbl (bi * BS) BS
      params ← IreeSession.mlpTrainStepV tsSess "m.cifar_bn_train_step"
                  xb params shapes yb BS.toUSize D0.toUSize NCLASS.toUSize
    let mut correct := 0
    for bi in [0:nbt] do
      let xb := F32.sliceImages testImg (bi * BS) BS D0
      let logits ← IreeSession.forwardF32 fwdSess "m.cifar_bn_fwd" params shapes
                      xb xShape BS.toUSize NCLASS.toUSize
      for j in [0:BS] do
        let pred := (F32.argmax10 logits (j * NCLASS).toUSize).toNat
        let lbl  := (testLbl.get! (4 * (bi * BS + j))).toNat
        if pred == lbl then correct := correct + 1
    let acc := correct.toFloat / (nbt * BS).toFloat * 100.0
    IO.println s!"  epoch {ep + 1}: test_acc = {correct}/{nbt * BS} = {acc}%"
  IO.println "done (trained on the proof-rendered BN-CIFAR StableHLO)."
