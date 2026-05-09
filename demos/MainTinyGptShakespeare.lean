import LeanMlir

/-! Char-level tinyGPT trained on tinyshakespeare.

Architecture (212K params):
  tokenPositionEmbed (V=65, T=64, D=64)  →  shapes [B, V*T] one-hot  →  [B, T, D]
  transformerEncoder (D=64, h=2, mlp=256, blocks=4, causalMask=true)  →  [B, T, D]
  lmHead (D=64, V=65, T=64)  →  [B, V, T, 1]  for per-token softmax CE

Training rides the existing `useSeg` path (per-pixel CE) by treating
the LM head's [B, V, T, 1] output as NCHW segmentation logits with
T positions × 1 width. Labels are `[B, T, 1]` int32 from the
random-chunk sampler.

Subcommands:
  lake exe tinygpt-shakespeare train [steps=2000] [batch=32]
  lake exe tinygpt-shakespeare sample [n_chars=400] [temperature_x100=80] [prompt="ROMEO:\n"]
-/

def vocabSize : Nat := 65
def seqLen    : Nat := 64
def dModel    : Nat := 64
def numHeads  : Nat := 2
def mlpDim    : Nat := 256
def numLayers : Nat := 4

def tinyGpt : NetSpec where
  name := "tinygpt-shakespeare"
  imageH := seqLen     -- so y_seg shape becomes [B, T, 1]
  imageW := 1
  layers := [
    .tokenPositionEmbed vocabSize seqLen dModel,
    .transformerEncoder dModel numHeads mlpDim numLayers (causalMask := true),
    .lmHead dModel vocabSize seqLen
  ]

def trainConfig : TrainConfig where
  learningRate := 0.003
  batchSize    := 32
  epochs       := 1
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := false
  warmupEpochs := 0
  augment      := false
  labelSmoothing := 0.0

/-- Load vocab.txt → byte values indexed by token id. -/
def loadVocab (path : String) : IO (Array UInt8) := do
  let raw ← IO.FS.readFile path
  let mut out : Array UInt8 := #[]
  for line in raw.splitOn "\n" do
    if line.isEmpty then continue
    let parts := line.splitOn "\t"
    if parts.length >= 2 then
      match parts[1]!.toNat? with
      | some b => out := out.push b.toUInt8
      | none => pure ()
  return out

def reverseVocab (vocab : Array UInt8) : Array Nat := Id.run do
  let mut m : Array Nat := Array.replicate 256 0
  for i in [:vocab.size] do
    m := m.set! vocab[i]!.toNat i
  return m

/-- Greedy / temperature-sampled token from a logits Array. -/
def sampleToken (logits : Array Float) (temperature : Float) (seed : USize) : Nat := Id.run do
  if temperature <= 0.0 then
    let mut best := 0
    let mut bestV := logits[0]!
    for i in [1:logits.size] do
      if logits[i]! > bestV then bestV := logits[i]!; best := i
    return best
  let mut mx := logits[0]!
  for i in [1:logits.size] do
    if logits[i]! > mx then mx := logits[i]!
  let invT := 1.0 / temperature
  let mut probs : Array Float := Array.mkEmpty logits.size
  let mut sum : Float := 0.0
  for v in logits do
    let p := ((v - mx) * invT).exp
    probs := probs.push p
    sum := sum + p
  let inv := 1.0 / sum
  let mut s := seed; if s == 0 then s := 1
  s := s ^^^ (s <<< 13); s := s ^^^ (s >>> 7); s := s ^^^ (s <<< 17)
  let r : Float := (s.toNat % 1000003).toFloat / 1000003.0
  let mut acc : Float := 0.0
  for i in [:probs.size] do
    acc := acc + probs[i]! * inv
    if r <= acc then return i
  return probs.size - 1

/-- Reshape [B*T] int32 ids → [B, T, 1] int32 (just a byte-identical
    rebind — labels' MLIR shape declaration is `tensor<BxTx1xi32>`). -/
def asSegLabels (ids : ByteArray) : ByteArray := ids

/-- Run training. Returns final params buffer and a loss history. -/
def runTinyGptTrain (steps : Nat) (batch : Nat) (lr : Float)
    : IO (ByteArray × Array Float) := do
  let spec := tinyGpt
  let cfg : TrainConfig := { trainConfig with batchSize := batch, learningRate := lr }
  IO.eprintln s!"compiling train step (B={batch}, T={seqLen}, V={vocabSize}, params={spec.totalParams}) ..."
  let _ ← spec.compileVmfbs cfg (useSeg := true)
  let pfx := spec.buildPrefix
  let trainVmfb := s!"{pfx}_train_step.vmfb"
  let trainSess ← IreeSession.create trainVmfb

  IO.eprintln "loading data/shakespeare/train.bin ..."
  let (tokens, nTokens) ← F32.loadTokenStream "data/shakespeare/train.bin"
  IO.eprintln s!"  {nTokens} tokens"

  let mut params ← spec.heInitParams
  let nP := F32.size params
  let m0 ← F32.const nP.toUSize 0.0
  let v0 ← F32.const nP.toUSize 0.0
  let mut packed := F32.concat #[params, m0, v0]
  let allShapes := spec.shapesBA
  let xShape := spec.xShape batch
  let bnShapes := spec.bnShapesBA
  let nT3 := nP * 3

  IO.eprintln s!"training: {steps} steps × batch {batch} × T {seqLen}, lr={lr}, Adam"
  let mut history : Array Float := #[]
  for step in [:steps] do
    let seedU : USize := (step * 65537 + 17).toUSize
    let chunks ← F32.sampleChunks tokens nTokens batch.toUSize seqLen.toUSize seedU
    let inputBytes := batch * seqLen * 4
    let inputIds  := chunks.extract 0 inputBytes
    let targetIds := chunks.extract inputBytes (2 * inputBytes)
    -- One-hot encode → [B, V*T] f32 (matches inputFlatDim = V*T).
    let xba ← F32.tokenOneHot inputIds batch.toUSize seqLen.toUSize vocabSize.toUSize
    let t : Float := (step + 1).toFloat
    let outBA ← IreeSession.trainStepAdamF32Seg trainSess spec.trainFnName
                  packed allShapes xba xShape (asSegLabels targetIds) lr t
                  bnShapes batch.toUSize seqLen.toUSize 1
    packed := outBA.extract 0 (nT3 * 4)
    let loss := F32.read outBA nT3.toUSize
    history := history.push loss
    if step % 100 == 0 || step + 1 == steps then
      IO.eprintln s!"  step {step + 1}/{steps}: loss={loss}"
  let finalParams := packed.extract 0 (nP * 4)
  return (finalParams, history)

/-- Read N float32s from a ByteArray (element offset 0). -/
def readFloats (ba : ByteArray) (start n : Nat) : Array Float := Id.run do
  let mut out := Array.mkEmpty n
  for i in [:n] do
    out := out.push (F32.read ba (start + i).toUSize)
  return out

/-- Pack `[T]` int32 ids into a `[1, V*T]` f32 one-hot ByteArray. -/
def packOneHotContext (ids : Array Nat) (T V : Nat) : IO ByteArray := do
  -- Build a [T] int32 buffer, then call tokenOneHot.
  let mut idsBA : ByteArray := ByteArray.emptyWithCapacity (T * 4)
  for t in [:T] do
    let id := if t < ids.size then ids[t]! else 0
    idsBA := idsBA.push (id % 256).toUInt8
                |>.push ((id / 256) % 256).toUInt8
                |>.push ((id / 65536) % 256).toUInt8
                |>.push ((id / 16777216) % 256).toUInt8
  F32.tokenOneHot idsBA 1 T.toUSize V.toUSize

/-- Autoregressive sampling. -/
def runTinyGptSample (paramsPath : String) (nChars : Nat)
    (temperature : Float) (prompt : String) : IO String := do
  let spec := tinyGpt
  -- Compile eval forward at batch=1.
  let evalCfg : TrainConfig := { trainConfig with batchSize := 1 }
  let _ ← spec.compileVmfbs evalCfg (useSeg := true)
  let pfx := spec.buildPrefix
  let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
  let sess ← IreeSession.create evalVmfb

  let params ← IO.FS.readBinFile paramsPath
  -- Append zero-byte BN stats (no BN in this spec).
  let nBn := spec.nBnStats
  let bnPad ← F32.const nBn.toUSize 0.0
  let evalParams := params.append bnPad

  let vocab ← loadVocab "data/shakespeare/vocab.txt"
  let rev := reverseVocab vocab
  let evalShapesBA := spec.evalShapesBA
  let xShape := spec.xShape 1

  -- Encode prompt → list of token ids; left-pad with zeros to length T.
  let promptIds : Array Nat := prompt.toList.toArray.map fun c => rev[c.toNat]!
  IO.eprintln s!"  prompt encoded: {promptIds.size} ids"
  let mut context : Array Nat := Array.replicate seqLen 0
  -- Place prompt in the rightmost positions (so index T-1 is the most recent).
  let startCtx := if promptIds.size >= seqLen then 0 else seqLen - promptIds.size
  let promptStart := if promptIds.size >= seqLen then promptIds.size - seqLen else 0
  for i in [:seqLen - startCtx] do
    context := context.set! (startCtx + i) promptIds[promptStart + i]!

  let mut generated : Array Nat := #[]
  -- Forward returns a [1, T*V] flat tensor (the eval forward emits flat).
  let outElems : USize := (seqLen * vocabSize).toUSize
  for stepN in [:nChars] do
    let xba ← packOneHotContext context seqLen vocabSize
    let logitsFlat ← IreeSession.forwardF32 sess spec.evalFnName
                       evalParams evalShapesBA xba xShape 1 outElems
    -- logitsFlat layout: [1, T*V] reshape of [1, T, V]: row major in (1, t, v).
    -- We want logits at the LAST position = index (T-1) * V to T*V.
    let baseIdx := (seqLen - 1) * vocabSize
    let logits := readFloats logitsFlat baseIdx vocabSize
    let seedU : USize := (stepN * 7919 + promptIds.size + 1).toUSize
    let next := sampleToken logits temperature seedU
    generated := generated.push next
    -- Shift context left by 1, append next at end.
    for t in [:seqLen - 1] do
      context := context.set! t context[t + 1]!
    context := context.set! (seqLen - 1) next

  -- Decode (prompt + generated) to text.
  let mut s : String := ""
  for id in promptIds do
    if id < vocab.size then s := s ++ (Char.ofNat vocab[id]!.toNat).toString
  for id in generated do
    if id < vocab.size then s := s ++ (Char.ofNat vocab[id]!.toNat).toString
  return s

def main (args : List String) : IO Unit := do
  match args with
  | "train" :: rest =>
    let steps := (rest.head?.bind String.toNat?).getD 2000
    let batch := (rest.tail.head?.bind String.toNat?).getD 32
    let lrPct := (rest.tail.tail.head?.bind String.toNat?).getD 30  -- lr_x10000 default 0.003
    let lr : Float := lrPct.toFloat / 10000.0
    let (params, hist) ← runTinyGptTrain steps batch lr
    let pfx := tinyGpt.buildPrefix
    let outPath := s!"{pfx}_params.bin"
    IO.FS.writeBinFile outPath params
    IO.eprintln s!"saved params to {outPath}  (final loss: {hist.back!})"
  | "sample" :: rest =>
    let nChars := (rest.head?.bind String.toNat?).getD 400
    let tempPct := (rest.tail.head?.bind String.toNat?).getD 80
    let temperature : Float := tempPct.toFloat / 100.0
    let prompt := (rest.tail.tail.head?).getD "ROMEO:\n"
    let pfx := tinyGpt.buildPrefix
    let paramsPath := s!"{pfx}_params.bin"
    if !(← System.FilePath.pathExists paramsPath) then
      IO.eprintln s!"missing {paramsPath}; run `train` first"; IO.Process.exit 1
    let text ← runTinyGptSample paramsPath nChars temperature prompt
    IO.println "════════════════ SAMPLE ════════════════"
    IO.println text
    IO.println "════════════════════════════════════════"
  | _ =>
    IO.eprintln "usage: tinygpt-shakespeare train [steps=2000] [batch=32] [lr_x10000=30]"
    IO.eprintln "       tinygpt-shakespeare sample [n_chars=400] [temperature_x100=80] [prompt='ROMEO:\\n']"
    IO.Process.exit 1
