import LeanMlir

/-! Char-level bigram language model on tinyshakespeare.

The simplest language model: one dense layer `V → V`, predicting
P(next_char | current_char). ~4K params. Trains in seconds.

This isn't the tinyGPT demo — that needs new backward codegen for
`tokenPositionEmbed` / `lmHead`. The bigram exists to (a) prove the
new Shakespeare data pipeline (tokenizer → int32 streams → random
chunk sampling → one-hot → trainStep) works end-to-end, and (b) ship
a working text generator today against the existing classification
train-step ABI.

Subcommands:
  lake exe bigram-shakespeare train [epochs=30] [batch=512]
  lake exe bigram-shakespeare sample [n_chars=500] [temperature=1.0] [prompt="ROMEO:\n"]

The trainer leans on the existing `trainStepAdamF32` ABI: one-hot
inputs `[B, V]` f32 + int32 next-token labels `[B]`. Sampling uses the
eval forward (pure inference). -/

def vocabSize : Nat := 65

def bigramSpec : NetSpec where
  name := "bigram-shakespeare"
  imageH := 1
  imageW := vocabSize
  layers := [
    .dense vocabSize vocabSize .identity
  ]

def bigramConfig : TrainConfig where
  learningRate := 0.05
  batchSize    := 512
  epochs       := 30
  useAdam      := true
  weightDecay  := 0.0
  cosineDecay  := true
  warmupEpochs := 1
  augment      := false
  labelSmoothing := 0.0

/-- Load vocab.txt (id<TAB>byte<TAB>repr) into `Array UInt8`, indexed by token id. -/
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

/-- Build a reverse map byte→id from a vocab array. -/
def reverseVocab (vocab : Array UInt8) : Array Nat := Id.run do
  let mut m : Array Nat := Array.replicate 256 0
  for i in [:vocab.size] do
    m := m.set! vocab[i]!.toNat i
  return m

/-- Decode an int32 LE ByteArray of token IDs to a String via `vocab`. -/
def decodeTokens (tokens : ByteArray) (count : Nat) (vocab : Array UInt8) : String := Id.run do
  let mut s : String := ""
  for i in [:count] do
    let id := tokens.data[i * 4]!.toNat
      ||| tokens.data[i * 4 + 1]!.toNat <<< 8
      ||| tokens.data[i * 4 + 2]!.toNat <<< 16
      ||| tokens.data[i * 4 + 3]!.toNat <<< 24
    if id < vocab.size then
      s := s ++ (Char.ofNat vocab[id]!.toNat).toString
  return s

/-- Read N float32s from a ByteArray starting at element offset 0. -/
def readFloats (ba : ByteArray) (n : Nat) : Array Float := Id.run do
  let mut out := Array.mkEmpty n
  for i in [:n] do
    out := out.push (F32.read ba i.toUSize)
  return out

/-- Sample one token from a logits array via softmax + temperature.
    `temperature` ≤ 0 → argmax (greedy). -/
def sampleToken (logits : Array Float) (temperature : Float) (seed : USize) : Nat := Id.run do
  if temperature <= 0.0 then
    let mut best : Nat := 0
    let mut bestV : Float := logits[0]!
    for i in [1:logits.size] do
      if logits[i]! > bestV then bestV := logits[i]!; best := i
    return best
  -- Softmax with temperature, then sample by xorshift uniform.
  let mut mx : Float := logits[0]!
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
  -- xorshift64 → uniform [0, 1).  USize.size is the byte width, not the
  -- max value; convert via a millionth-mod for a stable [0, 1) draw.
  let mut s := seed; if s == 0 then s := 1
  s := s ^^^ (s <<< 13); s := s ^^^ (s >>> 7); s := s ^^^ (s <<< 17)
  let denom : Float := 1000003.0
  let r : Float := (s.toNat % 1000003).toFloat / denom
  let mut acc : Float := 0.0
  for i in [:probs.size] do
    acc := acc + probs[i]! * inv
    if r <= acc then return i
  return probs.size - 1

/-- Run the train loop. Returns the trained params + loss history. -/
def runBigramTrain (epochs : Nat) (batch : Nat) (seqLen : Nat) : IO (ByteArray × Array Float) := do
  let spec := bigramSpec
  let cfg : TrainConfig := { bigramConfig with epochs := epochs, batchSize := batch }
  IO.eprintln s!"compiling vmfbs for {spec.name} (V={vocabSize}, batch={batch}) ..."
  let _ ← spec.compileVmfbs cfg
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
  let nT := nP * 3

  let stepsPerEpoch : Nat := 200
  IO.eprintln s!"training: {epochs} epochs × {stepsPerEpoch} steps × batch {batch} (seqLen {seqLen} per chunk)"
  let mut history : Array Float := #[]
  let mut globalStep : Nat := 0
  for epoch in [:epochs] do
    let mut sumLoss : Float := 0.0
    for _step in [:stepsPerEpoch] do
      -- Sample fresh B*seqLen pairs per step.
      let seedU : USize := (epoch * 100003 + globalStep * 17 + 31).toUSize
      let chunks ← F32.sampleChunks tokens nTokens (batch * seqLen).toUSize 1 seedU
      -- chunks layout: first half is input ids [B*seqLen], second half is targets.
      let inputBytes := batch * seqLen * 4
      let inputIds := chunks.extract 0 inputBytes
      let targetIds := chunks.extract inputBytes (2 * inputBytes)
      -- One-hot encode input ids → [B*seqLen, V] f32.
      let xba ← F32.tokenOneHot inputIds (batch * seqLen).toUSize 1 vocabSize.toUSize
      -- targets are int32 [B*seqLen]; that's our `y`.
      let lr := cfg.learningRate
      let t : Float := (globalStep + 1).toFloat
      let outBA ← IreeSession.trainStepAdamF32 trainSess spec.trainFnName
                    packed allShapes xba xShape targetIds lr t bnShapes (batch * seqLen).toUSize
      packed := outBA.extract 0 (nT * 4)
      let loss := F32.read outBA nT.toUSize
      sumLoss := sumLoss + loss
      globalStep := globalStep + 1
    let avg := sumLoss / stepsPerEpoch.toFloat
    history := history.push avg
    IO.eprintln s!"  epoch {epoch + 1}/{epochs}: loss={avg}"
  let finalParams := packed.extract 0 (nP * 4)
  return (finalParams, history)

/-- Sample text autoregressively. -/
def runBigramSample (paramsPath : String) (nChars : Nat)
    (temperature : Float) (prompt : String) : IO String := do
  let spec := bigramSpec
  -- Compile eval vmfb at batch 1 (we sample one token at a time).
  let evalCfg : TrainConfig := { bigramConfig with batchSize := 1 }
  let _ ← spec.compileVmfbs evalCfg
  let pfx := spec.buildPrefix
  let evalVmfb := s!"{pfx}_fwd_eval.vmfb"
  let sess ← IreeSession.create evalVmfb

  let params ← IO.FS.readBinFile paramsPath
  -- The eval forward also wants a (zero-sized for no-BN) bn-stats append.
  let nBn := spec.nBnStats
  let bnPad ← F32.const nBn.toUSize 0.0
  let evalParams := params.append bnPad

  let vocab ← loadVocab "data/shakespeare/vocab.txt"
  let rev := reverseVocab vocab

  let evalShapesBA := spec.evalShapesBA
  let xShape := spec.xShape 1

  -- Encode prompt → list of token ids.
  let mut tokIds : List Nat := prompt.toList.map fun c => rev[c.toNat]!
  let nClasses : USize := vocabSize.toUSize

  -- Autoregressive loop.
  for _ in [:nChars] do
    let lastId := tokIds.getLast?.getD 0
    -- One-hot [1, V].
    let mut xba := ByteArray.emptyWithCapacity (vocabSize * 4)
    for v in [:vocabSize] do
      let f := if v == lastId then (1.0 : Float) else (0.0 : Float)
      xba := xba.append (← F32.const 1 f)
    let logits ← IreeSession.forwardF32 sess spec.evalFnName
                    evalParams evalShapesBA xba xShape 1 nClasses
    let arr := readFloats logits vocabSize
    let seedU : USize := (tokIds.length * 7919 + 1).toUSize
    let next := sampleToken arr temperature seedU
    tokIds := tokIds ++ [next]

  return decodeTokens (← (do
    let mut buf : ByteArray := ByteArray.emptyWithCapacity (tokIds.length * 4)
    for id in tokIds do
      buf := buf.append (← do
        let mut b : ByteArray := .empty
        b := b.push (id % 256).toUInt8
        b := b.push ((id / 256) % 256).toUInt8
        b := b.push ((id / 65536) % 256).toUInt8
        b := b.push ((id / 16777216) % 256).toUInt8
        return b)
    pure buf)) tokIds.length vocab

def main (args : List String) : IO Unit := do
  match args with
  | "train" :: rest =>
    let epochs := (rest.head?.bind String.toNat?).getD 30
    let batch := (rest.tail.head?.bind String.toNat?).getD 512
    let seqLen := 1
    let (params, _hist) ← runBigramTrain epochs batch seqLen
    let pfx := bigramSpec.buildPrefix
    let outPath := s!"{pfx}_params.bin"
    IO.FS.writeBinFile outPath params
    IO.eprintln s!"saved params to {outPath}"
  | "sample" :: rest =>
    let nChars := (rest.head?.bind String.toNat?).getD 500
    -- Temperature passed as an integer x100 to dodge `String.toFloat?`
    -- (not in this Lean stdlib). e.g. `100` → 1.0, `80` → 0.8, `0` → greedy.
    let tempPct := (rest.tail.head?.bind String.toNat?).getD 100
    let temperature : Float := tempPct.toFloat / 100.0
    let prompt := (rest.tail.tail.head?).getD "ROMEO:\n"
    let pfx := bigramSpec.buildPrefix
    let paramsPath := s!"{pfx}_params.bin"
    if !(← System.FilePath.pathExists paramsPath) then
      IO.eprintln s!"missing {paramsPath}; run `train` first"
      IO.Process.exit 1
    let text ← runBigramSample paramsPath nChars temperature prompt
    IO.println "════════════════ SAMPLE ════════════════"
    IO.println text
    IO.println "════════════════════════════════════════"
  | _ =>
    IO.eprintln "usage: bigram-shakespeare train [epochs=30] [batch=512]"
    IO.eprintln "       bigram-shakespeare sample [n_chars=500] [temperature=1.0] [prompt='ROMEO:\\n']"
    IO.Process.exit 1
