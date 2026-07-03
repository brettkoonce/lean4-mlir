import LeanMlir

/-! BPE-tokenized GPT trained on TinyStories (planning/tinygpt_demo_v2.md
    Part II).

    ~8.5M-param decoder-only transformer, vocab 4096, T=256:
      tokenPositionEmbed (V=4096, T=256, D=256, idsInput=true)
        → transformerEncoder (D=256, h=8, mlp=1024, blocks=8, causal)
        → lmHead (D=256, V=4096, T=256)

    Key difference from the char-level demo: the model input is [B, T]
    f32 token ids and the one-hot is built *inside* the MLIR graph
    (tokenPositionEmbed idsInput), so the host never uploads the
    [B, V·T] = [32, 4096·256] ≈ 134MB/step one-hot. Same per-token CE
    (`useSeg`) loss ride; vocab-agnostic.

    Reuses the Shakespeare data path verbatim: F32.loadTokenStream +
    F32.sampleChunks read the int32 stream `preprocess_tinystories.py`
    writes to data/tinystories/{train,val}.bin.

    Subcommands:
      lake exe tinystories train  [steps=12000] [batch=32] [lr_x10000=30]
      lake exe tinystories sample [n_toks=200] [temp_x100=80] [topk=40] [topp_x100=95] [seed=1] [prompt]
-/

def vocabSize : Nat := 4096
def seqLen    : Nat := 256
def dModel    : Nat := 256
def numHeads  : Nat := 8
def mlpDim    : Nat := 1024
def numLayers : Nat := 8

def storiesSpec : NetSpec :=
  { name := "tinystories-8m"
    imageH := seqLen
    imageW := 1
    layers := [
      .tokenPositionEmbed vocabSize seqLen dModel (idsInput := true),
      .transformerEncoder dModel numHeads mlpDim numLayers (causalMask := true),
      .lmHead dModel vocabSize seqLen
    ] }

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

def ln2 : Float := 0.6931471805599453

/-- Cosine decay + linear warmup (200 steps), host-side. -/
def lrAt (lrMax : Float) (step steps : Nat) : Float :=
  let warmup : Nat := 200
  if steps <= warmup then lrMax
  else if step < warmup then lrMax * (step + 1).toFloat / warmup.toFloat
  else
    let prog := (step - warmup).toFloat / (steps - warmup).toFloat
    let lrMin := lrMax * 0.1
    lrMin + 0.5 * (1.0 + Float.cos (3.141592653589793 * prog)) * (lrMax - lrMin)

def asSegLabels (ids : ByteArray) : ByteArray := ids

/-- Mean per-token CE (nats) over fixed val chunks — the in-graph
    one-hot path, so the model input is [B, T] f32 ids. -/
def evalValLoss (sess : IreeSession) (spec : NetSpec)
    (packed allShapes xShape bnShapes : ByteArray)
    (valTokens : ByteArray) (nValTokens : USize)
    (batch T : Nat) (nBatches : Nat) : IO Float := do
  let inputBytes := batch * T * 4
  let mut tot : Float := 0.0
  let nT3 := 3 * spec.totalParams
  for i in [:nBatches] do
    let seedU : USize := (0x57019000 + i * 9973).toUSize
    let chunks ← F32.sampleChunks valTokens nValTokens batch.toUSize T.toUSize seedU
    let inputIds  := chunks.extract 0 inputBytes
    let targetIds := chunks.extract inputBytes (2 * inputBytes)
    let xba ← F32.idsToFloats inputIds
    let outBA ← IreeSession.trainStepAdamF32Seg sess spec.trainFnName
                  packed allShapes xba xShape (asSegLabels targetIds) 0.0 1.0
                  bnShapes batch.toUSize T.toUSize 1
    tot := tot + F32.read outBA nT3.toUSize
  return tot / nBatches.toFloat

def runTrain (steps batch : Nat) (lrMax : Float) : IO (ByteArray × Float) := do
  let spec := storiesSpec
  let T := seqLen
  let cfg : TrainConfig := { trainConfig with batchSize := batch, learningRate := lrMax }
  IO.eprintln s!"compiling train step (B={batch}, T={T}, V={vocabSize}, params={spec.totalParams}) ..."
  let _ ← spec.compileVmfbs cfg (useSeg := true)
  let pfx := spec.buildPrefix
  let trainSess ← IreeSession.create s!"{pfx}_train_step.vmfb"

  IO.eprintln "loading data/tinystories/{train,val}.bin ..."
  let (tokens, nTokens) ← F32.loadTokenStream "data/tinystories/train.bin"
  let (valTokens, nValTokens) ← F32.loadTokenStream "data/tinystories/val.bin"
  IO.eprintln s!"  {nTokens} train tokens, {nValTokens} val tokens"

  let mut params ← spec.heInitParams
  let nP := F32.size params
  let m0 ← F32.const nP.toUSize 0.0
  let v0 ← F32.const nP.toUSize 0.0
  let mut packed := F32.concat #[params, m0, v0]
  let allShapes := spec.shapesBA
  let xShape := spec.xShape batch
  let bnShapes := spec.bnShapesBA
  let nT3 := nP * 3

  let evalEvery : Nat := 500
  IO.eprintln s!"training: {steps} steps × batch {batch} × T {T}, lr peak {lrMax} (cosine + 200 warmup), Adam"
  let mut lastLoss : Float := 0.0
  for step in [:steps] do
    let lr := lrAt lrMax step steps
    let seedU : USize := (step * 65537 + 17).toUSize
    let chunks ← F32.sampleChunks tokens nTokens batch.toUSize T.toUSize seedU
    let inputBytes := batch * T * 4
    let inputIds  := chunks.extract 0 inputBytes
    let targetIds := chunks.extract inputBytes (2 * inputBytes)
    let xba ← F32.idsToFloats inputIds
    let t : Float := (step + 1).toFloat
    let outBA ← IreeSession.trainStepAdamF32Seg trainSess spec.trainFnName
                  packed allShapes xba xShape (asSegLabels targetIds) lr t
                  bnShapes batch.toUSize T.toUSize 1
    packed := outBA.extract 0 (nT3 * 4)
    lastLoss := F32.read outBA nT3.toUSize
    if step % 100 == 0 || step + 1 == steps then
      IO.eprintln s!"  step {step + 1}/{steps}: loss={lastLoss} ({lastLoss / ln2} bits/tok) lr={lr}"
    if (step + 1) % evalEvery == 0 || step + 1 == steps then
      let valNats ← evalValLoss trainSess spec packed allShapes xShape bnShapes
                      valTokens nValTokens batch T 8
      IO.eprintln s!"  ── val @ step {step + 1}: {valNats} nats/tok = {valNats / ln2} bits/tok"
      -- Periodic checkpoint (long run; survive interruption).
      IO.FS.writeBinFile s!"{pfx}_params.bin" (packed.extract 0 (nP * 4))
  return (packed.extract 0 (nP * 4), lastLoss)

def readFloats (ba : ByteArray) (start n : Nat) : Array Float := Id.run do
  let mut out := Array.mkEmpty n
  for i in [:n] do out := out.push (F32.read ba (start + i).toUSize)
  return out

/-- Sample a token id with temperature + top-k + top-p. -/
def sampleToken (logits : Array Float) (temperature : Float)
    (topK : Nat) (topP : Float) (seed : USize) : Nat := Id.run do
  if temperature <= 0.0 then
    let mut best := 0; let mut bestV := logits[0]!
    for i in [1:logits.size] do
      if logits[i]! > bestV then bestV := logits[i]!; best := i
    return best
  let ids := (Array.range logits.size).qsort (fun a b => logits[a]! > logits[b]!)
  let keep := if topK == 0 || topK > ids.size then ids.size else topK
  let mx := logits[ids[0]!]!
  let invT := 1.0 / temperature
  let mut probs : Array Float := Array.mkEmpty keep
  let mut sum : Float := 0.0
  for i in [:keep] do
    let p := ((logits[ids[i]!]! - mx) * invT).exp
    probs := probs.push p; sum := sum + p
  let mut cut := keep
  if topP < 1.0 then
    let mut acc : Float := 0.0
    for i in [:keep] do
      if acc >= topP * sum then cut := i; break
      acc := acc + probs[i]!
  let mut cutSum : Float := 0.0
  for i in [:cut] do cutSum := cutSum + probs[i]!
  let mut s := seed; if s == 0 then s := 1
  s := s ^^^ (s <<< 13); s := s ^^^ (s >>> 7); s := s ^^^ (s <<< 17)
  let r : Float := (s.toNat % 1000003).toFloat / 1000003.0
  let mut acc : Float := 0.0
  for i in [:cut] do
    acc := acc + probs[i]! / cutSum
    if r <= acc then return ids[i]!
  return ids[cut - 1]!

/-- Autoregressive sampling. Prompt token ids are supplied pre-encoded
    (one BPE id per line) via `data/tinystories/prompt_ids.txt`, or fall
    back to a single eot-primed start. Decoding is done in Python
    afterward (BPE detokenization lives in the tokenizer), so this dumps
    generated ids to stdout as space-separated integers. -/
def runSample (nToks : Nat) (temperature : Float) (topK : Nat)
    (topP : Float) (userSeed : Nat) (promptIds : Array Nat) : IO (Array Nat) := do
  let spec := storiesSpec
  let T := seqLen
  let evalCfg : TrainConfig := { trainConfig with batchSize := 1 }
  let _ ← spec.compileVmfbs evalCfg (useSeg := true)
  let pfx := spec.buildPrefix
  let sess ← IreeSession.create s!"{pfx}_fwd_eval.vmfb"
  let params ← IO.FS.readBinFile s!"{pfx}_params.bin"
  let bnPad ← F32.const spec.nBnStats.toUSize 0.0
  let evalParams := params.append bnPad
  let evalShapesBA := spec.evalShapesBA
  let xShape := spec.xShape 1

  let pStart := if promptIds.size > T then promptIds.size - T else 0
  let mut context : Array Nat := Array.replicate T 0
  let mut curLen : Nat := 0
  for i in [pStart:promptIds.size] do
    context := context.set! (i - pStart) promptIds[i]!; curLen := curLen + 1
  if curLen == 0 then curLen := 1  -- id 0 primes generation

  let mut generated : Array Nat := #[]
  let outElems : USize := (T * vocabSize).toUSize
  for stepN in [:nToks] do
    -- Build [1, T] f32 ids input.
    let mut idsBA : ByteArray := ByteArray.emptyWithCapacity (T * 4)
    for t in [:T] do
      let id := context[t]!
      idsBA := idsBA.push (id % 256).toUInt8 |>.push ((id / 256) % 256).toUInt8
                  |>.push ((id / 65536) % 256).toUInt8 |>.push ((id / 16777216) % 256).toUInt8
    let xba ← F32.idsToFloats idsBA
    let logitsFlat ← IreeSession.forwardF32 sess spec.evalFnName
                       evalParams evalShapesBA xba xShape 1 outElems
    let baseIdx := (curLen - 1) * vocabSize
    let logits := readFloats logitsFlat baseIdx vocabSize
    let seedU : USize := (userSeed * 1099511628211 + stepN * 2654435761 + 1).toUSize
    let next := sampleToken logits temperature topK topP seedU
    generated := generated.push next
    if curLen < T then context := context.set! curLen next; curLen := curLen + 1
    else
      for t in [:T - 1] do context := context.set! t context[t + 1]!
      context := context.set! (T - 1) next
  return generated

/-- Read space/line-separated int ids from a file (encoded prompt). -/
def loadPromptIds (path : String) : IO (Array Nat) := do
  if !(← System.FilePath.pathExists path) then return #[]
  let raw ← IO.FS.readFile path
  return (raw.split (fun c => c == ' ' || c == '\n')).toArray.filterMap (·.trim.toNat?)

def main (args : List String) : IO Unit := do
  match args with
  | "train" :: rest =>
    let steps := (rest[0]?.bind String.toNat?).getD 12000
    let batch := (rest[1]?.bind String.toNat?).getD 32
    let lrPct := (rest[2]?.bind String.toNat?).getD 30
    let lr : Float := lrPct.toFloat / 10000.0
    let (params, loss) ← runTrain steps batch lr
    let pfx := storiesSpec.buildPrefix
    IO.FS.writeBinFile s!"{pfx}_params.bin" params
    IO.eprintln s!"saved params to {pfx}_params.bin (final loss {loss})"
  | "sample" :: rest =>
    let nToks   := (rest[0]?.bind String.toNat?).getD 200
    let tempPct := (rest[1]?.bind String.toNat?).getD 80
    let topK    := (rest[2]?.bind String.toNat?).getD 40
    let toppPct := (rest[3]?.bind String.toNat?).getD 95
    let seed    := (rest[4]?.bind String.toNat?).getD 1
    let pfx := storiesSpec.buildPrefix
    if !(← System.FilePath.pathExists s!"{pfx}_params.bin") then
      IO.eprintln s!"missing {pfx}_params.bin; run `train` first"; IO.Process.exit 1
    let promptIds ← loadPromptIds "data/tinystories/prompt_ids.txt"
    let gen ← runSample nToks (tempPct.toFloat / 100.0) topK (toppPct.toFloat / 100.0) seed promptIds
    -- Emit ids for Python-side BPE decode.
    IO.println (String.intercalate " " (gen.toList.map toString))
  | _ =>
    IO.eprintln "usage: tinystories train  [steps=12000] [batch=32] [lr_x10000=30]"
    IO.eprintln "       tinystories sample [n_toks=200] [temp_x100=80] [topk=40] [topp_x100=95] [seed=1]"
    IO.Process.exit 1
