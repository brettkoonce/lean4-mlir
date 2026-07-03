import LeanMlir

/-! Char-level tinyGPT trained on tinyshakespeare.

Two model rungs (planning/tinygpt_demo_v2.md Part I):

  nano (212K params): tokenPositionEmbed (V=65, T=64,  D=64)
                      → transformerEncoder (D=64,  h=2, mlp=256, blocks=4, causal)
                      → lmHead
  tiny (~1.2M params): tokenPositionEmbed (V=65, T=128, D=128)
                      → transformerEncoder (D=128, h=4, mlp=512, blocks=6, causal)
                      → lmHead

Training rides the existing `useSeg` path (per-pixel CE) by treating
the LM head's [B, V, T, 1] output as NCHW segmentation logits with
T positions × 1 width. Labels are `[B, T, 1]` int32 from the
random-chunk sampler.

v2 additions over the original demo:
  * validation bits/char every `evalEvery` steps (fixed val chunks →
    train-step forward, update discarded) — the val split was
    previously never read;
  * cosine LR decay + linear warmup, computed host-side per step
    (the TrainConfig fields existed but were never threaded);
  * sampler right-pads short prompts and reads logits at the last
    *real* position instead of left-padding with token id 0 (a real
    character) and always reading position T-1;
  * top-k / top-p sampling + an explicit RNG seed;
  * `suite` subcommand: fixed prompts × seeds → a diffable sample
    file for the blueprint.

Subcommands:
  lake exe tinygpt-shakespeare train  [nano|tiny] [steps=2000] [batch=32] [lr_x10000=30]
  lake exe tinygpt-shakespeare sample [nano|tiny] [n_chars=400] [temp_x100=80] [topk=0] [topp_x100=100] [seed=1] [prompt]
  lake exe tinygpt-shakespeare suite  [nano|tiny]
-/

def vocabSize : Nat := 65

structure GptCfg where
  key       : String
  specName  : String
  seqLen    : Nat
  dModel    : Nat
  numHeads  : Nat
  mlpDim    : Nat
  numLayers : Nat

/-- The original 212K-param demo config (spec name unchanged so old
    checkpoints keep their build prefix). -/
def nanoCfg : GptCfg :=
  { key := "nano", specName := "tinygpt-shakespeare"
    seqLen := 64, dModel := 64, numHeads := 2, mlpDim := 256, numLayers := 4 }

/-- The v2 scale-up rung: D=128, 4 heads, 6 blocks, T=128. -/
def tinyCfg : GptCfg :=
  { key := "tiny", specName := "tinygpt-shakespeare-tiny"
    seqLen := 128, dModel := 128, numHeads := 4, mlpDim := 512, numLayers := 6 }

def pickCfg : String → Option GptCfg
  | "nano" => some nanoCfg
  | "tiny" => some tinyCfg
  | _ => none

def mkSpec (g : GptCfg) : NetSpec :=
  { name := g.specName
    imageH := g.seqLen     -- so y_seg shape becomes [B, T, 1]
    imageW := 1
    layers := [
      .tokenPositionEmbed vocabSize g.seqLen g.dModel,
      .transformerEncoder g.dModel g.numHeads g.mlpDim g.numLayers (causalMask := true),
      .lmHead g.dModel vocabSize g.seqLen
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

/-- Cosine decay with linear warmup, host-side (the train-step vmfb
    takes lr as a runtime scalar, so no recompile per step). Peak
    `lrMax`, floor 10% of peak, 100 warmup steps. -/
def lrAt (lrMax : Float) (step steps : Nat) : Float :=
  let warmup : Nat := 100
  if steps <= warmup then lrMax
  else if step < warmup then
    lrMax * (step + 1).toFloat / warmup.toFloat
  else
    let prog := (step - warmup).toFloat / (steps - warmup).toFloat
    let lrMin := lrMax * 0.1
    lrMin + 0.5 * (1.0 + Float.cos (3.141592653589793 * prog)) * (lrMax - lrMin)

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

/-- Sample a token id from `logits` with temperature, then top-k
    (0 = off), then top-p (1.0 = off). Greedy when `temperature ≤ 0`. -/
def sampleToken (logits : Array Float) (temperature : Float)
    (topK : Nat) (topP : Float) (seed : USize) : Nat := Id.run do
  if temperature <= 0.0 then
    let mut best := 0
    let mut bestV := logits[0]!
    for i in [1:logits.size] do
      if logits[i]! > bestV then bestV := logits[i]!; best := i
    return best
  -- Sort ids by logit, descending.
  let ids := (Array.range logits.size).qsort (fun a b => logits[a]! > logits[b]!)
  let keep := if topK == 0 || topK > ids.size then ids.size else topK
  -- Softmax over the kept set at the given temperature.
  let mx := logits[ids[0]!]!
  let invT := 1.0 / temperature
  let mut probs : Array Float := Array.mkEmpty keep
  let mut sum : Float := 0.0
  for i in [:keep] do
    let p := ((logits[ids[i]!]! - mx) * invT).exp
    probs := probs.push p
    sum := sum + p
  -- Top-p: cut the sorted tail once cumulative mass ≥ topP.
  let mut cut := keep
  if topP < 1.0 then
    let mut acc : Float := 0.0
    for i in [:keep] do
      if acc >= topP * sum then
        cut := i
        break
      acc := acc + probs[i]!
  let mut cutSum : Float := 0.0
  for i in [:cut] do cutSum := cutSum + probs[i]!
  -- xorshift → uniform in [0, 1)
  let mut s := seed; if s == 0 then s := 1
  s := s ^^^ (s <<< 13); s := s ^^^ (s >>> 7); s := s ^^^ (s <<< 17)
  let r : Float := (s.toNat % 1000003).toFloat / 1000003.0
  let mut acc : Float := 0.0
  for i in [:cut] do
    acc := acc + probs[i]! / cutSum
    if r <= acc then return ids[i]!
  return ids[cut - 1]!

/-- Reshape [B*T] int32 ids → [B, T, 1] int32 (just a byte-identical
    rebind — labels' MLIR shape declaration is `tensor<BxTx1xi32>`). -/
def asSegLabels (ids : ByteArray) : ByteArray := ids

/-- Mean per-token CE (nats) over `nBatches` fixed-seed chunks of the
    val stream, via the train-step vmfb with the updated state
    discarded (the loss output is computed on the *incoming* params,
    so this is a pure eval — no separate eval codegen needed). -/
def evalValLoss (sess : IreeSession) (spec : NetSpec)
    (packed allShapes xShape bnShapes : ByteArray)
    (valTokens : ByteArray) (nValTokens : USize)
    (batch T : Nat) (nBatches : Nat) : IO Float := do
  let inputBytes := batch * T * 4
  let mut tot : Float := 0.0
  let nP := spec.totalParams
  let nT3 := 3 * nP
  for i in [:nBatches] do
    -- Fixed seeds → the same val chunks every eval → comparable curve.
    let seedU : USize := (0xdead0000 + i * 9973).toUSize
    let chunks ← F32.sampleChunks valTokens nValTokens batch.toUSize T.toUSize seedU
    let inputIds  := chunks.extract 0 inputBytes
    let targetIds := chunks.extract inputBytes (2 * inputBytes)
    let xba ← F32.tokenOneHot inputIds batch.toUSize T.toUSize vocabSize.toUSize
    let outBA ← IreeSession.trainStepAdamF32Seg sess spec.trainFnName
                  packed allShapes xba xShape (asSegLabels targetIds) 0.0 1.0
                  bnShapes batch.toUSize T.toUSize 1
    tot := tot + F32.read outBA nT3.toUSize
  return tot / nBatches.toFloat

/-- Run training. Returns final params buffer and a loss history. -/
def runTinyGptTrain (g : GptCfg) (steps : Nat) (batch : Nat) (lrMax : Float)
    : IO (ByteArray × Array Float) := do
  let spec := mkSpec g
  let T := g.seqLen
  let cfg : TrainConfig := { trainConfig with batchSize := batch, learningRate := lrMax }
  IO.eprintln s!"compiling train step (model={g.key}, B={batch}, T={T}, V={vocabSize}, params={spec.totalParams}) ..."
  let _ ← spec.compileVmfbs cfg (useSeg := true)
  let pfx := spec.buildPrefix
  let trainVmfb := s!"{pfx}_train_step.vmfb"
  let trainSess ← IreeSession.create trainVmfb

  IO.eprintln "loading data/shakespeare/{train,val}.bin ..."
  let (tokens, nTokens) ← F32.loadTokenStream "data/shakespeare/train.bin"
  let (valTokens, nValTokens) ← F32.loadTokenStream "data/shakespeare/val.bin"
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
  let nValBatches : Nat := 12

  IO.eprintln s!"training: {steps} steps × batch {batch} × T {T}, lr peak {lrMax} (cosine + 100-step warmup), Adam"
  let mut history : Array Float := #[]
  for step in [:steps] do
    let lr := lrAt lrMax step steps
    let seedU : USize := (step * 65537 + 17).toUSize
    let chunks ← F32.sampleChunks tokens nTokens batch.toUSize T.toUSize seedU
    let inputBytes := batch * T * 4
    let inputIds  := chunks.extract 0 inputBytes
    let targetIds := chunks.extract inputBytes (2 * inputBytes)
    -- One-hot encode → [B, V*T] f32 (matches inputFlatDim = V*T).
    let xba ← F32.tokenOneHot inputIds batch.toUSize T.toUSize vocabSize.toUSize
    let t : Float := (step + 1).toFloat
    let outBA ← IreeSession.trainStepAdamF32Seg trainSess spec.trainFnName
                  packed allShapes xba xShape (asSegLabels targetIds) lr t
                  bnShapes batch.toUSize T.toUSize 1
    packed := outBA.extract 0 (nT3 * 4)
    let loss := F32.read outBA nT3.toUSize
    history := history.push loss
    if step % 100 == 0 || step + 1 == steps then
      IO.eprintln s!"  step {step + 1}/{steps}: loss={loss} lr={lr}"
    if (step + 1) % evalEvery == 0 || step + 1 == steps then
      let valNats ← evalValLoss trainSess spec packed allShapes xShape bnShapes
                      valTokens nValTokens batch T nValBatches
      IO.eprintln s!"  ── val @ step {step + 1}: {valNats} nats/char = {valNats / ln2} bits/char"
  let finalParams := packed.extract 0 (nP * 4)
  return (finalParams, history)

/-- Read N float32s from a ByteArray (element offset `start`). -/
def readFloats (ba : ByteArray) (start n : Nat) : Array Float := Id.run do
  let mut out := Array.mkEmpty n
  for i in [:n] do
    out := out.push (F32.read ba (start + i).toUSize)
  return out

/-- Pack `[T]` int32 ids into a `[1, V*T]` f32 one-hot ByteArray. -/
def packOneHotContext (ids : Array Nat) (T V : Nat) : IO ByteArray := do
  let mut idsBA : ByteArray := ByteArray.emptyWithCapacity (T * 4)
  for t in [:T] do
    let id := if t < ids.size then ids[t]! else 0
    idsBA := idsBA.push (id % 256).toUInt8
                |>.push ((id / 256) % 256).toUInt8
                |>.push ((id / 65536) % 256).toUInt8
                |>.push ((id / 16777216) % 256).toUInt8
  F32.tokenOneHot idsBA 1 T.toUSize V.toUSize

/-- Autoregressive sampling. The context is right-padded: positions
    `≥ curLen` hold token 0 but the causal mask keeps them from
    influencing position `curLen - 1`, whose logits we read. (The old
    version left-padded with token id 0 — a real character — and read
    position T-1, feeding the model a prefix distribution it never
    saw in training.) -/
def runTinyGptSample (g : GptCfg) (paramsPath : String) (nChars : Nat)
    (temperature : Float) (topK : Nat) (topP : Float) (userSeed : Nat)
    (prompt : String) : IO String := do
  let spec := mkSpec g
  let T := g.seqLen
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

  -- Encode prompt → token ids; keep the last T if it overflows.
  let promptIdsAll : Array Nat := prompt.toList.toArray.map fun c => rev[c.toNat]!
  let pStart := if promptIdsAll.size > T then promptIdsAll.size - T else 0
  let mut context : Array Nat := Array.replicate T 0
  let mut curLen : Nat := 0
  for i in [pStart:promptIdsAll.size] do
    context := context.set! (i - pStart) promptIdsAll[i]!
    curLen := curLen + 1
  if curLen == 0 then
    -- Empty prompt: seed with newline so position 0 is a real token.
    context := context.set! 0 (rev[10]!)
    curLen := 1
  IO.eprintln s!"  prompt encoded: {promptIdsAll.size} ids (context fill {curLen}/{T})"

  let mut generated : Array Nat := #[]
  let outElems : USize := (T * vocabSize).toUSize
  for stepN in [:nChars] do
    let xba ← packOneHotContext context T vocabSize
    let logitsFlat ← IreeSession.forwardF32 sess spec.evalFnName
                       evalParams evalShapesBA xba xShape 1 outElems
    -- logitsFlat layout: [1, T*V] row-major in (t, v); read the last
    -- REAL position.
    let baseIdx := (curLen - 1) * vocabSize
    let logits := readFloats logitsFlat baseIdx vocabSize
    let seedU : USize := (userSeed * 1099511628211 + stepN * 2654435761 + 1).toUSize
    let next := sampleToken logits temperature topK topP seedU
    generated := generated.push next
    if curLen < T then
      context := context.set! curLen next
      curLen := curLen + 1
    else
      for t in [:T - 1] do
        context := context.set! t context[t + 1]!
      context := context.set! (T - 1) next

  -- Decode (prompt + generated) to text.
  let mut s : String := ""
  for id in promptIdsAll do
    if id < vocab.size then s := s ++ (Char.ofNat vocab[id]!.toNat).toString
  for id in generated do
    if id < vocab.size then s := s ++ (Char.ofNat vocab[id]!.toNat).toString
  return s

/-- Fixed prompts × seeds — regenerate after every training change and
    diff. The LM version of the DDPM fixed-seed sample grid. -/
def suitePrompts : Array String := #[
  "ROMEO:\n",
  "KING HENRY VI:\n",
  "First Citizen:\n",
  "QUEEN MARGARET:\nMy lord,",
  "To be, or not to be"
]

def suiteSeeds : Array Nat := #[1, 2]

def runSuite (g : GptCfg) (paramsPath : String) : IO Unit := do
  let outDir := "blueprint/src/figures/tinygpt"
  IO.FS.createDirAll outDir
  let outPath := s!"{outDir}/prompt_suite_{g.key}.txt"
  let mut out : String :=
    s!"tinygpt prompt suite — model={g.key} (T={g.seqLen}, D={g.dModel}, " ++
    s!"{g.numLayers} blocks)\ntemperature=0.8 top-p=0.95, 300 chars per sample\n"
  for p in suitePrompts do
    for seed in suiteSeeds do
      IO.eprintln s!"suite: prompt={p.quote} seed={seed}"
      let text ← runTinyGptSample g paramsPath 300 0.8 0 0.95 seed p
      out := out ++ "\n════════════════════════════════════════\n"
      out := out ++ s!"prompt: {p.quote}   seed: {seed}\n────────────────────────────────────────\n"
      out := out ++ text ++ "\n"
  IO.FS.writeFile outPath out
  IO.eprintln s!"wrote {outPath}"

/-- Peel an optional leading model key off the arg list (defaults to
    nano so the pre-v2 CLI shapes still work). -/
def peelModel (rest : List String) : GptCfg × List String :=
  match rest with
  | key :: tl =>
    match pickCfg key with
    | some g => (g, tl)
    | none => (nanoCfg, rest)
  | [] => (nanoCfg, [])

def main (args : List String) : IO Unit := do
  match args with
  | "train" :: rest =>
    let (g, rest) := peelModel rest
    let steps := (rest.head?.bind String.toNat?).getD 2000
    let batch := (rest.tail.head?.bind String.toNat?).getD 32
    let lrPct := (rest.tail.tail.head?.bind String.toNat?).getD 30  -- lr_x10000, default 0.003
    let lr : Float := lrPct.toFloat / 10000.0
    let (params, hist) ← runTinyGptTrain g steps batch lr
    let pfx := (mkSpec g).buildPrefix
    let outPath := s!"{pfx}_params.bin"
    IO.FS.writeBinFile outPath params
    IO.eprintln s!"saved params to {outPath}  (final loss: {hist.back!})"
  | "sample" :: rest =>
    let (g, rest) := peelModel rest
    let nChars  := (rest[0]?.bind String.toNat?).getD 400
    let tempPct := (rest[1]?.bind String.toNat?).getD 80
    let topK    := (rest[2]?.bind String.toNat?).getD 0
    let toppPct := (rest[3]?.bind String.toNat?).getD 100
    let seed    := (rest[4]?.bind String.toNat?).getD 1
    let prompt  := (rest[5]?).getD "ROMEO:\n"
    let temperature : Float := tempPct.toFloat / 100.0
    let topP : Float := toppPct.toFloat / 100.0
    let pfx := (mkSpec g).buildPrefix
    let paramsPath := s!"{pfx}_params.bin"
    if !(← System.FilePath.pathExists paramsPath) then
      IO.eprintln s!"missing {paramsPath}; run `train {g.key}` first"; IO.Process.exit 1
    let text ← runTinyGptSample g paramsPath nChars temperature topK topP seed prompt
    IO.println "════════════════ SAMPLE ════════════════"
    IO.println text
    IO.println "════════════════════════════════════════"
  | "suite" :: rest =>
    let (g, _) := peelModel rest
    let pfx := (mkSpec g).buildPrefix
    let paramsPath := s!"{pfx}_params.bin"
    if !(← System.FilePath.pathExists paramsPath) then
      IO.eprintln s!"missing {paramsPath}; run `train {g.key}` first"; IO.Process.exit 1
    runSuite g paramsPath
  | _ =>
    IO.eprintln "usage: tinygpt-shakespeare train  [nano|tiny] [steps=2000] [batch=32] [lr_x10000=30]"
    IO.eprintln "       tinygpt-shakespeare sample [nano|tiny] [n_chars=400] [temp_x100=80] [topk=0] [topp_x100=100] [seed=1] [prompt]"
    IO.eprintln "       tinygpt-shakespeare suite  [nano|tiny]"
    IO.Process.exit 1
