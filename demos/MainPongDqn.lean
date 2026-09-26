import LeanMlir.Train
import LeanMlir.Pong

/-! DQN on the Lean Pong — rung 3 of `planning/pong_dqn_demo.md`.

    Mnih et al. 2015's loop on the game in `LeanMlir/Pong.lean`: frame skip 4,
    replay, a target network, ε-greedy, one gradient step per four agent steps.
    The loss is blackjack's: the rank-2 DDPM MSE block with the Bellman target
    built on the host, the net's own prediction kept in the two untaken slots.

    The replay stores one observation per agent step and rebuilds the input by
    index at sampling time (`k` observations stacked, clamped at the episode's
    first step), so `state` and `pixels` share every line of the loop but the
    observation and the net:

    - `state`: the six-number state, a 6→64→64→3 MLP; the ceiling row.
    - `pixels`: 84 × 84 u8 frames, `k` stacked (4, or 1 for the ablation),
      Chapter 3's CNN with pooling standing in for stride (Phase 2).

    Evaluation every `evalEvery` agent steps plays `evalGames` games in lockstep
    at ε = 0.05 through one batched forward; the score is mean points per game
    (own minus opponent's). XLA only.

    `lake exe pong-dqn [mode=state] [steps=500000] [seed=1] [lr=1e-4] [double]
     [k=4] [opp=1.5] [tag=<name>]`; writes `<prefix>_curve.csv` and
     `<prefix>_params.bin` under `.lake/build/`.

    Run it with `PJRT_FFI_RESIDENT=1`: [θ|m|v] then stays on the card across train
    steps and the forwards hold their parameters, 24 → 14.6 ms per pixel update on
    one 4060 Ti. Bit-identical to the copying path on the deterministic shim
    (`scripts/det_shim.sh`); on the shipping shim the two differ in the last bits
    from update 2 on, as autotuned kernels do across buffer origins. -/

open PongEnv

namespace PongDqn

def stateNet : NetSpec where
  name := "pong dqn state"
  imageH := 1
  imageW := 1
  layers := [
    .dense 6 64 .relu,
    .dense 64 64 .relu,
    .dense 64 3 .identity     -- stay, up, down
  ]

/-- Chapter 3's kit at 84 × 84 on a stack of `k` frames. The Nature net's strides
    become pools (`.conv2d` has no stride, and BatchNorm has no place in a
    Q-function whose batches mix a moving policy's states), sized so every pool
    divides evenly and the flatten is the paper's 7 × 7 × 64; its 8×8 and 4×4
    kernels become 7×7 and 5×5, since `.same` is spelled for odd kernels. 1.7M
    params, the dense layer nearly all of them. -/
def pixelNet (k : Nat) : NetSpec where
  name := s!"pong dqn pixels k{k}"
  imageH := 84
  imageW := 84
  layers := [
    .conv2d k 32 7 .same .relu, .maxPool 3 3,       -- 84 → 28
    .conv2d 32 64 5 .same .relu, .maxPool 2 2,      -- 28 → 14
    .conv2d 64 64 3 .same .relu, .maxPool 2 2,      -- 14 → 7
    .flatten,
    .dense (64 * 7 * 7) 512 .relu,
    .dense 512 3 .identity
  ]

/-- The six-number state as 24 bytes of f32, the replay's per-step record. -/
def stateObs (p : Pong) : ByteArray :=
  (Pong.stateVec p).foldl pushF32LE ByteArray.empty

/-- A ring of per-step records. Slot `i` holds the observation BEFORE the action
    taken at step `i`, the action, the reward, whether the step ended the game,
    and whether the observation is the game's first (the stack clamps there).
    The next observation of a non-terminal step is slot `i + 1`'s. -/
structure Replay where
  cap : Nat
  obs : Array ByteArray
  act : Array Nat
  rew : Array Float
  done : Array Bool
  first : Array Bool
  wr : Nat := 0
  size : Nat := 0

def Replay.empty (cap : Nat) : Replay :=
  { cap, obs := Array.mkEmpty cap, act := Array.mkEmpty cap, rew := Array.mkEmpty cap,
    done := Array.mkEmpty cap, first := Array.mkEmpty cap }

def Replay.push (rb : Replay) (o : ByteArray) (a : Nat) (r : Float) (d f : Bool) : Replay :=
  if rb.obs.size < rb.cap then
    { rb with obs := rb.obs.push o, act := rb.act.push a, rew := rb.rew.push r,
              done := rb.done.push d, first := rb.first.push f,
              wr := (rb.wr + 1) % rb.cap, size := rb.size + 1 }
  else
    { rb with obs := rb.obs.set! rb.wr o, act := rb.act.set! rb.wr a, rew := rb.rew.set! rb.wr r,
              done := rb.done.set! rb.wr d, first := rb.first.set! rb.wr f,
              wr := (rb.wr + 1) % rb.cap }

/-- The `k` slot indices ending at `i`, oldest first, clamped at the game's
    first observation (repeated), the way the paper's stack pads a game's start. -/
def Replay.stackIdx (rb : Replay) (i k : Nat) : Array Nat := Id.run do
  let mut out : Array Nat := Array.replicate k i
  let mut j := i
  for t in [1:k] do
    if !rb.first[j]! then j := (j + rb.cap - 1) % rb.cap
    out := out.set! (k - 1 - t) j
  return out

/-- A sampleable slot: its successor has been written (it is not the newest
    step), and once the ring has wrapped, its stack stays on the oldest-written
    side of the write head (at least `k - 1` slots of this data before it). -/
def Replay.valid (rb : Replay) (i k : Nat) : Bool :=
  if i == (rb.wr + rb.cap - 1) % rb.cap then false
  else if rb.size < rb.cap then true
  else (i + rb.cap - rb.wr) % rb.cap + 1 >= k

/-- `1.5`, `0.001`, `1e-4`, `2.5e-4`: enough of a decimal parser for the knobs. -/
def parseFloat (s : String) : Option Float := do
  let (mant, ex) ← match s.splitOn "e" with
    | [m] => pure (m, 0)
    | [m, e] =>
      if e.startsWith "-" then ((e.drop 1).toString.toNat?).map fun n => (m, -(n : Int))
      else e.toNat?.map fun n => (m, (n : Int))
    | _ => none
  let (ip, fp) := match mant.splitOn "." with
    | [i] => (i, "")
    | [i, f] => (i, f)
    | _ => ("x", "")
  let iv ← if ip.isEmpty then some 0 else ip.toNat?
  let fv ← if fp.isEmpty then some 0 else fp.toNat?
  let x := iv.toFloat + fv.toFloat / Float.pow 10.0 fp.length.toFloat
  return x * Float.pow 10.0 (Float.ofInt ex)

end PongDqn

open PongDqn in
def main (args : List String) : IO Unit := do
  let kv (key : String) : Option String :=
    (args.find? (·.startsWith (key ++ "="))).map (·.drop (key.length + 1) |>.toString)
  let natArg (key : String) (d : Nat) : Nat := ((kv key) >>= String.toNat?).getD d
  let floatArg (key : String) (d : Float) : Float := ((kv key) >>= parseFloat).getD d
  let mode := (kv "mode").getD "state"
  let steps := natArg "steps" 500000
  let seed := natArg "seed" 1
  let lr := floatArg "lr" 1.0e-4
  let double := args.any (· == "double")
  let opp : Opp := { speed := floatArg "opp" 1.5 }
  let tag := kv "tag"
  let pixels ← match mode with
    | "state" => pure false
    | "pixels" => pure true
    | _ => throw <| IO.userError s!"mode={mode}: expected state or pixels"
  let k : Nat := if pixels then natArg "k" 4 else 1
  let spec := if pixels then pixelNet k else stateNet
  -- the replay's per-step record: 24 bytes of f32 state, or one 7056-byte u8 frame
  let obsOf : Pong → ByteArray := if pixels then Pong.render else stateObs
  -- the network's input from concatenated records
  let enc (x : ByteArray) : IO ByteArray := if pixels then F32.u8Scaled x (1.0 / 255.0) else pure x
  let recBytes : USize := if pixels then 84 * 84 else 24
  let gather (idx : ByteArray) (recs : Array ByteArray) (dst : ByteArray) : IO ByteArray :=
    if pixels then F32.gatherU8Scaled recs idx dst recBytes (1.0 / 255.0)
    else F32.gatherConcat recs idx dst recBytes
  -- last update's batches, handed back to the gather so it writes in place
  let mut xsBuf := ByteArray.empty
  let mut xnBuf := ByteArray.empty
  let B : Nat := 32
  let gamma : Float := 0.99
  let ring : Nat := 100000
  let warm : Nat := 10000
  let epsHi : Float := 1.0
  let epsLo : Float := 0.1
  let epsSteps : Nat := 100000
  let epsEval : Float := 0.05
  let trainEvery : Nat := 4
  let targetEvery : Nat := 1000
  let evalEvery := natArg "evalEvery" 25000
  let evalGames := natArg "evalGames" 20
  let nA : Nat := 3
  IO.eprintln s!"{spec.name}: {spec.totalParams} params, {steps} agent steps, batch {B}, lr {lr}, \
{if double then "double" else "plain"} DQN, opponent speed {opp.speed} delay {opp.delay}, seed {seed}"
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "pong-dqn runs on the XLA backend only"

  -- ── graphs: train step at B, eval forward at B, at 1 (acting) and at evalGames ──
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix ++ (match tag with | some t => "_" ++ t | none => "")
  let trainMlir := MlirCodegen.generateTrainStep spec B
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := 0.0) (useAdam := true)
    (useDdpm := true) (ddpmOutShape := [B, nA, 1, 1])
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec B)
  IO.FS.writeFile s!"{pfx}_fwd_eval_act.mlir" (MlirCodegen.generateEval spec 1)
  IO.FS.writeFile s!"{pfx}_fwd_eval_games.mlir" (MlirCodegen.generateEval spec evalGames)
  let sess ← LowererSession.create (← NetSpec.graphArtifact pfx "train_step")
  -- The forwards hold their parameters on the device (`forwardF32`'s HOLD mode), one
  -- session per parameter set so the online and target nets never evict each other.
  -- The generation token is what keeps a held set honest: the online net's is
  -- `updates + 1`, which moves with every train step; the target's moves with every copy.
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval")
  let targetSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval")
  let actSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval_act")
  let gamesSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval_games")
  IO.eprintln "  sessions loaded"

  let nP := spec.totalParams
  let nRes : USize := (spec.layers.foldl (fun n (l : Layer) => match l with
    | .conv2d .. | .dense .. => n + 2 | _ => n) 0).toUSize
  let nT := 3 * nP
  let allShapes := spec.shapesBA
  let evalShapes := spec.evalShapesBA
  let bnShapes := spec.bnShapesBA
  let xShB := spec.xShape B
  let xSh1 := spec.xShape 1
  let xShG := spec.xShape evalGames
  let p0 ← spec.heInitParams
  -- `pmv` is the train step's own output, params ‖ Adam m ‖ Adam v ‖ loss, fed straight
  -- back as the next step's input: the shim reads each tensor at its offset from the
  -- shapes and ignores the tail, so there is no per-step slice or append. With
  -- PJRT_FFI_RESIDENT=1 the step keeps [θ|m|v] on the device (`trainStepAdamF32DdpmR`)
  -- and `pmv`'s param region is unwritten after the seed; `theta` — what every forward
  -- reads — comes back through `readParamsPrefix`, θ alone, m and v never leaving the card.
  let mut pmv := ((p0.append (← F32.const nP.toUSize 0.0)).append (← F32.const nP.toUSize 0.0)).append
    (← F32.const 1 0.0)
  let mut theta := p0
  let nTrainRes : USize := 3 * nRes
  let mut target := p0
  let mut targetGen := 1

  let argmax3 (q : ByteArray) (i : Nat) : Nat :=
    let a := F32.read q (nA * i).toUSize
    let b := F32.read q (nA * i + 1).toUSize
    let c := F32.read q (nA * i + 2).toUSize
    if a >= b && a >= c then 0 else if b >= c then 1 else 2

  -- ── evaluation: `evalGames` games in lockstep, one batched forward per agent step ──
  let evaluate (params : ByteArray) (gen round : Nat) : IO (Float × Float × Nat) := do
    let mut gms : Array Game := (Array.range evalGames).map fun i => Game.reset (900000 + 1000 * round + i)
    let mut hist : Array (Array ByteArray) := gms.map fun gm => Array.replicate k (obsOf gm.p)
    let mut live : Array Bool := Array.replicate evalGames true
    let mut g := mkStdGen (77 + round)
    let mut agentSteps := 0
    for _ in [0:frameCap / 4 + 1] do
      unless live.any id do break
      let x ← enc (hist.foldl (fun acc h => h.foldl (· ++ ·) acc) ByteArray.empty)
      -- what the net saw, game 0 a few steps in: the check that the tensor is the game
      if pixels && agentSteps == 40 * evalGames then
        let mut strip := ByteArray.mk (Array.replicate (84 * 84 * k) 0)
        for c in [0:k] do
          for yy in [0:84] do
            for xx in [0:84] do
              let f := F32.read x (c * 7056 + yy * 84 + xx).toUSize
              strip := strip.set! (yy * 84 * k + c * 84 + xx) (Float.toUInt8 (f * 255.0 + 0.5))
        writePgm s!"{pfx}_seen_{round}.pgm" (84 * k) 84 strip
      let q ← LowererSession.forwardF32 gamesSess spec.evalFnName params evalShapes x xShG
                evalGames.toUSize nA.toUSize nRes gen.toUSize
      for i in [0:evalGames] do
        if live[i]! then
          let (u, g1) := randNat g 0 999999
          let (ra, g2) := randNat g1 0 (nA - 1)
          g := g2
          let a := if u.toFloat / 1000000.0 < epsEval then ra else argmax3 q i
          let (gm', _, d) := gms[i]!.step opp (Act.ofNat a)
          gms := gms.set! i gm'
          hist := hist.set! i ((hist[i]!.extract 1 k).push (obsOf gm'.p))
          agentSteps := agentSteps + 1
          if d then live := live.set! i false
    let diffs := gms.map fun gm => gm.scoreP.toFloat - gm.scoreO.toFloat
    let mean := diffs.foldl (· + ·) 0.0 / evalGames.toFloat
    let var := diffs.foldl (fun acc d => acc + (d - mean) * (d - mean)) 0.0 / evalGames.toFloat
    return (mean, Float.sqrt var / Float.sqrt evalGames.toFloat, agentSteps)

  let mut rb := Replay.empty ring
  let mut g := mkStdGen seed
  let mut gm := Game.reset seed
  let mut hist : Array ByteArray := Array.replicate k (obsOf gm.p)
  let mut isFirst := true
  let mut updates := 0
  let mut games := 0
  let mut lossAcc := 0.0
  let mut lossN := 0
  let mut curve : Array (Nat × Nat × Float × Float) := #[]
  let mut firstWin : Option Nat := none
  let (e0, se0, _) ← evaluate theta 1 0
  curve := curve.push (0, 0, e0, se0)
  IO.eprintln s!"  step 0  eval {fmt e0 2} ± {fmt se0 2}"
  -- per-phase wall clock in ns (act forward, replay gather, target forwards, train step)
  let mut tAct := 0
  let mut tGather := 0
  let mut tFwd := 0
  let mut tTrain := 0
  let t0 ← IO.monoMsNow
  for step in [1:steps + 1] do
    -- ── act ──
    let eps := if step >= epsSteps then epsLo
               else epsHi + (epsLo - epsHi) * step.toFloat / epsSteps.toFloat
    let (u, g1) := randNat g 0 999999
    let (ra, g2) := randNat g1 0 (nA - 1)
    g := g2
    let ta ← IO.monoNanosNow
    let a ← if step <= warm || u.toFloat / 1000000.0 < eps then pure ra else do
      let x ← enc (hist.foldl (· ++ ·) ByteArray.empty)
      let q ← LowererSession.forwardF32 actSess spec.evalFnName theta evalShapes x xSh1 1 nA.toUSize
                nRes (updates + 1).toUSize
      pure (argmax3 q 0)
    tAct := tAct + ((← IO.monoNanosNow) - ta)
    let o := obsOf gm.p
    let (gm', r, d) := gm.step opp (Act.ofNat a)
    rb := rb.push o a r d isFirst
    if d then
      games := games + 1
      gm := Game.reset (seed * 1000003 + games)
      hist := Array.replicate k (obsOf gm.p)
      isFirst := true
    else
      gm := gm'
      hist := (hist.extract 1 k).push (obsOf gm.p)
      isFirst := false
    -- ── learn ──
    if step > warm && step % trainEvery == 0 then
      let tg ← IO.monoNanosNow
      let mut xsIdx := ByteArray.empty
      let mut xnIdx := ByteArray.empty
      let mut idx : Array Nat := Array.mkEmpty B
      while idx.size < B do
        let (i, g') := randNat g 0 (rb.size - 1)
        g := g'
        if rb.valid i k then
          idx := idx.push i
          xsIdx := (rb.stackIdx i k).foldl pushU32LE xsIdx
          xnIdx := (rb.stackIdx ((i + 1) % ring) k).foldl pushU32LE xnIdx
      let xsF ← gather xsIdx rb.obs xsBuf
      let xnF ← gather xnIdx rb.obs xnBuf
      let tf ← IO.monoNanosNow
      tGather := tGather + (tf - tg)
      let qs ← LowererSession.forwardF32 evalSess spec.evalFnName theta evalShapes xsF xShB B.toUSize nA.toUSize
                 nRes (updates + 1).toUSize
      let qn ← LowererSession.forwardF32 targetSess spec.evalFnName target evalShapes xnF xShB B.toUSize nA.toUSize
                 nRes targetGen.toUSize
      let qo ← if double
        then LowererSession.forwardF32 evalSess spec.evalFnName theta evalShapes xnF xShB B.toUSize nA.toUSize
               nRes (updates + 1).toUSize
        else pure qn
      let mut y := ByteArray.empty
      for bi in [0:B] do
        let i := idx[bi]!
        let boot := if double then F32.read qn (nA * bi + argmax3 qo bi).toUSize
          else max (F32.read qn (nA * bi).toUSize)
                 (max (F32.read qn (nA * bi + 1).toUSize) (F32.read qn (nA * bi + 2).toUSize))
        let tgt := rb.rew[i]! + (if rb.done[i]! then 0.0 else gamma * boot)
        for j in [0:nA] do
          y := pushF32LE y (if j == rb.act[i]! then tgt else F32.read qs (nA * bi + j).toUSize)
      let tt ← IO.monoNanosNow
      tFwd := tFwd + (tt - tf)
      updates := updates + 1
      let out ← LowererSession.trainStepAdamF32DdpmR sess spec.trainFnName
                  pmv allShapes xsF xShB y lr updates.toFloat bnShapes B.toUSize nA.toUSize 1 1
                  nTrainRes
      lossAcc := lossAcc + F32.extractLoss out nT
      lossN := lossN + 1
      pmv := out
      theta ← LowererSession.readParamsPrefix sess pmv (4 * nP).toUSize
      tTrain := tTrain + ((← IO.monoNanosNow) - tt)
      xsBuf := xsF   -- the batches' last use was above; the next gather writes over them
      xnBuf := xnF
      if updates % targetEvery == 0 then
        target := theta
        targetGen := targetGen + 1
    if step % evalEvery == 0 then
      let (e, se, _) ← evaluate theta (updates + 1) (step / evalEvery)
      curve := curve.push (step, updates, e, se)
      if firstWin.isNone && e > 0.0 then firstWin := some step
      let t1 ← IO.monoMsNow
      IO.eprintln s!"  step {step}  updates {updates}  games {games}  eps {fmt eps 3}  \
loss {fmt (lossAcc / (max 1 lossN).toFloat) 5}  eval {fmt e 2} ± {fmt se 2}  ({t1 - t0} ms; act {tAct / 1000000} gather {tGather / 1000000} \
fwd {tFwd / 1000000} train {tTrain / 1000000} ms)"
      lossAcc := 0.0
      lossN := 0
  let t1 ← IO.monoMsNow
  IO.eprintln s!"trained: {steps} agent steps, {updates} updates, {games} games, {t1 - t0} ms"
  let (ef, sef, _) ← evaluate theta (updates + 1) 9999
  IO.println s!"final ({evalGames} games, eps {epsEval}): {fmt ef 2} ± {fmt sef 2} points per game; \
first positive eval at {match firstWin with | some s => toString s | none => "never"} agent steps"
  IO.FS.writeBinFile s!"{pfx}_params.bin" theta
  let curveCsv := "agent_steps,updates,eval_mean,eval_se\n" ++
    String.join (curve.toList.map fun (s, u, e, se) => s!"{s},{u},{fmt e 3},{fmt se 3}\n")
  IO.FS.writeFile s!"{pfx}_curve.csv" curveCsv
  IO.eprintln s!"wrote {pfx}_curve.csv, {pfx}_params.bin"
