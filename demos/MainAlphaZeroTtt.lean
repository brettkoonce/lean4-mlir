import LeanMlir.Train
import LeanMlir.TicTacToe

/-! AlphaZero on the Lean tic-tac-toe, scored against the solved game.

    Silver et al. 2017's loop on `LeanMlir/TicTacToe.lean`: self-play with PUCT tree
    search over the network's priors and value, the visit distribution and the game's
    outcome as the targets, then the next iteration plays with the new network. The
    net is AlphaGo's plain conv + ReLU stack (`Bestiary/AlphaGo.lean`) at tic-tac-toe
    width, the two heads merged into one dense output of `n² + 1` slots — policy
    logits over the cells, then the value pre-tanh. Not the AlphaZero conv-BN tower:
    BatchNorm's batch statistics over nine binary cells are a liability — the first
    run, on the bestiary's `tinyAlphaZero` body, diverged at iteration 11 with the
    loss going 1.26 → 10.8 in three iterations (`runs/2026-09-29-alphazero-ttt/
    n3_convbn_collapse.txt`) — and without BN the eval forward the loss reads and the
    train step's forward are the same function, which the loss trick needs.

    The loss is the paper's `(z − v)² − πᵀ log p`, delivered through the rank-2 DDPM
    MSE block the way the NQS demo delivers its energy gradient: the host asks for the
    output cotangent it wants by writing `y = out − nOut·g/2`, which makes the block's
    gradient `g/B`, the gradient of the batch-mean loss (`lean_ttt_targets` in
    `ffi/f32_helpers.c`). Zero new codegen; the block's own printed loss is meaningless
    and the log prints the AlphaZero loss instead. Every forward is the eval graph,
    held on the device under a generation token.

    Everything batched runs in lockstep: `bf` self-play games advance one simulation at
    a time — one PUCT descent per game in C (`lean_mcts_*`: the trees are flat per-game
    arrays in one Lean-owned arena), one forward over their pending leaves, expand and
    back up in C; the matches against the solved game's perfect player run `bf` games
    with the net on one side; the sweep over the reachable decision positions runs `bf`
    positions per forward.

    `lake exe alphazero-ttt [n=3] [k=n] [iters=20] [sims=25] [bf=256] [batch=64]
     [epochs=10] [lr=1e-3] [window=20] [cpuct=1.5] [alpha=1.0] [noise=0.25]
     [tempPlies=n²] [sweep=0 (all)] [probe=0,81] [seed=1] [tag=<name>]`;
    writes `<prefix>_curve.csv`, `<prefix>_sweep.csv`, `<prefix>_policy.csv` and
    `<prefix>_params.bin` under `.lake/build/`. XLA only. -/

open FloatFmt TTT

namespace AlphaZeroTtt

/-- AlphaGo's plain conv + ReLU stack at board size `n`: two input planes (the
    mover's stones, the opponent's), three 3×3 convs, a 1×1 head conv, and the policy
    and value heads merged into one dense output — slots `0 .. n²−1` the move logits,
    slot `n²` the value pre-tanh. No BatchNorm (see the module docstring). -/
def net (n : Nat) : NetSpec where
  name := s!"alphazero ttt {n}x{n}"
  imageH := n
  imageW := n
  layers := [
    .conv2d 2 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .conv2d 64 4 1 .same .relu,
    .flatten,
    .dense (4 * n * n) 64 .relu,
    .dense 64 (n * n + 1) .identity
  ]

/-- Replay gather with one random dihedral view per sample, applied alike to the
    planes and the policy target: `(x [count, 2, n, n], π [count, n²])`. -/
@[extern "lean_ttt_gather_aug"]
opaque gatherAug (planes pi idx : @& ByteArray) (count n : USize) (seed : UInt64) :
    IO (ByteArray × ByteArray)

/-- The MSE block's target for `(z − tanh v)² − πᵀ log softmax(p)` on a logits block
    `[count, nOut]`, and the batch's mean AlphaZero loss. -/
@[extern "lean_ttt_targets"]
opaque targets (out pi z : @& ByteArray) (count nOut : USize) (scale : Float) :
    IO (ByteArray × ByteArray)

def readU32 (ba : ByteArray) (i : Nat) : Nat :=
  ba[4 * i]!.toNat ||| (ba[4 * i + 1]!.toNat <<< 8) ||| (ba[4 * i + 2]!.toNat <<< 16) |||
    (ba[4 * i + 3]!.toNat <<< 24)

/-- `1.5`, `0.001`, `1e-4`: enough of a decimal parser for the knobs. -/
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

-- ── Random draws ──

def uniform (g : StdGen) : Float × StdGen :=
  let (u, g) := randNat g 0 999999999
  ((u.toFloat + 0.5) / 1.0e9, g)

def sampleFrom (p : Array Float) (g : StdGen) : Nat × StdGen := Id.run do
  let (u, g) := uniform g
  let mut acc := 0.0
  let mut last := 0
  for i in [0:p.size] do
    if p[i]! > 0.0 then
      last := i
      acc := acc + p[i]!
      if u < acc then return (i, g)
  return (last, g)

/-- A fresh 64-bit seed for one C-side draw. -/
def seed64 (g : StdGen) : UInt64 × StdGen :=
  let (a, g) := randNat g 0 0xFFFFFFFF
  let (b, g) := randNat g 0 0xFFFFFFFF
  ((a.toUInt64 <<< 32) ||| b.toUInt64, g)

-- ── The search: trees in C (`lean_mcts_*` in ffi/f32_helpers.c), the lockstep in Lean ──

/-- The arena for `G` games at board `n`, `k` in a row, `cap` nodes each: every game's
    nodes, its position-index → node hash and its pending path, as flat arrays. Lean
    never reads it; the C calls mutate it in place. -/
@[extern "lean_mcts_alloc"]
opaque mctsAlloc (G n k cap : USize) : IO ByteArray

/-- Forget every tree: before a self-play iteration or a match. -/
@[extern "lean_mcts_reset"]
opaque mctsReset (arena : @& ByteArray) : IO Unit

/-- One PUCT descent per live game from its root (u64 LE indices): `G` records of
    `{u32 kind, f32 value, u64 leaf index}` — kind 1 needs the net, kind 2 is a terminal
    leaf carrying its value from its mover's view, 0 was not live. -/
@[extern "lean_mcts_select"]
opaque mctsSelect (arena roots live : @& ByteArray) (cpuct : Float) : IO ByteArray

/-- Expand the pending leaves from the logits block (row `r` is the `r`-th game whose
    kind was 1, in game order), Dirichlet noise on a freshly expanded root when `noise`,
    then back every value up its path. -/
@[extern "lean_mcts_expand_backup"]
opaque mctsExpandBackup (arena logits : @& ByteArray) (nOut : USize) (noise : UInt8)
    (alpha eps : Float) (seed : UInt64) : IO Unit

/-- Mix Dirichlet noise into the priors of every live game's root already in its tree. -/
@[extern "lean_mcts_root_noise"]
opaque mctsRootNoise (arena roots live : @& ByteArray) (alpha eps : Float) (seed : UInt64) : IO Unit

/-- The roots' visit distributions, f32 `[G, n²]`, zeros where a root is not in its tree. -/
@[extern "lean_mcts_root_visits"]
opaque mctsRootVisits (arena roots : @& ByteArray) : IO ByteArray

/-- The net's priors at `pos` from logits row `row`: softmax over the legal cells. The
    search does its own in C; this is the figure's readout. -/
def priors (logits : ByteArray) (row nOut : Nat) (pos : Pos) : Array Float := Id.run do
  let nc := pos.n * pos.n
  let legal := pos.legal
  let mut mx := -1.0e30
  for c in legal do mx := max mx (F32.read logits (row * nOut + c).toUSize)
  let mut P := Array.replicate nc 0.0
  let mut se := 0.0
  for c in legal do
    let e := Float.exp (F32.read logits (row * nOut + c).toUSize - mx)
    P := P.set! c e
    se := se + e
  return P.map (· / se)

/-- What a search needs: the board, the widths, the arena, and the forward itself, which
    takes up to `bf` position indices as u32 and returns logits `[bf, nOut]`. -/
structure Ctx where
  n : Nat
  nOut : Nat
  bf : Nat
  cpuct : Float
  alpha : Float
  eps : Float
  arena : ByteArray
  forward : ByteArray → Nat → IO ByteArray

def Ctx.nc (ctx : Ctx) : Nat := ctx.n * ctx.n

def pushU64LE (acc : ByteArray) (v : Nat) : ByteArray :=
  pushU32LE (pushU32LE acc (v % 4294967296)) (v / 4294967296)

def rootsOf (pos : Array Pos) : ByteArray :=
  pos.foldl (fun acc p => pushU64LE acc p.index) ByteArray.empty

def liveOf (live : Array Bool) : ByteArray :=
  ByteArray.mk (live.map fun b => if b then 1 else 0)

/-- `sims` simulations on every live game at once: one descent per game in C, one
    forward over the pending leaves, expand and back up in C. Root noise (self-play
    only) is mixed into the root's priors, whether it was expanded this move or reused. -/
def search (ctx : Ctx) (roots : Array Pos) (live : Array Bool) (sims : Nat) (noise : Bool)
    (g0 : StdGen) : IO StdGen := do
  let rootsBA := rootsOf roots
  let liveBA := liveOf live
  let mut g := g0
  if noise then
    let (sd, g') := seed64 g
    g := g'
    mctsRootNoise ctx.arena rootsBA liveBA ctx.alpha ctx.eps sd
  for _ in [0:sims] do
    let sel ← mctsSelect ctx.arena rootsBA liveBA ctx.cpuct
    let mut pend := ByteArray.empty
    let mut nPend := 0
    for i in [0:roots.size] do
      if readU32 sel (4 * i) == 1 then
        pend := pushU32LE pend (readU32 sel (4 * i + 2))   -- the leaf index, below 2³² at n ≤ 4
        nPend := nPend + 1
    let logits ← if nPend == 0 then pure ByteArray.empty else ctx.forward pend nPend
    let (sd, g') := seed64 g
    g := g'
    mctsExpandBackup ctx.arena logits ctx.nOut.toUSize (if noise then 1 else 0) ctx.alpha ctx.eps sd
  return g

/-- The roots' visit distributions, one array per game. -/
def visitDists (ctx : Ctx) (roots : Array Pos) : IO (Array (Array Float)) := do
  let v ← mctsRootVisits ctx.arena (rootsOf roots)
  return (Array.range roots.size).map fun i =>
    (Array.range ctx.nc).map fun c => F32.read v (i * ctx.nc + c).toUSize

def argmaxLegal (score : Nat → Float) (pos : Pos) : Nat := Id.run do
  let mut best := 0
  let mut bestS := -1.0e30
  for c in pos.legal do
    if score c > bestS then
      bestS := score c
      best := c
  return best

/-- One iteration of self-play: `bf` games in lockstep, `sims` simulations a move,
    sampled from the visit distribution for the first `tempPlies` plies then greedy.
    Returns the samples (position index, π, z from the mover's view) and the games'
    results from X's view. -/
def selfPlay (ctx : Ctx) (root : Pos) (sims tempPlies : Nat) (g0 : StdGen) :
    IO (Array Nat × Array (Array Float) × Array Float × Array Int × StdGen) := do
  let G := ctx.bf
  mctsReset ctx.arena
  let mut pos : Array Pos := Array.replicate G root
  let mut live : Array Bool := Array.replicate G true
  let mut hist : Array (Array (Nat × Array Float × UInt8)) := Array.replicate G #[]
  let mut results : Array Int := Array.replicate G 0
  let mut g := g0
  let mut ply := 0
  while live.any id do
    g ← search ctx pos live sims true g
    let pis ← visitDists ctx pos
    for i in [0:G] do
      if live[i]! then
        let p := pos[i]!
        let pi := pis[i]!
        hist := hist.modify i (·.push (p.index, pi, p.mover))
        let (a, g') := if ply < tempPlies then sampleFrom pi g
                       else (argmaxLegal (fun c => pi[c]!) p, g)
        g := g'
        let p' := p.play a
        pos := pos.set! i p'
        if p'.terminal then
          live := live.set! i false
          results := results.set! i p'.result
    ply := ply + 1
  let mut idx : Array Nat := #[]
  let mut pis : Array (Array Float) := #[]
  let mut zs : Array Float := #[]
  for i in [0:G] do
    for (ix, pi, mover) in hist[i]! do
      idx := idx.push ix
      pis := pis.push pi
      zs := zs.push (Float.ofInt (if mover == 1 then results[i]! else -results[i]!))
  return (idx, pis, zs, results, g)

/-- `bf` games with the net on one side against the perfect player (uniform over the
    optimal set), in lockstep. `sims = 0` is the net alone: the argmax of its masked
    logits. W/D/L from the net's view. -/
def matchVsPerfect (ctx : Ctx) (t : Table) (root : Pos) (netIsX : Bool) (sims : Nat)
    (g0 : StdGen) : IO (WDL × StdGen) := do
  let G := ctx.bf
  mctsReset ctx.arena
  let mut pos : Array Pos := Array.replicate G root
  let mut live : Array Bool := Array.replicate G true
  let mut res : WDL := {}
  let mut g := g0
  let perfect := perfectPlayer t
  let mut ply := 0
  while live.any id do
    -- every live game sits at ply `ply`: X's turn on even plies (a finished game's
    -- position has fewer stones, so its mover says nothing about the live ones)
    let netTurn := (ply % 2 == 0) == netIsX
    let mut moves : Array Nat := Array.replicate G 0
    if netTurn then
      if sims == 0 then
        let mut idx := ByteArray.empty
        for i in [0:G] do idx := pushU32LE idx pos[i]!.index
        let logits ← ctx.forward idx G
        for i in [0:G] do
          moves := moves.set! i (argmaxLegal (fun c => F32.read logits (i * ctx.nOut + c).toUSize) pos[i]!)
      else
        g ← search ctx pos live sims false g
        let pis ← visitDists ctx pos
        for i in [0:G] do
          if live[i]! then
            moves := moves.set! i (argmaxLegal (fun c => pis[i]![c]!) pos[i]!)
    else
      for i in [0:G] do
        if live[i]! then
          let (a, g') := perfect pos[i]! g
          g := g'
          moves := moves.set! i a
    for i in [0:G] do
      if live[i]! then
        let p' := pos[i]!.play moves[i]!
        pos := pos.set! i p'
        if p'.terminal then
          live := live.set! i false
          res := res.add (if netIsX then p'.result else -p'.result)
    ply := ply + 1
  return (res, g)

/-- The sweep: every position of `idx` (u32, `count` of them) through the net in
    chunks of `bf`, scored in C against the table. Returns
    (agree fraction, value MSE, sign-agree fraction, value MAE). -/
def sweep (ctx : Ctx) (t : Table) (idx : ByteArray) (count : Nat) :
    IO (Float × Float × Float × Float) := do
  let mut agree := 0.0
  let mut mse := 0.0
  let mut sign := 0.0
  let mut mae := 0.0
  let mut off := 0
  while off < count do
    let m := min ctx.bf (count - off)
    let chunk := idx.extract (4 * off) (4 * (off + m))
    let logits ← ctx.forward chunk m
    let r ← scoreLogits t.tbl chunk m.toUSize ctx.n.toUSize logits ctx.nOut.toUSize
    agree := agree + F32.read r 0
    mse := mse + F32.read r 1
    sign := sign + F32.read r 2
    mae := mae + F32.read r 3
    off := off + m
  let c := count.toFloat
  return (agree / c, mse / c, sign / c, mae / c)

/-- One CSV row per position: index, exact value, the net's value, its move, whether
    that move is optimal, and its softmax over the cells. -/
def dumpSweep (ctx : Ctx) (t : Table) (idx : ByteArray) (count : Nat) (seen : ByteArray)
    (path : String) : IO Unit := do
  let mut rows := "index,exact,value,move,optimal,mover,seen" ++
    String.join ((List.range ctx.nc).map fun c => s!",p{c}") ++ "\n"
  let mut off := 0
  while off < count do
    let m := min ctx.bf (count - off)
    let chunk := idx.extract (4 * off) (4 * (off + m))
    let logits ← ctx.forward chunk m
    for i in [0:m] do
      let ix := readU32 chunk i
      let p := posOfIndex t ix
      let P := priors logits i ctx.nOut p
      let a := argmaxLegal (fun c => P[c]!) p
      let v := Float.tanh (F32.read logits (i * ctx.nOut + ctx.nc).toUSize)
      rows := rows ++ s!"{ix},{t.value p},{fmt v 4},{a},{if (t.optimal p).contains a then 1 else 0},\
{if p.mover == 1 then "X" else "O"},{seen[ix]!}" ++
        String.join ((List.range ctx.nc).map fun c => s!",{fmt P[c]! 4}") ++ "\n"
    off := off + m
  IO.FS.writeFile path rows
where
  posOfIndex (t : Table) (idx : Nat) : Pos := Id.run do
    let mut p := Pos.empty t.n t.k
    let mut i := idx
    for c in [0:t.n * t.n] do
      let d := (i % 3).toUInt8
      if d != 0 then p := { p with cells := p.cells.set! c d, stones := p.stones + 1 }
      i := i / 3
    return p

end AlphaZeroTtt

open AlphaZeroTtt in
def main (args : List String) : IO Unit := do
  let kv (key : String) : Option String :=
    (args.find? (·.startsWith (key ++ "="))).map (·.drop (key.length + 1) |>.toString)
  let natArg (key : String) (d : Nat) : Nat := ((kv key) >>= String.toNat?).getD d
  let floatArg (key : String) (d : Float) : Float := ((kv key) >>= parseFloat).getD d
  let n := natArg "n" 3
  let k := natArg "k" n
  let iters := natArg "iters" 20
  let sims := natArg "sims" 25
  let bf := natArg "bf" 256
  let B := natArg "batch" 64
  let epochs := natArg "epochs" 10
  let lr := floatArg "lr" 1.0e-3
  let window := natArg "window" 20
  let cpuct := floatArg "cpuct" 1.5
  let alpha := floatArg "alpha" 1.0
  let eps := floatArg "noise" 0.25
  let tempPlies := natArg "tempPlies" (n * n)
  let sweepN := natArg "sweep" 0
  let seed := natArg "seed" 1
  let tag := kv "tag"
  let centre := (n / 2) * n + n / 2
  let probe : List Nat := ((kv "probe").getD s!"0,{3 ^ centre}").splitOn "," |>.filterMap String.toNat?
  let spec := net n
  let nc := n * n
  let nOut := nc + 1
  IO.eprintln s!"{spec.name}: {spec.totalParams} params, {k} in a row, {iters} iterations of \
{bf} games at {sims} sims, batch {B} × {epochs} epochs over the last {window} iterations, \
lr {lr}, cpuct {cpuct}, Dir({alpha}) at {eps}, seed {seed}"
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "alphazero-ttt runs on the XLA backend only"
  let t0 ← IO.monoMsNow
  let table ← Table.build n k
  let decIdx ← reachableIdx table.tbl n.toUSize 0
  let nDec := decIdx.size / 4
  let root := Pos.empty n k
  IO.eprintln s!"  table {table.tbl.size} bytes, {nDec} decision positions ({(← IO.monoMsNow) - t0} ms)"
  -- the sweep set: every decision position, or a fixed random subsample of `sweep`
  let (sweepIdx, nSweep) ← if sweepN == 0 || sweepN >= nDec then pure (decIdx, nDec) else do
    let mut g := mkStdGen 4242
    let mut out := ByteArray.empty
    for _ in [0:sweepN] do
      let (i, g') := randNat g 0 (nDec - 1)
      g := g'
      out := pushU32LE out (readU32 decIdx i)
    pure (out, sweepN)

  -- ── graphs: the train step at B, one eval forward at bf ──
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix ++ (match tag with | some t => "_" ++ t | none => "")
  IO.FS.writeFile s!"{pfx}_train_step.mlir" (MlirCodegen.generateTrainStep spec B
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := 0.0) (useAdam := true) (useDdpm := true) (ddpmOutShape := [B, nOut, 1, 1]))
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec bf)
  let sess ← LowererSession.create (← NetSpec.graphArtifact pfx "train_step")
  -- two sessions on the one eval graph: the search's forwards hold the parameters on
  -- the device across an iteration (re-seeded once, when training ends); the train
  -- step's own forward, whose parameters move every step, takes the copying path
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval")
  let stepSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval")
  IO.eprintln "  sessions loaded"
  let nP := spec.totalParams
  let nT := 3 * nP
  let allShapes := spec.shapesBA
  let evalShapes := spec.evalShapesBA
  let bnShapes := spec.bnShapesBA
  let nRes : USize := spec.evalShapes.size.toUSize
  let xShB := spec.xShape B
  let xShF := spec.xShape bf
  let mut p ← spec.heInitParams
  let mut m ← F32.const nP.toUSize 0.0
  let mut v ← F32.const nP.toUSize 0.0
  let mut evalParams := p
  let mut gen := 1        -- moves whenever evalParams does; the held forward re-seeds on it
  let evalParamsRef ← IO.mkRef evalParams
  let genRef ← IO.mkRef gen
  let forward (idx : ByteArray) (count : Nat) : IO ByteArray := do
    let mut idxF := idx
    for _ in [count:bf] do idxF := pushU32LE idxF 0   -- pad with the empty board
    let x ← planesOf idxF bf.toUSize n.toUSize
    LowererSession.forwardF32 evalSess spec.evalFnName (← evalParamsRef.get) evalShapes x xShF
      bf.toUSize nOut.toUSize nRes (← genRef.get).toUSize
  -- a game's tree holds at most one node per simulation per ply, plus the root
  let arena ← mctsAlloc bf.toUSize n.toUSize k.toUSize (nc * sims + 16).toUSize
  let ctx : Ctx := { n, nOut, bf, cpuct, alpha, eps, arena, forward }
  -- a shape check before anything trains: bf boards in, bf × nOut logits out
  let probeOut ← forward (pushU32LE ByteArray.empty 0) 1
  unless probeOut.size == bf * nOut * 4 do
    throw <| IO.userError s!"eval forward returned {probeOut.size} bytes, expected {bf * nOut * 4}"

  let instrument (iter : Nat) (g : StdGen) : IO (String × StdGen) := do
    let (agree, mse, sign, mae) ← sweep ctx table sweepIdx nSweep
    let (aX, g) ← matchVsPerfect ctx table root true 0 g
    let (aO, g) ← matchVsPerfect ctx table root false 0 g
    let (mX, g) ← matchVsPerfect ctx table root true sims g
    let (mO, g) ← matchVsPerfect ctx table root false sims g
    let rootOut ← forward (pushU32LE ByteArray.empty 0) 1
    let rootV := Float.tanh (F32.read rootOut nc.toUSize)
    IO.eprintln s!"  iter {iter}: sweep agree {fmt (100.0 * agree) 2}% value mse {fmt mse 4} \
sign {fmt (100.0 * sign) 2}%  net alone vs perfect X {aX.str} O {aO.str}  +MCTS X {mX.str} O {mO.str}  \
root value {fmt rootV 3}"
    let row := s!"{fmt agree 4},{fmt mse 4},{fmt sign 4},{fmt mae 4},{aX.w},{aX.d},{aX.l},\
{aO.w},{aO.d},{aO.l},{mX.w},{mX.d},{mX.l},{mO.w},{mO.d},{mO.l},{fmt rootV 4}"
    return (row, g)

  let mut curve := "iter,samples,steps,loss,agree,value_mse,sign_agree,value_mae,\
alone_x_w,alone_x_d,alone_x_l,alone_o_w,alone_o_d,alone_o_l,mcts_x_w,mcts_x_d,mcts_x_l,\
mcts_o_w,mcts_o_d,mcts_o_l,root_value,seen,ms\n"
  let mut g := mkStdGen seed
  let (row0, g0) ← instrument 0 g
  g := g0
  curve := curve ++ s!"0,0,0,,{row0},0,{(← IO.monoMsNow) - t0}\n"
  -- the replay: per-iteration (planes, π, z) blocks, the last `window` kept
  let mut pool : Array (ByteArray × ByteArray × ByteArray × Nat) := #[]
  let mut steps := 0
  -- which decision positions self-play ever stood at: the instrument for the gap
  -- between the match record and the sweep
  let mut seen : ByteArray := ByteArray.mk (Array.replicate table.tbl.size 0)
  let mut nSeen := 0
  for iter in [1:iters + 1] do
    let tI ← IO.monoMsNow
    let (idx, pis, zs, results, g1) ← selfPlay ctx root sims tempPlies g
    g := g1
    let mut idxBA := ByteArray.empty
    for i in idx do
      idxBA := pushU32LE idxBA i
      if seen[i]! == 0 then
        seen := seen.set! i 1
        nSeen := nSeen + 1
    let planes ← planesOf idxBA idx.size.toUSize n.toUSize
    let mut piBA := ByteArray.emptyWithCapacity (idx.size * nc * 4)
    for pi in pis do
      for c in [0:nc] do piBA := pushF32LE piBA pi[c]!
    let mut zBA := ByteArray.emptyWithCapacity (idx.size * 4)
    for z in zs do zBA := pushF32LE zBA z
    pool := pool.push (planes, piBA, zBA, idx.size)
    if pool.size > window then pool := pool.extract 1 pool.size
    let xWins := results.foldl (fun a r => if r > 0 then a + 1 else a) 0
    let oWins := results.foldl (fun a r => if r < 0 then a + 1 else a) 0
    let tS ← IO.monoMsNow
    -- the window as one block
    let mut allPlanes := ByteArray.empty
    let mut allPi := ByteArray.empty
    let mut allZ := ByteArray.empty
    let mut N := 0
    for (pl, pi, z, cnt) in pool do
      allPlanes := allPlanes.append pl
      allPi := allPi.append pi
      allZ := allZ.append z
      N := N + cnt
    let nSteps := epochs * N / B
    let mut lossAcc := 0.0
    for s in [0:nSteps] do
      let mut bIdx := ByteArray.empty
      let mut zB := ByteArray.empty
      for _ in [0:B] do
        let (i, g') := randNat g 0 (N - 1)
        g := g'
        bIdx := pushU32LE bIdx i
        zB := pushF32LE zB (F32.read allZ i.toUSize)
      let (x, piB) ← gatherAug allPlanes allPi bIdx B.toUSize n.toUSize (seed * 7919 + steps + 1).toUInt64
      let out ← LowererSession.forwardF32 stepSess spec.evalFnName evalParams evalShapes
        (x.append (← F32.const ((bf - B) * 2 * nc).toUSize 0.0)) xShF bf.toUSize nOut.toUSize 0 0
      let (y, lossBA) ← targets out piB zB B.toUSize nOut.toUSize (nOut.toFloat / 2.0)
      steps := steps + 1
      let packed := (p.append m).append v
      let res ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName packed allShapes x xShB y
        lr steps.toFloat bnShapes B.toUSize nOut.toUSize 1 1
      p := F32.slice res 0 nP
      m := F32.slice res nP nP
      v := F32.slice res (2 * nP) nP
      lossAcc := lossAcc + F32.read lossBA 0
      evalParams := p
      if s == 0 && iter == 1 then
        IO.eprintln s!"  first step: alphazero loss {fmt (F32.read lossBA 0) 4}, \
block loss {fmt (F32.extractLoss res nT) 4}"
    -- the held forwards see the new parameters from here on: one re-seed per iteration
    evalParamsRef.set evalParams
    gen := gen + 1
    genRef.set gen
    let tE ← IO.monoMsNow
    let (row, g2) ← instrument iter g
    g := g2
    let tF ← IO.monoMsNow
    IO.eprintln s!"    self-play {idx.size} positions (X {xWins} / draw {bf - xWins - oWins} / O {oWins}), \
{nSteps} steps over {N} samples, loss {fmt (lossAcc / (max 1 nSteps).toFloat) 4}  \
(play {tS - tI} train {tE - tS} score {tF - tE} ms)"
    curve := curve ++ s!"{iter},{N},{steps},{fmt (lossAcc / (max 1 nSteps).toFloat) 4},{row},{nSeen},{tF - t0}\n"
    IO.FS.writeFile s!"{pfx}_curve.csv" curve
  IO.eprintln s!"trained: {iters} iterations, {steps} steps, {(← IO.monoMsNow) - t0} ms"
  -- the table's number: when the per-iteration sweep was a subsample, sweep every
  -- reachable decision position once with the final parameters
  if nSweep < nDec then
    let tA ← IO.monoMsNow
    let (agree, mse, sign, mae) ← sweep ctx table decIdx nDec
    IO.eprintln s!"full sweep, {nDec} decision positions: agree {fmt (100.0 * agree) 2}% value mse \
{fmt mse 4} sign {fmt (100.0 * sign) 2}% mae {fmt mae 4} ({(← IO.monoMsNow) - tA} ms)"
    IO.FS.writeFile s!"{pfx}_full_sweep.txt"
      s!"positions,agree,value_mse,sign_agree,value_mae\n{nDec},{fmt agree 4},{fmt mse 4},{fmt sign 4},{fmt mae 4}\n"
  -- the figure's inputs: the full sweep when it is small enough to plot, and the
  -- probe positions' policies
  IO.eprintln s!"self-play stood at {nSeen} of the {nDec} decision positions"
  if nSweep <= 100000 then dumpSweep ctx table sweepIdx nSweep seen s!"{pfx}_sweep.csv"
  let mut probeBA := ByteArray.empty
  for ix in probe do probeBA := pushU32LE probeBA ix
  dumpSweep ctx table probeBA probe.length seen s!"{pfx}_policy.csv"
  IO.FS.writeBinFile s!"{pfx}_params.bin" evalParams
  IO.eprintln s!"wrote {pfx}_curve.csv, {pfx}_sweep.csv, {pfx}_policy.csv, {pfx}_params.bin"
