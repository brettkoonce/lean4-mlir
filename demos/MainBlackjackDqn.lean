import LeanMlir
import LeanMlir.Blackjack

/-! DQN on blackjack — rung 2 of `planning/blackjack_dqn_demo.md` (§4).

    The Q-function is a three-layer dense net on a 29-float one-hot of the
    observation, trained through the stack on the rank-2 DDPM MSE block with no
    new codegen. Per step the host runs the eval graph on the replay batch,
    keeps the net's own prediction in the untaken action's slot and overwrites
    the taken one with the Bellman target, so the MSE gradient is zero on the
    untaken slot and a per-sample TD error on the taken one. Every 1000 steps
    the greedy policy is read off the online net over all 280 observations in
    one batched forward and scored EXACTLY by `BJ.gameValue`; that is the
    training curve, and its ceiling is the DP optimum.

    XLA backend only. `lake exe blackjack-dqn [steps=50000] [seed=1] [double] [lrdecay]`;
    `lrdecay` runs the learning rate linearly from 1e-3 to 1e-4 over the updates;
    `tag=<name>` suffixes every file the run writes under `.lake/build/`.
    Writes `<prefix>_curve.csv` and `<prefix>_policy.csv` next to the graphs. -/

open BJ

namespace DQN

/-- One-hot of the sum (4..21, 18 slots), one-hot of the dealer card (10), the
    ace bit. A linear map of this already represents any table policy. -/
def obsDim : Nat := 29

def net : NetSpec where
  name := "blackjack dqn"
  imageH := 1
  imageW := 1
  layers := [
    .dense obsDim 64 .relu,
    .dense 64 64 .relu,
    .dense 64 2 .identity     -- slot 0 = stick, slot 1 = hit
  ]

/-- Every observation the game can show: hard 4..21 × dealer 1..10 (180), then
    soft 12..21 × dealer (100). A soft hand is at least A+1 = 12. -/
def allObs : Array Obs := Id.run do
  let mut a : Array Obs := #[]
  for s in [4:22] do
    for up in [1:11] do a := a.push { sum := s, dealer := up, usable := false }
  for s in [12:22] do
    for up in [1:11] do a := a.push { sum := s, dealer := up, usable := true }
  return a

def nObs : Nat := 280

def obsIndex (o : Obs) : Nat :=
  if o.usable then 180 + (o.sum - 12) * 10 + (o.dealer - 1)
  else (o.sum - 4) * 10 + (o.dealer - 1)

@[inline] def pushF32 (acc : ByteArray) (x : Float) : ByteArray :=
  let u : UInt32 := x.toFloat32.toBits
  (((acc.push (u &&& 0xff).toUInt8).push ((u >>> 8) &&& 0xff).toUInt8).push
    ((u >>> 16) &&& 0xff).toUInt8).push ((u >>> 24) &&& 0xff).toUInt8

/-- Append the 29-float encoding of `o`. A bust sum (> 21) encodes as all-zero
    sum slots; it only ever appears as the `s'` of a terminal transition. -/
def pushObs (acc : ByteArray) (o : Obs) : ByteArray := Id.run do
  let mut acc := acc
  for i in [0:18] do acc := pushF32 acc (if o.sum == 4 + i then 1.0 else 0.0)
  for i in [0:10] do acc := pushF32 acc (if o.dealer == 1 + i then 1.0 else 0.0)
  acc := pushF32 acc (if o.usable then 1.0 else 0.0)
  return acc

structure Transition where
  s : Obs
  hit : Bool
  r : Float
  s' : Obs
  done : Bool
deriving Inhabited

end DQN

open DQN in
def main (args : List String) : IO Unit := do
  let nums := args.filterMap String.toNat?
  let steps := (nums[0]?).getD 50000
  let seed := (nums[1]?).getD 1
  let double := args.any (· == "double")
  let lrDecay := args.any (· == "lrdecay")
  -- `tag=<name>` keeps several arms' graphs and outputs apart under .lake/build
  let tag := (args.find? (·.startsWith "tag=")).map (·.drop 4)
  let B : Nat := 128
  let lr : Float := 0.001
  let gamma : Float := 1.0
  let ring : Nat := 20000
  let warm : Nat := 1000
  let epsHi : Float := 1.0
  let epsLo : Float := 0.05
  let epsHands : Nat := 20000
  let targetEvery : Nat := 500
  let refreshEvery : Nat := 50
  let logEvery : Nat := 1000
  let spec := net
  IO.eprintln s!"{spec.name}: {spec.totalParams} params, {steps} steps, batch {B}, \
{if double then "double" else "plain"} DQN{if lrDecay then ", lr decay" else ""}, seed {seed}"
  unless (← LowererSession.backendName) == "xla" do
    throw <| IO.userError "blackjack-dqn runs on the XLA backend only (the eval graphs \
are loaded as .mlir; there is no iree-compile step here)"

  -- ── graphs: the DDPM-MSE train step at B, the eval forward at B and at 280 ──
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix ++ (match tag with | some t => "_" ++ t | none => "")
  let outShape : List Nat := [B, 2, 1, 1]   -- rank-4 by the FFI's convention; the loss reshapes
  let trainMlir := MlirCodegen.generateTrainStep spec B
    ("jit_" ++ spec.sanitizedName ++ "_train_step")
    (weightDecay := 0.0) (useAdam := true)
    (useDdpm := true) (ddpmOutShape := outShape)
  IO.FS.writeFile s!"{pfx}_train_step.mlir" trainMlir
  IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec B)
  IO.FS.writeFile s!"{pfx}_fwd_eval_all.mlir" (MlirCodegen.generateEval spec nObs)
  let sess ← LowererSession.create (← NetSpec.graphArtifact pfx "train_step")
  let evalSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval")
  let evalAllSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval_all")
  IO.eprintln "  sessions loaded"

  let nP := spec.totalParams
  let nT := 3 * nP
  let allShapes := spec.shapesBA
  let evalShapes := spec.evalShapesBA
  let bnShapes := spec.bnShapesBA
  let xShB := spec.xShape B
  let xShAll := spec.xShape nObs
  let mut p ← spec.heInitParams
  let mut m ← F32.const nP.toUSize 0.0
  let mut v ← F32.const nP.toUSize 0.0
  let mut target := p

  -- the 280-observation batch, built once; one forward reads the whole policy
  let xAll := allObs.foldl pushObs ByteArray.empty
  let readGreedy (params : ByteArray) : IO (Array Bool) := do
    let q ← LowererSession.forwardF32 evalAllSess spec.evalFnName params evalShapes
              xAll xShAll nObs.toUSize 2
    return (Array.range nObs).map fun i =>
      F32.read q (2 * i + 1).toUSize > F32.read q (2 * i).toUSize
  let polOf (greedy : Array Bool) : Pol := fun o => if greedy[obsIndex o]! then 1.0 else 0.0

  let mut buf : Array Transition := Array.mkEmpty ring
  let mut wr := 0
  let mut g := mkStdGen seed
  let mut greedy : Array Bool := Array.replicate nObs true
  let mut hands := 0
  let mut updates := 0
  let mut lossAcc := 0.0
  let mut lossN := 0
  let mut curve : Array (Nat × Nat × Float × Nat) := #[]
  let (s0, g0) := reset g
  g := g0
  let mut s := s0
  let t0 ← IO.monoMsNow
  for _ in [0:steps + warm] do
    if updates >= steps then break
    -- ── act: ε-greedy from the last greedy readout (280 states, refreshed every 50 steps) ──
    let eps := if hands >= epsHands then epsLo
               else epsHi + (epsLo - epsHi) * hands.toFloat / epsHands.toFloat
    let o := s.obs
    let (u, g1) := randNat g 0 999999
    let (u2, g2) := randNat g1 0 1
    let hit := if u.toFloat / 1000000.0 < eps then u2 == 1 else greedy[obsIndex o]!
    let (s', r, d, g3) := BJ.step s hit g2
    let tr : Transition := { s := o, hit := hit, r := r, s' := s'.obs, done := d }
    if buf.size < ring then buf := buf.push tr else buf := buf.set! wr tr
    wr := (wr + 1) % ring
    if d then
      hands := hands + 1
      let (sn, g4) := reset g3
      s := sn
      g := g4
    else
      s := s'
      g := g3
    -- ── learn ──
    if buf.size >= warm then
      let mut xs := ByteArray.empty
      let mut xn := ByteArray.empty
      let mut idx : Array Nat := Array.mkEmpty B
      for _ in [0:B] do
        let (k, g') := randNat g 0 (buf.size - 1)
        g := g'
        idx := idx.push k
        xs := pushObs xs buf[k]!.s
        xn := pushObs xn buf[k]!.s'
      let qs ← LowererSession.forwardF32 evalSess spec.evalFnName p evalShapes xs xShB B.toUSize 2
      let qn ← LowererSession.forwardF32 evalSess spec.evalFnName target evalShapes xn xShB B.toUSize 2
      let qo ← if double
        then LowererSession.forwardF32 evalSess spec.evalFnName p evalShapes xn xShB B.toUSize 2
        else pure qn
      let mut y := ByteArray.empty
      for i in [0:B] do
        let t := buf[idx[i]!]!
        let q0 := F32.read qs (2 * i).toUSize
        let q1 := F32.read qs (2 * i + 1).toUSize
        let n0 := F32.read qn (2 * i).toUSize
        let n1 := F32.read qn (2 * i + 1).toUSize
        -- Double DQN: argmax from the online net, value from the target net.
        let boot := if double
          then (if F32.read qo (2 * i + 1).toUSize > F32.read qo (2 * i).toUSize then n1 else n0)
          else max n0 n1
        let tgt := t.r + (if t.done then 0.0 else gamma * boot)
        y := pushF32 y (if t.hit then q0 else tgt)
        y := pushF32 y (if t.hit then tgt else q1)
      updates := updates + 1
      let packed := (p.append m).append v
      let lrNow := if lrDecay then lr * (1.0 - 0.9 * updates.toFloat / steps.toFloat) else lr
      let out ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName
                  packed allShapes xs xShB y lrNow updates.toFloat bnShapes B.toUSize 2 1 1
      lossAcc := lossAcc + F32.extractLoss out nT
      lossN := lossN + 1
      p := F32.slice out 0 nP
      m := F32.slice out nP nP
      v := F32.slice out (2 * nP) nP
      if updates % targetEvery == 0 then target := p
      if updates % refreshEvery == 0 then greedy ← readGreedy p
      if updates % logEvery == 0 then
        let pol := polOf greedy
        let ev := gameValue (some pol)
        let ag := agreement pol
        curve := curve.push (updates, hands, ev, ag)
        let t1 ← IO.monoMsNow
        IO.eprintln s!"  update {updates}  hands {hands}  loss {fmt (lossAcc / lossN.toFloat) 4}  \
exact {fmt ev 4}  agree {ag}/200  ({t1 - t0} ms)"
        lossAcc := 0.0
        lossN := 0
  let t1 ← IO.monoMsNow
  IO.eprintln s!"trained: {updates} updates, {hands} hands, {t1 - t0} ms"

  -- ── final readout, chart, exact + Monte Carlo score ──
  greedy ← readGreedy p
  let pol := polOf greedy
  IO.println s!"DQN greedy policy after {updates} updates ({hands} hands, {if double then "double" else "plain"}{if lrDecay then ", lr decay" else ""}):"
  IO.println (chart pol false)
  IO.println (chart pol true)
  let ev := gameValue (some pol)
  let (mc, se) := mcEval pol 1000000 7
  let ag := agreement pol
  IO.println "arm                     exact value    Monte Carlo (1000000 hands)    agreement /200"
  IO.println s!"DQN                     {fmt ev 4}        {fmt mc 4} ± {fmt se 4}               {ag}"
  IO.println s!"exact optimum           {fmt (gameValue none) 4}"
  IO.FS.writeBinFile s!"{pfx}_params.bin" p
  let curveCsv := "updates,hands,exact,agreement\n" ++
    String.join (curve.toList.map fun (u, h, e, a) => s!"{u},{h},{fmt e 4},{a}\n")
  IO.FS.writeFile s!"{pfx}_curve.csv" curveCsv
  let polCsv := "usable,sum,dealer,dqn\n" ++
    String.join (allObs.toList.map fun o =>
      s!"{if o.usable then 1 else 0},{o.sum},{o.dealer},{if greedy[obsIndex o]! then "H" else "S"}\n")
  IO.FS.writeFile s!"{pfx}_policy.csv" polCsv
  IO.eprintln s!"wrote {pfx}_curve.csv, {pfx}_policy.csv, {pfx}_params.bin"
