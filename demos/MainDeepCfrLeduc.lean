import LeanMlir.Train
import LeanMlir.Leduc
import LeanMlir.CliArgs

/-! Deep CFR on Leduc hold'em, scored by exact exploitability. DRAFT — written before the
    tree was available; not yet elaborated. Mirrors `demos/MainAlphaZeroTtt.lean`'s session
    setup and loss trick; see planning/leduc_deep_cfr_demo.md §3.

    Brown, Lerer, Gross & Sandholm 2019's loop on `LeanMlir/Leduc.lean`: each iteration t and
    player p, one batched forward of p's advantage net over EVERY information set gives σ_t as
    a table (regret matching on the positive advantages; the table is exactly what per-node
    queries would return, since σ_t is fixed within an iteration), K external-sampling
    traversals in C write instantaneous regrets to p's advantage reservoir and the opponent's
    σ to the strategy reservoir, then p's advantage net is re-initialised and trained from
    scratch on its reservoir. After T iterations the average-strategy net is trained on the
    strategy reservoir. Both losses are weighted squared errors on the legal slots, delivered
    through the rank-2 DDPM MSE block as a host-built target (`lean_leduc_targets`): the host
    writes `y = out − (nOut/2)·g` with `g = w ⊙ m ⊙ (out − target)`, so the block's gradient is
    `g / B`. Zero new codegen; the block's own printed loss is meaningless.

    Three parameter sets share one compiled graph (the two advantage nets and the strategy
    net); the dense stack has no BatchNorm, so the eval forward the loss reads and the train
    step's forward are the same function.

    Arms scored at the end, all by the C instrument: the strategy net; SD-CFR (the exact
    average of the T stored advantage-net profiles, weighted by t and by own reach, no strategy
    net); tabular ES-MCCFR run to the same number of nodes touched (the matched-budget
    bracket); CFR+ at 1,000 iterations as the equilibrium; head-to-head against it.

    `lake exe deep-cfr-leduc [r=3] [T=100] [K=1000] [cap=1000000] [steps=1000]
     [stratSteps=4000] [batch=512] [lr=1e-3] [every=5] [seed=1] [tag=<name>]`;
    writes `<prefix>_curve.csv`, `<prefix>_strategy.bin`, `<prefix>_sdcfr.bin` under
    `.lake/build/`. XLA only.

    `mode=tabular` is Gate B (planning §7): the same loop with the advantage net replaced by a
    lookup table — σ_t regret-matched from the reservoir's summed sampled regrets, the average
    the t-weighted strategy reservoir — which is ES-MCCFR by another route, so its
    exploitability-against-nodes curve must be `leduc-env`'s. No GPU; separates a sampling
    bug from function-approximation error. -/

open FloatFmt Leduc

namespace DeepCfrLeduc

/-- The dense stack of planning §2 — the bestiary's `deepCfrLeducNet` (`Bestiary/DeepCFR.lean`),
    Steinberger 2019's three dense layers of 64 for Leduc: F → 64 → 64 → 64 → 3 (fold, call,
    raise), no BatchNorm. -/
def net (F : Nat) : NetSpec where
  name := s!"deep cfr leduc f{F}"
  imageH := 1
  imageW := 1
  layers := [
    .dense F 64 .relu,
    .dense 64 64 .relu,
    .dense 64 64 .relu,
    .dense 64 3 .identity
  ]

end DeepCfrLeduc

open DeepCfrLeduc

def main (args : List String) : IO Unit := do
  let natArg := CliArgs.natArg args
  let floatArg := CliArgs.floatArg args
  let kv := CliArgs.kv args
  let r := natArg "r" 3
  let T := natArg "T" 100
  let K := natArg "K" 1000
  let cap := natArg "cap" 1000000
  let steps := natArg "steps" 1000
  let stratSteps := natArg "stratSteps" 4000
  let B := natArg "batch" 512
  let lr := floatArg "lr" 1e-3
  let every := natArg "every" 5
  let seed := natArg "seed" ((← IO.getEnv "LEAN_MLIR_SEED").bind String.toNat? |>.getD 1)
  let tag := kv "tag"
  let tabular := kv "mode" == some "tabular"
  let t0 ← IO.monoMsNow
  let cnt ← counts r.toUSize
  let nI := readU64 cnt 0
  let F := readU64 cnt 1
  let (feats, mask) ← enumerate r.toUSize
  let spec := net F
  let nOut := 3
  IO.eprintln s!"Deep CFR on Leduc r = {r}: {nI} information sets, F = {F}, {spec.totalParams} params per net; \
T = {T}, K = {K}, reservoirs {cap}, {steps} / {stratSteps} Adam steps at batch {B}, lr {lr}, seed {seed}"
  -- the equilibrium the arms are scored against
  let eqArena ← cfrAlloc r.toUSize 0 0 0.0
  cfrIterate eqArena 1000
  let eq ← cfrAverage eqArena
  IO.eprintln s!"  CFR+ 1000: exploitability {← exploitability r eq}, value {fmt (headToHead r.toUSize eq eq) 5}"
  -- ── graphs: the train step at B, the eval forward over every information set and at B ──
  IO.FS.createDirAll ".lake/build"
  let pfx := spec.buildPrefix ++ (match tag with | some t => "_" ++ t | none => "")
  -- (train step, eval over every set, eval at B); none in the tabular mode, which has no net
  let sessions ← if tabular then pure none else do
    IO.FS.writeFile s!"{pfx}_train_step.mlir" (MlirCodegen.generateTrainStep spec B
      ("jit_" ++ spec.sanitizedName ++ "_train_step")
      (weightDecay := 0.0) (useAdam := true) (useDdpm := true) (ddpmOutShape := [B, nOut, 1, 1]))
    IO.FS.writeFile s!"{pfx}_fwd_eval.mlir" (MlirCodegen.generateEval spec nI)
    IO.FS.writeFile s!"{pfx}_fwd_eval_b.mlir" (MlirCodegen.generateEval spec B)
    let sess ← LowererSession.create (← NetSpec.graphArtifact pfx "train_step")
    let allSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval")
    let stepSess ← LowererSession.create (← NetSpec.graphArtifact pfx "fwd_eval_b")
    IO.eprintln "  sessions loaded"
    pure (some (sess, allSess, stepSess))
  let nP := spec.totalParams
  let allShapes := spec.shapesBA
  let evalShapes := spec.evalShapesBA
  let bnShapes := spec.bnShapesBA
  let xShB := spec.xShape B
  let xShAll := spec.xShape nI
  -- one forward over every information set (the copying path: parameters change every call)
  let forwardAll (p : ByteArray) : IO ByteArray := do
    let some (_, allSess, _) := sessions | throw <| IO.userError "deep-cfr-leduc: no net in the tabular mode"
    LowererSession.forwardF32 allSess spec.evalFnName p evalShapes feats xShAll nI.toUSize nOut.toUSize 0 0
  let forwardB (p x : ByteArray) : IO ByteArray := do
    let some (_, _, stepSess) := sessions | throw <| IO.userError "deep-cfr-leduc: no net in the tabular mode"
    LowererSession.forwardF32 stepSess spec.evalFnName p evalShapes x xShB B.toUSize nOut.toUSize 0 0
  -- train one parameter set from scratch on a reservoir; returns (params, mean loss)
  let train (init : ByteArray) (res : ByteArray) (nSteps : Nat) (seedBase : Nat) : IO (ByteArray × Float) := do
    let mut p := init
    let mut m ← F32.const nP.toUSize 0.0
    let mut v ← F32.const nP.toUSize 0.0
    let mut lossAcc := 0.0
    for s in [0:nSteps] do
      let (x, tail) ← reservoirSample res B.toUSize (seedBase * 1000003 + s + 1).toUInt64
      let out ← forwardB p x
      let (y, lossBA) ← targets out tail B.toUSize (nOut.toFloat / 2.0)
      let packed := (p.append m).append v
      let some (sess, _, _) := sessions | throw <| IO.userError "deep-cfr-leduc: no net in the tabular mode"
      let resS ← LowererSession.trainStepAdamF32Ddpm sess spec.trainFnName packed allShapes x xShB y
        lr (s + 1).toFloat bnShapes B.toUSize nOut.toUSize 1 1
      (p, m, v) := F32.unpackAdam resS nP
      lossAcc := lossAcc + F32.read lossBA 0
    return (p, lossAcc / (max 1 nSteps).toFloat)
  -- ── the loop ──
  let adv0 ← reservoirAlloc F.toUSize cap.toUInt64
  let adv1 ← reservoirAlloc F.toUSize cap.toUInt64
  let strat ← reservoirAlloc F.toUSize cap.toUInt64
  let mut params : Array ByteArray := #[← spec.heInitParams (42 + seed * 100003).toUSize,
                                        ← spec.heInitParams (43 + seed * 100003).toUSize]
  -- the current profile of both players, as tables, and the stored profiles for SD-CFR
  let mut sigma : Array ByteArray := #[← uniformTable r.toUSize, ← uniformTable r.toUSize]
  let mut stored := ByteArray.empty
  let mut weights := ByteArray.empty
  let mut nodes : Nat := 0
  let mut curve := "iter,nodes,steps,exploit_current,exploit_avg,loss0,loss1,ms\n"
  let mut stepsDone := 0
  -- σ_t for both players: from the advantage nets, or (Gate B) from the reservoirs' summed regrets
  let debug := kv "debug" == some "1"
  -- (`ps` is passed in: a closure over the `let mut` would hold the binding it was made with)
  let sigmaNow (t : Nat) (ps : Array ByteArray) : IO (Array ByteArray) := do
    if tabular then return #[← reservoirSigma r.toUSize adv0, ← reservoirSigma r.toUSize adv1]
    let mut out := #[]
    for q in [0, 1] do
      let adv ← forwardAll ps[q]!
      -- a net that returns a non-finite advantage would fold silently (the argmax of NaN is
      -- slot 0); fail loudly instead
      let mut bad := 0
      for i in [0:nI * nOut] do
        if !(F32.read adv i.toUSize).isFinite then bad := bad + 1
      if bad > 0 then throw <| IO.userError s!"deep-cfr-leduc: {bad} non-finite advantages from net {q} at iteration {t}"
      if debug then IO.FS.writeBinFile s!"{pfx}_adv{q}_t{t}.bin" adv
      out := out.push (← sigmaFromAdvantages r.toUSize adv)
    return out
  for t in [1:T + 1] do
    let tI ← IO.monoMsNow
    let mut losses := #[0.0, 0.0]
    for p in [0, 1] do
      sigma ← sigmaNow t params
      let res := if p == 0 then adv0 else adv1
      let n ← traverse r.toUSize sigma[0]! sigma[1]! feats mask p.toUSize K.toUSize t.toFloat
        (seed * 7919 + t * 2 + p + 1).toUInt64 res strat
      nodes := nodes + n.toNat
      if tabular then continue
      -- retrain from scratch
      let init ← spec.heInitParams (42 + seed * 100003 + t * 17 + p).toUSize
      let (pNew, loss) ← train init res steps (seed * 31 + t * 2 + p)
      if debug then
        IO.FS.writeBinFile s!"{pfx}_init{p}_t{t}.bin" init
        IO.FS.writeBinFile s!"{pfx}_params{p}_t{t}.bin" pNew
      params := params.set! p pNew
      losses := losses.set! p loss
      stepsDone := stepsDone + steps
    -- the profile after the iteration, stored for SD-CFR with weight t
    sigma ← sigmaNow t params
    -- one table holds both players' rows: P0's from sigma[0], P1's from sigma[1]
    let ks ← keys r.toUSize
    let mut prof := ByteArray.emptyWithCapacity (nI * 12)
    for i in [0:nI] do
      let src := if ks[i * 6]! == 0 then sigma[0]! else sigma[1]!
      for x in [0:3] do prof := pushF32LE prof (F32.read src (i * 3 + x).toUSize)
    stored := stored.append prof
    weights := pushF32LE weights t.toFloat
    if t % every == 0 || t == T then
      let eCur ← exploitability r prof
      -- the average: SD-CFR over the stored profiles, or (Gate B) the t-weighted strategy reservoir
      let avg ← if tabular then reservoirAverage r.toUSize strat
        else sdcfrAverage r.toUSize stored weights t.toUSize
      let eSd ← exploitability r avg
      let ms := (← IO.monoMsNow) - tI
      IO.eprintln s!"  iter {t}: nodes {nodes}, exploitability current {eCur} \
{if tabular then "average" else "sd-cfr"} {eSd}, losses {fmt losses[0]! 4} / {fmt losses[1]! 4} ({ms} ms)"
      curve := curve ++ s!"{t},{nodes},{stepsDone},{eCur},{eSd},{fmt losses[0]! 5},{fmt losses[1]! 5},{(← IO.monoMsNow) - t0}\n"
      IO.FS.writeFile s!"{pfx}_curve.csv" curve
  if tabular then
    let avg ← reservoirAverage r.toUSize strat
    let eAvg ← exploitability r avg
    IO.FS.writeFile s!"{pfx}_tabular_curve.csv" curve
    IO.FS.writeBinFile s!"{pfx}_tabular.bin" avg
    IO.eprintln s!"Gate B (tabular): exploitability {eAvg} at {nodes} nodes, vs CFR+ seat-averaged \
{fmt (seatAveraged r avg eq) 5}; compare `lake exe leduc-env`'s ES-MCCFR curve ({(← IO.monoMsNow) - t0} ms)"
    return
  -- ── the average-strategy net ──
  let st ← reservoirStats strat
  IO.eprintln s!"  strategy reservoir: {readU64 st 0} rows held of {readU64 st 1} seen; training {stratSteps} steps"
  let (pS, lossS) ← train (← spec.heInitParams (44 + seed * 100003).toUSize) strat stratSteps (seed * 97)
  let outS ← forwardAll pS
  let sigS ← sigmaFromStrategyNet r.toUSize outS
  let eS ← exploitability r sigS
  let sd ← sdcfrAverage r.toUSize stored weights T.toUSize
  let eSd ← exploitability r sd
  IO.eprintln s!"strategy net: exploitability {eS} (loss {fmt lossS 4}), vs CFR+ seat-averaged {fmt (seatAveraged r sigS eq) 5}"
  IO.eprintln s!"SD-CFR:       exploitability {eSd}, vs CFR+ seat-averaged {fmt (seatAveraged r sd eq) 5}"
  -- the matched-budget bracket: tabular ES-MCCFR run to the same number of nodes touched
  let es ← esAlloc r.toUSize seed.toUInt64
  let st ← esRun es nodes.toUInt64
  let esTbl ← esAverage es
  IO.eprintln s!"ES-MCCFR:     exploitability {← exploitability r esTbl} at {readU64 st 0} nodes \
(coverage {fmt ((readU64 st 2).toFloat / nI.toFloat) 3}), vs CFR+ seat-averaged {fmt (seatAveraged r esTbl eq) 5}"
  IO.FS.writeBinFile s!"{pfx}_esmccfr.bin" esTbl
  IO.FS.writeBinFile s!"{pfx}_strategy.bin" sigS
  IO.FS.writeBinFile s!"{pfx}_sdcfr.bin" sd
  IO.FS.writeBinFile s!"{pfx}_strategy_params.bin" pS
  IO.eprintln s!"trained: {T} iterations, {nodes} nodes touched, {stepsDone + stratSteps} steps, {(← IO.monoMsNow) - t0} ms; \
wrote {pfx}_curve.csv, {pfx}_strategy.bin, {pfx}_sdcfr.bin"
