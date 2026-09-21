import LeanMlir.VerifiedNets

/-! # The sync-BN gate, parameterised by net — `2×b` IS `1×2b`

The runner behind `mobilenetv2-syncbn-check` and `efficientnet-syncbn-check`: `resnet34-syncbn-check`'s
columns and verdict (its entry file under tests/, TestR34SyncBnCheck, carries the full account of
what each column measures), with the net, its batch, its committed artifacts and its run-time
renders handed in. Each net's entry file is a `SyncBnCheck.Cfg` and `main := SyncBnCheck.run cfg`.

    TEST         DP_sync([xA|xB]) at 2×b      ==  single_2b([xA|xB])
    COLLECTIVE   DP_sync([xA|xB])             vs  the one-replica SYNC graph at 2b
    FORMULATION  the one-replica SYNC graph   vs  single_2b          (the sync ops' arithmetic)
    CONTROL      DP_sync([xA|xB])             !=  mean(single_b(xA), single_b(xB))
    DUPLICATED   DP_sync([xA|xA])             vs  the one-replica SYNC graph at b on xA
    SENSITIVITY  single_2b vs itself with x perturbed by 1e-4·N(0,1) — the yardstick for m'

The verdict is on the handed-back BN statistics (sharp) and on the collective (DUPLICATED); the
gradient columns are printed against SENSITIVITY, because at a random-init operating point no
two f32 implementations agree on `m'` better than the forward's conditioning allows.

⚠ Only the per-replica `b` single-device step and the DP step are committed artifacts here; the
`2b` single-device step and both one-replica sync graphs are rendered to `.lake/build/` at run
time from the same functions. Needs TWO GPUs and the XLA backend.
-/

namespace SyncBnCheck

/-- One net's gate: the net, its per-replica batch, the committed `b` and `2×b` artifacts, and the
    single-device render at any batch (`forceSync := true` gives the one-replica sync graph). -/
structure Cfg where
  slug      : String                       -- artifact slug, e.g. `mobilenetv2`
  net       : VerifiedNet
  bs        : Nat                          -- per-replica batch of the committed DP artifact
  sgPath    : String                       -- committed single-device step at `bs`
  dpPath    : String                       -- committed sync-BN DP step at `2 × bs`
  render    : (B : Nat) → (forceSync : Bool) → String   -- single-device train step at batch `B`
  entry     : (B replicas : Nat) → String  -- its entry name, `m.<slug>_<variant>_train_step`
  /-- The split identity's bound on the handed-back statistics. `resnet34-syncbn-check` set 1e-3
      at 36 layers (measured 1.7e-4); a deeper net compounds more reduction-order rounding, and a
      net that needs more says why beside its value. -/
  statsTol  : Float := 1e-3

/-- Labels for one shard: class `(i + off) % nClasses`, packed as the driver's 4-byte records. -/
private def mkLabels (bs off nc : Nat) : ByteArray := Id.run do
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat ((i + off) % nc)); y := y.push 0; y := y.push 0; y := y.push 0
  y

/-- Max-abs difference and max-abs magnitude over `[lo, hi)` of `d` vs `ref`. -/
private def regionErr (lo hi : Nat) (d : ByteArray) (ref : Nat → Float) : Float × Float := Id.run do
  let mut err : Float := 0.0
  let mut mag : Float := 0.0
  for i in [lo:hi] do
    let x := F32.read d i.toUSize
    let r := ref i
    let e := (x - r).abs
    if e > err then err := e
    if r.abs > mag then mag := r.abs
  (err, mag)

def run (cfg : Cfg) : IO Unit := do
  let net := cfg.net
  let bs := cfg.bs
  let replicas := 2
  IO.println s!"{net.name} SYNC-BN gate — {replicas}×{bs} against 1×{2*bs}"
  IO.println s!"  TEST     DP_sync([xA|xB])  ==  single_{2*bs}([xA|xB])"
  IO.println s!"  CONTROL  DP_sync([xA|xB])  !=  mean(single_{bs}(xA), single_{bs}(xB))   (the old, per-replica identity)"
  IO.println s!"  single {bs}: {cfg.sgPath}\n  DP render: {cfg.dpPath}"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), {net.bnChannels.size} BN layers, \
backend {← LowererSession.backendName}"

  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  -- m = 0: m' = 0.1·g, linear in the gradient; v generic positive so θ' exercises Adam's nonlinearity
  let m ← F32.const net.nParams.toUSize 0.0
  let v ← F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat #[θ, m, v, tail, bnIn]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  -- two genuinely DIFFERENT shards — different pixels AND different labels
  let xA ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let xB ← F32.heInit 999 (bs * net.d0).toUSize 1.0
  let yA := mkLabels bs 0 net.nClasses
  let yB := mkLabels bs 5 net.nClasses
  let xAB := F32.concat #[xA, xB]
  let yAB := yA ++ yB

  -- the run-time renders: the 2b two-pass step, and the one-replica sync graph at 2b and at b
  let sg2Path := s!".lake/build/{cfg.slug}_syncgate_single{2*bs}_train_step.mlir"
  let syncPath := s!".lake/build/{cfg.slug}_syncgate_sync{2*bs}_train_step.mlir"
  let sync1Path := s!".lake/build/{cfg.slug}_syncgate_sync{bs}_train_step.mlir"
  IO.FS.writeFile sg2Path (cfg.render (2*bs) false)
  IO.FS.writeFile syncPath (cfg.render (2*bs) true)
  IO.FS.writeFile sync1Path (cfg.render bs true)

  IO.println s!"  single-device {2*bs} on [xA|xB]… ({sg2Path})"; (← IO.getStdout).flush
  let s2 ← mkSession sg2Path
  let e2 := cfg.entry (2*bs) 1
  let o2 ← LowererSession.mlpTrainStepV s2 e2
    xAB pbuf shapes yAB (2*bs).toUSize net.d0.toUSize net.nClasses.toUSize
  let s1 ← mkSession cfg.sgPath
  let e1 := cfg.entry bs 1
  IO.println s!"  single-device {bs} on xA, xB…"; (← IO.getStdout).flush
  let oA ← LowererSession.mlpTrainStepV s1 e1
    xA pbuf shapes yA bs.toUSize net.d0.toUSize net.nClasses.toUSize
  let oB ← LowererSession.mlpTrainStepV s1 e1
    xB pbuf shapes yB bs.toUSize net.d0.toUSize net.nClasses.toUSize
  IO.println s!"  single-device SYNC-graph {2*bs} on [xA|xB]… ({syncPath})"; (← IO.getStdout).flush
  let sS ← mkSession syncPath
  let oS ← LowererSession.mlpTrainStepV sS e2
    xAB pbuf shapes yAB (2*bs).toUSize net.d0.toUSize net.nClasses.toUSize
  -- SENSITIVITY: the two-pass graph on the same batch perturbed by 1e-4·N(0,1) per pixel
  let noise ← F32.heInit 7777 (2 * bs * net.d0).toUSize 1.0
  let xABp ← F32.axpySlice (F32.concat #[xA, xB]) 0 noise 0 (2 * bs * net.d0).toUSize 1.0e-4
  IO.println s!"  sensitivity probe (x + 1e-4·noise) on the two-pass {2*bs} graph…"; (← IO.getStdout).flush
  let o2p ← LowererSession.mlpTrainStepV s2 e2
    xABp pbuf shapes yAB (2*bs).toUSize net.d0.toUSize net.nClasses.toUSize
  -- the sync-BN data-parallel step on the split batch
  IO.println s!"  data-parallel {replicas}×{bs} on [xA|xB]…"; (← IO.getStdout).flush
  let sD ← mkSession cfg.dpPath
  let eD := cfg.entry bs replicas
  let oD ← LowererSession.mlpTrainStepVDP sD eD xAB pbuf shapes yAB
             (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize
  -- DUPLICATED: DP on [xA|xA] against the one-replica sync graph on xA at the same per-replica shape
  IO.println s!"  duplicated probe: DP on [xA|xA] vs the one-replica sync graph on xA ({sync1Path})…"
  (← IO.getStdout).flush
  let sS1 ← mkSession sync1Path
  let oS1 ← LowererSession.mlpTrainStepV sS1 e1
    xA pbuf shapes yA bs.toUSize net.d0.toUSize net.nClasses.toUSize
  let oDD ← LowererSession.mlpTrainStepVDP sD eD
    (F32.concat #[xA, xA]) pbuf shapes (yA ++ yA)
    (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  for (o, what) in [(o2, "single 2b"), (oA, "single b (A)"), (oB, "single b (B)"), (oS, "sync 2b"),
                    (oS1, "sync b"), (oDD, "DP duplicated")] do
    if o.size != oD.size then
      IO.eprintln s!"SIZE MISMATCH: {what} gives {o.size}, DP gives {oD.size}"; IO.Process.exit 1
  let nP := net.nParams
  let nOut := oD.size / 4
  let statsLabel := s!"bn stats ({2 * net.bnChannels.size} slots)"
  -- `[θ' | m' | v' | loss bc1 bc2 | bnstat]`: every region but the report-only loss scalar
  let regions : List (String × Nat × Nat) :=
    [("θ'", 0, nP), ("m' = 0.1·g", nP, 2*nP), ("v'", 2*nP, 3*nP), (statsLabel, 3*nP + 3, nOut)]
  let mut worstTest : Float := 0.0
  let mut worstCtrl : Float := 0.0
  let mut worstDup : Float := 0.0
  let mut statsTest : Float := 0.0
  let mut statsForm : Float := 0.0
  let mut statsDup : Float := 0.0
  let mut statsCtrl : Float := 0.0
  let mut mSens : Float := 0.0
  let mut nonFinite : Nat := 0
  for (what, lo, hi) in regions do
    for i in [lo:hi] do
      if !(F32.read oD i.toUSize).isFinite || !(F32.read o2 i.toUSize).isFinite then
        nonFinite := nonFinite + 1
    let (eT, mT) := regionErr lo hi oD (fun i => F32.read o2 i.toUSize)
    let (eC, mC) := regionErr lo hi oD
      (fun i => 0.5 * (F32.read oA i.toUSize + F32.read oB i.toUSize))
    let (eP, _)  := regionErr lo hi oD (fun i => F32.read oS i.toUSize)
    let (eF, _)  := regionErr lo hi oS (fun i => F32.read o2 i.toUSize)
    let (eSn, _) := regionErr lo hi o2p (fun i => F32.read o2 i.toUSize)
    let (eDup, mDup) := regionErr lo hi oDD (fun i => F32.read oS1 i.toUSize)
    let nrSn := if mT > 1e-30 then eSn / mT else 0.0
    let nrDup := if mDup > 1e-30 then eDup / mDup else 0.0
    let nrT := if mT > 1e-30 then eT / mT else 0.0
    let nrC := if mC > 1e-30 then eC / mC else 0.0
    let nrP := if mT > 1e-30 then eP / mT else 0.0
    let nrF := if mT > 1e-30 then eF / mT else 0.0
    IO.println s!"  ── {what} [{lo}, {hi}) ──"
    IO.println s!"    TEST        |DP − single_2b|      / max|single_2b| = {nrT}"
    IO.println s!"    COLLECTIVE  |DP − sync_2b|        / max|single_2b| = {nrP}"
    IO.println s!"    FORMULATION |sync_2b − single_2b| / max|single_2b| = {nrF}"
    IO.println s!"    CONTROL     |DP − mean(A,B)|      / max|mean|      = {nrC}"
    IO.println s!"    SENSITIVITY two-pass graph, x perturbed by 1e-4     = {nrSn}"
    IO.println s!"    DUPLICATED  |DP([A|A]) − sync_b(A)|  / max|sync_b|  = {nrDup}"
    if nrDup > worstDup then worstDup := nrDup
    if what == statsLabel then
      statsTest := nrT; statsForm := nrF; statsDup := nrDup; statsCtrl := nrC
    if what == "m' = 0.1·g" then mSens := nrSn
    if nrT > worstTest then worstTest := nrT
    if what != "θ'" && what != "v'" then
      if nrC > worstCtrl then worstCtrl := nrC
  -- the `*-dp-check` metric, for the duplicated probe: ‖DP([A|A]) − sync_b(A)‖ / ‖sync_b(A)‖ on m'
  let dupNormRel : Float := Id.run do
    let mut num : Float := 0.0
    let mut den : Float := 0.0
    for i in [nP:2*nP] do
      let a := F32.read oDD i.toUSize
      let b := F32.read oS1 i.toUSize
      num := num + (a - b) * (a - b)
      den := den + b * b
    return (if den > 0.0 then (num / den).sqrt else 0.0)
  IO.println s!"  DUPLICATED m' norm-rel (the *-dp-check metric, against the sync graph) = {dupNormRel}"
  -- `SYNCBN_VERBOSE=1`: the split identity per BN layer, in forward order. Compounded kernel
  -- rounding starts near 1e-7 at the first layer and grows with depth; a wrong statistic exchange
  -- is visible at the layer it enters.
  if (← IO.getEnv "SYNCBN_VERBOSE").isSome then
    IO.println "  ── per-layer BN statistics, TEST rel-err vs single_2b (mean | var) ──"
    let mut so := 3*nP + 3
    let mut j := 0
    for c in net.bnChannels do
      let (eM, mM) := regionErr so (so + c) oD (fun i => F32.read o2 i.toUSize)
      let (eV, mV) := regionErr (so + c) (so + 2*c) oD (fun i => F32.read o2 i.toUSize)
      IO.println s!"    bn{j} [{c}] : {if mM > 1e-30 then eM / mM else 0.0} | {if mV > 1e-30 then eV / mV else 0.0}"
      so := so + 2*c
      j := j + 1
  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite} non-finite outputs"; IO.Process.exit 1
  -- the verdict — the same four conditions and thresholds as `resnet34-syncbn-check`
  if statsForm > 1e-5 then
    IO.eprintln s!"SYNC-BN CHECK FAILED (formulation): the one-replica sync graph's statistics \
differ from the two-pass graph's by {statsForm} > 1e-5."
    IO.Process.exit 1
  if statsDup > 1e-5 || worstDup > 5e-3 then
    IO.eprintln s!"SYNC-BN CHECK FAILED (collective): DP_sync([A|A]) does not reproduce the \
one-replica sync graph on A — statistics {statsDup}, worst region {worstDup}."
    IO.Process.exit 1
  if statsCtrl < 2e-3 then
    IO.eprintln s!"VACUOUS: DP still reproduces mean(single_b(A), single_b(B)) to {statsCtrl} on \
the statistics — the BatchNorm statistics are NOT synchronised."
    IO.Process.exit 1
  if statsTest > cfg.statsTol then
    IO.eprintln s!"SYNC-BN CHECK FAILED (split): DP_sync([A|B]) statistics differ from \
single_2b([A|B]) by {statsTest} > {cfg.statsTol} — beyond what kernel rounding at two batch shapes explains."
    IO.Process.exit 1
  -- (5) the split identity at the FIRST BatchNorm layer, where nothing has compounded yet: its
  --     statistics depend only on the input and the stem conv, so a 32-row and a 64-row
  --     reduction may differ only in their last bits. A wrong exchange (Chan's correction, the
  --     packing, a collective's divisor) is visible here at full size.
  let c0 := net.bnChannels[0]!
  let (e0m, m0m) := regionErr (3*nP + 3) (3*nP + 3 + c0) oD (fun i => F32.read o2 i.toUSize)
  let (e0v, m0v) := regionErr (3*nP + 3 + c0) (3*nP + 3 + 2*c0) oD (fun i => F32.read o2 i.toUSize)
  let first := max (if m0m > 1e-30 then e0m / m0m else 0.0) (if m0v > 1e-30 then e0v / m0v else 0.0)
  if first > 1e-5 then
    IO.eprintln s!"SYNC-BN CHECK FAILED (first layer): DP_sync([A|B])'s first BN layer's statistics \
differ from single_2b's by {first} > 1e-5 — before any depth has compounded, so the exchange itself."
    IO.Process.exit 1
  IO.println s!"✓ SYNC-BN CONFIRMED for {cfg.slug}: the sync graph IS the two-pass graph \
(statistics {statsForm}), the collective composes it exactly (duplicated batch {statsDup}), and \
DP_sync([A|B]) = single_2b([A|B]) to {statsTest} on the statistics (first layer {first}) against a \
per-replica CONTROL of {statsCtrl} — {replicas}×{bs} IS 1×{2*bs}. m' TEST {worstTest} against a SENSITIVITY-to-1e-4 \
of {mSens}."

end SyncBnCheck
