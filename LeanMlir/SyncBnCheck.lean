import LeanMlir.VerifiedNets

/-! # The sync-BN gate, parameterised by net — `R×b` against `1×Rb`

The runner behind `mobilenetv2-syncbn-check`, `efficientnet-syncbn-check` and
`imagenet-syncbn-check`: `resnet34-syncbn-check`'s columns and verdict (its entry file under tests/,
TestR34SyncBnCheck, carries the full account of what each column measures), with the net, its
batch, its replica count, its committed artifacts and its run-time renders handed in. Each net's
entry file is a `SyncBnCheck.Cfg` and `main := SyncBnCheck.run cfg`. With `x₀ … x_{R-1}` the R
shards (different pixels and different labels):

    TEST         DP_sync([x₀|…]) at R×b       ==  single_Rb([x₀|…])
    COLLECTIVE   DP_sync([x₀|…])              vs  the one-replica SYNC graph at Rb
    FORMULATION  the one-replica SYNC graph   vs  single_Rb          (the sync ops' arithmetic)
    CONTROL      DP_sync([x₀|…])              !=  mean_k single_b(x_k)
    DUPLICATED   DP_sync([x₀|x₀|…])           vs  the one-replica SYNC graph at b on x₀
    SENSITIVITY  single_Rb vs itself with x perturbed by 1e-4·N(0,1) — the yardstick for the gradient
    REPEAT       single_Rb vs itself, same input, run twice — XLA's run-to-run floor

The verdict is on the handed-back BN statistics (sharp) and on the collective (DUPLICATED); the
gradient columns are printed against SENSITIVITY, because at a random-init operating point no
two f32 implementations agree on `m'` better than the forward's conditioning allows.

Only the DP step (and, where one exists, the per-replica `b` single-device step) is a
committed artifact here; the `Rb` single-device step and both one-replica sync graphs are rendered
to `.lake/build/` at run time from the same functions — and so is the DP step when `dpPath := ""`
(ResNet-50, gated below its committed batch). Needs `R` GPUs and the XLA backend.
-/

namespace SyncBnCheck

/-- One net's gate: the net, its per-replica batch and replica count, the committed `R×b` artifact
    (and `b`, if committed), and the single-device render at any batch (`forceSync := true` gives
    the one-replica sync graph). -/
structure Cfg where
  slug      : String                       -- artifact slug, e.g. `mobilenetv2`
  net       : VerifiedNet
  bs        : Nat                          -- per-replica batch of the committed DP artifact
  replicas  : Nat := 2
  sgPath    : String                       -- committed single-device step at `bs`; "" renders it
  dpPath    : String                       -- committed sync-BN DP step at `replicas × bs`; "" renders it
  render    : (B : Nat) → (forceSync : Bool) → String   -- single-device train step at batch `B`
  /-- The sync-BN DP step at per-replica batch `B` over `R` replicas, for `dpPath := ""` — a net
      whose committed DP shape is too large to gate (ResNet-50's 1×256 reference peaks at 94–95 %
      of the raised arena) is gated at a smaller `bs` by the same renderer. -/
  renderDp  : (B R : Nat) → String := fun _ _ => ""
  entry     : (B replicas : Nat) → String  -- its entry name, `m.<slug>_<variant>_train_step`
  /-- The split identity's bound on the handed-back statistics. `resnet34-syncbn-check` set 1e-3
      at 36 layers (measured 1.7e-4); a deeper net compounds more reduction-order rounding, and a
      net that needs more says why beside its value. -/
  statsTol  : Float := 1e-3
  /-- Which packed `[θ|m|v]` slot carries the gradient, and its label. Adam and RMSProp take `m`
      (`m = 0` in: `m' = 0.1·g`, resp. `g/√(s'+ε)`); heavy-ball keeps its velocity in `v` and
      passes `m` through untouched, so its gradient column is `v'` with `v = 0` in. -/
  gradSlot  : Nat := 1
  gradLabel : String := "m' = 0.1·g"
  vZero     : Bool := false
  /-- Shard `k`'s pixels are shifted by `shardShift · k`. With iid shards at 224² every replica's
      statistics sit within ~1/√(b·h·w) of the global ones, so CONTROL barely separates from TEST
      and Chan's `(μ_r − μ)²` term is ~1e-6 — a render that dropped it would pass. A shift makes
      the shards' statistics differ at O(1) from the first layer on. 0 keeps the Imagenette gates'
      committed inputs. -/
  shardShift : Float := 0.0
  /-- The DUPLICATED bound on the gradient slot's norm-rel, `‖DP − sync_b‖ / ‖sync_b‖` (the
      `*-dp-check` metric). A sum-not-mean collective reads `R − 1`. Measured 6.8e-4 – 1.0e-3 on
      every f32 gate; a bf16 net that needs more says why beside its value. Norm-rel rather than
      the regions' max-abs, which one squared entry of RMSProp's `v'` moved 4× between runs. -/
  dupTol    : Float := 5e-3
  /-- The FORMULATION bound on the statistics: the one-replica sync graph against the two-pass
      graph, same batch. Exact in f32; a bf16 net whose two graphs round apart says so here. -/
  formTol   : Float := 1e-5

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
  let replicas := cfg.replicas
  let bigB := replicas * bs
  let fr := (1.0 : Float) / replicas.toFloat
  let dpPath := if cfg.dpPath.isEmpty then
      s!".lake/build/{cfg.slug}_syncgate_dp{replicas}x{bs}_train_step.mlir" else cfg.dpPath
  if cfg.dpPath.isEmpty then IO.FS.writeFile dpPath (cfg.renderDp bs replicas)
  IO.println s!"{net.name} SYNC-BN gate — {replicas}×{bs} against 1×{bigB}"
  IO.println s!"  TEST     DP_sync([x₀|…|x{replicas-1}])  ==  single_{bigB}([x₀|…|x{replicas-1}])"
  IO.println s!"  CONTROL  DP_sync([x₀|…|x{replicas-1}])  !=  mean_k single_{bs}(x_k)   (the old, per-replica identity)"
  IO.println s!"  single {bs}: {if cfg.sgPath.isEmpty then "(rendered at run time)" else cfg.sgPath}\n  DP render: {dpPath}{if cfg.dpPath.isEmpty then " (rendered at run time)" else ""}"
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
  let v ← if cfg.vZero then F32.const net.nParams.toUSize 0.0
          else F32.scaleShift (← F32.heInit 8484 net.nParams.toUSize 0.01) 1.0 0.05
  let tail ← F32.const 3 0.0
  let tail ← F32.write3 tail 0 0.001 0.19 0.002
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat #[θ, m, v, tail, bnIn]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  -- R genuinely DIFFERENT shards — different pixels AND different labels. Seeds 555 + 444·k and
  -- label offsets 5·k, so at R = 2 these are the xA / xB every committed reading was taken on.
  let mut xs : Array ByteArray := #[]
  let mut ys : Array ByteArray := #[]
  for k in [0:replicas] do
    let x ← F32.heInit (555 + 444 * k).toUSize (bs * net.d0).toUSize 1.0
    xs := xs.push (← if k == 0 || cfg.shardShift == 0.0 then pure x
                     else F32.scaleShift x 1.0 (cfg.shardShift * k.toFloat))
    ys := ys.push (mkLabels bs (5 * k) net.nClasses)
  let xA := xs[0]!
  let yA := ys[0]!
  let xAll := F32.concat xs
  let yAll := ys.foldl (· ++ ·) ByteArray.empty

  -- the run-time renders: the Rb two-pass step, the one-replica sync graph at Rb and at b, and the
  -- b two-pass step when no committed one is named
  let sgBPath := s!".lake/build/{cfg.slug}_syncgate_single{bigB}_train_step.mlir"
  let syncPath := s!".lake/build/{cfg.slug}_syncgate_sync{bigB}_train_step.mlir"
  let sync1Path := s!".lake/build/{cfg.slug}_syncgate_sync{bs}_train_step.mlir"
  let sg1Path := if cfg.sgPath.isEmpty then
      s!".lake/build/{cfg.slug}_syncgate_single{bs}_train_step.mlir" else cfg.sgPath
  IO.FS.writeFile sgBPath (cfg.render bigB false)
  IO.FS.writeFile syncPath (cfg.render bigB true)
  IO.FS.writeFile sync1Path (cfg.render bs true)
  if cfg.sgPath.isEmpty then IO.FS.writeFile sg1Path (cfg.render bs false)

  IO.println s!"  single-device {bigB} on [x₀|…]… ({sgBPath})"; (← IO.getStdout).flush
  let s2 ← mkSession sgBPath
  let e2 := cfg.entry bigB 1
  let o2 ← LowererSession.mlpTrainStepV s2 e2
    xAll pbuf shapes yAll bigB.toUSize net.d0.toUSize net.nClasses.toUSize
  let o2r ← LowererSession.mlpTrainStepV s2 e2
    xAll pbuf shapes yAll bigB.toUSize net.d0.toUSize net.nClasses.toUSize
  let s1 ← mkSession sg1Path
  let e1 := cfg.entry bs 1
  IO.println s!"  single-device {bs} on each of the {replicas} shards…"; (← IO.getStdout).flush
  let mut oks : Array ByteArray := #[]
  for k in [0:replicas] do
    oks := oks.push (← LowererSession.mlpTrainStepV s1 e1
      xs[k]! pbuf shapes ys[k]! bs.toUSize net.d0.toUSize net.nClasses.toUSize)
  IO.println s!"  single-device SYNC-graph {bigB} on [x₀|…]… ({syncPath})"; (← IO.getStdout).flush
  let sS ← mkSession syncPath
  let oS ← LowererSession.mlpTrainStepV sS e2
    xAll pbuf shapes yAll bigB.toUSize net.d0.toUSize net.nClasses.toUSize
  -- SENSITIVITY: the two-pass graph on the same batch perturbed by 1e-4·N(0,1) per pixel
  let noise ← F32.heInit 7777 (bigB * net.d0).toUSize 1.0
  let xAllp ← F32.axpySlice (F32.concat xs) 0 noise 0 (bigB * net.d0).toUSize 1.0e-4
  IO.println s!"  sensitivity probe (x + 1e-4·noise) on the two-pass {bigB} graph…"; (← IO.getStdout).flush
  let o2p ← LowererSession.mlpTrainStepV s2 e2
    xAllp pbuf shapes yAll bigB.toUSize net.d0.toUSize net.nClasses.toUSize
  -- the sync-BN data-parallel step on the split batch
  IO.println s!"  data-parallel {replicas}×{bs} on [x₀|…]…"; (← IO.getStdout).flush
  let sD ← mkSession dpPath
  let eD := cfg.entry bs replicas
  let oD ← LowererSession.mlpTrainStepVDP sD eD xAll pbuf shapes yAll
             bigB.toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize
  -- DUPLICATED: DP on [x₀|x₀|…] against the one-replica sync graph on x₀ at the same per-replica shape
  IO.println s!"  duplicated probe: DP on [x₀|x₀|…] vs the one-replica sync graph on x₀ ({sync1Path})…"
  (← IO.getStdout).flush
  let sS1 ← mkSession sync1Path
  let oS1 ← LowererSession.mlpTrainStepV sS1 e1
    xA pbuf shapes yA bs.toUSize net.d0.toUSize net.nClasses.toUSize
  let oDD ← LowererSession.mlpTrainStepVDP sD eD
    (F32.concat (Array.replicate replicas xA)) pbuf shapes
    ((Array.replicate replicas yA).foldl (· ++ ·) ByteArray.empty)
    bigB.toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  for (o, what) in [(o2, "single Rb"), (o2r, "single Rb repeat"), (oS, "sync Rb"), (oS1, "sync b"),
                    (oDD, "DP duplicated")]
                   ++ (oks.toList.map (·, "single b")) do
    if o.size != oD.size then
      IO.eprintln s!"SIZE MISMATCH: {what} gives {o.size}, DP gives {oD.size}"; IO.Process.exit 1
  let nP := net.nParams
  let nOut := oD.size / 4
  let statsLabel := s!"bn stats ({2 * net.bnChannels.size} slots)"
  let gLo := cfg.gradSlot * nP
  let slotLabel (k : Nat) (dflt : String) := if k == cfg.gradSlot then cfg.gradLabel else dflt
  -- `[θ' | m' | v' | loss bc1 bc2 | bnstat]`: every region but the report-only loss scalar
  let regions : List (String × Nat × Nat) :=
    [("θ'", 0, nP), (slotLabel 1 "m'", nP, 2*nP), (slotLabel 2 "v'", 2*nP, 3*nP),
     (statsLabel, 3*nP + 3, nOut)]
  let mut worstTest : Float := 0.0
  let mut worstCtrl : Float := 0.0
  let mut worstDup : Float := 0.0
  let mut statsTest : Float := 0.0
  let mut statsForm : Float := 0.0
  let mut statsDup : Float := 0.0
  let mut statsCtrl : Float := 0.0
  let mut mSens : Float := 0.0
  let mut statsRep : Float := 0.0
  let mut nonFinite : Nat := 0
  for (what, lo, hi) in regions do
    for i in [lo:hi] do
      if !(F32.read oD i.toUSize).isFinite || !(F32.read o2 i.toUSize).isFinite then
        nonFinite := nonFinite + 1
    let (eT, mT) := regionErr lo hi oD (fun i => F32.read o2 i.toUSize)
    let (eC, mC) := regionErr lo hi oD
      (fun i => fr * oks.foldl (fun acc o => acc + F32.read o i.toUSize) 0.0)
    let (eP, _)  := regionErr lo hi oD (fun i => F32.read oS i.toUSize)
    let (eF, _)  := regionErr lo hi oS (fun i => F32.read o2 i.toUSize)
    let (eSn, _) := regionErr lo hi o2p (fun i => F32.read o2 i.toUSize)
    let (eDup, mDup) := regionErr lo hi oDD (fun i => F32.read oS1 i.toUSize)
    let (eRp, _) := regionErr lo hi o2r (fun i => F32.read o2 i.toUSize)
    let nrRp := if mT > 1e-30 then eRp / mT else 0.0
    let nrSn := if mT > 1e-30 then eSn / mT else 0.0
    let nrDup := if mDup > 1e-30 then eDup / mDup else 0.0
    let nrT := if mT > 1e-30 then eT / mT else 0.0
    let nrC := if mC > 1e-30 then eC / mC else 0.0
    let nrP := if mT > 1e-30 then eP / mT else 0.0
    let nrF := if mT > 1e-30 then eF / mT else 0.0
    IO.println s!"  ── {what} [{lo}, {hi}) ──"
    IO.println s!"    TEST        |DP − single_Rb|      / max|single_Rb| = {nrT}"
    IO.println s!"    COLLECTIVE  |DP − sync_Rb|        / max|single_Rb| = {nrP}"
    IO.println s!"    FORMULATION |sync_Rb − single_Rb| / max|single_Rb| = {nrF}"
    IO.println s!"    CONTROL     |DP − mean_k single_b| / max|mean|     = {nrC}"
    IO.println s!"    SENSITIVITY two-pass graph, x perturbed by 1e-4     = {nrSn}"
    IO.println s!"    DUPLICATED  |DP([x₀|x₀|…]) − sync_b(x₀)| / max|sync_b| = {nrDup}"
    IO.println s!"    REPEAT      single_Rb vs itself, same input         = {nrRp}"
    if nrDup > worstDup then worstDup := nrDup
    if what == statsLabel then
      statsTest := nrT; statsForm := nrF; statsDup := nrDup; statsCtrl := nrC; statsRep := nrRp
    if lo == gLo then mSens := nrSn
    if nrT > worstTest then worstTest := nrT
    if what != "θ'" && what != "v'" then
      if nrC > worstCtrl then worstCtrl := nrC
  -- the `*-dp-check` metric, for the duplicated probe: ‖DP([x₀|x₀|…]) − sync_b(x₀)‖ / ‖sync_b(x₀)‖
  -- on the gradient slot
  let dupNormRel : Float := Id.run do
    let mut num : Float := 0.0
    let mut den : Float := 0.0
    for i in [gLo:gLo + nP] do
      let a := F32.read oDD i.toUSize
      let b := F32.read oS1 i.toUSize
      num := num + (a - b) * (a - b)
      den := den + b * b
    return (if den > 0.0 then (num / den).sqrt else 0.0)
  IO.println s!"  DUPLICATED {cfg.gradLabel} norm-rel (the *-dp-check metric, against the sync graph) = {dupNormRel}"
  -- `SYNCBN_VERBOSE=1`: the split identity per BN layer, in forward order. Compounded kernel
  -- rounding starts near 1e-7 at the first layer and grows with depth; a wrong statistic exchange
  -- is visible at the layer it enters.
  if (← IO.getEnv "SYNCBN_VERBOSE").isSome then
    IO.println "  ── per-layer BN statistics, TEST rel-err vs single_Rb (mean | var) ──"
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
  if statsForm > cfg.formTol then
    IO.eprintln s!"SYNC-BN CHECK FAILED (formulation): the one-replica sync graph's statistics \
differ from the two-pass graph's by {statsForm} > {cfg.formTol}."
    IO.Process.exit 1
  if statsDup > 1e-5 || dupNormRel > cfg.dupTol then
    IO.eprintln s!"SYNC-BN CHECK FAILED (collective): DP_sync([x₀|x₀|…]) does not reproduce the \
one-replica sync graph on x₀ — statistics {statsDup}, {cfg.gradLabel} norm-rel {dupNormRel} \
(worst region max-abs {worstDup})."
    IO.Process.exit 1
  if statsCtrl < 2e-3 then
    IO.eprintln s!"VACUOUS: DP still reproduces mean_k single_b(x_k) to {statsCtrl} on \
the statistics — the BatchNorm statistics are NOT synchronised."
    IO.Process.exit 1
  if statsTest > cfg.statsTol then
    IO.eprintln s!"SYNC-BN CHECK FAILED (split): DP_sync([x₀|…]) statistics differ from \
single_Rb([x₀|…]) by {statsTest} > {cfg.statsTol} — beyond what kernel rounding at two batch shapes explains."
    IO.Process.exit 1
  -- (5) the split identity at the FIRST BatchNorm layer, where nothing has compounded yet: its
  --     statistics depend only on the input and the stem conv, so a b-row and an Rb-row
  --     reduction may differ only in their last bits. A wrong exchange (Chan's correction, the
  --     packing, a collective's divisor) is visible here at full size.
  let c0 := net.bnChannels[0]!
  let (e0m, m0m) := regionErr (3*nP + 3) (3*nP + 3 + c0) oD (fun i => F32.read o2 i.toUSize)
  let (e0v, m0v) := regionErr (3*nP + 3 + c0) (3*nP + 3 + 2*c0) oD (fun i => F32.read o2 i.toUSize)
  let first := max (if m0m > 1e-30 then e0m / m0m else 0.0) (if m0v > 1e-30 then e0v / m0v else 0.0)
  if first > 1e-5 then
    IO.eprintln s!"SYNC-BN CHECK FAILED (first layer): DP_sync([x₀|…])'s first BN layer's statistics \
differ from single_Rb's by {first} > 1e-5 — before any depth has compounded, so the exchange itself."
    IO.Process.exit 1
  IO.println s!"✓ SYNC-BN CONFIRMED for {cfg.slug}: the sync graph IS the two-pass graph \
(statistics {statsForm}), the collective composes it exactly (duplicated batch {statsDup}), and \
DP_sync([x₀|…]) = single_Rb([x₀|…]) to {statsTest} on the statistics (first layer {first}) against a \
per-replica CONTROL of {statsCtrl} — {replicas}×{bs} IS 1×{bigB}. Worst TEST region {worstTest}; \
{cfg.gradLabel} SENSITIVITY-to-1e-4 {mSens}; statistics REPEAT {statsRep}."

end SyncBnCheck
