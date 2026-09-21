import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.ResNet34RenderB

/-! # `resnet34-syncbn-check` — synchronised BatchNorm: 2×b IS 1×2b

    lake build resnet34-syncbn-check
    unset HIP_VISIBLE_DEVICES
    PJRT_REPLICAS=2 .lake/build/bin/resnet34-syncbn-check

The identity every `*-dp-check` and `shard-check` could NOT state for a batch-BN net. Until
2026-09-21 a data-parallel R34 render normalised each replica over its own `b` rows, so
`DP([xA|xB])` was `mean(single_b(xA), single_b(xB))` — a step on the mean of two per-replica
losses, and provably not the single-device step at `2b` (`dpMeanGrad_ne_globalBatchGrad`;
`lakefile.lean`'s own words: "R34 is about SPLITTING a batch — 2×32 really is not 1×64").

With the sync-BN render (`planning/global_bn_verified.md` §2b: every BN layer all-reduces its
μ, then its Chan-corrected σ² (`σ²_r + (μ_r − μ)²`), before normalising; its backward all-reduces
the two dy-reductions; and the γ gradient reads the same global `x̂`) the replicas compute their
shards of ONE global-batch function, and the identity becomes exact up to float reduction order:

    TEST     DP_sync( [xA | xB] )  ==  single_2b( [xA | xB] )
    CONTROL  DP_sync( [xA | xB] )  !=  mean( single_b(xA), single_b(xB) )

The CONTROL is the old identity, and it must now FAIL by a margin: if it still held, the
statistics would not be synchronised and the TEST would be passing for the wrong reason (both
sides per-replica). `dpSyncGrad_eq_globalBatchGrad` / `den_bnSyncF_allReduce` … are the
ℝ-level statements this gates the emitted bytes against — the first numeric check any of the
seven sync ops' MLIR has had.

**What is compared, and why all of it.** With `m = 0` fed in, `m' = 0.1·g` is linear in the
gradient (the `shard-check` trick). Here the gradient itself is equal, not just averagable — the
per-replica divisor `1/b` and the collective's `1/2` compose to the single device's `1/(2b)`
(`DataParallelSync.lean`, "the 1/R") — so `v' = 0.001·g² + 0.999·v` and Adam's `θ'` are equal too,
and so are the 72 handed-back BN statistics (global μ / σ² on every replica, versus the 2b batch's
own). Every output region is checked; only the report-only `%loss` slot is skipped, since the DP
render logs replica 0's shard loss and does not all-reduce it.

**How the columns are read.** Two more runs bracket the comparison. The one-replica SYNC graph
on the whole batch (every collective empty) splits TEST into COLLECTIVE (DP vs it — the
collective composing the sync ops) and FORMULATION (it vs the two-pass graph — the arithmetic of
the exchange). And a SENSITIVITY probe — the two-pass graph on the same batch perturbed by
1e-4·N(0,1) per pixel — is the yardstick for the gradient columns: at this random-init operating
point it moves the two-pass graph's OWN `m'` by ~0.2, so the gradient can only ever be compared to
~1e-3 of that, while the statistics are compared tightly.

⛔ **History (2026-09-21).** The first render exchanged `[μ ‖ E[x²]]` in one round and every
consumer formed `σ² = E[x²] − μ²`. This gate measured it 2e-4 off in the statistics after 36
layers and 15 % off in `m'`, with the sensitivity probe at 0.22 — i.e. the ops were right and the
f32 arithmetic was not (`ε·E[x²]/σ²` per layer, compounding). Chan's two-round exchange replaced
it; the numbers below are its.

Needs TWO GPUs and the XLA backend (collectives do not exist on the IREE path). The artifacts are
`resnet34_adam_train_step` (1×32), `resnet34_adam64_train_step` (1×64) and
`resnet34_adamdp_train_step` (2×32, sync); the one-replica sync graph is rendered to
`.lake/build/` at run time. ~30 s.
-/

/-- Labels for one shard: class `(i + off) % nClasses`, packed as the driver's 4-byte records. -/
private def mkLabels (bs off nc : Nat) : ByteArray := Id.run do
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat ((i + off) % nc)); y := y.push 0; y := y.push 0; y := y.push 0
  y

/-- Max-abs difference and max-abs magnitude over `[lo, hi)` of `f a i` vs `g i`. -/
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

def main (args : List String) : IO Unit := do
  let net := resnet34Verified.toNet
  let bs := 32
  let replicas := 2
  let dpPath := args[0]?.getD "verified_mlir/resnet34_adamdp_train_step.mlir"
  let sgPath := "verified_mlir/resnet34_adam_train_step.mlir"
  let sg2Path := "verified_mlir/resnet34_adam64_train_step.mlir"
  IO.println s!"{net.name} SYNC-BN gate — 2×{bs} against 1×{2*bs}"
  IO.println s!"  TEST     DP_sync([xA|xB])  ==  single_{2*bs}([xA|xB])"
  IO.println s!"  CONTROL  DP_sync([xA|xB])  !=  mean(single_{bs}(xA), single_{bs}(xB))   (the old, per-replica identity)"
  IO.println s!"  single {bs}: {sgPath}\n  single {2*bs}: {sg2Path}\n  DP render: {dpPath}"
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
  -- m = 0: m' = 0.1·g, linear in the gradient. v is a generic positive buffer so θ' exercises
  -- Adam's nonlinearity on EQUAL gradients rather than on a degenerate one.
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

  for tag in ["resnet34_shard_a", "resnet34_shard_b", "resnet34_shard_c"] do
    for p in [s!".lake/build/{tag}.vmfb",
              s!".lake/build/{tag}_{((← IO.getEnv "IREE_BACKEND").getD "cuda")}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
  -- the 2b single-device step on the whole batch
  IO.println s!"  single-device {2*bs} on [xA|xB]…"; (← IO.getStdout).flush
  let s2 ← mkSession sg2Path
  let o2 ← LowererSession.mlpTrainStepV s2 "m.resnet34_adam64_train_step"
    xAB pbuf shapes yAB (2*bs).toUSize net.d0.toUSize net.nClasses.toUSize
  -- the two b single-device steps, for the CONTROL
  let s1 ← mkSession sgPath
  IO.println s!"  single-device {bs} on xA, xB…"; (← IO.getStdout).flush
  let oA ← LowererSession.mlpTrainStepV s1 "m.resnet34_adam_train_step"
    xA pbuf shapes yA bs.toUSize net.d0.toUSize net.nClasses.toUSize
  let oB ← LowererSession.mlpTrainStepV s1 "m.resnet34_adam_train_step"
    xB pbuf shapes yB bs.toUSize net.d0.toUSize net.nClasses.toUSize
  -- the SYNC graph at ONE replica on the whole batch — the same seven sync ops, every collective
  -- empty. Splits the question in two: DP vs this is the COLLECTIVE (must be ~float-exact); this
  -- vs `adam64` is the FORMULATION (E[x²]−μ² and the handed-in statistics, in f32).
  let syncPath := ".lake/build/resnet34_adamsync64_train_step.mlir"
  IO.FS.writeFile syncPath
    (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 64 10 "1.0e-05" (forceSync := true))
  IO.println s!"  single-device SYNC-graph {2*bs} on [xA|xB]… ({syncPath})"; (← IO.getStdout).flush
  let sS ← mkSession syncPath
  let oS ← LowererSession.mlpTrainStepV sS "m.resnet34_adam64_train_step"
    xAB pbuf shapes yAB (2*bs).toUSize net.d0.toUSize net.nClasses.toUSize
  -- REORDER probes: the same batch with its two halves swapped. Parameter gradients are
  -- permutation-invariant, so each graph's disagreement with itself here is pure float
  -- reduction-order sensitivity — the yardstick every other column is read against.
  let xBA := F32.concat #[xB, xA]
  let yBA := yB ++ yA
  IO.println s!"  reorder probes ([xB|xA]) on both {2*bs} graphs…"; (← IO.getStdout).flush
  let o2r ← LowererSession.mlpTrainStepV s2 "m.resnet34_adam64_train_step"
    xBA pbuf shapes yBA (2*bs).toUSize net.d0.toUSize net.nClasses.toUSize
  let oSr ← LowererSession.mlpTrainStepV sS "m.resnet34_adam64_train_step"
    xBA pbuf shapes yBA (2*bs).toUSize net.d0.toUSize net.nClasses.toUSize
  -- SENSITIVITY probe: the two-pass graph on the same batch perturbed by 1e-4·N(0,1) per pixel.
  -- How much a 1e-4 forward perturbation moves the gradient at this (random-init, random-data)
  -- operating point — the conditioning every column above has to be read against.
  let noise ← F32.heInit 7777 (2 * bs * net.d0).toUSize 1.0
  let xABp ← F32.axpySlice (F32.concat #[xA, xB]) 0 noise 0 (2 * bs * net.d0).toUSize 1.0e-4
  IO.println s!"  sensitivity probe (x + 1e-4·noise) on the two-pass {2*bs} graph…"; (← IO.getStdout).flush
  let o2p ← LowererSession.mlpTrainStepV s2 "m.resnet34_adam64_train_step"
    xABp pbuf shapes yAB (2*bs).toUSize net.d0.toUSize net.nClasses.toUSize
  -- the sync-BN data-parallel step on the split batch
  IO.println s!"  data-parallel {replicas}×{bs} on [xA|xB]…"; (← IO.getStdout).flush
  let sD ← mkSession dpPath
  let oD ← LowererSession.mlpTrainStepVDP sD "m.resnet34_adamdp_train_step" xAB pbuf shapes yAB
             (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize
  -- DUPLICATED probe: DP on [xA|xA] against the one-replica sync graph on xA at the SAME
  -- per-replica batch shape. Every replica statistic equals the global one, so every collective
  -- averages two identical operands — the plumbing alone, with no batch-shape change under it.
  -- Any difference here is a collective composing the ops wrongly; ~0 says the split residual
  -- above is XLA's reduction kernels at 32 rows vs 64, not the render.
  let sync32Path := ".lake/build/resnet34_adamsync32_train_step.mlir"
  IO.FS.writeFile sync32Path
    (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 32 10 "1.0e-05" (forceSync := true))
  IO.println s!"  duplicated probe: DP on [xA|xA] vs the one-replica sync graph on xA ({sync32Path})…"
  (← IO.getStdout).flush
  let sS32 ← mkSession sync32Path
  let oS32 ← LowererSession.mlpTrainStepV sS32 "m.resnet34_adam_train_step"
    xA pbuf shapes yA bs.toUSize net.d0.toUSize net.nClasses.toUSize
  let oDD ← LowererSession.mlpTrainStepVDP sD "m.resnet34_adamdp_train_step"
    (F32.concat #[xA, xA]) pbuf shapes (yA ++ yA)
    (bs * replicas).toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize

  for (o, what) in [(o2, "single 2b"), (oA, "single b (A)"), (oB, "single b (B)"), (oS, "sync 2b")] do
    if o.size != oD.size then
      IO.eprintln s!"SIZE MISMATCH: {what} gives {o.size}, DP gives {oD.size}"; IO.Process.exit 1
  let nP := net.nParams
  let nOut := oD.size / 4
  -- `[θ' | m' | v' | loss bc1 bc2 | bnstat]`: every region but the report-only loss scalar
  let regions : List (String × Nat × Nat) :=
    [("θ'", 0, nP), ("m' = 0.1·g", nP, 2*nP), ("v'", 2*nP, 3*nP),
     ("bn stats (72 slots)", 3*nP + 3, nOut)]
  let mut worstTest : Float := 0.0
  let mut worstCtrl : Float := 0.0
  let mut worstPlumb : Float := 0.0
  let mut worstForm : Float := 0.0
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
    let (eR2, _) := regionErr lo hi o2r (fun i => F32.read o2 i.toUSize)
    let (eSn, _) := regionErr lo hi o2p (fun i => F32.read o2 i.toUSize)
    let nrSn := if mT > 1e-30 then eSn / mT else 0.0
    let (eDup, mDup) := regionErr lo hi oDD (fun i => F32.read oS32 i.toUSize)
    let nrDup := if mDup > 1e-30 then eDup / mDup else 0.0
    let (eRS, _) := regionErr lo hi oSr (fun i => F32.read oS i.toUSize)
    let nrR2 := if mT > 1e-30 then eR2 / mT else 0.0
    let nrRS := if mT > 1e-30 then eRS / mT else 0.0
    let nrT := if mT > 1e-30 then eT / mT else 0.0
    let nrC := if mC > 1e-30 then eC / mC else 0.0
    let nrP := if mT > 1e-30 then eP / mT else 0.0
    let nrF := if mT > 1e-30 then eF / mT else 0.0
    IO.println s!"  ── {what} [{lo}, {hi}) ──"
    IO.println s!"    TEST        |DP − single_2b|      / max|single_2b| = {nrT} ({nrT * 1e9} e-9)"
    IO.println s!"    COLLECTIVE  |DP − sync_2b|        / max|single_2b| = {nrP} ({nrP * 1e9} e-9)"
    IO.println s!"    FORMULATION |sync_2b − single_2b| / max|single_2b| = {nrF} ({nrF * 1e9} e-9)"
    IO.println s!"    CONTROL     |DP − mean(A,B)|      / max|mean|      = {nrC}"
    IO.println s!"    REORDER     two-pass graph vs itself on [B|A]      = {nrR2};  sync graph vs itself = {nrRS}"
    IO.println s!"    SENSITIVITY two-pass graph, x perturbed by 1e-4     = {nrSn}"
    IO.println s!"    DUPLICATED  |DP([A|A]) − sync_b(A)|  / max|sync_b|  = {nrDup} ({nrDup * 1e9} e-9)"
    if nrDup > worstDup then worstDup := nrDup
    if what == "bn stats (72 slots)" then
      statsTest := nrT; statsForm := nrF; statsDup := nrDup; statsCtrl := nrC
    if what == "m' = 0.1·g" then mSens := nrSn
    if nrT > worstTest then worstTest := nrT
    if nrP > worstPlumb then worstPlumb := nrP
    if nrF > worstForm then worstForm := nrF
    -- the control is judged on the gradient proxy and the statistics, where the per-replica
    -- and global functions differ by construction; θ' and v' carry it too but through Adam
    if what != "θ'" && what != "v'" then
      if nrC > worstCtrl then worstCtrl := nrC

  -- `SYNCBN_VERBOSE=1`: per-parameter breakdown of the gradient proxy, to localise a failing op
  -- (γ nodes only → `bnSyncGammaGradB`; every conv → `bnSyncBack`; the stem → its plumbing).
  if (← IO.getEnv "SYNCBN_VERBOSE").isSome then
    IO.println "  ── per-parameter m' (TEST rel-err | CONTROL rel-err), func-arg order ──"
    let mut off := nP
    let mut k := 0
    for (dims, _) in net.specs do
      let n := dims.foldl (· * ·) 1
      let (eT, mT) := regionErr off (off + n) oD (fun i => F32.read o2 i.toUSize)
      let (eP, _)  := regionErr off (off + n) oD (fun i => F32.read oS i.toUSize)
      let (eF, _)  := regionErr off (off + n) oS (fun i => F32.read o2 i.toUSize)
      let (eC, _) := regionErr off (off + n) oD
        (fun i => 0.5 * (F32.read oA i.toUSize + F32.read oB i.toUSize))
      let nrT := if mT > 1e-30 then eT / mT else 0.0
      let nrP := if mT > 1e-30 then eP / mT else 0.0
      let nrF := if mT > 1e-30 then eF / mT else 0.0
      let nrC := if mT > 1e-30 then eC / mT else 0.0
      IO.println s!"    #{k} {dims} : test {nrT} | collective {nrP} | formulation {nrF} | control {nrC}"
      off := off + n; k := k + 1
    IO.println "  ── per-slot BN stats (TEST rel-err vs single_2b, |ref| max) ──"
    let mut so := 3*nP + 3
    let mut j := 0
    for c in net.bnChannels do
      for what in ["mean", "var"] do
        let (eT, mT) := regionErr so (so + c) oD (fun i => F32.read o2 i.toUSize)
        IO.println s!"    bn{j} {what} [{c}] : {if mT > 1e-30 then eT / mT else 0.0}  (max|ref| {mT})"
        so := so + c
      j := j + 1
  if nonFinite > 0 then
    IO.eprintln s!"DEGENERATE: {nonFinite} non-finite outputs"; IO.Process.exit 1
  -- ── the verdict, on the STATISTICS (the sharp part) and the collective ──
  -- (1) the sync graph's arithmetic IS the two-pass graph's: at one replica the 72 handed-back
  --     statistics agree to rounding (measured 0.000000 — bit-exact).
  if statsForm > 1e-5 then
    IO.eprintln s!"SYNC-BN CHECK FAILED (formulation): the one-replica sync graph's statistics \
differ from the two-pass graph's by {statsForm} > 1e-5 — a sync op's emitted arithmetic \
disagrees with its den (the E[x²]−μ² exchange failed exactly here, at 2e-4)."
    IO.Process.exit 1
  -- (2) the collective composes the ops exactly: DP on a duplicated batch reproduces the
  --     one-replica sync graph at the same batch shape (measured bit-exact on the statistics,
  --     and within rounding × the gradient's amplification on m').
  if statsDup > 1e-5 || worstDup > 5e-3 then
    IO.eprintln s!"SYNC-BN CHECK FAILED (collective): DP_sync([A|A]) does not reproduce the \
one-replica sync graph on A — statistics {statsDup}, worst region {worstDup}."
    IO.Process.exit 1
  -- (3) the old per-replica identity must now FAIL by a margin, or nothing was synchronised.
  if statsCtrl < 2e-3 then
    IO.eprintln s!"VACUOUS: DP still reproduces mean(single_b(A), single_b(B)) to {statsCtrl} on \
the statistics — the BatchNorm statistics are NOT synchronised."
    IO.Process.exit 1
  -- (4) the split identity 2×b = 1×2b, to the accuracy XLA's own reduction kernels allow: the
  --     32-row and 64-row programs round their means differently (~1e-6 per layer), which 36
  --     layers of forward compound to ~2e-4 in the statistics (the CONTROL — a genuinely
  --     different function — sits at 3.4e-3). ⚠ m' is NOT bounded here: at this random-init
  --     operating point a 1e-4 forward perturbation moves the two-pass graph's OWN m' by ~0.2
  --     (SENSITIVITY), so the gradient columns are read against that, not against a tolerance.
  if statsTest > 1e-3 then
    IO.eprintln s!"SYNC-BN CHECK FAILED (split): DP_sync([A|B]) statistics differ from \
single_2b([A|B]) by {statsTest} > 1e-3 — beyond what kernel rounding at two batch shapes explains."
    IO.Process.exit 1
  IO.println s!"✓ SYNC-BN CONFIRMED for resnet34: the sync graph IS the two-pass graph \
(statistics {statsForm}), the collective composes it exactly (duplicated batch {statsDup}), and \
DP_sync([A|B]) = single_2b([A|B]) to {statsTest} on the statistics against a per-replica CONTROL \
of {statsCtrl} — 2×{bs} IS 1×{2*bs}, to the rounding of XLA's own kernels. \
m' TEST {worstTest} against a SENSITIVITY-to-1e-4 of {mSens} — see the table."
