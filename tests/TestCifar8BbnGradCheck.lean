module

public import LeanMlir.Verified.NetsCore
public import LeanMlir.Verified.Train

@[expose] public section

/-! # The batch-statistics cifar8 BN step's GRADIENT, gated — on the committed artifact

`cifar8wb_bbn_*_train_step.mlir` (`Proofs/Codegen/CnnBbnRender.lean`) is the eight-conv CIFAR net
with every BatchNorm reducing over `[0,2,3]` — the first cifar8 artifact whose BN couples the
batch, and the first with a running-statistics protocol. Its forward is `pretty` of nodes the
proofs tie (`bnBatchF` ↔ `bnBatchTensor4`); its backward is `pretty` of `bnBatchBack` /
`bnGammaGradB` / `bnBetaGradB`, whose per-op `den` theorems exist, and the whole-net step tie at
these nodes is `Cifar8BbnStepTieB.cifar8Bbn_train_step_tiedB`. This file gates the backward
numerically on the artifact beside that tie — the pattern of
`tests/TestR50GradCheck.lean`, at a net small enough to run on a CPU in seconds.

It needs no second framework: the AdamW step returns its own loss beside its own first moment,

    out = [θ' | m' | v' | loss, bc₁, bc₂ | 16 BN batch stats]

so ONE invoke from `m = v = 0` yields `L(θ)` and `m' = (1−β₁)·g = 0.1·g`. AdamW's decay is
decoupled, so `m` sees the raw gradient.

## TIER 1 — a closed-form identity, no finite differences

Every conv here is BN-followed, so the loss is 0-homogeneous in each conv's `(W, b)` **pair**:
scaling both by `s` scales the conv output by `s` and the BatchNorm divides it straight back out
(batch statistics or per-image, the same way). Hence `⟨g_W, W⟩ + ⟨g_b, b⟩ = 0` at each of the 8
convs, up to `O(ε/var)` from BN's ε. The dense head is the CONTROL: `W9`, `Wa`, `Wb` have no BN
after them, so the same cosine must be far from zero there.

(The BN-affine identity `⟨g_γ,γ⟩ + ⟨g_β,β⟩ = 0` that R50 also checks does NOT hold here: the op
after each BN is relu → [pool →] conv **with a bias**, and the bias breaks the homogeneity. It is
deliberately not stated rather than stated at a bias-free base point, which would make tier 1 A
degenerate in `b`.)

## TIER 2 — the adjoint probe, which pins the scale

Per parameter group (each conv+BN layer, the head, and the whole net):

    ⟨g, δ⟩  ≟  (L(θ+δ) − L(θ−δ)) / 2,     δ = α·m' restricted to the group, ‖δ‖ = h·‖θ_group‖

`δ ∝ g` rather than random, for R50's reason: the inner product is then `10α‖m'‖²`, maximal and
never degenerate, and a wrong backward gives `⟨g_true, g_wrong⟩` against a prediction of
`‖g_wrong‖²`, which agree only by coincidence. The scale control re-checks each finite difference
against `2·⟨g,δ⟩` and must miss. The fp32 noise floor is read along the tier-1 invariance
direction of layer 1 at the same ‖δ‖ (an exact invariance, so the difference there is pure
arithmetic noise).

    lake build cifar8-bbn-gradcheck
    LEAN_MLIR_LOWERER=iree IREE_BACKEND=vmvx .lake/build/bin/cifar8-bbn-gradcheck
    C8BBN_GC_SLUG=cifar8wb_bn .lake/build/bin/cifar8-bbn-gradcheck   # the per-image sibling, as a control
    C8BBN_GC_SCAN=1 .lake/build/bin/cifar8-bbn-gradcheck             # the step-size curve
    C8BBN_GC_TENSORS=1 .lake/build/bin/cifar8-bbn-gradcheck          # the probe per tensor
    C8BBN_GC_DUMP=/tmp/d/bbn .lake/build/bin/cifar8-bbn-gradcheck    # + tests/cifar8_bbn_jax_ref.py

## ⚠ Under IREE 3.11 run it on `IREE_BACKEND=vmvx` (or XLA), NOT llvm-cpu — the lowerer, not the render

(On IREE 3.13.0's llvm-cpu — `deploy/iree_cpu_setup.sh`'s stack — both this artifact and the
per-image control PASS, 2026-10-28, worst rel 0.0051 / 0.0067, tier-1 |cos∠| ≤ 1e-6, 17 s each on
88 cores. What follows is the 3.11 finding, which is why the vmvx route exists.)

Found 2026-10-19, the first time this gate ran. On IREE 3.11's **llvm-cpu** target this artifact's
gradient is wrong below the stage-3 pool — layers 1-6 at cosine 0.4-0.8 against JAX, layers 7-8
and the head exact — and the per-image `cifar8wb_bn_adam_train_step.mlir` (the chapter's
committed artifact) shows the same pattern. It is NOT the text: the same artifact compiled for
IREE's reference `vmvx` backend gives cosine 1.0000 in every layer; the forward matches JAX on
either backend; tapping extra outputs out of the same llvm-cpu compile CHANGES which layers come
out right; and `-O0`, no-fusion and no-vectorisation builds all agree with each other and are all
wrong. The compile warns `reads uninitialized values from an operand produced by a tensor.empty
op` at every `select_and_scatter` (the `maxPoolBackB` emit): the running-max accumulator of its
lowering is uninitialised, and in a program with buffer reuse that is what it reads. The whole
investigation is reproducible from a `C8BBN_GC_DUMP` with `tests/cifar8_bbn_jax_ref.py`. The GPU
trainers use XLA and are unaffected; the CI gradchecks on llvm-cpu are not gates on any pooled
net's backward until IREE is fixed or they move to vmvx.

Knobs (micro-units): `C8BBN_GC_EPS_U` (step, default 60 = 6e-5), `C8BBN_GC_TOL_U` (tier-2 rel
tolerance, default 20000 = 0.02), `C8BBN_GC_EXACT_U` (tier-1 |cos∠| tolerance, default 300 =
3e-4), `C8BBN_GC_VARIANT` (default `adam`; `bf16adam` reaches the bf16 twin), `C8BBN_GC_SLUG`
(default `cifar8wb_bbn`; `cifar8wb_bn` is the per-image net, same 38 parameters).

The tier-2 tolerance (0.02) was set from the vmvx run of 2026-10-19; the planning note
`planning/cifar-batch-bn-rerender.md` carries that run's table. -/

/-- The driver's init shape (He fan-out for conv, Glorot for dense, γ = 1) with ONE departure:
    kind 2 (conv bias, β, dense bias) is small noise rather than 0, so the base point is
    non-degenerate for tier 1 — at `b = 0` the `⟨g_b, b⟩` term would vanish whatever the render
    computes. -/
def mkParamNoisyBias (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.heInit (seed + 90000).toUSize n.toUSize 0.05
  | _ =>
    let variance :=
      if dims.size == 4 then 2.0 / (dims[0]! * dims[2]! * dims[3]!).toFloat
      else if dims.size == 2 then 2.0 / (dims[0]! + dims[1]!).toFloat
      else 2.0 / (dims[0]!).toFloat
    F32.heInit seed.toUSize n.toUSize (Float.sqrt variance)

/-- A contiguous run of parameter TENSORS, in func-arg order. -/
structure Grp where
  label : String
  form  : String
  t0    : Nat
  nT    : Nat
deriving Inhabited

/-- The 9 groups: 8 × (W, b, γ, β) then the 6-tensor dense head. 38 tensors. -/
def c8Groups : Array Grp := Id.run do
  let mut gs : Array Grp := #[]
  let forms := #["conv 3→16 @32²", "conv 16→16 @32²", "conv 16→16 @16²", "conv 16→16 @16²",
                 "conv 16→32 @8²", "conv 32→32 @8²", "conv 32→32 @4²", "conv 32→32 @4²"]
  for i in [0:8] do
    gs := gs.push ⟨s!"L{i+1}", s!"{forms[i]!} + BN + relu", 4 * i, 4⟩
  return gs.push ⟨"head", "dense 128→512→512→10", 32, 6⟩

def main : IO Unit := do
  let slug := (← IO.getEnv "C8BBN_GC_SLUG").getD "cifar8wb_bbn"
  let net ← match slug with
    | "cifar8wb_bbn" => pure cifar8wbBbnVerified.toNet
    | "cifar8wb_bn"  => pure cifar8wbBnVerified.toNet
    | s => throw <| IO.userError s!"C8BBN_GC_SLUG={s}: want cifar8wb_bbn or cifar8wb_bn"
  let bs := 128
  let nP := net.nParams
  let nT := net.specs.size
  let variant := (← IO.getEnv "C8BBN_GC_VARIANT").getD "adam"
  let artifact := s!"verified_mlir/{net.slug}_{variant}_train_step.mlir"
  if !(variant.endsWith "adam") then
    throw <| IO.userError s!"C8BBN_GC_VARIANT={variant}: the gradient is recovered from m', so \
only the AdamW renders (adam, bf16adam) can be gated here"
  let gScale := 10.0                      -- g = 10·m'  (m = 0, 1−β₁ = 0.1)
  let lossOff := 3 * nP
  let uEnv (k : String) (dflt : Float) : IO Float := do
    pure (((← IO.getEnv k).bind (·.toNat?)).map (fun u => u.toFloat * 1e-6) |>.getD dflt)
  let h        ← uEnv "C8BBN_GC_EPS_U"   6.0e-5
  let tol      ← uEnv "C8BBN_GC_TOL_U"   0.02
  let tolExact ← uEnv "C8BBN_GC_EXACT_U" 3.0e-4
  let scan := (← IO.getEnv "C8BBN_GC_SCAN").isSome
  let lr := 0.0
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]

  IO.println s!"GRADCHECK — @{net.slug}_{variant}_train_step on the committed artifact"
  IO.println s!"  artifact {artifact}"
  IO.println s!"  {nT} params ({nP} floats), bs {bs}, {net.bnChannels.size} running-stat BN layers \
({nBnStats} stat floats), backend {← LowererSession.backendName}"
  IO.println s!"  tier 1 |cos∠| ≤ {tolExact}   ·   tier 2 h = {h}, rel ≤ {tol}"

  let mut offs : Array Nat := #[0]
  for sh in net.paramShapes do offs := offs.push (offs.back! + sh.foldl (· * ·) 1)
  if offs.back! != nP then
    throw <| IO.userError s!"offset table totals {offs.back!}, net.nParams is {nP}"
  if c8Groups.foldl (fun a g => a + g.nT) 0 != nT then
    throw <| IO.userError s!"the group table covers {c8Groups.foldl (fun a g => a + g.nT) 0} \
tensors, the net has {nT}"

  -- ── one (θ, x, y) everything below sees ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParamNoisyBias sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  let z ← F32.const nP.toUSize 0.0
  -- The passthrough stat inputs: the graph never reads them, but the slots must be filled.
  let bnIn ← if nBnStats == 0 then F32.const 0 0.0
             else F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  let sess ← mkSession artifact
  let fn := s!"m.{net.slug}_{variant}_train_step"
  let tl ← F32.write3 (← F32.const 3 0.0) 0 lr 0.1 0.001
  let mut nInvokes := 0
  let runAt (θx : ByteArray) : IO ByteArray :=
    LowererSession.mlpTrainStepV sess fn x
      (F32.concat (if nBnStats == 0 then #[θx, z, z, tl] else #[θx, z, z, tl, bnIn])) shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize

  -- ── the base point: L(θ) and g = 10·m' ──
  let out0 ← runAt θ
  nInvokes := nInvokes + 1
  let l0 := F32.read out0 lossOff.toUSize
  let gNorm2 := gScale * gScale * F32.dotSlice out0 nP.toUSize out0 nP.toUSize nP.toUSize
  IO.println s!"  base loss {l0}   ‖g‖ {Float.sqrt gNorm2}   (ln 10 = 2.302585)"
  if !l0.isFinite || l0 < 1e-3 then
    throw <| IO.userError s!"DEGENERATE: base loss {l0} — nothing is being differentiated"
  if !gNorm2.isFinite || gNorm2 < 1e-12 then
    throw <| IO.userError s!"DEGENERATE: ‖g‖² = {gNorm2} — the recovered gradient is ~0"
  -- The returned batch statistics must be finite and the variances positive: a stat slot that
  -- read the wrong tensor would still be a plausible number, but a NaN or a negative variance
  -- is caught here.
  if nBnStats > 0 then
    let statOff := 3 * nP + 3
    let mut k := 0
    for c in net.bnChannels do
      for j in [0:c] do
        let mu := F32.read out0 (statOff + k + j).toUSize
        let va := F32.read out0 (statOff + k + c + j).toUSize
        if !mu.isFinite || !va.isFinite || va < 0.0 then
          throw <| IO.userError s!"BN STAT slot {k + j}: μ = {mu}, var = {va} — not a statistic"
      k := k + 2 * c
    IO.println s!"  {nBnStats} returned batch statistics finite, variances ≥ 0"

  -- ════════════════════════════════════════════════════════════════
  -- § TIER 1 — the (W, b) homogeneity identity at the 8 BN-followed convs; the head as control
  -- ════════════════════════════════════════════════════════════════
  let cosine (ts : Array Nat) : Float := Id.run do
    let mut s := 0.0; let mut gn2 := 0.0; let mut pn2 := 0.0
    for t in ts do
      let (fo, fl) := (offs[t]!, offs[t+1]! - offs[t]!)
      s   := s   + gScale * F32.dotSlice out0 (nP + fo).toUSize θ    fo.toUSize        fl.toUSize
      gn2 := gn2 + gScale * gScale * F32.dotSlice out0 (nP + fo).toUSize out0 (nP + fo).toUSize fl.toUSize
      pn2 := pn2 +         F32.dotSlice θ    fo.toUSize        θ    fo.toUSize        fl.toUSize
    return s.abs / max (Float.sqrt gn2 * Float.sqrt pn2) 1e-30
  let mut idA : Array (Array Nat × String) := #[]
  for i in [0:8] do idA := idA.push (#[4 * i, 4 * i + 1], s!"L{i+1}.(W,b)")
  let ctl : Array (Array Nat × String) := #[(#[32], "head.W9"), (#[34], "head.Wa"), (#[36], "head.Wb")]
  let mut wA := 0.0; let mut atA := ""
  for (ts, nm) in idA do
    let c := cosine ts
    if c > wA then wA := c; atA := nm
  let mut bC := 1e30; let mut atC := ""
  for (ts, nm) in ctl do
    let c := cosine ts
    if c < bC then bC := c; atC := nm
  IO.println ""
  IO.println s!"  TIER 1  A  ⟨g_W,W⟩+⟨g_b,b⟩ = 0   8 BN-followed convs   worst |cos∠| {wA} at {atA}"
  IO.println s!"          ⟂  control: 3 head weights with no BN after them, weakest violation {bC} at {atC}"
  for (ts, nm) in idA ++ ctl do
    IO.println s!"             {nm}\t{cosine ts}"

  -- ════════════════════════════════════════════════════════════════
  -- § TIER 2 — the adjoint probe, per group
  -- ════════════════════════════════════════════════════════════════
  let probe (fo fl : Nat) (hh : Float) : IO (Float × Float) := do
    let tn2 := F32.dotSlice θ fo.toUSize θ fo.toUSize fl.toUSize
    let mn2 := F32.dotSlice out0 (nP + fo).toUSize out0 (nP + fo).toUSize fl.toUSize
    if mn2 < 1e-30 || tn2 < 1e-30 then return (0.0, 0.0)
    let α := hh * Float.sqrt tn2 / Float.sqrt mn2
    let θp ← F32.axpySlice (F32.concat #[θ]) fo.toUSize out0 (nP + fo).toUSize fl.toUSize α
    let θm ← F32.axpySlice (F32.concat #[θ]) fo.toUSize out0 (nP + fo).toUSize fl.toUSize (-α)
    let op ← runAt θp
    let om ← runAt θm
    return ((F32.read op lossOff.toUSize - F32.read om lossOff.toUSize) / 2.0, gScale * α * mn2)
  -- The fp32 noise floor: along the tier-1 invariance direction of layer 1 — its `(W, b)` pair,
  -- contiguous as tensors 0 and 1 — the loss is exactly invariant, so the central difference
  -- measured there at ‖δ‖ = `sNorm` is pure arithmetic noise at full depth.
  let radial (sNorm : Float) : IO Float := do
    let (fo, fl) := (offs[0]!, offs[2]! - offs[0]!)
    let s := sNorm / Float.sqrt (F32.dotSlice θ fo.toUSize θ fo.toUSize fl.toUSize)
    let θp ← F32.axpySlice (F32.concat #[θ]) fo.toUSize θ fo.toUSize fl.toUSize s
    let θm ← F32.axpySlice (F32.concat #[θ]) fo.toUSize θ fo.toUSize fl.toUSize (-s)
    let op ← runAt θp
    let om ← runAt θm
    return ((F32.read op lossOff.toUSize - F32.read om lossOff.toUSize) / 2.0).abs

  -- `C8BBN_GC_DUMP=<prefix>` — write the base point (θ, x, y as int32) and the step's full output
  -- (`[θ'|m'|v'|loss,bc₁,bc₂|stats]`) as raw little-endian f32/int32 files, for an independent
  -- autodiff of the same forward (`tests/cifar8_bbn_jax_ref.py`) to read.
  if let some pre ← IO.getEnv "C8BBN_GC_DUMP" then
    IO.FS.writeBinFile s!"{pre}_theta.bin" θ
    IO.FS.writeBinFile s!"{pre}_x.bin" x
    IO.FS.writeBinFile s!"{pre}_y.bin" y
    IO.FS.writeBinFile s!"{pre}_out.bin" out0
    IO.println s!"  dumped θ/x/y/out to {pre}_*.bin"

  -- `C8BBN_GC_TENSORS=1` — the adjoint probe per TENSOR at the chosen h, to localise a group's
  -- residual to W, b, γ or β. Diagnostic only, no verdict.
  if (← IO.getEnv "C8BBN_GC_TENSORS").isSome then
    IO.println ""
    IO.println s!"  per-tensor adjoint probe at h = {h}   tensor   <g,δ>   (L₊−L₋)/2   rel"
    let names := #["W", "b", "γ", "β"]
    for t in [0:nT] do
      let (fo, fl) := (offs[t]!, offs[t+1]! - offs[t]!)
      let (fd, pred) ← probe fo fl h
      nInvokes := nInvokes + 2
      let nm := if t < 32 then s!"L{t / 4 + 1}.{names[t % 4]!}"
                else s!"head.{#["W9", "b9", "Wa", "ba", "Wb", "bb"][t - 32]!}"
      IO.println s!"    {nm}\t{pred}\t{fd}\t{(fd - pred).abs / max pred.abs 1e-30}"
    IO.println s!"  {nInvokes} invokes — per-tensor diagnostic only, no verdict"
    return ()

  if scan then
    IO.println ""
    IO.println "  step-size scan — rel error per h, L1 (deepest) and head (shallowest), with L1's noise floor"
    IO.println "     h          L1 rel        L1 floor      L5 rel        head rel      whole rel"
    for u in [20000, 6000, 2000, 600, 200, 60, 20, 6] do
      let hh := u.toFloat * 1e-6
      let mut row := s!"  {hh}"
      for lbl in ["L1", "L5", "head"] do
        let g := (c8Groups.find? (·.label == lbl)).get!
        let (fo, fl) := (offs[g.t0]!, offs[g.t0 + g.nT]! - offs[g.t0]!)
        let (fd, pred) ← probe fo fl hh
        nInvokes := nInvokes + 2
        row := row ++ s!"\t{(fd - pred).abs / max pred.abs 1e-30}"
        if lbl == "L1" then
          let tn2 := F32.dotSlice θ fo.toUSize θ fo.toUSize fl.toUSize
          let fl0 ← radial (hh * Float.sqrt tn2)
          nInvokes := nInvokes + 2
          row := row ++ s!"\t{fl0 / max pred.abs 1e-30}"
      let (fd, pred) ← probe 0 nP hh
      nInvokes := nInvokes + 2
      row := row ++ s!"\t{(fd - pred).abs / max pred.abs 1e-30}"
      IO.println row
    IO.println s!"  {nInvokes} invokes — scan only, no verdict"
    return ()

  IO.println ""
  IO.println "  TIER 2  group   form                           <g,δ>          (L₊−L₋)/2        rel"
  let mut worst := 0.0; let mut worstAt := ""
  let mut scaleCtl := 1e30; let mut scaleCtlAt := ""
  for g in c8Groups ++ #[⟨"whole", "all 38 tensors", 0, nT⟩] do
    let (fo, fl) := (offs[g.t0]!, offs[g.t0 + g.nT]! - offs[g.t0]!)
    let (fd, pred) ← probe fo fl h
    nInvokes := nInvokes + 2
    let rel := (fd - pred).abs / max pred.abs 1e-30
    let rel2 := (fd - 2.0 * pred).abs / max pred.abs 1e-30
    if rel > worst then worst := rel; worstAt := g.label
    if rel2 < scaleCtl then scaleCtl := rel2; scaleCtlAt := g.label
    IO.println s!"          {g.label}\t{g.form}\t{pred}\t{fd}\t{rel}"
  let l1 := c8Groups[0]!
  let l1Fl := offs[l1.t0 + l1.nT]! - offs[l1.t0]!
  let (_, l1Pred) ← probe offs[l1.t0]! l1Fl h
  let l1Floor ← radial (h * Float.sqrt (F32.dotSlice θ offs[l1.t0]!.toUSize θ
                          offs[l1.t0]!.toUSize l1Fl.toUSize))
  nInvokes := nInvokes + 4
  IO.println s!"  worst rel {worst} at {worstAt};  L1's fp32 NOISE FLOOR at the same ‖δ‖ is \
{l1Floor / max l1Pred.abs 1e-30}"
  IO.println s!"  ⟂ scale control (fd vs 2·⟨g,δ⟩)  weakest {scaleCtl} at {scaleCtlAt}"
  IO.println s!"  {nInvokes} invokes of @{fn}"

  -- ════════════════════════════════════════════════════════════════
  -- § Verdict
  -- ════════════════════════════════════════════════════════════════
  IO.println ""
  if wA > tolExact then
    throw <| IO.userError s!"TIER 1 A FAILED: ⟨g_W,W⟩+⟨g_b,b⟩ ≠ 0 at {atA} (|cos∠| {wA} > \
{tolExact}) — the conv+BN VJP does not respect the scale invariance it is built on"
  if bC <= 5.0 * wA then
    throw <| IO.userError s!"CONTROL DEAD: {atC} (no BN after it) violates the identity by only \
{bC} against a homogeneous worst of {wA} — the cosine is ~0 for reasons unrelated to the identity"
  if bC <= tolExact then
    throw <| IO.userError s!"TOLERANCE TOO LOOSE: {atC} genuinely violates the identity at {bC}, \
inside the {tolExact} tolerance"
  if worst > tol then
    throw <| IO.userError s!"TIER 2 FAILED: {worstAt} disagrees by rel {worst} > {tol} — the \
rendered backward's MAGNITUDE is not the gradient of the rendered forward"
  if scaleCtl <= 3.0 * worst then
    throw <| IO.userError s!"CONTROL DEAD: a gradient twice as large fits {scaleCtlAt} as well as \
the real one ({scaleCtl} vs a tie of {worst})"
  IO.println "  ✅ GRADCHECK PASSES."
  IO.println s!"     TIER 1: the 8 BN-followed convs satisfy ⟨g_W,W⟩+⟨g_b,b⟩ = 0 to |cos∠| ≤ {wA}, \
against 3 head weights that violate it by ≥ {bC} ({bC / max wA 1e-30}×)."
  IO.println s!"     TIER 2: all {c8Groups.size} groups and the whole net agree in magnitude to \
rel ≤ {worst}, against a doubled-gradient control that misses by {scaleCtl} \
({scaleCtl / max worst 1e-30}× the tie)."
  IO.println s!"     The backward of {artifact} is the gradient of its forward."
