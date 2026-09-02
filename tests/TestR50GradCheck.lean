import LeanMlir.VerifiedNets

/-! # ResNet-50's GRADIENT, gated — on the committed artifact, in two tiers

`planning/next_session_pipeline_then_r50.md` §3.2's debt, paid. R50 phases 1–3 shipped a net that
renders, compiles, trains and descends behind a **layout** gate (`tests/TestR50Contract.lean`) and
nothing at all on the backward. §3.2's own words: *"The gradient is ungated. Say which one licensed
the swap. Neither has been run."*

## Why not the two checks §3.2 named

§3.2 proposed two `vjp_oracle` cases and a matched-init loss tie against
`jax/MainResnet50Imagenet.lean`. ⚠⚠ **The oracle cases would gate the wrong emitter.**
`tests/vjp_oracle/run.sh` drives `NetSpec.train`, i.e. `LeanMlir/MlirCodegen.lean`'s
`emitBottleneckBlock` — a different lowering from `Proofs/Codegen/ResNet50RenderB.lean`, which is
what `verified_mlir/resnet50in_*_train_step.mlir` is rendered from and what
`resnet50-imagenet-verified` actually runs. Those cases are still worth adding (they cover
`apps/baselines/MainResnet50Train.lean`) and they are not this.

This drives **the committed bytes** and needs no second framework, because the train step returns
its own loss beside its own first moment:

    out = [θ' | m' | v' | loss, bc₁, bc₂ | 106 BN stats]

so ONE invoke from `m = v = 0` yields both `L(θ)` and `m' = (1−β₁)·g = 0.1·g` — §2k's construction,
reused by `r34-mom-tie`, `rms-tie` and `conv-bias-zero`. AdamW's decay is DECOUPLED, so `m` sees the
raw gradient.

## ⭐ TIER 1 — two closed-form identities, no finite differences anywhere

Every conv in R50 is BN-followed and bias-free (`convBnNB`). Two consequences, both exact:

| | identity | sites | what it certifies |
|---|---|---|---|
| **A** | `⟨g_W, W⟩ = 0` | 53 conv kernels | `L` is 0-homogeneous in each kernel — BN divides the scale straight back out |
| **B** | `⟨g_γ, γ⟩ + ⟨g_β, β⟩ = 0` | 33 pre-conv BN affines | the same for the BN affine, whose scale the NEXT conv's BN removes |

⚠ **A and B are not the same check, and B is the one with teeth.** A factors as `⟨c, J_W W⟩` with
`J_W W = 0`, so it holds for ANY cotangent `c` arriving at the BN output: it certifies the local
conv+BN VJP and is structurally BLIND to whatever fed it. B expands to `⟨c, γx̂ + β⟩`, a statement
about the arriving cotangent itself, so it fails if anything between this BN and the next one
differentiates wrongly. ⭐ **In particular the stem's B crosses `maxPool3s2`** — R50's one stem-path
op, and the only route by which the stem's parameters get a gradient at all.

⚠ 33 affines, not 53: `bn3` and `bnp` feed the RESIDUAL ADD, which is not homogeneous, so the
invariance genuinely fails there. **That is the control** — see below.

⚠ Neither identity is exactly 0 in the render: BN's `ε = 1e-5` does not scale with the variance, so
the true cosines are `O(ε/var)`. Measured worst **6.1e-5**, which is that order and not a
coincidence. ⭐ The tolerance is **3e-4**, and it is not a round number picked for comfort — it sits
between the two measured populations: every homogeneous site is ≤ 6.1e-5 and every non-homogeneous
CONTROL site is ≥ 7.3e-4 (median 2.4e-2, max 0.12). The gate refuses to run with a tolerance loose
enough to admit the weakest real violation.

## TIER 2 — the adjoint probe, which is what fixes the SCALE

Tier 1 is scale-blind: multiply the whole gradient by any constant and both identities still hold.
So each parameter group additionally gets

    ⟨g, δ⟩  ≟  (L(θ+δ) − L(θ−δ)) / 2 + O(‖δ‖³),     δ = α·m' restricted to that group

with `α` set so `‖δ‖ = h·‖θ_group‖`. `δ ∝ g` rather than random: `⟨g, δ⟩` for a random direction
shrinks like `1/√n` and R50's largest tensor has 4.7M entries, so a random probe would sit under the
forward's own fp32 noise. Taking `δ = α·m'` makes `⟨g, δ⟩ = 10α‖m'‖²` — maximal signal, always
positive, never degenerate. It is a fair test despite coming from `g`: a wrong backward gives
`⟨g_true, g_wrong⟩` against a prediction of `‖g_wrong‖²`, and those agree only by coincidence.

⚠⚠ **THE RESOLUTION OF THIS TIER IS DEPTH-DEPENDENT, AND THAT IS MEASURED, NOT ASSUMED.**
`R50_GC_SCAN=1` prints the step-size curve. Perturbing the head leaves 49 layers bit-identical, so
their rounding CANCELS in `L₊ − L₋`; perturbing the stem changes every layer downstream and drags
the forward across millions of ReLU and max-pool kinks, whose central-difference error decays far
slower than `h²`. Measured at `h = 6e-5`, one run, the residual falls monotonically with depth:

    stem 0.17 · s1b0 0.12 · s1b1 0.089 · s2b0 0.041 · s3b0 0.019 · s4b0 0.0027 · head 0.00097

▶ So the honest sentence is: **tier 2 pins the gradient's magnitude to ~0.1% at the head and stage
4, loosening to ~17% at the stem**, and tier 1 pins its structure to ~6e-5 everywhere including the
stem. The two cover each other's blind spots — that is the design, not an apology for the tolerance.

⚠ A whole-net RANDOM-SIGN direction was tried and DROPPED rather than quietly kept: at `n = 25.5M`
its `⟨g,δ⟩` came out at 2e-4 against per-group signals of 3e-2, i.e. degenerate by construction, and
it reported rel 0.12–0.52 on a gradient the rest of this file certifies. The blind spot it was meant
to cover — error strictly orthogonal to `g` — is instead covered by tier 1, whose identities are
statements about `g`'s component along `θ` and are not along `g` at all.

## The controls, because a green with no control is not a reading (§6)

* ⭐⭐ **Tier 1's control is arithmetic, and it is free.** `bn3` and `bnp` feed the residual add and
  the dense `Wd` has no BN after it, so those 21 sites must **VIOLATE** the identity. If they came
  out at 6e-5 too, the cosine would be ~0 for trivial reasons and tier 1 would be vacuous. What the
  harness requires is a POPULATION SEPARATION, not an order statistic on one site: every control
  above every passing site (≥5×), the MEDIAN control ≥100× the worst passing site, and the
  tolerance strictly between the two. ⚠ The weakest single control, `s1b0.proj.bnp`, lands at 7.3e-4
  against peers at 1e-2 — its cotangent happens to sit nearly orthogonal to its own BN output at
  this base point — so a "weakest violation ≥ 100×" rule would have failed a working harness.
* ⭐ **The four projection shortcuts are their own groups.** `sXb0.proj` carries only `Wp/gp/btp`, so
  its adjoint probe measures the shortcut branch's gradient and nothing else — the branch §3.2 says
  is covered by neither existing check. (A first design predicted `sXb0` with the shortcut's terms
  DROPPED and required the fit to break; it did not, because the shortcut is only ~4% of `‖m'‖²` at
  s1b0. Recorded because "the control was too weak to fire" is the failure mode §6 is about.)
* **Scale control.** Each group's finite difference is re-checked against `2·⟨g,δ⟩`. A harness that
  cannot tell a gradient from twice a gradient is measuring nothing; required to miss by ≥3× the
  tie, which at the stem is 6.9× and at stage 4 is ~700×.

## ⚠ This gates `adam64`; the driver defaults to `adamdp64`

Tier 2 cannot run on the DP render: it all-reduces the GRADIENT but not the loss, so `%loss` is
replica 0's own shard and the adjoint identity would compare a 256-sample gradient against a
64-sample loss. **`tests/r50_dp_render_tie.py` carries the verdict across by text instead** — it
shows the two artifacts are the same program line for line once each gradient is routed through
its `all_reduce`, so everything that COMPUTES the gradient is literally the same. Run both.

    lake build r50-gradcheck && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/r50-gradcheck
    python3 tests/r50_dp_render_tie.py                                   # extends it to adamdp64
    CUDA_VISIBLE_DEVICES=0 R50_GC_SCAN=1 .lake/build/bin/r50-gradcheck   # the step-size curve

Knobs (micro-units, so one `Nat` reaches 1e-6): `R50_GC_EPS_U` (step, default 60 = 6e-5),
`R50_GC_TOL_U` (tier-2 tolerance, default 300000 = 0.30), `R50_GC_EXACT_U` (tier-1 tolerance,
default 300 = 3e-4), `R50_GC_VARIANT` (default `adam64`, the single-device render).
-/

/-- The driver's own init (`VerifiedTrain.mkParam`, which is private): He **fan-out** for conv,
    Glorot for dense, γ = 1. Using the real init matters — a constant splat makes BN see zero
    variance, and the gradient this recovers would be of a degenerate forward.

    ⚠ **ONE deliberate departure: β (kind 2) is small noise, not 0.** The driver zeroes it, and
    `β = 0` is a DEGENERATE point for identity B — `⟨g_β, β⟩` is then 0 whatever the render
    computes, so half of that check would be satisfied by construction rather than by being right.
    A non-degenerate base point is worth more to a gradcheck than a faithful one. -/
private def mkParamBiasSeeded (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
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

/-- A contiguous run of parameter TENSORS. Parameters are packed in func-arg order, so every group
    here is a SLICE and the perturbation needs no mask. -/
private structure Grp where
  label : String
  form  : String
  t0    : Nat
  nT    : Nat
deriving Inhabited

/-- The 22 groups, derived from the `[3,4,6,3]` stage structure rather than written out, so they
    cannot drift from `r50ShapeList`'s own `bnkStageSig`. Block 0 of every stage carries 12 tensors
    — 9 on the main path plus `Wp/gp/btp` — and **the shortcut is split off into its own group** so
    its gradient is probed on its own rather than diluted by the main path's. -/
private def r50Groups : Array Grp := Id.run do
  let mut gs : Array Grp := #[⟨"stem", "conv7x7-s2 + BN + maxPool3s2", 0, 3⟩]
  let mut t := 3
  for (s, count, form0) in
      [("s1", 3, "PROJ stride-1 ⭐"), ("s2", 4, "PROJ strided"),
       ("s3", 6, "PROJ strided"),    ("s4", 3, "PROJ strided")] do
    gs := gs.push ⟨s!"{s}b0",      form0,            t,     9⟩
    gs := gs.push ⟨s!"{s}b0.proj", "shortcut branch", t + 9, 3⟩
    t := t + 12
    for b in [1:count] do
      gs := gs.push ⟨s!"{s}b{b}", "identity", t, 9⟩
      t := t + 9
  return gs.push ⟨"head", "dense 2048→1000", t, 2⟩

def main : IO Unit := do
  -- ▶ Resolution selects the NET (slug + `d0`). Parameter layout is identical by construction, so
  -- every offset, group and tolerance below is unchanged — only `x`'s width moves.
  let net ← match (← IO.getEnv "R50_GC_RES").getD "224" with
    | "224" => pure resnet50ImagenetVerified.toNet
    | "160" => pure resnet50Imagenet160Verified.toNet
    | r     => throw <| IO.userError s!"R50_GC_RES={r}: want 224 or 160"
  let bs   := 64
  let nP   := net.nParams
  let nT   := net.specs.size
  let variant := (← IO.getEnv "R50_GC_VARIANT").getD "adam64"
  -- ⭐⭐ ACCUMULATION-AWARE. A `*acc<k>x*` render has FOUR parameter regions and FIVE scalars, so
  -- the blob, the loss offset and the gradient recovery all shift. Driving it with `%aup = 1`
  -- (apply) and `%akeep = 0` (discard the incoming accumulator) makes ONE invoke a complete cycle:
  -- `Gt = 0·G + g = g`, so the artifact behaves as its non-accumulating peer except that `%ob1`
  -- carries `(1−β₁)/k` rather than `(1−β₁)`.
  -- ▶ Hence `m' = ((1−β₁)/k)·g = (0.1/k)·g` and the gradient is recovered at **10k·m'**, not 10·m'.
  -- ⚠ Getting that factor wrong is invisible to tier 1 — both homogeneity identities are SCALE
  -- INVARIANT in `g`, so a wrong `k` would sail through them and only tier 2's absolute fit would
  -- notice. That is why it is derived from the name rather than assumed.
  let accOn := (variant.splitOn "acc").length > 1
  let accK  := if accOn then
      (let after := (variant.splitOn "acc").getD 1 ""
       let after := if after.startsWith "dp" then after.drop 2 else after
       ((after.takeWhile (· != 'x')).toNat?).getD 0)
    else 1
  if accOn && accK < 2 then
    throw <| IO.userError s!"could not read k from accumulating variant '{variant}'"
  -- ⚠ The driver's own, not a copy: the region count became 3, 4 or 5 when EMA and accumulation
  -- stopped sharing the fourth slot (`VerifiedVariant.nRegions`, RSB-A2/A1, 2026-08-27), and a
  -- frozen `if accOn then 4 else 3` here would size the blob one region short for an `ema…acc…`
  -- variant — every parameter after θ misaligned, and nothing throws.
  let nRegions := VerifiedVariant.nRegions variant
  let gScale := 10.0 * accK.toFloat        -- g = gScale · m'
  let lossOff := nRegions * nP
  -- Micro-units throughout, so a single `Nat` env var reaches 1e-6 without a float parser.
  let uEnv (k : String) (dflt : Float) : IO Float := do
    pure (((← IO.getEnv k).bind (·.toNat?)).map (fun u => u.toFloat * 1e-6) |>.getD dflt)
  let h        ← uEnv "R50_GC_EPS_U"   6.0e-5
  let tol      ← uEnv "R50_GC_TOL_U"   0.30
  let tolExact ← uEnv "R50_GC_EXACT_U" 3.0e-4
  let scan := (← IO.getEnv "R50_GC_SCAN").isSome
  let lr   := 0.0      -- θ' is never read; 0 makes that explicit rather than incidental
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]

  IO.println "§3.2's owed gate — ResNet-50's GRADIENT on the committed train step"
  IO.println s!"  artifact verified_mlir/{net.slug}_{variant}_train_step.mlir"
  IO.println s!"  {nT} params ({nP} floats), bs {bs}, {net.bnChannels.size} BN layers, \
backend {← LowererSession.backendName}"
  IO.println s!"  tier 1 |cos∠| ≤ {tolExact}   ·   tier 2 h = {h}, rel ≤ {tol}"

  -- ── the tensor→float offset table, so a group is a slice ──
  let mut offs : Array Nat := #[0]
  for sh in net.paramShapes do offs := offs.push (offs.back! + sh.foldl (· * ·) 1)
  if offs.back! != nP then
    throw <| IO.userError s!"offset table totals {offs.back!}, net.nParams is {nP}"
  if r50Groups.foldl (fun a g => a + g.nT) 0 != nT then
    throw <| IO.userError s!"the group table covers {r50Groups.foldl (fun a g => a + g.nT) 0} \
tensors, the net has {nT} — the [3,4,6,3] derivation is out of step with the spec"

  -- ── one (θ, x, y) everything below sees ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParamBiasSeeded sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  let z ← F32.const nP.toUSize 0.0
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let shapes := packShapes ((List.replicate nRegions net.paramShapes).foldl (· ++ ·) #[]
                            ++ Array.replicate (VerifiedVariant.nScalars variant) #[] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % 251)); y := y.push 0; y := y.push 0; y := y.push 0

  -- ── the session, opened once: same shapes every invoke ⇒ XLA autotunes ONCE and every probe
  --    runs the identical program, so `L(θ+δ)` and `L(θ−δ)` differ by the perturbation and not by
  --    a per-process algorithm choice (the effect `scripts/det_shim.sh` exists for) ──
  let sess ← mkSession s!"verified_mlir/{net.slug}_{variant}_train_step.mlir"
  let fn := s!"m.{net.slug}_{variant}_train_step"
  -- lr, 1−β₁¹, 1−β₂¹ at t = 1; then `%aup = 1`, `%akeep = 0` on the accumulating render.
  let tl ← if accOn then do
      let t5 ← F32.write3 (← F32.const 5 0.0) 0 lr 0.1 0.001
      let pair ← F32.write3 (← F32.const 3 0.0) 0 1.0 0.0 0.0
      F32.blit t5 3 pair 0 2
    else F32.write3 (← F32.const 3 0.0) 0 lr 0.1 0.001
  let mut nInvokes := 0
  let runAt (θx : ByteArray) : IO ByteArray :=
    LowererSession.mlpTrainStepV sess fn x
      (F32.concat (if accOn then #[θx, z, z, z, tl, bnIn] else #[θx, z, z, tl, bnIn])) shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize

  -- ── the base point: L(θ) and g = 10·m' ──
  let out0 ← runAt θ
  nInvokes := nInvokes + 1
  let l0 := F32.read out0 lossOff.toUSize
  let gNorm2 := gScale * gScale * F32.dotSlice out0 nP.toUSize out0 nP.toUSize nP.toUSize
  IO.println s!"  base loss {l0}   ‖g‖ {Float.sqrt gNorm2}   (ln 1000 = 6.907755)"
  if !l0.isFinite || l0 < 1e-3 then
    throw <| IO.userError s!"DEGENERATE: base loss {l0} — nothing is being differentiated"
  if !gNorm2.isFinite || gNorm2 < 1e-12 then
    throw <| IO.userError s!"DEGENERATE: ‖g‖² = {gNorm2} — the recovered gradient is ~0"

  -- ════════════════════════════════════════════════════════════════
  -- § TIER 1 — the two homogeneity identities, and the sites that must VIOLATE them
  -- ════════════════════════════════════════════════════════════════

  -- `|cos∠(g_S, θ_S)|` over a set of tensor slices taken together.
  let cosine (ts : Array Nat) : Float := Id.run do
    let mut s := 0.0; let mut gn2 := 0.0; let mut pn2 := 0.0
    for t in ts do
      let (fo, fl) := (offs[t]!, offs[t+1]! - offs[t]!)
      s   := s   + gScale * F32.dotSlice out0 (nP + fo).toUSize θ    fo.toUSize        fl.toUSize
      gn2 := gn2 + gScale * gScale * F32.dotSlice out0 (nP + fo).toUSize out0 (nP + fo).toUSize fl.toUSize
      pn2 := pn2 +         F32.dotSlice θ    fo.toUSize        θ    fo.toUSize        fl.toUSize
    return s.abs / max (Float.sqrt gn2 * Float.sqrt pn2) 1e-30

  -- ⚠⚠ **THE CONDITIONING OF THE IDENTITY, and it is what `a3_paper_fidelity.md` §3.1 is about.**
  --
  -- `cosine` reports `|Σ_t ⟨g_t,θ_t⟩| / (‖g‖‖θ‖)`, and for identity **B** that numerator is a
  -- CANCELLATION: `⟨g_γ,γ⟩` and `⟨g_β,β⟩` are each large and must sum to zero. `kappa` measures how
  -- much — `(|⟨g_γ,γ⟩| + |⟨g_β,β⟩|) / (‖g‖‖θ‖)`, in `[0,1]` — so the cosine cannot resolve anything
  -- below roughly `eps_f32 · κ · √n` no matter how right the render is.
  --
  -- ▶ Identity **A** has no such term: it is a SINGLE inner product `⟨g_W,W⟩`, so κ ≈ |cos| and its
  -- floor is the plain fp32 one. That asymmetry is the whole reason B degrades under BCE while A
  -- does not — measured, not assumed: at BCE, A's worst is 2.2e-05 and B's is 7.65e-04.
  let kappa (ts : Array Nat) : Float := Id.run do
    let mut a := 0.0; let mut gn2 := 0.0; let mut pn2 := 0.0
    for t in ts do
      let (fo, fl) := (offs[t]!, offs[t+1]! - offs[t]!)
      a   := a + (gScale * F32.dotSlice out0 (nP + fo).toUSize θ fo.toUSize fl.toUSize).abs
      gn2 := gn2 + gScale * gScale * F32.dotSlice out0 (nP+fo).toUSize out0 (nP+fo).toUSize fl.toUSize
      pn2 := pn2 +         F32.dotSlice θ fo.toUSize θ fo.toUSize fl.toUSize
    return a / max (Float.sqrt gn2 * Float.sqrt pn2) 1e-30

  -- A: the 53 BN-followed conv kernels.  B: the 33 BN affines whose scale the next conv removes.
  -- CONTROL: `bn3`/`bnp` feed the residual add and `Wd` has no BN after it — 21 sites where the
  -- invariance is FALSE and the cosine must therefore be large.
  let mut idA : Array (Array Nat × String) := #[(#[0], "stem.W")]
  let mut idB : Array (Array Nat × String) := #[(#[1, 2], "stem.bn ⭐through maxPool3s2")]
  let mut ctl : Array (Array Nat × String) := #[(#[nT - 2], "head.Wd")]
  for g in r50Groups do
    if g.form == "identity" || g.form.startsWith "PROJ" then
      idA := idA.push (#[g.t0], s!"{g.label}.W1") |>.push (#[g.t0+3], s!"{g.label}.W2")
                      |>.push (#[g.t0+6], s!"{g.label}.W3")
      idB := idB.push (#[g.t0+1, g.t0+2], s!"{g.label}.bn1")
                      |>.push (#[g.t0+4, g.t0+5], s!"{g.label}.bn2")
      ctl := ctl.push (#[g.t0+7, g.t0+8], s!"{g.label}.bn3 (residual add)")
    if g.form == "shortcut branch" then
      idA := idA.push (#[g.t0], s!"{g.label}.Wp")
      ctl := ctl.push (#[g.t0+1, g.t0+2], s!"{g.label}.bnp (residual add)")
  let worstOf (xs : Array (Array Nat × String)) : Float × String := Id.run do
    let mut w := 0.0; let mut whr := ""
    for (ts, nm) in xs do
      let c := cosine ts
      if c > w then w := c; whr := nm
    return (w, whr)
  let bestOf (xs : Array (Array Nat × String)) : Float × String := Id.run do
    let mut b := 1e30; let mut whr := ""
    for (ts, nm) in xs do
      let c := cosine ts
      if c < b then b := c; whr := nm
    return (b, whr)
  let (wA, atA) := worstOf idA
  let (wB, atB) := worstOf idB
  let (bC, atC) := bestOf ctl
  let stemB := cosine #[1, 2]
  IO.println ""
  IO.println s!"  TIER 1  A  ⟨g_W,W⟩ = 0        {idA.size} conv kernels    worst |cos∠| {wA} at {atA}"
  IO.println s!"          B  ⟨g_γ,γ⟩+⟨g_β,β⟩=0  {idB.size} BN affines      worst |cos∠| {wB} at {atB}"
  IO.println s!"             ⭐ the stem's B, which is the only path through maxPool3s2: {stemB}"
  IO.println s!"          ⟂  control: {ctl.size} sites where the invariance is FALSE (residual-add \
BNs + the head)"
  let ctlSorted := (ctl.map (fun (ts, nm) => cosine ts)).qsort (fun a b => a < b)
  let ctlMed := ctlSorted[ctlSorted.size / 2]!
  -- ⭐⭐ **THE 10th-PERCENTILE VIOLATION, and it is what the verdict below now rests on.**
  --
  -- ⚠⚠ The check used to be `bC` — the MINIMUM over the 21 control sites — which is exactly the
  -- order statistic the comment at the verdict says not to use. `scripts/r50_gradcheck_stability.py`
  -- measured the cost of that contradiction (`a3_paper_fidelity.md` §3.1b): over three runs on the
  -- SAME seeded base point, `bC` spreads **2.75× under CE and 10.5× under BCE**, so under BCE the
  -- gate's answer depended on which run you happened to do — 2 of 3 reps cleared the separation and
  -- 1 did not. The base point is fixed; the GPU execution is not, and
  -- `--xla_gpu_deterministic_ops=true` does not fix it.
  --
  -- ▶ This quantile is the same statement made robustly: at `size/10` (index 2 of 21) it spreads
  -- **1.1×** across the same three runs, because it takes one site's collapse to move the minimum
  -- and three simultaneous collapses to move this. ⚠ It is deliberately NOT the median — the median
  -- already has its own, much stronger check below, and a separation claim wants the WEAK end of
  -- the violating population, just not its single weakest member.
  let ctlQ10 := ctlSorted[ctlSorted.size / 10]!
  IO.println s!"             weakest violation {bC} at {atC};  10th pct {ctlQ10};  median \
{ctlMed};  strongest {ctlSorted.back!}"
  -- ⚠ Reported, never thrown on: `bC` is the noisiest number in this report, so "the populations
  -- strictly separate" is stated as an observation and the VERDICT uses the quantile. A run where
  -- this line reads `no` is not evidence of a wrong render — check the quantile and the median.
  IO.println s!"             populations strictly separate (min ⟂ > worst passing)? \
{if bC > max wA wB then "yes" else "NO — see §3.1b, this is the unstable extremum"}"
  -- ▶ The per-site conditioning dump, behind `R50_GC_DIAG=1`. Read it before touching `tolExact`:
  -- a site whose |cos| is high AND whose κ is high is unresolved, not wrong.
  if (← IO.getEnv "R50_GC_DIAG").isSome then
    IO.println "  ── conditioning (R50_GC_DIAG) ──   site            |cos∠|         κ      cos/κ"
    for (ts, nm) in idB do
      let c := cosine ts; let k := kappa ts
      IO.println s!"                                    B  {nm}\t{c}\t{k}\t{c / max k 1e-30}"
    for (ts, nm) in ctl do
      let c := cosine ts; let k := kappa ts
      IO.println s!"                                    ⟂  {nm}\t{c}\t{k}\t{c / max k 1e-30}"

  -- ════════════════════════════════════════════════════════════════
  -- § TIER 2 — the adjoint probe, per group
  -- ════════════════════════════════════════════════════════════════

  -- `δ = α·m'` on the float slice `[fo, fo+fl)`, α set so `‖δ‖ = hh·‖θ_slice‖`. Returns the finite
  -- difference and the predicted `⟨g, δ⟩`. ⚠ `F32.concat #[θ]` is a deliberate COPY: `axpySlice`
  -- mutates its destination when unshared, and θ is the base point every probe starts from.
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

  -- ⭐ THE NOISE FLOOR, and R50 hands it to us for free. Scaling one BN-followed conv kernel is an
  -- exact invariance of the loss (tier 1 A is its derivative), so the finite difference measured
  -- along that direction is pure fp32 noise at that depth — which is what says the residuals below
  -- are the probe's TRUNCATION rather than its arithmetic. `sNorm` is passed so the floor is read
  -- at the same ‖δ‖ as the signal it is compared against.
  let radial (t : Nat) (sNorm : Float) : IO Float := do
    let (fo, fl) := (offs[t]!, offs[t+1]! - offs[t]!)
    let s := sNorm / Float.sqrt (F32.dotSlice θ fo.toUSize θ fo.toUSize fl.toUSize)
    let θp ← F32.axpySlice (F32.concat #[θ]) fo.toUSize θ fo.toUSize fl.toUSize s
    let θm ← F32.axpySlice (F32.concat #[θ]) fo.toUSize θ fo.toUSize fl.toUSize (-s)
    let op ← runAt θp
    let om ← runAt θm
    return ((F32.read op (3 * nP).toUSize - F32.read om (3 * nP).toUSize) / 2.0).abs

  -- ══ R50_GC_SCAN — the step-size curve, which is how `h` was CHOSEN and how the depth-dependent
  -- resolution in this file's header was MEASURED. Truncation falls with h, fp32 noise rises as
  -- 1/h, and the usable step is the basin between them — a different basin at each depth.
  if scan then
    IO.println ""
    IO.println "  step-size scan — rel error, with the BN-invariance noise floor beside the two \
groups that plateau"
    IO.println "     h       stem rel   stem floor    s1b0 rel   s1b0 floor    s4b0 rel     head rel"
    for u in [20000, 6000, 2000, 600, 200, 60, 20, 6] do
      let hh := u.toFloat * 1e-6
      let mut row := s!"  {hh}"
      for lbl in ["stem", "s1b0", "s4b0", "head"] do
        let g := (r50Groups.find? (·.label == lbl)).get!
        let (fo, fl) := (offs[g.t0]!, offs[g.t0 + g.nT]! - offs[g.t0]!)
        let (fd, pred) ← probe fo fl hh
        nInvokes := nInvokes + 2
        row := row ++ s!"\t{(fd - pred).abs / max pred.abs 1e-30}"
        if lbl == "stem" || lbl == "s1b0" then
          let tn2 := F32.dotSlice θ fo.toUSize θ fo.toUSize fl.toUSize
          let fl0 ← radial g.t0 (hh * Float.sqrt tn2)
          nInvokes := nInvokes + 2
          row := row ++ s!"\t{fl0 / max pred.abs 1e-30}"
      IO.println row
    IO.println s!"  {nInvokes} invokes — scan only, no verdict"
    return ()

  IO.println ""
  IO.println "  TIER 2  group        form                          <g,δ>        (L₊−L₋)/2       rel"
  let mut worst := 0.0
  let mut worstAt := ""
  let mut scaleCtl := 1e30
  let mut scaleCtlAt := ""
  for g in r50Groups do
    let (fo, fl) := (offs[g.t0]!, offs[g.t0 + g.nT]! - offs[g.t0]!)
    let (fd, pred) ← probe fo fl h
    nInvokes := nInvokes + 2
    let rel := (fd - pred).abs / max pred.abs 1e-30
    -- the scale control: the same finite difference against a gradient twice as large
    let rel2 := (fd - 2.0 * pred).abs / max pred.abs 1e-30
    if rel > worst then worst := rel; worstAt := g.label
    if rel2 < scaleCtl then scaleCtl := rel2; scaleCtlAt := g.label
    IO.println s!"          {g.label}\t{g.form}\t{pred}\t{fd}\t{rel}"
  -- the floor at the stem, at the same ‖δ‖ as the stem's own probe — the number that says the
  -- stem's residual is truncation and not arithmetic
  let stemG := (r50Groups.find? (·.label == "stem")).get!
  let stemFl := offs[stemG.t0 + stemG.nT]! - offs[stemG.t0]!
  let (_, stemPred) ← probe offs[stemG.t0]! stemFl h
  let stemFloor ← radial 0 (h * Float.sqrt (F32.dotSlice θ offs[stemG.t0]!.toUSize θ
                              offs[stemG.t0]!.toUSize stemFl.toUSize))
  nInvokes := nInvokes + 4
  IO.println s!"  worst rel {worst} at {worstAt};  stem's fp32 NOISE FLOOR at the same ‖δ‖ is \
{stemFloor / max stemPred.abs 1e-30} — the residual is truncation, not arithmetic"
  IO.println s!"  ⟂ scale control (fd vs 2·⟨g,δ⟩)  weakest {scaleCtl} at {scaleCtlAt}"
  IO.println s!"  {nInvokes} invokes of @{fn}"

  -- ════════════════════════════════════════════════════════════════
  -- § Verdict
  -- ════════════════════════════════════════════════════════════════
  IO.println ""
  if wA > tolExact then
    throw <| IO.userError s!"TIER 1 A FAILED: ⟨g_W, W⟩ ≠ 0 at {atA} (|cos∠| {wA} > {tolExact}) — \
the conv+BN VJP does not respect the scale invariance it is built on"
  if wB > tolExact then
    throw <| IO.userError s!"TIER 1 B FAILED: ⟨g_γ,γ⟩+⟨g_β,β⟩ ≠ 0 at {atB} (|cos∠| {wB} > \
{tolExact}) — the cotangent ARRIVING at that BN is wrong, i.e. something between it and the next \
convolution differentiates wrongly"
  -- ⚠ The control is a SEPARATION statement, not an order statistic on one site. Requiring the
  -- weakest of 21 violations to clear a fixed multiple is brittle: `s1b0.proj.bnp`'s cotangent
  -- happens to sit nearly orthogonal to its own BN output at this base point and lands at 7e-4,
  -- an order below its peers. What has to hold is that the two POPULATIONS do not overlap —
  -- every non-homogeneous site above every homogeneous one, with the tolerance between them —
  -- and that the typical violation is large enough that the separation is not one lucky site.
  --
  -- ⚠⚠ **THIS COMMENT AND THE CODE BELOW IT DISAGREED UNTIL 2026-08-14** — the check read
  -- `bC <= 5.0 * …`, i.e. the very order statistic ruled out two lines up, and had done since the
  -- gate was written. `a3_paper_fidelity.md` §3.1b has the measurement: `bC` spreads 2.75× (CE) and
  -- 10.5× (BCE) over three runs on the same seeded base point, so the BCE verdict flipped 2-of-3.
  -- The quantile spreads 1.1×. ▶ A comment that states the right rule is not a gate; this is the
  -- code catching up to it.
  if ctlQ10 <= 5.0 * max wA wB then
    throw <| IO.userError s!"CONTROL DEAD: the two populations overlap — the 10th-percentile \
non-homogeneous site sits at {ctlQ10} against a homogeneous worst of {max wA wB}, so the cosine is \
~0 for reasons unrelated to the identity and tier 1 proves nothing. (The single weakest is {bC} at \
{atC}; that extremum is deliberately NOT what this tests — §3.1b.)"
  if ctlMed <= 100.0 * max wA wB then
    throw <| IO.userError s!"CONTROL DEAD: the MEDIAN non-homogeneous site violates by only \
{ctlMed} against a homogeneous worst of {max wA wB} — the separation rests on outliers"
  if bC <= tolExact then
    throw <| IO.userError s!"TOLERANCE TOO LOOSE: {atC} genuinely violates the identity at {bC}, \
which is inside the {tolExact} tolerance — a real violation of this size would be accepted"
  if worst > tol then
    throw <| IO.userError s!"TIER 2 FAILED: {worstAt} disagrees by rel {worst} > {tol} — the \
rendered backward's MAGNITUDE is not the gradient of the rendered forward"
  if scaleCtl <= 3.0 * worst then
    throw <| IO.userError s!"CONTROL DEAD: a gradient twice as large fits {scaleCtlAt} as well as \
the real one ({scaleCtl} vs a tie of {worst}) — tier 2 is not resolving the scale it exists to pin"
  IO.println s!"  ✅ GRADCHECK PASSES."
  IO.println s!"     TIER 1: {idA.size} conv kernels and {idB.size} BN affines satisfy their \
homogeneity identities to |cos∠| ≤ {max wA wB} — including the stem's, the only gradient path \
through maxPool3s2 — against {ctl.size} non-homogeneous control sites that do not overlap them at all: weakest \
at the 10th percentile {ctlQ10} ({ctlQ10 / max (max wA wB) 1e-30}× the worst passing site), \
median {ctlMed} ({ctlMed / max (max wA wB) 1e-30}×), with the tolerance {tolExact} sitting between \
the two populations. ⚠ The single weakest violation is {bC}, reported but NOT the verdict's \
statistic — it is an unstable extremum (§3.1b)."
  IO.println s!"     TIER 2: all {r50Groups.size} groups agree in magnitude to rel ≤ {worst}, \
against a doubled-gradient control that misses by {scaleCtl} ({scaleCtl / max worst 1e-30}× the tie)."
  IO.println "     All three bottleneck forms are covered — the stride-1 projection (s1b0), the \
three strided projections, and the 12 identity blocks — plus each shortcut branch on its own."
  IO.println s!"     §3.2's \"the gradient is ungated\" is discharged for \
verified_mlir/{net.slug}_{variant}_train_step.mlir."
