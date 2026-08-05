import LeanMlir.VerifiedNets

/-! # Gradient accumulation, numerically certified — k micro-steps ARE one step

`planning/next_session_pipeline_then_r50.md` §4's blocker, gated. `verified_mlir/`'s new
`resnet50in_acc4x64_train_step.mlir` renders AdamW over **k accumulated micro-batches**: a fourth
parameter region `G`, and two runtime scalars deciding per invoke whether this is an accumulate or
an apply. This is the numeric argument that the mechanism computes what it claims.

## The construction, and why it is a known answer rather than a tolerance

Run the accumulation render **k times on the SAME batch**. Every micro-gradient is then the same
`g`, so the accumulated mean is exactly `g` — and the whole cycle must therefore reproduce ONE step
of the committed `resnet50in_adam64_train_step.mlir` on that batch, from the same `(θ, m, v)`.
That peer is not a re-derivation of the same arithmetic: it is a **different committed artifact**,
rendered before accumulation existed, and `lake build r50-gradcheck` certifies its gradient.

⚠ **A duplicated batch is a deliberate blind spot, and it is the same one every `*-dp-check` has.**
It cannot see whether DIFFERENT micro-batches are combined correctly — only that the accumulate /
apply machinery, the `1/k` folded into `%ob1`/`%ob2`, and the reset are right. The complementary
check (k micro-batches of b == one step at k·b on genuinely different data) is exact only where
there is no BatchNorm, exactly as `cifar8-dp-check` is the collective's exact peer: R50 normalises
per micro-batch by design (that IS Ghost-BN, which is what the JAX reference takes), so no exact
tie exists here and asserting one would be wrong rather than strict.

## What is checked, in order of what it would catch

| | claim | the defect it catches |
|---|---|---|
| ① | after micro-steps 1..k−1, `θ`, `m`, `v` are **bit-identical** to their inputs | an accumulate micro-batch that moves the optimizer — e.g. a COUPLED-L2 decay, which `lr = 0` would not freeze |
| ② | after micro-step k, `(θ', m', v')` match the plain AdamW render's one step | the `1/k`, the `1/k²` on the second moment, and the accumulator itself |

⭐ ② is where the asymmetry lives. `v` is QUADRATIC in the gradient, so `%ob2` carries `(1−β₂)/k²`
while `%ob1` carries `(1−β₁)/k`. A single shared scale — the obvious implementation — gives
`β₂v + (1−β₂)·(1/k)Σg²` instead of `β₂v + (1−β₂)·((1/k)Σg)²`: the mean of the per-micro-batch second
moments rather than the second moment of the mean. It descends, it looks completely normal, and it
is a different optimizer.

## The controls, because a green with no control is not a reading (§6)

* ⭐⭐ **WRONG-k.** Apply after `k−1` micro-batches instead of `k`. The graph still divides by `k`,
  so the update is `(k−1)/k` of the right one — **exactly the failure mode of a driver whose apply
  cadence disagrees with the artifact's baked `1/k`**, which is why the driver reads `k` off the
  artifact NAME. It must MISS by far more than the tie.
* **Degeneracy.** `θ' ≠ θ` by a wide margin, or everything above ties trivially.

⚠ Not bit-exact, and the reason is arithmetic rather than a defect: the accumulator forms
`g + g + g + g` by repeated addition, and `3g` needs two mantissa bits more than `g` has. `2g` and
`4g` are exact (exponent shifts); `3g` is not, so the k = 4 total is one rounding away from `4g`.
The bit-exact coordinate count is reported alongside, because `Float.toString` prints a genuine
3e-8 as `0.000000` (§2e-bis).

    lake build r50-accum-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/r50-accum-tie

Knobs: `R50_ACC_VARIANT` (default `acc4x64`), `R50_ACC_TOL_U` (micro-units, default 10 = 1e-5).
-/

/-- The driver's own init (`VerifiedTrain.mkParam`, private): He fan-out conv, Glorot dense, γ = 1,
    β = 0. Faithful here — unlike the gradcheck, nothing in this file is degenerate at β = 0. -/
private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    let variance :=
      if dims.size == 4 then 2.0 / (dims[0]! * dims[2]! * dims[3]!).toFloat
      else if dims.size == 2 then 2.0 / (dims[0]! + dims[1]!).toFloat
      else 2.0 / (dims[0]!).toFloat
    F32.heInit seed.toUSize n.toUSize (Float.sqrt variance)

/-- Max |a−b| and max |a| over a float range of two blobs, plus the bit-exact count. -/
private def cmp (a b : ByteArray) (offA offB n : Nat) : Float × Float × Nat := Id.run do
  let mut d := 0.0; let mut m := 0.0; let mut ex := 0
  for i in [0:n] do
    let x := F32.read a (offA + i).toUSize
    let y := F32.read b (offB + i).toUSize
    if x == y then ex := ex + 1
    if (x - y).abs > d then d := (x - y).abs
    if x.abs > m then m := x.abs
  return (d, m, ex)

def main : IO Unit := do
  let net  := resnet50ImagenetVerified.toNet
  let bs   := 64
  let nP   := net.nParams
  let variant := (← IO.getEnv "R50_ACC_VARIANT").getD "acc4x64"
  let tol  := (((← IO.getEnv "R50_ACC_TOL_U").bind (·.toNat?)).map (fun u => u.toFloat * 1e-6)
                |>.getD 1.0e-5)
  -- `k` off the artifact NAME, the same read the driver does — so this gate and the run it licenses
  -- cannot disagree about what the artifact was rendered for.
  let k := (((variant.drop (if variant.startsWith "accdp" then 5 else 3)).takeWhile (· != 'x')).toNat?).getD 0
  if k < 2 then
    throw <| IO.userError s!"could not read k from variant '{variant}' (want acc<k>x<B>)"
  let lr := 0.01                       -- large enough that θ' ≠ θ by a wide margin
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]

  IO.println s!"§4's gradient accumulation, certified — {k} micro-steps on one batch = one AdamW step"
  IO.println s!"  under test  verified_mlir/{net.slug}_{variant}_train_step.mlir   (4 regions [θ|m|v|G])"
  IO.println s!"  peer        verified_mlir/{net.slug}_adam64_train_step.mlir      (3 regions, committed \
before accumulation existed; its gradient is certified by `lake build r50-gradcheck`)"
  IO.println s!"  {net.specs.size} params ({nP} floats), bs {bs}, lr {lr}, backend {← IreeSession.backendName}"

  -- ── one (θ, m, v, x, y) both renders see ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  -- m and v are deliberately NON-ZERO and distinct: at m = v = 0 the β₁/β₂ passthrough terms
  -- vanish, so ① would hold for a render that dropped them and ② could not see a wrong β.
  let mIn ← F32.scaleShift (← F32.heInit 4242 nP.toUSize 0.02) 1.0 0.05
  let vIn ← F32.scaleShift (← F32.heInit 7373 nP.toUSize 0.001) 1.0 0.02
  -- G's initial contents are ARBITRARY on purpose: `%akeep = 0` on the first micro-batch must
  -- discard them. A zero fill would let a missing reset pass.
  let gIn ← F32.scaleShift (← F32.heInit 9191 nP.toUSize 0.5) 1.0 0.3
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % 251)); y := y.push 0; y := y.push 0; y := y.push 0

  let pShapes := net.paramShapes
  let accShapes := packShapes (pShapes ++ pShapes ++ pShapes ++ pShapes
                               ++ Array.replicate 5 #[] ++ bnStatShapes)
  let admShapes := packShapes (pShapes ++ pShapes ++ pShapes
                               ++ #[#[], #[], #[]] ++ bnStatShapes)

  let mkRun (v : String) : IO (ByteArray → IO ByteArray) := do
    let sess ← mkSession s!"verified_mlir/{net.slug}_{v}_train_step.mlir"
                         s!".lake/build/r50_accum_tie_{v}.vmfb"
    pure (fun buf => IreeSession.mlpTrainStepV sess s!"m.{net.slug}_{v}_train_step" x buf
            (if v == variant then accShapes else admShapes) y
            bs.toUSize net.d0.toUSize net.nClasses.toUSize)
  let runAcc ← mkRun variant
  let runAdm ← mkRun "adam64"

  -- Adam's bias correction at t = 1, the step the peer takes. The accumulate micro-batches read
  -- these too, but at lr = 0 `adamWParamF` contributes nothing, so they are irrelevant there.
  let bc1 := 1.0 - 0.9
  let bc2 := 1.0 - 0.999

  /- One accumulation cycle of `steps` micro-batches, applying on the last. Returns the final
     output blob and the max |Δ| over `[θ|m|v]` between the input and the state after the LAST
     ACCUMULATE micro-batch — claim ①'s measurement, and it is taken inside the cycle because after
     the apply everything has legitimately moved. -/
  let cycle (steps : Nat) : IO (ByteArray × Float × Nat) := do
    let mut buf := F32.concat #[θ, mIn, vIn, gIn, ← F32.const 5 0.0, bnIn]
    let mut frozenΔ := 0.0
    let mut frozenExact := 0
    for i in [0:steps] do
      let applyNow := i + 1 == steps
      buf ← F32.write3 buf (4 * nP).toUSize (if applyNow then lr else 0.0) bc1 bc2
      let pair ← F32.write3 (← F32.const 3 0.0) 0
                   (if applyNow then 1.0 else 0.0) (if i == 0 then 0.0 else 1.0) 0.0
      buf ← F32.blit buf (4 * nP + 3).toUSize pair 0 2
      buf ← F32.blit buf (4 * nP + 5).toUSize bnIn 0 nBnStats.toUSize
      let out ← runAcc buf
      if !applyNow then
        -- ① θ, m and v must be bit-identical across an accumulate micro-batch.
        let (dT, _, eT) := cmp out θ   0        0 nP
        let (dM, _, eM) := cmp out mIn (nP)     0 nP
        let (dV, _, eV) := cmp out vIn (2 * nP) 0 nP
        frozenΔ := max frozenΔ (max dT (max dM dV))
        frozenExact := eT + eM + eV
      -- the output blob is the next micro-batch's input, exactly as in the driver
      buf := out
    return (buf, frozenΔ, frozenExact)

  let (accOut, frozenΔ, frozenExact) ← cycle k
  -- the peer: ONE plain AdamW step from the same (θ, m, v)
  let admBuf := F32.concat #[θ, mIn, vIn, ← F32.write3 (← F32.const 3 0.0) 0 lr bc1 bc2, bnIn]
  let admOut ← runAdm admBuf

  -- ── ② the tie, region by region ──
  let (dT, mT, eT) := cmp accOut admOut 0        0        nP
  let (dM, mM, eM) := cmp accOut admOut nP       nP       nP
  let (dV, mV, eV) := cmp accOut admOut (2 * nP) (2 * nP) nP
  let (dMove, _, _) := cmp accOut θ 0 0 nP        -- degeneracy: did θ actually move?
  let relT := dT / max mT 1e-30
  let relM := dM / max mM 1e-30
  let relV := dV / max mV 1e-30
  let worst := max relT (max relM relV)
  IO.println ""
  IO.println s!"  ① accumulate micro-batches leave [θ|m|v] frozen"
  IO.println s!"      max abs Δ {frozenΔ}   bit-exact {frozenExact}/{3 * nP}"
  IO.println s!"  ② after {k} micro-steps == one AdamW step on the same batch"
  IO.println s!"      θ'  max abs Δ {dT}  rel {relT}   bit-exact {eT}/{nP}"
  IO.println s!"      m'  max abs Δ {dM}  rel {relM}   bit-exact {eM}/{nP}"
  IO.println s!"      v'  max abs Δ {dV}  rel {relV}   bit-exact {eV}/{nP}"
  IO.println s!"      (θ' moved from θ by {dMove} — the degeneracy guard)"

  -- ── ⟂ the WRONG-k control: apply one micro-batch early ──
  let (badOut, _, _) ← cycle (k - 1)
  let (dTb, _, _) := cmp badOut admOut 0 0 nP
  let relTb := dTb / max mT 1e-30
  IO.println s!"  ⟂ WRONG-k control: apply after {k-1} micro-batches, graph still divides by {k}"
  IO.println s!"      θ' rel {relTb}"

  -- ── verdict ──
  IO.println ""
  if dMove < 1e-6 then
    throw <| IO.userError s!"DEGENERATE: θ' == θ (max Δ {dMove}) — nothing stepped, so both \
renders agree on doing nothing"
  if frozenExact != 3 * nP then
    throw <| IO.userError s!"① FAILED: an accumulate micro-batch moved the optimizer state \
({frozenExact}/{3 * nP} bit-exact, max Δ {frozenΔ}) — θ, m and v must be untouched until the apply"
  if worst > tol then
    throw <| IO.userError s!"② FAILED: {k} accumulated micro-steps != one AdamW step \
(θ' rel {relT}, m' rel {relM}, v' rel {relV}, tolerance {tol}). If v' is the outlier, suspect the \
1/k² on %ob2 — a shared 1/k gives the mean of the second moments instead of the second moment of \
the mean"
  if relTb <= 10.0 * worst then
    throw <| IO.userError s!"CONTROL DEAD: applying after {k-1} micro-batches fits as well as \
after {k} ({relTb} vs a tie of {worst}) — this harness cannot tell the apply cadence from the \
baked 1/k, which is the one disagreement that is otherwise silent"
  IO.println s!"  ✅ CERTIFIED: {k} accumulated micro-batches are ONE AdamW step at {k}x the batch \
— θ' rel {relT}, m' rel {relM}, v' rel {relV}, against a committed peer rendered before \
accumulation existed, with [θ|m|v] {3 * nP}/{3 * nP} bit-exact across every accumulate micro-batch \
and a wrong-k control that misses by {relTb} ({relTb / max worst 1e-30}x the tie)."
  IO.println "     ⚠ Duplicated batch, so this covers the accumulate/apply machinery and the 1/k, \
NOT the combination of different micro-batches — see the header."
