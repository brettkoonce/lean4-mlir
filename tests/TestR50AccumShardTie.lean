import LeanMlir.VerifiedNets

/-! # Accumulation over DIFFERENT micro-batches — the identity `r50-accum-tie` is blind to

`lake build r50-accum-tie` runs k micro-steps on the **same** batch, so every micro-gradient is the
same `g`. That gates the accumulate/apply machinery and the `1/k`, and it is structurally blind to
whether different micro-batches are combined correctly — the same hole every `*-dp-check` has and
that `shard-check` exists to close one level up. Its own header says so. This is the other half.

## ⭐ The identity, and why it is EXACT rather than approximate

The obvious complement — "k micro-batches of b == one step at batch k·b" — is **false here, by
design**: R50 normalises over whatever batch the forward sees, so k micro-batches give k separate
BatchNorm groups where one big batch gives one. That is Ghost-BN, it is what the JAX reference
takes, and asserting the naive identity would be wrong rather than strict.

⭐⭐ **But R50 already has a render that computes exactly the same thing: the DATA-PARALLEL one.**
`emitGradAllReduce` averages k replicas' gradients, and `shard-check`'s docstring records the fact
that makes this work — *"each replica normalises over its own b rows"*. So:

| | groups | combination |
|---|---|---|
| `acc4x64` × 4 micro-batches | 4 BN groups of 64 | `Σgᵢ`, then `1/k` folded into `%ob1`/`%ob2` |
| `adamdp64` × 1 step | 4 BN groups of 64 | `all_reduce(add)` then `/4` |

**Same function.** Ghost-BN over k micro-batches on one device and per-replica BN over k replicas
are the same grouping of the same data, so

    acc(x₁, x₂, x₃, x₄)  ==  adamdp( [x₁|x₂|x₃|x₄] )

to fp rounding — and the two sides reach it through completely different machinery: a serial
accumulator against a collective, a folded `1/k` against a divide, a fourth blob region against
none. Neither is a re-derivation of the other.

⚠ **This compares θ', m' AND v'.** `shard-check` can only compare `m` (it averages two *separately
optimised* single-device steps, and AdamW is nonlinear in the gradient, so `θ'` would be
meaningless). Here nothing is averaged after the fact: both sides form the same `ĝ` and then run
the same AdamW on it, so every region is comparable and `v'` — the quadratic one, where a shared
`1/k` instead of `1/k²` shows up — is included.

## ⟂ The control is the blind spot itself

Re-run the accumulation with **x₁ four times** — exactly what `r50-accum-tie` does — and require it
to MISS. That is the statement that this harness sees *which* micro-batch is which, i.e. that it
closes the hole rather than restating the gate it complements.

Needs FOUR GPUs and the XLA backend (collectives do not exist on the IREE path).

    lake build r50-accum-shard-tie
    CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_REPLICAS=4 .lake/build/bin/r50-accum-shard-tie

⚠ `PJRT_REPLICAS` is required and is NOT redundant with `CUDA_VISIBLE_DEVICES`: it is what makes
the shim compile a module for more than one device. It is safe to set here even though half this
harness is single-device, because the shim decides PER SESSION — `reps = (g_replicas > 1 &&
strstr(mlir, "all_reduce")) ? g_replicas : 1` — and the accumulation render contains no collective.
Without it the DP invoke refuses outright ("asked for 4 replicas but … was compiled for 1"), which
is the right failure: a silent single-device run would have tied against itself.

Knobs: `R50_ACC_VARIANT` (default `acc4x64`; k is read from the name and must equal the replica
count), `R50_ACC_TOL_U` (micro-units, default 200 = 2e-4).
-/

/-- Labels for one micro-batch: class `(i + off) % nClasses`, the driver's 4-byte records. The
    offset makes the shards differ in their LABELS as well as their pixels, so a run that read the
    wrong rows cannot coincidentally agree. -/
private def mkLabels (bs off nc : Nat) : ByteArray := Id.run do
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat ((i + off) % (min nc 251))); y := y.push 0; y := y.push 0; y := y.push 0
  y

private def cmp (a b : ByteArray) (off n : Nat) : Float × Float × Nat := Id.run do
  let mut d := 0.0; let mut m := 0.0; let mut ex := 0
  for i in [0:n] do
    let x := F32.read a (off + i).toUSize
    let y := F32.read b (off + i).toUSize
    if x == y then ex := ex + 1
    if (x - y).abs > d then d := (x - y).abs
    if x.abs > m then m := x.abs
  return (d, m, ex)

def main : IO Unit := do
  -- ▶ Resolution selects the NET (slug + `d0`); the 160 artifacts are their own family. Parameter
  -- layout is identical by construction (`VerifiedNets.lean` `#guard`s `toSpecs` equal), so only
  -- the input width moves.
  let net ← match (← IO.getEnv "R50_ACC_RES").getD "224" with
    | "224" => pure resnet50ImagenetVerified.toNet
    | "160" => pure resnet50Imagenet160Verified.toNet
    | r     => throw <| IO.userError s!"R50_ACC_RES={r}: want 224 or 160"
  let bs   := 64
  let nP   := net.nParams
  let variant := (← IO.getEnv "R50_ACC_VARIANT").getD "acc4x64"
  -- ⚠⚠ The DP peer must match `variant` on optimizer, loss and resolution — see the header. For the
  -- composed RSB-A3 render that is `lambdp64bce`, NOT `adamdp64`.
  let peer := (← IO.getEnv "R50_ACC_PEER").getD "adamdp64"
  if peer == variant then
    throw <| IO.userError s!"R50_ACC_PEER == R50_ACC_VARIANT ('{peer}') — the tie would compare the \
artifact to itself and pass unconditionally"
  let tol  := (((← IO.getEnv "R50_ACC_TOL_U").bind (·.toNat?)).map (fun u => u.toFloat * 1e-6)
                |>.getD 2.0e-4)
  -- ⚠ SUBSTRING parse — `lambacc4x64bce` does not lead with the marker (defect #4).
  let k :=
    let after := (variant.splitOn "acc").getD 1 ""
    let after := if after.startsWith "dp" then after.drop 2 else after
    ((after.takeWhile (· != 'x')).toNat?).getD 0
  if k < 2 then
    throw <| IO.userError s!"could not read k from variant '{variant}' (want [<opt>]acc<k>x<B>[<loss>])"
  let lr := 0.01
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let pShapes := net.paramShapes

  IO.println s!"§4's accumulation over DIFFERENT micro-batches — the identity r50-accum-tie is blind to"
  IO.println s!"  acc(x1..x{k})  ==  DP([x1|..|x{k}])   ({k} micro-batches of {bs} vs {k} replicas x {bs})"
  IO.println s!"  accumulation : verified_mlir/{net.slug}_{variant}_train_step.mlir"
  IO.println s!"  data-parallel: verified_mlir/{net.slug}_{peer}_train_step.mlir"
  IO.println s!"  {net.specs.size} params ({nP} floats), lr {lr}, backend {← LowererSession.backendName}"

  -- ── one (θ, m, v) both sides see. m and v are non-zero: at m = v = 0 the β₁/β₂ passthrough
  --    terms vanish and a render that dropped them would still tie. ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  let mIn ← F32.scaleShift (← F32.heInit 4242 nP.toUSize 0.02) 1.0 0.05
  let vIn ← F32.scaleShift (← F32.heInit 7373 nP.toUSize 0.001) 1.0 0.02
  let gIn ← F32.const nP.toUSize 0.0
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let bc1 := 1.0 - 0.9
  let bc2 := 1.0 - 0.999

  -- ── k genuinely different micro-batches, and their concatenation in shard order ──
  -- ⚠ The DP shim splits `x` by ROWS: replica r takes `[r*bs, (r+1)*bs)`. So the concatenation
  -- order below IS the replica assignment, and it must match the order the accumulation visits.
  let mut xs : Array ByteArray := #[]
  let mut ys : Array ByteArray := #[]
  for i in [0:k] do
    xs := xs.push (← F32.heInit (555 + 444 * i).toUSize (bs * net.d0).toUSize 1.0)
    ys := ys.push (mkLabels bs (5 * i) net.nClasses)
  let xAll := F32.concat xs
  let mut yAll : ByteArray := .empty
  for y in ys do yAll := yAll ++ y

  -- Delete both the bare and the backend-scoped .vmfb first (§4): `compileVmfb` keys its cache on
  -- the OUTPUT path and an mtime, never the source, so a re-run under the same tag with a
  -- different candidate silently reuses the first.
  let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
  for tag in ["r50_accshard_acc", "r50_accshard_dp"] do
    for p in [s!".lake/build/{tag}.vmfb", s!".lake/build/{tag}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p

  let accShapes := packShapes (pShapes ++ pShapes ++ pShapes ++ pShapes
                               ++ Array.replicate 5 #[] ++ bnStatShapes)
  let dpShapes  := packShapes (pShapes ++ pShapes ++ pShapes
                               ++ #[#[], #[], #[]] ++ bnStatShapes)
  let accSess ← mkSession s!"verified_mlir/{net.slug}_{variant}_train_step.mlir"
  let dpSess  ← mkSession s!"verified_mlir/{net.slug}_{peer}_train_step.mlir"

  /- One accumulation cycle over `batches`, applying on the last micro-batch. -/
  let cycle (batches : Array (ByteArray × ByteArray)) : IO ByteArray := do
    let mut buf := F32.concat #[θ, mIn, vIn, gIn, ← F32.const 5 0.0, bnIn]
    for i in [0:batches.size] do
      let applyNow := i + 1 == batches.size
      buf ← F32.write3 buf (4 * nP).toUSize (if applyNow then lr else 0.0) bc1 bc2
      let pair ← F32.write3 (← F32.const 3 0.0) 0
                   (if applyNow then 1.0 else 0.0) (if i == 0 then 0.0 else 1.0) 0.0
      buf ← F32.blit buf (4 * nP + 3).toUSize pair 0 2
      buf ← F32.blit buf (4 * nP + 5).toUSize bnIn 0 nBnStats.toUSize
      buf ← LowererSession.mlpTrainStepV accSess s!"m.{net.slug}_{variant}_train_step"
              batches[i]!.1 buf accShapes batches[i]!.2
              bs.toUSize net.d0.toUSize net.nClasses.toUSize
    return buf

  IO.println s!"  accumulating {k} different micro-batches…"; (← IO.getStdout).flush
  let accOut ← cycle ((Array.range k).map (fun i => (xs[i]!, ys[i]!)))
  IO.println s!"  one {k}-replica data-parallel step on their concatenation…"; (← IO.getStdout).flush
  let dpBuf := F32.concat #[θ, mIn, vIn, ← F32.write3 (← F32.const 3 0.0) 0 lr bc1 bc2, bnIn]
  let dpOut ← LowererSession.mlpTrainStepVDP dpSess s!"m.{net.slug}_{peer}_train_step"
                xAll dpBuf dpShapes yAll (k * bs).toUSize net.d0.toUSize net.nClasses.toUSize
                k.toUSize
  -- ⟂ the control: the DUPLICATED batch, i.e. exactly what `r50-accum-tie` runs.
  IO.println s!"  ⟂ control: the same cycle on x1 repeated {k} times…"; (← IO.getStdout).flush
  let dupOut ← cycle (Array.replicate k (xs[0]!, ys[0]!))

  let (dT, mT, eT) := cmp accOut dpOut 0        nP
  let (dM, mM, eM) := cmp accOut dpOut nP       nP
  let (dV, mV, eV) := cmp accOut dpOut (2 * nP) nP
  let (dC, _,  _)  := cmp dupOut dpOut 0        nP
  let (dMove, _, _) := cmp accOut θ 0 nP        -- θ' vs θ: did anything step at all?
  let relT := dT / max mT 1e-30
  let relM := dM / max mM 1e-30
  let relV := dV / max mV 1e-30
  let relC := dC / max mT 1e-30
  let worst := max relT (max relM relV)
  IO.println ""
  IO.println s!"  θ'  max abs Δ {dT}  rel {relT}   bit-exact {eT}/{nP}"
  IO.println s!"  m'  max abs Δ {dM}  rel {relM}   bit-exact {eM}/{nP}"
  IO.println s!"  v'  max abs Δ {dV}  rel {relV}   bit-exact {eV}/{nP}"
  IO.println s!"  (θ' moved from θ by {dMove} — the degeneracy guard)"
  IO.println s!"  ⟂ DUPLICATED-batch control  θ' rel {relC}"

  IO.println ""
  if dMove < 1e-6 then
    throw <| IO.userError s!"DEGENERATE: θ' == θ (max Δ {dMove}) — both sides agree on doing nothing"
  if worst > tol then
    throw <| IO.userError s!"TIE FAILED: {k} accumulated micro-batches != one {k}-replica step on \
their concatenation (θ' rel {relT}, m' rel {relM}, v' rel {relV}, tolerance {tol}). The two paths \
group the data identically — Ghost-BN over micro-batches vs per-replica BN — so this is the \
COMBINATION of different micro-batches, not the machinery `r50-accum-tie` already covers"
  if relC <= 10.0 * worst then
    throw <| IO.userError s!"CONTROL DEAD: running the cycle on ONE batch repeated {k} times fits \
as well as the {k} different ones ({relC} vs a tie of {worst}) — this harness cannot see which \
micro-batch is which, which is the entire hole it exists to close"
  IO.println s!"  ✅ CERTIFIED: {k} accumulated micro-batches ARE one {k}-replica data-parallel \
step on their concatenation — θ' rel {relT}, m' rel {relM}, v' rel {relV} — reached by a serial \
accumulator with a folded 1/k on one side and a collective all_reduce on the other, against a \
duplicated-batch control that misses by {relC} ({relC / max worst 1e-30}x the tie)."
  IO.println s!"     Together with `r50-accum-tie` (the machinery) this closes both halves: the \
accumulate/apply switching AND the combination of genuinely different micro-batches."
