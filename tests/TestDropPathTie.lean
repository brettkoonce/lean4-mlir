import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.EfficientNetRender

/-! # Stochastic depth — the two gates that cover the op's INTERIOR

`planning/stochastic_depth.md` §7 lists eight gates. Six were run when the feature landed
(2026-08-02) and they all pin **endpoints**: `dropPath = 0` re-renders every artifact
byte-identically, the keep = 1 train step is bit-identical to AdamW (0 of 4,020,358 against a 0
floor, real recipe firing at 1.89), and `tests/TestDropPathRamp.lean` pins the keep ramp across the
driver/renderer seam. **Nothing yet checks what happens strictly between those endpoints**, and
that is what this file is for:

| gate | what it establishes | why nothing else can |
|---|---|---|
| **A — the known answer** | a supplied scale multiplies the branch by EXACTLY that, per example | every existing tie compares the render against itself or against a peer built from the SAME constants (§6.5 of that doc). Only a host-computed answer sees a wrong multiply |
| **B — the all-zero-mask control** | the site is on the RESIDUAL BRANCH, not on the block output | `out = s·(branch + x)` compiles, trains and descends. It is a different net, and no structural check distinguishes it from `out = s·branch + x` |

    lake build droppath-tie
    scripts/det_shim.sh /tmp/detshim
    LD_LIBRARY_PATH=/tmp/detshim CUDA_VISIBLE_DEVICES=0 .lake/build/bin/droppath-tie
    .lake/build/bin/droppath-tie --op     # gate A only (tiny, no artifact needed)
    .lake/build/bin/droppath-tie --net    # gate B only
    .lake/build/bin/droppath-tie --net --eval        # at @efficientnet_drop_fwd_eval
    .lake/build/bin/droppath-tie --op --break        # gate A is falsifiable: a 1% wrong scale
    .lake/build/bin/droppath-tie --net --cand <misplaced.mlir>   # gate B goes red; expect rc=1

⚠ **Run it under `scripts/det_shim.sh`.** Gate B compares two DIFFERENT HLO programs
(`efficientnet_drop_fwd` against the drop-free `efficientnet_fwd`), and on CUDA the committed
compile options autotune — §2d.3's Finding 1 ("the floor IS bit-exact across processes") is
ROCm-specific. The harness measures its own A-vs-A floor first and says so; without the det shim
that floor is noise and the bit-exact claim in B1 has no resolution.

**What this does NOT establish.** Gate A is SITE-LOCAL: it drives the op through the same `pretty`
emitter the render uses, at small dims, against a closed form recomputed from the inputs — it says
the op multiplies correctly, not that the real net wires it anywhere in particular. Gate B is
whole-net but its logits are a NONLINEAR function of the site (BN, swish, SE downstream), so it
cannot read `branch · scale` off the output; what it can do is falsify the placements, which is a
different and weaker claim. The two halves are independent and both are needed — the same split
§2b records for the artifact-vs-theorem pair.
-/

open Proofs Proofs.StableHLO

/-- The driver's own init (`VerifiedTrain.mkParam`, which is private): He(fan-in) weights, γ = 1,
    β/bias = 0. A constant-splat parameter set makes BN see zero variance, and two wrong renders
    would then agree on the resulting garbage. -/
private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.const n.toUSize 0.0
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

/-- A float32 buffer from explicit values. Used for masks, which are tiny and must be exact. -/
private def mkVec (vals : Array Float) : IO ByteArray := do
  let mut cells : Array ByteArray := #[]
  for v in vals do cells := cells.push (← F32.const 1 v)
  pure (F32.concat cells)

/-- `(max |a−b|, max(|a|,|b|), bit-exact count)` over `n` coordinates. The exact count is not
    decoration: `Float.toString` gives six decimals, so a genuine 3e-8 prints as `0.000000` and
    reads as bit-exact when it is not (§2e-bis).

    ⚠ **The magnitude is over BOTH buffers, and a first version of this took it over `a` alone.**
    That is not a nicety — the misplaced-render control (`--cand`, below) drove `a` to identically
    zero, so the denominator went to zero, `rel` returned `0.0`, and a TOTAL COLLAPSE of the logits
    was reported as *"the logits did not move"*. The gate then fired on the wrong check with a
    message naming the wrong cause. `fwd-tie` already normalises over both; this is that, learned
    again. A control does not only prove the gate can go red — it proves the gate goes red for the
    reason it claims. -/
private def cmpBufs (a b : ByteArray) (n : Nat) : Float × Float × Nat := Id.run do
  let mut d := 0.0
  let mut m := 0.0
  let mut e := 0
  for i in [0:n] do
    let u := F32.read a i.toUSize
    let w := F32.read b i.toUSize
    if u == w then e := e + 1
    if (u - w).abs > d then d := (u - w).abs
    if max u.abs w.abs > m then m := max u.abs w.abs
  (d, m, e)

private def rel (d m : Float) : Float := if m > 1e-30 then d / m else 0.0

private def nonFinite (a : ByteArray) (n : Nat) : Nat := Id.run do
  let mut c : Nat := 0
  for i in [0:n] do
    if !(F32.read a i.toUSize).isFinite then c := c + 1
  return c

/-- Compile `src` fresh. Deletes both the bare and the `_$IREE_BACKEND`-scoped `.vmfb` first (§4):
    `compileVmfb` reuses any output newer than the `.mlir`, keyed on the output path and an mtime
    rather than on the source, so a second run under one tag silently reuses the first binary. -/
private def freshSession (path tag : String) : IO IreeSession := do
  let vmfb := s!".lake/build/droppath_tie_{tag}.vmfb"
  let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
  for p in [vmfb, s!".lake/build/droppath_tie_{tag}_{target}.vmfb"] do
    if ← System.FilePath.pathExists p then IO.FS.removeFile p
  mkSession path vmfb

-- ════════════════════════════════════════════════════════════════
-- § GATE A — the known answer. Does a supplied scale multiply the branch by EXACTLY that?
-- ════════════════════════════════════════════════════════════════

private def OB : Nat := 8      -- examples. The mask is per-EXAMPLE, so this is the axis under test
private def ON : Nat := 5      -- per-example width. ⚠ ON ≠ OB deliberately — see `opModule`

private def zOX : Vec (OB * ON) := fun _ => 0

/-- **The op, through the SAME `pretty` emitter the render uses** — not a restatement of it. One
    `dropPathB` over `%x` with the mask as a graph input, which is exactly the subtree
    `eFwd` splices onto the residual branch.

    ⚠ `ON ≠ OB` on purpose. With `n = B` a mask broadcast along the wrong axis still typechecks,
    and the whole point of this probe is that the scale is indexed by EXAMPLE. At `8 × 5` a
    `dims = [1]` broadcast of a `tensor<8xf32>` onto `tensor<8x5xf32>` is a hard type error rather
    than a silent different function — so the failure this gate is left to catch is the subtler
    one the op's own docstring names: a `BatchableOp` descriptor, which would denote "every example
    shares example 0's mask". That is control C1 below. -/
private def opModule : String :=
  let go : StateM Nat String := do
    let (c, o) ← pretty OB (.dropPathB (N := OB) (n := ON) "%dp" (fun _ => 0 : Vec OB)
                              (.operand "%x" zOX))
    pure (c ++ s!"    return {o} : {ty [OB, ON]}\n")
  let body : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @dp(%x: {ty [OB, ON]}, %dp: {ty [OB]}) -> {ty [OB, ON]} " ++ "{\n" ++
  body ++ "  }\n}\n"

/-- The closed form, recomputed from the inputs: `y[j·n + i] = s[j] · x[j·n + i]`.

    **Bit-exact is the right bar, and that is an argument rather than a hope.** Both operands are
    float32, so their exact product needs at most 48 mantissa bits and is therefore EXACT in the
    float64 Lean computes it in; rounding that exact product to float32 is by definition what
    float32 multiplication returns. No double rounding, so any difference at all is the render
    computing something else. ⚠ The mask is read back out of the buffer the device receives, never
    from the literal, so both sides see the identical float32 value. -/
private def dropRef (x s : ByteArray) : IO ByteArray := do
  let mut cells : Array ByteArray := #[]
  for j in [0:OB] do
    let sj := F32.read s j.toUSize
    for i in [0:ON] do
      cells := cells.push (← F32.const 1 (sj * F32.read x (j * ON + i).toUSize))
  pure (F32.concat cells)

/-- **Control C1** — the descriptor bug: every example scaled by example 0's mask. A
    `BatchableOp` descriptor may carry only batch-INVARIANT data (§4), and this is what one would
    denote here. It must NOT match, which is why the mask below is non-uniform. -/
private def dropRefBroadcast0 (x s : ByteArray) : IO ByteArray := do
  let s0 := F32.read s 0
  let mut cells : Array ByteArray := #[]
  for k in [0:OB * ON] do
    cells := cells.push (← F32.const 1 (s0 * F32.read x k.toUSize))
  pure (F32.concat cells)

/-- The mask under test. Three kinds of value in one vector, so one invoke covers three claims:

* **`1/keep_i` — the values the driver ACTUALLY supplies.** `F32.dropScales` emits
  `bernoulli(keep_i)/keep_i`, i.e. `0` or `1/keep_i`, and `1/keep_i > 1` for every site. A gate
  written only over `(0,1)` would test a range the feature never uses;
* **the endpoints `1.0` and `0.0`** — `dropPath_ones_id` and `dropPath_zeros_zero`, on device;
* **interior values `0.5 / 0.25 / 0.75`** — exactly representable in binary, so the known answer
  is not testing the harness's own rounding.

⚠ Non-uniform, and index 0 is deliberately NOT the zero: control C1 compares against "everything
scaled by `s[0]`", which a zero at `s[0]` would turn into the trivially-different all-zero vector. -/
private def opMask (keeps : Array Float) : Array Float :=
  let inv := fun (i : Nat) => if h : i < keeps.size then 1.0 / keeps[i] else 1.0
  #[inv 0, 0.0, 1.0, inv 8, 0.5, 0.25, inv 4, 0.75]

private def gateOp (doBreak : Bool) : IO Unit := do
  IO.println "── GATE A — the known answer: does a supplied scale multiply the branch by exactly that?"
  let keeps := efficientnetVerified.dropKeeps
  let maskVals := opMask keeps
  IO.println s!"  B {OB}, n {ON} (n ≠ B on purpose), backend {← IreeSession.backendName}"
  IO.println s!"  mask (per example) = {maskVals}"
  IO.println s!"    ↑ 1/keep at sites 0/4/8 of the real ramp {keeps.size} sites, plus both endpoints \
and three interior values"
  let x ← F32.heInit 20260802 (OB * ON).toUSize 1.0
  let s ← mkVec maskVals
  let path := "/tmp/droppath_op.mlir"
  IO.FS.writeFile path opModule
  let sess ← freshSession path "op"
  let y ← IreeSession.forwardF32 sess "m.dp" s (packShapes #[#[OB]]) x
            (packXShape #[OB, ON]) OB.toUSize ON.toUSize
  let n := OB * ON
  if nonFinite y n > 0 then
    throw (IO.userError "DEGENERATE: non-finite output — the gate proves nothing")

  -- ── the gate ──
  let ref ← dropRef x s
  let (d, m, e) := cmpBufs y ref n
  IO.println s!"  emitted vs host `s[j]·x[j,i]` : max abs {d}   |ref|max {m}   bit-exact {e}/{n}"
  if m < 1e-6 then
    throw (IO.userError "DEGENERATE: the reference is ~0 — the gate proves nothing")

  -- ── the two endpoints, read out separately because they are named theorems ──
  let rowExact (j : Nat) (against : ByteArray) : Nat := Id.run do
    let mut c : Nat := 0
    for i in [0:ON] do
      if F32.read y (j * ON + i).toUSize == F32.read against (j * ON + i).toUSize then c := c + 1
    return c
  let zeroRow := maskVals.findIdx? (· == 0.0)
  let onesRow := maskVals.findIdx? (· == 1.0)
  match onesRow with
  | some j => IO.println s!"  `dropPath_ones_id`   on device: row {j} (s=1) equals x in \
{rowExact j x}/{ON} coordinates"
  | none => pure ()
  match zeroRow with
  | some j =>
      let z ← F32.const (OB * ON).toUSize 0.0
      IO.println s!"  `dropPath_zeros_zero` on device: row {j} (s=0) is zero in \
{rowExact j z}/{ON} coordinates"
  | none => pure ()

  -- ── C1: the descriptor bug must NOT match ──
  let refB ← dropRefBroadcast0 x s
  let (dB, _, eB) := cmpBufs y refB n
  IO.println s!"  ⚠ CONTROL C1 (every example gets s[0] — the descriptor bug): max abs {dB} \
(rel {rel dB m}), bit-exact {eB}/{n}"
  if rel dB m < 1e-3 then
    throw (IO.userError s!"CONTROL C1 FAILED: scaling every example by s[0] agrees with the render \
(rel {rel dB m}). Either the mask is uniform (it must not be) or the emitted broadcast is not \
per-example — either way the gate above means nothing.")

  -- ── C2: prove the gate can go red ──
  if doBreak then
    let bad := maskVals.set! 4 (maskVals[4]! * 1.01)
    let sBad ← mkVec bad
    let refBad ← dropRef x sBad
    let (dX, _, eX) := cmpBufs y refBad n
    IO.println s!"  ⚠ CONTROL C2 (--break: one scale perturbed by 1%): max abs {dX} \
(rel {rel dX m}), bit-exact {eX}/{n}"
    if eX == n then
      throw (IO.userError "CONTROL C2 FAILED: a 1% wrong scale still reports bit-exact — the \
comparison is not wired to the output")
    IO.println "  ✓ the gate is falsifiable: a 1% wrong scale is caught"

  if e != n then
    throw (IO.userError s!"GATE A FAILED: {n - e} of {n} coordinates differ from the host answer \
(max abs {d}, rel {rel d m}). A single f32 multiply is exact, so this is not tolerance — the \
emitted op computes something other than `s[j]·x[j,i]`.")
  IO.println s!"✓ GATE A: the emitted scale is EXACTLY `s[j]·x[j,i]`, bit-exact on all {n} \
coordinates, at values spanning 0, 1 and the real 1/keep ramp"

-- ════════════════════════════════════════════════════════════════
-- § GATE B — the all-zero-mask control. Is the site on the BRANCH or on the BLOCK OUTPUT?
-- ════════════════════════════════════════════════════════════════

/-! The renderer splices the drop op onto the residual branch, *before* the skip add
(`EfficientNetRender.eFwd`), which is where the reference puts it:

    correct      block_out = s ⊙ branch + x
    misplaced    block_out = s ⊙ (branch + x)

Both compile, both train, both descend, and **every structural check in the repo is blind to the
difference** — same op count, same arity, same types, same prefix. What separates them is what a
ZERO mask does: the correct one leaves the block an IDENTITY, the misplaced one annihilates the
signal. So the three checks below are all statements about a zeroed site, and each is impossible
under the misplacement:

* **B2** zeroing a site moves the logits at all (the mask reaches the site);
* **B3** with an upstream site already zeroed, zeroing a DOWNSTREAM one *still* moves the logits.
  Under the misplacement the tensor is already identically zero at the upstream site, so the
  downstream mask cannot matter and the two runs would agree **bit-exactly**;
* **B4** with every site zeroed the net still DEPENDS ON ITS INPUT. Under the misplacement the
  first site (block 2, near the stem) zeroes the activation and every later layer is a function of
  zero, so the logits are constant in `x`.

B3 and B4 are the load-bearing pair: B2 alone is satisfied by both placements.
-/

private structure NetRun where
  sess   : IreeSession
  params : ByteArray      -- params (+ BN stats when eval), WITHOUT the trailing masks
  shapes : ByteArray      -- the matching shape table, masks included
  bs     : Nat
  nOut   : Nat

private def uniform (bs : Nat) (v : Float) : Array Float := Array.replicate bs v

private def gateNet (isEval : Bool) (cand : Option String) : IO Unit := do
  let fn   := if isEval then "efficientnet_drop_fwd_eval" else "efficientnet_drop_fwd"
  let refFn := if isEval then "efficientnet_fwd_eval" else "efficientnet_fwd"
  -- ⚠ `--cand` is what makes a green run BELIEVABLE, and it is the `vit-dp-check` lesson (§2j):
  -- that harness hardcoded both paths and took no argv, so its bit-exact PASS was UNFALSIFIABLE
  -- until an argument was added and the sum-not-mean control built. Point this at a render with
  -- the drop moved onto the block output and B3/B4 must go red.
  let dropPath := cand.getD s!"verified_mlir/{fn}.mlir"
  IO.println s!"── GATE B — the all-zero-mask control, at @{fn}"
  if cand.isSome then IO.println s!"  ⚠ CANDIDATE render: {dropPath}"
  let spec := efficientnetVerified
  let net := spec.toNet
  let bs := 32
  let sites := enetDropIdxs
  IO.println s!"  {net.specs.size} params, {sites.length} drop sites {sites}, bs {bs}, \
backend {← IreeSession.backendName}"

  -- ── one deterministic (θ, x) every run sees ──
  let mut parts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    parts := parts.push (← mkParam sd dims kind)
    sd := sd + 1
  let mut params := F32.concat parts
  let mut shapeList := net.paramShapes
  if isEval then
    let mut stats : Array ByteArray := #[]
    let mut ss := 5000
    for c in net.bnChannels do
      stats := stats.push (← F32.heInit ss.toUSize c.toUSize 0.30)                        -- μ
      stats := stats.push (← F32.scaleShift (← F32.heInit (ss+1).toUSize c.toUSize 0.20)
                               1.0 1.0)                                                    -- var ≈ 1 ± 0.2
      ss := ss + 2
    params := F32.concat (#[params] ++ stats)
    shapeList := shapeList ++ net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
    IO.println s!"  + {net.bnChannels.size} BN layers of frozen stats"
  let dropShapes := (List.replicate sites.length (#[bs] : Array Nat)).toArray
  let x  ← F32.heInit 987654 (bs * net.d0).toUSize 1.0
  let x2 ← F32.heInit 135791 (bs * net.d0).toUSize 1.0
  let n := bs * net.nClasses

  let mkRun (tag : String) : IO NetRun := do
    let sess ← freshSession dropPath tag
    pure { sess, params, shapes := packShapes (shapeList ++ dropShapes), bs, nOut := net.nClasses }
  let runAt (r : NetRun) (xx : ByteArray) (masks : Array (Array Float)) : IO ByteArray := do
    let mut cells : Array ByteArray := #[r.params]
    for m in masks do cells := cells.push (← mkVec m)
    IreeSession.forwardF32 r.sess s!"m.{fn}" (F32.concat cells) r.shapes xx
      (packXShape #[r.bs, net.d0]) r.bs.toUSize r.nOut.toUSize

  let ones  := (List.replicate sites.length (uniform bs 1.0)).toArray
  let zeroAt := fun (js : List Nat) =>
    (sites.zipIdx.map (fun (_, i) => if js.contains i then uniform bs 0.0 else uniform bs 1.0)).toArray

  let rA ← mkRun "netA"
  let rB ← mkRun "netB"       -- a SECOND compile of the same artifact — the A-vs-A floor

  -- ── B0: the floor, measured BEFORE any cross-graph number is read (finding 3, 2026-08-02) ──
  let yOnesA ← runAt rA x ones
  let yOnesB ← runAt rB x ones
  let (dF, mF, eF) := cmpBufs yOnesA yOnesB n
  IO.println s!"  FLOOR  same artifact, two compiles : max abs {dF}   bit-exact {eF}/{n}   \
(rel {rel dF mF})"
  if nonFinite yOnesA n > 0 then
    throw (IO.userError "DEGENERATE: non-finite logits — every gate below proves nothing")
  if mF < 1e-6 then
    throw (IO.userError s!"DEGENERATE: logits are all ~0 (|max| {mF}) — every gate below proves nothing")
  if eF != n then
    IO.println s!"  ⚠ the floor is NOT bit-exact ({n - eF} of {n} differ). On CUDA that is \
autotuning, not the render — re-run under `scripts/det_shim.sh` or read B1 as a bound, not as an \
identity."

  -- ── B1: the ones mask is the exact identity, against the DROP-FREE committed forward ──
  let rRef ← freshSession s!"verified_mlir/{refFn}.mlir" "ref"
  let yRef ← IreeSession.forwardF32 rRef s!"m.{refFn}" params (packShapes shapeList) x
               (packXShape #[bs, net.d0]) bs.toUSize net.nClasses.toUSize
  let (d1, m1, e1) := cmpBufs yOnesA yRef n
  IO.println s!"  B1  ones mask vs drop-free @{refFn} : max abs {d1}   bit-exact {e1}/{n}   \
(rel {rel d1 m1})"
  if e1 != n && eF == n then
    throw (IO.userError s!"B1 FAILED: at an all-ones mask the drop render must be BIT-EXACT to the \
drop-free forward ({n - e1} of {n} differ, rel {rel d1 m1}) — `1 * x = x` is exact in IEEE, and \
the floor above IS bit-exact, so this is graph-attributable.")

  -- ── B2: the mask reaches the site ──
  let k  := 0                      -- the FIRST site (block 2) — near the stem, so everything is downstream
  let k' := sites.length - 1       -- the LAST site (block 14)
  let yZk ← runAt rA x (zeroAt [k])
  let (d2, m2, _) := cmpBufs yZk yOnesA n
  let magZk := (cmpBufs yZk yZk n).2.1        -- max |logits| of the zeroed run, on its own
  IO.println s!"  B2  zero at site {k} (block {sites[k]!}) vs ones : rel {rel d2 m2}   \
|logits|max {magZk}"
  -- ⚠ The COLLAPSE check goes first, or its symptom is misread as the one below. Site 0 is block 2,
  -- near the stem: a drop on the BLOCK OUTPUT zeroes the activation there and every later layer is
  -- a function of zero, so the logits go to ~0 rather than merely moving. Reporting that as "the
  -- mask is not reaching the site" would name the wrong cause with the right verdict.
  if magZk < 1e-6 then
    throw (IO.userError s!"B2 FAILED — THE PLACEMENT IS WRONG: zeroing site {k} COLLAPSES the \
logits to ~0 (|max| {magZk}). A zeroed per-example scale on a residual BRANCH leaves the block an \
identity and the net intact; annihilating the signal is what scaling the BLOCK OUTPUT does.")
  if rel d2 m2 < 1e-3 then
    throw (IO.userError s!"B2 FAILED: zeroing site {k} barely moves the logits (rel {rel d2 m2}) — \
the mask input is not reaching that site.")

  -- ── B3: an upstream zero must NOT absorb a downstream site ──
  let yZkk ← runAt rA x (zeroAt [k, k'])
  let (d3, m3, e3) := cmpBufs yZkk yZk n
  IO.println s!"  B3  + also zero at site {k'} (block {sites[k']!}) : rel {rel d3 m3}   \
bit-exact {e3}/{n}"
  if rel d3 m3 < 1e-3 then
    throw (IO.userError s!"B3 FAILED — THE PLACEMENT IS WRONG: with site {k} zeroed, zeroing site \
{k'} changes nothing (rel {rel d3 m3}, {e3}/{n} bit-exact). That is the signature of a drop scaling \
the BLOCK OUTPUT rather than the residual branch: the activation is already identically zero at the \
upstream site, so no later mask can matter. The render claims `s ⊙ branch + x`.")

  -- ── B4: with every site zeroed the net must still depend on its input ──
  let allZero := (List.replicate sites.length (uniform bs 0.0)).toArray
  let yZ  ← runAt rA x  allZero
  let yZ2 ← runAt rA x2 allZero
  let (d4, m4, e4) := cmpBufs yZ yZ2 n
  IO.println s!"  B4  all {sites.length} sites zeroed, two different x : rel {rel d4 m4}   \
|logits|max {m4}   bit-exact {e4}/{n}"
  if m4 < 1e-6 then
    throw (IO.userError s!"B4 FAILED — THE PLACEMENT IS WRONG: with every site zeroed the logits \
collapse to ~0 (|max| {m4}). Zeroing a per-example scale on a RESIDUAL BRANCH leaves each block an \
identity and the rest of the net intact; collapsing to zero is what scaling the BLOCK OUTPUT does.")
  if rel d4 m4 < 1e-3 then
    throw (IO.userError s!"B4 FAILED — THE PLACEMENT IS WRONG: with every site zeroed the logits no \
longer depend on x (rel {rel d4 m4} between two different inputs). The first site is near the stem, \
so a drop on the BLOCK OUTPUT would make every later layer a function of zero.")

  IO.println s!"✓ GATE B: the ones mask is the identity ({e1}/{n} bit-exact against the drop-free \
forward), and a zeroed site leaves its block an IDENTITY rather than annihilating it — B3 \
{rel d3 m3} and B4 {rel d4 m4}, both impossible if the scale sat on the block output"

def main (args : List String) : IO Unit := do
  let doOp   := args.contains "--op" || !(args.contains "--net")
  let doNet  := args.contains "--net" || !(args.contains "--op")
  let isEval := args.contains "--eval"
  IO.println "stochastic depth — the interior gates (planning/stochastic_depth.md §7)"
  let ldPath := (← IO.getEnv "LD_LIBRARY_PATH").getD ""
  if (ldPath.splitOn "detshim").length == 1 then
    IO.println "  ⚠ no `detshim` on LD_LIBRARY_PATH — gate B compares two HLO programs and the \
committed compile options AUTOTUNE on CUDA. Run `scripts/det_shim.sh /tmp/detshim` first, or read \
B1 as a bound."
  let cand := match args.dropWhile (· != "--cand") with
    | _ :: p :: _ => some p
    | _ => none
  if doOp  then gateOp (args.contains "--break")
  if doNet then gateNet isEval cand
