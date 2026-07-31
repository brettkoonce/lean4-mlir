import LeanMlir.VerifiedNets

/-! # §2l step B — are the conv biases really inert?

`planning/xla_pjrt_handoff.md` §2l wants the conv biases dropped so the verified R34's parameter
layout lines up with the JAX `.convBn` reference (−8,512 params). Its argument that this is
**layout-only, not functional** is:

> every conv here is immediately followed by BN, and BN subtracts the batch mean, so
> `(x + b) − mean(x + b) = x − mean(x)`; the bias gradient is exactly zero, and since biases are
> zero-initialised it stays 0 forever.

§2l flags that as *"an argument, not a measurement"* and asks for the measurement before anything
relies on it. This is it — and it is sharper than the ten-second version suggested there.

**The construction (§2k's, reused).** Run the committed `resnet34_adam_train_step.mlir` for ONE
step from **`m = v = 0`**. AdamW's first moment is `m' = (1−β₁)·g`, so `m'` recovers the gradient
exactly: `g = 10·m'` at β₁ = 0.9. Reading `m'` is therefore reading the gradient, with no tolerance
argument and no dependence on how θ moved. Checking θ instead would be much weaker — Adam's update
is scale-free, so a zero-gradient parameter still moves ±lr (§3).

**What is checked, and why the controls are the point.** A harness that reported "zero" for every
rank-1 slot would pass the headline claim while proving nothing, so the same run must show the
same-shaped params that are NOT BN-followed moving:

| region | expectation | why |
|---|---|---|
| the 36 **conv biases** (`[c]`, kind 2, index ≡ 1 mod 4) | **exactly 0** | swallowed by the following BN |
| the 36 **conv weights** (rank-4) | non-zero | else the whole step is degenerate |
| BN **γ** (kind 1) and **β** (kind 2) | non-zero | β is the parameter that actually plays the bias role |
| the **dense bias** (`[10]`, kind 2, last slot) | **non-zero** | ⚠ THE CONTROL: same shape and same
  init kind as a conv bias, but it feeds the loss directly with no BN after it. If this were zero
  too, the reading would be "the harness sees zeros", not "BN kills conv-bias gradients" |

⚠ Scope: this measures the **gradient at step 1**, plus zero-init (`ResNet34Layout.specs` kind 2),
which together give the induction — a parameter whose gradient is identically zero and starts at 0
stays at 0, with `m`/`v` at 0 too. It does not measure a trained checkpoint; if you want that
belt-and-braces, dump `[θ|m|v]` from a real run and read the same slots.

    lake build conv-bias-zero && HIP_VISIBLE_DEVICES=0 .lake/build/bin/conv-bias-zero
    #   optional argv[1]: a candidate render to measure instead of the committed one
-/

private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0          -- BN γ
  | 2 => F32.const n.toUSize 0.0          -- BN β / biases
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

/-- Classify param `i` by the R34 layout: 36 groups of `[W, b, γ, β]` (indices 0-143), then the
    dense head `[W, b]` (144, 145). The conv biases are the `≡ 1 (mod 4)` slots below 144; the
    dense bias is 145 and is deliberately NOT one of them. -/
private inductive Slot | convW | convB | bnG | bnB | denseW | denseB
  deriving BEq, Inhabited

private def Slot.idx : Slot → Nat
  | .convW => 0 | .convB => 1 | .bnG => 2 | .bnB => 3 | .denseW => 4 | .denseB => 5

private def classify (nSpecs i : Nat) : Slot :=
  if i == nSpecs - 2 then .denseW
  else if i == nSpecs - 1 then .denseB
  else match i % 4 with
    | 0 => .convW | 1 => .convB | 2 => .bnG | _ => .bnB

private def slotName : Slot → String
  | .convW => "conv W" | .convB => "conv b" | .bnG => "BN γ"
  | .bnB => "BN β" | .denseW => "dense W" | .denseB => "dense b"

/-- `--ckpt <path>`: read a TRAINED `[θ|m|v]` blob instead of running a step, and report how far
    the conv biases actually drifted. This is the belt-and-braces half — the one-step gradient says
    what the math does, this says what 80 epochs of AdamW did with it. Needs no GPU. -/
private def ckptMode (net : VerifiedNet) (path : String) : IO Unit := do
  let blob ← IO.FS.readBinFile path
  let nP := net.nParams
  let nS := net.specs.size
  if blob.size < 3 * nP * 4 then
    throw (IO.userError s!"{path}: {blob.size} bytes, expected ≥ {3*nP*4} for [θ|m|v]")
  IO.println s!"§2l step B — the TRAINED conv biases, from {path}"
  IO.println s!"  {blob.size} bytes = [θ|m|v] over {nP} floats"
  let mut off := 0
  let mut maxBySlot : Array Float := Array.replicate 6 0.0
  let mut cntBySlot : Array Nat := Array.replicate 6 0
  let mut nzBySlot : Array Nat := Array.replicate 6 0
  for i in [0:nS] do
    let sz := net.paramShapes[i]!.foldl (· * ·) 1
    let k := (classify nS i).idx
    for j in [0:sz] do
      let θ := (F32.read blob (off + j).toUSize).abs      -- the θ region
      if θ != 0.0 then nzBySlot := nzBySlot.set! k (nzBySlot[k]! + 1)
      if θ > maxBySlot[k]! then maxBySlot := maxBySlot.set! k θ
    cntBySlot := cntBySlot.set! k (cntBySlot[k]! + sz)
    off := off + sz
  IO.println ""
  IO.println "  region      |θ|max            non-zero coords"
  for sl in [Slot.convW, .convB, .bnG, .bnB, .denseW, .denseB] do
    IO.println s!"  {slotName sl}       {maxBySlot[sl.idx]!}      \
{nzBySlot[sl.idx]!}/{cntBySlot[sl.idx]!}"
  IO.println ""
  if maxBySlot[Slot.convB.idx]! == 0.0 then
    IO.println "  ✅ the trained conv biases are still EXACTLY zero — they never moved."
  else
    IO.println s!"  ⚠ the trained conv biases are NOT zero: |θ|max = {maxBySlot[Slot.convB.idx]!}, \
{nzBySlot[Slot.convB.idx]!}/{cntBySlot[Slot.convB.idx]!} coords moved off 0 — compare BN β at \
{maxBySlot[Slot.bnB.idx]!}. They drifted despite a gradient ~1e-6 of every other parameter's, \
because AdamW's update is SCALE-FREE (§3): m̂/(√v̂+ε) is O(1) however small the gradient is."

/-- `--ablate <ckpt>`: THE GATE THAT DECIDES STEP B. Take the TRAINED θ, run `@resnet34_fwd`, then
    zero the 36 conv-bias slots and run it again. If dropping the biases is layout-only, the two
    logit sets must agree — and that is a claim about the *function*, which is what §2l actually
    needs, rather than about the gradient or the parameter values.

    The control is built in: the same ablation applied to BN β (which BN does NOT remove) must
    move the logits a lot. Without it, "logits unchanged" could just mean the harness re-ran the
    same buffer. -/
private def ablateMode (net : VerifiedNet) (ckpt : String) : IO Unit := do
  let blob ← IO.FS.readBinFile ckpt
  let nP := net.nParams
  let nS := net.specs.size
  if blob.size < nP * 4 then throw (IO.userError s!"{ckpt}: too small for θ")
  IO.println s!"§2l step B — the FORWARD with the trained conv biases vs with them ZEROED"
  IO.println s!"  ckpt  {ckpt}"
  -- θ as trained, and two ablations: conv biases → 0, and (the control) BN β → 0.
  let build (kill : Slot) : IO ByteArray := do
    let mut parts : Array ByteArray := #[]
    let mut off := 0
    for i in [0:nS] do
      let sz := net.paramShapes[i]!.foldl (· * ·) 1
      if classify nS i == kill then parts := parts.push (← F32.const sz.toUSize 0.0)
      else parts := parts.push (blob.extract (off*4) ((off+sz)*4))
      off := off + sz
    pure (F32.concat parts)
  let θ    := blob.extract 0 (nP*4)
  let θnb  ← build .convB        -- conv biases zeroed  → must be inert
  let θnβ  ← build .bnB          -- BN β zeroed         → the control, must NOT be inert
  let bs := 32
  let x ← F32.heInit 777 (bs * net.d0).toUSize 1.0
  let xsh := net.xShape bs
  let vmfb := ".lake/build/conv_bias_ablate.vmfb"
  let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
  for p in [vmfb, s!".lake/build/conv_bias_ablate_{target}.vmfb"] do
    if ← System.FilePath.pathExists p then IO.FS.removeFile p
  let sess ← mkSession s!"verified_mlir/{net.slug}_fwd.mlir" vmfb
  let run (p : ByteArray) : IO ByteArray :=
    IreeSession.forwardF32 sess s!"m.{net.slug}_fwd" p net.shapesBA x xsh
      bs.toUSize net.nClasses.toUSize
  let la ← run θ
  let lb ← run θnb
  let lc ← run θnβ
  let cmp (u w : ByteArray) : Float × Float := Id.run do
    let mut d := 0.0; let mut m := 0.0
    for i in [0:bs * net.nClasses] do
      let a := F32.read u i.toUSize; let b := F32.read w i.toUSize
      if (a-b).abs > d then d := (a-b).abs
      if a.abs > m then m := a.abs
    (d, m)
  let (dNb, mag) := cmp la lb
  let (dNβ, _)   := cmp la lc
  IO.println s!"  |logits|max = {mag}"
  IO.println s!"  conv biases → 0 : max abs Δ = {dNb}   (rel {dNb / mag})"
  IO.println s!"  BN β → 0        : max abs Δ = {dNβ}   (rel {dNβ / mag})   ← control"
  if mag < 1e-6 then throw (IO.userError "DEGENERATE: logits are ~0")
  if dNβ / mag < 1e-3 then
    throw (IO.userError "CONTROL FAILED: zeroing BN β barely moved the logits — the harness is \
not actually re-running with the ablated parameters, so the conv-bias result means nothing")
  if dNb / mag > 1e-4 then
    IO.println s!"  ⛔ zeroing the trained conv biases MOVES the logits (rel {dNb / mag}) — \
dropping them is NOT layout-only"
    throw (IO.userError "conv biases are not functionally inert")
  IO.println s!"  ✅ the trained conv biases are FUNCTIONALLY INERT: zeroing all 8,512 of them \
moves the logits by rel {dNb / mag}, while the same ablation on BN β moves them by {dNβ / mag}."
  IO.println "  ⇒ §2l step B is safe on the FUNCTION even though its stated reason (\"the gradient \
is exactly zero, so they stay 0\") is false in f32 — see the one-step and --ckpt modes."

/-- `--tie <nobias.mlir>`: THE SWAP GATE. Run the committed 146-param render and the candidate
    110-param one on the SAME (θ, m, v, x, y), and compare every shared output slot.

    This should be **bit-exact**, and that is a stronger claim than it looks: the candidate binds
    each conv bias to a zero constant instead of an argument, and `x + 0.0` is exact in IEEE for
    every finite `x`, so the two graphs compute the identical float — not merely a close one. A
    non-zero difference here means the bias is reaching the output some other way.

    Index mapping: the committed layout is 146 params, the candidate drops the 36 conv biases in
    place, so walking both in func-arg order and skipping the bias slots on the left pairs them.
    Regions compared: θ', m', v' (per param), then %loss and the 72 BN running stats. -/
private def tieMode (net : VerifiedNet) (candidate : String) : IO Unit := do
  let bs := 32
  let nS := net.specs.size
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let ref := "verified_mlir/resnet34_adam_train_step.mlir"
  IO.println s!"§2l step B — the swap tie"
  IO.println s!"  A (committed, with conv biases) = {ref}"
  IO.println s!"  B (candidate, no conv biases)   = {candidate}"
  -- shapes/params for each side: B keeps every slot A has EXCEPT the conv biases.
  let mut θA : Array ByteArray := #[]
  let mut θB : Array ByteArray := #[]
  let mut shA : Array (Array Nat) := #[]
  let mut shB : Array (Array Nat) := #[]
  let mut sd := 1234
  for i in [0:nS] do
    let (dims, kind) := net.specs[i]!
    let p ← mkParam sd dims kind
    sd := sd + 1
    θA := θA.push p; shA := shA.push dims
    if classify nS i != .convB then θB := θB.push p; shB := shB.push dims
  let nPA := net.nParams
  let nPB := (shB.map (fun d => d.foldl (· * ·) 1)).foldl (· + ·) 0
  IO.println s!"  A: {nS} params / {nPA} floats     B: {shB.size} params / {nPB} floats"
  -- ⚠ m and v must be the SAME NUMBERS on both sides, which means generating them ONCE at A's
  -- size and slicing B's out of it. Generating each side at its own size looks equivalent and is
  -- not: the buffers are position-indexed, so dropping 36 slots shifts every later value and the
  -- two runs see different moments. That is a harness bug that reads exactly like a render bug —
  -- it produced 36448/68015201 bit-exact before this was fixed.
  let mA ← F32.heInit 4242 nPA.toUSize 0.02
  let vA ← F32.scaleShift (← F32.heInit 8484 nPA.toUSize 0.01) 1.0 0.05
  let dropBias (buf : ByteArray) : ByteArray := Id.run do
    let mut parts : Array ByteArray := #[]
    let mut off := 0
    for i in [0:nS] do
      let sz := net.paramShapes[i]!.foldl (· * ·) 1
      if classify nS i != .convB then parts := parts.push (buf.extract (off*4) ((off+sz)*4))
      off := off + sz
    F32.concat parts
  let tl ← F32.const 3 0.0
  let tl ← F32.write3 tl 0 0.001 0.19 0.002
  let bn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let bufA := F32.concat #[F32.concat θA, mA, vA, tl, bn]
  let bufB := F32.concat #[F32.concat θB, dropBias mA, dropBias vA, tl, bn]
  let shpA := packShapes (shA ++ shA ++ shA ++ #[#[], #[], #[]] ++ bnStatShapes)
  let shpB := packShapes (shB ++ shB ++ shB ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0
  let run (path tag : String) (buf shp : ByteArray) : IO ByteArray := do
    let vmfb := s!".lake/build/cbz_tie_{tag}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for q in [vmfb, s!".lake/build/cbz_tie_{tag}_{target}.vmfb"] do
      if ← System.FilePath.pathExists q then IO.FS.removeFile q
    let sess ← mkSession path vmfb
    IreeSession.mlpTrainStepV sess "m.resnet34_adam_train_step" x buf shp y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  let oA ← run ref "a" bufA shpA
  let oA2 ← run ref "a2" bufA shpA        -- the A-vs-A determinism floor (§4)
  let oB ← run candidate "b" bufB shpB

  -- walk both in func-arg order, skipping A's conv-bias slots
  -- One pass, used for BOTH the A-vs-A floor and the A-vs-B tie, so they are measured the same
  -- way. `nPR` is the right-hand side's param count (nPA for the floor, nPB for the tie).
  let compare (oR : ByteArray) (nPR : Nat) : Float × Float × Nat × Nat × String := Id.run do
    let mut offA := 0; let mut offB := 0
    let mut maxD := 0.0; let mut maxM := 0.0
    let mut exact := 0; let mut total := 0; let mut worst := ""
    for i in [0:nS] do
      let sz := net.paramShapes[i]!.foldl (· * ·) 1
      if classify nS i == .convB && nPR != nPA then offA := offA + sz else
        for r in [0:3] do                                   -- θ', m', v'
          for j in [0:sz] do
            let a := F32.read oA (r * nPA + offA + j).toUSize
            let b := F32.read oR (r * nPR + offB + j).toUSize
            total := total + 1
            if a == b then exact := exact + 1
            if (a-b).abs > maxD then maxD := (a-b).abs; worst := s!"param {i} region {r}"
            if a.abs > maxM then maxM := a.abs
        offA := offA + sz; offB := offB + sz
    for j in [0:3 + nBnStats] do
      let a := F32.read oA (3 * nPA + j).toUSize
      let b := F32.read oR (3 * nPR + j).toUSize
      total := total + 1
      if a == b then exact := exact + 1
      if (a-b).abs > maxD then maxD := (a-b).abs; worst := s!"loss/bnstat {j}"
      if a.abs > maxM then maxM := a.abs
    (maxD, maxM, exact, total, worst)
  -- Per REGION (§4): `bnstat` and `%loss` are FORWARD-only outputs, so if they are bit-exact the
  -- forward is identical and any difference is confined to the backward — which is what a
  -- reduction-reorder from dropping 36 gradient chains looks like, as opposed to a real change.
  let regionMax (oR : ByteArray) (nPR : Nat) (lo hi : Nat) : Float × Nat × Nat := Id.run do
    let mut d := 0.0; let mut e := 0; let mut n := 0
    for j in [lo:hi] do
      let a := F32.read oA (3 * nPA + j).toUSize
      let b := F32.read oR (3 * nPR + j).toUSize
      n := n + 1
      if a == b then e := e + 1
      if (a-b).abs > d then d := (a-b).abs
    (d, e, n)
  let (fD, _, fE, fT, _) := compare oA2 nPA
  let (maxD, maxM, exact, total, worst) := compare oB nPB
  let sc := 1000000000.0
  IO.println s!"  A-vs-A floor : bit-exact {fE}/{fT}, max abs diff {fD * sc}e-9"
  IO.println s!"  A-vs-B tie   : bit-exact {exact}/{total}, max abs diff {maxD * sc}e-9 \
(rel {maxD / maxM}, |A|max {maxM}), worst at {worst}"
  let (dL, eL, nL) := regionMax oB nPB 0 3
  let (dS, eS, nS') := regionMax oB nPB 3 (3 + nBnStats)
  IO.println s!"    forward-only regions:  %loss/bc {eL}/{nL} bit-exact (max {dL * sc}e-9), \
BN running stats {eS}/{nS'} bit-exact (max {dS * sc}e-9)"
  if maxM < 1e-6 then throw (IO.userError "DEGENERATE: A's outputs are ~0")
  -- The two graphs are NOT identical text — B has 36 fewer parameters and 36 fewer gradient
  -- chains — so XLA fuses and schedules them differently and the reduction orders diverge. Gate
  -- on the same relative bound §2b used for R34, and report the floor beside it so the reader can
  -- see how much of the difference is the backend rather than the change.
  if maxD / maxM > 1e-5 then
    IO.println s!"  ⛔ tie FAILED: rel {maxD / maxM} > 1e-5 — the renders compute different functions"
    throw (IO.userError "tie failed")
  IO.println s!"  ✅ the candidate ties the committed render at rel {maxD / maxM} over \
{total} shared floats ({exact} of them bit-exact), with 36 fewer parameters."

def main (args : List String) : IO Unit := do
  let path := args.headD "verified_mlir/resnet34_adam_train_step.mlir"
  let net := resnet34Verified.toNet
  match args with
  | "--ckpt" :: p :: _ => ckptMode net p; return
  | "--ablate" :: p :: _ => ablateMode net p; return
  | "--tie" :: p :: _ => tieMode net p; return
  | _ => pure ()
  let bs  := 32
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  IO.println s!"§2l step B — conv-bias gradients, from m = v = 0 (so m' = (1−β₁)·g)"
  IO.println s!"  render  {path}"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), bs {bs}, \
backend {← IreeSession.backendName}"

  -- θ at the driver's own init; m = v = 0 EXACTLY, which is what makes m' the gradient.
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind)
    sd := sd + 1
  let θ := F32.concat θparts
  let m ← F32.const net.nParams.toUSize 0.0
  let v ← F32.const net.nParams.toUSize 0.0
  let tail ← F32.const 3 0.0
  -- lr, then the bias-correction denominators at t = 1: 1−β₁¹ = 0.1, 1−β₂¹ = 0.001.
  let tail ← F32.write3 tail 0 0.001 0.1 0.001
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let pbuf := F32.concat #[θ, m, v, tail, bnIn]
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  -- §4: delete the .vmfb first, or a re-run with a different candidate reuses the old binary.
  let vmfb := ".lake/build/conv_bias_zero.vmfb"
  let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
  for p in [vmfb, s!".lake/build/conv_bias_zero_{target}.vmfb"] do
    if ← System.FilePath.pathExists p then IO.FS.removeFile p
  let sess ← mkSession path vmfb
  IO.println "  running one step…"; (← IO.getStdout).flush
  let out ← IreeSession.mlpTrainStepV sess "m.resnet34_adam_train_step" x pbuf shapes y
    bs.toUSize net.d0.toUSize net.nClasses.toUSize

  -- ── read the m' region: [θ' | m' | v' | loss,bc1,bc2 | bnstat] ──
  let nP := net.nParams
  let nS := net.specs.size
  let mut off := 0
  let mut maxBySlot : Array Float := Array.replicate 6 0.0
  let mut nzBySlot  : Array Nat   := Array.replicate 6 0
  let mut cntBySlot : Array Nat   := Array.replicate 6 0
  let mut worstConvB : Float := 0.0
  let mut worstConvBIdx : Nat := 0
  for i in [0:nS] do
    let sz := net.paramShapes[i]!.foldl (· * ·) 1
    let sl := classify nS i
    let k  := sl.idx
    let mut mx := 0.0
    let mut nz := 0
    for j in [0:sz] do
      let g := (F32.read out (nP + off + j).toUSize).abs   -- the m' region
      if g != 0.0 then nz := nz + 1
      if g > mx then mx := g
    if sl == .convB && mx > worstConvB then worstConvB := mx; worstConvBIdx := i
    maxBySlot := maxBySlot.set! k (max maxBySlot[k]! mx)
    nzBySlot  := nzBySlot.set! k (nzBySlot[k]! + nz)
    cntBySlot := cntBySlot.set! k (cntBySlot[k]! + sz)
    off := off + sz

  -- ⚠ Float.toString truncates at 6 decimals, which prints a 1e-9 residue as "0.000000" — the
  -- exact reading this check must not get wrong. Report ×1e9 and the ratio to the conv WEIGHT
  -- gradient, so a tiny-but-non-zero value cannot be mistaken for an exact zero.
  let scale := 1000000000.0
  IO.println ""
  IO.println "  region      |m'|max ×1e9      rel to conv-W      non-zero coords"
  for sl in [Slot.convW, .convB, .bnG, .bnB, .denseW, .denseB] do
    let rel := if maxBySlot[Slot.convW.idx]! > 0.0 then
                 maxBySlot[sl.idx]! / maxBySlot[Slot.convW.idx]! else 0.0
    IO.println s!"  {slotName sl}       {maxBySlot[sl.idx]! * scale}      {rel}      \
{nzBySlot[sl.idx]!}/{cntBySlot[sl.idx]!}"

  -- ── the verdict, with the controls first: a degenerate run must not read as a pass ──
  let mut ok := true
  for sl in [Slot.convW, .bnG, .bnB, .denseW] do
    if maxBySlot[sl.idx]! == 0.0 then
      IO.println s!"  ⛔ DEGENERATE: {slotName sl} gradients are all zero — the step proves nothing"
      ok := false
  if maxBySlot[Slot.denseB.idx]! == 0.0 then
    IO.println "  ⛔ CONTROL FAILED: the DENSE bias gradient is zero too. It is the same shape and \
init kind as a conv bias but has no BN after it, so this reading is \"the harness sees zeros\", \
NOT \"BN swallows the conv bias\"."
    ok := false
  if !ok then throw (IO.userError "measurement is degenerate — do not rely on it")

  -- ▶ MEASURED 2026-07-30, and §2l's stated reason is WRONG. The gradient is NOT exactly zero:
  -- `(x+b) − mean(x+b) = x − mean(x)` is exact in ℝ, but the BN mean is a ROUNDED f32 sum, so the
  -- cancellation leaves a residue on ~93% of coordinates. Two runs disagree on which coordinates,
  -- which is what identifies it as rounding noise rather than signal. So the gate is a RATIO, not
  -- an equality: the residue must be orders below every real gradient in the same step.
  let relB := maxBySlot[Slot.convB.idx]! / maxBySlot[Slot.convW.idx]!
  let relD := maxBySlot[Slot.denseB.idx]! / maxBySlot[Slot.convW.idx]!
  if relB > 1e-4 then
    IO.println s!"  ⛔ conv-bias gradients are {relB} of the conv-WEIGHT gradient — too large to be \
a rounding residue. Dropping the biases would change the function."
    throw (IO.userError "conv-bias gradient is not a rounding residue")
  IO.println s!"  ✅ conv-bias gradient is {relB} of the conv-weight gradient (a rounding residue \
on {nzBySlot[Slot.convB.idx]!}/{cntBySlot[Slot.convB.idx]!} coords), while the DENSE bias — same \
shape, same zero init, no BN after it — is {relD}."
  IO.println "  ⚠ NOT exactly zero, so §2l's \"it stays 0 forever\" is FALSE: AdamW's update is \
scale-free (§3), so a 1e-6 residue still moves θ by ~lr per step. Measured on the 80-epoch \
checkpoint, all 8,512 conv biases drifted to |θ|max 0.041 (`--ckpt`). What makes step B safe is \
not that they stay zero — it is that the FORWARD does not depend on them (`--ablate`)."

