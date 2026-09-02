import LeanMlir.VerifiedNets

/-! # §2k's owed gate — the heavy-ball momentum render, numerically certified

`verified_mlir/resnet34_mom_train_step.mlir` was **rendered** on 2026-07-30 and never numerically
gated; §2k says so in as many words — *"Until it is run, say 'rendered', not 'certified'."* This is
that run, built exactly as §2k prescribes: a **cross-render known answer**, not a tolerance
argument, reusing the `shard-check` construction.

**How the gradient is recovered.** Run the committed **AdamW** render on `(θ, x, onehot)` from
`m = v = 0`. Its stored first moment is `m' = β₁·m + (1−β₁)·g = 0.1·g`, so

    g = 10 · m'_adam

exactly — no tolerance, no dependence on the bias correction (which only enters `θ'`). AdamW's
decay is DECOUPLED, so `m` sees the raw gradient; that is what makes it usable as an oracle here.

**What the momentum render must then satisfy**, on the same `(θ, x, onehot)` with its velocity slot
zeroed (`optOne .heavyBall`, `optConstsB`: μ = 0.9, wd = 1e-4):

| | claim | why it is the interesting one |
|---|---|---|
| ① | `v' = g + wd·θ` | the **COUPLED** L2 — the decay flows through the velocity, unlike AdamW's decoupled `−lr·wd·θ`. This is the clause that makes `.adamw` unable to stand in for the reference recipe |
| ② | `θ' = θ − lr·v'` | **HEAVY-BALL** |
| ③ | `m' = m` | the passthrough the shared `[θ\|m\|v]` signature depends on |

**▶ THE CONTROL IS THE POINT, and it is the trap §2k dodged.** The repo's `momParamF` is
**Nesterov** — `θ − lr·(g + μ·v')` — and reaching for it is what "add the momentum variant"
obviously means. At `v = 0` that is `θ − lr·(1+μ)·v'`, i.e. **1.9×** the heavy-ball step. So this
harness computes BOTH predictions and requires that heavy-ball matches while Nesterov does *not*.
A gate that only checked ② against the render it was derived from would pass either optimizer.

Second control: ① is re-checked against `v' = g` with the decay dropped. `wd·θ` is ~3e-4 of `g`
here — small, but four orders above the f32 floor, so a missing coupled-L2 term is detectable and
is shown to be.

    lake build r34-mom-tie && HIP_VISIBLE_DEVICES=0 .lake/build/bin/r34-mom-tie
-/

def main : IO Unit := do
  let net := resnet34Verified.toNet
  let bs  := 32
  let nP  := net.nParams
  let μ   := 0.9
  let wd  := 0.0001
  let lr  := 0.1                     -- passed as %lr; large enough that θ' ≠ θ by a wide margin
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  IO.println "§2k's owed gate — heavy-ball momentum vs the AdamW gradient oracle"
  IO.println s!"  {net.specs.size} params ({nP} floats), bs {bs}, μ {μ}, wd {wd}, lr {lr}, \
backend {← LowererSession.backendName}"

  -- ── one (θ, x, onehot) both renders see ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParamHeFanIn sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  let z ← F32.const nP.toUSize 0.0
  -- `m` for the momentum run is deliberately NON-zero and distinctive: it must ride through
  -- untouched, and a zero fill would let a dropped passthrough match by luck.
  let mIn ← F32.scaleShift (← F32.heInit 4242 nP.toUSize 0.02) 1.0 0.7
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  let run (slug variant : String) (buf : ByteArray) : IO ByteArray := do
    let vmfb := s!".lake/build/mom_tie_{variant}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for p in [vmfb, s!".lake/build/mom_tie_{variant}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession s!"verified_mlir/{slug}_{variant}_train_step.mlir"
    LowererSession.mlpTrainStepV sess s!"m.{slug}_{variant}_train_step" x buf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize

  -- ── the ORACLE: AdamW from m = v = 0 ⇒ m' = 0.1·g ⇒ g = 10·m' ──
  let tlA ← F32.const 3 0.0
  let tlA ← F32.write3 tlA 0 lr 0.1 0.001            -- lr, 1−β₁¹, 1−β₂¹ at t = 1
  let oA ← run "resnet34" "adam" (F32.concat #[θ, z, z, tlA, bnIn])
  -- ── the render under test: heavy-ball with velocity 0, m distinctive ──
  let tlM ← F32.const 3 0.0
  let tlM ← F32.write3 tlM 0 lr 0.1 0.001            -- bc slots are unread by .heavyBall
  let oM ← run "resnet34" "mom" (F32.concat #[θ, mIn, z, tlM, bnIn])

  -- ── ①②③ and the two controls, in one pass over the parameters ──
  let mut dV := 0.0; let mut mV := 0.0        -- ① v' vs g + wd·θ
  let mut dVnoWd := 0.0                       --   control: v' vs g (decay dropped)
  let mut dT := 0.0; let mut mT := 0.0        -- ② θ' vs θ − lr·v'   (heavy-ball)
  let mut dTnes := 0.0                        --   CONTROL: θ' vs θ − lr·(1+μ)·v' (Nesterov)
  let mut mExact := 0                         -- ③ passthrough
  let mut gMax := 0.0
  for i in [0:nP] do
    let g   := 10.0 * F32.read oA (nP + i).toUSize      -- oracle gradient
    let θi  := F32.read θ i.toUSize
    let v'  := F32.read oM (2*nP + i).toUSize           -- the momentum render's velocity
    let θ'  := F32.read oM i.toUSize
    if g.abs > gMax then gMax := g.abs
    let wantV := g + wd * θi
    if (v' - wantV).abs > dV then dV := (v' - wantV).abs
    if wantV.abs > mV then mV := wantV.abs
    if (v' - g).abs > dVnoWd then dVnoWd := (v' - g).abs
    let wantT := θi - lr * v'
    if (θ' - wantT).abs > dT then dT := (θ' - wantT).abs
    if wantT.abs > mT then mT := wantT.abs
    let wantNes := θi - lr * (1.0 + μ) * v'
    if (θ' - wantNes).abs > dTnes then dTnes := (θ' - wantNes).abs
    if F32.read oM (nP + i).toUSize == F32.read mIn i.toUSize then mExact := mExact + 1

  let relV := dV / mV
  let relT := dT / mT
  IO.println ""
  IO.println s!"  oracle |g|max = {gMax}   (from AdamW's m' at m = v = 0)"
  -- ×1e9, because Float.toString truncates at 6 decimals and would print a 1e-8 tie as "0.000000"
  -- — the same formatting trap that nearly mis-read the conv-bias residue (§2l step B).
  let sc := 1000000000.0
  IO.println s!"  ① v' = g + wd·θ      max abs Δ {dV * sc}e-9   rel {relV * sc}e-9"
  IO.println s!"  ② θ' = θ − lr·v'     max abs Δ {dT * sc}e-9   rel {relT * sc}e-9"
  IO.println s!"  ③ m passthrough      bit-exact {mExact}/{nP}"
  IO.println s!"  ⟂ control, decay dropped   (v' vs g)                rel {dVnoWd / mV}"
  IO.println s!"  ⟂ control, NESTEROV        (θ' vs θ − lr·(1+μ)·v')  rel {dTnes / mT}"

  -- ── verdict. Degeneracy first: a zero gradient would satisfy everything. ──
  if gMax < 1e-6 then
    throw (IO.userError "DEGENERATE: the oracle gradient is ~0 — the check proves nothing")
  if mExact != nP then
    throw (IO.userError s!"③ FAILED: m is not a passthrough ({mExact}/{nP} bit-exact)")
  if relV > 1e-4 then
    throw (IO.userError s!"① FAILED: v' ≠ g + wd·θ (rel {relV}) — the coupled L2 is wrong")
  if relT > 1e-4 then
    throw (IO.userError s!"② FAILED: θ' ≠ θ − lr·v' (rel {relT})")
  -- The controls must FIRE. A gate that cannot tell heavy-ball from Nesterov is not a gate.
  if dTnes / mT <= 10.0 * relT then
    throw (IO.userError s!"CONTROL DEAD: the Nesterov prediction fits as well as heavy-ball \
({dTnes / mT} vs {relT}) — this harness cannot tell the two optimizers apart, so ② means nothing")
  if dVnoWd / mV <= 10.0 * relV then
    throw (IO.userError s!"CONTROL DEAD: dropping wd·θ fits as well as keeping it \
({dVnoWd / mV} vs {relV}) — the coupled-L2 term is not being tested")
  IO.println ""
  IO.println s!"  ✅ CERTIFIED: the momentum render is heavy-ball with coupled L2 — ① rel {relV * sc}e-9, \
② rel {relT * sc}e-9, ③ {nP}/{nP} bit-exact, against a Nesterov control that misses by \
{dTnes / mT} ({(dTnes / mT) / relT}× the tie) and a no-decay control that misses by \
{dVnoWd / mV} ({(dVnoWd / mV) / relV}× the tie)."
  IO.println "     §2k's \"say rendered, not certified\" is discharged."
