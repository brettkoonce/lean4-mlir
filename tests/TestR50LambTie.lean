import LeanMlir.VerifiedNets

/-! # LAMB, numerically certified — the closed form, against three plausible wrong neighbours

`planning/rsb_a3_r50_verified.md` §2.3's LAMB row was the ONE line that file flags as an estimate
rather than a measurement ("2–3 ops"). It is now built — measured at **two** new `SHlo`
constructors, because `gradSumSqAccF` was already present for the global-norm clip and `sgdParamF`
for heavy-ball — and this is the run that says the render computes LAMB.

## The construction — `r34-mom-tie`'s, reused for the third time

**How the gradient is recovered.** Run the committed **AdamW** render on `(θ, x, onehot)` from
`m = v = 0`. Its stored first moment is `m' = β₁·m + (1−β₁)·g = 0.1·g`, so `g = 10·m'_adam`
exactly — no tolerance, no dependence on the bias correction. AdamW's decay is DECOUPLED, so `m`
sees the raw gradient; that is what makes it usable as an oracle. §2k's construction, reused by
`r34-mom-tie`, `rms-tie` and `conv-bias-zero`.

**What the LAMB render must then satisfy**, on the same `(θ, x, onehot)` with DISTINCTIVE non-zero
`m`/`v` (a zero fill would let a dropped β₁/β₂ term match by luck):

    m' = β₁·m + (1−β₁)·g                          β₁ = 0.9   — Adam's, unchanged
    v' = β₂·v + (1−β₂)·g²                         β₂ = 0.999 — Adam's, unchanged
    r  = (m'/bc₁) / (√(v'/bc₂) + ε) + wd·θ        ε = 1e-6, wd = 0.02   ⚠ NOT AdamW's 1e-8 / 1e-4
    trust = ‖θ‖/‖r‖   PER PARAMETER TENSOR, = 1 when either norm is 0
    θ' = θ − lr·trust·r

⚠ `bc₁`/`bc₂` are supplied at **t = 10**, not t = 1. At t = 1 the correction is `1/(1−β₁) = 10×`,
which with non-zero incoming moments is an inconsistent state to test at; t = 10's `0.651`/`0.00995`
is what a real step sees.

## ▶▶ THE CONTROLS ARE THE POINT — three wrong LAMBs that all train and descend

Every one of these is a real implementation people ship, and none of them is distinguishable from
the real thing by a loss curve:

| ⟂ | the wrong neighbour | where it comes from |
|---|---|---|
| ① | **`trust ≡ 1`** | plain bias-corrected Adam with decoupled decay — i.e. forgetting the layer-wise part, which IS the algorithm |
| ② | **`√(v̂ + ε)`** instead of `√v̂ + ε` | RMSProp-TF's ε placement, which `RmsPropStep` exists because of |
| ③ | **decay AFTER the trust ratio** | AdamW's placement — `θ − lr·(trust·r₀ + wd·θ)`. The decay then never enters the norm |

Each is computed on the same numbers and required to MISS by far more than the tie. A harness that
cannot separate them is not measuring LAMB, it is measuring "something scaled something".

⚠ The comparison is on the **STEP** `θ' − θ`, not on `θ'`. The step is ~1e-3 of `θ`, so a relative
error on `θ'` divides by the wrong thing and would report a passing 1e-3 as 1e-6. Same reading
trap as §2e-bis's `Float.toString`.

⚠ Expect ~1e-4, not bit-exact: the graph reduces `‖·‖²` over up to 4.7M f32 elements while this
harness sums in f64, so the two norms differ in their last few bits and the trust ratio inherits it.

    lake build r50-lamb-tie && CUDA_VISIBLE_DEVICES=0 .lake/build/bin/r50-lamb-tie
-/

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

def main : IO Unit := do
  let net  := resnet50ImagenetVerified.toNet
  let bs   := 64
  let nP   := net.nParams
  let β₁   := 0.9
  let β₂   := 0.999
  let ε    := 1.0e-6          -- ⚠ LAMB's, not AdamW's 1e-8
  let wd   := 0.02            -- ⚠ timm a3's `wd0.02`, not AdamW's 1e-4
  let lr   := 0.008           -- RSB-A3's LR at bs2048
  let bc1  := 1.0 - Float.exp (10.0 * Float.log β₁)      -- t = 10
  let bc2  := 1.0 - Float.exp (10.0 * Float.log β₂)
  let tol  := (((← IO.getEnv "R50_LAMB_TOL_U").bind (·.toNat?)).map (fun u => u.toFloat * 1e-6)
                |>.getD 1.0e-3)
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]

  IO.println "§2.3's LAMB row, measured — the render against LAMB's closed form"
  IO.println s!"  under test  verified_mlir/{net.slug}_lamb64_train_step.mlir"
  IO.println s!"  oracle      verified_mlir/{net.slug}_adam64_train_step.mlir  (g = 10·m' at m = v = 0)"
  IO.println s!"  {net.specs.size} params ({nP} floats), bs {bs}, lr {lr}, eps {ε}, wd {wd}, \
bc1 {bc1} bc2 {bc2} (t = 10), backend {← IreeSession.backendName}"

  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParam sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  let z ← F32.const nP.toUSize 0.0
  -- DISTINCTIVE and non-zero: a zero fill would let a render that dropped the β₁·m or β₂·v
  -- passthrough match by luck, which is the same argument `r34-mom-tie` makes for its `m`.
  let mIn ← F32.scaleShift (← F32.heInit 4242 nP.toUSize 0.02) 1.0 0.05
  -- ⚠⚠ TWO SECOND-MOMENT REGIMES, and the reason is a MEASUREMENT this harness made rather than a
  -- precaution. At `v̂ ≈ 1` the two ε placements — `√v̂ + ε` and `√(v̂ + ε)` — differ by `ε/(2√v̂)`
  -- against `ε`, i.e. by **5e-7 at ε = 1e-6**, which is BELOW this harness's own 3e-6 floor. So at a
  -- typical `v` the placement is not gated at all, and a run that reported ⟂② as "passing" there
  -- would be reporting that its control is dead.
  --
  -- ⭐ The placement bites where `v̂ ≪ ε²`, which is early training and small gradients — exactly
  -- the regime `RmsPropStep` was written for. `vSmall` puts `v̂ ≈ 1e-7`, where `√v̂ = 3.2e-4` and
  -- `√(v̂+ε) = 1.0e-3`: a 3× separation. The tie is exact in BOTH regimes; only the control's
  -- sensitivity moves, so both are run and only the second is allowed to carry ⟂②.
  let vIn ← F32.scaleShift (← F32.heInit 7373 nP.toUSize 0.002) 1.0 0.01
  let vSmall ← F32.scaleShift (← F32.heInit 5151 nP.toUSize 2.0e-10) 1.0 1.0e-9
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % 251)); y := y.push 0; y := y.push 0; y := y.push 0

  let run (variant : String) (buf : ByteArray) : IO ByteArray := do
    let vmfb := s!".lake/build/r50_lamb_tie_{variant}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for p in [vmfb, s!".lake/build/r50_lamb_tie_{variant}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession s!"verified_mlir/{net.slug}_{variant}_train_step.mlir" vmfb
    IreeSession.mlpTrainStepV sess s!"m.{net.slug}_{variant}_train_step" x buf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize

  -- ── the ORACLE: AdamW from m = v = 0 ⇒ m' = 0.1·g ⇒ g = 10·m' ──
  let tlA ← F32.write3 (← F32.const 3 0.0) 0 0.0 0.1 0.001   -- lr = 0: θ' unread, and it is explicit
  let oA ← run "adam64" (F32.concat #[θ, z, z, tlA, bnIn])
  -- ── the render under test ──
  let tlL ← F32.write3 (← F32.const 3 0.0) 0 lr bc1 bc2
  let vv0 ← run "lamb64" (F32.concat #[θ, mIn, vIn, tlL, bnIn])
  let oS ← run "lamb64" (F32.concat #[θ, mIn, vSmall, tlL, bnIn])

  -- ── the closed form, per parameter TENSOR (the trust ratio's quantifier) ──
  let mut offs : Array Nat := #[0]
  for sh in net.paramShapes do offs := offs.push (offs.back! + sh.foldl (· * ·) 1)
  -- One regime: given the incoming `v` and the render's output, return
  -- (tie rel on the step, m' rel, v' rel, ⟂① rel, ⟂② rel, ⟂③ rel, |g|max, guard count).
  let evalRegime (vv oL : ByteArray) :
      IO (Float × Float × Float × Float × Float × Float × Float × Nat) := do
    let mut dStep := 0.0; let mut mStep := 0.0    -- ② the tie, on the STEP
    let mut dM := 0.0; let mut mM := 0.0          -- ① the moments
    let mut dV := 0.0; let mut mV := 0.0
    let mut c1 := 0.0; let mut c2 := 0.0; let mut c3 := 0.0   -- ⟂ the three controls
    let mut gMax := 0.0
    let mut nTrust1 := 0                          -- tensors where the zero-norm guard fired
    for t in [0:net.specs.size] do
      let (fo, fl) := (offs[t]!, offs[t+1]! - offs[t]!)
      -- ‖θ‖ and ‖r‖ for THIS tensor, in f64
      let mut wn2 := 0.0; let mut rn2 := 0.0; let mut rn2b := 0.0; let mut rn2c := 0.0
      let mut rs : Array Float := #[]
      let mut rsB : Array Float := #[]   -- ⟂② ε inside the root
      let mut rsC : Array Float := #[]   -- ⟂③ decay after the trust ratio (r without wd)
      for i in [0:fl] do
        let θi := F32.read θ (fo + i).toUSize
        let g  := 10.0 * F32.read oA (nP + fo + i).toUSize
        if g.abs > gMax then gMax := g.abs
        let m' := β₁ * F32.read mIn (fo + i).toUSize + (1.0 - β₁) * g
        let v' := β₂ * F32.read vv (fo + i).toUSize + (1.0 - β₂) * g * g
        let mh := m' / bc1
        let vh := v' / bc2
        let r  := mh / (Float.sqrt vh + ε) + wd * θi
        let rB := mh / Float.sqrt (vh + ε) + wd * θi
        let rC := mh / (Float.sqrt vh + ε)
        rs := rs.push r; rsB := rsB.push rB; rsC := rsC.push rC
        wn2 := wn2 + θi * θi; rn2 := rn2 + r * r; rn2b := rn2b + rB * rB; rn2c := rn2c + rC * rC
        -- ① the moments, which LAMB shares with Adam
        let dm := (F32.read oL (nP + fo + i).toUSize - m').abs
        if dm > dM then dM := dm
        if m'.abs > mM then mM := m'.abs
        let dv := (F32.read oL (2 * nP + fo + i).toUSize - v').abs
        if dv > dV then dV := dv
        if v'.abs > mV then mV := v'.abs
      let trust  := if wn2 > 0.0 && rn2  > 0.0 then Float.sqrt wn2 / Float.sqrt rn2  else 1.0
      let trustB := if wn2 > 0.0 && rn2b > 0.0 then Float.sqrt wn2 / Float.sqrt rn2b else 1.0
      let trustC := if wn2 > 0.0 && rn2c > 0.0 then Float.sqrt wn2 / Float.sqrt rn2c else 1.0
      if wn2 <= 0.0 then nTrust1 := nTrust1 + 1
      for i in [0:fl] do
        let θi   := F32.read θ (fo + i).toUSize
        let step := F32.read oL (fo + i).toUSize - θi          -- what the render did
        let want := -lr * trust * rs[i]!                        -- ② LAMB
        if (step - want).abs > dStep then dStep := (step - want).abs
        if want.abs > mStep then mStep := want.abs
        -- ⟂① trust ≡ 1: plain bias-corrected Adam with decoupled decay
        let w1 := -lr * rs[i]!
        if (step - w1).abs > c1 then c1 := (step - w1).abs
        -- ⟂② ε INSIDE the root (RMSProp-TF's placement)
        let w2 := -lr * trustB * rsB[i]!
        if (step - w2).abs > c2 then c2 := (step - w2).abs
        -- ⟂③ decay AFTER the trust ratio (AdamW's placement)
        let w3 := -lr * (trustC * rsC[i]! + wd * θi)
        if (step - w3).abs > c3 then c3 := (step - w3).abs

    return (dStep / max mStep 1e-30, dM / max mM 1e-30, dV / max mV 1e-30,
            c1 / max mStep 1e-30, c2 / max mStep 1e-30, c3 / max mStep 1e-30, gMax, nTrust1)

  -- ── run both regimes ──
  let (rS, rM, rV, k1, k2, k3, gMax, nT1) ← evalRegime vIn vv0
  let (sS, sM, sV, s1, s2, s3, _, _)      ← evalRegime vSmall oS
  IO.println ""
  IO.println s!"  oracle |g|max {gMax}   ({nT1} tensors hit the zero-‖θ‖ guard ⇒ trust = 1)"
  IO.println s!"  regime A — v̂ ≈ 1 (a typical mid-training second moment)"
  IO.println s!"    ① m' rel {rM}   v' rel {rV}"
  IO.println s!"    ② θ' − θ = −lr·trust·r   rel {rS}"
  IO.println s!"    ⟂① trust ≡ 1                      rel {k1}"
  IO.println s!"    ⟂② √(v̂+ε)                          rel {k2}   ⚠ NOT a live control here — see below"
  IO.println s!"    ⟂③ decay after the trust ratio    rel {k3}"
  IO.println s!"  regime B — v̂ ≈ 1e-7 (early training / small gradients, where ε's placement bites)"
  IO.println s!"    ① m' rel {sM}   v' rel {sV}"
  IO.println s!"    ② θ' − θ = −lr·trust·r   rel {sS}"
  IO.println s!"    ⟂① trust ≡ 1                      rel {s1}"
  IO.println s!"    ⟂② √(v̂+ε)                          rel {s2}"
  IO.println s!"    ⟂③ decay after the trust ratio    rel {s3}"

  IO.println ""
  if gMax < 1e-6 then
    throw (IO.userError "DEGENERATE: the oracle gradient is ~0 — the check proves nothing")
  for (nm, r, m, v) in [("A", rS, rM, rV), ("B", sS, sM, sV)] do
    if m > tol then throw (IO.userError s!"① FAILED in regime {nm}: m' is not Adam's first moment (rel {m})")
    if v > tol then throw (IO.userError s!"① FAILED in regime {nm}: v' is not Adam's second moment (rel {v})")
    if r > tol then
      throw (IO.userError s!"② FAILED in regime {nm}: the step is not −lr·trust·r (rel {r} > {tol}) \
— check the ε placement, the wd value (0.02, not 1e-4), and whether the norm fold is per-tensor")
  -- ⟂① and ⟂③ are live in BOTH regimes; ⟂② only in B, and that asymmetry is the measurement.
  if k1 <= 10.0 * rS then
    throw (IO.userError s!"CONTROL DEAD: 'trust ≡ 1' fits as well as LAMB in regime A ({k1} vs {rS}) \
— the layer-wise part, which IS the algorithm, is not being tested")
  if k3 <= 10.0 * rS then
    throw (IO.userError s!"CONTROL DEAD: 'decay after the trust ratio' fits as well as LAMB in \
regime A ({k3} vs {rS}) — the decay's placement inside the norm is not being tested")
  -- ⚠⚠ ⟂② IS REQUIRED TO BE DEAD IN A AND LIVE IN B. Both directions are asserted: if it ever fires
  -- in A, ε's effective size has changed and this file's own reasoning about the floor is stale; if
  -- it stops firing in B, the ε placement has ceased to be gated anywhere.
  if k2 > 10.0 * rS then
    throw (IO.userError s!"regime A's ⟂② is LIVE ({k2} vs a tie of {rS}), which this harness \
predicted it would not be. ε is no longer negligible against √v̂ there — re-derive the floor \
argument in the header rather than trusting it")
  if s2 <= 10.0 * sS then
    throw (IO.userError s!"CONTROL DEAD: '√(v̂+ε)' fits as well as LAMB even at v̂ ≈ 1e-7 ({s2} vs \
{sS}) — the ε placement is gated NOWHERE, so ② says nothing about it")
  IO.println s!"  ✅ CERTIFIED: the render is LAMB in both regimes — step rel {rS} (A) and {sS} (B)."
  IO.println s!"     ⟂ Live controls: trust ≡ 1 misses by {k1} ({k1 / max rS 1e-30}x the tie), decay \
after the ratio by {k3}, and ε-inside-the-root by {s2} at v̂ ≈ 1e-7 ({s2 / max sS 1e-30}x)."
  IO.println s!"     ⭐ MEASURED, and worth keeping: at v̂ ≈ 1 the two ε placements agree to {k2}, \
i.e. BELOW this harness's own floor. LAMB's ε = 1e-6 makes the placement unobservable at a typical \
second moment — unlike RMSProp's ε = 1e-3, which `RmsPropStep` records at ~30x. It is gated only \
where it matters."
  IO.println "     ⚠ This certifies the OPTIMIZER, not the recipe: rsb-faithful is LAMB at \
effective batch 2048 with BCE-with-logits and 160/224, and none of those three is composed here."
