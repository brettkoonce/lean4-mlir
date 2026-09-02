import LeanMlir.VerifiedNets
-- for `mnv2RmsHyper` / `enetRmsHyper` — the SAME records `rmsConstsBlock` emits the graph
-- constants from, so this gate reads its ρ/ε/wd from the render's own source rather than a copy.
import LeanMlir.Proofs.Codegen.StableHLO

/-! # The RMSProp render, numerically certified — recipe_gaps v1.2's gate

`verified_mlir/{mobilenetv2,efficientnet}_rms_train_step.mlir` render RMSProp-with-momentum, the
optimizer both nets' ImageNet references actually use — the **only** gap between MobileNetV2 and
JAX's 68.33%, and one of two for EfficientNet's 72.31% (`planning/recipe_gaps.md` §2). This is
their numeric gate, built as §2k built `r34-mom-tie`: a **cross-render known answer**, not a
tolerance argument.

**How the gradient is recovered.** Run the committed **AdamW** render on `(θ, x, onehot)` from
`m = v = 0`. Its stored first moment is `m' = β₁·m + (1−β₁)·g = 0.1·g`, so

    g = 10 · m'_adam

exactly. AdamW's decay is DECOUPLED, so `m` sees the raw gradient — which is what makes it usable
as an oracle for a render whose decay is *coupled*.

**What the RMSProp render must satisfy**, on the same `(θ, x, onehot)` with both moment slots
zeroed (ρ = μ = 0.9 on both; ε/wd are **1.0 / 4e-5** on mnv2 and **1e-3 / 1e-5** on enet),
writing `gw = g + wd·θ`:

| | claim | why it is the interesting one |
|---|---|---|
| ① | `s' = (1−ρ)·gw²` | the mean-square — and because it is built from `gw`, this simultaneously tests that the L2 is **COUPLED** and that it enters **BEFORE** the accumulator, which is the reference's ordering |
| ② | `b' = gw / √(s' + ε)` | ⚠ **ε INSIDE the root** — TensorFlow's RMSProp, not the textbook one |
| ③ | `θ' = θ − lr·b'` | the parameter update |

**▶ THE CONTROLS ARE THE POINT.** ② is re-checked against the textbook prediction
`gw / (√s' + ε)`, and ① against `(1−ρ)·g²` with the decay dropped. A gate that only checked ②
against the render it came from would pass either ε placement, and that placement is the exact trap
`timm`'s `RMSpropTF` exists to avoid — the JAX reference calls it out in its own comment.

⚠ **The two nets sit on opposite sides of the ε placement.** At mnv2's ε = 1.0 the difference is
mild; at EfficientNet's ε = 1e-3 it is worth ~31.6× at a collapsed mean-square
(`Proofs.rmsBufNext_eps_placement_at_zero`). **Neither net's green run licenses the other** — run
both. The harness prints each control's separation, so the gap between them is measured rather
than assumed.

**ONE harness, both nets** — `rms-tie [mobilenetv2|efficientnet]`. Written as a family rather than
copied per net for the reason `TestShardCheck.lean` was: a second copy is the double-writer disease
one level down, in code. Everything per-net comes from the selected `VerifiedNetSpec` and from
`Proofs.StableHLO.{mnv2,enet}RmsHyper` — **the same records the renderer emits its constants
from**, so the gate cannot drift from the render it gates.

    lake build rms-tie
    CUDA_VISIBLE_DEVICES=0 .lake/build/bin/rms-tie mobilenetv2
    CUDA_VISIBLE_DEVICES=0 .lake/build/bin/rms-tie efficientnet
-/

def main (argv : List String) : IO Unit := do
  let slug := argv.headD "mobilenetv2"
  -- The hyperparameters come from the SAME records `rmsConstsBlock` emits from, never a second
  -- hand-copied pair — the K-constant lesson (§2k) applied to the gate rather than the render.
  let hyper ← match slug with
    | "mobilenetv2" => pure Proofs.StableHLO.mnv2RmsHyper
    | "efficientnet" => pure Proofs.StableHLO.enetRmsHyper
    | other => throw (IO.userError s!"unknown net '{other}' — expected mobilenetv2 | efficientnet")
  let vnet := if slug == "efficientnet" then efficientnetVerified else mobilenetv2Verified
  let net := vnet.toNet
  let bs  := 32
  let nP  := net.nParams
  let ρ   := hyper.rho
  let ε   := hyper.eps
  let wd  := hyper.wd
  let lr  := 0.1                     -- passed as %lr; large enough that θ' ≠ θ by a wide margin
  let bnStatShapes := net.bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]]) #[]
  let nBnStats := net.bnChannels.foldl (fun acc c => acc + 2 * c) 0
  IO.println s!"recipe_gaps v1.2 gate — RMSProp (TF flavour) vs the AdamW gradient oracle — {slug}"
  IO.println s!"  {net.specs.size} params ({nP} floats), bs {bs}, ρ {ρ}, ε {ε}, wd {wd}, lr {lr}, \
backend {← LowererSession.backendName}"

  -- ── one (θ, x, onehot) both renders see ──
  let mut θparts : Array ByteArray := #[]
  let mut sd := 1234
  for (dims, kind) in net.specs do
    θparts := θparts.push (← mkParamHeFanIn sd dims kind); sd := sd + 1
  let θ := F32.concat θparts
  let z ← F32.const nP.toUSize 0.0
  let bnIn ← F32.scaleShift (← F32.heInit 3131 nBnStats.toUSize 0.01) 1.0 0.3
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                            ++ #[#[], #[], #[]] ++ bnStatShapes)
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0

  let run (netSlug variant : String) (buf : ByteArray) : IO ByteArray := do
    let vmfb := s!".lake/build/rms_tie_{netSlug}_{variant}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for p in [vmfb, s!".lake/build/rms_tie_{netSlug}_{variant}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession s!"verified_mlir/{netSlug}_{variant}_train_step.mlir"
    LowererSession.mlpTrainStepV sess s!"m.{netSlug}_{variant}_train_step" x buf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize

  -- ── the ORACLE: AdamW from m = v = 0 ⇒ m' = 0.1·g ⇒ g = 10·m' ──
  let tlA ← F32.const 3 0.0
  let tlA ← F32.write3 tlA 0 lr 0.1 0.001            -- lr, 1−β₁¹, 1−β₂¹ at t = 1
  let oA ← run slug "adam" (F32.concat #[θ, z, z, tlA, bnIn])
  -- ── the render under test: RMSProp from buf = 0, sq = 0 ──
  let tlR ← F32.const 3 0.0
  let tlR ← F32.write3 tlR 0 lr 0.1 0.001            -- bc slots are unread by .rmsprop
  let oR ← run slug "rms" (F32.concat #[θ, z, z, tlR, bnIn])

  -- ── ①②③ and the two controls, in one pass over the parameters ──
  let mut dS := 0.0; let mut mS := 0.0        -- ① s' vs (1−ρ)·gw²
  let mut dSnoWd := 0.0                       --   control: s' vs (1−ρ)·g²  (decay dropped)
  let mut dB := 0.0; let mut mB := 0.0        -- ② b' vs gw/√(s'+ε)     (TENSORFLOW)
  let mut dBvan := 0.0                        --   CONTROL: b' vs gw/(√s'+ε)  (textbook)
  let mut dT := 0.0; let mut mT := 0.0        -- ③ θ' vs θ − lr·b'
  let mut gMax := 0.0
  for i in [0:nP] do
    let g   := 10.0 * F32.read oA (nP + i).toUSize      -- oracle gradient
    let θi  := F32.read θ i.toUSize
    let b'  := F32.read oR (nP + i).toUSize             -- the RMSProp render's momentum buffer
    let s'  := F32.read oR (2*nP + i).toUSize           -- ... and its mean-square
    let θ'  := F32.read oR i.toUSize
    if g.abs > gMax then gMax := g.abs
    let gw := g + wd * θi
    -- ① the mean-square, built from the COUPLED gradient
    let wantS := (1.0 - ρ) * gw * gw
    if (s' - wantS).abs > dS then dS := (s' - wantS).abs
    if wantS.abs > mS then mS := wantS.abs
    let wantSnoWd := (1.0 - ρ) * g * g
    if (s' - wantSnoWd).abs > dSnoWd then dSnoWd := (s' - wantSnoWd).abs
    -- ② the buffer — against the render's OWN s', so this isolates the ε placement
    let wantB := gw / Float.sqrt (s' + ε)
    if (b' - wantB).abs > dB then dB := (b' - wantB).abs
    if wantB.abs > mB then mB := wantB.abs
    let wantBvan := gw / (Float.sqrt s' + ε)
    if (b' - wantBvan).abs > dBvan then dBvan := (b' - wantBvan).abs
    -- ③ the parameter update, against the render's OWN b'
    let wantT := θi - lr * b'
    if (θ' - wantT).abs > dT then dT := (θ' - wantT).abs
    if wantT.abs > mT then mT := wantT.abs

  let relS := dS / mS
  let relB := dB / mB
  let relT := dT / mT
  let relSnoWd := dSnoWd / mS
  let relBvan := dBvan / mB
  IO.println ""
  IO.println s!"  oracle |g|max = {gMax}   (from AdamW's m' at m = v = 0)"
  -- ×1e9, because Float.toString truncates at 6 decimals and would print a 1e-8 tie as "0.000000".
  let sc := 1000000000.0
  IO.println s!"  ① s' = (1−ρ)·gw²      max abs Δ {dS * sc}e-9   rel {relS * sc}e-9"
  IO.println s!"  ② b' = gw/√(s'+ε)     max abs Δ {dB * sc}e-9   rel {relB * sc}e-9"
  IO.println s!"  ③ θ' = θ − lr·b'      max abs Δ {dT * sc}e-9   rel {relT * sc}e-9"
  IO.println s!"  ⟂ control, TEXTBOOK ε  (b' vs gw/(√s'+ε))   rel {relBvan}"
  IO.println s!"  ⟂ control, decay dropped (s' vs (1−ρ)·g²)   rel {relSnoWd}"

  -- ── verdict. Degeneracy first: a zero gradient would satisfy everything. ──
  if gMax < 1e-6 then
    throw (IO.userError "DEGENERATE: the oracle gradient is ~0 — the check proves nothing")
  if relS > 1e-4 then
    throw (IO.userError s!"① FAILED: s' ≠ (1−ρ)·gw² (rel {relS}) — wrong ρ, or the L2 is not \
coupled into the gradient before the accumulator")
  if relB > 1e-4 then
    throw (IO.userError s!"② FAILED: b' ≠ gw/√(s'+ε) (rel {relB}) — the ε placement or μ is wrong")
  if relT > 1e-4 then
    throw (IO.userError s!"③ FAILED: θ' ≠ θ − lr·b' (rel {relT})")
  -- The controls must FIRE. A gate that cannot tell TF's RMSProp from the textbook one is not a gate.
  if relBvan <= 10.0 * relB then
    throw (IO.userError s!"CONTROL DEAD: the TEXTBOOK ε placement fits as well as TensorFlow's \
({relBvan} vs {relB}) — this harness cannot tell the two spellings apart, so ② means nothing. \
NOTE ε = {ε} here; at a LARGE ε and a small mean-square the two converge, which is exactly why \
EfficientNet (ε = 1e-3) needs its own run of this gate.")
  if relSnoWd <= 10.0 * relS then
    throw (IO.userError s!"CONTROL DEAD: dropping wd·θ fits as well as keeping it \
({relSnoWd} vs {relS}) — the coupled-L2 term is not being tested (wd = {wd} may be too small \
against |g| = {gMax} to be separable at f32)")
  IO.println ""
  IO.println s!"  ✅ CERTIFIED: the render is TF-flavoured RMSProp with coupled L2 — ① rel \
{relS * sc}e-9, ② rel {relB * sc}e-9, ③ rel {relT * sc}e-9, against a textbook-ε control that \
misses by {relBvan} ({relBvan / relB}× the tie) and a no-decay control that misses by \
{relSnoWd} ({relSnoWd / relS}× the tie)."
