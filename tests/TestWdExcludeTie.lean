import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.ViTRender
import LeanMlir.Proofs.Codegen.ConvNeXtRender

/-! # `wdExcludeNormBias` — the timm `no_weight_decay` render, numerically certified (v1.4)

`verified_mlir/vit_adamwx_train_step.mlir` is `vit_adam` with decoupled weight decay switched OFF
for the 126 params timm/DeiT exclude — every 1-D param (biases, LayerNorm γ/β, the CLS token) and
the positional embedding. `vitTinyImagenetConfig.wdExcludeNormBias := true`, so the ImageNet pair
needs it; it is one of the two gaps `recipe_gaps.md` §2 lists for both ViT and ConvNeXt.

**Why a cross-render KNOWN ANSWER and not a tie.** The two renders compute genuinely different
functions, so there is nothing to tie — and every structural check passes on both, because the
change moves **no arity, no type and no region**, only which constant feeds `%wd` at 126 of 200
sites. What separates them is exactly one term, and it is exactly predictable:

    adam:  θ' = θ − lr·( m̂/(√v̂+ε) + wd·θ )
    wx:    θ' = θ − lr·( m̂/(√v̂+ε) + wd·msk·θ )        msk ∈ {0,1} per TENSOR

so on one shared `(θ, x, onehot)` with `m = v = 0`:

| | claim | gated at |
|---|---|---|
| ① | a **decayed** param's θ' is **BIT-EXACT** between the two renders | 74 params, no tolerance |
| ② | an **excluded** param's `θ'_wx − θ'_adam` is exactly `lr·wd·θ` | 126 params, norm-relative |
| ③ | `m'`, `v'` are **bit-exact on all 200** — decay is DECOUPLED, so it touches neither | 400 regions |
| ④ | `%loss` is bit-exact — it is a forward-only output and cannot see the optimizer | |

**▶ THE PARTITION IS THE CONTROL, and that is the whole design.** The gate does not check that 74
params match and 126 differ — a count is satisfied by *any* 74. It recovers, per parameter, which
bucket that parameter EMPIRICALLY falls in (θ' bit-exact ⇒ decayed; θ' offset by `lr·wd·θ` ⇒
excluded) and requires the empirical partition to **equal `vitWdDecays`'s, name for name**. A mask
that excluded the wrong 126 — the plausible failure, since a misaligned mask is silent in the
arity, the types and the prefix audit (§2e) — lands params in the wrong bucket and fires.

⚠ **The instrument has to be conditioned, and this is the ViT-EMA lesson (§0, 2026-08-02) in the
same place it bit before.** ② is a difference of two nearly-equal f32 numbers: `θ'_wx − θ'_adam`
against `|θ'|`. Its resolution is `wd·|θ| / 1` ≈ 5e-6 at wd = 1e-4 — about 80× f32's own 6e-8, so
it is readable but not generously so, and it is INDEPENDENT of `lr` (both the difference and
`|θ'|` scale with it, which is why turning `%lr` up does not help here the way it did for EMA).
That is why ① and ③ carry the weight: they are bit-exact and need no resolution argument at all.

    lake build wdx-tie
    CUDA_VISIBLE_DEVICES=0 .lake/build/bin/wdx-tie vit         # 200 params, 74/126
    CUDA_VISIBLE_DEVICES=0 .lake/build/bin/wdx-tie convnext    # 180 params, 59/121

**ONE harness, both nets**, for the reason `TestShardCheck.lean` and `rms-tie` are families: a
second copy is the double-writer disease one level down, in code. Everything per-net comes from the
selected net's own signature list and mask predicate — `vitParamSig`/`vitWdDecays` and
`allParams`/`cnxWdDecays` — i.e. from the SAME sources the renderers choose `%wd`/`%wdz` from, so
the gate cannot drift from the render it gates.

⚠ **The two nets' rules are NOT the same, and that is the point of running both.** ViT excludes the
positional embedding by NAME on top of the rank test; ConvNeXt has no such param (its generated
reference sets `_WD_POS_SHAPE = None`), so it is the plain rank test. A green ViT run does not
license ConvNeXt — the `rms-tie` ε-placement lesson, one knob over.
-/

open Proofs.StableHLO

/-- Everything per-net: the slug, the signature list (names + shapes, in func-arg order) and the
    mask predicate. Both entries read the RENDERER's own definitions, never a copy. -/
private structure WdNet where
  slug : String
  sig  : List (String × List Nat)
  mask : String → List Nat → Bool
  spec : VerifiedNetSpec

private def netBySlug (s : String) : IO WdNet :=
  match s with
  | "vit"      => pure { slug := "vit", sig := vitParamSig 10, mask := vitWdDecays,
                         spec := vitVerified }
  | "convnext" => pure { slug := "convnext", sig := cnxAllParams 10, mask := cnxWdDecays,
                         spec := convnextVerified }
  | _ => throw (IO.userError s!"unknown net '{s}' — expected vit | convnext")

/-- The driver's init (`VerifiedTrain.mkParam`, private) with **exactly one deliberate change**.

    ⚠ **`kind = 2` params are ZERO under the driver, and a zero θ makes ② VACUOUS** — the predicted
    offset `lr·wd·θ` collapses to 0, which is indistinguishable from "decayed". ViT's kind-2 params
    are the biases, β and the CLS token: precisely a large part of the 126 the mask excludes, so
    the driver's own init would have made this gate blind on the half it exists to test. They get
    small centred noise here instead.

    ⚠ **And the first attempt at "non-degenerate" broke the net**: giving every param a value
    centred at 0.6 (weights included) made the ViT forward overflow, and the harness reported
    `%loss NaN` with `0/5,526,346` bit-exact and `max abs 0.000000` — a contradiction that is the
    tell, since NaN ≠ NaN makes every coordinate "differ" while every comparison against `>` stays
    false. **Keep the driver's He scaling for the weights**; only the zeros may move. -/
private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0                                   -- γ: already non-zero
  | 2 => F32.heInit seed.toUSize n.toUSize 0.02                    -- was 0.0 — see above
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

private def cmpAt (a b : ByteArray) (off n : Nat) : Float × Float × Nat := Id.run do
  let mut d := 0.0; let mut m := 0.0; let mut e := 0
  for i in [0:n] do
    let u := F32.read a (off + i).toUSize
    let w := F32.read b (off + i).toUSize
    if u == w then e := e + 1
    if (u - w).abs > d then d := (u - w).abs
    if max u.abs w.abs > m then m := max u.abs w.abs
  (d, m, e)

def main (argv : List String) : IO Unit := do
  -- ⚠ `--cand <path>` drives a CANDIDATE wx render instead of the committed one, and it is what
  -- makes a green run believable — the `vit-dp-check` lesson (§2j): that harness hardcoded both
  -- paths, so its bit-exact PASS was unfalsifiable until an argument was added.
  -- `scripts/perturb_wd_mask.py` builds the two controls.
  let cand := match argv.dropWhile (· != "--cand") with
    | _ :: p :: _ => some p
    | _ => none
  let wn ← netBySlug ((argv.filter (fun a => a != "--cand" && !a.startsWith "/")).head?.getD "vit")
  let net := wn.spec.toNet
  let sig := wn.sig
  let bs  := 32
  let nDec := (sig.filter (fun (nm, ds) => wn.mask nm ds)).length
  let nExc := sig.length - nDec
  IO.println s!"wdExcludeNormBias — {wn.slug}, the timm no_weight_decay render (v1.4)"
  IO.println s!"  {sig.length} params, {nDec} decayed / {nExc} excluded, bs {bs}, \
backend {← IreeSession.backendName}"

  -- ⚠ TWO ROUTES TO THE SAME LAYOUT, checked. the signature list names the params and drives the
  -- render's `%wd`/`%wdz` choice; `net.specs` is the LAYOUT the driver packs a blob from. They are
  -- independent hand-lists, and this gate reads offsets from one while the mask came from the
  -- other — §2m's whole lesson (mnv2 shipped a 160-param forward past a green tie because only one
  -- of two arity routes was pinned).
  if net.specs.size != sig.length then
    throw (IO.userError s!"LAYOUT SKEW: net.specs has {net.specs.size} entries, vitParamSig has \
{sig.length} — the offsets below would be meaningless")
  for i in [0:sig.length] do
    let (nm, ds) := sig[i]!
    let (dims, _) := net.specs[i]!
    if dims.toList != ds then
      throw (IO.userError s!"LAYOUT SKEW at {i} ({nm}): net.specs says {dims.toList}, \
the signature list says {ds}")

  -- ── one shared (θ, x, onehot); m = v = 0 so the moments are the same on both sides ──
  let mut parts : Array ByteArray := #[]
  let mut sd := 4242
  for i in [0:sig.length] do
    let (dims, kind) := net.specs[i]!
    parts := parts.push (← mkParam sd dims kind); sd := sd + 1
  let θ := F32.concat parts
  let z ← F32.const net.nParams.toUSize 0.0
  -- `%lr`, then the bias-correction denominators at t = 1: 1−β₁¹ = 0.1, 1−β₂¹ = 0.001.
  let lrF : Float := 1.0
  let mut tl ← F32.const 3 0.0
  tl ← F32.write3 tl 0 lrF 0.1 0.001
  let x ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let mut y : ByteArray := .empty
  for i in [0:bs] do
    y := y.push (UInt8.ofNat (i % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                              ++ #[#[], #[], #[]])
  let buf := F32.concat #[θ, z, z, tl]

  let run (variant : String) : IO ByteArray := do
    let vmfb := s!".lake/build/wdx_tie_{wn.slug}_{variant}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for p in [vmfb, s!".lake/build/wdx_tie_{wn.slug}_{variant}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let dflt := s!"verified_mlir/{wn.slug}_{variant}_train_step.mlir"
    let path := if variant == "adamwx" then cand.getD dflt else dflt
    let sess ← mkSession path vmfb
    IreeSession.mlpTrainStepV sess s!"m.{wn.slug}_{variant}_train_step" x buf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize
  if cand.isSome then IO.println s!"  ⚠ CANDIDATE wx render: {cand.get!}"
  let oA ← run "adam"
  let oX ← run "adamwx"

  -- ⚠ REFUSE A NON-FINITE RUN BEFORE READING ANYTHING OUT OF IT. NaN ≠ NaN makes every
  -- coordinate "differ" and every `>` comparison false, so a blown-up forward reports as
  -- `0/N bit-exact, max abs 0.000000` — which reads like a catastrophic failure of the render
  -- rather than a broken input. It cost one run here.
  let P := net.nParams
  let mut nonFin := 0
  for i in [0:3*P + 1] do
    if !(F32.read oA i.toUSize).isFinite || !(F32.read oX i.toUSize).isFinite then
      nonFin := nonFin + 1
  if nonFin > 0 then
    throw (IO.userError s!"DEGENERATE: {nonFin} non-finite outputs — the INIT blew the forward up, not the render. Nothing below this line would mean anything.")

  -- ── ③ m' and v' must be bit-exact: decoupled decay touches neither ──
  let (dM, mM, eM) := cmpAt oA oX P P
  let (dV, mV, eV) := cmpAt oA oX (2*P) P
  IO.println s!"  ③ m' bit-exact {eM}/{P} (max abs {dM}, |m|max {mM})"
  IO.println s!"  ③ v' bit-exact {eV}/{P} (max abs {dV}, |v|max {mV})"
  -- ④ %loss — the forward-only output, immediately after the three regions
  let lA := F32.read oA (3*P).toUSize
  let lX := F32.read oX (3*P).toUSize
  IO.println s!"  ④ %loss  adam {lA}   wx {lX}   {if lA == lX then "BIT-EXACT" else "DIFFER"}"

  -- ── ①/② the partition, recovered per parameter ──
  let wd : Float := 1.0e-4      -- `vitAdamConsts`' baked `%wd`
  let mut off := 0
  let mut wrong : List String := []
  let mut nExactSeen := 0
  let mut nOffsetSeen := 0
  let mut worstOffRel : Float := 0.0
  let mut degenerate : List String := []
  for i in [0:sig.length] do
    let (nm, ds) := sig[i]!
    let n := ds.foldl (· * ·) 1
    let (d, m, e) := cmpAt oA oX off n
    -- the predicted offset at an EXCLUDED param: θ'_wx − θ'_adam = lr·wd·θ
    -- ⚠ THE ERROR IS MEASURED IN ULPs OF θ', NOT AS A RELATIVE BOUND, and that is forced rather
    -- than lenient. `θ'_wx − θ'_adam` is a difference of two nearly-equal f32 numbers, so the
    -- best achievable relative accuracy is `ulp(θ') / (lr·wd·|θ|)`. At wd = 1e-4 and |θ| ~ 0.02
    -- that ratio is ~17, i.e. a **~6e-2 relative floor** — and it is INDEPENDENT of lr, because
    -- both the difference and |θ'| scale with it (turning `%lr` up is what fixed the EMA gate;
    -- it does nothing here). A first version gated ② at an absolute 1e-3 and FAILED A CORRECT
    -- RENDER at 1.39e-2, which is inside that floor. §2d.1's rule: calibrate against what the
    -- instrument can resolve, never against a round number.
    let mut predMax : Float := 0.0
    let mut absErr  : Float := 0.0
    let mut ulpMax  : Float := 0.0
    for k in [0:n] do
      let want := lrF * wd * F32.read θ (off + k).toUSize
      let tp   := F32.read oX (off + k).toUSize
      let got  := tp - F32.read oA (off + k).toUSize
      if want.abs > predMax then predMax := want.abs
      if (want - got).abs > absErr then absErr := (want - got).abs
      if tp.abs * 1.1920929e-7 > ulpMax then ulpMax := tp.abs * 1.1920929e-7
    let relErr := absErr
    let bitExact := e == n
    if bitExact then nExactSeen := nExactSeen + 1 else nOffsetSeen := nOffsetSeen + 1
    -- ⚠ a param whose predicted offset is ~0 cannot be classified at all — say so rather than
    -- counting it as agreement. (It does not arise at this init, which is why `mkParam` above
    -- deliberately does not use the driver's zeros for the 1-D params.)
    if predMax < 1e-12 then degenerate := degenerate ++ [nm]
    let empiricalDecayed := bitExact
    if empiricalDecayed != wn.mask nm ds then
      wrong := wrong ++ [s!"{nm}{ds} render={if bitExact then "decayed" else "excluded"} \
expected={if wn.mask nm ds then "decayed" else "excluded"}"]
    if !bitExact && ulpMax > 0.0 then
      let r := relErr / ulpMax          -- the error in ULPs of θ' — the unit the f32 output has
      if r > worstOffRel then worstOffRel := r
    if bitExact && d != 0.0 then
      wrong := wrong ++ [s!"{nm}: classified exact but max abs {d}"]
    off := off + n
    let _ := m
  IO.println s!"  ① θ' BIT-EXACT on {nExactSeen} params   ② θ' OFFSET on {nOffsetSeen} params"
  IO.println s!"  ② worst |(θ'_wx − θ'_adam) − lr·wd·θ| = {worstOffRel} ULPs of θ' \
(the f32 output's own unit; a relative bound here would be measuring the format, not the render)"

  if !degenerate.isEmpty then
    throw (IO.userError s!"DEGENERATE: {degenerate.length} params have a ~0 predicted offset \
({degenerate.take 4}) — they cannot be classified and the partition check is vacuous for them")
  if eM != P || eV != P then
    throw (IO.userError s!"③ FAILED: m'/v' are NOT bit-exact ({P - eM} / {P - eV} coords differ). \
AdamW's decay is DECOUPLED — it is applied to θ directly and must not reach either moment. A \
difference here means the exclusion was wired into the gradient path, i.e. COUPLED L2, which is a \
different optimizer (Loshchilov & Hutter).")
  if lA != lX then
    throw (IO.userError s!"④ FAILED: %loss differs ({lA} vs {lX}) — it is a forward-only output \
and cannot see a change to the optimizer tail. That localises the difference to the wrong half.")
  if !wrong.isEmpty then
    throw (IO.userError s!"①/② FAILED — THE MASK IS WRONG on {wrong.length} params. The empirical \
partition must EQUAL the renderer's mask, name for name; a count alone is satisfied by any {nDec}. \
First few: {wrong.take 4}")
  if nExactSeen != nDec || nOffsetSeen != nExc then
    throw (IO.userError s!"①/② FAILED: {nExactSeen}/{nOffsetSeen} decayed/excluded, expected \
{nDec}/{nExc}")
  -- 4 ULPs: the difference is one subtraction of two rounded f32s, so 1 ULP each side plus the
  -- rounding of `θ'` itself. Anything at or under this is the format, not the render.
  if worstOffRel > 4.0 then
    throw (IO.userError s!"② FAILED: the offset at excluded params is not lr·wd·θ \
({worstOffRel} ULPs of θ', over a 4-ULP bar)")
  IO.println s!"✓ wdExcludeNormBias: the empirical partition EQUALS the renderer's mask on all \
{sig.length} params ({nDec} decayed bit-exact, {nExc} excluded offset by lr·wd·θ to \
{worstOffRel} ULPs), with m'/v' bit-exact on all {2*P} moment coordinates and %loss bit-exact"
