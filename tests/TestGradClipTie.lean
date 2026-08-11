import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.ViTRender
import LeanMlir.Proofs.Codegen.ConvNeXtRender

/-! # Global-norm gradient clipping, numerically certified (`recipe_gaps.md` v1.4b)

`verified_mlir/<net>_adamclip_train_step.mlir` is `<net>_adam` with the reference's two extra lines
in front of the optimizer (`jax/Jax/Codegen.lean:2262`):

    gn    = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, CLIP / (gn + 1e-6)), grads)

`vitTinyImagenetConfig` and `convnextTinyImagenetConfig` both set `gradClipNorm := 1.0`; the
reference calls it *"the unlock for the 5e-4 LR"*. EfficientNet sets **0.0 deliberately** and
R34/mnv2 do not use it, so this harness covers the two nets that have it and no others.

**Why a cross-render KNOWN ANSWER and not a tie.** The two renders compute different functions, and
every structural check passes on both — the clip moves no arity, no type and no region. What
separates them is one scalar, and at `m = v = 0` the moment slot recovers it exactly:
`adamMNext β₁ 0 g = (1−β₁)·g = 0.1·g`, so

    m'_clip[k] / m'_adam[k]  =  fac  =  min(1, CLIP/(gn + 1e-6))

for **every coordinate of every parameter** — one number, 5.5M times.

**▶ THE CONSTANCY OF THAT RATIO IS THE GATE, and that is the whole design.** The norm being GLOBAL
is the entire semantic content of the feature, and it is the only thing a per-parameter clip gets
wrong: a per-parameter clip scales, never amplifies, and is the identity below the threshold —
`Proofs.clipFactor_le_one`, `clipFactor_eq_one_below` and every other property in `GradClip.lean`
hold for it. It differs from the reference in exactly one place, `clipFactor_shared`, and a gate
that asks *"did the gradients get smaller"* passes it. `scripts/perturb_clip.py perparam` builds it.

| gate | claim |
|---|---|
| ⓪ | the clip is ACTIVE (`fac < 1` by a margin) — else ①-③ are vacuous and the run REFUSES |
| ① | `m'_clip/m'_adam` is ONE CONSTANT across all params — the global-vs-per-parameter distinction |
| ② | that constant equals `min(1, CLIP/(gn+1e-6))` for the host-recomputed `gn` |
| ③ | `%loss` is bit-exact — a forward-only output cannot see a change to the gradient path |
| ④ | at a threshold ABOVE the norm the render is **BYTE-IDENTICAL** to the unclipped one |

⚠ **② is the weakest of the five and says so in its own output.** `gn` on device is an f32 fold of
~5.5M positive squares; the host recomputes it in f64 from gradients themselves recovered through
one f32 rounding. Those cannot agree to the bit and nothing is wrong when they do not — it is the
mixup-λ finding (*recover a constant by READING it, not by fitting it*) hitting a quantity that
genuinely has no exact host reading. **① reads the factor off the device and needs no host norm at
all**, which is why it, not ②, is the one with a tight bound.

⚠ **④ is BIT-EXACT BY ARGUMENT, not by luck**: above the threshold `min(1, c/(gn+ε))` is exactly
`1.0` and `x * 1.0` is exact in IEEE-754 binary32 (`Proofs.clipScaleF_id_below`). But it is also
**blind on its own** — at factor 1 a global clip and a per-parameter clip are the SAME FUNCTION, so
④ passes on `perparam`. That is the stochastic-depth ones-mask finding one feature over: an identity
gate cannot see where, or from what, the intervention was computed. ① and ④ are evidence together.

⚠⚠ **RUN IT UNDER `scripts/det_shim.sh`. THIS IS NOT OPTIONAL AND IT IS NOT A STYLE POINT.**
Without it, ConvNeXt reads gate ④ as 137,229 of 83,478,847 outputs differing at a 24,149-ULP floor
— carried by ONE parameter of 180, `d0W`'s even-kernel 2×2/s2 downsample weight gradient — and the
number MOVES between runs. With it: **83,478,847/83,478,847 bit-identical, gate ① at 1.15 ULPs.**
Nothing about the render changes; XLA autotuning picks a different convolution algorithm per
process. ViT happens to be clean either way, which is exactly how this trap stays hidden: a gate
developed on ViT and ported to ConvNeXt inherits ViT's conditioning (handoff §0.4, finding 1, now
in a fourth place). Gate ④ refuses with the recipe rather than reporting a phantom defect.

    lake build clip-tie
    scripts/det_shim.sh /tmp/detshim
    python3 scripts/perturb_clip.py verified_mlir/vit_adamclip_train_step.mlir \
      .lake/build/clip_hi_vit.mlir hi
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/detshim .lake/build/bin/clip-tie vit       # 200
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/detshim .lake/build/bin/clip-tie convnext  # 180

**ONE harness, both nets**, per `wdx-tie` / `rms-tie` / `shard-check`: a second copy is the
double-writer disease one level down, in code. ⚠ And running both is not ceremony — the same edit
behaved differently on the two renderers twice now (ConvNeXt derives its entry name from the
variant, ViT takes it explicitly), and this thread found a third instance of it.

The controls, each of which must be verified to FIRE:

    python3 scripts/perturb_clip.py <committed> /tmp/c.mlir perparam   # ① fires — the real one
    python3 scripts/perturb_clip.py <committed> /tmp/c.mlir nosqrt     # ② fires
    python3 scripts/perturb_clip.py <committed> /tmp/c.mlir epsout     # ② fires
    LD_LIBRARY_PATH=/tmp/detshim .lake/build/bin/clip-tie vit --cand /tmp/c.mlir

Measured 2026-08-02, under the det shim, all six red and rc=1:

| control | ViT | ConvNeXt | fires |
|---|---|---|---|
| `perparam` (per-parameter norm) | **7,601,258 ULPs** vs an 8-ULP bar | **30,372,642 ULPs** | ① |
| `nosqrt`   (clip on ‖g‖²) | 961,717 ppm vs a 5 ppm bar | 977,995 ppm | ② |
| `epsout`   (`c/gn + ε`) | **26.28 ppm** (predicted ε/fac = 26.1) | **45.46 ppm** (45.5) | ② |

against true-render readings of ① **1.12 / 1.15 ULPs** and ② **0.105 / 0.0070 ppm**.
⚠ `perparam` passes ⓪, ③ and ④ — it is a working clip, just not a GLOBAL one — and `nosqrt`
passes ①, because ‖g‖² is still one shared scalar. The gates are independent on purpose.
-/

open Proofs.StableHLO

private structure ClipNet where
  slug : String
  sig  : List (String × List Nat)
  spec : VerifiedNetSpec

private def netBySlug (s : String) : IO ClipNet :=
  match s with
  | "vit"      => pure { slug := "vit", sig := vitParamSig 10, spec := vitVerified }
  | "convnext" => pure { slug := "convnext", sig := cnxAllParams 10, spec := convnextVerified }
  | _ => throw (IO.userError s!"unknown net '{s}' — expected vit | convnext")

/-- The driver's init with `wdx-tie`'s one deliberate change: kind-2 params get small centred noise
    instead of zero. Here the reason is different and weaker than there — a zero θ does not make
    this gate vacuous — but a net whose 1-D params are all zero has a smaller gradient norm, and ⓪
    needs the norm ABOVE the threshold. Same function, so the two harnesses stay comparable.

    ⚠ Keep the He scaling on the weights. `wdx-tie` measured what happens otherwise: every param
    centred at 0.6 overflowed the ViT forward, and the harness reported `%loss NaN` with
    `0/5,526,346` bit-exact and `max abs 0.000000` — the tell being that NaN ≠ NaN makes every
    coordinate "differ" while every `>` comparison stays false. -/
private def mkParam (seed : Nat) (dims : Array Nat) (kind : Nat) : IO ByteArray := do
  let n := dims.foldl (· * ·) 1
  match kind with
  | 1 => F32.const n.toUSize 1.0
  | 2 => F32.heInit seed.toUSize n.toUSize 0.02
  | _ =>
    let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
    F32.heInit seed.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat))

def main (argv : List String) : IO Unit := do
  let optArg (flag : String) : Option String :=
    match argv.dropWhile (· != flag) with
    | _ :: p :: _ => some p
    | _ => none
  let cand := optArg "--cand"
  let net0 := (argv.filter (fun a => !a.startsWith "--" && !a.startsWith "/"
                                     && !a.startsWith ".")).head?.getD "vit"
  let cn ← netBySlug net0
  let net := cn.spec.toNet
  let sig := cn.sig
  let bs  := 32
  let P   := net.nParams
  let clipC : Float := 1.0        -- the baked `%clip` in `<net>_adamclip_train_step.mlir`
  let clipE : Float := 1.0e-6     -- the baked `%eps`  (the reference's literal)
  let ob1   : Float := 0.1        -- 1 − β₁, so `m' = ob1·g` at m = 0
  let hiPath := (optArg "--hi").getD s!".lake/build/clip_hi_{cn.slug}.mlir"
  IO.println s!"grad clip — {cn.slug}, global-norm clipping at CLIP = {clipC} (v1.4b)"
  IO.println s!"  {sig.length} params / {P} coords, bs {bs}, backend {← LowererSession.backendName}"

  -- ⚠ TWO ROUTES TO THE SAME LAYOUT, checked — `wdx-tie`'s check, kept for §2m's reason: the
  -- signature list drives the render's per-param loop while `net.specs` drives the driver's blob,
  -- and this gate reads offsets from one having taken the parameter count from the other.
  if net.specs.size != sig.length then
    throw (IO.userError s!"LAYOUT SKEW: net.specs has {net.specs.size} entries, the signature \
list has {sig.length} — every offset below would be meaningless")
  for i in [0:sig.length] do
    let (nm, ds) := sig[i]!
    let (dims, _) := net.specs[i]!
    if dims.toList != ds then
      throw (IO.userError s!"LAYOUT SKEW at {i} ({nm}): net.specs says {dims.toList}, \
the signature list says {ds}")

  -- ── one shared (θ, x, onehot); m = v = 0 so the moment slot recovers the gradient ──
  let mut parts : Array ByteArray := #[]
  let mut sd := 4242
  for i in [0:sig.length] do
    let (dims, kind) := net.specs[i]!
    parts := parts.push (← mkParam sd dims kind); sd := sd + 1
  let θ := F32.concat parts
  let z ← F32.const P.toUSize 0.0
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

  -- ⚠ `variant` names the ENTRY as well as the path, and the two must agree — the shim refuses a
  -- mismatch outright rather than running the wrong graph. That check earned its keep on this very
  -- feature: a second clip render at a different threshold spelled the same Bool-derived ConvNeXt
  -- variant and came out declaring another artifact's entry (`planning/grad_clip.md` §6).
  let run (variant : String) (path : Option String := none) : IO ByteArray := do
    let vmfb := s!".lake/build/clip_tie_{cn.slug}_{variant}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    -- delete first: `compileVmfb`'s cache key is the OUTPUT path plus an mtime, never the source,
    -- so a re-run with a different candidate under the same tag silently reuses the first binary
    -- and reports the second as a perfect match (§4's `.vmfb` false-PASS hazard).
    for p in [vmfb, s!".lake/build/clip_tie_{cn.slug}_{variant}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let src := path.getD s!"verified_mlir/{cn.slug}_{variant}_train_step.mlir"
    let sess ← mkSession src
    LowererSession.mlpTrainStepV sess s!"m.{cn.slug}_{variant}_train_step" x buf shapes y
      bs.toUSize net.d0.toUSize net.nClasses.toUSize

  if cand.isSome then IO.println s!"  ⚠ CANDIDATE clip render: {cand.get!}"
  let oA ← run "adam"
  let oC ← run "adamclip" cand

  -- ── ④ FIRST, because it is also ①'s FLOOR ──
  -- The `hi` render has the clip block's exact graph structure with the factor pinned at 1.0. So
  -- it answers a question ① cannot ask of itself: **does the clip block's mere presence perturb
  -- the gradient?** If `hi` is bit-identical to `adam`, `g` is the same number in both graphs and
  -- any spread ① sees is the render; if it is not, XLA scheduled the backward differently and ①
  -- must be read against that, not against an absolute ULP count. Measuring the floor before
  -- reading a cross-graph number is §2d.3's rule, and it has twice been the difference between a
  -- green feature and a phantom defect.
  if !(← System.FilePath.pathExists hiPath) then
    throw (IO.userError s!"④ CANNOT RUN: {hiPath} does not exist. It is GENERATED, not committed — \
an artifact baking a threshold no config sets is a silent-hyperparameter artifact. Build it with:\n\
  python3 scripts/perturb_clip.py verified_mlir/{cn.slug}_adamclip_train_step.mlir {hiPath} hi\n\
Refusing to report a pass without it: it is both gate ④ and the floor ① is read against.")
  let oH ← run "adamclip" (some hiPath)

  let mut diff := 0
  let mut floorUlp : Float := 0.0
  for i in [0:3*P + 1] do
    if F32.read oA i.toUSize != F32.read oH i.toUSize then
      diff := diff + 1
      let a := F32.read oA i.toUSize
      let h := F32.read oH i.toUSize
      if a.abs > 1.0e-18 then
        let r := ((h - a) / a).abs / 1.1920929e-7
        if r > floorUlp then floorUlp := r
  IO.println s!"  ④ threshold ABOVE the norm: {3*P + 1 - diff}/{3*P + 1} outputs BIT-IDENTICAL \
to the unclipped render{if diff == 0 then "" else s!"  ⚠ FLOOR {floorUlp} ULPs on {diff} outputs"}"
  -- ⚠ WHICH parameters carry the floor, in the m' region. A floor concentrated in a handful of
  -- known ill-conditioned reduces is a scheduling story; one spread over every parameter would be
  -- a wiring story, and they need different responses.
  if diff != 0 then
    let mut wide : List String := []
    let mut off2 := 0
    for i in [0:sig.length] do
      let (nm, ds) := sig[i]!
      let n := ds.foldl (· * ·) 1
      let mut w : Float := 0.0
      for k in [0:n] do
        let a := F32.read oA (P + off2 + k).toUSize
        let h := F32.read oH (P + off2 + k).toUSize
        if a.abs > 1.0e-18 && a != h then
          let r := ((h - a) / a).abs / 1.1920929e-7
          if r > w then w := r
      if w > 8.0 then wide := wide ++ [s!"{nm}:{w}"]
      off2 := off2 + n
    IO.println s!"     floor is carried by {wide.length}/{sig.length} params: {wide.take 6}"

  let mut nonFin := 0
  for i in [0:3*P + 1] do
    if !(F32.read oA i.toUSize).isFinite || !(F32.read oC i.toUSize).isFinite then
      nonFin := nonFin + 1
  if nonFin > 0 then
    throw (IO.userError s!"DEGENERATE: {nonFin} non-finite outputs — the INIT blew the forward up, \
not the render. Nothing below this line would mean anything.")

  -- ── recover the factor from the m' region, per parameter ──
  -- `m'_adam = ob1·g` and `m'_clip = ob1·(fac·g)`, both at m = 0, so the ratio IS `fac` up to the
  -- two f32 roundings. Coordinates whose `m'_adam` is denormal-small are skipped: their ratio is
  -- dominated by the rounding, not by `fac`, and counting them would widen the spread for a
  -- reason that has nothing to do with the render.
  let mut facMin : Float := 1.0e30
  let mut facMax : Float := -1.0e30
  let mut facSum : Float := 0.0
  let mut nUsed  := 0
  let mut perParamMin : Array Float := #[]
  let mut perParamMax : Array Float := #[]
  let mut gn2 : Float := 0.0            -- Σ g² in f64, for ②
  let mut off := 0
  for i in [0:sig.length] do
    let (_, ds) := sig[i]!
    let n := ds.foldl (· * ·) 1
    let mut pMin : Float := 1.0e30
    let mut pMax : Float := -1.0e30
    for k in [0:n] do
      let a := F32.read oA (P + off + k).toUSize
      let c := F32.read oC (P + off + k).toUSize
      let g := a / ob1
      gn2 := gn2 + g * g
      if a.abs > 1.0e-18 then
        let r := c / a
        if r < pMin then pMin := r
        if r > pMax then pMax := r
        if r < facMin then facMin := r
        if r > facMax then facMax := r
        facSum := facSum + r
        nUsed := nUsed + 1
    perParamMin := perParamMin.push pMin
    perParamMax := perParamMax.push pMax
    off := off + n
  if nUsed == 0 then
    throw (IO.userError "DEGENERATE: every m' coordinate is ~0 — nothing to recover a factor from")
  let fac := facSum / nUsed.toFloat
  let gn := Float.sqrt gn2
  let want := min 1.0 (clipC / (gn + clipE))

  IO.println s!"  ⓪ recovered factor  {fac}   (min {facMin}, max {facMax}, over {nUsed} coords)"
  IO.println s!"  ② host ‖g‖ = {gn}  ⇒  predicted min(1, {clipC}/(‖g‖+{clipE})) = {want}"

  -- ── ⓪ the clip must be ACTIVE, or ① and ② are statements about the number 1 ──
  if fac > 0.999 then
    throw (IO.userError s!"VACUOUS: the recovered factor is {fac} — the gradient norm ({gn}) is at \
or below the threshold ({clipC}), so `min(1, ·)` returned the identity and NOTHING about the clip \
was exercised. ① and ④ both pass trivially here, and ④ passes on a per-parameter clip too. \
Condition the instrument by scaling the input, not the LR: the norm is a property of the DATA, and \
both the difference and |θ'| scale with `%lr` (the `wdx-tie` finding — the EMA fix does not \
transfer).")

  -- ── ① THE RATIO IS ONE CONSTANT ACROSS PARAMETERS — the gate that defines the feature ──
  -- Reported in ULPs of the factor: `m'_clip` carries two f32 roundings relative to `m'_adam`
  -- (the `fac·g` product and the `ob1·` one), so ~2 ULPs is the floor and anything at that scale
  -- is the format. A per-parameter clip separates the per-parameter factors by ORDERS.
  let ulp : Float := 1.1920929e-7
  let spread := (facMax - facMin) / (fac * ulp)
  -- the per-parameter reading, which is what actually distinguishes global from local: each
  -- parameter's own [min,max] band must overlap every other's.
  let mut worstParam : Float := 0.0
  for i in [0:sig.length] do
    if perParamMax[i]! > -1.0e29 then
      let d := max ((perParamMax[i]! - fac).abs) ((perParamMin[i]! - fac).abs)
      if d / (fac * ulp) > worstParam then worstParam := d / (fac * ulp)
  IO.println s!"  ① ratio spread across ALL params: {spread} ULPs of the factor \
(worst single param {worstParam} ULPs)"
  IO.println s!"  ③ %loss  adam {F32.read oA (3*P).toUSize}   clip {F32.read oC (3*P).toUSize}"

  if worstParam > 8.0 then
    throw (IO.userError s!"① FAILED — THE FACTOR IS NOT SHARED ({worstParam} ULPs of spread, over \
an 8-ULP bar). The reference's norm is GLOBAL: one scalar folded from every parameter's gradient \
and applied to all of them. A spread this wide means each parameter was scaled by its OWN norm, \
which is a different function that scales, never amplifies, and is the identity below the \
threshold — i.e. it passes every other gate here.")

  -- ── ③ %loss: forward-only, so it cannot see the optimizer tail ──
  let lA := F32.read oA (3*P).toUSize
  let lC := F32.read oC (3*P).toUSize
  if lA != lC then
    throw (IO.userError s!"③ FAILED: %loss differs ({lA} vs {lC}) — it is a forward-only output \
and the clip is on the gradient path. That localises the difference to the wrong half of the graph.")

  -- ── ② the known answer, at the resolution the instrument actually has ──
  let rel := ((fac - want) / want).abs
  -- in ppm, because at 6 decimals a correct render prints "0.000000" and an ε-placement error
  -- prints "0.000026" — two numbers whose RATIO is the whole question, rendered as one digit.
  IO.println s!"  ② |recovered − predicted| / predicted = {rel * 1.0e6} ppm"
  -- ⚠ THE 5 ppm BAR IS CALIBRATED FROM BOTH NETS, NOT PICKED ROUND (§2d.1's rule).
  --   floor    — the true renders measure **0.105 ppm (ViT) / 0.0070 ppm (ConvNeXt)**. That is the
  --              device's f32 fold of 5.5M / 27.8M positive squares against this f64 host sum, plus
  --              one f32 rounding in the recovered `g`.
  --   control  — `epsout` (ε outside the root) lands at **26.3 / 45.5 ppm**, and those are a KNOWN
  --              ANSWER rather than an observation: the error is exactly `ε/fac`, i.e.
  --              1e-6/0.0383 = 26.1 and 1e-6/0.0220 = 45.5. Measured 26.28 / 45.46.
  -- 5 ppm sits ~48×/718× above the floor and 5.3×/9.1× below the control. ⚠ It is the ONE bound in
  -- this harness that is a tolerance rather than a bit-exactness or ULP statement, which is why the
  -- other three carry the weight.
  if rel > 5.0e-6 then
    throw (IO.userError s!"② FAILED: the recovered factor {fac} is not min(1, CLIP/(‖g‖+ε)) = \
{want} ({rel * 1.0e6} ppm, over a 5 ppm bar calibrated against a measured 0.105/0.0070 ppm floor). \
Check, in order: the sqrt (dropping it reads ~960,000 ppm), the ε PLACEMENT — `c/(gn+ε)` not \
`c/gn + ε`, which is worth exactly ε/fac ≈ {clipE / fac * 1.0e6} ppm here — and whether the \
threshold is the one the render baked ({clipC}).")

  if diff != 0 then
    throw (IO.userError s!"④ FAILED: {diff} of {3*P + 1} outputs differ, worst {floorUlp} ULPs.\n\
\n\
⚠⚠ CHECK THE INSTRUMENT FIRST, AND IT IS ALMOST CERTAINLY THE INSTRUMENT. Run under the \
DETERMINISTIC SHIM:\n\
    scripts/det_shim.sh /tmp/detshim\n\
    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/detshim .lake/build/bin/clip-tie {cn.slug}\n\
\n\
Measured on ConvNeXt 2026-08-02: WITHOUT the det shim this read 137,229 of 83,478,847 differing at \
a 24,149-ULP floor carried by ONE parameter of 180 (`d0W`, the even-kernel 2x2/s2 downsample weight \
gradient), and it MOVED between runs. WITH it: 83,478,847/83,478,847 bit-identical and gate 1 at \
1.15 ULPs. Nothing was wrong with the render — XLA autotuning picked a different convolution \
algorithm per process. That is handoff §2d.3 Finding 1 (ROCm-specific) and the standing rule \
*measure the A-vs-A floor before reading ANY cross-graph number*, in a third place.\n\
\n\
If the floor is still non-zero UNDER the det shim, then it is the render: above the threshold the \
factor is EXACTLY 1.0 and `x * 1.0` is exact in binary32 (`Proofs.clipScaleF_id_below`), so this \
is a bit-exactness claim with no tolerance to tune. Check the `min` against 1.0 — it is what stops \
a SMALL gradient being AMPLIFIED by c/‖g‖.")

  IO.println s!"✓ grad clip on {cn.slug}: the factor is ONE SHARED SCALAR across all \
{sig.length} params ({worstParam} ULPs of spread), equals min(1, CLIP/(‖g‖+ε)) to {rel} relative, \
leaves %loss bit-exact, and is bit-identically INERT at a threshold above the norm"
