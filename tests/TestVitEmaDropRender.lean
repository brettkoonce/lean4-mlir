import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.ViTRenderB

/-! # EMA **and** stochastic depth in one ViT render: the arity the driver expects is the arity the
    artifact has

`lake build vit-ema-drop-render && .lake/build/bin/vit-ema-drop-render`

`verified_mlir/vitin_emadp128x4wxclipdropbf16_train_step.mlir` is the first render anywhere in this
repo that carries **EMA and stochastic depth at the same time**, and it exists because the ImageNet
ViT pair needs both: the phase-2 reference (blueprint §9.6) trains with `useEMA := true` *and*
`dropPath := 0.1`, and until 2026-08-29 the only committed EMA render for this net,
`vitin_emadp128x4`, was emitted with `(ema := true)` alone — no `wx`, no `clip`, no `sd`. Choosing
it to obtain EMA therefore silently gave up gradient clipping, which §9.6 measures as load-bearing
for ViT-Ti: without it the model collapses to chance the moment warmup ramps past ~1.6e-4.

**What this file gates, and why a string test cannot.** `tests/TestVariantPredicates.lean` already
pins the five axis predicates against the variant *name*. That is necessary and not sufficient. The
predicates decide how many regions the driver packs into the blob and how long the scalar tail is;
whether the **artifact** agrees is a different proposition, and it is the one that fails silently.
`planning/ema.md` records the failure mode directly: a wrongly-packed region **trains and reports a
loss**. There is no crash and no NaN to notice — the run simply optimises a misaligned view of its
own parameters and produces a number that looks like an answer.

So this reads the committed artifact and checks its measured arity against
`VerifiedVariant.nRegions` / `nScalars` / `emaRegion` computed from the same string the driver will
see, plus the stochastic-depth mask count the net's own `dropKeeps` implies. Two independent
derivations of one number, which is the only shape of check that catches a packing error.

⚠ **The two axes are checked as a PAIR, deliberately.** Each is exercised alone by existing
artifacts (`vitin_emadp128x4` for EMA, `vitin_adamdp128x4wxclipdrop` for SD), and both passed for
months while their composition did not exist. This repo's naming has collided three times and every
one was two markers *meeting* rather than a new marker misbehaving — `TestVariantPredicates`'
docstring is that history. The same reasoning applies a level down: EMA adds a region, SD adds
operands, and nothing had ever forced the emitter to lay both out at once.

⚠ **What this does NOT establish.** That the EMA update is numerically right — `opt_step_tie.py`'s
`ema*` rows are that, against the reference's own `ema_update`. This file is about *layout*: the
right number of things in the right order. A shadow that decays at the wrong rate has correct
arity and passes here (and would have: `trainAdamSched`'s `emaDecay` defaults to 0.9999 against
the reference's 0.99996, which is why `MainViTImagenet.lean` now passes it by name).

No GPU — it is a parse and three counts, so it belongs in a pre-commit sweep rather than behind a
device.
-/

open Proofs.StableHLO

/-- The variant the ImageNet ViT pair runs, spelled once. -/
def variantUnderTest : String := "emadp128x4wxclipdropbf16"

/-- Count `func.func` operands whose name matches a prefix, by parsing the committed signature.
    Deliberately a parse of the ARTIFACT rather than a re-render: a re-render that shares the
    emitter's bug reproduces it, and the file on disk is what the FFI will actually load. -/
def operandNames (src : String) : List String :=
  match (src.splitOn "func.func @").tail? with
  | some (body :: _) =>
    match (body.splitOn ")").head? with
    | some sig =>
      (sig.splitOn "%").tail!.filterMap fun frag =>
        match (frag.splitOn ":").head? with
        | some nm => if nm.isEmpty then none else some nm.trim
        | none    => none
    | none => []
  | _ => []

/-- Fail via `throw`, never `IO.Process.exit` — under `#eval` the elaborator buffers output and
    `exit` discards every diagnostic. -/
def main : IO Unit := do
  let path := s!"verified_mlir/vitin_{variantUnderTest}_train_step.mlir"
  let src ← IO.FS.readFile path
  let names := operandNames src
  let net := vitImagenetVerified.toNet

  IO.println "── ViT: EMA + stochastic depth in one render ──"
  IO.println s!"  artifact : {path}"
  IO.println s!"  operands : {names.length}"

  -- What the DRIVER will do, from the variant string alone.
  let wantRegions := VerifiedVariant.nRegions variantUnderTest
  let wantScalars := VerifiedVariant.nScalars variantUnderTest
  let wantEmaReg  := VerifiedVariant.emaRegion variantUnderTest
  let nP          := net.specs.size
  let wantMasks   := net.dropKeeps.size

  IO.println s!"  driver   : {wantRegions} regions, {wantScalars} scalars, ema region {wantEmaReg}, {wantMasks} drop masks, {nP} params"

  let mut bad : List String := []

  -- 1. Both axes must actually be ON for this to be the test it claims to be. A predicate that
  --    quietly reads false would make every count below vacuously agree with a 3-region graph.
  unless VerifiedVariant.emaOn variantUnderTest do
    bad := bad ++ [s!"emaOn '{variantUnderTest}' is FALSE — this file would gate nothing"]
  unless VerifiedVariant.sdOn variantUnderTest do
    bad := bad ++ [s!"sdOn '{variantUnderTest}' is FALSE — this file would gate nothing"]
  unless wantRegions == 4 do
    bad := bad ++ [s!"nRegions = {wantRegions}, expected 4 ([θ|m|v|ema])"]
  unless wantScalars == 5 do
    bad := bad ++ [s!"nScalars = {wantScalars}, expected 5 (lr,bc1,bc2,emad,oemad)"]
  unless wantEmaReg == some 3 do
    bad := bad ++ [s!"emaRegion = {wantEmaReg}, expected some 3"]

  -- 2. The scalar tail is NAMED in the artifact, so check the names rather than only the count —
  --    five scalars in the wrong order packs the same width and misreads every one of them.
  let tail := ["lr", "bc1", "bc2", "emad", "oemad"]
  for s in tail do
    unless names.contains s do
      bad := bad ++ [s!"scalar operand %{s} is missing from the artifact"]

  -- 3. Stochastic depth survived the composition: one `dpN` operand per drop site.
  let masks := names.filter (fun n => n.startsWith "dp" && (n.drop 2).all Char.isDigit)
  unless masks.length == wantMasks do
    bad := bad ++ [s!"{masks.length} dp* mask operands, but the net declares {wantMasks} drop sites"]

  -- 4. The EMA region is a FULL param-sized region, not a truncated one. The whole signature is
  --    `%x` + `%onehot` + θ|m|v|ema + scalars + masks, so the arithmetic closes only if every
  --    region is `nP` wide — which is exactly what a mispacked shadow gets wrong.
  --    ⚠ The two non-param operands are BOTH data: `%x` the images and `%onehot` the targets. The
  --    latter is a `tensor<B×1000xf32>` DISTRIBUTION rather than int labels, which is what lets
  --    Mixup/CutMix ride the shim; forgetting it is an off-by-one this file caught on its first run.
  let dataOperands := 2
  for nm in ["x", "onehot"] do
    unless names.contains nm do
      bad := bad ++ [s!"data operand %{nm} is missing from the artifact"]
  let expected := dataOperands + wantRegions * nP + wantScalars + wantMasks
  unless names.length == expected do
    bad := bad ++ [s!"operand count {names.length} ≠ 2 + {wantRegions}×{nP} + {wantScalars} + {wantMasks} = {expected}"]

  -- 5. bf16 actually reached the body. The name says bf16 and a name is a label, not evidence —
  --    the same lesson as §0.5's "list what the artifact BAKES".
  let bf16Ops := (src.splitOn "bf16").length - 1
  unless bf16Ops > 100 do
    bad := bad ++ [s!"only {bf16Ops} bf16 mentions — the render is not actually bf16"]

  if bad.isEmpty then
    IO.println s!"  masks    : {masks.length}  scalars: {tail.length} named and present"
    IO.println s!"  arity    : {names.length} = 2 + {wantRegions}×{nP} + {wantScalars} + {wantMasks} ✓"
    IO.println s!"  bf16     : {bf16Ops} mentions in the body ✓"
    IO.println "  ✅ EMA and stochastic depth compose: the artifact's layout is the one the driver packs"
  else
    for b in bad do IO.println s!"  ✗ {b}"
    throw <| IO.userError s!"{bad.length} check(s) failed — see above"
