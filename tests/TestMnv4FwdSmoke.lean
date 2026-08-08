import LeanMlir
import LeanMlir.Proofs.Codegen.MobileNetV4RenderB

/-! # MNv4 forward-chain structural smoke (`planning/mnv4_verified.md` phase 3)

Renders `mnv4FwdChainB` and counts the ops it emitted against what the Conv-S block table
says it should have. This catches a **family dispatch** error — an FFN block that emitted
depthwises, or a ConvNeXt-like block that emitted two — because those change the counts.

⚠ It does **not** catch a pre/post-DW swap. Same `k`, same channels ⇒ same op counts as well
as the same parameter shapes. Only a forward tie against the reference pins the order. Stated
in every file that touches this, because a green structural gate is exactly when it gets
forgotten.
-/

open Proofs.StableHLO

/-- Expected `feature_group_count = c` multiset over the 20 depthwise convs, derived by hand from
    the Conv-S table: a pre-DW groups at `ic`, a post-DW at `mid = ic * expand`. This is the strong
    half of the gate — it pins each depthwise to a BLOCK, where a bare total would not. -/
def expectedDwGroups : List (Nat × Nat) :=
  [ (48, 1),    -- b1 pre  (ic 48)
    (192, 1),   -- b1 post (48*4)
    (80, 1),    -- b2 pre
    (160, 7),   -- b2 post (80*2) + pre of b4,b5,b6,b8,b10,b11
    (480, 1),   -- b3 post (80*6)
    (640, 4),   -- post of b4,b5,b7,b10 (160*4)
    (960, 1),   -- b11 post (160*6)
    (256, 2),   -- b12,b14 pre
    (1024, 2) ] -- b12,b13 post (256*4)

def main : IO Unit := do
  let (code, logits) := (mnv4FwdChainB 2 10 "1.0e-05").run' 0
  let lines := code.splitOn "\n"
  let n (pat : String) : Nat := (lines.filter (fun l => (l.splitOn pat).length > 1)).length
  -- ⚠ Parse the group count NUMERICALLY. A substring match on "feature_group_count = 1" also
  -- matches "= 160", "= 192" and "= 1024", which is how the first version of this gate reported
  -- 42 regular convs instead of 32 — a false FAILURE on a correct render, which is the more
  -- expensive direction of wrong for a gate.
  let fgcOf (l : String) : Option Nat :=
    match l.splitOn "feature_group_count = " with
    | _ :: rest :: _ => (rest.takeWhile Char.isDigit).toNat?
    | _ => none
  let fgcs : List Nat := lines.filterMap fgcOf
  let grp (c : Nat) : Nat := (fgcs.filter (· == c)).length
  IO.FS.writeFile ".lake/build/mnv4_fwd_chain_smoke.mlir" code
  IO.println s!"  rendered {lines.length} lines, logits {logits}"
  let mut bad := 0
  let chk (what : String) (got want : Nat) : IO Bool := do
    if got == want then IO.println s!"  ✓ {what}: {got}"; pure true
    else IO.println s!"  ✗ {what}: got {got}, want {want}"; pure false
  if !(← chk "regular convs (fgc = 1)" (grp 1) 32) then bad := bad + 1
  if !(← chk "swish (fused stage only)" (n "stablehlo.logistic") 1) then bad := bad + 1
  if !(← chk "relu" (n "stablehlo.maximum") 36) then bad := bad + 1
  -- (No skip-add count: `addVB` emits a bare `stablehlo.add`, indistinguishable from the many
  -- adds inside each expanded batch-BN. 273 of them, so the check would be noise, not a gate.)
  if !(← chk "total convs" fgcs.length 52) then bad := bad + 1
  -- Per-block depthwise widths: the strong check.
  let mut dwTot := 0
  for (c, want) in expectedDwGroups do
    let got := grp c
    dwTot := dwTot + got
    if got != want then
      IO.println s!"  ✗ depthwise @ groups={c}: got {got}, want {want}"; bad := bad + 1
  if !(← chk "depthwise total" dwTot 20) then bad := bad + 1

  -- ── the module wrapper: self-containment and the layout tie ──
  let m := mnv4FwdFaithfulV 2 10 "1.0e-05"
  IO.FS.writeFile ".lake/build/mnv4_fwd.mlir" m
  let mlines := m.splitOn "\n"
  -- Every %zbN the body references must be bound by the prelude. An unbound one is an
  -- undefined SSA name — caught here at `lake build` rather than at iree-compile.
  let refZb : List Nat := mlines.filterMap (fun l =>
    match l.splitOn "%zb" with
    | _ :: rest :: _ => (rest.takeWhile Char.isDigit).toNat?
    | _ => none)
  let boundZb : List Nat := mlines.filterMap (fun l =>
    if (l.splitOn "= stablehlo.constant").length > 1 then
      match l.splitOn "%zb" with
      | _ :: rest :: _ => (rest.takeWhile Char.isDigit).toNat?
      | _ => none
    else none)
  let unbound := refZb.filter (fun c => !(boundZb.contains c))
  if unbound.isEmpty then IO.println s!"  ✓ all %zb widths bound ({boundZb.length} constants)"
  else
    IO.println s!"  ✗ UNBOUND %zb widths: {unbound.eraseDups}"; bad := bad + 1

  -- ⭐ The signature and VLayer.toSpecs are two hand-written readings of one layout. Tie them.
  let sigShapes := (mnv4ShapeList 10).map (fun (_, ds) => ds)
  let specShapes : List (List Nat) :=
    ([VLayer.convBnNB 3 32 3 2, VLayer.fusedMbConvNB 32 48 4 3 2,
      VLayer.uib  48  80 4 2 3 5, VLayer.uib  80  80 2 1 3 3, VLayer.uib  80 160 6 2 0 3,
      VLayer.uib 160 160 4 1 3 3, VLayer.uib 160 160 4 1 3 5, VLayer.uib 160 160 4 1 5 0,
      VLayer.uib 160 160 4 1 0 3, VLayer.uib 160 160 4 1 3 0, VLayer.uib 160 160 4 1 0 0,
      VLayer.uib 160 160 4 1 3 3, VLayer.uib 160 256 6 2 5 5, VLayer.uib 256 256 4 1 5 5,
      VLayer.uib 256 256 4 1 0 3, VLayer.uib 256 256 4 1 3 0,
      VLayer.convBnNB 256 1280 1 1, VLayer.dense 1280 10]).flatMap
        (fun l => (l.toSpecs.map (fun (d, _) => d.toList)).toList)
  if sigShapes == specShapes then
    IO.println s!"  ✓ signature ties VLayer.toSpecs: {sigShapes.length} params, shape-for-shape"
  else
    IO.println s!"  ✗ SIGNATURE/LAYOUT MISMATCH: sig has {sigShapes.length}, spec has {specShapes.length}"
    let z := sigShapes.zip specShapes
    for (a, b) in (z.filter (fun (a, b) => a != b)).take 5 do
      IO.println s!"      sig {a} vs spec {b}"
    bad := bad + 1
  let tot := sigShapes.foldl (fun acc d => acc + d.foldl (· * ·) 1) 0
  IO.println s!"  total params: {tot}"
  if bad != 0 then
    IO.eprintln s!"MNV4 FORWARD SMOKE FAILED ({bad} mismatches)"
    IO.Process.exit 1
  IO.println "  ✓ mnv4 forward chain: op counts and per-block depthwise widths match the Conv-S table"
