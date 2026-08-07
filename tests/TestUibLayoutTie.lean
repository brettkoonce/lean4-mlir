import LeanMlir

/-! # `uib` layout tie — `VLayer.toSpecs` against the baseline `Layer.nParams`

Phase 1's gate (`planning/mnv4_verified.md`). `VLayer.toSpecs` (the verified parameter
layout, which is what the driver allocates and threads) and `Layer.nParams` (the baseline
count, which the JAX reference's shim was built against) are **two independent readings of
the same block**. They are written in different files by different means — shapes versus
arithmetic — so agreement is evidence, not tautology.

⚠ **This gate is necessary and NOT sufficient.** A pre-DW and a post-DW at the same `k` and
the same channel count contribute identical parameter shapes, so this check passes on a
renderer that swaps them. It pins the *layout*; only a forward tie against the reference
pins the *order*. Said again here because a green gate is exactly when that gets forgotten.

Block table: the 15 UIB blocks of `jax/MainMobilenetV4.lean` (the Conv-S-sized 4.1M
Imagenette demo — the one `RESULTS.md`'s 84.58% belongs to, NOT faithful Conv-M). All four
families appear: ExtraDW (both DWs), IB (`preDWk = 0`), ConvNext-like (`postDWk = 0`),
FFN (neither).
-/

/-- `(ic, oc, expand, stride, preDWk, postDWk)` — transcribed from `jax/MainMobilenetV4.lean`. -/
def mnv4UibTable : List (Nat × Nat × Nat × Nat × Nat × Nat) :=
  [ ( 48,  80, 4, 2, 3, 5),   -- ExtraDW
    ( 80,  80, 2, 1, 3, 3),   -- ExtraDW
    ( 80, 160, 6, 2, 0, 3),   -- IB
    (160, 160, 4, 1, 3, 3),   -- ExtraDW
    (160, 160, 4, 1, 3, 5),   -- ExtraDW
    (160, 160, 4, 1, 5, 0),   -- ConvNext
    (160, 160, 4, 1, 0, 3),   -- IB
    (160, 160, 4, 1, 3, 0),   -- ConvNext
    (160, 160, 4, 1, 0, 0),   -- FFN
    (160, 160, 4, 1, 3, 3),   -- ExtraDW
    (160, 256, 6, 2, 5, 5),   -- ExtraDW
    (256, 256, 4, 1, 5, 5),   -- ExtraDW
    (256, 256, 4, 1, 0, 3),   -- IB
    (256, 256, 4, 1, 3, 0) ]  -- ConvNext

/-- The verified layout's count: sum of `∏ dims` over `toSpecs`. -/
def vlayerCount (t : Nat × Nat × Nat × Nat × Nat × Nat) : Nat :=
  let (ic, oc, e, s, pre, post) := t
  ((VLayer.uib ic oc e s pre post).toSpecs).foldl (fun acc (d, _) => acc + d.foldl (· * ·) 1) 0

/-- The baseline's count, straight off `Layer.nParams`. -/
def baselineCount (t : Nat × Nat × Nat × Nat × Nat × Nat) : Nat :=
  let (ic, oc, e, s, pre, post) := t
  (Layer.uib ic oc e s pre post).nParams

def main : IO Unit := do
  let mut bad := 0
  let mut totV := 0
  let mut totB := 0
  for t in mnv4UibTable do
    let v := vlayerCount t
    let b := baselineCount t
    totV := totV + v; totB := totB + b
    let (ic, oc, e, _, pre, post) := t
    let fam := if pre > 0 && post > 0 then "ExtraDW"
               else if pre == 0 && post > 0 then "IB     "
               else if pre > 0 && post == 0 then "ConvNext"
               else "FFN    "
    if v != b then
      bad := bad + 1
      IO.println s!"  ✗ {fam} uib {ic}->{oc} e{e} pre{pre} post{post}: VLayer {v} ≠ baseline {b}"
    else
      IO.println s!"  ✓ {fam} uib {ic}->{oc} e{e} pre{pre} post{post}: {v}"
  IO.println s!"\n  UIB-block total: VLayer {totV}  baseline {totB}"
  if bad != 0 || totV != totB then
    IO.eprintln s!"UIB LAYOUT TIE FAILED ({bad} mismatched blocks)"
    IO.Process.exit 1
  IO.println "  ✓ uib layout tie: VLayer.toSpecs == Layer.nParams on all 14 blocks, all 4 families"
