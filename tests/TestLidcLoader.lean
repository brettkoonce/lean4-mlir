import LeanMlir.F32Array

/-!
# Does `F32.loadCtPair` hand the trainer the pairs `preprocess_lidc.py` wrote?

`test-lidc-loader <file.bin> [record …]` loads a CT pair file through the C loader in each
pairing — supervised at every realisation, and Noise2Noise when the file has two — plus the
masks, and prints, for each requested record (default: first, middle, last), integer statistics
of what came back:

    count <n>  N <realisations>
    rec <k> pair <p> r <r>  in <Σ input·4096>  tgt <Σ target·4096>  ssd <Σ (input − target)²·4096²>
    rec <k> mask <Σ mask> nz <pixels with a reader>

Every pixel is a multiple of 1/4096 below 1, so each statistic is an integer and exact in f64:
`scripts/probes/lidc_loader_check.py` decodes the same records with numpy and requires equality,
not a tolerance. It also checks the declared record sizes (`count · S²` floats each side). The
realisation count comes from the file size (`F32.ctRealisations`), as it does in the trainer,
and a read at the wrong count must be refused.
-/

open F32

private def S : Nat := 512

private def stats (inp tgt : ByteArray) (k : Nat) : Nat × Nat × Nat := Id.run do
  let base := k * S * S
  let mut a := 0.0
  let mut b := 0.0
  let mut d := 0.0
  for j in [0:S*S] do
    let x := F32.read inp (base + j).toUSize * 4096.0
    let y := F32.read tgt (base + j).toUSize * 4096.0
    a := a + x
    b := b + y
    d := d + (x - y) * (x - y)
  return (a.round.toUInt64.toNat, b.round.toUInt64.toNat, d.round.toUInt64.toNat)

def main (args : List String) : IO UInt32 := do
  let some path := args.head? | IO.eprintln "usage: test-lidc-loader <file.bin> [record …]"; return 2
  let mut ok := true
  let nReal ← F32.ctRealisations path S
  let arms : List (USize × USize) :=
    ((List.range nReal).map fun r => ((0 : USize), r.toUSize)) ++ (if nReal ≥ 2 then [(1, 0)] else [])
  let mut recs : List Nat := args.tail.filterMap String.toNat?
  for (p, r) in arms do
    let (inp, tgt, n) ← F32.loadCtPair path S.toUSize nReal.toUSize p r
    if inp.size != n * S * S * 4 || tgt.size != n * S * S * 4 then
      IO.println s!"FAIL sizes pair {p} r {r}: n={n} input {inp.size} B target {tgt.size} B"
      ok := false
    if recs.isEmpty then recs := [0, n / 2, n - 1]
    IO.println s!"count {n} N {nReal}"
    for k in recs do
      let (a, b, d) := stats inp tgt k
      IO.println s!"rec {k} pair {p} r {r} in {a} tgt {b} ssd {d}"
  let m ← F32.loadCtMask path S.toUSize nReal.toUSize
  for k in recs do
    let mut sum := 0
    let mut nz := 0
    for j in [0:S*S] do
      let v := (m.get! (k * S * S + j)).toNat
      sum := sum + v
      if v > 0 then nz := nz + 1
    IO.println s!"rec {k} mask {sum} nz {nz}"
  -- A wrong realisation count must be refused, not misparsed.
  try
    let _ ← F32.loadCtPair path S.toUSize (nReal + 1).toUSize
    IO.println s!"FAIL a {nReal + 1}-realisation read of a {nReal}-realisation file was accepted"
    ok := false
  catch e => IO.println s!"refused N={nReal + 1}: {e}"
  return if ok then 0 else 1
