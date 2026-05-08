import LeanMlir

/-! Synthetic CAM kernel test.

Builds a tiny `[B=1, C=2, H=3, W=3]` activation and a 2-class dense
weight matrix, hand-computes the expected ReLU-CAM heatmap, and checks
the FFI-side `F32.camCompute` matches within ε.

Also smoke-tests bilinear upsample and viridis colormap.

Pass criterion: max absolute error < 1e-5 across all heatmap pixels. -/

private def expectedHeatmap : List (List Float) :=
  -- Activation A[c=0..1, i=0..2, j=0..2] (laid out below).
  -- W = [[1.0, -0.5], [0.5, -1.0]] row-major; tgt = 0.
  -- raw[i,j] = 1.0 * A0[i,j] + 0.5 * A1[i,j]
  --
  -- A0 =
  --   1 2 3
  --   4 5 6
  --   7 8 9
  -- A1 =
  --   9 8 7
  --   6 5 4
  --   3 2 1
  -- raw =
  --   1.0*A0 + 0.5*A1 =
  --     5.5  6.0  6.5
  --     7.0  7.5  8.0
  --     8.5  9.0  9.5
  -- All non-negative, max = 9.5 ⇒ heat = raw / 9.5.
  [[5.5, 6.0, 6.5],
   [7.0, 7.5, 8.0],
   [8.5, 9.0, 9.5]]

private def packF32 (vs : List Float) : IO ByteArray := do
  let mut ba ← F32.const vs.length.toUSize 0.0
  -- Write each float by allocating a tiny one-cell ByteArray and copying.
  -- `F32.const` already memcpys, so build via concat of singletons.
  let mut parts : Array ByteArray := #[]
  for v in vs do
    parts := parts.push (← F32.const 1 v)
  return F32.concat parts.toList.toArray

private def maxAbsErr (got : ByteArray) (want : List (List Float))
    (H W : Nat) : Float := Id.run do
  let mut mx : Float := 0.0
  for i in [:H] do
    for j in [:W] do
      let g := F32.read got (i * W + j).toUSize
      let w := (want[i]!)[j]!
      let d := if g - w < 0.0 then w - g else g - w
      if d > mx then mx := d
  return mx

def main : IO Unit := do
  IO.println "=== CAM kernel synthetic test ==="

  -- Activation: [B=1, C=2, H=3, W=3] flat NCHW.
  let act ← packF32 [
    -- channel 0
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    -- channel 1
    9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0
  ]
  -- Dense weights: [C=2, NC=2] row-major.
  --   W[0, 0]=1.0  W[0, 1]=-0.5
  --   W[1, 0]=0.5  W[1, 1]=-1.0
  let dw ← packF32 [1.0, -0.5, 0.5, -1.0]

  -- camCompute for tgt=0 expects max-normalized heatmap matching the table above.
  let heat ← F32.camCompute dw act 0 2 3 3 2 0
  IO.println s!"heatmap size = {F32.size heat} (expect {3*3})"
  for i in [:3] do
    let row := (List.range 3).map fun j =>
      F32.read heat (i * 3 + j).toUSize
    IO.println s!"  row {i}: {row}"

  -- Hand-computed expected (after dividing by 9.5).
  let expected := expectedHeatmap.map fun row => row.map (· / 9.5)
  let err := maxAbsErr heat expected 3 3
  IO.println s!"max abs error vs hand-computed: {err}"
  if err > 1.0e-5 then
    IO.println "FAIL: CAM kernel output diverges from analytic expectation"
    IO.Process.exit 1

  -- Negative-class sanity: tgt=1 weights are all negative ⇒ raw is all-negative
  -- ⇒ ReLU clamps to zero ⇒ max-norm leaves the buffer at exactly zero.
  let heatNeg ← F32.camCompute dw act 0 2 3 3 2 1
  let mut allZero := true
  for i in [:9] do
    if F32.read heatNeg i.toUSize != 0.0 then allZero := false
  if !allZero then
    IO.println "FAIL: tgt=1 should ReLU to all-zero but didn't"
    IO.Process.exit 1
  IO.println "  tgt=1 (all-negative) -> all-zero heatmap as expected"

  -- Bilinear upsample sanity: [3, 3] -> [9, 9] should reproduce the corners.
  let up ← F32.bilinearUpsample2D heat 3 3 9 9
  let upTL := F32.read up 0
  let upTR := F32.read up 8
  let upBL := F32.read up (8 * 9)
  let upBR := F32.read up (8 * 9 + 8)
  let expTL := (expectedHeatmap[0]!)[0]! / 9.5
  let expTR := (expectedHeatmap[0]!)[2]! / 9.5
  let expBL := (expectedHeatmap[2]!)[0]! / 9.5
  let expBR := (expectedHeatmap[2]!)[2]! / 9.5
  IO.println s!"upsample corners: TL={upTL}/{expTL} TR={upTR}/{expTR}"
  IO.println s!"                  BL={upBL}/{expBL} BR={upBR}/{expBR}"
  let fmax (a b : Float) : Float := if a > b then a else b
  let cornerErr := List.foldl (fun acc x => fmax acc x) 0.0
    [Float.abs (upTL - expTL), Float.abs (upTR - expTR),
     Float.abs (upBL - expBL), Float.abs (upBR - expBR)]
  if cornerErr > 1.0e-5 then
    IO.println s!"FAIL: bilinear upsample corner error {cornerErr} > 1e-5"
    IO.Process.exit 1

  -- Viridis sanity: 0.0 → dark purple, 1.0 → yellow.
  let (r0, g0, b0) := Cam.viridis 0.0
  let (r1, g1, b1) := Cam.viridis 1.0
  IO.println s!"viridis(0.0) = ({r0}, {g0}, {b0})  (expect ~(68, 1, 84))"
  IO.println s!"viridis(1.0) = ({r1}, {g1}, {b1})  (expect ~(253, 231, 36))"
  if r0 != 68 || g0 != 1 || b0 != 84 then
    IO.println "FAIL: viridis(0.0) does not match palette"
    IO.Process.exit 1
  if r1.toNat < 240 || g1.toNat < 220 || b1.toNat > 60 then
    IO.println "FAIL: viridis(1.0) does not match palette"
    IO.Process.exit 1

  IO.println "=== PASS ==="
