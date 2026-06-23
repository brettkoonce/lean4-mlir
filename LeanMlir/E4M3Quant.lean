import LeanMlir.F32Array

/-! # Pure-Lean E4M3 (fp8) fake-quant over raw-f32 `ByteArray`s

The quantizer the **Lean fp8 trainer** (`MainMnistLinearE4M3Verified`) wraps around
the verified linear train-step kernel. It implements the *same* E4M3 (1-4-3, bias 7)
round-to-nearest grid as the numpy oracle (`scripts/mnist_e4m3_demo.py:to_e4m3`):
subnormals on the `e = −6` grid (step `2⁻⁹`), saturating at ±448.

This is the host-side "operand byte preparation" half of the §3b render-tie
(`LeanMlir/Proofs/E4M3FaithfulPoC.lean`): fp8 = fp32 arithmetic on operands
projected onto the E4M3 grid, with fp32 accumulate inside the kernel. No fp8
hardware or fp8 StableHLO type is needed — `q` runs here, in Lean, before the
verified matmul sees the bytes.

Values move through `F32.read` (extern f32→Float) and are re-encoded f32→`Float32`
→ 4 little-endian bytes; the master weights stay fp32 across the gradient update. -/

namespace F32E4M3

/-- E4M3 largest finite magnitude: `S.1111.110 = 2⁸·1.75`. -/
@[inline] def e4m3Max : Float := 448.0

/-- Round one value to the E4M3 grid (round-to-nearest, subnormals on the
    `e = −6` grid, saturating). Mirrors `to_e4m3` in the numpy oracle. -/
def roundE4M3 (x : Float) : Float :=
  let s : Float := if x < 0.0 then -1.0 else 1.0
  let a := min x.abs e4m3Max
  if a == 0.0 then 0.0
  else
    -- binade exponent, clamped to the normal range (subnormals share e = −6)
    let e := max (-6.0) (min 8.0 (Float.floor (Float.log2 a)))
    let step := Float.exp2 (e - 3.0)          -- 3-bit mantissa LSB
    let q := min (Float.round (a / step) * step) e4m3Max
    s * q

/-- Append `x` as 4 little-endian f32 bytes (narrowing f64 → `Float32`). -/
@[inline] def pushF32LE (acc : ByteArray) (x : Float) : ByteArray :=
  let u : UInt32 := x.toFloat32.toBits
  ((acc.push (u &&& 0xff).toUInt8).push ((u >>> 8) &&& 0xff).toUInt8).push
    ((u >>> 16) &&& 0xff).toUInt8 |>.push ((u >>> 24) &&& 0xff).toUInt8

/-- **Per-tensor E4M3 quant** (one scale `s = max|·|/448`). Round-trips through
    the grid: returns `s · q(vᵢ/s)` as f32 bytes (the dequantized operand). -/
def quantPerTensor (ba : ByteArray) : ByteArray := Id.run do
  let n := F32.size ba
  let mut m : Float := 0.0
  for i in [0:n] do
    let v := (F32.read ba i.toUSize).abs
    if v > m then m := v
  let s := if m > 0.0 then m / e4m3Max else 1.0
  let mut out := ByteArray.emptyWithCapacity (n * 4)
  for i in [0:n] do
    let v := F32.read ba i.toUSize
    out := pushF32LE out (roundE4M3 (v / s) * s)
  return out

/-- **Per-output-column E4M3 quant** for a row-major `[d0 × d1]` matrix (the
    "block scale" `sWⱼ` of §3b: each output column scaled independently). -/
def quantPerColumn (ba : ByteArray) (d0 d1 : Nat) : ByteArray := Id.run do
  let mut scales : Array Float := #[]
  for j in [0:d1] do
    let mut m : Float := 0.0
    for i in [0:d0] do
      let v := (F32.read ba (i * d1 + j).toUSize).abs
      if v > m then m := v
    scales := scales.push (if m > 0.0 then m / e4m3Max else 1.0)
  let mut out := ByteArray.emptyWithCapacity (d0 * d1 * 4)
  for i in [0:d0] do
    for j in [0:d1] do
      let s := scales[j]!
      let v := F32.read ba (i * d1 + j).toUSize
      out := pushF32LE out (roundE4M3 (v / s) * s)
  return out

/-- **fp32-master gradient-delta update.** The fused kernel returns
    `wOut = wq − lr·∇` (it updated the *quantized* operand). Applying the same
    gradient to the fp32 master is `master + (wOut − wq) = master − lr·∇`. -/
def addDelta (master wOut wq : ByteArray) : ByteArray := Id.run do
  let n := F32.size master
  let mut out := ByteArray.emptyWithCapacity (n * 4)
  for i in [0:n] do
    let m := F32.read master i.toUSize
    let o := F32.read wOut i.toUSize
    let q := F32.read wq i.toUSize
    out := pushF32LE out (m + (o - q))
  return out

/-- **Per-leading-block E4M3 quant**: `nBlocks` contiguous blocks of `blockSize`
    elements, each with its own scale. A conv kernel `[oc, ic, k, k]` (row-major)
    is `oc` blocks of `ic·k·k`, so this is per-output-channel quant. -/
def quantPerLeadingBlock (ba : ByteArray) (nBlocks blockSize : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (nBlocks * blockSize * 4)
  for o in [0:nBlocks] do
    let base := o * blockSize
    let mut m : Float := 0.0
    for t in [0:blockSize] do
      let v := (F32.read ba (base + t).toUSize).abs
      if v > m then m := v
    let s := if m > 0.0 then m / e4m3Max else 1.0
    for t in [0:blockSize] do
      let v := F32.read ba (base + t).toUSize
      out := pushF32LE out (roundE4M3 (v / s) * s)
  return out

/-- **Quantize a packed param buffer slot-by-slot**, per the `(dims, initKind)`
    layout (`VerifiedNetSpec.toSpecs`, func-arg order). `initKind == 0` weights:
    dense `[ic, oc]` → `quantPerColumn` (per output column), conv `[oc, ic, k, k]`
    → `quantPerLeadingBlock` (per output channel). Biases / γ / β (`initKind`
    1 or 2) are **copied** (kept fp32). This is the packed analogue of the
    per-tensor/per-column quant the linear trainer does — the *weight* operand
    prep for the verified MLP/CNN train step. -/
def quantPackedParams (params : ByteArray) (specs : Array (Array Nat × Nat)) : ByteArray := Id.run do
  let mut parts : Array ByteArray := #[]
  let mut off := 0                                  -- element offset into the packed buffer
  for spec in specs do
    let dims := spec.1
    let n := dims.foldl (· * ·) 1
    let slot := params.extract (off * 4) ((off + n) * 4)
    let q :=
      if spec.2 == 0 && dims.size == 2 then
        quantPerColumn slot dims[0]! dims[1]!                       -- dense [ic, oc]
      else if spec.2 == 0 && dims.size == 4 then
        quantPerLeadingBlock slot dims[0]! (dims[1]! * dims[2]! * dims[3]!)  -- conv [oc, ic, k, k]
      else
        slot                                                        -- bias / γ / β: fp32
    parts := parts.push q
    off := off + n
  return F32.concat parts

end F32E4M3
