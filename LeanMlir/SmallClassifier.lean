import LeanMlir.SpecHelpers
import LeanMlir.IreeRuntime

/-! # The small-classifier demos' shared kit

The Chapter 10 image-classification demos (`MainAraslSigns`, `MainPlantLeaf`, `MainGwDetect`,
`MainRsBands`) each train one `NetSpec` on an int32-labelled f32 set with the same host loop:
a seeded Fisher–Yates order per epoch, a gathered batch per step, and a scoring pass over a
held-out part. This is that loop's pieces, one copy. -/

namespace SmallClassifier

/-- `x` rounded to `d` decimals, printed with `toString` and cut at 10 characters. -/
def fmt (x : Float) (d : Nat) : String :=
  let m := (10.0 : Float) ^ d.toFloat
  let r := (x * m).round / m
  let s := toString r
  if s.length > 10 then (s.toRawSubstring.take 10).toString else s

/-- xorshift64 step. -/
@[inline] def xs (s : UInt64) : UInt64 :=
  let s := s ^^^ (s <<< 13)
  let s := s ^^^ (s >>> 7)
  s ^^^ (s <<< 17)

/-- Fisher–Yates permutation of `0..n-1` from a seed. -/
def permutation (n : Nat) (seed : UInt64) : Array Nat := Id.run do
  let mut a : Array Nat := Array.range n
  let mut s := if seed == 0 then 0x9E3779B97F4A7C15 else seed
  for i in [1:n] do
    let j := n - i
    s := xs s
    let k := (s % (j + 1).toUInt64).toNat
    let tmp := a[j]!
    a := a.set! j a[k]!
    a := a.set! k tmp
  return a

/-- Gather a batch of `B` images (`nPix` floats each) by index from a flat f32 set, and their
    labels. -/
def gather (img lbl : ByteArray) (idx : Array Nat) (start B nPix : Nat) :
    ByteArray × ByteArray := Id.run do
  let mut x := ByteArray.emptyWithCapacity (B * nPix * 4)
  let mut y := ByteArray.emptyWithCapacity (B * 4)
  for i in [:B] do
    let k := idx[start + i]!
    x := x ++ F32.sliceImages img k 1 nPix
    y := y ++ F32.sliceLabels lbl k 1
  return (x, y)

/-- Run the eval graph over `n` examples at batch `evalB` and return the logits as f32
    `[n, classes]` plus the argmax accuracy (%) against the int32 labels `lbl`. `batchAt bi` is
    the input of batch `bi`, padded to `evalB` rows; only the real rows are scored. -/
def scoreBatches (sess : LowererSession) (spec : NetSpec) (evalParams evalShapes xSh : ByteArray)
    (lbl : ByteArray) (n evalB : Nat) (batchAt : Nat → IO ByteArray) :
    IO (ByteArray × Float) := do
  let nC := spec.numClasses
  let mut logits := ByteArray.emptyWithCapacity (n * nC * 4)
  let mut correct : Nat := 0
  let nb := (n + evalB - 1) / evalB
  for bi in [:nb] do
    let xba ← batchAt bi
    let out ← LowererSession.forwardF32 sess spec.evalFnName evalParams evalShapes xba xSh
                evalB.toUSize nC.toUSize
    let avail := min evalB (n - bi * evalB)
    logits := logits ++ out.extract 0 (avail * nC * 4)
    for i in [:avail] do
      let pred := F32.argmaxN out (i * nC).toUSize nC.toUSize
      if pred.toNat == F32.readLabel lbl (bi * evalB + i) then correct := correct + 1
  return (logits, correct.toFloat / n.toFloat * 100.0)

/-- `scoreBatches` over a flat f32 set of `n` images of `nPix` floats, the tail batch
    zero-padded. -/
def scoreSet (sess : LowererSession) (spec : NetSpec) (evalParams evalShapes xSh : ByteArray)
    (img lbl : ByteArray) (n evalB nPix : Nat) : IO (ByteArray × Float) :=
  scoreBatches sess spec evalParams evalShapes xSh lbl n evalB fun bi =>
    pure (F32.sliceImagesPad img (bi * evalB) evalB nPix n)

end SmallClassifier
