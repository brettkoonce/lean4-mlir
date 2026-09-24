/-! # Little-endian packing for the FFI buffers

Every buffer the runtimes read — shape descriptors, labels, token ids, anchor priors, f32
tensors — is 4-byte little-endian. These are the two writers; the reader is `F32.readLabel`.
Import-free, so the pure-data `ParamLayouts` can use them. -/

/-- Append `v mod 2³²` as 4 little-endian bytes (an int32 / uint32 record). -/
@[inline] def pushU32LE (acc : ByteArray) (v : Nat) : ByteArray :=
  let u : UInt32 := v.toUInt32
  ((acc.push (u &&& 0xff).toUInt8).push ((u >>> 8) &&& 0xff).toUInt8).push
    ((u >>> 16) &&& 0xff).toUInt8 |>.push ((u >>> 24) &&& 0xff).toUInt8

/-- Append `x` as 4 little-endian f32 bytes (narrowing f64 → `Float32`). -/
@[inline] def pushF32LE (acc : ByteArray) (x : Float) : ByteArray :=
  let u : UInt32 := x.toFloat32.toBits
  ((acc.push (u &&& 0xff).toUInt8).push ((u >>> 8) &&& 0xff).toUInt8).push
    ((u >>> 16) &&& 0xff).toUInt8 |>.push ((u >>> 24) &&& 0xff).toUInt8

#guard (pushU32LE .empty 0x04030201).data == #[1, 2, 3, 4]
#guard (pushU32LE .empty (2 ^ 32 + 7)).data == #[7, 0, 0, 0]
#guard (pushF32LE .empty 1.0).data == #[0, 0, 0x80, 0x3f]
