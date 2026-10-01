/-! # Little-endian packing for the FFI buffers

Every buffer the runtimes read — shape descriptors, labels, token ids, anchor priors, f32
tensors — is little-endian. These are the writers and the readers (`F32.readLabel` is
`readU32LE` at record granularity). Import-free, so the pure-data `ParamLayouts` can use them. -/

/-- Append `v mod 2³²` as 4 little-endian bytes (an int32 / uint32 record). -/
@[inline] def pushU32LE (acc : ByteArray) (v : Nat) : ByteArray :=
  let u : UInt32 := v.toUInt32
  ((acc.push (u &&& 0xff).toUInt8).push ((u >>> 8) &&& 0xff).toUInt8).push
    ((u >>> 16) &&& 0xff).toUInt8 |>.push ((u >>> 24) &&& 0xff).toUInt8

/-- Append `v mod 2⁶⁴` as 8 little-endian bytes (a uint64 record). -/
@[inline] def pushU64LE (acc : ByteArray) (v : Nat) : ByteArray :=
  pushU32LE (pushU32LE acc (v % 4294967296)) (v / 4294967296)

/-- Append `x` as 4 little-endian f32 bytes (narrowing f64 → `Float32`). -/
@[inline] def pushF32LE (acc : ByteArray) (x : Float) : ByteArray :=
  let u : UInt32 := x.toFloat32.toBits
  ((acc.push (u &&& 0xff).toUInt8).push ((u >>> 8) &&& 0xff).toUInt8).push
    ((u >>> 16) &&& 0xff).toUInt8 |>.push ((u >>> 24) &&& 0xff).toUInt8

/-- Append `x` as 8 little-endian f64 bytes. -/
@[inline] def pushF64LE (acc : ByteArray) (x : Float) : ByteArray :=
  let u := x.toBits
  pushU32LE (pushU32LE acc (u &&& 0xffffffff).toNat) (u >>> 32).toNat

/-- The little-endian uint32 at BYTE offset `off`. -/
@[inline] def readU32LE (ba : ByteArray) (off : Nat) : Nat :=
  (ba.get! off).toNat ||| ((ba.get! (off + 1)).toNat <<< 8)
    ||| ((ba.get! (off + 2)).toNat <<< 16) ||| ((ba.get! (off + 3)).toNat <<< 24)

/-- The little-endian uint64 at BYTE offset `off`. -/
@[inline] def readU64LE (ba : ByteArray) (off : Nat) : Nat :=
  readU32LE ba off ||| (readU32LE ba (off + 4) <<< 32)

#guard (pushU32LE .empty 0x04030201).data == #[1, 2, 3, 4]
#guard (pushU32LE .empty (2 ^ 32 + 7)).data == #[7, 0, 0, 0]
#guard (pushF32LE .empty 1.0).data == #[0, 0, 0x80, 0x3f]
#guard readU32LE (pushU32LE (pushU32LE .empty 9) 0xdeadbeef) 4 == 0xdeadbeef
#guard readU64LE (pushU64LE .empty (2 ^ 40 + 3)) 0 == 2 ^ 40 + 3
#guard (pushF64LE .empty 1.0).data == #[0, 0, 0, 0, 0, 0, 0xf0, 0x3f]
