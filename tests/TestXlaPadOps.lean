import LeanMlir

/-! # Known-answer guard for the XLA-`SAME` strided ops (`planning/mnv4_verified.md` §3e)

`convStridedXla` and `depthwiseStridedXla` are the asymmetric-pad peers of `convStrided` and
`depthwiseStrided`. **They have identical types, identical output shapes, identical op counts and
identical feature-group widths** — the entire difference is four numbers in the emitted `pad`. So
nothing structural can distinguish a correct render from one that picked the wrong token, and
`#guard`s on shapes or arity are worthless here by construction (`planning/mnv4_verified.md` §3,
the same invisibility class as R50's stride-on-the-3×3).

What this emits is therefore deliberately *small*: two one-op modules, at the two kernel sizes the
affected nets actually use, so `scripts/xla_pad_op_check.py` can run them through IREE and compare
against `jax.lax.conv_general_dilated(…, padding='SAME')` directly. A whole-net tie can only say
"something is off somewhere"; this says which op.

⚠ Both are rendered at EVEN spatial inputs, which is the only case the tokens' types admit
(`c*(2*h)*(2*w)`) and the only case where XLA `SAME` is asymmetric. At an odd input `SAME` is
symmetric and the *existing* `convStrided`/`depthwiseStrided` are already correct — that boundary
is the one thing a reader of these ops most needs to know, so it is asserted in the checker rather
than left as a comment.

Run: `lake env lean tests/TestXlaPadOps.lean` (writes the modules), then
     `.venv/bin/python3 scripts/xla_pad_op_check.py`. -/

open Proofs Proofs.StableHLO

-- ⚠ File-level, not `set_option … in` per def: `renderModule`'s index argument forces the
-- `B*(c*(2*h)*(2*w))` arithmetic through `whnf`, which blows the 200k default even at these toy
-- sizes. Same kernel-blowup shape the batched EfficientNet render hit.
set_option maxHeartbeats 2000000

namespace Proofs.XlaPadProbe

/-- Wrap one batched op as a module. ⚠ NOT `renderModule`: that takes `g : SHlo retLen` and returns
    `tensor<B × retLen>`, which is the per-example convention — a `batchOp` graph's index is already
    `B * n`, so `renderModule` would both fail to unify and declare the wrong return type. Same
    `N` (batch) vs `n` (per-example width) distinction the `BatchableOp` docs call out. -/
private def wrap (name argSig : String) (B outN : Nat) (body res : String) : String :=
  "module @m {\n" ++
  s!"  func.func @{name}({argSig}) -> {ty [B, outN]} " ++ "{\n" ++
  body ++ s!"    return {res} : {ty [B, outN]}\n" ++ "  }\n}\n"

/-- `@conv_xla` — one `convStridedXla`, ic 3 → oc 8, k=3, 16×16 → 8×8, batch 2. -/
def convXlaModule : String :=
  let B := 2
  let zx : Vec (B*(3*(2*8)*(2*8))) := fun _ => 0
  let zK : Kernel4 8 3 3 3 := fun _ _ _ _ => 0
  let z8 : Vec 8 := fun _ => 0
  let (body, res) := (pretty B (.batchOp (N := B)
    (.convStridedXla (ic := 3) (oc := 8) (h := 8) (w := 8) (kH := 3) (kW := 3) "%W" "%b" zK z8)
    (.operand "%x" zx))).run' 0
  wrap "conv_xla" s!"%x: {ty [B, 3*16*16]}, %W: {ty [8,3,3,3]}, %b: {ty [8]}" B (8*8*8) body res

/-- `@dw_xla_k3` — one `depthwiseStridedXla`, c = 6, k=3, 16×16 → 8×8, batch 2. -/
def dwXlaK3Module : String :=
  let B := 2
  let zx : Vec (B*(6*(2*8)*(2*8))) := fun _ => 0
  let zK : DepthwiseKernel 6 3 3 := fun _ _ _ => 0
  let z6 : Vec 6 := fun _ => 0
  let (body, res) := (pretty B (.batchOp (N := B)
    (.depthwiseStridedXla (c := 6) (h := 8) (w := 8) (kH := 3) (kW := 3) "%W" "%b" zK z6)
    (.operand "%x" zx))).run' 0
  wrap "dw_xla_k3" s!"%x: {ty [B, 6*16*16]}, %W: {ty [6,1,3,3]}, %b: {ty [6]}" B (6*8*8) body res

/-- `@dw_xla_k5` — the k=5 case, where XLA `SAME` pads `(1,2)`: total 3, which is NOT `2·((k-1)/2)`.
    EfficientNet renders depthwises at k=5, so this is the case that would catch a `pad_low`
    formula that happens to be right only at k=3. -/
def dwXlaK5Module : String :=
  let B := 2
  let zx : Vec (B*(6*(2*8)*(2*8))) := fun _ => 0
  let zK : DepthwiseKernel 6 5 5 := fun _ _ _ => 0
  let z6 : Vec 6 := fun _ => 0
  let (body, res) := (pretty B (.batchOp (N := B)
    (.depthwiseStridedXla (c := 6) (h := 8) (w := 8) (kH := 5) (kW := 5) "%W" "%b" zK z6)
    (.operand "%x" zx))).run' 0
  wrap "dw_xla_k5" s!"%x: {ty [B, 6*16*16]}, %W: {ty [6,1,5,5]}, %b: {ty [6]}" B (6*8*8) body res

/-- The SYMMETRIC control, same shape as `@conv_xla`. If the checker finds this one ALSO matches
    XLA `SAME`, the test is vacuous — the two conventions coincide at that shape and nothing is
    being measured. It exists to make that failure mode loud instead of silent. -/
def convSymModule : String :=
  let B := 2
  let zx : Vec (B*(3*(2*8)*(2*8))) := fun _ => 0
  let zK : Kernel4 8 3 3 3 := fun _ _ _ _ => 0
  let z8 : Vec 8 := fun _ => 0
  let (body, res) := (pretty B (.batchOp (N := B)
    (.convStrided (ic := 3) (oc := 8) (h := 8) (w := 8) (kH := 3) (kW := 3) "%W" "%b" zK z8)
    (.operand "%x" zx))).run' 0
  wrap "conv_sym" s!"%x: {ty [B, 3*16*16]}, %W: {ty [8,3,3,3]}, %b: {ty [8]}" B (8*8*8) body res

-- ══════════════════════════════════════════════════════════════════════════
-- § BACKWARD probes — the part that can be wrong SILENTLY
-- ══════════════════════════════════════════════════════════════════════════
-- The forward probes above compare against a reference that exists (`jax.lax` with
-- `padding='SAME'`). The backward emits have no such oracle to copy from: their padding was
-- DERIVED (`pad_low = p-1`, `pad_high = p+1`, because the Xla forward reads one position later).
-- A one-off error in either direction still type-checks, still produces the right shape, and
-- still descends — it just computes the gradient of a different net. So these are checked against
-- `jax.vjp` of the actual Xla-padded forward, which is the only thing that can catch it.

/-- `@dw_xla_back` — `depthwiseStridedXlaBackBatched`: the input cotangent, at `2h×2w`. -/
def dwXlaBackModule : String :=
  let B := 2
  let zdy : Vec (B*(6*8*8)) := fun _ => 0
  let zK : DepthwiseKernel 6 3 3 := fun _ _ _ => 0
  let z6 : Vec 6 := fun _ => 0
  let (body, res) := (pretty B (.depthwiseStridedXlaBackBatched (N := B) (c := 6) (h := 8) (w := 8)
    "%W" zK z6 (.operand "%dy" zdy))).run' 0
  "module @m {\n" ++
  s!"  func.func @dw_xla_back(%dy: {ty [B, 6*8*8]}, %W: {ty [6,1,3,3]}) -> {ty [B, 6*16*16]} " ++ "{\n" ++
  body ++ s!"    return {res} : {ty [B, 6*16*16]}\n" ++ "  }\n}\n"

/-- `@conv_xla_wgrad` — `convStridedXlaWeightGradB`: the kernel gradient, summed over the batch. -/
def convXlaWGradModule : String :=
  let B := 2
  let zdy : Vec (B*(8*8*8)) := fun _ => 0
  let zx : Vec (B*(3*16*16)) := fun _ => 0
  let zK : Kernel4 8 3 3 3 := fun _ _ _ _ => 0
  let z8 : Vec 8 := fun _ => 0
  let (body, res) := (pretty B (.convStridedXlaWeightGradB (N := B) (ic := 3) (oc := 8)
    (h := 8) (w := 8) "%x" z8 zx zK (.operand "%dy" zdy))).run' 0
  "module @m {\n" ++
  s!"  func.func @conv_xla_wgrad(%dy: {ty [B, 8*8*8]}, %x: {ty [B, 3*16*16]}) -> {ty [8,3,3,3]} " ++ "{\n" ++
  body ++ s!"    return {res} : {ty [8,3,3,3]}\n" ++ "  }\n}\n"

/-- `@dw_xla_wgrad` — `depthwiseStridedXlaWeightGradB`: the depthwise kernel gradient. -/
def dwXlaWGradModule : String :=
  let B := 2
  let zdy : Vec (B*(6*8*8)) := fun _ => 0
  let zx : Vec (B*(6*16*16)) := fun _ => 0
  let zK : DepthwiseKernel 6 3 3 := fun _ _ _ => 0
  let z6 : Vec 6 := fun _ => 0
  let (body, res) := (pretty B (.depthwiseStridedXlaWeightGradB (N := B) (c := 6)
    (h := 8) (w := 8) "%x" z6 zx zK (.operand "%dy" zdy))).run' 0
  "module @m {\n" ++
  s!"  func.func @dw_xla_wgrad(%dy: {ty [B, 6*8*8]}, %x: {ty [B, 6*16*16]}) -> {ty [6,1,3,3]} " ++ "{\n" ++
  body ++ s!"    return {res} : {ty [6,1,3,3]}\n" ++ "  }\n}\n"

end Proofs.XlaPadProbe

-- The emitted `pad` is the whole content of these ops, so pin it here too: a `lake env lean` that
-- succeeds while the padding is symmetric would send the checker looking in the wrong place.
#guard (Proofs.XlaPadProbe.convXlaModule.splitOn "pad = [[0, 1], [0, 1]]").length == 2
#guard (Proofs.XlaPadProbe.dwXlaK3Module.splitOn "pad = [[0, 1], [0, 1]]").length == 2
#guard (Proofs.XlaPadProbe.dwXlaK5Module.splitOn "pad = [[1, 2], [1, 2]]").length == 2
#guard (Proofs.XlaPadProbe.convSymModule.splitOn "pad = [[1, 1], [1, 1]]").length == 2
-- …and that the depthwise ones really are grouped (a `feature_group_count = 1` here would mean the
-- emit fell through to the dense-conv case and the test would be measuring the wrong operator).
#guard (Proofs.XlaPadProbe.dwXlaK3Module.splitOn "feature_group_count = 6 : i64").length == 2
#guard (Proofs.XlaPadProbe.dwXlaK5Module.splitOn "feature_group_count = 6 : i64").length == 2

#eval IO.FS.writeFile ".lake/build/xlapad_conv.mlir"   Proofs.XlaPadProbe.convXlaModule
#eval IO.FS.writeFile ".lake/build/xlapad_dw_k3.mlir"  Proofs.XlaPadProbe.dwXlaK3Module
#eval IO.FS.writeFile ".lake/build/xlapad_dw_k5.mlir"  Proofs.XlaPadProbe.dwXlaK5Module
#eval IO.FS.writeFile ".lake/build/xlapad_conv_sym.mlir" Proofs.XlaPadProbe.convSymModule

-- The backward emits all shift to `[p-1, p+1]`; at k=3 (p=1) that is `[0, 2]`. Pin it — an
-- accidental `[1, 1]` here is the silent wrong-gradient this whole exercise exists to prevent.
-- ⚠ `[[2, 0]]` — the input-VJP shifts the OTHER way from the weight grads below, because its
-- kernel is reversed. This asymmetry is the single easiest thing to get wrong here.
#guard (Proofs.XlaPadProbe.dwXlaBackModule.splitOn "pad = [[2, 0], [2, 0]]").length == 2
#guard (Proofs.XlaPadProbe.convXlaWGradModule.splitOn "pad = [[0, 2], [0, 2]]").length == 2
#guard (Proofs.XlaPadProbe.dwXlaWGradModule.splitOn "pad = [[0, 2], [0, 2]]").length == 2

#eval IO.FS.writeFile ".lake/build/xlapad_dw_back.mlir"    Proofs.XlaPadProbe.dwXlaBackModule
#eval IO.FS.writeFile ".lake/build/xlapad_conv_wgrad.mlir" Proofs.XlaPadProbe.convXlaWGradModule
#eval IO.FS.writeFile ".lake/build/xlapad_dw_wgrad.mlir"   Proofs.XlaPadProbe.dwXlaWGradModule
#eval IO.println "✓ TestXlaPadOps: 7 probe modules written to .lake/build/xlapad_*.mlir"
