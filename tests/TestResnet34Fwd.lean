import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # B9a — full ResNet-34 forward renderer + `iree-compile` validation

Programmatic StableHLO for a real ResNet-34 forward (CIFAR 3×32×32 input):
strided 3×3 stem (3→64, 32→16) → 2×2 maxpool (16→8) → 4 stages of basic blocks
`[3,4,6,3]` at channels `64/128/256/512` (stages 2–4 open with a strided downsample
block, 8→4→2→1) → global-average-pool → dense(512→10). Every fragment is the same
StableHLO the VERIFIED per-op emitters produce (conv, stride-2 conv, per-channel BN
= bnPerChannelF/Back, relu, maxpool, GAP, dense); this de-risks the whole 34-layer
forward on `iree-compile` before wiring the train step.

Naming: helper `o` is a bare prefix; the helper's result SSA is `%{o}`, its
intermediates `%{o}…`. All value inputs carry a leading `%`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestResnet34Fwd.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 2
private def EPS : String := "1.0e-5"

-- ── 4-D fragment helpers ([B,C,H,W] throughout; structured names, no global counter) ──

/-- 3×3 SAME conv (stride `s`, pad 1) + bias broadcast. Result SSA = `%{o}`. -/
private def conv (o x w bnm : String) (oc ic Hin Win Hout Wout s : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hin,Win]}, {ty [oc,ic,3,3]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hout,Wout]}\n"

/-- Per-channel BatchNorm forward (4-D), reduce μ/var over spatial `[2,3]`, rank-1
    γ/β `dims=[1]`. The bnPerChannelF op pattern, kept 4-D. Result SSA = `%{o}`. -/
private def bnPC (o x g bt : String) (oc Hh Ww m : Nat) : String :=
  s!"    %{o}nf = stablehlo.constant dense<{m}.0> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ep = stablehlo.constant dense<{EPS}> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}smr = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [BS,oc]}\n" ++
  s!"    %{o}sm = stablehlo.broadcast_in_dim %{o}smr, dims = [0, 1] : ({ty [BS,oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}mu = stablehlo.divide %{o}sm, %{o}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xc = stablehlo.subtract {x}, %{o}mu : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sq = stablehlo.multiply %{o}xc, %{o}xc : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vsr = stablehlo.reduce(%{o}sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [BS,oc]}\n" ++
  s!"    %{o}vs = stablehlo.broadcast_in_dim %{o}vsr, dims = [0, 1] : ({ty [BS,oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vr = stablehlo.divide %{o}vs, %{o}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ve = stablehlo.add %{o}vr, %{o}ep : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}istd = stablehlo.rsqrt %{o}ve : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xh = stablehlo.multiply %{o}xc, %{o}istd : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}btb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gx = stablehlo.multiply %{o}xh, %{o}gb : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}gx, %{o}btb : {ty [BS,oc,Hh,Ww]}\n"

private def relu (o x : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o}z = stablehlo.constant dense<0.0> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.maximum {x}, %{o}z : {ty [BS,oc,Hh,Ww]}\n"

private def maxpool (o x : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}ni = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
  s!"    %{o} = \"stablehlo.reduce_window\"({x}, %{o}ni) (" ++ "{\n" ++
  "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
  "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
  "        stablehlo.return %pm : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c,Hh/2,Ww/2]}\n"

private def addOp (o a b : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.add {a}, {b} : {ty [BS,oc,Hh,Ww]}\n"

-- ── blocks ──

/-- Identity basic block: `relu(BN(conv(relu(BN(conv x)))) + x)`, channels `c` @ `Hh×Ww`. -/
private def idBlock (p x : String) (c Hh Ww : Nat) : String × String :=
  let m := Hh*Ww
  let code :=
    conv s!"{p}c1" x s!"%{p}W1" s!"%{p}b1" c c Hh Ww Hh Ww 1 ++
    bnPC s!"{p}n1" s!"%{p}c1" s!"%{p}g1" s!"%{p}bt1" c Hh Ww m ++
    relu s!"{p}r1" s!"%{p}n1" c Hh Ww ++
    conv s!"{p}c2" s!"%{p}r1" s!"%{p}W2" s!"%{p}b2" c c Hh Ww Hh Ww 1 ++
    bnPC s!"{p}n2" s!"%{p}c2" s!"%{p}g2" s!"%{p}bt2" c Hh Ww m ++
    addOp s!"{p}a" s!"%{p}n2" x c Hh Ww ++
    relu s!"{p}o" s!"%{p}a" c Hh Ww
  (code, s!"%{p}o")

/-- Strided downsample block `c→oc`, `2Hh×2Ww → Hh×Ww`:
    `relu(BN(conv₃ₓ₃(relu(BN(conv₂ₛ x)))) + BN(proj₂ₛ x))`. -/
private def downBlock (p x : String) (c oc Hh Ww : Nat) : String × String :=
  let m := Hh*Ww; let Hin := 2*Hh; let Win := 2*Ww
  let code :=
    conv s!"{p}c1" x s!"%{p}W1" s!"%{p}b1" oc c Hin Win Hh Ww 2 ++
    bnPC s!"{p}n1" s!"%{p}c1" s!"%{p}g1" s!"%{p}bt1" oc Hh Ww m ++
    relu s!"{p}r1" s!"%{p}n1" oc Hh Ww ++
    conv s!"{p}c2" s!"%{p}r1" s!"%{p}W2" s!"%{p}b2" oc oc Hh Ww Hh Ww 1 ++
    bnPC s!"{p}n2" s!"%{p}c2" s!"%{p}g2" s!"%{p}bt2" oc Hh Ww m ++
    conv s!"{p}cp" x s!"%{p}Wp" s!"%{p}bp" oc c Hin Win Hh Ww 2 ++
    bnPC s!"{p}np" s!"%{p}cp" s!"%{p}gp" s!"%{p}btp" oc Hh Ww m ++
    addOp s!"{p}a" s!"%{p}n2" s!"%{p}np" oc Hh Ww ++
    relu s!"{p}o" s!"%{p}a" oc Hh Ww
  (code, s!"%{p}o")

/-- `n` identity blocks chained, names `{base}b0..b{n-1}`. -/
private def idChain (base x : String) (n c Hh Ww : Nat) : String × String := Id.run do
  let mut code := ""; let mut cur := x
  for i in [:n] do
    let (c2, out) := idBlock s!"{base}b{i}" cur c Hh Ww
    code := code ++ c2; cur := out
  return (code, cur)

-- ── parameter-signature generators (must mirror the body's names + types) ──

private def bnSig (p : String) (oc : Nat) : List String :=
  [s!"%{p}g: {ty [oc]}", s!"%{p}bt: {ty [oc]}"]
private def idBlockSig (p : String) (c : Nat) : List String :=
  [s!"%{p}W1: {ty [c,c,3,3]}", s!"%{p}b1: {ty [c]}", s!"%{p}g1: {ty [c]}", s!"%{p}bt1: {ty [c]}",
   s!"%{p}W2: {ty [c,c,3,3]}", s!"%{p}b2: {ty [c]}", s!"%{p}g2: {ty [c]}", s!"%{p}bt2: {ty [c]}"]
private def downBlockSig (p : String) (c oc : Nat) : List String :=
  [s!"%{p}W1: {ty [oc,c,3,3]}", s!"%{p}b1: {ty [oc]}", s!"%{p}g1: {ty [oc]}", s!"%{p}bt1: {ty [oc]}",
   s!"%{p}W2: {ty [oc,oc,3,3]}", s!"%{p}b2: {ty [oc]}", s!"%{p}g2: {ty [oc]}", s!"%{p}bt2: {ty [oc]}",
   s!"%{p}Wp: {ty [oc,c,3,3]}", s!"%{p}bp: {ty [oc]}", s!"%{p}gp: {ty [oc]}", s!"%{p}btp: {ty [oc]}"]
private def idChainSig (base : String) (n c : Nat) : List String := Id.run do
  let mut acc := []
  for i in [:n] do acc := acc ++ idBlockSig s!"{base}b{i}" c
  return acc

private def gapDense (o x : String) (c nC Hh Ww : Nat) : String :=
  s!"    %{o}gs = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c]}\n" ++
  s!"    %{o}gnf = stablehlo.constant dense<{Hh*Ww}.0> : {ty [BS,c]}\n" ++
  s!"    %{o}g = stablehlo.divide %{o}gs, %{o}gnf : {ty [BS,c]}\n" ++
  s!"    %{o}dd = stablehlo.dot_general %{o}g, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [c,nC]}) -> {ty [BS,nC]}\n" ++
  s!"    %{o}db = stablehlo.broadcast_in_dim %bd, dims = [1] : ({ty [nC]}) -> {ty [BS,nC]}\n" ++
  s!"    %{o} = stablehlo.add %{o}dd, %{o}db : {ty [BS,nC]}\n"

-- ── whole net ──

private def resnet34Fwd : String := Id.run do
  -- stride-1 3×3 stem (3→64 @32×32) + 2×2 maxpool (32→16); downsamples 16→8→4→2
  let stemCode :=
    conv "stc" "%x" "%sW" "%sb" 64 3 32 32 32 32 1 ++
    bnPC "stn" "%stc" "%sg" "%sbt" 64 32 32 (32*32) ++
    relu "str" "%stn" 64 32 32 ++
    maxpool "stp" "%str" 64 32 32
  let (s1, o1) := idChain "s1" "%stp" 3 64 16 16
  let (d2, o2) := downBlock "d2" o1 64 128 8 8
  let (s2, o2b) := idChain "s2" o2 3 128 8 8
  let (d3, o3) := downBlock "d3" o2b 128 256 4 4
  let (s3, o3b) := idChain "s3" o3 5 256 4 4
  let (d4, o4) := downBlock "d4" o3b 256 512 2 2
  let (s4, o4b) := idChain "s4" o4 2 512 2 2
  let tail := gapDense "out" o4b 512 10 2 2
  let body :=
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    stemCode ++ s1 ++ d2 ++ s2 ++ d3 ++ s3 ++ d4 ++ s4 ++ tail
  let sig : List String :=
    ["%x: " ++ ty [BS,3,32,32]]
    ++ [s!"%sW: {ty [64,3,3,3]}", s!"%sb: {ty [64]}"] ++ bnSig "s" 64
    ++ idChainSig "s1" 3 64
    ++ downBlockSig "d2" 64 128 ++ idChainSig "s2" 3 128
    ++ downBlockSig "d3" 128 256 ++ idChainSig "s3" 5 256
    ++ downBlockSig "d4" 256 512 ++ idChainSig "s4" 2 512
    ++ [s!"%Wd: {ty [512,10]}", s!"%bd: {ty [10]}"]
  let argSig := String.intercalate ", " sig
  return "module @m {\n" ++ s!"  func.func @resnet34_fwd({argSig}) -> {ty [BS,10]} " ++ "{\n" ++
    body ++ s!"    return %out : {ty [BS,10]}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := resnet34Fwd
  IO.println s!"rendered @resnet34_fwd: {mlir.length} chars"
  IO.FS.createDirAll ".lake/build"
  let path := ".lake/build/resnet34_fwd_v.mlir"
  IO.FS.writeFile path mlir
  let cargs ← ireeCompileArgs path ".lake/build/resnet34_fwd_v.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "resnet34_fwd iree-compile OK → .lake/build/resnet34_fwd_v.vmfb"

#eval main
