import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # B9a — ResNet-34 **eval** forward renderer (running-stats BN) + `iree-compile` validation

Programmatic StableHLO for the ResNet-34 *inference* forward (IMAGENETTE 3×224×224 input —
the paper's native ImageNet resolution): 7×7 stride-2 stem (3→64, 224→112) → 2×2
maxpool (112→56) → 4 stages of basic blocks `[3,4,6,3]` at channels `64/128/256/512`
(stages 2–4 open with a strided downsample block, 56→28→14→7) → global-average-pool →
dense(512→10), with **affine BN consuming per-layer running mean/var** rather than
batch statistics. This is the eval partner of the Adam train step
(`tests/TestResnet34Train.lean`), which trains with true batch-norm and EMAs its batch
stats out through passthrough slots.

**`@resnet34_fwd` no longer lives here** (`planning/xla_pjrt_handoff.md` §2a). It is rendered by
`LeanMlir/Proofs/Codegen/ResNet34Render.lean` as `pretty(provenGraph)`, sharing its forward chain
with the certified train step — its body is a byte-identical prefix of
`verified_mlir/resnet34_train_step.mlir`. The hand-written copy that used to be here computed a
**different function**: true batch-norm (reduce over `[0,2,3]`, n = B·H·W) where the certified
train step normalises **per example** (reduce over `[2,3]`, n = H·W). See the §2a notes.

Naming: helper `o` is a bare prefix; the helper's result SSA is `%{o}`, its
intermediates `%{o}…`. All value inputs carry a leading `%`.

Run (rocm):
  export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestResnet34Fwd.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
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

/-- 7×7 SAME conv, stride 2, pad 3 (the ImageNet ResNet stem: 224→112). Same proven
    stride-2 pattern as the 3×3 `conv s=2`, just kernel-7. Result SSA = `%{o}`. -/
private def convStem (o x w bnm : String) (oc ic Hin Win Hout Wout : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hin,Win]}, {ty [oc,ic,7,7]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hout,Wout]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hout,Wout]}\n"

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

-- ════════════ inference-BN (running-stats) eval forward ════════════
-- Affine-only BN consuming per-layer running mean/var (func inputs `%{o}mu`/`%{o}var`), instead of
-- computing batch stats. The `<slug>_fwd_eval.mlir` the driver evals with once running-stats are
-- threaded; exact-parity eval (class-batch-independent), unlike batch-BN eval.

/-- Affine BN with running stats: `y = γ·(x − μ)·rsqrt(var + ε) + β`, μ/var from inputs `%{o}mu`/`%{o}var`. -/
private def bnEval (o x g bt : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o}mub = stablehlo.broadcast_in_dim %{o}mu, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xc = stablehlo.subtract {x}, %{o}mub : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vb = stablehlo.broadcast_in_dim %{o}var, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ep = stablehlo.constant dense<{EPS}> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ve = stablehlo.add %{o}vb, %{o}ep : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}istd = stablehlo.rsqrt %{o}ve : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xh = stablehlo.multiply %{o}xc, %{o}istd : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}btb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gx = stablehlo.multiply %{o}xh, %{o}gb : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}gx, %{o}btb : {ty [BS,oc,Hh,Ww]}\n"

private def idBlockEval (p x : String) (c Hh Ww : Nat) : String × String :=
  let code :=
    conv s!"{p}c1" x s!"%{p}W1" s!"%{p}b1" c c Hh Ww Hh Ww 1 ++
    bnEval s!"{p}n1" s!"%{p}c1" s!"%{p}g1" s!"%{p}bt1" c Hh Ww ++
    relu s!"{p}r1" s!"%{p}n1" c Hh Ww ++
    conv s!"{p}c2" s!"%{p}r1" s!"%{p}W2" s!"%{p}b2" c c Hh Ww Hh Ww 1 ++
    bnEval s!"{p}n2" s!"%{p}c2" s!"%{p}g2" s!"%{p}bt2" c Hh Ww ++
    addOp s!"{p}a" s!"%{p}n2" x c Hh Ww ++
    relu s!"{p}o" s!"%{p}a" c Hh Ww
  (code, s!"%{p}o")

private def downBlockEval (p x : String) (c oc Hh Ww : Nat) : String × String :=
  let Hin := 2*Hh; let Win := 2*Ww
  let code :=
    conv s!"{p}c1" x s!"%{p}W1" s!"%{p}b1" oc c Hin Win Hh Ww 2 ++
    bnEval s!"{p}n1" s!"%{p}c1" s!"%{p}g1" s!"%{p}bt1" oc Hh Ww ++
    relu s!"{p}r1" s!"%{p}n1" oc Hh Ww ++
    conv s!"{p}c2" s!"%{p}r1" s!"%{p}W2" s!"%{p}b2" oc oc Hh Ww Hh Ww 1 ++
    bnEval s!"{p}n2" s!"%{p}c2" s!"%{p}g2" s!"%{p}bt2" oc Hh Ww ++
    conv s!"{p}cp" x s!"%{p}Wp" s!"%{p}bp" oc c Hin Win Hh Ww 2 ++
    bnEval s!"{p}np" s!"%{p}cp" s!"%{p}gp" s!"%{p}btp" oc Hh Ww ++
    addOp s!"{p}a" s!"%{p}n2" s!"%{p}np" oc Hh Ww ++
    relu s!"{p}o" s!"%{p}a" oc Hh Ww
  (code, s!"%{p}o")

private def idChainEval (base x : String) (n c Hh Ww : Nat) : String × String := Id.run do
  let mut code := ""; let mut cur := x
  for i in [:n] do
    let (c2, out) := idBlockEval s!"{base}b{i}" cur c Hh Ww
    code := code ++ c2; cur := out
  return (code, cur)

/-- BN running-stats input pair `(%{p}mu, %{p}var)`, both `[oc]` — canonical (forward) order. -/
private def bnStatSig (p : String) (oc : Nat) : List String :=
  [s!"%{p}mu: {ty [oc]}", s!"%{p}var: {ty [oc]}"]
private def idBlockStatSig (p : String) (c : Nat) : List String :=
  bnStatSig s!"{p}n1" c ++ bnStatSig s!"{p}n2" c
private def downBlockStatSig (p : String) (oc : Nat) : List String :=
  bnStatSig s!"{p}n1" oc ++ bnStatSig s!"{p}n2" oc ++ bnStatSig s!"{p}np" oc
private def idChainStatSig (base : String) (n c : Nat) : List String := Id.run do
  let mut acc := []
  for i in [:n] do acc := acc ++ idBlockStatSig s!"{base}b{i}" c
  return acc

private def gapDense (o x : String) (c nC Hh Ww : Nat) : String :=
  s!"    %{o}gs = stablehlo.reduce({x} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c]}\n" ++
  s!"    %{o}gnf = stablehlo.constant dense<{Hh*Ww}.0> : {ty [BS,c]}\n" ++
  s!"    %{o}g = stablehlo.divide %{o}gs, %{o}gnf : {ty [BS,c]}\n" ++
  s!"    %{o}dd = stablehlo.dot_general %{o}g, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,c]}, {ty [c,nC]}) -> {ty [BS,nC]}\n" ++
  s!"    %{o}db = stablehlo.broadcast_in_dim %bd, dims = [1] : ({ty [nC]}) -> {ty [BS,nC]}\n" ++
  s!"    %{o} = stablehlo.add %{o}dd, %{o}db : {ty [BS,nC]}\n"

-- ── whole net ──

/-- `@resnet34_fwd_eval` — the eval forward with affine running-stats BN. Same params as
    `@resnet34_fwd`, plus per-BN-layer `%{p}mu`/`%{p}var` `[oc]` inputs in BN forward order
    (the driver passes `θ ++ runningBnStats`). Returns logits `[BS,10]`. -/
private def resnet34FwdEval : String := Id.run do
  let stemCode :=
    s!"    %xr = stablehlo.reshape %x : ({ty [BS,150528]}) -> {ty [BS,3,224,224]}\n" ++
    convStem "stc" "%xr" "%sW" "%sb" 64 3 224 224 112 112 ++
    bnEval "stn" "%stc" "%sg" "%sbt" 64 112 112 ++
    relu "str" "%stn" 64 112 112 ++
    maxpool "stp" "%str" 64 112 112
  let (s1, o1) := idChainEval "s1" "%stp" 3 64 56 56
  let (d2, o2) := downBlockEval "d2" o1 64 128 28 28
  let (s2, o2b) := idChainEval "s2" o2 3 128 28 28
  let (d3, o3) := downBlockEval "d3" o2b 128 256 14 14
  let (s3, o3b) := idChainEval "s3" o3 5 256 14 14
  let (d4, o4) := downBlockEval "d4" o3b 256 512 7 7
  let (s4, o4b) := idChainEval "s4" o4 2 512 7 7
  let tail := gapDense "out" o4b 512 10 7 7
  let body :=
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    stemCode ++ s1 ++ d2 ++ s2 ++ d3 ++ s3 ++ d4 ++ s4 ++ tail
  let paramSig : List String :=
    ["%x: " ++ ty [BS,150528]]
    ++ [s!"%sW: {ty [64,3,7,7]}", s!"%sb: {ty [64]}"] ++ bnSig "s" 64
    ++ idChainSig "s1" 3 64
    ++ downBlockSig "d2" 64 128 ++ idChainSig "s2" 3 128
    ++ downBlockSig "d3" 128 256 ++ idChainSig "s3" 5 256
    ++ downBlockSig "d4" 256 512 ++ idChainSig "s4" 2 512
    ++ [s!"%Wd: {ty [512,10]}", s!"%bd: {ty [10]}"]
  -- BN running-stats inputs, BN forward order (the driver's runningBnStats layout)
  let statSig : List String :=
    bnStatSig "stn" 64
    ++ idChainStatSig "s1" 3 64
    ++ downBlockStatSig "d2" 128 ++ idChainStatSig "s2" 3 128
    ++ downBlockStatSig "d3" 256 ++ idChainStatSig "s3" 5 256
    ++ downBlockStatSig "d4" 512 ++ idChainStatSig "s4" 2 512
  let argSig := String.intercalate ", " (paramSig ++ statSig)
  return "module @m {\n" ++ s!"  func.func @resnet34_fwd_eval({argSig}) -> {ty [BS,10]} " ++ "{\n" ++
    body ++ s!"    return %out : {ty [BS,10]}\n" ++ "  }\n}\n"

private def tryCompile (src dst label : String) : IO Unit := do
  try
    let cargs ← ireeCompileArgs src dst
    let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
    if r.exitCode != 0 then IO.eprintln s!"iree-compile ({label}) FAILED:\n{r.stderr.take 3000}"
    else IO.println s!"{label} iree-compile OK → {src}"
  catch e => IO.eprintln s!"iree-compile ({label}) skipped (compiler unavailable): {e}"

-- NOTE: `verified_mlir/resnet34_fwd.mlir` is deliberately NOT written here — it belongs to
-- `LeanMlir/Proofs/Codegen/ResNet34Render.lean` (§2a). Adding a second writer back would
-- re-open the silent last-writer-wins clobber this file used to cause.
def main : IO Unit := do
  IO.FS.createDirAll "verified_mlir"
  IO.FS.createDirAll ".lake/build"
  let evalMlir := resnet34FwdEval
  IO.println s!"rendered @resnet34_fwd_eval (BS={BS}): {evalMlir.length} chars"
  IO.FS.writeFile "verified_mlir/resnet34_fwd_eval.mlir" evalMlir
  tryCompile "verified_mlir/resnet34_fwd_eval.mlir" ".lake/build/resnet34_fwd_eval_v.vmfb" "fwd_eval"

#eval main
