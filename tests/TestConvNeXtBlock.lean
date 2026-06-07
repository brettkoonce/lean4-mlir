import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # ch9 N2+N4 — one ConvNeXt block (fwd + backward) render + iree-compile

Validates the new ConvNeXt-specific StableHLO fragments on a tiny single block before
the full ConvNeXt-T renderer. One block (stride-1, identity skip):

  x → depthwise 7×7 (dim c) → LN(global scalar γ/β) → 1×1 expand c→e → GELU
    → 1×1 project e→c → layerScale (per-channel γ:[c]) → + x

Every fragment is the StableHLO a VERIFIED per-op emitter produces:
- GELU (`geluF`/`geluBack`, N1, tanh-approx + closed-form derivative),
- LN = per-example GLOBAL scalar-γ/β batch-norm (`bnF`/`bnBack` flat, = `layerNormForward`),
- layerScale (per-channel diagonal multiply, an instance of the proven `layerScale`),
- depthwise k×k (`depthwiseF`/`depthwiseBack`, kernel-general), 1×1 conv, residual `addV`.

The backward threads the cotangent in reverse: addV fan-in → layerScale back → conv1 back
(project) → geluBack → conv1 back (expand) → lnBack → depthwiseBack → +skip. Compile-only
(shape/type check); gradient correctness is validated by the full-net training run.

Run (rocm): export PATH="$PWD/.venv/bin:$PATH"; export IREE_BACKEND=rocm
  lake env lean tests/TestConvNeXtBlock.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 2
private def EPS : String := "1.0e-6"

-- ════════════ shared forward fragments (4-D [B,C,H,W]) ════════════

/-- 1×1 conv (pad 0, stride 1) + bias. -/
private def conv1 (o x w bnm : String) (oc ic Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hh,Ww]}, {ty [oc,ic,1,1]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hh,Ww]}\n"

/-- Depthwise k×k SAME conv, STRIDE 1 (feature_group_count=c, [c,1,k,k], pad (k-1)/2). -/
private def dwconv (o x w bnm : String) (c Hh Ww k : Nat) : String :=
  let p := (k-1)/2
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,k,k]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,c,Hh,Ww]}\n"

private def addOp (o a b : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.add {a}, {b} : {ty [BS,oc,Hh,Ww]}\n"

/-- GELU forward (tanh approx), the 4-D render of `geluF`:
    `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. Saves nothing — backward recomputes. -/
private def geluAct (o x : String) (c Hh Ww : Nat) : String :=
  let t := ty [BS,c,Hh,Ww]
  s!"    %{o}x2 = stablehlo.multiply {x}, {x} : {t}\n" ++
  s!"    %{o}x3 = stablehlo.multiply %{o}x2, {x} : {t}\n" ++
  s!"    %{o}ck = stablehlo.constant dense<0.044715> : {t}\n" ++
  s!"    %{o}kx3 = stablehlo.multiply %{o}ck, %{o}x3 : {t}\n" ++
  s!"    %{o}inn = stablehlo.add {x}, %{o}kx3 : {t}\n" ++
  s!"    %{o}cs = stablehlo.constant dense<0.7978845608028654> : {t}\n" ++
  s!"    %{o}u = stablehlo.multiply %{o}cs, %{o}inn : {t}\n" ++
  s!"    %{o}t = stablehlo.tanh %{o}u : {t}\n" ++
  s!"    %{o}one = stablehlo.constant dense<1.0> : {t}\n" ++
  s!"    %{o}opt = stablehlo.add %{o}one, %{o}t : {t}\n" ++
  s!"    %{o}half = stablehlo.constant dense<0.5> : {t}\n" ++
  s!"    %{o}hx = stablehlo.multiply %{o}half, {x} : {t}\n" ++
  s!"    %{o} = stablehlo.multiply %{o}hx, %{o}opt : {t}\n"

/-- GELU backward `dy ⊙ gelu'(pre)`, recomputing `t = tanh(u(pre))` from the saved
    pre-activation — the 4-D render of `geluBack`. -/
private def geluBack (o pre dy : String) (c Hh Ww : Nat) : String :=
  let t := ty [BS,c,Hh,Ww]
  s!"    %{o}x2 = stablehlo.multiply {pre}, {pre} : {t}\n" ++
  s!"    %{o}x3 = stablehlo.multiply %{o}x2, {pre} : {t}\n" ++
  s!"    %{o}ck = stablehlo.constant dense<0.044715> : {t}\n" ++
  s!"    %{o}kx3 = stablehlo.multiply %{o}ck, %{o}x3 : {t}\n" ++
  s!"    %{o}inn = stablehlo.add {pre}, %{o}kx3 : {t}\n" ++
  s!"    %{o}cs = stablehlo.constant dense<0.7978845608028654> : {t}\n" ++
  s!"    %{o}u = stablehlo.multiply %{o}cs, %{o}inn : {t}\n" ++
  s!"    %{o}t = stablehlo.tanh %{o}u : {t}\n" ++
  s!"    %{o}one = stablehlo.constant dense<1.0> : {t}\n" ++
  s!"    %{o}opt = stablehlo.add %{o}one, %{o}t : {t}\n" ++
  s!"    %{o}half = stablehlo.constant dense<0.5> : {t}\n" ++
  s!"    %{o}term1 = stablehlo.multiply %{o}half, %{o}opt : {t}\n" ++
  s!"    %{o}t2 = stablehlo.multiply %{o}t, %{o}t : {t}\n" ++
  s!"    %{o}omt2 = stablehlo.subtract %{o}one, %{o}t2 : {t}\n" ++
  s!"    %{o}hx = stablehlo.multiply %{o}half, {pre} : {t}\n" ++
  s!"    %{o}hxo = stablehlo.multiply %{o}hx, %{o}omt2 : {t}\n" ++
  s!"    %{o}c3b = stablehlo.constant dense<0.134145> : {t}\n" ++
  s!"    %{o}a3x2 = stablehlo.multiply %{o}c3b, %{o}x2 : {t}\n" ++
  s!"    %{o}in2 = stablehlo.add %{o}one, %{o}a3x2 : {t}\n" ++
  s!"    %{o}up = stablehlo.multiply %{o}cs, %{o}in2 : {t}\n" ++
  s!"    %{o}term2 = stablehlo.multiply %{o}hxo, %{o}up : {t}\n" ++
  s!"    %{o}gp = stablehlo.add %{o}term1, %{o}term2 : {t}\n" ++
  s!"    %{o} = stablehlo.multiply {dy}, %{o}gp : {t}\n"

/-- LayerNorm forward = per-example GLOBAL scalar-γ/β batch-norm over the flattened
    c·h·w feature vec (`= layerNormForward`, the flat render of `bnF`): reshape to
    `[B, c·h·w]`, reduce μ/var over [1], scalar γ/β (rank-0). -/
private def lnFwd (o x g bt : String) (c Hh Ww : Nat) : String :=
  let n := c*Hh*Ww
  let tn := ty [BS,n]
  s!"    %{o}ri = stablehlo.reshape {x} : ({ty [BS,c,Hh,Ww]}) -> {tn}\n" ++
  s!"    %{o}nf = stablehlo.constant dense<{n}.0> : {tn}\n" ++
  s!"    %{o}ep = stablehlo.constant dense<{EPS}> : {tn}\n" ++
  s!"    %{o}smr = stablehlo.reduce(%{o}ri init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %{o}sm = stablehlo.broadcast_in_dim %{o}smr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    %{o}mu = stablehlo.divide %{o}sm, %{o}nf : {tn}\n" ++
  s!"    %{o}xc = stablehlo.subtract %{o}ri, %{o}mu : {tn}\n" ++
  s!"    %{o}sq = stablehlo.multiply %{o}xc, %{o}xc : {tn}\n" ++
  s!"    %{o}vsr = stablehlo.reduce(%{o}sq init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %{o}vs = stablehlo.broadcast_in_dim %{o}vsr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    %{o}vr = stablehlo.divide %{o}vs, %{o}nf : {tn}\n" ++
  s!"    %{o}ve = stablehlo.add %{o}vr, %{o}ep : {tn}\n" ++
  s!"    %{o}istd = stablehlo.rsqrt %{o}ve : {tn}\n" ++
  s!"    %{o}xh = stablehlo.multiply %{o}xc, %{o}istd : {tn}\n" ++
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [] : (tensor<f32>) -> {tn}\n" ++
  s!"    %{o}btb = stablehlo.broadcast_in_dim {bt}, dims = [] : (tensor<f32>) -> {tn}\n" ++
  s!"    %{o}gx = stablehlo.multiply %{o}xh, %{o}gb : {tn}\n" ++
  s!"    %{o}fl = stablehlo.add %{o}gx, %{o}btb : {tn}\n" ++
  s!"    %{o} = stablehlo.reshape %{o}fl : ({tn}) -> {ty [BS,c,Hh,Ww]}\n"

/-- Per-channel layerScale forward `γ ⊙ x` (γ:[c] broadcast over spatial) — an
    instance of the proven `layerScale` with channel-constant γ. -/
private def layerScaleF (o x g : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply {x}, %{o}gb : {ty [BS,c,Hh,Ww]}\n"

-- ════════════ backward fragments ════════════

private def conv1Back (o dy w : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,1,1]}) -> {ty [ic,oc,1,1]}\n" ++
  s!"    %{o} = stablehlo.convolution({dy}, %{o}t)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,oc,Hh,Ww]}, {ty [ic,oc,1,1]}) -> {ty [BS,ic,Hh,Ww]}\n"

private def conv1WGrad (o inp dy : String) (ic oc Hh Ww : Nat) : String :=
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Ww]}) -> {ty [ic,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Ww]}) -> {ty [oc,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Ww]}, {ty [oc,BS,Hh,Ww]}) -> {ty [ic,oc,1,1]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,1,1]}) -> {ty [oc,ic,1,1]}\n"

private def dwconvBack (o dy w : String) (c Hh Ww k : Nat) : String :=
  let p := (k-1)/2
  s!"    %{o}rev = stablehlo.reverse {w}, dims = [2, 3] : {ty [c,1,k,k]}\n" ++
  s!"    %{o} = stablehlo.convolution({dy}, %{o}rev)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = " ++ toString c ++ " : i64}" ++
  s!" : ({ty [BS,c,Hh,Ww]}, {ty [c,1,k,k]}) -> {ty [BS,c,Hh,Ww]}\n"

private def dwconvWGrad (o inp dy : String) (c Hh Ww k : Nat) : String :=
  let p := (k-1)/2
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    %{o}dt = stablehlo.transpose {dy}, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{p}, {p}], [{p}, {p}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [c,BS,Hh,Ww]}, {ty [c,BS,Hh,Ww]}) -> {ty [1,c,k,k]}\n" ++
  s!"    %{o} = stablehlo.reshape %{o}raw : ({ty [1,c,k,k]}) -> {ty [c,1,k,k]}\n"

private def convBiasGrad (o dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

/-- LayerNorm backward (global scalar): input-grad (3-term `bn_grad_input`) recomputed
    from the saved LN input `xin`, plus scalar γ-grad `%{o}dg` and β-grad `%{o}db`. -/
private def lnBack (o xin dy g : String) (c Hh Ww : Nat) : String :=
  let n := c*Hh*Ww
  let tn := ty [BS,n]
  s!"    %{o}ri = stablehlo.reshape {xin} : ({ty [BS,c,Hh,Ww]}) -> {tn}\n" ++
  s!"    %{o}rdy = stablehlo.reshape {dy} : ({ty [BS,c,Hh,Ww]}) -> {tn}\n" ++
  s!"    %{o}nf = stablehlo.constant dense<{n}.0> : {tn}\n" ++
  s!"    %{o}ep = stablehlo.constant dense<{EPS}> : {tn}\n" ++
  s!"    %{o}smr = stablehlo.reduce(%{o}ri init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %{o}sm = stablehlo.broadcast_in_dim %{o}smr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    %{o}mu = stablehlo.divide %{o}sm, %{o}nf : {tn}\n" ++
  s!"    %{o}xc = stablehlo.subtract %{o}ri, %{o}mu : {tn}\n" ++
  s!"    %{o}sq = stablehlo.multiply %{o}xc, %{o}xc : {tn}\n" ++
  s!"    %{o}vsr = stablehlo.reduce(%{o}sq init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %{o}vs = stablehlo.broadcast_in_dim %{o}vsr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    %{o}vr = stablehlo.divide %{o}vs, %{o}nf : {tn}\n" ++
  s!"    %{o}ve = stablehlo.add %{o}vr, %{o}ep : {tn}\n" ++
  s!"    %{o}istd = stablehlo.rsqrt %{o}ve : {tn}\n" ++
  s!"    %{o}xh = stablehlo.multiply %{o}xc, %{o}istd : {tn}\n" ++
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [] : (tensor<f32>) -> {tn}\n" ++
  s!"    %{o}dxh = stablehlo.multiply %{o}gb, %{o}rdy : {tn}\n" ++
  s!"    %{o}sdxr = stablehlo.reduce(%{o}dxh init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %{o}sdx = stablehlo.broadcast_in_dim %{o}sdxr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    %{o}xd = stablehlo.multiply %{o}xh, %{o}dxh : {tn}\n" ++
  s!"    %{o}sxdr = stablehlo.reduce(%{o}xd init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %{o}sxd = stablehlo.broadcast_in_dim %{o}sxdr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    %{o}t1 = stablehlo.multiply %{o}dxh, %{o}nf : {tn}\n" ++
  s!"    %{o}i1 = stablehlo.subtract %{o}t1, %{o}sdx : {tn}\n" ++
  s!"    %{o}xs = stablehlo.multiply %{o}xh, %{o}sxd : {tn}\n" ++
  s!"    %{o}i2 = stablehlo.subtract %{o}i1, %{o}xs : {tn}\n" ++
  s!"    %{o}sN = stablehlo.divide %{o}istd, %{o}nf : {tn}\n" ++
  s!"    %{o}gin = stablehlo.multiply %{o}sN, %{o}i2 : {tn}\n" ++
  s!"    %{o} = stablehlo.reshape %{o}gin : ({tn}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}dgp = stablehlo.multiply %{o}rdy, %{o}xh : {tn}\n" ++
  s!"    %{o}dg = stablehlo.reduce(%{o}dgp init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tn}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    %{o}db = stablehlo.reduce(%{o}rdy init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tn}, tensor<f32>) -> tensor<f32>\n"

/-- Per-channel layerScale backward: input-grad `γ ⊙ dy` + γ-grad `Σ_{batch,spatial}(x ⊙ dy)`. -/
private def layerScaleBack (o xin dy g : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply %{o}gb, {dy} : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}xdy = stablehlo.multiply {xin}, {dy} : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}dg = stablehlo.reduce(%{o}xdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [c]}\n"

-- ════════════ one block (c=8, e=32, h=w=8, k=7) ════════════

private def C : Nat := 8
private def E : Nat := 32
private def H : Nat := 8
private def K : Nat := 7

private def blockModule : String := Id.run do
  let body :=
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    -- forward
    dwconv "b1d" "%x" "%b1dW" "%b1db" C H H K ++
    lnFwd "b1n" "%b1d" "%b1ng" "%b1nbt" C H H ++
    conv1 "b1e" "%b1n" "%b1eW" "%b1eb" E C H H ++
    geluAct "b1g" "%b1e" E H H ++
    conv1 "b1p" "%b1g" "%b1pW" "%b1pb" C E H H ++
    layerScaleF "b1ls" "%b1p" "%b1lg" C H H ++
    addOp "b1o" "%b1ls" "%x" C H H ++
    -- backward (incoming cotangent %dy into the block output = ls + x)
    layerScaleBack "b1dls" "%b1p" "%dy" "%b1lg" C H H ++
    conv1Back "b1dp" "%b1dls" "%b1pW" E C H H ++
    conv1WGrad "b1dpW" "%b1g" "%b1dls" E C H H ++
    convBiasGrad "b1dpb" "%b1dls" C H H ++
    geluBack "b1dg" "%b1e" "%b1dp" E H H ++
    conv1Back "b1de" "%b1dg" "%b1eW" C E H H ++
    conv1WGrad "b1deW" "%b1n" "%b1dg" C E H H ++
    convBiasGrad "b1deb" "%b1dg" E H H ++
    lnBack "b1dn" "%b1d" "%b1de" "%b1ng" C H H ++
    dwconvBack "b1dd" "%b1dn" "%b1dW" C H H K ++
    dwconvWGrad "b1ddW" "%x" "%b1dn" C H H K ++
    convBiasGrad "b1ddb" "%b1dn" C H H ++
    addOp "b1dx" "%b1dd" "%dy" C H H
  let sig := String.intercalate ", "
    [s!"%x: {ty [BS,C,H,H]}", s!"%dy: {ty [BS,C,H,H]}",
     s!"%b1dW: {ty [C,1,K,K]}", s!"%b1db: {ty [C]}",
     s!"%b1ng: tensor<f32>", s!"%b1nbt: tensor<f32>",
     s!"%b1eW: {ty [E,C,1,1]}", s!"%b1eb: {ty [E]}",
     s!"%b1pW: {ty [C,E,1,1]}", s!"%b1pb: {ty [C]}",
     s!"%b1lg: {ty [C]}"]
  -- return: block output, input-grad, all weight/bias/scale grads
  let retTy := String.intercalate ", "
    [ty [BS,C,H,H], ty [BS,C,H,H], ty [C,1,K,K], ty [C], "tensor<f32>", "tensor<f32>",
     ty [E,C,1,1], ty [E], ty [C,E,1,1], ty [C], ty [C]]
  let retV := "%b1o, %b1dx, %b1ddW, %b1ddb, %b1dndg, %b1dndb, %b1deW, %b1deb, %b1dpW, %b1dpb, %b1dlsdg"
  return "module @m {\n" ++ s!"  func.func @convnext_block({sig}) -> ({retTy}) " ++ "{\n" ++
    body ++ s!"    return {retV} : {retTy}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := blockModule
  IO.println s!"rendered one ConvNeXt block fwd+back (BS={BS}, c={C}, e={E}, {H}²): {mlir.length} chars"
  IO.FS.createDirAll ".lake/build"
  let path := ".lake/build/convnext_block.mlir"
  IO.FS.writeFile path mlir
  let cargs ← ireeCompileArgs path ".lake/build/convnext_block.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 3000}"
  else
    IO.println "convnext_block iree-compile OK → .lake/build/convnext_block.vmfb"

#eval main
