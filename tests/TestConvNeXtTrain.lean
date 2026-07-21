import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.ViTRender
import LeanMlir.Types

/-! # ch9 N5 — ConvNeXt-T train-step renderer (faithful [3,3,9,3] config) + iree

Full single-batch SGD train step for the ConvNeXt-T forward of TestConvNeXtFwd.lean,
on Imagenette 224². Data-driven over the [3,3,9,3] @ [96,192,384,768] architecture,
threading spatial dims forward AND reverse. Each fragment is the StableHLO of a
proven-faithful per-op emitter (GELU `geluF`/`geluBack`, LN = global scalar BN
`bnF`/`bnBack`, layerScale, depthwise 7×7, 1×1 convs, even-kernel strided patchify/
downsample with the hand-verified transposed backward, residual, GAP, dense).

Backward threads in reverse: softmax-CE cotangent → dense+GAP → head-LN → [stage4
blocks → down2 → stage3 blocks → down1 → stage2 blocks → down0 → stage1 blocks] →
patchify weight-grad. Block backward: addV fan-in → layerScale → project → gelu →
expand → LN → depthwise → +skip.

Run (rocm): export IREE_BACKEND=rocm; lake env lean tests/TestConvNeXtTrain.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def IMG : Nat := 224
private def EPS : String := "1.0e-6"
private def LR : String := "0.1"

-- ════════════ forward fragments ════════════

private def conv1 (o x w bnm : String) (oc ic Hh Ww : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,Hh,Ww]}, {ty [oc,ic,1,1]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Hh,Ww]}\n"

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

private def layerScaleF (o x g : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply {x}, %{o}gb : {ty [BS,c,Hh,Ww]}\n"

private def patchify (o x w bnm : String) (oc ic Ho Wo s : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [{s}, {s}], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,s*Ho,s*Wo]}, {ty [oc,ic,s,s]}) -> {ty [BS,oc,Ho,Wo]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Ho,Wo]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,Ho,Wo]}\n"

private def convDown (o x w bnm : String) (oc ic H W : Nat) : String :=
  s!"    %{o}c = stablehlo.convolution({x}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,ic,2*H,2*W]}, {ty [oc,ic,2,2]}) -> {ty [BS,oc,H,W]}\n" ++
  s!"    %{o}bb = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,H,W]}\n" ++
  s!"    %{o} = stablehlo.add %{o}c, %{o}bb : {ty [BS,oc,H,W]}\n"

-- ════════════ backward fragments ════════════

private def convBiasGrad (o dy : String) (oc Hh Ww : Nat) : String :=
  s!"    %{o} = stablehlo.reduce({dy} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

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

/-- LayerNorm backward (global scalar): 3-term input-grad recomputed from saved LN
    input `xin`, plus scalar γ-grad `%{o}dg`, β-grad `%{o}db`. -/
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

private def layerScaleBack (o xin dy g : String) (c Hh Ww : Nat) : String :=
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [c]}) -> {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.multiply %{o}gb, {dy} : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}xdy = stablehlo.multiply {xin}, {dy} : {ty [BS,c,Hh,Ww]}\n" ++
  s!"    %{o}dg = stablehlo.reduce(%{o}xdy init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [c]}\n"

/-- Downsample input-grad: dilate dy (interior 1, high 1 → 2H), reverse(Wᵀ), conv pad [[1,0],[1,0]]. -/
private def convDownBack (o dy w : String) (ic oc H W : Nat) : String :=
  s!"    %{o}u = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [BS,oc,H,W]}, tensor<f32>) -> {ty [BS,oc,2*H,2*W]}\n" ++
  s!"    %{o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,ic,2,2]}) -> {ty [ic,oc,2,2]}\n" ++
  s!"    %{o}r = stablehlo.reverse %{o}t, dims = [2, 3] : {ty [ic,oc,2,2]}\n" ++
  s!"    %{o} = stablehlo.convolution(%{o}u, %{o}r)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [BS,oc,2*H,2*W]}, {ty [ic,oc,2,2]}) -> {ty [BS,ic,2*H,2*W]}\n"

/-- Downsample weight-grad: dilate dy (interior 1, no high → 2H-1), valid conv (x lhs, dilated dy rhs). -/
private def convDownWGrad (o inp dy : String) (ic oc H W : Nat) : String :=
  s!"    %{o}u = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : ({ty [BS,oc,H,W]}, tensor<f32>) -> {ty [BS,oc,2*H-1,2*W-1]}\n" ++
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,2*H,2*W]}) -> {ty [ic,BS,2*H,2*W]}\n" ++
  s!"    %{o}dt = stablehlo.transpose %{o}u, dims = [1, 0, 2, 3] : ({ty [BS,oc,2*H-1,2*W-1]}) -> {ty [oc,BS,2*H-1,2*W-1]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,2*H,2*W]}, {ty [oc,BS,2*H-1,2*W-1]}) -> {ty [ic,oc,2,2]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,2,2]}) -> {ty [oc,ic,2,2]}\n"

/-- Patchify weight-grad: dilate dy by s (interior s-1, no high → s·Ho-(s-1)), valid conv. -/
private def patchifyWGrad (o inp dy : String) (ic oc Ho Wo s : Nat) : String :=
  let dilH := s*Ho - (s-1)
  let dilW := s*Wo - (s-1)
  s!"    %{o}u = stablehlo.pad {dy}, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, {s-1}, {s-1}] : ({ty [BS,oc,Ho,Wo]}, tensor<f32>) -> {ty [BS,oc,dilH,dilW]}\n" ++
  s!"    %{o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [BS,ic,s*Ho,s*Wo]}) -> {ty [ic,BS,s*Ho,s*Wo]}\n" ++
  s!"    %{o}dt = stablehlo.transpose %{o}u, dims = [1, 0, 2, 3] : ({ty [BS,oc,dilH,dilW]}) -> {ty [oc,BS,dilH,dilW]}\n" ++
  s!"    %{o}raw = stablehlo.convolution(%{o}xt, %{o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,s*Ho,s*Wo]}, {ty [oc,BS,dilH,dilW]}) -> {ty [ic,oc,s,s]}\n" ++
  s!"    %{o} = stablehlo.transpose %{o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,s,s]}) -> {ty [oc,ic,s,s]}\n"

-- ════════════ block / downsample forward + backward ════════════

private def blockFwd (p x : String) (c e h : Nat) : String × String :=
  let code :=
    dwconv s!"{p}d" x s!"%{p}dW" s!"%{p}db" c h h 7 ++
    lnFwd s!"{p}n" s!"%{p}d" s!"%{p}ng" s!"%{p}nbt" c h h ++
    conv1 s!"{p}e" s!"%{p}n" s!"%{p}eW" s!"%{p}eb" e c h h ++
    geluAct s!"{p}g" s!"%{p}e" e h h ++
    conv1 s!"{p}p" s!"%{p}g" s!"%{p}pW" s!"%{p}pb" c e h h ++
    layerScaleF s!"{p}ls" s!"%{p}p" s!"%{p}lg" c h h ++
    addOp s!"{p}o" s!"%{p}ls" x c h h
  (code, s!"%{p}o")

private def blockBack (p dy xin : String) (c e h : Nat) : String × String :=
  let code :=
    layerScaleBack s!"{p}dls" s!"%{p}p" dy s!"%{p}lg" c h h ++
    conv1Back s!"{p}dp" s!"%{p}dls" s!"%{p}pW" e c h h ++
    conv1WGrad s!"{p}dpW" s!"%{p}g" s!"%{p}dls" e c h h ++
    convBiasGrad s!"{p}dpb" s!"%{p}dls" c h h ++
    geluBack s!"{p}dg" s!"%{p}e" s!"%{p}dp" e h h ++
    conv1Back s!"{p}de" s!"%{p}dg" s!"%{p}eW" c e h h ++
    conv1WGrad s!"{p}deW" s!"%{p}n" s!"%{p}dg" c e h h ++
    convBiasGrad s!"{p}deb" s!"%{p}dg" e h h ++
    lnBack s!"{p}dn" s!"%{p}d" s!"%{p}de" s!"%{p}ng" c h h ++
    dwconvBack s!"{p}dd" s!"%{p}dn" s!"%{p}dW" c h h 7 ++
    dwconvWGrad s!"{p}ddW" xin s!"%{p}dn" c h h 7 ++
    convBiasGrad s!"{p}ddb" s!"%{p}dn" c h h ++
    addOp s!"{p}dx" s!"%{p}dd" dy c h h
  (code, s!"%{p}dx")

private def downFwd (d x : String) (ci co hin : Nat) : String × String :=
  let code :=
    lnFwd s!"{d}n" x s!"%{d}ng" s!"%{d}nbt" ci hin hin ++
    convDown s!"{d}c" s!"%{d}n" s!"%{d}W" s!"%{d}b" co ci (hin/2) (hin/2)
  (code, s!"%{d}c")

private def downBack (d dy xin : String) (ci co hin : Nat) : String × String :=
  let code :=
    convDownBack s!"{d}dc" dy s!"%{d}W" ci co (hin/2) (hin/2) ++
    convDownWGrad s!"{d}dW" s!"%{d}n" dy ci co (hin/2) (hin/2) ++
    convBiasGrad s!"{d}db" dy co (hin/2) (hin/2) ++
    lnBack s!"{d}dn" xin s!"%{d}dc" s!"%{d}ng" ci hin hin
  (code, s!"%{d}dn")

private def depths : Array Nat := #[3, 3, 9, 3]
private def dims   : Array Nat := #[96, 192, 384, 768]
private def spats  : Array Nat := #[56, 28, 14, 7]

-- ════════════ param list (forward order); single source for sig + grads + update ════════════

private def blockParams (p : String) (c e : Nat) : List (String × String × List Nat) :=
  [(s!"{p}dW", s!"%{p}ddW", [c,1,7,7]), (s!"{p}db", s!"%{p}ddb", [c]),
   (s!"{p}ng", s!"%{p}dndg", []), (s!"{p}nbt", s!"%{p}dndb", []),
   (s!"{p}eW", s!"%{p}deW", [e,c,1,1]), (s!"{p}eb", s!"%{p}deb", [e]),
   (s!"{p}pW", s!"%{p}dpW", [c,e,1,1]), (s!"{p}pb", s!"%{p}dpb", [c]),
   (s!"{p}lg", s!"%{p}dlsdg", [c])]

private def downParams (d : String) (ci co : Nat) : List (String × String × List Nat) :=
  [(s!"{d}ng", s!"%{d}dndg", []), (s!"{d}nbt", s!"%{d}dndb", []),
   (s!"{d}W", s!"%{d}dW", [co,ci,2,2]), (s!"{d}b", s!"%{d}db", [co])]

private def allParams : List (String × String × List Nat) := Id.run do
  let mut ps : List (String × String × List Nat) :=
    [("psW", "%psdW", [96,3,4,4]), ("psb", "%psdb", [96])]
  for si in [0:4] do
    let c := dims[si]!
    let e := 4 * c
    for j in [0:depths[si]!] do
      ps := ps ++ blockParams s!"s{si}b{j}" c e
    if si < 3 then
      ps := ps ++ downParams s!"d{si}" c dims[si+1]!
  ps := ps ++ [("hng", "%hddg", []), ("hnbt", "%hddb", []),
               ("Wd", "%dWd", [768,10]), ("bd", "%dbd", [10])]
  return ps

private def sgd (θ dθ ty' : String) : String :=
  s!"    %{θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    %{θ}s = stablehlo.multiply {dθ}, %{θ}l : {ty'}\n" ++
  s!"    %{θ}n = stablehlo.subtract %{θ}, %{θ}s : {ty'}\n"

-- ════════════ whole train step ════════════

/-- The forward + backward body, SHARED by the SGD (`trainStep`) and AdamW (`trainStepAdamSched`)
    renders. `cot` is spliced between forward and backward; it must define `%dy` (the cotangent the
    backward reads) — and `%loss` for the Adam path. -/
private def renderBody (cot : String) : String := Id.run do
  let mut fwd := "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n"
    ++ s!"    %xr = stablehlo.reshape %x : ({ty [BS,3*IMG*IMG]}) -> {ty [BS,3,IMG,IMG]}\n"
    ++ patchify "ps" "%xr" "%psW" "%psb" 96 3 56 56 4
  let mut cur := "%ps"
  -- record (isBlock, prefix, inputName, a, b, h) in forward order
  let mut io : List (Bool × String × String × Nat × Nat × Nat) := []
  for si in [0:4] do
    let c := dims[si]!
    let e := 4 * c
    let h := spats[si]!
    for j in [0:depths[si]!] do
      io := io ++ [(true, s!"s{si}b{j}", cur, c, e, h)]
      let (code, out) := blockFwd s!"s{si}b{j}" cur c e h
      fwd := fwd ++ code; cur := out
    if si < 3 then
      io := io ++ [(false, s!"d{si}", cur, c, dims[si+1]!, h)]
      let (code, out) := downFwd s!"d{si}" cur c dims[si+1]! h
      fwd := fwd ++ code; cur := out
  -- head: GAP → LN(768) → dense 768→10
  fwd := fwd
    ++ s!"    %gaps = stablehlo.reduce({cur} init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,768,7,7]}, tensor<f32>) -> {ty [BS,768]}\n"
    ++ s!"    %gapnf = stablehlo.constant dense<49.0> : {ty [BS,768]}\n"
    ++ s!"    %gap = stablehlo.divide %gaps, %gapnf : {ty [BS,768]}\n"
    ++ s!"    %gapr = stablehlo.reshape %gap : ({ty [BS,768]}) -> {ty [BS,768,1,1]}\n"
    ++ lnFwd "hn" "%gapr" "%hng" "%hnbt" 768 1 1
    ++ s!"    %hnf = stablehlo.reshape %hn : ({ty [BS,768,1,1]}) -> {ty [BS,768]}\n"
    ++ s!"    %ld = stablehlo.dot_general %hnf, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,768]}, {ty [768,10]}) -> {ty [BS,10]}\n"
    ++ s!"    %ldb = stablehlo.broadcast_in_dim %bd, dims = [1] : ({ty [10]}) -> {ty [BS,10]}\n"
    ++ s!"    %logits = stablehlo.add %ld, %ldb : {ty [BS,10]}\n"
  -- backward: dense + GAP → head-LN → blocks/downs reversed → patchify (reads `%dy` from `cot`)
  let mut bwd :=
    s!"    %dhnf = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : ({ty [BS,10]}, {ty [768,10]}) -> {ty [BS,768]}\n"
    ++ s!"    %dWd = stablehlo.dot_general %hnf, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,768]}, {ty [BS,10]}) -> {ty [768,10]}\n"
    ++ s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    ++ s!"    %dhnr = stablehlo.reshape %dhnf : ({ty [BS,768]}) -> {ty [BS,768,1,1]}\n"
    ++ lnBack "hd" "%gapr" "%dhnr" "%hng" 768 1 1
    ++ s!"    %hdf = stablehlo.reshape %hd : ({ty [BS,768,1,1]}) -> {ty [BS,768]}\n"
    ++ s!"    %dgd = stablehlo.divide %hdf, %gapnf : {ty [BS,768]}\n"
    ++ s!"    %dgap = stablehlo.broadcast_in_dim %dgd, dims = [0, 1] : ({ty [BS,768]}) -> {ty [BS,768,7,7]}\n"
  let mut d := "%dgap"
  for (isBlk, p, xin, a, b, h) in io.reverse do
    if isBlk then
      let (code, out) := blockBack p d xin a b h
      bwd := bwd ++ code; d := out
    else
      let (code, out) := downBack p d xin a b h
      bwd := bwd ++ code; d := out
  -- stem (patchify) weight + bias grad (first layer; no input grad)
  bwd := bwd
    ++ patchifyWGrad "psdW" "%xr" d 3 96 56 56 4
    ++ convBiasGrad "psdb" d 96 56 56
  return fwd ++ cot ++ bwd

/-- SGD loss cotangent dy = (softmax(logits) − onehot) / B. -/
private def sgdCot : String :=
  s!"    %le = stablehlo.exponential %logits : {ty [BS,10]}\n"
  ++ s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [BS,10]}, tensor<f32>) -> {ty [BS]}\n"
  ++ s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [BS]}) -> {ty [BS,10]}\n"
  ++ s!"    %lsm = stablehlo.divide %le, %lsb : {ty [BS,10]}\n"
  ++ s!"    %dyr = stablehlo.subtract %lsm, %onehot : {ty [BS,10]}\n"
  ++ s!"    %bnc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n"
  ++ s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [BS,10]}\n"

private def trainStep : String :=
  let body := renderBody sgdCot
  -- SGD + signature/return from the single param list
  let upd := String.join (allParams.map (fun (nm, gr, ds) => sgd nm gr (ty ds)))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS,3*IMG*IMG]) :: allParams.map (fun (nm, _, ds) => s!"%{nm}: {ty ds}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, _, ds) => ty ds))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _, _) => s!"%{nm}n"))
  "module @m {\n" ++ s!"  func.func @convnext_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    body ++ upd ++ s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

-- ════════════ AdamW scheduled train step (loss-curve parity with the convnext reference) ════════════

/-- β₁/β₂/ε/wd baked (the convnext reference recipe); `%lr`/`%bc1`/`%bc2` arrive as runtime args. -/
private def adamConsts : String :=
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"

/-- AdamW loss cotangent with label smoothing α=0.1 (off-class mass α/K, K=10) + the in-graph
    smoothed-CE loss `%loss` for logging — same mechanism as the ViT/mnv2/enet/r34 sched renders.
    Defines `%lsm` (softmax), `%dy` (smoothed cotangent), `%loss`. -/
private def adamCot : String :=
  let ls : Float := 0.1
  let lsK : Float := ls / 10.0
  s!"    %le = stablehlo.exponential %logits : {ty [BS,10]}\n"
  ++ s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [BS,10]}, tensor<f32>) -> {ty [BS]}\n"
  ++ s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [BS]}) -> {ty [BS,10]}\n"
  ++ s!"    %lsm = stablehlo.divide %le, %lsb : {ty [BS,10]}\n"
  ++ s!"    %dyr0 = stablehlo.subtract %lsm, %onehot : {ty [BS,10]}\n"
  ++ s!"    %lsa = stablehlo.constant dense<{ls}> : {ty [BS,10]}\n"
  ++ s!"    %lsaoh = stablehlo.multiply %lsa, %onehot : {ty [BS,10]}\n"
  ++ s!"    %dyr1 = stablehlo.add %dyr0, %lsaoh : {ty [BS,10]}\n"
  ++ s!"    %lsaik = stablehlo.constant dense<{lsK}> : {ty [BS,10]}\n"
  ++ s!"    %dyr = stablehlo.subtract %dyr1, %lsaik : {ty [BS,10]}\n"
  ++ s!"    %bnc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n"
  ++ s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [BS,10]}\n"
  ++ s!"    %llog = stablehlo.log %lsm : {ty [BS,10]}\n"
  ++ s!"    %ohll = stablehlo.multiply %onehot, %llog : {ty [BS,10]}\n"
  ++ s!"    %t1s = stablehlo.reduce(%ohll init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [BS,10]}, tensor<f32>) -> {ty [BS]}\n"
  ++ s!"    %lls = stablehlo.reduce(%llog init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [BS,10]}, tensor<f32>) -> {ty [BS]}\n"
  ++ s!"    %omac = stablehlo.constant dense<{1.0 - ls}> : {ty [BS]}\n"
  ++ s!"    %aKc = stablehlo.constant dense<{lsK}> : {ty [BS]}\n"
  ++ s!"    %lt1 = stablehlo.multiply %omac, %t1s : {ty [BS]}\n"
  ++ s!"    %lt2 = stablehlo.multiply %aKc, %lls : {ty [BS]}\n"
  ++ s!"    %lpe = stablehlo.add %lt1, %lt2 : {ty [BS]}\n"
  ++ s!"    %lsum2 = stablehlo.reduce(%lpe init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS]}, tensor<f32>) -> tensor<f32>\n"
  ++ s!"    %lbfc = stablehlo.constant dense<{BS}.0> : tensor<f32>\n"
  ++ s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n"
  ++ s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"

/-- `@convnext_adam_train_step` — the proof-rendered fwd/bwd/param-grads with the SGD update swapped
    for `ViTRender.emitAdamV` and the `[θ|m|v]` + scalar-tail packed signature the generic
    `VerifiedNet.trainAdamSched` driver expects:
    `(x, θ×k, m×k, v×k, lr, bc1, bc2, onehot) → (θ'×k, m'×k, v'×k, loss, bc1, bc2)` (k=180).
    `lr`/`bc1`/`bc2` runtime (cosine+warmup + per-step bias correction); `bc1`/`bc2` pass through.
    Scalar LN params (dims `[]` → `tensor<f32>`) go through `emitAdamV` as rank-0 (identity casts). -/
private def trainStepAdamSched : String :=
  let body := renderBody adamCot
  let updParts := allParams.map (fun (nm, gr, ds) =>
    ViTRender.emitAdamV ("%" ++ nm) gr ("%" ++ nm ++ "m") ("%" ++ nm ++ "v") ds nm)
  let upd := String.join (updParts.map (·.1))
  let thetaN := updParts.map (·.2.1)
  let mN := updParts.map (·.2.2.1)
  let vN := updParts.map (·.2.2.2)
  let psig := String.intercalate ", " (allParams.map (fun (nm, _, ds) => s!"%{nm}: {ty ds}"))
  let msig := String.intercalate ", " (allParams.map (fun (nm, _, ds) => s!"%{nm}m: {ty ds}"))
  let vsig := String.intercalate ", " (allParams.map (fun (nm, _, ds) => s!"%{nm}v: {ty ds}"))
  let argSig := ("%x: " ++ ty [BS,3*IMG*IMG]) ++ ", " ++ psig ++ ", " ++ msig ++ ", " ++ vsig ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: " ++ ty [BS,10]
  let pdims := allParams.map (fun (_, _, ds) => ds)
  let allDims := pdims ++ pdims ++ pdims
  let retTy := String.intercalate ", " ((allDims.map (fun ds => ty ds)) ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"])
  let retVals := String.intercalate ", " (thetaN ++ mN ++ vN ++ ["%loss", "%bc1", "%bc2"])
  "module @m {\n" ++ s!"  func.func @convnext_adam_train_step({argSig}) -> ({retTy}) " ++ "{\n" ++
    body ++ adamConsts ++ upd ++ s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

/-- iree-compile smoke that degrades gracefully when the compiler isn't on PATH (the render +
    write already happened, so the artifact exists regardless). -/
private def tryCompile (src dst label : String) : IO Unit := do
  try
    let cargs ← ireeCompileArgs src dst
    let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
    if r.exitCode != 0 then IO.eprintln s!"iree-compile ({label}) FAILED:\n{r.stderr.take 3000}"
    else IO.println s!"{label} iree-compile OK → {src}"
  catch e => IO.eprintln s!"iree-compile ({label}) skipped (compiler unavailable): {e}"

def main : IO Unit := do
  IO.FS.createDirAll "verified_mlir"
  IO.FS.createDirAll ".lake/build"
  -- Render + write BOTH artifacts first, then compile (so a missing iree-compile can't abort
  -- before the AdamW artifact is written).
  let mlir := trainStep
  IO.println s!"rendered ConvNeXt-T train step (BS={BS}, [3,3,9,3]): {mlir.length} chars, {allParams.length} params"
  IO.FS.writeFile "verified_mlir/convnext_train_step.mlir" mlir
  let amlir := trainStepAdamSched
  IO.println s!"rendered ConvNeXt-T AdamW-sched train step: {amlir.length} chars"
  IO.FS.writeFile "verified_mlir/convnext_adam_train_step.mlir" amlir
  -- SGD smoke (the committed train step; verifies the shared `renderBody`).
  tryCompile "verified_mlir/convnext_train_step.mlir" ".lake/build/convnext_train_step_v.vmfb" "SGD"
  -- AdamW scheduled train step — the artifact `convnext-verified-adam` trains on.
  tryCompile "verified_mlir/convnext_adam_train_step.mlir" "/tmp/convnext_adam_ts.vmfb" "AdamW"

#eval main
