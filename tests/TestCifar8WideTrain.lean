import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.ViTRender
import LeanMlir.Types

/-! # `cifar8{,_bn}_adam_train_step` — the cifar8 verified train step with AdamW

The Adam peer of `verified_mlir/cifar8w_train_step.mlir` (no-BN) and
`verified_mlir/cifar8w_bn_train_step.mlir` (per-channel BN). Same forward + backward +
param-gradient body as the SGD render (`Proofs.StableHLO.cifar8{,Bn}TrainStepText` — the
readable predecessor that exposes every `%dW*`/`%db*`/`%dg*`/`%db*` gradient as a named SSA
value), with the per-param SGD update `θ − lr·∇` swapped for `ViTRender.emitAdamV`
(`θ' = θ − lr·(m̂/(√v̂+ε)) − lr·wd·θ`, op-for-op `Proofs.adamWParam`) and the
`[θ|m|v]` + scalar-tail packed signature the generic `VerifiedNet.trainAdamSched` driver
expects. The cotangent is divided by `B` (mean gradients — Adam needs the true mean, the
1/B factor cannot fold into `lr` the way it does for SGD), and the in-graph mean softmax-CE
`%loss` is emitted in the slot after `[θ'|m'|v']` for logging. No label smoothing, so the
*only* difference vs the SGD path is the optimizer map — a clean optimizer ablation.

Gotcha: cifar8's conv-bias params are named `%b1..%b8`, which would collide with
`emitAdamV`/`adamConsts`' β₁/β₂ constants `%b1`/`%b2`. The FFI packs params by shape order
(names are irrelevant to it), so we rename the conv biases `%cb1..%cb8` here. Conv-bias
*gradients* stay `%db1..%db8` (no collision).

Run (renders both .mlir): `lake env lean tests/TestCifar8AdamTrain.lean`
-/

open Proofs Proofs.StableHLO

-- ── concrete cifar8 dims (match CnnRender.lean's faithful render: [16,16,32,32], 32→2) ──
private def B : Nat := 128
private def IC : Nat := 3
private def C1 : Nat := 16
private def C2 : Nat := 16
private def C3 : Nat := 32
private def C4 : Nat := 32
private def IMH : Nat := 32
private def IMW : Nat := 32
private def KH : Nat := 3
private def KW : Nat := 3
private def D1 : Nat := 512
private def NC : Nat := 10

/-- β₁/β₂/ε/wd baked (standard AdamW); `%lr`/`%bc1`/`%bc2` arrive as runtime args. -/
private def adamConsts : String :=
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"

-- ════════════ shared body helpers (transcribed from cifar8TrainStepText) ════════════

private def pH : Nat := (KH - 1) / 2
private def pW : Nat := (KW - 1) / 2

private def dg (o a w cA cB tA tB tO : String) : String :=
  s!"    {o} = stablehlo.dot_general {a}, {w}, contracting_dims = [{cA}] x [{cB}], precision = [DEFAULT, DEFAULT] : ({tA}, {tB}) -> {tO}\n"

private def dense (oh a w bnm : String) (mm nn : Nat) : String :=
  dg s!"{oh}d" a w "1" "0" (ty [B,mm]) (ty [mm,nn]) (ty [B,nn]) ++
  s!"    {oh}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [nn]}) -> {ty [B,nn]}\n" ++
  s!"    {oh} = stablehlo.add {oh}d, {oh}b : {ty [B,nn]}\n"

private def relu2 (o h : String) (nn : Nat) : String :=
  s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
  s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,nn]}\n"

private def relu4 (o h : String) (C Hh Ww : Nat) : String :=
  s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
  s!"    {o} = stablehlo.maximum {h}, {o}z : {ty [B,C,Hh,Ww]}\n"

private def reduce0 (o dyk : String) (nn : Nat) : String :=
  s!"    {o} = stablehlo.reduce({dyk} init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [B,nn]}, tensor<f32>) -> {ty [nn]}\n"

private def selMask2 (o pre dgrad : String) (nn : Nat) : String :=
  s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,nn]}\n" ++
  s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,nn]}, {ty [B,nn]}) -> {tyI1 [B,nn]}\n" ++
  s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,nn]}, {ty [B,nn]}\n"

private def selMask4 (o pre dgrad : String) (C Hh Ww : Nat) : String :=
  s!"    {o}z = stablehlo.constant dense<0.0> : {ty [B,C,Hh,Ww]}\n" ++
  s!"    {o}m = stablehlo.compare GT, {pre}, {o}z : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}) -> {tyI1 [B,C,Hh,Ww]}\n" ++
  s!"    {o} = stablehlo.select {o}m, {dgrad}, {o}z : {tyI1 [B,C,Hh,Ww]}, {ty [B,C,Hh,Ww]}\n"

private def convFwd (o lhs w bnm : String) (oc icc Hh Ww : Nat) : String :=
  s!"    {o}c = stablehlo.convolution({lhs}, {w})\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [B,icc,Hh,Ww]}, {ty [oc,icc,KH,KW]}) -> {ty [B,oc,Hh,Ww]}\n" ++
  s!"    {o}b = stablehlo.broadcast_in_dim {bnm}, dims = [1] : ({ty [oc]}) -> {ty [B,oc,Hh,Ww]}\n" ++
  s!"    {o} = stablehlo.add {o}c, {o}b : {ty [B,oc,Hh,Ww]}\n"

private def convBack (o dh w : String) (icc oc Hh Ww : Nat) : String :=
  s!"    {o}t = stablehlo.transpose {w}, dims = [1, 0, 2, 3] : ({ty [oc,icc,KH,KW]}) -> {ty [icc,oc,KH,KW]}\n" ++
  s!"    {o}r = stablehlo.reverse {o}t, dims = [2, 3] : {ty [icc,oc,KH,KW]}\n" ++
  s!"    {o} = stablehlo.convolution({dh}, {o}r)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [B,oc,Hh,Ww]}, {ty [icc,oc,KH,KW]}) -> {ty [B,icc,Hh,Ww]}\n"

private def convWGrad (o inp grad : String) (icc oc Hh Ww : Nat) : String :=
  s!"    {o}xt = stablehlo.transpose {inp}, dims = [1, 0, 2, 3] : ({ty [B,icc,Hh,Ww]}) -> {ty [icc,B,Hh,Ww]}\n" ++
  s!"    {o}dt = stablehlo.transpose {grad}, dims = [1, 0, 2, 3] : ({ty [B,oc,Hh,Ww]}) -> {ty [oc,B,Hh,Ww]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = " ++ "{" ++ s!"stride = [1, 1], pad = [[{pH}, {pH}], [{pW}, {pW}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [icc,B,Hh,Ww]}, {ty [oc,B,Hh,Ww]}) -> {ty [icc,oc,KH,KW]}\n" ++
  s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [icc,oc,KH,KW]}) -> {ty [oc,icc,KH,KW]}\n"

private def convBiasGrad (o dh : String) (oc Hh Ww : Nat) : String :=
  s!"    {o} = stablehlo.reduce({dh} init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [B,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

private def maxpoolFwd (o a : String) (C Hh Ww : Nat) : String :=
  s!"    {o}ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>\n" ++
  s!"    {o} = \"stablehlo.reduce_window\"({a}, {o}ninf) (" ++ "{\n" ++
  "      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):\n" ++
  "        %pm = stablehlo.maximum %pa, %pb : tensor<f32>\n" ++
  "        stablehlo.return %pm : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({ty [B,C,Hh,Ww]}, tensor<f32>) -> {ty [B,C,Hh/2,Ww/2]}\n"

private def scatter (o src dgrad : String) (C Hh Ww : Nat) : String :=
  s!"    {o} = \"stablehlo.select_and_scatter\"({src}, {dgrad}, %sc) (" ++ "{\n" ++
  "      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):\n" ++
  "        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
  "        stablehlo.return %sge : tensor<i1>\n" ++
  "    }, " ++ "{\n" ++
  "      ^bb0(%su: tensor<f32>, %sv: tensor<f32>):\n" ++
  "        %ss = stablehlo.add %su, %sv : tensor<f32>\n" ++
  "        stablehlo.return %ss : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({ty [B,C,Hh,Ww]}, {ty [B,C,Hh/2,Ww/2]}, tensor<f32>) -> {ty [B,C,Hh,Ww]}\n"

-- ════════════ the shared forward+backward+grad body (no BN) ════════════

/-- Forward (conv→relu)×2→pool ×4 → flatten → (dense→relu)×2 → dense, mean softmax-CE
    cotangent `%dy` (divided by `B`), `%loss`, then the full reverse pass producing every
    param gradient `%dW1..%dW8`, `%db1..%db8`, `%dW9/%dWa/%dWb`, `%db9/%dba/%dbb`. Conv-bias
    *params* are `%cb1..%cb8` (collision-free); their grads stay `%db1..%db8`. -/
private def cifar8AdamBody : String :=
  let H := IMH; let W := IMW
  let H2 := H / 2;  let W2 := W / 2
  let H3 := H2 / 2; let W3 := W2 / 2
  let H4 := H3 / 2; let W4 := W3 / 2
  let Hp := H4 / 2; let Wp := W4 / 2
  let flat := C4 * Hp * Wp
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→relu)×2→pool ×4 →flatten→(dense→relu)×2→dense ──\n" ++
  s!"    %xr = stablehlo.reshape %x : ({ty [B,IC*IMH*IMW]}) -> {ty [B,IC,H,W]}\n" ++
  convFwd "%hc1" "%xr" "%W1" "%cb1" C1 IC H W ++ relu4 "%ac1" "%hc1" C1 H W ++
  convFwd "%hc2" "%ac1" "%W2" "%cb2" C1 C1 H W ++ relu4 "%ac2" "%hc2" C1 H W ++
  maxpoolFwd "%pool1" "%ac2" C1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%cb3" C2 C1 H2 W2 ++ relu4 "%ac3" "%hc3" C2 H2 W2 ++
  convFwd "%hc4" "%ac3" "%W4" "%cb4" C2 C2 H2 W2 ++ relu4 "%ac4" "%hc4" C2 H2 W2 ++
  maxpoolFwd "%pool2" "%ac4" C2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%cb5" C3 C2 H3 W3 ++ relu4 "%ac5" "%hc5" C3 H3 W3 ++
  convFwd "%hc6" "%ac5" "%W6" "%cb6" C3 C3 H3 W3 ++ relu4 "%ac6" "%hc6" C3 H3 W3 ++
  maxpoolFwd "%pool3" "%ac6" C3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%cb7" C4 C3 H4 W4 ++ relu4 "%ac7" "%hc7" C4 H4 W4 ++
  convFwd "%hc8" "%ac7" "%W8" "%cb8" C4 C4 H4 W4 ++ relu4 "%ac8" "%hc8" C4 H4 W4 ++
  maxpoolFwd "%pool4" "%ac8" C4 H4 W4 ++
  s!"    %flat = stablehlo.reshape %pool4 : ({ty [B,C4,Hp,Wp]}) -> {ty [B,flat]}\n" ++
  dense "%h9" "%flat" "%W9" "%b9" flat D1 ++ relu2 "%a9" "%h9" D1 ++
  dense "%ha" "%a9" "%Wa" "%ba" D1 D1 ++ relu2 "%aa" "%ha" D1 ++
  dense "%logits" "%aa" "%Wb" "%bb" D1 NC ++
  "    // ── mean loss cotangent dy = (softmax(logits) − onehot) / B + scalar %loss ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,NC]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,NC]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,NC]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,NC]}\n" ++
  s!"    %dyr = stablehlo.subtract %lsm, %onehot : {ty [B,NC]}\n" ++
  s!"    %bnc = stablehlo.constant dense<{B}.0> : {ty [B,NC]}\n" ++
  s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [B,NC]}\n" ++
  s!"    %llog = stablehlo.log %lsm : {ty [B,NC]}\n" ++
  s!"    %ohll = stablehlo.multiply %onehot, %llog : {ty [B,NC]}\n" ++
  s!"    %csum = stablehlo.reduce(%ohll init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,NC]}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    %cneg = stablehlo.negate %csum : tensor<f32>\n" ++
  s!"    %lbf = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
  s!"    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>\n" ++
  "    // ── backward: dense (dotOut)+relu masks → scatter → convBack, four stages ──\n" ++
  dg "%dxb" "%dy" "%Wb" "1" "1" (ty [B,NC]) (ty [D1,NC]) (ty [B,D1]) ++
  selMask2 "%dya" "%ha" "%dxb" D1 ++
  dg "%dxa" "%dya" "%Wa" "1" "1" (ty [B,D1]) (ty [D1,D1]) (ty [B,D1]) ++
  selMask2 "%dy9" "%h9" "%dxa" D1 ++
  dg "%dx9" "%dy9" "%W9" "1" "1" (ty [B,D1]) (ty [flat,D1]) (ty [B,flat]) ++
  s!"    %dpool4 = stablehlo.reshape %dx9 : ({ty [B,flat]}) -> {ty [B,C4,Hp,Wp]}\n" ++
  scatter "%dac8" "%ac8" "%dpool4" C4 H4 W4 ++
  selMask4 "%dhc8" "%hc8" "%dac8" C4 H4 W4 ++
  convBack "%dac7" "%dhc8" "%W8" C4 C4 H4 W4 ++
  selMask4 "%dhc7" "%hc7" "%dac7" C4 H4 W4 ++
  convBack "%dpool3" "%dhc7" "%W7" C3 C4 H4 W4 ++
  scatter "%dac6" "%ac6" "%dpool3" C3 H3 W3 ++
  selMask4 "%dhc6" "%hc6" "%dac6" C3 H3 W3 ++
  convBack "%dac5" "%dhc6" "%W6" C3 C3 H3 W3 ++
  selMask4 "%dhc5" "%hc5" "%dac5" C3 H3 W3 ++
  convBack "%dpool2" "%dhc5" "%W5" C2 C3 H3 W3 ++
  scatter "%dac4" "%ac4" "%dpool2" C2 H2 W2 ++
  selMask4 "%dhc4" "%hc4" "%dac4" C2 H2 W2 ++
  convBack "%dac3" "%dhc4" "%W4" C2 C2 H2 W2 ++
  selMask4 "%dhc3" "%hc3" "%dac3" C2 H2 W2 ++
  convBack "%dpool1" "%dhc3" "%W3" C1 C2 H2 W2 ++
  scatter "%dac2" "%ac2" "%dpool1" C1 H W ++
  selMask4 "%dhc2" "%hc2" "%dac2" C1 H W ++
  convBack "%dac1" "%dhc2" "%W2" C1 C1 H W ++
  selMask4 "%dhc1" "%hc1" "%dac1" C1 H W ++
  "    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dWb" "%aa" "%dy" "0" "0" (ty [B,D1]) (ty [B,NC]) (ty [D1,NC]) ++ reduce0 "%dbb" "%dy" NC ++
  dg "%dWa" "%a9" "%dya" "0" "0" (ty [B,D1]) (ty [B,D1]) (ty [D1,D1]) ++ reduce0 "%dba" "%dya" D1 ++
  dg "%dW9" "%flat" "%dy9" "0" "0" (ty [B,flat]) (ty [B,D1]) (ty [flat,D1]) ++ reduce0 "%db9" "%dy9" D1 ++
  convWGrad "%dW8" "%ac7" "%dhc8" C4 C4 H4 W4 ++ convBiasGrad "%db8" "%dhc8" C4 H4 W4 ++
  convWGrad "%dW7" "%pool3" "%dhc7" C3 C4 H4 W4 ++ convBiasGrad "%db7" "%dhc7" C4 H4 W4 ++
  convWGrad "%dW6" "%ac5" "%dhc6" C3 C3 H3 W3 ++ convBiasGrad "%db6" "%dhc6" C3 H3 W3 ++
  convWGrad "%dW5" "%pool2" "%dhc5" C2 C3 H3 W3 ++ convBiasGrad "%db5" "%dhc5" C3 H3 W3 ++
  convWGrad "%dW4" "%ac3" "%dhc4" C2 C2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" C2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" C1 C2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" C2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" C1 C1 H W ++ convBiasGrad "%db2" "%dhc2" C1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" IC C1 H W ++ convBiasGrad "%db1" "%dhc1" C1 H W

-- ════════════ no-BN Adam train step ════════════

/-- The 22 cifar8 params: `(SSA-name-without-%, gradient-SSA, dims)`, in func-arg order
    (= `cifar8Verified.toSpecs`). Conv biases are `%cb1..%cb8` (collision-free). -/
private def params : List (String × String × List Nat) :=
  [("W1","%dW1",[C1,IC,KH,KW]), ("cb1","%db1",[C1]),
   ("W2","%dW2",[C1,C1,KH,KW]), ("cb2","%db2",[C1]),
   ("W3","%dW3",[C2,C1,KH,KW]), ("cb3","%db3",[C2]),
   ("W4","%dW4",[C2,C2,KH,KW]), ("cb4","%db4",[C2]),
   ("W5","%dW5",[C3,C2,KH,KW]), ("cb5","%db5",[C3]),
   ("W6","%dW6",[C3,C3,KH,KW]), ("cb6","%db6",[C3]),
   ("W7","%dW7",[C4,C3,KH,KW]), ("cb7","%db7",[C4]),
   ("W8","%dW8",[C4,C4,KH,KW]), ("cb8","%db8",[C4]),
   ("W9","%dW9",[C4*(IMH/16)*(IMW/16),D1]), ("b9","%db9",[D1]),
   ("Wa","%dWa",[D1,D1]), ("ba","%dba",[D1]),
   ("Wb","%dWb",[D1,NC]), ("bb","%dbb",[NC])]

/-- `@cifar8w_adam_train_step` — the no-BN body + per-param `emitAdamV`, packed `[θ|m|v]` +
    `%lr`/`%bc1`/`%bc2` signature, returning `[θ'|m'|v'|loss|bc1|bc2]`. -/
private def cifar8AdamTrainStep : String :=
  let updParts := params.map (fun (nm, gr, ds) =>
    ViTRender.emitAdamV ("%" ++ nm) gr ("%" ++ nm ++ "m") ("%" ++ nm ++ "v") ds nm)
  let upd := String.join (updParts.map (·.1))
  let thetaN := updParts.map (·.2.1)
  let mN := updParts.map (·.2.2.1)
  let vN := updParts.map (·.2.2.2)
  let psig := String.intercalate ", " (params.map (fun (nm, _, ds) => s!"%{nm}: {ty ds}"))
  let msig := String.intercalate ", " (params.map (fun (nm, _, ds) => s!"%{nm}m: {ty ds}"))
  let vsig := String.intercalate ", " (params.map (fun (nm, _, ds) => s!"%{nm}v: {ty ds}"))
  let dims := params.map (fun (_, _, ds) => ds)
  let allDims := dims ++ dims ++ dims
  let retTy := String.intercalate ", " ((allDims.map (fun ds => ty ds)) ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"])
  let retVals := String.intercalate ", " (thetaN ++ mN ++ vN ++ ["%loss", "%bc1", "%bc2"])
  let argSig := s!"%x: {ty [B,IC*IMH*IMW]}, " ++ psig ++ ", " ++ msig ++ ", " ++ vsig ++
    s!", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [B,NC]}"
  "module @m {\n" ++ s!"  func.func @cifar8w_adam_train_step({argSig}) -> ({retTy}) " ++ "{\n" ++
    cifar8AdamBody ++ adamConsts ++ upd ++ s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

-- ════════════ BN variant (per-channel BatchNorm) ════════════

private def epsStr : String := "1.0e-05"

private def rs (o src : String) (dimsFrom dimsTo : List Nat) : String :=
  s!"    {o} = stablehlo.reshape {src} : ({ty dimsFrom}) -> {ty dimsTo}\n"

/-- Per-channel (per-example) BN forward over `[B,C,S]`; saves `_xhat/_nf/_istd` for the
    backward. `g`=γ:[C], `bt`=β:[C]. Result `%{o}` flat `[B,C·S]`. -/
private def bnFwd (o x g bt : String) (C S : Nat) : String :=
  let Mn := C * S
  s!"    {o}_xr = stablehlo.reshape {x} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
  s!"    {o}_nf = stablehlo.constant dense<{S}.0> : {ty [B,C,S]}\n" ++
  s!"    {o}_ep = stablehlo.constant dense<{epsStr}> : {ty [B,C,S]}\n" ++
  s!"    {o}_smr = stablehlo.reduce({o}_xr init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
  s!"    {o}_sm = stablehlo.broadcast_in_dim {o}_smr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
  s!"    {o}_mu = stablehlo.divide {o}_sm, {o}_nf : {ty [B,C,S]}\n" ++
  s!"    {o}_xc = stablehlo.subtract {o}_xr, {o}_mu : {ty [B,C,S]}\n" ++
  s!"    {o}_sq = stablehlo.multiply {o}_xc, {o}_xc : {ty [B,C,S]}\n" ++
  s!"    {o}_vsr = stablehlo.reduce({o}_sq init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
  s!"    {o}_vs = stablehlo.broadcast_in_dim {o}_vsr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
  s!"    {o}_var = stablehlo.divide {o}_vs, {o}_nf : {ty [B,C,S]}\n" ++
  s!"    {o}_ve = stablehlo.add {o}_var, {o}_ep : {ty [B,C,S]}\n" ++
  s!"    {o}_istd = stablehlo.rsqrt {o}_ve : {ty [B,C,S]}\n" ++
  s!"    {o}_xhat = stablehlo.multiply {o}_xc, {o}_istd : {ty [B,C,S]}\n" ++
  s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
  s!"    {o}_bb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
  s!"    {o}_gx = stablehlo.multiply {o}_xhat, {o}_gb : {ty [B,C,S]}\n" ++
  s!"    {o}_y3 = stablehlo.add {o}_gx, {o}_bb : {ty [B,C,S]}\n" ++
  s!"    {o} = stablehlo.reshape {o}_y3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"

private def bnBack (o bn g dyf : String) (C S : Nat) : String :=
  let Mn := C * S
  s!"    {o}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
  s!"    {o}_gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [C]}) -> {ty [B,C,S]}\n" ++
  s!"    {o}_dxh = stablehlo.multiply {o}_gb, {o}_dyr : {ty [B,C,S]}\n" ++
  s!"    {o}_sdxr = stablehlo.reduce({o}_dxh init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
  s!"    {o}_sdx = stablehlo.broadcast_in_dim {o}_sdxr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
  s!"    {o}_xd = stablehlo.multiply {bn}_xhat, {o}_dxh : {ty [B,C,S]}\n" ++
  s!"    {o}_sxdr = stablehlo.reduce({o}_xd init: %sc) applies stablehlo.add across dimensions = [2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [B,C]}\n" ++
  s!"    {o}_sxd = stablehlo.broadcast_in_dim {o}_sxdr, dims = [0, 1] : ({ty [B,C]}) -> {ty [B,C,S]}\n" ++
  s!"    {o}_t1 = stablehlo.multiply {o}_dxh, {bn}_nf : {ty [B,C,S]}\n" ++
  s!"    {o}_i1 = stablehlo.subtract {o}_t1, {o}_sdx : {ty [B,C,S]}\n" ++
  s!"    {o}_xs = stablehlo.multiply {bn}_xhat, {o}_sxd : {ty [B,C,S]}\n" ++
  s!"    {o}_i2 = stablehlo.subtract {o}_i1, {o}_xs : {ty [B,C,S]}\n" ++
  s!"    {o}_s = stablehlo.divide {bn}_istd, {bn}_nf : {ty [B,C,S]}\n" ++
  s!"    {o}_dx3 = stablehlo.multiply {o}_s, {o}_i2 : {ty [B,C,S]}\n" ++
  s!"    {o} = stablehlo.reshape {o}_dx3 : ({ty [B,C,S]}) -> {ty [B,Mn]}\n"

private def bnParamGrad (dgr dbe bn dyf : String) (C S : Nat) : String :=
  let Mn := C * S
  s!"    {dgr}_dyr = stablehlo.reshape {dyf} : ({ty [B,Mn]}) -> {ty [B,C,S]}\n" ++
  s!"    {dgr}_p = stablehlo.multiply {dgr}_dyr, {bn}_xhat : {ty [B,C,S]}\n" ++
  s!"    {dgr} = stablehlo.reduce({dgr}_p init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n" ++
  s!"    {dbe} = stablehlo.reduce({dgr}_dyr init: %sc) applies stablehlo.add across dimensions = [0, 2] : ({ty [B,C,S]}, tensor<f32>) -> {ty [C]}\n"

/-- BN forward+backward+grad body (conv→BN→relu)×2→pool ×4, mean cotangent + `%loss`. Conv
    biases `%cb1..%cb8` (collision-free); BN γ `%g*`, β `%bt*`; grads `%dW*/%db*/%dg*/%dbt*`. -/
private def cifar8BnAdamBody : String :=
  let H := IMH; let W := IMW
  let H2 := H / 2;  let W2 := W / 2
  let H3 := H2 / 2; let W3 := W2 / 2
  let H4 := H3 / 2; let W4 := W3 / 2
  let Hp := H4 / 2; let Wp := W4 / 2
  let flat := C4 * Hp * Wp
  let M1 := C1 * H * W;   let S1 := H * W
  let M2 := C2 * H2 * W2; let S2 := H2 * W2
  let M3 := C3 * H3 * W3; let S3 := H3 * W3
  let M4 := C4 * H4 * W4; let S4 := H4 * W4
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    // ── forward: (conv→BN→relu)×2→pool ×4 →flatten→(dense→relu)×2→dense ──\n" ++
  rs "%xr" "%x" [B,IC*IMH*IMW] [B,IC,H,W] ++
  convFwd "%hc1" "%xr" "%W1" "%cb1" C1 IC H W ++ rs "%hc1f" "%hc1" [B,C1,H,W] [B,M1] ++
  bnFwd "%bn1" "%hc1f" "%g1" "%bt1" C1 S1 ++ relu2 "%ac1f" "%bn1" M1 ++ rs "%ac1" "%ac1f" [B,M1] [B,C1,H,W] ++
  convFwd "%hc2" "%ac1" "%W2" "%cb2" C1 C1 H W ++ rs "%hc2f" "%hc2" [B,C1,H,W] [B,M1] ++
  bnFwd "%bn2" "%hc2f" "%g2" "%bt2" C1 S1 ++ relu2 "%ac2f" "%bn2" M1 ++ rs "%ac2" "%ac2f" [B,M1] [B,C1,H,W] ++
  maxpoolFwd "%pool1" "%ac2" C1 H W ++
  convFwd "%hc3" "%pool1" "%W3" "%cb3" C2 C1 H2 W2 ++ rs "%hc3f" "%hc3" [B,C2,H2,W2] [B,M2] ++
  bnFwd "%bn3" "%hc3f" "%g3" "%bt3" C2 S2 ++ relu2 "%ac3f" "%bn3" M2 ++ rs "%ac3" "%ac3f" [B,M2] [B,C2,H2,W2] ++
  convFwd "%hc4" "%ac3" "%W4" "%cb4" C2 C2 H2 W2 ++ rs "%hc4f" "%hc4" [B,C2,H2,W2] [B,M2] ++
  bnFwd "%bn4" "%hc4f" "%g4" "%bt4" C2 S2 ++ relu2 "%ac4f" "%bn4" M2 ++ rs "%ac4" "%ac4f" [B,M2] [B,C2,H2,W2] ++
  maxpoolFwd "%pool2" "%ac4" C2 H2 W2 ++
  convFwd "%hc5" "%pool2" "%W5" "%cb5" C3 C2 H3 W3 ++ rs "%hc5f" "%hc5" [B,C3,H3,W3] [B,M3] ++
  bnFwd "%bn5" "%hc5f" "%g5" "%bt5" C3 S3 ++ relu2 "%ac5f" "%bn5" M3 ++ rs "%ac5" "%ac5f" [B,M3] [B,C3,H3,W3] ++
  convFwd "%hc6" "%ac5" "%W6" "%cb6" C3 C3 H3 W3 ++ rs "%hc6f" "%hc6" [B,C3,H3,W3] [B,M3] ++
  bnFwd "%bn6" "%hc6f" "%g6" "%bt6" C3 S3 ++ relu2 "%ac6f" "%bn6" M3 ++ rs "%ac6" "%ac6f" [B,M3] [B,C3,H3,W3] ++
  maxpoolFwd "%pool3" "%ac6" C3 H3 W3 ++
  convFwd "%hc7" "%pool3" "%W7" "%cb7" C4 C3 H4 W4 ++ rs "%hc7f" "%hc7" [B,C4,H4,W4] [B,M4] ++
  bnFwd "%bn7" "%hc7f" "%g7" "%bt7" C4 S4 ++ relu2 "%ac7f" "%bn7" M4 ++ rs "%ac7" "%ac7f" [B,M4] [B,C4,H4,W4] ++
  convFwd "%hc8" "%ac7" "%W8" "%cb8" C4 C4 H4 W4 ++ rs "%hc8f" "%hc8" [B,C4,H4,W4] [B,M4] ++
  bnFwd "%bn8" "%hc8f" "%g8" "%bt8" C4 S4 ++ relu2 "%ac8f" "%bn8" M4 ++ rs "%ac8" "%ac8f" [B,M4] [B,C4,H4,W4] ++
  maxpoolFwd "%pool4" "%ac8" C4 H4 W4 ++
  rs "%flat" "%pool4" [B,C4,Hp,Wp] [B,flat] ++
  dense "%h9" "%flat" "%W9" "%b9" flat D1 ++ relu2 "%a9" "%h9" D1 ++
  dense "%ha" "%a9" "%Wa" "%ba" D1 D1 ++ relu2 "%aa" "%ha" D1 ++
  dense "%logits" "%aa" "%Wb" "%bb" D1 NC ++
  "    // ── mean loss cotangent dy = (softmax(logits) − onehot) / B + scalar %loss ──\n" ++
  s!"    %le = stablehlo.exponential %logits : {ty [B,NC]}\n" ++
  s!"    %lsum = stablehlo.reduce(%le init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [B,NC]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,NC]}\n" ++
  s!"    %lsm = stablehlo.divide %le, %lsb : {ty [B,NC]}\n" ++
  s!"    %dyr = stablehlo.subtract %lsm, %onehot : {ty [B,NC]}\n" ++
  s!"    %bnc = stablehlo.constant dense<{B}.0> : {ty [B,NC]}\n" ++
  s!"    %dy = stablehlo.divide %dyr, %bnc : {ty [B,NC]}\n" ++
  s!"    %llog = stablehlo.log %lsm : {ty [B,NC]}\n" ++
  s!"    %ohll = stablehlo.multiply %onehot, %llog : {ty [B,NC]}\n" ++
  s!"    %csum = stablehlo.reduce(%ohll init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,NC]}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    %cneg = stablehlo.negate %csum : tensor<f32>\n" ++
  s!"    %lbf = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
  s!"    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>\n" ++
  "    // ── backward: dense+relu → scatter → (relu→BN-back→convBack)×stage, four stages ──\n" ++
  dg "%dxb" "%dy" "%Wb" "1" "1" (ty [B,NC]) (ty [D1,NC]) (ty [B,D1]) ++
  selMask2 "%dya" "%ha" "%dxb" D1 ++
  dg "%dxa" "%dya" "%Wa" "1" "1" (ty [B,D1]) (ty [D1,D1]) (ty [B,D1]) ++
  selMask2 "%dy9" "%h9" "%dxa" D1 ++
  dg "%dx9" "%dy9" "%W9" "1" "1" (ty [B,D1]) (ty [flat,D1]) (ty [B,flat]) ++
  rs "%dpool4" "%dx9" [B,flat] [B,C4,Hp,Wp] ++
  scatter "%dac8" "%ac8" "%dpool4" C4 H4 W4 ++ rs "%dac8f" "%dac8" [B,C4,H4,W4] [B,M4] ++
  selMask2 "%dbn8" "%bn8" "%dac8f" M4 ++
  bnBack "%dhc8f" "%bn8" "%g8" "%dbn8" C4 S4 ++ bnParamGrad "%dg8" "%dbt8" "%bn8" "%dbn8" C4 S4 ++
  rs "%dhc8" "%dhc8f" [B,M4] [B,C4,H4,W4] ++
  convBack "%dac7" "%dhc8" "%W8" C4 C4 H4 W4 ++ rs "%dac7f" "%dac7" [B,C4,H4,W4] [B,M4] ++
  selMask2 "%dbn7" "%bn7" "%dac7f" M4 ++
  bnBack "%dhc7f" "%bn7" "%g7" "%dbn7" C4 S4 ++ bnParamGrad "%dg7" "%dbt7" "%bn7" "%dbn7" C4 S4 ++
  rs "%dhc7" "%dhc7f" [B,M4] [B,C4,H4,W4] ++
  convBack "%dpool3" "%dhc7" "%W7" C3 C4 H4 W4 ++
  scatter "%dac6" "%ac6" "%dpool3" C3 H3 W3 ++ rs "%dac6f" "%dac6" [B,C3,H3,W3] [B,M3] ++
  selMask2 "%dbn6" "%bn6" "%dac6f" M3 ++
  bnBack "%dhc6f" "%bn6" "%g6" "%dbn6" C3 S3 ++ bnParamGrad "%dg6" "%dbt6" "%bn6" "%dbn6" C3 S3 ++
  rs "%dhc6" "%dhc6f" [B,M3] [B,C3,H3,W3] ++
  convBack "%dac5" "%dhc6" "%W6" C3 C3 H3 W3 ++ rs "%dac5f" "%dac5" [B,C3,H3,W3] [B,M3] ++
  selMask2 "%dbn5" "%bn5" "%dac5f" M3 ++
  bnBack "%dhc5f" "%bn5" "%g5" "%dbn5" C3 S3 ++ bnParamGrad "%dg5" "%dbt5" "%bn5" "%dbn5" C3 S3 ++
  rs "%dhc5" "%dhc5f" [B,M3] [B,C3,H3,W3] ++
  convBack "%dpool2" "%dhc5" "%W5" C2 C3 H3 W3 ++
  scatter "%dac4" "%ac4" "%dpool2" C2 H2 W2 ++ rs "%dac4f" "%dac4" [B,C2,H2,W2] [B,M2] ++
  selMask2 "%dbn4" "%bn4" "%dac4f" M2 ++
  bnBack "%dhc4f" "%bn4" "%g4" "%dbn4" C2 S2 ++ bnParamGrad "%dg4" "%dbt4" "%bn4" "%dbn4" C2 S2 ++
  rs "%dhc4" "%dhc4f" [B,M2] [B,C2,H2,W2] ++
  convBack "%dac3" "%dhc4" "%W4" C2 C2 H2 W2 ++ rs "%dac3f" "%dac3" [B,C2,H2,W2] [B,M2] ++
  selMask2 "%dbn3" "%bn3" "%dac3f" M2 ++
  bnBack "%dhc3f" "%bn3" "%g3" "%dbn3" C2 S2 ++ bnParamGrad "%dg3" "%dbt3" "%bn3" "%dbn3" C2 S2 ++
  rs "%dhc3" "%dhc3f" [B,M2] [B,C2,H2,W2] ++
  convBack "%dpool1" "%dhc3" "%W3" C1 C2 H2 W2 ++
  scatter "%dac2" "%ac2" "%dpool1" C1 H W ++ rs "%dac2f" "%dac2" [B,C1,H,W] [B,M1] ++
  selMask2 "%dbn2" "%bn2" "%dac2f" M1 ++
  bnBack "%dhc2f" "%bn2" "%g2" "%dbn2" C1 S1 ++ bnParamGrad "%dg2" "%dbt2" "%bn2" "%dbn2" C1 S1 ++
  rs "%dhc2" "%dhc2f" [B,M1] [B,C1,H,W] ++
  convBack "%dac1" "%dhc2" "%W2" C1 C1 H W ++ rs "%dac1f" "%dac1" [B,C1,H,W] [B,M1] ++
  selMask2 "%dbn1" "%bn1" "%dac1f" M1 ++
  bnBack "%dhc1f" "%bn1" "%g1" "%dbn1" C1 S1 ++ bnParamGrad "%dg1" "%dbt1" "%bn1" "%dbn1" C1 S1 ++
  rs "%dhc1" "%dhc1f" [B,M1] [B,C1,H,W] ++
  "    // ── param grads: dense W/b; conv dW (transpose trick), db (reduce) ──\n" ++
  dg "%dWb" "%aa" "%dy" "0" "0" (ty [B,D1]) (ty [B,NC]) (ty [D1,NC]) ++ reduce0 "%dbb" "%dy" NC ++
  dg "%dWa" "%a9" "%dya" "0" "0" (ty [B,D1]) (ty [B,D1]) (ty [D1,D1]) ++ reduce0 "%dba" "%dya" D1 ++
  dg "%dW9" "%flat" "%dy9" "0" "0" (ty [B,flat]) (ty [B,D1]) (ty [flat,D1]) ++ reduce0 "%db9" "%dy9" D1 ++
  convWGrad "%dW8" "%ac7" "%dhc8" C4 C4 H4 W4 ++ convBiasGrad "%db8" "%dhc8" C4 H4 W4 ++
  convWGrad "%dW7" "%pool3" "%dhc7" C3 C4 H4 W4 ++ convBiasGrad "%db7" "%dhc7" C4 H4 W4 ++
  convWGrad "%dW6" "%ac5" "%dhc6" C3 C3 H3 W3 ++ convBiasGrad "%db6" "%dhc6" C3 H3 W3 ++
  convWGrad "%dW5" "%pool2" "%dhc5" C2 C3 H3 W3 ++ convBiasGrad "%db5" "%dhc5" C3 H3 W3 ++
  convWGrad "%dW4" "%ac3" "%dhc4" C2 C2 H2 W2 ++ convBiasGrad "%db4" "%dhc4" C2 H2 W2 ++
  convWGrad "%dW3" "%pool1" "%dhc3" C1 C2 H2 W2 ++ convBiasGrad "%db3" "%dhc3" C2 H2 W2 ++
  convWGrad "%dW2" "%ac1" "%dhc2" C1 C1 H W ++ convBiasGrad "%db2" "%dhc2" C1 H W ++
  convWGrad "%dW1" "%xr" "%dhc1" IC C1 H W ++ convBiasGrad "%db1" "%dhc1" C1 H W

/-- The 38 cifar8-BN params `(name, grad, dims)` in func-arg order (= `cifar8BnVerified.toSpecs`):
    per conv layer i = `(Wi, cbi, gi, bti)`, then the 3 dense `(W,b)`. -/
private def paramsBn : List (String × String × List Nat) :=
  [("W1","%dW1",[C1,IC,KH,KW]), ("cb1","%db1",[C1]), ("g1","%dg1",[C1]), ("bt1","%dbt1",[C1]),
   ("W2","%dW2",[C1,C1,KH,KW]), ("cb2","%db2",[C1]), ("g2","%dg2",[C1]), ("bt2","%dbt2",[C1]),
   ("W3","%dW3",[C2,C1,KH,KW]), ("cb3","%db3",[C2]), ("g3","%dg3",[C2]), ("bt3","%dbt3",[C2]),
   ("W4","%dW4",[C2,C2,KH,KW]), ("cb4","%db4",[C2]), ("g4","%dg4",[C2]), ("bt4","%dbt4",[C2]),
   ("W5","%dW5",[C3,C2,KH,KW]), ("cb5","%db5",[C3]), ("g5","%dg5",[C3]), ("bt5","%dbt5",[C3]),
   ("W6","%dW6",[C3,C3,KH,KW]), ("cb6","%db6",[C3]), ("g6","%dg6",[C3]), ("bt6","%dbt6",[C3]),
   ("W7","%dW7",[C4,C3,KH,KW]), ("cb7","%db7",[C4]), ("g7","%dg7",[C4]), ("bt7","%dbt7",[C4]),
   ("W8","%dW8",[C4,C4,KH,KW]), ("cb8","%db8",[C4]), ("g8","%dg8",[C4]), ("bt8","%dbt8",[C4]),
   ("W9","%dW9",[C4*(IMH/16)*(IMW/16),D1]), ("b9","%db9",[D1]),
   ("Wa","%dWa",[D1,D1]), ("ba","%dba",[D1]),
   ("Wb","%dWb",[D1,NC]), ("bb","%dbb",[NC])]

/-- `@cifar8w_bn_adam_train_step` — the BN body + per-param `emitAdamV`, packed `[θ|m|v]`. -/
private def cifar8BnAdamTrainStep : String :=
  let updParts := paramsBn.map (fun (nm, gr, ds) =>
    ViTRender.emitAdamV ("%" ++ nm) gr ("%" ++ nm ++ "m") ("%" ++ nm ++ "v") ds nm)
  let upd := String.join (updParts.map (·.1))
  let thetaN := updParts.map (·.2.1)
  let mN := updParts.map (·.2.2.1)
  let vN := updParts.map (·.2.2.2)
  let psig := String.intercalate ", " (paramsBn.map (fun (nm, _, ds) => s!"%{nm}: {ty ds}"))
  let msig := String.intercalate ", " (paramsBn.map (fun (nm, _, ds) => s!"%{nm}m: {ty ds}"))
  let vsig := String.intercalate ", " (paramsBn.map (fun (nm, _, ds) => s!"%{nm}v: {ty ds}"))
  let dims := paramsBn.map (fun (_, _, ds) => ds)
  let allDims := dims ++ dims ++ dims
  let retTy := String.intercalate ", " ((allDims.map (fun ds => ty ds)) ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"])
  let retVals := String.intercalate ", " (thetaN ++ mN ++ vN ++ ["%loss", "%bc1", "%bc2"])
  let argSig := s!"%x: {ty [B,IC*IMH*IMW]}, " ++ psig ++ ", " ++ msig ++ ", " ++ vsig ++
    s!", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [B,NC]}"
  "module @m {\n" ++ s!"  func.func @cifar8w_bn_adam_train_step({argSig}) -> ({retTy}) " ++ "{\n" ++
    cifar8BnAdamBody ++ adamConsts ++ upd ++ s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

-- ════════════ Nesterov-momentum SGD variant (same body, momentum update) ════════════

/-- μ baked (heavy-ball/Nesterov coefficient 0.9); `%lr` arrives as a runtime arg (cosine+warmup
    schedule, shared with the Adam driver). No weight decay (clean "add momentum to plain SGD"). -/
private def momentumConsts : String :=
  "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n"

/-- **Nesterov-momentum SGD update for one parameter.** `v' = μ·v + g`,
    `θ' = θ − lr·(μ·v' + g)`. Reads `%mu` (baked) and `%lr` (runtime). Returns
    `(ir, θ'SSA, m'SSA, v'SSA)` — the `m` slot is **passed through unchanged** (momentum needs
    one buffer, but we keep the `[θ|m|v]` packing so the Adam driver is reused verbatim). The
    Nesterov look-ahead is the `μ·v' + g` term; drop it (use `%momvel`) for classic heavy-ball. -/
private def emitMomentum (θ g m v : String) (ds : List Nat) (t : String) : String × String × String × String :=
  let T := ty ds
  let s :=
    s!"    %mommu{t} = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %momvg{t} = stablehlo.multiply %mommu{t}, {v} : {T}\n" ++
    s!"    %momvel{t} = stablehlo.add %momvg{t}, {g} : {T}\n" ++
    s!"    %momnv{t} = stablehlo.multiply %mommu{t}, %momvel{t} : {T}\n" ++
    s!"    %momlk{t} = stablehlo.add %momnv{t}, {g} : {T}\n" ++
    s!"    %momlr{t} = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %momst{t} = stablehlo.multiply %momlr{t}, %momlk{t} : {T}\n" ++
    s!"    %momnew{t} = stablehlo.subtract {θ}, %momst{t} : {T}\n"
  (s, s!"%momnew{t}", m, s!"%momvel{t}")

/-- `@cifar8w_mom_train_step` — the no-BN body + per-param `emitMomentum`, same packed
    `[θ|m|v]`+`lr`/`bc1`/`bc2` signature as the Adam step (the m/bc slots are passthrough). -/
private def cifar8MomTrainStep : String :=
  let updParts := params.map (fun (nm, gr, ds) =>
    emitMomentum ("%" ++ nm) gr ("%" ++ nm ++ "m") ("%" ++ nm ++ "v") ds nm)
  let upd := String.join (updParts.map (·.1))
  let thetaN := updParts.map (·.2.1)
  let mN := updParts.map (·.2.2.1)
  let vN := updParts.map (·.2.2.2)
  let psig := String.intercalate ", " (params.map (fun (nm, _, ds) => s!"%{nm}: {ty ds}"))
  let msig := String.intercalate ", " (params.map (fun (nm, _, ds) => s!"%{nm}m: {ty ds}"))
  let vsig := String.intercalate ", " (params.map (fun (nm, _, ds) => s!"%{nm}v: {ty ds}"))
  let dims := params.map (fun (_, _, ds) => ds)
  let allDims := dims ++ dims ++ dims
  let retTy := String.intercalate ", " ((allDims.map (fun ds => ty ds)) ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"])
  let retVals := String.intercalate ", " (thetaN ++ mN ++ vN ++ ["%loss", "%bc1", "%bc2"])
  let argSig := s!"%x: {ty [B,IC*IMH*IMW]}, " ++ psig ++ ", " ++ msig ++ ", " ++ vsig ++
    s!", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [B,NC]}"
  "module @m {\n" ++ s!"  func.func @cifar8w_mom_train_step({argSig}) -> ({retTy}) " ++ "{\n" ++
    cifar8AdamBody ++ momentumConsts ++ upd ++ s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

/-- `@cifar8w_bn_mom_train_step` — the BN body + per-param `emitMomentum`. -/
private def cifar8BnMomTrainStep : String :=
  let updParts := paramsBn.map (fun (nm, gr, ds) =>
    emitMomentum ("%" ++ nm) gr ("%" ++ nm ++ "m") ("%" ++ nm ++ "v") ds nm)
  let upd := String.join (updParts.map (·.1))
  let thetaN := updParts.map (·.2.1)
  let mN := updParts.map (·.2.2.1)
  let vN := updParts.map (·.2.2.2)
  let psig := String.intercalate ", " (paramsBn.map (fun (nm, _, ds) => s!"%{nm}: {ty ds}"))
  let msig := String.intercalate ", " (paramsBn.map (fun (nm, _, ds) => s!"%{nm}m: {ty ds}"))
  let vsig := String.intercalate ", " (paramsBn.map (fun (nm, _, ds) => s!"%{nm}v: {ty ds}"))
  let dims := paramsBn.map (fun (_, _, ds) => ds)
  let allDims := dims ++ dims ++ dims
  let retTy := String.intercalate ", " ((allDims.map (fun ds => ty ds)) ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"])
  let retVals := String.intercalate ", " (thetaN ++ mN ++ vN ++ ["%loss", "%bc1", "%bc2"])
  let argSig := s!"%x: {ty [B,IC*IMH*IMW]}, " ++ psig ++ ", " ++ msig ++ ", " ++ vsig ++
    s!", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [B,NC]}"
  "module @m {\n" ++ s!"  func.func @cifar8w_bn_mom_train_step({argSig}) -> ({retTy}) " ++ "{\n" ++
    cifar8BnAdamBody ++ momentumConsts ++ upd ++ s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

-- ════════════ plain SGD on the SAME pipeline (controlled optimizer ablation) ════════════

/-- **Plain SGD update for one parameter.** `θ' = θ − lr·g`. Reads only `%lr` (runtime). The
    `m`/`v` slots pass through unchanged — kept solely so this shares the Adam driver's `[θ|m|v]`
    packing, hence the SAME shuffle + hflip + cosine-warmup pipeline as the momentum/Adam runs.
    This makes the three-way optimizer comparison controlled (only the update rule differs). -/
private def emitSgd (θ g m v : String) (ds : List Nat) (t : String) : String × String × String × String :=
  let T := ty ds
  let s :=
    s!"    %sgdlr{t} = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> {T}\n" ++
    s!"    %sgdst{t} = stablehlo.multiply %sgdlr{t}, {g} : {T}\n" ++
    s!"    %sgdnew{t} = stablehlo.subtract {θ}, %sgdst{t} : {T}\n"
  (s, s!"%sgdnew{t}", m, v)

/-- `@cifar8w_sgd_train_step` — the no-BN body + per-param `emitSgd`, same packed signature. -/
private def cifar8SgdTrainStep : String :=
  let updParts := params.map (fun (nm, gr, ds) =>
    emitSgd ("%" ++ nm) gr ("%" ++ nm ++ "m") ("%" ++ nm ++ "v") ds nm)
  let upd := String.join (updParts.map (·.1))
  let thetaN := updParts.map (·.2.1)
  let mN := updParts.map (·.2.2.1)
  let vN := updParts.map (·.2.2.2)
  let psig := String.intercalate ", " (params.map (fun (nm, _, ds) => s!"%{nm}: {ty ds}"))
  let msig := String.intercalate ", " (params.map (fun (nm, _, ds) => s!"%{nm}m: {ty ds}"))
  let vsig := String.intercalate ", " (params.map (fun (nm, _, ds) => s!"%{nm}v: {ty ds}"))
  let dims := params.map (fun (_, _, ds) => ds)
  let allDims := dims ++ dims ++ dims
  let retTy := String.intercalate ", " ((allDims.map (fun ds => ty ds)) ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"])
  let retVals := String.intercalate ", " (thetaN ++ mN ++ vN ++ ["%loss", "%bc1", "%bc2"])
  let argSig := s!"%x: {ty [B,IC*IMH*IMW]}, " ++ psig ++ ", " ++ msig ++ ", " ++ vsig ++
    s!", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [B,NC]}"
  "module @m {\n" ++ s!"  func.func @cifar8w_sgd_train_step({argSig}) -> ({retTy}) " ++ "{\n" ++
    cifar8AdamBody ++ upd ++ s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

/-- `@cifar8w_bn_sgd_train_step` — the BN body + per-param `emitSgd`. -/
private def cifar8BnSgdTrainStep : String :=
  let updParts := paramsBn.map (fun (nm, gr, ds) =>
    emitSgd ("%" ++ nm) gr ("%" ++ nm ++ "m") ("%" ++ nm ++ "v") ds nm)
  let upd := String.join (updParts.map (·.1))
  let thetaN := updParts.map (·.2.1)
  let mN := updParts.map (·.2.2.1)
  let vN := updParts.map (·.2.2.2)
  let psig := String.intercalate ", " (paramsBn.map (fun (nm, _, ds) => s!"%{nm}: {ty ds}"))
  let msig := String.intercalate ", " (paramsBn.map (fun (nm, _, ds) => s!"%{nm}m: {ty ds}"))
  let vsig := String.intercalate ", " (paramsBn.map (fun (nm, _, ds) => s!"%{nm}v: {ty ds}"))
  let dims := paramsBn.map (fun (_, _, ds) => ds)
  let allDims := dims ++ dims ++ dims
  let retTy := String.intercalate ", " ((allDims.map (fun ds => ty ds)) ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"])
  let retVals := String.intercalate ", " (thetaN ++ mN ++ vN ++ ["%loss", "%bc1", "%bc2"])
  let argSig := s!"%x: {ty [B,IC*IMH*IMW]}, " ++ psig ++ ", " ++ msig ++ ", " ++ vsig ++
    s!", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: {ty [B,NC]}"
  "module @m {\n" ++ s!"  func.func @cifar8w_bn_sgd_train_step({argSig}) -> ({retTy}) " ++ "{\n" ++
    cifar8BnAdamBody ++ upd ++ s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

private def tryCompile (src dst label : String) : IO Unit := do
  try
    let cargs ← ireeCompileArgs src dst
    let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
    if r.exitCode != 0 then IO.eprintln s!"iree-compile ({label}) FAILED:\n{r.stderr.take 4000}"
    else IO.println s!"{label} iree-compile OK → {src}"
  catch e => IO.eprintln s!"iree-compile ({label}) skipped (compiler unavailable): {e}"

def main : IO Unit := do
  IO.FS.createDirAll "verified_mlir"
  IO.FS.createDirAll ".lake/build"
  let mlir := cifar8AdamTrainStep
  IO.println s!"rendered cifar8 AdamW train step: {mlir.length} chars, {params.length} params"
  IO.FS.writeFile "verified_mlir/cifar8w_adam_train_step.mlir" mlir
  let bmlir := cifar8BnAdamTrainStep
  IO.println s!"rendered cifar8_bn AdamW train step: {bmlir.length} chars, {paramsBn.length} params"
  IO.FS.writeFile "verified_mlir/cifar8w_bn_adam_train_step.mlir" bmlir
  let mmlir := cifar8MomTrainStep
  IO.println s!"rendered cifar8 Nesterov-mom train step: {mmlir.length} chars, {params.length} params"
  IO.FS.writeFile "verified_mlir/cifar8w_mom_train_step.mlir" mmlir
  let bmmlir := cifar8BnMomTrainStep
  IO.println s!"rendered cifar8_bn Nesterov-mom train step: {bmmlir.length} chars, {paramsBn.length} params"
  IO.FS.writeFile "verified_mlir/cifar8w_bn_mom_train_step.mlir" bmmlir
  let smlir := cifar8SgdTrainStep
  IO.println s!"rendered cifar8 SGD-sched train step: {smlir.length} chars, {params.length} params"
  IO.FS.writeFile "verified_mlir/cifar8w_sgd_train_step.mlir" smlir
  let bsmlir := cifar8BnSgdTrainStep
  IO.println s!"rendered cifar8_bn SGD-sched train step: {bsmlir.length} chars, {paramsBn.length} params"
  IO.FS.writeFile "verified_mlir/cifar8w_bn_sgd_train_step.mlir" bsmlir
  -- eval-forward graphs at d1=512 (the StableHLO renderers emit @cifar8_fwd / @cifar8_bn_fwd;
  -- rename to the cifar8w slug so trainAdamSched's `m.cifar8w_fwd` eval call resolves).
  let fwd := (cifar8FwdText B IC C1 C2 C3 C4 IMH IMW KH KW D1 NC).replace "@cifar8_fwd" "@cifar8w_fwd"
  IO.println s!"rendered cifar8w fwd: {fwd.length} chars"
  IO.FS.writeFile "verified_mlir/cifar8w_fwd.mlir" fwd
  let bnfwd := (cifar8BnFwdTextPC B IC C1 C2 C3 C4 IMH IMW KH KW D1 NC "1.0e-05").replace "@cifar8_bn_fwd" "@cifar8w_bn_fwd"
  IO.println s!"rendered cifar8w_bn fwd: {bnfwd.length} chars"
  IO.FS.writeFile "verified_mlir/cifar8w_bn_fwd.mlir" bnfwd
  tryCompile "verified_mlir/cifar8w_fwd.mlir" "/tmp/cifar8w_fwd.vmfb" "cifar8w fwd"
  tryCompile "verified_mlir/cifar8w_bn_fwd.mlir" "/tmp/cifar8w_bn_fwd.vmfb" "cifar8w_bn fwd"
  tryCompile "verified_mlir/cifar8w_adam_train_step.mlir" "/tmp/cifar8w_adam_ts.vmfb" "cifar8 AdamW"
  tryCompile "verified_mlir/cifar8w_bn_adam_train_step.mlir" "/tmp/cifar8w_bn_adam_ts.vmfb" "cifar8_bn AdamW"
  tryCompile "verified_mlir/cifar8w_mom_train_step.mlir" "/tmp/cifar8w_mom_ts.vmfb" "cifar8 Nesterov-mom"
  tryCompile "verified_mlir/cifar8w_bn_mom_train_step.mlir" "/tmp/cifar8w_bn_mom_ts.vmfb" "cifar8_bn Nesterov-mom"
  tryCompile "verified_mlir/cifar8w_sgd_train_step.mlir" "/tmp/cifar8w_sgd_ts.vmfb" "cifar8 SGD-sched"
  tryCompile "verified_mlir/cifar8w_bn_sgd_train_step.mlir" "/tmp/cifar8w_bn_sgd_ts.vmfb" "cifar8_bn SGD-sched"

#eval main
