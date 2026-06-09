import LeanMlir.Proofs.ResNet34RenderPC
import LeanMlir.Types

/-! # r34 Item B — structured ResNet-34 train-step render (per-channel BN, proof-rendered)

The ResNet-34 peer of `tests/TestMobilenetV2TrainPC.lean`. Forward AND the whole backward cotangent
chain are proof-rendered through `pretty` over the per-channel tokens of `resnet34FwdGraphFullPC`
(r34 Item A): forward (`flatConvStridedF`/`flatConvF`/`bnPerChannelF`/`reluF`/`maxPoolF`/`addV`/`gapF`/
`denseF`) and backward (`dotOut`, `bnPerChannelBack`, **`selectPos`** (single-sided relu mask),
`convBack`/`convStridedBack`, **`maxPoolBack`**, `addV` residual fan-in). Only the no-SHlo-constructor
pieces are hand-emitted: GAP backward, conv/strided-conv weight+bias grads, the 7×7 stem weight-grad,
and per-channel BN dγ/dβ; reshape glue (flat→NCHW) only at those.

Same 146-param func signature/order as the committed `tests/TestResnet34Train.lean` → drop-in.
Validate with `scripts/render_parity.py --fn resnet34_train_step --ref verified_mlir/resnet34_train_step.mlir
--cand /tmp/r34pc/train_step.mlir`.

Run: `IREE_BACKEND=rocm lake env lean tests/TestResnet34TrainPC.lean`
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def EPS : String := "1.0e-5"
private def LR : String := "0.1"

private def zK {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0
private def zV {n : Nat} : Vec n := fun _ => 0
private def zM {a b : Nat} : Mat a b := fun _ _ => 0

-- ════════════ hand-emitted tail templates (NCHW) ════════════

private def rs4 (o flatN : String) (C Hh Ww : Nat) : String :=
  s!"    {o} = stablehlo.reshape {flatN} : ({ty [BS, C*Hh*Ww]}) -> {ty [BS,C,Hh,Ww]}\n"

/-- kk×kk conv weight-grad (transpose trick), inputs flat. -/
private def convWGrad (o inpFlat dyFlat : String) (ic oc Hh Ww kk : Nat) : String :=
  rs4 s!"{o}xi" inpFlat ic Hh Ww ++ rs4 s!"{o}di" dyFlat oc Hh Ww ++
  s!"    {o}xt = stablehlo.transpose {o}xi, dims = [1, 0, 2, 3] : ({ty [BS,ic,Hh,Ww]}) -> {ty [ic,BS,Hh,Ww]}\n" ++
  s!"    {o}dt = stablehlo.transpose {o}di, dims = [1, 0, 2, 3] : ({ty [BS,oc,Hh,Ww]}) -> {ty [oc,BS,Hh,Ww]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{(kk-1)/2}, {(kk-1)/2}], [{(kk-1)/2}, {(kk-1)/2}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ic,BS,Hh,Ww]}, {ty [oc,BS,Hh,Ww]}) -> {ty [ic,oc,kk,kk]}\n" ++
  s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [ic,oc,kk,kk]}) -> {ty [oc,ic,kk,kk]}\n"

/-- Zero-upsample a flat cotangent [BS,c,Hh,Ww] → flat [BS,c,2Hh,2Ww] (pad interior/high=1). -/
private def upsampleFlat (o dyFlat : String) (c Hh Ww : Nat) : String :=
  rs4 s!"{o}i" dyFlat c Hh Ww ++
  s!"    {o}p = stablehlo.pad {o}i, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c,2*Hh,2*Ww]}\n" ++
  s!"    {o} = stablehlo.reshape {o}p : ({ty [BS,c,2*Hh,2*Ww]}) -> {ty [BS, c*(2*Hh)*(2*Ww)]}\n"

private def biasGrad (o dyFlat : String) (oc Hh Ww : Nat) : String :=
  rs4 s!"{o}i" dyFlat oc Hh Ww ++
  s!"    {o} = stablehlo.reduce({o}i init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

/-- per-channel BN dγ_c=Σ_{b,h,w} dy·x̂, dβ_c=Σ_{b,h,w} dy; recompute x̂ from conv out `convFlat`. -/
private def bnParamGrad (dgr dbe convFlat dyFlat : String) (C Hh Ww : Nat) : String :=
  rs4 s!"{dgr}xr" convFlat C Hh Ww ++ rs4 s!"{dgr}dyr" dyFlat C Hh Ww ++
  s!"    {dgr}nf = stablehlo.constant dense<{Hh*Ww}.0> : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}ep = stablehlo.constant dense<{EPS}> : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}smr = stablehlo.reduce({dgr}xr init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,C,Hh,Ww]}, tensor<f32>) -> {ty [BS,C]}\n" ++
  s!"    {dgr}sm = stablehlo.broadcast_in_dim {dgr}smr, dims = [0, 1] : ({ty [BS,C]}) -> {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}mu = stablehlo.divide {dgr}sm, {dgr}nf : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}xc = stablehlo.subtract {dgr}xr, {dgr}mu : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}sq = stablehlo.multiply {dgr}xc, {dgr}xc : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}vsr = stablehlo.reduce({dgr}sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : ({ty [BS,C,Hh,Ww]}, tensor<f32>) -> {ty [BS,C]}\n" ++
  s!"    {dgr}vs = stablehlo.broadcast_in_dim {dgr}vsr, dims = [0, 1] : ({ty [BS,C]}) -> {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}var = stablehlo.divide {dgr}vs, {dgr}nf : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}ve = stablehlo.add {dgr}var, {dgr}ep : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}istd = stablehlo.rsqrt {dgr}ve : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}xh = stablehlo.multiply {dgr}xc, {dgr}istd : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr}p = stablehlo.multiply {dgr}dyr, {dgr}xh : {ty [BS,C,Hh,Ww]}\n" ++
  s!"    {dgr} = stablehlo.reduce({dgr}p init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,C,Hh,Ww]}, tensor<f32>) -> {ty [C]}\n" ++
  s!"    {dbe} = stablehlo.reduce({dgr}dyr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,C,Hh,Ww]}, tensor<f32>) -> {ty [C]}\n"

private def sgd (θ dθ ty' : String) : String :=
  s!"    {θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
  s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"

/-- Maxpool backward (select_and_scatter), flat in/out. Hand-emitted (not the `maxPoolBack` token):
    the token's `emitTok` hardcodes region args `%sa/%sb/%sc/%sd`, which collide with the stem-bias
    param `%sb` and the `%sc` init constant; we use `%q`-prefixed region names (as the committed
    renderer does). `src` = saved pre-pool input (flat @2Hh), `dy` = cot at pool output (flat @Hh). -/
private def maxpoolBackFlat (o srcFlat dyFlat : String) (c Hh Ww : Nat) : String :=
  rs4 s!"{o}sr" srcFlat c (2*Hh) (2*Ww) ++ rs4 s!"{o}dr" dyFlat c Hh Ww ++
  s!"    {o}ss = \"stablehlo.select_and_scatter\"({o}sr, {o}dr, %sc) (" ++ "{\n" ++
  "      ^bb0(%qa: tensor<f32>, %qb: tensor<f32>):\n" ++
  "        %qge = stablehlo.compare GE, %qa, %qb : (tensor<f32>, tensor<f32>) -> tensor<i1>\n" ++
  "        stablehlo.return %qge : tensor<i1>\n" ++
  "    }, " ++ "{\n" ++
  "      ^bb0(%qc: tensor<f32>, %qd: tensor<f32>):\n" ++
  "        %qs = stablehlo.add %qc, %qd : tensor<f32>\n" ++
  "        stablehlo.return %qs : tensor<f32>\n" ++
  "    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}" ++
  s!" : ({ty [BS,c,2*Hh,2*Ww]}, {ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c,2*Hh,2*Ww]}\n" ++
  s!"    {o} = stablehlo.reshape {o}ss : ({ty [BS,c,2*Hh,2*Ww]}) -> {ty [BS, c*(2*Hh)*(2*Ww)]}\n"

-- ════════════ captured forward names per basic block ════════════
private structure FNames where
  xin : String  -- block input
  c1 : String   -- conv1 out
  n1 : String   -- bn1 out (relu1 pre)
  r1 : String   -- relu1 out (conv2 in)
  c2 : String   -- conv2 out
  a : String    -- addV out (block relu pre)
  cp : String   -- proj conv out (down only; "" for id)
  o : String    -- block out

/-- (p, isDown, ic, c, Hh).  id: ic=c, input@Hh.  down: ic→c, input@2Hh, output@Hh. -/
private def blocks : List (String × Bool × Nat × Nat × Nat) :=
  [("s1b0", false, 64, 64, 56), ("s1b1", false, 64, 64, 56), ("s1b2", false, 64, 64, 56),
   ("d2", true, 64, 128, 28), ("s2b0", false, 128, 128, 28), ("s2b1", false, 128, 128, 28), ("s2b2", false, 128, 128, 28),
   ("d3", true, 128, 256, 14), ("s3b0", false, 256, 256, 14), ("s3b1", false, 256, 256, 14), ("s3b2", false, 256, 256, 14), ("s3b3", false, 256, 256, 14), ("s3b4", false, 256, 256, 14),
   ("d4", true, 256, 512, 7), ("s4b0", false, 512, 512, 7), ("s4b1", false, 512, 512, 7)]

/-- One basic block forward via `pretty`, capturing flat names. -/
private def fwdBlock (p xin : String) (isDown : Bool) (ic c Hh : Nat) : StateM Nat (String × FNames) := do
  let (k1, c1) ←
    if isDown then pretty BS (.flatConvStridedF (h := Hh) (w := Hh) s!"%{p}W1" s!"%{p}b1" (zK : Kernel4 c ic 3 3) zV (.operand xin zV))
    else pretty BS (.flatConvF (h := Hh) (w := Hh) s!"%{p}W1" s!"%{p}b1" (zK : Kernel4 c ic 3 3) zV (.operand xin zV))
  let (k2, n1) ← pretty BS (.bnPerChannelF (oc := c) (h := Hh) (w := Hh) s!"%{p}g1" s!"%{p}bt1" EPS 0 zV zV (.operand c1 zV))
  let (k3, r1) ← pretty BS (.reluF (.operand n1 (zV : Vec (c*Hh*Hh))))
  let (k4, c2) ← pretty BS (.flatConvF (h := Hh) (w := Hh) s!"%{p}W2" s!"%{p}b2" (zK : Kernel4 c c 3 3) zV (.operand r1 zV))
  let (k5, n2) ← pretty BS (.bnPerChannelF (oc := c) (h := Hh) (w := Hh) s!"%{p}g2" s!"%{p}bt2" EPS 0 zV zV (.operand c2 zV))
  let fwd := k1 ++ k2 ++ k3 ++ k4 ++ k5
  if isDown then
    let (k6, cp) ← pretty BS (.flatConvStridedF (h := Hh) (w := Hh) s!"%{p}Wp" s!"%{p}bp" (zK : Kernel4 c ic 3 3) zV (.operand xin zV))
    let (k7, np) ← pretty BS (.bnPerChannelF (oc := c) (h := Hh) (w := Hh) s!"%{p}gp" s!"%{p}btp" EPS 0 zV zV (.operand cp zV))
    let (k8, a) ← pretty BS (.addV (.operand n2 (zV : Vec (c*Hh*Hh))) (.operand np zV))
    let (k9, o) ← pretty BS (.reluF (.operand a (zV : Vec (c*Hh*Hh))))
    pure (fwd ++ k6 ++ k7 ++ k8 ++ k9, ⟨xin, c1, n1, r1, c2, a, cp, o⟩)
  else
    let (k6, a) ← pretty BS (.addV (.operand n2 (zV : Vec (c*Hh*Hh))) (.operand xin zV))
    let (k7, o) ← pretty BS (.reluF (.operand a (zV : Vec (c*Hh*Hh))))
    pure (fwd ++ k6 ++ k7, ⟨xin, c1, n1, r1, c2, a, "", o⟩)

/-- One basic block backward cotangent chain via `pretty` (flat tokens). `dy` = cot at block output.
    Returns (code, cot-at-block-input, cot_a, cot_c1, cot_n1, cot_c2, cot_cp). -/
private def bwdBlock (p dy : String) (b : FNames) (isDown : Bool) (ic c Hh : Nat) :
    StateM Nat (String × String × String × String × String × String × String) := do
  let (k1, cot_a) ← pretty BS (.selectPos b.a (zV : Vec (c*Hh*Hh)) (.operand dy zV))
  -- main path: bn2 → conv2 → relu1 → bn1 → conv1
  let (k2, cot_c2) ← pretty BS (.bnPerChannelBack (oc := c) (h := Hh) (w := Hh) s!"%{p}g2" b.c2 EPS 0 zV zV (.operand cot_a zV))
  let (k3, cot_r1) ← pretty BS (.convBack (h := Hh) (w := Hh) s!"%{p}W2" (zK : Kernel4 c c 3 3) zV zV (.operand cot_c2 zV))
  let (k4, cot_n1) ← pretty BS (.selectPos b.n1 (zV : Vec (c*Hh*Hh)) (.operand cot_r1 zV))
  let (k5, cot_c1) ← pretty BS (.bnPerChannelBack (oc := c) (h := Hh) (w := Hh) s!"%{p}g1" b.c1 EPS 0 zV zV (.operand cot_n1 zV))
  let chain := k1 ++ k2 ++ k3 ++ k4 ++ k5
  if isDown then
    let (k6, cot_xmain) ← pretty BS (.convStridedBack (h := Hh) (w := Hh) s!"%{p}W1" (zK : Kernel4 c ic 3 3) zV zV (.operand cot_c1 zV))
    -- proj path: bnp → convp (strided)
    let (k7, cot_cp) ← pretty BS (.bnPerChannelBack (oc := c) (h := Hh) (w := Hh) s!"%{p}gp" b.cp EPS 0 zV zV (.operand cot_a zV))
    let (k8, cot_xproj) ← pretty BS (.convStridedBack (h := Hh) (w := Hh) s!"%{p}Wp" (zK : Kernel4 c ic 3 3) zV zV (.operand cot_cp zV))
    let (k9, cot_xin) ← pretty BS (.addV (.operand cot_xmain (zV : Vec (ic*(2*Hh)*(2*Hh)))) (.operand cot_xproj zV))
    pure (chain ++ k6 ++ k7 ++ k8 ++ k9, cot_xin, cot_a, cot_c1, cot_n1, cot_c2, cot_cp)
  else
    let (k6, cot_xmain) ← pretty BS (.convBack (h := Hh) (w := Hh) s!"%{p}W1" (zK : Kernel4 c ic 3 3) zV zV (.operand cot_c1 zV))
    -- residual fan-in: cot at block input = main + skip (skip cot = cot_a, since relu is AFTER the add)
    let (k7, cot_xin) ← pretty BS (.addV (.operand cot_xmain (zV : Vec (ic*Hh*Hh))) (.operand cot_a zV))
    pure (chain ++ k6 ++ k7, cot_xin, cot_a, cot_c1, cot_n1, cot_c2, "")

/-- block param-grad text (hand-emitted), given captured fwd names + cotangents. -/
private def blockParamGrads (p : String) (b : FNames) (cot_a cot_c1 cot_n1 cot_c2 cot_cp : String)
    (isDown : Bool) (ic c Hh : Nat) : String :=
  -- conv2 (3×3 s1): W/b/γ/β
  convWGrad s!"%{p}dW2" b.r1 cot_c2 c c Hh Hh 3 ++ biasGrad s!"%{p}db2" cot_c2 c Hh Hh ++
  bnParamGrad s!"%{p}dg2" s!"%{p}dbt2" b.c2 cot_a c Hh Hh ++
  -- conv1: W (strided for down: upsample cot_c1 then weight-grad against xin@2Hh; else 3×3 s1)/b/γ/β
  (if isDown then upsampleFlat s!"%{p}dW1u" cot_c1 c Hh Hh ++ convWGrad s!"%{p}dW1" b.xin s!"%{p}dW1u" ic c (2*Hh) (2*Hh) 3
   else convWGrad s!"%{p}dW1" b.xin cot_c1 ic c Hh Hh 3) ++
  biasGrad s!"%{p}db1" cot_c1 c Hh Hh ++
  bnParamGrad s!"%{p}dg1" s!"%{p}dbt1" b.c1 cot_n1 c Hh Hh ++
  -- projection (down only): strided 3×3 W/b/γ/β
  (if isDown then
    upsampleFlat s!"%{p}dWpu" cot_cp c Hh Hh ++ convWGrad s!"%{p}dWp" b.xin s!"%{p}dWpu" ic c (2*Hh) (2*Hh) 3 ++
    biasGrad s!"%{p}dbp" cot_cp c Hh Hh ++ bnParamGrad s!"%{p}dgp" s!"%{p}dbtp" b.cp cot_a c Hh Hh
   else "")

private def blockSgd (p : String) (isDown : Bool) (ic c : Nat) : String :=
  sgd s!"%{p}W1" s!"%{p}dW1" (if isDown then ty [c,ic,3,3] else ty [c,c,3,3]) ++ sgd s!"%{p}b1" s!"%{p}db1" (ty [c]) ++
  sgd s!"%{p}g1" s!"%{p}dg1" (ty [c]) ++ sgd s!"%{p}bt1" s!"%{p}dbt1" (ty [c]) ++
  sgd s!"%{p}W2" s!"%{p}dW2" (ty [c,c,3,3]) ++ sgd s!"%{p}b2" s!"%{p}db2" (ty [c]) ++
  sgd s!"%{p}g2" s!"%{p}dg2" (ty [c]) ++ sgd s!"%{p}bt2" s!"%{p}dbt2" (ty [c]) ++
  (if isDown then
    sgd s!"%{p}Wp" s!"%{p}dWp" (ty [c,ic,3,3]) ++ sgd s!"%{p}bp" s!"%{p}dbp" (ty [c]) ++
    sgd s!"%{p}gp" s!"%{p}dgp" (ty [c]) ++ sgd s!"%{p}btp" s!"%{p}dbtp" (ty [c])
   else "")

private def trainStep : String := Id.run do
  let go : StateM Nat String := do
    -- ═══ forward ═══ stem 7×7-s2 → bn → relu → maxpool
    let (cSc, stc) ← pretty BS (.flatConvStridedF (h := 112) (w := 112) "%sW" "%sb" (zK : Kernel4 64 3 7 7) zV (.operand "%x" zV))
    let (cSn, stn) ← pretty BS (.bnPerChannelF (oc := 64) (h := 112) (w := 112) "%sg" "%sbt" EPS 0 zV zV (.operand stc zV))
    let (cSr, str) ← pretty BS (.reluF (.operand stn (zV : Vec (64*112*112))))
    let (cMp, stp) ← pretty BS (.maxPoolF (c := 64) (h := 56) (w := 56) (.operand str zV))
    let mut fwd := cSc ++ cSn ++ cSr ++ cMp
    let mut cur := stp
    let mut bns : List (FNames × (String × Bool × Nat × Nat × Nat)) := []
    for blk in blocks do
      let (p, isDown, ic, c, Hh) := blk
      let (code, bn) ← fwdBlock p cur isDown ic c Hh
      fwd := fwd ++ code
      cur := bn.o
      bns := bns ++ [(bn, blk)]
    -- GAP(512,7,7) → dense(512→10)
    let (cGap, gap) ← pretty BS (.gapF (c := 512) (h := 7) (w := 7) (.operand cur zV))
    let (cLog, logits) ← pretty BS (denseF "%Wd" "%bd" (zM : Mat 512 10) zV (.operand gap zV))
    let (cSub, dyr) ← pretty BS (.sub (.softmaxDiv (.expe (.operand logits (zV : Vec 10)))) (.operand "%onehot" (zV : Vec 10)))
    fwd := fwd ++ cGap ++ cLog ++ cSub
    -- ═══ backward ═══ dy/BS, dense-back, GAP-back, blocks reversed, maxpool, stem
    let (cDg, cotGap) ← pretty BS (.dotOut "%Wd" (zM : Mat 512 10) (.operand "%dy" zV))
    let mut bwd :=
      s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [BS,10]}\n" ++ cDg ++
      rs4 "%dgi" cotGap 512 1 1 ++
      s!"    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : ({ty [BS,512,1,1]}) -> {ty [BS,512,7,7]}\n" ++
      s!"    %dgn = stablehlo.constant dense<49.0> : {ty [BS,512,7,7]}\n" ++
      s!"    %dgd = stablehlo.divide %dgb, %dgn : {ty [BS,512,7,7]}\n" ++
      s!"    %dgapf = stablehlo.reshape %dgd : ({ty [BS,512,7,7]}) -> {ty [BS, 512*7*7]}\n"
    let mut paramG := ""
    let mut d := "%dgapf"
    for (bn, blk) in bns.reverse do
      let (p, isDown, ic, c, Hh) := blk
      let (code, dxin, cot_a, cot_c1, cot_n1, cot_c2, cot_cp) ← bwdBlock p d bn isDown ic c Hh
      bwd := bwd ++ code
      paramG := paramG ++ blockParamGrads p bn cot_a cot_c1 cot_n1 cot_c2 cot_cp isDown ic c Hh
      d := dxin
    -- maxpool back (hand, saved pre-pool str) → stem relu → stem bn (both via tokens)
    let cMpB := maxpoolBackFlat "%dmp" str d 64 56 56
    let (cSrB, cot_stn) ← pretty BS (.selectPos stn (zV : Vec (64*112*112)) (.operand "%dmp" zV))
    let (cSnB, cot_stc) ← pretty BS (.bnPerChannelBack (oc := 64) (h := 112) (w := 112) "%sg" stc EPS 0 zV zV (.operand cot_stn zV))
    bwd := bwd ++ cMpB ++ cSrB ++ cSnB
    -- ═══ remaining param grads (hand) ═══ stem (7×7 strided W) + dense
    let stemG :=
      upsampleFlat "%dsu" cot_stc 64 112 112 ++ convWGrad "%dsW" "%x" "%dsu" 3 64 224 224 7 ++
      biasGrad "%dsb" cot_stc 64 112 112 ++ bnParamGrad "%dsg" "%dsbt" stc cot_stn 64 112 112
    let denseG :=
      s!"    %dWd = stablehlo.dot_general {gap}, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,512]}, {ty [BS,10]}) -> {ty [512,10]}\n" ++
      s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    pure (fwd ++ bwd ++ paramG ++ stemG ++ denseG)
  let body : String := go.run' 0
  -- ═══ SGD over all 146 params + signature ═══
  let stemSgd := sgd "%sW" "%dsW" (ty [64,3,7,7]) ++ sgd "%sb" "%dsb" (ty [64]) ++ sgd "%sg" "%dsg" (ty [64]) ++ sgd "%sbt" "%dsbt" (ty [64])
  let blkSgd := String.join (blocks.map (fun (p, isDown, ic, c, _) => blockSgd p isDown ic c))
  let denseSgd := sgd "%Wd" "%dWd" (ty [512,10]) ++ sgd "%bd" "%dbd" (ty [10])
  let blkParams (p : String) (isDown : Bool) (ic c : Nat) : List (String × String) :=
    [(s!"{p}W1", if isDown then ty [c,ic,3,3] else ty [c,c,3,3]), (s!"{p}b1", ty [c]), (s!"{p}g1", ty [c]), (s!"{p}bt1", ty [c]),
     (s!"{p}W2", ty [c,c,3,3]), (s!"{p}b2", ty [c]), (s!"{p}g2", ty [c]), (s!"{p}bt2", ty [c])]
    ++ (if isDown then [(s!"{p}Wp", ty [c,ic,3,3]), (s!"{p}bp", ty [c]), (s!"{p}gp", ty [c]), (s!"{p}btp", ty [c])] else [])
  let allParams : List (String × String) :=
    [("sW", ty [64,3,7,7]), ("sb", ty [64]), ("sg", ty [64]), ("sbt", ty [64])]
    ++ (blocks.map (fun (p, isDown, ic, c, _) => blkParams p isDown ic c)).flatten
    ++ [("Wd", ty [512,10]), ("bd", ty [10])]
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS,150528]) :: allParams.map (fun (nm, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @resnet34_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n" ++
    body ++ stemSgd ++ blkSgd ++ denseSgd ++
    s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered structured ResNet-34 train step: {mlir.length} chars"
  IO.FS.createDirAll "/tmp/r34pc"
  IO.FS.writeFile "/tmp/r34pc/train_step.mlir" mlir
  let cargs ← ireeCompileArgs "/tmp/r34pc/train_step.mlir" "/tmp/r34pc/train_step.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 5000}"
  else
    IO.println "structured ResNet-34 train step iree-compile OK → /tmp/r34pc/train_step.mlir"

#eval main
