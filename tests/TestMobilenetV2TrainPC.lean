import LeanMlir.Proofs.MobileNetV2RenderPC
import LeanMlir.Types

/-! # Item B — structured MobileNetV2 train-step render (per-channel BN, proof-rendered)

The MobileNetV2 peer of `cifarBnTrainStepStructured` (CnnRender.lean). The **forward** AND the
**whole backward cotangent chain** are proof-rendered through `pretty` over the per-channel tokens
of `mobilenetv2FwdGraphFullPC` (Item A) — forward (`flatConvStridedF`/`flatConvF`/`depthwiseF`/
`depthwiseStridedF`/`bnPerChannelF`/`relu6F`/`addV`/`gapF`/`denseF`) and backward (`dotOut`,
`bnPerChannelBack`, `selectMid`, `convBack`, `depthwiseBack`/`depthwiseStridedBack`, `addV` residual
fan-in). Only the pieces with **no SHlo constructor** are hand-emitted: the global-avg-pool backward,
the conv/depthwise weight+bias grads (transpose trick / reduce), and the per-channel BN dγ/dβ
(recomputed x̂). Everything stays flat except the hand weight/bias/BN-param grads, which reshape
flat→NCHW at their boundary (`reshape` is a buffer no-op).

Same func signature (82 params, same names/order) as the committed `TestMobilenetV2Train.lean`, so it
**swap-trains** `mobilenetv2-verified`. Trains EQUIVALENTLY (the per-channel BN input-grad/param-grad
recompute x̂/istd from the saved conv-output rather than reusing forward intermediates), not
bit-identically — exactly as the CIFAR-BN structured render.

Run: `IREE_BACKEND=rocm lake env lean tests/TestMobilenetV2TrainPC.lean`
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def EPS : String := "1.0e-5"
private def LR : String := "0.3"

-- placeholder values (pretty/emitTok render names only; values are irrelevant)
private def zK {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0
private def zD {c kh kw : Nat} : DepthwiseKernel c kh kw := fun _ _ _ => 0
private def zV {n : Nat} : Vec n := fun _ => 0
private def zM {a b : Nat} : Mat a b := fun _ _ => 0

-- ════════════ hand-emitted tail templates (NCHW, same op text as TestMobilenetV2Train) ════════════

/-- flat → NCHW reshape. -/
private def rs4 (o flatN : String) (C Hh Ww : Nat) : String :=
  s!"    {o} = stablehlo.reshape {flatN} : ({ty [BS, C*Hh*Ww]}) -> {ty [BS,C,Hh,Ww]}\n"

/-- 1×1 / 3×3 conv weight-grad (transpose trick), inputs flat. `kk` = kernel spatial. -/
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

/-- Depthwise 3×3 weight-grad (batch_group_count=c), inputs flat. -/
private def dwWGrad (o inpFlat dyFlat : String) (c Hh Ww : Nat) : String :=
  rs4 s!"{o}xi" inpFlat c Hh Ww ++ rs4 s!"{o}di" dyFlat c Hh Ww ++
  s!"    {o}xt = stablehlo.transpose {o}xi, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    {o}dt = stablehlo.transpose {o}di, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [c,BS,Hh,Ww]}, {ty [c,BS,Hh,Ww]}) -> {ty [1,c,3,3]}\n" ++
  s!"    {o} = stablehlo.reshape {o}raw : ({ty [1,c,3,3]}) -> {ty [c,1,3,3]}\n"

/-- Zero-upsample a flat cotangent [BS,c,Hh,Ww] → flat [BS,c,2Hh,2Ww] (pad interior/high=1). -/
private def upsampleFlat (o dyFlat : String) (c Hh Ww : Nat) : String :=
  rs4 s!"{o}i" dyFlat c Hh Ww ++
  s!"    {o}p = stablehlo.pad {o}i, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : ({ty [BS,c,Hh,Ww]}, tensor<f32>) -> {ty [BS,c,2*Hh,2*Ww]}\n" ++
  s!"    {o} = stablehlo.reshape {o}p : ({ty [BS,c,2*Hh,2*Ww]}) -> {ty [BS, c*(2*Hh)*(2*Ww)]}\n"

/-- conv/depthwise bias-grad: reduce flat cotangent over batch+spatial → [oc]. -/
private def biasGrad (o dyFlat : String) (oc Hh Ww : Nat) : String :=
  rs4 s!"{o}i" dyFlat oc Hh Ww ++
  s!"    {o} = stablehlo.reduce({o}i init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

/-- per-channel BN dγ_c=Σ_{b,h,w} dy·x̂, dβ_c=Σ_{b,h,w} dy; recompute x̂ from the saved conv out
    (flat BN input `convFlat`), dy = flat cotangent at BN output `dyFlat`. -/
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

-- ════════════ captured forward names per inverted-residual block ════════════
private structure FNames where  -- all flat SSA names from `pretty`
  xin : String   -- block input
  ec : String    -- expand conv out
  en : String    -- expand bn out
  er : String    -- expand relu6 out
  dc : String    -- depthwise conv out
  dn : String    -- depthwise bn out
  dr : String    -- depthwise relu6 out
  pc : String    -- project conv out
  pn : String    -- project bn out
  bout : String  -- block output (pn or addV(pn,xin))

private def blocks : List (String × Nat × Nat × Nat × Nat × Nat) :=
  -- (p, ic, mid, oc, s, Hin)
  [("b1", 16, 64,  24, 2, 112), ("b2", 24, 96,  24, 1, 56),
   ("b3", 24, 96,  32, 2, 56),  ("b4", 32, 128, 32, 1, 28),
   ("b5", 32, 128, 64, 2, 28),  ("b6", 64, 256, 64, 2, 14)]

/-- One inverted-residual block forward via `pretty`, capturing flat names. -/
private def fwdBlock (p xin : String) (ic mid oc s Hin : Nat) : StateM Nat (String × FNames) := do
  let Hout := Hin / s
  let (c1, ec) ← pretty BS (.flatConvF (h := Hin) (w := Hin) s!"%{p}eW" s!"%{p}eb" (zK : Kernel4 mid ic 1 1) zV (.operand xin zV))
  let (c2, en) ← pretty BS (.bnPerChannelF (oc := mid) (h := Hin) (w := Hin) s!"%{p}eg" s!"%{p}ebt" EPS 0 zV zV (.operand ec zV))
  let (c3, er) ← pretty BS (.relu6F (.operand en (zV : Vec (mid*Hin*Hin))))
  let (c4, dc) ←
    if s == 2 then pretty BS (.depthwiseStridedF (h := Hout) (w := Hout) s!"%{p}dW" s!"%{p}db" (zD : DepthwiseKernel mid 3 3) zV (.operand er zV))
    else pretty BS (.depthwiseF (h := Hin) (w := Hin) s!"%{p}dW" s!"%{p}db" (zD : DepthwiseKernel mid 3 3) zV (.operand er zV))
  let (c5, dn) ← pretty BS (.bnPerChannelF (oc := mid) (h := Hout) (w := Hout) s!"%{p}dg" s!"%{p}dbt" EPS 0 zV zV (.operand dc zV))
  let (c6, dr) ← pretty BS (.relu6F (.operand dn (zV : Vec (mid*Hout*Hout))))
  let (c7, pc) ← pretty BS (.flatConvF (h := Hout) (w := Hout) s!"%{p}pW" s!"%{p}pb" (zK : Kernel4 oc mid 1 1) zV (.operand dr zV))
  let (c8, pn) ← pretty BS (.bnPerChannelF (oc := oc) (h := Hout) (w := Hout) s!"%{p}pg" s!"%{p}pbt" EPS 0 zV zV (.operand pc zV))
  let fwd := c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++ c8
  if s == 1 && ic == oc then
    let (c9, bout) ← pretty BS (.addV (.operand pn (zV : Vec (oc*Hout*Hout))) (.operand xin zV))
    pure (fwd ++ c9, ⟨xin, ec, en, er, dc, dn, dr, pc, pn, bout⟩)
  else
    pure (fwd, ⟨xin, ec, en, er, dc, dn, dr, pc, pn, pn⟩)

/-- One inverted-residual block backward cotangent chain via `pretty` (all flat tokens), capturing
    the cotangents the param grads need. `dy` = flat cotangent at block output. Returns
    (code, cot-at-block-input, [cot_pc, cot_dc, cot_ec, cot_en, cot_dn]). -/
private def bwdBlock (p dy : String) (b : FNames) (ic mid oc s Hin : Nat) :
    StateM Nat (String × String × String × String × String × String × String) := do
  let Hout := Hin / s
  -- project: bn-back → conv-back (1×1)
  let (k1, cot_pc) ← pretty BS (.bnPerChannelBack (oc := oc) (h := Hout) (w := Hout) s!"%{p}pg" b.pc EPS 0 zV zV (.operand dy zV))
  let (k2, cot_dr) ← pretty BS (.convBack (h := Hout) (w := Hout) s!"%{p}pW" (zK : Kernel4 oc mid 1 1) zV zV (.operand cot_pc zV))
  -- depthwise: relu6-mask → bn-back → depthwise-back
  let (k3, cot_dn) ← pretty BS (.selectMid b.dn (zV : Vec (mid*Hout*Hout)) (.operand cot_dr zV))
  let (k4, cot_dc) ← pretty BS (.bnPerChannelBack (oc := mid) (h := Hout) (w := Hout) s!"%{p}dg" b.dc EPS 0 zV zV (.operand cot_dn zV))
  let (k5, cot_er) ←
    if s == 2 then pretty BS (.depthwiseStridedBack (h := Hout) (w := Hout) s!"%{p}dW" (zD : DepthwiseKernel mid 3 3) zV zV (.operand cot_dc zV))
    else pretty BS (.depthwiseBack (h := Hin) (w := Hin) s!"%{p}dW" (zD : DepthwiseKernel mid 3 3) zV zV (.operand cot_dc zV))
  -- expand: relu6-mask → bn-back → conv-back (1×1)
  let (k6, cot_en) ← pretty BS (.selectMid b.en (zV : Vec (mid*Hin*Hin)) (.operand cot_er zV))
  let (k7, cot_ec) ← pretty BS (.bnPerChannelBack (oc := mid) (h := Hin) (w := Hin) s!"%{p}eg" b.ec EPS 0 zV zV (.operand cot_en zV))
  let (k8, cot_xpre) ← pretty BS (.convBack (h := Hin) (w := Hin) s!"%{p}eW" (zK : Kernel4 mid ic 1 1) zV zV (.operand cot_ec zV))
  let chain := k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7 ++ k8
  if s == 1 && ic == oc then
    let (k9, cot_xin) ← pretty BS (.addV (.operand cot_xpre (zV : Vec (ic*Hin*Hin))) (.operand dy zV))
    pure (chain ++ k9, cot_xin, cot_pc, cot_dc, cot_ec, cot_en, cot_dn)
  else
    pure (chain, cot_xpre, cot_pc, cot_dc, cot_ec, cot_en, cot_dn)

/-- block param-grad + SGD text (hand-emitted), given captured fwd names + cotangents. -/
private def blockParamGrads (p : String) (b : FNames) (cot_pc cot_dc cot_ec cot_en cot_dn dy : String)
    (ic mid oc s Hin : Nat) : String :=
  let Hout := Hin / s
  -- project (1×1 @ Hout): W/b/γ/β
  convWGrad s!"%{p}dpW" b.dr cot_pc mid oc Hout Hout 1 ++ biasGrad s!"%{p}dpb" cot_pc oc Hout Hout ++
  bnParamGrad s!"%{p}dpg" s!"%{p}dpbt" b.pc dy oc Hout Hout ++
  -- depthwise (3×3): strided weight-grad upsamples dy first
  (if s == 2 then upsampleFlat s!"%{p}ddu" cot_dc mid Hout Hout ++ dwWGrad s!"%{p}ddW" b.er s!"%{p}ddu" mid (2*Hout) (2*Hout)
   else dwWGrad s!"%{p}ddW" b.er cot_dc mid Hin Hin) ++
  biasGrad s!"%{p}ddb" cot_dc mid Hout Hout ++
  bnParamGrad s!"%{p}ddg" s!"%{p}ddbt" b.dc cot_dn mid Hout Hout ++
  -- expand (1×1 @ Hin, ic→mid): W/b/γ/β
  convWGrad s!"%{p}deW" b.xin cot_ec ic mid Hin Hin 1 ++ biasGrad s!"%{p}deb" cot_ec mid Hin Hin ++
  bnParamGrad s!"%{p}deg" s!"%{p}debt" b.ec cot_en mid Hin Hin

/-- per-block SGD over the 12 params (matching `allParams` order/names). -/
private def blockSgd (p : String) (ic mid oc : Nat) : String :=
  sgd s!"%{p}eW" s!"%{p}deW" (ty [mid,ic,1,1]) ++ sgd s!"%{p}eb" s!"%{p}deb" (ty [mid]) ++
  sgd s!"%{p}eg" s!"%{p}deg" (ty [mid]) ++ sgd s!"%{p}ebt" s!"%{p}debt" (ty [mid]) ++
  sgd s!"%{p}dW" s!"%{p}ddW" (ty [mid,1,3,3]) ++ sgd s!"%{p}db" s!"%{p}ddb" (ty [mid]) ++
  sgd s!"%{p}dg" s!"%{p}ddg" (ty [mid]) ++ sgd s!"%{p}dbt" s!"%{p}ddbt" (ty [mid]) ++
  sgd s!"%{p}pW" s!"%{p}dpW" (ty [oc,mid,1,1]) ++ sgd s!"%{p}pb" s!"%{p}dpb" (ty [oc]) ++
  sgd s!"%{p}pg" s!"%{p}dpg" (ty [oc]) ++ sgd s!"%{p}pbt" s!"%{p}dpbt" (ty [oc])

private def trainStep : String := Id.run do
  let go : StateM Nat String := do
    -- ═══ forward (proof-rendered) ═══
    let (cStemC, stc) ← pretty BS (.flatConvStridedF (h := 112) (w := 112) "%sW" "%sb" (zK : Kernel4 16 3 3 3) zV (.operand "%x" zV))
    let (cStemB, stn) ← pretty BS (.bnPerChannelF (oc := 16) (h := 112) (w := 112) "%sg" "%sbt" EPS 0 zV zV (.operand stc zV))
    let (cStemR, str) ← pretty BS (.relu6F (.operand stn (zV : Vec (16*112*112))))
    -- 6 inverted-residual blocks
    let mut fwd := cStemC ++ cStemB ++ cStemR
    let mut cur := str
    let mut bns : List (FNames × (String × Nat × Nat × Nat × Nat × Nat)) := []
    for blk in blocks do
      let (p, ic, mid, oc, s, Hin) := blk
      let (code, bn) ← fwdBlock p cur ic mid oc s Hin
      fwd := fwd ++ code
      cur := bn.bout
      bns := bns ++ [(bn, blk)]
    -- head: 1×1 conv (64→128) → bn → relu6 @7
    let (cHc, hc) ← pretty BS (.flatConvF (h := 7) (w := 7) "%hW" "%hb" (zK : Kernel4 128 64 1 1) zV (.operand cur zV))
    let (cHb, hn) ← pretty BS (.bnPerChannelF (oc := 128) (h := 7) (w := 7) "%hg" "%hbt" EPS 0 zV zV (.operand hc zV))
    let (cHr, hr) ← pretty BS (.relu6F (.operand hn (zV : Vec (128*7*7))))
    let (cGap, gap) ← pretty BS (.gapF (c := 128) (h := 7) (w := 7) (.operand hr zV))
    let (cLog, logits) ← pretty BS (denseF "%Wd" "%bd" (zM : Mat 128 10) zV (.operand gap zV))
    -- loss cotangent: (softmax(logits) − onehot)/BS
    let (cSub, dyr) ← pretty BS (.sub (.softmaxDiv (.expe (.operand logits (zV : Vec 10)))) (.operand "%onehot" (zV : Vec 10)))
    fwd := fwd ++ cHc ++ cHb ++ cHr ++ cGap ++ cLog ++ cSub
    -- ═══ backward cotangent chain (proof-rendered) ═══
    -- dy = dyr / BS ; dense-back (dotOut) → gap-back (broadcast/÷49) → head relu6 → head bn → head conv
    let (cDg, cotGap) ← pretty BS (.dotOut "%Wd" (zM : Mat 128 10) (.operand "%dy" zV))
    let mut bwd :=
      s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [BS,10]}\n" ++ cDg ++
      rs4 "%dgi" cotGap 128 1 1 ++  -- [BS,128] → [BS,128,1,1]
      s!"    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : ({ty [BS,128,1,1]}) -> {ty [BS,128,7,7]}\n" ++
      s!"    %dgn = stablehlo.constant dense<49.0> : {ty [BS,128,7,7]}\n" ++
      s!"    %dgd = stablehlo.divide %dgb, %dgn : {ty [BS,128,7,7]}\n" ++
      s!"    %dgapf = stablehlo.reshape %dgd : ({ty [BS,128,7,7]}) -> {ty [BS, 128*7*7]}\n"
    let (cHrB, cot_hn) ← pretty BS (.selectMid hn (zV : Vec (128*7*7)) (.operand "%dgapf" zV))
    let (cHbB, cot_hc) ← pretty BS (.bnPerChannelBack (oc := 128) (h := 7) (w := 7) "%hg" hc EPS 0 zV zV (.operand cot_hn zV))
    let (cHcB, cot_b6) ← pretty BS (.convBack (h := 7) (w := 7) "%hW" (zK : Kernel4 128 64 1 1) zV zV (.operand cot_hc zV))
    bwd := bwd ++ cHrB ++ cHbB ++ cHcB
    -- block backward (reversed), threading the cotangent + accumulating param grads
    let mut paramG := ""
    let mut d := cot_b6
    for (bn, blk) in bns.reverse do
      let (p, ic, mid, oc, s, Hin) := blk
      let (code, dxin, cot_pc, cot_dc, cot_ec, cot_en, cot_dn) ← bwdBlock p d bn ic mid oc s Hin
      bwd := bwd ++ code
      paramG := paramG ++ blockParamGrads p bn cot_pc cot_dc cot_ec cot_en cot_dn d ic mid oc s Hin
      d := dxin
    -- stem backward: relu6 mask → bn-back → (strided 3×3 weight-grad). `d` = cot at stem relu6 out.
    let (cStR, cot_stn) ← pretty BS (.selectMid stn (zV : Vec (16*112*112)) (.operand d zV))
    let (cStB, cot_stc) ← pretty BS (.bnPerChannelBack (oc := 16) (h := 112) (w := 112) "%sg" stc EPS 0 zV zV (.operand cot_stn zV))
    bwd := bwd ++ cStR ++ cStB
    -- ═══ param grads (hand-emitted) ═══
    -- head: 1×1 conv W/b + bn γ/β
    let headG :=
      convWGrad "%dhW" cur cot_hc 64 128 7 7 1 ++ biasGrad "%dhb" cot_hc 128 7 7 ++
      bnParamGrad "%dhg" "%dhbt" hc cot_hn 128 7 7
    -- stem: strided 3×3 conv W (upsample dy) + b + bn γ/β
    let stemG :=
      upsampleFlat "%dsu" cot_stc 16 112 112 ++ convWGrad "%dsW" "%x" "%dsu" 3 16 224 224 3 ++
      biasGrad "%dsb" cot_stc 16 112 112 ++ bnParamGrad "%dsg" "%dsbt" stc cot_stn 16 112 112
    -- dense: Wd (gap ⊗ dy), bd (reduce dy)
    let denseG :=
      s!"    %dWd = stablehlo.dot_general {gap}, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,128]}, {ty [BS,10]}) -> {ty [128,10]}\n" ++
      s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    pure (fwd ++ bwd ++ paramG ++ headG ++ stemG ++ denseG)
  let body : String := go.run' 0
  -- ═══ SGD over all 82 params (allParams order) + signature ═══
  let stemSgd := sgd "%sW" "%dsW" (ty [16,3,3,3]) ++ sgd "%sb" "%dsb" (ty [16]) ++ sgd "%sg" "%dsg" (ty [16]) ++ sgd "%sbt" "%dsbt" (ty [16])
  let blkSgd := String.join (blocks.map (fun (p, ic, mid, oc, _, _) => blockSgd p ic mid oc))
  let headSgd := sgd "%hW" "%dhW" (ty [128,64,1,1]) ++ sgd "%hb" "%dhb" (ty [128]) ++ sgd "%hg" "%dhg" (ty [128]) ++ sgd "%hbt" "%dhbt" (ty [128])
  let denseSgd := sgd "%Wd" "%dWd" (ty [128,10]) ++ sgd "%bd" "%dbd" (ty [10])
  -- param list (name, type) in allParams order
  let blkParams (p : String) (ic mid oc : Nat) : List (String × String) :=
    [(s!"{p}eW", ty [mid,ic,1,1]), (s!"{p}eb", ty [mid]), (s!"{p}eg", ty [mid]), (s!"{p}ebt", ty [mid]),
     (s!"{p}dW", ty [mid,1,3,3]), (s!"{p}db", ty [mid]), (s!"{p}dg", ty [mid]), (s!"{p}dbt", ty [mid]),
     (s!"{p}pW", ty [oc,mid,1,1]), (s!"{p}pb", ty [oc]), (s!"{p}pg", ty [oc]), (s!"{p}pbt", ty [oc])]
  let allParams : List (String × String) :=
    [("sW", ty [16,3,3,3]), ("sb", ty [16]), ("sg", ty [16]), ("sbt", ty [16])]
    ++ (blocks.map (fun (p, ic, mid, oc, _, _) => blkParams p ic mid oc)).flatten
    ++ [("hW", ty [128,64,1,1]), ("hb", ty [128]), ("hg", ty [128]), ("hbt", ty [128]), ("Wd", ty [128,10]), ("bd", ty [10])]
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS,150528]) :: allParams.map (fun (nm, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @mobilenetv2_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n" ++
    body ++ stemSgd ++ blkSgd ++ headSgd ++ denseSgd ++
    s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered structured MobileNetV2 train step: {mlir.length} chars"
  IO.FS.createDirAll "/tmp/mnv2pc"
  IO.FS.writeFile "/tmp/mnv2pc/train_step.mlir" mlir
  let cargs ← ireeCompileArgs "/tmp/mnv2pc/train_step.mlir" "/tmp/mnv2pc/train_step.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 5000}"
  else
    IO.println "structured MobileNetV2 train step iree-compile OK → /tmp/mnv2pc/train_step.mlir"

#eval main
