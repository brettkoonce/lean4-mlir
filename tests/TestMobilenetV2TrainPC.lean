import LeanMlir.Proofs.MobileNetV2RenderPC
import LeanMlir.ViTRender
import LeanMlir.Types

/-! # Item B — structured MobileNetV2 train-step render (TRUE batch-norm, exact-parity)

The MobileNetV2 peer of `cifarBnTrainStepStructured` (CnnRender.lean). The convs, depthwise, relu6,
residual `addV`, GAP and dense forward + backward are proof-rendered through `pretty` over the tokens
of `mobilenetv2FwdGraphFullPC` (Item A) — forward (`flatConvStridedF`/`flatConvF`/`depthwiseF`/
`depthwiseStridedF`/`relu6F`/`addV`/`gapF`/`denseF`) and backward (`dotOut`, `selectMid`, `convBack`,
`depthwiseBack`/`depthwiseStridedBack`, `addV` residual fan-in).

**BatchNorm is hand-emitted (`bnB`/`bnBackB`), NOT a proof token** — the exact-parity change (matches
r34/enet, see `planning/mnv2_verified.md`). The reference uses TRUE batch-norm (reduce μ/var over
`[0,2,3]`), but the SHlo `.bnBatchF` token has no `pretty`/emit case, so giving mnv2 batch-norm trades
away its proof-rendered-BN property: BN forward+backward become hand-emitted flat↔NCHW fragments
(reshape is a buffer no-op), like the existing hand-emitted gap-backward and conv/depthwise grads.
`bnB` saves x̂/istd/nf/γb + the `[oc]` batch sums; `bnBackB` reuses them and folds dγ/dβ. The adam
step also carries per-layer batch mean/var out in passthrough slots (running-stats BN eval — see
`bnLayers` + `mobilenetv2Verified.bnChannels` + `@mobilenetv2_fwd_eval`).

Full-paper MobileNetV2 (17 inverted-residual blocks, 214 param tensors / ~2.25M scalars) — the layout
`mobilenetv2Verified.toSpecs` / `MobileNetV2Layout.specs` (#guard-locked) the verified-adam driver
trains on (NOTE: `TestMobilenetV2Train.lean`, the committed SGD renderer, is still the reduced 6-block
net — this PC/adam path is the full one).

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

/-- True batch-norm forward, FLAT in/out (reshape to NCHW to reduce μ/var over `[0,2,3]`,
    `nf = BS·H·W`). `x` = flat BN input `[BS, oc·Hh·Ww]`; output flat `%{o}`. Saves NCHW
    `%{o}xh`/`%{o}istd`/`%{o}nf`/`%{o}gb` for the backward + `[oc]` batch sums `%{o}smr`/`%{o}vsr`
    (the running-stats passthrough). HAND-EMITTED, not a proof token: `.bnBatchF` has no `pretty`
    case, so giving mnv2 true batch-norm trades away the proof-rendered-BN property (the convs,
    depthwise, relu6, residual, gap and dense all stay `pretty`-rendered). Same op as r34's `bnPC`. -/
private def bnB (o x g bt : String) (oc Hh Ww : Nat) : String :=
  rs4 s!"%{o}xi" x oc Hh Ww ++
  s!"    %{o}nf = stablehlo.constant dense<{BS*Hh*Ww}.0> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ep = stablehlo.constant dense<{EPS}> : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}smr = stablehlo.reduce(%{o}xi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}sm = stablehlo.broadcast_in_dim %{o}smr, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}mu = stablehlo.divide %{o}sm, %{o}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xc = stablehlo.subtract %{o}xi, %{o}mu : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sq = stablehlo.multiply %{o}xc, %{o}xc : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vsr = stablehlo.reduce(%{o}sq init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}vs = stablehlo.broadcast_in_dim %{o}vsr, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}vr = stablehlo.divide %{o}vs, %{o}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}ve = stablehlo.add %{o}vr, %{o}ep : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}istd = stablehlo.rsqrt %{o}ve : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xh = stablehlo.multiply %{o}xc, %{o}istd : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gb = stablehlo.broadcast_in_dim {g}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}btb = stablehlo.broadcast_in_dim {bt}, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}gx = stablehlo.multiply %{o}xh, %{o}gb : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}n4 = stablehlo.add %{o}gx, %{o}btb : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.reshape %{o}n4 : ({ty [BS,oc,Hh,Ww]}) -> {ty [BS, oc*Hh*Ww]}\n"

/-- Batch-norm backward, FLAT in/out. `bn` = forward BN save-prefix; `dy` = flat upstream cotangent
    `[BS, oc·Hh·Ww]`. Result flat dx `%{o}` + param grads `%{o}dg` (dγ) / `%{o}db` (dβ), both `[oc]`.
    Reuses the NCHW forward saves `%{bn}gb`/`%{bn}xh`/`%{bn}nf`/`%{bn}istd`. Same op as r34's `bnBackPC`. -/
private def bnBackB (o bn dy : String) (oc Hh Ww : Nat) : String :=
  rs4 s!"%{o}dyi" dy oc Hh Ww ++
  s!"    %{o}dxh = stablehlo.multiply %{bn}gb, %{o}dyi : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sdxr = stablehlo.reduce(%{o}dxh init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}sdx = stablehlo.broadcast_in_dim %{o}sdxr, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xd = stablehlo.multiply %{bn}xh, %{o}dxh : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sxdr = stablehlo.reduce(%{o}xd init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}sxd = stablehlo.broadcast_in_dim %{o}sxdr, dims = [1] : ({ty [oc]}) -> {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}t1 = stablehlo.multiply %{o}dxh, %{bn}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}i1 = stablehlo.subtract %{o}t1, %{o}sdx : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}xs = stablehlo.multiply %{bn}xh, %{o}sxd : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}i2 = stablehlo.subtract %{o}i1, %{o}xs : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}sN = stablehlo.divide %{bn}istd, %{bn}nf : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}dxn = stablehlo.multiply %{o}sN, %{o}i2 : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o} = stablehlo.reshape %{o}dxn : ({ty [BS,oc,Hh,Ww]}) -> {ty [BS, oc*Hh*Ww]}\n" ++
  s!"    %{o}dgp = stablehlo.multiply %{o}dyi, %{bn}xh : {ty [BS,oc,Hh,Ww]}\n" ++
  s!"    %{o}dg = stablehlo.reduce(%{o}dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n" ++
  s!"    %{o}db = stablehlo.reduce(%{o}dyi init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

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
  -- (p, ic, mid, oc, s, Hin)  — full paper MobileNetV2
  [("b1",  32,  32,  16, 1, 112),
   ("b2",  16,  96,  24, 2, 112), ("b3",  24, 144,  24, 1, 56),
   ("b4",  24, 144,  32, 2, 56),  ("b5",  32, 192,  32, 1, 28), ("b6",  32, 192,  32, 1, 28),
   ("b7",  32, 192,  64, 2, 28),  ("b8",  64, 384,  64, 1, 14), ("b9",  64, 384,  64, 1, 14), ("b10", 64, 384,  64, 1, 14),
   ("b11", 64, 384,  96, 1, 14),  ("b12", 96, 576,  96, 1, 14), ("b13", 96, 576,  96, 1, 14),
   ("b14", 96, 576, 160, 2, 14),  ("b15",160, 960, 160, 1, 7),  ("b16",160, 960, 160, 1, 7),
   ("b17",160, 960, 320, 1, 7)]

/-- One inverted-residual block forward via `pretty`, capturing flat names. -/
private def fwdBlock (p xin : String) (ic mid oc s Hin : Nat) : StateM Nat (String × FNames) := do
  let Hout := Hin / s
  let (c1, ec) ← pretty BS (.flatConvF (h := Hin) (w := Hin) s!"%{p}eW" s!"%{p}eb" (zK : Kernel4 mid ic 1 1) zV (.operand xin zV))
  let c2 := bnB s!"{p}en" ec s!"%{p}eg" s!"%{p}ebt" mid Hin Hin
  let en := s!"%{p}en"
  let (c3, er) ← pretty BS (.relu6F (.operand en (zV : Vec (mid*Hin*Hin))))
  let (c4, dc) ←
    if s == 2 then pretty BS (.depthwiseStridedF (h := Hout) (w := Hout) s!"%{p}dW" s!"%{p}db" (zD : DepthwiseKernel mid 3 3) zV (.operand er zV))
    else pretty BS (.depthwiseF (h := Hin) (w := Hin) s!"%{p}dW" s!"%{p}db" (zD : DepthwiseKernel mid 3 3) zV (.operand er zV))
  let c5 := bnB s!"{p}dn" dc s!"%{p}dg" s!"%{p}dbt" mid Hout Hout
  let dn := s!"%{p}dn"
  let (c6, dr) ← pretty BS (.relu6F (.operand dn (zV : Vec (mid*Hout*Hout))))
  let (c7, pc) ← pretty BS (.flatConvF (h := Hout) (w := Hout) s!"%{p}pW" s!"%{p}pb" (zK : Kernel4 oc mid 1 1) zV (.operand dr zV))
  let c8 := bnB s!"{p}pn" pc s!"%{p}pg" s!"%{p}pbt" oc Hout Hout
  let pn := s!"%{p}pn"
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
    StateM Nat (String × String × String × String × String) := do
  let Hout := Hin / s
  -- project: bn-back (folds dγ/dβ → %{p}dpndg/%{p}dpndb) → conv-back (1×1)
  let k1 := bnBackB s!"{p}dpn" s!"{p}pn" dy oc Hout Hout
  let cot_pc := s!"%{p}dpn"
  let (k2, cot_dr) ← pretty BS (.convBack (h := Hout) (w := Hout) s!"%{p}pW" (zK : Kernel4 oc mid 1 1) zV zV (.operand cot_pc zV))
  -- depthwise: relu6-mask → bn-back (%{p}ddndg/%{p}ddndb) → depthwise-back
  let (k3, cot_dn) ← pretty BS (.selectMid b.dn (zV : Vec (mid*Hout*Hout)) (.operand cot_dr zV))
  let k4 := bnBackB s!"{p}ddn" s!"{p}dn" cot_dn mid Hout Hout
  let cot_dc := s!"%{p}ddn"
  let (k5, cot_er) ←
    if s == 2 then pretty BS (.depthwiseStridedBack (h := Hout) (w := Hout) s!"%{p}dW" (zD : DepthwiseKernel mid 3 3) zV zV (.operand cot_dc zV))
    else pretty BS (.depthwiseBack (h := Hin) (w := Hin) s!"%{p}dW" (zD : DepthwiseKernel mid 3 3) zV zV (.operand cot_dc zV))
  -- expand: relu6-mask → bn-back (%{p}dendg/%{p}dendb) → conv-back (1×1)
  let (k6, cot_en) ← pretty BS (.selectMid b.en (zV : Vec (mid*Hin*Hin)) (.operand cot_er zV))
  let k7 := bnBackB s!"{p}den" s!"{p}en" cot_en mid Hin Hin
  let cot_ec := s!"%{p}den"
  let (k8, cot_xpre) ← pretty BS (.convBack (h := Hin) (w := Hin) s!"%{p}eW" (zK : Kernel4 mid ic 1 1) zV zV (.operand cot_ec zV))
  let chain := k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7 ++ k8
  if s == 1 && ic == oc then
    let (k9, cot_xin) ← pretty BS (.addV (.operand cot_xpre (zV : Vec (ic*Hin*Hin))) (.operand dy zV))
    pure (chain ++ k9, cot_xin, cot_pc, cot_dc, cot_ec)
  else
    pure (chain, cot_xpre, cot_pc, cot_dc, cot_ec)

/-- block conv/depthwise weight+bias param-grads (hand-emitted), given captured fwd names +
    cotangents. The BN γ/β grads are folded into `bnBackB` (`%{p}d{e,d,p}n{dg,db}`). -/
private def blockParamGrads (p : String) (b : FNames) (cot_pc cot_dc cot_ec : String)
    (ic mid oc s Hin : Nat) : String :=
  let Hout := Hin / s
  -- project (1×1 @ Hout): W/b
  convWGrad s!"%{p}dpW" b.dr cot_pc mid oc Hout Hout 1 ++ biasGrad s!"%{p}dpb" cot_pc oc Hout Hout ++
  -- depthwise (3×3): strided weight-grad upsamples dy first
  (if s == 2 then upsampleFlat s!"%{p}ddu" cot_dc mid Hout Hout ++ dwWGrad s!"%{p}ddW" b.er s!"%{p}ddu" mid (2*Hout) (2*Hout)
   else dwWGrad s!"%{p}ddW" b.er cot_dc mid Hin Hin) ++
  biasGrad s!"%{p}ddb" cot_dc mid Hout Hout ++
  -- expand (1×1 @ Hin, ic→mid): W/b
  convWGrad s!"%{p}deW" b.xin cot_ec ic mid Hin Hin 1 ++ biasGrad s!"%{p}deb" cot_ec mid Hin Hin

/-- per-block SGD over the 12 params (matching `allParams` order/names). -/
private def blockSgd (p : String) (ic mid oc : Nat) : String :=
  sgd s!"%{p}eW" s!"%{p}deW" (ty [mid,ic,1,1]) ++ sgd s!"%{p}eb" s!"%{p}deb" (ty [mid]) ++
  sgd s!"%{p}eg" s!"%{p}dendg" (ty [mid]) ++ sgd s!"%{p}ebt" s!"%{p}dendb" (ty [mid]) ++
  sgd s!"%{p}dW" s!"%{p}ddW" (ty [mid,1,3,3]) ++ sgd s!"%{p}db" s!"%{p}ddb" (ty [mid]) ++
  sgd s!"%{p}dg" s!"%{p}ddndg" (ty [mid]) ++ sgd s!"%{p}dbt" s!"%{p}ddndb" (ty [mid]) ++
  sgd s!"%{p}pW" s!"%{p}dpW" (ty [oc,mid,1,1]) ++ sgd s!"%{p}pb" s!"%{p}dpb" (ty [oc]) ++
  sgd s!"%{p}pg" s!"%{p}dpndg" (ty [oc]) ++ sgd s!"%{p}pbt" s!"%{p}dpndb" (ty [oc])

/-- The proof-rendered fwd + backward-cotangent-chain + hand param grads, SHARED by the SGD
    (`trainStep`) and AdamW (`trainStepAdamSched`) renders. The softmax `sm` `[BS,10]` is captured
    and handed to `cot`, which emits the loss cotangent (and must define `%dy` in scope — plus
    `%loss` for the Adam path). Everything downstream (dense param grads, dense-back) reads `%dy`. -/
private def renderBody (cot : String → String) : String := Id.run do
  let go : StateM Nat String := do
    -- ═══ forward (proof-rendered) ═══
    let (cStemC, stc) ← pretty BS (.flatConvStridedF (h := 112) (w := 112) "%sW" "%sb" (zK : Kernel4 32 3 3 3) zV (.operand "%x" zV))
    let cStemB := bnB "stn" stc "%sg" "%sbt" 32 112 112
    let stn := "%stn"
    let (cStemR, str) ← pretty BS (.relu6F (.operand stn (zV : Vec (32*112*112))))
    -- 17 inverted-residual blocks
    let mut fwd := cStemC ++ cStemB ++ cStemR
    let mut cur := str
    let mut bns : List (FNames × (String × Nat × Nat × Nat × Nat × Nat)) := []
    for blk in blocks do
      let (p, ic, mid, oc, s, Hin) := blk
      let (code, bn) ← fwdBlock p cur ic mid oc s Hin
      fwd := fwd ++ code
      cur := bn.bout
      bns := bns ++ [(bn, blk)]
    -- head: 1×1 conv (320→1280) → bn → relu6 @7
    let (cHc, hc) ← pretty BS (.flatConvF (h := 7) (w := 7) "%hW" "%hb" (zK : Kernel4 1280 320 1 1) zV (.operand cur zV))
    let cHb := bnB "hn" hc "%hg" "%hbt" 1280 7 7
    let hn := "%hn"
    let (cHr, hr) ← pretty BS (.relu6F (.operand hn (zV : Vec (1280*7*7))))
    let (cGap, gap) ← pretty BS (.gapF (c := 1280) (h := 7) (w := 7) (.operand hr zV))
    let (cLog, logits) ← pretty BS (denseF "%Wd" "%bd" (zM : Mat 1280 10) zV (.operand gap zV))
    -- softmax(logits) [BS,10] — `cot` turns it into the loss cotangent `%dy` (+ `%loss`).
    let (cSm, sm) ← pretty BS (.softmaxDiv (.expe (.operand logits (zV : Vec 10))))
    fwd := fwd ++ cHc ++ cHb ++ cHr ++ cGap ++ cLog ++ cSm ++ cot sm
    -- ═══ backward cotangent chain (proof-rendered) ═══
    -- dense-back (dotOut, reads `%dy`) → gap-back (broadcast/÷49) → head relu6 → head bn → head conv
    let (cDg, cotGap) ← pretty BS (.dotOut "%Wd" (zM : Mat 1280 10) (.operand "%dy" zV))
    let mut bwd :=
      cDg ++
      rs4 "%dgi" cotGap 1280 1 1 ++  -- [BS,1280] → [BS,1280,1,1]
      s!"    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : ({ty [BS,1280,1,1]}) -> {ty [BS,1280,7,7]}\n" ++
      s!"    %dgn = stablehlo.constant dense<49.0> : {ty [BS,1280,7,7]}\n" ++
      s!"    %dgd = stablehlo.divide %dgb, %dgn : {ty [BS,1280,7,7]}\n" ++
      s!"    %dgapf = stablehlo.reshape %dgd : ({ty [BS,1280,7,7]}) -> {ty [BS, 1280*7*7]}\n"
    let (cHrB, cot_hn) ← pretty BS (.selectMid hn (zV : Vec (1280*7*7)) (.operand "%dgapf" zV))
    let cHbB := bnBackB "dhn" "hn" cot_hn 1280 7 7
    let cot_hc := "%dhn"
    let (cHcB, cot_b6) ← pretty BS (.convBack (h := 7) (w := 7) "%hW" (zK : Kernel4 1280 320 1 1) zV zV (.operand cot_hc zV))
    bwd := bwd ++ cHrB ++ cHbB ++ cHcB
    -- block backward (reversed), threading the cotangent + accumulating param grads
    let mut paramG := ""
    let mut d := cot_b6
    for (bn, blk) in bns.reverse do
      let (p, ic, mid, oc, s, Hin) := blk
      let (code, dxin, cot_pc, cot_dc, cot_ec) ← bwdBlock p d bn ic mid oc s Hin
      bwd := bwd ++ code
      paramG := paramG ++ blockParamGrads p bn cot_pc cot_dc cot_ec ic mid oc s Hin
      d := dxin
    -- stem backward: relu6 mask → bn-back → (strided 3×3 weight-grad). `d` = cot at stem relu6 out.
    let (cStR, cot_stn) ← pretty BS (.selectMid stn (zV : Vec (32*112*112)) (.operand d zV))
    let cStB := bnBackB "dstn" "stn" cot_stn 32 112 112
    let cot_stc := "%dstn"
    bwd := bwd ++ cStR ++ cStB
    -- ═══ param grads (hand-emitted) ═══
    -- head: 1×1 conv W/b  (BN γ/β folded into bnBackB → %dhndg/%dhndb)
    let headG :=
      convWGrad "%dhW" cur cot_hc 320 1280 7 7 1 ++ biasGrad "%dhb" cot_hc 1280 7 7
    -- stem: strided 3×3 conv W (upsample dy) + b  (BN γ/β → %dstndg/%dstndb)
    let stemG :=
      upsampleFlat "%dsu" cot_stc 32 112 112 ++ convWGrad "%dsW" "%x" "%dsu" 3 32 224 224 3 ++
      biasGrad "%dsb" cot_stc 32 112 112
    -- dense: Wd (gap ⊗ dy), bd (reduce dy)
    let denseG :=
      s!"    %dWd = stablehlo.dot_general {gap}, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,1280]}, {ty [BS,10]}) -> {ty [1280,10]}\n" ++
      s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    pure (fwd ++ bwd ++ paramG ++ headG ++ stemG ++ denseG)
  pure (go.run' 0)

/-- SGD loss cotangent: `%dy = (softmax − onehot)/BS`. -/
private def sgdCot (sm : String) : String :=
  s!"    %dyr = stablehlo.subtract {sm}, %onehot : {ty [BS,10]}\n" ++
  s!"    %dy = stablehlo.divide %dyr, %bsc : {ty [BS,10]}\n"

private def trainStep : String := Id.run do
  let body : String := renderBody sgdCot
  -- ═══ SGD over all params (allParams order) + signature ═══
  let stemSgd := sgd "%sW" "%dsW" (ty [32,3,3,3]) ++ sgd "%sb" "%dsb" (ty [32]) ++ sgd "%sg" "%dstndg" (ty [32]) ++ sgd "%sbt" "%dstndb" (ty [32])
  let blkSgd := String.join (blocks.map (fun (p, ic, mid, oc, _, _) => blockSgd p ic mid oc))
  let headSgd := sgd "%hW" "%dhW" (ty [1280,320,1,1]) ++ sgd "%hb" "%dhb" (ty [1280]) ++ sgd "%hg" "%dhndg" (ty [1280]) ++ sgd "%hbt" "%dhndb" (ty [1280])
  let denseSgd := sgd "%Wd" "%dWd" (ty [1280,10]) ++ sgd "%bd" "%dbd" (ty [10])
  -- param list (name, type) in allParams order
  let blkParams (p : String) (ic mid oc : Nat) : List (String × String) :=
    [(s!"{p}eW", ty [mid,ic,1,1]), (s!"{p}eb", ty [mid]), (s!"{p}eg", ty [mid]), (s!"{p}ebt", ty [mid]),
     (s!"{p}dW", ty [mid,1,3,3]), (s!"{p}db", ty [mid]), (s!"{p}dg", ty [mid]), (s!"{p}dbt", ty [mid]),
     (s!"{p}pW", ty [oc,mid,1,1]), (s!"{p}pb", ty [oc]), (s!"{p}pg", ty [oc]), (s!"{p}pbt", ty [oc])]
  let allParams : List (String × String) :=
    [("sW", ty [32,3,3,3]), ("sb", ty [32]), ("sg", ty [32]), ("sbt", ty [32])]
    ++ (blocks.map (fun (p, ic, mid, oc, _, _) => blkParams p ic mid oc)).flatten
    ++ [("hW", ty [1280,320,1,1]), ("hb", ty [1280]), ("hg", ty [1280]), ("hbt", ty [1280]), ("Wd", ty [1280,10]), ("bd", ty [10])]
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS,150528]) :: allParams.map (fun (nm, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @mobilenetv2_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n" ++
    body ++ stemSgd ++ blkSgd ++ headSgd ++ denseSgd ++
    s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

-- ════════════ AdamW scheduled train step (loss-curve parity with mobilenet-v2-train) ════════════

/-- (paramName, gradName, dims) for all 214 param tensors, in `allParams` order (= `net.specs` order).
    Drives the per-param AdamW update and the packed `[θ|m|v]` signature. The grad names match
    those emitted by `renderBody`'s param-grad section (`%{p}d…` per block, `%d…` for stem/head/dense). -/
private def adamParams : List (String × String × List Nat) :=
  [("%sW", "%dsW", [32,3,3,3]), ("%sb", "%dsb", [32]), ("%sg", "%dstndg", [32]), ("%sbt", "%dstndb", [32])]
  ++ (blocks.map (fun (p, ic, mid, oc, _, _) =>
       [(s!"%{p}eW", s!"%{p}deW", [mid,ic,1,1]), (s!"%{p}eb", s!"%{p}deb", [mid]),
        (s!"%{p}eg", s!"%{p}dendg", [mid]), (s!"%{p}ebt", s!"%{p}dendb", [mid]),
        (s!"%{p}dW", s!"%{p}ddW", [mid,1,3,3]), (s!"%{p}db", s!"%{p}ddb", [mid]),
        (s!"%{p}dg", s!"%{p}ddndg", [mid]), (s!"%{p}dbt", s!"%{p}ddndb", [mid]),
        (s!"%{p}pW", s!"%{p}dpW", [oc,mid,1,1]), (s!"%{p}pb", s!"%{p}dpb", [oc]),
        (s!"%{p}pg", s!"%{p}dpndg", [oc]), (s!"%{p}pbt", s!"%{p}dpndb", [oc])])).flatten
  ++ [("%hW", "%dhW", [1280,320,1,1]), ("%hb", "%dhb", [1280]), ("%hg", "%dhndg", [1280]), ("%hbt", "%dhndb", [1280]),
      ("%Wd", "%dWd", [1280,10]), ("%bd", "%dbd", [10])]

/-- β₁/β₂/ε/wd baked (the mnv2 reference recipe); `%lr`/`%bc1`/`%bc2` arrive as runtime args. -/
private def adamConsts : String :=
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"

/-- AdamW loss cotangent with label smoothing α=0.1 (off-class mass α/K, K=10), plus the in-graph
    smoothed-CE loss `%loss` for logging — same mechanism as `ViTRender.vitTrainStepModuleAdamSched`.
    Defines `%dy` (the cotangent the backward chain reads) and `%loss`. -/
private def adamCot (sm : String) : String :=
  let ls : Float := 0.1
  let lsK : Float := ls / 10.0
  s!"    %dyr0 = stablehlo.subtract {sm}, %onehot : {ty [BS,10]}\n" ++
  s!"    %lsa = stablehlo.constant dense<{ls}> : {ty [BS,10]}\n" ++
  s!"    %lsaoh = stablehlo.multiply %lsa, %onehot : {ty [BS,10]}\n" ++
  s!"    %dyr1 = stablehlo.add %dyr0, %lsaoh : {ty [BS,10]}\n" ++
  s!"    %lsaik = stablehlo.constant dense<{lsK}> : {ty [BS,10]}\n" ++
  s!"    %dyr = stablehlo.subtract %dyr1, %lsaik : {ty [BS,10]}\n" ++
  s!"    %dy = stablehlo.divide %dyr, %bsc : {ty [BS,10]}\n" ++
  s!"    %llog = stablehlo.log {sm} : {ty [BS,10]}\n" ++
  s!"    %ohll = stablehlo.multiply %onehot, %llog : {ty [BS,10]}\n" ++
  s!"    %t1s = stablehlo.reduce(%ohll init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [BS,10]}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %lls = stablehlo.reduce(%llog init: %sc) applies stablehlo.add across dimensions = [1] : ({ty [BS,10]}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    %omac = stablehlo.constant dense<{1.0 - ls}> : {ty [BS]}\n" ++
  s!"    %aKc = stablehlo.constant dense<{lsK}> : {ty [BS]}\n" ++
  s!"    %lt1 = stablehlo.multiply %omac, %t1s : {ty [BS]}\n" ++
  s!"    %lt2 = stablehlo.multiply %aKc, %lls : {ty [BS]}\n" ++
  s!"    %lpe = stablehlo.add %lt1, %lt2 : {ty [BS]}\n" ++
  s!"    %lsum2 = stablehlo.reduce(%lpe init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS]}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    %lbfc = stablehlo.constant dense<{BS}.0> : tensor<f32>\n" ++
  s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
  s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"

/-- BN layers (forward-BN prefix, channels, H·W) in forward order — the running-stats layout shared
    by the train-step batch-stat outputs, the driver's `runningBnStats` buffer, and
    `@mobilenetv2_fwd_eval` inputs. The forward `bnB` saves `%{prefix}smr`/`%{prefix}vsr` (`[oc]` batch
    sums over `[0,2,3]`). Must match `mobilenetv2Verified.bnChannels` and `TestMobilenetV2Fwd`'s eval
    stat order. -/
private def bnLayers : List (String × Nat × Nat) :=
  ("stn", 32, 112*112) ::
  (blocks.flatMap (fun (p, _ic, mid, oc, s, Hin) =>
    let Hout := Hin / s
    [(s!"{p}en", mid, Hin*Hin), (s!"{p}dn", mid, Hout*Hout), (s!"{p}pn", oc, Hout*Hout)]))
  ++ [("hn", 1280, 7*7)]

/-- `@mobilenetv2_adam_train_step` — the proof-rendered (BN now hand-emitted) fwd/bwd/param-grads
    with the SGD update swapped for `ViTRender.emitAdamV` and the `[θ|m|v]` + scalar-tail packed
    signature the generic `VerifiedNet.trainAdamSched` driver expects, EXTENDED with per-BN-layer
    batch mean/var carried out in passthrough slots (running-stats BN; the func also takes matching
    dummy `[oc]` inputs so `#outputs = #inputs`):
    `(x, θ×214, m×214, v×214, lr, bc1, bc2, μ/var×53, onehot) → (θ'×214, m'×214, v'×214, loss, bc1, bc2, μ/var×53)`.
    `lr`/`bc1`/`bc2` runtime; `bc1`/`bc2` + the stat-in slots pass through unchanged. -/
private def trainStepAdamSched : String :=
  let body  := renderBody adamCot
  let names := adamParams.map (fun (nm, _, _) => nm)
  let dims  := adamParams.map (fun (_, _, ds) => ds)
  let updParts := adamParams.map (fun (nm, gr, ds) =>
    ViTRender.emitAdamV nm gr (nm ++ "m") (nm ++ "v") ds (String.ofList (nm.toList.drop 1)))
  let upd := String.join (updParts.map (·.1))
  let thetaN := updParts.map (·.2.1)
  let mN := updParts.map (·.2.2.1)
  let vN := updParts.map (·.2.2.2)
  let psig := String.intercalate ", " ((names.zip dims).map (fun (nm, ds) => s!"{nm}: {ty ds}"))
  let msig := String.intercalate ", " ((names.zip dims).map (fun (nm, ds) => s!"{nm}m: {ty ds}"))
  let vsig := String.intercalate ", " ((names.zip dims).map (fun (nm, ds) => s!"{nm}v: {ty ds}"))
  -- Per-BN-layer batch mean/var = smr/(BS·H·W), vsr/(BS·H·W). Carried out in passthrough slots
  -- (the func also takes 2 dummy `[oc]` inputs per layer so #outputs = #inputs for the generic FFI).
  let statIn := String.intercalate ", " (bnLayers.flatMap (fun (p, oc, _) =>
    [s!"%{p}mui: {ty [oc]}", s!"%{p}vari: {ty [oc]}"]))
  let statCode := String.join (bnLayers.map (fun (p, oc, hw) =>
    s!"    %{p}bnnf = stablehlo.constant dense<{BS*hw}.0> : {ty [oc]}\n" ++
    s!"    %{p}bnmu = stablehlo.divide %{p}smr, %{p}bnnf : {ty [oc]}\n" ++
    s!"    %{p}bnvar = stablehlo.divide %{p}vsr, %{p}bnnf : {ty [oc]}\n"))
  let statOutNames := bnLayers.flatMap (fun (p, _, _) => [s!"%{p}bnmu", s!"%{p}bnvar"])
  let statOutTy := bnLayers.flatMap (fun (_, oc, _) => [ty [oc], ty [oc]])
  let argSig := ("%x: " ++ ty [BS,150528]) ++ ", " ++ psig ++ ", " ++ msig ++ ", " ++ vsig ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, " ++ statIn ++ ", %onehot: " ++ ty [BS,10]
  let allDims := dims ++ dims ++ dims
  let retTy := String.intercalate ", " ((allDims.map (fun ds => ty ds)) ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statOutTy)
  let retVals := String.intercalate ", " (thetaN ++ mN ++ vN ++ ["%loss", "%bc1", "%bc2"] ++ statOutNames)
  "module @m {\n" ++ s!"  func.func @mobilenetv2_adam_train_step({argSig}) -> ({retTy}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n" ++
    adamConsts ++
    body ++ upd ++ statCode ++
    s!"    return {retVals} : {retTy}\n" ++ "  }\n}\n"

/-- iree-compile smoke that degrades gracefully when the compiler isn't on PATH (the render +
    write already happened, so the artifact exists regardless). -/
private def tryCompile (src dst label : String) : IO Unit := do
  try
    let cargs ← ireeCompileArgs src dst
    let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
    if r.exitCode != 0 then IO.eprintln s!"iree-compile ({label}) FAILED:\n{r.stderr.take 5000}"
    else IO.println s!"{label} iree-compile OK → {src}"
  catch e => IO.eprintln s!"iree-compile ({label}) skipped (compiler unavailable): {e}"

def main : IO Unit := do
  IO.FS.createDirAll "/tmp/mnv2pc"
  -- Render + write BOTH artifacts first, then compile — so a missing iree-compile can't abort
  -- before the AdamW artifact is written.
  let mlir := trainStep
  IO.println s!"rendered structured MobileNetV2 train step: {mlir.length} chars"
  IO.FS.writeFile "/tmp/mnv2pc/train_step.mlir" mlir
  let amlir := trainStepAdamSched
  IO.println s!"rendered MobileNetV2 AdamW-sched train step: {amlir.length} chars"
  IO.FS.writeFile "verified_mlir/mobilenetv2_adam_train_step.mlir" amlir
  -- SGD smoke (swap-compatible with the committed train step; verifies the shared `renderBody`).
  tryCompile "/tmp/mnv2pc/train_step.mlir" "/tmp/mnv2pc/train_step.vmfb" "SGD"
  -- AdamW scheduled train step — the artifact `mobilenetv2-verified-adam` trains on.
  tryCompile "verified_mlir/mobilenetv2_adam_train_step.mlir" "/tmp/mnv2pc/adam_train_step.vmfb" "AdamW"

#eval main
