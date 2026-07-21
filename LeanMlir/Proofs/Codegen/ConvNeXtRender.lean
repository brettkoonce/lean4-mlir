import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Types

/-! # ConvNeXt-T train step rendered ENTIRELY from the verified AST (the §1 render)

The ConvNeXt peer of `MobileNetV2Render`/`EfficientNetRender`: the FULL [3,3,9,3] ConvNeXt-T train
step (BS=32, 3×224²→10) rendered as `pretty` of verified `SHlo` nodes — forward, backward-cotangent
chain, AND the param-SGD tail (the new `ConvNeXtFaithfulPoC` ops + the existing conv/depthwise/dense
ops). Adapted from the committed emitter `tests/TestConvNeXtTTrainPC.lean`: its forward + backward
cotangent chain were already `pretty(SHlo)`; here the hand-written param-GRAD strings are replaced by
the SHlo param-SGD ops, which BUNDLE the gradient + SGD wrap into one op (producing the updated param).

Two documented residuals (the only non-`render(provenGraph)` pieces):
* the **stem 4×4/s4 weight** (`psW`) stays a hand-written `patchWGrad`+SGD — there is no stride-4
  weight-grad VJP yet (`flatConvStride4_weight_grad_has_vjp` would be a 4th op + new proof);
* the **scalar-LN γ/β** params render as `tensor<1xf32>` (since the ops output `SHlo 1`), a func-sig
  change vs the committed `tensor<f32>` (the trainer regenerates against this sig).

Every other param (depthwise-7×7 W/b, 1×1 expand/project W/b, per-channel layer-scale γ, scalar-LN
γ/β, downsample 2×2 W/b, dense W/b) denotes the certified loss-descent step (`ConvNeXtFaithfulPoC` +
`ConvNeXtClose`/M2/M3). Render is value-independent (`skel` erases values), so placeholders + `lr:=0`
are passed; the emitted `lrStr`/`epsStr` literals carry the real values. -/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

private def cBS : Nat := 32
private def cEPS : String := "1.0e-6"
private def cLR : String := "0.1"
private def cDepths : Array Nat := #[3, 3, 9, 3]
private def cDims   : Array Nat := #[96, 192, 384, 768]
private def cSpats  : Array Nat := #[56, 28, 14, 7]

private def zK {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0
private def zD {c kh kw : Nat} : DepthwiseKernel c kh kw := fun _ _ _ => 0
private def zV {n : Nat} : Vec n := fun _ => 0
private def zM {a b : Nat} : Mat a b := fun _ _ => 0
private def zT {c h w : Nat} : Tensor3 c h w := fun _ _ _ => 0

-- ── hand-emitted stem-weight grad (the 4×4/s4 patchify — the one documented gap) + SGD wrap ──
private def rs4 (o flatN : String) (Cc Hh Ww : Nat) : String :=
  s!"    {o} = stablehlo.reshape {flatN} : ({ty [cBS, Cc*Hh*Ww]}) -> {ty [cBS,Cc,Hh,Ww]}\n"

private def patchWGrad (o dyFlat : String) : String :=
  rs4 s!"{o}xi" "%x" 3 224 224 ++ rs4 s!"{o}di" dyFlat 96 56 56 ++
  s!"    {o}u = stablehlo.pad {o}di, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : ({ty [cBS,96,56,56]}, tensor<f32>) -> {ty [cBS,96,221,221]}\n" ++
  s!"    {o}xt = stablehlo.transpose {o}xi, dims = [1, 0, 2, 3] : ({ty [cBS,3,224,224]}) -> {ty [3,cBS,224,224]}\n" ++
  s!"    {o}dt = stablehlo.transpose {o}u, dims = [1, 0, 2, 3] : ({ty [cBS,96,221,221]}) -> {ty [96,cBS,221,221]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [3,cBS,224,224]}, {ty [96,cBS,221,221]}) -> {ty [3,96,4,4]}\n" ++
  s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [3,96,4,4]}) -> {ty [96,3,4,4]}\n"

private def sgd (nm t : String) : String :=
  s!"    %{nm}l = stablehlo.constant dense<{cLR}> : {t}\n" ++
  s!"    %{nm}s = stablehlo.multiply %d{nm}, %{nm}l : {t}\n" ++
  s!"    %{nm}n = stablehlo.subtract %{nm}, %{nm}s : {t}\n"

/-- 2×2/s2 downsample weight-grad (the committed even-kernel `convDownWGrad` formulation). Hand-written
    — the even-kernel strided weight grad has no matching VJP-cert SHlo op (the 2nd documented gap). -/
private def downWGrad (o inFlat dyFlat : String) (ci co h2 : Nat) : String :=
  rs4 s!"{o}xi" inFlat ci (2*h2) (2*h2) ++ rs4 s!"{o}di" dyFlat co h2 h2 ++
  s!"    {o}u = stablehlo.pad {o}di, %sc, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : ({ty [cBS,co,h2,h2]}, tensor<f32>) -> {ty [cBS,co,2*h2-1,2*h2-1]}\n" ++
  s!"    {o}xt = stablehlo.transpose {o}xi, dims = [1, 0, 2, 3] : ({ty [cBS,ci,2*h2,2*h2]}) -> {ty [ci,cBS,2*h2,2*h2]}\n" ++
  s!"    {o}dt = stablehlo.transpose {o}u, dims = [1, 0, 2, 3] : ({ty [cBS,co,2*h2-1,2*h2-1]}) -> {ty [co,cBS,2*h2-1,2*h2-1]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}\n" ++
  "      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [ci,cBS,2*h2,2*h2]}, {ty [co,cBS,2*h2-1,2*h2-1]}) -> {ty [ci,co,2,2]}\n" ++
  s!"    {o} = stablehlo.transpose {o}raw, dims = [1, 0, 2, 3] : ({ty [ci,co,2,2]}) -> {ty [co,ci,2,2]}\n"

-- ── captured forward names per ConvNeXt block ──
private structure FNames where
  xin : String   -- block input (residual skip / depthwise in)
  d : String     -- depthwise out (LN in)
  n : String     -- LN out (expand in)
  e : String     -- expand conv out (gelu pre-act)
  g : String     -- gelu out (project in)
  p : String     -- project out (layer-scale in)
  bout : String  -- block out (addV)
  deriving Inhabited

-- ── forward + backward-cotangent block helpers (verbatim pretty(SHlo), from the committed emitter) ──
private def fwdBlock (pfx xin : String) (c e h : Nat) : StateM Nat (String × FNames) := do
  let (k1, d) ← pretty cBS (.depthwiseF (h := h) (w := h) s!"%{pfx}dW" s!"%{pfx}db" (zD : DepthwiseKernel c 7 7) zV (.operand xin zV))
  let (k2, n) ← pretty cBS (.bnF s!"%{pfx}ng" s!"%{pfx}nbt" cEPS 0 0 0 (.operand d (zV : Vec (c*h*h))))
  let (k3, e') ← pretty cBS (.flatConvF (h := h) (w := h) s!"%{pfx}eW" s!"%{pfx}eb" (zK : Kernel4 e c 1 1) zV (.operand n zV))
  let (k4, g) ← pretty cBS (.geluF (.operand e' (zV : Vec (e*h*h))))
  let (k5, p) ← pretty cBS (.flatConvF (h := h) (w := h) s!"%{pfx}pW" s!"%{pfx}pb" (zK : Kernel4 c e 1 1) zV (.operand g zV))
  let (k6, ls) ← pretty cBS (.layerScaleChF (h := h) (w := h) s!"%{pfx}lg" (zV : Vec c) (.operand p zV))
  let (k7, bout) ← pretty cBS (.addV (.operand ls (zV : Vec (c*h*h))) (.operand xin zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, ⟨xin, d, n, e', g, p, bout⟩)

private def bwdBlock (pfx dy : String) (b : FNames) (c e h : Nat) :
    StateM Nat (String × String × String × String × String × String) := do
  let (k1, cot_p) ← pretty cBS (.layerScaleChF (h := h) (w := h) s!"%{pfx}lg" (zV : Vec c) (.operand dy zV))
  let (k2, cot_g) ← pretty cBS (.convBack (h := h) (w := h) s!"%{pfx}pW" (zK : Kernel4 c e 1 1) zV zV (.operand cot_p zV))
  let (k3, cot_e) ← pretty cBS (.geluBack b.e (zV : Vec (e*h*h)) (.operand cot_g zV))
  let (k4, cot_n) ← pretty cBS (.convBack (h := h) (w := h) s!"%{pfx}eW" (zK : Kernel4 e c 1 1) zV zV (.operand cot_e zV))
  let (k5, cot_d) ← pretty cBS (.bnBack s!"%{pfx}ng" b.d cEPS 0 0 zV (.operand cot_n (zV : Vec (c*h*h))))
  let (k6, cot_main) ← pretty cBS (.depthwiseBack (h := h) (w := h) s!"%{pfx}dW" (zD : DepthwiseKernel c 7 7) zV zV (.operand cot_d zV))
  let (k7, cot_xin) ← pretty cBS (.addV (.operand cot_main (zV : Vec (c*h*h))) (.operand dy zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, cot_xin, cot_p, cot_e, cot_n, cot_d)

private def fwdDown (pfx xin : String) (ci co h2 : Nat) : StateM Nat (String × String × String) := do
  let (k1, n) ← pretty cBS (.bnF s!"%{pfx}ng" s!"%{pfx}nbt" cEPS 0 0 0 (.operand xin (zV : Vec (ci*(2*h2)*(2*h2)))))
  let (k2, o) ← pretty cBS (.flatConvStridedF (h := h2) (w := h2) s!"%{pfx}W" s!"%{pfx}b" (zK : Kernel4 co ci 2 2) zV (.operand n zV))
  pure (k1 ++ k2, n, o)

private def bwdDown (pfx dy xin : String) (ci co h2 : Nat) :
    StateM Nat (String × String × String) := do
  let (k1, cot_n) ← pretty cBS (.convStridedBack (h := h2) (w := h2) s!"%{pfx}W" (zK : Kernel4 co ci 2 2) zV zV (.operand dy (zV : Vec (co*h2*h2))))
  let (k2, cot_x) ← pretty cBS (.bnBack s!"%{pfx}ng" xin cEPS 0 0 (zV : Vec (ci*(2*h2)*(2*h2))) (.operand cot_n zV))
  pure (k1 ++ k2, cot_n, cot_x)

-- ── NEW: param-SGD via the SHlo ops (the updated param IS the op output) ──
private def blockParamSgd (pfx : String) (b : FNames) (cot_p cot_e cot_n cot_d dy : String) (c e h : Nat) :
    StateM Nat (String × List (String × String)) := do
  let (cLg, nLg) ← pretty cBS (.layerScaleChGammaSgd s!"%{pfx}lg" b.p cLR (zV : Vec (c*h*h)) (zV : Vec c) 0 (.operand dy zV))
  let (cPw, nPw) ← pretty cBS (.convWeightSgd b.g s!"%{pfx}pW" cLR (zV : Vec c) (zT : Tensor3 e h h) (zK : Kernel4 c e 1 1) 0 (.operand cot_p zV))
  let (cPb, nPb) ← pretty cBS (.convBiasSgd s!"%{pfx}pb" cLR (zK : Kernel4 c e 1 1) (zT : Tensor3 e h h) (zV : Vec c) 0 (.operand cot_p zV))
  let (cEw, nEw) ← pretty cBS (.convWeightSgd b.n s!"%{pfx}eW" cLR (zV : Vec e) (zT : Tensor3 c h h) (zK : Kernel4 e c 1 1) 0 (.operand cot_e zV))
  let (cEb, nEb) ← pretty cBS (.convBiasSgd s!"%{pfx}eb" cLR (zK : Kernel4 e c 1 1) (zT : Tensor3 c h h) (zV : Vec e) 0 (.operand cot_e zV))
  let (cNg, nNg) ← pretty cBS (.lnGammaSgd s!"%{pfx}ng" b.d cEPS cLR 0 (zV : Vec (c*h*h)) (zV : Vec 1) 0 (.operand cot_n zV))
  let (cNb, nNb) ← pretty cBS (.lnBetaSgd s!"%{pfx}nbt" cLR (zV : Vec 1) 0 (.operand cot_n (zV : Vec (c*h*h))))
  let (cDw, nDw) ← pretty cBS (.depthwiseWeightSgd b.xin s!"%{pfx}dW" cLR (zV : Vec c) (zT : Tensor3 c h h) (zD : DepthwiseKernel c 7 7) 0 (.operand cot_d zV))
  let (cDb, nDb) ← pretty cBS (.depthwiseBiasSgd s!"%{pfx}db" cLR (zD : DepthwiseKernel c 7 7) (zT : Tensor3 c h h) (zV : Vec c) 0 (.operand cot_d zV))
  pure (cLg ++ cPw ++ cPb ++ cEw ++ cEb ++ cNg ++ cNb ++ cDw ++ cDb,
    [(s!"{pfx}dW", nDw), (s!"{pfx}db", nDb), (s!"{pfx}ng", nNg), (s!"{pfx}nbt", nNb),
     (s!"{pfx}eW", nEw), (s!"{pfx}eb", nEb), (s!"{pfx}pW", nPw), (s!"{pfx}pb", nPb), (s!"{pfx}lg", nLg)])

private def downParamSgd (pfx downLn downIn cot_n dy : String) (ci co h2 : Nat) :
    StateM Nat (String × List (String × String)) := do
  -- dXb (convStridedBiasSgd, channel-sum) + dXng/dXnbt (SHlo ln ops); dXW hand-written (even-kernel gap)
  let (cB, nB) ← pretty cBS (.convStridedBiasSgd s!"%{pfx}b" cLR (zK : Kernel4 co ci 2 2) (zV : Vec (ci*(2*h2)*(2*h2))) (zV : Vec co) 0 (.operand dy zV))
  let (cNg, nNg) ← pretty cBS (.lnGammaSgd s!"%{pfx}ng" downIn cEPS cLR 0 (zV : Vec (ci*(2*h2)*(2*h2))) (zV : Vec 1) 0 (.operand cot_n zV))
  let (cNb, nNb) ← pretty cBS (.lnBetaSgd s!"%{pfx}nbt" cLR (zV : Vec 1) 0 (.operand cot_n (zV : Vec (ci*(2*h2)*(2*h2)))))
  let wcode := downWGrad s!"%d{pfx}W" downLn dy ci co h2 ++ sgd s!"{pfx}W" (ty [co, ci, 2, 2])
  pure (cB ++ cNg ++ cNb ++ wcode,
    [(s!"{pfx}ng", nNg), (s!"{pfx}nbt", nNb), (s!"{pfx}W", s!"%{pfx}Wn"), (s!"{pfx}b", nB)])

-- ── full param signature (committed forward order; LN params now tensor<1xf32>) ──
private def blkParams (pfx : String) (c e : Nat) : List (String × String) :=
  [(s!"{pfx}dW", ty [c,1,7,7]), (s!"{pfx}db", ty [c]),
   (s!"{pfx}ng", "tensor<f32>"), (s!"{pfx}nbt", "tensor<f32>"),
   (s!"{pfx}eW", ty [e,c,1,1]), (s!"{pfx}eb", ty [e]),
   (s!"{pfx}pW", ty [c,e,1,1]), (s!"{pfx}pb", ty [c]),
   (s!"{pfx}lg", ty [c])]

private def allParams : List (String × String) := Id.run do
  let mut ps : List (String × String) := [("psW", ty [96,3,4,4]), ("psb", ty [96])]
  for si in [0:4] do
    let c := cDims[si]!
    let e := 4 * c
    for j in [0:cDepths[si]!] do
      ps := ps ++ blkParams s!"s{si}b{j}" c e
    if si < 3 then
      ps := ps ++ [(s!"d{si}ng", "tensor<f32>"), (s!"d{si}nbt", "tensor<f32>"),
                   (s!"d{si}W", ty [cDims[si+1]!, c, 2, 2]), (s!"d{si}b", ty [cDims[si+1]!])]
  ps := ps ++ [("hng", "tensor<f32>"), ("hnbt", "tensor<f32>"), ("Wd", ty [768,10]), ("bd", ty [10])]
  return ps

-- ════════════════════════════════════════════════════════════════
-- § The whole-net renderer
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 8000 in
/-- **ConvNeXt-T (full [3,3,9,3]) train step rendered ENTIRELY from the verified AST** (except the
    stem 4×4/s4 weight — the documented stride-4-weight-grad gap). Every other line is `pretty` of a
    verified `SHlo` node: forward + backward cotangent chain + the param-SGD ops (the updated param is
    each op's output). -/
def convNextTrainStepFaithfulV (funcName : String := "convnext_train_step") : String := Id.run do
  let go : StateM Nat (String × List (String × String)) := do
    -- ═══ forward ═══
    let (cS, stem) ← pretty cBS (.flatConvStride4F (h := 56) (w := 56) "%psW" "%psb"
      (zK : Kernel4 96 3 4 4) zV (.operand "%x" (zV : Vec (3*(2*(2*56))*(2*(2*56))))))
    let mut fwd := cS
    let mut cur := stem
    let mut blksAll : Array (Array FNames) := #[]
    let mut downLn : Array String := #[]
    let mut downIn : Array String := #[]
    for si in [0:4] do
      let c := cDims[si]!; let e := 4 * c; let h := cSpats[si]!
      let mut blks : Array FNames := #[]
      for j in [0:cDepths[si]!] do
        let (code, bn) ← fwdBlock s!"s{si}b{j}" cur c e h
        fwd := fwd ++ code; cur := bn.bout; blks := blks.push bn
      blksAll := blksAll.push blks
      if si < 3 then
        downIn := downIn.push cur
        let (code, n, o) ← fwdDown s!"d{si}" cur c cDims[si+1]! cSpats[si+1]!
        fwd := fwd ++ code; downLn := downLn.push n; cur := o
    let (cG, gap) ← pretty cBS (.gapF (c := 768) (h := 7) (w := 7) (.operand cur zV))
    let (cHn, hn) ← pretty cBS (.bnF "%hng" "%hnbt" cEPS 0 0 0 (.operand gap (zV : Vec 768)))
    let (cLog, logits) ← pretty cBS (denseF "%Wd" "%bd" (zM : Mat 768 10) zV (.operand hn zV))
    let (cSub, dyr) ← pretty cBS (.sub (.softmaxDiv (.expe (.operand logits (zV : Vec 10)))) (.operand "%onehot" zV))
    fwd := fwd ++ cG ++ cHn ++ cLog ++ cSub
    -- ═══ backward: head cotangent chain + param-SGD ═══
    let (cDd, cot_hn) ← pretty cBS (.dotOut "%Wd" (zM : Mat 768 10) (.operand "%dy" zV))
    let (cHnB, cot_gap) ← pretty cBS (.bnBack "%hng" gap cEPS 0 0 zV (.operand cot_hn (zV : Vec 768)))
    let (cWd, nWd) ← pretty cBS (.weightSgd hn "%Wd" cLR (zV : Vec 768) (zM : Mat 768 10) 0 (.operand "%dy" zV))
    let (cBd, nBd) ← pretty cBS (.biasSgd "%bd" cLR (zV : Vec 10) 0 (.operand "%dy" zV))
    let (cHg, nHg) ← pretty cBS (.lnGammaSgd "%hng" gap cEPS cLR 0 (zV : Vec 768) (zV : Vec 1) 0 (.operand cot_hn zV))
    let (cHb, nHb) ← pretty cBS (.lnBetaSgd "%hnbt" cLR (zV : Vec 1) 0 (.operand cot_hn (zV : Vec 768)))
    let mut updMap : List (String × String) :=
      [("hng", nHg), ("hnbt", nHb), ("Wd", nWd), ("bd", nBd)]
    let mut bwd :=
      s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [cBS, 10]}\n" ++ cDd ++ cHnB ++
      cWd ++ cBd ++ cHg ++ cHb ++
      s!"    %dgi = stablehlo.reshape {cot_gap} : ({ty [cBS,768]}) -> {ty [cBS,768,1,1]}\n" ++
      s!"    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : ({ty [cBS,768,1,1]}) -> {ty [cBS,768,7,7]}\n" ++
      s!"    %dgn = stablehlo.constant dense<49.0> : {ty [cBS,768,7,7]}\n" ++
      s!"    %dgd = stablehlo.divide %dgb, %dgn : {ty [cBS,768,7,7]}\n" ++
      s!"    %dgapf = stablehlo.reshape %dgd : ({ty [cBS,768,7,7]}) -> {ty [cBS, 768*7*7]}\n"
    let mut dy := "%dgapf"
    for si' in [0:4] do
      let si := 3 - si'
      let c := cDims[si]!; let e := 4 * c; let h := cSpats[si]!
      for j' in [0:cDepths[si]!] do
        let j := cDepths[si]! - 1 - j'
        let b := (blksAll[si]!)[j]!
        let (code, cot_xin, cot_p, cot_e, cot_n, cot_d) ← bwdBlock s!"s{si}b{j}" dy b c e h
        let (pcode, pairs) ← blockParamSgd s!"s{si}b{j}" b cot_p cot_e cot_n cot_d dy c e h
        bwd := bwd ++ code ++ pcode; updMap := updMap ++ pairs; dy := cot_xin
      if si > 0 then
        let ci := cDims[si-1]!; let h2 := cSpats[si]!
        let (code, cot_n, cot_x) ← bwdDown s!"d{si-1}" dy (downIn[si-1]!) ci c h2
        let (pcode, pairs) ← downParamSgd s!"d{si-1}" (downLn[si-1]!) (downIn[si-1]!) cot_n dy ci c h2
        bwd := bwd ++ code ++ pcode; updMap := updMap ++ pairs; dy := cot_x
    -- stem: psb via convBiasSgd (channel-sum, correct); psW hand-written (the stride-4 gap) + SGD
    let (cPsb, nPsb) ← pretty cBS (.convBiasSgd "%psb" cLR (zK : Kernel4 96 3 4 4) (zT : Tensor3 3 56 56) (zV : Vec 96) 0 (.operand dy zV))
    bwd := bwd ++ patchWGrad "%dpsW" dy ++ sgd "psW" (ty [96,3,4,4]) ++ cPsb
    updMap := updMap ++ [("psW", "%psWn"), ("psb", nPsb)]
    pure (fwd ++ bwd, updMap)
  let (body, updMap) := go.run' 0
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [cBS, 3*224*224]) :: allParams.map (fun (nm, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [cBS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, t) => t))
  let retVals := String.intercalate ", "
    (allParams.map (fun (nm, _) => (updMap.lookup nm).getD s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{cBS}.0> : {ty [cBS,10]}\n" ++
    body ++
    s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

end Proofs.StableHLO

-- Regenerate verified_mlir/convnext_train_step.mlir from the faithful renderer (BS=32, ε=1e-6, lr=0.1).
#eval IO.FS.writeFile "verified_mlir/convnext_train_step.mlir"
  (Proofs.StableHLO.convNextTrainStepFaithfulV "convnext_train_step")
