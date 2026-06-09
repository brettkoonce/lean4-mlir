import LeanMlir.Proofs.StableHLO
import LeanMlir.Types

/-! # ConvNeXt Item B — structured representative train-step render (proof-rendered)

The ConvNeXt peer of `tests/TestMobilenetV2TrainPC.lean` / `TestResnet34TrainPC.lean`, at the
**representative** `convNextForward` config (the proven graph: 1×1 patchify stem → scalar-LN →
2 residual ConvNeXt blocks → GAP → head-LN → dense), CIFAR-shaped: 3×32² in, c=32, cExp=128,
dw 7×7, 10 classes. Forward AND the whole backward cotangent chain are proof-rendered through
`pretty` over the very tokens of `convNextFwdGraph` (Item A) — forward (`flatConvF`/`bnF`/
`depthwiseF`/`geluF`/`layerScaleF`/`addV`/`gapF`/`denseF`) and backward (`dotOut`, `bnBack`
(scalar-LN input-VJP), `geluBack`, `convBack`, `depthwiseBack`, `addV` residual fan-in). The
layer-scale backward IS the forward token applied to the cotangent (`layerScale`'s input-VJP is
`γ ⊙ dy` — diagonal/symmetric), so no new backward token. Only the no-SHlo-constructor pieces
are hand-emitted: the GAP backward, conv/depthwise/dense weight+bias grads, layer-scale
`dγ = Σ_b x⊙dy` (`layerScale_grad_gamma`/`cnx_render_lsgamma_certified`), and scalar-LN
`dγ = Σ dy·x̂`, `dβ = Σ dy` (`bn_grad_gamma/beta`, certified via the `Vec 1` embedding in
`ConvNeXtClose.lean`); reshape glue (flat→NCHW) only at those.

Unlike the MNV2/r34 peers there is no committed same-signature renderer (the committed
`TestConvNeXtTrain.lean` is the full ConvNeXt-T [3,3,9,3]; "come back to scaling later"), so
validation is the `scripts/render_parity.py` ref-only smoke: compile + run on the GPU, all 26
updated params finite and non-zero:
  `scripts/render_parity.py --fn convnext_rep_train_step --ref /tmp/cnxpc/train_step.mlir`

Run: `IREE_BACKEND=rocm lake env lean tests/TestConvNeXtTrainPC.lean`
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def IC : Nat := 3    -- input channels
private def C : Nat := 32    -- block width
private def CE : Nat := 128  -- expanded width (4×)
private def H : Nat := 32    -- spatial (stride-1 everywhere: blocks keep resolution)
private def EPS : String := "1.0e-6"
private def LR : String := "0.1"

-- placeholder values (pretty/emitTok render names only; values are irrelevant)
private def zK {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0
private def zD {c kh kw : Nat} : DepthwiseKernel c kh kw := fun _ _ _ => 0
private def zV {n : Nat} : Vec n := fun _ => 0
private def zM {a b : Nat} : Mat a b := fun _ _ => 0

-- ════════════ hand-emitted tail templates (param grads; NCHW only here) ════════════

/-- flat → NCHW reshape. -/
private def rs4 (o flatN : String) (Cc Hh Ww : Nat) : String :=
  s!"    {o} = stablehlo.reshape {flatN} : ({ty [BS, Cc*Hh*Ww]}) -> {ty [BS,Cc,Hh,Ww]}\n"

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

/-- Depthwise k×k weight-grad (batch_group_count=c), inputs flat. -/
private def dwWGrad (o inpFlat dyFlat : String) (c Hh Ww k : Nat) : String :=
  rs4 s!"{o}xi" inpFlat c Hh Ww ++ rs4 s!"{o}di" dyFlat c Hh Ww ++
  s!"    {o}xt = stablehlo.transpose {o}xi, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    {o}dt = stablehlo.transpose {o}di, dims = [1, 0, 2, 3] : ({ty [BS,c,Hh,Ww]}) -> {ty [c,BS,Hh,Ww]}\n" ++
  s!"    {o}raw = stablehlo.convolution({o}xt, {o}dt)\n" ++
  "      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n" ++
  "      window = {" ++ s!"stride = [1, 1], pad = [[{(k-1)/2}, {(k-1)/2}], [{(k-1)/2}, {(k-1)/2}]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]" ++ "}\n" ++
  "      {batch_group_count = " ++ toString c ++ " : i64, feature_group_count = 1 : i64}" ++
  s!" : ({ty [c,BS,Hh,Ww]}, {ty [c,BS,Hh,Ww]}) -> {ty [1,c,k,k]}\n" ++
  s!"    {o} = stablehlo.reshape {o}raw : ({ty [1,c,k,k]}) -> {ty [c,1,k,k]}\n"

/-- conv/depthwise bias-grad: reduce flat cotangent over batch+spatial → [oc]. -/
private def biasGrad (o dyFlat : String) (oc Hh Ww : Nat) : String :=
  rs4 s!"{o}i" dyFlat oc Hh Ww ++
  s!"    {o} = stablehlo.reduce({o}i init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : ({ty [BS,oc,Hh,Ww]}, tensor<f32>) -> {ty [oc]}\n"

/-- layer-scale γ-grad `dγ = Σ_b x ⊙ dy` (per-element over the flat c·h·w map):
    the rendered form of `layerScale_grad_gamma` (certified `cnx_render_lsgamma_certified`). -/
private def lsGrad (o xFlat dyFlat : String) (n : Nat) : String :=
  s!"    {o}p = stablehlo.multiply {xFlat}, {dyFlat} : {ty [BS, n]}\n" ++
  s!"    {o} = stablehlo.reduce({o}p init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS, n]}, tensor<f32>) -> {ty [n]}\n"

/-- scalar-LN dγ = Σ_{b,k} dy·x̂, dβ = Σ_{b,k} dy; recompute x̂ from the saved LN input
    `inFlat` (per-example mean/var over [1] — `bnF`'s own emission text). The rendered
    `bn_grad_gamma`/`bn_grad_beta` (certified `cnx_render_ln{gamma,beta}_certified`). -/
private def lnParamGrad (dgr dbe inFlat dyFlat : String) (n : Nat) : String :=
  let tn := ty [BS, n]
  s!"    {dgr}nf = stablehlo.constant dense<{n}.0> : {tn}\n" ++
  s!"    {dgr}ep = stablehlo.constant dense<{EPS}> : {tn}\n" ++
  s!"    {dgr}smr = stablehlo.reduce({inFlat} init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    {dgr}sm = stablehlo.broadcast_in_dim {dgr}smr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    {dgr}mu = stablehlo.divide {dgr}sm, {dgr}nf : {tn}\n" ++
  s!"    {dgr}xc = stablehlo.subtract {inFlat}, {dgr}mu : {tn}\n" ++
  s!"    {dgr}sq = stablehlo.multiply {dgr}xc, {dgr}xc : {tn}\n" ++
  s!"    {dgr}vsr = stablehlo.reduce({dgr}sq init: %sc) applies stablehlo.add across dimensions = [1] : ({tn}, tensor<f32>) -> {ty [BS]}\n" ++
  s!"    {dgr}vs = stablehlo.broadcast_in_dim {dgr}vsr, dims = [0] : ({ty [BS]}) -> {tn}\n" ++
  s!"    {dgr}vr = stablehlo.divide {dgr}vs, {dgr}nf : {tn}\n" ++
  s!"    {dgr}ve = stablehlo.add {dgr}vr, {dgr}ep : {tn}\n" ++
  s!"    {dgr}istd = stablehlo.rsqrt {dgr}ve : {tn}\n" ++
  s!"    {dgr}xh = stablehlo.multiply {dgr}xc, {dgr}istd : {tn}\n" ++
  s!"    {dgr}p = stablehlo.multiply {dyFlat}, {dgr}xh : {tn}\n" ++
  s!"    {dgr} = stablehlo.reduce({dgr}p init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tn}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    {dbe} = stablehlo.reduce({dyFlat} init: %sc) applies stablehlo.add across dimensions = [0, 1] : ({tn}, tensor<f32>) -> tensor<f32>\n"

private def sgd (θ dθ ty' : String) : String :=
  s!"    {θ}l = stablehlo.constant dense<{LR}> : {ty'}\n" ++
  s!"    {θ}s = stablehlo.multiply {dθ}, {θ}l : {ty'}\n" ++
  s!"    {θ}n = stablehlo.subtract {θ}, {θ}s : {ty'}\n"

-- ════════════ captured forward names per ConvNeXt block ════════════
private structure FNames where  -- all flat SSA names from `pretty`
  xin : String   -- block input (= the residual skip)
  d : String     -- depthwise out (LN input)
  n : String     -- LN out (expand in)
  e : String     -- expand conv out (gelu pre-activation)
  g : String     -- gelu out (project in)
  p : String     -- project out (layer-scale in)
  ls : String    -- layer-scale out
  bout : String  -- block out (addV)

/-- One ConvNeXt block forward via `pretty` — exactly the `convNextFwdGraph` block tokens
    (`b{i}Body` + `addV` skip), with the graph's own param names. -/
private def fwdBlock (i : Nat) (xin : String) : StateM Nat (String × FNames) := do
  let (k1, d) ← pretty BS (.depthwiseF (h := H) (w := H) s!"%Wdw{i}" s!"%bdw{i}" (zD : DepthwiseKernel C 7 7) zV (.operand xin zV))
  let (k2, n) ← pretty BS (.bnF s!"%gn{i}" s!"%btn{i}" EPS 0 0 0 (.operand d (zV : Vec (C*H*H))))
  let (k3, e) ← pretty BS (.flatConvF (h := H) (w := H) s!"%Wex{i}" s!"%bex{i}" (zK : Kernel4 CE C 1 1) zV (.operand n zV))
  let (k4, g) ← pretty BS (.geluF (.operand e (zV : Vec (CE*H*H))))
  let (k5, p) ← pretty BS (.flatConvF (h := H) (w := H) s!"%Wpr{i}" s!"%bpr{i}" (zK : Kernel4 C CE 1 1) zV (.operand g zV))
  let (k6, ls) ← pretty BS (.layerScaleF s!"%gls{i}" (zV : Vec (C*H*H)) (.operand p zV))
  let (k7, bout) ← pretty BS (.addV (.operand ls (zV : Vec (C*H*H))) (.operand xin zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, ⟨xin, d, n, e, g, p, ls, bout⟩)

/-- One ConvNeXt block backward cotangent chain via `pretty`. `dy` = flat cotangent at block
    output (= at layer-scale output: the residual add passes it through). The layer-scale
    backward is `layerScaleF` itself applied to the cotangent (input-VJP `γ ⊙ dy`). Returns
    (code, cot-at-block-input, cot_p, cot_e, cot_n, cot_d) — what the param grads need. -/
private def bwdBlock (i : Nat) (dy : String) (b : FNames) :
    StateM Nat (String × String × String × String × String × String) := do
  let (k1, cot_p) ← pretty BS (.layerScaleF s!"%gls{i}" (zV : Vec (C*H*H)) (.operand dy zV))
  let (k2, cot_g) ← pretty BS (.convBack (h := H) (w := H) s!"%Wpr{i}" (zK : Kernel4 C CE 1 1) zV zV (.operand cot_p zV))
  let (k3, cot_e) ← pretty BS (.geluBack b.e (zV : Vec (CE*H*H)) (.operand cot_g zV))
  let (k4, cot_n) ← pretty BS (.convBack (h := H) (w := H) s!"%Wex{i}" (zK : Kernel4 CE C 1 1) zV zV (.operand cot_e zV))
  let (k5, cot_d) ← pretty BS (.bnBack s!"%gn{i}" b.d EPS 0 0 zV (.operand cot_n (zV : Vec (C*H*H))))
  let (k6, cot_main) ← pretty BS (.depthwiseBack (h := H) (w := H) s!"%Wdw{i}" (zD : DepthwiseKernel C 7 7) zV zV (.operand cot_d zV))
  let (k7, cot_xin) ← pretty BS (.addV (.operand cot_main (zV : Vec (C*H*H))) (.operand dy zV))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, cot_xin, cot_p, cot_e, cot_n, cot_d)

/-- block param-grad text (hand-emitted), given captured fwd names + cotangents.
    `dy` = cot at block output (= at layer-scale output, the x⊙dy partner for dγ_ls). -/
private def blockParamGrads (i : Nat) (b : FNames) (cot_p cot_e cot_n cot_d dy : String) : String :=
  -- layer-scale: dγ = Σ_b p ⊙ dy
  lsGrad s!"%dgls{i}" b.p dy (C*H*H) ++
  -- project 1×1 (CE→C): W/b
  convWGrad s!"%dWpr{i}" b.g cot_p CE C H H 1 ++ biasGrad s!"%dbpr{i}" cot_p C H H ++
  -- expand 1×1 (C→CE): W/b
  convWGrad s!"%dWex{i}" b.n cot_e C CE H H 1 ++ biasGrad s!"%dbex{i}" cot_e CE H H ++
  -- scalar-LN: dγ/dβ from the saved depthwise out
  lnParamGrad s!"%dgn{i}" s!"%dbtn{i}" b.d cot_n (C*H*H) ++
  -- depthwise 7×7: W/b
  dwWGrad s!"%dWdw{i}" b.xin cot_d C H H 7 ++ biasGrad s!"%dbdw{i}" cot_d C H H

/-- per-block param (name, type) list, forward order (matches `convNextFwdGraph` arg order). -/
private def blkParams (i : Nat) : List (String × String) :=
  [(s!"Wdw{i}", ty [C,1,7,7]), (s!"bdw{i}", ty [C]), (s!"gn{i}", "tensor<f32>"), (s!"btn{i}", "tensor<f32>"),
   (s!"Wex{i}", ty [CE,C,1,1]), (s!"bex{i}", ty [CE]), (s!"Wpr{i}", ty [C,CE,1,1]), (s!"bpr{i}", ty [C]),
   (s!"gls{i}", ty [C*H*H])]

private def trainStep : String := Id.run do
  let go : StateM Nat String := do
    -- ═══ forward (proof-rendered; the convNextFwdGraph tokens in graph order) ═══
    let (cP, patch) ← pretty BS (.flatConvF (h := H) (w := H) "%Wst" "%bst" (zK : Kernel4 C IC 1 1) zV (.operand "%x" zV))
    let (cL, stemLn) ← pretty BS (.bnF "%gst" "%btst" EPS 0 0 0 (.operand patch (zV : Vec (C*H*H))))
    let (cB1, b1) ← fwdBlock 1 stemLn
    let (cB2, b2) ← fwdBlock 2 b1.bout
    let (cG, gap) ← pretty BS (.gapF (c := C) (h := H) (w := H) (.operand b2.bout zV))
    let (cHn, hn) ← pretty BS (.bnF "%ghd" "%bthd" EPS 0 0 0 (.operand gap (zV : Vec C)))
    let (cLog, logits) ← pretty BS (denseF "%Wd" "%bd" (zM : Mat C 10) zV (.operand hn zV))
    -- loss cotangent: (softmax(logits) − onehot)/BS
    let (cSub, dyr) ← pretty BS (.sub (.softmaxDiv (.expe (.operand logits (zV : Vec 10)))) (.operand "%onehot" (zV : Vec 10)))
    let fwd := cP ++ cL ++ cB1 ++ cB2 ++ cG ++ cHn ++ cLog ++ cSub
    -- ═══ backward cotangent chain (proof-rendered) ═══
    -- dy = dyr/BS ; dense-back (dotOut) → head-LN back (bnBack) → GAP back (hand) → blocks
    let (cDd, cot_hn) ← pretty BS (.dotOut "%Wd" (zM : Mat C 10) (.operand "%dy" zV))
    let (cHnB, cot_gap) ← pretty BS (.bnBack "%ghd" gap EPS 0 0 zV (.operand cot_hn (zV : Vec C)))
    let mut bwd :=
      s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [BS, 10]}\n" ++ cDd ++ cHnB ++
      rs4 "%dgi" cot_gap C 1 1 ++  -- [BS,C] → [BS,C,1,1]
      s!"    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : ({ty [BS,C,1,1]}) -> {ty [BS,C,H,H]}\n" ++
      s!"    %dgn = stablehlo.constant dense<{H*H}.0> : {ty [BS,C,H,H]}\n" ++
      s!"    %dgd = stablehlo.divide %dgb, %dgn : {ty [BS,C,H,H]}\n" ++
      s!"    %dgapf = stablehlo.reshape %dgd : ({ty [BS,C,H,H]}) -> {ty [BS, C*H*H]}\n"
    let (cB2b, cot_b1out, cot_p2, cot_e2, cot_n2, cot_d2) ← bwdBlock 2 "%dgapf" b2
    let (cB1b, cot_stemLn, cot_p1, cot_e1, cot_n1, cot_d1) ← bwdBlock 1 cot_b1out b1
    -- stem-LN back (input-VJP; the patchify conv is the first layer — no input grad)
    let (cSLb, cot_patch) ← pretty BS (.bnBack "%gst" patch EPS 0 0 zV (.operand cot_stemLn (zV : Vec (C*H*H))))
    bwd := bwd ++ cB2b ++ cB1b ++ cSLb
    -- ═══ param grads (hand-emitted) ═══
    let paramG :=
      blockParamGrads 2 b2 cot_p2 cot_e2 cot_n2 cot_d2 "%dgapf" ++
      blockParamGrads 1 b1 cot_p1 cot_e1 cot_n1 cot_d1 cot_b1out ++
      -- stem: LN γ/β (saved patchify out) + 1×1 conv W/b
      lnParamGrad "%dgst" "%dbtst" patch cot_stemLn (C*H*H) ++
      convWGrad "%dWst" "%x" cot_patch IC C H H 1 ++ biasGrad "%dbst" cot_patch C H H ++
      -- head LN γ/β (saved GAP out, n = C) + dense Wd (hn ⊗ dy), bd (reduce dy)
      lnParamGrad "%dghd" "%dbthd" gap cot_hn C ++
      s!"    %dWd = stablehlo.dot_general {hn}, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : ({ty [BS,C]}, {ty [BS,10]}) -> {ty [C,10]}\n" ++
      s!"    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : ({ty [BS,10]}, tensor<f32>) -> {ty [10]}\n"
    pure (fwd ++ bwd ++ paramG)
  let body : String := go.run' 0
  -- ═══ SGD over all 26 params (forward order) + signature ═══
  let allParams : List (String × String) :=
    [("Wst", ty [C,IC,1,1]), ("bst", ty [C]), ("gst", "tensor<f32>"), ("btst", "tensor<f32>")]
    ++ blkParams 1 ++ blkParams 2
    ++ [("ghd", "tensor<f32>"), ("bthd", "tensor<f32>"), ("Wd", ty [C,10]), ("bd", ty [10])]
  let upd := String.join (allParams.map (fun (nm, t) => sgd s!"%{nm}" s!"%d{nm}" t))
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [BS, IC*H*H]) :: allParams.map (fun (nm, t) => s!"%{nm}: {t}") ++ ["%onehot: " ++ ty [BS,10]])
  let retTyL := String.intercalate ", " (allParams.map (fun (_, t) => t))
  let retVals := String.intercalate ", " (allParams.map (fun (nm, _) => s!"%{nm}n"))
  return "module @m {\n" ++ s!"  func.func @convnext_rep_train_step({argSig}) -> ({retTyL}) " ++ "{\n" ++
    "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    s!"    %bsc = stablehlo.constant dense<{BS}.0> : {ty [BS,10]}\n" ++
    body ++ upd ++
    s!"    return {retVals} : {retTyL}\n" ++ "  }\n}\n"

def main : IO Unit := do
  let mlir := trainStep
  IO.println s!"rendered structured ConvNeXt (representative) train step: {mlir.length} chars"
  IO.FS.createDirAll "/tmp/cnxpc"
  IO.FS.writeFile "/tmp/cnxpc/train_step.mlir" mlir
  let cargs ← ireeCompileArgs "/tmp/cnxpc/train_step.mlir" "/tmp/cnxpc/train_step.vmfb"
  let r ← IO.Process.output { cmd := "iree-compile", args := cargs }
  if r.exitCode != 0 then
    IO.eprintln s!"iree-compile FAILED:\n{r.stderr.take 5000}"
  else
    IO.println "structured ConvNeXt representative train step iree-compile OK → /tmp/cnxpc/train_step.mlir"

#eval main
