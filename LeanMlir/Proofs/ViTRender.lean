import LeanMlir.Proofs.ViTMultiHead

/-! # ViT-Tiny train step rendered from the verified AST (the §1 render) — FORWARD portion

The ViT peer of `MobileNetV2Render`/`ConvNeXtRender`: the full depth-12 ViT-Tiny forward rendered as
`pretty` of the verified multi-head vector-LN graph (`vitBlockGraphMHV` × 12 + patch embed + final
vector-LN + CLS-slice dense head). The committed `LeanMlir/ViTRender.lean` is a hand-written String
emitter (faithful per-op, NOT `pretty(provenGraph)`); this renders the SAME forward as `pretty` of the
proven `SHlo` graph, so `den(graph) = vitForward` (via `vitFwdGraphMHV_faithful`, here at depth-12).

Render is value-independent (`skel` erases the `ℝ`/`Mat`/`Vec` fields), so placeholders (`0`, zero
mats/vecs) are passed; the emitted `epsStr`/`sStr` literals carry the real ε / SDPA-scale. This file
is the FORWARD half of the §1 train-step render; the backward-cotangent chain (via the `*Back` ops)
+ the param-SGD tail (`veclnGammaSgd`/`patchEmbedWeightSgd`/`denseWeightSgdB`/`denseBiasSgdB`) follow.

ViT-Tiny: ic=3, 224², patch 16×16/s16 (N=196 patches, 197 tokens), D=192 = 3 heads × 64, MLP 768,
12 blocks, 10 classes, BS=32, ε=1e-5, SDPA scale = 1/√64 = 0.125. -/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

private def vBS : Nat := 32
private def vEPS : String := "1.0e-5"
private def vSCALE : String := "0.125"
private def vLR : String := "0.1"
private def vDEPTH : Nat := 12

private def zVv {n : Nat} : Vec n := fun _ => 0
private def zMm {a b : Nat} : Mat a b := fun _ _ => 0
private def zKk {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0

-- ── node-by-node renderers (the computable ConvNeXt pattern: each `pretty` emits ONE op with
--    `.operand <prevSSA> <zero-placeholder>`, threading SSA name strings; `vitBlockGraphMHV` is the
--    composed-graph reference) ──

/-- One **vector-LN** site (`lnRow(1,0) → rowScale γ → rowBias β`) on the `[197,192]` token matrix,
    with explicit γ/β param names. Returns the LN-output SSA. -/
private def vlnFwd (gName btName xin : String) : StateM Nat (String × String) := do
  let (c1, a) ← pretty vBS (.lnRowF "%one" "%zero" vEPS 0 1 0 (.operand xin (zVv : Vec (197*192))))
  let (c2, b) ← pretty vBS (.rowScaleF gName (zVv : Vec 192) (.operand a (zVv : Vec (197*192))))
  let (c3, o) ← pretty vBS (.rowBiasF btName (zVv : Vec 192) (.operand b (zVv : Vec (197*192))))
  pure (c1 ++ c2 ++ c3, o)

/-- The forward SSA names a block's backward + param-SGD reference (the ConvNeXt-`FNames` analogue).
    Per-head arrays hold the 3 heads' slices + pre-softmax + softmax-output. -/
private structure BSaves where
  xin : String       -- block input (LN1 input, residual1)
  ln1 : String       -- LN1 output (Q/K/V dense input)
  q : String         -- Q output
  k : String         -- K output
  v : String         -- V output
  qss : Array String -- per-head Q slices
  kss : Array String -- per-head K slices
  vss : Array String -- per-head V slices
  scs : Array String -- per-head pre-softmax (softmaxRowBack)
  sms : Array String -- per-head softmax output
  att : String       -- attn output (Wo dense input)
  hres : String      -- residual1 (LN2 input, residual2)
  ln2 : String       -- LN2 output (fc1 dense input)
  f1 : String        -- pre-gelu (geluBack)
  g : String         -- gelu output (fc2 dense input)
  bout : String      -- block output
  deriving Inhabited

/-- One **transformer block** forward (pre-norm, multi-head vector-LN), prefix `pfx`. Mirrors
    `vitBlockGraphMHV` node-by-node: LN1 → Q/K/V dense → per-head SDPA (slice→QKᵀ→scale→softmax→·V→pad,
    summed) → out dense → +res → LN2 → fc1 → GELU → fc2 → +res. Returns (code, the saved SSA names). -/
private def vBlockFwd (pfx xin : String) : StateM Nat (String × BSaves) := do
  let (c1, ln1) ← vlnFwd s!"%{pfx}g1" s!"%{pfx}bt1" xin
  let (cq, q) ← pretty vBS (.denseRowF s!"%{pfx}Wq" s!"%{pfx}bq" (zMm : Mat 192 192) zVv (.operand ln1 (zVv : Vec (197*192))))
  let (ck, k) ← pretty vBS (.denseRowF s!"%{pfx}Wk" s!"%{pfx}bk" (zMm : Mat 192 192) zVv (.operand ln1 (zVv : Vec (197*192))))
  let (cv, v) ← pretty vBS (.denseRowF s!"%{pfx}Wv" s!"%{pfx}bv" (zMm : Mat 192 192) zVv (.operand ln1 (zVv : Vec (197*192))))
  -- per-head SDPA, accumulate the padded heads with addV
  let mut code := c1 ++ cq ++ ck ++ cv
  let mut acc : String := ""
  let mut qss : Array String := #[]; let mut kss : Array String := #[]; let mut vss : Array String := #[]
  let mut scs : Array String := #[]; let mut sms : Array String := #[]
  for hh in [0:3] do
    let h : Fin 3 := ⟨hh % 3, by omega⟩
    let (cqs, qs) ← pretty vBS (.headSliceF (N := 197) (heads := 3) (d := 64) h (.operand q (zVv : Vec (197*(3*64)))))
    let (cks, ks) ← pretty vBS (.headSliceF (N := 197) (heads := 3) (d := 64) h (.operand k (zVv : Vec (197*(3*64)))))
    let (cvs, vs) ← pretty vBS (.headSliceF (N := 197) (heads := 3) (d := 64) h (.operand v (zVv : Vec (197*(3*64)))))
    let (ckt, kt) ← pretty vBS (.transposeF (m := 197) (n := 64) (.operand ks (zVv : Vec (197*64))))
    let (cmm, qk) ← pretty vBS (.matmulF (m := 197) (k := 64) (n := 197) (.operand qs (zVv : Vec (197*64))) (.operand kt (zVv : Vec (64*197))))
    let (csc, sc) ← pretty vBS (.scaleF vSCALE 0 (.operand qk (zVv : Vec (197*197))))
    let (csm, sm) ← pretty vBS (.softmaxRowF (m := 197) (n := 197) (.operand sc (zVv : Vec (197*197))))
    let (cpv, pv) ← pretty vBS (.matmulF (m := 197) (k := 197) (n := 64) (.operand sm (zVv : Vec (197*197))) (.operand vs (zVv : Vec (197*64))))
    let (cpd, pd) ← pretty vBS (.headPadF (N := 197) (heads := 3) (d := 64) h (.operand pv (zVv : Vec (197*64))))
    code := code ++ cqs ++ cks ++ cvs ++ ckt ++ cmm ++ csc ++ csm ++ cpv ++ cpd
    qss := qss.push qs; kss := kss.push ks; vss := vss.push vs; scs := scs.push sc; sms := sms.push sm
    if hh == 0 then
      acc := pd
    else
      let (cad, s) ← pretty vBS (.addV (.operand acc (zVv : Vec (197*192))) (.operand pd (zVv : Vec (197*192))))
      code := code ++ cad; acc := s
  let (co, o) ← pretty vBS (.denseRowF s!"%{pfx}Wo" s!"%{pfx}bo" (zMm : Mat 192 192) zVv (.operand acc (zVv : Vec (197*192))))
  let (ch, hres) ← pretty vBS (.addV (.operand xin (zVv : Vec (197*192))) (.operand o (zVv : Vec (197*192))))
  let (c2, ln2) ← vlnFwd s!"%{pfx}g2" s!"%{pfx}bt2" hres
  let (cf1, f1) ← pretty vBS (.denseRowF s!"%{pfx}Wfc1" s!"%{pfx}bfc1" (zMm : Mat 192 768) zVv (.operand ln2 (zVv : Vec (197*192))))
  let (cg, g) ← pretty vBS (.geluF (.operand f1 (zVv : Vec (197*768))))
  let (cf2, f2) ← pretty vBS (.denseRowF s!"%{pfx}Wfc2" s!"%{pfx}bfc2" (zMm : Mat 768 192) zVv (.operand g (zVv : Vec (197*768))))
  let (cr, bout) ← pretty vBS (.addV (.operand hres (zVv : Vec (197*192))) (.operand f2 (zVv : Vec (197*192))))
  pure (code ++ co ++ ch ++ c2 ++ cf1 ++ cg ++ cf2 ++ cr,
    { xin, ln1, q, k, v, qss, kss, vss, scs, sms, att := acc, hres, ln2, f1, g, bout })

/-- Forward saves the whole-net backward references: the patch embed SSA, the per-block saves, the
    final-LN input (last block output) + output, and the logits SSA. -/
private structure FwdSaves where
  embed : String
  blocks : Array BSaves
  flnIn : String        -- final-LN input (= last block output)
  fln : String          -- final-LN output (= CLS-slice input)
  clsTok : String       -- CLS-slice output (= head-dense input)
  logits : String
  deriving Inhabited

/-- The depth-12 ViT-Tiny **forward**, node-by-node. Returns (body, saves). -/
private def vitFwd12 : StateM Nat (String × FwdSaves) := do
  let (ce, embed) ← pretty vBS (.patchEmbedF "%wConv" "%bConv" "%cls" "%pos"
    (zKk : Kernel4 192 3 16 16) zVv zVv (zMm : Mat 197 192) (.operand "%x" (zVv : Vec (3*224*224))))
  let mut code := ce
  let mut cur := embed
  let mut blocks : Array BSaves := #[]
  for i in [0:vDEPTH] do
    let (cb, sv) ← vBlockFwd s!"b{i}_" cur
    code := code ++ cb; cur := sv.bout; blocks := blocks.push sv
  let (cf, fl) ← vlnFwd "%gF" "%btF" cur
  let (cs, sl) ← pretty vBS (.clsSliceF (N := 196) (D := 192) (.operand fl (zVv : Vec (197*192))))
  let (cl, logits) ← pretty vBS (denseF "%Wc" "%bc" (zMm : Mat 192 10) zVv (.operand sl (zVv : Vec 192)))
  pure (code ++ cf ++ cs ++ cl, { embed, blocks, flnIn := cur, fln := fl, clsTok := sl, logits })

/-- Per-block func-arg signature (committed forward order). -/
private def blkArgSig (i : Nat) : String :=
  String.intercalate ", "
    [s!"%b{i}_g1: {ty [192]}", s!"%b{i}_bt1: {ty [192]}",
     s!"%b{i}_Wq: {ty [192,192]}", s!"%b{i}_bq: {ty [192]}",
     s!"%b{i}_Wk: {ty [192,192]}", s!"%b{i}_bk: {ty [192]}",
     s!"%b{i}_Wv: {ty [192,192]}", s!"%b{i}_bv: {ty [192]}",
     s!"%b{i}_Wo: {ty [192,192]}", s!"%b{i}_bo: {ty [192]}",
     s!"%b{i}_g2: {ty [192]}", s!"%b{i}_bt2: {ty [192]}",
     s!"%b{i}_Wfc1: {ty [192,768]}", s!"%b{i}_bfc1: {ty [768]}",
     s!"%b{i}_Wfc2: {ty [768,192]}", s!"%b{i}_bfc2: {ty [192]}"]

/-- **ViT-Tiny depth-12 forward rendered ENTIRELY from the verified AST.** Every line is `pretty` of a
    verified `SHlo` node; `den(graph) = vitForward` by `vitFwdGraphMHV_faithful` (at depth-12). The
    output is the `[BS,10]` logits. (FORWARD half of the §1 train-step render.) -/
def vitFwdRenderV (funcName : String := "vit_fwd") : String :=
  let (body, sv) := vitFwd12.run' 0
  let res := sv.logits
  let blkSigs := String.intercalate ", " ((List.range vDEPTH).map blkArgSig)
  let argSig := s!"%x: {ty [vBS, 3*224*224]}, %wConv: {ty [192,3,16,16]}, %bConv: {ty [192]}, " ++
    s!"%cls: {ty [192]}, %pos: {ty [197,192]}, " ++ blkSigs ++
    s!", %gF: {ty [192]}, %btF: {ty [192]}, %Wc: {ty [192,10]}, %bc: {ty [10]}"
  "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> {ty [vBS, 10]} " ++ "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  body ++ s!"    return {res} : {ty [vBS, 10]}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════════════════════════════
-- § BACKWARD render — node-by-node reverse of `vBlockFwd`/`vitFwd12`, then the 200-param SGD tail
-- ════════════════════════════════════════════════════════════════════════════════════════

/-- One **vector-LN backward** site (the reverse of `vlnFwd`): given the LN-output cotangent `dyOut`
    on `[197,192]` and the saved LN INPUT `xin`, emit (a) the β-SGD (`rowDenseBiasSgd`, dβ=Σ dy),
    (b) the γ-SGD (`veclnGammaSgd`, dγ=Σ dy⊙x̂ recomputed from `xin`), and (c) the input cotangent
    `dxin = lnRowBack(γ=1)(rowScale γ (dy))` (back through normalize after the γ-scale).
    Returns (code, dxin, ngamma, nbeta). `lr` is the mean-loss-equiv literal. -/
private def vlnBack (gName btName xin dyOut lrStr : String) : StateM Nat (String × String × String × String) := do
  let (cb, nb) ← pretty vBS (.rowDenseBiasSgd (N := 197) (c := 192) btName lrStr (zVv : Vec 192) 0
                              (.operand dyOut (zVv : Vec (197*192))))
  let (cg, ng) ← pretty vBS (.veclnGammaSgd (N := 197) (D := 192) gName xin vEPS lrStr 0
                              (zVv : Vec (197*192)) (zVv : Vec 192) 0 (.operand dyOut (zVv : Vec (197*192))))
  let (cs, da) ← pretty vBS (.rowScaleF (m := 197) (n := 192) gName (zVv : Vec 192)
                              (.operand dyOut (zVv : Vec (197*192))))
  let (cn, dx) ← pretty vBS (.lnRowBack (m := 197) (n := 192) "%one" xin vEPS 0 1 (zVv : Vec (197*192))
                              (.operand da (zVv : Vec (197*192))))
  pure (cb ++ cg ++ cs ++ cn, dx, ng, nb)

/-- One **transformer block backward** (the reverse of `vBlockFwd`), prefix `pfx`, with the forward
    saves `sv` and the block-output cotangent `dyOut`. Threads the cotangent in reverse:
    +res₂ fan-out → fc2-back → GELU-back → fc1-back → LN2-back → +res₁ fan-in (dhres) → out-dense-back
    → per-head SDPA backward (slice-back / matmul-backs / softmax-back / scale / transpose-backs) →
    Q/K/V-dense-back (summed) → LN1-back → +res₁ fan-in (dxin). Returns (code, dxin, the 16 param
    SGD-update SSAs in `BlockParams` order: g1,bt1, Wq,bq,Wk,bk,Wv,bv,Wo,bo, g2,bt2, Wfc1,bfc1,Wfc2,bfc2). -/
private def vBlockBack (pfx : String) (sv : BSaves) (dyOut lrStr : String) :
    StateM Nat (String × String × List String) := do
  let p := pfx
  -- ─ MLP sublayer back: bout = addV(hres, f2); df2 = dyOut, dhres ⊇ dyOut ─
  -- fc2: f2 = denseRow(Wfc2,bfc2)(g)  [g:197*768 → f2:197*192]
  let (c1, dg) ← pretty vBS (.denseRowBack (N := 197) (a := 768) (c := 192) s!"%{p}Wfc2" (zMm : Mat 768 192)
                              (.operand dyOut (zVv : Vec (197*192))))
  let (c2, nWfc2) ← pretty vBS (.rowDenseWeightSgd (N := 197) (a := 768) (c := 192) sv.g s!"%{p}Wfc2" lrStr
                              (zVv : Vec (197*768)) (zMm : Mat 768 192) 0 (.operand dyOut (zVv : Vec (197*192))))
  let (c3, nbfc2) ← pretty vBS (.rowDenseBiasSgd (N := 197) (c := 192) s!"%{p}bfc2" lrStr (zVv : Vec 192) 0
                              (.operand dyOut (zVv : Vec (197*192))))
  -- gelu: g = gelu(f1)  [197*768]
  let (c4, df1) ← pretty vBS (.geluBack (n := 197*768) sv.f1 (zVv : Vec (197*768))
                              (.operand dg (zVv : Vec (197*768))))
  -- fc1: f1 = denseRow(Wfc1,bfc1)(ln2)  [ln2:197*192 → f1:197*768]
  let (c5, dln2) ← pretty vBS (.denseRowBack (N := 197) (a := 192) (c := 768) s!"%{p}Wfc1" (zMm : Mat 192 768)
                              (.operand df1 (zVv : Vec (197*768))))
  let (c6, nWfc1) ← pretty vBS (.rowDenseWeightSgd (N := 197) (a := 192) (c := 768) sv.ln2 s!"%{p}Wfc1" lrStr
                              (zVv : Vec (197*192)) (zMm : Mat 192 768) 0 (.operand df1 (zVv : Vec (197*768))))
  let (c7, nbfc1) ← pretty vBS (.rowDenseBiasSgd (N := 197) (c := 768) s!"%{p}bfc1" lrStr (zVv : Vec 768) 0
                              (.operand df1 (zVv : Vec (197*768))))
  -- LN2 back (input = hres)
  let (c8, dhresLn2, ng2, nbt2) ← vlnBack s!"%{p}g2" s!"%{p}bt2" sv.hres dln2 lrStr
  -- dhres = dyOut (res₂ skip) + dhresLn2 (LN2 path)
  let (c9, dhres) ← pretty vBS (.addV (.operand dyOut (zVv : Vec (197*192))) (.operand dhresLn2 (zVv : Vec (197*192))))
  -- ─ Attention sublayer back: hres = addV(xin, o); do = dhres, dxin ⊇ dhres ─
  -- out-dense: o = denseRow(Wo,bo)(acc)  [acc=att:197*192 → o:197*192]
  let (c10, dacc) ← pretty vBS (.denseRowBack (N := 197) (a := 192) (c := 192) s!"%{p}Wo" (zMm : Mat 192 192)
                              (.operand dhres (zVv : Vec (197*192))))
  let (c11, nWo) ← pretty vBS (.rowDenseWeightSgd (N := 197) (a := 192) (c := 192) sv.att s!"%{p}Wo" lrStr
                              (zVv : Vec (197*192)) (zMm : Mat 192 192) 0 (.operand dhres (zVv : Vec (197*192))))
  let (c12, nbo) ← pretty vBS (.rowDenseBiasSgd (N := 197) (c := 192) s!"%{p}bo" lrStr (zVv : Vec 192) 0
                              (.operand dhres (zVv : Vec (197*192))))
  -- per-head SDPA backward; accumulate dq/dk/dv over the 3 heads
  let mut code := c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++ c8 ++ c9 ++ c10 ++ c11 ++ c12
  let mut dqAcc : String := ""; let mut dkAcc : String := ""; let mut dvAcc : String := ""
  for hh in [0:3] do
    let h : Fin 3 := ⟨hh % 3, by omega⟩
    -- pd[h] = headPad(h)(pv[h]); dpv = headSlice(h)(dacc)
    let (ca, dpv) ← pretty vBS (.headSliceF (N := 197) (heads := 3) (d := 64) h (.operand dacc (zVv : Vec (197*(3*64)))))
    -- pv[h] = matmul(sm[h][197,197], vs[h][197,64]) → [197,64]
    let (cb, vsT) ← pretty vBS (.transposeF (m := 197) (n := 64) (.operand (sv.vss[hh]!) (zVv : Vec (197*64))))
    let (cc, dsm) ← pretty vBS (.matmulF (m := 197) (k := 64) (n := 197) (.operand dpv (zVv : Vec (197*64))) (.operand vsT (zVv : Vec (64*197))))
    let (cd, smT) ← pretty vBS (.transposeF (m := 197) (n := 197) (.operand (sv.sms[hh]!) (zVv : Vec (197*197))))
    let (ce, dvs) ← pretty vBS (.matmulF (m := 197) (k := 197) (n := 64) (.operand smT (zVv : Vec (197*197))) (.operand dpv (zVv : Vec (197*64))))
    -- sm[h] = softmaxRow(sc[h]); dsc = softmaxRowBack(sc[h])(dsm)
    let (cf, dsc) ← pretty vBS (.softmaxRowBack (m := 197) (n := 197) (sv.scs[hh]!) (zVv : Vec (197*197)) (.operand dsm (zVv : Vec (197*197))))
    -- sc[h] = scale(qk[h]); dqk = scale(dsc)
    let (cg2, dqk) ← pretty vBS (.scaleF vSCALE 0 (.operand dsc (zVv : Vec (197*197))))
    -- qk[h] = matmul(qs[h][197,64], kt[h][64,197]); dqs = dqk·ktᵀ = matmul(dqk[197,197], ks[h][197,64])
    let (ch, dqs) ← pretty vBS (.matmulF (m := 197) (k := 197) (n := 64) (.operand dqk (zVv : Vec (197*197))) (.operand (sv.kss[hh]!) (zVv : Vec (197*64))))
    -- dkt = qsᵀ·dqk = matmul(qsᵀ[64,197], dqk[197,197]) → [64,197]; dks = transpose(dkt) → [197,64]
    let (ci, qsT) ← pretty vBS (.transposeF (m := 197) (n := 64) (.operand (sv.qss[hh]!) (zVv : Vec (197*64))))
    let (cj, dkt) ← pretty vBS (.matmulF (m := 64) (k := 197) (n := 197) (.operand qsT (zVv : Vec (64*197))) (.operand dqk (zVv : Vec (197*197))))
    let (ck, dks) ← pretty vBS (.transposeF (m := 64) (n := 197) (.operand dkt (zVv : Vec (64*197))))
    -- scatter each head's grad back into the [197,192] feature block
    let (cl, dqH) ← pretty vBS (.headPadF (N := 197) (heads := 3) (d := 64) h (.operand dqs (zVv : Vec (197*64))))
    let (cm, dkH) ← pretty vBS (.headPadF (N := 197) (heads := 3) (d := 64) h (.operand dks (zVv : Vec (197*64))))
    let (cn, dvH) ← pretty vBS (.headPadF (N := 197) (heads := 3) (d := 64) h (.operand dvs (zVv : Vec (197*64))))
    code := code ++ ca ++ cb ++ cc ++ cd ++ ce ++ cf ++ cg2 ++ ch ++ ci ++ cj ++ ck ++ cl ++ cm ++ cn
    if hh == 0 then
      dqAcc := dqH; dkAcc := dkH; dvAcc := dvH
    else
      let (cq, dqs2) ← pretty vBS (.addV (.operand dqAcc (zVv : Vec (197*192))) (.operand dqH (zVv : Vec (197*192))))
      let (cr, dks2) ← pretty vBS (.addV (.operand dkAcc (zVv : Vec (197*192))) (.operand dkH (zVv : Vec (197*192))))
      let (cs, dvs2) ← pretty vBS (.addV (.operand dvAcc (zVv : Vec (197*192))) (.operand dvH (zVv : Vec (197*192))))
      code := code ++ cq ++ cr ++ cs; dqAcc := dqs2; dkAcc := dks2; dvAcc := dvs2
  -- Q/K/V dense backward: q/k/v = denseRow(W*,b*)(ln1)  [ln1:197*192 → 197*192]
  let (cq1, dln1q) ← pretty vBS (.denseRowBack (N := 197) (a := 192) (c := 192) s!"%{p}Wq" (zMm : Mat 192 192) (.operand dqAcc (zVv : Vec (197*192))))
  let (cq2, nWq) ← pretty vBS (.rowDenseWeightSgd (N := 197) (a := 192) (c := 192) sv.ln1 s!"%{p}Wq" lrStr (zVv : Vec (197*192)) (zMm : Mat 192 192) 0 (.operand dqAcc (zVv : Vec (197*192))))
  let (cq3, nbq) ← pretty vBS (.rowDenseBiasSgd (N := 197) (c := 192) s!"%{p}bq" lrStr (zVv : Vec 192) 0 (.operand dqAcc (zVv : Vec (197*192))))
  let (ck1, dln1k) ← pretty vBS (.denseRowBack (N := 197) (a := 192) (c := 192) s!"%{p}Wk" (zMm : Mat 192 192) (.operand dkAcc (zVv : Vec (197*192))))
  let (ck2, nWk) ← pretty vBS (.rowDenseWeightSgd (N := 197) (a := 192) (c := 192) sv.ln1 s!"%{p}Wk" lrStr (zVv : Vec (197*192)) (zMm : Mat 192 192) 0 (.operand dkAcc (zVv : Vec (197*192))))
  let (ck3, nbk) ← pretty vBS (.rowDenseBiasSgd (N := 197) (c := 192) s!"%{p}bk" lrStr (zVv : Vec 192) 0 (.operand dkAcc (zVv : Vec (197*192))))
  let (cv1, dln1v) ← pretty vBS (.denseRowBack (N := 197) (a := 192) (c := 192) s!"%{p}Wv" (zMm : Mat 192 192) (.operand dvAcc (zVv : Vec (197*192))))
  let (cv2, nWv) ← pretty vBS (.rowDenseWeightSgd (N := 197) (a := 192) (c := 192) sv.ln1 s!"%{p}Wv" lrStr (zVv : Vec (197*192)) (zMm : Mat 192 192) 0 (.operand dvAcc (zVv : Vec (197*192))))
  let (cv3, nbv) ← pretty vBS (.rowDenseBiasSgd (N := 197) (c := 192) s!"%{p}bv" lrStr (zVv : Vec 192) 0 (.operand dvAcc (zVv : Vec (197*192))))
  -- dln1 = dln1q + dln1k + dln1v
  let (cs1, dln1a) ← pretty vBS (.addV (.operand dln1q (zVv : Vec (197*192))) (.operand dln1k (zVv : Vec (197*192))))
  let (cs2, dln1) ← pretty vBS (.addV (.operand dln1a (zVv : Vec (197*192))) (.operand dln1v (zVv : Vec (197*192))))
  -- LN1 back (input = xin)
  let (cl1, dxinLn1, ng1, nbt1) ← vlnBack s!"%{p}g1" s!"%{p}bt1" sv.xin dln1 lrStr
  -- dxin = dhres (res₁ skip) + dxinLn1 (LN1 path)
  let (cx, dxin) ← pretty vBS (.addV (.operand dhres (zVv : Vec (197*192))) (.operand dxinLn1 (zVv : Vec (197*192))))
  let names := [ng1, nbt1, nWq, nbq, nWk, nbk, nWv, nbv, nWo, nbo, ng2, nbt2, nWfc1, nbfc1, nWfc2, nbfc2]
  pure (code ++ cq1 ++ cq2 ++ cq3 ++ ck1 ++ ck2 ++ ck3 ++ cv1 ++ cv2 ++ cv3 ++ cs1 ++ cs2 ++ cl1 ++ cx, dxin, names)

/-- The 16 per-block return TYPE strings (BlockParams order), matching `blkArgSig`. -/
private def blkRetTys : List String :=
  [ty [192], ty [192], ty [192,192], ty [192], ty [192,192], ty [192], ty [192,192], ty [192],
   ty [192,192], ty [192], ty [192], ty [192], ty [192,768], ty [768], ty [768,192], ty [192]]

/-- **ViT-Tiny depth-12 train step rendered ENTIRELY from the verified AST** — the §1 backward render.
    Forward (`vitFwd12`) → softmax-CE cotangent (`softmax(logits) − onehot`, the `lossCotGraph` form) →
    head-dense back (`dotOut` + `weightSgd`/`biasSgd`) → `clsPadF` → final-LN back (`vlnBack`) → 12×
    `vBlockBack` (reversed, cotangent threaded) → patch-embed back (`patchEmbedWeightSgd`/`patchEmbedBiasSgd`
    + `clsSliceF`→`denseBiasSgdB` for cls + `posEmbedSgd` for pos). Returns the 200 SGD-updated params in
    func-arg order. `lrStr` is the mean-loss-equiv literal (base/BS); cotangent has NO /B (folded into lr). -/
def vitTrainStepRenderV (funcName : String := "vit_train_step") (lrStr : String := "0.003125") : String :=
  let go : StateM Nat String := do
    let (fwd, sv) ← vitFwd12
    -- loss cotangent dy = softmax(logits) − onehot  (lossCotGraph_isCEgrad; mean folded into lr)
    let (cDy, nDy) ← pretty vBS (.sub (.softmaxDiv (.expe (.operand sv.logits (zVv : Vec 10)))) (.operand "%onehot" (zVv : Vec 10)))
    -- head: logits = denseF(Wc,bc)(clsTok)  [clsTok:192 → logits:10]
    let (cDc, dcls) ← pretty vBS (.dotOut (m := 192) (n := 10) "%Wc" (zMm : Mat 192 10) (.operand nDy (zVv : Vec 10)))
    let (cWc, nWc) ← pretty vBS (.weightSgd sv.clsTok "%Wc" lrStr (zVv : Vec 192) (zMm : Mat 192 10) 0 (.operand nDy (zVv : Vec 10)))
    let (cbc, nbc) ← pretty vBS (.biasSgd "%bc" lrStr (zVv : Vec 10) 0 (.operand nDy (zVv : Vec 10)))
    -- scatter the CLS-token cotangent back into the final-LN output (row 0), zero elsewhere
    let (cPad, dfln) ← pretty vBS (.clsPadF (N := 196) (D := 192) (.operand dcls (zVv : Vec 192)))
    -- final LN back (input = flnIn = last block output)
    let (cFln, dflnIn, ngF, nbtF) ← vlnBack "%gF" "%btF" sv.flnIn dfln lrStr
    -- 12 blocks reversed; thread the cotangent from the final-LN input down to the embed
    let mut code := fwd ++ cDy ++ cDc ++ cWc ++ cbc ++ cPad ++ cFln
    let mut dcur := dflnIn
    let mut blkNames : Array (List String) := #[]   -- per-block update names, fwd-index order
    for j in [0:vDEPTH] do
      let i := vDEPTH - 1 - j
      let (cb, dx, names) ← vBlockBack s!"b{i}_" (sv.blocks[i]!) dcur lrStr
      code := code ++ cb; dcur := dx; blkNames := blkNames.push names
    -- `dcur` is now the patch-embed output cotangent (dembed)
    -- patch-embed param grads: wConv (patchEmbedWeightSgd), bConv (patchEmbedBiasSgd),
    -- cls (clsSlice→denseBiasSgdB), pos (posEmbedSgd)
    let (cwC, nwConv) ← pretty vBS (.patchEmbedWeightSgd (ic := 3) (H := 224) (W := 224) (P := 16) (N := 196) (D := 192)
                          "%wConv" "%ximg" lrStr (zVv : Vec (3*224*224)) (zKk : Kernel4 192 3 16 16) 0
                          (.operand dcur (zVv : Vec (197*192))))
    let (cbC, nbConv) ← pretty vBS (.patchEmbedBiasSgd (N := 196) (c := 192) "%bConv" lrStr (zVv : Vec 192) 0
                          (.operand dcur (zVv : Vec (197*192))))
    let (cClSl, dclsRow) ← pretty vBS (.clsSliceF (N := 196) (D := 192) (.operand dcur (zVv : Vec (197*192))))
    let (cCl, ncls) ← pretty vBS (.denseBiasSgdB (N := 1) (c := 192) "%cls" lrStr (zVv : Vec 192) 0
                          (.operand dclsRow (zVv : Vec 192)))
    let (cPo, npos) ← pretty vBS (.posEmbedSgd (N := 196) (D := 192) "%pos" lrStr (zMm : Mat 197 192) 0
                          (.operand dcur (zVv : Vec (197*192))))
    -- return 200 updated params in func-arg order. `blkNames` was pushed in reverse build order
    -- (j=0 → block 11, …, j=11 → block 0), so `blkNames[vDEPTH-1-i]` = block i's 16 update SSAs.
    let blkOutOrdered := (List.range vDEPTH).flatMap (fun i => blkNames[vDEPTH - 1 - i]!)
    let retNames := [nwConv, nbConv, ncls, npos] ++ blkOutOrdered ++ [ngF, nbtF, nWc, nbc]
    let retTys := [ty [192,3,16,16], ty [192], ty [192], ty [197,192]] ++
      ((List.range vDEPTH).flatMap (fun _ => blkRetTys)) ++ [ty [192], ty [192], ty [192,10], ty [10]]
    pure <|
      "    // ── ViT-Tiny depth-12 train step: every line is pretty(verified AST node) ──\n" ++
      code ++ cwC ++ cbC ++ cClSl ++ cCl ++ cPo ++
      s!"    return {String.intercalate ", " retNames} : {String.intercalate ", " retTys}\n"
  let body : String := go.run' 0
  let blkSigs := String.intercalate ", " ((List.range vDEPTH).map blkArgSig)
  let argSig := s!"%x: {ty [vBS, 3*224*224]}, %wConv: {ty [192,3,16,16]}, %bConv: {ty [192]}, " ++
    s!"%cls: {ty [192]}, %pos: {ty [197,192]}, " ++ blkSigs ++
    s!", %gF: {ty [192]}, %btF: {ty [192]}, %Wc: {ty [192,10]}, %bc: {ty [10]}, %onehot: {ty [vBS, 10]}"
  let retTys := [ty [192,3,16,16], ty [192], ty [192], ty [197,192]] ++
    ((List.range vDEPTH).flatMap (fun _ => blkRetTys)) ++ [ty [192], ty [192], ty [192,10], ty [10]]
  "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> ({String.intercalate ", " retTys}) " ++ "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %ximg = stablehlo.reshape %x : ({ty [vBS, 3*224*224]}) -> {ty [vBS, 3, 224, 224]}\n" ++
  body ++ "  }\n}\n"

end Proofs.StableHLO

-- Regenerate the committed verified_mlir/vit_{fwd,train_step}.mlir from the certified renderer
-- (pure Lean, no iree) — the drift-guard source: `lake env lean LeanMlir/Proofs/ViTRender.lean`
-- rewrites both, and proofs.yml git-diffs them. The bytes `MainViTVerified` trains on ARE these.
-- (tests/TestViT{Train,Fwd}.lean write the SAME render + additionally iree-compile on the rocm box.)
#eval IO.FS.writeFile "verified_mlir/vit_fwd.mlir" (Proofs.StableHLO.vitFwdRenderV "vit_fwd")
#eval IO.FS.writeFile "verified_mlir/vit_train_step.mlir"
  (Proofs.StableHLO.vitTrainStepRenderV "vit_train_step" "0.003125")
