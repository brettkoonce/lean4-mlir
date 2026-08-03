import LeanMlir.Proofs.Architectures.ViTMultiHead
import LeanMlir.ViTRender

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
12 blocks, `nClasses` classes (10 as committed), BS=32, ε=1e-5, SDPA scale = 1/√64 = 0.125. -/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

-- ⚠ NOT `private`: `ViTRenderB` shares these rather than restating them. A second copy of an
-- epsilon or a depth is the double-writer disease with a silently-wrong hyperparameter as its
-- failure mode (§2a-quater), and the SDPA scale in particular is a number no type would catch.
def vEPS : String := "1.0e-5"
def vSCALE : String := "0.125"
private def vLR : String := "0.1"
def vDEPTH : Nat := 12

private def zVv {n : Nat} : Vec n := fun _ => 0
private def zMm {a b : Nat} : Mat a b := fun _ _ => 0
private def zKk {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0

-- ── node-by-node renderers (the computable ConvNeXt pattern: each `pretty` emits ONE op with
--    `.operand <prevSSA> <zero-placeholder>`, threading SSA name strings; `vitBlockGraphMHV` is the
--    composed-graph reference) ──

/-- One **vector-LN** site (`lnRow(1,0) → rowScale γ → rowBias β`) on the `[197,192]` token matrix,
    with explicit γ/β param names. Returns the LN-output SSA. -/
private def vlnFwd (bs : Nat) (gName btName xin : String) : StateM Nat (String × String) := do
  let (c1, a) ← pretty bs (.lnRowF "%one" "%zero" vEPS 0 1 0 (.operand xin (zVv : Vec (197*192))))
  let (c2, b) ← pretty bs (.rowScaleF gName (zVv : Vec 192) (.operand a (zVv : Vec (197*192))))
  let (c3, o) ← pretty bs (.rowBiasF btName (zVv : Vec 192) (.operand b (zVv : Vec (197*192))))
  pure (c1 ++ c2 ++ c3, o)

/-- The forward SSA names a block's backward + param-SGD reference (the ConvNeXt-`FNames` analogue).
    Per-head arrays hold the 3 heads' slices + pre-softmax + softmax-output. -/
structure BSaves where
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
private def vBlockFwd (bs : Nat) (pfx xin : String) : StateM Nat (String × BSaves) := do
  let (c1, ln1) ← vlnFwd bs s!"%{pfx}g1" s!"%{pfx}bt1" xin
  let (cq, q) ← pretty bs (.denseRowF s!"%{pfx}Wq" s!"%{pfx}bq" (zMm : Mat 192 192) zVv (.operand ln1 (zVv : Vec (197*192))))
  let (ck, k) ← pretty bs (.denseRowF s!"%{pfx}Wk" s!"%{pfx}bk" (zMm : Mat 192 192) zVv (.operand ln1 (zVv : Vec (197*192))))
  let (cv, v) ← pretty bs (.denseRowF s!"%{pfx}Wv" s!"%{pfx}bv" (zMm : Mat 192 192) zVv (.operand ln1 (zVv : Vec (197*192))))
  -- per-head SDPA, accumulate the padded heads with addV
  let mut code := c1 ++ cq ++ ck ++ cv
  let mut acc : String := ""
  let mut qss : Array String := #[]; let mut kss : Array String := #[]; let mut vss : Array String := #[]
  let mut scs : Array String := #[]; let mut sms : Array String := #[]
  for hh in [0:3] do
    let h : Fin 3 := ⟨hh % 3, by omega⟩
    let (cqs, qs) ← pretty bs (.headSliceF (N := 197) (heads := 3) (d := 64) h (.operand q (zVv : Vec (197*(3*64)))))
    let (cks, ks) ← pretty bs (.headSliceF (N := 197) (heads := 3) (d := 64) h (.operand k (zVv : Vec (197*(3*64)))))
    let (cvs, vs) ← pretty bs (.headSliceF (N := 197) (heads := 3) (d := 64) h (.operand v (zVv : Vec (197*(3*64)))))
    let (ckt, kt) ← pretty bs (.transposeF (m := 197) (n := 64) (.operand ks (zVv : Vec (197*64))))
    let (cmm, qk) ← pretty bs (.matmulF (m := 197) (k := 64) (n := 197) (.operand qs (zVv : Vec (197*64))) (.operand kt (zVv : Vec (64*197))))
    let (csc, sc) ← pretty bs (.scaleF vSCALE 0 (.operand qk (zVv : Vec (197*197))))
    let (csm, sm) ← pretty bs (.softmaxRowF (m := 197) (n := 197) (.operand sc (zVv : Vec (197*197))))
    let (cpv, pv) ← pretty bs (.matmulF (m := 197) (k := 197) (n := 64) (.operand sm (zVv : Vec (197*197))) (.operand vs (zVv : Vec (197*64))))
    let (cpd, pd) ← pretty bs (.headPadF (N := 197) (heads := 3) (d := 64) h (.operand pv (zVv : Vec (197*64))))
    code := code ++ cqs ++ cks ++ cvs ++ ckt ++ cmm ++ csc ++ csm ++ cpv ++ cpd
    qss := qss.push qs; kss := kss.push ks; vss := vss.push vs; scs := scs.push sc; sms := sms.push sm
    if hh == 0 then
      acc := pd
    else
      let (cad, s) ← pretty bs (.addV (.operand acc (zVv : Vec (197*192))) (.operand pd (zVv : Vec (197*192))))
      code := code ++ cad; acc := s
  let (co, o) ← pretty bs (.denseRowF s!"%{pfx}Wo" s!"%{pfx}bo" (zMm : Mat 192 192) zVv (.operand acc (zVv : Vec (197*192))))
  let (ch, hres) ← pretty bs (.addV (.operand xin (zVv : Vec (197*192))) (.operand o (zVv : Vec (197*192))))
  let (c2, ln2) ← vlnFwd bs s!"%{pfx}g2" s!"%{pfx}bt2" hres
  let (cf1, f1) ← pretty bs (.denseRowF s!"%{pfx}Wfc1" s!"%{pfx}bfc1" (zMm : Mat 192 768) zVv (.operand ln2 (zVv : Vec (197*192))))
  let (cg, g) ← pretty bs (.geluF (.operand f1 (zVv : Vec (197*768))))
  let (cf2, f2) ← pretty bs (.denseRowF s!"%{pfx}Wfc2" s!"%{pfx}bfc2" (zMm : Mat 768 192) zVv (.operand g (zVv : Vec (197*768))))
  let (cr, bout) ← pretty bs (.addV (.operand hres (zVv : Vec (197*192))) (.operand f2 (zVv : Vec (197*192))))
  pure (code ++ co ++ ch ++ c2 ++ cf1 ++ cg ++ cf2 ++ cr,
    { xin, ln1, q, k, v, qss, kss, vss, scs, sms, att := acc, hres, ln2, f1, g, bout })

/-- Forward saves the whole-net backward references: the patch embed SSA, the per-block saves, the
    final-LN input (last block output) + output, and the logits SSA. -/
structure FwdSaves where
  embed : String
  blocks : Array BSaves
  flnIn : String        -- final-LN input (= last block output)
  fln : String          -- final-LN output (= CLS-slice input)
  clsTok : String       -- CLS-slice output (= head-dense input)
  logits : String
  deriving Inhabited

/-- The depth-12 ViT-Tiny **forward**, node-by-node. Returns (body, saves). -/
private def vitFwd12 (bs : Nat) (nClasses : Nat) : StateM Nat (String × FwdSaves) := do
  let (ce, embed) ← pretty bs (.patchEmbedF "%wConv" "%bConv" "%cls" "%pos"
    (zKk : Kernel4 192 3 16 16) zVv zVv (zMm : Mat 197 192) (.operand "%x" (zVv : Vec (3*224*224))))
  let mut code := ce
  let mut cur := embed
  let mut blocks : Array BSaves := #[]
  for i in [0:vDEPTH] do
    let (cb, sv) ← vBlockFwd bs s!"b{i}_" cur
    code := code ++ cb; cur := sv.bout; blocks := blocks.push sv
  let (cf, fl) ← vlnFwd bs "%gF" "%btF" cur
  let (cs, sl) ← pretty bs (.clsSliceF (N := 196) (D := 192) (.operand fl (zVv : Vec (197*192))))
  let (cl, logits) ← pretty bs (denseF "%Wc" "%bc" (zMm : Mat 192 nClasses) zVv (.operand sl (zVv : Vec 192)))
  pure (code ++ cf ++ cs ++ cl, { embed, blocks, flnIn := cur, fln := fl, clsTok := sl, logits })

/-- Per-block func-arg signature (committed forward order). -/
def blkArgSig (i : Nat) : String :=
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
def vitFwdRenderV (funcName : String := "vit_fwd") (bs : Nat := 32)
    (nClasses : Nat := 10) : String :=
  let (body, sv) := (vitFwd12 bs nClasses).run' 0
  let res := sv.logits
  let blkSigs := String.intercalate ", " ((List.range vDEPTH).map blkArgSig)
  let argSig := s!"%x: {ty [bs, 3*224*224]}, %wConv: {ty [192,3,16,16]}, %bConv: {ty [192]}, " ++
    s!"%cls: {ty [192]}, %pos: {ty [197,192]}, " ++ blkSigs ++
    s!", %gF: {ty [192]}, %btF: {ty [192]}, %Wc: {ty [192,nClasses]}, %bc: {ty [nClasses]}"
  "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> {ty [bs, nClasses]} " ++ "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  body ++ s!"    return {res} : {ty [bs, nClasses]}\n" ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════════════════════════════
-- § BACKWARD render — node-by-node reverse of `vBlockFwd`/`vitFwd12`, then the 200-param SGD tail
-- ════════════════════════════════════════════════════════════════════════════════════════

/-- Rowwise-dense **bias** tail: the un-fused gradient in `adam` mode, the fused SGD update
    otherwise. The two ops have the same output shape and the `*Grad` emit is a byte-prefix of the
    `*Sgd` one (`tests/TestBatchedEmitTie.lean`), so this is the only place the two tails differ —
    one backward traversal, two endings, which is what keeps `vit_train_step.mlir` byte-identical
    while `vit_adam_train_step.mlir` gets its gradients. -/
private def rdB (bs : Nat) (adam : Bool) (c : Nat) (bN lrS dy : String) : StateM Nat (String × String) :=
  if adam then pretty bs (.rowDenseBiasGrad (N := 197) (c := c) (.operand dy (zVv : Vec (197*c))))
  else pretty bs (.rowDenseBiasSgd (N := 197) (c := c) bN lrS (zVv : Vec c) 0
                    (.operand dy (zVv : Vec (197*c))))

/-- Rowwise-dense **weight** tail, same dispatch. -/
private def rdW (bs : Nat) (adam : Bool) (a c : Nat) (xSSA wN lrS dy : String) : StateM Nat (String × String) :=
  if adam then pretty bs (.rowDenseWeightGrad (N := 197) (a := a) (c := c) xSSA
                            (zVv : Vec (197*a)) (.operand dy (zVv : Vec (197*c))))
  else pretty bs (.rowDenseWeightSgd (N := 197) (a := a) (c := c) xSSA wN lrS
                    (zVv : Vec (197*a)) (zMm : Mat a c) 0 (.operand dy (zVv : Vec (197*c))))

/-- One **vector-LN backward** site (the reverse of `vlnFwd`): given the LN-output cotangent `dyOut`
    on `[197,192]` and the saved LN INPUT `xin`, emit (a) the β tail (`rowDenseBias{Sgd,Grad}`,
    dβ=Σ dy), (b) the γ tail (`veclnGamma{Sgd,Grad}`, dγ=Σ dy⊙x̂ recomputed from `xin`), and (c) the
    input cotangent `dxin = lnRowBack(γ=1)(rowScale γ (dy))` (back through normalize after the
    γ-scale). Returns (code, dxin, ngamma, nbeta) — updates at `adam := false`, gradients at `true`.
    `lr` is the mean-loss-equiv literal, and is unused in adam mode. -/
private def vlnBack (bs : Nat) (gName btName xin dyOut lrStr : String) (adam : Bool) :
    StateM Nat (String × String × String × String) := do
  let (cb, nb) ← rdB bs adam 192 btName lrStr dyOut
  let (cg, ng) ← if adam then
      pretty bs (.veclnGammaGrad (N := 197) (D := 192) xin vEPS 0
                    (zVv : Vec (197*192)) (.operand dyOut (zVv : Vec (197*192))))
    else
      pretty bs (.veclnGammaSgd (N := 197) (D := 192) gName xin vEPS lrStr 0
                    (zVv : Vec (197*192)) (zVv : Vec 192) 0 (.operand dyOut (zVv : Vec (197*192))))
  let (cs, da) ← pretty bs (.rowScaleF (m := 197) (n := 192) gName (zVv : Vec 192)
                              (.operand dyOut (zVv : Vec (197*192))))
  let (cn, dx) ← pretty bs (.lnRowBack (m := 197) (n := 192) "%one" xin vEPS 0 1 (zVv : Vec (197*192))
                              (.operand da (zVv : Vec (197*192))))
  pure (cb ++ cg ++ cs ++ cn, dx, ng, nb)

/-- One **transformer block backward** (the reverse of `vBlockFwd`), prefix `pfx`, with the forward
    saves `sv` and the block-output cotangent `dyOut`. Threads the cotangent in reverse:
    +res₂ fan-out → fc2-back → GELU-back → fc1-back → LN2-back → +res₁ fan-in (dhres) → out-dense-back
    → per-head SDPA backward (slice-back / matmul-backs / softmax-back / scale / transpose-backs) →
    Q/K/V-dense-back (summed) → LN1-back → +res₁ fan-in (dxin). Returns (code, dxin, the 16 param
    SGD-update SSAs in `BlockParams` order: g1,bt1, Wq,bq,Wk,bk,Wv,bv,Wo,bo, g2,bt2, Wfc1,bfc1,Wfc2,bfc2). -/
private def vBlockBack (bs : Nat) (pfx : String) (sv : BSaves) (dyOut lrStr : String) (adam : Bool) :
    StateM Nat (String × String × List String) := do
  let p := pfx
  -- ─ MLP sublayer back: bout = addV(hres, f2); df2 = dyOut, dhres ⊇ dyOut ─
  -- fc2: f2 = denseRow(Wfc2,bfc2)(g)  [g:197*768 → f2:197*192]
  let (c1, dg) ← pretty bs (.denseRowBack (N := 197) (a := 768) (c := 192) s!"%{p}Wfc2" (zMm : Mat 768 192)
                              (.operand dyOut (zVv : Vec (197*192))))
  let (c2, nWfc2) ← rdW bs adam 768 192 sv.g s!"%{p}Wfc2" lrStr dyOut
  let (c3, nbfc2) ← rdB bs adam 192 s!"%{p}bfc2" lrStr dyOut
  -- gelu: g = gelu(f1)  [197*768]
  let (c4, df1) ← pretty bs (.geluBack (n := 197*768) sv.f1 (zVv : Vec (197*768))
                              (.operand dg (zVv : Vec (197*768))))
  -- fc1: f1 = denseRow(Wfc1,bfc1)(ln2)  [ln2:197*192 → f1:197*768]
  let (c5, dln2) ← pretty bs (.denseRowBack (N := 197) (a := 192) (c := 768) s!"%{p}Wfc1" (zMm : Mat 192 768)
                              (.operand df1 (zVv : Vec (197*768))))
  let (c6, nWfc1) ← rdW bs adam 192 768 sv.ln2 s!"%{p}Wfc1" lrStr df1
  let (c7, nbfc1) ← rdB bs adam 768 s!"%{p}bfc1" lrStr df1
  -- LN2 back (input = hres)
  let (c8, dhresLn2, ng2, nbt2) ← vlnBack bs s!"%{p}g2" s!"%{p}bt2" sv.hres dln2 lrStr adam
  -- dhres = dyOut (res₂ skip) + dhresLn2 (LN2 path)
  let (c9, dhres) ← pretty bs (.addV (.operand dyOut (zVv : Vec (197*192))) (.operand dhresLn2 (zVv : Vec (197*192))))
  -- ─ Attention sublayer back: hres = addV(xin, o); do = dhres, dxin ⊇ dhres ─
  -- out-dense: o = denseRow(Wo,bo)(acc)  [acc=att:197*192 → o:197*192]
  let (c10, dacc) ← pretty bs (.denseRowBack (N := 197) (a := 192) (c := 192) s!"%{p}Wo" (zMm : Mat 192 192)
                              (.operand dhres (zVv : Vec (197*192))))
  let (c11, nWo) ← rdW bs adam 192 192 sv.att s!"%{p}Wo" lrStr dhres
  let (c12, nbo) ← rdB bs adam 192 s!"%{p}bo" lrStr dhres
  -- per-head SDPA backward; accumulate dq/dk/dv over the 3 heads
  let mut code := c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++ c8 ++ c9 ++ c10 ++ c11 ++ c12
  let mut dqAcc : String := ""; let mut dkAcc : String := ""; let mut dvAcc : String := ""
  for hh in [0:3] do
    let h : Fin 3 := ⟨hh % 3, by omega⟩
    -- pd[h] = headPad(h)(pv[h]); dpv = headSlice(h)(dacc)
    let (ca, dpv) ← pretty bs (.headSliceF (N := 197) (heads := 3) (d := 64) h (.operand dacc (zVv : Vec (197*(3*64)))))
    -- pv[h] = matmul(sm[h][197,197], vs[h][197,64]) → [197,64]
    let (cb, vsT) ← pretty bs (.transposeF (m := 197) (n := 64) (.operand (sv.vss[hh]!) (zVv : Vec (197*64))))
    let (cc, dsm) ← pretty bs (.matmulF (m := 197) (k := 64) (n := 197) (.operand dpv (zVv : Vec (197*64))) (.operand vsT (zVv : Vec (64*197))))
    let (cd, smT) ← pretty bs (.transposeF (m := 197) (n := 197) (.operand (sv.sms[hh]!) (zVv : Vec (197*197))))
    let (ce, dvs) ← pretty bs (.matmulF (m := 197) (k := 197) (n := 64) (.operand smT (zVv : Vec (197*197))) (.operand dpv (zVv : Vec (197*64))))
    -- sm[h] = softmaxRow(sc[h]); dsc = softmaxRowBack(sc[h])(dsm)
    let (cf, dsc) ← pretty bs (.softmaxRowBack (m := 197) (n := 197) (sv.scs[hh]!) (zVv : Vec (197*197)) (.operand dsm (zVv : Vec (197*197))))
    -- sc[h] = scale(qk[h]); dqk = scale(dsc)
    let (cg2, dqk) ← pretty bs (.scaleF vSCALE 0 (.operand dsc (zVv : Vec (197*197))))
    -- qk[h] = matmul(qs[h][197,64], kt[h][64,197]); dqs = dqk·ktᵀ = matmul(dqk[197,197], ks[h][197,64])
    let (ch, dqs) ← pretty bs (.matmulF (m := 197) (k := 197) (n := 64) (.operand dqk (zVv : Vec (197*197))) (.operand (sv.kss[hh]!) (zVv : Vec (197*64))))
    -- dkt = qsᵀ·dqk = matmul(qsᵀ[64,197], dqk[197,197]) → [64,197]; dks = transpose(dkt) → [197,64]
    let (ci, qsT) ← pretty bs (.transposeF (m := 197) (n := 64) (.operand (sv.qss[hh]!) (zVv : Vec (197*64))))
    let (cj, dkt) ← pretty bs (.matmulF (m := 64) (k := 197) (n := 197) (.operand qsT (zVv : Vec (64*197))) (.operand dqk (zVv : Vec (197*197))))
    let (ck, dks) ← pretty bs (.transposeF (m := 64) (n := 197) (.operand dkt (zVv : Vec (64*197))))
    -- scatter each head's grad back into the [197,192] feature block
    let (cl, dqH) ← pretty bs (.headPadF (N := 197) (heads := 3) (d := 64) h (.operand dqs (zVv : Vec (197*64))))
    let (cm, dkH) ← pretty bs (.headPadF (N := 197) (heads := 3) (d := 64) h (.operand dks (zVv : Vec (197*64))))
    let (cn, dvH) ← pretty bs (.headPadF (N := 197) (heads := 3) (d := 64) h (.operand dvs (zVv : Vec (197*64))))
    code := code ++ ca ++ cb ++ cc ++ cd ++ ce ++ cf ++ cg2 ++ ch ++ ci ++ cj ++ ck ++ cl ++ cm ++ cn
    if hh == 0 then
      dqAcc := dqH; dkAcc := dkH; dvAcc := dvH
    else
      let (cq, dqs2) ← pretty bs (.addV (.operand dqAcc (zVv : Vec (197*192))) (.operand dqH (zVv : Vec (197*192))))
      let (cr, dks2) ← pretty bs (.addV (.operand dkAcc (zVv : Vec (197*192))) (.operand dkH (zVv : Vec (197*192))))
      let (cs, dvs2) ← pretty bs (.addV (.operand dvAcc (zVv : Vec (197*192))) (.operand dvH (zVv : Vec (197*192))))
      code := code ++ cq ++ cr ++ cs; dqAcc := dqs2; dkAcc := dks2; dvAcc := dvs2
  -- Q/K/V dense backward: q/k/v = denseRow(W*,b*)(ln1)  [ln1:197*192 → 197*192]
  let (cq1, dln1q) ← pretty bs (.denseRowBack (N := 197) (a := 192) (c := 192) s!"%{p}Wq" (zMm : Mat 192 192) (.operand dqAcc (zVv : Vec (197*192))))
  let (cq2, nWq) ← rdW bs adam 192 192 sv.ln1 s!"%{p}Wq" lrStr dqAcc
  let (cq3, nbq) ← rdB bs adam 192 s!"%{p}bq" lrStr dqAcc
  let (ck1, dln1k) ← pretty bs (.denseRowBack (N := 197) (a := 192) (c := 192) s!"%{p}Wk" (zMm : Mat 192 192) (.operand dkAcc (zVv : Vec (197*192))))
  let (ck2, nWk) ← rdW bs adam 192 192 sv.ln1 s!"%{p}Wk" lrStr dkAcc
  let (ck3, nbk) ← rdB bs adam 192 s!"%{p}bk" lrStr dkAcc
  let (cv1, dln1v) ← pretty bs (.denseRowBack (N := 197) (a := 192) (c := 192) s!"%{p}Wv" (zMm : Mat 192 192) (.operand dvAcc (zVv : Vec (197*192))))
  let (cv2, nWv) ← rdW bs adam 192 192 sv.ln1 s!"%{p}Wv" lrStr dvAcc
  let (cv3, nbv) ← rdB bs adam 192 s!"%{p}bv" lrStr dvAcc
  -- dln1 = dln1q + dln1k + dln1v
  let (cs1, dln1a) ← pretty bs (.addV (.operand dln1q (zVv : Vec (197*192))) (.operand dln1k (zVv : Vec (197*192))))
  let (cs2, dln1) ← pretty bs (.addV (.operand dln1a (zVv : Vec (197*192))) (.operand dln1v (zVv : Vec (197*192))))
  -- LN1 back (input = xin)
  let (cl1, dxinLn1, ng1, nbt1) ← vlnBack bs s!"%{p}g1" s!"%{p}bt1" sv.xin dln1 lrStr adam
  -- dxin = dhres (res₁ skip) + dxinLn1 (LN1 path)
  let (cx, dxin) ← pretty bs (.addV (.operand dhres (zVv : Vec (197*192))) (.operand dxinLn1 (zVv : Vec (197*192))))
  let names := [ng1, nbt1, nWq, nbq, nWk, nbk, nWv, nbv, nWo, nbo, ng2, nbt2, nWfc1, nbfc1, nWfc2, nbfc2]
  pure (code ++ cq1 ++ cq2 ++ cq3 ++ ck1 ++ ck2 ++ ck3 ++ cv1 ++ cv2 ++ cv3 ++ cs1 ++ cs2 ++ cl1 ++ cx, dxin, names)

/-- The 16 per-block return TYPE strings (BlockParams order), matching `blkArgSig`. -/
private def blkRetTys : List String :=
  [ty [192], ty [192], ty [192,192], ty [192], ty [192,192], ty [192], ty [192,192], ty [192],
   ty [192,192], ty [192], ty [192], ty [192], ty [192,768], ty [768], ty [768,192], ty [192]]

/-- The whole-net backward traversal, SHARED by the SGD and AdamW renders. Returns the emitted code
    and, in func-arg order, one SSA per parameter — the **updated param** at `adam := false`, the
    **un-fused gradient** at `adam := true`. One traversal, two tails: the alternative was a second
    copy of the depth-12 backward, which is the double-writer disease one level down. -/
private def vitBackAll (bs : Nat) (nClasses : Nat) (lrStr : String) (adam : Bool)
    (smooth : Option (String × String × String) := none) :
    StateM Nat (String × List String × String) := do
    let (fwd, sv) ← vitFwd12 bs nClasses
    -- loss cotangent. The softmax is `pretty`d on its own line rather than nested inside the
    -- `.sub`, so its SSA can also feed the report-only `%loss`; `.operand` is a leaf that emits
    -- nothing, so the fresh-name sequence — and therefore the text — is unchanged (checked: the
    -- SGD artifact stays byte-identical).
    let (cSm, nSm) ← pretty bs (.softmaxDiv (.expe (.operand sv.logits (zVv : Vec nClasses))))
    let (cD0, nD0) ← pretty bs (.sub (.operand nSm (zVv : Vec nClasses)) (.operand "%onehot" (zVv : Vec nClasses)))
    -- `none` → plain CE with the batch mean folded into `lrStr` (the SGD recipe, unchanged).
    -- `some (α, −α/K, B)` → the LABEL-SMOOTHED cotangent with an explicit ÷B, which is what the
    -- AdamW recipe uses: dy = ((softmax − onehot) + α·onehot − α/K) / B. `shiftB`/`divConstB` at
    -- `N := 1` are the per-example forms — their emit reads the width off `n` and ignores `N`, and
    -- both are POINTWISE (not batch-reducing), so the §2b `N := 1` hazard does not apply.
    -- `shiftB` emits `add x, dense<−α/K>` where the hand-written render emits `subtract x, α/K`;
    -- IEEE subtraction *is* addition of the exact negation, so the two are bit-identical.
    let (cSmooth, nDy) ← match smooth with
      | none => pure ("", nD0)
      | some (aStr, negAK, bStr) => do
          let (c1, n1) ← pretty bs (.scaleF (n := nClasses) aStr 0 (.operand "%onehot" (zVv : Vec nClasses)))
          let (c2, n2) ← pretty bs (.addV (.operand nD0 (zVv : Vec nClasses)) (.operand n1 (zVv : Vec nClasses)))
          let (c3, n3) ← pretty bs (.shiftB (N := 1) (n := nClasses) negAK 0 (.operand n2 (zVv : Vec (1 * nClasses))))
          let (c4, n4) ← pretty bs (.divConstB (N := 1) (n := nClasses) bStr 0 (.operand n3 (zVv : Vec (1 * nClasses))))
          pure (c1 ++ c2 ++ c3 ++ c4, n4)
    let cDy := cSm ++ cD0 ++ cSmooth
    -- head: logits = denseF(Wc,bc)(clsTok)  [clsTok:192 → logits:nClasses]
    let (cDc, dcls) ← pretty bs (.dotOut (m := 192) (n := nClasses) "%Wc" (zMm : Mat 192 nClasses) (.operand nDy (zVv : Vec nClasses)))
    let (cWc, nWc) ← if adam then
        pretty bs (.weightGrad sv.clsTok (zVv : Vec 192) (.operand nDy (zVv : Vec nClasses)))
      else
        pretty bs (.weightSgd sv.clsTok "%Wc" lrStr (zVv : Vec 192) (zMm : Mat 192 nClasses) 0 (.operand nDy (zVv : Vec nClasses)))
    let (cbc, nbc) ← if adam then
        pretty bs (.biasGrad (.operand nDy (zVv : Vec nClasses)))
      else
        pretty bs (.biasSgd "%bc" lrStr (zVv : Vec nClasses) 0 (.operand nDy (zVv : Vec nClasses)))
    -- scatter the CLS-token cotangent back into the final-LN output (row 0), zero elsewhere
    let (cPad, dfln) ← pretty bs (.clsPadF (N := 196) (D := 192) (.operand dcls (zVv : Vec 192)))
    -- final LN back (input = flnIn = last block output)
    let (cFln, dflnIn, ngF, nbtF) ← vlnBack bs "%gF" "%btF" sv.flnIn dfln lrStr adam
    -- 12 blocks reversed; thread the cotangent from the final-LN input down to the embed
    let mut code := fwd ++ cDy ++ cDc ++ cWc ++ cbc ++ cPad ++ cFln
    let mut dcur := dflnIn
    let mut blkNames : Array (List String) := #[]   -- per-block param SSAs, fwd-index order
    for j in [0:vDEPTH] do
      let i := vDEPTH - 1 - j
      let (cb, dx, names) ← vBlockBack bs s!"b{i}_" (sv.blocks[i]!) dcur lrStr adam
      code := code ++ cb; dcur := dx; blkNames := blkNames.push names
    -- `dcur` is now the patch-embed output cotangent (dembed)
    -- patch-embed params: wConv, bConv, cls (clsSlice→denseBias), pos
    let (cwC, nwConv) ← if adam then
        pretty bs (.patchEmbedWeightGrad (ic := 3) (H := 224) (W := 224) (P := 16) (N := 196) (D := 192)
                      "%ximg" (zVv : Vec (3*224*224)) (.operand dcur (zVv : Vec (197*192))))
      else
        pretty bs (.patchEmbedWeightSgd (ic := 3) (H := 224) (W := 224) (P := 16) (N := 196) (D := 192)
                      "%wConv" "%ximg" lrStr (zVv : Vec (3*224*224)) (zKk : Kernel4 192 3 16 16) 0
                      (.operand dcur (zVv : Vec (197*192))))
    let (cbC, nbConv) ← if adam then
        pretty bs (.patchEmbedBiasGrad (N := 196) (c := 192) (.operand dcur (zVv : Vec (197*192))))
      else
        pretty bs (.patchEmbedBiasSgd (N := 196) (c := 192) "%bConv" lrStr (zVv : Vec 192) 0
                      (.operand dcur (zVv : Vec (197*192))))
    let (cClSl, dclsRow) ← pretty bs (.clsSliceF (N := 196) (D := 192) (.operand dcur (zVv : Vec (197*192))))
    let (cCl, ncls) ← if adam then
        pretty bs (.denseBiasGradB (N := 1) (c := 192) (.operand dclsRow (zVv : Vec 192)))
      else
        pretty bs (.denseBiasSgdB (N := 1) (c := 192) "%cls" lrStr (zVv : Vec 192) 0
                      (.operand dclsRow (zVv : Vec 192)))
    let (cPo, npos) ← if adam then
        pretty bs (.posEmbedGrad (N := 196) (D := 192) (.operand dcur (zVv : Vec (197*192))))
      else
        pretty bs (.posEmbedSgd (N := 196) (D := 192) "%pos" lrStr (zMm : Mat 197 192) 0
                      (.operand dcur (zVv : Vec (197*192))))
    -- 200 params in func-arg order. `blkNames` was pushed in reverse build order (j=0 → block 11,
    -- …, j=11 → block 0), so `blkNames[vDEPTH-1-i]` = block i's 16 SSAs.
    let blkOutOrdered := (List.range vDEPTH).flatMap (fun i => blkNames[vDEPTH - 1 - i]!)
    pure (code ++ cwC ++ cbC ++ cClSl ++ cCl ++ cPo,
          [nwConv, nbConv, ncls, npos] ++ blkOutOrdered ++ [ngF, nbtF, nWc, nbc], nSm)

/-- The 200 parameter `(name, shape)` pairs in func-arg order — the single source for the argument
    signature, the return types, and (in the AdamW render) the `%<nm>m`/`%<nm>v` moment slots.

    `nClasses` is a real parameter as of 2026-07-31: it was the literal 10 here and in ~28 other
    places, which pinned the whole render to Imagenette and blocked the matched pair with
    `jax/MainVitImagenet.lean` (a 1000-class ViT-Tiny that already exists). -/
def vitParamSig (nClasses : Nat := 10) : List (String × List Nat) :=
  [("wConv", [192,3,16,16]), ("bConv", [192]), ("cls", [192]), ("pos", [197,192])] ++
  (List.range vDEPTH).flatMap (fun i =>
    [(s!"b{i}_g1", [192]), (s!"b{i}_bt1", [192]),
     (s!"b{i}_Wq", [192,192]), (s!"b{i}_bq", [192]),
     (s!"b{i}_Wk", [192,192]), (s!"b{i}_bk", [192]),
     (s!"b{i}_Wv", [192,192]), (s!"b{i}_bv", [192]),
     (s!"b{i}_Wo", [192,192]), (s!"b{i}_bo", [192]),
     (s!"b{i}_g2", [192]), (s!"b{i}_bt2", [192]),
     (s!"b{i}_Wfc1", [192,768]), (s!"b{i}_bfc1", [768]),
     (s!"b{i}_Wfc2", [768,192]), (s!"b{i}_bfc2", [192])]) ++
  [("gF", [192]), ("btF", [192]), ("Wc", [192,nClasses]), ("bc", [nClasses])]

-- ════════════════════════════════════════════════════════════════
-- ── ▶ `wdExcludeNormBias` — timm/DeiT `no_weight_decay` (`recipe_gaps.md` v1.4) ────────────────
--
-- The reference (`jax/Jax/Codegen.lean`'s `_wd_mask`) decays only ≥2-D weight matrices: every 1-D
-- param (biases, LayerNorm γ/β, the CLS token) and the POSITIONAL EMBEDDING (2-D, and the reason
-- the rule is not just "ndim ≥ 2") are excluded. `vitTinyImagenetConfig` sets it; `vitTinyConfig`
-- does not, which is why this is a variant rather than a default.
--
-- ⚠ IT NEEDS NO NEW OP AND NO INTERFACE CHANGE, which is what makes it a one-evening item.
-- `adamWParamF` already takes `wd` as a runtime OPERAND NAME (`wdName`) beside the ℝ `den` uses —
-- the `%lr` shape, for the `%lr` reason. So "exclude" is: bind that operand to a ZERO constant and
-- pass `wd := 0` to `den`. Same arity, same types, same everything; only 126 of 200 operand
-- strings move. That is `e9c2729`'s conv-bias spelling (§2m) one op over, and it is why the
-- DRIVER needs nothing at all — unlike EMA (a 4th region) or dropPath (extra inputs).
-- `adamWParamF_faithful` then denotes `Proofs.adamWStep … (wd := 0) …`, i.e. the no-decay update,
-- so the proof side is untouched too.

/-- **Does this parameter get weight decay?** The renderer's half of the timm rule.

    ⚠ It keys the positional embedding by NAME where the reference keys it by SHAPE
    (`p.shape == _WD_POS_SHAPE`), and that is deliberate rather than a transcription slip: the
    reference walks an unnamed pytree and has nothing else to key on, while a shape test here
    would also exclude any *other* param that happened to be 197×192. The name is the more precise
    identifier when you have one. `#guard`s below pin the resulting counts against the reference's
    own, which is what stops the two readings drifting.

    ⚠ The rule reads the RANK, and rank is the one thing that survives the layout difference
    between the two sides — the render carries `Wfc1` as `[192,768]` where the reference has
    `(768,192)`. Both are 2-D, so both decay. Measured, not assumed (§4's one-layout rule). -/
def vitWdDecays (nm : String) (ds : List Nat) : Bool := ds.length ≥ 2 && nm != "pos"

/-- The decayed / excluded split, as the renderer computes it. -/
def vitWdCounts (nClasses : Nat := 10) : Nat × Nat :=
  let d := ((vitParamSig nClasses).filter (fun (nm, ds) => vitWdDecays nm ds)).length
  (d, (vitParamSig nClasses).length - d)

-- ⭐ The reference's OWN `_wd_mask`, run over its OWN `init_params` for
-- `generated_vit_tiny_imagenet.py`, reports **200 tensors: 74 decayed, 126 excluded**, with every
-- mask leaf uniform (the decision is per-TENSOR, which is what licenses a scalar zero operand).
-- Two independent routes — that pytree walk and this signature list — must agree, and the count
-- is the cheapest thing that can disagree. §2m's `toSpecs == Layout.specs` move, one recipe knob
-- over. ⚠ nClasses does not move it: `Wc` is 2-D and `bc` 1-D at every K.
#guard vitWdCounts 10 == (74, 126)
#guard vitWdCounts 1000 == (74, 126)
-- The four that are NOT 1-D biases, spelled out, so a reader can check the interesting cases by eye.
#guard vitWdDecays "pos" [197,192] == false      -- 2-D, and excluded ANYWAY — the whole point
#guard vitWdDecays "cls" [192] == false          -- the CLS token
#guard vitWdDecays "wConv" [192,3,16,16] == true -- patch embed, 4-D
#guard vitWdDecays "Wq" [192,192] == true

/-- **ViT-Tiny depth-12 train step rendered ENTIRELY from the verified AST** — the §1 backward render.
    Forward (`vitFwd12`) → softmax-CE cotangent (`softmax(logits) − onehot`, the `lossCotGraph` form) →
    head-dense back (`dotOut` + `weightSgd`/`biasSgd`) → `clsPadF` → final-LN back (`vlnBack`) → 12×
    `vBlockBack` (reversed, cotangent threaded) → patch-embed back (`patchEmbedWeightSgd`/`patchEmbedBiasSgd`
    + `clsSliceF`→`denseBiasSgdB` for cls + `posEmbedSgd` for pos). Returns the 200 SGD-updated params in
    func-arg order. `lrStr` is the mean-loss-equiv literal (base/BS); cotangent has NO /B (folded into lr).
    The traversal itself is `vitBackAll false`, shared with the AdamW render. -/
def vitTrainStepRenderV (funcName : String := "vit_train_step") (lrStr : String := "0.003125")
    (nClasses : Nat := 10)
    (bs : Nat := 32) : String :=
  let go : StateM Nat String := do
    let (code, retNames, _) ← vitBackAll bs nClasses lrStr false
    let retTys := [ty [192,3,16,16], ty [192], ty [192], ty [197,192]] ++
      ((List.range vDEPTH).flatMap (fun _ => blkRetTys)) ++ [ty [192], ty [192], ty [192,nClasses], ty [10]]
    pure <|
      "    // ── ViT-Tiny depth-12 train step: every line is pretty(verified AST node) ──\n" ++
      code ++
      s!"    return {String.intercalate ", " retNames} : {String.intercalate ", " retTys}\n"
  let body : String := go.run' 0
  let blkSigs := String.intercalate ", " ((List.range vDEPTH).map blkArgSig)
  let argSig := s!"%x: {ty [bs, 3*224*224]}, %wConv: {ty [192,3,16,16]}, %bConv: {ty [192]}, " ++
    s!"%cls: {ty [192]}, %pos: {ty [197,192]}, " ++ blkSigs ++
    s!", %gF: {ty [192]}, %btF: {ty [192]}, %Wc: {ty [192,nClasses]}, %bc: {ty [nClasses]}, %onehot: {ty [bs, nClasses]}"
  let retTys := [ty [192,3,16,16], ty [192], ty [192], ty [197,192]] ++
    ((List.range vDEPTH).flatMap (fun _ => blkRetTys)) ++ [ty [192], ty [192], ty [192,nClasses], ty [10]]
  "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> ({String.intercalate ", " retTys}) " ++ "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %ximg = stablehlo.reshape %x : ({ty [bs, 3*224*224]}) -> {ty [bs, 3, 224, 224]}\n" ++
  body ++ "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § The certified AdamW train step (handoff §2a-quinquies follow-on, step 2b)
-- ════════════════════════════════════════════════════════════════

/-- `(θ', m', v')` for one parameter from its un-fused gradient — the proven
    `adamMNextF`/`adamVNextF`/`adamWParamF` triple (`adamW_triple_faithful` bundles their `den`s
    into `Proofs.adamWStep` by `rfl`). β₁/β₂/ε/wd are baked literals; `%lr`/`%bc1`/`%bc2` are
    runtime `tensor<f32>` args, so one render serves the whole cosine+warmup schedule. Mirrors
    `ResNet34RenderB.adamOne`, minus the replica collective (ViT has no DP render yet). -/
private def vitAdamOne (bs : Nat) (nm : String) (ds : List Nat) (gradSSA : String) (replicas : Nat)
    (ema : Bool := false) (wdName : String := "%wd") (preAvg : Bool := false) :
    StateM Nat (String × String × String × String × String) := do
  let n := ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  -- ⚠ `preAvg` says the caller has ALREADY averaged (and clipped) this gradient, so the collective
  -- must not be emitted a second time. It exists because **under data parallelism the clip has to
  -- come AFTER the `all_reduce`**: the reference clips the whole, already-combined gradient, so
  -- clipping per replica clips 200 PARTIAL gradients — a different function that compiles, trains,
  -- descends, and that no structural check sees (`planning/grad_clip.md` §4). The clip needs every
  -- gradient at once, and this op is per parameter, so at `clip := true` the caller hoists both the
  -- collective and the clip above the loop and passes the result here.
  let replicas := if preAvg then 1 else replicas
  -- At `replicas > 1` the gradient is averaged across devices first. **That collective is a TRUSTED
  -- CARVE-OUT** — emitted text, not `pretty` of an AST node, so it sits outside every faithfulness
  -- theorem here, exactly as in §2b-quater. The `den` side does not shift: the AdamW triple consumes
  -- the averaged gradient as an `.operand` just as it consumed the raw one. Claim ceiling is §5's —
  -- *the gradient averaging is a proven identity; the collective implementing it is trusted, exactly
  -- like the lowerer.* At `replicas ≤ 1` this emits NOTHING, which is the cheap self-check that the
  -- insertion is inert (the single-device render re-renders byte-identical).
  let (arS, gAvg) := ViTRender.emitGradAllReduce gradSSA ds nm replicas
  let gr : SHlo n := .operand gAvg z
  let (cM, nM) ← pretty bs (.adamMNextF s!"%{nm}m" "%b1" "%ob1" ds 0 z gr)
  let (cV, nV) ← pretty bs (.adamVNextF s!"%{nm}v" "%b2" "%ob2" ds 0 z gr)
  -- ⚠ `wdName` and `den`'s `wd` must move TOGETHER. They are both 0 here only because every
  -- literal in this constructor is already 0 (the ℝ slots are placeholders the emit ignores); if
  -- that ever changes, an excluded site must pass `wd := 0` as well as `%wdz`, or the artifact and
  -- the denotation describe different optimizers — §2a's two-writers disease inside one op.
  let (cT, nT) ← pretty bs (.adamWParamF s!"%{nm}" s!"%{nm}m" s!"%{nm}v" "%b1" "%ob1"
                    "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" wdName ds 0 0 0 0 0 0 0 z z z gr)
  -- ▶ THE EMA SHADOW, and as on ConvNeXt/EfficientNet it needs NO new op:
  -- `Proofs.adamMNext β₁ m g = β₁·m + (1−β₁)·g` IS the reference's `ema_update`
  -- (`jax/Jax/Codegen.lean:2459`) at `(β₁ := d, m := ema, g := θ')`, so `adamMNextF` renders it and
  -- `adamMNextF_faithful` closes the denotation side by `rfl`. Fourth net, same reading.
  --
  -- ⚠ It consumes `nT`, the UPDATED parameter, not the gradient — the shadow averages WEIGHTS. It
  -- therefore sits downstream of the whole AdamW triple, and (at `replicas > 1`) downstream of the
  -- collective, which is why the shadow and the all_reduce cannot interact.
  -- ⚠ `%emad`/`%oemad` are function ARGS, not constants, because the reference's decay is
  -- TIME-VARYING: `d = min(decay, (1+t)/(10+t))`, TF's warmup-corrected form. `planning/ema.md` §2
  -- has the reference's own measurement of dropping it — a shadow holding 12.8% of the random init
  -- and scoring 0.00% top-1 while the live weights scored 70.48%.
  --
  -- At `ema := false` NO `pretty` call happens, so the fresh-name counter does not move and all
  -- TEN committed `vit*`/`vitin*` artifacts re-render byte-identically. Gate 1's strong form, free.
  let (cE, nE) ← if ema then
      pretty bs (.adamMNextF s!"%{nm}e" "%emad" "%oemad" ds 0 z (.operand nT z))
    else pure ("", "")
  pure (arS ++ cM ++ cV ++ cT ++ cE, nT, nM, nV, nE)

/-- The driver's **variant slug** for a (per-device batch, replica count, EMA) triple: the artifact
    is `verified_mlir/vit_<variant>_train_step.mlir`, the entry point is `@vit_<variant>_train_step`
    and `LEAN_MLIR_VARIANT=<variant>` selects it at run time.

    This is the ViT peer of `cnxAdamVariant` / `r34AdamVariant` / `mnv2AdamVariant`, and unlike
    theirs it is **documentation plus a drift guard rather than the name's producer** —
    `vitAdamTrainStepFaithful` takes `funcName` explicitly (it predates the slug convention) and the
    `#eval` paths must stay string literals for `regen_verified_mlir.sh`'s writer audit to see them.
    So the `#guard`s at the bottom of this file are what tie the literals to this function; the
    contract is checked at `lake build` rather than merely described.

    ⚠ ViT's spelling breaks the "the number is the per-device batch" convention at 4 replicas
    (`adamdp32x4`, `adamdp128x4`) and that is deliberate — `vit_adamdp_train_step.mlir` is a
    COMMITTED 2-replica artifact at bs32, so a 4-replica render reusing `adamdp` would give one path
    two writers computing different graphs. Encoded here so the exception cannot be forgotten.

    ⚠ **The `ema` marker LEADS.** `trainAdamSched` keys its 4-region `[θ|m|v|ema]` blob off
    `variant.startsWith "ema"`, so a trailing marker would silently select the 3-region layout for a
    4-region graph — every parameter misaligned. And note what that cost on EfficientNet: its
    RMSProp+EMA variant is `emarms`, which does **not** start with `"rms"`, so the mean-square would
    have initialised to 0 through a prefix test. ViT is AdamW-only, so there is no second axis here
    today; if one is ever added, make both predicates substring tests first. -/
def vitAdamVariant (bs : Nat := 32) (replicas : Nat := 1) (ema : Bool := false)
    (wdExclude : Bool := false) (clip : Bool := false) : String :=
  (if ema then "ema" else "adam")
    ++ (if replicas ≤ 1 then "" else "dp")
    ++ (if bs == 32 && replicas ≤ 2 then "" else toString bs)
    ++ (if replicas > 2 then s!"x{replicas}" else "")
    -- ▶ `wx` = timm no_weight_decay. TRAILING, and checked against every CONCATENATION rather than
    -- against the other markers one at a time — that is §0's `sd`/`rmsdp` finding, where the
    -- collision was between two OTHER markers meeting. `tests/TestVariantPredicates.lean` runs the
    -- `wx` spellings through all three driver predicates.
    --
    -- ⚠ Unlike `ema`, `rms` and `drop`, this marker needs NO driver predicate at all: excluding a
    -- param from decay binds a different CONSTANT to `%wd` and changes no arity, no type and no
    -- region. It is a pure render variant. The name exists so the artifact says which recipe it is,
    -- not so anything switches on it.
    --
    -- ▶ `clip` = global-norm gradient clipping (`planning/grad_clip.md`), TRAILING and AFTER `wx`,
    -- because the ViT/ConvNeXt reference sets BOTH (`gradClipNorm := 1.0` and
    -- `wdExcludeNormBias := true`) — so `wx` ++ `clip` is the shipping spelling, not either alone,
    -- and a feature that is fine alone and wrong composed is the `emarms` failure. Like `wx` it
    -- needs no driver predicate: the clip changes no arity, no type and no region.
    ++ (if wdExclude then "wx" else "")
    ++ (if clip then "clip" else "")

/-- β₁/β₂/ε/wd as graph constants — the ViT-Tiny AdamW recipe (`vitTinyConfig`: lr 3e-4, wd 1e-4).

    ⚠ **`wdStr` is a parameter because the two ViT configs DISAGREE ON IT BY 500×**, and that was
    found while gating `wdExcludeNormBias` rather than by reading the configs: `vitTinyConfig`
    (Imagenette) sets `weightDecay := 1e-4`, which is the literal this file baked for every ViT
    render — but **`vitTinyImagenetConfig` sets 0.05**, the DeiT value. So an ImageNet render at
    the baked default trains at 1/500th of its reference's decay. It is the `RenderCifar8Sgd02` /
    EfficientNet-16× shape (§2a-quater): a silently wrong hyperparameter in a committed artifact,
    which compiles, runs and descends. The default is unchanged, so every existing artifact keeps
    its bytes; only the ImageNet `wx` render passes 0.05.

    ⚠ **`vitin_adam128` and `vitin_adamdp128x4` are STILL at 1e-4** and are not touched here —
    changing them is a separate call with its own blast radius (the DP peer, the residency-gate
    row). Recorded as owed in `recipe_gaps.md` rather than fixed in passing. -/
private def vitAdamConsts (wdExclude : Bool := false) (wdStr : String := "0.0001") : String :=
  (if wdExclude then
    -- The no-decay operand. Declared ONLY when the variant is on, so every committed artifact
    -- re-renders byte-identically at the default — gate 1 in its strong form, for free.
    "    // ── timm no_weight_decay (wdExcludeNormBias): 126 of 200 params take %wdz, not %wd ──\n" ++
    "    %wdz = stablehlo.constant dense<0.0> : tensor<f32>\n"
   else "") ++
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  s!"    %wd = stablehlo.constant dense<{wdStr}> : tensor<f32>\n"

/-- **ViT-Tiny depth-12 AdamW train step, rendered from the verified AST.** The certified peer of
    the hand-written `ViTRender.vitTrainStepModuleAdamSched` that `vit-verified-adam` has been
    emitting at startup.

    Same backward as `vit_train_step` (`vitBackAll`, one traversal) but taking the **un-fused
    gradients**, each fed to the proven AdamW triple. The cotangent is the LABEL-SMOOTHED one with
    an explicit ÷B, matching the AdamW recipe — the SGD render folds the mean into `lr` and does no
    smoothing, so the two are different functions and this parameter is not optional.

    Interface: 605 in (`%x`, 200 θ, 200 m, 200 v, `%lr`/`%bc1`/`%bc2`, `%onehot`) / 603 out
    (200 θ', 200 m', 200 v', `%loss`/`%bc1`/`%bc2`) — positionally identical to the hand-written
    render, so `trainAdamSched`'s packed `[θ|m|v]` protocol is unchanged.

    At `ema := true` (`planning/ema.md`) the blob gains a **fourth region** and the scalar tail goes
    3 → 5, so the interface becomes **807 in / 805 out** = 605/603 + 200 (the shadow) + 2
    (`%emad`/`%oemad`). ⚠ `ema` is LAST in this signature on purpose: inserted mid-list it would
    capture an existing positional argument at every call site, which is the mnv2/enet `convBias`
    lesson (§2m). -/
def vitAdamTrainStepFaithful (funcName : String := "vit_adam_train_step")
    (bStr : String := "32.0") (replicas : Nat := 1) (bs : Nat := 32)
    (nClasses : Nat := 10) (alpha : Float := 0.1) (ema : Bool := false)
    (wdExclude : Bool := false) (wdStr : String := "0.0001")
    (clip : Bool := false) (clipStr : String := "1.0") : String :=
  -- ⚠ α and K are the ONLY knobs; every emitted smoothing constant is derived from them here.
  -- Passing the cotangent's `−α/K` as a separate string (which is what this took until
  -- 2026-07-31) is the same two-writers-for-one-fact shape §2a spent a thread removing: the two
  -- agree until someone changes K, and then the gradient and the loss disagree silently.
  let alphaStr := fmt6 alpha
  let negAlphaKStr := "-" ++ alphaOverK nClasses alpha
  let go : StateM Nat String := do
    let (code, gradNames, nSm) ← vitBackAll bs nClasses "0.0" true (some (alphaStr, negAlphaKStr, bStr))
    -- ▶ GLOBAL-NORM GRADIENT CLIPPING (`planning/grad_clip.md`, `recipe_gaps.md` v1.4b) — the
    -- reference's `gn = sqrt(sum(jnp.sum(g*g) for g in tree.leaves(grads)))` then
    -- `g * min(1, CLIP/(gn + 1e-6))`, applied to ALL 200 gradients before the optimizer sees them.
    --
    -- ⚠⚠ THE ORDER IS THE SEMANTICS, TWICE OVER:
    --   1. the norm is GLOBAL — one scalar folded from every parameter, so the fold must run to
    --      completion before any parameter is scaled. That is why it is hoisted out of the loop.
    --   2. under DP the clip goes AFTER the `all_reduce`. `vitAdamOne` normally emits the collective
    --      per parameter, which is downstream of where the clip has to be, so at `clip := true`
    --      the collective is hoisted here too and the loop is told (`preAvg`) not to repeat it.
    --
    -- ⚠ At `clip := false` NOT ONE `pretty` CALL HAPPENS BELOW, so the fresh-name counter does not
    -- move and every committed `vit*`/`vitin*` artifact re-renders byte-identically — gate 1's
    -- strong form, free. That is the `ema := false` route, and taking it is deliberate: splitting
    -- the loop unconditionally would shift the all_reduce text relative to the adam blocks and
    -- re-render every `*dp*` artifact differently with the feature OFF.
    let zero1 : Vec 1 := fun _ => 0
    let mut avgNames := gradNames
    let mut clipCode := ""
    let mut clipped : List String := []
    if clip then
      -- 1. the collective, hoisted — emits NOTHING at `replicas ≤ 1`, so the single-device clip
      --    render is unaffected by this branch existing.
      let mut avg : List String := []
      for i in [0:(vitParamSig nClasses).length] do
        let (nm, ds) := (vitParamSig nClasses)[i]!
        let (arS, gAvg) := ViTRender.emitGradAllReduce (gradNames[i]!) ds nm replicas
        clipCode := clipCode ++ arS
        avg := avg ++ [gAvg]
      avgNames := avg
      -- 2. the fold, seeded at `%zero` (already in this render's preamble). Left-nested
      --    `gradSumSqAccF`, one leaf per parameter — an ordinary `SHlo` TREE, not a shared DAG
      --    node, because every gradient is already an `.operand` leaf and nothing is recomputed.
      let mut total : SHlo 1 := .operand "%zero" zero1
      for i in [0:(vitParamSig nClasses).length] do
        let (_, ds) := (vitParamSig nClasses)[i]!
        let n := ds.foldl (· * ·) 1
        total := .gradSumSqAccF (n := n) ds total (.operand (avgNames[i]!) (fun _ => 0))
      let (cN, normSSA) ← pretty bs total
      clipCode := clipCode ++ cN
      -- 3. the rescale, one per parameter, every site reading the SAME `normSSA` and the same
      --    `clipStr`/`epsStr` — which is what makes the factor shared (`clipShared_faithful`).
      for i in [0:(vitParamSig nClasses).length] do
        let (_, ds) := (vitParamSig nClasses)[i]!
        let n := ds.foldl (· * ·) 1
        let z : Vec n := fun _ => 0
        let (cS, sSSA) ← pretty bs (.clipScaleF (n := n) clipStr "0.000001" 0 0 ds
                            (.operand normSSA zero1) (.operand (avgNames[i]!) z))
        clipCode := clipCode ++ cS
        clipped := clipped ++ [sSSA]
    -- one triple per parameter, in func-arg order
    let mut adamCode := clipCode
    let mut thetaN : List String := []
    let mut mN : List String := []
    let mut vN : List String := []
    let mut eN : List String := []
    for i in [0:(vitParamSig nClasses).length] do
      let (nm, ds) := (vitParamSig nClasses)[i]!
      -- The wd operand is chosen from the SAME `vitParamSig` entry that names the site, never a
      -- parallel list — §2e's silent-slot rule: a misaligned mask would decay the wrong 126 params
      -- and nothing in the arity, the types or the prefix audit would notice.
      let wdN := if wdExclude && !vitWdDecays nm ds then "%wdz" else "%wd"
      let gSSA := if clip then clipped[i]! else gradNames[i]!
      let (c, nT, nM, nV, nE) ← vitAdamOne bs nm ds gSSA replicas ema wdN clip
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]; mN := mN ++ [nM]; vN := vN ++ [nV]
      if ema then eN := eN ++ [nE]
    -- `%loss` is REPORT-ONLY and on no gradient path, so NO theorem covers it — §2b shipped plain
    -- CE here against a smoothed-CE cotangent and only the numeric tie caught it. It is therefore
    -- built from the SAME smoothed recipe the cotangent implies, and declared as a carve-out.
    let lossCode :=
      "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
      s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %llog = stablehlo.log {nSm} : {ty [bs, nClasses]}\n" ++
      s!"    %lohll = stablehlo.multiply %onehot, %llog : {ty [bs, nClasses]}\n" ++
      s!"    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [bs, nClasses]}, tensor<f32>) -> {ty [bs]}\n" ++
      s!"    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [bs, nClasses]}, tensor<f32>) -> {ty [bs]}\n" ++
      -- ⚠ These were the literals `0.900000` / `0.010000` — i.e. (1−α) and α/K baked at K = 10.
      -- That is EXACTLY the bug §2k found in the R34 ImageNet render, where the same hardcode sat
      -- on the COTANGENT and made the objective 100× wrong at K = 1000, caught only because the
      -- reported loss was implausible. Here it was confined to the report-only `%loss` (the ViT
      -- cotangent takes its constant as an argument), so it would have produced a WRONG LOSS
      -- NUMBER against a correct gradient — the same trap read backwards, and harder to notice.
      s!"    %lomac = stablehlo.constant dense<{oneMinusAlpha}> : {ty [bs]}\n" ++
      s!"    %laKc = stablehlo.constant dense<{alphaOverK nClasses}> : {ty [bs]}\n" ++
      s!"    %llt1 = stablehlo.multiply %lomac, %lt1s : {ty [bs]}\n" ++
      s!"    %llt2 = stablehlo.multiply %laKc, %llsr : {ty [bs]}\n" ++
      s!"    %llpe = stablehlo.add %llt1, %llt2 : {ty [bs]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [bs]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{bs}.0> : tensor<f32>\n" ++
      s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
      s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
    let pTy := (vitParamSig nClasses).map (fun (_, ds) => ty ds)
    -- ⚠ THE RETURN LAYOUT MUST EQUAL THE INPUT LAYOUT, region for region and scalar for scalar.
    -- The driver does `pbuf := out` — each step's output IS the next step's input (§2d.3's no-copy
    -- handover) — so a return list that dropped the shadow, or carried fewer scalars than the
    -- signature takes, would silently re-interpret the blob from step 2 onward rather than fail. It
    -- is also exactly what the resident shim checks (inputs `[res_in, res_in+n)` and outputs
    -- `[res_out, res_out+n)` must agree tensor for tensor), which is why a 4th region needs no C
    -- change at all. `%emad`/`%oemad` ride through unread, as `%bc1`/`%bc2` already do.
    let retVals := thetaN ++ mN ++ vN ++ eN ++ ["%loss", "%bc1", "%bc2"]
                     ++ (if ema then ["%emad", "%oemad"] else [])
    let retTys := pTy ++ pTy ++ pTy ++ (if ema then pTy else [])
                     ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"]
                     ++ (if ema then ["tensor<f32>", "tensor<f32>"] else [])
    pure <|
      (if replicas <= 1 then
        "    // ── ViT-Tiny depth-12 AdamW train step: gradients + optimizer are pretty(AST) ──\n"
       else
        "    // ── ViT-Tiny depth-12 AdamW train step, DATA-PARALLEL over " ++ toString replicas ++
        " replicas ──\n" ++
        "    // The gradients and the AdamW triple are pretty(verified AST). The per-parameter\n" ++
        "    // all_reduce(add)/N between them is NOT — it is a TRUSTED CARVE-OUT, emitted text\n" ++
        "    // outside every faithfulness theorem, exactly like the lowerer (handoff §5).\n") ++
      -- The shadow is NOT a carve-out and the banner says which it is, because §2h-quater found a
      -- committed artifact under-describing its own certification level and that is still a wrong
      -- statement in the one place a reader trusts.
      (if ema then
        "    // ── EMA WEIGHT SHADOW (planning/ema.md): a 4th [θ|m|v|ema] region, one adamMNextF\n" ++
        "    // per parameter at (β₁ := %emad) on the UPDATED weight. It is pretty(verified AST)\n" ++
        "    // like the rest of the optimizer — NOT a carve-out. EVAL AND CHECKPOINTS SCORE IT.\n"
       else "") ++
      code ++ vitAdamConsts wdExclude wdStr ++ adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let pSig := String.intercalate ", " ((vitParamSig nClasses).map (fun (nm, ds) => s!"%{nm}: {ty ds}"))
  let mSig := String.intercalate ", " ((vitParamSig nClasses).map (fun (nm, ds) => s!"%{nm}m: {ty ds}"))
  let vSig := String.intercalate ", " ((vitParamSig nClasses).map (fun (nm, ds) => s!"%{nm}v: {ty ds}"))
  let eSig := String.intercalate ", " ((vitParamSig nClasses).map (fun (nm, ds) => s!"%{nm}e: {ty ds}"))
  let argSig := s!"%x: {ty [bs, 3*224*224]}, " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++
    (if ema then ", " ++ eSig else "") ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>" ++
    (if ema then ", %emad: tensor<f32>, %oemad: tensor<f32>" else "") ++
    s!", %onehot: {ty [bs, nClasses]}"
  let pTy := (vitParamSig nClasses).map (fun (_, ds) => ty ds)
  let retTys := pTy ++ pTy ++ pTy ++ (if ema then pTy else [])
    ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"]
    ++ (if ema then ["tensor<f32>", "tensor<f32>"] else [])
  let body : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @{funcName}({argSig}) -> ({String.intercalate ", " retTys}) " ++ "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %ximg = stablehlo.reshape %x : ({ty [bs, 3*224*224]}) -> {ty [bs, 3, 224, 224]}\n" ++
  body ++ "  }\n}\n"

end Proofs.StableHLO

-- Regenerate the committed verified_mlir/vit_{fwd,train_step}.mlir from the certified renderer
-- (pure Lean, no iree) — the drift-guard source: `lake env lean LeanMlir/Proofs/Codegen/ViTRender.lean`
-- rewrites both, and proofs.yml git-diffs them. The bytes `MainViTVerified` trains on ARE these.
-- (tests/TestViT{Train,Fwd}.lean write the SAME render + additionally iree-compile on the rocm box.)
#eval IO.FS.writeFile "verified_mlir/vit_fwd.mlir" (Proofs.StableHLO.vitFwdRenderV "vit_fwd")
#eval IO.FS.writeFile "verified_mlir/vit_train_step.mlir"
  (Proofs.StableHLO.vitTrainStepRenderV "vit_train_step" "0.003125")

-- The **AdamW** train step, `pretty(provenGraph)` — the artifact `vit-verified-adam` trains on, and
-- from 2026-07-28 this `#eval` is its ONLY writer. It used to be written by the DRIVER
-- (`apps/imagenette/MainViTVerifiedAdam.lean`) at every startup, from the hand-written
-- `LeanMlir/ViTRender.vitTrainStepModuleAdamSched` — a writer pattern the artifact audit cannot
-- see, and one where the committed bytes were never authoritative because each run overwrote them.
--
-- The swap was licensed by `lake build vit-adam-tie` (retired render vs this one, one AdamW step,
-- all 16,579,041 returned floats): gradient norm-rel 1e-6, **%loss bit-exact**, 0/200 parameters
-- disagreeing, against a bit-exact A-vs-A determinism floor. `%loss` carries real weight here — ViT
-- has no BN, so it is the only output that reads the forward directly, and it is precisely what §2b
-- got wrong elsewhere. To re-run the tie, recover the retired render:
--   git show 2957188:verified_mlir/vit_adam_train_step.mlir > /tmp/retired.mlir
--   .lake/build/bin/vit-adam-tie /tmp/retired.mlir verified_mlir/vit_adam_train_step.mlir
-- α = 0.1 and K = nClasses are the knobs; −α/K is DERIVED (2026-07-31), batch 32 —
-- `vitTinyConfig`'s label smoothing + mean.
#eval IO.FS.writeFile "verified_mlir/vit_adam_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_adam_train_step" "32.0")

-- The DATA-PARALLEL render, the ViT peer of `resnet34_adamdp_train_step` (§2b-quater): the same
-- graph plus one `all_reduce(add)/N` per parameter gradient before its AdamW triple. Selected at
-- run time by `LEAN_MLIR_VARIANT=adamdp`, and `2` must match `PJRT_REPLICAS` because the graph
-- bakes `replica_groups`. Rendering to its OWN path is what stops the §2a race in which producing
-- a DP render meant editing a knob and clobbering the single-device artifact the trainer runs.
--
-- **NOT RUNNABLE ON THIS BOX, and not gated above 1 replica.** Collectives live on the XLA/PJRT
-- path, but the ViT graph fails there in the patch-embed weight-grad convolution
-- (`miopenStatusUnknownError`) — which is why `vit-adam-tie` links IREE — and `vit-verified-adam`
-- is itself an IREE binary, where the shim refuses the DP entry point rather than silently running
-- single-device. What IS checked: the `replicas = 1` re-render is byte-identical (so the insertion
-- is provably inert), the collective count, and the emitted syntax.
--
-- ✅ 2026-07-30: the 2-GPU numeric gate now exists and PASSES — `vit-dp-check` reproduces the
-- single-device step BIT-EXACTLY on all 16,579,041 returned floats on a duplicated batch, against
-- a sum-not-mean control that fires at 0.996. The paragraph above's "not runnable on this box"
-- is retired: the graph executes (handoff §2j tail).
#eval IO.FS.writeFile "verified_mlir/vit_adamdp_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_adamdp_train_step" "32.0" 2)

-- ── The LARGER-BATCH pair, added 2026-07-30 (bs 64 per device) ─────────────────────────────────
-- `bs` is a renderer PARAMETER as of this change; it used to be a private constant `vBS := 32`,
-- which is the same thing §5 flags for ConvNeXt's `cBS` and the reason no ViT batch other than 32
-- could be rendered. Threading it is licensed by the strongest gate available: at `bs := 32` all
-- four artifacts above re-render BYTE-IDENTICAL (`vit_fwd` 626cc192, `vit_train_step` f57aff00,
-- `vit_adam_train_step` 2d176895, `vit_adamdp_train_step` 02f85184), so the refactor is inert.
--
-- The number in a variant name is the PER-DEVICE batch, matching `adam128`/`adamdp128` on
-- EfficientNet and ResNet-34. So `adamdp64` on two replicas is GLOBAL batch 128.
--
-- ⚠ Two things that must move together, because the batch is baked into the graph rather than
-- being a runtime dimension:
--   * the cotangent divisor `bStr` — the render means over its OWN device's batch, and the
--     `all_reduce(add)/N` then averages across devices, so per-device mean × N-average = global
--     mean. A `bStr` left at "32.0" here would silently double every gradient.
--   * `LEAN_MLIR_BATCH=64` at run time. A mismatch is a shape error at the first invoke, not a
--     silent limp — which is the good failure mode.
--
-- ⚠⚠ **bs64 REQUIRES `MIOPEN_DEBUG_CONV_GEMM=0`; bs32 does not.** This is the sharpest evidence
-- yet on the MIOpen im2col fault, because at bs64 it is RELIABLE rather than the once-only
-- flake seen at bs32. Measured 2026-07-30 — without the variable, `vit_adam64_train_step` dies at
-- execution with `miopenStatusUnknownError`; with it, `vit-dp-check` on the bs64 pair is BIT-EXACT
-- on all 16,579,041 floats against a sum-not-mean control that fires at 1.003.
-- Why the batch matters: XLA requests the patch-embed weight-grad conv (an interior-dilated `pad`
-- fused in as `rhs_dilation = 16`) with a **zero-byte workspace**, which confines MIOpen to
-- no-workspace solvers; it lands on `GemmFwdRest`, whose `MIOpenIm2d2Col.cpp` uses the OpenCL
-- builtins `get_global_id`/`get_global_size` but is JIT-compiled through HIPRTC, where they do not
-- exist. That im2col workspace is LINEAR IN BATCH — MIOpen asks for 6,422,528 bytes at bs32 and
-- exactly 2× that, 12,845,056, at bs64 — so the larger batch pushes solver selection onto the
-- broken kernel deterministically. The variable drops that solver family, at ~7% throughput.
-- Diagnosis + a 20-line JAX reproducer: `upstream-issues/2026-06-jax-rocm-miopen-im2col-hiprtc/`.
-- The eval forwards stay at bs 32 and that is fine: `trainAdamSched` reads the width off the
-- forward artifact (`evalBs`), so eval runs at 32 while training runs at 64.
#eval IO.FS.writeFile "verified_mlir/vit_adam64_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_adam64_train_step" "64.0" 1 64)
#eval IO.FS.writeFile "verified_mlir/vit_adamdp64_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_adamdp64_train_step" "64.0" 2 64)

-- ── The FOUR-REPLICA render, added 2026-08-01 ──────────────────────────────────────────────────
-- Same graph as `vit_adamdp_train_step` at the same per-device batch (32); only `replicas` moves,
-- so `replica_groups` becomes `[[0,1,2,3]]` and every collective's divisor becomes 4.0. Global
-- batch 128.
--
-- ⚠ **The name encodes BATCH×REPLICAS, deliberately breaking the `adamdp64` convention.** Elsewhere
-- the number in a variant name is the per-device batch and the replica count is absent —
-- `r34AdamVariant 64 2` and `r34AdamVariant 64 4` both return `"momdp64"`. That is fine for R34
-- only because nothing renders the 2-replica bs64 peer; here it is not, because
-- `vit_adamdp_train_step.mlir` is a COMMITTED 2-replica artifact at bs32, so reusing `adamdp`
-- would give one path two writers computing different graphs — §2a's exact disease, and the
-- failure mode is a silent clobber rather than an error. `32x4` cannot collide with anything.
--
-- Gated by `vit-dp-check` at `VIT_DP_REPLICAS=4`: on a batch duplicated FOUR ways,
-- `all_reduce(add)/4` is again the identity, so this must reproduce the single-device bs32 step
-- output-for-output. Control: the 200 divisors 4.0 → 1.0.
--
-- ⚠ The `MIOPEN_DEBUG_CONV_GEMM=0` warnings on the bs64 pair above are **ROCm-only** — MIOpen is
-- AMD's library and there is no analogue on the CUDA path this render was gated on.
#eval IO.FS.writeFile "verified_mlir/vit_adamdp32x4_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_adamdp32x4_train_step" "32.0" 4 32)

-- ── bs128, single-device and 4-replica ────────────────────────────────────────────────────────
-- Rendered to answer "do these 16 GB cards hold bs128, i.e. global 512 on four?" — they do, with
-- room (measured 2026-08-01: 3.2 GiB peak per device at bs128×4, 20% of the card).
--
-- The single-device `adam128` is NOT optional scaffolding: `vit-dp-check` gates a DP render against
-- the single-device render AT THE SAME PER-DEVICE BATCH, so without this the 4-replica one could
-- not be gated at all. Same pairing as `adam64`/`adamdp64`.
--
-- ⚠ `bStr` MUST track the per-device batch (the render means over its own device's batch, and the
-- `all_reduce(add)/N` then averages across devices ⇒ global mean). Left at "32.0" here every
-- gradient would be 4× and nothing but a numeric gate would say so.
--
-- ⚠⚠ **Global 512 is a THROUGHPUT config, not a training one.** Imagenette is 9,469 images, so
-- global 512 is **18 steps/epoch** — 1,480 updates over 80 epochs against the 23,600 the 71.31%
-- run took. §2d.2 measured accuracy tracking step count at ~1 point per halving, accelerating at
-- the bottom (R34 lost 3.4 points going 295 → 36). Use it to measure scaling; do not read an
-- accuracy off it and compare.
#eval IO.FS.writeFile "verified_mlir/vit_adam128_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_adam128_train_step" "128.0" 1 128)
#eval IO.FS.writeFile "verified_mlir/vit_adamdp128x4_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_adamdp128x4_train_step" "128.0" 4 128)

-- ── ViT-Tiny on FULL 1000-class ImageNet, slug `vitin` — the scale tier, added 2026-08-01 ───────
-- The ViT peer of `resnet34in_*` (§2k). No renderer change: `nClasses`, `bs` and `replicas` are
-- all ordinary parameters, so this is three `#eval`s.
--
-- ⚠⚠ **THE SLUG IS LOAD-BEARING AND IT IS THE ONE TRAP THAT MATTERS HERE.** Forward artifacts
-- carry no variant in their path (`<slug>_fwd.mlir`), so rendering a 1000-class ViT forward under
-- the `vit` slug would silently OVERWRITE `verified_mlir/vit_fwd.mlir` — the 10-class Imagenette
-- forward that the 71.31% run, the prefix audit and every `fwd-tie vit` invocation depend on.
-- §2k hit exactly this on R34 and it is why `slug` exists there. Distinct paths, distinct entries.
--
-- Batch: **128 per device × 4 replicas = global 512**, which is `jax/MainVitImagenet.lean`'s
-- `vitTinyImagenetConfig.batchSize` — matching the reference's global batch is what keeps the two
-- runs a comparable pair rather than two experiments (the §2k argument, and R34's `momdp64` was
-- chosen the same way). The single-device `adam128` peer exists so `vit-dp-check` has a
-- same-per-device-batch reference to gate against; it is not scaffolding.
--
-- ⚠ Label smoothing at K = 1000: `alphaOverK` derives the cotangent constant from `nClasses`, so
-- the emitted `-α/K` must be **-0.000100**, not the K=10 literal `-0.010000`. That hardcoding was
-- a REAL BUG on the R34 ImageNet render (§2k — the first smoke reported loss ≈ 87 where 1000-class
-- CE at init must be ≈ ln(1000) = 6.9, and it was on the GRADIENT path). Fixed for ViT on
-- 2026-07-31; `TestVitInSmoke` re-checks the emitted constant rather than trusting that.
--
-- ⚠ Not matched to the reference yet, and deliberately: `vitTinyImagenetConfig` also carries
-- mixup + cutmix (soft labels — this render's cotangent is smoothed-CE over a ONE-HOT and cannot
-- express them), stochastic depth, EMA and grad clipping. The pipeline-level augs (RandAugment,
-- random erasing, repeated aug) DO come across for free via the shim. See the handoff §2p.
-- ⚠⚠ `wdStr := "0.05"` ON BOTH, AND IT IS A FIX, NOT A NEW KNOB. These baked `vitAdamConsts`'
-- 1e-4 default until 2026-08-02 — which is `vitTinyConfig`'s IMAGENETTE value. **No config
-- anywhere says "ImageNet ViT at wd 1e-4"**: `vitTinyImagenetConfig.weightDecay := 0.05`, the DeiT
-- value, so these two were training at 1/500th of their reference's decay. It is the
-- `RenderCifar8Sgd02` / EfficientNet-16× shape (§2a-quater) — a silently wrong hyperparameter that
-- compiles, runs and descends — and it was found while gating `wdExcludeNormBias`, i.e. by
-- building the NEXT feature, not by reading the configs.
--
-- ⚠ These two remain SHORT of the reference recipe deliberately (no `wx`, no clip, no EMA, no
-- stochastic depth, one-hot targets) and that is what the docstring above says. The decay was
-- never on that list — it was a typo, and 0.05 is now the reference's number with the mask and the
-- clip legitimately absent, which is an honest ablation point rather than an unlabelled mistake.
-- The variant that MATCHES the reference is `vitin_adamdp128x4wxclip` below.
#eval IO.FS.writeFile "verified_mlir/vitin_adam128_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vitin_adam128_train_step" "128.0" 1 128 1000
    (wdStr := "0.05"))
#eval IO.FS.writeFile "verified_mlir/vitin_adamdp128x4_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vitin_adamdp128x4_train_step" "128.0" 4 128 1000
    (wdStr := "0.05"))
#eval IO.FS.writeFile "verified_mlir/vitin_fwd.mlir"
  (Proofs.StableHLO.vitFwdRenderV "vitin_fwd" 256 1000)

-- ── ▶ v1.4: `wdExcludeNormBias` — timm/DeiT `no_weight_decay` (`recipe_gaps.md` v1.4) ──────────
-- `vitTinyImagenetConfig.wdExcludeNormBias := true`, so the ImageNet pair needs this render, not
-- the plain `adam128` one; `vitTinyConfig` does NOT set it, which is why the Imagenette artifacts
-- keep their bytes and this is a variant rather than a flipped default.
--
-- 126 of the 200 params take `%wdz` (a zero constant) instead of `%wd`: every 1-D param plus the
-- positional embedding. **Same arity, same types, same regions** — the only thing that moves is
-- 126 operand strings, so the driver, the checkpoint layout and every harness are untouched.
--
-- `vit_adamwx` is the Imagenette-shaped peer, and it is NOT scaffolding: `vit-wdx-tie` drives it
-- against `vit_adam` at bs32/K=10, where the two renders differ in EXACTLY the thing being gated
-- and the compile is seconds rather than a minute.
#eval IO.FS.writeFile "verified_mlir/vit_adamwx_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_adamwx_train_step" "32.0" 1 32 10 0.1
    (ema := false) (wdExclude := true))
-- ⚠ wd = **0.05**, not the file's 1e-4 default: `vitTinyImagenetConfig.weightDecay := 0.05` (the
-- DeiT value) where `vitTinyConfig` uses 1e-4. Both halves of the reference's decay recipe — the
-- MAGNITUDE and the MASK — have to be right for this render to be the pair's, and only the mask
-- was in scope when this variant was named. See `vitAdamConsts`.
#eval IO.FS.writeFile "verified_mlir/vitin_adam128wx_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vitin_adam128wx_train_step" "128.0" 1 128 1000 0.1
    (ema := false) (wdExclude := true) (wdStr := "0.05"))

-- ── ▶ v1.4b: GLOBAL-NORM GRADIENT CLIPPING (`planning/grad_clip.md`) ───────────────────────────
-- `vitTinyImagenetConfig.gradClipNorm := 1.0` — the DeiT default, and the reference's own comment
-- calls it *"the unlock for the 5e-4 LR"*. `vitTinyConfig` sets NOTHING, so (like `wx` and `ema`)
-- the Imagenette artifacts keep their bytes and this is a variant, not a flipped default.
--
-- ⚠ The Imagenette `clip` render is therefore a **GATE VEHICLE, NOT A MATCHED PAIR**: no Imagenette
-- reference run uses clipping, so its accuracy is not comparable to anything. It exists because
-- `clip-tie` drives it against `vit_adam` at bs32/K=10, where the two renders differ in EXACTLY the
-- thing being gated and the compile is seconds rather than a minute. Same role `vit_adamwx` plays.
--
-- The committed threshold is the reference's 1.0, and that is the CLIPPING regime: the reference
-- measured the pre-clip global grad norm at init at 14.28 (timm init) / 44.09 (Xavier), i.e. 10x+
-- over the threshold (`jax/MainVitImagenet.lean:89`), so a real run spends its early steps there.
--
-- ⚠⚠ THE BELOW-THRESHOLD RENDER IS **NOT COMMITTED**, AND THAT IS DELIBERATE — it is generated by
-- `scripts/perturb_clip.py hi`, the `perturb_wd_mask.py` / `misplace_drop_sites.py` pattern.
-- Two reasons, and the second was found the hard way:
--   1. an artifact baking a threshold NO CONFIG SETS is a silent-hyperparameter artifact — the
--      `RenderCifar8Sgd02` / EfficientNet-16× / ViT-500× shape (§2a-quater), and there is no
--      reference to say whether it is right;
--   2. ⚠ **A BOOL-DERIVED VARIANT NAME CANNOT DISTINGUISH TWO RENDERS THAT DIFFER ONLY IN A BAKED
--      CONSTANT.** Rendering a second threshold under `cnxAdamVariant`'s `clip : Bool` produced
--      `convnext_adamcliphi_train_step.mlir` declaring `@convnext_adamclip_train_step` — an entry
--      disagreeing with its own path, caught only because the paths were compared. ViT's explicit
--      `funcName` hid it here; ConvNeXt derives its name and did not. That is §0.4's
--      derived-vs-explicit finding meeting §2a-quater's silent-hyperparameter one.
#eval IO.FS.writeFile "verified_mlir/vit_adamclip_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_adamclip_train_step" "32.0" 1 32 10 0.1
    (ema := false) (wdExclude := false) (wdStr := "0.0001") (clip := true) (clipStr := "1.0"))
-- The ImageNet render — BOTH halves of the reference's recipe, `wx` ++ `clip`, because
-- `vitTinyImagenetConfig` sets `wdExcludeNormBias := true` AND `gradClipNorm := 1.0`. ⚠ wd = 0.05
-- for the same 500× reason `vitin_adam128wx` carries it.
#eval IO.FS.writeFile "verified_mlir/vitin_adam128wxclip_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vitin_adam128wxclip_train_step" "128.0" 1 128 1000 0.1
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0"))
-- ▶ **THE DATA-PARALLEL PEER, AND IT IS THE ONE AN IMAGENET RUN ACTUALLY LOADS.** 128 per device ×
-- 4 = the reference's global 512. Until this landed the DP ImageNet renders were THREE features
-- behind their single-device peers — wrong decay, no `wx`, no clip — so the artifact a real ViT
-- pair run would have loaded matched none of its reference's optimizer recipe, while the corrected
-- single-device ones sat beside it unused. ⚠ A feature is not done when its single-device artifact
-- renders (§0.4 finding 5, one axis over: there it was Imagenette-vs-ImageNet, here it is
-- single-vs-DP), and the way this was found was by LISTING what each artifact bakes.
--
-- ⚠ THE CLIP IS AFTER THE COLLECTIVE, and that is the whole DP question for this feature: the
-- reference clips the already-combined gradient, so clipping per replica would clip 200 PARTIAL
-- gradients — a different function that trains and descends. `vitAdamTrainStepFaithful` hoists both
-- the collective and the clip above the optimizer loop at `clip := true` and passes `preAvg`, so
-- this render emits **200 all_reduces, not 400**, all of them before the norm fold.
#eval IO.FS.writeFile "verified_mlir/vitin_adamdp128x4wxclip_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vitin_adamdp128x4wxclip_train_step" "128.0" 4 128 1000
    0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0"))

-- ── ▶ THE EMA VARIANT (`planning/ema.md`), selected by `LEAN_MLIR_VARIANT=ema` ─────────────────
-- ViT is the last of the three nets whose reference uses EMA (`vitTinyImagenetConfig.emaDecay :=
-- 0.99996`; ConvNeXt landed first, then EfficientNet's `emarms`), and it is the CHEAPEST of the
-- three: LayerNorm means there are no BN running buffers, so there is no `ema_bn` peer to carry —
-- the parameter shadow alone. `hasBn` is false for this net, so the driver's whole `ema_bn` arm is
-- skipped rather than special-cased.
--
-- Same graph plus one `adamMNextF` per parameter on the UPDATED weight — `d·ema + (1−d)·θ'`, which
-- is `Proofs.adamMNext` at `(β₁ := d, m := ema, g := θ')`. **No new op, no new `den`, no new
-- faithfulness theorem, no new VJP.** Fourth time enumerating a reference update against existing
-- ops AT THEIR OTHER READINGS has collapsed a scoped op family to zero (§2k heavy-ball,
-- recipe_gaps v1.2 RMSProp, ConvNeXt/EfficientNet EMA, here).
--
-- ⚠ THE BLOB GAINS A FOURTH REGION: `[θ|m|v|ema]`, 807 in / 805 out, and the scalar tail goes
-- 3 → 5 (`%emad`, `%oemad`). That is why it renders to its OWN slug — a 4-region graph fed a
-- 3-region blob is not a subtle numeric wrong answer, it is every parameter misaligned, and
-- `vit_adam_train_step.mlir` (which the 71.31% 80-epoch run, `vit-adam-tie` and `vit-dp-check` all
-- depend on) must stay exactly what it is. `trainAdamSched`'s checkpoint SIZE GUARD is the other
-- half of that: checkpoints carry no header, so a 3-region file read as 4 resumes silent garbage.
--
-- ⚠ `%emad`/`%oemad` are ARGS rather than constants because the reference's decay is time-varying,
-- `d = min(decay, (1+t)/(10+t))` — TF's warmup-corrected `ExponentialMovingAverage`. That
-- correction is REQUIRED at our scale, not optional: `ema.md` §2 has the reference's own
-- measurement of dropping it (a shadow still holding 12.8% of the random init at 3.1 τ, scoring
-- **0.00% top-1** while the live weights scored 70.48%), and an 80-epoch Imagenette run is 23,600
-- steps = 2.4 τ at decay 0.9999 — squarely inside that regime.
--
-- ⚠ Read this net's smoke as a DELTA, never an absolute. ViT's 80-epoch Imagenette result (71.31%)
-- is the weakest of the five by a wide margin — the expected outcome for a ViT with no pretraining
-- on 9,469 images, not a defect — so the gate is "the shadow tracks then exceeds the live
-- weights", which is a comparison within one run.
#eval IO.FS.writeFile "verified_mlir/vit_ema_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vit_ema_train_step" "32.0" 1 32 10 0.1 (ema := true))

-- ── The naming contract, pinned (§2d.1) ───────────────────────────────────────────────────────
-- `vitAdamVariant` is the single description of ViT's variant spelling, and these `#guard`s are
-- what tie it to the literal `funcName`s above — a rename on either side fails at `lake build`
-- rather than at run time as an "entry mismatch". (The artifact paths must stay literals: the
-- writer audit greps for the literal string `IO.FS.writeFile "verified_mlir/`.)
#guard Proofs.StableHLO.vitAdamVariant 32 1 == "adam"
#guard Proofs.StableHLO.vitAdamVariant 32 2 == "adamdp"
#guard Proofs.StableHLO.vitAdamVariant 64 1 == "adam64"
#guard Proofs.StableHLO.vitAdamVariant 64 2 == "adamdp64"
#guard Proofs.StableHLO.vitAdamVariant 128 1 == "adam128"
-- The two that break the "the number is the per-device batch" convention, encoded so the exception
-- cannot be quietly re-broken: a 4-replica render reusing `adamdp` would give one artifact path two
-- writers computing different graphs (§2a), and the failure mode is a silent clobber.
#guard Proofs.StableHLO.vitAdamVariant 32 4 == "adamdp32x4"
#guard Proofs.StableHLO.vitAdamVariant 128 4 == "adamdp128x4"
-- The EMA peer. A distinct slug from the AdamW one is the point: the EMA render carries a FOURTH
-- `[θ|m|v|ema]` region, so it and the AdamW render cannot share an artifact path, a checkpoint or a
-- driver invocation. ⚠ The marker LEADS — `trainAdamSched` keys its 4-region layout off
-- `variant.startsWith "ema"`, and on EfficientNet a *prefix* test on a two-axis name (`emarms`)
-- already misclassified a variant once and would have initialised RMSProp's mean-square to 0.
#guard Proofs.StableHLO.vitAdamVariant 32 1 true == "ema"
-- ▶ The `wx` (timm no_weight_decay) spellings. TRAILING, so it composes with every other axis
-- without displacing one — and every combination below is run through the driver's three
-- predicates in `tests/TestVariantPredicates.lean`, because with N markers the collisions are
-- between PAIRS and reading a name at a time cannot find them (§0's `sd`/`rmsdp` finding).
#guard Proofs.StableHLO.vitAdamVariant 32 1 false true == "adamwx"
#guard Proofs.StableHLO.vitAdamVariant 128 1 false true == "adam128wx"
#guard Proofs.StableHLO.vitAdamVariant 128 4 false true == "adamdp128x4wx"
#guard Proofs.StableHLO.vitAdamVariant 32 1 true true == "emawx"
-- ▶ the `clip` spellings, and the COMPOSED one is the point: the reference sets `wx` AND `clip`,
-- so `adamwxclip` is what ships and `adamclip` alone is the gate vehicle.
#guard Proofs.StableHLO.vitAdamVariant 32 1 false false true == "adamclip"
#guard Proofs.StableHLO.vitAdamVariant 32 1 false true true == "adamwxclip"
#guard Proofs.StableHLO.vitAdamVariant 128 1 false true true == "adam128wxclip"
#guard Proofs.StableHLO.vitAdamVariant 128 4 false true true == "adamdp128x4wxclip"
#guard Proofs.StableHLO.vitAdamVariant 32 1 true false true == "emaclip"

-- ── ▶ v1.2c: THE IMAGENET EMA PEER (`planning/recipe_gaps.md` v1.2c) ──────────────────────────
-- `vitTinyImagenetConfig.emaDecay := 0.99996` (the DeiT default), so this is the render an ImageNet
-- ViT pair actually needs — `vit_ema` is Imagenette-scale and, as `ViTAdamCommon` records, a GATE
-- VEHICLE rather than a matched pair (`vitTinyConfig` sets no EMA at all). Batch 128 × 4 replicas
-- = global 512, matching the reference, exactly as the `vitin_adam128` pair does.
#eval IO.FS.writeFile "verified_mlir/vitin_ema128_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vitin_ema128_train_step" "128.0" 1 128 1000 0.1
    (ema := true))
#eval IO.FS.writeFile "verified_mlir/vitin_emadp128x4_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithful "vitin_emadp128x4_train_step" "128.0" 4 128 1000 0.1
    (ema := true))
