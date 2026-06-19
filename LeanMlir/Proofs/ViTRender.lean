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
  gap : String          -- final-LN output (= CLS-slice input)
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
  pure (code ++ cf ++ cs ++ cl, { embed, blocks, flnIn := cur, gap := fl, logits })

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

end Proofs.StableHLO

-- Render the depth-12 forward to a scratch file for iree validation.
#eval IO.FS.writeFile "/tmp/vit_fwd_rendered.mlir" (Proofs.StableHLO.vitFwdRenderV "vit_fwd")
