import LeanMlir.Proofs.Codegen.ViTRender

/-! # ViT-Tiny at the BATCHED index `N := B` — the forward (handoff §0.2 ▶3)

The ViT peer of `ConvNeXtRenderB` / `ResNet34RenderB` / `MobileNetV2RenderB`, and the reason it
exists is the same one: **stochastic depth's mask is per-EXAMPLE**, and in the per-example-indexed
render (`ViTRender.lean`) a node denotes ONE example — `pretty B` lifts it across the batch, so the
node cannot see `j`. ViT is the last net without that move, and `vitTinyImagenetConfig` sets
`dropPath 0.1` (24 sites, two per block).

⚠⚠ **ON ViT THE INDEX COLLISION IS IN THE SOURCE TEXT, NOT ONLY IN THE SEMANTICS.** The per-example
renderer names its TOKEN axis `N` — `denseRowF {N a c}`, `clsSliceF {N D}`, `headSliceF {N heads d}`
— and a batched constructor names the BATCH `N`. Both are `Nat`, both appear multiplied, and the
swap type-checks in either direction. Every batched call below therefore passes the token count as
an EXPLICIT named argument (`(N := 197)`, `(tk := 196)`) and never positionally; the batch is `vbB`
and appears only as `pretty`'s argument and as `(N := vbB)` on the `*B` constructors.

⚠ The sharpest instance is `clsSlice`, which CONTRACTS `(tk+1)*D → D`. A render reading the batch as
the token axis takes `(N+1)*D → D` — it keeps ONE example and drops the rest — and at `N = tk` it
type-checks *and agrees*. `den_batchOp_clsSlice_per_example` is what pins those apart.

**What this file is NOT (yet).** The forward only. The backward, the optimizer tail and the `#eval`
writers stay in `ViTRender.lean` until this chain is tied — §2b's order, which ConvNeXt followed and
which produced a byte tie before anything swapped. Nothing here writes an artifact, so
`verified_mlir/` is untouched by construction.

**The gate** (`lake build vit-fwd-b-tie`): this chain and the committed `verified_mlir/vit_fwd.mlir`
must emit **byte-identical** text. That is available *because* every batched form was built to emit
its per-example peer's text byte-for-byte and `tests/TestBatchedEmitTie.lean` pins all 47
individually — so the whole-net claim is the per-form claim composed, and when it fails that file
localises which form did it in one run.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

/-! ## The batched shapes

⚠ Restated, not re-derived — `ViTRender`'s own `vEPS`/`vSCALE`/`vDEPTH` are shared (they stopped
being `private` for this file), but the shape numbers are spelled here so that a drift between the
two renderers fails the BYTE tie loudly rather than propagating silently. That is the same choice
`ConvNeXtRenderB` made and for the same reason.

ViT-Tiny: patch 16×16/s16 over 224² ⇒ **196 patch tokens, 197 with CLS**; `D = 192 = 3 heads × 64`;
MLP 768; depth 12. -/
private def vbB : Nat := 32          -- THE BATCH. The only number in this file that is one.
private def vbTk : Nat := 196        -- patch tokens (the token axis is `vbTk + 1`)
private def vbTok : Nat := 197       -- tokens including CLS
private def vbD : Nat := 192         -- model dim
private def vbH : Nat := 3           -- heads
private def vbHd : Nat := 64         -- head dim (vbH * vbHd = vbD)
private def vbM : Nat := 768         -- MLP hidden

#guard vbTok == vbTk + 1
#guard vbH * vbHd == vbD

private def zVb {n : Nat} : Vec n := fun _ => 0
private def zMb {a b : Nat} : Mat a b := fun _ _ => 0
private def zKb {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0

/-! ## The sites, one per per-example site in `ViTRender.lean` -/

/-- One **vector-LN** site, batched: `lnRow(1,0) → rowScale γ → rowBias β` on the `[197,192]` token
    matrix. Three `batchOp`s where the per-example peer has three bare nodes.

    ⚠ `m := vbTok` is the TOKEN count PER EXAMPLE and `N := vbB` is the batch. Collapsing those two
    into one index is exactly the defect this file exists to remove. -/
private def vlnFwdB (gName btName xin : String) : StateM Nat (String × String) := do
  let (c1, a) ← pretty vbB (.batchOp (N := vbB)
      (.lnRow (m := vbTok) (n := vbD) "%one" "%zero" vEPS 0 1 0)
      (.operand xin (zVb : Vec (vbB*(vbTok*vbD)))))
  let (c2, b) ← pretty vbB (.batchOp (N := vbB)
      (.rowScale (m := vbTok) (n := vbD) gName (zVb : Vec vbD))
      (.operand a (zVb : Vec (vbB*(vbTok*vbD)))))
  let (c3, o) ← pretty vbB (.batchOp (N := vbB)
      (.rowBias (m := vbTok) (n := vbD) btName (zVb : Vec vbD))
      (.operand b (zVb : Vec (vbB*(vbTok*vbD)))))
  pure (c1 ++ c2 ++ c3, o)

set_option maxRecDepth 8000 in
/-- One **transformer block** forward, batched: LN1 → Q/K/V dense → per-head SDPA
    (slice → QKᵀ → scale → softmax → ·V → pad, summed) → out dense → +res → LN2 → fc1 → GELU →
    fc2 → +res.

    ⚠⚠ **THE TWO `matmulFB`s ARE THE POINT OF THE WHOLE INCREMENT.** `QKᵀ` and `·V` have BOTH
    operands per-example, where every other batched binary in the kit (`addVB`, `subB`) is
    pointwise-same-shape. `den_matmulFB_per_example` states what that means: example `k`'s output is
    `Qₖ·Kₖᵀ` — its own `Q` against its own `K`. A descriptor would hand every example operand 0's
    left factor, and a `den` reading the batched index as one big matrix would multiply the whole
    batch together; **both type-check and both emit the identical `dot_general`**. -/
private def vBlockFwdB (pfx xin : String) : StateM Nat (String × BSaves) := do
  let (c1, ln1) ← vlnFwdB s!"%{pfx}g1" s!"%{pfx}bt1" xin
  let qkv := fun (w b : String) => pretty vbB (.batchOp (N := vbB)
      (.denseRow (N := vbTok) (a := vbD) (c := vbD) w b (zMb : Mat vbD vbD) (zVb : Vec vbD))
      (.operand ln1 (zVb : Vec (vbB*(vbTok*vbD)))))
  let (cq, q) ← qkv s!"%{pfx}Wq" s!"%{pfx}bq"
  let (ck, k) ← qkv s!"%{pfx}Wk" s!"%{pfx}bk"
  let (cv, v) ← qkv s!"%{pfx}Wv" s!"%{pfx}bv"
  let mut code := c1 ++ cq ++ ck ++ cv
  let mut acc : String := ""
  let mut qss : Array String := #[]; let mut kss : Array String := #[]; let mut vss : Array String := #[]
  let mut scs : Array String := #[]; let mut sms : Array String := #[]
  for hh in [0:vbH] do
    -- ⚠ `Nat.mod_lt` rather than `omega`: `vbH` is a `private def`, so omega sees an opaque
    -- `Nat` and cannot discharge `hh % vbH < vbH` the way it does for the literal 3.
    let h : Fin vbH := ⟨hh % vbH, Nat.mod_lt _ (by decide)⟩
    let hslice := fun (src : String) => pretty vbB (.batchOp (N := vbB)
        (.headSlice (N := vbTok) (heads := vbH) (d := vbHd) h)
        (.operand src (zVb : Vec (vbB*(vbTok*(vbH*vbHd))))))
    let (cqs, qs) ← hslice q
    let (cks, ks) ← hslice k
    let (cvs, vs) ← hslice v
    let (ckt, kt) ← pretty vbB (.batchOp (N := vbB) (.transpose (m := vbTok) (n := vbHd))
        (.operand ks (zVb : Vec (vbB*(vbTok*vbHd)))))
    let (cmm, qk) ← pretty vbB (.matmulFB (N := vbB) (m := vbTok) (k := vbHd) (n := vbTok)
        (.operand qs (zVb : Vec (vbB*(vbTok*vbHd))))
        (.operand kt (zVb : Vec (vbB*(vbHd*vbTok)))))
    let (csc, sc) ← pretty vbB (.scaleB (N := vbB) (n := vbTok*vbTok) vSCALE 0
        (.operand qk (zVb : Vec (vbB*(vbTok*vbTok)))))
    let (csm, sm) ← pretty vbB (.batchOp (N := vbB) (.softmaxRow (m := vbTok) (n := vbTok))
        (.operand sc (zVb : Vec (vbB*(vbTok*vbTok)))))
    let (cpv, pv) ← pretty vbB (.matmulFB (N := vbB) (m := vbTok) (k := vbTok) (n := vbHd)
        (.operand sm (zVb : Vec (vbB*(vbTok*vbTok))))
        (.operand vs (zVb : Vec (vbB*(vbTok*vbHd)))))
    let (cpd, pd) ← pretty vbB (.batchOp (N := vbB)
        (.headPad (N := vbTok) (heads := vbH) (d := vbHd) h)
        (.operand pv (zVb : Vec (vbB*(vbTok*vbHd)))))
    code := code ++ cqs ++ cks ++ cvs ++ ckt ++ cmm ++ csc ++ csm ++ cpv ++ cpd
    qss := qss.push qs; kss := kss.push ks; vss := vss.push vs; scs := scs.push sc; sms := sms.push sm
    if hh == 0 then
      acc := pd
    else
      let (cad, s) ← pretty vbB (.addVB (.operand acc (zVb : Vec (vbB*(vbTok*vbD))))
          (.operand pd (zVb : Vec (vbB*(vbTok*vbD)))))
      code := code ++ cad; acc := s
  let (co, o) ← pretty vbB (.batchOp (N := vbB)
      (.denseRow (N := vbTok) (a := vbD) (c := vbD) s!"%{pfx}Wo" s!"%{pfx}bo"
        (zMb : Mat vbD vbD) (zVb : Vec vbD))
      (.operand acc (zVb : Vec (vbB*(vbTok*vbD)))))
  let (ch, hres) ← pretty vbB (.addVB (.operand xin (zVb : Vec (vbB*(vbTok*vbD))))
      (.operand o (zVb : Vec (vbB*(vbTok*vbD)))))
  let (c2, ln2) ← vlnFwdB s!"%{pfx}g2" s!"%{pfx}bt2" hres
  let (cf1, f1) ← pretty vbB (.batchOp (N := vbB)
      (.denseRow (N := vbTok) (a := vbD) (c := vbM) s!"%{pfx}Wfc1" s!"%{pfx}bfc1"
        (zMb : Mat vbD vbM) (zVb : Vec vbM))
      (.operand ln2 (zVb : Vec (vbB*(vbTok*vbD)))))
  let (cg, g) ← pretty vbB (.batchOp (N := vbB) (.gelu (n := vbTok*vbM))
      (.operand f1 (zVb : Vec (vbB*(vbTok*vbM)))))
  let (cf2, f2) ← pretty vbB (.batchOp (N := vbB)
      (.denseRow (N := vbTok) (a := vbM) (c := vbD) s!"%{pfx}Wfc2" s!"%{pfx}bfc2"
        (zMb : Mat vbM vbD) (zVb : Vec vbD))
      (.operand g (zVb : Vec (vbB*(vbTok*vbM)))))
  let (cr, bout) ← pretty vbB (.addVB (.operand hres (zVb : Vec (vbB*(vbTok*vbD))))
      (.operand f2 (zVb : Vec (vbB*(vbTok*vbD)))))
  pure (code ++ co ++ ch ++ c2 ++ cf1 ++ cg ++ cf2 ++ cr,
    { xin, ln1, q, k, v, qss, kss, vss, scs, sms, att := acc, hres, ln2, f1, g, bout })

set_option maxRecDepth 8000 in
/-- **The depth-12 ViT-Tiny forward at the batched index.** Node for node the same chain
    `vitFwd12` emits — patch embed (16×16/s16, 196 patches + CLS + pos) → 12 blocks → final
    vector-LN → CLS slice → dense head.

    ⚠ Every node is a `batchOp`/`*B` form, so `den` is a `batchMap`/`batchMapAux` at `N := vbB` and
    the batch is an index of the AST rather than a number only `pretty` knows. That is the entire
    content of the move; the emitted text is unchanged, which the tie checks. -/
def vitFwd12B (nClasses : Nat) : StateM Nat (String × FwdSaves) := do
  let (ce, embed) ← pretty vbB (.batchOp (N := vbB)
      (.patchEmbed (ic := 3) (H := 224) (W := 224) (P := 16) (N := vbTk) (D := vbD)
        "%wConv" "%bConv" "%cls" "%pos"
        (zKb : Kernel4 vbD 3 16 16) (zVb : Vec vbD) (zVb : Vec vbD) (zMb : Mat vbTok vbD))
      (.operand "%x" (zVb : Vec (vbB*(3*224*224)))))
  let mut code := ce
  let mut cur := embed
  let mut blocks : Array BSaves := #[]
  for i in [0:vDEPTH] do
    let (cb, sv) ← vBlockFwdB s!"b{i}_" cur
    code := code ++ cb; cur := sv.bout; blocks := blocks.push sv
  let (cf, fl) ← vlnFwdB "%gF" "%btF" cur
  -- ⚠ `(N := vbTk)` is the PATCH count (196), so the operand's token axis is 197 and the result is
  -- one `[192]` row per example. Passing the batch here instead type-checks and keeps example 0.
  let (cs, sl) ← pretty vbB (.batchOp (N := vbB) (.clsSlice (N := vbTk) (D := vbD))
      (.operand fl (zVb : Vec (vbB*(vbTok*vbD)))))
  let (cl, logits) ← pretty vbB (.batchOp (N := vbB)
      (.dense "%Wc" "%bc" (zMb : Mat vbD nClasses) (zVb : Vec nClasses))
      (.operand sl (zVb : Vec (vbB*vbD))))
  pure (code ++ cf ++ cs ++ cl, { embed, blocks, flnIn := cur, fln := fl, clsTok := sl, logits })

set_option maxRecDepth 8000 in
/-- **`@vit_fwd_b`** — the batched-index peer of `vitFwdRenderV`, same 200-parameter signature and
    same `%x`. Not written to `verified_mlir/`: it exists to be TIED against the committed
    per-example artifact, and an artifact nothing loads is a silent-hyperparameter hazard waiting to
    happen (§2a-quater). The writer lands with the swap, not before. -/
def vitFwdRenderB (funcName : String := "vit_fwd_b") (nClasses : Nat := 10) : String :=
  let (body, sv) := (vitFwd12B nClasses).run' 0
  let res := sv.logits
  let blkSigs := String.intercalate ", " ((List.range vDEPTH).map blkArgSig)
  let argSig := s!"%x: {ty [vbB, 3*224*224]}, %wConv: {ty [vbD,3,16,16]}, %bConv: {ty [vbD]}, " ++
    s!"%cls: {ty [vbD]}, %pos: {ty [vbTok,vbD]}, " ++ blkSigs ++
    s!", %gF: {ty [vbD]}, %btF: {ty [vbD]}, %Wc: {ty [vbD,nClasses]}, %bc: {ty [nClasses]}"
  "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> {ty [vbB, nClasses]} " ++ "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  body ++ s!"    return {res} : {ty [vbB, nClasses]}\n" ++ "  }\n}\n"

end Proofs.StableHLO
