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
-- ⚠⚠ **THE BATCH WAS A PRIVATE CONSTANT AND IS NOW A PARAMETER (2026-08-03).** It had to become
-- one to render ViT's SHIPPING data-parallel artifact: an ImageNet run loads the DP render, and
-- `vitin_adamdp128x4wxclip` carries no stochastic depth while `vitin_adamwxclipdrop` (batch 32,
-- single-device) sits beside it unused. The DP peer must be at batch **128** to match
-- `vitTinyImagenetConfig`, and 128 was not expressible while this was a `def`.
--
-- ⚠ It is threaded as a parameter NAMED `vbB` on purpose: all 134 uses in this file are function
-- BODIES, and naming the parameter after the constant leaves every one of them byte-identical. The
-- diff is signatures and call sites only, which is what makes the byte-identity gate below
-- meaningful — at `vbB := 32` every committed artifact must re-render unchanged.
--
-- ⚠ ConvNeXt has the same shape (`cBS`, a private constant, 104 uses) and handoff §5 calls making
-- it a parameter "the whole prerequisite" for the stronger split-identity gate. Same move, and it
-- is cheap: the bodies do not move.
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
private def vlnFwdB (vbB : Nat) (gName btName xin : String) : StateM Nat (String × String) := do
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
private def vBlockFwdB (vbB : Nat) (pfx xin : String) (drop : Option Nat := none) :
    StateM Nat (String × BSaves) := do
  -- `drop = some i` is the BLOCK index; the two mask ordinals are `vitSiteIdx i 0` (attention) and
  -- `vitSiteIdx i 1` (MLP). ⚠ They must be DIFFERENT inputs at the SAME keep — one mask per block
  -- halves the noise and no structural check sees it (`stochastic_depth.md` §6.3).
  let dpA := drop.map (fun i => dpName (vitSiteIdx i 0))
  let dpM := drop.map (fun i => dpName (vitSiteIdx i 1))
  let (c1, ln1) ← vlnFwdB vbB s!"%{pfx}g1" s!"%{pfx}bt1" xin
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
  -- ▶ SITE 1 of 2: the ATTENTION branch, between the out-dense and the skip add. The reference's
  -- `x = x + _drop_branch(mhsa(…), ka, keep_prob)` — the drop scales `mhsa`, never `x`.
  let (cdA, oD) ← match dpA with
    | some m => pretty vbB (.dropPathB (N := vbB) (n := vbTok*vbD) m (fun _ => 0 : Vec vbB)
                              (.operand o (zVb : Vec (vbB*(vbTok*vbD)))))
    | none   => pure ("", o)
  let (ch, hres) ← pretty vbB (.addVB (.operand xin (zVb : Vec (vbB*(vbTok*vbD))))
      (.operand oD (zVb : Vec (vbB*(vbTok*vbD)))))
  let (c2, ln2) ← vlnFwdB vbB s!"%{pfx}g2" s!"%{pfx}bt2" hres
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
  -- ▶ SITE 2 of 2: the MLP branch, between fc2 and the skip add.
  let (cdM, f2D) ← match dpM with
    | some m => pretty vbB (.dropPathB (N := vbB) (n := vbTok*vbD) m (fun _ => 0 : Vec vbB)
                              (.operand f2 (zVb : Vec (vbB*(vbTok*vbD)))))
    | none   => pure ("", f2)
  let (cr, bout) ← pretty vbB (.addVB (.operand hres (zVb : Vec (vbB*(vbTok*vbD))))
      (.operand f2D (zVb : Vec (vbB*(vbTok*vbD)))))
  pure (code ++ co ++ cdA ++ ch ++ c2 ++ cf1 ++ cg ++ cf2 ++ cdM ++ cr,
    { xin, ln1, q, k, v, qss, kss, vss, scs, sms, att := acc, hres, ln2, f1, g, bout })

set_option maxRecDepth 8000 in
/-- **The depth-12 ViT-Tiny forward at the batched index.** Node for node the same chain
    `vitFwd12` emits — patch embed (16×16/s16, 196 patches + CLS + pos) → 12 blocks → final
    vector-LN → CLS slice → dense head.

    ⚠ Every node is a `batchOp`/`*B` form, so `den` is a `batchMap`/`batchMapAux` at `N := vbB` and
    the batch is an index of the AST rather than a number only `pretty` knows. That is the entire
    content of the move; the emitted text is unchanged, which the tie checks. -/
def vitFwd12B (vbB : Nat) (nClasses : Nat) (sd : Bool := false) : StateM Nat (String × FwdSaves) := do
  let (ce, embed) ← pretty vbB (.batchOp (N := vbB)
      (.patchEmbed (ic := 3) (H := 224) (W := 224) (P := 16) (N := vbTk) (D := vbD)
        "%wConv" "%bConv" "%cls" "%pos"
        (zKb : Kernel4 vbD 3 16 16) (zVb : Vec vbD) (zVb : Vec vbD) (zMb : Mat vbTok vbD))
      (.operand "%x" (zVb : Vec (vbB*(3*224*224)))))
  let mut code := ce
  let mut cur := embed
  let mut blocks : Array BSaves := #[]
  for i in [0:vDEPTH] do
    let (cb, sv) ← vBlockFwdB vbB s!"b{i}_" cur (if sd then some i else none)
    code := code ++ cb; cur := sv.bout; blocks := blocks.push sv
  let (cf, fl) ← vlnFwdB vbB "%gF" "%btF" cur
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
def vitFwdRenderB (funcName : String := "vit_fwd_b") (nClasses : Nat := 10)
    -- ⚠ TRAILING, per §2m.
    (sd : Bool := false)
    -- ⚠ `vbB` LAST and defaulted, so every existing positional call site is untouched and the
    -- committed artifacts re-render byte-identically at 32. See the note on `vbB` above.
    (vbB : Nat := 32) : String :=
  let (body, sv) := (vitFwd12B vbB nClasses sd).run' 0
  let res := sv.logits
  let blkSigs := String.intercalate ", " ((List.range vDEPTH).map blkArgSig)
  let argSig := s!"%x: {ty [vbB, 3*224*224]}, %wConv: {ty [vbD,3,16,16]}, %bConv: {ty [vbD]}, " ++
    s!"%cls: {ty [vbD]}, %pos: {ty [vbTok,vbD]}, " ++ blkSigs ++
    s!", %gF: {ty [vbD]}, %btF: {ty [vbD]}, %Wc: {ty [vbD,nClasses]}, %bc: {ty [nClasses]}" ++
    -- ⚠ The mask inputs go LAST, after every parameter, matching the train step and the driver's
    -- blob layout. Mid-list they would capture an existing positional slot (§2m).
    vitDropSig vbB sd
  "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> {ty [vbB, nClasses]} " ++ "{\n" ++
  "    %one = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  "    %zero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  "    %sc = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  body ++ s!"    return {res} : {ty [vbB, nClasses]}\n" ++ "  }\n}\n"

end Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════════════════════════════
-- § The BACKWARD at `N := B` — node-by-node reverse of `vBlockFwdB`/`vitFwd12B`
-- ════════════════════════════════════════════════════════════════════════════════════════
--
-- ⚠ **AdamW only, deliberately**, exactly as `ConvNeXtRenderB` is. The per-example traversal serves
-- both renders off one `adam` flag; this one does not, because the SGD path bakes `lr` as a literal
-- where AdamW takes it as a runtime `%lr` operand, and an SGD render at the batched index is not
-- something any config asks for. Adding the flag later is cheap; adding an artifact nobody loads is
-- §2a-quater's silent-hyperparameter hazard.
--
-- ⚠⚠ THREE OPS CHANGE THEIR `N` FROM 1 TO THE BATCH AND EMIT THE SAME TEXT. `denseBiasGradB`,
-- `shiftB` and `divConstB` are written `(N := 1)` in the per-example render — the batch-unit
-- convention §2b removed everywhere else. Their emits read the width off `n` and ignore `N`, so the
-- text is identical either way; what changes is the DENOTATION, and for `denseBiasGradB` it changes
-- from "sum one thing" to "sum the batch", which is what a shared parameter's gradient must be.
-- That is the entire batched-index move in one line, and it is invisible to the byte tie.

namespace Proofs.StableHLO

/-- One **vector-LN backward** site, batched. Returns `(code, dxin, dγ, dβ)`. -/
private def vlnBackB (vbB : Nat) (gName btName xin dyOut : String) :
    StateM Nat (String × String × String × String) := do
  let (cb, nb) ← pretty vbB (.rowDenseBiasGradB (N := vbB) (R := vbTok) (c := vbD)
      (.operand dyOut (zVb : Vec (vbB*(vbTok*vbD)))))
  let (cg, ng) ← pretty vbB (.veclnGammaGradB (N := vbB) (R := vbTok) (D := vbD) xin vEPS 0
      (zVb : Vec (vbB*(vbTok*vbD))) (.operand dyOut (zVb : Vec (vbB*(vbTok*vbD)))))
  let (cs, da) ← pretty vbB (.batchOp (N := vbB)
      (.rowScale (m := vbTok) (n := vbD) gName (zVb : Vec vbD))
      (.operand dyOut (zVb : Vec (vbB*(vbTok*vbD)))))
  let (cn, dx) ← pretty vbB (.lnRowBackB (N := vbB) (m := vbTok) (n := vbD) "%one" xin vEPS 0 1
      (zVb : Vec (vbB*(vbTok*vbD))) (.operand da (zVb : Vec (vbB*(vbTok*vbD)))))
  pure (cb ++ cg ++ cs ++ cn, dx, ng, nb)

set_option maxRecDepth 8000 in
/-- One **transformer block backward**, batched. Returns `(code, dxin, the 16 gradient SSAs in
    `blkArgSig` order)`.

    ⚠ The per-head SDPA backward is where `matmulFB` earns the increment: `dsm = dpv·vsᵀ`,
    `dvs = smᵀ·dpv`, `dqs = dqk·ks`, `dkt = qsᵀ·dqk` — four matmuls, every one with BOTH operands
    per-example. A descriptor would pair example `k`'s cotangent with example 0's saved activation:
    it type-checks, it emits the identical `dot_general`, and it trains. -/
private def vBlockBackB (vbB : Nat) (pfx : String) (sv : BSaves) (dyOut : String) (drop : Option Nat := none) :
    StateM Nat (String × String × List String) := do
  let p := pfx
  let zTok : Vec (vbB*(vbTok*vbD)) := fun _ => 0
  let dpA := drop.map (fun i => dpName (vitSiteIdx i 0))
  let dpM := drop.map (fun i => dpName (vitSiteIdx i 1))
  -- ─ MLP sublayer back ─
  -- ▶ THE DROP'S BACKWARD IS THE SAME OP AT THE SAME MASK (`Proofs.dropPath_vjp_is_self`).
  -- ⚠⚠ IT FEEDS **THREE** CONSUMERS AND THE SKIP FAN-IN MUST NOT BE ONE OF THEM. `bout = hres + s⊙f2`
  -- ⇒ `df2 = s ⊙ dyOut` and `dhres ⊇ dyOut` RAW. All three of fc2's input-VJP, its weight gradient
  -- and its bias gradient read the cotangent at the fc2 OUTPUT, so all three take the dropped one.
  -- ConvNeXt's LayerScale-γ finding (one consumer) with the count multiplied: feeding any of them
  -- the undropped cotangent type-checks, trains and descends.
  let (cdM, dyD) ← match dpM with
    | some m => pretty vbB (.dropPathB (N := vbB) (n := vbTok*vbD) m (fun _ => 0 : Vec vbB)
                              (.operand dyOut zTok))
    | none   => pure ("", dyOut)
  let (c1, dg) ← pretty vbB (.batchOp (N := vbB)
      (.denseRowBack (rows := vbTok) (a := vbM) (c := vbD) s!"%{p}Wfc2" (zMb : Mat vbM vbD))
      (.operand dyD zTok))
  let (c2, nWfc2) ← pretty vbB (.rowDenseWeightGradB (N := vbB) (tk := vbTok) (a := vbM) (c := vbD)
      sv.g (zVb : Vec (vbB*(vbTok*vbM))) (.operand dyD zTok))
  let (c3, nbfc2) ← pretty vbB (.rowDenseBiasGradB (N := vbB) (R := vbTok) (c := vbD)
      (.operand dyD zTok))
  let (c4, df1) ← pretty vbB (.geluBackB sv.f1 (zVb : Vec (vbB*(vbTok*vbM)))
      (.operand dg (zVb : Vec (vbB*(vbTok*vbM)))))
  let (c5, dln2) ← pretty vbB (.batchOp (N := vbB)
      (.denseRowBack (rows := vbTok) (a := vbD) (c := vbM) s!"%{p}Wfc1" (zMb : Mat vbD vbM))
      (.operand df1 (zVb : Vec (vbB*(vbTok*vbM)))))
  let (c6, nWfc1) ← pretty vbB (.rowDenseWeightGradB (N := vbB) (tk := vbTok) (a := vbD) (c := vbM)
      sv.ln2 zTok (.operand df1 (zVb : Vec (vbB*(vbTok*vbM)))))
  let (c7, nbfc1) ← pretty vbB (.rowDenseBiasGradB (N := vbB) (R := vbTok) (c := vbM)
      (.operand df1 (zVb : Vec (vbB*(vbTok*vbM)))))
  let (c8, dhresLn2, ng2, nbt2) ← vlnBackB vbB s!"%{p}g2" s!"%{p}bt2" sv.hres dln2
  let (c9, dhres) ← pretty vbB (.addVB (.operand dyOut zTok) (.operand dhresLn2 zTok))
  -- ─ Attention sublayer back ─
  -- ▶ The ATTENTION branch's site, same shape: `hres = xin + s⊙o` ⇒ `do = s ⊙ dhres`, and the
  -- res₁ fan-in below keeps the RAW `dhres`.
  let (cdA, dhD) ← match dpA with
    | some m => pretty vbB (.dropPathB (N := vbB) (n := vbTok*vbD) m (fun _ => 0 : Vec vbB)
                              (.operand dhres zTok))
    | none   => pure ("", dhres)
  let (c10, dacc) ← pretty vbB (.batchOp (N := vbB)
      (.denseRowBack (rows := vbTok) (a := vbD) (c := vbD) s!"%{p}Wo" (zMb : Mat vbD vbD))
      (.operand dhD zTok))
  let (c11, nWo) ← pretty vbB (.rowDenseWeightGradB (N := vbB) (tk := vbTok) (a := vbD) (c := vbD)
      sv.att zTok (.operand dhD zTok))
  let (c12, nbo) ← pretty vbB (.rowDenseBiasGradB (N := vbB) (R := vbTok) (c := vbD)
      (.operand dhD zTok))
  let mut code := cdM ++ c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++ c8 ++ c9 ++ cdA ++ c10 ++ c11 ++ c12
  let mut dqAcc : String := ""; let mut dkAcc : String := ""; let mut dvAcc : String := ""
  let zHd : Vec (vbB*(vbTok*vbHd)) := fun _ => 0
  let zAtt : Vec (vbB*(vbTok*vbTok)) := fun _ => 0
  let zHdT : Vec (vbB*(vbHd*vbTok)) := fun _ => 0
  for hh in [0:vbH] do
    let h : Fin vbH := ⟨hh % vbH, Nat.mod_lt _ (by decide)⟩
    let (ca, dpv) ← pretty vbB (.batchOp (N := vbB)
        (.headSlice (N := vbTok) (heads := vbH) (d := vbHd) h)
        (.operand dacc (zVb : Vec (vbB*(vbTok*(vbH*vbHd))))))
    let (cb, vsT) ← pretty vbB (.batchOp (N := vbB) (.transpose (m := vbTok) (n := vbHd))
        (.operand (sv.vss[hh]!) zHd))
    let (cc, dsm) ← pretty vbB (.matmulFB (N := vbB) (m := vbTok) (k := vbHd) (n := vbTok)
        (.operand dpv zHd) (.operand vsT zHdT))
    let (cd, smT) ← pretty vbB (.batchOp (N := vbB) (.transpose (m := vbTok) (n := vbTok))
        (.operand (sv.sms[hh]!) zAtt))
    let (ce, dvs) ← pretty vbB (.matmulFB (N := vbB) (m := vbTok) (k := vbTok) (n := vbHd)
        (.operand smT zAtt) (.operand dpv zHd))
    let (cf, dsc) ← pretty vbB (.softmaxRowBackB (N := vbB) (m := vbTok) (n := vbTok)
        (sv.scs[hh]!) zAtt (.operand dsm zAtt))
    let (cg2, dqk) ← pretty vbB (.scaleB (N := vbB) (n := vbTok*vbTok) vSCALE 0
        (.operand dsc zAtt))
    let (ch, dqs) ← pretty vbB (.matmulFB (N := vbB) (m := vbTok) (k := vbTok) (n := vbHd)
        (.operand dqk zAtt) (.operand (sv.kss[hh]!) zHd))
    let (ci, qsT) ← pretty vbB (.batchOp (N := vbB) (.transpose (m := vbTok) (n := vbHd))
        (.operand (sv.qss[hh]!) zHd))
    let (cj, dkt) ← pretty vbB (.matmulFB (N := vbB) (m := vbHd) (k := vbTok) (n := vbTok)
        (.operand qsT zHdT) (.operand dqk zAtt))
    let (ck, dks) ← pretty vbB (.batchOp (N := vbB) (.transpose (m := vbHd) (n := vbTok))
        (.operand dkt zHdT))
    let hpad := fun (src : String) => pretty vbB (.batchOp (N := vbB)
        (.headPad (N := vbTok) (heads := vbH) (d := vbHd) h) (.operand src zHd))
    let (cl, dqH) ← hpad dqs
    let (cm, dkH) ← hpad dks
    let (cn, dvH) ← hpad dvs
    code := code ++ ca ++ cb ++ cc ++ cd ++ ce ++ cf ++ cg2 ++ ch ++ ci ++ cj ++ ck ++ cl ++ cm ++ cn
    if hh == 0 then
      dqAcc := dqH; dkAcc := dkH; dvAcc := dvH
    else
      let (cq, dqs2) ← pretty vbB (.addVB (.operand dqAcc zTok) (.operand dqH zTok))
      let (cr, dks2) ← pretty vbB (.addVB (.operand dkAcc zTok) (.operand dkH zTok))
      let (cs, dvs2) ← pretty vbB (.addVB (.operand dvAcc zTok) (.operand dvH zTok))
      code := code ++ cq ++ cr ++ cs; dqAcc := dqs2; dkAcc := dks2; dvAcc := dvs2
  -- Q/K/V dense backward
  let qkvBack := fun (w acc : String) => do
    let (c1', dln) ← pretty vbB (.batchOp (N := vbB)
        (.denseRowBack (rows := vbTok) (a := vbD) (c := vbD) w (zMb : Mat vbD vbD))
        (.operand acc zTok))
    let (c2', nW) ← pretty vbB (.rowDenseWeightGradB (N := vbB) (tk := vbTok) (a := vbD) (c := vbD)
        sv.ln1 zTok (.operand acc zTok))
    let (c3', nb') ← pretty vbB (.rowDenseBiasGradB (N := vbB) (R := vbTok) (c := vbD)
        (.operand acc zTok))
    pure (c1' ++ c2' ++ c3', dln, nW, nb')
  let (cQ, dln1q, nWq, nbq) ← qkvBack s!"%{p}Wq" dqAcc
  let (cK, dln1k, nWk, nbk) ← qkvBack s!"%{p}Wk" dkAcc
  let (cV, dln1v, nWv, nbv) ← qkvBack s!"%{p}Wv" dvAcc
  let (cs1, dln1a) ← pretty vbB (.addVB (.operand dln1q zTok) (.operand dln1k zTok))
  let (cs2, dln1) ← pretty vbB (.addVB (.operand dln1a zTok) (.operand dln1v zTok))
  let (cl1, dxinLn1, ng1, nbt1) ← vlnBackB vbB s!"%{p}g1" s!"%{p}bt1" sv.xin dln1
  let (cx, dxin) ← pretty vbB (.addVB (.operand dhres zTok) (.operand dxinLn1 zTok))
  let names := [ng1, nbt1, nWq, nbq, nWk, nbk, nWv, nbv, nWo, nbo, ng2, nbt2,
                nWfc1, nbfc1, nWfc2, nbfc2]
  pure (code ++ cQ ++ cK ++ cV ++ cs1 ++ cs2 ++ cl1 ++ cx, dxin, names)

set_option maxRecDepth 16000 in
/-- **The whole-net batched traversal** — forward + cotangent + all 200 parameter gradients, the
    batched peer of `vitBackAll bs nClasses lrStr true (some …)`. Returns
    `(code, gradients-in-func-arg-order, softmaxSSA)`, the same shape, so the AdamW tail in
    `ViTRender.lean` can consume either. -/
def vitBackAllB (vbB : Nat) (nClasses : Nat) (smooth : Option (String × String × String) := none)
    (sd : Bool := false) : StateM Nat (String × List String × String) := do
    let (fwd, sv) ← vitFwd12B vbB nClasses sd
    let zCls : Vec (vbB*nClasses) := fun _ => 0
    let (cSm, nSm) ← pretty vbB (.batchOp (N := vbB) (.softmaxDiv (n := nClasses))
        (.batchOp (N := vbB) (.expe (n := nClasses)) (.operand sv.logits zCls)))
    let (cD0, nD0) ← pretty vbB (.subB (.operand nSm zCls) (.operand "%onehot" zCls))
    let (cSmooth, nDy) ← match smooth with
      | none => pure ("", nD0)
      | some (aStr, negAK, bStr) => do
          let (c1, n1) ← pretty vbB (.scaleB (N := vbB) (n := nClasses) aStr 0
              (.operand "%onehot" zCls))
          let (c2, n2) ← pretty vbB (.addVB (.operand nD0 zCls) (.operand n1 zCls))
          -- ⚠ `N := vbB` where the per-example render writes `N := 1`. Same emitted text (the
          -- emitter reads `n`), different denotation — and here the denotation was already right,
          -- because both are POINTWISE. It is `denseBiasGradB` below where the change bites.
          let (c3, n3) ← pretty vbB (.shiftB (N := vbB) (n := nClasses) negAK 0 (.operand n2 zCls))
          let (c4, n4) ← pretty vbB (.divConstB (N := vbB) (n := nClasses) bStr 0
              (.operand n3 zCls))
          pure (c1 ++ c2 ++ c3 ++ c4, n4)
    let cDy := cSm ++ cD0 ++ cSmooth
    -- head
    let (cDc, dcls) ← pretty vbB (.batchOp (N := vbB)
        (.dotOut (m := vbD) (n := nClasses) "%Wc" (zMb : Mat vbD nClasses)) (.operand nDy zCls))
    let (cWc, nWc) ← pretty vbB (.weightGradB (N := vbB) (m := vbD) (n := nClasses) sv.clsTok
        (zVb : Vec (vbB*vbD)) (.operand nDy zCls))
    let (cbc, nbc) ← pretty vbB (.biasGradB (N := vbB) (n := nClasses) (.operand nDy zCls))
    let (cPad, dfln) ← pretty vbB (.batchOp (N := vbB) (.clsPad (N := vbTk) (D := vbD))
        (.operand dcls (zVb : Vec (vbB*vbD))))
    let (cFln, dflnIn, ngF, nbtF) ← vlnBackB vbB "%gF" "%btF" sv.flnIn dfln
    let mut code := fwd ++ cDy ++ cDc ++ cWc ++ cbc ++ cPad ++ cFln
    let mut dcur := dflnIn
    let mut blkNames : Array (List String) := #[]
    for j in [0:vDEPTH] do
      let i := vDEPTH - 1 - j
      let (cb, dx, names) ← vBlockBackB vbB s!"b{i}_" (sv.blocks[i]!) dcur (if sd then some i else none)
      code := code ++ cb; dcur := dx; blkNames := blkNames.push names
    -- patch-embed params
    let zTok : Vec (vbB*(vbTok*vbD)) := fun _ => 0
    let (cwC, nwConv) ← pretty vbB (.patchEmbedWeightGradB (N := vbB) (ic := 3) (H := 224) (W := 224)
        (P := 16) (tk := vbTk) (D := vbD) "%ximg" (zVb : Vec (vbB*(3*224*224)))
        (.operand dcur zTok))
    let (cbC, nbConv) ← pretty vbB (.patchEmbedBiasGradB (N := vbB) (tk := vbTk) (c := vbD)
        (.operand dcur zTok))
    let (cClSl, dclsRow) ← pretty vbB (.batchOp (N := vbB) (.clsSlice (N := vbTk) (D := vbD))
        (.operand dcur zTok))
    -- ⚠⚠ THE ONE LINE WHERE `N := 1 → N := vbB` CHANGES THE FUNCTION. The CLS token is ONE shared
    -- `[192]` parameter, so its gradient is the sum of every example's CLS-row cotangent. The
    -- per-example render writes `(N := 1)` and gets "sum one thing" — correct there, because
    -- `pretty B` was doing the batch lift outside the AST. Here the AST owns the batch, so the sum
    -- must be over it. Same emitted text either way (the emit reduces the `B` axis); the byte tie
    -- cannot see this and `den_rowDenseBiasGradB_at_one`'s argument is why.
    let (cCl, ncls) ← pretty vbB (.denseBiasGradB (N := vbB) (c := vbD)
        (.operand dclsRow (zVb : Vec (vbB*vbD))))
    let (cPo, npos) ← pretty vbB (.posEmbedGradB (N := vbB) (tk := vbTk) (D := vbD)
        (.operand dcur zTok))
    let blkOutOrdered := (List.range vDEPTH).flatMap (fun i => blkNames[vDEPTH - 1 - i]!)
    pure (code ++ cwC ++ cbC ++ cClSl ++ cCl ++ cPo,
          [nwConv, nbConv, ncls, npos] ++ blkOutOrdered ++ [ngF, nbtF, nWc, nbc], nSm)

end Proofs.StableHLO

namespace Proofs.StableHLO

set_option maxRecDepth 16000 in
/-- **The ViT-Tiny AdamW train step at the batched index.** ⚠ It is the SAME renderer the
    per-example path uses — `vitAdamTrainStepFaithful` with `traversal` pointed at `vitBackAllB` —
    not a copy.

    That is possible because the AdamW tail is entirely **parameter-space**: `adamMNextF`,
    `adamVNextF`, `adamWParamF`, `gradSumSqAccF` and `clipScaleF` are indexed by the parameter's own
    size and never see the batch. So "the AdamW tail at the batched index" is no work at all — the
    batch had already been factored out of it by the ops' own shapes, and the only thing that moves
    is which traversal produced the gradients. ConvNeXt's increment 7 found the same and it
    generalises: the optimizer is where the batch has already been summed away. -/
def vitAdamTrainStepFaithfulB (funcName : String := "vit_adam_train_step_b")
    (bStr : String := "32.0") (replicas : Nat := 1) (nClasses : Nat := 10)
    (alpha : Float := 0.1) (ema : Bool := false)
    (wdExclude : Bool := false) (wdStr : String := "0.0001")
    (clip : Bool := false) (clipStr : String := "1.0") (sd : Bool := false)
    -- ⚠ LAST and defaulted, for `vitFwdRenderB`'s reason.
    (vbB : Nat := 32) : String :=
  let alphaStr := fmt6 alpha
  let negAlphaKStr := "-" ++ alphaOverK nClasses alpha
  -- ⚠⚠ `sd` IS SPELLED ONCE AND REACHES BOTH HALVES FROM HERE — the traversal (which places the 24
  -- sites) and the wrapper (which declares the 24 inputs, the 24 pass-through outputs and the
  -- signature). Letting a caller set them independently is the shape of defect this thread keeps
  -- finding; here there is nothing to keep in step.
  vitAdamTrainStepFaithful funcName bStr replicas vbB nClasses alpha ema wdExclude wdStr clip clipStr
    (traversal := some (vitBackAllB vbB nClasses (some (alphaStr, negAlphaKStr, bStr)) sd))
    (sd := sd)

end Proofs.StableHLO

/-- The SD forward's banner. Its own, because these bytes ARE a different render and a banner
    claiming otherwise is §0.9 finding 3 in the artifact itself. -/
def vitDropFwdBanner : String :=
  "    // ── ViT-Tiny forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──\n"

-- ════════════════════════════════════════════════════════════════
-- § ▶ THE STOCHASTIC-DEPTH ARTIFACTS (`planning/stochastic_depth.md`, handoff §0.2 ▶3)
-- ════════════════════════════════════════════════════════════════
--
-- ⚠ These are the only artifacts this file writes, and they are all NEW. Unlike ConvNeXt, ViT's
-- drop-free batched chain is byte-identical to its committed artifacts, so the SWAP here would move
-- nothing and needs no numeric licence — but it is still not made, because there is nothing to gain
-- from it until something renders off the batched chain. These do.
--
-- ⚠ 24 sites, TWO per block, at 12 keeps: the reference splits one sub-key per residual branch
-- (`ka, km = split(drop_key)`) and gives both the same `keep_prob`. One mask per BLOCK instead
-- would halve the noise and pass every structural check (`stochastic_depth.md` §6.3).

#eval IO.FS.writeFile "verified_mlir/vit_adamdrop_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adamdrop_train_step" "32.0" 1 10 0.1
    (ema := false) (wdExclude := false) (wdStr := "0.0001") (clip := false) (clipStr := "1.0")
    (sd := true))

-- Its prefix partner. ⚠ The SD variant gets its OWN forward rather than reusing `vit_fwd`: letting
-- the SD trainer eval through the drop-free forward is what the reference literally does, but it
-- would leave the SD train step with no `forward ⊂ train-step` partner at all — i.e. SPEND one of
-- the two load-bearing structural gates in the repo rather than pay 24 dead multiplies at eval.
#eval IO.FS.writeFile "verified_mlir/vit_drop_fwd.mlir"
  (Proofs.StableHLO.vitFwdRenderB "vit_drop_fwd" 10 (sd := true))

-- ⚠ BOTH SCALES and the DP peer, per §0.4 finding 5 and §0.5: a feature is not done when its
-- Imagenette artifact renders, and an ImageNet run loads the DP render. `wx` ++ `clip` ++ `drop` at
-- wd 0.05 is `vitTinyImagenetConfig`'s optimizer-and-regulariser set entire.
#eval IO.FS.writeFile "verified_mlir/vitin_adamwxclipdrop_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adamwxclipdrop_train_step" "32.0" 1 1000 0.1
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true))
-- ▶▶ **THE SHIPPING DATA-PARALLEL RENDER — the artifact an ImageNet run actually loads.**
-- Found 2026-08-03 by listing what each artifact BAKES: `vitin_adamdp128x4wxclip` (201 all_reduce)
-- declares **zero** `%dp<n>` mask inputs, while `vitin_adamwxclipdrop` (24 inputs, 0 all_reduce,
-- batch 32) sat beside it unused. So a 4-replica ViT run trained with NO stochastic depth, silently
-- — `vitTinyImagenetConfig` sets `dropPath 0.1`.
--
-- ⚠⚠ §0.5's DEFECT ON A NEW AXIS, and the same check found it: list what the artifact bakes rather
-- than reading the recipe matrix, which said ✅ because the FEATURE existed at Imagenette scale and
-- single-device. A matrix records a capability; only the artifact records the state.
--
-- ⚠ **Batch 128, and that is why `vbB` had to stop being a constant.** 128 × 4 replicas = global
-- 512 = `vitTinyImagenetConfig.batchSize`, matching `vitin_adamdp128x4wxclip`'s geometry so the two
-- are comparable. At batch 32 the step count would be 4× the reference's, which §2d.2 measured as
-- the axis accuracy actually tracks — a render nobody should pair-run.
#eval IO.FS.writeFile "verified_mlir/vitin_adamdp128x4wxclipdrop_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adamdp128x4wxclipdrop_train_step" "128.0" 4 1000
    0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128))

#eval IO.FS.writeFile "verified_mlir/vitin_drop_fwd.mlir"
  (Proofs.StableHLO.vitFwdRenderB "vitin_drop_fwd" 1000 (sd := true))

-- The 2-replica peer, and the replica count is not a choice: `drop-shard-check`'s known answer is
-- exact only at TWO (f32 addition is commutative; above two the collective is a tree and
-- associativity does not hold).
#eval IO.FS.writeFile "verified_mlir/vit_adamdpdrop_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adamdpdrop_train_step" "32.0" 2 10 0.1
    (ema := false) (wdExclude := false) (wdStr := "0.0001") (clip := false) (clipStr := "1.0")
    (sd := true))

-- ⚠ ViT takes `funcName` EXPLICITLY where ConvNeXt derives it from the variant, so the entry name
-- and the artifact path are two writers for one fact here. These pin them against each other.
#guard Proofs.StableHLO.vitAdamVariant 32 1 false false false true == "adamdrop"
#guard Proofs.StableHLO.vitAdamVariant 32 2 false false false true == "adamdpdrop"
#guard Proofs.StableHLO.vitAdamVariant 32 1 false true true true == "adamwxclipdrop"
-- The marker must not LEAD: the driver keys its 4-region blob off `startsWith "ema"`.
#guard (Proofs.StableHLO.vitAdamVariant 32 1 true false false true).startsWith "ema"
