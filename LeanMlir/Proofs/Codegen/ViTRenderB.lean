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

**What this file is, as of 4c leg 4 (2026-09-07): the sole writer of every ViT artifact but one.**
It began as the forward only — the backward, the optimizer tail and the `#eval` writers stayed in
`ViTRender.lean` until this chain was tied, §2b's order, which ConvNeXt followed and which produced
a byte tie before anything swapped. The tie held: all NINETEEN drop-free artifacts re-render
byte-identically off `vitBackAllB`, measured whole-net before a writer moved, so they now render
from here and `git diff verified_mlir/` after the move is empty.

⛔ **The one exception is `vit_train_step.mlir`**, the SGD-inline step, which stays in
`ViTRender.lean`: `vitBackAllB` has no fused-SGD arm (it emits the raw gradient only), and ViT's
T3 §1a tie — `ViTStepTie.lean`, all 200 parameters — is stated at exactly those bytes. Retiring it
before that tie has a batched peer is the ordering mistake `planning/archive/renderer_convergence.md`
leg 1 wrote down.

⭐⭐ **The bytes did not move and one denotation did.** See the CLS-token emission below: this
chain's `denseBiasGradB (N := vbB)` sums the batch inside `den` where the per-example one wrote
`(N := 1)` and let `pretty B` lift outside the AST. `Proofs.ViTPoCGB.clsGrad_denB` is the theorem
that becomes available, and `ViTFoldGB.lean` carries the other nine nodes with it.

**The gate** (`lake build vit-fwd-b-tie`): this chain and the committed `verified_mlir/vit_fwd.mlir`
must emit **byte-identical** text. That is available *because* every batched form was built to emit
its per-example peer's text byte-for-byte and [`tests/TestBatchedEmitTie.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestBatchedEmitTie.lean) pins all 47
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
-- (`VitDims` and the Ti/S instances live in `ViTRender.lean`: the parameter-signature
-- builders there need them too, and this file imports that one.)

private def zVb {n : Nat} : Vec n := fun _ => 0
private def zMb {a b : Nat} : Mat a b := fun _ _ => 0
private def zKb {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0

/-- The abstract rounding ViT's six bf16 ops carry. ⚠ `id` in the RENDER, for the reason
    `ConvNeXtRenderB.zrndB` is: `skel`/`pretty` never look at it (the emitted text is decided by the
    tag), and the accuracy statement is made where `rnd` is instantiated at bf16 round-to-nearest.
    A render that baked a concrete rounding here would claim the emitter knows about it. -/
private def vzrnd : ℝ → ℝ := fun r => r

/-! ## The sites, one per per-example site in `ViTRender.lean` -/

/-- One **vector-LN** site, batched: `lnRow(1,0) → rowScale γ → rowBias β` on the `[197,192]` token
    matrix. Three `batchOp`s where the per-example peer has three bare nodes.

    ⚠ `m := vbTok` is the TOKEN count PER EXAMPLE and `N := vbB` is the batch. Collapsing those two
    into one index is exactly the defect this file exists to remove. -/
private def vlnFwdB (V : VitDims) (vbB : Nat) (gName btName xin : String) :
    StateM Proofs.StableHLO.EmitS (String × String) := do
  let vbTok := V.tok
  let vbD := V.d
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
private def vBlockFwdB (V : VitDims) (vbB : Nat) (pfx xin : String) (drop : Option Nat := none)
    -- ⭐ bf16 TRAILING and defaulted, the `wx`/`clip`/`sd` idiom — so every existing call site
    -- elaborates unchanged and every committed artifact re-renders byte-identically (gate 1).
    -- ⚠ It reaches the SIX matmuls and the TWO SDPA products only. LN, GELU, softmax, the
    -- transposes, the head slices/pads, the residual adds and the whole AdamW tail stay f32 —
    -- the same carve-out every bf16 render in this repo makes.
    (bf16 : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × BSaves) := do
  let vbTok := V.tok
  let vbD := V.d
  let vbH := V.heads
  let vbHd := V.hd
  let vbM := V.m
  -- `drop = some i` is the BLOCK index; the two mask ordinals are `vitSiteIdx i 0` (attention) and
  -- `vitSiteIdx i 1` (MLP). ⚠ They must be DIFFERENT inputs at the SAME keep — one mask per block
  -- halves the noise and no structural check sees it (`stochastic_depth.md` §6.3).
  let dpA := drop.map (fun i => dpName (vitSiteIdx i 0))
  let dpM := drop.map (fun i => dpName (vitSiteIdx i 1))
  let (c1, ln1) ← vlnFwdB V vbB s!"%{pfx}g1" s!"%{pfx}bt1" xin
  let qkv := fun (w b : String) => pretty vbB (.batchOp (N := vbB)
      (if bf16 then
        .denseRowBf16 (N := vbTok) (a := vbD) (c := vbD) vzrnd w b (zMb : Mat vbD vbD) (zVb : Vec vbD)
       else
        .denseRow (N := vbTok) (a := vbD) (c := vbD) w b (zMb : Mat vbD vbD) (zVb : Vec vbD))
      (.operand ln1 (zVb : Vec (vbB*(vbTok*vbD)))))
  let (cq, q) ← qkv s!"%{pfx}Wq" s!"%{pfx}bq"
  let (ck, k) ← qkv s!"%{pfx}Wk" s!"%{pfx}bk"
  let (cv, v) ← qkv s!"%{pfx}Wv" s!"%{pfx}bv"
  let mut code := c1 ++ cq ++ ck ++ cv
  let mut acc : String := ""
  let mut qss : Array String := #[]; let mut kss : Array String := #[]; let mut vss : Array String := #[]
  let mut scs : Array String := #[]; let mut sms : Array String := #[]
  for hh in [0:vbH] do
    -- ⚠ `Nat.mod_lt` rather than `omega`, and the positivity now comes from the RECORD: `vbH` was
    -- a `private def` (omega saw an opaque `Nat`), and is now a field, so `by decide` cannot close
    -- `0 < V.heads` either. That is the whole reason `VitDims` carries `heads_pos`.
    let h : Fin vbH := ⟨hh % vbH, Nat.mod_lt _ V.heads_pos⟩
    let hslice := fun (src : String) => pretty vbB (.batchOp (N := vbB)
        (.headSlice (N := vbTok) (heads := vbH) (d := vbHd) h)
        (.operand src (zVb : Vec (vbB*(vbTok*(vbH*vbHd))))))
    let (cqs, qs) ← hslice q
    let (cks, ks) ← hslice k
    let (cvs, vs) ← hslice v
    let (ckt, kt) ← pretty vbB (.batchOp (N := vbB) (.transpose (m := vbTok) (n := vbHd))
        (.operand ks (zVb : Vec (vbB*(vbTok*vbHd)))))
    let (cmm, qk) ← pretty vbB (if bf16 then
        .matmulFBBf16 (N := vbB) (m := vbTok) (k := vbHd) (n := vbTok) vzrnd
          (.operand qs (zVb : Vec (vbB*(vbTok*vbHd))))
          (.operand kt (zVb : Vec (vbB*(vbHd*vbTok))))
      else
        .matmulFB (N := vbB) (m := vbTok) (k := vbHd) (n := vbTok)
          (.operand qs (zVb : Vec (vbB*(vbTok*vbHd))))
          (.operand kt (zVb : Vec (vbB*(vbHd*vbTok)))))
    let (csc, sc) ← pretty vbB (.scaleB (N := vbB) (n := vbTok*vbTok) vSCALE 0
        (.operand qk (zVb : Vec (vbB*(vbTok*vbTok)))))
    let (csm, sm) ← pretty vbB (.batchOp (N := vbB) (.softmaxRow (m := vbTok) (n := vbTok))
        (.operand sc (zVb : Vec (vbB*(vbTok*vbTok)))))
    let (cpv, pv) ← pretty vbB (if bf16 then
        .matmulFBBf16 (N := vbB) (m := vbTok) (k := vbTok) (n := vbHd) vzrnd
          (.operand sm (zVb : Vec (vbB*(vbTok*vbTok))))
          (.operand vs (zVb : Vec (vbB*(vbTok*vbHd))))
      else
        .matmulFB (N := vbB) (m := vbTok) (k := vbTok) (n := vbHd)
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
      (if bf16 then
        .denseRowBf16 (N := vbTok) (a := vbD) (c := vbD) vzrnd s!"%{pfx}Wo" s!"%{pfx}bo"
          (zMb : Mat vbD vbD) (zVb : Vec vbD)
       else
        .denseRow (N := vbTok) (a := vbD) (c := vbD) s!"%{pfx}Wo" s!"%{pfx}bo"
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
  let (c2, ln2) ← vlnFwdB V vbB s!"%{pfx}g2" s!"%{pfx}bt2" hres
  let (cf1, f1) ← pretty vbB (.batchOp (N := vbB)
      (if bf16 then
        .denseRowBf16 (N := vbTok) (a := vbD) (c := vbM) vzrnd s!"%{pfx}Wfc1" s!"%{pfx}bfc1"
          (zMb : Mat vbD vbM) (zVb : Vec vbM)
       else
        .denseRow (N := vbTok) (a := vbD) (c := vbM) s!"%{pfx}Wfc1" s!"%{pfx}bfc1"
          (zMb : Mat vbD vbM) (zVb : Vec vbM))
      (.operand ln2 (zVb : Vec (vbB*(vbTok*vbD)))))
  let (cg, g) ← pretty vbB (.batchOp (N := vbB) (.gelu (n := vbTok*vbM))
      (.operand f1 (zVb : Vec (vbB*(vbTok*vbM)))))
  let (cf2, f2) ← pretty vbB (.batchOp (N := vbB)
      (if bf16 then
        .denseRowBf16 (N := vbTok) (a := vbM) (c := vbD) vzrnd s!"%{pfx}Wfc2" s!"%{pfx}bfc2"
          (zMb : Mat vbM vbD) (zVb : Vec vbD)
       else
        .denseRow (N := vbTok) (a := vbM) (c := vbD) s!"%{pfx}Wfc2" s!"%{pfx}bfc2"
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
def vitFwd12B (V : VitDims) (vbB : Nat) (nClasses : Nat) (sd : Bool := false)
    -- ⭐ bf16 TRAILING, per the same rule. ⚠ The classifier head below stays f32 in every net —
    -- one `[192,1000]` dense against 12 blocks of six, and it is where the loss is read.
    (bf16 : Bool := false)
    -- ⭐⭐ **`bf16Conv` IS A SEPARATE AXIS FROM `bf16`, AND ON ViT IT MUST BE `false`** — measured,
    -- not assumed. See `vitBackAllB`'s note: the stem's WEIGHT GRADIENT has no usable bf16 cuDNN
    -- kernel at ViT's shape and costs more than every dot in the net gains. The name matches the
    -- JAX side's own per-net knob (`TrainConfig.bf16Conv`), which has always been separate from
    -- `bf16` for exactly this class of reason.
    (bf16Conv : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × FwdSaves) := do
  let vbTk := V.tk
  let vbTok := V.tok
  let vbD := V.d
  -- ⭐ **The 16×16/s16 patchify stem — the ONE `convolution` in this net**, and therefore the one
  -- op here that takes the conv emit shape (bf16-TYPED result + convert) rather than the dot one.
  let (ce, embed) ← pretty vbB (.batchOp (N := vbB)
      (if bf16 && bf16Conv then
        .patchEmbedBf16 (ic := 3) (H := 224) (W := 224) (P := 16) (N := vbTk) (D := vbD) vzrnd
          "%wConv" "%bConv" "%cls" "%pos"
          (zKb : Kernel4 vbD 3 16 16) (zVb : Vec vbD) (zVb : Vec vbD) (zMb : Mat vbTok vbD)
       else
        .patchEmbed (ic := 3) (H := 224) (W := 224) (P := 16) (N := vbTk) (D := vbD)
          "%wConv" "%bConv" "%cls" "%pos"
          (zKb : Kernel4 vbD 3 16 16) (zVb : Vec vbD) (zVb : Vec vbD) (zMb : Mat vbTok vbD))
      (.operand "%x" (zVb : Vec (vbB*(3*224*224)))))
  let mut code := ce
  let mut cur := embed
  let mut blocks : Array BSaves := #[]
  for i in [0:vDEPTH] do
    let (cb, sv) ← vBlockFwdB V vbB s!"b{i}_" cur (if sd then some i else none) bf16
    code := code ++ cb; cur := sv.bout; blocks := blocks.push sv
  let (cf, fl) ← vlnFwdB V vbB "%gF" "%btF" cur
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
    (vbB : Nat := 32) (V : VitDims := vitTiDims)
    -- ⭐ bf16, and NO bf16 forward artifact is written today — exactly ConvNeXt's choice (§16.5's
    -- peer). The parameter exists so this render can be tied against a bf16 train step if one ever
    -- needs a `forward ⊂ train-step` partner; writing an artifact nothing loads is the
    -- silent-hyperparameter hazard this file warns about two comments up.
    (bf16 : Bool := false) (bf16Conv : Bool := false) : String :=
  let vbTok := V.tok
  let vbD := V.d
  let (body, sv) := (vitFwd12B V vbB nClasses sd bf16 bf16Conv).run' (0, [])
  let res := sv.logits
  -- ⚠ `fun i => blkArgSig i V`, NOT `.map blkArgSig`: the bare form passes only `i` and lets
  -- `V` fall back to its Tiny default, which renders a ViT-S body under a ViT-Tiny block
  -- signature. It type-checks and the artifact is wrong. Caught by shape-checking the emit.
  let blkSigs := String.intercalate ", " ((List.range vDEPTH).map (fun i => blkArgSig i V))
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
private def vlnBackB (V : VitDims) (vbB : Nat) (gName _btName xin dyOut : String) :
    StateM Proofs.StableHLO.EmitS (String × String × String × String) := do
  let vbTok := V.tok
  let vbD := V.d
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
private def vBlockBackB (V : VitDims) (vbB : Nat) (pfx : String) (sv : BSaves) (dyOut : String)
    (drop : Option Nat := none)
    -- ⭐ bf16 TRAILING. ⚠ It reaches the six input-VJPs, the six WEIGHT gradients and the four SDPA
    -- backward matmuls. Every BIAS gradient beside them stays f32, in this net as in all six:
    -- `Σ dy` is a reduction, not a contraction, and there is nothing for a tensor core to do.
    -- ⚠⚠ And the backward is where the money is — §9.4 measured it at ~60 % of a conv step and the
    -- forward-only arm at 1.09×. A render that flipped only `vBlockFwdB` would look wired and buy
    -- almost nothing.
    (bf16 : Bool := false) : StateM Proofs.StableHLO.EmitS (String × String × List String) := do
  let vbTok := V.tok
  let vbD := V.d
  let vbH := V.heads
  let vbHd := V.hd
  let vbM := V.m
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
      (if bf16 then
        .denseRowBackBf16 (rows := vbTok) (a := vbM) (c := vbD) vzrnd s!"%{p}Wfc2" (zMb : Mat vbM vbD)
       else
        .denseRowBack (rows := vbTok) (a := vbM) (c := vbD) s!"%{p}Wfc2" (zMb : Mat vbM vbD))
      (.operand dyD zTok))
  let (c2, nWfc2) ← pretty vbB (if bf16 then
      .rowDenseWeightGradBBf16 (N := vbB) (tk := vbTok) (a := vbM) (c := vbD) vzrnd
        sv.g (zVb : Vec (vbB*(vbTok*vbM))) (.operand dyD zTok)
    else
      .rowDenseWeightGradB (N := vbB) (tk := vbTok) (a := vbM) (c := vbD)
        sv.g (zVb : Vec (vbB*(vbTok*vbM))) (.operand dyD zTok))
  let (c3, nbfc2) ← pretty vbB (.rowDenseBiasGradB (N := vbB) (R := vbTok) (c := vbD)
      (.operand dyD zTok))
  let (c4, df1) ← pretty vbB (.geluBackB sv.f1 (zVb : Vec (vbB*(vbTok*vbM)))
      (.operand dg (zVb : Vec (vbB*(vbTok*vbM)))))
  let (c5, dln2) ← pretty vbB (.batchOp (N := vbB)
      (if bf16 then
        .denseRowBackBf16 (rows := vbTok) (a := vbD) (c := vbM) vzrnd s!"%{p}Wfc1" (zMb : Mat vbD vbM)
       else
        .denseRowBack (rows := vbTok) (a := vbD) (c := vbM) s!"%{p}Wfc1" (zMb : Mat vbD vbM))
      (.operand df1 (zVb : Vec (vbB*(vbTok*vbM)))))
  let (c6, nWfc1) ← pretty vbB (if bf16 then
      .rowDenseWeightGradBBf16 (N := vbB) (tk := vbTok) (a := vbD) (c := vbM) vzrnd
        sv.ln2 zTok (.operand df1 (zVb : Vec (vbB*(vbTok*vbM))))
    else
      .rowDenseWeightGradB (N := vbB) (tk := vbTok) (a := vbD) (c := vbM)
        sv.ln2 zTok (.operand df1 (zVb : Vec (vbB*(vbTok*vbM)))))
  let (c7, nbfc1) ← pretty vbB (.rowDenseBiasGradB (N := vbB) (R := vbTok) (c := vbM)
      (.operand df1 (zVb : Vec (vbB*(vbTok*vbM)))))
  let (c8, dhresLn2, ng2, nbt2) ← vlnBackB V vbB s!"%{p}g2" s!"%{p}bt2" sv.hres dln2
  let (c9, dhres) ← pretty vbB (.addVB (.operand dyOut zTok) (.operand dhresLn2 zTok))
  -- ─ Attention sublayer back ─
  -- ▶ The ATTENTION branch's site, same shape: `hres = xin + s⊙o` ⇒ `do = s ⊙ dhres`, and the
  -- res₁ fan-in below keeps the RAW `dhres`.
  let (cdA, dhD) ← match dpA with
    | some m => pretty vbB (.dropPathB (N := vbB) (n := vbTok*vbD) m (fun _ => 0 : Vec vbB)
                              (.operand dhres zTok))
    | none   => pure ("", dhres)
  let (c10, dacc) ← pretty vbB (.batchOp (N := vbB)
      (if bf16 then
        .denseRowBackBf16 (rows := vbTok) (a := vbD) (c := vbD) vzrnd s!"%{p}Wo" (zMb : Mat vbD vbD)
       else
        .denseRowBack (rows := vbTok) (a := vbD) (c := vbD) s!"%{p}Wo" (zMb : Mat vbD vbD))
      (.operand dhD zTok))
  let (c11, nWo) ← pretty vbB (if bf16 then
      .rowDenseWeightGradBBf16 (N := vbB) (tk := vbTok) (a := vbD) (c := vbD) vzrnd
        sv.att zTok (.operand dhD zTok)
    else
      .rowDenseWeightGradB (N := vbB) (tk := vbTok) (a := vbD) (c := vbD)
        sv.att zTok (.operand dhD zTok))
  let (c12, nbo) ← pretty vbB (.rowDenseBiasGradB (N := vbB) (R := vbTok) (c := vbD)
      (.operand dhD zTok))
  let mut code := cdM ++ c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++ c8 ++ c9 ++ cdA ++ c10 ++ c11 ++ c12
  let mut dqAcc : String := ""; let mut dkAcc : String := ""; let mut dvAcc : String := ""
  let zHd : Vec (vbB*(vbTok*vbHd)) := fun _ => 0
  let zAtt : Vec (vbB*(vbTok*vbTok)) := fun _ => 0
  let zHdT : Vec (vbB*(vbHd*vbTok)) := fun _ => 0
  for hh in [0:vbH] do
    let h : Fin vbH := ⟨hh % vbH, Nat.mod_lt _ V.heads_pos⟩
    let (ca, dpv) ← pretty vbB (.batchOp (N := vbB)
        (.headSlice (N := vbTok) (heads := vbH) (d := vbHd) h)
        (.operand dacc (zVb : Vec (vbB*(vbTok*(vbH*vbHd))))))
    let (cb, vsT) ← pretty vbB (.batchOp (N := vbB) (.transpose (m := vbTok) (n := vbHd))
        (.operand (sv.vss[hh]!) zHd))
    let (cc, dsm) ← pretty vbB (if bf16 then
        .matmulFBBf16 (N := vbB) (m := vbTok) (k := vbHd) (n := vbTok) vzrnd
          (.operand dpv zHd) (.operand vsT zHdT)
      else
        .matmulFB (N := vbB) (m := vbTok) (k := vbHd) (n := vbTok)
          (.operand dpv zHd) (.operand vsT zHdT))
    let (cd, smT) ← pretty vbB (.batchOp (N := vbB) (.transpose (m := vbTok) (n := vbTok))
        (.operand (sv.sms[hh]!) zAtt))
    let (ce, dvs) ← pretty vbB (if bf16 then
        .matmulFBBf16 (N := vbB) (m := vbTok) (k := vbTok) (n := vbHd) vzrnd
          (.operand smT zAtt) (.operand dpv zHd)
      else
        .matmulFB (N := vbB) (m := vbTok) (k := vbTok) (n := vbHd)
          (.operand smT zAtt) (.operand dpv zHd))
    let (cf, dsc) ← pretty vbB (.softmaxRowBackB (N := vbB) (m := vbTok) (n := vbTok)
        (sv.scs[hh]!) zAtt (.operand dsm zAtt))
    let (cg2, dqk) ← pretty vbB (.scaleB (N := vbB) (n := vbTok*vbTok) vSCALE 0
        (.operand dsc zAtt))
    let (ch, dqs) ← pretty vbB (if bf16 then
        .matmulFBBf16 (N := vbB) (m := vbTok) (k := vbTok) (n := vbHd) vzrnd
          (.operand dqk zAtt) (.operand (sv.kss[hh]!) zHd)
      else
        .matmulFB (N := vbB) (m := vbTok) (k := vbTok) (n := vbHd)
          (.operand dqk zAtt) (.operand (sv.kss[hh]!) zHd))
    let (ci, qsT) ← pretty vbB (.batchOp (N := vbB) (.transpose (m := vbTok) (n := vbHd))
        (.operand (sv.qss[hh]!) zHd))
    let (cj, dkt) ← pretty vbB (if bf16 then
        .matmulFBBf16 (N := vbB) (m := vbHd) (k := vbTok) (n := vbTok) vzrnd
          (.operand qsT zHdT) (.operand dqk zAtt)
      else
        .matmulFB (N := vbB) (m := vbHd) (k := vbTok) (n := vbTok)
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
        (if bf16 then
          .denseRowBackBf16 (rows := vbTok) (a := vbD) (c := vbD) vzrnd w (zMb : Mat vbD vbD)
         else
          .denseRowBack (rows := vbTok) (a := vbD) (c := vbD) w (zMb : Mat vbD vbD))
        (.operand acc zTok))
    let (c2', nW) ← pretty vbB (if bf16 then
        .rowDenseWeightGradBBf16 (N := vbB) (tk := vbTok) (a := vbD) (c := vbD) vzrnd
          sv.ln1 zTok (.operand acc zTok)
      else
        .rowDenseWeightGradB (N := vbB) (tk := vbTok) (a := vbD) (c := vbD)
          sv.ln1 zTok (.operand acc zTok))
    let (c3', nb') ← pretty vbB (.rowDenseBiasGradB (N := vbB) (R := vbTok) (c := vbD)
        (.operand acc zTok))
    pure (c1' ++ c2' ++ c3', dln, nW, nb')
  let (cQ, dln1q, nWq, nbq) ← qkvBack s!"%{p}Wq" dqAcc
  let (cK, dln1k, nWk, nbk) ← qkvBack s!"%{p}Wk" dkAcc
  let (cV, dln1v, nWv, nbv) ← qkvBack s!"%{p}Wv" dvAcc
  let (cs1, dln1a) ← pretty vbB (.addVB (.operand dln1q zTok) (.operand dln1k zTok))
  let (cs2, dln1) ← pretty vbB (.addVB (.operand dln1a zTok) (.operand dln1v zTok))
  let (cl1, dxinLn1, ng1, nbt1) ← vlnBackB V vbB s!"%{p}g1" s!"%{p}bt1" sv.xin dln1
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
    (sd : Bool := false) (V : VitDims := vitTiDims)
    -- ⭐⭐ bf16, TRAILING and defaulted, so every existing render is byte-identical (gate 1).
    -- ⚠ The loss, the softmax, the label smoothing, the classifier head and its two gradients, the
    -- LN/GELU/softmax backwards, every bias gradient, the global-norm clip and the whole AdamW tail
    -- stay f32 — the carve-out every bf16 render in this repo makes. §14 measured that those
    -- carve-outs are NOT a fixed tax: they cost MobileNetV2 nothing and EfficientNet-B0 almost
    -- everything, so what they cost ViT is a question for the measurement, not for this comment.
    (bf16 : Bool := false)
    -- ⭐⭐⭐ **ViT'S TWO CONVOLUTIONS GET THEIR OWN FLAG AND IT DEFAULTS TO `false`. MEASURED.**
    -- Turning the stem and its weight gradient bf16 alongside the dots makes the whole step
    -- **0.52×** — nearly twice as SLOW as f32 — and an `nsys` profile says why in one line: the
    -- bf16 arm's `__cudnn$convBackwardFilter` lowers to
    -- `sm80_xmma_wgrad_implicit_gemm_indexed_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_*` at ~30 ms,
    -- where the f32 arm's same op lowers to `conv2d_grouped_direct_kernel<float>` at ~5.8 ms.
    --
    -- ▶ The shape is why. This wgrad is the transpose trick — `[3,B,224,224] × [192,B,209,209]`
    -- with the DILATED cotangent as the filter — so cuDNN sees a 209×209 window. It has a direct
    -- f32 kernel for that and no bf16 one, so bf16 falls back to implicit GEMM over a window 170×
    -- larger than any real kernel, plus an NCHW→NHWC layout transform.
    --
    -- ⚠⚠ **THIS IS §9.1's DEPTHWISE FINDING ON A NEW OP, AND IT IS THE SAME LESSON**: gate 2 proves
    -- the operands are bf16; it proves NOTHING about whether cuDNN picked a good kernel for the
    -- shape. §14.3 item 2 asked for exactly this check on EfficientNet-B0 and it had never been run
    -- on any net. ViT is where it finally fired.
    (bf16Conv : Bool := false)
    -- ⚠ And the stem's FORWARD is split from its WEIGHT GRADIENT, because the two do not behave the
    -- same and lumping them would have hidden which one costs. See the probe numbers in
    -- `planning/archive/bf16_renderer.md` §18.
    (bf16ConvW : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × List String × String) := do
    let vbTk := V.tk
    let vbTok := V.tok
    let vbD := V.d
    let (fwd, sv) ← vitFwd12B V vbB nClasses sd bf16 bf16Conv
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
    let (cFln, dflnIn, ngF, nbtF) ← vlnBackB V vbB "%gF" "%btF" sv.flnIn dfln
    let mut code := fwd ++ cDy ++ cDc ++ cWc ++ cbc ++ cPad ++ cFln
    let mut dcur := dflnIn
    let mut blkNames : Array (List String) := #[]
    for j in [0:vDEPTH] do
      let i := vDEPTH - 1 - j
      let (cb, dx, names) ← vBlockBackB V vbB s!"b{i}_" (sv.blocks[i]!) dcur
        (if sd then some i else none) bf16
      code := code ++ cb; dcur := dx; blkNames := blkNames.push names
    -- patch-embed params
    let zTok : Vec (vbB*(vbTok*vbD)) := fun _ => 0
    -- ⭐ **The stem weight grad — ViT's second and last `convolution`.** ⚠ There is no
    -- `patchEmbedBack` in this traversal and therefore no bf16 twin of one: the stem's input is
    -- `%x`, so it has no input gradient. ConvNeXt's `convStride4` exactly (§16.5), and the reason
    -- this net needs six ops rather than seven.
    let (cwC, nwConv) ← pretty vbB (if bf16 && bf16ConvW then
        .patchEmbedWeightGradBBf16 (N := vbB) (ic := 3) (H := 224) (W := 224)
          (P := 16) (tk := vbTk) (D := vbD) vzrnd "%ximg" (zVb : Vec (vbB*(3*224*224)))
          (.operand dcur zTok)
      else
        .patchEmbedWeightGradB (N := vbB) (ic := 3) (H := 224) (W := 224)
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
    (vbB : Nat := 32) (V : VitDims := vitTiDims)
    -- ⭐⭐ bf16, TRAILING and defaulted.
    -- ⚠⚠ **ViT's ENTRY-NAME ROUTE IS THE THIRD ONE, NOT THE FIRST TWO.** `cnxAdamVariant` DERIVES
    -- ConvNeXt's entry name from its variant, so a flag that misses that call writes a mismatched
    -- artifact; `vitAdamTrainStepFaithful` takes `funcName` EXPLICITLY, so here the risk is route
    -- (c) — the artifact PATH and the `#eval`'s `funcName` disagreeing. The `#guard`s under the
    -- bf16 `#eval` pin the two spellings against each other, which is the check that fits this
    -- route. `planning/archive/bf16_renderer.md` §15.2 enumerates all three.
    (bf16 : Bool := false)
    -- ⭐ `bf16Conv`, defaulted `false`. ⚠ It adds NO variant marker: it is not a recipe choice, it
    -- is a measured statement that this net's two convolutions have no usable bf16 kernel, and a
    -- marker would invite someone to flip it. `vitBackAllB` carries the measurement.
    (bf16Conv : Bool := false) (bf16ConvW : Bool := false) : String :=
  let alphaStr := fmt6 alpha
  let negAlphaKStr := "-" ++ alphaOverK nClasses alpha
  -- ⚠⚠ `sd` IS SPELLED ONCE AND REACHES BOTH HALVES FROM HERE — the traversal (which places the 24
  -- sites) and the wrapper (which declares the 24 inputs, the 24 pass-through outputs and the
  -- signature). Letting a caller set them independently is the shape of defect this thread keeps
  -- finding; here there is nothing to keep in step.
  vitAdamTrainStepFaithful funcName bStr replicas vbB nClasses alpha ema wdExclude wdStr clip clipStr
    -- ⚠⚠ AND HERE. `bf16` reaching this traversal is what makes the graph bf16; nothing else in
    -- this function's body needs it, because the AdamW tail is parameter-space and stays f32.
    (traversal := some (vitBackAllB vbB nClasses (some (alphaStr, negAlphaKStr, bStr)) sd V bf16 bf16Conv bf16ConvW))
    (V := V)
    (sd := sd)

end Proofs.StableHLO

/-- The SD forward's banner. Its own, because these bytes ARE a different render and a banner
    claiming otherwise is §0.9 finding 3 in the artifact itself. -/
def vitDropFwdBanner : String :=
  "    // ── ViT-Tiny forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──\n"

-- ════════════════════════════════════════════════════════════════
-- § ▶ THE STOCHASTIC-DEPTH ARTIFACTS (`planning/archive/stochastic_depth.md`, handoff §0.2 ▶3)
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

-- ════════════════════════════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE bf16 ARM — `planning/archive/bf16_renderer.md` §17, and read §17.3 before quoting a number
-- ════════════════════════════════════════════════════════════════════════════════════════
--
-- ⚠⚠ **THIS RENDER WAS MEASURED BEFORE IT WAS BUILT AND THE MEASUREMENT SAID 1.03×.** §17.3 timed
-- ViT-Tiny's own 387 matmuls three ways at B = 32: f32 26.9 ms, bf16 in THIS emit shape 26.2 ms
-- (1.03×), bf16 with activations staying bf16 BETWEEN ops 15.7 ms (1.71×). ViT's matmuls are skinny
-- (contracting dim 192, or 768 at the MLP) and bandwidth-bound on the activations, so the
-- f32→bf16→f32 round trip at every op costs about what the tensor cores save. That is NOT true of
-- the convnets — §16.3 measured ConvNeXt keeping 64 % of its saving through the same boundary,
-- because a convolution reuses each loaded input across many output positions.
--
-- ▶ So this arm exists to CONFIRM the predicted 1.03× on a real artifact and to give the successor
-- project (`planning/archive/bf16_dtype_ir.md`) a wired net to flip. It is not a shipping recipe, and the
-- honest reading of its ms/step is "the six ops are correct", not "ViT is faster now".
--
-- ⚠ **SINGLE-DEVICE ONLY, deliberately** — §13.2: a 1-GPU pair is what isolates the RENDERER, and
-- a 4-replica number is a system result carrying the shim feed, the f32 all-reduce and (§16.2) the
-- parameter round trip. No bf16 DP peer is rendered: ViT's shipping DP artifact is the 128×4 one,
-- and at 1.03× there is nothing to ship. An artifact nothing loads is the silent-hyperparameter
-- hazard this file warns about 40 lines up, and rendering one ahead of a result would be it.
--
-- ⚠⚠ **THE ENTRY NAME.** ViT takes `funcName` explicitly rather than deriving it from
-- `vitAdamVariant`, so the defect route here is (c) — the artifact PATH and this string
-- disagreeing — not (a)/(b), which are ConvNeXt's and EfficientNet's. The `#guard`s below pin the
-- two spellings against each other and against the driver's three substring predicates.
#eval IO.FS.writeFile "verified_mlir/vitin_adamwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adamwxclipdropbf16_train_step" "32.0" 1 1000 0.1
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (bf16 := true))

-- ⚠ **HOW THE `bf16Conv`/`bf16ConvW` DEFAULTS WERE ESTABLISHED, so the claim can be re-run rather
-- than believed.** Three probe renders, identical to the `#eval` above except for the two flags,
-- written outside `verified_mlir/` and timed with `scripts/bf16_device_step.py` against the f32
-- peer (bare device, B = 32, median of 25). They are NOT committed: a probe under `verified_mlir/`
-- is what gate 1 watches and what the driver loads, and one living there is the
-- silent-hyperparameter hazard this file warns about 60 lines up.
--
--     (bf16Conv := false) (bf16ConvW := false)   24.39 ms   1.23x   ← what is rendered above
--     (bf16Conv := true)  (bf16ConvW := false)   24.64 ms   1.21x   ← stem forward: ~free, no gain
--     (bf16Conv := false) (bf16ConvW := true)    57.22 ms   0.52x   ← THE WEIGHT GRADIENT ALONE
--     (bf16Conv := true)  (bf16ConvW := true)    57.29 ms   0.52x
--                              f32 control       29.89 ms
--
-- ▶ **One op, one site, and it costs more than every dot in the net gains.** The stem forward is
-- within noise either way; the weight gradient is +32.8 ms on a 29.89 ms step.

-- ── The entry-name guards, route (c). ────────────────────────────────────────────────────────
-- ⚠ The bf16 slug is the f32 one with `bf16` APPENDED, so every committed spelling is untouched.
#guard "vitin_adamwxclipdropbf16_train_step" ==
  "vitin_" ++ Proofs.StableHLO.vitAdamVariant 32 1 false true true true ++ "bf16" ++ "_train_step"
-- ⚠ And the DRIVER's three substring predicates, which size the checkpoint blob off this string.
-- `cdOn` is a test for `"do"`, not for `"cd"` — a false positive silently adds a blob region.
#guard !(Proofs.StableHLO.vitAdamVariant 32 1 false true true true ++ "bf16").startsWith "ema"
#guard ((Proofs.StableHLO.vitAdamVariant 32 1 false true true true ++ "bf16").splitOn "do").length == 1
#guard ((Proofs.StableHLO.vitAdamVariant 32 1 false true true true ++ "bf16").splitOn "acc").length == 1
-- ⚠ `drop` must survive the append: the driver reads it to declare the 24 mask inputs, and the
-- bf16 arm has the same 24 sites as its f32 peer.
#guard ((Proofs.StableHLO.vitAdamVariant 32 1 false true true true ++ "bf16").splitOn "drop").length == 2
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

-- ⭐ **The 4-replica bf16 peer of the shipping DP render**, at the same 128×4 = global 512 geometry.
-- ⚠⚠ A 4-replica ms/step is a SYSTEM result (§13.2) — shim feed and f32 all-reduce included. ViT's
-- RENDERER number is the single-device bare-device **1.46×** (§20), and that is what the emit is
-- worth. This artifact exists so the net can be scheduled and costed at the geometry an ImageNet
-- run actually uses, not to restate the speedup.
-- ⚠ `bf16Conv`/`bf16ConvW` stay at their measured defaults (both false) — §19.1's stem weight
-- gradient is 0.19× its f32 peer and the replica axis does not change that.
#eval IO.FS.writeFile "verified_mlir/vitin_adamdp128x4wxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adamdp128x4wxclipdropbf16_train_step" "128.0" 4
    1000 0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128) (bf16 := true))
#guard "vitin_adamdp128x4wxclipdropbf16_train_step" ==
  "vitin_" ++ Proofs.StableHLO.vitAdamVariant 128 4 false true true true ++ "bf16" ++ "_train_step"

-- ── ⭐ THE FULL-RECIPE PAIR RENDER: EMA **AND** wx + clip + drop, bf16 ────────────────────────
-- ⚠⚠ **THE MISSING COMBINATION, AND IT WAS MISSING RATHER THAN UNBUILDABLE.** EMA was rendered
-- for this net back at v1.2c (`vitin_emadp128x4`, `ViTRender.lean`), at this exact 128×4 geometry
-- and for this exact reason — but with ONLY `(ema := true)`, so `wdExclude`, `clip` and `sd` all
-- fell to their `false` defaults. Picking that artifact to get EMA therefore silently gives up
-- **gradient clipping**, which blueprint §9.6 measures as load-bearing for this net: without it
-- ViT-Ti collapses to chance the moment warmup ramps past ~1.6e-4 and never recovers. Nobody
-- would trade that for EMA knowingly, so the pair had no runnable artifact at all.
--
-- ⚠ It is one call, not new machinery: `vitAdamTrainStepFaithfulB` already takes all four flags,
-- and `vitAdamVariant` already composes the name from all four (`ema` REPLACES the `adam` prefix,
-- then `wx` ++ `clip` ++ `drop` append). The driver predicate already routes it —
-- `VerifiedVariant.emaOn` is `startsWith "ema"`, and `emadp128x4wxclipdrop` does.
--
-- ⚠⚠ **EMA + dropPath had never been rendered together before this line.** EMA adds a FOURTH blob
-- region and takes the scalar tail 3 → 5; dropPath adds 24 mask operands (12 blocks × 2 branches).
-- Each is exercised alone. `tests/TestVitEmaDropRender.lean` is where that composition gets gated —
-- a wrongly-packed region TRAINS and REPORTS A LOSS, which is `planning/archive/ema.md`'s own defect.
--
-- ⭐ bf16 because the phase-2 reference run this pairs against is bf16 (§9.6): an f32 verified run
-- would differ from it in PRECISION as well as in lowerer, and the comparison is about the lowerer.
-- Same argument as the R50-2018 pair.
#eval IO.FS.writeFile "verified_mlir/vitin_emadp128x4wxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_emadp128x4wxclipdropbf16_train_step" "128.0" 4
    1000 0.1 (ema := true) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128) (bf16 := true))
#guard "vitin_emadp128x4wxclipdropbf16_train_step" ==
  "vitin_" ++ Proofs.StableHLO.vitAdamVariant 128 4 true true true true ++ "bf16" ++ "_train_step"
#guard ((Proofs.StableHLO.vitAdamVariant 128 4 false true true true ++ "bf16").splitOn "do").length == 1
#guard ((Proofs.StableHLO.vitAdamVariant 128 4 false true true true ++ "bf16").splitOn "drop").length == 2

-- The **2-GPU** peer: 256 per replica × 2 = the same global 512 the 128×4 render above trains at,
-- so the recipe, the steps/epoch and the LR are unchanged and the two wall-clocks are comparable.
-- ⚠ ViT is the one net whose slug spells the replica count (`128x4` → `256x2`), so BOTH numbers
-- move here and the name is passed explicitly rather than derived — which also means the entry
-- name, the artifact path and `LEAN_MLIR_VARIANT` are three copies of one string. `vbB` is the
-- fourth and is easiest to miss: it must track the per-replica batch or the render disagrees with
-- its own geometry.
#eval IO.FS.writeFile "verified_mlir/vitin_adamdp256x2wxclipdrop_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adamdp256x2wxclipdrop_train_step" "256.0" 2 1000
    0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 256))

-- ⚠ **256 per replica does not RUN on gfx1100**: MIOpen refuses the patch-embed convolution at
-- that batch — `Failed to enqueue convolution on stream: miopenStatusUnknownError`, a returned
-- status rather than a crash, on the first step. 128 per replica is the same net at half the
-- per-card batch (global 256 instead of 512), which is what this render is for: it keeps ViT
-- runnable on two cards while the 256 variant stays as the recipe-matched artifact for boxes whose
-- MIOpen accepts it. The two differ ONLY in `B` (and `vbB`, which must track it).
#eval IO.FS.writeFile "verified_mlir/vitin_adamdp128x2wxclipdrop_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adamdp128x2wxclipdrop_train_step" "128.0" 2 1000
    0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128))

#eval IO.FS.writeFile "verified_mlir/vitin_drop_fwd.mlir"
  (Proofs.StableHLO.vitFwdRenderB "vitin_drop_fwd" 1000 (sd := true))

-- ════════════════════════════════════════════════════════════════════════════════════════
-- § ViT-**Small** on ImageNet-1k — the same renderer at `vitSDims`
-- ════════════════════════════════════════════════════════════════════════════════════════
--
-- ⭐ **Nothing above this line changed to add these two artifacts except the `V` parameter.** S is
-- Tiny widened: `D = 384 = 6 × 64` instead of `192 = 3 × 64`, MLP 1536 instead of 768. Same depth
-- (12), same patch grid (196 + CLS), same block chain, same backward. That is why the proof side
-- needs nothing: `vitForwardKV_has_vjp` is already `∀ heads d_head mlpDim k` and it is a GLOBAL
-- `HasVJP`, since GELU/softmax/LayerNorm carry no kink.
--
-- ⚠ Only the DP train step and the forward are rendered, and that is the complete set for this
-- net: ViT has no BatchNorm, so there is no running-stat eval forward to pair with them. The
-- 10-class `vit_*` artifacts stay ViT-Tiny — they come from the per-example renderer, which is
-- still pinned at Tiny by ~154 dim literals (`ViTRender.lean`). Widening THAT is a separate job.
#eval IO.FS.writeFile "verified_mlir/vitsin_adamdp128x4wxclipdrop_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitsin_adamdp128x4wxclipdrop_train_step" "128.0" 4 1000
    0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128) (V := Proofs.StableHLO.vitSDims))

-- ⭐ **S's bf16 peer** (`verified_side_quest_counterparts.md` §4c). ViT-Tiny carried one and S did
-- not, the same accident of ordering ConvNeXt-S had. One `(bf16 := true)`, no new operator: S is
-- Tiny WIDENED, so every bf16 op it needs is one Tiny already instantiates, at a wider shape.
-- ⚠⚠ `bf16Conv`/`bf16ConvW` STAY FALSE, inherited from Tiny's bf16 render and load-bearing:
-- §19.1 measured this net's stem weight gradient at **0.19×** its f32 peer — a 209×209 window
-- with no bf16 cuDNN kernel — and the width axis does not touch the stem. So the one op where
-- bf16 is a LOSS on this architecture stays out of the emit at every size, which is why a green
-- gate here is allowed to mean what it usually means.
#eval IO.FS.writeFile "verified_mlir/vitsin_adamdp128x4wxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitsin_adamdp128x4wxclipdropbf16_train_step" "128.0" 4
    1000 0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128) (V := Proofs.StableHLO.vitSDims) (bf16 := true))
#guard "vitsin_adamdp128x4wxclipdropbf16_train_step" ==
  "vitsin_" ++ Proofs.StableHLO.vitAdamVariant 128 4 false true true true ++ "bf16" ++ "_train_step"

#eval IO.FS.writeFile "verified_mlir/vitsin_drop_fwd.mlir"
  (Proofs.StableHLO.vitFwdRenderB "vitsin_drop_fwd" 1000 (sd := true) (V := Proofs.StableHLO.vitSDims))

-- ⚠ The driver loads `<slug>_fwd.mlir` by NAME for its eval pass, so the `*_drop_fwd` above does
-- not satisfy it — `vitsin_drop_fwd` declares the 24 `%dp<n>` mask inputs an eval forward has no
-- values for. ViT-Tiny has both and so must S. Found by running the trainer, which failed with
-- `cannot open verified_mlir/vitsin_fwd.mlir`; no build-time check covers an artifact NAME.
#eval IO.FS.writeFile "verified_mlir/vitsin_fwd.mlir"
  (Proofs.StableHLO.vitFwdRenderB "vitsin_fwd" 1000 (V := Proofs.StableHLO.vitSDims))

-- ⚠⚠ **THE `32×4` PAIR IS GONE** (brett, 2026-08-27) — two artifacts deleted, and the reason is
-- that they had no axis left to win on. They existed because a phase-2 OOM at 4×128 was recorded
-- as a hardware limit; it was the CUDA plugin's `memory_fraction = 0.75` default, and at 0.97 the
-- 128×4 pair below both fits and runs. Against it, `32×4` was global 128 where DeiT's recipe is
-- **512**, it applied the reference's batch-512 LR to a batch four times too small, and it was
-- **slower** per epoch (322 h against 270 fp32, 228 against 155 bf16, device-only over 300
-- epochs) because a quarter of the per-device batch pays the per-invoke overhead four times as
-- often. ▶ This is the ViT half of what `2b2b15b` did to R50's `8×64`.
-- ⚠ The measurements that licensed the move are kept in `runs/2026-08-27-vitb-global512/`; the
-- artifacts are not.

-- ════════════════════════════════════════════════════════════════════════════════════════
-- § ▶ ViT-Base at **128 per device** — the probe that asks whether §4d's accumulation loop
--     is needed at all (`planning/archive/verified_side_quest_counterparts.md` §6a, step 1)
-- ════════════════════════════════════════════════════════════════════════════════════════
--
-- ⭐⭐ **The `32`-per-device pin above is a memory verdict, and the budget it was taken against
-- was WRONG.** The comment says the phase-2 JAX probe found ViT-B OOM at 4×128 "on these 16 GB
-- cards"; it did not — it found it OOM against **11.68 GiB**, which is the CUDA plugin's BFC
-- `memory_fraction = 0.75` DEFAULT and not the card. `ffi/pjrt_ffi.c` now passes the option and
-- `LEAN_MLIR_MEM_FRACTION=0.97` yields **15.11 GiB** (§6c, and XLA's own log line). So the pin has
-- to be re-taken against the real budget, and re-taking it needs the artifact to exist.
--
-- ▶ **Why this matters beyond a batch size.** DeiT's recipe is global **512**. At 32×4 this net
-- renders at global 128, which is a RECIPE deviation rather than a hardware footnote — a ViT-B
-- number produced at global 128 is not comparable to DeiT-B's 81.8% even in principle. 128×4 is
-- 512 in one shot, and one shot is what removes the need for `ViTRenderB` to grow an accumulation
-- loop it does not have (§4d, the only "real feature" left on the side-quest list).
--
-- ⭐⭐ **IT FITS, AND IT EXECUTED** (2026-08-27, `runs/2026-08-27-vitb-global512/`). Not a compile
-- figure: four 4060 Ti, ten steps each, `scripts/bf16_device_step.py --replicas 4`.
--
--   | render                    | peak    | of 15.11 | device ms/step | at the 11.68 default |
--   |---------------------------|---------|----------|----------------|----------------------|
--   | `adamdp128x4wxclipdrop`   | 13.99 G |   93 %   |   1298.92      | ⛔ `RESOURCE_EXHAUSTED` |
--   | `…dropbf16`               | 12.61 G |   83 %   |    746.25      | ✅ runs (10.88 G there) |
--
-- ▶ **The control is the finding.** The fp32 artifact above is ONE ENVIRONMENT VARIABLE from
-- failing: unset, the same bytes on the same four cards die *"Out of memory while trying to
-- allocate 11.96GiB"*. So §4d's accumulation loop was never a ViT-B fact, it was an unset
-- `memory_fraction`.
--
-- ⚠ **And the un-rematerialised graph really is 20.39 GiB** — XLA says so in its own words when
-- compiled against the default budget (*"Can't reduce memory use below 10.16GiB … down from
-- 20.39GiB originally"*). 13.99 is what rematerialisation buys, which is why a peak must be quoted
-- with the budget it was taken against: this pair reads 13.97/10.88 at 11.68 and 13.99/12.61 at
-- 15.11, because XLA rematerialises less when it has room (§6c saw the same on R50).
--
-- ⭐ **The all-reduce costs nothing measurable in the peak**, which is what the 1-replica peers
-- below were rendered to establish — §6a's third way this could fail, priced by DIFFERENCE rather
-- than assumed absent. A graph with no collective in it at all reads the same 13.99 / 12.61.
--
-- ⭐ **The 128×4 shape is not new to this renderer**, which is why this is four `#eval`s and not a
-- feature: `vitsin_adamdp128x4wxclipdrop` and `vitin_adamdp128x4wxclipdrop` both ship. ViT-B is
-- the only size that was pinned narrower, and only for the budget reason above.
#eval IO.FS.writeFile "verified_mlir/vitbin_adamdp128x4wxclipdrop_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitbin_adamdp128x4wxclipdrop_train_step" "128.0" 4
    1000 0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128) (V := Proofs.StableHLO.vitBDims))
#guard "vitbin_adamdp128x4wxclipdrop_train_step" ==
  "vitbin_" ++ Proofs.StableHLO.vitAdamVariant 128 4 false true true true ++ "_train_step"

#eval IO.FS.writeFile "verified_mlir/vitbin_adamdp128x4wxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitbin_adamdp128x4wxclipdropbf16_train_step" "128.0" 4
    1000 0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128) (V := Proofs.StableHLO.vitBDims) (bf16 := true))
#guard "vitbin_adamdp128x4wxclipdropbf16_train_step" ==
  "vitbin_" ++ Proofs.StableHLO.vitAdamVariant 128 4 false true true true ++ "bf16" ++ "_train_step"

-- ⚠⚠ **The 1-replica peers are a CONTROL, not a second recipe.** ViT-S and ViT-B have no
-- single-device render at all today, and that is what made §6a's third failure mode unfalsifiable:
-- "the all-reduce buffer is not in a single-device peak" cannot be checked without a graph that
-- has no all-reduce in it. These two are that graph — same net, same batch, same flags, `replicas
-- := 1` — so the collective's cost is the DIFFERENCE between two measurements rather than a
-- correction someone estimates.
-- ⚠ They render at global 128, so they are not a DeiT recipe and must not be quoted as one.
#eval IO.FS.writeFile "verified_mlir/vitbin_adam128wxclipdrop_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitbin_adam128wxclipdrop_train_step" "128.0" 1
    1000 0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128) (V := Proofs.StableHLO.vitBDims))
#guard "vitbin_adam128wxclipdrop_train_step" ==
  "vitbin_" ++ Proofs.StableHLO.vitAdamVariant 128 1 false true true true ++ "_train_step"

#eval IO.FS.writeFile "verified_mlir/vitbin_adam128wxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitbin_adam128wxclipdropbf16_train_step" "128.0" 1
    1000 0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (vbB := 128) (V := Proofs.StableHLO.vitBDims) (bf16 := true))
#guard "vitbin_adam128wxclipdropbf16_train_step" ==
  "vitbin_" ++ Proofs.StableHLO.vitAdamVariant 128 1 false true true true ++ "bf16" ++ "_train_step"


#eval IO.FS.writeFile "verified_mlir/vitbin_fwd.mlir"
  (Proofs.StableHLO.vitFwdRenderB "vitbin_fwd" 1000 (V := Proofs.StableHLO.vitBDims))

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

-- ════════════════════════════════════════════════════════════════
-- § ▶ 4c LEG 4 (2026-09-07): EVERY DROP-FREE ViT ARTIFACT, MOVED HERE FROM `ViTRender.lean`
-- ════════════════════════════════════════════════════════════════
--
-- ⭐⭐ These nineteen writers used to call `vitAdamTrainStepFaithful` / `vitFwdRenderV`, i.e. the
-- PER-EXAMPLE traversal, while the drop-bearing artifacts below already came from `vitBackAllB`.
-- One net, two chains. They now all call the batched wrapper, and the swap was licensed by
-- MEASUREMENT rather than argument: all nineteen re-render BYTE-IDENTICALLY, which is what
-- `TestBatchedEmitTie.lean`'s per-form ties predict and what a whole-net diff confirmed before a
-- single line moved.
--
-- ⭐⭐ The bytes did not move and the DENOTATION did, on exactly one parameter. The CLS token is
-- one shared `[192]` vector, so its gradient is the sum of every example's CLS-row cotangent —
-- `denseBiasGradB (N := vbB)` here against `(N := 1)` there, where `pretty B` did the batch lift
-- outside the AST. `Proofs.ViTPoCGB.clsGrad_denB` is the statement that becomes available, and
-- `ViTPoCG.clsGrad_den` is the same theorem at `N = 1`. See the note on that emission above.
--
-- ⚠ The comments below are the originals, unedited except for the call and where a claim moved.
-- ⛔ `vit_train_step.mlir` did NOT move and stays in `ViTRender.lean`: `vitBackAllB` has no
-- fused-SGD arm, and `ViTStepTie.lean`'s 200-parameter tie is stated at those bytes.

-- ⭐ The Imagenette forward, off the BATCHED chain since 4c leg 4 (2026-09-07) — byte-identical
-- to what `vitFwdRenderV` wrote, which `vit-fwd-b-tie` had been asserting since the chain landed.
#eval IO.FS.writeFile "verified_mlir/vit_fwd.mlir" (Proofs.StableHLO.vitFwdRenderB "vit_fwd")

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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adam_train_step" "32.0")

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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adamdp_train_step" "32.0" 2)

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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adam64_train_step" "64.0" 1 10 (vbB := 64))
#eval IO.FS.writeFile "verified_mlir/vit_adamdp64_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adamdp64_train_step" "64.0" 2 10 (vbB := 64))

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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adamdp32x4_train_step" "32.0" 4 10 (vbB := 32))

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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adam128_train_step" "128.0" 1 10 (vbB := 128))
#eval IO.FS.writeFile "verified_mlir/vit_adamdp128x4_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adamdp128x4_train_step" "128.0" 4 10 (vbB := 128))

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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adam128_train_step" "128.0" 1 1000
    (wdStr := "0.05") (vbB := 128))
#eval IO.FS.writeFile "verified_mlir/vitin_adamdp128x4_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adamdp128x4_train_step" "128.0" 4 1000
    (wdStr := "0.05") (vbB := 128))
#eval IO.FS.writeFile "verified_mlir/vitin_fwd.mlir"
  (Proofs.StableHLO.vitFwdRenderB "vitin_fwd" 1000 (vbB := 256))

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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adamwx_train_step" "32.0" 1 10 0.1
    (ema := false) (wdExclude := true) (vbB := 32))
-- ⚠ wd = **0.05**, not the file's 1e-4 default: `vitTinyImagenetConfig.weightDecay := 0.05` (the
-- DeiT value) where `vitTinyConfig` uses 1e-4. Both halves of the reference's decay recipe — the
-- MAGNITUDE and the MASK — have to be right for this render to be the pair's, and only the mask
-- was in scope when this variant was named. See `vitAdamConsts`.
#eval IO.FS.writeFile "verified_mlir/vitin_adam128wx_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adam128wx_train_step" "128.0" 1 1000 0.1
    (ema := false) (wdExclude := true) (wdStr := "0.05") (vbB := 128))

-- ── ▶ v1.4b: GLOBAL-NORM GRADIENT CLIPPING (`planning/archive/grad_clip.md`) ───────────────────────────
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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_adamclip_train_step" "32.0" 1 10 0.1
    (ema := false) (wdExclude := false) (wdStr := "0.0001") (clip := true) (clipStr := "1.0")
    (vbB := 32))
-- The ImageNet render — BOTH halves of the reference's recipe, `wx` ++ `clip`, because
-- `vitTinyImagenetConfig` sets `wdExcludeNormBias := true` AND `gradClipNorm := 1.0`. ⚠ wd = 0.05
-- for the same 500× reason `vitin_adam128wx` carries it.
#eval IO.FS.writeFile "verified_mlir/vitin_adam128wxclip_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adam128wxclip_train_step" "128.0" 1 1000 0.1
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (vbB := 128))
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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_adamdp128x4wxclip_train_step" "128.0" 4 1000
    0.1 (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (vbB := 128))

-- ── ▶ THE EMA VARIANT (`planning/archive/ema.md`), selected by `LEAN_MLIR_VARIANT=ema` ─────────────────
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
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vit_ema_train_step" "32.0" 1 10 0.1 (ema := true) (vbB := 32))

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

-- ── ▶ v1.2c: THE IMAGENET EMA PEER (`planning/archive/recipe_gaps.md` v1.2c) ──────────────────────────
-- `vitTinyImagenetConfig.emaDecay := 0.99996` (the DeiT default), so this is the render an ImageNet
-- ViT pair actually needs — `vit_ema` is Imagenette-scale and, as `MainViTVerifiedAdam` records, a GATE
-- VEHICLE rather than a matched pair (`vitTinyConfig` sets no EMA at all). Batch 128 × 4 replicas
-- = global 512, matching the reference, exactly as the `vitin_adam128` pair does.
#eval IO.FS.writeFile "verified_mlir/vitin_ema128_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_ema128_train_step" "128.0" 1 1000 0.1
    (ema := true) (vbB := 128))
#eval IO.FS.writeFile "verified_mlir/vitin_emadp128x4_train_step.mlir"
  (Proofs.StableHLO.vitAdamTrainStepFaithfulB "vitin_emadp128x4_train_step" "128.0" 4 1000 0.1
    (ema := true) (vbB := 128))
