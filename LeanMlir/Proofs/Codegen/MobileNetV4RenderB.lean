import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Codegen.MobileNetV2Render
import LeanMlir.ViTRender

/-! # MobileNetV4 — the Universal Inverted Bottleneck render (`planning/mnv4_verified.md` phase 3)

The UIB block, batch-BN, at `N := B`:

```
  optional pre-DW (k×k)  → BN → relu      -- takes the block's stride
  expand 1×1 (ic → mid)  → BN → relu
  optional post-DW (k×k) → BN → relu      -- at stride 1 if a pre-DW already consumed it
  project 1×1 (mid → oc) → BN             -- NO activation
  + skip                                  -- iff stride = 1 ∧ ic = oc; NO post-add activation
```

`mid = ic * expand`, every conv BN-followed and therefore **bias-free** — `VLayer.toSpecs` and
the baseline's `Layer.nParams` both assume that, and `uib-layout-tie` pins them to each other
(3,737,088 params over the 14-block table, all four families).

⭐ **`k = 0` omits that depthwise**, which is how one block expresses MNv4's four families:
ExtraDW (both), IB/MBConv (post only), ConvNeXt-like (pre only), FFN (neither). Those are `if`s
here, not separate functions, because omitting a shape-preserving op does not change any type.

**Phase 0 found this needs no new op and no new position** (`planning/mnv4_verified.md` §2): the
depthwise VJP is kernel-general (`cnx_render_dw7*_certified`; the descriptor carries `kH kW`), and
a leading depthwise already exists — `MobileNetV2RenderB.lean:196`'s `t = 1` inverted residual
emits `.depthwise (c := ic)` straight onto the block input. What is new is the **composition**:
ExtraDW puts a depthwise on *both* sides of the pointwise expand.

## Why three functions and not one

`.depthwise : BatchableOp (c*h*w) (c*h*w)` and `.depthwiseStrided : BatchableOp (c*(2h)*(2w))
(c*h*w)` have different INPUT types, so a stride-polymorphic block cannot typecheck — the same
reason `MobileNetV2RenderB` splits `irFwdStridedB` from `irFwdSkipB`. The stride-2 case splits
again by **which depthwise consumes the stride**, because that decides the spatial size the expand
runs at. Read off the Conv-M table (`jax/MainMobilenetV4.lean`), which lands cleanly:

| | stride | `ic` vs `oc` | function |
|---|---|---|---|
| 11 blocks | 1 | `ic = oc` | `uibFwdSkipB` |
| 2 blocks | 2 | `ic ≠ oc`, `preDWk > 0` | `uibFwdPreStridedB` |
| 1 block  | 2 | `ic ≠ oc`, `preDWk = 0` | `uibFwdPostStridedB` |

⚠⚠ **ACTIVATION IS PLAIN `relu`, NOT `relu6`.** MobileNetV2's blocks use relu6 and this file sits
next to that renderer, so the wrong one is one keystroke away. Read off the baseline emitter
(`MlirCodegen.lean:6357`, "Plain ReLU throughout").

⚠ **A pre/post-DW swap is invisible to every count.** Same `k`, same channels ⇒ same parameter
shapes, so `uib-layout-tie` passes on a renderer that swaps them, and so does any arity or op-count
audit. At stride 1 it is invisible to the TYPES too, since both positions are shape-preserving —
which is why the four families are `if`s that the compiler cannot check. Only a forward tie against
the reference on shared weights pins the order. Same class as R50's stride-on-the-3×3.

⚠ The baseline drops the stride entirely for a stride-2 FFN block (no depthwise to carry it,
`MlirCodegen.lean:6364`). No such block exists in the table; this file has no function for that
shape, so the case is absent rather than silently wrong.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-- One UIB block's parameters, in **`VLayer.toSpecs` order**: pre-DW? → expand → post-DW? →
    project, each `{W, γ, β}` and bias-free. The names are exactly what `uibFwd*B` emits, and the
    order is exactly what `toSpecs` lays out — those two facts are what make the signature and the
    driver's parameter blob describe the same thing.

    ⚠ A depthwise kernel is `[c, 1, k, k]`, not `[c, c, k, k]`. That is the whole difference between
    a depthwise and a regular conv at this layer, and it is 3 orders of magnitude of parameters. -/
private def uibSig (p : String) (ic oc expand preDWk postDWk : Nat) : List (String × List Nat) :=
  let mid := ic * expand
  (if preDWk > 0 then
     [(s!"%u{p}qW", [ic, 1, preDWk, preDWk]), (s!"%u{p}qg", [ic]), (s!"%u{p}qbt", [ic])] else []) ++
  [(s!"%u{p}eW", [mid, ic, 1, 1]), (s!"%u{p}eg", [mid]), (s!"%u{p}ebt", [mid])] ++
  (if postDWk > 0 then
     [(s!"%u{p}dW", [mid, 1, postDWk, postDWk]), (s!"%u{p}dg", [mid]), (s!"%u{p}dbt", [mid])] else []) ++
  [(s!"%u{p}pW", [oc, mid, 1, 1]), (s!"%u{p}pg", [oc]), (s!"%u{p}pbt", [oc])]

/-- The fused stage's parameters, in `toSpecs` order: the `k×k` regular conv then the 1×1 project. -/
private def fusedSig (p : String) (ic oc expand k : Nat) : List (String × List Nat) :=
  let mid := if expand == 1 then oc else ic * expand
  [(s!"%f{p}cW", [mid, ic, k, k]), (s!"%f{p}cg", [mid]), (s!"%f{p}cbt", [mid])] ++
  (if expand == 1 then [] else
     [(s!"%f{p}pW", [oc, mid, 1, 1]), (s!"%f{p}pg", [oc]), (s!"%f{p}pbt", [oc])])

/-- **One row of the MobileNetV4-Conv-M block table.** `h` is the block's OUTPUT spatial size, so
    a `stride2` block reads its input at `2h`. -/
structure UibSpec where
  /-- parameter-name prefix: `"1"` … `"21"` (Conv-M; Conv-S ran to `"14"`). -/
  p : String
  ic : Nat
  oc : Nat
  expand : Nat
  /-- pre-depthwise kernel, `0` = absent. -/
  preDWk : Nat
  /-- post-depthwise kernel, `0` = absent. -/
  postDWk : Nat
  h : Nat
  stride2 : Bool
deriving Inhabited, DecidableEq

/-- **THE BLOCK TABLE — transcribed once, from `jax/MainMobilenetV4.lean`.**

    ⭐⭐ Everything downstream folds over this list: the parameter signature, the BN stat slots, the
    forward chain, the backward chain and the running-statistic recomputes. Before it existed the
    same rows were hand-written FOUR times, and §3/§7.2's whole point is that a divergence
    between two such readings is invisible — same ops, same channel counts, same types, different
    net. One table means a dispatch error is a typo in one place rather than a mismatch nothing
    checks.

    Families in order (Conv-**M**): ExtraDW ×7, ConvNeXt, FFN, ConvNeXt, ExtraDW ×4, FFN, ConvNeXt,
    ExtraDW ×2, FFN ×2, ConvNeXt — 13 ExtraDW / 4 ConvNeXt / 4 FFN, and **no IB at all**, where
    Conv-S used three. Spatial ladder 56 → 28 → 14 → 7.

    ⚠ Verified against **timm 1.0.28** (`mobilenetv4_conv_medium`, walking `model.blocks[1:4]`):
    all 21 rows agree on `(ic, oc, expand, preDWk, postDWk, h, stride2)`. The `#guard`s in
    `Proofs/Foundation/MobileNetV4BackB0.lean` pin that reading; they are derived from timm rather
    than re-read off this table, or they would gate nothing. -/
def mnv4Blocks : List UibSpec :=
  [ ⟨"1",   48,  80, 4, 3, 5, 28, true⟩,   -- ExtraDW  56→28
    ⟨"2",   80,  80, 2, 3, 3, 28, false⟩,  -- ExtraDW  28
    ⟨"3",   80, 160, 6, 3, 5, 14, true⟩,   -- ExtraDW  28→14
    ⟨"4",  160, 160, 4, 3, 3, 14, false⟩,  -- ExtraDW  14
    ⟨"5",  160, 160, 4, 3, 3, 14, false⟩,  -- ExtraDW  14
    ⟨"6",  160, 160, 4, 3, 5, 14, false⟩,  -- ExtraDW  14
    ⟨"7",  160, 160, 4, 3, 3, 14, false⟩,  -- ExtraDW  14
    ⟨"8",  160, 160, 4, 3, 0, 14, false⟩,  -- ConvNeXt 14
    ⟨"9",  160, 160, 2, 0, 0, 14, false⟩,  -- FFN      14
    ⟨"10", 160, 160, 4, 3, 0, 14, false⟩,  -- ConvNeXt 14
    ⟨"11", 160, 256, 6, 5, 5,  7, true⟩,   -- ExtraDW  14→7
    ⟨"12", 256, 256, 4, 5, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"13", 256, 256, 4, 3, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"14", 256, 256, 4, 3, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"15", 256, 256, 4, 0, 0,  7, false⟩,  -- FFN      7
    ⟨"16", 256, 256, 4, 3, 0,  7, false⟩,  -- ConvNeXt 7
    ⟨"17", 256, 256, 2, 3, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"18", 256, 256, 4, 5, 5,  7, false⟩,  -- ExtraDW  7
    ⟨"19", 256, 256, 4, 0, 0,  7, false⟩,  -- FFN      7
    ⟨"20", 256, 256, 4, 0, 0,  7, false⟩,  -- FFN      7
    ⟨"21", 256, 256, 2, 5, 0,  7, false⟩ ] -- ConvNeXt 7

/-- **The MobileNetV4-Conv-M parameter inputs**, in func-arg order: stem (3), the fused stage, the
    21 UIB blocks, the TWO head convs, the classifier. Single source for the signature and the return
    order, the same role `r50ShapeList` plays for R50.

    ⚠ This list and `VLayer.toSpecs` are TWO HAND-WRITTEN READINGS of the same layout — the
    renderer cannot import the spec without inverting the dependency, which is the same two-lists
    shape as `toSpecs == XLayout.specs` elsewhere. `mnv4-fwd-smoke` is the gate that pins them. -/
def mnv4ShapeList (nClasses : Nat) : List (String × List Nat) :=
  [("%sW", [32, 3, 3, 3]), ("%sg", [32]), ("%sbt", [32])] ++
  fusedSig "0" 32 48 4 3 ++
  mnv4Blocks.flatMap (fun b => uibSig b.p b.ic b.oc b.expand b.preDWk b.postDWk) ++
  [("%h1W", [960, 256, 1, 1]), ("%h1g", [960]), ("%h1bt", [960])] ++
  [("%hW", [1280, 960, 1, 1]), ("%hg", [1280]), ("%hbt", [1280])] ++
  [("%Wd", [1280, nClasses]), ("%bd", [nClasses])]

/-- The same list as MLIR types. Derived, so the shapes have one definition. -/
def mnv4SigList (nClasses : Nat) : List (String × String) :=
  (mnv4ShapeList nClasses).map (fun (n, ds) => (n, ty ds))

/-- The running-mean/var slots for one BN layer. -/
private def bnStatSig4 (nm : String) (c : Nat) : List (String × List Nat) :=
  [(s!"%{nm}mu", [c]), (s!"%{nm}var", [c])]

/-- One UIB block's BN stat slots, in the SAME order `uibSig` lays out its parameters — pre-DW
    (at `ic`), expand (at `mid`), post-DW (at `mid`), project (at `oc`), each present exactly when
    its conv is. The `if`s are the same `k = 0` dispatch, so a family that drops a depthwise drops
    its stat slot too and the two lists cannot go out of step. -/
private def uibStatSig (p : String) (ic oc expand preDWk postDWk : Nat) : List (String × List Nat) :=
  let mid := ic * expand
  (if preDWk > 0 then bnStatSig4 s!"u{p}qn" ic else []) ++
  bnStatSig4 s!"u{p}en" mid ++
  (if postDWk > 0 then bnStatSig4 s!"u{p}dn" mid else []) ++
  bnStatSig4 s!"u{p}pn" oc

/-- The fused stage's BN stat slots — the `k×k` conv's, then the 1×1 project's. -/
private def fusedStatSig (p : String) (ic oc expand : Nat) : List (String × List Nat) :=
  let mid := if expand == 1 then oc else ic * expand
  bnStatSig4 s!"f{p}cn" mid ++ (if expand == 1 then [] else bnStatSig4 s!"f{p}pn" oc)

/-- **The 104 BN running-statistic slots** = 52 BN layers × (μ, var), in forward-traversal order:
    stem, the fused stage's two, each UIB block's (2–4 depending on family), the head.

    ⚠ A misaligned stat slot is **SILENT**: the arities still match and the wrong layer's statistics
    simply flow into the wrong `@mnv4_fwd_eval` slot. That is why the order here and the order the
    train step returns them in are both derived from the same block table, and why the eval forward
    reads them through `mnv4Bn`'s `statP` rather than an independently-numbered list. -/
def mnv4StatShapeList : List (String × List Nat) :=
  bnStatSig4 "stn" 32 ++
  fusedStatSig "0" 32 48 4 ++
  mnv4Blocks.flatMap (fun b => uibStatSig b.p b.ic b.oc b.expand b.preDWk b.postDWk) ++
  bnStatSig4 "h1n" 960 ++
  bnStatSig4 "hn" 1280

/-- The stat slots as MLIR types. Derived, so the widths have one definition. -/
def mnv4StatSigList : List (String × String) :=
  mnv4StatShapeList.map (fun (n, ds) => (n, ty ds))

/-- **One BN site at the batched index** — the ONLY place the two BN worlds are chosen between.

    `statP` names the running-stat slot: in `.eval` it is read (`%{statP}mu`/`%{statP}var`), in
    `.train` it is the slot the train step hands the batch statistics back in. Both modes call
    `pretty` exactly once, so the fresh-name counter advances identically and `@mnv4_fwd`
    re-renders byte-identical after this threading — the cheap self-check that it is inert.

    ⚠⚠ **This exists so `@mnv4_fwd` and `@mnv4_fwd_eval` cannot be different nets.** That is not a
    hypothetical: `planning/mnv4_verified.md` §3d(b) measured `mobilenetv2_fwd` in a *different BN
    world* from the Adam train step that trains it, and `regen_verified_mlir.sh check` went green
    anyway because it only ever pairs a forward with the SGD step. One traversal, one switch, per
    the `ResNet50RenderB` rule — the divergence has nowhere to live. -/
private def mnv4Bn (B oc h : Nat) (mode : BnMode) (epsStr gName btName statP xin : String) :
    StateM Nat (String × String) := do
  let zc  : Vec oc := fun _ => 0
  let zin : Vec (B * (oc*h*h)) := fun _ => 0
  match mode with
  | .train => pretty B (.bnBatchF (N := B) (oc := oc) (h := h) (w := h)
                          gName btName epsStr 0 zc zc (.operand xin zin))
  | .eval  => pretty B (.batchOp (N := B) (.bnEval (oc := oc) (h := h) (w := h)
                          gName btName s!"%{statP}mu" s!"%{statP}var" epsStr 0 zc zc zc zc)
                          (.operand xin zin))

/-- Saved forward SSA names the UIB backward + gradient passes reference. The optional fields are
    `""` when their depthwise is absent (`k = 0`), which is how the backward reads off which family
    it is looking at without re-deriving it from the kernel sizes. -/
structure UibFwdB where
  code : String
  o  : String        -- block output (project-BN out, or the skip add)
  qc : String        -- pre-DW conv / BN / relu out; "" when preDWk = 0
  qn : String
  qr : String
  ec : String        -- expand conv out (= expand-BN input)
  en : String        -- expand BN out   (= expand-relu pre-activation)
  er : String        -- expand relu out (= post-DW input, or project input when postDWk = 0)
  dc : String        -- post-DW conv / BN / relu out; "" when postDWk = 0
  dn : String
  dr : String
  pc : String        -- project conv out (= project-BN input)
deriving Inhabited

/-- **Stride-1 UIB with the identity skip** (`ic = oc = c`) — 11 of the table's 14 blocks, and all
    four families. Everything runs at `h×h`: both depthwise positions are shape-preserving, so the
    `k = 0` omissions are plain `if`s. Block output = `addVB (project-BN out) (block input)`; the
    bottleneck is LINEAR, no activation after the add. -/
private def uibFwdSkipB (B c expand preDWk postDWk h : Nat) (mode : BnMode)
    (epsStr p xName : String)
    (bf16 : Bool := false) : StateM Nat UibFwdB := do
  let mid := c * expand
  -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are: the render produces TEXT
  -- and `skel` erases every ℝ payload before a token is emitted.
  let zrnd : ℝ → ℝ := fun r => r
  let zc   : Vec c := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zqk  : DepthwiseKernel c preDWk preDWk := fun _ _ _ => 0
  let zdk  : DepthwiseKernel mid postDWk postDWk := fun _ _ _ => 0
  let zke  : Kernel4 mid c 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 c mid 1 1 := fun _ _ _ _ => 0
  let zcb  : Vec (B*(c*h*h)) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0

  let mut code := ""
  let mut qc := ""; let mut qn := ""; let mut qr := ""
  let mut cur := xName
  if preDWk > 0 then
    let (c1, n1) ← pretty B (.batchOp (N := B)
      (if bf16 then .depthwiseBf16 (c := c) (h := h) (w := h) zrnd s!"%u{p}qW" s!"%zb{c}" zqk zc else .depthwise (c := c) (h := h) (w := h) s!"%u{p}qW" s!"%zb{c}" zqk zc) (.operand cur zcb))
    let (c2, n2) ← mnv4Bn B c h mode epsStr s!"%u{p}qg" s!"%u{p}qbt" s!"u{p}qn" n1
    let (c3, n3) ← pretty B (.batchOp (N := B) (.relu (n := c*h*h)) (.operand n2 zcb))
    code := code ++ c1 ++ c2 ++ c3
    qc := n1; qn := n2; qr := n3; cur := n3

  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := c) (oc := mid) (h := h) (w := h) zrnd s!"%u{p}eW" s!"%zb{mid}" zke zm else .conv (ic := c) (oc := mid) (h := h) (w := h) s!"%u{p}eW" s!"%zb{mid}" zke zm) (.operand cur zcb))
  let (cEn, nEn) ← mnv4Bn B mid h mode epsStr s!"%u{p}eg" s!"%u{p}ebt" s!"u{p}en" nEc
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand nEn zmb))
  code := code ++ cEc ++ cEn ++ cEr
  cur := nEr

  let mut dc := ""; let mut dn := ""; let mut dr := ""
  if postDWk > 0 then
    let (c1, n1) ← pretty B (.batchOp (N := B)
      (if bf16 then .depthwiseBf16 (c := mid) (h := h) (w := h) zrnd s!"%u{p}dW" s!"%zb{mid}" zdk zm else .depthwise (c := mid) (h := h) (w := h) s!"%u{p}dW" s!"%zb{mid}" zdk zm) (.operand cur zmb))
    let (c2, n2) ← mnv4Bn B mid h mode epsStr s!"%u{p}dg" s!"%u{p}dbt" s!"u{p}dn" n1
    let (c3, n3) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand n2 zmb))
    code := code ++ c1 ++ c2 ++ c3
    dc := n1; dn := n2; dr := n3; cur := n3

  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := mid) (oc := c) (h := h) (w := h) zrnd s!"%u{p}pW" s!"%zb{c}" zkp zc else .conv (ic := mid) (oc := c) (h := h) (w := h) s!"%u{p}pW" s!"%zb{c}" zkp zc) (.operand cur zmb))
  let (cPn, nPn) ← mnv4Bn B c h mode epsStr s!"%u{p}pg" s!"%u{p}pbt" s!"u{p}pn" nPc
  let (cA, nA) ← pretty B (.addVB (.operand nPn zcb) (.operand xName zcb))
  code := code ++ cPc ++ cPn ++ cA

  pure { code := code, o := nA, qc := qc, qn := qn, qr := qr,
         ec := nEc, en := nEn, er := nEr, dc := dc, dn := dn, dr := dr, pc := nPc }

/-- **Stride-2 UIB where the PRE-DW carries the stride** (`preDWk > 0`). The pre-DW downsamples
    `2h×2h → h×h`, so the expand, the optional post-DW (now at stride 1) and the project all run at
    `h×h`. No skip — `ic ≠ oc`. -/
private def uibFwdPreStridedB (B ic oc expand preDWk postDWk h : Nat) (mode : BnMode)
    (epsStr p xName : String)
    (bf16 : Bool := false) : StateM Nat UibFwdB := do
  let mid := ic * expand
  -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are: the render produces TEXT
  -- and `skel` erases every ℝ payload before a token is emitted.
  let zrnd : ℝ → ℝ := fun r => r
  let zic  : Vec ic := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zqk  : DepthwiseKernel ic preDWk preDWk := fun _ _ _ => 0
  let zdk  : DepthwiseKernel mid postDWk postDWk := fun _ _ _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zin  : Vec (B*(ic*(2*h)*(2*h))) := fun _ => 0
  let zqb  : Vec (B*(ic*h*h)) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0

  let (cQc, nQc) ← pretty B (.batchOp (N := B)
    (if bf16 then .depthwiseStridedBf16 (c := ic) (h := h) (w := h) zrnd s!"%u{p}qW" s!"%zb{ic}" zqk zic else .depthwiseStrided (c := ic) (h := h) (w := h) s!"%u{p}qW" s!"%zb{ic}" zqk zic) (.operand xName zin))
  let (cQn, nQn) ← mnv4Bn B ic h mode epsStr s!"%u{p}qg" s!"%u{p}qbt" s!"u{p}qn" nQc
  let (cQr, nQr) ← pretty B (.batchOp (N := B) (.relu (n := ic*h*h)) (.operand nQn zqb))

  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := ic) (oc := mid) (h := h) (w := h) zrnd s!"%u{p}eW" s!"%zb{mid}" zke zm else .conv (ic := ic) (oc := mid) (h := h) (w := h) s!"%u{p}eW" s!"%zb{mid}" zke zm) (.operand nQr zqb))
  let (cEn, nEn) ← mnv4Bn B mid h mode epsStr s!"%u{p}eg" s!"%u{p}ebt" s!"u{p}en" nEc
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand nEn zmb))

  let mut code := cQc ++ cQn ++ cQr ++ cEc ++ cEn ++ cEr
  let mut cur := nEr
  let mut dc := ""; let mut dn := ""; let mut dr := ""
  if postDWk > 0 then
    let (c1, n1) ← pretty B (.batchOp (N := B)
      (if bf16 then .depthwiseBf16 (c := mid) (h := h) (w := h) zrnd s!"%u{p}dW" s!"%zb{mid}" zdk zm else .depthwise (c := mid) (h := h) (w := h) s!"%u{p}dW" s!"%zb{mid}" zdk zm) (.operand cur zmb))
    let (c2, n2) ← mnv4Bn B mid h mode epsStr s!"%u{p}dg" s!"%u{p}dbt" s!"u{p}dn" n1
    let (c3, n3) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand n2 zmb))
    code := code ++ c1 ++ c2 ++ c3
    dc := n1; dn := n2; dr := n3; cur := n3

  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := mid) (oc := oc) (h := h) (w := h) zrnd s!"%u{p}pW" s!"%zb{oc}" zkp zoc else .conv (ic := mid) (oc := oc) (h := h) (w := h) s!"%u{p}pW" s!"%zb{oc}" zkp zoc) (.operand cur zmb))
  let (cPn, nPn) ← mnv4Bn B oc h mode epsStr s!"%u{p}pg" s!"%u{p}pbt" s!"u{p}pn" nPc
  code := code ++ cPc ++ cPn

  pure { code := code, o := nPn, qc := nQc, qn := nQn, qr := nQr,
         ec := nEc, en := nEn, er := nEr, dc := dc, dn := dn, dr := dr, pc := nPc }

/-- **Stride-2 UIB where the POST-DW carries the stride** (`preDWk = 0`, `postDWk > 0`) — the IB /
    MBConv family at a downsample. The expand runs at the INPUT size `2h×2h`; the post-DW then
    downsamples to `h×h`. No skip — `ic ≠ oc`. -/
private def uibFwdPostStridedB (B ic oc expand postDWk h : Nat) (mode : BnMode)
    (epsStr p xName : String)
    (bf16 : Bool := false) : StateM Nat UibFwdB := do
  let mid := ic * expand
  -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are: the render produces TEXT
  -- and `skel` erases every ℝ payload before a token is emitted.
  let zrnd : ℝ → ℝ := fun r => r
  let zoc  : Vec oc := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zdk  : DepthwiseKernel mid postDWk postDWk := fun _ _ _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zin  : Vec (B*(ic*(2*h)*(2*h))) := fun _ => 0
  let zeb  : Vec (B*(mid*(2*h)*(2*h))) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0

  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := ic) (oc := mid) (h := 2*h) (w := 2*h) zrnd s!"%u{p}eW" s!"%zb{mid}" zke zm else .conv (ic := ic) (oc := mid) (h := 2*h) (w := 2*h) s!"%u{p}eW" s!"%zb{mid}" zke zm) (.operand xName zin))
  let (cEn, nEn) ← mnv4Bn B mid (2*h) mode epsStr s!"%u{p}eg" s!"%u{p}ebt" s!"u{p}en" nEc
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu (n := mid*(2*h)*(2*h))) (.operand nEn zeb))

  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (if bf16 then .depthwiseStridedBf16 (c := mid) (h := h) (w := h) zrnd s!"%u{p}dW" s!"%zb{mid}" zdk zm else .depthwiseStrided (c := mid) (h := h) (w := h) s!"%u{p}dW" s!"%zb{mid}" zdk zm) (.operand nEr zeb))
  let (cDn, nDn) ← mnv4Bn B mid h mode epsStr s!"%u{p}dg" s!"%u{p}dbt" s!"u{p}dn" nDc
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand nDn zmb))

  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := mid) (oc := oc) (h := h) (w := h) zrnd s!"%u{p}pW" s!"%zb{oc}" zkp zoc else .conv (ic := mid) (oc := oc) (h := h) (w := h) s!"%u{p}pW" s!"%zb{oc}" zkp zoc) (.operand nDr zmb))
  let (cPn, nPn) ← mnv4Bn B oc h mode epsStr s!"%u{p}pg" s!"%u{p}pbt" s!"u{p}pn" nPc

  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, qc := "", qn := "", qr := "",
         ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc }

/-- **Fused inverted bottleneck, stride 2, no SE** — MobileNetV4's stage 0
    (`.fusedMbConv 32 48 4 3 2 1 false`) and EfficientNetV2's early stages.

    ```
      k×k regular conv ic→mid at stride 2  → BN → swish     -- 2h×2h → h×h
      1×1 project mid→oc                   → BN             -- NO activation
    ```

    "Fused" means the MBConv expand-1×1 and its depthwise collapse into ONE regular `k×k` conv, so
    despite living in a mobile net there is nothing depthwise here. No skip: `ic ≠ oc` and stride 2.

    ⚠⚠ **SWISH, NOT RELU — and that is a deviation from the MNv4 paper, on purpose.** MobileNetV4-Conv
    is a ReLU network, but the reference that produced 84.58% uses swish at this site
    (`jax/Jax/Codegen.lean:1031`), inherited from the block being shared with EfficientNetV2. This
    render must match the REFERENCE or the forward tie cannot pass and the number cannot be
    reproduced. `uibFwd*` above are relu, correctly — the two activations sit twenty lines apart and
    the difference is real, not a copy-paste slip. -/
private def fusedMbConvFwdStridedB (B ic oc expand k h : Nat) (mode : BnMode)
    (epsStr p xName : String)
    (bf16 : Bool := false) : StateM Nat UibFwdB := do
  let mid := if expand == 1 then oc else ic * expand
  -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are: the render produces TEXT
  -- and `skel` erases every ℝ payload before a token is emitted.
  let zrnd : ℝ → ℝ := fun r => r
  let zoc  : Vec oc := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zkf  : Kernel4 mid ic k k := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zin  : Vec (B*(ic*(2*h)*(2*h))) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0

  let (cFc, nFc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convStridedBf16 (ic := ic) (oc := mid) (h := h) (w := h) (kH := k) (kW := k) zrnd
      s!"%f{p}cW" s!"%zb{mid}" zkf zm else .convStrided (ic := ic) (oc := mid) (h := h) (w := h) (kH := k) (kW := k)
      s!"%f{p}cW" s!"%zb{mid}" zkf zm) (.operand xName zin))
  let (cFn, nFn) ← mnv4Bn B mid h mode epsStr s!"%f{p}cg" s!"%f{p}cbt" s!"f{p}cn" nFc
  let (cFs, nFs) ← pretty B (.batchOp (N := B) (.swish (n := mid*h*h)) (.operand nFn zmb))

  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := mid) (oc := oc) (h := h) (w := h) zrnd s!"%f{p}pW" s!"%zb{oc}" zkp zoc else .conv (ic := mid) (oc := oc) (h := h) (w := h) s!"%f{p}pW" s!"%zb{oc}" zkp zoc) (.operand nFs zmb))
  let (cPn, nPn) ← mnv4Bn B oc h mode epsStr s!"%f{p}pg" s!"%f{p}pbt" s!"f{p}pn" nPc

  pure { code := cFc ++ cFn ++ cFs ++ cPc ++ cPn,
         o := nPn, qc := "", qn := "", qr := "",
         ec := nFc, en := nFn, er := nFs, dc := "", dn := "", dr := "", pc := nPc }

/-- **Block-shape dispatch, in one place.** Which of the three forwards a row uses is forced by
    the row: stride 1 ⇒ the identity skip (`ic = oc`, pinned by `mnv4-fwd-smoke`); stride 2 splits
    on which depthwise carries the stride, because that decides the spatial size the expand runs at.
    The three cannot be one function — `.depthwise` and `.depthwiseStrided` differ in INPUT type. -/
private def uibFwdDispatch (B : Nat) (b : UibSpec) (mode : BnMode) (epsStr xName : String)
    (bf16 : Bool := false) : StateM Nat UibFwdB :=
  if b.stride2 then
    if b.preDWk > 0 then
      uibFwdPreStridedB B b.ic b.oc b.expand b.preDWk b.postDWk b.h mode epsStr b.p xName bf16
    else
      uibFwdPostStridedB B b.ic b.oc b.expand b.postDWk b.h mode epsStr b.p xName bf16
  else
    uibFwdSkipB B b.ic b.expand b.preDWk b.postDWk b.h mode epsStr b.p xName bf16

/-- Everything the whole-net render needs out of one forward traversal: the emitted code, the
    logits, and every saved activation the backward reads. `inputs` is each block's INPUT SSA name
    (`inputs[i]` is what block `i` consumed), which the weight gradients contract against. -/
structure Mnv4FwdRec where
  code : String
  logits : String
  stc : String          -- stem conv out (= stem-BN input)
  stn : String          -- stem BN out   (= stem-relu pre-activation)
  str : String          -- stem relu out
  f0 : UibFwdB          -- the fused stage
  blocks : List UibFwdB -- the 14 UIB blocks, in table order
  inputs : List String  -- each block's input SSA, same order
  h1c : String          -- head conv 1 out (256→960)  (= its BN input)
  h1n : String          -- head BN 1 out              (= its relu pre-activation)
  h1r : String          -- head relu 1 out            (= head conv 2's input)
  hc : String           -- head conv 2 out (960→1280) (= head-BN input)
  hn : String           -- head BN out   (= head-relu pre-activation)
  hr : String           -- head relu out
  gap : String          -- GAP out (= dense input)
  last : String         -- the last block's output (= head conv input)
deriving Inhabited

/-- **The MobileNetV4-Conv-M forward chain**, batch BN, at `N := B`, 224² → 10 classes.

    Transcribed 1:1 from `jax/MainMobilenetV4.lean`, which is the faithful Conv-M table as of
    2026-08-14 (`RESULTS.md`'s **84.58%** belongs to the SUPERSEDED Conv-S table). Spatial ladder:

    ```
      224 --stem s2--> 112 --fused s2--> 56 --uib s2--> 28 --uib s2--> 14 --uib s2--> 7 --GAP--> 1
    ```

    Block dispatch is forced by the table and checked by the types: the three stride-2 blocks are
    `ic ≠ oc` and split by which depthwise carries the stride; the eleven stride-1 blocks are all
    `ic = oc`, hence all skip. Families in order: Fused, ExtraDW, ExtraDW, IB, ExtraDW, ExtraDW,
    ConvNeXt, IB, ConvNeXt, FFN, ExtraDW, ExtraDW, ExtraDW, IB, ConvNeXt.

    **Activations, and they are not uniform** — each read off the emitter that produced the number,
    not assumed:
    * stem and head `.convBn` → **relu** (`MlirCodegen.lean:5852`, `emitConvBnTrain … useRelu := true`)
    * the fused stage → **swish** (`jax/Jax/Codegen.lean:1031`) — a deliberate paper deviation, see
      `fusedMbConvFwdStridedB`
    * every UIB block → **relu** (`MlirCodegen.lean:6357`, "Plain ReLU throughout")

    Returns the full forward RECORD, not just `(code, logits)`: the train step needs every saved
    activation the backward reads, and the alternative — a second copy of the chain inside the
    train step — is the two-readings defect this file exists to avoid. -/
def mnv4FwdChainB (B nClasses : Nat) (epsStr : String) (mode : BnMode := .train)
    -- ▶ TRAILING and defaulted, so `@mnv4_fwd` / `@mnv4_fwd_eval` re-render byte-identical.
    (bf16 : Bool := false) : StateM Nat Mnv4FwdRec := do
  -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are: the render produces TEXT
  -- and `skel` erases every ℝ payload before a token is emitted.
  let zrnd : ℝ → ℝ := fun r => r
  -- ═══ stem: 3×3/s2 conv (3→32), 224→112 → batch BN → relu ═══
  -- ⭐ `.convStridedXla`, NOT `.convStrided` — and this net is the reason that token exists.
  -- The reference stem is `conv_bn(…, stride=(2,2), padding='SAME')`, and XLA `'SAME'` on a 3×3/s2
  -- at 224 pads **(0,1)**, not (1,1). Both give 112×112, so no shape check, `#guard`, op count or
  -- arity audit can see the difference — the forward tie is the only thing that can, and it
  -- measured **6.16e-2** with the symmetric token against **1.79e-6** with the reference patched
  -- to match (`planning/mnv4_verified.md` §3b). Every OTHER stride-2 site in this net is genuinely
  -- symmetric: `uib_block` and `fused_mbconv_block` pass an explicit `(pad,pad)` tuple, which is
  -- why patching this one line alone closed the whole tie.
  let zx    : Vec (B*(3*224*224)) := fun _ => 0
  let zSk   : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
  let z32   : Vec 32 := fun _ => 0
  let z112  : Vec (B*(32*112*112)) := fun _ => 0
  let (cStc, nStc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convStridedXlaBf16 (ic := 3) (oc := 32) (h := 112) (w := 112) (kH := 3) (kW := 3) zrnd "%sW" "%zb32" zSk z32 else .convStridedXla (ic := 3) (oc := 32) (h := 112) (w := 112) (kH := 3) (kW := 3) "%sW" "%zb32" zSk z32)
    (.operand "%x" zx))
  let (cStn, nStn) ← mnv4Bn B 32 112 mode epsStr "%sg" "%sbt" "stn" nStc
  let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu (n := 32*112*112)) (.operand nStn z112))

  -- ═══ stage 0: the fused inverted bottleneck, 112→56 (swish) ═══
  let f0 ← fusedMbConvFwdStridedB B 32 48 4 3 56 mode epsStr "0" nStr bf16

  -- ═══ the 14 UIB blocks — ONE fold over `mnv4Blocks`, dispatch by the row ═══
  let mut cur := f0.o
  let mut bcode := ""
  let mut blocks : List UibFwdB := []
  let mut inputs : List String := []
  for b in mnv4Blocks do
    let r ← uibFwdDispatch B b mode epsStr cur bf16
    bcode := bcode ++ r.code
    blocks := blocks ++ [r]
    inputs := inputs ++ [cur]
    cur := r.o

  -- ═══ head, TWO convs (Conv-M's `cn_r1_k1_s1_c960` then `conv_head` to 1280):
  --     1×1 (256→960) → BN → relu → 1×1 (960→1280) → BN → relu → GAP(7×7) → dense ═══
  let z7     : Vec (B*(256*7*7)) := fun _ => 0
  let zH1k   : Kernel4 960 256 1 1 := fun _ _ _ _ => 0
  let z960   : Vec 960 := fun _ => 0
  let zH17   : Vec (B*(960*7*7)) := fun _ => 0
  let zHk    : Kernel4 1280 960 1 1 := fun _ _ _ _ => 0
  let z1280  : Vec 1280 := fun _ => 0
  let zH7    : Vec (B*(1280*7*7)) := fun _ => 0
  let z1280b : Vec (B*1280) := fun _ => 0
  let zWd    : Mat 1280 nClasses := fun _ _ => 0
  let zNC    : Vec nClasses := fun _ => 0
  let (cH1c, nH1c) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := 256) (oc := 960) (h := 7) (w := 7) zrnd "%h1W" "%zb960" zH1k z960 else .conv (ic := 256) (oc := 960) (h := 7) (w := 7) "%h1W" "%zb960" zH1k z960) (.operand cur z7))
  let (cH1n, nH1n) ← mnv4Bn B 960 7 mode epsStr "%h1g" "%h1bt" "h1n" nH1c
  let (cH1r, nH1r) ← pretty B (.batchOp (N := B) (.relu (n := 960*7*7)) (.operand nH1n zH17))
  let (cHc, nHc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := 960) (oc := 1280) (h := 7) (w := 7) zrnd "%hW" "%zb1280" zHk z1280 else .conv (ic := 960) (oc := 1280) (h := 7) (w := 7) "%hW" "%zb1280" zHk z1280) (.operand nH1r zH17))
  let (cHn, nHn) ← mnv4Bn B 1280 7 mode epsStr "%hg" "%hbt" "hn" nHc
  let (cHr, nHr) ← pretty B (.batchOp (N := B) (.relu (n := 1280*7*7)) (.operand nHn zH7))
  let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 1280) (h := 7) (w := 7))
    (.operand nHr zH7))
  let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC)
    (.operand nGap z1280b))

  pure { code := cStc ++ cStn ++ cStr ++ f0.code ++ bcode ++
                 cH1c ++ cH1n ++ cH1r ++ cHc ++ cHn ++ cHr ++ cGap ++ cLog,
         logits := nLog, stc := nStc, stn := nStn, str := nStr,
         f0 := f0, blocks := blocks, inputs := inputs,
         h1c := nH1c, h1n := nH1n, h1r := nH1r,
         hc := nHc, hn := nHn, hr := nHr, gap := nGap, last := cur }

/-- Every distinct channel width a bias-free conv in this net binds `%zb{c}` at: the stem, the fused
    stage's two convs, each block's pre-DW (`ic`), expand and post-DW (`mid`) and project (`oc`), and
    the head. Derived by hand and pinned by `mnv4-fwd-smoke`, which fails on an unbound `%zb`. -/
def mnv4ZbWidths : List Nat :=
  [32, 48, 80, 128, 160, 192, 256, 320, 480, 512, 640, 960, 1024, 1280]

/-- **`@mnv4_fwd`** — the MobileNetV4-Conv-M forward as one MLIR module.

    `%x` plus `mnv4SigList`'s parameters in `VLayer.toSpecs` order, logits `[B, nClasses]`. Every
    conv is bias-free, so the proven conv ops' bias operands bind to the `%zb{c}` zero constants the
    prelude declares — same op, `bias = 0`, and `x + 0.0` is exact (§2l step B).

    ⚠ `zeroBiasPrelude` must cover every width the chain references or the module has an unbound
    SSA name. That is a link error at parse time rather than a wrong number, which is the good
    direction, but `mnv4-fwd-smoke` checks it anyway so the failure arrives at `lake build` and not
    at `iree-compile`. -/
def mnv4FwdFaithfulV (B nClasses : Nat) (epsStr : String)
    (slug : String := "mnv4") (vSuffix : String := "") : String :=
  let sigList := mnv4SigList nClasses
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let r := (mnv4FwdChainB B nClasses epsStr .train).run' 0
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd{vSuffix}({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── MobileNetV4-Conv-M forward: every line is pretty(verified AST node) ──\n" ++
  zeroBiasPrelude false mnv4ZbWidths ++ r.code ++
  s!"    return {r.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

set_option maxRecDepth 4000000 in
/-- **`@mnv4_fwd_eval`** — the inference forward, every BN site reading frozen running stats.
    `%x` + 158 params + 104 stat inputs = **263 inputs**. This is what the driver scores through.

    ⭐ It is `mnv4FwdChainB` at `.eval` — the SAME traversal `@mnv4_fwd` and the train step use, so
    its BN order matches `mnv4StatSigList` by construction rather than by a second reading. -/
def mnv4FwdEvalFaithfulV (B nClasses : Nat) (epsStr : String)
    (slug : String := "mnv4") (vSuffix : String := "") : String :=
  let sigList := mnv4SigList nClasses ++ mnv4StatSigList
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let r := (mnv4FwdChainB B nClasses epsStr .eval).run' 0
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd_eval{vSuffix}({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── MobileNetV4-Conv-M eval forward (running-stats BN): every line is pretty(AST node) ──\n" ++
  zeroBiasPrelude false mnv4ZbWidths ++ r.code ++
  s!"    return {r.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § Block backward + un-fused parameter gradients
--   (project → post-DW? → expand → pre-DW?; `dy` flows straight into the project-BN backward,
--    because the UIB bottleneck is LINEAR — no activation after project, none after the skip add)
-- ════════════════════════════════════════════════════════════════

/-- A trainable parameter: emitted name (no `%`), its un-fused gradient SSA name, and its shape.
    The AdamW tail is a fold over this list, so the θ/m/v output order cannot drift from the
    signature order. (`MobileNetV2RenderB`/`ResNet34RenderB` carry same-shaped peers.) -/
private structure PGradV4 where
  nm   : String
  grad : String
  ds   : List Nat
deriving Inhabited

/-- Backward result: code, the dx cotangent to the previous block, and the block's parameter
    gradients in func-arg order. -/
private structure UibBackB where
  code : String
  dx : String
  ps : List PGradV4

/-- Pair a block's `uibSig`/`fusedSig` slice with the gradient SSA names its backward produced, in
    the same order.

    ⭐ **ONE source for the names AND the shapes.** The peers hand-write the parameter list twice —
    once in the signature, once in the backward's `ps` — and rely on eyes to keep them in step. Here
    the list the func header is built from is literally the list the AdamW tail folds over, so a
    parameter cannot be one shape in the signature and another in the optimizer. The `k = 0`
    dispatch is then written once, in `uibSig`, instead of once per family per direction. -/
private def zipPs (sig : List (String × List Nat)) (grads : List String) : List PGradV4 :=
  -- `nm` is the emitted name WITHOUT the leading `%`, because the AdamW tail spells `%{nm}m` /
  -- `%{nm}v` off it. (`String.drop` returns a `String.Slice` on this toolchain.)
  (sig.zip grads).map (fun ((n, ds), g) => ⟨String.ofList n.toList.tail, g, ds⟩)

/-- **STRIDE-1 UIB backward + its 6/9/12 un-fused gradients** — all four families, `ic = oc = c`.
    The skip is linear, so `dy` reaches the project BN unchanged AND fans back in at the block dx.

    The two `if`s are the same `k = 0` dispatch the forward used. ⚠ A backward that omits the
    *other* depthwise from the one the forward emitted still type-checks (both positions are
    shape-preserving at stride 1) and still descends — it computes the gradient of a different net.
    That is §3's trap arriving in the backward, so the forward record `f` is read for which
    positions exist (`f.qn`/`f.dn` are `""` when absent) rather than re-deriving it. -/
private def uibBackSkipGradB (B c expand preDWk postDWk h : Nat)
    (epsStr p xName : String) (f : UibFwdB) (dyName : String)
    (bf16 : Bool := false) : StateM Nat UibBackB := do
  let mid := c * expand
  -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are: the render produces TEXT
  -- and `skel` erases every ℝ payload before a token is emitted.
  let zrnd : ℝ → ℝ := fun r => r
  let zc   : Vec c := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zqk  : DepthwiseKernel c preDWk preDWk := fun _ _ _ => 0
  let zdk  : DepthwiseKernel mid postDWk postDWk := fun _ _ _ => 0
  let zke  : Kernel4 mid c 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 c mid 1 1 := fun _ _ _ _ => 0
  let zcb  : Vec (B*(c*h*h)) := fun _ => 0
  let zcp  : Vec (B*(c*(h*h))) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0
  let zmp  : Vec (B*(mid*(h*h))) := fun _ => 0
  -- ── project 1×1 + BN ──
  let pIn := if postDWk > 0 then f.dr else f.er
  let (cPn, nPn) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := h) (w := h)
    s!"%u{p}pg" f.pc epsStr 0 zc zcp (.operand dyName zcp))
  let (cPW, nPW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := mid) (oc := c) (h := h) (w := h) zrnd
    pIn zc zmb zkp (.operand nPn zcb) else .convWeightGradB (N := B) (ic := mid) (oc := c) (h := h) (w := h)
    pIn zc zmb zkp (.operand nPn zcb))
  let (cPg, nPg) ← pretty B (.bnGammaGradB (N := B) (oc := c) (h := h) (w := h)
    f.pc epsStr 0 zcp (.operand dyName zcp))
  let (cPt, nPt) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := h) (w := h)
    (.operand dyName zcp))
  let (cPx, nPx) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := c) (h := h) (w := h) zrnd
    s!"%u{p}pW" zkp zc (.operand nPn zcb) else .convBackBatched (N := B) (ic := mid) (oc := c) (h := h) (w := h)
    s!"%u{p}pW" zkp zc (.operand nPn zcb))
  let mut code := cPn ++ cPW ++ cPg ++ cPt ++ cPx
  let mut cur := nPx
  -- ── post-DW (present iff postDWk > 0), stride 1 on `mid` channels ──
  let mut dGrads : List String := []
  if postDWk > 0 then
    let (c1, n1) ← pretty B (.selectPosB f.dn zmb (.operand cur zmb))
    let (c2, n2) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := h) (w := h)
      s!"%u{p}dg" f.dc epsStr 0 zm zmp (.operand n1 zmp))
    let (c3, n3) ← pretty B (if bf16 then .depthwiseWeightGradBBf16 (N := B) (c := mid) (h := h) (w := h) zrnd
      f.er zm zmb zdk (.operand n2 zmb) else .depthwiseWeightGradB (N := B) (c := mid) (h := h) (w := h)
      f.er zm zmb zdk (.operand n2 zmb))
    let (c4, n4) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := h) (w := h)
      f.dc epsStr 0 zmp (.operand n1 zmp))
    let (c5, n5) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := h) (w := h)
      (.operand n1 zmp))
    let (c6, n6) ← pretty B (if bf16 then .depthwiseBackBatchedBf16 (N := B) (c := mid) (h := h) (w := h) zrnd
      s!"%u{p}dW" zdk zm (.operand n2 zmb) else .depthwiseBackBatched (N := B) (c := mid) (h := h) (w := h)
      s!"%u{p}dW" zdk zm (.operand n2 zmb))
    code := code ++ c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6
    dGrads := [n3, n4, n5]
    cur := n6
  -- ── expand 1×1 (c → mid) + BN ──
  let eIn := if preDWk > 0 then f.qr else xName
  let (cEm, nEm) ← pretty B (.selectPosB f.en zmb (.operand cur zmb))
  let (cEn, nEn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := h) (w := h)
    s!"%u{p}eg" f.ec epsStr 0 zm zmp (.operand nEm zmp))
  let (cEW, nEW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := c) (oc := mid) (h := h) (w := h) zrnd
    eIn zm zcb zke (.operand nEn zmb) else .convWeightGradB (N := B) (ic := c) (oc := mid) (h := h) (w := h)
    eIn zm zcb zke (.operand nEn zmb))
  let (cEg, nEg) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := h) (w := h)
    f.ec epsStr 0 zmp (.operand nEm zmp))
  let (cEt, nEt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := h) (w := h)
    (.operand nEm zmp))
  let (cEx, nEx) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := c) (oc := mid) (h := h) (w := h) zrnd
    s!"%u{p}eW" zke zm (.operand nEn zmb) else .convBackBatched (N := B) (ic := c) (oc := mid) (h := h) (w := h)
    s!"%u{p}eW" zke zm (.operand nEn zmb))
  code := code ++ cEm ++ cEn ++ cEW ++ cEg ++ cEt ++ cEx
  cur := nEx
  -- ── pre-DW (present iff preDWk > 0), stride 1 on `c` channels, reading the BLOCK INPUT ──
  let mut qGrads : List String := []
  if preDWk > 0 then
    let (c1, n1) ← pretty B (.selectPosB f.qn zcb (.operand cur zcb))
    let (c2, n2) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := h) (w := h)
      s!"%u{p}qg" f.qc epsStr 0 zc zcp (.operand n1 zcp))
    let (c3, n3) ← pretty B (if bf16 then .depthwiseWeightGradBBf16 (N := B) (c := c) (h := h) (w := h) zrnd
      xName zc zcb zqk (.operand n2 zcb) else .depthwiseWeightGradB (N := B) (c := c) (h := h) (w := h)
      xName zc zcb zqk (.operand n2 zcb))
    let (c4, n4) ← pretty B (.bnGammaGradB (N := B) (oc := c) (h := h) (w := h)
      f.qc epsStr 0 zcp (.operand n1 zcp))
    let (c5, n5) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := h) (w := h)
      (.operand n1 zcp))
    let (c6, n6) ← pretty B (if bf16 then .depthwiseBackBatchedBf16 (N := B) (c := c) (h := h) (w := h) zrnd
      s!"%u{p}qW" zqk zc (.operand n2 zcb) else .depthwiseBackBatched (N := B) (c := c) (h := h) (w := h)
      s!"%u{p}qW" zqk zc (.operand n2 zcb))
    code := code ++ c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6
    qGrads := [n3, n4, n5]
    cur := n6
  -- ── skip fan-in: (body dx) + dy, at the block-input shape ──
  let (cDx, nDx) ← pretty B (.addVB (.operand cur zcb) (.operand dyName zcb))
  pure { code := code ++ cDx, dx := nDx,
         ps := zipPs (uibSig p c c expand preDWk postDWk)
                 (qGrads ++ [nEW, nEg, nEt] ++ dGrads ++ [nPW, nPg, nPt]) }

/-- **STRIDE-2 UIB backward where the PRE-DW carries the stride.** The pre-DW is always present
    here, and it is the only op whose input-VJP crosses the resolution change — so the block's dx
    lands at `2h×2h` and everything upstream of the expand runs at `h×h`. No skip (`ic ≠ oc`). -/
private def uibBackPreStridedGradB (B ic oc expand preDWk postDWk h : Nat)
    (epsStr p xName : String) (f : UibFwdB) (dyName : String)
    (bf16 : Bool := false) : StateM Nat UibBackB := do
  let mid := ic * expand
  -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are: the render produces TEXT
  -- and `skel` erases every ℝ payload before a token is emitted.
  let zrnd : ℝ → ℝ := fun r => r
  let zic  : Vec ic := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zqk  : DepthwiseKernel ic preDWk preDWk := fun _ _ _ => 0
  let zdk  : DepthwiseKernel mid postDWk postDWk := fun _ _ _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zin  : Vec (B*(ic*(2*h)*(2*h))) := fun _ => 0
  let zqb  : Vec (B*(ic*h*h)) := fun _ => 0
  let zqp  : Vec (B*(ic*(h*h))) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0
  let zmp  : Vec (B*(mid*(h*h))) := fun _ => 0
  let zob  : Vec (B*(oc*h*h)) := fun _ => 0
  let zop  : Vec (B*(oc*(h*h))) := fun _ => 0
  let pIn := if postDWk > 0 then f.dr else f.er
  let (cPn, nPn) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := h) (w := h)
    s!"%u{p}pg" f.pc epsStr 0 zoc zop (.operand dyName zop))
  let (cPW, nPW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := mid) (oc := oc) (h := h) (w := h) zrnd
    pIn zoc zmb zkp (.operand nPn zob) else .convWeightGradB (N := B) (ic := mid) (oc := oc) (h := h) (w := h)
    pIn zoc zmb zkp (.operand nPn zob))
  let (cPg, nPg) ← pretty B (.bnGammaGradB (N := B) (oc := oc) (h := h) (w := h)
    f.pc epsStr 0 zop (.operand dyName zop))
  let (cPt, nPt) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := h) (w := h)
    (.operand dyName zop))
  let (cPx, nPx) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := oc) (h := h) (w := h) zrnd
    s!"%u{p}pW" zkp zoc (.operand nPn zob) else .convBackBatched (N := B) (ic := mid) (oc := oc) (h := h) (w := h)
    s!"%u{p}pW" zkp zoc (.operand nPn zob))
  let mut code := cPn ++ cPW ++ cPg ++ cPt ++ cPx
  let mut cur := nPx
  let mut dGrads : List String := []
  if postDWk > 0 then
    let (c1, n1) ← pretty B (.selectPosB f.dn zmb (.operand cur zmb))
    let (c2, n2) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := h) (w := h)
      s!"%u{p}dg" f.dc epsStr 0 zm zmp (.operand n1 zmp))
    let (c3, n3) ← pretty B (if bf16 then .depthwiseWeightGradBBf16 (N := B) (c := mid) (h := h) (w := h) zrnd
      f.er zm zmb zdk (.operand n2 zmb) else .depthwiseWeightGradB (N := B) (c := mid) (h := h) (w := h)
      f.er zm zmb zdk (.operand n2 zmb))
    let (c4, n4) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := h) (w := h)
      f.dc epsStr 0 zmp (.operand n1 zmp))
    let (c5, n5) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := h) (w := h)
      (.operand n1 zmp))
    let (c6, n6) ← pretty B (if bf16 then .depthwiseBackBatchedBf16 (N := B) (c := mid) (h := h) (w := h) zrnd
      s!"%u{p}dW" zdk zm (.operand n2 zmb) else .depthwiseBackBatched (N := B) (c := mid) (h := h) (w := h)
      s!"%u{p}dW" zdk zm (.operand n2 zmb))
    code := code ++ c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6
    dGrads := [n3, n4, n5]
    cur := n6
  -- expand 1×1 (ic → mid) at h; its input is the pre-DW's relu output
  let (cEm, nEm) ← pretty B (.selectPosB f.en zmb (.operand cur zmb))
  let (cEn, nEn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := h) (w := h)
    s!"%u{p}eg" f.ec epsStr 0 zm zmp (.operand nEm zmp))
  let (cEW, nEW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := ic) (oc := mid) (h := h) (w := h) zrnd
    f.qr zm zqb zke (.operand nEn zmb) else .convWeightGradB (N := B) (ic := ic) (oc := mid) (h := h) (w := h)
    f.qr zm zqb zke (.operand nEn zmb))
  let (cEg, nEg) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := h) (w := h)
    f.ec epsStr 0 zmp (.operand nEm zmp))
  let (cEt, nEt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := h) (w := h)
    (.operand nEm zmp))
  let (cEx, nEx) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := ic) (oc := mid) (h := h) (w := h) zrnd
    s!"%u{p}eW" zke zm (.operand nEn zmb) else .convBackBatched (N := B) (ic := ic) (oc := mid) (h := h) (w := h)
    s!"%u{p}eW" zke zm (.operand nEn zmb))
  -- pre-DW: STRIDED, so its input-VJP is the one that upsamples h → 2h
  let (cQm, nQm) ← pretty B (.selectPosB f.qn zqb (.operand nEx zqb))
  let (cQn, nQn) ← pretty B (.bnBatchBack (N := B) (oc := ic) (h := h) (w := h)
    s!"%u{p}qg" f.qc epsStr 0 zic zqp (.operand nQm zqp))
  let (cQW, nQW) ← pretty B (if bf16 then .depthwiseStridedWeightGradBBf16 (N := B) (c := ic) (h := h) (w := h) zrnd
    xName zic zin zqk (.operand nQn zqb) else .depthwiseStridedWeightGradB (N := B) (c := ic) (h := h) (w := h)
    xName zic zin zqk (.operand nQn zqb))
  let (cQg, nQg) ← pretty B (.bnGammaGradB (N := B) (oc := ic) (h := h) (w := h)
    f.qc epsStr 0 zqp (.operand nQm zqp))
  let (cQt, nQt) ← pretty B (.bnBetaGradB (N := B) (oc := ic) (h := h) (w := h)
    (.operand nQm zqp))
  let (cQx, nQx) ← pretty B (if bf16 then .depthwiseStridedBackBatchedBf16 (N := B) (c := ic) (h := h) (w := h) zrnd
    s!"%u{p}qW" zqk zic (.operand nQn zqb) else .depthwiseStridedBackBatched (N := B) (c := ic) (h := h) (w := h)
    s!"%u{p}qW" zqk zic (.operand nQn zqb))
  pure { code := code ++ cEm ++ cEn ++ cEW ++ cEg ++ cEt ++ cEx ++
                 cQm ++ cQn ++ cQW ++ cQg ++ cQt ++ cQx,
         dx := nQx,
         ps := zipPs (uibSig p ic oc expand preDWk postDWk)
                 ([nQW, nQg, nQt] ++ [nEW, nEg, nEt] ++ dGrads ++ [nPW, nPg, nPt]) }

/-- **STRIDE-2 UIB backward where the POST-DW carries the stride** (`preDWk = 0`) — the IB/MBConv
    family at a downsample. The expand runs at the INPUT resolution `2h×2h`, so its weight gradient
    contracts against `%x` at `2h` while the project's runs at `h`. No skip (`ic ≠ oc`). -/
private def uibBackPostStridedGradB (B ic oc expand postDWk h : Nat)
    (epsStr p xName : String) (f : UibFwdB) (dyName : String)
    (bf16 : Bool := false) : StateM Nat UibBackB := do
  let mid := ic * expand
  -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are: the render produces TEXT
  -- and `skel` erases every ℝ payload before a token is emitted.
  let zrnd : ℝ → ℝ := fun r => r
  let zoc  : Vec oc := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zdk  : DepthwiseKernel mid postDWk postDWk := fun _ _ _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zin  : Vec (B*(ic*(2*h)*(2*h))) := fun _ => 0
  let zeb  : Vec (B*(mid*(2*h)*(2*h))) := fun _ => 0
  let zep  : Vec (B*(mid*((2*h)*(2*h)))) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0
  let zmp  : Vec (B*(mid*(h*h))) := fun _ => 0
  let zob  : Vec (B*(oc*h*h)) := fun _ => 0
  let zop  : Vec (B*(oc*(h*h))) := fun _ => 0
  let (cPn, nPn) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := h) (w := h)
    s!"%u{p}pg" f.pc epsStr 0 zoc zop (.operand dyName zop))
  let (cPW, nPW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := mid) (oc := oc) (h := h) (w := h) zrnd
    f.dr zoc zmb zkp (.operand nPn zob) else .convWeightGradB (N := B) (ic := mid) (oc := oc) (h := h) (w := h)
    f.dr zoc zmb zkp (.operand nPn zob))
  let (cPg, nPg) ← pretty B (.bnGammaGradB (N := B) (oc := oc) (h := h) (w := h)
    f.pc epsStr 0 zop (.operand dyName zop))
  let (cPt, nPt) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := h) (w := h)
    (.operand dyName zop))
  let (cPx, nPx) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := oc) (h := h) (w := h) zrnd
    s!"%u{p}pW" zkp zoc (.operand nPn zob) else .convBackBatched (N := B) (ic := mid) (oc := oc) (h := h) (w := h)
    s!"%u{p}pW" zkp zoc (.operand nPn zob))
  -- post-DW: STRIDED, so its input-VJP is the one that upsamples h → 2h
  let (cDm, nDm) ← pretty B (.selectPosB f.dn zmb (.operand nPx zmb))
  let (cDn, nDn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := h) (w := h)
    s!"%u{p}dg" f.dc epsStr 0 zm zmp (.operand nDm zmp))
  let (cDW, nDW) ← pretty B (if bf16 then .depthwiseStridedWeightGradBBf16 (N := B) (c := mid) (h := h) (w := h) zrnd
    f.er zm zeb zdk (.operand nDn zmb) else .depthwiseStridedWeightGradB (N := B) (c := mid) (h := h) (w := h)
    f.er zm zeb zdk (.operand nDn zmb))
  let (cDg, nDg) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := h) (w := h)
    f.dc epsStr 0 zmp (.operand nDm zmp))
  let (cDt, nDt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := h) (w := h)
    (.operand nDm zmp))
  let (cDx, nDx) ← pretty B (if bf16 then .depthwiseStridedBackBatchedBf16 (N := B) (c := mid) (h := h) (w := h) zrnd
    s!"%u{p}dW" zdk zm (.operand nDn zmb) else .depthwiseStridedBackBatched (N := B) (c := mid) (h := h) (w := h)
    s!"%u{p}dW" zdk zm (.operand nDn zmb))
  -- expand 1×1 (ic → mid) at the INPUT resolution 2h
  let (cEm, nEm) ← pretty B (.selectPosB f.en zeb (.operand nDx zeb))
  let (cEn, nEn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := 2*h) (w := 2*h)
    s!"%u{p}eg" f.ec epsStr 0 zm zep (.operand nEm zep))
  let (cEW, nEW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := ic) (oc := mid) (h := 2*h) (w := 2*h) zrnd
    xName zm zin zke (.operand nEn zeb) else .convWeightGradB (N := B) (ic := ic) (oc := mid) (h := 2*h) (w := 2*h)
    xName zm zin zke (.operand nEn zeb))
  let (cEg, nEg) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := 2*h) (w := 2*h)
    f.ec epsStr 0 zep (.operand nEm zep))
  let (cEt, nEt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := 2*h) (w := 2*h)
    (.operand nEm zep))
  let (cEx, nEx) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := ic) (oc := mid) (h := 2*h) (w := 2*h) zrnd
    s!"%u{p}eW" zke zm (.operand nEn zeb) else .convBackBatched (N := B) (ic := ic) (oc := mid) (h := 2*h) (w := 2*h)
    s!"%u{p}eW" zke zm (.operand nEn zeb))
  pure { code := cPn ++ cPW ++ cPg ++ cPt ++ cPx ++ cDm ++ cDn ++ cDW ++ cDg ++ cDt ++ cDx ++
                 cEm ++ cEn ++ cEW ++ cEg ++ cEt ++ cEx,
         dx := nEx,
         ps := zipPs (uibSig p ic oc expand 0 postDWk)
                 ([nEW, nEg, nEt] ++ [nDW, nDg, nDt] ++ [nPW, nPg, nPt]) }

/-- **Fused inverted-bottleneck backward, stride 2** — MobileNetV4's stage 0.

    ⚠⚠ **`swishBackB`, NOT `selectPosB`.** The forward is swish here and relu twenty lines up
    (`fusedMbConvFwdStridedB` records why: the reference that produced 84.58% uses swish at this
    site). A relu mask against a swish forward type-checks, has the right shape, and descends —
    it is the same silent-wrong-gradient class as the pre/post-DW swap, in the activation. -/
private def fusedMbConvBackStridedGradB (B ic oc expand k h : Nat)
    (epsStr p xName : String) (f : UibFwdB) (dyName : String)
    (bf16 : Bool := false) : StateM Nat UibBackB := do
  let mid := if expand == 1 then oc else ic * expand
  -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are: the render produces TEXT
  -- and `skel` erases every ℝ payload before a token is emitted.
  let zrnd : ℝ → ℝ := fun r => r
  let zoc  : Vec oc := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zkf  : Kernel4 mid ic k k := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zin  : Vec (B*(ic*(2*h)*(2*h))) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0
  let zmp  : Vec (B*(mid*(h*h))) := fun _ => 0
  let zob  : Vec (B*(oc*h*h)) := fun _ => 0
  let zop  : Vec (B*(oc*(h*h))) := fun _ => 0
  let (cPn, nPn) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := h) (w := h)
    s!"%f{p}pg" f.pc epsStr 0 zoc zop (.operand dyName zop))
  let (cPW, nPW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := mid) (oc := oc) (h := h) (w := h) zrnd
    f.er zoc zmb zkp (.operand nPn zob) else .convWeightGradB (N := B) (ic := mid) (oc := oc) (h := h) (w := h)
    f.er zoc zmb zkp (.operand nPn zob))
  let (cPg, nPg) ← pretty B (.bnGammaGradB (N := B) (oc := oc) (h := h) (w := h)
    f.pc epsStr 0 zop (.operand dyName zop))
  let (cPt, nPt) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := h) (w := h)
    (.operand dyName zop))
  let (cPx, nPx) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := oc) (h := h) (w := h) zrnd
    s!"%f{p}pW" zkp zoc (.operand nPn zob) else .convBackBatched (N := B) (ic := mid) (oc := oc) (h := h) (w := h)
    s!"%f{p}pW" zkp zoc (.operand nPn zob))
  let (cSw, nSw) ← pretty B (.swishBackB f.en zmb (.operand nPx zmb))
  let (cCn, nCn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := h) (w := h)
    s!"%f{p}cg" f.ec epsStr 0 zm zmp (.operand nSw zmp))
  let (cCW, nCW) ← pretty B (if bf16 then .convStridedWeightGradBBf16 (N := B) (ic := ic) (oc := mid)
    (h := h) (w := h) zrnd xName zm zin zkf (.operand nCn zmb) else .convStridedWeightGradB (N := B) (ic := ic) (oc := mid)
    (h := h) (w := h) xName zm zin zkf (.operand nCn zmb))
  let (cCg, nCg) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := h) (w := h)
    f.ec epsStr 0 zmp (.operand nSw zmp))
  let (cCt, nCt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := h) (w := h)
    (.operand nSw zmp))
  let (cCx, nCx) ← pretty B (if bf16 then .convStridedBackBatchedBf16 (N := B) (ic := ic) (oc := mid)
    (h := h) (w := h) zrnd s!"%f{p}cW" zkf zm (.operand nCn zmb) else .convStridedBackBatched (N := B) (ic := ic) (oc := mid)
    (h := h) (w := h) s!"%f{p}cW" zkf zm (.operand nCn zmb))
  pure { code := cPn ++ cPW ++ cPg ++ cPt ++ cPx ++ cSw ++ cCn ++ cCW ++ cCg ++ cCt ++ cCx,
         dx := nCx,
         ps := zipPs (fusedSig p ic oc expand k) [nCW, nCg, nCt, nPW, nPg, nPt] }

/-- **Backward dispatch — the SAME `if`s as `uibFwdDispatch`, on the SAME row.**

    ⚠⚠ That the two dispatchers read one `UibSpec` is the point. Differentiating a block as a
    different family than the forward emitted it type-checks at stride 1 (both depthwise positions
    are shape-preserving), produces the right shapes, trains and descends — and computes the
    gradient of a net nobody built. With one table and one row per call there is no second place
    for the dispatch to be written down differently. -/
private def uibBackDispatch (B : Nat) (b : UibSpec) (epsStr xName : String)
    (f : UibFwdB) (dyName : String) (bf16 : Bool := false) : StateM Nat UibBackB :=
  if b.stride2 then
    if b.preDWk > 0 then
      uibBackPreStridedGradB B b.ic b.oc b.expand b.preDWk b.postDWk b.h epsStr b.p xName f dyName bf16
    else
      uibBackPostStridedGradB B b.ic b.oc b.expand b.postDWk b.h epsStr b.p xName f dyName bf16
  else
    uibBackSkipGradB B b.ic b.expand b.preDWk b.postDWk b.h epsStr b.p xName f dyName bf16

-- ════════════════════════════════════════════════════════════════
-- § The AdamW tail — one proven triple per parameter, folded in signature order
-- ════════════════════════════════════════════════════════════════

/-- `(θ', m', v')` for one parameter, from its un-fused gradient: the proven
    `adamMNextF`/`adamVNextF`/`adamWParamF` triple (`adamW_triple_faithful` bundles their `den`s
    into `Proofs.adamWStep` by `rfl`). β₁/β₂/ε/wd are baked; `%lr`/`%bc1`/`%bc2` are runtime
    `tensor<f32>` args, so one render serves a whole LR schedule.

    At `replicas > 1` the gradient is first averaged by `ViTRender.emitGradAllReduce`. **That
    collective is a TRUSTED CARVE-OUT** — emitted text, not `pretty` of an AST node, so outside
    every faithfulness theorem here; the AdamW triple consumes the averaged gradient as an
    `.operand` exactly as it consumed the raw one, so the `den` side does not shift. At
    `replicas ≤ 1` it emits nothing and the single-device render stays byte-identical. -/
private def adamOne4 (B : Nat) (replicas : Nat) (g : PGradV4) :
    StateM Nat (String × String × String × String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let (arS, gAvg) := ViTRender.emitGradAllReduce g.grad g.ds g.nm replicas
  let gr : SHlo n := .operand gAvg z
  let (cM, nM) ← pretty B (.adamMNextF s!"%{g.nm}m" "%b1" "%ob1" g.ds 0 z gr)
  let (cV, nV) ← pretty B (.adamVNextF s!"%{g.nm}v" "%b2" "%ob2" g.ds 0 z gr)
  let (cT, nT) ← pretty B (.adamWParamF s!"%{g.nm}" s!"%{g.nm}m" s!"%{g.nm}v" "%b1" "%ob1"
                    "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" "%wd" g.ds 0 0 0 0 0 0 0 z z z gr)
  pure (arS ++ cM ++ cV ++ cT, nT, nM, nV)

/-- β₁/β₂/ε/wd as graph constants — the Imagenette AdamW recipe, identical to MobileNetV2's and
    ResNet-34's, because MNv4 sits in the same tier and runs the same schedule. -/
private def adamConsts4 : String :=
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"

/-- The driver's **variant slug** for a given `(B, replicas)`: the artifact is
    `verified_mlir/mnv4_<variant>_train_step.mlir`, the entry point is
    `@mnv4_<variant>_train_step`, and `LEAN_MLIR_VARIANT` selects it. All three must agree — the
    shim checks the entry name and refuses a mismatch outright rather than running the wrong graph.
    `B = 32` is deliberately unsuffixed so the Imagenette artifact keeps a stable name. -/
def mnv4AdamVariant (B replicas : Nat)
    -- ⚠⚠ `bf16` MUST reach here and not merely the block renderers: the entry NAME derives from
    -- this, so a flag that reaches the emission but not the name writes `…bf16_train_step.mlir`
    -- declaring `@…_train_step` inside and the driver refuses at load ("entry mismatch").
    (bf16 : Bool := false) : String :=
  (if replicas ≤ 1 then "adam" else "adamdp") ++ (if B == 32 then "" else toString B) ++
  (if bf16 then "bf16" else "")

-- ════════════════════════════════════════════════════════════════
-- § The whole-net batched AdamW train step
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000000 in
/-- **MobileNetV4-Conv-M AdamW train step, batch BN, rendered from the verified AST at `N := B`.**

    583 inputs (`%x`, 158 θ, 158 m, 158 v, `%lr`/`%bc1`/`%bc2`, 104 running-stat slots, `%onehot`)
    and 581 outputs (158 θ', 158 m', 158 v', `%loss`/`%bc1`/`%bc2`, 104 batch stats). Parameter
    order comes from `mnv4ShapeList` — through `zipPs`, which builds each block's gradient list from
    the very same `uibSig` slice the signature does — and stat order from `mnv4StatShapeList`.

    Forward: stem 3×3/s2 XLA-`SAME` (3→32, 224→112) → fused MBConv (32→48, 112→56, **swish**) →
    the 14 UIB blocks (three stride-2 downsamples, eleven identity skips, all four families) →
    1×1 conv-BN-relu head (256→1280) → GAP → dense.

    The cotangent is composed from kit ops (`softmaxRow → subB → scaleB → addVB → shiftB →
    divConstB`, α = 0.1, K = nClasses), and `%loss` is report-only and stays outside the AST — the
    same carve-out `resnet34`/`mobilenetv2` take. -/
def mobilenetv4AdamTrainStepFaithfulB (B nClasses : Nat) (epsStr : String)
    (replicas : Nat := 1) (slug : String := "mnv4")
    -- ⭐⭐ **bf16**, TRAILING and defaulted so every existing render is byte-identical (gate 1).
    -- MNv4's UIB blocks carry BOTH depthwise families: the stride-1 `depthwise` and the
    -- SYMMETRIC-pad `depthwiseStrided` — the latter is new here (MNv2 used the XLA-`SAME` one),
    -- and its three bf16 twins are the only ops this net needed that MobileNetV2 did not build.
    (bf16 : Bool := false) : String :=
  let alphaStr := fmt6 0.1
  let negAlphaKStr := "-" ++ alphaOverK nClasses 0.1
  let go : StateM Nat String := do
    -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are — see `uibFwdSkipB`.
    let zrnd : ℝ → ℝ := fun r => r
    -- ═══ forward: THE SHARED CHAIN, not a second copy ═══
    -- ⭐⭐ `@mnv4_fwd`, `@mnv4_fwd_eval` and this train step are all `mnv4FwdChainB`. The peers
    -- inline a second transcription of the block table into their train step and rely on eyes to
    -- keep the two in step; here there is only one, so the train/score divergence §3d(b) measured
    -- in MobileNetV2 — and that `regen_verified_mlir.sh check` reported green — cannot arise.
    let fwd ← mnv4FwdChainB B nClasses epsStr .train bf16
    let zx    : Vec (B*(3*224*224)) := fun _ => 0
    let zSk   : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let z32   : Vec 32 := fun _ => 0
    let z112  : Vec (B*(32*112*112)) := fun _ => 0
    let z112p : Vec (B*(32*(112*112))) := fun _ => 0
    let z7     : Vec (B*(256*7*7)) := fun _ => 0
    let zH1k   : Kernel4 960 256 1 1 := fun _ _ _ _ => 0
    let z960   : Vec 960 := fun _ => 0
    let zH17   : Vec (B*(960*7*7)) := fun _ => 0
    let zH17p  : Vec (B*(960*(7*7))) := fun _ => 0
    let zHk    : Kernel4 1280 960 1 1 := fun _ _ _ _ => 0
    let z1280  : Vec 1280 := fun _ => 0
    let zH7    : Vec (B*(1280*7*7)) := fun _ => 0
    let zH7p   : Vec (B*(1280*(7*7))) := fun _ => 0
    let z1280b : Vec (B*1280) := fun _ => 0
    let zWd    : Mat 1280 nClasses := fun _ _ => 0
    let zNCb   : Vec (B*(1*nClasses)) := fun _ => 0
    let zNCp   : Vec (B*nClasses) := fun _ => 0
    let nStc := fwd.stc; let nStn := fwd.stn
    let nH1c := fwd.h1c; let nH1n := fwd.h1n
    let nHc := fwd.hc; let nHn := fwd.hn; let nGap := fwd.gap; let nLog := fwd.logits
    -- ═══ label-smoothed softmax-CE cotangent, COMPOSED from kit ops (α = 0.1, K = nClasses):
    --     dy = (softmax(logits) − onehot + α·onehot − α/K) / B. Every line is a verified node. ═══
    let (cSm,  nSm)  ← pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses))
      (.operand nLog zNCb))
    let (cD0,  nD0)  ← pretty B (.subB (.operand nSm zNCb) (.operand "%onehot" zNCb))
    let (cLsa, nLsa) ← pretty B (.scaleB alphaStr 0 (.operand "%onehot" zNCb))
    let (cD1,  nD1)  ← pretty B (.addVB (.operand nD0 zNCb) (.operand nLsa zNCb))
    let (cD2,  nD2)  ← pretty B (.shiftB negAlphaKStr 0 (.operand nD1 zNCb))
    let (cDy,  nDy)  ← pretty B (.divConstB s!"{B}.0" 0 (.operand nD2 zNCb))
    -- ═══ head backward + the 8 head/dense gradients (bias-free convs ⇒ no `hb`).
    --     TWO conv-BN-relu stages now, unwound outermost-first: 960→1280, then 256→960. ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (N := B)
      (.denseRowBack (rows := 1) (a := 1280) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    let (cWdg, nWdg) ← pretty B (.denseWeightGradB (c := nClasses) nGap z1280b (.operand nDy zNCp))
    let (cbdg, nbdg) ← pretty B (.denseBiasGradB (N := B) (.operand nDy zNCp))
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 1280) (h := 7) (w := 7)
      (.operand nDgi z1280b))
    let (cDhm, nDhm) ← pretty B (.selectPosB nHn zH7 (.operand nDgp zH7))
    let (cDhn, nDhn) ← pretty B (.bnBatchBack (N := B) (oc := 1280) (h := 7) (w := 7)
      "%hg" nHc epsStr 0 z1280 zH7p (.operand nDhm zH7p))
    let (cDhx, nDhx) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := 960) (oc := 1280) (h := 7) (w := 7) zrnd
      "%hW" zHk z1280 (.operand nDhn zH7) else .convBackBatched (N := B) (ic := 960) (oc := 1280) (h := 7) (w := 7)
      "%hW" zHk z1280 (.operand nDhn zH7))
    let (cHW, nHW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := 960) (oc := 1280) (h := 7) (w := 7) zrnd
      fwd.h1r z1280 zH17 zHk (.operand nDhn zH7) else .convWeightGradB (N := B) (ic := 960) (oc := 1280) (h := 7) (w := 7)
      fwd.h1r z1280 zH17 zHk (.operand nDhn zH7))
    let (cHg, nHg) ← pretty B (.bnGammaGradB (N := B) (oc := 1280) (h := 7) (w := 7)
      nHc epsStr 0 zH7p (.operand nDhm zH7p))
    let (cHt, nHt) ← pretty B (.bnBetaGradB (N := B) (oc := 1280) (h := 7) (w := 7)
      (.operand nDhm zH7p))
    -- the first head conv's stage: relu mask at `h1n`, BN back, then input/weight grads at 256→960
    let (cDh1m, nDh1m) ← pretty B (.selectPosB nH1n zH17 (.operand nDhx zH17))
    let (cDh1n, nDh1n) ← pretty B (.bnBatchBack (N := B) (oc := 960) (h := 7) (w := 7)
      "%h1g" nH1c epsStr 0 z960 zH17p (.operand nDh1m zH17p))
    let (cDh1x, nDh1x) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := 256) (oc := 960) (h := 7) (w := 7) zrnd
      "%h1W" zH1k z960 (.operand nDh1n zH17) else .convBackBatched (N := B) (ic := 256) (oc := 960) (h := 7) (w := 7)
      "%h1W" zH1k z960 (.operand nDh1n zH17))
    let (cH1W, nH1W) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := 256) (oc := 960) (h := 7) (w := 7) zrnd
      fwd.last z960 z7 zH1k (.operand nDh1n zH17) else .convWeightGradB (N := B) (ic := 256) (oc := 960) (h := 7) (w := 7)
      fwd.last z960 z7 zH1k (.operand nDh1n zH17))
    let (cH1g, nH1g) ← pretty B (.bnGammaGradB (N := B) (oc := 960) (h := 7) (w := 7)
      nH1c epsStr 0 zH17p (.operand nDh1m zH17p))
    let (cH1t, nH1t) ← pretty B (.bnBetaGradB (N := B) (oc := 960) (h := 7) (w := 7)
      (.operand nDh1m zH17p))
    -- ═══ backward: ONE fold over `mnv4Blocks` reversed, dispatched by the same row the
    --     forward used, so a family can never be differentiated as a different family ═══
    let mut dy := nDh1x
    let mut gcode := ""
    let mut blockPs : List (List PGradV4) := []
    for (b, f, xin) in (mnv4Blocks.zip (fwd.blocks.zip fwd.inputs)).reverse.map
        (fun (b, f, xin) => (b, f, xin)) do
      let g ← uibBackDispatch B b epsStr xin f dy bf16
      gcode := gcode ++ g.code
      blockPs := [g.ps] ++ blockPs
      dy := g.dx
    let g0 ← fusedMbConvBackStridedGradB B 32 48 4 3 56 epsStr "0" fwd.str fwd.f0 dy bf16
    -- ═══ stem backward: relu mask → BN back, then the 3 stem gradients (NO conv-back past %x) ═══
    let (cDsm, nDsm) ← pretty B (.selectPosB nStn z112 (.operand g0.dx z112))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := B) (oc := 32) (h := 112) (w := 112)
      "%sg" nStc epsStr 0 z32 z112p (.operand nDsm z112p))
    let (csW, nsW) ← pretty B (if bf16 then .convStridedXlaWeightGradBBf16 zrnd "%x" z32 zx zSk (.operand nDsn z112) else .convStridedXlaWeightGradB "%x" z32 zx zSk (.operand nDsn z112))
    let (csg, nsg) ← pretty B (.bnGammaGradB (N := B) (oc := 32) (h := 112) (w := 112)
      nStc epsStr 0 z112p (.operand nDsm z112p))
    let (cst, nst) ← pretty B (.bnBetaGradB (N := B) (oc := 32) (h := 112) (w := 112)
      (.operand nDsm z112p))
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT.
    --     Derived from the SAME forward record that computes them (`f.qc`/`f.ec`/`f.dc`/`f.pc`)
    --     rather than from an independent 52-entry table — a misaligned stat slot is SILENT, since
    --     the arities still match and the wrong layer's statistics flow into the wrong
    --     `@mnv4_fwd_eval` slot. ═══
    let bnStat (oc hh : Nat) (xn : String) : StateM Nat (String × List String) := do
      let zb : Vec (B*(oc*(hh*hh))) := fun _ => 0
      let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
      let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
      pure (cM ++ cV, [nM, nV])
    -- One UIB block's stats in `uibStatSig` order. ⚠ `eh` is the EXPAND BN's spatial size, which
    -- is `2h` for the post-strided family alone (its expand runs before the downsample) and `h`
    -- everywhere else — the one place the three block shapes are not interchangeable here.
    let uibStats (b : UibSpec) (f : UibFwdB) : StateM Nat (String × List String) := do
      let mid := b.ic * b.expand
      -- the expand BN sits at the INPUT resolution `2h` for the post-strided family alone (its
      -- expand runs before the downsample); every other site is at the block's own `h`
      let eh := if b.stride2 && b.preDWk == 0 then 2 * b.h else b.h
      let mut code := ""
      let mut ns : List String := []
      if b.preDWk > 0 then
        let (c, n) ← bnStat b.ic b.h f.qc
        code := code ++ c; ns := ns ++ n
      let (ce, ne) ← bnStat mid eh f.ec
      code := code ++ ce; ns := ns ++ ne
      if b.postDWk > 0 then
        let (c, n) ← bnStat mid b.h f.dc
        code := code ++ c; ns := ns ++ n
      let (cp, np) ← bnStat b.oc b.h f.pc
      pure (code ++ cp, ns ++ np)
    let (cQs, qs) ← bnStat 32 112 nStc
    let (cQ0c, q0c) ← bnStat 128 56 fwd.f0.ec
    let (cQ0p, q0p) ← bnStat  48 56 fwd.f0.pc
    let mut qcode := ""
    let mut qnames : List String := []
    for (b, f) in mnv4Blocks.zip fwd.blocks do
      let (c, n) ← uibStats b f
      qcode := qcode ++ c; qnames := qnames ++ n
    let (cQh1, qh1) ← bnStat 960 7 nH1c
    let (cQh, qh) ← bnStat 1280 7 nHc
    -- ═══ the 158 parameter gradients in func-arg order ═══
    let stemPs : List PGradV4 :=
      [⟨"sW", nsW, [32,3,3,3]⟩, ⟨"sg", nsg, [32]⟩, ⟨"sbt", nst, [32]⟩]
    let headPs : List PGradV4 :=
      [⟨"h1W", nH1W, [960,256,1,1]⟩, ⟨"h1g", nH1g, [960]⟩, ⟨"h1bt", nH1t, [960]⟩,
       ⟨"hW", nHW, [1280,960,1,1]⟩, ⟨"hg", nHg, [1280]⟩, ⟨"hbt", nHt, [1280]⟩,
       ⟨"Wd", nWdg, [1280, nClasses]⟩, ⟨"bd", nbdg, [nClasses]⟩]
    let allPs : List PGradV4 := stemPs ++ g0.ps ++ blockPs.flatten ++ headPs
    -- ═══ AdamW: one proven triple per parameter ═══
    let mut adamCode := ""
    let mut thetaN : List String := []
    let mut mNames : List String := []
    let mut vNames : List String := []
    for g in allPs do
      let (c, nT, nM, nV) ← adamOne4 B replicas g
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]
      mNames := mNames ++ [nM]
      vNames := vNames ++ [nV]
    -- ═══ assemble ═══
    let statCode := cQs ++ cQ0c ++ cQ0p ++ qcode ++ cQh1 ++ cQh
    let statNames : List String := qs ++ q0c ++ q0p ++ qnames ++ qh1 ++ qh
    -- `%loss` is REPORT-ONLY: mean smoothed-CE for logging, on no gradient path. It is NOT
    -- `pretty` of an AST node and says so in the emitted text — the same carve-out
    -- `resnet34`/`mobilenetv2`'s `%loss` takes. The SMOOTHED cross-entropy, matching the
    -- cotangent's soft target; §2b shipped plain CE against a smoothed cotangent on R34 and only
    -- the numeric tie caught it.
    let lossCode :=
      "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
      s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %llog = stablehlo.log {nSm} : {ty [B, nClasses]}\n" ++
      s!"    %lohll = stablehlo.multiply %onehot, %llog : {ty [B, nClasses]}\n" ++
      s!"    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B, nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
      s!"    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B, nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
      s!"    %lomac = stablehlo.constant dense<{oneMinusAlpha 0.1}> : {ty [B]}\n" ++
      s!"    %laKc = stablehlo.constant dense<{alphaOverK nClasses 0.1}> : {ty [B]}\n" ++
      s!"    %llt1 = stablehlo.multiply %lomac, %lt1s : {ty [B]}\n" ++
      s!"    %llt2 = stablehlo.multiply %laKc, %llsr : {ty [B]}\n" ++
      s!"    %llpe = stablehlo.add %llt1, %llt2 : {ty [B]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [B]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
      s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
      s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
    let body := fwd.code ++
      cSm ++ cD0 ++ cLsa ++ cD1 ++ cD2 ++ cDy ++
      cDgi ++ cWdg ++ cbdg ++ cDgp ++ cDhm ++ cDhn ++ cDhx ++ cHW ++ cHg ++ cHt ++
      cDh1m ++ cDh1n ++ cDh1x ++ cH1W ++ cH1g ++ cH1t ++
      gcode ++ g0.code ++ cDsm ++ cDsn ++ csW ++ csg ++ cst ++ statCode
    let pTypes : List String := allPs.map (fun g => ty g.ds)
    let statTypes : List String := mnv4StatSigList.map (·.2)
    let retVals := thetaN ++ mNames ++ vNames ++ ["%loss", "%bc1", "%bc2"] ++ statNames
    let retTys  := pTypes ++ pTypes ++ pTypes ++
      ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statTypes
    pure <|
      (if replicas ≤ 1 then
        "    // ── MobileNetV4-Conv-M batch-BN AdamW train step: every line is pretty(AST node) ──\n"
       else
        s!"    // ── MobileNetV4-Conv-M batch-BN AdamW train step, DATA-PARALLEL over {replicas} replicas ──\n" ++
        "    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`\n" ++
        "    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT, emitted text outside\n" ++
        "    // the faithfulness theorems. NOTE this does NOT equal a single-device step at the\n" ++
        "    // global batch — BN normalises per replica, so N×b != 1×(N·b) by design.\n") ++
      zeroBiasPrelude false mnv4ZbWidths ++ body ++ adamConsts4 ++ adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let sigList : List (String × String) := mnv4SigList nClasses
  let pSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let mSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}m: {t}"))
  let vSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}v: {t}"))
  let statSig := String.intercalate ", " (mnv4StatSigList.map (fun (n, t) => s!"{n}i: {t}"))
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, " ++ statSig ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (·.2)
  let outSig := String.intercalate ", "
    (pTy ++ pTy ++ pTy ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++
     (mnv4StatSigList.map (·.2)))
  let inner : String := go.run' 0
  let fname := s!"{slug}_{mnv4AdamVariant B replicas bf16}_train_step"
  "module @m {\n" ++
  s!"  func.func @{fname}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The layout contract, checked at elaboration
-- ════════════════════════════════════════════════════════════════

-- 233 parameters and 154 BN stat slots (= 2 × 77 BN layers) ⇒ the train step is
-- 1 + 3×233 + 3 + 154 + 1 = 858 inputs and 3×233 + 3 + 154 = 856 outputs, and the eval forward is
-- 1 + 233 + 154 = 388 inputs. `mnv4-fwd-smoke` ties the 233 to `VLayer.toSpecs` shape-for-shape.
#guard (Proofs.StableHLO.mnv4ShapeList 10).length == 233
#guard Proofs.StableHLO.mnv4StatShapeList.length == 154

-- ⭐⭐ **THE STAT-ALIGNMENT GATE, and it is the strong one.** Every conv in this net is
-- BN-followed, so the BN stat slots must be, in order, two per conv weight at that conv's OUTPUT
-- channel count — which is the kernel's first dimension for a regular conv `[oc,ic,kH,kW]` AND for
-- a depthwise `[c,1,k,k]` alike. This pins the stat list's LENGTH, WIDTHS and ORDER against the
-- parameter list, so the two `k = 0` dispatches (`uibSig`'s and `uibStatSig`'s) cannot diverge.
--
-- ⚠ It is worth having as a `#guard` rather than a comment because a misaligned stat slot is
-- SILENT at run time: the arities still match, and the wrong layer's statistics simply flow into
-- the wrong `@mnv4_fwd_eval` slot. Nothing downstream would fail — the net would just score
-- slightly wrong, forever.
#guard (Proofs.StableHLO.mnv4StatShapeList.map (·.2)) ==
  ((Proofs.StableHLO.mnv4ShapeList 10).filter (fun p => p.2.length == 4)).flatMap
    (fun p => [[p.2.headD 0], [p.2.headD 0]])

-- The entry name, the artifact path and `LEAN_MLIR_VARIANT` must agree or the shim refuses the
-- call ("entry mismatch"). These pin the literal paths below against `mnv4AdamVariant`, so a
-- rename fails at `lake build` rather than at run time.
#guard Proofs.StableHLO.mnv4AdamVariant 32 1 == "adam"
#guard Proofs.StableHLO.mnv4AdamVariant 32 2 == "adamdp"
#guard Proofs.StableHLO.mnv4AdamVariant 64 1 == "adam64"

-- ── The three Imagenette artifacts. B=32, nClasses=10, ε=1e-5 — the tier's shape. ──────────────
-- ⭐ All three come from ONE chain (`mnv4FwdChainB`, switched by `BnMode`) and one block table, so
-- the train/score split that §3d(b) found in MobileNetV2 — a per-example forward paired with a
-- batch-BN Adam train step, green under `regen_verified_mlir.sh check` because that script only
-- ever pairs a forward with the SGD step — has nowhere to live here.

#eval IO.FS.writeFile "verified_mlir/mnv4_fwd.mlir"
  (Proofs.StableHLO.mnv4FwdFaithfulV 32 10 "1.0e-5")

#eval IO.FS.writeFile "verified_mlir/mnv4_fwd_eval.mlir"
  (Proofs.StableHLO.mnv4FwdEvalFaithfulV 32 10 "1.0e-5")

-- **This is the artifact the MNv4 Imagenette trainer runs**, and this `#eval` is its only writer.
-- Target: `RESULTS.md`'s 84.58%, the baseline path's number for this block table
-- (`planning/mnv4_verified.md` phase 4). ⚠ Unlike MobileNetV2's, that number belongs to the JAX
-- baseline and does NOT move when this render changes — the stem's `convStridedXla` was chosen so
-- the two are the same net (§3e), and the forward tie measured 1.423e-06 against it unpatched.
#eval IO.FS.writeFile "verified_mlir/mnv4_adam_train_step.mlir"
  (Proofs.StableHLO.mobilenetv4AdamTrainStepFaithfulB 32 10 "1.0e-5")

-- ── The 1000-class ImageNet artifacts, for `mnv4ImagenetVerified` (slug `mnv4in`). ─────────────
-- Same renderer, same block table, same chain as the three above — the ONLY deltas are
-- `nClasses` 10 → 1000, B 32 → 64 and the slug, which is why this is three `#eval`s and not a new
-- proof chain. Exactly how `resnet50in` was added on top of `resnet50`.
--
-- ⭐ B = 64 PER DEVICE, so `mnv4AdamVariant 64 1` is `"adam64"` and the artifact, the
-- `@mnv4in_adam64_train_step` entry point and the driver's default `LEAN_MLIR_VARIANT` are one
-- string. Four replicas of 64 give the global 256 the other ImageNet drivers use.
--
-- ⚠ The forwards are rendered at B = 64 too, because the driver scores eval through them at the
-- train batch. Any other batch needs re-rendered forwards or `LEAN_MLIR_SKIP_EVAL=1`, and would
-- otherwise be a shape error at the first invoke.
--
-- ✅ **THE DP PAIR IS RENDERED AND TIED**, as of 2026-08-27 — see the block below. This comment
-- used to say no DP variant was rendered, on the grounds that nothing had tied MNv4's collectives
-- and an untied artifact looks as trustworthy as the rest. Both halves of that tie now exist: the
-- `mnv4in` row in `tests/TestShardCheck.lean` and `tests/TestMnv4DpCheck.lean`.
#eval IO.FS.writeFile "verified_mlir/mnv4in_fwd.mlir"
  (Proofs.StableHLO.mnv4FwdFaithfulV 64 1000 "1.0e-5" "mnv4in")

#eval IO.FS.writeFile "verified_mlir/mnv4in_fwd_eval.mlir"
  (Proofs.StableHLO.mnv4FwdEvalFaithfulV 64 1000 "1.0e-5" "mnv4in")

#eval IO.FS.writeFile "verified_mlir/mnv4in_adam64_train_step.mlir"
  (Proofs.StableHLO.mobilenetv4AdamTrainStepFaithfulB 64 1000 "1.0e-5" 1 "mnv4in")

-- ⭐⭐ **The bf16 peer** — `adam64bf16`, the same graph with every convolution AND every depthwise
-- replaced by its bf16 twin: bf16 operands, a **bf16-TYPED** result, then a convert back to f32.
-- BN, the loss, AdamW and the master weights stay f32.
--
-- ⚠ SINGLE-DEVICE. Its 4-replica peer is `adamdp64bf16` in the block below, and it is tied —
-- ⭐ the precision axis did NOT inherit the replica axis's tie, it was given its own:
-- `DP_VARIANT=adam64bf16 DP_VARIANT_DP=adamdp64bf16 mnv4-dp-check` is a separate run, and it
-- comes back bit-exact on all 9,715,512 floats. ▶ A probe off THIS artifact is still a 1-GPU
-- number and must not be compared to R34/R50/MNv2's 4×bs64 figures without saying so.
--
-- ⭐ MNv4 is the first net to use the SYMMETRIC-pad `depthwiseStrided` family in bf16 (MNv2 used
-- the XLA-`SAME` one). Its three twins are the only ops this net needed that MobileNetV2 did not
-- already build — 47 call sites, 3 new ops.
#eval IO.FS.writeFile "verified_mlir/mnv4in_adam64bf16_train_step.mlir"
  (Proofs.StableHLO.mobilenetv4AdamTrainStepFaithfulB 64 1000 "1.0e-5" 1 "mnv4in" true)

-- ✅ **THE 4-REPLICA PAIR, TIED 2026-08-27 — the caveat that stood here is lifted.**
-- It read: "nothing has tied MNv4's collectives … these exist for ONE purpose, to COST this net at
-- the 4×bs64 geometry … ⛔ do not train off these … what would lift the caveat is a DP tie for
-- MNv4's collectives, the way R34/R50/MNv2/ConvNeXt have one." That tie was then built, exactly as
-- specified, and both halves of it are green:
--
--   `mnv4-dp-check`         (duplicated batch, `all_reduce(add)/4 = g`)
--        fp32 — bnstat BIT-EXACT 67,904/67,904, gradient norm-rel 8.45e-7
--        bf16 — BIT-EXACT on all 9,715,512 floats in θ, m, v AND bnstat
--   `shard-check mnv4in`    (asymmetric batch, `DP([x0..x3]) = mean of 4 single steps`)
--        TEST 1.10e-6 against a CONTROL of 2.00 — 1.8e6× apart
--
-- ⭐ And both went RED against a sum-not-mean render (every divisor 4.0 → 1.0): the shard TEST
-- lands on **3.000000**, which is `|4g − g| / |g|` exactly, so the gate reproduces the arithmetic
-- its failure mode implies rather than merely returning a big number.
-- ▶ `runs/2026-08-27-mnv4-dp-shard-gates/` holds the logs, the controls and a `run.sh`.
-- ⚠ Four GPUs, forced: these are the only DP renders MNv4 has, and there is no 2-replica peer, so
-- `PJRT_REPLICAS=2` hits the shim's replica-count guard rather than degrading to a 2-way run.
#eval IO.FS.writeFile "verified_mlir/mnv4in_adamdp64_train_step.mlir"
  (Proofs.StableHLO.mobilenetv4AdamTrainStepFaithfulB 64 1000 "1.0e-5" 4 "mnv4in")
#eval IO.FS.writeFile "verified_mlir/mnv4in_adamdp64bf16_train_step.mlir"
  (Proofs.StableHLO.mobilenetv4AdamTrainStepFaithfulB 64 1000 "1.0e-5" 4 "mnv4in" true)
#guard Proofs.StableHLO.mnv4AdamVariant 64 4 == "adamdp64"
#guard Proofs.StableHLO.mnv4AdamVariant 64 4 true == "adamdp64bf16"
#guard ("adamdp64bf16".splitOn "do").length == 1
#guard ("adamdp64bf16".splitOn "acc").length == 1
#guard !"adamdp64bf16".startsWith "ema"

-- ⭐ The bf16 marker, and the wiring that actually breaks: the entry name derives from
-- `mnv4AdamVariant`, so `bf16` must reach THAT call and not merely the block renderers.
#guard Proofs.StableHLO.mnv4AdamVariant 64 1 true == "adam64bf16"
#guard Proofs.StableHLO.mnv4AdamVariant 64 1 == "adam64"
-- ▶ And the slug must not trip the DRIVER's substring variant predicates. `cdOn` tests for "do".
#guard ("adam64bf16".splitOn "do").length == 1
#guard ("adam64bf16".splitOn "acc").length == 1
#guard !"adam64bf16".startsWith "ema"
