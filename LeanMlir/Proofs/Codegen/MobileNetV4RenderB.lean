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
runs at. Read off the Conv-S table (`jax/MainMobilenetV4.lean`), which lands cleanly:

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

/-- **The MobileNetV4-Conv-S parameter inputs**, in func-arg order: stem (3), the fused stage, the
    14 UIB blocks, the head conv, the classifier. Single source for the signature and the return
    order, the same role `r50ShapeList` plays for R50.

    ⚠ This list and `VLayer.toSpecs` are TWO HAND-WRITTEN READINGS of the same layout — the
    renderer cannot import the spec without inverting the dependency, which is the same two-lists
    shape as `toSpecs == XLayout.specs` elsewhere. `mnv4-sig-tie` is the `#guard` that pins them. -/
def mnv4ShapeList (nClasses : Nat) : List (String × List Nat) :=
  [("%sW", [32, 3, 3, 3]), ("%sg", [32]), ("%sbt", [32])] ++
  fusedSig "0" 32 48 4 3 ++
  uibSig "1"   48  80 4 3 5 ++
  uibSig "2"   80  80 2 3 3 ++
  uibSig "3"   80 160 6 0 3 ++
  uibSig "4"  160 160 4 3 3 ++
  uibSig "5"  160 160 4 3 5 ++
  uibSig "6"  160 160 4 5 0 ++
  uibSig "7"  160 160 4 0 3 ++
  uibSig "8"  160 160 4 3 0 ++
  uibSig "9"  160 160 4 0 0 ++
  uibSig "10" 160 160 4 3 3 ++
  uibSig "11" 160 256 6 5 5 ++
  uibSig "12" 256 256 4 5 5 ++
  uibSig "13" 256 256 4 0 3 ++
  uibSig "14" 256 256 4 3 0 ++
  [("%hW", [1280, 256, 1, 1]), ("%hg", [1280]), ("%hbt", [1280])] ++
  [("%Wd", [1280, nClasses]), ("%bd", [nClasses])]

/-- The same list as MLIR types. Derived, so the shapes have one definition. -/
def mnv4SigList (nClasses : Nat) : List (String × String) :=
  (mnv4ShapeList nClasses).map (fun (n, ds) => (n, ty ds))

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
private def uibFwdSkipB (B c expand preDWk postDWk h : Nat)
    (epsStr p xName : String) : StateM Nat UibFwdB := do
  let mid := c * expand
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
      (.depthwise (c := c) (h := h) (w := h) s!"%u{p}qW" s!"%zb{c}" zqk zc) (.operand cur zcb))
    let (c2, n2) ← pretty B (.bnBatchF (N := B) (oc := c) (h := h) (w := h)
      s!"%u{p}qg" s!"%u{p}qbt" epsStr 0 zc zc (.operand n1 zcb))
    let (c3, n3) ← pretty B (.batchOp (N := B) (.relu (n := c*h*h)) (.operand n2 zcb))
    code := code ++ c1 ++ c2 ++ c3
    qc := n1; qn := n2; qr := n3; cur := n3

  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (.conv (ic := c) (oc := mid) (h := h) (w := h) s!"%u{p}eW" s!"%zb{mid}" zke zm) (.operand cur zcb))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := h) (w := h)
    s!"%u{p}eg" s!"%u{p}ebt" epsStr 0 zm zm (.operand nEc zmb))
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand nEn zmb))
  code := code ++ cEc ++ cEn ++ cEr
  cur := nEr

  let mut dc := ""; let mut dn := ""; let mut dr := ""
  if postDWk > 0 then
    let (c1, n1) ← pretty B (.batchOp (N := B)
      (.depthwise (c := mid) (h := h) (w := h) s!"%u{p}dW" s!"%zb{mid}" zdk zm) (.operand cur zmb))
    let (c2, n2) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := h) (w := h)
      s!"%u{p}dg" s!"%u{p}dbt" epsStr 0 zm zm (.operand n1 zmb))
    let (c3, n3) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand n2 zmb))
    code := code ++ c1 ++ c2 ++ c3
    dc := n1; dn := n2; dr := n3; cur := n3

  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (.conv (ic := mid) (oc := c) (h := h) (w := h) s!"%u{p}pW" s!"%zb{c}" zkp zc) (.operand cur zmb))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := c) (h := h) (w := h)
    s!"%u{p}pg" s!"%u{p}pbt" epsStr 0 zc zc (.operand nPc zcb))
  let (cA, nA) ← pretty B (.addVB (.operand nPn zcb) (.operand xName zcb))
  code := code ++ cPc ++ cPn ++ cA

  pure { code := code, o := nA, qc := qc, qn := qn, qr := qr,
         ec := nEc, en := nEn, er := nEr, dc := dc, dn := dn, dr := dr, pc := nPc }

/-- **Stride-2 UIB where the PRE-DW carries the stride** (`preDWk > 0`). The pre-DW downsamples
    `2h×2h → h×h`, so the expand, the optional post-DW (now at stride 1) and the project all run at
    `h×h`. No skip — `ic ≠ oc`. -/
private def uibFwdPreStridedB (B ic oc expand preDWk postDWk h : Nat)
    (epsStr p xName : String) : StateM Nat UibFwdB := do
  let mid := ic * expand
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
  let zob  : Vec (B*(oc*h*h)) := fun _ => 0

  let (cQc, nQc) ← pretty B (.batchOp (N := B)
    (.depthwiseStrided (c := ic) (h := h) (w := h) s!"%u{p}qW" s!"%zb{ic}" zqk zic) (.operand xName zin))
  let (cQn, nQn) ← pretty B (.bnBatchF (N := B) (oc := ic) (h := h) (w := h)
    s!"%u{p}qg" s!"%u{p}qbt" epsStr 0 zic zic (.operand nQc zqb))
  let (cQr, nQr) ← pretty B (.batchOp (N := B) (.relu (n := ic*h*h)) (.operand nQn zqb))

  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (.conv (ic := ic) (oc := mid) (h := h) (w := h) s!"%u{p}eW" s!"%zb{mid}" zke zm) (.operand nQr zqb))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := h) (w := h)
    s!"%u{p}eg" s!"%u{p}ebt" epsStr 0 zm zm (.operand nEc zmb))
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand nEn zmb))

  let mut code := cQc ++ cQn ++ cQr ++ cEc ++ cEn ++ cEr
  let mut cur := nEr
  let mut dc := ""; let mut dn := ""; let mut dr := ""
  if postDWk > 0 then
    let (c1, n1) ← pretty B (.batchOp (N := B)
      (.depthwise (c := mid) (h := h) (w := h) s!"%u{p}dW" s!"%zb{mid}" zdk zm) (.operand cur zmb))
    let (c2, n2) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := h) (w := h)
      s!"%u{p}dg" s!"%u{p}dbt" epsStr 0 zm zm (.operand n1 zmb))
    let (c3, n3) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand n2 zmb))
    code := code ++ c1 ++ c2 ++ c3
    dc := n1; dn := n2; dr := n3; cur := n3

  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (.conv (ic := mid) (oc := oc) (h := h) (w := h) s!"%u{p}pW" s!"%zb{oc}" zkp zoc) (.operand cur zmb))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := h) (w := h)
    s!"%u{p}pg" s!"%u{p}pbt" epsStr 0 zoc zoc (.operand nPc zob))
  code := code ++ cPc ++ cPn

  pure { code := code, o := nPn, qc := nQc, qn := nQn, qr := nQr,
         ec := nEc, en := nEn, er := nEr, dc := dc, dn := dn, dr := dr, pc := nPc }

/-- **Stride-2 UIB where the POST-DW carries the stride** (`preDWk = 0`, `postDWk > 0`) — the IB /
    MBConv family at a downsample. The expand runs at the INPUT size `2h×2h`; the post-DW then
    downsamples to `h×h`. No skip — `ic ≠ oc`. -/
private def uibFwdPostStridedB (B ic oc expand postDWk h : Nat)
    (epsStr p xName : String) : StateM Nat UibFwdB := do
  let mid := ic * expand
  let zoc  : Vec oc := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zdk  : DepthwiseKernel mid postDWk postDWk := fun _ _ _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zin  : Vec (B*(ic*(2*h)*(2*h))) := fun _ => 0
  let zeb  : Vec (B*(mid*(2*h)*(2*h))) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0
  let zob  : Vec (B*(oc*h*h)) := fun _ => 0

  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (.conv (ic := ic) (oc := mid) (h := 2*h) (w := 2*h) s!"%u{p}eW" s!"%zb{mid}" zke zm) (.operand xName zin))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := 2*h) (w := 2*h)
    s!"%u{p}eg" s!"%u{p}ebt" epsStr 0 zm zm (.operand nEc zeb))
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu (n := mid*(2*h)*(2*h))) (.operand nEn zeb))

  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (.depthwiseStrided (c := mid) (h := h) (w := h) s!"%u{p}dW" s!"%zb{mid}" zdk zm) (.operand nEr zeb))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := h) (w := h)
    s!"%u{p}dg" s!"%u{p}dbt" epsStr 0 zm zm (.operand nDc zmb))
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu (n := mid*h*h)) (.operand nDn zmb))

  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (.conv (ic := mid) (oc := oc) (h := h) (w := h) s!"%u{p}pW" s!"%zb{oc}" zkp zoc) (.operand nDr zmb))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := h) (w := h)
    s!"%u{p}pg" s!"%u{p}pbt" epsStr 0 zoc zoc (.operand nPc zob))

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
private def fusedMbConvFwdStridedB (B ic oc expand k h : Nat)
    (epsStr p xName : String) : StateM Nat UibFwdB := do
  let mid := if expand == 1 then oc else ic * expand
  let zoc  : Vec oc := fun _ => 0
  let zm   : Vec mid := fun _ => 0
  let zkf  : Kernel4 mid ic k k := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zin  : Vec (B*(ic*(2*h)*(2*h))) := fun _ => 0
  let zmb  : Vec (B*(mid*h*h)) := fun _ => 0
  let zob  : Vec (B*(oc*h*h)) := fun _ => 0

  let (cFc, nFc) ← pretty B (.batchOp (N := B)
    (.convStrided (ic := ic) (oc := mid) (h := h) (w := h) (kH := k) (kW := k)
      s!"%f{p}cW" s!"%zb{mid}" zkf zm) (.operand xName zin))
  let (cFn, nFn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := h) (w := h)
    s!"%f{p}cg" s!"%f{p}cbt" epsStr 0 zm zm (.operand nFc zmb))
  let (cFs, nFs) ← pretty B (.batchOp (N := B) (.swish (n := mid*h*h)) (.operand nFn zmb))

  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (.conv (ic := mid) (oc := oc) (h := h) (w := h) s!"%f{p}pW" s!"%zb{oc}" zkp zoc) (.operand nFs zmb))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := h) (w := h)
    s!"%f{p}pg" s!"%f{p}pbt" epsStr 0 zoc zoc (.operand nPc zob))

  pure { code := cFc ++ cFn ++ cFs ++ cPc ++ cPn,
         o := nPn, qc := "", qn := "", qr := "",
         ec := nFc, en := nFn, er := nFs, dc := "", dn := "", dr := "", pc := nPc }

/-- **The MobileNetV4-Conv-S forward chain**, batch BN, at `N := B`, 224² → 10 classes.

    Transcribed 1:1 from `jax/MainMobilenetV4.lean` — the Conv-S-sized demo that `RESULTS.md`'s
    **84.58%** belongs to, NOT faithful Conv-M (~9.7M, no accuracy run). Spatial ladder:

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

    Returns `(code, logits-SSA)` with logits `[B, nClasses]`. -/
def mnv4FwdChainB (B nClasses : Nat) (epsStr : String) : StateM Nat (String × String) := do
  -- ═══ stem: 3×3/s2 conv (3→32), 224→112 → batch BN → relu ═══
  let zx    : Vec (B*(3*224*224)) := fun _ => 0
  let zSk   : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
  let z32   : Vec 32 := fun _ => 0
  let z112  : Vec (B*(32*112*112)) := fun _ => 0
  let (cStc, nStc) ← pretty B (.batchOp (N := B)
    (.convStrided (ic := 3) (oc := 32) (h := 112) (w := 112) (kH := 3) (kW := 3) "%sW" "%zb32" zSk z32)
    (.operand "%x" zx))
  let (cStn, nStn) ← pretty B (.bnBatchF (N := B) (oc := 32) (h := 112) (w := 112)
    "%sg" "%sbt" epsStr 0 z32 z32 (.operand nStc z112))
  let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu (n := 32*112*112)) (.operand nStn z112))

  -- ═══ stage 0: the fused inverted bottleneck, 112→56 (swish) ═══
  let f0 ← fusedMbConvFwdStridedB B 32 48 4 3 56 epsStr "0" nStr

  -- ═══ the 14 UIB blocks ═══
  let b1  ← uibFwdPreStridedB  B  48  80 4 3 5 28 epsStr "1"  f0.o   -- ExtraDW  56→28
  let b2  ← uibFwdSkipB        B  80     2 3 3 28 epsStr "2"  b1.o   -- ExtraDW  28
  let b3  ← uibFwdPostStridedB B  80 160 6   3 14 epsStr "3"  b2.o   -- IB       28→14
  let b4  ← uibFwdSkipB        B 160     4 3 3 14 epsStr "4"  b3.o   -- ExtraDW  14
  let b5  ← uibFwdSkipB        B 160     4 3 5 14 epsStr "5"  b4.o   -- ExtraDW  14
  let b6  ← uibFwdSkipB        B 160     4 5 0 14 epsStr "6"  b5.o   -- ConvNeXt 14
  let b7  ← uibFwdSkipB        B 160     4 0 3 14 epsStr "7"  b6.o   -- IB       14
  let b8  ← uibFwdSkipB        B 160     4 3 0 14 epsStr "8"  b7.o   -- ConvNeXt 14
  let b9  ← uibFwdSkipB        B 160     4 0 0 14 epsStr "9"  b8.o   -- FFN      14
  let b10 ← uibFwdSkipB        B 160     4 3 3 14 epsStr "10" b9.o   -- ExtraDW  14
  let b11 ← uibFwdPreStridedB  B 160 256 6 5 5  7 epsStr "11" b10.o  -- ExtraDW  14→7
  let b12 ← uibFwdSkipB        B 256     4 5 5  7 epsStr "12" b11.o  -- ExtraDW  7
  let b13 ← uibFwdSkipB        B 256     4 0 3  7 epsStr "13" b12.o  -- IB       7
  let b14 ← uibFwdSkipB        B 256     4 3 0  7 epsStr "14" b13.o  -- ConvNeXt 7

  -- ═══ head: 1×1 conv (256→1280) → batch BN → relu → GAP(7×7) → dense ═══
  let z7     : Vec (B*(256*7*7)) := fun _ => 0
  let zHk    : Kernel4 1280 256 1 1 := fun _ _ _ _ => 0
  let z1280  : Vec 1280 := fun _ => 0
  let zH7    : Vec (B*(1280*7*7)) := fun _ => 0
  let z1280b : Vec (B*1280) := fun _ => 0
  let zWd    : Mat 1280 nClasses := fun _ _ => 0
  let zNC    : Vec nClasses := fun _ => 0
  let (cHc, nHc) ← pretty B (.batchOp (N := B)
    (.conv (ic := 256) (oc := 1280) (h := 7) (w := 7) "%hW" "%zb1280" zHk z1280) (.operand b14.o z7))
  let (cHn, nHn) ← pretty B (.bnBatchF (N := B) (oc := 1280) (h := 7) (w := 7)
    "%hg" "%hbt" epsStr 0 z1280 z1280 (.operand nHc zH7))
  let (cHr, nHr) ← pretty B (.batchOp (N := B) (.relu (n := 1280*7*7)) (.operand nHn zH7))
  let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 1280) (h := 7) (w := 7))
    (.operand nHr zH7))
  let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC)
    (.operand nGap z1280b))

  pure (cStc ++ cStn ++ cStr ++ f0.code
        ++ b1.code ++ b2.code ++ b3.code ++ b4.code ++ b5.code ++ b6.code ++ b7.code
        ++ b8.code ++ b9.code ++ b10.code ++ b11.code ++ b12.code ++ b13.code ++ b14.code
        ++ cHc ++ cHn ++ cHr ++ cGap ++ cLog, nLog)

/-- Every distinct channel width a bias-free conv in this net binds `%zb{c}` at: the stem, the fused
    stage's two convs, each block's pre-DW (`ic`), expand and post-DW (`mid`) and project (`oc`), and
    the head. Derived by hand and pinned by `mnv4-fwd-smoke`, which fails on an unbound `%zb`. -/
def mnv4ZbWidths : List Nat :=
  [32, 48, 80, 128, 160, 192, 256, 480, 640, 960, 1024, 1280]

/-- **`@mnv4_fwd`** — the MobileNetV4-Conv-S forward as one MLIR module.

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
  let (code, logits) := (mnv4FwdChainB B nClasses epsStr).run' 0
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd{vSuffix}({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── MobileNetV4-Conv-S forward: every line is pretty(verified AST node) ──\n" ++
  zeroBiasPrelude false mnv4ZbWidths ++ code ++
  s!"    return {logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

end Proofs.StableHLO
