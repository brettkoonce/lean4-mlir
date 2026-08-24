import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Codegen.MobileNetV2Render
import LeanMlir.ViTRender

/-! # MobileNetV2 AdamW train step rendered from the verified AST, at the BATCHED index

The §2f peer of `MobileNetV2Render.lean`, and structurally the mnv2 twin of `ResNet34RenderB.lean`
rather than of `EfficientNetRender.lean`. That distinction is the whole reason this is a separate
file, and it was settled by measurement:

* **`MobileNetV2Render.lean` renders PER-EXAMPLE BN** (`bnPerChannelF`, reduce `[2,3]` — 417 of them
  in `mobilenetv2_train_step.mlir`, with 156 = 52×3 `rsqrt`), while
  `verified_mlir/mobilenetv2_adam_train_step.mlir` — what `mobilenetv2-verified-adam` actually
  trains on — is **BATCH BN** (52 `rsqrt`, 364 `reduce[0,2,3]`). Two different functions: exactly the
  §2a two-worlds divergence, live.
* EfficientNet could thread one `adam : Bool` through ONE renderer because
  `efficientnet_train_step.mlir` was **already batch-BN** (539 `reduce[0,2,3]`), so §2b's `N := B`
  move came back byte-neutral there. Re-instantiating mnv2's per-example renderer at `N := B` would
  instead change `mobilenetv2_train_step.mlir`'s bytes AND its function, voiding the 80-epoch SGD
  log (`runs/mobilenetv2_verified_crop_gpu0.log`, 32.9% → 86.89%). So this file is AdamW-only and
  `MobileNetV2Render.lean` is not touched — the `ResNet34RenderB` shape.

**Consequence for the gates.** The §0 gate "the SGD artifact re-renders byte-identical after the
`adam : Bool` threading" does not apply literally here: nothing is threaded. It degrades to
*"`git diff verified_mlir/mobilenetv2_train_step.mlir` is empty"*, which is strictly weaker, because
nothing forces the two renderers to stay in step. §2b-ter's sibling-race warning applies.

**The whole graph sits at `N := B`**, so every batch-coupled `den` is honest: `bnBatchF`,
`bnBatchBack` and the whole `*GradB` family reduce over the batch, and at `N = 1` each would
describe a one-example function while the emitted text reduces over all `B` (§2b).

**The ops this net needed that no other did** (§2f): `BatchableOp.relu6` — mnv2 is the only ReLU6 net
in the kit, EfficientNet being all-swish — and `selectMidB`, its two-sided backward mask, which
reads the saved per-example pre-activation and therefore CANNOT be a `BatchableOp` descriptor. Plus
`depthwise{,Strided}BiasGradB`: enet's depthwise convs are followed by BN so their bias is folded
into it, mnv2's are not.

The optimizer is the proven `adamMNextF`/`adamVNextF`/`adamWParamF` triple applied to the un-fused
`*GradB` gradients. The cotangent is composed from kit ops (`softmaxRow → subB → scaleB → addVB →
shiftB → divConstB`, α = 0.1, K = nClasses), so this render does NOT match the hand-written artifact
op-for-op and the tie against it must be numeric. `%loss` is report-only and stays outside the AST,
exactly as `resnet34`/`cifar8`'s does (§5).

Render is value-independent (`skel` erases values), so placeholder zeros and `ε := 0` are passed;
the emitted literals carry the real values.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-- A trainable parameter: emitted name (no `%`), its un-fused gradient SSA name, and its shape.
    The AdamW tail is a fold over this list, so the θ/m/v output order cannot drift from the
    signature order. (The `ResNet34RenderB` peer of the same name is `private` to that file.) -/
structure PGradM where
  nm   : String
  grad : String
  ds   : List Nat
deriving Inhabited

/-- Saved forward SSA names a block's backward + gradient passes reference. `ec`/`en`/`er` are the
    block input for the no-expand block (b1), which has no expand conv. -/
structure MBFwdB where
  code : String
  o  : String        -- block output (project-BN out, or the addVB result for skip blocks)
  ec : String        -- expand conv output    (= expand-BN input)
  en : String        -- expand BN output      (= expand-relu6 pre-activation)
  er : String        -- expand relu6 output   (= depthwise input)
  dc : String        -- depthwise conv output (= depthwise-BN input)
  dn : String        -- depthwise BN output   (= depthwise-relu6 pre-activation)
  dr : String        -- depthwise relu6 out   (= project input)
  pc : String        -- project conv output   (= project-BN input)
deriving Inhabited

/-- Backward result: code, the dx cotangent to the previous block, and the block's parameter
    gradients in func-arg order. -/
structure MBBackB where
  code : String
  dx : String
  ps : List PGradM

-- ════════════════════════════════════════════════════════════════
-- § Block forward (batch BN), all at `N := B`
--   inverted residual: expand(1×1)→BN→relu6 → depthwise(3×3)→BN→relu6 → project(1×1)→BN
-- ════════════════════════════════════════════════════════════════

/-- **STRIDED inverted-residual forward**: expand at the input `2hh×2ww`, depthwise downsamples
    `2hh×2ww → hh×ww`, project 1×1 at `hh×ww`. NO skip. -/
private def irFwdStridedB (B ic mid oc hh : Nat) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) : StateM Nat MBFwdB := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER, exactly as the `z*` zero kernels are: the render
  -- produces TEXT and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (B*(ic*(2*hh)*(2*ww))) := fun _ => 0
  let zeb  : Vec (B*(mid*(2*hh)*(2*ww))) := fun _ => 0
  let zdb  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zob  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) zrnd s!"%b{p}eW" (biasName convBias s!"%b{p}eb" mid) zke zmid else .conv (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%b{p}eW" (biasName convBias s!"%b{p}eb" mid) zke zmid)
    (.operand xName zxin))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := 2*hh) (w := 2*ww)
    s!"%b{p}eg" s!"%b{p}ebt" epsStr 0 zmid zmid (.operand nEc zeb))
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*(2*hh)*(2*ww))) (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (if bf16 then .depthwiseStridedXlaBf16 (c := mid) (h := hh) (w := ww) zrnd s!"%b{p}dW" (biasName convBias s!"%b{p}db" mid) zdk zmid else .depthwiseStridedXla (c := mid) (h := hh) (w := ww) s!"%b{p}dW" (biasName convBias s!"%b{p}db" mid) zdk zmid)
    (.operand nEr zeb))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww)
    s!"%b{p}dg" s!"%b{p}dbt" epsStr 0 zmid zmid (.operand nDc zdb))
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nDn zdb))
  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc else .conv (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc)
    (.operand nDr zdb))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pg" s!"%b{p}pbt" epsStr 0 zoc zoc (.operand nPc zob))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc }

/-- **STRIDE-1 inverted-residual forward with the identity skip** (`ic = oc`): everything at
    `hh×ww`, block output = `addVB (project-BN out) (block input)`. The bottleneck is LINEAR — no
    relu6 after the add. -/
private def irFwdSkipB (B ic mid oc hh : Nat) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) : StateM Nat MBFwdB := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER, exactly as the `z*` zero kernels are: the render
  -- produces TEXT and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zeb  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zob  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := ic) (oc := mid) (h := hh) (w := ww) zrnd s!"%b{p}eW" (biasName convBias s!"%b{p}eb" mid) zke zmid else .conv (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%b{p}eW" (biasName convBias s!"%b{p}eb" mid) zke zmid)
    (.operand xName zxin))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww)
    s!"%b{p}eg" s!"%b{p}ebt" epsStr 0 zmid zmid (.operand nEc zeb))
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (if bf16 then .depthwiseBf16 (c := mid) (h := hh) (w := ww) zrnd s!"%b{p}dW" (biasName convBias s!"%b{p}db" mid) zdk zmid else .depthwise (c := mid) (h := hh) (w := ww) s!"%b{p}dW" (biasName convBias s!"%b{p}db" mid) zdk zmid) (.operand nEr zeb))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww)
    s!"%b{p}dg" s!"%b{p}dbt" epsStr 0 zmid zmid (.operand nDc zeb))
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nDn zeb))
  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc else .conv (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc)
    (.operand nDr zeb))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pg" s!"%b{p}pbt" epsStr 0 zoc zoc (.operand nPc zob))
  let (cA, nA) ← pretty B (.addVB (.operand nPn zob) (.operand xName zob))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn ++ cA,
         o := nA, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc }

/-- **EXPAND-NO-SKIP stride-1 forward** (b11/b17): as `irFwdSkipB` but `ic ≠ oc`, so the block
    output is the project-BN output directly. -/
private def irFwdNoSkipB (B ic mid oc hh : Nat) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) : StateM Nat MBFwdB := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER, exactly as the `z*` zero kernels are: the render
  -- produces TEXT and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zeb  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zob  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := ic) (oc := mid) (h := hh) (w := ww) zrnd s!"%b{p}eW" (biasName convBias s!"%b{p}eb" mid) zke zmid else .conv (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%b{p}eW" (biasName convBias s!"%b{p}eb" mid) zke zmid)
    (.operand xName zxin))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww)
    s!"%b{p}eg" s!"%b{p}ebt" epsStr 0 zmid zmid (.operand nEc zeb))
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (if bf16 then .depthwiseBf16 (c := mid) (h := hh) (w := ww) zrnd s!"%b{p}dW" (biasName convBias s!"%b{p}db" mid) zdk zmid else .depthwise (c := mid) (h := hh) (w := ww) s!"%b{p}dW" (biasName convBias s!"%b{p}db" mid) zdk zmid) (.operand nEr zeb))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww)
    s!"%b{p}dg" s!"%b{p}dbt" epsStr 0 zmid zmid (.operand nDc zeb))
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nDn zeb))
  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc else .conv (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc)
    (.operand nDr zeb))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pg" s!"%b{p}pbt" epsStr 0 zoc zoc (.operand nPc zob))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc }

/-- **NO-EXPAND forward** (b1, the canonical `t = 1` block): depthwise(stride-1, on `ic` channels)
    → BN → relu6 → project(1×1 `ic→oc`) → BN. No expand conv, no skip. `er` is the depthwise INPUT
    (= the block input), which is what the depthwise weight gradient reads. -/
private def irFwdNoExpB (B ic oc hh : Nat) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) : StateM Nat MBFwdB := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER, exactly as the `z*` zero kernels are: the render
  -- produces TEXT and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zic  : Vec ic := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zkp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel ic 3 3 := fun _ _ _ => 0
  let zib  : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zob  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (if bf16 then .depthwiseBf16 (c := ic) (h := hh) (w := ww) zrnd s!"%b{p}dW" (biasName convBias s!"%b{p}db" ic) zdk zic else .depthwise (c := ic) (h := hh) (w := ww) s!"%b{p}dW" (biasName convBias s!"%b{p}db" ic) zdk zic) (.operand xName zib))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := B) (oc := ic) (h := hh) (w := ww)
    s!"%b{p}dg" s!"%b{p}dbt" epsStr 0 zic zic (.operand nDc zib))
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu6 (n := ic*hh*ww)) (.operand nDn zib))
  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (if bf16 then .convBf16 (ic := ic) (oc := oc) (h := hh) (w := ww) zrnd s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc else .conv (ic := ic) (oc := oc) (h := hh) (w := ww) s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc)
    (.operand nDr zib))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pg" s!"%b{p}pbt" epsStr 0 zoc zoc (.operand nPc zob))
  pure { code := cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := xName, en := xName, er := xName, dc := nDc, dn := nDn, dr := nDr, pc := nPc }

-- ════════════════════════════════════════════════════════════════
-- § Block backward + un-fused parameter gradients
--   (project → depthwise → expand; dyOut flows straight into the project-BN backward,
--    because the linear bottleneck has no relu6 after project)
-- ════════════════════════════════════════════════════════════════

/-- **STRIDED backward + 12 un-fused gradients.** The depthwise input gradient lands at `2hh×2ww`;
    no skip, so the dx handed to the previous block is the expand-conv backward directly. -/
private def irBackStridedGradB (B ic mid oc hh : Nat) (epsStr p xName : String)
    (f : MBFwdB) (dyName : String) (convBias : Bool)
    (bf16 : Bool := false) : StateM Nat MBBackB := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER, exactly as the `z*` zero kernels are: the render
  -- produces TEXT and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (B*(ic*(2*hh)*(2*ww))) := fun _ => 0
  let zeb  : Vec (B*(mid*(2*hh)*(2*ww))) := fun _ => 0
  let zebp : Vec (B*(mid*((2*hh)*(2*ww)))) := fun _ => 0
  let zdb  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zdbp : Vec (B*(mid*(hh*ww))) := fun _ => 0
  let zob  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zobp : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let (cDpc, nDpc) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pg" f.pc epsStr 0 zoc zobp (.operand dyName zobp))
  let (cDdr, nDdr) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd
    s!"%b{p}pW" zkp zoc (.operand nDpc zob) else .convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pW" zkp zoc (.operand nDpc zob))
  let (cDdm, nDdm) ← pretty B (.selectMidB f.dn zdb (.operand nDdr zdb))
  let (cDdn, nDdn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww)
    s!"%b{p}dg" f.dc epsStr 0 zmid zdbp (.operand nDdm zdbp))
  let (cDer, nDer) ← pretty B (if bf16 then .depthwiseStridedXlaBackBatchedBf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd
    s!"%b{p}dW" zdk zmid (.operand nDdn zdb) else .depthwiseStridedXlaBackBatched (N := B) (c := mid) (h := hh) (w := ww)
    s!"%b{p}dW" zdk zmid (.operand nDdn zdb))
  let (cDem, nDem) ← pretty B (.selectMidB f.en zeb (.operand nDer zeb))
  let (cDen, nDen) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := 2*hh) (w := 2*ww)
    s!"%b{p}eg" f.ec epsStr 0 zmid zebp (.operand nDem zebp))
  let (cDxb, nDxb) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := ic) (oc := mid)
    (h := 2*hh) (w := 2*ww) zrnd s!"%b{p}eW" zke zmid (.operand nDen zeb) else .convBackBatched (N := B) (ic := ic) (oc := mid)
    (h := 2*hh) (w := 2*ww) s!"%b{p}eW" zke zmid (.operand nDen zeb))
  -- the 12 gradients, func-arg order: eW eb eg ebt | dW db dg dbt | pW pb pg pbt
  let (cEW, nEW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := ic) (oc := mid)
    (h := 2*hh) (w := 2*ww) zrnd xName zmid zxin zke (.operand nDen zeb) else .convWeightGradB (N := B) (ic := ic) (oc := mid)
    (h := 2*hh) (w := 2*ww) xName zmid zxin zke (.operand nDen zeb))
  let (cEb, nEb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := ic) (oc := mid)
        (h := 2*hh) (w := 2*ww) zke zxin zmid (.operand nDen zeb))
    else pure ("", "")
  let (cEg, nEg) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := 2*hh) (w := 2*ww)
    f.ec epsStr 0 zebp (.operand nDem zebp))
  let (cEt, nEt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := 2*hh) (w := 2*ww)
    (.operand nDem zebp))
  let (cDW, nDW) ← pretty B (if bf16 then .depthwiseStridedXlaWeightGradBBf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd
    f.er zmid zeb zdk (.operand nDdn zdb) else .depthwiseStridedXlaWeightGradB (N := B) (c := mid) (h := hh) (w := ww)
    f.er zmid zeb zdk (.operand nDdn zdb))
  let (cDb, nDb) ← if convBias then
      pretty B (.depthwiseStridedXlaBiasGradB (N := B) (c := mid) (h := hh) (w := ww)
        zdk zeb zmid (.operand nDdn zdb))
    else pure ("", "")
  let (cDg, nDg) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := hh) (w := ww)
    f.dc epsStr 0 zdbp (.operand nDdm zdbp))
  let (cDt, nDt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww)
    (.operand nDdm zdbp))
  let (cPW, nPW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd
    f.dr zoc zdb zkp (.operand nDpc zob) else .convWeightGradB (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww)
    f.dr zoc zdb zkp (.operand nDpc zob))
  let (cPb, nPb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww)
        zkp zdb zoc (.operand nDpc zob))
    else pure ("", "")
  let (cPg, nPg) ← pretty B (.bnGammaGradB (N := B) (oc := oc) (h := hh) (w := ww)
    f.pc epsStr 0 zobp (.operand dyName zobp))
  let (cPt, nPt) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww)
    (.operand dyName zobp))
  pure { code := cDpc ++ cDdr ++ cDdm ++ cDdn ++ cDer ++ cDem ++ cDen ++ cDxb ++
                 cEW ++ cEb ++ cEg ++ cEt ++ cDW ++ cDb ++ cDg ++ cDt ++ cPW ++ cPb ++ cPg ++ cPt,
         dx := nDxb,
         ps := [⟨s!"b{p}eW", nEW, [mid,ic,1,1]⟩] ++
                (if convBias then [⟨s!"b{p}eb", nEb, [mid]⟩] else []) ++
                [⟨s!"b{p}eg", nEg, [mid]⟩, ⟨s!"b{p}ebt", nEt, [mid]⟩, ⟨s!"b{p}dW", nDW, [mid,1,3,3]⟩] ++
                (if convBias then [⟨s!"b{p}db", nDb, [mid]⟩] else []) ++
                [⟨s!"b{p}dg", nDg, [mid]⟩, ⟨s!"b{p}dbt", nDt, [mid]⟩, ⟨s!"b{p}pW", nPW, [oc,mid,1,1]⟩] ++
                (if convBias then [⟨s!"b{p}pb", nPb, [oc]⟩] else []) ++
                [⟨s!"b{p}pg", nPg, [oc]⟩, ⟨s!"b{p}pbt", nPt, [oc]⟩] }

/-- **STRIDE-1 backward + 12 un-fused gradients**, shared by the skip (`skip := true`) and
    no-skip block kinds — the ONLY difference is the skip's `addVB` fan-in on the dx, which is why
    they are one function with a flag rather than two near-copies (the double-writer disease one
    level down, §2a-quater). `ic = oc` whenever `skip` is true. -/
private def irBackStride1GradB (B ic mid oc hh : Nat) (skip : Bool) (epsStr p xName : String)
    (f : MBFwdB) (dyName : String) (convBias : Bool)
    (bf16 : Bool := false) : StateM Nat MBBackB := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER, exactly as the `z*` zero kernels are: the render
  -- produces TEXT and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zeb  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zebp : Vec (B*(mid*(hh*ww))) := fun _ => 0
  let zob  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zobp : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let (cDpc, nDpc) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pg" f.pc epsStr 0 zoc zobp (.operand dyName zobp))
  let (cDdr, nDdr) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd
    s!"%b{p}pW" zkp zoc (.operand nDpc zob) else .convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pW" zkp zoc (.operand nDpc zob))
  let (cDdm, nDdm) ← pretty B (.selectMidB f.dn zeb (.operand nDdr zeb))
  let (cDdn, nDdn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww)
    s!"%b{p}dg" f.dc epsStr 0 zmid zebp (.operand nDdm zebp))
  let (cDer, nDer) ← pretty B (if bf16 then .depthwiseBackBatchedBf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd
    s!"%b{p}dW" zdk zmid (.operand nDdn zeb) else .depthwiseBackBatched (N := B) (c := mid) (h := hh) (w := ww)
    s!"%b{p}dW" zdk zmid (.operand nDdn zeb))
  let (cDem, nDem) ← pretty B (.selectMidB f.en zeb (.operand nDer zeb))
  let (cDen, nDen) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww)
    s!"%b{p}eg" f.ec epsStr 0 zmid zebp (.operand nDem zebp))
  let (cDxb, nDxb) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww) zrnd
    s!"%b{p}eW" zke zmid (.operand nDen zeb) else .convBackBatched (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww)
    s!"%b{p}eW" zke zmid (.operand nDen zeb))
  -- skip fan-in: (body dx) + dyOut, at the block-input shape (ic = oc for a skip block)
  let (cDx, nDx) ← if skip then
      pretty B (.addVB (.operand nDxb zxin) (.operand dyName zxin))
    else pure ("", nDxb)
  let (cEW, nEW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww) zrnd
    xName zmid zxin zke (.operand nDen zeb) else .convWeightGradB (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww)
    xName zmid zxin zke (.operand nDen zeb))
  let (cEb, nEb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww)
        zke zxin zmid (.operand nDen zeb))
    else pure ("", "")
  let (cEg, nEg) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := hh) (w := ww)
    f.ec epsStr 0 zebp (.operand nDem zebp))
  let (cEt, nEt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww)
    (.operand nDem zebp))
  let (cDW, nDW) ← pretty B (if bf16 then .depthwiseWeightGradBBf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd
    f.er zmid zeb zdk (.operand nDdn zeb) else .depthwiseWeightGradB (N := B) (c := mid) (h := hh) (w := ww)
    f.er zmid zeb zdk (.operand nDdn zeb))
  let (cDb, nDb) ← if convBias then
      pretty B (.depthwiseBiasGradB (N := B) (c := mid) (h := hh) (w := ww)
        zdk zeb zmid (.operand nDdn zeb))
    else pure ("", "")
  let (cDg, nDg) ← pretty B (.bnGammaGradB (N := B) (oc := mid) (h := hh) (w := ww)
    f.dc epsStr 0 zebp (.operand nDdm zebp))
  let (cDt, nDt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww)
    (.operand nDdm zebp))
  let (cPW, nPW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd
    f.dr zoc zeb zkp (.operand nDpc zob) else .convWeightGradB (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww)
    f.dr zoc zeb zkp (.operand nDpc zob))
  let (cPb, nPb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww)
        zkp zeb zoc (.operand nDpc zob))
    else pure ("", "")
  let (cPg, nPg) ← pretty B (.bnGammaGradB (N := B) (oc := oc) (h := hh) (w := ww)
    f.pc epsStr 0 zobp (.operand dyName zobp))
  let (cPt, nPt) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww)
    (.operand dyName zobp))
  pure { code := cDpc ++ cDdr ++ cDdm ++ cDdn ++ cDer ++ cDem ++ cDen ++ cDxb ++ cDx ++
                 cEW ++ cEb ++ cEg ++ cEt ++ cDW ++ cDb ++ cDg ++ cDt ++ cPW ++ cPb ++ cPg ++ cPt,
         dx := nDx,
         ps := [⟨s!"b{p}eW", nEW, [mid,ic,1,1]⟩] ++
                (if convBias then [⟨s!"b{p}eb", nEb, [mid]⟩] else []) ++
                [⟨s!"b{p}eg", nEg, [mid]⟩, ⟨s!"b{p}ebt", nEt, [mid]⟩, ⟨s!"b{p}dW", nDW, [mid,1,3,3]⟩] ++
                (if convBias then [⟨s!"b{p}db", nDb, [mid]⟩] else []) ++
                [⟨s!"b{p}dg", nDg, [mid]⟩, ⟨s!"b{p}dbt", nDt, [mid]⟩, ⟨s!"b{p}pW", nPW, [oc,mid,1,1]⟩] ++
                (if convBias then [⟨s!"b{p}pb", nPb, [oc]⟩] else []) ++
                [⟨s!"b{p}pg", nPg, [oc]⟩, ⟨s!"b{p}pbt", nPt, [oc]⟩] }

/-- **NO-EXPAND backward + 8 un-fused gradients** (b1). No expand conv and no skip, so the dx to
    the stem is the depthwise backward directly. -/
private def irBackNoExpGradB (B ic oc hh : Nat) (epsStr p xName : String)
    (f : MBFwdB) (dyName : String) (convBias : Bool)
    (bf16 : Bool := false) : StateM Nat MBBackB := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER, exactly as the `z*` zero kernels are: the render
  -- produces TEXT and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zic  : Vec ic := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zkp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel ic 3 3 := fun _ _ _ => 0
  let zib  : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zibp : Vec (B*(ic*(hh*ww))) := fun _ => 0
  let zob  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zobp : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let (cDpc, nDpc) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pg" f.pc epsStr 0 zoc zobp (.operand dyName zobp))
  let (cDdr, nDdr) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) zrnd
    s!"%b{p}pW" zkp zoc (.operand nDpc zob) else .convBackBatched (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww)
    s!"%b{p}pW" zkp zoc (.operand nDpc zob))
  let (cDdm, nDdm) ← pretty B (.selectMidB f.dn zib (.operand nDdr zib))
  let (cDdn, nDdn) ← pretty B (.bnBatchBack (N := B) (oc := ic) (h := hh) (w := ww)
    s!"%b{p}dg" f.dc epsStr 0 zic zibp (.operand nDdm zibp))
  let (cDxb, nDxb) ← pretty B (if bf16 then .depthwiseBackBatchedBf16 (N := B) (c := ic) (h := hh) (w := ww) zrnd
    s!"%b{p}dW" zdk zic (.operand nDdn zib) else .depthwiseBackBatched (N := B) (c := ic) (h := hh) (w := ww)
    s!"%b{p}dW" zdk zic (.operand nDdn zib))
  let (cDW, nDW) ← pretty B (if bf16 then .depthwiseWeightGradBBf16 (N := B) (c := ic) (h := hh) (w := ww) zrnd
    xName zic zib zdk (.operand nDdn zib) else .depthwiseWeightGradB (N := B) (c := ic) (h := hh) (w := ww)
    xName zic zib zdk (.operand nDdn zib))
  let (cDb, nDb) ← if convBias then
      pretty B (.depthwiseBiasGradB (N := B) (c := ic) (h := hh) (w := ww)
        zdk zib zic (.operand nDdn zib))
    else pure ("", "")
  let (cDg, nDg) ← pretty B (.bnGammaGradB (N := B) (oc := ic) (h := hh) (w := ww)
    f.dc epsStr 0 zibp (.operand nDdm zibp))
  let (cDt, nDt) ← pretty B (.bnBetaGradB (N := B) (oc := ic) (h := hh) (w := ww)
    (.operand nDdm zibp))
  let (cPW, nPW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) zrnd
    f.dr zoc zib zkp (.operand nDpc zob) else .convWeightGradB (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww)
    f.dr zoc zib zkp (.operand nDpc zob))
  let (cPb, nPb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww)
        zkp zib zoc (.operand nDpc zob))
    else pure ("", "")
  let (cPg, nPg) ← pretty B (.bnGammaGradB (N := B) (oc := oc) (h := hh) (w := ww)
    f.pc epsStr 0 zobp (.operand dyName zobp))
  let (cPt, nPt) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww)
    (.operand dyName zobp))
  pure { code := cDpc ++ cDdr ++ cDdm ++ cDdn ++ cDxb ++
                 cDW ++ cDb ++ cDg ++ cDt ++ cPW ++ cPb ++ cPg ++ cPt,
         dx := nDxb,
         ps := [⟨s!"b{p}dW", nDW, [ic,1,3,3]⟩] ++
                (if convBias then [⟨s!"b{p}db", nDb, [ic]⟩] else []) ++
                [⟨s!"b{p}dg", nDg, [ic]⟩, ⟨s!"b{p}dbt", nDt, [ic]⟩, ⟨s!"b{p}pW", nPW, [oc,ic,1,1]⟩] ++
                (if convBias then [⟨s!"b{p}pb", nPb, [oc]⟩] else []) ++
                [⟨s!"b{p}pg", nPg, [oc]⟩, ⟨s!"b{p}pbt", nPt, [oc]⟩] }

-- ════════════════════════════════════════════════════════════════
-- § Signatures — ONE source for the arg order, the return types and the AdamW slots
-- ════════════════════════════════════════════════════════════════

/-- A 12-param inverted-residual block's signature, func-arg order. -/
private def mnv2BlockSig (i : String) (ic mid oc : Nat) (convBias : Bool) : List (String × String) :=
  let b (nm : String) (c : Nat) : List (String × String) := if convBias then [(nm, ty [c])] else []
  [(s!"%b{i}eW", ty [mid,ic,1,1])] ++ b s!"%b{i}eb" mid ++
  [(s!"%b{i}eg", ty [mid]), (s!"%b{i}ebt", ty [mid]),
   (s!"%b{i}dW", ty [mid,1,3,3])] ++ b s!"%b{i}db" mid ++
  [(s!"%b{i}dg", ty [mid]), (s!"%b{i}dbt", ty [mid]),
   (s!"%b{i}pW", ty [oc,mid,1,1])] ++ b s!"%b{i}pb" oc ++
  [(s!"%b{i}pg", ty [oc]), (s!"%b{i}pbt", ty [oc])]

/-- The no-expand block's signature (b1): depthwise on `ic` channels + project `ic→oc`. 8 params. -/
private def mnv2NoExpSig (i : String) (ic oc : Nat) (convBias : Bool) : List (String × String) :=
  let b (nm : String) (c : Nat) : List (String × String) := if convBias then [(nm, ty [c])] else []
  [(s!"%b{i}dW", ty [ic,1,3,3])] ++ b s!"%b{i}db" ic ++
  [(s!"%b{i}dg", ty [ic]), (s!"%b{i}dbt", ty [ic]),
   (s!"%b{i}pW", ty [oc,ic,1,1])] ++ b s!"%b{i}pb" oc ++
  [(s!"%b{i}pg", ty [oc]), (s!"%b{i}pbt", ty [oc])]

/-- **The 210 parameters of the 17-block paper-spec net, in func-arg order.** stem (4) + b1
    no-expand (8) + b2..b17 (16 × 12 = 192) + head (4) + dense (2) = 210 — the same decomposition
    `MobileNetV2Render.paperSig` uses, at the names the committed AdamW artifact presents. -/
def mnv2SigList (nClasses : Nat) (convBias : Bool) : List (String × String) :=
  [("%sW", ty [32,3,3,3])] ++ (if convBias then [("%sb", ty [32])] else []) ++
  [("%sg", ty [32]), ("%sbt", ty [32])] ++
  mnv2NoExpSig "1" 32 16 convBias ++
  mnv2BlockSig "2"  16  96  24 convBias ++ mnv2BlockSig "3"  24 144  24 convBias ++
  mnv2BlockSig "4"  24 144  32 convBias ++ mnv2BlockSig "5"  32 192  32 convBias ++
  mnv2BlockSig "6"  32 192  32 convBias ++ mnv2BlockSig "7"  32 192  64 convBias ++
  mnv2BlockSig "8"  64 384  64 convBias ++ mnv2BlockSig "9"  64 384  64 convBias ++
  mnv2BlockSig "10" 64 384  64 convBias ++ mnv2BlockSig "11" 64 384  96 convBias ++
  mnv2BlockSig "12" 96 576  96 convBias ++ mnv2BlockSig "13" 96 576  96 convBias ++
  mnv2BlockSig "14" 96 576 160 convBias ++ mnv2BlockSig "15" 160 960 160 convBias ++
  mnv2BlockSig "16" 160 960 160 convBias ++ mnv2BlockSig "17" 160 960 320 convBias ++
  [("%hW", ty [1280,320,1,1])] ++ (if convBias then [("%hb", ty [1280])] else []) ++
  [("%hg", ty [1280]), ("%hbt", ty [1280])] ++
  [("%Wd", ty [1280, nClasses]), ("%bd", ty [nClasses])]

/-- The running-mean/var slots for one BN layer. -/
private def bnStatSig (nm : String) (c : Nat) : List (String × String) :=
  [(s!"%{nm}mu", ty [c]), (s!"%{nm}var", ty [c])]

/-- A 3-BN block's stat slots (expand-BN, depthwise-BN, project-BN). -/
private def blockStatSig (i : String) (mid oc : Nat) : List (String × String) :=
  bnStatSig s!"b{i}en" mid ++ bnStatSig s!"b{i}dn" mid ++ bnStatSig s!"b{i}pn" oc

/-- **The 104 BN running-statistic slots** = 52 BN layers × (μ, var): stem 1, b1 two (no expand
    BN), b2..b17 three each (48), head 1. Both an input (`…i`) and an output slot. -/
def mnv2StatSigList : List (String × String) :=
  bnStatSig "stn" 32 ++
  (bnStatSig "b1dn" 32 ++ bnStatSig "b1pn" 16) ++
  blockStatSig "2"  96  24 ++ blockStatSig "3" 144  24 ++
  blockStatSig "4" 144  32 ++ blockStatSig "5" 192  32 ++
  blockStatSig "6" 192  32 ++ blockStatSig "7" 192  64 ++
  blockStatSig "8" 384  64 ++ blockStatSig "9" 384  64 ++
  blockStatSig "10" 384  64 ++ blockStatSig "11" 384  96 ++
  blockStatSig "12" 576  96 ++ blockStatSig "13" 576  96 ++
  blockStatSig "14" 576 160 ++ blockStatSig "15" 960 160 ++
  blockStatSig "16" 960 160 ++ blockStatSig "17" 960 320 ++
  bnStatSig "hn" 1280

-- ════════════════════════════════════════════════════════════════
-- § The AdamW tail — one proven triple per parameter, folded in signature order
-- ════════════════════════════════════════════════════════════════

/-- `(θ', m', v')` for one parameter, from its un-fused gradient: the proven
    `adamMNextF`/`adamVNextF`/`adamWParamF` triple (`adamW_triple_faithful` bundles their `den`s
    into `Proofs.adamWStep` by `rfl`). β₁/β₂/ε/wd are baked; `%lr`/`%bc1`/`%bc2` are runtime
    `tensor<f32>` args, so one render serves a whole LR schedule.

    At `replicas > 1` the gradient is first averaged by `ViTRender.emitGradAllReduce`. **That
    collective is a TRUSTED CARVE-OUT** (handoff §5) — emitted text, not `pretty` of an AST node,
    so outside every faithfulness theorem here. The AdamW triple consumes the averaged gradient as
    an `.operand` exactly as it consumed the raw one, so the `den` side does not shift. At
    `replicas ≤ 1` this emits nothing and threads the raw gradient, so the single-device render
    stays byte-identical — the cheap self-check that the insertion is inert. -/
private def adamOneM (B : Nat) (replicas : Nat) (g : PGradM) :
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

/-- `(θ', b', s')` for one parameter under **RMSProp with momentum** — the `adamOneM` peer.

    Only ONE of the four ops is new. Reading the reference
    (`jax/Jax/Codegen.lean`, the `.rmsprop` branch) top to bottom:

    | reference line | emitted here |
    |---|---|
    | `grads = g + WD * p` | `momVNextF` at `(μ := wd, v := θ)` — `Proofs.momVNext_as_coupled_l2` |
    | `sq = RHO*s + (1-RHO)*g*g` | **`adamVNextF` at `β₂ := ρ`** — `Proofs.rmsSqNext_eq_adamVNext` |
    | `buf = MOMENTUM*b + g/sqrt(sq+EPS)` | `rmsBufNextF` — the new op, ε INSIDE the root |
    | `params = p - lr*buf` | `sgdParamF` on the buffer's SSA |

    ⚠ **The weight decay is COUPLED and goes FIRST**, so the accumulator sees the decayed gradient.
    Reversing that order — decaying after the accumulator, AdamW-style — is a different optimizer
    and would not show up as an arity or type error anywhere.

    Slot mapping: the packed `[θ|m|v]` signature is reused verbatim with **`m` carrying the
    momentum buffer and `v` the running mean-square**, the same slot reinterpretation the Nesterov
    render does for its velocity. That is why the driver and the interface do not move. -/
private def rmsOneM (B : Nat) (replicas : Nat) (g : PGradM) :
    StateM Nat (String × String × String × String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let (arS, gAvg) := ViTRender.emitGradAllReduce g.grad g.ds g.nm replicas
  let (cW, nW) ← pretty B (.momVNextF s!"%{g.nm}" "%wd" g.ds 0 z (.operand gAvg z))
  let gr : SHlo n := .operand nW z
  let (cS, nS) ← pretty B (.adamVNextF s!"%{g.nm}v" "%rho" "%orho" g.ds 0 z gr)
  let (cB, nB) ← pretty B (.rmsBufNextF s!"%{g.nm}v" s!"%{g.nm}m" "%rho" "%orho" "%mu" "%eps"
                    g.ds 0 0 0 z z gr)
  -- θ' threads b' by SSA NAME, not by re-nesting `rmsBufNextF` inside `sgdParamF`: `pretty` has no
  -- CSE (§4), so re-nesting would emit the whole 13-op buffer block a second time.
  let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nB z))
  pure (arS ++ cW ++ cS ++ cB ++ cT, nT, nB, nS)

/-- β₁/β₂/ε/wd as graph constants — the committed MobileNetV2 AdamW recipe, identical to
    ResNet-34's and read off `verified_mlir/mobilenetv2_adam_train_step.mlir`. -/
private def adamConstsM : String :=
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"

/-- The driver's **variant slug** for a given `(B, replicas)`: the artifact is
    `verified_mlir/mobilenetv2_<variant>_train_step.mlir`, the entry point is
    `@mobilenetv2_<variant>_train_step`, and `LEAN_MLIR_VARIANT` selects it. All three must agree —
    the shim checks the entry name and refuses a mismatch outright ("entry mismatch") rather than
    running the wrong graph. `B = 32` is deliberately unsuffixed so the committed artifact keeps its
    name. The `#guard`s at the bottom pin the literal `#eval` paths against this. -/
def mnv2AdamVariant (B replicas : Nat) (opt : OptKind := .adamw)
    -- ▶ `bf16` LAST and defaulted, so every committed spelling is untouched.
    -- ⚠⚠ It MUST reach this function and not merely the block renderers: the entry NAME is
    -- derived from the variant, so a flag that reaches the emission but not the name writes
    -- `…bf16_train_step.mlir` declaring `@…_train_step` inside, and the driver refuses at load
    -- with an entry mismatch. ConvNeXt shipped that twice and R34's bf16 a third time.
    (bf16 : Bool := false) : String :=
  (match opt with
   | .adamw   => if replicas ≤ 1 then "adam" else "adamdp"
   | .rmsprop => if replicas ≤ 1 then "rms"  else "rmsdp") ++
  (if B == 32 then "" else toString B) ++
  (if bf16 then "bf16" else "")

-- ════════════════════════════════════════════════════════════════
-- § The whole-net batched AdamW train step
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000000 in
/-- **MobileNetV2 (17-block paper spec) AdamW train step, batch BN, rendered from the verified AST
    at `N := B`.** 739 inputs (`%x`, 210 θ, 210 m, 210 v, `%lr`/`%bc1`/`%bc2`, 104 running-stat
    slots, `%onehot`) and 737 outputs (210 θ', 210 m', 210 v', `%loss`/`%bc1`/`%bc2`, 104 batch
    stats) — the interface the committed hand-written artifact already presents, so the driver is
    unchanged. Parameter ORDER comes from `mnv2SigList` and stat order from `mnv2StatSigList`, the
    single sources, so the arity/order contract cannot drift.

    Stem 3×3/s2 (3→32, 224→112, NO maxpool) → b1 (no-expand `t=1`, 32→16) → b2..b17 (4 stride-2
    downsamples, 10 identity skips, 2 stage-first widenings) → 1×1 conv-BN-relu6 head (320→1280) →
    GAP → dense (1280→nClasses). -/
def mobilenetv2AdamTrainStepFaithfulB (B nClasses : Nat) (epsStr : String)
    (replicas : Nat := 1) (convBias : Bool := false) (slug : String := "mobilenetv2")
    (opt : OptKind := .adamw)
    -- ⭐⭐ **bf16**, TRAILING and defaulted so every existing render is byte-identical (gate 1).
    -- Every conv AND every DEPTHWISE — stem, expand/project 1×1s, the depthwise, both dgrads,
    -- every wgrad — becomes its bf16 twin: bf16 operands, a **bf16-TYPED** result, convert back.
    -- BN, the bias grads, the loss, AdamW and the master weights stay f32.
    -- ⚠ This is the first net whose bf16 path includes GROUPED convolutions. The f32-result
    -- shape folds for those exactly as it does for ordinary conv — measured on a real MNv2
    -- layer before the ops were written — so `feature_group_count` buys no exemption from §9.2.
    (bf16 : Bool := false) : String :=
  -- ⚠ α and K are spelled ONCE here. Until 2026-08-02 this render carried `0.100000` and
  -- `-0.010000` as inline literals in the cotangent AND a third copy, `0.010000`, in the
  -- report-only loss — the K=10 values. mnv2 is the WORST of the four ImageNet ports on this axis
  -- because one of them is on the GRADIENT path, which is §2k's original bug rather than the
  -- report-only variant found in ConvNeXt and EfficientNet. At K=10 all three render byte-identical
  -- to the literals they replace, so the fix is inert on every committed artifact.
  let alphaStr    := fmt6 0.1                 -- α itself ("0.100000")
  let negAlphaKStr := "-" ++ alphaOverK nClasses 0.1
  let go : StateM Nat String := do
    -- ▶ Placeholder rounding, exactly as the `z*` zero kernels are — see `irFwdStridedB`.
    let zrnd : ℝ → ℝ := fun r => r
    -- ═══ stem: 3×3/s2 conv (3→32, 224→112) → batch BN → relu6 (NO maxpool) ═══
    let zx    : Vec (B*(3*224*224)) := fun _ => 0
    let zSk   : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let z32   : Vec 32 := fun _ => 0
    let z112  : Vec (B*(32*112*112)) := fun _ => 0
    let z112p : Vec (B*(32*(112*112))) := fun _ => 0
    let (cStc, nStc) ← pretty B (.batchOp (N := B)
      (if bf16 then .convStridedXlaBf16 (ic := 3) (oc := 32) (h := 112) (w := 112) zrnd "%sW" (biasName convBias "%sb" 32) zSk z32 else .convStridedXla (ic := 3) (oc := 32) (h := 112) (w := 112) "%sW" (biasName convBias "%sb" 32) zSk z32)
      (.operand "%x" zx))
    let (cStn, nStn) ← pretty B (.bnBatchF (N := B) (oc := 32) (h := 112) (w := 112)
      "%sg" "%sbt" epsStr 0 z32 z32 (.operand nStc z112))
    let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu6 (n := 32*112*112)) (.operand nStn z112))
    -- ═══ forward: the 17 inverted-residual blocks ═══
    let f1  ← irFwdNoExpB   B 32      16 112 epsStr "1"  nStr convBias bf16
    let f2  ← irFwdStridedB B 16  96  24  56 epsStr "2"  f1.o convBias bf16
    let f3  ← irFwdSkipB    B 24 144  24  56 epsStr "3"  f2.o convBias bf16
    let f4  ← irFwdStridedB B 24 144  32  28 epsStr "4"  f3.o convBias bf16
    let f5  ← irFwdSkipB    B 32 192  32  28 epsStr "5"  f4.o convBias bf16
    let f6  ← irFwdSkipB    B 32 192  32  28 epsStr "6"  f5.o convBias bf16
    let f7  ← irFwdStridedB B 32 192  64  14 epsStr "7"  f6.o convBias bf16
    let f8  ← irFwdSkipB    B 64 384  64  14 epsStr "8"  f7.o convBias bf16
    let f9  ← irFwdSkipB    B 64 384  64  14 epsStr "9"  f8.o convBias bf16
    let f10 ← irFwdSkipB    B 64 384  64  14 epsStr "10" f9.o convBias bf16
    let f11 ← irFwdNoSkipB  B 64 384  96  14 epsStr "11" f10.o convBias bf16
    let f12 ← irFwdSkipB    B 96 576  96  14 epsStr "12" f11.o convBias bf16
    let f13 ← irFwdSkipB    B 96 576  96  14 epsStr "13" f12.o convBias bf16
    let f14 ← irFwdStridedB B 96 576 160   7 epsStr "14" f13.o convBias bf16
    let f15 ← irFwdSkipB    B 160 960 160  7 epsStr "15" f14.o convBias bf16
    let f16 ← irFwdSkipB    B 160 960 160  7 epsStr "16" f15.o convBias bf16
    let f17 ← irFwdNoSkipB  B 160 960 320  7 epsStr "17" f16.o convBias bf16
    -- ═══ head: 1×1 conv (320→1280) → batch BN → relu6 → GAP(7×7) → dense ═══
    let z7     : Vec (B*(320*7*7)) := fun _ => 0
    let zHk    : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
    let z1280  : Vec 1280 := fun _ => 0
    let zH7    : Vec (B*(1280*7*7)) := fun _ => 0
    let zH7p   : Vec (B*(1280*(7*7))) := fun _ => 0
    let z1280b : Vec (B*1280) := fun _ => 0
    let zWd    : Mat 1280 nClasses := fun _ _ => 0
    let zNC    : Vec nClasses := fun _ => 0
    let zNCb   : Vec (B*(1*nClasses)) := fun _ => 0
    let zNCp   : Vec (B*nClasses) := fun _ => 0
    let (cHc, nHc) ← pretty B (.batchOp (N := B)
      (if bf16 then .convBf16 (ic := 320) (oc := 1280) (h := 7) (w := 7) zrnd "%hW" (biasName convBias "%hb" 1280) zHk z1280 else .conv (ic := 320) (oc := 1280) (h := 7) (w := 7) "%hW" (biasName convBias "%hb" 1280) zHk z1280) (.operand f17.o z7))
    let (cHn, nHn) ← pretty B (.bnBatchF (N := B) (oc := 1280) (h := 7) (w := 7)
      "%hg" "%hbt" epsStr 0 z1280 z1280 (.operand nHc zH7))
    let (cHr, nHr) ← pretty B (.batchOp (N := B) (.relu6 (n := 1280*7*7)) (.operand nHn zH7))
    let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 1280) (h := 7) (w := 7))
      (.operand nHr zH7))
    let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC)
      (.operand nGap z1280b))
    -- ═══ label-smoothed softmax-CE cotangent, COMPOSED from kit ops (α = 0.1, K = nClasses):
    --     dy = (softmax(logits) − onehot + α·onehot − α/K) / B. Every line is a verified node;
    --     the hand-written render fuses this into one [B,K] block, so the two graphs differ. ═══
    let (cSm,  nSm)  ← pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses))
      (.operand nLog zNCb))
    let (cD0,  nD0)  ← pretty B (.subB (.operand nSm zNCb) (.operand "%onehot" zNCb))
    let (cLsa, nLsa) ← pretty B (.scaleB alphaStr 0 (.operand "%onehot" zNCb))
    let (cD1,  nD1)  ← pretty B (.addVB (.operand nD0 zNCb) (.operand nLsa zNCb))
    let (cD2,  nD2)  ← pretty B (.shiftB negAlphaKStr 0 (.operand nD1 zNCb))
    let (cDy,  nDy)  ← pretty B (.divConstB s!"{B}.0" 0 (.operand nD2 zNCb))
    -- ═══ head backward + the 6 head/dense gradients ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (N := B)
      (.denseRowBack (rows := 1) (a := 1280) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    let (cWdg, nWdg) ← pretty B (.denseWeightGradB (c := nClasses) nGap z1280b (.operand nDy zNCp))
    let (cbdg, nbdg) ← pretty B (.denseBiasGradB (N := B) (.operand nDy zNCp))
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 1280) (h := 7) (w := 7)
      (.operand nDgi z1280b))
    let (cDhm, nDhm) ← pretty B (.selectMidB nHn zH7 (.operand nDgp zH7))
    let (cDhn, nDhn) ← pretty B (.bnBatchBack (N := B) (oc := 1280) (h := 7) (w := 7)
      "%hg" nHc epsStr 0 z1280 zH7p (.operand nDhm zH7p))
    let (cDhx, nDhx) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7) zrnd
      "%hW" zHk z1280 (.operand nDhn zH7) else .convBackBatched (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7)
      "%hW" zHk z1280 (.operand nDhn zH7))
    let (cHW, nHW) ← pretty B (if bf16 then .convWeightGradBBf16 (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7) zrnd
      f17.o z1280 z7 zHk (.operand nDhn zH7) else .convWeightGradB (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7)
      f17.o z1280 z7 zHk (.operand nDhn zH7))
    let (cHb, nHb) ← if convBias then
        pretty B (.convBiasGradB (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7)
            zHk z7 z1280 (.operand nDhn zH7))
      else pure ("", "")
    let (cHg, nHg) ← pretty B (.bnGammaGradB (N := B) (oc := 1280) (h := 7) (w := 7)
      nHc epsStr 0 zH7p (.operand nDhm zH7p))
    let (cHt, nHt) ← pretty B (.bnBetaGradB (N := B) (oc := 1280) (h := 7) (w := 7)
      (.operand nDhm zH7p))
    -- ═══ backward: the 17 blocks reversed (the cotangent threads from nDhx) ═══
    let b17 ← irBackStride1GradB B 160 960 320  7 false epsStr "17" f16.o f17 nDhx convBias bf16
    let b16 ← irBackStride1GradB B 160 960 160  7 true  epsStr "16" f15.o f16 b17.dx convBias bf16
    let b15 ← irBackStride1GradB B 160 960 160  7 true  epsStr "15" f14.o f15 b16.dx convBias bf16
    let b14 ← irBackStridedGradB B 96 576 160   7       epsStr "14" f13.o f14 b15.dx convBias bf16
    let b13 ← irBackStride1GradB B 96 576  96  14 true  epsStr "13" f12.o f13 b14.dx convBias bf16
    let b12 ← irBackStride1GradB B 96 576  96  14 true  epsStr "12" f11.o f12 b13.dx convBias bf16
    let b11 ← irBackStride1GradB B 64 384  96  14 false epsStr "11" f10.o f11 b12.dx convBias bf16
    let b10 ← irBackStride1GradB B 64 384  64  14 true  epsStr "10" f9.o  f10 b11.dx convBias bf16
    let b9  ← irBackStride1GradB B 64 384  64  14 true  epsStr "9"  f8.o  f9  b10.dx convBias bf16
    let b8  ← irBackStride1GradB B 64 384  64  14 true  epsStr "8"  f7.o  f8  b9.dx convBias bf16
    let b7  ← irBackStridedGradB B 32 192  64  14       epsStr "7"  f6.o  f7  b8.dx convBias bf16
    let b6  ← irBackStride1GradB B 32 192  32  28 true  epsStr "6"  f5.o  f6  b7.dx convBias bf16
    let b5  ← irBackStride1GradB B 32 192  32  28 true  epsStr "5"  f4.o  f5  b6.dx convBias bf16
    let b4  ← irBackStridedGradB B 24 144  32  28       epsStr "4"  f3.o  f4  b5.dx convBias bf16
    let b3  ← irBackStride1GradB B 24 144  24  56 true  epsStr "3"  f2.o  f3  b4.dx convBias bf16
    let b2  ← irBackStridedGradB B 16  96  24  56       epsStr "2"  f1.o  f2  b3.dx convBias bf16
    let b1  ← irBackNoExpGradB   B 32      16 112       epsStr "1"  nStr  f1  b2.dx convBias bf16
    -- ═══ stem backward: relu6 mask → BN back, then the 4 stem gradients (NO conv-back past %x) ═══
    let (cDsm, nDsm) ← pretty B (.selectMidB nStn z112 (.operand b1.dx z112))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := B) (oc := 32) (h := 112) (w := 112)
      "%sg" nStc epsStr 0 z32 z112p (.operand nDsm z112p))
    let (csW, nsW) ← pretty B (if bf16 then .convStridedXlaWeightGradBBf16 zrnd "%x" z32 zx zSk (.operand nDsn z112) else .convStridedXlaWeightGradB "%x" z32 zx zSk (.operand nDsn z112))
    let (csb, nsb) ← if convBias then
        pretty B (.convStridedXlaBiasGradB (h := 112) (w := 112) zSk zx z32
            (.operand nDsn z112))
      else pure ("", "")
    let (csg, nsg) ← pretty B (.bnGammaGradB (N := B) (oc := 32) (h := 112) (w := 112)
      nStc epsStr 0 z112p (.operand nDsm z112p))
    let (cst, nst) ← pretty B (.bnBetaGradB (N := B) (oc := 32) (h := 112) (w := 112)
      (.operand nDsm z112p))
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT.
    --     Derived from the SAME forward record that computes them (`f.ec`/`f.dc`/`f.pc`) rather
    --     than from an independent 52-entry table — a misaligned stat slot is SILENT, since the
    --     arities still match and the wrong layer's statistics simply flow into the wrong
    --     `@mobilenetv2_fwd_eval` slot (§2e). ═══
    let bnStat (oc hh : Nat) (xn : String) : StateM Nat (String × String × String) := do
      let zb : Vec (B*(oc*(hh*hh))) := fun _ => 0
      let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh)
        (.operand xn zb))
      let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh)
        (.operand xn zb))
      pure (cM ++ cV, nM, nV)
    -- a STRIDED block's expand BN sits at the INPUT resolution 2hh; everything else at hh
    let blkStatsS (mid oc hh : Nat) (f : MBFwdB) : StateM Nat (String × List String) := do
      let (ce, me, ve) ← bnStat mid (2*hh) f.ec
      let (cd, md, vd) ← bnStat mid hh f.dc
      let (cp, mp, vp) ← bnStat oc hh f.pc
      pure (ce ++ cd ++ cp, [me, ve, md, vd, mp, vp])
    let blkStats1 (mid oc hh : Nat) (f : MBFwdB) : StateM Nat (String × List String) := do
      let (ce, me, ve) ← bnStat mid hh f.ec
      let (cd, md, vd) ← bnStat mid hh f.dc
      let (cp, mp, vp) ← bnStat oc hh f.pc
      pure (ce ++ cd ++ cp, [me, ve, md, vd, mp, vp])
    let (cQ0, m0, v0) ← bnStat 32 112 nStc
    -- b1 is the no-expand block: TWO BN layers, not three
    let (cQ1d, m1d, v1d) ← bnStat 32 112 f1.dc
    let (cQ1p, m1p, v1p) ← bnStat 16 112 f1.pc
    let (cQ2,  q2)  ← blkStatsS  96  24 56 f2
    let (cQ3,  q3)  ← blkStats1 144  24 56 f3
    let (cQ4,  q4)  ← blkStatsS 144  32 28 f4
    let (cQ5,  q5)  ← blkStats1 192  32 28 f5
    let (cQ6,  q6)  ← blkStats1 192  32 28 f6
    let (cQ7,  q7)  ← blkStatsS 192  64 14 f7
    let (cQ8,  q8)  ← blkStats1 384  64 14 f8
    let (cQ9,  q9)  ← blkStats1 384  64 14 f9
    let (cQ10, q10) ← blkStats1 384  64 14 f10
    let (cQ11, q11) ← blkStats1 384  96 14 f11
    let (cQ12, q12) ← blkStats1 576  96 14 f12
    let (cQ13, q13) ← blkStats1 576  96 14 f13
    let (cQ14, q14) ← blkStatsS 576 160  7 f14
    let (cQ15, q15) ← blkStats1 960 160  7 f15
    let (cQ16, q16) ← blkStats1 960 160  7 f16
    let (cQ17, q17) ← blkStats1 960 320  7 f17
    let (cQh, mh, vh) ← bnStat 1280 7 nHc
    -- ═══ the 210 parameter gradients in func-arg order ═══
    let stemPs : List PGradM :=
      [⟨"sW", nsW, [32,3,3,3]⟩] ++ (if convBias then [⟨"sb", nsb, [32]⟩] else []) ++
      [⟨"sg", nsg, [32]⟩, ⟨"sbt", nst, [32]⟩]
    let headPs : List PGradM :=
      [⟨"hW", nHW, [1280,320,1,1]⟩] ++ (if convBias then [⟨"hb", nHb, [1280]⟩] else []) ++
      [⟨"hg", nHg, [1280]⟩, ⟨"hbt", nHt, [1280]⟩,
       ⟨"Wd", nWdg, [1280, nClasses]⟩, ⟨"bd", nbdg, [nClasses]⟩]
    let allPs : List PGradM := stemPs ++
      b1.ps ++ b2.ps ++ b3.ps ++ b4.ps ++ b5.ps ++ b6.ps ++ b7.ps ++ b8.ps ++ b9.ps ++
      b10.ps ++ b11.ps ++ b12.ps ++ b13.ps ++ b14.ps ++ b15.ps ++ b16.ps ++ b17.ps ++ headPs
    -- ═══ AdamW: one proven triple per parameter ═══
    let mut adamCode := ""
    let mut thetaN : List String := []
    let mut mNames : List String := []
    let mut vNames : List String := []
    for g in allPs do
      let (c, nT, nM, nV) ← match opt with
        | .adamw   => adamOneM B replicas g
        | .rmsprop => rmsOneM  B replicas g
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]
      mNames := mNames ++ [nM]
      vNames := vNames ++ [nV]
    -- ═══ assemble ═══
    let statCode := cQ0 ++ cQ1d ++ cQ1p ++ cQ2 ++ cQ3 ++ cQ4 ++ cQ5 ++ cQ6 ++ cQ7 ++ cQ8 ++
      cQ9 ++ cQ10 ++ cQ11 ++ cQ12 ++ cQ13 ++ cQ14 ++ cQ15 ++ cQ16 ++ cQ17 ++ cQh
    let statNames : List String :=
      [m0, v0, m1d, v1d, m1p, v1p] ++ q2 ++ q3 ++ q4 ++ q5 ++ q6 ++ q7 ++ q8 ++ q9 ++ q10 ++
      q11 ++ q12 ++ q13 ++ q14 ++ q15 ++ q16 ++ q17 ++ [mh, vh]
    -- `%loss` is REPORT-ONLY: mean smoothed-CE for logging, on no gradient path. It is NOT
    -- `pretty` of an AST node and says so in the emitted text — the same carve-out
    -- `resnet34`/`cifar8`'s `%loss` takes (§5). The SMOOTHED cross-entropy, matching the
    -- cotangent's soft target:
    --   loss = −(1/B)·Σ_b [ (1−α)·Σ_k onehot·log sm  +  (α/K)·Σ_k log sm ].
    -- Getting this wrong is invisible to every proof in the repo, and §2b shipped exactly that bug
    -- on R34 (plain CE against a smoothed cotangent) — only the numeric tie caught it.
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
    let body := cStc ++ cStn ++ cStr ++
      f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++ f8.code ++
      f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++ f15.code ++
      f16.code ++ f17.code ++
      cHc ++ cHn ++ cHr ++ cGap ++ cLog ++ cSm ++ cD0 ++ cLsa ++ cD1 ++ cD2 ++ cDy ++
      cDgi ++ cWdg ++ cbdg ++ cDgp ++ cDhm ++ cDhn ++ cDhx ++ cHW ++ cHb ++ cHg ++ cHt ++
      b17.code ++ b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++
      b10.code ++ b9.code ++ b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++
      b2.code ++ b1.code ++
      cDsm ++ cDsn ++ csW ++ csb ++ csg ++ cst ++ statCode
    let pTypes : List String := allPs.map (fun g => ty g.ds)
    let statTypes : List String := mnv2StatSigList.map (·.2)
    let retVals := thetaN ++ mNames ++ vNames ++ ["%loss", "%bc1", "%bc2"] ++ statNames
    let retTys  := pTypes ++ pTypes ++ pTypes ++
      ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statTypes
    pure <|
      (match opt with
       | .adamw => ""
       | .rmsprop =>
         "    // ── OPTIMIZER: RMSProp + momentum, TENSORFLOW flavour (the MobileNetV2 reference's\n" ++
         "    //    own: jax/MainMobilenetV2Imagenet.lean). Per parameter, in this order:\n" ++
         "    //      g  <- g + wd*θ        COUPLED L2, BEFORE the accumulator  (momVNextF)\n" ++
         "    //      s' <- ρ*s + (1-ρ)*g²                                      (adamVNextF at ρ)\n" ++
         "    //      b' <- μ*b + g/sqrt(s' + ε)   ⚠ ε INSIDE the sqrt          (rmsBufNextF)\n" ++
         "    //      θ' <- θ - lr*b'                                           (sgdParamF)\n" ++
         "    //    Packed [θ|m|v] is reused with m = momentum buffer, v = mean-square, so the\n" ++
         "    //    interface is byte-identical to the AdamW render's apart from the entry name.\n" ++
         "    //    %bc1/%bc2 are Adam bias corrections: unused here, passed through unchanged.\n" ++
         "    //    ⚠ The mean-square must be INITIALISED TO 1.0, not 0 — part of the recipe, not\n" ++
         "    //    an implementation detail, since this optimizer is not bias-corrected.\n") ++
      (if replicas ≤ 1 then
        "    // ── MobileNetV2 batch-BN AdamW train step: every line is pretty(verified AST node) ──\n"
       else
        s!"    // ── MobileNetV2 batch-BN AdamW train step, DATA-PARALLEL over {replicas} replicas ──\n" ++
        "    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`\n" ++
        "    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5), emitted\n" ++
        "    // text outside the faithfulness theorems. Each replica evaluates the same tied graph\n" ++
        "    // at the batch it was rendered for; the collective averages that function's gradients\n" ++
        "    // over disjoint equal batches. NOTE this does NOT equal a single-device step at the\n" ++
        "    // global batch — BN normalises per replica, so N×b != 1×(N·b) by design (§10.3b).\n") ++
      zeroBiasPrelude convBias [16, 24, 32, 64, 96, 128, 144, 160, 192, 256, 320, 384, 576, 960, 1280] ++ body ++
      (match opt with | .adamw => adamConstsM | .rmsprop => rmsConstsBlock mnv2RmsHyper) ++
      adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let sigList : List (String × String) := mnv2SigList nClasses convBias
  let pSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let mSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}m: {t}"))
  let vSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}v: {t}"))
  let statSig := String.intercalate ", " (mnv2StatSigList.map (fun (n, t) => s!"{n}i: {t}"))
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, " ++ statSig ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (·.2)
  let outSig := String.intercalate ", "
    (pTy ++ pTy ++ pTy ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++
     (mnv2StatSigList.map (·.2)))
  let inner : String := go.run' 0
  let fname := s!"{slug}_{mnv2AdamVariant B replicas opt bf16}_train_step"
  "module @m {\n" ++
  s!"  func.func @{fname}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- Regenerate `verified_mlir/mobilenetv2_adam_train_step.mlir` — the batched (`N := B`) MobileNetV2
-- AdamW train step as `pretty(provenGraph)`. B=32, nClasses=10, ε=1e-5. **This is the artifact
-- `mobilenetv2-verified-adam` trains on**, and this `#eval` is its ONLY writer.
--
-- It rendered to a separate `…_b.mlir` while the hand-written emitter in
-- `tests/TestMobilenetV2TrainPC.lean` still owned this path — two writers for one artifact is the
-- last-writer-wins race §2a found. The swap happened (§2f) once the gates were in:
--
--   * the numeric tie (`mobilenetv2-adam-tie`, IREE) — forward BIT-EXACT on all 52 BN layers'
--     batch statistics, `%loss` bit-exact, gradient bit-exact, spread 0/210, over all 6,795,329
--     returned floats, against a bit-exact A-vs-A determinism floor;
--   * that tie VERIFIED TO FAIL, three ways: a perturbed cotangent fires the gradient gate
--     (spread 111/210) with the forward still bit-exact, a perturbed BN ε fires the forward gate
--     (`bnstat` exact only 80/34112), and a perturbed `%loss` constant fires the loss gate with
--     every other region bit-exact.
--
-- The driver needed no change: it resolves the path from the net slug, so taking over the
-- canonical name IS the swap. `…_b.mlir` is deleted; the bytes now at this path are byte-identical
-- to the `_b.mlir` render that passed the tie (checked before deleting).
#eval IO.FS.writeFile "verified_mlir/mobilenetv2_adam_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 32 10 "1.0e-5")

-- The **DATA-PARALLEL** render (handoff §2h-bis), selected at run time by
-- `LEAN_MLIR_VARIANT=adamdp`. Same graph, plus one `all_reduce(add)/N` per parameter gradient
-- between the certified gradient and the certified AdamW triple: *certified gradient → trusted
-- collective → certified AdamW*. The collective is a DECLARED carve-out and the render says so in
-- its own output banner at `replicas > 1`, per the §5/§2b `%loss` lesson that an undeclared
-- carve-out is how wrong things ship. Claim ceiling is unchanged (§5): the gradient averaging is a
-- proven identity; the collective implementing it is trusted, exactly like the lowerer.
--
-- Like EfficientNet's and unlike ResNet-34's, this variant never had a hand-written emitter to
-- migrate off — this file has been the only writer of both mnv2 AdamW artifacts since the §2f swap.
--
-- It renders to its OWN path, which is what stops the §2a race where producing a DP render meant
-- editing a knob and clobbering the artifact the trainer runs. `2` is the replica count these are
-- rendered at and it must match `PJRT_REPLICAS` at run time, because the graph bakes
-- `replica_groups`. Re-render here to change it.
--
-- It needs the XLA build (`mobilenetv2-verified-adam`, §2h): collectives exist only on the PJRT
-- path, and the IREE shim refuses a DP entry point outright rather than silently running
-- single-device.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2_adamdp_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 32 10 "1.0e-5" 2)

-- ── RMSProp at the IMAGENETTE shape (B=32, K=10) ──────────────────────────────────────────────
-- Rendered deliberately at the shape the existing gates and trainer exercise TODAY, exactly as
-- §2k did for ResNet-34's heavy-ball `mom` variant before the ImageNet one existed. The ImageNet
-- renders below are the same renderer at `B := 64, nClasses := 1000` — both are true renderer
-- parameters — so anything this shape establishes about the OPTIMIZER carries, and this is the one
-- that `mobilenetv2-adam-tie` and the Imagenette trainer can compile and run without the shim.
--
-- It renders to its OWN path, so the artifact `mobilenetv2-verified-adam` runs is untouched.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2_rms_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 32 10 "1.0e-5" 1 false "mobilenetv2" .rmsprop)

-- ── MobileNetV2 / ImageNet-1k train steps, slug `mobilenetv2in` ──────────────────────────────────────
-- Batch 64 x 4 replicas = global 256 = `mobilenetV2ImagenetConfig.batchSize`, so the step count
-- per epoch (5004) matches the reference exactly. All three label-smoothing constants are derived
-- from `nClasses` as of this change; at K=1000 the cotangent shift is -0.000100 and the loss's
-- α/K is 0.000100, where both were the K=10 value before.
--
-- ⚠ `mnv2AdamVariant B replicas` encodes the PER-DEVICE batch, not the replica count, so
-- `adamdp64` would name both a 2- and a 4-replica render at B=64. Only the 4-replica one exists.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_adam64_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-5" 1 false "mobilenetv2in")

-- ⭐ **The SINGLE-DEVICE bf16 peer**, and it exists to answer one question the 4-replica numbers
-- cannot: **how much of the bf16 win does the f32 all-reduce eat?** `adamdp64bf16` measured 1.37×
-- at 4 replicas while MNv4 — which renders no DP variant at all — measured 1.88× at 1. Those two
-- differ in BOTH architecture and replica count, so neither explains the other. This render holds
-- the architecture fixed and moves only the replica count. ▶ Not a recipe; a control.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_adam64bf16_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-5" 1 false "mobilenetv2in"
    Proofs.StableHLO.OptKind.adamw true)
#guard Proofs.StableHLO.mnv2AdamVariant 64 1 Proofs.StableHLO.OptKind.adamw true == "adam64bf16"
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_adamdp64_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-5" 4 false "mobilenetv2in")

-- ⭐⭐ **The bf16 peer** — `adamdp64bf16`, the same graph with every convolution AND every
-- DEPTHWISE replaced by its bf16 twin: bf16 operands, a **bf16-TYPED** result, then a convert
-- back to f32. Same 4×64 config as the f32 render above, so the two differ by precision alone.
--
-- ⭐ **This is the first verified net whose bf16 path includes GROUPED convolutions.** Before the
-- eight new ops were written, a hand-built StableHLO module at a real MNv2 depthwise layer
-- (c=144, 56², 3×3, fgc=144) was compiled three ways: f32 → f32 operands; bf16 operands with an
-- **f32-typed result** → **FOLDED back to f32**; bf16 operands with a bf16-typed result and a
-- convert → bf16 reaches the hardware. ▶ Identical to the ordinary-conv finding in §9.2, so
-- `feature_group_count` buys no exemption. Check with `scripts/bf16_gate2.py`, never by grepping
-- the op line, which shows only the result type.
--
-- ⚠ The depthwise convs are ~13% of MNv2's step and bf16 is a mild LOSS on them in isolation
-- (0.86× at MNv2's own layers — cuDNN has a better f32 depthwise kernel on Ada). The win comes
-- from the 1×1 expand/project convs, which is why the reference sets `bf16Conv := true` here.
-- ▶ `planning/bf16_renderer.md` §9.1: the doc's "depthwise nets won't pay" was REFUTED.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_adamdp64bf16_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-5" 4 false "mobilenetv2in"
    Proofs.StableHLO.OptKind.adamw true)

-- ⭐ The bf16 marker, and the wiring that actually breaks: `mobilenetv2AdamTrainStepFaithfulB`
-- derives its entry name from `mnv2AdamVariant`, so `bf16` must reach THAT call.
#guard Proofs.StableHLO.mnv2AdamVariant 64 4 Proofs.StableHLO.OptKind.adamw true == "adamdp64bf16"
#guard Proofs.StableHLO.mnv2AdamVariant 64 4 Proofs.StableHLO.OptKind.adamw == "adamdp64"
-- ▶ And the slug must not trip the DRIVER's substring variant predicates, which read the same
-- string to size the checkpoint blob. `cdOn` tests for "do" — a false positive would silently add
-- a dropout region to the layout with no error anywhere.
#guard ("adamdp64bf16".splitOn "do").length == 1
#guard ("adamdp64bf16".splitOn "acc").length == 1
#guard !"adamdp64bf16".startsWith "ema"

-- ── ▶ RMSProp: the optimizer the MobileNetV2 reference ACTUALLY USES ──────────────────────────
-- `planning/recipe_gaps.md` §2: RMSProp is the ONLY gap between this net and the JAX reference's
-- **68.33%** (everything else — batch 256, 90 epochs, 5-epoch warmup, no label smoothing — already
-- matches). recipe_gaps files this as Tier D, "a new proven `SHlo` op family, ten sites each";
-- measured, it is **one** op: `momVNextF` already spells the coupled L2 and `adamVNextF` at
-- `β₂ := ρ` already IS the running mean-square (`Proofs.rmsSqNext_eq_adamVNext`, by `rfl`), so only
-- the ε-inside-the-root normalise had to be built.
--
-- Same shape/batch/replicas as the `adam64` peer above, so the two are comparable row for row: the
-- signature is byte-identical apart from the entry name, and `%bc1`/`%bc2` ride through unused.
--
-- ⚠ THE DRIVER OWES TWO THINGS BEFORE THIS TRAINS CORRECTLY, and neither is a render change:
--   1. the mean-square slot (`v`) must be **initialised to 1.0, not 0** — TF's convention, and the
--      reason this optimizer trains stably at the paper LR. A zero init is not a crash, it is a
--      different and much larger first step;
--   2. exponential LR decay (0.98/epoch), which is recipe_gaps' Tier C and still open.
-- Until both land this artifact is a correct render of the right optimizer, not a matched pair.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_rms64_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-5" 1 false "mobilenetv2in" .rmsprop)
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_rmsdp64_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-5" 4 false "mobilenetv2in" .rmsprop)
-- The **2-GPU** peer of the line above: `B := 128` per replica, so the global batch is still
-- 128×2 = 256 and the recipe, the steps/epoch and the LR all stay exactly what the 4×64 config
-- runs. That is what makes a 2-card wall-clock comparable to the 4-card one rather than a new
-- experiment. The batch is in the slug (`rmsdp128` vs `rmsdp64`), so the two artifacts cannot
-- collide even though both are `dp` renders of the same optimizer.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_rmsdp128_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 128 1000 "1.0e-5" 2 false "mobilenetv2in" .rmsprop)

-- The entry name, the artifact path and `LEAN_MLIR_VARIANT` must agree or the shim refuses the
-- call ("entry mismatch"). These pin the literal path above against `mnv2AdamVariant`, so a rename
-- fails at `lake build` rather than at run time.
#guard Proofs.StableHLO.mnv2AdamVariant 32 1 == "adam"
#guard Proofs.StableHLO.mnv2AdamVariant 32 2 == "adamdp"
#guard Proofs.StableHLO.mnv2AdamVariant 128 1 == "adam128"
-- The RMSProp peers. Distinct slugs from the AdamW ones is the whole point: rendering the other
-- optimizer must never be able to overwrite the artifact the AdamW trainer runs (§2a's
-- last-writer-wins race, which is how `resnet34_train_step` ended up with two writers computing
-- genuinely different functions).
#guard Proofs.StableHLO.mnv2AdamVariant 64 1 .rmsprop == "rms64"
#guard Proofs.StableHLO.mnv2AdamVariant 64 4 .rmsprop == "rmsdp64"
#guard Proofs.StableHLO.mnv2AdamVariant 128 2 .rmsprop == "rmsdp128"
#guard Proofs.StableHLO.mnv2AdamVariant 64 1 .adamw   == "adam64"
-- The interface contract, checked at elaboration: 210 parameters and 104 BN stat slots ⇒
-- 1 + 3×210 + 3 + 104 + 1 = 739 inputs and 3×210 + 3 + 104 = 737 outputs.
#guard (Proofs.StableHLO.mnv2SigList 10 true).length == 210   -- with conv biases (pre-swap)
-- 210 − 52 conv biases = 158. Both are pinned, so dropping the biases cannot silently
-- change the arity of the render that ships (§2m).
#guard (Proofs.StableHLO.mnv2SigList 10 false).length == 158
#guard Proofs.StableHLO.mnv2StatSigList.length == 104
