import LeanMlir.Proofs.Codegen.StableHLOPretty
import LeanMlir.Proofs.Codegen.SyncBnSites
import LeanMlir.Proofs.Codegen.RenderKit

/-! # MobileNetV2 rendered from the verified AST, at the BATCHED index — the SOLE renderer

⭐⭐ **This file writes every MobileNetV2 artifact** as of 2026-09-06, when leg 2 of
`planning/archive/renderer_convergence.md` retired `MobileNetV2Render.lean`. Before that the net had two
renderers and they were two different functions:

* **`MobileNetV2Render.lean` rendered PER-EXAMPLE BN** (`bnPerChannelF`, reduce `[2,3]`) and wrote
  the SGD-inline `mobilenetv2_train_step.mlir` and both train forwards, while every artifact in
  this file — including `mobilenetv2in_rmsdp64`, whose accuracy the book quotes — is **BATCH BN**
  (reduce `[0,2,3]`). `scripts/regen_verified_mlir.sh`'s `check_adam_prefix` carried the split as
  its LAST `KNOWN_SPLIT` entry.
* That renderer's train step was reachable only from `mobilenetv2-verified`, whose own header
  measured its accuracy at chance (387/3925, byte identical every epoch — running-statistic
  threading lives only in `trainAdamSched`) and said not to quote it. Both are retired.

**What the convergence changed, and what it did not.** `@mobilenetv2_fwd` and `@mobilenetv2in_fwd`
now come from `mnv2FwdChainB`, the ONE traversal every train step below differentiates, so the net
that scores and the net that trains are one graph by construction. ⚠ Their BatchNorm world and
their parameter names both change; the driver binds positionally, so the rename reaches nothing.
⚠ The EVAL forwards did NOT move: `bnPerChannelEvalF` reads frozen statistics and reduces nothing,
so they are BatchNorm-world-agnostic and re-render byte-identically from the per-example chain,
which came here with them. See that section's banner for the second reason not to move them.

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

⭐ **The Proofs tier this file's train steps are tied at** is the batch-BN one:
`MobileNetV2FullB.lean` (T1 forward, T2), `MobileNetV2FullBVJP.lean` (T1's VJP),
`GradNodesB` (T3 §1 fold, un-fused) and `MobileNetV2StepTieB.lean` (T3 §1a
tie) — §4.2 of `planning/archive/proofs_tier_to_paper_nets.md`, all 2026-09-06. The per-example
fold and tie were retired (2026-09-08 and 2026-09-19), since no committed bytes exercised them.

Render is value-independent (`skel` erases values), so placeholder zeros and `ε := 0` are passed;
the emitted literals carry the real values.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

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
  -- ⭐ SYNC-BN (`replicas > 1`): each BN site's all-reduced packed `[μ ‖ σ²]`, read by its backward,
  -- its γ gradient and the handed-back running stats. `""` at one replica.
  stE : String := ""
  stD : String := ""
  stP : String := ""
deriving Inhabited

/-- Backward result: code, the dx cotangent to the previous block, and the block's parameter
    gradients in func-arg order. -/
structure MBBackB where
  code : String
  dx : String
  ps : List PGrad

-- ════════════════════════════════════════════════════════════════
-- § Block forward (batch BN), all at `N := B`
--   inverted residual: expand(1×1)→BN→relu6 → depthwise(3×3)→BN→relu6 → project(1×1)→BN
-- ════════════════════════════════════════════════════════════════

/-- **STRIDED inverted-residual forward**: expand at the input `2hh×2ww`, depthwise downsamples
    `2hh×2ww → hh×ww`, project 1×1 at `hh×ww`. NO skip. -/
def irFwdStridedB (B ic mid oc hh : Nat) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS MBFwdB := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (B*(ic*(2*hh)*(2*ww))) := fun _ => 0
  let zeb  : Vec (B*(mid*(2*hh)*(2*ww))) := fun _ => 0
  let zdb  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (.convAt bf16 (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) zrnd s!"%b{p}eW" (biasName convBias s!"%b{p}eb" mid) zke zmid)
    (.operand xName zxin))
  let (cEn, nEn, stE) ← bnFwdSite B mid (2*hh) (2*ww) sync replicas epsStr s!"%b{p}eg" s!"%b{p}ebt" s!"b{p}eg" nEc
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*(2*hh)*(2*ww))) (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (.depthwiseStridedXlaAt bf16 (c := mid) (h := hh) (w := ww) zrnd s!"%b{p}dW" (biasName convBias s!"%b{p}db" mid) zdk zmid)
    (.operand nEr zeb))
  let (cDn, nDn, stD) ← bnFwdSite B mid (hh) (ww) sync replicas epsStr s!"%b{p}dg" s!"%b{p}dbt" s!"b{p}dg" nDc
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nDn zdb))
  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (.convAt bf16 (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc)
    (.operand nDr zdb))
  let (cPn, nPn, stP) ← bnFwdSite B oc (hh) (ww) sync replicas epsStr s!"%b{p}pg" s!"%b{p}pbt" s!"b{p}pg" nPc
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc,
         stE := stE, stD := stD, stP := stP }

/-- **STRIDE-1 inverted-residual forward with the identity skip** (`ic = oc`): everything at
    `hh×ww`, block output = `addVB (project-BN out) (block input)`. The bottleneck is LINEAR — no
    relu6 after the add. -/
def irFwdSkipB (B ic mid oc hh : Nat) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS MBFwdB := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zeb  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zob  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (.convAt bf16 (ic := ic) (oc := mid) (h := hh) (w := ww) zrnd s!"%b{p}eW" (biasName convBias s!"%b{p}eb" mid) zke zmid)
    (.operand xName zxin))
  let (cEn, nEn, stE) ← bnFwdSite B mid (hh) (ww) sync replicas epsStr s!"%b{p}eg" s!"%b{p}ebt" s!"b{p}eg" nEc
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (.depthwiseAt bf16 (c := mid) (h := hh) (w := ww) zrnd s!"%b{p}dW" (biasName convBias s!"%b{p}db" mid) zdk zmid) (.operand nEr zeb))
  let (cDn, nDn, stD) ← bnFwdSite B mid (hh) (ww) sync replicas epsStr s!"%b{p}dg" s!"%b{p}dbt" s!"b{p}dg" nDc
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nDn zeb))
  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (.convAt bf16 (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc)
    (.operand nDr zeb))
  let (cPn, nPn, stP) ← bnFwdSite B oc (hh) (ww) sync replicas epsStr s!"%b{p}pg" s!"%b{p}pbt" s!"b{p}pg" nPc
  let (cA, nA) ← pretty B (.addVB (.operand nPn zob) (.operand xName zob))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn ++ cA,
         o := nA, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc,
         stE := stE, stD := stD, stP := stP }

/-- **EXPAND-NO-SKIP stride-1 forward** (b11/b17): as `irFwdSkipB` but `ic ≠ oc`, so the block
    output is the project-BN output directly. -/
def irFwdNoSkipB (B ic mid oc hh : Nat) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS MBFwdB := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zeb  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B)
    (.convAt bf16 (ic := ic) (oc := mid) (h := hh) (w := ww) zrnd s!"%b{p}eW" (biasName convBias s!"%b{p}eb" mid) zke zmid)
    (.operand xName zxin))
  let (cEn, nEn, stE) ← bnFwdSite B mid (hh) (ww) sync replicas epsStr s!"%b{p}eg" s!"%b{p}ebt" s!"b{p}eg" nEc
  let (cEr, nEr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (.depthwiseAt bf16 (c := mid) (h := hh) (w := ww) zrnd s!"%b{p}dW" (biasName convBias s!"%b{p}db" mid) zdk zmid) (.operand nEr zeb))
  let (cDn, nDn, stD) ← bnFwdSite B mid (hh) (ww) sync replicas epsStr s!"%b{p}dg" s!"%b{p}dbt" s!"b{p}dg" nDc
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu6 (n := mid*hh*ww)) (.operand nDn zeb))
  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (.convAt bf16 (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc)
    (.operand nDr zeb))
  let (cPn, nPn, stP) ← bnFwdSite B oc (hh) (ww) sync replicas epsStr s!"%b{p}pg" s!"%b{p}pbt" s!"b{p}pg" nPc
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc,
         stE := stE, stD := stD, stP := stP }

/-- **NO-EXPAND forward** (b1, the canonical `t = 1` block): depthwise(stride-1, on `ic` channels)
    → BN → relu6 → project(1×1 `ic→oc`) → BN. No expand conv, no skip. `er` is the depthwise INPUT
    (= the block input), which is what the depthwise weight gradient reads. -/
def irFwdNoExpB (B ic oc hh : Nat) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS MBFwdB := do
  let ww := hh
  let zic  : Vec ic := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zkp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel ic 3 3 := fun _ _ _ => 0
  let zib  : Vec (B*(ic*hh*ww)) := fun _ => 0
  let (cDc, nDc) ← pretty B (.batchOp (N := B)
    (.depthwiseAt bf16 (c := ic) (h := hh) (w := ww) zrnd s!"%b{p}dW" (biasName convBias s!"%b{p}db" ic) zdk zic) (.operand xName zib))
  let (cDn, nDn, stD) ← bnFwdSite B ic (hh) (ww) sync replicas epsStr s!"%b{p}dg" s!"%b{p}dbt" s!"b{p}dg" nDc
  let (cDr, nDr) ← pretty B (.batchOp (N := B) (.relu6 (n := ic*hh*ww)) (.operand nDn zib))
  let (cPc, nPc) ← pretty B (.batchOp (N := B)
    (.convAt bf16 (ic := ic) (oc := oc) (h := hh) (w := ww) zrnd s!"%b{p}pW" (biasName convBias s!"%b{p}pb" oc) zkp zoc)
    (.operand nDr zib))
  let (cPn, nPn, stP) ← bnFwdSite B oc (hh) (ww) sync replicas epsStr s!"%b{p}pg" s!"%b{p}pbt" s!"b{p}pg" nPc
  pure { code := cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := xName, en := xName, er := xName, dc := nDc, dn := nDn, dr := nDr, pc := nPc,
         stD := stD, stP := stP }

-- ════════════════════════════════════════════════════════════════
-- § Block backward + un-fused parameter gradients
--   (project → depthwise → expand; dyOut flows straight into the project-BN backward,
--    because the linear bottleneck has no relu6 after project)
-- ════════════════════════════════════════════════════════════════

/-- **STRIDED backward + 12 un-fused gradients.** The depthwise input gradient lands at `2hh×2ww`;
    no skip, so the dx handed to the previous block is the expand-conv backward directly. -/
private def irBackStridedGradB (B ic mid oc hh : Nat) (epsStr p xName : String)
    (f : MBFwdB) (dyName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS MBBackB := do
  let ww := hh
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
  let (cDpc, nDpc) ← bnBackSite B oc (hh) (ww) sync replicas epsStr s!"%b{p}pg" f.pc s!"b{p}pgdst" dyName f.stP
  let (cDdr, nDdr) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd
    s!"%b{p}pW" zkp zoc (.operand nDpc zob))
  let (cDdm, nDdm) ← pretty B (.selectMidB f.dn zdb (.operand nDdr zdb))
  let (cDdn, nDdn) ← bnBackSite B mid (hh) (ww) sync replicas epsStr s!"%b{p}dg" f.dc s!"b{p}dgdst" nDdm f.stD
  let (cDer, nDer) ← pretty B (.depthwiseStridedXlaBackBatchedAt bf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd
    s!"%b{p}dW" zdk zmid (.operand nDdn zdb))
  let (cDem, nDem) ← pretty B (.selectMidB f.en zeb (.operand nDer zeb))
  let (cDen, nDen) ← bnBackSite B mid (2*hh) (2*ww) sync replicas epsStr s!"%b{p}eg" f.ec s!"b{p}egdst" nDem f.stE
  let (cDxb, nDxb) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := ic) (oc := mid)
    (h := 2*hh) (w := 2*ww) zrnd s!"%b{p}eW" zke zmid (.operand nDen zeb))
  -- the 12 gradients, func-arg order: eW eb eg ebt | dW db dg dbt | pW pb pg pbt
  let (cEW, nEW) ← pretty B (.convWeightGradBAt bf16 (N := B) (ic := ic) (oc := mid)
    (h := 2*hh) (w := 2*ww) zrnd xName zmid zxin zke (.operand nDen zeb))
  let (cEb, nEb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := ic) (oc := mid)
        (h := 2*hh) (w := 2*ww) zke zxin zmid (.operand nDen zeb))
    else pure ("", "")
  let (cEg, nEg) ← bnGammaSite B mid (2*hh) (2*ww) sync epsStr f.ec nDem f.stE
  let (cEt, nEt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := 2*hh) (w := 2*ww)
    (.operand nDem zebp))
  let (cDW, nDW) ← pretty B (.depthwiseStridedXlaWeightGradBAt bf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd
    f.er zmid zeb zdk (.operand nDdn zdb))
  let (cDb, nDb) ← if convBias then
      pretty B (.depthwiseStridedXlaBiasGradB (N := B) (c := mid) (h := hh) (w := ww)
        zdk zeb zmid (.operand nDdn zdb))
    else pure ("", "")
  let (cDg, nDg) ← bnGammaSite B mid (hh) (ww) sync epsStr f.dc nDdm f.stD
  let (cDt, nDt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww)
    (.operand nDdm zdbp))
  let (cPW, nPW) ← pretty B (.convWeightGradBAt bf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd
    f.dr zoc zdb zkp (.operand nDpc zob))
  let (cPb, nPb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww)
        zkp zdb zoc (.operand nDpc zob))
    else pure ("", "")
  let (cPg, nPg) ← bnGammaSite B oc (hh) (ww) sync epsStr f.pc dyName f.stP
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
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS MBBackB := do
  let ww := hh
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
  let (cDpc, nDpc) ← bnBackSite B oc (hh) (ww) sync replicas epsStr s!"%b{p}pg" f.pc s!"b{p}pgdst" dyName f.stP
  let (cDdr, nDdr) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd
    s!"%b{p}pW" zkp zoc (.operand nDpc zob))
  let (cDdm, nDdm) ← pretty B (.selectMidB f.dn zeb (.operand nDdr zeb))
  let (cDdn, nDdn) ← bnBackSite B mid (hh) (ww) sync replicas epsStr s!"%b{p}dg" f.dc s!"b{p}dgdst" nDdm f.stD
  let (cDer, nDer) ← pretty B (.depthwiseBackBatchedAt bf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd
    s!"%b{p}dW" zdk zmid (.operand nDdn zeb))
  let (cDem, nDem) ← pretty B (.selectMidB f.en zeb (.operand nDer zeb))
  let (cDen, nDen) ← bnBackSite B mid (hh) (ww) sync replicas epsStr s!"%b{p}eg" f.ec s!"b{p}egdst" nDem f.stE
  let (cDxb, nDxb) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww) zrnd
    s!"%b{p}eW" zke zmid (.operand nDen zeb))
  -- skip fan-in: (body dx) + dyOut, at the block-input shape (ic = oc for a skip block)
  let (cDx, nDx) ← if skip then
      pretty B (.addVB (.operand nDxb zxin) (.operand dyName zxin))
    else pure ("", nDxb)
  let (cEW, nEW) ← pretty B (.convWeightGradBAt bf16 (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww) zrnd
    xName zmid zxin zke (.operand nDen zeb))
  let (cEb, nEb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww)
        zke zxin zmid (.operand nDen zeb))
    else pure ("", "")
  let (cEg, nEg) ← bnGammaSite B mid (hh) (ww) sync epsStr f.ec nDem f.stE
  let (cEt, nEt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww)
    (.operand nDem zebp))
  let (cDW, nDW) ← pretty B (.depthwiseWeightGradBAt bf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd
    f.er zmid zeb zdk (.operand nDdn zeb))
  let (cDb, nDb) ← if convBias then
      pretty B (.depthwiseBiasGradB (N := B) (c := mid) (h := hh) (w := ww)
        zdk zeb zmid (.operand nDdn zeb))
    else pure ("", "")
  let (cDg, nDg) ← bnGammaSite B mid (hh) (ww) sync epsStr f.dc nDdm f.stD
  let (cDt, nDt) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww)
    (.operand nDdm zebp))
  let (cPW, nPW) ← pretty B (.convWeightGradBAt bf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd
    f.dr zoc zeb zkp (.operand nDpc zob))
  let (cPb, nPb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww)
        zkp zeb zoc (.operand nDpc zob))
    else pure ("", "")
  let (cPg, nPg) ← bnGammaSite B oc (hh) (ww) sync epsStr f.pc dyName f.stP
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
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS MBBackB := do
  let ww := hh
  let zic  : Vec ic := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zkp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel ic 3 3 := fun _ _ _ => 0
  let zib  : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zibp : Vec (B*(ic*(hh*ww))) := fun _ => 0
  let zob  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zobp : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let (cDpc, nDpc) ← bnBackSite B oc (hh) (ww) sync replicas epsStr s!"%b{p}pg" f.pc s!"b{p}pgdst" dyName f.stP
  let (cDdr, nDdr) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) zrnd
    s!"%b{p}pW" zkp zoc (.operand nDpc zob))
  let (cDdm, nDdm) ← pretty B (.selectMidB f.dn zib (.operand nDdr zib))
  let (cDdn, nDdn) ← bnBackSite B ic (hh) (ww) sync replicas epsStr s!"%b{p}dg" f.dc s!"b{p}dgdst" nDdm f.stD
  let (cDxb, nDxb) ← pretty B (.depthwiseBackBatchedAt bf16 (N := B) (c := ic) (h := hh) (w := ww) zrnd
    s!"%b{p}dW" zdk zic (.operand nDdn zib))
  let (cDW, nDW) ← pretty B (.depthwiseWeightGradBAt bf16 (N := B) (c := ic) (h := hh) (w := ww) zrnd
    xName zic zib zdk (.operand nDdn zib))
  let (cDb, nDb) ← if convBias then
      pretty B (.depthwiseBiasGradB (N := B) (c := ic) (h := hh) (w := ww)
        zdk zib zic (.operand nDdn zib))
    else pure ("", "")
  let (cDg, nDg) ← bnGammaSite B ic (hh) (ww) sync epsStr f.dc nDdm f.stD
  let (cDt, nDt) ← pretty B (.bnBetaGradB (N := B) (oc := ic) (h := hh) (w := ww)
    (.operand nDdm zibp))
  let (cPW, nPW) ← pretty B (.convWeightGradBAt bf16 (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) zrnd
    f.dr zoc zib zkp (.operand nDpc zob))
  let (cPb, nPb) ← if convBias then
      pretty B (.convBiasGradB (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww)
        zkp zib zoc (.operand nDpc zob))
    else pure ("", "")
  let (cPg, nPg) ← bnGammaSite B oc (hh) (ww) sync epsStr f.pc dyName f.stP
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
    `paperSig` uses, at the names the committed AdamW artifact presents. -/
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
    (bf16 : Bool := false)
    -- ▶ `wx` (no decay on the 1-D params), `do` (classifier dropout) and the label-smoothing
    -- mass, TRAILING and defaulted for `bf16`'s reason: every committed spelling is untouched.
    -- α is spelled `ls<100α>` only when it is not the default 0.1, so `ls0` is α = 0.
    (wx : Bool := false) (cd : Bool := false) (alpha : Float := 0.1)
    -- ▶ BatchNorm ε, as `bnEpsMarker` spells it: empty at the committed 1e-5.
    (epsMarker : String := "") : String :=
  (match opt with
   | .adamw   => if replicas ≤ 1 then "adam" else "adamdp"
   | .rmsprop => if replicas ≤ 1 then "rms"  else "rmsdp") ++
  (if B == 32 then "" else toString B) ++
  (if wx then "wx" else "") ++
  (if cd then "do" else "") ++
  (if alpha == 0.1 then "" else s!"ls{(alpha * 100.0).round.toUInt64}") ++
  epsMarker ++
  (if bf16 then "bf16" else "")

-- ════════════════════════════════════════════════════════════════
-- § The forward traversal — ONE chain, consumed by `@mobilenetv2_fwd` and every train step
-- ════════════════════════════════════════════════════════════════

/-- The forward record `mnv2FwdChainB` hands to its two consumers: the emitted code, the stem's
    three activations, the seventeen block records, the head's three, and the GAP/logits names. -/
structure MNV2FwdRecB where
  code : String
  stc : String            -- stem conv out (the stem BN's input)
  stn : String            -- stem BN out (the stem relu6's pre-activation)
  str : String            -- stem relu6 out (block 1's input)
  hc  : String            -- head conv out (the head BN's input)
  hn  : String            -- head BN out (the head relu6's pre-activation)
  hr  : String            -- head relu6 out (the GAP's input)
  gap : String            -- GAP out (= dense input)
  log : String            -- logits
  b   : Array MBFwdB      -- the 17 inverted-residual blocks, in forward order
  sst : String := ""      -- stem BN's packed global statistics (sync-BN only)
  hst : String := ""      -- head BN's packed global statistics (sync-BN only)
  cin : String := ""      -- the dense's input (= gap, or the dropout output when cd is on)
deriving Inhabited

/-- The stem's saved SSA names: conv, BN, BN stats (`""` at one replica), relu6 output. -/
structure MNV2StemFwdB where
  code : String
  c : String
  n : String
  st : String
  o : String

/-- Stem forward: 3×3/s2 XLA-`SAME` conv (3→32, 224→112) → batch BN → relu6 (no max-pool). -/
def mnv2StemFwdB (B : Nat) (epsStr : String) (convBias : Bool) (bf16 : Bool := false)
    (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS MNV2StemFwdB := do
  let zx    : Vec (B*(3*224*224)) := fun _ => 0
  let zSk   : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
  let z32   : Vec 32 := fun _ => 0
  let z112  : Vec (B*(32*112*112)) := fun _ => 0
  let (cStc, nStc) ← pretty B (.batchOp (N := B)
    (.convStridedXlaAt bf16 (ic := 3) (oc := 32) (h := 112) (w := 112) zrnd "%sW" (biasName convBias "%sb" 32) zSk z32)
    (.operand "%x" zx))
  let (cStn, nStn, sst) ← bnFwdSite B 32 (112) (112) sync replicas epsStr "%sg" "%sbt" "sg" nStc
  let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu6 (n := 32*112*112)) (.operand nStn z112))
  pure { code := cStc ++ cStn ++ cStr, c := nStc, n := nStn, st := sst, o := nStr }

/-- The head's saved SSA names: conv, BN, BN stats, relu6, GAP, logits. -/
structure MNV2HeadFwdB where
  code : String
  hc : String
  hn : String
  hst : String
  hr : String
  gap : String
  log : String
  cin : String := ""      -- the dense's input: `gap`, or `gap` under the dropout mask `%do`

/-- Head forward: 1×1 conv (320→1280) → batch BN → relu6 → GAP(7×7) → [dropout] → dense, on
    block 17's output `xName`. -/
def mnv2HeadFwdB (B nClasses : Nat) (epsStr xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) (cd : Bool := false) :
    StateM Proofs.StableHLO.EmitS MNV2HeadFwdB := do
  let z7     : Vec (B*(320*7*7)) := fun _ => 0
  let zHk    : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
  let z1280  : Vec 1280 := fun _ => 0
  let zH7    : Vec (B*(1280*7*7)) := fun _ => 0
  let z1280b : Vec (B*1280) := fun _ => 0
  let zWd    : Mat 1280 nClasses := fun _ _ => 0
  let zNC    : Vec nClasses := fun _ => 0
  let (cHc, nHc) ← pretty B (.batchOp (N := B)
    (.convAt bf16 (ic := 320) (oc := 1280) (h := 7) (w := 7) zrnd "%hW" (biasName convBias "%hb" 1280) zHk z1280) (.operand xName z7))
  let (cHn, nHn, hst) ← bnFwdSite B 1280 (7) (7) sync replicas epsStr "%hg" "%hbt" "hg" nHc
  let (cHr, nHr) ← pretty B (.batchOp (N := B) (.relu6 (n := 1280*7*7)) (.operand nHn zH7))
  let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 1280) (h := 7) (w := 7))
    (.operand nHr zH7))
  -- ▶ CLASSIFIER DROPOUT between GAP and the dense, where the reference puts it (`emitForward`'s
  -- `.dense` case) and where EfficientNet's render does. At `cd = false` no `pretty` call happens,
  -- so the fresh-name counter does not move and every committed artifact re-renders byte-identical.
  let (cDo, nCin) ← if cd then
      pretty B (.dropoutB (N := B) (n := 1280) doName z1280b (.operand nGap z1280b))
    else pure ("", nGap)
  let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC)
    (.operand nCin z1280b))
  pure { code := cHc ++ cHn ++ cHr ++ cGap ++ cDo ++ cLog, hc := nHc, hn := nHn, hst := hst,
         hr := nHr, gap := nGap, log := nLog, cin := nCin }

/-- **The MobileNetV2 forward chain at the BATCHED index** — one traversal, consumed by both
    `@mobilenetv2_fwd` and every train step that differentiates it.

    ⭐⭐ **This exists so `@mobilenetv2_fwd` and the batch-BN train steps cannot be different nets.**
    They were: the retired `MobileNetV2Render.lean` built its forward from the PER-EXAMPLE chain —
    `bnPerChannelF`, reduce `[2,3]`, divisor `H·W` — while every train step in this file is batch
    BN, reduce `[0,2,3]`, divisor `B·H·W`. `scripts/regen_verified_mlir.sh`'s `check_adam_prefix`
    carried the divergence as the LAST `KNOWN_SPLIT` entry for as long as both existed. This is
    `ResNet34RenderB.r34FwdChainB`'s shape, for `ResNet50RenderB.r50FwdChainB`'s reason
    (`planning/archive/renderer_convergence.md`, leg 2).

    ⚠ The EVAL forward is deliberately NOT moved onto this chain, exactly as ResNet-34's and
    ResNet-50's are not: `bnPerChannelEvalF` reads frozen per-channel statistics and reduces
    nothing, so `mobilenetv2_fwd_eval.mlir` is BatchNorm-world-agnostic and correct against both
    chains. (Until 2026-09-08 there was a second reason: a whole-net float budget's provenance
    claim named that artifact's SSA names line for line; the budget is deleted, and the eval
    forward stays where it is only because moving it buys nothing.)

    ⭐ Extracting the traversal is byte-neutral for the train step: `pretty`'s SSA counter follows
    the call SEQUENCE, and the sequence is unchanged. -/
def mnv2FwdChainB (B nClasses : Nat) (epsStr : String) (convBias : Bool := false)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) (cd : Bool := false) :
    StateM Proofs.StableHLO.EmitS MNV2FwdRecB := do
  -- ═══ stem: 3×3/s2 conv (3→32, 224→112) → batch BN → relu6 (NO maxpool) ═══
  let st ← mnv2StemFwdB B epsStr convBias bf16 replicas sync
  let (nStc, nStn, sst, nStr) := (st.c, st.n, st.st, st.o)
  -- ═══ forward: the 17 inverted-residual blocks ═══
  let f1  ← irFwdNoExpB   B 32      16 112 epsStr "1"  nStr convBias bf16 replicas sync
  let f2  ← irFwdStridedB B 16  96  24  56 epsStr "2"  f1.o convBias bf16 replicas sync
  let f3  ← irFwdSkipB    B 24 144  24  56 epsStr "3"  f2.o convBias bf16 replicas sync
  let f4  ← irFwdStridedB B 24 144  32  28 epsStr "4"  f3.o convBias bf16 replicas sync
  let f5  ← irFwdSkipB    B 32 192  32  28 epsStr "5"  f4.o convBias bf16 replicas sync
  let f6  ← irFwdSkipB    B 32 192  32  28 epsStr "6"  f5.o convBias bf16 replicas sync
  let f7  ← irFwdStridedB B 32 192  64  14 epsStr "7"  f6.o convBias bf16 replicas sync
  let f8  ← irFwdSkipB    B 64 384  64  14 epsStr "8"  f7.o convBias bf16 replicas sync
  let f9  ← irFwdSkipB    B 64 384  64  14 epsStr "9"  f8.o convBias bf16 replicas sync
  let f10 ← irFwdSkipB    B 64 384  64  14 epsStr "10" f9.o convBias bf16 replicas sync
  let f11 ← irFwdNoSkipB  B 64 384  96  14 epsStr "11" f10.o convBias bf16 replicas sync
  let f12 ← irFwdSkipB    B 96 576  96  14 epsStr "12" f11.o convBias bf16 replicas sync
  let f13 ← irFwdSkipB    B 96 576  96  14 epsStr "13" f12.o convBias bf16 replicas sync
  let f14 ← irFwdStridedB B 96 576 160   7 epsStr "14" f13.o convBias bf16 replicas sync
  let f15 ← irFwdSkipB    B 160 960 160  7 epsStr "15" f14.o convBias bf16 replicas sync
  let f16 ← irFwdSkipB    B 160 960 160  7 epsStr "16" f15.o convBias bf16 replicas sync
  let f17 ← irFwdNoSkipB  B 160 960 320  7 epsStr "17" f16.o convBias bf16 replicas sync
  -- ═══ head: 1×1 conv (320→1280) → batch BN → relu6 → GAP(7×7) → dense ═══
  let hd ← mnv2HeadFwdB B nClasses epsStr f17.o convBias bf16 replicas sync cd
  let (nHc, nHn, hst, nHr, nGap, nLog) := (hd.hc, hd.hn, hd.hst, hd.hr, hd.gap, hd.log)
  pure { code := st.code ++
           f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++
           f8.code ++ f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++
           f15.code ++ f16.code ++ f17.code ++ hd.code,
         stc := nStc, stn := nStn, str := nStr,
         hc := nHc, hn := nHn, hr := nHr, gap := nGap, log := nLog,
         b := #[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17],
         sst := sst, hst := hst, cin := hd.cin }

/-- **`@mobilenetv2_fwd` rendered from the BATCHED chain** — the same traversal every batch-BN
    train step in this file differentiates, so the net that scores and the net that trains are one
    graph by construction. Replaces the retired `MobileNetV2Render.lean` as the writer of
    `verified_mlir/mobilenetv2_fwd.mlir` (2026-09-06, `planning/archive/renderer_convergence.md` leg 2).
    Takes `%x` plus the parameters in `mnv2SigList` order — 159 inputs at the shipped
    `convBias := false` — and returns logits `[B, nClasses]`.

    ⚠ **This CHANGES what `@mobilenetv2_fwd` computes, and that is the point.** The retired render
    normalised PER EXAMPLE while the AdamW and RMSProp steps whose accuracies the book quotes
    normalise over the BATCH. ⚠ It also renames every parameter — `%sW`/`%b2eW`/`%Wd` where the
    retired one said `%Ws`/`%We2`/`%Wfc` — because the names now come from `mnv2SigList`, this
    file's single source. The driver binds positionally, so nothing downstream sees the rename. -/
def mobilenetv2FwdFaithfulB (B nClasses : Nat) (epsStr : String)
    (slug : String := "mobilenetv2") (convBias : Bool := false) (bf16 : Bool := false) : String :=
  let sigList := mnv2SigList nClasses convBias
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let F : MNV2FwdRecB := (mnv2FwdChainB B nClasses epsStr convBias bf16).run' (0, [])
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // -- MobileNetV2 (17-block paper) batch-BN forward: every line is pretty(verified AST node) --\n" ++
  zeroBiasPrelude convBias [16, 24, 32, 64, 96, 128, 144, 160, 192, 256, 320, 384, 576, 960, 1280] ++ F.code ++
  s!"    return {F.log} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § The whole-net batched AdamW train step
-- ════════════════════════════════════════════════════════════════

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
    (bf16 : Bool := false)
    -- ▶ `forceSync`: the sync-BN graph at ONE replica (every collective empty), for the numeric
    -- gate `mobilenetv2-syncbn-check`. Never a committed artifact.
    (forceSync : Bool := false)
    -- ▶ The recipe knobs, TRAILING and defaulted so every committed render is byte-identical:
    --   `wdExclude` — `wx`, no decay on the 1-D parameters (BN γ/β, biases): `r34WdName`'s rule;
    --   `cd` — classifier dropout at the driver's `%do` mask (EfficientNet's and MNv4's slot);
    --   `alpha` — the label-smoothing mass. At α = 0 the smoothing ops are not emitted at all.
    (wdExclude : Bool := false) (cd : Bool := false) (alpha : Float := 0.1) : String :=
  let sync : Bool := replicas > 1 || forceSync
  -- ⚠ α and K are spelled ONCE here. Until 2026-08-02 this render carried `0.100000` and
  -- `-0.010000` as inline literals in the cotangent AND a third copy, `0.010000`, in the
  -- report-only loss — the K=10 values. mnv2 is the WORST of the four ImageNet ports on this axis
  -- because one of them is on the GRADIENT path, which is §2k's original bug rather than the
  -- report-only variant found in ConvNeXt and EfficientNet. At K=10 all three render byte-identical
  -- to the literals they replace, so the fix is inert on every committed artifact.
  let alphaStr    := fmt6 alpha               -- α itself ("0.100000" at the default)
  let negAlphaKStr := "-" ++ alphaOverK nClasses alpha
  let go : StateM Proofs.StableHLO.EmitS String := do
    -- ═══ forward — the SAME traversal `@mobilenetv2_fwd` renders, so the forward this
    --     differentiates and the forward the driver scores with are one graph by construction
    --     (leg 2 of `planning/archive/renderer_convergence.md`) ═══
    let F : MNV2FwdRecB ← mnv2FwdChainB B nClasses epsStr convBias bf16 replicas sync cd
    let zx    : Vec (B*(3*224*224)) := fun _ => 0
    let zSk   : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let z32   : Vec 32 := fun _ => 0
    let z112  : Vec (B*(32*112*112)) := fun _ => 0
    let z112p : Vec (B*(32*(112*112))) := fun _ => 0
    let nStc := F.stc; let nStn := F.stn; let nStr := F.str
    let f1  := F.b[0]!;  let f2  := F.b[1]!;  let f3  := F.b[2]!;  let f4  := F.b[3]!
    let f5  := F.b[4]!;  let f6  := F.b[5]!;  let f7  := F.b[6]!;  let f8  := F.b[7]!
    let f9  := F.b[8]!;  let f10 := F.b[9]!;  let f11 := F.b[10]!; let f12 := F.b[11]!
    let f13 := F.b[12]!; let f14 := F.b[13]!; let f15 := F.b[14]!; let f16 := F.b[15]!
    let f17 := F.b[16]!
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
    let nHc := F.hc; let nHn := F.hn; let nHr := F.hr
    let nGap := F.gap; let nLog := F.log
    let _ := (zx, zSk, z32, z112, z112p, z7, zHk, z1280, zH7p, z1280b, zWd, zNC, zNCb,
              zNCp, nStr, nHr, nGap)
    -- ═══ label-smoothed softmax-CE cotangent, COMPOSED from kit ops (α = 0.1, K = nClasses):
    --     dy = (softmax(logits) − onehot + α·onehot − α/K) / B. Every line is a verified node;
    --     the hand-written render fuses this into one [B,K] block, so the two graphs differ. ═══
    let (cSm,  nSm)  ← pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses))
      (.operand nLog zNCb))
    let (cD0,  nD0)  ← pretty B (.subB (.operand nSm zNCb) (.operand "%onehot" zNCb))
    -- At α = 0 the smoothing chain is not emitted: `dy = (softmax − onehot)/B`.
    let (cLsa, cD1, cD2, nD2) ← if alpha == 0.0 then pure ("", "", "", nD0) else do
      let (cLsa, nLsa) ← pretty B (.scaleB alphaStr 0 (.operand "%onehot" zNCb))
      let (cD1,  nD1)  ← pretty B (.addVB (.operand nD0 zNCb) (.operand nLsa zNCb))
      let (cD2,  nD2)  ← pretty B (.shiftB negAlphaKStr 0 (.operand nD1 zNCb))
      pure (cLsa, cD1, cD2, nD2)
    let (cDy,  nDy)  ← pretty B (.divConstB s!"{B}.0" 0 (.operand nD2 zNCb))
    -- ═══ head backward + the 6 head/dense gradients ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (N := B)
      (.denseRowBack (rows := 1) (a := 1280) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    -- ⚠⚠ `F.cin`, NOT `nGap`: the classifier weight gradient reads the DENSE'S INPUT, which with
    -- classifier dropout on is the dropped activation (EfficientNet's `ENetFwd.cin` note).
    let (cWdg, nWdg) ← pretty B (.denseWeightGradB (c := nClasses) F.cin z1280b (.operand nDy zNCp))
    let (cbdg, nbdg) ← pretty B (.denseBiasGradB (N := B) (.operand nDy zNCp))
    -- ▶ Dropout's backward is the same op at the same mask (`Proofs.dropout_vjp_is_self`), between
    -- the dense's input-VJP and the GAP backward. At `cd = false` nothing is emitted.
    let (cDdo, nDdo) ← if cd then
        pretty B (.dropoutB (N := B) (n := 1280) doName z1280b (.operand nDgi z1280b))
      else pure ("", nDgi)
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 1280) (h := 7) (w := 7)
      (.operand nDdo z1280b))
    let (cDhm, nDhm) ← pretty B (.selectMidB nHn zH7 (.operand nDgp zH7))
    let (cDhn, nDhn) ← bnBackSite B 1280 (7) (7) sync replicas epsStr "%hg" nHc "hgdst" nDhm F.hst
    let (cDhx, nDhx) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7) zrnd
      "%hW" zHk z1280 (.operand nDhn zH7))
    let (cHW, nHW) ← pretty B (.convWeightGradBAt bf16 (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7) zrnd
      f17.o z1280 z7 zHk (.operand nDhn zH7))
    let (cHb, nHb) ← if convBias then
        pretty B (.convBiasGradB (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7)
            zHk z7 z1280 (.operand nDhn zH7))
      else pure ("", "")
    let (cHg, nHg) ← bnGammaSite B 1280 (7) (7) sync epsStr nHc nDhm F.hst
    let (cHt, nHt) ← pretty B (.bnBetaGradB (N := B) (oc := 1280) (h := 7) (w := 7)
      (.operand nDhm zH7p))
    -- ═══ backward: the 17 blocks reversed (the cotangent threads from nDhx) ═══
    let b17 ← irBackStride1GradB B 160 960 320  7 false epsStr "17" f16.o f17 nDhx convBias bf16 replicas sync
    let b16 ← irBackStride1GradB B 160 960 160  7 true  epsStr "16" f15.o f16 b17.dx convBias bf16 replicas sync
    let b15 ← irBackStride1GradB B 160 960 160  7 true  epsStr "15" f14.o f15 b16.dx convBias bf16 replicas sync
    let b14 ← irBackStridedGradB B 96 576 160   7       epsStr "14" f13.o f14 b15.dx convBias bf16 replicas sync
    let b13 ← irBackStride1GradB B 96 576  96  14 true  epsStr "13" f12.o f13 b14.dx convBias bf16 replicas sync
    let b12 ← irBackStride1GradB B 96 576  96  14 true  epsStr "12" f11.o f12 b13.dx convBias bf16 replicas sync
    let b11 ← irBackStride1GradB B 64 384  96  14 false epsStr "11" f10.o f11 b12.dx convBias bf16 replicas sync
    let b10 ← irBackStride1GradB B 64 384  64  14 true  epsStr "10" f9.o  f10 b11.dx convBias bf16 replicas sync
    let b9  ← irBackStride1GradB B 64 384  64  14 true  epsStr "9"  f8.o  f9  b10.dx convBias bf16 replicas sync
    let b8  ← irBackStride1GradB B 64 384  64  14 true  epsStr "8"  f7.o  f8  b9.dx convBias bf16 replicas sync
    let b7  ← irBackStridedGradB B 32 192  64  14       epsStr "7"  f6.o  f7  b8.dx convBias bf16 replicas sync
    let b6  ← irBackStride1GradB B 32 192  32  28 true  epsStr "6"  f5.o  f6  b7.dx convBias bf16 replicas sync
    let b5  ← irBackStride1GradB B 32 192  32  28 true  epsStr "5"  f4.o  f5  b6.dx convBias bf16 replicas sync
    let b4  ← irBackStridedGradB B 24 144  32  28       epsStr "4"  f3.o  f4  b5.dx convBias bf16 replicas sync
    let b3  ← irBackStride1GradB B 24 144  24  56 true  epsStr "3"  f2.o  f3  b4.dx convBias bf16 replicas sync
    let b2  ← irBackStridedGradB B 16  96  24  56       epsStr "2"  f1.o  f2  b3.dx convBias bf16 replicas sync
    let b1  ← irBackNoExpGradB   B 32      16 112       epsStr "1"  nStr  f1  b2.dx convBias bf16 replicas sync
    -- ═══ stem backward: relu6 mask → BN back, then the 4 stem gradients (NO conv-back past %x) ═══
    let (cDsm, nDsm) ← pretty B (.selectMidB nStn z112 (.operand b1.dx z112))
    let (cDsn, nDsn) ← bnBackSite B 32 (112) (112) sync replicas epsStr "%sg" nStc "sgdst" nDsm F.sst
    let (csW, nsW) ← pretty B (.convStridedXlaWeightGradBAt bf16 zrnd "%x" z32 zx zSk (.operand nDsn z112))
    let (csb, nsb) ← if convBias then
        pretty B (.convStridedXlaBiasGradB (h := 112) (w := 112) zSk zx z32
            (.operand nDsn z112))
      else pure ("", "")
    let (csg, nsg) ← bnGammaSite B 32 (112) (112) sync epsStr nStc nDsm F.sst
    let (cst, nst) ← pretty B (.bnBetaGradB (N := B) (oc := 32) (h := 112) (w := 112)
      (.operand nDsm z112p))
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT.
    --     Derived from the SAME forward record that computes them (`f.ec`/`f.dc`/`f.pc`) rather
    --     than from an independent 52-entry table — a misaligned stat slot is SILENT, since the
    --     arities still match and the wrong layer's statistics simply flow into the wrong
    --     `@mobilenetv2_fwd_eval` slot (§2e). ═══
    -- ⭐ At `replicas > 1` they are read off the all-reduced packed vector (`bnStatsMeanB` /
    -- `bnStatsVarB`), so the host EMAs the GLOBAL batch statistics rather than replica 0's shard's.
    let bnStat (oc hh : Nat) (xn st : String) : StateM Proofs.StableHLO.EmitS (String × String × String) := do
      let zb : Vec (B*(oc*(hh*hh))) := fun _ => 0
      let zst : Vec (oc+oc) := fun _ => 0
      if !sync then
        let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh)
          (.operand xn zb))
        let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh)
          (.operand xn zb))
        pure (cM ++ cV, nM, nV)
      else
        let (cM, nM) ← pretty B (.bnStatsMeanB (oc := oc) (.operand st zst))
        let (cV, nV) ← pretty B (.bnStatsVarB (oc := oc) (.operand st zst))
        pure (cM ++ cV, nM, nV)
    -- a STRIDED block's expand BN sits at the INPUT resolution 2hh; everything else at hh
    let blkStatsS (mid oc hh : Nat) (f : MBFwdB) : StateM Proofs.StableHLO.EmitS (String × List String) := do
      let (ce, me, ve) ← bnStat mid (2*hh) f.ec f.stE
      let (cd, md, vd) ← bnStat mid hh f.dc f.stD
      let (cp, mp, vp) ← bnStat oc hh f.pc f.stP
      pure (ce ++ cd ++ cp, [me, ve, md, vd, mp, vp])
    let blkStats1 (mid oc hh : Nat) (f : MBFwdB) : StateM Proofs.StableHLO.EmitS (String × List String) := do
      let (ce, me, ve) ← bnStat mid hh f.ec f.stE
      let (cd, md, vd) ← bnStat mid hh f.dc f.stD
      let (cp, mp, vp) ← bnStat oc hh f.pc f.stP
      pure (ce ++ cd ++ cp, [me, ve, md, vd, mp, vp])
    let (cQ0, m0, v0) ← bnStat 32 112 nStc F.sst
    -- b1 is the no-expand block: TWO BN layers, not three
    let (cQ1d, m1d, v1d) ← bnStat 32 112 f1.dc f1.stD
    let (cQ1p, m1p, v1p) ← bnStat 16 112 f1.pc f1.stP
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
    let (cQh, mh, vh) ← bnStat 1280 7 nHc F.hst
    -- ═══ the 210 parameter gradients in func-arg order ═══
    let stemPs : List PGrad :=
      [⟨"sW", nsW, [32,3,3,3]⟩] ++ (if convBias then [⟨"sb", nsb, [32]⟩] else []) ++
      [⟨"sg", nsg, [32]⟩, ⟨"sbt", nst, [32]⟩]
    let headPs : List PGrad :=
      [⟨"hW", nHW, [1280,320,1,1]⟩] ++ (if convBias then [⟨"hb", nHb, [1280]⟩] else []) ++
      [⟨"hg", nHg, [1280]⟩, ⟨"hbt", nHt, [1280]⟩,
       ⟨"Wd", nWdg, [1280, nClasses]⟩, ⟨"bd", nbdg, [nClasses]⟩]
    let allPs : List PGrad := stemPs ++
      b1.ps ++ b2.ps ++ b3.ps ++ b4.ps ++ b5.ps ++ b6.ps ++ b7.ps ++ b8.ps ++ b9.ps ++
      b10.ps ++ b11.ps ++ b12.ps ++ b13.ps ++ b14.ps ++ b15.ps ++ b16.ps ++ b17.ps ++ headPs
    -- ═══ AdamW: one proven triple per parameter ═══
    let mut adamCode := ""
    let mut thetaN : List String := []
    let mut mNames : List String := []
    let mut vNames : List String := []
    for g in allPs do
      let wdN := r34WdName wdExclude g.nm g.ds
      let (c, nT, nM, nV) ← match opt with
        | .adamw   => adamOne B replicas g wdN
        | .rmsprop => rmsOne  B replicas g wdN
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
      s!"    %lomac = stablehlo.constant dense<{oneMinusAlpha alpha}> : {ty [B]}\n" ++
      s!"    %laKc = stablehlo.constant dense<{alphaOverK nClasses alpha}> : {ty [B]}\n" ++
      s!"    %llt1 = stablehlo.multiply %lomac, %lt1s : {ty [B]}\n" ++
      s!"    %llt2 = stablehlo.multiply %laKc, %llsr : {ty [B]}\n" ++
      s!"    %llpe = stablehlo.add %llt1, %llt2 : {ty [B]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [B]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
      s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
      s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
    let body := F.code ++ cSm ++ cD0 ++ cLsa ++ cD1 ++ cD2 ++ cDy ++
      cDgi ++ cWdg ++ cbdg ++ cDdo ++ cDgp ++ cDhm ++ cDhn ++ cDhx ++ cHW ++ cHb ++ cHg ++ cHt ++
      b17.code ++ b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++
      b10.code ++ b9.code ++ b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++
      b2.code ++ b1.code ++
      cDsm ++ cDsn ++ csW ++ csb ++ csg ++ cst ++ statCode
    let pTypes : List String := allPs.map (fun g => ty g.ds)
    let statTypes : List String := mnv2StatSigList.map (·.2)
    -- The dropout mask goes LAST, handed back as a passthrough (EfficientNet's and MNv4's slot).
    let retVals := thetaN ++ mNames ++ vNames ++ ["%loss", "%bc1", "%bc2"] ++ statNames ++
      (if cd then [doName] else [])
    let retTys  := pTypes ++ pTypes ++ pTypes ++
      ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statTypes ++
      (if cd then [ty [B, 1280]] else [])
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
        "    // Every line is pretty(verified AST node), the per-parameter `%arsum*` all_reduce /\n" ++
        "    // `%armean*` blocks included: pretty(allReduceMeanF), whose den is the replica MEAN of\n" ++
        "    // the per-replica gradient nodes (4d piece 2). BatchNorm is SYNCHRONISED: every BN\n" ++
        "    // layer all-reduces its mu, then var_r + (mu_r - mu)^2 (bnBatchVarAtB, Chan's parallel\n" ++
        "    // variance), before normalising with the global [mu | var] (bnSyncF); its\n" ++
        "    // backward all-reduces the two dy-reductions (bnSyncDyStatsB -> bnSyncBack), and the gamma\n" ++
        "    // gradient reads the same global x-hat (bnSyncGammaGradB). Each replica therefore computes\n" ++
        "    // its shard of the GLOBAL-batch function, and this step IS the single-device step at the\n" ++
        "    // global batch N x b: proved as MobileNetV2SyncTieB.mnv2_net_syncTiedB (every all-reduced\n" ++
        "    // gradient) and StableHLO.mobilenetv2FwdGraphSyncFull_shard (the forward), both in\n" ++
        "    // LeanMlir/Proofs/Nets/MobileNet/ (planning/global_bn_verified.md).\n" ++
        (if bf16 then
          "    // (Both are stated at the f32 nodes; this artifact's bf16 conv twins, which round\n" ++
          "    // their operands per element, are not in that statement.)\n"
         else "")) ++
      zeroBiasPrelude convBias [16, 24, 32, 64, 96, 128, 144, 160, 192, 256, 320, 384, 576, 960, 1280] ++ body ++
      (match opt with | .adamw => adamWConsts | .rmsprop => rmsConstsBlock mnv2RmsHyper) ++
      wdzConst wdExclude ++ adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let sigList : List (String × String) := mnv2SigList nClasses convBias
  let statSig := String.intercalate ", " (mnv2StatSigList.map (fun (n, t) => s!"{n}i: {t}"))
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++ packedTrainSig sigList ++ ", " ++ statSig ++
    (if cd then s!", {doName}: {ty [B, 1280]}" else "") ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (·.2)
  let outSig := String.intercalate ", "
    (packedTrainRetTys pTy ++
     (mnv2StatSigList.map (·.2)) ++ (if cd then [ty [B, 1280]] else []))
  let inner : String := go.run' (0, [])
  let fname := s!"{slug}_{mnv2AdamVariant B replicas opt bf16 wdExclude cd alpha (bnEpsMarker epsStr)}_train_step"
  "module @m {\n" ++
  s!"  func.func @{fname}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § The PER-EXAMPLE forward chain, kept for the EVAL forward only
--   Migrated here 2026-09-06 when `MobileNetV2Render.lean` was retired (leg 2 of
--   `planning/archive/renderer_convergence.md`). `bnPerChannelEvalF` reads frozen per-channel statistics
--   and reduces nothing, so `@mobilenetv2_fwd_eval` is BatchNorm-world-agnostic and correct
--   against either chain — the same call ResNet-34 and ResNet-50 make.
--   It is not re-pointed at `mnv2FwdChainB` because nothing needs it to move. (Until 2026-09-08
--   a whole-net float budget's provenance claim named that artifact's 263 SSA names line for
--   line; that budget is deleted.)
--   ⚠ `paperSig` and `mnv2SigList` are the SAME 210/158 parameters in the SAME order under two
--   naming conventions — `%Ws`/`%We2`/`%Wfc` here, `%sW`/`%b2eW`/`%Wd` there. The `#guard`s at
--   the bottom of this file pin both arities, which is what keeps the two lists one contract.
-- ════════════════════════════════════════════════════════════════

structure MBFwd where
  code : String
  o  : String        -- block output (project-BN out, or the addV result for skip blocks)
  ec : String        -- expand conv output (= expand-BN input)
  en : String        -- expand BN output (= expand-relu6 pre-activation)
  er : String        -- expand relu6 output (= depthwise input)
  dc : String        -- depthwise conv output (= depthwise-BN input)
  dn : String        -- depthwise BN output (= depthwise-relu6 pre-activation)
  dr : String        -- depthwise relu6 output (= project input)
  pc : String        -- project conv output (= project-BN input)
  /-- The block's BN layers in forward order, `(stat prefix, channels, spatial side)`. The eval
      forward turns each into a `%{prefix}mu`/`%{prefix}var` input pair; `MobileNetV2RenderB`'s
      AdamW step hands the matching batch μ/var back in the SAME order (it walks the same block
      list). Order is expand-BN → depthwise-BN → project-BN, with the expand entry ABSENT for the
      no-expand block b1 — the layout `mobilenetv2Verified.bnChannels` is listed in, which is how
      the driver packs `runningBnStats`. A misaligned slot is SILENT: the arities still match and
      the wrong layer's statistics simply flow into the wrong site (§2e). -/
  bns : List (String × Nat × Nat)
  deriving Inhabited

-- ════════════════════════════════════════════════════════════════
-- § Block forward
--   inverted residual: expand(1×1)→BN→relu6 → depthwise(3×3)→BN→relu6 → project(1×1)→BN
-- ════════════════════════════════════════════════════════════════

/-- **STRIDED inverted-residual forward** (b1/b3/b5/b6): expand at the input `2hh×2ww`, depthwise
    downsamples `2hh×2ww → hh×ww`, project 1×1 at `hh×ww`. NO skip. -/
private def irFwdStrided (B ic mid oc hh : Nat) (epsStr p xName : String) (convBias : Bool) : StateM Proofs.StableHLO.EmitS MBFwd := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (ic*(2*hh)*(2*ww)) := fun _ => 0
  let zeb  : Vec (mid*(2*hh)*(2*ww)) := fun _ => 0
  let zdb  : Vec (mid*hh*ww) := fun _ => 0
  let _zob  : Vec (oc*hh*ww) := fun _ => 0
  let (cEc, nEc) ← pretty B (.flatConvF (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%We{p}" (biasName convBias s!"%be{p}" mid) zke zmid (.operand xName zxin))
  let (cEn, nEn) ← bnEvalSite B mid (2*hh) (2*ww) epsStr s!"%ge{p}" s!"%bte{p}" s!"b{p}en" nEc
  let (cEr, nEr) ← pretty B (.relu6F (.operand nEn zeb))
  -- ⚠ XLA-`SAME` (`depthwiseStridedXlaF`), the TF-origin convention. The symmetric token has the
  -- same type and output shape, so nothing structural would notice the wrong one here — only
  -- `scripts/gates/convention_audit.py` (pad profile) and `scripts/parity/mnv2_forward_tie.py` (values) can.
  let (cDc, nDc) ← pretty B (.depthwiseStridedXlaF (h := hh) (w := ww) s!"%Wd{p}" (biasName convBias s!"%bd{p}" mid) zdk zmid (.operand nEr zeb))
  let (cDn, nDn) ← bnEvalSite B mid hh ww epsStr s!"%gd{p}" s!"%btd{p}" s!"b{p}dn" nDc
  let (cDr, nDr) ← pretty B (.relu6F (.operand nDn zdb))
  let (cPc, nPc) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" (biasName convBias s!"%bp{p}" oc) zkp zoc (.operand nDr zdb))
  let (cPn, nPn) ← bnEvalSite B oc hh ww epsStr s!"%gp{p}" s!"%btp{p}" s!"b{p}pn" nPc
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc,
         bns := [(s!"b{p}en", mid, 2*hh), (s!"b{p}dn", mid, hh), (s!"b{p}pn", oc, hh)] }

/-- **STRIDE-1 inverted-residual forward** (b2/b4): everything at `hh×ww`, with an `addV` skip on the
    block input (ic = oc). -/
private def irFwd (B ic mid oc hh : Nat) (epsStr p xName : String) (convBias : Bool) : StateM Proofs.StableHLO.EmitS MBFwd := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (ic*hh*ww) := fun _ => 0
  let zeb  : Vec (mid*hh*ww) := fun _ => 0
  let zob  : Vec (oc*hh*ww) := fun _ => 0
  let (cEc, nEc) ← pretty B (.flatConvF (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%We{p}" (biasName convBias s!"%be{p}" mid) zke zmid (.operand xName zxin))
  let (cEn, nEn) ← bnEvalSite B mid hh ww epsStr s!"%ge{p}" s!"%bte{p}" s!"b{p}en" nEc
  let (cEr, nEr) ← pretty B (.relu6F (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.depthwiseF (h := hh) (w := ww) s!"%Wd{p}" (biasName convBias s!"%bd{p}" mid) zdk zmid (.operand nEr zeb))
  let (cDn, nDn) ← bnEvalSite B mid hh ww epsStr s!"%gd{p}" s!"%btd{p}" s!"b{p}dn" nDc
  let (cDr, nDr) ← pretty B (.relu6F (.operand nDn zeb))
  let (cPc, nPc) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" (biasName convBias s!"%bp{p}" oc) zkp zoc (.operand nDr zeb))
  let (cPn, nPn) ← bnEvalSite B oc hh ww epsStr s!"%gp{p}" s!"%btp{p}" s!"b{p}pn" nPc
  let (cA, nA) ← pretty B (.addV (.operand nPn zob) (.operand xName zob))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn ++ cA,
         o := nA, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc,
         bns := [(s!"b{p}en", mid, hh), (s!"b{p}dn", mid, hh), (s!"b{p}pn", oc, hh)] }


-- ════════════════════════════════════════════════════════════════
-- § NO-EXPAND block (b1): depthwise(stride-1, on `ic` ch)→BN→relu6 → project(1×1 ic→oc)→BN.
--   NO expand conv, NO skip. 8 params (Wd bd gd btd Wp bp gp btp).
-- ════════════════════════════════════════════════════════════════

/-- **NO-EXPAND inverted-residual forward** (b1): depthwise(stride-1, on `ic` channels)→BN→relu6
    → project(1×1 ic→oc)→BN. NO expand, NO skip. `f.er` = the depthwise INPUT (= block input
    `xName`), `f.dr` = the project input. (`ec`/`en` are unused for this block kind.) -/
private def irFwdNoExp (B ic oc hh : Nat) (epsStr p xName : String) (convBias : Bool) : StateM Proofs.StableHLO.EmitS MBFwd := do
  let ww := hh
  let zic  : Vec ic := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zkp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel ic 3 3 := fun _ _ _ => 0
  let zib  : Vec (ic*hh*ww) := fun _ => 0
  let _zob  : Vec (oc*hh*ww) := fun _ => 0
  let (cDc, nDc) ← pretty B (.depthwiseF (h := hh) (w := ww) s!"%Wd{p}" (biasName convBias s!"%bd{p}" ic) zdk zic (.operand xName zib))
  let (cDn, nDn) ← bnEvalSite B ic hh ww epsStr s!"%gd{p}" s!"%btd{p}" s!"b{p}dn" nDc
  let (cDr, nDr) ← pretty B (.relu6F (.operand nDn zib))
  let (cPc, nPc) ← pretty B (.flatConvF (ic := ic) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" (biasName convBias s!"%bp{p}" oc) zkp zoc (.operand nDr zib))
  let (cPn, nPn) ← bnEvalSite B oc hh ww epsStr s!"%gp{p}" s!"%btp{p}" s!"b{p}pn" nPc
  pure { code := cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := xName, en := xName, er := xName, dc := nDc, dn := nDn, dr := nDr, pc := nPc,
         -- NO expand entry: b1 has two BN layers, not three.
         bns := [(s!"b{p}dn", ic, hh), (s!"b{p}pn", oc, hh)] }

-- ════════════════════════════════════════════════════════════════
-- § EXPAND-NO-SKIP stride-1 block (b11, b17): == irFwd/irBack but NO addV skip.
--   block output = project-BN out directly; backward dx = expand-conv-back (no fan-in).
-- ════════════════════════════════════════════════════════════════

/-- **EXPAND-NO-SKIP stride-1 forward** (b11/b17): expand(1×1)→BN→relu6 → depthwise(3×3)→BN→relu6
    → project(1×1)→BN. Everything at `hh×ww`; `ic ≠ oc` so NO skip (block output = project-BN out). -/
private def irFwdNoSkip (B ic mid oc hh : Nat) (epsStr p xName : String) (convBias : Bool) : StateM Proofs.StableHLO.EmitS MBFwd := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (ic*hh*ww) := fun _ => 0
  let zeb  : Vec (mid*hh*ww) := fun _ => 0
  let _zob  : Vec (oc*hh*ww) := fun _ => 0
  let (cEc, nEc) ← pretty B (.flatConvF (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%We{p}" (biasName convBias s!"%be{p}" mid) zke zmid (.operand xName zxin))
  let (cEn, nEn) ← bnEvalSite B mid hh ww epsStr s!"%ge{p}" s!"%bte{p}" s!"b{p}en" nEc
  let (cEr, nEr) ← pretty B (.relu6F (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.depthwiseF (h := hh) (w := ww) s!"%Wd{p}" (biasName convBias s!"%bd{p}" mid) zdk zmid (.operand nEr zeb))
  let (cDn, nDn) ← bnEvalSite B mid hh ww epsStr s!"%gd{p}" s!"%btd{p}" s!"b{p}dn" nDc
  let (cDr, nDr) ← pretty B (.relu6F (.operand nDn zeb))
  let (cPc, nPc) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" (biasName convBias s!"%bp{p}" oc) zkp zoc (.operand nDr zeb))
  let (cPn, nPn) ← bnEvalSite B oc hh ww epsStr s!"%gp{p}" s!"%btp{p}" s!"b{p}pn" nPc
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc,
         bns := [(s!"b{p}en", mid, hh), (s!"b{p}dn", mid, hh), (s!"b{p}pn", oc, hh)] }

-- ════════════════════════════════════════════════════════════════
-- § Param signature lists (func-arg order — names + types, shared by sig + return types)
-- ════════════════════════════════════════════════════════════════

private def irSig (p : String) (ic mid oc : Nat) (convBias : Bool) : List (String × String) :=
  let b (nm : String) (c : Nat) : List (String × String) := if convBias then [(nm, ty [c])] else []
  [(s!"%We{p}", ty [mid,ic,1,1])] ++ b s!"%be{p}" mid ++
  [(s!"%ge{p}", ty [mid]), (s!"%bte{p}", ty [mid]),
   (s!"%Wd{p}", ty [mid,1,3,3])] ++ b s!"%bd{p}" mid ++
  [(s!"%gd{p}", ty [mid]), (s!"%btd{p}", ty [mid]),
   (s!"%Wp{p}", ty [oc,mid,1,1])] ++ b s!"%bp{p}" oc ++
  [(s!"%gp{p}", ty [oc]), (s!"%btp{p}", ty [oc])]

/-- **NO-EXPAND block sig** (b1): depthwise on `ic` channels + project `ic→oc`. 8 params. -/
private def irSigNoExp (p : String) (ic oc : Nat) (convBias : Bool) : List (String × String) :=
  let b (nm : String) (c : Nat) : List (String × String) := if convBias then [(nm, ty [c])] else []
  [(s!"%Wd{p}", ty [ic,1,3,3])] ++ b s!"%bd{p}" ic ++
  [(s!"%gd{p}", ty [ic]), (s!"%btd{p}", ty [ic]),
   (s!"%Wp{p}", ty [oc,ic,1,1])] ++ b s!"%bp{p}" oc ++
  [(s!"%gp{p}", ty [oc]), (s!"%btp{p}", ty [oc])]

/-- **Reduced 6-block param signature** (the demo net), func-arg order: stem (4) + 6×12 + head (4)
    + dense (2) = 82 tensors, or 64 at `convBias := false`. One source for the func signature, the
    return types and the demo's own arity — it was written out twice, which is the same
    two-lists-one-net shape that let the stem/head gate go missing in `paperSig`. -/

private def paperSig (nClasses : Nat) (convBias : Bool) : List (String × String) :=
  [("%Ws", ty [32,3,3,3])] ++ (if convBias then [("%bs", ty [32])] else []) ++
  [("%gs", ty [32]), ("%bts", ty [32])] ++
  irSigNoExp "1" 32 16 convBias ++
  irSig "2"  16  96  24 convBias ++ irSig "3"  24 144  24 convBias ++ irSig "4"  24 144  32 convBias ++
  irSig "5"  32 192  32 convBias ++ irSig "6"  32 192  32 convBias ++ irSig "7"  32 192  64 convBias ++
  irSig "8"  64 384  64 convBias ++ irSig "9"  64 384  64 convBias ++ irSig "10" 64 384  64 convBias ++
  irSig "11" 64 384  96 convBias ++ irSig "12" 96 576  96 convBias ++ irSig "13" 96 576  96 convBias ++
  irSig "14" 96 576 160 convBias ++ irSig "15" 160 960 160 convBias ++ irSig "16" 160 960 160 convBias ++
  irSig "17" 160 960 320 convBias ++
  [("%Wh", ty [1280,320,1,1])] ++ (if convBias then [("%bh", ty [1280])] else []) ++
  [("%gh", ty [1280]), ("%bth", ty [1280])] ++
  [("%Wfc", ty [1280, nClasses]), ("%bfc", ty [nClasses])]


-- ════════════════════════════════════════════════════════════════
-- § The FULL 17-block paper-spec renderer
-- ════════════════════════════════════════════════════════════════

/-- Every SSA name the 17-block MobileNetV2 forward produces, plus the 52-entry BN stat layout.
    `mnv2Fwd{,Eval}FaithfulV` return just `logits`; the train step additionally consumes the stem,
    head and per-block names on the way back. -/
structure MNV2Fwd where
  code   : String            -- stem -> 17 blocks -> head -> GAP -> dense, in emission order
  stc    : String            -- stem conv out (= stem BN input)
  stn    : String            -- stem BN out (= stem relu6 pre-act)
  str    : String            -- stem relu6 out (= b1 input)
  blocks : Array MBFwd       -- the 17 inverted-residual forwards, in forward order
  hc     : String            -- head 1x1 conv out (= head BN input)
  hn     : String            -- head BN out (= head relu6 pre-act)
  hr     : String            -- head relu6 out (= GAP input)
  gap    : String            -- global-average-pool out (= dense input)
  logits : String            -- dense out
  /-- The 52 BN layers as `(stat prefix, channels, spatial side)`, stem -> blocks in forward order
      -> head. Single source for the eval signature and the eval BN sites. -/
  bns    : List (String × Nat × Nat)
  deriving Inhabited

/-- **The full 17-block paper MobileNetV2 forward as `pretty` of the verified AST**, at the
    PER-EXAMPLE index. 3x3/s2 stem (3->32, 224->112) -> the `[t,c,n,s]` inverted-residual stack
    (112->56->28->14->7) -> 1x1 head (320->1280) -> GAP(7x7) -> dense(1280->`nClasses`).

    The EVAL forward: every BN site is `RenderKit.bnEvalSite` (frozen running statistics), so this
    writes `@mobilenetv2_fwd_eval` only; the training forward is the batched `mnv2FwdChainB`. -/
private def mnv2FwdChain (B nClasses : Nat) (epsStr : String) (convBias : Bool) :
    StateM Proofs.StableHLO.EmitS MNV2Fwd := do
    -- stem: 3x3/s2 conv (3->32, 224->112) -> BN -> relu6 (NO maxpool)
    let zx   : Vec (3*224*224) := fun _ => 0
    let zSk  : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let z32  : Vec 32 := fun _ => 0
    let z112 : Vec (32*112*112) := fun _ => 0
    let (cStc, nStc) ← pretty B (.flatConvStridedXlaF (ic := 3) (oc := 32) (h := 112) (w := 112) "%Ws" (biasName convBias "%bs" 32) zSk z32 (.operand "%x" zx))
    let (cStn, nStn) ← bnEvalSite B 32 112 112 epsStr "%gs" "%bts" "stn" nStc
    let (cStr, nStr) ← pretty B (.relu6F (.operand nStn z112))
    -- forward: 17 inverted-residual blocks
    let f1  ← irFwdNoExp   B 32      16 112 epsStr "1"  nStr convBias
    let f2  ← irFwdStrided B 16  96  24  56 epsStr "2"  f1.o convBias
    let f3  ← irFwd        B 24 144  24  56 epsStr "3"  f2.o convBias
    let f4  ← irFwdStrided B 24 144  32  28 epsStr "4"  f3.o convBias
    let f5  ← irFwd        B 32 192  32  28 epsStr "5"  f4.o convBias
    let f6  ← irFwd        B 32 192  32  28 epsStr "6"  f5.o convBias
    let f7  ← irFwdStrided B 32 192  64  14 epsStr "7"  f6.o convBias
    let f8  ← irFwd        B 64 384  64  14 epsStr "8"  f7.o convBias
    let f9  ← irFwd        B 64 384  64  14 epsStr "9"  f8.o convBias
    let f10 ← irFwd        B 64 384  64  14 epsStr "10" f9.o convBias
    let f11 ← irFwdNoSkip  B 64 384  96  14 epsStr "11" f10.o convBias
    let f12 ← irFwd        B 96 576  96  14 epsStr "12" f11.o convBias
    let f13 ← irFwd        B 96 576  96  14 epsStr "13" f12.o convBias
    let f14 ← irFwdStrided B 96 576 160   7 epsStr "14" f13.o convBias
    let f15 ← irFwd        B 160 960 160   7 epsStr "15" f14.o convBias
    let f16 ← irFwd        B 160 960 160   7 epsStr "16" f15.o convBias
    let f17 ← irFwdNoSkip  B 160 960 320   7 epsStr "17" f16.o convBias
    -- head: 1x1 conv (320->1280) -> BN -> relu6 -> GAP(7x7) -> dense(1280->nClasses)
    let z7    : Vec (320*7*7) := fun _ => 0
    let zHk   : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
    let z1280 : Vec 1280 := fun _ => 0
    let zH7   : Vec (1280*7*7) := fun _ => 0
    let zWd   : Mat 1280 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    let (cHc, nHc) ← pretty B (.flatConvF (ic := 320) (oc := 1280) (h := 7) (w := 7) "%Wh" (biasName convBias "%bh" 1280) zHk z1280 (.operand f17.o z7))
    let (cHn, nHn) ← bnEvalSite B 1280 7 7 epsStr "%gh" "%bth" "hn" nHc
    let (cHr, nHr) ← pretty B (.relu6F (.operand nHn zH7))
    let (cGap, nGap) ← pretty B (.gapF (c := 1280) (h := 7) (w := 7) (.operand nHr zH7))
    let (cLog, nLog) ← pretty B (denseF "%Wfc" "%bfc" zWd zNC (.operand nGap z1280))
    pure { code := cStc ++ cStn ++ cStr ++
             f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++
             f8.code ++ f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++
             f15.code ++ f16.code ++ f17.code ++ cHc ++ cHn ++ cHr ++ cGap ++ cLog,
           stc := nStc, stn := nStn, str := nStr,
           blocks := #[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17],
           hc := nHc, hn := nHn, hr := nHr, gap := nGap, logits := nLog,
           bns := ("stn", 32, 112) ::
             (f1.bns ++ f2.bns ++ f3.bns ++ f4.bns ++ f5.bns ++ f6.bns ++ f7.bns ++ f8.bns ++
              f9.bns ++ f10.bns ++ f11.bns ++ f12.bns ++ f13.bns ++ f14.bns ++ f15.bns ++
              f16.bns ++ f17.bns ++ [("hn", 1280, 7)]) }

/-- The `@mobilenetv2_fwd_eval` argument signature. The 104 stat slots come off
    the SAME `bns` the traversal built — never a parallel 52-entry table (§2e). -/
private def mnv2FwdSig (B nClasses : Nat) (epsStr : String) (convBias : Bool) : String :=
  let F : MNV2Fwd := (mnv2FwdChain B nClasses epsStr convBias).run' (0, [])
  let params := (paperSig nClasses convBias).map (fun (nm, t) => s!"{nm}: {t}")
  let stats := F.bns.flatMap (fun (sp, c, _) => [s!"%{sp}mu: {ty [c]}", s!"%{sp}var: {ty [c]}"])
  String.intercalate ", " ((s!"%x: {ty [B, 3*224*224]}") :: (params ++ stats))


/-- **`@mobilenetv2_fwd_eval` rendered ENTIRELY from the verified AST** — the inference forward,
    every BN site consuming frozen per-channel running stats (`bnPerChannelEvalF`) instead of
    reducing statistics out of its activation. Same 210 params in the same order, plus the 104 stat
    inputs (52 BN layers × μ/var, interleaved per layer in `bnChannels` order): **315 inputs**.

    Its train-step partner is the batch-BN `mobilenetv2_adam_train_step` in `MobileNetV2RenderB`,
    whose returned batch μ/var the driver EMAs into exactly these slots. Being frozen-stat affine,
    this graph is the same in either BN world — which is why it can live beside the per-example
    chain: `bnPerChannelEvalF` performs no reduction, so there is no batch to be honest about. -/
def mnv2FwdEvalFaithfulV (B nClasses : Nat) (epsStr : String) (convBias : Bool := false)
    (slug : String := "mobilenetv2") : String :=
  let entry := fwdEvalEntry slug epsStr
  -- ⭐ The eval forward must be the SAME NET as the train step that produces the running
  -- statistics it consumes, and that partner is `mobilenetv2_adam_train_step` in
  -- `MobileNetV2RenderB`, XLA-`SAME` since 2026-08-08 (`planning/archive/mnv4_verified.md` §3h). Since
  -- 2026-09-05 `mnv2FwdChain` is XLA-`SAME` unconditionally, so this and its per-example sibling
  -- `@mobilenetv2_fwd` are one net at every stride-2 site (they still differ in BN world: frozen
  -- stats here, per-example there) and `LEAN_MLIR_EVAL_BATCHSTATS=1` — which scores through
  -- `@mobilenetv2_fwd` — is back to being transductive-only rather than also cross-net.
  let F : MNV2Fwd := (mnv2FwdChain B nClasses epsStr convBias).run' (0, [])
  "module @m {\n" ++
  s!"  func.func @{entry}({mnv2FwdSig B nClasses epsStr convBias}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // -- MobileNetV2 eval forward (running-stats BN): every line is pretty(verified AST node) --\n" ++
  zeroBiasPrelude convBias [16, 24, 32, 64, 96, 128, 144, 160, 192, 256, 320, 384, 576, 960, 1280] ++ F.code ++
  s!"    return {F.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"


#guard (paperSig 10 true).length == 210
#guard (paperSig 10 false).length == 158        -- 210 − 52 conv biases (50 in blocks + stem + head)

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
-- ⭐⭐ **`@mobilenetv2_fwd`, moved onto THIS chain 2026-09-06** (leg 2 of
-- `planning/archive/renderer_convergence.md`). It came from `MobileNetV2Render.mnv2FwdFaithfulV`, the
-- PER-EXAMPLE render, while every train step below is batch BN — so the artifact the driver scores
-- with and the artifact it trains on were different functions of the same architecture. That was
-- the last `KNOWN_SPLIT` entry in `scripts/regen_verified_mlir.sh`. It is now a byte-identical
-- PREFIX of `mobilenetv2_adam_train_step.mlir`, machine-checked by `check_adam_prefix`.
--
-- ⚠ Two things change in these bytes and both are intended: the BatchNorm world (per-example →
-- batch), and every parameter NAME (`%Ws`/`%We2`/`%Wfc` → `%sW`/`%b2eW`/`%Wd`, from `mnv2SigList`).
-- The driver binds positionally, so the rename reaches nothing.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2_fwd.mlir"
  (Proofs.StableHLO.mobilenetv2FwdFaithfulB 32 10 "1.0e-5")

-- The ImageNet forward, 64x1000. ⭐ It was split the same way and no audit could see it:
-- `check_adam_prefix`'s PAIRS list holds only the Imagenette names, which is exactly how
-- `resnet34in_fwd` hid in leg 1.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_fwd.mlir"
  (Proofs.StableHLO.mobilenetv2FwdFaithfulB 64 1000 "1.0e-5" "mobilenetv2in")

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
-- `feature_group_count` buys no exemption. Check with `scripts/probes/bf16_gate2.py`, never by grepping
-- the op line, which shows only the result type.
--
-- ⚠ The depthwise convs are ~13% of MNv2's step and bf16 is a mild LOSS on them in isolation
-- (0.86× at MNv2's own layers — cuDNN has a better f32 depthwise kernel on Ada). The win comes
-- from the 1×1 expand/project convs, which is why the reference sets `bf16Conv := true` here.
-- ▶ `planning/archive/bf16_renderer.md` §9.1: the doc's "depthwise nets won't pay" was REFUTED.
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
#guard !"adamdp64bf16".contains "do"
#guard !"adamdp64bf16".contains "acc"
#guard !"adamdp64bf16".startsWith "ema"

-- ── ▶ RMSProp: the optimizer the MobileNetV2 reference ACTUALLY USES ──────────────────────────
-- `planning/archive/recipe_gaps.md` §2: RMSProp is the ONLY gap between this net and the JAX reference's
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
-- ⭐⭐ **The bf16 twin of the line above, and the arm `mnv2-default-4gpu` should run.** Until now
-- this net's ONLY committed bf16 renders were the AdamW family, so the one job that trains it had
-- no bf16 arm at all and paid f32 for the whole 350-epoch schedule — the gap was in the render
-- matrix rather than in anything measured, which is the kind that survives longest because every
-- gate on the existing artifacts stays green while it does.
-- ⚠ RMSProp is the axis that matters here, not the precision: `.rmsprop` keeps ε INSIDE the root
-- and the mean-square initialised to 1.0 (TF's form), and the bf16 flag is orthogonal to both.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_rmsdp64bf16_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-5" 4 false "mobilenetv2in"
    Proofs.StableHLO.OptKind.rmsprop true)
-- The entry name is derived from the variant, so this guard is what stops the artifact declaring
-- `@mobilenetv2in_rmsdp64_train_step` inside a file named `…rmsdp64bf16…` — the load-time entry
-- mismatch three other nets have each shipped once.
#guard Proofs.StableHLO.mnv2AdamVariant 64 4 Proofs.StableHLO.OptKind.rmsprop true == "rmsdp64bf16"
-- ⭐ **The JAX reference's `full` recipe as of 0ccd6ad9, row for row** (BN still at ε 1e-5; the arm
-- `mnv2-default-4gpu` runs is its TF-slim BN peer `rmsdp64wxdols0eps0001bf16`, below).
-- `rmsdp64bf16` above differs from `mobilenetV2ImagenetConfigFull` in three ways: it decays BN γ/β
-- and biases, has no classifier dropout, and smooths labels at α = 0.1 where the reference (and
-- Sandler et al.) use none. This render closes all three: `wx`, `do` (keep 0.8 from
-- `mobilenetv2ImagenetVerified.dropoutKeep`) and `ls0`.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_rmsdp64wxdols0bf16_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-5" 4 false "mobilenetv2in"
    Proofs.StableHLO.OptKind.rmsprop true (wdExclude := true) (cd := true) (alpha := 0.0))
#guard Proofs.StableHLO.mnv2AdamVariant 64 4 .rmsprop true (wx := true) (cd := true) (alpha := 0.0)
  == "rmsdp64wxdols0bf16"
-- The driver reads the SAME string for its region layout: `do` must switch the mask slot on, and
-- nothing else may fire.
#guard "rmsdp64wxdols0bf16".contains "do" && "rmsdp64wxdols0bf16".contains "rms"
#guard !"rmsdp64wxdols0bf16".contains "drop" && !"rmsdp64wxdols0bf16".contains "acc"
#guard !"rmsdp64wxdols0bf16".startsWith "ema" && !"rmsdp64wxdols0bf16".contains "lamb"
#guard !"rmsdp64wxdols0bf16".contains "bce"
-- The **2-GPU** peer of the line above: `B := 128` per replica, so the global batch is still
-- 128×2 = 256 and the recipe, the steps/epoch and the LR all stay exactly what the 4×64 config
-- runs. That is what makes a 2-card wall-clock comparable to the 4-card one rather than a new
-- experiment. The batch is in the slug (`rmsdp128` vs `rmsdp64`), so the two artifacts cannot
-- collide even though both are `dp` renders of the same optimizer.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_rmsdp128_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 128 1000 "1.0e-5" 2 false "mobilenetv2in" .rmsprop)
-- ⭐⭐ **The TF-slim BatchNorm peer, and the arm `mnv2-default-4gpu` runs** (2026-09-25): `rmsdp64wxdols0bf16`
-- at BN ε = 1e-3, TF-slim's value, which the JAX `full` recipe now uses (with decay 0.997, a
-- host-side knob, and the staircase schedule with no warmup, both driver-side). ε is baked, so
-- this is a new artifact, and its eval forward is `mobilenetv2in_fwd_eval_eps0001.mlir`.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_rmsdp64wxdols0eps0001bf16_train_step.mlir"
  (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-3" 4 false "mobilenetv2in"
    Proofs.StableHLO.OptKind.rmsprop true (wdExclude := true) (cd := true) (alpha := 0.0))
#guard Proofs.StableHLO.mnv2AdamVariant 64 4 .rmsprop true (wx := true) (cd := true) (alpha := 0.0)
    (epsMarker := Proofs.StableHLO.bnEpsMarker "1.0e-3") == "rmsdp64wxdols0eps0001bf16"
#guard Proofs.StableHLO.bnEpsMarker "1.0e-5" == ""
#guard "rmsdp64wxdols0eps0001bf16".contains "do" && "rmsdp64wxdols0eps0001bf16".contains "rms"
#guard !"rmsdp64wxdols0eps0001bf16".contains "drop" && !"rmsdp64wxdols0eps0001bf16".contains "acc"
#guard !"rmsdp64wxdols0eps0001bf16".startsWith "ema" && !"rmsdp64wxdols0eps0001bf16".contains "lamb"
#guard !"rmsdp64wxdols0eps0001bf16".contains "bce"

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

#eval IO.FS.writeFile "verified_mlir/mobilenetv2_fwd_eval.mlir"
  (Proofs.StableHLO.mnv2FwdEvalFaithfulV 32 10 "1.0e-5")

-- ── MobileNetV2 on FULL 1000-class ImageNet, slug `mobilenetv2in` — 2026-08-02 ───────────────────────
-- The forward pair for the fifth and last scale-tier trainer (§2p). `B`/`nClasses` were already
-- parameters; the `slug` is new, and it is what stops these overwriting the 10-class pair the
-- 86.73% Imagenette run and the §2g prefix audit depend on. §2g is the reason to be careful here
-- specifically: `mobilenetv2_fwd` is the artifact that was found to be the WRONG BN WORLD, so this
-- net has already been burned once by a forward that did not match its train step.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_fwd_eval.mlir"
  (Proofs.StableHLO.mnv2FwdEvalFaithfulV 64 1000 "1.0e-5" false "mobilenetv2in")
-- The eval partner of `rmsdp64wxdols0eps0001bf16`: the same graph at TF-slim's BN ε = 1e-3. The
-- ε-1e-5 file above stays, because the checkpoints trained at 1e-5 score through it.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2in_fwd_eval_eps0001.mlir"
  (Proofs.StableHLO.mnv2FwdEvalFaithfulV 64 1000 "1.0e-3" false "mobilenetv2in")

-- The reduced 6-block render kept as a demo / stepping-stone (the worked foundation that built the
-- depthwise SGD core ops); NOT what the trainer reads.
