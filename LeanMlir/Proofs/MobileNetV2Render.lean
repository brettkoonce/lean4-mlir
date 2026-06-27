import LeanMlir.Proofs.StableHLO

/-! # MobileNetV2 train step rendered ENTIRELY from the verified AST

The Chapter-6 peer of `resnet34TrainStepFaithfulV` (`ResNet34Render.lean`), for the committed
reduced-6-block MobileNetV2 (the net `mobilenetv2Forward_full_pc` / `mobilenetv2FwdGraphFullPC`
compute). A strided stem (no maxpool), 6 inverted-residual blocks (`b1/b3/b5/b6` stride-2
downsample, `b2/b4` stride-1 with an `addV` skip), a 1×1 conv-bn-relu6 head, global-average-pool
and a final dense. `MainMobilenetV2Verified` trains on `verified_mlir/mobilenetv2_train_step.mlir`;
this renderer emits that file as `pretty(provenGraph)` — every line is `pretty` of a verified `SHlo`
node, so the committed bytes ARE the certified render.

**The depthwise wrinkle (vs ResNet's plain convs).** The inverted-residual body is
expand(1×1)→BN→relu6 → depthwise(3×3)→BN→relu6 → project(1×1)→BN (a LINEAR bottleneck — NO relu6
after project), so the block-output cotangent flows straight into the project-BN backward. The
depthwise param updates use the new `depthwiseWeightSgd`/`depthwiseStridedWeightSgd` ops (the
per-channel `batch_group_count = c` transpose-trick) and the depthwise bias ops
`depthwiseBiasSgd`/`depthwiseStridedBiasSgd` (whose `skel` aliases `convBiasSgd`'s spatial reduce);
the relu6 kink uses `selectMid` (the two-sided `0 < x < 6` mask) rather than ResNet's `selectPos`.

Render is value-independent (`skel` erases values), so the renderer passes placeholder zeros and
`lr := 0`/`ε := 0`; the emitted `lrStr`/`epsStr` literals carry the real values. The committed
trainer (`tests/TestMobilenetV2Train.lean`) uses `BS=32`, `ε=1.0e-5`, `lr=0.3`.

Strided vs stride-1 blocks are SEPARATE emitters (`irFwdStrided`/`irFwd`, `irBackStrided`/`irBack`),
each with CONCRETE spatial dims — this avoids `Vec (… if strided …)` placeholder-type mismatches. -/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-- Saved forward SSA names a block's backward + SGD passes reference. -/
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

/-- Backward result: code, the dx cotangent to the previous block, and the block's param-update
    output SSA names in func-arg order. -/
structure MBBack where
  code : String
  dx : String
  names : List String

-- ════════════════════════════════════════════════════════════════
-- § Block forward
--   inverted residual: expand(1×1)→BN→relu6 → depthwise(3×3)→BN→relu6 → project(1×1)→BN
-- ════════════════════════════════════════════════════════════════

/-- **STRIDED inverted-residual forward** (b1/b3/b5/b6): expand at the input `2hh×2ww`, depthwise
    downsamples `2hh×2ww → hh×ww`, project 1×1 at `hh×ww`. NO skip. -/
private def irFwdStrided (B ic mid oc hh : Nat) (epsStr p xName : String) : StateM Nat MBFwd := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (ic*(2*hh)*(2*ww)) := fun _ => 0
  let zeb  : Vec (mid*(2*hh)*(2*ww)) := fun _ => 0
  let zdb  : Vec (mid*hh*ww) := fun _ => 0
  let zob  : Vec (oc*hh*ww) := fun _ => 0
  let (cEc, nEc) ← pretty B (.flatConvF (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%We{p}" s!"%be{p}" zke zmid (.operand xName zxin))
  let (cEn, nEn) ← pretty B (.bnPerChannelF (oc := mid) (h := 2*hh) (w := 2*ww) s!"%ge{p}" s!"%bte{p}" epsStr 0 zmid zmid (.operand nEc zeb))
  let (cEr, nEr) ← pretty B (.relu6F (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.depthwiseStridedF (h := hh) (w := ww) s!"%Wd{p}" s!"%bd{p}" zdk zmid (.operand nEr zeb))
  let (cDn, nDn) ← pretty B (.bnPerChannelF (oc := mid) (h := hh) (w := ww) s!"%gd{p}" s!"%btd{p}" epsStr 0 zmid zmid (.operand nDc zdb))
  let (cDr, nDr) ← pretty B (.relu6F (.operand nDn zdb))
  let (cPc, nPc) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" s!"%bp{p}" zkp zoc (.operand nDr zdb))
  let (cPn, nPn) ← pretty B (.bnPerChannelF (oc := oc) (h := hh) (w := ww) s!"%gp{p}" s!"%btp{p}" epsStr 0 zoc zoc (.operand nPc zob))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc }

/-- **STRIDE-1 inverted-residual forward** (b2/b4): everything at `hh×ww`, with an `addV` skip on the
    block input (ic = oc). -/
private def irFwd (B ic mid oc hh : Nat) (epsStr p xName : String) : StateM Nat MBFwd := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (ic*hh*ww) := fun _ => 0
  let zeb  : Vec (mid*hh*ww) := fun _ => 0
  let zob  : Vec (oc*hh*ww) := fun _ => 0
  let (cEc, nEc) ← pretty B (.flatConvF (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%We{p}" s!"%be{p}" zke zmid (.operand xName zxin))
  let (cEn, nEn) ← pretty B (.bnPerChannelF (oc := mid) (h := hh) (w := ww) s!"%ge{p}" s!"%bte{p}" epsStr 0 zmid zmid (.operand nEc zeb))
  let (cEr, nEr) ← pretty B (.relu6F (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.depthwiseF (h := hh) (w := ww) s!"%Wd{p}" s!"%bd{p}" zdk zmid (.operand nEr zeb))
  let (cDn, nDn) ← pretty B (.bnPerChannelF (oc := mid) (h := hh) (w := ww) s!"%gd{p}" s!"%btd{p}" epsStr 0 zmid zmid (.operand nDc zeb))
  let (cDr, nDr) ← pretty B (.relu6F (.operand nDn zeb))
  let (cPc, nPc) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" s!"%bp{p}" zkp zoc (.operand nDr zeb))
  let (cPn, nPn) ← pretty B (.bnPerChannelF (oc := oc) (h := hh) (w := ww) s!"%gp{p}" s!"%btp{p}" epsStr 0 zoc zoc (.operand nPc zob))
  let (cA, nA) ← pretty B (.addV (.operand nPn zob) (.operand xName zob))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn ++ cA,
         o := nA, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc }

-- ════════════════════════════════════════════════════════════════
-- § Block backward + param SGD (project → depthwise → expand; dyOut → project-BN out directly)
-- ════════════════════════════════════════════════════════════════

/-- **STRIDED inverted-residual backward + 12 param SGD ops.** depthwise input grad at `2hh×2ww`;
    NO skip — the dx to the previous block is the body dx (at the `2hh×2ww` grid). -/
private def irBackStrided (B ic mid oc hh : Nat) (epsStr lrStr p xName : String)
    (f : MBFwd) (dyName : String) : StateM Nat MBBack := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxinV : Vec (ic*(2*hh)*(2*ww)) := fun _ => 0
  let zeb   : Vec (mid*(2*hh)*(2*ww)) := fun _ => 0
  let zdb   : Vec (mid*hh*ww) := fun _ => 0
  let zob   : Vec (oc*hh*ww) := fun _ => 0
  -- project: BN back (cot at project-conv out) → 1×1 conv back (cot at depthwise relu6 out)
  let (cDpc, nDpc) ← pretty B (.bnPerChannelBack (oc := oc) (h := hh) (w := ww) s!"%gp{p}" f.pc epsStr 0 zoc zob (.operand dyName zob))
  let (cDdr, nDdr) ← pretty B (.convBack (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" zkp zoc zdb (.operand nDpc zob))
  -- depthwise: relu6 mask (cot at depthwise-BN out) → BN back → strided depthwise back (cot at expand relu6 out, 2hh×2ww)
  let (cDdmask, nDdmask) ← pretty B (.selectMid f.dn zdb (.operand nDdr zdb))
  let (cDdn, nDdn) ← pretty B (.bnPerChannelBack (oc := mid) (h := hh) (w := ww) s!"%gd{p}" f.dc epsStr 0 zmid zdb (.operand nDdmask zdb))
  let (cDer, nDer) ← pretty B (.depthwiseStridedBack (h := hh) (w := ww) s!"%Wd{p}" zdk zmid zeb (.operand nDdn zdb))
  -- expand: relu6 mask (cot at expand-BN out) → BN back → 1×1 conv back (cot at block input, 2hh×2ww)
  let (cDemask, nDemask) ← pretty B (.selectMid f.en zeb (.operand nDer zeb))
  let (cDen, nDen) ← pretty B (.bnPerChannelBack (oc := mid) (h := 2*hh) (w := 2*ww) s!"%ge{p}" f.ec epsStr 0 zmid zeb (.operand nDemask zeb))
  let (cDxb, nDxb) ← pretty B (.convBack (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%We{p}" zke zmid zxinV (.operand nDen zeb))
  -- param SGD (func-arg order: We be ge bte | Wd bd gd btd | Wp bp gp btp)
  let (cWe, nWe) ← pretty B (.convWeightSgd xName s!"%We{p}" lrStr zmid (fun _ _ _ => 0 : Tensor3 ic (2*hh) (2*ww)) zke 0 (.operand nDen zeb))
  let (cbe, nbe) ← pretty B (.convBiasSgd s!"%be{p}" lrStr zke (fun _ _ _ => 0 : Tensor3 ic (2*hh) (2*ww)) zmid 0 (.operand nDen zeb))
  let (cge, nge) ← pretty B (.bnGammaSgd s!"%ge{p}" f.ec epsStr lrStr 0 zmid zeb 0 (.operand nDemask zeb))
  let (cte, nte) ← pretty B (.bnBetaSgd s!"%bte{p}" lrStr zmid 0 (.operand nDemask zeb))
  let (cWd, nWd) ← pretty B (.depthwiseStridedWeightSgd f.er s!"%Wd{p}" lrStr zmid zeb zdk 0 (.operand nDdn zdb))
  let (cbd, nbd) ← pretty B (.depthwiseStridedBiasSgd s!"%bd{p}" lrStr zdk zeb zmid 0 (.operand nDdn zdb))
  let (cgd, ngd) ← pretty B (.bnGammaSgd s!"%gd{p}" f.dc epsStr lrStr 0 zmid zdb 0 (.operand nDdmask zdb))
  let (ctd, ntd) ← pretty B (.bnBetaSgd s!"%btd{p}" lrStr zmid 0 (.operand nDdmask zdb))
  let (cWp, nWp) ← pretty B (.convWeightSgd f.dr s!"%Wp{p}" lrStr zoc (fun _ _ _ => 0 : Tensor3 mid hh ww) zkp 0 (.operand nDpc zob))
  let (cbp, nbp) ← pretty B (.convBiasSgd s!"%bp{p}" lrStr zkp (fun _ _ _ => 0 : Tensor3 mid hh ww) zoc 0 (.operand nDpc zob))
  let (cgp, ngp) ← pretty B (.bnGammaSgd s!"%gp{p}" f.pc epsStr lrStr 0 zoc zob 0 (.operand dyName zob))
  let (ctp, ntp) ← pretty B (.bnBetaSgd s!"%btp{p}" lrStr zoc 0 (.operand dyName zob))
  pure { code := cDpc ++ cDdr ++ cDdmask ++ cDdn ++ cDer ++ cDemask ++ cDen ++ cDxb ++
                 cWe ++ cbe ++ cge ++ cte ++ cWd ++ cbd ++ cgd ++ ctd ++ cWp ++ cbp ++ cgp ++ ctp,
         dx := nDxb, names := [nWe, nbe, nge, nte, nWd, nbd, ngd, ntd, nWp, nbp, ngp, ntp] }

/-- **STRIDE-1 inverted-residual backward + 12 param SGD ops.** Everything at `hh×ww`; the skip sums
    (body dx) + dyOut at the block input. -/
private def irBack (B ic mid oc hh : Nat) (epsStr lrStr p xName : String)
    (f : MBFwd) (dyName : String) : StateM Nat MBBack := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxinV : Vec (ic*hh*ww) := fun _ => 0
  let zeb   : Vec (mid*hh*ww) := fun _ => 0
  let zob   : Vec (oc*hh*ww) := fun _ => 0
  let (cDpc, nDpc) ← pretty B (.bnPerChannelBack (oc := oc) (h := hh) (w := ww) s!"%gp{p}" f.pc epsStr 0 zoc zob (.operand dyName zob))
  let (cDdr, nDdr) ← pretty B (.convBack (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" zkp zoc zeb (.operand nDpc zob))
  let (cDdmask, nDdmask) ← pretty B (.selectMid f.dn zeb (.operand nDdr zeb))
  let (cDdn, nDdn) ← pretty B (.bnPerChannelBack (oc := mid) (h := hh) (w := ww) s!"%gd{p}" f.dc epsStr 0 zmid zeb (.operand nDdmask zeb))
  let (cDer, nDer) ← pretty B (.depthwiseBack (h := hh) (w := ww) s!"%Wd{p}" zdk zmid zeb (.operand nDdn zeb))
  let (cDemask, nDemask) ← pretty B (.selectMid f.en zeb (.operand nDer zeb))
  let (cDen, nDen) ← pretty B (.bnPerChannelBack (oc := mid) (h := hh) (w := ww) s!"%ge{p}" f.ec epsStr 0 zmid zeb (.operand nDemask zeb))
  let (cDxb, nDxb) ← pretty B (.convBack (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%We{p}" zke zmid zxinV (.operand nDen zeb))
  -- skip fan-in: (body dx) + dyOut  (ic = oc, so dyOut lives at the block-input shape)
  let (cDx, nDx) ← pretty B (.addV (.operand nDxb zxinV) (.operand dyName zxinV))
  -- param SGD
  let (cWe, nWe) ← pretty B (.convWeightSgd xName s!"%We{p}" lrStr zmid (fun _ _ _ => 0 : Tensor3 ic hh ww) zke 0 (.operand nDen zeb))
  let (cbe, nbe) ← pretty B (.convBiasSgd s!"%be{p}" lrStr zke (fun _ _ _ => 0 : Tensor3 ic hh ww) zmid 0 (.operand nDen zeb))
  let (cge, nge) ← pretty B (.bnGammaSgd s!"%ge{p}" f.ec epsStr lrStr 0 zmid zeb 0 (.operand nDemask zeb))
  let (cte, nte) ← pretty B (.bnBetaSgd s!"%bte{p}" lrStr zmid 0 (.operand nDemask zeb))
  let (cWd, nWd) ← pretty B (.depthwiseWeightSgd f.er s!"%Wd{p}" lrStr zmid (fun _ _ _ => 0 : Tensor3 mid hh ww) zdk 0 (.operand nDdn zeb))
  let (cbd, nbd) ← pretty B (.depthwiseBiasSgd s!"%bd{p}" lrStr zdk (fun _ _ _ => 0 : Tensor3 mid hh ww) zmid 0 (.operand nDdn zeb))
  let (cgd, ngd) ← pretty B (.bnGammaSgd s!"%gd{p}" f.dc epsStr lrStr 0 zmid zeb 0 (.operand nDdmask zeb))
  let (ctd, ntd) ← pretty B (.bnBetaSgd s!"%btd{p}" lrStr zmid 0 (.operand nDdmask zeb))
  let (cWp, nWp) ← pretty B (.convWeightSgd f.dr s!"%Wp{p}" lrStr zoc (fun _ _ _ => 0 : Tensor3 mid hh ww) zkp 0 (.operand nDpc zob))
  let (cbp, nbp) ← pretty B (.convBiasSgd s!"%bp{p}" lrStr zkp (fun _ _ _ => 0 : Tensor3 mid hh ww) zoc 0 (.operand nDpc zob))
  let (cgp, ngp) ← pretty B (.bnGammaSgd s!"%gp{p}" f.pc epsStr lrStr 0 zoc zob 0 (.operand dyName zob))
  let (ctp, ntp) ← pretty B (.bnBetaSgd s!"%btp{p}" lrStr zoc 0 (.operand dyName zob))
  pure { code := cDpc ++ cDdr ++ cDdmask ++ cDdn ++ cDer ++ cDemask ++ cDen ++ cDxb ++ cDx ++
                 cWe ++ cbe ++ cge ++ cte ++ cWd ++ cbd ++ cgd ++ ctd ++ cWp ++ cbp ++ cgp ++ ctp,
         dx := nDx, names := [nWe, nbe, nge, nte, nWd, nbd, ngd, ntd, nWp, nbp, ngp, ntp] }

-- ════════════════════════════════════════════════════════════════
-- § NO-EXPAND block (b1): depthwise(stride-1, on `ic` ch)→BN→relu6 → project(1×1 ic→oc)→BN.
--   NO expand conv, NO skip. 8 params (Wd bd gd btd Wp bp gp btp).
-- ════════════════════════════════════════════════════════════════

/-- **NO-EXPAND inverted-residual forward** (b1): depthwise(stride-1, on `ic` channels)→BN→relu6
    → project(1×1 ic→oc)→BN. NO expand, NO skip. `f.er` = the depthwise INPUT (= block input
    `xName`), `f.dr` = the project input. (`ec`/`en` are unused for this block kind.) -/
private def irFwdNoExp (B ic oc hh : Nat) (epsStr p xName : String) : StateM Nat MBFwd := do
  let ww := hh
  let zic  : Vec ic := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zkp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel ic 3 3 := fun _ _ _ => 0
  let zib  : Vec (ic*hh*ww) := fun _ => 0
  let zob  : Vec (oc*hh*ww) := fun _ => 0
  let (cDc, nDc) ← pretty B (.depthwiseF (h := hh) (w := ww) s!"%Wd{p}" s!"%bd{p}" zdk zic (.operand xName zib))
  let (cDn, nDn) ← pretty B (.bnPerChannelF (oc := ic) (h := hh) (w := ww) s!"%gd{p}" s!"%btd{p}" epsStr 0 zic zic (.operand nDc zib))
  let (cDr, nDr) ← pretty B (.relu6F (.operand nDn zib))
  let (cPc, nPc) ← pretty B (.flatConvF (ic := ic) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" s!"%bp{p}" zkp zoc (.operand nDr zib))
  let (cPn, nPn) ← pretty B (.bnPerChannelF (oc := oc) (h := hh) (w := ww) s!"%gp{p}" s!"%btp{p}" epsStr 0 zoc zoc (.operand nPc zob))
  pure { code := cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := xName, en := xName, er := xName, dc := nDc, dn := nDn, dr := nDr, pc := nPc }

/-- **NO-EXPAND inverted-residual backward + 8 param SGD.** project-BN-back(dyOut) → project-conv-back
    → depthwise relu6 mask → depthwise-BN-back → depthwise-back → dx (= block input grad, `ic` ch).
    No skip fan-in. -/
private def irBackNoExp (B ic oc hh : Nat) (epsStr lrStr p xName : String)
    (f : MBFwd) (dyName : String) : StateM Nat MBBack := do
  let ww := hh
  let zic  : Vec ic := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zkp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel ic 3 3 := fun _ _ _ => 0
  let zib  : Vec (ic*hh*ww) := fun _ => 0
  let zob  : Vec (oc*hh*ww) := fun _ => 0
  -- project: BN back (cot at project-conv out) → 1×1 conv back (cot at depthwise relu6 out)
  let (cDpc, nDpc) ← pretty B (.bnPerChannelBack (oc := oc) (h := hh) (w := ww) s!"%gp{p}" f.pc epsStr 0 zoc zob (.operand dyName zob))
  let (cDdr, nDdr) ← pretty B (.convBack (ic := ic) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" zkp zoc zib (.operand nDpc zob))
  -- depthwise: relu6 mask (cot at depthwise-BN out) → BN back → depthwise back (cot at block input)
  let (cDdmask, nDdmask) ← pretty B (.selectMid f.dn zib (.operand nDdr zib))
  let (cDdn, nDdn) ← pretty B (.bnPerChannelBack (oc := ic) (h := hh) (w := ww) s!"%gd{p}" f.dc epsStr 0 zic zib (.operand nDdmask zib))
  let (cDxb, nDxb) ← pretty B (.depthwiseBack (h := hh) (w := ww) s!"%Wd{p}" zdk zic zib (.operand nDdn zib))
  -- param SGD (func-arg order: Wd bd gd btd | Wp bp gp btp)
  let (cWd, nWd) ← pretty B (.depthwiseWeightSgd xName s!"%Wd{p}" lrStr zic (fun _ _ _ => 0 : Tensor3 ic hh ww) zdk 0 (.operand nDdn zib))
  let (cbd, nbd) ← pretty B (.depthwiseBiasSgd s!"%bd{p}" lrStr zdk (fun _ _ _ => 0 : Tensor3 ic hh ww) zic 0 (.operand nDdn zib))
  let (cgd, ngd) ← pretty B (.bnGammaSgd s!"%gd{p}" f.dc epsStr lrStr 0 zic zib 0 (.operand nDdmask zib))
  let (ctd, ntd) ← pretty B (.bnBetaSgd s!"%btd{p}" lrStr zic 0 (.operand nDdmask zib))
  let (cWp, nWp) ← pretty B (.convWeightSgd f.dr s!"%Wp{p}" lrStr zoc (fun _ _ _ => 0 : Tensor3 ic hh ww) zkp 0 (.operand nDpc zob))
  let (cbp, nbp) ← pretty B (.convBiasSgd s!"%bp{p}" lrStr zkp (fun _ _ _ => 0 : Tensor3 ic hh ww) zoc 0 (.operand nDpc zob))
  let (cgp, ngp) ← pretty B (.bnGammaSgd s!"%gp{p}" f.pc epsStr lrStr 0 zoc zob 0 (.operand dyName zob))
  let (ctp, ntp) ← pretty B (.bnBetaSgd s!"%btp{p}" lrStr zoc 0 (.operand dyName zob))
  pure { code := cDpc ++ cDdr ++ cDdmask ++ cDdn ++ cDxb ++
                 cWd ++ cbd ++ cgd ++ ctd ++ cWp ++ cbp ++ cgp ++ ctp,
         dx := nDxb, names := [nWd, nbd, ngd, ntd, nWp, nbp, ngp, ntp] }

-- ════════════════════════════════════════════════════════════════
-- § EXPAND-NO-SKIP stride-1 block (b11, b17): == irFwd/irBack but NO addV skip.
--   block output = project-BN out directly; backward dx = expand-conv-back (no fan-in).
-- ════════════════════════════════════════════════════════════════

/-- **EXPAND-NO-SKIP stride-1 forward** (b11/b17): expand(1×1)→BN→relu6 → depthwise(3×3)→BN→relu6
    → project(1×1)→BN. Everything at `hh×ww`; `ic ≠ oc` so NO skip (block output = project-BN out). -/
private def irFwdNoSkip (B ic mid oc hh : Nat) (epsStr p xName : String) : StateM Nat MBFwd := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxin : Vec (ic*hh*ww) := fun _ => 0
  let zeb  : Vec (mid*hh*ww) := fun _ => 0
  let zob  : Vec (oc*hh*ww) := fun _ => 0
  let (cEc, nEc) ← pretty B (.flatConvF (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%We{p}" s!"%be{p}" zke zmid (.operand xName zxin))
  let (cEn, nEn) ← pretty B (.bnPerChannelF (oc := mid) (h := hh) (w := ww) s!"%ge{p}" s!"%bte{p}" epsStr 0 zmid zmid (.operand nEc zeb))
  let (cEr, nEr) ← pretty B (.relu6F (.operand nEn zeb))
  let (cDc, nDc) ← pretty B (.depthwiseF (h := hh) (w := ww) s!"%Wd{p}" s!"%bd{p}" zdk zmid (.operand nEr zeb))
  let (cDn, nDn) ← pretty B (.bnPerChannelF (oc := mid) (h := hh) (w := ww) s!"%gd{p}" s!"%btd{p}" epsStr 0 zmid zmid (.operand nDc zeb))
  let (cDr, nDr) ← pretty B (.relu6F (.operand nDn zeb))
  let (cPc, nPc) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" s!"%bp{p}" zkp zoc (.operand nDr zeb))
  let (cPn, nPn) ← pretty B (.bnPerChannelF (oc := oc) (h := hh) (w := ww) s!"%gp{p}" s!"%btp{p}" epsStr 0 zoc zoc (.operand nPc zob))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr, pc := nPc }

/-- **EXPAND-NO-SKIP stride-1 backward + 12 param SGD.** == `irBack` but NO skip fan-in: the dx to
    the previous block is the expand-conv-back directly (no `addV` with dyOut). -/
private def irBackNoSkip (B ic mid oc hh : Nat) (epsStr lrStr p xName : String)
    (f : MBFwd) (dyName : String) : StateM Nat MBBack := do
  let ww := hh
  let zmid : Vec mid := fun _ => 0
  let zoc  : Vec oc := fun _ => 0
  let zke  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zdk  : DepthwiseKernel mid 3 3 := fun _ _ _ => 0
  let zxinV : Vec (ic*hh*ww) := fun _ => 0
  let zeb   : Vec (mid*hh*ww) := fun _ => 0
  let zob   : Vec (oc*hh*ww) := fun _ => 0
  let (cDpc, nDpc) ← pretty B (.bnPerChannelBack (oc := oc) (h := hh) (w := ww) s!"%gp{p}" f.pc epsStr 0 zoc zob (.operand dyName zob))
  let (cDdr, nDdr) ← pretty B (.convBack (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%Wp{p}" zkp zoc zeb (.operand nDpc zob))
  let (cDdmask, nDdmask) ← pretty B (.selectMid f.dn zeb (.operand nDdr zeb))
  let (cDdn, nDdn) ← pretty B (.bnPerChannelBack (oc := mid) (h := hh) (w := ww) s!"%gd{p}" f.dc epsStr 0 zmid zeb (.operand nDdmask zeb))
  let (cDer, nDer) ← pretty B (.depthwiseBack (h := hh) (w := ww) s!"%Wd{p}" zdk zmid zeb (.operand nDdn zeb))
  let (cDemask, nDemask) ← pretty B (.selectMid f.en zeb (.operand nDer zeb))
  let (cDen, nDen) ← pretty B (.bnPerChannelBack (oc := mid) (h := hh) (w := ww) s!"%ge{p}" f.ec epsStr 0 zmid zeb (.operand nDemask zeb))
  let (cDxb, nDxb) ← pretty B (.convBack (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%We{p}" zke zmid zxinV (.operand nDen zeb))
  -- param SGD (no skip fan-in; dx = nDxb directly)
  let (cWe, nWe) ← pretty B (.convWeightSgd xName s!"%We{p}" lrStr zmid (fun _ _ _ => 0 : Tensor3 ic hh ww) zke 0 (.operand nDen zeb))
  let (cbe, nbe) ← pretty B (.convBiasSgd s!"%be{p}" lrStr zke (fun _ _ _ => 0 : Tensor3 ic hh ww) zmid 0 (.operand nDen zeb))
  let (cge, nge) ← pretty B (.bnGammaSgd s!"%ge{p}" f.ec epsStr lrStr 0 zmid zeb 0 (.operand nDemask zeb))
  let (cte, nte) ← pretty B (.bnBetaSgd s!"%bte{p}" lrStr zmid 0 (.operand nDemask zeb))
  let (cWd, nWd) ← pretty B (.depthwiseWeightSgd f.er s!"%Wd{p}" lrStr zmid (fun _ _ _ => 0 : Tensor3 mid hh ww) zdk 0 (.operand nDdn zeb))
  let (cbd, nbd) ← pretty B (.depthwiseBiasSgd s!"%bd{p}" lrStr zdk (fun _ _ _ => 0 : Tensor3 mid hh ww) zmid 0 (.operand nDdn zeb))
  let (cgd, ngd) ← pretty B (.bnGammaSgd s!"%gd{p}" f.dc epsStr lrStr 0 zmid zeb 0 (.operand nDdmask zeb))
  let (ctd, ntd) ← pretty B (.bnBetaSgd s!"%btd{p}" lrStr zmid 0 (.operand nDdmask zeb))
  let (cWp, nWp) ← pretty B (.convWeightSgd f.dr s!"%Wp{p}" lrStr zoc (fun _ _ _ => 0 : Tensor3 mid hh ww) zkp 0 (.operand nDpc zob))
  let (cbp, nbp) ← pretty B (.convBiasSgd s!"%bp{p}" lrStr zkp (fun _ _ _ => 0 : Tensor3 mid hh ww) zoc 0 (.operand nDpc zob))
  let (cgp, ngp) ← pretty B (.bnGammaSgd s!"%gp{p}" f.pc epsStr lrStr 0 zoc zob 0 (.operand dyName zob))
  let (ctp, ntp) ← pretty B (.bnBetaSgd s!"%btp{p}" lrStr zoc 0 (.operand dyName zob))
  pure { code := cDpc ++ cDdr ++ cDdmask ++ cDdn ++ cDer ++ cDemask ++ cDen ++ cDxb ++
                 cWe ++ cbe ++ cge ++ cte ++ cWd ++ cbd ++ cgd ++ ctd ++ cWp ++ cbp ++ cgp ++ ctp,
         dx := nDxb, names := [nWe, nbe, nge, nte, nWd, nbd, ngd, ntd, nWp, nbp, ngp, ntp] }

-- ════════════════════════════════════════════════════════════════
-- § Param signature lists (func-arg order — names + types, shared by sig + return types)
-- ════════════════════════════════════════════════════════════════

private def irSig (p : String) (ic mid oc : Nat) : List (String × String) :=
  [(s!"%We{p}", ty [mid,ic,1,1]), (s!"%be{p}", ty [mid]), (s!"%ge{p}", ty [mid]), (s!"%bte{p}", ty [mid]),
   (s!"%Wd{p}", ty [mid,1,3,3]), (s!"%bd{p}", ty [mid]), (s!"%gd{p}", ty [mid]), (s!"%btd{p}", ty [mid]),
   (s!"%Wp{p}", ty [oc,mid,1,1]), (s!"%bp{p}", ty [oc]), (s!"%gp{p}", ty [oc]), (s!"%btp{p}", ty [oc])]

/-- **NO-EXPAND block sig** (b1): depthwise on `ic` channels + project `ic→oc`. 8 params. -/
private def irSigNoExp (p : String) (ic oc : Nat) : List (String × String) :=
  [(s!"%Wd{p}", ty [ic,1,3,3]), (s!"%bd{p}", ty [ic]), (s!"%gd{p}", ty [ic]), (s!"%btd{p}", ty [ic]),
   (s!"%Wp{p}", ty [oc,ic,1,1]), (s!"%bp{p}", ty [oc]), (s!"%gp{p}", ty [oc]), (s!"%btp{p}", ty [oc])]

/-- **FULL 17-block paper param signature**, func-arg order: stem (4) + b1 no-exp (8) +
    b2..b17 inverted-residual (16×12) + head (4) + dense (2) = 4+8+192+4+2 = 210 tensors. -/
private def paperSig (nClasses : Nat) : List (String × String) :=
  [("%Ws", ty [32,3,3,3]), ("%bs", ty [32]), ("%gs", ty [32]), ("%bts", ty [32])] ++
  irSigNoExp "1" 32 16 ++
  irSig "2"  16  96  24 ++ irSig "3"  24 144  24 ++ irSig "4"  24 144  32 ++
  irSig "5"  32 192  32 ++ irSig "6"  32 192  32 ++ irSig "7"  32 192  64 ++
  irSig "8"  64 384  64 ++ irSig "9"  64 384  64 ++ irSig "10" 64 384  64 ++
  irSig "11" 64 384  96 ++ irSig "12" 96 576  96 ++ irSig "13" 96 576  96 ++
  irSig "14" 96 576 160 ++ irSig "15" 160 960 160 ++ irSig "16" 160 960 160 ++
  irSig "17" 160 960 320 ++
  [("%Wh", ty [1280,320,1,1]), ("%bh", ty [1280]), ("%gh", ty [1280]), ("%bth", ty [1280])] ++
  [("%Wfc", ty [1280, nClasses]), ("%bfc", ty [nClasses])]

-- ════════════════════════════════════════════════════════════════
-- § The whole-net renderer
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 1000000 in
/-- **Reduced-6-block MobileNetV2 train step rendered ENTIRELY from the verified AST.** `B` batch,
    `nClasses` outputs (=10 for the committed trainer). Every emitted line is `pretty` of a verified
    `SHlo` node. Strided stem 3×3/s2 (3→16, 224→112, NO maxpool), 6 inverted-residual blocks
    (b1/b3/b5/b6 stride-2 downsample, b2/b4 stride-1 skip), 1×1 conv-bn-relu6 head → GAP → dense. -/
def mnv2TrainStepFaithfulV (B nClasses : Nat) (epsStr lrStr : String) : String :=
  let go : StateM Nat String := do
    -- ═══ stem: 3×3/s2 conv (3→16, 224→112) → BN → relu6 (NO maxpool) ═══
    let zx   : Vec (3*224*224) := fun _ => 0
    let zSk  : Kernel4 16 3 3 3 := fun _ _ _ _ => 0
    let z16  : Vec 16 := fun _ => 0
    let z112 : Vec (16*112*112) := fun _ => 0
    let (cStc, nStc) ← pretty B (.flatConvStridedF (ic := 3) (oc := 16) (h := 112) (w := 112) "%Ws" "%bs" zSk z16 (.operand "%x" zx))
    let (cStn, nStn) ← pretty B (.bnPerChannelF (oc := 16) (h := 112) (w := 112) "%gs" "%bts" epsStr 0 z16 z16 (.operand nStc z112))
    let (cStr, nStr) ← pretty B (.relu6F (.operand nStn z112))
    -- ═══ forward: 6 inverted-residual blocks (ic, mid, oc, outH) ═══
    let f1 ← irFwdStrided B 16  64  24 56 epsStr "1" nStr
    let f2 ← irFwd        B 24  96  24 56 epsStr "2" f1.o
    let f3 ← irFwdStrided B 24  96  32 28 epsStr "3" f2.o
    let f4 ← irFwd        B 32 128  32 28 epsStr "4" f3.o
    let f5 ← irFwdStrided B 32 128  64 14 epsStr "5" f4.o
    let f6 ← irFwdStrided B 64 256  64  7 epsStr "6" f5.o
    -- ═══ head: 1×1 conv (64→128) → BN → relu6 → GAP(7×7) → dense(128→nClasses) → softmax-CE cot ═══
    let z7   : Vec (64*7*7) := fun _ => 0
    let zHk  : Kernel4 128 64 1 1 := fun _ _ _ _ => 0
    let z128 : Vec 128 := fun _ => 0
    let zH7  : Vec (128*7*7) := fun _ => 0
    let zWd  : Mat 128 nClasses := fun _ _ => 0
    let zNC  : Vec nClasses := fun _ => 0
    let zHxT : Tensor3 64 7 7 := fun _ _ _ => 0
    let (cHc, nHc) ← pretty B (.flatConvF (ic := 64) (oc := 128) (h := 7) (w := 7) "%Wh" "%bh" zHk z128 (.operand f6.o z7))
    let (cHn, nHn) ← pretty B (.bnPerChannelF (oc := 128) (h := 7) (w := 7) "%gh" "%bth" epsStr 0 z128 z128 (.operand nHc zH7))
    let (cHr, nHr) ← pretty B (.relu6F (.operand nHn zH7))
    let (cGap, nGap) ← pretty B (.gapF (c := 128) (h := 7) (w := 7) (.operand nHr zH7))
    let (cLog, nLog) ← pretty B (denseF "%Wfc" "%bfc" zWd zNC (.operand nGap z128))
    let (cDy,  nDy)  ← pretty B (.sub (.softmaxDiv (.expe (.operand nLog zNC))) (.operand "%onehot" zNC))
    -- ═══ head backward: dense input-grad → GAP-back → relu6 mask → BN-back → 1×1 conv back ═══
    let (cDg,  nDg)  ← pretty B (.dotOut "%Wfc" zWd (.operand nDy zNC))
    let (cDgi, nDgi) ← pretty B (.gapBack (c := 128) (h := 7) (w := 7) (.operand nDg z128))
    let (cWfc, nWfc) ← pretty B (.weightSgd nGap "%Wfc" lrStr z128 zWd 0 (.operand nDy zNC))
    let (cbfc, nbfc) ← pretty B (.biasSgd "%bfc" lrStr zNC 0 (.operand nDy zNC))
    let (cDhmask, nDhmask) ← pretty B (.selectMid nHn zH7 (.operand nDgi zH7))
    let (cDhn, nDhn) ← pretty B (.bnPerChannelBack (oc := 128) (h := 7) (w := 7) "%gh" nHc epsStr 0 z128 zH7 (.operand nDhmask zH7))
    let (cDhx, nDhx) ← pretty B (.convBack (ic := 64) (oc := 128) (h := 7) (w := 7) "%Wh" zHk z128 z7 (.operand nDhn zH7))
    let (cWh, nWh) ← pretty B (.convWeightSgd f6.o "%Wh" lrStr z128 zHxT zHk 0 (.operand nDhn zH7))
    let (cbh, nbh) ← pretty B (.convBiasSgd "%bh" lrStr zHk zHxT z128 0 (.operand nDhn zH7))
    let (cgh, ngh) ← pretty B (.bnGammaSgd "%gh" nHc epsStr lrStr 0 z128 zH7 0 (.operand nDhmask zH7))
    let (cth, nth) ← pretty B (.bnBetaSgd "%bth" lrStr z128 0 (.operand nDhmask zH7))
    -- ═══ backward: 6 blocks reversed (cotangent threads from nDhx) ═══
    let b6 ← irBackStrided B 64 256  64  7 epsStr lrStr "6" f5.o f6 nDhx
    let b5 ← irBackStrided B 32 128  64 14 epsStr lrStr "5" f4.o f5 b6.dx
    let b4 ← irBack        B 32 128  32 28 epsStr lrStr "4" f3.o f4 b5.dx
    let b3 ← irBackStrided B 24  96  32 28 epsStr lrStr "3" f2.o f3 b4.dx
    let b2 ← irBack        B 24  96  24 56 epsStr lrStr "2" f1.o f2 b3.dx
    let b1 ← irBackStrided B 16  64  24 56 epsStr lrStr "1" nStr f1 b2.dx
    -- ═══ stem backward: relu6 mask → BN-back, then stem param SGD (NO conv-back past %x) ═══
    let (cDsr, nDsr) ← pretty B (.selectMid nStn z112 (.operand b1.dx z112))
    let (cDsn, nDsn) ← pretty B (.bnPerChannelBack (oc := 16) (h := 112) (w := 112) "%gs" nStc epsStr 0 z16 z112 (.operand nDsr z112))
    let (csW, nsW) ← pretty B (.convStridedWeightSgd "%x" "%Ws" lrStr z16 zx zSk 0 (.operand nDsn z112))
    let (csb, nsb) ← pretty B (.convStridedBiasSgd "%bs" lrStr zSk zx z16 0 (.operand nDsn z112))
    let (csg, nsg) ← pretty B (.bnGammaSgd "%gs" nStc epsStr lrStr 0 z16 z112 0 (.operand nDsr z112))
    let (cst, nst) ← pretty B (.bnBetaSgd "%bts" lrStr z16 0 (.operand nDsr z112))
    -- ═══ assemble body + return (params in func-arg order: stem, blocks fwd-order, head, dense) ═══
    let fwdCode := cStc ++ cStn ++ cStr ++
      f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++
      cHc ++ cHn ++ cHr ++ cGap ++ cLog ++ cDy
    let bwdCode := cDg ++ cDgi ++ cWfc ++ cbfc ++
      cDhmask ++ cDhn ++ cDhx ++ cWh ++ cbh ++ cgh ++ cth ++
      b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDsr ++ cDsn ++ csW ++ csb ++ csg ++ cst
    let outNames : List String :=
      [nsW, nsb, nsg, nst] ++
      b1.names ++ b2.names ++ b3.names ++ b4.names ++ b5.names ++ b6.names ++
      [nWh, nbh, ngh, nth] ++ [nWfc, nbfc]
    let outTypes : List String :=
      (([("%Ws", ty [16,3,3,3]), ("%bs", ty [16]), ("%gs", ty [16]), ("%bts", ty [16])] :
          List (String × String)) ++
        irSig "1" 16 64 24 ++ irSig "2" 24 96 24 ++ irSig "3" 24 96 32 ++
        irSig "4" 32 128 32 ++ irSig "5" 32 128 64 ++ irSig "6" 64 256 64 ++
        [("%Wh", ty [128,64,1,1]), ("%bh", ty [128]), ("%gh", ty [128]), ("%bth", ty [128])] ++
        [("%Wfc", ty [128, nClasses]), ("%bfc", ty [nClasses])]).map (·.2)
    pure <|
      "    // ── MobileNetV2 train step: every line is pretty(verified AST node) ──\n" ++
      fwdCode ++ bwdCode ++
      s!"    return {String.intercalate ", " outNames} : {String.intercalate ", " outTypes}\n"
  -- func signature: %x, all params (forward-arg order), %onehot
  let sigList : List (String × String) :=
    [("%Ws", ty [16,3,3,3]), ("%bs", ty [16]), ("%gs", ty [16]), ("%bts", ty [16])] ++
    irSig "1" 16 64 24 ++ irSig "2" 24 96 24 ++ irSig "3" 24 96 32 ++
    irSig "4" 32 128 32 ++ irSig "5" 32 128 64 ++ irSig "6" 64 256 64 ++
    [("%Wh", ty [128,64,1,1]), ("%bh", ty [128]), ("%gh", ty [128]), ("%bth", ty [128])] ++
    [("%Wfc", ty [128, nClasses]), ("%bfc", ty [nClasses])]
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}")) ++
    s!", %onehot: {ty [B, nClasses]}"
  let outSig := String.intercalate ", " (sigList.map (·.2))
  let inner : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @mobilenetv2_train_step({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § The FULL 17-block paper-spec renderer
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000000 in
/-- **FULL paper-spec (17-block) MobileNetV2 train step rendered ENTIRELY from the verified AST.**
    The whole-net peer of `mnv2TrainStepFaithfulV`, scaled to the real `[t,c,n,s]` table. Every
    emitted line is `pretty` of a verified `SHlo` node. Strided stem 3×3/s2 (3→32, 224→112, NO
    maxpool) → b1 (no-expand t=1, 32→16) → b2..b17 (4 stride-2 downsamples, 10 identity skips,
    2 stage-first widenings) → 1×1 conv-bn-relu6 head (320→1280) → GAP → dense (1280→nClasses).
    Per-block prefixes are the block index (`1`..`17`), matching `irSig`/`irSigNoExp`. -/
def mnv2TrainStepFaithfulVPaper (B nClasses : Nat) (epsStr lrStr : String)
    (funcName : String := "mobilenetv2_paper_train_step") : String :=
  let go : StateM Nat String := do
    -- ═══ stem: 3×3/s2 conv (3→32, 224→112) → BN → relu6 (NO maxpool) ═══
    let zx   : Vec (3*224*224) := fun _ => 0
    let zSk  : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let z32  : Vec 32 := fun _ => 0
    let z112 : Vec (32*112*112) := fun _ => 0
    let (cStc, nStc) ← pretty B (.flatConvStridedF (ic := 3) (oc := 32) (h := 112) (w := 112) "%Ws" "%bs" zSk z32 (.operand "%x" zx))
    let (cStn, nStn) ← pretty B (.bnPerChannelF (oc := 32) (h := 112) (w := 112) "%gs" "%bts" epsStr 0 z32 z32 (.operand nStc z112))
    let (cStr, nStr) ← pretty B (.relu6F (.operand nStn z112))
    -- ═══ forward: 17 inverted-residual blocks ═══
    let f1  ← irFwdNoExp   B 32      16 112 epsStr "1"  nStr
    let f2  ← irFwdStrided B 16  96  24  56 epsStr "2"  f1.o
    let f3  ← irFwd        B 24 144  24  56 epsStr "3"  f2.o
    let f4  ← irFwdStrided B 24 144  32  28 epsStr "4"  f3.o
    let f5  ← irFwd        B 32 192  32  28 epsStr "5"  f4.o
    let f6  ← irFwd        B 32 192  32  28 epsStr "6"  f5.o
    let f7  ← irFwdStrided B 32 192  64  14 epsStr "7"  f6.o
    let f8  ← irFwd        B 64 384  64  14 epsStr "8"  f7.o
    let f9  ← irFwd        B 64 384  64  14 epsStr "9"  f8.o
    let f10 ← irFwd        B 64 384  64  14 epsStr "10" f9.o
    let f11 ← irFwdNoSkip  B 64 384  96  14 epsStr "11" f10.o
    let f12 ← irFwd        B 96 576  96  14 epsStr "12" f11.o
    let f13 ← irFwd        B 96 576  96  14 epsStr "13" f12.o
    let f14 ← irFwdStrided B 96 576 160   7 epsStr "14" f13.o
    let f15 ← irFwd        B 160 960 160   7 epsStr "15" f14.o
    let f16 ← irFwd        B 160 960 160   7 epsStr "16" f15.o
    let f17 ← irFwdNoSkip  B 160 960 320   7 epsStr "17" f16.o
    -- ═══ head: 1×1 conv (320→1280) → BN → relu6 → GAP(7×7) → dense(1280→nClasses) → softmax-CE cot ═══
    let z7    : Vec (320*7*7) := fun _ => 0
    let zHk   : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
    let z1280 : Vec 1280 := fun _ => 0
    let zH7   : Vec (1280*7*7) := fun _ => 0
    let zWd   : Mat 1280 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    let zHxT  : Tensor3 320 7 7 := fun _ _ _ => 0
    let (cHc, nHc) ← pretty B (.flatConvF (ic := 320) (oc := 1280) (h := 7) (w := 7) "%Wh" "%bh" zHk z1280 (.operand f17.o z7))
    let (cHn, nHn) ← pretty B (.bnPerChannelF (oc := 1280) (h := 7) (w := 7) "%gh" "%bth" epsStr 0 z1280 z1280 (.operand nHc zH7))
    let (cHr, nHr) ← pretty B (.relu6F (.operand nHn zH7))
    let (cGap, nGap) ← pretty B (.gapF (c := 1280) (h := 7) (w := 7) (.operand nHr zH7))
    let (cLog, nLog) ← pretty B (denseF "%Wfc" "%bfc" zWd zNC (.operand nGap z1280))
    let (cDy,  nDy)  ← pretty B (.sub (.softmaxDiv (.expe (.operand nLog zNC))) (.operand "%onehot" zNC))
    -- ═══ head backward: dense input-grad → GAP-back → relu6 mask → BN-back → 1×1 conv back ═══
    let (cDg,  nDg)  ← pretty B (.dotOut "%Wfc" zWd (.operand nDy zNC))
    let (cDgi, nDgi) ← pretty B (.gapBack (c := 1280) (h := 7) (w := 7) (.operand nDg z1280))
    let (cWfc, nWfc) ← pretty B (.weightSgd nGap "%Wfc" lrStr z1280 zWd 0 (.operand nDy zNC))
    let (cbfc, nbfc) ← pretty B (.biasSgd "%bfc" lrStr zNC 0 (.operand nDy zNC))
    let (cDhmask, nDhmask) ← pretty B (.selectMid nHn zH7 (.operand nDgi zH7))
    let (cDhn, nDhn) ← pretty B (.bnPerChannelBack (oc := 1280) (h := 7) (w := 7) "%gh" nHc epsStr 0 z1280 zH7 (.operand nDhmask zH7))
    let (cDhx, nDhx) ← pretty B (.convBack (ic := 320) (oc := 1280) (h := 7) (w := 7) "%Wh" zHk z1280 z7 (.operand nDhn zH7))
    let (cWh, nWh) ← pretty B (.convWeightSgd f17.o "%Wh" lrStr z1280 zHxT zHk 0 (.operand nDhn zH7))
    let (cbh, nbh) ← pretty B (.convBiasSgd "%bh" lrStr zHk zHxT z1280 0 (.operand nDhn zH7))
    let (cgh, ngh) ← pretty B (.bnGammaSgd "%gh" nHc epsStr lrStr 0 z1280 zH7 0 (.operand nDhmask zH7))
    let (cth, nth) ← pretty B (.bnBetaSgd "%bth" lrStr z1280 0 (.operand nDhmask zH7))
    -- ═══ backward: 17 blocks reversed (cotangent threads from nDhx) ═══
    let b17 ← irBackNoSkip  B 160 960 320   7 epsStr lrStr "17" f16.o f17 nDhx
    let b16 ← irBack        B 160 960 160   7 epsStr lrStr "16" f15.o f16 b17.dx
    let b15 ← irBack        B 160 960 160   7 epsStr lrStr "15" f14.o f15 b16.dx
    let b14 ← irBackStrided B 96 576 160   7 epsStr lrStr "14" f13.o f14 b15.dx
    let b13 ← irBack        B 96 576  96  14 epsStr lrStr "13" f12.o f13 b14.dx
    let b12 ← irBack        B 96 576  96  14 epsStr lrStr "12" f11.o f12 b13.dx
    let b11 ← irBackNoSkip  B 64 384  96  14 epsStr lrStr "11" f10.o f11 b12.dx
    let b10 ← irBack        B 64 384  64  14 epsStr lrStr "10" f9.o  f10 b11.dx
    let b9  ← irBack        B 64 384  64  14 epsStr lrStr "9"  f8.o  f9  b10.dx
    let b8  ← irBack        B 64 384  64  14 epsStr lrStr "8"  f7.o  f8  b9.dx
    let b7  ← irBackStrided B 32 192  64  14 epsStr lrStr "7"  f6.o  f7  b8.dx
    let b6  ← irBack        B 32 192  32  28 epsStr lrStr "6"  f5.o  f6  b7.dx
    let b5  ← irBack        B 32 192  32  28 epsStr lrStr "5"  f4.o  f5  b6.dx
    let b4  ← irBackStrided B 24 144  32  28 epsStr lrStr "4"  f3.o  f4  b5.dx
    let b3  ← irBack        B 24 144  24  56 epsStr lrStr "3"  f2.o  f3  b4.dx
    let b2  ← irBackStrided B 16  96  24  56 epsStr lrStr "2"  f1.o  f2  b3.dx
    let b1  ← irBackNoExp   B 32      16 112 epsStr lrStr "1"  nStr  f1  b2.dx
    -- ═══ stem backward: relu6 mask → BN-back, then stem param SGD (NO conv-back past %x) ═══
    let (cDsr, nDsr) ← pretty B (.selectMid nStn z112 (.operand b1.dx z112))
    let (cDsn, nDsn) ← pretty B (.bnPerChannelBack (oc := 32) (h := 112) (w := 112) "%gs" nStc epsStr 0 z32 z112 (.operand nDsr z112))
    let (csW, nsW) ← pretty B (.convStridedWeightSgd "%x" "%Ws" lrStr z32 zx zSk 0 (.operand nDsn z112))
    let (csb, nsb) ← pretty B (.convStridedBiasSgd "%bs" lrStr zSk zx z32 0 (.operand nDsn z112))
    let (csg, nsg) ← pretty B (.bnGammaSgd "%gs" nStc epsStr lrStr 0 z32 z112 0 (.operand nDsr z112))
    let (cst, nst) ← pretty B (.bnBetaSgd "%bts" lrStr z32 0 (.operand nDsr z112))
    -- ═══ assemble body + return (params in func-arg order: stem, blocks fwd-order, head, dense) ═══
    let fwdCode := cStc ++ cStn ++ cStr ++
      f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++ f8.code ++
      f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++ f15.code ++
      f16.code ++ f17.code ++
      cHc ++ cHn ++ cHr ++ cGap ++ cLog ++ cDy
    let bwdCode := cDg ++ cDgi ++ cWfc ++ cbfc ++
      cDhmask ++ cDhn ++ cDhx ++ cWh ++ cbh ++ cgh ++ cth ++
      b17.code ++ b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++
      b10.code ++ b9.code ++ b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++
      b2.code ++ b1.code ++
      cDsr ++ cDsn ++ csW ++ csb ++ csg ++ cst
    let outNames : List String :=
      [nsW, nsb, nsg, nst] ++
      b1.names ++ b2.names ++ b3.names ++ b4.names ++ b5.names ++ b6.names ++ b7.names ++
      b8.names ++ b9.names ++ b10.names ++ b11.names ++ b12.names ++ b13.names ++ b14.names ++
      b15.names ++ b16.names ++ b17.names ++
      [nWh, nbh, ngh, nth] ++ [nWfc, nbfc]
    let outTypes : List String := (paperSig nClasses).map (·.2)
    pure <|
      "    // ── MobileNetV2 (17-block paper) train step: every line is pretty(verified AST node) ──\n" ++
      fwdCode ++ bwdCode ++
      s!"    return {String.intercalate ", " outNames} : {String.intercalate ", " outTypes}\n"
  -- func signature: %x, all params (forward-arg order), %onehot
  let sigList : List (String × String) := paperSig nClasses
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}")) ++
    s!", %onehot: {ty [B, nClasses]}"
  let outSig := String.intercalate ", " (sigList.map (·.2))
  let inner : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @{funcName}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- Regenerate `verified_mlir/mobilenetv2_train_step.mlir` (what MainMobilenetV2Verified trains on)
-- from the faithful renderer: the FULL 17-block paper-spec net (210 params, canonical t=1 no-expand
-- b1, matching mobilenetv2Verified's 210-param spec). B=32, nClasses=10, ε=1e-5, lr=0.3.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2_train_step.mlir"
  (Proofs.StableHLO.mnv2TrainStepFaithfulVPaper 32 10 "1.0e-5" "0.3" "mobilenetv2_train_step")

-- The reduced 6-block render kept as a demo / stepping-stone (the worked foundation that built the
-- depthwise SGD core ops); NOT what the trainer reads.
#eval IO.FS.writeFile "verified_mlir/mobilenetv2_reduced_train_step.mlir"
  (Proofs.StableHLO.mnv2TrainStepFaithfulV 32 10 "1.0e-5" "0.3")
