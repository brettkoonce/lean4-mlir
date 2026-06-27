import LeanMlir.Proofs.StableHLO

/-! # EfficientNet-B0 train step rendered ENTIRELY from the verified AST (batched)

The Chapter-7 peer of `mnv2TrainStepFaithfulVPaper` (`MobileNetV2Render.lean`), for the committed
full-16-MBConv EfficientNet-B0 (262 params, the real `[t,c,n,s,k]` B0 spec). Unlike MobileNetV2/
ResNet34 (per-example `SHlo` index, batch threaded only in emit), EfficientNet emits **true
batch-norm**, which couples the batch — so the whole net lives at the **batched index** `N·(c·h·w)`
(`StableHLO.batchOp`/`bnBatchF`/the batched backward + param-SGD ops, all Item B).

**The SE wrinkle (vs MobileNetV2's relu6 blocks).** Each MBConv has a squeeze-excite gate
`x ⊙ sigmoid(dense W₂ (swish (dense W₁ (GAP x))))`, and the committed trainer **trains all 4 SE dense
params**. The fused `batchOp seBlock` / `seBackBatched` give the forward value + the SE *input*
cotangent but NOT the SE param grads, so the renderer **un-fuses** SE: it keeps the fused `seBlock`
for the forward `out` and ADDITIONALLY emits the un-fused gate subnet `s = batchOp gap → e1 = batchOp
dense W₁ → z = swishF → e2 = batchOp dense W₂` (only to expose `s/e1/z/e2`); the SE param grads chain
`seReduceB → sigmoidBack(e2) → denseWeightSgdB/denseBiasSgdB (W₂) → denseRowBack(W₂) → swishBack(e1)
→ denseWeightSgdB/denseBiasSgdB (W₁)`, and `dx` reuses the fused `seBackBatched`. Activations
are **swish** (smooth, no relu6 kink), the head GAP-back uses the batched `gapBackBatched`.

Render is value-independent (`skel` erases values), so placeholder zeros + `lr := 0`/`ε := 0` are
passed; the emitted `lrStr`/`epsStr` literals carry the real values. -/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-! The renderer builds the SHlo graph at the **single-batch-unit** index `N=1` (so the per-example
pointwise ops `swishF`/`sigmoidF` emit `[B, c·h·w]`), while the emit's `B` parameter supplies the real
batch dimension — the `batchOp`/`bnBatch*`/param-SGD emits read their per-example dims from `info`
and `B` from the emit, ignoring the (=1) `N` in the type. So every `(N := 1)` below is the SHlo
batch-unit; `pretty B` carries the actual batch. -/

/-- Saved forward SSA names a block's backward + SGD passes reference. -/
structure EFwd where
  code : String
  o  : String        -- block output (project-BN out, or the addV result for skip blocks)
  ec : String        -- expand conv out (= expand-BN input)         [noExp: = block input]
  en : String        -- expand BN out   (= expand-swish pre-act)    [noExp: unused]
  er : String        -- expand swish out (= depthwise input)        [noExp: = block input]
  dc : String        -- depthwise conv out (= depthwise-BN input)
  dn : String        -- depthwise BN out (= depthwise-swish pre-act)
  dr : String        -- depthwise swish out (= SE input)
  se : String        -- SE out (= project input)
  s  : String        -- SE squeeze (GAP out)
  e1 : String        -- SE reduce dense out (= SE swish pre-act)
  z  : String        -- SE reduce swish out (= SE excite dense input)
  e2 : String        -- SE excite dense out (= SE sigmoid pre-act)
  pc : String        -- project conv out (= project-BN input)

structure EBack where
  code : String
  dx : String
  names : List String

-- ════════════════════════════════════════════════════════════════
-- § Squeeze-excite forward (un-fused, for activation-saving) + backward (param grads)
-- ════════════════════════════════════════════════════════════════

/-- **Un-fused SE forward** on `c` channels at `hh×ww`, reduce dim `r`. Emits the squeeze (GAP),
    reduce dense (`W₁`), swish, excite dense (`W₂`) to expose `s/e1/z/e2`, AND the fused `seBlock`
    for the actual SE output `seOut`. Returns `(code, s, e1, z, e2, seOut)`. -/
private def seFwd (B c hh r : Nat) (p drName : String) :
    StateM Nat (String × String × String × String × String × String) := do
  let ww := hh
  let zChw : Vec (1 * (c * hh * ww)) := fun _ => 0
  let zCc  : Vec (1 * c) := fun _ => 0
  let zRr  : Vec (1 * r) := fun _ => 0
  let zW1  : Mat c r := fun _ _ => 0
  let zb1  : Vec r := fun _ => 0
  let zW2  : Mat r c := fun _ _ => 0
  let zb2  : Vec c := fun _ => 0
  let (cS,  nS)  ← pretty B (.batchOp (N := 1) (.gap (c := c) (h := hh) (w := ww)) (.operand drName zChw))
  let (cE1, nE1) ← pretty B (.batchOp (N := 1) (.dense s!"%{p}zW1" s!"%{p}zb1" zW1 zb1) (.operand nS zCc))
  let (cZ,  nZ)  ← pretty B (.swishF (.operand nE1 zRr))
  let (cE2, nE2) ← pretty B (.batchOp (N := 1) (.dense s!"%{p}zW2" s!"%{p}zb2" zW2 zb2) (.operand nZ zRr))
  let (cSe, nSe) ← pretty B (.batchOp (N := 1)
      (.seBlock (h := hh) (w := ww) s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2" zW1 zb1 zW2 zb2)
      (.operand drName zChw))
  pure (cS ++ cE1 ++ cZ ++ cE2 ++ cSe, nS, nE1, nZ, nE2, nSe)

/-- **SE backward + 4 param SGD ops.** `dx` (SE input cot) via the fused `seBackBatched`; the SE dense
    param grads via `seReduceB → sigmoidBack(e2) → {W₂} → denseRowBack(W₂) → swishBack(e1) → {W₁}`.
    Returns `(code, dx, [zW1, zb1, zW2, zb2 updated names])`. -/
private def seBack (B c hh r : Nat) (lrStr p drName sName e1Name zName e2Name seCot : String) :
    StateM Nat (String × String × List String) := do
  let ww := hh
  let zChw : Vec (1 * (c * hh * ww)) := fun _ => 0
  let zCc  : Vec (1 * c) := fun _ => 0
  let zRr  : Vec (1 * r) := fun _ => 0
  let zW1  : Mat c r := fun _ _ => 0
  let zb1  : Vec r := fun _ => 0
  let zW2  : Mat r c := fun _ _ => 0
  let zb2  : Vec c := fun _ => 0
  let (cDx, nDx) ← pretty B (.seBackBatched (N := 1) (c := c) (h := hh) (w := ww)
      s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2" drName zW1 zb1 zW2 zb2 zChw (.operand seCot zChw))
  let (cDg, nDg) ← pretty B (.seReduceB (N := 1) (c := c) (h := hh) (w := ww) drName zChw (.operand seCot zChw))
  let (cE2c, nE2c) ← pretty B (.sigmoidBack (n := 1 * c) e2Name zCc (.operand nDg zCc))
  let (cW2, nW2) ← pretty B (.denseWeightSgdB (N := 1) (a := r) (c := c) zName s!"%{p}zW2" lrStr zRr zW2 0 (.operand nE2c zCc))
  let (cb2, nb2) ← pretty B (.denseBiasSgdB (N := 1) (c := c) s!"%{p}zb2" lrStr zb2 0 (.operand nE2c zCc))
  let (cDz, nDz) ← pretty B (.denseRowBack (N := 1) (a := r) (c := c) s!"%{p}zW2" zW2 (.operand nE2c zCc))
  let (cE1c, nE1c) ← pretty B (.swishBack (n := 1 * r) e1Name zRr (.operand nDz zRr))
  let (cW1, nW1) ← pretty B (.denseWeightSgdB (N := 1) (a := c) (c := r) sName s!"%{p}zW1" lrStr zCc zW1 0 (.operand nE1c zRr))
  let (cb1, nb1) ← pretty B (.denseBiasSgdB (N := 1) (c := r) s!"%{p}zb1" lrStr zb1 0 (.operand nE1c zRr))
  pure (cDx ++ cDg ++ cE2c ++ cW2 ++ cb2 ++ cDz ++ cE1c ++ cW1 ++ cb1, nDx, [nW1, nb1, nW2, nb2])

-- ════════════════════════════════════════════════════════════════
-- § MBConv forward emitters (stride-1 expand / no-skip expand / strided expand / no-expand b1)
-- ════════════════════════════════════════════════════════════════

/-- Stride-1 expand MBConv forward body (shared by residual + no-skip): expand 1×1 conv-bn-swish →
    depthwise(kd) conv-bn-swish → SE → project 1×1 conv-bn. Returns the EFwd WITHOUT the final
    residual (caller adds the `addV` for residual blocks). -/
private def eFwdBody (B ic mid oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd := do
  let ww := hh
  let zIn  : Vec (1 * (ic * hh * ww)) := fun _ => 0
  let zMid : Vec (1 * (mid * hh * ww)) := fun _ => 0
  let zOut : Vec (1 * (oc * hh * ww)) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := 1) (.conv (h := hh) (w := ww) s!"%{p}eW" s!"%{p}eb" zKe zVm) (.operand xName zIn))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}eg" s!"%{p}ebt" epsStr 0 zVm zVm (.operand nEc zMid))
  let (cEr, nEr) ← pretty B (.swishF (.operand nEn zMid))
  let (cDc, nDc) ← pretty B (.batchOp (N := 1) (.depthwise (h := hh) (w := ww) s!"%{p}dW" s!"%{p}db" zDk zVm) (.operand nEr zMid))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" s!"%{p}dbt" epsStr 0 zVm zVm (.operand nDc zMid))
  let (cDr, nDr) ← pretty B (.swishF (.operand nDn zMid))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B mid hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := 1) (.conv (h := hh) (w := ww) s!"%{p}pW" s!"%{p}pb" zKp zVo) (.operand nSe zMid))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" s!"%{p}pbt" epsStr 0 zVo zVo (.operand nPc zOut))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc }

/-- **Residual stride-1 MBConv forward** (ic = oc): body + `addV` skip. -/
private def eFwd (B ic mid oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd := do
  let f ← eFwdBody B ic mid oc hh kd r epsStr p xName
  let zOut : Vec (1 * (oc * hh * hh)) := fun _ => 0
  let (cA, nA) ← pretty B (.addV (.operand f.o zOut) (.operand xName zOut))
  pure { f with code := f.code ++ cA, o := nA }

/-- **No-skip stride-1 MBConv forward** (ic ≠ oc, b9/b16): body, output = project-BN out. -/
private def eFwdNoSkip (B ic mid oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd :=
  eFwdBody B ic mid oc hh kd r epsStr p xName

/-- **Strided MBConv forward** (b2/b4/b6/b12): expand at the input `2hh×2ww`, depthwise downsamples
    `2hh×2ww → hh×ww`, project 1×1 at `hh×ww`. NO skip. -/
private def eFwdStrided (B ic mid oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd := do
  let ww := hh
  let zIn  : Vec (1 * (ic * (2*hh) * (2*ww))) := fun _ => 0
  let zMidH : Vec (1 * (mid * (2*hh) * (2*ww))) := fun _ => 0
  let zMid : Vec (1 * (mid * hh * ww)) := fun _ => 0
  let zOut : Vec (1 * (oc * hh * ww)) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := 1) (.conv (h := 2*hh) (w := 2*ww) s!"%{p}eW" s!"%{p}eb" zKe zVm) (.operand xName zIn))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := 1) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eg" s!"%{p}ebt" epsStr 0 zVm zVm (.operand nEc zMidH))
  let (cEr, nEr) ← pretty B (.swishF (.operand nEn zMidH))
  let (cDc, nDc) ← pretty B (.batchOp (N := 1) (.depthwiseStrided (h := hh) (w := ww) s!"%{p}dW" s!"%{p}db" zDk zVm) (.operand nEr zMidH))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" s!"%{p}dbt" epsStr 0 zVm zVm (.operand nDc zMid))
  let (cDr, nDr) ← pretty B (.swishF (.operand nDn zMid))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B mid hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := 1) (.conv (h := hh) (w := ww) s!"%{p}pW" s!"%{p}pb" zKp zVo) (.operand nSe zMid))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" s!"%{p}pbt" epsStr 0 zVo zVo (.operand nPc zOut))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc }

/-- **No-expand MBConv forward** (b1, t=1): depthwise(kd, on `ic` channels)-bn-swish → SE → project
    1×1 (ic→oc)-bn. NO expand, NO skip. `ec/en` unused; `er` = block input (= depthwise input). -/
private def eFwdNoExp (B ic oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd := do
  let ww := hh
  let zIn  : Vec (1 * (ic * hh * ww)) := fun _ => 0
  let zOut : Vec (1 * (oc * hh * ww)) := fun _ => 0
  let zKp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel ic kd kd := fun _ _ _ => 0
  let zVi  : Vec ic := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cDc, nDc) ← pretty B (.batchOp (N := 1) (.depthwise (h := hh) (w := ww) s!"%{p}dW" s!"%{p}db" zDk zVi) (.operand xName zIn))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := 1) (oc := ic) (h := hh) (w := ww) s!"%{p}dg" s!"%{p}dbt" epsStr 0 zVi zVi (.operand nDc zIn))
  let (cDr, nDr) ← pretty B (.swishF (.operand nDn zIn))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B ic hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := 1) (.conv (h := hh) (w := ww) s!"%{p}pW" s!"%{p}pb" zKp zVo) (.operand nSe zIn))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" s!"%{p}pbt" epsStr 0 zVo zVo (.operand nPc zOut))
  pure { code := cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := xName, en := xName, er := xName, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc }

-- ════════════════════════════════════════════════════════════════
-- § MBConv backward emitters (project → SE → depthwise → expand; param-SGD in func-arg order)
-- ════════════════════════════════════════════════════════════════

/-- Stride-1 expand MBConv backward body (shared by residual + no-skip): returns the EBack with `dx`
    = the expand-conv-back cotangent (caller adds the residual `+ dyOut` for residual blocks). -/
private def eBackBody (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack := do
  let ww := hh
  let zInF  : Vec (1 * (ic * hh * ww)) := fun _ => 0
  let zMidF : Vec (1 * (mid * hh * ww)) := fun _ => 0
  let zMidB : Vec (1 * (mid * (hh * ww))) := fun _ => 0
  let zOutF : Vec (1 * (oc * hh * ww)) := fun _ => 0
  let zOutB : Vec (1 * (oc * (hh * ww))) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  -- project: BN back (cot at project conv out) → 1×1 conv back (cot at SE out)
  let (cPbn, nPbn) ← pretty B (.bnBatchBack (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" f.pc epsStr 0 zVo zOutB (.operand dyName zOutB))
  let (cPdr, nPdr) ← pretty B (.convBackBatched (N := 1) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}pW" zKp zVo (.operand nPbn zOutF))
  let (cgp, ngp) ← pretty B (.bnGammaSgdB (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" f.pc epsStr lrStr 0 zVo zOutB 0 (.operand dyName zOutB))
  let (ctp, ntp) ← pretty B (.bnBetaSgdB (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pbt" lrStr zVo 0 (.operand dyName zOutB))
  let (cWp, nWp) ← pretty B (.convWeightSgdB (N := 1) (ic := mid) (oc := oc) (h := hh) (w := ww) f.se s!"%{p}pW" lrStr zVo zMidF zKp 0 (.operand nPbn zOutF))
  let (cbp, nbp) ← pretty B (.bnBetaSgdB (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pb" lrStr zVo 0 (.operand nPbn zOutB))
  -- SE back (dx at depthwise-swish out) + 4 SE param grads
  let (cSe, nDxSe, seNames) ← seBack B mid hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise: swish mask (cot at dw-BN out) → BN back (cot at dw conv out) → conv back (cot at expand-swish out)
  let (cDsw, nDsw) ← pretty B (.swishBack (n := 1 * (mid * hh * ww)) f.dn zMidF (.operand nDxSe zMidF))
  let (cDbn, nDbn) ← pretty B (.bnBatchBack (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr 0 zVm zMidB (.operand nDsw zMidB))
  let (cDer, nDer) ← pretty B (.depthwiseBackBatched (N := 1) (c := mid) (h := hh) (w := ww) s!"%{p}dW" zDk zVm (.operand nDbn zMidF))
  let (cgd, ngd) ← pretty B (.bnGammaSgdB (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr lrStr 0 zVm zMidB 0 (.operand nDsw zMidB))
  let (ctd, ntd) ← pretty B (.bnBetaSgdB (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}dbt" lrStr zVm 0 (.operand nDsw zMidB))
  let (cWd, nWd) ← pretty B (.depthwiseWeightSgdB (N := 1) (c := mid) (h := hh) (w := ww) f.er s!"%{p}dW" lrStr zVm zMidF zDk 0 (.operand nDbn zMidF))
  let (cbd, nbd) ← pretty B (.bnBetaSgdB (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}db" lrStr zVm 0 (.operand nDbn zMidB))
  -- expand: swish mask (cot at expand-BN out) → BN back → 1×1 conv back (cot at block input)
  let (cEsw, nEsw) ← pretty B (.swishBack (n := 1 * (mid * hh * ww)) f.en zMidF (.operand nDer zMidF))
  let (cEbn, nEbn) ← pretty B (.bnBatchBack (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}eg" f.ec epsStr 0 zVm zMidB (.operand nEsw zMidB))
  let (cExb, nExb) ← pretty B (.convBackBatched (N := 1) (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%{p}eW" zKe zVm (.operand nEbn zMidF))
  let (cge, nge) ← pretty B (.bnGammaSgdB (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}eg" f.ec epsStr lrStr 0 zVm zMidB 0 (.operand nEsw zMidB))
  let (cte, nte) ← pretty B (.bnBetaSgdB (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}ebt" lrStr zVm 0 (.operand nEsw zMidB))
  let (cWe, nWe) ← pretty B (.convWeightSgdB (N := 1) (ic := ic) (oc := mid) (h := hh) (w := ww) xName s!"%{p}eW" lrStr zVm zInF zKe 0 (.operand nEbn zMidF))
  let (cbe, nbe) ← pretty B (.bnBetaSgdB (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}eb" lrStr zVm 0 (.operand nEbn zMidB))
  let names := [nWe, nbe, nge, nte, nWd, nbd, ngd, ntd] ++ seNames ++ [nWp, nbp, ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDer ++ cgd ++ ctd ++ cWd ++ cbd ++
                 cEsw ++ cEbn ++ cExb ++ cge ++ cte ++ cWe ++ cbe,
         dx := nExb, names := names }

/-- **Residual stride-1 MBConv backward** (ic = oc): body + skip fan-in `+ dyOut`. -/
private def eBack (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack := do
  let b ← eBackBody B ic mid oc hh kd r epsStr lrStr p xName f dyName
  let zIn : Vec (1 * (ic * hh * hh)) := fun _ => 0
  let (cDx, nDx) ← pretty B (.addV (.operand b.dx zIn) (.operand dyName zIn))
  pure { b with code := b.code ++ cDx, dx := nDx }

/-- **No-skip stride-1 MBConv backward** (ic ≠ oc, b9/b16): body, dx = expand-conv-back directly. -/
private def eBackNoSkip (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack :=
  eBackBody B ic mid oc hh kd r epsStr lrStr p xName f dyName

/-- **Strided MBConv backward** (b2/b4/b6/b12): depthwise-back upsamples `hh×ww → 2hh×2ww`; the
    expand stage backward runs at `2hh×2ww`. NO skip. -/
private def eBackStrided (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack := do
  let ww := hh
  let zInF  : Vec (1 * (ic * (2*hh) * (2*ww))) := fun _ => 0
  let zMidHF : Vec (1 * (mid * (2*hh) * (2*ww))) := fun _ => 0
  let zMidHB : Vec (1 * (mid * ((2*hh) * (2*ww)))) := fun _ => 0
  let zMidF : Vec (1 * (mid * hh * ww)) := fun _ => 0
  let zMidB : Vec (1 * (mid * (hh * ww))) := fun _ => 0
  let zOutF : Vec (1 * (oc * hh * ww)) := fun _ => 0
  let zOutB : Vec (1 * (oc * (hh * ww))) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  -- project (at hh)
  let (cPbn, nPbn) ← pretty B (.bnBatchBack (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" f.pc epsStr 0 zVo zOutB (.operand dyName zOutB))
  let (cPdr, nPdr) ← pretty B (.convBackBatched (N := 1) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}pW" zKp zVo (.operand nPbn zOutF))
  let (cgp, ngp) ← pretty B (.bnGammaSgdB (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" f.pc epsStr lrStr 0 zVo zOutB 0 (.operand dyName zOutB))
  let (ctp, ntp) ← pretty B (.bnBetaSgdB (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pbt" lrStr zVo 0 (.operand dyName zOutB))
  let (cWp, nWp) ← pretty B (.convWeightSgdB (N := 1) (ic := mid) (oc := oc) (h := hh) (w := ww) f.se s!"%{p}pW" lrStr zVo zMidF zKp 0 (.operand nPbn zOutF))
  let (cbp, nbp) ← pretty B (.bnBetaSgdB (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pb" lrStr zVo 0 (.operand nPbn zOutB))
  -- SE back (at hh)
  let (cSe, nDxSe, seNames) ← seBack B mid hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise (swish + BN at hh, strided conv-back upsamples to 2hh)
  let (cDsw, nDsw) ← pretty B (.swishBack (n := 1 * (mid * hh * ww)) f.dn zMidF (.operand nDxSe zMidF))
  let (cDbn, nDbn) ← pretty B (.bnBatchBack (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr 0 zVm zMidB (.operand nDsw zMidB))
  let (cDer, nDer) ← pretty B (.depthwiseStridedBackBatched (N := 1) (c := mid) (h := hh) (w := ww) s!"%{p}dW" zDk zVm (.operand nDbn zMidF))
  let (cgd, ngd) ← pretty B (.bnGammaSgdB (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr lrStr 0 zVm zMidB 0 (.operand nDsw zMidB))
  let (ctd, ntd) ← pretty B (.bnBetaSgdB (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}dbt" lrStr zVm 0 (.operand nDsw zMidB))
  let (cWd, nWd) ← pretty B (.depthwiseStridedWeightSgdB (N := 1) (c := mid) (h := hh) (w := ww) f.er s!"%{p}dW" lrStr zVm zMidHF zDk 0 (.operand nDbn zMidF))
  let (cbd, nbd) ← pretty B (.bnBetaSgdB (N := 1) (oc := mid) (h := hh) (w := ww) s!"%{p}db" lrStr zVm 0 (.operand nDbn zMidB))
  -- expand (at 2hh)
  let (cEsw, nEsw) ← pretty B (.swishBack (n := 1 * (mid * (2*hh) * (2*ww))) f.en zMidHF (.operand nDer zMidHF))
  let (cEbn, nEbn) ← pretty B (.bnBatchBack (N := 1) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eg" f.ec epsStr 0 zVm zMidHB (.operand nEsw zMidHB))
  let (cExb, nExb) ← pretty B (.convBackBatched (N := 1) (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eW" zKe zVm (.operand nEbn zMidHF))
  let (cge, nge) ← pretty B (.bnGammaSgdB (N := 1) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eg" f.ec epsStr lrStr 0 zVm zMidHB 0 (.operand nEsw zMidHB))
  let (cte, nte) ← pretty B (.bnBetaSgdB (N := 1) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}ebt" lrStr zVm 0 (.operand nEsw zMidHB))
  let (cWe, nWe) ← pretty B (.convWeightSgdB (N := 1) (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) xName s!"%{p}eW" lrStr zVm zInF zKe 0 (.operand nEbn zMidHF))
  let (cbe, nbe) ← pretty B (.bnBetaSgdB (N := 1) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eb" lrStr zVm 0 (.operand nEbn zMidHB))
  let names := [nWe, nbe, nge, nte, nWd, nbd, ngd, ntd] ++ seNames ++ [nWp, nbp, ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDer ++ cgd ++ ctd ++ cWd ++ cbd ++
                 cEsw ++ cEbn ++ cExb ++ cge ++ cte ++ cWe ++ cbe,
         dx := nExb, names := names }

/-- **No-expand MBConv backward** (b1): project back → SE back → depthwise back → dx (block input).
    8 params (Wd bd gd btd zW1 zb1 zW2 zb2 ... wait, 4 dw + 4 SE + 4 proj = 12). -/
private def eBackNoExp (B ic oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack := do
  let ww := hh
  let zInF  : Vec (1 * (ic * hh * ww)) := fun _ => 0
  let zInB  : Vec (1 * (ic * (hh * ww))) := fun _ => 0
  let zOutF : Vec (1 * (oc * hh * ww)) := fun _ => 0
  let zOutB : Vec (1 * (oc * (hh * ww))) := fun _ => 0
  let zKp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel ic kd kd := fun _ _ _ => 0
  let zVi  : Vec ic := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  -- project
  let (cPbn, nPbn) ← pretty B (.bnBatchBack (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" f.pc epsStr 0 zVo zOutB (.operand dyName zOutB))
  let (cPdr, nPdr) ← pretty B (.convBackBatched (N := 1) (ic := ic) (oc := oc) (h := hh) (w := ww) s!"%{p}pW" zKp zVo (.operand nPbn zOutF))
  let (cgp, ngp) ← pretty B (.bnGammaSgdB (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" f.pc epsStr lrStr 0 zVo zOutB 0 (.operand dyName zOutB))
  let (ctp, ntp) ← pretty B (.bnBetaSgdB (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pbt" lrStr zVo 0 (.operand dyName zOutB))
  let (cWp, nWp) ← pretty B (.convWeightSgdB (N := 1) (ic := ic) (oc := oc) (h := hh) (w := ww) f.se s!"%{p}pW" lrStr zVo zInF zKp 0 (.operand nPbn zOutF))
  let (cbp, nbp) ← pretty B (.bnBetaSgdB (N := 1) (oc := oc) (h := hh) (w := ww) s!"%{p}pb" lrStr zVo 0 (.operand nPbn zOutB))
  -- SE back (on ic channels)
  let (cSe, nDxSe, seNames) ← seBack B ic hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise (on ic channels)
  let (cDsw, nDsw) ← pretty B (.swishBack (n := 1 * (ic * hh * ww)) f.dn zInF (.operand nDxSe zInF))
  let (cDbn, nDbn) ← pretty B (.bnBatchBack (N := 1) (oc := ic) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr 0 zVi zInB (.operand nDsw zInB))
  let (cDxb, nDxb) ← pretty B (.depthwiseBackBatched (N := 1) (c := ic) (h := hh) (w := ww) s!"%{p}dW" zDk zVi (.operand nDbn zInF))
  let (cgd, ngd) ← pretty B (.bnGammaSgdB (N := 1) (oc := ic) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr lrStr 0 zVi zInB 0 (.operand nDsw zInB))
  let (ctd, ntd) ← pretty B (.bnBetaSgdB (N := 1) (oc := ic) (h := hh) (w := ww) s!"%{p}dbt" lrStr zVi 0 (.operand nDsw zInB))
  let (cWd, nWd) ← pretty B (.depthwiseWeightSgdB (N := 1) (c := ic) (h := hh) (w := ww) xName s!"%{p}dW" lrStr zVi zInF zDk 0 (.operand nDbn zInF))
  let (cbd, nbd) ← pretty B (.bnBetaSgdB (N := 1) (oc := ic) (h := hh) (w := ww) s!"%{p}db" lrStr zVi 0 (.operand nDbn zInB))
  let names := [nWd, nbd, ngd, ntd] ++ seNames ++ [nWp, nbp, ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDxb ++ cgd ++ ctd ++ cWd ++ cbd,
         dx := nDxb, names := names }

-- ════════════════════════════════════════════════════════════════
-- § Param signature lists (func-arg order — names + types)
-- ════════════════════════════════════════════════════════════════

private def eSig (p : String) (ic mid oc r kd : Nat) : List (String × String) :=
  [(s!"%{p}eW", ty [mid,ic,1,1]), (s!"%{p}eb", ty [mid]), (s!"%{p}eg", ty [mid]), (s!"%{p}ebt", ty [mid]),
   (s!"%{p}dW", ty [mid,1,kd,kd]), (s!"%{p}db", ty [mid]), (s!"%{p}dg", ty [mid]), (s!"%{p}dbt", ty [mid]),
   (s!"%{p}zW1", ty [mid,r]), (s!"%{p}zb1", ty [r]), (s!"%{p}zW2", ty [r,mid]), (s!"%{p}zb2", ty [mid]),
   (s!"%{p}pW", ty [oc,mid,1,1]), (s!"%{p}pb", ty [oc]), (s!"%{p}pg", ty [oc]), (s!"%{p}pbt", ty [oc])]

private def eSigNoExp (p : String) (ic oc r kd : Nat) : List (String × String) :=
  [(s!"%{p}dW", ty [ic,1,kd,kd]), (s!"%{p}db", ty [ic]), (s!"%{p}dg", ty [ic]), (s!"%{p}dbt", ty [ic]),
   (s!"%{p}zW1", ty [ic,r]), (s!"%{p}zb1", ty [r]), (s!"%{p}zW2", ty [r,ic]), (s!"%{p}zb2", ty [ic]),
   (s!"%{p}pW", ty [oc,ic,1,1]), (s!"%{p}pb", ty [oc]), (s!"%{p}pg", ty [oc]), (s!"%{p}pbt", ty [oc])]

/-- **Full 262-param EfficientNet-B0 signature**, func-arg order: stem(4) + b1 no-exp(12) +
    b2..b16 expand(15×16) + head(4) + dense(2) = 4+12+240+4+2 = 262 tensors. -/
private def enetSig (nClasses : Nat) : List (String × String) :=
  [("%sW", ty [32,3,3,3]), ("%sb", ty [32]), ("%sg", ty [32]), ("%sbt", ty [32])] ++
  eSigNoExp "b1" 32 16 8 3 ++
  eSig "b2"  16  96  24  4 3 ++ eSig "b3"  24 144  24  6 3 ++
  eSig "b4"  24 144  40  6 5 ++ eSig "b5"  40 240  40 10 5 ++
  eSig "b6"  40 240  80 10 3 ++ eSig "b7"  80 480  80 20 3 ++ eSig "b8"  80 480  80 20 3 ++
  eSig "b9"  80 480 112 20 5 ++ eSig "b10" 112 672 112 28 5 ++ eSig "b11" 112 672 112 28 5 ++
  eSig "b12" 112 672 192 28 5 ++ eSig "b13" 192 1152 192 48 5 ++ eSig "b14" 192 1152 192 48 5 ++
  eSig "b15" 192 1152 192 48 5 ++ eSig "b16" 192 1152 320 48 3 ++
  [("%hW", ty [1280,320,1,1]), ("%hb", ty [1280]), ("%hg", ty [1280]), ("%hbt", ty [1280])] ++
  [("%Wd", ty [1280, nClasses]), ("%bd", ty [nClasses])]

-- ════════════════════════════════════════════════════════════════
-- § The whole-net renderer
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000000 in
/-- **EfficientNet-B0 (full 16-MBConv) train step rendered ENTIRELY from the verified AST**, at the
    batched index `N·(c·h·w)`. Every emitted line is `pretty` of a verified `SHlo` node. Strided stem
    3×3/s2 (3→32, 224→112) → b1 (no-expand) → b2..b16 (4 strided downsamples 112→7, 9 residual skips,
    2 no-skip widenings) → 1×1 conv-bn-swish head (320→1280) → GAP → dense (1280→nClasses). -/
def efficientnetTrainStepFaithfulV (B nClasses : Nat) (epsStr lrStr : String)
    (funcName : String := "efficientnet_train_step") : String :=
  let go : StateM Nat String := do
    -- ═══ stem: 3×3/s2 conv (3→32, 224→112) → bn → swish ═══
    let zx   : Vec (1 * (3*224*224)) := fun _ => 0
    let zSk  : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let z32  : Vec 32 := fun _ => 0
    let z112F : Vec (1 * (32*112*112)) := fun _ => 0
    let z112B : Vec (1 * (32*(112*112))) := fun _ => 0
    let (cStc, nStc) ← pretty B (.batchOp (N := 1) (.convStrided (h := 112) (w := 112) "%sW" "%sb" zSk z32) (.operand "%x" zx))
    let (cStn, nStn) ← pretty B (.bnBatchF (N := 1) (oc := 32) (h := 112) (w := 112) "%sg" "%sbt" epsStr 0 z32 z32 (.operand nStc z112F))
    let (cStr, nStr) ← pretty B (.swishF (.operand nStn z112F))
    -- ═══ forward: 16 MBConv blocks ═══
    let f1  ← eFwdNoExp   B 32      16 112 3  8 epsStr "b1"  nStr
    let f2  ← eFwdStrided B 16  96  24  56 3  4 epsStr "b2"  f1.o
    let f3  ← eFwd        B 24 144  24  56 3  6 epsStr "b3"  f2.o
    let f4  ← eFwdStrided B 24 144  40  28 5  6 epsStr "b4"  f3.o
    let f5  ← eFwd        B 40 240  40  28 5 10 epsStr "b5"  f4.o
    let f6  ← eFwdStrided B 40 240  80  14 3 10 epsStr "b6"  f5.o
    let f7  ← eFwd        B 80 480  80  14 3 20 epsStr "b7"  f6.o
    let f8  ← eFwd        B 80 480  80  14 3 20 epsStr "b8"  f7.o
    let f9  ← eFwdNoSkip  B 80 480 112  14 5 20 epsStr "b9"  f8.o
    let f10 ← eFwd        B 112 672 112 14 5 28 epsStr "b10" f9.o
    let f11 ← eFwd        B 112 672 112 14 5 28 epsStr "b11" f10.o
    let f12 ← eFwdStrided B 112 672 192  7 5 28 epsStr "b12" f11.o
    let f13 ← eFwd        B 192 1152 192 7 5 48 epsStr "b13" f12.o
    let f14 ← eFwd        B 192 1152 192 7 5 48 epsStr "b14" f13.o
    let f15 ← eFwd        B 192 1152 192 7 5 48 epsStr "b15" f14.o
    let f16 ← eFwdNoSkip  B 192 1152 320 7 3 48 epsStr "b16" f15.o
    -- ═══ head: 1×1 conv (320→1280) → bn → swish → GAP → dense → softmax-CE cot ═══
    let z7F   : Vec (1 * (320*7*7)) := fun _ => 0
    let zHk   : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
    let z1280 : Vec 1280 := fun _ => 0
    let zH7F  : Vec (1 * (1280*7*7)) := fun _ => 0
    let zH7B  : Vec (1 * (1280*(7*7))) := fun _ => 0
    let z1280c : Vec (1 * 1280) := fun _ => 0
    let zWd   : Mat 1280 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    let zNCb  : Vec (1 * nClasses) := fun _ => 0
    let (cHc, nHc) ← pretty B (.batchOp (N := 1) (.conv (h := 7) (w := 7) "%hW" "%hb" zHk z1280) (.operand f16.o z7F))
    let (cHn, nHn) ← pretty B (.bnBatchF (N := 1) (oc := 1280) (h := 7) (w := 7) "%hg" "%hbt" epsStr 0 z1280 z1280 (.operand nHc zH7F))
    let (cHr, nHr) ← pretty B (.swishF (.operand nHn zH7F))
    let (cGap, nGap) ← pretty B (.batchOp (N := 1) (.gap (c := 1280) (h := 7) (w := 7)) (.operand nHr zH7F))
    let (cLog, nLog) ← pretty B (.batchOp (N := 1) (.dense "%Wd" "%bd" zWd zNC) (.operand nGap z1280c))
    let (cSm, nSm) ← pretty B (.softmaxRowF (m := 1) (n := nClasses) (.operand nLog zNCb))
    let (cDy, nDy) ← pretty B (.sub (.operand nSm zNCb) (.operand "%onehot" zNCb))
    -- ═══ head backward: dense back → GAP back → swish mask → bn back → 1×1 conv back ═══
    let (cDgi, nDgi) ← pretty B (.denseRowBack (N := 1) (a := 1280) (c := nClasses) "%Wd" zWd (.operand nDy zNCb))
    let (cWfc, nWfc) ← pretty B (.denseWeightSgdB (N := 1) (a := 1280) (c := nClasses) nGap "%Wd" lrStr z1280c zWd 0 (.operand nDy zNCb))
    let (cbfc, nbfc) ← pretty B (.denseBiasSgdB (N := 1) (c := nClasses) "%bd" lrStr zNC 0 (.operand nDy zNCb))
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := 1) (c := 1280) (h := 7) (w := 7) (.operand nDgi z1280c))
    let (cHsw, nHsw) ← pretty B (.swishBack (n := 1 * (1280*7*7)) nHn zH7F (.operand nDgp zH7F))
    let (cHbn, nHbn) ← pretty B (.bnBatchBack (N := 1) (oc := 1280) (h := 7) (w := 7) "%hg" nHc epsStr 0 z1280 zH7B (.operand nHsw zH7B))
    let (cHxb, nHxb) ← pretty B (.convBackBatched (N := 1) (ic := 320) (oc := 1280) (h := 7) (w := 7) "%hW" zHk z1280 (.operand nHbn zH7F))
    let (cgh, ngh) ← pretty B (.bnGammaSgdB (N := 1) (oc := 1280) (h := 7) (w := 7) "%hg" nHc epsStr lrStr 0 z1280 zH7B 0 (.operand nHsw zH7B))
    let (cth, nth) ← pretty B (.bnBetaSgdB (N := 1) (oc := 1280) (h := 7) (w := 7) "%hbt" lrStr z1280 0 (.operand nHsw zH7B))
    let (cWh, nWh) ← pretty B (.convWeightSgdB (N := 1) (ic := 320) (oc := 1280) (h := 7) (w := 7) f16.o "%hW" lrStr z1280 z7F zHk 0 (.operand nHbn zH7F))
    let (cbh, nbh) ← pretty B (.bnBetaSgdB (N := 1) (oc := 1280) (h := 7) (w := 7) "%hb" lrStr z1280 0 (.operand nHbn zH7B))
    -- ═══ backward: 16 blocks reversed (cotangent threads from nHxb) ═══
    let b16 ← eBackNoSkip  B 192 1152 320 7 3 48 epsStr lrStr "b16" f15.o f16 nHxb
    let b15 ← eBack        B 192 1152 192 7 5 48 epsStr lrStr "b15" f14.o f15 b16.dx
    let b14 ← eBack        B 192 1152 192 7 5 48 epsStr lrStr "b14" f13.o f14 b15.dx
    let b13 ← eBack        B 192 1152 192 7 5 48 epsStr lrStr "b13" f12.o f13 b14.dx
    let b12 ← eBackStrided B 112 672 192  7 5 28 epsStr lrStr "b12" f11.o f12 b13.dx
    let b11 ← eBack        B 112 672 112 14 5 28 epsStr lrStr "b11" f10.o f11 b12.dx
    let b10 ← eBack        B 112 672 112 14 5 28 epsStr lrStr "b10" f9.o  f10 b11.dx
    let b9  ← eBackNoSkip  B 80 480 112  14 5 20 epsStr lrStr "b9"  f8.o  f9  b10.dx
    let b8  ← eBack        B 80 480  80  14 3 20 epsStr lrStr "b8"  f7.o  f8  b9.dx
    let b7  ← eBack        B 80 480  80  14 3 20 epsStr lrStr "b7"  f6.o  f7  b8.dx
    let b6  ← eBackStrided B 40 240  80  14 3 10 epsStr lrStr "b6"  f5.o  f6  b7.dx
    let b5  ← eBack        B 40 240  40  28 5 10 epsStr lrStr "b5"  f4.o  f5  b6.dx
    let b4  ← eBackStrided B 24 144  40  28 5  6 epsStr lrStr "b4"  f3.o  f4  b5.dx
    let b3  ← eBack        B 24 144  24  56 3  6 epsStr lrStr "b3"  f2.o  f3  b4.dx
    let b2  ← eBackStrided B 16  96  24  56 3  4 epsStr lrStr "b2"  f1.o  f2  b3.dx
    let b1  ← eBackNoExp   B 32      16 112 3  8 epsStr lrStr "b1"  nStr  f1  b2.dx
    -- ═══ stem backward: swish mask → bn back, then stem param SGD (NO conv-back past %x) ═══
    let (cDsr, nDsr) ← pretty B (.swishBack (n := 1 * (32*112*112)) nStn z112F (.operand b1.dx z112F))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := 1) (oc := 32) (h := 112) (w := 112) "%sg" nStc epsStr 0 z32 z112B (.operand nDsr z112B))
    let (csW, nsW) ← pretty B (.convStridedWeightSgdB (N := 1) (ic := 3) (oc := 32) (h := 112) (w := 112) "%x" "%sW" lrStr z32 zx zSk 0 (.operand nDsn z112F))
    let (csb, nsb) ← pretty B (.bnBetaSgdB (N := 1) (oc := 32) (h := 112) (w := 112) "%sb" lrStr z32 0 (.operand nDsn z112B))
    let (csg, nsg) ← pretty B (.bnGammaSgdB (N := 1) (oc := 32) (h := 112) (w := 112) "%sg" nStc epsStr lrStr 0 z32 z112B 0 (.operand nDsr z112B))
    let (cst, nst) ← pretty B (.bnBetaSgdB (N := 1) (oc := 32) (h := 112) (w := 112) "%sbt" lrStr z32 0 (.operand nDsr z112B))
    -- ═══ assemble (params in func-arg order: stem, blocks fwd-order, head, dense) ═══
    let fwdCode := cStc ++ cStn ++ cStr ++
      f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++ f8.code ++
      f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++ f15.code ++ f16.code ++
      cHc ++ cHn ++ cHr ++ cGap ++ cLog ++ cSm ++ cDy
    let bwdCode := cDgi ++ cWfc ++ cbfc ++ cDgp ++ cHsw ++ cHbn ++ cHxb ++ cgh ++ cth ++ cWh ++ cbh ++
      b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++ b10.code ++ b9.code ++
      b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDsr ++ cDsn ++ csW ++ csb ++ csg ++ cst
    let outNames : List String :=
      [nsW, nsb, nsg, nst] ++
      b1.names ++ b2.names ++ b3.names ++ b4.names ++ b5.names ++ b6.names ++ b7.names ++ b8.names ++
      b9.names ++ b10.names ++ b11.names ++ b12.names ++ b13.names ++ b14.names ++ b15.names ++
      b16.names ++ [nWh, nbh, ngh, nth] ++ [nWfc, nbfc]
    let outTypes : List String := (enetSig nClasses).map (·.2)
    pure <|
      "    // ── EfficientNet-B0 (16-MBConv) train step: every line is pretty(verified AST node) ──\n" ++
      fwdCode ++ bwdCode ++
      s!"    return {String.intercalate ", " outNames} : {String.intercalate ", " outTypes}\n"
  let sigList : List (String × String) := enetSig nClasses
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

-- Regenerate `verified_mlir/efficientnet_train_step.mlir` (what MainEfficientNetVerified trains on)
-- from the faithful renderer: the FULL 16-MBConv B0 net (262 params). B=32, nClasses=10, ε=1e-5.
#eval IO.FS.writeFile "verified_mlir/efficientnet_train_step.mlir"
  (Proofs.StableHLO.efficientnetTrainStepFaithfulV 32 10 "1.0e-5" "0.05" "efficientnet_train_step")
