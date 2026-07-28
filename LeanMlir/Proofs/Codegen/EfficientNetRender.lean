import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.ViTRender      -- `emitGradAllReduce`, the data-parallel collective (a carve-out)

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
dense W₁ → z = batchOp swish → e2 = batchOp dense W₂` (only to expose `s/e1/z/e2`); the SE param grads chain
`seReduceB → sigmoidBack(e2) → denseWeightSgdB/denseBiasSgdB (W₂) → denseRowBack(W₂) → swishBack(e1)
→ denseWeightSgdB/denseBiasSgdB (W₁)`, and `dx` reuses the fused `seBackBatched`. Activations
are **swish** (smooth, no relu6 kink), the head GAP-back uses the batched `gapBackBatched`.

Render is value-independent (`skel` erases values), so placeholder zeros + `lr := 0`/`ε := 0` are
passed; the emitted `lrStr`/`epsStr` literals carry the real values. -/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-! ## The batched index: why every `N` below is `B`, not 1

This renderer used to build the graph at the **batch-unit** index `N = 1` and let `pretty B` supply
the real batch. That was a disclosed convention, and it was sound for the ops where the batch is a
*parallel* index — `batchOp`'s `den` is `batchMap N (denOp op)`, which at `N = 1` is the per-example
op, exactly what the emit applies across the batch. It was **not** sound for the ops where the batch
is a *reduction* axis, and there are ten of those in this graph: `bnBatchF` and `bnBatchBack` reduce
μ/var over `[0,2,3]`, and the whole `*SgdB` param family (`bnGammaSgdB`, `bnBetaSgdB`,
`dense{Weight,Bias}SgdB`, `conv{,Strided}WeightSgdB`, `depthwise{,Strided}WeightSgdB`) sums the
per-example gradient over `Fin N`. At `N = 1` each of those `den`s describes a ONE-EXAMPLE function
while the emitted text reduces over all `B` — the node and its render were different functions, the
op-level form of the two-writers bug.

The fix is to put the whole graph at `N := B`, where those `den`s are honest. What blocked that was
not the batch-coupled ops (their emitters discard `N` and use `B`, so they render identically at any
`N`) but the *pointwise* ones: `swishF`/`swishBack`/`sigmoidBack`/`addV`/`sub` carry only the SHlo
index and emit `tensor<B×n>` from it, so at the batched index `N·s` they emit `tensor<B×(N·s)>` —
which does not even typecheck against its own operand. Hence the batched forms, which all separate
the batch `N` from the per-example emit width `n`:

* `BatchableOp.swish`/`softmaxRow`/`denseRowBack` — descriptors, `den = batchMap N (denOp op)`.
  Sound because what they lift is a FIXED function: swish and row-softmax carry no data, and
  `denseRowBack` carries `W`, a parameter shared by every example.
* `swishBackB`/`sigmoidBackB` — their own constructors, NOT descriptors, because their VJP
  `fun x dy i => dy i * deriv (x i)` depends on the saved pre-activation, which varies per example.
  `batchMap N` of that would denote "every example shares one example's activation". They carry the
  whole-batch `x : Vec (N*n)` instead — which is what the emitted `xName` actually holds.
* `addVB`/`subB` — pointwise binary, via the binary `batched2` tag.

`softmaxRow`'s `m` and `denseRowBack`'s `rows` are NOT the batch — they are rows per example (ViT
uses `m := 197` tokens; a classifier head has one logit row), and they stay 1 here.

The artifact is byte-identical across this change, which is what proves the EMIT side
behaviour-preserving. It cannot witness the `den` side — the render is value-independent, so a
descriptor holding the wrong saved activation would render exactly the same bytes. That half is
carried by the `rfl` faithfulness theorems in `StableHLO.lean`. -/

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
  /-- The block's BN layers in forward order: `(BN-input SSA, channels, spatial side)`. The AdamW
      render turns each into a `bnBatchMeanB`/`bnBatchVarB` pair — the batch statistics a batch-BN
      train step has to hand back so the host can EMA them into the eval forward's frozen stats.
      The SGD render has no such outputs and ignores this field.

      Order is expand-BN → depthwise-BN → project-BN, with the expand entry ABSENT for the
      no-expand block (b1, t = 1), because that is the layout the driver's `bnChannels` metadata
      and `@efficientnet_fwd_eval` read positionally. Getting it wrong is silent: the arities still
      match and the wrong layer's statistics simply flow into the wrong eval slot. -/
  bns : List (String × Nat × Nat)

structure EBack where
  code : String
  dx : String
  names : List String

-- ════════════════════════════════════════════════════════════════
-- § Parameter tails — the un-fused gradient, or the fused SGD update
-- ════════════════════════════════════════════════════════════════

/-! Every leaf of the backward ends at a parameter, and there are exactly two things it can emit:
the **un-fused gradient** (`adam := true`) or the **fused SGD update** `θ − lr·g` (`false`). The
`*SgdB_eq_grad` theorems say `den (xSgdB …) = θ − lr · den (xGradB …)` by `rfl`, and
`tests/TestBatchedEmitTie.lean` checks the emit side of the same statement: each `*GradB` render is
a byte-PREFIX of its `*SgdB` peer's, the tail being exactly the const-lr / multiply / subtract.

These six helpers are what let ONE backward traversal serve both renders. The alternative — a second
copy of the 16-MBConv backward for AdamW — is the double-writer disease one level down, in code
rather than in artifacts, and it is how `efficientnet_train_step` ended up with two emitters
computing different functions in the first place (§2a-quinquies).

`lrStr` is threaded but unused in `adam` mode: the AdamW render's learning rate is the runtime
`%lr` argument, not a baked literal. The placeholder values are irrelevant either way — `skel`
erases them, so the render is value-independent. -/

/-- BN γ. -/
private def bnG (adam : Bool) (B oc hh ww : Nat) (gName vName epsStr lrStr dy : String) :
    StateM Nat (String × String) := do
  let zc : Vec oc := fun _ => 0
  let zb : Vec (B * (oc * (hh * ww))) := fun _ => 0
  if adam then
    pretty B (.bnGammaGradB (N := B) (oc := oc) (h := hh) (w := ww) vName epsStr 0 zb (.operand dy zb))
  else
    pretty B (.bnGammaSgdB (N := B) (oc := oc) (h := hh) (w := ww) gName vName epsStr lrStr 0 zc zb 0
                (.operand dy zb))

/-- BN β — and, because `Σ_{batch,spatial} dy` is exactly a conv bias gradient, also every conv
    bias in this net (`%eb`, `%db`, `%pb`, `%sb`). That reuse is why EfficientNet needed no
    depthwise BIAS gradient op: its depthwise convs are followed by BN, so the bias is folded. -/
private def bnBt (adam : Bool) (B oc hh ww : Nat) (bName lrStr dy : String) : StateM Nat (String × String) := do
  let zc : Vec oc := fun _ => 0
  let zb : Vec (B * (oc * (hh * ww))) := fun _ => 0
  if adam then
    pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand dy zb))
  else
    pretty B (.bnBetaSgdB (N := B) (oc := oc) (h := hh) (w := ww) bName lrStr zc 0 (.operand dy zb))

/-- 1×1 conv weight (expand / project / head — every non-depthwise conv here but the stem). -/
private def convW1 (adam : Bool) (B ic oc hh ww : Nat) (xName wName lrStr dy : String) :
    StateM Nat (String × String) := do
  let zb : Vec oc := fun _ => 0
  let zx : Vec (B * (ic * hh * ww)) := fun _ => 0
  let zk : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zd : Vec (B * (oc * hh * ww)) := fun _ => 0
  if adam then
    pretty B (.convWeightGradB (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) xName zb zx zk
                (.operand dy zd))
  else
    pretty B (.convWeightSgdB (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) xName wName lrStr
                zb zx zk 0 (.operand dy zd))

/-- Depthwise `kd × kd` weight, stride 1. -/
private def dwW (adam : Bool) (B c hh ww kd : Nat) (xName wName lrStr dy : String) :
    StateM Nat (String × String) := do
  let zb : Vec c := fun _ => 0
  let zx : Vec (B * (c * hh * ww)) := fun _ => 0
  let zk : DepthwiseKernel c kd kd := fun _ _ _ => 0
  let zd : Vec (B * (c * hh * ww)) := fun _ => 0
  if adam then
    pretty B (.depthwiseWeightGradB (N := B) (c := c) (h := hh) (w := ww) xName zb zx zk
                (.operand dy zd))
  else
    pretty B (.depthwiseWeightSgdB (N := B) (c := c) (h := hh) (w := ww) xName wName lrStr
                zb zx zk 0 (.operand dy zd))

/-- Depthwise `kd × kd` weight, stride 2 (input at `2hh × 2ww`). -/
private def dwWS (adam : Bool) (B c hh ww kd : Nat) (xName wName lrStr dy : String) :
    StateM Nat (String × String) := do
  let zb : Vec c := fun _ => 0
  let zx : Vec (B * (c * (2*hh) * (2*ww))) := fun _ => 0
  let zk : DepthwiseKernel c kd kd := fun _ _ _ => 0
  let zd : Vec (B * (c * hh * ww)) := fun _ => 0
  if adam then
    pretty B (.depthwiseStridedWeightGradB (N := B) (c := c) (h := hh) (w := ww) xName zb zx zk
                (.operand dy zd))
  else
    pretty B (.depthwiseStridedWeightSgdB (N := B) (c := c) (h := hh) (w := ww) xName wName lrStr
                zb zx zk 0 (.operand dy zd))

/-- Dense weight (the two SE gate matrices and the classifier head). -/
private def dnW (adam : Bool) (B a c : Nat) (xName wName lrStr dy : String) : StateM Nat (String × String) := do
  let zx : Vec (B * a) := fun _ => 0
  let zW : Mat a c := fun _ _ => 0
  let zd : Vec (B * c) := fun _ => 0
  if adam then
    pretty B (.denseWeightGradB (N := B) (a := a) (c := c) xName zx (.operand dy zd))
  else
    pretty B (.denseWeightSgdB (N := B) (a := a) (c := c) xName wName lrStr zx zW 0 (.operand dy zd))

/-- Dense bias. -/
private def dnB (adam : Bool) (B c : Nat) (bName lrStr dy : String) : StateM Nat (String × String) := do
  let zb : Vec c := fun _ => 0
  let zd : Vec (B * c) := fun _ => 0
  if adam then
    pretty B (.denseBiasGradB (N := B) (c := c) (.operand dy zd))
  else
    pretty B (.denseBiasSgdB (N := B) (c := c) bName lrStr zb 0 (.operand dy zd))

-- ════════════════════════════════════════════════════════════════
-- § Squeeze-excite forward (un-fused, for activation-saving) + backward (param grads)
-- ════════════════════════════════════════════════════════════════

/-- **Un-fused SE forward** on `c` channels at `hh×ww`, reduce dim `r`. Emits the squeeze (GAP),
    reduce dense (`W₁`), swish, excite dense (`W₂`) to expose `s/e1/z/e2`, AND the fused `seBlock`
    for the actual SE output `seOut`. Returns `(code, s, e1, z, e2, seOut)`. -/
private def seFwd (B c hh r : Nat) (p drName : String) :
    StateM Nat (String × String × String × String × String × String) := do
  let ww := hh
  let zChw : Vec (B * (c * hh * ww)) := fun _ => 0
  let zCc  : Vec (B * c) := fun _ => 0
  let zRr  : Vec (B * r) := fun _ => 0
  let zW1  : Mat c r := fun _ _ => 0
  let zb1  : Vec r := fun _ => 0
  let zW2  : Mat r c := fun _ _ => 0
  let zb2  : Vec c := fun _ => 0
  let (cS,  nS)  ← pretty B (.batchOp (N := B) (.gap (c := c) (h := hh) (w := ww)) (.operand drName zChw))
  let (cE1, nE1) ← pretty B (.batchOp (N := B) (.dense s!"%{p}zW1" s!"%{p}zb1" zW1 zb1) (.operand nS zCc))
  let (cZ,  nZ)  ← pretty B (.batchOp (.swish) (.operand nE1 zRr))
  let (cE2, nE2) ← pretty B (.batchOp (N := B) (.dense s!"%{p}zW2" s!"%{p}zb2" zW2 zb2) (.operand nZ zRr))
  let (cSe, nSe) ← pretty B (.batchOp (N := B)
      (.seBlock (h := hh) (w := ww) s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2" zW1 zb1 zW2 zb2)
      (.operand drName zChw))
  pure (cS ++ cE1 ++ cZ ++ cE2 ++ cSe, nS, nE1, nZ, nE2, nSe)

/-- **SE backward + 4 param SGD ops.** `dx` (SE input cot) via the fused `seBackBatched`; the SE dense
    param grads via `seReduceB → sigmoidBack(e2) → {W₂} → denseRowBack(W₂) → swishBack(e1) → {W₁}`.
    Returns `(code, dx, [zW1, zb1, zW2, zb2 updated names])`. -/
private def seBack (adam : Bool) (B c hh r : Nat)
    (lrStr p drName sName e1Name zName e2Name seCot : String) :
    StateM Nat (String × String × List String) := do
  let ww := hh
  let zChw : Vec (B * (c * hh * ww)) := fun _ => 0
  let zCc  : Vec (B * c) := fun _ => 0
  let zRr  : Vec (B * r) := fun _ => 0
  -- The SE gate is ONE row per example, so the row ops sit at `rows = 1` and their
  -- index is `B·(1·c)` rather than `B·c`. Same vector, non-defeq type (`1 * c` does
  -- not reduce for a variable `c`), so the leaf needs its own placeholder; the emit
  -- is unaffected — `1 * c` evaluates to `c` when the render runs.
  let zCc1 : Vec (B * (1 * c)) := fun _ => 0
  let zW1  : Mat c r := fun _ _ => 0
  let zb1  : Vec r := fun _ => 0
  let zW2  : Mat r c := fun _ _ => 0
  let zb2  : Vec c := fun _ => 0
  let (cDx, nDx) ← pretty B (.seBackBatched (N := B) (c := c) (h := hh) (w := ww)
      s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2" drName zW1 zb1 zW2 zb2 zChw (.operand seCot zChw))
  let (cDg, nDg) ← pretty B (.seReduceB (N := B) (c := c) (h := hh) (w := ww) drName zChw (.operand seCot zChw))
  let (cE2c, nE2c) ← pretty B (.sigmoidBackB e2Name (fun _ => 0) (.operand nDg zCc))
  let (cW2, nW2) ← dnW adam B r c zName s!"%{p}zW2" lrStr nE2c
  let (cb2, nb2) ← dnB adam B c s!"%{p}zb2" lrStr nE2c
  let (cDz, nDz) ← pretty B (.batchOp (.denseRowBack (rows := 1) (a := r) (c := c) s!"%{p}zW2" zW2) (.operand nE2c zCc1))
  let (cE1c, nE1c) ← pretty B (.swishBackB e1Name (fun _ => 0) (.operand nDz zRr))
  let (cW1, nW1) ← dnW adam B c r sName s!"%{p}zW1" lrStr nE1c
  let (cb1, nb1) ← dnB adam B r s!"%{p}zb1" lrStr nE1c
  pure (cDx ++ cDg ++ cE2c ++ cW2 ++ cb2 ++ cDz ++ cE1c ++ cW1 ++ cb1, nDx, [nW1, nb1, nW2, nb2])

-- ════════════════════════════════════════════════════════════════
-- § MBConv forward emitters (stride-1 expand / no-skip expand / strided expand / no-expand b1)
-- ════════════════════════════════════════════════════════════════

/-- Stride-1 expand MBConv forward body (shared by residual + no-skip): expand 1×1 conv-bn-swish →
    depthwise(kd) conv-bn-swish → SE → project 1×1 conv-bn. Returns the EFwd WITHOUT the final
    residual (caller adds the `addV` for residual blocks). -/
private def eFwdBody (B ic mid oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd := do
  let ww := hh
  let zIn  : Vec (B * (ic * hh * ww)) := fun _ => 0
  let zMid : Vec (B * (mid * hh * ww)) := fun _ => 0
  let zOut : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}eW" s!"%{p}eb" zKe zVm) (.operand xName zIn))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}eg" s!"%{p}ebt" epsStr 0 zVm zVm (.operand nEc zMid))
  let (cEr, nEr) ← pretty B (.batchOp (.swish) (.operand nEn zMid))
  let (cDc, nDc) ← pretty B (.batchOp (N := B) (.depthwise (h := hh) (w := ww) s!"%{p}dW" s!"%{p}db" zDk zVm) (.operand nEr zMid))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" s!"%{p}dbt" epsStr 0 zVm zVm (.operand nDc zMid))
  let (cDr, nDr) ← pretty B (.batchOp (.swish) (.operand nDn zMid))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B mid hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}pW" s!"%{p}pb" zKp zVo) (.operand nSe zMid))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" s!"%{p}pbt" epsStr 0 zVo zVo (.operand nPc zOut))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc,
         bns := [(nEc, mid, hh), (nDc, mid, hh), (nPc, oc, hh)] }

/-- **Residual stride-1 MBConv forward** (ic = oc): body + `addV` skip. -/
private def eFwd (B ic mid oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd := do
  let f ← eFwdBody B ic mid oc hh kd r epsStr p xName
  let zOut : Vec (B * (oc * hh * hh)) := fun _ => 0
  let (cA, nA) ← pretty B (.addVB (.operand f.o zOut) (.operand xName zOut))
  pure { f with code := f.code ++ cA, o := nA }

/-- **No-skip stride-1 MBConv forward** (ic ≠ oc, b9/b16): body, output = project-BN out. -/
private def eFwdNoSkip (B ic mid oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd :=
  eFwdBody B ic mid oc hh kd r epsStr p xName

/-- **Strided MBConv forward** (b2/b4/b6/b12): expand at the input `2hh×2ww`, depthwise downsamples
    `2hh×2ww → hh×ww`, project 1×1 at `hh×ww`. NO skip. -/
private def eFwdStrided (B ic mid oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd := do
  let ww := hh
  let zIn  : Vec (B * (ic * (2*hh) * (2*ww))) := fun _ => 0
  let zMidH : Vec (B * (mid * (2*hh) * (2*ww))) := fun _ => 0
  let zMid : Vec (B * (mid * hh * ww)) := fun _ => 0
  let zOut : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B) (.conv (h := 2*hh) (w := 2*ww) s!"%{p}eW" s!"%{p}eb" zKe zVm) (.operand xName zIn))
  let (cEn, nEn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eg" s!"%{p}ebt" epsStr 0 zVm zVm (.operand nEc zMidH))
  let (cEr, nEr) ← pretty B (.batchOp (.swish) (.operand nEn zMidH))
  let (cDc, nDc) ← pretty B (.batchOp (N := B) (.depthwiseStrided (h := hh) (w := ww) s!"%{p}dW" s!"%{p}db" zDk zVm) (.operand nEr zMidH))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" s!"%{p}dbt" epsStr 0 zVm zVm (.operand nDc zMid))
  let (cDr, nDr) ← pretty B (.batchOp (.swish) (.operand nDn zMid))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B mid hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}pW" s!"%{p}pb" zKp zVo) (.operand nSe zMid))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" s!"%{p}pbt" epsStr 0 zVo zVo (.operand nPc zOut))
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc,
         -- the expand stage runs at the block INPUT resolution, `2hh`; only the depthwise
         -- downsamples. That asymmetry is the one thing the stat layout gets wrong by default.
         bns := [(nEc, mid, 2*hh), (nDc, mid, hh), (nPc, oc, hh)] }

/-- **No-expand MBConv forward** (b1, t=1): depthwise(kd, on `ic` channels)-bn-swish → SE → project
    1×1 (ic→oc)-bn. NO expand, NO skip. `ec/en` unused; `er` = block input (= depthwise input). -/
private def eFwdNoExp (B ic oc hh kd r : Nat) (epsStr p xName : String) : StateM Nat EFwd := do
  let ww := hh
  let zIn  : Vec (B * (ic * hh * ww)) := fun _ => 0
  let zOut : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel ic kd kd := fun _ _ _ => 0
  let zVi  : Vec ic := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cDc, nDc) ← pretty B (.batchOp (N := B) (.depthwise (h := hh) (w := ww) s!"%{p}dW" s!"%{p}db" zDk zVi) (.operand xName zIn))
  let (cDn, nDn) ← pretty B (.bnBatchF (N := B) (oc := ic) (h := hh) (w := ww) s!"%{p}dg" s!"%{p}dbt" epsStr 0 zVi zVi (.operand nDc zIn))
  let (cDr, nDr) ← pretty B (.batchOp (.swish) (.operand nDn zIn))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B ic hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}pW" s!"%{p}pb" zKp zVo) (.operand nSe zIn))
  let (cPn, nPn) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" s!"%{p}pbt" epsStr 0 zVo zVo (.operand nPc zOut))
  pure { code := cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := xName, en := xName, er := xName, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc,
         -- NO expand entry: `ec` is the block input here, not a BN input.
         bns := [(nDc, ic, hh), (nPc, oc, hh)] }

-- ════════════════════════════════════════════════════════════════
-- § MBConv backward emitters (project → SE → depthwise → expand; param-SGD in func-arg order)
-- ════════════════════════════════════════════════════════════════

/-- Stride-1 expand MBConv backward body (shared by residual + no-skip): returns the EBack with `dx`
    = the expand-conv-back cotangent (caller adds the residual `+ dyOut` for residual blocks). -/
private def eBackBody (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack := do
  let ww := hh
  let zMidF : Vec (B * (mid * hh * ww)) := fun _ => 0
  let zMidB : Vec (B * (mid * (hh * ww))) := fun _ => 0
  let zOutF : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zOutB : Vec (B * (oc * (hh * ww))) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  -- project: BN back (cot at project conv out) → 1×1 conv back (cot at SE out)
  let (cPbn, nPbn) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" f.pc epsStr 0 zVo zOutB (.operand dyName zOutB))
  let (cPdr, nPdr) ← pretty B (.convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}pW" zKp zVo (.operand nPbn zOutF))
  let (cgp, ngp) ← bnG adam B oc hh ww s!"%{p}pg" f.pc epsStr lrStr dyName
  let (ctp, ntp) ← bnBt adam B oc hh ww s!"%{p}pbt" lrStr dyName
  let (cWp, nWp) ← convW1 adam B mid oc hh ww f.se s!"%{p}pW" lrStr nPbn
  let (cbp, nbp) ← bnBt adam B oc hh ww s!"%{p}pb" lrStr nPbn
  -- SE back (dx at depthwise-swish out) + 4 SE param grads
  let (cSe, nDxSe, seNames) ← seBack adam B mid hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise: swish mask (cot at dw-BN out) → BN back (cot at dw conv out) → conv back (cot at expand-swish out)
  let (cDsw, nDsw) ← pretty B (.swishBackB f.dn (fun _ => 0) (.operand nDxSe zMidF))
  let (cDbn, nDbn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr 0 zVm zMidB (.operand nDsw zMidB))
  let (cDer, nDer) ← pretty B (.depthwiseBackBatched (N := B) (c := mid) (h := hh) (w := ww) s!"%{p}dW" zDk zVm (.operand nDbn zMidF))
  let (cgd, ngd) ← bnG adam B mid hh ww s!"%{p}dg" f.dc epsStr lrStr nDsw
  let (ctd, ntd) ← bnBt adam B mid hh ww s!"%{p}dbt" lrStr nDsw
  let (cWd, nWd) ← dwW adam B mid hh ww kd f.er s!"%{p}dW" lrStr nDbn
  let (cbd, nbd) ← bnBt adam B mid hh ww s!"%{p}db" lrStr nDbn
  -- expand: swish mask (cot at expand-BN out) → BN back → 1×1 conv back (cot at block input)
  let (cEsw, nEsw) ← pretty B (.swishBackB f.en (fun _ => 0) (.operand nDer zMidF))
  let (cEbn, nEbn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}eg" f.ec epsStr 0 zVm zMidB (.operand nEsw zMidB))
  let (cExb, nExb) ← pretty B (.convBackBatched (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%{p}eW" zKe zVm (.operand nEbn zMidF))
  let (cge, nge) ← bnG adam B mid hh ww s!"%{p}eg" f.ec epsStr lrStr nEsw
  let (cte, nte) ← bnBt adam B mid hh ww s!"%{p}ebt" lrStr nEsw
  let (cWe, nWe) ← convW1 adam B ic mid hh ww xName s!"%{p}eW" lrStr nEbn
  let (cbe, nbe) ← bnBt adam B mid hh ww s!"%{p}eb" lrStr nEbn
  let names := [nWe, nbe, nge, nte, nWd, nbd, ngd, ntd] ++ seNames ++ [nWp, nbp, ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDer ++ cgd ++ ctd ++ cWd ++ cbd ++
                 cEsw ++ cEbn ++ cExb ++ cge ++ cte ++ cWe ++ cbe,
         dx := nExb, names := names }

/-- **Residual stride-1 MBConv backward** (ic = oc): body + skip fan-in `+ dyOut`. -/
private def eBack (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack := do
  let b ← eBackBody adam B ic mid oc hh kd r epsStr lrStr p xName f dyName
  let zIn : Vec (B * (ic * hh * hh)) := fun _ => 0
  let (cDx, nDx) ← pretty B (.addVB (.operand b.dx zIn) (.operand dyName zIn))
  pure { b with code := b.code ++ cDx, dx := nDx }

/-- **No-skip stride-1 MBConv backward** (ic ≠ oc, b9/b16): body, dx = expand-conv-back directly. -/
private def eBackNoSkip (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack :=
  eBackBody adam B ic mid oc hh kd r epsStr lrStr p xName f dyName

/-- **Strided MBConv backward** (b2/b4/b6/b12): depthwise-back upsamples `hh×ww → 2hh×2ww`; the
    expand stage backward runs at `2hh×2ww`. NO skip. -/
private def eBackStrided (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack := do
  let ww := hh
  let zMidHF : Vec (B * (mid * (2*hh) * (2*ww))) := fun _ => 0
  let zMidHB : Vec (B * (mid * ((2*hh) * (2*ww)))) := fun _ => 0
  let zMidF : Vec (B * (mid * hh * ww)) := fun _ => 0
  let zMidB : Vec (B * (mid * (hh * ww))) := fun _ => 0
  let zOutF : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zOutB : Vec (B * (oc * (hh * ww))) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  -- project (at hh)
  let (cPbn, nPbn) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" f.pc epsStr 0 zVo zOutB (.operand dyName zOutB))
  let (cPdr, nPdr) ← pretty B (.convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}pW" zKp zVo (.operand nPbn zOutF))
  let (cgp, ngp) ← bnG adam B oc hh ww s!"%{p}pg" f.pc epsStr lrStr dyName
  let (ctp, ntp) ← bnBt adam B oc hh ww s!"%{p}pbt" lrStr dyName
  let (cWp, nWp) ← convW1 adam B mid oc hh ww f.se s!"%{p}pW" lrStr nPbn
  let (cbp, nbp) ← bnBt adam B oc hh ww s!"%{p}pb" lrStr nPbn
  -- SE back (at hh)
  let (cSe, nDxSe, seNames) ← seBack adam B mid hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise (swish + BN at hh, strided conv-back upsamples to 2hh)
  let (cDsw, nDsw) ← pretty B (.swishBackB f.dn (fun _ => 0) (.operand nDxSe zMidF))
  let (cDbn, nDbn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr 0 zVm zMidB (.operand nDsw zMidB))
  let (cDer, nDer) ← pretty B (.depthwiseStridedBackBatched (N := B) (c := mid) (h := hh) (w := ww) s!"%{p}dW" zDk zVm (.operand nDbn zMidF))
  let (cgd, ngd) ← bnG adam B mid hh ww s!"%{p}dg" f.dc epsStr lrStr nDsw
  let (ctd, ntd) ← bnBt adam B mid hh ww s!"%{p}dbt" lrStr nDsw
  let (cWd, nWd) ← dwWS adam B mid hh ww kd f.er s!"%{p}dW" lrStr nDbn
  let (cbd, nbd) ← bnBt adam B mid hh ww s!"%{p}db" lrStr nDbn
  -- expand (at 2hh)
  let (cEsw, nEsw) ← pretty B (.swishBackB f.en (fun _ => 0) (.operand nDer zMidHF))
  let (cEbn, nEbn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eg" f.ec epsStr 0 zVm zMidHB (.operand nEsw zMidHB))
  let (cExb, nExb) ← pretty B (.convBackBatched (N := B) (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eW" zKe zVm (.operand nEbn zMidHF))
  let (cge, nge) ← bnG adam B mid (2*hh) (2*ww) s!"%{p}eg" f.ec epsStr lrStr nEsw
  let (cte, nte) ← bnBt adam B mid (2*hh) (2*ww) s!"%{p}ebt" lrStr nEsw
  let (cWe, nWe) ← convW1 adam B ic mid (2*hh) (2*ww) xName s!"%{p}eW" lrStr nEbn
  let (cbe, nbe) ← bnBt adam B mid (2*hh) (2*ww) s!"%{p}eb" lrStr nEbn
  let names := [nWe, nbe, nge, nte, nWd, nbd, ngd, ntd] ++ seNames ++ [nWp, nbp, ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDer ++ cgd ++ ctd ++ cWd ++ cbd ++
                 cEsw ++ cEbn ++ cExb ++ cge ++ cte ++ cWe ++ cbe,
         dx := nExb, names := names }

/-- **No-expand MBConv backward** (b1): project back → SE back → depthwise back → dx (block input).
    8 params (Wd bd gd btd zW1 zb1 zW2 zb2 ... wait, 4 dw + 4 SE + 4 proj = 12). -/
private def eBackNoExp (adam : Bool) (B ic oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) : StateM Nat EBack := do
  let ww := hh
  let zInF  : Vec (B * (ic * hh * ww)) := fun _ => 0
  let zInB  : Vec (B * (ic * (hh * ww))) := fun _ => 0
  let zOutF : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zOutB : Vec (B * (oc * (hh * ww))) := fun _ => 0
  let zKp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel ic kd kd := fun _ _ _ => 0
  let zVi  : Vec ic := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  -- project
  let (cPbn, nPbn) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}pg" f.pc epsStr 0 zVo zOutB (.operand dyName zOutB))
  let (cPdr, nPdr) ← pretty B (.convBackBatched (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) s!"%{p}pW" zKp zVo (.operand nPbn zOutF))
  let (cgp, ngp) ← bnG adam B oc hh ww s!"%{p}pg" f.pc epsStr lrStr dyName
  let (ctp, ntp) ← bnBt adam B oc hh ww s!"%{p}pbt" lrStr dyName
  let (cWp, nWp) ← convW1 adam B ic oc hh ww f.se s!"%{p}pW" lrStr nPbn
  let (cbp, nbp) ← bnBt adam B oc hh ww s!"%{p}pb" lrStr nPbn
  -- SE back (on ic channels)
  let (cSe, nDxSe, seNames) ← seBack adam B ic hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise (on ic channels)
  let (cDsw, nDsw) ← pretty B (.swishBackB f.dn (fun _ => 0) (.operand nDxSe zInF))
  let (cDbn, nDbn) ← pretty B (.bnBatchBack (N := B) (oc := ic) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr 0 zVi zInB (.operand nDsw zInB))
  let (cDxb, nDxb) ← pretty B (.depthwiseBackBatched (N := B) (c := ic) (h := hh) (w := ww) s!"%{p}dW" zDk zVi (.operand nDbn zInF))
  let (cgd, ngd) ← bnG adam B ic hh ww s!"%{p}dg" f.dc epsStr lrStr nDsw
  let (ctd, ntd) ← bnBt adam B ic hh ww s!"%{p}dbt" lrStr nDsw
  let (cWd, nWd) ← dwW adam B ic hh ww kd xName s!"%{p}dW" lrStr nDbn
  let (cbd, nbd) ← bnBt adam B ic hh ww s!"%{p}db" lrStr nDbn
  let names := [nWd, nbd, ngd, ntd] ++ seNames ++ [nWp, nbp, ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDxb ++ cgd ++ ctd ++ cWd ++ cbd,
         dx := nDxb, names := names }

-- ════════════════════════════════════════════════════════════════
-- § Param signature lists (func-arg order — names + SHAPES)
-- ════════════════════════════════════════════════════════════════

/-! Names are stored WITHOUT the leading `%` and shapes as `List Nat` rather than a rendered
`tensor<…>` string, because the AdamW render needs both forms: `%{nm}`/`%{nm}m`/`%{nm}v` for the
moment slots, and the raw dimensions for the emitted Adam ops (`adamMNextF`'s `ds`). The SGD
render's emitted text is unchanged — `ty ds` reproduces exactly the strings that used to be
stored. -/

private def eSig (p : String) (ic mid oc r kd : Nat) : List (String × List Nat) :=
  [(s!"{p}eW", [mid,ic,1,1]), (s!"{p}eb", [mid]), (s!"{p}eg", [mid]), (s!"{p}ebt", [mid]),
   (s!"{p}dW", [mid,1,kd,kd]), (s!"{p}db", [mid]), (s!"{p}dg", [mid]), (s!"{p}dbt", [mid]),
   (s!"{p}zW1", [mid,r]), (s!"{p}zb1", [r]), (s!"{p}zW2", [r,mid]), (s!"{p}zb2", [mid]),
   (s!"{p}pW", [oc,mid,1,1]), (s!"{p}pb", [oc]), (s!"{p}pg", [oc]), (s!"{p}pbt", [oc])]

private def eSigNoExp (p : String) (ic oc r kd : Nat) : List (String × List Nat) :=
  [(s!"{p}dW", [ic,1,kd,kd]), (s!"{p}db", [ic]), (s!"{p}dg", [ic]), (s!"{p}dbt", [ic]),
   (s!"{p}zW1", [ic,r]), (s!"{p}zb1", [r]), (s!"{p}zW2", [r,ic]), (s!"{p}zb2", [ic]),
   (s!"{p}pW", [oc,ic,1,1]), (s!"{p}pb", [oc]), (s!"{p}pg", [oc]), (s!"{p}pbt", [oc])]

/-- **Full 262-param EfficientNet-B0 signature**, func-arg order: stem(4) + b1 no-exp(12) +
    b2..b16 expand(15×16) + head(4) + dense(2) = 4+12+240+4+2 = 262 tensors. -/
def enetSig (nClasses : Nat) : List (String × List Nat) :=
  [("sW", [32,3,3,3]), ("sb", [32]), ("sg", [32]), ("sbt", [32])] ++
  eSigNoExp "b1" 32 16 8 3 ++
  eSig "b2"  16  96  24  4 3 ++ eSig "b3"  24 144  24  6 3 ++
  eSig "b4"  24 144  40  6 5 ++ eSig "b5"  40 240  40 10 5 ++
  eSig "b6"  40 240  80 10 3 ++ eSig "b7"  80 480  80 20 3 ++ eSig "b8"  80 480  80 20 3 ++
  eSig "b9"  80 480 112 20 5 ++ eSig "b10" 112 672 112 28 5 ++ eSig "b11" 112 672 112 28 5 ++
  eSig "b12" 112 672 192 28 5 ++ eSig "b13" 192 1152 192 48 5 ++ eSig "b14" 192 1152 192 48 5 ++
  eSig "b15" 192 1152 192 48 5 ++ eSig "b16" 192 1152 320 48 3 ++
  [("hW", [1280,320,1,1]), ("hb", [1280]), ("hg", [1280]), ("hbt", [1280])] ++
  [("Wd", [1280, nClasses]), ("bd", [nClasses])]

-- ════════════════════════════════════════════════════════════════
-- § The whole-net renderer
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000000 in
/-- **The whole-net forward + cotangent + backward traversal, SHARED by the SGD and AdamW renders.**

    Returns `(code, params, softmax, bns)`:

    * `code` — every emitted line, `pretty` of a verified `SHlo` node;
    * `params` — one SSA per parameter in `enetSig` order: the **updated param** at `adam := false`,
      the **un-fused gradient** at `adam := true`;
    * `softmax` — the softmax SSA, which the AdamW render's report-only `%loss` reads;
    * `bns` — the 49 BN layers as `(BN-input SSA, channels, spatial side)`, in the layout the
      driver's `bnChannels` and `@efficientnet_fwd_eval` expect.

    One traversal, two tails, for the reason `ViTRender.vitBackAll` has the same shape: duplicating
    the 16-MBConv backward for AdamW would be the double-writer disease one level down, in code
    rather than in artifacts — and two emitters for `efficientnet_train_step` computing *different*
    functions is exactly what §2a-quinquies found.

    The gate on that claim is cheap and exact: `efficientnet_train_step.mlir` must come back
    **byte-identical**. It does, because `pretty`'s fresh-name counter only advances on nodes that
    emit, and at `smooth := none` the smoothing chain emits nothing at all.

    `smooth` is the cotangent recipe. `none` → plain `softmax − onehot` with the batch mean folded
    into `lrStr` (the SGD render, unchanged). `some (α, −α/K, B)` → the label-smoothed cotangent
    with an explicit ÷B, `((softmax − onehot) + α·onehot − α/K)/B`, which is what the AdamW recipe
    trains on. The two are **different functions**, so this is not an optional refinement: a render
    built without it would fail the tie in a way that looks like a bug in the gradient ops. -/
private def enetBackAll (B nClasses : Nat) (epsStr lrStr : String) (adam : Bool)
    (smooth : Option (String × String × String) := none) :
    StateM Nat (String × List String × String × List (String × Nat × Nat)) := do
    -- ═══ stem: 3×3/s2 conv (3→32, 224→112) → bn → swish ═══
    let zx   : Vec (B * (3*224*224)) := fun _ => 0
    let zSk  : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let z32  : Vec 32 := fun _ => 0
    let z112F : Vec (B * (32*112*112)) := fun _ => 0
    let z112B : Vec (B * (32*(112*112))) := fun _ => 0
    let (cStc, nStc) ← pretty B (.batchOp (N := B) (.convStrided (h := 112) (w := 112) "%sW" "%sb" zSk z32) (.operand "%x" zx))
    let (cStn, nStn) ← pretty B (.bnBatchF (N := B) (oc := 32) (h := 112) (w := 112) "%sg" "%sbt" epsStr 0 z32 z32 (.operand nStc z112F))
    let (cStr, nStr) ← pretty B (.batchOp (.swish) (.operand nStn z112F))
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
    let z7F   : Vec (B * (320*7*7)) := fun _ => 0
    let zHk   : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
    let z1280 : Vec 1280 := fun _ => 0
    let zH7F  : Vec (B * (1280*7*7)) := fun _ => 0
    let zH7B  : Vec (B * (1280*(7*7))) := fun _ => 0
    let z1280c : Vec (B * 1280) := fun _ => 0
    let zWd   : Mat 1280 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    -- `zNCb` is the ROW-indexed logit leaf (`rows = 1`, so `B·(1·nClasses)`) that the softmax /
    -- row-back / sub / smoothing nodes all sit at. The two head-dense parameter tails want the
    -- PLAIN batched index `B·nClasses` instead — same vector, non-defeq type — so `dnW`/`dnB`
    -- build that leaf themselves. See `zCc1` for the same wrinkle inside the SE gate.
    let zNCb  : Vec (B * (1 * nClasses)) := fun _ => 0
    let (cHc, nHc) ← pretty B (.batchOp (N := B) (.conv (h := 7) (w := 7) "%hW" "%hb" zHk z1280) (.operand f16.o z7F))
    let (cHn, nHn) ← pretty B (.bnBatchF (N := B) (oc := 1280) (h := 7) (w := 7) "%hg" "%hbt" epsStr 0 z1280 z1280 (.operand nHc zH7F))
    let (cHr, nHr) ← pretty B (.batchOp (.swish) (.operand nHn zH7F))
    let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 1280) (h := 7) (w := 7)) (.operand nHr zH7F))
    let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC) (.operand nGap z1280c))
    let (cSm, nSm) ← pretty B (.batchOp (.softmaxRow (m := 1) (n := nClasses)) (.operand nLog zNCb))
    let (cD0, nD0) ← pretty B (.subB (.operand nSm zNCb) (.operand "%onehot" zNCb))
    -- ═══ the cotangent tail. `none` emits NOTHING (so the SGD render is byte-identical); `some`
    --     appends the label-smoothing chain `scaleB → addVB → shiftB → divConstB`, every line
    --     `pretty` of a verified node. `shiftB` emits `add x, dense<−α/K>` where the hand-written
    --     render emits `subtract x, α/K` — IEEE subtraction IS addition of the exact negation, so
    --     the two are bit-identical. `divConstB` emits a real `divide` rather than a multiply by
    --     `1/B`, which is only exact in binary32 at powers of two. ═══
    let (cSmooth, nDy) ← match smooth with
      | none => pure ("", nD0)
      | some (aStr, negAK, bStr) => do
          let (c1, n1) ← pretty B (.scaleB aStr 0 (.operand "%onehot" zNCb))
          let (c2, n2) ← pretty B (.addVB (.operand nD0 zNCb) (.operand n1 zNCb))
          let (c3, n3) ← pretty B (.shiftB negAK 0 (.operand n2 zNCb))
          let (c4, n4) ← pretty B (.divConstB bStr 0 (.operand n3 zNCb))
          pure (c1 ++ c2 ++ c3 ++ c4, n4)
    let cDy := cD0 ++ cSmooth
    -- ═══ head backward: dense back → GAP back → swish mask → bn back → 1×1 conv back ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (.denseRowBack (rows := 1) (a := 1280) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    let (cWfc, nWfc) ← dnW adam B 1280 nClasses nGap "%Wd" lrStr nDy
    let (cbfc, nbfc) ← dnB adam B nClasses "%bd" lrStr nDy
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 1280) (h := 7) (w := 7) (.operand nDgi z1280c))
    let (cHsw, nHsw) ← pretty B (.swishBackB nHn (fun _ => 0) (.operand nDgp zH7F))
    let (cHbn, nHbn) ← pretty B (.bnBatchBack (N := B) (oc := 1280) (h := 7) (w := 7) "%hg" nHc epsStr 0 z1280 zH7B (.operand nHsw zH7B))
    let (cHxb, nHxb) ← pretty B (.convBackBatched (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7) "%hW" zHk z1280 (.operand nHbn zH7F))
    let (cgh, ngh) ← bnG adam B 1280 7 7 "%hg" nHc epsStr lrStr nHsw
    let (cth, nth) ← bnBt adam B 1280 7 7 "%hbt" lrStr nHsw
    let (cWh, nWh) ← convW1 adam B 320 1280 7 7 f16.o "%hW" lrStr nHbn
    let (cbh, nbh) ← bnBt adam B 1280 7 7 "%hb" lrStr nHbn
    -- ═══ backward: 16 blocks reversed (cotangent threads from nHxb) ═══
    let b16 ← eBackNoSkip  adam B 192 1152 320 7 3 48 epsStr lrStr "b16" f15.o f16 nHxb
    let b15 ← eBack        adam B 192 1152 192 7 5 48 epsStr lrStr "b15" f14.o f15 b16.dx
    let b14 ← eBack        adam B 192 1152 192 7 5 48 epsStr lrStr "b14" f13.o f14 b15.dx
    let b13 ← eBack        adam B 192 1152 192 7 5 48 epsStr lrStr "b13" f12.o f13 b14.dx
    let b12 ← eBackStrided adam B 112 672 192  7 5 28 epsStr lrStr "b12" f11.o f12 b13.dx
    let b11 ← eBack        adam B 112 672 112 14 5 28 epsStr lrStr "b11" f10.o f11 b12.dx
    let b10 ← eBack        adam B 112 672 112 14 5 28 epsStr lrStr "b10" f9.o  f10 b11.dx
    let b9  ← eBackNoSkip  adam B 80 480 112  14 5 20 epsStr lrStr "b9"  f8.o  f9  b10.dx
    let b8  ← eBack        adam B 80 480  80  14 3 20 epsStr lrStr "b8"  f7.o  f8  b9.dx
    let b7  ← eBack        adam B 80 480  80  14 3 20 epsStr lrStr "b7"  f6.o  f7  b8.dx
    let b6  ← eBackStrided adam B 40 240  80  14 3 10 epsStr lrStr "b6"  f5.o  f6  b7.dx
    let b5  ← eBack        adam B 40 240  40  28 5 10 epsStr lrStr "b5"  f4.o  f5  b6.dx
    let b4  ← eBackStrided adam B 24 144  40  28 5  6 epsStr lrStr "b4"  f3.o  f4  b5.dx
    let b3  ← eBack        adam B 24 144  24  56 3  6 epsStr lrStr "b3"  f2.o  f3  b4.dx
    let b2  ← eBackStrided adam B 16  96  24  56 3  4 epsStr lrStr "b2"  f1.o  f2  b3.dx
    let b1  ← eBackNoExp   adam B 32      16 112 3  8 epsStr lrStr "b1"  nStr  f1  b2.dx
    -- ═══ stem backward: swish mask → bn back, then the 4 stem params (NO conv-back past %x) ═══
    let (cDsr, nDsr) ← pretty B (.swishBackB nStn (fun _ => 0) (.operand b1.dx z112F))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := B) (oc := 32) (h := 112) (w := 112) "%sg" nStc epsStr 0 z32 z112B (.operand nDsr z112B))
    let (csW, nsW) ← if adam then
        pretty B (.convStridedWeightGradB (N := B) (ic := 3) (oc := 32) (h := 112) (w := 112) "%x" z32 zx zSk (.operand nDsn z112F))
      else
        pretty B (.convStridedWeightSgdB (N := B) (ic := 3) (oc := 32) (h := 112) (w := 112) "%x" "%sW" lrStr z32 zx zSk 0 (.operand nDsn z112F))
    let (csb, nsb) ← bnBt adam B 32 112 112 "%sb" lrStr nDsn
    let (csg, nsg) ← bnG adam B 32 112 112 "%sg" nStc epsStr lrStr nDsr
    let (cst, nst) ← bnBt adam B 32 112 112 "%sbt" lrStr nDsr
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
    -- BN layers in the driver's order: stem → each block's (expand?, depthwise, project) → head.
    let bnList : List (String × Nat × Nat) := [(nStc, 32, 112)] ++
      f1.bns ++ f2.bns ++ f3.bns ++ f4.bns ++ f5.bns ++ f6.bns ++ f7.bns ++ f8.bns ++
      f9.bns ++ f10.bns ++ f11.bns ++ f12.bns ++ f13.bns ++ f14.bns ++ f15.bns ++ f16.bns ++
      [(nHc, 1280, 7)]
    pure (fwdCode ++ bwdCode, outNames, nSm, bnList)

set_option maxRecDepth 4000000 in
/-- **EfficientNet-B0 (full 16-MBConv) SGD train step rendered ENTIRELY from the verified AST**, at
    the batched index `N·(c·h·w)`. Every emitted line is `pretty` of a verified `SHlo` node. Strided
    stem 3×3/s2 (3→32, 224→112) → b1 (no-expand) → b2..b16 (4 strided downsamples 112→7, 9 residual
    skips, 2 no-skip widenings) → 1×1 conv-bn-swish head (320→1280) → GAP → dense (1280→nClasses).

    The cotangent is plain `softmax − onehot` with the batch mean folded into `lrStr` — so the
    committed `lrStr = 0.05` is an effective **1.6** on the mean loss. That is a tuned value, not a
    slip (`runs/efficientnet_verified_crop_gpu1.log`: 40.6% → 87.81% over 80 epochs, matching
    README's 87.58%); the AdamW render below spells the mean explicitly instead. -/
def efficientnetTrainStepFaithfulV (B nClasses : Nat) (epsStr lrStr : String)
    (funcName : String := "efficientnet_train_step") : String :=
  let go : StateM Nat String := do
    let (code, outNames, _, _) ← enetBackAll B nClasses epsStr lrStr false
    let outTypes : List String := (enetSig nClasses).map (fun p => ty p.2)
    pure <|
      "    // ── EfficientNet-B0 (16-MBConv) train step: every line is pretty(verified AST node) ──\n" ++
      code ++
      s!"    return {String.intercalate ", " outNames} : {String.intercalate ", " outTypes}\n"
  let sigList := enetSig nClasses
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, ds) => s!"%{n}: {ty ds}")) ++
    s!", %onehot: {ty [B, nClasses]}"
  let outSig := String.intercalate ", " (sigList.map (fun p => ty p.2))
  let inner : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @{funcName}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § The AdamW tail — one proven triple per parameter, folded in signature order
-- ════════════════════════════════════════════════════════════════

/-- `(θ', m', v')` for one parameter, from its un-fused gradient. The three ops are the proven
    `adamMNextF`/`adamVNextF`/`adamWParamF` (`adamW_triple_faithful` bundles their `den`s into
    `Proofs.adamWStep` by `rfl`). β₁/β₂/ε/wd are baked literals; `%lr`/`%bc1`/`%bc2` are runtime
    `tensor<f32>` args, so one render serves the whole cosine+warmup schedule.

    At `replicas > 1` the gradient is first averaged across devices by
    `ViTRender.emitGradAllReduce`. **That collective is a TRUSTED CARVE-OUT** — it is emitted text,
    not `pretty` of an AST node, so it sits outside every faithfulness theorem here. What the proofs
    still cover is the whole rest of the graph: the AdamW triple consumes the averaged gradient as
    an `.operand` exactly as it consumed the raw one, so the `den` side does not shift. What is
    trusted is that `all_reduce(add)/N` computes the mean — handoff §5, *"the gradient averaging is
    a proven identity; the collective implementing it is trusted, exactly like the lowerer."*

    At `replicas ≤ 1` this emits **nothing** and threads the raw gradient, so the single-device
    render stays byte-identical — the cheap self-check that the insertion is inert.

    Mirrors `ResNet34RenderB.adamOne` and `ViTRender.vitAdamOne`. -/
private def enetAdamOne (B : Nat) (nm : String) (ds : List Nat) (gradSSA : String)
    (replicas : Nat) : StateM Nat (String × String × String × String) := do
  let n := ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let (arS, gAvg) := ViTRender.emitGradAllReduce gradSSA ds nm replicas
  let gr : SHlo n := .operand gAvg z
  let (cM, nM) ← pretty B (.adamMNextF s!"%{nm}m" "%b1" "%ob1" ds 0 z gr)
  let (cV, nV) ← pretty B (.adamVNextF s!"%{nm}v" "%b2" "%ob2" ds 0 z gr)
  let (cT, nT) ← pretty B (.adamWParamF s!"%{nm}" s!"%{nm}m" s!"%{nm}v" "%b1" "%ob1"
                    "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" "%wd" ds 0 0 0 0 0 0 0 z z z gr)
  pure (arS ++ cM ++ cV ++ cT, nT, nM, nV)

/-- β₁/β₂/ε/wd as graph constants — the committed EfficientNet-B0 AdamW recipe
    (`efficientNetB0Config`: lr 1e-3, wd 1e-4, cosine + 3-epoch warmup).

    `%b1`/`%b2` do NOT collide with the MBConv blocks also called `b1`/`b2`: every block parameter
    carries a suffix (`%b1dW`, `%b2eW`, …), so the bare names are free. The hand-written render
    relies on the same thing. -/
private def enetAdamConsts : String :=
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"

/-- The driver's **variant slug** for a given replica count: the artifact is
    `verified_mlir/efficientnet_<variant>_train_step.mlir`, the entry point is
    `@efficientnet_<variant>_train_step`, and `LEAN_MLIR_VARIANT` selects it.

    All three must agree, or the shim refuses the call outright ("entry mismatch") rather than
    running the wrong graph — which is exactly what it did the first time R34's DP render kept the
    single-device name (§2b-quater). Deriving the name here is what stops it drifting from the
    `#eval` paths below; the `#guard`s at the bottom pin those literal paths against this function. -/
def enetAdamVariant (replicas : Nat) : String :=
  if replicas ≤ 1 then "adam" else "adamdp"

set_option maxRecDepth 4000000 in
/-- **EfficientNet-B0 AdamW train step rendered from the verified AST.** The certified peer of the
    hand-written `tests/TestEfficientNetTrain.lean` render that `efficientnet-verified-adam` has
    been training on.

    Same backward as `efficientnet_train_step` (`enetBackAll`, one traversal) but taking the
    **un-fused gradients**, each fed to the proven AdamW triple. Two things differ from the SGD
    render and both are load-bearing:

    * the cotangent is **label-smoothed** (α = 0.1, K = nClasses) with an **explicit ÷B**, where the
      SGD render is plain CE with the mean folded into `lr`. Measured against the hand-written
      AdamW emitter, this is the same gap ViT had; get it wrong and the tie fails in a way that
      looks like a bug in the gradient ops.
    * it returns the **BN running statistics** — batch μ/var per BN layer, `bnBatchMeanB`/
      `bnBatchVarB` recomputed from that layer's BN input — which the host EMAs into
      `@efficientnet_fwd_eval`'s frozen stats. The SGD render has no such outputs.

    Interface: 889 in (`%x`, 262 θ, 262 m, 262 v, `%lr`/`%bc1`/`%bc2`, 98 running-stat slots,
    `%onehot`) / 887 out (262 θ', 262 m', 262 v', `%loss`/`%bc1`/`%bc2`, 98 batch stats) —
    positionally identical to the hand-written render, so `trainAdamSched`'s packed `[θ|m|v]`
    protocol is unchanged.

    Unlike ViT's, this tie can pin the **forward bit-exactly**: EfficientNet has BatchNorm, so the
    returned batch statistics are a whole-net forward fingerprint no gradient touches. -/
def efficientnetAdamTrainStepFaithful (B nClasses : Nat) (epsStr : String)
    (alphaStr negAlphaKStr bStr : String) (replicas : Nat := 1) : String :=
  let sigList := enetSig nClasses
  -- `go` hands back the BN channel counts alongside the body, so the argument signature is built
  -- from the SAME list the traversal walked. Deriving the 49 slots independently would be a second
  -- source for the stat layout — and a misaligned stat slot is silent: the arities still match and
  -- the wrong layer's statistics simply flow into the wrong `@efficientnet_fwd_eval` slot.
  let go : StateM Nat (String × List Nat) := do
    let (code, gradNames, nSm, bnList) ←
      enetBackAll B nClasses epsStr "0.0" true (some (alphaStr, negAlphaKStr, bStr))
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT. `den` is the
    --     same `bnMean`/`bnVar` `bnBatchF` normalises by, so these ARE the statistics the forward
    --     used rather than a separately-derived approximation. ═══
    let mut statCode := ""
    let mut statNames : List String := []
    let mut statTypes : List String := []
    for (xn, oc, hh) in bnList do
      let zb : Vec (B * (oc * (hh * hh))) := fun _ => 0
      let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
      let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
      statCode := statCode ++ cM ++ cV
      statNames := statNames ++ [nM, nV]
      statTypes := statTypes ++ [ty [oc], ty [oc]]
    -- ═══ AdamW: one proven triple per parameter, in func-arg order ═══
    let mut adamCode := ""
    let mut thetaN : List String := []
    let mut mN : List String := []
    let mut vN : List String := []
    for i in [0:sigList.length] do
      let (nm, ds) := sigList[i]!
      let (c, nT, nM, nV) ← enetAdamOne B nm ds (gradNames[i]!) replicas
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]; mN := mN ++ [nM]; vN := vN ++ [nV]
    -- `%loss` is REPORT-ONLY: mean smoothed-CE for logging, on no gradient path. It is NOT
    -- `pretty` of an AST node and says so in the emitted text — the carve-out `resnet34_`/
    -- `cifar8_adam_train_step` also take (handoff §5).
    --   loss = −(1/B)·Σ_b [ (1−α)·Σ_k onehot·log sm  +  (α/K)·Σ_k log sm ]
    -- No theorem covers this, and nothing on a gradient path touches it, so it is precisely where
    -- §2b shipped PLAIN CE against a smoothed-CE cotangent and only the numeric tie caught it. It
    -- is therefore built from the SAME smoothed recipe the cotangent implies, and gated.
    let lossCode :=
      "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
      s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %llog = stablehlo.log {nSm} : {ty [B, nClasses]}\n" ++
      s!"    %lohll = stablehlo.multiply %onehot, %llog : {ty [B, nClasses]}\n" ++
      s!"    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B, nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
      s!"    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B, nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
      s!"    %lomac = stablehlo.constant dense<0.900000> : {ty [B]}\n" ++
      s!"    %laKc = stablehlo.constant dense<0.010000> : {ty [B]}\n" ++
      s!"    %llt1 = stablehlo.multiply %lomac, %lt1s : {ty [B]}\n" ++
      s!"    %llt2 = stablehlo.multiply %laKc, %llsr : {ty [B]}\n" ++
      s!"    %llpe = stablehlo.add %llt1, %llt2 : {ty [B]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [B]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
      s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
      s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
    let pTy := sigList.map (fun p => ty p.2)
    let retVals := thetaN ++ mN ++ vN ++ ["%loss", "%bc1", "%bc2"] ++ statNames
    let retTys := pTy ++ pTy ++ pTy ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statTypes
    pure (
      (if replicas ≤ 1 then
        "    // ── EfficientNet-B0 AdamW train step: gradients + optimizer are pretty(AST node) ──\n"
       else
        s!"    // ── EfficientNet-B0 AdamW train step, DATA-PARALLEL over {replicas} replicas ──\n" ++
        "    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`\n" ++
        "    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5), emitted\n" ++
        "    // text outside the faithfulness theorems. Each replica evaluates the same tied graph\n" ++
        "    // at the batch it was rendered for; the collective averages that function's gradients\n" ++
        "    // over disjoint equal batches. NOTE this does NOT equal a single-device step at the\n" ++
        "    // global batch — BN normalises per replica, so N×b != 1×(N·b) by design (§10.3b).\n") ++
      code ++ statCode ++ enetAdamConsts ++ adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n",
      bnList.map (fun t => t.2.1))
  let (inner, bnOc) := go.run' 0
  -- The 49 BN layers each get a dummy `[oc]` input slot, unused by the graph — they exist so the
  -- driver's argument buffer keeps the shape the generic FFI hands it, and the hand-written render
  -- has the same unused slots. It names them after its BN-layer prefixes; `pretty`'s intermediates
  -- are counter-named, so these are indexed instead. Inputs only: the returned statistics are
  -- recomputed above from each layer's BN input.
  let statSig := String.intercalate ", " ((List.range bnOc.length).map (fun i =>
    s!"%bnmu{i}i: {ty [bnOc[i]!]}, %bnvar{i}i: {ty [bnOc[i]!]}"))
  let pSig := String.intercalate ", " (sigList.map (fun (n, ds) => s!"%{n}: {ty ds}"))
  let mSig := String.intercalate ", " (sigList.map (fun (n, ds) => s!"%{n}m: {ty ds}"))
  let vSig := String.intercalate ", " (sigList.map (fun (n, ds) => s!"%{n}v: {ty ds}"))
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, " ++ statSig ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (fun p => ty p.2)
  let outSig := String.intercalate ", "
    (pTy ++ pTy ++ pTy ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++
     bnOc.flatMap (fun oc => [ty [oc], ty [oc]]))
  -- The entry name must track the driver's `{slug}_{variant}_train_step` convention, or the shim
  -- refuses the call ("entry mismatch"). `enetAdamVariant` is the single source for the name, the
  -- artifact path and `LEAN_MLIR_VARIANT`.
  let fname := s!"efficientnet_{enetAdamVariant replicas}_train_step"
  "module @m {\n" ++
  s!"  func.func @{fname}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- Regenerate `verified_mlir/efficientnet_train_step.mlir` (what MainEfficientNetVerified trains on)
-- from the faithful renderer: the FULL 16-MBConv B0 net (262 params). B=32, nClasses=10, ε=1e-5.
#eval IO.FS.writeFile "verified_mlir/efficientnet_train_step.mlir"
  (Proofs.StableHLO.efficientnetTrainStepFaithfulV 32 10 "1.0e-5" "0.05" "efficientnet_train_step")

-- The **AdamW** train step — **the artifact `efficientnet-verified-adam` trains on**, and from
-- 2026-07-28 this `#eval` is its ONLY writer. The hand-written emitter in
-- `tests/TestEfficientNetTrain.lean` is retired; that file now only iree-compiles the committed
-- bytes. The driver needed no change at all: it resolves the path from the net slug, so taking
-- over the canonical name IS the swap.
--
-- It rendered to a separate `…_b.mlir` until the tie passed, because two writers for one artifact
-- is the last-writer-wins race §2a found — and adding one would have recreated exactly what
-- §2a-quinquies removed. `lake build efficientnet-adam-tie` licensed the swap (one AdamW step,
-- all 12,166,117 returned floats): forward BIT-EXACT over the 98 BN batch statistics, `%loss`
-- bit-exact, gradient bit-exact, against a bit-exact A-vs-A determinism floor. To re-run it:
--
--   git show c96bd36:verified_mlir/efficientnet_adam_train_step.mlir > /tmp/retired.mlir
--   IREE_BACKEND=rocm .lake/build/bin/efficientnet-adam-tie /tmp/retired.mlir \
--     verified_mlir/efficientnet_adam_train_step.mlir
--
-- Literals: α = 0.1, −α/K = −0.01 (K = 10), batch 32 — `efficientNetB0Config`'s label smoothing
-- and explicit mean, matching the retired `adamCot` term for term.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adam_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0")

-- The **DATA-PARALLEL** render (handoff §2e-bis), selected at run time by
-- `LEAN_MLIR_VARIANT=adamdp`. Same graph, plus one `all_reduce(add)/N` per parameter gradient
-- between the certified gradient and the certified AdamW triple: *certified gradient → trusted
-- collective → certified AdamW*. The collective is a DECLARED carve-out and the render says so in
-- its own output banner at `replicas > 1`, per the §5/§2b `%loss` lesson that an undeclared
-- carve-out is how wrong things ship.
--
-- Unlike ResNet-34's, this variant never had a hand-written emitter to migrate off — the certified
-- renderer is the only writer of both EfficientNet AdamW artifacts from the start.
--
-- It renders to its OWN path, which is what stops the §2a race where producing a DP render meant
-- editing a knob and clobbering the artifact the trainer runs. `2` is the replica count these are
-- rendered at and it must match `PJRT_REPLICAS` at run time, because the graph bakes
-- `replica_groups`. Re-render here to change it.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adamdp_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" 2)

-- Pin the two literal artifact paths above against the name the renderer actually emits. If a
-- variant is renamed this fails at `lake build` instead of at run time as an "entry mismatch".
#guard Proofs.StableHLO.enetAdamVariant 1 == "adam"
#guard Proofs.StableHLO.enetAdamVariant 2 == "adamdp"
