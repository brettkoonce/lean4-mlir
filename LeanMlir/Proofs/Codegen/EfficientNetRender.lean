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
      match and the wrong layer's statistics simply flow into the wrong eval slot.

      The second component is the layer's **stat prefix** — `@efficientnet_fwd_eval` takes
      `%{prefix}mu`/`%{prefix}var` there, and the AdamW train step hands the matching batch μ/var
      back from the SAME entry. So the eval signature, the eval BN sites and the train step's stat
      outputs all come off this one list; there is deliberately no parallel 49-entry table. -/
  bns : List (String × String × Nat × Nat)
  deriving Inhabited

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
-- ── ▶ STOCHASTIC DEPTH (`planning/stochastic_depth.md`) ───────────────────────────────────────
-- EfficientNet-B0 has 16 MBConv blocks, and the drop fires on the 9 that carry a skip.
--
-- ⚠⚠ THE RAMP INDEX IS THE BLOCK INDEX, NOT THE SITE ORDINAL — and getting that wrong is the
-- expensive silent bug here. The reference advances `dbi` on EVERY block
-- (`if cfg.dropPath > 0 then dbi := dbi + 1`, unconditional) while the drop only FIRES inside the
-- skip guard `residual.shape == x.shape and stride == 1`. So `keep_i = 1 − dropRate·i/(16−1)` with
-- `i` the BLOCK index, and the denominator is **15**, not 8. Re-indexing by site would give nine
-- evenly-spaced keeps instead of the reference's nine unevenly-spaced ones: it compiles, runs,
-- descends, and trains a different objective. §2k's `α/K` bug in a new place.
--
-- b1 noExp · b2 strided · b3 SKIP · b4 strided · b5 SKIP · b6 strided · b7 SKIP · b8 SKIP
-- b9 noSkip · b10 SKIP · b11 SKIP · b12 strided · b13 SKIP · b14 SKIP · b15 SKIP · b16 noSkip
/-- Total MBConv blocks = the reference's `totalDrop`, i.e. the ramp DENOMINATOR is this minus 1. -/
def enetDropTotal : Nat := 16

/-- The block indices (0-based) that carry a drop site, in signature order. **This is the single
    source**: `enetDropSig` maps over it to build the `%dp<i>` inputs, the traversal passes the same
    `i` at each `eFwd` call site, and the driver reads it to know how many scales to supply and at
    which ramp index. The two routes fail LOUDLY if they disagree — an entry with no call site
    leaves an unused input (arity mismatch at the driver), and a call site with no entry emits an
    undeclared `%dp<i>` (rejected by the lowerer). Neither is silent, which is the §2m property. -/
def enetDropIdxs : List Nat := [2, 4, 6, 7, 9, 10, 12, 13, 14]

/-- The number of per-example drop-path scale inputs a stochastic-depth render takes. -/
def enetDropSites : Nat := enetDropIdxs.length

#guard enetDropTotal == 16
#guard enetDropSites == 9
-- Every site is a real block, and the list is strictly increasing (= signature order).
#guard enetDropIdxs.all (· < enetDropTotal)
#guard (enetDropIdxs.zip (enetDropIdxs.drop 1)).all (fun (a, b) => a < b)

/-- The drop-path scale input name for block index `i`. Signature and emit both read this. -/
def dpName (i : Nat) : String := s!"%dp{i}"

/-- The `%dp<i>: tensor<Bxf32>` inputs, appended to a render's signature when stochastic depth is
    on. Empty when off, which is what keeps gate 1 byte-identical. -/
def enetDropSig (B : Nat) (sd : Bool) : String :=
  if sd then String.join (enetDropIdxs.map (fun i => s!", {dpName i}: {ty [B]}")) else ""

-- § MBConv forward emitters (stride-1 expand / no-skip expand / strided expand / no-expand b1)
-- ════════════════════════════════════════════════════════════════

/-- One BN site at the batched index. `statP` is the running-stat input prefix
    (`%{statP}mu` / `%{statP}var`), used only in `.eval` mode; in `.train` mode the statistics are
    reduced out of `xin` and `statP` names the slot the train step hands them back in.

    This is the ONLY place the two BN worlds are chosen between, which is the point: `@efficientnet_fwd`
    and `@efficientnet_fwd_eval` come from one traversal, so they cannot be the same net with
    different normalisation the way `resnet34_fwd` and `resnet34_train_step` were (§2a). -/
private def bnSiteB (B oc hh ww : Nat) (mode : BnMode) (epsStr gName btName statP xin : String) :
    StateM Nat (String × String) := do
  let zc  : Vec oc := fun _ => 0
  let zin : Vec (B * (oc*hh*ww)) := fun _ => 0
  match mode with
  | .train => pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww)
                          gName btName epsStr 0 zc zc (.operand xin zin))
  | .eval  => pretty B (.batchOp (N := B) (.bnEval (oc := oc) (h := hh) (w := ww)
                          gName btName s!"%{statP}mu" s!"%{statP}var" epsStr 0 zc zc zc zc)
                          (.operand xin zin))

/-- Stride-1 expand MBConv forward body (shared by residual + no-skip): expand 1×1 conv-bn-swish →
    depthwise(kd) conv-bn-swish → SE → project 1×1 conv-bn. Returns the EFwd WITHOUT the final
    residual (caller adds the `addV` for residual blocks). -/
private def eFwdBody (B ic mid oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String) (convBias : Bool) : StateM Nat EFwd := do
  let ww := hh
  let zIn  : Vec (B * (ic * hh * ww)) := fun _ => 0
  let zMid : Vec (B * (mid * hh * ww)) := fun _ => 0
  let zOut : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}eW" (biasName convBias s!"%{p}eb" mid) zKe zVm) (.operand xName zIn))
  let (cEn, nEn) ← bnSiteB B mid hh ww mode epsStr s!"%{p}eg" s!"%{p}ebt" s!"{p}en" nEc
  let (cEr, nEr) ← pretty B (.batchOp (.swish) (.operand nEn zMid))
  let (cDc, nDc) ← pretty B (.batchOp (N := B) (.depthwise (h := hh) (w := ww) s!"%{p}dW" (biasName convBias s!"%{p}db" mid) zDk zVm) (.operand nEr zMid))
  let (cDn, nDn) ← bnSiteB B mid hh ww mode epsStr s!"%{p}dg" s!"%{p}dbt" s!"{p}dn" nDc
  let (cDr, nDr) ← pretty B (.batchOp (.swish) (.operand nDn zMid))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B mid hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}pW" (biasName convBias s!"%{p}pb" oc) zKp zVo) (.operand nSe zMid))
  let (cPn, nPn) ← bnSiteB B oc hh ww mode epsStr s!"%{p}pg" s!"%{p}pbt" s!"{p}pn" nPc
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc,
         bns := [(nEc, s!"{p}en", mid, hh), (nDc, s!"{p}dn", mid, hh), (nPc, s!"{p}pn", oc, hh)] }

/-- **Residual stride-1 MBConv forward** (ic = oc): body + `addV` skip. -/
private def eFwd (B ic mid oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String)
    (convBias : Bool) (drop : Option Nat := none) : StateM Nat EFwd := do
  let f ← eFwdBody B ic mid oc hh kd r mode epsStr p xName convBias
  let zOut : Vec (B * (oc * hh * hh)) := fun _ => 0
  -- ▶ THE DROP SITE — on the RESIDUAL BRANCH, before the skip add, which is where the reference
  -- puts it (`x = x * keep / keep_prob` then `x = x + residual`). Scaling after the add would
  -- attenuate the identity path too: a different net that still trains, and invisible to every
  -- structural check. `dropPath_zeros_zero` plus the all-zero-mask control is what pins it here.
  -- At `drop = none` no `pretty` call happens, so the fresh-name counter does not move and every
  -- committed artifact re-renders byte-identically — gate 1's strong form, for free.
  let (cD, nD) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (dpName i) (fun _ => 0 : Vec B) (.operand f.o zOut))
    | none   => pure ("", f.o)
  let (cA, nA) ← pretty B (.addVB (.operand nD zOut) (.operand xName zOut))
  pure { f with code := f.code ++ cD ++ cA, o := nA }

/-- **No-skip stride-1 MBConv forward** (ic ≠ oc, b9/b16): body, output = project-BN out. -/
private def eFwdNoSkip (B ic mid oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String) (convBias : Bool) : StateM Nat EFwd :=
  eFwdBody B ic mid oc hh kd r mode epsStr p xName convBias

/-- **Strided MBConv forward** (b2/b4/b6/b12): expand at the input `2hh×2ww`, depthwise downsamples
    `2hh×2ww → hh×ww`, project 1×1 at `hh×ww`. NO skip. -/
private def eFwdStrided (B ic mid oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String) (convBias : Bool) : StateM Nat EFwd := do
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
  let (cEc, nEc) ← pretty B (.batchOp (N := B) (.conv (h := 2*hh) (w := 2*ww) s!"%{p}eW" (biasName convBias s!"%{p}eb" mid) zKe zVm) (.operand xName zIn))
  let (cEn, nEn) ← bnSiteB B mid (2*hh) (2*ww) mode epsStr s!"%{p}eg" s!"%{p}ebt" s!"{p}en" nEc
  let (cEr, nEr) ← pretty B (.batchOp (.swish) (.operand nEn zMidH))
  let (cDc, nDc) ← pretty B (.batchOp (N := B) (.depthwiseStrided (h := hh) (w := ww) s!"%{p}dW" (biasName convBias s!"%{p}db" mid) zDk zVm) (.operand nEr zMidH))
  let (cDn, nDn) ← bnSiteB B mid hh ww mode epsStr s!"%{p}dg" s!"%{p}dbt" s!"{p}dn" nDc
  let (cDr, nDr) ← pretty B (.batchOp (.swish) (.operand nDn zMid))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B mid hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}pW" (biasName convBias s!"%{p}pb" oc) zKp zVo) (.operand nSe zMid))
  let (cPn, nPn) ← bnSiteB B oc hh ww mode epsStr s!"%{p}pg" s!"%{p}pbt" s!"{p}pn" nPc
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc,
         -- the expand stage runs at the block INPUT resolution, `2hh`; only the depthwise
         -- downsamples. That asymmetry is the one thing the stat layout gets wrong by default.
         bns := [(nEc, s!"{p}en", mid, 2*hh), (nDc, s!"{p}dn", mid, hh), (nPc, s!"{p}pn", oc, hh)] }

/-- **No-expand MBConv forward** (b1, t=1): depthwise(kd, on `ic` channels)-bn-swish → SE → project
    1×1 (ic→oc)-bn. NO expand, NO skip. `ec/en` unused; `er` = block input (= depthwise input). -/
private def eFwdNoExp (B ic oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String) (convBias : Bool) : StateM Nat EFwd := do
  let ww := hh
  let zIn  : Vec (B * (ic * hh * ww)) := fun _ => 0
  let zOut : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel ic kd kd := fun _ _ _ => 0
  let zVi  : Vec ic := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cDc, nDc) ← pretty B (.batchOp (N := B) (.depthwise (h := hh) (w := ww) s!"%{p}dW" (biasName convBias s!"%{p}db" ic) zDk zVi) (.operand xName zIn))
  let (cDn, nDn) ← bnSiteB B ic hh ww mode epsStr s!"%{p}dg" s!"%{p}dbt" s!"{p}dn" nDc
  let (cDr, nDr) ← pretty B (.batchOp (.swish) (.operand nDn zIn))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B ic hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}pW" (biasName convBias s!"%{p}pb" oc) zKp zVo) (.operand nSe zIn))
  let (cPn, nPn) ← bnSiteB B oc hh ww mode epsStr s!"%{p}pg" s!"%{p}pbt" s!"{p}pn" nPc
  pure { code := cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := xName, en := xName, er := xName, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc,
         -- NO expand entry: `ec` is the block input here, not a BN input.
         bns := [(nDc, s!"{p}dn", ic, hh), (nPc, s!"{p}pn", oc, hh)] }

-- ════════════════════════════════════════════════════════════════
-- § MBConv backward emitters (project → SE → depthwise → expand; param-SGD in func-arg order)
-- ════════════════════════════════════════════════════════════════

/-- Stride-1 expand MBConv backward body (shared by residual + no-skip): returns the EBack with `dx`
    = the expand-conv-back cotangent (caller adds the residual `+ dyOut` for residual blocks). -/
private def eBackBody (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool) : StateM Nat EBack := do
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
  let (cbp, nbp) ← if convBias then bnBt adam B oc hh ww s!"%{p}pb" lrStr nPbn else pure ("", "")
  -- SE back (dx at depthwise-swish out) + 4 SE param grads
  let (cSe, nDxSe, seNames) ← seBack adam B mid hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise: swish mask (cot at dw-BN out) → BN back (cot at dw conv out) → conv back (cot at expand-swish out)
  let (cDsw, nDsw) ← pretty B (.swishBackB f.dn (fun _ => 0) (.operand nDxSe zMidF))
  let (cDbn, nDbn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr 0 zVm zMidB (.operand nDsw zMidB))
  let (cDer, nDer) ← pretty B (.depthwiseBackBatched (N := B) (c := mid) (h := hh) (w := ww) s!"%{p}dW" zDk zVm (.operand nDbn zMidF))
  let (cgd, ngd) ← bnG adam B mid hh ww s!"%{p}dg" f.dc epsStr lrStr nDsw
  let (ctd, ntd) ← bnBt adam B mid hh ww s!"%{p}dbt" lrStr nDsw
  let (cWd, nWd) ← dwW adam B mid hh ww kd f.er s!"%{p}dW" lrStr nDbn
  let (cbd, nbd) ← if convBias then bnBt adam B mid hh ww s!"%{p}db" lrStr nDbn else pure ("", "")
  -- expand: swish mask (cot at expand-BN out) → BN back → 1×1 conv back (cot at block input)
  let (cEsw, nEsw) ← pretty B (.swishBackB f.en (fun _ => 0) (.operand nDer zMidF))
  let (cEbn, nEbn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}eg" f.ec epsStr 0 zVm zMidB (.operand nEsw zMidB))
  let (cExb, nExb) ← pretty B (.convBackBatched (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww) s!"%{p}eW" zKe zVm (.operand nEbn zMidF))
  let (cge, nge) ← bnG adam B mid hh ww s!"%{p}eg" f.ec epsStr lrStr nEsw
  let (cte, nte) ← bnBt adam B mid hh ww s!"%{p}ebt" lrStr nEsw
  let (cWe, nWe) ← convW1 adam B ic mid hh ww xName s!"%{p}eW" lrStr nEbn
  let (cbe, nbe) ← if convBias then bnBt adam B mid hh ww s!"%{p}eb" lrStr nEbn else pure ("", "")
  let names := [nWe] ++ biasSlot convBias nbe ++ [nge, nte, nWd] ++ biasSlot convBias nbd ++
               [ngd, ntd] ++ seNames ++ [nWp] ++ biasSlot convBias nbp ++ [ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDer ++ cgd ++ ctd ++ cWd ++ cbd ++
                 cEsw ++ cEbn ++ cExb ++ cge ++ cte ++ cWe ++ cbe,
         dx := nExb, names := names }

/-- **Residual stride-1 MBConv backward** (ic = oc): body + skip fan-in `+ dyOut`. -/
private def eBack (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool) (drop : Option Nat := none) : StateM Nat EBack := do
  -- ▶ THE DROP'S BACKWARD IS THE SAME OP AT THE SAME SCALE (`Proofs.dropPath_vjp_is_self`) — a
  -- diagonal linear map is its own transpose, so there is no `*Grad` peer to keep in step.
  -- ⚠ IT APPLIES TO THE BRANCH ONLY. `eBackBody` consumes this cotangent for the whole branch
  -- INCLUDING its parameter gradients, so feeding it `dyd` is what makes the project/depthwise/
  -- expand grads see the dropped signal; the skip's fan-in below keeps the RAW `dyName`. Dropping
  -- there too would attenuate the identity path — the mirror of the forward's placement trap.
  let zOut : Vec (B * (oc * hh * hh)) := fun _ => 0
  let (cD, dyd) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (dpName i) (fun _ => 0 : Vec B) (.operand dyName zOut))
    | none   => pure ("", dyName)
  let b ← eBackBody adam B ic mid oc hh kd r epsStr lrStr p xName f dyd convBias
  let zIn : Vec (B * (ic * hh * hh)) := fun _ => 0
  let (cDx, nDx) ← pretty B (.addVB (.operand b.dx zIn) (.operand dyName zIn))
  pure { b with code := cD ++ b.code ++ cDx, dx := nDx }

/-- **No-skip stride-1 MBConv backward** (ic ≠ oc, b9/b16): body, dx = expand-conv-back directly. -/
private def eBackNoSkip (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool) : StateM Nat EBack :=
  eBackBody adam B ic mid oc hh kd r epsStr lrStr p xName f dyName convBias

/-- **Strided MBConv backward** (b2/b4/b6/b12): depthwise-back upsamples `hh×ww → 2hh×2ww`; the
    expand stage backward runs at `2hh×2ww`. NO skip. -/
private def eBackStrided (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool) : StateM Nat EBack := do
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
  let (cbp, nbp) ← if convBias then bnBt adam B oc hh ww s!"%{p}pb" lrStr nPbn else pure ("", "")
  -- SE back (at hh)
  let (cSe, nDxSe, seNames) ← seBack adam B mid hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise (swish + BN at hh, strided conv-back upsamples to 2hh)
  let (cDsw, nDsw) ← pretty B (.swishBackB f.dn (fun _ => 0) (.operand nDxSe zMidF))
  let (cDbn, nDbn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr 0 zVm zMidB (.operand nDsw zMidB))
  let (cDer, nDer) ← pretty B (.depthwiseStridedBackBatched (N := B) (c := mid) (h := hh) (w := ww) s!"%{p}dW" zDk zVm (.operand nDbn zMidF))
  let (cgd, ngd) ← bnG adam B mid hh ww s!"%{p}dg" f.dc epsStr lrStr nDsw
  let (ctd, ntd) ← bnBt adam B mid hh ww s!"%{p}dbt" lrStr nDsw
  let (cWd, nWd) ← dwWS adam B mid hh ww kd f.er s!"%{p}dW" lrStr nDbn
  let (cbd, nbd) ← if convBias then bnBt adam B mid hh ww s!"%{p}db" lrStr nDbn else pure ("", "")
  -- expand (at 2hh)
  let (cEsw, nEsw) ← pretty B (.swishBackB f.en (fun _ => 0) (.operand nDer zMidHF))
  let (cEbn, nEbn) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eg" f.ec epsStr 0 zVm zMidHB (.operand nEsw zMidHB))
  let (cExb, nExb) ← pretty B (.convBackBatched (N := B) (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}eW" zKe zVm (.operand nEbn zMidHF))
  let (cge, nge) ← bnG adam B mid (2*hh) (2*ww) s!"%{p}eg" f.ec epsStr lrStr nEsw
  let (cte, nte) ← bnBt adam B mid (2*hh) (2*ww) s!"%{p}ebt" lrStr nEsw
  let (cWe, nWe) ← convW1 adam B ic mid (2*hh) (2*ww) xName s!"%{p}eW" lrStr nEbn
  let (cbe, nbe) ← if convBias then bnBt adam B mid (2*hh) (2*ww) s!"%{p}eb" lrStr nEbn else pure ("", "")
  let names := [nWe] ++ biasSlot convBias nbe ++ [nge, nte, nWd] ++ biasSlot convBias nbd ++
               [ngd, ntd] ++ seNames ++ [nWp] ++ biasSlot convBias nbp ++ [ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDer ++ cgd ++ ctd ++ cWd ++ cbd ++
                 cEsw ++ cEbn ++ cExb ++ cge ++ cte ++ cWe ++ cbe,
         dx := nExb, names := names }

/-- **No-expand MBConv backward** (b1): project back → SE back → depthwise back → dx (block input).
    8 params (Wd bd gd btd zW1 zb1 zW2 zb2 ... wait, 4 dw + 4 SE + 4 proj = 12). -/
private def eBackNoExp (adam : Bool) (B ic oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool) : StateM Nat EBack := do
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
  let (cbp, nbp) ← if convBias then bnBt adam B oc hh ww s!"%{p}pb" lrStr nPbn else pure ("", "")
  -- SE back (on ic channels)
  let (cSe, nDxSe, seNames) ← seBack adam B ic hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise (on ic channels)
  let (cDsw, nDsw) ← pretty B (.swishBackB f.dn (fun _ => 0) (.operand nDxSe zInF))
  let (cDbn, nDbn) ← pretty B (.bnBatchBack (N := B) (oc := ic) (h := hh) (w := ww) s!"%{p}dg" f.dc epsStr 0 zVi zInB (.operand nDsw zInB))
  let (cDxb, nDxb) ← pretty B (.depthwiseBackBatched (N := B) (c := ic) (h := hh) (w := ww) s!"%{p}dW" zDk zVi (.operand nDbn zInF))
  let (cgd, ngd) ← bnG adam B ic hh ww s!"%{p}dg" f.dc epsStr lrStr nDsw
  let (ctd, ntd) ← bnBt adam B ic hh ww s!"%{p}dbt" lrStr nDsw
  let (cWd, nWd) ← dwW adam B ic hh ww kd xName s!"%{p}dW" lrStr nDbn
  let (cbd, nbd) ← if convBias then bnBt adam B ic hh ww s!"%{p}db" lrStr nDbn else pure ("", "")
  let names := [nWd] ++ biasSlot convBias nbd ++ [ngd, ntd] ++ seNames ++
               [nWp] ++ biasSlot convBias nbp ++ [ngp, ntp]
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

private def eSig (p : String) (ic mid oc r kd : Nat) (convBias : Bool) : List (String × List Nat) :=
  -- ⚠ `zb1`/`zb2` are the SQUEEZE-EXCITE biases and STAY: those convs are followed by an
  -- activation, not BN, so nothing absorbs them and the reference carries them too (§2m).
  let b (nm : String) (c : Nat) : List (String × List Nat) := if convBias then [(nm, [c])] else []
  [(s!"{p}eW", [mid,ic,1,1])] ++ b s!"{p}eb" mid ++ [(s!"{p}eg", [mid]), (s!"{p}ebt", [mid]),
   (s!"{p}dW", [mid,1,kd,kd])] ++ b s!"{p}db" mid ++ [(s!"{p}dg", [mid]), (s!"{p}dbt", [mid]),
   (s!"{p}zW1", [mid,r]), (s!"{p}zb1", [r]), (s!"{p}zW2", [r,mid]), (s!"{p}zb2", [mid]),
   (s!"{p}pW", [oc,mid,1,1])] ++ b s!"{p}pb" oc ++ [(s!"{p}pg", [oc]), (s!"{p}pbt", [oc])]

private def eSigNoExp (p : String) (ic oc r kd : Nat) (convBias : Bool) : List (String × List Nat) :=
  let b (nm : String) (c : Nat) : List (String × List Nat) := if convBias then [(nm, [c])] else []
  [(s!"{p}dW", [ic,1,kd,kd])] ++ b s!"{p}db" ic ++ [(s!"{p}dg", [ic]), (s!"{p}dbt", [ic]),
   (s!"{p}zW1", [ic,r]), (s!"{p}zb1", [r]), (s!"{p}zW2", [r,ic]), (s!"{p}zb2", [ic]),
   (s!"{p}pW", [oc,ic,1,1])] ++ b s!"{p}pb" oc ++ [(s!"{p}pg", [oc]), (s!"{p}pbt", [oc])]

/-- **Full 262-param EfficientNet-B0 signature**, func-arg order: stem(4) + b1 no-exp(12) +
    b2..b16 expand(15×16) + head(4) + dense(2) = 4+12+240+4+2 = 262 tensors. -/
def enetSig (nClasses : Nat) (convBias : Bool) : List (String × List Nat) :=
  (if convBias then [("sW", [32,3,3,3]), ("sb", [32])] else [("sW", [32,3,3,3])]) ++
  [("sg", [32]), ("sbt", [32])] ++
  eSigNoExp "b1" 32 16 8 3 convBias ++
  eSig "b2"  16  96  24  4 3 convBias ++ eSig "b3"  24 144  24  6 3 convBias ++
  eSig "b4"  24 144  40  6 5 convBias ++ eSig "b5"  40 240  40 10 5 convBias ++
  eSig "b6"  40 240  80 10 3 convBias ++ eSig "b7"  80 480  80 20 3 convBias ++ eSig "b8"  80 480  80 20 3 convBias ++
  eSig "b9"  80 480 112 20 5 convBias ++ eSig "b10" 112 672 112 28 5 convBias ++ eSig "b11" 112 672 112 28 5 convBias ++
  eSig "b12" 112 672 192 28 5 convBias ++ eSig "b13" 192 1152 192 48 5 convBias ++ eSig "b14" 192 1152 192 48 5 convBias ++
  eSig "b15" 192 1152 192 48 5 convBias ++ eSig "b16" 192 1152 320 48 3 convBias ++
  (if convBias then [("hW", [1280,320,1,1]), ("hb", [1280])] else [("hW", [1280,320,1,1])]) ++
  [("hg", [1280]), ("hbt", [1280])] ++
  [("Wd", [1280, nClasses]), ("bd", [nClasses])]

/-- Every channel width EfficientNet-B0 uses as a **conv** bias — stem 32, each block's `mid` and
    `oc` off the `[t,c,n,s,k]` table, head 1280. One list feeding all four `zeroBiasPrelude` calls,
    so a `convBias := false` render cannot declare the constants in one artifact and not another.

    ⚠ **NOT the SE widths.** SE's two 1×1 convs are followed by an ACTIVATION, not BN, so nothing
    absorbs their biases and the reference carries them (§2m); they stay real parameters and are
    never bound to a zero constant. The audit's rule — a rank-1 kind-2 param after a **rank-4**
    kernel — excludes them because SE's params are rank-2, which is why enet's +21,008 gap closed
    exactly on the first attempt. -/
def enetBiasWidths : List Nat :=
  [16, 24, 32, 40, 80, 96, 112, 144, 192, 240, 320, 480, 672, 1152, 1280]

-- ════════════════════════════════════════════════════════════════
-- § The whole-net renderer
-- ════════════════════════════════════════════════════════════════

/-- Every SSA name the EfficientNet-B0 forward produces, plus the 49-entry BN stat layout.
    `efficientnetFwd{,Eval}FaithfulV` return just `logits`; the train steps additionally consume the
    stem/head names and the 16 block records on the way back. -/
structure ENetFwd where
  code   : String            -- stem → 16 MBConv blocks → head → GAP → dense, in emission order
  stc    : String            -- stem conv out (= stem BN input)
  stn    : String            -- stem BN out (= stem swish pre-act)
  str    : String            -- stem swish out (= b1 input)
  blocks : Array EFwd        -- the 16 MBConv forwards, in forward order
  hc     : String            -- head 1×1 conv out (= head BN input)
  hn     : String            -- head BN out (= head swish pre-act)
  hr     : String            -- head swish out (= GAP input)
  gap    : String            -- global-average-pool out (= dense input)
  logits : String            -- dense out
  /-- The 49 BN layers as `(BN-input SSA, stat prefix, channels, spatial side)`, stem → blocks in
      forward order → head. Single source for the eval signature, the eval BN sites and the AdamW
      train step's returned batch statistics — see `EFwd.bns`. -/
  bns    : List (String × String × Nat × Nat)
  deriving Inhabited

set_option maxRecDepth 4000000 in
/-- **The full EfficientNet-B0 forward as `pretty` of the verified AST**, at the batched index
    `N := B`. 3×3/s2 stem (3→32, 224→112) → 16 MBConv blocks (the paper `[t,c,n,s]` table, SE in
    every block) → 1×1 head (320→1280) → GAP(7×7) → dense(1280→`nClasses`).

    `mode` picks the BN world and NOTHING else, which is the whole point of routing both forward
    artifacts through one chain: at `.train` every BN reduces its own batch statistics (`bnBatchF`)
    and this is exactly the prefix the train step differentiates; at `.eval` every BN consumes the
    driver's frozen running stats (the `bnEval` descriptor) and the net becomes
    class-batch-independent. -/
private def enetFwdChain (B nClasses : Nat) (mode : BnMode) (epsStr : String) (convBias : Bool)
    (sd : Bool := false) : StateM Nat ENetFwd := do
  -- `dp i` is `some i` exactly when block `i` is in `enetDropIdxs` AND stochastic depth is on —
  -- so the one list drives the signature and every call site, and the ramp index carried is the
  -- BLOCK index (see `enetDropIdxs`' note on why the site ordinal would be wrong).
  let dp : Nat → Option Nat := fun i => if sd && enetDropIdxs.contains i then some i else none
    -- ═══ stem: 3×3/s2 conv (3→32, 224→112) → bn → swish ═══
    let zx   : Vec (B * (3*224*224)) := fun _ => 0
    let zSk  : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let z32  : Vec 32 := fun _ => 0
    let z112F : Vec (B * (32*112*112)) := fun _ => 0
    let (cStc, nStc) ← pretty B (.batchOp (N := B) (.convStrided (h := 112) (w := 112) "%sW" (biasName convBias "%sb" 32) zSk z32) (.operand "%x" zx))
    let (cStn, nStn) ← bnSiteB B 32 112 112 mode epsStr "%sg" "%sbt" "stn" nStc
    let (cStr, nStr) ← pretty B (.batchOp (.swish) (.operand nStn z112F))
    -- ═══ forward: 16 MBConv blocks ═══
    let f1  ← eFwdNoExp   B 32      16 112 3  8 mode epsStr "b1"  nStr convBias
    let f2  ← eFwdStrided B 16  96  24  56 3  4 mode epsStr "b2"  f1.o convBias
    let f3  ← eFwd        B 24 144  24  56 3  6 mode epsStr "b3"  f2.o convBias (dp 2)
    let f4  ← eFwdStrided B 24 144  40  28 5  6 mode epsStr "b4"  f3.o convBias
    let f5  ← eFwd        B 40 240  40  28 5 10 mode epsStr "b5"  f4.o convBias (dp 4)
    let f6  ← eFwdStrided B 40 240  80  14 3 10 mode epsStr "b6"  f5.o convBias
    let f7  ← eFwd        B 80 480  80  14 3 20 mode epsStr "b7"  f6.o convBias (dp 6)
    let f8  ← eFwd        B 80 480  80  14 3 20 mode epsStr "b8"  f7.o convBias (dp 7)
    let f9  ← eFwdNoSkip  B 80 480 112  14 5 20 mode epsStr "b9"  f8.o convBias
    let f10 ← eFwd        B 112 672 112 14 5 28 mode epsStr "b10" f9.o convBias (dp 9)
    let f11 ← eFwd        B 112 672 112 14 5 28 mode epsStr "b11" f10.o convBias (dp 10)
    let f12 ← eFwdStrided B 112 672 192  7 5 28 mode epsStr "b12" f11.o convBias
    let f13 ← eFwd        B 192 1152 192 7 5 48 mode epsStr "b13" f12.o convBias (dp 12)
    let f14 ← eFwd        B 192 1152 192 7 5 48 mode epsStr "b14" f13.o convBias (dp 13)
    let f15 ← eFwd        B 192 1152 192 7 5 48 mode epsStr "b15" f14.o convBias (dp 14)
    let f16 ← eFwdNoSkip  B 192 1152 320 7 3 48 mode epsStr "b16" f15.o convBias
    -- ═══ head: 1×1 conv (320→1280) → bn → swish → GAP → dense ═══
    let z7F   : Vec (B * (320*7*7)) := fun _ => 0
    let zHk   : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
    let z1280 : Vec 1280 := fun _ => 0
    let zH7F  : Vec (B * (1280*7*7)) := fun _ => 0
    let z1280c : Vec (B * 1280) := fun _ => 0
    let zWd   : Mat 1280 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    let (cHc, nHc) ← pretty B (.batchOp (N := B) (.conv (h := 7) (w := 7) "%hW" (biasName convBias "%hb" 1280) zHk z1280) (.operand f16.o z7F))
    let (cHn, nHn) ← bnSiteB B 1280 7 7 mode epsStr "%hg" "%hbt" "hn" nHc
    let (cHr, nHr) ← pretty B (.batchOp (.swish) (.operand nHn zH7F))
    let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 1280) (h := 7) (w := 7)) (.operand nHr zH7F))
    let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC) (.operand nGap z1280c))
    pure { code := cStc ++ cStn ++ cStr ++
             f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++
             f8.code ++ f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++
             f15.code ++ f16.code ++ cHc ++ cHn ++ cHr ++ cGap ++ cLog,
           stc := nStc, stn := nStn, str := nStr,
           blocks := #[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16],
           hc := nHc, hn := nHn, hr := nHr, gap := nGap, logits := nLog,
           bns := (nStc, "stn", 32, 112) ::
             (f1.bns ++ f2.bns ++ f3.bns ++ f4.bns ++ f5.bns ++ f6.bns ++ f7.bns ++ f8.bns ++
              f9.bns ++ f10.bns ++ f11.bns ++ f12.bns ++ f13.bns ++ f14.bns ++ f15.bns ++
              f16.bns ++ [(nHc, "hn", 1280, 7)]) }

/-- The 263-input `@efficientnet_fwd` / 361-input `@efficientnet_fwd_eval` argument signature.
    The stat half is derived from the SAME `bns` the traversal built (never a parallel table), so
    the eval forward's slots cannot drift out of the order the driver packs `runningBnStats` in. -/
private def enetFwdSig (B nClasses : Nat) (mode : BnMode) (epsStr : String) (convBias : Bool)
    (sd : Bool := false) : String :=
  let F : ENetFwd := (enetFwdChain B nClasses mode epsStr convBias sd).run' 0
  let params := (enetSig nClasses convBias).map (fun (nm, d) => s!"%{nm}: {ty d}")
  let stats := if mode == .train then [] else
    F.bns.flatMap (fun (_, sp, c, _) => [s!"%{sp}mu: {ty [c]}", s!"%{sp}var: {ty [c]}"])
  -- ⚠ The drop inputs go LAST, after the BN stats, so adding them cannot shift an existing
  -- positional slot — the mnv2 `convBias` lesson (§2m): a parameter inserted mid-list captures
  -- an existing argument, and the driver walks this signature positionally.
  String.intercalate ", " ((s!"%x: {ty [B, 3*224*224]}") :: (params ++ stats)) ++ enetDropSig B sd

set_option maxRecDepth 4000000 in
/-- **`@efficientnet_fwd` rendered ENTIRELY from the verified AST** — 263 inputs (`%x` plus the 262
    params in `enetSig` order), returning logits `[B, nClasses]`. Shares `enetFwdChain` with the
    train step, so it is a byte-identical PREFIX of `efficientnet_train_step.mlir`, ending exactly
    where the loss begins. Replaces the hand-written emitter in `tests/TestEfficientNetFwd.lean`. -/
def efficientnetFwdFaithfulV (B nClasses : Nat) (epsStr : String) (convBias : Bool := false)
    (slug : String := "efficientnet") (sd : Bool := false) : String :=
  let F : ENetFwd := (enetFwdChain B nClasses .train epsStr convBias sd).run' 0
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd({enetFwdSig B nClasses .train epsStr convBias sd}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── EfficientNet-B0 forward: every line is pretty(verified AST node) ──\n" ++
  zeroBiasPrelude convBias enetBiasWidths ++ F.code ++
  s!"    return {F.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

set_option maxRecDepth 4000000 in
/-- **`@efficientnet_fwd_eval` rendered ENTIRELY from the verified AST** — the inference forward,
    every BN site consuming frozen per-channel running stats (the `bnEval` descriptor, `den` =
    `batchMap N bnPerChannelEvalTensor3`) instead of reducing statistics out of its activation.
    Same 262 params in the same order, plus the 98 stat inputs (49 BN layers × μ/var, interleaved
    per layer in `bnChannels` order): **361 inputs**.

    This is the eval partner of `efficientnet_adam_train_step`, whose returned batch μ/var the
    driver EMAs into exactly these slots — and both sides of that contract now come off one
    `bns` list rather than two independently-written ones. -/
def efficientnetFwdEvalFaithfulV (B nClasses : Nat) (epsStr : String) (convBias : Bool := false)
    (slug : String := "efficientnet") (sd : Bool := false) : String :=
  let F : ENetFwd := (enetFwdChain B nClasses .eval epsStr convBias sd).run' 0
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd_eval({enetFwdSig B nClasses .eval epsStr convBias sd}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── EfficientNet-B0 eval forward (running-stats BN): every line is pretty(verified AST node) ──\n" ++
  zeroBiasPrelude convBias enetBiasWidths ++ F.code ++
  s!"    return {F.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

set_option maxRecDepth 4000000 in
/-- **The whole-net forward + cotangent + backward traversal, SHARED by the SGD and AdamW renders.**

    Returns `(code, params, softmax, bns)`:

    * `code` — every emitted line, `pretty` of a verified `SHlo` node;
    * `params` — one SSA per parameter in `enetSig` order: the **updated param** at `adam := false`,
      the **un-fused gradient** at `adam := true`;
    * `softmax` — the softmax SSA, which the AdamW render's report-only `%loss` reads;
    * `bns` — the 49 BN layers as `(BN-input SSA, stat prefix, channels, spatial side)`, taken
      straight off `enetFwdChain` so the slots this step RETURNS are the slots
      `@efficientnet_fwd_eval` READS.

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
    built without it would fail the tie in a way that looks like a bug in the gradient ops.

    The forward half is `enetFwdChain` at `.train`, which `@efficientnet_fwd` also renders — so the
    forward this differentiates and the forward the driver evals with are one graph by
    construction, not by inspection (§2a). -/
private def enetBackAll (B nClasses : Nat) (epsStr lrStr : String) (adam : Bool)
    (smooth : Option (String × String × String) := none) (convBias : Bool := false)
    (sd : Bool := false) :
    StateM Nat (String × List String × String × List (String × String × Nat × Nat)) := do
    let F : ENetFwd ← enetFwdChain B nClasses .train epsStr convBias sd
    -- The SAME `dp` the forward used, from the SAME `enetDropIdxs`. The backward walks the
    -- blocks in REVERSE, so a carried counter would have to be reversed too — the easy place
    -- to be off by one. Both directions derive it from the literal block index instead.
    let dp : Nat → Option Nat := fun i => if sd && enetDropIdxs.contains i then some i else none
    let f1 := F.blocks[0]!; let f2 := F.blocks[1]!; let f3 := F.blocks[2]!; let f4 := F.blocks[3]!
    let f5 := F.blocks[4]!; let f6 := F.blocks[5]!; let f7 := F.blocks[6]!; let f8 := F.blocks[7]!
    let f9 := F.blocks[8]!; let f10 := F.blocks[9]!; let f11 := F.blocks[10]!
    let f12 := F.blocks[11]!; let f13 := F.blocks[12]!; let f14 := F.blocks[13]!
    let f15 := F.blocks[14]!; let f16 := F.blocks[15]!
    let nStc := F.stc; let nStn := F.stn; let nStr := F.str
    let nHc := F.hc; let nHn := F.hn; let nGap := F.gap; let nLog := F.logits
    let z32  : Vec 32 := fun _ => 0
    let z112B : Vec (B * (32*(112*112))) := fun _ => 0
    let z112F : Vec (B * (32*112*112)) := fun _ => 0
    let zSk  : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let zx   : Vec (B * (3*224*224)) := fun _ => 0
    let z1280 : Vec 1280 := fun _ => 0
    let zH7F  : Vec (B * (1280*7*7)) := fun _ => 0
    let zH7B  : Vec (B * (1280*(7*7))) := fun _ => 0
    let zHk   : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
    let z1280c : Vec (B * 1280) := fun _ => 0
    let zWd   : Mat 1280 nClasses := fun _ _ => 0
    -- `zNCb` is the ROW-indexed logit leaf (`rows = 1`, so `B·(1·nClasses)`) that the softmax /
    -- row-back / sub / smoothing nodes all sit at. The two head-dense parameter tails want the
    -- PLAIN batched index `B·nClasses` instead — same vector, non-defeq type — so `dnW`/`dnB`
    -- build that leaf themselves. See `zCc1` for the same wrinkle inside the SE gate.
    let zNCb  : Vec (B * (1 * nClasses)) := fun _ => 0
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
    let (cbh, nbh) ← if convBias then bnBt adam B 1280 7 7 "%hb" lrStr nHbn else pure ("", "")
    -- ═══ backward: 16 blocks reversed (cotangent threads from nHxb) ═══
    let b16 ← eBackNoSkip  adam B 192 1152 320 7 3 48 epsStr lrStr "b16" f15.o f16 nHxb convBias
    let b15 ← eBack        adam B 192 1152 192 7 5 48 epsStr lrStr "b15" f14.o f15 b16.dx convBias (dp 14)
    let b14 ← eBack        adam B 192 1152 192 7 5 48 epsStr lrStr "b14" f13.o f14 b15.dx convBias (dp 13)
    let b13 ← eBack        adam B 192 1152 192 7 5 48 epsStr lrStr "b13" f12.o f13 b14.dx convBias (dp 12)
    let b12 ← eBackStrided adam B 112 672 192  7 5 28 epsStr lrStr "b12" f11.o f12 b13.dx convBias
    let b11 ← eBack        adam B 112 672 112 14 5 28 epsStr lrStr "b11" f10.o f11 b12.dx convBias (dp 10)
    let b10 ← eBack        adam B 112 672 112 14 5 28 epsStr lrStr "b10" f9.o  f10 b11.dx convBias (dp 9)
    let b9  ← eBackNoSkip  adam B 80 480 112  14 5 20 epsStr lrStr "b9"  f8.o  f9  b10.dx convBias
    let b8  ← eBack        adam B 80 480  80  14 3 20 epsStr lrStr "b8"  f7.o  f8  b9.dx convBias (dp 7)
    let b7  ← eBack        adam B 80 480  80  14 3 20 epsStr lrStr "b7"  f6.o  f7  b8.dx convBias (dp 6)
    let b6  ← eBackStrided adam B 40 240  80  14 3 10 epsStr lrStr "b6"  f5.o  f6  b7.dx convBias
    let b5  ← eBack        adam B 40 240  40  28 5 10 epsStr lrStr "b5"  f4.o  f5  b6.dx convBias (dp 4)
    let b4  ← eBackStrided adam B 24 144  40  28 5  6 epsStr lrStr "b4"  f3.o  f4  b5.dx convBias
    let b3  ← eBack        adam B 24 144  24  56 3  6 epsStr lrStr "b3"  f2.o  f3  b4.dx convBias (dp 2)
    let b2  ← eBackStrided adam B 16  96  24  56 3  4 epsStr lrStr "b2"  f1.o  f2  b3.dx convBias
    let b1  ← eBackNoExp   adam B 32      16 112 3  8 epsStr lrStr "b1"  nStr  f1  b2.dx convBias
    -- ═══ stem backward: swish mask → bn back, then the 4 stem params (NO conv-back past %x) ═══
    let (cDsr, nDsr) ← pretty B (.swishBackB nStn (fun _ => 0) (.operand b1.dx z112F))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := B) (oc := 32) (h := 112) (w := 112) "%sg" nStc epsStr 0 z32 z112B (.operand nDsr z112B))
    let (csW, nsW) ← if adam then
        pretty B (.convStridedWeightGradB (N := B) (ic := 3) (oc := 32) (h := 112) (w := 112) "%x" z32 zx zSk (.operand nDsn z112F))
      else
        pretty B (.convStridedWeightSgdB (N := B) (ic := 3) (oc := 32) (h := 112) (w := 112) "%x" "%sW" lrStr z32 zx zSk 0 (.operand nDsn z112F))
    let (csb, nsb) ← if convBias then bnBt adam B 32 112 112 "%sb" lrStr nDsn else pure ("", "")
    let (csg, nsg) ← bnG adam B 32 112 112 "%sg" nStc epsStr lrStr nDsr
    let (cst, nst) ← bnBt adam B 32 112 112 "%sbt" lrStr nDsr
    -- ═══ assemble (params in func-arg order: stem, blocks fwd-order, head, dense) ═══
    -- `F.code` is exactly what `@efficientnet_fwd` renders, so the artifact is its byte-prefix.
    let fwdCode := F.code ++ cSm ++ cDy
    let bwdCode := cDgi ++ cWfc ++ cbfc ++ cDgp ++ cHsw ++ cHbn ++ cHxb ++ cgh ++ cth ++ cWh ++ cbh ++
      b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++ b10.code ++ b9.code ++
      b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDsr ++ cDsn ++ csW ++ csb ++ csg ++ cst
    -- Gated the same way `enetSig` is: a dropped bias must leave the list, not sit in it as the
    -- empty string `pure ("", "")` handed back (which renders `return %a, , %b` — malformed text
    -- that no arity `#guard` catches, because the list keeps its full length).
    let outNames : List String :=
      [nsW] ++ biasSlot convBias nsb ++ [nsg, nst] ++
      b1.names ++ b2.names ++ b3.names ++ b4.names ++ b5.names ++ b6.names ++ b7.names ++ b8.names ++
      b9.names ++ b10.names ++ b11.names ++ b12.names ++ b13.names ++ b14.names ++ b15.names ++
      b16.names ++ [nWh] ++ biasSlot convBias nbh ++ [ngh, nth] ++ [nWfc, nbfc]
    -- BN layers in the driver's order: stem → each block's (expand?, depthwise, project) → head.
    -- Taken straight off the forward chain, so the stat slots this train step RETURNS and the slots
    -- `@efficientnet_fwd_eval` READS are the same list — not two that happen to agree today.
    pure (fwdCode ++ bwdCode, outNames, nSm, F.bns)

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
    (funcName : String := "efficientnet_train_step") (convBias : Bool := false) : String :=
  let go : StateM Nat String := do
    let (code, outNames, _, _) ← enetBackAll B nClasses epsStr lrStr false none convBias
    let outTypes : List String := (enetSig nClasses convBias).map (fun p => ty p.2)
    pure <|
      "    // ── EfficientNet-B0 (16-MBConv) train step: every line is pretty(verified AST node) ──\n" ++
      zeroBiasPrelude convBias enetBiasWidths ++ code ++
      s!"    return {String.intercalate ", " outNames} : {String.intercalate ", " outTypes}\n"
  let sigList := enetSig nClasses convBias
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

/-- `(θ', b', s')` for one parameter under **RMSProp with momentum** — the `enetAdamOne` peer, and
    the same four-op composition `MobileNetV2RenderB.rmsOneM` uses:

    | reference (`jax/Jax/Codegen.lean`, `.rmsprop`) | emitted here |
    |---|---|
    | `grads = g + WD * p` | `momVNextF` at `(μ := wd, v := θ)` — `Proofs.momVNext_as_coupled_l2` |
    | `sq = RHO*s + (1-RHO)*g*g` | **`adamVNextF` at `β₂ := ρ`** — `Proofs.rmsSqNext_eq_adamVNext` |
    | `buf = MOMENTUM*b + g/sqrt(sq+EPS)` | `rmsBufNextF` — ε INSIDE the root |
    | `params = p - lr*buf` | `sgdParamF` on the buffer's SSA |

    ⚠ **EfficientNet's ε is 1e-3, where MobileNetV2's is 1.0** — and that is the placement's
    sensitive end: at a collapsed mean-square the textbook spelling takes a step **31.6×** larger
    (`Proofs.rmsBufNext_eps_placement_at_zero`). The JAX config says this in as many words —
    *"vanilla diverges/erodes at the paper LR, the TF form trains stably"* — and it is why the
    reference could drop its gradient clipping. **A green mnv2 tie does not license this render**;
    `rms-tie efficientnet` is its own gate.

    Slots: packed `[θ|m|v]` reused with `m` = momentum buffer, `v` = mean-square, so the interface
    is byte-identical to the AdamW render's apart from the entry name and `%bc1`/`%bc2` ride
    through unread. -/
private def enetRmsOne (B : Nat) (nm : String) (ds : List Nat) (gradSSA : String)
    (replicas : Nat) : StateM Nat (String × String × String × String) := do
  let n := ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let (arS, gAvg) := ViTRender.emitGradAllReduce gradSSA ds nm replicas
  let (cW, nW) ← pretty B (.momVNextF s!"%{nm}" "%wd" ds 0 z (.operand gAvg z))
  let gr : SHlo n := .operand nW z
  let (cS, nS) ← pretty B (.adamVNextF s!"%{nm}v" "%rho" "%orho" ds 0 z gr)
  let (cB, nB) ← pretty B (.rmsBufNextF s!"%{nm}v" s!"%{nm}m" "%rho" "%orho" "%mu" "%eps"
                    ds 0 0 0 z z gr)
  -- θ' threads b' by SSA NAME, not by re-nesting: `pretty` has no CSE (§4).
  let (cT, nT) ← pretty B (.sgdParamF s!"%{nm}" "%lr" ds 0 z (.operand nB z))
  pure (arS ++ cW ++ cS ++ cB ++ cT, nT, nB, nS)

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

/-- The driver's **variant slug** for a given `(B, replicas)`: the artifact is
    `verified_mlir/efficientnet_<variant>_train_step.mlir`, the entry point is
    `@efficientnet_<variant>_train_step`, and `LEAN_MLIR_VARIANT` selects it.

    All three must agree, or the shim refuses the call outright ("entry mismatch") rather than
    running the wrong graph — which is exactly what it did the first time R34's DP render kept the
    single-device name (§2b-quater). Deriving the name here is what stops it drifting from the
    `#eval` paths below; the `#guard`s at the bottom pin those literal paths against this function.

    `B = 32` is deliberately unsuffixed, so the two existing artifacts keep their names and bytes.
    Same convention as `r34AdamVariant`. -/
def enetAdamVariant (B replicas : Nat) (opt : OptKind := .adamw) (ema : Bool := false)
    (sd : Bool := false) : String :=
  -- ⚠ The `sd` marker TRAILS, and that is checked rather than assumed. The driver's two live
  -- predicates are `variant.startsWith "ema"` (4-region blob) and a `"rms"` SUBSTRING (mean-square
  -- init 1.0); a leading `sd` would break the first — `sdema` does not start with "ema", which is
  -- the `emarms` defect (`planning/ema.md`) reintroduced through a third axis. Trailing collides
  -- with neither. Three axes now share this string, so add the fourth only after re-running the
  -- predicate check.
  -- ⚠ The `ema` marker LEADS, because the driver keys its 4-region `[θ|m|v|ema]` blob layout off
  -- `variant.startsWith "ema"` — the same reverse-of-this-function reading it uses for `"rms"`.
  -- Optimizer and EMA are independent axes here (unlike ConvNeXt, which has only AdamW), so the
  -- name carries both: `emarms` is RMSProp + EMA, which IS the EfficientNet reference's recipe.
  (if ema then "ema" else "") ++
  (match opt with
   | .adamw   => if replicas ≤ 1 then "adam" else "adamdp"
   | .rmsprop => if replicas ≤ 1 then "rms"  else "rmsdp") ++
  (if B == 32 then "" else toString B) ++
  (if sd then "sd" else "")

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
    (alphaStr negAlphaKStr bStr : String) (replicas : Nat := 1)
    (convBias : Bool := false) (slug : String := "efficientnet")
    (opt : OptKind := .adamw) (ema : Bool := false) (sd : Bool := false) : String :=
  -- ⚠ `negAlphaKStr` is DERIVED from `nClasses` when empty. Passing −α/K as a string independent
  -- of K is the two-writers-for-one-fact shape that shipped a K=10 constant into R34's first
  -- ImageNet render ON THE GRADIENT PATH (§2k), and again into ConvNeXt's report-only loss
  -- (§2p, 2026-08-01) where a positive-signed copy hid from the grep that caught the cotangent.
  -- Empty ⇒ derived, so the K=1000 spelling cannot be got wrong; non-empty ⇒ honoured verbatim,
  -- which keeps every committed Imagenette artifact byte-identical.
  let negAlphaKStr := if negAlphaKStr.isEmpty then "-" ++ alphaOverK nClasses 0.1 else negAlphaKStr
  let sigList := enetSig nClasses convBias
  -- `go` hands back the BN channel counts alongside the body, so the argument signature is built
  -- from the SAME list the traversal walked. Deriving the 49 slots independently would be a second
  -- source for the stat layout — and a misaligned stat slot is silent: the arities still match and
  -- the wrong layer's statistics simply flow into the wrong `@efficientnet_fwd_eval` slot.
  let go : StateM Nat (String × List Nat) := do
    let (code, gradNames, nSm, bnList) ←
      enetBackAll B nClasses epsStr "0.0" true (some (alphaStr, negAlphaKStr, bStr)) convBias sd
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT. `den` is the
    --     same `bnMean`/`bnVar` `bnBatchF` normalises by, so these ARE the statistics the forward
    --     used rather than a separately-derived approximation. ═══
    let mut statCode := ""
    let mut statNames : List String := []
    let mut statTypes : List String := []
    for (xn, _sp, oc, hh) in bnList do
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
    let mut eN : List String := []
    for i in [0:sigList.length] do
      let (nm, ds) := sigList[i]!
      let (c, nT, nM, nV) ← match opt with
        | .adamw   => enetAdamOne B nm ds (gradNames[i]!) replicas
        | .rmsprop => enetRmsOne  B nm ds (gradNames[i]!) replicas
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]; mN := mN ++ [nM]; vN := vN ++ [nV]
      -- ▶ THE EMA SHADOW (`planning/ema.md`), emitted HERE rather than inside the two `*One`
      -- helpers, because it reads `nT` — the UPDATED parameter — and both tails produce one. A copy
      -- in each helper would be the double-writer disease one level down, in code (§2a-quater), and
      -- it would have to be kept in step across an optimizer axis that already exists.
      --
      -- `Proofs.adamMNext β₁ m g = β₁·m + (1−β₁)·g` IS the reference's `ema_update` at
      -- `(β₁ := d, m := ema, g := θ')`, so this is **no new op** — `adamMNextF_faithful` closes the
      -- denotation side by `rfl`. `%emad`/`%oemad` are ARGS, not constants: the reference's decay is
      -- time-varying, `d = min(decay, (1+t)/(10+t))`.
      --
      -- At `ema := false` no `pretty` call happens, so the fresh-name counter does not move and
      -- every committed artifact re-renders byte-identically — gate 1, for free.
      if ema then
        let n := ds.foldl (· * ·) 1
        let z : Vec n := fun _ => 0
        let (cE, nE) ← pretty B (.adamMNextF s!"%{nm}e" "%emad" "%oemad" ds 0 z (.operand nT z))
        adamCode := adamCode ++ cE
        eN := eN ++ [nE]
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
      -- ⚠ DERIVED, both of them. `%laKc` is α/K and was hardcoded at the K=10 value until
      -- 2026-08-02 — the THIRD copy of §2k's bug (R34's cotangent, ConvNeXt's loss, this). It
      -- survives a grep for the cotangent's `-0.010000` because this copy is positive-signed, and
      -- it is on no gradient path, so nothing but an implausible reported loss would show it.
      -- `%lomac` is (1−α), K-independent, derived anyway so α has one spelling here.
      s!"    %lomac = stablehlo.constant dense<{oneMinusAlpha 0.1}> : {ty [B]}\n" ++
      s!"    %laKc = stablehlo.constant dense<{alphaOverK nClasses 0.1}> : {ty [B]}\n" ++
      s!"    %llt1 = stablehlo.multiply %lomac, %lt1s : {ty [B]}\n" ++
      s!"    %llt2 = stablehlo.multiply %laKc, %llsr : {ty [B]}\n" ++
      s!"    %llpe = stablehlo.add %llt1, %llt2 : {ty [B]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [B]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
      s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
      s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
    let pTy := sigList.map (fun p => ty p.2)
    let retVals := thetaN ++ mN ++ vN ++ eN ++ ["%loss", "%bc1", "%bc2"]
                     ++ (if ema then ["%emad", "%oemad"] else []) ++ statNames
    let retTys := pTy ++ pTy ++ pTy ++ (if ema then pTy else [])
                     ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"]
                     ++ (if ema then ["tensor<f32>", "tensor<f32>"] else []) ++ statTypes
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
      (match opt with
       | .adamw => ""
       | .rmsprop =>
         "    // ── OPTIMIZER: RMSProp + momentum, TENSORFLOW flavour (EfficientNet's own:\n" ++
         "    //    jax/MainEfficientNetImagenet.lean). Per parameter, in this order:\n" ++
         "    //      g  <- g + wd*θ        COUPLED L2, BEFORE the accumulator  (momVNextF)\n" ++
         "    //      s' <- ρ*s + (1-ρ)*g²                                      (adamVNextF at ρ)\n" ++
         "    //      b' <- μ*b + g/sqrt(s' + ε)   ⚠ ε INSIDE the sqrt          (rmsBufNextF)\n" ++
         "    //      θ' <- θ - lr*b'                                           (sgdParamF)\n" ++
         "    //    ⚠ ε = 1e-3 here, against MobileNetV2's 1.0 — this is the SENSITIVE end of the\n" ++
         "    //    placement: textbook g/(sqrt(s')+ε) takes a ~31.6x larger step at a collapsed\n" ++
         "    //    mean-square, which is what the reference means by \"vanilla diverges at the\n" ++
         "    //    paper LR\" and why it carries no gradient clipping.\n" ++
         "    //    Packed [θ|m|v] reused with m = momentum buffer, v = mean-square; %bc1/%bc2 are\n" ++
         "    //    Adam bias corrections, unread here and passed through unchanged.\n" ++
         "    //    ⚠ The mean-square must be INITIALISED TO 1.0, not 0 — part of the recipe.\n") ++
      zeroBiasPrelude convBias enetBiasWidths ++ code ++ statCode ++
      (match opt with | .adamw => enetAdamConsts | .rmsprop => rmsConstsBlock enetRmsHyper) ++
      adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n",
      bnList.map (fun t => t.2.2.1))
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
  let eSig := String.intercalate ", " (sigList.map (fun (n, ds) => s!"%{n}e: {ty ds}"))
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++
    (if ema then ", " ++ eSig else "") ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>" ++
    (if ema then ", %emad: tensor<f32>, %oemad: tensor<f32>" else "") ++ ", " ++ statSig ++
    -- ⚠ The drop scales go LAST, after the BN stats and before `%onehot` is appended, matching
    -- `enetFwdSig`'s placement — inserted mid-list they would capture an existing positional slot,
    -- which is the mnv2 `convBias` failure (§2m) and is silent until the driver mis-walks the blob.
    enetDropSig B sd ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (fun p => ty p.2)
  let outSig := String.intercalate ", "
    (pTy ++ pTy ++ pTy ++ (if ema then pTy else [])
     ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"]
     ++ (if ema then ["tensor<f32>", "tensor<f32>"] else [])
     ++ bnOc.flatMap (fun oc => [ty [oc], ty [oc]]))
  -- The entry name must track the driver's `{slug}_{variant}_train_step` convention, or the shim
  -- refuses the call ("entry mismatch"). `enetAdamVariant` is the single source for the name, the
  -- artifact path and `LEAN_MLIR_VARIANT`.
  let fname := s!"{slug}_{enetAdamVariant B replicas opt ema sd}_train_step"
  "module @m {\n" ++
  s!"  func.func @{fname}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- Regenerate `verified_mlir/efficientnet_train_step.mlir` (what MainEfficientNetVerified trains on)
-- from the faithful renderer: the FULL 16-MBConv B0 net (262 params). B=32, nClasses=10, ε=1e-5.
#eval IO.FS.writeFile "verified_mlir/efficientnet_train_step.mlir"
  (Proofs.StableHLO.efficientnetTrainStepFaithfulV 32 10 "1.0e-5" "0.05" "efficientnet_train_step")

-- Regenerate `verified_mlir/efficientnet_fwd.mlir` (the SGD driver's eval forward) and
-- `verified_mlir/efficientnet_fwd_eval.mlir` (what the AdamW driver evals with, once the running
-- stats are threaded) from the SAME `enetFwdChain` the train steps differentiate. These replace the
-- hand-written emitter in `tests/TestEfficientNetFwd.lean`; that copy is retired to an
-- `iree-compile` smoke over the committed bytes.
--
-- The `.train` artifact is a byte-identical PREFIX of `efficientnet_train_step.mlir`, which
-- `scripts/regen_verified_mlir.sh check` audits — the check that caught ResNet-34 training a
-- per-example-BN net and scoring it with batch statistics (§2a). EfficientNet was never skewed that
-- way (both sides were already batch-BN), and the prefix audit is what now keeps it that way.
#eval IO.FS.writeFile "verified_mlir/efficientnet_fwd.mlir"
  (Proofs.StableHLO.efficientnetFwdFaithfulV 32 10 "1.0e-5")

#eval IO.FS.writeFile "verified_mlir/efficientnet_fwd_eval.mlir"
  (Proofs.StableHLO.efficientnetFwdEvalFaithfulV 32 10 "1.0e-5")

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

-- The **bs128** pair (handoff §2e-quater), single-device and 2-replica. `B` is a true parameter of
-- the renderer, so this is the whole change: the graph structure is identical and only the tensor
-- dimensions and the mean-CE divisor (32.0 → 128.0) move. At 2 replicas this is a GLOBAL batch of
-- 256, the batch ImageNet wants.
--
-- They render to their OWN paths, so the artifacts the trainer runs today are untouched and the
-- §2e tie/DP-gate baselines stay valid. Select with `LEAN_MLIR_VARIANT=adam128` / `adamdp128` and
-- `LEAN_MLIR_BATCH=128`. **The eval forwards are still bs32**, so train with `LEAN_MLIR_SKIP_EVAL=1`
-- or re-render them — the same caveat §2d.1 carries for R34's bs256.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adam128_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 128 10 "1.0e-5"
    "0.100000" "-0.010000" "128.0")

#eval IO.FS.writeFile "verified_mlir/efficientnet_adamdp128_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 128 10 "1.0e-5"
    "0.100000" "-0.010000" "128.0" 2)

-- ── EfficientNet-B0 on FULL 1000-class ImageNet, slug `enetin` — 2026-08-02 ────────────────────
-- The EfficientNet peer of `resnet34in_*` (§2k), `vitin_*` and `cnxin_*` (§2p). No renderer
-- restructuring was needed — `B` and `nClasses` were already parameters; what this change added is
-- a `slug` (the three entry names were baked) and the derived −α/K above.
--
-- ⚠ **The slug is load-bearing and EfficientNet is the case where it bites hardest**: it is the
-- only net here with BOTH a `_fwd` and a `_fwd_eval` artifact, and neither carries a variant in its
-- path. A 1000-class forward emitted under the `efficientnet` slug would silently overwrite the
-- 10-class pair that the 88.20% Imagenette run, the §2g prefix audit and `fwd-tie efficientnet
-- --eval` all depend on.
--
-- Batch **64 per device × 4 replicas = global 256**, which is
-- `efficientNetB0ImagenetConfig.batchSize`. Matching the reference's global batch is what makes the
-- two runs a comparable pair rather than two experiments — the same reasoning as R34's `momdp64`.
--
-- ⚠ `enetAdamVariant B replicas` encodes the PER-DEVICE batch and NOT the replica count, so
-- `adamdp64` would name both a 2-replica and a 4-replica render at B=64. Only the 4-replica one is
-- emitted here, so nothing collides today; anyone adding the 2-replica peer must rename first.
#eval IO.FS.writeFile "verified_mlir/enetin_fwd.mlir"
  (Proofs.StableHLO.efficientnetFwdFaithfulV 64 1000 "1.0e-5" false "enetin")
#eval IO.FS.writeFile "verified_mlir/enetin_fwd_eval.mlir"
  (Proofs.StableHLO.efficientnetFwdEvalFaithfulV 64 1000 "1.0e-5" false "enetin")
#eval IO.FS.writeFile "verified_mlir/enetin_adam64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 1 false "enetin")
#eval IO.FS.writeFile "verified_mlir/enetin_adamdp64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 4 false "enetin")

-- ── ▶ RMSProp: the optimizer the EfficientNet reference ACTUALLY USES ─────────────────────────
-- `planning/recipe_gaps.md` §2: RMSProp is one of TWO gaps between this net and the reference's
-- **72.31%** (the other being dropPath + EMA, both driver/architectural). ρ = μ = 0.9,
-- **ε = 1e-3**, wd = 1e-5 — `Proofs.StableHLO.enetRmsHyper`.
--
-- ⚠ **ε = 1e-3 is the SENSITIVE end of the ε-placement difference**, unlike MobileNetV2's ε = 1.0.
-- At a collapsed mean-square the textbook spelling steps 31.6× larger, which is exactly what the
-- reference config means by *"vanilla diverges/erodes at the paper LR, the TF form trains stably"*
-- and why its `gradClipNorm` is 0. So this net is where the placement is load-bearing, and it gets
-- its own numeric gate — `rms-tie efficientnet` — rather than inheriting mnv2's.
--
-- The Imagenette-shape render below differs from `efficientnet_adam_train_step` in EXACTLY ONE
-- thing, the optimizer: same B, K, ε, α, −α/K and ÷B literals. That is what makes a tie against it
-- attributable (§2m — a candidate that differs in two ways cannot license either).
#eval IO.FS.writeFile "verified_mlir/efficientnet_rms_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" 1 false "efficientnet" .rmsprop)

-- ── ▶ RMSProp **+ EMA** — this net's ACTUAL reference recipe (`planning/ema.md`) ────────────────
-- `efficientNetB0ImagenetConfig` is RMSProp + exp-decay + **EMA (decay 0.9999)** + dropPath, and
-- its 72.31% is the EMA shadow's number. RMSProp and its schedule landed in recipe_gaps v1.2 and
-- its driver half; this is the third of the four, leaving only stochastic depth.
--
-- ⚠ The variant is `emarms`, not `ema`: optimizer and EMA are INDEPENDENT axes on this net (unlike
-- ConvNeXt, which has only AdamW), so the name carries both — and the `ema` marker LEADS because
-- the driver keys its 4-region `[θ|m|v|ema]` layout off the prefix.
--
-- ⚠ EfficientNet is the first EMA net with **BatchNorm**, and the reference shadows the BN running
-- buffers too (`ema_bn`): eval pairs EMA weights with EMA-LAGGED statistics, because pairing them
-- with LIVE stats is the mismatch its own comment says "blows up early eval". That half is
-- driver-side and nearly free — `runningBnStats` already lives on the host and is already EMA'd
-- there — so it is NOT in this render. Nothing here emits a BN shadow; the graph's stat slots are
-- unchanged.
#eval IO.FS.writeFile "verified_mlir/efficientnet_emarms_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" 1 false "efficientnet" .rmsprop (ema := true))

-- The ImageNet peers: batch 64 × 4 replicas = global 256 = `efficientNetB0ImagenetConfig.batchSize`,
-- and −α/K DERIVED from nClasses (empty string), so the emitted shift is -0.000100 at K = 1000.
--
-- ⚠ THE DRIVER STILL OWES TWO THINGS, neither a render change: the mean-square slot must be
-- INITIALISED TO 1.0 (TF's convention — a zero init is not a crash, it is a different and much
-- larger first step), and the LR schedule must be exponential 0.97/epoch rather than cosine.
-- Until both land these are correct renders of the right optimizer, not a matched pair.
#eval IO.FS.writeFile "verified_mlir/enetin_rms64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 1 false "enetin" .rmsprop)
#eval IO.FS.writeFile "verified_mlir/enetin_rmsdp64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 4 false "enetin" .rmsprop)

-- Pin the two literal artifact paths above against the name the renderer actually emits. If a
-- variant is renamed this fails at `lake build` instead of at run time as an "entry mismatch".
#guard Proofs.StableHLO.enetAdamVariant 32 1 == "adam"
#guard Proofs.StableHLO.enetAdamVariant 32 2 == "adamdp"
#guard Proofs.StableHLO.enetAdamVariant 128 1 == "adam128"
#guard Proofs.StableHLO.enetAdamVariant 128 2 == "adamdp128"
-- The RMSProp peers. Distinct slugs from the AdamW ones is the point: rendering the other
-- optimizer must never overwrite the artifact the AdamW trainer runs (§2a's last-writer-wins race).
#guard Proofs.StableHLO.enetAdamVariant 32 1 .rmsprop == "rms"
#guard Proofs.StableHLO.enetAdamVariant 64 1 .rmsprop == "rms64"
#guard Proofs.StableHLO.enetAdamVariant 64 4 .rmsprop == "rmsdp64"
#guard Proofs.StableHLO.enetAdamVariant 64 1 .adamw   == "adam64"
-- The EMA peers. The marker LEADS so the driver's `startsWith "ema"` finds it whichever
-- optimizer it is paired with, and `emarms` — RMSProp + EMA — is this net's reference recipe.
#guard Proofs.StableHLO.enetAdamVariant 32 1 .rmsprop true == "emarms"
#guard Proofs.StableHLO.enetAdamVariant 32 1 .adamw   true == "emaadam"
#guard Proofs.StableHLO.enetAdamVariant 64 4 .rmsprop true == "emarmsdp64"
-- ...and OFF it is byte-identical to what it always was, which is what keeps gate 1 free.
#guard Proofs.StableHLO.enetAdamVariant 32 1 .rmsprop false == "rms"

-- §2m: both arities pinned, so dropping the conv biases cannot silently change the render that
-- ships. 262 − 49 = 213; the SE biases (`zb1`/`zb2`) are NOT among the 49, because those convs are
-- followed by an activation rather than BN and the reference carries them too.
#guard (Proofs.StableHLO.enetSig 10 true).length == 262
#guard (Proofs.StableHLO.enetSig 10 false).length == 213

-- ── ▶ STOCHASTIC DEPTH (`planning/stochastic_depth.md`), selected by `LEAN_MLIR_VARIANT=adamsd` ──
-- EfficientNet-B0 is the net this landed on FIRST, and the reason inverts the spec's own
-- recommendation. `stochastic_depth.md` §8 recommends ConvNeXt as "the cheapest"; measured, it is
-- not, and the axis it was scoped on was the wrong one:
--
--   renderer            batched forms   per-example forms
--   ResNet34RenderB              36            0
--   MobileNetV2RenderB           53            0
--   EfficientNetRender           45            0     ← at N := B, so the op drops straight in
--   ConvNeXtRender                2 (N := 1)  10     ← per-example
--   ViTRender                     2           13     ← per-example
--
-- A per-example mask CANNOT be expressed honestly at the per-example index: `den` at index `n`
-- describes one example, so the mask becomes §4's descriptor trap ("a descriptor may carry only
-- batch-INVARIANT data"). ConvNeXt and ViT therefore need the §2b batched-index move FIRST — the
-- step the handoff calls the most expensive and most badly mis-estimated in the whole thread.
-- EfficientNet needs none of it. This is §2f's lesson one net over: scope by the INDEX CONVENTION,
-- not by op count.
--
-- ⚠ 9 sites, not 16, and the ramp index is the BLOCK index — see `enetDropIdxs`.
-- ⚠ The scale is a graph INPUT with `1/keep_i` FOLDED IN by the driver, never `stablehlo.rng` and
-- never a baked constant. `Proofs.dropPath`'s note has the argument; the short version is that a
-- baked `1/keep_i` and §3's "the forward emits the sites too" cannot both hold, because a ones
-- mask would then compute `x/keep_i` rather than `x`, and the reference is explicit that eval
-- returns the branch untouched.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adamsd_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" (sd := true))

-- The forward peers. ⚠ These exist so the SD variant has its OWN `forward ⊂ train-step` pair —
-- `stochastic_depth.md` §3's design, and the reason is that the prefix audit is one of the two
-- load-bearing structural gates in the repo (it caught `resnet34_fwd` and `mobilenetv2_fwd`
-- scoring nets they had not trained). The alternative — let the SD trainer eval through the
-- drop-free `efficientnet_fwd` — is what the reference literally does, but it would leave the SD
-- train step with no prefix partner at all, i.e. spend the gate rather than pay 9 dead multiplies.
-- At eval the driver supplies an all-ones scale, so these compute the identity EXACTLY
-- (`Proofs.dropPath_ones_id`, and `1 * x = x` is exact in IEEE).
#eval IO.FS.writeFile "verified_mlir/efficientnet_sd_fwd.mlir"
  (Proofs.StableHLO.efficientnetFwdFaithfulV 32 10 "1.0e-5" false "efficientnet_sd" (sd := true))

#eval IO.FS.writeFile "verified_mlir/efficientnet_sd_fwd_eval.mlir"
  (Proofs.StableHLO.efficientnetFwdEvalFaithfulV 32 10 "1.0e-5" false "efficientnet_sd" (sd := true))
