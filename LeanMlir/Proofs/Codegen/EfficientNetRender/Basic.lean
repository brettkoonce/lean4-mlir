import LeanMlir.Proofs.Codegen.SyncBnSites
import LeanMlir.Proofs.Codegen.RenderKit

/-! # EfficientNet-B0 train step rendered ENTIRELY from the verified AST (batched)

The Chapter-7 peer of `MobileNetV2RenderB`, for the committed full-16-MBConv EfficientNet-B0
(262 params, the real `[t,c,n,s,k]` B0 spec). EfficientNet emits **true batch-norm**, which
couples the batch — so the whole net lives at the **batched index** `N·(c·h·w)`
(`StableHLO.batchOp`/`bnBatchF`/the batched backward + param-SGD ops).

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
passed; the emitted `lrStr`/`epsStr` literals carry the real values.

## References

- Tan & Le 2019, *EfficientNet: Rethinking Model Scaling for CNNs*. <https://arxiv.org/abs/1905.11946> -/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-! ## The batched index: why every `N` below is `B`, not 1

Building the graph at the **batch-unit** index `N = 1` and letting `pretty B` supply the real
batch would be sound for the ops where the batch is a *parallel* index — `batchOp`'s `den` is
`batchMap N (denOp op)`, which at `N = 1` is the per-example op, exactly what the emit applies
across the batch. It is **not** sound for the ops where the batch is a *reduction* axis:
`bnBatchF` and `bnBatchBack` reduce μ/var over `[0,2,3]`, and the whole `*SgdB` param family
(`bnGammaSgdB`, `bnBetaSgdB`, `dense{Weight,Bias}SgdB`, `conv{,Strided}WeightSgdB`, `depthwise{,Strided}WeightSgdB`) sums the
per-example gradient over `Fin N`. At `N = 1` each of those `den`s would describe a ONE-EXAMPLE
function while the emitted text reduces over all `B`.

So the whole graph sits at `N := B`, where those `den`s are honest. The obstacle there is not the
batch-coupled ops (their emitters discard `N` and use `B`, so they render identically at any
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

The emitted text does not depend on `N`, and it cannot witness the `den` side — the render is
value-independent, so a descriptor holding the wrong saved activation would render exactly the
same bytes. That half is carried by the `rfl` faithfulness theorems in `StableHLO.Basic`. -/

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
  -- SYNC-BN (`replicas > 1`): each BN site's all-reduced packed `[μ ‖ σ²]`, read by its
  -- backward, its γ gradient and the handed-back running stats; `bnSt` is aligned with `bns`.
  -- `""` / `[]` at one replica.
  stE : String := ""
  stD : String := ""
  stP : String := ""
  bnSt : List String := []
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
[`tests/TestBatchedEmitTie.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestBatchedEmitTie.lean) checks the emit side of the same statement: each `*GradB` render is
a byte-PREFIX of its `*SgdB` peer's, the tail being exactly the const-lr / multiply / subtract.

These six helpers are what let ONE backward traversal serve both renders. The alternative — a second
copy of the 16-MBConv backward for AdamW — is the double-writer disease one level down, in code
rather than in artifacts: two emitters for one step that can compute different functions.

`lrStr` is threaded but unused in `adam` mode: the AdamW render's learning rate is the runtime
`%lr` argument, not a baked literal. The placeholder values are irrelevant either way — `skel`
erases them, so the render is value-independent. -/

/-- BN γ. -/
private def bnG (adam : Bool) (B oc hh ww : Nat) (gName vName epsStr lrStr dy : String)
    (sync : Bool := false) (st : String := "") :
    StateM Proofs.StableHLO.EmitS (String × String) := do
  let zc : Vec oc := fun _ => 0
  let zb : Vec (B * (oc * (hh * ww))) := fun _ => 0
  if adam then
    bnGammaSite B oc hh ww sync epsStr vName dy st
  else
    pretty B (.bnGammaSgdB (N := B) (oc := oc) (h := hh) (w := ww) gName vName epsStr lrStr 0 zc zb 0
                (.operand dy zb))

/-- BN β — and, because `Σ_{batch,spatial} dy` is exactly a conv bias gradient, also every conv
    bias in this net (`%eb`, `%db`, `%pb`, `%sb`). That reuse is why EfficientNet needed no
    depthwise BIAS gradient op: its depthwise convs are followed by BN, so the bias is folded. -/
private def bnBt (adam : Bool) (B oc hh ww : Nat) (bName lrStr dy : String) : StateM Proofs.StableHLO.EmitS (String × String) := do
  let zc : Vec oc := fun _ => 0
  let zb : Vec (B * (oc * (hh * ww))) := fun _ => 0
  if adam then
    pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand dy zb))
  else
    pretty B (.bnBetaSgdB (N := B) (oc := oc) (h := hh) (w := ww) bName lrStr zc 0 (.operand dy zb))

/-- 1×1 conv weight (expand / project / head — every non-depthwise conv here but the stem). -/
private def convW1 (adam : Bool) (B ic oc hh ww : Nat) (xName wName lrStr dy : String)
    -- bf16 reaches ONLY the `adam` branch's un-fused `*GradB`. The `else` branch is the
    -- fused plain-SGD tail (`*SgdB`), which no bf16 artifact renders — it stays f32.
    (bf16 : Bool := false) : StateM Proofs.StableHLO.EmitS (String × String) := do
  let zb : Vec oc := fun _ => 0
  let zx : Vec (B * (ic * hh * ww)) := fun _ => 0
  let zk : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zd : Vec (B * (oc * hh * ww)) := fun _ => 0
  if adam then
    pretty B (.convWeightGradBAt bf16 (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) zrnd xName zb zx zk
                (.operand dy zd))
  else
    pretty B (.convWeightSgdB (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) xName wName lrStr
                zb zx zk 0 (.operand dy zd))

/-- Depthwise `kd × kd` weight, stride 1. -/
private def dwW (adam : Bool) (B c hh ww kd : Nat) (xName wName lrStr dy : String)
    -- bf16 reaches ONLY the `adam` branch's un-fused `*GradB`. The `else` branch is the
    -- fused plain-SGD tail (`*SgdB`), which no bf16 artifact renders — it stays f32.
    (bf16 : Bool := false) : StateM Proofs.StableHLO.EmitS (String × String) := do
  let zb : Vec c := fun _ => 0
  let zx : Vec (B * (c * hh * ww)) := fun _ => 0
  let zk : DepthwiseKernel c kd kd := fun _ _ _ => 0
  let zd : Vec (B * (c * hh * ww)) := fun _ => 0
  if adam then
    pretty B (.depthwiseWeightGradBAt bf16 (N := B) (c := c) (h := hh) (w := ww) zrnd xName zb zx zk
                (.operand dy zd))
  else
    pretty B (.depthwiseWeightSgdB (N := B) (c := c) (h := hh) (w := ww) xName wName lrStr
                zb zx zk 0 (.operand dy zd))

/-- Depthwise `kd × kd` weight, stride 2 (input at `2hh × 2ww`). -/
private def dwWS (adam : Bool) (B c hh ww kd : Nat) (xName wName lrStr dy : String)
    -- bf16 reaches ONLY the `adam` branch's un-fused `*GradB`. The `else` branch is the
    -- fused plain-SGD tail (`*SgdB`), which no bf16 artifact renders — it stays f32.
    (bf16 : Bool := false) : StateM Proofs.StableHLO.EmitS (String × String) := do
  let zb : Vec c := fun _ => 0
  let zx : Vec (B * (c * (2*hh) * (2*ww))) := fun _ => 0
  let zk : DepthwiseKernel c kd kd := fun _ _ _ => 0
  let zd : Vec (B * (c * hh * ww)) := fun _ => 0
  if adam then
    pretty B (.depthwiseStridedWeightGradBAt bf16 (N := B) (c := c) (h := hh) (w := ww) zrnd xName zb zx zk
                (.operand dy zd))
  else
    pretty B (.depthwiseStridedWeightSgdB (N := B) (c := c) (h := hh) (w := ww) xName wName lrStr
                zb zx zk 0 (.operand dy zd))

/-- Dense weight (the two SE gate matrices and the classifier head). -/
private def dnW (adam : Bool) (B a c : Nat) (xName wName lrStr dy : String) : StateM Proofs.StableHLO.EmitS (String × String) := do
  let zx : Vec (B * a) := fun _ => 0
  let zW : Mat a c := fun _ _ => 0
  let zd : Vec (B * c) := fun _ => 0
  if adam then
    pretty B (.denseWeightGradB (N := B) (a := a) (c := c) xName zx (.operand dy zd))
  else
    pretty B (.denseWeightSgdB (N := B) (a := a) (c := c) xName wName lrStr zx zW 0 (.operand dy zd))

/-- Dense bias. -/
private def dnB (adam : Bool) (B c : Nat) (bName lrStr dy : String) : StateM Proofs.StableHLO.EmitS (String × String) := do
  let zb : Vec c := fun _ => 0
  let zd : Vec (B * c) := fun _ => 0
  if adam then
    pretty B (.denseBiasGradB (N := B) (c := c) (.operand dy zd))
  else
    pretty B (.denseBiasSgdB (N := B) (c := c) bName lrStr zb 0 (.operand dy zd))

-- ════════════════════════════════════════════════════════════════
-- § Squeeze-excite forward (un-fused, for activation-saving) + backward (param grads)
-- ════════════════════════════════════════════════════════════════

/-- **SE forward** on `c` channels at `hh×ww`, reduce dim `r`: one fused `seBlock`, whose text
    computes the squeeze (GAP), reduce dense (`W₁`), swish and excite dense (`W₂`) on the way to the
    gate. Their SSA names (`seBlockSavedNames`) are the activations the SE backward reads, so the
    forward computes the SE once — it is `pretty` of the typed graph's `seBlock` node — rather than
    a second, un-fused copy beside it.
    Returns `(code, s, e1, z, e2, seOut)`. -/
def seFwd (B c hh r : Nat) (p drName : String) :
    StateM Proofs.StableHLO.EmitS (String × String × String × String × String × String) := do
  let ww := hh
  let zChw : Vec (B * (c * hh * ww)) := fun _ => 0
  let zW1  : Mat c r := fun _ _ => 0
  let zb1  : Vec r := fun _ => 0
  let zW2  : Mat r c := fun _ _ => 0
  let zb2  : Vec c := fun _ => 0
  let (k0, _) ← get
  let (cSe, nSe) ← pretty B (.batchOp (N := B)
      (.seBlock (h := hh) (w := ww) s!"%{p}zW1" s!"%{p}zb1" s!"%{p}zW2" s!"%{p}zb2" zW1 zb1 zW2 zb2)
      (.operand drName zChw))
  let (nS, nE1, nZ, nE2) := seBlockSavedNames k0
  pure (cSe, nS, nE1, nZ, nE2, nSe)

/-- **SE backward + 4 param SGD ops.** `dx` (SE input cot) via the fused `seBackBatched`; the SE dense
    param grads via `seReduceB → sigmoidBack(e2) → {W₂} → denseRowBack(W₂) → swishBack(e1) → {W₁}`.
    Returns `(code, dx, [zW1, zb1, zW2, zb2 updated names])`. -/
private def seBack (adam : Bool) (B c hh r : Nat)
    (lrStr p drName sName e1Name zName e2Name seCot : String) :
    StateM Proofs.StableHLO.EmitS (String × String × List String) := do
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
-- ── STOCHASTIC DEPTH ───────────────────────────────────────
-- EfficientNet-B0 has 16 MBConv blocks, and the drop fires on the 9 that carry a skip.
--
-- THE RAMP INDEX IS THE BLOCK INDEX, NOT THE SITE ORDINAL — and getting that wrong is the
-- expensive silent bug here. The reference advances `dbi` on EVERY block
-- (`if cfg.dropPath > 0 then dbi := dbi + 1`, unconditional) while the drop only FIRES inside the
-- skip guard `residual.shape == x.shape and stride == 1`. So `keep_i = 1 − dropRate·i/(16−1)` with
-- `i` the BLOCK index, and the denominator is **15**, not 8. Re-indexing by site would give nine
-- evenly-spaced keeps instead of the reference's nine unevenly-spaced ones: it compiles, runs,
-- descends, and trains a different objective.
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
    undeclared `%dp<i>` (rejected by the lowerer). Neither is silent. -/
def enetDropIdxs : List Nat := [2, 4, 6, 7, 9, 10, 12, 13, 14]

/-- The number of per-example drop-path scale inputs a stochastic-depth render takes. -/
def enetDropSites : Nat := enetDropIdxs.length

#guard enetDropTotal == 16
#guard enetDropSites == 9
-- Every site is a real block, and the list is strictly increasing (= signature order).
#guard enetDropIdxs.all (· < enetDropTotal)
#guard (enetDropIdxs.zip (enetDropIdxs.drop 1)).all (fun (a, b) => a < b)

-- `dpName` lives in `StableHLO.Basic` (beside `fresh`), shared with ConvNeXt's SD render:
-- `scripts/probes/misplace_drop_sites.py` matches `%dp\d+` textually, so a second definition would
-- put a committed shell script in the middle of a two-writer drift.

/-- The `%dp<i>: tensor<Bxf32>` inputs, appended to a render's signature when stochastic depth is
    on. Empty when off. -/
def enetDropSig (B : Nat) (sd : Bool) : String := dropMaskSig B sd enetDropIdxs

-- ── CLASSIFIER DROPOUT ─────────────────────────────────────────────
-- `efficientNetB0ImagenetConfig` sets `dropout := 0.2` (`jax/MainEfficientNetImagenet.lean`).
-- Stochastic depth and classifier dropout are different regularisers, at different places, with
-- different mask ranks: covering one does not cover the other.
--
-- ONE SITE, AND IT IS NOT A RAMP. The reference applies it in the `.dense` case
-- (`emitForward`'s classifier dropout in `jax/Jax/Codegen.lean`), immediately before the single classifier — so unlike stochastic
-- depth there is no per-block schedule, no `totalDrop` denominator, and therefore no
-- silent-constant ramp bug is even spellable here. What replaces that risk is the mask
-- RANK (`Proofs.dropout` vs `Proofs.dropPath`) and the weight-gradient operand below.
--
-- THE WIDTH IS THE HEAD'S, NOT THE CLASS COUNT. Dropout sits between GAP and the dense, so the
-- mask is `tensor<B×1280>` at every `nClasses` — Imagenette and ImageNet renders take the SAME
-- mask shape, which is why `enetDropoutSig` does not read `nClasses` and must not be "fixed" to.

/-- The classifier's input width — EfficientNet-B0's head channel count, i.e. the GAP output and
    hence the dropout mask's per-example width. Independent of `nClasses`. -/
def enetHeadWidth : Nat := 1280

/-- The `%do: tensor<B×1280xf32>` input, appended when classifier dropout is on. Empty when off,
    which is what keeps the inertness gate byte-identical.

    It goes **after** `enetDropSig`, i.e. dead last in every signature. Two independent reasons,
    and the second is the one that bites: a parameter inserted mid-list captures an existing
    positional slot and the driver walks these signatures
    positionally; and the drop-mask tail is what the DP shim shards by COUNT from the end
    (`n_shard_tail`), so a per-example input placed before them would be counted as one of them. -/
def enetDropoutSig (B : Nat) (cd : Bool) : String :=
  if cd then s!", {doName}: {ty [B, enetHeadWidth]}" else ""

-- § MBConv forward emitters (stride-1 expand / no-skip expand / strided expand / no-expand b1)
-- ════════════════════════════════════════════════════════════════

/-- One BN site at the batched index. `statP` is the running-stat input prefix
    (`%{statP}mu` / `%{statP}var`), used only in `.eval` mode; in `.train` mode the statistics are
    reduced out of `xin` and `statP` names the slot the train step hands them back in.

    This is the ONLY place the two BN worlds are chosen between, which is the point: `@efficientnet_fwd`
    and `@efficientnet_fwd_eval` come from one traversal, so they cannot differ in anything but
    the BN mode. -/
private def bnSiteB (B oc hh ww : Nat) (mode : BnMode) (epsStr gName btName statP xin : String)
    (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × String × String) := do
  let zc  : Vec oc := fun _ => 0
  let zin : Vec (B * (oc*hh*ww)) := fun _ => 0
  match mode with
  | .train => bnFwdSite B oc hh ww sync replicas epsStr gName btName (gName.drop 1).toString xin
  | .eval  => do
      let (c, n) ← pretty B (.batchOp (N := B) (.bnEval (oc := oc) (h := hh) (w := ww)
                          gName btName s!"%{statP}mu" s!"%{statP}var" epsStr 0 zc zc zc zc)
                          (.operand xin zin))
      pure (c, n, "")

/-- Stride-1 expand MBConv forward body (shared by residual + no-skip): expand 1×1 conv-bn-swish →
    depthwise(kd) conv-bn-swish → SE → project 1×1 conv-bn. Returns the EFwd WITHOUT the final
    residual (caller adds the `addV` for residual blocks). -/
def eFwdBody (B ic mid oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EFwd := do
  let ww := hh
  let zIn  : Vec (B * (ic * hh * ww)) := fun _ => 0
  let zMid : Vec (B * (mid * hh * ww)) := fun _ => 0
  let _zOut : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B) (.convAt bf16 (h := hh) (w := ww) zrnd s!"%{p}eW" (biasName convBias s!"%{p}eb" mid) zKe zVm) (.operand xName zIn))
  let (cEn, nEn, stE) ← bnSiteB B mid hh ww mode epsStr s!"%{p}eg" s!"%{p}ebt" s!"{p}en" nEc (replicas := replicas) (sync := sync)
  let (cEr, nEr) ← pretty B (.batchOp (.swish) (.operand nEn zMid))
  let (cDc, nDc) ← pretty B (.batchOp (N := B) (.depthwiseAt bf16 (h := hh) (w := ww) zrnd s!"%{p}dW" (biasName convBias s!"%{p}db" mid) zDk zVm) (.operand nEr zMid))
  let (cDn, nDn, stD) ← bnSiteB B mid hh ww mode epsStr s!"%{p}dg" s!"%{p}dbt" s!"{p}dn" nDc (replicas := replicas) (sync := sync)
  let (cDr, nDr) ← pretty B (.batchOp (.swish) (.operand nDn zMid))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B mid hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := B) (.convAt bf16 (h := hh) (w := ww) zrnd s!"%{p}pW" (biasName convBias s!"%{p}pb" oc) zKp zVo) (.operand nSe zMid))
  let (cPn, nPn, stP) ← bnSiteB B oc hh ww mode epsStr s!"%{p}pg" s!"%{p}pbt" s!"{p}pn" nPc (replicas := replicas) (sync := sync)
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc,
         bns := [(nEc, s!"{p}en", mid, hh), (nDc, s!"{p}dn", mid, hh), (nPc, s!"{p}pn", oc, hh)],
         stE := stE, stD := stD, stP := stP, bnSt := [stE, stD, stP] }

/-- **Residual stride-1 MBConv forward** (ic = oc): body + `addV` skip. -/
def eFwd (B ic mid oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String)
    (convBias : Bool) (drop : Option Nat := none) (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EFwd := do
  let f ← eFwdBody B ic mid oc hh kd r mode epsStr p xName convBias bf16 replicas sync
  let zOut : Vec (B * (oc * hh * hh)) := fun _ => 0
  -- THE DROP SITE — on the RESIDUAL BRANCH, before the skip add, which is where the reference
  -- puts it (`x = x * keep / keep_prob` then `x = x + residual`). Scaling after the add would
  -- attenuate the identity path too: a different net that still trains, and invisible to every
  -- structural check. `dropPath_zeros_zero` plus the all-zero-mask control is what pins it here.
  -- At `drop = none` no `pretty` call happens, so the fresh-name counter does not move and every
  -- committed artifact re-renders byte-identically — the inertness gate's strong form, for free.
  let (cD, nD) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (dpName i) (fun _ => 0 : Vec B) (.operand f.o zOut))
    | none   => pure ("", f.o)
  let (cA, nA) ← pretty B (.addVB (.operand nD zOut) (.operand xName zOut))
  pure { f with code := f.code ++ cD ++ cA, o := nA }

/-- **No-skip stride-1 MBConv forward** (ic ≠ oc, b9/b16): body, output = project-BN out. -/
def eFwdNoSkip (B ic mid oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EFwd :=
  eFwdBody B ic mid oc hh kd r mode epsStr p xName convBias bf16 replicas sync

/-- **Strided MBConv forward** (b2/b4/b6/b12): expand at the input `2hh×2ww`, depthwise downsamples
    `2hh×2ww → hh×ww`, project 1×1 at `hh×ww`. NO skip. -/
def eFwdStrided (B ic mid oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EFwd := do
  let ww := hh
  let zIn  : Vec (B * (ic * (2*hh) * (2*ww))) := fun _ => 0
  let zMidH : Vec (B * (mid * (2*hh) * (2*ww))) := fun _ => 0
  let zMid : Vec (B * (mid * hh * ww)) := fun _ => 0
  let _zOut : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cEc, nEc) ← pretty B (.batchOp (N := B) (.convAt bf16 (h := 2*hh) (w := 2*ww) zrnd s!"%{p}eW" (biasName convBias s!"%{p}eb" mid) zKe zVm) (.operand xName zIn))
  let (cEn, nEn, stE) ← bnSiteB B mid (2*hh) (2*ww) mode epsStr s!"%{p}eg" s!"%{p}ebt" s!"{p}en" nEc (replicas := replicas) (sync := sync)
  let (cEr, nEr) ← pretty B (.batchOp (.swish) (.operand nEn zMidH))
  let (cDc, nDc) ← pretty B (.batchOp (N := B) (.depthwiseStridedAt bf16 (h := hh) (w := ww) zrnd s!"%{p}dW" (biasName convBias s!"%{p}db" mid) zDk zVm) (.operand nEr zMidH))
  let (cDn, nDn, stD) ← bnSiteB B mid hh ww mode epsStr s!"%{p}dg" s!"%{p}dbt" s!"{p}dn" nDc (replicas := replicas) (sync := sync)
  let (cDr, nDr) ← pretty B (.batchOp (.swish) (.operand nDn zMid))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B mid hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := B) (.convAt bf16 (h := hh) (w := ww) zrnd s!"%{p}pW" (biasName convBias s!"%{p}pb" oc) zKp zVo) (.operand nSe zMid))
  let (cPn, nPn, stP) ← bnSiteB B oc hh ww mode epsStr s!"%{p}pg" s!"%{p}pbt" s!"{p}pn" nPc (replicas := replicas) (sync := sync)
  pure { code := cEc ++ cEn ++ cEr ++ cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := nEc, en := nEn, er := nEr, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc,
         -- the expand stage runs at the block INPUT resolution, `2hh`; only the depthwise
         -- downsamples. That asymmetry is the one thing the stat layout gets wrong by default.
         bns := [(nEc, s!"{p}en", mid, 2*hh), (nDc, s!"{p}dn", mid, hh), (nPc, s!"{p}pn", oc, hh)],
         stE := stE, stD := stD, stP := stP, bnSt := [stE, stD, stP] }

/-- **No-expand MBConv forward** (b1, t=1): depthwise(kd, on `ic` channels)-bn-swish → SE → project
    1×1 (ic→oc)-bn. NO expand, NO skip. `ec/en` unused; `er` = block input (= depthwise input). -/
def eFwdNoExp (B ic oc hh kd r : Nat) (mode : BnMode) (epsStr p xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EFwd := do
  let ww := hh
  let zIn  : Vec (B * (ic * hh * ww)) := fun _ => 0
  let _zOut : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel ic kd kd := fun _ _ _ => 0
  let zVi  : Vec ic := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  let (cDc, nDc) ← pretty B (.batchOp (N := B) (.depthwiseAt bf16 (h := hh) (w := ww) zrnd s!"%{p}dW" (biasName convBias s!"%{p}db" ic) zDk zVi) (.operand xName zIn))
  let (cDn, nDn, stD) ← bnSiteB B ic hh ww mode epsStr s!"%{p}dg" s!"%{p}dbt" s!"{p}dn" nDc (replicas := replicas) (sync := sync)
  let (cDr, nDr) ← pretty B (.batchOp (.swish) (.operand nDn zIn))
  let (cSe, nS, nE1, nZ, nE2, nSe) ← seFwd B ic hh r p nDr
  let (cPc, nPc) ← pretty B (.batchOp (N := B) (.convAt bf16 (h := hh) (w := ww) zrnd s!"%{p}pW" (biasName convBias s!"%{p}pb" oc) zKp zVo) (.operand nSe zIn))
  let (cPn, nPn, stP) ← bnSiteB B oc hh ww mode epsStr s!"%{p}pg" s!"%{p}pbt" s!"{p}pn" nPc (replicas := replicas) (sync := sync)
  pure { code := cDc ++ cDn ++ cDr ++ cSe ++ cPc ++ cPn,
         o := nPn, ec := xName, en := xName, er := xName, dc := nDc, dn := nDn, dr := nDr,
         se := nSe, s := nS, e1 := nE1, z := nZ, e2 := nE2, pc := nPc,
         -- NO expand entry: `ec` is the block input here, not a BN input.
         bns := [(nDc, s!"{p}dn", ic, hh), (nPc, s!"{p}pn", oc, hh)],
         stD := stD, stP := stP, bnSt := [stD, stP] }

-- ════════════════════════════════════════════════════════════════
-- § MBConv backward emitters (project → SE → depthwise → expand; param-SGD in func-arg order)
-- ════════════════════════════════════════════════════════════════

/-- Stride-1 expand MBConv backward body (shared by residual + no-skip): returns the EBack with `dx`
    = the expand-conv-back cotangent (caller adds the residual `+ dyOut` for residual blocks). -/
private def eBackBody (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EBack := do
  let ww := hh
  let zMidF : Vec (B * (mid * hh * ww)) := fun _ => 0
  let zOutF : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  -- project: BN back (cot at project conv out) → 1×1 conv back (cot at SE out)
  let (cPbn, nPbn) ← bnBackSite B oc (hh) (ww) sync replicas epsStr s!"%{p}pg" f.pc s!"{p}pgdst" dyName f.stP
  let (cPdr, nPdr) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%{p}pW" zKp zVo (.operand nPbn zOutF))
  let (cgp, ngp) ← bnG adam B oc hh ww s!"%{p}pg" f.pc epsStr lrStr dyName (sync := sync) (st := f.stP)
  let (ctp, ntp) ← bnBt adam B oc hh ww s!"%{p}pbt" lrStr dyName
  let (cWp, nWp) ← convW1 adam B mid oc hh ww f.se s!"%{p}pW" lrStr nPbn (bf16 := bf16)
  let (cbp, nbp) ← if convBias then bnBt adam B oc hh ww s!"%{p}pb" lrStr nPbn else pure ("", "")
  -- SE back (dx at depthwise-swish out) + 4 SE param grads
  let (cSe, nDxSe, seNames) ← seBack adam B mid hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise: swish mask (cot at dw-BN out) → BN back (cot at dw conv out) → conv back (cot at expand-swish out)
  let (cDsw, nDsw) ← pretty B (.swishBackB f.dn (fun _ => 0) (.operand nDxSe zMidF))
  let (cDbn, nDbn) ← bnBackSite B mid (hh) (ww) sync replicas epsStr s!"%{p}dg" f.dc s!"{p}dgdst" nDsw f.stD
  let (cDer, nDer) ← pretty B (.depthwiseBackBatchedAt bf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd s!"%{p}dW" zDk zVm (.operand nDbn zMidF))
  let (cgd, ngd) ← bnG adam B mid hh ww s!"%{p}dg" f.dc epsStr lrStr nDsw (sync := sync) (st := f.stD)
  let (ctd, ntd) ← bnBt adam B mid hh ww s!"%{p}dbt" lrStr nDsw
  let (cWd, nWd) ← dwW adam B mid hh ww kd f.er s!"%{p}dW" lrStr nDbn (bf16 := bf16)
  let (cbd, nbd) ← if convBias then bnBt adam B mid hh ww s!"%{p}db" lrStr nDbn else pure ("", "")
  -- expand: swish mask (cot at expand-BN out) → BN back → 1×1 conv back (cot at block input)
  let (cEsw, nEsw) ← pretty B (.swishBackB f.en (fun _ => 0) (.operand nDer zMidF))
  let (cEbn, nEbn) ← bnBackSite B mid (hh) (ww) sync replicas epsStr s!"%{p}eg" f.ec s!"{p}egdst" nEsw f.stE
  let (cExb, nExb) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := ic) (oc := mid) (h := hh) (w := ww) zrnd s!"%{p}eW" zKe zVm (.operand nEbn zMidF))
  let (cge, nge) ← bnG adam B mid hh ww s!"%{p}eg" f.ec epsStr lrStr nEsw (sync := sync) (st := f.stE)
  let (cte, nte) ← bnBt adam B mid hh ww s!"%{p}ebt" lrStr nEsw
  let (cWe, nWe) ← convW1 adam B ic mid hh ww xName s!"%{p}eW" lrStr nEbn (bf16 := bf16)
  let (cbe, nbe) ← if convBias then bnBt adam B mid hh ww s!"%{p}eb" lrStr nEbn else pure ("", "")
  let names := [nWe] ++ biasSlot convBias nbe ++ [nge, nte, nWd] ++ biasSlot convBias nbd ++
               [ngd, ntd] ++ seNames ++ [nWp] ++ biasSlot convBias nbp ++ [ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDer ++ cgd ++ ctd ++ cWd ++ cbd ++
                 cEsw ++ cEbn ++ cExb ++ cge ++ cte ++ cWe ++ cbe,
         dx := nExb, names := names }

/-- **Residual stride-1 MBConv backward** (ic = oc): body + skip fan-in `+ dyOut`. -/
private def eBack (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool) (drop : Option Nat := none)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EBack := do
  -- THE DROP'S BACKWARD IS THE SAME OP AT THE SAME SCALE (`Proofs.dropPath_vjp_is_self`) — a
  -- diagonal linear map is its own transpose, so there is no `*Grad` peer to keep in step.
  -- IT APPLIES TO THE BRANCH ONLY. `eBackBody` consumes this cotangent for the whole branch
  -- INCLUDING its parameter gradients, so feeding it `dyd` is what makes the project/depthwise/
  -- expand grads see the dropped signal; the skip's fan-in below keeps the RAW `dyName`. Dropping
  -- there too would attenuate the identity path — the mirror of the forward's placement trap.
  let zOut : Vec (B * (oc * hh * hh)) := fun _ => 0
  let (cD, dyd) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (dpName i) (fun _ => 0 : Vec B) (.operand dyName zOut))
    | none   => pure ("", dyName)
  let b ← eBackBody adam B ic mid oc hh kd r epsStr lrStr p xName f dyd convBias (bf16 := bf16)
    (replicas := replicas) (sync := sync)
  let zIn : Vec (B * (ic * hh * hh)) := fun _ => 0
  let (cDx, nDx) ← pretty B (.addVB (.operand b.dx zIn) (.operand dyName zIn))
  pure { b with code := cD ++ b.code ++ cDx, dx := nDx }

/-- **No-skip stride-1 MBConv backward** (ic ≠ oc, b9/b16): body, dx = expand-conv-back directly. -/
private def eBackNoSkip (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool) (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EBack :=
  eBackBody adam B ic mid oc hh kd r epsStr lrStr p xName f dyName convBias bf16 replicas sync

/-- **Strided MBConv backward** (b2/b4/b6/b12): depthwise-back upsamples `hh×ww → 2hh×2ww`; the
    expand stage backward runs at `2hh×2ww`. NO skip. -/
private def eBackStrided (adam : Bool) (B ic mid oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EBack := do
  let ww := hh
  let zMidHF : Vec (B * (mid * (2*hh) * (2*ww))) := fun _ => 0
  let zMidF : Vec (B * (mid * hh * ww)) := fun _ => 0
  let zOutF : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKe  : Kernel4 mid ic 1 1 := fun _ _ _ _ => 0
  let zKp  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel mid kd kd := fun _ _ _ => 0
  let zVm  : Vec mid := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  -- project (at hh)
  let (cPbn, nPbn) ← bnBackSite B oc (hh) (ww) sync replicas epsStr s!"%{p}pg" f.pc s!"{p}pgdst" dyName f.stP
  let (cPdr, nPdr) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%{p}pW" zKp zVo (.operand nPbn zOutF))
  let (cgp, ngp) ← bnG adam B oc hh ww s!"%{p}pg" f.pc epsStr lrStr dyName (sync := sync) (st := f.stP)
  let (ctp, ntp) ← bnBt adam B oc hh ww s!"%{p}pbt" lrStr dyName
  let (cWp, nWp) ← convW1 adam B mid oc hh ww f.se s!"%{p}pW" lrStr nPbn (bf16 := bf16)
  let (cbp, nbp) ← if convBias then bnBt adam B oc hh ww s!"%{p}pb" lrStr nPbn else pure ("", "")
  -- SE back (at hh)
  let (cSe, nDxSe, seNames) ← seBack adam B mid hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise (swish + BN at hh, strided conv-back upsamples to 2hh)
  let (cDsw, nDsw) ← pretty B (.swishBackB f.dn (fun _ => 0) (.operand nDxSe zMidF))
  let (cDbn, nDbn) ← bnBackSite B mid (hh) (ww) sync replicas epsStr s!"%{p}dg" f.dc s!"{p}dgdst" nDsw f.stD
  let (cDer, nDer) ← pretty B (.depthwiseStridedBackBatchedAt bf16 (N := B) (c := mid) (h := hh) (w := ww) zrnd s!"%{p}dW" zDk zVm (.operand nDbn zMidF))
  let (cgd, ngd) ← bnG adam B mid hh ww s!"%{p}dg" f.dc epsStr lrStr nDsw (sync := sync) (st := f.stD)
  let (ctd, ntd) ← bnBt adam B mid hh ww s!"%{p}dbt" lrStr nDsw
  let (cWd, nWd) ← dwWS adam B mid hh ww kd f.er s!"%{p}dW" lrStr nDbn (bf16 := bf16)
  let (cbd, nbd) ← if convBias then bnBt adam B mid hh ww s!"%{p}db" lrStr nDbn else pure ("", "")
  -- expand (at 2hh)
  let (cEsw, nEsw) ← pretty B (.swishBackB f.en (fun _ => 0) (.operand nDer zMidHF))
  let (cEbn, nEbn) ← bnBackSite B mid (2*hh) (2*ww) sync replicas epsStr s!"%{p}eg" f.ec s!"{p}egdst" nEsw f.stE
  let (cExb, nExb) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := ic) (oc := mid) (h := 2*hh) (w := 2*ww) zrnd s!"%{p}eW" zKe zVm (.operand nEbn zMidHF))
  let (cge, nge) ← bnG adam B mid (2*hh) (2*ww) s!"%{p}eg" f.ec epsStr lrStr nEsw (sync := sync) (st := f.stE)
  let (cte, nte) ← bnBt adam B mid (2*hh) (2*ww) s!"%{p}ebt" lrStr nEsw
  let (cWe, nWe) ← convW1 adam B ic mid (2*hh) (2*ww) xName s!"%{p}eW" lrStr nEbn (bf16 := bf16)
  let (cbe, nbe) ← if convBias then bnBt adam B mid (2*hh) (2*ww) s!"%{p}eb" lrStr nEbn else pure ("", "")
  let names := [nWe] ++ biasSlot convBias nbe ++ [nge, nte, nWd] ++ biasSlot convBias nbd ++
               [ngd, ntd] ++ seNames ++ [nWp] ++ biasSlot convBias nbp ++ [ngp, ntp]
  pure { code := cPbn ++ cPdr ++ cgp ++ ctp ++ cWp ++ cbp ++ cSe ++
                 cDsw ++ cDbn ++ cDer ++ cgd ++ ctd ++ cWd ++ cbd ++
                 cEsw ++ cEbn ++ cExb ++ cge ++ cte ++ cWe ++ cbe,
         dx := nExb, names := names }

/-- **No-expand MBConv backward** (b1): project back → SE back → depthwise back → dx (block input).
    12 params at `convBias := true`: depthwise W b γ β, SE W₁ b₁ W₂ b₂, project W b γ β; the two
    conv biases drop out at `convBias := false` (10). -/
private def eBackNoExp (adam : Bool) (B ic oc hh kd r : Nat) (epsStr lrStr p xName : String)
    (f : EFwd) (dyName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS EBack := do
  let ww := hh
  let zInF  : Vec (B * (ic * hh * ww)) := fun _ => 0
  let zOutF : Vec (B * (oc * hh * ww)) := fun _ => 0
  let zKp  : Kernel4 oc ic 1 1 := fun _ _ _ _ => 0
  let zDk  : DepthwiseKernel ic kd kd := fun _ _ _ => 0
  let zVi  : Vec ic := fun _ => 0
  let zVo  : Vec oc := fun _ => 0
  -- project
  let (cPbn, nPbn) ← bnBackSite B oc (hh) (ww) sync replicas epsStr s!"%{p}pg" f.pc s!"{p}pgdst" dyName f.stP
  let (cPdr, nPdr) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := ic) (oc := oc) (h := hh) (w := ww) zrnd s!"%{p}pW" zKp zVo (.operand nPbn zOutF))
  let (cgp, ngp) ← bnG adam B oc hh ww s!"%{p}pg" f.pc epsStr lrStr dyName (sync := sync) (st := f.stP)
  let (ctp, ntp) ← bnBt adam B oc hh ww s!"%{p}pbt" lrStr dyName
  let (cWp, nWp) ← convW1 adam B ic oc hh ww f.se s!"%{p}pW" lrStr nPbn (bf16 := bf16)
  let (cbp, nbp) ← if convBias then bnBt adam B oc hh ww s!"%{p}pb" lrStr nPbn else pure ("", "")
  -- SE back (on ic channels)
  let (cSe, nDxSe, seNames) ← seBack adam B ic hh r lrStr p f.dr f.s f.e1 f.z f.e2 nPdr
  -- depthwise (on ic channels)
  let (cDsw, nDsw) ← pretty B (.swishBackB f.dn (fun _ => 0) (.operand nDxSe zInF))
  let (cDbn, nDbn) ← bnBackSite B ic (hh) (ww) sync replicas epsStr s!"%{p}dg" f.dc s!"{p}dgdst" nDsw f.stD
  let (cDxb, nDxb) ← pretty B (.depthwiseBackBatchedAt bf16 (N := B) (c := ic) (h := hh) (w := ww) zrnd s!"%{p}dW" zDk zVi (.operand nDbn zInF))
  let (cgd, ngd) ← bnG adam B ic hh ww s!"%{p}dg" f.dc epsStr lrStr nDsw (sync := sync) (st := f.stD)
  let (ctd, ntd) ← bnBt adam B ic hh ww s!"%{p}dbt" lrStr nDsw
  let (cWd, nWd) ← dwW adam B ic hh ww kd xName s!"%{p}dW" lrStr nDbn (bf16 := bf16)
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
moment slots, and the raw dimensions for the emitted Adam ops (`adamMNextF`'s `ds`); `ty ds`
renders the type string. -/

private def eSig (p : String) (ic mid oc r kd : Nat) (convBias : Bool) : List (String × List Nat) :=
  -- `zb1`/`zb2` are the SQUEEZE-EXCITE biases and STAY: those convs are followed by an
  -- activation, not BN, so nothing absorbs them and the reference carries them too.
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

    **NOT the SE widths.** SE's two 1×1 convs are followed by an ACTIVATION, not BN, so nothing
    absorbs their biases and the reference carries them; they stay real parameters and are
    never bound to a zero constant. The rule for which biases fold — a rank-1 parameter after a
    **rank-4** kernel — excludes them because SE's params are rank-2. -/
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
  /-- **THE CLASSIFIER'S ACTUAL INPUT** — the pooled activation with classifier dropout OFF, the
      `dropoutB` output with it ON. The record carries no separate pooled name, because there are
      TWO consumers of this value and one of them is easy to miss:

      * the dense forward, which obviously reads it; and
      * **the dense WEIGHT gradient**, `∂L/∂W = Σ_b dy_b ⊗ (input_b)` — which reads the dense's
        input, i.e. the DROPPED activation, not the pooled one.

      Feeding `dnW` the undropped pooled activation type-checks, trains, descends, and is wrong on
      the one parameter dropout acts through. It is invisible to every ones-mask gate this feature
      has, because at `mask ≡ 1` the two values are equal; leaving the pooled name off the record
      means there is nothing else to read. -/
  cin    : String            -- dense input (= gap, or the dropout output when cd is on)
  logits : String            -- dense out
  /-- The 49 BN layers as `(BN-input SSA, stat prefix, channels, spatial side)`, stem → blocks in
      forward order → head. Single source for the eval signature, the eval BN sites and the AdamW
      train step's returned batch statistics — see `EFwd.bns`. -/
  bns    : List (String × String × Nat × Nat)
  sst    : String := ""       -- stem BN's packed global statistics (sync-BN only)
  hst    : String := ""       -- head BN's packed global statistics (sync-BN only)
  bnSt   : List String := []  -- every BN site's packed statistics, aligned with `bns`
  deriving Inhabited

/-- Stem forward: 3×3/s2 XLA-`SAME` conv (3→32, 224→112) → BN → swish, on `%x`. -/
def enetStemFwdB (B : Nat) (mode : BnMode) (epsStr : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS StemFwdB := do
  let zx   : Vec (B * (3*224*224)) := fun _ => 0
  let zSk  : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
  let z32  : Vec 32 := fun _ => 0
  let z112F : Vec (B * (32*112*112)) := fun _ => 0
  let (cStc, nStc) ← pretty B (.batchOp (N := B) (.convStridedXlaAt bf16 (h := 112) (w := 112) zrnd "%sW" (biasName convBias "%sb" 32) zSk z32) (.operand "%x" zx))
  let (cStn, nStn, sst) ← bnSiteB B 32 112 112 mode epsStr "%sg" "%sbt" "stn" nStc (replicas := replicas) (sync := sync)
  let (cStr, nStr) ← pretty B (.batchOp (.swish) (.operand nStn z112F))
  pure { code := cStc ++ cStn ++ cStr, c := nStc, n := nStn, st := sst, o := nStr }

/-- The head's saved SSA names up to the GAP: conv, BN, BN stats, swish, GAP. -/
structure ENetHeadFwdB where
  code : String
  hc : String
  hn : String
  hst : String
  hr : String
  gap : String

/-- Head forward up to the GAP: 1×1 conv (320→1280) → BN → swish → GAP(7×7), on block 16's output
    `xName`. The classifier dropout and the dense stay in the chain (`cd` decides the dropout). -/
def enetHeadFwdB (B _nClasses : Nat) (mode : BnMode) (epsStr xName : String) (convBias : Bool)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS ENetHeadFwdB := do
  let z7F   : Vec (B * (320*7*7)) := fun _ => 0
  let zHk   : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
  let z1280 : Vec 1280 := fun _ => 0
  let zH7F  : Vec (B * (1280*7*7)) := fun _ => 0
  let (cHc, nHc) ← pretty B (.batchOp (N := B) (.convAt bf16 (h := 7) (w := 7) zrnd "%hW" (biasName convBias "%hb" 1280) zHk z1280) (.operand xName z7F))
  let (cHn, nHn, hst) ← bnSiteB B 1280 7 7 mode epsStr "%hg" "%hbt" "hn" nHc (replicas := replicas) (sync := sync)
  let (cHr, nHr) ← pretty B (.batchOp (.swish) (.operand nHn zH7F))
  let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 1280) (h := 7) (w := 7)) (.operand nHr zH7F))
  pure { code := cHc ++ cHn ++ cHr ++ cGap, hc := nHc, hn := nHn, hst := hst, hr := nHr, gap := nGap }

/-- **The full EfficientNet-B0 forward as `pretty` of the verified AST**, at the batched index
    `N := B`. 3×3/s2 stem (3→32, 224→112) → 16 MBConv blocks (the paper `[t,c,n,s]` table, SE in
    every block) → 1×1 head (320→1280) → GAP(7×7) → dense(1280→`nClasses`).

    `mode` picks the BN world and NOTHING else, which is the whole point of routing both forward
    artifacts through one chain: at `.train` every BN reduces its own batch statistics (`bnBatchF`)
    and this is exactly the prefix the train step differentiates; at `.eval` every BN consumes the
    driver's frozen running stats (the `bnEval` descriptor) and the net becomes
    class-batch-independent. -/
private def enetFwdChain (B nClasses : Nat) (mode : BnMode) (epsStr : String) (convBias : Bool)
    (sd : Bool := false) (cd : Bool := false)
    -- TRAILING and defaulted, so every committed forward re-renders byte-identical.
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS ENetFwd := do
  -- `dp i` is `some i` exactly when block `i` is in `enetDropIdxs` AND stochastic depth is on —
  -- so the one list drives the signature and every call site, and the ramp index carried is the
  -- BLOCK index (see `enetDropIdxs`' note on why the site ordinal would be wrong).
  let dp : Nat → Option Nat := fun i => if sd && enetDropIdxs.contains i then some i else none
    -- ═══ stem: 3×3/s2 conv (3→32, 224→112) → bn → swish ═══
    let st ← enetStemFwdB B mode epsStr convBias bf16 replicas sync
    let (nStc, nStn, sst, nStr) := (st.c, st.n, st.st, st.o)
    -- ═══ forward: 16 MBConv blocks ═══
    let f1  ← eFwdNoExp   B 32      16 112 3  8 mode epsStr "b1"  nStr convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f2  ← eFwdStrided B 16  96  24  56 3  4 mode epsStr "b2"  f1.o convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f3  ← eFwd        B 24 144  24  56 3  6 mode epsStr "b3"  f2.o convBias (dp 2) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f4  ← eFwdStrided B 24 144  40  28 5  6 mode epsStr "b4"  f3.o convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f5  ← eFwd        B 40 240  40  28 5 10 mode epsStr "b5"  f4.o convBias (dp 4) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f6  ← eFwdStrided B 40 240  80  14 3 10 mode epsStr "b6"  f5.o convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f7  ← eFwd        B 80 480  80  14 3 20 mode epsStr "b7"  f6.o convBias (dp 6) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f8  ← eFwd        B 80 480  80  14 3 20 mode epsStr "b8"  f7.o convBias (dp 7) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f9  ← eFwdNoSkip  B 80 480 112  14 5 20 mode epsStr "b9"  f8.o convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f10 ← eFwd        B 112 672 112 14 5 28 mode epsStr "b10" f9.o convBias (dp 9) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f11 ← eFwd        B 112 672 112 14 5 28 mode epsStr "b11" f10.o convBias (dp 10) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f12 ← eFwdStrided B 112 672 192  7 5 28 mode epsStr "b12" f11.o convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f13 ← eFwd        B 192 1152 192 7 5 48 mode epsStr "b13" f12.o convBias (dp 12) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f14 ← eFwd        B 192 1152 192 7 5 48 mode epsStr "b14" f13.o convBias (dp 13) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f15 ← eFwd        B 192 1152 192 7 5 48 mode epsStr "b15" f14.o convBias (dp 14) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let f16 ← eFwdNoSkip  B 192 1152 320 7 3 48 mode epsStr "b16" f15.o convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    -- ═══ head: 1×1 conv (320→1280) → bn → swish → GAP → dense ═══
    let hd ← enetHeadFwdB B nClasses mode epsStr f16.o convBias bf16 replicas sync
    let (nHc, nHn, hst, nGap) := (hd.hc, hd.hn, hd.hst, hd.gap)
    let z1280c : Vec (B * 1280) := fun _ => 0
    let zWd   : Mat 1280 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    -- CLASSIFIER DROPOUT: the per-ELEMENT inverted mask, exactly where the reference puts it —
    -- between GAP and the dense (`emitForward`'s classifier dropout in `jax/Jax/Codegen.lean`, the `.dense` case).
    -- At `cd = false` NO `pretty` call happens, so the fresh-name counter does not move and every
    -- committed artifact re-renders byte-identically. Same convention as `drop`'s `Option Nat`; it
    -- is what makes the inertness gate a byte claim rather than a diff-review.
    -- Emitted in the FORWARD too, at the driver's all-ones mask, so `@efficientnet_do_fwd` stays
    -- a byte-prefix of the train step and the `forward ⊂ train-step` audit survives. The identity
    -- is exact, not close (`Proofs.dropout_ones_id`; `1 * x = x` in IEEE).
    let (cDo, nCin) ← if cd then
        pretty B (.dropoutB (N := B) (n := enetHeadWidth) doName (fun _ => 0 : Vec (B * enetHeadWidth))
          (.operand nGap z1280c))
      else pure ("", nGap)
    let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC) (.operand nCin z1280c))
    pure { code := st.code ++
             f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++
             f8.code ++ f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++
             f15.code ++ f16.code ++ hd.code ++ cDo ++ cLog,
           stc := nStc, stn := nStn, str := nStr,
           blocks := #[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16],
           hc := nHc, hn := nHn, cin := nCin, logits := nLog,
           bns := (nStc, "stn", 32, 112) ::
             (f1.bns ++ f2.bns ++ f3.bns ++ f4.bns ++ f5.bns ++ f6.bns ++ f7.bns ++ f8.bns ++
              f9.bns ++ f10.bns ++ f11.bns ++ f12.bns ++ f13.bns ++ f14.bns ++ f15.bns ++
              f16.bns ++ [(nHc, "hn", 1280, 7)]),
           sst := sst, hst := hst,
           bnSt := sst ::
             (f1.bnSt ++ f2.bnSt ++ f3.bnSt ++ f4.bnSt ++ f5.bnSt ++ f6.bnSt ++ f7.bnSt ++ f8.bnSt ++
              f9.bnSt ++ f10.bnSt ++ f11.bnSt ++ f12.bnSt ++ f13.bnSt ++ f14.bnSt ++ f15.bnSt ++
              f16.bnSt ++ [hst]) }

/-- The 263-input `@efficientnet_fwd` / 361-input `@efficientnet_fwd_eval` argument signature.
    The stat half is derived from the SAME `bns` the traversal built (never a parallel table), so
    the eval forward's slots cannot drift out of the order the driver packs `runningBnStats` in. -/
private def enetFwdSig (B nClasses : Nat) (mode : BnMode) (epsStr : String) (convBias : Bool)
    (sd : Bool := false) (cd : Bool := false) : String :=
  let F : ENetFwd := (enetFwdChain B nClasses mode epsStr convBias sd cd).run' (0, [])
  let params := (enetSig nClasses convBias).map (fun (nm, d) => s!"%{nm}: {ty d}")
  let stats := if mode == .train then [] else
    F.bns.flatMap (fun (_, sp, c, _) => [s!"%{sp}mu: {ty [c]}", s!"%{sp}var: {ty [c]}"])
  -- The drop inputs go LAST, after the BN stats, so adding them cannot shift an existing positional
  -- slot: a parameter inserted mid-list captures an existing argument, and the driver walks this
  -- signature positionally. And the dropout mask goes after THOSE — see `enetDropoutSig` on why the
  -- order within the per-example tail matters as well as the tail's position.
  String.intercalate ", " ((s!"%x: {ty [B, 3*224*224]}") :: (params ++ stats)) ++
    enetDropSig B sd ++ enetDropoutSig B cd

/-- **`@efficientnet_fwd` rendered ENTIRELY from the verified AST** — 263 inputs (`%x` plus the 262
    params in `enetSig` order), returning logits `[B, nClasses]`. Shares `enetFwdChain` with the
    train step, so it is a byte-identical PREFIX of `efficientnet_train_step.mlir`, ending exactly
    where the loss begins. -/
def efficientnetFwdFaithfulV (B nClasses : Nat) (epsStr : String) (convBias : Bool := false)
    (slug : String := "efficientnet") (sd : Bool := false) (cd : Bool := false) : String :=
  let F : ENetFwd := (enetFwdChain B nClasses .train epsStr convBias sd cd).run' (0, [])
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd({enetFwdSig B nClasses .train epsStr convBias sd cd}) -> {ty [B, nClasses]} " ++ "{\n" ++
  s!"    // ── EfficientNet-B0 forward: every op is pretty(verified AST node){if convBias then "" else " except the %zb zero-bias constants"} ──\n" ++
  zeroBiasPrelude convBias enetBiasWidths ++ F.code ++
  s!"    return {F.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

/-- **`@efficientnet_fwd_eval` rendered ENTIRELY from the verified AST** — the inference forward,
    every BN site consuming frozen per-channel running stats (the `bnEval` descriptor, `den` =
    `batchMap N bnPerChannelEvalTensor3`) instead of reducing statistics out of its activation.
    Same 262 params in the same order, plus the 98 stat inputs (49 BN layers × μ/var, interleaved
    per layer in `bnChannels` order): **361 inputs**.

    This is the eval partner of `efficientnet_adam_train_step`, whose returned batch μ/var the
    driver EMAs into exactly these slots — and both sides of that contract come off one
    `bns` list rather than two independently-written ones. -/
def efficientnetFwdEvalFaithfulV (B nClasses : Nat) (epsStr : String) (convBias : Bool := false)
    (slug : String := "efficientnet") (sd : Bool := false) (cd : Bool := false) : String :=
  let F : ENetFwd := (enetFwdChain B nClasses .eval epsStr convBias sd cd).run' (0, [])
  "module @m {\n" ++
  s!"  func.func @{fwdEvalEntry slug epsStr}({enetFwdSig B nClasses .eval epsStr convBias sd cd}) -> {ty [B, nClasses]} " ++ "{\n" ++
  s!"    // ── EfficientNet-B0 eval forward (running-stats BN): every op is pretty(verified AST node){if convBias then "" else " except the %zb zero-bias constants"} ──\n" ++
  zeroBiasPrelude convBias enetBiasWidths ++ F.code ++
  s!"    return {F.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

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
    rather than in artifacts, with two emitters for `efficientnet_train_step` free to compute
    *different* functions.

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
    construction, not by inspection. -/
private def enetBackAll (B nClasses : Nat) (epsStr lrStr : String) (adam : Bool)
    (smooth : Option (String × String × String) := none) (convBias : Bool := false)
    (sd : Bool := false) (cd : Bool := false) (bf16 : Bool := false)
    (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS
      (String × List String × String × List (String × String × Nat × Nat) × List String) := do
    let F : ENetFwd ← enetFwdChain B nClasses .train epsStr convBias sd cd bf16 replicas sync
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
    -- `ENetFwd` carries no pooled (pre-dropout) name: the classifier weight gradient must read
    -- `F.cin` (the dense's actual input — see `ENetFwd.cin`). The cotangent path reaches GAP through
    -- `gapBackBatched`, which needs no forward name at all.
    let nHc := F.hc; let nHn := F.hn; let nLog := F.logits
    let z32  : Vec 32 := fun _ => 0
    let z112F : Vec (B * (32*112*112)) := fun _ => 0
    let zSk  : Kernel4 32 3 3 3 := fun _ _ _ _ => 0
    let zx   : Vec (B * (3*224*224)) := fun _ => 0
    let z1280 : Vec 1280 := fun _ => 0
    let zH7F  : Vec (B * (1280*7*7)) := fun _ => 0
    let zHk   : Kernel4 1280 320 1 1 := fun _ _ _ _ => 0
    let z1280c : Vec (B * 1280) := fun _ => 0
    let zWd   : Mat 1280 nClasses := fun _ _ => 0
    -- `zNCb` is the ROW-indexed logit leaf (`rows = 1`, so `B·(1·nClasses)`) that the softmax /
    -- row-back / sub / smoothing nodes all sit at. The two head-dense parameter tails want the
    -- PLAIN batched index `B·nClasses` instead — same vector, non-defeq type — so `dnW`/`dnB`
    -- build that leaf themselves. See `zCc1` for the same wrinkle inside the SE gate.
    let zNCb  : Vec (B * (1 * nClasses)) := fun _ => 0
    let (cSm, nSm) ← pretty B (.batchOp (.softmaxRow (m := 1) (n := nClasses)) (.operand nLog zNCb))
    -- ═══ the cotangent tail. `none` emits the subtraction alone (so the SGD render is
    --     byte-identical); `some` emits `smoothedCotB`, which with the softmax above is `pretty` of
    --     `smoothedLossCotGraph`, the graph the step tie starts from. `shiftB` emits
    --     `add x, dense<−α/K>` where the hand-written render emits `subtract x, α/K` — IEEE
    --     subtraction IS addition of the exact negation, so the two are bit-identical. `divConstB`
    --     emits a real `divide` rather than a multiply by `1/B`, which is only exact in binary32 at
    --     powers of two. ═══
    let (cDy, nDy) ← match smooth with
      | none => pretty B (.subB (.operand nSm zNCb) (.operand "%onehot" zNCb))
      | some (aStr, negAK, bStr) => smoothedCotB B (1 * nClasses) aStr negAK bStr nSm
    -- ═══ head backward: dense back → GAP back → swish mask → bn back → 1×1 conv back ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (.denseRowBack (rows := 1) (a := 1280) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    -- `F.cin`, NOT `nGap` — the classifier weight gradient reads the DENSE'S INPUT, which with
    -- classifier dropout on is the dropped activation. `∂L/∂W = Σ_b dy_b ⊗ (mask_b ⊙ gap_b)`.
    -- Passing `nGap` here type-checks, trains, descends, and is wrong on the one parameter dropout
    -- acts through — invisible to every ones-mask gate, because there the two values are equal.
    -- See `ENetFwd.cin` (ConvNeXt's LayerScale-γ carries the same hazard).
    let (cWfc, nWfc) ← dnW adam B 1280 nClasses F.cin "%Wd" lrStr nDy
    let (cbfc, nbfc) ← dnB adam B nClasses "%bd" lrStr nDy
    -- CLASSIFIER DROPOUT'S BACKWARD IS THE SAME OP AT THE SAME MASK
    -- (`Proofs.dropout_vjp_is_self`) — a diagonal linear map is its own transpose. It sits between
    -- the dense's input-VJP and the GAP backward, mirroring the forward's position between GAP and
    -- the dense. The DENSE side is above and reads `F.cin`; this side scales the cotangent on its
    -- way DOWN. Both are needed and neither implies the other: the first is the weight gradient,
    -- the second is everything upstream of the classifier.
    let (cDdo, nDdo) ← if cd then
        pretty B (.dropoutB (N := B) (n := enetHeadWidth) doName (fun _ => 0 : Vec (B * enetHeadWidth))
          (.operand nDgi z1280c))
      else pure ("", nDgi)
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 1280) (h := 7) (w := 7) (.operand nDdo z1280c))
    let (cHsw, nHsw) ← pretty B (.swishBackB nHn (fun _ => 0) (.operand nDgp zH7F))
    let (cHbn, nHbn) ← bnBackSite B 1280 (7) (7) sync replicas epsStr "%hg" nHc "hgdst" nHsw F.hst
    let (cHxb, nHxb) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := 320) (oc := 1280) (h := 7) (w := 7) zrnd "%hW" zHk z1280 (.operand nHbn zH7F))
    let (cgh, ngh) ← bnG adam B 1280 7 7 "%hg" nHc epsStr lrStr nHsw (sync := sync) (st := F.hst)
    let (cth, nth) ← bnBt adam B 1280 7 7 "%hbt" lrStr nHsw
    let (cWh, nWh) ← convW1 adam B 320 1280 7 7 f16.o "%hW" lrStr nHbn (bf16 := bf16)
    let (cbh, nbh) ← if convBias then bnBt adam B 1280 7 7 "%hb" lrStr nHbn else pure ("", "")
    -- ═══ backward: 16 blocks reversed (cotangent threads from nHxb) ═══
    let b16 ← eBackNoSkip  adam B 192 1152 320 7 3 48 epsStr lrStr "b16" f15.o f16 nHxb convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b15 ← eBack        adam B 192 1152 192 7 5 48 epsStr lrStr "b15" f14.o f15 b16.dx convBias (dp 14) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b14 ← eBack        adam B 192 1152 192 7 5 48 epsStr lrStr "b14" f13.o f14 b15.dx convBias (dp 13) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b13 ← eBack        adam B 192 1152 192 7 5 48 epsStr lrStr "b13" f12.o f13 b14.dx convBias (dp 12) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b12 ← eBackStrided adam B 112 672 192  7 5 28 epsStr lrStr "b12" f11.o f12 b13.dx convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b11 ← eBack        adam B 112 672 112 14 5 28 epsStr lrStr "b11" f10.o f11 b12.dx convBias (dp 10) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b10 ← eBack        adam B 112 672 112 14 5 28 epsStr lrStr "b10" f9.o  f10 b11.dx convBias (dp 9) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b9  ← eBackNoSkip  adam B 80 480 112  14 5 20 epsStr lrStr "b9"  f8.o  f9  b10.dx convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b8  ← eBack        adam B 80 480  80  14 3 20 epsStr lrStr "b8"  f7.o  f8  b9.dx convBias (dp 7) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b7  ← eBack        adam B 80 480  80  14 3 20 epsStr lrStr "b7"  f6.o  f7  b8.dx convBias (dp 6) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b6  ← eBackStrided adam B 40 240  80  14 3 10 epsStr lrStr "b6"  f5.o  f6  b7.dx convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b5  ← eBack        adam B 40 240  40  28 5 10 epsStr lrStr "b5"  f4.o  f5  b6.dx convBias (dp 4) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b4  ← eBackStrided adam B 24 144  40  28 5  6 epsStr lrStr "b4"  f3.o  f4  b5.dx convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b3  ← eBack        adam B 24 144  24  56 3  6 epsStr lrStr "b3"  f2.o  f3  b4.dx convBias (dp 2) (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b2  ← eBackStrided adam B 16  96  24  56 3  4 epsStr lrStr "b2"  f1.o  f2  b3.dx convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    let b1  ← eBackNoExp   adam B 32      16 112 3  8 epsStr lrStr "b1"  nStr  f1  b2.dx convBias (bf16 := bf16) (replicas := replicas) (sync := sync)
    -- ═══ stem backward: swish mask → bn back, then the 4 stem params (NO conv-back past %x) ═══
    let (cDsr, nDsr) ← pretty B (.swishBackB nStn (fun _ => 0) (.operand b1.dx z112F))
    let (cDsn, nDsn) ← bnBackSite B 32 (112) (112) sync replicas epsStr "%sg" nStc "sgdst" nDsr F.sst
    let (csW, nsW) ← if adam then
        pretty B (.convStridedXlaWeightGradBAt bf16 (N := B) (ic := 3) (oc := 32) (h := 112) (w := 112) zrnd "%x" z32 zx zSk (.operand nDsn z112F))
      else
        pretty B (.convStridedXlaWeightSgdB (N := B) (ic := 3) (oc := 32) (h := 112) (w := 112) "%x" "%sW" lrStr z32 zx zSk 0 (.operand nDsn z112F))
    let (csb, nsb) ← if convBias then bnBt adam B 32 112 112 "%sb" lrStr nDsn else pure ("", "")
    let (csg, nsg) ← bnG adam B 32 112 112 "%sg" nStc epsStr lrStr nDsr (sync := sync) (st := F.sst)
    let (cst, nst) ← bnBt adam B 32 112 112 "%sbt" lrStr nDsr
    -- ═══ assemble (params in func-arg order: stem, blocks fwd-order, head, dense) ═══
    -- `F.code` is exactly what `@efficientnet_fwd` renders, so the artifact is its byte-prefix.
    let fwdCode := F.code ++ cSm ++ cDy
    let bwdCode := cDgi ++ cWfc ++ cbfc ++ cDdo ++ cDgp ++ cHsw ++ cHbn ++ cHxb ++ cgh ++ cth ++ cWh ++ cbh ++
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
    pure (fwdCode ++ bwdCode, outNames, nSm, F.bns, F.bnSt)

/-- **EfficientNet-B0 (full 16-MBConv) SGD train step rendered ENTIRELY from the verified AST**, at
    the batched index `N·(c·h·w)`. Every emitted line is `pretty` of a verified `SHlo` node. Strided
    stem 3×3/s2 (3→32, 224→112) → b1 (no-expand) → b2..b16 (4 strided downsamples 112→7, 9 residual
    skips, 2 no-skip widenings) → 1×1 conv-bn-swish head (320→1280) → GAP → dense (1280→nClasses).

    The cotangent is plain `softmax − onehot` with the batch mean folded into `lrStr` — so the
    committed `lrStr = 0.05` is an effective **1.6** on the mean loss. That is a tuned value, not a
    slip; the AdamW render below spells the mean explicitly instead. -/
-- Evidence for the tuned lr: runs/efficientnet_verified_crop_gpu1.log, the per-epoch val accuracy
-- of an 80-epoch Imagenette run at this lr.
def efficientnetTrainStepFaithfulV (B nClasses : Nat) (epsStr lrStr : String)
    (funcName : String := "efficientnet_train_step") (convBias : Bool := false) : String :=
  let go : StateM Proofs.StableHLO.EmitS String := do
    let (code, outNames, _, _, _) ← enetBackAll B nClasses epsStr lrStr false none convBias
    let outTypes : List String := (enetSig nClasses convBias).map (fun p => ty p.2)
    pure <|
      s!"    // ── EfficientNet-B0 (16-MBConv) train step: every op is pretty(verified AST node){if convBias then "" else " except the %zb zero-bias constants"} ──\n" ++
      zeroBiasPrelude convBias enetBiasWidths ++ code ++
      s!"    return {String.intercalate ", " outNames} : {String.intercalate ", " outTypes}\n"
  let sigList := enetSig nClasses convBias
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, ds) => s!"%{n}: {ty ds}")) ++
    s!", %onehot: {ty [B, nClasses]}"
  let outSig := String.intercalate ", " (sigList.map (fun p => ty p.2))
  let inner : String := go.run' (0, [])
  "module @m {\n" ++
  s!"  func.func @{funcName}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

-- ════════════════════════════════════════════════════════════════
-- § The AdamW tail — one proven triple per parameter, folded in signature order
-- ════════════════════════════════════════════════════════════════

/-- The driver's **variant slug** for a given `(B, replicas)`: the artifact is
    `verified_mlir/efficientnet_<variant>_train_step.mlir`, the entry point is
    `@efficientnet_<variant>_train_step`, and `LEAN_MLIR_VARIANT` selects it.

    All three must agree, or the shim refuses the call outright ("entry mismatch") rather than
    running the wrong graph. Deriving the name here is what stops it drifting from the
    `#eval` paths below; the `#guard`s at the bottom pin those literal paths against this function.

    `B = 32` is deliberately unsuffixed, so the two existing artifacts keep their names and bytes.
    Same convention as `r34AdamVariant`. -/
def enetAdamVariant (B replicas : Nat) (opt : OptKind := .adamw) (ema : Bool := false)
    (sd : Bool := false) (cd : Bool := false)
    -- `bf16` LAST — the newest axis, so appending leaves every committed spelling untouched.
    -- It MUST reach this function, not merely the block renderers: the entry NAME derives from
    -- it, and a flag reaching the emission but not the name writes `…bf16_train_step.mlir`
    -- declaring `@…_train_step` inside, which the driver refuses at load ("entry mismatch").
    -- This net has the most crowded marker space in the repo — `ema`/`drop`/`do`/`dp` — and
    -- `tests/TestVariantPredicates.lean` exists because the collisions are between PAIRS of
    -- markers. `bf16` collides with none of them (no "ema" prefix, no "do", no "sd", no "acc").
    (bf16 : Bool := false)
    -- `wx` (no decay on BN γ/β and biases) and the BN-ε marker (`bnEpsMarker`), TRAILING and
    -- defaulted for `bf16`'s reason. They print after `do` and before `bf16`: `…dropdowxeps0001bf16`.
    (wx : Bool := false) (epsMarker : String := "") : String :=
  -- THE STOCHASTIC-DEPTH MARKER IS `"drop"`, NOT `"sd"`.
  -- `"sd"` collides: `rms` ++ `dp` spells **`rmsdp`**, which CONTAINS "sd", so a `"sd"` substring
  -- test fires on `rmsdp64` and `emarmsdp64` — every RMSProp DATA-PARALLEL variant, including the
  -- committed and gated `efficientnetin_rmsdp64`. No placement of an `sd` marker avoids that; the collision
  -- is between two OTHER markers meeting; `drop` collides with nothing.
  --
  -- The predicate table in `tests/TestVariantPredicates.lean` is run rather than reasoned
  -- about for the same reason: with three markers the collisions are between PAIRS, which is not
  -- something you see by reading one name at a time.
  --
  -- The marker TRAILS: a leading one would break `variant.startsWith "ema"` (the 4-region
  -- blob test) — `dropema` does not start with "ema".
  -- The `ema` marker LEADS, because the driver keys its 4-region `[θ|m|v|ema]` blob layout off
  -- `variant.startsWith "ema"` — the same reverse-of-this-function reading it uses for `"rms"`.
  -- Optimizer and EMA are independent axes here (unlike ConvNeXt, which has only AdamW), so the
  -- name carries both: `emarms` is RMSProp + EMA, which IS the EfficientNet reference's recipe.
  (if ema then "ema" else "") ++
  (match opt with
   | .adamw   => if replicas ≤ 1 then "adam" else "adamdp"
   | .rmsprop => if replicas ≤ 1 then "rms"  else "rmsdp") ++
  (if B == 32 then "" else toString B) ++
  (if sd then "drop" else "") ++
  -- AND THE CLASSIFIER-DROPOUT MARKER IS `"do"`, WHICH IS A CHOICE, NOT A DEFAULT.
  -- The obvious spelling is `"dropout"`, and it is unusable: it CONTAINS `"drop"`, so the driver's
  -- `variant.splitOn "drop"` test — which is how stochastic depth is detected — would fire on a
  -- dropout-only render and try to pack nine mask slots that graph does not have. The rule,
  -- as with `emarms`/`rmsdp` and `"sd"`/`"drop"`:
  -- **with N markers the collisions are between PAIRS, so a new marker must be checked against
  -- every existing one, not read on its own.** `tests/TestVariantPredicates.lean` runs that check
  -- rather than reasoning about it.
  -- It TRAILS `drop` for the same reason `drop` trails everything: a leading marker would break
  -- `variant.startsWith "ema"`, the driver's 4-region `[θ|m|v|ema]` blob test.
  (if cd then "do" else "") ++
  -- `bf16` LAST, after even the `do` marker — the newest axis, so appending is the only
  -- placement that leaves every committed spelling byte-identical.
  -- A flag threaded into this function's signature but missing from this line still reaches the
  -- parameter and the `fname` site, and the variant still returns `rms64` — so the artifact lands
  -- at `…rms64bf16_train_step.mlir` declaring `@efficientnetin_rms64_train_step` inside: the flag
  -- reaches the name function and the name function ignores it. The `#guard`s below catch that
  -- before anything runs.
  (if wx then "wx" else "") ++ epsMarker ++
  (if bf16 then "bf16" else "")

/-- **EfficientNet-B0 AdamW train step rendered from the verified AST.**

    Same backward as `efficientnet_train_step` (`enetBackAll`, one traversal) but taking the
    **un-fused gradients**, each fed to the proven AdamW triple. Two things differ from the SGD
    render and both are load-bearing:

    * the cotangent is **label-smoothed** (α = 0.1, K = nClasses) with an **explicit ÷B**, where the
      SGD render is plain CE with the mean folded into `lr`. Get it wrong and the tie fails in a
      way that looks like a bug in the gradient ops.
    * it returns the **BN running statistics** — batch μ/var per BN layer, `bnBatchMeanB`/
      `bnBatchVarB` recomputed from that layer's BN input — which the host EMAs into
      `@efficientnet_fwd_eval`'s frozen stats. The SGD render has no such outputs.

    Interface: 889 in (`%x`, 262 θ, 262 m, 262 v, `%lr`/`%bc1`/`%bc2`, 98 running-stat slots,
    `%onehot`) / 887 out (262 θ', 262 m', 262 v', `%loss`/`%bc1`/`%bc2`, 98 batch stats) —
    the positional layout `trainAdamSched`'s packed `[θ|m|v]` protocol reads.

    Unlike ViT's, this tie can pin the **forward bit-exactly**: EfficientNet has BatchNorm, so the
    returned batch statistics are a whole-net forward fingerprint no gradient touches. -/
def efficientnetAdamTrainStepFaithful (B nClasses : Nat) (epsStr : String)
    (alphaStr negAlphaKStr bStr : String) (replicas : Nat := 1)
    (convBias : Bool := false) (slug : String := "efficientnet")
    (opt : OptKind := .adamw) (ema : Bool := false) (sd : Bool := false)
    (cd : Bool := false)
    -- **bf16**, TRAILING and defaulted so every existing render is byte-identical. EfficientNet
    -- needs **ZERO new ops** — every conv and depthwise kind it uses on the AdamW/RMSProp path has
    -- a bf16 twin shared with MobileNetV2 and MobileNetV4. The **squeeze-excitation** stays f32 ON
    -- PURPOSE: its 1×1s act on 1×1-spatial pooled tensors, where there is no bf16 win to have.
    -- `seBlock` / `seReduceB` / `seBackBatched` are bundled ops and are simply not rewritten. The
    -- classifier dense, every BN, the loss and the optimizer stay f32 too.
    (bf16 : Bool := false)
    -- `forceSync`: the sync-BN graph at ONE replica (every collective empty), for the numeric
    -- gate `efficientnet-syncbn-check`. Never a committed artifact.
    (forceSync : Bool := false)
    -- `wdExclude` — `wx`, no decay on the 1-D parameters (BN γ/β, biases): `wdNameBy`'s rule,
    -- the JAX `wdExcludeNormBias` mask. TRAILING and defaulted: every committed render is untouched.
    (wdExclude : Bool := false) : String :=
  let sync : Bool := replicas > 1 || forceSync
  -- `negAlphaKStr` is DERIVED from `nClasses` when empty. Passing −α/K as a string independent
  -- of K is the two-writers-for-one-fact shape: a K=10 constant in a K=1000 render sits ON THE
  -- GRADIENT PATH, and a positive-signed copy in a report-only loss hides from a grep for the
  -- cotangent's.
  -- Empty ⇒ derived, so the K=1000 spelling cannot be got wrong; non-empty ⇒ honoured verbatim,
  -- which keeps every committed Imagenette artifact byte-identical.
  let negAlphaKStr := if negAlphaKStr.isEmpty then "-" ++ alphaOverK nClasses 0.1 else negAlphaKStr
  let sigList := enetSig nClasses convBias
  -- `go` hands back the BN channel counts alongside the body, so the argument signature is built
  -- from the SAME list the traversal walked. Deriving the 49 slots independently would be a second
  -- source for the stat layout — and a misaligned stat slot is silent: the arities still match and
  -- the wrong layer's statistics simply flow into the wrong `@efficientnet_fwd_eval` slot.
  let go : StateM Proofs.StableHLO.EmitS (String × List Nat) := do
    let (code, gradNames, nSm, bnList, bnSt) ←
      enetBackAll B nClasses epsStr "0.0" true (some (alphaStr, negAlphaKStr, bStr)) convBias sd cd bf16
        replicas sync
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT. `den` is the
    --     same `bnMean`/`bnVar` `bnBatchF` normalises by, so these ARE the statistics the forward
    --     used rather than a separately-derived approximation. ═══
    let mut statCode := ""
    let mut statNames : List String := []
    let mut statTypes : List String := []
    -- At `replicas > 1` they are read off the all-reduced packed vector (`bnStatsMeanB` /
    -- `bnStatsVarB`), so the host EMAs the GLOBAL batch statistics rather than replica 0's shard's.
    for ((xn, _sp, oc, hh), st) in bnList.zip bnSt do
      let zb : Vec (B * (oc * (hh * hh))) := fun _ => 0
      let zst : Vec (oc + oc) := fun _ => 0
      let (cM, nM) ← if sync then pretty B (.bnStatsMeanB (oc := oc) (.operand st zst))
        else pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
      let (cV, nV) ← if sync then pretty B (.bnStatsVarB (oc := oc) (.operand st zst))
        else pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
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
      let wdN := wdNameBy wdExclude nm ds
      let (c, nT, nM, nV) ← match opt with
        | .adamw   => adamOne B replicas ⟨nm, gradNames[i]!, ds⟩ wdN
        | .rmsprop => rmsOne  B replicas ⟨nm, gradNames[i]!, ds⟩ wdN
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]; mN := mN ++ [nM]; vN := vN ++ [nV]
      -- THE EMA SHADOW, emitted HERE rather than inside the two `*One` helpers, because it reads
      -- `nT` — the UPDATED parameter — and both tails produce one. A copy in each helper would be
      -- the double-writer disease one level down, in code, and it would have to be kept in step
      -- across an optimizer axis that already exists.
      --
      -- `Proofs.adamMNext β₁ m g = β₁·m + (1−β₁)·g` IS the reference's `ema_update` at
      -- `(β₁ := d, m := ema, g := θ')`, so this is **no new op** — `adamMNextF_faithful` closes the
      -- denotation side by `rfl`. `%emad`/`%oemad` are ARGS, not constants: the reference's decay is
      -- time-varying, `d = min(decay, (1+t)/(10+t))`.
      --
      -- At `ema := false` no `pretty` call happens, so the fresh-name counter does not move and
      -- every committed artifact re-renders byte-identically — the inertness gate, for free.
      if ema then
        let n := ds.foldl (· * ·) 1
        let z : Vec n := fun _ => 0
        let (cE, nE) ← pretty B (.adamMNextF s!"%{nm}e" "%emad" "%oemad" ds 0 z (.operand nT z))
        adamCode := adamCode ++ cE
        eN := eN ++ [nE]
    -- `%loss`: the report-only smoothed CE (`reportSmoothedCeLoss`).
    let lossCode := reportSmoothedCeLoss B nClasses nSm
    let pTy := sigList.map (fun p => ty p.2)
    -- THE RETURN LAYOUT MUST MIRROR THE INPUT LAYOUT, tensor for tensor. The driver does `pbuf :=
    -- out` — each step's output IS the next step's input (the no-copy handover) — and the shim's G4
    -- guard checks the counts agree. So the drop scales RIDE THROUGH unread, exactly as
    -- `%bc1`/`%bc2` and `%emad`/`%oemad` already do. Omitting them is not a subtle error: G4
    -- refuses the call with "returns 740 outputs, caller supplied 749 destinations" before a single
    -- step runs. Loud, and the right way round.
    let dpNames := (if sd then enetDropIdxs.map dpName else []) ++ (if cd then [doName] else [])
    let dpTys   := (if sd then enetDropIdxs.map (fun _ => ty [B]) else [])
                     ++ (if cd then [ty [B, enetHeadWidth]] else [])
    let retVals := thetaN ++ mN ++ vN ++ eN ++ ["%loss", "%bc1", "%bc2"]
                     ++ (if ema then ["%emad", "%oemad"] else []) ++ statNames ++ dpNames
    let retTys := packedTrainRetTys pTy (ema := ema) ++ statTypes ++ dpTys
    pure (
      (if replicas ≤ 1 then
        s!"    // ── EfficientNet-B0 AdamW train step: {trainStepHandNote} ──\n"
       else
        s!"    // ── EfficientNet-B0 AdamW train step, DATA-PARALLEL over {replicas} replicas ──\n" ++
        s!"    // {trainStepHandNote}.\n" ++
        syncBnBanner "EnetSyncTieG.efficientnet_net_syncTiedG"
          "StableHLO.efficientnetFwdGraphSyncFull_shard" "EfficientNet" ++
        (if sd || cd then
          "    // (Both are stated without drop-path and dropout; this artifact's per-example masks\n" ++
          "    // are not in that statement.)\n"
         else "") ++
        (if bf16 then syncBnBf16TwinsNote else "")) ++
      (match opt with
       | .adamw => ""
       | .rmsprop =>
         "    // ── OPTIMIZER: RMSProp + momentum, TENSORFLOW flavour (EfficientNet's own:\n" ++
         "    //    jax/MainEfficientNetImagenet.lean). Per parameter, in this order:\n" ++
         "    //      g  <- g + wd*θ        COUPLED L2, BEFORE the accumulator  (momVNextF)\n" ++
         "    //      s' <- ρ*s + (1-ρ)*g²                                      (adamVNextF at ρ)\n" ++
         "    //      b' <- μ*b + g/sqrt(s' + ε)   ε INSIDE the sqrt            (rmsBufNextF)\n" ++
         "    //      θ' <- θ - lr*b'                                           (sgdParamF)\n" ++
         "    //    ε = 1e-3 here, against MobileNetV2's 1.0 — this is the SENSITIVE end of the\n" ++
         "    //    placement: textbook g/(sqrt(s')+ε) takes a ~31.6x larger step at a collapsed\n" ++
         "    //    mean-square, which is what the reference means by \"vanilla diverges at the\n" ++
         "    //    paper LR\" and why it carries no gradient clipping.\n" ++
         "    //    Packed [θ|m|v] reused with m = momentum buffer, v = mean-square; %bc1/%bc2 are\n" ++
         "    //    Adam bias corrections, unread here and passed through unchanged.\n" ++
         "    //    The mean-square must be INITIALISED TO 1.0, not 0 — part of the recipe.\n") ++
      zeroBiasPrelude convBias enetBiasWidths ++ code ++ statCode ++
      (match opt with | .adamw => adamWConsts | .rmsprop => rmsConstsBlock enetRmsHyper) ++
      wdzConst wdExclude ++ adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n",
      bnList.map (fun t => t.2.2.1))
  let (inner, bnOc) := go.run' (0, [])
  -- The 49 BN layers each get a dummy `[oc]` input slot, unused by the graph — they exist so the
  -- driver's argument buffer keeps the shape the generic FFI hands it, and the hand-written render
  -- has the same unused slots. It names them after its BN-layer prefixes; `pretty`'s intermediates
  -- are counter-named, so these are indexed instead. Inputs only: the returned statistics are
  -- recomputed above from each layer's BN input.
  let statSig := String.intercalate ", " ((List.range bnOc.length).map (fun i =>
    s!"%bnmu{i}i: {ty [bnOc[i]!]}, %bnvar{i}i: {ty [bnOc[i]!]}"))
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    packedTrainSig (sigList.map fun (n, ds) => (s!"%{n}", ty ds)) (ema := ema) ++ ", " ++ statSig ++
    -- The drop scales go LAST, after the BN stats and before `%onehot` is appended, matching
    -- `enetFwdSig`'s placement — inserted mid-list they would capture an existing positional slot,
    -- which is silent until the driver mis-walks the blob.
    enetDropSig B sd ++ enetDropoutSig B cd ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (fun p => ty p.2)
  let outSig := String.intercalate ", "
    (packedTrainRetTys pTy (ema := ema)
     ++ bnOc.flatMap (fun oc => [ty [oc], ty [oc]])
     ++ (if sd then enetDropIdxs.map (fun _ => ty [B]) else [])
     ++ (if cd then [ty [B, enetHeadWidth]] else []))
  -- The entry name must track the driver's `{slug}_{variant}_train_step` convention, or the shim
  -- refuses the call ("entry mismatch"). `enetAdamVariant` is the single source for the name, the
  -- artifact path and `LEAN_MLIR_VARIANT`.
  let fname := s!"{slug}_{enetAdamVariant B replicas opt ema sd cd bf16 wdExclude (bnEpsMarker epsStr)}_train_step"
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
-- stats are threaded) from the SAME `enetFwdChain` the train steps differentiate.
--
-- The `.train` artifact is a byte-identical PREFIX of `efficientnet_train_step.mlir`, which
-- `scripts/regen_verified_mlir.sh check` audits — the check that catches a net trained with
-- per-example BN and scored with batch statistics. Both EfficientNet sides are batch-BN, and the
-- prefix audit is what keeps it that way.
#eval IO.FS.writeFile "verified_mlir/efficientnet_fwd.mlir"
  (Proofs.StableHLO.efficientnetFwdFaithfulV 32 10 "1.0e-5")

#eval IO.FS.writeFile "verified_mlir/efficientnet_fwd_eval.mlir"
  (Proofs.StableHLO.efficientnetFwdEvalFaithfulV 32 10 "1.0e-5")

-- The **AdamW** train step — **the artifact `efficientnet-verified-adam` trains on**, and this
-- `#eval` is its ONLY writer; `tests/TestEfficientNetTrain.lean` only iree-compiles the committed
-- bytes. The driver resolves the path from the net slug.
--
-- Two writers for one artifact is a last-writer-wins race, so a candidate peer renders to its own
-- path and is tied with `lake build efficientnet-adam-tie` (one AdamW step, all 12,166,117
-- returned floats: forward BIT-EXACT over the 98 BN batch statistics, `%loss` bit-exact, gradient
-- bit-exact, against a bit-exact A-vs-A determinism floor).
--
-- Literals: α = 0.1, −α/K = −0.01 (K = 10), batch 32 — `efficientNetB0Config`'s label smoothing
-- and explicit mean.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adam_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0")

-- The **Imagenette bf16 twin** of the row above — same 32/10/α/−α÷K literals, only the cast
-- differs. It exists as the cheap END-TO-END sanity check on the bf16 + 4-D-pointwise path:
-- Imagenette has a known 80-epoch result (`historical/RESULTS.md`), with a per-epoch trajectory to
-- compare against, where an ImageNet run costs days.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adambf16_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" (bf16 := true))
#guard Proofs.StableHLO.enetAdamVariant 32 1 .adamw false false false true == "adambf16"

-- The **DATA-PARALLEL** render, selected at run time by `LEAN_MLIR_VARIANT=adamdp`. Same graph,
-- plus one `all_reduce(add)/N` per parameter gradient between the certified gradient and the
-- certified AdamW triple: *certified gradient → trusted collective → certified AdamW*. The
-- collective is a DECLARED carve-out and the render says so in its own output banner at `replicas >
-- 1`, because an undeclared carve-out is how wrong things ship.
--
-- The certified renderer is the only writer of both EfficientNet AdamW artifacts.
--
-- It renders to its OWN path, so producing a DP render never means editing a knob and clobbering
-- the artifact the trainer runs. `2` is the replica count these are rendered at and it must match
-- `PJRT_REPLICAS` at run time, because the graph bakes `replica_groups`. Re-render here to change
-- it.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adamdp_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" 2)

-- The **bs128** pair, single-device and 2-replica. `B` is a true parameter of
-- the renderer, so this is the whole change: the graph structure is identical and only the tensor
-- dimensions and the mean-CE divisor (32.0 → 128.0) move. At 2 replicas this is a GLOBAL batch of
-- 256, the batch ImageNet wants.
--
-- They render to their OWN paths, so the bs32 artifacts and their tie/DP-gate baselines are
-- untouched. Select with `LEAN_MLIR_VARIANT=adam128` / `adamdp128` and
-- `LEAN_MLIR_BATCH=128`. **The eval forwards are still bs32**, so train with `LEAN_MLIR_SKIP_EVAL=1`
-- or re-render them — the same caveat as R34's bs256.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adam128_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 128 10 "1.0e-5"
    "0.100000" "-0.010000" "128.0")

#eval IO.FS.writeFile "verified_mlir/efficientnet_adamdp128_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 128 10 "1.0e-5"
    "0.100000" "-0.010000" "128.0" 2)

-- ── EfficientNet-B0 on FULL 1000-class ImageNet, slug `efficientnetin` ────────────────────
-- The EfficientNet peer of `resnet34in_*`, `vitin_*` and `convnextin_*`. `B`, `nClasses` and the
-- `slug` are renderer parameters, and −α/K is derived above.
--
-- **The slug is load-bearing and EfficientNet is the case where it bites hardest**: it is the
-- only net here with BOTH a `_fwd` and a `_fwd_eval` artifact, and neither carries a variant in its
-- path. A 1000-class forward emitted under the `efficientnet` slug would silently overwrite the
-- 10-class pair that the Imagenette runs (`runs/2026-08-31-imagenette-n3/`), the prefix audit and `fwd-tie efficientnet
-- --eval` all depend on.
--
-- Batch **64 per device × 4 replicas = global 256**, which is
-- `efficientNetB0ImagenetConfig.batchSize`. Matching the reference's global batch is what makes the
-- two runs a comparable pair rather than two experiments — the same reasoning as R34's `momdp64`.
--
-- `enetAdamVariant B replicas` encodes the PER-DEVICE batch and NOT the replica count, so
-- `adamdp64` would name both a 2-replica and a 4-replica render at B=64. Only the 4-replica one is
-- emitted here, so nothing collides; anyone adding the 2-replica peer must rename first.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_fwd.mlir"
  (Proofs.StableHLO.efficientnetFwdFaithfulV 64 1000 "1.0e-5" false "efficientnetin")
#eval IO.FS.writeFile "verified_mlir/efficientnetin_fwd_eval.mlir"
  (Proofs.StableHLO.efficientnetFwdEvalFaithfulV 64 1000 "1.0e-5" false "efficientnetin")
#eval IO.FS.writeFile "verified_mlir/efficientnetin_adam64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 1 false "efficientnetin")
#eval IO.FS.writeFile "verified_mlir/efficientnetin_adamdp64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 4 false "efficientnetin")

-- ── RMSProp: the optimizer the EfficientNet reference ACTUALLY USES ─────────────────────────
-- The EfficientNet reference's top-1 (`planning/imagenet_parity.md`) is an RMSProp number. ρ = μ = 0.9,
-- **ε = 1e-3**, wd = 1e-5 — `Proofs.StableHLO.enetRmsHyper`.
--
-- **ε = 1e-3 is the SENSITIVE end of the ε-placement difference**, unlike MobileNetV2's ε = 1.0.
-- At a collapsed mean-square the textbook spelling steps 31.6× larger, which is exactly what the
-- reference config means by *"vanilla diverges/erodes at the paper LR, the TF form trains stably"*
-- and why its `gradClipNorm` is 0. So this net is where the placement decides the outcome, and it gets
-- its own numeric gate — `rms-tie efficientnet` — rather than inheriting mnv2's.
--
-- The Imagenette-shape render below differs from `efficientnet_adam_train_step` in EXACTLY ONE
-- thing, the optimizer: same B, K, ε, α, −α/K and ÷B literals. That is what makes a tie against it
-- attributable (a candidate that differs in two ways cannot license either).
#eval IO.FS.writeFile "verified_mlir/efficientnet_rms_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" 1 false "efficientnet" .rmsprop)

-- ── RMSProp **+ EMA** — this net's ACTUAL reference recipe ────────────────
-- `efficientNetB0ImagenetConfig` is RMSProp + exp-decay + **EMA (decay 0.9999)** + dropPath, and
-- its reference top-1 is the EMA shadow's number.
--
-- The variant is `emarms`, not `ema`: optimizer and EMA are INDEPENDENT axes on this net (unlike
-- ConvNeXt, which has only AdamW), so the name carries both — and the `ema` marker LEADS because
-- the driver keys its 4-region `[θ|m|v|ema]` layout off the prefix.
--
-- EfficientNet is the first EMA net with **BatchNorm**, and the reference shadows the BN running
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
-- Two driver-side requirements, neither a render change: the mean-square slot must be
-- INITIALISED TO 1.0 (TF's convention — a zero init is not a crash, it is a different and much
-- larger first step), and the LR schedule must be exponential 0.97/epoch rather than cosine.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_rms64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 1 false "efficientnetin" .rmsprop)

-- **The bf16 peer** — `rms64bf16`. RMSProp is EfficientNet's own optimizer, and this is the
-- SINGLE-DEVICE render, deliberately: a 4-replica bf16 number on this box is a SYSTEM result (shim
-- feed + f32 all-reduce), not a statement about the emit — MobileNetV2's bf16 gain shrinks sharply
-- from one GPU to four on the same graph (planning/archive/bf16_renderer.md). A 1-GPU pair is the
-- measurement that isolates the renderer.
--
-- **EfficientNet needs ZERO new ops.** Every conv and depthwise kind on its AdamW/RMSProp path
-- has a bf16 twin shared with MobileNetV2 (8 ops) and MobileNetV4 (3 ops) — 23 call
-- sites, nothing new to build or prove.
--
-- The **squeeze-excitation stays f32 on purpose**: its 1×1s act on 1×1-spatial pooled
-- tensors, where there is no bf16 win to have. `seBlock`/`seReduceB`/`seBackBatched` are bundled
-- ops and are simply never rewritten. The classifier dense, every BN, the loss and the optimizer
-- stay f32 too — the same carve-out every other bf16 render in this repo makes.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_rms64bf16_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 1 false "efficientnetin" .rmsprop (bf16 := true))

-- The bf16 marker. This net has the repo's most crowded marker space — `ema`/`drop`/`do`/`dp`
-- — and `tests/TestVariantPredicates.lean` exists because the collisions are between PAIRS of
-- markers (`rms` ++ `dp` spells `rmsdp`, which contains "sd"). `bf16` collides with none of them.
#guard Proofs.StableHLO.enetAdamVariant 64 1 .rmsprop false false false true == "rms64bf16"
#guard Proofs.StableHLO.enetAdamVariant 64 1 .rmsprop == "rms64"
#guard !"rms64bf16".contains "do"
#guard !"rms64bf16".contains "acc"
#guard !"rms64bf16".contains "sd"
#guard !"rms64bf16".startsWith "ema"
#eval IO.FS.writeFile "verified_mlir/efficientnetin_rmsdp64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 4 false "efficientnetin" .rmsprop)

-- **The 4-replica bf16 peer** — the DP arm of `rms64bf16`, so B0 can be probed at the same 4×bs64
-- geometry as R34/R50/MNv2 rather than only single-device. Read its ms/step as a SYSTEM number, not
-- a renderer one: a 4-replica figure carries the shim feed and the f32 all-reduce. B0's RENDERER
-- number is the single-device one (planning/archive/bf16_renderer.md), and that is the one that
-- says what the emit is worth.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_rmsdp64bf16_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 4 false "efficientnetin" .rmsprop (bf16 := true))

-- Pin the two literal artifact paths above against the name the renderer actually emits. If a
-- variant is renamed this fails at `lake build` instead of at run time as an "entry mismatch".
#guard Proofs.StableHLO.enetAdamVariant 32 1 == "adam"
#guard Proofs.StableHLO.enetAdamVariant 32 2 == "adamdp"
#guard Proofs.StableHLO.enetAdamVariant 128 1 == "adam128"
#guard Proofs.StableHLO.enetAdamVariant 128 2 == "adamdp128"
-- The RMSProp peers. Distinct slugs from the AdamW ones is the point: rendering the other
-- optimizer must never overwrite the artifact the AdamW trainer runs (a last-writer-wins race).
#guard Proofs.StableHLO.enetAdamVariant 32 1 .rmsprop == "rms"
#guard Proofs.StableHLO.enetAdamVariant 64 1 .rmsprop == "rms64"
#guard Proofs.StableHLO.enetAdamVariant 64 4 .rmsprop == "rmsdp64"
#guard Proofs.StableHLO.enetAdamVariant 64 1 .adamw   == "adam64"
-- The EMA peers. The marker LEADS so the driver's `startsWith "ema"` finds it whichever
-- optimizer it is paired with, and `emarms` — RMSProp + EMA — is this net's reference recipe.
#guard Proofs.StableHLO.enetAdamVariant 32 1 .rmsprop true == "emarms"
#guard Proofs.StableHLO.enetAdamVariant 32 1 .adamw   true == "emaadam"
#guard Proofs.StableHLO.enetAdamVariant 64 4 .rmsprop true == "emarmsdp64"
-- ...and OFF it is byte-identical to the plain spelling, which is what keeps the inertness gate free.
#guard Proofs.StableHLO.enetAdamVariant 32 1 .rmsprop false == "rms"

-- Both arities pinned, so dropping the conv biases cannot silently change the render that
-- ships. 262 − 49 = 213; the SE biases (`zb1`/`zb2`) are NOT among the 49, because those convs are
-- followed by an activation rather than BN and the reference carries them too.
#guard (Proofs.StableHLO.enetSig 10 true).length == 262
#guard (Proofs.StableHLO.enetSig 10 false).length == 213

-- ── STOCHASTIC DEPTH, selected by `LEAN_MLIR_VARIANT=adamsd` ──
-- EfficientNet-B0 is the cheapest net to carry stochastic depth, and the reason is the INDEX
-- CONVENTION, not the op count:
--
--   renderer            batched forms   per-example forms
--   ResNet34RenderB              36            0
--   MobileNetV2RenderB           53            0
--   EfficientNetRender           45            0     ← at N := B, so the op drops straight in
--   ConvNeXtRender                2 (N := 1)  10     ← per-example
--   ViTRender                     2           13     ← per-example
--
-- A per-example mask CANNOT be expressed honestly at the per-example index: `den` at index `n`
-- describes one example, so the mask becomes the descriptor trap ("a descriptor may carry only
-- batch-INVARIANT data"). ConvNeXt and ViT therefore need the batched-index move FIRST, and that
-- move is expensive. EfficientNet needs none of it: scope by the INDEX CONVENTION, not by op
-- count.
--
-- 9 sites, not 16, and the ramp index is the BLOCK index — see `enetDropIdxs`.
-- The scale is a graph INPUT with `1/keep_i` FOLDED IN by the driver, never `stablehlo.rng` and
-- never a baked constant. `Proofs.dropPath`'s note has the argument; the short version is that a
-- baked `1/keep_i` and "the forward emits the sites too" cannot both hold, because a ones
-- mask would then compute `x/keep_i` rather than `x`, and the reference is explicit that eval
-- returns the branch untouched.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adamdrop_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" (sd := true))

-- **THE DATA-PARALLEL PEER — and it exists to be GATED.**
-- The mask is a PER-EXAMPLE input, so under data parallelism it must be **SHARDED like `x`**, not replicated like the parameters.
--
-- The masks ride in the parameter blob (`Verified.Train`'s `dropShapes` are appended to
-- `adamShapes`), and a DP shim that marks exactly `x` and the labels sharded and everything between
-- them replicated (`iree_lean_ffi.c`) hands every replica replica 0's mask, applied to its OWN
-- rows. `pjrt_ffi_invoke_f32_dp2` takes a sharded-tail count, and `lake build drop-shard-check` is
-- the gate.
--
-- 2 replicas, deliberately: the gate's known answer is that swapping the two mask halves leaves
-- a correctly-sharded DP step BIT-IDENTICAL, which rests on f32 addition being COMMUTATIVE
-- (`a+b == b+a` exactly). At more than two replicas the reduction is a tree whose ORDER changes
-- under a permutation, and associativity does not hold — so the known answer is exact at 2 and
-- only approximate above it.
#eval IO.FS.writeFile "verified_mlir/efficientnet_adamdpdrop_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" (replicas := 2) (sd := true))

-- The forward peers. These exist so the SD variant has its OWN `forward ⊂ train-step` pair, because
-- the prefix audit is one of the two load-bearing structural gates in the repo (it catches a `_fwd`
-- scoring a net it did not train). The alternative — let the SD trainer eval through the drop-free
-- `efficientnet_fwd` — is what the reference literally does, but it would leave the SD train step
-- with no prefix partner at all, i.e. spend the gate rather than pay 9 dead multiplies. At eval the
-- driver supplies an all-ones scale, so these compute the identity EXACTLY
-- (`Proofs.dropPath_ones_id`, and `1 * x = x` is exact in IEEE).
#eval IO.FS.writeFile "verified_mlir/efficientnet_drop_fwd.mlir"
  (Proofs.StableHLO.efficientnetFwdFaithfulV 32 10 "1.0e-5" false "efficientnet_drop" (sd := true))

#eval IO.FS.writeFile "verified_mlir/efficientnet_drop_fwd_eval.mlir"
  (Proofs.StableHLO.efficientnetFwdEvalFaithfulV 32 10 "1.0e-5" false "efficientnet_drop" (sd := true))

-- ── THE IMAGENET PEERS of the EMA and stochastic-depth renders ────────────────────────
-- A feature is not "done" when its Imagenette artifact renders. Both scales are one `#eval` apart
-- (`nClasses`/`B`/`slug` are ordinary parameters), which is exactly why it is easy to stop at one
-- and not notice; LIST the artifacts rather than reasoning about them.
--
-- `emarms64` is the reference's ACTUAL recipe at ImageNet scale — RMSProp + exponential decay +
-- EMA. The config also sets `dropout := 0.2` (`jax/MainEfficientNetImagenet.lean`); the render
-- that carries it too is `efficientnetin_emarms64dropdo` below.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_emarms64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 1 false "efficientnetin" .rmsprop (ema := true))
#eval IO.FS.writeFile "verified_mlir/efficientnetin_emarmsdp64_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 4 false "efficientnetin" .rmsprop (ema := true))

-- Stochastic depth at ImageNet scale. Same 9 sites and the same block-index ramp — `enetDropIdxs`
-- is a property of the ARCHITECTURE (16 MBConv blocks, 9 with skips), not of the class count or the
-- batch, so `efficientnetin` reuses it unchanged and `tests/TestDropPathRamp.lean` covers both.
-- THE PATH MUST SPELL THE VARIANT EXACTLY. `enetAdamVariant 64 1 .rmsprop true true` emits
-- **`emarms64drop`** — the batch suffix precedes the regulariser markers. The driver derives the
-- artifact path from `variant` (`VerifiedNet.trainAdamSched`) AND the entry name from the same
-- `variant` (`:868`), so a path spelled `emarmsdrop64` would be unloadable: the driver finds the
-- file and asks for an entry it does not contain. Every structural gate misses that — the prefix
-- audit reads the file, and nothing loads it through the driver — so
-- `scripts/regen_verified_mlir.sh check` audits basename == entry across all artifacts.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_emarms64drop_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 1 false "efficientnetin" .rmsprop (ema := true) (sd := true))
#eval IO.FS.writeFile "verified_mlir/efficientnetin_drop_fwd.mlir"
  (Proofs.StableHLO.efficientnetFwdFaithfulV 64 1000 "1.0e-5" false "efficientnetin_drop" (sd := true))

-- ── CLASSIFIER DROPOUT ──────────────────────────────────
-- `efficientNetB0ImagenetConfig` sets `dropout := 0.2`; these renders carry it.
--
-- IT IS NOT STOCHASTIC DEPTH AT A DIFFERENT SITE. The reference draws
-- `bernoulli(key, keep, x.shape)` — one Bernoulli per (example, feature) — where stochastic depth
-- draws `(B, 1, …, 1)`. Same op (`layerScale`), different mask RANK, and each is what the other's
-- comments have spent this file warning about. `Proofs.dropout_of_dropScale` states the
-- containment and `Proofs.dropPath_scales_uniformly` the gap; `tests/TestBatchedEmitTie.lean` pins
-- both directions in the emitted bytes.
--
-- ONE site and NO ramp — so `enetDropIdxs`' expensive block-index/site-ordinal distinction has
-- no analogue here, and no silent-constant ramp bug is spellable. What replaces it
-- is the weight-gradient operand (`ENetFwd.cin`), which no ones-mask gate can see.

-- The Imagenette AdamW peer: dropout alone, so it pairs with `efficientnet_adam` for the keep = 1
-- tie. That tie is this feature's floor measurement — at an all-ones mask the two renders must
-- agree BIT-EXACTLY, because `1 * x = x` is exact in IEEE (`Proofs.dropout_ones_id`).
#eval IO.FS.writeFile "verified_mlir/efficientnet_adamdo_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 32 10 "1.0e-5"
    "0.100000" "-0.010000" "32.0" (cd := true))

-- The forward peers, for the same reason the SD ones exist: the dropout variant needs its OWN
-- `forward ⊂ train-step` pair, or the prefix audit quietly stops covering it. The site is emitted
-- here too and the driver supplies an all-ones mask at eval, so it is the exact identity.
#eval IO.FS.writeFile "verified_mlir/efficientnet_do_fwd.mlir"
  (Proofs.StableHLO.efficientnetFwdFaithfulV 32 10 "1.0e-5" false "efficientnet_do" (cd := true))
#eval IO.FS.writeFile "verified_mlir/efficientnet_do_fwd_eval.mlir"
  (Proofs.StableHLO.efficientnetFwdEvalFaithfulV 32 10 "1.0e-5" false "efficientnet_do" (cd := true))

-- **THE FULL REFERENCE RECIPE AT IMAGENET SCALE — `efficientNetB0ImagenetConfig` entire on the
-- regulariser axis**: RMSProp (TF flavour, ε inside the sqrt, ms-init 1.0) + EMA 0.9999 +
-- stochastic depth 0.1 + classifier dropout 0.2. The peer of ConvNeXt's
-- `convnextin_adamdpwxclipdrop`, and what the reference pair needs to be reachable through
-- the verified path.
--
-- NOTE THE PATH: `efficientnetin_emarms64dropdo`, batch suffix BEFORE the two regulariser markers,
-- because that is what `enetAdamVariant` emits and the driver derives the artifact PATH and the
-- entry NAME from the same string (`VerifiedNet.trainAdamSched`). The other order is **unloadable
-- at any `LEAN_MLIR_VARIANT`** — see the note on the stochastic-depth `#eval` above. The `#guard`s
-- at the bottom of this file pin literal paths against the function that derives them rather than
-- writing both by hand.
--
-- It carries BOTH mask families at once, which is exactly why it is worth committing rather than
-- assembling ad hoc: nine `tensor<64xf32>` per-example scales followed by one
-- `tensor<64x1280xf32>` per-element mask, in that order, and the two must not be confused by the
-- driver (which packs them), the shim (which shards the tail by COUNT) or a reader. It is the only
-- artifact in the repo where getting the mask rank wrong would be a type error rather than a silent
-- regulariser swap — which is a property of this pairing, not something to rely on elsewhere.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_emarms64dropdo_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 1 false "efficientnetin" .rmsprop (ema := true) (sd := true) (cd := true))
-- **THE SHIPPING DATA-PARALLEL RENDER — and it exists because an ImageNet run loads the DP
-- artifact, not the single-device one.** `efficientnetin_emarmsdp64` (214 all_reduce) carries
-- NEITHER stochastic depth NOR classifier dropout, and a 4-replica run on it trains without the two
-- regularisers its reference sets, silently.
--
-- A DP render can sit features behind its single-device peer. The detection method that works:
-- **list what the artifact bakes — do not read a recipe matrix**, which records a capability;
-- only the artifact records the state.
--
-- 4 replicas × batch 64 = global 256 = `efficientNetB0ImagenetConfig.batchSize`, matching
-- `efficientnetin_emarmsdp64`'s geometry exactly so the two are comparable.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_emarmsdp64dropdo_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 4 false "efficientnetin" .rmsprop (ema := true) (sd := true) (cd := true))

-- The **bf16 twin of the production job's artifact** (`scripts/jobs/enet-default-4gpu.conf`). Same
-- geometry and the same three regularisers as the f32 row above — only the cast differs — so the
-- two are a like-for-like pair the job can be flipped between.
-- With flat-activation NHWC↔NCHW relayouts in the graph B0's bf16 gain nearly vanishes, because that
-- traffic is f32 and sits BEFORE the cast; with the pointwise ops emitted 4-D (`liftPointwise`) the cast
-- pays.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_emarmsdp64dropdobf16_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-5"
    "0.100000" "" "64.0" 4 false "efficientnetin" .rmsprop (ema := true) (sd := true) (cd := true)
    (bf16 := true))
#guard Proofs.StableHLO.enetAdamVariant 64 4 .rmsprop true true true true == "emarmsdp64dropdobf16"

-- **The TF recipe, and the arm `enet-default-4gpu` runs**: the line above plus `wx` (no decay on BN γ/β or biases, as TF and timm do) at TF's BN
-- ε = 1e-3. The staircase schedule and the i/16 drop-connect ramp are driver-side
-- (`enetImagenetRmsSchedule`, `efficientnetImagenetVerified.dropKeeps`). ε is baked, so the eval
-- partner is its own artifact, `efficientnetin_fwd_eval_eps0001.mlir`.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_emarmsdp64dropdowxeps0001bf16_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 64 1000 "1.0e-3"
    "0.100000" "" "64.0" 4 false "efficientnetin" .rmsprop (ema := true) (sd := true) (cd := true)
    (bf16 := true) (wdExclude := true))
#eval IO.FS.writeFile "verified_mlir/efficientnetin_fwd_eval_eps0001.mlir"
  (Proofs.StableHLO.efficientnetFwdEvalFaithfulV 64 1000 "1.0e-3" false "efficientnetin")
#guard Proofs.StableHLO.enetAdamVariant 64 4 .rmsprop true true true true (wx := true)
    (epsMarker := Proofs.StableHLO.bnEpsMarker "1.0e-3") == "emarmsdp64dropdowxeps0001bf16"

-- The **2-GPU** peer: 2 replicas × batch 128 = the same global 256 =
-- `efficientNetB0ImagenetConfig.batchSize`, so it keeps the geometry the 4×64 render above has and
-- stays comparable to it row for row. Both regularisers ride along explicitly (`sd`, `cd`) — the
-- defect this block documents is a DP render silently sitting a feature behind its single-device
-- peer, and the way to not repeat it is to copy the flag list, not to trust that a default matches.
-- The batch is in the slug, so `emarmsdp128dropdo` cannot overwrite `emarmsdp64dropdo`.
#eval IO.FS.writeFile "verified_mlir/efficientnetin_emarmsdp128dropdo_train_step.mlir"
  (Proofs.StableHLO.efficientnetAdamTrainStepFaithful 128 1000 "1.0e-5"
    "0.100000" "" "128.0" 2 false "efficientnetin" .rmsprop (ema := true) (sd := true) (cd := true))

#eval IO.FS.writeFile "verified_mlir/efficientnetin_dropdo_fwd.mlir"
  (Proofs.StableHLO.efficientnetFwdFaithfulV 64 1000 "1.0e-5" false "efficientnetin_dropdo"
    (sd := true) (cd := true))

-- Pin the variant spellings the four paths above depend on, so a rename fails at `lake build`
-- rather than at run time as an "entry mismatch".
-- The last two are the collision checks that matter, and they are why the marker is `"do"` and
-- not `"dropout"`: `dropdo` must still contain `"drop"` exactly once as the SD marker, and a
-- dropout-only variant must NOT contain it at all. `tests/TestVariantPredicates.lean` runs the
-- full pairwise table; these three are the spellings this file commits to.
#guard Proofs.StableHLO.enetAdamVariant 32 1 .adamw false false true == "adamdo"
#guard Proofs.StableHLO.enetAdamVariant 64 1 .rmsprop true true true == "emarms64dropdo"
-- The SHIPPING DP spelling. `rmsdp` (not `rms`) because replicas > 1, then the batch, then both
-- regulariser markers — the artifact an ImageNet run at 4 replicas actually loads.
#guard Proofs.StableHLO.enetAdamVariant 64 4 .rmsprop true true true == "emarmsdp64dropdo"
#guard Proofs.StableHLO.enetAdamVariant 128 2 .rmsprop true true true == "emarmsdp128dropdo"
-- The collision checks, in the driver's own predicate (`variant.splitOn "drop"`), stated as the
-- two facts that would break if the marker were spelled `"dropout"`:
--   a dropout-ONLY variant must NOT look like a stochastic-depth one …
#guard !(Proofs.StableHLO.enetAdamVariant 32 1 .adamw false false true).contains "drop"
--   … and the combined one must look like exactly ONE stochastic-depth marker, not two.
#guard ((Proofs.StableHLO.enetAdamVariant 64 1 .rmsprop true true true).splitOn "drop").length == 2
-- And OFF, every spelling is byte-identical to the flag-free one — the inertness gate in the
-- naming layer, which is what keeps every committed artifact at 0 diff.
#guard Proofs.StableHLO.enetAdamVariant 32 1 .adamw false false false == "adam"
#guard Proofs.StableHLO.enetAdamVariant 64 1 .rmsprop true true false == "emarms64drop"
