import LeanMlir.Proofs.Codegen.StableHLO.Pretty
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Foundation.BceLossCot

/-! # The renderers' shared optimizer tail

Every batched train-step renderer ends the same way: fold a per-parameter optimizer step over the
parameter list, so the θ/m/v outputs come out in signature order. This file holds that step once:
`OptRecipe` names the optimizer and `optOne` emits one parameter's tail under it.

| piece | role |
|---|---|
| `OptRecipe` | AdamW, RMSProp, heavy-ball, SGD, LAMB, and AdamW / LAMB over `k` accumulated micro-batches |
| `optOne` | one parameter: all-reduce (DP only), the recipe's ops, the optional model-EMA shadow |
| `optAllParams` | the whole stage: the hoisted global-norm clip, then `optOne` per parameter (ResNet-34/50, MobileNetV4) |
| `optConstsB` | the recipe's baked constants |

MobileNetV2 and EfficientNet call `optOne` per parameter, and ViT and ConvNeXt call it after their
own hoisted clip (`preAvg`).

Then the train step's **packed interface** — the one positional contract the driver's blob walks:
`packedTrainSig` (arguments) and `packedTrainRetTys` (results), used by every batched renderer.
Last, the precision-switched constructors `XAt bf16 rnd …` (`convAt`, `convBackBatchedAt`, …): one
`if bf16 then .XBf16 rnd … else .X …` per op instead of one per call site.

The loss cotangent's text comes from here too: `smoothedCotB` / `bceCotB` print the capstones'
own cotangent graphs (`smoothedLossCotGraph`, `smoothedLossCotGraphDiv`, `bceLossCotGraph`) after
their softmax or sigmoid head, which each render prints first because its `%loss` report reads
that name. The hand-written text every render shares is here once as well: the report-only
`%loss` blocks (`reportSmoothedCeLoss`, `reportBceLoss`, `reportCeLossOfLogits`,
`reportCeLossOfSm`), the weight-decay exclusion (`rankWdDecays`, `wdNameBy`) and the sync-BN
banner (`syncBnBanner`). So is one shared `pretty` site, ViT's and ConvNeXt's vector LN
(`vecLnSite` / `vecLnSiteB`).

At `replicas ≤ 1` the all-reduce emits nothing and threads the raw gradient, so a single-device
render is unchanged by it. The AdamW triple consumes the averaged gradient as an `.operand`,
exactly as it consumed the raw one, so the `den` side does not shift.
-/

namespace Proofs.StableHLO

/-- A trainable parameter: emitted name (no `%`), gradient SSA name, and shape. The optimizer tail
    is a fold over this list, so the θ/m/v output order cannot drift from the signature order. -/
structure PGrad where
  nm   : String          -- parameter name without `%` (`%{nm}`, `%{nm}m`, `%{nm}v` are the args)
  grad : String          -- SSA name of its un-fused gradient
  ds   : List Nat        -- parameter shape, for the emitted optimizer ops
deriving Inhabited

/-- A BatchNorm layer's running-statistic slots `%{nm}mu`, `%{nm}var`, each `[c]`, μ before var —
    the order the driver packs `runningBnStats` in. Every render's stat signature is built from this
    one pair: a misaligned slot keeps the arities and silently feeds the wrong layer's statistics. -/
def bnStatSlots (nm : String) (c : Nat) : List (String × List Nat) :=
  [(s!"%{nm}mu", [c]), (s!"%{nm}var", [c])]

/-- A block's backward: its code, the `dx` cotangent to the previous block, and the block's
    parameter gradients in func-arg order. -/
structure BlockBack where
  code : String
  dx : String
  ps : List PGrad

/-- A stem's saved SSA names: conv out, BN out, the BN's packed statistics (`""` at one replica),
    activation out. -/
structure StemFwdB where
  code : String
  c : String
  n : String
  st : String
  o : String

-- ════════════════════════════════════════════════════════════════
-- § Weight-decay exclusion (`wx`): shared by every optimizer tail (`optOne`)
-- ════════════════════════════════════════════════════════════════

/-- **Does this parameter get weight decay?** timm's `no_weight_decay` rule, and it is the PLAIN
    RANK TEST with no name carve-out — every 1-D parameter is excluded: BN γ, BN β and every bias.
    The `wx` renders bind the excluded parameters' decay operand to `%wdz` (`wdNameBy`).

    Every net with no positional parameter uses it (ResNet, MobileNet, EfficientNet, ConvNeXt):
    the rule is timm's, not the net's. ViT's `vitWdDecays` adds its `nm != "pos"` carve-out, which
    is why the predicate takes the name it ignores here. -/
-- Why it matters: decay on pre-BN conv weights is renormalised away by BN and acts only as an
-- effective-LR control; decay on γ/β is not, because γ directly scales the layer's output. The
-- effect concentrates at low LR, i.e. in the cosine endgame. The first RSB-A3 R50 run used
-- a non-`wx` artifact, so it decayed BN γ/β and every bias at wd = 0.02 where its reference
-- (`resnet50ImagenetConfigRSBFaithful`, `wdExcludeNormBias := true`) did not.
def rankWdDecays (_nm : String) (ds : List Nat) : Bool := ds.length ≥ 2

/-- The decay operand for one parameter: the real `%wd`, or the zero constant when `wdExclude` is
    on and `decays` excludes it. -/
def wdNameBy (wdExclude : Bool) (nm : String) (ds : List Nat)
    (decays : String → List Nat → Bool := rankWdDecays) : String :=
  if wdExclude && !decays nm ds then "%wdz" else "%wd"

/-- The variant marker for a BatchNorm ε other than the committed `1.0e-5`, in the `wd`/`ls` decimal
    grammar (first digit the integer part): `eps0001` is 1e-3, the TF papers' value (MobileNetV2 in
    slim, EfficientNet). ε is baked into every BN site of the train step AND of the eval forward, so
    a different ε is a different pair of artifacts: the train step's entry carries the marker, and
    the eval forward it scores through is `<slug>_fwd_eval_<marker>` (`VerifiedVariant.evalTag`
    reads it back). -/
def bnEpsMarker (epsStr : String) : String :=
  if epsStr == "1.0e-5" then "" else if epsStr == "1.0e-3" then "eps0001" else s!"eps({epsStr})"

/-- `@<slug>_fwd_eval`, or `@<slug>_fwd_eval_<marker>` at a non-default ε (an artifact's entry is its
    file name). -/
def fwdEvalEntry (slug epsStr : String) : String :=
  match bnEpsMarker epsStr with | "" => s!"{slug}_fwd_eval" | m => s!"{slug}_fwd_eval_{m}"

/-- The banner clause of a train step: which of its lines are not `pretty` of a node.
    `acc` adds the accumulation β scalars, which the render computes from `%aup` by hand; `hand`
    names any further hand-written block, followed by `", "` (ConvNeXt's GAP backward). -/
def trainStepHandNote (acc : Bool := false) (hand : String := "") : String :=
  "every op is pretty(verified AST node) except the constants, " ++
    (if acc then "the β scalars computed from %aup, " else "") ++ hand ++
    "the input passthroughs and the marked report-only %loss"

/-- The banner of a **sync-BN data-parallel** train step, after its title line: every gradient and
    update op is `pretty` of a node, the all-reduce blocks included; BN is synchronised; and the
    step is the single-device step at the global batch, proved as `tieThm` (the all-reduced
    gradients) and `fwdThm` (the forward), both in `LeanMlir/Proofs/Nets/<dir>/`. Scope notes the
    statements leave out (drop-path masks, accumulation, bf16: `syncBnBf16WgradNote` /
    `syncBnBf16TwinsNote`) follow it, from the caller. -/
def syncBnBanner (tieThm fwdThm dir : String) : String :=
  "    // Every gradient and update op is pretty(verified AST node), the per-parameter `%arsum*` all_reduce /\n" ++
  "    // `%armean*` blocks included: pretty(allReduceMeanF), whose den is the replica MEAN of\n" ++
  "    // the per-replica gradient nodes. BatchNorm is SYNCHRONISED: every BN\n" ++
  "    // layer all-reduces its mu, then var_r + (mu_r - mu)^2 (bnBatchVarAtB, Chan's parallel\n" ++
  "    // variance), before normalising with the global [mu | var] (bnSyncF); its\n" ++
  "    // backward all-reduces the two dy-reductions (bnSyncDyStatsB -> bnSyncBack), and the gamma\n" ++
  "    // gradient reads the same global x-hat (bnSyncGammaGradB). Each replica therefore computes\n" ++
  "    // its shard of the GLOBAL-batch function, and this step IS the single-device step at the\n" ++
  s!"    // global batch N x b: proved as {tieThm} (every all-reduced\n" ++
  s!"    // gradient) and {fwdThm} (the forward), both in\n" ++
  s!"    // LeanMlir/Proofs/Nets/{dir}/.\n"

/-- The sync-BN banner's bf16 note for the nets whose conv weight gradients have a sharded bf16
    statement (ResNet, `Proofs.den_allReduceMeanF_convWeightGradBBf16_sub_global`): only the
    weight gradient's rounding differs from one device's. -/
def syncBnBf16WgradNote : String :=
  "    // (Both are stated at the f32 nodes. At bf16 the conv forward and input-VJP nodes\n" ++
  "    // still shard exactly; each conv weight gradient rounds its replica's partial sum\n" ++
  "    // before the all-reduce, where one device rounds the whole sum once. That is the\n" ++
  "    // one difference: den_allReduceMeanF_convWeightGradBBf16_sub_global and its strided\n" ++
  "    // peer, in LeanMlir/Proofs/Foundation/DataParallel/SyncBf16.lean.)\n"

/-- The sync-BN banner's bf16 note for the nets with no such peer (depthwise and XLA-strided convs:
    MobileNetV2, MobileNetV4, EfficientNet): the bf16 twins are outside the statement. -/
def syncBnBf16TwinsNote : String :=
  "    // (Both are stated at the f32 nodes; this artifact's bf16 conv twins, which round\n" ++
  "    // their operands per element, are not in that statement.)\n"

/-- The `%wdz` declaration an excluding render needs. Emitted only when the flag is on, so at
    `wdExclude := false` not one byte moves and every committed artifact is untouched. `params`
    names which parameters take it, in the banner comment only. -/
def wdzConst (wdExclude : Bool) (params : String := "1-D params") : String :=
  if wdExclude then
    s!"    // ── timm no_weight_decay (wdExcludeNormBias): {params} take %wdz, not %wd ──\n" ++
    "    %wdz = stablehlo.constant dense<0.0> : tensor<f32>\n"
  else ""

-- ════════════════════════════════════════════════════════════════
-- § The optimizer stage — one recipe type, one per-parameter step, folded in signature order
-- ════════════════════════════════════════════════════════════════

/-- Which optimizer tail a render emits. The forward, the backward, the un-fused parameter
    gradients and the packed signature do not depend on it: only the per-parameter tail
    (`optOne`), the baked constants (`optConstsB`) and the variant marker (`OptRecipe.slug`) do.

    * `.adamw` — every net's committed default.
    * `.rmsprop` — the MobileNetV2 and EfficientNet ImageNet references' optimizer.
    * `.heavyBall` — **the [`jax/MainResnetImagenet.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/MainResnetImagenet.lean) reference rule**: coupled L2 decay, then
      heavy-ball momentum. See `optOne` for why it needs no new `SHlo` op.

    The two accumulating recipes need the fourth parameter region `G` and the `%aup`/`%akeep`
    scalars in the signature (`packedTrainSig` at `acc := true`); ResNet-50 and MobileNetV4 build
    that signature, and ResNet-34 refuses them. -/
inductive OptRecipe
  /-- `θ' = θ − lr·(m̂/(√v̂+ε)) − lr·wd·θ`; both moments live. -/
  | adamw
  /-- **RMSProp with momentum**, TensorFlow's flavour: coupled L2 first, then the mean-square, then
      the momentum buffer on the normalised gradient (ε INSIDE the root). The buffer rides in the
      `m` slot and the mean-square in `v`; see `optOne`'s `.rmsprop` arm. Its constants are the
      net's own `RmsHyper` (`optConstsB`'s `rms`). -/
  | rmsprop
  /-- `g ← g + wd·θ`, `v' = μ·v + g`, `θ' = θ − lr·v'`; velocity in the `v` slot, `m` untouched. -/
  | heavyBall
  /-- **PLAIN SGD** with coupled L2: `g ← g + wd·θ`, `θ' = θ − lr·g`. No velocity, so BOTH the
      `m` and `v` slots ride through untouched and the packed `[θ|m|v]` signature is unchanged —
      the same convention `.heavyBall` uses for `m` alone.

      It exists for the book's optimizer ablation. Removing AdamW and putting `.heavyBall` in its
      place measures *AdamW against momentum*, not against nothing, and momentum is most of what an
      adaptive optimizer buys at this depth — so that arm routes around the thing it is supposed to
      remove. This case is the honest bottom of the ladder.
      It is `.heavyBall` MINUS step ②, which is why it needs no new `SHlo` op either. -/
  | sgd
  /-- **LAMB** (You et al. 2019) — RSB-A3's optimizer, at **two new ops** (`lambDirF`,
      `lambScaleF`). Adam moments give a direction
      `r = m̂/(√v̂+ε) + wd·θ`, then a PER-PARAMETER-TENSOR trust ratio `‖θ‖/‖r‖` rescales the step.
      `Optim.Lamb` is the ℝ reference; see `optOne`. -/
  | lamb
  /-- **AdamW over `k` accumulated micro-batches.** A FOURTH parameter region `G` holds the running gradient sum, and the graph is
      one function for both phases with two runtime scalars deciding which it is. See `optOne`. -/
  | adamwAccum (k : Nat)
  /-- **LAMB over `k` accumulated micro-batches — RSB-A3's ACTUAL optimizer.**

      **The observation that makes this one constructor rather than a redesign:** the accumulator
      `Gt = akeep·G + g` sits UPSTREAM of the optimizer and does not care who consumes it. So the
      accumulate/apply machinery is shared verbatim with `.adamwAccum` (see `accumScalarConsts`,
      which both arms emit), and only the tail that consumes `Gt` differs.

      An accumulate micro-batch needs `m' = m`, `v' = v`, `θ' = θ`. The first two come from
      `%b1 = %b2 = 1`, `%ob1 = %ob2 = 0` exactly as for AdamW. **`θ' = θ` comes from `lr = 0`**,
      because LAMB's parameter step is `sgdParamF θ lr (trust·r)` — at `lr = 0` that is `θ − 0·(…)`
      exactly, with no decay term left running, since LAMB's `wd` lives INSIDE `r` and the zero
      multiplies it away. (AdamW gets the same result for a different reason: its decay is
      DECOUPLED, so `lr = 0` kills it too.)

      `lambDirF` also reads `%b1..%ob2`, so on an accumulate micro-batch it computes an `r` built
      from `β₁·m` rather than the real moment. **That is harmless and is said out loud here**: `r`
      feeds only `lambScaleF → sgdParamF`, and `lr = 0` discards it. Nothing stateful is written —
      the moments are passthroughs and θ is frozen, so the accumulate phase's ONLY effect is `Gt`. -/
  | lambAccum (k : Nat)
deriving DecidableEq, Repr

/-- The optimizer's name in an artifact's banner. -/
def OptRecipe.label : OptRecipe → String
  | .adamw         => "AdamW"
  | .rmsprop       => "RMSProp"
  | .heavyBall     => "heavy-ball momentum + coupled L2"
  | .sgd           => "plain SGD + coupled L2 (no momentum)"
  | .lamb          => "LAMB (per-tensor trust ratio)"
  | .adamwAccum k  => s!"AdamW over {k} ACCUMULATED micro-batches"
  | .lambAccum k   => s!"LAMB (per-tensor trust ratio) over {k} ACCUMULATED micro-batches"

/-- **The optimizer's marker in a variant slug**, with the data-parallel `dp` placed the way every
    net spells it: `adam`/`adamdp`, `rms`/`rmsdp`, `mom`/`momdp`, `sgd`/`sgddp`, `lamb`/`lambdp`.

    `k` is IN THE NAME for the accumulating recipes (`acc4x`, `accdp4x`) because the driver has to
    know it (to decide which micro-batch applies) and the graph has it baked (in `%ob1`/`%ob2`).
    Two places, and a disagreement between them is silent: the run would apply on a cadence the
    `1/k` does not match, i.e. a wrong effective learning rate with no error anywhere. Carrying `k`
    in the artifact NAME makes the driver read it off the same string that selects the file. -/
def OptRecipe.slug (opt : OptRecipe) (replicas : Nat) : String :=
  match opt with
  | .adamw     => if replicas ≤ 1 then "adam" else "adamdp"
  | .rmsprop   => if replicas ≤ 1 then "rms"  else "rmsdp"
  | .heavyBall => if replicas ≤ 1 then "mom"  else "momdp"
  | .sgd       => if replicas ≤ 1 then "sgd"  else "sgddp"
  | .lamb      => if replicas ≤ 1 then "lamb" else "lambdp"
  | .adamwAccum k => (if replicas ≤ 1 then "acc" else "accdp") ++ toString k ++ "x"
  -- `lamb` ++ `acc` — the marker is NO LONGER LEADING, and that is what broke the driver's
  -- `startsWith "acc"` predicate (defect #4 in `tests/TestVariantPredicates.lean`). The name is
  -- spelled this way rather than as `accdp8x64lamb` because the OPTIMIZER is the primary axis and
  -- every other variant leads with it; the fix belongs in the predicate, not in the spelling.
  -- `dp` goes INSIDE, right after `acc`, matching `.adamwAccum`'s placement exactly — so the
  -- `k` parse is one rule for both, not two.
  | .lambAccum k => (if replicas ≤ 1 then "lambacc" else "lambaccdp") ++ toString k ++ "x"

/-- The `%aup`-driven scalar block that turns ONE graph into both accumulation phases.

    **ONE WRITER, shared by `.adamwAccum` and `.lambAccum`.** These eleven lines are the whole
    accumulate/apply mechanism; one definition keeps the two optimizers from carrying two copies
    of it.

        accumulate (%aup = 0):  β₁ = 1, (1−β₁) = 0  ⇒  m' = m,  v' = v   exactly
        apply      (%aup = 1):  β₁ = 0.9, (1−β₁)/k  ⇒  m' = 0.9·m + (1−β₁)·(Gt/k)

    `1/k` is folded in HERE, and asymmetrically: `%ob1` carries `1/k` while `%ob2` carries `1/k²`,
    because `v` consumes the gradient SQUARED. `v' = β₂v + ((1−β₂)/k²)·Gt² = β₂v + (1−β₂)·(Gt/k)²` —
    the identity that makes accumulation equal a real large-batch step rather than the "mean of
    per-micro-batch second moments" a naive implementation produces.

    `fmt12`, not `fmt6`: at k = 4, `(1−β₂)/k² = 6.25e-5`, and `fmt6` emits `0.000063` — 0.8%
    wrong, baked, in the optimizer.

    β₁/β₂ are 0.9/0.999 for BOTH optimizers (LAMB's moments ARE Adam's), which is why this block
    needs no per-optimizer parameter. Only `%eps`/`%wd` differ, and those stay in `optConstsB`. -/
def accumScalarConsts (k : Nat) : String :=
  let kf := k.toFloat
  "    %aone = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  s!"    %aob1 = stablehlo.constant dense<{fmt12 (1.0 - 0.9)}> : tensor<f32>\n" ++
  "    %ab1d = stablehlo.multiply %aup, %aob1 : tensor<f32>\n" ++
  "    %b1 = stablehlo.subtract %aone, %ab1d : tensor<f32>\n" ++
  s!"    %aob1k = stablehlo.constant dense<{fmt12 ((1.0 - 0.9) / kf)}> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.multiply %aup, %aob1k : tensor<f32>\n" ++
  s!"    %aob2 = stablehlo.constant dense<{fmt12 (1.0 - 0.999)}> : tensor<f32>\n" ++
  "    %ab2d = stablehlo.multiply %aup, %aob2 : tensor<f32>\n" ++
  "    %b2 = stablehlo.subtract %aone, %ab2d : tensor<f32>\n" ++
  s!"    %aob2k = stablehlo.constant dense<{fmt12 ((1.0 - 0.999) / (kf * kf))}> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.multiply %aup, %aob2k : tensor<f32>\n"

/-- **Is this parameter in timm's `no_weight_decay` group?**, recovered from the ONE name
    `wdNameBy` produces rather than passed alongside it.

    Derived and not a second argument, on purpose. The skip-list controls TWO things —
    whether the decay term enters `r` (`%wdz`) and whether the trust ratio applies at all
    — and a caller threading a `Bool` beside the name is exactly the
    two-writers shape that lets them disagree: a parameter decayed but not adapted, or the reverse.
    One name, one predicate, both consumers downstream of it. -/
def wdNameExcludes (wdName : String) : Bool := wdName != "%wd"

/-- `(θ', m', v')` for one parameter, from its un-fused gradient.

    For `.adamw` the three ops are the proven `adamMNextF`/`adamVNextF`/`adamWParamF`
    (`adamW_triple_faithful` bundles their `den`s into `Proofs.adamWStep` by `rfl`). β₁/β₂/ε/wd are
    baked literals; `%lr`/`%bc1`/`%bc2` are runtime `tensor<f32>` args, so one render serves a whole
    LR schedule. For `.heavyBall` see the inline notes — same discipline, different three ops, and
    `m` becomes a passthrough so the packed signature does not move. For `.rmsprop` the buffer is
    returned in the `m` slot and the mean-square in `v`.

    At `replicas > 1` the gradient is first averaged across devices by
    `prettyAllReduceMean` — `pretty` of the `allReduceMeanF` node, whose `den` is the replica
    MEAN of the per-replica gradient nodes (`den_allReduceMeanF`). The
    optimizer tail consumes the averaged gradient as an `.operand`, exactly as it consumes the raw
    one. What stays trusted is the lowerer's `all_reduce`, as every op's lowering is, so the
    collective has its own numeric check: the DP render is SYNC-BN (`bnFwdSite`/`bnBackSite`/
    `bnGammaSite`), so the data-parallel step at 2×32 should equal the single-device step at 1×64
    on every output region, and `resnet34-syncbn-check` checks that numerically.

    At `replicas ≤ 1` this emits **nothing** and threads the raw gradient, so the single-device
    render stays byte-identical — which is the cheap self-check that this insertion is inert. -/
-- Every batched render calls this one step, directly or through `optAllParams`: a per-net copy
-- would put AdamW/heavy-ball semantics in two places — the double-writer failure.
def optOne (opt : OptRecipe) (B : Nat) (replicas : Nat) (g : PGrad)
    -- DEFAULTED to `"%wd"`. The caller decides per PARAMETER
    -- whether this is `"%wd"` or the zero constant `"%wdz"` — timm's `no_weight_decay` skip-list —
    -- which ConvNeXt and ViT implement this exact way. It needs NO new op:
    -- `adamWParamF`/`lambDirF`/`momVNextF` all take the decay as an OPERAND NAME, so excluding a
    -- parameter is binding that name to a zero rather than changing a graph.
    (wdName : String := "%wd")
    -- **`preAvg` — the caller has already all-reduced (and clipped) this gradient**, so the
    -- collective must not be emitted a second time. The global-norm clip needs EVERY gradient at
    -- once while this function is per parameter, and under DP the clip must come AFTER the
    -- `all_reduce` (the reference clips the combined gradient; clipping per replica clips 161
    -- PARTIAL gradients — a different function that still trains and still descends). So at
    -- `gradClip := true` the caller hoists both and sets this. ViT's and ConvNeXt's clip
    -- renders set it for the identical reason.
    -- It needs no interface change beyond the flag: `emitGradAllReduce` at `replicas ≤ 1` emits
    -- NOTHING and threads its input name straight through, so forcing 1 here is exactly "skip it".
    (preAvg : Bool := false)
    -- **`accIn` — the caller has already emitted the ACCUMULATOR too, and this is its RAW
    -- output name.** Only the accumulating optimizers read it, and only under the clip.
    --
    -- The reference clips the MEAN ACCUMULATED gradient, not the micro-batch one
    -- (`emitLossAndTraining` in `jax/Jax/Codegen.lean` — `grads = _gsum / _K` and only THEN the clip line), so the fold
    -- has to run on `Gt`, which is computed here, per parameter. The caller therefore hoists the
    -- `momVNextF` as well, folds the norm across all 161 of them, clips, and passes the clipped
    -- total back in `g.grad` while naming the UNCLIPPED one here.
    --
    -- **The two must stay distinct, and that is the whole subtlety.** `%<p>a` rides out as the
    -- fourth region and is the carry the NEXT micro-batch accumulates onto; the reference's carry
    -- (`_gsum`) is raw, clipped only on the way into the optimizer. Returning the clipped total
    -- here would compound the clip across the k micro-batches of every cycle — a contraction that
    -- trains and descends and is not the recipe.
    (accIn : Option String := none)
    -- **`ema` — the MODEL-EMA shadow, a region of its own** (the residual family's RSB-A2/A1
    -- recipes use it). One extra op per parameter, emitted AFTER the
    -- optimizer's own tail because the shadow tracks the UPDATED weight: `e' = %emad·e + %oemad·θ'`.
    --
    -- It needs NO new `SHlo` constructor. `Proofs.adamMNext b ob m g = b·m + ob·g` instantiated
    -- at `(b := %emad, ob := %oemad, m := e, g := θ')` denotes exactly the exponential moving
    -- average, which is the same reading `.adamwAccum`'s accumulator takes of `momVNextF`. So the
    -- faithfulness theorems carry over untouched and this costs none of the ten-site surgery an
    -- added op does.
    --
    -- **IT READS `nT`, THE UPDATED PARAMETER — NOT `%<p>`.** The reference EMAs the weights
    -- AFTER the optimizer moves them (`ema_params = ema_update(ema_params, params, step)` follows
    -- the `train_step` call, `emitMainImagenet` in `jax/Jax/Codegen.lean`). Reading the incoming θ instead gives a
    -- shadow lagging by one step — a number that trains, descends and is quietly not the
    -- reference's.
    --
    -- **The DECAY is a runtime scalar, not a constant**, because it is warmup-corrected:
    -- `d = min(decay, (1+t)/(10+t))` moves every step and the driver computes it. Baking 0.9999
    -- here is the EMA-warmup defect.
    (ema : Bool := false)
    -- **`emaSuf` — the shadow's SSA suffix**, `%{nm}{emaSuf}`, and `packedTrainSig`'s argument of
    -- the same name. `"e"` for ViT, ConvNeXt and EfficientNet; `"ema"` for the ResNet family
    -- (`optAllParams`), because at suffix `e` the stem BN gamma `%sg` produces **`%sge`** — the
    -- hardcoded block-local name inside `select_and_scatter`'s comparator, the maxpool backward.
    -- MLIR's textual name scope is flat across nested regions, so the artifact is rejected at
    -- parse with *"redefinition of SSA value '%sge'"*. It renders, passes byte-identity and render
    -- coverage, and its arity checks — only a parse catches it
    -- (`scripts/gates/parse_verified_mlir.py`). The nets with no maxpool have no `%s*` block-locals.
    (emaSuf : String := "e") :
    StateM Proofs.StableHLO.EmitS (String × String × String × String × Option String × Option String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let replicas := if preAvg then 1 else replicas
  let (arS, gAvg) ← Proofs.StableHLO.prettyAllReduceMean g.grad g.ds g.nm replicas
  let gr : SHlo n := .operand gAvg z
  -- Emitted by every arm below rather than once here, because it consumes each arm's OWN `nT`.
  -- Hoisting it would need the updated-parameter name before the arm that produces it has run.
  let emaTail : String → StateM Proofs.StableHLO.EmitS (String × Option String) := fun nT =>
    if ema then do
      -- At `ema := false` no `pretty` call happens, so the fresh-name counter does not move.
      let (c, nE) ← pretty B (.adamMNextF s!"%{g.nm}{emaSuf}" "%emad" "%oemad" g.ds 0 z
                      (.operand nT z))
      pure (c, some nE)
    else pure ("", none)
  match opt with
  | .adamw =>
    let (cA, nT, nM, nV) ← prettyAdamW B g.nm g.ds gAvg wdName
    let (cE, nE) ← emaTail nT
    pure (arS ++ cA ++ cE, nT, nM, nV, none, nE)
  | .rmsprop =>
    -- Four ops, ONE of them new. Reading the reference
    -- ([`jax/Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean), the `.rmsprop` branch) top to bottom:
    --
    --   `grads = g + WD * p`                → `momVNextF` at `(μ := wd, v := θ)`
    --                                         (`Proofs.momVNext_as_coupled_l2`)
    --   `sq = RHO*s + (1-RHO)*g*g`          → **`adamVNextF` at `β₂ := ρ`**: `s'` is `adamVNext ρ`
    --   `buf = MOMENTUM*b + g/sqrt(sq+EPS)` → `rmsBufNextF`, the new op, ε INSIDE the root
    --   `params = p - lr*buf`               → `sgdParamF` on the buffer's SSA
    --
    -- **The weight decay is COUPLED and goes FIRST**, so the accumulator sees the decayed
    -- gradient. Reversing that order — decaying after the accumulator, AdamW-style — is a
    -- different optimizer and would not show up as an arity or type error anywhere.
    --
    -- **EfficientNet's ε is 1e-3, where MobileNetV2's is 1.0** — the placement's sensitive end: at
    -- a collapsed mean-square the textbook spelling takes a step **31.6×** larger
    -- (`Proofs.rmsBufNext_eps_placement_at_zero`). A green MobileNetV2 tie does not license the
    -- EfficientNet render; `rms-tie efficientnet` is its own gate.
    --
    -- Slot mapping: the packed `[θ|m|v]` signature is reused verbatim with **`m` carrying the
    -- momentum buffer and `v` the running mean-square**, the same slot reinterpretation the
    -- Nesterov render does for its velocity. That is why the driver and the interface do not move.
    -- An excluded parameter's `wdName` is `"%wdz"`, so its coupled L2 is `g + 0·θ = g` exactly.
    let (cW, nW) ← pretty B (.momVNextF s!"%{g.nm}" wdName g.ds 0 z gr)
    let gw : SHlo n := .operand nW z
    let (cS, nS) ← pretty B (.adamVNextF s!"%{g.nm}v" "%rho" "%orho" g.ds 0 z gw)
    let (cB, nB) ← pretty B (.rmsBufNextF s!"%{g.nm}v" s!"%{g.nm}m" "%rho" "%orho" "%mu" "%eps"
                      g.ds 0 0 0 z z gw)
    -- θ' threads b' by SSA NAME, not by re-nesting `rmsBufNextF` inside `sgdParamF`: `pretty` has
    -- no CSE, so re-nesting would emit the whole buffer block a second time.
    let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nB z))
    let (cE, nE) ← emaTail nT
    pure (arS ++ cW ++ cS ++ cB ++ cT ++ cE, nT, nB, nS, none, nE)
  | .lamb =>
    -- LAMB, in four ops per parameter, TWO of which are LAMB's own (`gradSumSqAccF` serves the
    -- clip too, and `sgdParamF` heavy-ball). `Optim.Lamb` carries the ℝ reference and the clauses.
    let z1 : Vec 1 := fun _ => 0
    -- ① the DIRECTION, `r = m̂/(√v̂+ε) + wd·θ`, from the incoming moments and this step's gradient.
    -- ε OUTSIDE the root and the decay INSIDE `r` — both placements are load-bearing and both
    -- have a plausible wrong neighbour (RMSProp-TF's `√(v̂+ε)`, AdamW's decay after the ratio).
    let (cR, nR) ← pretty B (.lambDirF s!"%{g.nm}" s!"%{g.nm}m" s!"%{g.nm}v" "%b1" "%ob1"
                      "%b2" "%ob2" "%bc1" "%bc2" "%eps" wdName g.ds 0 0 0 0 0 0 z z z gr)
    -- ② `‖θ‖²`, THIS parameter's own. Seeded at `%lzero` and never folded across parameters —
    -- that single-leaf fold is the entire difference from the global-norm clip, whose whole
    -- semantic content is that ONE scalar is shared (`Proofs.clipFactor_shared` against
    -- `Proofs.lambScale_not_shared`). The two features emit nearly the same lines.
    -- **The no_weight_decay group is NOT layer-adapted, so this op is SKIPPED for it.**
    -- timm reads `if weight_decay != 0 or group['always_adapt']:` before computing the ratio, so an
    -- excluded parameter takes a plain Adam step at `trust = 1`. Feeding `%lzero` here — the same
    -- zero this op would otherwise seed from — makes `wn2 = 0`, and `Proofs.lambTrust_zero_weight`
    -- (`lambTrust 0 rn2 = 1`, already `@[simp]`) says that IS 1. No new op, no new constructor,
    -- no new theorem; the excluded params emit one op FEWER.
    -- The existing zero-norm guard does not already cover this. It fires at `‖θ‖ = 0` exactly,
    -- i.e. step one, where every BN β and bias starts; from step two the parameter is
    -- small-but-nonzero and `‖θ‖/‖r‖` collapses to ~0.01–0.1 against timm's 1.0. So
    -- `lambTrust_zero_weight` can hold on both sides while both are wrong.
    -- INERT unless the skip-list is on: with `wdExclude := false` every `wdName` is `"%wd"`, so
    -- this takes the `else` and every committed non-`wx` artifact re-renders byte-identically.
    let (cN, nN) ← if wdNameExcludes wdName then pure ("", "%lzero")
                   else pretty B (.gradSumSqAccF (n := n) g.ds (.operand "%lzero" z1)
                                   (.operand s!"%{g.nm}" z))
    -- ③ `trust · r`, with `‖r‖²` reduced inside the op from its own tensor child.
    let (cS, nS) ← pretty B (.lambScaleF (n := n) g.ds (.operand nN z1) (.operand nR z))
    -- ④ `θ' = θ − lr·(trust·r)` — `sgdParamF`, an op that already exists, applied to the scaled
    -- direction exactly as `.heavyBall` applies it to the velocity.
    let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nS z))
    -- the moments themselves, unchanged from `.adamw` — LAMB's m and v ARE Adam's.
    let (cM, nM) ← pretty B (.adamMNextF s!"%{g.nm}m" "%b1" "%ob1" g.ds 0 z gr)
    let (cV, nV) ← pretty B (.adamVNextF s!"%{g.nm}v" "%b2" "%ob2" g.ds 0 z gr)
    let (cE, nE) ← emaTail nT
    pure (arS ++ cM ++ cV ++ cR ++ cN ++ cS ++ cT ++ cE, nT, nM, nV, none, nE)
  | .adamwAccum _ =>
    -- **The whole feature, and it adds exactly ONE op per parameter.**
    --
    -- ① the accumulator, `Gt = akeep·G + g`. That is `momVNextF` read the way `.heavyBall` reads it
    -- for coupled L2 — `Proofs.momVNext μ v g = μ·v + g`, so instantiating `(μ := %akeep, v := G)`
    -- denotes exactly the accumulation. **No new `SHlo` constructor**, so none of the ten-site
    -- surgery an added op costs, and the faithfulness theorem carries over untouched.
    --
    -- `%akeep` is 0 on the FIRST micro-batch of a cycle and 1 after, so the cycle RESETS by
    -- dropping the previous total rather than by a separate zeroing pass — which is why one scalar
    -- and one op suffice, and why there is no "clear the accumulator" step that could be skipped.
    -- Under `accIn` the caller emitted this op itself (it needed `Gt` to fold the global norm),
    -- and `gr` is already the CLIPPED total — so the tail reads `gr` while the fourth region still
    -- reports the raw `Gt`. See `accIn`'s note: those two must not collapse into one name.
    let (cG, nG, gt) ← match accIn with
      | some a => pure ("", a, gAvg)
      | none   => do
          let (c, nm') ← pretty B (.momVNextF s!"%{g.nm}a" "%akeep" g.ds 0 z gr)
          pure (c, nm', nm')
    -- ② the moments and the parameter, **byte-identical to `.adamw`'s** except that they consume
    -- `Gt` rather than `g`. The `1/k` that turns a SUM into a MEAN is not applied here: it is folded
    -- into `%ob1 = (1−β₁)/k` and `%ob2 = (1−β₂)/k²` by `optConstsB`, because `v` is QUADRATIC in the
    -- gradient and a single shared scale factor cannot serve both moments. Folding it is what keeps
    -- this from needing a scalar-multiply op the vocabulary does not have.
    --
    -- On an ACCUMULATE micro-batch `optConstsB` sets `%b1 = %b2 = 1` and `%ob1 = %ob2 = 0`, so both
    -- moments are exact passthroughs; `%lr = 0` freezes θ, and AdamW's decay is DECOUPLED (`−lr·wd·θ`)
    -- so lr = 0 freezes it COMPLETELY rather than leaving a decay term running k times per step.
    let (cA, nT, nM, nV) ← prettyAdamW B g.nm g.ds gt wdName
    let (cE, nE) ← emaTail nT
    pure (arS ++ cG ++ cA ++ cE, nT, nM, nV, some nG, nE)
  | .lambAccum _ =>
    -- **RSB-A3's optimizer.** Structurally: `.adamwAccum`'s ① accumulator, then `.lamb`'s tail
    -- reading `Gt` where it read `g`. Nothing else changes, and nothing new is introduced — no new
    -- `SHlo` constructor, so no ten-site surgery and every faithfulness theorem carries over.
    let z1 : Vec 1 := fun _ => 0
    -- ① the accumulator, `Gt = akeep·G + g` — the SAME `momVNextF` instantiation `.adamwAccum`
    -- uses, for the same reason (`Proofs.momVNext μ v g = μ·v + g` at `(μ := %akeep, v := G)`).
    -- It is upstream of the optimizer and does not know which one follows: that independence is
    -- the whole reason this composition is a constructor and not a redesign.
    -- Same `accIn` carve-out as `.adamwAccum`'s, and this is the arm RSB-A3 actually renders:
    -- timm's `Lamb` is the optimizer whose `max_grad_norm = 1.0` default makes the clip
    -- mandatory, so `.lambAccum` + clip is the composition that has to be right.
    let (cG, nG, gt) ← match accIn with
      | some a => pure ("", a, gr)
      | none   => do
          let (c, nm') ← pretty B (.momVNextF s!"%{g.nm}a" "%akeep" g.ds 0 z gr)
          pure (c, nm', (.operand nm' z : SHlo n))
    -- ② LAMB's four ops, **byte-identical to `.lamb`'s except that they consume `Gt` rather than
    -- `g`** — the same substitution `.adamwAccum` makes to AdamW's three.
    let (cR, nR) ← pretty B (.lambDirF s!"%{g.nm}" s!"%{g.nm}m" s!"%{g.nm}v" "%b1" "%ob1"
                      "%b2" "%ob2" "%bc1" "%bc2" "%eps" wdName g.ds 0 0 0 0 0 0 z z z gt)
    -- `‖θ‖²` reads θ ALONE — no gradient, so accumulation cannot reach it and this line is
    -- character-for-character `.lamb`'s, INCLUDING its no_weight_decay skip: an excluded parameter feeds
    -- `%lzero` so `lambTrust 0 _ = 1`. See the `.lamb` branch above for why.
    let (cN, nN) ← if wdNameExcludes wdName then pure ("", "%lzero")
                   else pretty B (.gradSumSqAccF (n := n) g.ds (.operand "%lzero" z1)
                                   (.operand s!"%{g.nm}" z))
    let (cS, nS) ← pretty B (.lambScaleF (n := n) g.ds (.operand nN z1) (.operand nR z))
    -- ④ `θ' = θ − lr·(trust·r)`. THIS is where the accumulate phase is frozen: `%lr = 0` makes it
    -- `θ − 0·(…) = θ` exactly. LAMB's decay is inside `r`, which the zero multiplies away, so there
    -- is no decay term left running k times per optimizer step.
    let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nS z))
    -- the moments, from `Gt`, with `%b1`/`%ob1` computed from `%aup` — passthroughs while accumulating.
    let (cM, nM) ← pretty B (.adamMNextF s!"%{g.nm}m" "%b1" "%ob1" g.ds 0 z gt)
    let (cV, nV) ← pretty B (.adamVNextF s!"%{g.nm}v" "%b2" "%ob2" g.ds 0 z gt)
    let (cE, nE) ← emaTail nT
    pure (arS ++ cG ++ cM ++ cV ++ cR ++ cN ++ cS ++ cT ++ cE, nT, nM, nV, some nG, nE)
  | .heavyBall =>
    -- ── Three applications of ops that ALREADY EXIST. No new `SHlo` constructor, so none of the
    -- ten-site surgery (and none of the `StableHLO.Parse` roundtrip risk) an added op costs.
    --
    -- ① COUPLED L2 decay, `g ← g + wd·θ`. This reuses **`momVNextF`**, which is not a pun:
    -- `Proofs.momVNext μ v g = μ·v + g`, so instantiating `(μ := wd, v := θ)` denotes exactly
    -- `wd·θ + g`. Same function, so the faithfulness theorem carries over unchanged; only the
    -- *reading* of the two slots differs. NOTE this is COUPLED (into the gradient, so it flows
    -- through the velocity), not AdamW's DECOUPLED `−lr·wd·θ` — that difference is the whole
    -- reason `.adamw` cannot stand in for the reference recipe.
    let (cD, nD) ← pretty B (.momVNextF s!"%{g.nm}" wdName g.ds 0 z gr)
    let gwd : SHlo n := .operand nD z
    -- ② velocity, `v' = μ·v + g`.
    let (cV, nV) ← pretty B (.momVNextF s!"%{g.nm}v" "%mu" g.ds 0 z gwd)
    -- ③ HEAVY-BALL parameter step, `θ' = θ − lr·v'` — `sgdParamF` applied to the velocity rather
    -- than to the gradient. This is deliberately NOT `momParamF`: that op is **Nesterov**
    -- (`θ − lr·(g + μ·v')`, see `Proofs.momParam_heavyBall_diff`), and the JAX reference this
    -- render exists to match steps by `v'` alone. Using the momentum-named op here would have been
    -- the obvious move and would have silently produced a different optimizer.
    let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nV z))
    -- `m` rides through untouched, so the packed `[θ|m|v]` signature is shared with `.adamw` and
    -- the driver is byte-identical across variants (the `CnnRender.optTail` `.sgd` convention).
    let (cE, nE) ← emaTail nT
    pure (arS ++ cD ++ cV ++ cT ++ cE, nT, s!"%{g.nm}m", nV, none, nE)
  -- PLAIN SGD: `.heavyBall` with step ② deleted. Two ops, both already here.
  | .sgd =>
    -- ① COUPLED L2 decay, `g ← g + wd·θ` — identical to `.heavyBall`'s ①, `momVNextF` read at
    -- `(μ := wd, v := θ)`. Kept, so the ONLY difference between the two arms is the velocity.
    let (cD, nD) ← pretty B (.momVNextF s!"%{g.nm}" wdName g.ds 0 z gr)
    -- ② the step, `θ' = θ − lr·g`, applied to the DECAYED GRADIENT rather than to a velocity.
    -- Same `sgdParamF` the heavy-ball arm ends with; only its operand differs. That is the
    -- whole of "no momentum", and it is worth seeing that it is one operand and not one flag.
    let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nD z))
    let (cE, nE) ← emaTail nT
    -- BOTH `m` and `v` ride through untouched — `.heavyBall` passes `m` and writes `v`; this
    -- passes both. The packed `[θ|m|v]` arity is unchanged, so the driver needs no predicate.
    pure (arS ++ cD ++ cT ++ cE, nT, s!"%{g.nm}m", s!"%{g.nm}v", none, nE)

/-- **How many micro-batches the optimizer accumulates over** — `k` for the two accumulating
    constructors and `1` for every other, so a caller can ask the question without a second `match`
    that could disagree with `accOn`'s.

    It exists for the CLIP (`clipNormStr`/`clipEpsStr` below), which is the first feature whose
    emitted constants depend on `k` from OUTSIDE `accumScalarConsts`. -/
def optAccumK : OptRecipe → Nat
  | .adamwAccum k => k
  | .lambAccum k  => k
  | _             => 1

/-- **The clip threshold as the render bakes it, `k·C`** — and the `k` is not a typo.

    **THE REFERENCE CLIPS THE MEAN ACCUMULATED GRADIENT, NOT THE MICRO-BATCH ONE.**
    `emitLossAndTraining` in [`jax/Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean) is unambiguous about the order:

    ```python
    grads = jax.tree.map(lambda _a: _a / _K, _gsum)   # the MEAN over k micro-batches
    loss  = jnp.mean(_ls)
    gn    = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, C / (gn + 1e-6)), grads)
    ```

    This render never materialises that mean: `optOne`'s accumulator carries the SUM `Gt`, and the
    `1/k` is folded into `%ob1 = (1−β₁)/k` and `%ob2 = (1−β₂)/k²` downstream (`accumScalarConsts`,
    and it is split that way because `v` is QUADRATIC in the gradient). So the fold here runs on
    `Gt`, whose norm is `k·‖mean‖`, and the threshold must move with it:

    `min(1, kC / (‖Gt‖ + k·ε))  =  min(1, C / (‖Gt‖/k + ε))`

    — the reference's factor on the mean, exactly, with no new op and no division emitted. And the
    factor is then applied to `Gt` rather than to the mean, which is the same thing for the same
    reason: scaling commutes with the `1/k` the moments fold in afterwards.

    `fmt12`, not `fmt6`, for `accumScalarConsts`' stated reason — these are baked literals in the
    optimizer, where nothing downstream would question a truncated one. At `k = 1` this is the
    identity and emits the plain threshold. -/
def clipNormStr (clipNorm : Float) (k : Nat) : String := fmt12 (clipNorm * k.toFloat)

/-- The clip's `ε`, scaled by the same `k` and for the same reason — see `clipNormStr`.

    It is NOT cosmetic and it does not cancel: the reference's `+ 1e-6` is what keeps the factor
    from being `0/0` at a zero gradient (`Proofs.clipDenom_pos`), and leaving it unscaled while the
    numerator scales would shift the factor by `k` in exactly the regime the guard exists for. -/
def clipEpsStr (k : Nat) : String := fmt12 (0.000001 * k.toFloat)

/-- The rank-0 zero that seeds the global-norm fold. Its own name rather than `%lzero`: that one
    exists only under the two LAMB constructors (`optConstsB`), and the clip is an independent axis
    that has to work over `.adamw` too. Emitted only when the flag is on, so at `gradClip := false`
    not one byte moves and every committed artifact is untouched — the same discipline `wdzConst`
    keeps. -/
def clipZeroConst (gradClip : Bool) : String :=
  if gradClip then
    "    // ── timm Lamb.max_grad_norm (D1): seed of the GLOBAL squared-norm fold ──\n" ++
    "    %czero = stablehlo.constant dense<0.0> : tensor<f32>\n"
  else ""

/-- **The weight decay each optimizer bakes when the caller does not override it.**

    The two families genuinely differ, and by 200×: AdamW's `1e-4` against LAMB's `0.02`, the
    latter off timm's a3 arg string (`lamb-cosine-lr0.008-wd0.02-…`). `optConstsB`'s `.lamb` arm
    records what reusing AdamW's number here would produce — a LAMB that is structurally right and
    200× off on the decay.

    It exists so that `wdStr` can mean "the optimizer's own value" by DEFAULT rather than by the
    caller restating a number that is already decided by the constructor. -/
def optWdDefault : OptRecipe → String
  | .adamw | .heavyBall | .sgd | .adamwAccum _ => "0.0001"
  | .lamb  | .lambAccum _              => "0.02"
  -- RMSProp's decay is the net's `RmsHyper.wd`, which `optConstsB` bakes; `wdStr` is not read.
  | .rmsprop => ""

/-- **The decay actually baked**: the caller's override, or `optWdDefault`. Empty means default.

    **`%wd` IS A BAKED `stablehlo.constant`, NOT A RUNTIME OPERAND — this parameterises the
    literal, it does not make the decay schedulable.** Unlike `%lr`, which stays a `tensor<f32>`
    argument so one graph serves a whole cosine, changing the decay is a RE-RENDER. That is the same
    shape `ConvNeXtRender.convnextAdamConsts` already has (`wdStr := "0.0001"`, with the ImageNet
    render passing `0.05`), copied rather than re-invented.

    Why it exists: RSB-**A1** uses wd = 0.01 where A3 uses 0.02, so A1 costs a re-render rather
    than a new op. -/
def optWdStr (opt : OptRecipe) (wdStr : String := "") : String :=
  if wdStr.isEmpty then optWdDefault opt else wdStr

/-- **The variant marker for a NON-DEFAULT decay**, and it is not optional bookkeeping.

    **Two renders that differ only in a baked constant MUST NOT share a path.** `%wd` lives in
    the artifact, so an A1 render (0.01) and an A3 render (0.02) at the same optimizer, batch and
    replica count would otherwise both be `lambaccdp8x64wxclipbce`, and the last writer would win.
    `scripts/regen_verified_mlir.sh check` would catch it as a two-writer collision, but a collision
    that cannot be SPELLED is better than one that is merely detected.

    Spelling: `wd` ++ the digits with the point removed, so `0.01` → `wd001` and `0.005` →
    `wd0005`. Mechanical, and unambiguous because the leading `0` is kept. Empty at the default,
    so every committed artifact keeps its name and its bytes. -/
def wdVariantMark (opt : OptRecipe) (wdStr : String := "") : String :=
  if wdStr.isEmpty || wdStr == optWdDefault opt then "" else "wd" ++ wdStr.replace "." ""

/-- **The label-smoothing marker**, `wdVariantMark`'s peer and there for the identical reason: α is
    BAKED into the smoothed-CE cotangent, so two renders differing only in it would collide on one
    artifact path. Empty at the default 0.1, so every committed spelling is unchanged.
    `ls0`, not `ls0000000`: `fmt6 0.0` is `"0.000000"` and stripping its point leaves seven
    zeros, so OFF gets the short spelling it deserves and any other α keeps the general one. The
    `#guard`s in `ResNet34RenderB` pin that.
    It must reach `r34AdamVariant` and not merely the renderer — the rule `wx`, `clip` and `bf16`
    each state there: an
    artifact whose declared entry disagrees with its own path is refused by the shim outright. -/
def lsVariantMark (alpha : Float := 0.1) : String :=
  if alpha == 0.1 then "" else if alpha == 0.0 then "ls0"
  else "ls" ++ (fmt6 alpha).replace "." ""

/-- **The WHOLE optimizer stage for a net: the hoisted global-norm clip, then `optOne` per
    parameter.** Returns `(code, θ', m', v', G', E')`, with `G'` empty unless the optimizer
    accumulates and `E'` empty unless `ema`. The two are INDEPENDENT — see
    `VerifiedVariant.nRegions` for the region count.

    **THIS IS A FUNCTION SO THAT THE ONE-STEP GATE CAN DRIVE THE SHIPPED PATH.** The one-step
    optimizer gate — one step of each optimizer on the same `(θ, g, state)` — needs the optimizer
    stage ALONE, and a second copy of it written for the gate would gate a transcription rather than
    the emission. [`tests/TestOptStepFixtures.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestOptStepFixtures.lean) calls exactly this.

    **THE ORDER IS THE SEMANTICS, and there are two orderings to get right, not one.**

      ① the clip goes AFTER the `all_reduce`. Each replica holds a PARTIAL gradient; clipping those
         and then averaging is a clip of nothing in particular. `optOne` all-reduces per parameter,
         so under the clip the collective is hoisted here and `optOne` is told (`preAvg`) not to
         repeat it.
      ② the clip goes AFTER the ACCUMULATION. The reference is explicit
         (`emitLossAndTraining` in [`jax/Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean)): `grads = _gsum / _K` and only THEN the clip line, so the
         norm is of the MEAN over the k micro-batches. Clipping the micro-batch gradient instead
         would clip k times per optimizer step against a threshold meant for their mean — again
         something that trains and descends. So the accumulator is hoisted here too, and the
         threshold moves to `k·C` to read the fold on `Gt` as a fold on `Gt/k` (`clipNormStr`).

    Neither ordering is visible to a gate that only checks "the gradients got smaller", which is
    why `Proofs.clipFactor_shared` is the statement to drive and why it must be driven in the
    CLIPPING regime — the identity-below-threshold gate is structurally blind to placement.

    At `gradClip := false` NOT ONE `pretty` CALL happens in the clip block, so the fresh-name
    counter does not move and every committed artifact re-renders byte-identically. -/
def optAllParams (opt : OptRecipe) (B replicas : Nat) (ps : List PGrad)
    (wdExclude : Bool := false) (gradClip : Bool := false) (clipNorm : Float := 1.0)
    -- **`ema` — the model-EMA shadow region**, threaded straight to `optOne`.
    -- TRAILING and defaulted, so every committed R50 artifact re-renders byte-identically and the
    -- one-step gate's existing call is unchanged.
    -- It is INDEPENDENT of the accumulator: `G` and `E` are two regions, not one slot two
    -- features share (`VerifiedVariant.nRegions`).
    (ema : Bool := false)
    -- The shadow's suffix, `"ema"` by default for the ResNet family's maxpool (see `optOne`).
    (emaSuf : String := "ema") :
    StateM Proofs.StableHLO.EmitS (String × List String × List String × List String × List String × List String) := do
  let accOn := match opt with | .adamwAccum _ => true | .lambAccum _ => true | _ => false
  let z1 : Vec 1 := fun _ => 0
  let accK := optAccumK opt
  let mut clipCode := ""
  let mut clipped : List (String × String) := []
  let mut accRaw : List (String × String) := []
  if gradClip then
    -- ① average across replicas first. At `replicas ≤ 1` this emits nothing and threads the name
    -- through, so the single-device clip render carries no collective at all.
    let mut avg : List (String × String) := []
    for g in ps do
      let (arS, gAvg) ← Proofs.StableHLO.prettyAllReduceMean g.grad g.ds g.nm replicas
      clipCode := clipCode ++ arS
      avg := avg ++ [(g.nm, gAvg)]
    -- ② accumulate, when the optimizer accumulates. `Gt = akeep·G + g`, the SAME `momVNextF`
    -- instantiation `optOne` would have emitted — moved, not duplicated, and handed back to it by
    -- name so the fourth region still reports the RAW total.
    -- On an ACCUMULATE micro-batch the clip is computed on a PARTIAL `Gt` and then discarded:
    -- `%lr = 0` freezes θ and `%b1 = %b2 = 1` / `%ob1 = %ob2 = 0` make both moments exact
    -- passthroughs, so only the APPLY micro-batch's factor — the one taken on the full sum — can
    -- reach a weight. That is what makes ② expressible without a second buffer.
    let mut src : List (String × String) := avg
    if accOn then
      let mut acc : List (String × String) := []
      for g in ps do
        let n := g.ds.foldl (· * ·) 1
        let z : Vec n := fun _ => 0
        let gAvg := (avg.lookup g.nm).getD g.grad
        let (cG, nG) ← pretty B (.momVNextF s!"%{g.nm}a" "%akeep" g.ds 0 z (.operand gAvg z))
        clipCode := clipCode ++ cG
        acc := acc ++ [(g.nm, nG)]
      accRaw := acc
      src := acc
    -- ③ ONE scalar, folded across every parameter before any of them is scaled. The fold must run
    -- to completion first — that is the global-vs-local distinction, made structural by
    -- `Proofs.clipScale` taking the factor as a PARAMETER it cannot compute from its own tensor.
    let mut total : SHlo 1 := .operand "%czero" z1
    for g in ps do
      let n := g.ds.foldl (· * ·) 1
      let gS := (src.lookup g.nm).getD g.grad
      total := .gradSumSqAccF (n := n) g.ds total (.operand gS (fun _ => 0))
    let (cN, normSSA) ← pretty B total
    clipCode := clipCode ++ cN
    -- ④ scale each parameter's gradient by that one shared factor.
    for g in ps do
      let n := g.ds.foldl (· * ·) 1
      let z : Vec n := fun _ => 0
      let gS := (src.lookup g.nm).getD g.grad
      let (cS, sSSA) ← pretty B (.clipScaleF (n := n) (clipNormStr clipNorm accK) (clipEpsStr accK)
                          0 0 g.ds (.operand normSSA z1) (.operand gS z))
      clipCode := clipCode ++ cS
      clipped := clipped ++ [(g.nm, sSSA)]
  -- ═══ the optimizer: one proven triple per parameter ═══
  let mut code := clipCode
  let mut thetaN : List String := []
  let mut mNames : List String := []
  let mut vNames : List String := []
  let mut aNames : List String := []
  let mut eNames : List String := []
  for g in ps do
    -- The decay operand comes from the SAME `PGrad` that names the site, so the parameter whose
    -- shape decides exclusion is the parameter being updated. Reading the shape
    -- off a parallel list is how this class of bug ships.
    -- Under the clip the gradient `optOne` consumes is the CLIPPED one, while `accIn` names the
    -- unclipped accumulator it must still return: see `optOne`'s note for why those two names
    -- cannot collapse into one.
    let gIn : PGrad :=
      if gradClip then { g with grad := (clipped.lookup g.nm).getD g.grad } else g
    let (c, nT, nM, nV, nA, nE) ← optOne opt B replicas gIn (wdNameBy wdExclude g.nm g.ds)
                                (preAvg := gradClip) (accIn := accRaw.lookup g.nm) (ema := ema)
                                (emaSuf := emaSuf)
    code := code ++ c
    thetaN := thetaN ++ [nT]
    mNames := mNames ++ [nM]
    vNames := vNames ++ [nV]
    -- The accumulator's output name, present only under the accumulating constructors. It becomes
    -- the FOURTH region of the packed blob — the same shape the EMA renders use, so the driver's
    -- `nRegions = 4` path is reused rather than a second one being written.
    match nA with | some a => aNames := aNames ++ [a] | none => pure ()
    -- The shadow's output name, present only under `ema`. It becomes the FIFTH region — AFTER
    -- the accumulator, never instead of it. Region order `[θ|m|v|G|E]` is what keeps every blob
    -- written without a fifth region readable at the index it was written at.
    match nE with | some e => eNames := eNames ++ [e] | none => pure ()
  pure (code, thetaN, mNames, vNames, aNames, eNames)

/-- The optimizer's baked constants. `.adamw` is byte-for-byte the committed block; `.heavyBall`
    emits only what it reads, so there are no dead constants in the momentum artifact.

    `%wd` is baked rather than a runtime arg because weight decay is not scheduled — unlike `%lr`,
    which stays a `tensor<f32>` argument so one graph serves the whole cosine schedule.

    `.rmsprop` bakes the net's own `rms` (`mnv2RmsHyper`, `enetRmsHyper`): ε is where the two
    references differ, so there is no shared default to fall back on. -/
def optConstsB (opt : OptRecipe) (wdStr : String := "") (rms : Option RmsHyper := none) :
    String :=
  -- ONE binding, used by every arm below, so the two families cannot drift apart in how they
  -- honour the override — `optWdStr` owns "the caller's value or this optimizer's default".
  let wd := optWdStr opt wdStr
  match opt with
  | .adamw => adamWConsts wd
  | .rmsprop =>
    match rms with
    | some h => rmsConstsBlock h
    | none => panic! "optConstsB .rmsprop: pass the net's RmsHyper as (rms := some …)"
  | .heavyBall =>
    "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n"
  -- NO `%mu`. Emitting one would be harmless MLIR (an unused constant) and exactly the kind of
  -- decoration that makes a reader think the velocity is in there somewhere.
  | .sgd =>
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n"
  | .lamb =>
    -- **`%eps` is 1e-6, NOT AdamW's 1e-8**, and `%wd` is 0.02, NOT 1e-4. Both come off timm's a3
    -- arg string (`lamb-cosine-lr0.008-wd0.02-…`), which `jax/MainResnet50Imagenet.lean` decodes in
    -- its own comment. Reusing AdamW's numbers here would render a LAMB that is structurally right
    -- and 200x off on the decay.
    -- `%lzero` seeds each parameter's OWN norm fold, one leaf deep. The clip's `%zero` seeds a
    -- fold across ALL parameters. Same op, and the seed placement is the whole difference.
    "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
    "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
    "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
    "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
    "    %eps = stablehlo.constant dense<1.0e-6> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n" ++
    "    %lzero = stablehlo.constant dense<0.0> : tensor<f32>\n"
  | .adamwAccum k =>
    -- **A TRUSTED CARVE-OUT, and deliberately the smallest one that does the job**: eight lines of
    -- SCALAR arithmetic emitted ONCE, next to the constants that are already emitted text. The 161
    -- per-parameter tails stay `pretty(verified AST node)` and are byte-identical to `.adamw`'s.
    --
    -- `%aup ∈ {0, 1}` is the APPLY flag, supplied per micro-batch by the driver. It selects between
    -- the two phases by arithmetic rather than by two artifacts — one graph, one compile, one
    -- resident parameter set, and no way for an "accumulate" and an "apply" render to drift:
    --
    --     accumulate (%aup = 0):  β₁ = 1, (1−β₁) = 0  ⇒  m' = m,  v' = v   exactly
    --     apply      (%aup = 1):  β₁ = 0.9, (1−β₁)/k  ⇒  m' = 0.9·m + (1−β₁)·(Gt/k)
    --
    -- `1/k` is folded in HERE, and asymmetrically: `%ob1` carries `1/k` while `%ob2` carries
    -- `1/k²`, because `v` consumes the gradient SQUARED. `v' = β₂v + ((1−β₂)/k²)·Gt² =
    -- β₂v + (1−β₂)·(Gt/k)²` — the identity that makes accumulation equal a real large-batch step
    -- rather than the "mean of per-micro-batch second moments" a naive implementation produces.
    --
    -- `fmt12`, not `fmt6` — see `accumScalarConsts`, which owns those eight lines so that
    -- `.lambAccum` emits the SAME mechanism rather than a second copy of it.
    "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n" ++
    accumScalarConsts k
  | .lambAccum k =>
    -- **THE COMPOSITION, AND IT IS EXACTLY "LAMB's CONSTANTS + THE SHARED ACCUMULATOR".**
    -- `%eps` 1e-6 and `%wd` 0.02 are LAMB's, off timm's a3 arg string — NOT AdamW's 1e-8/1e-4.
    -- Reusing AdamW's numbers here would render a LAMB that is structurally right and 200× off on
    -- the decay, which is the exact trap `.lamb`'s own comment records.
    -- `%lzero` seeds each parameter's OWN norm fold, one leaf deep — carried over from `.lamb`
    -- unchanged, because accumulation does not touch the trust ratio.
    -- `%b1`/`%ob1`/`%b2`/`%ob2` are COMPUTED from `%aup`, not baked, which is the only difference
    -- from `.lamb`'s constant block and is precisely what `accumScalarConsts` supplies.
    "    %eps = stablehlo.constant dense<1.0e-6> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n" ++
    "    %lzero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    accumScalarConsts k

-- ════════════════════════════════════════════════════════════════
-- § The packed train-step interface `[θ|m|v|G|E] + scalars`
-- ════════════════════════════════════════════════════════════════

/-- The packed train step's **argument list after `%x`**: the parameter regions in the order the
    driver packs them — `θ`, `m`, `v`, then the accumulator `G` (`<p>a`, only under gradient
    accumulation) and the model-EMA shadow `E` (`<p>{emaSuf}`, only under `ema`) — followed by the
    runtime scalars `%lr, %bc1, %bc2`, `%aup, %akeep` (accumulation) and `%emad, %oemad` (EMA).
    `ps` is `(%name, type)` per parameter, in signature order.

    `G` precedes `E` and never follows it: `[θ|m|v|G|E]` is the order that leaves both single-axis
    layouts at the index they already occupy. `emaSuf` is `"e"` except on the ResNet family, whose
    stem BN gamma `%sg` + `e` would be `%sge` — `select_and_scatter`'s block-local name in the
    max-pool backward — so there it is `"ema"` (`optOne`'s `emaSuf`). -/
def packedTrainSig (ps : List (String × String)) (acc : Bool := false) (ema : Bool := false)
    (emaSuf : String := "e") : String :=
  let region (suf : String) := String.intercalate ", " (ps.map fun (n, t) => s!"{n}{suf}: {t}")
  let sufs := ["", "m", "v"] ++ (if acc then ["a"] else []) ++ (if ema then [emaSuf] else [])
  String.intercalate ", " (sufs.map region) ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>" ++
    (if acc then ", %aup: tensor<f32>, %akeep: tensor<f32>" else "") ++
    (if ema then ", %emad: tensor<f32>, %oemad: tensor<f32>" else "")

/-- The packed train step's **leading result types**, in `packedTrainSig`'s order: the updated
    regions `θ', m', v'[, G'][, E']`, then `%loss, %bc1, %bc2`, then the handed-back
    accumulation / EMA scalars. `pTy` is the parameter types in signature order. -/
def packedTrainRetTys (pTy : List String) (acc : Bool := false) (ema : Bool := false) :
    List String :=
  pTy ++ pTy ++ pTy ++ (if acc then pTy else []) ++ (if ema then pTy else []) ++
    ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++
    (if acc then ["tensor<f32>", "tensor<f32>"] else []) ++
    (if ema then ["tensor<f32>", "tensor<f32>"] else [])

/-- The stochastic-depth mask arguments `, %dp<i>: tensor<Bxf32>` for the sites `idxs`, in the
    order given — the order the driver's `dropScales` writes them into the blob. Empty when `sd` is
    off, which keeps every non-SD render byte-identical. Each net passes its own site list
    (`vitDropSig`, `cnxDropSig`, `enetDropSig`, `r50DropSig`). -/
def dropMaskSig (B : Nat) (sd : Bool) (idxs : List Nat) : String :=
  if sd then String.join (idxs.map (fun i => s!", {dpName i}: {ty [B]}")) else ""

-- ════════════════════════════════════════════════════════════════
-- § Precision-switched constructors: `XAt bf16 rnd …` is `XBf16 rnd …` or `X …`
-- ════════════════════════════════════════════════════════════════

/-! Every bf16 op is its f32 peer with one extra leading argument, the rounding `rnd`. `XAt bf16
rnd …` is the `if bf16 then .convBf16 (h := h) zrnd … else .conv (h := h) …` choice, written once
per constructor instead of at every call site. `pretty` evaluates it, so the emitted text is
exactly the chosen branch's. -/

/-- `conv`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.convAt (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (bias : Vec oc) :
    BatchableOp (ic*h*w) (oc*h*w) :=
  if bf16 then .convBf16 rnd wName bName W bias else .conv wName bName W bias

/-- `convBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convBackBatchedAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    SHlo (N * (oc * h * w)) → SHlo (N * (ic * h * w)) :=
  if bf16 then .convBackBatchedBf16 rnd wName W b else .convBackBatched wName W b

/-- `convStride4`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.convStride4At (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (bias : Vec oc) :
    BatchableOp (ic*(2*(2*h))*(2*(2*w))) (oc*h*w) :=
  if bf16 then .convStride4Bf16 rnd wName bName W bias else .convStride4 wName bName W bias

/-- `convStride4WeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convStride4WeightGradBAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec oc) (x : Vec (N * (ic*(2*(2*h))*(2*(2*w)))))
    (W : Kernel4 oc ic kH kW) :
    SHlo (N * (oc*h*w)) → SHlo (oc*ic*kH*kW) :=
  if bf16 then .convStride4WeightGradBBf16 rnd xName b x W else .convStride4WeightGradB xName b x W

/-- `convStrided`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.convStridedAt (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (bias : Vec oc) :
    BatchableOp (ic*(2*h)*(2*w)) (oc*h*w) :=
  if bf16 then .convStridedBf16 rnd wName bName W bias else .convStrided wName bName W bias

/-- `convStridedBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convStridedBackBatchedAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    SHlo (N * (oc * h * w)) → SHlo (N * (ic * (2 * h) * (2 * w))) :=
  if bf16 then .convStridedBackBatchedBf16 rnd wName W b else .convStridedBackBatched wName W b

/-- `convStridedWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convStridedWeightGradBAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) :
    SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW) :=
  if bf16 then .convStridedWeightGradBBf16 rnd xName b x W else .convStridedWeightGradB xName b x W

/-- `convStridedXla`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.convStridedXlaAt (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (bias : Vec oc) :
    BatchableOp (ic*(2*h)*(2*w)) (oc*h*w) :=
  if bf16 then .convStridedXlaBf16 rnd wName bName W bias else .convStridedXla wName bName W bias

/-- `convStridedXlaWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convStridedXlaWeightGradBAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec oc) (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (W : Kernel4 oc ic kH kW) :
    SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW) :=
  if bf16 then .convStridedXlaWeightGradBBf16 rnd xName b x W
  else .convStridedXlaWeightGradB xName b x W

/-- `convWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.convWeightGradBAt (bf16 : Bool) {N ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec oc) (x : Vec (N * (ic * h * w)))
    (W : Kernel4 oc ic kH kW) :
    SHlo (N * (oc * h * w)) → SHlo (oc * ic * kH * kW) :=
  if bf16 then .convWeightGradBBf16 rnd xName b x W else .convWeightGradB xName b x W

/-- `denseRow`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.denseRowAt (bf16 : Bool) {N a c : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Mat a c) (b : Vec c) :
    BatchableOp (N*a) (N*c) :=
  if bf16 then .denseRowBf16 rnd wName bName W b else .denseRow wName bName W b

/-- `denseRowBack`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.denseRowBackAt (bf16 : Bool) {rows a c : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : Mat a c) :
    BatchableOp (rows*c) (rows*a) :=
  if bf16 then .denseRowBackBf16 rnd wName W else .denseRowBack wName W

/-- `depthwise`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.depthwiseAt (bf16 : Bool) {c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : DepthwiseKernel c kH kW) (bias : Vec c) :
    BatchableOp (c*h*w) (c*h*w) :=
  if bf16 then .depthwiseBf16 rnd wName bName W bias else .depthwise wName bName W bias

/-- `depthwiseBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseBackBatchedAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : DepthwiseKernel c kH kW) (b : Vec c) :
    SHlo (N * (c * h * w)) → SHlo (N * (c * h * w)) :=
  if bf16 then .depthwiseBackBatchedBf16 rnd wName W b else .depthwiseBackBatched wName W b

/-- `depthwiseStrided`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.depthwiseStridedAt (bf16 : Bool) {c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : DepthwiseKernel c kH kW) (bias : Vec c) :
    BatchableOp (c*(2*h)*(2*w)) (c*h*w) :=
  if bf16 then .depthwiseStridedBf16 rnd wName bName W bias
  else .depthwiseStrided wName bName W bias

/-- `depthwiseStridedBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseStridedBackBatchedAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : DepthwiseKernel c kH kW) (b : Vec c) :
    SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w))) :=
  if bf16 then .depthwiseStridedBackBatchedBf16 rnd wName W b
  else .depthwiseStridedBackBatched wName W b

/-- `depthwiseStridedWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseStridedWeightGradBAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) :
    SHlo (N * (c * h * w)) → SHlo (c * kH * kW) :=
  if bf16 then .depthwiseStridedWeightGradBBf16 rnd xName b x W
  else .depthwiseStridedWeightGradB xName b x W

/-- `depthwiseStridedXla`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def BatchableOp.depthwiseStridedXlaAt (bf16 : Bool) {c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : DepthwiseKernel c kH kW) (bias : Vec c) :
    BatchableOp (c*(2*h)*(2*w)) (c*h*w) :=
  if bf16 then .depthwiseStridedXlaBf16 rnd wName bName W bias
  else .depthwiseStridedXla wName bName W bias

/-- `depthwiseStridedXlaBackBatched`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseStridedXlaBackBatchedAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName : String) (W : DepthwiseKernel c kH kW) (b : Vec c) :
    SHlo (N * (c * h * w)) → SHlo (N * (c * (2 * h) * (2 * w))) :=
  if bf16 then .depthwiseStridedXlaBackBatchedBf16 rnd wName W b
  else .depthwiseStridedXlaBackBatched wName W b

/-- `depthwiseStridedXlaWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseStridedXlaWeightGradBAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec c) (x : Vec (N * (c * (2 * h) * (2 * w))))
    (W : DepthwiseKernel c kH kW) :
    SHlo (N * (c * h * w)) → SHlo (c * kH * kW) :=
  if bf16 then .depthwiseStridedXlaWeightGradBBf16 rnd xName b x W
  else .depthwiseStridedXlaWeightGradB xName b x W

/-- `depthwiseWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.depthwiseWeightGradBAt (bf16 : Bool) {N c h w kH kW : Nat}
    (rnd : ℝ → ℝ) (xName : String) (b : Vec c) (x : Vec (N * (c * h * w)))
    (W : DepthwiseKernel c kH kW) :
    SHlo (N * (c * h * w)) → SHlo (c * kH * kW) :=
  if bf16 then .depthwiseWeightGradBBf16 rnd xName b x W else .depthwiseWeightGradB xName b x W

/-- `flatConvF`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.flatConvFAt (bf16 : Bool) {ic oc h w kH kW : Nat}
    (rnd : ℝ → ℝ) (wName bName : String) (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    SHlo (ic*h*w) → SHlo (oc*h*w) :=
  if bf16 then .flatConvFBf16 rnd wName bName W b else .flatConvF wName bName W b

/-- `matmulFB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.matmulFBAt (bf16 : Bool) {N m k n : Nat}
    (rnd : ℝ → ℝ) :
    SHlo (N*(m*k)) → SHlo (N*(k*n)) → SHlo (N*(m*n)) :=
  if bf16 then .matmulFBBf16 rnd  else .matmulFB

/-- `rowDenseWeightGradB`, or its bf16 peer at rounding `rnd` when `bf16`. -/
@[reducible] def SHlo.rowDenseWeightGradBAt (bf16 : Bool) {N tk a c : Nat}
    (rnd : ℝ → ℝ) (xName : String) (x : Vec (N*(tk*a))) :
    SHlo (N*(tk*c)) → SHlo (a*c) :=
  if bf16 then .rowDenseWeightGradBBf16 rnd xName x else .rowDenseWeightGradB xName x

/-- **One frozen-statistic BN site at the per-example index** — `bnPerChannelEvalF` on `xin`, with
    the running statistics arriving as graph inputs `%{statP}mu` / `%{statP}var`. The BN site of the
    per-example eval chains (`r34FwdChain`, `r50FwdChain`, `mnv2FwdChain`), which write
    `@resnet34_fwd_eval`, `@resnet50in_fwd_eval` and `@mobilenetv2_fwd_eval`: frozen-stat affine BN
    performs no reduction, so these forwards are class-batch-independent and partner the batch-BN
    train steps whose EMA'd μ/σ² they read. There is deliberately no training arm — the per-example
    training BN those chains once also emitted (`bnPerChannelF`) is not the BN any shipped train
    step uses. -/
def bnEvalSite (B oc hh ww : Nat) (epsStr gName btName statP xin : String) :
    StateM EmitS (String × String) := do
  let zc  : Vec oc := fun _ => 0
  let zin : Vec (oc*hh*ww) := fun _ => 0
  pretty B (.bnPerChannelEvalF (oc := oc) (h := hh) (w := ww)
    gName btName s!"%{statP}mu" s!"%{statP}var" epsStr 0 zc zc zc zc (.operand xin zin))

/-- **One vector-LN site at the per-example index**: `lnRowF` at the scalar identities
    `%one`/`%zero` (γ = 1, β = 0), then the real `[n]` affine `rowScaleF gN` and `rowBiasF btN`, on
    the `[m, n]` matrix `xin`, at ε `epsStr`. ViT's token LN (`m` tokens) and ConvNeXt's head LN
    (`m = 1`, after GAP). Returns the LN-output SSA. The ℝ arguments do not print, so `0` and `1`
    stand for them. -/
def vecLnSite (B m n : Nat) (epsStr gN btN xin : String) : StateM EmitS (String × String) := do
  let (c1, a) ← pretty B (.lnRowF (m := m) (n := n) "%one" "%zero" epsStr 0 1 0
                            (.operand xin (0 : Vec (m*n))))
  let (c2, b) ← pretty B (.rowScaleF (m := m) (n := n) gN (0 : Vec n) (.operand a (0 : Vec (m*n))))
  let (c3, o) ← pretty B (.rowBiasF (m := m) (n := n) btN (0 : Vec n) (.operand b (0 : Vec (m*n))))
  pure (c1 ++ c2 ++ c3, o)

/-- **`vecLnSite` at the batched index**: the same three ops as `batchOp`s over `N` examples of
    `[m, n]` each. `m` is the row count PER EXAMPLE; folding the batch into it is a different
    graph. -/
def vecLnSiteB (N m n : Nat) (epsStr gN btN xin : String) : StateM EmitS (String × String) := do
  let (c1, a) ← pretty N (.batchOp (N := N) (.lnRow (m := m) (n := n) "%one" "%zero" epsStr 0 1 0)
                            (.operand xin (0 : Vec (N*(m*n)))))
  let (c2, b) ← pretty N (.batchOp (N := N) (.rowScale (m := m) (n := n) gN (0 : Vec n))
                            (.operand a (0 : Vec (N*(m*n)))))
  let (c3, o) ← pretty N (.batchOp (N := N) (.rowBias (m := m) (n := n) btN (0 : Vec n))
                            (.operand b (0 : Vec (N*(m*n)))))
  pure (c1 ++ c2 ++ c3, o)

/-- The label-smoothed cotangent after a render's softmax: `pretty` of `smoothedCotTail` over the
    softmax's name `smN` and `%onehot`, at width `n` per example (`1 * K` under the row softmax,
    `K` under `softmaxDiv ∘ expe`). The ℝ arguments do not print, so `0` stands for them. With the
    softmax printed first, the text is `pretty` of `smoothedLossCotGraph` (row) or
    `smoothedLossCotGraphDiv`, the graphs the step ties start from. -/
def smoothedCotB (B n : Nat) (aStr negAK bStr smN : String) : StateM EmitS (String × String) :=
  pretty B (smoothedCotTail (N := B) (n := n) 0 0 0 aStr negAK bStr
    (.operand smN 0) (.operand "%onehot" 0))

/-- The BCE cotangent after a render's sigmoid: `pretty` of `bceCotTail` over the sigmoid's name
    `sgN` and `%onehot`; with the sigmoid printed first, the text is `pretty` of `bceLossCotGraph`. -/
def bceCotB (B n : Nat) (bStr sgN : String) : StateM EmitS (String × String) :=
  pretty B (bceCotTail (N := B) (n := n) 0 bStr (.operand sgN 0) (.operand "%onehot" 0))

-- ════════════════════════════════════════════════════════════════
-- § The report-only `%loss` blocks
-- ════════════════════════════════════════════════════════════════

/-! The scalar `%loss` every train step returns is for logging: it is on no gradient path, and it
is NOT `pretty` of an AST node, which the emitted text says. No theorem covers it, so a wrong loss
here (plain CE against a smoothed cotangent, a K = 10 literal at K = 1000) shows only as a reported
loss that disagrees with the reference's under an otherwise bit-identical forward — only the
numeric tie catches it. That is why each block is written once, here, with its hyperparameters
derived (`oneMinusAlpha`, `alphaOverK`) rather than spelled as literals. Each block introduces only
`%l*` names and reads only the logits or softmax and `%onehot`, so it never shifts a proven
output. -/

/-- **Mean label-smoothed CE** over the softmax `nSm` (`[B, nClasses]`):
    `loss = −(1/B)·Σ_b [ (1−α)·Σ_k onehot·log sm + (α/K)·Σ_k log sm ]`, the objective whose
    cotangent `smoothedCotB` prints. Every batched CE render's `%loss`. -/
def reportSmoothedCeLoss (B nClasses : Nat) (nSm : String) (alpha : Float := 0.1) : String :=
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

/-- **Mean BCE-with-logits** over the logits `nLog` (`[B, nClasses]`), the `reportSmoothedCeLoss`
    peer for the renders whose cotangent is `bceCotB`. It is the stable form `softplus(z) − t·z`:
    expanding `t·softplus(−z) + (1−t)·softplus(z)` with `softplus(−x) = softplus(x) − x` collapses
    the reference's two softplus calls to one, and `softplus(z) = max(z,0) + log(1 + exp(−|z|))`
    never exponentiates a positive number. The mean is over all `B·K` entries, not of the
    per-example sum. -/
def reportBceLoss (B nClasses : Nat) (nLog : String) : String :=
  "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
  "    // BCE-with-logits, mean over B x K: softplus(z) - t*z, softplus stable as\n" ++
  "    // max(z,0) + log(1 + exp(-|z|)). Mean over B*K, NOT mean of the per-example sum.\n" ++
  s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %lzb = stablehlo.constant dense<0.0> : {ty [B, nClasses]}\n" ++
  s!"    %labs = stablehlo.abs {nLog} : {ty [B, nClasses]}\n" ++
  s!"    %lneg = stablehlo.negate %labs : {ty [B, nClasses]}\n" ++
  s!"    %lexp = stablehlo.exponential %lneg : {ty [B, nClasses]}\n" ++
  s!"    %lone = stablehlo.constant dense<1.0> : {ty [B, nClasses]}\n" ++
  s!"    %l1pe = stablehlo.add %lone, %lexp : {ty [B, nClasses]}\n" ++
  s!"    %llg = stablehlo.log %l1pe : {ty [B, nClasses]}\n" ++
  s!"    %lmax = stablehlo.maximum {nLog}, %lzb : {ty [B, nClasses]}\n" ++
  s!"    %lsp = stablehlo.add %lmax, %llg : {ty [B, nClasses]}\n" ++
  s!"    %ltz = stablehlo.multiply %onehot, {nLog} : {ty [B, nClasses]}\n" ++
  s!"    %lbce = stablehlo.subtract %lsp, %ltz : {ty [B, nClasses]}\n" ++
  s!"    %lsum2 = stablehlo.reduce(%lbce init: %lz) applies stablehlo.add across dimensions = [0, 1] : ({ty [B, nClasses]}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    %lbfc = stablehlo.constant dense<{B * nClasses}.0> : tensor<f32>\n" ++
  s!"    %loss = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n"

/-- **Mean plain CE from the logits** `nLog` (`[B, nClasses]`): the block re-derives the softmax
    itself, because the per-example SGD renders (MLP, MNIST CNN) never name theirs. -/
def reportCeLossOfLogits (B nClasses : Nat) (nLog : String) : String :=
  "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
  s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
  s!"    %lex = stablehlo.exponential {nLog} : {ty [B,nClasses]}\n" ++
  s!"    %lsum = stablehlo.reduce(%lex init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsmb = stablehlo.broadcast_in_dim %lsum, dims = [0] : ({ty [B]}) -> {ty [B,nClasses]}\n" ++
  s!"    %lsm = stablehlo.divide %lex, %lsmb : {ty [B,nClasses]}\n" ++
  s!"    %llog = stablehlo.log %lsm : {ty [B,nClasses]}\n" ++
  s!"    %lohll = stablehlo.multiply %onehot, %llog : {ty [B,nClasses]}\n" ++
  s!"    %lrow = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B,nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
  s!"    %lsum2 = stablehlo.reduce(%lrow init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [B]}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    %lbf = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
  s!"    %lossm = stablehlo.divide %lsum2, %lbf : tensor<f32>\n" ++
  s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"

/-- The banner `reportCeLossOfSm` opens with unless its caller passes its own. -/
def reportCeLossBanner : String :=
  "    // ── report-only scalar loss (NOT pretty(AST): the kit has no rank-0 loss op; it\n" ++
  "    //    feeds no parameter, only the driver's progress line) ──\n"

/-- **Mean plain CE from the softmax** `nSm` (`[B, nClasses]`), read from the same softmax the
    cotangent uses: the packed-optimizer CIFAR renders. It reduces against `%lzero`, which those
    renders' constants block declares. -/
def reportCeLossOfSm (B nClasses : Nat) (nSm : String) (banner : String := reportCeLossBanner) :
    String :=
  banner ++
  s!"    %llog = stablehlo.log {nSm} : {ty [B,nClasses]}\n" ++
  s!"    %ohll = stablehlo.multiply %onehot, %llog : {ty [B,nClasses]}\n" ++
  s!"    %csum = stablehlo.reduce(%ohll init: %lzero) applies stablehlo.add across dimensions = [0, 1] : ({ty [B,nClasses]}, tensor<f32>) -> tensor<f32>\n" ++
  s!"    %cneg = stablehlo.negate %csum : tensor<f32>\n" ++
  s!"    %lbf = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
  s!"    %loss = stablehlo.divide %cneg, %lbf : tensor<f32>\n"

end Proofs.StableHLO
