import LeanMlir.Proofs.Codegen.StableHLO.Pretty
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Foundation.BceLossCot

/-! # The renderers' shared optimizer tail

Every batched train-step renderer ends the same way: fold a per-parameter optimizer step over the
parameter list, so the θ/m/v outputs come out in signature order. This file holds that step once.

| step | used by | ops per parameter |
|---|---|---|
| `adamOne` | MobileNetV2, MobileNetV4, EfficientNet | all-reduce (DP only) + the AdamW triple |
| `rmsOne` | MobileNetV2, EfficientNet | all-reduce + coupled L2 + mean-square + buffer + SGD |
| `adamOneEma` | ViT, ConvNeXt | `adamOne` + the model-EMA shadow, with the clip's `preAvg` |

and the train step's **packed interface** — the one positional contract the driver's blob walks:
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

`ResNet34RenderB.optOne` is the multi-optimizer step (AdamW / LAMB / heavy-ball / accumulation /
EMA) that ResNet-34 and ResNet-50 fold; it reads the same `PGrad`.

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

/-- `(θ', m', v')` for one parameter under **AdamW**: the replica mean of its gradient
    (`prettyAllReduceMean`, `pretty` of the `allReduceMeanF` node) and then the proven
    `adamMNextF`/`adamVNextF`/`adamWParamF` triple (`prettyAdamW`). -/
def adamOne (B : Nat) (replicas : Nat) (g : PGrad) (wdName : String := "%wd") :
    StateM Proofs.StableHLO.EmitS (String × String × String × String) := do
  let (arS, gAvg) ← Proofs.StableHLO.prettyAllReduceMean g.grad g.ds g.nm replicas
  let (c, nT, nM, nV) ← prettyAdamW B g.nm g.ds gAvg wdName
  pure (arS ++ c, nT, nM, nV)

/-- `(θ', b', s')` for one parameter under **RMSProp with momentum** — the `adamOne` peer. Only ONE
    of the four ops is new. Reading the reference
    ([`jax/Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean), the `.rmsprop` branch) top to bottom:

    | reference line | emitted here |
    |---|---|
    | `grads = g + WD * p` | `momVNextF` at `(μ := wd, v := θ)` — `Proofs.momVNext_as_coupled_l2` |
    | `sq = RHO*s + (1-RHO)*g*g` | **`adamVNextF` at `β₂ := ρ`** — `s'` is `adamVNext ρ` |
    | `buf = MOMENTUM*b + g/sqrt(sq+EPS)` | `rmsBufNextF` — the new op, ε INSIDE the root |
    | `params = p - lr*buf` | `sgdParamF` on the buffer's SSA |

    **The weight decay is COUPLED and goes FIRST**, so the accumulator sees the decayed gradient.
    Reversing that order — decaying after the accumulator, AdamW-style — is a different optimizer
    and would not show up as an arity or type error anywhere.

    **EfficientNet's ε is 1e-3, where MobileNetV2's is 1.0** — the placement's sensitive end: at a
    collapsed mean-square the textbook spelling takes a step **31.6×** larger
    (`Proofs.rmsBufNext_eps_placement_at_zero`). A green MobileNetV2 tie does not license the
    EfficientNet render; `rms-tie efficientnet` is its own gate.

    Slot mapping: the packed `[θ|m|v]` signature is reused verbatim with **`m` carrying the
    momentum buffer and `v` the running mean-square**, the same slot reinterpretation the Nesterov
    render does for its velocity. That is why the driver and the interface do not move.

    `wdName` is `"%wd"`, or `"%wdz"` for a parameter `wdNameBy` excludes (`wx`). The coupled L2
    reads its coefficient as an operand name, so excluding a parameter binds it to the zero
    constant: `g + 0·θ = g` exactly. -/
def rmsOne (B : Nat) (replicas : Nat) (g : PGrad) (wdName : String := "%wd") :
    StateM Proofs.StableHLO.EmitS (String × String × String × String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let (arS, gAvg) ← Proofs.StableHLO.prettyAllReduceMean g.grad g.ds g.nm replicas
  let (cW, nW) ← pretty B (.momVNextF s!"%{g.nm}" wdName g.ds 0 z (.operand gAvg z))
  let gr : SHlo n := .operand nW z
  let (cS, nS) ← pretty B (.adamVNextF s!"%{g.nm}v" "%rho" "%orho" g.ds 0 z gr)
  let (cB, nB) ← pretty B (.rmsBufNextF s!"%{g.nm}v" s!"%{g.nm}m" "%rho" "%orho" "%mu" "%eps"
                    g.ds 0 0 0 z z gr)
  -- θ' threads b' by SSA NAME, not by re-nesting `rmsBufNextF` inside `sgdParamF`: `pretty` has no
  -- CSE, so re-nesting would emit the whole 13-op buffer block a second time.
  let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nB z))
  pure (arS ++ cW ++ cS ++ cB ++ cT, nT, nB, nS)

/-- `(θ', m', v', e')` for one parameter under **AdamW with the model-EMA shadow** — ViT's and
    ConvNeXt's step. `e'` is `""` at `ema := false`.

    * `wdName` — `"%wd"`, or the zero constant for timm's `no_weight_decay` parameters. The AdamW
      ops take the decay as an OPERAND NAME, so excluding a parameter binds that name to a zero.
    * `preAvg` — the caller has ALREADY averaged (and clipped) this gradient, so the collective is
      not emitted a second time. Under data parallelism the clip must come AFTER the `all_reduce`
      (the reference clips the combined gradient; clipping per replica clips PARTIAL gradients — a
      different function that still trains and descends), and the clip needs every gradient at
      once while this step is per parameter, so at `clip := true` the caller hoists both.
    * `ema` — the shadow `e' = d·e + (1−d)·θ'` is `adamMNextF` at `(β₁ := d, m := e, g := θ')`:
      `Proofs.adamMNext` IS the reference's `ema_update` (`ema_update` in [`jax/Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean)), so it needs
      no new op and `adamMNextF_faithful` closes the `den` side by `rfl`. It reads `nT`, the
      UPDATED parameter — the shadow averages weights after the optimizer moves them.
      `%emad`/`%oemad` are function ARGS, not constants: the reference's decay is warmup-corrected,
      `d = min(decay, (1+t)/(10+t))`; without the warmup a shadow still holds a large share of the
      random init early in training. At `ema := false` no `pretty` call happens, so the
      fresh-name counter does not move and the non-EMA renders are unchanged by the flag.

    The shadow's SSA name is `%{nm}e` here; `ResNet34RenderB.optOne` uses `%{nm}ema` because at
    suffix `e` the stem BN gamma `%sg` collides with `select_and_scatter`'s block-local `%sge`.
    ViT and ConvNeXt have no max-pool, so `e` is safe for them. -/
def adamOneEma (B : Nat) (replicas : Nat) (g : PGrad)
    (ema : Bool := false) (wdName : String := "%wd") (preAvg : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × String × String × String × String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let replicas := if preAvg then 1 else replicas
  let (arS, gAvg) ← Proofs.StableHLO.prettyAllReduceMean g.grad g.ds g.nm replicas
  let (cA, nT, nM, nV) ← prettyAdamW B g.nm g.ds gAvg wdName
  let (cE, nE) ← if ema then
      pretty B (.adamMNextF s!"%{g.nm}e" "%emad" "%oemad" g.ds 0 z (.operand nT z))
    else pure ("", "")
  pure (arS ++ cA ++ cE, nT, nM, nV, nE)

-- ════════════════════════════════════════════════════════════════
-- § Weight-decay exclusion (`wx`): shared by every optimizer tail above and `ResNet34RenderB.optOne`
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
    max-pool backward — so there it is `"ema"` (`ResNet34RenderB.optOne`). -/
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
