# 4c — one chain per net: the renderer convergence

**Opened 2026-09-06 as its own thread, out of `planning/archive/proofs_tier_to_paper_nets.md` §4c; ✅ CLOSED
2026-09-07 — all four legs done, every net one chain.** The
decision it executes was taken by the user that day: *the Imagenette and ImageNet artifacts of a net
come from the SAME renderer, the batched one, and the Proofs tiers are stated at that chain.* This
log carries the per-net legs, the seams outside Lean, and what each leg costs. START at "State of
play".

## Why the axis exists

Four nets have two renderers. `ResNet34Render` / `MobileNetV2Render` / `ConvNeXtRender` / `ViTRender`
render the **per-example-indexed** chain — `pretty B` lifts a one-example node across the batch, and
on the `*Grad` ops the batch sum lives in the EMITTER, not in `den`. The `*RenderB` files render the
**batched** chain — `N := B` inside the AST, `*B` constructors, `den = batchMap N …` with the `Σ_n`
inside `den`. Every Proofs tier that predates section 4 is at the per-example chain; every ImageNet
artifact, every `*dp*`, every drop variant and every quoted accuracy is on the batched one.

They are not two spellings of one function. Per-example BatchNorm reduces `[2,3]` and batched
BatchNorm reduces `[0,2,3]`: different functions of the same architecture, and the only op in the
suite that couples examples.

## ⭐⭐ The precedent: ResNet-50 already did this, and it is the template

`ResNet50RenderB.lean`'s `r50FwdChainB` exists for exactly this reason, and its docstring says so:

> **This exists so `@resnet50_fwd` and `@resnet50_adam_train_step` cannot be different nets.** They
> were: `resnet50FwdFaithfulV` built its forward from the PER-EXAMPLE chain while the train step is
> batch BN. Its own docstring claimed "the same forward the train step differentiates", which was
> the invariant that did not hold.

The fix was **one forward traversal, consumed by both** — not a deletion, not a driver change. And
⚠ **the EVAL forward was deliberately left off that chain**: `bnEval` reads frozen per-channel
statistics and reduces nothing, so `<net>_fwd_eval` is BN-world-agnostic and correct for both.
That is the same fact the float-budget thread records as "the eval-mode forward budgets are
world-agnostic".

⭐ **So each leg is smaller than §4c scoped.** The unit of work is "factor the forward traversal out
of `<net>RenderB` and render `<net>_fwd` from it", plus whatever the driver and the guards then need.
Retiring the per-example renderer is a SEPARATE, later question — R50 did not need it, and the
per-example chain is still the CIFAR chapter's pedagogical ladder.

## The measurable acceptance criterion

`scripts/regen_verified_mlir.sh`'s `check_adam_prefix` carries a ratchet:

```
KNOWN_SPLIT = {
  "resnet34_fwd.mlir":    "per-example BN vs the batch-BN Adam step — two renderers …",
  "mobilenetv2_fwd.mlir": "per-example BN vs the batch-BN Adam step — same two-renderer split",
}
```

**4c is done for a net when its entry leaves that dict.** The script already prints
`✓ RESOLVED — <fwd> is now a N-line prefix of <ts>` and tells you to drop the entry; the list "may
shrink, never grow". ConvNeXt and ViT were never in it, because their `convnext_fwd`/`vit_fwd` and
`convnext_adam_train_step`/`vit_adam_train_step` came BOTH from the per-example renderer —
internally consistent, and the split there is between the drop-free Imagenette pair and everything
else. ⚠ So for legs 3 and 4 this ratchet is not the criterion; leg 4's was an empty
`git diff verified_mlir/` after the swap (ViT's are now all on the batched chain and the ratchet
never moved).

## State of play (2026-09-07)

| net | what is split | leg |
|---|---|---|
| **ResNet-34** | — | 1 ✅ **DONE 2026-09-06** |
| **MobileNetV2** | — | 2 ✅ **DONE 2026-09-06** |
| **ConvNeXt-T** | — (35 of 36 artifacts; the SGD-inline step stays, deliberately) | 3 ✅ **DONE 2026-09-07** |
| **ViT-Tiny** | — (19 of 20 artifacts; the SGD-inline step stays, deliberately) | 4 ✅ **DONE 2026-09-07** |
| EfficientNet-B0, ResNet-50, MNv4 | nothing — one renderer already | — |

⭐⭐ **`KNOWN_SPLIT` IS EMPTY, AND EVERY NET IS ONE CHAIN.** `check_adam_prefix` reads **20 paired,
0 known-split, 0 unaccounted** — the criterion this thread was opened to reach, and at the ImageNet
tier since the PAIRS extension. ⚠ Legs 3 and 4 were a different shape (no BN-world split at all)
and were never in that ratchet, so each needed its own acceptance criterion: leg 4's was an EMPTY
`git diff verified_mlir/` after the swap; leg 3's was a diff of EXACTLY thirteen files × 78 lines,
every changed line a conv-VJP `transpose`/`reverse`, and the four forwards untouched — measured
before a writer moved and again after.

## Leg 1 — ResNet-34 ✅ DONE 2026-09-06

**Result, and it is the thread's acceptance criterion in one line:**
`check_adam_prefix` goes from `5 paired, 2 known-split` to **`6 paired, 1 known-split`**, and
`resnet34_fwd.mlir` is a byte-identical **1220-line prefix of `resnet34_adam_train_step.mlir`**.
⭐ It is also still a 1220-line prefix of `resnet34_sgd_train_step.mlir`, so `check_fwd_prefix`
passes on the SAME forward — which is exactly what one chain per net buys.

**What landed.**

1. **The forward traversal, factored.** `R34FwdRecB` + `r34FwdChainB` extracted from
   `resnet34AdamTrainStepFaithfulB`'s opening, `ResNet50RenderB.r50FwdChainB`'s shape. ⭐ **Every
   committed artifact re-rendered byte-identically** — `pretty`'s SSA counter follows the call
   SEQUENCE, and the sequence is unchanged, so extracting a traversal into a helper is free.
2. **`resnet34FwdFaithfulB`**, and both 10-class forward writers moved into `ResNet34RenderB`.
3. ⛔ **`ResNet34Render.lean` deleted**, with `verified_mlir/resnet34_train_step.mlir` and
   `apps/imagenette/MainResnet34Verified.lean` (and its `resnet34-verified` exe).
4. What had to move rather than go: `R34Bn` / `bnSite` (⚠ **ResNet-50 shares the train/eval BN
   switch** — its bottleneck forward calls `bnSite` with both modes), the signature lists
   `idSig` / `downSig` / `r34SigList` / `r34StatSigList` (the single source for every r34
   artifact's arg order), and the whole per-example forward chain, which the **eval** forward
   needs. All now live in `ResNet34RenderB.lean`, the sole writer of every ResNet-34 artifact.
5. Guards: `KNOWN_SPLIT` shrank by one, `check_fwd_prefix`'s r34 pair moved to
   `resnet34_sgd_train_step`, the regen writer list lost a module, and `proofs.yml` now diffs
   `resnet34_fwd`/`resnet34_fwd_eval` in the `ResNet34RenderB` step instead of a retired one.

**⭐ The split was bigger than the audit could see.** `resnet34in_fwd.mlir` — the **ImageNet**
forward, at 256×1000 — also came from `resnet34FwdFaithfulV`, i.e. per-example BatchNorm, while
every `resnet34in_*` train step is batch BN. `check_adam_prefix`'s PAIRS list has only the
Imagenette names, so this instance was invisible to the very audit built to catch it. Both
ImageNet forwards are on the batched chain now. ⚠ **Extend that PAIRS list to the `*in_*` artifacts
before declaring any later leg done.**

**⚠ The eval forwards did not move, and could not meaningfully.** `resnet34_fwd_eval.mlir` and
`resnet34in_fwd_eval.mlir` re-render byte-identically from the migrated chain: frozen per-channel
statistics reduce nothing, so `bnPerChannelEvalF` is BatchNorm-world-agnostic. Same call R50 makes.

**⛔ There was no number to re-run.** `MainResnet34Verified`'s own header measured
`390/3925 = 9.936306%`, byte identical every epoch — chance — because running-stat threading lives
only in `trainAdamSched`, and it said *"do not quote its accuracy"*. The batched SGD trainer is
`LEAN_MLIR_VARIANT=sgd .lake/build/bin/resnet34-verified-adam`, which already existed. So §4c's
"re-run the Imagenette SGD numbers" was vacuous for this net. ⚠ **Do not assume that for legs 2–4**:
MobileNetV2's per-example SGD trainer is a different case and has not been checked.

**⚠ What the retirement costs, stated plainly.** `ResNet34FaithfulPoC.lean` (the §1 fold) and
`ResNet34TiePoC.lean` (the §1a tie, every parameter at its chain cotangent) are now about an
artifact that does not exist. Every theorem in them is unchanged and still true — they are about
the per-example ResNet-34 and the SGD-inline op family, both of which still exist as mathematics —
but no committed bytes exercise them. Their live peers landed the same day:
`ResNet34FaithfulPoCB.lean` (4.1e) and `ResNet34TiePoCB.lean` (4.2a). Both file headers now say so.
`ResNet34RenderPC.lean` is NOT affected the same way: `resnet34Forward_full_pc` is the subject of
the float budgets, the T6 tie and the witness, none of which is about a train step's bytes.

## Leg 2 — MobileNetV2 ✅ DONE 2026-09-06

**Result:** `check_adam_prefix` goes from `6 paired, 1 known-split` to **`7 paired, 0 known-split,
0 unaccounted`**, and `mobilenetv2_fwd.mlir` is a byte-identical **1714-line prefix of
`mobilenetv2_adam_train_step.mlir`**. The ratchet is empty and the last `KNOWN_SPLIT` entry is
gone.

**The pre-flight question the log told leg 2 to ask, answered: no re-runs, same as leg 1.**
`mobilenetv2-verified`'s own header measured `387/3925 = 9.859873%`, byte identical every epoch —
chance — because running-statistic threading lives only in `trainAdamSched`, and it said in bold
*"THIS DRIVER CANNOT PRODUCE A MEANINGFUL ACCURACY ON THIS NET… Do not quote its accuracy."* The
86.89% in `runs/mobilenetv2_verified_crop_gpu0.log` was already disclaimed by `mnv2FwdFaithfulV`'s
own docstring as having gone through the wrong forward. The real number comes from
`mobilenetv2-verified-adam`, which was already on the batched chain. ⚠ **Do not assume this for
legs 3 and 4 either** — check the driver, as this leg did.

**What landed.**

1. **The forward traversal, factored.** `MNV2FwdRecB` + `mnv2FwdChainB` extracted from
   `mobilenetv2AdamTrainStepFaithfulB`'s opening, `r34FwdChainB`'s shape. ⭐ **Every committed
   artifact re-rendered byte-identically**, as leg 1 predicted: `pretty`'s SSA counter follows the
   call SEQUENCE and the sequence is unchanged.
2. **`mobilenetv2FwdFaithfulB`**, and both train forwards (`mobilenetv2_fwd`, `mobilenetv2in_fwd`)
   moved onto it.
3. ⛔ **`MobileNetV2Render.lean` deleted**, with `verified_mlir/mobilenetv2_train_step.mlir`,
   `verified_mlir/mobilenetv2_reduced_train_step.mlir`, `apps/imagenette/MainMobilenetV2Verified.lean`
   and the `mobilenetv2-verified` exe.
4. What had to move rather than go: the whole PER-EXAMPLE forward chain (`MBFwd`, `bnSiteP`, the
   four `irFwd*`, `MNV2Fwd`, `mnv2FwdChain`, `mnv2FwdSig`, `mnv2FwdEvalFaithfulV`) and the
   `irSig`/`irSigNoExp`/`paperSig` signature lists — because the **eval** forward needs them. All
   now live in `MobileNetV2RenderB.lean`, the sole writer of every MobileNetV2 artifact.
5. Guards: `KNOWN_SPLIT` emptied, mnv2's `check_fwd_prefix` pair REMOVED (it had no batched SGD
   step to re-pair with — see below), `EXEMPT` lost `mobilenetv2_reduced_train_step`, the regen
   writer list lost a module, `proofs.yml` now diffs all four mnv2 forwards in the
   `MobileNetV2RenderB` step, and four entries left `scripts/render_guard_baseline.txt`.

**⚠ Two parameter-naming conventions now live in one file, and that is deliberate.** The batched
chain names parameters `%sW`/`%b2eW`/`%Wd` (`mnv2SigList`); the migrated per-example chain names
them `%Ws`/`%We2`/`%Wfc` (`paperSig`). Same 210/158 parameters in the same order — the `#guard`s
pin both arities — but ⛔ **the eval forward must NOT be re-pointed at the batched chain**, and for
a reason ResNet-34 did not have: `mnv2Paper_float_logits_le_committed`'s provenance claim is that
the typed graph "diffs against `mobilenetv2_fwd_eval` line for line", at those 263 inputs and
those names. Moving it would leave the theorem true and the sentence false.

**⚠ `check_fwd_prefix` lost its MobileNetV2 entry rather than gaining a new partner.** ResNet-34
kept a pair there because `resnet34_sgd_train_step.mlir` already existed on the batched chain.
MobileNetV2 ships no batched SGD step at all — `OptKind` is AdamW/RMSProp only — so its per-example
partner had no replacement. ⭐ The coverage is not lost: `check_adam_prefix` now forms exactly the
pairing that check would have, against `mobilenetv2_adam_train_step.mlir`. ⚠ §4c's note that this
renderer "already has an `sgdParamF` tail" is **wrong** — `sgdParamF` appears only in
`ResNet34RenderB` — so adding a `.sgd` variant would have meant a new `OptKind` case shared with
EfficientNet plus a new artifact and its gates. Declined as scope.

**⚠ What retirement costs, stated plainly.** `MobileNetV2FaithfulPoCPaper.lean` (the per-example §1
fold) and `MobileNetV2TiePoCPaper.lean` (its §1a tie) are now about an artifact that does not
exist. Every theorem in them is unchanged and still true; no committed bytes exercise them, and
both headers say so. That is only acceptable because their batched peers landed first —
`MobileNetV2FaithfulPoCPaperG.lean` (4b.4) and `Nets/MobileNet/MobileNetV2TiePoCB.lean` (§4.2c), the
latter the same day. ⭐ **This is the ordering rule leg 1 wrote down, honoured deliberately for the
first time**: §4.2 was done before the retirement rather than alongside it.
`MobileNetV2RenderPC.lean` is NOT affected — its per-example net is the subject of the float
budgets, the T6 tie and the witness, none of which is about a train step's bytes.

## Leg 3 — ConvNeXt-T ✅ DONE 2026-09-07 — the only leg that moved bytes

**The decision was the user's (2026-09-07), and the measurement came first.** Different shape from
legs 1 and 2: ConvNeXt has no BN-world split (LayerNorm, train == eval) and its two chains render
the same FORWARD byte-for-byte; what differs is the BACKWARD, 78 lines of the conv input-VJP's
`transpose`/`reverse` in a commuting order (`convBack` per-example, `convBackBatched` batched —
disjoint axes, one kernel). Rendered off the batched chain before a writer moved: the four
drop-free forwards (`convnext_fwd`, `convnextin_fwd`, `convnextsin_fwd`, `convnextbin_fwd`)
**byte-identical**, and each of the thirteen AdamW/EMA train steps differing on **exactly 78 lines,
all that pair, line counts equal**. After the move, `git diff --stat verified_mlir/` is thirteen
files, 1014 insertions / 1014 deletions, zero non-pair lines.

**The licence.** `planning/archive/xla_pjrt_handoff.md` §0.10's keep = 1 gate had already compared the two
chains (0 of 83,478,846 floats differing after three AdamW steps, `scripts/perturb_conv_vjp.py`
firing at 0.0343 as the negative control); `convnext-adam-tie` re-run on the swapped bytes — the
pre-swap `convnext_adam_train_step.mlir` against the committed one, on XLA — ties at gradient
norm-rel 0.000000 against a 0.000002 reorder control, `%loss` bit-exact, `v` bit-exact on all
27,827,818 coordinates, 0/182 parameters disagreeing. ⛔ Not re-run: the 84.41% 80-epoch Imagenette
run. Its artifact's bytes moved on those 78 lines and nothing else, and the numeric gate says the
function did not.

**What landed.**

1. **`Nets/ConvNeXt/ConvNeXtFaithfulPoCGB.lean` FIRST** (18 declarations, ~2 s, 3-axiom clean) —
   the §1 fold at the batched constructors, the peer of 4b.2's `ConvNeXtFaithfulPoCG.lean`. ⭐⭐
   **For this net the fold was OWED before the swap**: every `convnextin_*` train step, every
   `*drop*` variant and the S/B artifacts had rendered from the batched chain since they existed,
   so the artifact behind the quoted ImageNet accuracy (`convnextin_adamdpwxclipdrop`) had a fold
   only at the per-example constructors no committed byte of it is `pretty` of. 4b's "one lemma per
   op kind certifies every optimizer tail" was, for ConvNeXt, a statement about the Imagenette
   pair. No new mathematics — `Finset.sum_congr rfl` over the batch, then the per-example bridge at
   `batchSlice n`; the 22 channel-LN sites add one `batchSlice_batchMap` step because the render
   hands the LN ops the per-example `[h·w, c]` view lifted over the batch.
2. ⭐ **The four bf16 weight-gradient nodes are stated, for the first time in any fold file.** The
   bf16 artifacts emit `convWeightGradBBf16`, `depthwiseWeightGradBBf16`,
   `convStridedWeightGradBBf16` and `convStride4WeightGradBBf16` — not the f32 nodes — and their
   `den` is ONE rounding outside the batch sum of the certified VJP at rounded operands. ⚠ The r34,
   B0 and MobileNetV2 fold headers say their bf16 twins "consume the same node"; those renders emit
   the `*Bf16` constructors too (`ResNet34RenderB.lean:409`, `EfficientNetRender.lean:158`), so the
   sentence is loose there in the same way and their folds should gain the same four-line lemmas.
3. **The seventeen writers moved** from `ConvNeXtRender.lean` to `ConvNeXtRenderB.lean`, each with
   its own comment record, calling `convNextAdamTrainStepFaithfulB` / `convNextFwdRenderB`. ⭐ The
   two wrappers take their arguments in the SAME positions (the batched one delegates to the
   per-example one with `traversal :=`), so unlike ViT's leg there was no positional trap; the
   per-artifact byte diff was run anyway. `cnxFwdPerExampleBanner` became `cnxFwdBanner V`, derived
   from the size table exactly as the per-example line is, so S and B carry their own names.
4. **`tests/TestConvNeXtFwdBTie.lean` flipped**: the committed bytes are now the batched chain's,
   so the gate renders the PER-EXAMPLE chain and compares it — forward byte-identical, train step
   identical apart from the pair. Same statement, other side; load-bearing as long as both chains
   exist.
5. Guards: `proofs.yml`'s ConvNeXt diff list moved from the `ConvNeXtRender` step (now one file)
   to the `ConvNeXtRenderB` step; `regen_verified_mlir.sh`'s SD-pair comment corrected (the
   `convnext_fwd`/`convnext_train_step` pair in `check_fwd_prefix` is CROSS-chain now, and holds
   because the forwards are byte-identical); the yaml rows and status section 4q.

⛔ **One artifact did NOT move, and it is the ordering rule again.** `convnext_train_step.mlir` is
the SGD-inline step; `convNextBackAllB` has no fused-SGD arm, and ConvNeXt's T3 §1a tie —
`ConvNeXtTiePoC.lean`, all 182 parameters — is stated at exactly those bytes. Its batched peer is
4b's ConvNeXt capstone, which this leg unblocks. So `ConvNeXtRender.lean` keeps that one writer and
its traversal, and nothing is orphaned. ⭐ ConvNeXt is one chain per net on 35 of 36 artifacts.

**Gates.** `lake build Certs` 3993 → **3994**; `lake env lean tests/AuditAxioms.lean` 3-axiom clean
on all eighteen; `lake exe docstring-checkrefs` 1668; `check_audit_coverage.py`,
`check_render_coverage.py` (241 files, one writer each); `regen_verified_mlir.sh check` (20 paired /
0 known-split / 0 unaccounted); `convnext-fwd-b-tie` (flipped) green; `convnext-adam-tie` old vs new
as above.

## Leg 4 — ViT-Tiny ✅ DONE 2026-09-07

**The leg started with a measurement and the measurement is the licence.** All **19 of 19**
drop-free ViT artifacts — `vit_fwd`, `vitin_fwd` and the seventeen AdamW/EMA train steps —
re-render **byte-identically** off the batched traversal (`ViTRenderB.vitBackAllB`), checked
whole-net before a single writer moved. `git diff verified_mlir/` after the swap is **empty**.

That is the outcome this log predicted, and the prediction's reasoning held: the 78-line
divergence that makes ConvNeXt's leg a decision is entirely `convBack` vs `convBackBatched`, and
ViT's 16×16 patch embed is not a `conv2d`, so there is nothing for the two chains to disagree
about.

**What landed.**

1. **`Nets/ViT/ViTFaithfulPoCGB.lean` FIRST** (10 declarations, ~2 s, 3-axiom clean) — the
   §1 fold at the batched constructors, the batched peer of 4b.3's `ViTFaithfulPoCG.lean`. ⭐ No new
   mathematics: each proof is `Finset.sum_congr rfl` over the batch and then the per-example bridge
   at `batchSlice n`, because every batched `den` arm is literally the per-example one under a
   batch sum. `ResNet34FaithfulPoCB.denseWGradB_den`'s shape.
2. **The nineteen writers moved** from `ViTRender.lean` to `ViTRenderB.lean`, calling
   `vitAdamTrainStepFaithfulB` / `vitFwdRenderB`. ⚠ The two wrappers take the batch in DIFFERENT
   positions (`fn bStr replicas bs nClasses …` against `fn bStr replicas nClasses … (vbB := …)`),
   which is the one place a positional slip would have shipped a wrong artifact; the byte diff is
   what catches it, and it was run per artifact.
3. `tests/TestViTFwdBTie.lean`'s standing caveat, retired — see below.
4. Guards: `proofs.yml`'s ViT diff list moved from the `ViTRender` step to the `ViTRenderB` step
   (one filename left behind), `regen_verified_mlir.sh`'s SD-pair comment corrected, the yaml row
   and status section 4p.

⭐⭐ **The bytes did not move and one DENOTATION did, and that is the leg's actual product.** The
CLS token is one shared `[192]` vector, so its gradient is the sum of every example's CLS-row
cotangent. The per-example render emits `denseBiasGradB (N := 1)` — "sum one thing", correct there
because `pretty B` performed the batch lift OUTSIDE the AST — where the batched one emits
`(N := vbB)` and the sum is inside `den`. Same emitted text either way (`biasGrad`'s reduce takes
the `B` axis regardless), so the byte tie **provably cannot see it**; `den_rowDenseBiasGradB_at_one`
is the general form of the trap. `vit-fwd-b-tie` had been printing that as a standing ⚠ ever since
the batched chain landed, and `Proofs.ViTPoCGB.clsGrad_denB` closes it — the gate now says so.

⛔ **One artifact did NOT move, and it is the ordering rule.** `verified_mlir/vit_train_step.mlir`
is the SGD-inline step; `vitBackAllB` has no fused-SGD arm (`vitBackAll` takes an `adam : Bool` and
the batched peer only ever emits the raw gradient), and ViT's T3 §1a tie — `ViTTiePoC.lean`, all
200 parameters — is stated at exactly those bytes. Its batched peer is 4b's last open capstone,
which this leg unblocks. So `ViTRender.lean` keeps that one writer and its per-example traversal,
and **nothing is orphaned** — the cost legs 1 and 2 both paid is not paid here.

⛔⛔ **What the audit mispriced, and it is a NEW shape of the "named gap can be the wrong one"
rule.** `planning/archive/proofs_tier_to_paper_nets.md` said leg 4 would be *"a one-line renderer change
with no artifact movement and no decision to take"*. Right about the renderer; wrong about the
tier. **Byte-identity is not tier-identity**: the two traversals emit different CONSTRUCTORS
(`veclnGammaGradB` against `veclnGammaGrad`, `rowDenseWeightGradB` against `rowDenseWeightGrad`, and
so on for all ten), and every `den` lemma is about the AST, not about the bytes. Swapping the
writers without step 1 would have left 4b.3's ten ViT lemmas about an AST that no committed
artifact is `pretty` of — leg 1's orphaning cost, incurred by a change that moves no bytes at all.
▶ **A leg whose artifacts are byte-identical still moves the tier, because a tier is stated about
the graph.** That is the fourth instance of the rule and the first where the gap was invisible to
every byte-level gate in the repo.

⚠ **No re-runs, and this time the check was cheap.** Legs 1 and 2 each had to establish that the
per-example driver's Imagenette number was vacuous. Here no number is at risk at all: the bytes are
unchanged, so every ViT run ever made is a run of exactly the artifacts still committed.

**Gates.** `lake build Certs` 3992 → **3993** green; `lake env lean tests/AuditAxioms.lean` 3-axiom
clean on all ten new declarations; `lake exe docstring-checkrefs` 1668 citations;
`python3 scripts/check_audit_coverage.py`; `python3 scripts/check_render_coverage.py` (241 files,
one writer each); `bash scripts/regen_verified_mlir.sh` — writer audit **252 artifacts, one writer
each**, `check_fwd_prefix` green, `check_adam_prefix` **20 paired, 0 known-split, 0 unaccounted**,
`git diff verified_mlir/` empty; `.lake/build/bin/vit-fwd-b-tie` byte-identical on all 14,457 lines.

## The ImageNet tier of `check_adam_prefix` ✅ DONE 2026-09-06

The item legs 1 and 2 both carried. `check_adam_prefix`'s PAIRS list held only the seven Imagenette
names for a year, so `resnet34in_fwd` and `mobilenetv2in_fwd` were each split exactly as their
Imagenette twins were and **both were moved by hand, not because a guard fired**. The list now
covers the ImageNet artifacts too: **20 paired, 0 known-split, 0 unaccounted**, up from 7.

⭐ **The result was not what this log predicted, and the difference matters.** The expectation was
more instances of the r34/mnv2 split — a forward from one renderer against a train step from
another. There are none. Every ImageNet forward that HAS a train step at its configuration is
already a byte-identical prefix of it, including ConvNeXt-S/B and ViT-S/B, which legs 3 and 4 have
not touched. The forwards that do not pair are unpaired for a completely different reason:

⚠ **Six ImageNet forwards have no partner because no train step exists at their `(batch, drop)`
configuration** — `convnextsin_fwd`, `convnextbin_fwd`, `vitin_fwd`, `vitsin_fwd`,
`vitsin_drop_fwd`, `vitbin_fwd`. Measured: every one diverges at a `broadcast_in_dim %dp0` drop
site or at the batch dimension of the first op, **never at a BatchNorm or a convolution**. Three
are drop-free forwards whose net ships only drop-bearing train steps (the drop-bearing peer IS
paired); three are rendered at B=32 or B=256 against train steps at B=128. They are listed in a
`NO_PARTNER` dict with the reason, so the omission is a recorded decision rather than a silent gap.

⭐⭐ **The real deliverable is the completeness assertion, not the twelve new pairs.** The script now
fails if any `*in*_fwd.mlir` on disk is in NEITHER `PAIRS` nor `NO_PARTNER`, so an ImageNet forward
can no longer be added without being classified — which is precisely how the two instances legs 1
and 2 fixed by hand stayed invisible. It earned its keep immediately: writing PAIRS from the
data-parallel artifacts alone missed `efficientnetin_drop_fwd`, which has no DP peer, and the
assertion caught it on the first run. Both failure modes carry a negative control (an unclassified
forward → exit 1; a mis-pointed pair → exit 1 with the diverging line printed).

## Open

* ⭐ `KNOWN_SPLIT` is empty. The dict is kept, not deleted: an entry appearing again is the §3d(b)
  failure recurring, and it should have to be argued for rather than silently re-added.
* ⭐ All four legs are done (2026-09-07). Every net is one chain.
* ViT's `vit_train_step.mlir` and ConvNeXt's `convnext_train_step.mlir` — one artifact each, still
  per-example. Each moves when its batched traversal gains a fused-SGD arm or its `*TiePoC.lean`
  gains a batched peer (4b's last two capstones, which legs 3 and 4 unblocked). Neither is scoped
  here.
* ✅ 4d piece 2 (2026-09-07): the all-reduce is an AST node in every `*dp*` artifact, swapped in
  byte-identically the way leg 4 was — see `proofs_tier_to_paper_nets.md` §4d.2.
* `convBack` and `convBackBatched` — two emitters for one VJP — are still not tied to each other in
  Lean. The byte tie records the pair and the numeric gate the equivalence; a `den`-level lemma
  that the two constructors denote the same function would make the 78-line allowance a theorem.
* ⚠ The bf16 `*GradBBf16` nodes: stated in ConvNeXt's batched fold, not yet in r34's, B0's or
  MobileNetV2's (leg 3, item 2). Four lines each; batch them into the cleanup pass.

## Seams outside Lean, for every leg

* `scripts/regen_verified_mlir.sh`: `check_fwd_prefix`'s PAIRS (a forward must be a byte-prefix of
  the train step it is paired with — moving the forward moves the pair), `check_adam_prefix`'s
  `KNOWN_SPLIT` ratchet, and the two regen lists.
* `scripts/check_render_coverage.py` and `.github/workflows/proofs.yml`'s diff list — a new artifact
  or a moved writer needs adding, and no local build runs the coverage check.
* `LeanMlir/VerifiedTrain.lean`: the un-varianted load at :988 (`VerifiedNet.train`) and the
  variant-resolved one at :1365 (`trainAdamPacked`, with its per-variant `<slug>_<variant>_fwd`
  fallback and the BN-world assertion).
* The book's chapter numbers, for any net whose quoted accuracy moves.

## Traps, carried from §4c

* **The wrong thing typechecks.** A `broadcast_in_dim` mask against a per-example node compiles,
  trains and descends with no `den` behind it (`ConvNeXtRenderB` header).
* **On ViT the token axis and the batch axis are both called `N`.** Pass the token count by name.
* **A flag that reaches the emission but not the entry name** ships an artifact whose `@name`
  disagrees with its path — three nets, four times.
* **`pretty B` of a per-example node and the `*B` constructor of the same op emit the same bytes.**
  That is what the byte ties are, and it is why a converged render is checked byte-for-byte against
  the artifact it replaces on the forward and numerically on the backward.
* ⭐ **Extracting a traversal into a helper preserves SSA numbering**, because `pretty`'s counter
  follows the call sequence and the sequence is unchanged. That is what makes the R50 refactor a
  byte-identical one for the train step.
* ⛔ **Assume any render docstring's parameter census is the `convBias := true` one** until the
  artifact is counted. Three files were caught on this on 2026-09-06 (`MobileNetV2FaithfulPoCPaper`
  said 210 against 158; `ResNet34TiePoC` and `ResNet34Render` said 146 against 110);
  `resnet34AdamTrainStepFaithfulB`'s own "515 inputs, 146 θ" is the fourth, against a measured 407.
