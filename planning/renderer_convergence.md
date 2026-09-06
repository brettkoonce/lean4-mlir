# 4c — one chain per net: the renderer convergence

**Opened 2026-09-06 as its own thread, out of `planning/proofs_tier_to_paper_nets.md` §4c.** The
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
shrink, never grow". ConvNeXt and ViT are not in it, because their `convnext_fwd`/`vit_fwd` and
`convnext_adam_train_step`/`vit_adam_train_step` are BOTH from the per-example renderer — internally
consistent, and the split there is between the drop-free Imagenette pair and everything else.

## State of play (2026-09-06)

| net | what is split | leg |
|---|---|---|
| **ResNet-34** | — | 1 ✅ **DONE 2026-09-06** |
| **MobileNetV2** | — | 2 ✅ **DONE 2026-09-06** |
| **ConvNeXt-T** | the drop-free Imagenette pair is per-example; every `*drop*` and every `*in*` is batched | 3 ← NEXT |
| **ViT-Tiny** | as ConvNeXt | 4 |
| EfficientNet-B0, ResNet-50, MNv4 | nothing — one renderer already | — |

⭐⭐ **`KNOWN_SPLIT` IS EMPTY.** `check_adam_prefix` reads **7 paired, 0 known-split, 0
unaccounted** — the criterion this thread was opened to reach. Legs 3 and 4 are a different shape
(no BN-world split at all) and are not in that ratchet.

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
`MobileNetV2FaithfulPoCPaperG.lean` (4b.4) and `Foundation/MobileNetV2TiePoCB.lean` (§4.2c), the
latter the same day. ⭐ **This is the ordering rule leg 1 wrote down, honoured deliberately for the
first time**: §4.2 was done before the retirement rather than alongside it.
`MobileNetV2RenderPC.lean` is NOT affected — its per-example net is the subject of the float
budgets, the T6 tie and the witness, none of which is about a train step's bytes.

## Legs 3 and 4 — ConvNeXt-T and ViT-Tiny

Different shape: these two do NOT have a BN-world split (LayerNorm, train == eval), and their
per-example and batched chains render the same FORWARD byte-for-byte (`convnext-fwd-b-tie`,
`vit-fwd-b-tie`). What differs is the BACKWARD: 78 lines of conv-VJP `transpose`/`reverse` in a
commuting order. So the leg is a SWAP, not a re-render, and the gate that licenses it
(`convnext-adam-tie`) is IREE-linked and does not link on this box.

⭐ Per §4c the swap goes under an XLA-side numeric gate instead — a one-batch A/B of the two
artifacts' outputs, the shape the `*-dp-check` gates already have.

## Open, carried from legs 1 and 2

* ⚠ **`check_adam_prefix`'s PAIRS list covers only the Imagenette artifacts.** `resnet34in_fwd` was
  split the same way and no audit saw it; `mobilenetv2in_fwd` was too, and leg 2 moved it by hand
  for the same reason rather than because a guard said so. Add the `*in_*` forwards and their train
  steps — `resnet34in_fwd`/`resnet34in_mom256`, `mobilenetv2in_fwd`/`mobilenetv2in_adam64`, and the
  ConvNeXt/ViT/B0/MNv4 peers — before declaring legs 3 and 4 done. **This is now the only carried
  item, and it is the one that would have caught both instances.**
* ⭐ `KNOWN_SPLIT` is empty. The dict is kept, not deleted: an entry appearing again is the §3d(b)
  failure recurring, and it should have to be argued for rather than silently re-added.

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
