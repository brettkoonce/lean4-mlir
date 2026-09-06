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
| **MobileNetV2** | `mobilenetv2_fwd` (per-example) vs the batched Adam/RMSProp family | 2 |
| **ConvNeXt-T** | the drop-free Imagenette pair is per-example; every `*drop*` and every `*in*` is batched | 3 |
| **ViT-Tiny** | as ConvNeXt | 4 |
| EfficientNet-B0, ResNet-50, MNv4 | nothing — one renderer already | — |

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

## Leg 2 — MobileNetV2 ← NEXT

The last `KNOWN_SPLIT` entry. Same split, same fix: factor the forward traversal out of
`MobileNetV2RenderB` and render `mobilenetv2_fwd` from it. That file is AdamW/RMSProp-only and
already has `OptKind` and an `sgdParamF` tail in the RMSProp arm, so §4c's "gains the `.sgd` tail"
is small. ⚠ Unlike r34, this net's per-example render is SGD-inline only (no `adam` flag anywhere
in it), which 4b.4 already recorded — so there is no one-traversal-two-endings structure to
preserve here.

⛔⛔ **The ordering rule leg 1 established applies, and MobileNetV2 does not satisfy it yet.** A leg
must not retire a per-example renderer before the batched tier that replaces it exists. ResNet-34
was safe because 4.1b–4.1e and 4.2a landed the same day; **MobileNetV2's batched column is empty**
— no batch-BN T1, T2 or T3 (`planning/proofs_tier_to_paper_nets.md` §4.2). So leg 2 is either

* **§4.2's MobileNetV2 half first, then the full leg** — the recommended order, and r34's sequence
  replayed. `MobileNetV2BackB0.lean` carries the same `*BackBatchedGraph_faithful` family
  `ResNet34BackB0.lean` does, so the `_eq_vjp` lemmas take the same `rfl` route 4.2a found; or
* **the renderer only** — converge `mobilenetv2_fwd` onto the batched chain and leave
  `mobilenetv2_train_step.mlir` and its per-example renderer alive, so
  `MobileNetV2FaithfulPoCPaper` / `MobileNetV2TiePoCPaper` keep a live artifact. That empties
  `KNOWN_SPLIT` without orphaning anything, and defers the retirement.

⚠ Check before starting, and do not assume r34's answer: **does MobileNetV2's per-example SGD
trainer produce a number anyone quotes?** r34's did not (chance, and its header said so), which is
what made leg 1 free of re-runs. This net's driver has not been looked at.

## Legs 3 and 4 — ConvNeXt-T and ViT-Tiny

Different shape: these two do NOT have a BN-world split (LayerNorm, train == eval), and their
per-example and batched chains render the same FORWARD byte-for-byte (`convnext-fwd-b-tie`,
`vit-fwd-b-tie`). What differs is the BACKWARD: 78 lines of conv-VJP `transpose`/`reverse` in a
commuting order. So the leg is a SWAP, not a re-render, and the gate that licenses it
(`convnext-adam-tie`) is IREE-linked and does not link on this box.

⭐ Per §4c the swap goes under an XLA-side numeric gate instead — a one-batch A/B of the two
artifacts' outputs, the shape the `*-dp-check` gates already have.

## Open, carried from leg 1

* ⚠ **`check_adam_prefix`'s PAIRS list covers only the Imagenette artifacts.** `resnet34in_fwd` was
  split the same way and no audit saw it. Add the `*in_*` forwards and their train steps —
  `resnet34in_fwd`/`resnet34in_mom256`, `mobilenetv2in_fwd`/`mobilenetv2in_adam64`, and the
  ConvNeXt/ViT/B0/MNv4 peers — before declaring any later leg done.
* ⚠ `KNOWN_SPLIT` has one entry left. When leg 2 lands it is empty, and the dict itself becomes the
  thing to delete rather than a list to maintain.

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
