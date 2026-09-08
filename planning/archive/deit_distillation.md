# Planning — DeiT: the distillation-objective rung on the ViT chapter

**The thesis:** the book's trainers all optimize one objective (supervised cross-entropy on ImageNet
labels). DeiT (Touvron et al. 2021) is the cleanest *different objective* that's squarely an
"ImageNet trainer" and reuses the net you already finished (ViT). It's the rebuttal to "the future is
just bigger data": DeiT trains a competitive ViT on **ImageNet-1k alone** (no JFT-300M) via a strong
recipe + **distillation from a teacher**. You already have the recipe (crop/flip/LS; the jax track has
Mixup/CutMix/RE/AutoAug). What's missing is the distillation mechanism — and in the verified-trainer
frame, most of it is nearly free.

**Narrative hook (the chapter closer):** DeiT's best teacher was a *CNN* (RegNet), and distilling a
CNN into a ViT transfers the convolutional inductive bias to the attention model. So the architecture
ladder stops being a sequence and becomes a **relay — the CNN era literally tutors the ViT era.** Much
better ViT-chapter ending than "here's another classifier." Chapter one-liner: *"the teacher is just
another label source — distillation is two cross-entropies, and you only verify your half."*

**This is the first brick of the *self-bootstrapping* spine** (see `math_threads.md` / the Part II arc):
the book's own *verified* artifacts become load-bearing inputs to later stages — Muon (proved optimizer)
already trains the ViT; here a proved CNN *teaches* it; later the Muon spectral engine computes the
`‖W‖₂` the Lipschitz cert consumes (`power_iteration_lipschitz.md`). Because every hand-off is
faithfulness-checked, the bootstrap chain compounds *guarantees*, not the errors that make ordinary
distillation/self-training fragile. DeiT is the opening move.

**The headline experiment — the teacher-ladder sweep (turns DeiT from a trainer into a payoff for the
whole book).** DeiT's real finding is counterintuitive: a **CNN teacher beats a transformer teacher for
a ViT student even at lower accuracy** — it's the *conv prior* that transfers, not the accuracy. Your
book is already organized along exactly the axis that tests this, from most-classically-convolutional to
most-transformer-like:
```
ResNet-50 → EfficientNet-B0 → ConvNeXt-T → (ViT student)
   most conv prior ─────────────────▶ most transformer-like
```
Distill *each rung* into the *same* ViT student, everything else fixed, and measure the student.
**Hypothesis:** student quality tracks the teacher's *conv-ness*, not its raw accuracy — so the classic
ResNet-50 teacher may beat the higher-accuracy ConvNeXt-T teacher. Note the ConvNeXt twist: it's the CNN
*deliberately engineered to mimic a transformer* (patchify stem, 7×7 depthwise ≈ windowed attention,
inverted-bottleneck MLP, LayerNorm/GELU), so it sits on the conv↔attention boundary — the *weakest*
clean demonstration of the effect but the *most interesting* research question ("does the bias transfer
survive when the teacher is transformer-ified?"). Use **ResNet-50 as the clean-demo teacher** (the
canonical conv baseline, and what RSB itself studied); run the **full ladder as the research hook**. If the hypothesis holds you've reproduced + sharpened DeiT with
only your own nets, and the teacher axis retroactively justifies the entire architecture progression.
Honest caveats: the effect was measured at **full-ImageNet scale** (imagenette magnitudes may be muted/
noisy → wants the ImageNet run to be conclusive), and it costs one distillation run per teacher.

**The second axis (the accuracy control — disentangles DeiT's claim).** The teacher's *recipe* sets its
*accuracy* while holding the conv prior fixed, so RSB recipe variants of the **same R50** give a clean
accuracy axis orthogonal to the conv-ness axis (published RSB ResNet-50 numbers):
```
recipe (accuracy, R50 fixed):  90-ep baseline (~76.1%) → A3 (78.1%, 100ep/160px) → A2 (79.8%, 300ep) → A1 (80.4%, 600ep)
architecture (conv-ness):      R50 → EfficientNet-B0 → ConvNeXt-T
```
- **Test 1 (recipe sweep, R50 fixed):** vary teacher *accuracy* (~76→80%), hold architecture. If the
  student barely moves while teacher accuracy climbs ⟹ **accuracy doesn't dominate** (the clean DeiT claim).
- **Test 2 (architecture sweep):** if the student tracks *conv-ness* even when ConvNeXt has higher
  accuracy ⟹ **conv-ness is the driver.** Together they separate "architecture vs accuracy" — the
  question DeiT only partially controlled.
- **Default / first teacher = RSB-A3 R50** (the model being trained next — cheap 100ep/160px recipe,
  reuses the existing A2-R50 build infra; ~78.1% is a genuinely solid teacher and the first accuracy-axis
  dot). A2-R50 (already built, 79.8%) and A1-R50 (80.4%) are the stronger-accuracy extensions; the
  90-epoch baseline is the weak-teacher probe. DeiT used RegNetY-16GF (~83%); A-recipe R50 is the
  in-house stand-in, A3 the affordable first cut. Pair with **hard distillation** (recipe-robust; the
  paper's preferred method) — soft distillation's dark-knowledge benefits more from the Mixup-trained
  RSB teacher's softer/calibrated outputs, so reserve the recipe-quality sensitivity for the D5 KL path.

## 0. What DeiT adds to a vanilla ViT

Vanilla ViT (have it: `ViTRender.vitFwd`, `vit_net_tied_certified` in `ViTTiePoC.lean`,
`MainViTVerified`/`MainViTVerifiedAdam`): prepend a `[class]` token (`clsPosFwd`, row 0), run blocks,
slice the class row and run one head (`headFwd`: `[b,n,d]` row 0 → `[b,d]` → `Wc[d,nc]` → logits) →
CE vs the true label.

DeiT adds a **second learnable `[distillation]` token** alongside the class token, a **second head**
on that token, and a **two-target loss**:
```
loss = ½·CE(class_head,    y_true)
     + ½·CE(distill_head,  y_teacher)
```
- **Hard distillation (do this first):** `y_teacher` = the teacher's argmax → a plain hard label →
  the second term is *just another CE*. Parameter-free, no temperature, maximal reuse.
- **Soft distillation (later):** temperature-scaled KL between student/teacher softmaxes — needs a
  KL/temperature backward op the codegen does not have yet.

At inference, fuse the two heads (average the softmaxes).

## 1. Layered plan (engineering vs. verification cleanly split)

- ⬜ **D1 — the distillation token (forward render).** In `ViTRender.lean`, generalize `clsPosFwd`
  to prepend/append a *second* `[d]` learnable token → sequence `[b, n0+2, d]`; add a second
  `headFwd`-style slice+dense on the distill row (parameterize `headFwd` by the slice row, or clone
  it). Blocks are sequence-length-agnostic, so attention/MLP are unchanged. **The only architectural
  change.** Mirror `clsPosBack`/`headBack` for the backward (two head-backs; their `dz` contributions
  pad into `[b,n,d]` at rows 0 and the distill row, then **sum** before flowing into the block
  backward — sum-of-VJP, already in hand).
- ⬜ **D2 — the two-target loss.** In `Train.lean`, the DeiT loss is `½ CE+LS(cls, y_true) +
  ½ CE+LS(distill, y_teacher)`. Hard distillation ⟹ both terms are the *existing* CE+label-smoothing
  op on two logit tensors against two label tensors. Gradient = sum of two CE backward paths.
- ⬜ **D3 — the teacher (pure engineering, ZERO verification).** A frozen classifier emitting one
  target per image (argmax for hard distillation). **The teacher is data, not a verified component** —
  you never prove anything about it, only that your loss consumes its predictions. The part you called
  "the teacher step"; the *easiest* piece. Two decisions:
  - **Which teacher: use your own CNN.** DeiT's headline finding is that a **CNN teacher beats a
    transformer teacher** for a ViT student — *even at lower accuracy* — because distillation transfers
    the CNN's locality/translation-equivariance prior, exactly what the data-hungry ViT lacks. So pick
    your strongest in-house convnet (ConvNeXt-T or EfficientNet-B0) as the teacher: the relay narrative
    ("CNN era tutors ViT era") becomes the literal training graph, self-contained, no external dep. (A
    stronger external teacher also works — the teacher is data — but the in-house net is the dog-fooding
    win.) The student can *exceed* the teacher: it also sees the true labels (class-token CE), so it
    gets teacher-knowledge + ground truth.
  - **Online vs. cached (the augmentation wrinkle).** Faithful DeiT runs the teacher **online, on the
    same *augmented* batch the student sees, every step** (teacher in eval/frozen) — because
    RandAug/Mixup/CutMix change the view per step, a single cached "label per clean image" wouldn't
    match what the student looks at. Cost: one teacher forward per step. The cheaper **cached** variant
    (run teacher once on clean images, store argmax) is only correct if augmentation is frozen — fine
    for a first smoke build, weaker with strong aug. Either way the teacher stays unverified data.
- ⬜ **D4 — the faithfulness tie extension** (the real bounded work). Extend `vit_net_tied_certified`
  so the certified forward gains the second token + second head, and the certified backward gains the
  second head's path (two `headBack`s summing into the `[b,n,d]` cotangent). The per-token machinery
  (`perRowFlat`/`perRowFlatPR`) already handles an extra token row; the cls token is already
  special-cased — this is "the cls-token swap, but two special tokens + two heads." Effort ≈ the
  cls-token trainer swap you already did, ~doubled.
- ⬜ **D5 — soft distillation (optional, later).** Temperature-scaled KL on the distill head. Needs a
  KL + temperature-scaling forward/backward op the codegen lacks. Strictly-later add; hard
  distillation captures the paradigm without it.

## 2. Honest scoping (tiers, no overclaim)

- **Free / data:** the teacher (D3) and, with hard distillation, the loss (D2) — both reuse existing
  machinery, no new verification.
- **The real work:** the distillation-token thread through the *certified* forward+backward (D1+D4).
  Bounded, comparable to the cls-token swap.
- **Out of scope for v1:** soft distillation (D5, needs a KL op); **descent** on the two-term loss is
  a separate optional follow-up — the *faithfulness tie* (emitted DeiT step = certified math) is the
  deliverable, mirroring how the other nets landed faithfulness before descent.

## 3. Files / wiring

`ViTRender.lean` (D1: token + 2nd head fwd/back), `Train.lean` (D2: two-CE loss), a new
`apps/imagenette/MainDeiTVerified.lean` (D3 + the trainer, cloning `MainViTVerified` + teacher-label
feed), `ViTTiePoC.lean` (D4: extend `vit_net_tied_certified` → `deit_net_tied_certified`), and
`tests/AuditAxioms.lean` for the new capstone. Reference: Touvron et al. 2021 (DeiT), and
[[vit-tie-scope]] / [[section1a-tie-sweep]] for the existing ViT tie recipe + gotchas.

## 4. Session handoff — start at D1

1. `ViTRender.lean`: add the distill token to `clsPosFwd`/`clsPosBack` (or a `clsDistillPosFwd`
   variant) and parameterize `headFwd`/`headBack` by the slice row; validate the new render in a
   `tests/TestViT*` smoke test (shape `[b, n0+2, d]`, two `[b,nc]` heads).
2. `Train.lean` + `MainDeiTVerified`: the two-CE loss + teacher-label feed; smoke-train one epoch.
3. `ViTTiePoC.lean`: extend the tie to `deit_net_tied_certified` (the per-token fold already covers
   the extra row; the new content is the second head's fwd/back tie + the summed cotangent). Keep
   3-axiom clean; add the `#print axioms` line. Soft distillation (D5) only if wanted.
