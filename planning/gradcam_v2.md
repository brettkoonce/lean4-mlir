# gradcam_v2.md — Grad-CAM / CAM demo, second pass

Goal: grow the CAM demo along the three axes v1 deliberately
deferred: **coverage** (2 hardwired models → the CAM-eligible zoo,
plus the comparison figure the v1 plan specced but never produced),
**trust** (a quantitative faithfulness metric instead of "the
heatmap looks right"), and **reach** (true gradient-based Grad-CAM
for the non-closed-form heads — now much cheaper than v1 priced it,
because the whole-network backward emit landed in the meantime).
Plus the proof tie-in this demo is unusually well-shaped for: the
closed-form shortcut at the heart of v1 is a two-line theorem.

Prerequisite reading: `planning/gradcam.md` (v1 plan; its Phase 1
shipped essentially as written).

## Where v1 landed (recap, one paragraph)

Phase 1 is complete and committed (`ae7e4ac`, `d1f7002`): for any
spec ending `… → globalAvgPool → dense`, CAM collapses to
`ReLU(Σ_k W[c,k]·A^k)/HW` — no autodiff. Shipped:
`generateForwardCam` (a `stopAtGAP` forward walk that returns the
pre-GAP activation), `preGAPShape` eligibility check, FFI kernels
(`camCompute`, `camLogits`, `bilinearUpsample2D`), `Cam.lean`
(hardcoded viridis LUT, α-blend overlay, PPM writers),
`MainGradCAM.lean` with two pre-wired Imagenette models
(ConvNeXt-T-GELU, ResNet-34) emitting input | overlay | heatmap
strips, `TestCam.lean` hand-computed sanity check, and two blueprint
figures (`demos/figures/gradcam_{resnet34,convnext_t}.png`).

## What's open (v1's own deferrals, audited 2026-07-03)

1. **Coverage stopped at 2 of the ~8 eligible zoo members.** The v1
   plan scoped ResNets, MobileNet-V2/V3, EfficientNet, ConvNeXt,
   and the small CNNs; only ConvNeXt-T and R34 got wired. The
   plan's headline figure — **one image, four CNNs** ("architecture
   choice changes what the network looks at") — was never made.
2. **`--class` flag never shipped** (argmax only); no image
   selection either (always the first N val records at a hardcoded
   batch of 32).
3. **No quantitative story.** The demo's claim is "the heatmap
   carries codegen pedigree," but nothing measures whether the
   heatmap is *faithful* (would occluding the hot region actually
   change the prediction?).
4. **Phase 2 (ViT / true Grad-CAM) was punted** (option 2c) at an
   estimated "~6 hr, high risk" — priced *before* the
   whole-network-backward work existed.
5. **Operational**: no Imagenette classifier checkpoints currently
   exist in `.lake/build` (only MNIST + VJP-oracle fixtures) — the
   two models `MainGradCAM` points at need their training runs
   re-executed (or artifacts fetched from mars) before *any* v2
   work can be demonstrated. Budget this first.

## Repricing that changes the picture

v1's Phase 2a ("`gradcam_step` MLIR module: forward + backward from
a `−onehot` seed, stop at the chosen activation, no param grads")
was rated high-effort/high-risk because per-layer backward coverage
was uneven. Since then the **whole-network backward emit landed for
every layer kind, including the transformer stack** (the
ViT/ConvNeXt backward-graph-faithfulness and whole-net VJP work).
The train-step backward *already computes activation gradients as
intermediates* at every layer boundary — a `gradcam_step` is that
walk with (a) a onehot-row seed instead of `(softmax−onehot)/B`,
(b) an early stop at the capture layer, (c) `d_act` returned and
all param-grad emission skipped. It reuses the existing dispatcher
arms untouched. Re-estimate: **1–2 sessions**, same tier as the
other demos' codegen items, not a research risk.

## Workstream A — coverage + CLI (the unfinished 20% of v1)

1. **Regenerate checkpoints** (operational, mostly wall-clock):
   retrain or fetch the Imagenette runs for R34, ConvNeXt-T,
   EfficientNet-B0, MobileNet-V2 (each is an existing exe;
   Imagenette trainings are hours, not days, and klawd can run
   them in parallel with the usual duty-cycle caveat).
2. **Model registry instead of two hardwired specs** (~half
   session): table of `(key, NetSpec, ckptPfx)` covering every
   CAM-eligible zoo member with a trained checkpoint; the
   eligibility check (`preGAPShape` + trailing-dense) already
   exists, and v1's refuse-loudly behavior stays.
3. **CLI polish** (~half session): `--class N` (v1's own table
   promised it), `--indices i,j,k` to pick specific val images,
   and a `--panel` mode that renders **one image × N models** —
   the missing headline figure.
4. Regenerate both v1 figures + the new four-CNN comparison panel
   from the fresh checkpoints.

Gate A: the four-CNN panel exists and the per-model strips
reproduce. Pure demo debt, zero new math.

## Workstream B — faithfulness metric (trust, no codegen)

The standard deletion/insertion test needs only the existing eval
forward and host-side loops:

- **Deletion curve**: rank pixels by CAM value, progressively mask
  the top-k% (replace with dataset mean), re-run the forward,
  record `p(target)` vs k. Faithful heatmaps → steep early drop
  (low AUC). **Insertion** is the mirror (start from blur/mean,
  reveal top-k%).
- Implement as a host loop over k ∈ {1, 5, 10, 20, 50}% × N val
  images, batch the perturbed images through the eval vmfb. One
  script/exe, ~1 session.
- Report deletion/insertion AUC per model in RESULTS.md, and — the
  honest control — the same curves for a **random-ranking
  baseline**. CAM beating random by a wide margin is the
  quantitative version of "the heatmap works."
- **Cross-demo option** (cheap, cute): a cat/dog classifier
  fine-tuned from the R34 checkpoint on the Pets images (both
  already in-repo) gives ground-truth foreground masks from the
  UNet demo's trimaps — report **CAM energy inside the mask**. Zero
  new annotation; connects three demos. Optional garnish, not the
  core metric.

Gate B: CAM deletion-AUC ≪ random baseline on every registered
model. If any model *fails* this, that's a finding, not an
embarrassment — investigate before showcasing its heatmaps.

## Workstream C — true Grad-CAM (`gradcam_step`) + the consistency seal

The re-priced v1 Phase 2a, with a twist that makes it
verification-flavored rather than feature-flavored:

1. **`generateGradCamStep`** (~1–2 sessions): backward walk from a
   `−onehot[c]` seed to a designated capture layer, returning
   `(A, ∂y_c/∂A, logits)`. Param-grad emission skipped; existing
   backward arms reused as-is. The capture-layer designation can
   reuse the `stopAtGAP` mechanics generalized to "stop at index
   i" — which is also a step toward the `.saveFeature` primitive
   the YOLO/UNet v2 docs want (three docs now touch this seam;
   build it once, coordinate).
2. **The consistency check** (the on-brand payoff, ~half session):
   on R34/ConvNeXt, assert numerically that `gradcam_step`'s
   α-weights equal `W[c,·]/HW` — the closed form v1 *derived on
   paper* is now checked against the actual compiled backward.
   This is exactly the repo's oracle-test pattern applied to an
   explanation method.
3. **ViT-Tiny Grad-CAM**: capture at the last pre-head token
   feature map, reshape tokens → grid (the `spatialUnflatten` math,
   host-side), heatmap as usual. Grad-CAM-for-ViT via token
   gradients is standard practice; note in the figure caption that
   attention-rollout-style methods are a different (out-of-scope)
   family.

Gate C: consistency check passes to float tolerance; ViT panel
renders something sane on Imagenette (ViT CAMs are blockier —
that's intrinsic, document it).

## Workstream D — the theorem (small, and the whole demo's thesis)

v1's shortcut *is* a mathematical claim: for a head
`dense ∘ globalAvgPool`, the gradient `∂y_c/∂A_{ij}^k = W[c,k]/HW`
is constant in `(i,j)`, hence GradCAM ≡ CAM (up to the positive
factor the ReLU respects). The proof is linearity-of-the-head plus
the existing GAP and dense VJP lemmas — likely a short composition
over the real-valued semantics, no new machinery. Landing it turns
the demo's premise into a sealed statement ("the visualization
shortcut is proved, and Workstream C checked the compiled artifact
against it") — the two halves of the repo's standard claim, applied
to interpretability. ~1 session; genuinely optional but unusually
cheap for what it says.

## Sequencing

```
Phase 0 (wall-clock + ½ session):  A1  regenerate checkpoints
Phase 1 (1 session):               A2–4 registry + CLI + comparison panel (Gate A)
Phase 2 (1 session):               B   deletion/insertion AUC + random control (Gate B)
Phase 3 (1–2 sessions):            C   gradcam_step + consistency check + ViT (Gate C)
Phase 4 (1 session, optional):     D   CAM≡GradCAM lemma
```

Phases 0–2 are the committed core (≈2–3 sessions + retraining
wall-clock): they finish v1's own spec and give the demo a number.
Phase 3 is the reach extension whose codegen overlaps the
`.saveFeature` seam shared with the YOLO/UNet v2 plans; Phase 4 is
the cherry.

## Deliverables

- Model registry + `--class` / `--indices` / `--panel` in
  `MainGradCAM.lean`; refreshed strips + the one-image-four-CNNs
  figure in `demos/figures/` and the blueprint
- Deletion/insertion AUC table (with random baseline) in
  RESULTS.md; optional Pets mask-energy column
- If C lands: `generateGradCamStep`, the α-weight consistency test
  in `tests/`, and a ViT-Tiny panel
- If D lands: the CAM≡GradCAM lemma in the proof tree + one
  blueprint paragraph closing the loop
- `planning/gradcam.md` gets a status header pointing here

## Out of scope (unchanged from v1, plus)

GradCAM++ / ScoreCAM / Eigen-CAM (still separate plumbing, still
skip), attention rollout and the ViT-interpretability literature
(one caption sentence, no implementation), counterfactual or
contrastive heatmaps, sanity checks of the Adebayo
cascading-randomization flavor (worth a line in the blueprint text;
not worth a harness), user-facing PNG conversion (PPM + `convert`
stays fine).
