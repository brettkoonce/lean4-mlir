# The mask chain: step ties through drop-path and classifier dropout

Written 2026-10-07, at the close of planning/bf16_tie.md (tier A done: every whole-net step, sync
and loss-gradient tie is stated at either precision). A starting brief, not yet a plan: the table
and site counts are checked against the job confs, the manifest and the renderers; the proof shape
below is a read of the tie files, not a build. The first session turns it into a plan.

## ▶ Start here (next session)

**The gap.** Every step tie, sync tie and `*_net_lossGrad` is stated on the chain WITHOUT the
training masks. The book's seven jobs (`content.tex` job table, ~18370; each conf's
`LEAN_MLIR_VARIANT` names the artifact) train these:

| job → artifact | drop-path sites | classifier dropout | other outside the tie |
|---|---|---|---|
| `r34-default-bf16-4gpu` → `resnet34in_momdp64bf16` | — | — | — (fully reached) |
| `r50-2018-bf16-4gpu` → `resnet50in_momdp64bf16` | — | — | — (fully reached) |
| `r50-a3-wxclip4x128-bf16-4gpu` → `resnet50in160_lambaccdp4x128wxclipbcebf16` | — (A3 sets `dropPath := 0.0`, `ResNet50RenderB`; the `*drop*` R50 renders are A2's, no job) | — | accumulation |
| `mnv2-default-4gpu` → `mobilenetv2in_rmsdp64wxdols0eps0001bf16` | — | 1 (`%do`, per element, before the dense) | — |
| `enet-default-4gpu` → `efficientnetin_emarmsdp64dropdowxeps0001bf16` | 9 (the skip-carrying MBConvs of 16) | 1 | EMA |
| `cnx-default-4gpu` → `convnextin_adamdpwxclipdroperfbf16` | 18 (one per block) | — | — |
| `vit-default-emabf16-4gpu` → `vitin_emadp128x4wxclipdropeps0000001erfbf16` | 24 (two per block) | — | EMA |

Side quest, not a book job: MobileNetV4-Conv-M `mnv4in_emaaccdp8x128wxdowd005bf16` (dropout;
EMA, accumulation) and its `*wxdropdowd01bf16` axis sibling (drop-path + dropout).

EMA and accumulation are optimizer tails that consume the gradient nodes; the ties already say
they're outside, and that's a different kind of statement. So the masks are the last gap at the
gradient nodes for four of the seven jobs, and for ConvNeXt and ViT they are the only one — every
`convnextin_*bf16` and `vitin_*bf16` artifact in the manifest carries `drop`, so tier A's "at
either precision" for those two nets is stated at a chain no bf16 artifact has.

**Why it should be tractable.**
* Both masks are a diagonal scaling and the VJP is the op itself at the same mask:
  `Proofs.dropPath_vjp_is_self`, `dropout_vjp_is_self` (Training/DropPath.lean);
  `dropout_of_dropScale` says drop-path IS dropout at a lifted (per-example-constant) mask. Every
  renderer's backward emits it that way and cites the lemma (`ConvNeXtRenderB`, `ViTRenderB`,
  `ResNet50RenderB`, `EfficientNetRender/Basic`, `MobileNetV2RenderB`, `MobileNetV4RenderB`).
* The masked FORWARDS are already proved for three nets, with the sites as `Option` binders —
  `none` = no node = the drop-free artifact, `some s` = the `dropPathB` / `dropoutB` node
  (`dropPathOptG` / `dropoutOptG`, EfficientNetFullB0Drop.lean): `efficientnetFwdGraphBFullDrop`
  (`sd : Option (Fin 9 → Vec N)`, `cd : Option (Vec (N * 1280))`), `mnv4FwdGraphBFullDrop` and the
  `Do` heads (MNv2, MNv4), and ViT's `vitFwdGraphBDrop_faithful` (ViTFwdDrop.lean, masks
  `sdA sdM : Fin k → Vec B`, per-example `blockVDrop`). ConvNeXt and ResNet-50 have no typed drop
  forward graph (ConvNeXt has no typed batched forward at all — bf16_tie.md, ConvNeXt note (a)).
* Where the mask lands in the chain decides the cost. The CNN ties are BATCHED chains (BN), so a
  site is one batched node, `dropPath N n s` / `dropout m`, dropped into the chain where the
  renderer puts it; the node predicates (`ConvWTiedBAt … x … dy`) do not change, only the `x` (the
  dropped activation — gate W below) and `dy` they are fed. ConvNeXt's and ViT's ties are
  `batchMap` / `batchMapAux` lifts of a UNIFORM per-example function (Foundation/Batched/Basic.lean
  has only those two; `HasGradAt.param_batchMap_through` likewise), and a per-example mask makes
  it a different function per example. `vitForwardKVDropB` dodges this by being defined pointwise
  (`fun x idx => let p := …`), which the backward cannot do under the ParamGrad lift. So the LN
  nets need either an indexed lift (`batchMapIdx` + its `_through`) or the mask carried in
  `batchMapAux`'s saved slot (`Vec (N * s)`, per-example-sliced) beside the activation. Design
  this before touching ViT; it is the one piece of new Foundation in the thread.

**First session.**
1. Read: `ConvNeXtStepTieGB` / `ViTStepTieGB` scope paragraphs (both name the gap),
   `Training/DropPath.lean`, `ViTFwdDrop.lean`, `EfficientNetFullB0Drop.lean`, and how each
   renderer places the backward drop (site order matters — planning/archive/stochastic_depth.md
   §7b: the ones-mask gate is blind to placement; `tests/TestDropPathTie.lean` gate B is the
   numeric control).
2. Statement shape: the sites as `Option` binders, as the forward graphs already have them — the
   existing drop-free tie is the `none` instance verbatim, the drop artifact's is `some s`;
   nothing restated. `dropPath_ones_id` is the separate eval statement (the drop artifact's
   forward at a ones mask is the drop-free function), not how the drop-free tie is recovered.
3. Order, cheapest whole-net win first: **MobileNetV2 classifier dropout** (one batched site, the
   job's only gap — closes `mnv2-default-4gpu` entirely; the dense weight node reads the dropped
   buffer, which is `tests/TestDropoutTie.lean` gate W as a theorem), then **B0** (nine drop
   sites + the dropout, typed drop forward in hand, batched chain), then the lift design, then
   **ViT** (masked forward in hand, 24 sites), then **ConvNeXt** (18 sites, no typed forward,
   `@[irreducible]` stem). MNv4 follows B0's recipe if the side quest is ever written up.
4. Same discipline as bf16_tie: stage per net, stop for the commit word; comparator tier + yaml +
   book sentences move with the Lean (the book's "classifier dropout sits outside that statement"
   / "the drop-path chain" sentences, one chapter per net).

## Background

* planning/archive/stochastic_depth.md — the August design of the drop-path render (host-drawn
  mask inputs, the depth ramp, why the RNG is not in the graph, the gates and their blind spots).
* tests/TestDropPathTie.lean (`lake build droppath-tie`; gate A: the scale is exactly the supplied
  per-example scalar, gate B: the site is on the residual branch) and tests/TestDropoutTie.lean
  (gate A: per element, gate W: `∂L/∂W` reads the dropped activation) — the numeric gates whose
  content the mask chain states.
* planning/bf16_tie.md — the thread that closed the precision gap; its recipe (flag on the
  statement, erase in a first `simp only`, binders by line number) is the template for threading
  a new binder through a net's tie files.
