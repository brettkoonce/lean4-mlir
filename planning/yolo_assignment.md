# yolo_assignment.md — target assignment (detection-infra brick #4)

Handoff for the VisDrone detector's **last untested hypothesis**. Prereqs: `yolo_fpn.md`
(the FPN build + the Tier 1/2 refutations), `yolo_drone.md` (overall plan), memories
`yolo-fpn-thread` + `visdrone-fetch-and-wsa`. Written 2026-07-19, after Tier 2 was
measured out.

## Why this doc exists

Four levers have been built, trained, and refuted. mAP@0.5 was **0.0001 in every one**:

| lever | what it changed | result |
|---|---|---|
| T1a objectness `/num_pos` | loss balance | **refuted before building** — background is 6% of total loss |
| T1b class-weighted CE | class term | works (class spread 5/10 → 7/10 predicted), mAP unmoved |
| T2-bias + RetinaNet prior | head init | **washed out** — converged objectness identical to baseline |
| T2a 4-conv head tower (+7M) | head capacity | **refuted** — didn't even lower the TRAIN loss |

Recall sits at 11.8–12.6% across all of them: one noise band. Every arm converges to the
same objectness equilibrium (gap ≈ +0.25, std ≈ 0.24/0.33, means ≈ −1.57/−1.83).

## The measured constraint — read this before proposing anything

`scripts/fpn_neighbor_separation.py` splits background into cells **8-adjacent to a
positive** ("ring") vs everything else. On the T2a e12 logits (and it replicates on the
T2-bias arm to 3 decimals):

| population | n | mean objectness logit | std |
|---|---|---|---|
| positive | 34,341 | **−1.5818** | 0.249 |
| ring (adjacent) | 192,747 | **−1.6035** | 0.261 |
| far background | 6,539,616 | −1.8509 | 0.329 |

    ring − far  = +0.247     the head KNOWS where the objects are
    pos  − ring = +0.022     it CANNOT split the assigned cell from its neighbour
    ring sits 92% of the way from far background to positive

**The head has solved localization at neighbourhood resolution.** On top of that it is
being asked to pick which one of ~6 cells in the neighbourhood is "the" assigned cell. For
a 2–5px VisDrone object on the 56² P3 grid those cells see essentially the same receptive
field — they are **near-identical inputs, and no function of them can rank them apart**.

That is why capacity, initialization and loss weighting all washed out: none of them
changes the fact that the task as posed is unlearnable. It also explains the pinned mAP
mechanically — **5.6 ring cells per positive, all within 0.022 of the positives**, so the
top of the score-ranked detection list is a near-miss-duplicate avalanche that no
confidence threshold can clear.

**Corollary: stop tuning the network. The next change must be to WHAT IS ASKED OF IT.**

## The hypothesis

Make the ring cells positive too — **multi-anchor-per-GT / FCOS-style center sampling**.

1. It stops demanding an impossible discrimination.
2. It multiplies positives ~6.6×, which the T1a measurement says the loss can absorb
   (positives already carry 75% of objectness loss; this is not a balance regression, but
   see Risk 1).
3. Near-duplicate detections become **NMS-mergeable clusters** instead of false positives.

## ⚠️ BITE 0 — PRE-MEASURE. Do NOT write codegen first.

Measure-before-build has now paid off four times in this thread (T1a refuted on paper,
T1b's ceiling predicted, T2-bias and T2a both explained by one measurement). Do the same
here. **All of bite 0 is numpy on existing artifacts — no training, no GPU, no Lean.**

**0a — oracle recall.** Extend `scripts/visdrone_fpn_coverage.py` with a center-sampling
assignment (every cell whose centre falls inside the GT box, or within radius r of the box
centre, at the scale the size-threshold picks) and report the achievable recall@0.5 ceiling
the way it already reports the 88.2% encodability ceiling. **If the oracle does not move,
stop — the hypothesis is dead and this doc is wrong.**

**0b — the NMS question, which is the real risk.** With ~6.6× positives, adjacent cells
each emit a box. Those merge only if their mutual IoU exceeds the NMS threshold. **At 2–5px
box sizes IoU is brutal** — a one-pixel centre shift can drop IoU below 0.5. So measure, on
the *existing* e12 logits: take the boxes predicted by ring cells around each GT and
compute their pairwise IoU distribution against the positive cell's box. If those IoUs sit
below the NMS threshold, center sampling converts an FP avalanche into a *differently
shaped* FP avalanche and buys nothing without also fixing NMS (see Risk 2).

**0c — duplicate accounting.** From the same logits, quantify how much of the current mAP
loss is actually duplicate-driven: re-score with an oracle that keeps only the
highest-objectness detection within each GT's neighbourhood. That gives an upper bound on
what perfect duplicate handling alone would buy, independent of assignment.

Write results into this doc before touching `preprocess_visdrone.py`.

## The build, if bite 0 says go

**The good news: this is very likely a ZERO-CODEGEN change.** The multi-scale loss masks
positives by the **target's own objectness channel** (`%{pa}_m4`, sliced from the target at
`base+4`) — there is no FFI mask and no hard-coded positive count anywhere in
`emitAnchorYoloLoss`. Making more cells positive in the encoded target is therefore picked
up by the existing, FD-verified loss with **no emitter change at all**.

- **Bite 1 — encoder.** `encode_targets_fpn` in `scripts/preprocess_visdrone.py`. Each
  newly-positive cell needs its OWN box target: the DIoU block predicts
  `w = anchor·exp(tw)` and centres relative to that cell, so a ring cell must encode the
  same GT box as offsets from *its* cell origin and *its* anchor — not a copy of the
  centre cell's numbers. Getting this wrong is silent (training runs, boxes are just
  wrong), so `--smoke` it the way the original encoder was smoke-verified: decode the
  encoded targets back to boxes and assert they reproduce the GT.
- **Bite 2 — data.** Re-run `process_split_fpn --fpn` to a NEW dir (`data/visdrone_fpn_cs`)
  — 8.3 GB, do not clobber the existing set, which is the control's input.
- **Bite 3 — train + A/B.** New spec name (checkpoint prefix!) so the four existing arms
  survive. **Control = the T2-bias arm** (`…_448_wcls_pb__visdrone_`, recall 12.62%,
  mAP 0.0001) — same weights, same bias, same everything but the assignment.
- **Bite 4 — the diagnostic that matters.** Re-run `fpn_neighbor_separation.py`. Under
  center sampling the "ring" is now mostly positive, so the meaningful new question is
  whether the boundary between the enlarged positive region and the background outside it
  is separable. If `pos − ring` is still ≈0 at the NEW boundary, the problem has just moved
  outward one cell and the hypothesis is refuted.

## Risks, in the order they are likely to bite

1. **Loss re-balance.** 6.6× positives shifts the focal loss further toward positives
   (already 75% of the objectness term). This may need `λ_noobj` or focal α revisited — but
   **measure before adjusting**, exactly as T1a demanded. Do not pre-emptively "fix" it.
2. **NMS at tiny box sizes** (bite 0b). If ring boxes don't merge, consider a lower NMS IoU,
   or soft-NMS, or scoring by cluster rather than by cell. This is the most likely reason
   for the hypothesis to fail in a way that still leaves it *fixable*.
3. **Recall vs precision trade.** More positives should raise recall; mAP only moves if
   ranking improves too. Recall has been stuck at ~12% across four arms, so treat a recall
   jump as necessary-but-not-sufficient and always read mAP alongside it.
4. **The 88.2% encodability ceiling still stands.** Center sampling does not fix cell
   collisions on the 56² grid — that is a P2-scale question (Tier 3, deferred).

## Inherited gotchas — all of these have already cost time

- **`FPN_TOWER` selects the SPEC, so set it on `infer` too, not just training.** Forgetting
  it silently evaluates a *different arm's* checkpoint: every prefix/size/vmfb is
  self-consistent for the wrong spec, so no size check catches it. The tell is an epoch
  sweep with **zero variation between checkpoints**. `inferDump` now prints
  spec/prefix/expected-floats on line 2 — read it. (`runs/fpn_t2a_eval_watch_BROKEN_wrong_arm.log`
  is the failure kept for reference.)
- **Build/run the train step with `IREE_BACKEND=rocm`.** Without it `ireeCompileArgs`
  defaults to CUDA/sm_86 and drops
  `--iree-codegen-llvmgpu-use-reduction-vector-distribution=false`, and the loss reduce dies
  with `'func.func' op failed to distribute`. The flag exists for exactly that failure.
- **Any unbounded op (`exp`) + global-norm grad clip = latent `inf·0 = NaN`.** Cap at
  source; DIoU's `tw,th ≤ 8` cap is why the FPN arm trains at all.
- **An interim measurement confirming the MECHANISM is not evidence the LEVER works.**
  T2-bias moved its target metric 3–5× at e4 and gave every bit of it back by e12. Judge at
  e12, on mAP, against a named control.
- `lake build <exe>` mid-run does NOT kill a running trainer (new inode) — verified.
- Checkpoint prefix = sanitized `NetSpec.name`; give every arm a distinct name.

## The train → eval loop (current, for reference)

```
lake build yolov1-visdrone-fpn
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 ./.lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
# eval a checkpoint on the OTHER gpu while training runs (see run_fpn_pb_eval_watch.sh):
B=.lake/build/resnet_34___fpn_detector_448_wcls_pb__visdrone_
cp ${B}_params_e12.bin ${B}_params.bin; cp ${B}_bn_stats_e12.bin ${B}_bn_stats.bin
IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=1 ./.lake/build/bin/yolov1-visdrone-fpn infer \
  data/visdrone_fpn figures/yolo_fpn_pb_e12
<jax-venv>/bin/python3 scripts/yolo_map_visdrone.py figures/yolo_fpn_pb_e12/logits.bin \
  data/visdrone448/val.bin --grid 14 --fpn data/visdrone
<jax-venv>/bin/python3 scripts/fpn_neighbor_separation.py \
  figures/yolo_fpn_pb_e12/logits.bin data/visdrone_fpn/val.bin
```
`<jax-venv>` = `/home/skoonce/lean/claude_max/lean4-jax/.venv`. ~22 min/epoch at tower=0
(34 at tower=4) on gfx1100, + ~7 min one-time compile; checkpoints every 2 epochs. GT for
mAP comes from the single-box `data/visdrone448/val.bin`.

## Diagnostics available

| script | answers |
|---|---|
| `fpn_neighbor_separation.py` | is the cap the assignment? (positive vs ring vs far) |
| `fpn_loss_breakdown.py` | where does the loss actually go? (box/obj/cls, pos vs neg) |
| `fpn_obj_separation.py` | objectness AUC + logit spread + class-head collapse |
| `fpn_class_freq.py` | encoded-target class frequencies + weight literals |
| `visdrone_fpn_coverage.py` | encodability ceiling per assignment scheme (extend for 0a) |
| `fpn_prior_bias_check.py` | did a prior-bias init land on the right channels? |
