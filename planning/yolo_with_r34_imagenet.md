# yolo_with_r34_imagenet.md — YOLOv1 push using the 1000-class R34 backbone

Handoff doc for the next session. Captures where YOLOv1 is, what the
R34 ImageNet run unlocked, and the concrete next steps. Read this
*after* skimming `yolo_demo_v2.md` (smallest-viable scope, pinned
decisions) and `yolo_demo_v3.md` (Phases 2-6) for architectural
context.

## What just landed (2026-05-29)

The R34 ImageNet run finished. The 1000-class backbone is on disk:

```
jax/.lake/build/jax_r34_imagenet.bin     # 87 MB, 37 param groups
jax/.lake/build/jax_r34_imagenet_e30.bin # same payload, named copy
```

Final result: **69.26% top-1**, val\_loss 1.232 on the standard 50K
validation split. ~15 wall-clock hours for 30 epochs across 2× 7900 XTX.
Recipe: SGD+momentum 0.9, batch 256 (128/device), cosine LR + 5-epoch
warmup from peak 0.1, label smoothing 0.1, random-crop + hflip.

Crucially: written via `emitParamsToFile`, byte-for-byte compatible
with `paramShapes` and therefore with `TrainConfig.bootstrapBackbone`.
No conversion step. See Ch 6 of the blueprint for the prose.

## What's already in place (don't redo)

YOLOv1 Phases 1-5 of `yolo_demo_v3.md` already shipped:

| Commit | What |
|---|---|
| `08d0c2d` | Phase 1 — codegen + FFI + 6 unit tests on R34+YOLO head |
| `f46182e` | Phase 2 + R1 — VOC 2007 trains end-to-end through unified path |
| `2762faf` | Phase 3 — bbox-aware hflip + random crop |
| `a3c908d` | Phase 4+5 — backbone bootstrap loader + inference dump + render |

Specifically `a3c908d` added:
- `LeanMlir/SpecHelpers.lean::patchInitWithPretrainedPrefix`
- `TrainConfig.bootstrapBackbone : Option (String × Nat)`
- `demos/MainYolov1VocTrainBootstrap.lean` (`yolov1-voc-train-bootstrap`)
- `demos/MainYolov1VocInfer.lean` (`yolov1-voc-infer`)
- `tools/yolo_render.py` (render boxes onto images)

The bootstrap exe as committed pointed at an **Imagenette** R34
checkpoint (`.lake/build/resnet_34_params.bin`, prefix 21,284,672).
That checkpoint trained on 10 classes and won't transfer well to 20-
class VOC. **The R34 ImageNet checkpoint is the upgrade this whole
session was about.**

## The push (next session)

### Step 0 — Sanity-check the bootstrap path

The current `MainYolov1VocTrainBootstrap.lean` hardcodes a path and
prefix. Before training, confirm the byte layout still matches:

```
ls -l jax/.lake/build/jax_r34_imagenet.bin
# expect 87,190,688 bytes = 21,797,672 floats × 4 bytes
# R34-Imagenette had a 512×10 dense head (5,120 params)
# R34-ImageNet has a 512×1000 dense head (512,000 params)
# So the YOLO head replaces 512,000 floats, not 5,120
# The bootstrap prefix is `totalParams - 512*1000 - 1000 = 21,284,672`
# Same number as before because Imagenette had `512*10 + 10 = 5,130`.
# But verify it's structured identically up through globalAvgPool.
```

Run `lake exe resnet34-imagenet --print-shapes` (or eyeball
`SpecHelpers.paramShapes resnet34Imagenet`) and confirm the first 37
groups match `r34Yolov1` up through `.globalAvgPool`. Probably they
do — the spec is shared up to the head — but **don't skip this check**;
a silent byte-misalignment will corrupt training in a way that's hard
to debug at hour 6 of an overnight run.

### Step 1 — Point bootstrap at the ImageNet checkpoint

Update `demos/MainYolov1VocTrainBootstrap.lean` to:

```lean
bootstrapBackbone := some
  ("../jax/.lake/build/jax_r34_imagenet.bin", 21284672)
```

(Path is relative to the phase-3 working directory. Adjust if it's
launched from project root.)

Also: the `_bn_stats.bin` companion file the loader auto-detects —
the JAX side writes per-batch BN running stats too. Check whether
`emitParamsToFile` wrote them; if not, the YOLO bootstrap will
initialize BN stats fresh, which is fine (fine-tuning will retrain
them). Verify in `Train.lean::runTraining` log output.

### Step 2 — Run the bootstrapped VOC training

```
tmux new-window -n yolo-voc
cd /home/skoonce/lean/claude_max/lean4-jax
nohup lake exe yolov1-voc-train-bootstrap > runs/yolo-voc.log 2>&1
```

Use tmux not bash background — same lesson as R34 (systemd-logind
reaps bash-tool descendants after hours). Expected ~6-10 hours
overnight on one GPU; YOLO is much smaller than R34 ImageNet so
single-GPU is fine.

Target: ~45-50% mAP@0.5 (vs paper's 63% at 448 input + full
Darknet backbone + 2012 data). The R34 backbone is *better* than
Darknet but the input is 224 not 448, so the two roughly cancel.

### Step 3 — Inference + render

```
lake exe yolov1-voc-infer > runs/yolo-voc-preds.jsonl
python3 tools/yolo_render.py runs/yolo-voc-preds.jsonl \
        --out blueprint/src/figures/yolo_voc/grid.png
```

Render goes to the blueprint figures dir so it can be cited in
Ch~\ref{chap:bestiary}. Task #32 (pending) is exactly this render.

## Things to NOT do in the next session

- **Don't retrain R34.** It's done. 69.26% is plenty for fine-tuning.
- **Don't try multi-GPU on YOLO.** It's pointless — single GPU is
  more than fast enough for 5K VOC images.
- **Don't switch to 448×448 input.** Stay on 224. The whole point
  of v2 was "smallest viable"; the input bump is a v3-polish item.
- **Don't add VOC 2012 data.** Same reason. Stay at 5K trainval.
- **Don't second-guess the bootstrap prefix arithmetic** without
  cross-checking against `paramShapes`. The numerology is load-bearing
  and a silent shift here will eat hours.

## If something goes sideways

The most likely failure mode is byte-layout mismatch between the JAX
R34 dump and the phase-3 R34 spec. If `patchInitWithPretrainedPrefix`
loads cleanly but training diverges immediately (loss explodes in
the first ~100 steps), suspect:

1. BN parameter ordering — JAX emits `(scale, bias, mean, var)`,
   `paramShapes` might expect a different tuple order
2. Conv weight layout — JAX uses `HWIO`, phase-3 might expect `OIHW`
3. The 1000-class head bytes aren't being skipped correctly

Cross-check by loading the same R34 ImageNet bin into the phase-2
JAX trainer at batch 1, single step, and confirming the loss matches
what `jax_r34_imagenet_e30.bin`'s training-time loss was at epoch 30
(~1.32). If it doesn't, the dump→load roundtrip is the bug; fix it
on the JAX side before touching YOLO.

## Open follow-ups (not blocking)

- bf16 pass on JAX codegen (Ch 6 compute budget TODO) — gets the R34
  recipe to ~2× throughput on 7900 XTX WMMA / Lovelace tensor cores.
- 90-epoch ImageNet R34 run for the paper-headline number — only
  worth doing if YOLO mAP is bottlenecked by backbone quality.
- TFRC TPU pitch (full-ImageNet ablation matrix for every demo
  architecture) — discussed in `r34_imagenet.md` and the session.

## Sources

- `planning/yolo_demo_v2.md` — Phase 1 design, smallest-viable scope
- `planning/yolo_demo_v3.md` — Phases 2-6, scoped on top of v2
- `planning/r34_imagenet.md` — operational notes from the R34 run
- Blueprint Ch 6 "Compute budget" section — measured R34 numbers
