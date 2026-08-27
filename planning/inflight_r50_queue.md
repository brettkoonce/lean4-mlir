# inflight_r50_queue.md — what is running right now, and how to pick it up

**Opened 2026-08-16 12:20; leg 3 added 12:41. ✅ CHAIN DRAINED 2026-08-18 19:42.** This was the
live handoff for a four-leg chain; it is now the record of one. Companion to
`imagenet_rerun_sweep.md`, which is the inventory; this was the operational state.

▶ **Nothing here is running.** Read it for the traps (§4's Ghost-BN correction, §6's kill order,
§1's completion markers), not for status. Results live in `imagenet_rerun_sweep.md`'s status
banner and blueprint §5.7/§5.8.

---

## 1. THE CHAIN

| leg | what | status | ETA |
|---|---|---|---|
| 0 | R50 `2018`, JAX, 90 ep | ✅ **76.95 / 93.44**, 24.0 h | landed 2026-08-16 10:40 |
| 1 | R50 RSB-A3, **JAX reference**, `rsb-faithful`, 100 ep | ✅ **78.26 / 93.79**, 15.1 h | landed 2026-08-17 01:46 |
| 2 | R50 RSB-A3, **VERIFIED** `lambaccdp8x64wxclipbce`, 100 ep | ✅ **77.91 / 93.84**, 32.1 h | landed 2026-08-18 09:50 |
| 3 | ~~R34 RSB-A3, timm `resnet34.a3_in1k`~~ | ⚠ ran (73.70 / 91.61) then **DESCOPED** | see below |

⚠ **Leg 3 was a misread and is descoped.** The ask was R34 on the **2018 recipe under the current
(timm) validation setup**; leg 3 instead ran timm's *A3 recipe* on R34. Its config and driver were
removed from the tree on 2026-08-22. The run itself completed and its artifacts survive at
`/home/skoonce/r34_a3_jax_100ep/` (73.70 / 91.61, which does beat timm's published 72.996 for that
recipe) — but nothing quotes it, and it is not in the blueprint.

▶ **The run that was actually wanted** was done separately: **R34 `2018`, 90 ep → 74.16 / 91.92**,
14 h 49 m, artifacts at `/home/skoonce/r34_2018_90ep/`. Blueprint §5.7 carries it.

Driver: `scripts/queue_r50_a3_pair.sh` for legs 0–2, launched `setsid nohup`, parented to init.
**Terminal hangup cannot reach it** — it and both supervisors are their own session leaders.
Verified with `ps -o sid,stat`, not assumed.

⚠ **Leg 3 has a SEPARATE driver**, `scripts/queue_r34_a3_after_leg2.sh` (sid 3241150), added
after the chain was already in flight — editing a running bash script is undefined, so chaining
had to be a second watcher rather than a fourth stanza in the first one. It waits on leg 2's
marker and is a no-op if leg 2 never lands. See §8.

### Where to look

```
/tmp/queue_r50_a3_pair.out                            queue narration (leg transitions)
/tmp/r50_a3_jax_100ep_master.log                      leg 1 supervisor
/home/skoonce/r50_a3_jax_100ep/r50_a3_jax_full.log    leg 1 training stdout (cumulative)
/tmp/supervise_r50-a3-wxclip-4gpu/master.log          leg 2 supervisor (exists once leg 2 starts)
/tmp/supervise_r50-a3-wxclip-4gpu/full.log            leg 2 training stdout
/home/skoonce/r50_2018_90ep/                          leg 0's finished checkpoints + log
/tmp/queue_r34_a3.out                                 leg 3 queue narration
/tmp/r34_a3_jax_100ep_master.log                      leg 3 supervisor (exists once leg 3 starts)
/home/skoonce/r34_a3_jax_100ep/r34_a3_jax_full.log    leg 3 training stdout (cumulative)
```

⚠ **The two supervisors print DIFFERENT completion markers**, and grepping the wrong one is a
silent hang rather than an error:

* `jax/scripts/supervise_r50_a3_100ep.sh` (legs 0, 1, **3**) → `TRAINING COMPLETE`
* `scripts/supervise.sh` (leg 2, the generic engine) → `COMPLETE — N/M epochs`

⭐ Leg 3's driver greps for leg 2's marker, so it spans both conventions. The em-dash was checked
byte-for-byte (`e2 80 94`) against what `supervise.sh` emits rather than eyeballed — a hyphen
there would have been a silent 34-hour wait ending in nothing.

---

## 2. LEG 0's RESULT, AND WHY IT NEEDS NO ASTERISK

**76.95 % / 93.44 %**, 90 epochs, 24 h 0 m, 4 attempts / 3 thermal cooldowns / 0 crashes.
Final weights `/home/skoonce/r50_2018_90ep/r50_2018_imagenet.bin` (+ `.state.npz`).

⭐ It clears torchvision's ~76.1 % 90-epoch reference, which is what the run existed to establish,
and it is **post-C2/C3/C4/C5/C6 end to end and scored over 50,000** — so unlike every number in
`imagenet_rerun_sweep.md` §6 it is directly quotable and directly comparable to legs 1 and 2.
**§5 item 1 of the sweep is closed.**

▶ Every resume landed on an exact step boundary — 125,100 / 250,200 / 375,300, each exactly
`epoch × 5004` — so the 90-epoch cosine was continuous across all three cooldowns. The only cost
of the duty cycle was ~22 s of recompilation on the first epoch after each pause.

---

## 3. LEG 1 — the JAX reference, and the gate it already passed

`rsb-faithful`: LAMB, BCE, lr 0.008, effective batch 2048 (4 accum × 512, itself 4 devices × 128),
100 epochs, train@160 / eval@224, `wdExcludeNormBias` + `gradClipNorm 1.0`.

⭐ **The BCE trivial-solution gate passed.** `scripts/jobs/r50-a3-4gpu.conf` warns that BCE
admits a degenerate solution — drive every logit to ≈ −8, mean-BCE settles at ≈ 0.008, top-1 pins
at chance — and that top-1 must be decisively off 0.1 % by ~epoch 5 or the remaining hours are
wasted. Measured: **ep 5 = 11.62 %, ep 10 = 35.15 %, ep 12 = 36.58 %.**

⚠ **The loss value is NOT the signal and looks alarming if you read it as one.** `loss(train_avg)`
sits at 0.005–0.006, i.e. right where the degenerate solution sits, because mean-BCE is small by
construction when 999 of 1000 targets are negative. **Top-1 is the discriminator.**

Measured throughput **470 s/epoch** (625 optimizer steps), so ~15 h including three cooldowns —
about 2 h faster than the sweep's ~17 h estimate.

---

## 4. LEG 2 — what it is, and the two traps around it

`scripts/jobs/r50-a3-wxclip-4gpu.conf` (new, deliberately a SIBLING of `r50-a3-4gpu.conf`
rather than an edit — that file documents the completed 77.43 % `lambaccdp8x64bce` run in its own
comments, and repointing its variant would leave those measurements describing a run nobody could
reproduce).

Variant `lambaccdp8x64wxclipbce` = the 77.43 % artifact **+ `wx`** (wdExcludeNormBias) **+ `clip`**
(D1's gradient clip). Both were open `a3_paper_fidelity.md` deltas when 77.43 % went out.

⭐ **The pairing with leg 1 is one-variable, checked rather than assumed** (2026-08-16):
`resnet50ImagenetConfigRSBFaithful` ALREADY sets `wdExcludeNormBias := true` and inherits
`gradClipNorm := 1.0` (`MainResnet50Imagenet.lean:104`), plus BCE, LAMB, 100 epochs and effective
batch 2048. The reference needed no edit to be a peer.

⚠ **The one shape difference, and it is not fixable without changing a recipe.** Leg 2 accumulates
8 × 256 (4 replicas × bs 64); leg 1 accumulates 4 × 512 (512 sharded 4 × 128). Both reach effective
2048, but BN normalises over **64 vs 512** — different Ghost-BN granularity. Do not
attribute a small verified-vs-reference gap to the verified path without accounting for it.

⚠⚠ **CORRECTED 2026-08-22 — this said "64/GPU vs 128/GPU" and that was wrong**, in the direction
that understates the confound 4×. Leg 1's trainer is `@jit` under a `NamedSharding` mesh, NOT
`pmap`, so the `jnp.mean(x, axis=(0,2,3))` in `_bn` carries **global** semantics and XLA inserts
the cross-device reduction itself — there is no `psum` in the emitted file precisely because none
is needed. The statistic group is the whole 512-image micro-batch, not the 128 resident on each
card. Leg 2 is a genuine per-replica 64 (no collective touches the batch statistics), so the real
difference is **8×**, not 2×. Verified by reading the emitted trainer, not assumed.

⚠ The queue runs `scripts/gen_shims.sh` before leg 2. The job's PRECHECK checks the shim EXISTS,
which a stale one also does; shims went stale twice in the week of 2026-08-14, once silently.

---

## 5. ▶ WHAT TO DO WHEN EACH LEG LANDS

1. **Leg 1 lands** — nothing to do; the queue advances itself. Sanity-check top-1 against the July
   run's **77.22 %**: this is the same recipe and the same 62,500 total steps, but under the new
   eval protocol (C3) and the new network (C6), so it will NOT reproduce 77.22 % and should not be
   expected to. Whatever it prints IS the reference from now on.
2. **Leg 2 lands** — the verified-vs-reference comparison the whole sweep is for. Both numbers are
   over 50,000 under the same protocol, so quote them side by side.
3. **Update `imagenet_rerun_sweep.md`** §5 items 2 and 3, and §6's table.
4. ⚠ **Do not re-score either with `eval_full50k.py` expecting a different number** — both runs are
   already post-C3/C4/C6, so their in-training figure is already the correct-protocol one. That
   script is for OLD checkpoints, and §5.4's correction explains why even then it cannot isolate
   C3/C4.
5. **Leg 3 lands** — closes `imagenet_rerun_sweep.md` §6b, the one net a re-score could not
   rescue. Compare against timm's published `resnet34.a3_in1k`; ⚠ that number was NOT recorded
   here because timm ships no accuracy table offline, so read it off timm's results CSV rather
   than trusting a remembered figure. Then update §2.2's R34 row and §6b.

---

## 6. IF SOMETHING BREAKS — re-queuing

Both supervisors resume losslessly from the newest full-state checkpoint, so a crash costs at most
the epochs since the last one (`CKPT_EVERY=5` on leg 1; leg 2 checkpoints every epoch).

**Re-run just leg 1** (JAX reference):
```bash
cd jax && PY=.lake/build/generated_resnet50_imagenet_rsbfaithful.py \
  TAG=r50_a3_jax_100ep CKPT_BASE=/home/skoonce/r50_a3_jax_100ep/r50_a3_jax \
  VENV_PY=/home/skoonce/lean/klawd_max_power/lean4-jax-mlir/.venv/bin/python \
  CKPT_EVERY=5 COOLDOWN_AT="25 50 75" COOLDOWN_SECS=1800 CUDA_DEVS=0,2,3,4 \
  setsid nohup bash scripts/supervise_r50_a3_100ep.sh \
    > /home/skoonce/r50_a3_jax_100ep/supervisor.out 2>&1 &
```

**Re-run just leg 2** (verified):
```bash
DRY_RUN=1 scripts/supervise.sh r50-a3-wxclip-4gpu     # PRECHECK only, launches nothing
setsid nohup bash scripts/supervise.sh r50-a3-wxclip-4gpu \
  > /tmp/supervise_r50-a3-wxclip-4gpu.out 2>&1 &
```

**Re-run the whole chain** — `scripts/queue_r50_a3_pair.sh` waits on leg 0's marker, which is
already present in `/tmp/r50_2018_90ep_master.log`, so it falls straight through to leg 1. ⚠ If
`/tmp` has been cleared, that marker is gone and the queue will wait forever; launch the legs
directly with the commands above instead.

⚠ **Kill order matters for leg 2.** `scripts/supervise.sh` traps EXIT and takes its run down with
it, but `setsid` means the reverse is not true — killing the trainer alone leaves the supervisor to
relaunch it. Stop the supervisor, not the trainer.

### Box constraints that decide all of this

* **Only 4 usable GPUs.** Devices **1 and 5 throw PCIe BadTLP under load** and are masked
  everywhere (`DEVS="0,2,3,4"`). So the legs are strictly sequential; there is no concurrency to
  recover.
* **Thermal duty cycle is not optional** — the box has no fans and goes flaky ~24 h into continuous
  load. Leg 1 rests 30 min after epochs 25/50/75; leg 2 rests on temperature (`TEMP_MAX=78`,
  `TEMP_RESUME=62`), which is the better mechanism because it adapts to ambient.
* ⚠ **Disk was 98 % / 51 GB free at 2026-08-16 12:20**, down from 58 GB at the chain's start. The
  chain needs ~5 GB more. It fits, but it is creeping — check before writing anything large.
* ⚠ **Never `pip install timm` into `.venv`.** It pulls the CUDA 13 stack, `nvidia-nccl-cu13`
  overwrites the pinned cu12 `libnccl.so.2`, and every multi-GPU run dies with a misleading
  "CUDA driver version is insufficient". timm belongs in `.venv-timm`.

---

## 7. ⚠ UNCOMMITTED AS OF 2026-08-16 12:20

Level with `origin/main`; none of this session's work is committed, and a fresh session will not
know why any of it exists.

| file | state | what |
|---|---|---|
| `planning/imagenet_rerun_sweep.md` | modified | **C6** added to §1, §5.4's re-score advice CORRECTED, new §6 with the re-score results + equality gate |
| `jax/scripts/eval_full50k.py` | new | net-agnostic re-score; reads the state-tuple layout from the generating module and refuses on a leaf-count mismatch |
| `scripts/jobs/r50-a3-wxclip-4gpu.conf` | new | leg 2's job config |
| `scripts/queue_r50_a3_pair.sh` | new | the chain driver |
| `jax/scripts/supervise_r50_a3_100ep.sh` | modified | `PY`/`TAG`/`CUDA_DEVS` made env-driven so one supervisor drives legs 0 and 1 |
| `planning/inflight_r50_queue.md` | new | this file |
| `jax/MainResnetImagenet.lean` | modified | **leg 3**: `resnet34ImagenetConfigA3` + the `a3` recipe — timm's `resnet34.a3_in1k` |
| `scripts/queue_r34_a3_after_leg2.sh` | new | leg 3's driver + its PRECHECK |

▶ The §5.4 correction is the one that matters most: without it, a puller still reads
"re-scoring is far cheaper than re-training" and checks for surviving checkpoints, which
2026-08-15 established is unsound for every checkpoint predating 2026-08-04.

⚠ `jax/scripts/supervise_r50_a3_100ep.sh` is part of the family `scripts/supervise.sh` says it
replaces. Parameterizing it was the low-risk choice mid-flight, not the right long-term home —
the JAX legs should eventually become job configs, which needs the engine's jax-path `epoch_now()`
branch exercised (no existing config uses it).

---

## 8. LEG 3 — R34 RSB-A3, added 2026-08-16 12:41 (mid-flight)

The ResNet-34 re-train `imagenet_rerun_sweep.md` §6b calls for, run to **timm's own recipe**.

**Why a re-train and not a re-score.** R34 is the one net in the sweep a re-score cannot rescue:
its surviving JAX `.bin` (May 31) has **no companion `.state.npz`**, so there are no BN running
stats to evaluate with, and it predates C6 besides. ⚠ §2.2 credits R34's number to
`runs/r34in_30ep_4gpu_sup*.log`, but that is the **verified** path (70.71 % over 49,920), not a
JAX run.

**What the recipe is.** `resnet34ImagenetConfigA3` in `jax/MainResnetImagenet.lean`, recipe arg
`a3` → `.lake/build/generated_resnet34_imagenet_a3.py`. It is timm's `resnet34.a3_in1k`: the RSB-A3
*procedure* on the 34-layer basic-block backbone. timm ships that name from the RSB release itself
(`v0.1-rsb-weights/resnet34_a3_0-a20cabb6.pth`), so it is a reproduction target, not an invention.

⭐ **The geometry was read off timm 1.0.28, not assumed** — `input_size=(3,160,160)`,
`crop_pct=0.95`, `test_input_size=(3,224,224)`, byte-identical to `resnet50.a3_in1k`. Same method
as `MainResnet50Imagenet.lean:82`'s `testCropRatio` note.

⭐⭐ **The batch shape is deliberately leg 1's, and it is load-bearing.** 512 micro × 4 accum,
sharded 4 × 128/GPU — verified identical in the emitted file
(`MICRO_BATCH = (512 // n_devices) * n_devices`). ⚠ Under `@jit` + `NamedSharding` the BN
statistic group is the full 512, not the per-card 128 (see §4's correction). R34 at 160px is
light enough that a larger micro-batch would fit, but raising it would hand R34 a different
Ghost-BN granularity than the R50 reference beside it. That is precisely the confound §4 flags
between legs 1 and 2; leg 3 avoids inheriting it. ⚠ Do not "optimise" this for throughput.

⚠ **R34 leaves C1's safe list here.** `imagenet_rerun_sweep.md` §1a lists R34 as augmentation
"none", `mstd 0.0`. This is R34's first RandAugment recipe (`mstd := 0.5`) — the exact setting that
made Solarize's magnitude symbolic and emitted a zero-byte shim. Fixed in `8f8fdd1`, but the §1a
row stops describing R34 the moment leg 3 runs.

### Verified before launch, not assumed

| check | result |
|---|---|
| `lake build resnet34-imagenet` | ✅ built, `nice -n 19` — leg 1 held 470.0 → 471.4 s/epoch across it |
| emitted param count | ✅ 21,797,672 — exact torchvision ResNet-34 |
| leg 1's trainer untouched by the generate | ✅ md5 unchanged |
| lr / epochs / accum / trainRes / crop | ✅ 0.008 / 100 / 4 / 160 / 0.95, greped out of the emitted file |
| PRECHECK refuses a stale build product | ✅ tested both directions |
| PRECHECK refuses while GPUs are held | ✅ fired against leg 1's live processes |
| leg 2's em-dash marker | ✅ `e2 80 94`, byte-matched |

⚠ **The JAX path needs NO shim.** Confirmed by inspection: the only "shim" string in the generated
trainer is a comment about C1; there is no `Popen`. So leg 3 required no `gen_shims.sh` run and
could be built mid-flight without touching anything leg 1 or leg 2 reloads on resume.

### ETA and how to check it

**~12 h**, and this is an extrapolation, not a measurement: R34 and R50 have been run head to head
on this box at matched settings (90 ep, bs256, @224) at ~18 h and 22.5 h — a 0.8× ratio — and
0.8 × leg 1's measured 470 s/epoch ≈ 375 s/epoch ⇒ ~10.4 h + 3 × 30 min cooldowns.
▶ **The first `[Epoch 1]` line settles it. Check that rather than trusting this paragraph.**

### If it needs re-queuing

```bash
DRY_RUN=1 scripts/queue_r34_a3_after_leg2.sh                  # PRECHECK only, launches nothing
SKIP_WAIT=1 setsid nohup scripts/queue_r34_a3_after_leg2.sh \
  > /tmp/queue_r34_a3.out 2>&1 &                              # box already idle: skip the wait
```

⚠ Regenerate first if `jax/MainResnetImagenet.lean` has been touched since — the PRECHECK will
refuse, but the fix is `(cd jax && ./.lake/build/bin/resnet34-imagenet a3)`. ⚠⚠ Do **not** run
that exe expecting it to only generate: it generates *and then launches training*, and bare it
picks up `/home/skoonce/anaconda3`'s jax-less python (observed 2026-08-16). Only the pinned
`.venv` interpreter can run it.
