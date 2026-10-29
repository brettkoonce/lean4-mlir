# recipe_prep.md — every ImageNet chapter's `[TODO: …]` marker ready to run

Started 2026-10-11. The ch 5–9 markers (`blueprint/src/content.tex`, one bold `[TODO: what is still out;
the job that closes it.]` after each side-by-side table) are the contract: each names the recipe items a
pair still owes its paper and the confs that pay them. This doc is the code plan for closing every one
of them *in code*, in cost order, so that whichever box frees next has its run ready. The runs
themselves go in the user's ROI order (ResNet → ViT → MNv2 → EfficientNet → ConvNeXt) and each needs
the user's word. `imagenet_parity.md` §10 holds the F-items (cross-net) and D-items (per-net) this doc
cites; the method is the ResNet A3/A2 prep of 2026-10-11 (4d612bb9), repeated per net.

**All seven sections are in code (2026-10-12; §9).** Nothing has run. §7 closed in two halves — D3
(a89fa9af) and the rest as TF's own code has it (its second "Done" paragraph): `main.py`'s filter
excludes BN alone, so the biases decay and the depthwise does too (no `dw`); D13 rides on §6's
`OptRecipe.rmspropTf`. §6's two rows that had stayed disclosed — slim's validation crop and TF 1's
bilinear kernel — landed later the same day (its addendum; D23), so MobileNetV2 is the README's
every row like MNv4 and B0. What is left is the runs, in the user's ROI order, each on the user's word —
and the four rows the markers still call "disclosed", costed in §10 (R1–R4), each on the user's word too.

## 1. Done, 2026-10-11

| chapter | state | commit |
|---|---|---|
| ch 5 ResNet R34, R50-2018 | fp32 rescore paid the markers; R34's deleted; McNemar paired | 8a659b2f, 5be4f349 |
| ch 5 ResNet A3, A2, A1 | timm's recipe on both paths (F1 / F3 / F4 / F5 / F8), gated, smoked; the A3 pair ran 2026-10-12/13 on ares — JAX 77.60 / 93.61, verified 78.08 / 93.84 (`runs/2026-10-12-r50-a3-jax/`, `runs/2026-10-12-r50-a3-rsb-bf16/`), the ch 5 A3 marker deleted, McNemar 4,038 discordant / +236 / p = 0.0002 (the JAX checkpoint rescored on atlas's CPU, `rescore.sh`); A2 / A1 pending | 4d612bb9 |
| ch 9 ViT-Ti | DeiT as timm runs it on both paths, dp256x4 no-clip render, smoked; runs pending | 320f5045, 3336f4a7 |
| ch 9 ViT-S, ViT-B | Tiny's recipe at both widths; S one-shot 4 × 256, B the first `acc` render; both JAX refs accumulate; smoked (§3) | a39f5cf7 |

What "done" means, per net, and what each later section repeats:

1. **Flags on the JAX config.** `TrainConfig` (`LeanMlir/Types.lean`) carries `mixSwitchProb`,
   `schedPerEpoch` + `warmupLRInit`, `minLR` (the per-step cosine takes it too), `randAugmentFillMean`,
   `repeatedAugAdjacent`, `emaWarmup`, `evalF32`, `gradClipNorm`, `gradAccumSteps`, `wdExcludeNormBias`,
   `cropTorchvision`, `randAugmentInc`, `geluExact`, `bnMomentum`, `bceTargetThresh`. A net whose marker
   is covered by these needs config lines, a re-emit (`scripts/regen_jax_generated.sh write`, then
   `sync`), and nothing in the emitter.
2. **The verified half.** The shim is emitted from the same config, so augmentation items ride for
   free. The driver (`LeanMlir/Verified/Train.lean`, `trainAdamSched`) takes `schedPerEpoch` /
   `warmupLRInit` / `minLR` / `emaWarmup` as arguments the net's app passes. Anything in the *graph* —
   accumulation, the clip, resolution, an optimizer rule, padding — is a render (`LeanMlir/Proofs/Codegen/
   *RenderB.lean`, a `#eval` + `#guard`, the GELU twin where the net has one, `MANIFEST.md`,
   `.github/workflows/proofs.yml`'s diff list, `scripts/regen_verified_mlir.sh` rows) and, where the
   change moves what a tie proves, proof work.
3. **Confs.** Fresh checkpoint lineages for a rerun, a grep per new emitted line (the ViT and A3 confs are
   the templates), and the verified conf appending `LEAN_MLIR_DUMP_CORRECT=$RUNDIR/bitmaps/<tag>` in
   PRECHECK, so the run writes per-epoch bitmaps and the McNemar pairing is free.
4. **Gates.** `sched_timm_gate.py --trainer <py>` (per-epoch form against timm 1.0.28),
   `repeated_aug_gate.py --script <shim>` (control synthesized from the flagged shim),
   `aug_bicubic_pil_check.py <shim>` (fill), the shim hash under `SHIM_DETERMINISM=1` at two seeds,
   `comment_numbers.py`, `regen_jax_generated.sh check` (every other artifact byte-identical),
   `regen_verified_mlir.sh check` after a render, `DRY_RUN=1 scripts/supervise.sh <conf>` for each conf.
5. **Smokes.** JAX: the trainer with `steps_per_epoch` patched short, two or three validations read for
   the epoch rates and the `eval: f32 forward` line. Verified: `ONCE=1 LEAN_MLIR_MAX_STEPS=400
   LEAN_MLIR_PROBE_WARM=200 RUNDIR=<dir> scripts/supervise.sh <conf>` for the ms/step and the schedule
   banner; delete any checkpoint the smoke leaves under the new tag.
6. **After the pair lands.** Score the reference post hoc through `jax/scripts/eval_full50k.py`
   (`EVAL_DTYPE=f32`, `DUMP_CORRECT`, `PROTOCOL=timm` for D18), pair with the verified bitmap through
   `scripts/demos/mcnemar.py`, update the chapter, delete the marker.

## 2. Order and cost

| step | net | code | new in the emitter / renders | decision needed |
|---|---|---|---|---|
| 1 | ViT-S, ViT-B (ch 9 side quest) | ½ day — DONE 2026-10-11 | S one-shot at 4 × 256; B the first accumulating ViT render (`acc`) | ViT-S micro-batch — probed: one-shot |
| 2 | ConvNeXt-T, then S / B (ch 8) | 1½ days | accumulating no-clip renders | none |
| 3 | MNv4-Conv-M (ch 6) | 1 day — DONE 2026-10-12 | the `mnv4in256` renders (final side 8) | the paper's own yaml is the source of truth; D7 settled by it, D8 withdrawn, D9/D10 corrected (§5) |
| 4 | MNv2 (ch 6) | 1 day — DONE 2026-10-12 | `rmspropTf` recipe + `rmsBufNextTfF`, the `dw` decay set, the first EMA MNv2 render; the slim pipeline + init on the emitter; later that day `evalCropSlim` + TF 1's bilinear (D23) | D12's reference = a numpy twin (the user); the recipe = the README's command, every row, 0.12 @ 256 / ×0.98 per 2.5 ep / 420 ep (the user) |
| 5 | EfficientNet-B0 (ch 7) | 3–4 days — DONE 2026-10-12 (D3 a89fa9af; the rest the same day) | RMSProp op (shared); SAME padding through the ties turned out to be a re-pointing onto MNv2's Xla op family, not new math; then five pipeline / init / decay flags and the `wn` decay set read off TF's own code | D3: proofs (the user) |

About two weeks of code against about two weeks of ResNet and ViT runs on the two boxes. Each step
ends in one commit per net (code + confs + smokes), the chapter untouched until its pair lands.

## 3. ViT-S and ViT-B — "as Tiny's"

Marker (ch 9, after the S/B table): as Tiny's; neither has run — `vits-default-jax-4gpu` /
`vits-default-emabf16-4gpu`, `vitb-accum-jax-4gpu` / `vitb-default-emabf16-4gpu`.

Today (`jax/MainVitSImagenet.lean`, `jax/MainVitBImagenet.lean`): batch 512, lr 5e-4, clip 1.0, minLR
1e-5, repeated aug 3, EMA; none of Tiny's flags. Renders on file: `vit{s,b}in_emadp128x4wxclipdrop
eps0000001{erf,}bf16` (4×128, clip on).

1. Configs: Tiny's lines verbatim — `batchSize 1024`, `learningRate 0.001` (DeiT's 5e-4·1024/512),
   `gradClipNorm 0.0`, `schedPerEpoch true`, `warmupLRInit 0.000001`, `mixSwitchProb 0.5`,
   `randAugmentFillMean true`, `repeatedAugAdjacent true`, `emaWarmup false`, `evalF32 true`; minLR 1e-5
   stays (DeiT's). ViT-B: DeiT-B is 8×128 = 1024; here 4×128 with `gradAccumSteps 2`, exact under
   LayerNorm (`vitb-accum-jax-4gpu` already does the accumulation; keep it).
2. Verified: the apps pass `(emaWarmup := false) (schedPerEpoch := true) (warmupLRInit := 0.000001)` as
   Tiny's does; `batchSize` per replica 256 for S if it fits, else 128 with accumulation 2.
   Renders: `vitsin_emadp256x4wxdropeps0000001{erf,}bf16` (or `…acc2dp128x4…`) and
   `vitbin_emaacc2dp128x4wxdropeps0000001{erf,}bf16`, from `ViTRenderB.lean` as Tiny's were. Checked
   2026-10-11: the ViT renderer has NO accumulation form (R50's `OptRecipe.lambAccum` is LAMB-only;
   `ViTRenderB.lean`:902 notes B fits the arena at global 512 without one). An AdamW accumulation
   recipe in `OptRecipe` + the renderer is the one real piece of code here, shared with §4 — unless
   ViT-S at 4×256 fits and ViT-B is run at 2× accumulation on the JAX side only, disclosed.
3. Decision: ViT-S at 4×256 — a memory probe (`LEAN_MLIR_MEM_FRACTION` 0.90–0.97) before the render
   is chosen. Measure both halves' ms/step; the 10-05 ETAs (`side_quest_etas_measured`) predate every
   flag.
4. Confs: four, on the Tiny confs' grep blocks; new checkpoint names.
5. Still out, disclosed, all three ViTs: the patch-embedding bias at PyTorch's default (small; do it
   here if the ViT renders move anyway, since it is a render change).
6. Runs: ~S and ~B ETAs on file are 10-05 probes; re-probe. The ch 9 side-quest marker goes when both
   pairs land.

**Done 2026-10-11.** The decision in 3 went to a probe before any render was chosen: ViT-S at 4 × 256
peaks at 10.75 GiB of the 11.68 GiB default arena and executes (587 ms a step, bare device at 4
replicas, `runs/2026-10-11-vits-default-emabf16-4gpu/device_probe.log`), so S is one-shot as Tiny is and
the S conf keeps its "no `LEAN_MLIR_MEM_FRACTION`" rule. B is the accumulating one: `vitAdamVariant` and
`vitAdamTrainStepText` (`ViTRender.lean`) take `acc`, trailing and defaulted to 1 (every committed ViT
artifact re-renders byte-identical), and at `acc > 1` hand the per-parameter tail to RenderKit's
`optOne (.adamwAccum k)` with `optConstsB`'s `%aup`-computed β block and `packedTrainSig`'s `G` region —
the "one real piece of code" of 2, ~40 lines, and the same `acc` serves ConvNeXt's writer in §4 (its
`cnxAdamVariant` derives the entry name, so the marker goes through that). The clip is refused in
composition (the reference clips the accumulated mean; `optAllParams`'s `accIn` route would be needed,
and DeiT does not clip). Renders: `vitsin_emadp256x4wxdropeps0000001{erf,}bf16`,
`vitbin_emaaccdp2x128x4wxdropeps0000001{erf,}bf16`, each with its eval forward (S at B = 256 as Tiny's,
B at its 32) and `#guard`s on the spelling; B's probe 13.83 GiB of 15.11 at 0.97 and 813 ms a 512-row
micro-batch (`runs/2026-10-11-vitb-default-emabf16-4gpu/device_probe.log`). `tests/TestVitEmaDropRender`
accepts the five-region layout; `TestVariantPredicates` pins the spelling (k from the FIRST `x`) and
106 / 53 GELU-form pairs. JAX: Tiny's lines verbatim on both configs, and BOTH references accumulate,
`vit{S,B}ImagenetConfigAccum` = 512 × 2 — the JAX ViT-S trainer at the one-shot 4 × 256 dies
RESOURCE_EXHAUSTED at the default arena (12.08 GiB asked) and at 0.97 (14.32), where the verified render
of the same step fits the default; the two compilers lay the backward out differently, and under LayerNorm
the pair is one function either way. So the JAX S conf is `vits-accum-jax-4gpu` (renamed; the marker
and §9.6's conf names follow), `vitb-accum-jax-4gpu` keeps its name at 512 × 2, and both JAX confs raise
`XLA_PYTHON_CLIENT_MEM_FRACTION` where their smoke needed it. Confs: the four, on the Ti confs' blocks,
fresh lineages, bitmaps appended. Gates and smokes per §1.4–1.5 (`imagenet_parity.md` §10.4 has the
numbers). Item 5 (the patch-embed bias) stays out, disclosed.

## 4. ConvNeXt-T, then S and B — the official recipe in four places

Marker (ch 8): batch 4096 (accumulated 16× at 256, exact without BatchNorm), no gradient clip, Mixup or
CutMix drawn per batch, torchvision's crop; plus the exact erf GELU, the cosine floor at 1e-6,
RandAugment's fill at the image mean, and scoring at timm's 288 / 1.0. The EMA pair has not run —
`cnx-default-jax-4gpu` / `cnx-default-emabf16-4gpu`, ~77 + 107 h. S/B: as T's.

**✅ T landed 2026-10-28.** JAX `runs/2026-10-21-cnx-full-jax-bf16-300ep/` 81.990 / 95.87 (74.07 h), verified
`runs/2026-10-21-cnx-verified-bf16-300ep/` 82.146 / 95.910 (102.66 h), both EMA at e300 through an f32
forward; McNemar EMA p = 0.18, live p = 0.64. The ch 8 section is this pair alone and its marker is gone
(the book commit after 1c98b112). Not done from the list above: scoring at timm's 288 / 1.0 (the book
reports the paper's 224 / 0.875). S/B: still pending, their marker stands.

Today (`jax/MainConvNeXtImagenet.lean`): batch 256 (4×64, no accumulation), lr 2.5e-4 (the 4096 rate
scaled down), warmup 20, clip 1.0, drop-path 0.1, EMA, `cropTorchvision`, `randAugmentInc`, erf GELU
(D1, landed 2026-10-06); no minLR, no switch, no fill, no f32 eval. Renders: `convnext{,s,b}in_
emadpwxclipdrop{erf,}bf16` (clip on, no accumulation).

1. Config: `gradAccumSteps 16` with `learningRate 0.004` (the official rate at 4096), `gradClipNorm 0.0`,
   `mixSwitchProb 0.5`, `minLR 0.000001` (the per-step cosine takes the floor, `Codegen.lean`
   `cosLine`), `randAugmentFillMean true`, `evalF32 true`. The schedule stays per step: the official
   ConvNeXt trainer builds its cosine per iteration, not timm's per-epoch form (verify
   `warmup_start_lr` in `main.py` before setting `warmupLRInit`). `emaWarmup`: the official EMA is
   timm's `ModelEmaV2`, constant decay — `emaWarmup false`.
2. Verified: the app passes `(emaWarmup := false)` and `minLR`; renders `convnextin_emaacc16dpwxdrop
   erfbf16` + tanh twin (accumulation 16 at 4×64 = 4096; no clip) — the first accumulating ConvNeXt
   render. The AdamW accumulation recipe of §3.2 now exists on ViT (a39f5cf7) and is a copy here —
   see the hand-off below; after T, S and B follow the same way (`convnextsin_…`, `convnextbin_…`),
   four more artifacts.
3. D15: the T reference on file (81.53) is pre-C6; this rerun replaces both halves, so the pair is
   one-axis again. D18: `PROTOCOL=timm` on the JAX scorer; the verified half's 288 renders exist
   (`convnext{,s,b}in_fwd_s288.mlir`, checked 2026-10-11).
4. Gates: fill and hash on the three shims; `regen_verified_mlir.sh check` (GELU twins). The schedule
   gate does not apply (per step).
5. Confs: six; `LEAN_MLIR_DUMP_CORRECT` appended in the three verified prechecks.
6. Runs: T ~77 + 107 h (10-07 re-probe), S and B larger; the ch 8 marker goes when T lands, the S/B
   marker when both land.

**Done 2026-10-12.** The official values were read off facebookresearch/ConvNeXt `main.py` and `utils.py`
before anything was set: `--lr 4e-3` at 4096, `--min_lr 1e-6`, `--warmup_epochs 20` ramped PER ITERATION from
`cosine_scheduler`'s `start_warmup_value = 0` (so item 1's question is answered: `warmupLRInit` stays 0 and
the schedule stays the emitter's per-step form — this is not timm's per-epoch scheduler), `--clip_grad None`,
`--mixup_switch_prob 0.5` at `--mixup_mode batch`, timm's `ModelEma` at `--model_ema_decay 0.9999` (constant;
`emaWarmup false`), `--model_ema_eval`. Renderer: `cnxAdamVariant` and `convNextAdamTrainStepText` take `acc`
(trailing, default 1; every committed ConvNeXt artifact re-renders byte-identical) and spell it into the
entry — `emaaccdp16xwxdroperfbf16`, `acc` after `ema`, `dp` inside, `16x` before `wx` (the driver's `accK`
ends at the FIRST `x`, so `wx`'s cannot reach it; `TestVariantPredicates` pins `accK = 16`) — routing the
tail through `optOne (.adamwAccum 16)`, the `G` region, `%aup` / `%akeep`, `optConstsB`'s `1/16` and a
`packedTrainSig (acc := true)`; clip ∧ acc refused as on ViT. Six renders, DP only: `convnext{,s,b}in_
emaaccdp16xwxdrop{erf,}bf16` (937 / 1,765 / 1,765 operands = 2 + 5 × P + 7 + masks), no new forwards
(the driver opens `<slug>_erf_fwd` at 64 by name). Probed bare-device at 4 replicas: T 4.46 GiB / 194 ms a
256-row micro-batch, S 7.21 / 300 ms, B 10.42 / 462 ms at 0.90 — the accumulation is the recipe's, not
memory's. JAX: `convNeXt{Tiny,S,B}ImagenetConfig` carry `gradAccumSteps 16`, `learningRate 0.004`,
`gradClipNorm 0`, `minLR`, `mixSwitchProb`, `randAugmentFillMean`, `evalF32`, `emaWarmup false`; S and B's
legacy `accum` / `bs512` recipes are deleted with their eight generated files (batch-256 shapes that would
have inherited the 4096 rate). Apps: `(minLR := 0.000001) (emaWarmup := false)`, the rate defaulting to
4e-3 on an `acc` variant. Confs: the three verified ones rewritten standalone on the ViT-B template
(`cnx-default-emabf16-4gpu` used to source `cnx-default-4gpu`, which asserts the batch-256 rate) with
`LEAN_MLIR_G2_STEPS=4992` (trap 1: 5,004 mod 16 = 12; = the reference's 312 steps × 16, the same
1,277,952 images) asserted together with its divisibility, `LEAN_MLIR_BASE_LR_U=4000`, the render / shim /
app greps and the bitmaps; the three JAX ones on fresh lineages (`~/cnx_{t,s,b}_jax_2611/`) with a grep
per emitted recipe line. Trap 2 bit once: the JAX ConvNeXt-B trainer trains at the default arena but its
f32 VALIDATION dies RESOURCE_EXHAUSTED (7.46 GiB in `jit_eval_batch` — `evalF32`'s cost, where the one-shot
trainer validated in bf16), so `cnxb-default-jax-4gpu` carries `XLA_PYTHON_CLIENT_MEM_FRACTION=0.97` in
`ENV_EXTRA` and asserts it; T and S validate at the default. Smokes (`runs/2026-10-12-cnx{,s,b}-jax-smoke/`,
two 25-step epochs each, the warmup reading 2e-4 then 4e-4 as the per-step ramp has it, the f32 eval line
before every validation): JAX 2.83 / 3.96 / 5.63 s a 4096-row step = 177 / 247 / 352 ms a micro-batch →
~76 / ~106 / ~151 h; verified (`runs/2026-10-12-cnx{,s,b}-default-emabf16-4gpu/`, the driver's `k = 16 …
4992 micro-batches/epoch = 312 updates/epoch` banner, the floor and the constant decay in its schedule
line): 242 / 323 / 491 ms a micro-step → ~109 / ~144 / ~219 h. Gates per §1.4 (`imagenet_parity.md`
§10.4 has the list; the schedule gate is n/a per step, and `mixup_gate.py`'s "inert when off" leg cannot
run on a shim whose mixing is on by default — the `SHIM_HASH` was taken directly: reproducible at seed 0,
distinct at seed 1). Disclosed, not a recipe item: Mixup's λ granularity (JAX once per 4096, the shim once
per 256, the official trainer once per 128). Found on the way: ViT-B's `emaaccdp2x128x4wxdropbf16` was in
`TestVariantPredicates`' table but not its `accumSpellings` since a39f5cf7 (red on the unpushed branch);
fixed. The ch 8 markers stand; the S/B "what is queued" paragraph names the new renders.

**Hand-off from §3 (2026-10-12) — what §4 inherits and the order to do it in.**

* *The accumulation form.* `ViTRender.lean`'s `vitAdamTrainStepText (acc : Nat := 1)` is the template:
  `opt := if acc > 1 then .adamwAccum acc else .adamw`, `optOne opt …` per parameter with the fifth
  return (`nA`) collected as the `G` region, `packedTrainSig` / `packedTrainRetTys` at `(acc := accOn)`,
  the return list `θ ++ m ++ v ++ G ++ E ++ [loss, bc1, bc2] ++ [aup, akeep] ++ [emad, oemad] ++ masks`,
  the consts `wdzConst … ++ optConstsB opt wdStr` instead of the baked `adamWConsts`, `trainStepHandNote
  (acc := …)`, and `clip ∧ acc` refused (the reference clips the accumulated mean; DeiT and the official
  ConvNeXt recipe do not clip). ConvNeXt's writer has the same shape — `ConvNeXtRender.lean` :996
  (`optOne .adamw`), :1043 (`packedTrainSig (ema := ema)`), :865 (`adamWConsts`) — so it is the same
  edit, PLUS `cnxAdamVariant` (:808) takes `acc` and spells `acc` + `dp` + `<k>x` in the optimizer slot
  (`emaaccdp16xwxdroperfbf16`; `vitAdamVariant` has the spelling and the `#guard`s), because ConvNeXt
  DERIVES its entry name from the variant (route (a) in that file's words) where ViT takes `funcName`.
  Every committed ConvNeXt artifact must re-render byte-identical at `acc = 1` (`regen_verified_mlir.sh
  check`); `vit-ema-drop-render convnextin <variant>` already accepts five regions; `TestVariantPredicates`
  gets the spelling and the GELU-form count (106 → 110 with T's pair; S/B add four more).
* *Trap 1 — the per-epoch micro-batch count.* The driver refuses an epoch whose micro-batch count is
  not a multiple of k (`Train.lean` ~:2100): at 4 × 64, 1,281,167 / 256 = 5,004 micro-batches and
  5,004 mod 16 = 12. Cap with `LEAN_MLIR_G2_STEPS=4992` in the conf (= the JAX trainer's 312 optimizer
  steps × 16; its `steps_per_epoch = train_examples // 4096` drops the same remainder), and grep it in
  the precheck. ViT-B's k = 2 never met this (2,502 is even).
* *Trap 2 — the JAX trainer's arena.* The verified render and the JAX graph of one step do not peak
  alike: ViT-S's render fit the default arena at 4 × 256 while the JAX trainer OOMed at 0.75 AND at
  0.97 (XLA's rematerialisation spends the room it is given), and ViT-B's JAX 512 × 2 OOMed at the
  default. So: smoke the JAX trainer's first step at the shape before writing its ETA, and the conf's
  lever is `ENV_EXTRA=(XLA_PYTHON_CLIENT_MEM_FRACTION=0.97)` + `pc_env` (`vitb-accum-jax-4gpu.conf`).
  ConvNeXt-T at 4 × 64 micro-batches is the shape its JAX trainer already runs, so this should be
  moot for T; check S and B.
* *The method, in order:* probe (`scripts/probes/bf16_peak_memory.py` / `bf16_device_step.py
  --replicas 4` on a scratch render) → renderer `acc` → renders + `#guard`s → proofs.yml list, regen
  rows, `gen_mlir_manifest.py` → JAX config + `regen_jax_generated.sh` write / sync / check → apps →
  confs (fresh lineages, `LEAN_MLIR_DUMP_CORRECT`) → gates (§1.4, schedule gate n/a here) → smokes
  (`ONCE=1 LEAN_MLIR_MAX_STEPS=400 LEAN_MLIR_PROBE_WARM=200`; the JAX three-line patch
  `runs/2026-10-11-vits-jax-smoke/trainer_smoke_accum.patch`) → run records → `docstring-checkrefs`,
  `comment_numbers` (cite the run dir, not the number) → stage. The driver's EMA banner reads
  `emaWarmup` since a39f5cf7, so the smoke log says "constant from step 0" when the app passes it.

## 5. MobileNetV4-Conv-M — the paper's tier, and what is still out of it

Marker (ch 6): the paper's tier — 500 epochs, wd 0.1, dropout 0.2, RandAugment m15 at p 0.7, drop-path
0.075 — exists on both paths and has not run: `mnv4-full-jax-4gpu` / `mnv4-full-4gpu`, ~112 + 95 h.
Still out of it: the paper's 256² (ours 224), timm's inc1 magnitude mappings, BN decay 0.9, the
drop-path ramp over all 23 blocks, the verified depthwise init; and both tiers scored at timm's 256 / 1.0.

Today (`jax/MainMobilenetV4Imagenet.lean`): `Full` = 500 epochs, dropout 0.2, RA m15, drop-path 0.075,
no EMA, on the base's lr 4e-3 at 512×8 = 4096, warmup 5, `cropTorchvision`, f32 head; the base has
no `randAugmentInc`, BN decay 0.99.

1. Config (`Full`): `trainRes 256` + `testCropRatio`/size per the tag; `randAugmentInc true` (D8);
   `bnMomentum 0.9` (D9, eval only); the drop-path ramp over all 23 blocks (D10: the per-block rate is
   computed on the host for both paths — `Spec`/driver, not the render); `evalF32 true`;
   `mixSwitchProb` if the tier mixes (check timm's `ra4` hparams); `randAugmentFillMean true`.
2. D7 decision: RandAugment m15 — TF extrapolates past 10, the pinned timm clamps; the book's tier
   says m15, so the emitter's extrapolation stands unless the user prefers timm's clamp. Say which.
3. The timm tag: `jax/timm_eval_protocols.json` has one MNv4 row, `mobilenetv4_conv_medium.e500_
   r224_in1k` (train 224, test 256 / 1.0); the marker's "the paper's 256²" is the `…e500_r256_in1k` tag,
   whose protocol is not in the table — look it up from the pinned timm, pin which tag the tier
   reproduces, and the train and eval render sizes follow.
4. Verified: D4 (depthwise fan-in init) landed 2026-10-07 and the rerun picks it up; a 256² render
   `mobilenetv4in256_…` from `MobileNetV4Render*` (the R50 160² render is the template: a second
   resolution of the same spec, eval render at the test size), plus the `_fwd_eval_s<S>` for D18.
5. Gates: fill, hash, `comment_numbers`; `mnv4_timm_parity.py` must stay green at 224 (the parity
   gate's resolution) — the 256 render is a new artifact, not a change to the 224 one.
6. Runs: ~112 + 95 h (10-07); the ch 6 MNv4 marker goes when the pair lands and both tiers are rescored
   at timm's protocol.

**Done 2026-10-12.** The two decisions dissolved into one fact: the paper's own training config is
public — tensorflow/models `official/vision/configs/experiments/image_classification/
imagenet_mobilenetv4_conv_medium_tpu.yaml` ("Top-1 Acc: 79.9% @ 500 epochs") — and timm's
`e500_r256_in1k` recipe is not ("inspired by the paper with timm enhancements"), so the tier reproduces
the yaml on timm's net and the timm tag supplies only the comparison numbers (79.93 @ 256, 80.86 @ 320)
and a second scoring protocol. Read off the yaml and the tf-models sources it inherits from (`augment.py`,
`classification_input.py`, `preprocess_ops.py`, `lr_schedule.py`, `nn_layers.py`, `mobilenet.py`), and
set on both paths (the user: build every row; the JAX config names its source per line):
`input_size 256` (train and eval); `norm_momentum 0.997`, `norm_epsilon 1e-3` (D9's "timm 0.9" was
wrong — the paper's code is TF's 0.997); the drop-path ramp `0.075·(block_idx+1)/26` over tf-models'
26-entry decoded block list (stem, fused_ib, 21 UIBs, `convbn 960`, `gpooling`, `convbn 1280`) with
`flat_stochastic_depth_drop_rate: false` — UIB `i` at `(i+3)/26`, residual rows only
(D10's "/23" was timm's; first read here as /25 over a list without the pool, **corrected 2026-10-17 to
/26**, committed 3333bbd5 — `len(self._decoded_specs)` counts the `gpooling` entry, `imagenet_parity.md`
§11 A1; the landed 100-epoch pair carries /25); RandAugment m15 at p 0.7 under tf-models' `augment.py`: `magnitude_std 0` so
no clamp — m15 extrapolates (D7 settled; Posterize 6 bits, Solarize a no-op, enhance 2.8, rotate 45°),
TF's non-increasing mappings (D8 withdrawn: inc1 at m15 would be Posterize −2 bits and Solarize at −128,
black and inverted images), an identity branch in the 16-way draw, `translate_const 100` (±150 px),
nearest resampling at fill 128, `exclude_ops [Cutout]` (our 15-op list); TF's crop sampler at
`min_object_covered = area_range[0] = 0.08` with the whole image on exhaustion, bilinear resize without
antialias (`tf.image.resize`'s default), the eval centre crop 0.875 of the SHORTER side then resize
(`center_crop_image_v2`); AdamW ε 1e-7 (Table 10, Keras' default); the schedule `LinearWarmup` over
`CosineDecay(decay_steps = 156000)` — the cosine over every step from step 0, warmup 0 → cosine(1560)
over 1,560 steps, the step 0-based; no EMA, no Mixup, no clip, wd 0.1 excluding BN. Disclosed, not
reproduced: the init (timm's, the 10-07 D4 call), the decay reaching the classifier bias (1,000 floats),
and the head order — timm pools before its 1280 conv where tf-models' list runs that conv at 8×8 and
pools after (the net is timm's, the parity gate's) [**withdrawn 2026-10-17**: tf-models' spec list is
`convbn 960`, `gpooling`, `convbn 1280` — it pools first too; `imagenet_parity.md` §11 A3]. Code: `TrainConfig` gains `imgSize`, `evalCropTf`,
`resizeAntialias`, `cropMinCovered`, `randAugmentIdentityOp`, `randAugmentTranslateConst`, `augNearest`,
`adamEps`, `dropPathAllBlocks`, `cosineFromStep0`, every default byte-identical (66 artifacts re-emitted,
two changed); the emitter keeps timm's eval form as `_imagenet_decode_center_crop_timm` under
`evalCropTf` and the shim swaps it in under `SHIM_EVAL_SIZE`, `eval_full50k.py PROTOCOL=timm` likewise.
Verified: `mobilenetv4AdamTrainStepText` takes the final side `f` (R50's `q`; every 224 artifact
re-renders byte-identical) and `adamEps`; the `mnv4in256` family — `mnv4Imagenet256FullVerified`
(`d0 = 3·256·256`, `dropDenom 25`, the new `VerifiedNetSpec.dropIdxOffset 3`), renders `mnv4in256_fwd`,
`_fwd_eval` (256), `_fwd_eval_s320` (timm's test size), `_acc{,dp}8x128wxdropdowd01bf16_train_step` at
BN ε 1e-3 and AdamW ε 1e-7 (1,112 operands, arity green); the app passes `bnMomentum 0.997` and the
driver's new `cosineFromStep0` on `full`; `score-checkpoint mnv4-in256`, `score_timm.sh mnv4-in256`,
`timm_eval_protocols.json` row `mobilenet_v4_imagenet_full` (r256: 320 / 1.0). The 256 render at
4 × 128 peaks at 4.02 GiB of 11.68 (34 %, `runs/2026-10-12-mnv4-full-4gpu/peak_memory_256.log`), so the
geometry stays `accdp8x128`. Confs rewritten on the §3/§4 template with a grep per emitted line and the
bitmaps appended; smokes in `runs/2026-10-12-mnv4-jax-smoke/` and `runs/2026-10-12-mnv4-full-4gpu/`
(JAX 2.56 s a 4096-row step at the default arena = the 224 pace, feed-bound, ~112 h; verified 214 ms a 512-row micro-step, compute-bound at 4 ms starvation — faster than the 224 tier's 274 because TF's geometry is cheaper than the PIL-bicubic path — ~75 h; the §2 row's "~112 + 95 h" becomes ~112 + 75 h). The apex tie is generalised (the user's call): `mnv4InputGradB`, `mnv4InputGradB_eq_mnv4B_full_vjp` and
`mnv4InputGradB_correct` take `(N f : Nat)` with the sides `16f, 8f, 4f, 2f, f` and the input `2·16f` in
place of 112/56/28/14/7 and 224 — the stage ties were generic in `h w` all along and the apex binds its
widths, so the generalised module built in 12 s with the proof untouched; the 224 statements are the
instance `f := 7`. `mobilenetv4ForwardBFull_eq_slots`, which identifies the opaque stages with
`MobileNetV4FullB`'s literal-resolution forward, stays at 224 (that file's note on resolution literals),
as do the per-block ties' instances; the per-block statements bind the row's `UibSpec` and so cover `f = 8`. The ch 6 marker stands, rewritten to the yaml's
items; D7–D10 in `imagenet_parity.md` §10.3 carry the correction.

## 6. MobileNetV2 — TF-slim in three places

Marker (ch 6): TF-slim in three places, on both paths: weight decay off the depthwise kernels, slim's
Inception colour distortion, RMSProp's learning rate inside the momentum term; slim's truncated-normal
initialization (σ 0.09) against He-uniform; and the pair predates the depthwise fan-in init on the
verified path, the suspect for its −0.51. One rerun of both halves takes all of it —
`mnv2-full-jax-4gpu` / `mnv2-default-4gpu`, ~38 + 66 h. The sixteen workers of 96 stay 4×64.

Today (`jax/MainMobilenetV2Imagenet.lean`): lr 0.045 at 256, RMSProp, `wdExcludeNormBias`, dropout
0.2, no warmup, BN decay 0.997, no AutoAugment, f32 head. Nothing here is a flag yet.

1. D11, weight decay off the depthwise kernels: a per-tensor decay mask. JAX: a flag
   (`wdExcludeDepthwise`) extending the norm/bias skip-list by kernel shape. Verified: the decay is in
   the optimizer step of the render, so a variant (`wxdw`) whose per-tensor wd vector zeroes the
   depthwise slots — `OptRecipe` already carries `wx` (norm/bias off); extend the mask builder. M.
2. D12, slim's Inception colour distortion: `inception_preprocessing.distort_color` (fast mode:
   brightness 32/255, saturation 0.5–1.5, hue 0.2, contrast 0.5–1.5, two op orders) in the shared
   `emitDataLoading`, so the trainer and the shim get it together; a flag (`colorDistortSlim`). The
   reference is unverified offline — a gate against a slim transcript needs TF-slim in `.venv-timm`
   or a hand-rolled numpy twin; decide which. M.
3. D13, RMSProp's learning rate inside the momentum term (TF1: `mom = μ·mom + lr·g/√(v+ε)`,
   `w −= mom`; ours applies lr outside). JAX: a flag on the emitted update. Verified: a new optimizer
   op spelling (`rmspropTf`) with its `opt_step_tie` row and T3 tie; it differs only while lr moves,
   so the Imagenette twin at constant lr cannot catch a wrong placement — the tie must. M; shared
   with B0 (§7).
4. Slim's truncated-normal init σ 0.09 on the convolutions against He-uniform: an init kind on both
   paths (`VerifiedConfig` has kinds 1–7; add 8), JAX a flag. S.
5. D4 landed 2026-10-07; the rerun picks it up. `evalF32 true`, `bnMomentum` as slim (0.997 stays).
6. Gates: `mnv2_timm_parity.py --controls` stays green (forward unchanged); `opt_step_tie.py` gains the
   TF-RMSProp row; a colour-distortion known-answer gate; fill n/a (no RandAugment).
7. Runs: ~38 + 66 h; the ch 6 MNv2 marker goes when the pair lands.

**Done 2026-10-12.** Three things changed before any code. (1) D12 was misdescribed: `train_image_classifier.py`
passes no `fast_mode`, so slim's `distort_color` ran in fast mode — brightness ±32/255 then saturation
×U(0.5, 1.5), ONE order, clipped; hue, contrast and the four orderings are the non-fast mode. (2) D11, D13
and the init were confirmed from `mobilenet.training_scope` (`l2_regularizer` on `slim.conv2d` only,
`weights_regularizer=None` on `separable_conv2d`; one `truncated_normal_initializer(0.09)` for conv2d,
separable_conv2d AND fully_connected, so the depthwise kernels and the logits take it too),
`conv_blocks.expanded_conv` (the depthwise IS `slim.separable_conv2d`) and TF's `training_ops.cc`
(`mom <- μ·mom + lr·g/√(ms+ε)`, `var <- var − mom`, `ms` from 1). (3) `research/slim/nets/mobilenet/README.md`'s
reproduction command names rows the marker did not: `--label_smoothing=0.1`, `--moving_average_decay=0.9999`
(TF's warmup-corrected `ExponentialMovingAverage(decay, global_step)`, restored at eval), `--learning_rate=0.045 ×
NUM_GPUS` on `--batch_size=96` per clone (0.045 per 96 images → 0.12 at this pair's 256),
`--num_epochs_per_decay=2.5 / NUM_GPUS` per-clone epochs (= ×0.98 every 2.5 epochs of data, not the paper
sentence's "per epoch"), "about 700K steps" of 768 (≈ 420 epochs), `--preprocessing_name="inception_v2"` =
`inception_preprocessing` (crop area ≥ 0.05, aspect (0.75, 1.33), 100 attempts, bilinear without antialias,
`(x − 0.5) × 2` on train and eval). The user's call, as for MNv4: the README's command, every row, 0.12 at 256,
×0.98 per 2.5 epochs, 420 epochs; D12's reference a numpy twin.

JAX: `TrainConfig` gains `rmspropTf`, `wdExcludeDepthwise`, `initTruncNormal`, `colorDistortSlim`,
`inputScalePm1`, `cropAreaMin`, `cropMaxAttempts`, `cropAspectLo` / `cropAspectHi`, every default byte-identical
(66 artifacts re-emitted, the four MNv2 files moved); every conv-BN and depthwise init site now routes through
`emitConvBnInitCfg` / `emitDwBnInitCfg` so an init flag cannot reach some kernels and not others (53 of 53 take
the truncated normal: 35 conv-BN, 17 depthwise, the logits). `mobilenetV2ImagenetConfig` is the README's recipe
with its line beside each value; `default` = 90 epochs, `full` = 420. Verified: `OptRecipe.rmspropTf` (`rmstf` /
`rmstfdp`) = the new op `rmsBufNextTfF` (`%lr` multiplied onto the normalised gradient inside the buffer) +
`sgdParamF` at a baked `%one`; `Proofs.rmsBufNextTf` / `rmsPropTfStep` with `rmsPropTf_triple_faithful` (rfl),
`rmsBufNextTf_smul`, `rmsPropTfStep_theta_at_const` and `rmsPropTfStep_theta_lr_change` (the two placements
differ by exactly `μ·b·(lr₁ − lr₀)` per coordinate when the rate moves — the D13 statement);
`rankWdDecaysNoDw` + `wdNameBy`'s `decays` argument (the `dw` marker; `optAllParams` takes it too);
`mnv2AdamVariant` / `mobilenetv2AdamTrainStepText` take `ema` and `dw` (B0's fifth-region threading), every
committed MNv2 artifact byte-identical; the render `mobilenetv2in_emarmstfdp64wxdwdoeps0001bf16` (314 all-reduces,
744 operands, 122 `%wdz` = 105 one-dimensional + 17 depthwise, 158 `%one` / `%lr` / `%emad` broadcasts) with
`#guard`s, proofs.yml, MANIFEST, `TestVariantPredicates` (the third ε row; `tfdp` and `dwdo` spell nothing);
`VerifiedConfig.truncNormalSigma` → `mkParam` at std 0.8796σ (TF's ±2σ redraw, variance-matched) above every fan
rule, `LEAN_MLIR_TRUNC_NORMAL=0|1`; `mnv2ImagenetRmsSchedule` 0.12 / 0.98 / 2.5 / staircase; the app at 420 epochs,
σ 0.09, and `rmsOn` as a substring test (its `startsWith "rms"` would have missed `emarms…`). Gates:
`opt_step_tie.py` generalised to four fixture parameters (`p3 = [7,1,2,2]`, a depthwise shape) with rows
`rmswx` (B0's trainer), `rmstfwxdw` and `emarmstfwxdw` (MNv2's `full`) — 10/10 at 1.5e-7, the counterfactuals
9.8e-1 and 5.4e+1; `scripts/gates/color_distort_gate.py` — the numpy twin of TF's `adjust_saturation` kernel tied
to `tf.image.adjust_saturation` (4.8e-7) and to the EMITTED `_imagenet_decode_random_crop_flip` run verbatim with
its three draws pinned (4.8e-7 at five (δ, f)), the text order, and controls at f = 1.5 (brightness and
saturation commute exactly except through the S clamp, so f = 0.5 is a dead control — in the gate's header);
the shim hash reproducible at seed 0, distinct at seed 1 (`SHIM_DETERMINISM=1`); parse 372, arity 281, manifest,
`regen_jax_generated` write / sync / check, `regen_verified_mlir check` 76/76, comment_numbers, module_refs,
repo_shape, target names, docstring-checkrefs, both DRY_RUN prechecks. Confs rewritten standalone on the §5
template: fresh lineage by construction (the variant is in the checkpoint name; `~/mnv2_full_jax_2612/` for
JAX), a grep per emitted line (the TF optimizer lines, the depthwise mask leaf, 53 truncated normals, the
staircase at 2.5, LR 0.12, 420 epochs, EMA + its warmup form, the colour lines, 127.5, the crop ranges, bilinear
without antialias, f32 eval; the render's `rmsBufNextTfF`, `%one`, 122 `%wdz`, the ε and wd constants, −α/K),
`LEAN_MLIR_DUMP_CORRECT` appended. Smokes (`runs/2026-10-12-mnv2-jax-smoke/`, two 100-step epochs validated in f32, lr 0.12 on both — inside
the first 2.5-epoch stair — exit 0 at the default arena, 81 ms a 256-row step compile excluded → ~49 h;
`runs/2026-10-12-mnv2-default-4gpu/`, one capped window through the supervisor: the RMSPROP / DROPOUT / EMA
(4 regions, TF warmup-corrected, the shadow scored) / truncated-normal banners, the `exp x0.98/2.5ep
staircase+warmup 0ep, baseLR 0.12` line, 742 outputs, 96 ms a step median at 6 ms starvation — slim's
bilinear feed is cheaper than timm's bicubic, as MNv4's was — → ~63 h; the render peaks at 1.53 GiB of
11.68, 13 %). §2's "~38 + 66 h" becomes ~49 + 63 h at 420 epochs. Disclosed, not reproduced: slim's eval
(`central_crop(0.875)` on both sides, then a non-aspect-preserving bilinear resize; scored at timm's 224 / 0.875
like every pair, post-hoc scorable) and the sixteen asynchronous workers of 96. Not done: `rms-tie`'s TF form (an
Imagenette-size `rmstf` artifact would be needed; `opt_step_tie` is the gate against the reference's own code).
The ch 6 marker stands, rewritten to the README's items; D11–D13 in `imagenet_parity.md` §10.3 and the manifest
row carry the sources. B0 inherits `rmspropTf` for its D13.

**Addendum 2026-10-12 (later; D23) — the two disclosed rows, closed.** The user's call ("match MNv2 to the
original paper, not timm's variant"): the recipe already was the README's, and the audit found exactly one
row still timm's — the validation crop — plus one TF 2 kernel where slim ran TF 1's. Both landed, so MNv2
is every row like MNv4 (§5) and B0 (§7). (1) slim's `preprocess_for_eval`: `tf.image.central_crop(0.875)`
keeps the centred 0.875 of BOTH sides of the decoded image (offsets `int(h · 0.125 / 2)` in float64,
sizes `h − 2 · offset`), so the window has the JPEG's own aspect, then `resize_bilinear` to 224 × 224,
which does not keep it; timm's form (resize the short side to 256 bicubic + antialias, crop 224) was what
both paths scored through. `TrainConfig.evalCropSlim` (with `evalCropTf`, as B0's `evalCropTfEfficientNet`
is) emits it as `_imagenet_decode_center_crop`, timm's form staying under `_timm` for `SHIM_EVAL_SIZE` /
`PROTOCOL=timm` (the post-hoc second score, like every pair); the verified eval render is untouched (a
224² input either way), the shim carries the same function, so the pair stays matched. (2) TF 1's
bilinear: slim's `tf.image.resize_images` in the TF 1 that trained MobileNetV2 was `resize_bilinear` with
`half_pixel_centers=False` (the grid aligned to pixel corners), where §6 had set TF 2's `tf.image.resize`
(half-pixel centres) — `resizeLegacyTf1`, which §7 added for B0's bicubic only, now covers `.bilinear` too
and MNv2 sets it, on the train crop and the eval. Found by the gate's first run, and the reason the emitted
form decodes whole then `crop_to_bounding_box`: `tf.io.decode_and_crop_jpeg` (TF TPU EfficientNet's and
tf-models' own call, which the `evalCropTf` forms keep) is NOT the whole decode's window at an iMCU-unaligned
window — one column differs, by up to ~20 grey levels, which the resize then spreads (5 of 8 test shapes,
up to 0.18 of the [−1, 1] range after resizing); slim's dataset decoder decoded whole and `central_crop` cut
the tensor. Gate: `scripts/gates/slim_eval_crop_gate.py` on the trainer and the shim — the emitted decode
exec'd verbatim against slim's own ops (`decode_jpeg → convert_image_dtype → central_crop → resize_bilinear
→ (x − 0.5)·2`) at 1.4e-5 of the range over eight shapes, against a numpy twin of TF 1's grid at 0.003 grey
levels, and three red controls (TF 2's half-pixel bilinear up to 121 grey levels, timm's form 254, tf-models'
shorter-side square 253); in jax.yml. `pc_crop` gains `tf1-bilinear`; both confs grep the two legacy calls,
slim's comment line, `central_crop`'s arithmetic and the `_timm` form, and refuse `ResizeMethod.BILINEAR`.
Re-emit: only the four MNv2 files moved (62 byte-identical). Smokes: `runs/2026-10-12-mnv2-jax-smoke-evalcrop/`
(one 50-step epoch, the f32 validation over 50,000 through slim's decode in 12.4 s, exit 0) and
`runs/2026-10-12-mnv2-verified-evalcrop-smoke/` (one capped epoch through the supervisor, the epoch-end
validation through the shim's decode). The ch 6 marker now lists the crop and the kernel and drops "stays
disclosed"; the manifest row and D23 in `imagenet_parity.md` carry the sources. No run launched.

## 7. EfficientNet-B0 — the final pair

Marker (ch 7): the final pair — the staircase floored on the global step from step 0, no decay on BN or
biases, drop-connect 0.2·i/16, BN ε 1e-3, AutoAugment v0 with bicubic geometry, TensorFlow's crop
sampler without its centre-crop fallback, the fp32 classifier head, synchronised BatchNorm,
TensorFlow's classifier init U(±1/√1000) with the depthwise and squeeze-excite fan inits, RMSProp's
learning rate inside the momentum term, and SAME padding on the strided depthwise convolutions;
`enet-full-jax-4gpu` / `enet-default-4gpu`, ~47 + 80 h.

Today (`jax/MainEfficientNetImagenet.lean`): RMSProp lr 0.016 at 256, `wdExcludeNormBias`, dropout
0.2, drop-path 0.2, AutoAugment, EMA, f32 head, clip 0; BN ε 1e-3 and the bicubic v0 geometry landed
with X1; the sync-BN render exists (`imagenet_syncbn_probe`); D4 / D5 (depthwise and SE fan inits)
landed 2026-10-07. Of the marker's list, owed in code:

1. TF's crop sampler without its centre-crop fallback: the emitted `sample_distorted_bounding_box`
   path keeps TF's fallback today; a flag that drops it (TF's reference uses `max_attempts 100` and no
   fallback? — read `preprocessing.py` first; the fallback may be TF's own). S once pinned.
2. D6, the classifier init U(±1/√1000): an init kind for the head on both paths. S.
3. D13, RMSProp's rate inside the momentum: §6.3, the same op. Shared.
4. The staircase floored on the global step from step 0 and drop-connect 0.2·i/16: check the emitted
   `expStaircase` / `dropPath` ramp against the TF reference's `global_step` arithmetic; likely
   already right (they are in the pair's recipe, not the "owed" sentence of 3bc91118) — confirm, and
   list here as "verified, no change" or as an item.
5. D3, SAME padding on the strided depthwise convolutions (TF) against our symmetric padding:
   `enet_timm_parity --pad tf` puts it at 0.37 / 0.048 of the logit scale. JAX: a flag on the conv
   emit (S). Verified: the render's padded conv op and the T2 / T3 ties that are stated on the
   symmetric form — the padding changes the function, so the tie moves to the TF-padded spelling:
   a new op or a padding argument on the existing one, its VJP, the fold lemmas for the asymmetric
   case, and the whole-net tie re-pointed. M–L; parked 2026-09-25 for this reason. This is the one
   item in the five chapters that is proof work, and it decides whether B0's pair is "the paper's" or
   "the paper's at symmetric padding, disclosed".
6. `evalF32 true`; `mixSwitchProb` n/a (B0 has no Mixup); fill n/a (AutoAugment's fill is X1's).
7. Gates: `enet_timm_parity.py --controls` (the `--pad tf` control becomes the pass once D3 lands),
   `opt_step_tie.py` (TF-RMSProp), `crop_sampler_gate.py` for the sampler, hash.
8. Decision: do D3 now (the pair is "the paper's", ~2 days of proofs) or run the pair at symmetric
   padding with the marker keeping that one clause. The user's call before §7 starts.

**D3 done 2026-10-12 (the user: proofs).** Item 5's estimate was stale: the XLA-`SAME` strided-depthwise
op family — `depthwiseStridedXla{,Bf16,F,Back,BackBatched,WeightGradB,BiasGradB,…}` with
`depthwiseStride2FlatXla`, its input / weight / bias VJPs, the bias split, the batched back-link
faithfulness, the `*TiedBAt` / `_hasGradAt` predicates and the erasure lemmas — had landed with
MobileNetV2's switch (3d9b14d6, 2026-08-08: "the asymmetry is a decimation PHASE"), and B0's render
simply used the symmetric tokens at its four strided sites. So the verified half is a re-pointing. One
new op: `depthwiseStridedXlaWeightSgdB`, the fused-SGD-B form for the Imagenette SGD-inline artifact
(constructor, `den`, `_eq_grad`, skel and printer arms; a `TestBatchedEmitTie` row). MNv2's private
copies of the Xla helpers moved into the kit so both nets share them: `dStridedXlaInB` with its `den`
lemma, `_eq_batchMapBackward`, `_smul`, `_shard` (`BackLinks`, `ParamGradNodes`, `SyncKit`),
`hasGradAt_depthwiseStridedXla`, `den_allReduceMeanF_depthwiseStridedXlaWeightGradB_shard`,
`DepthwiseStridedXlaWSync{,At}` with their `_of_scaled`; new for the swish stage, `dwbsSXlaB` with its
`_differentiable` / `HasVJP`, `dwbsSXlaBackBatchedGraph{,_faithful}`, `dwbsSXlaB_back_eq`. The 13 B0
proof files and the writer are restated on the Xla forms (a whole-word sweep, `depthwiseStrided →
depthwiseStridedXla`, `depthwiseStride2Flat → …Xla`, `dwbsSB → dwbsSXlaB`, `dStridedInB →
dStridedXlaInB`; the one hand fix was the bias split's namespace, `Bf16Fold.`), and every tie built
unchanged — `efficientnet_net_tiedG`, `efficientnet_net_syncTiedG`, `enet_net_lossGrad`,
`efficientnetInputGradBFull_correct`, the fused Imagenette tie — the stage ties were generic in the op
all along and the apex binds its tokens. The render's forward now prints TF's literal `SAME` pads
(`[[0, 1]]` at k = 3, `[[1, 2]]` at k = 5), the dgrad `[[2, 0]]` / `[[3, 1]]` and the wgrad `[[0, 2]]` /
`[[1, 3]]`, as MNv2's; 36 B0 artifacts re-rendered (Imagenette and ImageNet; `MANIFEST.md` follows).
JAX: `mbconv_block` pads its depthwise `'SAME'` under `NetSpec.convPadStyle` (`.xlaSame` is B0's
default; MNv4's blocks are `uib_block` / `fused_mbconv_block` and keep timm's symmetric pad), so only
the B0 emitters move. Gates: `enet_timm_parity.py` defaults to `--pad tf` with the old hybrid as a
red control (ImageNet 1.7e-6 / 7.3e-7; hybrid 4.8e-2, symmetric 4.6e-2, ε 5.8e-3). Also green:
`regen_verified_mlir check`, manifest, parse 372, arity 281, `fwd-tie efficientnet` and `--eval` bit-exact,
`TestBatchedEmitTie` (a `depthwiseStridedXlaWeightGradB` / `…SgdB` prefix row added), `TestXlaPadOps`,
`TestVariantPredicates`, `AuditAxioms` (the moved MNv2 lemma re-pointed to `Proofs.SyncKit`),
`efficientnet-dp-check` (forward bit-exact, gradient 8.1e-4 ≤ 1e-2) and `efficientnet-syncbn-check`
at `PJRT_REPLICAS=2`, comment_numbers, module_refs, repo_shape, target names, docstring-checkrefs,
book_xrefs, both DRY_RUN prechecks; preview PDF 0 errors, overfull 82 → 82, 232 pages, 2 changed
paragraphs. Pre-existing and not D3's: `enet_forward_tie.py` fails its absolute 1e-4 at 2.9e-4 after and
1.9e-4 before (A/B: the HEAD render against a symmetric-patched reference), 200× under its padding
control. As at MNv2's switch (c4d07bd8, "switch and re-run"), the Imagenette B0 pair is re-run on
the faithful net — the chapter's 89.96 % was trained on the symmetric artifact (41 min a half). The
marker's "which moves the T2 / T3 ties" clause is gone (process, not a recipe gap); the padding item
stays in it until the pair lands. Items 1–4, 6 and 7 are the rest of §7.

**The rest of §7 done 2026-10-12 — TF TPU EfficientNet's own code, every row.** Read off
tensorflow/tpu `models/official/efficientnet/{main,utils,efficientnet_model,preprocessing,
autoaugment}.py` (master, fetched 2026-10-12) before anything was set, as at MNv4 and MNv2, and three
of the marker's items changed shape on the way. (1) Item 1 is void: `_decode_and_random_crop` samples
at `max_attempts=10` and a crop that came back whole (`_at_least_x_are_equal(original_shape,
tf.shape(image), 3)`) is replaced by `_decode_and_center_crop` — TF's sampler HAS the fallback,
`cropFallbackCenter` is it, and the marker's "without" was wrong (F2 corrected). (2) "No decay on BN
or biases" is timm's rule, not TF's: `main.py` adds `weight_decay · Σ l2_loss(v)` over every
trainable variable whose name lacks `'batch_normalization'`, so the SE and classifier biases decay
(D21). (3) "AutoAugment v0 with bicubic geometry" was timm's rendering: `autoaugment.py` runs the
geometry through `contrib_image.{transform,rotate,translate}` at their default NEAREST, fills with
128, moves TranslateY by `(l/10)·250` PIXELS (`distort_image_with_autoaugment`'s `translate_const`),
and posterizes through TF's clamped shift (0 bits keeps the top bit); the 25-row table was already
TF's op for op (D20). Confirmed, no change: the staircase on the global step from step 0, the
`0.2·i/16` drop-connect ramp, BN 0.99 / 1e-3, label smoothing 0.1, dropout 0.2, EMA 0.9999
warmup-corrected (`num_updates=global_step`) and scored, the SAME stem and head convs without bias,
the SE convs with, lr 0.016 at 256. Also read off and set: the classifier at `dense_kernel_initializer`'s
`U(±1/√shape[1])` — the CLASS count sets the range (D6); `tf.image.resize_bicubic` (TF 1: no
antialias, corner-aligned) on every resize and `_decode_and_center_crop`'s own arithmetic at eval
(D22); the eval forward in f32 (F8). Disclosed, not reproduced: the conv kernels' distribution
(`tf.random_normal` at He fan-out where every init here is variance-matched uniform), warmup at
`lr · step / W` from step 0 (ours `(step + 1) / W`, one step ahead), the eval crop scored at timm's
protocol too (post hoc, `PROTOCOL=timm`).

JAX: `TrainConfig` gains `autoAugmentTf`, `wdDecayBiases`, `initHeadTfUniform`, `resizeLegacyTf1`,
`evalCropTfEfficientNet`, every default byte-identical (66 artifacts re-emitted, the four B0 files
moved: trainers and shims); `efficientNetB0ImagenetConfig` names its source per line (`rmspropTf`,
`wdDecayBiases`, `initHeadTfUniform`, `autoAugmentTf` + `augNearest` + `randAugmentTranslateConst
250` with `augBicubic` off, `resizeLegacyTf1`, `evalCropTf` + `evalCropTfEfficientNet`, `evalF32`),
the 80-epoch `default` tier with it. The `_wd_mask` under `wdDecayBiases` is STRUCTURAL — a
`(W, γ, β)` triple is a conv-BN site, a `(W, b)` pair decays both — since the rank cannot tell a bias
from a γ. Verified: `rankWdDecaysKeepBias` (by name: γ ends in `g`, β in `bt`; guarded on `enetSig` —
98 BN parameters excluded, 33 biases decaying, 131 under `wx`) behind the `wn` marker in `wx`'s slot;
the render `efficientnetin_emarmstfdp64dropdowneps0001bf16` (`OptRecipe.rmspropTf` at `enetRmsHyper`,
100 lines naming `%wdz` = 98 + its constant and banner; every committed artifact byte-identical);
`VerifiedConfig.headTfUniform` → `mkParam`'s kind-0 rank-2 branch at variance `1/(3·out)`,
`LEAN_MLIR_HEAD_TF_UNIFORM=0|1`, an INIT banner, set in the ImageNet driver; the shim carries the
pipeline rows. Gates: `scripts/gates/aa_v0_tf_check.py` (the v0 table, TranslateY at `(l/10)·250`,
a 7° rotation emitting only source pixels and the fill, the TF shift clamp measured on
`tf.bitwise.right_shift` itself, TF's level mappings; HEAD's timm-form trainer red on 6 rows as the
control; in jax.yml), `opt_step_tie.py` row `emarmstfwn` 7.9e-8 (the `rmswx` row retired — both B0
trainers carry TF's placement now, so it has no reference; 5.19e+01 as the counterfactual, recorded),
`enet_timm_parity --controls` unchanged (1.7e-6 / 7.3e-7; the forward did not move),
`TestVariantPredicates` (136 spellings, the fourth ε row), `TestOptStepFixtures`, the shim hash
reproducible at seed 0 and distinct at seed 1 under `SHIM_DETERMINISM=1`, `regen_jax_generated`
write / sync / check, `regen_verified_mlir check`, manifest (+ `wn` in its decoder), parse 373, arity
282, comment_numbers, module_refs, docstring-checkrefs, `pc_crop`'s new `tf1-bicubic` kernel. Confs:
`enet-default-4gpu` on the new variant with a grep per emitted line (TranslateY, 250 px, NEAREST, two
legacy resizes, the eval crop, the render's 100 `%wdz`) and `LEAN_MLIR_DUMP_CORRECT` appended;
`enet-full-jax-4gpu` on a fresh lineage (`~/enet_tf350_2612/`) with a grep per line (the TF update,
`_entry`, `1/√1000`, the pipeline, `evalF32`). Smokes: `runs/2026-10-12-enet-default-4gpu/` (the conf's own precheck and launch, `ONCE=1
LEAN_MLIR_MAX_STEPS=400 LEAN_MLIR_PROBE_WARM=200`: 121 ms/step median at 2 ms starvation — 148 on
the PIL-bicubic feed on 09-25; TF's nearest AutoAugment and TF 1's bicubic are the cheaper shim —
606 s/epoch + ~85 s eval/ckpt → ~67 h; no checkpoint under the new variant) and
`runs/2026-10-12-enet-jax-smoke/` (two 100-step epochs on four cards at the default arena, the
warmup reading 0.0032 then 0.0064 as `0.016·(step+1)/500` has it, the f32 eval line before every
validation, 99 ms a step in the second epoch → ~49 h); the §2 row's "~47 + 80 h" becomes ~49 + 67 h.
Both DRY_RUN prechecks pass on the new greps. And the Imagenette rerun D3 owed landed the same
morning (`runs/2026-10-12-enet-imagenette-d3/`): **90.04 / 98.29** at epoch 80 (best 90.17) on the
`SAME`-padded net against the symmetric artifact's 89.96 / 98.45, 40.7 min on one card — inside
the split's CI either way, so ch 7's Imagenette section, its result table and the two appendix
copies carry the new run (the log excerpt, the first-four-epochs sentence, "eight tenths of a point
over MobileNetV2"). The ch 7 marker is
rewritten to TF's rows (the fallback WITH, decay off BN alone, nearest geometry, TF 1's bicubic, the
f32 eval); `imagenet_parity.md` D6 / D13 / D20–D22 / F2 and §10.4 carry the sources.

## 8. What this does not cover

* The runs' order and boxes (`imagenet_parity.md` §10.4): A3 pair first, ViT-Ti, then A2, MNv2 … as
  boxes free; every launch is the user's word.
* The 2018 R50 pair's McNemar p 0.015 — a seeds-per-arm question, not a marker.
* timm's `cooldown_epochs 10` on the RSB tiers — disclosed in the configs, not reproduced.
* The Imagenette twins: nothing here changes a chapter's Imagenette number.

## 9. Status

| net | config | verified driver | render | confs | gates | smokes | commit | run | marker |
|---|---|---|---|---|---|---|---|---|---|
| R34, R50-2018 | — | — | — | — | — | — | 8a659b2f, 5be4f349 | landed | R34 deleted; 2018 paid |
| R50 A3 / A2 / A1 | done | done | existing | done | green | done | 4d612bb9 | A3 landed 10-13/14 (77.60 / 78.08); A2 / A1 pending | A3's deleted; A2/A1's stands (RASampler, the runs) |
| ViT-Ti | done | done | done (the `smx` render, 2170bd9f) | done | green | done | 320f5045, 3336f4a7 | ✅ landed 2026-10-16 (04c72553): verified 71.612 / 90.948, JAX 71.638 / 90.980 | stands for the patch-embed bias (R4) only |
| ViT-S / B | done | done | done (B: the first `acc` render) | done | green | done | a39f5cf7 | pending | stands |
| ConvNeXt-T / S / B | done | done | done (the first `acc` ConvNeXt renders, six) | done | green | done | a0c125b3 | ✅ T landed 2026-10-28: verified 82.146 / 95.910, JAX 81.990 / 95.87 (p = 0.18); S / B pending | T's deleted; S/B's stands |
| MNv4-Conv-M | done (the paper's yaml) | done | done (`mnv4in256`, five) | done | green | done | 263da67a (+ 3333bbd5: /26 and the crop fallback) | ✅ the 100-epoch pair landed 2026-10-15 (6bcdb7f3): verified 77.35 / 93.53, JAX 77.39 / 93.46 — before 3333bbd5; the 500-epoch tier pending | stands, rewritten |
| MNv2 | done (the README's command, every row: + `evalCropSlim` / TF 1's bilinear, D23) | done | done (the first `rmstf` + EMA MNv2 render) | done | green | done | 6762494c + the D23 addendum | pending | stands, rewritten (the crop and the kernel in it) |
| EfficientNet-B0 | done (TF's code, every row: 5 new flags + `rmspropTf` / `evalF32`) | done (`headTfUniform`) | done (D3's 36 Xla artifacts; the `emarmstfdp64dropdowneps0001bf16` render) | done | green | done | a89fa9af + dedb34f5 | pending | stands, rewritten to TF's rows |

## 10. What is still disclosed, per net, and what closing each would take (2026-10-12)

Every ch 5–9 marker now reads items / state / job / hours, and four of them keep a row "disclosed" that
their rerun does not pay. Asked on 2026-10-12 what each would take, with two confirmations first:

* **ResNet A2 / A1 — timm's index-level `RASampler` IS in code** (4d612bb9): `repeatedAugAdjacent := true`
  on the A2 base config (A1 inherits it; A3 has no repeated augmentation, `repeatedAug := 1`), the
  `a2accum` / `a1` shims carry it, and `repeated_aug_gate.py` ties the a2accum shim at 0.681 against the
  re-shuffled control. The A2/A1 marker's item is paid by the rerun as it says.
* **ViT-Ti at 320f5045 (the commit the pair runs from) — everything its marker lists is in it.** The only
  ViT-Ti change after it is 3336f4a7's `evalF32` re-plumbing (module globals in place of one-element lists;
  the trainer's own validation was f32 already — the fix was for `vit_timm_parity.py`'s forcing of
  `mod.DT`), plus the JAX conf's grep line and a shim comment. The renders, the verified app, the verified
  conf and `NetsCore`'s ViT parts are byte-identical to HEAD, and the reference is rescored in f32 post hoc
  regardless (§1.6). The patch-embedding bias (R4) landed 2026-10-20 for S and B; the Ti pair carries it disclosed.

| # | net | row | the reference's own form | ours today | what it takes, both paths | size | proofs |
|---|---|---|---|---|---|---|---|
| R1 | MNv4 | decay on the classifier bias | the yaml excludes BN only — tf-models' L2 over every trainable variable but BN's | `wx`: norm γ/β AND biases off | JAX `wdDecayBiases := true` on `mobilenetV4ConvMImagenetConfigFull` (the structural `_wd_mask` already feeds the AdamW update's `WD * msk * p` term); verified `wn` for `wx` on the `mnv4in256` render — `rankWdDecaysKeepBias` is name-generic (BN names end `g` / `bt`), B0's count guard repeated on MNv4's signature — then MANIFEST, proofs.yml, `TestVariantPredicates`, an `opt_step_tie` row (AdamW + `wn`), both confs flipped. 1,000 floats under decoupled wd 0.1: nil on accuracy, "every row" | S, ~3–4 h | none — the decay set is a constant of the optimizer tail |
| R2 | MNv4 | init | tf-models' initializers, to be READ off `nn_blocks.py`, `mobilenet.py`, `classification_model.py` and the yaml before anything is set (as σ 0.09 was off slim's `training_scope`); the expectation is Keras `VarianceScaling` (fan-in, truncated normal) on the conv blocks and the classifier's own rule | timm's `_init_weight_goog` with D4's depthwise fan k² | one init kind per layer class: JAX through `emitConvBnInitCfg` / `emitDwBnInitCfg` (every conv-BN and depthwise site already routes through them) and the head's init site; verified `VerifiedConfig` → `mkParam` (the templates are `truncNormalSigma`, `headTfUniform`, `seFanOutInit`), the app sets it, an INIT banner; only the MNv4 files re-emit; gates: per-tensor variance against the rule on both paths, the shim hash | S–M, about half a day | none — host-side |
| R3 | MNv4 | head order — **⛔ void 2026-10-17**: `_mnv4_conv_medium_block_specs` ends `convbn 960`, `gpooling`, `convbn 1280`, the same order as timm (`imagenet_parity.md` §11 A3); nothing to close | ~~tf-models' block list runs `convbn 960 → 1280` at 8 × 8 and pools after~~ | timm's: pool, then `conv_head` on the pooled vector | a different NET — BN and ReLU6 do not commute with the pool. JAX: two spec lines. Verified: a new render (the 1280 conv at 8 × 8 is a new op shape) and the whole-net ties re-pointed — `mobilenetv4ForwardBFull_eq_slots`, `mnv4InputGradB_correct`, the sync tie — D3-shaped; `mnv4_timm_parity.py` red by construction, so a tf-models-form oracle for the head; +78.6 MMAC (~8 % of Conv-M); the Imagenette twin and the landed 224 tier are then timm's net — the twin "switch and re-run" (~40 min), the 224 tier disclosed or rerun (100 epochs) | M–L, 1–2 days | yes — the apex, the slots equality and the sync tie restate on the new head |
| R4 | ViT-Ti / S / B | patch-embedding bias | PyTorch's `Conv2d` default, `U(±1/√fan_in)` with fan_in = 3 · 16 · 16 = 768 → ±0.036 (timm's `init_weights_vit_timm` and DeiT's `_init_weights` touch `nn.Linear` only, so the Conv2d keeps the default) | 0 on both paths; `vitInit` already draws the patch WEIGHT at `U(±1/√fan_in)` | ✅ **2026-10-20** — built as described: `VLayer.patchEmbed` (init kind 8) + `VerifiedConfig.patchEmbedBiasTorch`, JAX `TrainConfig.initPatchEmbedBiasTorch`; set by the ViT-S/B drivers on both paths, prechecked by their four confs; Ti's landed pair keeps 0, disclosed. Was: JAX: one init line under a flag (`initPatchEmbedBiasTorch`); verified: a `VerifiedConfig` flag → `mkParam` for that one bias (host-side; no render moves), the three ViT ImageNet apps set it, an INIT banner, an init-statistics check on both paths. Ti's running pair keeps it disclosed; S / B, not launched, take it | S, 1–2 h | none |

**Decision, R3 first:** the paper's own MAdd count for Conv-M says which head it counted. If it is the pooled
one, the 8 × 8 conv is tf-models' implementation and not the paper's net — leave it disclosed and say so in
the marker. **Order proposed:** R1 + R4 (one afternoon, no proofs), R2 after the read, R3 on the MAdd
answer. Each is the user's call, and none moves a landed number.

**Still disclosed elsewhere, for the ledger** (none is a marker item):
* MNv2 — the sixteen asynchronous workers of 96 (asynchronous SGD has no synchronous equivalent; stays).
* B0 — the conv kernels' distribution (`tf.random_normal` at He fan-out where every init here is
  variance-matched uniform; a normal-draw kind in `mkParam` and the JAX init would close it — S, both paths,
  and it touches a convention every net shares), and the warmup one step ahead (`lr · step / W` from step 0
  where ours is `(step + 1) / W`; S, the schedule emitter and the driver's warmup, with the schedule gate).
* ConvNeXt — Mixup's λ granularity (once per 4096 / per 256 / per 128 in the official trainer); a variance
  detail, not a row.
* ResNet RSB — timm's `cooldown_epochs 10` (§8; disclosed in the configs).
* The second scoring protocols (timm's for MNv4's paper tier, MNv2, B0) are post-hoc scores kept beside the
  papers' own, not gaps.

**§7 pair landed 2026-10-21.** JAX `runs/2026-10-16-enet-full-jax-bf16-350ep/` 77.168 / 93.39 (42.49 h,
no restart); verified `runs/2026-10-16-enet-verified-bf16-350ep/` 77.208 / 93.464 (65.14 h). A tie:
+0.04 / +0.07, McNemar 2,210 vs 2,190, p = 0.77 (EMA), p = 0.41 (live). The 09-12 pair's −0.32 window
offset is gone. Chapter 7's ImageNet section carries the pair and the marker is out; the B0 tie
headers name the `emarmstf` render. The one ch 7 row left disclosed is nothing: every line of the
listing ran.
