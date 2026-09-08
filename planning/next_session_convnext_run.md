# next_session_convnext_run.md — run ConvNeXt-T on the current artifact, then reprint Ch8

**Opened 2026-09-04.** Temp handoff. Same shape as the EfficientNet-B0 session that just closed
(`jax/runs/enet_b0_imagenet_bf16_350ep/`, blueprint Ch7, commits `b34c431f`/`a558cfbe`/`458cd158`):
run the JAX trainer to completion on the *current* emitted artifact, rescore over the full 50,000,
then reprint the chapter from that run and delete its `\imagenettimmnote`.

▶ State at open: `main` = `458cd158`, in sync with origin. GPUs idle. Nothing running.

---

## ▶ LAUNCHED 2026-09-04 17:51:20 UTC (12:51 PM Central) — RUNNING

Supervisor `jax/scripts/supervise_convnext_t_300ep_3060.sh` (new, forked from the B0 3060 one).
Pre-flight all green: `regen_jax_generated.sh box` ✅ 78 artifacts in sync · params **28,589,128**
· arch line carries `GAP → LN(768) → 768→1000` · `cnxInit` `*0.02` present in `init_params`
· 4 GPUs at 1 MiB before launch · 177 G free.

Two deliberate deviations from §1a/§2:
* **`MASTER` is persistent**, at `${CKPT_BASE}_master.log`, not `/tmp` — §2.5's power-cut lesson
  applied rather than just noted. Only the per-attempt scratch `RUNLOG` stays in `/tmp`.
* **Every `.bin` is kept** (`LEAN_MLIR_KEEP_BIN` unset), matching B0: 34.3 G of 177 G, and it
  leaves every epoch rescorable.

### ⭐ §1b's WARNING DID NOT APPLY — the ConvNeXt anchor was already right

Marginal ms/step, differenced off the trainer's cumulative average:

| window | 200→600 | 1000→2000 | 1500→3000 | 2000→3000 | 2500→3000 |
|---|---|---|---|---|---|
| ms/step | 179.0 | 180.0 | 180.0 | 180.0 | 179.0 |

**Flat** — so the 200→600 window, the one §1b distrusts, is already correct here (anchor 179.5).
Confirmed two independent ways off wall-clock: epoch 2 (first clean epoch; epoch 1 carries
~102 s of autotune) printed **900.1 s train + 17.7 s val = 917.8 s**, and consecutive
`_e{N}.bin` mtimes are **919 s** apart ⇒ checkpoint write ≈1.2 s.

    steady epoch 919 s = 15.32 min/ep  ⇒  76.6 h / 3.19 d for 300 epochs
    ETA Mon 2026-09-07 22:28 UTC = Mon 2026-09-07 5:28 PM Central

vs the anchor's 74.9 h train-only: **+2.4%**, essentially all per-epoch validation (1.48 h).
▶ §5's "the rest of the table is likely 15–20% optimistic" is therefore **too broad**. Differencing
cancels a *one-time* compile constant exactly; it only fails while the net is still shedding compile
inside the window, which is what ENet was doing (cumulative avg still 173 ms at step 600, 105 by
step 5000). ConvNeXt finishes autotune before step 200. **Test the rows, don't assume the bias:
a flat marginal across several windows is the evidence.** Anchor memory updated.

Health at epoch 2: 4× 99% util, 54–65 °C (trip is 80), ~155 W, 9.2–9.4 GiB of 12 GiB.
Loss 7.09 → 6.86; still inside the 20-epoch warmup.

---

## 0. WHY THIS IS A RE-RUN AND NOT A RE-SCORE

Ch8 currently reports **81.10% / 95.37%** from a 300-epoch run on 4× 4060 Ti. **Two changes since
have moved the network**, so that checkpoint is a different architecture from what the emitter
builds today — the same C6-class hazard that made B0's old number non-comparable:

| # | change | commit | effect |
|---|---|---|---|
| 1 | **`cnxInit`** — ConvNeXt paper init, `trunc_normal(0.02)` on every conv/dense | `a8bda98b` 2026-08-27 | *"the stem was 5.9× too wide"* — an init-scale change on every weight |
| 2 | **head LayerNorm added** before the classifier | `4ee94c01` 2026-08-30 | architecture line went `GAP → 768→1000` ⇒ `GAP → LN(768) → 768→1000`; params **28,587,592 ⇒ 28,589,128** |

⭐ #2 is the ConvNeXt analogue of B0's swish stem/head: a small parameter delta at a place the whole
signal passes. ConvNeXt's paper has the final LN; the phase-2 spec did not.
▶ There is an Imagenette check of the head-LN at `runs/2026-08-30-convnext-imagenette-headln`.

---

## 1. THE RUN

Artifact `jax/.lake/build/generated_convnext_tiny_imagenet_full.py`, already in sync
(`scripts/regen_jax_generated.sh box` was green at close of the B0 session — **re-check it first**).

    EPOCHS 300 · batch 256 (4×64) · bf16 + bf16Conv · lr 2.5e-4 (4e-3@4096 scaled)
    params 28,589,128 · steps_per_epoch 5004 · AdamW + cosine + RandAugment/mixup/cutmix

### 1a. Fork the supervisor — the ares one will not run here

`jax/scripts/supervise_convnext_t_300ep_4gpu_duty.sh` is hardcoded to ares and fails instantly on
this box. Copy `jax/scripts/supervise_enet_b0_350ep_3060.sh` (written last session, works) and change:

* `DEVS=0,1,2,3` — ares' `0,2,3,4` names a card this box does not have
* `PY_BIN=/home/skoonce/.venv-cuda/bin/python3` — **`../.venv/bin/python` DOES NOT EXIST HERE**
* `PY=.lake/build/generated_convnext_tiny_imagenet_full.py`
* `CKPT_BASE=/home/skoonce/convnext_t300_3060/convnext_tiny_imagenet`
* keep the temperature-driven rest (80 °C trip / 62 °C resume, taken at an epoch boundary).
  B0 never tripped it at 51–65 °C; ConvNeXt is heavier, so this one might actually fire.

Launch detached, exactly as B0 was:

    setsid nohup bash jax/scripts/supervise_convnext_t_300ep_3060.sh \
      > /tmp/convnext_t_300ep_3060_sup.out 2>&1 < /dev/null & disown

### 1b. Cost — budget MORE than the stored anchor says

The throughput memory says ConvNeXt-T **179.5 ms/step ⇒ 14.97 min/ep ⇒ 74.9 h**. ⛔ That row was
differenced over steps **200→600**, which is the method that made B0 read 39.9 h against a real
**46.6 h** (+17%). Assume the same and **budget ~85–90 h (3.5–3.8 days)**, and measure it properly:

* difference the cumulative average past step 1500, not 600
* better, take it from the trainer's own `[Epoch N] … [Xs train, Ys val]` line once steady
* record the real number and correct the anchor, as was done for B0

---

## 2. THE CHECKLIST THAT MADE B0 GO SMOOTHLY

1. `scripts/regen_jax_generated.sh box` **must be green before launching.** `.lake/build` is
   gitignored and goes stale silently; benchmarking or training a stale graph is the MNv2 trap.
2. Confirm all 4 GPUs read **0 MiB** in `nvidia-smi` — a stale process fakes OOM/NCCL errors.
3. `LEAN_MLIR_CKPT_EVERY=1` so a stop costs at most one epoch. Resume is bit-for-bit: the state
   carries params, opt state, EMA, and step; the LR and the drop-path RNG are pure functions of the
   step. **Verify it after any resume** by checking the first post-resume epoch's accuracy against
   the last pre-resume one (B0: 75.07 vs 75.07).
4. Watch disk: 300 × ~114 MB `.bin` ≈ 34 GB (ConvNeXt is 5.4× B0's parameter count). Check free
   space first, or set `LEAN_MLIR_KEEP_BIN`.
5. ⚠ `/tmp` does not survive a reboot — the B0 supervisor's master log was lost to a power cut.
   The persistent log under `CKPT_BASE` is what saved the record; keep using it.

---

## 3. SCORING

`jax/scripts/eval_convnext_full50k.py` **is correct as written** — verified last session. ConvNeXt
defines `forward(params, x, drop_key=None)` (LayerNorm net, no BN to thread) and its `.bin` **is**
`ema_params`, so params-only loading with a 2-arg forward is right. Do **not** "fix" it to match
the B0/R34 pattern.

    CKPT=/home/skoonce/convnext_t300_3060/convnext_tiny_imagenet.bin \
      /home/skoonce/.venv-cuda/bin/python3 jax/scripts/eval_convnext_full50k.py

⚠ Point `GEN` at the `_full` artifact that trained the checkpoint.
▶ Worth also scoring the raw (non-EMA) arm: on B0 the EMA was worth **+0.82 points**, and that
number turned out to be the most quotable thing in the chapter.

---

## 4. THE CHAPTER (Ch8, `\chapter{ConvNeXt}` ≈ line 8349)

Mirror what Ch7 got. Concretely:

* **Delete `\imagenettimmnote`** (line ~8991). The convention comment at `content.tex:35` is
  explicit: a re-run section carries nothing.
* **Results table** (~line 9106) — one row, this run's box/epochs/min-per-epoch/wall-clock/top-1/top-5.
  Ch7 ended with a single row; drop the 4060 Ti row rather than keeping both.
* **Both curve data lines** (~9157 top-1, ~9161 top-5) — regenerate from the run's own log, every
  epoch, all scored over 50,000.
* **Caption + the `\textbf{Compute budget.}` paragraph** (~9082–9112) — the 177 ms/step, 14.9 min/ep
  and ~75/80 hr figures are all the old box.
* **Verify the printed `NetSpec` listing against `jax/MainConvNeXtImagenet.lean`.** Ch7's listing was
  missing `convBnAct := .swish` and so contradicted its own prose. Expect the **head LN** to be the
  analogous omission here — check the layer list and the parameter count (28,589,128).
* Rebuild: `cd blueprint/src && latexmk -pdf -interaction=nonstopmode print.tex`, expect 0 errors.

⛔ **Brett's editorial line, learned the hard way in Ch7:** describe *what the code does*, not the
debugging history. No "an earlier version of the codegen…", no lab-notebook notes, no power-outage
narration, no comparisons to the prior run, no mention of the short/validation variant. Four titled
paragraphs stating behaviour beat four bug stories.

---

## 5. OPEN, NOT BLOCKING

* **The verified-path resume is lossy** and probably worse than its comment claims:
  `LeanMlir/Train.lean:688-689` zeroes the optimizer moments while `globalStep` resumes mature, so
  the render's bias correction divides fresh zeros by `1−β^t` at large `t`. EMA restarts too
  (`Train.lean:810`; nothing ever reads `_ema_params.bin`). **A 200-step loss trace across a
  deliberate resume settles it** — cheap, and the GPUs are free between runs.
* **The rest of the JAX-codegen throughput table is likely 15–20% optimistic** — every row was
  differenced 200→600. Only ENet's row has been corrected.
* `[TODO: run convnext-imagenet-verified.]` (line ~9246) stays true: no phase-4 ImageNet result
  exists for ConvNeXt either.
