# next_session_vit_verified_run.md — run `vit-imagenet-verified`, then land §9.6's phase-4 half

**Opened 2026-09-07.** Temp handoff, same shape as `next_session_convnext_run.md` (the session that
just closed: ConvNeXt-T 300ep → **81.53 / 95.50**, Ch8 reprinted). One job: run the **phase-4
verified/PJRT ViT** to completion, then write it into §9.6 the way Ch8 was just written.

▶ **The standing plan is `planning/vit_verified_run.md`** (committed `10ae7478`) — the config body,
the five driver announcements to verify, the trap list and the LaTeX edits are all there and all
still good. **This doc is only the delta**: what changed since it was written, what the ConvNeXt
session established that transfers, and the one thing that is genuinely blocking.

---

## 0. ⛔ THE BLOCKER — RESUME IS LOSSY, AND A CHECKPOINT IS ALREADY ON DISK

`.lake/build/vitin_emadp128x4wxclipdropbf16_ckpt_xla.bin` exists (91 MB, 2026-08-29) and its
`.epoch` sidecar reads **`1`**. `scripts/jobs/vit-default-emabf16-4gpu.conf` says so out loud:

    ⚠ $CKPT_EPOCH_FILE exists — this run will RESUME from it, not start fresh.

**So the very first action of the next session hits the resume path.** And that path drops state:

| `LeanMlir/Train.lean` | what it does |
|---|---|
| **689–690** | `adamM`/`adamV ← F32.const … 0.0` — **unconditional**, resume or not |
| **806** | `globalStep := startEpoch * bpE` — resumes **mature** |
| **810** | `emaParams := params` — the EMA shadow **restarts at the weights** |

⭐ **The moments and the step counter disagree, and Adam's bias correction is what breaks.** With
fresh `m`,`v` at a mature `t`, `(1-β^t) ≈ 1`, so the correction that is supposed to un-bias a cold
start does nothing. One step after a resume:

    mc = 0.1·g / (1-0.9^t)   ≈ 0.1·g        vc = 0.001·g² / (1-0.999^t) ≈ 0.001·g²
    update = mc/(√vc + ε)    ≈ 0.1g / 0.0316|g|  ≈ **3.16 · sign(g)**

against the **1.0** a correct cold start gives. That is a ~3.2× oversized step decaying over the
following few hundred steps, plus the total loss of momentum history, plus an EMA shadow that
restarts — and on a net whose reported number **is** the shadow (`EVAL AND CHECKPOINT SCORE THE
SHADOW`), a restarted EMA is not a cosmetic problem.

### The decision to make first

* **(a) Delete the epoch-1 checkpoint and run fresh.** Costs one epoch. Removes the issue *at
  launch* but not *mid-run* — a ~78 h run on a box that reboots is very likely to resume at least
  once, and every resume takes the same hit.
* **(b) Fix the resume path.** Persist `m`,`v` and the EMA shadow alongside the weights, the way
  the phase-2 JAX trainer's `save_train_state` already does (`(params, opt_state, ema_params)` +
  step, one `.npz`). This is the real answer and it is what makes a long verified run safe.

▶ **Settle it with the 200-step loss trace across a deliberate resume** — carried over unrun from
`next_session_convnext_run.md` §5. Cheap, and it converts "probably worse than its comment claims"
into a measurement. Do it BEFORE committing 78 hours.

---

## 1. THE COST — BUDGET ~78 h, AND §9.6'S OWN TABLE IS WRONG

|  | ms/step | min/epoch | 300 ep |
|---|---|---|---|
| §9.6's phase-4 table claims | 159 | 6.6 | ~36.3 h |
| **measured 2026-08-29 on this box** | **375** | **15.6** | **~78 h** |

⛔ The 159 came from `scripts/bf16_probe_4gpu.sh`, which sets **`LEAN_MLIR_MAX_STEPS=40`**. With
`SHIM PREFETCH` depth 8 over 8 producers, 40 steps drain a queue the producers filled during
compile and the ~90 s val drain — it reads **burst rate, not production rate**. That script is also
hardcoded to ares (`CUDA_VISIBLE_DEVICES=0,2,3,4`, `xla_cuda12`), so the "4× 3060" row was never
produced on this box at all. ⚠ Neither EMA nor precision explains the gap; the measured 375 is the
**bf16** render, and bf16 is the *faster* direction here.

⚠ **The verified path is shim-data-starved, not GPU-bound.** 8 producers drew **1230% CPU** of this
box's 24 threads while the trainer took 85% — ~11 threads idle, GPUs 70–95%, never pegged. The
verified path pushes **602 KB/img of fp32 through a subprocess pipe, 308 MB per step** at global
512. ▶ **Sweep `SHIM_WORKERS` 10/12/14 before the long run** — the conf's `SHIM_WORKERS=8` carries
"⚠ Do NOT raise to 16: measured slower (710 vs 665)", which is an **ares** number (32 cores)
imported into a Xeon w5-2455X (12c/24t) conf. That is exactly the mistake the same file calls out
one field over about `DEVS` and `PJRT_PLUGIN`.

### ⭐ Measure it the ConvNeXt way, not with a step probe

The ConvNeXt session settled how to do this and the method is the transferable part:

* Differencing two cumulative averages cancels a **one-time** compile constant *exactly*. It only
  fails while the net is still shedding compile inside the window. **So test it: is the cumulative
  average still FALLING across your window?** ConvNeXt was flat (179.0 at 200→600 vs 180.0 at
  1500→3000) and its anchor was right; ENet was still falling at step 600 and its anchor was 17% low.
  ⇒ Do not assume the bias and do not assume its absence — a flat marginal across several windows
  is the evidence.
* Better still, take **epoch-to-epoch wall clock two independent ways** and require agreement:
  consecutive checkpoint mtimes, and the driver's own per-epoch line. ConvNeXt: 919 s vs 917.8 s,
  the 1.2 s gap being the checkpoint write. That is what makes an ETA trustworthy at 4% in.

---

## 2. WHAT THE CONVNEXT SESSION ESTABLISHED THAT TRANSFERS

⚠ **The mechanism differs** — ViT runs under `scripts/supervise.sh` + a job conf, not a bespoke
`jax/scripts/supervise_*.sh`. So what transfers is the practice, not the script.

1. **Keep the master log OFF `/tmp`.** B0's was lost to a power cut mid-run. The ConvNeXt supervisor
   puts both master and full logs under `CKPT_BASE`. Check where `supervise.sh` writes and make the
   persistent copy exist before launching, not after.
2. **Archive the run in-repo when it lands**: `jax/runs/<run>/` with `RESULTS.md`, the curve `.tex`,
   and the **full per-epoch log**. ⛔ `/home/skoonce/vit/` (the phase-2 300ep ViT) has **no log at
   all** — its 300 `.bin` files are the only surviving record of that curve, which is why they can
   never be pruned. Do not create a second one of those.
3. ⭐⭐ **Audit the LaTeX against the ARTIFACT, not just the numbers.** Swapping figures in Ch8 would
   have left four defects standing: a printed `NetSpec` missing the head LayerNorm (while the prose
   two pages up already gave the post-LN parameter count), a `TrainConfig` missing `cnxInit`, a
   residual attributed to "test-crop" when the artifact already evaluates at timm's 0.875, and a
   sentence claiming the resampler and validation protocol "still differ" when the artifact has
   `antialias=True` + BICUBIC on both resizes. **Diff every printed field against the source, then
   every claim against the emitted trainer.**
4. **Do the no-GPU chapter work while the run grinds.** The listing audit costs nothing and is the
   part most likely to contain a bug.

---

## 3. PRE-FLIGHT (the ConvNeXt checklist, ported)

1. ⚠⚠ **`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` must be EMPTY.**
   A stale GPU process fakes OOM and NCCL errors on this box. Do not trust any CUDA failure before
   checking this.
2. ⛔ **Assert `LEAN_MLIR_VARIANT=emadp128x4wxclipdropbf16`.** `adamdp128x4wxclipdropbf16` is a
   committed artifact one character-class away with **no EMA** — it would train fine, report a
   number, and not be the pair. The conf already gates this; confirm the gate fires.
3. **Verify from the run's own output, never the conf** — all five driver announcements, and
   especially `decay min(0.999960, …)`. `trainAdamSched`'s `emaDecay` default is 0.9999; the
   reference's is 0.99996, and a shadow averaging 4× too fast still trains and still reports.
4. Disk: check free space against 300 epochs of checkpoints before launching.
5. The artifact-staleness question for the verified path is `regen_verified_mlir.sh`, not
   `regen_jax_generated.sh` — and ⛔ note that guard only ever pairs the forward with the **SGD**
   train step, never Adam.

---

## 4. THE CHAPTER (§9.6)

* The phase-4 subsection ends on a bare `[TODO: run vit-imagenet-verified.]` and says outright that
  no phase-4 ImageNet result exists. This fills it and gives ViT the {reference, verified} pair
  Chapter 5 already has for R50.
* **Fix the 159 ms/step row** while you are in there — see §1. It is wrong for a knowable reason.
* ⚠ `content.tex` reportedly carries a **4-way-duplicate subsection heading** in this area that once
  ate 4,100 lines. Verify the heading structure before editing.
* Editorial line, unchanged: describe what the code does, not the debugging history. No resume
  narration, no thermal/compute-budget paragraphs (Brett cut both from Ch8).

---

## 5. THE PAIR ALREADY TRACKS

Epoch-1 train loss **6.8167** (verified) vs **6.7945** (reference) at the same lr 1e-4 — two
lowerers, 0.02 apart. ⚠ The reference validates every 5 epochs, so **epoch 5** (14.07 / 32.08) is
the first comparable *accuracy* point; the verified path evals every epoch (ep 1 = 2.028 / 6.80).
Target to beat/match: the phase-2 reference landed **72.31 / 91.12** in 34.22 h.
