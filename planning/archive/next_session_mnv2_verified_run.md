# next_session_mnv2_verified_run.md — run `mobilenetv2-imagenet-verified`, then land §6's phase-4 half

**Opened 2026-09-10.** Temp handoff, the same shape as `next_session_vit_verified_run.md` — the
session that just closed, which ran the verified ViT to **72.350 / 91.216 in 48 h 00 m, one
attempt**, and wrote it into §9.6. One job here: run the **phase-4 verified/PJRT MobileNetV2** to
completion, then write it into §6 the way §9.6 was just written.

▶ **§9.6 is now the worked template for a pair.** Read it before editing §6: side-by-side table,
convergence table, both curves overlaid on one plot, and an explicit statement of what *not* to
compare. Copy that shape.

---

## 0. ⛔⛔ THE TRAP — THE CONF NAMES THE **f32** ARM, AND ITS OWN HEADER SAYS NOT TO RUN IT

`scripts/jobs/mnv2-default-4gpu.conf` sets:

    LEAN_MLIR_VARIANT=rmsdp64          # ← f32
    CKPT_EPOCH_FILE=".lake/build/mobilenetv2in_rmsdp64_ckpt_xla.bin.epoch"

while the same file's header says, twenty lines up:

    ⚠ LEAN_MLIR_VARIANT stays f32 as the DEFAULT; the run itself is meant to use the bf16 twin.
    Flip it to `rmsdp64bf16` when launching — worth ~25 h

**Launching this job as it stands runs the wrong arm**, ~25 hours slower, and produces an artifact
that is not the one §6 wants. `verified_mlir/mobilenetv2in_rmsdp64bf16_train_step.mlir` is
committed and is the arm to run. This is the ViT `adamdp`-vs-`emadp` trap in a different costume,
and ViT's conf caught it only because someone had written a `PRECHECK` gate that asserted the
exact variant string. **This conf has no such gate.**

⚠ **And `CKPT_EPOCH_FILE` is keyed to the variant name.** Flipping to `rmsdp64bf16` changes the
checkpoint path, so the conf's `epoch_now()` would track a file the run never writes — the
supervisor would think it is permanently at epoch 0. Fix the conf properly (variant + ckpt path +
a precheck gate asserting both), don't override with an env var at launch.

⛔ **AND THE PRECHECK IS WORSE THAN ABSENT — IT VALIDATES THE WRONG ARM.** `precheck_mnv2`
hardcodes `[ -f verified_mlir/mobilenetv2in_rmsdp64_train_step.mlir ]`, the **f32** render. Flip
the variant to bf16 and the gate still passes, having checked an artifact the run does not use.
It asserts no variant string at all. Compare ViT's, which asserts the string, the operand count,
the EMA marker, the wd-exclusion marker and the clip signature.

### ✅ DECIDED (2026-09-10, Brett): run the **bf16** arm, `rmsdp64bf16`

Not just for the ~25 h. It is the same argument `r50-2018-bf16-4gpu.conf` already makes for its
own net, and the one §9.6 makes: *the JAX reference this run sits beside is itself bf16, so an
f32 verified run would differ from it in PRECISION as well as in lowerer — and the comparison is
about the lowerer.* bf16 makes it apples to apples.

⇒ **Edit the conf**: `LEAN_MLIR_VARIANT=rmsdp64bf16`, `CKPT_EPOCH_FILE` to the matching
`mobilenetv2in_rmsdp64bf16_ckpt_xla.bin.epoch`, and a real `PRECHECK` asserting the variant string
and the bf16 render. ~20 minutes, and it makes the job launchable by anyone.

---

## 0b. ⛔⛔ SECOND BLOCKER — THIS CONF CANNOT START ON THIS BOX AT ALL

    PJRT_PLUGIN=".venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so"

**That file does not exist here.** This box has only `xla_cuda13` (driver 610.57.04 / CUDA 13.3);
`xla_cuda12` is ares'. And `SHIM_PYTHON` is **absent** from the conf, which ViT's header calls
required on this box. Verified 2026-09-10: the path is missing, and the only plugin directory
present is `xla_cuda13/`.

⚠ **14 of the job confs carry the same stale line** — every ImageNet job except
`vit-default-emabf16-4gpu` (written for this box) and `r50-2018-bf16-4gpu`.

⭐ **The fix already exists, do not invent one.** Upstream `f3d89574` taught
`r50-2018-bf16-4gpu.conf` to pick its box by what is on disk — ares' `.venv` + `xla_cuda12` with
no `SHIM_PYTHON`, or the 3060 box's `/home/skoonce/.venv-cuda` + `xla_cuda13` with `SHIM_PYTHON`
set. Copy that block verbatim. Its header also notes the underlying cause: `ffi/pjrt_ffi.c`'s
default plugin search list knows `xla_cuda12` and `xla_rocm7` but not `xla_cuda13`, and fixing
that would delete the `PJRT_PLUGIN` line from every conf.

▶ Whoever fixes this for MNv2 should consider doing the other twelve in the same pass; it is one
copied block per file and it is the difference between a job that runs and a job that dies at
step 0 with a plugin-load error.

---

## 1. THE COST — AND ⭐ ONE THING CHANGED UNDER THIS NET THIS WEEK

|  | ms/step | 350 ep |
|---|---|---|
| conf header, `rmsdp64bf16` @ w8 (ares) | 136 | ~70 h |
| §6's printed table, bf16 arm | 140 | ~72 h |
| bare graph, 4 replicas, all-reduce in | **68.4** | — |

⭐⭐ **MNv2 IS PRODUCER-BOUND AND THE CONF SAYS SO OUT LOUD**: *"~68 of the 136 ms is no longer the
graph."* Half the step is the shim. That is the same condition that made the ViT run
1.8× slower than its own compute floor.

⭐⭐⭐ **`SHIM_DETERMINISM` NOW DEFAULTS OFF** (commit `4a0a2781`, 2026-09-10). On ViT that single
knob took the 300-epoch job from **69 h to 48 h**. It has **never been measured on MNv2**, and
MNv2 may not behave like ViT at all — this conf is the one that records

    ▶ THIS IS THE NET THAT DOES **NOT** WANT MORE PRODUCERS

with `SHIM_WORKERS=1` at 201 ms/step against `=4`'s 208. That is the *opposite* signature from
ViT's. **Producer-bound and wants-few-producers are not contradictory** — it means the producers
are expensive per-item rather than insufficient in number, which is exactly what determinism-off
addresses. But it has to be measured, not assumed.

▶ **Sweep on THIS box before committing 70 hours** — the script exists and takes ~30 min:

    WORKERS=1 ARMS="fed synth" PRECS=bf16 NETS=mnv2 scripts/bf16_probe_3060.sh runs/mnv2-sweep.tsv
    # then WORKERS=2, 4, 8 — fed arm only after the first synth run

⚠⚠ **RANK ON `MEAN`, NOT `med`.** The ViT session's central finding: under determinism the median
sits within ~5 ms of the compute floor at *every* worker count while p90 is catastrophic (1906 ms
at w4), so a median-ranked sweep is blind to the thing that sets wall clock. The 2026-08-31 fleet
sweep that put `SHIM_WORKERS=4` in this conf ranked on the median **and was run on ares (32 cores)**.
Treat it as unmeasured here.

⚠ The conf's own numbers are 40-step probes. §9.7 was annotated this week to say why that matters:
40 steps drain a `SHIM PREFETCH` queue the producers filled during compile, so it times the graph
with the data feed all but removed — i.e. **those are compute floors, not throughputs.** For ViT
the floor was 153 and the fed rate 199. Expect the same gap here and budget from the fed number.

---

## 2. THE PAIR — AND THE ONE CONFOUND ViT DIDN'T HAVE

**Reference (phase-2, JAX):** `71.90 / 90.41`, 350 epochs, **38.4 h on this same 4× 3060 box**,
one attempt, no restarts, no thermal pauses; best epoch 331 at 71.99. Committed `85daffbc`,
§6.5. The paper's figure is 72.0 and the Wilson 95% interval is [71.50, 72.29].

Recipe: global batch **256** (64 per replica × 4), RMSProp + exponential decay, LR 0.045, 5-epoch
warmup, ×0.98/epoch, constants shared with the Imagenette peer via `mnv2RmsSchedule` /
`mnv2RmsHyper`.

⚠⚠ **MNv2 HAS BATCHNORM. ViT DID NOT.** §9.6 got to say "what is left between the two columns is
the lowerer" precisely because ViT has no BN and both sides were bf16. That sentence **must not be
copied into §6.** The verified path runs four replicas of 64 with no collective touching the batch
statistics, so its BN statistic group is **64**; the JAX reference under a `NamedSharding` mesh
reduces globally, so its group is **256**. §5.7 measured that exact fourfold difference moving
ResNet-34's answer by 0.02 points. Expect a small offset and *say which confound it is*.

⚠ **Check the reference's provenance before pairing against it.** The number in §6.5 was re-run
2026-08-30 *after* the stale-emit fix; the earlier `71.44` was trained on a Jun-22 artifact whose
source had said `labelSmoothing 0.0` for three weeks. `scripts/regen_jax_generated.sh check` is
green as of this session — keep it green, and confirm the 71.90 run postdates the fix.

---

## 3. PRE-FLIGHT (the ViT checklist, which worked — one attempt, zero resumes)

1. ⚠⚠ `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` **must be EMPTY.**
   A stale GPU process fakes OOM and NCCL errors on this box.
2. `scripts/regen_jax_generated.sh box` must be ✅ — it is, as of 2026-09-10. This is the guard
   that would have caught the MNv2 stale-emit bug that cost 45 h of GPU.
3. **Verify the variant from the RUN'S OWN OUTPUT**, never the conf. ViT's driver announces its
   init, EMA decay, drop sites, replica count and steps/epoch; read them all before walking away.
4. Disk: the checkpoint is one overwritten file, so this is cheap — but check anyway.
5. ⛔ **THE RESUME PATH IS STILL LOSSY AND WAS NEVER FIXED.** `LeanMlir/Train.lean` zeroes
   `adamM`/`adamV` (689-690) against a mature `globalStep` (806) and restarts the EMA shadow at
   the weights (810). ViT got away with it by never restarting; do not count on that twice. If
   `master.log` shows more than one `attempt`, the result carries a caveat.
   ▶ The fix — persist `m`, `v`, shadow, as the JAX trainer's `save_train_state` already does — is
   still unwritten and is the single highest-value piece of infrastructure work here.
6. **Launch via `systemd-run --user`**, not a bare `&`. The agent harness SIGKILLs background
   processes under memory pressure and killed a supervisor mid-run this session; `Linger=yes` is
   already set on this account, so a systemd user unit survives both that and logout. Put `RUNDIR`
   in-repo — the default is `/tmp/supervise_<job>` and a power cut has eaten a master log before.
   ⚠ Identify the supervisor with `pgrep -f supervise.sh`, **not** the pid from `$!` — `setsid`
   forks, so `$!` is a transient wrapper. That misread cost this session an unnecessary watchdog.

---

## 4. THE CHAPTER (§6) — WHAT TO UPDATE

* The phase-4 subsection ends on **`[TODO: run mobilenetv2-imagenet-verified.]`** (~line 7092) and
  says outright *"No phase-4 ImageNet result exists for this network."* That is what this fills.
* **The throughput table** (~7075) has `166 → 140`, `~84 → ~72 h`, and both `Val top-1` cells TBD.
  Replace with the measured fed rate, and give the 40-step figures the same compute-floor
  annotation §9.7 just got.
* ⛔ **AN INDEPENDENT DEFECT, ALREADY CONFIRMED, WORTH FIXING WHILE YOU ARE IN THERE.** Line ~7233
  still compares MNv4 against **"MobileNetV2's paper-faithful $71.44\%$"** with **"+4.04 points"**
  and **"45 wall-clock hours"** — all three superseded by `85daffbc`, which moved §6.5 to
  **71.90 / 90.41 in 38.4 h**. Same chapter, stale cross-reference. It should read **+3.58** and
  **38.4**. (75.48 − 71.90 = 3.58.)
* Editorial line, unchanged: describe what the code does, not the debugging history. No resume
  narration, no thermal paragraphs. Brett cut both from Ch8 and cut two paragraphs from §9.6.

---

## 5. ⛔⛔ WHAT COST THIS SESSION THE MOST — READ BEFORE TOUCHING `content.tex`

**`blueprint/src/content.tex` has systematic duplicate anchors across the seven net chapters.**
A global string replace on one of them deleted **3,939 lines and three whole chapters**, silently,
and only a line-count check caught it. Measured occurrences:

| anchor | count | where |
|---|---|---|
| `\medskip\noindent\textbf{What it has not done is run.}` | **3** | mnv2, enet, convnext |
| `\subsection*{Phase 4: the verified trainer}` | **4** | same four |
| `Trainer steps, all measured on real ImageNet over $40$` | **2** | convnext, vit |
| `but no committed artifact combines it with` | **1** | convnext |

⚠ Two of those counts were higher before 2026-09-10 — ViT's copies were fixed this session, which
is *why* the trap fired. **MNv2's is the FIRST of each that remains.** A `str.index()` anchored on any of them lands on MobileNetV2
by default — which is what made the failure so easy to hit while editing ViT, and means editing
MNv2 will *appear* to work while a later chapter is the one that actually needed the change.

▶ **The rule: every edit to this file asserts `count == 1` on its anchor, or operates on an
explicit line range verified to sit inside the target chapter.** Take a backup first
(`cp content.tex /tmp/...`), and after every edit check `grep -c '^\\chapter{'` (must be 13) and
that `\begin{center}` / `\end{center}` still balance. An unbalanced environment fails the build
with a message that points at `\end{document}`, hundreds of lines from the real edit.

Other things that transfer:

* **Verify a printed listing field-by-field against its source.** §9.6's `VerifiedConfig` was
  missing `vitInit := true` — the one axis the section itself credits for reaching the paper's
  number. Ch8 had the identical defect with `cnxInit`. Assume §6's listing has one too.
* **The train loss is not a fidelity metric.** ViT's ran 0.47 apart where top-1 agreed to 0.04,
  because label-smoothed CE sums `log p` over all classes and is hypersensitive to the
  small-probability tail. Audited to the render's loss subgraph; the formula is the reference's
  term for term. State ties on **val top-1/top-5**.
* **Two independent reads of any ETA.** Marginal step rate × steps/epoch, *and* consecutive
  checkpoint mtimes. They agreed to 199 vs 203 ms/step on ViT; the 48.1 h estimate landed within
  33 seconds of the actual 48 h 00 m 33 s.
* **Archive the run in-repo when it lands**: `RESULTS.md`, the full per-epoch log, and a curve CSV
  — see `runs/2026-09-08-vit-verified-300ep-det0/` for the shape. `/home/skoonce/vit/` has no log
  at all and its 300 `.bin` files are the only record of that curve; do not create a second one.
