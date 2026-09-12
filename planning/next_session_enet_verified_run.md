# next_session_enet_verified_run.md — run `efficientnet-imagenet-verified`, then land §7's phase-4 half

**Opened 2026-09-12.** Successor to `next_session_mnv2_verified_run.md`, which closed the same
day: MobileNetV2 ran to **71.912 / 90.520 in 54 h 13 m, one attempt**, and §6 got its phase-4
half. One job here: run the **phase-4 verified/PJRT EfficientNet-B0** to completion, then write
it into §7 the way §6 was just written.

▶ **§6 is now the worked template**, and it is closer to this net than §9.6 is: side-by-side
lowerer table with **BN statistic group above the rule**, convergence table, both curves on one
plot, and an explicit statement that the pair does NOT isolate the lowerer. Copy that shape.
Read §6's phase-4 half before editing §7.

---

## 0. ⛔⛔ THE SAME TRAP MNv2 HAD — THE CONF NAMES THE **f32** ARM

`scripts/jobs/enet-default-4gpu.conf` sets:

    LEAN_MLIR_VARIANT=emarmsdp64dropdo            # ← f32
    CKPT_EPOCH_FILE=".lake/build/efficientnetin_emarmsdp64dropdo_ckpt_xla.bin.epoch"

while its own header, thirty lines up, says:

    ▶ So: flip to the bf16 twin AND raise SHIM_WORKERS to 14.

`verified_mlir/efficientnetin_emarmsdp64dropdobf16_train_step.mlir` is committed and is the arm
to run. ⚠ `CKPT_EPOCH_FILE` is keyed to the variant name, so it must move with it or
`supervise.sh` tracks a file the run never writes and sits at epoch 0 forever.

⛔ **AND THE PRECHECK VALIDATES THE WRONG ARM**, exactly as MNv2's did: `precheck_enet`
hardcodes `[ -f verified_mlir/efficientnetin_emarmsdp64dropdo_train_step.mlir ]` — the **f32**
render — and asserts no variant string at all. Flip the variant and the gate still passes,
having checked an artifact the run does not use.

⛔⛔ **AND IT ACTIVELY BLOCKS ITS OWN HEADER.** The precheck asserts `SHIM_WORKERS=8`. The
header says to raise it to 14. Following the conf's own recommendation makes the conf refuse to
launch. This one is worse than MNv2's, which merely failed to check.

⇒ **Fix the conf the way `mnv2-default-4gpu.conf` was fixed** (`44426027`): variant to
`emarmsdp64dropdobf16`, `CKPT_EPOCH_FILE` to match, and a PRECHECK that asserts the variant
string, the bf16 render, ckpt/variant agreement, shim freshness, residency, the worker count
you actually chose, stray per-variant checkpoints and GPU idleness. ~20 minutes, and it makes
the job launchable by anyone.

## 0b. ⭐ THE BOX BLOCKER IS ALREADY FIXED — unlike MNv2's

`6d3cb267` made all 16 job confs box-aware. `enet-default-4gpu.conf` now picks ares' `.venv` +
`xla_cuda12` or this box's `/home/skoonce/.venv-cuda` + `xla_cuda13` + `SHIM_PYTHON` by what is
on disk. Nothing to do here; just do not re-hardcode it.

---

## 1. ⛔⛔ EVERY THROUGHPUT NUMBER IN THAT CONF IS VOID, INCLUDING `SHIM_WORKERS=14`

The header's measured table — `f32 @ w8 96.1 h`, `bf16 @ w8 97.1 h`, `bf16 @ w14 82.9 h`, floor
50.3 h — cites `runs/probe3060_prod.tsv`, **dated 2026-08-31**. Since then:

* `4a0a2781` (**2026-09-10**) flipped the tf.data determinism default OFF.
* `13d90e68` (**2026-09-11**) fixed the mimalloc RSS growth *and* found that the runtime shims in
  `jax/.lake/build/` still defaulted determinism ON, **capping every producer at ~1.7 cores**.

⇒ **The w14 recommendation is an artifact of the bug.** It was compensating for producers running
at a fifth of their throughput. With determinism off each producer is far more effective, so the
optimum should move **DOWN, not up** — upstream measured **6 best for ViT** post-fix. The header's
"at 14 they are near 21 of this box's 24 threads" is solving a problem that no longer exists.

⚠ It is also unreachable: the precheck asserts 8. Nobody has ever run this job at 14.

### The only post-fix numbers that exist (2026-09-11, `runs/2026-09-11-imagenet-probe-postfix/fed.tsv`)

| variant | prec | w | med | min | p90 | **mean** |
|---|---|---|---|---|---|---|
| `emarmsdp64dropdo` | f32 | 8 | 221 | 188 | 232 | **220** |
| `emarmsdp64dropdobf16` | bf16 | 8 | 149 | 112 | 170 | **279** |

⭐⭐ **READ THE bf16 ROW AGAIN: the mean (279) is ABOVE its own p90 (170).** That is not noise,
it is a handful of enormous stalls dragging the average past the 90th percentile. The bimodality
the old header describes — "most steps land at the compute floor and a minority stall on an empty
queue" — **SURVIVED both fixes at w8.** Median says 72 h for 350 epochs; mean says 136 h. Wall
clock follows the mean.

▶ **SWEEP BEFORE LAUNCHING. Rank on `mean`, and look at `mean` vs `p90` as a pair** — this net is
the one case in the fleet where they disagree in that direction, and a median-ranked sweep will
confidently pick an arm that takes twice as long.

    for w in 4 6 8 10 12; do WORKERS=$w ARMS=fed PRECS=bf16 NETS=enetema \
      scripts/bf16_probe_3060.sh runs/2026-09-XX-enet-sweep.tsv; done
    # then the synth arm once, for the compute floor

⚠ `enetema` is the right row — it is the production graph (EMA + drop-path + dropout), and the
plain `enet` row is the LIGHT variant no job conf trains. Same class of mistake as MNv2's
`mnv2`-vs-`mnv2rms`, which cost a re-probe.

⚠ **This is the most feed-bound net in the fleet and that is a property of its shim, not its
graph.** Its device compute is 144.5 ms — cheaper than ResNet-34's 188.3 — and its shim runs
AutoAugment + RandAugment per image on CPU, where R34 and MobileNetV2 do flip only. Expect the
verified-vs-reference wall-clock gap to be the widest in the book, and expect most of it to be
pipeline rather than lowerer.

---

## 2. THE PAIR

**Reference (phase-2, JAX):** `77.15 / 93.30`, 350 epochs, **46.6 h on this same 4× 3060 box**,
~8.0 min/epoch (§7, chapter line ~8183). Matches EfficientNet-B0's paper 77.1 / 93.3.

Recipe: RMSProp, global batch 256 (64 × 4), AutoAugment, stochastic depth, **EMA decay 0.9999**,
bf16 conv.

⚠⚠ **THE EMA SHADOWS THE BATCHNORM BUFFERS AS WELL AS THE WEIGHTS** (§7, ~line 8209). The
reference's scored weights are EMA weights. The verified variant `emarmsdp64dropdo*` carries EMA
for exactly this reason — but confirm from the run's own output that the EMA decay it announces
is `0.9999`, not a different constant. ViT's pair was one-variable only after its EMA artifact
existed; do not assume this one matches without reading it.

⚠⚠ **BN CONFOUND — AND THIS NET IS THE DENSEST BN CASE IN THE BOOK.** Audited 2026-09-12:
`allReduceMeanF` is the only `SHlo` constructor taking a replica family, every BatchNorm
constructor is `SHlo n → SHlo n`, so **the verified render normalizes over 64 per replica** while
the JAX reference under `@jit` + `NamedSharding` reduces globally over **256**. §5.7 measured that
fourfold difference as 0.02 points on ResNet-34. EfficientNet has more BN per unit of compute than
ResNet-34 does, so do not assume the same magnitude transfers.
▶ §7 must carry the `BN statistic group` row above the rule, as §5, §6 and `[TODO: global bn]`
do. See `planning/global_bn_verified.md` — and note §1 of that doc gives a way to MEASURE the
confound on the JAX side for free, which would be worth doing before this run rather than after.

---

## 3. PRE-FLIGHT

1. ⛔⛔ **REBUILD THE EXE.** `13d90e68` changed `ffi/f32_helpers.c` (`lean_mlir_read_into`) and the
   prefetch path. A stale binary runs the leak, which cost the MNv2 run ~3 h and made its last 100
   epochs 1.6× slower. `lake build efficientnet-imagenet-verified` — and check the mtime moved.
   ⚠ MNv2's exe was silently 11 days stale at launch; `lake build A` will not rebuild exe B.
2. `scripts/regen_jax_generated.sh sync` — ⚠ **`sync`, not `box`.** `13d90e68` found 68 of 78
   artifacts stale on this box, including the emitted JAX trainers, and the runtime shims were the
   half that still had determinism ON. MNv2's precheck ran `box` and passed while that was true.
3. `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` **must be EMPTY.**
4. **Verify the variant, the EMA decay, the drop sites, the replica count and steps/epoch from the
   RUN'S OWN OUTPUT**, never the conf.
5. ✅ **RESUME — CORRECTED AND FIXED 2026-09-12.** ⛔ The claim that stood here was about the WRONG
   FILE: `LeanMlir/Train.lean` is the IREE/Lean path. The verified exe runs
   `LeanMlir/VerifiedTrain.lean`, which has always checkpointed `[θ|m|v|(G)|E]` — moments and EMA
   shadow included. What it did NOT save: the BN running stats and their EMA (`ema_bn`), which a
   resume restarted at ZERO under the mature 0.9999 decay (0.9999^5004 ≈ 61% zero after one epoch,
   ~10 epochs to wash out — the weights resumed exactly, the eval off them did not); and it wrote
   the blob in place, so a crash mid-write left a truncated file the size guard refuses on every
   restart. Fixed (uncommitted): `<ckpt>.bn` companion + `writeBinAtomic` (tmp + rename), with a
   seed-from-running fallback for pre-companion checkpoints. **Tested live on this run**: SIGTERM
   right after epoch 1 → attempt 2 read back the companion hash-identical
   (`runs/2026-09-12-enet-verified-350ep/resume_test_epoch1.log`).
6. **Launch via `systemd-run --user`**, `RUNDIR` in-repo. Copy `run_mnv2_verified.sh`. The agent
   harness SIGKILLs background processes under memory pressure — it killed two helpers during the
   MNv2 run while the systemd unit survived untouched.
   ⚠ Identify the supervisor with `pgrep -f supervise.sh`, not `$!`.

---

## 4. THE CHAPTER (§7) — WHAT TO UPDATE

* The phase-4 subsection ends on **`[TODO: run efficientnet-imagenet-verified.]`** (line **8371**)
  after **"What it has not done is run."** (line **8348**).
* **The throughput table** (~8360) has `188 → 107`, `~95 → ~56 h`, and both `Val top-1` cells TBD.
  Replace with the measured fed rate. ⚠ Those are 4060 Ti 40-step figures — compute floors, not
  throughputs.
* Add the `BN statistic group` row and the paragraph that goes with it (§6's wording).
* Editorial line, enforced on §6 this session: **describe what the code does, not the run's
  operational history.** No resume narration, no thermal paragraphs, no host-memory forensics, no
  "the box it ran on". Software and theory only; the run's own story goes in its `RESULTS.md`.

## 5. ⛔⛔ `content.tex` DUPLICATE ANCHORS — THE COUNTS MOVED THIS SESSION

| anchor | count | where |
|---|---|---|
| `\medskip\noindent\textbf{What it has not done is run.}` | **2** | **enet (8348)**, convnext (9330) |
| `\subsection*{Phase 4: the verified trainer}` | **4** | unchanged |

⚠ **EfficientNet's is now the FIRST of the pair** — MobileNetV2's was consumed when §6 was
written. A `str.index()` on that anchor now lands on EfficientNet by default, which is
convenient for this job and a trap for the next one.

▶ **The rule: every edit asserts `count == 1` on its anchor, or operates on an explicit line range
verified to sit inside the target chapter** (EfficientNet is **7472–8464**). A global replace on
one of these once ate 3,939 lines and three chapters. Back up first, and after every edit check
`grep -c '^\\chapter{'` is 13 and that `\begin{center}`/`\end{center}` still balance.

Other things that transfer from the MNv2 session:

* **Two independent reads of any ETA**, and take the second from consecutive checkpoint mtimes —
  the checkpoint is one overwritten file, so it is a free clock. ⚠ And compute pace **per window**,
  not cumulatively: MNv2's cumulative average still read 546 s/epoch when the current rate was 847,
  because 246 good epochs were diluting 90 bad ones. A cumulative mean hides a step change.
* **Verify a printed listing field-by-field against its source.** §9.6's was missing `vitInit`;
  Ch8 had the same defect with `cnxInit`. Assume §7's has one.
* **The train loss is not a fidelity metric.** State ties on val top-1/top-5.
* **Archive in-repo when it lands**: `RESULTS.md`, the full per-epoch log, a curve CSV. See
  `runs/2026-09-10-mnv2-verified-350ep/` for the shape; the extractor is 30 lines and pairs the
  `Epoch N/350: loss= lr=` line with the `epoch N: test_acc =` line.
