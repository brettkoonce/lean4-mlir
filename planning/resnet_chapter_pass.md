# resnet_chapter_pass.md — Chapter 5 to the standard Chapters 1-4 now meet

**Opened 2026-09-01**, at the end of the session that took Chapters 1-4 through a transcript,
listing and statistics pass. Chapter 5 is next and is the biggest of them.

**This is self-contained.** A session that has read nothing else can execute it.

---

## 0. What the first four chapters turned up, because it all recurs here

⭐⭐ **The dominant defect was a listing introduced as complete that was not.** Six of them across
four chapters:

| where | missing |
|---|---|
| §1.4 `linearVerified` | `blurb` |
| §1.6 `trainLinear` | used `tsFn`, `fwdFn`, `nResident` **without binding them** — did not compile as printed |
| §1.7 `linear_train_step.mlir` | showed **19 of 27** operations, described as "only `precision` dropped" |
| §2.4 `mlpVerified` | `blurb` **and** `lossSlot` |
| §3.3 `cnnVerified` | `blurb`, `lossSlot`, wrong `import` |
| §4.4 `cifar8w{,Bn}Verified` | `blurb`, both listings |

▶ **Check every listing in Chapter 5 against its source first.** It is the highest-yield hour.

⚠ Other recurring shapes: a number restated in prose disagreeing with the transcript that produced
it (§2.4's "5.4 seconds" against §2.1's 2103 ms); a claim inherited from a driver nobody chose for
that chapter (§4's cosine schedule); and a run directory cited that no longer matches the binary.

---

## 1. The concrete Chapter 5 list

### 1.1 ⛔ §5.1's transcript is format-stale

`content.tex:4695`, from `runs/2026-08-12-r34-imagenette-xla-cuda/`, shows **PJRT 0.112** and no
interval. `trainAdamSched` has appended `[95% CI lo–hi]` to every eval line since `c205cea`, so
the printed commands no longer produce the printed output.

⚠ **The epoch lines are already 78 characters** (`val_acc = … top5 = …`). Adding the interval
takes them past 100. Chapter 4 solved exactly this: wrap the CI onto an indented continuation and
widen the elision. See `content.tex` §4.1 for the shape, and say so in the lead-in.

▶ A fresh 80-epoch run is ~45 min (measured 2026-08-31: R34 Imagenette is 44.7 min at n=5 under
contention, faster alone). ⚠ `PJRT_FFI_RESIDENT=1` is OFF by default; §5.1's four printed commands
do not set it, so run it **without** residency or the timing prose will not match.

### 1.2 ⚠⚠ ResNet-34's published Imagenette number does not reproduce

`planning/imagenette_error_intervals.md` §8.2: the book prints **89.71**; five seeds give
**90.08 ± 0.25**, a 95% seed interval of [89.77, 90.39] that **excludes** it. Every other published
net brackets its book value.

⭐ It pairs with a timing miss in the same direction — 44.7 min measured against a published 1.5 h,
while ConvNeXt, B0, MNv2 and ViT all reproduce their timings *despite* six-way contention. Two
independent misses on one net point at the published row coming from an **older code state**.

▶ **Next step is `git log` on the R34 Imagenette path, not more seeds.** Settle this before
rewriting §5.4's numbers, because the answer decides whether they are stale or wrong.

### 1.2-bis ✅ §1.2 IS SOLVED — it is `f4e4172`, and it is "stale", not "wrong"

Do not spend the `git log` hour §1.2 asks for; it was spent 2026-09-01. The answer:

* The published **89.71** came from `runs/2026-08-12-r34-imagenette-xla-cuda/`, which loaded
  `verified_mlir/resnet34_adam_train_step.mlir` — the **same** artifact the n=5 seeds use. So it is
  not a two-renderer mixup. (⚠ Worth knowing anyway: R34 has TWO renderers with DIFFERENT BN
  semantics — `resnet34_train_step.mlir` is per-example, 289 × `reduce[2,3]`, and
  `resnet34_adam_train_step.mlir` is batch, 468 × `reduce[0,2,3]`. The MobileNetV2 two-worlds
  divergence, live on R34. It is not the cause here, but it is a live trap.)
* That artifact **changed** on 2026-08-29 in **`f4e4172`** *"perf(render): pointwise ops at their
  4-D shape"*. Diff against the pre-publication blob: **+261 reshapes (585 → 846) and NOTHING
  else** — every reduction axis identical, every other op count identical.
* Same seed, before and after: **89.707 → 90.038**, a **+0.33** shift. Seed variation is ±0.25, so
  the graph change is detectable above it.

⭐ **So the published row is a single run on a superseded graph.** Reshapes are numerically inert
in exact arithmetic; what moved is XLA's fusion and therefore the rounding, compounded over 80
epochs. The net is the same net. ▶ Reprint from the current-graph measurement; do not hunt a bug.

⚠ **This is not R34-only.** `f4e4172` touched **124 artifacts**, and ALL SIX `runs/2026-08-12-*`
transcripts the book prints predate it — r34, convnext, enet, mnv2, vit and cifar8w. Every
Imagenette chapter transcript is graph-stale, not just format-stale. Chapters 6-9 inherit this.

### 1.2-ter ⚠ Before tagging: which numbers predate `f4e4172`?

The release intends to declare MNIST / CIFAR / Imagenette / ImageNet-with-ResNet **stable**.

| set | graph state | action |
|---|---|---|
| MNIST, CIFAR | `runs/2026-09-01-*`, **post**-fix | ✅ nothing to do |
| Imagenette Ch5 | 2026-08-12 transcript is **pre**-fix | the fresh §5.1 run (§1.1, ~45 min) fixes format AND graph in one go — do it |
| Imagenette n=5 | `2026-08-31-*`, **post**-fix | ✅ already current; reprint §5.4 from these |
| **R50 ImageNet** (76.95 / 77.07 / 78.26 / 77.91) | **pre**-fix (runs are 2026-08-07…08-24) | ▶ see below |

▶ **Recommendation: do NOT re-run the R50 ImageNet cells.** A 100-epoch ImageNet cell is ~24 h, and
the expected shift is smaller than the number's own measurement noise: the Imagenette shift was
+0.33, while the 95% binomial interval on ImageNet's 50k val at 77% is **±0.37**. You would be
buying a change you cannot distinguish from the interval already printed beside it.

⭐⭐ **And the measurement is free anyway.** The A3-RSB run just finished on the other box. If it
used current artifacts (check: does its log's `resnet50in*_train_step.mlir` match the working tree,
or did that box pull after 2026-08-29?), then **A3-new vs the published 77.91 IS the measurement of
`f4e4172`'s ImageNet impact.** If it lands inside ±0.37, the pre-fix ImageNet rows need no re-run
and you can say so in one sentence. If it moves more than that, re-running becomes a real question.
▶ Do that comparison before deciding to spend 24 h.

### 1.3 Cosine annealing is now Chapter 5's to introduce

Chapter 4 was converted to a **constant** learning rate this session (see
`planning/bf16_batchnorm.md` §6). §4.1 now forward-references Chapter 5 for the schedule:

> *"Chapter~\ref{chap:residual} adds cosine annealing back and shows what it buys."*

⛔ **Chapter 5 does not currently pay that off.** Its transcript shows `cosine+warmup 3ep` with no
prose explaining the schedule or what it is worth. ▶ Write that. The measurement is in hand:
removing cosine cost the CIFAR BN momentum arm **0.66 points** (77.14 → 76.48 median, n=5 each) and
cost SGD and AdamW essentially nothing — so the schedule flatters the arm that already wins.

### 1.3-bis Chapter 5 must now pay off Lever 3, exactly as it pays off cosine

Chapter 4's §4.5 was rewritten 2026-09-01 (`planning/bf16_batchnorm.md`) and now ends:

> *"Thirty arm-runs, no NaN, nothing collapsed to chance. That is the result
> Chapter~\ref{chap:residual} was going to assume, and it no longer has to."*

▶ Chapter 5 should collect that: bf16 on the normalized CIFAR net costs +0.31 / −0.02 / −0.33 with
every delta inside its own fp32 row's spread, which is the licence for training the ImageNet jobs
in bf16. One or two sentences where the precision choice is first made.

⚠⚠ **And be precise about which runs are bf16, because Chapter 4 now is.** The old §4.5 said *"every
network from Chapter 5 onward is trained that way"*, which is false: the **Imagenette** runs behind
Chapters 5-10 all load **f32** artifacts (`{net}_adam_train_step.mlir`, zero bf16 in any of them);
the bf16 rows are the **full-ImageNet** jobs. §4.5 was corrected to say so. Chapter 5 is the chapter
that has to be consistent with it — and §5.8 already carries the symptom, the caveat that *"the PJRT
column is not internally uniform: its 2018 entry is bf16, its A3 entry fp32."*

⭐ Chapter 4 also now warns that the **op family** is a lever invisible to the proofs: it moves the
un-normalized CIFAR net by +2.0 / +1.7 and the normalized one by ~0.2. §1.2-bis is the same class
of finding on R34 — the rendering path moved a published number by 0.33 with no proof disturbed.
The two belong together if Chapter 5 says anything about what the certificates do not cover.

### 1.4 Phase-N vocabulary

`planning/blueprint_lowerer_pattern.md` rule 2: *"'PJRT lowerer' and 'JAX lowerer' survive;
'phase 2 / phase 3 / phase 4' date the book."* 68 references remain book-wide. Chapter 5's five are
all **structural**, not prose:

    5376  \label{sec:r34_phase4}
    5735  \S\ref{sec:r50_phase4}
    5761  \label{sec:r50_phase4}
    5786  \S\ref{sec:r34_phase4}
    6010  \S\ref{sec:r50_phase4}

▶ Rename to `sec:r34_pjrt` / `sec:r50_pjrt` **before** Chapters 6-9 are converted, because those
chapters will `\ref` into them.

### 1.5 What is already right — do not touch

* §5.7 and §5.8 are the **template** for the ImageNet-chapter pattern and are already converted.
  §5.8's grid is complete (2018: JAX 76.95 / PJRT 77.07; A3: JAX 78.26 / PJRT 77.91).
* ⚠ `blueprint_lowerer_pattern.md` is **stale on two counts** — it says the PJRT×2018 cell is "not
  run" and that the two-lowerer argument "is not currently stated anywhere in the book." `2859cd7`
  (2026-08-27) filled the cell and stated the argument in §5.8's comparison. Fix the doc or ignore it.
* The A3 **bf16/PJRT** run is finishing on another machine. It does not add a cell; it removes a
  caveat — §5.8 currently has to print *"the PJRT column is not internally uniform: its 2018 entry
  is bf16 on the four-3060 box, its A3 entry fp32."* ▶ When it lands, decide: replace the fp32 A3
  entry (per `85daffb`'s rule that superseded numbers are deleted, not annotated) or grow a
  precision dimension. Recommend the former; the table's job is lowerer-vs-lowerer.

---

## 2. Traps that cost time in the Chapters 1-4 pass

1. ⛔⛔ **Never slice `content.tex` with `s.index(...)` on a string that repeats.** This session
   deleted **2,894 lines** — all of Chapters 1-3 — because the anchor
   `[pjrt_ffi] XLA backend: PJRT 0.114` had by then been added to four chapters and matched the
   first. Recovered only because the chapters were committed. Use uniqueness-asserted replacement
   (`assert s.count(old)==1`) for every edit, and rebuild after each.
2. ⚠ **`latexmk` rc=0 is not "it worked."** Check `grep -c "Reference.*undefined" print.log` too —
   that is what caught the deletion above.
3. ⚠ **A checkpoint at the target epoch makes a re-run a silent no-op** — it resumes, prints `done`,
   emits no epoch line. All three `cifar8w_bn` untagged checkpoints sat at epoch 40. Use
   `LEAN_MLIR_CKPT_TAG`; it does not appear in the output, so the transcript is still what a clean
   checkout produces.
4. ⚠ **Render the page and look at it.** Two claims this session were contradicted by numbers
   printed two paragraphs above them, and both were caught by reading the PDF rather than the source.
5. ⚠ **Full-page renders at 150 dpi are not enough to judge a few-pixel feature.** A whisker cap was
   called "clipped" on that evidence and was not. Zoom with `pdftoppm -x -y -W -H`.

---

## 3. Suggested order

1. Listings audit (§1.5 of the four-chapter pattern) — cheap, high yield.
2. ~~`git log` on R34 Imagenette to settle §1.2~~ — ✅ DONE, see §1.2-bis: it is `f4e4172`,
   the numbers are stale rather than wrong, and §5.4 reprints from the 2026-08-31 seeds.
3. Fresh §5.1 run without residency; splice with the wrapped-CI shape.
4. Write the cosine payoff §4.1 now promises.
5. Rename the phase-N labels.
6. Fold in the A3 bf16 result when the other machine reports — and use it for §1.2-ter's
   free measurement of `f4e4172`'s ImageNet impact before deciding on any 24 h re-run.
7. Before tagging: `lake build Certs` (3923 jobs — plain `lake build` skips the corpus),
   `python3 scripts/check_render_coverage.py`, and re-elaborate the renderers to confirm every
   artifact regenerates byte-identically.
