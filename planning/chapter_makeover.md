# Chapter makeover: porting a chapter to the verified XLA path

**Who this is for.** An agent working on a CUDA box, picking up
**chapter 8 (ConvNeXt-T)** and then the rest. Chapters 1–7 are done.

▶ **START AT §4a-sexies**, the ch8 handoff. Read §4a-quinquies (the ch7 post-mortem)
first — it is where the traps ch8 inherits were found. Before touching anything, read ch5's §5.3 (`sec:verified_trainer_pattern`) and
then chapter 7 in `blueprint/src/content.tex` — §5.3 is the spec and ch7 is the most
recent worked example of applying it. §1 and §2 of this doc are the shape and the voice
rules; §5 is the verification discipline and is non-negotiable.

⚠⚠ **RUN `scripts/verify_excerpt.py` OVER EVERY CHAPTER BEFORE YOU TOUCH ONE.** It found
**five bad log lines in three chapters this doc had already marked DONE** (§4a-quinquies).
One was outright fabricated. A "DONE" mark in the table below means the prose was passed,
not that the numbers were re-verified.

⚠ **The single most expensive lesson in this document, learned four separate times:
RUN THE BINARY THE CHAPTER CLAIMS TO DOCUMENT, BEFORE DOCUMENTING IT.** Every real defect
found in ch4, ch5 and ch6 was invisible to reading and to `lake build`, and two of them
printed normal-looking output while doing nothing at all.

**Why CUDA.** The verified trainers cannot complete a run on ROCm. See
`upstream-issues/2026-08-jax-rocm-command-buffer-launchgraph-segv/`:
confirmed ROCm-specific, 11 of 11 deaths on gfx1100 against 6 of 6 clean
on CUDA at the identical JAX version. Anything needing a captured
training log has to happen on NVIDIA. Do not try to work around AMD here.

---

## 1. What a made-over chapter looks like

Chapter 1 (`\chapter{MNIST: linear classifier}`) is the mold. Its shape:

| § | what |
|---|---|
| opening | 2 paragraphs: what the chapter builds, why it is the hardest/easiest |
| **1.1 Run it first** | commands, then a **real captured log**, then what it means |
| 1.2 How it works | the roadmap diagram, how the proofs are written |
| 1.3 The theorems | untouched formal content |
| 1.4 Example | the `VerifiedNetSpec`, the spec-to-proof tie, the whole driver, results |
| 1.5 MLIR: *operator* | the committed render, as committed |
| 1.6 What's inside `.train`? | the verified driver walked line by line |
| 1.7 MLIR: Training Step | the whole rendered step |

The ordering is the point. Book 1 put Results at position 4 of 8 — the
reader sees the number **before** the explanation. A chapter that opens
with history and buries the demo at 65% depth is a monograph; one that
opens with a working run is not. This is the difference between
proof-as-spine and proof-as-receipt, and it is why 1.1 exists.

### The 1.1 recipe

1. Four commands: `lake exe cache get`, the download script, `lake build
   <target>`, run it.
2. The captured log, verbatim, with only the XLA startup banner removed
   (say that you removed it).
3. Wall-clock and accuracy in prose.
4. **The paragraph that matters**: that run did not execute a
   reimplementation — it executed `verified_mlir/<slug>_train_step.mlir`,
   which is what `<renderer>` emits, and every theorem in the chapter is
   about the graph that just trained.
5. The compile lines, which carry the in-process-XLA story on their own.
6. A pointer to `lake run mnist` for the whole tier.

---

## 2. Style rules

These were derived by measuring book 1 (*Convolutional Neural Networks
with Swift for TensorFlow*, Apress 2021) against this book. They are
mechanical and checkable. **Tone is syntax, not diction** — chapter 1's
intro rewrite moved the word count 4% and changed the voice completely.

### The measurements that matter

```
                    words  mean sentence  em-dash/1k  semicolons
book 1 intro          738           24.6         0.0           0
blueprint intro (was)1756           16.1        19.4           5
blueprint intro (now)1604           16.4         0.0           0
```

Book 1's sentences are **longer** than ours. Length was never the
problem. Book 1 writes long sentences that run left to right and never
stop; ours were shorter but **interrupted**. The rule is *don't interrupt
the sentence*.

### The four patterns to cut

1. **Em-dash asides.** The signature tic. Move the aside to the end of
   the sentence, or use parentheses (book 1 uses those freely — they read
   as skippable, em-dashes pretend to be part of the flow).
   - `Every architecture in this book --- MLP, CNN, ... --- decomposes into eight things`
   - → `Every architecture in this book decomposes into eight things, and that includes MLP, CNN, ...`
2. **Denial clauses.** A second half refuting what nobody claimed.
   - `it needs one because of how its data is streamed, not because of how it computes`
   - → `and only to stream data`
   - `the claim is correct --- not because you trust the author, but because the compiler checked it`
   - → `the claim is correct. You don't have to trust the author, because the compiler checked it.`
3. **Throat-clearing lead-ins.** Announcing that a thing matters instead
   of saying it. `The part worth knowing about is how it splits` → `It
   splits in two`. Also `That division is the point:`, `That asymmetry is
   the point of the section below`.
4. **Semicolon chains that are really lists.** If a paragraph joins ≥3
   parallel items with semicolons, make it an `itemize`. Chapter 1's
   "Where this goes" chained eight chapters across seven semicolons; as a
   list, the reader can find their chapter.

### What NOT to cut

- **Explanatory repetition.** Book 1 restates technical points in plain
  words on purpose, and for a reader working in English as a second
  language that redundancy is error correction. Cut *ornamental*
  repetition (rhetoric restating rhetoric); keep a plain restatement of a
  technical point.
- **Lines that are already working.** "It turned out 'eventually' was
  about five years." / "That's it. Three rules, five tricks... There is
  no ninth." / "The compiler is the referee." / "This is correct and
  totally impractical." Short, concrete, landing a technical point. A
  careless pass flattens these; they are the mechanism, not decoration.
- **Emphasis that disambiguates.** Appendix B keeps one italic pair —
  `\emph{compile}` vs `\emph{runtime}` — because it separates two
  similarly-named IREE knobs. Everything decorative goes.

### Register by section type

Chapters get this voice. **Theorem, proof, and definition bodies are left
alone** — the register there should stay precise, and brett asked for
this explicitly. Appendix C's trust accounting likewise.

### The audience

Not beginners. Experts, written so the expertise looks easy — a reader
should notice the subject, never the sentence. Technical nouns stay hard
(Fréchet derivative, axiom closure, `select_and_scatter`); the connective
tissue between them gets simple. A grade-6 sentence containing
`stablehlo.select_and_scatter` is still a grade-6 sentence.

### The rule I broke, so you don't

**A voice pass changes punctuation and clause order. It never changes
words.** I let words drift in the intro pass — `production-quality` →
`have changed`, `verify` → `check` (in a book about verification),
`resources` → `material`. Caught by a word-frequency diff, reverted in
`f0c3a30`. Any word substitution is a separate decision that gets
flagged, not folded in. The check:

```python
# words present/absent between two revisions of a chapter, minus stopwords
git show <before>:blueprint/src/content.tex   # extract chapter, Counter() the words
# diff the two Counters; every delta should be explainable
```

---

## 3. Measuring

Run before and after. Theorem/proof bodies excluded, since they are out
of scope:

```python
import re
t = open('blueprint/src/content.tex').read()
p = [x for x in re.split(r'\n(?=\\chapter)', t)
     if x.startswith('\\chapter{MNIST: 2D CNN}')][0]
b = re.sub(r'\\begin\{(theorem|proof|definition|lemma|verbatim|tabular'
           r'|tikzpicture|axis|align\*?)\}.*?\\end\{\1\}', '', p, flags=re.S)
print(f"em-dash {b.count('---')}, semicolons {b.count(';')}")
for l in b.split('\n'):
    if '---' in l or ';' in l: print('  ', l.strip()[:100])
```

Target: **0 and 0** in prose. Chapters 1 and 2 are both there.

Chapter-wide ranking, to pick what's next:

| chapter | em-dash/1k | semicolons | state |
|---|---|---|---|
| Introduction, How this book is organized, Foundations | 0.0 | 0 | done |
| MNIST: linear classifier (ch1) | 0.0 | 26 (all in proofs) | done |
| MNIST: 1D MLP (ch2) | 0.0 | 0 | **DONE** |
| MNIST: 2D CNN (ch3) | 0.0 | 0 | **DONE** |
| CIFAR with BatchNorm (ch4) | 0.0 | 0 | **DONE** |
| ResNet-34 (ch5) | 0.0 | 0 | **DONE** |
| On Verification (app C) | 17.8 | 36 | **needs an argument rethink, see §6** |
| Getting started (app B) | 17.0 | 8 | flourishes cut, joinery not |
| Data availability (app A) | 16.4 | 14 | |
| MobileNetV2 (ch6) | 0.0 | 0 | **DONE** |
| EfficientNet (ch7) | 0.0 | 0 | **DONE** — max sentence 64 w |
| Bestiary | 8.1 | 58 | 110 em-dash raw, the biggest single job left |
| Vision Transformer (ch9) | 7.1 | 16 | 46 em-dash raw |
| ConvNeXt (ch8) | 6.1 | 16 | **▶ NEXT** — 31 em-dash raw; max sentence is **68 w, not 142** (§4a-sexies) |

▶ Measure with `scripts/measure_prose.py '\chapter{ConvNeXt}'`, which is the §3 script with
all three of its false-join bugs fixed (it now replaces stripped environments with a full
stop, and normalises `.)`, `.''` and `.}` before splitting). The raw §3 script over-reports
max sentence by 10–30%.

Re-measured 2026-08-12 with the §3 script, so these supersede the older
per-1k figures above. ▶ Max sentence is the metric this doc keeps forgetting:
ch1/ch2 sit at 52–54 words, ch5 finished at 94, ConvNeXt has a 142.

⚠ **The §3 script over-counts max sentence**, two ways, and both inflate it.
It does not strip `enumerate`/`itemize`, so a lead-in plus all its list items
reads as one sentence; and its split regex `(?<=[.!?])\s+` does not fire after
`.)`, so a sentence ending in a parenthetical swallows the next one. Strip the
list environments and normalise `.)` → `).` before splitting. Re-measured that
way: ch2 55, ch4 70, ch5 96, ch6 63 (and ch6's 63 is itself a colon lead-in
joined across a table, so its real max is ~55).

---

## 4. Chapters 2 and 3: DONE (2026-08-11/12)

Both shipped. Commits `e9a7081`, `8620eb8`, `3f404e8`, `b9402ac`, plus the
`IreeSession` rename `1d34920`. What landed, so you can copy the shape for
chapter 4:

- §2.1 / §3.1 **Run it first** from real CUDA logs
  (`runs/2026-08-11-{mlp,cnn}-verified-xla-cuda/`): MLP 12 epochs 97.83%,
  CNN 10 epochs 98.75%.
- The width sweeps re-measured on XLA (§2.4 at n=1, §3.5 at **n=5**).
- The unverified ablation runner **deleted** from both chapters, its loss
  plots kept and repointed at verified data.
- Voice pass: ch1/ch2/ch3 all now 0 em-dashes, 0 prose semicolons.
- `\phasethreenote` removed from both (markers 8 → 6).
- `IreeSession` → `LowererSession`, 269 sites, and chapter 1's apology
  paragraph for the name deleted with it.

### ⚠ PLACEMENT: this doc told you the wrong thing, and it cost a rewrite

§4 step 2 said to put "Run it first" *before* `\section{The theorems}`.
Following that literally put the demo **130 lines deep** in chapter 2,
behind six prosesections of Jacobian teaching, against chapter 1's 21.
That defeats the ordering principle §1 calls the whole point.

▶ **Put it after the FIRST prosesection.** Chapter 2 opens in 22 lines now,
chapter 3 in 27. Do the same for chapter 4.

### ⭐ The loss carve-out — you do NOT need to repeat this for CIFAR

Chapters 2 and 3 could not plot a verified loss because `mlp`/`cnn_train_step`
returned parameters and nothing else. `3f404e8` added a trailing `%loss`
scalar to both renderers as a **declared report-only carve-out** (banner in
the emitted MLIR, same as ConvNeXt/EfficientNet/R50) plus a `%lslot` unused
input to keep the shared C entry's single shape list symmetric.

**cifar8 already returns a loss** — `lake run cifar` prints
`Epoch 40/40: loss=0.400674 lr=...` today, because those nets run through
`trainAdamSched`, which already reads the scalar off the packed tail. So
chapter 4 needs none of that plumbing.

---

## 4a. Chapter 4, CIFAR with BatchNorm: DONE (2026-08-12)

Shipped. 0 em-dashes / 0 prose semicolons (from 60/27), max sentence 91 → 70 w,
`\phasethreenote` removed (markers 6 → 5), `latexmk` 0 errors.

⚠⚠ **THIS DOC WAS WRONG ABOUT THE DATA, and it cost the whole GPU budget.** §4a
said the ch4 measured work was "already banked" in
`runs/2026-08-11-cifar8-6arm-xla-cuda/`. That log is the **narrow-head**
(`128→64→64→10`) net that `lake run cifar` runs. Chapter 4's board is the
**wide 2×512-head** net (`cifar8w`), which was still on IREE via
`cifar8w-ablation`. They are different networks and the doc never said so.
▶ Before trusting any "already measured" pointer, `grep` the log's own banner
line for the architecture and diff it against what the chapter claims.

**The port.** `ireeLink` → `lowererLink` on `cifar8w-ablation` and
`cifar8w-bn-ablation` was the whole code change — `trainAdamSched` and all eight
`cifar8w*` renders were already in place. Re-measured at **n=5** (n=4 AdamW) in
`runs/2026-08-12-cifar8w-6arm-xla-cuda/`.

⭐ **`LEAN_MLIR_CKPT_TAG` is new and §3c depends on it.** Every pass of one
(net, variant, backend) shared ONE checkpoint path, so the parallel sweeps §3c
mandates could not be run: concurrent passes clobber each other's blob, and a
later pass resumes from an earlier one's finished epoch 40 and trains **nothing**
while printing a normal-looking log. Set it to the pass index. Also: clear stale
`.lake/build/<slug>_*_ckpt_xla*.bin{,.epoch}` before any re-measurement — the
first attempt here silently trained zero epochs off July checkpoints.

⭐ **The result changed, for the better.** BN's payoff at this depth is
**stability**, not the sub-noise "wash" the old IREE board showed:

| arm | n | median | range | diverged |
|---|---|---|---|---|
| SGD, no BN | 5 | 72.64 | 1.77 | 0/5 |
| SGD, BN | 5 | 75.00 | 1.13 | 0/5 |
| momentum, no BN | 5 | — | — | **5/5 NaN** |
| momentum, BN | 5 | **77.14** | 0.77 | 0/5 |
| AdamW, no BN | 4 | 73.71 | 0.97 | **1/4 NaN** |
| AdamW, BN | 4 | 74.36 | 0.83 | 0/4 |

The un-normalized net's loss goes to `NaN` in 6 of 14 runs; the BN net's in 0 of
14. It trains normally to loss ≈0.27, then dies between epochs 29 and 34 at an
lr the cosine had already decayed to ≈0.002, and 3 of 5 collapse to 10%. This is
**not new** — `robustness_handoff.md:140` and `render_close_handoff.md:67` both
recorded "BN avoids the divergence entirely" for the plain-SGD case. What moved
on XLA is *which* optimizer exposes it. The momentum cell prints *diverged*
rather than a number.

▶ **OPEN, deliberately not chased:** whether the wide-head no-BN momentum NaN is
XLA-specific or a property of the recipe. An IREE run of the same arm answers it
in one command (`LEAN_MLIR_LOWERER=iree`); it was started and killed. All five
logs are kept.

### Also landed: the blurb pattern

`VerifiedNetSpec.blurb` hard-coded its transport, so the banner described the
build rather than the run. Three of seven print sites patched it with
`.replace "IREE FFI" "XLA/PJRT"` and **four printed it raw**, so nets on
`trainAdamPacked` and the three E4M3 trainers announced IREE while training on
XLA. Blurbs now carry the literal `%LOWERER%` and `VerifiedNet.printBlurb`
resolves it once. No committed book log was affected (checked). ▶ Use
`printBlurb`, never `IO.println net.blurb`.

### Chapter 4's specs were fiction

§4.4 printed `cifar8NoBn`/`cifar8Bn` as `NetSpec` — **names that exist nowhere in
the codebase**, in a layer syntax (`.conv2d 3 16 3 .same .relu`) the real type
does not use. Replaced with the actual `cifar8wVerified` / `cifar8wBnVerified`
`VerifiedNetSpec`s plus the §1.4 spec-to-proof tie (slug → committed render,
`toSpecs` `#guard`, and `cifarCnn8_has_vjp_at` being parametric in head width,
which is why wide and narrow share one proof). ▶ **Worth checking in every
remaining chapter**: a listing that predates the verified types still typesets
fine and reads as current.

---

## 4a-bis. Chapter 5, ResNet-34: DONE (2026-08-12)

57 em-dashes / 30 prose semicolons → **0 / 0**. `\phasethreenote` markers 5 → 4.
latexmk 0 errors. Commits `bf248a6` … `bbcce23`.

⭐⭐ **§5.3 IS NOW THE REFERENCE TRAINER PATTERN — `\label{sec:verified_trainer_pattern}`.**
This is the thing to reuse. It is the real `resnet34Verified : VerifiedNetSpec` +
`VerifiedConfig` + `trainAdamSched` entry point, with the four annotated points
every later chapter copies:

- **`slug` names the artifact.** The trainer LOADS
  `verified_mlir/<slug>_<variant>_train_step.mlir`; it does not build a net from
  the spec at run time. The spec says which file and is `#guard`ed against it.
- **`bnChannels` drives BN statistic threading**, and getting it wrong reports
  chance forever (measured, see the bug list below).
- **`trainAdamSched`, not `.train`** — the latter has no BN threading.
- **The recipe is arguments, not architecture**, which is what lets one proven
  gradient serve every optimizer comparison.

⚠ §5.3 previously printed a `NetSpec`/`TrainConfig` listing matching no code and
an `IREE_BACKEND=rocm … gfx1100` log. Both gone. **Check every remaining chapter
for the same thing**: a pre-verified-era listing typesets fine and reads current.

What else landed: §5.1 "Run it first" from a fresh 80-epoch XLA capture
(`runs/2026-08-12-r34-imagenette-xla-cuda/`, 89.71% / 98.27%, ~65 s/epoch);
§5.7 restructured so the verified path is the result and JAX is the reference,
now describing the **90-epoch paper recipe** (the driver default moved 30 → 90);
§5.8 comparing **R50-to-R50** so the A3-vs-2018 table is a recipe diff rather
than one confounded with a backbone diff.

### ⚠ The two `[TODO]` runs are real commands, not aspirations

Both verified to start and print `Epoch 1/90`. Neither has been run.

```bash
# R34 / ImageNet, 90 ep — ~27.9 h. 90 is now the driver default.
CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  SHIM_WORKERS=8 PJRT_FFI_RESIDENT=1 \
  LEAN_MLIR_VARIANT=momdp64 LEAN_MLIR_BATCH=64 \
  .lake/build/bin/resnet34-imagenet-verified data

# R50 / ImageNet on the 2018 recipe, 90 ep — the controlled peer for §5.8.
CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  SHIM_WORKERS=8 PJRT_FFI_RESIDENT=1 \
  LEAN_MLIR_VARIANT=momdp64 LEAN_MLIR_BATCH=64 LEAN_MLIR_EPOCHS=90 \
  LEAN_MLIR_BASE_LR_U=100000 \
  .lake/build/bin/resnet50-imagenet-verified data
```

⭐ **`LEAN_MLIR_EPOCHS` is new and it SETS where `LEAN_MLIR_MAX_EPOCHS` only
CAPS** (`min n cfg.epochs`). Not cosmetic: `totalSteps := cfg.epochs * nb / accK`
is what the cosine anneals over, so `EPOCHS=30` is a complete 30-epoch experiment
while `MAX_EPOCHS=30` is a PREFIX of a 90-epoch decay stopped with the LR high.
⚠ Clear checkpoints when switching schedules; resuming across them fuses two LR
curves silently.

⚠ `resnet50in_mom256` (single device, bs256) **OOMs** — 14.64 GiB for the d2h on
a 16 GB card. `momdp64` (4×64, same global 256) is the runnable one. Found by
running it.

⚠ Do not read "we're doing to JAX what we did to IREE" as retiring JAX. Confirmed
2026-08-12: the reference implementations stay, and the work is bringing PJRT to
**parity** with them. The migration model brett described: run the phase-4
trainer, redo the phase-2 section's graph with that data, move it to the phase-4
section, and the phase-2 sections get eaten one at a time.

---

## 4a-quater. Chapter 6, MobileNetV2: DONE (2026-08-12)

51 em-dashes / 10 prose semicolons → **0 / 0**. Max sentence 87 → 55 w.
`\phasethreenote` markers 4 → 3. latexmk 0 errors, 0 undefined refs.

⭐ **§6.1 is a fresh 80-epoch XLA capture** (`runs/2026-08-12-mnv2-imagenette-xla-cuda/`),
and the result is much better than the retired IREE number the chapter had:
**89.25% top-1 / 98.68% top-5 in 35 minutes** on one 4060 Ti, against the old
87.09%. Run `mobilenetv2-verified-adam` (NOT `mobilenetv2-verified`, which is one
of the three `.train` drivers that report chance). ~26 s/epoch, ~90 ms/step.

▶ **That reframes the chapter's thesis.** MNv2 is 0.46 points of top-1 behind
R34 (89.71%), not 3.2, and it BEATS R34 on top-5 (98.68 vs 98.27) at 9.5× fewer
parameters and 2.4× the throughput. The old "you give up a couple of points"
framing was an artifact of the stale IREE run.

⚠ **The lakefile's own estimate was 3× too slow** (`probeXla ... epochs := 80` says
XLA 1h25m; actual 35 min). That comment is annotated "measured on the PRE-§2m net",
i.e. before the 52 conv biases were dropped. Don't budget off it.

⭐ **What replaced the fiction.** §6.3 printed `mobilenetV2 : NetSpec` +
`TrainConfig`; now it is the real `mobilenetv2Verified : VerifiedNetSpec` +
`trainAdamSched` in §5.3's shape. ⚠ The *prose* was wrong too, not just the
listing: it described `.invertedResidual ic oc t stride n` (the phase-2
`Types.lean` `Layer`), where the verified type is `.invertedResidualNB ic mid oc
stride` — expanded width, not ratio, and one constructor is one block.
▶ **Check the prose around every listing you replace, not just the listing.**

**§6.4 restructured on brett's instruction (2026-08-12):**

- The **90-epoch/cosine tier is deleted** — table row, its 44-line plot, the
  68.77%/88.53% prose, the +0.44-over-SGD comparison, and the "One honesty note"
  paragraph. ⚠ Four other passages leaned on it and had to be rewritten, not just
  cut: the 350-ep tier's opening ("The 90-epoch run above is a validation tier"),
  its "+2.67 over the 90-epoch tier", the "−3.23 gap was training budget" analysis,
  and the recipe-diff table's Epochs row.
- **Phase 4 is now its own labelled subsection at the END** (`sec:mnv2_phase4`),
  immediately before "What the MobileNetV2 recipe changes", mirroring ch5 where
  `sec:r34_phase4` also closes the ImageNet section. Its lead-in flipped from
  "why the numbers below are still JAX" to "What it has not done is run".

### ⭐⭐ A phase-4 config's job is to MATCH ITS PHASE-2 RUN (brett, 2026-08-12)

`mobilenetv2ImagenetConfig.epochs` was **300**; it is now **350**. The phase-2 number this net is
measured against is the paper-faithful **350-epoch** tier (71.44% / 90.34%), so a 300-epoch
phase-4 run answers a question nobody asked. `totalSteps := cfg.epochs * nb / accK` is what the
schedule anneals over, so 300 vs 350 is a **different LR curve end to end**, not a prefix — the
two results would not be comparable, which defeats the entire point of running phase 4.

▶ **This is a rule, not an MNv2 fix. Check it for every remaining chapter**: the phase-4
`VerifiedConfig` must carry the epoch count of the phase-2 tier whose number the chapter prints.
I had originally *documented* the 300/350 disagreement in a two-column schedule table, which was
the wrong instinct — the disagreement was a bug in the config, not a fact about the world.
▶ **When a listing and the section it belongs to disagree, fix the listing before you annotate
it.** ch6's table is now one column, 350 epochs, ~91.5 h.

MNv4's driver is already consistent with this rule at 100 epochs, matching the Conv-M tier-2
schedule its section reports. ⚠ ch7/ch8/ch9 are unchecked.

⭐ **The MNv2 ImageNet driver can run the PAPER's optimizer**, which the parked
commit did not know. `LEAN_MLIR_VARIANT=rms64` loads
`mobilenetv2in_rms64_train_step.mlir` with RMSProp ρ.9/μ.9/ε1.0, wd 4e-5, peak
0.045, ×0.98/epoch baked in, off `mnv2RmsSchedule` + `mnv2RmsHyper` (shared with
the Imagenette peer so they cannot drift). `adam64` is only the default.

### ⚠ Numbers that look wrong and are NOT — check provenance before "fixing"

The ImageNet headlines disagree with their own training logs, and both are right:

| | headline (quoted) | in-loop log (plotted) |
|---|---|---|
| MNv2 90ep | 68.77 / 88.53 | 68.39 / 88.15 @ 49,920 |
| MNv2 350ep | 71.44 / 90.34 | 71.46 / 90.33 @ 49,920 |
| MNv4-Conv-M 100ep | 75.48 / 92.37 | 75.51 / 92.37 @ 49,664 |

**The headline is an offline full-50k EMA eval; the curve is the in-loop eval over
the batch-divisible subset with live weights.** `planning/paper_faithfulness.md:153`
documents this for the 90ep row ("offline full-50k EMA eval"). I nearly "corrected"
75.48 → 75.51 before finding it. ▶ ch6 now states the distinction once, in §6.4.

### ⚠⚠ The cross-chapter comparison table is stale in THREE places

`ch7 / ch8 / ch9` each carry a cumulative table whose R34 row still reads
`21.29M / 518 KB / 1400 ms / 9.5 h / 90.29%`. **Chapter 5's pass missed it**, and
ch6's would have made it worse. Mixing vintages inside one table breaks the
like-for-like comparison that is its whole point, so all three now carry a
provenance note (§5.5's "label it instead") pointing at `sec:r34_runit` and
`sec:mnv2_runit`. ▶ **Re-measure these rows as each chapter is made over.** Real
values so far: R34 729 KB / 220 ms / ~1.5 h / 89.71%; MNv2 1,047 KB / 90 ms /
35 min / 89.25%.

### ⚠ The whole-net VJP fold is NOT proved at full depth

`mobilenetv2_has_vjp_at` is proved for **stem + 2 inverted-residual blocks + head**,
not all 17 (`VerifiedNets.lean:619` says so; the theorem's binders only go to ₂).
The full 17-block tie is *denotational* and about the **forward**
(`mobilenetv2Verified_denote_eq`, `mobilenetv2Verified_fwd_faithful`). The
per-operation VJP theorems ARE unconditional and do cover all 17. §6.1 now says
exactly that. ▶ **Same check is owed by ch7 and ch8** — `efficientnet_has_vjp` is
flagged "representative" at `VerifiedNets.lean:720`.

### ⭐ MNv4 ImageNet phase 4: BUILT AND RUNNING (2026-08-12)

`mnv4-imagenet-verified` exists and trains. Built exactly the way R50's was, in four pieces:

1. `mnv4ImagenetVerified : VerifiedNetSpec` (slug `mnv4in`, `VerifiedNets.lean`) — Conv-S trunk
   copied from `mobilenetv4Verified`, head `1280×10 → 1280×1000`, params
   4,124,426 → **5,392,616**. Carries MNv2's three-way `#guard` pin (`toSpecs.size`,
   `toSpecs.pop.pop`, `back! == (#[1000], 2)`), the param-count guard, the stat-alignment guard,
   and the `d0 == 3*224*224` row in the cross-net `.imagenet` invariant.
2. Three `#eval`s in `MobileNetV4RenderB.lean` → `mnv4in_{fwd,fwd_eval,adam64}_train_step.mlir`.
   The renderer already took `slug` and `replicas` as defaulted args, so this was three lines.
3. `apps/imagenette/MainMobilenetV4Imagenet.lean` + `lean_exe mnv4-imagenet-verified`.
4. `gen_shims.sh` row `mobilenet-v4-imagenet:default:generated_mobilenet_v4_imagenet_shim.py`,
   shim generated (23,786 bytes).

**Verified working end to end**: all three artifacts compile under PJRT (9.9 s / 0.4 s / 0.3 s),
shim feeds it, first step loss **6.99** against ln 1000 = 6.91, eval **0.106%** top-1 over
49,920 val = 1/1000. **114 ms/step** single device at bs64 (marginal over 40 steps, so start-up
and the 28 GB val drain cancel) → 38.1 min/epoch → 63.5 h for 100 epochs on ONE card.

⚠ **That single-card figure is NOT in the book, on brett's instruction (2026-08-12), and the
reason generalises.** Every other ImageNet row in ch6 — including the Conv-M tier's ~9.0 min/epoch
/ ~16 h — was measured at **4× on this same box**. Printing a 1-GPU row next to them invites the
reader to diff 63.5 h against 16 h and conclude something about the architectures, when the only
real difference is the card count. The blueprint table therefore carries the 4× row alone, all
`TBD`. ▶ **Match the box to the rows already on the page, or the table lies by arithmetic.** The
114 ms/step measurement is kept HERE because it is what someone budgeting the run needs.

⚠ **The stale-checkpoint trap bit again, and it looked like success.** The first throughput probe
resumed from the previous probe's finished epoch-1 checkpoint, trained ZERO steps, printed no
step lines and still printed `done (trained ...)` with exit 0. Two probes 40 steps apart came
back 0.34 s apart, which is the only reason it was caught. ▶ `rm -f
.lake/build/mnv4in_adam64_ckpt_xla*` between probes, or set `LEAN_MLIR_CKPT_TAG`.

⚠⚠ **It is Conv-S; the chapter's 75.48%/92.37% is Conv-M.** Different networks (4.1M vs ~9.7M,
different block tables). Conv-S has NO published ImageNet target, so a run here is a first
measurement, not a reproduction. The blueprint's §6.5 phase-4 subsection says this in bold and
leaves its Val top-1 column `TBD`. The shim is generated from the **Conv-M** reference recipe and
that is fine and deliberate — a shim carries augmented batches, not weights.

### ▶ What is NOT implemented on the PJRT/verified side (all checked against code, not guessed)

| gap | status | evidence |
|---|---|---|
| **Data parallel** | **not rendered** | `mnv4AdamVariant 64 2` names `adamdp64` and the renderer takes `replicas`, but MNv4 has no `shard-check` row (`convnext\|efficientnet\|mobilenetv2` only) and no `TestMnv4DpCheck.lean`. Deliberately NOT emitted: an untied collective artifact would look as trustworthy as the tied ones. → single-device only, hence 63.5 h |
| **Drop-path / stochastic depth** | **exists, not wired into UIB** | `LeanMlir/Proofs/Codegen/DropPath.lean` is a shared module and ENet/ConvNeXt render `*_adamdrop_*` variants off it. `MobileNetV4RenderB.lean` has 0 hits for it and `mnv4AdamVariant` takes only `(B, replicas)`, so there is no marker to ask for it. ⚠ `jax/MainMobilenetV4Imagenet.lean:114` says the reference side is not wired either (`dropPath := 0.075 -- NB: not yet wired into UIB`), so this gap is on BOTH paths |
| **Classifier dropout** | **exists, not wired into UIB** | the `"do"` marker, rendered for ENet (`efficientnet_adamdo_train_step.mlir`). Same story: no MNv4 hook. Reference uses 0.1 (tier-2) / 0.2 (paper) |
| **RandAugment m15** | **not expressible** | shim's `default` recipe is m9; `jax/...:113` notes "codegen clamps M to 0–10", so the paper's 15 cannot be asked for |
| **Effective batch 4096** | renderable, not rendered | `accK` grad-accum exists and R50 has `resnet50in_acc4x64_train_step.mlir`, so the machinery is there; no accumulating `mnv4in` variant was emitted. Driver runs bs64 @ LR 1e-3 against the reference's 4096 @ 0.004 |
| **bf16** | **whole verified path is fp32** | 0 bf16 artifacts in `verified_mlir/`; `mnv4in` train step is 18,706 × `f32`, 0 × `bf16`. Not MNv4-specific — ch5's phase-4 table says fp32 too |

**Implemented and working**, so not gaps: EMA with the TF warmup correction
`min(d, (1+t)/(10+t))` (`VerifiedTrain.lean:1560` — the exact fix §6.5's "One trap in that EMA
row" describes), cosine + warmup, AdamW + coupled decay, BN running-stat threading (52 layers,
42,592 stat floats), gradient accumulation, and the phase-2 gradient tie
(`scripts/grad_tie.py --net mnv4`).

⚠ **Two of those rows I got wrong on the first pass and the regen gate caught it.** I had written
drop-path and dropout as "absent from the spec language" on the strength of `VLayer` having no
constructor for them. They are not layers — they are *render variants*, selected by the `"drop"`
and `"do"` markers in `enetAdamVariant`, off the shared `Proofs/Codegen/DropPath.lean`. So the
mechanism exists and is proven for MBConv and ConvNeXt blocks, and the MNv4 gap is wiring it into
UIB plus giving `mnv4AdamVariant` the markers. That is a much smaller job than building it, and
the wrong version of this note would have sent someone off to write one from scratch. ▶ **Look
for a render VARIANT before concluding a feature is missing**; `ls verified_mlir/ | grep <marker>`
answers it in one command.

▶ **Cheapest next step, and it is now the blocking one**: render `mnv4in_adamdp64` and add the
`shard-check`/dp-check rows. That is what fills in the blueprint's 4× row at all — without it
the net simply cannot be run in the configuration the chapter reports everything else in, and a
single-card run would produce a number that cannot be printed beside the others.

### ▶ 4-GPU time ESTIMATE: ~20 h for 100 epochs (2026-08-12)

Derived without rendering the untied DP artifact, by borrowing **MobileNetV2's** measured
parallel efficiency — closest architecture, same box, same batch, same shim:

| net | 1× bs64 | 4× global 256 | speedup |
|---|---|---|---|
| MobileNetV2 | 146 ms/step → 48.7 min/ep | 188 ms/step → 15.7 min/ep *(measured 2026-08-11)* | **3.10× (78% eff)** |
| MNv4-Conv-S | 114 ms/step → 38.1 min/ep *(measured)* | ~147 ms/step → **~12.3 min/ep** *(est.)* | assumed same |

→ **100 epochs ≈ 20 h (0.9 d) on 4× 4060 Ti.**

⚠ **Two things this estimate is NOT.** (1) It is a second-order extrapolation — MNv4's own 4×
ms/step has never been measured, because the artifact that would allow it does not exist. That is
why the **blueprint row stays `TBD`**: the book's other schedule columns extrapolate from a
ms/step measured *on that net*, and this one cannot. (2) It is **fp32 at global 256**, against the
phase-2 Conv-M row's **bf16 at effective batch 4096**. That is why this estimate (12.3 min/ep) is
*slower* than the Conv-M tier's measured 9.0 min/ep despite Conv-S being the smaller net — the
verified path has no bf16 at all. Do not read the two as a Conv-S-vs-Conv-M speed comparison.

⭐ MNv4-Conv-S is **faster per step than MNv2** single-device (114 vs 146 ms) despite more
parameters (5.4M vs 3.5M). That is the UIB design goal working as advertised: the block was
chosen for latency on real hardware, not for parameter count.

### ⚠⚠ `mobilenetv2-imagenet-verified` HAD NO `LEAN_MLIR_EPOCHS` — fixed

Measuring the above hit it. The R50 and MNv4 ImageNet drivers both read `LEAN_MLIR_EPOCHS`;
MNv2's did not, and **silently ignored it**, so `EPOCHS=1` ran the full committed schedule. With
`G2_STEPS=20` that is 350 epochs of 20 steps, and the probe ran until it was killed 10 minutes
later, having written a checkpoint at epoch 179. Combined with the 350-epoch change above, a
short probe of this net was simply impossible. Now added, matching R50's spelling.

▶ **When you change a committed `epochs`, check the driver actually has the knob to override it.**
The two changes are individually harmless and jointly make the net unprobeable.

---

## 4a-quinquies. ▶▶ START HERE: Chapter 7, EfficientNet-B0: DONE (2026-08-12)

48 em-dashes / 12 prose semicolons → **0 / 0**. Max sentence 75 → **64 w** (ch5 is 64, ch6
is 56). `\phasethreenote` uses 3 → 2. latexmk 0 errors, **0 undefined refs**. ch7 is now
**IREE-free**: 0 hits for `IREE|vmfb|gfx1100` in the chapter.

### ⭐ §7.1 is a fresh 80-epoch XLA capture, and B0 WINS the Imagenette column

`runs/2026-08-12-enet-imagenette-xla-cuda/`, one 4060 Ti, `efficientnet-verified-adam`:
**89.96% top-1 / 98.45% top-5 in 40.5 minutes**, best epoch 90.14%. 30.4 s/epoch,
103 ms/step. Against the retired IREE number the chapter had (87.58% in 6.2 h) that is
**+2.4 points and 9× faster**.

| net | params | MLIR | ms/step | total | top-1 | top-5 |
|---|---|---|---|---|---|---|
| ResNet-34 | 21.29M | 729 KB | 220 | 1.5 h | 89.71 | 98.27 |
| MobileNetV2 | 2.24M | 1,047 KB | 90 | 35 min | 89.25 | **98.68** |
| EfficientNet-B0 | **4.02M** | 1,316 KB | 103 | 41 min | **89.96** | 98.45 |

⚠ **The lakefile's own bench row said XLA 1h34m; actual 40.5 min.** Third chapter running
where that estimate is 2–3× pessimistic (ch6's was 3×). Do not budget off it.

### ⭐⭐ The chapter's parameter count was wrong and it INVERTED a conclusion

ch7 printed **7.16M** params for the Imagenette B0. The verified spec derives
**4,020,358**. The 7.16M is a **pre-fix** number from when the codegen sized the SE
bottleneck off the expanded width — a bug *the same chapter describes two sections later*,
where it correctly quotes the fixed 5.29M for the 1000-class net. The chapter was carrying
the buggy count and the fixed count simultaneously.

Cross-checked: `efficientnetImagenetVerified` = **5,288,548** = B0's canonical 5.29M,
`mobilenetv2Verified` = 2,236,682, `resnet34Verified` = 21,289,802. The last two match the
book exactly, so only ENet's row was stale.

▶ **This changed the analysis, not just the number.** The old bullet read "EfficientNet-B0
is 3.2× the parameters of MobileNet V2 for a 0.5 point lift — SE is an expensive addition."
Real figures: **1.80×** the parameters for **+0.71** top-1, and B0 takes the top-1 crown
from R34 at 5.3× fewer parameters. Rewritten, not renumbered.

### ⚠⚠ FIVE BAD LOG LINES IN THREE CHAPTERS THIS DOC CALLED DONE

`scripts/verify_excerpt.py` checks every quoted `Epoch n/N:` / `epoch n:` / `[pjrt_ffi]` /
`done (` line in a .tex range against the real log, handling the book's line-wrapping. It
audited 112 quoted lines across ch1–ch6 and found:

| chapter | defect | mode |
|---|---|---|
| ch5 | `done (trained ResNet-34 via the proof-rendered StableHLO)` — **the binary never printed this**; it printed `done (trained ResNet-34 adam + cosine/warmup via packed threading)` | **fabrication** |
| ch2 | compile lines said `6 outputs / 2398 ms / 122 ms`; log says `7 outputs / 2103 ms / 118 ms` | stale carryover |
| ch3 | compile lines said `10 outputs / 2655 ms / 389 ms`; log says `11 outputs / 2490 ms / 383 ms` | stale carryover |

All fixed; all seven chapters now verify clean.

▶ **The two modes need different defences and §5 only covers one.** §5 says to diff the
excerpt when you ELIDE a run, which catches fabrication. It does not catch staleness,
because a stale line was genuine when pasted and nothing about eliding draws the eye to it.
**Re-capturing a run silently invalidates every quoted line, not just the ones you edited.**
⚠ The ch2/ch3 tell was that the *prose* already had the right answer: ch2 says
`MlpFaithfulPoC` proves "each of the six parameter outputs" plus a trailing report-only
`%loss`, i.e. 7 total — the prose had been updated and the pasted log had not, and the two
sat contradicting each other in the same section.

### ⚠⚠ A LIVE DRIVER BUG that blocked the one thing ch7 was supposed to be able to do

`MainEfficientNetImagenet.lean` tested `variant.startsWith "rms"`. The paper-recipe variants
are spelled **`emarms…`**, which does not start with `rms`. Six committed artifacts —
including `efficientnetin_emarmsdp64dropdo`, the exact one this doc wanted for ch7's
phase-4 row — would have run with **RMSProp's state** (the shared trainer's own test is a
substring, so it initialised the mean-square to 1.0 correctly) and **AdamW's 1e-3 with a
cosine schedule** instead of the paper's 0.016 with ×0.97 every 2.4 epochs. It descends and
prints a normal-looking log.

⚠ `tests/TestVariantPredicates.lean` exists FOR this collision class and records it as case
1 — but it defines a `private def rmsOn` **copy** of the predicate, so it proves the
predicate is right and never gated a single driver. Fixed here; **verified by running**
(`lr=0.003200` = 0.016/5 after, against 0.0002 before — 16× apart).

▶ **Two more drivers still carry the prefix test**, latent only because no `emarms`
artifact exists for their net yet: `MainMobileNetV2Imagenet.lean:52` and
`MainMobilenetV2VerifiedAdam.lean:71`. Render one MNv2 EMA variant and they go live.

### ⭐ ch7's whole-net VJP IS proved at full depth — unlike ch6

This doc told ch7 it owed ch6's "representative scale" apology because
`efficientnet_has_vjp` is flagged representative. That is true of *that* theorem, but
there is another one: **`efficientnetForwardB_full_has_vjp`**
(`Proofs/Architectures/EfficientNetFullB0.lean:381`) chains stem → **all 16 MBConv blocks**
→ head through `vjp_comp`, batched over N, closing on `exact vjp_comp _ _ f16 dH e16 vH`.
Zero `sorry` in all three ENet proof files. §7.1 now makes the strong claim.

▶ **ch8 is owed the same check before it repeats ch6's caveat.** Look for a
`*FullT*`/`*Full*` file beside the representative theorem, not just the theorem the spec's
docstring names.

### Code changes, all typechecked

1. **`efficientnetImagenetConfig.epochs` 80 → 350.** brett approved 2026-08-12. Forced, not
   preferred: **there is no momentum/SGD render for `efficientnetin`** (only the AdamW and
   RMSProp families), so the 80-epoch SGD tier is unreproducible on the verified path and
   the 350-epoch RMSProp tier is the only phase-2 number these artifacts can target.
2. **`LEAN_MLIR_EPOCHS` added** to `MainEfficientNetImagenet.lean` (R50's spelling), in the
   same commit as the epoch change. ⚠ **`MainConvNeXtImagenet.lean` and
   `MainViTImagenet.lean` still have 0 hits — ch8 and ch9 inherit this.**
3. `VerifiedNets.lean` / `MainEfficientNetVerified.lean`: the layout comment said "262
   params"; `toSpecs.size` is **213**. ⚠ No gate could see it — the `#guard` compares
   `toSpecs == EfficientNetLayout.specs`, array against array, so the number in the comment
   beside it was never an operand.

### ▶ Phase 4: NOT run, deliberately

brett's call 2026-08-12: use the already-measured **371 ms/step → 30.9 min/ep → ~180.5 h
(7.5 d)** for 350 epochs on 4× 4060 Ti rather than probe or run it. `sec:enet_phase4`
prints that with Val top-1 **TBD**. ⚠ The subsection states plainly that the verified path
is fp32 while every phase-2 row is bf16, so 30.9 min/ep must not be read against phase 2's
8.6 min/ep as an architecture result.

⭐ ENet's collectives ARE gated, by two gates that prove different things:
`TestEfficientNetDpCheck.lean` pins the DP forward bit-exactly but hands both replicas the
SAME rows, so it is structurally blind to a shard-offset bug; the `shard-check efficientnet`
row closes exactly that by giving them different data. ▶ Do not write "tied" as if it were
one property.

### ⚠ The phase-2 ImageNet listing had drifted too — check ch8's

This doc said to KEEP ch7's phase-2 config listing as the reference. It is real code, but
the book printed `learningRate := 0.1`, `useAdam := false`, `cosineDecay := true` while
claiming to mirror `jax/MainEfficientNetImagenet.lean`, which now holds **RMSProp at 0.016**
with exponential decay, AutoAugment and classifier dropout. Both book tiers come from that
one config today (`default` 80 ep, `full` 350 ep). Fixed, plus a note that the 80-epoch SGD
row is a real measurement of a recipe the listing no longer reproduces.
▶ **"Keep as the phase-2 reference" is not the same as "it is accurate."**

### ▶▶ TODO: FIX — MobileNetV2's whole-net VJP fold is still 2 blocks of 17

Raised by brett 2026-08-12, reading ch7's §7.1 contrast paragraph. `mobilenetv2_has_vjp_at`
(`Proofs/Architectures/MobileNetV2.lean:489`) binds stem + `We₁/Wd₁/Wp₁` + `We₂/Wd₂/Wp₂` +
head. **Two** inverted-residual blocks; the net has seventeen. ch6's §6.1 states this out
loud and ch7's §7.1 now contrasts against it, so closing it edits the book in two places.

⚠⚠ **The obvious plan — "copy `EfficientNetFullB0.lean`" — DOES NOT TRANSFER, and the
reason is load-bearing.** `efficientnetForwardB_full_has_vjp` is a **global** `HasVJP` over
all 16 blocks, and it can be global only because EfficientNet's activation is **swish**,
smooth everywhere (ch7's MLIR caveat: the SE fan-in carries no kink condition, and the only
smooth-point hypotheses are the BatchNorms' `0 < ε`). MobileNetV2 is **relu6**. A global
`HasVJP` through a kink is FALSE. `MobileNetV2FullPaper.lean`'s own header already says
this: "relu6 is kinked, so the whole-net input-VJP stays pointwise-only (the repo standard
for relu-family nets)" — which is why that file delivers forward + faithfulness for all 17
and stops.

▶ **So there are two axes and only one is movable.** Depth (2 → 17) is the gap. Pointwise →
global is not, and must stay the `_at` form. Anyone who conflates them will spend the day
proving something untrue.

**What exists already**, so this is assembly rather than new mathematics:

- `MobileNetV2FullPaper.lean` has the four block shapes (`ivNoExpW`, `ivExpOnlyW`,
  `ivResidW`, `ivStridedW`) and the full 17-block chain `mobilenetv2ForwardPaper`, 0 sorries.
- The per-operation VJPs it would compose (depthwise, pointwise, BN, additive skip) are all
  proved and unconditional.
- EfficientNet shows the target shape: eight lemmas
  `mb{NoExp,Exp,Resid,Strided}W_{has_vjp,differentiable}` chained by `vjp_comp`.
- The `iv*` peers of those eight are the missing piece: **zero of them exist today.**

⭐ Note this is the SECOND time a "representative scale" caveat turned out to be worth
re-checking, and the two went opposite ways: ch7's was **understated** (the full-depth
theorem existed and nobody had found it), ch6's is **real**. Check the code, not the
caveat, and check it in both directions.

### ▶ Open, not chased

- **`lakefile.lean:1805`** documents `efficientnet-adam-tie` as `ireeLink` because
  "`efficientnet-verified-adam` is an IREE binary, and a tie must run on the backend the
  trainer actually uses." That premise is now false — the trainer is on `lowererLink` and
  defaults to XLA. Either the docstring is stale or the tie runs on the wrong backend, and
  telling those apart means RUNNING the gate, not editing it.
- **Cheap hardening that closes classes, not instances**: a `#guard` on the *printed*
  quantity (`scalarParams efficientnetVerified.toSpecs == 4020358`) would have caught the
  262; having the drivers import one shared `rmsOn` instead of hand-rolling three copies
  would have collapsed them into the one `TestVariantPredicates` already covers.

---

## 4a-sexies. ▶▶ NEXT: Chapter 8, ConvNeXt-T

**ch8 is `content.tex` 7658–8438.** Baseline with `scripts/measure_prose.py`:
**31 em-dashes, 16 prose semicolons, max sentence 68 w.**

⚠ **This doc has claimed for weeks that ch8 has "one 142-word sentence." It does not.**
That came from the raw §3 script, which welds sentences across stripped environments and
across `.)`. The real max is 68 w, which is 4 w over ch5/ch7's finished 64. So ch8's prose
job is the em-dashes and semicolons, not a monster sentence hunt.

▶ **ch8 is in the best starting position of any chapter so far.** Three of the things that
cost ch7 the most time are already correct here. Check them rather than assume, but expect
them to pass.

### 1. §8.1 "Run it first" — one command, and the estimate is probably RIGHT this time

```bash
lake build convnext-verified-adam
CUDA_VISIBLE_DEVICES=0 PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
  ./.lake/build/bin/convnext-verified-adam data \
  > runs/<date>-convnext-imagenette-xla-cuda/convnext-imagenette-xla.log 2>&1
```

`convnext-verified-adam` is already on `lowererLink`. lakefile line 2939 says IREE 13.3 h
against **XLA 1h54m01s**.

⭐ **Unlike ch6's and ch7's, this estimate is likely accurate**, and the reason is in the
comment beside it: `⚠ was 6960 (1h56m) = the retired SCALAR-LN net; §2o Part B re-ran the
channel-LN net at 6841s`. Somebody actually re-measured it against the current net. ch6's
and ch7's were 3× and 2.3× pessimistic because they were never re-run after the net
changed. ▶ So budget ~2 h, but **ASK BEFORE STARTING** (`user_runtime_prefs`), and note
ConvNeXt is the heaviest per-epoch net in the book.

⚠ Clear `.lake/build/convnext_adam_ckpt_xla*` first and between probes, or set
`LEAN_MLIR_CKPT_TAG`.

### 2. ⭐⭐ DO NOT repeat ch6's "representative scale" caveat — ConvNeXt has the full fold

**`convNextForwardTCh_has_vjp` (`Proofs/Architectures/ConvNeXtFullT.lean:270`), with
`convNextForwardTCh_has_vjp_correct` at :341. 0 sorries in the file.** It has the full
per-block ladder underneath it too: `cnxBodyWith_has_vjp`, `cnxBlockChW_has_vjp`,
`convNextStageChK_has_vjp`, `cnxDownChW_has_vjp`.

And it can be a **global** statement for the same reason EfficientNet's can: GELU is smooth
and LayerNorm has no running buffers or kink. So ch8 gets ch7's strong sentence, not ch6's
apology. ▶ **The three-way split now reads: ENet and ConvNeXt have full-depth global folds;
MobileNetV2 is the only one still at representative depth, and its relu6 kink is why (see
the TODO above).** If ch8's prose currently carries a representative-scale caveat, delete
it — it is wrong.

### 3. The fiction to replace (real line numbers, not offsets)

| line | what is there now | replace with |
|---|---|---|
| 8005 | `def convNextTiny : NetSpec` | `convnextVerified : VerifiedNetSpec`, §5.3 shape |
| 8023 | `def convNextTinyConfig : TrainConfig` | `VerifiedConfig` + `trainAdamSched` entry |
| 8036 | `convNextTiny.train convNextTinyConfig` | ⚠ `.train` again — `trainAdamSched` is the real entry |
| 8057 | `$ IREE_BACKEND=rocm IREE_CHIP=gfx1100` log | the §8.1 capture. **ch8's last IREE holdout** |
| 8175 | `def convNeXtTinyImagenet : NetSpec` | keep as the labelled phase-2 reference, **but diff it against `jax/MainConvNeXtImagenet.lean` first** |
| 8193 | `def convNeXtTinyImagenetConfig : TrainConfig` | ditto |
| 8390 | "`.train` entry point is identical; every row is a one-line `NetSpec` or `TrainConfig` change" | the verified vocabulary, as ch7's now reads |
| 7910 | `\phasethreenote` | delete once §8.x's numbers are XLA (uses 2 → 1) |

⚠ **§4a-quinquies' hardest-won lesson applies to row 8175/8193**: this doc told ch7 to
"keep the phase-2 listing as the reference" and it had silently drifted from the file it
claimed to mirror. **Diff the book's listing against `jax/MainConvNeXtImagenet.lean` line
by line before keeping it.** That file today has `learningRate := 2.5e-4`, `epochs := 80`,
`cosineDecay := true`, `gradClipNorm := 1.0`, `useEMA := true`, `dropPath := 0.1`, and a
`convNeXtTinyImagenetConfigFull` at `epochs := 300`.

### 4. ⭐ Things ch7 had to fix that ch8 already has right — VERIFY, then move on

- **Parameter count matches.** `convnextVerified` derives **180 tensors / 27,826,282
  scalars**, and the book's comparison table already says 27.83M. (`convnextImagenetVerified`
  is 28,587,592.) ch7's was 78% wrong; this one is fine.
- **`convnextImagenetConfig.epochs := 300` already matches** the phase-2 `Full` tier, so the
  match-phase-2 rule is satisfied out of the box. ⚠ But ch8 prints **two** bolded tiers
  (80 ep → 78.13/94.05, 300 ep → 81.10/95.37). Decide which is the chapter's headline and
  make sure 300 is it, or the config is matching the wrong one.
- **The DP collectives are gated, both ways**: `tests/TestConvNeXtDpCheck.lean` AND
  `tests/TestConvNeXtShardCheck.lean` exist (ConvNeXt is where `shard-check` was
  generalised from). ▶ Describe them as ch7's §7.4 now does — the dp-check pins the forward
  bit-exactly but hands both replicas the SAME rows, and shard-check is what closes the
  shard-offset hole. They are not one property.

### 5. ⚠ What ch8 still inherits and must fix

1. **`MainConvNeXtImagenet.lean` has NO `LEAN_MLIR_EPOCHS`** (0 hits, same as ViT). With
   `epochs := 300` committed, the net is **unprobeable** — you cannot ask for a short smoke
   run. Add it copying R50's spelling, exactly as ch7 did.
2. **`lakefile.lean:1828` says "ireeLink, because `convnext-verified-adam` is an IREE
   binary."** That premise is false — line 2120 puts it on `lowererLink`. This is the SAME
   stale docstring ch7 found on `efficientnet-adam-tie` (§4a-quinquies "Open"). ▶ Whether
   the docstring is stale or the tie runs on the wrong backend can only be settled by
   RUNNING the gate.
3. **The variant markers are denser here than anywhere else**: `wx`, `clip`, `drop`, `ema`,
   `dp`, in combinations up to `convnextin_adamdpwxclipdrop`. ▶ Before trusting any driver's
   variant predicate, check it against `tests/TestVariantPredicates.lean`'s rule — **with N
   markers the collisions are between PAIRS, so test every CONCATENATION.** ch7's live bug
   was exactly this class and the test file never gated a driver because each driver
   hand-rolls its own copy of the predicate.
4. **ch8 owns the second of the three cumulative comparison tables.** Its ConvNeXt row reads
   `27.83M / 790 KB / 2030 ms / 13.3 h / 84.94%`. Params are right; the other four are
   phase-3 IREE. Re-measure while you are in the chapter. The other three rows are known:
   R34 729 KB / 220 ms / 1.5 h / 89.71%, MNv2 1,047 KB / 90 ms / 35 min / 89.25%, ENet-B0
   1,316 KB / 103 ms / 41 min / 89.96%.

### 6. Verification, in order

`scripts/verify_excerpt.py` over EVERY chapter → `latexmk` 0 errors AND 0 undefined refs →
`scripts/measure_prose.py` to 0/0 → typecheck any Lean touched → **never a number you did
not measure**. ⚠ And re-measure the chapters you did NOT touch: ch7's TODO note silently
regressed ch6 from 0 to 2 prose semicolons, caught only by re-running the script on ch6.

---

## 4a-ter. Chapter 7 scoping (superseded by §4a-quinquies, kept for what it got right)

**Read ch5 §5.3 (`sec:verified_trainer_pattern`) and ch6 first — they are the mold, and
ch6 is the most recent worked example.** Baseline: 48 em-dashes, 12 prose semicolons,
max sentence 81 w. ch7 is `content.tex` 6662–7413.

▶ **This chapter is mostly a RUN plus a COPY.** The pattern is settled, the code exists,
and ch7 is in a *better* starting position than ch6 was. What follows is the whole job.

### 1. §7.1 "Run it first" — one command, ~1.6 h

```bash
lake build efficientnet-verified-adam
CUDA_VISIBLE_DEVICES=0 PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
  ./.lake/build/bin/efficientnet-verified-adam data \
  > runs/<date>-enet-imagenette-xla-cuda/enet-imagenette-xla.log 2>&1
```

`efficientnet-verified-adam` goes through `trainAdamSched` (80 ep, bs 32, AdamW 1e-3,
cosine + 3-ep warmup, 49 BN layers), so it threads BN properly and needs no plumbing.
lakefile's own bench row (line ~2937) says **XLA 1h34m** against IREE 6.2 h.
⚠ **Clear `.lake/build/efficientnet_adam_ckpt_xla*` first** and between any probes.
⚠ Verify every quoted log line against the source before committing the excerpt.

### 2. The fiction to replace (offsets are +lines from `\chapter{EfficientNet}`)

| where | what is there now | replace with |
|---|---|---|
| +262 | `def efficientNetB0 : NetSpec` | `efficientnetVerified : VerifiedNetSpec`, §5.3 shape |
| +281 | `def efficientNetB0Config : TrainConfig` | `VerifiedConfig` + `trainAdamSched` entry point |
| +312 | `$ IREE_BACKEND=rocm IREE_CHIP=gfx1100` log | the §7.1 capture, or a pointer to it |
| +475 | `def efficientNetB0Imagenet : NetSpec` | keep as the labelled **phase-2** reference |
| +494 | `def efficientNetB0ImagenetConfig` | ditto |
| +172 | `\phasethreenote` | delete once §7.3's numbers are XLA (markers 3 → 2) |

⚠ **Check the prose AROUND each listing, not just the listing** — that is what bit ch6,
where the surrounding paragraph described a constructor signature from the phase-2 type.

### 3. §7.x ImageNet → a phase-4 subsection at the END

Same move as ch6: phase-2 material stays where it is, and a
`\subsection*{Phase 4: ...}` with its own `\label` goes immediately before the
recipe-diff subsection, mirroring `sec:r34_phase4` / `sec:mnv2_phase4` / `sec:mnv4_phase4`.

⭐⭐ **AND HERE ch7 IS BETTER OFF THAN EVERY CHAPTER SO FAR: the full paper recipe is
ALREADY RENDERED, DATA-PARALLEL, AND TIED.** `efficientnetin_emarmsdp64dropdo_train_step.mlir`
is EMA + RMSProp + DP@64 + drop-path + dropout — the paper recipe end to end, on four
cards. ENet's collectives ARE tied (`shard-check` has an `efficientnet` row,
`tests/TestEfficientNetDpCheck.lean` exists), unlike MNv4's. So **ch7's phase-4 row can be
MEASURED rather than estimated** — it is the first chapter where that is true.

Phase-2 target to match: **76.80% top-1 / 93.26% top-5**, 350 epochs RMSProp, ~8.6 min/ep,
~55.5 h on 4× 4060 Ti (content.tex ~7216, ~7295). Against B0's paper 77.1 / 93.3.
⚠ That is a ~2-day run. **ASK BEFORE STARTING IT.**

### 4. ⚠ Two gaps ch7 inherits, both of which ch6 hit and fixed

1. **`efficientnetImagenetConfig.epochs := 80`, but the phase-2 tier it must reproduce is
   350.** Apply the match-phase-2 rule (see the MNv2 entry above): a phase-4 config must
   carry the epoch count of the phase-2 tier its chapter prints, or the run answers a
   question nobody asked. ⚠ Note the driver's 80 is *also* a real tier ENet documents, so
   decide deliberately rather than by reflex, and say which in the listing.
2. **`MainEfficientNetImagenet.lean` has NO `LEAN_MLIR_EPOCHS`** (0 hits; R50 and MNv4
   have it, MNv2 got it in `092001c`). Without it a 350-epoch config is **unprobeable** —
   exactly the trap that cost ten minutes on MNv2. Add it in the same commit as any epoch
   change, copying R50's spelling.

### 5. The stale cross-chapter table

ch7 owns the **first** of the three cumulative comparison tables whose ResNet-34 row still
reads `518 KB / 1400 ms / 9.5 h / 90.29%`. All three now carry a provenance note instead of
correct numbers. Re-measure ENet's own row while you are in the chapter, and the R34 and
MNv2 rows are already known: R34 729 KB / 220 ms / ~1.5 h / 89.71%, MNv2 1,047 KB / 90 ms /
35 min / 89.25%.

### 6. ⚠ ch7-specific trap: this is the EMA chapter

The EMA warmup defect is ch7's own material (content.tex ~7290 discusses it) and it is in
memory as `ema_warmup_bug`: a shadow seeded at random init and decayed away leaves `d^t` of
that init behind, which puts the average at **chance** rather than merely behind on short or
grad-accumulated runs, and **pre-fix `.bin` checkpoints hold poisoned weights**. Do not
re-derive it, do not trust an old ENet EMA checkpoint, and note that the `emarms` variant
marker collisions (`planning/ema.md`) are the reason the markers are spelled `drop` and `do`
rather than `sd` and `dropout`.

### 7. Verification, in order

Same as §5: latexmk 0 errors → 0 em-dash / 0 prose semicolons in prose (script in §3, and
mind its two over-counting bugs) → typecheck any Lean touched → regen/shim gates if any
artifact or spec moved → **never a number you did not measure**.

**The ch6 material below is kept for the parts still unused.**

## 4a-ter-old. Chapter 6, MobileNetV2 (scoping, now superseded)

51 em-dashes, 10 prose semicolons, max sentence 87 w. Carries one
`\imagenetphasenote`. Smaller prose job than ch4 or ch5.

**Apply the §5.3 pattern.** The chapter still prints `mobilenetV2Imagenet :
NetSpec` + `TrainConfig`; the real spec is `mobilenetv2ImagenetVerified`
(slug `mobilenetv2in`) and the exe is `mobilenetv2-imagenet-verified`, already on
`lowererLink` and building.

⭐ **There is parked work for exactly this.** Two tags hold a first pass at ch6–9
that was dropped on 2026-08-12 to nail the pattern on ch5 first:

- `parked/ch6-9-verified-trainers` — verified listings + per-chapter schedule
  tables with a blank 2×7900 XTX row for estimates
- `parked/mnv2-enet-350ep` — MNv2/ENet costed at the 350-epoch schedule

`git cherry-pick` them, then reconcile against §5.3's shape, which is newer and
better than what those commits used.

⚠ **No verified ImageNet run exists for MNv2, ENet, ConvNeXt or ViT.** All four
trainers build; none has been run to completion. So a ported chapter shows a
verified trainer above phase-2 JAX numbers, which is honest only if the chapter
says so. The parked commits did say so.

Measured schedule, 4×4060 Ti at eight loader workers (2026-08-11):

| net | ms/step | min/ep | 350 ep |
|---|---|---|---|
| MobileNetV2 | 188 | 15.7 | ~91.5 h (3.8 d) |
| EfficientNet-B0 | 371 | 30.9 | ~180.5 h (7.5 d) |
| ConvNeXt-T | 223 | 37.2 | ~186 h (7.8 d) at 300 ep |
| ViT-Tiny | 287 | 12.0 | ~60 h (2.5 d) at 300 ep |

### ⭐ And while in this chapter: MNv4 needs the R50 treatment

`mobilenetv4Verified` is **Conv-S on Imagenette** (10 classes, slug `mnv4`), and
that is all there is: renders `mnv4_{adam_train_step,fwd,fwd_eval}`, exe
`mobilenetv4-verified-adam` (80 ep, bs32). **There is no ImageNet spec, render or
driver.** Building one is the same three steps R50 just took:

1. `mnv4ImagenetVerified : VerifiedNetSpec` — slug `mnv4in`, `nClasses := 1000`,
   `data := .imagenet`, its own `shimScript`, `#guard`s pinning it to the
   10-class spec at everything but the head.
2. The renders, as `#eval`s in the MNv4 renderer, exactly as
   `resnet50in_{mom256,momdp64}` were two lines in `ResNet50RenderB.lean`.
   ⚠ Emit with `lake build <module>`, NOT `lake env lean` — that file segfaults
   under `lake env lean` at baseline (rc 139), which is pre-existing.
3. `mnv4-imagenet-verified` exe + driver with the `LEAN_MLIR_EPOCHS` knob.

⚠⚠ **The reference is Conv-M, the repo's spec is Conv-S.** The 100-epoch JAX
reference (75.51%) lives OUTSIDE the repo at `/home/skoonce/mnv4_convm_100ep`, so
repo searches miss it, and it is a **different network** from what
`mobilenetv4Verified` renders. Decide which of the two the ImageNet spec should
be before writing it, and do not quote the Conv-M number against a Conv-S run.

---

## 4a-old. The original ch4 scoping (kept for the numbers it got right)

**The prose was the big job here**, much bigger than ch2 or ch3:

| chapter | em-dashes | prose-semicolon lines | max sentence |
|---|---|---|---|
| ch2 (was) | 25 | 11 | — |
| ch3 (was) | 32 | 22 | 92 w |
| **ch4 (now)** | **60** | **82** | **91 w** |

⚠ Measure with the script in §3, and **exclude theorem/proof/definition
bodies** — in ch3, 35 of 71 sites were inside them and are out of scope.
⚠ `\;=\;` and `\;+\;` are LaTeX math spacing, not prose semicolons. A naive
`count(';')` counts them; `(?<!\\);` does not.

**The measured work is already done.** The six-arm ablation was re-run on
XLA on 2026-08-11 and lives in
`runs/2026-08-11-cifar8-6arm-xla-cuda/cifar8-6arm-xla.log` — 40 epochs each,
`lake run cifar`, RTX 4060 Ti:

| arm | BN | top-1 | top-5 | final loss |
|---|---|---|---|---|
| sgdsched | — | 72.47 | 97.57 | 0.5463 |
| sgdsched | ✓ | 74.49 | 97.83 | 0.4029 |
| momentum | — | **77.30** | 98.24 | 0.3639 |
| momentum | ✓ | 77.11 | **98.33** | 0.2958 |
| adam | — | 73.98 | 98.21 | 0.4941 |
| adam | ✓ | 73.80 | 97.72 | 0.4014 |

What the data says, for the prose: momentum wins by ~3 points over both SGD
and AdamW; BN helps plain SGD (+2.02) but is a wash under momentum (−0.19)
and AdamW (−0.18), while reaching a markedly lower *training* loss in every
pair. So BN fits harder without generalising better at this scale.

⚠⚠ **Run-to-run spread is ~1 point on these arms** — `bn-adam` measured
72.84% and 73.80% on the same seed and schedule. So the BN deltas on the
momentum and adam rows are inside the noise and must NOT be reported as
ordered. The momentum advantage is well outside it. If the chapter wants to
rank the BN pairs, that needs n≥3 (see §3c).

**Steps:**

1. `\section{Run it first}` after the first prosesection, from a fresh
   `lake run cifar` capture or the log above.
2. Voice pass, prose only.
3. Re-point any figure/table at the XLA numbers; rewrite the interpretive
   sentences to match, since several will move.
4. Remove ch4's `\phasethreenote` once its numbers are XLA (markers 6 → 5).
5. `grep` the *other* chapters for numbers you changed. Deleting ch2's 98.57%
   left **chapter 1** quoting it 1,000 lines away.

---

## 3c. ⚠⚠ REPRODUCIBILITY — the lesson that cost the most time

**Conv nets do not reproduce run to run on CUDA.** `tests/prefetch_tie.sh`'s
header records why: XLA picks convolution algorithms **per process**, so two
runs of an identical command differ. Dense-only graphs (the MLP) are exact.

Measured on the CNN head sweep, five passes:

    d=512   96.66  97.94  98.49  98.58  98.66     range 2.00
    d=4096  93.52  98.08  98.75  98.79  98.81     range 5.29
    other eight widths                            range <= 0.4

▶ **A single pass twice led to a wrong conclusion this session.** Pass 1 said
d=4096 scored 93.52% against the book's 98.96%, and it was one commit away
from being written up as a refuted number. Re-running gave 98.76%. The truth
is bimodality: one bad draw in five, four clustered near 98.7.

**Rules that follow:**
- Any conv-net number that a claim rests on: **n≥3, plot the median.** Means
  get dragged by the outlier and draw a fake dip exactly where the net is
  least stable.
- State n and the observed range in the caption.
- The MLP needs n=1. Say which regime you are in.
- **Sweeps are parallel.** Four passes on GPUs 0/2/3/4 cost the wall time of
  one. Avoid 1 and 5 (PCIe AER, per `scripts/jobs/*.conf`).
- ⚠ Do not drop a point because it misbehaved. Dropping inconvenient data is
  how the unreproducible numbers got into the book. Keep it, plot the median,
  and say it is bimodal.

---

## 4b. Measured experiments that need re-running on XLA

These are sections whose numbers came off the old path. Brett wants XLA
versions. They are not copy edits — each needs GPU time, and each has
**interpretive prose quoting specific values**, so the analysis around
the table has to be rewritten with the new numbers, not just the table.

### 2.3 "Return on width" (`content.tex` ~2378)

A 10-point sweep, $d \in \{8, 16, \ldots, 4096\}$ for
$784{\to}d{\to}d{\to}10$, 12-epoch SGD each, plotted as accuracy vs
width. Driver: `mnist-mlp-grid <d1> <d2> [epochs]`.

Already on the verified renderer (the prose says so), so this is a
lowerer re-measurement, not a path change. **Cheap** — the biggest point
is d=4096 and the smallest is d=8.

Prose that quotes numbers and must move with them: the knee annotation
("knee ≈ 64: 97.2% at 55K params", drawn as a `\node` in the
`tikzpicture`), "$64\times$ the width and $364\times$ the parameters buys
only $+0.8$", and the $784{\to}8{\to}4096{\to}10$ collapse-to-chance
(9.8%) aside.

### 3.2 "Example: MNIST 2D CNN" (`content.tex` ~2889)

Currently runs the **unverified** `ablation cnn-nobn-sgd` through IREE at
~39 s/epoch, reporting 98.98%. Same problem chapter 1 had: theorems about
one program, log from another.

Port it the way chapter 1 and 2 were: `mnist-cnn-verified` on XLA, a
"Run it first" section, `cnnVerified` as a `VerifiedNetSpec`, the
spec-to-proof tie, the whole driver. **Cheap.**

Note the CNN is one of the two nets that never completes on ROCm, so this
genuinely could not be done here.

### 5.5 "Ablation: what each ingredient contributes" (`content.tex` ~4735)

**Do not re-run this. Label it instead.**

Seven runs — full recipe plus six leave-one-out variants — on ResNet-34 /
Imagenette at 80 epochs each. That is hours of GPU time, and it would buy
provenance hygiene and nothing scientific: both lowerers consume the
*same* proven StableHLO, so the deltas this section measures are
lowerer-independent by construction. What changes between IREE and XLA is
wall-clock, and this section does not report wall-clock.

So mark it rather than repeat it. Add `\phasethreenote` at the top of the
section, which already says exactly the right thing:

> *Which trainer is this?* This demo runs on the phase-3 verified-IREE
> path — the proven graph, compiled by IREE and executed on the GPU —
> which is what the numbers below were measured on. It has a phase-4
> (PJRT) peer already built, and phase 4 is where it is headed.

That macro's whole purpose is stated in its own source comment: the set
of these markers is the map of what phase 4 still has to absorb. Tagging
5.5 puts it honestly on that map instead of silently implying the numbers
came off the current default. Same treatment for any other expensive
measured section where the conclusion is lowerer-independent.

Two caveats for whoever does this:

- The macro's prose contains em-dashes, so it is due the same voice pass
  as everything else. Fix it at the definition (`content.tex` ~21) and
  every use benefits. `\imagenetphasenote` beside it has the same problem.
- If you *do* eventually re-run it, the whole section is quantitative and
  every claim moves with the table: "augmentation is by far the largest
  single contribution ($-7.58$, more than $2\times$ any other knob)",
  "cosine decay and Adam are roughly tied ($-3.48$ and $-3.25$)", and
  "sum of all six leave-one-out deltas: $-17.38$ points". Those three
  "observations worth naming" are conclusions, not decoration — if the
  ordering changes they get rewritten to match the data rather than
  preserved.

---

## 5. Verification discipline

Non-negotiable, in order:

1. `cd blueprint/src && latexmk -xelatex -interaction=nonstopmode print.tex`
   → **0 errors** (`grep -ac '^!' print.log`).
2. Re-run the trainer after any code change and diff the epoch
   trajectory against the previous run. The `mkSession` refactor was
   accepted only because the trajectory came back bit-identical.
3. If you touch Lean, typecheck every file you touched:
   `for f in $(git diff --name-only -- tests/); do lake env lean "$f"; done`
4. Never put a number on the page you did not measure. If a run cannot be
   captured, leave the section out and say so — do not synthesise a log.

### ⚠⚠ Three bugs the chapter work uncovered (2026-08-12), all found by RUNNING

Every one was invisible to reading and to `lake build`. Two printed
plausible-looking output; one refused loudly. The lesson is the same in all
three: **run the binary the chapter claims to document, before documenting it.**

1. **The loss slot was driver-wide, should be per-render** (`ff1ef3d`).
   `3f404e8` taught `VerifiedNet.train` to append a `%loss` destination
   unconditionally, but only `mlp` and `cnn` were re-rendered to return one.
   Nine nets — `resnet34`, `cifar8`, `cifar8_bn`, `cifar`, `cifar_bn`,
   `mobilenetv2`, `efficientnet`, `convnext`, `vit` — could not run at all. The
   G4 arity gate refused every one, which is exactly what it is for. Fixed with
   `VerifiedNetSpec.lossSlot`.
2. **One checkpoint path per (net, variant, backend)**, so §3c's mandated
   parallel sweeps silently trained NOTHING: concurrent passes clobber each
   other, and a later pass resumes from an earlier one's finished epoch 40.
   Fixed with `LEAN_MLIR_CKPT_TAG`. ▶ Also clear stale
   `.lake/build/<slug>_*_ckpt_xla*.bin{,.epoch}` before any re-measurement.
3. **`.train` cannot evaluate a BN net** (`0be3357`, documented not fixed).
   Running-stat threading lives ONLY in `trainAdamSched`. `resnet34-verified`,
   `mobilenetv2-verified` and `efficientnet-verified` all report **exactly
   chance**, byte-identical every epoch (390/3925 = 9.936306%), because they
   evaluate through `@<slug>_fwd` without statistics anyone computed. Exactly the
   three nets with `bnChannels`. ▶ The fix is teaching `.train` the same
   threading; it wants its own commit and its own trajectory diff.

⚠ Also fixed: the epoch line no longer prints `loss = 0.000000` for a net with no
loss slot, it omits the field. A fabricated zero in a captured log is the one
thing this book cannot ship.

### ⚠⚠ Verify hand-assembled log excerpts against the source — THIS IS NOW A SCRIPT, RUN IT

Writing §5.1 I **fabricated two log lines** — epoch 4's loss and epoch 76's —
while hand-eliding the middle of a run, and caught them only by diffing the
excerpt against the log before committing. Elision is exactly where invented
numbers get in. **A third one got past that pass and shipped**, and two more chapters
carried stale lines; see §4a-quinquies. So this is no longer a habit, it is a gate:

```bash
scripts/verify_excerpt.py blueprint/src/content.tex <logfile> <tex-start> <tex-end>
```

It checks every quoted `Epoch n/N:` / `epoch n:` / `[pjrt_ffi]` / `done (` line, rejoins
the book's wrapped continuation lines before comparing, and exits non-zero on any miss.
Run it over **every** chapter, not just the one you touched.

⚠ It has two failure modes to catch and only one is fabrication. The other is **stale
carryover**: a line that was genuine when pasted and was invalidated by a later re-capture.
Nothing about eliding draws your eye to those, and they survive any number of prose passes.

### Traps

- **`\subsection*{Code: the full training program}` appears in both ch2
  and ch3.** Anchor edits by line number, not by string match.
- **Scripted call-site edits across newlines eat code.** A regex that
  skips whitespace to find "the next argument" will happily consume the
  next statement's `let`. Restrict to same-line or a continuation line
  holding only a string.
- **The springer branch.** `git rebase main springer` leaves you *on*
  springer. Two commits landed there by accident this session. Check
  `git branch --show-current` before committing.
- **Squashing main breaks springer's rebase**, since it carries the
  pre-squash commits. Rebuild it instead: `git branch -f springer main`
  then cherry-pick its three own commits.

---

## 6. The open question: Appendix C

Not a copy pass. Appendix C argues trust by runtime size — IREE at 1.3 MB
against the XLA plugin's 505 MB, "the fast path asks you to trust a great
deal more unverified code than the small one does." That was right when
IREE was the default. The book now recommends XLA everywhere, so the
appendix argues against the book's own choice, and a reader who follows
the setup instructions then reads the trust accounting finds it
condemning what they just installed.

Three ways out, and it needs brett's decision, not ours:

1. Size was never the right metric — what you trust is whether the
   lowerer preserves StableHLO semantics, neither is verified, and 400× is
   a red herring the appendix should retire.
2. Size still counts and XLA is a deliberate trade — keep the number,
   state plainly that the default buys speed at a real cost in trusted
   surface, and that IREE remains the smaller-TCB option.
3. Something else.

**Ask before rewriting it.**

---

## 7. Known-stale, book-wide

- ✅ **Chapters 1–7 are IREE-free** as of 2026-08-12. ch7's holdout was its Results block's
  `IREE_BACKEND=rocm … gfx1100` log, replaced by the XLA capture in `sec:enet_runit`; the
  chapter now has 0 hits for `IREE|vmfb|gfx1100`. ch6's last holdout had also contained a
  **fabricated line** (`Epoch 2/80: loss=(dropping) lr=0.000667`). The two mentions left in
  ch5 are legitimate (PJRT implements "the same C surface as the IREE shim"; gfx1100 in the
  ROCm fault note).
  ▶ **ch8 is next.** Note the fabricated-line problem was NOT confined to IREE blocks —
  see §4a-quinquies, where three chapters' XLA captures were also wrong.
- ⚠⚠ **`latexmk` AND THE PDF YOU READ ARE DIFFERENT FILES.** §5's gate builds
  `blueprint/src/print.pdf` in place. The copy that gets opened is
  `blueprint/lean4-mlir-blueprint.pdf`, and `blueprint/README.md` says the intended
  command is `leanblueprint pdf`, which writes somewhere else again. Measured
  2026-08-12: the read copy had drifted **seven days** while four rounds of "latexmk 0
  errors" were all true and none of them visible. ▶ Either point §5's gate at
  `leanblueprint pdf`, or copy `src/print.pdf` over the other two after building. Both
  are build outputs and are deliberately left untracked.
- ⚠ **98 targets are still on `ireeLink`** against 31 on `lowererLink`. The swap
  is one line per target and the migration comment in `lakefile.lean` says so,
  but ▶ **swapping the link does not mean the binary works** — that is how the
  loss-slot and BN-threading bugs surfaced. Port, then RUN.
- ⚠ **`VerifiedNetSpec.blurb` carries `%LOWERER%`** and is resolved by
  `VerifiedNet.printBlurb`. Never `IO.println net.blurb` — that was the bug where
  four print sites announced IREE while training on XLA.
- ~74 `iree`/`vmfb` references remain outside chapters 1–2, concentrated
  in appendix B (18) and appendix C (13). Brett wants IREE gone from the
  book; removing it from the *repo* is a separate and much larger
  decision (`ffi/iree_ffi.c` is 765 lines, `lakefile.lean` has 226
  references and 3 `lake run *-iree` scripts).
- ~~`IreeSession` is still the session type name~~ ✅ **DONE** (`1d34920`):
  renamed `LowererSession` across 269 sites, and chapter 1's paragraph
  apologising for the name is deleted. ▶ Still IREE-named and left for a
  follow-up: the module file `LeanMlir/IreeRuntime.lean` (7 imports + the
  lakefile), and **31 driver docstrings** that still say
  `Run (GPU): IREE_BACKEND=rocm ...` when XLA is the default.
- Docker references: 0 remaining. Done.
