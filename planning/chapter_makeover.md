# Chapter makeover: porting a chapter to the verified XLA path

**Who this is for.** An agent working on a CUDA box, picking up chapter 2
and then the rest. Chapter 1 is done and is the worked example — read it
in `blueprint/src/content.tex` before starting, it is the spec.

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
| On Verification (app C) | 19.3 | 28 | **needs an argument rethink, see §6** |
| Data availability (app A) | 17.8 | 14 | |
| Getting started (app B) | 17.0 | 8 | flourishes cut, joinery not |
| EfficientNet | 15.8 | 20 | |
| MobileNetV2 | 14.4 | 22 | |
| Bestiary | 12.7 | 70 | |
| ResNet-34 (ch5) | 12.6 | 33 | **▶ NEXT** (57 em-dash / 30 semi raw count) |
| ConvNeXt | 11.9 | 31 | |
| Vision Transformer | 8.7 | 59 | |

Watch the `max sentence length` too — several chapters have single
sentences over 100 words (MNIST 2D CNN has one at 133).

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

## 4a-bis. ▶▶ NEXT: Chapter 5, ResNet-34

57 em-dashes, 30 prose semicolons, comparable to ch4's 60/27. Carries one
`\phasethreenote` (§5.2 Imagenette) and two `\imagenetphasenote` (§5.7, §5.8).
§5.5's ablation is the one §4b says to **label, not re-run**. Note ch4's own text
calls this "the easiest chapter in the book."

⚠ Do not read "we're doing to JAX what we did to IREE" as retiring JAX. Confirmed
2026-08-12: the reference implementations stay, and the work is bringing PJRT to
**parity** with them. IREE was a swappable engine because both lowerers eat the
same proven StableHLO; JAX is the oracle `vjp_oracle` diffs 14 layer families
against.

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
