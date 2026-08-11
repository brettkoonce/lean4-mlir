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
| MNIST: 1D MLP (ch2) | 0.0 | 0 | prose done, run pending |
| On Verification (app C) | 19.3 | 28 | **needs an argument rethink, see §6** |
| Data availability (app A) | 17.8 | 14 | |
| Getting started (app B) | 17.0 | 8 | flourishes cut, joinery not |
| EfficientNet | 15.8 | 20 | |
| MobileNetV2 | 14.4 | 22 | |
| CIFAR with BatchNorm | 14.2 | 59 | |
| Bestiary | 12.7 | 70 | |
| ResNet-34 | 12.6 | 33 | |
| ConvNeXt | 11.9 | 31 | |
| MNIST: 2D CNN | 9.8 | 39 | |
| Vision Transformer | 8.7 | 59 | |

Watch the `max sentence length` too — several chapters have single
sentences over 100 words (MNIST 2D CNN has one at 133).

---

## 4. Chapter 2: what is left

Done already (commit `7a9a1dd`):

- Prose voice pass, 25 em-dashes and 11 semicolons → 0 and 0
- New "The verified spec and program" subsection: `mlpVerified`, the
  spec-to-proof tie (`mlpVerified_has_vjp` over `denoteMLP
  mlpVerified.layers`), the whole 44-line driver
- Old listing relabelled "The earlier unverified path"
- Two factual fixes (the `.train` pointer, the Caveats trusted-surface bullet)
- The book's last Docker reference killed

**What you do:**

1. **Capture the run.** On CUDA:
   ```
   lake build mnist-mlp-verified
   CUDA_VISIBLE_DEVICES=0 .lake/build/bin/mnist-mlp-verified data \
     2>&1 | tee runs/<date>-mlp-verified-xla/mlp-verified-xla.log
   ```
   Expect 12 epochs, ~97.83% — that is what the CUDA control produced,
   bit-identically across six runs.
2. **Add §2.1 "Run it first"** per the recipe in §1, before
   `\section{The theorems}`. Copy chapter 1's section and adapt.
3. **Replace the historical results.** The current results block runs the
   *unverified* ablation runner through IREE at ~9.4 s/epoch and reports
   98.57%. Once §2.1 has a real log, demote or delete it the way chapter
   1 did. Note the loss plot (`tikzpicture`, per-epoch training loss)
   depends on those numbers — the verified trainer does **not** print
   per-batch loss, so either drop the plot or keep it labelled as the
   historical run. Brett's call; ask.
4. **Strip the remaining IREE.** 2 mentions left, both in the historical
   results verbatim (`IREE_BACKEND=rocm`, `Compiling vmfbs...`). They go
   with the results block.
5. **`NetSpec` → `VerifiedNetSpec`.** 6 hits, 2 are the substring inside
   `VerifiedNetSpec`; the rest are the historical listing.
6. **Remove `\phasethreenote`** (line ~2085) once the trainer completes a
   run. Its text claims the numbers were measured on the phase-3
   verified-IREE path, which stops being true. The macro's own comment
   says the set of these markers maps what phase 4 still has to absorb —
   9 chapters originally, 8 after chapter 1, 7 after this.

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
- `IreeSession` is still the session type name on the XLA path. Chapter 1
  says so out loud rather than pretending otherwise. It is on the rename
  list.
- Docker references: 0 remaining. Done.
