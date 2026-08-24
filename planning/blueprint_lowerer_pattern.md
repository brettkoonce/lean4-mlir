# blueprint_lowerer_pattern.md — converting the ImageNet chapters to the §5.7 shape

**Opened 2026-08-24.** §5.7 (ResNet-34 / ImageNet) was restructured on 2026-08-24 and is the
template every other ImageNet section should converge on. This is what the pattern *is*, why each
piece is there, and what converting each remaining chapter actually costs.

▶ §5.7 is the only worked example. Read it before converting anything — the shape is easier to
copy than to describe.

---

## 1. WHY THIS SHAPE, IN ONE PARAGRAPH

⭐ **With one backend, the Lean/proof apparatus is a fancy metaprogramming layer for Python.** The
proofs decorate a code generator and nothing checks the generator. With a *second, independent*
lowerer, the verified path becomes the artifact and JAX becomes a **reference oracle** — an
independent implementation the verified number is checked against. That is the whole argument for
phase 4, and the chapter layout should make a reader feel it without being told. Hence: the two
lowerers get equal, parallel billing, and a third section exists only to put them against each
other.

⚠ This argument is **not currently stated anywhere in the book.** A reader can finish chapter 5
thinking the verified path is a performance footnote. It belongs in the introduction or at the top
of "On Verification", and writing it is a prerequisite for the pattern reading as intentional
rather than as a layout quirk.

---

## 2. THE PATTERN

```
\section{ImageNet recipe}

  <intro>                     the net at ImageNet scale; the recipe, run twice
  <the recipe in full>        shared by both lowerers, stated once

  \prosesection{The PJRT lowerer}      ← leads. it is the artifact.
      what the lowerer is (C surface, PJRT, no Python at run time)
      the trainer listing + what makes it a claim (slug / #guards / shimScript)
      the run command
      results table   (hardware rows; TODO rows stay visible)
      curve

  \prosesection{The JAX lowerer}       ← the oracle
      where it lives, one paragraph
      precision/throughput note (why its column is faster)
      results table   (hardware rows)
      curve

  \prosesection{The two paths, side by side}
      ONE comparison table, metrics as rows, lowerers as columns
      the agreement claim, and what it is NOT evidence of
```

### 2a. The rules that fell out of doing it

1. ⭐⭐ **Every number has exactly one home.** Prose points at tables; it never restates them.
   This is not style. §5.7 carried "10.5 min/epoch, ~16 wall-clock hours" in prose for days after
   the tables said 8.8 and 14.8 — the re-run updated the table and nobody re-read the paragraph.
   A number in two places is a number that will disagree with itself.
   ▶ The one tolerated exception: a lowerer's own results table and the comparison table may both
   carry its headline accuracy, because each has to stand alone. Prose still doesn't.
2. ⭐ **Name the thing, not the project's history of it.** "PJRT lowerer" and "JAX lowerer"
   survive; "phase 2 / phase 3 / phase 4" date the book and mean nothing to a reader who did not
   watch it get built. §5.7 has **zero** phase-N references left.
3. ⭐ **Only the comparison section makes a claim.** Both lowerer sections are description. The
   assertion — the agreement, and the caveats on it — lives in one place.
4. **Cut the "how we got here".** "Earlier printings of this chapter asked whether…", "when this
   chapter was first written it did not…", "the earlier 30-epoch run is superseded but…" — all
   removed. Keep the *evidence* such passages carried (the 17,312 / 17,313 lowerer agreement
   survived; its archival framing did not).
5. **Cut training minutiae.** Checkpoint file sizes, param-group counts, microbenchmark
   corroboration of a speedup figure. Keep mechanism that explains behaviour; drop bookkeeping.
6. **A gap is a row, not a sentence.** An un-run configuration stays in the table as `---` or
   `\withheld`, marked `[TODO]`. Demoting it to a clause in a caption is how the 7900 XTX row
   nearly vanished.

### 2b. The degenerate case

A chapter with no completed verified run is **not** a different structure. It is:

```
  <intro> / <recipe>
  \prosesection{The PJRT lowerer}   ← the trainer, the run command, an all-TODO results table
  \prosesection{The JAX lowerer}    ← table + curve
  (no comparison section yet)
```

The comparison section appears when the second number does. That graceful degradation is the
evidence the pattern is right rather than fitted to §5.7.

---

## 3. WHAT EACH CHAPTER ACTUALLY NEEDS

Surveyed 2026-08-24 against `blueprint/src/content.tex`.

| chapter | ImageNet §: tbl / fig | verified material present? | `\imagenettimmnote` | conversion cost |
|---|---|---|---|---|
| **5 · ResNet-34** §5.7 | ✅ **done — the template** | yes, run, 74.14 | removed | — |
| **5 · ResNet-50** §5.8 | 3 / 2 | yes, run (77.91) | removed | **medium** — see §4 |
| **6 · MobileNetV2** | 3 / 1 | 21 refs, 7 "phase 4", 3 TODOs | present | medium |
| **6 · MNv4 side quest** | 3 / 1 | same | present | medium |
| **7 · EfficientNet** | 3 / 1 | 21 refs, 1 TODO | present | medium |
| **8 · ConvNeXt** | 3 / 1 | 20 refs, 1 TODO | present | medium |
| **9 · ViT** | 2 / 1 | 19 refs, 2 TODOs | present | medium |

⭐ **Every chapter already has PJRT material and a verified render on disk** (`mobilenetv2in`,
`efficientnetin`, `convnextin`, `vitin`, `mnv4in`). None of them needs new code to *have* a PJRT
lowerer section — the section can be written today, with its results table all-TODO where no run
exists. That is the cheap half of the work and it is worth doing before any GPU time.

⚠ **Whether each has a completed ImageNet-scale verified run is NOT established by this survey.**
Bold accuracies in a chapter may belong to the JAX path, the Imagenette demo, or the verified path
— the current layout does not distinguish them, which is itself an argument for the conversion.
▶ **Check per chapter before writing its comparison section.** Do not assume a number is a
verified number because it is in a chapter that mentions PJRT.

---

## 4. §5.8 IS THE REAL TEST, AND IT SHOULD GO FIRST

§5.7 had the easy shape: one net, one recipe, both lowerers run. §5.8 has **two recipes (2018 and
RSB-A3) × two lowerers**, and the grid is three-quarters filled:

| | 2018 | RSB-A3 |
|---|---|---|
| JAX | 76.95 | 78.26 |
| PJRT | ⏳ **not run (~46 h, measured)** | 77.91 |

Converting §5.8 answers the question the rest of the book depends on: **does "one section per
lowerer" survive when a section has to carry two recipes?** If it needs "one section per lowerer,
recipes as table rows", that is a refinement to fold back into this document before touching
chapters 6–9.

▶ The missing cell is a measured **~46 h** run and needs no build work — render, eval forward,
shim, and a post-C3/C4 binary all exist (`resnet50in_momdp64_train_step.mlir`, verified
2026-08-24). An epoch-1 checkpoint from the timing probe is sitting in `.lake/build/`, so a real
run resumes from epoch 1. ⚠ That checkpoint is only safe while the schedule stays 90 epochs;
at any other epoch count, clear it or the two cosines fuse silently.

---

## 5. SUGGESTED ORDER

1. **§5.8** — the two-recipe stress test. Do it before generalising. (Conversion is independent of
   the missing run; the grid can carry a TODO cell.)
2. **Write the oracle argument** (§1's ⚠) into the introduction or "On Verification". Cheap, and
   everything else reads better once a reader knows why two lowerers matter.
3. **Chapters 6–9, one at a time**, easiest first — ConvNeXt and EfficientNet have 1 TODO each;
   MobileNetV2 has 3 plus a side quest.
4. **Retire the phase-N vocabulary book-wide** once no ImageNet section uses it. `\imagenetphasenote`
   defines those terms and is the last thing to remove; it currently sits on 5 sections.

⚠ **Do not convert a chapter and re-run its net in the same pass.** §5.7's conversion was
interleaved with a live 37 h run and a re-measured ETA, and the result was three separate stale-number
corrections. Convert against the numbers you have; update the table when a run lands.

---

## 6. WHAT IS DELIBERATELY NOT IN THE PATTERN

* **Per-lowerer *Imagenette* sections.** The Imagenette demos are one path and pedagogical; leave
  them.
* **A book-wide results table.** Tempting, and it would immediately go stale against the per-chapter
  tables. Rule 1 applies at book scale too.
* **Symmetric code listings.** §5.7's PJRT section has two code blocks to JAX's zero, and that is
  correct: what the verified path *executes* is the claim, so its spec and its invocation are load
  bearing. The JAX lowerer only needs identifying.
