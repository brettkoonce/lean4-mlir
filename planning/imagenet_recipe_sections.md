# One shape for the ImageNet recipe sections (Chapters 5–9)

**Goal.** Every ImageNet section in the book reads the same way: the recipe, then one table with
the two paths as rows, one figure with both curves, one agreement paragraph. The protocol is stated
once, in Chapter 5, and the other chapters point at it in a sentence. Every (network, recipe) pair
that ran on both paths is a numbered subsection; "Side quest" is reserved for what is rendered but
not run.

**Status.** Not started as a pass. The shape was agreed 2026-10-24 (the Chapter 5 reorg was the
pilot of the headings; the inner shape has reached no chapter). Every pair the pass needs has
landed: R34 (476f7878), R50 2018 + A3 (dcd382fd, e6ffda75), MNv2 (7f61ba89), MNv4 100-ep
(2582b32d), B0 (c7307da4), ConvNeXt-T (62982e02), ViT-Ti (04c72553). Pending pairs (A2/A1, MNv4
500-ep, ConvNeXt-S/B, ViT-S/B) stay side quests and are not blocking.

**Why.** The three-beat (reference → verified → side by side) repeats seven times, each with the
learner's piece — the agreement — last, after two tables that say the same thing the side-by-side
table says again. The 2026-10-26 review (`planning/book_review_2026-10-26.md` §1.1, Review D) and
the copy audit of 2026-10-21 both read the repetition; the user's steer is "the whole book the same
treatment" as Chapter 5.

## 1. The template, per (network, recipe) subsection

```
\section{ImageNet recipe}
\subsection{<Net>: <the recipe's name>}          % one per pair that ran on both paths
  1. The recipe.   Lineage sentences (whose code, which rows are the paper's and which are its
                   successors'), then the diff-from-2018 table (today's "What X changes from the
                   2018 recipe", moved up from the end), then the reference config listing trimmed
                   to recipe rows (ViT's prose form is the model: no listing at all is acceptable
                   where the diff table carries every row). Disclosed-not-matched items in one
                   sentence.
  2. The result.   ONE table: rows = the two paths; columns = GPU, per epoch, total, top-1, top-5,
                   scored arm. No separate reference / verified tables.
  3. The curves.   ONE figure, both paths, top-1 (top-5 only where the chapter reads it). The
                   selected-epochs table stays only if the prose reads a window from it; otherwise
                   the figure carries it.
  4. Agreement.    ONE paragraph: the gap against the Wilson half-width, McNemar on the scored arm
                   (and the live arm where both exist), what the EMA is worth, the one-seed caveat,
                   and what genuinely differs between the paths (λ granularity, mask source). The
                   "held equal by construction" rows go: the recipe is stated once above, and both
                   paths run it by construction (the shim is generated from the reference config —
                   App C says so).
  5. Cost.         One sentence: the per-epoch ratio on the same cards. No feed/lowering
                   decomposition unless measured for that run.
\section{Side quest: ...}                          % rendered, not run: A2/A1, MNv4 500, S/B
```

Prose rules in force: one claim per sentence; no epoch narration (behaviour and its cause; per-epoch
numbers live in the table or the figure); numbers that drift live in tables.

## 2. The protocol, stated once

Chapter 5's R34 subsection keeps, and the others drop: what the two paths are (JAX traces the
generated program / `pretty(provenGraph)` through PJRT, no Python in the step); what is held equal
and why (same spec, same shim, same compiler, the shim generated from the reference config); how a
pair is scored (the final checkpoint through an fp32 forward on all 50,000; the Wilson half-width at
50k, ±0.34–0.38 across the book's accuracies; McNemar on the per-image bitmaps); what one seed per
arm can and cannot say. Later chapters: "Scored as §5.x scores its pairs." Today the "reference
first, then the verified path, then the two side by side" sentence appears 3×, "held equal by
construction" 9×, Wilson 13×, McNemar 7×.

## 3. Chapter by chapter — what is there and what moves

Census at 62982e02 (lines / tables / figures / listings in the ImageNet + side-quest span):

| ch | span | tables | figs | listings | subsections today | target subsections |
|---|---|---|---|---|---|---|
| 5 R34/R50 | 889 | 12 | 4 | 3 | R34 2018; R50 2018 + A3 (numbered) | **as is**: the pilot. Inner shape still three-beat → apply §1. Fold "A reasonable stopping point" (review D) into the R34 agreement paragraph. R50's grid stays one subsection with a two-block table. |
| 6 MNv2 | 894 | 11 | 4 | 6 | none; MNv4 is a "Side quest" with its own three-beat | **MobileNetV2: the TF-slim recipe**; **MobileNetV4-Conv-M: the paper recipe at 100 epochs** (it ran on both paths, so it is a pair, not a side quest). The 500-epoch tier is the side quest. `tab:mnv2_config_sources` becomes the recipe step's provenance table; the six listings drop to the recipe rows. |
| 7 B0 | 557 | 5 | 2 | 2 | none | **EfficientNet-B0: TensorFlow's recipe**. Already closest to the shape (one reference table, no verified table). `tab:enet_config_sources` as in Ch 6. |
| 8 CNX | 570 | 7 | 2 | 3 | none; S/B side quest | **ConvNeXt-T: the official recipe**. Landed 62982e02 on the three-beat; the recipe paragraph is already written once up front, so step 1 is mostly moving the diff table up. |
| 9 ViT | 552 | 6 | 2 | 3 | none; S/B side quest | **ViT-Tiny: the DeiT recipe**. The recipe is already prose, no diff-from-2018 table exists → add one (the template needs it; DeiT vs 2018 is the cleanest diff in the book). |

Common to all five: `tab:<net>_jax` + `tab:<net>_pjrt` + `tab:<net>_paths` → one `tab:<net>_pair`;
`fig:<net>_jax_curve` + `fig:<net>_paths_curve` → one `fig:<net>_curves` (the reference's top-5
curve goes unless the chapter reads it); `tab:<net>_paths_epochs` kept or dropped per §1.3.
Expected: roughly −10 tables, −5 figures, and the span down by a third; `book_floats --strict`
re-cites every figure by number, so no float is left uncited.

## 4. Appendix C: the scoreboard

`tab:track4_runners` gains three columns — reference top-1, verified top-1, Δ — and becomes the
book's single scoreboard (the "How this book is organized" chapter and the Introduction's thesis
paragraph point at it instead of quoting numbers). Rows: R34, R50-2018, R50-A3, MNv2, MNv4-100,
B0, CNX-T, ViT-Ti. The pending pairs are not rows. `book_counts` may want a regex if any prose
states the count of pairs.

## 4a. The cheap cuts, first (agreed 2026-10-29)

Three mechanical edits per section that need no new prose and remove most of the repetition. One
chapter per commit, Ch 8 first (its text is the freshest), then 5, 6, 7, 9; each with the preview
counts and the four gates. The structural pass (§1–§4) follows once these are in.

1. **The two run tables go.** The side-by-side table already carries both results; its differing
   rows take the GPU / per-epoch / total cells, and `tab:<net>_jax` + `tab:<net>_pjrt` are deleted.
   Prose that said "Table X is the run" points at the side-by-side table; the headline number
   stays in the prose where it was already quoted.
2. **The "held equal by construction" block goes.** The recipe is stated once per section and both
   paths run it by construction (the shim is generated from the reference config; App C says so).
   The side-by-side table keeps the rows that differ and the results.
3. **The reference-only curve goes.** It is the reference line of the two-path figure plus a top-5
   trace; `fig:<net>_jax_curve` is deleted and the chapter that reads top-5 (Ch 8) adds the trace
   to the pair figure if the prose still needs it. Anything the prose read off the deleted figure
   moves to the curves paragraph of the side-by-side.

## 5. Order

1. Chapter 5 inner shape (the pilot of the inner shape; R34 first, then the R50 grid). Writes the
   protocol paragraph the others will cite.
2. Chapter 6, both subsections (the full template on a chapter with a provenance table and a
   second pair).
3. Chapters 7, 8, 9 — one commit each, mechanical after 1–2.
4. Appendix C scoreboard + the front-matter pointers.
5. A read-through for the cross-chapter sentences the collapse leaves dangling (`book_xrefs.py
   --summary`), and the "Every chapter after this one uses the same six ingredients … the recipe
   stays" sentence in Chapter 5, which conflicts with the diff tables.

One chapter per commit; each commit carries the preview counts (changed paragraphs, errors,
overfull, pages) and the four gates (`book_listings --check`, `book_floats --strict`,
`book_counts`, `book_prose_idents`). Each chapter's numbers are re-read from its run dirs'
RESULTS.md at the time, not copied from the old tables.

## 6. Open

- Keep the selected-epochs table anywhere? Leaning: only in Ch 5 (R50's grid) and Ch 8 (the EMA
  climb), where the prose reads a window.
- The reference config listings: keep a trimmed listing (B0's and MNv2's are provenance, every
  line names its source) or go fully to prose as ViT does? Leaning: keep where a provenance table
  exists (Ch 6, 7), prose elsewhere.
- The side-quest TODO markers (A2/A1, MNv4 500, S/B ×2) stay as markers; this pass does not
  rewrite them beyond pointing at the new subsection names.

## 7. The Imagenette Results subsections (Chapters 5–9), same idea — superseded by §8 where they differ

**What is there.** Each chapter's `Example` section ends in a `Results` subsection (Ch 5 L6958,
Ch 6 L9217, Ch 7 L10852, Ch 8 L12418, Ch 9 L14748; ~250 lines in all at 62982e02) that tangles three
kinds of content, each written on the day that chapter's run landed:

- a repeated headline — every one opens by restating the number `Run it first` printed a section
  earlier ("§X has the run: 80 epochs … finishing at …");
- a cumulative table written three times — `tab:imagenette_three` (Ch 7), `_four` (Ch 8), `_five`
  (Ch 9), each caption pointing back at the first;
- log observations as bullets (17 across the five chapters): step time, the label-smoothing floor,
  BN counts, MLIR size, plus one-off asides (Ch 8's dated layer-scale paragraph at L12383, Ch 9's
  bit-exact reproduction). Most are about the column, not the chapter.

**The shape.** Separate the three and give each one home.

1. *The chapter's Example ends at the run.* `Run it first` keeps the transcript and the headline.
   `Results` shrinks to two sentences — the number, and what it says about this net at this data
   scale — with a forward pointer to the column; or it goes and those sentences close "The verified
   trainer".
2. *The column appears once, where it is complete.* Ch 9 already says "that completes the Part 1
   column": the five-row table lives there (one label, `tab:imagenette_five` or a new
   `tab:part1_column`), Ch 7 and Ch 8 point forward to it, and `_three` / `_four` go.
3. *The observations become the column's reading, one paragraph per column of the table, written
   once under it:* accuracy against parameters at 9,469 images (data scale, inductive bias; the ViT
   gap and the ConvNeXt gap are the same paragraph); step time against parameters (depthwise is
   memory-bound; the 7×7 and the LayerNorms); MLIR size against operation count (per-op emission;
   normalizations per block). The label-smoothing floor (0.500 at ε = 0.1, K = 10) is explained once,
   in Ch 5, and never again.
4. *Properties of a net move into its Architecture subsection:* 36 / 52 BN layers, MobileNetV2's
   three artifacts, SE's ops in the render, one norm per ConvNeXt block, ViT's 435 `dot_general`s
   and two convolutions (which is why it reproduces bit for bit). None is a result.
5. *The dated asides go, under the no-dev-history rule.* The layer-scale paragraph becomes one
   sentence about the property (a 10⁻⁶ layer scale delays every residual branch; on this schedule
   that costs about three points against starting at one; kept because it is the paper's and the
   reference's). The epoch-by-epoch comparison and the "until 2026-09-13" go; the headln run dir
   can stay as a citation.

**Spread.** The column quotes one seed per net except ResNet-34's five (±0.32), so every cross-net
difference is read against R34's spread alone. Five seeds for the other four is ~15 card-hours on
one 4060 Ti (24–74 min a run). Decide before writing the column's paragraphs: with the seeds the
column is a measurement; without, its paragraph says so in one sentence.

**Size.** ~250 lines → one ~80-line section at the end of Ch 9 plus five two-sentence closers.
Independent of the ImageNet pass; can go before or after it.

## 8. Decisions of 2026-10-29 (session 2)

Argued before editing. Where this section disagrees with §1–§7 above, this section wins.

**The one rule for both halves: anchor in full, then diffs.** Chapter 5 (ResNet-34) is the anchor for
both the Imagenette loop and the ImageNet recipe. It keeps its full shape. Chapters 6–9 say only
what differs from it.

**ImageNet (§1–§4a), amended:**

- Ch 5's R34 subsection keeps the reference / verified / side-by-side / stopping-point shape as is.
  "Imagenette demo → bigger data, same loop → ImageNet (2018)" is the chapter's narrative.
- In Ch 6–9 the `*_vs_2018` diff table **moves to the front** of the section: here is what changed
  from R34's recipe, here is the run, here is the number. R50's `tab:r50_a3_vs_2018` and MNv4's
  `tab:mnv4_vs_mnv2` fit the same order.
- The `tab:*_config_sources` tables (Ch 6 L9426, Ch 7 L11106) go. One sentence stays saying the
  config is paper-exact and where each value comes from.
- The run-level tables and the reference-only curve go (the §4a cuts); the three prose headings per
  pair collapse when only one table is left under them.
- The Ch 8 cuts of 2026-10-29 are parked as `git stash@{0}` ("ch8 imagenet cheap cuts") and as
  `scratchpad/ch8_imagenet_cuts.patch`; redo them with the diff-table move rather than popping
  the stash blind.
- Order of work: Imagenette pass first (below), then ImageNet Ch 6, 7, 8, 9.

**Imagenette (§7), as executed in the working tree on 2026-10-29:**

- `Run it first` keeps the transcript and the headline, and ends with one pointer to Results. Ch 6
  and Ch 7 lose the comparison paragraph that Results repeated; Ch 6's epoch-2 paragraph moves.
- `The verified trainer` carries the net's properties: Ch 5 gains the 36-BatchNorm paragraph in the
  shape Ch 6/7 already had; Ch 6 gains a `runningBN` paragraph (the third artifact, and why
  validation can fall during warmup while the loss drops, stated as behaviour and cause); Ch 8's
  layer-scale paragraph loses the date, the epoch-by-epoch pairs, the ones-init comparison and its
  run dir, and becomes a property: a superseded run is not cited in the curated text, even as a
  measurement (user, 2026-10-29).
- `Results` in every chapter is: one placement paragraph (this net against the previous ones, and
  the architectural reason), one "Where the time goes" paragraph (step time and its cause; MLIR
  size where it says something), nothing else. No headline repeated. Each says which row of the
  column it is.
- The column appears once: `tab:imagenette_five` in Ch 9, self-contained caption (seed note, recipe
  note that ViT runs at 3e-4 / 5-epoch warmup). `tab:imagenette_three` and `_four` are gone. Under
  the table, one paragraph reads the column (accuracy vs parameters, step time vs ops, render size
  vs ops per block, and the one-seed caveat).
- The label-smoothing floor is explained once, in Ch 5 Results, with "here and in every later
  chapter's run". Ch 6/7/8 no longer mention it.
- Kept as is: Ch 8 `Run it first`'s "two compile lines" paragraph (LN vs BN, read off the
  transcript), Ch 9's "It reproduces exactly" paragraph, Ch 6's scope note on the pointwise theorem.
- Not done: five seeds for the other four nets (§7 "Spread"); the column's paragraph says every row
  but R34's is one seed.

Net: 253 lines out, 168 in; gates green; preview 2026-10-29 evening.

**Landed 2026-10-29 (evening), all on main, unpushed at the time of writing:**

- df213b38 — the Imagenette pass, Ch 5–9 (above).
- f0cdbd9b — Ch 6 ImageNet + MNv4 side quest.
- 51174dee — Ch 7 ImageNet; `efficientNetB0ImagenetConfig` → 350 epochs and `mobilenetV2ImagenetConfig`
  → 420, the `*ConfigFull` names kept as aliases for the jobs/CI; both default trainers re-emitted
  (epoch count only; `_full` files and shims byte-identical). 04b85a34 — preview tool stats line.
- a17b9786 — Ch 8 ImageNet; `convNeXtTinyImagenetConfig` → 300 the same way. Supersedes the stash
  "ch8 imagenet cheap cuts" (stash@{0}), which can be dropped.
- 5c5f0525 — Ch 9 ImageNet, with the new `tab:vit_vs_2018`.
- 8c39ef98 — Ch 9: `tab:vit_vs_deit`, the recipe lever on a fixed architecture (the ViT paper's
  ImageNet-1k-only row is from scratch at 224 then a 384 fine-tune; disclosed in the table).
- 067b43e2 — Ch 5 R50 subsection, same shape with the 2×2 grid kept. Every ImageNet pair is now
  R34-in-full-then-diffs.

Two more rules came out of the review, now in memory: the curated text never cites a superseded run,
even as a measurement (the Ch 8 layer-scale paragraph); and a reference config carries one schedule,
the official one, with no short validation tier (the 80/90-epoch `default` tiers are gone from three
Lean files and the book). Pages 269 → 262, tables 86 → 76, figures 64 → 59 over the day.

Untouched by design: Ch 5's R34 anchor and the A2/A1 side quest; the ConvNeXt-S/B and ViT-S/B side quests (TODO runs; the ViT-B Lean config still has a `short` tier to drop when its section lands).
Still open from §6–§7: the Appendix C scoreboard (§4), five seeds for the Imagenette column.

**Queue item — five seeds for the Imagenette column (2026-10-29).** `tab:imagenette_five` has one
averaged row (R34, 89.99 ± 0.32 over five seeds from `runs/2026-09-12-r34-ablation-fp32-seeds/`) and
four single runs (MNv2, B0, CNX-T, ViT-Ti). Run the other four at seeds 2–5 (seed 1 is the printed
run) so every row is a mean with a spread and the under-a-point comparisons in each chapter's
Results read against a real interval. Driver: `scripts/sweeps/seed_sweep.sh` (SUITE=imagenette,
`LEAN_MLIR_SEED`), packed one job per card; or the four `*-verified-adam` binaries by hand with
`LEAN_MLIR_SEED=n`. Cost on one 4060 Ti per net, four more runs: MNv2 2.3 h, B0 2.7 h, CNX-T 4.9 h,
ViT-Ti 1.6 h (~12 card-hours; ~3 h on four cards). ViT-Ti is bit-reproducible across cards, so its
spread is the seed's alone. On landing: every row becomes mean ± half-width, the caption drops its
one-seed note, Ch 6–9 Results restate their comparisons against the interval, and the column
paragraph's one-seed caveat goes.
