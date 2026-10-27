# platforms_ladder.md — one ladder, named by its commands, generated into the book

Started 2026-10-26, from the book review (planning/book_review_2026-10-26.md §1.6). The
platform *suite* is planning/platform_integration.md; this is the layer above it: how the
book, the README and PLATFORMS.md describe where each platform stands, and how a new platform
flows into all three without hand-typing.

## 0. The problem

Three numberings of one ladder:

| where | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| README tour table | `lake run mnist` | `cifar` | `imagenette` | `imagenet` |
| Appendix A, Table A-1 ("Track") | proofs only | MNIST / CIFAR / Imagenette ("2, 3") | — | ImageNet-1k |
| PLATFORMS.md / `support.tsv` | names only: suite, mnist, cifar, imagenette, imagenet | | | |

"Track 2" vs "Track 3" is defined nowhere in the book since Getting started was rewritten in
tiers (adb2e750); Appendix A's platforms table uses both ("Track 2 on record, and of Track 3 the
MNIST CNN …") and carries queue prose ("is the next run", "their full runs to come") that the
generated PLATFORMS.md cannot. The book's table is hand-typed and already disagrees with
`support.tsv` in both directions (the Mac's MNIST numbers are in the book, `-` in the tsv).

## 1. The ladder

Rungs are named by the command that climbs them; nothing in prose is "Track N" or "tier N".

| rung | what it is | needs | time |
|---|---|---|---|
| proofs | `lake build Proofs` / `Certs`; `tests/comparator/run.sh` | Lean, no GPU | minutes to an hour |
| suite | `scripts/platform/check.sh` (its tiers 0–2 stay internal to the script) | the plugin, the shim; no Lean build of the proof tree | minutes |
| mnist | `lake run mnist` | one plugin file | ~1 min |
| cifar | `lake run cifar` | " | ~6 min (re-time: the tier gained `cifar8wb-bbn-ablation` on 10-24) |
| imagenette | `lake run imagenette` (`imagenette-orin` at the board's memory) | " | ~6 h on a 4060 Ti; a day on the Orin |
| imagenet | `lake run imagenet` | four cards, Python + tfds | weeks |

The side quests (A2/A1, MNv4-500, ConvNeXt-S/B, ViT-S/B) are imagenet-rung jobs that ride
`lake run imagenet` after their chapter's row; they are not a rung.

## 2. Where each platform stands (2026-10-26, the user's word; `support.tsv` is the record)

| platform | state | next |
|---|---|---|
| NVIDIA CUDA 12 (ares) and CUDA 13 (3060 box) | working; every rung on record between them | — |
| Intel Arc Pro B50 (minerva) | working; suite, mnist, cifar, imagenette on record | — |
| Jetson Orin Nano | running; mnist, cifar on record, imagenette (`imagenette-orin`) in progress | the tsv cell when it lands |
| AMD ROCm | was running (2026-08); fighting back right now | row stays with its last record and `historical/ROCM.md`; no promise in the book |
| Apple M1 / Metal | mnist run on the laptop (92.10 / 97.79 / 98.71, deploy/MAC.md), not in the tsv | pick the form it runs in (the suite + mnist, or mnist alone), record it in the tsv |
| AWS Trainium | waiting on AWS approval; the probe plan is planning/book_copy_audit_2026-10-21.md §5 | a `neuron` arm in `env.sh`, an expected file, `deploy/TRAINIUM.md`, a tsv row |
| TPU (TFRC) | the plan after Trainium: the primary sequence and the side quests, i.e. every imagenet-rung row, on TFRC time | a tsv row; the imagenet column is where it would show |

What the book says per platform is a projection of this table, so the book cannot say more than
the tsv does. "Running", "fighting", "waiting on approval" are this doc's words, never the book's:
a rung is on record or it is `-`.

## 3. The work, in order

1. **Reconcile `support.tsv` with what has run.** Add the Mac's mnist record (date, deploy/MAC.md
   § Measured); confirm the Orin cells; decide the ROCm row's wording (last record, no forecast);
   add a `TRAINIUM` and a `TPU` row only when each has a record. Add a `proofs` column? No: the
   proofs rung is platform-free (CI runs it); the tsv stays GPU rungs only, and the book's ladder
   table says so.
2. **`platforms_md.py --tex`.** Emit a LaTeX tabular from the same rows: platform, plugin (pinned),
   then one cell per rung with the date (`-` as a dash); write
   `blueprint/src/figures/platforms.tex`; `--check` covers both outputs. Appendix A's platforms
   float `\input`s it (book_floats already allows `\input{figures/...}` inside a float). The
   runbook column becomes a footnote line under the table (the book cannot link into the repo
   the way markdown does). Markdown-only cell text (links, `✅`) needs a plain rendering: date and
   a short tag, link dropped.
3. **Appendix A's prose.** Table A-1 becomes the ladder table of §1 (rung / command / needs /
   where in the book); "Track 1" section title → "Proofs only"; the thirteen "Track" mentions
   become rung names (Track 4 → "the imagenet tier", "Tracks 2 and 3" → "the mnist, cifar and
   imagenette tiers"); labels (`sec:getting_started_track3`, `_track4`) can stay. Appendix C's
   opening sentence and Table C-1 caption follow. The hand-typed platforms table is deleted once
   the generated one is checked against it, same commit.
4. **README tour table**: drop the 1–4 column for the command; its "Track 4" link text follows.
   Getting started's one-command-per-tier block is already in rung names and needs nothing.
5. **New platform flow** (the thing this buys): a row in `support.tsv` + an `env.sh` arm + an
   expected file + a `deploy/<X>.md` runbook; `check.sh` regenerates PLATFORMS.md and the book's
   table in the same run; the book's prose never names a platform count ("the six GPU platforms"
   at L421, L20033 become "the GPU platforms of Table A-2", so Trainium and TPU do not need a
   prose edit when they land).

Each step is its own commit; step 2 and the deletion in step 3 are the one that needs the
preview diffed before commit.

## 4. Open

* Does the book's table show the suite as its own column (one rung a new box clears in minutes
  without Lean) or fold it into the plugin column as "checked <date>"? Proposed: its own column.
* The two "Track-4" comments in content.tex's preamble (`\sqstatus`, never used) go in step 3.
* Whether the TPU row's imagenet cell, when it exists, is per (network, recipe) like the chapters'
  tables, or one date for the whole sequence. Proposed: one date + a pointer to the runs/ README
  that lists the rows, as the ares and 3060 cells already do.

## 5. IREE is a row, and the tier rule (2026-10-27, from the copy/positioning conversation)

IREE gets a platform row in Table A-2 instead of its own section. Its cells are honest in the
table's three columns: platform "IREE, over CUDA, ROCm or CPU; no PJRT plugin, it compiles every
kernel itself"; pinned "IREE 3.11 (the release Appendix B names), `iree-compile` on `$PATH`,
`ffi/libiree_ffi.so`"; where it stands "mnist, cifar, and five of the seven imagenette nets
(ResNet-50 and MobileNetV4 have no driver); no collectives, so no imagenet". That makes it
literally an mnist-to-imagenette platform, the same regime as the Orin and the Mac.

The row has to carry three things or a new reader trips:

* the CPU caveat: `llvm-cpu` miscompiles the pooled train steps, so the CPU gradient gate runs on
  vmvx / JAX (Appendix B already says this; the row says it in half a line, since CPU is the
  row's appeal for a reader without a card);
* the admission path: `check.sh` admits a box by "plugin loads + goldens match", and IREE has no
  plugin, so either the script grows an IREE mode or the row says it is admitted by the `-iree`
  twins (`lake run mnist-iree`, …);
* a `support.tsv` row (step 1), or the book and PLATFORMS.md disagree on the count. Step 5's
  "never name a platform count" already covers the prose.

**"The second lowerer" section (Appendix A) goes away.** Its mechanics are the row plus one
sentence under the table beside the device-variable sentence (`LEAN_MLIR_LOWERER=iree`, the
`-iree` twins). Its argument — two independent compilers with different codegen, runtime and
kernels are what make agreement evidence of anything; the 392× runtime-size gap is not a trust
gap; the small runtime relocates the trusted base rather than shrinking it, with the 2.3× / 4.9×
timings — moves to Appendix B, at the top of "Against JAX", directly before the `llvm-cpu`
miscompile. The bug is the evidence for the argument and today they sit 400 lines apart in
different appendices. "Why now"'s one line (IREE runs where no vendor library exists) stays: it
is a platform fact.

**The tier rule, stated once.** The ladder has two regimes with a boundary at cifar. Below it,
theorem families that need the whole training story: the SGD descent lemmas (Linear, MLP, CNN,
CIFAR) and the randomized-smoothing scorecards (MLP, CNN, CIFAR-CNN, 2026-07-12). Above it, the
VJP contracts, the step tie and the JAX cross-check only. IREE as a platform stops at the same
line (no collectives). Say the rule in Appendix B once; then descent, smoothing and the IREE row
all read as instances of it and none needs its own excuse.

Smoothing, under that rule: drop the `convnext-smooth` target, its two logs and
`runs/smoothing_frontier.tex` (20 images, n = 2016, 2 of 20 certified at r ≤ 0.5 after 2 h 10 m
on one card: a smoke test, and the lakefile comment calls it non-vacuous); keep the three
scorecards and their CI in `Certs` beside descent; the TRUST.md hypothesis row stays, beside the
descent row. Nothing enters the book beyond, at most, one paragraph in Appendix B framing the
scorecard as the one bridge from the Measured row to the Proven row (a decided binomial tail per
image), with the sentence that this is the other meaning of "verified". Not a section.

## 6. Landed (2026-10-27)

* Step 1, `support.tsv` reconciled: 4dd620ad. The Mac's mnist record (2026-10-14), the mars
  and minerva suite cells from their manifests (`auto:`), the Orin without queue prose, ROCm
  named by its last box, IREE as a row with the records it has (mnist 2026-04-21 traces, cifar
  2026-04 IREE era, imagenette 2026-05-27 ConvNeXt-T; the `-iree` twins have not been run as
  such since). The cell grammar `DATE · WHERE (TAG); more` is in the header.
* Step 2, `platforms_md.py --tex`: eeece9d6. `blueprint/src/figures/platforms.tex`, the suite as
  its own column (§4's first question, yes), `--check` on both outputs in the blueprint
  workflow.
* Steps 4 and 5 (README, home page, lakefile comments, deploy index): committed with the
  smoothing cut of §5 (`convnext-smooth`, its driver, three logs, the frontier figure).
* Step 3 and §5's book half are in the working tree, uncommitted, for the preview: Table A-1 is
  the ladder, the platforms float `\input`s the generated table, "The second lowerer" is the
  IREE row plus one paragraph under the table, its argument sits in Appendix B directly before
  the `llvm-cpu` miscompile, the tier rule and the scorecard paragraph follow the chapter
  tally, Appendix C and the preamble are in rung names, `\sqstatus` is gone.
* Open after this: the TPU row's imagenet cell (§4's third question) is still a question for
  when it exists; `repo_shape.txt` has no line for `runs/platform/*/manifest.tsv` (the three
  atlas neuron dirs of 3f9039d7 fail the gate today, before this work).
