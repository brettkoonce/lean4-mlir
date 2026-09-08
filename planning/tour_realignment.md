# Tour, not survey — realigning the front door around the canonical set

**Standing doc, opened 2026-09-08.** The repo presents a survey of everything that was built (the
three lowerer phases, every verification result, ninety-odd trainers) when what a new reader wants
is a tour: a course of models, met in order, each with one command and one number. The book
already IS that tour — its "How this book is organized" and "Getting started" chapters — and the
README is a stale second copy of the survey. This doc records the decision and the work to make
the repo say what the book says. Every item is a separate session and a separate commit; §1 and
§2 first (they fix what the others point at). Written at the end of the cleanup-backlog session
(`planning/cleanup_backlog.md`, §1–§11 all done); the clean session starts here.

**The decision (2026-09-08).** The canonical set is:

* the four `lake run` tiers — `mnist`, `cifar`, `imagenette`, `imagenet` — with each chapter's
  side-quest trainer riding in its tier (r50 and mnv4 in `imagenette`, the ConvNeXt/ViT S and B
  sizes in `imagenet`);
* the nine top-level demos in `demos/`, the ones the book carries.

New users are pointed at that and nothing else. Everything else is **the lab** — `apps/baselines/`,
the ablation exes, the MNIST/CIFAR robustness and low-precision exes, the `-iree` twins,
`demos/archive/`, `demos/probes/`, the tests, the Bestiary exes — and it stays, one level down,
labeled as the lab: it is the evidence behind the tour's numbers and the book's ablations cite it.
`apps/baselines/` keeps its trainers and may absorb some of the ablation exes.

Gates for every item: `lake exe docstring-checkrefs` when Lean prose moves (it resolves every
backticked identifier), `lake exe blueprint-checkdecls blueprint/lean_decls` when the book is
touched, `lake build` + `python3 scripts/check_audit_coverage.py` when `lakefile.lean` is touched
(the coverage script parses its root lists), a `grep` residue check after any path move (the
`sed` machinery of `cleanup_backlog.md` §1 and §10), `git diff verified_mlir/` empty. Land by
fast-forward, never a merge commit.

## 0. The canonical set, as it stands today

Numbers are the ones the book's chapters quote (CIFAR, Imagenette and ImageNet-1k) or
`RESULTS.md` / `demos/README.md` quote (the rest); the two cells no chapter quotes (R50 and MNv4 on
Imagenette) are read off their runs, and §1 says how. A cell marked ⚠ is one the sources disagree
on, and §7 reconciles them. Times are XLA wall-clocks on the box the source names.

| stop | command | runs | data | time | the number |
|---|---|---|---|---|---|
| Ch. 1–3 | `lake run mnist` | `mnist-linear-verified`, `mnist-mlp-verified`, `mnist-cnn-verified` | MNIST | ~1 min | CNN 99.50% (`RESULTS.md`) |
| Ch. 4 | `lake run cifar` | `cifar8w-ablation`, `cifar8w-bn-ablation` (SGD / momentum / AdamW × no-BN / BN, 40 epochs at a constant lr) | CIFAR-10 | ~19 min | no-BN 68.8 · 72.2 · 72.8, BN 74.5 · **76.3** · 74.3 (SGD · momentum · AdamW — Chapter 4's Lever-2 table, medians of five seeds from `runs/2026-09-01-cifar8w-6arm-constlr/`); ⚠ `RESULTS.md`'s 83.50% is a different net (§7 e) |
| Ch. 5–9 | `lake run imagenette` | `resnet34-`, `resnet50-`, `mobilenetv2-`, `mobilenetv4-`, `efficientnet-`, `convnext-`, `vit-verified-adam`, in book order | Imagenette | ~7 h for the five chapter nets, plus ~1 h 50 min for r50 (75 min) and mnv4 (35 min), one 4060 Ti per run | R34 89.50 · R50 89.71 · MNv2 89.25 · MNv4-Conv-M 86.24 · B0 89.96 · ConvNeXt-T 85.07 · ViT-Tiny 68.74 (the chapters' single runs; R50 and MNv4 are medians of five seeds from `runs/2026-08-31-imagenette-n3/`, §1); ⚠ R34 is also 89.71 elsewhere in the book (§7 d) |
| ImageNet-1k | `lake run imagenet` ⚠ does not exist yet (§2) | the book's Track 4 rows: `resnet34-`, `resnet50-` (two recipes), `mobilenetv2-`, `efficientnet-`, `convnext-`, `vit-imagenet-verified`; four more exes exist (`mobilenetv4-`, `convnext-s-`, `convnext-b-`, `vit-s-`, `vit-b-`) | ImageNet-1k | days, 4× 4060 Ti; per-job in `scripts/jobs/` and `runs/` | R34 74.16 · R50-A3 78.26 · MNv2 71.90 · MNv4-Conv-M 75.48 · B0 77.15 · ConvNeXt-T 81.53 · ViT-Tiny 72.31 |
| demo: segmentation | `lake exe unet-brats-r34` → `brats-predict` | BraTS, R34 encoder + UNet | ~hours, 2 GPUs | mIoU 0.742 (`demos/README.md`) |
| demo: detection | `lake exe yolov1-visdrone-fpn` | VisDrone, R34+FPN at 448 | ~2 h | mAP@0.5 0.2363 (`demos/README.md`); ⚠ `RESULTS.md`'s table stops at 0.1961 |
| demo: diffusion | `lake exe mnist-ddpm-train` → `mnist-ddpm-sample` | MNIST | 50 epochs | the sample grid (no scalar) |
| demo: language | `lake exe tinygpt-shakespeare`, `bigram-shakespeare`, `tinystories` | tinyshakespeare, TinyStories | ~11 min (TinyGPT) | 1.45 nats/char (TinyGPT); TinyStories in `RESULTS.md` |

The census the tour sits on: **233 `lean_exe`s**, of which the tour uses 12 (tiers) + 11
(ImageNet) + 9 (demos) + 2 (the gates). The rest by home: `tests/` 56, `Bestiary/` 41,
`apps/cifar/` 23, `apps/imagenette/` 22 (11 are the ImageNet runners), `apps/baselines/` 17,
`demos/archive/` 15, `apps/mnist/` 15, `tests/vjp_oracle/` 14, `demos/probes/` 12,
`apps/ablation/` 8, `apps/tools/` 1. ⚠ The `cifar` tier's two exes live in `apps/ablation/`, the
lab directory — the one place the canonical set and the lab share a home (§6 decides).

## 1. The canonical set is written down

**Done 2026-09-08.** This §0 is the record; the two cells that were empty are filled from the
runs, this file only.

* **R50 and MNv4 on Imagenette.** No chapter quotes them, so the cells are the final-epoch (80)
  top-1 of the tier's own exes, median of the five seeds in `runs/2026-08-31-imagenette-n3/`
  (`scripts/seed_sweep.sh`, `SUITE=imagenette`: 80 epochs, bs 32, AdamW with cosine + 3-epoch
  warmup, one 4060 Ti per run). R50: 89.58 / 89.68 / **89.71** / 89.73 / 89.96. MNv4-Conv-M:
  85.15 / 86.09 / **86.24** / 86.39 / 86.90. The `logs/ablation_*` files this section first pointed
  at are the 2026-06 narrow-net ablations and hold neither. The same sweep has the other five nets,
  and the chapters' single-run numbers sit inside their spreads — medians R34 90.14 · MNv2 89.07 ·
  B0 89.91 · ConvNeXt-T 85.40 · ViT-Tiny 68.87. Nothing in the book cites the sweep. Whether the
  README quotes a chapter's run or the median is a §3 call, made once for all seven (§7 a/d).
* **The CIFAR tier's six arms.** The tier's two exes have run at a constant lr since `1682bef5`
  (2026-09-01: warmup 0, decay 1.0), and Chapter 4's Lever-2 table is exactly their five-seed
  medians from `runs/2026-09-01-cifar8w-6arm-constlr/` — re-derived from the `bn_s*` / `nobn_s*`
  logs, every cell and every range reproduces. The earlier cosine-schedule pass
  `runs/2026-08-12-cifar8w-6arm-xla-cuda/` is a different experiment (BN + momentum 77.1 median;
  the no-BN momentum arm diverged to 10.00% in 3 of 5) and is no longer what the binaries do.

Filling the cells turned up three drifts and a stale docstring; they are §7's (d)–(g). (e)–(g)
were fixed the same day by decision; (d) is the user's note.

**Decision 2026-09-08, what a tour number is.** Every stop that has a seed sweep reports the
**mean and a 95% confidence interval over seeds** — not a single run, not the median. The CIFAR
and Imagenette cells above are placeholders until then (the book's medians and the chapters'
single runs, which is what the sources say today). ⚠ The sweeps need re-running before the
README quotes them — a note, not yet scheduled; which stops, how many seeds and what makes the
2026-08-31 / 2026-09-01 sweeps stale is §9's item 6.

## 2. `lake run imagenet`

The fourth tier does not exist as a command; the book's Track 4 lists its rows as separate `lake
exe` targets. Add `script imagenet` beside the other three in `lakefile.lean`, driving the seven
Track-4 rows through `runDemoGroup` in chapter order, so the README's tour is four symmetric
lines. ⚠ Unlike the other tiers this one is days of 4-GPU time and the box has crashed under it —
the script should print the plan and the per-job estimate first and require a confirmation flag
(the `benchmark` script already knows how to estimate), never start on a bare invocation. Decide
whether the S/B sizes and MNv4 ride in the tier (they are side quests in the book's sense) or
stay `lake exe` only. Gates: `lake build`, the coverage script, the book's Track 4 text (§8).

## 3. `README.md` becomes the front door

713 lines, 16 sections, and the survey: "Three phases", "Pipeline", "Cross-backend verification",
"VJP oracle", 250 lines of "What is and isn't verified", a "Results" table whose numbers come from
an older recipe on a 7900 XTX (R34 90.29, ViT-Tiny 71.70, an EfficientNetV2-S and a MobileNetV3
row) and disagree with every chapter of the book, a project-structure tree that describes a repo
with 103 proof files in six buckets and 18 demos, and a quick start that still carries the
2026-08-10 tier-rename warning. IREE is named 33 times, PJRT 13.

Rewrite it as ~120 lines: one paragraph on what this is; the tour — the §0 table in miniature,
four `lake run` lines and the demos, each with its number and a link to its chapter; how the
proofs are checked (`lake build ProofsMinimal` / `Certs`, the blueprint link, the comparator);
then pointers — the book, `demos/README.md`, `RESULTS.md`, the setup docs. What leaves:

* "Three phases", "Pipeline", "Cross-backend verification", "VJP oracle" → `historical/`
  (`historical/IREE.md` and `historical/Lean_MLIR.md` are already the story; these join them);
* "What is and isn't verified" and its four sub-sections → two paragraphs and a link to the book's
  "On Verification" chapter, which is where that argument lives now;
* "Results" → the tour table, with the book's numbers; the histories are `RESULTS.md`'s (§7);
* "Project structure" → a ten-line tree regenerated from the tree, or dropped in favour of
  `LeanMlir/Proofs/README.md`'s table and `demos/README.md`'s layout section;
* "Supported layers (phase 3 codegen)", "Lean specs" → the book's appendix, or `LeanMlir/README`.

`demos/README.md` is the model of a tour stop (the command, the number, the figure, one paragraph
of why) and the README's tour section should read like it in miniature. One commit; the docstring
gate does not read markdown, so the check is by hand: every path and command named must exist.

## 4. IREE to one line

PJRT/XLA is the training engine and every quoted number comes from it; IREE is the differential
compiler `tests/vjp_oracle/` runs and the `-iree` twins' lowerer. The README should say that in
one sentence and no more. `IREE_BUILD.md` stays (the oracle needs the shim), `upstream-issues/`
stays (they are IREE reproducers), the `-iree` scripts stay and are documented where the book
already documents them ("The second lowerer", the end of Getting started). The one-line rule
applies to `demos/README.md` and `RESULTS.md` too ("These runs: … via CUDA / IREE").

## 5. Root hygiene

* **57 `.log` files at the root.** Untracked (`.gitignore` ignores `*.log` except under `logs/`
  and `runs/`), but `run.sh` tees every trainer's output to `<trainer>.log` in the repo root, so
  they come back after every run. Change `run.sh` to tee into `runs/<date>-<trainer>/` (or `logs/`,
  the curated dir) and delete the local ones.
* **Phase-1 and phase-2 remnants beside the live code.** `mnist-lean4/` (pure Lean + C BLAS, 11
  files) and `mlir_poc/` (the Python exporters, 15 tracked) → `historical/`. The README's "Three
  phases" is their only front-door citation; the book's introduction names them as history, which
  is where they will then be.
* **Six reference docs at the root.** `RESULTS.md`, `BENCHMARK.md`, `CUDA.md`, `ROCM.md`,
  `IREE_BUILD.md` → `docs/`; `CHANGELOG.md` stays. Cited from the README, the book (`\texttt{}`
  paths), `demos/README.md`, `CHANGELOG.md`, `planning/archive/`, the workflows and `deploy/` —
  a `sed` sweep like §1 of the cleanup backlog, residue-checked. ⚠ Decide `docs/` vs leaving them
  where they are; the README pointing at them is what matters, the directory is taste.
* Leave: `Bestiary/` (Part 2 of the book), `traces/`, `deploy/` (the Orin path), `home_page/`
  (GitHub Pages), `upstream-issues/`, `jax/` (the reference implementation the ImageNet path
  ports), `data/`.

## 6. `lakefile.lean` says which is which

233 exes in one flat run of `lean_exe`s. Regroup under section banners — the tour first (the
three tier groups, the ImageNet runners, the demos, the two gates), then the lab by home
(`apps/baselines/`, the ablations, the MNIST/CIFAR robustness and low-precision exes, the demo
archive and probes, the tests, the oracle, the Bestiary) — with one comment line per group saying
what it is. Nothing is deleted or renamed; `RESULTS.md` and the book's ablations cite these
names. Decide the `apps/ablation/` question here: the `cifar` tier's two exes either move to
`apps/cifar/` (the tier's home) or `apps/ablation/` is declared part of the tour; and which of the
remaining ablation exes `apps/baselines/` absorbs (decision 2026-09-08: it may keep some).
Cosmetic, but the file every session touches; `check_audit_coverage.py` parses its root lists, so
run it after.

## 7. `RESULTS.md`, and the numbers that disagree

629 lines of per-epoch histories, ablations and scorecards. The tour wants one number per stop;
the histories belong in the book's appendix or beside their runs (`runs/<run>/README.md`). Before
anything moves, reconcile the drifts §0 and §1 found — one number, one source, cited everywhere
else by pointer:

* (a) the README's Imagenette table against the book's chapter numbers (different recipe,
  different card — say which is canonical, the book's);
* (b) VisDrone at 0.1961 in `RESULTS.md` against 0.2363 in `demos/README.md` (the demos README has
  the later run);
* (c) the `certs.yml` step-summary labels that still say 146 / 210 params for r34 / mnv2 (the
  census is 110 / 158, `cleanup_backlog.md` §10);
* (d) (the user's note, 2026-09-08) **ResNet-34 on Imagenette is two numbers inside the book.** Chapter 5's listing ends at
  89.50% and cites `runs/2026-09-01-r34-imagenette-rerun.log`, which is not in the repo (no
  tracked log has that epoch-80 line); the MobileNetV2 and EfficientNet chapters and both
  comparison tables say 89.71%, which is epoch 80 of `runs/2026-08-12-r34-imagenette-xla-cuda/`.
  The five-seed median is 90.14 (§1). Pick one, and cite a log that exists;
* (e) **Done 2026-09-08.** `RESULTS.md`'s CIFAR row was not the tier — its 83.50% was a 4-conv,
  30-epoch net. It now carries Chapter 4's constant-lr six-arm table (the latest, medians of five
  seeds); the old row is in `git log`;
* (f) **Done 2026-09-08.** Chapter 4's §4.1 listing cited
  `runs/2026-08-12-cifar8w-6arm-xla-cuda/cifar8w-bn-p1.log` (the cosine pass, momentum arm 77.48%)
  but printed the constant-lr momentum arm (75.94%) from
  `runs/2026-09-01-cifar8w-bn-xla-cuda/cifar8w-bn.log`. Decision: the constant-lr version is the
  one Chapter 4 wants (cosine enters with ResNet in Chapter 5); the cite, the PJRT version, one
  compile time and the elision range now match that log, every epoch line already did.
  `scripts/seed_sweep.sh`'s header said the same stale things and was fixed with it;
* (g) **Done 2026-09-08.** `apps/ablation/MainCifar8WideAblation.lean`'s docstring said
  "cosine-warmup" for code that has run a constant lr since `1682bef5`; it now says so, like its BN
  twin. ⚠ One residue is left for §6, where the lakefile gate runs anyway: `script cifar-iree`'s
  docstring still says the 2026-08-12 cosine run is "behind §4.1's listing".

## 8. The book is the onboarding text, once

"Getting started" (Tracks 1–4, troubleshooting, the second lowerer) is already what the README's
"Native setup" and "Quick start" duplicate at lab granularity. After §3 the README points there
and stops; the only book edits are Track 4 learning the `lake run imagenet` line (§2) and
"How this book is organized" gaining a sentence that names the tiers as the book's runnable spine.
Gate: `blueprint-checkdecls`.

## 9. Open decisions, to settle at the start of the clean session

1. `docs/` for the reference markdown, or leave them at the root and only fix the README's pointers.
2. `mnist-lean4/` and `mlir_poc/`: move to `historical/`, or delete (the book's introduction and
   `historical/*.md` already tell the story; `git log` keeps the code).
3. Which ablation exes `apps/baselines/` absorbs, and where the `cifar` tier's two exes live.
4. Whether the `imagenet` tier is the book's seven Track-4 rows or all eleven ImageNet exes.
5. The two empty cells in §0 — settled by §1 (2026-09-08), and the form of the number is settled
   too: mean ± 95% CI over seeds (§1's decision).
6. Re-run the seed sweeps before the README quotes mean ± CI: which stops (Imagenette's seven,
   the CIFAR six arms; MNIST?), how many seeds, and what makes the 2026-08-31 / 2026-09-01
   sweeps stale. Added 2026-09-08: **the ResNet chapter's §5.6 recipe ablation, in both
   precisions.** Eight arms (`resnet34-ablation data <arm> [bf16]`: full, nowd, nowarm, nols,
   noadam, nocos, noaug, bare), each ONE 80-epoch run today (`runs/2026-09-01-r34-ablation/`;
   the figure's band is the Wilson half-width at n = 3,925, not a spread over seeds), so the
   fp32 and bf16 panels both go to n seeds — ~80 min an arm on one 4060 Ti, ~21 GPU-hours per
   seed for the pair. ⚠ Ask before launching — the Imagenette sweep alone is ~20 GPU-hours.
