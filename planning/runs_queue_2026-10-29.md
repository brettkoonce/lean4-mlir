# Runs queue — 2026-10-29

What the book needs run, in the order to run it, for a free 4-card box. Written after the
dev-history audit of 2026-10-29 (commits d6341742, 7d54f110), which checked every run directory
the book cites against the last change to the artifact it compiled and the driver it ran under.
The ImageNet-scale runs stay where they are, in `planning/imagenet_finish_line.md` "Runs owed";
this file is the short and medium work that can land while those queue.

Every launch below goes through `scripts/sweeps/seed_sweep.sh` or a `scripts/jobs/*.conf`, never a
binary by hand: the sweep driver owns residency, per-seed checkpoint tags, the `-adam` binaries
and the AER watchdog, and its header lists the traps.

## Why these and not others

Three facts drive the list.

1. **EfficientNet-B0's artifact changed semantically after its Imagenette run.** The row in
   `tab:imagenette_five` is `runs/2026-08-12-enet-imagenette-xla-cuda/`; on 2026-10-12 (a89fa9af)
   TF `SAME` padding went onto B0's four strided depthwise convolutions. The shipped 90.04 % was
   trained by a different network.
2. **Every ResNet artifact was re-emitted on 2026-10-28** (ea934bc6, the stem pool's backward as
   nine stride-2 slices instead of `select_and_scatter`). Same math, new lowering. Every ResNet
   verified number in the book predates it: the R34 Imagenette ablation seeds (09-11, 09-12), the
   R34 2018 verified half (10-20), the R50 2018 and A3 verified halves (10-12, 10-24), R50
   Imagenette. Nothing is known to be wrong; nothing has been re-measured either.
3. **Chapter 4's board spans two dates.** The "no BN" and "BN, per image" rows of
   `tab:cifar_optimizers`, and the un-normalized fp32-vs-bf16 comparison, are
   `runs/2026-09-01-cifar8w-6arm-constlr/`; the batch-statistics rows are
   `runs/2026-10-24-bbn-seeds/`. The artifacts changed only in banners between the two
   (83e1085d), so the numbers should hold, but one date for the whole board is the clean story,
   and the September NaN symptom on the no-BN AdamW arms gets a fresh look.

Everything else the book cites either post-dates its artifact's last change or changed only in
comments (the 09-27 and 09-30 banner passes). The driver fix of 2026-10-28 (728bb7b5) touched
logging and gating only.

## Tier 1 — today, book consistency

### 1a. Imagenette, five seeds, all seven nets

```
SUITE=imagenette scripts/sweeps/seed_sweep.sh 0,1,2,3 "1 2 3 4 5"
```

One sweep settles three things at once: the B0 re-validation (fact 1), the ResNet check on the new
pool backward (fact 2, at Imagenette cost), and the five-seed column already queued for
`tab:imagenette_five` (`planning/imagenet_recipe_sections.md` §8). Seven nets × 5 seeds, LPT-packed
one per card; the driver measured ~19.4 GPU-h for three seeds, so about 32 GPU-h, roughly 8 h on
four cards. Resumable; a crashed box restarts where it was.

| net | shipped (one run, or R34's five-seed mean) | what to look at |
|---|---|---|
| EfficientNet-B0 | 90.04 | the median after `SAME` padding; this row is the one that may move |
| ResNet-34 | 89.99 | the five-seed median against the 09-12 five; same band = pool backward is a no-op numerically |
| ResNet-50 | 89.71 | same test as R34 |
| MobileNetV2, ConvNeXt-T, ViT-Tiny | 89.25, 82.27, 68.74 | seed spread for the column; no artifact change to test |
| MobileNetV4 | side quest, no table row | seed spread for its paragraph |

Lands as: `tab:imagenette_five` becomes median ± range over five seeds for every row, with one run
directory in the caption; the Ch 5 and Ch 7 Run-it-first transcripts point at a seed from this sweep.
If R34 or R50 moves outside its Sept band, the ImageNet ResNet verified halves (Tier 3 note) become
re-runs; if not, they stand.

### 1b. CIFAR, the whole board on one date

```
SUITE=small scripts/sweeps/seed_sweep.sh 0,1,2,3 "1 2 3 4 5"      # cifar8w, cifar8w-bn, the MNIST three
```

`small` covers the two per-image binaries (`cifar8w-ablation`, `cifar8w-bn-ablation`, three
optimizer arms each, ~6 and ~13 min a seed) and the MNIST ladder at 12 epochs (a minute each).
The un-normalized precision pair, `cifar8wb-ablation` and `cifar8wb-bf16-ablation` (three arms
each, ~6 min a seed), is in no suite yet: add a `SUITE=nobnprec` case to `seed_sweep.sh` on the
`bnprec` pattern before launch rather than running the binaries by hand. About 2.5 GPU-h in all,
under an hour on four cards.

Lands as: `tab:cifar_optimizers` "no BN" and "BN, per image" rows and the un-normalized bf16
paragraph re-read from the new directory; the two captions name one September directory fewer.
Check the no-BN AdamW arms for `loss=NaN`: the Oct-24 batch-BN run had none in thirty arm-runs, and
the paragraph that explained the September symptom is already out of the book (d6341742).

Not included: `runs/2026-09-01-mnist-n30/` (the thirty-run MNIST dots). Banner-only artifact change;
rerun only if the box is otherwise idle (~30 min).

## Tier 2 — the open question: ResNet-50 A3, seeds per arm

The Ch 5 A3 pair parts by 0.47 top-1 (verified 78.05, reference 77.58), paired p = 0.0002, one seed
per arm. The chapter says a paired lead at one seed is a statement about two trained models, and
"whether A3's lead is the path or the seed needs seeds per arm". The one mechanism it names and
leaves untested is where the Mixup/CutMix draw happens (per 512-image micro-batch on the verified
path, per accumulated step on the reference).

| half | conf | measured | per extra seed |
|---|---|---|---|
| verified | `scripts/jobs/r50-a3-wxclip4x128-bf16-4gpu.conf` | 21.9 h (`runs/2026-10-12-r50-a3-rsb-bf16/`) | ~22 h |
| reference | `scripts/jobs/r50-a3-jax-4gpu.conf` | ~13.5 h (`runs/2026-10-12-r50-a3-jax/`) | ~13.5 h |

Two more seeds per arm is ~71 box-hours, three days on one four-card box. Before launch:

- the verified conf tags its checkpoint (`CKPT_TAG`, the `2610` lineage); each seed needs its own
  tag and `LEAN_MLIR_SEED=<n>` in the environment, the way the sweep driver does it. The confs
  carry neither today; add both as parameters first.
- the JAX half has no seed flag: the emitted trainer carries its seeds as constants (the config's
  `seed` 314159, the shuffle's 42, the augmentation seed). A second reference seed means changing
  `seed` in the Lean config and re-emitting (`--emit`, `--shim`, then `scripts/regen_jax_generated.sh
  sync`), or adding an env override to the emitter first.
- launch with `lake run <conf> plan` first, then `once`.

Lands as: three seeds per arm turn the Ch 5 "two trained models at one seed" paragraph and the
Ch 5 recap's "a difference … that the chapter leaves unexplained" into a measurement either way:
a lead that survives seeds is a path difference and the Mixup/CutMix placement becomes the thing to
test; one inside the seed spread retires the question.

## Tier 3 — ImageNet scale

The owed pairs and their box-hours are the table in `planning/imagenet_finish_line.md` "Runs
owed". Since it was written: MNv2 420 landed (10-20), B0 350 landed (10-21) and is now the
reference config's default, the official ConvNeXt-T pair landed (10-29, 746e7a9d), MNv4's 500-epoch
tier is still the one TODO block in Chapter 6. Open there: ConvNeXt-S/B, ViT-S/B, MNv4 500, RSB-A2,
RSB-A1.

One conditional addition, from fact 2: if Tier 1a moves a ResNet, the R34 2018 verified half
(`r34-default-bf16-4gpu`, ~21 h) and the R50 2018 and A3 verified halves were trained on the old
pool backward and join this tier. If Tier 1a holds, they do not.

## Not on the list, and why

- **NEU-DET single-grid head, one run.** The caption says one run; the arm is a contrast, not a
  headline. Seeds would be nice, not needed.
- **ConvNeXt-T and ViT-Tiny Imagenette on exact-GELU twins** (`planning/book_copy_audit_2026-10-21.md`
  item 3). Both Imagenette artifacts still carry tanh GELU (no `erf` in either). That is a
  decision about how the book tells the GELU story, not a consistency re-run; decide it before
  spending the card time.
- **The Chapter 5 R34 five-seed ablation** (`runs/2026-09-1[12]-r34-ablation-*-seeds/`, two
  precisions × arms × five seeds). Covered by Tier 1a's R34 check; re-run the ablation only if
  that check fails.

## Landed

- **2026-10-30, Tier 1 (mars, 4× 3060).** 1b: `runs/2026-10-29-small-seeds/`,
  `runs/2026-10-29-nobnprec-seeds/`; every arm's median inside its 09-01 range, so the two 09-01
  rows of `tab:cifar_optimizers` hold. One no-BN Nesterov seed never trained (10.00, range cell
  would read 10.00–74.00); no-BN AdamW `loss=NaN` in 4/5, 4/5, 3/5 with test accuracy intact.
  1a: `runs/2026-10-29-imagenette-seeds/`; B0 and both ResNets hold (the Tier 3 conditional does not
  fire), ConvNeXt's median is 83.75 against the shipped 82.27 (a low draw on a two-point seed spread),
  MNv2 88.87 against 89.25. Book not edited; the ConvNeXt row and its ch 8 transcript need a decision.
- **2026-10-30, Tier 1 in the book** (`blueprint/src/content.tex`). `tab:imagenette_five` is median
  (range) over the five seeds for every row, timing from the median seed on the 3060, and the Ch 5–9
  Run-it-first transcripts are the median seeds (R34 s4, MNv2 s5, B0 s4, CNX-T s2, ViT s1); the
  comparison paragraphs read medians against medians (MNv2 0.9 behind R34 with the ranges clear, B0
  level with R34, ConvNeXt 83.75 and the widest row). The ViT "reproduces exactly" paragraph is out:
  seed 1 on the 3060 stack lands 68.61 against 68.74 on the 4060 Ti stack, so bit-identity across
  cards is not a claim the current evidence supports. The R34 ablation keeps its own 09-12 five seeds
  as the paired baseline. Ch 4: the two per-image rows, the SGD curves and the un-normalized
  precision pair are the 10-29 directories; the claims that bf16 costs the bare net 1.4 points and
  that the op family moves it by 2 did not replicate (medians move under a point, inside the seed
  ranges) and are rewritten as such; the dead no-BN momentum seed is in the table with a dagger and
  carries the "verified gradient is not trainability" point.
- **Tier 2 queued** by `runs/2026-10-30-r50-a3-seeds/queue.sh`: seeds 2 and 3 per arm, JAX s2
  launched 2026-10-30 08:54 UTC. Both confs take `RUN_SEED`; the JAX emitter reads
  `LEAN_MLIR_SEED_OFFSET`.

## After each run

RESULTS.md in the run directory (the `bbn-seeds` one is the shape); the caption's run directory
updated in `blueprint/src/content.tex`; the gates (`book_listings.py --check`, `book_floats.py
--strict`, `book_prose_idents.py`); a line here saying what landed and what it changed. The book
cites the run that produced the shipped number and no other (`book-prose-no-prior-runs`).
