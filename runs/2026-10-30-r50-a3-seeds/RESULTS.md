# ResNet-50 RSB-A3, three seeds per arm — mars (4× RTX 3060), 2026-10-30 → 11-02

Tier 2 of `planning/runs_queue_2026-10-29.md`: does the Ch 5 A3 pair's verified lead (78.08 vs 77.60,
paired p = 0.0002 at one seed) survive seeds? `queue.sh` ran seeds 2 and 3 of each arm back to back
(`QUEUE_EVENT`), each one attempt, no resumes; `pair.sh` scored the pairs (`pairs/`, `pairs.log`).

| arm | conf | seeds 2, 3 |
|---|---|---|
| verified | `scripts/jobs/r50-a3-wxclip4x128-bf16-4gpu.conf`, `RUN_SEED=<n>` (ckpt tag `2611s<n>`, shim seeds 10<n>) | 22.2 h, 21.0 h |
| JAX | `scripts/jobs/r50-a3-jax-4gpu.conf`, `RUN_SEED=<n>` (`LEAN_MLIR_SEED_OFFSET`) | 14.6 h, 14.5 h |

Run directories: `../2026-10-30-r50-a3-{jax,ver}-s{2,3}/`. Seed 1 is the 10-12 pair
(`../2026-10-12-r50-a3-jax/`, `../2026-10-12-r50-a3-rsb-bf16/`).

## Top-1 at epoch 100, all 50,000 val images

| seed | JAX | verified | gap | discordant | net flips | McNemar exact p |
|---|---|---|---|---|---|---|
| 1 (10-12, ares) | 77.60 (38,801) | 78.08 (39,039) | +0.48 | 4,038 | +236 | **0.0002** |
| 2 | 77.66 (38,829) | 77.82 (38,911) | +0.16 | 4,008 | +78 | 0.22 |
| 3 | 77.64 (38,821) | 77.77 (38,886) | +0.13 | 4,039 | +65 | 0.31 |
| **mean (range)** | **77.63** (77.60–77.66) | **77.89** (77.77–78.08) | +0.26 | | | |

Top-5, seeds 2 / 3: JAX 93.65 / 93.53, verified 93.81 / 93.70. Seed 1's pair row is the 10-12 RESULTS.md's
(JAX side its 38,803 re-score).

## Reading

* The verified arm leads on all three seeds; only seed 1's pair separates. Seeds 2 and 3 each sit at
  p > 0.2, with the same ~8 % discordant mass as seed 1 and a third of its net flips.
* The JAX arm is tight (0.06 range); the verified arm spreads 0.31, and seed 1 is its top. Seed 1 is
  also the confounded pair: it ran on ares (4× 4060 Ti), and its verified half predates the 10-28 stem
  pool backward re-emit (ea934bc6). Seeds 2 and 3 are one box, one artifact.
* So the 0.47 is not a path difference that survives seeds. What is left is a lead of about 0.15 on the
  matched seeds that no single pair resolves. The Mixup/CutMix placement question the chapter leaves open
  does not need a test on this evidence.

## Pairing guards (`pair.sh`)

Each bitmap has to reproduce its own run's count before it is paired. The verified e100 bitmaps match their
logs exactly (38,911, 38,886). The JAX checkpoint, re-scored on one card at batch 200 in f32
(`jax/scripts/eval_full50k.py`, `PROTOCOL=train`), reads 38,833 against the trainer's 38,829 for seed 2 and
38,821 = 38,821 for seed 3; seed 1 was 38,803 against 38,801. The first `pair.sh` demanded equality and
refused seed 2 (`pairs.log`); the guard is now ±10 (a wrong checkpoint or protocol is off by hundreds), and
`pairs/mcnemar_s2.txt` was produced by the same `scripts/demos/mcnemar.py` call by hand. Bitmaps are
gitignored (`runs/**/*.bin`).
