# 2026-10-27 — the TinyGPT ladder on XLA: rows T1–T5 of planning/part2_ship_audit.md §2.1, three seeds each

The question `planning/tinygpt_ngram_ladder.md` left open on 10-21 (its Phase 1, three `nano`
seeds at 10k steps on the CPU box: the net sits 0.04–0.06 bits/char *behind* the Kneser-Ney
7-gram) was whether the transformer crosses the counts with more steps, a larger batch, the
larger `tiny` config, or a modern position encoding. Every row of that follow-up list ran here on
one 4060 Ti under XLA, three seeds each (`LEAN_MLIR_SEED` 0 = the recipe's own init, 1, 2), and
was scored against the same ladder under both of the plan's protocols. The answer is no: **no row
crosses the 7-gram, and the one lever the ladder has not measured is dropout** — row T6, which
needed emitter code and is `runs/2026-10-27-tinygpt-dropout-xla/`.

```
LEAN_MLIR_SEED=$s LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=0 \
  .lake/build/bin/tinygpt-shakespeare train <key> <steps> <batch> 30        # run.sh's `job`
  .lake/build/bin/tinygpt-shakespeare score <key> <job>/<key>_params.bin <job>
python3 scripts/demos/ngram_ladder.py fit --orders 10 --out ngram_kn.npz    # once, 5 s
python3 scripts/demos/ngram_ladder.py score --model ngram_kn.npz --protocol chunked:64  --out .
python3 scripts/demos/ngram_ladder.py score --model ngram_kn.npz --protocol chunked:128 --out .
python3 scripts/demos/ngram_ladder.py score --model ngram_kn.npz --protocol streaming --T 64  --out kn_T64
python3 scripts/demos/ngram_ladder.py score --model ngram_kn.npz --protocol streaming --T 128 --out kn_T128
./kn_table.sh      # symlinks each job to the KN arrays at its window (tiny 128, nano 64), runs `table`
./summary.sh       # the table below
```

The jobs ran back to back on card 0 (`queue.log`): `tiny` 10k is 6.5 min, `tiny` 5k 3.7, `nano`
10k 3.5, 20k 6.3, 40k 11.5. The trainer saves to a fixed path per key, so each job copies its
checkpoint out before the next starts (`run.sh`). The checkpoints (18, 40 MB) are on ares and
ignored here; `train.log` (loss every 100 steps, val every 500), `score.log`, `table.{md,csv}`
and `context_curve.csv` are committed per job.

## The table (val, bits/char; `summary.sh`)

`train_b` is the final training loss in bits; `logged` the trainer's own 12-batch val; P1 the
`score` mode's chunked pass at the net's window (T = 128 for `tiny`, 64 for `nano`); P2 the
streaming pass; the gap is P2 against the ladder's best order (the 7-gram, 2.1883 at T = 64 /
2.1887 at T = 128), with a 95% block-bootstrap CI over the ~111k val positions.

```
job                 train_b   logged       P1       P2  gap to KN (P2), 95% CI
nano20k_s0            1.839 2.264989   2.2800   2.2379  +0.0496 [+0.0369, +0.0635]
nano20k_s1            1.831 2.255386   2.2820   2.2387  +0.0504 [+0.0371, +0.0634]
nano20k_s2            1.897 2.243597   2.2691   2.2296  +0.0414 [+0.0272, +0.0554]
nano40k_s0            1.683 2.307001   2.3412   2.3017  +0.1134 [+0.0955, +0.1306]
nano40k_s1            1.703 2.296255   2.3322   2.2900  +0.1017 [+0.0853, +0.1188]
nano40k_s2            1.784 2.287166   2.3095   2.2744  +0.0861 [+0.0675, +0.1046]
nanob64_s0            1.956 2.278277   2.2992   2.2561  +0.0679 [+0.0533, +0.0813]
nanob64_s1            1.822 2.279889   2.2937   2.2549  +0.0666 [+0.0535, +0.0799]
nanob64_s2            1.826 2.254351   2.2764   2.2277  +0.0394 [+0.0266, +0.0531]
rope_s0               1.978 2.270399   2.2789   2.2213  +0.0331 [+0.0233, +0.0428]
rope_s1               1.874 2.273542   2.2800   2.2245  +0.0362 [+0.0268, +0.0457]
rope_s2               1.838 2.273250   2.2745   2.2148  +0.0266 [+0.0156, +0.0368]
tiny5k_s0             1.531 2.292085   2.2702   2.2611  +0.0724 [+0.0600, +0.0853]
tiny5k_s1             1.623 2.310900   2.2959   2.2939  +0.1052 [+0.0905, +0.1232]
tiny5k_s2             1.650 2.301309   2.2904   2.2735  +0.0848 [+0.0700, +0.0997]
tiny_s0               1.290 2.756311   2.7461   2.8092  +0.6206 [+0.5940, +0.6484]
tiny_s1               1.259 2.778542   2.7458   2.8071  +0.6185 [+0.5920, +0.6481]
tiny_s2               1.264 2.781408   2.7524   2.7990  +0.6103 [+0.5810, +0.6408]
```

The ladder itself (val, P2 streaming; P1 at T = 64 / 128 in each job's `table.md`): uniform
6.022, KN 1 4.829, 2 3.583, 3 2.940, 4 2.480, 5 2.253, 6 2.197, **7 2.188**, 8 2.192, 9 2.196,
10 2.197 — the plateau at order 6 by the plan's 0.01 rule, the best at 7, identical to 10-21.
Gate 1 holds on every row: P1 − logged is within 0.035 of zero (the 12-batch sampling sd is 0.026).

| row (audit §2.1) | key, steps, batch | gap to KN 7 (P2), by seed | mean | min of the val curve (step) |
|---|---|---|---|---|
| 10-21 reference, CPU box | `nano` 10k × 32 | +0.056 / +0.039 / +0.038 | +0.044 | — |
| T2 | `nano` 20k × 32 | +0.050 / +0.050 / +0.041 | +0.047 | 2.257 (14.5k) / 2.247 (17.5k) / 2.235 (17k) |
| T2 | `nano` 40k × 32 | +0.113 / +0.102 / +0.086 | **+0.100** | 2.272 (24k) / 2.270 (20.5k) / 2.252 (19k) |
| T3 | `nano` 10k × 64 | +0.068 / +0.067 / +0.039 | +0.058 | 2.270 (10k) / 2.270 (8.5k) / 2.245 (10k) |
| T4 | `nano-rope-nopos` 10k × 32 | +0.033 / +0.036 / +0.027 | **+0.032** | 2.269 (9k) / 2.271 (9.5k) / 2.272 (9.5k) |
| T5 | `tiny` 5k × 32 | +0.072 / +0.105 / +0.085 | +0.087 | 2.244 (3.5k) / 2.251 (3.5k) / 2.253 (3k) |
| T1 | `tiny` 10k × 32 | +0.621 / +0.619 / +0.610 | **+0.616** | 2.264 (4k) / 2.263 (4k) / 2.263 (3k) |

## What each row says

- **T1, `tiny` at 10k, is the "val turns up" exhibit, three times over.** The val curve
  bottoms at 2.263–2.264 bits at step 3000–4000 on every seed (the seeds agree to 0.001) and
  climbs for the remaining 6,000 steps to 2.76–2.78 while the train loss falls to 1.26–1.29 bits.
  Scored on the stream it ends 0.62 behind the 7-gram. `historical/RESULTS.md`'s 2.78 and "val
  bottomed ≈2.27 @ step ~4500" are reproduced on XLA. The curve for the book
  (`lm_demos_modernization.md` §3) is in each `tiny_s*/train.log` (`── val @ step N` every 500).
- **T5, `tiny` at 5k, is the better `tiny` row and still behind.** The 5k cosine schedule puts the
  minimum at 2.244–2.253 (step 3000–3500) and the end at 2.29–2.31, +0.07 to +0.11 against the
  counts: 1.2M parameters buy nothing over `nano`'s 212K on 1 MB of text without regularisation.
- **T2: more steps do not move `nano` toward the counts.** 20k lands where 10k did (+0.047
  against the 10-21 run's +0.044; the minima at 14–17k are 2.235–2.257, the same 2.25 as 10k's
  end). 40k is worse: the minimum comes at 19–24k (2.25–2.27) and the val rises 0.03–0.05 from
  there to the end — the net has what it can take from the corpus by 10k, and the cosine tail
  of a longer schedule spends the rest overfitting. The plan's "undertrained" case
  (`tinygpt_ngram_ladder.md` §1, third case) is the wrong name for this gap.
- **T3: batch 64 is batch 32.** Twice the tokens per step at the same 10k steps gives +0.058,
  inside the batch-32 rows' spread; the 2× tokens are spent on the same overfit.
- **T4: RoPE without a position table is the best net row, and it does not extrapolate at this
  scale.** +0.027 to +0.036, 0.015 ahead of absolute positions under P2 and level under P1
  (2.2745–2.2800 against 2.269–2.282): the gain is all in the unbounded-history pass (P1 − P2 is
  +0.054 to +0.058 for RoPE against +0.042 for the position table), i.e. relative positions make
  the net better at using a long stream, not better within a window. The gate's two halves hold
  as `lm_demos_modernization.md` §5 wrote them — `xeval nano 128` on a copied `nano` checkpoint
  refuses to load ("checkpoint 850180B ≠ expected 866564B … posEmb-off is required to
  extrapolate", the contrast) and the RoPE checkpoint runs at T = 64 / 128 / 256 — but what it
  produces is not coherent: T = 64 2.275–2.282, T = 128 2.85–3.36, T = 256 4.38–5.41 bits/char,
  i.e. above the KN bigram (3.58) at 4× the window on every seed. The weights are
  length-independent; the function learned on 64-character windows is not. "Stays coherent" in
  §5 is wrong as a claim about this net and should be stated as the load/refuse gate only.

## What this says for the book

Against the right bracket, every transformer row in §10.3.1's ladder is behind a 6-character
count on its own corpus: by +0.03 (RoPE) to +0.10 (longer schedules), and `tiny` is behind by
+0.6 at the configuration the catalogue prints. The seed spread is a quarter of the smallest gap,
so the ordering is settled. The gap is not steps, batch, width or positions; the train/val
separation on every row (train 1.3–2.0 bits against val 2.2–2.8) says regularisation — the
dropout arm (T6, `runs/2026-10-27-tinygpt-dropout-xla/`) is the ladder's next row and the first
one that can change the sign — and it does, the same evening: `tiny` with residual dropout 0.2 at
10k steps is 0.06 bits/char *ahead* of the 7-gram on every seed. Figures here: `ngram_ladder_tiny5k_s0.png` (T = 128) and
`ngram_ladder_rope_s0.png` (T = 64), the plan's three panels for one seed of each window
(`scripts/demos/ngram_ladder_figure.py <job> <png> --T <T> --nets <key>`).
