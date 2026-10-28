# 2026-10-27 — TinyGPT `tiny` with residual dropout: row T6 of planning/part2_ship_audit.md §2.1

The ladder run the same afternoon (`runs/2026-10-27-tinygpt-ladder-xla/README.md`) ends with every
transformer row behind the Kneser-Ney 7-gram on its own corpus — `tiny` by 0.62 bits/char at the
catalogue's 10k steps, where its val curve bottoms at step 3–4k and climbs for the rest of the
schedule — and names dropout as the one lever left (`lm_demos_modernization.md` §1.4: "the
indicated fix, not more steps"). This is that row. **With GPT-2's residual dropout at 0.2 the
same net at the same 10k steps is 0.06 bits/char ahead of the 7-gram on all three seeds, with
every CI clear of zero, and its val curve is still falling at the end of the schedule.** It is the
first transformer row in the ladder to cross the counts.

## The code (uncommitted at launch, 10-27 afternoon)

`transformerEncoder` gained `(dropout : Float := 0.0)` (`LeanMlir/Types.lean`). In the generic
emitter's train step (`MlirCodegen.lean`, `emitDropoutMask` + the `transformerEncoder` case of
`emitTrainForward`) each block's two residual-branch outputs — the attention projection and the
MLP's second linear, GPT-2's placement — are multiplied by an in-graph inverted-dropout mask: a
`stablehlo.rng_bit_generator` (Philox) seeded from the step scalar `%t` and a per-site constant,
top 24 bits to a uniform, keep where it is below `1 − p`, scaled by `1/(1 − p)`. The VJP reuses
the forward's mask SSA. The eval forward has no dropout, and at 0 the train-step graph is
byte-identical to the previous emitter's (XLA reported `tiny`'s step cached). The TinyGPT exe
takes `<base>-do<pct>` keys (`demos/MainTinyGptShakespeare.lean`, `pickCfg`), so `tiny-do10` and
`tiny-do20` are `tiny` (T = 128, D = 128, 4 heads, 6 blocks, 1.2M parameters) at p = 0.1 / 0.2.
Not implemented: embedding and attention-probability dropout.

⚠ The trainer's own val number — the `── val @ step N` lines in `train.log` and the
"trainer-logged" line `score` prints — runs the val chunks through the *train-step* graph with
the update discarded (`evalValLoss`, lr 0 and `t = 1`), so on a dropout key it is the loss under
one fixed dropout realisation and sits ~0.07 bits above the eval forward's. Its shape (where the
minimum is) is right; its level is not. The numbers below are the `score` mode's P1 / P2 passes
through the eval forward, the same instrument as every other row of the ladder. Routing the
trainer's val through the eval graph is a small follow-up.

```
LEAN_MLIR_SEED=$s LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=0 \
  .lake/build/bin/tinygpt-shakespeare train tiny-do20 10000 32 30          # run.sh (10k), run20k.sh
  .lake/build/bin/tinygpt-shakespeare score tiny-do20 <job>/tiny-do20_params.bin <job>
../2026-10-27-tinygpt-ladder-xla/kn_table.sh runs/2026-10-27-tinygpt-dropout-xla   # KN at T = 128
../2026-10-27-tinygpt-ladder-xla/summary.sh  runs/2026-10-27-tinygpt-dropout-xla
```

~7 min a 10k run on one 4060 Ti (card 0, after the ladder; `queue.log`). Checkpoints (4.9 MB
each) stay on ares; `train.log`, `score.log`, `table.{md,csv}`, `context_curve.csv` per job.

## 10k steps, batch 32, three seeds (`summary.sh`; val, bits/char; KN 7-gram = 2.2056 P1 / 2.1887 P2)

```
job                 train_b   logged       P1       P2  gap to KN (P2), 95% CI
tinydo10_s0           1.696 2.239605   2.1670   2.1316  -0.0570 [-0.0689, -0.0429]
tinydo10_s1           1.794 2.242941   2.1730   2.1410  -0.0476 [-0.0594, -0.0337]
tinydo10_s2           1.763 2.244279   2.1802   2.1537  -0.0349 [-0.0483, -0.0191]
tinydo20_s0           1.828 2.238798   2.1657   2.1263  -0.0623 [-0.0725, -0.0501]
tinydo20_s1           1.894 2.240739   2.1712   2.1320  -0.0566 [-0.0680, -0.0430]
tinydo20_s2           1.894 2.240801   2.1595   2.1199  -0.0687 [-0.0791, -0.0564]
```

| row | P2, by seed | gap to KN 7 (P2), by seed | mean | gap (P1) | val curve at 10k |
|---|---|---|---|---|---|
| `tiny`, no dropout (ladder T1) | 2.809 / 2.807 / 2.799 | +0.621 / +0.619 / +0.610 | +0.616 | +0.54 | minimum 2.263 at step 3–4k, then up |
| `tiny`, 5k steps (ladder T5) | 2.261 / 2.294 / 2.274 | +0.072 / +0.105 / +0.085 | +0.087 | +0.06 to +0.09 | minimum 2.25 at 3–3.5k |
| best `nano` row (RoPE, ladder T4) | 2.221 / 2.225 / 2.215 | +0.033 / +0.036 / +0.027 | +0.032 | +0.054 | flat from 9k |
| **`tiny-do10`** | 2.132 / 2.141 / 2.154 | −0.057 / −0.048 / −0.035 | **−0.047** | −0.039 / −0.033 / −0.025 | minimum at 8.5–9.5k, within 0.007 of the end |
| **`tiny-do20`** | 2.126 / 2.132 / 2.120 | −0.062 / −0.057 / −0.069 | **−0.063** | −0.040 / −0.034 / −0.046 | minimum at the last or second-to-last eval |

## What it says

- **The sign flips.** Every other row of the ladder is behind a 6-character count; both dropout
  rows are ahead, under both protocols, with the 95% block-bootstrap CIs clear of zero on all six
  jobs. The margin is of the size the plan hoped for (`tinygpt_ngram_ladder.md` §1's first case):
  0.03–0.07 bits/char under P2, 0.025–0.046 under P1.
- **It is the overfit that was in the way, not the capacity.** The no-dropout `tiny` has the
  lowest train loss in the ladder (1.26–1.29 bits) and the worst val; at p = 0.2 the train loss
  is 1.83–1.89, nearly the no-dropout `nano`'s, and the val is 0.68 lower. The 1.2M parameters
  were always enough; without regularisation they were spent memorising 1 MB by step 4,000.
- **0.2 beats 0.1, and both are still improving at 10k.** The three p = 0.2 curves reach their
  minimum at the last or second-to-last evaluation and the p = 0.1 ones at 8.5–9.5k, where the
  no-dropout curve turned at 3–4k. The 20k rows below say whether the schedule keeps paying.
- **Where the net wins** (`ngram_ladder_tinydo20_s0.png`, panel b; `context_curve.csv`, net −
  KN 7 averaged over history-length bins, seed 0): behind at histories 1–5 (+0.08) and 6–10
  (+0.04), ahead from 11 on — −0.02 at 11–20, −0.05 at 21–40, −0.06 at 41–64, −0.05 at 65–128 —
  ahead at 95 of the 128 lengths (p = 0.1: +0.07 / +0.03 / −0.04 / −0.05 / −0.06 / −0.04, 91 of
  128). The 10-21 nano lost by 0.17 at 2–5 and only drew level past 11; the regularised `tiny`
  still loses the first ten characters of a chunk to the counts and earns its margin on the rest
  of the window. Panel (c): the same single-character spikes (a speaker tag's first letter, a rare
  word) remain the net's losses; the wins are spread across ordinary text.

## 20k steps (`run20k.sh`; ~11.5 min a run, card 0, 20:15–21:25 UTC)

```
job                 train_b   logged       P1       P2  gap to KN (P2), 95% CI
tinydo10_20k_s0       1.600 2.286050   2.2106   2.1799  -0.0088 [-0.0253, +0.0091]
tinydo10_20k_s1       1.574 2.302549   2.2165   2.1929  +0.0042 [-0.0131, +0.0223]
tinydo10_20k_s2       1.568 2.287970   2.2091   2.1806  -0.0081 [-0.0238, +0.0116]
tinydo20_20k_s0       1.757 2.235915   2.1658   2.1297  -0.0590 [-0.0726, -0.0426]
tinydo20_20k_s1       1.741 2.228575   2.1576   2.1287  -0.0600 [-0.0751, -0.0427]
tinydo20_20k_s2       1.716 2.216658   2.1356   2.1005  -0.0881 [-0.0998, -0.0747]
```

| row | gap to KN 7 (P2), by seed | mean | gap (P1) | val curve (train-graph level) |
|---|---|---|---|---|
| `tiny-do10`, 10k | −0.057 / −0.048 / −0.035 | −0.047 | −0.039 / −0.033 / −0.025 | minimum at 8.5–9.5k |
| `tiny-do10`, 20k | −0.009 / +0.004 / −0.008 | **−0.004** | +0.011 / +0.004 / +0.005 | minimum 2.236–2.253 at 9–11k, end 2.29–2.30 |
| `tiny-do20`, 10k | −0.062 / −0.057 / −0.069 | −0.063 | −0.040 / −0.034 / −0.046 | minimum at the last two evals |
| `tiny-do20`, 20k | −0.059 / −0.060 / −0.088 | **−0.069** | −0.040 / −0.048 / −0.070 | minimum 2.214–2.231 at 17.5–18.5k, end within 0.005 |

- **The rate has to match the schedule.** At p = 0.1 the longer schedule gives the overfit
  back: the val curve turns at 9–11k, exactly where the 10k run ended, and climbs 0.05 by 20k;
  the net lands level with the counts (every P2 CI straddles zero, P1 behind by 0.004–0.011)
  with a train loss of 1.57–1.60 bits, back toward the unregularised `tiny`'s 1.26. At p = 0.2
  the curve bottoms at 17.5–18.5k and the end is within 0.005 of it; the gain over 10k is 0.006
  on the mean (−0.069 against −0.063), with the spread between seeds (0.03) now larger than the
  gain. The longer schedule is worth one more seed's worth of noise at 0.2 and a sign change at 0.1.
- The row to print is `tiny-do20` at 10k: the same steps as the catalogue's recipe, one knob
  changed, 0.06 ahead of the 7-gram with three seeds inside 0.012 of each other. The 20k rows
  are the dose–schedule check, not a better number.

## For the book (§5 of the audit)

The §10.3.1 ladder's sentence becomes: a 1.2M-parameter transformer on 1 MB of Shakespeare is
behind a Kneser-Ney 7-gram by 0.6 bits/char at the catalogue's recipe and ahead of it by 0.06 with
residual dropout at 0.2 and nothing else changed, three seeds each, the gap's sign set by one
regularisation knob rather than by steps, batch, width or position encoding — and doubling the
steps buys nothing at 0.2 and gives the overfit back at 0.1.
