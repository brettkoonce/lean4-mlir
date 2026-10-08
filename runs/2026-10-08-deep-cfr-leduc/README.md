# Deep CFR on Leduc hold'em — planning/leduc_deep_cfr_demo.md Phases 2–3

Every number here is read by the exact instrument (`LeanMlir/Leduc.lean`, C in
`ffi/f32_helpers.c`): exploitability is the best responder's winnings per hand averaged over the
two seats (NashConv / 2, OpenSpiel's convention), chips with ante 1, so 1 chip/hand = 1,000
milli-antes per game. XLA, one 4060 Ti per run; the trainer was run as the built binary
(`.lake/build/bin/deep-cfr-leduc`, no lake flock held), commit 0c4304a3 + the Gate B / matched-budget
additions.

```
.lake/build/bin/deep-cfr-leduc r=3 T=100 K=1000 steps=1000 stratSteps=4000 every=1 seed=<1|2|3> tag=r3s<seed>
.lake/build/bin/deep-cfr-leduc r=6 ... tag=r6s<seed>        # the scale arm
.lake/build/bin/deep-cfr-leduc r=13 ... tag=r13s<seed>
.lake/build/bin/deep-cfr-leduc mode=tabular T=200 K=100 every=10 tag=gateb      # Gate B, CPU
.lake/build/bin/leduc-env budget=4145000 esSeeds=3 seeds=0                        # ES-MCCFR at the r = 3 budget
```

Files: `r<r>_s<seed>.log` (the run), `r<r>_s<seed>_curve.csv`
(`iter,nodes,steps,exploit_current,exploit_avg,loss0,loss1,ms` — `exploit_avg` is SD-CFR's
exploitability after that iteration), `r<r>_s<seed>_{strategy,sdcfr,esmccfr}.txt` (`[nInfo, 3]` tables as
`idx σ_fold σ_call σ_raise` rows — the strategy net's σ, SD-CFR's average, tabular ES-MCCFR at the
same nodes touched; the trainer's f32 `.bin` originals are gitignored under runs/);
`r<r>_esmccfr.csv` (ES-MCCFR's decade curve, three seeds).

## r = 3 (288 information sets, F = 30, 10,499 parameters per net)

T = 100 iterations, K = 1,000 traversals per player per iteration, reservoirs 1M, 1,000 Adam steps at
batch 512 per advantage net (re-initialised each iteration), lr 1e-3, 4,000 steps for the strategy
net. 4.12–4.17M nodes touched, 204,000 steps, 5.7 min a seed. The strategy reservoir overflowed
(1.35M rows seen, 1M held — reservoir sampling as the paper has it).

| arm (seed 1 / 2 / 3) | exploitability | head-to-head vs CFR+, seat-averaged |
|---|---|---|
| strategy net | 0.229 / 0.185 / 0.152 (mean 0.189) | −0.055 / −0.039 / −0.050 |
| SD-CFR (the T stored advantage nets, own-reach weighted) | 0.127 / 0.150 / 0.109 (mean 0.129) | −0.022 / −0.031 / −0.020 |
| tabular ES-MCCFR at 4.145M nodes (seeds 1 / 2 / 3) | 0.038 / 0.043 / 0.041 (mean 0.041) | |
| CFR+ 1,000 iterations (exact) | 0.000238 | 0 |

Fixed arms (`lake exe leduc-env`): uniform 2.3736, scripted honest 1.050 (−0.123/hand), CFR+ with the
bluffs removed 0.1856 (−0.0115/hand).

Gate C: SD-CFR's exploitability falls from 1.4 at iteration 5 to 0.11–0.15 at 100 (seed noise ±0.03
iteration to iteration; the per-iteration current profile swings 0.4–2.8, as a current CFR profile
does), ends under the honest arm (1.05) and the bluffs-removed arm (0.19). The sampled table wins at
this size by 3×, which is the plan's prediction for r = 3 (§5).

Published bar (read from Steinberger 2019, *Single Deep CFR*, Fig. 1a, Leduc in milli-antes per
game against algorithm iterations at 1,500 traversals per iteration, networks 3 × 64, reservoirs
1M, 750 updates at batch 2,048): both curves start near 650 mA/g at 10 iterations, pass roughly
150–200 at 100 iterations and end near 80 (Deep CFR) and 60 (SD-CFR) at 5,000. Ours at 100
iterations with 1,000 traversals: SD-CFR 109–150 mA/g, strategy net 152–229 — at the published
curve. The figure's exploitability convention (total or average) is not stated in the paper; if it
is total, halve it before comparing. Brown et al. 2019 has no Leduc curve for Deep CFR (its
Leduc number is NFSP's, 37 mbb/g in a footnote); its experiments are in FHP / HULH.

Gate B (`mode=tabular`, T = 200, K = 100 per player, the trainer's traversal with a lookup table in
the net's place; 0.2 s, CPU): the t-weighted average's exploitability 1.13 / 0.28 / 0.24 / 0.25 /
0.21 at 43k / 217k / 424k / 634k / 836k nodes — on ES-MCCFR's curve (0.45 at 100k, 0.10 at 1M).

## The scale arm — r = 6 and r = 13 at the same recipe

Same T, K, reservoirs, steps and net widths (F = 36 and 50; 10.9k and 11.8k parameters); 3.6M and
3.4M nodes touched (a traversal is shorter at larger r: fewer raises survive), 5.8 min a seed. The
tabular ES-MCCFR row is the trainer's own matched-budget bracket (`r<r>_s<seed>_esmccfr.txt`).

| r (sets) | arm | exploitability, seeds 1 / 2 / 3 | head-to-head vs CFR+ |
|---|---|---|---|
| 6 (1,116) | strategy net | 0.113 / 0.157 / 0.157 | −0.038 / −0.041 / −0.041 |
| 6 | SD-CFR | 0.112 / 0.123 / 0.113 (mean 0.116) | −0.024 / −0.025 / −0.025 |
| 6 | tabular ES-MCCFR, 3.6M nodes | 0.053 / 0.066 / 0.058 (mean 0.059) | −0.014 / −0.017 / −0.018 |
| 6 | scripted honest / bluffs removed / CFR+ | 1.285 / 0.160 / 0.00044 | |
| 13 (5,148) | strategy net | 0.115 / 0.132 / 0.168 | −0.041 / −0.034 / −0.036 |
| 13 | SD-CFR | 0.099 / 0.120 / 0.106 (mean 0.108) | −0.029 / −0.026 / −0.028 |
| 13 | tabular ES-MCCFR, 3.4M nodes | 0.113 / 0.107 / 0.102 (mean 0.107) | −0.036 / −0.035 / −0.034 |
| 13 | scripted honest / bluffs removed / CFR+ | 1.466 / 0.087 / 0.00037 | |

Table over net at matched budget: 3.1× at r = 3, 2.0× at r = 6, 1.0× at r = 13 (and the net is
the better head-to-head there, −0.028 against −0.035 per hand). The sampled table's coverage is
1.000 at every budget here, so the gap is not the uniform-where-unvisited effect the plan
guessed at; it is the table's noise at a fixed budget growing with the number of sets while the
net's error does not.

Gate B at r = 6 and 13 (`mode=tabular`, T = 100, K = 100): 0.36 at 364k nodes and 0.43 at 338k —
on ES-MCCFR's decade curves (r = 13: 1.11 at 100k, 0.24 at 1M).

A bug worth the log: the first r = 6 launch froze at a seed-dependent exploitability from iteration
1 (4.4, worse than uniform — the always-fold profile a net's argmax gives when every advantage is
read from the same parameters). A closure defined before the loop had captured the `let mut`
parameter binding at creation; the forward never saw a trained net. Found by dumping the net's
outputs at t = 1 and t = 2 (identical) and the parameters (different); fixed by passing the
parameters in. The r = 3 runs predate that refactor. The trainer now throws on a non-finite
advantage instead of folding silently.
