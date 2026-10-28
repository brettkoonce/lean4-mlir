# 2026-10-27 — the Part 2 grind: planning/part2_ship_audit.md §2 on ares, four cards

The MNv4 500-epoch verified half (unit mnv4-e500-queue) was stopped at its epoch-190 checkpoint at
17:30 UTC to free the box; it resumes from there afterwards. Each card runs its own chain
(`queue_card<N>.sh`, launched with `setsid -f nohup`); every run directory has its own `run.sh`
and `queue.log`.

| card | first | then | then (`queue_card<N>v.sh`, the evening's addition) |
|---|---|---|---|
| 0 | `runs/2026-10-27-tinygpt-ladder-xla/` — T1–T5, three seeds (~2.5 h); beside it, at `LEAN_MLIR_MEM_FRACTION=0.15`, `runs/2026-10-27-pong-state-arms/` (state `k=4` and `width=512`, three seeds each, ~30 min) | `runs/2026-10-27-boltzmann-seeds/` — flow seeds 1–3 (minutes); then `queue_card0b.sh`: `runs/2026-10-27-tinygpt-dropout-xla/` — T6, `tiny-do10` / `tiny-do20` × 3 seeds (~40 min) | the dropout pair at 20k steps (`run20k.sh`, ~80 min); `runs/2026-10-27-visdrone-rows-seeds/` `aug50` and `ctrl12` at seed 1 (~4.3 h) |
| 1 | `runs/2026-10-27-visdrone-aff30-seeds/` seed 1 (~2.7 h) | `runs/2026-10-27-brats-r34-seeds/` r34 / scratch / noskip at seed 1 (~2.5 h) | `long50` and `cfoc2` at seed 1 (~4.3 h) |
| 2 | `runs/2026-10-27-visdrone-aff30-seeds/` seed 2 (~2.7 h) | `runs/2026-10-27-brats-r34-seeds/` the three arms at seed 2 (~2.5 h) | `long50` and `cfoc2` at seed 2 (~4.3 h) |
| 3 | `runs/2026-10-27-neudet-seeds/` — four arms × seeds 1, 2 (~3 h) | its val and test sweeps (~2.5 h) | `aug50` and `ctrl12` at seed 2 (~4.3 h) |

The two §3 code bits that blocked a run landed the same afternoon, uncommitted at launch:
`transformerEncoder`'s `dropout` (GPT-2 residual dropout in the generic emitter's train step,
in-graph Philox masks keyed on the step; `tiny`'s graph byte-identical at 0) and `pong-dqn`'s
`k=` / `width=` for the state net.

The fourth column was added at 19:30 UTC when the user chose to spend a day or two of the box on
Part 2 rather than resume the MNv4 pair: §7.4's other four rows of `tab:visdrone` at seeds 1 and 2
(every row of the table becomes three seeds; ~17 GPU-h) and the dropout arm at 20k steps. Each
waiter polls for its card's chain (`pgrep -f 'bash .*queue_card<N>\.sh'`) and starts when it
ends. Not run: the MuZero branch (§4), the denoising demo (§6).

**DONE 07:58 UTC on 10-28, 14.3 h after launch.** Landed, READMEs written: the ladder T1–T5
(19:45), Pong (18:38), Boltzmann (19:45), VisDrone aff30 seeds (20:33), dropout 10k (20:15) and
20k (21:25), BraTS seeds 1 and 2 (23:18), NEU-DET four arms × two seeds + sixteen sweeps (03:47;
the FPN sweeps ran ~35 min a split beside three busy chains, not the ~10 the script expected).
VisDrone rows: cards 0–2 done by 04:12 (`aug50_s1`, `ctrl12_s1`, `long50_s1/s2`, `cfoc2_s1/s2`);
card 3 ran `aug50_s2` from 03:47 (3.2 h once the other cards went idle) and `ctrl12_s2` to 07:58,
the last run of the grind. Card 0, idle from 03:16, rescored the four seed-0 checkpoints under
both decodes (every historical argmax number reproduced), then the `cfoc2` e6/e8/e10 ladders for
three seeds (the row came in 0.1961 / 0.1350 / 0.1516 — the cosine tail decides it;
`…-visdrone-rows-seeds/README.md`), then the seed-0 `aug50` / `long50` e10–e40 ladders. Disk at
08:00: 63 GB free. The MNv4 pair resumed on the user's word at 12:12 UTC (unit
`mnv4-e500-resume`; the verified half from epoch 190, after a stale-shim precheck refusal fixed
by rebuilding `ffi/libpjrt_ffi.so` — `runs/2026-10-26-mnv4-full500-jax/queue.log`).

**What the grind settled, one line a run** (each run dir's README has the tables):

| run | the book's number | three seeds | what moves |
|---|---|---|---|
| TinyGPT ladder T1–T5 | "RoPE stays coherent at T=128/256" | no row crosses the KN 7-gram (best gap +0.032); xeval 2.9–5.4 b/c | the claim falls; only the load/refuse gate holds |
| TinyGPT dropout T6 (new code) | — | `tiny-do20` 10k crosses the 7-gram on every seed (−0.057 to −0.069, CIs exclude 0); 20k buys nothing at 0.2, hands the overfit back at 0.1 | the sign flips; the row to print |
| Pong state arms (new code) | pixels 15.47 over state 13.45, "not dissected here" | state k=4 11.07 (history hurts), width 512 14.75 ≈ pixels | September's ordering was the 64-wide MLP's capacity, not the representation |
| Boltzmann seeds | gate A basin bar 0.03 | energy 3.2–6.3× on four seeds; raw basin 0.031–0.039 fails on 3/4, reweighted within 0.008 on all | gate A restated on the reweighted row |
| VisDrone aff30 | 0.2363 | 0.232 ± 0.004 | the book's is the best of three; rare classes agree to 0.002 |
| BraTS r34 / scratch / noskip | "bootstrap ties; buys epoch 1" | +0.002/+0.004/+0.005 paired (same sign ×3); e1 gap +0.064/−0.035/−0.003; skips +0.103 ± 0.005 | bootstrap is a small consistent gain, not a tie; "epoch 1" is gone |
| NEU-DET four arms | 0.623 / 0.607 grid | recipe 0.632 ± 0.008 (book's = low seed); bootstrap +0.071 ± 0.011; grid −0.023 ("within 0.02", not "matches"); HSV off +0.014 same sign on all six rows | bootstrap holds; grid and augmentation sentences soften |
| VisDrone rows ×4 | 0.1961 / 0.1674 / 0.1526 / 0.1243 (mixed decodes) | multilabel: `cfoc2` 0.161 ± 0.032, `aug50` 0.170 ± 0.003, `ctrl12` 0.157 ± 0.004, `long50` 0.1287 ± 0.0011 | "12 epochs still beats 50" falls (the lucky seed of a focal recipe whose cosine tail decides it); "the pack sets whether the long schedule pays" stands on the ladders |
