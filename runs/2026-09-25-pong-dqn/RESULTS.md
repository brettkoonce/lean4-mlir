# Pong DQN, 2026-09-25 (planning/pong_dqn_demo.md Phases 1–3)

`lake exe pong-dqn mode=state|pixels [k=4|1] [opp=1.5] [seed=1] tag=…`, XLA, 500k
agent steps (2M frames), replay 100k, batch 32, Adam 1e-4, γ 0.99, ε 1.0 → 0.1 over
100k steps, target copy every 1000 updates, 10k random seed steps, one update per four
agent steps. Evaluation: 20 games in lockstep at ε 0.05 every 25k steps; the final row is
20 fresh games after training. Score = own points − opponent's in a game to 21.
State runs 4.5–6 min alone (15 min two to a card); pixel runs ~102 min, two to a card
on cards 1–3 (`launch_all.sh`).

## Table 1 — default opponent (speed 1.5 px/frame, delay 4)

| arm | final points/game | first positive eval | at 200k |
|---|---|---|---|
| random (`pong-env 100`) | −17.88 ± 0.20 | — | — |
| scripted tracker (`pong-env 100`) | +11.18 ± 0.37 | — | — |
| state DQN, seed 1 | +14.15 ± 0.59 | 125k | +11.35 |
| state DQN, seed 2 | +12.70 ± 0.60 | 125k | +9.50 |
| state DQN, seed 3 | +13.50 ± 0.57 | 100k | +6.20 |
| pixel DQN, 1 frame | +4.60 ± 1.48 | 100k | +2.30 |
| pixel DQN, 4 frames, seed 1 | +15.60 ± 0.42 | 75k | +14.15 |
| pixel DQN, 4 frames, seed 2 (resident path) | +15.90 ± 0.71 | 100k | +14.90 |
| pixel DQN, 4 frames, seed 3 (resident path) | +14.90 ± 0.62 | 75k | +15.00 |

Gate A (state beats the opponent within 200k): passes, all three seeds.
Gate B (4-frame pixels within 5 points of state): passes, and pixels sit ABOVE the
state net on every seed: pixels +15.47 (sd 0.51 across 3 seeds) vs state +13.45 (sd 0.73).

## Table 2 — opponent speed (delay 4)

| opponent speed | state DQN | pixel DQN, 4 frames |
|---|---|---|
| 1.0 | +14.45 ± 0.46 (first + at 100k) | +17.75 ± 0.37 (50k) |
| 1.5 | +13.45 (3 seeds) | +15.60 ± 0.42 (75k) |
| 2.0 | +9.80 ± 0.86 (150k) | +14.45 ± 0.42 (125k) |

## Departures from the plan

- Pixel net: 7×7 / 5×5 kernels for the paper's 8×8 / 4×4 (`.same` is spelled for odd
  kernels) and pools 3, 2, 2 (84 → 28 → 14 → 7) so the flatten is the paper's 7×7×64;
  1.70M params. A 2×2 max-pool on an odd input (21) breaks the generic train step:
  the forward pads to 22, the backward reads 21 (`%plpad3` type mismatch at compile).
- Random scores −17.9, not the plan's −21 (random returns win some points off the
  delayed opponent; not dissected). The gate is re-stated at the measured number.
- Pixel runs are ~6× the plan's 15 min: an update is ~24 ms of which the GPU is busy
  ~4 ms (nsys); the rest is the DDPM train step's shim re-pushing params + Adam m, v
  (~20 MB) every step. These runs predate the fix below.

## Speed after device residency (same day, `PJRT_FFI_RESIDENT=1`)

`trainStepAdamF32DdpmR` keeps [θ|m|v] on the card, `readParamsPrefix` returns θ
alone for the Q-forwards (new optional shim export `pjrt_ffi_resident_read_prefix`),
the forwards hold their parameters (the env var was what their `nResident` needed all
along), and the replay gather is one C pass into last batch's buffer.

| per update, pixels k=4, one card alone | copying | resident |
|---|---|---|
| train step | 12.2 ms | 5.1 ms |
| 100k agent steps incl. evals | — | 329 s (14.6 ms/update) |
| full 500k run incl. evals | ~50 min (est.; 102 min two to a card) | 27.3 min (seed 2) |

Gate (`scripts/det_shim.sh` shim, state 1000 updates and pixels 100 updates): copy ×2
and resident bit-identical; `PJRT_FFI_FAULT=2` (stale retained params) differs. On
the shipping shim resident ≠ copy in the last bits from update 2 (1–49 floats per
tensor, ≤ 2e-5 relative), autotuning's kernel choice, not transport.

`pong_dqn.png` = `~/loiter-venv/bin/python scripts/demos/pong_figure.py runs/2026-09-25-pong-dqn out.png 20`.
The `*_seen_20.pgm` files are the input tensor as the net saw it at evaluation 20
(decoded back from the f32 batch).
