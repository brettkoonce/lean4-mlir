# Pong DQN, 2026-10-27 — the state net's two missing arms (planning/part2_ship_audit.md §2.5)

The September table (`runs/2026-09-25-pong-dqn/RESULTS.md`, Table 1) has the pixel net above the
state net on every seed — pixels +15.47 against state +13.45 — and the book says "not dissected
here". The two arms below are the dissection. Same recipe as September: `lake exe pong-dqn
mode=state … opp=1.5 seed=s tag=…`, XLA, 500k agent steps, 20-game evaluations at ε 0.05 every
25k steps, the final row 20 fresh games; three seeds each, one card, ~5–7 min a run at
`LEAN_MLIR_MEM_FRACTION=0.15` beside the TinyGPT ladder (`run.sh`, `queue.log`, the six logs and
curve CSVs are here).

The six-number state (`Pong.stateVec`) is the ball's position and velocity and both paddles,
already scaled to about [−1, 1], so there is no normalisation arm. What the state net lacks against
the pixel net is two things, and `demos/MainPongDqn.lean` gained a knob for each:

- `k=4` — the last four states stacked, as the pixel net stacks four frames (24 inputs);
- `width=512` — the one-state MLP at the pixel net's dense width (two hidden layers of 512 for
  64; 267,779 parameters for 4,803).

## Table — default opponent (speed 1.5, delay 4), three seeds each

| arm | params | final points/game (seeds 1, 2, 3) | mean (sd) | at 200k | first positive eval |
|---|---|---|---|---|---|
| state, k = 1, width 64 (September) | 4.8K | +14.15, +12.70, +13.50 | 13.45 (0.73) | +11.35, +9.50, +6.20 | 125k, 125k, 100k |
| **state, k = 4, width 64** | 6.0K | +13.45 ± 0.83, +9.20 ± 1.04, +10.55 ± 0.70 | **11.07 (2.17)** | −3.25, +2.10, +5.25 | 225k, 150k, 175k |
| **state, k = 1, width 512** | 268K | +16.85 ± 0.44, +12.75 ± 0.64, +14.65 ± 0.75 | **14.75 (2.05)** | +15.50, +14.60, +13.10 | 100k, 100k, 75k |
| pixels, k = 4 (September) | 1.70M | +15.60, +15.90, +14.90 | 15.47 (0.51) | +14.15, +14.90, +15.00 | 75k, 100k, 75k |

## What it says

- **History hurts the state net.** The six numbers already carry the ball's velocity, so four
  stacked states are the same information four times over 24 inputs; the net is slower to its
  first positive evaluation (150–225k against 100–125k), 2.4 points lower at the end, and the seed
  spread triples. The pixel net stacks frames because a single frame has no velocity; the state
  vector does, and the stacking is not transferable.
- **Capacity closes the gap.** At the pixel net's dense width the one-state MLP finishes at 14.75
  against the pixel net's 15.47 — inside one seed sd of either — and at 200k it is level with the
  pixel net (15.5 / 14.6 / 13.1 against 14.2 / 14.9 / 15.0) where the 64-wide net was 4–9 points
  behind. The September ordering was the 64-wide MLP's capacity, not the representation.
- The two wide-net seeds that end below seed 1 (12.75, 14.65) peaked earlier — seed 2 at 15.50
  at 200k — so the final-row spread is the ε-greedy policy's late drift, the same thing the
  September pixel seeds show at a smaller scale.

For the book (§5 of the audit): the sentence "not dissected here" becomes one line — a state net of
the pixel net's width matches it; stacking states, which the pixel net needs for velocity, costs
the state net 2.4 points because its vector already has velocity.
