# pong_dqn_demo.md — DQN from pixels on a Pong written in Lean

Goal: the Chapter 3 pairing. The MNIST CNN's kit as a Q-function on 84×84
frames, Mnih et al. 2015's recipe (frame skip, a stack of four frames,
replay, a target network, ε-greedy) on a Pong written in Lean, so the demo
has no emulator, no ROM and no Python in the loop. The bracket is built in:
the same DQN on the game's six-number state is the ceiling for what pixels
can learn, random is the floor, and a one-frame input is the ablation that
shows why the stack of four exists. The loop is `blackjack_dqn_demo.md`'s,
generalised over the environment; blackjack is rung 2 there, this is rung 3.

## 0. Three ways to get a Pong, and why this one

- **ALE through a Python sidecar.** Real Atari Pong; `ale-py` ships the ROM
  now. The repo already streams ImageNet through a generated Python shim, so
  a bidirectional env shim (action in, frame and reward out over a pipe) is
  the same pattern with a reply channel. Cost: the shim, one to two million
  frames to reach +20, hours per run with tuning risk, and "Atari" in the
  sentence. A one-pager later if that sentence is wanted.
- **MinAtar, Young & Tian 2019.** Breakout, Freeway, Seaquest and two more
  on 10×10 binary planes, built for cheap DQN research, with published DQN
  scores per game. About 200 lines of numpy per game to port to Lean. The
  fallback if a literature number is wanted; it has no Pong.
- **Pong in Lean.** About 150 lines of pure functions, rendered straight to
  84×84, deterministic from a seed, opponent skill as a knob, and the state
  vector is ours so the state-DQN ceiling exists. Dependency-free the way
  the 2-D diffusion toy is. This one.

## 1. The game

- Field 84 × 84. Paddles 2 px wide and 12 tall at x = 4 and x = 79. Ball
  2 × 2.
- Ball speed |vx| = 1.5 px per frame at serve, +0.25 per paddle hit, capped
  at 4. vy is set by where the ball meets the paddle (offset from centre ×
  0.4) and reflected at the walls.
- Player actions {stay, up, down}, paddle speed 2 px per frame.
- Opponent: a scripted paddle that tracks the ball's y at speed s_o after a
  reaction delay of d frames. s_o = 1.5, d = 4 is beatable and not trivial;
  both are knobs and one is Table 2.
- A point when the ball passes a paddle: reward ±1, re-serve toward the
  scorer. Episode is a game to 21 points, capped at 4000 frames.
- Render: u8 grayscale, background 0, paddles and ball 255. Frame skip 4
  (action repeated, reward summed, last frame kept; nothing flickers, so no
  max-pool over frames). Observation: the last four skipped frames stacked,
  [4, 84, 84], scaled to [0, 1] at the network.
- `step : State → Action → StdGen → State × Float × Bool × StdGen` and
  `render : State → ByteArray`, pure, the RNG threaded for serves.

Sanity before anything trains: the scripted paddle against itself scores
about 0 per game by symmetry; random against it scores about −21. Those are
the bottom two rows of Table 1 and the check that the game is a game.

## 2. The network — Chapter 3's kit at 84 × 84

```
.conv2d 4 32 8 .same .relu, .maxPool 4 4,        -- 84 → 21
.conv2d 32 64 4 .same .relu, .maxPool 2 2,       -- 21 → 10
.conv2d 64 64 3 .same .relu,
.flatten, .dense (64 * 10 * 10) 512 .relu, .dense 512 3 .identity
```

The Nature architecture with pooling standing in for stride: `.conv2d` has no
stride argument, `.convBn` does, and BatchNorm has no business in a
Q-function whose batches mix a moving policy's states. About 3.4M params,
the dense dominating. `imageH := imageW := 84`, four input channels for the
stack.

The state-vector twin: `.dense 6 64 .relu, .dense 64 64 .relu, .dense 64 3
.identity` on (ball x, y, vx, vy, own paddle y, opponent y), scaled to
[−1, 1]. That is the blackjack net with six inputs, and it is the ceiling
row: nothing the pixels contain is missing from it.

## 3. The loop — blackjack's, with frames

- Replay: 100k transitions. Frames stored once as u8 (7 KB each, 700 MB)
  and stacks rebuilt by index at sampling time; never store f32 stacks.
- Batch 32; Adam 1e-4; γ = 0.99; ε from 1.0 to 0.1 over the first 100k
  agent steps then held, 0.05 at evaluation; target parameters copied every
  1000 gradient steps; one gradient step per four agent steps; 10k random
  steps to seed the buffer.
- Loss: the rank-2 DDPM MSE block with `ddpmOutShape := [B, 3, 1, 1]` and
  the target built on the host, the taken slot replaced by
  `r + γ (1 − done) max_a' Q_target(s', a')`, one extra forward at batch 32.
  The paper's Huber loss is a stability clip; at reward ±1 the MSE is fine,
  and a fifteen-line Huber branch is the fallback if it is not.
- Budget: 500k agent steps, two million frames, 125k gradient steps.
  Acting forwards at batch 1 about 0.5 ms → four minutes; training steps
  about 3 ms → six minutes; the environment is free. A run is a quarter
  hour on one card, which is a probe by the house rules, not a long run.
- Evaluate every 25k steps: 20 games at ε = 0.05, mean points per game, and
  dump the frame the network saw as a PPM so a blank or clipped observation
  cannot hide.

The environment interface shared with blackjack is a structure of `reset`,
`step`, `render`, `obsDim`; the replay, ε schedule, target copy and
evaluation loop are written once.

## 4. Bracket and tables

**Table 1.** Rows: random, scripted paddle against itself, pixel DQN with
one frame, pixel DQN with four frames, state-vector DQN. Columns: mean
points per game over 100 evaluation games with standard error, agent steps
to the first positive game, wall-clock. The lead-in: one frame cannot see
velocity, and the row says by how much.

**Table 2.** Opponent speed s_o ∈ {1.0, 1.5, 2.0} against the four-frame
pixel DQN: where learning stops, and whether the state-vector twin stops at
the same place.

## 5. Figure

Left: four consecutive frames of a rally with the three Q-values printed
under each, the paper's own Figure 3 idea — Q rising as the ball approaches
an open return. Middle: learning curves for state, pixels with four frames
and pixels with one, against agent steps, with the random and self-play
lines. Right: Grad-CAM of the Q-network on a frame, through
`demos/probes/MainGradCAM.lean` reused on this net: where it looks should be
the ball and the paddles, and if it is the score digits the frame has
leaked the score. (It cannot: the render has no digits. Say so.)

## 6. Section and bestiary

The RL subsection's second demo paragraph, after blackjack, sharing the
`Bestiary/DQN.lean` entry. Three things change from blackjack: the
observation is an image, so the Q-function is the Chapter 3 CNN; the state
is partially observed from one frame, so the input is a stack; the target
network exists because the regression target moves with the parameters.
No dataset row.

## 7. Phases

```
Phase 0 (½ session, no GPU):  the game, the renderer, self-play and random baselines, a PPM dump
Phase 1 (½ session):          state-vector DQN on the shared loop;
                               Gate A: beats the opponent within 200k agent steps
Phase 2 (1 session + 15 min): four-frame pixel DQN; Gate B: within 5 points of the state row
Phase 3 (½ session):          one-frame ablation, the opponent-speed sweep, figure, section
```

## 8. Gates that fail loudly

- Self-play must sit near 0 and random near −21 before any training run;
  a game that fails either is not Pong.
- The state-vector DQN must win first. If pixels fail while state succeeds,
  the loop is right and the vision is wrong; if state fails, the loop is
  wrong, and no amount of CNN tuning will show it.
- Every evaluation dumps the observed frame. The failure mode of a
  from-pixels demo is a network staring at the wrong tensor for an hour.

## 9. Out of scope

Real Atari through ALE; Double, Dueling and Rainbow beyond a flag;
policy-gradient methods; any game other than this one.
