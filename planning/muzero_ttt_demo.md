# muzero_ttt_demo.md — MuZero on tic-tac-toe, its learned model scored against the solved game

**Opened 2026-10-09.** Goal: a Player of games demo that plans **without the rules**, on the
tic-tac-toe instrument the AlphaZero demo already built (`alphazero_ttt_demo.md`). AlphaZero
is handed `step`; MuZero (Schrittwieser et al. 2020, *Nature*) learns a model of the game
and searches inside it. Win rate cannot tell the two apart — both should be unbeaten at
3×3. The solved game can: every reachable position has an exact value and optimal-move set,
so the learned model can be scored at every position, after every number of imagined
moves, on lines the agent plays and lines it never plays.

The question the section asks: **did MuZero learn tic-tac-toe, or only its values?**

The literature predicts the second. MuZero's model is trained to be *value-equivalent*
(Grimm et al. 2020): it only has to predict the quantities planning reads — values,
policies, rewards along the agent's own lines — not the board. He, Moerland, de Vries &
Oliehoek (2023) find that such a model is accurate on the policy it was trained alongside
and degrades on other policies. Their environments (CartPole, LunarLander and the like)
have no exact answer, so they measure the degradation against returns that are themselves
estimated. Here the answer is exact, exhaustive at 3×3 and 4×4, and split by whether
self-play ever visited the position.

## 0. Prior art (searched 2026-10-09, three queries; "not found" means not found)

| work | what it did | what it leaves open here |
|---|---|---|
| Schrittwieser et al. 2020 (*Nature*; arXiv:1911.08265) | MuZero: h / g / f, latent MCTS, K-step unrolled training; Go, chess, shogi, Atari | model fidelity not measured against the true game |
| Grimm, Barreto, Singh & Silver 2020 (NeurIPS) | the value-equivalence principle: a model need only match the Bellman updates of the policies and values that matter | a principle, no board game scored |
| He, Moerland, de Vries & Oliehoek 2023 (arXiv:2306.00840; ECAI 2024) | "What model does MuZero learn?": value-equivalent in practice; poor at evaluating unseen policies; the policy prior keeps search on lines where the model is accurate | control tasks, estimated returns — no exact ground truth |
| de Vries, Voskuil, Moerland & Plaat 2021 (arXiv:2102.12924; ICML 2021 workshop) | "Visualizing MuZero Models": latent space near-collapsed at init, groups states by value; observed latent vs imagined trajectories diverge; two regularisers | MountainCar-scale; qualitative |
| arXiv:2411.04580 (title not checked) | reconstruction + state-consistency losses on MuZero; latent analysis on 9×9 Go, Gomoku, Atari; dynamics less accurate over longer unrolls | no exact per-position scoring |
| Duvaud, muzero-general (GitHub) | MuZero on tic-tac-toe, Connect Four, CartPole … | win rate only |

The new piece is the instrument: exact, exhaustive scoring of a learned game model on a
solved game. Confirm before the section claims it (one more search pass at Phase 5; check
2411.04580's title and authors).

## 1. What already exists (reused unchanged)

- The game, the dense table (3×3, 4×4), the 5×5 solver, the scripted players, the sweep
  scorer, the plane builder, the dihedral augmentation: `LeanMlir/TicTacToe.lean`,
  `lean_ttt_*` in `ffi/f32_helpers.c`, `lake exe ttt-env`.
- The AlphaZero trainer `demos/MainAlphaZeroTtt.lean` (`lake exe alphazero-ttt`) and its
  runs (`runs/2026-09-29-alphazero-ttt/`): the AlphaZero row of the table, and the code
  shape (lockstep self-play, the held eval forward, matches vs perfect).
- The loss trick: a host-written target for the squared-error block so its gradient is a
  chosen cotangent g (`lean_ttt_targets`; Pong's pattern).
- `Bestiary/MuZero.lean` (`tinyMuZero*`) — the three-network split. Its conv-BN bodies are
  NOT the trained nets, for AlphaZero's reason (§2 there: BN over nine binary cells
  diverged at iteration 11; with BN the eval and train forwards differ).

## 2. The one new codegen piece — a VJP module with the input cotangent

The kit emits two modules per spec: `generateTrainStep` (forward, backward, optimizer in
one call) and `generateEval`. MuZero's loss reaches h and g through the unroll
`s₀ = h(o)`, `s_{k+1} = g(s_k, a_k)`, so g's backward has to hand a cotangent to the
previous step and h's receives the sum of everything downstream. That needs:

- `generateVjp spec B`: `(θ, x, ḡ) ↦ (∂θ, ∂x)`. The backward walk is
  `emitTrainBackward` as it stands; the change is to keep the first layer's input
  gradient (every conv already emits one through `conv2dHasVJP3`; the first layer's is
  currently dead) and return it, with no optimizer tail.
- `generateApplyAdam spec`: `(θ, m, v, ∂θ, t) ↦ (θ', m', v')`, the existing
  `emitAdamUpdate` per parameter. g is applied K times per sample, so its ∂θ is the sum
  of K VJP calls; the sum is one C loop (`lean_f32_axpy`-style in `f32_helpers.c`).

Gates (Phase 1, before any game code):
- **Parameter-gradient identity:** `generateTrainStep` at SGD, lr = 1, no momentum, no
  weight decay gives θ − ∂θ; the VJP module's ∂θ must reproduce that difference to the
  float on the same batch.
- **Input cotangent:** central finite differences of ⟨ḡ, net(x)⟩ in x on a tiny spec
  (f32, a handful of directions), and the JAX reference's `jax.vjp` on the same weights
  (`jax/` is the reference implementation).
- Demo modules only: no `verified_mlir/` artifact, so no render-guard row. If it later
  wants a tie, the backward is the same walk the step ties already cover.

## 3. The networks — the AlphaZero trunk, split four ways

All plain conv, no BN, `imageH := imageW := n`, latent `[C, n, n]` with C = 32 (a knob).

```
h  representation  [2, n, n] → [C, n, n]   conv 2→64 3×3, conv 64→64 3×3, conv 64→C 3×3
g  dynamics        [C+1, n, n] → [C, n, n] conv (C+1)→64 3×3, conv 64→64 3×3, conv 64→C 3×3
f  prediction      [C, n, n] → n² + 1      AlphaZero's head: conv C→4 1×1, flatten, dense 64, dense n²+1
r  reward          [C, n, n] → 1           conv C→4 1×1, flatten, dense 64, dense 1
```

- The action enters g as one extra plane, one-hot at the played cell (the paper's
  board-game encoding), concatenated on the host side of the call (in C).
- The reward reads the NEXT latent, `r(s_{k+1})`: in a board game the reward is the
  outcome of the move just made, a function of the resulting state, and a separate spec
  keeps NetSpec linear (no branch off g's trunk).
- Latent normalisation: min-max to [0, 1] per example after h and after every g (the
  paper's), in C with its VJP (`lean_muz_minmax` / `_back`). The degenerate-latent
  failure de Vries et al. describe is what it guards against; a range under 1e-6 is
  clamped, and the number of clamped examples is printed every iteration.
- Values and rewards in two-player zero-sum form from the mover's view, as AlphaZero's
  demo: the value of a child is the negation of its own, the reward of a winning move +1.

## 4. The loss and the backward

Per sample, a position o_t from a stored game, the next K actions a_t … a_{t+K−1}
(K = 5, a knob), and targets for k = 0 … K: the MCTS visit distribution π_{t+k}, the game
outcome z from the mover's view at t+k, and the reward u_{t+k} (+1 on the winning move,
0 otherwise). Past the end of the game: absorbing states, uniform policy target, value
and reward 0 (the paper's convention).

```
L = Σ_k [ (z_k − v_k)² − π_kᵀ log p_k ] + Σ_{k≥1} (u_k − r_k)²
```

delivered per call as the MSE-block target, exactly as AlphaZero's demo does for f, and
the same way for r. Backward in reverse over the unroll: f's and r's VJPs give ∂s at each
step, the min-max VJP, g's VJP gives ∂θ_g (summed) and ∂s_k, which joins f's cotangent at
s_k; at k = 0 h's VJP. The paper's two scalings are host multiplies: the loss of each
unrolled step by 1/K, and the cotangent entering each g by 1/2.

Per training step at K = 5: one h, five g, six f, five r forwards (cached), the same VJPs
in reverse, four Adam applies. Small nets at batch 64–256; the call count, not the FLOPs,
is the cost.

## 5. The search — MCTS in the latent

`lean_mcts_*` keys nodes by the true position's index (the hash that gives AlphaZero its
transpositions). A latent tree cannot: a node is a path. New C section `lean_muz_mcts_*`,
same arena discipline (flat per-game arrays, one Lean-owned buffer):

- node = (parent, action, prior[n²], N, W, reward, latent slot); the latent itself lives in
  a `[nodes, C, n, n]` host buffer the next batched g call reads.
- select by PUCT with the paper's min-max-normalised Q; expand = one batched g over every
  game's pending leaf, then one batched f and one batched r; backup with the negamax sign
  flip and the reward.
- legal moves masked at the root only, from the real rules (the paper's choice for board
  games); inside the tree every cell is allowed — the model has to have learned which
  cells are taken. How often the search's principal line plays an occupied cell is
  measured (§6 M4).

**Oracle gate (the one that catches search bugs):** replace g with the true dynamics
(the latent of the real next position, `h(planes(step(s, a)))`, terminal and reward from
the rules). With f ∘ h a trained AlphaZero net's two heads read through one representation,
latent MCTS must reproduce `lean_mcts_*`'s root visit counts **exactly** on the same
seeds — the same tree up to transpositions, so the check runs at depth ≤ 2 where no
transpositions occur, and on visit distributions at full depth within a stated tolerance.

## 6. The instrument

Each measurement is exact per position. At 3×3 every set is exhaustive; at 4×4 the
sequence sets are sampled with a fixed seed and the sizes stated.

- **M1 Play.** W/D/L vs perfect as X and as O, net alone (argmax of f ∘ h, real legality
  mask) and with latent search; vs random. The AlphaZero demo's protocol, pooled seeds.
- **M2 Root.** f ∘ h over every reachable decision position: policy agreement with the
  optimal set, value MSE and sign — directly beside AlphaZero's 97.30% / 99.42%.
- **M3 Imagined lines.** From each reachable position s, a k-move legal line
  (k = 0 … 2K, so beyond the trained horizon): f(g^k(h(s), a₁ … a_k)) scored against the
  exact value and optimal set of the TRUE position the line reaches. Lines from three
  sources: the agent's own policy, perfect play, uniform random legal moves. He et al.'s
  claim is the gap between the first and the third; here it is a curve against k with
  an exact y-axis.
- **M4 Rules.** (a) Terminal detection: r on every true move — the winning moves' recall,
  the non-winning moves' false-positive rate, by k. (b) Occupied cells: after an illegal
  move inside the model, the change in predicted value and policy; the share of latent
  search lines that pass through one. (c) Transpositions: pairs of move orders reaching
  the same board — latent distance and value disagreement between the two imagined
  states, and against h of the true board. (d) Board probe: a linear readout from the
  latent to the n² cell states, closed-form ridge in C, fit on h(o) of real positions and
  scored on g^k latents. "Decodable at depth k" is the nearest thing to "knows the board".
- **M5 Coverage.** Every M2–M4 number split visited-by-self-play / never visited (the
  AlphaZero runs stood at 44.5% of 3×3 decision positions and 1.2% of 4×4's).

## 7. The one table

Rows: random, win-or-block, AlphaZero (net alone, + search; the existing runs, same
budgets re-stated), MuZero (net alone, + latent search), perfect. Columns per board: root
agreement (M2), imagined-line agreement at k = K on own / random lines (M3), terminal
recall (M4a), W/D/L vs perfect as X and as O, wall clock.

## 8. The one figure

Left (the object itself): one 3×3 position and a five-move line played only inside the
model — the true boards along the line beside the board probe's readout of each imagined
latent, the policy heatmap at each step, cells the model has lost track of marked.
Middle: agreement against imagined depth k, own / perfect / random lines, AlphaZero's
root agreement as a horizontal reference, K marked. Right: terminal recall and the board
probe's accuracy against k.

## 9. Phases

```
Phase 0 (no GPU):  this doc; placement in the book (§10, open); the prior-art pass.
Phase 1:           generateVjp + generateApplyAdam; the two gates of §2. Small spec only.
Phase 2:           demos/MainMuZeroTtt.lean (`lake exe muzero-ttt`, AlphaZero's knobs + K, C);
                   lean_muz_mcts_*, lean_muz_minmax, the trajectory replay; the oracle gate
                   of §5; a 3×3 smoke (one iteration).
Phase 3:           3×3 run. Gate A as AlphaZero's: unbeaten vs perfect both sides with
                   search. Then M2–M5 on the saved net (`iters=0 params=<file>`, the
                   AlphaZero trainer's knob).
Phase 4:           4×4, same binary (ask first: AlphaZero's 4×4 was 10 min with the C tree;
                   MuZero is ~3–5× the calls a step and a deeper search a move).
Phase 5:           figure, section, README, run README, bestiary cross-link
                   (`Bestiary/MuZero.lean` ↔ the trained specs), prior-art re-check.
```

Optional arm, after Phase 4: the state-consistency loss (g(s_k, a_k) pulled toward
stop-grad h(o_{k+1}); EfficientZero, and the 2411.04580 line). Same instrument; the
question becomes whether one extra term turns a value-equivalent model into one that
knows the board. Costs one more h forward per unrolled step and no new codegen.

## 10. Open decisions (user's)

- **Placement.** (a) its own subsection after tic-tac-toe, "Tic-tac-toe without the rules
  — MuZero against the solved game", one figure and one table; or (b) a second half of the
  tic-tac-toe subsection, its table gaining MuZero rows. (a) keeps the one-figure rule;
  (b) keeps the section at three games (`demo_slate.md` §1).
- **5×5.** The instrument works there through the solver and sampled sweeps
  (`alphazero_ttt_demo.md` §11); M3 at 5×5 costs a solve per line endpoint. Default: not
  in scope until 4×4 reads.
- **Blackjack's removal** (Leduc plan §8) is independent and can land first.

## 11. Gates that fail loudly

- §2's VJP gates before any game code; §5's oracle gate before any MuZero training.
- The min-max clamp count printed every iteration: a collapsed latent stops the run
  rather than training on constants.
- Batched shapes checked at startup for all four specs and the action-plane concat.
- M3 and M4 score against the table (n ≤ 4) or the solver (n = 5), never against a
  heuristic; the scorer refuses a position the table marks unreached.

## 12. Out of scope

Stochastic MuZero, Sampled MuZero, Gumbel MuZero's root selection, reanalyse, Connect Four
(well-trodden: alpha-zero-general, Oracle's 2018 series, Kaggle ConnectX), Go.

## 13. Log

- 2026-10-09: doc opened. Prior-art pass (three standard searches): He et al. 2023,
  de Vries et al. 2021, arXiv:2411.04580, muzero-general; nothing found that scores a
  learned MuZero model against a solved game.
