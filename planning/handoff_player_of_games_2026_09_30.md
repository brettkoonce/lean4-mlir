# handoff_player_of_games_2026_09_30.md — where the games stand, for the next session

State of the tree at hand-off: **two commits on main, not pushed** (`120021f2` the AlphaZero
tic-tac-toe demo, `55ad3a5a` the search in C), and **34 files staged, not committed**: the
5×5 board, the book's §10.4 Player of games (blackjack, Pong, tic-tac-toe, the AlphaGo family),
the Pong entry, the README's mirror of it, the run record, the planning log. `git status`
shows the staged set; nothing in it needs a rebuild to read. Next session's first jobs: the
copy/content pass over §10.4 and the READMEs with the user, then commit (their word), then
push (their word, separately).

## What exists

- `LeanMlir/TicTacToe.lean` — the game (n×n, k in a row, X first, a position is a base-3
  number over the cells), the dense-table bindings (n ≤ 4), the on-demand solver bindings,
  the players (random, win-or-block, perfect), `Table` dispatching table / solver.
- `ffi/f32_helpers.c`, the `lean_ttt_*` and `lean_mcts_*` sections — the dense table,
  planes, scorers, the MCTS arena (flat per-game trees), the solver (below), the sampler and
  the exhaustive opening enumeration.
- `demos/MainTttEnv.lean` (`lake exe ttt-env`) — the instrument's own gates and the scripted
  pairings; `demos/MainAlphaZeroTtt.lean` (`lake exe alphazero-ttt`) — the trainer; with
  `iters=0 params=<file>` it is the instrument alone on a saved net.
- `scripts/demos/ttt_figure.py` (the stacked 3×3 / 5×5 boards, three curves, the value
  scatter), `scripts/demos/ttt_sweep_stats.py` (the miss breakdown).
- `runs/2026-09-29-alphazero-ttt/README.md` — every number in the table with its log; the
  logs, curves, sweeps, policies; `n5_convbn_collapse.txt` (an excerpt, the log was lost).
- `planning/alphazero_ttt_demo.md` — the plan and, in §10, the whole log of what went wrong
  and why, in order. Read §10 before touching the solver.
- Book: `blueprint/src/content.tex`, `\section{Player of games}` (`sec:player_of_games`) at
  the end of chapter 10, before the three lines `\addtocontents…tocdepth 1`,
  `\hypersetup{bookmarksdepth=1}`, `\appendix` — anything inserted after those prints as
  ".1" with no chapter number (it did, once). Preview: `python3
  scripts/book/blueprint_preview.py --base 5b50416d`, then the user runs
  `! setsid nohup python3 -m http.server 8765 --bind 0.0.0.0 --directory
  /tmp/blueprint_preview > /dev/null 2>&1 &` and reads http://100.76.1.97:8765/diff.html.
- Memory: `alphazero_ttt_demo_state`, `book_game_theory_section`, `pkill_matches_own_shell`,
  `lean_stdout_block_buffered`, `pong-dqn-state`.

## The table, as the documents state it

| | 3×3 | 4×4 | 5×5, four in a row |
|---|---|---|---|
| agree, net alone | 97.3% (all 4,520 decisions) | 99.4% (all 9,062,619) | 87.4% (20k sample ≥ 6 stones); **92.1% over the exhaustive ≤ 3-stone opening** (7,526) |
| vs perfect, net alone, X · O | 0/768/0 · 0/768/0 | 0/768/0 · 0/768/0 | 0/768/0 · 0/708/60 |
| vs perfect, net + search | 0/768/0 · 0/768/0 | 0/768/0 · 0/768/0 | 0/768/0 · 0/745/23 |
| random / win-or-block agree | 58.0 / 94.0% | 60.4 / 95.6% | 45.0 / 77.0% (over the sample) |
| run | 20 it × 25 sims, 2.5 min | 40 × 100, 10.0 min | 100 × 400, 63.8 min |

The net rows pool three independent evaluations of the saved net (seeds 2–4, 256 games
each, `iters=0 params=…`). The in-run readings ("unbeaten with search from iteration 59")
were one draw of the perfect player's random lines; the evaluations are the truth quoted.
The net never loses as X on any board; at 5×5 it still loses as O, and the 200-sim run
lost far more as O (243/13 at its last reading): the search closes the second player's
games and has not closed them yet.

## The 5×5 logic, for the walk-through the user asked for

1. **Why there is no table.** The dense table is one byte per base-3 index: 3¹⁶ = 43 MB at
   4×4, 3²⁵ = 847 GB at 5×5. The reachable space at (5,5,4) is on the order of 10¹⁰
   canonical positions (÷8 for the dihedral symmetries) — beyond a table and beyond the
   solver's time — so "every position" ends at 4×4 for good.
2. **The solver** (`lean_ttt_solver_*`): alpha-beta negamax over the values {0 loss, 1
   draw, 2 win} from the mover's view, to the end of the game. A transposition table keyed
   by the *symmetry-canonical* index (the minimum over the eight views) in a Lean-owned
   ByteArray (keys u64, entries u16; 2²⁸ slots = 2.7 GB, replace-always within 8-slot
   probe runs). Each entry carries a **lower and an upper bound** (merged on store; equal =
   exact) and the **principal move in canonical coordinates** (mapped back through the
   view on read). Before recursing: an immediate win is a win; two opponent threats are a
   loss; one threat forces the block. Move order: the table's move, two killers, then
   moves scored by the lines the mover owns unopposed (threat-makers first), centre-out.
   Win and threat tests are one popcount per precomputed winning line over bitboards.
3. **Why both bounds.** A null-window search leaves a bound; a later query with another
   window re-searches and, with one value+flag per entry, overwrites the earlier bound —
   a draw-valued position flips between "≥ 1" and "≤ 1" forever. Keeping both, two
   one-sided searches add up to an exact entry. It also cut the exhaustive 4×4 check from
   80M to 13M nodes.
4. **The root.** A single null-window search from the empty board never returned; the 25
   openings solved one at a time with the full window take 33 s in total, so the empty
   board's entry is the max over its children. Facts: (5,5,4) is a draw; every opening
   draws; after X centre O's only drawing replies are the four diagonal neighbours (the
   figure's rings, and the net's 27/24/23/22%); the inner-corner opening leaves O one
   drawing reply; a corner opening leaves all 24.
5. **The gate.** `ttt-env n=3` / `n=4` run the solver over every reachable position of the
   dense table: 0 value mismatches and 0 principal moves outside the optimal set, on both
   boards. Four solver versions were caught here (an unproven bound promotion; lost
   positions and the double-threat shortcut storing no move). Nothing at 5×5 is trusted
   that this gate did not pass.
6. **What is measured at 5×5, and why those sets.** (a) The **sample**: 20,000 random-play
   decision positions with at least six stones (`sweepMin=6`), because a fresh 3-stone
   subtree costs seconds to solve and a 6-stone one milliseconds; exact per position, not
   exhaustive, and off the self-play distribution (self-play stood at 4 of them). (b) The
   **exhaustive opening** (`opening=3`): every decision position with at most three stones,
   7,526 — the theory lives there and a sample never sees it; two solves a position (value,
   and the net's move's child). Four stones (76k positions, each child a fresh subtree)
   was tried and stopped. (c) **Match play**: 256 games a side against the perfect player.
7. **The perfect player above 4×4.** Uniform over the optimal set for each side's first
   move (the cached opening book makes that free; the variety between games comes from
   it), then the search's **principal move** from the third stone on — one solve a move,
   the position's own. Valuing every child to draw uniformly was 40 minutes an iteration.
   The distribution at the first move is unchanged: the first optimal move of a uniformly
   random order *is* a uniform draw from the optimal set.
8. **The cache.** `.lake/build/ttt_solver_5x5_k4.bin` (2.6 GB, ignored by git), saved by
   `ttt-env` and by every trainer run, reloaded by the next; the root and the sample are in
   it. Delete it after any change to the entry format. `cap=29` (5 GB) if a table shows
   heavy `replaced` counts.
9. **Cost, warm.** Self-play ~7 s, the instrument ~6 s, training up to 29 s per 5×5
   iteration at 400 sims; a first-time pass over a fresh set of positions is minutes.

## Open items

- Copy / content pass over §10.4 and the READMEs (the user's), then commit, then push.
- The book sentence on the talk's "a day at 4×4" is neutral ("the Python implementation
  this loop follows"); the user's call to keep, cut, or cite the talk.
- The O side at 5×5: more search or a bigger net is the obvious next run; 400 sims × 100
  iterations is where it stands.
- The scripted players' expected agreement over the exhaustive opening needs every child
  of every opening position and ran 40 min unfinished; the baselines stay on the sample.
- The PJRT shim prints its RESIDENT banner on every re-seed of a held forward (Pong's log
  too); a print-once in `ffi/pjrt_ffi.c` is the fix (needs the gcc one-liner rebuild).
- Grad-CAM / Q-value panel for Pong (plan §5) never drawn; `Bestiary/DQN.lean` deferred.
