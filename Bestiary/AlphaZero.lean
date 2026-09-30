import LeanMlir.Spec

/-! # AlphaZero — Bestiary entry

AlphaZero (Silver et al., 2018) is a self-play reinforcement-learning system
that plays Go, chess, and shogi from scratch. At its heart is a very
simple neural network: a stack of residual blocks feeding **two heads** —
one predicting the move probability distribution (policy), one predicting
the expected game outcome (value).

The body is "just a ResNet." The novelty is the self-play training loop,
not the architecture. From a `NetSpec` perspective, AlphaZero is exactly
a ResNet body (`Bestiary/ResNet.lean`), forked at the end into two tiny heads.

```
           Input planes (board state encoding)
                       │
                       ▼
          ┌─────────────────────────┐
          │  convBn: ic → 256, 3×3  │
          │                         │
          │    residualBlock × 19   │   ← "tower"
          │      256 → 256, s=1     │      (AlphaGo Zero's larger
          │                         │       net has 39; not specced here)
          └──────────┬──────────────┘
                     │
          ┌──────────┴──────────────┐
          ▼                         ▼
  Policy head                Value head (convBn→1)
   Go: convBn→2, flatten,     → flatten
       dense(2·H·W → nMoves)  → dense(H·W → 256, ReLU)
   chess: convBn 256→256,     → dense(256 → 1)
       conv 1×1 → 73 planes,  → tanh   (not in the Activation enum;
       flatten (the logits)             identity here, applied downstream)
```

Our `NetSpec` is a linear list of layers, so we represent the two heads
as two independent specs sharing the body conceptually. A reader who
wants to think "what does AlphaZero look like in Lean" sees **both**
branches laid out — the shared body in each, the heads diverging only at
the last few layers. In the codegen, sharing the body parameters is an
orchestration concern; the pure shape/architecture story is the two
specs below.

## Variants

- `alphaGoZeroPolicy` / `alphaGoZeroValue` — the 19-block / 256-channel /
  19×19 board / 17-plane Go network.
- `alphaZeroChessPolicy` / `alphaZeroChessValue` — AlphaZero's chess network:
  the same 19-block / 256-channel tower on the 8×8 board with 119 input planes,
  and a convolutional policy head whose 73 output planes are the 73 × 8 × 8 = 4672
  move logits.
- `tinyAlphaZeroPolicy` / `tinyAlphaZeroValue` — a scale-model with 3
  blocks for quick inspection / testing. Useful as a fixture.

All are pure `NetSpec` values; no training runs here. The book's Part 2
(Bestiary) uses these as read-only examples of architecture idioms. The
self-play loop itself is trained in `demos/MainAlphaZeroTtt.lean`, on the Lean
tic-tac-toe of `LeanMlir/TicTacToe.lean` and scored against the solved game —
with AlphaGo's plain conv stack rather than this conv-BN tower, whose batch
statistics over nine binary cells diverged (see that file's docstring).

## References

- Silver et al. 2017, *Mastering the game of Go without human knowledge* (AlphaGo Zero; PUCT, the 20- and 40-block nets). <https://doi.org/10.1038/nature24270>
- Silver et al. 2018, *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* (AlphaZero). <https://doi.org/10.1126/science.aar6404>
-/

-- ════════════════════════════════════════════════════════════════
-- § AlphaGo Zero (original, Go board 19×19, 17 input planes)
-- ════════════════════════════════════════════════════════════════

/-- Policy branch: outputs 19·19 + 1 = 362 move probabilities (361 board
    positions + 1 pass). -/
def alphaGoZeroPolicy : NetSpec where
  name := "AlphaGo Zero (policy head)"
  imageH := 19
  imageW := 19
  layers := [
    .convBn 17 256 3 1 .same,                        -- 17 planes → 256
    .residualBlock 256 256 19 1,                     -- 19 residual blocks
    .convBn 256 2 1 1 .same,                         -- policy head: conv→BN→ReLU (2 filters)
    .flatten,
    .dense (2 * 19 * 19) 362 .identity               -- 361 moves + pass
  ]

/-- Value branch: outputs a single scalar (expected outcome, pre-tanh). -/
def alphaGoZeroValue : NetSpec where
  name := "AlphaGo Zero (value head)"
  imageH := 19
  imageW := 19
  layers := [
    .convBn 17 256 3 1 .same,
    .residualBlock 256 256 19 1,
    .convBn 256 1 1 1 .same,                         -- value head: conv→BN→ReLU (1 filter)
    .flatten,
    .dense (1 * 19 * 19) 256 .relu,
    .dense 256 1 .identity                            -- tanh applied downstream
  ]

-- ════════════════════════════════════════════════════════════════
-- § AlphaZero chess (8×8 board, 119 input planes, 19 blocks)
-- ════════════════════════════════════════════════════════════════

/-- Policy: 73 × 8 × 8 = 4672 move logits (AlphaZero's move encoding). The head is
    convolutional, as in the paper: a rectified, batch-normalised 3×3 conv, then a
    1×1 conv to 73 planes whose flattened output is the logit vector. -/
def alphaZeroChessPolicy : NetSpec where
  name := "AlphaZero chess (policy head)"
  imageH := 8
  imageW := 8
  layers := [
    .convBn 119 256 3 1 .same,
    .residualBlock 256 256 19 1,
    .convBn 256 256 3 1 .same,                       -- policy head: conv→BN→ReLU
    .conv2d 256 73 1 .same .identity,                -- 73 move planes
    .flatten
  ]

def alphaZeroChessValue : NetSpec where
  name := "AlphaZero chess (value head)"
  imageH := 8
  imageW := 8
  layers := [
    .convBn 119 256 3 1 .same,
    .residualBlock 256 256 19 1,
    .convBn 256 1 1 1 .same,
    .flatten,
    .dense (1 * 8 * 8) 256 .relu,
    .dense 256 1 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny AlphaZero (fixture for testing / small-scale pedagogy)
-- ════════════════════════════════════════════════════════════════

/-- 3 residual blocks, 64 channels, 9×9 tiny-Go-style board. -/
def tinyAlphaZeroPolicy : NetSpec where
  name := "Tiny AlphaZero (policy head)"
  imageH := 9
  imageW := 9
  layers := [
    .convBn 17 64 3 1 .same,
    .residualBlock 64 64 3 1,
    .convBn 64 2 1 1 .same,
    .flatten,
    .dense (2 * 9 * 9) 82 .identity
  ]

def tinyAlphaZeroValue : NetSpec where
  name := "Tiny AlphaZero (value head)"
  imageH := 9
  imageW := 9
  layers := [
    .convBn 17 64 3 1 .same,
    .residualBlock 64 64 3 1,
    .convBn 64 1 1 1 .same,
    .flatten,
    .dense (1 * 9 * 9) 64 .relu,
    .dense 64 1 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary of every Bestiary entry in this file.
-- ════════════════════════════════════════════════════════════════


def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — AlphaZero"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Two-headed network: shared residual body + policy / value heads."
  IO.println "  Not trained here — just the architecture, as NetSpec values."

  alphaGoZeroPolicy.summarize (unit := .bare) (okNote := " (channel dims chain cleanly)")
  alphaGoZeroValue.summarize (unit := .bare) (okNote := " (channel dims chain cleanly)")
  alphaZeroChessPolicy.summarize (unit := .bare) (okNote := " (channel dims chain cleanly)")
  alphaZeroChessValue.summarize (unit := .bare) (okNote := " (channel dims chain cleanly)")
  tinyAlphaZeroPolicy.summarize (unit := .bare) (okNote := " (channel dims chain cleanly)")
  tinyAlphaZeroValue.summarize (unit := .bare) (okNote := " (channel dims chain cleanly)")

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • Policy and value heads share the first two layers (body),"
  IO.println "    which would share parameters in a real training run. NetSpec"
  IO.println "    as a linear list can't express that sharing; we show the"
  IO.println "    two forks as separate specs."
  IO.println "  • Value head ends with dense(→1, identity). The original paper"
  IO.println "    applies tanh afterwards; `Activation` has no tanh, so it is"
  IO.println "    applied downstream."
  IO.println "  • The chess policy head is convolutional: its 73 output planes"
  IO.println "    over the 8×8 board are the 4672 move logits, flattened, with"
  IO.println "    no dense layer."
