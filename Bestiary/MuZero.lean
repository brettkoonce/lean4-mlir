import LeanMlir

/-! # MuZero — Bestiary entry

MuZero (Schrittwieser, Antonoglou, Hubert, et al., 2020 — "Mastering
Atari, Go, chess and shogi by planning with a learned model") is the
successor to AlphaZero. Same MCTS-guided reinforcement learning — but
**without being given the rules of the game**. MuZero learns a model
of the environment's dynamics and plans in that learned latent space.

## Three networks instead of one

Where AlphaZero has a single network with two heads (policy + value)
and uses the real game engine for MCTS rollouts, MuZero factors the
work across **three** networks:

```
   observation  o_t                                               hidden state
        │                                                              ▲
        ▼                                                              │
   representation  h(o_t) ─────────────────────────────────────────►  s_t
                                                                      │
                                         ┌────────────────────────────┤
                                         │                            │
                              prediction f(s_t)           dynamics g(s_t, a_t)
                                         │                            │
                              ┌──────────┴──────────┐       ┌─────────┴────────┐
                              ▼                     ▼       ▼                  ▼
                         policy π_t            value v_t   r_t           next state s_{t+1}
                              │                     │       │                  │
                              └── MCTS selection ───┘       └── reward target ─┘
```

- **Representation** `h` — encodes observation (pixels, board state)
  into a compact hidden state. Never sees the real environment after
  training; MCTS operates entirely in the learned latent space.
- **Dynamics** `g` — given `(hidden_state, action)`, predicts
  `(next_hidden_state, immediate_reward)`. The **learned simulator**.
  This is the MuZero-exclusive piece.
- **Prediction** `f` — given hidden state, predicts `(policy prior,
  value estimate)`. Identical role to AlphaZero's two heads.

MCTS uses `f` to pick promising actions and `g` to advance the tree,
never touching the real environment during planning.

## Why no new primitives?

Each of the three networks is a ResNet-style stack — AlphaZero's
body with minor variations. Dynamics and representation use
encoder/recurrent-conv structures; prediction mirrors AlphaZero's
two-headed tower. Nothing architecturally novel at the **layer**
level; the novelty is in (a) the three-network factoring and (b) the
training objective (value + reward + policy + encoder-dynamics
consistency losses). Both are outside NetSpec's shape concern.

## Variants

| Variant | Board / input   | Backbone | Blocks |
|---------|-----------------|----------|--------|
| Go MuZero   | 19×19×17 Go     | 256 ch   | 16 ResBlocks |
| Atari MuZero | 96×96×128 frames | 256 ch   | ~10 ResBlocks |
| tiny MuZero | 9×9×17 toy-Go   | 64 ch    | 3 ResBlocks |

Params are per-network; total MuZero-for-Go is roughly 3× AlphaZero
(since it has three networks of comparable scale). MuZero-for-Atari
is lighter thanks to global-average-pooling in the prediction head
and fewer residual blocks.
-/

-- ════════════════════════════════════════════════════════════════
-- § Go MuZero — three-network decomposition
-- ════════════════════════════════════════════════════════════════

/-- Representation network `h`: observation → hidden state.
    Input: 19×19 Go board encoded as 17 planes.
    Output: (256, 19, 19) latent. -/
def muZeroGoRepresentation : NetSpec where
  name := "MuZero Go — representation h"
  imageH := 19
  imageW := 19
  layers := [
    .convBn 17 256 3 1 .same,
    .residualBlock 256 256 16 1     -- 16 ResBlocks, matching AlphaZero body
  ]

/-- Dynamics network `g`: (hidden state, action) → next hidden state.
    Action is encoded as one additional channel plane, so input channels
    = hidden-state channels + 1. Output is the same shape as the hidden
    state (the reward scalar head branches off and is shown separately). -/
def muZeroGoDynamics : NetSpec where
  name := "MuZero Go — dynamics g (next-state path)"
  imageH := 19
  imageW := 19
  layers := [
    .convBn 257 256 3 1 .same,        -- 256 hidden + 1 action plane
    .residualBlock 256 256 16 1       -- same body depth as representation
    -- output: (256, 19, 19) next hidden state
  ]

/-- Reward head of the dynamics network: scalar per step. -/
def muZeroGoDynamicsReward : NetSpec where
  name := "MuZero Go — dynamics g (reward head)"
  imageH := 19
  imageW := 19
  layers := [
    .conv2d 256 1 1 .same .identity,  -- 1×1 conv to 1 channel
    .flatten,
    .dense (1 * 19 * 19) 128 .relu,
    .dense 128 1 .identity            -- scalar reward
  ]

/-- Prediction network `f`: hidden state → policy distribution.
    Same pattern as AlphaZero's policy head. -/
def muZeroGoPredictionPolicy : NetSpec where
  name := "MuZero Go — prediction f (policy head)"
  imageH := 19
  imageW := 19
  layers := [
    .residualBlock 256 256 2 1,       -- small head body (2 ResBlocks)
    .conv2d 256 2 1 .same .identity,
    .flatten,
    .dense (2 * 19 * 19) 362 .identity   -- 361 board positions + pass
  ]

/-- Prediction network `f`: hidden state → scalar value. -/
def muZeroGoPredictionValue : NetSpec where
  name := "MuZero Go — prediction f (value head)"
  imageH := 19
  imageW := 19
  layers := [
    .residualBlock 256 256 2 1,
    .conv2d 256 1 1 .same .identity,
    .flatten,
    .dense (1 * 19 * 19) 256 .relu,
    .dense 256 1 .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Atari MuZero — representation network (the interesting one)
-- ════════════════════════════════════════════════════════════════

/-- Atari representation network: stacks of residual blocks with
    spatial downsampling. Takes 32 stacked frames × 4 channels = 128
    input planes, produces (256, 6, 6) hidden state (after 4 stride-2
    reductions from the 96×96 input).

    Dynamics and prediction heads follow the same pattern as Go (omitted
    here for brevity — identical structure with different spatial dim). -/
def muZeroAtariRepresentation : NetSpec where
  name := "MuZero Atari — representation h"
  imageH := 96
  imageW := 96
  layers := [
    .convBn 128 128 3 2 .same,        -- 96 → 48
    .residualBlock 128 128 2 1,       -- 2 ResBlocks, 128 ch
    .convBn 128 256 3 2 .same,        -- 48 → 24
    .residualBlock 256 256 3 1,       -- 3 ResBlocks, 256 ch
    .maxPool 2 2,                      -- 24 → 12 (avg-pool in paper; maxPool is our proxy)
    .residualBlock 256 256 3 1,
    .maxPool 2 2,                      -- 12 → 6
    .residualBlock 256 256 3 1         -- output: (256, 6, 6)
  ]

-- ════════════════════════════════════════════════════════════════
-- § tinyMuZero fixture
-- ════════════════════════════════════════════════════════════════

def tinyMuZeroRepresentation : NetSpec where
  name := "tiny MuZero — representation"
  imageH := 9
  imageW := 9
  layers := [
    .convBn 17 64 3 1 .same,
    .residualBlock 64 64 3 1
  ]

def tinyMuZeroDynamics : NetSpec where
  name := "tiny MuZero — dynamics"
  imageH := 9
  imageW := 9
  layers := [
    .convBn 65 64 3 1 .same,           -- 64 hidden + 1 action
    .residualBlock 64 64 3 1
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

private def summarize (spec : NetSpec) : IO Unit := do
  IO.println s!""
  IO.println s!"  ── {spec.name} ──"
  IO.println s!"  input       : {spec.imageH} × {spec.imageW}"
  IO.println s!"  layers      : {spec.layers.length}"
  IO.println s!"  params      : {spec.totalParams} (~{spec.totalParams / 1000000}M)"
  IO.println s!"  architecture:"
  IO.println s!"    {spec.archStr}"
  match spec.validate with
  | none     => IO.println s!"  validate    : OK"
  | some err => IO.println s!"  validate    : FAIL — {err}"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — MuZero"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  AlphaZero, but without the rulebook. Three networks:"
  IO.println "  representation h, dynamics g, prediction f. MCTS runs"
  IO.println "  entirely in the learned latent space."

  IO.println ""
  IO.println "──────────── Go MuZero (three networks) ────────────"
  summarize muZeroGoRepresentation
  summarize muZeroGoDynamics
  summarize muZeroGoDynamicsReward
  summarize muZeroGoPredictionPolicy
  summarize muZeroGoPredictionValue

  IO.println ""
  IO.println "──────────── Atari MuZero (representation only) ────────────"
  summarize muZeroAtariRepresentation

  IO.println ""
  IO.println "──────────── tiny MuZero (fixture) ────────────"
  summarize tinyMuZeroRepresentation
  summarize tinyMuZeroDynamics

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • No new Layer primitives. MuZero's novelty is the DYNAMICS"
  IO.println "    network — a learned environment simulator — and the"
  IO.println "    training objective (value + reward + policy + consistency"
  IO.println "    losses). The per-network architectures are ResNet-style."
  IO.println "  • Action encoding: actions enter the dynamics network as an"
  IO.println "    extra channel plane (one-hot or similar). Hence the dynamics"
  IO.println "    input has 257 channels (256 hidden + 1 action) for Go."
  IO.println "  • Prediction head split: policy + value shown as two specs,"
  IO.println "    same as AlphaZero's treatment. The body (2 ResBlocks) is"
  IO.println "    shared in practice; our linear NetSpec can't express that."
  IO.println "  • Avg-pool in the Atari encoder is approximated with maxPool."
  IO.println "    Identical spatial effect; slight difference in activation"
  IO.println "    statistics at training time."
  IO.println "  • MuZero is the spiritual bridge between game-AI (AlphaZero)"
  IO.println "    and model-based RL (Dreamer, PlaNet). Same rep + dyn + pred"
  IO.println "    factoring — different backbones, different training losses."
