# Adjoint chain: depth-linear float composition (option-2 of the scaling question)

Status 2026-07-04: **v1 LANDED (uncommitted)** — `LeanMlir/Proofs/AdjointChainBridge.lean`
(3-axiom clean, registered in `tests/AuditAxioms.lean`) + `scripts/adjoint_chain_probe.py`
(the MEASURED-tier discharge artifact, validated on gfx1100).

## The problem this solves

`FloatClose.comp` composes per-op certificates generically — and `floatClose_iterate` /
`floatBridges_towerBack` already make whole-net folds depth-generic by induction. But the
composed modulus `Lg ∘ Lf` multiplies inherited error by each layer's **worst-case** gain
(dense: the fan-in row sum `m·w`), so the whole-net budget pays `∏ᵢ gainᵢ ≈ (m·w)^depth`.
Measured on real silicon (IREE/gfx1100, relu∘dense stacks, width 64), that bound is sound
but exponentially vacuous while true drift stays flat (~√depth):

| depth | true GPU max\|f32−f64\| | interval fold | loose by |
|---|---|---|---|
| 3  | 1.3e-06 | 2.0e+00  | 1.5×10⁶ |
| 6  | 1.6e-06 | 2.0e+05  | 1.3×10¹¹ |
| 12 | 2.1e-06 | 1.6e+15  | 7.7×10²⁰ |
| 24 | 4.4e-06 | 7.4e+34  | 1.7×10⁴⁰ |

The actual error pays the gain of the **composed tail** (‖J_tail‖ along the trajectory,
O(1) for He-init/normalized nets), not the product of per-layer worst cases. Empirics
(scratch prototypes, 2026-07-04 session): propagating each layer's measured local rounding
residual through the tail linearization predicts the true whole-net f32 error to rel-err
≤ 3e-5 at every depth incl. softmax and train-mode BN; the triangle-inequality shape
Σᵢ‖Jᵢrᵢ‖ is within 2–5× of truth.

## What v1 proves (`AdjointChainBridge.lean`)

The telescoping (hybrid) argument, ONCE, by induction on the layer list — **exact, no
linearization**:

- `LipOnWindow A H f` — windowed Lipschitz gain (elementwise, matching the FloatClose
  currency): inputs within `|·|≤A`, per-coordinate spread `e` ⟹ output spread `≤ H·e`.
- `LayerCert m A` — one layer's local contract: real `f`, float `fF`, both hold the
  window, float within FRESH budget `b` of real at every window point (= the per-op
  `*_close` modulus at `e=0`; `LayerCert.of_floatClose` converts any existing
  `FloatClose A B` instance with `B ≤ A`).
- `TailGains ls Hs` — position i's `Hᵢ` bounds the REAL suffix `fₙ∘…∘fᵢ₊₁`.
- **`chain_adjointClose`**: `|chainF ls x j − chainR ls x j| ≤ chainBudget = Σᵢ Hᵢ·bᵢ`.
  Depth-LINEAR in the fresh budgets: no gain products between budgets — each `bᵢ` is
  amplified once, by its own tail.
- **`tailGains_suffixProd`** (the PROVEN face): per-layer gains
  (`lipOnWindow_dense` = `m·w'`, `lipOnWindow_relu` = 1) give `Hᵢ = ∏_{j>i} gⱼ` — which
  reproduces the old interval fold exactly. So the theorem **subsumes** the `.comp`
  chain; any tighter tail gain strictly improves it with zero change to the chain proof.

Proof shape: hybrid i (float prefix, real tail) vs hybrid i−1 differ by one fresh budget
pushed through one real tail; `abs_sub_le` telescopes. ~230 lines, builds in ~1.5s.

## The measured discharge (`scripts/adjoint_chain_probe.py`)

`Hᵢ` = L∞→L∞ norm (max abs row sum) of the tail Jacobian on-trajectory, maxed over
samples — exactly what one backward/VJP sweep computes. Same `bᵢ` = `layerBudget(…, 0)`
at the measured window, same nets, same silicon:

| depth | true GPU err | chainBudget (measured H) | /true | (proven H) | /true |
|---|---|---|---|---|---|
| 3  | 1.3e-06 | 2.4e-02 | 1.8e+04 | 2.0e+00  | 1.5e+06 |
| 6  | 1.6e-06 | 9.8e-02 | 6.3e+04 | 2.0e+05  | 1.3e+11 |
| 12 | 2.1e-06 | 1.9e-01 | 8.8e+04 | 1.6e+15  | 7.7e+20 |
| 24 | 4.4e-06 | 8.2e-01 | 1.8e+05 | 7.4e+34  | 1.7e+40 |

Non-vacuous at every depth (0.8 vs activations O(10) at d=24); the residual ~10⁴–10⁵×
is per-layer Higham-vs-FMA conservatism (the known ~500×/layer from
`kernel_faithfulness_probe.py`) times window-sup-vs-typical magnitudes — flat in depth,
NOT compounding. Ratio grows only 10× from d=3→24 while the interval fold grows 10³⁴×.

**Honesty line (keep it crisp):** the measured `Hᵢ` is an on-trajectory probe of a
window-supremum hypothesis — MEASURED tier, quarantined exactly like `esig`/`egelu`:
supplied as a named hypothesis at the application site with provenance, never an axiom;
the Lean statement stays 3-axiom clean.

## Softmax / BN fit

The window formulation absorbs both problem children:
- softmax: **DONE in v1** — `lipOnWindow_softmax`, gain `(e^{4A}−1)/(2A)`: the nonlinear
  modulus `e^{2δ}−1` (`softmax_perturb`) is linear-on-a-window by convexity of `exp`
  through the origin (`convexOn_exp`), plus window points never differ by more than `2A`;
- train-mode BN's real-map gain on a window is bounded by the existing `bnReluBudget`
  input-shift machinery (`G·(2+8A²/ε)/√ε`-shaped) — `lipOnWindow_bn` from the same parts.
Neither is needed for the fold to work (they're just gain instances); v1 ships
dense/relu/softmax.

## v2 ladder (open, in order of value)

1. **Per-op gain instances**: `lipOnWindow_softmax`, `lipOnWindow_bn`, conv (row-sum =
   `ic·kH·kW·w'`), gelu (the magnitude-poly constant from `floatClose_gelu`) — each a
   short lemma from existing `*_perturb`/`*_close` parts.
2. **Trajectory-tube gains**: replace window-sup with a tube of radius = accumulated
   budget around the real trajectory (needs the pairwise prefix-gain bookkeeping —
   `Hⱼ→ᵢ` for j<i — threaded through the induction). Closes most of the measured-vs-
   proven gap *formally* for smooth ops; relu needs a margin condition at mask
   boundaries (the `MaxPool2MarginQ.isArgmax_iff` pattern).
3. **Heterogeneous windows/dims**: per-position `Aᵢ` (indexed lists or sigma-typed
   chains); v1 is uniform-width/uniform-window (the `towerBack` shape) with stems/heads
   composed at the ends via `FloatClose.comp`.
4. ~~**A committed-net instance**~~ ✅ DONE 2026-07-04 — `adjoint_chain_probe.py` §2 runs
   the committed MNIST-CNN render (kernel_faithfulness's per-stage emission, gfx1100),
   heterogeneous per-stage windows (the item-3 shape, evaluated numerically):

   | | true GPU logits drift | chainBudget measured-H | chainBudget proven-H (= interval fold) |
   |---|---|---|---|
   | MNIST-CNN | 1.6e-07 | 4.1e+00 (2.5e7×) | 2.1e+04 (1.3e11×) |

   Adjoint chain beats the interval fold by ~5×10³ even at only 6 stages — but the
   HONEST reading: on this shallow-wide net the budget (4.1) still exceeds the logits
   magnitude (0.38), and the slack is NOT the gains (H_meas ≈ 20–100, fine) — it's the
   fan-in-6272 Higham fresh budget `b_dense0 = 0.21` (the known 10³–10⁴× dot_close
   conservatism at large n, cf. kernel_faithfulness_probe). Composition is no longer
   the bottleneck; the per-op local budget is. So the leverage order flips per regime:
   deep-narrow nets → gains dominate (adjoint chain wins big); shallow-wide nets →
   local dot budgets dominate (a tighter proven dot bound — FMA-aware or probabilistic
   — would multiply straight through the chain).

## Session artifacts (scratchpad, 2026-07-04)

`vjp_error_prop.py` / `vjp_roundoff_predict.py` (linearization tightness: exact at
roundoff scale, degrades only ≥1e-3 perturbations, BN batch-coupling worst),
`iree_depth_sweep.py` / `depth_sweep_bounds.py` (the interval-fold depth cliff, incl.
softmax/BN variants), `validate_adjoint_chain.py` (= the probe, pre-scripts/ copy).
