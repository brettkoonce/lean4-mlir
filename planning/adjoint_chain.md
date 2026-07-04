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

   **CIFAR-8 (no BN, the committed `cifar8Verified` net — probe §3, 2026-07-04): the
   depth-dominated regime, and the headline.** 15 stages (8 convs [16,16,32,32] + 4
   pools + 3-dense head), small fan-ins (≤288) ⇒ tame local budgets, composition is
   the whole game:

   | | true GPU logits drift | logits magnitude | chainBudget measured-H | chainBudget proven-H (= interval fold) |
   |---|---|---|---|---|
   | CIFAR-8 | 3.8e-06 | 4.6 | **2.6e+00 (6.8e5×) — BELOW the logit scale** | 4.1e+14 (1.1e20×) |

   The adjoint chain gives the first NON-VACUOUS whole-net float certificate on the
   deepest committed no-BN net (budget < logits magnitude; argmax-safe wherever the
   margin exceeds 2·budget ≈ 5), while the interval fold overshoots the logit scale by
   ~14 orders. Per-stage contributions H_i·b_i are flat (0.03–0.3 each — genuinely
   depth-linear); the interval fold's suffix products reach 2e18 at conv1 vs measured
   tail gain 210. (Standalone §3 run: budget 1.76 vs logits 5.0 — numbers vary with
   the shared RNG stream position; conclusion identical.)

   **CIFAR-8-BN (`cifar8BnVerified`, per-EXAMPLE per-channel BN — probe §4,
   2026-07-04): BN is where the remaining work lives.** 23 stages; BN fresh budgets
   from the BnFloatBridge parts (bn_mean/var/istd/norm budgets at the A-POSTERIORI
   floor σ²min+ε; ers = 2u32 pinned); per-example stats ⇒ no batch coupling, the
   per-sample tail Jacobians stay valid:

   | | true GPU logits drift | logits magnitude | chainBudget measured-H | chainBudget proven-H (a-post floor) |
   |---|---|---|---|---|
   | CIFAR-8-BN | 6.6e-06 | 3.2 | 5.1e+01 (7.7e6×) — 16× ABOVE logit scale | 2.8e+31 (4.2e36×) |

   The adjoint chain is ~10³⁰× tighter than the interval fold (which is vacuous by 31
   orders even at the charitable a-posteriori floor — the strict ε-floor Lean face adds
   another ~(σ²/ε)^1.5 ≈ 10⁷ PER BN, ~10⁶⁰ total), but the budget is not yet below the
   logit scale. Two identified causes, both actionable:
   1. **bn1/bn2 dominate (33 + 11.6 of the 50.7 total)**: the probe takes D, S, and the
      istd floor as maxes/mins over ALL channels — one low-variance channel at init
      poisons the whole stage budget. Per-channel budget bookkeeping (which the Lean
      `bnPerChannel` machinery already supports structurally) would shrink b_bn by the
      worst-to-typical channel ratio.
   2. **BN genuinely amplifies tail gains** (H_meas ≈ 500–1300 early, vs 210/87 in the
      no-BN twin): normalization divides by per-channel σ, so perturbation directions
      that shrink variance get magnified. This part is real, not slack — BN helps
      optimization but hurts float-error certificates.

   **ResNet-34 @224² (`resnet34Verified` twin — probe §5, 2026-07-04): the composition
   mechanism scales; the ladder's summary row.** Eval-mode BN (running-stats per-channel
   affine, batch-calibrated — the DEPLOYED forward, `BnEvalFloatBridge`'s case), chain
   granularity = the `r34_floatBridges` granularity (stem / maxpool / 16 residual blocks
   / GAP / head = 20 stages; block fresh budget = the mini fold inside the block). Two
   methodological upgrades that live in §5 and should back-propagate to §2–§4:
   (a) budgets use the EXACT data-dependent `dot_close`/`denseErr` coefficients
   (`Σᵢ|Wᵢⱼ||xᵢ|` via abs-conv at the measured profile — what the Lean lemmas actually
   prove; `layerBudget`'s `m·w·A` is just their uniform upper bound and costs 257× at
   this scale); (b) gains use the exact row sum `Σ|W|` (what `m·w'` upper-bounds).
   Tail gains measured in f32 on the GPU (jax rocm) — ample for O(10³) numbers.

   | | true GPU logits drift | logits magnitude | chainBudget measured-H | chainBudget proven-H |
   |---|---|---|---|---|
   | r34 @224² | 1.1e-05 | 3.6 | 7.0e+02 (6.5e7×) | 6.5e+51 (6.1e56×) |

   Per-block H_i·b_i is FLAT (27–77 each across all 16 blocks — pristine depth-
   linearity; the total is just 16 × ~44). The interval fold is vacuous by 51 orders
   (was 10⁷⁸ before the exact-coefficient upgrade). The remaining ~200× to a logit-scale
   certificate: (i) the Higham fresh budget per block (~u·m·Σ|x||w| at m=4608 — the
   irreducible worst-case face; actual local drift is ~√m smaller, so the last order or
   two needs probabilistic/FMA-aware local bounds); (ii) early-block tail gains ~10³ at
   He-init + calibrated BN (stem perturbations amplify ~2200× through the residual
   stream — real sensitivity, not slack; re-measuring on the TRAINED checkpoint
   (.lake/build/resnet34_adam_ckpt.bin, 272MB, layout = ResNet34Layout.specs) is the
   natural follow-up).

   **Ladder summary (all on gfx1100, logits-scale certificates):**

   | net | stages | interval fold | adjoint chain | logit scale | verdict |
   |---|---|---|---|---|---|
   | MLP d=24 | 24 | 7e+34 | 0.82 | ~21 | non-vacuous ✓ |
   | MNIST-CNN | 6 | 2e+04 | 4.1 | 0.38 | fan-in-budget-bound |
   | CIFAR-8 | 15 | 4e+14 | 2.6 | 4.6 | **non-vacuous ✓** |
   | CIFAR-8-BN | 23 | 3e+31 | 51 | 3.2 | 16× over |
   | r34 @224² | 20 | 7e+51 | 698 | 3.6 | 200× over |

   Composition is solved at every scale tried — the adjoint chain stays within 1–3
   orders of the logit scale where the interval fold loses 4–51 orders; what remains is
   local (per-op budgets, BN channel bookkeeping, trained-vs-init gains).

## Session artifacts (scratchpad, 2026-07-04)

`vjp_error_prop.py` / `vjp_roundoff_predict.py` (linearization tightness: exact at
roundoff scale, degrades only ≥1e-3 perturbations, BN batch-coupling worst),
`iree_depth_sweep.py` / `depth_sweep_bounds.py` (the interval-fold depth cliff, incl.
softmax/BN variants), `validate_adjoint_chain.py` (= the probe, pre-scripts/ copy).
