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

   **EfficientNet-B0 (probe §6, 2026-07-04): the COMMITTED AUDITED RENDER itself**
   (`verified_mlir/efficientnet_fwd_eval.mlir` run as-is on gfx1100, batch 32 @224²,
   args generated from its parsed signature — He convs, γ=1/β=0, batch-calibrated
   running stats; f64 oracle mirrors it op-for-op; esig = 2u32 for the 65 logistics):

   | | true GPU logits drift | logits magnitude | chainBudget measured-H | chainBudget proven-H |
   |---|---|---|---|---|
   | ENet-B0 (committed render) | 5.7e-06 | 1.3 | 6.8e+04 (1.2e10×) | **7.8e+106** (1.4e112×) |

   The interval fold sets the vacuity record (~10¹⁰⁶ — MBConv's per-block gain product
   is ~10⁴–10⁶: two swishes, three convs, and the SE path all multiply). The adjoint
   chain compresses that by **102 orders of magnitude** but lands ~5e4× above the logit
   scale — and the table pinpoints why: the late-block fresh budgets b_i ≈ 10³ are
   inflated by the WITHIN-block worst-case treatment of squeeze-excite. SE's gate path
   (pool → dense(1152→48) → swish → dense(48→1152) → σ) enters the block mini-fold as
   an interval product of dense row-sums (rs1 ≈ 40, rs2 ≈ 8 at He-init ⇒ ~×100 on the
   inherited error per block), while its measured sensitivity is tiny (σ saturated,
   0.25 max slope, pooled input). Cross-block composition is fine (H_meas ≤ 425,
   decaying to 6). **Identified v2 item: block-INTERNAL gain treatment — a residual/
   gate combinator with measured (or per-path proven) sub-gains for parallel-path
   blocks (SE, skip), instead of interval-folding inside the block.**

   **ConvNeXt-T (probe §7, 2026-07-04): the committed render
   (`verified_mlir/convnext_fwd.mlir`) as-is on gfx1100, repo init (layerScale γ = ONES
   per the kind-1 convention, not the paper's 1e-6):**

   | | true GPU logits drift | logits magnitude | chainBudget measured-H | chainBudget proven-H |
   |---|---|---|---|---|
   | ConvNeXt-T (committed render) | 2.9e-06 | 2.9 | 9.4e+07 (3.2e13×) | **3.2e+139** (1.1e145×) |

   All-time interval-fold record (~10¹³⁹). Cross-block composition is again clean
   (H_meas ≤ 103 decaying to 18), but per-block fresh budgets hit ~10⁵ — the worst of
   the ladder — from two identified WITHIN-block worst-case faces multiplying:
   1. **The committed scalar-LN normalizes over the whole C·H·W extent (n = 301056 at
      stage 0)** — its Higham mean/var face pays (1+u)^{n+1}−1 ≈ n·u ≈ 1.8e-2 where the
      true reduction error is ~√n·u (≈500× smaller), and the D² variance-shift term at
      D ≈ 20 inflates eistd. (Standard per-position channels-LN would have n = 96–768
      and be trivial — the scalar-LN render is architecturally the expensive-to-certify
      choice.)
   2. **`floatClose_gelu`'s magnitude-polynomial gain** `1 + √(2/π)/2·A·(1+3·0.044715·A²)`
      evaluates to ~400 at the actual expand magnitudes (A ≈ 20), while the TRUE global
      GELU Lipschitz constant is ≈ 1.13 (tanh saturation).
      **→ CLOSED same day: `LeanMlir/Proofs/GeluLipschitz.lean` (3-axiom clean, in
      AuditAxioms) proves `|gelu′| ≤ 3/2` globally** — `sech²u ≤ 4e^{−2|u|}` beats the
      cubic growth; the whole core is `(cs−½)² ≥ 0` + `π ≤ 4` + the cubic exp Taylor
      bound. Products: `geluScalarDeriv_abs_le`, `geluScalar_lipschitz` (MVT),
      `lipOnWindow_gelu` (adjoint-chain gain, window-free), `floatClose_gelu_sat`
      (drop-in `floatClose_gelu` with flat modulus `egelu + 3/2·e`). Probe §7 rerun
      with the proven flat gain: **adjoint chainBudget 9.4e7 → 3.2e6 (29× from one
      lemma); interval fold 3.2e139 → 2.8e113 (10²⁶×)**. The dominant residual is now
      purely the scalar-LN big-n Higham face (block budgets ~4e3–1e4, all LN).

   **ViT-Tiny (probe §8, 2026-07-04): the committed render (`verified_mlir/vit_fwd.mlir`)
   as-is on gfx1100, repo init (He denses, LN γ=1/β=0, biases/CLS/pos = ZEROS) — and the
   prediction "friendliest deep net" was WRONG in an instructive way:**

   | | true GPU logits drift | logits magnitude | chainBudget measured-H | chainBudget proven-H |
   |---|---|---|---|---|
   | ViT-Tiny (committed render) | 5.2e-06 | 4.3 | 1.6e+16 (3.0e21×) | **~1e+364** (~1e370×) |

   Three findings, in order of discovery:
   1. **The zero-init CLS token makes block-0's LN genuinely near-singular** (var = 0 ⇒
      istd = 1/√ε ≈ 316): a REAL property of the committed init (kind-2 zeros for
      CLS/pos), the extreme case of the BN min-channel issue. Fixed in the probe by
      **per-token LN budget bookkeeping** (each token's deviation paired with its own
      variance floor — the `perRowFlat` granularity the Lean LN lemmas already have);
      the zero token then contributes a large gain but ~zero fresh budget.
   2. **Cross-block composition is the cleanest of the whole ladder**: measured tail
      gains 84 → 2.8, decaying smoothly through all 12 blocks.
   3. **The within-block fold dies at the softmax exponent** — the attention analogue
      of the SE finding, but sharper: scores at init sit at A_s ≈ 9–12, the score-path
      linear coefficient (≈ 8·(A_q+A_k)·rowsum_W ≈ 100–200) amplifies the LN fresh
      budget (~4e-3, itself the n=192 Higham face ~10³× above true) to E_s ≈ 0.4–6,
      and the proven softmax modulus `e^{2δ}−1` is vacuous for δ ≳ 1 ⇒ block fresh
      budgets ~1e11–1e14. The interval fold hits ~1e364 (softmax window gain
      `(e^{4A_s}−1)/2A_s ≈ 1e17` PER BLOCK — reported in log-space now).
      **The fix is the identified v2 combinator, now with a precise spec**: sub-block
      chain granularity with the stage state carrying the residual stream (stage k
      output = (h, scores) pair), so the softmax burden moves from the nonlinear
      modulus to the MEASURED tail gain — where it is tiny (softmax Jacobian L∞ ≤ ½).
      Same lemma shape needed for SE; attention and SE are the same obstruction.

   **Ladder summary (all on gfx1100, logits-scale certificates):**

   | net | stages | interval fold | adjoint chain | logit scale | verdict |
   |---|---|---|---|---|---|
   | MLP d=24 | 24 | 7e+34 | 0.82 | ~21 | non-vacuous ✓ |
   | MNIST-CNN | 6 | 2e+04 | 4.1 | 0.38 | fan-in-budget-bound |
   | CIFAR-8 | 15 | 4e+14 | 2.6 | 4.6 | **non-vacuous ✓** |
   | CIFAR-8-BN | 23 | 3e+31 | 51 | 3.2 | 16× over |
   | r34 @224² | 20 | 7e+51 | 698 | 3.6 | 200× over |
   | ENet-B0 @224² (committed render) | 20 | 8e+106 | 6.8e+04 | 1.3 | 5e4× over (SE-in-block) |
   | ConvNeXt-T @224² (committed render) | 25 | 3e+139 | 9.4e+07 | 2.9 | 3e7× over (scalar-LN n=301k + gelu poly-gain) |
   | ConvNeXt-T + `gelu_lipschitz` (3/2 gain) | 25 | 2.8e+113 | 3.2e+06 | 2.9 | 1e6× over (scalar-LN only) |
   | ViT-Tiny @224² (committed render) | 15 | ~1e+364 | 1.6e+16 | 4.3 | softmax-exponent-in-block (needs the residual-carrying sub-block combinator) |

   Composition is solved at every scale tried — the adjoint chain stays within 1–3
   orders of the logit scale where the interval fold loses 4–51 orders; what remains is
   local (per-op budgets, BN channel bookkeeping, trained-vs-init gains).

## Session artifacts (scratchpad, 2026-07-04)

`vjp_error_prop.py` / `vjp_roundoff_predict.py` (linearization tightness: exact at
roundoff scale, degrades only ≥1e-3 perturbations, BN batch-coupling worst),
`iree_depth_sweep.py` / `depth_sweep_bounds.py` (the interval-fold depth cliff, incl.
softmax/BN variants), `validate_adjoint_chain.py` (= the probe, pre-scripts/ copy).
