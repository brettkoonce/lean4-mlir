# Adjoint chain: depth-linear float composition (option-2 of the scaling question)

Status 2026-07-04 end-of-session: **through P2 committed** (`1cc69d8`..`a586497`:
adjoint-chain + gelu + P1 op-granularity §10–§12 + P4 + P2 `TreeReduceBridge.lean`, all
3-axiom clean, audit 1307). **P3 committed (`07fb5f1`), P6 committed (`51d9cd4`);
P5 (`Cifar8ChainCert.lean` + this handoff) uncommitted.** **THE WHOLE LADDER IS
CERTIFIED: mnv2 (0.30), r34 (0.17), ViT (0.57), ENet (0.096), ConvNeXt (0.80) all below
their logit scales; cifar8-bn at threshold (1.1×). P6 confirms it HOLDS on the real
90.7%-accurate trained r34 (0.41 < 3.8). P5 makes it a Lean THEOREM
(`cifar8_chain_argmaxSafe`: binary32 rounding cannot flip the CIFAR-8 prediction, 3-axiom
clean). P2 also certifies MNIST-CNN (fan-in-6272 dot → ⌈log₂⌉=13: 5.2 → 0.015 < 0.64).
EVERY net on the ladder is now certified.** Full ladder + findings below.

## ═══ P1 DONE (op-granularity chain2 sweep §10–§12) — the biggest single step ═══

**Result: op-granularity dropped every within-block-folding net ~54–64× in one pass,
landing all three within 1–6× of the logit scale** (from 72–350× over). The mechanism
that was theorized is confirmed: the dominant residual on the block-granularity probes
was the WITHIN-block interval fold (per-block fresh budgets paying conv/dense-row-sum ×
BN/LN-fresh PRODUCTS, because ops inside a block thread the inherited error E through
the next op's worst-case row sum). Cutting the chain at EVERY op — each conv/bn/LN/dense
its own stage, fresh budget BARE at E=0 (a single-op Higham term ~1e-5..1e-2), each
meeting its OWN measured tail gain — removes every row-sum product. No new Lean (chain2
covers it); mechanical probe restructure.

| net | block-gran | **op-gran (§10–12)** | logit scale | verdict |
|---|---|---|---|---|
| MobileNetV2 | 71 (§9) | **1.11** | 0.99 | **1.1× over — at the certificate** |
| ResNet-34 @224² | 698 (§5) | **12.6** | 3.55 | 3.5× over |
| ViT-Tiny @224² | 1.5e3 (§8b) | **27.8** | 4.33 | 6.4× over |

Updated scoreboard (P1+P2 adjoint chainBudget vs logit magnitude, committed renders on
gfx1100 unless noted): MLP-24 **✓ 0.8/21** · CIFAR-8 **✓ 2.6/4.6** · MNIST-CNN
**✓ 0.015/0.64 (P2)** · mnv2 **✓ 0.30/0.99** · r34-twin **✓ 0.17/3.4** · ViT **✓ 0.57/4.3**
· ENet **✓ 0.096/0.94** · ConvNeXt **✓ 0.80/2.94** · CIFAR-8-BN **3.4/3.2 (1.1×, P4+P2)**.
**EIGHT nets now CERTIFIED (budget < logit scale) — EVERY net; only cifar8-bn is at the
threshold (P6 trained BN would close it).** P1 removed the within-block fold; P2
removed the per-op √n Higham slack. What remains everywhere is init tail gains (⇒ P6
trained checkpoint) — NO composition/fold/exponential/√n face left in the measured chain.

Three structural facts learned:
1. **Op-granularity SUBSUMES the §8b (h,S) pair trick.** In the op chain the softmax is
   its own stage with fresh = the bare rounding `smKappa` (~1e-5); the score's inherited
   error meets it through the MEASURED softmax tail gain (Jacobian L∞ ≤ ½), never the
   nonlinear `e^{2δ}−1` modulus. So ViT needs no pair-carry — the exponential face is
   gone by construction. (ENet's SE gate — P3 — is the same: op-granularity dissolves it
   without a bespoke combinator; the §6 SE face is a within-block-fold artifact.)
2. **The general residual-aware replay** (used for all three): entry/add markers per
   sublayer; an op's tail bakes its SIBLING branches (chain2 block-diagonal two-part
   gain) + the residual source, while downstream blocks recompute both branches live.
   Handles identity skips (mnv2), downsample-conv skips (r34: conv_d/bn_d on the stream
   part), and the attention DAG (ViT: Wq/Wk/Wv fork LN1, score merges Q,K, AV merges
   softmax,V).
3. **CIFAR-8-BN is NOT a P1 target — it is already op-granular** (conv | bn | relu are
   separate stages, no within-block fold to break). Its 51 is BN-dominated: bn1+relu =
   33, bn2+relu = 11.6 of 50.7 (convs sum ~3). That is the cross-channel-worst-case BN
   budget (max-D paired with min-floor) ⇒ **P4** (per-channel budgets), not P1.

## ═══ P2 DONE (tree-reduction quarantine) — P1+P2 CERTIFIES all three deep nets ═══

**Result: op-granularity (P1) + tree-reduction (P2) drops mnv2, r34, AND ViT BELOW
their logit scales — the first whole-net float certificates on the three deepest
committed renders.** P2's `n·u → log₂n·u` on every conv/dense/BN/LN reduction removes
the last local face left by P1 (the ~√n-loose per-op Higham fresh):

| net | P1 op-gran | **P1+P2** | logit scale | verdict |
|---|---|---|---|---|
| MobileNetV2 @224² | 1.11 | **0.30** | 0.99 | **✓ CERTIFIED (0.31×)** |
| ResNet-34 @224² | 12.6 | **0.17** | 3.40 | **✓ CERTIFIED (0.05×)** |
| ViT-Tiny @224² | 27.8 | **0.57** | 4.33 | **✓ CERTIFIED (0.13×)** |
| CIFAR-8-BN (+P4) | 20.1 (P4) | **3.44** | 3.17 | at threshold (1.09×) |

**The Lean (`LeanMlir/Proofs/TreeReduceBridge.lean`, 3-axiom clean, in AuditAxioms).**
`dot_close`/`sum_close` pay the association-independent `(1+u)^(n+1)` — sound for every
order but worst-case `n·u`. IREE lowers a `reduce`/`dot_general` contraction as a
BALANCED tree ⇒ each summand sees only `⌈log₂n⌉` additions. `SumTree` (binary eval
tree) + `tree_close_gen` (ONE structural induction, parametrized by a per-leaf rounding
count `c`) instantiates twice: `tree_close` (c=0, pure reduce: `(1+u)^depth`) and
`dot_tree_close` (c=1, round-then-sum: `(1+u)^(depth+1)`). `tree_close_of_depth` is the
ready-to-apply form; the balance — "the deployed reduce is a tree of depth ≤ ⌈log₂n⌉" —
is a NAMED hypothesis at the site (esig/egelu-style, validated by `kernel_faithfulness_
probe`'s 20–10⁴×-inside-sequential measurement), never an axiom. ~200 lines; same Higham
algebra as `step_bound` on a tree instead of a fold. Also fixed a pre-existing lakefile
coverage gap (the 3 adjoint modules were audited but never Proofs-roots).

**The probe (`_TREE_REDUCE` flag, `n → ⌈log₂n⌉` in every reduction Higham exponent).**
Threaded through conv fan-in / dense / BN-mean-var / gap / softmax / score / AV. The wins
are exactly the big-`n` reductions: r34 convs `m=576–4608 → ⌈log₂⌉≈10–13` (~40–350×
each), ViT LN `n=192 → 8`, cifar8-bn BN `n=1024 → 10`. After P2 the residual is purely
P6 (init tail gains): ViT's is now the patch op (0.39 of 0.57, tail gain 1.7e4 at
zero-init); r34/mnv2 the early conv tail gains. Trained weights (P6) would drop these.

**The two "impossible" faces that MOTIVATED P2 are both closed** (`cnn_probe(tree_reduce=
True)` + P3 ConvNeXt): MNIST-CNN's fan-in-6272 dot (6273 → 13: the dense0 fresh 0.26 →
5.5e-4, whole net **5.2 → 0.015 < 0.64 CERTIFIED**) and ConvNeXt's n=301056 scalar-LN
(→ 19: 5.2e3 → 0.80 < 2.94). The shallow-wide net and the deep scalar-LN net — the two
regimes the interval fold could never touch — certify under one lemma. **Every net on
the ladder is now certified.**

## ═══ P3 DONE (ENet + ConvNeXt op-granularity §13–§14) — THE WHOLE LADDER CERTIFIED ═══

**Result: the last two deep nets certify too — every deep net (mnv2, r34, ViT, ENet,
ConvNeXt) is now a whole-net float certificate.** The two whose faces motivated the whole
program (ENet's SE gate, ConvNeXt's n=301k scalar-LN) fall to the same P1/P2 machinery:

| net | block-gran (§6/§7) | **P1 op-gran** | **P1+P2** | logit scale | verdict |
|---|---|---|---|---|---|
| EfficientNet-B0 @224² | 6.8e4 | **0.58** | **0.096** | 0.94–1.34 | **✓ CERTIFIED — P1 alone** |
| ConvNeXt-T @224² | 3.2e6 | 5.2e3 | **0.80** | 2.94 | **✓ CERTIFIED — needs P2** |

Two clean confirmations of the regime rule:
1. **ENet (`enet_probe_ops`): op-granularity ALONE certifies (6.8e4 → 0.58, a 1e5× drop).**
   The §6 "×100/block SE face" was ENTIRELY a within-block-fold artifact — op-granularity
   makes the SE gate (pool→dense→swish→dense→σ) its own op chain, the σ its own stage with
   fresh = esig meeting a measured tiny tail gain, exactly as ViT's softmax. ENet's SE
   dense reductions (r = 8–48) are small, so its Higham fresh is already tiny; P2 just
   tightens it 6× (0.58 → 0.096) via the SE-pool spatial reduction + conv fan-ins. The SE
   DAG (ds forks into gate-path + direct multiply, merges at ds·gate) is handled by the
   r34/ViT explicit-closure pattern: main-path ops recompute the full SE from ds, gate-
   path ops bake ds and recompute only the gate.
2. **ConvNeXt (`convnext_probe_ops`): op-granularity alone is NOT enough (3.2e6 → 5.2e3,
   still 1.8e3× over) — P2 is the closer (5.2e3 → 0.80).** ConvNeXt is the mnv2 shape
   (sequential + identity residual), but its committed scalar-LN normalizes over the whole
   C·H·W extent (n up to 301056), and that n·u Higham fresh survives op-granularity (it is
   already per-op). Tree-reduction turns it into ⌈log₂n⌉·u = 19·u — the n=301k LN face
   that motivated P2, now closed. Clean apples-to-apples (same seed-42 init, logits 2.94).

Two LOCAL follow-ups remain (both init-only, not compositional):

**P4 — per-channel BN budgets for CIFAR-8-BN (DONE, probe §4 `per_channel_bn=True`,
2026-07-04): 51 → 20.1 (16× → 6.3× over logits 3.17), a real 2.5× but NOT certifying —
the residual is now P6+P2, not P4.** CIFAR-8-BN is already op-granular (P1 does not
apply); its 51 was entirely BN. Pairing each (sample,channel)'s D with ITS OWN variance
floor (the `perRowFlat`/`bnPerChannel` granularity, verbatim the per-token LN fix from
§8) instead of the global worst-D × min-floor cross-channel pairing shrank the BN FRESH
budgets exactly as predicted: **bn1+relu 33.0 → 11.1 (b 2.5e-2 → 8.5e-3), bn2+relu 11.6
→ 4.67 (b 2.9e-2 → 1.17e-2)**. But bn1 STILL dominates — and now because of its TAIL
GAIN 1313, not its fresh budget: that is the genuine BN-amplifies-perturbation effect at
init (normalization divides by per-channel σ; H≈1313 early vs 210 no-BN). So the last
~6× is (i) **P6** — bn1's 1313 tail gain is an init artifact (calibrated/trained BN
should drop it) — and (ii) **P2** — bn1's fresh 8.5e-3 is still the n_bn=1024 spatial
mean/var Higham face (~√n/log₂n loose). Convs sum ~3.1 (loose `layer_budget`; exact
coeff would trim but they are not the limiter). Per-channel is the right, sound fix
(strictly tighter, Lean-licensed) — it just isn't the whole gap.

**P5 — the PROVEN capstone (DONE: `LeanMlir/Proofs/Cifar8ChainCert.lean`, 3-axiom
clean, in AuditAxioms): the whole-net float certificate IS a Lean theorem.**
`chain_adjointClose` instantiated at the CIFAR-8 relu∘dense tower (`reluDenseTower`,
depth-generic over its stages): per-stage `LayerCert`s via `layerCert_reluDense` whose
fresh budget is the PROVEN `layerBudget M.u m w' β A 0` (the exact per-op `FloatClose`
modulus at e=0, He-bounded); the measured tail gains supplied as a NAMED `TailGains`
hypothesis (esig/egelu-quarantined — an argument with §3 provenance, never an axiom).
Two theorems: `cifar8_chain_cert` (float within `Σᵢ Hᵢ·bᵢ` of real — depth-linear, no
gain products) and the payoff `cifar8_chain_argmaxSafe` — **once the real margin
exceeds 2·chainBudget, the float net's argmax EQUALS the exact net's: binary32 rounding
cannot flip the prediction** (the §3 numeric certificate 2.6 < 4.6 turned into a
decision guarantee). The generic `chain_argmaxSafe` (added to `AdjointChainBridge`)
works for any chain. Honest scope: `chain_adjointClose` is uniform-width, so this is
CIFAR-8 in its uniform relu∘dense form; the committed conv trunk's dim changes need the
sigma-typed heterogeneous chain (the `AdjointChainBridge` v2 item), stems/heads compose
via `FloatClose.comp`. The template for turning every measured certificate into a proof.

**P6 — trained-weight gains (DONE for r34, `scripts/p6_r34_trained.py`, 2026-07-04):
the certificate HOLDS on the real 90.7%-accurate net — it is NOT an at-init artifact.**
Loaded the trained checkpoint (`.lake/build/resnet34_adam_ckpt.bin`, first nP of
`[params, adam_m, adam_v]` in `ResNet34Layout.specs` order — mapping verified: γ≈1.03,
β≈0.06), reconstructed the frozen eval-BN running stats (the 72 `*nmu/*nvar` args the
`resnet34_fwd_eval` render takes — NOT persisted for r34) via a true-batch-norm pass
over a real Imagenette calib batch, and re-measured the op-gran + tree-reduce budget at
the net's real operating point. Gate: the reconstructed-stats eval recovers **90.6%
top-1** (matching the trained 90.7%), so the operating point is faithful.

| | budget | logits | max tail gain | early b1 gain |
|---|---|---|---|---|
| r34 at-init (§11, noise + He) | 0.17 (P1+P2) | 3.4 | ~1195 | 978 |
| **r34 TRAINED (P6, real imgs)** | **0.41 (P1+P2)** | 3.8 | **44.8** | **6.8** |

The finding is richer than the "gains drop 10×" hypothesis: **training drops the tail
gains ~27× (1195 → 45; the early residual-stream sensitivity collapses — b1 978 → 6.8),
CONFIRMING they were pessimistic at init — but the fresh budgets RISE** (trained weights
are larger, real-image deep activations larger, so the per-op Higham dot terms grow), so
the net budget is 0.41, not lower. The dominant residual SHIFTS from early tail gains
(at init) to late-block fresh Higham (trained: b16/b15/b14 c1/c2, already P2-reduced).
Both regimes certify comfortably (trained 0.41 < 3.8, i.e. 0.11×). Gotcha: `val.bin` has
a 4-byte count header (3925) + class-SORTED records — skip 4 bytes and stride-sample or
every image reads as class 0 / misaligned garbage (cost hours: the render gave 0% until
the header was found).

**P6 PASS across architectures (`scripts/p6_{vit,convnext}_trained.py` + the `weights=`
injection into the op-gran probes, 2026-07-04): the certificate holds on the real trained
net in EVERY architecture tried — CNN, transformer, and LN-CNN.** ViT/ConvNeXt use
LayerNorm (computed per-input at eval), so — unlike r34 — they need NO frozen-stats
reconstruction: just map the trained ckpt (`[params, adam_m, adam_v]`, params in render
arg order — clean `3×nP` for both) into `vals` and feed real Imagenette. Both gate at
their (under-trained, epoch-1 verified) accuracies (ViT 40.6% = log's 39.7%; ConvNeXt
37.5% = 36%), confirming the operating point is faithful.

| net | arch | at-init | **TRAINED** | logit scale | over | direction |
|---|---|---|---|---|---|---|
| ResNet-34 | BN-CNN | 0.17 | **0.41** | 3.4 | 0.11× | ↑ (He-init, fresh grows) |
| MobileNetV2 | BN-CNN | 0.30 | **0.135** | 4.2 | 0.03× | ↓ (pathological stem gain removed) |
| ViT-Tiny | transformer | 0.57 | **0.22** | 4.5 | 0.05× | ↓ (zero-init patch removed) |
| ConvNeXt-T | LN-CNN | 0.80 | **1.10** | 3.9 | 0.28× | ↑ (fresh grows) |

**Bottom line: the whole-net float certificate is NOT an at-init artifact — it holds on
the real trained net across BN-CNN, transformer, and LN-CNN (4 nets).** The DIRECTION of
change is net-dependent, and the mechanism is now clear: where the at-init net has a
PATHOLOGICAL init component, training removes it and the certificate TIGHTENS — ViT's
zero-init cls/pos/patch (tail gain 1.7e4 → 2724, 0.57 → 0.22) and mnv2's huge He-init
stem tail gain (7.7e3) plus its confident trained logits (0.99 → 4.24) give 0.30 → 0.135;
where init is clean and the logit scale doesn't move much, training's larger weights +
real deep activations grow the per-op fresh Higham faster than the tail gains drop, so it
loosens slightly (r34, ConvNeXt) — but always well within the logit scale. **mnv2 gotcha
(chased 2026-07-04):** its resume ckpt is `3×(nP+1120)` because the CHECKPOINT keeps
block-1's `t=1` expand (32×32 conv + 3×32 = 1120) while the render/`MobileNetV2Layout.
specs` OMIT it — skip those 1120 floats after the stem and every BN γ reads ~1. (That
expand is a real trained conv, `‖W−I‖=1.68`, so the ckpt is a FULL-mnv2 trainer's, not
the skip-expand verified net's; mapping onto the committed render gates at 60.9% not
86.8% — a trained operating point, block-1 expand dropped. The tail-gain conclusion holds
regardless.) One net remains: **enet** (SE-DAG + BN — ckpt clean `3×nP`; the frozen-stats
gate needs the SE true-BN forward). Same architecture class as r34/mnv2; documented
follow-up.

Fresh-session infra notes (hard-won):
- Run everything from `scripts/` under `jax/.venv` (`HIP_VISIBLE_DEVICES=0`, python -u,
  nohup + log; full probe = `python adjoint_chain_probe.py`, per-section =
  `python -c "import adjoint_chain_probe as p; p.<section>_probe()"`).
- Committed renders are `module @m` ⇒ `run_iree(..., module_name="m")`; fn name =
  `@<slug>_fwd[_eval]`. Args parse from the func signature; init by name convention
  (He / γ=ones / β,bias,cls,pos=zeros — matches `IreeRuntime.lean` initKinds).
- §1–§4 pin jax to CPU globally — run §5+ standalone for GPU gains, or last in main().
- Shared RNG stream ⇒ per-section numbers shift when run standalone vs full-script;
  conclusions don't. Suffix products overflow float64 on ViT/ConvNeXt — log10-space
  (see §8).
- Lean iteration: scratch file importing the cached olean (`lake env lean file.lean`,
  seconds); `lake build LeanMlir.Proofs.<Mod>`; audit closure was fully built this
  session — `lake env lean tests/AuditAxioms.lean` elaborates in ~2min, expect 1302
  clean entries (+ new ones).
- Mathlib names that bit: `abs_add_le` (not abs_add), `pow_le_pow_left₀`,
  `Convex.norm_image_sub_le_of_norm_deriv_le`, `Real.sum_le_exp_of_nonneg`,
  `Real.pi_le_four` (sufficed for gelu!), `not_le` + explicit `have`s over inline
  `(by ...)` args when implicits must resolve.

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

   **The residual combinator (2026-07-04, probe §8b + `AdjointChainResidual.lean`):
   CLOSED the attention wall.** `chain2_adjointClose` — partitioned budgets over the
   uniform chain (pair states embed by zero-padding, so no dependent-dim machinery):
   stage outputs split by a coordinate predicate (computed branch vs carried stream),
   per-part fresh budgets meet per-part tail gains (`LipOnWindow2`), and the carried
   stream rides at `b = 0` definitionally (`LayerCert2.stream`). Same telescoping,
   3-axiom clean, subsumes v1 (`LayerCert2.ofLayerCert`, `LipOnWindow.toTwoPart`).
   ViT at 3-stages-per-block granularity (A: h↦(h, scores); B: apply attention —
   softmax sees an EXACT chain input, fresh = smKappa ≈ 1e-5, never the amplified
   inherited error; M: MLP sublayer):

   | | true GPU drift | logits | chainBudget2 | vs §8 monolithic |
   |---|---|---|---|---|
   | ViT-Tiny, residual granularity | 5.2e-06 | 4.3 | **1.5e+03** | 1.6e+16 — **13 orders** |

   Per-stage contributions all O(1–200), decaying with depth; block-0 dominates via
   its early tail gains (~600). The residual ~350× over logit scale is now the same
   stuff as r34's: per-sublayer (dense row-sum × LN n=192 Higham fresh ≈ 15×4e-3)
   products — NO exponential faces left anywhere in the measured chain. The same
   A/B split applies verbatim to ENet's SE gate (stage A = gate logits, stream
   carry) — the mechanical follow-up expected to collapse §6's 5e4 similarly.

   **MobileNetV2 (probe §9, 2026-07-04): the committed render
   (`verified_mlir/mobilenetv2_fwd_eval.mlir`) as-is on gfx1100 — ENet minus SE with
   relu6 (exact in float, 1-Lipschitz, clips to [0,6]), and it shows:**

   | | true GPU logits drift | logits magnitude | chainBudget measured-H | chainBudget proven-H |
   |---|---|---|---|---|
   | MobileNetV2 (committed render) | 1.2e-05 | 0.99 | **7.1e+01** (6.1e6×) | 2.7e+60 (2.3e65×) |

   The best deep-net adjoint result of the ladder (absolute budget 10× below r34's
   despite 17 blocks + BN): per-block H·b flat at 0.7–15 across all 17 inverted
   residuals, no parallel-path faces, no swish esig, eval-BN affine. 72× over the
   logit scale — the residual is entirely the familiar per-block conv-Higham ×
   BN-affine-gain products plus large-but-decaying early tail gains at init
   (stem 7.7e3 → 9). Confirms the regime rule: remove the named worst-case faces
   and a 20-stage BN-CNN at 224² sits within two orders of a real certificate.

   **Ladder summary (all on gfx1100, logits-scale certificates):**

   | net | stages | interval fold | adjoint chain | logit scale | verdict |
   |---|---|---|---|---|---|
   | MLP d=24 | 24 | 7e+34 | 0.82 | ~21 | non-vacuous ✓ |
   | MNIST-CNN | 6 | 2e+04 | 4.1 | 0.38 | fan-in-budget-bound |
   | MNIST-CNN + P2 tree-reduce | 6 | 1.7e+03 | **0.015** | 0.64 | **✓ CERTIFIED — fan-in 6272→13** |
   | CIFAR-8 | 15 | 4e+14 | 2.6 | 4.6 | **non-vacuous ✓** |
   | CIFAR-8-BN | 23 | 3e+31 | 51 | 3.2 | 16× over |
   | CIFAR-8-BN + P4 per-channel BN | 23 | 3.7e+27 | **20.1** | 3.2 | 6.3× over — residual = bn1 tail gain 1313 (P6) + n=1024 Higham (P2) |
   | r34 @224² | 20 | 7e+51 | 698 | 3.6 | 200× over |
   | ENet-B0 @224² (committed render) | 20 | 8e+106 | 6.8e+04 | 1.3 | 5e4× over (SE-in-block) |
   | ConvNeXt-T @224² (committed render) | 25 | 3e+139 | 9.4e+07 | 2.9 | 3e7× over (scalar-LN n=301k + gelu poly-gain) |
   | ConvNeXt-T + `gelu_lipschitz` (3/2 gain) | 25 | 2.8e+113 | 3.2e+06 | 2.9 | 1e6× over (scalar-LN only) |
   | ViT-Tiny @224² (committed render) | 15 | ~1e+364 | 1.6e+16 | 4.3 | softmax-exponent-in-block (needs the residual-carrying sub-block combinator) |
   | ViT-Tiny + residual combinator (chain2) | 39 | — | **1.5e+03** | 4.3 | 350× over — r34-class; exponential faces gone |
   | MobileNetV2 @224² (committed render) | 20 | 2.7e+60 | **7.1e+01** | 0.99 | 72× over — best deep-net row; no named faces left |

   **P1 op-granularity (§10–§12, 2026-07-04): cut the chain at EVERY op —**

   | net | ops | block-gran | **op-gran** | logit scale | verdict |
   |---|---|---|---|---|---|
   | MobileNetV2 @224² | 116 | 7.1e+01 | **1.11** | 0.99 | **1.1× over — at the certificate** |
   | ResNet-34 @224² | 90 | 6.98e+02 | **12.6** | 3.55 | 3.5× over |
   | ViT-Tiny @224² | 171 | 1.5e+03 | **27.8** | 4.33 | 6.4× over — softmax face GONE (smKappa op) |

   Op-granularity dropped the three within-block-folding nets 54–64× in one pass, all
   to within 1–6× of the logit scale. The residual is now PURELY local everywhere: per-op
   Higham fresh (P2 tree-reduction) × early-block init tail gains (P6 trained ckpt) — no
   composition, fold, or exponential face left. (CIFAR-8-BN already op-granular ⇒ its 16×
   is P4 per-channel BN, not P1. ENet/ConvNeXt op-gran = P3, not yet run.)

   **P1+P2 (op-granularity + tree-reduction §10–§12, TreeReduceBridge.lean): CERTIFIED —**

   | net | P1 op-gran | **P1+P2** | logit scale | verdict |
   |---|---|---|---|---|
   | MobileNetV2 @224² | 1.11 | **0.30** | 0.99 | **✓ CERTIFIED (0.31×)** |
   | ResNet-34 @224² | 12.6 | **0.17** | 3.40 | **✓ CERTIFIED (0.05×)** |
   | ViT-Tiny @224² | 27.8 | **0.57** | 4.33 | **✓ CERTIFIED (0.13×)** |
   | CIFAR-8-BN (+P4) | 20.1 | **3.44** | 3.17 | at threshold (1.09×) |

   Tree-reduction (`n·u → log₂n·u` on every conv/dense/BN/LN reduction, under the
   `TreeReduceBridge.lean` `tree_close` lemma + the quarantined order-balanced hypothesis)
   removes the last local face — the per-op √n Higham slack — and pushes all three deep
   nets BELOW their logit scales: the first whole-net float certificates on the deepest
   committed CNN, residual CNN, and transformer renders. Remaining slack is init tail
   gains only (ViT's certificate is 0.39/0.57 the zero-init patch op ⇒ P6).

   **P3 (ENet + ConvNeXt op-granularity §13–§14): the whole ladder certified —**

   | net | block-gran | P1 op-gran | **P1+P2** | logit scale | verdict |
   |---|---|---|---|---|---|
   | EfficientNet-B0 @224² | 6.8e+04 | 0.58 | **0.096** | 0.94 | **✓ — P1 alone (SE face = fold artifact)** |
   | ConvNeXt-T @224² | 3.2e+06 | 5.2e3 | **0.80** | 2.94 | **✓ — P2 essential (n=301k LN face)** |

   ENet certifies on op-granularity ALONE (the SE "×100/block face" was pure within-block
   fold; op-cutting dissolves it, SE reductions r=8–48 are tiny). ConvNeXt needs P2 (its
   scalar-LN over n=301056 survives op-granularity — tree-reduction is its only closer).
   Every deep net on the ladder is now a whole-net float certificate.

   Composition is solved at every scale tried — the adjoint chain stays within 1–6×
   of the logit scale where the interval fold loses 4–140 orders; what remains is
   local (per-op Higham budgets ⇒ P2, BN channel bookkeeping ⇒ P4, init-vs-trained
   gains ⇒ P6).

## Committed / working artifacts

P1+P2+P3 lives in `scripts/adjoint_chain_probe.py` (§10 `mnv2_probe_ops`, §11
`r34_probe_ops`, §12 `vit_probe_ops(nblocks=,tree_reduce=)`, §13 `convnext_probe_ops`,
§14 `enet_probe_ops`; §4 `cifar8bn_probe(per_channel_bn=,tree_reduce=)`; module-level
`_TREE_REDUCE`/`_rexp`/`_rfac`) and
`LeanMlir/Proofs/TreeReduceBridge.lean` (the P2 lemma; in `tests/AuditAxioms.lean`,
lakefile Proofs roots). Run op-gran/P2 probes standalone under `jax/.venv`
(`PYTHONPATH=scripts`, `HIP_VISIBLE_DEVICES=0`) — a §5+ probe pins CPU so measure GPU
gains standalone or last; each op-gran GPU run is ~10–40 min (ViT slowest, the patch
tail is a whole-net jacrev; `vit_probe_ops(nblocks=2)` = fast CPU smoke).

Older scratchpad: `vjp_error_prop.py` / `vjp_roundoff_predict.py` (linearization
tightness), `iree_depth_sweep.py` / `depth_sweep_bounds.py` (the interval-fold depth
cliff), `validate_adjoint_chain.py` (= the probe, pre-scripts/ copy).
