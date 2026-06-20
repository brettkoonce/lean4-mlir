# Planning — FloatBridge: finish the MNIST chain, then low-precision quantization (→ an E4M3 demo)

_Status note, updated 2026-06-20._ Large parts now LANDED (all axiom-clean, in `AuditAxioms.lean`):
**Item D/G1 §1a** (`linear_float_sgd_descends`), **§1c** (`dot_close_mixed` + `dense_close_mixed`:
two-roundoff dot AND dense), **Items A/B/C** (`cnn_float_close` whole-net forward, `cnn_conv{W,b}_step_float_close`
gradient-step, `mnist_cnn_convW_step_float_budget` numeric) — so FloatBridge covers all 3 MNIST nets,
forward + gradient-step + numeric. **E4M3 demo §3a DONE** (`scripts/mnist_e4m3_demo.py`: fp32 92.25% →
E4M3 92.30%, 92.89% margin>2B verified-region). **§3c DONE** (`FloatBridge.lean`:
`argmax_preserved` + `dense_close_mixed_uniform_budget` + `linear_e4m3_logit_budget` (worst-case `B ≤ 61`)
+ the capstone `linear_e4m3_argmax_preserved` — margin > 2B ⟹ provably same prediction; all 3-axiom clean,
audited). **§3b DONE** (`E4M3FaithfulPoC.lean`: `e4m3_render_faithful` — the emitted block-scaled
int-matmul graph denotes the intended dequant-first algorithm, `dequant_factors` the scale-factors-out
heart; zero new SHlo constructors, 3-axiom clean, audited). **All of §3 (3a/3b/3c) now landed.**
Two threads that share one foundation
(`LeanMlir/Proofs/FloatBridge.lean`, the `FloatModel` relative-error model, parametric in the
unit roundoff `u`):

1. **Finish the MNIST float chain** — close the two structural gaps so all three MNIST nets
   (linear / mlp / cnn) have the complete `binary32 → certified proximity → proven smoothness →
   loss decreases` story. Mostly reuse; no new analysis.
2. **Low-precision quantization** — generalize the dot/dense budget to a *two-roundoff* model
   (`u_leaf` for the matmul inputs, `u_acc` for the accumulation), which covers the deployed
   **bf16-mixed** config for free, and sets up **fp8 (E4M3)** — with an honest, scoped
   **E4M3 MNIST demo** as the concrete target.

The key design fact this all rests on: `FloatModel` already carries `u : ℝ` as a field
(`FloatBridge.lean:42`), with `u32 := (2^24)⁻¹` (`:49`). Every budget theorem
(`dot_close`, `layerBudget`, `sgdErr`, `cotErr`, the SGD-descent η-oracle) is stated for
**abstract `u`**. So precision is an instantiation, not a rewrite — *up to the point where the
model's assumptions break* (§2).

---

## 0. Where FloatBridge is now (baseline → current)

| Net | Rounding-proximity (`*_float_close`) | Descent (Lipschitz + `sgd_descends`) |
|---|---|---|
| **linear** | ✅ `linear_float_close` (fwd) + `linear_grad_close` + **`linear_float_sgd_descends`** (η closed) | ✅ `linear_loss_grad_lipschitz` + `linear_sgd_descends` |
| **mlp** | ✅ fwd + **all 6** param `*_step_float_close` + 2 numeric capstones (`mnist_mlp_float_budget`, `mnist_w2_step_float_budget`) | ✅ per-layer `mlp_{output,hidden,input}_sgd_descends` |
| **cnn** | ✅ **`cnn_float_close`** (whole-net fwd) + `cnn_conv{W,b}_step_float_close` + `mnist_cnn_convW_step_float_budget` | ✅ `cnn_conv{1,2}{,_bias}_sgd_descends` + Lipschitz |

**The two structural gaps that drove this plan — both now CLOSED:**

- **G1 — the η-composition ("the two halves never meet").** Was: `*_sgd_descends` took gradient
  accuracy `η` as an **abstract** parameter with no theorem feeding a FloatBridge budget into it.
  **Closed on linear** (`linear_float_sgd_descends`, §1a) — rounding and descent now compose into one
  statement. (mlp/cnn η-composition still open — needs the mlp joint-step refinement; §4.)
- **G2 — CNN has no rounding side.** Was: linear/mlp had `*_float_close`, cnn didn't. **Closed**
  (§1b, Items A/B/C) — cnn now has forward + gradient-step + numeric rounding budgets.

**The precision-scaling insight (from the bf16/fp8/fp4 discussion):**
- `u`: fp32 `2⁻²⁴`, bf16 `2⁻⁸`, fp8-E4M3 `2⁻⁴` (6.25%), fp8-E5M2 `2⁻³`, fp4-E2M1 `2⁻²` (25%).
- The Higham `γ_k = k·u/(1−k·u)` goes vacuous at fan-in `k = 1/u`: bf16 256, **fp8 16, fp4 4**.
- What saves low precision is **mixed precision**: the deployed config (`jax/scripts/jax_r34_bf16_bench.py`)
  is *fp32 master weights, bf16 leaf compute, fp32 accumulate, fp32 BN/softmax/GAP/classifier*. The
  fan-in amplification stays at `u_acc = 2⁻²⁴`; the low precision only contributes a **flat per-leaf
  term** at `u_leaf`. That is the difference between vacuous and useful.
- Past bf16 the model itself changes regime (§2): the relative bound `|rnd x−x| ≤ u|x|` stops holding
  (dynamic range too small), and quantization becomes **block-scaled** — so the "verified" claim
  migrates from *numerical accuracy* toward *structural faithfulness of the quantization scheme*.

---

## 1. Finish the MNIST float chain (all near-term, all tractable)

### 1a. Item D — the η-composition (do this FIRST, on linear) — ✅ DONE (2026-06-19)

**The cheapest, highest-leverage thing in the whole float story.** Instantiate the descent η-slot
with the actual rounding budget so the two halves become one theorem.

**Landed** in `SgdDescentLinear.lean` (3-axiom clean, audited):
- `FloatModel.linearFloatGrad` — the *actual* binary32 gradient the rendered trainer computes:
  float forward logits `M.dense W b x` → rounded softmax−onehot cotangent (`softmaxCECotF`) → one
  rounded multiply by the exact input `xᵢ`, flattened to the `Vec (m*n)` parameter layout.
- `linear_grad_close` — that gradient is within `mulErr u a 1 0 (cotErr u eexp δ n)` of the
  certified `∂L/∂Wᵢⱼ`, per entry (head via `softmax_ce_cot_close`; the input multiply via one
  `mul_close` with an exact left operand `ea = 0`, `C = 1` since `softmax−onehot ∈ [−1,1]`).
- `linear_float_sgd_descends` — discharges `linear_sgd_descends`' abstract `η` with that proven
  budget: **one binary32 SGD step on MNIST-linear provably decreases the cross-entropy loss, no
  abstract gradient-accuracy parameter.** Residue = the documented FloatModel→kernel trust boundary
  (`exp` accuracy `eexp`, a-posteriori logit drift `δ`) + checkable arithmetic (small-step, the two
  dominance conditions). Depth-1 ⇒ no per-layer η-threading, exactly as predicted.

*Still open (§4):* replicate the η-composition for mlp (per-layer η from `mlp_*_step_float_close`) —
heavier only because of the **joint-step** refinement (logits aren't affine when all params move at
once; needs a joint Lipschitz or a per-coordinate decomposition — already flagged in `SgdDescentMlp`).

### 1b. Items A/B/C — bring the rounding side to CNN (= "FloatBridge to the rest of MNIST")

CNN already had the *descent* side (`SgdDescentCnn.lean`: `MaxPool2MarginQ`, pool drift,
conv = dense-with-sharing); it lacked the *rounding* side. **Now added** (A/B/C below), ~70% reuse.

**Item A is DONE (2026-06-19) — `cnn_float_close` closed end-to-end.** All 3-axiom clean, audited.
- *Exact maxpool* — `max_close` / `maxPool2_close` / `maxPoolFlat_close` (+ `maxPoolFlat_abs_le`)
  (`CNN.lean`): max is compare-and-select, rounds nothing, so inherited error `e` passes through with
  no rounding term and no amplification — the `max`-peer of `relu_close`.
- *Conv forward budget* — `conv2d_eq_dense` / `FloatModel.convF` / `FloatModel.convF_close` plus the
  `Vec`-space `FloatModel.flatConvF` / `flatConvF_close` and magnitude bound `flatConv_abs_le`
  (`SgdDescentCnn.lean`): `conv2d_eq_dense` makes "conv = dense-with-sharing" exact (each output coord
  is `Proofs.dense` of the kernel slab against the flattened window, via `sum_w3` collapsing the
  triple sum to one fan-in sum); `convF` = `M.dense` on the window; `convF_close`/`flatConvF_close` =
  `dense_close`/`layerBudget` at fan-in `ic·kH·kW` — so a conv layer threads **identically to a dense
  layer**.
- *Whole-net capstone* — `FloatModel.mnistCnnNoBnForwardF` (float forward) + `FloatModel.cnn_float_close`
  (`SgdDescentCnn.lean`): the binary32 forward-error bound for the whole Chapter-4 CNN, an explicit
  closed-form `layerBudget` nest over `conv→relu→conv→relu→maxpool→dense→relu→dense→relu→dense`
  (the `mlp_float_close_uniform` pattern extended to six layers; relu/maxpool exact-in-float pass
  error through unamplified). **The chain `binary32 → certified proximity` is now closed for all three
  MNIST nets (linear / mlp / cnn).**

- **B — `cnn_convW/convb_step_float_close` (gradient-step rounding). ✅ DONE (2026-06-19).** The conv
  weight grad is a **correlation = a dot** over the `h·w` spatial positions; the bias grad is a
  spatial **sum**. Both rounded SGD steps reduce to two reusable generic cores in `FloatBridge.lean` —
  `dotSgd_step_close` / `sumSgd_step_close` (= `dot_close` / `sum_close` feeding `sgd_step_close`).
  `convWeightGrad_eq_dot` / `convBiasGrad_eq_sum` (`SgdDescentCnn.lean`, via `sum_s2` + the
  `convPadWin` / `cotWin` flattenings) re-express the certified conv gradient (`conv2d_weight_pdiv`)
  as that flat dot/sum; `cnn_convW_step_float_close` / `cnn_convb_step_float_close` are then the
  generic cores instantiated — the rounded conv weight/bias update within `sgdErr` of the real step,
  the dot/sum Higham γ (fan-in `h·w`) as the gradient-error slot. The cotangent is hypothesis-supplied
  (same as `mlp_w2_step_float_close`; the loss-head `exp` accuracy lives in `cotErr`). All 3-axiom
  clean, audited.
- **C — numeric capstone (`mnist_cnn_convW_step_float_budget`). ✅ DONE (2026-06-20).** At the
  committed Chapter-4 dims (conv2 `32→32`, `3×3`, `28×28` ⇒ weight-grad fan-in `28·28 = 784`),
  `u ≤ 2⁻²⁴`, `lr = 1/10`, `|W| ≤ 3/5` (the trained-magnitude bound, matching the MLP capstone): every
  rounded conv2 weight SGD entry is within **`(a·g)/250 + 10⁻⁷`** of the certified step. Here `a`
  bounds the conv-input activation and `g` the conv cotangent — both **a-posteriori / measured**
  (supplied as hypotheses, since the conv input and back-propagated cotangent are not intrinsically
  `≤ 1`, unlike the softmax−onehot head). The `1/250 ≈ 0.4%` rate is `lr·γ₇₈₅` — the gradient's
  Higham error at learning-rate scale: the conv weight step is as accurate as the gradient itself.
  3-axiom clean, audited. (Plugging the measured `a, g` from a `margin_probe`-style run yields the
  final single decimal.)

Effort: A/B/C **all DONE** — the CNN rounding side is complete.

### 1c. Do A/B *parametric in two roundoffs* (the free bf16 + fp8 setup) — ✅ foundation DONE (2026-06-19)

When writing the conv/dense dot budget for A/B, split the single `u` into **`u_leaf`** (rounding the
matmul inputs) and **`u_acc`** (the accumulation):

```
dot_close_mixed : |fl_mixed(x·y) − x·y| ≤ (per-leaf term at u_leaf) + (Higham γ_k at u_acc)
```

This is a localized generalization of `dot_close` (`FloatBridge.lean:218`).

**Landed** in `FloatBridge.lean` (3-axiom clean, audited): the leaf precision is a second
`FloatModel L` (`u_leaf`), the accumulate is `M` (`u_acc`).
- `FloatModel.dotMixed L x y` = `M.dot (L.rnd ∘ x) (L.rnd ∘ y)` — the bf16-mixed kernel shape.
- `dot_close_mixed` — `|dotMixed − x·y| ≤ ((1+u_acc)^(n+1) − 1)·Σ|x̃ỹ| + (2·u_leaf + u_leaf²)·Σ|xy|`:
  the leaf term is **flat** (not fan-in amplified); the fan-in γ rides entirely on `u_acc`. The
  formal statement of "the `1/u` fan-in wall sits at `u_acc`, not the leaf."
- `dot_close_mixed_uniform` — folded to one `Σ|xy|` factor `[γ_acc·(1+u_leaf)² + 2u_leaf + u_leaf²]`,
  the directly-instantiable shipped-artifact form.
- `dotMixed_exact_leaf` — `u_leaf = 0` collapses it to `dot_close` (a genuine generalization).
- **`FloatModel.denseMixed` + `dense_close_mixed`** — `dotMixed` threaded through the **dense layer**
  (leaf precision `L` on the matmul, accumulate `M` on the bias add): the leaf precision enters only
  via the flat `dotMixed` term, the accumulate rides the bias add + fan-in γ. The deployed bf16-mixed
  dense layer; bf16 / fp8 dense fall out by setting `L.u`.

The three numeric instantiations now drop straight out of `dot_close_mixed_uniform` / `dense_close_mixed`
by choosing `L.u`:
- **fp32**: `u_leaf = u_acc = 2⁻²⁴` (current behavior).
- **bf16-mixed** (the deployed config): `u_leaf = 2⁻⁸`, `u_acc = 2⁻²⁴` — non-vacuous because the
  fan-in term rides at fp32; the leaf term is a flat `~2·2⁻⁸ ≈ 0.8%`. Reductions (BN/softmax/GAP)
  stay at `u_acc` and reuse the existing fp32 budgets verbatim.
- **fp8 (next section)**: `u_leaf = 2⁻⁴`, with the block-scale caveat of §2.

So: write A/B once with `(u_leaf, u_acc)`, instantiate three ways. Verifies the **actually-shipped**
artifact (the ImageNet checkpoints are bf16-mixed: `r34_imagenet_bf16.bin`), not a hypothetical fp32 one.

---

## 2. Simple quantization — the model extension (and where it changes regime)

"Simple quantization" = **per-tensor (or per-row/per-block) scaling + low-precision mantissa +
fp32 accumulate**:

```
x ≈ s · q      where  q = round_to_fpK(x / s),   s = scale (per tensor / per row / per 32-block)
```

Two regime changes vs. fp32/bf16:

1. **The rounding op is block-scaled, not globally relative.** The error becomes **per-block
   absolute**: `|rnd x − x| ≤ u·s_block`, with `s_block` a data-dependent quantity to track.
   `FloatModel` needs a **block-scale field** (and the "subnormals / absolute regime" — currently a
   half-open footnote — becomes the main case, because fpK's tiny exponent range means most values
   are near the scale, not in a wide normal band).
2. **Worst-case end-to-end accuracy gives way.** With fp32 accumulate the **per-matmul** bound is
   non-vacuous (the leaf term `u_leaf` is flat, not fan-in-amplified) but *large* (E4M3 ~6%); over
   depth the worst-case compounds and goes vacuous. So past one layer, an honest accuracy claim is
   **a-posteriori / probabilistic**, and fp4 "working" is fundamentally a *network-robustness*
   (training-dynamics) property, outside the arithmetic model.

**Where the verification value actually is at low precision: structural faithfulness.** The tractable,
*complete* claim is the **render-tie of the quantization scheme** — model `quantize`/`dequantize`/
`block-scale` as `den`-able ops and prove `den(emitted fpK graph) = the intended block-scaled
algorithm (block-quantize inputs → fp32 accumulate → dequantize)`. This is a *correctness-of-
implementation* claim (the existing `den`/render machinery is the right tool) — not an accuracy
bound, and it does not pretend to one.

Mental model to keep crisp:
`fp32` accuracy-provable · `bf16-mixed` accuracy-provable (§1c) · `fp8` per-matmul accuracy-provable,
end-to-end only a-posteriori · `fp4` mostly structural-faithfulness + statistical robustness.

---

## 3. The E4M3 MNIST demo (the concrete target)

**Why MNIST-linear at E4M3 is the sweet spot:** it is **depth-1** (a single 784→10 matmul), so the
per-matmul leaf bound *is* the end-to-end bound — **no vacuous depth compounding.** This is the one
realistic case where an **honest end-to-end accuracy bound at fp8 exists**. Three deliverables,
increasing in ambition:

### 3a. Empirical demo (runnable, numpy — the "it works" headline) — ✅ DONE (2026-06-20)
- **Landed:** `scripts/mnist_e4m3_demo.py` (numpy-only, sibling of `scripts/margin_probe.py`; no JAX
  dep needed). Trains an fp32 MNIST-linear baseline, then fake-quantizes the trained weights AND test
  activations to **E4M3** (per-row weight scale, per-tensor activation scale, **fp32 accumulate**,
  fp32 softmax) — exactly the `dotMixed` model (u_leaf = E4M3, u_acc = fp32). `to_e4m3` is a faithful
  round-to-nearest E4M3 (1-4-3, bias 7, max 448, subnormals to 2⁻⁹, saturating).
- **Result (seed 0, 20 epochs):** fp32 **92.25%**, E4M3 **92.30%** — a *+0.05pt* "drop" (statistical
  noise); prediction agreement 99.63%; logit drift mean 0.047 / max 0.38. **"Precision drops
  elegantly" confirmed.** Also logs the fp32 logit-margin distribution (mean 4.25) for 3c.
- The script also computes the **3c argmax-preservation fraction empirically** (see 3c): 92.89% of
  the test set has margin > 2B, and a built-in check confirms 100% of those keep their prediction.

### 3b. Structural faithfulness (the verified part that's *complete*) — ✅ DONE (2026-06-20)
**Landed** in `E4M3FaithfulPoC.lean` (3-axiom clean, audited):
- `actCode`/`weightCode` — the stored integer-grid codes: activation `q(xᵢ/sx)` (per-tensor `sx`),
  weight `q(Wᵢⱼ/sWⱼ)` (per-output-column block scale `sWⱼ`). `q : ℝ → ℝ` is the quantizer, left
  **abstract** (E4M3 round-to-nearest is one instance) — the scheme is faithful for any grid.
- `e4m3LinearGraph` — the emitted block-scaled-E4M3 graph: `operand` (int activation code) → `dotIn`
  (int weight code; its `den` `∑` is the fp32 accumulate) → `layerScaleF` (the per-output dequant
  block-scale `sx·sWⱼ`) → `addBcast` (fp32 bias). Built **only from existing `den`-faithful ops** —
  **zero new `SHlo` constructors** (so it's also fully printable via the existing `pretty`/parser).
- `dequant_factors` — the arithmetic heart: `(sx·sWⱼ)·∑ᵢ q(xᵢ/sx)·q(Wᵢⱼ/sWⱼ) = ∑ᵢ (sx·q(xᵢ/sx))·(sWⱼ·q(Wᵢⱼ/sWⱼ))`.
  The per-output dequant scale **factors out of the accumulate**, so "int matmul then one dequant" =
  "dequantize each operand then matmul" — exactly what fp32 accumulate buys (scales constant across
  the reduction).
- `e4m3_render_faithful` — the **render-tie**: `den(e4m3LinearGraph) = quantLinear` (the intended
  dequant-first algorithm = `mnistLinear` on the round-tripped tensors). "The bytes correctly
  implement block-scaled-E4M3 matmul with fp32 accumulate," **no accuracy claim**.
- *Design note.* Modeled quantize-to-code as the offline/runtime byte preparation that produces the
  operands (how real fp8 inference works), so no abstract `quantF` op was needed — dequant/block-scale
  *is* a `den`-able op (`layerScaleF`). The committed-bytes tie (a quantized `.mlir` trainer) isn't set
  up (no fp8 trainer exists yet); the `renderModule` text half is mechanical (all ops printable) but
  unwired — the den-level faithfulness is the verified render-tie, same boundary as the other PoCs.

### 3c. Per-matmul accuracy bound (the honest fp8 accuracy statement — only because depth-1) — ✅ DONE (2026-06-20)
**Landed** in `FloatBridge.lean` (all 3-axiom clean, audited):
- `argmax_preserved` — the pure conditional core: if every logit of `z'` is within `B` of `z` and `z`'s
  strict top-1 margin at `k` exceeds `2B`, then `k` is still the strict argmax of `z'`. `B` is a
  *hypothesis*, so the same theorem covers both the proven worst-case bound and the demo's measured
  a-posteriori drift. Conditional exactly like the suite's quantitative ReLU margins.
- `denseMixedBudget` + `dense_close_mixed_uniform_budget` (+ the `denseMixedBudget_le_of` monotone
  helper, the `layerBudget_le_of` analogue) — one *uniform* per-logit `B` over all outputs, from the
  §1c `dense_close_mixed`. The fan-in power is kept abstract through `denseMixedBudget_le_of` so the
  concrete instance never unfolds the 785-fold `npow`.
- `u_e4m3 := 2⁻⁴`; `linear_e4m3_logit_budget` — at the committed 784→n dims, `u_leaf ≤ 2⁻⁴`,
  `u_acc ≤ 2⁻²⁴`, `|x| ≤ 1`, `|W| ≤ 3/5`, `|b| ≤ 1`: every E4M3-mixed logit is within **61** of the
  exact-ℝ logit (worst-case; the flat `2·2⁻⁴ ≈ 12.5%` leaf term dominates, the fp32 fan-in
  γ₇₈₅ ≈ 5·10⁻⁵ is negligible).
- `linear_e4m3_argmax_preserved` — the capstone: margin > `2·61 = 122` ⟹ the E4M3 forward keeps the
  top class, **provably the same prediction**. Depth-1 makes the single-matmul bound the end-to-end
  bound (no vacuous compounding).
- **The honesty.** The *worst-case* threshold 122 is vacuous on real data (mean fp32 margin ≈ 4.25).
  The same `argmax_preserved` with the demo's **measured** `B = 0.38` (errors cancel; worst-case
  assumes them aligned) covers the `>0.76`-margin inputs — empirically **92.89%** of the MNIST test
  set, and a built-in check confirms 100% of those keep their prediction (`scripts/mnist_e4m3_demo.py`).
  So: *provably same prediction on the margin-`>2B` inputs; with the measured `B` that is 92.89% of
  the test set.* (`fp32 ≈ exact-ℝ` within `u_acc`, so the demo's fp32 margins are the relevant quantity.)

**Stretch:** the 784→64→10 mlp at E4M3 — now depth-2, so the end-to-end accuracy bound starts to
compound and you'd lean on 3c per-layer + the margin fraction (still honest, larger decimals). Good
to *show* the compounding so the regime change from §2 is visible on a real net.

---

## 4. Sequencing & effort

| Order | Item | Effort | Payoff |
|---|---|---|---|
| ✅ | **G1/Item D on linear** (`linear_float_sgd_descends`) | light | **DONE 2026-06-19** — closes the chain end-to-end for one net, the biggest honesty win |
| ✅ | **§1c two-`u` `dot_close_mixed`** (foundation) | light | **DONE 2026-06-19** — bf16-mixed (shipped artifact) falls out + sets up fp8; dense/conv threading lands with A/B |
| ✅ | **A/B/C — CNN rounding side** | medium | **DONE 2026-06-19/20.** A (forward `cnn_float_close`), B (`cnn_conv{W,b}_step_float_close`), C (`mnist_cnn_convW_step_float_budget`, decimal `(a·g)/250 + 10⁻⁷`). FloatBridge is now on all 3 MNIST nets, forward + gradient-step + numeric |
| ✅ | **3a E4M3 MNIST empirical demo** | light | **DONE 2026-06-20** — fp32 92.25% → E4M3 92.30%, the "precision drops elegantly" headline; computes the 3c margin fraction empirically (92.89%) |
| ✅ | **3c E4M3 per-matmul accuracy + margin fraction** | medium | **DONE 2026-06-20** — `argmax_preserved` + `linear_e4m3_logit_budget` (worst-case B ≤ 61) + `linear_e4m3_argmax_preserved` (margin > 122 ⟹ same prediction); measured B = 0.38 ⟹ 92.89% of test set. The honest end-to-end fp8 accuracy bound (depth-1); the Lean side of what 3a measured |
| ✅ | **3b E4M3 structural faithfulness** | medium | **DONE 2026-06-20** — `e4m3_render_faithful` (emitted block-scaled int-matmul graph denotes the dequant-first algorithm) + `dequant_factors` (scale factors out of fp32 accumulate); zero new SHlo constructors. The *complete* verified claim at fp8 (the right kind). **All of §3 landed.** |
| — | G1 for mlp/cnn + mlp joint-step | medium | finishes the descent composition across the chain |
| — | fp4 / block-scaled `FloatModel` field / probabilistic budgets | heavy/research | only if pushing below fp8; expect structural-only claims |

**Definition of done (per item):** every new theorem `#print axioms`-closes under
`[propext, Classical.choice, Quot.sound]`, added to `tests/AuditAxioms.lean`; no `sorry`, no project
axiom. The demo scripts live in `scripts/` with reference numbers in the header (like
`scripts/margin_probe.py` — `scripts/mnist_e4m3_demo.py` follows it). README/blueprint float section updated.

---

## 5. Out of scope / honest caveats (state these, don't let them blur)

- `FloatModel` is an abstract ℝ rounding model (`|rnd x−x| ≤ u|x|`), **not** Lean's `Float`/IEEE; the
  link `FloatModel → IREE bf16/fp8 kernels` stays empirical (the same trust boundary as today).
- Worst-case accuracy is honest only through bf16-mixed and (depth-1) fp8; below that the claim must
  be **structural faithfulness** + **a-posteriori margin fractions** — say so plainly. fp4 inference
  "working" is a network-robustness fact, not an arithmetic one.
- The numeric capstones must be re-instantiated at **trained magnitudes** (the `|W| ≤ 3/5` lesson) —
  the parametric theorems cover any `W`; only the decimal headlines assume a magnitude bound.
