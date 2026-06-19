# Planning — FloatBridge: finish the MNIST chain, then low-precision quantization (→ an E4M3 demo)

_Status note, 2026-06-19._ Forward-looking plan. Two threads that share one foundation
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

## 0. Where FloatBridge is now (the honest baseline)

| Net | Rounding-proximity (`*_float_close`) | Descent (Lipschitz + `sgd_descends`) |
|---|---|---|
| **linear** | ✅ `linear_float_close` (fwd) | ✅ `linear_loss_grad_lipschitz` + `linear_sgd_descends` |
| **mlp** | ✅ fwd + **all 6** param `*_step_float_close` + 2 numeric capstones (`mnist_mlp_float_budget`, `mnist_w2_step_float_budget`) | ✅ per-layer `mlp_{output,hidden,input}_sgd_descends` |
| **cnn** | ❌ **none** (`SgdDescentCnn.lean` has only the descent side) | ✅ `cnn_conv{1,2}{,_bias}_sgd_descends` + Lipschitz |

**Two structural gaps:**

- **G1 — the η-composition ("the two halves never meet").** `*_sgd_descends` takes the gradient
  accuracy `η` as an **abstract** parameter (`hgh : |gh − gradAt f| ≤ η`); **no theorem feeds a
  FloatBridge budget (`sgdErr`/`cotErr`) into that `η`-slot.** So the rounding side and the descent
  side coexist but never compose into one statement. This is the single biggest honesty win
  available, and it is *independent of which net*.
- **G2 — CNN has no rounding side.** Linear/mlp have `*_float_close`; cnn does not. Bringing
  FloatBridge "to the rest of the MNIST models" is essentially this.

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

### 1a. Item D — the η-composition (do this FIRST, on linear)

**The cheapest, highest-leverage thing in the whole float story.** Instantiate the descent η-slot
with the actual rounding budget so the two halves become one theorem.

- `linear_sgd_descends` (`SgdDescentLinear.lean`) currently: `∀ η, (gradient is within η) → loss
  drops by ≥ lr‖∇‖²/2 − taxes(η)`. The FloatBridge budget that bounds the rounded gradient is the
  per-entry `sgdErr`/`cotErr` family (`FloatBridge.lean:683,1446`).
- **The theorem to add:** `linear_float_sgd_descends` — discharge `η` by the FloatBridge budget at
  the linear net's single layer, so the statement becomes unconditional-given-float:
  *"one binary32 SGD step on MNIST-linear decreases the cross-entropy loss"* — no abstract `η`.
- Linear is **depth-1**, so there is no per-layer η-threading and no joint-step subtlety — this is
  the clean pilot. Effort: **light** (it is a wiring + one inequality chain). Payoff: the chain
  `binary32 → proximity → smoothness → descent` is *closed end-to-end for one net*.

Then replicate for mlp (per-layer η from `mlp_*_step_float_close`) — heavier only because of the
**joint-step** refinement (logits aren't affine when all params move at once; needs a joint Lipschitz
or a per-coordinate decomposition — already flagged as open in `SgdDescentMlp`).

### 1b. Items A/B/C — bring the rounding side to CNN (= "FloatBridge to the rest of MNIST")

CNN already has the *descent* side (`SgdDescentCnn.lean`, 6771 lines: `MaxPool2MarginQ`, pool drift,
conv = dense-with-sharing). It lacks the *rounding* side. Reuse ~70%.

- **A — `cnn_float_close` (forward rounding budget).** Conv is a sum-of-products ⇒ the **dense
  Higham budget (`dot_close`/`layerBudget`) at conv fan-in `kH·kW·ic`**. The structural fact
  `conv = dense-with-sharing` already exists (`SgdDescentCnn`: `conv2d_eq_convPad`, affine-in-kernel).
  Two kink/pool facts:
  - **maxpool is exact in float** — `max(a,b)` is one of `a,b` (compare-and-select, no arithmetic, no
    rounding). The float peer of exact-ℝ max; FloatBridge already notes "relu = max-with-0 is exact in
    float" (`:84`) — this just lifts it to the 2×2 window. **The one genuinely-new lemma, and it's easy.**
  - **relu exact in float** — already have (`reluMask_close`, the quantitative margin `ez < |z|`).
  So `cnn_float_close` = the dense float-close at conv fan-in, threaded through exact relu + exact
  maxpool + the (done) dense head.
- **B — `cnn_convW/convb_step_float_close` (gradient-step rounding).** The conv weight grad is a
  **correlation** = another dot product ⇒ reuse the dense-gradient machinery (`mlp_w*_step_float_close`)
  at the conv grad fan-in. The backward routing is exact-in-float under the margins *already proven*
  on the descent side (`MaxPool2MarginQ` freezes the argmax; the relu mask is the margin condition).
- **C — numeric capstone at trained magnitudes** (`mnist_cnn_*_step_float_budget`). Instantiate B at
  the CNN's measured trained `|W|` (mirroring `mnist_w2_step_float_budget`). `norm_num` once B exists.

Effort: A/B **medium** (mostly gluing dense budgets + existing CNN margin lemmas; the only fresh
content is maxpool-exact-in-float); C **easy**.

### 1c. Do A/B *parametric in two roundoffs* (the free bf16 + fp8 setup)

When writing the conv/dense dot budget for A/B, split the single `u` into **`u_leaf`** (rounding the
matmul inputs) and **`u_acc`** (the accumulation):

```
dot_close_mixed : |fl_mixed(x·y) − x·y| ≤ (per-leaf term at u_leaf) + (Higham γ_k at u_acc)
```

This is a localized generalization of `dot_close` (`FloatBridge.lean:218`). It costs little and buys:
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

### 3a. Empirical demo (runnable, JAX/numpy — the "it works" headline)
- Quantize the trained MNIST-linear (or a 784→64→10 mlp) to **E4M3**: per-row weight scale, per-tensor
  activation scale, **fp32 accumulate**, fp32 softmax. Mirror `jax/scripts/jax_r34_bf16_bench.py`'s
  mixed-precision structure (a `compute_dtype` knob; here add a fake-quant `to_e4m3(x/s)*s`).
- Measure: top-1 vs. the fp32 baseline (92.1% for linear). Expectation: small drop — MNIST is
  well-separated; this is the "precision drops elegantly" demo. Also log the **logit-margin
  distribution** (needed for 3c).
- Deliverable: `jax/scripts/mnist_e4m3_demo.py` + a one-line result in the README/blueprint.
- Effort: **light** (a day of numpy/JAX).

### 3b. Structural faithfulness (the verified part that's *complete*)
- Add E4M3 `quantize`/`dequantize` (with a per-row scale) as `den`-able ops, and prove the emitted
  quantized-linear graph denotes `dequant(fp32-accumulate(quant(W), quant(x)))` — the **render-tie
  for the E4M3 scheme**. This says "the bytes correctly implement block-scaled-E4M3 matmul with fp32
  accumulate," with no accuracy claim. Reuses the `SHlo`/`den`/`pretty` machinery.
- Effort: **medium** (new quantize/dequantize op `den`s + a faithfulness lemma; the linear render is
  the smallest existing template).

### 3c. Per-matmul accuracy bound (the honest fp8 accuracy statement — only because depth-1)
- Instantiate the §1c two-`u` dot budget at `u_leaf = 2⁻⁴` (E4M3), `u_acc = 2⁻²⁴` (fp32 accumulate),
  with the per-row scale `s`. Result: each logit is within `B = (E4M3 leaf term) + (fp32 Higham at
  fan-in 784)` of the exact-ℝ logit. The leaf term dominates (~6%·‖row‖-scaled).
- **Argmax-preservation form** (the useful claim, conditional like the existing ReLU margins):
  *for any input whose fp32 logit margin exceeds `2B`, the E4M3 prediction equals the fp32
  prediction.* Then **measure** (from 3a's margin histogram) the fraction of the MNIST test set that
  satisfies it — the a-posteriori number (expected ~95%+). This is the honest "verified E4M3 MNIST"
  statement: *provably same prediction on the margin-`>2B` inputs, empirically that's X% of the test set.*
- Effort: **medium** (the bound is the two-`u` `dot_close_mixed` at one layer + an argmax-margin
  lemma; the depth-1 structure removes all the compounding pain).

**Stretch:** the 784→64→10 mlp at E4M3 — now depth-2, so the end-to-end accuracy bound starts to
compound and you'd lean on 3c per-layer + the margin fraction (still honest, larger decimals). Good
to *show* the compounding so the regime change from §2 is visible on a real net.

---

## 4. Sequencing & effort

| Order | Item | Effort | Payoff |
|---|---|---|---|
| 1 | **G1/Item D on linear** (`linear_float_sgd_descends`) | light | closes the chain end-to-end for one net — the biggest honesty win |
| 2 | **§1c two-`u` `dot_close_mixed`** | light | bf16-mixed (shipped artifact) falls out + sets up fp8 |
| 3 | **A/B/C — CNN rounding side** | medium | brings the 3rd MNIST net to mlp parity ("FloatBridge to the rest of MNIST") |
| 4 | **3a E4M3 MNIST empirical demo** | light | the "precision drops elegantly" headline |
| 5 | **3b E4M3 structural faithfulness** | medium | a *complete* verified claim at fp8 (the right kind) |
| 6 | **3c E4M3 per-matmul accuracy + margin fraction** | medium | the honest end-to-end fp8 accuracy bound (depth-1) |
| — | G1 for mlp/cnn + mlp joint-step | medium | finishes the descent composition across the chain |
| — | fp4 / block-scaled `FloatModel` field / probabilistic budgets | heavy/research | only if pushing below fp8; expect structural-only claims |

**Definition of done (per item):** every new theorem `#print axioms`-closes under
`[propext, Classical.choice, Quot.sound]`, added to `tests/AuditAxioms.lean`; no `sorry`, no project
axiom. The demo scripts live in `jax/scripts/` with reference numbers in the header (like
`scripts/margin_probe.py`). README/blueprint float section updated.

---

## 5. Out of scope / honest caveats (state these, don't let them blur)

- `FloatModel` is an abstract ℝ rounding model (`|rnd x−x| ≤ u|x|`), **not** Lean's `Float`/IEEE; the
  link `FloatModel → IREE bf16/fp8 kernels` stays empirical (the same trust boundary as today).
- Worst-case accuracy is honest only through bf16-mixed and (depth-1) fp8; below that the claim must
  be **structural faithfulness** + **a-posteriori margin fractions** — say so plainly. fp4 inference
  "working" is a network-robustness fact, not an arithmetic one.
- The numeric capstones must be re-instantiated at **trained magnitudes** (the `|W| ≤ 3/5` lesson) —
  the parametric theorems cover any `W`; only the decimal headlines assume a magnitude bound.
