# Float bridge §3 — the descent pass (handoff for a clean session)

The last open item of `planning/floatbridge_certificate_gaps.md`: push "a rounded
training step still **decreases the loss**" past the linear net. Everything else
in the bridge is *closeness* (`|float − real| ≤ budget`); descent says "it provably
trains." This doc is the cold-start plan — current state, the exact next rung, and
the honest stop line.

The whole pattern is **one master theorem + a per-rung η source.**
`SgdDescentLinear.linear_float_sgd_descends` is the master: it takes the *actual*
float gradient and an accuracy `η` *proven* (not assumed), and concludes one
binary32 SGD step decreases the CE loss by ≥ `lr·‖∇L‖₂²/2`. Each new rung = supply
that `η` from a per-layer **float-backward grad-close** and discharge the same
small-step + dominance arithmetic.

## State (verified 2026-06-25)

| rung | grad-close (η source) | descent wiring | status |
|---|---|---|---|
| **linear** | `linear_grad_close` | `linear_float_sgd_descends` | ✅ CLOSED end-to-end |
| **MLP output `W₂`** | (= linear at `a₁`) | `mlp_output_float_sgd_descends` | ✅ CLOSED (no ReLU below ⇒ no margin) |
| **MLP hidden `W₁`** | `mlp_w1_grad_close` ✅ | `mlp_hidden_float_sgd_descends` ✅ | ✅ CLOSED (Step 1, 2026-06-25) |
| **MLP input `W₀`** | `mlp_w0_grad_close` ✅ | `mlp_input_float_sgd_descends` ✅ | ✅ CLOSED (Step 2, 2026-06-25) |
| **CNN conv `W₁/W₂`** | `cnn_conv*_grad_close` ✅ | `cnn_conv*_float_sgd_descends` ✅ | ✅ CLOSED (Step 3, Increments 1–4) |
| **CNN conv `b₁/b₂` bias** | `cnn_conv*_bias_grad_close` ✅ | `cnn_conv*_bias_float_sgd_descends` ✅ | ✅ CLOSED (Step 3, Increment 5) |
| **deep nets / joint step** | — | — | OUT OF SCOPE (honest stop, below) |

**Step 3 is fully closed**: every conv weight AND bias of the Chapter-4 CNN is a
float-faithful descent step. With the MLP (all three layers) already done, the §3
descent program is complete for the deployed MLP and CNN — only the honest-stop
line (joint all-layers step, deep nets) remains, by design.

The **abstract-η smoothness side is fully proven** everywhere it matters:
`mlp_{output,hidden,input}_sgd_descends` (explicit constants `2d₃w₂²a²/(1−2w₂aD)`,
`2d₃d₂²w₁²w₂²a²/(1−2d₂w₁w₂aD)`) and the CNN per-conv ingredients
(`cnn_conv{1,2}_sgd_descends` + bias, max-pool selection margins, conv-kernel
drift). What's missing is only the **float-fusion** — feeding a *proven* float
budget into the `η`-slot those theorems already accept abstractly.

## Step 1 — wire the MLP hidden rung — ✅ DONE (2026-06-25)

**Landed** in `SgdDescentMlp.lean` (3-axiom clean, audited):
- `FloatModel.mlpHiddenFloatGrad` (+ `mlpHiddenFloatGrad_apply`) — the *actual*
  binary32 `W₁` gradient the trainer computes, `fl(a₀ᵢ·c̃₁ⱼ)`, flattened to the
  `Vec (d₁*d₂)` layout (the hidden-layer peer of `linearFloatGrad`).
- `mlp_hidden_loss_gradAt_reluMask` — the bridge from `mlp_hidden_loss_gradAt`'s
  `gradAt` closed form (`a₀ᵢ·relu'(z₁ⱼ)·∑ₖ W₂ⱼₖ·(softmax−onehot)ₖ`) to the
  `reluMask`/masked-`W₂ᵀ` form that `mlp_w1_grad_close` bounds against, at the
  off-kink point (`dense W₁ b₁ a₀ k ≠ 0`, supplied by the rounding margin since
  `layerBudget ≥ 0`). One `congr` + a ReLU-sign case split + `mul_comm`.
- `mlp_hidden_float_sgd_descends` — discharges `mlp_hidden_sgd_descends`' abstract
  `η` with the proven `mlp_w1_grad_close` budget (via the bridge): **one binary32
  hidden-layer SGD step provably decreases the CE loss, no abstract
  gradient-accuracy parameter.** Both margins carried as hypotheses (the honest
  first cut, as planned): the **rounding** margin `hmargin_round`
  (`layerBudget < |z₁|`, the grad-close precondition) and the **step** margin
  `hmargin_step` (`a·D < |z₁|`, the smoothness precondition). Collapsing one into
  the other (rounding-at-operating-point ⇒ step-at-radius-`D`) is left as a
  refinement; same shape ("nothing flips the layer-1 ReLU"). Mirrors
  `mlp_output_float_sgd_descends` / `linear_float_sgd_descends`.

The original plan is below for reference.

`mlp_w1_grad_close` (SgdDescentMlp.lean:1433) already proves the binary32 `W₁`
gradient `fl(a₀ᵢ · c̃₁ⱼ)` is within `mulErr u a … 0 (layerBudget … (cotErr …))` of
the certified `a₀ᵢ · mask(z₁, W₂ᵀ·(softmax−onehot))ⱼ` (= `mlp_hidden_loss_gradAt`),
**under the rounding margin** `hmargin : layerBudget u d₁ w₁ β₁ a 0 < |dense W₁ b₁ a₀ j'|`
(forward rounding must not flip the layer-1 ReLU). The grad-close is done; only the
descent statement is missing.

**Task:** prove `mlp_hidden_float_sgd_descends`, the exact analogue of
`mlp_output_float_sgd_descends` (SgdDescentMlp.lean:1361) — which is a 3-line call
to `linear_float_sgd_descends`. Here instead:
- instantiate `mlp_hidden_sgd_descends`' abstract gradient `gh := ` the inline
  float `W₁` gradient (`fun idx => M.mul (a₀ i) (reluMask … )`, the LHS of
  `mlp_w1_grad_close`), and `η := mulErr u a … 0 (layerBudget … (cotErr …))`;
- discharge its `hgh : |gh idx − gradAt … idx| ≤ η` hypothesis with
  `mlp_w1_grad_close` (per-entry; sum/flatten to the `gradAt` indexing as the
  linear rung does);
- **reconcile the two margins.** `mlp_w1_grad_close` carries the *rounding* margin
  `layerBudget < |z₁|`; `mlp_hidden_sgd_descends` carries the *step* margin
  `a·D < |z₁|` (mask freezes along the segment). They are the same shape ("nothing
  flips the layer-1 ReLU"). Either show the rounding margin at the operating point
  implies the step margin at radius `D`, or carry both as hypotheses (the honest,
  lower-risk first cut — the linear rung already carries several hypotheses).

Effort: **small** — the grad-close (the real work) is done; this is wiring +
margin bookkeeping. Highest value/effort in the whole pass.

## Step 2 — the MLP input rung — ✅ DONE (2026-06-25)

**Landed** in `SgdDescentMlp.lean` (3-axiom clean, audited), the mechanical twin of
Step 1 one mask deeper:
- `reluMask_dense_transpose_eq` — the reusable per-step identity
  `relu'(zₗ)·∑ₖ Wₗₖ·cₖ = reluMask z (Wᵀ·c) l` (one ReLU-sign case split + `mul_comm`);
  factors the masked-contraction ↔ if-then-else bridge step.
- `FloatModel.mlpInputFloatGrad` (+ `mlpInputFloatGrad_apply`) — the actual binary32
  `W₀` gradient, head run back through **two** masked `Wᵀ` contractions.
- `mlp_input_loss_gradAt_reluMask` — the nested bridge (`mlp_input_loss_gradAt`'s
  two-mask if-then-else form ⇒ the nested `reluMask` form), discharged by
  `simp_rw [reluMask_dense_transpose_eq]` (fires inner W₂ᵀ then outer W₁ᵀ).
- `mlp_w0_grad_close` — the grad-close: `mlp_w1_grad_close`'s shape **plus one extra
  `cot_step_close`** (the second masked contraction), forward chain one layer deeper,
  under **two** rounding margins (`E₀ < |z₀|`, `E₁ < |z₁|`). Budget
  `mulErr u a (layerAct…) 0 (layerBudget … (layerBudget … (cotErr …)))`.
- `mlp_input_float_sgd_descends` — discharges `mlp_input_sgd_descends`' abstract η.
  Carries **four** margins (two rounding + two step) as the honest first cut. With
  this, **all three MLP weight layers are float-fused descent** — "one binary32 SGD
  step on any single MLP weight layer provably decreases the loss" is a closed
  statement.

The original plan is below for reference.

`mlp_w0_grad_close` then `mlp_input_float_sgd_descends`. The float `W₀` gradient
runs back through **two** ReLU masks + the `W₁`,`W₂` cotangent fan-in, so it needs
**two** rounding margins and picks up the `ℓ1→ℓ1` factor `d₂·w₁` (already in
`mlp_input_sgd_descends`' Lipschitz constant). `mlp_w0_grad_close` is built like
`mlp_w1_grad_close` from the same reusable closes — head (`softmax_ce_cot_close`,
`cotErr`), one extra masked `Wᵀ` contraction (`cot_step_close` under the second
margin), final exact-operand multiply (`mul_close`, `ea = 0`). Then wire as Step 1.
Effort: **medium** (two margins, deeper cotangent chain; no new machinery).

## Step 3 — CNN float fusion (stretch)  → full cold-start plan: `planning/floatbridge_descent_cnn.md`

`cnn_conv2_grad_close` (then `cnn_conv1_grad_close`) reusing the conv weight-grad
bridges, wired into the existing abstract-η `cnn_conv{1,2}_sgd_descends`. The float
backward runs through the dense head **and** the max-pool selection (frozen by the
existing `MaxPool2MarginQ.isArgmax_iff` under a rounding margin) **and** a ReLU mask
before the conv-weight correlation. The single **largest** rung; scoped into
independently-audited increments (pool-back primitive + bridge → grad-close → wiring
→ conv1 → biases) in **`planning/floatbridge_descent_cnn.md`** — the abstract-η
descent side is already done (`planning/sgd_descent_cnn.md`). Effort: **high**; its
own focused session, after Steps 1–2 (done).

## The honest stop line (do NOT cross)

- **Joint all-layers step** — when every parameter moves at once the logits are no
  longer affine in the moving parameters, so the segment-Lipschitz route breaks.
  Open by design; the per-layer rungs are the honest unit of the descent claim.
- **Deep nets (ViT / ConvNeXt / r34 / enet / mnv2)** — descent needs a
  loss-gradient smoothness (Lipschitz) constant; at depth it is brutal (compounding
  per-layer operator norms ⇒ vanishing admissible `lr`), and there is **no**
  `*_sgd_descends` theorem for any of them (verified). Bank *closeness* there and
  state it honestly — exactly the §5 honesty-pass flag F1: "loss-descent **step**"
  (the certified update, proven for all nets via the §1a ties) is **not** "the loss
  provably **decreases**." Don't let the new MLP rungs make "descent" read as
  net-wide.

## Reusable template + gotchas

- **Master/η pattern:** copy `mlp_output_float_sgd_descends`' body shape; the only
  per-rung change is the η source (a grad-close theorem) and the margin count.
- **grad-close recipe** (`mlp_w1_grad_close` is the model): head close
  (`softmax_ce_cot_close`, accuracy `cotErr`; `cotErr_nonneg` exists) → masked `Wᵀ`
  contraction (`cot_step_close`, under the quantitative margin) → input multiply
  (`mul_close`, exact left operand `ea = 0`).
- **Lean gotchas** (from prior float-chain work): `nlinarith` ring-normalizes the
  whole context → `npow` blowup; keep hypothesis contexts small / clear unused.
  Use `open FloatModel in` **before** the docstring for Proofs-namespace lemmas
  that mention `layerBudget`/`reluMask`/`cotErr`. Combining-tilde identifiers
  (x̃) are invalid Lean idents — use `zt`/`ct`/`gt`.
- **Wiring:** `SgdDescentMlp` is already a lakefile Proofs root; each new theorem
  just needs a `#print axioms` line in `tests/AuditAxioms.lean` (keep 3-axiom
  clean) and the coverage check passes as-is.

## Suggested order

1. ~~**Step 1** (hidden wiring)~~ ✅ DONE — `mlp_hidden_float_sgd_descends`.
2. ~~**Step 2** (input rung)~~ ✅ DONE — `mlp_w0_grad_close` + `mlp_input_float_sgd_descends`;
   the **MLP is complete** — "one binary32 SGD step on any single MLP weight layer
   provably decreases the loss" is now a clean closed statement for all three layers.
3. ~~**Step 3** (CNN)~~ ✅ DONE — `cnn_conv{1,2}_float_sgd_descends` (weights, Increments 1–4)
   + `cnn_conv{1,2}_bias_float_sgd_descends` (biases, Increment 5). The CNN deployed-descent
   headline is closed; see `planning/floatbridge_descent_cnn.md` for the full increment log.
4. **NEXT:** Re-run the §5 honesty pass (`planning/floatbridge_honesty_pass.md`) —
   keep the per-layer vs net-wide / step vs decreases lines exactly honest. The new CNN bias
   rungs do **not** make "descent" read net-wide (the per-param rung is still the honest unit).
