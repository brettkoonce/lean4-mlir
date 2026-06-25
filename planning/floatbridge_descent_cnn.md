# Float bridge §3 — the CNN descent rung (cold-start handoff)

The last open rung of the descent program (`planning/floatbridge_descent_pass.md`
Step 3). The **linear net and the whole MLP are float-fused descent**
(`linear_float_sgd_descends`, `mlp_{output,hidden,input}_float_sgd_descends`, committed
`39f05f9`); this doc is the cold-start plan to do the same for the **Chapter-4 MNIST
CNN's two conv kernels**, then (optionally) the conv biases.

The whole pattern is unchanged: **one abstract-η descent theorem + a per-rung float
grad-close as the proven η source.** The descent side is *already done* — the abstract-η
`cnn_conv{1,2}_sgd_descends` (+ bias variants) are proven (`SgdDescentCnn.lean`,
`planning/sgd_descent_cnn.md`). What's missing is the **float fusion**: feeding a *proven*
float gradient budget into the η-slot those theorems already accept abstractly.

This is the single **largest** rung — the float backward runs through the dense head **and**
the max-pool selection **and** a ReLU mask before the conv-weight correlation, and the wiring
hypotheses are enormous. Budget it as its own focused effort and land it in the increments
below, each independently audited (3-axiom clean).

## State

| rung | grad-close (η source) | descent wiring (abstract η ✅) | status |
|---|---|---|---|
| linear / MLP (all 3 layers) | ✅ | ✅ | ✅ CLOSED (committed 39f05f9) |
| **CNN conv2 `W₂`** | `cnn_conv2_grad_close` ✅ | `cnn_conv2_float_sgd_descends` ✅ | ✅ CLOSED (Increments 1–3) |
| **CNN conv1 `W₁`** | `cnn_conv1_grad_close` ❌ | `cnn_conv1_sgd_descends` ✅ | OPEN (Increment 4 — one layer deeper) |
| CNN conv2/1 **bias** | `cnn_conv{2,1}_bias_grad_close` ❌ | `cnn_conv{2,1}_bias_sgd_descends` ✅ | OPEN (strictly easier; optional) |
| deep nets / joint step | — | — | OUT OF SCOPE (honest stop) |

## The certified target (what the float gradient must approximate)

`cnn_conv2_loss_gradAt` (SgdDescentCnn.lean:1545) gives the closed form of
`gradAt (loss-of-W₂) (Kernel4.flatten W₂) (k4Idx o cc kh kw)`. Net is
`conv2 → relu → maxPoolFlat → dense W₃ → relu d₃ → dense W₄ → relu d₄ → dense W₅ → CE`.
The gradient is a spatial correlation of the conv input with the **conv-output cotangent**:

```
∑ ci hi wi,
  (if ci = o then convPad kH kW x₁ cc kh kw hi wi else 0)        -- conv weight Jacobian (k4Idx)
  · (if z_conv(ci,hi,wi) > 0 then 1 else 0)                       -- relu mask at conv output
  · (if MaxPool2IsArgmax(relu post-conv) ci hi wi                 -- pool argmax selection
     then ∑ l, W₃(t3Idx ci (winRow hi)(winCol wi)) l              -- W₃ᵀ contraction (at the pooled idx)
       · (if z₃ l > 0 then 1 else 0)                              -- relu d₃ mask
       · ∑ q, W₄ l q · (if z₄ q > 0 then 1 else 0)                -- relu d₄ mask
            · ∑ k, W₅ q k · (softmax − onehot)                    -- W₅ᵀ + head
     else 0)
```

So the **conv-output cotangent** is
`c_conv = reluMask(z_conv) · poolSelect · W₃ᵀ · reluMask(z₃) · W₄ᵀ · reluMask(z₄) · W₅ᵀ · (softmax−onehot)`,
and the gradient is `∑_{spatial} convPad · c_conv` (the `if ci=o` picks the output channel).
The **float** gradient rounds every step of this and uses the **float** forward values.

The abstract-η slot to discharge — `cnn_conv2_sgd_descends` (SgdDescentCnn.lean:2458),
hypothesis `hgh`:
```
hgh : ∀ idx, |gh idx − gradAt (loss-of-W₂) (Kernel4.flatten W₂) idx| ≤ η
```
`cnn_conv2_grad_close` supplies `gh := the float conv2 gradient` and a *proven* `η`.

## Reuse inventory (most pieces exist)

**Float forward (the whole chain is plumbing, no new math):**
- `FloatModel.cnn_float_close` (SgdDescentCnn.lean:695) — whole-net forward budget; copy its
  per-layer `set A_/E_` skeleton for the intermediate float values you need
  (`z̃_conv`, `z̃₃`, `z̃₄`, `z̃₅`).
- `FloatModel.convF` / `flatConvF` / `flatConvF_close` / `flatConv_abs_le` (≈571–660) — conv
  forward = dense at fan-in `ic·kH·kW` (`conv2d_eq_dense`); threads like a dense layer.
- `maxPoolFlat_close` (CNN.lean:2747, `|poolF − pool| ≤ e` pass-through) + `maxPoolFlat_abs_le`
  (2773) — pool forward is error-transparent (max rounds nothing).
- `dense_close`/`dense_close_fresh`/`denseErr_le_uniform`, `relu_close`/`relu_abs_le`/
  `dense_abs_le` (FloatBridge) — exactly as in `mlp_w0_grad_close`.

**Backward (head + masked-Wᵀ — identical to the MLP):**
- `softmax_ce_cot_close` (head cotangent within `cotErr`), `cotErr_nonneg`.
- `cot_step_close` (FloatBridge:1019) — `reluMask z (Wᵀ·c)` close under a forward rounding
  margin `ez < |z|`; reuse **2×** (W₅ᵀ under z̃₄ mask, W₄ᵀ under z̃₃ mask), exactly as the
  MLP input rung used it twice.
- `reluMask_close` (FloatBridge:990) — `|reluMask zt vt − reluMask z v| ≤ ev` under the
  forward sign-freeze margin `ez < |z|`; this is the **relu mask at the conv output**.

**Pool selection frozen under the rounding margin (the keystone is nearly free):**
- `MaxPool2MarginQ δ x` (SgdDescentCnn.lean:286) — every two cells of every 2×2 window differ
  by `> 2δ`. Instantiate `δ := the forward rounding budget at the post-relu conv output`,
  `x := the real post-relu tensor`.
- `MaxPool2MarginQ.isArgmax_iff` (311): `(hm) (hclose : |y − x| ≤ δ) → (IsArgmax y ↔ IsArgmax x)`.
  With `y := float post-relu`, `x := real post-relu`, this gives **float argmax = real argmax**
  — the pool routing is frozen by rounding, for free.

**Conv weight gradient = dot (the final correlation, float-close exists):**
- `convWeightGrad_eq_dot` (1376): the certified conv weight grad `= ∑ convPad · cot`.
- `FloatModel.cnn_convW_step_float_close` (1400) + `dot_close`/`dotSgd_step_close`: the rounded
  conv-weight dot within Higham γ (fan-in `h·w`) — but it **supplies** the cotangent. Here you
  feed the **proven** float cotangent `c_conv` (with its accuracy `ecot`), so use `dot_close`
  directly with the cotangent-error term, not the step form.
- `conv2d_weight_pdiv` (1246), `convPad`/`convPadWin`/`cotWin`, `t3Idx`/`k4Idx` — the index
  plumbing that `cnn_conv2_loss_gradAt` already uses; match it exactly.

## What is genuinely new

1. **Float pool-backward close** (small — falls out of `isArgmax_iff`). The pool-backward in
   the gradient is `if MaxPool2IsArgmax(post-relu) … then <pooled cotangent> else 0`. The float
   version reads the **float** post-relu tensor; `isArgmax_iff` (at `δ = conv-output rounding
   budget`) makes the indicator equal to the real one, so the difference is
   `indicator · (c̃_pooled − c_pooled)`, magnitude `≤ |c̃_pooled − c_pooled|` (indicator ∈ {0,1})
   — a `reluMask_close`-style pass-through. State it to match the certified form's
   `t3Idx ci (winRow hi)(winCol wi)` pooled-index access.
2. **`cnn_conv2_loss_gradAt_reluMask`** — the bridge from the 4-deep nested if-then-else form
   (`cnn_conv2_loss_gradAt`) to the `reluMask`/`poolSelect`/masked-`Wᵀ` form the grad-close
   bounds against. The MLP analogue is `mlp_input_loss_gradAt_reluMask` + the reusable
   `reluMask_dense_transpose_eq` (use it for the W₅ᵀ/W₄ᵀ/W₃ᵀ steps); the pool + conv-channel
   (`if ci=o`) selectors are the new parts. **This is the fiddly part** — heavy `t3Idx`/`k4Idx`/
   `convPad`/`winRow`/`winCol` plumbing (see gotchas).

## Increment plan (land each independently, 3-axiom clean + audited)

**Increment 1 — keystone primitives. ✅ DONE (3-axiom clean, audited).**
- The float pool-backward close lemma (new #1 above) = `MaxPool2MarginQ.poolBack_close`
  (SgdDescentCnn.lean, just after `pdiv3_eq`).
- `cnn_conv2_loss_gradAt_reluMask` bridge (new #2, just after `cnn_conv2_loss_gradAt`),
  built on two helpers: `head3_cot_reluMask` (the 3-dense head restated to
  `dense W₃ᵀ 0 (mask z₃ (dense W₄ᵀ 0 (mask z₄ (dense W₅ᵀ 0 (softmax−onehot))))`) and
  `dense_transpose_eq` (the unmasked peer of `reluMask_dense_transpose_eq`, for the W₃ rung).
  **Gotcha learned:** `dense_transpose_eq` is generic in `(W, c)`, so a blanket
  `simp_rw [dense_transpose_eq]` ALSO collapses the spatial `∑ convPad·cot` sum (matched as
  `∑ k W l k · c k`). Fix: confine the W₃ collapse to the head-local `head3_cot_reluMask`
  (no spatial sum present), and only THEN package via `convWeightGrad_eq_dot`. The bridge
  keeps the conv-output ReLU mask `𝟙[z₂>0]` and the pool argmax selector explicit — their
  float closeness is `reluMask_close` / `poolBack_close` in Increment 2.
- All 4 new theorems added to `tests/AuditAxioms.lean`; `lake build …SgdDescentCnn` clean,
  `check_audit_coverage.py` passes. *(Effort was medium-high, as estimated.)*
Land + audit before touching the grad-close. *Effort: medium-high (the bridge plumbing).*

**Increment 2 — `cnn_conv2_grad_close`. ✅ DONE (3-axiom clean, audited).**
Mirrors `mlp_w0_grad_close`, deeper by the pool + the dot. Landed:
- `FloatModel.cnnConv2FloatGrad` (+ `_apply`) — the rendered trainer's `W₂` gradient =
  `M.dot (convPadWin x₁) (cotWin c̃Conv o)`, the float conv-output cotangent rounding every step.
- `FloatModel.cnnConv2GradBudget` — the closed-form `η` as a `let`-nest **def** (NOT inlined —
  the deep nest is far less error-prone named once; the final `exact dot_perturbed_close` closes
  by defeq after `simp only [cnnConv2GradBudget]`).
- `cnn_conv2_grad_close` — `|cnnConv2FloatGrad … (k4Idx o cc kh kw) − gradAt …| ≤ cnnConv2GradBudget`.
  States against `gradAt` directly and applies the bridge **inside** (cleaner than the MLP's
  separate-bridge split). Chain exactly as planned: forward (`convF_close`→`dense_close`×3, relu/pool
  error-transparent) → head `softmax_ce_cot_close` → `cot_step_close`×2 (W₅/W₄) → unmasked W₃
  `dense_close` → `poolBack_close` (Inc 1) → `mask_scalar_close` (conv ReLU mask) → `dot_perturbed_close`.
- Two **reusable cores** + one helper, all generic & landed first: `mask_scalar_close` (scalar
  `𝟙[z>0]·x` peer of `reluMask_close`), `FloatModel.dot_perturbed_close` (float dot vs perturbed
  cotangent: `dot_close` Higham γ on `A·B̃` + per-entry drift `∑|A|·eB`), `t3Idx_surj` (lift per-cell
  conv bounds to `∀ k`).
- **Four margins carried** as hypotheses: conv-output `Econv`, pool `Econv` (POST-relu,
  `MaxPool2MarginQ`), z̃₃ `E₃`, z̃₄ `E₄`. Conv-2 input `x₁` is **exact** (the MLP `x`-exact pattern).
- **Gotchas learned:** the giant float-cotangent def is paren-fragile (the deepest softmax needs 9
  closes through `M.dense W₅ b₅` before `label`, not 8 — use the `awk … tr -cd '()'` per-line cumulative
  check); `softmax_nonneg`/`softmax_le_one` are **private** in FloatBridge — unfold softmax = `exp/∑exp`
  (the MLP `hC2` route) instead; derive the bridge's `hz2/hz3/hz4` off-kink hyps from the margins via
  `abs_pos.mp (lt_of_le_of_lt E_nn (hmargin …))` (NOT `rw [hl] at this` — `set` hides the expr behind
  its name so the rewrite pattern is gone). All 7 decls in `tests/AuditAxioms.lean`; build + coverage clean.
- *(Effort was high, ≈260 lines for the theorem, as estimated.)*

**Increment 3 — `cnn_conv2_float_sgd_descends`. ✅ DONE (3-axiom clean, audited).**
The wiring, mirroring `mlp_input_float_sgd_descends`: prove `0 ≤ cnnConv2GradBudget`, derive the
flattened-kernel bound (`hv2` via `flatten_k4Idx` + `k4Idx_surj`), discharge `hgh` per kernel
entry (`k4Idx_surj` → `cnn_conv2_grad_close`, which already folds the bridge in), then
`exact cnn_conv2_sgd_descends … hη0 hgh hm2 hmq hm3 hm4 hsmall h1 h2`. Closes **"one binary32
conv2-kernel SGD step provably decreases the cross-entropy loss"** with the gradient accuracy
*proven*, not assumed. Plus two index helpers: `flatten_k4Idx` / `k4Idx_surj` (the `k4Idx` peers of
`flatten_t3Idx` / `t3Idx_surj`).
- **The doc's paren-minefield warning did NOT materialize** — because Increment 2 made the η a
  **def** (`cnnConv2GradBudget`), the carried `hsmall`/`h1`/`h2`/step-margins reference η as a
  compact `M.cnnConv2GradBudget c h w … eexp` term (≈1.5 lines), NOT the 60-line inlined nest. The
  whole wiring is straight transcription of `cnn_conv2_sgd_descends`'s hyps with `η → the budget term`.
- **Two margin families carried** (the honest first cut): per-layer ROUND margins
  (`hmarginConv/Pool/3/4`, about `conv2d W₂ b₂ x₁`, fed to the grad-close via `Kernel4.unflatten_flatten`)
  + the gradient-radius STEP margins `hm2/hmq/hm3/hm4` + `hsmall`/`h1`/`h2` (fed straight to
  `cnn_conv2_sgd_descends`). **Gotchas:** the abstract descent uses `Kernel4.flatten W₂` as the param
  point, so the round margins (stated about `conv2d W₂ …`) are converted to the
  `conv2d (Kernel4.unflatten (Kernel4.flatten W₂)) …` form via `rw [Kernel4.unflatten_flatten]` when
  passed; `0 ≤ budget` proved by `simp only [cnnConv2GradBudget]` + `add_nonneg`/`mul_nonneg` on the
  component `layerAct`/`layerBudget`/`cotErr` nonnegs (positivity can't see the smRho condition);
  `layerAct_nonneg`/`layerBudget_nonneg` live in `FloatModel` → `open FloatModel in` (before the
  doc-comment, not after — the `open … in` must precede the `/-- -/`). All 3 decls in
  `tests/AuditAxioms.lean`; build + coverage clean. *Effort: medium (mostly transcription).*

**Increment 4 — conv1** (`cnn_conv1_grad_close` → `cnn_conv1_float_sgd_descends`). One layer
deeper: the cotangent runs back through conv2-as-input as well (locality, not spatial count —
`convTap`, `cnn1_*` machinery; see `planning/sgd_descent_cnn.md`). Reuse Increment 1–3 wholesale.
*Effort: high.*

**Increment 5 (optional) — the conv biases.** Strictly easier (bias is affine, no `a`, no kernel
mass; `conv2d_bias_pdiv` is the Kronecker channel indicator). Same chain with `a·‖e‖₁ ↦ ‖e‖₁`.

## The MLP template to mirror (exact names, committed 39f05f9)

- `reluMask_dense_transpose_eq` (SgdDescentMlp.lean) — reusable per-step `relu'(z)·∑W·c = reluMask z (Wᵀ·c)`.
- `FloatModel.mlpInputFloatGrad` (+ `mlpInputFloatGrad_apply`) — the float-gradient def pattern.
- `mlp_input_loss_gradAt_reluMask` — the `simp_rw [reluMask_dense_transpose_eq]` bridge pattern.
- `mlp_w0_grad_close` — the grad-close: forward chain → head → `cot_step_close`×2 → `mul_close`.
  (CNN swaps the final `mul_close` for `dot_close`, and inserts pool-back + a third Wᵀ + relu-mask.)
- `mlp_input_float_sgd_descends` — the wiring: `set η`, `0 ≤ η` by hand, off-kink from margins,
  `hgh` via grad-close+bridge, `exact …_sgd_descends`.

## Gotchas (don't rediscover — from the MLP rungs + `planning/sgd_descent_cnn.md`)

- **Paren-fragile giant budgets.** Each `η`→`mulErr`/`layerBudget` substitution in `hsmall`/`h1`/
  `h2`/margins needs the budget's internal closes **plus** the original post-`η` closes. Verify
  *per binder*: `awk 'NR>=a&&NR<=b' file | tr -cd '()' | awk '{print gsub(/\(/,"")-gsub(/\)/,"")}'`
  must be 0. (Two MLP binders, and *four* prior CNN sessions, got it wrong by one.)
- **Namespace.** `open FloatModel in` + `Proofs.dense` for helper-heavy proofs (the grad-close);
  bare `dense` + inline-qualified `FloatModel.*` for statements that must match the abstract
  theorem (the wiring). `M.foo` for methods either way.
- **Tensor-index plumbing** (the CNN-specific minefield): `t3Idx`/`k4Idx` leave raw
  `finProdFinEquiv` encodings after `simp only [conv2d_weight_grad_has_vjp, …]` — fold back with
  `t3Idx_def` in a **second** simp pass. `split_ifs` can't see through `let`-laden `dite`
  (`convPad` is deliberately let-free). The pool margin must be stated on the **post-relu**
  tensor. `open Classical` is at file scope (CNN.lean convention).
- **`simp_rw [pdiv_relu, ite_mul, zero_mul]`** distributes ite-masks into the goal's *own RHS* —
  re-normalize with `simp only [ite_mul, one_mul, zero_mul]` at the end (see `pool_relu_input_grad`).
- **`nlinarith` ring-normalizes the whole context** → `npow` blowup with concrete `(1+u)^k`; keep
  hypothesis contexts small, push `u` to a literal before the final arithmetic (`gamma_num`,
  `sgdErr_mono`), use `linarith` not `nlinarith` once `set s := a*g`.

## Honest stop line (do NOT cross)

- **Joint all-layers step** — logits not affine in all params at once; the per-layer rungs are
  the unit. Open by design.
- **Deep nets (ViT/ConvNeXt/r34/enet/mnv2)** — no `*_sgd_descends` exists for any (verified);
  descent needs a loss-gradient Lipschitz constant brutal at depth. Bank *closeness* there and
  state it honestly (honesty-pass flag F1: "loss-descent **step**" ≠ "the loss **decreases**").
  The CNN rungs do **not** make "descent" read as net-wide.

## Suggested order

1. **Increment 1** (pool-back primitive + bridge) — keystone, unblocks everything.
2. **Increment 2** (`cnn_conv2_grad_close`).
3. **Increment 3** (`cnn_conv2_float_sgd_descends`) — then "one binary32 conv2-kernel SGD step
   provably decreases the loss" is closed.
4. **Increment 4** (conv1), **5** (biases) — only if the full deployed-CNN-descent headline is wanted.
5. Re-run the §5 honesty pass (`planning/floatbridge_honesty_pass.md`) and update
   `planning/floatbridge_descent_pass.md` / `floatbridge_certificate_gaps.md` after each.

**Definition of done (per increment):** every new theorem `#print axioms`-closes under
`[propext, Classical.choice, Quot.sound]`, added to `tests/AuditAxioms.lean`; no `sorry`, no
project axiom; `python3 scripts/check_audit_coverage.py` passes; `lake build
LeanMlir.Proofs.SgdDescentCnn` clean.
