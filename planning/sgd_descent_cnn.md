# SGD descent through the CNN — remaining assembly

Status (this commit): the conv2-layer rung is DONE — `cnn_conv2_sgd_descends`
(LeanMlir/Proofs/SgdDescentCnn.lean) proves one inexact SGD step on the
second conv kernel decreases the CE loss, under the four margins at the
step radius. 3-axiom clean (tests/AuditAxioms.lean). Remaining: conv1 and
the biases.

## What's done

1. **Pool selection margin** — `MaxPool2MarginQ δ x` (every two cells of
   every 2×2 window differ by more than `2δ`), the quantitative form of
   `MaxPool2Smooth`. Under drift ≤ δ per entry:
   - `smooth_of_close`: no window ties anywhere near the point;
   - `isArgmax_iff`: every window's argmax is FROZEN;
   - `pdiv3_eq`: the pool's entire `pdiv3` routing pattern is frozen —
     the pool is a fixed linear selector along the whole step segment.
2. **Pool drift transport** — `maxPoolFlat_entry_lipschitz` (1-Lipschitz
   per entry, for the margins above the pool) and `maxPoolFlat_l1_contract`
   (`ℓ1`-contractive, for the budgets: windows partition the input via
   `sum_window_cells`/`winRowEquiv`).
3. **Conv kernel drift** — `conv2d` is affine in the kernel
   (`conv2d_kernel_sub` via `convPad`); per-entry drift
   `≤ a·(slab-o ℓ1) ≤ a·‖e‖₁` (`conv2d_kernel_drift`,
   `conv2d_kernel_drift_total`); `ℓ1` drift over all outputs
   `≤ (h·w)·a·‖e‖₁` (`conv2d_kernel_drift_sum`) — the spatial multiplicity
   is the price of weight sharing. Index plumbing: `t3Idx`/`k4Idx`,
   `sum_t3`, `sum_abs_k4`, `sum_abs_kernel_slab_le`.
4. **Dense head below the pool = free.** Loss-of-`W₅/W₄/W₃` are literal
   instances of `linear_sgd_descends` / `mlp_hidden_sgd_descends` /
   `mlp_input_sgd_descends` at `x := maxPoolFlat (…)` — the MLP theorems
   are generic in the fixed activation vector. Thin wrappers optional.
5. **The conv2-layer rung (NEW this commit)** — exactly the 5-step plan:
   - `ce_head3_input_grad`: one `pdiv_comp` hop (peel `dense W₃`) on
     `ce_head2_input_grad`; NO leading mask (the pool feeds `d₃` direct).
   - `pool_relu_input_grad`: the key glue. `pdiv` through relu (mask)
     then `maxPoolFlat`; the pool `pdiv` IS `pdiv3 maxPool2` after
     `unfold pdiv3; rw [Tensor3.flatten_unflatten]; rfl`, and
     `pdiv3_maxPool2_smooth` + a `Finset.sum_eq_single` chain collapse
     the pooled sum to the single argmax term.
   - `conv2d_weight_pdiv`: the conv weight-map Jacobian closed form
     (`if co = o then convPad … else 0`), extracted from the CERTIFIED
     VJP by contracting `conv_weight_grad_bridge` against `basisVec` —
     no re-derivation of the 200-line foundation proof. Point-free
     (conv is affine in the kernel), so along a segment only the head
     gradient moves. Row mass: `conv2d_weight_pdiv_row_l1 ≤ (h·w)·a`.
   - `cnn_conv2_loss_gradAt`: the EXISTING fold
     (`conv_total_loss_grad_fold`, generic in `G`) + `sum_t3` + the two
     pieces above.
   - `cnn_conv2_loss_grad_lipschitz`: margins freeze relu₂
     (`cnn_margin2_keeps_offkink`), the pool routing
     (`cnn_postrelu_close_seg` + `isArgmax_iff` — the margin/closeness
     stated on the POST-relu tensor), relu₃/relu₄
     (`cnn_margin{3,4}_keeps_offkink` via the drift chain
     `cnn_pool_l1_drift → cnn_z3_drift → cnn_z4_drift →
     cnn_conv2_logit_drift`); the difference collapses to the softmax
     drift (`head3_sum_drift`, masks generic 0/1-valued); constant
     `C₂ = 2·nC·(4hw)²·d₃²·d₄²·w₃²·w₄²·w₅²·a²/(1−2δ̄)` with
     `δ̄ = w₅·d₄·w₄·d₃·w₃·(4hw)·a·D`, assembled by `ring`.
   - `cnn_conv2_sgd_descends`: assembled via `sgd_descends` exactly as
     `mlp_input_sgd_descends` (margins at the step radius
     `D = lr·(‖∇f₂‖₁ + |kernel|·η)`).

## Remaining: conv1, biases

1. **conv1** — one more conv+relu crossing; the input-side conv drift
   needs the conv-as-function-of-INPUT `ℓ1→ℓ1` bound (factor
   `ic·kH·kW·w₂ᶜ`-shaped, entrywise kernel bound `w₂ᶜ`) — a sibling of
   `conv2d_kernel_drift` with the roles of kernel and input swapped.
   Margins: relu₁ + everything in the conv2 list. The gradAt closed form
   needs the conv2-as-INPUT `pdiv` (certified `conv2d_has_vjp3` /
   `conv2d_input_grad_formula`) in place of the frozen-activation step.
2. **Biases** — `conv_bias_total_loss_grad_fold` already exists
   (ConvLossFold.lean); the bias-map is affine with Jacobian a Kronecker
   indicator over the slab, drift `≤ ‖e‖₁` per entry (no `a` factor,
   no spatial multiplicity on the per-entry side). Same argument,
   strictly easier than the kernel.

## Gotchas encountered (don't rediscover)

- `Fintype.sum_prod_type` takes `f` EXPLICITLY on this pin; `.symm` in
  calc needs the lambda spelled out (HO unification fails on `?f (x, y)`).
- `split_ifs` can't see through `let`-laden `dite`s — `convPad` is
  deliberately let-free; `conv2d_eq_convPad` is still `rfl` (zeta).
- `simp only [iff-lemma]` inside `ite` conditions can report "no
  progress" — case-split on the prop and `simp [hA, hAy]` instead
  (see `MaxPool2MarginQ.pdiv3_eq`).
- The pool margin/`MaxPool2Smooth` must be stated on the POST-relu tensor
  (the net pools `relu (flatConv …)`).
- `simp_rw [pdiv_relu, ite_mul, zero_mul]` (the mask-collapse idiom)
  distributes ite-masks EVERYWHERE in the goal, including the statement's
  own RHS — re-normalize at the end with
  `simp only [ite_mul, one_mul, zero_mul]` (see `pool_relu_input_grad`).
- Unfolding the certified conv backward: `simp only
  [conv2d_weight_grad_has_vjp, k4Idx, Equiv.symm_apply_apply,
  basisVec_apply, convPad]` leaves RAW `finProdFinEquiv` encodings in the
  ite conditions; fold them back with `t3Idx_def` in a SECOND simp pass
  (folding in the same pass can race the kernel-side `symm_apply_apply`).
- Mask-`≤ 1` side goals on lambda-applied ites need `dsimp only` (beta)
  before `split_ifs`.
- `open Classical` at file level (CNN.lean convention) for `if`s over
  `MaxPool2IsArgmax`-style Props.
