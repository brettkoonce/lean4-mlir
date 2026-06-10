# SGD descent through the CNN — remaining assembly

Status (this commit): the three genuinely-new ingredient families are
proven and audited in `LeanMlir/Proofs/SgdDescentCnn.lean`; the conv-layer
capstone assembly (mirroring `SgdDescentMlp.lean`'s shape) is the remaining
work. Everything below is mechanical given what's now in place — no new
mathematics is required.

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

## Remaining: the conv2-layer rung (then conv1)

Target net: `mnistCnnNoBnForward` (MnistCNN.lean:79). Loss-of-conv2-kernel:
`f₂(v) = CE(d₅∘r∘d₄∘r∘d₃ (maxPoolFlat (relu (flatConv (unflatten v) b₂ a₁))))`
with `a₁ = relu (flatConv W₁ b₁ x)` fixed.

1. **`ce_head3_input_grad`** — input-gradient of the 3-dense head
   `CE∘d₅∘relu∘d₄∘relu∘d₃` at the pooled vector: one `pdiv_comp` hop
   (peel `dense W₃`) on top of `ce_head2_input_grad` (SgdDescentMlp.lean),
   exactly as `ce_head2` was one hop on `ce_head_relu`. Hypotheses: relu₃,
   relu₄ pre-acts off-kink.
2. **`pool_relu_input_grad`** — input-gradient of `G₂ := CE∘head3∘
   maxPoolFlat∘relu` at the conv output `z₂`: `pdiv_comp` through `relu`
   (mask, `pdiv_relu`) then `maxPoolFlat`. Key glue, free by definitional
   unfolding: `pdiv (maxPoolFlat c h w) (flatten x) (t3Idx ci hi wi)
   (t3Idx co ho wo) = pdiv3 maxPool2 x ci hi wi co ho wo` is `rfl`
   (`maxPoolFlat` IS `flatten ∘ maxPool2 ∘ unflatten`, and `pdiv3` is
   defined as the flat `pdiv` of that composite — Tensor.lean:1563). Then
   `pdiv3_maxPool2_smooth` collapses the sum over pooled coordinates to
   the single argmax term: `pdiv G₂ z₂ j 0 = relu'(z₂ⱼ) ·
   (if IsArgmax then head3grad(window j) else 0)`. Hypotheses: relu₂
   margin + `MaxPool2Smooth` of the post-relu tensor (NB the pool acts on
   the POST-relu activation; state the margin there).
3. **`cnn_conv2_loss_gradAt`** — closed form via the EXISTING fold
   `conv_total_loss_grad_fold` (ConvLossFold.lean:34, generic in `G` —
   no new fold needed) + step 2. Note the conv-weight `pdiv` factor is
   NOT a Kronecker delta (weight sharing): keep it as the certified
   Jacobian (`conv2d_weight_grad_has_vjp.backward` form via
   `conv_weight_grad_bridge`) contracted with the step-2 closed form.
4. **`cnn_conv2_loss_grad_lipschitz`** — frozen everything (relu₂ mask:
   margin `a·D < |z₂|` via `conv2d_kernel_drift_total`; pool routing:
   `MaxPool2MarginQ (a·D)` of post-relu via `pdiv3_eq` +
   `maxPoolFlat_entry_lipschitz`; relu₃/relu₄ masks: margins at
   `w₃·(4hw·a·D)` resp. `w₄·d₁·w₃·(4hw·a·D)` via `maxPoolFlat_l1_contract`
   + `conv2d_kernel_drift_sum`), the difference collapses to the softmax
   drift exactly as in `mlp_input_loss_grad_lipschitz`. Logit drift
   `δ̄ = w₅·d₁·w₄·d₁·w₃·(4hw)·a·D` (conv2 runs at spatial `(2h)·(2w)`);
   constant `C₂ = 2·nC·(4hw)²·d₁²·w₃²·w₄²·w₅²·a²/(1−2δ̄)`-shaped, assembled
   by `ring` like the MLP file — do not precompute, let the calc produce it.
   The Jacobian factor: `∑_k |J(idx,k)| ≤ (4hw)·a` (each kernel entry
   touches `(2h)(2w)` outputs of its slab; `convPad` bounded by `a`).
5. **`cnn_conv2_sgd_descends`** — assemble via `sgd_descends` exactly as
   `mlp_input_sgd_descends` (margins at the step radius
   `D = lr·(‖∇f₂‖₁ + |kernel|·η)`; differentiability along the segment
   from the frozen margins + `maxPoolFlat_differentiableAt`).
6. **conv1** — one more conv+relu crossing; the input-side conv drift
   needs the conv-as-function-of-INPUT `ℓ1→ℓ1` bound (factor
   `ic·kH·kW·w₂ᶜ`-shaped, entrywise kernel bound `w₂ᶜ`) — a sibling of
   `conv2d_kernel_drift` with the roles of kernel and input swapped.
   Margins: relu₁ + everything in the conv2 list.

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
