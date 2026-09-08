# SGD descent through the CNN — remaining assembly

Status (this commit): COMPLETE. All four conv rungs are DONE —
`cnn_conv2_sgd_descends`, `cnn_conv1_sgd_descends`,
`cnn_conv2_bias_sgd_descends`, and `cnn_conv1_bias_sgd_descends`
(LeanMlir/Proofs/SgdDescentCnn.lean) prove one inexact SGD step on
either conv kernel OR either conv bias decreases the CE loss, under the
four (resp. five) margins at the step radius. Together with the MLP
rungs for the dense head, EVERY parameter of the Chapter-4 CNN has a
proven descent statement. 3-axiom clean (tests/AuditAxioms.lean).

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

## What's done, continued: the conv1 rung (NEW this commit)

The deepest descent statement. The genuinely new mathematics is conv AS
A FUNCTION OF ITS INPUT — conv is linear there:

- `convTap` — the input-side Jacobian entry (single kernel tap), the
  peer of `convPad`; extracted POINT-FREE from the certified input-VJP
  by contracting `conv2d_has_vjp3.correct` against a basis cotangent
  (`conv2d_input_pdiv3` / `conv2d_flat_input_pdiv`) — same basisVec
  trick as `conv2d_weight_pdiv`, no re-derivation.
- **Locality, not spatial count**: each input entry feeds ≤ `oc·kH·kW`
  outputs, each output reads ≤ `ic·kH·kW` inputs (`convTap_out_l1` /
  `convTap_in_l1`). Proof device: expand `|convTap|` as a kernel-offset
  indicator sum (`abs_convTap_expand`) and collapse pinned sums
  (`sum_pinned_le`); same device gives the input-drift bounds
  (`conv2d_input_entry_drift`, `conv2d_input_l1_drift` via
  `abs_convPad_sub_expand`).
- Drift chain `cnn1_*`: conv1 (`ℓ1`, spatial multiplicity `4hw`) → relu
  → conv2-as-input (`ℓ1`, locality `c·kH·kW·w₂`) → relu → pool → head;
  FIVE margins freeze everything (`cnn1_margin{1,2,3,4}_keeps_offkink`,
  `cnn1_postrelu2_close_seg`).
- `cnn1_pool_head_input_grad`: peel relu₁ (mask), contract the
  point-free taps with the EXISTING pool-collapsed conv2-rung gradient
  (`pool_relu_input_grad` reused verbatim at z₂).
- `cnn_conv1_loss_gradAt` via the same fold; `cnn_conv1_loss_grad_lipschitz`
  with constant `2·nC·(4hw)²·(c·kH·kW)²·d₃²·d₄²·w₂²·w₃²·w₄²·w₅²·a²/(1−2δ̄₁)`,
  `δ̄₁ = w₅·d₄·w₄·d₃·w₃·(c·kH·kW)·w₂·(4hw)·a·D`; `cnn_conv1_sgd_descends`.

## What's done, concluded: the bias rungs (NEW this commit)

Exactly as predicted — same argument, strictly easier than the kernels:

- **Bias primitives**: `conv2d_bias_sub` (the difference is EXACTLY
  `e o` — conv is affine in the bias), `conv2d_flat_bias_drift_total`
  (per-entry ≤ `‖e‖₁`, no `a`, no kernel mass),
  `conv2d_flat_bias_drift_sum` (`ℓ1` ≤ `(h·w)·‖e‖₁` — one bias entry
  feeds a whole channel), `conv2d_bias_pdiv` (the Kronecker channel
  indicator `if co = o then 1 else 0`, extracted from the CERTIFIED
  bias VJP `conv2d_bias_grad_has_vjp` by the same basisVec trick as
  `conv2d_weight_pdiv`; point-free).
- **conv2-bias rung** (`cnnb2_*`): the conv2-kernel chain verbatim with
  the conv stage's `a·‖e‖₁` replaced by the bare `‖e‖₁`; margins at the
  bare radius `D`; `pool_relu_input_grad` reused verbatim in
  `cnn_conv2_bias_loss_gradAt` (the fold is
  `conv_bias_total_loss_grad_fold`); Lipschitz constant = the kernel
  constant with `a² ↦ 1`; `cnn_conv2_bias_sgd_descends` at
  `D = lr·(‖∇f‖₁ + c·η)`. NO flatten/unflatten plumbing anywhere — the
  bias IS a vector, so the descends proof is the kernel proof minus the
  `Kernel4.unflatten_flatten` margin restatements.
- **conv1-bias rung** (`cnnb1_*`): same one layer deeper;
  `cnn1_pool_head_input_grad` reused verbatim; five margins at bias
  radii (`D`, `(c·kH·kW)·w₂·D`, pool at the same, `w₃·…`, `w₄·…`);
  constant `2·nC·(4hw)²·(c·kH·kW)²·d₃²·d₄²·w₂²·w₃²·w₄²·w₅²/(1−2δ̄₁ᵇ)`,
  `δ̄₁ᵇ = w₅·d₄·w₄·d₃·w₃·(c·kH·kW)·w₂·(4hw)·D`;
  `cnn_conv1_bias_sgd_descends`.

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
- `congrArg abs (by rw [...])` with the equation's RHS a METAVAR: the
  first `rw`'s closing `rfl` unifies the metavar with the half-rewritten
  form and silently closes the goal. Pin the type first (helper lemma
  `abs_triple_sum_sub_le`, or a `have` with the statement spelled).
- `refine le_trans (Finset.sum_le_sum fun i _ => ?_) ?_` can't
  synthesize the middle sum — prepare the tail bound as a `have hlast`
  FIRST and pass it as the second argument to pin `g`.
- Term-level `calc` whose first expression starts mid-line: a trailing
  `:= by` on the FIRST step needs its tactics indented past the calc
  EXPRESSION's start column, not the step's.
- `DifferentiableAt.comp _ hg hf` with `_` for the point can misresolve
  `f`/`x` — pass the point explicitly when `hg`'s point is an applied
  composite.
- The 12-deep conv1 forward needs THIRTEEN closing parens after `x₀`
  before `label` (count them programmatically; three separate sessions
  got it wrong by one).
- Deriving the bias variants of the deep `(w₅ * (2·t·δ̄ / (1 − 2·δ̄)))`
  motifs by hand-adjusting paren runs WILL re-associate the division one
  level up (hM0's shape then mismatches hfinal's, surfacing as
  application-type errors far from the typo). The safe recipe: generate
  the bias text from the kernel text by BALANCED substring substitutions
  only — `(a * D)` → `D`, ` * a ^ 2` → ``, `a * (lr * (` → `lr * ((`,
  `(Kernel4.unflatten u') b₁` → `W₁ b'` — and machine-check the net
  paren balance per block before building (a fourth session got it
  wrong by one, in six places).
