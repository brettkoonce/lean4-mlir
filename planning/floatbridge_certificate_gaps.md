# Float bridge: from "precise number" to "certificate" — the cross-cutting gaps

Planning doc for what comes *after* the architecture-complete ℝ→Float32 bridge. The bridge
now covers every op of CNN / ResNet / EfficientNet / ViT (incl. attention rounding +
input-sensitivity, projections, multi-head — see `planning/floatbridge_enet_vit.md`,
**923 decls audited, 3-axiom-clean**). More *architecture* is now low-value. The returns
live in the §3 cross-cutting gaps — the ones that decide whether the body of work reads as
*a certificate* or *a very precise statement about an abstract model*.

## 0. Where we are — the three tiers

Be explicit about what tier each claim sits in; a reviewer trusts the whole more when the
boundary is crisp.

- **PROVEN (in Lean, 3-axiom-clean):** `|float_op − real_op| ≤ budget` over the abstract
  `FloatModel` (`rnd` with `|rnd x − x| ≤ u·|x|`, `u = u32 = 2⁻²⁴`), instantiated at the
  repo's reference ℝ ops. Whole-net folds (`FloatClose`/`FloatBridges`), the ViT block in
  full generality, the MNIST *descent* headline.
- **MEASURED (empirical, on real silicon):** the supplied transcendental constants —
  `eexp` (≈1–2 ULP, `margin_probe.py`), `ers` (≈2·u32, `cifar_bn_margin_probe.py`),
  `esig`/`egelu` (≈1.5/7.3·u32, `scripts/transcendental_probe.py`, gfx1100/IREE).
- **TRUSTED (today, unproven):** the `den` (ℝ) → Float32 → IREE-emitted-GPU-kernel boundary
  (FMA, tree-reduction order, accumulation precision); subnormal-free operation; that IEEE
  binary32 RN actually *is* a `FloatModel` (true for normals: ½ULP = 2⁻²⁴).

The four items below shrink the TRUSTED column.

## 1. Kernel faithfulness — turn the biggest trusted item into a MEASURED one  ✅ DONE (2026-06-25)

**Delivered:** `scripts/kernel_faithfulness_probe.py` (run under the IREE/JAX venv). Both
parts run on real gfx1100. **(1) dot_general core:** GPU f32 vs f64, checked against the
`dot_close` Higham budget `((1+u)^(n+1)−1)·Σ|xᵢyᵢ|` — the real kernel is INSIDE the proven
budget at every fan-in, which is ~20× (n=64) to ~10⁴× (n=25088) conservative w.r.t. silicon.
**(2) whole MNIST-CNN forward:** the committed render, emitted with every stage as a result,
run once on GPU; each conv/dense stage's measured drift sits at ratio 4e-2 … 2e-6 of its
proven `layerBudget`, maxpool passes error through without amplification. The TRUSTED residual
(IREE lowering, FFI boundary, single magnitude profile) is now stated precisely in the script's
closing block — boundary validated on real silicon, not formally closed. Original plan below.

**The gap (§3.1).** The bridge bounds the *model*. Does the model bound the *kernel IREE
actually runs*? Today: trusted. Can't be *formally* closed without a verified compiler — so
the honest move is **differential validation**: run the emitted StableHLO on the GPU in f32,
diff against an f64 exact-ℝ proxy, and check the measured drift stays **inside the proven
`FloatModel` budget** at the real fan-in. If it does (it should — GPU FMA / higher-precision
accumulate is *more* accurate than the model's separate-mul-then-add Higham bound, which is
also association-independent), that's strong evidence the bridge bounds the real computation.

**Unblocked.** IREE-on-gfx1100 works in the venv (proven this session):
`/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python`, with `iree.compiler.compile_str(
mlir, target_backends=["rocm"], extra_args=["--iree-rocm-target=gfx1100"],
input_type="stablehlo")` then `iree.runtime` on the `"hip"` driver. `scripts/transcendental_probe.py`
already does exactly this shape for `logistic`/`tanh`.

**Plan — `scripts/kernel_faithfulness_probe.py`** (the matmul/whole-graph analogue of the
transcendental probe):
1. **Core matmul/dot first (the FMA + reduction-order question).** Emit a tiny `stablehlo.dot_general`
   (or reuse `IRPrint.lean`'s emitters), random inputs at controlled magnitudes, fan-in `n`.
   - GPU f32 output vs f64 numpy `x·y`.
   - Check `|gpu − f64| ≤ ((1+u32)^(n+1) − 1)·Σ|xᵢyᵢ|` (the **`dot_close` Higham budget**).
   - Also report the CPU-f32 twin (the *formula* in f32) for contrast — the GPU should be
     *within* the model AND typically *closer* than the naive twin (FMA). That contrast is
     the headline: "the model is conservative w.r.t. the real kernel."
2. **Then a whole rendered-net forward.** Take a committed net's emitted StableHLO (the
   `den`/StableHLO render, or the MLIR the verified trainers feed IREE), run it on gfx1100,
   diff each stage's f32 vs f64 against the per-stage `FloatClose` budget evaluated at the
   measured magnitude profile — exactly `cifar_bn_margin_probe.py`'s "measured-vs-proven"
   table, but the f32 side is the **real GPU kernel** (via IREE), not a numpy twin. The numpy
   twin tests the formula's rounding; IREE tests the deployed kernel. *That distinction is
   the whole §3.1 gap.*
3. **Honest output:** a table per op/stage — `measured drift | proven FloatModel budget |
   margin`. Plus a precise written statement of what remains trusted (IREE lowering preserves
   StableHLO f32 semantics; the FFI/runtime boundary).

**What this does NOT do:** formally prove kernel = model. It *validates* the trusted boundary
on real silicon and documents it precisely. That is the honest, runnable, highest-credibility
deliverable. **Effort: medium, mostly Python + IREE plumbing, no Lean.** Reference style:
`scripts/transcendental_probe.py`, `scripts/cifar_bn_margin_probe.py`, `mlir_poc/validate_cnn.py`.

## 2. Subnormals — a genuinely *closeable* proof gap (§3.3)  ✅ DONE (2026-06-25)

**Delivered:** `LeanMlir/Proofs/FloatSubnormalBridge.lean` (root + audited, 3-axiom-clean).
`FaithfulFloatModel` is the honest binary32 rounder — the clean relative bound on the normal
range (`err_rel`) *plus* the gradual-underflow absolute floor `η≈2⁻¹⁵⁰` everywhere (`err_abs`)
*plus* `rnd 0 = 0`. `toFloatModel` proves `FloatModel` **is** its `η=0` (no-underflow) face, and
`err_of_normal` collapses the honest bound to the clean `FloatModel.err` on normal arguments —
the precise "stays-normal ⇒ the whole bridge applies verbatim." The stays-normal invariant is
proved for the BN/LN normalization denominator: `bnDenom_normal`/`bnSqrt_normal`/`istd_ge_minNormal`
show `var+ε`, `√(var+ε)`, and `istd=1/√(var+ε)` are all `≥ minNormal` (since `ε≫minNormal`), so the
`rsqrt` keystone never touches subnormals. `subFloor_total_negligible` handles the residual
near-zero coordinates (post-ReLU tails) honestly: even if all `n≤2⁶⁴` rounded values underflowed,
the total floor `≤2⁻⁸⁶`, below every budget. Caveat → lemmas, as planned. Original plan below.

**The gap.** The relative-error `FloatModel` (`|rnd x − x| ≤ u·|x|`) holds only in the normal
range; near 0 it should be `|rnd x − x| ≤ u·|x| + η` (a subnormal floor `η ≈ 2⁻¹⁴⁹`). Deep
activations *can* underflow.

**The clean closure (proof, not caveat).** Don't add the `η` term — instead prove activations
**stay normal**: a magnitude *lower* bound complementing the upper bounds the bridge already
threads. LN/BN exist precisely to keep activations O(1); the normalized output `γ·x̂ + β`
has a controlled scale. State and prove a "stays-normal" invariant for the normalized blocks
(`|activation| ≥ floor` on the a-posteriori operating domain), so the relative model applies.
For the genuinely-near-zero ops (post-ReLU zeros, softmax tails), note ReLU/softmax are exact
or already-bounded so subnormals there don't propagate error. **Effort: small–medium Lean;
clean, architecture-agnostic, converts a caveat into a lemma.** New file `*SubnormalBridge.lean`
or a section in `FloatBridge.lean`.

## 3. Descent, not just closeness — the scientific punchline (§3.5)  → handoff: `planning/floatbridge_descent_pass.md`

**The gap.** Everything past MNIST is *closeness* (`|float − real| ≤ budget`). The headline
"a rounded training step still **decreases the loss**" exists only for MNIST. Closeness says
"the float net computes ~the real gradient"; descent says "it provably trains."

**Where it stands (verified 2026-06-25; full cold-start plan in
`planning/floatbridge_descent_pass.md`).** Pattern = one master theorem
(`linear_float_sgd_descends`, η *proven* not assumed) + a per-rung float grad-close as the
η source. CLOSED: **linear** end-to-end + the **entire MLP** — output (`mlp_output_float_sgd_descends`),
hidden (`mlp_hidden_float_sgd_descends`, Step 1), and input (`mlp_input_float_sgd_descends`,
Step 2) — all DONE 2026-06-25: each per-layer grad-close (`mlp_w{1,0}_grad_close`) wired into
its `mlp_{hidden,input}_sgd_descends` via the `gradAt`↔`reluMask` bridges
(`mlp_{hidden,input}_loss_gradAt_reluMask`, the latter factored through the reusable
`reluMask_dense_transpose_eq`), margins carried as the honest first cut. OPEN: **cnn** float
fusion (Step 3). The abstract-η smoothness side (`mlp_{…}_sgd_descends`, cnn ingredients) is
fully proven — only the cnn float fusion is missing.

**Next (see the handoff doc for exact theorem names + the wiring template).** Step 3: the
**cnn** conv rungs — `cnn_conv{1,2}_grad_close` (reuse the conv weight-grad bridges) wired into
the existing abstract-η `cnn_conv{1,2}_sgd_descends`, plus the whole-CNN capstone assembly.
**Honest stop line:** the joint all-layers step (logits non-affine) and the deep nets stay
closeness-only — descent needs a loss-gradient Lipschitz constant brutal at depth (no
`*_sgd_descends` exists for any deep net). **Effort: Step 3 high.** Don't let "descent" be
*implied* net-wide (honesty-pass flag F1: "loss-descent *step*" ≠ "the loss provably
*decreases*").

## 4. Eval-mode normalization — a quick deployed-forward win (§3.6)  ✅ DONE (2026-06-25)

**Delivered:** `LeanMlir/Proofs/BnEvalFloatBridge.lean` (root + audited, 3-axiom-clean).
`bnEvalAffine a b` (per-coordinate `aᵢ·xᵢ + bᵢ`) + its rounded peer `bnEvalAffineF`
(`fl(fl(aᵢ·xᵢ)⊕bᵢ)`). `bnEvalAffine_fold` proves the eval-BN formula `γ(x−μ)/√(σ²+ε)+β`
**equals** `a·x+b` with `a=γ/√(σ²+ε)`, `b=β−γμ/√(σ²+ε)` — the `√` lives only in the offline
constants, so the runtime map is a bare affine. `floatClose_bnEval` is the `FloatClose`
instance: one rounded mul + one rounded add, **fan-in 1 ⇒ no Higham γ, no `rsqrt`** (the whole
`BnFloatBridge` keystone — `rsqrt_lipschitz`/`bnIstd_close_at` — is unneeded), modulus
`bnEvalErr` (a `mulErr` + a constant rounding floor, affine in the inherited error). Drops into
`FloatClose.comp` to fold a deployed eval-forward. Original plan below.

Deployed accuracy uses **running-stats BN/LN at eval** = a fixed per-channel affine (no
reduction, no `rsqrt`!) — *far* simpler to bridge than the training-mode BN already built (the
`rsqrt` keystone, operating-point `bnIstd_close_at`, all unneeded). If a headline is the
*deployed* forward, the eval-mode affine `FloatClose` instance is a near-trivial win and makes
the deployed-accuracy story tight. **Effort: small Lean.** A `floatClose_bnEval` (fixed affine)
+ swap it into the eval-forward fold.

## 5. Honesty pass (cross-cutting, do alongside whichever above)  ✅ DONE (2026-06-25)

**Delivered:** `planning/floatbridge_honesty_pass.md` — a claim-by-claim tier ledger (PROVEN/
MEASURED/TRUSTED) of the float-bridge headlines with 7 flags + recommended rewordings (author's
call on the actual README edits). Headlines: F1 disambiguate "loss-descent *step*" (certified
update, all nets) from "the loss provably *decreases*" (linear end-to-end; MLP/CNN per-layer;
deep nets none) — the one overclaim risk (README L167); F2 take the linear win — cite
`linear_float_sgd_descends` (the float budget IS now fused into descent, `η` proven via
`linear_grad_close`), retiring the prior audit's "the two halves never meet" caveat — the biggest
underclaim; F3/F4 promote the kernel boundary (now MEASURED on silicon, §1) and subnormals (now a
lemma, §2) out of the "unverified" list; F5 optional eval-BN add (§4); F6/F7 affirm the existing
vacuous-budget + degenerate-witness honesty (keep worst-case decimals paired with measured). Original plan below.

Given the submission context (memories `project-diderot-comparator`, `repo-verification-reality`):
a short pass aligning headline claims with the PROVEN / MEASURED / TRUSTED tiers above —
flag anything that reads as descent where only closeness holds, anything that implies the
kernel where only the model is proven, and any degenerate/vacuous-budget witnesses. Cheap,
and it's what makes a skeptical reviewer trust the *proven* parts.

## 6. Suggested order

1. ~~**§1 kernel faithfulness**~~ ✅ DONE — `scripts/kernel_faithfulness_probe.py`, both the
   dot/FMA core and the whole-CNN forward validated inside-budget on real gfx1100.
2. ~~**§2 subnormals**~~ ✅ DONE — `FloatSubnormalBridge.lean`: honest `FaithfulFloatModel`,
   `FloatModel` = its `η=0` face, BN/LN stays-normal invariant, residual floor negligible.
3. ~~**§4 eval-mode BN**~~ ✅ DONE — `BnEvalFloatBridge.lean`: eval BN = fixed affine
   (`bnEvalAffine_fold`), bridged with no `rsqrt`/fan-in γ (`floatClose_bnEval`).
4. **§3 descent** — highest ceiling, highest risk; be honest about depth.
   (NB: linear + the **whole MLP** float→descent are now closed —
   `linear_float_sgd_descends` / `mlp_{output,hidden,input}_float_sgd_descends`, η proven not
   assumed; the open work is the **cnn** rungs (Step 3), then the honest stop at deep nets.)
5. ~~**§5 honesty pass**~~ ✅ DONE — `planning/floatbridge_honesty_pass.md` (7 flags; key: F1
   "loss-descent step"≠"loss decreases", F2 cite `linear_float_sgd_descends`, F3/F4 promote
   kernel/subnormals to their true tiers). Re-run before any writeup/submission.

**The one-line recommendation:** do §1 first. It's the single thing that most changes how a
skeptical reviewer reads the entire (large, genuinely-proven) body of work — and it's a
Python/IREE harness on hardware that's confirmed working, not a multi-week proof.
