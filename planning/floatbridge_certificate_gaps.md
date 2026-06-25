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

## 1. Kernel faithfulness — turn the biggest trusted item into a MEASURED one  ⭐ recommended

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

## 2. Subnormals — a genuinely *closeable* proof gap (§3.3)

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

## 3. Descent, not just closeness — the scientific punchline (§3.5)

**The gap.** Everything past MNIST is *closeness* (`|float − real| ≤ budget`). The headline
"a rounded training step still **decreases the loss**" exists only for MNIST. Closeness says
"the float net computes ~the real gradient"; descent says "it provably trains."

**Where it stands (see memory `floatbridge-mnist-chain`).** The Item-D η-composition rungs are
partway: MLP **output** rung (`mlp_output_float_sgd_descends`) + **hidden** grad-close
(`mlp_w1_grad_close` + `cotErr_nonneg`) done. Remaining: the **input rung**, the
**descent-wiring** (compose the rungs into a single-step loss-decrease), then **cnn** / the
**joint-step**.

**Next.** Push the MLP to a full single-step descent (input rung + wiring), then decide per
architecture whether descent is the goal or whether to bank closeness and state it honestly.
Descent for a *deep* net needs a loss-gradient Lipschitz (smoothness) constant — brutal at
depth, tiny lr; likely only tractable for shallow nets. **Effort: high, highest ceiling,
highest risk.** Don't let "descent" be *implied* for the deep nets where only closeness holds.

## 4. Eval-mode normalization — a quick deployed-forward win (§3.6)

Deployed accuracy uses **running-stats BN/LN at eval** = a fixed per-channel affine (no
reduction, no `rsqrt`!) — *far* simpler to bridge than the training-mode BN already built (the
`rsqrt` keystone, operating-point `bnIstd_close_at`, all unneeded). If a headline is the
*deployed* forward, the eval-mode affine `FloatClose` instance is a near-trivial win and makes
the deployed-accuracy story tight. **Effort: small Lean.** A `floatClose_bnEval` (fixed affine)
+ swap it into the eval-forward fold.

## 5. Honesty pass (cross-cutting, do alongside whichever above)

Given the submission context (memories `project-diderot-comparator`, `repo-verification-reality`):
a short pass aligning headline claims with the PROVEN / MEASURED / TRUSTED tiers above —
flag anything that reads as descent where only closeness holds, anything that implies the
kernel where only the model is proven, and any degenerate/vacuous-budget witnesses. Cheap,
and it's what makes a skeptical reviewer trust the *proven* parts.

## 6. Suggested order

1. **§1 kernel faithfulness** — biggest credibility gain, unblocked, runnable now, no Lean.
   Start with the matmul/dot probe (the FMA + reduction-order core), then the whole-net forward.
2. **§2 subnormals** — best pure-proof follow-up; small, clean, closes a real gap with a lemma.
3. **§4 eval-mode BN** — quick win if the deployed forward is a headline.
4. **§3 descent** — highest ceiling, highest risk; push the MLP rungs, be honest about depth.
5. **§5 honesty pass** — fold into whichever, before any writeup/submission.

**The one-line recommendation:** do §1 first. It's the single thing that most changes how a
skeptical reviewer reads the entire (large, genuinely-proven) body of work — and it's a
Python/IREE harness on hardware that's confirmed working, not a multi-week proof.
