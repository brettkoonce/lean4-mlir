# Float bridge — honesty pass: aligning headline claims to the tiers

`planning/floatbridge_certificate_gaps.md` §5. A cross-cutting read of the
float-bridge *headline* claims against the three tiers, flagging anything that
reads as **descent where only closeness holds**, **kernel where only the model is
proven**, or a **vacuous/degenerate budget quoted as tight**. This is an audit +
recommendation doc — the actual README/blueprint edits are the author's call (the
repo's doc passes have been left to Brett; nothing here is committed).

Reflects current state as of 2026-06-25, **after** §1 (kernel-faithfulness probe),
§2 (`FloatSubnormalBridge`), §4 (`BnEvalFloatBridge`), and — importantly — after
`linear_float_sgd_descends` landed (which the last full audit,
`repo-verification-reality`, predates). Several prior findings are now stale; this
pass corrects them in both directions (overclaim *and* underclaim).

## The three tiers (what's in each, now)

- **PROVEN (Lean, 3-axiom-clean).** `|float_op − real_op| ≤ budget` over the
  abstract `FloatModel` (`|rnd x − x| ≤ u·|x|`, `u = u32 = 2⁻²⁴`), instantiated at
  the repo's reference ℝ ops. Whole-net `FloatClose`/`FloatBridges` folds; the ViT
  block in full generality; the MNIST descent headline. **NEW:** the honest
  subnormal model (`FaithfulFloatModel`, `FloatModel` = its `η=0` face) and the
  BN/LN stays-normal invariant (§2); the deployed eval-BN affine `floatClose_bnEval`
  (§4); the **linear** end-to-end float→descent composition `linear_float_sgd_descends`.
- **MEASURED (empirical, real silicon).** The transcendental constants
  (`eexp`/`ers`/`esig`/`egelu`); the f32/f64 margin probe (`scripts/margin_probe.py`).
  **NEW:** the FloatModel→GPU-kernel boundary itself (§1,
  `scripts/kernel_faithfulness_probe.py`) — real gfx1100 `dot_general` + whole-net
  forward, GPU f32 vs the *proven* budget, inside-budget at every fan-in.
- **TRUSTED (today, unproven).** The `den`(ℝ)→Float32→IREE-emitted-kernel lowering
  (no verified compiler); the FFI/runtime boundary; that IEEE binary32 RN *is* a
  `FaithfulFloatModel` (true for normals: ½ULP = 2⁻²⁴; subnormal floor now modeled,
  §2). The `MlirCodegen.lean` full-recipe path (zero theorems) is trusted and stays
  the headline-accuracy path.

## Flags

### F1 — "loss-descent step" ≠ "the loss provably decreases" (the central one)

Two different claims share the word *descent* and the README rides the ambiguity:

- **The certified update direction.** The §1a whole-net ties prove the rendered
  graph's `den` equals the certified `θ − lr·∂Loss/∂θ` step. This is the *update
  direction* being the certified gradient step — proven over ℝ for **all** nets
  (Tier 1/2/3), strong and real. README calls it the "loss-descent step".
- **The loss actually decreasing.** The `*_sgd_descends` theorems prove an
  η-accurate step *decreases the loss* (smoothness discharged). This holds for:
  **linear** end-to-end (F2); **MLP** per weight layer (abstract η); **CNN** per
  conv layer only (`cnn_conv{1,2}_sgd_descends` + bias — abstract η, whole-CNN
  capstone open); **no deep net at all** (verified: zero `*_sgd_descends` for
  ViT/ConvNeXt/r34/enet/mnv2).

**Flag.** README L167: *"forward, gradient, and SGD-step rounding budgets for all
three nets (linear / MLP / CNN), each with a proven loss-descent guarantee — the
CNN (`cnn_conv2_sgd_descends` &c.) carries descent through the max-pool selection
margins."* "loss-descent guarantee … carries descent" reads as *the loss
decreases* for all three, but for the CNN only the per-conv-layer ingredients are
proven (abstract η, no whole-net assembly), and it is not float-fused.

**Recommend.** Disambiguate the two senses once, explicitly: the **loss-descent
step** (certified update direction; all nets) vs the **loss provably decreasing**
(linear end-to-end; MLP/CNN per weight layer under margins; whole-CNN + deep nets
open). For the CNN, say "per-conv-layer descent ingredients (capstone assembly
open)" rather than "carries descent".

### F2 — the linear float→descent composition is now CLOSED (correct an underclaim)

The last full audit (`repo-verification-reality`, 2026-06-12) concluded: *"NO
theorem instantiates η from a FloatBridge budget … the two halves never meet in a
theorem … 'Float32 training decreases the loss' remains UNCLOSED end-to-end."*
**That is now stale for the linear net.** `SgdDescentLinear.lean:339`
`linear_float_sgd_descends` uses the **actual** float-computed gradient
(`M.linearFloatGrad`) with accuracy `η = mulErr u a 1 0 (cotErr u eexp δ n)`
**proven** by `linear_grad_close` (not assumed) and fused into the descent — one
binary32 SGD step on MNIST-linear provably decreases the CE loss, with no abstract
gradient-accuracy parameter. The honest residue is exactly: the input bound `a`,
`0 ≤ lr`, the GPU `exp` accuracy `eexp`, the a-posteriori logit drift `δ` (the
documented FloatModel→kernel trust boundary), and checkable small-step/dominance
arithmetic.

**Flag.** README's Descent bullet (L237–246) cites `sgd_descends`,
`linear_sgd_descends`, `mlp_{output,hidden,input}_sgd_descends` — all the
**abstract-η** versions — and omits `linear_float_sgd_descends`, the one theorem
where the float budget and the descent meet.

**Recommend.** Add the stronger, now-true claim: *for the linear net the chain
binary32 → proven proximity → proven smoothness → loss decreases is closed
end-to-end (`linear_float_sgd_descends`), η proven not assumed.* This is the single
biggest claim-improving update and it directly retires the prior audit's headline
caveat — keep the deep-net honesty of F1 intact while taking the linear win.

### F3 — `FloatModel` ≠ the GPU kernel — but the boundary is now MEASURED (§1)

`FloatModel` is an abstract ℝ rounding model (`|rnd x − x| ≤ u·|x|`), **not** Lean's
`Float`/IEEE and **not** IREE's emitted kernel. README is mostly careful ("binary32
instantiates it with `u = 2⁻²⁴` on the normal range"). The residual TRUSTED item —
*does the model bound the kernel IREE actually runs?* — was, until this week, backed
only by the `margin_probe.py` f32/f64 **formula** twin.

**Flag.** README L313–315 lists *"any link from the Lean-side `FloatModel` to IREE's
actual kernels beyond the empirical probe"* as not-yet-verified, referencing only
the margin probe.

**Recommend.** Cite §1 `scripts/kernel_faithfulness_probe.py`: it runs the **emitted
StableHLO on real gfx1100** (not a numpy twin) and checks the measured drift against
the *proven* budget — `dot_general` inside the `dot_close` Higham budget at every
fan-in (~20×→10⁴× conservative), and a whole MNIST-CNN forward inside its per-stage
`layerBudget`. Move this item from "unverified" to **MEASURED-validated** (still
TRUSTED, not formally closed — keep that wording, but the validation is now direct,
on silicon).

### F4 — subnormals: now a lemma, not just an open caveat (§2)

**Flag.** README L306 lists subnormals under "Not yet verified anywhere (the model
is relative-error-only)."

**Recommend.** §2 `FloatSubnormalBridge.lean` converts the caveat into lemmas:
`FaithfulFloatModel` (honest binary32 rounder = relative bound on normals **plus**
the gradual-underflow floor `η≈2⁻¹⁵⁰`), `toFloatModel` (the idealized `FloatModel`
IS its `η=0` face), `err_of_normal` (honest⇒clean on normals), the BN/LN stays-normal
invariant (`var+ε`/`√`/`istd` all ≥ minNormal ⇒ the `rsqrt` keystone never
underflows), and `subFloor_total_negligible` (residual floor ≤ 2⁻⁸⁶). Reword from
"open" to "characterized": the relative model's domain of validity is now a proven
boundary, and the residual is proven negligible. What stays TRUSTED is only that the
*kernel* realizes the IEEE rounder — not the modeling gap.

### F5 — eval-mode BN (deployed forward) exists now (§4), not yet in the headline

`BnEvalFloatBridge.lean` bridges running-stats BN-at-eval as a fixed affine
(`floatClose_bnEval`, `bnEvalAffine_fold`) — no batch reduce, no runtime `rsqrt`.
Not a flag against any existing claim; a candidate **addition** if the *deployed*
(eval-mode) forward becomes a headline, since it makes the deployed-accuracy story
tight with a near-trivial instance.

### F6 — vacuous worst-case budgets: already disclosed — keep, don't quote as tight

The repo is already honest here and should stay so:
- the worst-case-vs-measured gap (up to ~10⁸) is flagged as "the quantitative case
  for a-posteriori certificates" (README L259);
- `e^(2δ)−1` is vacuous at the worst-case logit budget ⇒ `δ` must be measured (the
  a-posteriori hand-off, `mnist_cot_budget`);
- the BN `istd` worst-case `1/(2ε√ε)` is explicitly labeled vacuous vs the
  operating-point `bnIstd_close_at` (`cifar_bn_margin_probe.py`);
- the fp8 worst-case `B ≤ 61` is "vacuous on real data (mean margin ≈ 4.25)", the
  measured `B = 0.38` feeds the same theorem ⇒ 92.89% (README L286–288).

**Recommend.** No change — affirm. The only thing to guard: never let a summary
table or abstract quote a worst-case decimal *without* its measured companion (the
existing tables already pair them). Spot-check passed: no headline currently quotes
a vacuous worst-case as tight.

### F7 — degenerate concrete witnesses: already disclosed + mostly closed

"Concrete-instance honesty" (README L317+) already discloses live vs degenerate
witnesses, and the deep ReLU/BN nets now carry nonzero-Jacobian-sealed **live**
witnesses (`Mnv2Live`, `ResNet34LivePC.liveFwd2`, full-depth `liveFwd2Full`).

**Recommend.** No change — affirm. One residual open item to keep visible: the
BN-CNN live witness (per the `whole-net-backward-b2` tracking). Don't let
"live witnesses for the deep nets" generalize silently to *every* concrete instance.

## Bottom line

The float-bridge claims are, on the whole, unusually honest — the work needed is
small and cuts both ways:

1. **F1** — disambiguate "loss-descent *step*" (certified update, all nets) from
   "the loss provably *decreases*" (linear end-to-end; MLP/CNN per-layer; deep nets
   none). The one genuine overclaim risk.
2. **F2** — take the linear win: cite `linear_float_sgd_descends`; retire the prior
   audit's "the two halves never meet" caveat for linear. The biggest underclaim.
3. **F3/F4** — promote the kernel boundary (now MEASURED on silicon, §1) and
   subnormals (now a lemma, §2) out of the "unverified" list into their true tiers.
4. **F5** — optionally add eval-mode BN if the deployed forward is a headline.
5. **F6/F7** — affirm the existing vacuous-budget and degenerate-witness honesty;
   keep worst-case decimals paired with their measured companions.

These edits make a skeptical reviewer trust the (large, genuinely-proven) body of
work *more*, by drawing the PROVEN / MEASURED / TRUSTED lines exactly where they
actually fall.
