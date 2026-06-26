# A3 backward float-closeness: the deep-net assembly (pick-up plan)

Standalone plan for finishing **A3** — "the deployed float backward ≈ the certified `ℝ`
gradient" — for the five deep nets. The framework, the hardest new op, and a working
whole-net fold are already landed; what remains is **per-op breadth + per-net assembly**,
which is mechanical-but-large. This doc is the map.

Parent: `planning/tier23_float_and_syntactic_faithfulness.md` (A3 section). Honest stop:
A3 is gradient **closeness**, NOT descent — deep-net descent stays open by design (vanishing
`lr` at depth; `floatbridge_certificate_gaps.md` §3). Everything here is at a **smooth point**
(the ReLU/activation kinks read fixed sign masks), mirroring the §1a backward ties' nonzero-kink
hypotheses (`cifarCnnBn_has_vjp_at`'s `h1..h6`, `r34`/`vit` peers).

---

## 0. Where we are (DONE — read before planning)

The **fold mechanism is proven**: a feed-forward net's input-gradient VJP at a smooth point is
itself a *forward* composition of maps on the cotangent, so it threads the **same
`FloatBridges.comp` backbone** the forward uses. No new composition machinery is needed — only
the per-op backward bridges.

Landed (all 3-axiom-clean, wired into `lakefile.lean` roots + `tests/AuditAxioms.lean`):

- **`BnBackFloatBridge.lean`** — the BatchNorm backward keystone (the hardest new op):
  - `bnBetaGrad_close` / `bnGammaGrad_close` — param grads (`Σ dy`, `Σ dy·x̂`).
  - `bnGradInput_close` — the three-term input gradient
    `dx = (1/n)·s·(n·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))`, budget `bnGradInputBudget`, supplied float
    `istd`/`x̂` (`es`/`exh`).
  - reusable helpers `reduction_close` (float `Σ` of close terms), `sub_close'` / `sub_mag`.
- **`LinBackFloatBridge.lean`** — the backward fold:
  - `floatBridges_linBack` — linear input-VJP `dx = Wᵀ·dy` = `dense (Mat.transpose W) 0`,
    **reuses `floatBridges_dense`** (no new proof).
  - `floatClose_reluMaskBack` / `floatBridges_reluMaskBack` — ReLU-back = the `selectPos` mask
    (`cond : Fin n → Prop` `[DecidablePred]`), **exact in float**, modulus `id`.
  - `mlpInputGrad_floatBridges` — **the witness**: the whole 3-layer MLP input-gradient VJP
    float-bridges in one `.comp` chain (the backward peer of `cifar8_floatBridges`).

### The reusable backbone (compose over these)

- Forward op bridges (the backward of a *linear* op is another linear op, so these are reused via
  the transpose trick): `floatBridges_dense`, `floatBridges_flatConv`, `floatBridges_linBack`.
- Pointwise/structural (exact, modulus `id`): `floatBridges_relu`, `floatBridges_maxPool`,
  `floatBridges_reluMaskBack`, `floatBridges_gather` (reshape/permute).
- Norm: `floatBridges_bnPerChannelTensor3` (forward), `bnGradInput_close` (backward).
- Combinators: `FloatBridges.comp`, `FloatBridges.residual` / `floatClose_residual`,
  `FloatClose.perRowIdx` (block-diagonal / per-channel / per-head), `FloatClose.comp`.

---

## Part 1 — remaining per-op BACKWARD bridges

Each is a `FloatClose`/`FloatBridges` certificate for one backward op, reusing the forward
moduli (the backward of a linear op IS a linear op; pointwise masks are exact; only the norm /
softmax grads carry genuine new budget). Ordered by friction.

### 1a. MaxPool backward — `select_and_scatter` (EASY, exact)
The rendered maxpool backward scatters `dy` to the arg-max position (per pooling window), 0
elsewhere — a **select/scatter**, exact in float, modulus `id`, magnitude-nonincreasing. Mirror
`floatClose_reluMaskBack` exactly: a fixed arg-max index map (the smooth-point assumption: float
and real arg-maxes agree — the `MaxPool2Smooth` margin, already a §1a hypothesis). Deliver
`floatBridges_maxPoolBack`. **Effort: S. Risk: low.**

### 1b. Conv input-VJP — `convBackDenote` (MEDIUM, its own bridge)
The conv backward is a **reversed-kernel conv** (`convBackDenote W`, `CnnChainClose.lean`), NOT a
`flatConv` of a transformed kernel without index gymnastics. Two routes:
- **(preferred)** prove `convBackDenote W = flatConv (kernelFlip W) 0` (or the `decimate`/padding
  variant for strided), then reuse `floatBridges_flatConv` exactly as `linBack` reused dense.
  The `kernelFlip` is a pure reindex (a `gather`); the equality is a `Finset.sum` reindex proof.
- **(fallback)** a fresh `floatClose_convBack` mirroring `floatClose_flatConv` (same fan-in
  `layerBudget`; the convBack fan-in = `oc·kH·kW`). ~the flatConv proof, re-indexed.
Deliver `floatBridges_convBack`. **Effort: M. Risk: low–med** (index bookkeeping).

### 1c. BN-back as a composable `FloatClose` MAP (MEDIUM)
`bnGradInput_close` is **per-entry** (`|dxF_i − dx_i| ≤ budget`). To put BN-back in a `.comp`
chain it must be a `FloatClose A B (bnGradInputMap) (bnGradInputMapF) L` over the cotangent
`dy`. The per-entry bound is most of the error clause; still needed:
- the magnitude clause `|dx_i| ≤ B` (read off `MTr·(S/n)` from the budget derivation),
- the modulus `L e` = Lipschitz-in-`dy` (the budget is already affine in `Cdy`; re-read it as a
  function of the incoming cotangent error `e`).
The supplied float `istd`/`x̂` (`fs`/`fxh`) come from the forward bridge at the BN input (the
`FwdRec` reuse). Deliver `floatClose_bnBack` / `floatBridges_bnBack`. **Effort: M. Risk: med**
(re-expressing the fixed-`dy` budget as a `dy`-error modulus).

### 1d. Smooth-activation grads — swish′ / gelu′ (MEDIUM)
Unlike ReLU (0/1 mask), swish/gelu backward multiplies `dy` by the **smooth derivative**
`σ′`/`gelu′` at the saved pre-activation. So it's a `mul_close` of `dy` by a supplied float
derivative within `esig`/`egelu` (the transcendental budgets already pinned on gfx1100,
`transcendental-constants-pinned`). Mirror the forward `floatClose_swish`/`floatClose_gelu`
structure but for the derivative. Deliver `floatBridges_swishBack` / `floatBridges_geluBack`.
**Effort: M. Risk: med.**

### 1e. Depthwise / SE grads (MEDIUM, reuse 1b/1d)
- Depthwise input-VJP = a depthwise conv (per-channel), reuse `floatBridges_depthwise` via the
  transpose/flip trick (1b at depthwise).
- SE backward = the gate/scale backward: a `mul_close` (scale) + the squeeze/excite dense+sigmoid
  grads (reuse 1d for the sigmoid, `linBack` for the denses, GAP-back = broadcast/mean).
Deliver `floatBridges_depthwiseBack`, `floatBridges_seBack`. **Effort: M. Risk: med.**

### 1f. Attention grad — sdpa backward (HARD)
The softmax Jacobian couples a whole row (`diag(p) − p pᵀ`). The forward `sdpa_close`
(`ViTAttentionFloatBridge.lean`) + `softmax_perturb` are the model; the backward needs the
softmax-Jacobian closeness (a per-row `mul_close`/`sum_close` over the coupled weights). This is
the genuinely hard one. Deliver `floatClose_sdpaBack`. **Effort: L. Risk: med–high.**

### 1g. Loss-head cotangent lift (EASY)
`M.softmax_ce_cot_close` (`FloatBridge.lean:1808`) already bounds the float loss cotangent
`softmax − onehot` per-entry (`cotErr`). Lift to a `FloatBridges`/seed term so the whole-net
backward starts *from the loss* (end-to-end "float gradient ≈ real gradient"), not from an
abstract `dy`. **Effort: S. Risk: low.**

---

## Part 2 — per-net BACKWARD assembly (`<net>_grad_floatBridges`)

Each is the `cifar8`-style `.comp` chain over Part-1 bridges, at a smooth point, mirroring the
net's forward bridge. The certified `ℝ` backward graphs are already faithful
(`*ChainClose`/`*BackB0`/`*_has_vjp_at`) — the float bridge mirrors those op-for-op.

| Net | Backward ops needed | New from Part 1 | Effort |
|---|---|---|---|
| **cifar8** (no-BN) | convBack, reluMaskBack, maxPoolBack, linBack | 1a, 1b | **S** (first conv witness; do right after 1a/1b) |
| **cifarBn** | + bnBack | 1c | S–M |
| **r34** | convBack, bnBack, reluMaskBack, residual fan-in, maxPoolBack | (1b,1c done) + `FloatBridges.residual` on the backward | M |
| **mnv2** | + depthwiseBack, seBack, smooth (relu6 mask) | 1e | M |
| **efficientnet** | + swishBack, seBack, depthwiseBack | 1d, 1e | M–L |
| **convnext** | LN-back (= bnBack), geluBack, depthwiseBack, layerScale (mul) | 1d | M |
| **vit** | sdpaBack, LN-back, geluBack, linBack, residual | 1f | **L** |

**Residual fan-in (backward):** the skip `relu(F(x)+x)` backward routes the cotangent to BOTH
branches and ADDS — modulus `id` add, exactly the forward `floatClose_residual` mirrored. Add a
`floatClose_addBack` (cotangent duplication + the two-branch sum) if not subsumed by `.comp` +
`residual`.

---

## Suggested order

1. **1a (maxPoolBack) + 1b (convBack)** → land **`cifar8_grad_floatBridges`** (the conv backward
   witness, exact peer of `cifar8_floatBridges`). Highest ratio of value to effort; proves the
   conv fold like the MLP fold.
2. **1c (bnBack map)** → **`cifarBn_grad_floatBridges`**. Closes the CIFAR backward family
   (matches the A1 forward family).
3. **r34** (residual + BN backward) — the first Imagenette backward; biggest single credibility
   jump after CIFAR.
4. **1d/1e (smooth + depthwise/SE)** → **mnv2**, **efficientnet**, **convnext**.
5. **1f (sdpaBack)** → **vit** last (hardest op).
6. **1g (loss-head lift)** anytime — upgrades each `<net>_grad_floatBridges` from "≈ at an
   abstract `dy`" to "≈ from the loss."

**One-line recommendation:** do **1a → 1b → `cifar8_grad_floatBridges` → 1c →
`cifarBn_grad_floatBridges`** first (the CIFAR backward family, low risk, proves both folds), then
**r34**. The fold backbone is done; this is breadth.

---

## Gotchas carried forward (from the session that built the keystones)

- **Two `if`s, same condition** ⇒ `simp only [if_pos h]` / `[if_neg h]` (rewrites BOTH); plain
  `rw` hits only one. (Bit the `reluMaskBack` proof.)
- **Smooth-point masks** = fixed parameters `cond : Fin n → Prop` `[DecidablePred cond]` (the
  common float/real sign pattern), NOT recomputed — the honest model, matches the §1a ties.
- **Big nested budget defs** (`bnGradInputBudget`-style): `set`-abbreviate each intermediate to
  mirror the def's `let`s EXACTLY, build the chain, then `simp only [theDefs]; exact hfinal` —
  defeq closes (don't hand-massage the giant expression).
- **`Mat.transpose W i j = W j i`** (defeq); pass `fun i j => hW j i`. Bias `(0 : Vec n)` ⇒
  `hβ := le_rfl`, `hb := fun j => by simp`.
- **`bnIstd_abs_le` already exists** (`BnInputBridge.lean:123`); **`bnMean_abs_le`** too — don't
  redefine (name clash).
- **`FloatClose.cod_nonneg`** gives the `0 ≤ B` for `FloatBridges` witnesses for free.
- **whnf "timeout" errors** were CASCADE artifacts of an earlier parse/decl error — fix the real
  error first, they vanish.
- Supplied-stats discipline: float `istd`/`x̂`/`exp`/`sig` are MODELLED (close within
  `es`/`exh`/`eexp`/`esig`), discharged at instantiation by the forward keystones
  (`bnIstd_close`, centered closeness, the pinned transcendental budgets) — `rsqrt`/`exp` have no
  IEEE spec, so this is the honest (and only) shape.
- Wire every new module into BOTH `lakefile.lean` Proofs `roots` AND `tests/AuditAxioms.lean`
  (import + `#print axioms`), then `lake build` (the audit invokes `lean` directly and needs the
  oleans first).

---

## Scope / honest stops (state wherever cited)

- A3 = gradient **closeness** at a **smooth point**. NOT descent (open by design at depth). NOT
  the param-gradient *update* (that's the SGD/Adam step, a separate rung).
- The bridges are `float ≈ ℝ` over the proven SHlo backward graph; per-op StableHLO-spec
  conformance, IREE lowering, and `float32 ≈ ℝ`-the-silicon stay validated by `iree-compile` +
  the GPU runs (same residue as the forward + the honesty pass).
