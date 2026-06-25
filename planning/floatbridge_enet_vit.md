# Float bridge: finishing EfficientNet + ViT (and the gaps in the bounds)

Planning doc for extending the ℝ→Float32 bridge past the CNN/ResNet line to the
two smooth modern architectures, and an honest inventory of what's still missing
for the *bounds themselves* to mean what we want.

## 0. Where we are

The float bridge proves `|float_op − real_op| ≤ budget` over an abstract
`FloatModel` (relative-error rounding `|rnd x − x| ≤ u·|x|`, `u = 2⁻²⁴` for
binary32), instantiated at the repo's reference ℝ ops. Status by file:

- **Elementary** (`FloatBridge.lean`): `dot_close`/`dense_close` (Higham γ),
  `mul_close`, `sgd_step_close`, the `exp`/`eexp` softmax model, `mlp_float_close_uniform`,
  the E4M3 argmax results.
- **MNIST CNN** (`SgdDescentCnn.lean`): `cnn_float_close`, `flatConvF_close`,
  conv weight/bias SGD-step closes + the numeric capstone.
- **CIFAR no-BN** (`CifarFloatBridge.lean`): `cifar_float_close` (whole fwd),
  `cifar_stage{1,2}_conv{W,b}_step_float_budget`.
- **BatchNorm** (`BnFloatBridge.lean`): the `rsqrt` keystone `rsqrt_lipschitz`,
  `bnIstd_close` + the operating-point `bnIstd_close_at`, `bnMean_close`,
  `bnVar_close`, `bnForward_close_of`, `bnForward_close`.
- **BN input-sensitivity** (`BnInputBridge.lean`): `bnMean/Var/Istd/Forward_input_close`.
- **ResNet-34 structural ops** (`Resnet34FloatBridge.lean`): `add_close`,
  `reluAdd_close`, `flatConvStride2F_close`, `bnPerChannelFlat_close_of`, `gapFlat_close`.
- **Block step** (`Resnet34BlockBridge.lean`): `bnRelu_close`, `bnReluBudget`.
- **Composition backbone** (`FloatComposeBridge.lean`): `FloatClose`,
  `FloatClose.comp`, instances `floatClose_relu/_flatConv/_maxPool/_bnRelu`,
  combinators `floatClose_residualBlock`, folded units `floatClose_reluConv/_cifarStage/_resBlock`,
  the depth-generic `floatClose_iterate` + `floatClose_r34_stages`.
- **EfficientNet activations** (`EnetFloatBridge.lean`): sigmoid bounded,
  `sigmoid_close`/`swish_close` (rounding), `sigmoidScalar_lipschitz_abs` (¼-Lip),
  `swishScalar_lipschitz_abs` (Swish (1+A/4)-Lip), `floatClose_swish`,
  `floatClose_seScale` (the SE multiplicative-branch combinator).

**§1 (EfficientNet) — DONE 2026-06-25** (all 3-axiom-clean, audited):
- §1a `floatClose_addResidual` (FloatComposeBridge, the no-relu skip) + `floatClose_smoothResBlock`
  (EnetFloatBridge, conv→swish→conv + skip).
- §1b `floatClose_bn` (BN alone, via the extracted relu-free `bnStep_close` in Resnet34BlockBridge),
  `floatClose_dense`, `floatClose_gap` (via new `globalAvgPoolFlat_eq_bnMean` helper +
  `bnMean_abs_le`/`bnMean_input_close`), `floatClose_broadcast`, `floatClose_sigmoid`.
- §1c `DepthwiseFloatBridge.lean` — the one new conv lemma: depthwise read IS `convPad`, so
  `depthwiseConv2d_eq_dense` (single-output dense at fan-in kH·kW) → `depthwiseFlatF_close`
  reuses `dense_close`; `floatClose_depthwise` is the wrap.
- §1d `floatClose_seGate` (the SE gate net, 6-stage `.comp` fold) + `floatClose_seBlockFull`
  (feeds `floatClose_seScale`). NEW abstraction **`FloatBridges`** (∃-closure of `FloatClose`
  over the magnitude domain) + `FloatBridges.comp` automate magnitude threading;
  `floatBridges_mbconvBody` folds the WHOLE MBConv body (the 3 BNs enter as operating-point
  `FloatBridges (bnForward …)` hypotheses, discharged by `floatClose_bn` + `bnIstd_close_at`, §3).

All 3-axiom-clean (`tests/AuditAxioms.lean`).

### The reusable framework
`FloatClose A B f fF L` := on inputs `≤ A` (real and float), `fF` is within `L e`
of `f` at input error `e`, and both outputs `≤ B`. `FloatClose.comp` proves it
**composes** (moduli `∘`, magnitudes `A→B→C`). So a whole net is `FloatClose`
with the folded modulus — the certificate is the fold.

Two organizing facts to keep in mind:
- **smooth vs kinked.** A discrete branch (ReLU sign, maxpool argmax) forces
  *both* a conditional VJP and a sign-flip margin in the rounding. enet/vit are
  all-smooth ⇒ clean (no margins), the easy regime.
- **a-posteriori magnitudes.** Worst-case magnitude propagation through depth is
  vacuous (the BN `istd` ε-floor is ~10¹³ loose — `cifar_bn_margin_probe.py`).
  Use operating-point bounds (`bnIstd_close_at`); supply measured magnitude
  bounds per layer rather than proving worst-case growth.

## 1. EfficientNet — remaining

The two architecturally-novel pieces are **done** (`floatClose_swish`,
`floatClose_seScale`). What's left is plumbing + one new conv lemma.

### 1a. Re-land the additive skip + a closed block (near-done, was reverted)
`floatClose_addResidual` (MBConv skip `F(x)+x`, no trailing activation — `add_close`,
output `(1+u)(B+A)`) and `floatClose_smoothResBlock` (a `conv→swish→conv` body
wrapped by the skip, the closed enet-flavored block) were written and compiled to
**two small errors** before being cut for hygiene:
- `addResidual` error conjunct: `(M.add_close (hFe …) (hd i)).trans (by gcongr …)`
  — `gcongr` "did not make progress" bounding `|F va i|≤B`, `|va i|≤A`. Fix: replace
  the `gcongr` with an explicit `add_le_add (mul_le_mul_of_nonneg_left … hu) le_rfl`
  chain, or `nlinarith [(hFm va hva i).1, hva i, M.u_nonneg]`.
- `smoothResBlock` statement: the `∃ B L (Ff : Vec _ → Vec _), …` parenthesized
  binder hit a parse error — use `∃ B L Ff, …` (let `Ff`'s type infer) or split
  the `∃`.
The proof body (`set B1`; `floatClose_swish (n := c*h*w) … hB1`; `hB2` via
`unfold mulErr; positivity`; `((conv₁).comp hsw).comp conv₂`; `floatClose_addResidual`)
was otherwise correct. **~20 min.**

### 1b. The remaining `FloatClose` instances (all wraps of existing closeness)
- `floatClose_bn` — BN alone (relu-free `floatClose_bnRelu`): error = `bnForward_close_of`
  (rounding) + `bnForward_input_close` (shift), magnitude = `bnForward` bound + rounding.
  Modulus is the same `bnReluBudget` (relu only shrinks). **~40 lines** (re-derive
  bnRelu's internals minus the final `relu_close`).
- `floatClose_dense` — wrap `dense_close` + `denseErr_le_uniform` (the `layerBudget`
  modulus, fan-in = in-dim). Needed for the SE excite **and** the classifier head.
- `floatClose_gap` — wrap `gapFlat_close` (`Vec(c·h·w) → Vec c`, the per-channel mean
  modulus). Needed for the SE squeeze.
- `floatClose_broadcast` — `Vec c → Vec(c·h·w)` channel-broadcast = a reindex, **exact
  in float** (modulus `id`, like `floatClose_relu`/`_maxPool`). Needed to turn the
  SE gate (per-channel) into the broadcast `Vec(c·h·w)` that `floatClose_seScale` eats.

### 1c. Depthwise conv (the one genuinely-new conv lemma)
`depthwiseFlat = flatten ∘ depthwiseConv2d ∘ unflatten`; per output `(ch,hi,wi)` it's
`b ch + Σ_{kh,kw} W·pad` — a dot over `kH·kW` (no channel sum), fan-in `kH·kW`.
Two routes:
1. **Direct:** define `depthwiseFlatF` (rounded), prove `depthwiseFlatF_close` mirroring
   `flatConvF_close` but bounding the `kH·kW` dot via `dot_close`; needs the padded-window
   gather/reindex (the fiddly part, cf. `convPadWin`/`sum_s2` in `SgdDescentCnn`).
2. **Reuse:** depthwise = `c` parallel `flatConv(1→1)` (single-channel, fan-in `1·kH·kW`),
   so it's `flatConvF_close` per channel — but relating the repo's direct `depthwiseConv2d`
   to that needs a `depthwiseConv2d = parallel flatConv₁` lemma (also gather plumbing).
Then `floatClose_depthwise` (wrap). **Medium — the only real new analysis left for enet.**

### 1d. The MBConv fold + whole net
With 1a–1c, an MBConv is a `.comp` chain:
`expand(1×1 conv) → BN → swish → depthwise → BN → swish → SE(seScale ∘ gate-net) → project(1×1 conv) → BN → addResidual`.
Whole net = stem `.comp` (`floatClose_iterate` per stage at the B0 block counts)
`.comp` GAP `.comp` dense. **Pure assembly, no new lemmas.** Feed the operating-point
`bnIstd_close_at` for every BN `eistd` so the budget is non-vacuous (§3).

## 2. ViT — the plan (most of it ports from BN)

ViT is all-smooth too. The expensive primitive is **already done**: LayerNorm is
BN over a different axis.

**§2 status (started 2026-06-25, `ViTFloatBridge.lean`):**
- §2a **DONE** — `floatClose_layerNorm` is literally `floatClose_bn` (`layerNormForward =
  bnForward` *definitionally* in this repo; the whole BN bridge ports verbatim).
- §2b **DONE** — `Real.tanh_lipschitz_abs` (tanh 1-Lipschitz, from the repo's
  `hasDerivAt_tanh`), `geluScalar_lipschitz_abs` (bounded-domain Lipschitz by the Swish
  algebra: split + tanh 1-Lip + `|a²+ab+b²| ≤ 3A²`, no global derivative analysis),
  `gelu_close` (rounding), `floatClose_gelu` + `floatBridges_gelu`.
- §2d (MLP half) **DONE** — `floatBridges_vitMlpResidual`: `LN→dense→GELU→dense + skip`
  folds via `FloatBridges` (LN enters as the operating-point hypothesis, like the MBConv
  BNs). All 3-axiom-clean.
- §2c **DONE 2026-06-25** (`ViTAttentionFloatBridge.lean`, 3-axiom-clean) — the `Mat`-space
  attention track. Capstone `sdpa_close`: each output entry of the float `sdpaF` is within
  `attnOutErr` of the real `sdpa` (Attention.lean), chaining four reused pieces — score
  `dot_close` (Higham γ over fan-in `d`) → `1/√d` `mul_close` → per-row `softmaxF_close_at`
  (= `softmaxF_close` rounding + `softmax_perturb` logit shift, within `smErr`) → output
  `dot_close` at perturbed softmax weights (`attnDot_close`). The reusable softmax engine
  `softmaxF_close_at` + `smErr_nonneg` + `softmax_abs_le_one` was extracted into
  `FloatBridge.lean`. All-smooth ⇒ no sign-flip margins; budget a-posteriori in the supplied
  `qA`/`kA`/`vA`/`scaleA` magnitudes, proved in rounding. The transformer-block fold
  (LN→MHSA→+→LN→MLP→+) now has both halves: this `Mat`-space MHSA + the §2d `Vec`-space MLP.

### 2a. LayerNorm (a re-axis port of the BN bridge)
LN normalizes per-token over the feature dim; BN normalizes per-channel over spatial.
Same `mean → var → istd → affine`. Port `bnMean_close`/`bnVar_close`/`bnIstd_close(_at)`/
`bnForward_close_of` + the `_input_close` chain, reducing over the feature axis. The
`rsqrt` keystone (`rsqrt_lipschitz`) and the operating-point fix (`bnIstd_close_at`)
carry over verbatim. **Medium — a port, not new math.** (Check the repo's LN def in
`LayerNorm.lean` for the exact reduction shape and the per-channel `[d]` γ/β.)

### 2b. GELU (one new transcendental, established pattern)
Model tanh-form GELU by a supplied `fgelu`/`egelu` (like `eexp`/`ers`/`esig`).
`gelu_close` (rounding) + a GELU input-sensitivity (bounded Lipschitz — GELU' is
bounded, so the same algebraic/`deriv`-bound route as `sigmoidScalar_lipschitz`).
Validate `egelu` empirically (the repo validates `eexp` at 1–2 ULP; reuse the harness).
`floatClose_gelu` (wrap). **Small.**

### 2c. Softmax + attention
The softmax float model already exists (`softmaxF`, `eexp`, `smErr`/`cotErr`).
Apply the per-row softmax closeness to the attention scores (over the sequence
axis). Then `attention_close`: compose `QKᵀ` (matmul = `dot_close`) → softmax →
`·V` (matmul). New **assembly** from existing pieces; the matmul count is higher
but every matmul is `dense_close`. **Medium.**

### 2d. Transformer block + fold
Block = `LN → MHSA → +x → LN → MLP(dense→gelu→dense) → +x`. The residual is the
**no-activation** additive skip = `floatClose_addResidual` (§1a) — reuse, simpler
than ResNet's. Depth-12 encoder = `floatClose_iterate` at `n=12`. Patch-embed =
standard conv (`floatClose_flatConv`) + CLS/pos adds (`add_close`). **Assembly.**

ViT effort ≈ LN port (2a) + GELU (2b) + attention assembly (2c) + fold. The hard
primitive (LN/rsqrt) is reused, so no `<` new analysis than the BN work.

## 3. Cross-cutting gaps — what's missing for the *bounds* to mean what we want

These apply to every architecture and are the real "is this a certificate or a
number" questions.

1. **`den` → Float32 → IREE kernels.** The bridge is to the abstract `FloatModel`,
   not IREE's emitted kernels. Closing this means: (a) the `den` (ℝ) ↔ a `Float32`
   semantics of the rendered graph, then (b) `Float32` semantics ↔ the actual GPU
   kernels (FMA, tensor-core accumulation order, the transcendental units). This is
   the biggest standing gap; today it's *trusted* (the `iree-compile`/runtime/FFI
   boundary). The float bridge bounds the *model*; the kernel faithfulness is separate.
2. **Supplied transcendental constants.** `eexp` is empirically validated; `ers`
   (rsqrt) was measured ≈`2u32` by the CIFAR-BN probe. `esig`/`egelu` now **pinned on real
   gfx1100 silicon** (`scripts/transcendental_probe.py`, IREE rocm, the deployed
   `stablehlo.logistic`/tanh-form ops, 4M-pt sweep vs f64-exact): **`esig` ≤ 9.0e-8 ≈ 1.5·u32**,
   **`egelu` ≤ 4.3e-7 ≈ 7.3·u32** (the gelu const-truncation `0.7978845608` vs `√(2/π)` adds
   only ~7e-9; the rest is the silicon `tanh`). Safe pinned values: `esig ≤ 2·u32`, `egelu ≤ 8·u32`.
   Each *un*validated constant is a soft spot — these two no longer are.
3. **Subnormals.** The relative-error `FloatModel` is normal-range only. Deep
   activations *can* go subnormal (BN/LN keeping things O(1) mitigates but doesn't
   prove it). Either add a subnormal-aware model term or prove activations stay normal.
4. **Magnitude compounding / a-posteriori bounds.** `FloatClose` threads a magnitude
   `B` per layer; worst-case it grows (and the BN/LN `istd` ε-floor is vacuous). The
   honest certificate supplies *measured* per-layer magnitude bounds (the activations
   stay within `A`, which normalization enforces) and proves the *error* — i.e.
   a-posteriori in magnitudes, proved in rounding. State this explicitly; don't claim
   worst-case magnitude propagation.
5. **Closeness vs descent.** Everything past MNIST is *closeness* (`|float − real| ≤ budget`).
   The MNIST headline was *descent* ("a rounded step still decreases the loss"). Descent
   for a deep net needs a loss-gradient Lipschitz (smoothness) constant — brutal at depth,
   tiny lr. Decide per architecture whether the goal is closeness (publishable as "the
   float net computes ~the real gradient") or descent. Don't let "descent" be implied.
6. **Eval-mode vs training-mode normalization.** Deployed accuracy uses *running-stats*
   BN at eval = a fixed per-channel affine (no reduction, no `rsqrt`!) — far simpler to
   bridge than the training-mode BN built here. If the headline is the *deployed* forward,
   the eval-mode affine instance is a quick win; the training-mode bridge is for the
   SGD/descent story. Same for LN.
7. **Per-channel vs per-example.** The built BN is per-example (`bnForward` over the
   feature vec) / per-channel via `bnPerChannelFlat` (= `bnForward` per row). Confirm
   each net's deployed BN/LN axis and instantiate the matching one.

## 4. Suggested order

1. ~~Re-land §1a~~ **DONE** — addResidual + smoothResBlock.
2. ~~§1b instances~~ **DONE** — bn / dense / gap / broadcast (+ sigmoid).
3. ~~§1c depthwise~~ **DONE** (`DepthwiseFloatBridge.lean`); ~~§1d MBConv fold~~ **DONE**
   (`floatBridges_mbconvBody` via the new `FloatBridges` whole-net-assembly abstraction).
4. ~~ViT §2a LN~~ **DONE**, ~~§2b GELU~~ **DONE**, ~~§2d MLP-block~~ **DONE**.
5. ~~§2c attention~~ **DONE** (`ViTAttentionFloatBridge.lean`, `sdpa_close`).
   ~~§3.2 — pin `esig`/`egelu`~~ **DONE** (`scripts/transcendental_probe.py`, real gfx1100).
   ~~the transformer-block fold~~ **DONE** (`ViTBlockFloatBridge.lean`, `floatBridges_vitBlock`):
   the `Mat`↔`Vec` seam is `perRowFlat` + `FloatClose.perRow`/`FloatBridges.perRow` (lift a
   per-token bridge to the flattened `Vec (n·d)` sequence, same magnitude + same modulus, rows
   independent). The block is one `FloatBridges.comp` — MLP+LN₂ sublayer fully discharged
   (§2d `.perRow`), attention sublayer supplied (`hattn`).
   ~~attention **input-sensitivity**~~ **DONE — THE CAPSTONE**: `sdpa_input_close` (the
   Lipschitz-through-softmax bound: score sensitivity → `1/√d` → per-row `softmax_perturb`
   `e^(2δ)−1`, the only nonlinear step, no derivatives → output matmul). With `sdpa_abs_le`
   (attention is a convex average ⇒ magnitude-stable, via `softmax_sum_one`), `floatClose_sdpaSelf`
   packages self-attention (Q=K=V=X) as a full `FloatClose` (rounding `sdpa_close` + sensitivity
   `sdpa_input_close`), and **`floatBridges_vitBlockSelf`** is the **UNCONDITIONAL** ViT encoder
   block — `hattn` discharged, nothing supplied, every piece proved in rounding. Composes to
   depth via `FloatBridges.comp`.
   ~~adding the Wq/Wk/Wv/Wo projections~~ **DONE** (`ViTBlockFloatBridge.lean`): Q=XWq, K=XWk,
   V=XWv are per-token denses of the same `X` (the three-way fan-in) — each projection's float
   drift threads into `sdpa`'s slots (`dense_close` → `layerBudget` rounding, `dense_abs_le` →
   `layerAct` magnitude). `floatClose_projAttn` = `sdpa_close` (rounding at the float projections)
   + `sdpa_input_close` (sensitivity, projection drift as δ); `floatBridges_mhsaProj` adds the
   output projection Wo (`perRowFlat (dense Wo bo)`); **`floatBridges_vitBlockProj`** is the
   **fully-projected unconditional ViT block** (the deployed single-head form — `…vitBlockSelf`
   is its Wq=Wk=Wv=Wo=I special case).
   ~~multi-head reshape~~ **DONE** (`ViTBlockFloatBridge.lean`): multi-head attention is `h`
   parallel single-heads over feature slabs — in a head-major layout that is exactly `perRowFlat`
   (heads = blocks); the token-major↔head-major split/concat is a pure coordinate PERMUTATION
   (`gather`/`floatClose_gather` — exact in float, magnitude-stable, modulus `id`), so it preserves
   `FloatClose`. `mhSdpaSelfFlat = gather(reshape⁻¹) ∘ perRowFlat h (n·dh) (sdpaSelfFlat n dh) ∘
   gather(reshape)` (per-head scale `1/√dh`); `floatBridges_mhSdpaSelf` is one `FloatBridges.comp`
   chain; **`floatBridges_vitBlockMH`** is the multi-head ViT block (`h=1` = `…vitBlockSelf`).
   The whole ViT float story — LN/GELU/MLP/projected & multi-head attention/full block — is CLOSED.
6. Whichever of §3.1/§3.5/§3.6 the writeup needs to be honest about (the kernel gap,
   the closeness-not-descent framing, the eval-mode quick win).

**Landed (all 3-axiom-clean, audited in `tests/AuditAxioms.lean`):**
`FloatComposeBridge.lean` — `floatClose_addResidual`/`_dense`/`_bn`/`_gap`/`_residual`,
the `FloatBridges` abstraction (`.comp`, `.residual`, `cod_nonneg`, `modulus_zero_nonneg`)
+ `floatBridges_relu`/`_maxPool`/`_flatConv`/`_dense`. `Resnet34BlockBridge.lean` —
`bnStep_close` (relu-free, extracted from `bnRelu_close`). `Resnet34FloatBridge.lean` —
`globalAvgPoolFlat_eq_bnMean`. `DepthwiseFloatBridge.lean` (new) — `depthwiseConv2d_eq_dense`,
`depthwiseFlatF_close`, `floatClose_depthwise`, `floatBridges_depthwise`.
`EnetFloatBridge.lean` — `floatClose_smoothResBlock`/`_broadcast`/`_sigmoid`/`_seGate`/
`_seBlockFull`, `floatBridges_swish`/`_seBlockFull`/`_mbconvBody`. `ViTFloatBridge.lean`
(new) — `floatClose_layerNorm`, `Real.tanh_lipschitz_abs`, `geluScalar_lipschitz_abs`,
`gelu_close`, `floatClose_gelu`, `floatBridges_gelu`, `floatBridges_vitMlpResidual`.
`FloatBridge.lean` — `softmaxF_close_at`, `smErr_nonneg`, `softmax_abs_le_one` (the reusable
softmax-at-perturbed-logits engine, extracted from `softmax_ce_cot_close`).
`ViTAttentionFloatBridge.lean` (new, §2c) — `matScore_eq`, `attnScore_abs_le`, `mulErr_nonneg`,
the `attn{Score,Scaled,Weight,Out}Err` budgets (+ nonneg), `attnScore_close`/`attnScaled_close`/
`attnDot_close`, `rowSoftmaxF`/`rowSoftmaxF_close`, `sdpaF`, and the capstone `sdpa_close`.
`ViTBlockFloatBridge.lean` (new, the block fold) — the `Mat`↔`Vec` seam `perRowFlat` +
`FloatClose.perRow`/`FloatBridges.perRow`, `floatBridges_vitBlock` (supplied-attn), and THE
CAPSTONE: `floatClose_sdpaSelf`/`floatBridges_sdpaSelf` + the unconditional `floatBridges_vitBlockSelf`.
`ViTAttentionFloatBridge.lean` also now carries `sdpa_input_close` (attention Lipschitz),
`sdpa_abs_le`, `softmax_sum_one`, and the `attn{Score,Weight,Out}InErr` sensitivity budgets.

The novel methodological core (compose rounding budgets as a fold, split by
smooth/kinked, instantiate a-posteriori) is in place; everything above is reuse,
wraps, one new conv lemma (depthwise), and one transcendental (GELU).
