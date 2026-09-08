# Planning — closing the **whole-network backward** gap (non-degenerate, full-depth VJPs)

Companion to the `*_close.md` ladder. Those docs close each net *both ways* at the
**train-step render** level (text ↔ proven forward graph ↔ certified `θ − lr·grad`). This doc
is about the **other** whole-net axis: the `HasVJP`/`HasVJPAt` apex — "the whole network's
backward pass equals its Mathlib-`fderiv` Jacobian-transpose" — and specifically the residual
honesty gap the audit (`tests/AUDIT_REPORT*.md`, `formalization.yaml` fidelity §4) flags:

> *Full-net results are "the network composes and has a VJP at a witness point". The concrete
> instances that make the conditional capstones unconditional are, for the kinked nets,
> **degenerate** (zero-weight / constant-output, hence zero Jacobian). Live witnesses for the
> deep / BN / ReLU-kink nets are follow-up.*

**The math is mostly done.** What remains is (1) non-degenerate witnesses for the kinked nets,
(2) a genuine nonzero-Jacobian seal, and (3) one piece of ViT plumbing. No new analysis is
needed; this is witness construction + a list-folding tower, not new VJP theory.

---

## 0. The bar — what "whole-network backward, honestly" means

For an all-smooth net (only `0 < ε`), a whole-net VJP at **every** input and **every** weight
is the honest, complete statement. For a **kinked** net (ReLU / ReLU6 / maxpool) it provably
*cannot* hold everywhere — `fderiv` takes its junk-zero default exactly at the kink — so the
honest targets are, in increasing strength:

1. **Conditional apex** `*_has_vjp_at` — VJP holds at any input whose pre-activations dodge the
   kinks (the hypothesis bundle). ✅ **Done generically, full depth, for every net.**
2. **A non-degenerate witness** — one concrete input/weight choice that (a) discharges the whole
   bundle with **genuine nonzero weights** (not a zero-weight collapse) and (b) has a **non-constant
   forward**, so the witness is not vacuous. ✅ **Done for `Mnv2Live` (MobileNetV2) and `ResNet34LivePC`
   (a 2-channel ResNet-34, `liveFwd2_nonconstant`).** Both prove `forward X ≠ forward 0`; `Mnv2Live`
   additionally has the level-3 seal (below), ResNet-34 not yet.
3. **A nonzero-Jacobian seal** — at the witness, `∃ i j, pdiv forward x i j ≠ 0`, i.e. the
   *backward map itself* is provably non-trivial there. ✅ **Done for `Mnv2Live`**
   (`MobileNetV2JacobianSeal.lean`, `fderiv ℝ forward 0 ≠ 0` ⇒ non-trivial backward, audited
   3-axiom-clean) **and for the live ResNet-34 `liveFwd2`** (`ResNet34LiveSeal.lean`,
   `fderiv ℝ liveFwd2 Y ≠ 0` ⇒ non-trivial backward, audited 3-axiom-clean — the maxpool kink
   handled by sealing at a channel-symmetric base `Y`, below). 🔴 still missing for the BN-CNN
   (a deep zero-weight kinked witness with no live forward yet).
4. **Almost-everywhere** — the bundle holds off a measure-zero set (the principled "the backward
   is correct for a.e. input/weight"). 🔴 **Not attempted.** (Stretch; needs `MeasureTheory`.)

The goal of this doc is to take every flagship net to **at least level 3**, and to scope level 4.

---

## 1. Current state (corrected — verified 3-axiom-clean via `tests/AuditAxioms.lean`)

### 1a. Smooth nets — full-depth backward is **DONE** (level: unconditional, non-degenerate)

These hold at **every input and every weight** (only `0 < ε`); no kink, so no witness needed.
Generic in weights ⇒ already non-degenerate. Nothing to do except the ViT plumbing in §2-C.

| Net | Full-depth backward theorem | Depth | File:line |
|---|---|---|---|
| **EfficientNet-B0** | `efficientnetForwardB_full_has_vjp{,_correct}` | **all 16 MBConv**, batched (`N`), true-BN, SE | `EfficientNetFullB0.lean:381,490` |
| **ConvNeXt-T** | `convNextForwardT_has_vjp{,_correct}` | **full [3,3,9,3] = 18 blocks** | `ConvNeXtFullT.lean:177,262` |
| **ViT-Tiny** | `vitTiny_has_vjp_correct` ✅ **(this work)** | **12 distinct-param blocks, 3 heads, D=192, MLP 768** | `ViTDepthK.lean` (§3) |
| ″ (generic depth) | `vitForwardKV_has_vjp{,_correct}` | **arbitrary `k`, distinct per-block `ps : Fin k → BlockParamsV`** | `ViTDepthK.lean:200,244` |

EfficientNet-B0 is the strongest result in the repo: a full-spec, batched, true-batch-norm +
squeeze-excite whole-net backward, generic in weights. ConvNeXt-T matches at full depth (caveat:
scalar-LN over `c·h·w`, not the faithful channel-LN — a separate, minor faithfulness upgrade).
ViT-Tiny now joins them: the depth-`k` distinct-param tower `vitForwardKV` already existed and was
audited; **Item C added the named production capstone `vitTiny_has_vjp_correct`** instantiating it
at the real `MainVitTrain.lean` `vitTiny` spec (audited 3-axiom-clean, 616/616 gate). All three
smooth flagships now have a full-architecture whole-net backward. (`vit_full_has_vjp`,
`Attention.lean:3665`, remains the weight-*tied* arbitrary-depth form — superseded for the
distinct-param case by `vitForwardKV`.)

### 1b. Kinked nets — depth fold **DONE**, witnesses **DEGENERATE**

The conditional apexes are full-depth and audited clean; the **machinery to fold arbitrary depth
already exists** (`vjp_comp_at`, `vjp_chain{,_at}`, `resStage_has_vjp_at`, all in `ResNet34.lean`
/ `Tensor.lean`). The gap is purely the **concrete instance**.

| Net | Conditional apex (full depth, ✅) | Unconditional witness | Witness verdict |
|---|---|---|---|
| **ResNet-34** | `resnet34_has_vjp_at` (`ResNet34.lean:174`) | `ResNet34Concrete.resnet34Concrete_has_vjp_correct` (`:735`) | 🔴 degenerate (1ch); **✅ live witness `ResNet34LivePC.liveFwd2_has_vjp_correct` + `liveFwd2_nonconstant` (level 2) — now ✅ level-3 sealed: `ResNet34LiveSeal.liveFwd2_jacobian_nonzero` ⇒ `liveFwd2_backward_nontrivial`** |
| **MobileNetV2** | `mobilenetv2_has_vjp_at_correct` (`MobileNetV2.lean:580`) | `MobileNetV2Concrete.*` (`:936`) | 🔴 **degenerate** — all-zero kernels, constant output |
| ″ (live) | ″ | `Mnv2Live.mnv2Live_has_vjp_correct` (`:1008`) | ✅ **level-3 sealed** — `mnv2Live_jacobian_nonzero : fderiv ℝ forward 0 ≠ 0` ⇒ `mnv2Live_backward_nontrivial` (`MobileNetV2JacobianSeal.lean`); nonzero weights, 1×2×2 |
| ″ (live, **full depth**) | ″ | `Mnv2Live.fwdFull_has_vjp_correct` (`MobileNetV2JacobianSealFull.lean`) | ✅ **level-3 sealed, full 17-block** — `fwdFull_jacobian_nonzero` ⇒ `fwdFull_backward_nontrivial` + `fwdFull_nonconstant`; 15 identity skip blocks (`ivId a = a+3`, no relu) wash out to `+45`, so `fwdFull = fwd + 45` and the seal reuses `Qq`/`g_hasDerivAt`; VJP genuinely composed through all 17 blocks |
| **BN-CNN** | `cnn_has_vjp_at` (`MnistCNN.lean`) | `CnnConcrete.cnnConcrete_has_vjp_correct` (`:957`) | 🔴 **degenerate** — zero-weight resblocks |
| **MNIST/CIFAR CNN** | `mnistCnnNoBn_has_vjp_at`, `cifarCnn8_has_vjp_at_correct` | `Mini`/`Spatial`/`Tiny.*` | ⚠️ mixed — `Spatial` (3×3 SAME) is genuinely-nonzero but tiny |

**The one reusable non-degenerate kit that exists** (from `Mnv2Live`, the template to copy):
- `bn13_window` (`MobileNetV2.lean:1016`) — `0 < bnForward n ε 1 3 z k < 6` for **any** `z`,
  `n ≤ 8`, `γ=1, β=3, ε>0`. Discharges a ReLU6 bundle **without** a constant collapse.
- `maxPool2Smooth_of_injective` (`MnistCNN.lean:802`) + `bnForward_injective` — no-tie from
  positional injectivity of the stem.
- `bnIstd_pos`, `bn1_devSum_scale` (`:1094`), `chSum_convX` (`:1427`), `mnv2Live_forward_nonconstant`
  (`:1463`) — the non-vacuity seal (level 2): BN rescales deviations by `istd > 0`, so an
  asymmetric stem survives to the output ⇒ `forward X ≠ forward 0`.

---

## 2. The remaining work

### Item A — ResNet-34 **Live** witness *(headline; genuinely multi-session)*

**Stage 1 — DONE and banked** (`LeanMlir/Proofs/ResNet34Live.lean`, a pure leaf; built but
deliberately *excluded* from the audit, lakefile note "a live witness and NOT in the AuditAxioms
headline set", because it is not yet a real witness). `liveDown` — a signal-carrying strided
downsample whose **projection skip** `relu(bn₂₀(decimate x) + 1)` reads the input (identity-decimate
proj via `rblkPStrided_has_vjp_at`, body zeroed, `βp=20` keeps proj `>0` so ReLU is smooth) — plus
`liveFwd_has_vjp_at`/`_correct` assembling the whole-net VJP with every smoothness/no-tie hypothesis
discharged. So the smoothness side (the old A1/A2) is solved; the discharge routes (`bnForward_lb`
positivity, `maxPool2Smooth_of_injective`) work as planned.

**The blocker (confirmed this session — a real structural obstruction, not a missing lemma):**
`liveFwd` is still **constant-output**. A **1-channel** net with **BN before the final GAP** is
necessarily constant: `gap` (1 channel) = spatial mean = BN's global mean = `β` (`bnForward_mean`),
and keeping ReLU in its identity region (required for smoothness) makes the post-BN map affine, so
`gap = β + consts` for *every* input — the signal lives in the BN-normalized *deviations*, which GAP
averages away. At 32²→1×1 there is a *second* collapse: BN over the final 1 spatial element is `β`
trivially. **This is the actual reason `ResNet34Concrete`/`CnnConcrete` are degenerate — structural,
not zeroed weights.** A 1-channel ResNet-34 witness is impossible; no choice of weights escapes it.

**The non-vacuity strategy (refined — the `chSum`/argmax plan is superseded):** the clean carrier is
a **pointwise strict channel-order invariant** — maintain `forward(z) ch 1 < forward(z) ch 0` at
*every* spatial position, through every layer. *Strict* order at the head gives `forward X ≠ forward 0`
(where the two channels coincide) **directly** — no channel-deviation sum, no maxpool-argmax tracking.
This works with the **scalar** `bnForward` the witness already uses (no move to per-channel BN needed):
even at the final 1×1 spatial collapse, scalar BN over the two channel values forces only their *mean*
to `β`, so an asymmetric stem keeps `ch0 ≠ ch1` and the per-channel head reads it. Every layer
preserves the invariant — and the maxpool, the supposed crux, is the easy case.

**Stage 2 — DONE and banked** (`LeanMlir/Proofs/ResNet34Live2.lean`, build-checked, 3-axiom-clean, a
Proofs root but — like Stage 1 — not yet in the AuditAxioms headline set):
- `maxPool2_chan_lt` — **max preserves strict pointwise channel domination**
  (`max ch0 ≥ ch0(argmax ch1) > ch1(argmax ch1) = max ch1`). The maxpool×order interaction the doc
  feared is a four-line `max_lt` — *no* argmax computation.
- `bnForward_chan_lt` — scalar BN (γ=1) preserves strict order between coordinates
  (`bn k₀ − bn k₁ = (z k₀ − z k₁)·istd`, `istd > 0`).
- `relu_chan_lt` / `relu_pos_eq` — ReLU preserves strict order in the kept-positive region.

**Stage 3 — the level-2 live ResNet-34 witness is COMPLETE** (`LeanMlir/Proofs/ResNet34LivePC.lean`,
**in the AuditAxioms headline set, gate 628/628 3-axiom-clean**). The first **non-degenerate**
ResNet-34 whole-net backward witness — retires the "degenerate constant-output witness" caveat:
- `liveDownPC` — the **2-channel signal-carrying strided downsample** (channel-diagonal
  identity-decimate projection `WsP2`, zeroed body), with its full whole-block VJP / `DifferentiableAt`
  / nonnegativity, mirroring Stage 1's `liveDown` at `oc = ic = 2`. `βp = 20 > √(2·h·w)` keeps proj `> 0`.
- `stem2` + `stem2_inj` + `stem2_maxpool_smooth` — the 2-channel stem. The maxpool no-tie turned out
  *clean*: a **channel-diagonal identity** stem conv (`WsId2 = δ_oi`) makes `flatConvStride2 = decimate`
  (lemma `flatConv_WsId2_X2`, the 2-channel peer of Stage-1 `flatConv_id_X`), so the stem is **globally**
  injective and the no-tie reuses Stage-1's `bnForward_injective` / `finProdFinEquiv.injective` pattern
  wholesale (asymmetry rides on the input, not the conv). `β = 30 > √512` positivity.
- **`liveFwd2_has_vjp_at`** — the **whole 2-channel ResNet-34 backward**: `dense ∘ gap ∘ liveDownPC×3 ∘
  maxpool ∘ stem2` with empty identity-block chains (`chainComp [] = id`; full depth = Item D), every
  hypothesis of the dimension-generic `resnet34_has_vjp_at` discharged. `liveFwd2_has_vjp_correct` exposes
  the `pdiv`-Jacobian. 3-axiom-clean.
- **`liveFwd2_nonconstant`** — the **non-vacuity**: `liveFwd2 X2 ≠ liveFwd2 0`. The Stage-2 channel-order
  invariant `Dom2` (channel 1 strictly dominates channel 0 at every spatial position) is threaded through
  the assembly via `Dom2_bn` / `Dom2_relu` / `Dom2_maxpool` / `Dom2_decimate` / `Dom2_add_const`: the
  positional input `X2 i = i` has channel 1 dominating (`Dom2_X2`, via the `finProdFinEquiv` channel-major
  index), so the per-channel head gives `out 0 < out 1`; the zero input collapses channel-symmetric
  (`liveFwd2_zero : liveFwd2 0 = const 21`).
- `bnForward_coord_inj` — scalar BN injective per coordinate (a reusable no-tie ingredient).

**What remains (now only depth):**
- **Level 3 — the nonzero-Jacobian seal** for ResNet ✅ **DONE** (`ResNet34LiveSeal.lean`,
  `liveFwd2_jacobian_nonzero : fderiv ℝ liveFwd2 Y ≠ 0` ⇒ `liveFwd2_backward_nontrivial`, audited
  3-axiom-clean, in the AuditAxioms headline set). The doc's worry — "the ReLUs/maxpool bind
  off-witness, so the `Mnv2Live` input-0 trick does not transfer" — is resolved by **moving the base
  point** rather than computing a BN-variance derivative: seal at a **channel-symmetric, spatially-
  decreasing base `Y`** (per-channel injective, so the maxpool has no ties and the VJP holds; and the
  *channel difference* — the live carrier — vanishes at `Y`, the ResNet analogue of Mnv2's `0`). The
  stem/ReLU6 sites are globally off (BN positivity), so the *only* kink is the maxpool, which is a fixed
  top-left selection *eventually* along the ray `Y + t·V` (`maxPool2_eq_at_max` + continuity). The
  **exact BN channel-difference identity** `bn z k₀ − bn z k₁ = (z k₀ − z k₁)·istd` then collapses the
  whole chain: `liveFwd2(Y+t·V) 0 − liveFwd2(Y+t·V) 1 = t · Π(t)` with `Π` a product of four positive
  `istd`s, so every `istd`-derivative cross-term carries a factor `t` and drops at `0` — `g'(0) = Π(0) ≠ 0`,
  no variance derivative ever needed (exactly the Mnv2 mechanism, transplanted to base `Y`).
- **Item D — full depth** ✅ **DONE** (`ResNet34LiveFull.lean`): the empty identity-block chains are
  filled with a 2-channel `idBlk2` to the real `[3,4,6,3] = 16`-block depth (3 strided downsamples +
  13 identity blocks). Both level 2 (`liveFwd2Full_nonconstant`) and level 3
  (`liveFwd2Full_jacobian_nonzero` ⇒ `liveFwd2Full_backward_nontrivial`) carry through, audited
  3-axiom-clean. The identity blocks have a **zeroed body**, so on a nonnegative activation each is
  the affine shift `relu(x+1) = x+1`; a chain of `k` is `x + k`, transparent to the channel-difference
  carrier (`cd(x+c) = cd x`) and to the Jacobian (it is the identity). The shift then **washes out
  through the next downsample's BN** (`bn(z+c) = bn(z)`), so `liveFwd2Full = liveFwd2 + 2` (the lone
  surviving shift, after the final chain where no BN follows, cancels in the channel *difference*) —
  and the whole seal reduces to `ResNet34LiveSeal`'s `gd_hasDerivAt` / `Rr` verbatim.

Effort: **Item A is DONE — a full-depth `[3,4,6,3]`, non-degenerate, nonzero-Jacobian-sealed
ResNet-34 whole-net backward (2-channel), audited 3-axiom-clean.**

### Item B — the **nonzero-Jacobian seal** (level 3, generic)

Today even `Mnv2Live` stops at `forward ≠ const`. The seal upgrades that to a genuine backward
non-triviality.

- **B1 — generic bridge** ✅ **DONE** (`LeanMlir/Proofs/JacobianSeal.lean`, audited; gate 622/622):
  - `HasVJP.backward_ne_zero_of_pdiv_ne` — one nonzero Jacobian entry `pdiv f x i₀ j₀ ≠ 0` ⇒ the
    proven backward is **not the zero map** at `x` (`h.backward x (basisVec j₀) i₀ ≠ 0`); the basis
    cotangent collapses `HasVJP.correct`'s sum to its diagonal term.
  - `fderiv_eq_zero_of_pdiv_all_zero` / `exists_pdiv_ne_of_fderiv_ne` /
    `HasVJP.backward_nontrivial_of_fderiv_ne` — the **`fderiv` form**: all entries zero ⇔
    `fderiv ℝ f x = 0` (via the standard-basis decomposition `sum_smul_basisVec`), so the clean
    analytic hypothesis `fderiv ℝ forward x ≠ 0` discharges the seal. No differentiability needed.
  - `mnistLinear_backward_nontrivial` — end-to-end demo: `pdiv (mnistLinear W b) = W`, so any
    `W i₀ j₀ ≠ 0` seals the linear classifier's backward as non-trivial.
- **B2 — discharge the premise at the kinked witnesses.** ✅ **`Mnv2Live` done**
  (`MobileNetV2JacobianSeal.lean`; also adds the pointwise `HasVJPAt` seal variants the kinked
  witnesses actually consume — B1 was `HasVJP`-only). The honest computation, as anticipated: an
  explicit directional derivative, **not** derivable from `forward ≠ const`. The decisive trick is
  the choice of witness point. Because `bn13_window` holds for *every* input, `Mnv2Live` is
  **globally smooth** (the ReLU6 kinks never bind), so `forward` equals a ReLU6-free closed form
  (`forward_eq`) and `forward(z)₀ = 3 + (1/4)·P(z)·L(z)` with `L = chSum∘conv` **linear**
  (`chSum_convX_smul`, `L(X) = -3`) and `P` a product of four **positive** `bnIstd` (`bnIstd_pos`).
  Along `t ↦ t·X` this is `3 - (3/4)·t·P(t·X)`, whose product-rule cross-term carries a factor `t`
  — so it **vanishes at `t = 0`**, leaving `g'(0) = -(3/4)·P(0) < 0` with *no* BN-variance derivative
  (the hairy `∂istd` piece) ever needed. So the seal is exact and constructive at the input `0`
  (not an MVT/nonconstructive point — and the doc's "MVT point may not be a smooth VJP point" worry
  is void here since *every* point is smooth). ✅ **`ResNet34Live` also done**
  (`ResNet34LiveSeal.lean`, see Item A "What remains" above): the same carrier-vanishes-at-base
  mechanism, but the base point is *moved* (channel-symmetric `Y`, not the input `0`) because the
  maxpool — unlike Mnv2's ReLU6 — genuinely binds, so it cannot be globally smooth; the maxpool kink is
  instead a fixed top-left selection *eventually* along the ray. ✅ **`Mnv2Live` now also full depth**
  (`MobileNetV2JacobianSealFull.lean`, the real 17-block count): the 15 added identity skip blocks
  have a zeroed body and **no final relu** (linear bottleneck), so `ivId a = a + 3` for *every* input —
  cleaner than ResNet's `idBlk2` (no nonnegativity bookkeeping). The chain shifts by `+45`, GAP + the
  identity head pass it, so `fwdFull = fwd + 45` and the seal/derivative reduce to
  `MobileNetV2JacobianSeal`'s `Qq`/`g_hasDerivAt` verbatim while the whole-net VJP is genuinely
  composed through all 17 block backward maps (the net-agnostic `chainComp`/`chain_vjp_diff_at` kit).
  Remaining: the BN-CNN — a deep zero-weight witness with no live forward yet.

### Item C — ViT **full-depth, distinct-param** backward — ✅ **DONE**

Turned out the depth-`k` distinct-param tower was **already built and audited** in `ViTDepthK.lean`:
`BlockParamsV` (the 16-field per-block bundle), `vitBodyKVFlat`/`vitBodyKVFlat_has_vjp` (the
`Fin k → BlockParamsV` fold — the authors' "Fin k parameter function" generalization of the
weight-tied `transformerTower`), and the whole-net `vitForwardKV_has_vjp{,_correct}`
(`ViTDepthK.lean:200,244`), unconditional except `0 < ε`. So **C1 was already complete.** The
file's own docstring noted the only remaining piece: *"Depth-12 ViT-Tiny shapes are now a config
change away (the production capstone needs only the P=16/D=192/heads=3 instantiation)."*

- **C2 — `vitTiny_has_vjp_correct`** ✅: added to `ViTDepthK.lean` §3 — `vitForwardKV_has_vjp_correct`
  specialized at the real `vitTiny` spec (`ic=3, H=W=224, P=16, N=196, D=3·64=192, mlpDim=768,
  k=12, nClasses=10`). Builds clean; `#print axioms` = `[propext, Classical.choice, Quot.sound]`;
  added to `tests/AuditAxioms.lean` (gate now 616/616). The ViT peer of `convNextForwardT_has_vjp`
  / `efficientnetForwardB_full_has_vjp`.

Net effect: all three **smooth** flagships have a named full-architecture whole-net backward.

### Item D — realistic input dimensions for the kinked witnesses *(scale hardening)*

The Live witnesses sit at 1×2×2 / 1×32×32. Lift at least `ResNet34Live` toward 3×224×224. The
dim-sensitive obligations are exactly A1/A2 (BN bounds and injectivity scale with `n`); keep them
dimension-generic so the witness construction is parametric and the dims are just instantiation.
Effort: **medium**, mostly re-checking that no discharge secretly used `n ≤ 8` (it shouldn't if
A1 takes the positivity route + a generalized window).

✅ **DONE for ResNet-34, level 2** (`LeanMlir/Proofs/ResNet34LiveRealistic.lean`,
`liveFwd224_has_vjp_correct` + `liveFwd224_nonconstant`, audited 3-axiom-clean). The whole-net
backward + non-vacuity now hold at real **ImageNet 224×224** resolution — the genuine ResNet-34
5-halving pyramid `224 →stem/s2 112 →maxpool/s2 56 →down 28 →down 14 →down 7 →GAP` (the
`conv1 → maxpool → layer2 → layer3 → layer4` skeleton). The audit confirmed the doc's worry is
void: **no discharge used a small `n`** — everything goes through the positivity route
`bnForward_lb : β − |γ|·√n ≤ bn`, so the only change was making `β` parametric (`liveDownβ`,
generalizing `liveDownPC`'s hardcoded `β=20`) and instantiating `β = 64 > √1568` for the
downsamples (largest BN length `2·28·28`) and `β = 160 > √25088` for the stem (`2·112·112`).
Everything else — the channel-diagonal decimating convs, the `Dom2` carrier, `resnet34_has_vjp_at`
— was already dimension-generic and reused verbatim. (2-channel carrier kept; channel-width
realism is orthogonal.)

✅ **ResNet-34 level 3 ALSO DONE at 224×224** (`LeanMlir/Proofs/ResNet34LiveRealisticSeal.lean`,
`R34RealSeal.liveFwd224_jacobian_nonzero` ⇒ `liveFwd224_backward_nontrivial`, audited 3-axiom-clean).
The toy seal's top-left-carrier + "maxpool eventually selects top-left" topology does **not** survive
the **7×7 GAP** (the carrier isn't the output). The fix was a **better witness, not more arithmetic**:
a *uniform* channel-0 perturbation `V224u` (the whole channel, not one coordinate) makes
`channel 0 = channel 1 + δ` at **every** position, so (a) GAP of a uniform difference is `δ` and (b)
`maxpool(ch0) = maxpool(ch1) + δ` holds for **all `t`** by `max(a+δ,b+δ)=max(a,b)+δ` — eliminating the
entire eventual-selection / `Eventually` / continuity block of `ResNet34LiveSeal`. The carrier is a
`UDiff δ u` invariant ("the two channels differ by the uniform `δ`") threaded through the net exactly
like `Dom2`: each BN multiplies `δ` by its positive global `istd` (`bnForward_chan_diff`),
decimate/maxpool/`+1` leave it fixed. So the output difference along the ray is `t · Rr(t)` (`Rr` = a
product of four positive `istd`s) and `g'(0) = Rr 0 ≠ 0` — same final mechanism as both prior seals,
no BN-variance derivative. The result is *cleaner* than the toy seal.

✅ **MobileNetV2 ALSO DONE at 224×224, level 3** (`LeanMlir/Proofs/MobileNetV2SealRealistic.lean`,
`Mnv2RealSeal.fwdR_jacobian_nonzero` ⇒ `fwdR_backward_nontrivial` + `fwdR_has_vjp_correct`,
audited 3-axiom-clean). The obstacle here is *different* from ResNet's: MobileNetV2's **ReLU6** needs
the BN output inside the **bounded** `(0,6)`, so the positivity route (`β` grows with `√n`) would
clamp at 6. The fix is to **scale `γ` down** — `γ = 1/128` keeps `bn ∈ (3 − |γ|√n, 3 + |γ|√n) ⊆ (0,6)`
at `n = 2·112·112` (`(1/128)·√25088 ≈ 2.48 < 3`). The MobileNetV2 weights are all 1×1 channel maps —
dimension-independent — so they're reused verbatim; only the spatial size and `γ` change. The seal
reuses the `UDiff` machinery: the asymmetric stem turns a uniform input perturbation `t` into the
channel difference `−t`, each BN multiplies it by `γ·istd`, so the output difference is `−t · Rr`
(`Rr` = four positive `γ·istd`s) and `g'(0) = −Rr 0 ≠ 0`. **Item D is now CLOSED: both ResNet-34 and
MobileNetV2 are level-3-sealed at real ImageNet 224×224.** (New foundation lemma `bnForward_ub`, the
`bnForward_lb` companion, and a γ-general `UDiff_bn_γ` were added along the way.)

### Item E — almost-everywhere backward *(stretch; the principled level-4 target)*

Replace "a witness exists" with "the `*_has_vjp_at` bundle holds off a measure-zero set": the
pre-activations equal a ReLU kink (exactly 0) or produce a maxpool tie only on a null set of
`(x, θ)`. Then `whole_net_has_vjp_ae`: for a.e. input the whole-net backward equals the
`pdiv`-Jacobian. This is the honest answer to "one point isn't the network." Needs
`Mathlib.MeasureTheory` (the kink locus is a finite union of hyperplanes ⇒ measure zero).
Effort: **heavy / research-flavored.** Scope it; don't block A–C on it.

### Item F — connect to deployed weights *(out of pure-ℝ scope; record only)*

Discharging the bundle at the *actual trained* θ on real data is either (i) interval arithmetic
over the committed weights or (ii) a corollary of Item E. This is where "whole-net backward"
meets the ℝ→Float32 / codegen trust boundary (see `formalization.yaml` fidelity §2, §6); track it
there, not here.

---

## 3. Ordering & effort

| Order | Item | Effort | Payoff | Status |
|---|---|---|---|---|
| 1 | **C** ViT distinct-param tower + `vitTiny` capstone | light–med | full-depth ViT-Tiny backward, looks real, no witness risk | ✅ **done** |
| 2 | **B1** generic nonzero-Jacobian seal | light | the missing honesty sentence, reusable | ✅ **done** |
| 2.5 | **B2** `Mnv2Live` seal | med | first kinked witness at level 3 (no longer level-2) | ✅ **done** (`MobileNetV2JacobianSeal.lean`; exact at input 0 via global smoothness) — **now also full 17-block depth** (`MobileNetV2JacobianSealFull.lean`, `fwdFull_jacobian_nonzero`/`_backward_nontrivial`) |
| 3 | **A** ResNet-34 Live + **B2** seal | research | kills the headline "degenerate witness" caveat | ✅ **DONE at level 3, full depth**: `ResNet34LivePC.liveFwd2_*` (level 2) + `ResNet34LiveSeal.liveFwd2_jacobian_nonzero` (level-3 seal) **and** `ResNet34LiveFull.liveFwd2Full_*` — the real `[3,4,6,3]` (16-block) live ResNet-34, level-3 sealed (`liveFwd2Full_jacobian_nonzero` ⇒ `liveFwd2Full_backward_nontrivial`). Non-degenerate, nonzero-Jacobian-sealed, full depth, 2-channel, audited 3-axiom-clean |
| 4 | **B2** BN-CNN Live | light (reuses A) | last degenerate kinked witness retired | |
| 5 | **D** realistic dims | med | scale credibility | ✅ **DONE — r34 AND mnv2 level-3-sealed at 224×224** (`R34RealSeal.liveFwd224_jacobian_nonzero`; `Mnv2RealSeal.fwdR_jacobian_nonzero`, both ⇒ `backward_nontrivial`, 3-axiom-clean). r34 via β-positivity, mnv2 via γ-scaling (ReLU6 window) |
| 6 | **E** almost-everywhere | heavy | the principled statement (stretch) | |

After 1–4, every flagship net has a **full-depth, non-degenerate, nonzero-Jacobian** whole-net
backward — smooth nets unconditionally, kinked nets at a genuine witness. That is the defensible
"the whole network's backward is proven correct" claim, with E as the long-tail upgrade.

**Session boundary.** Item C is the clean, self-contained win (a specialization of an existing
audited theorem — no new analysis). Items A and the *real* part of B (a nonzero-Jacobian seal on
the deep **kinked** witnesses `Mnv2Live` / a new `ResNet34Live`) are genuinely harder: B's generic
bridge `pdiv f x i j ≠ 0 ⇒ backward x (basisVec j) i ≠ 0` is a one-liner via `HasVJP.correct`, but
the *content* is discharging `pdiv ≠ 0` at the actual kinked witness (pushing through the
`bn1_devSum_scale` / `bnIstd_pos` seal chain) — and cannot be derived generically from
`forward ≠ const` (a non-constant map can still have a zero derivative at the witness point; the
honest route is either an explicit directional-derivative computation at the witness, or an MVT
"∃ a nearby point with nonzero Jacobian" — but for kinked nets that MVT point may not be a smooth
point where the VJP holds). That subtlety is the natural place to start a fresh session.

---

## 4. What stays trusted / out of scope

Unchanged from `verified_train_step.md` §"What stays trusted": ℝ→Float32, `iree-compile`, the op
templates, and the *render/codegen* path. This doc is entirely on the **ℝ `Proofs/` side** — the
`HasVJP.correct = pdiv`-Jacobian apex. It does **not** touch the train-step render close (that is
the `*_close.md` axis) and does **not** claim anything about the GPU float computation.

## 5. Definition of done (audit bar)

- Every new theorem `#print axioms`-closes under `[propext, Classical.choice, Quot.sound]` and is
  added to `tests/AuditAxioms.lean` (CI `proofs.yml` three-axiom gate + `comparator.yml` kernel
  recheck). No `sorry`, no project axiom.
- For each kinked net: a witness theorem whose construction uses **nonzero** weights, a
  `*_forward_nonconstant` (level 2) **and** a `*_jacobian_nonzero` (level 3, from Item B).
  ✅ Met for `Mnv2Live` (`mnv2Live_forward_nonconstant` + `mnv2Live_jacobian_nonzero` /
  `mnv2Live_backward_nontrivial`; **full 17-block depth** via `fwdFull_nonconstant` +
  `fwdFull_jacobian_nonzero` / `fwdFull_backward_nontrivial`) **and for the live ResNet-34**
  (`liveFwd2_nonconstant` + `ResNet34LiveSeal.liveFwd2_jacobian_nonzero` /
  `liveFwd2_backward_nontrivial`; full `[3,4,6,3]` via `ResNet34LiveFull`). Open for the BN-CNN.
- ViT: `vitFullTiny_has_vjp_correct` at 12 distinct-param blocks / 3 heads.
- Doc-drift: update `formalization.yaml` fidelity §4 and `README.md` §"Concrete-instance honesty"
  to point at the Live witnesses once they land (they currently say ResNet-34 live is "in progress").
