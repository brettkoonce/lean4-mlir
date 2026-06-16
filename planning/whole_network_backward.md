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
   forward**, so the witness is not vacuous. ⚠️ **Only `Mnv2Live` has this; it is tiny and proves
   only `forward ≠ const`, not a nonzero Jacobian.**
3. **A nonzero-Jacobian seal** — at the witness, `∃ i j, pdiv forward x i j ≠ 0`, i.e. the
   *backward map itself* is provably non-trivial there. 🔴 **Missing everywhere.**
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
| **ResNet-34** | `resnet34_has_vjp_at` (`ResNet34.lean:174`) | `ResNet34Concrete.resnet34Concrete_has_vjp_correct` (`:735`) | 🔴 **degenerate** — 16 zero-weight identity blocks, 1×32×32 |
| **MobileNetV2** | `mobilenetv2_has_vjp_at_correct` (`MobileNetV2.lean:580`) | `MobileNetV2Concrete.*` (`:936`) | 🔴 **degenerate** — all-zero kernels, constant output |
| ″ (live) | ″ | `Mnv2Live.mnv2Live_has_vjp_correct` (`:1008`) | ⚠️ **non-degenerate but level-2 only** — `forward ≠ const`, no nonzero-Jacobian seal; 1×2×2 |
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

### Item A — ResNet-34 **Live** witness *(headline; the audit's "Stage 1, in progress")*

Build `ResNet34Live.resnet34Live_has_vjp_correct`, the `Mnv2Live` analogue at ResNet-34 depth.
Pick genuine nonzero weights and discharge `resnet34_has_vjp_at`'s bundle:

- **A1 — ReLU smoothness at every block.** Two reusable routes, both already in the toolbox:
  - *positivity route*: `bnForward_lb` (dimension-robust BN lower bound) with `β` large enough
    that `bn ≥ β − |γ|√n > 0`, so the post-BN pre-ReLU is everywhere positive (ReLU acts as
    identity, smooth). Cleanest for the stem and the down-blocks.
  - *window route*: a `bn13_window`-style two-sided bound, generalized off `n ≤ 8` to the
    ResNet spatial sizes (new lemma `bnAB_window n ε γ β` — same `(zₖ−μ)² < …` algebra, no
    sqrt/variance evaluation). Needed where positivity is too strong an ask.
- **A2 — maxpool no-tie** via `maxPool2Smooth_of_injective` on an injective stem output.
  Injectivity of `flatConvStride2 ∘ (1×1 identity)` of an injective input + `bnForward_injective`
  (γ≠0). This is the heaviest single discharge; isolate it as `r34Stem_injective`.
- **A3 — non-degeneracy.** Choose ≥1 block with genuine identity convs (not zero) so the body is
  non-constant, exactly as `Mnv2Live` uses an identity `block2`. Reuse `vjp_chain_at` for the
  zero-or-identity blocks so depth stays a `List.length` fold.

Effort: **medium-heavy** (the injectivity discharge dominates). No new VJP math.

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
- **B2 — discharge the premise at the DEEP kinked witnesses** *(the remaining, harder part)*:
  exhibit `pdiv forward x i j ≠ 0` (equivalently `fderiv ℝ forward x ≠ 0`) at `Mnv2Live`, a future
  `ResNet34Live` (Item A), and the BN-CNN. This is **not** derivable from `forward ≠ const` (a
  non-constant map can have a zero derivative at the witness); the honest route is an explicit
  directional-derivative computation through the seal chain (`bn1_devSum_scale` gives
  `∂(Σ bn)/∂(deviation) = istd > 0`; push it through the linear head). Per-net, genuinely hairy —
  the natural fresh-session work, now unblocked by B1.

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

### Item D — realistic input dimensions for the kinked witnesses *(optional, scale hardening)*

The Live witnesses sit at 1×2×2 / 1×32×32. Lift at least `ResNet34Live` toward 3×224×224. The
dim-sensitive obligations are exactly A1/A2 (BN bounds and injectivity scale with `n`); keep them
dimension-generic so the witness construction is parametric and the dims are just instantiation.
Effort: **medium**, mostly re-checking that no discharge secretly used `n ≤ 8` (it shouldn't if
A1 takes the positivity route + a generalized window).

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
| 3 | **A** ResNet-34 Live + **B2** seal | med–heavy | kills the headline "degenerate witness" caveat | next (the hairy part) |
| 4 | **B2** BN-CNN Live | light (reuses A) | last degenerate kinked witness retired | |
| 5 | **D** realistic dims | med | scale credibility | |
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
- ViT: `vitFullTiny_has_vjp_correct` at 12 distinct-param blocks / 3 heads.
- Doc-drift: update `formalization.yaml` fidelity §4 and `README.md` §"Concrete-instance honesty"
  to point at the Live witnesses once they land (they currently say ResNet-34 live is "in progress").
