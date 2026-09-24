# Non-degeneracy seals on the full-width nets — retire the 2-channel proxies

**Standing doc, opened 2026-09-20. ✅ ALL FOUR PACKAGES DONE 2026-09-20.** `Training/BatchSealKit.lean`
(the shared machinery) plus `Nets/ResNet/ResNet34FullBSeal.lean`, `Nets/ResNet/ResNet50FullBSeal.lean`,
`Nets/MobileNet/MobileNetV2FullBSeal.lean` and `Nets/MobileNet/MobileNetV4FullBSeal.lean`. The
1,992 lines of ResNet-34 proxy and the 1,216 of MobileNetV2 proxy are deleted; audit
1,380 → 1,377 → 1,371 → **1,374**, all 3-axiom clean. **Every kinked net in the book is now sealed
on the forward its artifacts run**, and `formalization.yaml` §4 no longer discloses a gap.

⭐ ResNet-50's seal leaves the spatial size a **binder**, so one statement covers both shipped
resolutions (224 px and the 160 px net the 76.66% run trains) — `0 < q` and `q ≤ 7` are all the
witness needs, the bound being the `β = 160` margin against the stem's `2·(16q)²`.

⭐⭐ **The one finding that was not in the plan, from §4.4: a carrier crosses a relu because relu
is the IDENTITY inside the margin window, and swish is the identity on no window at all.** Every
other net's kinks are relu or relu6, so `EDiff` — which tracks only the gap between the two
examples — is enough. MobileNetV4's fused stage is swish, and a gap comes out of it as
`swish(a) − swish(b)`, which is neither a multiple of `a − b` nor even constant over the grid
unless `a` and `b` are. §4.4's fix, and the new kit §12–§14: make the witness's base
**grid-constant** instead of a ramp, carry the two VALUES (`BUnif`) rather than their difference up
to the fused BatchNorm, use the fact that batch BN on a grid-constant slab puts them symmetrically
about `β` (`bnBatchLA_pair`), and hand `EDiff` back a gap `swishGap β u` that is a function of the
half-gap alone. The readout is then `swishGap 160 (uF t 0) · Rr t`, not `t · Rr t`, and the seal
closes with a new `hasDerivAt_mul_of_zero` beside `hasDerivAt_mul_self_zero`. MobileNetV4 has no
pool, so nothing wanted the ramp. ⚠ The plan had called swish harmless because it contributes no
*clause*; it contributes no clause and is still the hardest thing in the package.

Corrections this doc needed, all applied below: the centre tap must **broadcast** (§3.1), the
carrier's BN count is **one per projection plus the stem** — five for ResNet-50, not four (§3.2),
`MobileNetV2SealRealistic` imported the R34 proxy seal (§5), and every collapse lemma has to be
proved at **variable** shapes (§3.5 — it is the one thing that could have sunk 4.3–4.4). The 4.3
inventory added four more: MobileNetV2's carrier threads **22** BNs, not four, because its
channel-changing blocks have no skip; the apex takes **19** clause bundles of **three** kinds; the
kit needs **seven** new lemmas, not three; and `relu6_id_window` lived inside the namespace the
retirement deletes. §4.4 retired two more: "the UIB relu-site inventory is the whole uncertainty"
was **wrong** (every MobileNetV4 clause is a relu on a BatchNorm output, so the bundle is
weight-only), and "the `CertLayer` composition with no `_eq_chain` is this package's real
uncertainty" was **also wrong** — `CertLayer.comp_fwd_apply` and `MobileNetV4FullB.lean`'s five
group `*_fwd_apply` lemmas already peel at variables, so the collapse chain was the cheapest part
of the file. The real work was the swish, which the plan never named.

## 0. The finding

ResNet-34 exists three times in the proof tree:

| | what | whole-net theorem | non-degeneracy |
|---|---|---|---|
| parametric apex `Nets/ResNet/ResNet34.lean` | generic in stem / pool / chains / downsamples | ∀ pieces with VJPs | n/a — a lemma |
| live 2-channel family `Nets/ResNet/ResNet34Live*`, `Training/ResNet34Live*Seal` (7 files, 1,992 lines) | structural instantiation of the apex: zero bodies, identity projections, whole-vector BN | at an exhibited point | levels 2 + 3, split across a depth × resolution grid no cell of which has both |
| full-width batched `Nets/ResNet/ResNet34FullB` + `FullBVJP` + `BackCertifiedTieB` + `StepTieB` | 64→512, batch BN, 7×7/s2 stem, 3×3/s2 pool, [3,4,6,3]; the ImageNet artifacts | `resnet34ForwardB_full_has_vjp_at`: ∀ `w`, ∀ `x`, under 32 relu clauses + a stem clause + a pool no-tie, all pointwise in `(w, x)` | none |

The seals sit on the proxy for a historical reason. The live family was built in August
(`planning/archive/whole_network_backward.md`, Items A/B/D) to show the *apex's* clauses are
jointly satisfiable with a nonzero Jacobian; two channels because a 1-channel net with BN before
GAP is provably constant. The full-width batched tier landed in September with the same kind of
clauses, and the question was never re-asked of it. MobileNetV2 has the same split — `Mnv2Live`
(`Nets/MobileNet/MobileNetV2.lean:531-918`, on the per-example, whole-vector-BN, two-block
`mobilenetv2Forward`) plus three `Training/MobileNetV2*Seal*` files — against
`mobilenetv2ForwardB_full`. ResNet-50 and MobileNetV4 have no witness at all. `formalization.yaml`
§4 discloses exactly this.

The fix is not to fill the proxy's grid (full depth × 224 on the live net, ~200 lines): it would
be thrown away. It is to seal the nets the artifacts run, at structural weights, and retire the
proxies. Everything transfers except the carrier (§3.2), which the real BatchNorm forces to change.

## 1. The bar

Unchanged from archive §0. For a kinked net (relu / relu6 / maxpool) the honest targets are:

1. **Conditional apex** `*_has_vjp_at` — the VJP at any `(w, x)` whose pre-activations dodge the
   kinks. Done, full depth, for every net.
2. **A non-degenerate witness** — one concrete `(w, x)` that discharges the whole clause bundle
   with genuine nonzero weights and has a non-constant forward.
3. **A nonzero-Jacobian seal** at that witness: `fderiv ℝ f x ≠ 0`, hence via
   `Training/JacobianSeal.lean` (`HasVJPAt.backward_nontrivial_of_fderiv_ne`) the proven backward
   is not the zero map there.
4. Almost-everywhere — not attempted, not in scope.

What this doc changes: levels 2 and 3 are to be stated **on `*ForwardB_full`**, at a
`(w : <Net>BWeights nCls, x : Vec (N * (3 * 224 * 224)))` of the real record type, with `N = 2`
(§3.2 says why not 1). Weights are structural, not trained — the trained-weight version is a
numeric fact about millions of floats (archive Item F) and is what the training runs evidence.

## 2. Census of the seven book nets

| net | activations | whole-net VJP | pointwise clauses | witness today | action |
|---|---|---|---|---|---|
| ResNet-34 | relu, maxpool | `resnet34ForwardB_full_has_vjp_at` | 32 (`R34IdSmoothAt`/`R34DownSmoothAt`, two relus each) + `R34StemSmoothAt` + `R34PoolSmoothAt` | ✅ `ResNet34FullBSeal` | §4.1 done |
| ResNet-50 | relu, maxpool | `resnet50ForwardB_full_has_vjp_at` (`q` binder: 224 and 160 px) | 48 (`R50IdSmoothAt`/`R50ProjSmoothAt`/`R50DownSmoothAt`, three relus each) + stem + pool | ✅ `ResNet50FullBSeal`, both resolutions | §4.2 done |
| MobileNetV2 | relu6 | `mobilenetv2ForwardB_full_has_vjp_at` | 19 bundles: stem + `IVNoExpSmoothAtB` + `IVStridedSmoothAtB` ×4 + `IVSmoothAtB` ×12 + head, 35 relu6 sites, each a window `≠ 0 ∧ ≠ 6` — all weight-only | ✅ `MobileNetV2FullBSeal`; the `Mnv2Live` proxy is deleted | §4.3 done |
| MobileNetV4-Conv-M | relu (UIB), swish (fused stage) | `mobilenetv4ForwardB_full_has_vjp_at` | one bundle `Mnv4SmoothAt`, 8 fields; per-group `.ok` unfolds per row; `fused` vacuous (swish); ⭐ every clause a relu on a BN output, so weight-only | ✅ `MobileNetV4FullBSeal` | §4.4 done |
| EfficientNet-B0 | SiLU, sigmoid (SE) | `efficientnetForwardB_full_has_vjp` | none — `HasVJP`, only `0 < ε` | — | nothing |
| ConvNeXt-T | GELU, LN | `convNextForwardTChB_has_vjp_at` | none — holds at every `x`, only `0 < ε` | — | nothing |
| ViT-Tiny | GELU, softmax, LN | `vitTiny_has_vjp_correct` | none — only `0 < ε` | — | nothing |

**Why the three smooth nets need nothing.** A statement quantified over every weight and every
input has no witness to be degenerate; the non-degeneracy program exists for pointwise
statements, whose hypotheses could in principle be jointly unsatisfiable. The only thing a seal
could add for a smooth net is "some instance has a nonzero Jacobian", which is a fact about the
architecture, not about the theorem: `*_correct` pins the backward to the `pdiv` Jacobian and
`chk_pdiv_is_fderiv` pins `pdiv` to `fderiv`, so a Jacobian that vanished for every weight would
be a bug in the forward definition, which the faithfulness tier and the training runs exclude.
Archive §1a reached the same conclusion; nothing since has changed it.

## 3. The construction (shared by the four kinked nets)

### 3.1 Structural weights

The live nets' recipe, typed at the real record (`R34BWeights nCls` etc.):

* every residual-body kernel and bias zero. A zero kernel gives a constant-zero pre-BN
  activation; batch BN of a constant channel is `β` (variance 0, `xhat = 0`), so the body is a
  constant regardless of `γ` — simpler than the live nets, which needed `γ = 0` for this. Set the
  body's last `β` to `1` (ResNet) or `3` (MobileNetV2, the relu6 centre): the block is the affine
  shift `a ↦ a + 1` on a nonnegative activation (`relu (a + 1) = a + 1`; `ResNet34LiveFull.idBlk2_eq`)
  or `a ↦ a + 3` for every input on a linear bottleneck (`MobileNetV2JacobianSealFull.ivId_eq`);
* every stem / projection / channel-changing kernel a **centre-tap broadcast from channel 0**:
  `W o i (kH/2) (kW/2) = if i = 0 then 1 else 0`, bias zero — EVERY output channel a decimated copy
  of input channel 0 (`BatchSeal.ctK`). ⛔ Not the `o = 0 ∧ i = 0` diagonal this doc first wrote:
  that leaves the stem's other 63 channels constant, and a constant channel ties every 3×3 window
  of the pool. `R34PoolSmoothAt` quantifies over channels, so the tap has to reach all of them.
  (For the three 1×1 projections either form works — no pool follows them — but one kernel for all
  four sites is cheaper, and it keeps the carrier's step uniform: `δ' o = s · δ 0` at every `o`.)
* every `β` positive with the margin `|γ| · √(N·h·w) < β` (relu) or `|γ| · √(N·h·w) < 3` with
  `β = 3` (relu6); `ε = 1` everywhere;
* the head dense `Wd` with `Wd 0 0 = 1`, otherwise zero, `bd = 0`.

### 3.2 The carrier is a batch difference, not a channel difference

⭐⭐ This is the one place the live construction does not transfer. `StableHLO.bnBatchLA` is
`bnBatchTensor4`, which is `bnPerChannelFlat oc (N * (h * w))`: **each channel is normalized over
all its `N·h·w` cells.** Two consequences:

* a channel-uniform offset (the live nets' `channel 0 = channel 1 + δ` carrier) is subtracted
  exactly by the per-channel mean. The `cd` / `UDiff` carrier does not survive one BN;
* at `N = 1` the structural net is constant in its input: the last block's output is
  `bn_p(proj) + const`, and GAP over `h·w` of a per-channel-centred activation is its `β`. This is
  the batch analogue of the 1-channel obstruction that forced two channels in August. So `N = 2`
  is forced for the structural witness, and it makes the witness exercise the one op that couples
  examples.

The carrier: `N = 2`, channel 0 of example 0 = channel 0 of example 1 plus `t`, uniformly
(`x = base + t • V`, `V` the indicator of example 0 / channel 0). Threading it:

| op | on the example difference `δ` (channel 0) | lemma to write (per-example ancestor) |
|---|---|---|
| batch BN | `γ₀ · istd₀(t) · δ` — the two examples share `μ` and `σ`, so centring keeps their difference and scaling multiplies it | `bnBatch_exdiff` (`ResNet34LiveRealisticSeal.bnForward_chan_diff_γ`) |
| batch BN, batch-uniform shift `+k` on a channel | removed exactly | `bnBatch_shift` (`ResNet34LiveFull.bnForward_shift`) |
| batch BN, constant channel | `β` | `bnBatch_const` (`bnForward_const`, `Architectures/BatchNorm.lean:111`) |
| batch BN, positivity | `β − |γ|√(N·h·w) ≤ bn ≤ β + |γ|√(N·h·w)` | `bnBatch_lb` / `bnBatch_ub` (`ResNet34.bnForward_lb`, `MobileNetV2SealRealistic.bnForward_ub`) |
| relu / relu6 inside the margin | identity | `relu_const_pos`, `relu6` window (`Mnv2Live` window lemma) |
| centre-tap stride-2 conv | decimation; `δ` preserved | `flatConvStride2_centreTap` (`ResNet34LivePC.flatConv_WsId2_X2`, `decimate_shift`). ⚠ `flatConvStride2` is symmetric padding (even positions); MobileNetV2's stem is `flatConvStride2Xla` (SAME, reads the odd positions) — one lemma per padding token |
| centre-tap 3×3 / 1×1 conv, depthwise centre tap | identity on channel 0 | `flatConv_centreTap`, `depthwiseFlat_centreTap` (`flatConv_eq_zero`, `MobileNetV2SealRealistic.depthwiseFlat_unit_id`) |
| `maxPool3s2Flat` | `max` of a uniformly shifted window shifts: `δ` preserved | `maxPool3s2_shift` (`ResNet34LiveRealisticSeal.maxPool2_shift`) |
| identity block | `+1` on both examples, batch-uniform: `δ` preserved, and the next BN removes the `+1` | `idBlk_chain_eq` (`ResNet34LiveFull.idBlk2_chain_eq`, generic in `h w` already) |
| GAP, dense | `δ`, then `Wd 0 0 · δ` | `UDiff_gap`-style (`ResNet34LiveRealisticSeal.UDiff_gap`, `gap_add_const`) |
| `batchMap` | every per-example op above distributes over the batch | `StableHLO.batchMap` unfolding |

⭐ **The carrier has one BN per projection, plus the stem.** ResNet-34 has three projections and
so four; ⛔ ResNet-50 has **four** — stage 1 block 0 projects too, at stride 1 (`64 → 256` at
unchanged resolution) — and so five. Count projections in the net, not downsamples.

⭐ As built, the invariant is `EDiff (δ : Fin c → ℝ)` — a **per-channel** offset, not one scalar.
That is what makes it cheap: BN multiplies channel `c`'s offset by `γ_c · istd_c` with no need to
prove the channels share an `istd`, the centre-tap conv collapses the whole function to
`fun _ => s · δ 0`, and only `δ 0` is ever read (by the head, and at each projection conv). The
identity blocks and the pool pass `δ` through untouched, so the trunk carries exactly FOUR values.

So the class-0 output difference between the two examples along the ray is `g(t) = t · R(t)`,
`R` a product of one `γ·istd(t)` per BN on the channel-0 path (ResNet-34: the stem and the three
projections, four factors), `R` continuous and `R(0) > 0`. The product-rule cross term carries a
factor `t` and vanishes at `0` exactly as in every existing seal, so `g'(0) = R(0) ≠ 0` and
`JacobianSeal.fderiv_ne_zero_of_ray` gives `fderiv ℝ (net w) x₀ ≠ 0` with the functional
"example 0 class 0 minus example 1 class 0". No BN-variance derivative is ever taken.

### 3.3 Discharging the clause bundle at the witness

Every clause is one of three shapes, and each has a one-lemma discharge at the structural
weights and the ramp input:

* **relu off the kink** (`bnBatchLA … ≠ 0`, and the post-residual `residual … ≠ 0`): the channel-0
  activation is inside `(β − |γ|√m, β + |γ|√m)` with `β > |γ|√m`; every other channel is the
  constant `β > 0`; a post-residual site is `proj + 1 > 0`;
* **relu6 window** (`≠ 0 ∧ ≠ 6`): `β = 3`, `|γ|√m < 3` (`m = 2·112·112 = 25,088`, `√m < 159`, so
  `γ = 1/64` works; `MobileNetV2SealRealistic` used `1/128` at `m = 12,544`);
* **pool no-tie** (`R34PoolSmoothAt`, per example): the stem input is a strictly decreasing
  per-channel ramp on both examples (`ResNet34LiveSeal.Ys`-style, position-injective), the stem
  conv decimates it and BN/relu are strictly monotone on the centred values, so every 3×3 window
  has distinct entries; the uniform `+t` on example 0 preserves strictness. `MaxPool3s2Smooth`
  (`Architectures/MaxPool3s2.lean:177`) is stated per window, which is exactly this.

MobileNetV2 and MobileNetV4 have no max-pool (checked: zero mentions in either `*FullB.lean`),
so their seals need no no-tie argument at all.

### 3.4 Statements to add, per net

With `<net>` ∈ {`resnet34`, `resnet50`, `mobilenetv2`, `mobilenetv4`}, in a new file
`Nets/<family>/<Net>FullBSeal.lean`:

```
noncomputable def <net>SealW (nCls) : <Net>BWeights nCls        -- §3.1
noncomputable def <net>SealX (t : ℝ) : Vec (2 * (3 * 224 * 224)) -- ramp + t • V
theorem <net>Seal_clauses : <every hypothesis of *_has_vjp_at at (SealW, SealX 0)>
noncomputable def <net>ForwardB_full_seal_has_vjp_at : HasVJPAt (<net>ForwardB_full 2 SealW) (SealX 0)
theorem <net>ForwardB_full_nonconstant : <net>ForwardB_full 2 SealW (SealX 1) ≠ <net>ForwardB_full 2 SealW (SealX 0)
theorem <net>ForwardB_full_jacobian_nonzero : fderiv ℝ (<net>ForwardB_full 2 SealW) (SealX 0) ≠ 0
theorem <net>ForwardB_full_backward_nontrivial : ∃ j₀ i₀, (…seal_has_vjp_at).backward (basisVec j₀) i₀ ≠ 0
```

The class count is a binder (`nCls`, with `Wd 0 0 = 1` needing `0 < nCls`); ResNet-50's `q` is
instantiated at `7` (224 px). The kit (§3.2's lemma column) goes in one shared file,
`Foundation/BatchSealKit.lean` or next to `bnBatchTensor4` in `Architectures/PerChannelBN.lean` —
the executor's call; what matters is that ResNet-50 and the MobileNets import it rather than
re-prove it.

### 3.5 ⚠⚠ Prove every collapse at VARIABLE shapes, instantiate afterwards

The single thing that can sink a package. `relu_id_of_pos` applied directly to
`cbReluStridedB 2 (h := 2*56) … x` leaves the **kernel** a defeq between two numeral-shaped
compositions, and it dies: "deep recursion" at `oc = 64, h = 112`, a deterministic timeout already
at `h = 16`, and 14 s of kernel time even at `h = 8`. The same statement with `N, ic, oc, h, w, kH,
kW` all variables elaborates and kernel-checks instantly, and *instantiating a proved lemma is
substitution* — no defeq at all. So: every block collapse, every nonnegativity, every clause
bundle is a lemma at variables (`sealIdB_eq`, `sealDnB_eq`, `r34StemB_eq`, `sealIdSmooth`, …), and
the witness's numerals appear only in one-line applications of them. The same rule is why
`sealStem_eq` was dropped in favour of a generic `r34StemB_eq`.

⚠ Second trap, same family: `2 * ?h =?= 56` is nonlinear, so unification cannot solve it. Any
lemma whose spatial dims are implicit and only reachable through a `Vec (… (2*h) …)` argument needs
`(h := 28) (w := 28)` passed explicitly, or the elaborator burns its budget in `whnf` (all three
downsample steps of the carrier chain hit this).

## 4. Work packages

### 4.1 ResNet-34 — DONE 2026-09-20

Ops on the path: `cbReluStridedB` (7×7/s2 stem), `maxPool3s2Flat` (via `batchMap`), `r34IdB`
(3×3 `cbReluB` + `projB`, `residual`, relu), `r34DownB` (`cbReluStridedB` + `projB`,
`projStridedB` 1×1/s2, `residualProj`, relu), `r34HeadB` (GAP + dense). Clauses: 32 + stem + pool.
Carrier BN sites: 4.

Mechanically this is `ResNet34LiveFull` + `ResNet34LiveRealisticSeal` re-instantiated: the
collapse `net = short-chain + const` (`liveFwd2Full_eq_add2` pattern, `ld_absorb`), the
ReLU-free twin along the ray (`liveFwd224S`), `Rr`-positivity, `gd_hasDerivAt`, the two seal
theorems. Effort: the live family is 1,992 lines for the same argument at 2 channels with
1×1 decimations; expect the same order, 1.5–2.5k lines including the kit, over two sessions —
the kit and the clause discharge first, the ray argument second. What inflates it: the 7×7
symmetric-padding centre-tap lemma if `flatConvStride2` unfolds badly at kernel size 7, and the
`maxPool3s2` no-tie proof over the batched, left-assoc index (`R34PoolSmoothAt` is per row of
`Mat.unflatten`).

Retired: `Nets/ResNet/ResNet34Live2` (84), `ResNet34LivePC` (499), `ResNet34LiveFull` (328),
`ResNet34LiveRealistic` (140), `ResNet34LiveGeneric` (108), `Training/ResNet34LiveSeal` (491),
`Training/ResNet34LiveRealisticSeal` (342): 1,992 lines. Stays: `Nets/ResNet/ResNet34.lean` (the
apex; consumed by `ResNet34BackCertifiedTie.lean` and `VerifiedNets.lean`) and
`Training/JacobianSeal.lean` (the bridge). `ResNet34LiveGeneric`'s "∀ downsample kernels"
generality is subsumed: the batched apex is already ∀ `w`.

⚠ `resnet34_has_vjp_at`, the *parametric* per-example apex, now has no concrete instantiation —
the proxies were it. It stays audited as the skeleton and as `ResNet34BackCertifiedTie`'s fold
target, and its docstring says so; the clause bundle it shares with the batched apex is what
`ResNet34FullBSeal` discharges.

**What landed.** `Training/BatchSealKit.lean` (494 lines): the `bcell` cell view, the index bridge
(`laIdx_cast` / `bnRowLA_apply` / `bnBatchLA_bcell`) and `bnBatchLA_pointwise`, then the BN
consequences (`bnBatchLA_const`, `bnBatchLA_abs_sub_le`/`_pos`, `bnBatchLA_exdiff`,
`bnBatchLA_cell_inj`), `ctK` and its conv values, `maxPool3s2_shift`, `globalAvgPool_shift`, and
the continuity odds and ends. `Nets/ResNet/ResNet34FullBSeal.lean` (1,194): weights, the collapses,
the 35 discharged clauses, `sealVJP`, `sealDiffAt`, the `EDiff` chain, and
`sealX_nonconstant` / `sealX_jacobian_nonzero` / `sealX_backward_nontrivial`. ⭐ The clause
discharge turned out nearly **input-independent**: the identity block's mid-relu sees the constant
`β₁ = 1` and the downsample's two clauses are weight-only, so only the pool's no-tie and the
identity blocks' post-residual clause touch the activation at all (the latter through
`0 ≤ activation`, which is `relu_nonneg`). No `bnBatchLA_shift` lemma was needed: the carrier is
transparent to a batch-uniform `+1` without the BN having to remove it.

### 4.2 ResNet-50 — DONE 2026-09-20

`Nets/ResNet/ResNet50FullBSeal.lean` (1,135 lines), on `resnet50ForwardB_full` itself. Retires
nothing (no proxy existed); closes the "ResNet-50 has no witness" gap, which §4.1 had made the yaml
disclose.

**What was actually new**, beyond bookkeeping at three convs per block:

* **`q` stays a binder.** The seal takes `0 < q` and `q ≤ 7` and covers 224 px *and* 160 px in one
  theorem — better than this doc's "instantiated at `q = 7`", and the margin is the only thing that
  wants a bound at all. ⚠ Every shape is written as the net's own `2 * (…)` nest, never a product
  like `8 * q`, for the reason `ResNet50FullB.lean`'s header records;
* **a stride-1 projection.** Stage 1 block 0's skip is `projB`, so the carrier needs the kit's
  stride-1 `EDiff_conv` beside the strided one — and it is the fifth carrier BN (§3.2);
* **three relu clauses per block, two of them weight-only.** The bottleneck's `hm2` sits after the
  3×3, but with `W₂` zeroed it sees a constant channel, so it is `β₂ = 1 ≠ 0` and needs nothing of
  the activation. As at ResNet-34, only the post-residual clause does, and only through `0 ≤ ·`.

**The kit absorbed the shared half** on the way (this is what makes 4.3/4.4 cheap):
`Training/BatchSealKit.lean` now holds `kv`/`zk`, the `margin160` check, the ray
(`rayRamp`/`rayBase`/`rayV`/`rayX` and `EDiff_rayX`), the carrier `EDiff` with all five per-op
steps, the stem's centre-tap conv with its positional injectivity and the pool's no-tie
(`ctConv`/`ctConv_inj`/`ctConv_pool_smooth`), and the head (`head_diff_ct`) — all generic in the
shapes. ResNet-50 reuses ResNet-34's block-level generics by name (`projB_zero_const`,
`cbReluStridedB_eq`, `r34StemB_eq`, `sealProj`, the four `*_continuous`), which is the same reuse
`ResNet50FullB.lean` already makes of `r34StemB` and `r34HeadB`.

### 4.3 MobileNetV2 — relu6, no pool, XLA-padded stem

**Inventory DONE 2026-09-20**, read off the nesting in `Nets/MobileNet/MobileNetV2FullB.lean` and
the hypothesis list of `MobileNetV2FullBVJP.lean`'s apex — not from prose, which is how §3.2's
ResNet-50 BN count went wrong.

| # | block | kind | `ic→mid→oc` | grid | role | BN | relu6 | `m = 2·h·w` at each relu6 BN |
|---|---|---|---|---|---|---|---|---|
| — | stem | `mnv2StemB`, 3×3/s2 `flatConvStride2Xla` | 3→32 | 224→112 | **carrier** | 1 | 1 | 25 088 |
| b1 | `IVWNoExp 32 16` | `mnv2NoExpB` (dw3×3, proj1×1) | 32→16 | 112 | **carrier** | 2 | 1 (dw) | 25 088 |
| b2 | `IVW 16 96 24` | `mnv2StridedB` | 16→96→24 | 112→56 | **carrier** | 3 | 2 | 25 088 (e), 6 272 (d) |
| b3 | `IVW 24 144 24` | `mnv2ResidB` | 24→144→24 | 56 | residual | 3 | 2 | 6 272, 6 272 |
| b4 | `IVW 24 144 32` | `mnv2StridedB` | 24→144→32 | 56→28 | **carrier** | 3 | 2 | 6 272 (e), 1 568 (d) |
| b5, b6 | `IVW 32 192 32` | `mnv2ResidB` ×2 | 32→192→32 | 28 | residual | 6 | 4 | 1 568 |
| b7 | `IVW 32 192 64` | `mnv2StridedB` | 32→192→64 | 28→14 | **carrier** | 3 | 2 | 1 568 (e), 392 (d) |
| b8, b9, b10 | `IVW 64 384 64` | `mnv2ResidB` ×3 | 64→384→64 | 14 | residual | 9 | 6 | 392 |
| b11 | `IVW 64 384 96` | `mnv2ExpOnlyB` | 64→384→96 | 14 | **carrier** | 3 | 2 | 392 |
| b12, b13 | `IVW 96 576 96` | `mnv2ResidB` ×2 | 96→576→96 | 14 | residual | 6 | 4 | 392 |
| b14 | `IVW 96 576 160` | `mnv2StridedB` | 96→576→160 | 14→7 | **carrier** | 3 | 2 | 392 (e), 98 (d) |
| b15, b16 | `IVW 160 960 160` | `mnv2ResidB` ×2 | 160→960→160 | 7 | residual | 6 | 4 | 98 |
| b17 | `IVW 160 960 320` | `mnv2ExpOnlyB` | 160→960→320 | 7 | **carrier** | 3 | 2 | 98 |
| — | head | `mnv2HeadB` (1×1 cbrB, GAP, dense) | 320→1280→`nCls` | 7 | **carrier** | 1 | 1 | 98 |

**52 BN sites, 35 relu6 sites**, which is what `MobileNetV2FullB.lean`'s header claims. Seven
non-residual blocks (b1, b2, b4, b7, b11, b14, b17) plus stem and head are the carrier; the ten
`mnv2ResidB` blocks pass it on the skip.

⭐ **The carrier has 22 BN factors**, not four: `1 (stem) + 2 (b1) + 3 × 6 (b2, b4, b7, b11, b14,
b17) + 1 (head)`. MobileNetV2's channel-changing blocks have **no skip at all** — the body *is* the
block — so the carrier threads every BN inside them, where ResNet's carrier saw only the projection.
`Rr` is a 22-fold product of `γ·istd` and `Rr_continuous` is 22 `Continuous.mul`s.

⚠ **Three clause-bundle kinds, not two**, and the head is not hypothesis-free. §2's census row said
"stem + 17 block bundles (`IVSmoothAtB` / `IVNoExpSmoothAtB`)"; the apex actually takes
`MNV2StemSmoothAtB` + `IVNoExpSmoothAtB` (b1) + `IVStridedSmoothAtB` ×4 (b2, b4, b7, b14) +
`IVSmoothAtB` ×12 + `MNV2HeadSmoothAtB` = **19 bundles**. `IVStridedSmoothAtB` differs from
`IVSmoothAtB` in both spatial arguments and in reading `depthwiseStride2FlatXla`, so it is a
separate discharge shape. Positivity is `0 < sε`, `0 < hε`, `IVNoExpPos`, `IVPos` ×16.

⭐⭐ **Every one of the 35 clauses is weight-only.** Better than ResNet-34, where the post-residual
relu needed `0 ≤ activation`. Two reasons compose:

* every relu6 in this net sits directly on a BN output (`cbrB`, `dwbrB`, `dwbrBstrided`, stem, head
  are all `relu6 ∘ bnBatchLA ∘ …`), and `bnBatchLA_abs_sub_le` bounds a BN output within
  `|γ|·√(N·h·w)` of `β` **at every input**. So `β = 3` with `|γ|·√m < 3` gives `≠ 0 ∧ ≠ 6` with no
  reference to the activation;
* the linear bottleneck has **no relu after the residual add** (`mnv2ResidB` returns `residual body`),
  so there is no post-residual clause to discharge and nothing needs a nonnegativity argument.

Consequence: no ramp is needed, no positional injectivity, no `0 ≤ ·` lemmas — the input is only
ever the carrier. ⛔ Do not port ResNet-34's `nn*` nonnegativity layer; it has no analogue here.

**The margin: one `γ = 1/64` for all 52 BN sites.** The widest relu6 BN is `m = 2·112² = 25 088`
(stem, b1's depthwise, b2's expand, all at 112×112), `√25 088 < 158.4`, and `158.4/64 = 2.48 < 3`.
The kit's `margin160` is `γ = 1, β = 160`; this net's peer is `margin192` — `|1/64|·√n < 3` whenever
`(n : ℝ) < 36 864 = (3·64)²` — proved the same way through `sqrt_lt_param`. The eleven project BNs
feed a conv, not a relu6, so they carry no margin at all; `γ = 1/64` there too only for uniformity
(all they need is `γ ≠ 0` for the carrier).

**Structural weights**, typed at `MNV2BWeights nCls`: `ε = 1` and `γ = kv _ (1/64)` everywhere;
`β = kv _ 3` at every BN followed by a relu6, `β = kv _ 0` at the project BNs (unconstrained —
a zeroed residual body then makes the block the exact identity, `EDiff_shift` at `s = 0`); stem,
expand, depthwise and project kernels on the carrier are centre taps; every residual block's three
kernels and biases are zero; `fcW 0 0 = 1`, rest zero, `fcb = 0`.

**Kit lemmas — seven families, not three. LANDED 2026-09-20**, in `Training/BatchSealKit.lean`
(+215 lines, sixteen declarations, `lake build LeanMlir.Proofs.Training.BatchSealKit` green). The
original list missed the depthwise family entirely, and the stem is a *regular* conv at XLA padding,
which is a third op again:

1. `decimateOdd_unflatten` — the odd peer of `decimate_unflatten`, reading `(2i+1, 2j+1)`;
2. `flatConvStride2Xla_ctK`, `bcell_convS2Xla_ctK`, `EDiff_convS2Xla` — the stem (new §3b).
   ⭐ Cheap, as predicted: `flatConvStride2Xla = decimateOddFlat ∘ flatConv`, so these are
   `flatConvStride2_ctK`'s proofs with `decimate_unflatten` swapped for (1), unchanged otherwise;
3. `ctDW` (the centre-tap **depthwise** kernel), `depthwise2d_ctDW`, `depthwiseFlat_ctDW`,
   `bcell_dw_ctDW`, `EDiff_dw` (new §3c) — `depthwiseConv2d`'s pad guard is `conv2d`'s, so
   `depthwise2d_ctDW` is `conv2d_ctK`'s proof with the channel sum deleted. ⭐ The interesting
   difference is semantic, not proof-theoretic: a depthwise **cannot broadcast**, so where
   `EDiff_conv` collapses the carrier to `fun _ => s · δ c₀`, `EDiff_dw` scales `δ` channel by
   channel and the whole function survives;
4. `depthwiseStride2FlatXla_ctDW`, `bcell_dwS2Xla_ctDW`, `EDiff_dwS2Xla` — same, decimated odd;
5. `batchMap_depthwiseFlat_zero` — the residual bodies' depthwise, from the existing
   `depthwiseFlat_eq_zero`. ⛔ No strided zero lemma is needed: all four strided blocks are on the
   carrier, so no zeroed kernel ever meets a stride;
6. `bnBatchLA_window` (`0 < bn < 6`, what the collapses want) and `bnBatchLA_smooth6`
   (`≠ 0 ∧ ≠ 6`, what the clause bundles want) — both from `bnBatchLA_abs_sub_le`, four lines each,
   and between them they discharge all 35 clauses. Plus `margin192` beside `margin160`:
   `|1/64|·√n < 3` whenever `n < 36 864 = (3·64)²`, which clears BOTH window hypotheses at once
   because `β = 3` is the centre of `(0, 6)`;
7. `relu6_continuous`, for `Rr_continuous`. `depthwiseFlat`, `depthwiseStride2FlatXla`,
   `flatConvStride2Xla` and `decimateOddFlat` all carry `@[fun_prop]` differentiability already, so
   `.continuous` covers them and no further continuity lemma is needed.

⚠⚠ **`relu6_id_window` had to move before the retirement, and it did — but not to the kit.** It is
generic in `n` and is exactly the "relu6 is the identity in the window" step the collapses need, yet
it lived **inside** `namespace Mnv2Live`, which 4.3 deletes. ⭐ Resolution taken: move it (and put
`relu6_continuous` beside it) to the **top level of `MobileNetV2.lean`**, next to `relu6` itself and
next to `depthwiseFlat_eq_zero`, which already sits *before* the namespace opens and survives
untouched. That is two lines up rather than into another file: it keeps the lemma beside the op it
is about, keeps `MobileNetV2SealRealistic`'s unqualified uses resolving (it does
`open Proofs Mnv2Live`, and `Proofs` is still an enclosing namespace), and makes the retirement diff
smaller. Nothing anywhere refers to either by a `Mnv2Live.`-qualified name, so the move is
self-contained. ⭐ `Mnv2Live.bnIstd_pos` is only a re-export of `Architectures/BatchNorm.lean`'s,
which is what `BatchSealKit.lean` actually resolves to — no migration, and no hazard there.

⭐ **No new imports were needed.** `Codegen/StableHLO.lean` already imports both
`Architectures/Depthwise.lean` and `Nets/MobileNet/MobileNetV2.lean`, so the kit could already see
`DepthwiseKernel`, `depthwiseFlat`, `depthwiseStride2FlatXla`, `decimateOddFlat`,
`flatConvStride2Xla` and `relu6`. ⚠ The flip side: editing either of those files rebuilds
`StableHLO.lean` (322 s) and everything downstream, so batch kit edits rather than iterating on
them — develop against a scratch file that imports `BatchSealKit` and only then transcribe.

**The seal file LANDED 2026-09-20**: `Nets/MobileNet/MobileNetV2FullBSeal.lean`, 1,418 lines,
221 declarations, rooted in `lakefile.lean` and printed in `tests/AuditAxioms.lean`
(`Mnv2FullBSeal.sealX_nonconstant` / `_jacobian_nonzero` / `_backward_nontrivial`).

What the inventory predicted and the proof confirmed:

* ⭐⭐ **all 35 clauses are weight-only.** `sealVJP` — the whole-net VJP with every one of the 19
  bundles discharged — elaborated on the first attempt, and not one of the 19 `sc*` lemmas reads
  the activation. `bnBatchLA_smooth6` at `β = 3`, `γ = 1/64` does the whole job;
* ⭐ **the residual blocks are the exact identity**, not ResNet's `a ↦ a + 1`: with `pβ = 0` a
  zeroed body is the constant `0`, so `sealResB_eq : mnv2ResidB … v = v` and the ten residual
  `pc`/`ed` steps are two lines each;
* **22 carrier BatchNorms**, as counted. `Rr` is a 22-fold product of `rf = 1/64 · istd`, and
  `gd_ray` closes by `ring` over 23 atoms after unfolding the 22 `δ` definitions.

Three things the plan did not anticipate, all small:

1. ⚠ **`rw` does not see through a structure projection.** `sealExpB_eq`'s `show` had to spell the
   witness's BN parameters as literals (`1`, `kv mid (1/64)`, `kv mid 3`); with `_` placeholders the
   goal keeps `(sealIVW ic mid oc).eε` and `cbrB_eq`'s syntactic pattern misses. Same fix in all
   three block collapses and in `headA`;
2. ⛔ **`repeat' apply mul_pos` splits inside `rf`.** `rf` is itself `1/64 * bnIstd …`, so the
   tactic keeps going and leaves `0 < 1/64` goals that `rf_pos` cannot close. `Rr_pos` is an
   explicit 22-deep `mul_pos (rf_pos _ _) (…)` chain instead, with a comment saying why;
3. each of the three block collapses ends in a bare `rfl` — `projB` is definitionally
   `bnBatchLA ∘ batchMap (flatConv …)`, and the `rw`s leave exactly that.

**Method note for §4.4.** The file was developed against a scratch file importing `BatchSealKit`
and `MobileNetV2FullBVJP` (both already built), in five stages — weights/window/stage collapses,
clause bundles + `sealVJP`, block collapses + continuity, the 22 activations + the collapsed trunk,
the carrier + `Rr` + the seal — each elaborated before the next was written, and the repetitive
two thirds generated from a 17-row site table rather than typed. Total: four errors across
1,418 lines, all of them the shapes listed above. ⚠ Do NOT iterate by editing the kit or
`MobileNetV2.lean`: either rebuilds `StableHLO.lean` (322 s) and everything downstream.

**The base input.** Reuse `rayX (2*112) (2*112)`: the types line up (`Vec (2 * (3 * 224 * 224))`),
`EDiff_rayX` is proved, and since every clause is weight-only the ramp is doing no work beyond
keeping one witness shape across the four nets.

Effort: ~1k lines plus ~150 of kit, one session.

**Retirement DONE 2026-09-20.** Deleted: `Mnv2Live` (384 lines out of
`Nets/MobileNet/MobileNetV2.lean`, which goes 930 → 546), `Training/MobileNetV2JacobianSeal` (255),
`MobileNetV2JacobianSealFull` (209), `MobileNetV2SealRealistic` (368). ⭐ The per-example
`mobilenetv2Forward`, its whole-net VJP, `relu6`, `relu6_id_window`, `relu6_continuous` and
`depthwiseFlat_eq_zero` all stay — only the witness goes, and that file's banner now says what the
survivors are for. Audit **1,380 → 1,371** (nine prints out).

Every place the retired names lived, all edited:

| what | where |
|---|---|
| `main_results` row + §4 prose | `formalization.yaml` |
| `DECLS` + `MODULES`, then the regenerated pair | `scripts/gen_comparator_tier.py`, `tests/comparator/{Challenge,Solution}Tier.lean`, `config-tier.json` |
| nine `#print axioms` + three imports | `tests/AuditAxioms.lean` |
| three roots | `lakefile.lean` |
| prose | `LeanMlir/Proofs/README.md`, `Training/JacobianSeal.lean`, `Training/TrainedMlpWitness.lean` |
| the docstring generator, kept in sync with the file above | `scripts/lipschitz_cert_witness_s8.py` |
| its own worked example of "a namespace named by a suffix" | `tests/DocstringCheckRefs.lean` |
| the audit count | `tests/comparator/README.md`, `blueprint/src/content.tex` |

⚠ **Not every mention went**, deliberately: `planning/archive/*` and `planning/audit_census.md`
record what was true when they were written. ⛔ And whether the rest of `MobileNetV2.lean` (the
per-example two-block net) still has consumers is a census question, not this package's.

### 4.4 MobileNetV4-Conv-M — DONE 2026-09-20

`Nets/MobileNet/MobileNetV4FullBSeal.lean` (1,238 lines), on `mobilenetv4ForwardB_full` itself, plus
205 lines of kit and 19 in `JacobianSeal.lean`. Retires nothing (no proxy existed); closes the last disclosed gap, as §4.2 did.
With it every kinked net in the book is sealed on the forward its artifacts run.

**What the recon got right, and it made two thirds of the file cheap:**

* ⭐⭐ **the clause bundle is weight-only.** Every kink is a relu on a `bnBatchLA` output, in four
  spellings; `projLayer.ok`, `mnv4FusedConvLayer.ok` and `CertLayer.id'.ok` are `True`. One
  `bnBatchLA_pos` at `γ = 1, β = 160, ε = 1` discharges all 54 (a `#guard` counts them off `mnv4Blocks`), and `sealVJP` — the whole-net VJP
  with all eight bundles closed — elaborated on the first attempt;
* ⭐ **and the discharge is generic in the table ROW.** `sealUib_ok` / `sealUibStrided_ok` take a
  `UibSpec` and the four kernels: the `k = 0` slots give `True` by `by_cases`, the rest are the
  same BatchNorm fact. 21 blocks, two lemmas. The same trick gives ONE weight record `sealP s Wq We
  Wd Wz`, instantiated as `sealCT` (centre taps, the three carrier rows) and `sealZ` (zeros, the
  eighteen skipped ones), so every block lemma is proved once;
* **no pool, no ramp, no positional injectivity, no `0 ≤ ·` layer** — as at MobileNetV2.

**⛔ What the recon got WRONG, in both directions.**

1. "The `CertLayer` composition with no `_eq_chain` is this package's real uncertainty" — **no.**
   `CertLayer.comp_fwd_apply` is already proved between variables, and `MobileNetV4FullB.lean`
   already carries `mnv4Res28Layer_fwd_apply` … `mnv4Res7bLayer_fwd_apply` for exactly this reason
   (its own T2 capstone needed them). The head's peel is three lines of `simp only` over
   `comp_fwd_apply` + the four `*_fwd_apply` projections. `pc0`–`pc6` are the shortest collapse
   chain of the four packages: with the project BatchNorm's `β = 0` a zeroed body is the constant
   `0`, so the eighteen skipped rows are the EXACT identity and two of the seven groups collapse to
   nothing at all.
2. ⭐⭐ **"The fused stage: the carrier crosses its BatchNorms but no kink argument is needed
   there" — true and beside the point, and this was the package.** A carrier crosses a relu
   because relu is the *identity* inside the margin window. Swish is the identity on no window.
   `EDiff` carries only the gap between the two examples, and `swish(a) − swish(b)` is neither a
   multiple of `a − b` nor constant over the grid unless `a` and `b` are — so `EDiff` cannot cross
   the fused stage at all, and a ramp base dies there.

**The fix, and the new kit (§12–§14, 217 lines).** Track the VALUES, not the gap, for the five
stages up to the fused BatchNorm:

* the base is **grid-constant**: `sealX t = t • rayV`, example 0's channel 0 lifted by `t`
  uniformly, everything else zero. ⛔ Not `rayX` — MobileNetV4 has no pool, so nothing wants the
  ramp, and the ramp is what breaks;
* `BUnif a v` says each example's slab is constant over the grid, one value per channel. Centre-tap
  convs preserve it (`BUnif_convS2Xla`, `BUnif_convS2`), pointwise activations preserve it
  (`BUnif_map`), and `EDiff_of_BUnif` hands the carrier back on the far side;
* ⭐⭐ `bnBatchLA_pair` is the lemma that makes it work: on a `BUnif` slab the channel's mean is the
  two values' midpoint, so batch BN outputs `β ± γ·(gap/2)·istd` — **symmetric about `β`**. The
  swish's two outputs are then a function of the half-gap `u` alone, and their difference is
  `swishGap β u := swish(β+u) − swish(β−u)`;
* §14 is `swishGap`: `swishGap β 0 = 0`, `HasDerivAt (swishGap β) (2·swish' β) 0`, and
  `0 < swish' β` for every `β ≥ 0` (all three factors of `σ(x)(1+x(1−σ(x)))` are positive there —
  proved from `x/(1+e^{-x})` by the quotient rule, so no sigmoid lemmas are needed). Plus
  `swishScalar_lt` (strictly increasing on the nonnegatives, two lines: numerator up, denominator
  down) for level 2, and `bnIstd_le_one` at `ε = 1` to keep `u` inside the window that needs.

**The readout is therefore `swishGap 160 (uF t 0) · Rr t`, not `t · Rr t`.** `uF t 0 = t · Q0 t`
with `Q0` the two pre-swish BatchNorm factors, so `HasDerivAt` composes:
`hasDerivAt_mul_self_zero` gives `u`, the chain rule gives `swishGap ∘ u`, and a new
`hasDerivAt_mul_of_zero` (`Training/JacobianSeal.lean`, beside its peer — `S · Q` at a zero of `S`,
`Q` merely continuous) closes it. ⚠ Still no BatchNorm variance derivative anywhere: the seventeen
`istd`s enter only as continuous factors, and the swish's slope is the one honest derivative in the
chain.

**⭐ The carrier threads 17 BatchNorms**, counted from the net: `1 (stem) + 2 (fused conv + fused
project) + 4 × 3 (rows 1, 3, 11 — the only channel-changing rows, hence the only ones without a
skip: strided pre-DW, expand, post-DW, project) + 2 (head)`. Two are inside `uF`; `Rr` is the other
fifteen. The eighteen `CertLayer.residual` rows pass the carrier on the skip.

**Two transcription traps, both already in this doc, both bit again:**

* ⚠ **`rw` does not see through a structure projection** (§4.3). `sealCTStrided_eq`'s `show` has to
  spell `ctK s.oc (s.ic * s.expand) 1 1 1` rather than `_`, and every `resid_id` step has to be
  restated at `(sealW nCls).b_k` before rewriting, because `sealZ mnv4Row_k` is only DEFEQ to it;
* ⚠⚠ **the `2 * ?h` nonlinearity** (§3.5's second trap). `BUnif_convS2` without
  `(h := 56) (w := 56)` is a `(deterministic) timeout at isDefEq` at 1,000,000 heartbeats — one
  minute of wall clock for one `refine`. With them it is instant.

⚠ One more, new: `positivity` on a goal mentioning `iS`/`iF` is a `maxRecDepth` failure — those
unfold through the whole activation chain. Explicit `mul_pos`/`div_pos` chains instead.

**Method.** Developed against a scratch file importing `MobileNetV4WholeBackCertifiedTieB` and
`ResNet34FullBSeal`, in six stages, each elaborated before the next was written; the repetitive
two thirds (15 activations, 15 carrier steps, 30 continuity lemmas, `Rr`) generated from a row
table. Total elaboration 3.1 s. The shipped file imports only `MobileNetV4FullBVJP` and
`ResNet34FullBSeal` — the head peel it wanted from the tie file is three lines, so it is local
(`headStack_apply`) and the seal does not depend on the backward tier.

## 5. Bookkeeping that moves with each package

* **`formalization.yaml`.** §4 (lines ~199-207) is rewritten once 4.1 lands: the non-degeneracy
  sentence names the batched seals and drops "no seal has both depth and resolution". The
  ResNet-34 witness row (lines ~368-369) re-points at `<Net>FullBSeal.lean`. 4.3 replaces the
  `Mnv2Live.mnv2Live_forward_nonconstant` `main_results` row with
  `mobilenetv2ForwardB_full_jacobian_nonzero` (or `_backward_nontrivial`), keeping
  `comparator_config: tests/comparator/config-tier.json`. House style: no emoji, one-line comments.
* **Comparator tier.** Every yaml `main_results` row must be in a comparator config
  (`gen_comparator_tier.py --check` enforces it). So a yaml row change is a `DECLS` change:
  edit `DECLS` and `MODULES`, run `python3 scripts/gen_comparator_tier.py` (regenerates and
  elaborates the solution in the parent package, seconds), then `tests/comparator/run.sh`
  (the tools are installed locally since 2026-09-20; ~25 min cold, all three configs).
* **`tests/AuditAxioms.lean`.** Drop the retired prints (`liveFwd2_*`, `liveFwd2Full_*`,
  `liveFwd224_*`, `mnv2Live_*`, `fwdFull_*`, `fwdR_*`), add the new seal theorems. The count in
  the book's comparator appendix (`1{,}380`) follows.
* **`LeanMlir/Proofs/README.md`** lines 76 and 162-170 name `*Live` / `Mnv2Live`.
* **The book** names none of these modules (checked 2026-09-20); its "sealed" language is in the
  yaml only. No book edit beyond the count above.
* **Imports.** ⛔ The claim that only `tests/AuditAxioms.lean` imports the live files was WRONG:
  `Training/MobileNetV2SealRealistic.lean` imported `ResNet34LiveRealisticSeal` for four
  per-example decls (`UDiff`, `UDiff_bn_γ`, `UDiff_gap`, `flatConv_diag_id`). Resolution in 4.1:
  those four moved INTO `MobileNetV2SealRealistic.lean` (they are per-example 2-channel facts and
  die with it in 4.3), while `sqrt_lt_param`, `bnIstd_cont` and `bnForward_chan_diff_γ` — general
  BN facts — moved into `BatchSealKit.lean`, which that file now imports. Also re-pointed: two
  docstrings in `Nets/ResNet/ResNet34.lean` cited the deleted modules (caught by
  `lake exe docstring-checkrefs`), and `scripts/lipschitz_cert_witness_s8.py`'s prose.
  `scripts/check_target_names.sh` and `python3 scripts/check_audit_coverage.py` after.
* **The audit count** went 1,380 → 1,374 at 4.1 (nine proxy prints out, three seal prints in),
  1,374 → 1,377 at 4.2, 1,377 → 1,371 at 4.3 (nine proxy prints out) and 1,371 → **1,374** at 4.4,
  in `tests/comparator/README.md` and `blueprint/src/content.tex`.
* **4.4's bookkeeping, all done**: the `main_results` row
  (`Proofs.Mnv4FullBSeal.sealX_backward_nontrivial`), the §4 prose (the "MobileNetV4 has no witness
  yet" sentence is gone), `DECLS` + `MODULES` and the regenerated comparator pair, three
  `#print axioms` plus one import in `tests/AuditAxioms.lean`, one `lakefile.lean` root, and
  `LeanMlir/Proofs/README.md`'s two seal paragraphs.

## 6. Gates

The standard set: `lake build Certs`, `lake env lean tests/AuditAxioms.lean` (3-axiom clean),
`lake exe docstring-checkrefs`, `python3 scripts/check_audit_coverage.py`,
`python3 scripts/check_render_coverage.py`, `git diff verified_mlir/` empty,
`bash scripts/check_target_names.sh`. Plus, whenever `DECLS` or the yaml changes,
`python3 scripts/gen_comparator_tier.py --check` and `tests/comparator/run.sh`. Land by
fast-forward, never a merge commit; stage, then stop for review before every commit.

## 7. Out of scope

* The three smooth nets (§2).
* Trained-weight witnesses for the big nets (archive Item F): numeric, not symbolic. The MLP/CNN
  rungs already have theirs (`Training/TrainedMlpWitness`, `TrainedCnnSeal`).
* Almost-everywhere correctness (level 4).
* The full-depth × 224 cross on the live net: superseded by 4.1.
* Retiring `Nets/ResNet/ResNet34.lean` or the per-example `MobileNetV2.lean`: both still have
  consumers; census questions for another doc.
