# Non-degeneracy seals on the full-width nets — retire the 2-channel proxies

**Standing doc, opened 2026-09-20.** Four work packages (§4), one commit each, ResNet-34 first
because it builds the kit the other three reuse. Each package is its own session or two. Gates in
§6; the bookkeeping that moves with every package in §5.

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
| ResNet-34 | relu, maxpool | `resnet34ForwardB_full_has_vjp_at` | 32 (`R34IdSmoothAt`/`R34DownSmoothAt`, two relus each) + `R34StemSmoothAt` + `R34PoolSmoothAt` | live 2-channel proxy, on the apex | §4.1 |
| ResNet-50 | relu, maxpool | `resnet50ForwardB_full_has_vjp_at` (`q` binder: 224 and 160 px) | 48 (`R50IdSmoothAt`/`R50ProjSmoothAt`/`R50DownSmoothAt`, three relus each) + stem + pool | none | §4.2 |
| MobileNetV2 | relu6 | `mobilenetv2ForwardB_full_has_vjp_at` | stem + 17 block bundles (`IVSmoothAtB` / `IVNoExpSmoothAtB`), each a window `≠ 0 ∧ ≠ 6` per site | `Mnv2Live` proxy, per-example two-block net | §4.3 |
| MobileNetV4-Conv-M | relu (UIB), swish (fused stage) | `mobilenetv4ForwardB_full_has_vjp_at` | one bundle `Mnv4SmoothAt` with per-group `.ok` fields; `fused` is vacuous (swish) | none | §4.4 |
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
* every stem / projection / channel-changing kernel a **centre-tap diagonal on channel 0**:
  `W o i (kH/2) (kW/2) = if o = 0 ∧ i = 0 then 1 else 0`, bias zero. Channel 0 carries a decimated
  copy of its input; every other channel is constant zero pre-BN, hence `β` post-BN, hence a
  positive constant that carries nothing and is off every kink;
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
`Foundation/BatchSealKit.lean` or next to `bnBatchTensor4` in `Foundation/PerChannelBN.lean` —
the executor's call; what matters is that ResNet-50 and the MobileNets import it rather than
re-prove it.

## 4. Work packages

### 4.1 ResNet-34 — first, builds the kit

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

Retires: `Nets/ResNet/ResNet34Live2` (84), `ResNet34LivePC` (499), `ResNet34LiveFull` (328),
`ResNet34LiveRealistic` (140), `ResNet34LiveGeneric` (108), `Training/ResNet34LiveSeal` (491),
`Training/ResNet34LiveRealisticSeal` (342): 1,992 lines. Only `tests/AuditAxioms.lean` imports any
of them (checked 2026-09-20). Stays: `Nets/ResNet/ResNet34.lean` (the apex; consumed by
`ResNet34BackCertifiedTie.lean` and `VerifiedNets.lean`) and `Training/JacobianSeal.lean` (the
bridge). `ResNet34LiveGeneric`'s "∀ downsample kernels" generality is subsumed: the batched apex
is already ∀ `w`.

### 4.2 ResNet-50 — same ops, bottleneck bodies

`r50IdB` / `r50ProjB` / `r50DownB` (1×1, 3×3, 1×1 with three relus; `projStridedB` on the
downsample blocks), the same stem and pool as 4.1, `q = 7`. 48 clauses + stem + pool; carrier
sites 4 (stem + three projections; a bottleneck's zeroed body contributes a constant the next BN
removes, exactly as a basic block's does). Nothing new beyond bookkeeping at three convs per
block. Effort: ~1k lines, one session, once 4.1's kit exists. Retires nothing (no proxy existed);
closes the "ResNet-50 has no witness" gap, which the yaml does not currently even disclose.

### 4.3 MobileNetV2 — relu6, no pool, XLA-padded stem

Blocks in `mobilenetv2ForwardB_full`: stem, one `mnv2NoExpB`, two `mnv2ExpOnlyB`, four
`mnv2StridedB`, ten `mnv2ResidB`, head. The carrier passes through the stem, the no-expansion
block and the six channel-changing blocks (expand / depthwise / project BNs, all centre-tap
diagonal on channel 0); the ten residual blocks pass it on the skip with a `+3` body the next BN
removes. Differences from 4.1: the relu6 window in place of the positivity margin (§3.3); the
stem is `flatConvStride2Xla` (odd positions); a depthwise centre-tap identity lemma; no pool, so
no no-tie argument and no ramp — the base input can be the constant `0` on both examples as in
every existing MobileNetV2 seal (`fwd_jacobian_nonzero` at `0`). Effort: ~1k lines, one session.

Retires: `Mnv2Live` (`Nets/MobileNet/MobileNetV2.lean:531-918`, 388 lines),
`Training/MobileNetV2JacobianSeal` (255), `MobileNetV2JacobianSealFull` (209),
`MobileNetV2SealRealistic` (309): ~1,161 lines. ⚠ `Proofs.Mnv2Live.mnv2Live_forward_nonconstant`
is a `formalization.yaml` `main_results` row (line 104) **and** a comparator tier theorem
(`scripts/gen_comparator_tier.py` `DECLS`, `tests/comparator/config-tier.json`): see §5. Whether
the rest of `MobileNetV2.lean` (the per-example two-block net) still has consumers is a census
question, not this package's.

### 4.4 MobileNetV4-Conv-M — UIB inventory first

One bundle `Mnv4SmoothAt` with fields `stem`, `fused` (vacuous, swish), `g28`, `g14a`, `g14b`,
`g7a`, `g7b`, `head`, each group's `.ok` a `CertLayer` conjunction over its rows. First step is
an inventory of the relu sites per UIB row from `Nets/MobileNet/MobileNetV4BackB0.lean`
(`dwbReluB` and the conv-BN-relu ops) and which rows are channel-changing (carrier path) versus
residual (zeroed body, `+β` shift). No pool. The fused stage is swish and needs the centre-tap
lemma only for the carrier, no kink argument. Effort: 1–1.5k lines; the inventory is the
uncertainty, the rest is 4.3's shape. Retires nothing; closes an undisclosed gap, as 4.2 does.

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
* **Imports.** Deleting the live files breaks only `tests/AuditAxioms.lean`.
  `scripts/check_target_names.sh` and `python3 scripts/check_audit_coverage.py` after.

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
