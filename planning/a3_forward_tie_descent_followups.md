# Post-A3-backward follow-ups: forward symmetry · certified-VJP tie · descent

Three directions worth taking after the A3 backward family was completed (all five nets +
batched enet + the fully-concrete vit whole-net; see `planning/a3_backward_deepnet_assembly.md`).
Written 2026-06-26, right after the backward push. Ordered by **what a skeptic attacks**, not by
ease. Honest framing throughout: the float bridges are `float32 ≈ ℝ` over the proven SHlo graph at a
**smooth point**; per-op StableHLO conformance + IREE lowering + `float32 ≈ ℝ-the-silicon` stay
validated by `iree-compile` + the GPU runs.

---

## §A — The forward whole-net bridges (close the symmetry)

**Why.** The backward is now AHEAD of the forward. We have whole-net BACKWARD bridges
(`r34_grad_floatBridges`, `mnv2_grad_floatBridges`, `convnext_grad_floatBridges`,
`vit_grad_floatBridges_concrete`, the batched enet backward) — but the whole-net FORWARD float
bridges are **block-level only** for the deep nets (r34/mnv2/convnext/enet), and the forward vit
whole-net does not exist. Only the shallow nets have forward whole-nets (`linear_float_close`,
`mlp_float_close`, `cnn_float_close`, `cifar_float_close`, `cifar8_floatBridges`,
`cifarBn_floatBridges`). This is pure asymmetry, not a hard problem.

**What's already in hand (reuse, don't rebuild):**
- Every per-op FORWARD bridge: `floatBridges_flatConv`, `floatBridges_dense`, `floatBridges_relu`,
  `floatBridges_maxPool`, `floatBridges_gap`/`floatClose_gap`, `floatBridges_depthwise`,
  `floatBridges_bn`/`floatBridges_bnPerChannelTensor3` (BN), `floatBridges_flatConvStride2`(?-verify),
  the SE/MBConv body (`floatBridges_mbconvBody`), the ViT block (`floatBridges_vitBlock*`).
- The fold combinators (all direction-agnostic): `FloatBridges.comp`, `FloatBridges.residual`,
  `FloatBridges.biPathSum`, `FloatBridges.perRow`, `FloatBridges.batchMap` (built this session),
  `floatBridges_towerBack` (the depth fold).

**Deliverables.** One forward whole-net per deep net, mirroring the backward blueprint
(concrete endpoints + per-block `FloatBridges` hypotheses folded by `.comp`):
- `r34_float_floatBridges` — reverse of `r34InputGrad`'s structure: `dense ∘ gap ∘ [3,4,6,3] blocks
  ∘ maxpool ∘ stem`, blocks supplied (discharge with the forward block bridges).
- `mnv2_float_floatBridges`, `convnext_float_floatBridges` — same pattern (the ch7 render targets).
- `enet_float_batchedFloatBridges` — `FloatBridges.batchMap N (per-example net)` (the lift now
  exists; the per-example net is the MBConv `.comp` chain).
- `vit_float_floatBridges` — `head ∘ finalLN ∘ towerBack(forward blocks) ∘ patchEmbed`, i.e. the
  forward analogue of `vitGradFlat` (reuse `floatBridges_towerBack` + the forward patch-embed/head).

**Effort: LOW-MEDIUM. Risk: LOW.** Pure `.comp` folds; the only friction is the `maxRecDepth
100000` + incremental-`have` gotcha on the ≳15-op chains (carried from `r34_grad_floatBridges`), and
the grep-miss naming (`<net>_float_*`, a `FloatModel.` method, NOT `*forward*`). A forward patch-embed
float bridge (the `patchEmbed_flat` forward, an im2col strided conv) is the one new per-op piece vit
needs — small, the analogue of `floatBridges_patchEmbedBack`.

---

## §B — Does the backward target the *certified* gradient? (the integrity check)

**Why this is the load-bearing one.** The backward float bridges prove **deployed-float ≈ (a
hand-assembled real backward map)**. For the LEAF ops the real map IS the certified object
(`patchEmbed_input_grad_formula`, `sdpa_back_{Q,K,V}` — bridged directly; `convFlatBack =
conv2d (reverseSwap W) 0`, `decimateBack = decimateFlat_has_vjp.backward` by `rfl`). But the
ASSEMBLIES — `r34IdBlockBack`/`r34DownBlockBack`/`r34InputGrad`, `mbconvBodyBack`, `mhsaBackFlat`,
`vitBlockBack`, `vitGradFlat` — are **hand-defined `∘`-chains**. Whether each equals the certified
whole-net VJP is a **separate tie that was NOT established** in the float-bridge files. By the chain
rule it should hold; "should" ≠ "proven." Until it's closed, the honest statement is "float ≈ our
reverse-mode transcription," not "float ≈ the certified gradient."

**What's certified to tie against (it exists):**
- r34: `ResNet34BackB0.lean` — `r34BasicBlockB_has_vjp_at`, `r34DownBlockB_has_vjp_at`,
  `cbReluB_has_vjp_at`, `projStridedB_has_vjp`, the stage VJPs.
- vit: `Attention.lean` — `sdpa_back_{Q,K,V}_correct`, `mhsa_has_vjp_mat`,
  `patchEmbed_flat_has_vjp`, `cls_slice_flat_has_vjp`, `classifier_flat_has_vjp`, `vit_full_has_vjp`.
- mnv2: `MobileNetV2JacobianSeal.lean` (`mnv2Live_has_vjp_at_input`); enet: the batched VJPs.
- Glue: `vjpMat_comp`, `vjp_comp`, `HasVJP.comp`, the per-op `_correct` theorems.

**Deliverables (per net, smallest-first):**
1. **r34 first** (no Mat-space attention): prove `r34IdBlockBack = (r34BasicBlockB_has_vjp_at …
   ).backward` (likely `rfl` or a short `unfold`; the block-back was *defined* as the reverse), then
   `r34InputGrad = (resnet34 …_has_vjp).backward` via the composition glue.
2. **vit** (hardest): `mhsaBackFlat = mhsa_has_vjp_mat.backward`, `vitGradFlat =
   vit_full_has_vjp.backward`. RISK: the certified side uses `pdivMat`/`colSlabwise`/flatten reindex;
   matching the assembly exactly may be fiddly (the §1a `*TiePoC` files are the precedent — they tied
   the forward param-grad to certified; this is the input-grad analogue).
3. mnv2/convnext/enet: same pattern.

**Effort: MEDIUM. Risk: MEDIUM** (concentrated in the vit Mat-space reindex match). Start with r34 to
learn whether the ties are `rfl`-cheap or need real glue; that decides the vit budget. **If a tie is
not cheap, the honest move is to STATE the gap precisely** (which `_close` targets a hand-map vs a
certified map) rather than leave it implicit.

### §B-r34 PROBE RESULT (2026-06-26) — the tie is NOT `rfl`-cheap; the gap, stated precisely

Ran the probe empirically (scratch elaboration vs the built oleans). The doc's "likely `rfl` or a
short `unfold`" optimism holds for exactly **one** leaf and fails everywhere else. There are **three**
distinct backward representations in the repo, and the float-bridge one was never wired to certified:

1. **Certified VJP** — `r34BasicBlockB_has_vjp_at.backward`, `resnet34_has_vjp_at` (`vjp_comp_at` /
   `HasVJPAt`, **batched**: `Vec (N·(c·h·w))`, `batchMap N (flatConv…)`, `bnBatchLA`).
2. **SHlo render graph** — `r34BasicBlockBackBatchedGraph`, ALREADY tied to #1 via `…_faithful`
   (`selectPos_faithful`, `convBackBatched_faithful`).
3. **Float-bridge** — `r34IdBlockBack`, `r34InputGrad` (the §B target; **non-batched**: `Vec (c·h·w)`,
   `convFlatBack`, per-channel BN). **Connected to neither #1 nor #2.**

**Per-leaf empirical result:**
- **relu: bare `rfl`** ✓ — `(relu_has_vjp_at).backward = reluMaskBack (matching mask)` closes by `rfl`
  (but only with the abstract mask `m` *pinned* to the true pre-relu sign `x i > 0`).
- **conv: `rfl` FAILS** (the load-bearing gap). `convFlatBack = conv2d (reverseSwap W) 0` (kernel-flip)
  is *propositionally* the certified `conv2d_input_grad_formula` (padded index-sum) — the conv adjoint
  — but **NOT definitionally**. The only proof in-repo (`IR.lean:195–212`, `conv_back_bridge_1to2/2to2`)
  is brute-force `fin_cases` over every spatial cell at **two toy shapes** (`Kernel4 2 1 3 3` / `2 2 3 3`,
  4×4), "by expansion at the concrete shape." No general reindex lemma. At r34's real conv shapes
  (64→64 @ 56², 512→512 @ 7², 7×7 stem @ 112²) `fin_cases` is intractable.

**The cost is concentrated in two places, neither cheap:**
- **(a) The gating leaf — a general conv-adjoint lemma:** `conv2d_input_grad_formula W =
  conv2d (reverseSwap W) 0` for arbitrary `ic oc h w kH kW`. A bounded but real `Finset.sum` reindex
  (the `hpad` index-arithmetic ↔ the `reverseSwap`+`kRev` flip). **It gates every conv-heavy net's §B
  tie** (r34/mnv2/convnext/enet hit the same wall); relu/maxpool/dense are comparatively free.
- **(b) Three structural mismatches** on top: batched (certified) vs non-batched (float-bridge)
  vocabulary; the abstract block/BN/mask params in `r34IdBlockBack`/`r34InputGrad` must be *pinned* to
  the certified ones (even the relu `rfl` is conditional on `m = preRelu>0`); and — the killer — **no
  concrete certified whole-net VJP at full Imagenette dims exists**: `resnet34_has_vjp_at` is
  parametric, the only concrete instance is `resnet34Concrete` at **toy 1-ch/2-class/32² dims**
  (`ResNet34.lean:421–431`). So "`r34InputGrad = certified r34 VJP`" can't be *stated* against a
  full-dim concrete term — you'd instantiate the parametric skeleton and discharge its abstract-stage
  hypotheses (per-channel BN VJPs + smoothness), which the certified side itself flagged "remain."

**Honest one-line status:** the r34 backward float bridge proves *deployed-float ≈ our non-batched
reverse-mode transcription (`r34InputGrad`)*; that transcription equals the certified VJP only via (i)
a not-yet-proven general conv-adjoint reindex, (ii) batched↔non-batched reconciliation + pinning the
abstract BN/mask params, and (iii) a full-dim concrete certified whole-net VJP that does not exist
(only the toy `resnet34Concrete`). relu/maxpool/dense leaves are `rfl`-cheap; conv is not.

**Budget calibration (answers the doc's "decides the vit budget" question):** conv-heavy nets
(r34/mnv2/convnext/enet) need real glue, gated by the conv-adjoint lemma. vit may be cheaper — its
leaves (`sdpa_back`, `patchEmbed_input_grad_formula`) are "bridged directly" — but needs its own probe.
**First concrete §B down payment = the general conv-adjoint lemma** (`conv2d_input_grad_formula =
conv2d ∘ reverseSwap`, all dims): reusable across all four conv nets, and the true gate. → §B-conv-adjoint
below.

### §B-conv-adjoint DONE (2026-06-26) — the gating leaf is proven, all dims, 3-axiom-clean

`Proofs.IR.convBackDenote_eq_input_grad_formula` (`IR.lean`, right after `convBackDenote`):
```
(hkH : 2 * ((kH-1)/2) + 1 = kH) (hkW : 2 * ((kW-1)/2) + 1 = kW)
  → conv2d (reverseSwap W) 0 dy = conv2d_input_grad_formula W dy   -- ALL ic oc h w kH kW
```
The reversed-kernel forward conv (`convFlatBack`'s shape) equals the certified conv input-gradient,
for **arbitrary** dims with **odd** kernels (`2·⌊(kH-1)/2⌋+1 = kH`; covers all of r34 — 3×3 convs +
7×7 stem — and the 1×1/3×3/5×5/7×7 of mnv2/enet/convnext). Proof: per output coord, both sides sum
over input-channel `co`; the LHS `(kh,kw)` kernel-window sum and the RHS `(ho,wo)` output-position sum
range over the **same valid alignments** via the partial bijection `(kh,kw) ↦ (kh+hi-pH, kw+wi-pW)` on
the pad supports (`Finset.sum_subset` to the pad filters → `Finset.sum_bij'` → all index arithmetic by
`omega`). Under oddness `2·pH = kH-1`, the reversed-kernel index `kH-1-kh` matches the formula's
`hi+pH-ho`. ~70 lines, default heartbeats, `[propext, Classical.choice, Quot.sound]`.

**Subsumes the toy bridges:** `conv_back_bridge_{1to2,2to2}` (previously exhaustive `fin_cases` over
every 4×4 cell) are now **one-line instances** (`by decide` discharges 3×3-oddness). The wall the probe
hit — "no general reindex, only `fin_cases` at toy shapes" — is gone for odd kernels.

**Even kernels stay open BY DESIGN** (the `(kH-1)/2` floor makes `2·pH = kH-2 ≠ kH-1`): the identity is
false as stated for even `kH`/`kW` — consistent with the standing "4 even-kernel weight-grad gaps"
(ConvNeXt). Not an r34 blocker (r34 is all-odd).

**What this unlocks / what's still gated for the full r34 §B tie:** the conv *leaf* is now
certified-tied (general). Remaining for `r34IdBlockBack`/`r34InputGrad = certified VJP`: (b) the three
structural mismatches — batched↔non-batched vocabulary, pinning the abstract BN/mask params, and the
missing full-dim concrete certified whole-net VJP (`resnet34Concrete` is toy-only). Those are the next
§B rungs; the load-bearing leaf math is done.

### §B-identity-block DONE (2026-06-26) — float backward = CERTIFIED VJP, b1-free, 3-axiom-clean

`LeanMlir/Proofs/Resnet34BackCertifiedTie.lean` (new module; in lib `roots` + `AuditAxioms`):
the r34 **identity block** float-bridge backward is now tied to the certified gradient.

**The b1 dodge (the key move).** b1 (batched↔non-batched) is *avoided*, not reconciled: the
float-bridge `r34IdBlockBack` is the reverse of `rblkPC` — the **per-channel-BN, non-batched**
block — so the right certified target is a VJP of `rblkPC` in the *same vocabulary*, NOT the
batched true-BN `r34BasicBlockB_has_vjp_at` (`ResNet34BackB0`). That object didn't exist, so:
- `rblkPC_has_vjp_at` — **built it** (the certified per-channel-BN identity-block VJP), mirroring
  the scalar-BN `resblock_has_vjp_at`, reusing `convBnReluPC_has_vjp_at` + `bnPerChannelTensor3_has_vjp`
  + `residual_has_vjp_at`. No `batchMap`/`N`/`Fin.cast` anywhere.
- `convFlatBack_eq_vjp_backward` — the conv **leaf** tie (general, odd kernels), via the committed
  `IR.convBackDenote_eq_input_grad_formula`.
- `r34IdBlockBack_eq_rblkPC_vjp` — **the tie**: `r34IdBlockBack` with its abstract BN-backs pinned to
  the certified per-channel backwards (`bnPerChannelTensor3_has_vjp.backward` at the conv outputs) and
  its ReLU masks pinned to the pre-activation signs **equals** `(rblkPC_has_vjp_at …).backward`. Both
  sides are `fun dy ↦ bodyBack(mask dy) + mask dy`; closes by rewriting the two conv leaves, the rest
  definitional (residual fan-in, `∘`-reversal, pinned BN-backs, relu masks). All 3-axiom-clean.

So **b2 is closed for the identity block** (the pinning is explicit and the tie proven), and **b1 is
shown dodgeable** (build the same-vocabulary certified object). For the r34 identity block the float
bridge's closeness is now closeness to **the certified gradient**, not a hand-map.

### §B-down-block DONE (2026-06-26) — both r34 block types now target the certified gradient

Same module, same recipe, b1-free. The down block `rblkPStridedPC = relu ∘ residualProj(proj, F_s)`
(strided proj + strided first conv):
- `flatConvStride2Back_eq_vjp_backward` — the **strided-conv leaf tie**: `flatConvStride2Back` (=
  `convFlatBack ∘ decimateBack`) = the certified `flatConvStride2_has_vjp.backward`. Decomposes into
  the conv leaf tie + the decimate leaf (`decimateBack_eq_vjp`, already `rfl`), matching
  `flatConvStride2 = decimateFlat ∘ flatConv`. The one genuinely new primitive — and it was cheap.
- `rblkPStridedPC_has_vjp_at` — **built** the certified per-channel-BN strided block VJP (mirrors the
  scalar-BN `resblockProj_has_vjp_at`, with the `residualProj` two-branch fan-in).
- `r34DownBlockBack_eq_rblkPStridedPC_vjp` — **the tie**: `r34DownBlockBack` with BN-backs pinned +
  ReLU masks pinned = `(rblkPStridedPC_has_vjp_at).backward`. Both sides `fun dy ↦ projBack(mask dy) +
  bodyBack(mask dy)`; closes by rewriting the 2 strided-conv leaves + 1 conv leaf, rest definitional.

All 3-axiom-clean, in `Resnet34BackCertifiedTie.lean` + `AuditAxioms`. **So both r34 block types
(identity + downsample) now have float-backward = certified-VJP.**

**Remaining §B:** the stem/GAP/maxpool/dense endpoints (small leaf ties); and the whole-net fold
`r34InputGrad` — still gated by b3 (the certified whole-net VJP `resnet34_has_vjp_at` is parametric,
only concretely instantiated at toy `resnet34Concrete` dims), so the honest whole-net statement remains
the per-block ties + the parametric skeleton, not a full-dim concrete certified term.

---

## §C — Descent: the joint step + BN-in-the-loop (NOT "shallow CNN" — that's done)

**Corrected state (descent is much further than the backward push implied).** Float-fused SGD
descent is DONE per-parameter for: linear (`linear_float_sgd_descends`), the whole MLP
(`mlp_{output,hidden,input}_float_sgd_descends`), AND the shallow CNN — `cnn_conv1_float_sgd_descends`,
`cnn_conv2_float_sgd_descends`, plus both biases (`cnn_conv{1,2}_bias_float_sgd_descends`), all in
`SgdDescentCnn.lean` (the conv weight-grad is `convWeightGrad_eq_dot`). So "the deployed float
training step decreases the loss" is proven for every parameter of a real (if shallow) CNN.

**What's actually open:**
1. **The JOINT step** — every `*_sgd_descends` moves ONE parameter with the others fixed. The
   deployed optimizer updates ALL parameters simultaneously. A genuine joint-step descent (loss
   decreases under the simultaneous float update) is the missing realism. The cross-parameter
   coupling (the Hessian off-diagonal × lr² term) is the new analysis — bounded for small `lr`, this
   is the natural next theorem (`cnn_joint_float_sgd_descends`).
2. **BN in the descent loop** — the CNN descent is the NO-BN cnn. The BN param grads exist
   (`bnBetaGrad_close`, `bnGammaGrad_close`); a BN-CNN float descent (`cifarBn`-style) folds them into
   the step. The honest wrinkle: BN's `γ/√(σ²+ε)` makes the loss landscape non-obvious — scope to a
   smooth operating point with saved stats (the A3 framing).
3. **Deep-net descent stays OPEN BY DESIGN** — vanishing `lr` at depth
   (`floatbridge_certificate_gaps.md` §3). Do NOT target r34+; the shallow CNN is the deepest honest
   descent rung, and the boundary is exactly where the rounding budget overwhelms the per-step
   decrement.

**Effort: HIGH (the joint step is real analysis). Risk: MEDIUM-HIGH.** This is the only direction
with genuine research content — it's what separates "the gradient is close" from "training provably
works in float32." The reusable spine: `sgd_step_close` (the rounded update) + `sgd_descends` (the
abstract descent lemma) + the per-param float descents as the diagonal terms of the joint bound.

---

## Suggested order

1. **§B-r34** — cheap integrity probe: is `r34InputGrad = certified r34 VJP` a `rfl`? It calibrates
   the whole tie effort and is the most load-bearing single check. Do this first regardless.
2. **§A** — the satisfying, low-risk symmetry close (forward whole-nets), reusing every combinator.
3. **§B-vit** — the harder tie, budgeted by what §B-r34 revealed.
4. **§C joint-step** — the real problem, when you want one (start no-BN, then BN).

## Non-goals / honest stops (state wherever cited)

- Deep-net (r34+) DESCENT — open by design, not a target.
- The bridges remain `float ≈ ℝ` over the proven SHlo graph; silicon-`float32`/IREE-lowering residue
  stays validated by `iree-compile` + GPU runs + the transcendental-constant pins, not by these proofs.
- A3 / §A / §B are gradient & forward **closeness** (and the VJP identity) at a smooth point — NOT the
  optimization claim. Only §C touches "training works."
