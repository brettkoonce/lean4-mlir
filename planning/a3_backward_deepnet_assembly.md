# A3 backward float-closeness: the deep-net assembly (pick-up plan)

Standalone plan for finishing **A3** — "the deployed float backward ≈ the certified `ℝ`
gradient" — for the five deep nets. Parent: `planning/tier23_float_and_syntactic_faithfulness.md`
(A3 section).

**Honest stop:** A3 is gradient **closeness**, NOT descent — deep-net descent stays open by design
(vanishing `lr` at depth; `floatbridge_certificate_gaps.md` §3). Everything here is at a **smooth
point** (the ReLU/activation kinks read fixed sign masks, the maxpool/conv kinks read fixed arg-max
maps), mirroring the §1a backward ties' nonzero-kink hypotheses.

---

## STATUS (2026-06-26, after the §1e + per-net + §1f/§1g session)

**DONE — committed, all 3-axiom-clean, full `tests/AuditAxioms.lean` elaborates clean (1040 decls):**

- **The entire CIFAR backward family** — `cifar8_grad_floatBridges`, `cifarBn_grad_floatBridges`
  (peers of the A1 forward family).
- **The whole ResNet-34 backward** — `r34_grad_floatBridges`, the first Imagenette whole-net
  input-gradient bridge (exact reverse of `resnet34Forward_full_pc`).
- **Every shared CNN/BN backward op** — see "The reusable backbone" below.
- **§1e depthwise + SE backward** (2026-06-26) — the mnv2/enet/convnext op blocker.
  `DepthwiseBackFloatBridge.lean`: `floatBridges_depthwiseBack` (= forward depthwise at the
  spatially-reversed kernel `dwReverse`, FREE reuse like convBack) + `floatBridges_depthwiseStride2Back`
  (`depthwiseFlatBack ∘ decimateBack`). `SEBackFloatBridge.lean`: `floatBridges_seBack` (the
  product-rule two-branch fan-in `biPathSum` of the two `diagBack`s, gate sub-net supplied abstractly)
  + `floatBridges_seGateBack` (gate backward fully assembled) + `floatBridges_broadcastBack` (the one
  new reduction op — the gate's broadcast adjoint = sum-over-spatial, via the BN-back `reduction_close`).

Prior commits: `e1504ad` (CIFAR family + the op bridges), `77f1741` (r34 id-block), `6a4f403`
(strided-conv back), `a656528` (r34 down-block), `7e91c0a` (whole r34 backward). The §1e + per-net +
stride-4 + §1f/§1g batch landed as **`029d29d`** (8 source files: 7 new `*BackFloatBridge.lean` +
`StridedConvBackFloatBridge.lean`, + lakefile/AuditAxioms).

- **Part 2 per-net assembly — mnv2 / efficientnet / convnext DONE** (2026-06-26,
  3-axiom-clean, audit=1032). `MobileNetV2BackFloatBridge.lean`: the inverted-residual body backward
  (`floatBridges_invresBodyBackPC` / `…StridedBackPC` = `expandBack ∘ depthwiseBack ∘ projectBack`,
  the §1e `depthwiseFlatBack`/`depthwiseStride2FlatBack` concrete — mnv2 has **no SE**) +
  `mnv2_grad_floatBridges` the WHOLE-NET fold (exact reverse of `mobilenetv2Forward_full_pc`).
  `EfficientNetBackFloatBridge.lean`: the per-example MBConv body backward
  (`floatBridges_mbconvBodyBack` = `expandBack ∘ depthwiseBack ∘ seBack ∘ projectBack` — **both** §1e
  ops land) + the skip variant; the **whole-net is batched** so the body is the in-scope peer of the
  forward `floatBridges_mbconvBody`. `ConvNeXtBackFloatBridge.lean`: the block body backward
  (`floatBridges_cnxBlockBodyBack` = `depthwiseBack ∘ lnBack ∘ convBack ∘ geluBack ∘ convBack ∘
  layerScaleBack`) + residual block + downsample (`floatBridges_cnxDownBack`) + `convnext_grad_floatBridges`
  the WHOLE-NET `[3,3,9,3]` fold (reverse of `convNextForwardT`).

- **convnext stride-4 stem DONE** (2026-06-26): `StridedConvBackFloatBridge.lean` now has
  `decimateOddBack` (+ `decimateOddIdx_injective`, the odd-position scatter) and
  `flatConvStride4Back = convFlatBack ∘ decimateOddBack ∘ decimateBack`, so `convnext_grad_floatBridges`'s
  stem is now **concrete** (only the stem/head LN-backs supplied).
- **§1g loss-head cotangent seed DONE** (2026-06-26): `LossHeadCotFloatBridge.lean` lifts
  `softmax_ce_cot_close` to `floatBridges_lossSeed` (the CE input-gradient `z ↦ softmax(z) − onehot`,
  bounded by `1 + cotErr(0)` since softmax ∈ [0,1]); `floatBridges_gradFromLoss` = the seed `.comp` any
  `<net>_grad` ⇒ the whole "logits → input-gradient" backward float-bridges **from the loss**.
- **§1f softmax-Jacobian backward (vit crux) DONE** (2026-06-26):
  `SoftmaxBackFloatBridge.lean` — the row-coupled VJP `softmaxBack p dy i = pᵢ·(dyᵢ − ⟨p, dy⟩)` =
  `(diag(p) − p·pᵀ)·dy`. Linear in `dy` ⇒ `FloatClose` modulus = magnitude at `Cdy := e` (like BN-back);
  float threads `mul_close`/`reduction_close`/`sub_close'` with weights supplied within `smErr`.
  `floatClose_softmaxBack` / `floatBridges_softmaxBack`, `P`-general (`P = 1` for softmax).

**REMAINING:**
- **vit whole-net** — the §1f softmax-Jacobian (the crux) is done; remaining is the **Mat-space sdpa
  assembly** (compose `softmaxBack` per query-row with the V-matmul + score-matmul backwards, each a
  rounded dot `dot_close`) + the transformer-block / whole-net fold. NOTE the `FloatBridges` backbone is
  Vec-space; attention is Mat-space (mixes across tokens), so this needs a Mat-space lift (or flatten).
- **efficientnet batched whole-net** — needs the batched-emit lift (the forward's Item-B stub); the
  per-example MBConv body is done.
- **Forward whole-net assembly** — NOTE the *forward* r34 whole-net float bridge is itself only
  block-level in the repo, so the backward is now AHEAD at whole-net scale. The
  `r34_grad_floatBridges` file is the BLUEPRINT for assembling both directions (concrete endpoints +
  abstract blocks supplied as `FloatBridges`).

---

## 0. The fold mechanism (the keystone insight)

A feed-forward net's input-gradient VJP at a smooth point is itself a *forward* composition of maps
on the cotangent, so it threads the **same `FloatBridges.comp` backbone** the forward uses. No new
composition machinery — only the per-op backward bridges + the fan-in combinators. Every backward op
is one of: a linear op's transpose (reuses the forward bridge), a pointwise/structural mask or
scatter (exact in float, modulus `id`), the norm/softmax grads (genuine new budget), or a fan-in
combinator (residual / two-branch sum).

### The reusable backbone (compose over these — ALL DONE)

Per-op backward bridges (file → theorem):
- **Linear input-VJP** `dx = Wᵀ·dy` — `LinBackFloatBridge.floatBridges_linBack` (reuses
  `floatBridges_dense` at the transpose).
- **Conv input-VJP** — `CnnBackFloatBridge.floatBridges_convBack` = `flatConv (reverseSwap W) 0`,
  reuses `floatBridges_flatConv` at the reversed kernel (FREE, the conv analogue of `linBack`).
- **Strided-conv input-VJP** — `StridedConvBackFloatBridge.floatBridges_flatConvStride2Back` =
  `convFlatBack ∘ decimateBack` (zero-upsample scatter then reversed-kernel conv); `decimateBack`
  (= the certified `decimateFlat` VJP, `decimateBack_eq_vjp` by `rfl`) is exact in float &
  magnitude-nonincreasing by `decimateIdx_injective` (REUSED from `ResNet34.lean`, don't redefine).
- **ReLU backward** — `LinBackFloatBridge.floatBridges_reluMaskBack` (`selectPos` mask, exact,
  modulus `id`).
- **Smooth-activation backward** (GELU/Swish/sigmoid/tanh, diagonal Jacobian) —
  `LinBackFloatBridge.floatBridges_diagBack` = `dy ⊙ act'(saved)` (one `mul_close`/coord; supply
  the float derivative within `egelu`/`esig`).
- **MaxPool backward** — `CnnBackFloatBridge.floatBridges_maxPoolBack` (`select_and_scatter` masked
  gather, exact, modulus `id`).
- **GAP backward** — `Resnet34WholeBackFloatBridge.floatBridges_gapBack` = `dy(channel)/(h·w)`
  (scaled broadcast, magnitude-nonincreasing).
- **BatchNorm/LayerNorm backward** — `BnBackComposeBridge.floatBridges_bnBack` (flat, the COTANGENT
  `FloatClose` map; real BN-back is linear in `dy`, so its modulus = the magnitude bound at
  `Cdy:=e`) + `BnPerChannelBackFloatBridge.floatBridges_bnPerChannelBack` (per-channel Tensor3 lift
  = `FloatClose.perRowIdx` of `floatClose_bnBack` conjugated by the reassoc gathers; bridges the
  certified `bnPerChannelTensor3_grad_input`).

Fan-in combinators:
- **Residual skip** `relu(F(x)+x)` backward = `residual bF ∘ reluMaskBack` — reuses
  `FloatBridges.residual` (NO new combinator; the rounded skip-add is the backward's too).
  (`Resnet34BackFloatBridge.floatBridges_r34IdBlockBack`.)
- **Two-branch sum** `f(x)+g(x)` backward — `Resnet34DownBackFloatBridge.FloatBridges.biPathSum`
  (generalizes `floatClose_addResidual`'s `f(x)+x` to `g≠id`; the downsample-skip fan-in).
  (`floatBridges_r34DownBlockBack`.)

Plus the forward bridges reused via the transpose trick, and `FloatBridges.comp`,
`FloatClose.perRowIdx`, `floatBridges_gather`, `FloatClose.cod_nonneg`.

---

## Part 1 — remaining per-op BACKWARD bridges

### ✅ 1a MaxPool back · 1b Conv back · 1c BN-back map · 1d smooth-act back · strided-conv back · GAP back — ALL DONE (see backbone above).

### ✅ 1e. Depthwise / SE grads — DONE (2026-06-26)
- **Depthwise input-VJP** ✅ `DepthwiseBackFloatBridge.floatBridges_depthwiseBack` = `depthwiseFlat
  (dwReverse W) 0` (forward depthwise at the **spatially-reversed** kernel — no channel transpose,
  depthwise has no cross-channel mixing). FREE reuse of `floatBridges_depthwise`, the depthwise twin
  of `convBack`. Strided variant `floatBridges_depthwiseStride2Back = depthwiseFlatBack ∘ decimateBack`.
- **SE backward** ✅ the product rule `seBack(dy) = (g⊙dy) + gateBack(x⊙dy)`. The two-branch
  multiplicative fan-in (`SEBackFloatBridge.floatBridges_seBack`) = `FloatBridges.biPathSum` of the
  main-path `diagBack g` (saved gate) and the gate-path `gateBack ∘ diagBack xinp` (saved input) —
  gate sub-net's backward supplied abstractly, exactly as `r34DownBlockBack` supplies its BN-backs.
  Gate fully assembled (`floatBridges_seGateBack`): `gapBack ∘ linBack W₁ ∘ swishBack ∘ linBack W₂ ∘
  sigmoidBack ∘ broadcastBack`. The one genuinely-new op `floatBridges_broadcastBack` (the broadcast
  adjoint = sum-over-spatial) bridged via the BN-back `reduction_close`.
  - *Honest budget note*: `broadcastBack`'s float model is the masked `M.sum` over the full `c·h·w`
    (budget rides `(1+u)^(c·h·w+1)`) — a valid existence-style over-approximation of the tighter
    per-channel `(1+u)^(h·w+1)` reduce. `ssig`/`ssw` (σ′/swish′ at the saved pre-acts) supplied with
    float closeness `esig`/`eswish`, exactly as the forward `floatClose_seGate` supplies its acts.
Delivered `floatBridges_depthwiseBack`, `floatBridges_depthwiseStride2Back`, `floatBridges_seBack`,
`floatBridges_seGateBack`, `floatBridges_broadcastBack`. **Effort was: M. Risk was: med.**

### ✅ 1f. Attention grad — softmax-Jacobian backward (the vit crux) — CORE DONE (2026-06-26)
The softmax Jacobian couples a whole row (`diag(p) − p pᵀ`). `SoftmaxBackFloatBridge.floatClose_softmaxBack`
/ `floatBridges_softmaxBack` deliver the row VJP `pᵢ·(dyᵢ − ⟨p, dy⟩)` as a composable Vec-space bridge
(threads `mul_close`/`reduction_close`/`sub_close'`, weights within `smErr`, linear-in-`dy` modulus).
REMAINING for the full vit backward: the **Mat-space sdpa assembly** (this row map ∘ the V-matmul +
score-matmul backwards, `dot_close`) — the `Mat`-space peer of the forward `sdpa_close`.

### ✅ 1g. Loss-head cotangent lift — DONE (2026-06-26)
`LossHeadCotFloatBridge.floatBridges_lossSeed` lifts `M.softmax_ce_cot_close` (`FloatBridge.lean:1808`,
the per-entry `softmax − onehot` `cotErr` bound) to a `FloatClose`/`FloatBridges` **seed**;
`floatBridges_gradFromLoss` = the seed `.comp` any `<net>_grad` — the whole-net backward now starts
*from the loss* (end-to-end "float gradient ≈ real gradient"), not from an abstract `dy`.

---

## Part 2 — per-net BACKWARD assembly

| Net | Status | Backward ops needed | New from Part 1 |
|---|---|---|---|
| **cifar8** (no-BN) | ✅ `cifar8_grad_floatBridges` | convBack, reluMaskBack, maxPoolBack, linBack | — |
| **cifarBn** | ✅ `cifarBn_grad_floatBridges` | + bnBack (per-channel) | — |
| **r34** | ✅ `r34_grad_floatBridges` (WHOLE NET) | convBack, strided-convBack, bnBack, residual + biPathSum fan-in, maxPoolBack, gapBack, linBack | — |
| **mnv2** | ✅ `mnv2_grad_floatBridges` (WHOLE NET) | + depthwiseBack ✅, smooth (relu6 mask ✅), NO SE | **1e ✅** |
| **efficientnet** | ✅ `floatBridges_mbconvBodyBack` (per-ex body; whole-net batched) | + swishBack (1d✅), seBack ✅, depthwiseBack ✅ | **1e ✅** |
| **convnext** | ✅ `convnext_grad_floatBridges` (WHOLE NET) + block body | LN-back (= bnBack ✅), geluBack (1d ✅), depthwiseBack ✅, layerScale (diagBack ✅) | **1e ✅** |
| **vit** | softmax-Jacobian ✅ (`floatBridges_softmaxBack`); Mat-space sdpa assembly TODO | sdpaBack-row ✅, LN-back ✅, geluBack ✅, linBack ✅, residual ✅ | **1f ✅ (core)** |

(mnv2: full whole-net, per-example forward. enet: whole-net is batched ⇒ per-example MBConv body is the
peer of the forward bridge. convnext: whole-net fold + block body, stem now concrete via
`flatConvStride4Back`. vit: §1f softmax-Jacobian crux done; Mat-space sdpa assembly is the remaining lift.)

**The whole-net assembly pattern (from `r34_grad_floatBridges` — reuse it):**
1. Define `<net>InputGrad` = the explicit reverse `∘` chain of the forward, each op → its backward,
   with the **structural endpoints concrete** (stem/GAP/head/maxpool) and the **repeated blocks
   supplied as `FloatBridges` hypotheses** (discharged separately by the per-block bridges) — exactly
   how the forward `cifarBn_floatBridges` supplies its BNs.
2. Prove `<net>_grad_floatBridges` by `unfold` + a `.comp` chain. **Two gotchas for big chains
   (≳15 ops):** (a) the one-shot `exact` hits `maxRecDepth` → put `set_option maxRecDepth 100000 in`
   BEFORE the docstring (it does NOT pass through a doc comment) AND (b) decompose the chain into
   incremental `have`s (the partial-composition types solidify and the final `exact` is cheap).

---

## Suggested order

1. ✅ **1e (depthwise + SE back)** — DONE.
2. ✅ **Part 2 per-net assembly** — DONE: mnv2 (whole-net), efficientnet (per-ex MBConv body),
   convnext (whole-net + block body + concrete `flatConvStride4Back` stem).
3. ✅ **1g (loss-head lift)** — DONE.
4. ✅ **1f softmax-Jacobian (the crux)** — DONE; remaining = the Mat-space sdpa assembly + vit whole-net.
5. Remaining follow-ons: vit Mat-space sdpa assembly; enet batched whole-net (Item-B).
3. **1g (loss-head lift)** anytime — upgrades each `<net>_grad_floatBridges` from "≈ at an abstract
   `dy`" to "≈ from the loss."
4. **(optional) forward whole-net assembly** — reuse the `r34InputGrad` blueprint to land the forward
   `r34_float_close`/`mnv2`/etc. whole-net (currently block-level only), closing both directions.

---

## Gotchas carried forward

**From the keystone session (BN-back, masks):**
- **Two `if`s, same condition** ⇒ `simp only [if_pos h]` / `[if_neg h]` (rewrites BOTH); plain `rw`
  hits only one.
- **Smooth-point masks/maps** = fixed parameters (`cond : Fin n → Prop` `[DecidablePred]`, saved
  pool tensors, saved derivatives) — NOT recomputed; the honest model, matches the §1a ties.
- **Big nested budget defs**: `set`-abbreviate each intermediate to mirror the def's `let`s, then
  `simp only [theDefs]; exact hfinal` (defeq closes).
- **`FloatClose.cod_nonneg`** gives `0 ≤ B` for `FloatBridges` witnesses for free (needs `0 < n` on
  the codomain — pass the channel/dim positivity, e.g. gapBack needs `0 < c`).
- Supplied-stats discipline: float `istd`/`x̂`/`exp`/`sig`/`act'` are MODELLED (close within
  `es`/`exh`/`esig`/`egelu`), discharged at instantiation by the forward keystones — `rsqrt`/`exp`
  have no IEEE spec, so this is the honest (and only) shape.

**From the CIFAR + r34 backward session (this round):**
- **convBack/strided-convBack are FREE reuses**, not from-scratch proofs: `convBackDenote W =
  conv2d (reverseSwap W) 0` (so flat = `flatConv (reverseSwap W) 0`); `flatConvStride2 =
  decimateFlat ∘ flatConv` (so back = `convFlatBack ∘ decimateBack`). The plan's old "convBack is
  its own bridge" was over-pessimistic.
- **`decimateIdx_injective` already exists** (`ResNet34.lean:503`) — import `ResNet34` and reuse, or
  you get an environment-clash on the auto-generated `._proof_*`.
- **BN-back is linear in `dy`** ⇒ its FloatClose modulus IS the real-magnitude bound at `Cdy:=e`
  (`bn_grad_input_diff_abs_le` = `bn_grad_input_abs_le` shape). The per-channel lift is the SAME
  conjugation as the forward (`FloatClose.perRowIdx` + reassoc gathers), with `bn_grad_input` in the
  per-row slot; `bnPerChannel_grad_input = perRowIdxFlat …` DEFINITIONALLY (`funext dy idx; rfl`).
- **`flatConvStride2Back`'s `{h,w}` are NOT inferable from `W`** (they're the output spatial dims) —
  pass `(h := h) (w := w)` explicitly in the conclusion.
- **gapBack coercion**: write `((h:ℝ)*(w:ℝ))` consistently; `abs_div` + `gcongr` +
  `div_sub_div_same` for the bound; float modelled as `M.mul (1/(h·w)) (dy channel)`.
- **whnf "timeout"/maxRecDepth** on giant `.comp` chains: `set_option maxRecDepth 100000 in` (before
  the docstring) + incremental `have` decomposition (see Part 2). 100000 was needed even with the
  decomposition for the 20-op r34 chain.
- Wire every new module into BOTH `lakefile.lean` Proofs `roots` AND `tests/AuditAxioms.lean`
  (import + `#print axioms`), then `lake build` (the audit invokes `lean` directly, needs the oleans
  first). Run `env LEAN_PATH="$(lake env printenv LEAN_PATH)" lean tests/AuditAxioms.lean` and grep
  for `error|sorry|warning` + any `depends on axioms` line missing `[propext, Classical.choice,
  Quot.sound]`.

---

## Scope / honest stops (state wherever cited)

- A3 = gradient **closeness** at a **smooth point**. NOT descent (open by design at depth). NOT the
  param-gradient *update* (the SGD/Adam step, a separate rung).
- Whole-net witnesses (`cifar8`/`cifarBn`/`r34` `_grad`) fold concrete endpoints over **per-block
  backward maps supplied as `FloatBridges` hypotheses** (discharged separately by the per-block
  bridges) — the same modular shape the forward capstones use for their BNs. "Whole-net" = the
  concrete fold; the blocks are concretely dischargeable, not inlined.
- The bridges are `float ≈ ℝ` over the proven SHlo backward graph; per-op StableHLO-spec conformance,
  IREE lowering, and `float32 ≈ ℝ`-the-silicon stay validated by `iree-compile` + the GPU runs (same
  residue as the forward + the honesty pass).
