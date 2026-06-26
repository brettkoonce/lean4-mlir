# Forward whole-net float-bridge — handoff

**Goal:** fold each net's *forward* float bridge at whole-net scale, the way the backward already
does for all five. ✅ **SWEEP COMPLETE — all 5 nets DONE** (r34, efficientnet, mnv2, vit, convnext).
The only remaining piece is the concrete vit forward patch-embed (peer of
`vit_grad_floatBridges_concrete`) — separable, ~200 lines, supplied abstractly for now. This doc is
the pick-up plan / record.

Parent context: `planning/a3_backward_deepnet_assembly.md` (the backward, which is the blueprint).
Memory: `[[float-tier23-and-lexer-gap]]` (A1/A3 state).

---

## STATUS (2026-06-26)

**convnext forward whole-net — DONE, uncommitted** (2026-06-26),
`LeanMlir/Proofs/ConvNeXtWholeFloatBridge.lean`, 3-axiom-clean (7 new decls), builds + `AuditAxioms`
green. The heaviest net, as predicted. `convnext_floatBridges` = the `[3,3,9,3]` fold (peer of
`convnext_grad_floatBridges`), `convnextForward` ∘-skeleton (concrete stem-conv/GAP/dense; stem/head
LN + 4 stages + 3 downsamples supplied — mirrors `convnextInputGrad`). TWO new op-bridges: (1)
`floatBridges_layerScale` — `layerScale γ = diagBack γ` DEFINITIONALLY (both `fun s x i => s i * x i`),
and γ is an exact stored weight (no transcendental), so it's `floatBridges_diagBack` at `fγ=γ, es=0`
(a 1-liner). (2) `floatBridges_flatConvStride4` — the 4×4/s4 patchify stem, `flatConvStride4 =
decimateFlat ∘ decimateOddFlat ∘ flatConv` read at the COMPOSITE `decimateOddIdx ∘ decimateIdx`
coordinate (the two-decimation cousin of r34's `floatBridges_flatConvStride2`). The named dischargers:
`floatBridges_convNextBlock` (the block = `residual(layerScale∘conv∘gelu∘conv∘LN∘depthwise)`, LN
supplied), `floatBridges_convNextStageK` (the depth-`k` stage fold BY INDUCTION on stage depth — the
ConvNeXt analogue of vit's `floatBridges_towerBack`, since blocks have distinct params), and
`floatBridges_cnxDownW` (the downsample `flatConvStride2∘LN`). Gotchas hit: the block's giant inline
`.comp` chain mis-counted parens → switched to incremental `have`s (the carried-forward gotcha);
`floatBridges_gelu`'s `n` isn't inferable in a bare `have` (only in the conclusion) → pin `(n := …)`.
Like r34/mnv2, `convnextForward` is a fresh ∘-skeleton (cosmetic tie open, symmetric with backward).

**vit forward whole-net (encoder-tower fold) — DONE, uncommitted** (2026-06-26),
`LeanMlir/Proofs/ViTWholeFloatBridge.lean`, 3-axiom-clean (3 new decls), builds + `AuditAxioms` green.
`vit_floatBridges` = `vit_full` reversed-from-the-backward = `classifier ∘ perRowFlat finalLN ∘ tower
blocks ∘ patchEmbed` — the forward peer of `vit_grad_floatBridges` (abstract patch-embed/finalLN/blocks
supplied, concrete head). **The thinnest net by far** — two big reuses: (1) the encoder tower is the
SAME `towerBack` (its head-first list fold IS the forward order — direction-agnostic), discharged by
the existing `floatBridges_towerBack`; (2) the per-row LN rides `FloatBridges.perRow`. The blocks are
discharged externally by the pre-existing `floatBridges_vitBlock`. The one new op-bridge is
`floatBridges_clsSlice` (the cls-slice gather = read row 0 of the `(N+1)×D` sequence; exact in float,
magnitude-stable `B=A`, modulus id — the forward peer of `clsScatter`) + `floatBridges_vitHead` (the
concrete `dense ∘ cls-slice`). **REMAINING for full vit parity: the concrete forward patch-embed
bridge** (peer of `PatchEmbedBackFloatBridge.lean`'s `floatBridges_patchEmbedBack`): `patchEmbed_flat`
is the guarded triple-sum (row 0 = cls_token+pos; rows>0 = pos + b_conv + per-patch conv-dot) — a
~200-line `dot_close` + nested `reduction_close` + guard `by_cases` job, exactly mirroring the
backward's, supplied abstractly for now (`hPatch`). The backward shipped abstract `vit_grad_floatBridges`
THEN concrete `vit_grad_floatBridges_concrete` — same staging here.

**mnv2 forward whole-net — DONE, uncommitted** (2026-06-26),
`LeanMlir/Proofs/MobileNetV2WholeFloatBridge.lean`, 3-axiom-clean (8 new decls), builds + `AuditAxioms`
green. Clean r34-shape replay (the forward target `mobilenetv2Forward_full_pc` is already in `∘`-form,
so NO efficientnet nested-app kernel-timeout): `mnv2Forward` skeleton (concrete stem/head/GAP/dense,
stem/head BNs + 6 invres blocks supplied — peer of the backward's `mnv2InputGrad` `b1B..b6B`) +
`mnv2Forward_floatBridges` (the same incremental-`have` left-fold as r34) + the named block bridges
`floatBridges_invresBody{,Strided}PC` (forward peers of the `*BackPC` blocks). **mnv2 has NO SE** — the
one new op-bridge is `floatBridges_relu6` (relu6 = `min(max(·,0),6)` is exact in float + 1-Lipschitz,
a mirror of `floatClose_relu` via mathlib `abs_max_sub_max_le_abs`/`abs_min_sub_min_le_max`); the
strided depthwise REUSED `floatBridges_depthwiseStride2Flat` (built for efficientnet). Like r34 (and
unlike efficientnet), `mnv2Forward` is a fresh skeleton NOT proven `= mobilenetv2Forward_full_pc`
(cosmetic tie #5, symmetric with the backward `mnv2InputGrad`).

**efficientnet forward whole-net — DONE, uncommitted** (2026-06-26),
`LeanMlir/Proofs/EfficientNetWholeFloatBridge.lean`, 3-axiom-clean (7 new decls audited), builds +
`AuditAxioms` green. `efficientnetForwardB_floatBridges` = `head ∘ mbResid ∘ mbStrided ∘ mbNoExp ∘
stem`, **stated on the ∘-composition that IS `efficientnetForwardB`** (identical to the backward
capstone `efficientnetForwardB_has_vjp` — so it ties to the REAL net, not a fresh skeleton). Key
corrections to this doc's old mental model, **read before doing mnv2/convnext/vit**:
* `efficientnetForwardB` is **NOT** `batchMap N (per-example net)` with eval-mode BN — it is a
  `.comp` chain of **per-block batched** stages (`stemB`/`cbsB`/`dwbsB`/`dwbsSB`/`seB`/`projB`) each
  built from `batchMap N (per-example op)` + **true batch-norm** `bnBatchLA` (batch-COUPLED). So the
  honest path was NOT "reuse `floatBridges_mbconvBody` + one outer `batchMap`"; it was to lift EACH
  batch-separable op with `FloatBridges.batchMap` and **supply the 10 `bnBatchLA`s abstractly** (the
  one batch-coupled op — `rsqrt`/`exp` no-spec, deferred exactly like r34's stem BN). swish is
  block-diagonal at the batched index (`floatBridges_swish` directly).
* **State the whole-net conclusion on the explicit `∘`-chain** (copy `efficientnetForwardB_has_vjp`'s
  type verbatim), do NOT `unfold efficientnetForwardB` and `exact` a `.comp` chain — the def is
  NESTED-APPLICATION, and forcing the kernel to match `g∘f` against `fun x => g (f x)` over the huge
  batched terms **deterministic-timeouts the kernel**. On the `∘`-form the `.comp` chain matches
  syntactically (the r34 cheap path). New op-bridge built: `floatBridges_depthwiseStride2Flat`
  (mbStrided's downsample, the depthwise peer of `floatBridges_flatConvStride2`).

**r34 forward whole-net — DONE, committed + pushed** (`e8f98f4` + `891a565`, on `main`),
`LeanMlir/Proofs/Resnet34WholeFloatBridge.lean`, 3-axiom-clean, full build + `AuditAxioms` green.
This file is **the worked example — copy its shape for the other four.** It contains:

- `floatBridges_flatConvStride2` — stem stride-2 conv (`flatConvStride2 = decimateFlat ∘ flatConv`
  ⇒ `floatClose_flatConv` on the `2h×2w` grid read at `decimateIdx`).
- `floatBridges_gap` — GAP, a one-line wrap of the pre-existing `floatClose_gap`.
- `r34Forward` + `r34_floatBridges` — the `[3,4,6,3]` `.comp` fold (concrete stem/maxpool/GAP/dense;
  stem-BN + 16 blocks supplied as `FloatBridges`).
- `floatBridges_r34IdBlock` / `floatBridges_r34DownBlock` — the named per-block dischargers (about the
  ACTUAL `rblkPC` / `rblkPStridedPC`), so the fold's block hyps discharge by name like the backward's.

**Remaining = the concrete vit forward patch-embed + the r34/mnv2/convnext/vit cosmetic skeleton ties
(efficientnet already ties to the real net via the ∘-form). All 5 whole-net forward folds are DONE.**

---

## THE RECIPE (mechanical — each net is "copy the backward, reverse the arrows")

Each net's **backward whole-net file is the exact blueprint** for its forward (same endpoints, same
block decomposition, same `FloatBridges.comp` backbone — just forward ops instead of their VJPs):

| net | forward target def | backward blueprint file | forward block bridge |
|---|---|---|---|
| mnv2 | ✅ **DONE** `MobileNetV2WholeFloatBridge.lean` (`mnv2Forward_floatBridges`) | `MobileNetV2BackFloatBridge.lean` (`mnv2_grad_floatBridges`) | `floatBridges_invresBody{,Strided}PC` (no SE; new `floatBridges_relu6`) |
| efficientnet | ✅ **DONE** `EfficientNetWholeFloatBridge.lean` (`efficientnetForwardB_floatBridges`) | `EfficientNetBackFloatBridge.lean` | per-block batched bridges (`floatBridges_{stemB,cbsB,dwbsB,dwbsSB,seB,projB,mbNoExpFwdB,mbStridedFwdB,mbResidFwdB,headFwdB}`) |
| convnext | ✅ **DONE** `ConvNeXtWholeFloatBridge.lean` (`convnext_floatBridges`) | `ConvNeXtBackFloatBridge.lean` (`convnext_grad_floatBridges`) | `floatBridges_convNextBlock` + `_convNextStageK` + `_cnxDownW` (new `_layerScale`, `_flatConvStride4`) |
| vit | ✅ **DONE** (tower fold) `ViTWholeFloatBridge.lean` (`vit_floatBridges`) — concrete patch-embed remaining | `MhsaBackFloatBridge.lean` + `PatchEmbedBackFloatBridge.lean` | `floatBridges_vitBlock` ✓ (reused `floatBridges_towerBack` + `FloatBridges.perRow`) |

Per-net steps:
1. **Define `<net>Forward`** = the structural `.comp` chain (concrete endpoints; repeated blocks +
   BNs supplied as `FloatBridges` hyps). Mirror the backward `…InputGrad` def, arrows reversed.
2. **Prove `<net>_floatBridges`** by `unfold` + an incremental-`have` `.comp` chain. Reuse the
   forward op-bridges that already exist (`floatBridges_flatConv`, `_dense`, `_maxPool`, `_relu`,
   `_bnPerChannelTensor3`, `_depthwise`, `_gelu`, `_swish`, `_gap`, `_flatConvStride2`, `_gather`,
   `_id`; `FloatBridges.residual` / `.biPathSum` / `.batchMap`).
3. **(if missing) build the forward block bridge** as a thin wrap (see r34's `floatBridges_r34IdBlock`).
4. **Wire** the new module into `lakefile.lean` Proofs roots **and** `tests/AuditAxioms.lean` (import
   + `#print axioms`), `lake build`, then `env LEAN_PATH="$(lake env printenv LEAN_PATH)" lean
   tests/AuditAxioms.lean` and grep `error|sorry|warning` + confirm each new decl is
   `[propext, Classical.choice, Quot.sound]`.

---

## REMAINING WORK (suggested order — easiest first)

### 1. efficientnet forward (batched) — ✅ DONE (2026-06-26). `EfficientNetWholeFloatBridge.lean`.
~~`efficientnetForwardB = batchMap N (per-example net)` (eval-mode BN)~~ — WRONG, see the STATUS-block
correction above: it's a `.comp` chain of per-block **batched** stages with **true batch-norm**
(`bnBatchLA`). Done by lifting each batch-separable op with `FloatBridges.batchMap`, supplying the 10
`bnBatchLA`s abstractly, swish block-diagonal; stated on the `∘`-form (= `efficientnetForwardB`).

### 2. mnv2 forward — ✅ DONE (2026-06-26). `MobileNetV2WholeFloatBridge.lean`.
Built the forward peer `floatBridges_invresBody{,Strided}PC` (mnv2 has NO SE — `floatBridges_mbconvBody`
does NOT apply; the block is `project∘depthwise∘expand`, each `relu6∘bnPC∘conv`). New op-bridge
`floatBridges_relu6` (the forward relu6 clamp, NOT the backward's reluMaskBack); strided depthwise
reused efficientnet's `floatBridges_depthwiseStride2Flat`. Same r34 ∘-skeleton + incremental-`have`
left-fold; targeted the ch7 6-block per-channel render.

### 3. vit forward — ✅ DONE (tower fold, 2026-06-26). `ViTWholeFloatBridge.lean`.
`vit_floatBridges` = the encoder-tower fold (peer of `vit_grad_floatBridges`). Turned out the THINNEST:
the forward **tower** is the SAME `towerBack` (head-first fold = forward order) ⇒ reused
`floatBridges_towerBack` verbatim; the per-row LN reused `FloatBridges.perRow`; cls-slice = the new
1-liner `floatBridges_clsSlice` (gather row 0, exact); head = `floatBridges_dense`. Blocks discharged
by the pre-existing `floatBridges_vitBlock`. **One follow-on for full parity: the concrete forward
patch-embed bridge** (see the STATUS block above) — the ~200-line guarded triple-sum, peer of
`floatBridges_patchEmbedBack`; supplied abstractly (`hPatch`) for now.

### 4. convnext forward — ✅ DONE (2026-06-26). `ConvNeXtWholeFloatBridge.lean`. (Was the heaviest.)
Built `floatBridges_convNextBlock` (block = `residual(layerScale∘conv∘gelu∘conv∘LN∘depthwise)`), the
depth-`k` stage fold `floatBridges_convNextStageK` (induction — the convnext towerBack), the downsample
`floatBridges_cnxDownW`, and the `[3,3,9,3]` fold `convnext_floatBridges`. layerScale turned out to be
`diagBack` definitionally (γ exact ⇒ `floatBridges_diagBack` at `es=0`, a 1-liner); the stride-4 stem
`floatBridges_flatConvStride4` is the two-decimation cousin of `flatConvStride2` (read at
`decimateOddIdx∘decimateIdx`). LNs supplied abstractly. See the STATUS block above for the gotchas.

### 5. (cosmetic) skeleton ↔ actual-net defeq ties — Effort S each, low value.
`r34Forward` (and each `<net>Forward`) is a fresh structural skeleton, NOT proven
`= <net>Forward_full_pc`. A `<net>Forward_eq` unfold/defeq lemma ties the bridge to *the* net. Same
gap exists on the backward side (`r34InputGrad` etc.) — do both directions or neither. Pure polish.

---

## GOTCHAS CARRIED FORWARD (these bit during r34; pre-empt them)

- **Structural `.comp` nesting (the #1 trap).** `FloatBridges.comp hf hg = g ∘ f` unifies
  `Function.comp` **structurally, not up to `∘`-assoc.** A body like `(b₂∘c₂)∘(relu∘b₁∘c₁)` must be
  built as `hH.comp hG` with `hH`/`hG` matching the def's exact nesting — NOT a flat left-fold. A
  right-assoc `∘` chain in the def needs **left-nested** `comp`. (r34 down-block + the backward's MLP/
  tower folds all hit this.)
- **Two-branch fan-in defeq.** For `residualProj`/`biPath` skips, `unfold <blk> residualProj biPath`
  to expose the `fun v j => proj v j + body v j` form that `FloatBridges.biPathSum` literally concludes.
- **Long chains → `maxRecDepth` + incremental `have`s.** Put `set_option maxRecDepth 100000 in`
  **before the docstring** (it does NOT pass through a doc comment) AND decompose the chain into
  incremental `have`s (the one-shot `exact` blows the stack even with the option). 100000 was needed
  for r34's 20-op chain.
- **Pass `(h := …) (w := …)` explicitly** to conv/pool/gap bridges — the spatial dims are NOT
  inferable from the kernel (nonlinear unif: `c*(2h)*(2w) =?= 64*112*112` won't solve).
- **`open FloatModel`** in the file (else `layerAct`/`layerBudget`/`*_nonneg` read as autoImplicits).
- **Supplied-stats discipline.** BN forward bridges (`floatBridges_bnPerChannelTensor3`) take the float
  mean/istd functions + accuracy moduli as data — `rsqrt`/`exp` have no IEEE spec, so the honest move
  is to supply the BN as an abstract `FloatBridges` hyp at the block/whole-net level and discharge it
  with `floatBridges_bnPerChannelTensor3` separately (exactly how r34's fold does it).
- **Audit linter is strict.** A bare `fun A …` where `A` is only used implicitly trips the
  unused-variable warning → use `fun _A …`. The audit greps for `warning`.

---

## HONEST STOPS (state wherever cited — NOT remaining work)

- A3 = gradient/forward **closeness at a smooth point**, NOT descent (deep-net descent open by design;
  vanishing `lr` at depth — `[[floatbridge-certificate-gaps]]` §3). Do NOT target descent.
- The bridges are `float ≈ ℝ` over the proven SHlo graph; per-op StableHLO conformance, IREE lowering,
  and `float32 ≈ ℝ`-the-silicon stay validated by `iree-compile` + the GPU runs, not proven here.
- "Whole-net" = the concrete fold over per-block maps **supplied as `FloatBridges`** (concretely
  dischargeable by the per-block bridges, not inlined) — same honest shape the backward uses.
