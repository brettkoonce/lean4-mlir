# Forward whole-net float-bridge — handoff

**Goal:** fold each net's *forward* float bridge at whole-net scale, the way the backward already
does for all five. r34 and **efficientnet** are **DONE**; the other three (mnv2 / convnext / vit)
are still block-level. This doc is the pick-up plan.

Parent context: `planning/a3_backward_deepnet_assembly.md` (the backward, which is the blueprint).
Memory: `[[float-tier23-and-lexer-gap]]` (A1/A3 state).

---

## STATUS (2026-06-26)

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

**Remaining = mnv2 / convnext / vit forwards + the r34/mnv2/convnext/vit cosmetic ties (efficientnet
already ties to the real net via the ∘-form).**

---

## THE RECIPE (mechanical — each net is "copy the backward, reverse the arrows")

Each net's **backward whole-net file is the exact blueprint** for its forward (same endpoints, same
block decomposition, same `FloatBridges.comp` backbone — just forward ops instead of their VJPs):

| net | forward target def | backward blueprint file | forward block bridge |
|---|---|---|---|
| mnv2 | `mobilenetv2Forward_full_pc` (`ResNet34RenderPC`-style PC file) | `MobileNetV2BackFloatBridge.lean` (`mnv2_grad_floatBridges`) | `floatBridges_mbconvBody` ✓ (no-SE invres variant — VERIFY/build) |
| efficientnet | ✅ **DONE** `EfficientNetWholeFloatBridge.lean` (`efficientnetForwardB_floatBridges`) | `EfficientNetBackFloatBridge.lean` | per-block batched bridges (`floatBridges_{stemB,cbsB,dwbsB,dwbsSB,seB,projB,mbNoExpFwdB,mbStridedFwdB,mbResidFwdB,headFwdB}`) |
| convnext | `convNextForwardT` | `ConvNeXtBackFloatBridge.lean` (`convnext_grad_floatBridges`) | **MISSING** — build `floatBridges_cnxBlock` (peer of `cnxBlockBodyBack`) |
| vit | `vit_full` | `MhsaBackFloatBridge.lean` + `PatchEmbedBackFloatBridge.lean` (`vit_grad_floatBridges_concrete`) | `floatBridges_vitBlock` ✓ |

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

### 2. mnv2 forward — Effort S–M, risk low.
Mirror `mnv2_grad_floatBridges`: concrete stem(`flatConvStride2 ∘ bn ∘ relu`, reuse r34's bridge)/
head/GAP/dense + the inverted-residual blocks. **VERIFY** whether `floatBridges_mbconvBody` covers
mnv2's no-SE block or whether a `floatBridges_invresBody` (expand∘depthwise∘project, no SE) wrap is
needed — the backward built `invresBodyBackPC` separately, so likely build the forward peer (thin:
`floatBridges_depthwise` ✓ + `floatBridges_flatConv` + BN + relu6-mask-as-relu). Target the **ch7
6-block per-channel render** (same target r34/backward used), NOT the 17-block paper trainer.

### 3. vit forward — Effort M, risk low–med.
Mirror `vit_grad_floatBridges_concrete` (`MhsaBackFloatBridge` + `PatchEmbedBackFloatBridge`).
`floatBridges_vitBlock` ✓ exists. Build: (a) a forward **tower fold** (`.comp` list-fold of distinct-
param blocks, peer of `floatBridges_towerBack`); (b) endpoints — patch-embed **forward** (peer of the
`patchEmbedBack` transposed-conv; the forward patch-embed is a plain conv/reshape, likely already has
a bridge or is a `floatBridges_flatConv` + `gather`), cls-slice (a `gather`/scatter, exact), head
(`floatBridges_dense`). Watch the **left-nested comp** gotcha hard here (the backward hit it on the
MLP + tower chains).

### 4. convnext forward — HEAVIEST (forward block bridge missing). Effort M, risk med.
Mirror `convnext_grad_floatBridges`. **Build `floatBridges_cnxBlock`** first (peer of
`cnxBlockBodyBack` = `depthwise ∘ LN ∘ conv ∘ gelu ∘ conv ∘ layerScale`, wrapped in `residual`): all
op-bridges exist (`floatBridges_depthwise` ✓, `_bnPerChannelTensor3`/LN ✓, `_flatConv` ✓, `_gelu` ✓,
layerScale = a `diag`/`gather`-style scale — check for a forward bridge or build a 1-liner). Then the
`[3,3,9,3]` fold with the **stride-4 stem** (`flatConvStride4 = decimateFlat ∘ decimateOddFlat ∘
flatConv` — the backward built `flatConvStride4Back`; build the forward `floatBridges_flatConvStride4`
the same way, or supply the stem BN abstractly).

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
