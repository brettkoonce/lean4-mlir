# Backward certified-VJP ties (§B for the other 4 nets) + the vit-forward tie — handoff

**Goal:** finish tying every whole-net float bridge to *the certified object*, in both directions, so
the closeness claims are "≈ the certified gradient / the real net" — not "≈ a hand-assembled skeleton."
The forward sweep is done (`planning/forward_wholenet_handoff.md`): all 5 forward bridges exist, vit is
concrete, and r34/mnv2/convnext forward skeletons are tied to their real net defs
(`WholeNetForwardTies.lean`). This doc is the **two remaining un-tied directions**:

1. **(Item 2) Backward §B certified-VJP ties for mnv2 / efficientnet / convnext / vit** — do for the
   other four what `Resnet34BackCertifiedTie.lean` did for r34.
2. **(Item 3) The vit-forward tie** — ✅ **DONE 2026-06-26.** `vit_full_eq_vitForwardFlat`
   (`ViTWholeFloatBridge.lean`), 3-axiom-clean — see PART B below for the closed-out record. The
   forward tie sweep now covers **all 5 nets**.

When this is finalized, the next pillar is **Gap-3 (syntactic / lexer faithfulness)** — see the bottom.

Memory: `[[float-tier23-and-lexer-gap]]`, `[[floatbridge-b-certified-tie]]` (the r34 §B record).

---

## PART A — the backward §B ties (Item 2) — convnext/mnv2/efficientnet ✅ DONE 2026-06-26

**Status (2026-06-26):** depthwise gate (shared prereq) + convnext + mnv2 + efficientnet §B ties all
landed, 3-axiom-clean, audited (1133 clean prints). vit §B = sdpa cores are certified-`sdpa_back`-grounded
**by construction** (no leaf gate needed, unlike CNNs) + block-level Mat↔flat reconciliation remaining
(see the vit note at the end of Part A).

- **Depthwise adjoint gate** (`DepthwiseBackCertifiedTie.lean`): `depthwiseConv2d_dwReverse_eq_input_grad_formula`
  (the depthwise twin of `IR.convBackDenote_eq_input_grad_formula` — `Finset.sum_bij'` on the pad supports,
  no `Σ co`) + `depthwiseFlatBack_eq_vjp_backward` (stride-1) + `depthwiseStride2FlatBack_eq_vjp_backward`.
  The gate did NOT pre-exist; built once here, unblocks all three CNNs.
- **convnext** (`ConvNeXtBackCertifiedTie.lean`): `cnxBlockBodyBack_eq_convNextBlockBody_vjp` (+ residual-
  wrapped `cnxBlockBack_eq_convNextBlock_vjp`). Certified `convNextBlockBody_has_vjp` already existed; pin
  LN/gelu/layerScale backs to certified at the saved activations, tie 1×1 convs + depthwise via leaf gates,
  `rfl`.
- **mnv2** (`MobileNetV2BackCertifiedTie.lean`): the certified VJP did NOT exist in the deployed per-channel
  vocabulary, so built `invresBodyPC_has_vjp_at` / `invresBodyStridedPC_has_vjp_at` fresh (like r34's
  `rblkPC_has_vjp_at`), then `invresBodyBackPC_eq_invresBodyPC_vjp` (+ strided). relu6 masks pinned to the
  `0 < preact < 6` clamp-window signs (relu6's certified backward); per-channel BN backs to
  `bnPerChannelTensor3_has_vjp`. MobileNetV2.lean has no `open Classical`, so the relu6 `DecidableAnd` mask
  instance matches `reluMaskBack`'s inferred one — `rfl` closes.
- **efficientnet** (`EfficientNetBackCertifiedTie.lean`): `mbconvBodyBack_eq_mbconvBody_vjp`. Certified
  `mbconvBody_has_vjp` already existed (global `bnForward`, per-example — the in-scope body, b1-free). The SE
  backward is **pinned** to `seBlockFull_has_vjp.backward` (abstract in the float bridge), NOT a leaf gate —
  the "product-rule" complexity lives in the float-bridge discharge (`floatBridges_seBack`), not the §B tie.

**Reusable recipe that worked first-try for all three:** pin every abstract back (BN / swish / gelu /
layerScale / SE) to the certified op-VJP `.backward` at the exact saved forward activation; tie the
concrete conv/depthwise leaves via the gates (conv/depthwise backwards ignore their linear primal, so any
`x` works); then `funext dy; unfold <floatBack>; rw [the 2-3 leaves]; rfl`. The certified `vjp_comp(_at)`
backward unfolds definitionally to the nested op-backwards; `set`-built certified VJPs still close by `rfl`.

### The r34 §B blueprint (the pattern to replicate)

`Resnet34BackCertifiedTie.lean` closed the §B integrity question for the r34 blocks. The backward
float bridges prove *deployed-float ≈ a hand-assembled reverse-mode transcription* (`r34IdBlockBack`);
§B proves *that transcription IS the certified input-gradient VJP*. **Three pieces per block type:**

1. **Leaf tie** — the float-bridge `convFlatBack` (reversed-kernel conv) IS the certified conv
   input-VJP, via the general odd-kernel `IR.convBackDenote_eq_input_grad_formula`
   (`convFlatBack_eq_vjp_backward`). The strided leaf adds the `decimateBack` `rfl`
   (`flatConvStride2Back_eq_vjp_backward`).
2. **Certified per-block VJP in the SAME vocabulary** — build `<block>_has_vjp_at` about the *actual
   per-channel-BN, non-batched* block the float bridge reverses (`rblkPC_has_vjp_at`), assembled from
   the per-op VJPs (`convBnReluPC_has_vjp_at` + `bnPerChannelTensor3_has_vjp` + `residual_has_vjp_at`).
   **Critical design choice (keeps it `b1`-free):** target the non-batched per-channel-BN block, NOT
   the batched true-BN `*_has_vjp_at` from the `*BackB0` files — so there's no batched↔non-batched
   `batchMap` reconciliation.
3. **The tie** — `<block>Back` with its abstract BN-backs pinned to the certified per-channel-BN
   backwards and its ReLU masks pinned to the actual pre-activation signs **=** `(<block>_has_vjp_at
   …).backward` (`r34IdBlockBack_eq_rblkPC_vjp`). Closes by rewriting the conv leaves; the residual
   fan-in, the `∘`-reversal, the masks, and the pinned BN-backs match definitionally.

The certified per-block VJPs the other four nets need **already exist**:
`mbconvBody_has_vjp` (`EfficientNet.lean:312`), `convNextBlockBody_has_vjp` (`ConvNeXt.lean:155`),
`invresBody_has_vjp_at` (`MobileNetV2.lean:343`), `vit_body_has_vjp_mat` (`Attention.lean:2508`).
So Part A is mostly **piece 1 (leaf ties for the new ops) + piece 3 (the per-block tie)**.

### Shared prerequisite: the depthwise adjoint gate

The conv leaf gate `IR.convBackDenote_eq_input_grad_formula` exists; mnv2/efficientnet/convnext all
reverse a **depthwise** conv, so they need the **depthwise analogue** (`depthwiseBackDenote =
depthwiseFlat (dwReverse W) 0` IS the certified depthwise input-VJP). **VERIFY whether this gate
already exists** (grep `depthwise.*input_grad_formula` / `depthwiseBackDenote_eq`); if not, build it
once (it's the depthwise twin of the conv gate — no channel transpose, the `dwReverse` spatial-only
flip the float bridge already uses). This single lemma unblocks the depthwise leaf for all three CNNs.

### Per-net plan (suggested order — easiest first)

| net | float-bridge target | certified VJP (exists) | NEW leaf ties needed | risk |
|---|---|---|---|---|
| **convnext** | `cnxBlockBodyBack` | `convNextBlockBody_has_vjp` | depthwise gate, gelu/LN/layerScale `diagBack`=stop-grad mul (cheap, exact) | low |
| **mnv2** | `invresBodyBackPC` (+strided) | `invresBody_has_vjp_at` | depthwise gate (+stride2), relu6-mask = the fixed `selectPos` (exact) | low |
| **efficientnet** | `mbconvBodyBack` | `mbconvBody_has_vjp` | depthwise gate, **SE-back** (the product-rule fan-in — the genuinely new tie), swish `diagBack` | med |
| **vit** | `vitBlockBack` | `vit_body_has_vjp_mat` | **sdpa-back** (the attention adjoint — Q/K/V matmuls + softmaxBack), LN/dense/gelu | med-high |

1. **convnext first** (no SE, no attention, no strided depthwise — cleanest CNN). Validates the
   depthwise gate + the `diagBack`/`bnBack` pin pattern on a body that is a plain `.comp` chain.
2. **mnv2** (no SE; adds the strided-depthwise leaf + the relu6 mask). The relu6 backward is the
   fixed-mask `reluMaskBack`, so its tie is the `selectPos`-is-the-derivative argument — exact.
3. **efficientnet** — the first with **SE**. The SE backward is a two-branch *multiplicative* fan-in
   (`seBack(dy) = (g⊙dy) + gateBack(x⊙dy)`); the certified `seBlockFull_has_vjp` exists, so the tie is
   the product-rule leaf. Per-example body is non-batched ⇒ `b1`-free like r34 (the batched whole-net
   stays at the batched-block level, as the forward does).
4. **vit** — the hardest: the sdpa adjoint. **MHSA LEAF TIE DONE 2026-06-26** (`mhsaBackFlat_eq_mhsa_vjp`,
   `ViTMhsaBackCertifiedTie.lean`, 3-axiom-clean): the float MHSA backward `mhsaBackFlat` (with Q/K/V pinned
   to the actual `dense W· b· (X·)` projections at the saved input X) **IS the certified
   `mhsa_has_vjp_mat.backward`, flattened**. The sdpa cores were already certified-`sdpa_back`-grounded by
   construction (`coreQFlat = flatten ∘ mhsaSdpaBackQ ∘ unflatten`, `mhsaSdpaBackQ = sdpa_back_Q` per
   `mhSlab` head); what's closed is the **assembly reconciliation** — the float's SEPARATE `dense Wᵀq/Wᵀk/Wᵀv`
   projBacks vs the certified qkv-MERGED VJP. Route: ViTBackB0's `mhsa_backward_collapseMH` (certified Mat
   backward = the per-head merged sum `mhsaBackCollapsedMH`) + a coordinate match — `dense Wᵀ 0 = Mat.mulVec
   W` (`projBack_core_coord`/`woback_unflatten`), the `Σ k` over `h·dh` reindexes to `Σₕ Σⱼ`
   (`Equiv.sum_comp` + `Fintype.sum_prod_type`), `mhSlab h Q' = Qg h` definitionally, and the float's three
   separate projBack sums regroup into the certified `Σₕ(Q+K+V)` via `Finset.sum_add_distrib` (+ `ring` for
   the `Q+(K+V)` vs `(Q+K)+V` association).

   **ATTN-SUBLAYER RECONCILIATION DONE 2026-06-26** (`transformerAttnSublayerBack_flat_decomp`, same file,
   3-axiom-clean): the certified attn-sublayer VJP, flattened, IS the residual skip `v` plus the certified
   per-token LayerNorm backward of (the unflatten of) `mhsaBackFlat` — so the MHSA leaf is now grounded in
   the certified gradient *through the sublayer*. Route: `transformerAttnSublayer_backward_decomp` (the
   `biPathMat`/`vjpMat_comp` unfold = skip + `LN₁-back ∘ mhsa-back`, `rfl` — but needs `set_option
   maxHeartbeats 10000000` and the build is ~222s: the general-heads `transformerAttnSublayer_has_vjp_mat.backward`
   is heavy for the kernel, unlike ViTBackB0's heads=1 `transformerBlock_backward_unfold`) + plug in the
   MHSA leaf (`congrFun mhsaBackFlat_eq_mhsa_vjp` + `unflatten_flatten`) + flatten the residual sum.

   **The structural finding (the genuine remaining piece):** the certified LayerNorm-back is
   `layerNorm_per_token_has_vjp_mat.backward A` = `rowwise` of the single-token LN VJP, which threads each
   token's saved input `A r` (its Jacobian differs per token). The float `vitBlockBack` instead lifts a
   SINGLE `lnB₁ : Vec → Vec` via `perRowFlat` (one cotangent→grad map for every token). For the FORWARD
   this is fine (`layerNormForward` is one pure function of each token's own input); for the BACKWARD a
   single map can't carry the per-token saved-input dependence. So the attn-sublayer reconciliation above
   uses the CORRECT per-token LN-back (not the float's `perRowFlat lnB₁`); closing the full `vitBlockBack`
   tie needs `vitBlockBack` enriched with a per-token-input-aware LN lift (not plain `perRowFlat` of a
   single `lnB₁`). The sdpa half ties; the LN-back half is this float-bridge enrichment. (The MLP sublayer
   is the analogous decomposition with dense/gelu instead of MHSA — same per-token-LN consideration, no new
   analysis.)

### Honest scope note

Like r34, the tie is at the **block** level (id/down/MBConv/cnx/vit block). The whole-net backward tie
is gated by a *fully-certified whole-net VJP in the same vocabulary* (r34's was gated on the toy-only
`resnet34Concrete`). So Part A's deliverable is "every per-block backward float bridge ties to the
certified per-block VJP" — the same honest stopping point r34 reached. State it that way.

---

## PART B — the vit-forward tie (Item 3) — ✅ DONE 2026-06-26

`vit_floatBridges` is stated on `vitForwardFlat = classifier ∘ perRowFlat finalLN ∘ towerBack blocks ∘
patchEmbed` (abstract blocks/finalLN). The real `vit_full = classifier ∘ (flatten ∘ vit_body ∘
unflatten) ∘ patchEmbed`. The tie is the **middle**: `flatten ∘ vit_body ∘ unflatten = perRowFlat
finalLN ∘ towerBack blocks`.

**Closed as `vit_full_eq_vitForwardFlat`** (`ViTWholeFloatBridge.lean`, alongside `vitForwardFlat`;
3-axiom-clean; audited in `tests/AuditAxioms.lean`). `vit_body` (`Attention.lean:2451`) IS already
`(per-token finalLN) ∘ transformerTower`, so the tie split exactly as predicted, plus two reusable
helpers:
* **finalLN reindex** — `flatten ∘ (per-token finalLN) ∘ unflatten = perRowFlat (N+1) D finalLN`: the
  one-`simp` `Mat.unflatten_flatten` round-trip inside `hmid` (no separate lemma needed).
* **tower fold** — `flatten ∘ transformerTower k ∘ unflatten = towerBack (List.replicate k blockFlat)`.
  The blocks share one parameter tuple, so the cleanest route is **through `Function.iterate`**, which
  absorbs the head/tail-fold order mismatch (the trap: `towerBack` adds the block head-first / applied
  first, `transformerTower`'s `Nat.rec` adds it tail-last / applied last — a naive combined induction
  hits a commutation obligation). Two lemmas:
  - `towerBack_replicate : towerBack (List.replicate k g) = g^[k]` (generic; `Function.iterate_succ`,
    the `f^[k] ∘ f` form, matches `towerBack`'s head-first cons).
  - `transformerTower_flatten_eq_iterate : (flatten ∘ transformerTower k ∘ unflatten) = blockFlat^[k]`
    (`Function.iterate_succ'`, the `f ∘ f^[k]` form, matches the tower's block-last `Nat.rec`; the step
    pushes `Mat.unflatten_flatten` through one block — the flatten/unflatten conjugation).

  Both reduce to the SAME `blockFlat^[k]`, so `hmid` rewrites one into the other. The final `vit_full`
  vs `vitForwardFlat` glue is `unfold` + `rw [hmid, Function.comp_assoc]` (the lone non-`rfl` step is the
  `∘`-reassociation, since `Function.comp` isn't reducible). `blocks := List.replicate kBlocks blockFlat`
  (identical per layer — matches the shared-param tower). Now `vit_full = vitForwardFlat (concrete
  blocks)`, so `vit_floatBridges` provably bounds the actual net, tied like r34/mnv2/convnext.

---

## ORDER / EFFORT

1. ~~**vit-forward tie (Part B)**~~ — ✅ DONE 2026-06-26 (`vit_full_eq_vitForwardFlat`). ALL 5 forwards
   now tie. **Part A (the backward §B ties) is the remaining work.**
2. **convnext + mnv2 backward ties** — validate the depthwise gate + `diagBack`/`bnBack` pins.
3. **efficientnet backward tie** — the SE product-rule leaf.
4. **vit backward tie** — the sdpa adjoint (hardest).

Each is its own commit + `AuditAxioms` 3-axiom check (the `[propext, Classical.choice, Quot.sound]`
gate). Reuse the r34 §B file as the literal template; most of the machinery (leaf gates, certified
VJPs, the float bridges) already exists — this is *connection* work, not new analysis.

---

## THE FOLLOW-ON: Gap-3 (syntactic / lexer faithfulness)

Once the ties are finalized, the next pillar is the **other faithfulness axis**: the emitted MLIR
*text* parses back to the exact proven graph (`parse (lex (pretty g)) = some (skel g)`). Plan:
`planning/tier23_float_and_syntactic_faithfulness.md`. **Much smaller than it looks** — `pretty =
serializeToks ∘ toToks ∘ skel` already factors through the ordered token stream, and the structural
roundtrip `parse (toToks (skel a)) = some (skel a)` is already proven (`StableHLOParse.lean:239`).
Remaining = a per-op `lexTok` inverse of `emitTok` (finite case-split) lifted by `foldl` induction;
SSA names are nameless/positional (De-Bruijn-ish), so no symbol-table. Closing it ⇒ the deployed
artifact is faithful **both numerically (the float bridges) and syntactically (the lexer)** — both
provable axes closed. Effort medium, risk low.

---

## HONEST STOPS (state wherever cited — NOT remaining work)

- The backward ties are **closeness to the certified gradient at a smooth point**, NOT descent
  (deep-net descent open by design — `[[floatbridge-certificate-gaps]]` §3). Do NOT target descent.
- Part A ties at the **per-block** level (the r34 §B stopping point); the whole-net backward tie is
  gated by a fully-certified whole-net VJP in the same vocabulary (toy-only today).
- Per-op StableHLO conformance, IREE lowering, and `float32 ≈ ℝ`-the-silicon stay validated by
  `iree-compile` + the GPU runs, not proven here (the float/lexer bridges are the provable parts).
