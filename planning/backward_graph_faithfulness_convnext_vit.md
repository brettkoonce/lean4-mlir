# Backward-graph faithfulness: close ConvNeXt + ViT

Goal of the session: give **ConvNeXt** and **ViT** the same whole-block
*backward-graph faithfulness* capstone that r34 / enet / mnv2 already have —
i.e. prove the **rendered StableHLO backward graph** of each block denotes the
**proven VJP** (`den (blockBackGraph …) = (block_has_vjp …).backward …`),
certifying to exactly `[propext, Classical.choice, Quot.sound]` (no `sorryAx`).

This doc is self-contained; you can start cold from it.

---

## 0. Status (as of 2026-06-14, commit `ae84b73`)

Whole-block backward-graph capstones (rendered backward ↔ VJP), per net:

| net | identity/residual block | downsample/strided block | file |
|---|:--:|:--:|---|
| EfficientNet | ✅ | ✅ | `LeanMlir/Proofs/EfficientNetBackB0.lean` |
| MobileNetV2 | ✅ | ✅ | `LeanMlir/Proofs/MobileNetV2BackB0.lean` |
| ResNet-34 | ✅ | ✅ | `LeanMlir/Proofs/ResNet34BackB0.lean` |
| **ConvNeXt** | ✅ | ✅ | `LeanMlir/Proofs/ConvNeXtBackB0.lean` |
| **ViT** | ✅ (block) | n/a | `LeanMlir/Proofs/ViTBackB0.lean` |

**✅ DONE (2026-06-14). All 5 nets have whole-block backward-graph faithfulness.**
See "Shipped" below — the §2/§3 plan assumptions below are partly WRONG; read the
Shipped notes for what actually landed.

---

## Shipped (2026-06-14)

Commits on `main`: `2cd8c9d` (ConvNeXt), `c0ae299` (ViT heads=1), `f590c5c`
(ViT → general multi-head), `e509514` (ViT → vec-LN), `d2e4507` (ViT whole-net).
All certify to `[propext, Classical.choice, Quot.sound]`, `sorryAx` 0,
`check_audit_coverage` passes.

**ConvNeXt — §2's "batched / new LayerNorm-batched primitive" assumption was WRONG.**
ConvNeXt's whole verified stack is **per-example / batch-1** (LayerNorm is per-example
separable, so NONE of EfficientNet's `batchMap`/`bnBatchLA` machinery applies —
`ConvNeXtChainClose.lean:8`). The capstone targets the per-example `convNextBlock_has_vjp`
/ `cnxDownW_has_vjp`, mirroring `mbconvResidual_backGraph_faithful`. **Zero new primitives**
were needed — every backward token already existed: `depthwiseBack`, `bnBack` (LN backward,
since `layerNorm_has_vjp` is *definitionally* `bn_has_vjp`), `convBack`, `geluBack`,
`layerScaleF`, `convStridedBack` (downsample). Capstones: `cnxResidBlockBackGraph_faithful`,
`cnxDownBackGraph_faithful`.

**ViT — went BEYOND the plan, to full production-config block parity.** §3 was right that
ViT lives in the `HasVJPMat` framework and the SDPA/attention backward is the crux, but the
delivered capstone is **general multi-head (heads = hm1+1) AND vec-LN**, matching the committed
`verified_mlir/vit_train_step.mlir` ViT-Tiny block config (heads=3 ⊂ general, vec-LN). Three
layers landed in `ViTBackB0.lean`: heads=1 scalar (`transformerBlockBackGraph_faithful`),
general multi-head (`transformerBlockBackGraphMH_faithful`), and multi-head + vec-LN
(`transformerBlockVBackGraphMH_faithful` ↔ `transformerBlockV_has_vjp_mat`). The MHSA crux
used a VJP-determinism lever (`hasVJPMat_backward_det`) to get past the opaque
`mhsa_has_vjp_mat` witness, then collapse to `sdpa_back_{Q,K,V}` (heads=1) / the per-head
`colSlab` fan-in via `headSliceF`/`headPadF`/`headsSumG` (multi-head). Vec-LN swaps the LN-back
to `lnRowBack(γ=1) ∘ rowScaleF(γv)` (bias backward = identity).

**Whole-NET ViT backward graph — ✅ DONE.** `vitNetBackGraph_faithful` (`ViTBackB0.lean`):
`den (vitNetBackGraph …) = (vitForwardKV_has_vjp …).backward x dy` at the full production config
(depth-k tower + multi-head + vec-LN + patch-embed stem + CLS/dense head). The net-level
extension beyond the block-level BackB0 series — ViT is the only net taken to whole-net backward
faithfulness. The patch-embed input-backward (image cotangent) needed one new SHlo token
`patchEmbedBack` (concrete transposed-patchify input grad `patchEmbedBackFlat`, proven = the VJP
backward; `den`+`skel` only, `skel` routed via the existing `.batched` tag — no parser change).
Stages: `classifierBackGraph` → `finalLNBackGraph` → `vitBodyBackGraphKMHV` (depth-k reverse
fold of the committed block backward) → `patchEmbedBackGraph` → `vitNetBackGraph`. 3-axiom clean.

All five nets ALREADY have, independently of this work: whole-net **VJP
correctness** (`*_has_vjp_correct`), **forward-graph faithfulness**
(`*FwdGraph*_faithful`), and **per-parameter rendered-gradient certs**
(`*_render_*_certified`). This session adds only the consolidated *whole-block
backward-graph* layer.

---

## 1. The proven recipe (followed 3× — copy it)

Read `EfficientNetBackB0.lean` (cleanest, global form) and `MobileNetV2BackB0.lean`
(pointwise `_at` form) first — they are the templates. The pattern per block:

1. **Batched stage defs** — each block stage is `activation ∘ bnBatchLA ∘ batchMap(conv-op)`
   (true batch-norm coupled on the `N·(c·h·w)` flat index; pointwise acts are `batchMap`).
   See `EfficientNetRenderPC.lean` ~L40–90 for `cbsB`/`dwbsB`/`projB` (swish) and
   `MobileNetV2BackB0.lean` for `cbrB`/`dwbrB` (relu6).
2. **Stage VJP + differentiability** — compose the per-op VJPs.
3. **Stage backward graph + `_faithful`** — chain the per-op backward-token
   faithfulness lemmas (each is `rfl`/short `simp`).
4. **Body VJP + body backward graph + `_faithful`** — `vjp_comp` (global) or
   `vjp_comp_at` (pointwise) over the stages.
5. **Block capstone** — wrap the body in the residual/skip:
   `residualBackGraph_faithful` (identity skip) or the projection-skip analog,
   plus any outer activation. Add `#print axioms` to `tests/AuditAxioms.lean`.

**Global vs `_at` (the key fork):**
- **Globally-smooth activations** (swish, **GELU**, sigmoid, LayerNorm, layerScale,
  dense, conv) → clean **global `HasVJP`** + `vjp_comp`. EfficientNet is the template.
  **ConvNeXt and ViT are entirely in this regime** — no kink threading. Easier than r34/mnv2.
- **Kinked activations** (relu, relu6) → pointwise `HasVJPAt` with a measure-zero
  smoothness hypothesis (`x k ≠ 0`, etc.), threaded via `vjp_comp_at` +
  `HasVJP.toHasVJPAt`. Only r34 (relu) and mnv2 (relu6) needed this. **N/A here.**

**Reusable core primitives (already exist, all in `StableHLO.lean` + faithfulness in `EfficientNetBackB0.lean`):**
- `convBackBatched` / `convStridedBackBatched` (stride-1 / stride-2 regular conv backward).
- `depthwiseBackBatched` / `depthwiseStridedBackBatched` (depthwise, both strides).
- `bnBatchLABack` (true batch-norm backward) + `bnBatchLABack_faithful`.
- `den`/`skel` for batched ops route through a generic `.batched` node ⇒ **adding a new
  batched-backward constructor needs only `den` + `skel` cases + a faithfulness lemma; NO
  parser/Raw/toToks change** (this is how the 4 strided primitives went in cheaply).

**Per-op backward-token faithfulness already proven (rfl-level unless noted):**
- relu: `selectPos_faithful` (`StableHLO.lean:782`); relu6: `selectMid_faithful` (`:796`).
- gelu: `(gelu_has_vjp n).backward = den (geluBack…)` (`IR.lean:274`, rfl).
- LayerNorm: `(layerNorm_has_vjp n ε γ β hε).backward …` (`IR.lean:356`).
- conv/dense/depthwise: the `*Back*_faithful` family in `StableHLO.lean` / `EfficientNetBackB0.lean`.

**Composition machinery (exists):** `vjp_comp`, `vjp_comp_at`, `HasVJP.toHasVJPAt`,
`residual_has_vjp` / `residual_has_vjp_at`, `residualProj`/`biPath` VJPs, `residualBackGraph(_faithful)`.

**Verification recipe (run after each capstone):**
```
export PATH=$HOME/.elan/bin:$PATH
lake build LeanMlir.Proofs.<NewFile>          # green, no sorry/admit/axiom
grep -nE '\bsorry\b|\badmit\b|^axiom ' LeanMlir/Proofs/<NewFile>.lean   # empty
lake env lean tests/AuditAxioms.lean 2>&1 | grep -iE '<capstone names>'  # [propext, Classical.choice, Quot.sound]
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c sorryAx              # 0
python3 scripts/check_audit_coverage.py        # passes
```
Register the new module in `lakefile.lean` (Proofs roots) and import + `#print axioms`
it in `tests/AuditAxioms.lean`.

---

## 2. ConvNeXt — difficulty: MEDIUM (global form, but new stage primitives)

**Block body** (`ConvNeXtChainClose.lean:18-20`, `ConvNeXt.lean:25`):
```
o = addV( layerScale γls ( proj₁ₓ₁ ( gelu ( exp₁ₓ₁ ( LN ( dw₇ₓ₇ (x) ) ) ) ) ), x )
```
i.e. `residual( layerScale ∘ proj ∘ gelu ∘ expand ∘ LN ∘ depthwise7×7 )` —
**no post-add activation** (simpler than r34's outer relu). All ops globally smooth.

**Reusable (exists):**
- Per-op VJP + backward faithfulness: `gelu_has_vjp` (rfl), `layerNorm_has_vjp`
  (`IR.lean:356`), `layerScale_has_vjp` (diagonal — its input-VJP is `layerScale γ`
  itself, so the render emits a *second* `layerScaleF` on the cotangent, see
  `ConvNeXtChainClose.lean:41-47` — no backward token needed).
- Forward stages + faithfulness: `ConvNeXtFullT.lean` (`layerScaleChF_faithful`,
  `geluF_faithful`, `bnF_faithful`, `flatConvF_faithful`), `convNextForwardT_has_vjp`,
  `convNextStageK_has_vjp`.
- **Downsample**: `cnxDownW = flatConvStride2(2×2) ∘ LN`, with `cnxDownW_has_vjp`/`_diff`
  already proven (`ConvNeXtFullT.lean:108-121`). The strided 2×2 conv backward is covered
  by the existing `convStridedBackBatched` (note: 2×2 stride-2, confirm the kernel/stride
  generalize — `convStridedBack` is the per-example op it lifts).

**Likely NEW pieces to build:**
- A **batched LayerNorm backward** stage. NB: LN normalizes over channels *per token*,
  so it does NOT couple the batch like true batch-norm — it should be a plain `batchMap`
  of the per-example `layerNorm_has_vjp`, i.e. simpler than `bnBatchLA`. Confirm whether a
  `bnBatchLA`-style LN node is even needed or LN is already `batchMap`-expressible.
- Batched stage backwards for `dw7×7-LN`, `expand-gelu`, `proj`, `layerScale`, composed.
  `depthwiseBackBatched` should be kernel-generic (7×7 ok); `convBackBatched` for the 1×1s.
- The two capstones: `cnxBlockBackBatchedGraph_faithful` (identity/residual) and
  `cnxDownBackBatchedGraph_faithful` (downsample = `cnxDownW`, no residual).

**Suggested order:** stage backwards (LN, gelu-expand, proj, layerScale) → body VJP +
backward graph → identity-block capstone → downsample capstone.

---

## 3. ViT — difficulty: HARD (different framework: per-token "mat" VJP)

**Block** (standard transformer): `x ↦ x + MLP(LN(x'))` where `x' = x + MHSA(LN(x))`;
`MLP(z) = dense(Wfc2,bfc2, gelu(dense(Wfc1,bfc1, z)))`. Two residual adds, two LNs.

**The key difference:** ViT lives in the **matrix / per-token VJP framework**
(`Attention.lean`), NOT the batched-flat `bnBatchLA`/`batchMap` framework the conv nets
use. So the capstone is "rendered backward graph denotes the **`vjpMat`** composition",
using `vjpMat_comp` / `biPathMat_has_vjp` / `*_has_vjp_mat`. Plan the ViT capstone in that
framework — don't try to force it into the batched-flat one.

**Reusable (exists — the hard parts are already done!):**
- Attention backward: `sdpa_back_Q/K/V` (`Attention.lean:678-688`),
  `rowSoftmax_has_vjp_mat` (`:532`), and the rendered-cotangent equalities
  `vitCotD{Q,K,V}_eq_sdpa_back` (in AuditAxioms) — the softmax/attention backward is proven.
- Per-token pieces: `dense_per_token_has_vjp_mat`, `gelu_per_token_has_vjp_mat`,
  `transformerMlp_has_vjp_mat` (`Attention.lean` ~1910-1960), `vjpMat_comp`, `biPathMat_has_vjp`.
- Whole-block forward VJP `vit_body_has_vjp_mat`; forward-graph `vitFwdGraph_faithful`;
  per-param certs `vit_render_*_certified`.

**NEW pieces to build:** the rendered **backward-graph** for the transformer block in the
mat framework — MHSA backward graph (compose Q/K/V dense-back + `sdpa_back` + out-dense-back)
+ LN-back + MLP backward graph + the two residual fan-ins — proven `= vit_body_has_vjp_mat.backward`.
The render tokens for the mat-side backward likely need their own `den`-faithfulness lemmas
(analogous to `selectPos_faithful` etc.) if not already present — check `Attention.lean` and
the ViT render file. This is the bulk of the ViT effort.

**Suggested order:** MLP block backward-graph faithful (two dense + gelu, easiest) → LN-back
→ MHSA backward-graph faithful (the attention core, reuse `sdpa_back`/`vitCotD*`) → assemble
the full transformer-block capstone with the two residuals.

---

## 4. Workflow that worked (recommend repeating)

The r34/enet/mnv2 capstones were each landed by a **single focused subagent** given (a) the
template file, (b) the exact reusable lemma locations, (c) the specific block structure, and
(d) "iterate `lake build` to green, no `sorry`, add `#print axioms`, report — stop and report
honestly if blocked rather than fudging." Each built on roughly the first or second try.
Then **independently verify** (the recipe in §1) before committing — don't trust the agent's
self-report on a proof claim.

Commit each net separately (per-commit sign-off is the repo rule; commit straight to `main`
is fine once approved). Suggested commits: one for ConvNeXt (incl. any new LN-batched
primitive), one for ViT.

## 5. Definition of done
- `ConvNeXtBackB0.lean` + `ViTBackB0.lean` exist, registered, build green.
- Identity AND downsample (ConvNeXt) / transformer-block (ViT) capstones proven.
- All new theorems certify to `[propext, Classical.choice, Quot.sound]`; `sorryAx` count 0;
  `check_audit_coverage.py` passes.
- All 5 nets then have complete whole-block backward-graph faithfulness.
