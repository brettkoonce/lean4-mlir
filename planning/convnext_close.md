# Planning — closing ConvNeXt (ch9) both ways

The ConvNeXt analogue of `planning/{efficientnet,mobilenetv2,resnet34}_close.md`. Goal: bring the
representative ("small", 2-block) ConvNeXt to the **"closed both ways"** bar — **(a)** train-step text =
name-threaded `pretty` of a *proven* `SHlo` forward graph, **(b)** every param output certified
`θ − lr·certified-gradient`, 3-axiom-clean.

**Headline: ConvNeXt is the CHEAPEST close of the flagship CNNs — and the deliberate CONTRAST to
EfficientNet.** Three reasons:
1. **The forward graph already exists and is proven faithful.** `convNextFwdGraph` +
   `convNextFwdGraph_faithful` (`StableHLO.lean`) — the representative 2-block net, denoting the proven
   `convNextForward`. So Item A is essentially DONE (only audit-wiring is missing).
2. **LayerNorm is per-example separable ⇒ NO batched index** (unlike EfficientNet). The operational
   render (`tests/TestConvNeXtFwd.lean`) reduces LN over dimension `[1]` — the whole feature axis, **per
   example** (`[BS, n] → [BS]`) — which is exactly the proof's *scalar* `layerNormForward` (= `bnForward`
   over `c·h·w`). So a batch-1 `den` is faithful, as for MNV2/r34. The whole `batchOp`/`bnBatchF`/`batchMap`
   machinery EfficientNet needed (`[[efficientnet-batched-render]]`) is **not required here.** (Even the
   *faithful* channel-LN-over-NCHW — the optional upgrade below — is per-token, hence still
   batch-separable: ConvNeXt never hits EfficientNet's batch-coupling.)
3. **The two genuinely-new structures are already tokenized AND VJP-proven AND audited:** GELU
   (`geluF`/`geluBack`, `gelu_has_vjp_correct`) and per-element layer-scale (`layerScaleF`,
   `layerScale_has_vjp_correct`). Nothing new to denote.

So the remaining work is concentrated in **Item C** (param close — mostly reuse; the only new param
families are layer-scale `γ` and scalar-LN `γ/β`), **Item B** (structured render via `pretty`), and
**Item D** (the block cotangent chain) — all at the easy batch-1 level.

---

## 0. Architecture (`ConvNeXt.lean`, representative 2-block "small" net)
stem-patchify (**1×1** conv `ic→c`) → stem-LN → **block₁** → **block₂** → GAP → head-LN → dense.
Each block = `residual( layerScale γ ∘ project(1×1) ∘ gelu ∘ expand(1×1, 4×) ∘ layerNorm ∘ depthwise(7×7) )`
(`convNextBlock`/`convNextBlockBody`). Identity skip (no projection, no post-add activation). All-smooth
(GELU smooth, LN smooth given `0<ε`, conv/layerScale linear) ⇒ only the `0<ε` LN hyps, no kink hyps.
Hand render: `tests/TestConvNeXtTrain.lean`; trainer `MainConvNeXtTrain.lean`.

**Two documented representation simplifications (same on the proof AND render sides — not gaps):**
- **Scalar LayerNorm.** `layerNormForward n ε γ β` normalizes over the *whole* flattened `n=c·h·w` with
  scalar `γ,β` (= `bnForward`); the render reduces `[1]` per example to match. True ConvNeXt LN is
  per-token over the channel axis C. Faithful channel-LN is the optional upgrade (still batch-separable).
- **1×1 patchify stem** (real ConvNeXt is 4×4 stride-4). A strided-stem upgrade if wanted (the
  `flatConvStride2` family / a 4×4 variant).

## 1. Starting state (op + block + whole-net VJPs all proven, 3-axiom clean)
`ConvNeXt.lean` (424 ln): `layerScale`/`layerScale_has_vjp`(+`pdiv_layerScale`); `convNextBlockBody` +
`convNextBlockBody_has_vjp[_at]`; `convNextBlock = residual(body)` + `convNextBlock_has_vjp[_at]`;
whole-net `convNextForward` + `convnext_has_vjp[_at]` (+`_correct`, an UNCONDITIONAL global VJP —
all-smooth, only `0<ε`). `LayerNorm.lean`: `layerNormForward`/`layerNorm_has_vjp`, `gelu`/`gelu_has_vjp`.
`StableHLO.lean`: `convNextFwdGraph` (tokens: `flatConvF`×5, `depthwiseF`×2, `bnF`×4 [scalar LN],
`geluF`×2, `layerScaleF`×2, `gapF`, `denseF`, `addV`×2 residual) + `convNextFwdGraph_faithful`. Backward
tokens that exist: `geluBack`, `bnBack` (scalar LN back), `convBack`/`depthwiseBack`, `dotOut` (dense back).
Audited (`tests/AuditAxioms.lean`): `convnext_has_vjp[_correct]`, `convnext_has_vjp_at_correct`,
`layerScale_has_vjp_correct`, `gelu_has_vjp_correct`, `IR.gelu_back_bridge`.

---

## 2. The four rungs

### Item A — forward graph — ✅ **DONE** (representative; audit-wired)
`convNextFwdGraph` + `convNextFwdGraph_faithful` (`StableHLO.lean`) denote the proven
`convNextForward` at the representative dims, **batch-1** (scalar LN, reduce `[1]` per example — no
batched index). **Audit-wired** (`tests/AuditAxioms.lean`, ConvNeXt RENDER section): `#print axioms
StableHLO.convNextFwdGraph_faithful` ⇒ `[propext, Classical.choice, Quot.sound]`, 3-axiom clean,
matching the other nets' forward graphs. No new tokens. (Contrast: EfficientNet needed a whole
batched-token layer because its BN couples the batch; ConvNeXt does not.)

### Item C — the param close — ✅ **DONE** (`ConvNeXtClose.lean`, audit-wired)
Every param output certified `θ − lr·(certified ∂forward/∂θ · cotangent)`, 3-axiom clean:

| family (render SSA)                       | forward fn            | certified by                                   |
|-------------------------------------------|-----------------------|------------------------------------------------|
| stem 1×1 conv `W/b`                       | `flatConv`/`conv2d`   | `conv_weight/bias_grad_bridge` (M3, **reuse**) |
| depthwise **7×7** `W/b`                   | `depthwiseConv2d`     | `cnx_render_dw7{W,b}_certified` (kernel-general bridges **pinned at 7×7**) |
| expand/project 1×1 conv `W/b`             | `flatConv`/`conv2d`   | `conv_weight/bias_grad_bridge` (M3, **reuse**) |
| dense head `Wd/bd`                        | matmul / +bias        | M2 `weight/bias_grad_bridge` (**reuse**)       |
| **layer-scale `γ`** (`layerScaleF`)       | `γ ⊙ x`               | `cnx_render_lsgamma_certified` (**new**): `pdiv_layerScale_gamma` (`∂(γⱼxⱼ)/∂γᵢ = xᵢδᵢⱼ`) ⇒ `dγ = x ⊙ dy` |
| **scalar-LN `γ/β`** (the `bnF` sites)     | `layerNormForward`    | `cnx_render_ln{gamma,beta}_certified` (**new**, see below) |
| gelu                                      | —                     | no parameters                                  |

**A planning correction discovered during the build:** the scalar-LN `γ/β` grads were NOT the free
"scalar special-case of `cifar_bn_render_{gamma,beta}`" the plan claimed — `BatchNorm.lean` had
deliberately left `bn_grad_gamma`/`bn_grad_beta` as *definitions only* ("scalar params don't fit the
`pdiv`/`HasVJP` framework cleanly"). `ConvNeXtClose.lean` closes that gap with the **`Vec 1` embedding**:
as a function of `γ' : Vec 1`, LN is affine (`fun y k => x̂ₖ·y 0 + β`), so the CifarBnClose Jacobian
recipe applies with the constant channel map `Fin n → Fin 1` — certifying the rendered whole-`n` reduces
`dγ = Σ dy·x̂`, `dβ = Σ dy` against the certified Jacobian. Affine in the params ⇒ no `0<ε`. The
layer-scale `γ` bridge is the symmetry mirror of `pdiv_layerScale` (roles of γ/x swapped). The 7×7
depthwise pins are verbatim instantiations of the kernel-general `mnv2_render_depthwise{W,b}_certified`
(stride-1 only — ConvNeXt blocks keep resolution). All audited in `tests/AuditAxioms.lean`.

### Item B — structured render — ✅ **DONE** (`tests/TestConvNeXtTrainPC.lean`)
Forward + the whole backward cotangent chain proof-rendered via `pretty` over the `convNextFwdGraph`
tokens (fn `@convnext_rep_train_step`, representative dims 3×32² / c=32 / cExp=128 / dw 7×7 / 10
classes, BS=32; 26 params + x + onehot). Backward all by token: `geluBack`, `bnBack` (scalar-LN
input-VJP), `convBack`/`depthwiseBack`, `dotOut` (dense). **Layer-scale backward reuses `layerScaleF`
itself** — `layerScale`'s input-VJP is `γ ⊙ dy` (diagonal/symmetric: applying the forward token to the
cotangent IS the backward), so no new backward token, exactly as planned. Hand-emitted only the param
grads (conv/dw/dense W/b, layer-scale `dγ = x⊙dy` via multiply+reduce[0], scalar-LN `dγ/dβ` via
recompute-x̂ + reduce `[0,1]` → `tensor<f32>`) and the GAP backward (reshape/broadcast/divide).
**Validation:** the committed renderer (`TestConvNeXtTrain.lean` → `verified_mlir/convnext_train_step.mlir`)
is FULL ConvNeXt-T ([3,3,9,3], 180 params, 4×4-stride-4 stem) — signatures don't match the
representative, so no swap-parity is possible; validated instead with the `render_parity.py` **ref-only
smoke** (iree-compile + gfx1100 run: 26/26 outputs all-finite, 26/26 non-zero). `render_parity.py` was
extended to parse 0-d scalar `tensor<f32>` params (the scalar-LN `γ/β`) — a fix that also newly enables
the harness on the committed full ConvNeXt-T (verified 182/180 parse, plus mnv2 84/82, r34 148/146).

### Item D — cotangent chain — ✅ **DONE** (`ConvNeXtChainClose.lean`, audit-wired)
The analogue of `MobileNetV2ChainClose`: each param pinned to the cotangent the **actual backward
chain delivers**, composing the rendered backward denotations through the block — layer-scale back
(= `layerScale γls` on the cotangent, `cnxCotP`) → project conv-back → gelu mask (`dy⊙geluScalarDeriv`,
`cnxCotE`) → expand conv-back (`cnxCotN`) → scalar-LN input-VJP (`bn_grad_input`, `cnxCotD`) →
depthwise back + the skip's `dyOut` (`cnxCotXin`); stem = `bn_grad_input` at the saved patch
(`cnxStemCot`). No stride split (blocks keep resolution — one cotangent set covers every block) and
no post-add activation (the `addV` passes `dyOut` straight through, so the layer-scale pin is the
exact passthrough). **Goes beyond the MNV2/r34 precedent:** the ConvNeXt-signature param families
are chain-pinned too — layer-scale `γ` at `dyOut` and the block scalar-LN `γ/β` at `cnxCotN` — so
every block param (not just conv/dw) is certified at its actual chain cotangent. Blocks compose by
instantiation (block 1's `dyOut` := block 2's `cnxCotXin`; `dyStem` := block 1's `cnxCotXin`).
**Pure-Lean, batch-1** — no batched-VJP machinery (the EfficientNet `batchMap_has_vjp` lift is NOT
needed; ConvNeXt stays per-example). 11 theorems audit-wired, 3-axiom clean (audit now 282/282).

---

## 3. Order & status
1. **Item A** ✅ DONE — `convNextFwdGraph` + `_faithful` (representative, scalar LN, batch-1), now
   audit-wired into `AuditAxioms.lean` (3-axiom clean).
2. **Item C** ✅ DONE — param close (`ConvNeXtClose.lean`): 7×7 depthwise pinned; layer-scale `γ`
   (`pdiv_layerScale_gamma` ⇒ `dγ = x⊙dy`) and scalar-LN `γ/β` (the `Vec 1` embedding — NOT free
   reuse; `bn_grad_gamma/beta` were definitions-only) bridged + certified. Audit-wired, 3-axiom clean.
3. **Item B** ✅ DONE — structured render (`tests/TestConvNeXtTrainPC.lean`, fn
   `convnext_rep_train_step`): all backward by token (layer-scale back = `layerScaleF` on the
   cotangent); validated iree-compile + ref-only GPU smoke (26/26 finite/non-zero — no
   same-signature committed ref; the committed renderer is full ConvNeXt-T). `render_parity.py`
   extended for 0-d scalar `tensor<f32>` params.
4. **Item D** ✅ DONE — block cotangent chain (`ConvNeXtChainClose.lean`, batch-1): every block
   param (incl. layer-scale `γ`, scalar-LN `γ/β` — beyond the MNV2/r34 conv/dw-only precedent)
   + the stem pinned to the actual chain cotangent. Audit-wired, 3-axiom clean.

**The small ConvNeXt is closed both ways, all four rungs done** (A+C+B+D). Remaining: only the
optional upgrades below (faithful channel-LN, 4×4 stride-4 stem, full `[3,3,9,3]` depth).

## Handoff notes
- **"Small version" = the representative `convNextForward`** (2 blocks, 1×1 stem, scalar LN) — already
  what's proven. Start there; the EfficientNet path shows the representative is a legitimate close at the
  same granularity as the proven VJP. Scaling (full depth `[3,3,9,3]`, 4×4 stride-4 stem, faithful
  channel-LN) is a later mechanical/optional pass — "come back to scaling later," as agreed.
- **Why ConvNeXt is easier than EfficientNet:** ConvNeXt's normalization (LayerNorm) is per-token/
  per-example separable, so the batch-1 `den` is faithful — the whole batched-token apparatus from Item A
  of EfficientNet (`StableHLO.batchOp`/`bnBatchF`/`batchMap`, and the `batchMap_has_vjp` backward lift in
  `EfficientNetChainClose.lean`) is **not needed**. See `[[efficientnet-batched-render]]` for that contrast
  and for the proof-engineering gotchas that DO carry over to Items B/D: prove faithfulness/VJP **per-block**
  then chain (don't reduce the whole net at once); write the ℝ-forward in **nested-application** form (not
  `∘`) so the forward close is pure delta; give free-floating stage lemmas **explicit `(h := …)(w := …)`**.
- **Templates to copy:** `ResNet34RenderPC.lean` (per-block forward graphs + `_faithful` + full-net chain),
  `MobileNetV2Close.lean`/`EfficientNetClose.lean` (param close + bridge reuse + the certified-θ table),
  `MobileNetV2ChainClose.lean` (the Item D cotangent-chain shape), `tests/TestMobilenetV2TrainPC.lean`
  (the `pretty`-rendered train step for Item B).
- **Optional upgrades (all batch-separable — none force a batched index):** faithful channel-LN-over-NCHW
  (reduce over C per `(n,h,w)` — a new `layerNormChannel` token + VJP, the real ConvNeXt LN); 4×4 stride-4
  patchify stem; full `[3,3,9,3]` depth.

---

## Full-architecture status (ladder audit, 2026-06-09)

**ConvNeXt is one of two nets NOT closed at full architecture** (the other: ViT — see
`planning/vit_close.md`'s handoff). What exists: the representative 2-block close (all four
rungs, above) and the committed full ConvNeXt-T `[3,3,9,3]` production render
(`tests/TestConvNeXtTrain.lean` — hand fragments, GPU-trained, NOT graph-closed). Contrast:
ResNet-34 is closed at the full 16-block `[3,4,6,3]` net (`ResNet34RenderPC`) and
EfficientNet-B0 at all 16 MBConv blocks (`EfficientNetFullB0`).

### Scaling handoff — full ConvNeXt-T `[3,3,9,3]`
Mirror the ViT depth-k recipe (`planning/vit_close.md` next-session handoff §2), which was
designed off this exact gap:
1. **Depth-k within a stage**: `convNextForwardK (k) (params : Fin k → CnxBlockParams)` —
   bundle the 9-param block tuple in a structure; VJP by induction with the existing
   `convNextBlock_has_vjp` as the step (the blocks are same-shape within a stage — easier
   than ViT, no index changes). Graph by fold; faithfulness by induction off a per-block
   den lemma (extract one from `convNextFwdGraph_faithful`'s block segment, the analogue of
   `vitBlockGraph_den_aux`).
2. **Downsample stages**: between stages, ConvNeXt-T has LN + 2×2/s2 conv downsamples and
   channel widening (96→192→384→768). The strided-conv tokens + VJPs exist (ch6
   `flatConvStridedF`, `StridedConv.lean`); what's new is only the stage-boundary chaining
   (same `vjp_comp` shape as the stem).
3. **4×4/s4 patchify stem**: the even-kernel non-overlapping strided conv — `patchEmbedF`'s
   conv core at P=4 without the CLS/pos plumbing, or ch9's existing patchify recipe; the
   ViT §E patch-close showed the kernel-Jacobian is the cheap const×reindex case.
4. **Faithful channel-LN** (optional, beyond the scalar/vector-over-flat forms): real
   ConvNeXt LN normalizes over C per (n,h,w) — a `layerNormChannel` token + rowwise-style
   VJP over the channel fibre. The ViT vector-LN file (`ViTVecLN.lean`) is the template for
   adding an LN variant end-to-end (forward+VJP+tokens+faithfulness+param bridges+chain pins).
5. Item C/D are then free-to-cheap: every param family is already bridged
   (`ConvNeXtClose`/`ConvNeXtChainClose` are dim-generic); only the stage-boundary
   downsample conv W/b needs the strided bridges (proven for MNV2/r34).
