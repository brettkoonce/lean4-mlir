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

### Item C — the param close — [NEXT; mostly reuse, two small new families]
`ConvNeXtClose.lean` — certify each param output `θ − lr·(certified ∂forward/∂θ · cotangent)`:

| family (render SSA)                       | forward fn            | certified by                                   |
|-------------------------------------------|-----------------------|------------------------------------------------|
| stem 1×1 conv `W/b`                       | `flatConv`/`conv2d`   | `conv_weight/bias_grad_bridge` (M3, **reuse**) |
| depthwise **7×7** `W/b`                   | `depthwiseConv2d`     | `mnv2_depthwise_weight/bias_grad_bridge` (kernel-general — **pin 7×7**) |
| expand/project 1×1 conv `W/b`             | `flatConv`/`conv2d`   | `conv_weight/bias_grad_bridge` (M3, **reuse**) |
| dense head `Wd/bd`                        | matmul / +bias        | M2 `weight/bias_grad_bridge` (**reuse**)       |
| **layer-scale `γ`** (`layerScaleF`)       | `γ ⊙ x`               | **new tiny bridge**: `∂(γⱼxⱼ)/∂γᵢ = xᵢδᵢⱼ` ⇒ `dγ = x ⊙ dy` |
| **scalar-LN `γ/β`** (the `bnF` sites)     | `layerNormForward`    | scalar special-case of the bn γ/β grad (cf. `cifar_bn_render_{gamma,beta}` at the whole-`n`, scalar shape) |
| gelu                                      | —                     | no parameters                                  |

The only genuinely-new param bridges are **layer-scale `γ`** (multiplicative-bias analogue; `pdiv_layerScale`
already gives the diagonal Jacobian) and the **scalar-LN `γ/β`** grads (single scalars over the whole `n`).
Everything else is verbatim reuse at the ConvNeXt shapes. The 7×7 depthwise is the only kernel size to pin
(prior nets used 3×3/5×5).

### Item B — structured render — [given C; the backward tokens already exist]
`tests/TestConvNeXtTrainPC.lean`: forward + backward via `pretty` over the `convNextFwdGraph` tokens.
Backward pieces all have tokens: `geluBack` (smooth, no kink), `bnBack` (scalar-LN input-VJP), `convBack`/
`depthwiseBack`, `dotOut` (dense). **Layer-scale backward reuses `layerScaleF` itself** — `layerScale`'s
input-VJP is `γ ⊙ dy` (diagonal/symmetric: applying the forward token to the cotangent IS the backward),
so no new backward token. Hand-emit only the param grads (conv/dw/dense W/b, layer-scale `dγ = x⊙dy`,
scalar-LN `dγ/dβ`). Validate with `scripts/render_parity.py` (cf. the MNV2/r34 `TrainPC` peers).

### Item D — cotangent chain — [optional; the block chain, batch-1]
`ConvNeXtChainClose.lean` — the analogue of `MobileNetV2ChainClose`. Pin each conv/dw param to the
cotangent the **actual backward chain delivers**, composing the rendered backward denotations through the
block: layer-scale back (`γ⊙·`) → project conv-back → gelu back (`dy⊙gelu'`) → expand conv-back → scalar-LN
input-VJP (`bn_grad_input`) → depthwise back; the residual adds the skip cotangent. `convNextBlock_has_vjp`
is the per-block head start. **Pure-Lean, batch-1** — no batched-VJP machinery (the EfficientNet
`batchMap_has_vjp` lift is NOT needed; ConvNeXt stays per-example).

---

## 3. Order & status
1. **Item A** ✅ DONE — `convNextFwdGraph` + `_faithful` (representative, scalar LN, batch-1), now
   audit-wired into `AuditAxioms.lean` (3-axiom clean).
2. **Item C** — param close (`ConvNeXtClose.lean`): reuse conv/depthwise/dense bridges; add the two small
   new families (layer-scale `γ`, scalar-LN `γ/β`); pin 7×7 depthwise.
3. **Item B** — structured render (`tests/TestConvNeXtTrainPC.lean`) + `iree-run-module` parity. All
   backward tokens exist (layer-scale back = `layerScaleF` on the cotangent).
4. **Item D** — optional block cotangent chain (`ConvNeXtChainClose.lean`), batch-1.

After Item B, the small ConvNeXt is closed both ways.

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
