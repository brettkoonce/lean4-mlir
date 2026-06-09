# Planning — closing EfficientNet-B0 (ch8) both ways

The enet analogue of `planning/{mobilenetv2,resnet34}_close.md`. Goal: bring EfficientNet-B0 to the
"closed both ways" bar — **(a)** train-step text = name-threaded `pretty` of a *proven* `SHlo` forward
graph, **(b)** every param output certified `θ − lr·certified-gradient`, 3-axiom-clean.

**Headline: like r34, Item C is a FREE close** — even the two genuinely-new structures (squeeze-excite,
true batch-norm) add no new *param*-gradient bridge. The new work is concentrated in A/B/D. The two
genuinely-new pieces turned out to be: **(1)** true batch-norm **couples the batch** (reduce `[0,2,3]`),
so — unlike MNV2/r34's per-example BN — the forward graph cannot stay batch-1; Item A moves the whole
graph to the **batched index** `N·(c·h·w)` via new `StableHLO` machinery (`batchOp` + `BatchableOp`,
`bnBatchF`), with pointwise ops reused at the batched index. **(2)** **squeeze-excite** — which enters
as `BatchableOp.seBlock` (= `batchMap N` of the proven `seBlockFull`) in the forward, with the SE-gate
backward still the fiddly part in Item B.

---

## 0. Architecture (`MainEfficientNetVerified.lean`, 262 params)
stem 3×3-s2 conv → bn → swish → 16 **MBConv** layers (the real B0 `[t,c,n,s,k]` spec, stages
`[(1,16,1,1,3),(6,24,2,2,3),(6,40,2,2,5),(6,80,3,2,3),(6,112,3,1,5),(6,192,4,2,5),(6,320,1,1,3)]`,
3×3 **and 5×5** depthwise) → head 1×1 conv → bn → swish → GAP → dense. Each MBConv =
`project(1×1) ∘ SE ∘ depthwise(bn-swish) ∘ expand(1×1 bn-swish)` with a residual skip when `s=1 ∧ ic=oc`
(MBConv1/`t=1` blocks skip the expand). SE = squeeze(GAP) → reduce-`dense[c,r]` → swish → expand-`dense[r,c]`
→ sigmoid → channel-scale. **swish/sigmoid** activations (smooth, no kink). **True batch-norm** (`bnBatch`,
reduce `[0,2,3]`). Hand render: `tests/TestEfficientNetTrain.lean`; trainer `efficientnet-verified`.

## 1. Starting state (op + block + whole-net VJPs all proven)
`EfficientNet.lean` (599 ln): `sigmoid_has_vjp`, `seGate_has_vjp`, `seBlockFull_has_vjp` (the SE block!),
`convBnSwish`/`dwBnSwish_has_vjp`, `mbconvBody`/`mbconvResidual_has_vjp[_at]`, whole-net
`efficientnet_has_vjp[_at]` (+`_correct`, an UNCONDITIONAL global VJP — all-smooth, only `0<ε`).
`bnBatchTensor4_has_vjp` + `bnBatchTensor4_grad_input` (PerChannelBN.lean) — the **true batch-norm VJP**,
proven. swish/sigmoid/gapF SHlo tokens exist; **no SE/channel-scale token**; **no enet forward graph**.

The proven `efficientnetForward` uses **scalar `bnForward`**; the render uses **batch-norm**. The
BN-flavor reconciliation (Item A) is `bnBatchTensor4 = bnchwBack ∘ bnPerChannelFlat oc (N·h·w) ε γ β ∘
bnchwFwd` — batch-norm IS per-channel BN over the merged batch+spatial axis `m = N·h·w`.

---

## 2. The four rungs

### Item C — the param closes — ✅ **DONE** (2026-06-09), a FREE close
`LeanMlir/Proofs/EfficientNetClose.lean` (6 theorems, 3-axiom clean, audited). **No new VJP.** Every
param family reuses an existing bridge:
- 1×1 conv W/b (expand/project/head) → `cnn_render_conv{W,b}_certified`; stem 3×3-s2 →
  `mnv2_render_stem_conv{W,b}_certified`.
- depthwise **3×3** W/b (s1/s2) → `mnv2_render_depthwise{W,b}[_strided]_certified` (**reuse**);
  depthwise **5×5** W/b (s1/s2) → the same bridges at `kH=kW=5`, **pinned** (`enet_render_dw5{W,b}[_strided]_certified`
  — the only kernel no prior net used).
- **batch-norm γ/β** → `cifar_bn_render_{gamma,beta}_certified` at `m=N·h·w` (`enet_render_bn{gamma,beta}_certified`):
  batch-norm = per-channel BN over the merged batch+spatial axis (`bnBatchTensor4` above), γ/β affine ⇒ the
  param grad has no batch coupling. (The hard batch-coupled *input*-VJP is `bnBatchTensor4_grad_input`.)
- **SE** squeeze/excite `dense[c,r]`/`[r,c]` W/b → M2 `weight/bias_grad_bridge` (dense, **reuse**); the
  channel-scale, sigmoid gate, swish carry no parameters. dense head → M2.

Cheaper than expected: SE introduces no param family beyond two dense layers, and batch-norm's param
grad collapses to per-channel-at-`N·h·w`.

### Item A — batched forward graph — ✅ **DONE** (2026-06-09), the genuinely-new structural assembly
`LeanMlir/Proofs/EfficientNetRenderPC.lean` (6 theorems, 3-axiom clean, audited). **The batch-coupling
realization:** EfficientNet's render emits **true batch-norm** (reduce `[0,2,3]`, `bnBatchTensor4`), which
couples the batch — unlike MNV2/r34 whose per-channel BN reduces `[2,3]` (per-example, so a batch-1 `den`
suffices). So a `bnBatchF` token "denoting `bnBatchTensor4`" **cannot** compose with the batch-1
conv/SE tokens; the graph had to move to the **batched index** `N·(c·h·w)`. New core machinery in
`StableHLO.lean`:
- `SHlo.batchOp` carrying a `BatchableOp` descriptor (conv / strided conv / depthwise / strided
  depthwise / dense / GAP / **the whole SE block**) — `den = batchMap N (denOp op)`, the proven
  per-example op lifted block-diagonally. The **squeeze-excite** enters here as `BatchableOp.seBlock`
  (= `batchMap N seBlockFull`), reusing the proven `seBlockFull`.
- `SHlo.bnBatchF` — the one batch-coupled op — `den = bnBatchLA` (= the proven `bnBatchTensor4`,
  reindexed to the network's left-assoc `N·(oc·h·w)` flat layout).
- pointwise swish/sigmoid/relu/`addV` (residual) **reuse their existing tokens** at the batched index —
  already block-diagonal there, no new token needed.

Faithfulness is **per-block** (`stem/mbNoExp/mbStrided/mbResid/head GraphB_faithful`: `den (block) =
block-fwd (den input)`), chained into `efficientnetFwdGraphB_faithful` — the r34 recipe, so the kernel
never reduces the whole net at once. (Two proof-engineering notes for B/D: the per-block lemmas keep the
shared residual subtree from being re-traversed; and the ℝ-forward is written in **nested-application**
form, NOT `∘`, so the final close is pure delta — a `∘` close blows the kernel up at these sizes.)

**Scope:** proven at TWO depths. (1) A representative batched B0 (`EfficientNetRenderPC.lean` /
`EfficientNetChainClose.lean`) exercising every structural element. (2) **The FULL 16-MBConv B0**
(`EfficientNetFullB0.lean`, 2026-06-09) — the real `[t,c,n,s,k]` spec, all 262 params, 4 stride-2
downsamples, 3×3+5×5 depthwise, 9 identity residuals — **closed both ways, 3-axiom clean**:
`efficientnetFwdGraphB_full_faithful` (forward graph) + `efficientnetForwardB_full_has_vjp` (backward VJP).
The scale-up was pure enumeration of the generic per-block machinery + one new block shape
(`mbExp` = expand + stride-1 + no-residual, stages 5/7); the per-block approach kept the kernel flat
(full forward + full VJP each ~2s). Params bundled via `MBW`/`MBWNoExp`/`B0Weights` structures.

### Item B — structured render — [given A; SE backward is the fiddly part]
`tests/TestEfficientNetTrainPC.lean`: forward + backward via `pretty`. swish/sigmoid backward tokens
exist (`swishBack`/`sigmoidBack`, smooth — no kink). The **SE-gate backward** (sigmoid VJP + channel-scale
back + the two dense backs + the squeeze/broadcast back) is the new piece. Validate with
`scripts/render_parity.py --fn efficientnet_train_step` (the harness is ready).

### Item D — batched backward (cotangent) math — ✅ **DONE (2026-06-09)**, `EfficientNetChainClose.lean`
The batched analogue of the backward half, on the subnet, at the same batched index `N·(c·h·w)` as the
forward graph — every gradient genuinely composed from the proven per-op VJPs (3-axiom clean throughout):

**Foundational batched VJP primitives:**
- `batchMap_has_vjp` — the genuinely-new lemma: a batch-separable op `batchMap N f` has a **block-diagonal**
  VJP (f's proven VJP applied per example). This is `seBlockFull_has_vjp` / the conv/depthwise/dense VJPs
  "lifted by batchMap" to the whole batch. Mechanically free: `batchMap N f = Mat.flatten ∘ (apply f per
  row) ∘ Mat.unflatten`, so `rowwise_has_vjp_mat` + `hasVJPMat_to_hasVJP` (Tensor.lean) close it.
  (`+ batchMap_differentiable`.)
- `bnBatchLA_has_vjp` — the one batch-coupled op: the proven `bnBatchTensor4_has_vjp` reindex-conjugated
  to the network's flat index (`+ bnBatchLA_differentiable`, `+ reindex_has_vjp`, a reusable generic
  reindex VJP). swish/sigmoid are pointwise ⇒ `swish_has_vjp` applies directly at the batched index.

**Per-stage → per-block → whole subnet (all via `vjp_comp`):** stage VJPs+diffs (cbsB/stemB/dwbsB/dwbsSB/
seB/projB), then the per-block gradients `mbNoExpFwdB/mbStridedFwdB/mbResidFwdB/headFwdB_has_vjp` (residual
via `residual_has_vjp`), then the capstone `efficientnetForwardB_has_vjp` — `HasVJP` of the whole batched
subnet (the batched, true-batch-norm + SE analogue of `efficientnet_has_vjp`). The subnet now has **both
directions proven** for the block decomposition we scale: forward faithfulness (Item A) + backward VJP.

(Proof-engineering note for scaling: the subnet VJP is stated on the `∘`-composition of the blocks — which
IS `efficientnetForwardB` by construction — because relating the nested-application forward spelling to a
`∘`-composition forces the kernel to re-reduce the whole net. The per-block VJPs stay opaque, so the
capstone closes structurally.) Optional remaining: the ChainClose-style param-pinning à la `MobileNetV2ChainClose`.

---

## 3. Order & status
1. **Item C** ✅ DONE — free close, `EfficientNetClose.lean`.
2. **Item A** ✅ DONE — batched forward graph (`EfficientNetRenderPC.lean`), true batch-norm + SE, at
   the batched index `N·(c·h·w)`; new `StableHLO` tokens `batchOp`/`bnBatchF` + the `BatchableOp.seBlock`.
   Representative depth; full 16-block enumeration is mechanical if wanted.
3. **Item B** — structured render + `iree-run-module` parity (harness ✅ ready). NOTE: `batchOp`/`bnBatchF`
   currently `skel` to an explicit `// [Item B render TODO]` marker (`emitTok`) — Item B fills in the real
   batched StableHLO emission (the `[N,C,H,W]` conv/dw/SE fragments + the reduce-`[0,2,3]` batch-norm,
   mirroring `tests/TestEfficientNetFwd.lean`).
4. **Item D** — ✅ DONE (`EfficientNetChainClose.lean`): batched backward math. The new lemma
   `batchMap_has_vjp` (block-diagonal VJP lift) + `bnBatchLA_has_vjp` (true-BN), composed via `vjp_comp`
   into per-stage → per-block (`mb*FwdB_has_vjp`) → whole-subnet `efficientnetForwardB_has_vjp`. SE rides
   in as `batchMap N seBlockFull` (reusing `seBlockFull_has_vjp`); swish pointwise. 3-axiom clean. The
   subnet now has BOTH directions proven. (Optional remaining: ChainClose-style param-pinning.)

The genuinely-new EfficientNet work was **(a)** the batch-coupling of true batch-norm (forcing the whole
graph to the batched index `N·(c·h·w)` — the new `batchOp`/`bnBatchF` machinery) and **(b)** squeeze-excite
(`BatchableOp.seBlock`). After Item B, EfficientNet is closed both ways.
