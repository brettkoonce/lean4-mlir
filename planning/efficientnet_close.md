# Planning — closing EfficientNet-B0 (ch8) both ways

The enet analogue of `planning/{mobilenetv2,resnet34}_close.md`. Goal: bring EfficientNet-B0 to the
"closed both ways" bar — **(a)** train-step text = name-threaded `pretty` of a *proven* `SHlo` forward
graph, **(b)** every param output certified `θ − lr·certified-gradient`, 3-axiom-clean.

**Headline: like r34, Item C is a FREE close** — even the two genuinely-new structures (squeeze-excite,
true batch-norm) add no new *param*-gradient bridge. The new work is concentrated in A/B/D, where the
**squeeze-excite** structure is the genuinely-new piece (a new SHlo channel-scale token + the SE-gate
backward).

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

### Item A — per-channel forward graph — [NEXT, real assembly + a new token]
`EfficientNetRenderPC.lean`: `efficientnetFwdGraph` (full 16-MBConv, batch-norm, SE) + `_faithful`. Needs
a **new SHlo channel-scale token** (the SE gate `[BS,c]` broadcast-multiplied with `[BS,c,h,w]`) — or
compose `gapF`/`denseF`/`sigmoidF`/`swishF` (all exist) plus the scale. Also a **batch-norm forward token**
(`bnBatchF` denoting `bnBatchTensor4`). The SE block VJP (`seBlockFull_has_vjp`) is already proven, so the
faithfulness is op-composition. This is the genuinely-new structural assembly.

### Item B — structured render — [given A; SE backward is the fiddly part]
`tests/TestEfficientNetTrainPC.lean`: forward + backward via `pretty`. swish/sigmoid backward tokens
exist (`swishBack`/`sigmoidBack`, smooth — no kink). The **SE-gate backward** (sigmoid VJP + channel-scale
back + the two dense backs + the squeeze/broadcast back) is the new piece. Validate with
`scripts/render_parity.py --fn efficientnet_train_step` (the harness is ready).

### Item D — cotangent chain — [optional; SE gate + batch-norm + swish]
The MBConv cotangent chain analogue of `MobileNetV2ChainClose`, crossing the SE gate
(`seBlockFull_has_vjp`), batch-norm (`bnBatchTensor4_grad_input`), and swish (smooth). `mbconvResidual_has_vjp`
is the per-block head start.

---

## 3. Order & status
1. **Item C** ✅ DONE — free close, `EfficientNetClose.lean`.
2. **Item A** — per-channel forward graph (needs a channel-scale + batch-norm SHlo token).
3. **Item B** — structured render + `iree-run-module` parity (harness ✅ ready).
4. **Item D** — optional SE/batch-norm/swish cotangent chain.

After Item B, EfficientNet is closed both ways. The new work vs MobileNetV2/r34 is **squeeze-excite**
(a new token + the SE-gate backward); everything else reuses the existing machinery.
