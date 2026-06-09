# Planning — closing MobileNetV2 (ch7) both ways

The next backbone after CIFAR-BN. Goal: bring MobileNetV2 to the same "closed both ways" bar the
linear→CIFAR-BN ladder reached (see `planning/render_close_handoff.md`) — **(a)** the train-step
text rendered as a name-threaded `pretty` of a *proven* `SHlo` forward graph, and **(b)** every
parameter output certified `θ − lr·certified-gradient`, audited 3-axiom-clean.

This is a **bigger lift than CIFAR was** — not "just instantiate." The op-level proofs and the
param-grad VJPs all exist and are reusable, but three genuinely-new pieces are missing: a typed
forward graph, a structured render, and the branchy inverted-residual cotangent chain. Read the
inventory before estimating.

---

## 0. Architecture (what we're closing)

`mobilenetv2Forward` (`LeanMlir/Proofs/MobileNetV2.lean:461`): stem 3×3-s2 conv→BN→relu6 → **6
inverted-residual blocks** (4 stride-2 downsampling, 224→112→56→28→14→7) → head 1×1 conv→BN→relu6
→ GAP → dense. **Inverted-residual block** = `expand(1×1)→BN→relu6 → depthwise(3×3)→BN→relu6 →
project(1×1)→BN` (linear bottleneck, *no* relu6 after project), with a **residual skip** when
stride=1 ∧ ic=oc (`invresSkip`, `invresBody`, `MobileNetV2.lean:312-424`). ~82 parameters.

The `mobilenetv2-verified` exe GPU-trains this on Imagenette 224² from the hand-written renderer
(`tests/TestMobilenetV2Train.lean` → `verified_mlir/mobilenetv2_train_step.mlir`). Spec:
`VerifiedNets.lean:145` (slug `mobilenetv2`).

---

## 1. Inventory — what's already proven (all 3-axiom clean, REUSABLE)

**Op tokens + faithfulness** (`StableHLO.lean`): every op MobileNetV2 needs is a faithful `SHlo`
token already:
- `flatConvF` (1×1 pointwise + stem 3×3) `flatConvF_faithful`; `flatConvStridedF` (stem s2)
- `depthwiseF` / `depthwiseStridedF` (`depthwiseF_faithful` / `depthwiseStridedF_faithful`), and
  their input-VJP backwards `depthwiseBack` / `depthwiseStridedBack` (proven faithful)
- `relu6F` / `selectMid` (the two-sided-kink backward, `selectMid_faithful` = `relu6_has_vjp_at`)
- `bnPerChannelF` / `bnPerChannelBack` (`bnPerChannelF_faithful`, `bnPerChannelBack_faithful` `0<ε`)
- `addV` (residual add, `addV_faithful`), `gapF` (`gapF_faithful`), `denseF`

**Param-grad VJPs** (the close ingredients, all with `.correct`):
- conv W/b → `conv_weight_grad_bridge` / `conv_bias_grad_bridge` (`CnnTrainStep.lean:32`) — reuse
  for expand/project/head 1×1 and (strided) stem
- **depthwise W/b → `depthwise_weight_grad_has_vjp3` / `depthwise_bias_grad_has_vjp`**
  (`Depthwise.lean:756`,`:1128`) — `.correct` exposes the pdiv contraction. **REUSE.**
- per-channel BN γ/β → `bnPerChannel_grad_{gamma,beta}_correct` (`CifarBnClose.lean:136`) — reuse
- relu6 / residual-add / GAP have no learnable params

**ℝ whole-net VJP**: `mobilenetv2_has_vjp_at` (+`_correct`) at a smooth point, and the concrete
witness `mnv2Concrete_has_vjp_correct` — but the concrete instance is **degenerate** (zero kernels
force every relu6 input to β=1∈(0,6) ⇒ constant activations ⇒ zero-Jacobian; `MobileNetV2.lean:897`).
A live parameterized witness needs genuine per-coordinate `(0,6)` bounds. *Not on the close
critical path*, but worth noting when claiming "verified."

## 1a. Two facts the close MUST account for (don't skip)

1. **BN flavor mismatch.** The **render emits per-channel BN** (reduce over spatial `[2,3]` →
   `[BS,oc]`, `TestMobilenetV2Train.lean:73`), but the **ℝ-proof `mobilenetv2_has_vjp_at` uses
   *scalar* `bnForward`** (one γ/β over the whole `oc·h·w`, `MobileNetV2.lean:215`). For the close,
   use the **per-channel** bridges (`bnPerChannelF` faithful + `CifarBnClose`) — they match the
   render. The scalar ℝ-VJP is a separate/older artifact. (Same scalar↔per-channel reconciliation
   that CIFAR-BN needed.)
2. **There is NO typed `SHlo` forward graph** (`grep ': SHlo' MobileNetV2.lean` is empty) and the
   train-step "render" is a **hand-written String** renderer (`TestMobilenetV2Train.lean`, the
   `cnnTrainStepText` stage), *not* a name-threaded `pretty(graph)` structured render. So the
   "text = render of proven graph" half does not exist yet.

---

## 2. The genuinely-new work (what closing actually requires)

### Item A — typed forward graph `mobilenetv2FwdGraph : SHlo nClasses`  ✅ **DONE** (2026-06-09)
Assemble the whole inverted-residual net from the faithful tokens (per-channel BN, to match the
render), with a `_faithful` theorem to the per-channel ℝ-forward — the analogue of
`cifarBnFwdGraph`/`cifarBnFwdGraph_faithful` but with depthwise, residual `addV`, stride-2, and the
block structure. **This is the prerequisite for a structured render.**

**Shipped** (`LeanMlir/Proofs/MobileNetV2RenderPC.lean`, 3-axiom clean, audited):
- Found that `StableHLO.lean` *already* had the full strided graph `mobilenetv2FwdGraphFull` +
  `mobilenetv2FwdGraphFull_faithful` — but using **scalar** `bnF`, tied to the scalar
  `mobilenetv2Forward_full`. That's a *different function* than the render (which emits per-channel
  BN), so it was not a faithful "render of a proven graph". The gap was purely the BN flavor.
- Built the **per-channel** twin: stage abbreviations `ivExpandPC`/`ivDepthwisePC`/
  `ivDepthwiseStridedPC`/`ivProjectPC` (per-channel `bnPerChannelTensor3`, `γ/β : Vec c`) →
  `mobilenetv2Forward_full_pc` (full ℝ-forward, same topology/stride/relu6 schedule) →
  `mobilenetv2FwdGraphFullPC` (typed `SHlo`, `bnPerChannelF` tokens, strided stem +
  `depthwiseStridedF` on the 4 downsampling blocks + `addV` skip on b2/b4 + conv-bn-relu6 head) →
  **`mobilenetv2FwdGraphFullPC_faithful`**: `den (graph) = mobilenetv2Forward_full_pc`, via
  `bnPerChannelF_faithful`, same `simp`-then-`unfold` recipe as the scalar peer (compiled first try).
- The "reconcile scalar↔per-channel BN" sub-task was handled by building per-channel *throughout*
  (no scalar ℝ-VJP in the loop), so there is no mismatch to reconcile away.

Net: MobileNetV2 now has a **proven per-channel forward graph** whose `pretty` will match the
render's forward text — the input Item B needs.

### Item B — structured render `mobilenetv2TrainStepStructured`  ✅ **DONE** (2026-06-09)
Pretty the forward graph (Item A) capturing names; the backward/grad/SGD **tail already exists**
as the hand-written `TestMobilenetV2Train.lean` text. Same **(a′) recipe** as CIFAR-BN.

**Shipped** (`tests/TestMobilenetV2TrainPC.lean`, `#eval`-rendered, iree-compiles, GPU-validated):
- Realized the recipe is *stronger* than planned: not just the forward, but the **entire backward
  cotangent chain** is proof-rendered through `pretty` — `dotOut` (dense), `bnPerChannelBack`,
  `selectMid` (relu6 two-sided kink), `convBack`, `depthwiseBack`/`depthwiseStridedBack`, and `addV`
  (residual fan-in) are all flat-in/flat-out SHlo tokens. Only the pieces with **no SHlo constructor**
  are hand-emitted: GAP backward, conv/depthwise weight+bias grads (transpose trick / reduce), and the
  per-channel BN dγ/dβ (recompute x̂). Reshape glue (flat→NCHW) only at the hand weight/bias/BN-param
  grads. So the structured render is *more* proof-rendered than the hand-written one.
- Same 82-param func signature/order as the committed renderer (verified `diff`-identical) → drop-in.
- iree-compiles OK (213 KB MLIR, gfx1100).
- **Validation: bit-identical, not just epoch-parity.** The swap-train harness (`mobilenetv2-verified`)
  is blocked here by a *pre-existing* imagenette data-loader bug ("short read") that hits the committed
  renderer equally — so instead ran **both** the committed and structured vmfbs through `iree-run-module`
  with identical fixed-random inputs and compared all 82 output param-updates: **worst rel-diff 0.0,
  `np.array_equal` True on every param** (and every output differs from its input — a real SGD step,
  gradients flowed end-to-end). The per-channel-BN recompute turned out bit-identical, not merely
  equivalent. This is a stronger parity check than the planned swap-train epoch check and bypasses the
  data bug entirely.

### Item C — the close `MobileNetV2Close.lean`  ✅ **DONE** (2026-06-09)
Instantiate the reusable bridges at each param, generic in the cotangent (the CIFAR-non-BN-style
"free close" — needs no chain). Per block: expand/project/head 1×1 → `cnn_render_conv{W,b}_certified`;
**depthwise → new `mnv2_render_depthwise{W,b}_certified`** wrapping `depthwise_weight_grad_has_vjp3
.correct` / `depthwise_bias_grad_has_vjp.correct` (the one genuinely-new bridge family — but it's
instantiation, the VJP is proven); BN γ/β → `cifar_bn_render_{gamma,beta}_certified`; stem strided
conv → the ch6 strided conv weight-grad. relu6 has no params. `#print axioms` each in AuditAxioms.
**This is the highest-value/effort piece and doesn't depend on A/B** — the operational render
already GPU-trains, so the close certifies *that* computation.

**Shipped** (`LeanMlir/Proofs/MobileNetV2Close.lean`, 8 certified theorems, all 3-axiom clean
`[propext, Classical.choice, Quot.sound]`, audited in `tests/AuditAxioms.lean`):
- `mnv2_render_depthwise{W,b}_certified` — stride-1 depthwise (blocks b2,b4), the plan's headline
  family. Thin `.correct` wrappers of `depthwise_weight_grad_has_vjp3` / `depthwise_bias_grad_has_vjp`.
- `mnv2_render_stem_conv{W,b}_certified` — stem 3×3 stride-2. W reuses `flatConvStride2_weight_grad_has_vjp`;
  b needed a **new** `flatConvStride2_bias_grad_has_vjp` (the plan only listed the stem *weight*).
- `mnv2_render_depthwise{W,b}_strided_certified` — **the gap the plan missed**: 4 of 6 blocks
  (b1,b3,b5,b6) downsample with a *strided* depthwise, so their dW/db feed `depthwiseStride2Flat`,
  not `depthwiseConv2d`. Built new `depthwiseStride2_{weight,bias}_grad_has_vjp` = `vjp_comp` of a
  proven stride-1 depthwise VJP with `decimateFlat_has_vjp` (the same `decimate ∘ stride-1` recipe
  as the ch6 strided conv weight-grad; backward = upsample-then-stride-1, matching the render's
  `dwconvWGradStrided`). +3 helper differentiability lemmas (`depthwise_weight/bias_differentiable`,
  `conv2d_bias_differentiable`).
- **Reuse (no new theorem):** 1×1 conv W/b → `cnn_render_conv{W,b}_certified`; BN γ/β →
  `cifar_bn_render_{gamma,beta}_certified`; dense Wd/bd → M2 `weight/bias_grad_bridge`. These are
  fully dim-polymorphic, so they apply at the MobileNetV2 shapes verbatim (documented in the file's
  coverage table + the AuditAxioms note, as CIFAR-BN did).

Net: **every** MobileNetV2 train-step parameter family is now certified `θ − lr·(certified
Jacobian · the layer cotangent)`. MobileNetV2 is at the CIFAR-non-BN "close-DONE" bar.

### Item D — the inverted-residual cotangent chain  ✅ **DONE** (2026-06-09)
`LeanMlir/Proofs/MobileNetV2ChainClose.lean` (9 theorems, 3-axiom clean, audited) — the analogue of
`CnnChainClose`/`ResNet34ChainClose`, pinning the Item C bridges' generic `c` to the cotangent the
inverted-residual backward chain delivers. The chain through a block composes the *rendered* backward
denotations — relu6 two-sided-kink mask (`selectMid`, `if 0<x<6`), per-channel BN input-VJP
(`bnPerChannelTensor3_grad_input`), 1×1 conv input-VJP (`conv2d_has_vjp3` via the flatten bridge),
depthwise input-VJP (`depthwiseFlat_has_vjp` / `depthwiseStride2Flat_has_vjp`) — `project→depthwise→
expand` into `invresCotPc` / `invresCotDc` / `invresCotEcS1` / `invresCotEcS2`:
- **project** (1×1): `invres_render_proj{W,b}_chain_certified` at `invresCotPc`. The **linear bottleneck**
  (no relu6 after the residual `addV`) makes the project-BN output cotangent `dyOut` *directly* — skip
  and no-skip alike, unlike r34's `relu(add(…))`. ((i) the residual fan-in is therefore trivial here.)
- **depthwise**: `invres_render_dw{W,b}_{s1,s2}_chain_certified` at `invresCotDc` — stride-1 (skip
  blocks, `mnv2_render_depthwise*`) and stride-2 (downsampling, the strided depthwise bridges). ((ii)+(v))
- **expand** (1×1): `invres_render_expW_{s1,s2}_chain_certified` at `invresCotEcS1`/`S2`. The stride-2
  blocks carry the expand-side cotangent at 2h×2w (the `_s2` split — (ii) the spatial size changes). ((iii)+(iv))
- **stem** (3×3 strided): `mnv2_stem_render_convW_chain_certified` at `mnv2StemCot` (relu6 mask + BN-back;
  no maxpool, simpler than r34's stem).

Each conv/depthwise θ output denotes `θ − lr·(certified ∂/∂θ · the-actual-chain-cotangent)`. Pins the
cotangent; the `= ∂loss/∂θ` fold stays separate (as for the CNN). The flatten/unflatten + stride-split +
relu6-kink juggling — the "fiddliest piece" — all compiled first try, reusing the `ResNet34ChainClose`
template.

---

## 3. Suggested order (value-first, like the original handoff)

1. **Item C (the close), generic-cotangent.** ✅ **DONE** — `MobileNetV2Close.lean`, 8 theorems,
   3-axiom clean, audited. "Every MobileNetV2 conv/depthwise/BN param output denotes `θ − lr·(certified
   Jacobian · the layer cotangent)`." The operational GPU-trained render is now param-certified.
   Scope note: the depthwise W/b wrappers were *not* the only new content — the 4 downsampling
   blocks' **strided** depthwise W/b + the stem strided **bias** each needed a new `decimate ∘
   stride-1` VJP (bounded analogues of the existing ch6 strided conv weight-grad, not just
   instantiation). See Item C above.
2. **Item A (typed forward graph)** + its `_faithful`. ✅ **DONE** — `MobileNetV2RenderPC.lean`
   (`mobilenetv2FwdGraphFullPC` + `_faithful`, per-channel BN matching the render). The scalar full
   graph already existed in `StableHLO.lean`; the new work was the per-channel twin.
3. **Item B (structured render)** from A. ✅ **DONE** — `tests/TestMobilenetV2TrainPC.lean`,
   forward + full backward cotangent chain proof-rendered, iree-compiles, **bit-identical** GPU
   outputs vs the committed renderer (82/82 params, `np.array_equal`). MobileNetV2 now has the
   "text = render of proven graph" half.
4. **Item D (cotangent chain)** ✅ **DONE** — `MobileNetV2ChainClose.lean` (9 theorems, 3-axiom clean).
   The branchy inverted-residual analogue of `CnnChainClose`/`ResNet34ChainClose`.

After step 1, MobileNetV2 is close-DONE (the CIFAR-non-BN bar). After step 3, closed both ways (the
CIFAR-BN bar). **After step 4 (now), MobileNetV2 is FULLY closed (C+A+B+D)** — matching ResNet-34.
Step 4 matches the conv-close upgrade. Validation recipe + audit gate: identical to
`planning/render_close_handoff.md` §"Validation recipe" (here the swap-train was replaced by an
`iree-run-module` same-inputs output diff via `scripts/render_parity.py`, since the imagenette loader
is broken in this env). The `= ∂loss/∂θ` fold (composing the per-block cotangents through all blocks to
the loss) is the one remaining program-wide open item, shared with the CNN — not MobileNetV2-specific.

## 4. Honest scope note
The Explore-style "no new proofs, ~200 LOC" estimate is too rosy: it ignores that there is **no
typed forward graph and no structured render** (Items A+B are real assembly), and that the
**cotangent chain (Item D) is branchier than anything done so far** (residual fan-in + multi-scale
stride-2). Item C (the param closes) was the cheap head start — but *not* "no new proofs":
the stride-1 depthwise + reuse families were instantiation, but the **4 downsampling blocks'
strided depthwise W/b and the stem strided bias each needed a new `decimate ∘ stride-1` VJP**
(`depthwiseStride2_{weight,bias}_grad_has_vjp`, `flatConvStride2_bias_grad_has_vjp` + 3 diff
lemmas). These are bounded analogues of the proven ch6 strided conv weight-grad, so they went in
quickly (~190 LOC total, all 3-axiom clean), but they *are* new proofs the original Item C list
missed. The genuine head start: every per-op VJP and stride-1 param-grad bridge was already proven.
