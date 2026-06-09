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

### Item A — typed forward graph `mobilenetv2FwdGraph : SHlo nClasses`  [NEW, real assembly]
Assemble the whole inverted-residual net from the faithful tokens (per-channel BN, to match the
render), with a `_faithful` theorem to the per-channel ℝ-forward — the analogue of
`cifarBnFwdGraph`/`cifarBnFwdGraph_faithful` but with depthwise, residual `addV`, stride-2, and the
block structure. **This is the prerequisite for a structured render.** Sub-tasks: reconcile the
ℝ-forward to per-channel BN; thread the residual skip (the `addV` reuses the block-input subtree in
both operands, so the graph stays a tree, cf. the `addV` doc comment); carry the stride-2 shape
changes (`flatConvStridedF`/`depthwiseStridedF` halve spatial).

### Item B — structured render `mobilenetv2TrainStepStructured`  [NEW, mechanical given A]
Pretty the forward graph (Item A) capturing names; the backward/grad/SGD **tail already exists**
as the hand-written `TestMobilenetV2Train.lean` text (depthwise-back regular+strided, relu6Back
two-sided kink, per-channel BN back + dγ/dβ, residual fan-in, conv W/b grads). Same **(a′) recipe**
as CIFAR-BN: flat forward + flat→NCHW `reshape` glue for the 4-D conv/depthwise consumers; wire the
tail to captured names. Validate by swap-train `mobilenetv2-verified` (expect parity with the
committed renderer; per-channel BN back via `bnPerChannelBack` token is *equivalent, not
bit-identical* if it recomputes like CIFAR-BN did).

### Item C — the close `MobileNetV2Close.lean`  [NEW wrappers, mostly cheap — DO THIS FIRST]
Instantiate the reusable bridges at each param, generic in the cotangent (the CIFAR-non-BN-style
"free close" — needs no chain). Per block: expand/project/head 1×1 → `cnn_render_conv{W,b}_certified`;
**depthwise → new `mnv2_render_depthwise{W,b}_certified`** wrapping `depthwise_weight_grad_has_vjp3
.correct` / `depthwise_bias_grad_has_vjp.correct` (the one genuinely-new bridge family — but it's
instantiation, the VJP is proven); BN γ/β → `cifar_bn_render_{gamma,beta}_certified`; stem strided
conv → the ch6 strided conv weight-grad. relu6 has no params. `#print axioms` each in AuditAxioms.
**This is the highest-value/effort piece and doesn't depend on A/B** — the operational render
already GPU-trains, so the close certifies *that* computation.

### Item D — the inverted-residual cotangent chain  [NEW, the hard/optional polish]
The analogue of `CnnChainClose.lean`, but harder: pin the generic `c` to the actual chain through
6 blocks, crossing **(i)** residual fan-in (`addV` backward: the cotangent at a block output flows
to BOTH the skip and the body, summed at the block input), **(ii)** stride-2 downsample boundaries
(the strided backward zero-upsamples; cotangent spatial size changes), **(iii)** the relu6
two-sided kink (`selectMid`), **(iv)** per-channel BN back, **(v)** depthwise + pointwise convs.
This is the genuinely-new, fiddliest piece (a `Back`/`Back3`-style chain over a branchy, multi-scale
net). *Optional* — pins the cotangent; the further "= ∂loss/∂θ" fold remains separate (as for CNN).

---

## 3. Suggested order (value-first, like the original handoff)

1. **Item C (the close), generic-cotangent.** Cheapest, highest value, no prerequisites. Gives
   "every MobileNetV2 conv/depthwise/BN param output denotes `θ − lr·(certified Jacobian · the
   layer cotangent)`." The depthwise W/b wrappers are the only new bridge family (instantiation).
   Land + audit. This alone is a real result: the operational GPU-trained render is now
   param-certified.
2. **Item A (typed forward graph)** + its `_faithful`. The prerequisite for the structured render;
   also reconciles the scalar↔per-channel BN.
3. **Item B (structured render)** from A. Swap-train validate. Now MobileNetV2 has the "text =
   render of proven graph" half too.
4. **Item D (cotangent chain)** — optional polish; the branchy inverted-residual analogue of
   `CnnChainClose`.

After step 1, MobileNetV2 is close-DONE (the CIFAR-non-BN bar). After step 3, it's closed both ways
(the CIFAR-BN bar). Step 4 matches the conv-close upgrade. Validation recipe + audit gate: identical
to `planning/render_close_handoff.md` §"Validation recipe".

## 4. Honest scope note
The Explore-style "no new proofs, ~200 LOC" estimate is too rosy: it ignores that there is **no
typed forward graph and no structured render** (Items A+B are real assembly), and that the
**cotangent chain (Item D) is branchier than anything done so far** (residual fan-in + multi-scale
stride-2). What *is* cheap is Item C (the param closes), because every per-op VJP and param-grad
bridge is already proven and 3-axiom-clean — that's the genuine head start.
