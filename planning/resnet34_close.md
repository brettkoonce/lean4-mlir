# Planning — closing ResNet-34 (ch6) both ways

The r34 analogue of `planning/mobilenetv2_close.md`. Goal: bring ResNet-34 to the "closed both ways"
bar — **(a)** the train-step text rendered as a name-threaded `pretty` of a *proven* per-channel-BN
`SHlo` forward graph, and **(b)** every parameter output certified `θ − lr·certified-gradient`,
3-axiom-clean.

**Headline: r34 is a *cleaner* close than MobileNetV2 was.** The thing that made MobileNetV2's close
real work — the depthwise bridge family — has no r34 counterpart; and the strided-conv W/b bridges
built for MobileNetV2's stem (`MobileNetV2Close.lean`) are exactly what r34's stem/downsample/projection
need. r34 uses only regular convs (3×3 + 7×7 stem), per-channel BN, plain relu, maxpool, residual add,
GAP, dense.

---

## 0. Architecture (`MainResnet34Verified.lean`, 146 params)
conv(3→64, 7×7-s2, pad 3) → BN → relu → maxpool(112→56) → 4 stages `[3,4,6,3]` basic blocks
(stages 2–4 open with a stride-2 **downsample** block + a 3×3 stride-2 **projection** skip),
channels 64/128/256/512, spatial 56→28→14→7 → GAP÷49 → dense 512→10 + softmax-CE. Hand render:
`tests/TestResnet34Train.lean`; trainer `resnet34-verified`.

## 1. Starting state (≈ where MobileNetV2 began)
- `ResNet34.lean` (885 ln): op VJPs all proven (`convBnStrided_has_vjp`, `convBnReluStrided_has_vjp_at`,
  `resblock_bodyStrided_has_vjp_at`, `rblkPStrided_has_vjp_at`) + the **depth-folded whole-net chain**
  `resnet34_has_vjp_at` (parametric over a `List.length`). Concrete witness is *degenerate* (1-channel).
- `resnetFwdGraph` (StableHLO.lean): a **representative** (stem + 1 id + 1 proj block + GAP + dense),
  **scalar `bnF`** — not full-depth, not per-channel. Same gap MobileNetV2 had.
- No Item C/A/B yet.

---

## 2. The four rungs

### Item C — the param closes — ✅ **DONE** (2026-06-09), a FREE close
`LeanMlir/Proofs/ResNet34Close.lean` (6 theorems, 3-axiom clean, audited). **No new VJP** — every r34
param family maps to an existing bridge:
- 3×3 s1 conv W/b (id `W1`/`W2`, down `W2`) → `cnn_render_conv{W,b}_certified` (M3).
- 3×3 s2 conv W/b (down `W1`, projection `Wp`) → `mnv2_render_stem_conv{W,b}_certified`.
- **7×7 s2 stem** W/b → `mnv2_render_stem_conv{W,b}_certified` at `kH=kW=7` (the only 7×7 in any net).
- per-channel BN γ/β → `cifar_bn_render_{gamma,beta}_certified`; dense → M2 `weight/bias_grad_bridge`.
- maxpool/relu/residual/GAP → no params.

The six `r34_render_{stem,block,down}Conv{W,b}_certified` theorems pin the generic strided/regular conv
bridges to r34's kernels — the audit-meaningful step (confirms the 7×7 stem and 3×3 strided projection,
the shapes no prior net exercised through these bridges, really are covered). BN/dense are verbatim
reuse (no kernel to pin). This was *cheaper* than MobileNetV2's Item C.

### Item A — per-channel forward graph — ✅ **DONE** (2026-06-09)
`LeanMlir/Proofs/ResNet34RenderPC.lean` (3-axiom clean, audited). Per-channel building blocks
`cbrStridedPC` (7×7 strided stem conv-bn-relu), `rblkPC` (identity block `relu(F(x)+x)`),
`rblkPStridedPC` (downsample block `relu(F_s(x)+proj_s(x))`, 3×3 strided projection) — per-channel
`bnPerChannelTensor3` mirrors of `cbr`/`rblk`/`rblkP`. **No new tokens.** Structured as:
- **per-block typed graphs + faithfulness** `idBlockGraphPC_faithful` / `downBlockGraphPC_faithful`
  (each block's token tree denotes its per-channel forward; the residual `addV` reuses the block-input
  subtree, tree-safe) — the general, reusable core, ~12 params each.
- **whole-net** `resnet34FwdGraphFullPC` + `resnet34Forward_full_pc` + `resnet34FwdGraphFullPC_faithful`
  at the render dims (3×224² → 7×7×512, 146 params, shared ε): 7×7 strided stem → maxpool → `[3,4,6,3]`
  blocks → GAP → dense. The whole-net faithfulness is a single `simp only` chaining the per-block
  lemmas + stem/maxpool/GAP/dense `_faithful` (compiled in ~2 s). Same recipe as
  `mobilenetv2FwdGraphFullPC_faithful`, but the per-block-lemma factoring keeps the 16-block proof one line.

### Item B — structured render — ✅ **DONE** (2026-06-09)
`tests/TestResnet34TrainPC.lean`: forward + backward cotangent chain proof-rendered via `pretty` —
forward (`flatConvStridedF`/`flatConvF`/`bnPerChannelF`/`reluF`/`maxPoolF`/`addV`/`gapF`/`denseF`),
backward (`dotOut`, `bnPerChannelBack`, **`selectPos`** (single-sided relu mask), `convBack`/
`convStridedBack`, `addV` residual fan-in). Hand-emit only: GAP back, conv/strided-conv weight+bias
grads, the 7×7 stem weight-grad, BN dγ/dβ — **and the maxpool backward** (the `maxPoolBack` token's
`emitTok` hardcodes region args `%sa/%sb/%sc/%sd`, which collide with the stem-bias param `%sb` and
the `%sc` init constant; hand-emitted with `%q`-prefixed names, as the committed renderer does).
**r34-specific wrinkle vs MobileNetV2:** the residual add is *inside* the block-output relu
(`relu(add(main, skip))`), so the skip cotangent is the post-relu-back value (`cot_a`), not `dy` —
matching the committed `addOp {p}dx %{p}dc1 %{p}da`.

iree-compiles OK (388 KB MLIR, gfx1100); same 146-param signature as the committed render (`diff`-
identical). **Validated bit-identical** via `scripts/render_parity.py`: 146/146 outputs `np.array_equal`,
worst rel-diff 0.0 (each a real SGD step — outputs differ from inputs). r34 is now **closed both ways**.

**Validation harness: ✅ SET UP & de-risked** (the hard part last time). `scripts/render_parity.py`
(reusable) parses the func sig, gens fixed-random `.npy` inputs, iree-compiles + GPU-runs both the
committed and structured `.mlir` via `iree-run-module`, and `np.array_equal`s all outputs. Proven on
the committed r34 baseline (`--fn resnet34_train_step --ref verified_mlir/resnet34_train_step.mlir`:
148 inputs/146 outputs, all finite; committed-vs-itself = 146/146 bit-identical) and on MobileNetV2
(82/82). So Item B validation is now one command — bypasses the broken imagenette swap-train loader.

### Item D — cotangent chain — [optional; head start]
r34 already has `resnet34_has_vjp_at` (the chain assembled parametrically), so pinning the Item C
bridges' generic cotangent to it is less from-scratch than MobileNetV2's would be.

---

## 3. Order & status
1. **Item C** ✅ DONE — free close, `ResNet34Close.lean`.
2. **Item A** ✅ DONE — per-channel forward graph, `ResNet34RenderPC.lean` (per-block + whole-net faithful).
3. **Item B** ✅ DONE — structured render `TestResnet34TrainPC.lean`, bit-identical to committed (146/146).
4. **Item D** — optional cotangent-chain polish (not started; `resnet34_has_vjp_at` gives a head start).

**r34 is closed both ways (CIFAR-BN bar): every param certified (C) + per-channel forward graph
proven faithful (A) + structured render = proven graph, GPU-validated bit-identical (B).** Only the
optional Item D (cotangent-chain pinning) remains.

After Item B, r34 is closed both ways (the CIFAR-BN bar). The new work is concentrated in A/B assembly,
mechanical given the MobileNetV2 templates (`MobileNetV2RenderPC.lean`, `tests/TestMobilenetV2TrainPC.lean`).
