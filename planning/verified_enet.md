# Verified EfficientNet (Chapter 8) — handoff

Continuation doc for the **EfficientNet** chapter of the verified-codegen ladder. ch8
brings EfficientNet (MBConv = inverted-residual + **squeeze-excite** + **swish**) into
the same "render via the proof-carrying StableHLO emitter + GPU-train on the rendered
text" line that ch6 (ResNet-34) and ch7 (MobileNetV2) closed.

> **Status in one line:** the whole-network VJP of EfficientNet —
> **`Proofs.efficientnet_has_vjp`** (`_correct`) — is **PROVEN, audit-clean, and
> UNCONDITIONAL** (all-smooth: swish + sigmoid SE gate + conv/BN, NO kinks, holds at
> *every* input under only `0 < ε`; even cleaner than MNv2, which had relu6 kinks).
> Also `efficientnet_has_vjp_at(_correct)` (smooth-point form). The SE/swish/sigmoid/
> MBConv VJPs are all done: `seBlock_has_vjp_correct`, `swish_has_vjp_correct`,
> `sigmoid_has_vjp`, `seGate_has_vjp`, `broadcastFlat_has_vjp`, `mbconvBody_has_vjp`,
> `mbconvResidual_has_vjp(_at)`. **The MATH is entirely in hand.** What remains for a
> *trained* model is the **verified codegen** — the `mobilenetv2-verified` treatment:
> a swish op + a sigmoid op + the **squeeze-excite module renderer** + the MBConv
> renderer/layout/Main + GPU train. **ch7 leaves ~85% reusable** (depthwise stride-1/
> stride-2, per-channel BN, 1×1 convs, GAP, dense, residual, the data-driven `[t,c,n,s]`
> strided-block harness); the only genuinely-new emitter work is swish, sigmoid, and the
> SE gate (the one hard combinator: a fan-out sub-network multiplied back into the main
> path). Read §§0–3 first; rest is reference.

Written 2026-06-07 by the session that finished ch7 (MobileNetV2, `mobilenetv2-verified`
GPU-trains the real downsampling net 10%→57.1%). `origin/main` is behind by ch6 + ch7 +
this doc; commit-to-main is fine, **never push without explicit per-push permission**.

---

## 0. The pattern (unchanged from every chapter)

Typed, proof-carrying StableHLO emitter. Per op-graph:
`SHlo a ─skel→ Raw ─toToks→ [Tok] ─emitTok→ StableHLO text`, two halves:
- **Semantic R4:** `den (graph) = <proven Mathlib fderiv quantity>` (faithfulness).
- **Syntactic R4:** `parse (toToks (skel a)) = skel a` (round-trip, `roundtrip`).

Deliverables per chapter: whole-net forward graph + faithfulness, full SGD train-step
renderer, a `*-verified` exe that GPU-trains on the rendered text, audit ℝ-headline(s)
3-axiom-clean.

Ladder: ch2 92.10%, ch3 97.86%, ch4 98.89%, ch5-A 66%, ch5-B 57%, ch6-A 60.8%,
**ch6-B (real ResNet-34) 62%+**, **ch7 (MobileNetV2, real downsampling) 57.1%** (verified
architecture AND GPU-trained). **ch8 = EfficientNet** (architecture verified + unconditional;
codegen + train = this doc).

---

## 1. ⭐ WHAT IS DONE — the EfficientNet proofs (audit-clean) + ch7 reuse

### The EfficientNet VJP stack (committed; in the audited headlines)
In **`LeanMlir/Proofs/EfficientNet.lean`** (imports `Depthwise`, `SE`, `LayerNorm`),
**`LeanMlir/Proofs/SE.lean`**, and **`LeanMlir/Proofs/LayerNorm.lean`** (swish lives here).

| Piece | Theorem / def | Notes |
|---|---|---|
| swish | `swish` (= `x·σ(x)`), `swish_has_vjp(_correct)`, `swishScalarDeriv` | **SMOOTH everywhere** (no kink). backward `dy·swish'(x)`, closed form `σ(x)·(1 + x·(1−σ(x)))`. |
| sigmoid | `sigmoid` (= `σ`), `sigmoid_has_vjp(_correct)`, `pdiv_sigmoid` | the SE gate's output nonlinearity. backward `dy·σ(x)(1−σ(x))`. |
| broadcast (channel→spatial) | `broadcastFlat`, `broadcastFlat_has_vjp` | `Vec c → Vec (c·h·w)`; backward = **sum each channel's spatial cotangents** (adjoint of broadcast). |
| SE gate | `seGate` (= `broadcast ∘ sigmoid ∘ dense₂ ∘ swish ∘ dense₁ ∘ GAP`), `seGate_has_vjp` | the squeeze-excite sub-network (`c·h·w → c·h·w`), chained via `vjp_comp`. |
| SE block (main × gate) | `seBlock gate x = x ⊙ gate(x)` (`SE.lean`), `seBlock_has_vjp(_correct)` = `elemwiseProduct_has_vjp (id) gate` | the fan-out: `back = gate(x)⊙dy + gate.back(x, x⊙dy)`. |
| MBConv body | `mbconvBody` (= `project(1×1 conv-bn) ∘ SE ∘ depthwise(bn-swish) ∘ expand(1×1 conv-bn-swish)`), `mbconvBody_has_vjp` | the EfficientNet block. |
| MBConv residual | `mbconvResidual_has_vjp(_at)` | stride-1, `cin=cout` block in an identity residual. |
| **WHOLE NET** | `efficientnetForward`, **`efficientnet_has_vjp(_correct)`** (UNCONDITIONAL) + `efficientnet_has_vjp_at(_correct)` | `dense ∘ GAP ∘ MBConv₂ ∘ residual(MBConv₁) ∘ (swish∘BN∘conv stem)`. |

Audited headlines: `swish_has_vjp_correct`, `sigmoid_has_vjp`, `seBlock_has_vjp_correct`,
`efficientnet_has_vjp_at_correct`, `efficientnet_has_vjp`, `efficientnet_has_vjp_correct`
(see `tests/AuditAxioms.lean`; audit currently **156/156** 3-axiom-clean). The old IRPrint
codegen also has audited backward bridges `IR.swish_back_bridge` / `IR.sigmoid_back_bridge`
— **a reference for the exact swish/sigmoid backward StableHLO**.

**NB — the proven apex is a SMALL EfficientNet** (stem + 1 residual-MBConv + 1 plain
MBConv, each with a full SE gate), the analogue of ch7's small apex / ch6-A's 2-block
net. Going deeper / downsampling is the codegen-side scaling job (see §2), exactly as
ch7 went small-apex → deep `[t,c,n,s]` net.

### ⭐ What ch7 (MNv2) leaves REUSABLE (this is why ch8 is cheap)
- **Depthwise conv op pairs** — `depthwiseF`/`depthwiseBack` (C1, stride-1) AND
  `depthwiseStridedF`/`depthwiseStridedBack` (C3, stride-2 downsample). MBConv's depthwise
  is identical. **5×5 depthwise needs NO new op** — the op is `kH/kW`-parameterized with
  pad `(k−1)/2`, so EfficientNet's 5×5 stages just pass `kH=kW=5`.
- **Per-channel BatchNorm op pair** — `bnPerChannelF`/`bnPerChannelBack`. Reuse verbatim.
- **1×1 convs, GAP `gapF`, dense, residual `addV`** — all present.
- **The depthwise weight-grad trick** — `batch_group_count = c` (stride-1) and upsample-then-
  that (stride-2). Reuse the ch7 train fragments `dwconvWGrad`/`dwconvWGradStrided`.
- **The whole `mobilenetv2-verified` codegen harness** — `tests/TestMobilenetV2{Fwd,Train}.lean`
  (data-driven `(p,ic,mid,oc,s)` block-list renderer threading spatial dims fwd+reverse,
  with strided depthwise), `MobileNetV2Layout` (IreeRuntime.lean), `MainMobilenetV2Verified.lean`,
  the `mlpTrainStepV` FFI loop. ch8's renderer/layout/Main are a near-clone: **swish in place
  of relu6, the SE module inserted before project, MBConv = inverted-residual + SE.**

---

## 2. ⭐ WHAT REMAINS — the verified codegen (ch8 = render + train)

The proof half is done; everything below is **render/GPU** (no new Mathlib). Suggested
order, mirroring ch7's de-risk-on-iree-compile-first discipline:

### E1 — swish SHlo op pair  [small, all-smooth — easier than relu6]
A `swishF`/`swishBack` op pair, 9-site lockstep mirroring the relu6 pair (`relu6F`/`selectMid`,
`StableHLO.lean`) but SIMPLER (swish is differentiable everywhere — no `select` mask, no
smoothness hyp, no two-sided kink):
- **`swishF`** forward: `x · sigmoid(x)`. StableHLO: `σ = logistic(x)` (`stablehlo.logistic`),
  `o = multiply(x, σ)`. `den = swish n` (rfl).
- **`swishBack`** backward: `dy · swish'(x)`, closed form `σ(x)·(1 + x·(1−σ(x)))`. Saves the
  pre-activation `x` (`xName`). `den = (swish_has_vjp n).backward x (den e)`, faithful via
  `swish_has_vjp_correct`. **Reference the existing `IR.swish_back_bridge` StableHLO**
  (IRPrint.lean) for the exact op sequence (logistic + a few multiplies/subtracts).
- 9 sites + standalone `tests/TestSwish.lean` iree-compile (mirror `tests/TestRelu6.lean`).
  swishF/swishBack are rfl-faithful → out of the audit; `roundtrip` covers them.

### E2 — sigmoid SHlo op pair  [small]
`sigmoidF` forward `stablehlo.logistic(x)` = `σ(x)`; `sigmoidBack` backward `dy·σ(x)(1−σ(x))`
(saves `x`; `σ = logistic(x)`, `mul(dy, mul(σ, sub(1, σ)))`). `den = sigmoid n` / its VJP via
`sigmoid_has_vjp_correct`. Reference `IR.sigmoid_back_bridge`. rfl-faithful forward.

### E3 — the SQUEEZE-EXCITE module  [the one genuinely-new combinator]
SE is `seBlock gate x = x ⊙ broadcast(gate(squeeze(x)))`. Render it as a sub-graph and its
two-path backward (NO single new SHlo op needed if you compose existing ops — GAP, dense/1×1,
swish, sigmoid, a channel-broadcast, and an elementwise multiply — but you DO need a renderer
fragment + its backward, like ch7's per-block fragments). **⭐ GOLDMINE REFERENCE:** the OLD
IRPrint codegen already emits the FULL MBConv-with-SE — `IRPrint.lean` ~**1413–1614** has the
squeeze-excite gate forward ("squeeze-excite (genuine gate sub-network)") AND its backward
fan-in ("squeeze-excite back (fan-in)"), plus `swishF`/`swishB` (~1462) and `swishFwdM`/
`swishBackM` (~807). Mirror that StableHLO directly. Forward (on `[B,c,h,w]`):
1. **squeeze** = GAP over spatial → `[B,c]`.
2. **reduce** = `dot_general` (`[B,c]·[c,r]`) + bias → `[B,r]`  (the SE bottleneck, r = c/4 typ).
3. **swish** (E1) → `[B,r]`.
4. **expand** = `dot_general` (`[B,r]·[r,c]`) + bias → `[B,c]`.
5. **sigmoid** (E2) → `[B,c]`  (= the per-channel gate, save it).
6. **broadcast** gate `[B,c]` → `[B,c,h,w]` (`broadcast_in_dim dims=[0,1]`) and **multiply**
   with the main path `x` → SE output.

Backward (the elemwise-product VJP `back = gate⊙dy + gate.back(x, x⊙dy)`):
- **main path:** `gate ⊙ dy` (saved gate × incoming cotangent) — one `multiply`.
- **gate path:** cotangent `x ⊙ dy` →
  GAP-back into `[B,c]` (reduce-sum over spatial — the adjoint of the step-6 broadcast is
  sum-over-spatial; `broadcastFlat_has_vjp`) →
  sigmoid-back (⊙ σ(1−σ), E2) →
  expand `dot_general` back (`[B,c]·[r,c]ᵀ → [B,r]`) + its weight/bias grads →
  swish-back (⊙ swish′, E1) →
  reduce `dot_general` back (`[B,r]·[c,r]ᵀ → [B,c]`) + its weight/bias grads →
  GAP-back to spatial (`broadcast dy/(h·w)`, the squeeze adjoint) → `[B,c,h,w]`.
- **dx = main + gate-path**, summed (`stablehlo.add`). The two SE dense weight grads
  (`dWs₁`, `dWs₂`) + biases are extra params.
- De-risk a standalone `tests/TestSE.lean` (one SE module fwd+back) on `iree-compile` BEFORE
  wiring into MBConv. **Watch:** the saved gate value is reused in BOTH the main-path scale
  and (implicitly) the sigmoid-back; save it once.

### E4 — the MBConv renderer + layout + Main + GPU train  [clone mobilenetv2-verified]
- **MBConv block fwd/back fragment** (mirror ch7's `irBlockFwd`/`irBlockBack`): expand
  `1×1 conv→BN→swish` (`ic→mid=t·ic`) → depthwise `dw→BN→swish` (stride 1 or 2; 3×3 or 5×5) →
  **SE module** (E3) → project `1×1 conv→BN` (no swish); add the residual skip iff stride-1 ∧
  `ic=oc`. Backward = the reverse, with the SE two-path inserted between project-back and
  depthwise-back. Swap ch7's `relu6`/`relu6Back` → `swish`/`swishBack` (E1).
- **block list** — the EfficientNet-B0-style `[t, c, n, s]` config as a `blocks`-style list
  (data-driven, exactly like ch7's `blocks`); a reduced-downsample CIFAR config (see §5).
- **`EfficientNetLayout`** (IreeRuntime.lean) — clone `MobileNetV2Layout`; add the 2 SE dense
  weights+biases per block (`Ws₁ [c,r]`, `bs₁ [r]`, `Ws₂ [r,c]`, `bs₂ [c]`), depthwise kernels
  `[mid,1,kH,kW]`. Data-driven from the same `blocks`.
- **`MainEfficientNetVerified.lean`** + lakefile `efficientnet-verified` exe — clone
  `MainMobilenetV2Verified.lean` (flat `[BS,3072]` `%x` reshaped internally; mean-loss cotangent
  ÷B; `mlpTrainStepV` FFI loop; programmatic He/γ=1/β=0 init). GPU-train CIFAR-10.
- **De-risk each fragment on `iree-compile` before the full chain** (swish → sigmoid → SE →
  one MBConv fwd → small-net train step → full depth, the ch7 de-risk rhythm).

**NB — naming clash:** `efficientnet-train` (lakefile) is the **OLD unverified** `NetSpec`/
`MlirCodegen` path. The verified one is a NEW file `MainEfficientNetVerified.lean` + exe
`efficientnet-verified` (do not overwrite the old one). Likewise an `efficientnet-v2-train`
exists — leave it.

---

## 3. Suggested first steps for the clean session
1. `git log --oneline -10`; build + audit green (`lake build Proofs`; audit 156/156 — see §4).
   Skim `EfficientNet.lean` (`efficientnet_has_vjp`, `mbconvBody`, `seGate`, `seBlock_has_vjp`),
   `SE.lean` (`seBlock`, `elemwiseProduct_has_vjp`), `LayerNorm.lean` (`swish`, `swish_has_vjp`).
2. Re-read ch7's `tests/TestMobilenetV2{Fwd,Train}.lean` + `MainMobilenetV2Verified.lean` +
   `MobileNetV2Layout` + the C1/C3 depthwise op pairs — ch8's renderer/layout/Main are near-clones.
3. **Do E1 (swish) then E2 (sigmoid)** first — small, iree-validate standalone (mirror
   `tests/TestRelu6.lean`). Then **E3 (SE module)** — the hard part; de-risk a standalone
   `tests/TestSE.lean` fwd+back on iree. Then E4 (the MBConv trainer).

---

## 4. Build / audit / run

```bash
cd /home/skoonce/lean/proof_verify_demo/lean4-mlir
lake build Proofs
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c 'depends on axioms: \[propext, Classical.choice, Quot.sound\]$'
#   must equal total 'depends on axioms:' lines (156 now — ENet headlines already in it)

export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
export IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0
DATA=/home/skoonce/lean/claude_max/lean4-jax/data            # cifar-10/ lives here
# ch7 reference trainer (works; the template to clone):
.lake/build/bin/mobilenetv2-verified "$DATA"                 # MobileNetV2, 10%→57.1%
# render+compile an op pair standalone (the E1/E2 de-risk pattern):
lake env lean tests/TestRelu6.lean                           # → @relu6_{fwd,back} iree-compile OK
lake env lean tests/TestDepthwiseStrided.lean                # → @dwstrided_{fwd,back} iree-compile OK
# ch8 target (once built): .lake/build/bin/efficientnet-verified "$DATA"
```
- **Running a trainer in this harness:** Bash tool `run_in_background: true`; do NOT
  `nohup … &` (detached child gets killed) and do NOT `pkill -f <exe>` from a shell whose
  own cmdline contains that string. The exe flushes per epoch (`(← IO.getStdout).flush`).
- **Rendering MLIRs:** the `tests/TestMobilenetV2*.lean` write `verified_mlir/*.mlir` via
  `#eval main` under `lake env lean …` (needs `IO.FS.createDirAll "verified_mlir"`).

---

## 5. Gotchas (carried from ch7 — read before editing)

- **swish/sigmoid are SMOOTH** — no kink, no mask, no smoothness hypothesis. `stablehlo.logistic`
  is `σ`; swish fwd = `multiply(x, logistic(x))`. This makes E1/E2 *easier* than relu6; the whole
  net is UNCONDITIONAL. (`select`/`compare` only appear in ch7's relu6 — none here.)
- **SE is a FAN-OUT multiplied back** — the SE backward has TWO paths into `dx` (main scale +
  gate gradient); both must be emitted and summed. The gate value is reused; save it once. The
  squeeze GAP-back (`broadcast dy/(h·w)`) and the broadcast-back (`reduce-sum over spatial`) are
  ADJOINTS of each other's forward steps — get the `÷h·w` only on the squeeze side.
- **5×5 depthwise = no new op** — pass `kH=kW=5` to the existing `depthwiseF`/`depthwiseStridedF`;
  the emit pads `(k−1)/2 = 2`. EfficientNet-B0 uses 3×3 and 5×5 stages.
- **Strided depthwise = the C3 op** (`depthwiseStridedF`/`depthwiseStridedBack`) — MBConv
  downsamples via stride-2 depthwise, already built + iree-validated in ch7. Backward = zero-
  upsample (`stablehlo.pad` interior=1) + reversed-kernel stride-1 depthwise; weight-grad =
  upsample dy then `batch_group_count=c` stride-1 weight-grad.
- **Per-example instance-norm washout in deep nets** — ch7's deep net needed **lr=0.3** (not 0.1)
  after a ~2-epoch warmup; at lr=0.1 it crept ~0.2%/epoch (NOT a bug — monotonic = correct
  gradients; the proven BN is per-example instance-norm, not batch-norm, so many norm layers erase
  signal). Instance-norm masks divergence (renormalizes activations) ⇒ too-high-lr shows as flat-
  chance, not NaN. Start EfficientNet at lr≈0.2–0.3, tune.
- **Dims must keep feature maps non-degenerate** — for CIFAR 32×32 use a reduced-downsample
  config (ch7 used stem-s2 + 2 strided-depthwise blocks: 32→16→8→4; no map below the kernel).
  A 5×5 depthwise needs the map ≥ 5 (so don't 5×5 below 8×8 unless padded — SAME pad handles it,
  but keep ≥ the stride·2 tiling rule).
- **`%x` is the flat `[BS,3072]` FFI buffer** — reshape to `[BS,3,32,32]` inside the func; the
  stem conv + its weight-grad read the reshaped `%xr`. **Mean-loss cotangent** (`dy = (softmax−
  onehot)/B`) so `lr` is well-scaled.
- **Param order is the single source of truth** — generate argSig / forward / backward / SGD /
  return / layout `specs` / init all from one `blocks` list (ch7's `blocks`/`allParams`), else the
  FFI silently binds the wrong buffer. The SE adds 2 dense W+b per block — thread them through.
- **GAP-of-instance-norm degeneracy** — ch7's net was frozen at 9.99% until a final
  `1×1 conv→BN→swish` HEAD was added before GAP (GAP of a raw instance-norm BN is the constant β,
  input-independent). EfficientNet's standard head (`1×1 conv→BN→swish`, `cout→1280`-ish) plays
  this role — KEEP IT (use swish, not relu6). Without an activation between the last linear BN and
  GAP, the net won't learn.
- **Adding an SHlo op = 9-site lockstep** (SHlo ctor / den / faithful / Raw / skel / Tok / toToks /
  parseStack / parseStack_toToks). `relu6F`/`selectMid` (ch7 C2) and `depthwiseStridedF`/`Back`
  (ch7 C3) are the worked op-pair examples. rfl-faithful forwards do NOT go in AuditAxioms.
- **emitter byte-stability:** after editing, `git diff --stat verified_mlir/` must be empty for
  unrelated `.mlir` files (resnet34, mobilenetv2).

---

## 6. The honest framing (don't oversell)

- The verified trainer is a **codegen artifact** — hand-rendered StableHLO, faithful **PER-OP**
  (each fragment is the StableHLO a proven-faithful emitter produces: swish = `swish_has_vjp_correct`,
  sigmoid = `sigmoid_has_vjp`, SE = `seBlock_has_vjp_correct` / `broadcastFlat_has_vjp`, depthwise/
  strided-depthwise/per-channel-BN/residual/GAP/dense as in ch6/ch7). It is **not** a single
  `den(trainStep)=…` theorem; the whole-net correctness theorem is `efficientnet_has_vjp` (UNCONDITIONAL,
  audit-clean), and the trainer mirrors its structure — validated by training (loss/accuracy decreasing),
  exactly like ch6/ch7's `*-verified` exes.
- The proven apex is a **small EfficientNet**; a deeper / downsampling B0-style config for a real
  accuracy number is the codegen scaling job (E4 + the block list), not new math.
- Per-channel BN here is per-example **instance-norm** (matches the proven `bnForward` per channel-
  slice), not batch-BN. The chapter win is a **correctness/codegen** result (MBConv + squeeze-excite +
  swish verified end-to-end and GPU-trained), not a SOTA number.

---

## 7. File map (ch8)

- `LeanMlir/Proofs/LayerNorm.lean` — `swish(_has_vjp(_correct))`, `swishScalarDeriv`. **PROOFS DONE.**
- `LeanMlir/Proofs/SE.lean` — `seBlock`, `elemwiseProduct(_has_vjp)`, `seBlock_has_vjp(_correct)`,
  `pdiv_mul`. **PROOFS DONE.**
- `LeanMlir/Proofs/EfficientNet.lean` — `sigmoid(_has_vjp)`, `broadcastFlat(_has_vjp)`, `seGate(_has_vjp)`,
  `mbconvBody(_has_vjp)`, `mbconvResidual_has_vjp(_at)`, apex `efficientnet_has_vjp(_at)(_correct)`
  (UNCONDITIONAL). imports Depthwise/SE/LayerNorm. **PROOFS DONE.**
- `LeanMlir/Proofs/StableHLO.lean` — **E1/E2 add here:** `swishF`/`swishBack` + `sigmoidF`/`sigmoidBack`
  SHlo op pairs (9-site lockstep). Reuses `depthwiseF/Back`, `depthwiseStridedF/Back`, `bnPerChannelF/Back`,
  `addV`, `gapF`, `denseF`, `flatConvF`.
- `LeanMlir/Proofs/StableHLOParse.lean` — parser cases for the new ops.
- `LeanMlir/Proofs/IRPrint.lean` — **⭐ reference:** `IR.swish_back_bridge` / `IR.sigmoid_back_bridge`
  (audited swish/sigmoid backward StableHLO), `swishF`/`swishB`/`swishFwdM`/`swishBackM`, and the
  **full MBConv-with-squeeze-excite forward+backward emit (~1413–1614)** — mirror it directly for E3/E4.
- `tests/TestSwish.lean` / `tests/TestSigmoid.lean` (NEW, E1/E2) / `tests/TestSE.lean` (NEW, E3) /
  `tests/TestEfficientNet{Fwd,Train}.lean` (NEW, E4) — standalone render + iree-compile, mirror
  `tests/TestRelu6.lean` and `tests/TestMobilenetV2{Fwd,Train}.lean`.
- `LeanMlir/IreeRuntime.lean` — add **`EfficientNetLayout`** (clone `MobileNetV2Layout`; +2 SE dense
  W/b per block; depthwise kernels `[mid,1,kH,kW]`).
- `MainEfficientNetVerified.lean` (NEW) + lakefile `efficientnet-verified` — clone
  `MainMobilenetV2Verified.lean`. (Do NOT touch the old unverified `efficientnet-train`/`-v2-train`.)
- `verified_mlir/efficientnet_{fwd,train_step}.mlir` — E4 rendered.
- **Reuse templates (ch7):** `tests/TestMobilenetV2Train.lean` (data-driven `[t,c,n,s]` renderer +
  strided depthwise fwd/back helpers), `MobileNetV2Layout` (layout generator), `MainMobilenetV2Verified.lean`
  (trainer), `tests/TestRelu6.lean` + `tests/TestDepthwiseStrided.lean` (standalone op-pair iree checks),
  `relu6F`/`selectMid` + `depthwiseStridedF`/`depthwiseStridedBack` (op-pair lockstep worked examples).
