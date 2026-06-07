# Verified MobileNetV2 (Chapter 7) — handoff

Continuation doc for the **MobileNetV2** chapter of the verified-codegen ladder. ch7
brings MNv2 into the same "render via the proof-carrying StableHLO emitter + GPU-train
on the rendered text" line that ch6 just closed for ResNet-34 (`resnet34-verified`).

> **Status in one line — ch7 is DONE (C1+C2+C4; C3 optional):** the whole-network VJP
> of MobileNetV2 — **`Proofs.mobilenetv2_has_vjp_at`** (`_correct`), UNCONDITIONAL via
> `MobileNetV2Concrete.mnv2Concrete_has_vjp_correct` — was already PROVEN + audit-clean,
> and now the **verified codegen GPU-TRAINS**: `mobilenetv2-verified` climbs CIFAR-10
> **12.3%→34.8% over 10 epochs** (monotonic, no NaN). Done this session: **C1** depthwise
> SHlo op pair (`depthwiseF`/`depthwiseBack`), **C2** relu6 op pair (`relu6F`/`selectMid`),
> **C4** inverted-residual renderer + `MobileNetV2Layout` + `MainMobilenetV2Verified` +
> `mobilenetv2-verified` exe. **C3** (strided depthwise) still **optional** — this config
> downsamples via a regular stride-2 stem + maxpool, stride-1 IR blocks = proven apex.

> **⚠ The one gotcha that cost a debug cycle (don't repeat):** ending the net on a
> linear-bottleneck projection BN → GAP is **degenerate** — per-example instance-norm
> zeroes each channel's spatial mean, so `GAP(γ·x̂+β) = β`, **constant across inputs** →
> every image gets the same logits → frozen at exactly 9.99%. **Fix = the standard MNv2
> "features" head** (`1×1 conv 64→128 → BN → relu6`) before GAP; the relu6 restores a
> per-input pooled mean. (ResNet-34 dodged it via its final `relu(BN+skip)`.) Any GAP that
> follows a raw instance-norm BN needs a nonlinearity in between.

Updated 2026-06-07 by the session that finished ch7 codegen (`mobilenetv2-verified`
GPU-trains 12.3%→34.8%). Commits `1096623`(C1) `f8f340b`(C2) `4724d49`(C4a) `ab1a376`(C4b)
`7f48229`(C4c), all local-not-pushed. `origin/main` is behind by ch6 + ch7; commit-to-main
is fine, **never push without explicit per-push permission**.

Original handoff (pre-codegen) written 2026-06-07 by the ch6 session; §§1–7 below are the
reference it left — still accurate, with C1/C2/C4 now marked done inline.

---

## 0. The pattern (unchanged from every chapter)

Typed, proof-carrying StableHLO emitter. Per op-graph:
`SHlo a ─skel→ Raw ─toToks→ [Tok] ─emitTok→ StableHLO text`, two halves:
- **Semantic R4:** `den (graph) = <proven Mathlib fderiv quantity>` (faithfulness).
- **Syntactic R4:** `parse (toToks (skel a)) = skel a` (round-trip, `roundtrip`).

Deliverables per chapter: whole-net forward graph + faithfulness, full SGD train-step
renderer, a `*-verified` exe that GPU-trains on the rendered text, audit ℝ-headline(s)
3-axiom-clean.

Ladder: ch2 92.10%, ch3 97.86%, ch4 98.89%, ch5-A 66%, ch5-B 57%, ch6-A (ResNet-style)
60.8%, **ch6-B (real ResNet-34) 62%+** (verified architecture AND GPU-trained). **ch7 =
MobileNetV2** (architecture verified; codegen + train = this doc).

---

## 1. ⭐ WHAT IS DONE — the MNv2 proofs (audit-clean) + ch6 reuse

### The MNv2 VJP stack (committed, e.g. `c6e13a3`; in the 155 audited headlines)
All in **`LeanMlir/Proofs/MobileNetV2.lean`** (imports `Depthwise`) and
**`LeanMlir/Proofs/Depthwise.lean`**.

| Piece | Theorem / def | Notes |
|---|---|---|
| depthwise conv input-VJP | `depthwiseFlat`, `depthwise_has_vjp3(_correct)`, `depthwiseFlat_has_vjp` | `DepthwiseKernel c kH kW` (one filter/channel); emit = `feature_group_count = c`. |
| depthwise weight/bias-VJP | `depthwise_weight_grad_has_vjp3`, `depthwise_bias_grad_has_vjp` | for the SGD weight/bias grads. |
| relu6 | `relu6` (= `min(max x 0) 6`), `relu6_has_vjp_at`, `pdiv_relu6` | **two-sided** kink — smooth iff `x≠0 ∧ x≠6`; backward mask `select(0<x<6, dy, 0)`. |
| 1×1-conv→BN→relu6 | `convBnRelu6_has_vjp_at(_differentiableAt)` | expand stage / stem. |
| depthwise→BN→relu6 | `dwBnRelu6_has_vjp_at` | depthwise stage. |
| inverted-residual block | `invresBody_has_vjp_at` (expand→dw→project, no final relu6), `invresSkip_has_vjp_at` (`residual` of the body, stride-1 / `ic=oc`) | the MNv2 block. |
| **WHOLE NET** | `mobilenetv2Forward`, **`mobilenetv2_has_vjp_at(_correct)`** | `dense ∘ GAP ∘ invresBody ∘ residual(invresBody) ∘ (relu6∘BN∘conv stem)`. |
| **UNCONDITIONAL** | `MobileNetV2Concrete.mnv2Concrete_has_vjp_correct` | concrete instance, every `≠0∧≠6` / smoothness hyp discharged (non-vacuity). |

Audited headlines: `depthwise_has_vjp3_correct`, `relu6_has_vjp_at`,
`mobilenetv2_has_vjp_at_correct`, `MobileNetV2Concrete.mnv2Concrete_has_vjp_correct`.

**NB — the proven apex is a SMALL MNv2** (stem + 1 skip-IR + 1 no-skip-IR, **stride-1,
no spatial downsampling**), the analogue of ch6-A's 2-block ResNet-*style* net. Going to a
deeper / downsampling MNv2 is the codegen-side scaling job (see §2), exactly as ch6 went
ch6-A → ch6-B (resnet34).

### ⭐ What ch6 leaves REUSABLE (this is why ch7 is cheap)
- **Per-channel BatchNorm op pair** — `bnPerChannelF`/`bnPerChannelBack` (B8b,
  `StableHLO.lean`); MNv2's BN is the same per-example per-channel norm. Reuse verbatim.
- **Residual `addV`** (ch6-A) — the IR skip fan-in. Reuse.
- **GAP `gapF`, dense `denseF`, conv `flatConvF`, relu, maxpool** — all present.
- **The strided-conv decimate trick** (B1: `decimateFlat`, `flatConvStride2 = decimate ∘
  conv`) — reuse for the **strided depthwise** (MNv2 downsamples via stride-2 depthwise).
- **The whole `resnet34-verified` codegen harness** — `tests/TestResnet34{Fwd,Train}.lean`
  (data-driven `Blk`-list renderer + fwd/back fragment helpers), `ResNet34Layout`
  generator (IreeRuntime.lean), `MainResnet34Verified.lean`, the `mlpTrainStepV` FFI loop.
  ch7's renderer/layout/Main are a near-clone with the IR block in place of the residual block.

---

## 2. ⭐ WHAT REMAINS — the verified codegen (ch7 = render + train)

> **DONE (this session).** C1, C2, C4 all landed; only C3 (strided depthwise) is left,
> and it's optional. The final shipped config: **stem** 3×3 stride-2 conv (3→32, reuses
> ch6's regular strided conv, NOT depthwise) → BN → relu6 → maxpool (16→8) → **IR-A**
> inverted-residual w/ skip (ic=oc=32, mid=64, stride-1 @8×8) → **IR-B** inverted-residual
> no-skip (32→64, mid=64, @8×8) → **head** 1×1 conv (64→128) → BN → relu6 → GAP → dense
> 128→10. 34 params, lr=0.1, GPU-trains 12.3%→34.8%. The depthwise **weight-grad** turned
> out to need `batch_group_count = c` (the XLA depthwise-filter-grad trick: transpose
> inp/dy → `[c,B,H,W]`, conv → `[1,c,3,3]`, reshape `[c,1,3,3]`) — de-risk it standalone
> on iree first. Files: `tests/TestDepthwise.lean` (C1), `tests/TestRelu6.lean` (C2),
> `tests/TestMobilenetV2{Fwd,Train}.lean` (C4), `MobileNetV2Layout` (IreeRuntime.lean),
> `MainMobilenetV2Verified.lean`, exe `mobilenetv2-verified`.

The proof half is done; everything below is **render/GPU** (no new Mathlib). Suggested
order, mirroring ch6's de-risk-on-iree-compile-first discipline:

### C1 — depthwise conv SHlo op pair  ✅ DONE  [the one genuinely-new emitter op]
A `depthwiseF` / `depthwiseBack` (+ weight-grad) op pair, 9-site lockstep mirroring B3's
strided-conv pair (`flatConvStridedF`/`convStridedBack`, `StableHLO.lean`):
- **`depthwiseF`** forward: `stablehlo.convolution` with **`feature_group_count = c`** and
  a `[c, 1, kH, kW]` kernel (one filter per channel), SAME pad, + bias. `den = depthwiseFlat
  W b` (rfl, like `flatConvStridedF_faithful`).
- **`depthwiseBack`** input-VJP: reversed-kernel depthwise conv (transpose/reverse the
  per-channel filters, again `feature_group_count = c`). `den = depthwiseFlat_has_vjp.backward`,
  faithful via `depthwise_has_vjp3_correct`.
- weight-grad fragment for the trainer (from `depthwise_weight_grad_has_vjp3`).
- 9 sites: SHlo ctor / den / faithful / Raw / skel / Tok / toToks / parseStack /
  parseStack_toToks. Then a standalone `tests/TestDepthwise.lean` render + `iree-compile`
  rocm (mirror `tests/TestPerChannelBn.lean`). **Watch:** `feature_group_count` + the
  `[c,1,kH,kW]` kernel shape are the only deltas from a normal conv.

### C2 — relu6 SHlo op  ✅ DONE  [small]
`relu6F` forward `clamp(x,0,6) = maximum(minimum(x, 6), 0)`; backward mask
`select(0 < x ∧ x < 6, dy, 0)` (two `compare` + `and` + `select`, or two chained selects).
`den = relu6 n` / its VJP. A two-sided `reluF`/`selectPos`. rfl-faithful forward.

### C3 — strided depthwise (downsampling)  ⏭ OPTIONAL / NOT DONE  [skipped — stride-1 IR blocks + regular strided stem reach a trainable net]
MNv2 halves spatial via **stride-2 depthwise**. Reuse the B1 identity:
`dwconv_stride2 = decimateFlat ∘ depthwiseFlat` (+ VJP via `vjp_comp` + `decimateFlat_has_vjp`),
then a `depthwiseStridedF`/`depthwiseStridedBack` op pair (forward `window_strides=[2,2]`;
backward zero-upsample via `stablehlo.pad` interior=1 then reversed-kernel depthwise —
exactly the `convStridedBack` shape). *Skippable for a first stride-1 verified MNv2* (the
proven apex is stride-1); needed for a downsampling config with a real accuracy story.

### C4 — the inverted-residual renderer + layout + Main + GPU train  ✅ DONE  [clone resnet34-verified]
- **IR block fwd/back fragment** (mirror `idBlock`/`downBlock` in `TestResnet34Train.lean`):
  expand `1×1 conv→BN→relu6` (`ic→mid=t·ic`) → depthwise `dw→BN→relu6` → project
  `1×1 conv→BN` (no relu6, `mid→oc`); add the residual skip iff stride-1 ∧ `ic=oc`.
  Backward = the reverse (project-BN back → 1×1 conv back → relu6 mask → dw-BN back →
  depthwise back → relu6 mask → expand-BN back → 1×1 conv back, residual fan-in).
- **block list** — the `[t, c, n, s]` config as a `Blk`-style list (data-driven, like ch6's
  `blocks`/`allParams`); a reduced-downsample CIFAR config (see §5 dims gotcha).
- **`MobileNetV2Layout`** (IreeRuntime.lean) — the `(dims, initKind)` param specs in
  func-arg order (depthwise kernels are `[c,1,kH,kW]`); generator like `ResNet34Layout`.
- **`MainMobilenetV2Verified.lean`** + lakefile `mobilenetv2-verified` exe — clone
  `MainResnet34Verified.lean` (flat `[BS,3072]` `%x` reshaped internally; mean-loss
  cotangent ÷B; `mlpTrainStepV` FFI loop; programmatic He/γ=1/β=0 init). GPU-train CIFAR-10.
- **De-risk each fragment on `iree-compile` before the full chain** (forward first, then
  the small-net train step, then full depth — the ch6 B9a→B9b→B9c rhythm).

**NB — naming clash:** `MainMobilenetV2Train.lean` already exists but is the **OLD
unverified** `NetSpec`/`MlirCodegen` path. The verified one is a NEW file
`MainMobilenetV2Verified.lean` + exe `mobilenetv2-verified` (do not overwrite the old one).

---

## 3. Suggested first steps for the clean session
1. `git log --oneline -5`; build + audit green (`lake build Proofs`; audit 155/155 — see §4).
   Skim `MobileNetV2.lean` (`mobilenetv2_has_vjp_at`, `mnv2Concrete_has_vjp_correct`) and
   `Depthwise.lean` (`depthwise_has_vjp3`, `depthwiseFlat_has_vjp`, `relu6_has_vjp_at`).
2. Re-read ch6's `tests/TestResnet34{Fwd,Train}.lean` + `MainResnet34Verified.lean` +
   `ResNet34Layout` — ch7's renderer/layout/Main are near-clones.
3. **Do C1** (depthwise op pair) first — it's the one new emitter op; iree-compile-validate
   it standalone (mirror `tests/TestPerChannelBn.lean`). Then C2 (relu6), then C4 (the
   trainer); fold C3 (strided depthwise) in when you want downsampling.

---

## 4. Build / audit / run

```bash
cd /home/skoonce/lean/proof_verify_demo/lean4-mlir
lake build Proofs
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c 'depends on axioms: \[propext, Classical.choice, Quot.sound\]$'
#   must equal total 'depends on axioms:' lines (155 now — MNv2 headlines already in it)

export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
export IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0
DATA=/home/skoonce/lean/claude_max/lean4-jax/data            # cifar-10/ lives here
# ch6 reference trainer (works; the template to clone):
.lake/build/bin/resnet34-verified "$DATA"                    # ResNet-34, ~62%+ climbing
# render+compile an op pair standalone (the C1/C2 de-risk pattern):
lake env lean tests/TestPerChannelBn.lean                    # → @perchannel_bn_{fwd,back} iree-compile OK
# ch7 target (once built): .lake/build/bin/mobilenetv2-verified "$DATA"
```
- **Running a trainer in this harness:** Bash tool `run_in_background: true`; do NOT
  `nohup … &` (detached child gets killed) and do NOT `pkill -f <exe>` from a shell whose
  own cmdline contains that string. The exe flushes per epoch (`(← IO.getStdout).flush`).
- **Rendering MLIRs:** the `tests/TestResnet34*.lean` write `verified_mlir/*.mlir` via
  `#eval main` under `lake env lean …` (needs `IO.FS.createDirAll "verified_mlir"`).

---

## 5. Gotchas (carried from ch6 — read before editing)

- **Depthwise = `feature_group_count = c` + `[c,1,kH,kW]` kernel.** That attribute (and the
  rank-4 kernel with `ic=1`) is the ONLY emitter delta from a normal conv. The backward is
  a reversed-kernel depthwise conv (same `feature_group_count`). `depthwiseConv2d` in
  `Depthwise.lean:67` documents the exact StableHLO shape.
- **relu6 is a TWO-SIDED kink** — smooth iff `x≠0 ∧ x≠6`; backward mask is `0<x<6` (not
  just `x>0`). The concrete instance already discharges both bounds — copy that style if you
  ever need a concrete MNv2 non-vacuity at training dims.
- **Strided depthwise = `decimate ∘ depthwise`** (NOT a from-scratch stride VJP) — reuse
  `decimateFlat` + `vjp_comp`, exactly like B1's `flatConvStride2`. Backward = zero-upsample
  (`stablehlo.pad` interior=[..,1,1]) then reversed-kernel stride-1 depthwise.
- **Dims must keep feature maps non-degenerate** — ch6 hit a `3×3 stride-2 on 2×2` failure
  (can't tile). For CIFAR 32×32, use a reduced-downsample config (e.g. fewer stride-2
  stages) so no map shrinks below the kernel. (ch6 used stride-1 stem + maxpool + 3
  downsamples 16→8→4→2.)
- **`%x` is the flat `[BS,3072]` FFI buffer** — reshape to `[BS,3,32,32]` inside the func
  (the FFI's `xShape` is `[batch,3072]`); the stem conv + its weight-grad read the reshaped
  `%xr`. **Mean-loss cotangent** (`dy = (softmax−onehot)/B`) so `lr` is well-scaled.
- **Param order is the single source of truth** — generate argSig / forward / backward /
  SGD / return / layout `paramShapes` / init all from one block list (ch6's `allParams`),
  else the FFI silently binds the wrong buffer to a param.
- **Adding an SHlo op = 9-site lockstep** (SHlo ctor / den / faithful / Raw / skel / Tok /
  toToks / parseStack / parseStack_toToks). `bnPerChannelF`/`bnPerChannelBack` (B8b) and
  `flatConvStridedF`/`convStridedBack` (B3) are the worked op-pair examples. rfl-faithful
  forwards do NOT go in AuditAxioms; `roundtrip` covers them structurally.
- **emitter byte-stability:** after editing, `git diff --stat verified_mlir/` must be empty
  for unrelated `.mlir` files.

---

## 6. The honest framing (don't oversell)

- The verified trainer is a **codegen artifact** — hand-rendered StableHLO, faithful
  **PER-OP** (each fragment is the StableHLO a proven-faithful emitter produces: depthwise =
  `depthwise_has_vjp3_correct`, per-channel BN = `bnPerChannelBack_faithful`, relu6 =
  `relu6_has_vjp_at`, residual/GAP/dense/conv as before). It is **not** a single
  `den(trainStep)=…` theorem; the whole-net correctness theorem is `mobilenetv2_has_vjp_at`
  (parametric, audit-clean), and the trainer mirrors its structure — validated by training
  (loss/accuracy decreasing), exactly like ch6's `resnet34-verified`.
- The proven apex is a **small, stride-1 MNv2**; a deeper / downsampling config for a real
  accuracy number is the codegen scaling job (C3 + the block list), not new math.
- Per-channel BN here is per-example **instance-norm** (matches the proven `bnForward` per
  channel-slice), not batch-BN. The chapter win is a **correctness/codegen** result
  (depthwise-separable convs + inverted residuals + relu6 verified end-to-end and
  GPU-trained), not a SOTA number.

---

## 7. File map (ch7)

- `LeanMlir/Proofs/Depthwise.lean` — depthwise conv VJPs (`depthwiseFlat`,
  `depthwise_has_vjp3(_correct)`, `depthwise_weight_grad_has_vjp3`, `depthwise_bias_grad_has_vjp`).
  imports CNN-level machinery. **PROOFS DONE.**
- `LeanMlir/Proofs/MobileNetV2.lean` — `relu6(_has_vjp_at)`, `convBnRelu6`/`dwBnRelu6`,
  `invresBody`/`invresSkip` blocks, apex `mobilenetv2_has_vjp_at(_correct)`,
  `MobileNetV2Concrete.mnv2Concrete_has_vjp_correct`. imports Depthwise. **PROOFS DONE.**
- `LeanMlir/Proofs/StableHLO.lean` — **C1/C2 add here:** `depthwiseF`/`depthwiseBack`
  (+ strided variant C3) + `relu6F` SHlo ops (9-site lockstep). Reuses `bnPerChannelF/Back`,
  `addV`, `gapF`, `denseF`, `flatConvF`, `decimateFlat`.
- `LeanMlir/Proofs/StableHLOParse.lean` — parser cases for the new ops.
- `tests/TestDepthwise.lean` (NEW, C1) / `tests/TestMobilenetV2{Fwd,Train}.lean` (NEW, C4) —
  standalone render + iree-compile, mirror `tests/TestResnet34{Fwd,Train}.lean`.
- `LeanMlir/IreeRuntime.lean` — add **`MobileNetV2Layout`** (clone `ResNet34Layout`; depthwise
  kernels `[c,1,kH,kW]`).
- `MainMobilenetV2Verified.lean` (NEW) + lakefile `mobilenetv2-verified` — clone
  `MainResnet34Verified.lean`. (Do NOT touch the old unverified `MainMobilenetV2Train`.)
- `verified_mlir/mobilenetv2_{fwd,train_step}.mlir` — C4 rendered.
- **Reuse templates (ch6):** `tests/TestResnet34Train.lean` (data-driven `Blk` renderer +
  fwd/back helpers), `ResNet34Layout` (layout generator), `MainResnet34Verified.lean`
  (trainer), `tests/TestPerChannelBn.lean` (standalone op-pair iree check),
  `flatConvStridedF`/`convStridedBack` + `bnPerChannelF`/`bnPerChannelBack` (op-pair lockstep
  worked examples).
