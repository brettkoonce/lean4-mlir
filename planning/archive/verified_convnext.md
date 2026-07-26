# Verified ConvNeXt (Chapter 9) — handoff / plan for a fresh session

Plan doc for the **ConvNeXt** chapter of the verified-codegen ladder. ch9 brings
ConvNeXt-T (Liu et al. 2022, "A ConvNet for the 2020s") into the same "render via the
proof-carrying StableHLO emitter + GPU-train on the rendered text **on Imagenette 224²**"
line as ch6 (ResNet-34), ch7 (MobileNetV2), and ch8 (EfficientNet-B0).

> **✅ CODEGEN DONE (2026-06-07); ⚠️ ACCURACY NOT YET VALIDATED.** Codegen built, renders +
> iree-compiles (ROCm), GPU-trains end-to-end (no crash/NaN, gradients flow through the FFI).
> The proofs were already there (`Proofs.convnext_has_vjp` / `_correct`, UNCONDITIONAL except
> the four `0 < ε` LayerNorm conditions, audit 157/157). The ch9 deliverable — the **VERIFIED
> STABLEHLO CODEGEN** — is complete: `geluF`/`geluBack` SHlo op pair (+ `geluScalarDeriv_eq`
> closed-form lemma, all 3-axiom clean, audit stays 157/157); the ConvNeXt-T renderer
> (`tests/TestConvNeXt{Fwd,Train}.lean`, [3,3,9,3] @ [96,192,384,768], 180 params / 27.8M floats
> ≈ ConvNeXt-T's 28M); `ConvNeXtLayout`; `MainConvNeXtVerified.lean` + lakefile `convnext-verified`;
> `verified_mlir/convnext_{fwd,train_step}.mlir`. Both MLIRs iree-compile. Commits de2d223 (N1) ·
> 8fb7880 (N2+N4 block) · eae89af (N3 strided) · 937f69b (N5 full net), local-not-pushed.
>
> **⚠️ Training result so far (stopped after epoch 1 — full 20-epoch run is ~13 h):** `convnext-verified`
> on imagenette 224² (BS=32, lr=0.1) → **epoch 1 = 387/3904 = 9.91% = CHANCE**. This is the §7-flagged
> "epoch-1 at chance" signal. Most likely **deep global-scalar-LN washout at lr=0.1** (the ch7 MNv2
> lesson: a deep per-example-norm net crept ~0.2 %/epoch at lr=0.1, needed lr=0.3 — and ConvNeXt's
> LN is global over the WHOLE c·h·w, even more aggressive than ch7's per-channel instance-norm, with
> 18 + head LNs); a `layerScale` γ init of 1.0 (vs the paper's 1e-6) makes the blocks fully perturb
> the residual stream early. **OR** a degenerate forward / backward bug. **FOLLOW-UP needed** (not yet
> done): (a) re-render with lr=0.3 and/or train more epochs; (b) per §7, run `convnext_fwd` on two
> different batches and check the logits VARY (if identical → degenerate forward, debug the head /
> global LN); (c) if still dead, add a finite-difference/adjoint gradcheck on the block backward
> (the hand-wired reverse threading or the even-kernel strided backward are the suspects). The
> **codegen pipeline is validated** (render→compile→GPU-train, gradients flow, no NaN); the **trained
> accuracy is NOT** — don't claim a ConvNeXt accuracy number until this is resolved.
>
> Implementation notes vs. the original plan below: the whole net is **hand-rendered 4-D
> StableHLO fragments** (ch7/ch8 style), with `geluF`/`geluBack` as the proof-carrying SHlo
> *witness* (TestGelu compiles the per-op text). LN = the flat global-scalar `bnF`/`bnBack`
> form (reshape [B,c,h,w]↔[B,c·h·w]). Patchify (4×4/s4) + downsample (2×2/s2) needed
> **hand-derived even-kernel** transposed-conv backwards (the proven `flatConvStride2` is
> odd-kernel only: pad (k-1)/2, kernel 2 ⇒ pad 0 ⇒ shape mismatch) — input-grad = dilate dy
> (interior s-1, high s-1), reverse(Wᵀ), conv pad [[s-1,0]]; weight-grad = dilate-no-high +
> valid conv (verified on concrete h=1,2 examples). LN is **per-example**, so eval uses the
> same stats as train (no batch-stat/EMA mismatch — nicer than EfficientNet's batch-norm).
> The original §§1–9 below are the as-planned design; read them for the rationale.

> **(historical, as-planned) Status in one line:** the **whole-network VJP is ALREADY PROVEN
> + AUDITED** — `Proofs.convnext_has_vjp` / `convnext_has_vjp_correct` (UNCONDITIONAL except
> the four `0 < ε` LayerNorm conditions; all-smooth, no ReLU/maxpool kinks), plus the entire
> component stack (`gelu_has_vjp_correct`, `layerNorm_has_vjp_correct`,
> `layerScale_has_vjp(_correct)`, `convNextBlock(Body)_has_vjp`). **What's missing is the
> VERIFIED STABLEHLO CODEGEN** — there is **no `geluF`/`geluBack` op, no ConvNeXt renderer,
> no `convnext-verified` exe, no `verified_mlir/convnext_*.mlir`.** This is exactly the ch8
> starting point (proofs done → build the codegen). **Target: a faithful ConvNeXt-T on
> IMAGENETTE 224²**, GPU-trained on the proof-rendered StableHLO. NB there is already an
> *UNVERIFIED* `convnext-tiny-train` (NetSpec/IRPrint path, `MainConvNeXtTrain.lean`) — the
> ch9 deliverable is the *verified* peer, `convnext-verified` (mirrors how ch8 has both
> `efficientnet-train` and `efficientnet-verified`).

Written 2026-06-07. Read §§0–5 first; §§6–9 are reference. Commit-to-`main` is the
workflow (no branches/PRs); **never push without explicit per-push permission.**

---

## 0. The pattern (unchanged from every chapter)

Typed, proof-carrying StableHLO emitter (`LeanMlir/Proofs/StableHLO.lean`). Per op:
- **Semantic R4:** `den (graph) = <proven Mathlib fderiv quantity>` (faithfulness).
- **Syntactic R4:** `parse (toToks (skel a)) = skel a` (round-trip, for typed-AST ops).

The **trainer** is hand-rendered batched StableHLO string fragments (ch7/ch8 style), each
fragment faithful PER-OP to a proven function — NOT a single `den(trainStep)` theorem.
Validated by training (loss/accuracy moving), like the other `*-verified` exes.

Ladder so far: ch2 92.10%, ch3 97.86%, ch4 98.89%, ch5-A 66% / ch5-B 57%, ch6 ResNet-34
(now **Imagenette 224²**), ch7 MobileNetV2 (now **Imagenette 224²**), ch8 EfficientNet-B0
(**Imagenette 224²**). ch9 ConvNeXt-T continues at the same native 224² resolution.

---

## 1. ⭐ WHAT IS ALREADY DONE (proofs — committed + audited in the 157)

All in **`LeanMlir/Proofs/LayerNorm.lean`** and **`LeanMlir/Proofs/ConvNeXt.lean`** (the
latter imports Depthwise + LayerNorm). Audited headlines are in `tests/AuditAxioms.lean`
(see lines ~73–130) and are part of the **157/157 3-axiom-clean** count.

| Symbol | Where | What |
|---|---|---|
| `geluScalar` / `gelu` / `geluScalarDeriv` / `pdiv_gelu` | LayerNorm.lean | GELU (tanh approximation) + its diagonal-Jacobian. |
| `gelu_has_vjp` / `gelu_has_vjp_correct` | LayerNorm.lean | GELU VJP (smooth, GLOBAL HasVJP — no kink, like swish). |
| `layerNormForward` / `layerNorm_has_vjp` / `_correct` | LayerNorm.lean | **LN ≡ `bnForward`** (per-example global norm, **scalar γ/β**). Its VJP *is* `bn_has_vjp`. |
| `layerScale` / `layerScale_has_vjp` / `_correct` | ConvNeXt.lean | per-element diagonal scale `γ ⊙ x` (`γ : Vec (c·h·w)`). |
| `convNextBlockBody(_has_vjp/_differentiable)` | ConvNeXt.lean | depthwise kH×kW → LN → 1×1 expand c→cExp → **GELU** → 1×1 project cExp→c → layerScale. |
| `convNextBlock(_has_vjp/_at)` | ConvNeXt.lean | `residual (convNextBlockBody …)` (identity skip). |
| `convNextForward` | ConvNeXt.lean | `dense ∘ headLN ∘ GAP ∘ block₂ ∘ block₁ ∘ stemLN ∘ stemConv`. |
| **`convnext_has_vjp` / `_correct`** | ConvNeXt.lean | **whole-net VJP, UNCONDITIONAL** (only the 4× `0<ε`). The apex. |
| `convnext_has_vjp_at(_correct)` | ConvNeXt.lean | the `_at` (point) restriction, for `_at` consumers. |

**Read the proof's exact forward (`convNextForward`, ConvNeXt.lean:267) — it is a
*representative* net, not full ConvNeXt-T:**
- **stem = a 1×1 conv** (`Wst : Kernel4 c ic 1 1`, via `flatConv`) — **NOT** the paper's
  4×4/s4 patchify, and **no spatial downsample.**
- **2 blocks, single channel `c`, single spatial `h×w` throughout — NO inter-stage
  downsampling.** (Same "representative fixed depth" spirit as `cnn_has_vjp_at`.)
- **LN = global per-example, scalar γ/β** (`layerNormForward (c·h·w)`) at stem-LN,
  block-internal LN, and head-LN. (This is the codebase's "LN ≡ global `bnForward`"
  identity — *not* the paper's per-token channel-LN. See §7.)

So the proof gives you every **component** VJP audit-clean; the codegen for the *faithful
multi-stage ConvNeXt-T* (patchify + downsamples + [3,3,9,3]) goes beyond the 2-block
representative proof, faithful PER-OP — exactly as ch8's E6 B0 went beyond its
representative `efficientnet_has_vjp`.

**Also already present (UNVERIFIED, the OLD IRPrint/NetSpec path — do NOT confuse with the
verified path):** `MainConvNeXtTrain.lean` (`convnext-tiny-train` exe, full ConvNeXt-T
NetSpec on `data/imagenette`), `tests/TestConvNext*.lean`, `MainAblation.lean` ch9 section,
`IR.gelu_back_bridge` / `IR.layernorm_back_bridge`. These are reference for the architecture
+ emit shapes, but the ch9 deliverable lives in the **`LeanMlir/Proofs/StableHLO.lean`**
verified path + a new `convnext-verified` exe.

---

## 2. ⭐ THE TARGET — faithful ConvNeXt-T on IMAGENETTE 224²

Mirror the unverified `convNextTiny` NetSpec (`MainConvNeXtTrain.lean`), at the paper-native
224² resolution (10 classes), real /32 spatial flow:

```
stem    : 4×4 conv stride 4 (3→96)  "patchify"             224→56
stage 1 : 3× ConvNeXt block @ 96    (LN + GELU)            @56
downsmpl: LN + 2×2 conv stride 2 (96→192)                  56→28
stage 2 : 3× ConvNeXt block @ 192                          @28
downsmpl: LN + 2×2 conv stride 2 (192→384)                 28→14
stage 3 : 9× ConvNeXt block @ 384                          @14
downsmpl: LN + 2×2 conv stride 2 (384→768)                 14→7
stage 4 : 3× ConvNeXt block @ 768                          @7
head    : globalAvgPool → LN(768) → dense 768→10 + softmax-CE
```

ConvNeXt block (depths [3,3,9,3], dims [96,192,384,768], ~28M params):
`x → depthwise 7×7 (dim) → LN → 1×1 conv dim→4·dim → GELU → 1×1 conv 4·dim→dim →
layerScale (per-channel γ) → + x` (identity residual; stride-1 blocks only — the
downsampling is the dedicated between-stage layer, NOT fused into the first block, unlike
ResNet/MNv2/EffNet).

Data (same as ch6/7/8 after this session's migration): `<DATA>/imagenette/{train.bin@256
centerCrop→224, val.bin@224}`, loaded via `F32.loadImagenetteSized`/`loadImagenette` +
`F32.centerCrop`. `DATA=/home/skoonce/lean/claude_max/lean4-jax/data`. BS=32 (224² memory),
`D0 = 3·224·224 = 150528`.

---

## 3. ⭐ WHAT'S REUSABLE FROM THE VERIFIED PATH (most of it)

| ConvNeXt piece | Reuse | From |
|---|---|---|
| depthwise 7×7 (stride 1) | `depthwiseF`/`depthwiseBack` (kernel-general — 7×7 fine; ch8 used 5×5) | ch7 C1 |
| 1×1 conv (expand/project/downsample-proj) | `conv1`-style 1×1 conv fragment | ch7/ch8 |
| LN (global scalar) | **`bnF`/`bnBack`** (ch5-B scalar-γ/β BN op = exactly `layerNormForward`) | ch5-B |
| residual add | `addV` (tree-safe skip) | ch6-A |
| global-avg-pool | `gapF` | ch6-A |
| dense head | `denseF` / dot_general | ch2–ch4 |
| 2×2 stride-2 downsample conv | `flatConvStridedF`/`convStridedBack` (stride-2 = decimate2∘stride-1) | ch6 B3 |
| strided-conv weight-grad template | `convWGradStrided` / this session's `convWGradStem` | ch6 B9 / r34 imagenette |
| imagenette loader + 224² Main/layout harness | `MainEfficientNetVerified` + `EfficientNetLayout` + `MainResnet34Verified` | ch8 / ch6 |

The **closest whole-net renderer templates** are `tests/TestEfficientNet{Fwd,Train}.lean`
(stage→block generator, depthwise + 1×1 + BN + smooth-activation + residual, 224² stem)
and this session's `tests/TestResnet34{Fwd,Train}.lean` (the 7×7-s2 **`convStem`** /
**`convWGradStem`** strided-stem fragments — directly adaptable to the 4×4/s4 patchify).

---

## 4. ⭐ WHAT'S GENUINELY NEW (the ch9 codegen work)

### N1 — GELU SHlo op pair `geluF`/`geluBack` (StableHLO.lean)
The one real new op. Template = **`swishF`/`swishBack`** (ch8 E1) — all-smooth, GLOBAL
`gelu_has_vjp`, no select-mask, no smoothness hyp. 9-site lockstep (inductive/den/skel/Raw/
Tok/toToks/emitTok + parseStack/parseStack_toToks).
- **emit fwd** (tanh approximation, matches `geluScalar`): `0.5·x·(1 + tanh(√(2/π)·(x +
  0.044715·x³)))` — constants + `multiply`/`add` + **`stablehlo.tanh`** (standard unary;
  iree-compiles). `den (geluF) = gelu` (rfl, roundtrip-covered → out of audit, like swishF).
- **emit back** = `dy ⊙ geluScalarDeriv(x)`, recomputed from saved pre-activation.
- ⚠️ **One small NEW proof:** `gelu_has_vjp.backward` uses `geluScalarDeriv x := deriv
  geluScalar x` (not a closed form). For `geluBack_faithful` to be rfl-clean you need a
  closed-form lemma `geluScalarDeriv_eq` giving the analytic derivative of the tanh-approx
  GELU — i.e. with `u = √(2/π)(x+0.044715x³)`, `t = tanh u`:
  `g'(x) = 0.5(1+t) + 0.5·x·(1−t²)·√(2/π)(1+3·0.044715·x²)`. Scalar calculus (`deriv`
  rewriting); this is the only Mathlib work in ch9. (Then emit `geluBack` from that formula.)

### N2 — LayerScale render fragment
`layerScale` = per-channel diagonal multiply `γ ⊙ x` (broadcast a `Vec c` over spatial).
Trivial: fwd = `broadcast_in_dim γ dims=[1]` + `multiply`; backward = `multiply dy by γ`
(input grad) + `dγ = Σ_{batch,spatial}(x ⊙ dy)` (a reduce over [0,2,3]). No new op needed —
a small hand-rendered fragment (peer of the BN γ/β grads). Faithful to `layerScale_has_vjp`.

### N3 — 4×4 stride-4 patchify stem (+ optional decimate-4)
The proven stem is 1×1; the faithful stem is **4×4 conv, stride 4** (224→56). Options:
- **(a) faithful, recommended:** generalize `decimateFlat` (StridedConv.lean, currently
  stride-2) to **decimate-s** (a `reindexCLM`, VJP via `pdiv_reindex` — the same mechanical
  proof, just stride `s`), giving `flatConvStrideN = decimateN ∘ (stride-1 SAME conv)`. Then
  the patchify forward = 4×4/s4 conv (pad 0, non-overlapping), backward weight-grad =
  zero-upsample-by-4 (`stablehlo.pad` interior=3) + stride-1 4×4 weight-grad — the exact
  shape of this session's `convWGradStem` but stride 4 / kernel 4. (Stem needs only dW + db,
  no input-grad — it's the first layer.)
- **(b) de-risk shortcut:** two stacked 2×2/s2 strided convs (each `flatConvStridedF`) =
  /4, reusing ch6 B3 verbatim — but that is two 2×2 convs, NOT the paper's single 4×4 patch
  embed (less faithful). Use only to unblock end-to-end before doing (a).

The between-stage **2×2/s2 downsample convs reuse ch6 B3 directly** (stride-2). ConvNeXt's
downsample is `LN → 2×2/s2 conv`; render LN (`bnF`) then the strided conv.

### N4 — ConvNeXt block renderer (fwd + backward), data-driven
One block = `depthwiseF(7×7) → bnF(LN) → conv1(expand c→4c) → geluF → conv1(project 4c→c)
→ layerScale → addV(skip)`. Backward threads in reverse: addV fan-in → layerScale back →
conv1 back (project) → geluBack → conv1 back (expand) → bnBack(LN) → depthwiseBack → +skip.
Data-driven over one `blocks` list `(name, dim, expDim, hw)` like ch7/ch8. Stride-1 only
(downsampling is separate, §N3).

### N5 — full ConvNeXt-T render + layout + Main + GPU-train
- `tests/TestConvNeXtFwd.lean` + `tests/TestConvNeXtTrain.lean` (peer of the EffNet/r34
  renderers): patchify stem → 4 stages [3,3,9,3] @ [96,192,384,768] with 3 between-stage
  downsamples (56→28→14→7) → GAP → head-LN → dense. Render `verified_mlir/convnext_{fwd,
  train_step}.mlir`; iree-compile rocm to de-risk before the Main.
- `ConvNeXtLayout` (IreeRuntime.lean — `(dims, initKind)` specs in func-arg order; depthwise
  `[c,1,7,7]`, 1×1 `[4c,c,1,1]`/`[c,4c,1,1]`, LN scalar γ/β, layerScale `[c]`, patchify
  `[96,3,4,4]`, downsample `[2c,c,2,2]`, dense `[768,10]`).
- `MainConvNeXtVerified.lean` + lakefile `convnext-verified` (clone `MainEfficientNetVerified`
  — imagenette loader, BS=32, `mlpTrainStepV` FFI, He init, mean-loss SGD). GPU-train.

---

## 5. Suggested milestones / de-risk order
1. **Build + audit green first** (`lake build Proofs`; audit must stay **157/157**; the
   ConvNeXt proofs are already in it — confirm `#print axioms convnext_has_vjp` etc.).
2. **N1 GELU op pair** standalone: `tests/TestGelu.lean` → `@gelu_fwd`/`@gelu_back`
   iree-compile rocm (mirror `tests/TestSwish.lean`). Land the `geluScalarDeriv_eq` lemma.
3. **N2 layerScale** + **N4 block** on a tiny config (`tests/TestConvNeXtBlock.lean`):
   render one block fwd+back, iree-compile. Optional finite-difference gradcheck on the
   compiled `.vmfb` (like ch8's `tests/TestSE.lean`) to validate the hand-wired backward.
4. **N3 patchify** (do option (b) first to unblock, then (a) faithful) + the 2×2 downsamples.
5. **N5 full renderer** (small depths first, e.g. [1,1,1,1], iree-compile, THEN full
   [3,3,9,3]) → layout → Main → `convnext-verified` → GPU-train imagenette 224².
6. Commit per milestone (to `main`, no push). Keep `git diff --stat verified_mlir/` empty
   for the other chapters' `.mlir` (byte-stability — the `StableHLO.lean` `#eval` regenerates
   them on elaboration).

---

## 6. Build / audit / run

```bash
cd /home/skoonce/lean/proof_verify_demo/lean4-mlir
lake build Proofs
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c 'depends on axioms: \[propext, Classical.choice, Quot.sound\]$'
#   must equal total 'depends on axioms:' lines (157 now; ConvNeXt/GELU/LN already in it)

export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
export IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0
DATA=/home/skoonce/lean/claude_max/lean4-jax/data        # imagenette/ + cifar-10/ live here

# render + iree-compile (once the renderers exist):
lake env lean tests/TestConvNeXtFwd.lean                 # writes verified_mlir/convnext_fwd.mlir
lake env lean tests/TestConvNeXtTrain.lean               # writes verified_mlir/convnext_train_step.mlir
lake build convnext-verified
.lake/build/bin/convnext-verified "$DATA"                # GPU-train imagenette 224², BS=32
```
- **Running a trainer:** Bash `run_in_background: true`; do NOT `nohup … &` (detached child
  killed at command exit). Flush per epoch (`(← IO.getStdout).flush`).
- **224² is slow** (~49× CIFAR compute; ConvNeXt-T ≈ EffNet-B0 scale) — minutes/epoch at
  BS=32; the train-step iree-compile takes a couple minutes. De-risk every fragment on
  iree-compile before the full chain.

---

## 7. Honest framing (don't oversell)

- **The whole-net VJP `convnext_has_vjp` is PROVEN + audit-clean + UNCONDITIONAL** (only the
  4× `0<ε`). That is the real verification result. The **trained model is a codegen artifact**
  faithful PER-OP (each fragment is the StableHLO of a proven-faithful emitter), validated by
  training — NOT a single `den(trainStep)` theorem, and the *faithful multi-stage ConvNeXt-T*
  (patchify + downsamples + [3,3,9,3]) extends beyond the proof's 2-block, single-spatial,
  1×1-stem *representative* witness (same relationship as ch8 E6 B0 ↔ `efficientnet_has_vjp`).
- **LayerNorm here is the codebase's global per-example scalar-γ/β LN (`= bnForward`), not
  the paper's per-token channel-LN.** It is what's proven (`layerNorm_has_vjp`) and what
  `bnF` renders — faithful to the theorem, a simplification of the paper. Channel-LN
  (reduce over the C axis per spatial location) would be a *different* reduction + a new
  fragment + a new proof; out of scope unless you want it (flag if the user asks for it).
- **Watch the GAP-of-normalization degeneracy** (ch7/ch8 lesson): per-example global/instance
  norm can make `GAP(act(LN))` input-invariant. ConvNeXt's head is `GAP → LN → dense` (LN
  AFTER pool) and blocks carry residual skips, so the pooled features keep inter-image
  variance — but if epoch-1 sits at exactly chance (≈9.45% on imagenette), suspect a
  degenerate forward, not an lr issue (see ch8 E4→E5 note).
- **Out of scope** (the paper recipe, none needs proofs): AdamW, cosine schedule + warmup,
  weight decay, stochastic depth / drop-path, label smoothing, RandAug/Mixup, EMA,
  population-stats eval. The verified trainer is plain mean-loss SGD with batch-stat eval —
  the win is **verified codegen on native-resolution data**, not a SOTA Imagenette number.
  (ConvNeXt is lr-sensitive from scratch; expect to tune lr, like ch7's lr=0.3 lesson.)

---

## 8. Gotchas to carry over (from ch6/ch7/ch8 + this session)

- **Adding an SHlo op = 9-site lockstep** (inductive/den/skel/Raw/Tok/toToks/emitTok +
  parseStack/parseStack_toToks). `swishF`/`sigmoidF` (ch8) are the all-smooth worked
  examples for `geluF`. rfl-faithful ops stay OUT of `AuditAxioms` (roundtrip covers them).
- **depthwise weight-grad needs `batch_group_count = c`** (XLA filter-grad trick: transpose
  inp/dy → [c,B,H,W], conv → [1,c,kH,kW], reshape [c,1,kH,kW]). De-risk standalone.
- **strided backward = pad-upsample then reuse the stride-1 backward** (decimate-backward =
  `lhs_dilation`/interior-pad), NOT a raw dilated conv (off-by-one). For 4×4/s4: interior=3.
- **BS is baked into the rendered MLIR** (`tensor<32x…>`); keep BS identical across
  TestConvNeXt{Fwd,Train}.lean AND MainConvNeXtVerified.lean (use 32).
- **`%x` is the flat `[BS,150528]` FFI buffer** reshaped to `[BS,3,224,224]` inside the
  graph (see the ch8/r34 renderers). Mean-loss cotangent (÷B) so a sane lr works.
- **emitter byte-stability:** after edits, `git diff --stat verified_mlir/` must stay empty
  for the OTHER chapters' `.mlir`; the `StableHLO.lean` `#eval` block regenerates them on
  elaboration.
- **StableHLO.lean import hygiene:** it already imports LayerNorm/EfficientNet/Depthwise/
  StridedConv/PerChannelBN. ConvNeXt.lean imports Depthwise+LayerNorm — adding `geluF`'s
  `den = gelu` only needs LayerNorm (already imported). No cycle expected.

---

## 9. File map (ch9)

**Already exists (proofs — reuse, don't rebuild):**
- `LeanMlir/Proofs/LayerNorm.lean` — `gelu(_has_vjp(_correct))`, `geluScalar(Deriv)`,
  `pdiv_gelu`, `layerNormForward`, `layerNorm_has_vjp(_correct)`. **PROOF (done).**
- `LeanMlir/Proofs/ConvNeXt.lean` — `layerScale(_has_vjp)`, `convNextBlock(Body)(_has_vjp)`,
  `convNextForward`, apex `convnext_has_vjp(_correct)` / `_at(_correct)`. **PROOF (done).**
- `LeanMlir/Proofs/Depthwise.lean` / `StridedConv.lean` — depthwise + strided-conv VJPs.
- `tests/AuditAxioms.lean` — already prints the ConvNeXt/GELU/LN/layerScale headlines (157).
- `MainConvNeXtTrain.lean` (+ `convnext-tiny-train`), `tests/TestConvNext*.lean`,
  `MainAblation.lean` ch9 — **UNVERIFIED** reference (architecture + emit shapes only).

**To create (the ch9 verified codegen):**
- add `geluF`/`geluBack` to `LeanMlir/Proofs/StableHLO.lean` (+ parser cases in
  `StableHLOParse.lean`); a `geluScalarDeriv_eq` lemma in `LayerNorm.lean`.
- `tests/TestGelu.lean` (standalone iree), `tests/TestConvNeXtBlock.lean` (block + gradcheck).
- `tests/TestConvNeXtFwd.lean` + `tests/TestConvNeXtTrain.lean` (the renderers).
- `ConvNeXtLayout` in `LeanMlir/IreeRuntime.lean`.
- `MainConvNeXtVerified.lean` + lakefile `convnext-verified` (clone `efficientnet-verified`).
- `verified_mlir/convnext_{fwd,train_step}.mlir` (rendered output).
- generalize `decimateFlat` → decimate-s in `StridedConv.lean` (for the 4×4/s4 patchify).

**Reuse templates:** `tests/TestEfficientNet{Fwd,Train}.lean` + `MainEfficientNetVerified.lean`
+ `EfficientNetLayout` (closest whole-net 224² renderer/Main/layout); `tests/TestResnet34*`
+ `MainResnet34Verified` (this session's `convStem`/`convWGradStem` = the strided-stem
template to adapt for patchify); `swishF`/`swishBack` (the GELU op-pair template).
