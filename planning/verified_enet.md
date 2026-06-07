# Verified EfficientNet (Chapter 8) — handoff / completion report

Continuation doc for the **EfficientNet** chapter of the verified-codegen ladder. ch8
brings EfficientNet (MBConv = inverted-residual + **squeeze-excite** + **swish** +
**batch-norm**) into the same "render via the proof-carrying StableHLO emitter + GPU-train
on the rendered text" line as ch6 (ResNet-34) and ch7 (MobileNetV2).

> **Status in one line:** ch8 is **DONE through E6** — a **faithful EfficientNet-B0**
> (real `[t,c,n,s,k]` config, all-swish, batch-norm, SE) is rendered per-op from the
> proven-faithful emitter, iree-compiles on ROCm, and GPU-trains on **Imagenette 224²**
> (B0's native resolution). The whole-network VJP `Proofs.efficientnet_has_vjp(_correct)`
> was already proven/unconditional; E1–E6 added the **verified codegen**. Audit **157/157**
> 3-axiom-clean (E5a added `bnBatchTensor4_grad_input_correct`). All commits LOCAL, **never
> push without explicit per-push permission.**

Updated 2026-06-07 by the session that built E1–E6. `origin/main` is behind by ch6 + ch7 +
ch8. Read §§0–2 first; rest is reference + honest framing + future work.

---

## 0. The pattern (unchanged from every chapter)

Typed, proof-carrying StableHLO. Two faithfulness halves per op:
- **Semantic R4:** `den (graph) = <proven Mathlib fderiv quantity>`.
- **Syntactic R4:** `parse (toToks (skel a)) = skel a` (round-trip, for the typed AST ops).

The **trainer** is hand-rendered batched StableHLO string fragments (ch7/ch8 style), each
fragment faithful PER-OP to a proven function — NOT a single `den(trainStep)` theorem.
Validated by training (loss/accuracy moving), like ch6/ch7's `*-verified` exes.

---

## 1. ⭐ WHAT WAS BUILT — E1–E6 (all committed, local)

| Step | Commit | What |
|---|---|---|
| E1 | `de4f3a7` | **swish** SHlo op pair `swishF`/`swishBack` (StableHLO.lean, full lockstep). All-smooth, GLOBAL `swish_has_vjp`. |
| E2 | `bd43cc2` | **sigmoid** op pair `sigmoidF`/`sigmoidBack` (the SE gate nonlinearity). imports EfficientNet (sigmoid lives there). |
| E3 | `fae03cd` | **squeeze-excite** module renderer `seFwd`/`seBack` (tests/TestSE.lean) — fan-out gate, 2-path backward + dense W/b grads. iree + **finite-difference GRADCHECK** all pass. |
| E4 | `b277439` | **MBConv renderer** + layout + Main + `efficientnet-verified` exe (reduced CIFAR config). [head was a relu6 STOPGAP — see E5] |
| E5a | `872ba25` | **batch-norm VJP** `bnBatchTensor4(_grad_input)(_correct)` (PerChannelBN.lean). Audit 156→**157**. |
| E5b | `bd6b5a4` | batch-norm codegen → **genuinely ALL-SWISH** (relu6 head removed). CIFAR 48.7%→**65.5%**. |
| E6 | `3be94b0` | **faithful EfficientNet-B0** `[t,c,n,s,k]` on **Imagenette 224²** (variable kernel 3×3/5×5, MBConv1 no-expand, stem stride-2, 224→7, 262 params). |

### The proofs (were already done before E1; in the audited 157)
`efficientnet_has_vjp(_correct)` (UNCONDITIONAL, all-smooth), `swish_has_vjp_correct`,
`sigmoid_has_vjp`, `seBlock_has_vjp_correct`, `seGate_has_vjp`, `broadcastFlat_has_vjp`,
`mbconvBody_has_vjp`. **E5a added** `bnBatchTensor4_grad_input_correct` — the only new proof
of the chapter, and it was mostly REUSE (see §2).

---

## 2. ⭐ KEY FINDINGS (read before touching anything)

### Batch-norm is what makes EfficientNet all-swish work (E5 — the crux)
The proven BN (carried since ch5) is **per-example INSTANCE-norm**. Instance-norm
standardizes every example identically, so `GAP(act(BN)) ≈ a per-image-INVARIANT constant`
for any *smooth* `act`. A **swish head froze the net at EXACTLY chance** (10.0±0.05% at BOTH
lr=0.3 AND lr=0.1 — a degenerate FORWARD, not an lr washout). E4's stopgap was a **relu6
head** (hard clamp breaks the degeneracy). **E5 fixed it properly with BATCH-norm** —
batch-norm keeps inter-image variance in the pooled features, so swish works at the final
GAP. The net is now **genuinely all-swish; relu6 is gone**. (DON'T reintroduce a relu6 head.)

### The batch-norm proof was mostly REUSE
Batch-norm per channel = normalizing each channel over its `N·H·W` (batch+spatial) cells =
**exactly `bnPerChannelFlat` with `m = N·h·w`** (the existing per-channel instance-norm VJP,
group enlarged from one example's spatial cells to the whole batch). The only NEW content
was a layout bridge `[N,C,H,W]↔[C,N·H·W]` (`bnchwFwd/Back` — a TRANSPOSE-reindex, vs the
instance-norm bridge's pure re-association), mirroring `reassocFwd/Back` + `bnPerChannelTensor3`
line-for-line. `bnBatch`/`bnBatchBack` fragments = the instance-norm fragments with the
Σ-reductions over `[0,2,3]` instead of `[2,3]`.

### B0-on-Imagenette is the right target (not CIFAR)
B0 was designed for 224×224 ImageNet (stem stride-2, 5 downsamples 224→7). Forced onto 32²
CIFAR it over-downsamples (5×5 convs on 4×4 maps — awkward). **Imagenette** (10-class
ImageNet subset, `<DATA>/imagenette/{train.bin@256, val.bin@224}`) is B0's native scale.
E6 trains on it at 224² with the real B0 spatial flow.

---

## 3. The EfficientNet-B0 architecture (E6, what's rendered)

Stem 3×3 **stride-2** conv (3→32) → BN → swish (224→112). Then 7 stages `[t,c,n,s,k]`:

| stage | expand t | ch c | repeats n | stride s | kernel k | spatial |
|---|---|---|---|---|---|---|
| s1 | 1 (MBConv1, **no expand conv**) | 16 | 1 | 1 | 3 | @112 |
| s2 | 6 | 24 | 2 | 2 | 3 | 112→56 |
| s3 | 6 | 40 | 2 | 2 | 5 | 56→28 |
| s4 | 6 | 80 | 3 | 2 | 3 | 28→14 |
| s5 | 6 | 112 | 3 | 1 | 5 | @14 |
| s6 | 6 | 192 | 4 | 2 | 5 | 14→7 |
| s7 | 6 | 320 | 1 | 1 | 3 | @7 |

16 MBConv layers; SE ratio 0.25 of block-input ch (r = ic/4) in every block. Head 1×1 conv
(320→1280) → BN → swish → GAP → dense(1280→10). 262 params / 4.04M floats (B0 range).

MBConv = expand 1×1→BN→swish (SKIPPED if t=1) → depthwise k×k (stride 1/2)→BN→swish → SE
(squeeze C→r→C, sigmoid, ×main) → project 1×1→BN; + residual iff s=1 ∧ ic=oc.

---

## 4. Build / audit / run

```bash
cd /home/skoonce/lean/proof_verify_demo/lean4-mlir
lake build Proofs
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c 'depends on axioms: \[propext, Classical.choice, Quot.sound\]$'
#   must equal total 'depends on axioms:' lines (157)

export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
export IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0
DATA=/home/skoonce/lean/claude_max/lean4-jax/data        # imagenette/ + cifar-10/ live here

# render + iree-compile the B0 renderers (writes verified_mlir/efficientnet_{fwd,train_step}.mlir):
lake env lean tests/TestEfficientNetFwd.lean             # @efficientnet_fwd  iree OK (138KB)
lake env lean tests/TestEfficientNetTrain.lean           # @efficientnet_train_step iree OK (410KB, 262 params)
lake env lean tests/TestSE.lean                          # SE module fwd+back iree OK (gradcheck: /tmp/se_gradcheck.py)

# GPU-train the verified B0 on Imagenette 224²:
lake build efficientnet-verified
.lake/build/bin/efficientnet-verified "$DATA"            # loads imagenette, BS=32, lr=0.1
```
- **Running a trainer:** Bash `run_in_background: true`; do NOT `nohup … &`. Flushes per epoch.
- **Imagenette is slow:** 224² B0 ≈ 49× CIFAR compute; ~5–8 min/epoch at BS=32. iree-compile of the
  410KB train step takes a couple minutes.
- The **reduced-CIFAR all-swish net** (E5, commit `bd6b5a4`) is the fast iteration target (65.5%, ~30s/epoch).

---

## 5. RESULTS + honest framing (don't oversell)

- **CONFIRMED:** fwd (138KB) + train (410KB, 262 params) iree-compile ROCm gfx1100; Imagenette
  loads (9469 train / 3925 val); `efficientnet-verified` GPU-trains without OOM. Audit 157/157;
  resnet34/mobilenetv2 `.mlir` byte-stable. The per-op faithfulness + E3 SE gradcheck + E5a
  batch-norm proof + the E5 CIFAR all-swish result (65.5%) establish the gradients are correct.
- **Training number on Imagenette:** B0 trained FROM SCRATCH on only 9469 images with **plain SGD,
  no augmentation/schedule/dropout/EMA** is a hard optimization (B0 was designed for ImageNet-1.3M
  + the full recipe). Epoch 1 ≈ chance (9.45%); see the live run for the climb. The chapter win is
  a **verified, paper-faithful B0 architecture GPU-training on native-resolution data**, NOT a SOTA
  Imagenette number.
- **The win is correctness/codegen**, faithful PER-OP (swish/sigmoid/SE/batch-norm/depthwise/conv/
  residual/GAP/dense), validated by training — not a single `den(trainStep)` theorem.

### Explicitly OUT OF SCOPE (user: "not ema/etc")
Paper training recipe — RMSProp, LR warmup+exp-decay, weight decay, dropout/stochastic-depth,
weight EMA; **population-stats eval** (we use batch stats at eval, BS=32); ImageNet-1000 (we use
Imagenette-10). None of these need new proofs (the optimizer/eval-stats aren't part of the VJP);
they're "normal deep-learning engineering."

---

## 6. File map (ch8)

- `LeanMlir/Proofs/LayerNorm.lean` — `swish(_has_vjp(_correct))`. **PROOF (pre-existing).**
- `LeanMlir/Proofs/SE.lean` — `seBlock`, `elemwiseProduct(_has_vjp)`, `seBlock_has_vjp(_correct)`. **PROOF.**
- `LeanMlir/Proofs/EfficientNet.lean` — `sigmoid(_has_vjp)`, `broadcastFlat`, `seGate`, `mbconvBody`,
  apex `efficientnet_has_vjp(_correct)` (UNCONDITIONAL). **PROOF.**
- `LeanMlir/Proofs/PerChannelBN.lean` — **E5a:** `bnchwFwd/Back` bridge + `bnBatchTensor4(_has_vjp)
  (_correct)` + `bnBatchTensor4_grad_input(_correct)` (the audited batch-norm headline).
- `LeanMlir/Proofs/StableHLO.lean` — **E1/E2:** `swishF`/`swishBack`, `sigmoidF`/`sigmoidBack` op pairs.
- `LeanMlir/Proofs/StableHLOParse.lean` — parser cases for the 4 new ops.
- `tests/TestSwish.lean` / `tests/TestSigmoid.lean` / `tests/TestSE.lean` — E1/E2/E3 standalone iree.
- `tests/TestEfficientNet{Fwd,Train}.lean` — **E6:** the B0 renderer (variable kernel, MBConv1
  no-expand, B0 stage→block generator, 224² stem-stride-2). `bnBatch`/`bnBatchBack` fragments (E5b).
- `LeanMlir/IreeRuntime.lean` — `EfficientNetLayout` (B0 specs, 262 params, xShape 224²).
- `MainEfficientNetVerified.lean` + lakefile `efficientnet-verified` — loads Imagenette (train@256
  centerCrop→224, val@224), BS=32, SGD lr=0.1. (Old unverified `efficientnet-train` left alone.)
- `verified_mlir/efficientnet_{fwd,train_step}.mlir` — E6 rendered (224²).

---

## 7. Future work (if continuing — all codegen/infra, NO new proofs except where noted)

1. **Get a real Imagenette number** — add the paper training recipe (the things scoped out): a LR
   schedule (pass lr as a train-step input), weight decay (`+λθ` in the grad), RMSProp/momentum
   (per-param state), dropout (in-graph RNG), weight EMA + **population-stats eval** (running mean/var
   as EMA "params" — the genuinely-fiddly bit). None need proofs. Likely the biggest accuracy lever
   alongside more data.
2. **Push** ch6 + ch7 + ch8 to origin (only with explicit per-push permission).
3. **ImageNet-1000** — bigger data pipeline + compute; FC 1280→1000.
4. The reduced-CIFAR all-swish net (E5) remains available for fast iteration.
