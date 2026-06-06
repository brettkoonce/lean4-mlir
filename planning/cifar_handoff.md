# CIFAR (Chapter 5) verified-codegen — handoff

Continuation doc for starting Chapter-5 (CIFAR-10) verified-codegen. Picks up
after ch2 (linear), ch3 (MLP), ch4 (CNN) were fully closed — each **proven
semantic R4 + on-GPU training** through the real Lean→IREE-FFI→ROCm path.
Written 2026-06-06 by the session that finished ch4.

> Status in one line: **ch4 is DONE** (conv+maxpool fwd+back per-op faithful;
> whole-CNN forward AND backward theorems; `mnist-cnn-verified` trains to 98.89%
> on GPU; audit 134/134). **ch5/CIFAR has NOTHING verified yet** — the
> production `MainCifarCnnTrain` uses the *old unverified* `NetSpec.train`
> codegen. This is a from-scratch chapter, but every conv/pool/dense op it needs
> is already built and GPU-validated.

---

## ⭐ UPDATE 2026-06-06 (b) — Milestone B (BatchNorm) is DONE

**The BN demon is slain: verified BN forward + its consolidated O(N) three-term
input-VJP, GPU-trained.** What landed (committed to `main`):
- **Typed SHlo BN ops** `bnF` (forward, denotes `bnForward`) + `bnBack` (denotes
  `bn_grad_input`), full lockstep through inductive/den/skel/Raw/Tok/toToks/emitTok
  + parser round-trip. Faithfulness: `bnF_faithful` (rfl), `bnBack_faithful`
  (= pdiv-Jacobian via `bn_input_grad_correct`, under 0<ε). emitTok copies
  `IRPrint.renderLN`/`renderLNBack`.
- **Semantics** (CifarCNN.lean): `cifarCnnBnForward` (conv→BN→relu ×4) + conditional
  `cifarCnnBn_has_vjp_at(_correct)` via `convBnRelu_has_vjp_at`.
- **Renderer**: `cifarBnFwdGraph(_faithful)`, `cifarBnFwdModuleV`, `cifarBnTrainStepText`
  (BN fwd=renderLN, back=renderLNBack 3-term, scalar dγ/dβ). `#eval` →
  `verified_mlir/cifar_bn_{fwd,train_step}.mlir`. Both iree-compile on rocm.
- **`CifarBnLayout`** (22 params, γ/β are rank-0 `#[]` scalars — the FFI shim's
  empty-product gives size 1; verified to flow end-to-end). `MainCifarBnVerified.lean`
  + `cifar-bn-verified` exe. Audit **140/140** 3-axiom-clean (+3 BN ℝ-headlines).

**⚠️ KEY MODELING CHOICE + result.** The proven `bnForward` is a **per-example
GLOBAL normalization** (reduce μ/var over the whole oc·H·W feature vec, **scalar**
γ/β), NOT per-channel batch-BN. This is what keeps the per-example "D1 shortcut"
intact and is faithfully verifiable (it's literally `IRPrint.renderLN` = LayerNorm-
shaped). Result: **57.1% @ epoch 10 (climbing, healthy)** — *below* no-BN's 66%.
That's faithful, not a bug: global normalization + scalar affine is very aggressive
(discards per-channel scale, can't restore it), so it slows early learning. The hard
chapter content (BN's tricky O(N) backward, verified + on GPU) is the win; a true
accuracy "lift" would need **per-channel** BN (apply `bnForward` per channel-slice
with per-channel γ/β — a real extension: a per-channel forward + chained VJP, more
proof work) and/or the init/lr-robustness framing from update (a) below.

**Next ideas if continuing:** (1) per-channel BN for a real lift; (2) the discharged
`Tiny` BN instance (non-vacuity witness — deferred; needs two exact-istd BN stages
with injectivity through both pools, harder than the no-BN `Tiny`); (3) variant C
(verified Adam) for the 72% number. §§2–7 below are the original (pre-A) plan.

---

## ⭐ UPDATE 2026-06-06 — Milestone A is DONE (and a surprise)

**Milestone A (no-BN SGD CIFAR) is complete and committed.** What landed:
- `LeanMlir/Proofs/CifarCNN.lean` — `cifarCnnForward` (4 conv, 2 pool, 3 dense),
  the conditional whole-network VJP `cifarCnn_has_vjp_at`(`_correct`), and an
  **unconditional** discharged tiny instance `Tiny.cifarTinyCnn_has_vjp_correct`
  (the Ch5-specific crux: the SECOND pool's no-tie needs the FIRST pool's output
  positionally injective — the four 2×2 window maxima `6,8,14,16` are distinct,
  proved by folding the real `max`es through `Nat.cast_max` + `omega`).
- `StableHLO.lean` — `cifarFwdGraph`(`_faithful`), `cifarFwdModuleV`,
  `cifarTrainStepText` (2-stage shape-parametric scale-up of `cnnTrainStepText`);
  `#eval` → `verified_mlir/cifar_{fwd,train_step}.mlir`. Both **iree-compile on
  rocm** (gfx1100) cleanly. Older verified_mlir bytes unchanged (no emitter drift).
- `MainCifarVerified.lean` + `cifar-verified` exe (lakefile). Trains on GPU.
- Audit **137/137** three-axiom-clean (was 134; +3 CIFAR ℝ-headlines).

**⚠️ THE SURPRISE — no-BN SGD does NOT fail at ~10%; it trains to ~66%.**
With textbook He init (conv fanin = ic·kH·kW) and the verified trainers' mean-loss
`lr = 0.1/128`, `cifar-verified` reaches **peak 66.9% @ epoch 8** (≈65.6% @ ep 10),
no aug/schedule. The Ch5 "no-BN fails (loss 2.30, ~10%)" prediction in §1 below did
NOT reproduce. The failure is **not intrinsic** — it's lr-scale / init sensitive.
Leading hypothesis: the production `MainCifarCnnTrain` lr=0.1 is on the *summed*-batch
gradient (≈128× larger here) → divergence → stuck at random; the mean-loss convention
every verified trainer uses is stable. (User will dig into the exact diff later.)
**Implication for Milestone B:** the BN "lift" is an init/lr-robustness story, not a
10%→trains cliff — frame the ablation honestly (e.g. show BN lets a *larger* lr or a
worse init still train, rather than claiming no-BN can't learn at all). Everything in
§§2–7 below (the BN op plan, file map, gotchas) still stands for Milestone B.

---

## 0. The pattern (what we build, every chapter)

Typed, proof-carrying StableHLO emitter. Per op-graph:

```
SHlo a ──skel──▶ Raw ──toToks──▶ List Tok ──serializeToks(emitTok)──▶ StableHLO text
   │ den (⟦·⟧ₐ, ℝ)               ▲ parse (StableHLOParse.lean)
   ▼                              │
proven Mathlib fderiv             └ parse (toToks (skel a)) = skel a  [round-trip]
```

- **Semantic R4:** `den (emit g) = <proven Mathlib fderiv quantity>`.
- **Syntactic R4:** `parse (toToks (skel a)) = skel a` (structural induction).
- **The deliverable each chapter:** (a) a whole-net forward graph + faithfulness,
  (b) a full SGD train-step renderer, (c) a `*-verified` trainer that trains on
  GPU through the real FFI, (d) optionally the whole-chain backward theorem.

Ladder: ch2 linear 92.10%, ch3 MLP 97.86%, ch4 CNN 98.89% — all proven (fwd+back)
AND GPU-trained on the proof-rendered StableHLO. ch5 is the CIFAR / **BatchNorm** tier.

---

## 1. ⚠️ FIRST DECISION — which CIFAR variant? (changes scope a LOT)

The CIFAR net (`MainCifarCnnTrain.cifarCnn`, no-BN) is:
`conv 3→32 ·relu → conv 32→32 ·relu → maxpool → conv 32→64 ·relu →
 conv 64→64 ·relu → maxpool → flatten(4096) → dense 4096→512 ·relu →
 dense 512→512 ·relu → dense 512→10`  (32×32 RGB, 10 classes, 2 pools 32→16→8).

**Chapter 5's whole pedagogical point** (per the docstring in MainCifarCnnTrain.lean):
this no-BN spec under **plain SGD lr=0.1 FAILS to train** — loss sticks at
log(10)≈2.302 (random). That failure is what motivates BatchNorm. The production
exe only reaches ~72% by switching to **Adam lr=1e-3 + cosine + warmup + augment**.

So a *plain-SGD verified CIFAR train step* (the cheap reuse of ch4's `cnnTrainStepText`)
will faithfully reproduce the **failure** (~10%). That's a legitimate — even elegant —
verified result, but it's not "72%". The three honest targets:

| Variant | Verified result | New emitter work | Effort |
|---|---|---|---|
| **A. no-BN + SGD** | reproduces Ch5 failure (~10%, loss 2.30) | none (reuse ch4 ops) | **low** |
| **B. BN + SGD (the ablation)** | no-BN fails ↔ BN trains — the chapter's lift | **add verified BN fwd+back ops** | **high** |
| **C. no-BN + Adam** | ~72% | add a verified **Adam** train step | high |

**Recommended:** **B** — it IS Chapter 5. Do A first as a cheap milestone (it
proves the conv-stack scale-up end-to-end and literally renders the failure the
chapter is about), then add BN to get the lift. C (Adam) is a separate axis
(optimizer, not normalization) — lower pedagogical priority for ch5.

The good news for B: the **BN semantic proofs already exist** (see §3) — only the
*typed emitter ops* are missing.

---

## 2. Repo / file map

Repo root: `/home/skoonce/lean/proof_verify_demo/lean4-mlir`

| file | holds |
|---|---|
| `LeanMlir/Proofs/StableHLO.lean` | typed AST `SHlo`, `den`, `pretty`, all faithfulness theorems, the `*Text`/`*ModuleV` renderers, build-time `#eval`s → `verified_mlir/`. **ch4 added** `cnnFwdGraph(_faithful)`, `cnnTrainStepText`, `cnnBackGraph(_faithful)`, conv/maxpool fwd+back ops + `emitTok`. **Mirror these for CIFAR.** |
| `LeanMlir/Proofs/StableHLOParse.lean` | `parse` + round-trip theorems (extend for any NEW ops, e.g. BN) |
| `LeanMlir/Proofs/CNN.lean` | `conv2d_has_vjp3`, `maxPoolFlat_has_vjp_at`, **BN: `bnForward`, `convBnRelu_has_vjp_at` (≈L869), `convBn_has_vjp`, `resblock_body_has_vjp_at`**, `conv2d_weight_grad*` |
| `LeanMlir/Proofs/MnistCNN.lean` | **the template to copy**: `mnistCnnNoBnForward`, `mnistCnnNoBn_has_vjp_at` (vjp_comp_at chain), `Micro.*` (smoothness discharged). Build `cifarCnnForward` + `cifarCnn_has_vjp_at` here, same style (just longer: 4 conv, 2 pool, 3 dense). |
| `LeanMlir/Proofs/Tensor.lean` | `HasVJPAt`, `vjp_comp_at`, `Tensor3`, the flatten bridges |
| `LeanMlir/IreeRuntime.lean` | `mlpTrainStepV` (params-general FFI — reuse), **`CifarLayout` (shapesBA + xShape) ALREADY EXISTS** for the 4-conv arch (L218), `trainStepAdamF32` (Adam, for variant C) |
| `LeanMlir/F32Array.lean` | `cifarBatch raw start count` (loads CIFAR `.bin`, normalizes [0,1]), heInit/const/concat/argmax10 |
| `LeanMlir/Proofs/IRPrint.lean` | production-mirror string builders incl. `renderLN`/`renderBNParamGrad` (BN fwd/back StableHLO to copy), `resnetTrainStepModule` (a BN train step) |
| `MainMnistCnnVerified.lean` | **the trainer to copy** (reads verified_mlir, compiles, drives via FFI) |

`CifarLayout.paramShapes` (IreeRuntime.lean:218, no-BN, already there):
```
#[32,3,3,3],#[32], #[32,32,3,3],#[32], #[64,32,3,3],#[64], #[64,64,3,3],#[64],
#[4096,512],#[512], #[512,512],#[512], #[512,10],#[10]
```
(BN variant would interleave γ/β scalar params — extend this.)

---

## 3. What's REUSABLE (a lot)

- **Every conv-stack op is built + GPU-validated** (ch4): `flatConvF`/`maxPoolF`
  (fwd), `convBack`/`maxPoolBack` (back), conv weight-grad (transpose trick),
  dense/relu/dotOut/selectPos. CIFAR's 2 maxpools = 2 `select_and_scatter`
  (already known-good on rocm). So the conv/pool/dense rendering is **pure
  scale-up** of `cnnTrainStepText`.
- **FFI:** `mlpTrainStepV` is params-general (4-D kernels, one-hot in C) — reuse
  for no-BN SGD exactly as ch4 did. `CifarLayout.shapesBA`/`xShape` exist.
- **Data:** CIFAR-10 at `…/lean4-jax/data/cifar-10/` (`data_batch_{1..5}.bin`,
  10000×3073 bytes each; **confirm `test_batch.bin` for eval** — wasn't in the
  listing, may need to grab it). Loader `F32.cifarBatch`.
- **BN semantics (for variant B) are PROVEN:** `bnForward`, its consolidated
  3-term backward (`bn_back_bridge`), `convBnRelu_has_vjp_at`. The production BN
  StableHLO is in `IRPrint.renderLN`/`renderBNParamGrad`/`resnetTrainStepModule`.
  **What's missing is only the typed `SHlo` BN ops + `emitTok` + faithfulness.**

---

## 4. Work breakdown (recommended order)

### Milestone A — no-BN SGD CIFAR (cheap; reproduces the Ch5 failure)
1. **Semantic:** in `MnistCNN.lean` (or a new `CifarCNN.lean`), define
   `cifarCnnForward` (the 10-layer compose above) + `cifarCnn_has_vjp_at` by
   chaining `convRelu_has_vjp_at`/`denseRelu_has_vjp_at`/`maxPoolFlat_has_vjp_at`
   through `vjp_comp_at` — **exactly** like `mnistCnnNoBn_has_vjp_at`, just
   longer and with TWO maxpool steps (each needs the `flatten∘unflatten` align +
   the `maxPoolFlat_has_vjp_at'`/`hasVJPAt_backward_det` trick from ch4's A2c).
2. **Renderer:** `cifarFwdGraph(_faithful)` + `cifarTrainStepText` in
   StableHLO.lean — scale up `cnnTrainStepText` (4 convBack, 4 conv-wgrad, 2
   select_and_scatter, 3 dense). `#eval` → `verified_mlir/cifar_{fwd,train_step}.mlir`.
3. **B0-style de-risk:** iree-compile the train step on rocm first (it's big +
   slow). 2 pools/4 convs — watch compile time.
4. **Trainer:** `MainCifarVerified.lean` (copy `MainMnistCnnVerified`): use
   `CifarLayout`, `F32.cifarBatch`, He-init (conv fanin = ic·kH·kW), `mlpTrainStepV`
   with d0=3072, nClasses=10. **Expect ~10% / loss 2.30 under SGD lr=0.1** — that
   IS the result (the BN-motivating failure, now in verified codegen). Try a
   smaller lr (the `cifar-nobn-sgd002` cell tops ~72% with tuned LR — verify).
5. Audit (add an ℝ-headline), commit.

### Milestone B — add BatchNorm (the chapter's real content)
6. **New `SHlo` ops:** `bnF` (forward: reduce mean/var → normalize → γ·+β) and
   `bnBack` (the 3-term consolidated backward). Lockstep update inductive/`den`/
   `skel`/`Raw`/`Tok`/`toToks`/`emitTok` + `parseStack`(+round-trip). `den` =
   `bnForward`/`bn_back_bridge`. Faithfulness via the existing CNN.lean BN proofs.
   StableHLO text: copy `IRPrint.renderLN`/`renderBNParamGrad` (reduce/broadcast/
   rsqrt). **De-risk the BN ops on rocm early.**
7. Insert BN after each conv (use `convBnRelu_has_vjp_at` for the semantic chain),
   re-render train step, extend `CifarLayout` with γ/β params, retrain. **Now SGD
   should train** (the lift) — show the no-BN-fails ↔ BN-works ablation on GPU.

### (Optional C — verified Adam) only if chasing the 72% number without BN.

---

## 5. Build / audit / run (same as ch4)

```bash
cd /home/skoonce/lean/proof_verify_demo/lean4-mlir
lake build Proofs                       # regenerates verified_mlir/ via #eval
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c 'depends on axioms: \[propext, Classical.choice, Quot.sound\]$'
#   must equal total 'depends on axioms:' lines (134 now; add a CIFAR ℝ-headline → 135+)

export PATH="$PWD/.venv/bin:$PATH"; export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
export IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0
DATA=/home/skoonce/lean/claude_max/lean4-jax/data            # cifar-10/ lives here
.lake/build/bin/mnist-cnn-verified "$DATA"                   # ch4 sanity → 98.89%
# .lake/build/bin/cifar-verified "$DATA"                     # the new exe
.venv/bin/iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx1100 \
  --iree-codegen-llvmgpu-use-reduction-vector-distribution=false  M.mlir -o /tmp/x.vmfb
```
`ffi/libiree_ffi.so` is built (gitignored); rebuild per `IREE_BUILD.md` §4 if missing.
`Vec`/`Mat` are `Fin → ℝ` (no runtime rep) → trainers read `verified_mlir/*.mlir`
(materialized at build time), they can't call the renderers at runtime.

---

## 6. Gotchas (carried from ch4 + CIFAR-specific)

- Adding an `SHlo` op = lockstep update of inductive/`den`/`skel`/`Raw`/`Tok`/
  `toToks`/`emitTok` (StableHLO.lean) + `parseStack`/`parseStack_toToks`
  (StableHLOParse.lean). Miss one → non-exhaustive-match.
- **Maxpool VJP cast (hit twice in CIFAR):** `maxPoolFlat_has_vjp_at`'s point is
  `flatten(unflatten v)`; composing needs the `set …; rw [← hpt]` transport. For
  the backward theorem use the **A2c pattern**: `maxPoolFlat_has_vjp_at'` (cast-free
  witness at a raw flat point) + `hasVJPAt_backward_det` (VJP-backward uniqueness)
  + `rfl`. (Both now in StableHLO.lean.)
- Audit exact-match: audit an **ℝ-carrying** headline (a `*_faithful`), not a
  `[propext]`-only lemma, or the `=[propext, Classical.choice, Quot.sound]` grep fails.
- Doc comments (`/-- -/`) can't precede `set_option … in` (or `open … in`); use
  `--` line comments there. (Bit me on `cnnBackGraph_faithful`.)
- After editing the emitter, `git diff --stat verified_mlir/` must be empty for
  the linear/MLP/CNN files (byte-stable) — non-empty = emitter drift.
- `#eval … "verified_mlir/…"` must `IO.FS.createDirAll "verified_mlir"` first.
- CIFAR conv train step compile is SLOW (4 convs + 4 conv-wgrads + 2
  select_and_scatter) — compile in background, expect minutes.
- **Reality check the accuracy:** no-BN SGD lr=0.1 ≈ 10% is the *expected* Ch5
  failure, not a bug. Don't "fix" it by silently switching optimizers — that
  erases the chapter's point. Make the variant explicit.

---

## 7. Commit / push status (at handoff time)

`origin/main` is at `ed0c69c` (ch4 fully pushed: B-train + A2c). Working tree clean.
Workflow: **commit straight to `main`; NEVER push without explicit per-push permission.**
`mnist-cnn-verified` (98.89%) and the ch4 theorems are the proven baseline to mirror.
