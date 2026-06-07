# Verified ResNet-34 (Chapter 6) — handoff

Continuation doc for the ResNet chapter of the verified-codegen ladder. Picks up
after **Milestone A (verified ResNet-*style* net) and Milestone B1–B6c (the whole
verified ResNet-34 architecture) are DONE and committed locally** (not pushed).

> **Status in one line:** the whole-network conditional VJP of a real 34-layer
> ResNet — **`Proofs.resnet34_has_vjp_at`** — is PROVEN and **audit-clean
> (149/149, 3-axiom)**, alongside every primitive it needs (strided conv +
> in/weight VJP + render, deep-block chain, all four block types). What remains
> is the *trained model*: a concrete-instance discharge (non-vacuity), per-channel
> BN, and the full GPU render+train. Read §§0–3 first; the rest is reference.

Written 2026-06-07 by the session that did ch6-A + ch6-B1…B6c. `origin/main` is
behind by **10 local commits** (`5f09e93`…`5df5743`); commit-to-main is fine,
**never push without explicit per-push permission**.

---

## 0. The pattern (unchanged from every chapter)

Typed, proof-carrying StableHLO emitter. Per op-graph:
`SHlo a ─skel→ Raw ─toToks→ [Tok] ─emitTok→ StableHLO text`, two halves:
- **Semantic R4:** `den (graph) = <proven Mathlib fderiv quantity>` (faithfulness).
- **Syntactic R4:** `parse (toToks (skel a)) = skel a` (round-trip, `roundtrip`).

Deliverables per chapter: whole-net forward graph + faithfulness, full SGD
train-step renderer, a `*-verified` exe that GPU-trains on the rendered text,
audit ℝ-headline(s) 3-axiom-clean.

Ladder: ch2 92.10%, ch3 97.86%, ch4 98.89%, ch5-A 66%, ch5-B 57%,
**ch6-A (ResNet-style) 60.8%**. ch6-B = real ResNet-34 (architecture verified; not
yet trained).

---

## 1. ⭐ WHAT IS DONE (committed, all local, not pushed)

### Milestone A — verified ResNet-*style* net, GPU-trained (`5f09e93`, `2818d12`)
The net `cnn_has_vjp_at` already covers (stem→maxpool→identity-block→
projection-block→GAP→dense), rendered + trained:
- SHlo ops **`addV`** (residual add; binary like `.sub`; skip reuses the
  block-input *subtree* in both operands → tree-safe, no DAG) and **`gapF`**
  (global-avg-pool → `globalAvgPoolFlat`). `den_addV`/`gapF_faithful`.
- **`resnetFwdGraph` + `resnetFwdGraph_faithful`** (denotes `cnnForward`).
- **`resnetTrainStepText`** — full SGD train step; residual fan-IN backward
  (`dx = dF + dskip`, a `stablehlo.add`) + GAP backward (broadcast `dy/(h·w)`).
- `resnetFwdModuleV`, `ResnetLayout` (26 params, IreeRuntime.lean),
  `MainResnetVerified.lean`, `resnet-verified` exe.
- GPU CIFAR-10, 10 ep: 27.6→…→**60.8%** (~50s/ep). Both MLIRs iree-compile rocm.

### Milestone B1–B6c — the verified ResNet-34 ARCHITECTURE
All in **`LeanMlir/Proofs/StridedConv.lean`** (B1–B2) and
**`LeanMlir/Proofs/ResNet34.lean`** (B4–B6c); B3 ops in `StableHLO.lean`.

| Step | Commit | What |
|---|---|---|
| B1 | `52f433a` | **strided conv input-VJP**. `decimateFlat`(+diff+vjp via `pdiv_reindex`), `flatConvStride2 = decimateFlat ∘ flatConv`(+diff), `flatConvStride2_has_vjp(_correct)`. |
| B2 | `b5053a0` | **strided conv weight-VJP**. `conv2d_weight_differentiable`, `flatConvStride2_weight_grad_has_vjp(_correct)`. |
| B3 | `04458b3` | **strided conv SHlo op pair**: `flatConvStridedF` (fwd, `window_strides=[2,2]`) + `convStridedBack` (input-VJP). Both iree-compile rocm. |
| B4 | `c2bd207` | **deep chain (global)**: `chainComp`, `vjp_chain(_correct)`. |
| B5 | `96a2c88` | **strided downsample block**: `convBnStrided_has_vjp`, `convBnReluStrided_has_vjp_at(_correct)`. |
| B6a | `3d71730` | **strided residual-proj block**: `resblock_bodyStrided_has_vjp_at`, `rblkPStrided_has_vjp_at(_correct)`. |
| B6b | `d9f5b82` | **stage combinators (_at)**: `ChainData`/`chain_vjp_diff_at`/`vjp_chain_at`, `resStage_has_vjp_at`. |
| B6c | `5df5743` | **WHOLE NET**: `resnet34_has_vjp_at` (+`vjp_comp_diff_at`). |

**THE KEY IDENTITY (B1):** stride-2 SAME conv = `decimate2 ∘ (stride-1 SAME conv)`
(both read `x_pad[2·hi+kh−pH, …]`). So the strided VJPs **reuse** the existing
800-line `conv2d_has_vjp3` + `conv2d_weight_grad_has_vjp` via `vjp_comp`; the only
new proof is `decimateFlat_has_vjp` (a `reindexCLM`).

**THE WHOLE NET (B6c):** `dense ∘ GAP ∘ stage₄ ∘ stage₃ ∘ stage₂ ∘ stage₁ ∘
maxpool ∘ stem`, each downsampling `stageᵢ = (identity-block chain) ∘ downsampleᵢ`.
The 3+4+6+3 = 16 basic blocks are carried as the **`idsᵢ` lists** (depth =
`List.length`, not 100 weight args). Parametric over the component functions +
per-component `(HasVJPAt, DifferentiableAt)` / `ChainData` at the running
activations; folded from `vjp_comp_at` + `vjp_chain_at`.

---

## 2. ⭐ WHAT REMAINS (for a TRAINED ResNet-34 — all large, not started)

In suggested order:

### B7 — concrete-instance discharge (non-vacuity)  [proof; "high" but mechanical]
`resnet34_has_vjp_at` is *conditional* (smoothness/no-tie hypotheses, parametric
over abstract `stem`/`down`/`ids`). Build an **unconditional witness** at concrete
CIFAR dims, exactly as **`CnnConcrete.cnnConcrete_has_vjp_correct`** does for the
2-block net (CNN.lean):
- Instantiate `stem := convBnReluStrided` closure, `downᵢ := rblkPStrided`
  closures, `idsᵢ := [rblk, …]` lists (the identity-block closures), `gap :=
  globalAvgPoolFlat`, `dense := dense`, `mp := maxPoolFlat`.
- Provide each `(HasVJPAt, DifferentiableAt)` pair from B5/B6a/CNN
  (`convBnReluStrided_has_vjp_at`, `rblkPStrided_has_vjp_at`, `resblock_has_vjp_at`
  for `rblk`) and each `ChainData` (a nested `PProd` of the per-block witnesses).
- Discharge every smoothness/no-tie hyp with a concrete weight choice (γ=0
  resblocks + exact-istd BN + injective-stem maxpool no-ties — copy the
  `CnnConcrete` tricks: `Nat.cast_max` + `omega` for the maxpool distinct-maxima).
- Headline: `resnet34Concrete_has_vjp_correct`, add to AuditAxioms (→150).

### B8 — per-channel BatchNorm  [proof + render; genuinely new]
Current BN is per-example GLOBAL scalar γ/β (the `bnForward` ch5 used). Real
ResNet wants **per-channel** BN: apply `bnForward (h·w) ε (γ c) (β c)` to each
channel-slice, γ/β `Vec oc`. The VJP is **block-diagonal** — a NEW "parallel /
blockwise VJP" lemma (each channel independent; the whole-vec VJP is the direct
sum of per-channel VJPs). Then a `bnPerChannelF`/`bnPerChannelBack` SHlo op pair
(γ/β render as rank-1 `Vec oc`, `dims=[1]`, vs the current rank-0 scalars).
*Optional for correctness; needed for an accuracy story.* See §6 honesty note.

### B9 — full ResNet-34 render + trainer + GPU train  [render; ~1000+ lines]
- `resnet34TrainStepText` — scale `resnetTrainStepText` (StableHLO.lean): strided
  stem (`flatConvStridedF`/`convStridedBack`), 16 blocks, residual fan-ins,
  strided downsample at stage starts, GAP, dense. Watch SSA-name management.
- `ResNet34Layout` (IreeRuntime.lean), `MainResnet34Verified.lean`,
  `resnet34-verified` exe (mirror `MainResnetVerified.lean`). GPU-train CIFAR-10
  (or downscaled ImageNet/imagenette).
- De-risk each new op fragment on `iree-compile` before the full chain.

---

## 3. Suggested first steps for the clean session
1. `git log --oneline -12` (confirm the 10 ch6 commits); skim
   `LeanMlir/Proofs/ResNet34.lean` end-to-end (`resnet34_has_vjp_at` is the apex)
   and `CnnConcrete` in `CNN.lean` (the discharge template for B7).
2. Build + audit green first (see §4): **149/149**.
3. **Do B7** (concrete instance) — the highest-value next proof, makes the
   verified ResNet-34 *unconditional*. Mirror `CnnConcrete` block-by-block.
4. Then B8 (per-channel BN) and B9 (render+train) as separate chapters.

---

## 4. Build / audit / run

```bash
cd /home/skoonce/lean/proof_verify_demo/lean4-mlir
lake build Proofs                       # type-checks the suite incl. StridedConv + ResNet34
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c 'depends on axioms: \[propext, Classical.choice, Quot.sound\]$'
#   must equal total 'depends on axioms:' lines (149 now; B7 → 150)

export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
export IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0
DATA=/home/skoonce/lean/claude_max/lean4-jax/data           # mnist + cifar-10/ live here
.lake/build/bin/resnet-verified "$DATA"                     # ch6-A trainer (~60.8% @ 10ep, ~50s/ep)
.venv/bin/iree-compile --iree-hal-target-backends=rocm --iree-rocm-target=gfx1100 \
  --iree-codegen-llvmgpu-use-reduction-vector-distribution=false M.mlir -o /tmp/x.vmfb
```
- **Running the trainer in this harness:** use the Bash tool's `run_in_background:
  true`; do NOT `nohup … &` (the detached child gets killed at command exit) and
  do NOT `pkill -f resnet-verified` from a shell whose own cmdline contains that
  string (it kills itself → exit 144). The exe's stdout is fully-buffered to a
  file — `MainResnetVerified` flushes after each epoch (`(← IO.getStdout).flush`),
  so epoch lines appear live; keep that pattern.

---

## 5. Gotchas (learned this chapter — read before editing)

- **`decimate`/strided VJP design:** never re-derive conv VJP with stride
  arithmetic; use `conv_stride2 = decimate ∘ conv1` + `vjp_comp`. `decimateFlat`
  is a `reindexCLM` ⇒ differentiable for free; its VJP via `pdiv_reindex`.
- **Type ascriptions on strided theorem statements:** `flatConvStride2`'s `h,w`
  appear only in the *type*, not in `W`/`b`, so a bare `flatConvStride2 W b` can't
  infer them — every statement ascribes `(… : Vec (ic*(2*h)*(2*w)) → Vec (oc*h*w))`
  (mirrors `flatConv_differentiable`). And nonlinear unification `oc*?*? =
  oc*(2h)*(2w)` won't solve dims for `.comp`/`vjp_comp` — pin with type-ascribed
  `have`s/`let`s.
- **`PProd`, not `×`:** `DifferentiableAt` is a `Prop`; `Prod` needs `Type`. The
  `_at`-chain (`ChainData`, `chain_vjp_diff_at`, `vjp_comp_diff_at`) carries
  `(HasVJPAt, DifferentiableAt)` as `PProd` (use `.fst`/`.snd`).
- **`vjp_chain` (global) vs `vjp_chain_at` (smooth-point):** identity blocks
  `rblk` have a ReLU kink ⇒ only `HasVJPAt`, so the *whole net* uses
  `vjp_chain_at`/`ChainData` (threads each block's diff+VJP at its running
  activation `chainComp rest x`). `vjp_chain` (global, B4) is for blocks with
  everywhere-VJPs only.
- **Strided backward render = pad + reuse convBack:** `convStridedBack` emits
  `stablehlo.pad` (interior=[..,1,1], high=[..,1,1]) to zero-upsample dy to the
  2h×2w grid (= `decimate.back` = `lhs_dilation`), then the reversed-kernel
  (transpose+reverse) stride-1 conv (= `conv.back`). NOT a raw `lhs_dilate` conv
  (that gives 2h−1, off-by-one).
- **Adding an SHlo op = 9-site lockstep** (inductive/den/skel/Raw/Tok/toToks/
  emitTok + parseStack/parseStack_toToks). `addV`/`gapF`/`flatConvStridedF`/
  `convStridedBack` are worked examples. rfl-faithful ops (e.g. `flatConvStridedF_faithful`)
  do NOT go in AuditAxioms; `roundtrip` covers them structurally.
- **`StableHLO.lean` now `import LeanMlir.Proofs.StridedConv`** (for the strided
  op `den`s). No cycle (StridedConv imports only CNN).
- **simp can't unfold `biPath` below its arity** — `resnetFwdGraph_faithful`
  uses `simp` then `unfold cnnForward cbr rblk rblkP residual residualProj biPath`.
- **Audit exact-match** wants ℝ-carrying headlines (`*_correct`/`*_faithful`
  through real analysis), not pure `rfl`. `#print axioms resnet34_has_vjp_at`
  works on the def directly (its `.correct` field is the ℝ pdiv-Jacobian).
- **emitter byte-stability:** after editing, `git diff --stat verified_mlir/` must
  be empty for ch2–ch6-A `.mlir` files. `#eval … writeFile` needs
  `IO.FS.createDirAll "verified_mlir"`.
- **rank-0 scalar γ/β** flow through the FFI (empty-product size 1,
  `broadcast_in_dim … dims = []`). Per-channel BN (B8) switches to `Vec oc`
  (rank-1, `dims = [1]`).

---

## 6. The honest framing (don't oversell)

- **ch6-A is a verified ResNet-*STYLE* net** (stem+maxpool+1 id-block+1 proj-block
  +GAP+dense), GPU-trained 60.8%. **ch6-B (`resnet34_has_vjp_at`) is the verified
  34-layer *architecture*** — the whole-net VJP is proven + audit-clean — but it is
  (a) *conditional* until B7 discharges a concrete instance, and (b) *not yet
  trained/rendered* (B9). Don't claim "trained ResNet-34" until B7+B9 land.
- **BN is per-example GLOBAL scalar γ/β**, not per-channel batch-BN. On CIFAR it
  *underperforms* no-BN (ch5: 57% vs 66%); ch6-A's 60.8% with this BN is fine. A
  real accuracy story needs per-channel BN (B8). The chapter win is a
  **correctness/codegen** result (residuals + GAP + 34-layer depth + strided
  downsampling all *verified*, gradients axiom-clean), not a SOTA number.
- The strided-conv backward, deep-chain `_at` VJP, and whole-net fold are the
  genuinely-new verified pieces; everything else reuses ch4/ch5 machinery.

---

## 7. File map (ch6)

- `LeanMlir/Proofs/StridedConv.lean` — B1/B2 (strided conv VJPs). imports CNN.
- `LeanMlir/Proofs/ResNet34.lean` — B4/B5/B6a/B6b/B6c. imports CNN + StridedConv.
  Apex: `resnet34_has_vjp_at`. (B7 concrete instance goes here.)
- `LeanMlir/Proofs/StableHLO.lean` — A ops (`addV`/`gapF`) + `resnetFwdGraph(_faithful)`
  + `resnetFwdModuleV` + `resnetTrainStepText` + B3 strided ops; `import …StridedConv`.
- `LeanMlir/Proofs/StableHLOParse.lean` — parser cases for all ch6 ops.
- `LeanMlir/IreeRuntime.lean` — `ResnetLayout` (26 params). (B9: `ResNet34Layout`.)
- `MainResnetVerified.lean` + lakefile `resnet-verified` — ch6-A trainer.
- `tests/AuditAxioms.lean` — 149 headlines (imports StridedConv + ResNet34).
- `verified_mlir/resnet_{fwd,train_step}.mlir` — ch6-A rendered (byte-stable).
- Reuse templates: `cnn_has_vjp_at`@CNN.lean:2817 + `CnnConcrete` (B7 template);
  `cifarBnTrainStepText`/`MainCifarBnVerified.lean` (render+trainer template).
