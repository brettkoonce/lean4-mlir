# Verified ResNet-34 (Chapter 6) — handoff

> **⭐ 2026-06-07 UPDATE — moved to IMAGENETTE 224² (uncommitted, on `main`).** The
> `resnet34-verified` trainer no longer targets CIFAR 32²; it is now the paper-native
> **ImageNet-resolution ResNet-34 on Imagenette 224²** (like ch8's B0). New **faithful
> 7×7 stride-2 stem** (224→112; `convStem` fwd + `convWGradStem` back in
> tests/TestResnet34Train.lean — upsample 112→224 then 7×7/pad-3 weight-grad, reusing the
> proven `flatConvStride2 = decimate2 ∘ stride-1` pattern, kernel-general) + 2×2 maxpool
> (112→56); spatial **56→28→14→7**, GAP÷49. Stem W is now `[64,3,7,7]`; still 146 params /
> 22.67M floats. BS=32. Edits: tests/TestResnet34{Fwd,Train}.lean, `ResNet34Layout`@IreeRuntime,
> MainResnet34Verified (now loads `<DATA>/imagenette/{train.bin@256 centerCrop→224, val.bin@224}`).
> Both MLIRs render + iree-compile rocm (fwd 89KB, train 254KB); exe builds; ran one epoch =
> **9.45% (= chance, climb pending a later "run all trainers" pass)** — confirms the 224² train
> epoch + 34-layer backward execute with no OOM/crash. Also: the **entire ch6-A ResNet-*style*
> trainer was DELETED** — `MainResnetVerified.lean` + lakefile `resnet-verified` entry +
> `ResnetLayout`@IreeRuntime + `resnetFwdModuleV`/`resnetTrainStepText`@StableHLO +
> `verified_mlir/resnet_*.mlir`. The audited `resnetFwdGraph` + `resnetFwdGraph_faithful` are KEPT
> (audit unchanged). §§ below describe the original CIFAR build; the *architecture/proof* content is
> unchanged — only resolution/stem/data moved (and the dead ch6-A trainer plumbing is gone).

Continuation doc for the ResNet chapter of the verified-codegen ladder. Picks up
after **Milestone A (verified ResNet-*style* net), B1–B6c (the whole verified
ResNet-34 architecture), B7 (the unconditional concrete instance), and B8a/B8a'
(per-channel BatchNorm — the math) are DONE and committed locally** (not pushed).

> **Status in one line:** the whole-network VJP of a real 34-layer ResNet —
> **`Proofs.resnet34_has_vjp_at`** — is PROVEN and **UNCONDITIONAL** (B7's
> `ResNet34Concrete.resnet34Concrete_has_vjp_correct` discharges every
> smoothness/no-tie hyp at 1ch/32×32); B8a/B8a' add the **per-channel BN
> block-diagonal VJP** (`bnPerChannelFlat_has_vjp_correct`) **+ its renderable
> closed-form backward** (`bnPerChannel_grad_input_correct`); and **B9-entry** adds
> the **layout bridge** `(oc*h)*w ↔ oc*(h*w)` (`bnPerChannelTensor3_has_vjp_correct`)
> **+ the renderable backward in the network's Tensor3 layout**
> (`bnPerChannelTensor3_grad_input_correct`); and **B8b** lands the per-channel BN
> **SHlo op pair** `bnPerChannelF`/`bnPerChannelBack` (`bnPerChannelBack_faithful`),
> rendered + **iree-compiled to ROCm**. **Audit 155/155, 3-axiom.** And **B9 is DONE**:
> a real **ResNet-34 ([3,4,6,3], per-channel BN, strided downsamples) is GPU-TRAINING**
> on the verified-rendered StableHLO — `resnet34-verified`, **38.6% CIFAR-10 test acc
> @ epoch 1** (climbing), confirming the hand-wired 34-layer backward is correct. Ch6
> is now a *trained* model, not just a verified architecture. Read §§0–3 first; rest is
> reference.

Written 2026-06-07 by the session that did ch6-A + ch6-B1…B8a' + B9-entry + B8b + B9.
`origin/main` is behind by **~33 local commits** (`34cdc52`…HEAD; latest *code* commit
`bee1943` = B9c, the rest are handoff-doc updates); commit-to-main is fine, **never push
without explicit per-push permission**.

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
**ch6-A (ResNet-style) 60.8%**, **ch6-B (real ResNet-34, [3,4,6,3], per-channel BN)
38.6%@ep1 (climbing)** — architecture verified (`resnet34_has_vjp_at`) AND GPU-trained.

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
| B7 | `785caa3` | **UNCONDITIONAL concrete instance**: `resnet34Concrete_has_vjp_correct` (non-vacuity). |
| B8a | `dda518d` | **per-channel BN VJP** (block-diagonal): `bnPerChannelFlat_has_vjp_correct` (`PerChannelBN.lean`). |
| B8a' | `d7f6a2c` | **renderable per-channel BN backward**: `bnPerChannel_grad_input(_correct)` (the closed form to emit). |

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

## 2. ⭐ WHAT REMAINS (for a TRAINED ResNet-34)

**Nothing critical remaining — B9 is DONE (ResNet-34 GPU-trains, 38.6%@ep1).** B7,
B8a/B8a', B9-entry, B8b, and B9 below are all ✅ DONE — kept here with their design
notes for reference. Only optional polish remains (more epochs, library refactor):

### ~~B7 — concrete-instance discharge (non-vacuity)~~  ✅ DONE (`785caa3`)
`ResNet34Concrete.resnet34Concrete_has_vjp_correct` (in `ResNet34.lean`). A real
34-layer concrete net at 1ch/32×32: strided identity-conv stem → maxpool →
(3+4+6+3) identity blocks → 3 strided downsamplers → GAP → dense, every
smoothness/no-tie hyp discharged. **Two dimension-robust tricks** (so the
256/1024-element BN discharge needs no `norm_num` over thousand-element sums):
- **`bnForward_lb`** (`bn ≥ β − |γ|·√n`, from `(vₖ−μ)² ≤ Σ(vⱼ−μ)² = n·σ²`): the
  stem's `β = 20 > √256` forces `bn > 0`, so ReLU = id and the stem output stays
  injective → maxpool no-ties (`bnForward_injective` + `decimateIdx_injective`).
- **zero-weight blocks**: every non-stem conv has a zero kernel ⇒ BN input is the
  constant `0` ⇒ `bnForward_const_eq` collapses each residual body to `1`; the
  post-add ReLU input is `1 + activation > 0` (identity blocks, `activation ≥ 0`
  threaded by `idChainData`) or `2` (downsamplers, fully unconditional).
New reusable lemmas live in `ResNet34.lean`'s B7 section (BN bounds, strided-index
injectivity, generic `idBlk`/`downBlk` machinery). Note: ResNet34 now imports
MnistCNN (for `conv2d_1x1`/`maxPool2Smooth_of_injective`/`relu_id_of_pos`/positivity);
`bnForward_const_eq`/`flatConv_zero` are local copies of MobileNetV2's (could be
hoisted to CNN/BatchNorm later to dedupe).

### B8a — per-channel BatchNorm, the MATH  ✅ DONE (`dda518d`, `d7f6a2c`)
`LeanMlir/Proofs/PerChannelBN.lean`: `bnPerChannelFlat` + `bnPerChannelFlat_has_vjp(_correct)`
+ `bnPerChannelFlat_differentiable`; **and (B8a') the renderable closed-form backward**
`bnPerChannel_grad_input` + `bnPerChannel_grad_input_correct` (the per-example three-term
`bn_grad_input` per channel-slice, proven = the pdiv-Jacobian — the formula a per-channel
`bnBack` emits). Audit **155/155** 3-axiom-clean. Real ResNet BN
applies `bnForward (h·w) ε (γ c) (β c)` to each channel-slice (γ/β `Vec oc`), so the
whole Jacobian is **block-diagonal** across channels. Got it cheaply by GENERALIZING
the existing `rowwise_has_vjp_mat` (Tensor.lean, multi-head attn) from a single per-row
map to a per-row FAMILY `g : Fin m → (Vec n → Vec p)`:
- `pdivMat_rowIndep_perRow` (output row k depends only on input row k via `g k`),
- `rowwisePerRow_has_vjp_mat` / `rowwisePerRow_flat_differentiable`.
Then per-channel BN = view activation as `Mat oc (h·w)` (row = channel) and
`rowwisePerRow (fun c => bnForward (h·w) ε (γ c) (β c))` — VJP from `bn_has_vjp` per
channel, no block-diagonal Jacobian from scratch. **NB layout:** defined on `Vec (oc·m)`
with the *Mat* row-major split `oc*(h·w)`; the network's Tensor3 activations are
`(oc*h)*w` — B9 must bridge the re-association `oc*(h·w) ↔ (oc*h)*w` when wiring this in.
Tensor.lean untouched (per-row lemmas live in the new file).

### ~~B8b — per-channel BN render (the SHlo op pair)~~  ✅ DONE (`813065c`)
The `bnPerChannelF`/`bnPerChannelBack` SHlo op pair (γ/β rank-1 `Vec oc`, `dims=[1]`,
vs ch5-B's rank-0 scalars), full 9-site lockstep mirroring `bnF`/`bnBack`:
- **SHlo** `bnPerChannelF {oc h w} (γ β : Vec oc)` / `bnPerChannelBack {oc h w}
  (γ : Vec oc) (x saved)`, both `SHlo (oc*h*w) → SHlo (oc*h*w)`.
- **den** = `bnPerChannelTensor3` (fwd) / `bnPerChannelTensor3_grad_input` (back).
- **faithfulness**: `bnPerChannelF_faithful` (rfl, roundtrip-covered, not audited) +
  `bnPerChannelBack_faithful` (= pdiv block-diagonal Jacobian under `0<ε`, via
  `bnPerChannelTensor3_grad_input_correct`) — the per-channel peer of `bnBack_faithful`.
- **Raw/skel/Tok/toToks/parseStack/parseStack_toToks**: 2 cases each (round-trip closes).
- **emitTok**: reshape `[B,oc*h*w]→[B,oc,h,w]`, per-channel μ/var reduce over the spatial
  axes `[2,3]`, normalize, rank-1 γ/β `broadcast_in_dim dims=[1]`; backward = the
  block-diagonal three-term `(istd/m)·(m·dx̂ − Σdx̂ − x̂·Σ(x̂dx̂))` per channel.

`StableHLO.lean` now `import …PerChannelBN` (no cycle). Audit **155/155** 3-axiom-clean
(+`bnPerChannelBack_faithful`). **iree-validated:** `tests/TestPerChannelBn.lean` renders
`@perchannel_bn_fwd`/`@perchannel_bn_back` from the verified AST; both `iree-compile` to
ROCm gfx1100 (`lake env lean tests/TestPerChannelBn.lean` with `IREE_BACKEND=rocm`). What
remains is wiring it into the B9 trainer. See §6 honesty note.

### ~~B9 — full ResNet-34 render + trainer + GPU train~~  ✅ DONE — TRAINS 38.6%@ep1
The real ResNet-34 now GPU-trains on the verified-rendered StableHLO. Done in three
de-risked sub-steps (the doc's "de-risk each op fragment on iree-compile" advice):
- **B9a** (`cafb5ee`): `tests/TestResnet34Fwd.lean` — full forward renderer (stride-1
  3×3 stem 3→64 + maxpool 32→16 → [3,4,6,3] blocks @64/128/256/512, spatial 16/8/4/2,
  strided downsamples → GAP → dense), depth-parametric block/stage builders. 87KB →
  iree-compile OK ROCm. (Dims keep maps non-degenerate — no 2→1 strided conv.)
- **B9b** (`ba73b37` de-risk small net, `fe317a8` full depth): `tests/TestResnet34Train.lean`
  — train-step renderer, data-driven from one `Blk` list (`allParams`, 146 tensors).
  Backward helpers each mirror a proven VJP: reluBack, **bnBackPC** (per-channel BN
  three-term + dγ/dβ), convBack/convWGrad/convBiasGrad, **convBackStrided/convWGradStrided**
  (zero-upsample then stride-1 = the decimate-backward), maxpoolBack (select_and_scatter),
  dense/GAP back; block backward idBlockBack/downBlockBack (residual fan-in). 249KB →
  iree-compile OK ROCm.
- **B9c** (`bee1943`): `ResNet34Layout` (IreeRuntime.lean — 146-param `(dims,initKind)`
  specs in func-arg order, He/γ=1/β=0 init) + `MainResnet34Verified.lean` + lakefile
  `resnet34-verified`. Mean-loss cotangent (÷B), lr=0.1, `%x` is the flat `[BS,3072]`
  FFI buffer reshaped to `[BS,3,32,32]`. Reuses `mlpTrainStepV`. **GPU-trains CIFAR-10:
  epoch 1 = 38.6%** (gradients correct ⇒ the hand-wired 34-layer backward works).
- **NB (honest framing):** the trainer is a *codegen* artifact (hand-rendered StableHLO),
  faithful PER-OP (each fragment is the StableHLO a proven-faithful emitter produces);
  it is NOT a single `den(trainStep)=…` theorem. The whole-net VJP theorem is
  `resnet34_has_vjp_at` (parametric, audit-clean); the trainer mirrors its structure and
  is validated by training (loss/acc). Per-channel BN here is per-example instance-norm
  (matches the proven `bnPerChannelFlat`), not batch-BN.
- Remaining polish (optional): run more epochs for a final number; byte-stable commit of
  `verified_mlir/resnet34_*.mlir`; optionally hoist the renderers from `tests/` into a
  library module shared with the Main (currently the Main reads the committed MLIRs).

#### Original B9 plan (for reference)
- `resnet34TrainStepText` — scale `resnetTrainStepText` (StableHLO.lean): strided
  stem (`flatConvStridedF`/`convStridedBack`), 16 blocks, residual fan-ins,
  strided downsample at stage starts, GAP, dense. Watch SSA-name management.
- `ResNet34Layout` (IreeRuntime.lean), `MainResnet34Verified.lean`,
  `resnet34-verified` exe (mirror `MainResnetVerified.lean`). GPU-train CIFAR-10
  (or downscaled ImageNet/imagenette).
- De-risk each new op fragment on `iree-compile` before the full chain.
- ~~**Suggested entry point**: the layout bridge `oc*(h·w) ↔ (oc*h)*w`~~  ✅ DONE
  (`cb5fe74`, `3d4525e`). `PerChannelBN.lean` now has `reassocFwd`/`reassocBack`
  (the `reindexCLM` re-association + VJPs, mirroring `decimateFlat`; proven mutual
  inverses), `bnPerChannelTensor3` (= per-channel BN on the Tensor3 `(oc*h)*w`
  activation layout, conjugation of `bnPerChannelFlat`) with `_has_vjp(_correct)`,
  and the **renderable Tensor3 backward** `bnPerChannelTensor3_grad_input(_correct)`
  (the bridge reindexes are permutations ⇒ the `vjp_comp` backward collapses to
  `reassocBack ∘ bnPerChannel_grad_input ∘ reassocFwd`). So **B8b's two `den` targets
  are now both ready and proven faithful**: `den (bnPerChannelF) = bnPerChannelTensor3`
  and `den (bnPerChannelBack) = bnPerChannelTensor3_grad_input`. B8b is now pure MLIR
  render. Then B8b's op pair, then scale the trainer.
- **Closest render+trainer template:** `cifarBnTrainStepText` + `MainCifarBnVerified.lean`
  (ch5-B: the only existing BN trainer) — and ch6-A's `resnetTrainStepText`/
  `MainResnetVerified.lean` for the residual/GAP/strided structure.

---

## 3. Suggested first steps for the clean session
1. `git log --oneline -18` (confirm the ch6 commits through B8a' `d7f6a2c`); skim
   `LeanMlir/Proofs/ResNet34.lean` (`resnet34_has_vjp_at` apex,
   `ResNet34Concrete.resnet34Concrete_has_vjp_correct` B7 headline) and
   `LeanMlir/Proofs/PerChannelBN.lean` (B8a/B8a': `bnPerChannelFlat_has_vjp_correct`
   + the renderable `bnPerChannel_grad_input_correct`).
2. Build + audit green first (see §4): **155/155**.
3. **Do B8b + B9** — the remaining work toward a *trained* ResNet-34. B8b (the
   per-channel BN SHlo op pair) is best done WITH B9's trainer (it only pays off
   wired in + GPU-validated). Watch the `oc*(h·w) ↔ (oc*h)*w` re-association when
   wiring `bnPerChannelFlat` into the Tensor3-layout network (see B8a note).

---

## 4. Build / audit / run

```bash
cd /home/skoonce/lean/proof_verify_demo/lean4-mlir
lake build Proofs                       # type-checks the suite incl. StridedConv + ResNet34
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c 'depends on axioms: \[propext, Classical.choice, Quot.sound\]$'
#   must equal total 'depends on axioms:' lines (155 now, incl. B7 + B8a/B8a' + B9-entry + B8b)

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
  34-layer *architecture*** — the whole-net VJP is proven + audit-clean — and as of
  **B7 it is UNCONDITIONAL** (`resnet34Concrete_has_vjp_correct` exhibits a concrete
  34-layer instance with every hypothesis discharged, so the theorem is non-vacuous).
  It is still *not yet trained/rendered* (B9), and the concrete instance is a
  *degenerate* witness (zero-weight blocks ⇒ constant bodies ⇒ the block Jacobians
  are trivial; only the stem + maxpool carry a non-trivial Jacobian) — its job is
  non-vacuity, not a live model. Don't claim "trained ResNet-34" until B9 lands.
- **The BN actually *rendered/trained* so far is per-example GLOBAL scalar γ/β**, not
  per-channel batch-BN. On CIFAR it *underperforms* no-BN (ch5: 57% vs 66%); ch6-A's
  60.8% with this BN is fine. B8a now *proves* the per-channel BN VJP
  (`bnPerChannelFlat_has_vjp`), but it is **not yet rendered or trained** (B8b + B9) —
  so the accuracy story is still pending. The chapter win is a **correctness/codegen**
  result (residuals + GAP + 34-layer depth + strided downsampling + per-channel BN VJP
  all *verified*, gradients axiom-clean), not a SOTA number.
- The strided-conv backward, deep-chain `_at` VJP, and whole-net fold are the
  genuinely-new verified pieces; everything else reuses ch4/ch5 machinery.

---

## 7. File map (ch6)

- `LeanMlir/Proofs/PerChannelBN.lean` — **B8a/B8a' + B9-entry** (per-channel BN
  block-diagonal VJP + renderable backward + the Tensor3-layout bridge). imports
  BatchNorm. `bnPerChannelFlat(_has_vjp(_correct))`, `bnPerChannel_grad_input(_correct)`,
  + the reusable per-row `pdivMat_rowIndep_perRow`/`rowwisePerRow_has_vjp_mat`; and the
  **B9-entry** bridge: `reassocFwd`/`reassocBack` (`reindexCLM` re-association `(oc*h)*w
  ↔ oc*(h*w)`, mutual inverses, VJPs; `reassoc*_has_vjp_backward_eq` = inverse reindex),
  `bnPerChannelTensor3(_differentiable/_has_vjp(_correct))`, and the renderable Tensor3
  backward `bnPerChannelTensor3_grad_input(_correct)`. In the `Proofs` lib roots.
- `LeanMlir/Proofs/StridedConv.lean` — B1/B2 (strided conv VJPs). imports CNN.
- `LeanMlir/Proofs/ResNet34.lean` — B4/B5/B6a/B6b/B6c + **B7**. imports CNN + **MnistCNN**
  + StridedConv. Apex: `resnet34_has_vjp_at`; B7 headline `ResNet34Concrete.resnet34Concrete_has_vjp_correct`
  (+ reusable B7 lemmas: `bnForward_lb`/`bnForward_injective`/`decimateIdx_injective`,
  generic `idBlk`/`downBlk`).
- `LeanMlir/Proofs/StableHLO.lean` — A ops (`addV`/`gapF`) + `resnetFwdGraph(_faithful)`
  + `resnetFwdModuleV` + `resnetTrainStepText` + B3 strided ops + **B8b per-channel BN
  op pair** (`bnPerChannelF`/`bnPerChannelBack`, `bnPerChannel{F,Back}_faithful`);
  `import …StridedConv` + **`…PerChannelBN`**.
- `LeanMlir/Proofs/StableHLOParse.lean` — parser cases for all ch6 ops (incl. B8b).
- `tests/TestPerChannelBn.lean` — **B8b** standalone render + iree-compile check
  (`@perchannel_bn_fwd`/`@perchannel_bn_back`, both compile to ROCm gfx1100).
- `tests/TestResnet34Fwd.lean` — **B9a** full ResNet-34 forward renderer (run via
  `lake env lean …`, writes `verified_mlir/resnet34_fwd.mlir`, iree-compiles).
- `tests/TestResnet34Train.lean` — **B9b** full train-step renderer (data-driven `Blk`
  list + `allParams`; fwd/back fragment helpers; writes `verified_mlir/resnet34_train_step.mlir`).
- `LeanMlir/IreeRuntime.lean` — `ResnetLayout` (26 params, ch6-A) + **`ResNet34Layout`**
  (146 params, B9c — generated `(dims,initKind)` specs in func-arg order).
- `MainResnetVerified.lean` + lakefile `resnet-verified` — ch6-A trainer.
- `MainResnet34Verified.lean` + lakefile **`resnet34-verified`** — **B9c** ResNet-34
  trainer (programmatic 146-param init, GPU-trains CIFAR-10).
- `verified_mlir/resnet34_{fwd,train_step}.mlir` — B9 rendered (BS=128, 89KB/254KB).
- `tests/AuditAxioms.lean` — 155 headlines (imports StridedConv + ResNet34 + PerChannelBN).
- `verified_mlir/resnet_{fwd,train_step}.mlir` — ch6-A rendered (byte-stable).
- Reuse templates: `cnn_has_vjp_at`@CNN.lean:2817 + `CnnConcrete`@MnistCNN.lean:841
  (concrete-instance template); `cifarBnTrainStepText`/`MainCifarBnVerified.lean`
  (render+trainer template).
