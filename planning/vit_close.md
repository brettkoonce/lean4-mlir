# Planning — closing ViT (ch10) both ways

The ViT analogue of `planning/{efficientnet,mobilenetv2,resnet34,convnext}_close.md`. Goal: bring a
representative (2-block, 1-head) ViT to the **"closed both ways"** bar — **(a)** train-step text =
name-threaded `pretty` of a *proven* `SHlo` forward graph, **(b)** every param output certified
`θ − lr·certified-gradient`, 3-axiom-clean.

**Headline: ViT is the most EXPENSIVE close of the ladder — but for the opposite reason from
EfficientNet.** The hard *math* is already done: the attention backward suite is fully proven and
audited (`sdpa_back_{Q,K,V}_correct`, `mhsa_has_vjp_mat_correct`, `transformerBlock_has_vjp_mat`,
`vit_full_has_vjp[_correct]` — all 3-axiom clean). What's missing is the *token layer*: `SHlo` has
exactly ONE attention token pair today (`softmaxRowF`/`softmaxRowBack`), and a ViT forward graph
needs the first genuinely-new token family since EfficientNet's `batchOp` — **activation×activation
matmul** (`Q@Kᵀ`, `P@V`: both sides are graph values, where every existing matmul token `dotIn`/
`dotOut` carries a *named fixed weight*), plus transpose, per-token LN, per-token dense, and the
CLS concat/slice plumbing. EfficientNet's cost was a batched *index*; ViT's cost is new *tokens*
(per-example everything is batch-separable — no `batchMap` apparatus needed).

Two structural facts make it tractable:
1. **Matmul backward IS matmul** — `d A = dC·Bᵀ`, `dB = Aᵀ·dC`. With one `matmulF` (plain `A·B`)
   and one `transposeF` token, every attention forward AND backward op is covered (the layer-scale
   trick again, scaled up: no backward-only token semantics, just forward tokens on cotangents).
   The row-softmax backward token already exists.
2. **Everything is per-example separable** (per-token LN, per-row softmax, per-example matmuls), so
   the batch-1 `den` is faithful and `pretty BS` just prefixes the batch dim — same regime as
   MNV2/r34/ConvNeXt. The `softmaxRowF` emitter is the established template for row-structured ops
   at a flat index: emit `[B, m·n] → reshape [B,m,n] → 3-D op → reshape back` (StableHLO.lean
   `emitTok`). Every new ViT token follows that bracket pattern.

---

## 0. Architecture (representative for the close)

**Representative = 2 *distinct-param* pre-norm blocks, heads=1, patchSize=1**, CIFAR-ish dims:
`ic=3, H=W=8 → N=64 patches, N+1=65 tokens (CLS), D=32 (= heads·d_head, heads=1), mlpDim=128 (4×),
nClasses=10`. Scores live at `65×65` per example. Forward:

  patchEmbed (per-pixel dense + CLS prepend + pos-embed add) → block₁ → block₂ → final-LN →
  CLS slice → classifier dense
  block = x + Wo·SDPA(Wq·LN₁(x), Wk·LN₁(x), Wv·LN₁(x))  then  · + Wfc2·gelu(Wfc1·LN₂(·))

**Representation choices (mirroring the proof side — same simplifications as prior nets, not gaps):**
- **Scalar LN γ/β.** The proven `vit_body`/`vit_full` use *scalar* `γ,β : ℝ` per LN site (per-token
  normalize over D, then scalar affine) — the ConvNeXt situation again. NB the committed ViT-Tiny
  render (`ViTRender.lean`) is MORE faithful (vector `γ,β : [D]`, decomposed scalar-LN(1,0) ∘
  `layerScale[D]` + bias); the representative closes the *proof's* scalar form; vector-[D] LN is an
  optional upgrade (the layerScale-γ bridge from ConvNeXtClose is exactly its γ-half).
- **heads=1.** `mhsa_has_vjp_mat` is proven for general `heads`, but multi-head *rendering* needs
  per-head column slice/concat tokens. heads=1 makes SDPA three matmuls + a row-softmax — the same
  granularity trade the 2-block/1×1-stem ConvNeXt representative made. Multi-head = optional upgrade.
- **patchSize=1.** `patchEmbed_flat` at patchSize=1 is a per-pixel dense (`ic→D`) + CLS + pos-embed —
  avoids even-kernel strided-conv tokens. The 16×16/s16 ViT-Tiny patchify is the scaling pass.
- **Separate Wq/Wk/Wv** (as the proof has them), not the committed render's fused QKV slab.

## 1. Starting state

**Proven & audited (Attention.lean, 3792 ln — the math is DONE):** `softmax_has_vjp` +
`pdiv_softmax`; `rowSoftmax_has_vjp_mat`; `sdpa` + closed-form `sdpa_back_{Q,K,V}` + `_correct`
(via `vjpMat_comp` chains); `mhsa_layer` + `mhsa_has_vjp_mat[_correct]` (general heads, the
column-stacked-slab proof); `transformerMlp_has_vjp_mat`; `transformerBlock_has_vjp_mat[_correct]`
(pre-norm, both sublayers, residuals); `transformerTower_has_vjp_mat` (k blocks — **but with a
single SHARED param tuple across blocks**, see the Item A caveat); `vit_body_has_vjp_mat` (tower +
final LN); `patchEmbed_flat_has_vjp` (general patchSize, incl. CLS + pos-embed);
`cls_slice_flat_has_vjp`; `classifier_flat_has_vjp`; `vit_full_has_vjp[_correct]` (the grand
composition). Matmul building blocks: `Mat.mul` + `matmul_{left,right}_const_has_vjp` +
`pdivMat` matmul Jacobians (Tensor.lean). Per-token lifts: `layerNorm/dense/gelu_per_token_has_vjp_mat`
(via `rowwise_has_vjp_mat`).

**Token layer (StableHLO.lean):** only `softmaxRowF`/`softmaxRowBack` (+ `rowSoftmaxFlat`/
`rowSoftmaxBackFlat` dens, faithfulness proven) are ViT-specific. `geluF`/`geluBack`, `addV`,
`dotIn`/`dotOut`/`addBcast` (fixed-weight), `layerScaleF` exist. NO activation×activation matmul,
NO transpose, NO per-token LN/dense tokens, NO concat/slice.

**Render side:** `ViTRender.lean` (514 ln) + `tests/TestViTTrain.lean` render the FULL production
ViT-Tiny (224², 16×16/s16, 197 tokens, depth-12, 3 heads, vector-[D] LN, fused QKV, 200 params,
BS=32) → committed `verified_mlir/vit_train_step.mlir`. Its fragments are per-op outputs of
proven-faithful emitters, gradcheck-validated (TestSDPA/TestMHSA/TestViTBlock/TestViTTiny) — but
there is **no `den(graph)` theorem and no name-threaded `pretty`**: it's the hand-fragment stage
MNV2 was at before its close. `render_parity.py` (post-ConvNeXt fix) handles its mixed shapes.

---

## 2. The four rungs

### Item A — forward graph + faithfulness — [the long pole: the new token family]
New `SHlo` tokens (each with `den` + emitter + `_faithful`; all follow the `softmaxRowF`
reshape-bracket pattern, flat per-example index, `pretty BS` prefixes the batch):
- **`matmulF {m k n} : SHlo (m·k) → SHlo (k·n) → SHlo (m·n)`** — `den = flatten (Mat.mul …)`;
  emits `dot_general` batching [0]. The ONE new semantic op. Covers `Q·Kᵀ`-via-transpose, `P·V`,
  and (on cotangents) every matmul backward.
- **`transposeF {m n} : SHlo (m·n) → SHlo (n·m)`** — pure reindex (linear); `stablehlo.transpose`.
  Also resolves the patchify layout flip (conv channel-major ↔ token row-major).
- **`scaleF (s : ℝ) (sStr : String)`** — constant scalar multiply (the `1/√d` SDPA scale).
- **`lnRowF`/`lnRowBack {m n}`** — per-row scalar-LN over `[m,n]` flat (row-wise `bnForward` /
  `bn_grad_input`), the LN analogue of `softmaxRowF`/`softmaxRowBack`; `den` = rowwise lift,
  faithful to the proof's per-token LN (`layerNorm_per_token…` via `rowwise` machinery).
- **`denseRowF`/`denseRowBack`** — per-token dense `[N,a]→[N,c]` with a *named weight* (the QKV/
  proj/MLP matmuls). Primary plan: a dedicated token pair (reshape-bracket + `dot_general`);
  fallback: reuse `batchOp (.dense …)` at N=tokens — but VERIFY its emitter composes under
  `pretty`'s leading BS first (EfficientNet used batchOp with N=the train batch itself).
- **CLS concat + slice:** `clsConcatF {N D} : SHlo D → SHlo (N·D) → SHlo ((N+1)·D)` and
  `clsSliceF : SHlo ((N+1)·D) → SHlo D`, plus the slice's backward (`clsPadF`: scatter into row 0).
  ⚠ Index arithmetic: `(N+1)·D` is NOT defeq to `D + N·D` (Nat.mul recurses on the right) — expect
  an explicit reindex/cast in `den`; budget fiddliness here.
- Pos-embed add is just `.addV` with an `.operand "%pos"` (param as operand — free).

Then **`vitFwdGraph` + `_faithful`**. ⚠ The proven tower SHARES one param tuple across blocks; a
train step needs DISTINCT per-block params (the committed render has 200 distinct). So first define
`vitForward2` (2 blocks, distinct params: `classifier ∘ finalLN ∘ block₂ ∘ block₁ ∘ patchEmbed`) and
get `vitForward2_has_vjp` by composing `transformerBlock_has_vjp_mat` twice + the existing bridges —
cheap (the tower proof does exactly this composition with shared params; per-block lemmas then
chain, per `[[efficientnet-batched-render]]` gotchas). Faithfulness target: `den vitFwdGraph =
vitForward2`.

### Item C — the param close
| family (render SSA)                  | forward fn              | certified by |
|--------------------------------------|-------------------------|--------------|
| Wq/Wk/Wv/Wo, Wfc1/Wfc2 + biases      | per-token dense         | **new family**: per-token dense W-grad `dW = Σ_tokens xᵢᵀ·dyᵢ` — the M2 outer-product bridge row-lifted (CifarBnClose's channel-gather/`reindexCLM` recipe is the template); bias `db = Σ_tokens dyᵢ` |
| classifier `Wcls/bcls`               | dense on the CLS row    | M2 `weight/bias_grad_bridge` (**reuse** — single-vector dense) |
| patch embed `Wp/bp` (patchSize=1)    | per-pixel dense         | same per-token dense family (**reuse of the new bridge**) |
| LN γ/β ×5 sites (scalar, per-token)  | rowwise `layerNormForward` | **new**: row-lift of ConvNeXtClose's `Vec 1` embedding — `dγ = Σ_{tokens,D} dy·x̂`, `dβ = Σ dy` (affine in params ⇒ no `0<ε`) |
| `pos_embed`                          | additive                | `dPos = Σ_batch dy` at the embed output — pdiv of `x+P` in `P` is identity (essentially free) |
| `cls_token`                          | concat row 0            | `dCls = row-0 slice of the embed cotangent` — a reindex Jacobian (cheap) |
| attention internals (softmax, scale) | —                       | no parameters |

### Item B — structured render — [given A; backward all by token]
`tests/TestViTTrainPC.lean`: forward + backward via `pretty` over the `vitFwdGraph` tokens, the
MNV2/r34/ConvNeXt `TrainPC` pattern (FNames per block, `StateM Nat`). Backward by token:
- attention: `dWo`-side dense back (`denseRowBack`), then SDPA backward = **`matmulF`/`transposeF`
  on cotangents** (`dP = dO·Vᵀ`, `dV = Pᵀ·dO`, `softmaxRowBack` (exists), `dQ = dS·K·scale`,
  `dK = dSᵀ·Q·scale`) — exactly the proven `sdpa_back_{Q,K,V}` shapes;
- MLP: `denseRowBack` / `geluBack` (exists) / `denseRowBack`;
- LN: `lnRowBack`; residuals: `addV`; CLS slice back: `clsPadF`; classifier back: `dotOut`.
Hand-emit only the param grads (per-token dense `dW/db` reduces, LN `dγ/dβ` reduces, `dPos`,
`dCls` slice). Validation: iree-compile + `render_parity.py` **ref-only smoke** (the committed
renderer is full ViT-Tiny — different signature, no swap-parity; same situation as ConvNeXt) +
optionally a Lean gradcheck reusing the TestViTBlock harness at the representative dims.

### Item D — cotangent chain — [optional; the attention block chain, batch-1]
`ViTChainClose.lean` — the `ConvNeXtChainClose` analogue. Pin each param to the actual chain
cotangent, composing the rendered backward denotations: classifier-back → `clsPad` → final-LN back
(`bn_grad_input` rowwise) → per block: MLP sublayer (dense-back → gelu mask → dense-back → LN-back,
+ residual passthrough) and attention sublayer (proj dense-back → **the proven `sdpa_back_{Q,K,V}`
closed forms** → Q/K/V dense-backs fan-IN (three cotangents sum at the LN₁ output) → LN-back, +
residual). New wrinkle vs all prior nets: the **three-way fan-in** at LN₁ (Q,K,V branches) — the
`biPath` fan-in machinery generalizes. Per-block lemmas then chain; nested-application form;
explicit dims on stage lemmas.

---

## 3. Order & status
1. **Item A — ✅ CLOSED (2026-06-09).** Tokens landed (`bf6ebde`: ten ops, `patchEmbedF` as one
   coarse token instead of `clsConcatF`); `LeanMlir/Proofs/ViTFwdGraph.lean` adds `vitForward2`
   (distinct-param 2-block, generic heads) + `vitForward2_has_vjp[_correct]` (UNCONDITIONAL,
   only `0 < ε`), `vitBlockGraph`/`vitFwdGraph` (heads = 1) + **`vitFwdGraph_faithful`**
   (`den vitFwdGraph = vitForward2` at heads := 1). The per-head plumbing collapses via
   `mhsa_layer_one_head` (the `Fin 1 × Fin d ≃ Fin (1·d)` sum reindex); the den side reduces
   through flat↔Mat commutation bridges (`*_flat` lemmas) + `vitBlockSpelled`. Audit 286/286.
2. **Item C — ✅ FULLY CLOSED (2026-06-09).** `LeanMlir/Proofs/ViTClose.lean`:
   per-token dense W/b (`vit_render_rowdense{W,b}_certified` — the row-lifted M2 family, covers
   Wq/Wk/Wv/Wo + Wfc1/Wfc2 + biases), row-lifted scalar-LN γ/β
   (`vit_render_rowln{gamma,beta}_certified`, all 5 sites, no `0<ε`), pos-embed identity
   (`vit_render_pos_certified`), CLS masked gather (`vit_render_cls_certified`); classifier =
   verbatim M2 reuse. Patch conv `Wp`/`bp` (§ E,
   `vit_render_patch{W,b}_certified`): `patchEmbed_flat` is LINEAR in the kernel with CONSTANT
   pad-guarded read coefficients — no pad-eval calculus needed; `dWp = Σ_p read·dy_(p+1,·)`,
   `dbp = Σ_p dy_(p+1,·)` (CLS row masked). Audit 307/307.
3. **Item B — ✅ CLOSED (2026-06-09).** `tests/TestViTTrainPC.lean`: forward + the whole backward
   cotangent chain proof-rendered via `pretty` over the `vitFwdGraph` tokens (fn
   `@vit_rep_train_step`, 3×8²/P=1/65 tokens/D=32/mlp=128, 40 params). The SDPA backward is the
   forward `matmulF`/`transposeF` on cotangents (dP=dO·Vᵀ, dV=Pᵀ·dO, softmaxRowBack, undo-scale,
   dQ=dS·K, dK=dSᵀ·Q); residual + Q/K/V fan-ins are `addV`. Hand-emitted only the Item-C-certified
   param grads (per-token dense dW/db, rowwise-LN dγ/dβ, dPos, dCls slice, patch-dense dWp/dbp,
   M2 head). Validation: iree-compile OK + gfx1100 ref-only smoke 40/40 outputs finite & non-zero
   (`scripts/render_parity.py --fn vit_rep_train_step`).
4. **Item D — ✅ CLOSED (2026-06-09).** `LeanMlir/Proofs/ViTChainClose.lean`: the Item C bridges
   pinned to the actual attention-block backward chain. Chain cots compose the rendered backward
   denotations (`vitCot{G,M1,Ln2,H,Att,DP,DS,DQ,DK,DV,Ln1,Xin,Fl,B2out}`); the substantive ties
   `vitCotD{Q,K,V}_eq_sdpa_back_{Q,K,V}` prove the matmul-spelled SDPA segments ARE the proven
   closed forms at the pinned saved activations. The Q/K/V three-way fan-in (`vitCotLn1`) is the
   new structural wrinkle. 18 chain-pinned param theorems (all block families + final-LN +
   pos/cls/patch). Audit 330/330.

**ALL FOUR RUNGS CLOSED (2026-06-09): the representative ViT is closed both ways with the
cotangent chain pinned — the ladder covers all four flagship families (CNN / inverted-residual /
ConvNeXt / transformer) at the full MNV2/r34/ConvNeXt bar.**

## Scaling pass status
- **vector-[D] LN — ✅ core closed (2026-06-09).** `LeanMlir/Proofs/ViTVecLN.lean`: `layerNormVec`
  (= ViTRender's scalar-LN(1,0) ∘ per-channel scale + bias) with VJP composed from
  `layerNorm_has_vjp`(1,0) + `layerScale_has_vjp` + bias translation; vector-LN sublayers/block
  (`transformerBlockV_has_vjp_mat`); `vitForward2V(_has_vjp[_correct])` (unconditional, only
  `0<ε`); two new broadcast tokens `rowScaleF`/`rowBiasF` (9-site lockstep; rowScaleF is its own
  input-VJP); `vitFwdGraphV_faithful` (each LN site = lnRowF(1,0) → rowScaleF → rowBiasF, the
  exact ViTRender decomposition); per-channel param bridges
  `vit_render_vecln{gamma,beta}_certified` (dγ_k = Σ_tokens dy·x̂ keeping the channel axis).
  **Render upgraded (B analogue):** `TestViTTrainPC.lean` now renders the production LN form
  (3-token decomposition; backward = rowScaleF-on-cotangent + lnRowBack(γ=1); per-channel
  dγ/dβ off the SAVED normalize output) — iree-compile OK + gfx1100 smoke 40/40.
  **Chain pins (D analogue):** `vitCot{H,Att,Xin,B2out}V` decompose the LN input-VJPs the
  same way; the dense/SDPA segments + ties are LN-form-agnostic and hold verbatim; six
  per-channel γ/β pins at the actual chain cotangents. Audit 347/347. **This item is DONE.**
- **16×16/s16 patchify — ✅ already general.** `patchEmbedF`, its faithfulness, the §E patch
  close, and the chain pins are all stated at general `patchSize` — nothing scalar-specific
  to lift. (The Item B render exercised P=1; a P=16 render is a config change.)
- **multi-head / fused QKV / depth-12** — open (per-head slice/concat tokens; the proofs are
  already general in `heads`; depth-k = the per-block generic lemmas chained k times).

## Handoff notes
- **Templates:** `softmaxRowF`/`softmaxRowBack` (StableHLO.lean ~198–250 + its `emitTok` case) is THE
  model for every new row-structured token (den as rowwise lift, reshape-bracketed 3-D emission);
  `ConvNeXtClose.lean` for the scalar-LN `Vec 1` embedding (row-lift it) and the param-close file
  shape; `tests/TestConvNeXtTrainPC.lean` for Item B; `ConvNeXtChainClose.lean` for Item D;
  `MobileNetV2RenderPC.lean`/`ResNet34RenderPC.lean` for per-block graph + faithful + chain structure.
- **Proof-engineering gotchas that carry over** (`[[efficientnet-batched-render]]`): per-block
  lemmas then chain (never reduce the whole net at once — especially the 2-block body); ℝ-forward in
  nested-application form, not `∘`; explicit dim args on free-floating stage lemmas. New ViT-specific
  ones to expect: `(N+1)·D` vs `D + N·D` Nat-index casts at the CLS boundary; the conv-vs-token
  layout transpose; `Mat.flatten`/`Mat.unflatten` round-trips in every matmul faithfulness proof.
- **The shared-params tower caveat is real:** `transformerTower_has_vjp_mat`/`vit_full_has_vjp` use
  ONE param tuple for all k blocks. Don't try to close the shared-params net (a train step updating
  the same tensor from two blocks' grads is a different statement); build `vitForward2` with
  distinct params from the block VJPs.
- **Validation assets:** TestSDPA/TestMHSA/TestViTBlock Lean gradchecks (reusable at representative
  dims); `verified_mlir/vit_train_step.mlir` + `vit_fwd.mlir` (committed, GPU-validated);
  `render_parity.py` (handles scalars post-ConvNeXt); local run recipe in
  `[[running-verified-trainers-locally]]`.
- **Optional upgrades (the scaling pass, in rough value order):** vector-[D] LN γ/β (matches the
  committed render; the γ-half is ConvNeXtClose's `layerScale` bridge); multi-head (per-head column
  slice/concat tokens; the math is already general in `heads`); 16×16/s16 patchify (even-kernel
  strided conv tokens); fused QKV slab (the proof's `mhsa_qkv_W` machinery exists); depth-12
  ViT-Tiny shapes.
