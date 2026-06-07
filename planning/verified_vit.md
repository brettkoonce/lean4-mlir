# Verified ViT (Chapter 10) — handoff / plan for a fresh session

Plan doc for the **Vision Transformer** chapter of the verified-codegen ladder. ch10 brings
ViT (Dosovitskiy et al. 2021, "An Image is Worth 16×16 Words") into the same "render via the
proof-carrying StableHLO emitter + GPU-train on the rendered text **on Imagenette 224²**"
line as ch6 (ResNet-34), ch7 (MobileNetV2), ch8 (EfficientNet-B0), and ch9 (ConvNeXt-T).

> **Status in one line:** the **whole-network VJP is ALREADY PROVEN + AUDITED** —
> `Proofs.vit_full_has_vjp` / `vit_full_has_vjp_correct` (image→logits; UNCONDITIONAL except
> the `0 < ε` LayerNorm conditions; all-smooth — GELU + softmax + LayerNorm have no kinks),
> plus the **entire** component stack: `softmax_has_vjp`, `sdpa_back_{Q,K,V}_correct`,
> `mhsa_has_vjp_mat_correct`, `transformerBlock_has_vjp_mat_correct`, `transformerTower_has_vjp_mat`,
> `vit_body_has_vjp_mat`, `patchEmbed_flat_has_vjp`, `classifier_flat`. **What's missing is the
> VERIFIED STABLEHLO CODEGEN for attention** — there is **no row-softmax SHlo op, no batched
> multi-head `dot_general` rendering, no MHSA renderer, no `vit-verified` exe, no
> `verified_mlir/vit_*.mlir`.** This is exactly the ch8/ch9 starting point (proofs done →
> build the codegen). **Target: a faithful ViT (ViT-Tiny class) on IMAGENETTE 224²**, GPU-trained
> on the proof-rendered StableHLO. NB there is already an *UNVERIFIED* `vit-*-train`
> (NetSpec/IRPrint path, `MainVitTrain.lean`, `vitTiny`) — the ch10 deliverable is the
> *verified* peer, `vit-verified` (mirrors how ch8/ch9 have both `*-train` and `*-verified`).

Written 2026-06-07 (just after ch9 ConvNeXt finished — see `planning/verified_convnext.md`).
Read §§0–5 first; §§6–9 are reference. Commit-to-`main` is the workflow (no branches/PRs);
**never push without explicit per-push permission.**

---

## 0. The pattern (unchanged from every chapter)

Typed, proof-carrying StableHLO emitter (`LeanMlir/Proofs/StableHLO.lean`). Per op:
- **Semantic R4:** `den (graph) = <proven Mathlib fderiv quantity>` (faithfulness).
- **Syntactic R4:** `parse (toToks (skel a)) = skel a` (round-trip, for typed-AST ops).

The **trainer** is hand-rendered batched StableHLO string fragments (ch7/ch8/ch9 style), each
fragment faithful PER-OP to a proven function — NOT a single `den(trainStep)` theorem.
Validated by training (loss/accuracy moving), like the other `*-verified` exes. New SHlo ops
(here: the **row-softmax** pair) get the full 9-site lockstep + are validated by `iree-compile`.

Ladder so far: ch2 92.10%, ch3 97.86%, ch4 98.89%, ch5-A 66% / ch5-B 57%, ch6 ResNet-34
(Imagenette 224²), ch7 MobileNetV2 (Imagenette 224²), ch8 EfficientNet-B0 (Imagenette 224²),
ch9 ConvNeXt-T (Imagenette 224²). ch10 ViT continues at the same native 224² resolution.

---

## 1. ⭐ WHAT IS ALREADY DONE (proofs — committed + audited)

All in **`LeanMlir/Proofs/Attention.lean`** (3,793 lines — the capstone proof file, imports
Tensor + LayerNorm). Audited headlines are in `tests/AuditAxioms.lean` (≈ lines 79–90) and
are part of the **157/157 3-axiom-clean** count.

| Symbol | Where (Attention.lean) | What |
|---|---|---|
| `softmax` / `pdiv_softmax` | §1 (~239) | softmax + its Jacobian `pᵢ(δᵢⱼ−pⱼ)` (diag−outer). |
| `softmax_has_vjp` | §1 (~339) | **softmax VJP, closed-form O(n):** `back(z,dy)ᵢ = pᵢ·(dyᵢ − ⟨p,dy⟩)`. |
| `softmaxCE_grad` | §1 (~377) | softmax-CE loss cotangent `softmax(z) − onehot`. |
| `rowSoftmax(_has_vjp_mat)` | §2 (~466/532) | row-wise softmax of a matrix; VJP = per-row `softmax_has_vjp`. |
| `sdpa` | §2 (~575) | scaled dot-product attn `softmax(QKᵀ/√d)·V`. |
| `sdpa_back_{Q,K,V}(_correct)` | §2 (~678/768) | the **3 backward paths** dQ/dK/dV (via `vjpMat_comp`). |
| `sdpa_has_vjp_mat3` | §2 (~917) | full SDPA as a 3-output `HasVJPMat` (Q,K,V together). |
| `mhsa_layer` / `mhsa_has_vjp_mat(_correct)` | §3 (~974/1790/3726) | **multi-head self-attn** (QKV proj → per-head SDPA → out proj) + VJP. |
| `transformerMlp(_has_vjp_mat)` | §4 (~1937/2015) | MLP sublayer `fc2 ∘ gelu ∘ fc1`. |
| `transformerBlock(_has_vjp_mat)(_correct)` | §4 (~2076/2294/3740) | **block** = `(MLP∘LN+res) ∘ (MHSA∘LN+res)`. |
| `transformerTower(_has_vjp_mat)` | §5 (~2335/2397) | stack of k identical-hyperparam blocks. |
| `vit_body(_has_vjp_mat)` | §5 (~2451/2508) | backbone = `finalLN ∘ transformerTower`. |
| `patchEmbed_flat(_has_vjp)` | §6 (~2747/2847) | image → tokens (patch conv + CLS + pos-embed); closed-form input/weight grad. |
| `classifier_flat` / `cls_slice_flat(_has_vjp)` | §6/§7 (~2686/2618) | CLS-token slice → dense head. |
| **`vit_full` / `vit_full_has_vjp` / `_correct`** | §7 (~3629/3665/3765) | **whole ViT, image→logits, UNCONDITIONAL** (only the `0<ε`). The apex. |

**Read the proof's exact forward (`vit_full`, Attention.lean:3629) before scoping the
renderer.** It is the real architecture: `classifier ∘ (flatten ∘ vit_body ∘ unflatten) ∘
patchEmbed`. NB two structural simplifications baked into the *proof* (the renderer is
faithful PER-OP, may go beyond, exactly like ch8 B0 ↔ representative `efficientnet_has_vjp`):
- **`transformerTower` shares ONE set of block weights across all k blocks** (docstring
  ~2328). The faithful ViT has *distinct* per-block weights (12× the params). The renderer
  data-drives over a per-block param list (like ch8/ch9) — beyond the shared-weight proof
  witness, faithful per-op.
- **LayerNorm = the codebase's `layerNormForward`** (per-token over the D axis; see §7 for
  the scalar-vs-`[D]` γ/β question to confirm from the proof). Same "LN ≡ `bnForward`"
  identity used in ch9 — *not* re-derived.

So the proof gives you every **component** VJP audit-clean; the codegen for the *faithful
multi-block ViT-Tiny* (12 distinct blocks) goes beyond the shared-weight tower, faithful PER-OP.

---

## 2. ⭐ THE TARGET — faithful ViT (ViT-Tiny class) on IMAGENETTE 224²

Mirror the unverified `vitTiny` NetSpec (`MainVitTrain.lean`), at the paper-native 224²
resolution (10 classes):

```
patch embed : 16×16 conv stride 16 (3→192) "patchify"        224→14×14 = 196 patches
            : flatten patches → [196, 192], prepend CLS token → [197, 192], + pos-embed
tower       : 12× transformer block @ dim 192, heads 3 (d_head 64), MLP 768 (4×)
final LN    : LayerNorm(192)
head        : take CLS token (row 0) → dense 192→10 + softmax-CE
```

Transformer block (pre-norm, the ViT/GPT convention):
`x → LN → MHSA → + x → LN → MLP(fc1 192→768 → GELU → fc2 768→192) → + (that residual)`.
MHSA: `LN(x)` → Q/K/V dense (192→192 each, OR fused 192→576) → reshape to 3 heads × 64 →
per-head SDPA `softmax(QKᵀ/√64)·V` → concat heads → out-proj dense 192→192.

Token count N+1 = 197 (196 patches + CLS). Sequence math is per-example over the token axis;
attention is O(N²)=197² per head. BS=32 (224² memory), `D0 = 3·224·224 = 150528`. Data (same
as ch6/7/8/9): `<DATA>/imagenette/{train.bin@256 centerCrop→224, val.bin@224}`, via
`F32.loadImagenetteSized`/`loadImagenette` + `F32.centerCrop`. `DATA=/home/skoonce/lean/claude_max/lean4-jax/data`.

(ViT-Tiny ≈ 5.7M params — smaller than ConvNeXt-T's 28M; the cost is the O(N²) attention, not
the param count. If 12 blocks is slow to iterate, prototype with depth 4 then scale, like ch9.)

---

## 3. ⭐ WHAT'S REUSABLE (most of the non-attention net)

| ViT piece | Reuse | From |
|---|---|---|
| **GELU** (MLP activation) | `geluF`/`geluBack` SHlo op + the 4-D/2-D `geluAct`/`geluBack` fragments | **ch9 N1** (done) |
| **16×16 patch embed** (conv s16) | the **even-kernel non-overlapping strided conv** (k=stride, pad 0) — patchify with k=16 | **ch9 N3** patchify/patchifyWGrad (just s=16: dilate dy interior 15, no high → 16·14−15=209, valid conv → 16×16) |
| LayerNorm (per-token over D) | **`bnF`/`bnBack`** flat global-scalar form, applied per token (reshape `[B,N,D]→[B·N, D]`, LN over D) | ch5-B / **ch9 N2** `lnFwd`/`lnBack` |
| dense (QKV/out-proj/MLP fc/head) | `denseF`/`dot_general` + bias | ch2–ch4 |
| residual add | `addV` (tree-safe skip) | ch6-A |
| softmax-CE cotangent | `softmaxCE_grad` render (`softmax − onehot`) | ch4/ch8/ch9 trainers |
| imagenette loader + 224² Main/layout harness | `MainConvNeXtVerified` + `ConvNeXtLayout` (or EfficientNet) | **ch9** / ch8 |

The **closest reference for the attention emit** is the UNVERIFIED path:
`LeanMlir/Proofs/IRPrint.lean` — `renderSoftmax` (~705), `softmaxFwdModule`/`softmaxBackModule`
(~712/721, the proven `p⊙(dy−⟨p,dy⟩)` backward), `sdpaFwdModule`/`sdpaBackModule` (~750/764,
single-head SDPA fwd + the dQ/dK/dV backward); and `LeanMlir/MlirCodegen.lean` —
`emitMHSAForward` (~1141, full multi-head: reshape→transpose→batched `dot_general`→softmax→…→
out-proj) + `emitTransformerBlockForward` (~1226). These are the emit-shape reference (NOT
verified) to mirror into the hand-rendered-fragment style — exactly how ch8's `seFwd`/`seBack`
mirrored IRPrint's `renderSEGate`.

---

## 4. ⭐ WHAT'S GENUINELY NEW (the ch10 attention codegen)

Attention is the one architecture family no prior chapter rendered. The new work, in de-risk
order:

### V1 — row-softmax SHlo op pair `softmaxRowF`/`softmaxRowBack` (StableHLO.lean)
The one real new op. `den (softmaxRowF) = rowSoftmax` (each row of an `[N,N]` matrix gets
`softmax`); `den (softmaxRowBack) = (rowSoftmax_has_vjp_mat …).backward` (per-row
`pᵢ·(dyᵢ−⟨pᵢ,dyᵢ⟩)`). 9-site lockstep (inductive/den/skel/Raw/Tok/toToks/emitTok +
parseStack/parseStack_toToks). Template = the existing `softmaxDiv`/`expe` ops + ch9's `bnF`.
- **emit fwd** (per row, over the last axis): `exp` → `reduce add [last]` → `broadcast` →
  `divide` (the proven `softmax` is plain exp/sum — NO max-shift; if you add a max-shift for
  stability note it's mathematically identical, `softmax(z)=softmax(z−max)`, so still faithful).
- **emit back** = `p ⊙ (dy − ⟨p,dy⟩)` per row: recompute `p` from the saved pre-softmax (or
  reuse the forward `p`), `pdy = p*dy`, `s = reduce add [last]` broadcast, `dy − s`, `* p`.
  This is the closed-form `softmax_has_vjp` (IRPrint `softmaxBackModule` is the byte reference).
- rfl-faithful → stays OUT of the audit (round-trip covers it), like swishF/geluF. **No new
  Mathlib** (softmax_has_vjp is already proven; ch9's GELU lemma was the last bit of analysis).

### V2 — batched multi-head `dot_general` fragments (no new SHlo op)
`QKᵀ` and `weights·V` per head are batched matmuls. Hand-rendered `stablehlo.dot_general`
fragments with `batching_dims` over `[batch, heads]` (the emit shapes are in `emitMHSAForward`):
- scores `Q·Kᵀ`: `[B,H,N,d]×[B,H,N,d] batching=[0,1] contracting=[3]×[3] → [B,H,N,N]`.
- weighted `weights·V`: `[B,H,N,N]×[B,H,N,d] batching=[0,1] contracting=[3]×[2] → [B,H,N,d]`.
- backward = the transposed `dot_general`s (dQ=dScores·K, dK=dScoresᵀ·Q, dV=weightsᵀ·dOut),
  exactly `sdpa_back_{Q,K,V}`. The scale `1/√d` is a `multiply` (and `divide` by `√d` on the
  backward of the scaled scores). These are NOT new ops — `dot_general` already renders in the
  dense layers; this just uses batching dims + the per-head shapes.

### V3 — MHSA renderer (fwd + 3-path backward), data-driven over heads
One MHSA = QKV-proj (dense 192→576 then split, OR three 192→192) → reshape `[B,N,3D]→[B,N,H,d]`
→ transpose `[B,H,N,d]` → batched SDPA (V2 + V1) → transpose back → reshape `[B,N,D]` → out-proj
dense. Backward threads in reverse: out-proj back → reshape/transpose back → SDPA 3-path
(dQ/dK/dV via V2 + the row-softmax backward V1, undo the `/√d`) → reshape/transpose back → QKV
back + the 4 weight/bias grads (Wq/Wk/Wv/Wo + biases). **This is the big new combinator** (the
ViT analogue of ch8's SE module). De-risk standalone first (single-head SDPA, then multi-head).

### V4 — transformer block renderer (fwd + backward)
One block = `LN → MHSA → +res → LN → MLP(fc1 → geluF → fc2) → +res` (pre-norm). Backward:
res fan-in → MLP back (fc2 back → geluBack → fc1 back) → LN back → +res fan-in → MHSA back →
LN back. Data-driven over one `blocks` list (12 distinct blocks). All pieces but MHSA reuse
existing fragments (LN = ch9 `lnFwd`/`lnBack` per-token, dense, geluF/geluBack, addV).

### V5 — patch embed + CLS + positional embedding
- patch conv = ch9's even-kernel strided patchify with **k=stride=16** (3→192, 224→14×14);
  weight-grad = dilate dy interior 15 (no high → 16·14−15 = 209), valid conv → `[192,3,16,16]`
  (no input-grad — first layer). Then flatten `[B,192,14,14] → [B,196,192]` (transpose +
  reshape).
- **CLS token**: a learned `[1,192]` param, broadcast to `[B,1,192]`, concatenate to the front
  of the 196 patch tokens → `[B,197,192]`. Backward = slice off row 0's grad → dCLS (`cls_slice_flat`).
- **positional embedding**: a learned `[197,192]` param, broadcast-add to the token tensor.
  Backward = sum dy over the batch → dPos `[197,192]`.

### V6 — full ViT render + layout + Main + GPU-train
- `tests/TestViTFwd.lean` + `tests/TestViTTrain.lean` (peer of ch9's renderers): patch embed →
  CLS+pos → 12 blocks → final LN → CLS-slice → dense. Render `verified_mlir/vit_{fwd,train_step}.mlir`;
  iree-compile rocm to de-risk before the Main (prototype depth 4 first, then 12).
- `ViTLayout` (IreeRuntime.lean): patch `[192,3,16,16]`, CLS `[1,192]`, pos `[197,192]`, per
  block {LN1 γ/β, Wq/Wk/Wv/Wo `[192,192]` + biases, LN2 γ/β, fc1 `[768,192]`+b, fc2 `[192,768]`+b},
  final-LN γ/β, head `[192,10]`+b. (dims, initKind) in func-arg order.
- `MainViTVerified.lean` + lakefile `vit-verified` (clone `MainConvNeXtVerified` — imagenette
  loader, BS=32, `mlpTrainStepV` FFI, init, mean-loss SGD). GPU-train.

---

## 5. Suggested milestones / de-risk order
1. **Build + audit green first** (`lake build Proofs`; audit must stay **157/157**; the ViT
   proofs are already in it — confirm `#print axioms vit_full_has_vjp_correct` etc.).
2. **V1 row-softmax op** standalone: `tests/TestSoftmaxRow.lean` → `@softmax_row_{fwd,back}`
   iree-compile rocm (mirror ch9 `tests/TestGelu.lean` + IRPrint `softmaxBackModule`).
3. **V2+V3 SDPA → MHSA**: `tests/TestSDPA.lean` (single-head fwd+back, **adjoint/finite-diff
   gradcheck** — the SDPA 3-path backward is the highest-risk piece, validate numerically like
   ch8's TestSE plan) → then `tests/TestMHSA.lean` (multi-head reshape/transpose/batched matmul).
4. **V4 block** on a tiny config (`tests/TestViTBlock.lean`): one block fwd+back, iree-compile.
5. **V5 patch embed + CLS + pos** standalone iree-compile (reuse ch9 patchify k=16).
6. **V6 full renderer** (depth 4 first, iree-compile, THEN depth 12) → layout → Main →
   `vit-verified` → GPU-train imagenette 224².
7. Commit per milestone (to `main`, no push). Keep `git diff --stat verified_mlir/` empty for
   the other chapters' `.mlir` (byte-stability — the `StableHLO.lean` `#eval` regenerates them).

---

## 6. Build / audit / run

```bash
cd /home/skoonce/lean/proof_verify_demo/lean4-mlir
lake build Proofs
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c 'depends on axioms: \[propext, Classical.choice, Quot.sound\]$'
#   must equal total 'depends on axioms:' lines (157 now; ViT/softmax/MHSA already in it)

export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
export IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0
DATA=/home/skoonce/lean/claude_max/lean4-jax/data        # imagenette/ lives here

# render + iree-compile (once the renderers exist):
lake env lean tests/TestViTFwd.lean                      # writes verified_mlir/vit_fwd.mlir
lake env lean tests/TestViTTrain.lean                    # writes verified_mlir/vit_train_step.mlir
lake build vit-verified
.lake/build/bin/vit-verified "$DATA"                     # GPU-train imagenette 224², BS=32
```
- **Running a trainer:** Bash `run_in_background: true`; do NOT `nohup … &`. Flush per epoch.
- **224² ViT is attention-heavy** (O(N²), N=197) — minutes/epoch at BS=32; de-risk every
  fragment on iree-compile (and the SDPA backward on a numerical gradcheck) before the Main.

---

## 7. Honest framing (don't oversell)

- **The whole-net VJP `vit_full_has_vjp` is PROVEN + audit-clean + UNCONDITIONAL** (only the
  `0<ε` LNs). That is the real verification result — and it is a *bigger* result than ConvNeXt:
  it includes the **softmax VJP** (the `diag−outer` Jacobian collapsed to the O(n) closed form)
  and the **3-path SDPA backward** (dQ/dK/dV through the per-row softmax). The **trained model is
  a codegen artifact** faithful PER-OP (each fragment is the StableHLO of a proven-faithful
  emitter), validated by training — NOT a single `den(trainStep)` theorem — and the *faithful
  12-distinct-block ViT-Tiny* extends beyond the proof's shared-weight `transformerTower` witness.
- **Softmax is rendered as plain exp/sum** to match the proven `softmax` (a max-shift is
  mathematically identical and still faithful, but the proof uses plain — note whichever you ship).
- **LayerNorm placement is per-token over D** (the proof's `layerNorm_per_token_flat_diff`);
  **CONFIRM from the proof whether γ/β are scalar or `[D]`** before writing `ViTLayout` (the
  `layerNormForward` signature is scalar γ/β — if the proof applies it per-token with scalar
  γ/β, render scalar like ch9; if it's per-channel `[D]`, render `[D]` like the per-channel BN).
  This is the one spec detail to nail down first.
- **Out of scope** (the paper recipe, none needs proofs): AdamW, cosine schedule + warmup,
  weight decay, dropout/stochastic-depth, label smoothing, RandAug/Mixup, EMA, population-stats.
  `MainVitTrain.lean` lists lr 3e-4 / Adam / 80 epochs / cosine — the *verified* trainer is
  plain mean-loss SGD; expect to tune lr (ViT from scratch on small data is notoriously hard —
  a low Imagenette number is fine; the win is **verified attention codegen on native-res data**).
- Like ch9's per-example LN, the ViT LN is per-token ⇒ **eval uses the same stats as train**
  (no batch-stat/EMA mismatch). Watch the usual GAP/CLS-of-normalization degeneracy: if epoch-1
  sits exactly at chance (~10% on Imagenette-10), suspect a degenerate forward or an attention
  backward bug, not lr (validate the SDPA backward with the §5.3 gradcheck first).

---

## 8. Gotchas to carry over (from ch6–ch9 + new attention ones)

- **Adding an SHlo op = 9-site lockstep** (inductive/den/skel/Raw/Tok/toToks/emitTok +
  parseStack/parseStack_toToks). `geluF`/`swishF`/`sigmoidF` are the all-smooth worked
  examples for `softmaxRowF`. rfl-faithful ops stay OUT of `AuditAxioms` (round-trip covers them).
- **Batched `dot_general` dim_numbers** are the new fiddly bit: get `batching_dims=[0,1]`
  (batch, heads) + the right `contracting_dims` for QKᵀ (`[3]×[3]`) vs weights·V (`[3]×[2]`).
  The backward transposes the contraction — copy the shapes from `emitMHSAForward`, don't guess.
- **The SDPA backward is the highest-risk hand-wiring** (3 paths through a per-row softmax VJP).
  NUMERICALLY GRADCHECK it standalone (adjoint dot-product test or finite differences on the
  compiled `.vmfb`) before wiring into the block — this is where a transpose/axis bug hides and
  compile-clean ≠ correct.
- **reshape/transpose for heads**: `[B,N,D] ↔ [B,N,H,d] ↔ [B,H,N,d]`. The forward and backward
  must be exact inverses (transpose dims reversed). A wrong perm compiles fine but trains dead.
- **Patch embed (k=stride=16, even)**: reuse ch9's hand-verified non-overlapping recipe — patch
  weight-grad = dilate dy interior 15 (no high → 16·14−15 = 209), valid conv → `[192,3,16,16]`.
  (No input-grad; first layer.) Then flatten `[B,192,14,14]→[B,196,192]` via transpose+reshape.
- **CLS + pos-embed**: CLS = learned `[1,D]` concatenated at row 0 (backward = slice row-0 grad);
  pos = learned `[N+1,D]` broadcast-added (backward = sum over batch). Both are params in the layout.
- **BS is baked into the rendered MLIR** (`tensor<32x…>`); keep BS identical across
  TestViT{Fwd,Train}.lean AND MainViTVerified.lean (use 32).
- **`%x` is the flat `[BS,150528]` FFI buffer** reshaped to `[BS,3,224,224]` inside the graph.
  Mean-loss cotangent (÷B) so a sane lr works.
- **emitter byte-stability:** after edits, `git diff --stat verified_mlir/` must stay empty for
  the OTHER chapters' `.mlir`.
- **StableHLO.lean import hygiene:** it already imports LayerNorm/EfficientNet/Depthwise/
  StridedConv/PerChannelBN. `softmaxRowF`'s `den = rowSoftmax` needs Attention.lean — check for
  an import cycle (Attention imports Tensor + LayerNorm; StableHLO imports LayerNorm). If a cycle
  appears, lift `rowSoftmax`/`rowSoftmax_has_vjp_mat` into a small leaf module, or render the
  softmax backward as a hand fragment (like ch9's LN) without a dedicated SHlo op if cleaner.

---

## 9. File map (ch10)

**Already exists (proofs — reuse, don't rebuild):**
- `LeanMlir/Proofs/Attention.lean` — `softmax_has_vjp`, `sdpa(_back_{Q,K,V}_correct)`,
  `mhsa_has_vjp_mat(_correct)`, `transformerBlock/Tower(_has_vjp_mat)`, `vit_body(_has_vjp_mat)`,
  `patchEmbed_flat(_has_vjp)`, apex `vit_full_has_vjp(_correct)`. **PROOF (done).**
- `LeanMlir/Proofs/LayerNorm.lean` — `gelu(_has_vjp)`, `geluScalarDeriv_eq` (ch9), LN. **done.**
- `tests/AuditAxioms.lean` — already prints the ViT/softmax/MHSA headlines (157).
- `MainVitTrain.lean` (`vitTiny` NetSpec), IRPrint `renderSoftmax`/`softmax{Fwd,Back}Module`/
  `sdpa{Fwd,Back}Module`, MlirCodegen `emitMHSAForward`/`emitTransformerBlockForward` —
  **UNVERIFIED** reference (architecture + emit shapes only).

**To create (the ch10 verified codegen):**
- add `softmaxRowF`/`softmaxRowBack` to `LeanMlir/Proofs/StableHLO.lean` (+ parser cases in
  `StableHLOParse.lean`). (No new Mathlib — softmax VJP is proven.)
- `tests/TestSoftmaxRow.lean`, `tests/TestSDPA.lean` (+ gradcheck), `tests/TestMHSA.lean`,
  `tests/TestViTBlock.lean`.
- `tests/TestViTFwd.lean` + `tests/TestViTTrain.lean` (the renderers).
- `ViTLayout` in `LeanMlir/IreeRuntime.lean`.
- `MainViTVerified.lean` + lakefile `vit-verified` (clone `convnext-verified`).
- `verified_mlir/vit_{fwd,train_step}.mlir` (rendered output).

**Reuse templates:** `tests/TestConvNeXt{Fwd,Train}.lean` + `MainConvNeXtVerified.lean` +
`ConvNeXtLayout` (closest whole-net 224² renderer/Main/layout; the patchify k-general strided
conv + per-token LN + GELU fragments transfer directly); `tests/TestGelu.lean` (the op-pair +
iree template); ch8 `tests/TestSE.lean` (the standalone-combinator-with-gradcheck template for
the SDPA/MHSA de-risk); IRPrint/MlirCodegen attention emit (the byte-shape reference).
