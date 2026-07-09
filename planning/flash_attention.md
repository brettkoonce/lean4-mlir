# Planning — FlashAttention codegen (kill the O(T²) attention wall)

Demo-first scaffold for **FlashAttention** in the StableHLO codegen. Goal: train the transformer
demos (tinyGPT/TinyStories/ViT) at long context (T≈8K) without materializing the dense
`[B,H,T,T]` score matrix — which is the single wall that makes 8K infeasible (planning notes
2026-07-09: T=8192 dense attention is ~640 GB at batch 32, ~20 GB even at batch 1).

## 0. TL;DR

- **What.** Replace the dense `softmax(QKᵀ/√d + mask)·V` emit (`MlirCodegen.lean` attention, score
  tensor `sTy = [B,H,T,T]` at `:1151` fwd / `:6253` bwd) with a **tiled online-softmax** loop that
  keeps only one `[B,H,T,Bk]` key-block live at a time. Memory O(T²) → O(T·Bk).
- **Scope decision (2026-07-09).** Emit the **full-query, blocked-keys** variant: a single
  `stablehlo.while` over ⌈T/Bk⌉ key blocks, carrying `(k_idx, O[B,H,T,d], m[B,H,T], l[B,H,T])`.
  The query axis stays full (`O`, `m`, `l` are O(T·d)/O(T) — cheap); only the keys are blocked.
  Simpler than the 2-D (query×key) tiling and captures the whole memory win. (Classic FlashAttention
  tiles both axes to fit SRAM; we tile only keys because the win we need is HBM materialization, not
  SRAM residency — IREE's own tiling handles the compute.)
- **IREE reality (checked 2026-07-09, iree 3.12.0rc).** IREE HAS a fused `iree_linalg_ext.attention`
  op, but there is **no auto-raise pass** from our `dot→softmax→dot` StableHLO into it — so "just flip
  a flag" is out. We emit the tiling explicitly with core StableHLO (`while` + `dynamic_slice` +
  `iota`), backend-agnostic (rocm today, cuda/MI300 later). An `iree_linalg_ext.attention`-emit path
  is a possible future alternative (less code, but couples to IREE internals + still needs a VJP).
- **Algorithm DE-RISKED (2026-07-09):** `jax/demos/flash_attention_ref.py` — the exact block
  recurrence (fwd) and the recompute+D-trick (bwd) match dense attention and dense autodiff to
  **~1e-15**, causal + non-causal, ragged blocks, AND the Bq=T single-loop special case. The codegen
  emits from this validated spec; the only open question is whether IREE realizes the memory bound.

## 1. The algorithm (validated in flash_attention_ref.py)

Per key block j (query axis full), online softmax:

```
init:  O=0 [B,H,T,d],  m=-inf [B,H,T],  l=0 [B,H,T]
for kj in 0, Bk, 2Bk, … < T:            #  stablehlo.while
  Kj,Vj = K[:,:,kj:kj+Bk,:], V[…]        #  dynamic_slice → [B,H,Bk,d]
  Sij   = Q·Kjᵀ · scale                  #  [B,H,T,Bk]   ← the ONLY O(T·Bk) tensor
  (causal) Sij = mask(q ≥ kj+k, Sij, -inf)
  mij   = rowmax(Sij)                     #  [B,H,T]
  m_new = max(m, mij)
  α     = exp(m − m_new)                  #  rescale prior block  (guard -inf→0)
  Pij   = exp(Sij − m_new)                #  [B,H,T,Bk]
  l     = α·l + rowsum(Pij)
  O     = α·O + Pij·Vj                    #  [B,H,T,d]
  m     = m_new
O   = O / l                               #  final normalize
Lse = m + log(l)                          #  logsumexp, SAVED for backward
```

Backward (recompute, no stored P; `D = rowsum(dO⊙O)` is the softmax-Jacobian term):

```
D = rowsum(dO ⊙ O)                        #  [B,H,T]
for kj … :                                #  same key loop
  Sij = Q·Kjᵀ·scale ; (causal mask)
  Pij = exp(Sij − Lse)                    #  recomputed  [B,H,T,Bk]
  dV[kj] += Pijᵀ · dO
  dPij    = dO · Vjᵀ                       #  [B,H,T,Bk]
  dSij    = Pij ⊙ (dPij − D) · scale
  dQ     += dSij · Kj
  dK[kj] += dSijᵀ · Q
```

## 2. Codegen plan

| where | change | effort |
|---|---|---|
| `Types.lean` | `transformerEncoder … (flashAttn : Bool := false)` (or a NetSpec/config flag threaded like `useShampoo`) | trivial |
| `MlirCodegen.lean` fwd attention (`emitMultiHeadAttn`, ~`:1140`) | `emitFlashAttnFwd`: the `stablehlo.while` above; returns O + Lse (Lse replaces the saved softmax `mh_sm` for the block backward) | **large** |
| `MlirCodegen.lean` bwd attention (~`:6253`) | `emitFlashAttnBwd`: the recompute loop; consumes Lse + O + dO | **large (the fiddly one)** |
| Q/K/V + head-split + output-proj | unchanged — flash swaps ONLY the inner sdpa (`softmax(S)V`), same as the dense path's seam | — |

**StableHLO specifics / risks.**
- `T`, `Bk` static (T = seqLen known at codegen; Bk compile-time const, e.g. 128; pad T to a multiple
  of Bk or handle a ragged tail block). Only `k_idx` is dynamic → `stablehlo.while` counter.
- `dynamic_slice(K, [0,0,k_idx·Bk,0], [B,H,Bk,d])` for the key/value block.
- Causal mask per block depends on the DYNAMIC offset `k_idx·Bk`: build `qidx = iota[T]`,
  `kidx = k_idx·Bk + iota[Bk]`, `mask = qidx ≥ kidx`, select `-inf`. Causal also lets the loop stop
  early (`k_idx·Bk ≤ q_max`) but full-query blocks span all q, so we run all key blocks and mask.
- **The load-bearing UNKNOWN:** does IREE keep the `while` body's per-iteration `[B,H,T,Bk]` tensors
  from persisting across iterations (→ real O(T·Bk) memory), or does it hoist/materialize them
  (→ no win)? This is correct-by-construction regardless (matches the reference), but the MEMORY win
  is a hypothesis until measured on GPU (§4 rung 3). If IREE won't bound it, fall back to the
  `iree_linalg_ext.attention` emit path.

## 3. Validation ladder (demo-first)

1. **numpy reference — DONE 2026-07-09** (`flash_attention_ref.py`): fwd == dense, bwd == dense VJP to
   ~1e-15 (causal/full, ragged blocks, Bq=T single-loop). The algorithm is pinned.
2. **StableHLO compiles — DONE 2026-07-09**: the emitted `while`/`dynamic_slice`/`iota` pattern
   legalizes on BOTH llvm-cpu and rocm gfx1100 (iree 3.12).
3. **Numerical match — DONE 2026-07-09 (forward, CPU)**: `emitFlashAttnSdpa` + `flashProbeModule` +
   the `flash-probe` exe emit a standalone `@main(Q,K,V)->O`; `scripts/flash_probe_check.py` compiles
   it (llvm-cpu) and runs it via `iree.runtime`, matching numpy dense attention to **~1e-6** (fp32)
   for full + causal across n=16..128 / 4..16 blocks. So the FORWARD emitter is correct end-to-end
   offline. STILL TODO: gfx1100 run (GPU busy at build time) + a tinyGPT `nano-flash` vs `nano`
   loss-equivalence run once the backward lands.
4. **MEMORY measured** (the actual point): compile a T=2048/4096/8192 train step both ways, compare
   peak device allocation. Confirms O(T²)→O(T·Bk). If it doesn't drop, diagnose (§2 risk) before
   scaling.
5. **Long-context train** (cloud): 8K TinyStories/GPT with flash on — the deliverable.

## 4. Staging

- **Phase 1 — forward — DONE 2026-07-09** (`emitFlashAttnSdpa` in MlirCodegen.lean, standalone-
  validated via `flashProbeModule`/`flash-probe`/`scripts/flash_probe_check.py`). Not yet wired into
  the transformer block (that needs the backward too, else training breaks) — it's the validated
  core, ready to drop into `emitMHSAForward`'s sdpa seam behind a flag.
- **Phase 2 — backward** (`emitFlashAttnBwd` + rungs 3-bwd/4/5) — NEXT. The training unlock; the
  recompute loop is fully specified by `flash_attention_ref.py`'s `flash_backward` (validated to
  ~1e-15). Then integrate fwd+bwd into `emitMHSAForward`/the block backward behind a `flashAttn`
  flag, GPU-validate, and MEASURE the memory drop (rung 4).
- Caveat carried from the 8K analysis: long context also needs the position embedding resized to
  `[T,D]` and retrained (or RoPE/ALiBi — separate codegen), independent of flash.

## 5. References
- Dao et al., *FlashAttention* (2022) + *FlashAttention-2* (2023) — the online-softmax + recompute.
- `jax/demos/flash_attention_ref.py` (this repo — the validated spec the codegen emits).
- `MlirCodegen.lean` dense attention: `emitMultiHeadAttn` (~:1140), score `sTy=[B,H,T,T]` (:1151 fwd,
  :6253 bwd) — the seam flash replaces.
- IREE 3.12: `iree_linalg_ext.attention` (fused op; no auto-raise from stablehlo in this build).
