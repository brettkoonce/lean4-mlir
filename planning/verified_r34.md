# Verified ResNet-34 (Chapter 6) verified-codegen — handoff

Continuation doc for the next chapter of the verified-codegen ladder: a
**ResNet** with residual/skip connections and global-average-pool, rendered to
proof-carrying StableHLO and GPU-trained through the real Lean→IREE-FFI→ROCm
path. Picks up after ch2 (linear), ch3 (MLP), ch4 (MNIST-CNN), ch5 (CIFAR, no-BN
**and** BatchNorm) all closed and committed to `main`.

> Status in one line: **the semantic whole-network VJP for a ResNet-style net
> ALREADY EXISTS** — `Proofs.cnn_has_vjp_at` (stem→maxpool→id-block→proj-block→
> GAP→dense), conditional, with a discharged concrete instance
> `CnnConcrete.cnnConcrete_has_vjp_correct`, both already audited 3-axiom-clean
> (part of the 140/140). So **Milestone A is almost entirely a RENDERING +
> trainer chapter** (add 2 SHlo ops, render the existing `cnnForward`, train).
> The jump to a *real* ResNet-34 (16 blocks, strided downsampling, per-channel
> BN) is **Milestone B** and is gated by ONE genuinely-new hard piece: **strided
> conv**. Read §§1–3 first; the rest is reference.

Written 2026-06-06 by the session that finished ch5 (CIFAR A+B). `origin/main`
is behind — the two ch5 commits (`34cdc52` ch5-A, `9c2a9f4` ch5-B) are **local,
not yet pushed** (per the commit-to-main-never-push-without-permission rule).

---

## 0. The pattern (unchanged from every chapter)

Typed, proof-carrying StableHLO emitter. Per op-graph:
`SHlo a ─skel→ Raw ─toToks→ [Tok] ─emitTok→ StableHLO text`, with two halves:
- **Semantic R4:** `den (graph) = <proven Mathlib fderiv quantity>` (faithfulness).
- **Syntactic R4:** `parse (toToks (skel a)) = skel a` (structural round-trip).

Deliverables each chapter: (a) whole-net forward graph + faithfulness, (b) full
SGD train-step renderer, (c) a `*-verified` exe that GPU-trains on the rendered
StableHLO, (d) audit ℝ-headline(s) 3-axiom-clean. ch5 also added per-op backward
faithfulness (`bnBack_faithful`) and a discharged tiny instance.

Ladder so far (all proven fwd+back AND GPU-trained on the proof-rendered text):
ch2 92.10%, ch3 97.86%, ch4 98.89%, ch5-A (CIFAR no-BN) 66%, ch5-B (CIFAR BN) 57%.
**ch6 = ResNet (residual + GAP).**

---

## 1. ⭐ THE BIG REUSE — the ResNet whole-net VJP is already proven

`LeanMlir/Proofs/CNN.lean` already builds and proves an end-to-end **ResNet-style
CNN** (the "CNN analogue of `vit_full_has_vjp`"):

```
cnnForward  (CNN.lean:2800) =
  dense ∘ globalAvgPoolFlat ∘ rblkP ∘ rblk ∘ maxPoolFlat ∘ cbr(stem)
```
- `cbr` = conv→BN→relu stem (= our ch5 `convBnRelu`, `bnForward` global scalar BN).
- `rblk`  (CNN.lean:~2780) = **identity** basic block
  `relu( BN₂conv₂(relu BN₁conv₁ x) + x )`  (`residual_has_vjp`).
- `rblkP` (CNN.lean:2790) = **projection** basic block
  `relu( BN₂conv₂(relu BN₁conv₁ x) + (BN_p conv_p x) )` (`residualProj_has_vjp`),
  used when channels change (`c → oc`).
- `globalAvgPoolFlat` (CNN.lean:2530) — mean over spatial, `Vec (c·h·w) → Vec c`;
  `globalAvgPoolFlat_has_vjp(_correct)` (2667/2683) — linear, backward broadcasts.

The whole-net VJP **`cnn_has_vjp_at`** (CNN.lean:2817) chains all of it through
`vjp_comp_at`/`residual(Proj)_has_vjp_at`/`resblock_body_has_vjp_at`, conditional
on `0<ε` per BN and ReLU/maxpool smoothness. **`CnnConcrete.cnnConcrete_has_vjp_correct`**
discharges every hypothesis (injective stem for maxpool no-ties + exact-istd BN +
γ=0 resblocks) — an *unconditional* witness, already in the audit.

**Implication:** the semantic backbone of ch6-A is DONE. Milestone A is: add the
two missing SHlo ops (residual-add, global-avg-pool), build `resnetFwdGraph`
denoting `cnnForward`, render the train step, and GPU-train. Same shape as ch5-A
but reusing `cnn_has_vjp_at` instead of writing a new whole-net VJP.

---

## 2. Architecture — what `cnn_has_vjp_at` covers vs real ResNet-34

| | `cnnForward` (proven) | real ResNet-34 |
|---|---|---|
| stem | 1 convBnRelu (SAME, stride 1) | 7×7 conv **stride 2** → BN → relu |
| pool | 1 maxPool 2×2 | 3×3 maxpool **stride 2** |
| blocks | 1 identity + 1 projection | 16 basic blocks (3+4+6+3), **strided** at stage starts |
| channels | c → oc (one bump) | 64→128→256→512 (3 bumps, projection skips) |
| BN | `bnForward` per-example **global, scalar γ/β** | per-channel batch-BN |
| head | GAP → dense | GAP → dense |

So `cnn_has_vjp_at` is a *representative* 2-block ResNet, not 34 layers. Two honest
targets:

| Variant | Verified result | New work | Effort |
|---|---|---|---|
| **A. ResNet-style (the proven 2-block net)** | trains on GPU through proof-rendered StableHLO; reuses `cnn_has_vjp_at` | **2 SHlo ops** (residualAdd, GAP) + render + trainer | **low–med** |
| **B. real ResNet-34** | deep 16-block net | **strided conv (hard, new)** + N-block chain + per-channel BN | **high** |

**Recommended: A first** (it's the cheap win that proves residual+GAP rendering
end-to-end and reuses the existing whole-net VJP), then chip at B.

---

## 3. Work breakdown

### Milestone A — verified ResNet-style net (reuse `cnn_has_vjp_at`)
1. **New SHlo ops** (lockstep: inductive/`den`/`skel`/`Raw`/`Tok`/`toToks`/`emitTok`
   + `parseStack`/`parseStack_toToks`, exactly like ch5's `bnF`/`bnBack`):
   - **`addV {n} : SHlo n → SHlo n → SHlo n`** — binary residual add, `den = den a + den b`.
     MIRROR the existing `.sub` constructor (already binary: `toToks a ++ toToks b ++
     [.sub n]`, `parseStack` pops two). The residual skip reuses the input as an
     **operand leaf** in BOTH branches (same `%x` name) — so the graph stays a TREE
     (two copies of the operand leaf), no DAG machinery needed. `addV_faithful` = rfl.
   - **`gapF {c h w} : SHlo (c*h*w) → SHlo c`** — global average pool, `den =
     globalAvgPoolFlat c h w (den e)`. `gapF_faithful` = rfl. emitTok: reshape to
     `[B,c,h,w]`, `reduce add over [2,3]`, multiply by `1/(h·w)`, reshape `[B,c]`.
   - (BN ops `bnF`/`bnBack`, conv/maxpool/dense/relu/selectPos — ALL already exist.)
2. **`resnetFwdGraph` + `_faithful`** denoting `cnnForward` (compose the existing
   `denseF`/`bnF`/`flatConvF`/`maxPoolF`/`reluF`/`addV`/`gapF`). The residual block
   subgraph: `reluF (addV (bnF…convF…reluF…bnF…convF %x) (skip %x))`. Mirror
   `cifarBnFwdGraph_faithful`.
3. **Train step** `resnetTrainStepText` — scale `cifarBnTrainStepText`, adding:
   the residual **backward** (cotangent after the block's relu mask flows to BOTH
   branches; `dx = Fback(dy) + skipback(dy)` — a `stablehlo.add` of the two), and
   the GAP backward (broadcast `dy/(h·w)` over spatial). The production reference is
   **`IRPrint.resnetTrainStepModule`** (IRPrint.lean:1318) — copy its residual/GAP
   StableHLO. De-risk on rocm early.
4. **Trainer** `MainResnetVerified.lean` + `resnet-verified` exe. Reuse
   `mlpTrainStepV`. Define a `ResnetLayout` (conv/BN-γ/β/dense params; γ/β rank-0
   scalars as in ch5-B `CifarBnLayout`). Train on CIFAR-10 (32×32) or MNIST.
5. Audit (the ℝ-headlines `cnn_has_vjp_at_correct` + `resnetFwdGraph_faithful`
   already-or-newly clean), commit.

### Milestone B — toward real ResNet-34
6. **Strided conv (THE hard new op).** `flatConv` is stride-1 SAME. Need a
   stride-2 forward (`Vec (ic·2h·2w) → Vec (oc·h·w)`) + its input-VJP (a *dilated /
   strided transpose* convolution) + weight-grad. This is a genuine proof + render
   addition (the conv-VJP reversed-kernel trick changes with stride). Likely the
   bulk of B. (StableHLO `stablehlo.convolution` supports `window_strides`; the
   backward needs `lhs_dilation`.)
7. **N-block chain.** Either generalize `cnn_has_vjp_at` to a fold over a list of
   blocks, or hand-chain the 16 (long but mechanical via `vjp_comp_at`). Watch
   `maxHeartbeats` — the chain is deep (ch4's `cnnBackGraph_faithful` already needed
   `set_option maxHeartbeats 2000000`).
8. **Per-channel BN** (for a real accuracy lift — see ch5-B caveat in §8). Apply
   `bnForward` per channel-slice (n = h·w) with per-channel γ/β `Vec oc`; prove the
   chained VJP. A real extension of the ch5 scalar BN.

---

## 4. What's REUSABLE (a lot — most of the semantics)

| reusable | where | for |
|---|---|---|
| `cnn_has_vjp_at` (+`_correct` via CnnConcrete) | CNN.lean:2817 | **whole-net VJP, ch6-A — DONE** |
| `cnnForward` | CNN.lean:2800 | the graph `resnetFwdGraph` must denote |
| `rblk`/`rblkP`, `residual_has_vjp`, `residualProj_has_vjp`, `resblock_body_has_vjp_at` | CNN.lean:~930–2797 | residual block semantics |
| `globalAvgPoolFlat_has_vjp(_correct)` | CNN.lean:2667/2683 | GAP semantics |
| `convBnRelu_has_vjp_at`, `convBn_has_vjp`, `convBnRelu_differentiableAt` | CNN.lean:869/905/2768 | conv→BN→relu block |
| `bnF`/`bnBack` SHlo ops + faithfulness | StableHLO.lean (ch5-B) | BN rendering |
| `flatConvF`/`maxPoolF`/`denseF`/`reluF`/`selectPos`/`convBack`/`maxPoolBack` | StableHLO.lean (ch4) | conv/pool/dense ops |
| `IRPrint.resnetTrainStepModule`, `renderLN`/`renderLNBack` | IRPrint.lean:1318/936/953 | production BN-ResNet StableHLO to copy |
| `MainResnetTrain` (`resnet34-train` exe) | lakefile:99 | data/arch reference (the OLD unverified path) |
| `mlpTrainStepV`, `forwardF32`, `CifarBnLayout` pattern (rank-0 γ/β) | IreeRuntime.lean | FFI + param packing |
| `cifar-bn-verified` / `MainCifarBnVerified.lean` | repo root | **the trainer template to copy** |

---

## 5. New SHlo ops needed (the whole delta for A)

- `addV` (binary residual add) — mirror `.sub`; `den a + den b`; emitTok = one
  `stablehlo.add`. The two operands are independent subtrees (both ending in the
  `%x` operand leaf) — tree-safe, no DAG.
- `gapF` (global average pool) — `den = globalAvgPoolFlat`; emitTok = reshape +
  `reduce add over [2,3]` + multiply `1/(h·w)` + reshape.
- (Milestone B only) `flatConvStrided` + `convStridedBack` — strided conv fwd/back.

Each op = lockstep update of inductive/`den`/`skel`/`Raw`/`Tok`/`toToks`/`emitTok`
(StableHLO.lean) + `parseStack`/`parseStack_toToks` (StableHLOParse.lean). Miss one
→ non-exhaustive match. (See ch5-B's `bnF`/`bnBack` for the exact 9-site recipe.)

---

## 6. Build / audit / run (same as every chapter)

```bash
cd /home/skoonce/lean/proof_verify_demo/lean4-mlir
lake build Proofs                       # regenerates verified_mlir/ via #eval
lake env lean tests/AuditAxioms.lean 2>&1 | grep -c 'depends on axioms: \[propext, Classical.choice, Quot.sound\]$'
#   must equal total 'depends on axioms:' lines (140 now; add ch6 ℝ-headlines → 141+)

export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/ffi:/opt/rocm/lib:$LD_LIBRARY_PATH"
export IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0
DATA=/home/skoonce/lean/claude_max/lean4-jax/data           # mnist + cifar-10/ live here
.lake/build/bin/cifar-bn-verified "$DATA"                   # ch5-B sanity (~57%)
# .lake/build/bin/resnet-verified "$DATA"                   # the new exe
.venv/bin/iree-compile --iree-hal-target-backends=rocm --iree-rocm-target=gfx1100 \
  --iree-codegen-llvmgpu-use-reduction-vector-distribution=false  M.mlir -o /tmp/x.vmfb
```
`ffi/libiree_ffi.so` is built (gitignored). `Vec`/`Mat` are `Fin → ℝ` (no runtime
rep) ⇒ trainers read `verified_mlir/*.mlir` (materialized at build time by `#eval`).

---

## 7. Gotchas (carried from ch4/ch5 + ResNet-specific)

- **Residual = DAG, but tree-safe via operand leaves.** `F(x)+x` reuses `x`; in the
  SHlo tree, put `.operand "%x" x` as a leaf in BOTH the F-branch and the skip-branch
  — same SSA name renders fine, `den` reads the same value, round-trip stays
  structural. No DAG/let machinery needed for the FORWARD graph. The train step
  (hand-written) sums the two backward branches at the fan-out: `dx = dF + dskip`.
- **Adding an SHlo op = 9-site lockstep** (inductive/den/skel/Raw/Tok/toToks/emitTok
  + parseStack/parseStack_toToks). Copy the ch5-B `bnF`/`bnBack` diffs verbatim as a
  template.
- **Paren-counting the fwd graph** bit me twice in ch5 (off-by-one closing parens on
  the deeply-nested `denseF(.reluF(…))` tree). Count opens, match closes; let the
  build error point you at the line.
- **`den (op) = …` faithfulness via `simp only [graph, forward, Function.comp_apply,
  <all *_faithful lemmas>, den_operand]`** — exactly like `cifarBnFwdGraph_faithful`.
- **Audit exact-match** wants an **ℝ-carrying** headline (`*_correct` / `*_faithful`
  that goes through the real analysis), NOT a pure `rfl` lemma (which can be
  `[propext]`-only and break the `=[propext, Classical.choice, Quot.sound]` grep).
  `cnn_has_vjp_at_correct` and `resnetFwdGraph_faithful` qualify; `addV_faithful`
  (rfl) does NOT — don't list rfl lemmas in AuditAxioms.lean.
- **rank-0 scalar params (γ/β)** flow through the FFI: the C shim's empty-product
  gives size 1. `broadcast_in_dim %g, dims = []`. (Validated in ch5-B.) Per-channel
  BN (Milestone B) switches γ/β to `Vec oc` (rank-1, `dims = [1]`).
- **Deep chain heartbeats:** `cnnBackGraph_faithful` needed `set_option
  maxHeartbeats 2000000`. The 16-block chain (B) will need more / smaller lemmas.
- **Conv has NO bias in standard ResNet** (BN absorbs it). The proofs' `convBnRelu`
  carries a bias `b`; just set `b = 0` (or keep it — its grad is harmless).
- **emitter byte-stability:** after editing, `git diff --stat verified_mlir/` must be
  empty for the ch2–ch5 files (non-empty = drift).
- **`#eval … writeFile "verified_mlir/…"`** must `IO.FS.createDirAll "verified_mlir"`.

---

## 8. The honest framing (don't oversell)

- **BN is per-example GLOBAL, scalar γ/β** (ch5's `bnForward`/`renderLN`), NOT
  per-channel batch-BN. That's what keeps the per-example "D1 shortcut" intact and
  is faithfully verifiable. On CIFAR it *underperformed* no-BN (57% vs 66%) because
  global+scalar normalization is aggressive. A ResNet rendered with THIS BN will
  likely also not beat its no-BN counterpart on accuracy — the win is **verifying
  residual connections + GAP + the deep block structure end-to-end**, not a SOTA
  number. A real accuracy story needs **per-channel BN** (Milestone B.8).
- **"Verified ResNet-34" ≠ 16 strided blocks until Milestone B lands strided conv.**
  Milestone A is honestly "a verified *ResNet-style* net (stem + maxpool + identity
  block + projection block + GAP + dense) — the structure `cnn_has_vjp_at` proves."
  Name the exe/chapter accordingly; don't claim 34 layers before they're rendered.
- The earlier ch5 surprise (no-BN SGD trains fine; the "BN-motivating failure" was
  lr-scale/init sensitive) means the *pedagogical* arc here is "we can VERIFY these
  architectures' gradients and run them," not "BN/ResNet unlock accuracy" — frame
  the chapter as a correctness/codegen result.

---

## 9. Suggested first steps for the clean session
1. `git log --oneline -3` (confirm ch5 A+B local commits) and skim `cnn_has_vjp_at`
   (CNN.lean:2817) + `CnnConcrete` (the discharged instance) to internalize the
   proven structure.
2. Read `cifarBnFwdGraph`/`cifarBnTrainStepText`/`MainCifarBnVerified.lean` (ch5-B)
   — the exact template to copy.
3. Add `addV` + `gapF` SHlo ops (lockstep, copy the `bnF` diff shape). Build.
4. `resnetFwdGraph` + `_faithful` (denote `cnnForward`). Build + check faithfulness.
5. De-risk a single residual-block + GAP train-step fragment on rocm (`iree-compile`)
   BEFORE the full chain.
6. `resnetTrainStepText` + `MainResnetVerified` + exe; GPU-train; audit; commit.
