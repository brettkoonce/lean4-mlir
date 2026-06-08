# Validation-sweep refactor — VerifiedNetSpec + the spec→MLIR ladder

Handoff for migrating the `*-verified` trainers onto a declarative `VerifiedNetSpec`
and tying each spec to its math + its generated MLIR. ch2–5 done; imagenette nets next
(order: r34 → mnv2 → efficientnet → convnext → vit). Keep context lean: this doc + the
per-net pointers below should be enough — don't read whole proof files, grep to the names.

## The thesis (what "done" means per net)

A net is a readable `VerifiedNetSpec` layer list, and we prove a ladder of ties:

| rung | statement | where |
|---|---|---|
| A shapes | `#guard spec.toSpecs == <layout>` (param interface) | `VerifiedNets.lean` |
| B spec→math | `denoteX spec.layers … = <proof Forward fn>` by `rfl` | `Proofs/SpecVJP.lean` |
| C VJP | `xVerified_has_vjp : HasVJP (denoteX spec.layers …)` | `Proofs/SpecVJP.lean` |
| E spec→MLIR | `den (xFwdGraph …) = denoteX spec.layers …` (generated StableHLO denotes the spec) | `Proofs/SpecVJP.lean` |

Plus: the trainer is a one-line `main` and is **GPU-validated** to a sane accuracy.

## The 4 modules (all light except SpecVJP)

- `LeanMlir/VerifiedSpec.lean` — `VLayer` DSL + `VerifiedNetSpec` + `toSpecs` (folds layers →
  `(dims,initKind)` param layout). initKind 0=He, 1=ones(γ), 2=zeros(β/bias).
  Existing constructors: `convBn`, `maxPool`, `residualStage`, `globalAvgPool`, `dense`,
  `relu`, `conv` (plain, no BN), `flatten`, `bn` (scalar-global `bnForward`, rank-0 γ/β).
- `LeanMlir/VerifiedTrain.lean` — `VerifiedNet`/`VerifiedConfig` + the driver. Two methods:
  `train` (packed-params `mlpTrainStepV`, He-init — the conv/MLP nets) and `trainLinear`
  (2-arg `linearTrainStepV`, zero-init — only ch2). Both share `compileVmfb`/`loadData`
  (`.mnist`/`.cifar`/`.imagenette`)/eval. `loadData` for `.imagenette` center-crops 256→224.
- `LeanMlir/VerifiedNets.lean` — the shared concrete specs (a spec lives here once a proof
  imports it; r34's lives in its Main until then). Holds the A-rung `#guard`s.
- `LeanMlir/Proofs/SpecVJP.lean` — rungs B/C/E. Imports VerifiedNets + the relevant proof
  modules + StableHLO. **NOT in the `lake build` aggregator** — verify with
  `lake env lean LeanMlir/Proofs/SpecVJP.lean` (heavy; Mathlib).

## Migration recipe (per net)

1. **Trainer**: add `xVerified : VerifiedNetSpec` to `VerifiedNets.lean` (layer list +
   slug + inC/imageH/imageW + nClasses + data) + an A-rung `#guard xVerified.toSpecs == …`
   (read `XLayout` in `IreeRuntime.lean` for the expected `(dims,kind)` list). Rewrite
   `MainXVerified.lean` to `import LeanMlir.VerifiedNets` + a `VerifiedConfig` + one-line
   `main := xVerified.train xConfig (argv.head?.getD "data")`.
2. **GPU-validate** (recipe below) — confirm a sane accuracy.
3. **Proof ties** in `SpecVJP.lean`: `denoteX` (match the exact layer list → the proof's
   Forward fn, pinning dependent dims like `(h:=…)(w:=…)`); `xVerified_denote_eq … := rfl`;
   `xVerified_has_vjp` (canonical witness: `backward := ∑ pdiv …; correct _ _ _ := rfl`);
   and `xVerified_fwd_faithful := xFwdGraph_faithful …` if the faithfulness infra exists.

### Gotchas (bit us already)
- **`#guard` cannot follow a `/-- … -/` doc-comment** (it's a command, not a decl). Use a
  plain `-- …` or `/- … -/` comment before `#guard`. (Hit this 3×.)
- **Re-seed**: the generic spec-driven init seeds every param slot (biases too), so the
  legacy hand-init nets (mnist/cifar) re-initialize differently — accepted (statistically
  equiv, not bit-identical to old runs).
- **Dependent dims**: conv forwards take `{h w}` only via products (`Mat (c*h*w) d1`), so
  `denote`/faithfulness must pass `(h:=…)(w:=…)` explicitly. `Vec 784 ≟ Vec (1*(2*14)*(2*14))`
  is defeq (Nat reduces) — `exact …`/`rfl` handle it.
- **`Kernel4 oc ic kH kW`** is the conv-kernel type (in `Proofs`, via CNN.lean).
- The spec must be in a **lib** for a proof to import it (can't import a Main exe root) —
  hence `VerifiedNets.lean`. (We deduped linear after first defining a twin.)

## GPU run recipe (this sandbox)
```
PATH=$PWD/.venv/bin:$PATH  LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:$LD_LIBRARY_PATH \
IREE_BACKEND=rocm IREE_CHIP=gfx1100 IREE_DEVICE=hip \
.lake/build/bin/<trainer> <dataDir>
```
- MNIST idx: `/home/skoonce/lean/mnist-lean4/data`.  CIFAR-10 bin: `/home/skoonce/lean/claude_max/lean4-jax/data` (has `cifar-10/`).  Imagenette: needs `<dir>/imagenette/{train.bin@256,val.bin@224}` — locate before running.
- `ffi/libiree_ffi.so` is a HIP+local-task build; `.venv/bin/iree-compile` supports rocm gfx1100. CPU fallback = `IREE_BACKEND=llvm-cpu IREE_DEVICE=local-task` (slow for wide nets). Single-GPU only. See `memory/lean4-mlir-env-assets.md`.

## ⭐ SESSION HANDOFF — 2026-06-08 (RESUME POINT — read this first)

Ladder per net (A=shape `#guard` · B/C=spec→math+VJP · E=spec→generated-MLIR, forward):

| ch | net | A | B/C | E(fwd) | notes |
|----|-----|---|-----|--------|-------|
| 2 | linear | ✅ | ✅ full | ✅ (+cotangent) | 92% |
| 3 | mlp | ✅ | ✅ full | ✅ fwd+**back** | 97.78% |
| 4 | cnn | ✅ | ✅ full | ✅ fwd | 98.99% |
| 5 | cifar / cifar-bn | ✅ | ✅ full | ✅ fwd | ~67% / **~71%** — **cifar-bn is now PER-CHANNEL BN (reconciled `a8b91ed`)** |
| 6 | r34 | ✅ | rep (skeleton `resnet34_has_vjp_at`) | ✅ rep (`resnetFwdGraph`, pre-existing) | render uses per-channel BN |
| 7 | **mnv2** | ✅ | ✅ **FULL** (real strided render) | ✅ **FULL strided** + rep | **the exemplar — full A+B+C+E(fwd) on the real render** |
| 8 | efficientnet | ✅ | rep | ⏳ needs an **SE/broadcast-mul op** | swish/sigmoid/conv/bn/dw F-ops exist |
| 9 | convnext | ✅ | rep | ✅ rep (built `layerScaleF` op) | scalar LN |
| 10 | vit | ✅ | rep (scalar-LN) | 🧱 needs **attention ops** | the wall (only `softmaxRowF` exists) |

All B/C + E ties live in `Proofs/SpecVJP.lean` (NOT in the build aggregator — check with
`lake env lean LeanMlir/Proofs/SpecVJP.lean`); the forward graphs + faithfulness live in
`Proofs/StableHLO.lean`. Commits `8e9ae0b … 60d5ed6` on `main`, **NOT pushed** (~86 ahead).

**Key lessons (the expensive map-making this session bought):**
1. **Full whole-net B/C VJP fold HIT A LEAN WALL** at concrete dims — `vjp_comp_at`+`set` do repeated
   `isDefEq`/`kabstract` over deeply-nested `@[reducible]` blocks ⇒ >10 min compile (even with opaque
   named-block defs + explicit `(h:=)(w:=)`). Reverted. **Only mnv2 has full B/C** (built
   `mobilenetv2Forward_full` + the strided block VJP `invresBodyStrided_has_vjp_at`/`_differentiableAt`,
   `convBnRelu6Strided*`, `dwBnRelu6Strided*`/`ivDepthwiseStrided`). **Revisit full B/C with
   `@[irreducible]` blocks or a fold combinator.** The other 4 nets' B/C is REPRESENTATIVE (denote a
   rep `VLayer` list → the proof's existing `<net>Forward`/skeleton by `rfl` + canonical `HasVJP`).
2. **E does NOT hit that wall** — it is `simp`-rewriting (op faithfulness lemmas are `@[simp] rfl`), so
   even the FULL concrete strided mnv2 graph faithfulness compiles in the normal ~35 s. **Full E is
   tractable where full B/C was not.**
3. **But forward-E per net first needs every op as an SHlo F-op.** Adding one = ~9 edits across the
   pipeline (`SHlo`/`Raw`/`Tok` ctors + `den`/`skel`/`toToks`/`emitTok` + `StableHLOParse.parseStack`
   + the `parseStack_toToks` round-trip) + a `rfl` faithfulness lemma. mnv2/r34 were lucky (all ops
   existed); convnext needed `layerScaleF` (done). efficientnet needs an SE/broadcast-mul op; vit needs
   attention. `layerScaleF` (8fdeacb) is the worked example to copy for the next op.
4. **Formalization (A/B/C/E) is architecture-MATH ⇒ INDEPENDENT of v3-parity** (Adam/aug/cosine/
   label-smoothing = optimizer/data loop). Don't gate the cheap reps on numerics; reserve the expensive
   full passes (full B/C; per-channel BN re-spec) for after the architecture is final.

**CIFAR-BN reconciliation DONE (2026-06-08, commit `a8b91ed`, not pushed) ↓ — generalize to the imagenette nets next.**

## BN reconciliation — ✅ DONE for CIFAR; imagenette nets remain

**✅ CIFAR-BN CLOSED (commit `a8b91ed`).** The verified cifar-bn net is now **per-example per-channel**
BN (`bnPerChannelTensor3`, `m=h·w`, γ/β `[c]`) end-to-end: render (`cifarBnTrainStepText` +
`cifarBnFwdTextPC`) + proofs (B `cifarBnVerified_denote_eq`, C `cifarCnnBn_has_vjp_at` via a new
`convBnReluPC_has_vjp_at` block, E `cifarBnFwdGraph_faithful` via `.bnPerChannelF`), all 3-axiom-clean,
byte-tied to the GPU-trained MLIR. **GPU-validated: per-channel BN 71% > no-BN 67% > old scalar 58%**
(BN now *helps*, inverting the prior "BN hurts"). New surface: `.bnPerChannel oc` VLayer
(`VerifiedSpec`), `CifarBnLayout` γ/β `#[c]`. **The exact same scalar→per-channel swap recipe now
generalizes to mnv2 / efficientnet / convnext** (which still use scalar BN/LN in their verified
forwards): build a per-channel `convBnReluPC`-style block on `bnPerChannelTensor3_has_vjp`, swap the
forward def's `bnForward (c*H*W)` → `bnPerChannelTensor3 c H W`, refold the VJP, swap the graph's
`.bnF` → `.bnPerChannelF`. **GOTCHA: `lake env lean <file>` reads STALE import `.olean`s — must
`lake build <Module>` to refresh after editing an imported proof file.**

**⚠️ Honest scope note:** per-*example* per-channel (instance-norm) gives the *accuracy* lift but
NOT LR headroom — cifar-bn collapses at lr≥0.2 like no-BN (its ceiling is *below* no-BN's). "BN
enables higher LR" needs *batch*-statistic BN (`m=N·h·w`, the enet-E5 `[N,C,H,W]↔[C,N·H·W]` bridge)
with running-stats / train≠eval handling — a bigger, separate build if ever wanted.

**THE GAP (now imagenette-only):** every imagenette VERIFIED PROOF forward still uses **scalar-global** `bnForward`
(one γ/β normalizing over the *whole* `c·h·w` map per example); every RENDER + every UNVERIFIED trainer
uses **per-channel** BN/LN. These are *different functions*, so today's proofs are about a slightly
different net than what actually trains. (ViT isn't special — its proof is scalar-LN like the rest, its
render is per-channel-`[D]` like the rest.)

**Closing it is a BUILD, not new theory** — the per-channel infra already exists and is fully wired:
- math: `bnPerChannelFlat (oc m ε γ β) : Vec (oc*m) → Vec (oc*m)` + `bnPerChannelFlat_has_vjp` (`Proofs/PerChannelBN.lean`).
- SHlo op: `bnPerChannelF` + `bnPerChannelF_faithful` (`den = bnPerChannelTensor3`), threaded through den/skel/toToks/emit/parse.
- the RENDERS already emit `bnPerChannelF` (e.g. r34 committed MLIR — `tests/TestResnet34Fwd.lean`). So
  only the *proof* side is the scalar holdout. To close: swap scalar `bnForward`/`bnF` →
  `bnPerChannelFlat`/`bnPerChannelF` in the verified forward + its VJP + its graph faithfulness.

**EVAL-STATS subtlety (decide this first):** the unverified `MainCifarCnnBnTrain` uses EMA running stats
(true batch-norm, momentum 0.1 per `RESULTS.md`) ⇒ train ≠ eval. The verified scalar `bnForward` is
PER-EXAMPLE (instance-norm-like) ⇒ train = eval, no EMA mismatch — **that is *why* the verified nets went
scalar** (it sidesteps the running-stats machinery the proofs don't model). `bnPerChannelFlat` can be
either: per-example per-channel (`m = h·w`, instance-norm — simplest, keeps train=eval) OR batch
(`m = N·h·w`, the efficientnet-E5 `[N,C,H,W]↔[C,N·H·W]` bridge — needs running-stats handling for eval).
**Pick per-example per-channel** unless you want to model running stats in the proof.

**WORKED EXAMPLE — CIFAR-BN, DONE (`a8b91ed`), the template to copy for the imagenette nets:**
chose **per-example per-channel** (`m=h·w`, train=eval). Touched: `cifarCnnBnForward` +
`cifarCnnBn_has_vjp_at` (`CifarCNN.lean`) — `bnForward (c*H*W)` → `bnPerChannelTensor3 c H W`, γ/β
`ℝ`→`Vec c`, refold via the new `convBnReluPC_has_vjp_at`/`_differentiableAt` (mirror of the scalar
`convBnRelu_has_vjp_at` on `bnPerChannelTensor3_has_vjp`); `cifarBnFwdGraph` `.bnF`→`.bnPerChannelF`
(explicit `(oc h w)`) + `cifarBnFwdGraph_faithful` via `bnPerChannelF_faithful`
(`StableHLO.lean`); `denoteCifarBn`/`cifarBnVerified_has_vjp`/`_fwd_faithful` (`SpecVJP.lean`);
`.bnPerChannel oc` VLayer + `CifarBnLayout` γ/β `#[c]`. Both apex theorems 3-axiom-clean; render is
hand-string `cifarBnTrainStepText`/`cifarBnFwdTextPC` (`[B,c·H·W]`→`[B,c,H·W]`, reduce spatial [2]).
**To generalize: repeat this swap on mnv2/efficientnet/convnext** (their verified forwards still use
scalar `bnForward`). GOTCHA: `lake build <Module>` (not `lake env lean`) to refresh stale import oleans.

### THREE BN regimes (the thing to keep straight — not one "per-channel")

There are *three* normalizations in play; "per-channel" is ambiguous between two of them:
1. **scalar-global** — `bnForward (n) ε γ β`, γ/β `ℝ`: one mean/var over the *whole* `c·h·w` map per
   example. Reduce axis (render) = flat `[1]`. Per-example ⇒ train=eval. *(what every imagenette PROOF
   forward still uses; what CIFAR used pre-`a8b91ed`.)*
2. **per-example per-channel** (instance-norm) — `bnPerChannelTensor3 oc h w` = `bnPerChannelFlat oc (h·w)`,
   γ/β `[c]`: each channel over its own `h·w`, per example. Render reduce = `[2,3]` on `[N,C,H,W]` (or
   CIFAR's `[2]` on the `[B,c,h·w]` reshape). Per-example ⇒ train=eval. *(what CIFAR is NOW; what the
   r34/mnv2 RENDERS already emit.)*
3. **batch per-channel** (textbook BN) — `bnchwFwd N oc h w` / `bnPerChannelFlat oc (N·h·w)` via the
   `[N,C,H,W]↔[C,N·H·W]` bridge, γ/β `[c]`: each channel over batch+spatial. Render reduce = `[0,2,3]`.
   **train ≠ eval** ⇒ needs running-stats/EMA the proofs don't model. *(what the efficientnet RENDER emits.)*

### PER-NET STATUS — proof BN vs render BN (verified from code + committed verified_mlir, 2026-06-08)

| net | PROOF forward BN | RENDER BN (committed `_train_step.mlir` reduce axis) | gap |
|---|---|---|---|
| **cifar-bn** | per-example per-channel `bnPerChannelTensor3` | per-example per-channel (reshape→`[2]`) | ✅ **matched (`a8b91ed`)** |
| r34 | scalar — parametric *skeleton* `resnet34_has_vjp_at`, NO whole-net forward | per-example per-channel (`[2,3]` ×145) | ⚠️ gap |
| **mnv2** | **scalar** `bnForward` + relu6 (`convBnRelu6_has_vjp_at`, `mobilenetv2Forward_full`) | per-example per-channel (`[2,3]` ×81) | ⚠️ gap — *identical to CIFAR's* |
| **enet** | **scalar** `bnForward` + swish (`convBnSwish_has_vjp`, `mbconvBody`) | **batch** per-channel (`[0,2,3]` ×343) | ⚠️ gap — *render is BATCH not instance* |
| **convnext** | **scalar** `layerNormForward` (≡ `bnForward`, `convNextBlockBody`) | scalar-global LN (reshape→`[1]` ×133) | ✅ **already consistent** |

KEY: three of four imagenette PROOFS are in exactly CIFAR's pre-`a8b91ed` scalar state. convnext is NOT —
its proof AND render are both scalar-global LN (consistent by design; the proof even has an explicit
"LN representation caveat"). convnext's real gap is the LN **axis** (LayerNorm over *channels* per spatial
position — the TRANSPOSE of `bnPerChannelTensor3`), a different/deferred op, and per-example either way.

### PORTING NOTES (how each differs from the CIFAR recipe)

- **mnv2 — closest fit, do FIRST.** Render already uses the exact normalization CIFAR now proves, so the
  swap applies directly. But instead of one `convBnReluPC` block, build per-channel versions of:
  `convBnRelu6` (relu6 not relu), `dwBnRelu6` (`depthwiseFlat` not `flatConv`), the **strided**
  `convBnRelu6Strided`/`dwBnRelu6Strided`, and the project `convBn`-no-activation block — then refold the
  inverted-residual body VJP (`invresBodyStrided_has_vjp_at`). Has the full concrete `mobilenetv2Forward_full`
  (no skeleton to build). relu6/swish smoothness hyps carry as-is (BN moves to post-BN pre-activations).
- **enet — different target op (BATCH).** To MATCH its render, reconcile to `bnchwFwd`/`bnPerChannelFlat
  m=N·h·w` (regime 3), NOT `bnPerChannelTensor3`. Blocks: `convBnSwish` + `dwBnSwish` + project (SE is
  BN-free, untouched). Plus decide the eval-stats story (running stats / batch-stat eval) — the per-example
  path sidesteps this; the batch path doesn't. Bigger than mnv2.
- **convnext — NOT this swap.** No scalar-vs-per-channel proof/render gap. If "real" ConvNeXt LN is ever
  wanted, it's a channel-axis LayerNorm op (new), independent of this pass.
- **r34 — build the forward first.** B/C is the representative skeleton; needs a concrete `resnet34Forward`
  (the awkward path) before the CIFAR recipe applies. Render is per-example per-channel like mnv2.

## Status

DONE — full ladder A+B+C+E, GPU-validated:
- ch2 linear (92%, fwd+cotangent E) · ch3 MLP (97.78%, fwd+**back** E) · ch4 CNN (98.99%,
  fwd E) · ch5 cifar (~67%, fwd E) · ch5 cifar-bn (**~71%, PER-CHANNEL BN**, fwd E — reconciled
  `a8b91ed`). Commits 649ef6a … 25847f2 (+ `a8b91ed`/`04e2087`).

DONE — rung A + trainer migrated + GPU codegen-validated (B/C/E pending) — **ALL imagenette**:
- ch6 r34 (146 params, 74818d1) · ch7 mnv2 (82, 39ee53d) · ch8 efficientnet (262, 7750b1c) ·
  ch9 convnext (180, 15c38af) · ch10 vit (200, ae8b1d2). Each: spec in `VerifiedNets`,
  `#guard toSpecs == XLayout.specs`, one-line `main`, GPU compiles rendered MLIR + loads
  imagenette + trains. **Numbers ≈chance** (r34/mnv2 = 9.45% flat — degenerate from scratch);
  that's the deferred v3-parity issue, NOT a codegen problem.

**A-route sweep COMPLETE — all ch2–10 trainers are on the declarative VerifiedNetSpec/driver.**
All on `main`, **NOT pushed** (~80 ahead).

**B/C DONE for all imagenette nets (2026-06-07, commits 9ce5674 + 3129a48, in `SpecVJP.lean`):**
- **mnv2 = FULL faithful** (9ce5674): spec rfl-denotes the *real* 6-block strided render
  (`mobilenetv2Forward_full`) + canonical `HasVJP` witness + NEW strided inverted-residual
  block VJP (`invresBodyStrided_has_vjp_at`/`_differentiableAt`, `convBnRelu6Strided*`,
  `dwBnRelu6Strided*`, `depthwiseStride2Flat` reuse) in `MobileNetV2.lean`.
- **r34/efficientnet/convnext/vit = REPRESENTATIVE** (3129a48): `denoteXRep <rep VLayer list>
  = <net>Forward/skeleton := rfl` (rung B) + canonical `HasVJP` (rung C), referencing the
  audited apex (`efficientnet_has_vjp`/`convnext_has_vjp`/`vit_full_has_vjp`/`resnet34_has_vjp_at`).
  All cheap to compile (symbolic dims). vit = scalar-LN witness (per-channel `[D]` gap); r34 ties
  to the parametric [3,4,6,3] skeleton (no concrete whole-net Forward).

**Full faithful build for r34/enet/convnext/vit is DEFERRED — known Lean wall.** The mnv2 full
whole-net conditional fold (`mobilenetv2_full_has_vjp_at`, ~14-stage `vjp_comp_at`) WAS built and
is mathematically correct, but compiles pathologically slowly (>10 min: `isDefEq` churns ~170k
heartbeats/stage over deeply-nested `@[reducible]` blocks at CONCRETE dims — symbolic reps don't
hit this). Tried: subst-inline, opaque named-block defs + explicit `(h:=)(w:=)` dims (got it
compiling but still >10min). Reverted the fold; kept the forward + rfl + canonical witness +
strided block VJP. Representative tie is the pragmatic current state; revisit the full fold with
a non-`isDefEq` assembly (e.g. `@[irreducible]` blocks, or a fold combinator) if pursued.

**E (forward) DONE for mnv2 — BOTH representative AND full strided (cd11958 + 384416e):**
- representative (`cd11958`): `mobilenetv2FwdGraph` + `_faithful : den=mobilenetv2Forward` (SAME 2-block).
- **FULL strided (`384416e`): `mobilenetv2FwdGraphFull` + `_faithful : den=mobilenetv2Forward_full`**
  — whole-net SHlo graph at REAL ch7 dims (3×224²→7×7×64; strided stem `flatConvStridedF` +
  `depthwiseStridedF` downsamples + `addV` skips), proven by ONE `simp only` + `unfold`. Axiom-clean,
  compiles in normal ~35s. SpecVJP `mobilenetv2Verified_fwd_faithful : den(graph)=denoteMobilenet
  spec.layers` (= faithfulness ∘ denote_eq) ⇒ **mnv2 has the full A+B+C+E(fwd) ladder tied to the
  real render** — the only imagenette net at that bar.
- **KEY LESSON: E is `simp`-rewriting (op lemmas are `@[simp] rfl`), NOT the `vjp_comp_at`/`isDefEq`
  fold — so it does NOT hit the concrete-dim wall that forced full B/C to be reverted.** Full E is
  TRACTABLE where full B/C was not.
- **forward-E per net = `simp` graph, BUT first needs every op as an SHlo F-op.** mnv2/r34 were lucky
  (all ops existed). Adding an op = ~9 edits across the pipeline (SHlo/Raw/Tok ctors + den + skel +
  toToks + emitTok + StableHLOParse parseStack + parseStack_toToks round-trip) + a faithfulness rfl.
  Status: **r34 rep E done** (`resnetFwdGraph`, pre-existing) · **mnv2 done** (full+rep) · **convnext
  rep E DONE (8fdeacb)** — needed a NEW `layerScaleF` op (γ⊙x), `convNextFwdGraph_faithful` +
  SpecVJP `convnextRep_fwd_faithful`, axiom-clean · **efficientnet** needs an SE-gate/broadcast-mul op
  (swish/sigmoid/conv/bn/depthwise exist) — medium · **vit** needs the attention ops (only `softmaxRowF`
  exists; QKV dot_generals + SDPA + MHSA missing) — the wall.

Remaining E: backward graph (the VJP graph denotes the backward — carries the relu6 smoothness hyps)
+ re-route the committed `tests/Test*` string `.mlir` to `pretty(emit graph)` (plumbing). Other deferred:
v3-parity (Adam/aug/schedule). Formalization (B/C/E) is architecture-math, INDEPENDENT of v3-parity
(optimizer/data) — don't gate the cheap reps on it; reserve the expensive full passes for after the
architecture (esp. scalar-vs-per-channel BN) is final. See the readiness sections + loose ends.

## THE PLAN (agreed)

**End goal = FULL builds (option b) for every net** — the most tedious/rigorous tie (the
spec's *full rendered* net = its math + VJP + generated MLIR), to preempt any "what does
verification mean" argument. The **representative tie (a) is NOT the destination.**

**First pass (now) = the A-route only** across all five imagenette nets
(r34✅→mnv2✅→enet→convnext→vit): add the block `VLayer`s, `#guard toSpecs == XLayout.specs`,
one-line `main`, GPU-smoke (compiles/loads/trains; ≈chance from scratch is fine). Then come
back for full B/C + E.

**Why B/C/E are a second pass, not the sweep:** the imagenette proof `Forward`/apex defs are
**representative witnesses** (smaller depth, scalar BN/LN), not the full rendered nets — so
full-net B/C needs *building* the real forward per net (the r34-awkward path, often with a
scalar-vs-per-channel BN gap). E needs whole-net `SHlo` assembly + re-routing the committed
`.mlir`. Both deferred.

**Also deferred — v3 parity (numbers):** the verified trainers run plain mean-loss SGD and
score ≈chance on imagenette; the v3 unverified `Main*Train.lean` use Adam + augmentation +
cosine + label-smoothing. Goal: get the verified trainers training to comparable numbers /
codegen-identical to v3. Separate workstream (see loose ends).

## Imagenette nets — per-net pointers (do in this order)

All Layouts (`ResNet34/MobileNetV2/EfficientNet/ConvNeXt/ViTLayout`) are in
`IreeRuntime.lean` and already expose `.specs` → rung A is trivial. The trainers already
use `mlpTrainStepV` + `.imagenette` data → they fit `VerifiedNet.train` (one-line `main`).

### B/C readiness — r34 is the AWKWARD one; the rest are clean
B/C = `denote spec.layers = <proof Forward fn> := rfl` + canonical-witness `has_vjp`.
- **r34 has NO whole-net `Forward` def** — only the parametric skeleton `resnet34_has_vjp_at`
  + block-level VJP lemmas + tiny 1-channel toy blocks (`idBlk`/`downBlk`). So B/C must
  *build* `resnet34Forward` (the full [3,4,6,3] op-composition at real dims), and its proof
  BN is scalar `bnForward` vs the render's per-channel `[c]` BN (granularity gap). Hardest B/C.
- **mnv2 / efficientnet / convnext / vit all HAVE a `Forward` def** → clean B/C like ch2–5
  (modulo representative-depth/scalar-witness gaps): `mobilenetv2Forward` (MobileNetV2.lean:461),
  `efficientnetForward` (EfficientNet.lean:424), `convNextForward` (ConvNeXt.lean:267, ~2-block
  representative), `vit_full` (Attention.lean:3629, scalar-LN witness).
- **Suggested B/C order: mnv2 first** (clean Forward), then enet/convnext/vit, **r34 last**
  (build its forward as a dedicated task).

### E readiness — gets HARDER later (mirror image of B/C). Don't sweep E.
E = `den(graph)=spec` + committed `.mlir` = `pretty(emit graph)`. The per-**op** verified IR
is *largely built*: `StableHLO.SHlo` has `depthwiseF`/`depthwiseStridedF`, `swishF`, `relu6F`,
`softmaxRowF`, `geluF` — each with `den` + faithfulness. **The real gap**: the imagenette
committed MLIRs are **hand-written string concatenation** (`tests/TestX{Train,Fwd}.lean`, ~200
concat lines each), gradcheck-validated — they do NOT go through `emit(SHlo)`. So E per net =
(1) assemble the whole-net graph in `SHlo`, (2) prove whole-net `den(graph)=math` (deep
faithfulness fold), (3) **re-route the committed render** from strings to `pretty(emit graph)`
so E ties to the real artifact (today even `resnetFwdGraph_faithful` is a *representative*
SHlo graph, NOT what's committed), (4) fill missing ops. It is plumbing, not new math, but a
sizable per-net assembly+re-render. Tractability:
| net | E-readiness | missing for E |
|---|---|---|
| r34 | easiest | ops all in SHlo; whole-net SHlo assembly + re-route + per-channel BN |
| mnv2 | easy | `depthwiseF`/`depthwiseStridedF`/`relu6F` exist; assembly + re-route |
| efficientnet | medium | SE sub-graph + per-channel BN (swish exists) |
| convnext | harder | layerScale + **even-kernel transposed conv** (gelu/depthwise exist) |
| vit | hardest | **batched multi-head SDPA** (softmaxRowF exists; the dot_general attention + 3-path backward is the wall) |
→ **E pass (later, opportunistic): start r34/mnv2; treat enet/convnext/vit E — esp. attention — as a deliberate separate project.**

Below = net-specific architecture + new `VLayer`s each B/C needs (B/C/E readiness is in the
two sections above).

- **r34** (`Proofs/ResNet34.lean`): ✅ **A + trainer + GPU done** (commit 74818d1). Apex
  `resnet34_has_vjp_at` (:174, **parametric skeleton**, no Forward def). VLayers all exist
  (`convBn`/`maxPool`/`residualStage`/`globalAvgPool`/`dense`). B/C = build `resnet34Forward`
  (real-dim [3,4,6,3] op-composition) — the awkward one; do last. `toBlocks` (spec→renderer
  blocks) was prototyped + proven `== blocks` in a reverted edit — re-derivable.
- **mnv2** (`Proofs/MobileNetV2.lean`): apex `mobilenetv2_has_vjp_at`; **`mobilenetv2Forward`
  (:461)** → clean B/C. Inverted-residual/MBConv (expand 1×1 → depthwise k×k strided → project
  1×1, per-channel BN, **relu6**, residual when s=1∧ic=oc). New VLayers: depthwise, relu6,
  invertedResidual, per-channel BN (distinct from scalar `.bn`). **B/C-first imagenette target.**
- **efficientnet** (`Proofs/EfficientNet.lean`): apex `efficientnet_has_vjp(_correct)` (:456/536,
  unconditional), `efficientnetForward` (:424). MBConv + **squeeze-excite** + **swish** +
  **batch-norm** (per-channel). New VLayers: swish, sigmoid, SE, MBConv-w-SE, per-channel BN.
- **convnext** (`Proofs/ConvNeXt.lean`): apex `convnext_has_vjp(_correct)` (:293/368),
  `convNextForward` (:267) — **representative ~2-block witness**, render is [3,3,9,3] (depth gap
  to handle). Block: depthwise 7×7 → scalar-LN → 1×1 expand → **GELU** → 1×1 project → layerScale;
  patchify 4×4/s4 + 2×2/s2 downsample (even-kernel transposed backward). New VLayers: convNextBlock,
  gelu, layerScale, patchify, downsample. LN is **scalar-global** = the existing `.bn` op.
- **vit** (`Proofs/Attention.lean`): apex `vit_full_has_vjp_correct` (softmax/SDPA/MHSA/block/
  patchEmbed stack). Transformer blocks (MHSA + MLP), patch-16 embed, CLS+pos. New VLayers:
  patchEmbed, transformerBlock (MHSA/SDPA), gelu, LN. **Proof-vs-render granularity gap**:
  the LN proof witness is **scalar** (`layerNormForward` = `bnForward`), but the rendered/trained
  ViT uses **per-channel `[D]` LN** — so denote can only tie the spec to the scalar witness; the
  `[D]` render is beyond it (state this explicitly). vit_full witness is also scalar-LN.

## Loose ends (do AFTER the rest are in the new pattern)

- **Wire `SpecVJP.lean` into the build** — currently only `lake env lean`-checked, not in the
  `LeanMlir.lean` aggregator, so it isn't verified on `lake build` (could bit-rot). Adding it
  pulls Mathlib into the default build (slow) — decide if worth it (maybe a separate test target).
- **Backward-E for the conv nets** — only forward-E is done for CNN/CIFAR. `cnnBackGraph_faithful`
  (StableHLO ~900) and the cifar backward equivalents exist but carry the 5 ReLU/maxpool
  smoothness hyps; wrap them as `xVerified_back_faithful` (transcribe the hyps; verbose). MLP
  already has backward-E (only 2 hyps).
- **Text/file trust boundary** — rung E proves `den(graph) = spec`; still trusted: `xFwdModuleV =
  pretty (emit graph)` (the pretty-printer) and that the committed `verified_mlir/*.mlir` equals
  that text (regeneration). `Proofs/StableHLOParse.lean` could close the text→IR round-trip.
- **CIFAR-BN GPU perf** — ~2 min/epoch (scalar-BN global reductions over c·h·w are slow on the
  rocm reduction path); fine, just slow. Not a correctness issue.
- **Push** — ~73 local commits on `main`, unpushed (needs explicit per-push OK per
  `memory/git-workflow-commit-to-main.md`).
- **Imagenette E pass** — the separate later project (see "E readiness" above): assemble each
  net's whole-net `SHlo` graph, prove `den(graph)=math`, and re-route the committed `.mlir` from
  the `tests/TestX*` string emitters to `pretty(emit graph)`. Per-op IR mostly exists; start
  r34/mnv2; attention (vit) is the wall. Do AFTER B/C is unified across all five.
